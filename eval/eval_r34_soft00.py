# eval_cale_top1.py
"""
加载已训练的 CALE V2 模型，用 top-1 口径重新评估测试集，与 Baseline argmax 对齐
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score, precision_score, recall_score
)

CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'
CKPT       = '/root/autodl-tmp/IEEE/cale_v2_10pct_1fold_resnet34_softlabel0/best_model_fold1.pth'

SOFT_LABEL      = 0.5
INFERENCE_SCALE = 0.5
BATCH_SIZE      = 128
NUM_WORKERS     = 8
SEED            = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(SEED); torch.manual_seed(SEED)

EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural']
NUM_CLASSES    = len(EMOTIONS)
NUM_CONDITIONS = 4
NUM_OUTPUTS    = NUM_CLASSES * NUM_CONDITIONS

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

LOW_BRIGHTNESS_THRESH  = 80
HIGH_BRIGHTNESS_THRESH = 120
BLUR_VAR_THRESH        = 10

def classify_condition(image_path):
    try:
        with Image.open(image_path) as img:
            gray = img.convert('L')
            arr = np.array(gray, dtype=np.float64)
        brightness = arr.mean()
        laplacian = cv2.Laplacian(arr, cv2.CV_64F)
        variance = laplacian.var()
        if variance < BLUR_VAR_THRESH:
            return 2
        elif brightness < LOW_BRIGHTNESS_THRESH:
            return 0
        elif brightness > HIGH_BRIGHTNESS_THRESH:
            return 1
        else:
            return 3
    except Exception:
        return 3

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_path, transform):
        self.transform = transform
        self.samples = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            img_no = parts[0]
            labels = np.array(list(map(int, parts[1:8])), dtype=np.float32)
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(img_dir, f'{img_no}{ext}')
                if os.path.exists(p):
                    cond = classify_condition(p)
                    self.samples.append((p, labels, cond))
                    break
        self.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cond = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), cond

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNet18CALE(nn.Module):
    def __init__(self, num_outputs=28, reduction=16,
                 dropout=0.0, freeze_shallow=False, use_se=False):
        super().__init__()
        backbone = torchvision.models.resnet34(weights=None)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        in_features = backbone.fc.in_features
        self.use_se  = use_se
        if use_se:
            self.se = SELayer(in_features, reduction)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        if self.use_se:
            x = self.se(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def cale_inference(logits, conds):
    probs = torch.sigmoid(logits).view(-1, NUM_CLASSES, NUM_CONDITIONS)
    one_hot = F.one_hot(conds.long(), NUM_CONDITIONS).float()
    scale   = INFERENCE_SCALE + (1 - INFERENCE_SCALE) * one_hot
    return (probs * scale.unsqueeze(1)).sum(dim=2)

def main():
    print(f'Device: {DEVICE}')
    print(f'Loading checkpoint: {CKPT}')

    test_set = TestDataset(TEST_DIR, LABEL_FILE, transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)
    print(f'测试集样本数: {len(test_set)}')

    model = ResNet18CALE(num_outputs=NUM_OUTPUTS, dropout=0.0,
                         freeze_shallow=False, use_se=False).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_probs, all_targets, all_conds = [], [], []
    with torch.no_grad():
        for inputs, targets, conds in test_loader:
            inputs = inputs.to(DEVICE)
            conds  = conds.to(DEVICE).long()
            logits = model(inputs)
            p      = cale_inference(logits, conds)
            all_probs.append(p.cpu().numpy())
            all_targets.append(targets.numpy())
            all_conds.extend(conds.cpu().numpy())

    probs   = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # ---------- 口径 1: top-1（与 Baseline argmax 完全对齐） ----------
    preds_idx = np.argmax(probs, axis=1)
    preds_top1 = np.zeros_like(targets)
    preds_top1[np.arange(len(preds_idx)), preds_idx] = 1

    uar_top1  = recall_score(targets, preds_top1, average='macro', zero_division=0)
    mf1_top1  = f1_score(targets, preds_top1, average='macro', zero_division=0)
    mic_top1  = f1_score(targets, preds_top1, average='micro', zero_division=0)
    ham_top1  = hamming_loss(targets, preds_top1)
    hit_top1  = (preds_top1 * targets).any(axis=1).mean()
    avg_top1  = preds_top1.sum(axis=1).mean()

    # ---------- 口径 2: 固定阈值 0.5（原训练脚本的原始口径） ----------
    preds_th = (probs > 0.5).astype(np.float32)
    uar_th  = recall_score(targets, preds_th, average='macro', zero_division=0)
    mf1_th  = f1_score(targets, preds_th, average='macro', zero_division=0)
    mic_th  = f1_score(targets, preds_th, average='micro', zero_division=0)
    ham_th  = hamming_loss(targets, preds_th)
    hit_th  = (preds_th * targets).any(axis=1).mean()
    avg_th  = preds_th.sum(axis=1).mean()

    print('\n================= CALE V2 测试集评估 =================')
    print('【Top-1 口径，与 Baseline argmax 对齐】')
    print(f'  UAR       = {uar_top1:.4f}')
    print(f'  Macro-F1  = {mf1_top1:.4f}')
    print(f'  Micro-F1  = {mic_top1:.4f}')
    print(f'  Hit Rate  = {hit_top1:.4f}')
    print(f'  Hamming   = {ham_top1:.4f}')
    print(f'  AvgLabels = {avg_top1:.4f}')
    print()
    print('【阈值 0.5 口径，原脚本口径】')
    print(f'  UAR       = {uar_th:.4f}')
    print(f'  Macro-F1  = {mf1_th:.4f}')
    print(f'  Micro-F1  = {mic_th:.4f}')
    print(f'  Hit Rate  = {hit_th:.4f}')
    print(f'  Hamming   = {ham_th:.4f}')
    print(f'  AvgLabels = {avg_th:.4f}')

    print('\n================= 与 Baseline 对比（Top-1 口径）=================')
    print('  Baseline (10% 1折): Macro-F1=0.3030, UAR=0.2136, HitRate=0.5801')
    print(f'  CALE V2  (10% 1折): Macro-F1={mf1_top1:.4f}, UAR={uar_top1:.4f}, HitRate={hit_top1:.4f}')
    print(f'  Delta: Macro-F1 = {mf1_top1 - 0.3030:+.4f}, UAR = {uar_top1 - 0.2136:+.4f}')

    out = {
        'top1': {'uar': float(uar_top1), 'macro_f1': float(mf1_top1),
                 'micro_f1': float(mic_top1), 'hit_rate': float(hit_top1),
                 'hamming': float(ham_top1), 'avg_labels': float(avg_top1)},
        'thresh_0.5': {'uar': float(uar_th), 'macro_f1': float(mf1_th),
                       'micro_f1': float(mic_th), 'hit_rate': float(hit_th),
                       'hamming': float(ham_th), 'avg_labels': float(avg_th)},
    }
    with open('/root/autodl-tmp/IEEE/cale_v2_10pct_1fold_resnet34_softlabel0/eval_top1.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('\n结果已保存至 cale_v2_10pct_1fold_resnet34_softlabel0/eval_top1.json')

if __name__ == '__main__':
    main()