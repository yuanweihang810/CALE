# eval_cale_weight_search.py
"""
在现有 CALE V2 checkpoint 上，扫描推理聚合权重 lambda
聚合公式: p_e = sigmoid(x_{e,c*}) + lambda * sum_{c != c*} sigmoid(x_{e,c})
扫描 lambda in [0.0, 0.1, ..., 1.0]，在验证集上选最优，在测试集上报结果
"""
import os, json
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
from sklearn.metrics import f1_score, recall_score, hamming_loss
from sklearn.model_selection import GroupKFold

CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'
CKPT       = '/root/autodl-tmp/IEEE/cale_v2_10pct_1fold/best_model_fold1.pth'

BATCH_SIZE  = 128
NUM_WORKERS = 8
SEED        = 42
TRAIN_RATIO = 0.1

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

def folder_to_emotion(name):
    m = {'angry':'angry','anger':'angry','disgust':'disgust','disgusted':'disgust',
         'fear':'fear','scared':'fear','happy':'happy','happiness':'happy',
         'sad':'sad','sadness':'sad','surprise':'surprise','surprised':'surprise',
         'natural':'natural','neutral':'natural'}
    return m.get(name.strip().lower())

def extract_subject_id(p):
    return os.path.basename(p).split('_')[0]

class EvalDataset(torch.utils.data.Dataset):
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
                    self.samples.append((p, labels, classify_condition(p)))
                    break
        self.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cond = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32), cond

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform, subset_indices):
        self.transform = transform
        self.samples = []
        root = Path(root_dir)
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            emo = folder_to_emotion(folder.name)
            if emo is None:
                continue
            idx = EMOTIONS.index(emo)
            label = np.zeros(NUM_CLASSES, dtype=np.float32); label[idx] = 1.0
            for f in sorted(folder.glob('*.*')):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    self.samples.append((str(f), label, classify_condition(str(f))))
        self.samples.sort(key=lambda x: x[0])
        self.samples = [self.samples[i] for i in subset_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cond = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32), cond

class SELayer(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c//r, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(c//r, c, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(F.adaptive_avg_pool2d(x, 1).view(b, c)).view(b, c, 1, 1)
        return x * y

class ResNet18CALE(nn.Module):
    def __init__(self, num_outputs=28, dropout=0.0, use_se=False):
        super().__init__()
        bb = torchvision.models.resnet18(weights=None)
        self.conv1, self.bn1, self.relu = bb.conv1, bb.bn1, bb.relu
        self.maxpool = bb.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.avgpool = bb.avgpool
        in_features = bb.fc.in_features
        self.use_se = use_se
        if use_se:
            self.se = SELayer(in_features)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        if self.use_se:
            x = self.se(x)
        x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x)
        return x

def get_branch_probs(model, loader):
    """返回 (probs, targets, conds)，probs 形状 (N, 7, 4)"""
    model.eval()
    all_probs, all_targets, all_conds = [], [], []
    with torch.no_grad():
        for inputs, targets, conds in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            p = torch.sigmoid(logits).view(-1, NUM_CLASSES, NUM_CONDITIONS)
            all_probs.append(p.cpu().numpy())
            all_targets.append(targets.numpy())
            all_conds.extend(np.array(conds))
    return np.concatenate(all_probs), np.concatenate(all_targets), np.array(all_conds)

def aggregate(probs, conds, lam):
    """p_e = prob[e, c*] + lam * sum_{c != c*} prob[e, c]，c* 由 conds 给出"""
    N = probs.shape[0]
    out = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    for i in range(N):
        c_star = conds[i]
        out[i] = probs[i, :, c_star] + lam * (probs[i].sum(axis=1) - probs[i, :, c_star])
    return out

def eval_top1(scores, targets):
    preds_idx = np.argmax(scores, axis=1)
    preds = np.zeros_like(targets)
    preds[np.arange(len(preds_idx)), preds_idx] = 1
    return {
        'macro_f1': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'uar': float(recall_score(targets, preds, average='macro', zero_division=0)),
        'hit_rate': float((preds * targets).any(axis=1).mean()),
    }

def eval_threshold(scores, targets, th=0.5):
    preds = (scores > th).astype(np.float32)
    return {
        'macro_f1': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'uar': float(recall_score(targets, preds, average='macro', zero_division=0)),
        'avg_labels': float(preds.sum(axis=1).mean()),
    }

def main():
    print(f'Device: {DEVICE}')
    model = ResNet18CALE(num_outputs=NUM_OUTPUTS, dropout=0.0, use_se=False).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print(f'Loaded: {CKPT}\n')

    # 重建验证集划分（与训练脚本一致）
    full_train_root = Path(CUSTOM_DIR)
    all_paths = []
    for folder in sorted(full_train_root.iterdir()):
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob('*.*')):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                all_paths.append(str(f))
    all_paths.sort()
    subject_ids = [extract_subject_id(p) for p in all_paths]
    gkf = GroupKFold(n_splits=5)
    _, val_idx = next(iter(gkf.split(np.zeros(len(subject_ids)), groups=subject_ids)))

    val_set = ValDataset(CUSTOM_DIR, transform_test, val_idx)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_set = EvalDataset(TEST_DIR, LABEL_FILE, transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f'验证集: {len(val_set)}  测试集: {len(test_set)}\n')

    val_probs, val_targets, val_conds = get_branch_probs(model, val_loader)
    test_probs, test_targets, test_conds = get_branch_probs(model, test_loader)

    print('================= 验证集扫描 lambda（Top-1 口径）=================')
    print(f'{"lambda":>8s} | {"Val Macro-F1":>14s} | {"Val UAR":>10s} | {"Val Hit":>10s}')
    best_lam, best_val_f1 = None, -1
    for lam in np.arange(0.0, 1.05, 0.1):
        scores = aggregate(val_probs, val_conds, lam)
        r = eval_top1(scores, val_targets)
        marker = ''
        if r['macro_f1'] > best_val_f1:
            best_val_f1 = r['macro_f1']; best_lam = lam; marker = ' *'
        print(f'{lam:8.1f} | {r["macro_f1"]:14.4f} | {r["uar"]:10.4f} | {r["hit_rate"]:10.4f}{marker}')
    print(f'\n验证集最优 lambda = {best_lam:.1f}, Macro-F1 = {best_val_f1:.4f}\n')

    print('================= 测试集 Top-1 口径 =================')
    print(f'{"lambda":>8s} | {"Test Macro-F1":>14s} | {"Test UAR":>10s} | {"Test Hit":>10s}')
    for lam in np.arange(0.0, 1.05, 0.1):
        scores = aggregate(test_probs, test_conds, lam)
        r = eval_top1(scores, test_targets)
        marker = '  <-- val best' if abs(lam - best_lam) < 1e-6 else ''
        print(f'{lam:8.1f} | {r["macro_f1"]:14.4f} | {r["uar"]:10.4f} | {r["hit_rate"]:10.4f}{marker}')

    print('\n================= 参考：Baseline 10% 1折 =================')
    print('  Top-1: Macro-F1=0.3030, UAR=0.2136, HitRate=0.5801\n')

    # 保存
    out = {'best_lambda_val': float(best_lam), 'best_val_macro_f1': float(best_val_f1)}
    for lam in np.arange(0.0, 1.05, 0.1):
        key = f'lambda_{lam:.1f}'
        val_r = eval_top1(aggregate(val_probs, val_conds, lam), val_targets)
        test_r = eval_top1(aggregate(test_probs, test_conds, lam), test_targets)
        out[key] = {'val': val_r, 'test': test_r}
    with open('/root/autodl-tmp/IEEE/cale_v2_10pct_1fold/weight_search.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('结果已保存至 cale_v2_10pct_1fold/weight_search.json')

if __name__ == '__main__':
    main()