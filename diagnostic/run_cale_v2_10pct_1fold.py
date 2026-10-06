# run_cale_v2_10pct_1fold.py
"""
CALE V2 (LE only) —— 10% subject 抽样，1 折，30 epoch
用于小样本鲁棒性对比实验
"""

import os
import json
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score, precision_score, recall_score
)
from sklearn.model_selection import GroupKFold

# ==================== 配置 ====================
CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'

VARIANT = 'V2'
SOFT_LABEL      = 0.5
INFERENCE_SCALE = 0.5
FIXED_THRESHOLD = 0.5

OUTPUT_DIR = '/root/autodl-tmp/IEEE/cale_v2_10pct_1fold'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE    = 128
LEARNING_RATE = 1e-5
EPOCHS        = 30
PATIENCE      = 10
SEED          = 42
WEIGHT_DECAY  = 5e-3
K_FOLDS       = 5          # 保留 5，但只跑第 1 折
NUM_WORKERS   = 8

FOCAL_GAMMA  = 2.0
CLASS_ALPHAS = [0.9, 0.8, 1.0, 0.3, 0.5, 0.7, 0.2]
DROPOUT      = 0.5
USE_WEIGHTED_SAMPLER = True

VARIANT_CONFIG = {
    'V2': {'use_ms': False, 'use_se': False},
    'V5': {'use_ms': True,  'use_se': False},
    'V6': {'use_ms': False, 'use_se': True},
    'V7': {'use_ms': True,  'use_se': True},
}
CFG     = VARIANT_CONFIG[VARIANT]
USE_MS  = CFG['use_ms']
USE_SE  = CFG['use_se']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural']
NUM_CLASSES    = len(EMOTIONS)
NUM_CONDITIONS = 4
NUM_OUTPUTS    = NUM_CLASSES * NUM_CONDITIONS
CONDITION_NAMES = ['weak_light', 'strong_light', 'low_res', 'standard']

# ==================== 数据增强 ====================
if USE_MS:
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ==================== 条件分辨器 ====================
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

def folder_to_emotion(folder_name):
    mapping = {
        'angry': 'angry', 'anger': 'angry',
        'disgust': 'disgust', 'disgusted': 'disgust',
        'fear': 'fear', 'scared': 'fear',
        'happy': 'happy', 'happiness': 'happy',
        'sad': 'sad', 'sadness': 'sad',
        'surprise': 'surprise', 'surprised': 'surprise',
        'natural': 'natural', 'neutral': 'natural',
    }
    return mapping.get(folder_name.strip().lower())

def extract_subject_id(img_path):
    return os.path.basename(img_path).split('_')[0]

# ==================== 数据集 ====================
class CALEDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None,
                 precomputed_samples=None, subset_indices=None):
        self.transform = transform
        self.samples   = []
        if precomputed_samples is not None:
            self.samples = precomputed_samples
        else:
            if split == 'Training':
                self._load_train(CUSTOM_DIR)
            else:
                self._load_test(TEST_DIR, LABEL_FILE)
        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]

    def _load_train(self, root_dir):
        root = Path(root_dir)
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            emo = folder_to_emotion(folder.name)
            if emo is None:
                continue
            emo_idx = EMOTIONS.index(emo)
            label   = np.zeros(NUM_CLASSES, dtype=np.float32)
            label[emo_idx] = 1.0
            for img_file in sorted(folder.glob('*.*')):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    cond = classify_condition(str(img_file))
                    self.samples.append((str(img_file), label, cond, emo_idx))
        print(f'训练集加载完毕: {len(self.samples)} 张图片')

    def _load_test(self, img_dir, label_path):
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
                    self.samples.append((p, labels, cond, -1))
                    break
        self.samples.sort(key=lambda x: x[0])
        print(f'测试集加载完毕: {len(self.samples)} 张图片')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cond, emo_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return (img,
                torch.tensor(label, dtype=torch.float32),
                cond,
                emo_idx)

# ==================== 加权采样器 ====================
def build_weighted_sampler(dataset):
    if not USE_MS or not USE_WEIGHTED_SAMPLER:
        return None
    class_counts = [0] * NUM_CLASSES
    for _, _, _, emo_idx in dataset.samples:
        if emo_idx >= 0:
            class_counts[emo_idx] += 1
    class_counts = [max(c, 1) for c in class_counts]
    class_weights = [1.0 / c for c in class_counts]
    weights = []
    for _, _, _, emo_idx in dataset.samples:
        weights.append(class_weights[emo_idx] if emo_idx >= 0 else 1.0)
    sampler = WeightedRandomSampler(weights=weights,
                                    num_samples=len(weights),
                                    replacement=True)
    print(f'加权采样器：各类别样本数 {class_counts}')
    return sampler

# ==================== SE 模块 ====================
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

# ==================== CALE 模型 ====================
class ResNet18CALE(nn.Module):
    def __init__(self, num_outputs=28, reduction=16,
                 dropout=0.5, freeze_shallow=False, use_se=True):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_shallow:
            for name, param in backbone.named_parameters():
                if 'layer3' in name or 'layer4' in name or 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.use_se:
            x = self.se(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== 28 维软标签构造 ====================
def build_targets_28(emo_indices, cond_indices):
    B = emo_indices.size(0)
    t28 = torch.zeros(B, NUM_CLASSES, NUM_CONDITIONS,
                      device=emo_indices.device)
    arange = torch.arange(B, device=emo_indices.device)
    for c in range(NUM_CONDITIONS):
        t28[arange, emo_indices, c] = SOFT_LABEL
    t28[arange, emo_indices, cond_indices] = 1.0
    return t28.view(B, NUM_OUTPUTS)

# ==================== 软标签 Focal Loss ====================
def soft_focal_loss(logits, targets, alphas_28, gamma=2.0):
    probs = torch.sigmoid(logits)
    pos_loss = -targets * (1 - probs).pow(gamma) * F.logsigmoid(logits)
    neg_loss = -(1 - targets) * probs.pow(gamma) * F.logsigmoid(-logits)
    alpha_w = torch.where(targets > 0,
                          alphas_28.unsqueeze(0).expand_as(targets),
                          torch.ones_like(targets))
    loss = alpha_w * (pos_loss + neg_loss)
    return loss.mean()

# ==================== 训练 ====================
def train_epoch(model, loader, optimizer, alphas_28):
    model.train()
    total_loss = 0.0
    for inputs, _, conds, emo_idx in loader:
        inputs  = inputs.to(DEVICE, non_blocking=True)
        conds   = conds.to(DEVICE).long()
        emo_idx = emo_idx.to(DEVICE).long()

        targets_28 = build_targets_28(emo_idx, conds)

        optimizer.zero_grad()
        logits = model(inputs)
        if USE_MS:
            loss = soft_focal_loss(logits, targets_28, alphas_28, FOCAL_GAMMA)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets_28)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

# ==================== 推理聚合 ====================
def cale_inference(logits, conds):
    probs = torch.sigmoid(logits).view(-1, NUM_CLASSES, NUM_CONDITIONS)
    one_hot = F.one_hot(conds.long(), NUM_CONDITIONS).float()
    scale   = INFERENCE_SCALE + (1 - INFERENCE_SCALE) * one_hot
    return (probs * scale.unsqueeze(1)).sum(dim=2)

# ==================== 评估 ====================
def evaluate(model, loader, mode='singlelabel', threshold=FIXED_THRESHOLD):
    model.eval()
    all_preds, all_targets, all_conds = [], [], []
    with torch.no_grad():
        for inputs, targets_7, conds, _ in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            conds  = conds.to(DEVICE).long()
            logits = model(inputs)
            p      = cale_inference(logits, conds)

            if mode == 'singlelabel':
                preds_idx = torch.argmax(p, dim=1)
                preds = torch.zeros_like(p)
                preds.scatter_(1, preds_idx.unsqueeze(1), 1.0)
            else:
                preds = (p > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets_7.numpy())
            all_conds.extend(conds.cpu().numpy())

    preds      = np.concatenate(all_preds, axis=0)
    targets    = np.concatenate(all_targets, axis=0)
    conditions = np.array(all_conds)

    acc      = accuracy_score(np.argmax(targets, axis=1),
                              np.argmax(preds, axis=1))
    uar      = recall_score(targets, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0)
    ham      = hamming_loss(targets, preds)
    hit_rate = (preds * targets).any(axis=1).mean()
    avg_lab  = preds.sum(axis=1).mean()
    per_f1   = f1_score(targets, preds, average=None, zero_division=0)

    cond_uar = {}
    for c in range(NUM_CONDITIONS):
        mask = conditions == c
        if mask.sum() > 0:
            cond_uar[c] = float(recall_score(
                targets[mask], preds[mask],
                average='macro', zero_division=0))
        else:
            cond_uar[c] = 0.0

    return {
        'accuracy': float(acc),
        'uar': float(uar), 'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1), 'hamming': float(ham),
        'hit_rate': float(hit_rate), 'avg_labels': float(avg_lab),
        'per_f1': per_f1, 'cond_uar': cond_uar,
    }

# ==================== 主程序 ====================
def main():
    print(f'Device: {DEVICE}')
    print(f'CALE 变体: {VARIANT} | USE_MS={USE_MS} | USE_SE={USE_SE}')
    print(f'软标签: 主分支=1.0, 其他分支={SOFT_LABEL}')
    print(f'推理: 主分支保持, 其他分支×{INFERENCE_SCALE}')
    print(f'验证集口径: singlelabel (argmax)')
    print(f'测试集口径: multilabel (阈值 {FIXED_THRESHOLD})')
    print(f'输出目录: {OUTPUT_DIR}')
    print(f'加权采样: {USE_WEIGHTED_SAMPLER}')
    print(f'训练数据比例: 10% subjects | 只跑第 1 折')
    print()

    alphas_28 = torch.tensor(CLASS_ALPHAS, dtype=torch.float32)
    alphas_28 = alphas_28.repeat_interleave(NUM_CONDITIONS).to(DEVICE)

    full_train = CALEDataset('Training', transform_train)
    all_samples = full_train.samples
    subject_ids = [extract_subject_id(s[0]) for s in all_samples]
    print(f'共提取到 {len(set(subject_ids))} 个不同实例\n')

    test_set = CALEDataset('Testing', transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)
    print(f'独立多标签测试集样本数: {len(test_set)}\n')

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(np.zeros(len(subject_ids)), groups=subject_ids)):
        if fold >= 1:
            break  # 只跑第 1 折

        fold_id = fold + 1
        print(f'\n{"="*80}')
        print(f'Fold {fold_id}/{K_FOLDS} (只跑此折)')
        print(f'{"="*80}')

        # ---- 10% subject 抽样 ----
        train_subjects = list(set([subject_ids[i] for i in train_idx]))
        rng = np.random.RandomState(SEED + fold)
        rng.shuffle(train_subjects)
        num_keep = max(1, int(len(train_subjects) * 0.1))
        kept_subjects = set(train_subjects[:num_keep])
        train_idx_sub = [i for i in train_idx if subject_ids[i] in kept_subjects]
        print(f'原始训练 subject 数: {len(train_subjects)}，'
              f'抽样后: {len(kept_subjects)}，'
              f'原始图像数: {len(train_idx)}，抽样后: {len(train_idx_sub)}')

        train_set = CALEDataset('Training', transform_train,
                                precomputed_samples=all_samples,
                                subset_indices=train_idx_sub)
        val_set   = CALEDataset('Validation', transform_test,
                                precomputed_samples=all_samples,
                                subset_indices=val_idx)

        train_sampler = build_weighted_sampler(train_set)
        if train_sampler is not None:
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                      sampler=train_sampler,
                                      num_workers=NUM_WORKERS,
                                      drop_last=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=NUM_WORKERS,
                                      drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True)

        model = ResNet18CALE(
            num_outputs=NUM_OUTPUTS, reduction=16,
            dropout=DROPOUT if USE_MS else 0.0,
            freeze_shallow=USE_MS,
            use_se=USE_SE,
        ).to(DEVICE)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-7)

        best_val_macro_f1 = -1.0
        best_val_epoch = 0
        patience_counter = 0
        best_state = None
        ckpt_path = os.path.join(OUTPUT_DIR, f'best_model_fold{fold_id}.pth')

        t_start = time.time()
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, alphas_28)
            val_metrics = evaluate(model, val_loader, mode='singlelabel')
            val_macro_f1 = val_metrics['macro_f1']

            print(f'[Fold {fold_id}] Epoch {epoch+1:3d}: '
                  f'loss={train_loss:.4f} | '
                  f'ValAcc={val_metrics["accuracy"]:.4f} | '
                  f'ValUAR={val_metrics["uar"]:.4f} | '
                  f'ValMacroF1={val_macro_f1:.4f}')

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_epoch = epoch + 1
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'[Fold {fold_id}] 早停 at epoch {epoch+1}')
                    break
            scheduler.step()

        elapsed = (time.time() - t_start) / 60
        print(f'[Fold {fold_id}] 训练耗时: {elapsed:.1f} 分钟')
        print(f'[Fold {fold_id}] 最佳 epoch: {best_val_epoch}, '
              f'Val MacroF1={best_val_macro_f1:.4f}')
        print(f'[Fold {fold_id}] 模型已保存: {ckpt_path}')

        model.load_state_dict(best_state)
        val_best = evaluate(model, val_loader, mode='singlelabel')
        test_metrics = evaluate(model, test_loader, mode='multilabel')

        print(f'\n[Fold {fold_id}] 验证集最终结果 (singlelabel):')
        print(f'  Accuracy={val_best["accuracy"]:.4f} | '
              f'UAR={val_best["uar"]:.4f} | '
              f'Macro-F1={val_best["macro_f1"]:.4f}')

        print(f'[Fold {fold_id}] 测试集补充结果 (multilabel, th={FIXED_THRESHOLD}):')
        print(f'  UAR={test_metrics["uar"]:.4f} | '
              f'Macro-F1={test_metrics["macro_f1"]:.4f} | '
              f'Micro-F1={test_metrics["micro_f1"]:.4f}')
        print(f'  HitRate={test_metrics["hit_rate"]:.4f} | '
              f'Hamming={test_metrics["hamming"]:.4f} | '
              f'AvgLabels={test_metrics["avg_labels"]:.4f}')

        fold_results.append({
            'fold': fold_id,
            'best_epoch': best_val_epoch,
            'val_accuracy': val_best['accuracy'],
            'val_uar': val_best['uar'],
            'val_macro_f1': val_best['macro_f1'],
            'test_uar': test_metrics['uar'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_micro_f1': test_metrics['micro_f1'],
            'test_hit_rate': test_metrics['hit_rate'],
            'test_hamming': test_metrics['hamming'],
            'test_avg_labels': test_metrics['avg_labels'],
            'cond_uar': val_best['cond_uar'],
            'per_f1': val_best['per_f1'].tolist(),
        })

    print(f'\n{"="*80}')
    print(f'{VARIANT} 10% 数据 1 折结果')
    print(f'{"="*80}')
    for k in ['val_accuracy', 'val_uar', 'val_macro_f1',
              'test_uar', 'test_macro_f1', 'test_micro_f1',
              'test_hit_rate', 'test_avg_labels']:
        vals = [r[k] for r in fold_results]
        print(f'{k:20s}: {np.mean(vals):.4f}')

    summary_path = os.path.join(OUTPUT_DIR, f'{VARIANT.lower()}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'fold_results': fold_results}, f, indent=2)
    print(f'\n结果已保存至 {summary_path}')
    print('训练完成。')

if __name__ == '__main__':
    main()