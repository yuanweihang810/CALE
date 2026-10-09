"""
Baseline training script - unified 5-fold split, comparable with CALE
Supports V0 / V1 / V3 / V4
- V0: ResNet-18 + 7-dim + CrossEntropy + fully trainable + basic augmentation
- V1: ResNet-18 + 7-dim + Sigmoid Focal + shallow freeze + Dropout + strong augmentation + weighted sampling
- V3: ResNet-18 + SE + 7-dim + CrossEntropy + fully trainable + basic augmentation
- V4: ResNet-18 + SE + 7-dim + Sigmoid Focal + shallow freeze + Dropout + strong augmentation + weighted sampling

Consistent with the CALE scripts:
- Same sample loading order (sorted by path)
- Same subject_id extraction (basename.split('_')[0])
- Same GroupKFold(5)
- Same best_val_macro_f1 model selection criterion
- Same single-label argmax validation protocol
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
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import GroupKFold

# ==================== Configuration ====================
CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'

VARIANT = 'V3'    # 'V0' | 'V1' | 'V3' | 'V4'
OUTPUT_DIR = f'/root/autodl-tmp/IEEE/baseline_{VARIANT.lower()}_singlelabel'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE    = 128
LEARNING_RATE = 1e-5
EPOCHS        = 100
PATIENCE      = 30
SEED          = 42
WEIGHT_DECAY  = 5e-3
K_FOLDS       = 5
NUM_WORKERS   = 8

FOCAL_GAMMA  = 2.0
CLASS_ALPHAS = [0.9, 0.8, 1.0, 0.3, 0.5, 0.7, 0.2]
DROPOUT      = 0.5

# Variant configuration
VARIANT_CONFIG = {
    'V0': {'use_ms': False, 'use_se': False},
    'V1': {'use_ms': True,  'use_se': False},
    'V3': {'use_ms': False, 'use_se': True},
    'V4': {'use_ms': True,  'use_se': True},
}
CFG     = VARIANT_CONFIG[VARIANT]
USE_MS  = CFG['use_ms']
USE_SE  = CFG['use_se']

USE_WEIGHTED_SAMPLER = USE_MS   # only enabled when MS is on

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
CONDITION_NAMES = ['weak_light', 'strong_light', 'low_res', 'standard']

# ==================== Data augmentation ====================
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

# ==================== Condition classifier (only used for grouping analysis) ====================
LOW_BRIGHTNESS_THRESH  = 80
HIGH_BRIGHTNESS_THRESH = 120
BLUR_VAR_THRESH        = 10

def classify_condition(image_path):
    try:
        with Image.open(image_path) as img:
            gray = img.convert('L')
            arr = np.array(gray, dtype=np.float64)
        brightness = arr.mean()
        laplacian  = cv2.Laplacian(arr, cv2.CV_64F)
        variance   = laplacian.var()
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

# ==================== Dataset ====================
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None,
                 precomputed_samples=None, subset_indices=None):
        self.transform = transform
        self.samples   = []
        if precomputed_samples is not None:
            self.samples = precomputed_samples
        else:
            self._load_train(CUSTOM_DIR)
        if subset_indices is not None:
            self.samples = [self.samples[i] for i in subset_indices]

    def _load_train(self, root_dir):
        root = Path(root_dir)
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            emo = folder_to_emotion(folder.name)
            if emo is None:
                continue
            emo_idx = EMOTIONS.index(emo)
            for img_file in folder.glob('*.*'):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    cond = classify_condition(str(img_file))
                    self.samples.append((str(img_file), emo_idx, cond))
        # Consistent with the CALE script: sorted by path
        self.samples.sort(key=lambda x: x[0])
        print(f'Training set loaded: {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, emo_idx, cond = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, emo_idx, cond

# ==================== Weighted sampler ====================
def build_weighted_sampler(dataset):
    if not USE_WEIGHTED_SAMPLER:
        return None
    class_counts = [0] * NUM_CLASSES
    for _, emo_idx, _ in dataset.samples:
        class_counts[emo_idx] += 1
    class_counts = [max(c, 1) for c in class_counts]
    class_weights = [1.0 / c for c in class_counts]
    weights = [class_weights[emo_idx] for _, emo_idx, _ in dataset.samples]
    sampler = WeightedRandomSampler(weights=weights,
                                    num_samples=len(weights),
                                    replacement=True)
    print(f'Weighted sampler: class counts {class_counts}')
    return sampler

# ==================== SE module ====================
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

# ==================== Baseline model (7-dim) ====================
class ResNet18Baseline(nn.Module):
    """
    ResNet-18 + optional SE + 7-dim output
    V0: fully trainable, no dropout, no SE
    V1: shallow freeze, dropout 0.5, no SE
    V3: fully trainable, no dropout, with SE
    V4: shallow freeze, dropout 0.5, with SE
    """
    def __init__(self, num_classes=7, reduction=16,
                 dropout=0.0, freeze_shallow=False, use_se=False):
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
        self.fc      = nn.Linear(in_features, num_classes)

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

# ==================== Sigmoid Focal Loss ====================
def sigmoid_focal_loss(inputs, targets_1hot, alphas, gamma=2.0):
    """
    inputs:      (B, 7) logits
    targets_1hot:(B, 7) one-hot
    alphas:      (7,) per-class positive sample weights
    """
    bce = F.binary_cross_entropy_with_logits(inputs, targets_1hot, reduction='none')
    p_t = torch.exp(-bce)
    focal_weight = (1 - p_t) ** gamma
    weight = torch.where(targets_1hot == 1,
                         alphas.to(inputs.device),
                         torch.ones_like(targets_1hot))
    loss = focal_weight * weight * bce
    return loss.mean()

# ==================== Training ====================
def train_epoch(model, loader, optimizer, alphas, criterion_type='ce'):
    model.train()
    total_loss = 0.0
    for inputs, emo_idx, _ in loader:
        inputs  = inputs.to(DEVICE, non_blocking=True)
        emo_idx = emo_idx.to(DEVICE).long()

        optimizer.zero_grad()
        logits = model(inputs)

        if criterion_type == 'ce':
            loss = F.cross_entropy(logits, emo_idx)
        else:
            targets_1hot = F.one_hot(emo_idx, NUM_CLASSES).float()
            loss = sigmoid_focal_loss(logits, targets_1hot, alphas, FOCAL_GAMMA)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

# ==================== Evaluation (single-label argmax) ====================
def evaluate_singlelabel(model, loader, criterion_type='ce'):
    model.eval()
    all_preds, all_targets, all_conds = [], [], []
    with torch.no_grad():
        for inputs, emo_idx, conds in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            logits = model(inputs)

            if criterion_type == 'ce':
                probs = F.softmax(logits, dim=1)
            else:
                probs = torch.sigmoid(logits)

            preds_idx = torch.argmax(probs, dim=1)
            all_preds.append(preds_idx.cpu().numpy())
            all_targets.append(emo_idx.numpy())
            all_conds.extend(conds.numpy())

    preds      = np.concatenate(all_preds)
    targets    = np.concatenate(all_targets)
    conditions = np.array(all_conds)

    acc      = accuracy_score(targets, preds)
    uar      = recall_score(targets, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
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
        'accuracy': float(acc), 'uar': float(uar),
        'macro_f1': float(macro_f1),
        'per_f1': per_f1, 'cond_uar': cond_uar,
    }

# ==================== Main ====================
def main():
    print(f'Device: {DEVICE}')
    print(f'Baseline variant: {VARIANT}')
    print(f'USE_MS={USE_MS} | USE_SE={USE_SE} | '
          f'DROPOUT={DROPOUT if USE_MS else 0.0} | '
          f'FREEZE={USE_MS} | WEIGHTED_SAMPLER={USE_WEIGHTED_SAMPLER}')
    print(f'Output dir: {OUTPUT_DIR}')
    print()

    criterion_type = 'ce' if not USE_MS else 'focal'

    alphas = torch.tensor(CLASS_ALPHAS, dtype=torch.float32).to(DEVICE)

    # Load full training set
    full_train = EmotionDataset(transform=transform_train)
    all_samples = full_train.samples
    subject_ids = [extract_subject_id(s[0]) for s in all_samples]
    print(f'Extracted {len(set(subject_ids))} unique subjects\n')

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(np.zeros(len(subject_ids)), groups=subject_ids)):
        fold_id = fold + 1
        print(f'\n{"="*80}')
        print(f'Fold {fold_id}/{K_FOLDS}')
        print(f'{"="*80}')

        train_set = EmotionDataset(transform=transform_train,
                                   precomputed_samples=all_samples,
                                   subset_indices=train_idx)
        val_set   = EmotionDataset(transform=transform_test,
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

        model = ResNet18Baseline(
            num_classes=NUM_CLASSES, reduction=16,
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
            train_loss = train_epoch(model, train_loader, optimizer, alphas,
                                     criterion_type)
            val_metrics = evaluate_singlelabel(model, val_loader, criterion_type)
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
                    print(f'[Fold {fold_id}] Early stopping at epoch {epoch+1}')
                    break
            scheduler.step()

        elapsed = (time.time() - t_start) / 60
        print(f'[Fold {fold_id}] Training time: {elapsed:.1f} min')
        print(f'[Fold {fold_id}] Best epoch: {best_val_epoch}, '
              f'Val MacroF1={best_val_macro_f1:.4f}')
        print(f'[Fold {fold_id}] Model saved: {ckpt_path}')

        # Load best model for final evaluation
        model.load_state_dict(best_state)
        val_best = evaluate_singlelabel(model, val_loader, criterion_type)

        print(f'\n[Fold {fold_id}] Final validation results:')
        print(f'  Accuracy={val_best["accuracy"]:.4f} | '
              f'UAR={val_best["uar"]:.4f} | '
              f'Macro-F1={val_best["macro_f1"]:.4f}')

        fold_results.append({
            'fold': fold_id,
            'best_epoch': best_val_epoch,
            'val_accuracy': val_best['accuracy'],
            'val_uar': val_best['uar'],
            'val_macro_f1': val_best['macro_f1'],
            'cond_uar': val_best['cond_uar'],
            'per_f1': val_best['per_f1'].tolist(),
        })

    # ==================== Summary ====================
    print(f'\n{"="*80}')
    print(f'{VARIANT} 5-fold summary')
    print(f'{"="*80}')
    for k, label in [('val_accuracy', 'Val Accuracy'),
                     ('val_uar', 'Val UAR'),
                     ('val_macro_f1', 'Val Macro-F1')]:
        vals = [r[k] for r in fold_results]
        print(f'{label:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    numeric_keys = [k for k in fold_results[0]
                    if k != 'fold'
                    and not isinstance(fold_results[0][k], (dict, list))]
    summary = {
        'variant': VARIANT,
        'use_ms': USE_MS,
        'use_se': USE_SE,
        'criterion_type': criterion_type,
        'fold_results': fold_results,
        'mean': {k: float(np.mean([r[k] for r in fold_results]))
                 for k in numeric_keys},
        'std': {k: float(np.std([r[k] for r in fold_results]))
                for k in numeric_keys},
    }
    summary_path = os.path.join(OUTPUT_DIR, f'{VARIANT.lower()}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved to {summary_path}')
    print('Training complete.')

if __name__ == '__main__':
    main()
