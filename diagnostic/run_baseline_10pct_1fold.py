# run_baseline_10pct_1fold.py
"""
V0 Baseline —— 10% subject 抽样，1 折，30 epoch
超参数已与 CALE V2 对齐：AdamW, lr=1e-5, wd=5e-3, 模型选择 Macro-F1
"""

import os
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import GroupKFold

# ==================== 配置 ====================
CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'
OUTPUT_DIR = '/root/autodl-tmp/IEEE/baseline_10pct_1fold'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE    = 128
LEARNING_RATE = 1e-5
EPOCHS        = 30
PATIENCE      = 10
SEED          = 42
WEIGHT_DECAY  = 5e-3
K_FOLDS       = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural']
NUM_CLASSES = len(EMOTIONS)

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

# 条件分类器（仅用于分组分析，不影响训练）
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

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None,
                 precomputed_samples=None, subset_indices=None):
        self.transform = transform
        self.samples = []
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
        if not root.exists():
            raise FileNotFoundError(f'训练目录不存在: {root_dir}')
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            emo = folder_to_emotion(folder.name)
            if emo is None:
                continue
            emo_idx = EMOTIONS.index(emo)
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            label[emo_idx] = 1.0
            for img_file in folder.glob('*.*'):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    cond = classify_condition(str(img_file))
                    self.samples.append((str(img_file), label, emo_idx, cond))
        self.samples.sort(key=lambda x: x[0])
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
                    self.samples.append((p, labels, -1, cond))
                    break
        self.samples.sort(key=lambda x: x[0])
        print(f'测试集加载完毕: {len(self.samples)} 张图片')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _, cond = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), cond


class ResNet18V0(nn.Module):
    """标准 ResNet-18，全部可训练，7 维输出"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inputs, targets, _ in loader:
        inputs = inputs.to(DEVICE)
        labels = torch.argmax(targets, dim=1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    """softmax + argmax → multi-hot → 多标签指标"""
    model.eval()
    all_probs, all_targets, all_conditions = [], [], []
    with torch.no_grad():
        for inputs, targets, conds in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_conditions.extend(conds.numpy())

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    conditions = np.array(all_conditions)

    preds = np.argmax(probs, axis=1)
    preds_multi = np.zeros_like(targets)
    preds_multi[np.arange(len(preds)), preds] = 1

    uar = recall_score(targets, preds_multi, average='macro', zero_division=0)
    macro_f1 = f1_score(targets, preds_multi, average='macro', zero_division=0)
    micro_f1 = f1_score(targets, preds_multi, average='micro', zero_division=0)
    ham = hamming_loss(targets, preds_multi)
    label_acc = 1 - ham
    inter = (preds_multi * targets).any(axis=1)
    hit_rate = inter.mean()
    per_prec = precision_score(targets, preds_multi, average=None, zero_division=0)
    per_rec = recall_score(targets, preds_multi, average=None, zero_division=0)
    per_f1 = f1_score(targets, preds_multi, average=None, zero_division=0)

    return {
        'uar': float(uar), 'macro_f1': float(macro_f1), 'micro_f1': float(micro_f1),
        'label_acc': float(label_acc), 'hit_rate': float(hit_rate), 'hamming': float(ham),
        'per_prec': per_prec, 'per_rec': per_rec, 'per_f1': per_f1,
    }, preds_multi, targets, conditions


def main():
    print(f'Device: {DEVICE}')
    print('V0 Baseline: ResNet-18 + 7维 + CrossEntropy + 全部可训练')
    print('超参对齐: AdamW lr=1e-5 wd=5e-3, 模型选择 Macro-F1')
    print('训练数据比例: 10% subjects | 只跑第 1 折')

    full_train = EmotionDataset('Training', transform_train)
    all_samples = full_train.samples
    subject_ids = [extract_subject_id(s[0]) for s in all_samples]
    print(f'共提取到 {len(set(subject_ids))} 个不同实例')

    test_set = EmotionDataset('Testing', transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8)
    print(f'独立测试集样本数: {len(test_set)}')

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(np.zeros(len(subject_ids)), groups=subject_ids)):
        if fold >= 1:
            break  # 只跑第 1 折

        print(f'\n==================== Fold {fold+1}/{K_FOLDS} (只跑此折) ====================')

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

        train_set = EmotionDataset('Training', transform_train,
                                   precomputed_samples=all_samples,
                                   subset_indices=train_idx_sub)
        val_set = EmotionDataset('Validation', transform_test,
                                 precomputed_samples=all_samples,
                                 subset_indices=val_idx)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=8)

        model = ResNet18V0(NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(),
                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-7)

        best_val_macro_f1 = -1.0
        best_val_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_metrics, _, _, _ = evaluate(model, val_loader)
            val_macro_f1 = val_metrics['macro_f1']
            print(f'Epoch {epoch+1}: loss={train_loss:.4f} | '
                  f'Val UAR={val_metrics["uar"]:.4f} | '
                  f'Macro-F1={val_macro_f1:.4f}')

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_epoch = epoch + 1
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'早停 at epoch {epoch+1}')
                    break
            scheduler.step()

        model.load_state_dict(best_state)
        test_metrics, _, _, _ = evaluate(model, test_loader)

        print(f'\n--- Fold {fold+1} 测试集结果 ---')
        print(f'UAR: {test_metrics["uar"]:.4f} | '
              f'Macro-F1: {test_metrics["macro_f1"]:.4f} | '
              f'Micro-F1: {test_metrics["micro_f1"]:.4f}')
        print(f'Hit Rate: {test_metrics["hit_rate"]:.4f} | '
              f'Hamming: {test_metrics["hamming"]:.4f}')

        fold_results.append({
            'fold': fold + 1,
            'best_epoch': best_val_epoch,
            'val_uar': float(val_metrics['uar']),
            'val_macro_f1': float(best_val_macro_f1),
            'test_uar': test_metrics['uar'],
            'test_macro_f1': test_metrics['macro_f1'],
            'test_micro_f1': test_metrics['micro_f1'],
            'test_hit_rate': test_metrics['hit_rate'],
            'test_hamming': test_metrics['hamming'],
            'test_label_acc': test_metrics['label_acc'],
        })

    print('\n==================== V0 Baseline 10% 数据 1 折结果 ====================')
    for k in ['val_uar', 'val_macro_f1', 'test_uar', 'test_macro_f1',
              'test_micro_f1', 'test_hit_rate', 'test_hamming', 'test_label_acc']:
        vals = [r[k] for r in fold_results]
        print(f'{k:20s}: {np.mean(vals):.4f}')

    with open(os.path.join(OUTPUT_DIR, 'v0_summary.json'), 'w') as f:
        json.dump({'fold_results': fold_results}, f, indent=2)
    print(f'\n结果已保存至 {OUTPUT_DIR}/v0_summary.json')


if __name__ == '__main__':
    main()