# run_cale_v2_10pct_1fold_softlabel0.py
"""
CALE V2 (LE only, SOFT_LABEL=0.0) - 10% subject sampling, 1 fold, 30 epochs
Diagnostic experiment: other condition positions set to 0 during training (only main branch supervised)
"""
import os, json, random, time
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
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, recall_score
from sklearn.model_selection import GroupKFold

# ==================== Configuration ====================
CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'

VARIANT = 'V2'
SOFT_LABEL      = 0.0          # <<< only change: 0.5 -> 0.0
INFERENCE_SCALE = 0.5
FIXED_THRESHOLD = 0.5

OUTPUT_DIR = '/root/autodl-tmp/IEEE/cale_v2_10pct_1fold_softlabel0'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE    = 128
LEARNING_RATE = 1e-5
EPOCHS        = 30
PATIENCE      = 10
SEED          = 42
WEIGHT_DECAY  = 5e-3
K_FOLDS       = 5
NUM_WORKERS   = 8

FOCAL_GAMMA  = 2.0
CLASS_ALPHAS = [0.9, 0.8, 1.0, 0.3, 0.5, 0.7, 0.2]
DROPOUT      = 0.5
USE_WEIGHTED_SAMPLER = True

USE_MS = False
USE_SE = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural']
NUM_CLASSES    = len(EMOTIONS)
NUM_CONDITIONS = 4
NUM_OUTPUTS    = NUM_CLASSES * NUM_CONDITIONS

# ==================== Data augmentation ====================
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

# ==================== Condition classifier ====================
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

# ==================== Dataset ====================
class CALEDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, precomputed_samples=None, subset_indices=None):
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
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            emo = folder_to_emotion(folder.name)
            if emo is None:
                continue
            emo_idx = EMOTIONS.index(emo)
            label = np.zeros(NUM_CLASSES, dtype=np.float32); label[emo_idx] = 1.0
            for img_file in sorted(folder.glob('*.*')):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    cond = classify_condition(str(img_file))
                    self.samples.append((str(img_file), label, cond, emo_idx))
        print(f'Training set loaded: {len(self.samples)} images')

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
                    self.samples.append((p, labels, classify_condition(p), -1))
                    break
        self.samples.sort(key=lambda x: x[0])
        print(f'Test set loaded: {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, cond, emo_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), cond, emo_idx

def build_weighted_sampler(dataset):
    if not USE_MS or not USE_WEIGHTED_SAMPLER:
        return None
    class_counts = [0] * NUM_CLASSES
    for _, _, _, emo_idx in dataset.samples:
        if emo_idx >= 0:
            class_counts[emo_idx] += 1
    class_counts = [max(c, 1) for c in class_counts]
    class_weights = [1.0 / c for c in class_counts]
    weights = [class_weights[emo_idx] if emo_idx >= 0 else 1.0
               for _, _, _, emo_idx in dataset.samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# ==================== SE / Model ====================
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
    def __init__(self, num_outputs=28, dropout=0.0, freeze_shallow=False, use_se=False):
        super().__init__()
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_shallow:
            for name, p in bb.named_parameters():
                p.requires_grad = ('layer3' in name or 'layer4' in name or 'fc' in name)
        self.conv1, self.bn1, self.relu = bb.conv1, bb.bn1, bb.relu
        self.maxpool = bb.maxpool
        self.layer1, self.layer2 = bb.layer1, bb.layer2
        self.layer3, self.layer4 = bb.layer3, bb.layer4
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

# ==================== 28-dim soft label (SOFT_LABEL=0.0) ====================
def build_targets_28(emo_indices, cond_indices):
    B = emo_indices.size(0)
    t28 = torch.zeros(B, NUM_CLASSES, NUM_CONDITIONS, device=emo_indices.device)
    arange = torch.arange(B, device=emo_indices.device)
    # When SOFT_LABEL=0.0, this loop is equivalent to writing nothing
    for c in range(NUM_CONDITIONS):
        t28[arange, emo_indices, c] = SOFT_LABEL
    t28[arange, emo_indices, cond_indices] = 1.0
    return t28.view(B, NUM_OUTPUTS)

def soft_focal_loss(logits, targets, alphas_28, gamma=2.0):
    probs = torch.sigmoid(logits)
    pos_loss = -targets * (1 - probs).pow(gamma) * F.logsigmoid(logits)
    neg_loss = -(1 - targets) * probs.pow(gamma) * F.logsigmoid(-logits)
    alpha_w = torch.where(targets > 0,
                          alphas_28.unsqueeze(0).expand_as(targets),
                          torch.ones_like(targets))
    return (alpha_w * (pos_loss + neg_loss)).mean()

# ==================== Training / Inference / Evaluation ====================
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

def cale_inference(logits, conds):
    probs = torch.sigmoid(logits).view(-1, NUM_CLASSES, NUM_CONDITIONS)
    one_hot = F.one_hot(conds.long(), NUM_CONDITIONS).float()
    scale = INFERENCE_SCALE + (1 - INFERENCE_SCALE) * one_hot
    return (probs * scale.unsqueeze(1)).sum(dim=2)

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
                idx = torch.argmax(p, dim=1)
                preds = torch.zeros_like(p)
                preds.scatter_(1, idx.unsqueeze(1), 1.0)
            else:
                preds = (p > threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets_7.numpy())
            all_conds.extend(conds.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    conditions = np.array(all_conds)
    acc = accuracy_score(np.argmax(targets, axis=1), np.argmax(preds, axis=1))
    uar = recall_score(targets, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0)
    ham = hamming_loss(targets, preds)
    hit_rate = (preds * targets).any(axis=1).mean()
    avg_lab = preds.sum(axis=1).mean()
    per_f1 = f1_score(targets, preds, average=None, zero_division=0)
    cond_uar = {}
    for c in range(NUM_CONDITIONS):
        mask = conditions == c
        cond_uar[c] = float(recall_score(targets[mask], preds[mask],
                                          average='macro', zero_division=0)) if mask.sum() > 0 else 0.0
    return {'accuracy': float(acc), 'uar': float(uar), 'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1), 'hamming': float(ham),
            'hit_rate': float(hit_rate), 'avg_labels': float(avg_lab),
            'per_f1': per_f1, 'cond_uar': cond_uar}

# ==================== Main ====================
def main():
    print(f'Device: {DEVICE}')
    print(f'CALE V2 | SOFT_LABEL={SOFT_LABEL} | USE_MS={USE_MS} | USE_SE={USE_SE}')
    print(f'Training data ratio: 10% subjects | only fold 1')
    print()

    alphas_28 = torch.tensor(CLASS_ALPHAS, dtype=torch.float32)
    alphas_28 = alphas_28.repeat_interleave(NUM_CONDITIONS).to(DEVICE)

    full_train = CALEDataset('Training', transform_train)
    all_samples = full_train.samples
    subject_ids = [extract_subject_id(s[0]) for s in all_samples]
    print(f'Extracted {len(set(subject_ids))} unique subjects\n')

    test_set = CALEDataset('Testing', transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f'Independent multi-label test set size: {len(test_set)}\n')

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(np.zeros(len(subject_ids)), groups=subject_ids)):
        if fold >= 1:
            break
        fold_id = fold + 1
        print(f'\n{"="*80}\nFold {fold_id}/5 (only this fold)\n{"="*80}')

        train_subjects = list(set([subject_ids[i] for i in train_idx]))
        rng = np.random.RandomState(SEED + fold)
        rng.shuffle(train_subjects)
        num_keep = max(1, int(len(train_subjects) * 0.1))
        kept_subjects = set(train_subjects[:num_keep])
        train_idx_sub = [i for i in train_idx if subject_ids[i] in kept_subjects]
        print(f'Original training subjects: {len(train_subjects)}, after sampling: {len(kept_subjects)}, '
              f'images: {len(train_idx)} -> {len(train_idx_sub)}')

        train_set = CALEDataset('Training', transform_train,
                                precomputed_samples=all_samples,
                                subset_indices=train_idx_sub)
        val_set = CALEDataset('Validation', transform_test,
                              precomputed_samples=all_samples,
                              subset_indices=val_idx)

        sampler = build_weighted_sampler(train_set)
        if sampler is not None:
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                                      num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                      num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        model = ResNet18CALE(num_outputs=NUM_OUTPUTS, dropout=0.0,
                             freeze_shallow=False, use_se=False).to(DEVICE)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

        best_val_macro_f1 = -1.0
        best_val_epoch = 0
        patience = 0
        best_state = None
        ckpt = os.path.join(OUTPUT_DIR, f'best_model_fold{fold_id}.pth')

        t0 = time.time()
        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, alphas_28)
            vm = evaluate(model, val_loader, mode='singlelabel')
            print(f'[Fold {fold_id}] Epoch {epoch+1:3d}: loss={loss:.4f} | '
                  f'ValAcc={vm["accuracy"]:.4f} | ValUAR={vm["uar"]:.4f} | '
                  f'ValMacroF1={vm["macro_f1"]:.4f}')
            if vm['macro_f1'] > best_val_macro_f1:
                best_val_macro_f1 = vm['macro_f1']
                best_val_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, ckpt)
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f'[Fold {fold_id}] Early stopping at epoch {epoch+1}')
                    break
            scheduler.step()

        print(f'[Fold {fold_id}] Time {(time.time()-t0)/60:.1f} min | '
              f'Best epoch {best_val_epoch} | Val MacroF1={best_val_macro_f1:.4f}')

        model.load_state_dict(best_state)
        val_best = evaluate(model, val_loader, mode='singlelabel')
        test_metrics = evaluate(model, test_loader, mode='multilabel')

        print(f'\n[Fold {fold_id}] Validation (singlelabel): '
              f'Acc={val_best["accuracy"]:.4f} | UAR={val_best["uar"]:.4f} | '
              f'MacroF1={val_best["macro_f1"]:.4f}')
        print(f'[Fold {fold_id}] Test (multilabel th={FIXED_THRESHOLD}): '
              f'UAR={test_metrics["uar"]:.4f} | MacroF1={test_metrics["macro_f1"]:.4f} | '
              f'MicroF1={test_metrics["micro_f1"]:.4f} | Hit={test_metrics["hit_rate"]:.4f} | '
              f'AvgLabels={test_metrics["avg_labels"]:.4f}')

        fold_results.append({
            'fold': fold_id, 'best_epoch': best_val_epoch,
            'val_accuracy': val_best['accuracy'], 'val_uar': val_best['uar'],
            'val_macro_f1': val_best['macro_f1'],
            'test_uar': test_metrics['uar'], 'test_macro_f1': test_metrics['macro_f1'],
            'test_micro_f1': test_metrics['micro_f1'],
            'test_hit_rate': test_metrics['hit_rate'],
            'test_avg_labels': test_metrics['avg_labels'],
        })

    print(f'\n{"="*80}\nSOFT_LABEL={SOFT_LABEL} 10% 1-fold results\n{"="*80}')
    for k in ['val_accuracy', 'val_uar', 'val_macro_f1',
              'test_uar', 'test_macro_f1', 'test_micro_f1',
              'test_hit_rate', 'test_avg_labels']:
        vals = [r[k] for r in fold_results]
        print(f'{k:20s}: {np.mean(vals):.4f}')

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump({'soft_label': SOFT_LABEL, 'fold_results': fold_results}, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}/summary.json')

if __name__ == '__main__':
    main()
