"""
V1 Baseline multi-label threshold scan
- V1 = ResNet-18 + 7-dim sigmoid output + Focal Loss + MS strategy
- Model structure: dropout=0.5, freeze_shallow=True, use_se=False (no SE)
- Inference: 7-dim logits -> sigmoid -> 7-dim probabilities
- Scan thresholds 0.01~0.99, find the operating point where AvgLabels is closest to 2.5
"""

import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    hamming_loss, f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== Configuration ====================
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'

MODEL_PATHS = [
    '/root/autodl-tmp/IEEE/baseline_v1_singlelabel/best_model_fold1.pth',
    '/root/autodl-tmp/IEEE/baseline_v1_singlelabel/best_model_fold2.pth',
    '/root/autodl-tmp/IEEE/baseline_v1_singlelabel/best_model_fold3.pth',
    '/root/autodl-tmp/IEEE/baseline_v1_singlelabel/best_model_fold4.pth',
    '/root/autodl-tmp/IEEE/baseline_v1_singlelabel/best_model_fold5.pth',
]

OUTPUT_DIR = '/root/autodl-tmp/IEEE/v1_multilabel_threshold_scan'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE  = 256
NUM_WORKERS = 8

# sigmoid probability range is 0~1, scan 0.01~0.99
THRESHOLDS = np.arange(0.01, 0.99, 0.01).round(3).tolist()

GUARANTEE_AT_LEAST_ONE = True
TARGET_DENSITY = 2.5

LOW_BRIGHTNESS_THRESH  = 80
HIGH_BRIGHTNESS_THRESH = 120
BLUR_VAR_THRESH        = 10

EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural']
NUM_CLASSES    = len(EMOTIONS)
CONDITIONS     = ['weak_light', 'strong_light', 'low_res', 'standard']
NUM_CONDITIONS = len(CONDITIONS)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Condition classifier ====================
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

# ==================== Dataset ====================
class CALETestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_path, transform, image_size=224):
        self.transform = transform
        self.image_size = image_size
        self.samples = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        print('Loading test set...')
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            img_no = parts[0]
            labels = list(map(int, parts[1:8]))
            label_vec = np.array(labels, dtype=np.float32)
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                p = os.path.join(img_dir, f'{img_no}{ext}')
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                continue
            cond = classify_condition(img_path)
            self.samples.append((img_path, label_vec, cond))
        self.samples.sort(key=lambda x: x[0])
        print(f'  Test set loaded: {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_vec_7, cond_idx = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return (img,
                torch.tensor(label_vec_7, dtype=torch.float32),
                cond_idx)

# ==================== V1 model (7-dim sigmoid, no SE) ====================
class ResNet18BaselineV1(nn.Module):
    """
    V1 structure:
    - ResNet-18 backbone
    - Shallow layers frozen (conv1-layer2)
    - dropout 0.5
    - 7-dim fc
    - No SE
    """
    def __init__(self, num_classes=7, dropout=0.5, freeze_shallow=True):
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
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== Metrics ====================
def compute_metrics(preds, targets):
    return {
        'uar': float(recall_score(targets, preds, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(targets, preds, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(targets, preds, average='macro', zero_division=0)),
        'ham': float(hamming_loss(targets, preds)),
        'avg_pred_labels': float(preds.sum(axis=1).mean()),
        'hit_rate': float((preds * targets).any(axis=1).mean()),
    }

def scan_thresholds(probs, targets, thresholds, guarantee_one=True):
    results = {}
    for th in thresholds:
        preds = (probs >= th).astype(np.int32)
        if guarantee_one:
            empty = preds.sum(axis=1) == 0
            if empty.any():
                preds[empty, np.argmax(probs[empty], axis=1)] = 1
        results[th] = compute_metrics(preds, targets)
    return results

# ==================== Main ====================
if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'V1 structure: ResNet-18 + 7-dim sigmoid + Focal + MS + Dropout 0.5 + shallow frozen')
    print(f'Threshold range: [{THRESHOLDS[0]}, {THRESHOLDS[-1]}] with {len(THRESHOLDS)} values')
    print(f'Target label density: {TARGET_DENSITY}')
    print(f'Output dir: {OUTPUT_DIR}\n')

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_set = CALETestDataset(TEST_DIR, LABEL_FILE, transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    avg_true_labels = np.mean([s[1].sum() for s in test_set.samples])
    print(f'True average number of labels: {avg_true_labels:.4f}\n')

    all_fold_results = []
    all_fold_best = []

    for fold_id, model_path in enumerate(MODEL_PATHS, start=1):
        print(f'{"="*80}')
        print(f'Fold {fold_id}: {os.path.basename(model_path)}')
        print(f'{"="*80}')

        if not os.path.exists(model_path):
            print(f'  [Skipped] Model not found')
            continue

        model = ResNet18BaselineV1(
            num_classes=NUM_CLASSES,
            dropout=0.5,
            freeze_shallow=True,
        ).to(DEVICE)

        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and 'net' in state:
            state = state['net']
        model.load_state_dict(state)
        model.eval()
        print(f'  Model loaded')

        all_probs, all_targets, all_conditions = [], [], []
        with torch.no_grad():
            for inputs, targets_7, cond_idx in test_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                logits = model(inputs)
                probs = torch.sigmoid(logits.float())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets_7.numpy())
                all_conditions.extend(cond_idx.numpy())

        probs = np.concatenate(all_probs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        conditions = np.array(all_conditions)

        # Threshold scan
        fold_results = scan_thresholds(
            probs, targets, THRESHOLDS,
            guarantee_one=GUARANTEE_AT_LEAST_ONE,
        )

        # Closest to density 2.5
        best_density_th = min(
            fold_results,
            key=lambda k: abs(fold_results[k]['avg_pred_labels'] - TARGET_DENSITY),
        )
        best_density_m = fold_results[best_density_th]

        # Best Macro-F1
        best_f1_th = max(fold_results, key=lambda k: fold_results[k]['macro_f1'])
        best_f1_m = fold_results[best_f1_th]

        # Best UAR
        best_uar_th = max(fold_results, key=lambda k: fold_results[k]['uar'])
        best_uar_m = fold_results[best_uar_th]

        # Fixed 0.5 reference
        th_05_key = min(fold_results.keys(), key=lambda k: abs(k - 0.5))
        m_05 = fold_results[th_05_key]

        print(f'  [Density≈2.5] th={best_density_th:.2f} | '
              f'AvgLabels={best_density_m["avg_pred_labels"]:.4f} | '
              f'UAR={best_density_m["uar"]:.4f} | MacroF1={best_density_m["macro_f1"]:.4f} | '
              f'MicroF1={best_density_m["micro_f1"]:.4f} | HitRate={best_density_m["hit_rate"]:.4f}')
        print(f'  [Best MacroF1] th={best_f1_th:.2f} | '
              f'AvgLabels={best_f1_m["avg_pred_labels"]:.4f} | '
              f'UAR={best_f1_m["uar"]:.4f} | MacroF1={best_f1_m["macro_f1"]:.4f}')
        print(f'  [Best UAR] th={best_uar_th:.2f} | '
              f'AvgLabels={best_uar_m["avg_pred_labels"]:.4f} | '
              f'UAR={best_uar_m["uar"]:.4f} | MacroF1={best_uar_m["macro_f1"]:.4f}')
        print(f'  [Fixed 0.5] AvgLabels={m_05["avg_pred_labels"]:.4f} | '
              f'UAR={m_05["uar"]:.4f} | MacroF1={m_05["macro_f1"]:.4f}')

        # Threshold sampling
        print(f'  Threshold sampling:')
        print(f'    {"th":>6} {"AvgLabels":>10} {"UAR":>8} {"MacroF1":>9} {"MicroF1":>9} {"HitRate":>9}')
        for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
            th_key = min(fold_results.keys(), key=lambda k: abs(k - th))
            m = fold_results[th_key]
            print(f'    {th_key:>6.2f} {m["avg_pred_labels"]:>10.4f} {m["uar"]:>8.4f} '
                  f'{m["macro_f1"]:>9.4f} {m["micro_f1"]:>9.4f} {m["hit_rate"]:>9.4f}')

        all_fold_results.append({
            'fold': fold_id,
            'model_path': model_path,
            'threshold_scan': {str(th): fold_results[th] for th in fold_results},
            'best_density_threshold': float(best_density_th),
            'best_density_metrics': best_density_m,
            'best_f1_threshold': float(best_f1_th),
            'best_f1_metrics': best_f1_m,
            'best_uar_threshold': float(best_uar_th),
            'best_uar_metrics': best_uar_m,
            'fixed_0.5_metrics': m_05,
        })
        all_fold_best.append({
            'fold': fold_id,
            'best_density_threshold': float(best_density_th),
            'best_density_metrics': best_density_m,
            'best_f1_threshold': float(best_f1_th),
            'best_f1_metrics': best_f1_m,
            'best_uar_threshold': float(best_uar_th),
            'best_uar_metrics': best_uar_m,
            'fixed_0.5_metrics': m_05,
        })

        del model
        torch.cuda.empty_cache()

    # ==================== 5-fold summary ====================
    print(f'\n{"="*80}')
    print(f'V1 5-fold threshold scan summary')
    print(f'{"="*80}')

    if len(all_fold_best) == 0:
        print('No valid results')
        raise SystemExit

    print('\nOperating point closest to density 2.5:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1'),
                       ('micro_f1', 'MicroF1'),
                       ('hit_rate', 'HitRate')]:
        vals = [r['best_density_metrics'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    ths = [r['best_density_threshold'] for r in all_fold_best]
    print(f'  {"Threshold":12s}: {np.mean(ths):.4f} ± {np.std(ths):.4f}')

    print('\nBest Macro-F1 operating point:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1'),
                       ('micro_f1', 'MicroF1'),
                       ('hit_rate', 'HitRate')]:
        vals = [r['best_f1_metrics'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    ths = [r['best_f1_threshold'] for r in all_fold_best]
    print(f'  {"Threshold":12s}: {np.mean(ths):.4f} ± {np.std(ths):.4f}')

    print('\nFixed 0.5 operating point:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1'),
                       ('micro_f1', 'MicroF1'),
                       ('hit_rate', 'HitRate')]:
        vals = [r['fixed_0.5_metrics'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    # ==================== Comparison table ====================
    print(f'\n{"="*80}')
    print(f'V1 vs V0 vs V2 (AvgLabels ≈ 2.5)')
    print(f'{"="*80}')
    print(f'{"Method":<24} {"AvgLabels":>10} {"UAR":>8} {"MacroF1":>9} {"MicroF1":>9} {"HitRate":>9}')
    print(f'{"V0":<24} {"2.5029":>10} {"0.5716":>8} {"0.5607":>9} {"0.5814":>9} {"0.9380":>9}')
    print(f'{"V2":<24} {"2.4720":>10} {"0.5817":>8} {"0.5761":>9} {"0.5968":>9} {"0.9553":>9}')
    v1_uar = np.mean([r['best_density_metrics']['uar'] for r in all_fold_best])
    v1_f1 = np.mean([r['best_density_metrics']['macro_f1'] for r in all_fold_best])
    v1_mf1 = np.mean([r['best_density_metrics']['micro_f1'] for r in all_fold_best])
    v1_hr = np.mean([r['best_density_metrics']['hit_rate'] for r in all_fold_best])
    v1_al = np.mean([r['best_density_metrics']['avg_pred_labels'] for r in all_fold_best])
    print(f'{"V1":<24} {v1_al:>10.4f} {v1_uar:>8.4f} {v1_f1:>9.4f} '
          f'{v1_mf1:>9.4f} {v1_hr:>9.4f}')
    print(f'  V1 vs V0: UAR {v1_uar - 0.5716:+.4f} | Macro-F1 {v1_f1 - 0.5607:+.4f}')
    print(f'  V1 vs V2: UAR {v1_uar - 0.5817:+.4f} | Macro-F1 {v1_f1 - 0.5761:+.4f}')

    # ==================== Save ====================
    summary = {
        'variant': 'V1',
        'target_density': TARGET_DENSITY,
        'thresholds': THRESHOLDS,
        'avg_true_labels': float(avg_true_labels),
        'per_fold': all_fold_results,
        'fold_summary': all_fold_best,
        'references': {
            'V0': {'avg_pred_labels': 2.5029, 'uar': 0.5716,
                   'macro_f1': 0.5607, 'micro_f1': 0.5814, 'hit_rate': 0.9380},
            'V2': {'avg_pred_labels': 2.4720, 'uar': 0.5817,
                   'macro_f1': 0.5761, 'micro_f1': 0.5968, 'hit_rate': 0.9553},
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'v1_multilabel_threshold_scan.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}')

    # ==================== Plot ====================
    common_ths = sorted(set(THRESHOLDS))
    metrics_keys = ['avg_pred_labels', 'uar', 'macro_f1', 'micro_f1', 'hit_rate']

    avg_curves = {k: [] for k in metrics_keys}
    for th in common_ths:
        per_fold = []
        for r in all_fold_results:
            th_str = str(th)
            if th_str in r['threshold_scan']:
                per_fold.append(r['threshold_scan'][th_str])
        if len(per_fold) == 0:
            for k in metrics_keys:
                avg_curves[k].append(np.nan)
            continue
        for k in metrics_keys:
            avg_curves[k].append(np.mean([m[k] for m in per_fold]))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(common_ths, avg_curves['uar'], 'o-', label='V1 UAR', markersize=3)
    plt.plot(common_ths, avg_curves['macro_f1'], 's-', label='V1 Macro-F1', markersize=3)
    plt.plot(common_ths, avg_curves['micro_f1'], '^-', label='V1 Micro-F1', markersize=3)
    plt.axhline(0.5716, color='red', linestyle='--', alpha=0.4, label='V0 UAR @ 2.5')
    plt.axhline(0.5607, color='blue', linestyle='--', alpha=0.4, label='V0 Macro-F1 @ 2.5')
    plt.axhline(0.5817, color='orange', linestyle='--', alpha=0.4, label='V2 UAR @ 2.5')
    plt.axhline(0.5761, color='purple', linestyle='--', alpha=0.4, label='V2 Macro-F1 @ 2.5')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('V1 Multi-label Threshold Scan')
    plt.legend(fontsize=8)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(common_ths, avg_curves['avg_pred_labels'], 'o-',
             label='V1 Avg Predicted Labels', markersize=3)
    plt.axhline(TARGET_DENSITY, color='red', linestyle='--', alpha=0.7,
                label=f'Target = {TARGET_DENSITY}')
    plt.axhline(avg_true_labels, color='gray', linestyle='--', alpha=0.5,
                label=f'True Avg = {avg_true_labels:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Avg Predicted Labels')
    plt.title('V1 Avg Predicted Labels vs Threshold')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'v1_threshold_scan.png'), dpi=200)
    plt.close()
    print(f'Curve plot saved to {OUTPUT_DIR}')
    print('\nScan complete.')
