"""
CALE V2 5-fold model threshold scan
- V2 configuration: USE_MS=False, USE_SE=False
- Model structure: dropout=0.0, freeze_shallow=False, use_se=False
- Scan thresholds 0.005~0.50 (V2 outputs are low)
- Directly comparable with V0/V7 results
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

VARIANT = 'V2'
INFERENCE_SCALE = 0.5
TARGET_DENSITY = 2.5

# Automatic model structure matching
VARIANT_CONFIG = {
    'V2': {'use_ms': False, 'use_se': False},
    'V5': {'use_ms': True,  'use_se': False},
    'V6': {'use_ms': False, 'use_se': True},
    'V7': {'use_ms': True,  'use_se': True},
}
CFG = VARIANT_CONFIG[VARIANT]
USE_MS = CFG['use_ms']
USE_SE = CFG['use_se']
DROPOUT = 0.5 if USE_MS else 0.0
FREEZE_SHALLOW = USE_MS

MODEL_PATHS = [
    f'/root/autodl-tmp/IEEE/cale_{VARIANT.lower()}_singlelabel/best_model_fold{i}.pth'
    for i in range(1, 6)
]

OUTPUT_DIR = f'/root/autodl-tmp/IEEE/cale_{VARIANT.lower()}_multilabel_threshold_scan'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE  = 256
NUM_WORKERS = 8

# V2 outputs are low, so the threshold range should be low
if VARIANT == 'V2':
    THRESHOLDS_SOFT = np.arange(0.005, 0.51, 0.005).round(4).tolist()
elif VARIANT in ('V5', 'V7'):
    THRESHOLDS_SOFT = np.arange(0.01, 2.51, 0.01).round(3).tolist()
else:  # V6
    THRESHOLDS_SOFT = np.arange(0.005, 1.51, 0.005).round(4).tolist()

THRESHOLDS_SUM  = np.arange(0.05, 4.01, 0.05).round(3).tolist()

GUARANTEE_AT_LEAST_ONE = True

LOW_BRIGHTNESS_THRESH  = 80
HIGH_BRIGHTNESS_THRESH = 120
BLUR_VAR_THRESH        = 10

EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'natural']
NUM_CLASSES    = len(EMOTIONS)
CONDITIONS     = ['weak_light', 'strong_light', 'low_res', 'standard']
NUM_CONDITIONS = len(CONDITIONS)
NUM_OUTPUTS    = NUM_CLASSES * NUM_CONDITIONS

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

# ==================== Model ====================
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
                 dropout=0.5, freeze_shallow=True, use_se=True):
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
        self.use_se = use_se
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

# ==================== Aggregation ====================
def aggregate_cale_soft(probs_28, conds, scale=0.5):
    B = probs_28.size(0)
    probs = probs_28.view(B, NUM_CLASSES, NUM_CONDITIONS)
    one_hot = F.one_hot(conds.long(), NUM_CONDITIONS).float().to(probs.device)
    weights = scale + (1.0 - scale) * one_hot
    return (probs * weights.unsqueeze(1)).sum(dim=2)


def aggregate_sum(probs_28, conds=None):
    B = probs_28.size(0)
    probs = probs_28.view(B, NUM_CLASSES, NUM_CONDITIONS)
    return probs.sum(dim=2)

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
    print(f'VARIANT: {VARIANT} | USE_MS={USE_MS} | USE_SE={USE_SE} | DROPOUT={DROPOUT} | FREEZE={FREEZE_SHALLOW}')
    print(f'Inference scale: INFERENCE_SCALE={INFERENCE_SCALE}')
    print(f'Target label density: {TARGET_DENSITY}')
    print(f'Threshold range: [{THRESHOLDS_SOFT[0]}, {THRESHOLDS_SOFT[-1]}] with {len(THRESHOLDS_SOFT)} values')
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

        model = ResNet18CALE(
            num_outputs=NUM_OUTPUTS, reduction=16,
            dropout=DROPOUT, freeze_shallow=FREEZE_SHALLOW, use_se=USE_SE,
        ).to(DEVICE)

        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and 'net' in state:
            state = state['net']
        model.load_state_dict(state)
        model.eval()
        print(f'  Model loaded')

        all_probs_28, all_targets, all_conditions = [], [], []
        with torch.no_grad():
            for inputs, targets_7, cond_idx in test_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                cond_idx_t = cond_idx.to(DEVICE).long()
                logits_28 = model(inputs)
                probs_28 = torch.sigmoid(logits_28.float())
                all_probs_28.append(probs_28.cpu())
                all_targets.append(targets_7.numpy())
                all_conditions.extend(cond_idx.numpy())

        probs_28 = torch.cat(all_probs_28, dim=0).to(DEVICE)
        targets = np.concatenate(all_targets, axis=0)
        conditions = np.array(all_conditions)
        conds_tensor = torch.tensor(conditions, dtype=torch.long, device=DEVICE)

        p_cale_soft = aggregate_cale_soft(probs_28, conds_tensor,
                                           scale=INFERENCE_SCALE).cpu().numpy()
        p_sum = aggregate_sum(probs_28, conds_tensor).cpu().numpy()

        fold_results_soft = scan_thresholds(
            p_cale_soft, targets, THRESHOLDS_SOFT,
            guarantee_one=GUARANTEE_AT_LEAST_ONE,
        )
        fold_results_sum = scan_thresholds(
            p_sum, targets, THRESHOLDS_SUM,
            guarantee_one=GUARANTEE_AT_LEAST_ONE,
        )

        best_density_th = min(
            fold_results_soft,
            key=lambda k: abs(fold_results_soft[k]['avg_pred_labels'] - TARGET_DENSITY),
        )
        best_density_m = fold_results_soft[best_density_th]

        best_f1_th = max(fold_results_soft, key=lambda k: fold_results_soft[k]['macro_f1'])
        best_f1_m = fold_results_soft[best_f1_th]

        best_uar_th = max(fold_results_soft, key=lambda k: fold_results_soft[k]['uar'])
        best_uar_m = fold_results_soft[best_uar_th]

        best_density_th_sum = min(
            fold_results_sum,
            key=lambda k: abs(fold_results_sum[k]['avg_pred_labels'] - TARGET_DENSITY),
        )
        best_density_m_sum = fold_results_sum[best_density_th_sum]

        print(f'  [cale_soft] Density closest to {TARGET_DENSITY}:')
        print(f'    th={best_density_th:.4f} | AvgLabels={best_density_m["avg_pred_labels"]:.4f} | '
              f'UAR={best_density_m["uar"]:.4f} | MacroF1={best_density_m["macro_f1"]:.4f} | '
              f'MicroF1={best_density_m["micro_f1"]:.4f} | HitRate={best_density_m["hit_rate"]:.4f}')
        print(f'  [cale_soft] Best Macro-F1:')
        print(f'    th={best_f1_th:.4f} | AvgLabels={best_f1_m["avg_pred_labels"]:.4f} | '
              f'UAR={best_f1_m["uar"]:.4f} | MacroF1={best_f1_m["macro_f1"]:.4f}')
        print(f'  [cale_soft] Best UAR:')
        print(f'    th={best_uar_th:.4f} | AvgLabels={best_uar_m["avg_pred_labels"]:.4f} | '
              f'UAR={best_uar_m["uar"]:.4f} | MacroF1={best_uar_m["macro_f1"]:.4f}')
        print(f'  [sum] Density closest to {TARGET_DENSITY}:')
        print(f'    th={best_density_th_sum:.4f} | AvgLabels={best_density_m_sum["avg_pred_labels"]:.4f} | '
              f'UAR={best_density_m_sum["uar"]:.4f} | MacroF1={best_density_m_sum["macro_f1"]:.4f}')

        # Sampled threshold printing
        print(f'  [cale_soft] Threshold sampling:')
        print(f'    {"th":>8} {"AvgLabels":>10} {"UAR":>8} {"MacroF1":>9} {"MicroF1":>9} {"HitRate":>9}')
        sample_ths = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
        for th in sample_ths:
            th_key = min(fold_results_soft.keys(), key=lambda k: abs(k - th))
            m = fold_results_soft[th_key]
            print(f'    {th_key:>8.4f} {m["avg_pred_labels"]:>10.4f} {m["uar"]:>8.4f} '
                  f'{m["macro_f1"]:>9.4f} {m["micro_f1"]:>9.4f} {m["hit_rate"]:>9.4f}')

        all_fold_results.append({
            'fold': fold_id,
            'model_path': model_path,
            'soft_threshold_scan': {str(th): fold_results_soft[th] for th in fold_results_soft},
            'sum_threshold_scan': {str(th): fold_results_sum[th] for th in fold_results_sum},
            'soft_best_density_threshold': float(best_density_th),
            'soft_best_density_metrics': best_density_m,
            'soft_best_f1_threshold': float(best_f1_th),
            'soft_best_f1_metrics': best_f1_m,
            'soft_best_uar_threshold': float(best_uar_th),
            'soft_best_uar_metrics': best_uar_m,
            'sum_best_density_threshold': float(best_density_th_sum),
            'sum_best_density_metrics': best_density_m_sum,
        })
        all_fold_best.append({
            'fold': fold_id,
            'soft_density': best_density_m,
            'soft_density_th': float(best_density_th),
            'soft_f1': best_f1_m,
            'soft_f1_th': float(best_f1_th),
            'soft_uar': best_uar_m,
            'soft_uar_th': float(best_uar_th),
            'sum_density': best_density_m_sum,
            'sum_density_th': float(best_density_th_sum),
        })

        del model
        torch.cuda.empty_cache()

    # ==================== 5-fold summary ====================
    print(f'\n{"="*80}')
    print(f'{VARIANT} 5-fold threshold scan summary')
    print(f'{"="*80}')

    if len(all_fold_best) == 0:
        print('No valid results')
        raise SystemExit

    print('\n[cale_soft] Density closest to 2.5:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1'),
                       ('micro_f1', 'MicroF1'),
                       ('hit_rate', 'HitRate')]:
        vals = [r['soft_density'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    ths = [r['soft_density_th'] for r in all_fold_best]
    print(f'  {"Threshold":12s}: {np.mean(ths):.4f} ± {np.std(ths):.4f}')

    print('\n[cale_soft] Best Macro-F1:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1'),
                       ('micro_f1', 'MicroF1'),
                       ('hit_rate', 'HitRate')]:
        vals = [r['soft_f1'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    ths = [r['soft_f1_th'] for r in all_fold_best]
    print(f'  {"Threshold":12s}: {np.mean(ths):.4f} ± {np.std(ths):.4f}')

    print('\n[cale_soft] Best UAR:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1')]:
        vals = [r['soft_uar'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    print('\n[sum] Density closest to 2.5:')
    for key, label in [('avg_pred_labels', 'AvgLabels'),
                       ('uar', 'UAR'),
                       ('macro_f1', 'MacroF1')]:
        vals = [r['sum_density'][key] for r in all_fold_best]
        print(f'  {label:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

    # ==================== Summary comparison table ====================
    print(f'\n{"="*80}')
    print(f'{VARIANT} vs V0 (AvgLabels ≈ 2.5)')
    print(f'{"="*80}')
    print(f'{"Method":<24} {"AvgLabels":>10} {"UAR":>8} {"MacroF1":>9} {"MicroF1":>9} {"HitRate":>9}')
    print(f'{"V0 (threshold)":<24} {"2.5029":>10} {"0.5716":>8} {"0.5607":>9} {"0.5814":>9} {"0.9380":>9}')
    v_uar = np.mean([r['soft_density']['uar'] for r in all_fold_best])
    v_f1 = np.mean([r['soft_density']['macro_f1'] for r in all_fold_best])
    v_mf1 = np.mean([r['soft_density']['micro_f1'] for r in all_fold_best])
    v_hr = np.mean([r['soft_density']['hit_rate'] for r in all_fold_best])
    v_al = np.mean([r['soft_density']['avg_pred_labels'] for r in all_fold_best])
    print(f'{VARIANT+" cale_soft":<24} {v_al:>10.4f} {v_uar:>8.4f} {v_f1:>9.4f} '
          f'{v_mf1:>9.4f} {v_hr:>9.4f}')
    print(f'  UAR improvement:     {v_uar - 0.5716:+.4f}')
    print(f'  Macro-F1 improvement: {v_f1 - 0.5607:+.4f}')

    # Save
    summary = {
        'variant': VARIANT,
        'use_ms': USE_MS, 'use_se': USE_SE,
        'inference_scale': INFERENCE_SCALE,
        'target_density': TARGET_DENSITY,
        'avg_true_labels': float(avg_true_labels),
        'per_fold': all_fold_results,
        'fold_summary': all_fold_best,
        'v0_reference': {
            'avg_pred_labels': 2.5029, 'uar': 0.5716, 'macro_f1': 0.5607,
            'micro_f1': 0.5814, 'hit_rate': 0.9380,
        },
    }
    with open(os.path.join(OUTPUT_DIR, f'{VARIANT.lower()}_multilabel_threshold_scan.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved to {OUTPUT_DIR}')

    # ==================== Plot ====================
    common_ths = sorted(set(THRESHOLDS_SOFT))
    metrics_keys = ['avg_pred_labels', 'uar', 'macro_f1', 'micro_f1', 'hit_rate']

    avg_curves = {k: [] for k in metrics_keys}
    for th in common_ths:
        per_fold = []
        for r in all_fold_results:
            th_str = str(th)
            if th_str in r['soft_threshold_scan']:
                per_fold.append(r['soft_threshold_scan'][th_str])
        if len(per_fold) == 0:
            for k in metrics_keys:
                avg_curves[k].append(np.nan)
            continue
        for k in metrics_keys:
            avg_curves[k].append(np.mean([m[k] for m in per_fold]))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(common_ths, avg_curves['uar'], 'o-', label=f'{VARIANT} UAR', markersize=3)
    plt.plot(common_ths, avg_curves['macro_f1'], 's-', label=f'{VARIANT} Macro-F1', markersize=3)
    plt.plot(common_ths, avg_curves['micro_f1'], '^-', label=f'{VARIANT} Micro-F1', markersize=3)
    plt.axhline(0.5716, color='red', linestyle='--', alpha=0.5, label='V0 UAR @ 2.5')
    plt.axhline(0.5607, color='blue', linestyle='--', alpha=0.5, label='V0 Macro-F1 @ 2.5')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'{VARIANT} Threshold Scan vs V0')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(common_ths, avg_curves['avg_pred_labels'], 'o-',
             label='Avg Predicted Labels', markersize=3)
    plt.axhline(TARGET_DENSITY, color='red', linestyle='--', alpha=0.7,
                label=f'Target = {TARGET_DENSITY}')
    plt.axhline(avg_true_labels, color='gray', linestyle='--', alpha=0.5,
                label=f'True Avg = {avg_true_labels:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Avg Predicted Labels')
    plt.title(f'{VARIANT} Avg Predicted Labels vs Threshold')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{VARIANT.lower()}_threshold_scan.png'), dpi=200)
    plt.close()
    print(f'Curve plot saved to {OUTPUT_DIR}')
    print('\nScan complete.')
