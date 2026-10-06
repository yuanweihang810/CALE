import os, json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

CUSTOM_DIR = '/root/autodl-tmp/IEEE/data_aligned'
TEST_DIR   = '/root/autodl-tmp/IEEE/test_aligned/test'
LABEL_FILE = '/root/autodl-tmp/IEEE/label.txt'

LOW_BRIGHTNESS_THRESH  = 80
HIGH_BRIGHTNESS_THRESH = 120
BLUR_VAR_THRESH        = 10

def classify_condition(image_path):
    try:
        with Image.open(image_path) as img:
            arr = np.array(img.convert('L'), dtype=np.float64)
        brightness = arr.mean()
        variance = cv2.Laplacian(arr, cv2.CV_64F).var()
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

def count_conditions(root_dir):
    counts = [0, 0, 0, 0]
    total = 0
    for folder in Path(root_dir).iterdir():
        if not folder.is_dir():
            continue
        for f in folder.glob('*.*'):
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                counts[classify_condition(str(f))] += 1
                total += 1
    return counts, total

def count_test_conditions(img_dir, label_path):
    counts = [0, 0, 0, 0]
    total = 0
    with open(label_path) as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        img_no = parts[0]
        for ext in ['.png', '.jpg', '.jpeg']:
            p = os.path.join(img_dir, f'{img_no}{ext}')
            if os.path.exists(p):
                counts[classify_condition(p)] += 1
                total += 1
                break
    return counts, total

train_counts, train_total = count_conditions(CUSTOM_DIR)
test_counts, test_total = count_test_conditions(TEST_DIR, LABEL_FILE)

train_pct = [c / train_total * 100 for c in train_counts]
test_pct  = [c / test_total * 100 for c in test_counts]

print(f'Train (total={train_total}):')
print(f'  weak_light   = {train_pct[0]:.1f}%  ({train_counts[0]})')
print(f'  strong_light = {train_pct[1]:.1f}%  ({train_counts[1]})')
print(f'  low_res      = {train_pct[2]:.1f}%  ({train_counts[2]})')
print(f'  standard     = {train_pct[3]:.1f}%  ({train_counts[3]})')
print()
print(f'Test (total={test_total}):')
print(f'  weak_light   = {test_pct[0]:.1f}%  ({test_counts[0]})')
print(f'  strong_light = {test_pct[1]:.1f}%  ({test_counts[1]})')
print(f'  low_res      = {test_pct[2]:.1f}%  ({test_counts[2]})')
print(f'  standard     = {test_pct[3]:.1f}%  ({test_counts[3]})')

with open('/root/autodl-tmp/IEEE/condition_stats.json', 'w') as f:
    json.dump({
        'train_pct': train_pct, 'test_pct': test_pct,
        'train_counts': train_counts, 'test_counts': test_counts,
        'train_total': train_total, 'test_total': test_total,
    }, f, indent=2)
print('\nSaved to condition_stats.json')
