import json
import numpy as np
from scipy import stats

BASE = '/root/autodl-tmp/IEEE'

FILES = {
    'V0': f'{BASE}/原始论文/v0_reproduce/v0_summary.json',
    'V2': f'{BASE}/cale_v2_singlelabel/v2_summary.json',
    'V3': f'{BASE}/baseline_v3_singlelabel/v3_summary.json',
    'V4': f'{BASE}/baseline_v4_singlelabel/v4_summary.json',
    'V7': f'{BASE}/cale_v7_singlelabel/v7_summary.json',
}

def extract(path, field):
    with open(path) as f:
        data = json.load(f)
    return np.array([r[field] for r in data['fold_results']])

PAIRS = [
    ('V2 vs V0 (LE vs Baseline)',    'V2', 'V0'),
    ('V3 vs V0 (SE vs Baseline)',    'V3', 'V0'),
    ('V4 vs V0 (MS+SE vs Baseline)', 'V4', 'V0'),
    ('V7 vs V0 (Full vs Baseline)',  'V7', 'V0'),
]

print('=' * 70)
print('Paired t-test on 5-fold validation metrics')
print('=' * 70)

# 用 val_uar 统一配对（V0 只有 val_uar）
for field in ['val_uar', 'val_macro_f1']:
    print(f'\n### Field = {field} ###')
    cache = {}
    for name, path in FILES.items():
        try:
            cache[name] = extract(path, field)
        except KeyError:
            pass

    for label, a_key, b_key in PAIRS:
        if a_key not in cache or b_key not in cache:
            print(f'\n{label}: SKIPPED (missing {field})')
            continue
        a, b = cache[a_key], cache[b_key]
        t, p = stats.ttest_rel(a, b)
        diff = a - b
        print(f'\n{label}')
        print(f'  A mean = {a.mean():.4f}, B mean = {b.mean():.4f}')
        print(f'  Mean diff = {diff.mean():+.4f} (std of diff = {diff.std(ddof=1):.4f})')
        print(f'  t = {t:.3f}, p = {p:.4f}')
        if p < 0.05:
            print(f'  -> Significant at 0.05')
        elif p < 0.10:
            print(f'  -> Marginally significant at 0.10')
        else:
            print(f'  -> Not significant')
