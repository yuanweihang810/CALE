"""
Fig. 9 (v3): Architecture dependence. Legend below to avoid covering bars.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

backbones = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'MobileNetV2', 'EfficientNet-B0']

avg_05  = [5.86, 3.08, 0.96, 2.71, 6.79]
avg_00  = [5.98, 1.72, 0.24, 1.68, np.nan]
test_05 = [0.5200, 0.5238, 0.3359, 0.4785, 0.5206]
test_00 = [0.5139, 0.3686, 0.1051, 0.3819, np.nan]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

# ---- (a) AvgLabels ----
ax = axes[0]
x = np.arange(len(backbones))
w = 0.38
ax.bar(x - w/2, avg_05, w, label='SOFT=0.5', color='#3B75AF',
       edgecolor='black', linewidth=0.6)
ax.bar(x + w/2, avg_00, w, label='SOFT=0.0', color='#D55E00',
       edgecolor='black', linewidth=0.6)
ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=0.9, label='Target (2.5)')
ax.set_xticks(x)
ax.set_xticklabels(backbones, rotation=15, ha='right')
ax.set_ylabel('AvgLabels @ th=0.5')
ax.set_title('(a) Predicted label density')
ax.set_ylim(0, 8.5)
ax.grid(axis='y', alpha=0.25, linewidth=0.5)

for i, (a, b) in enumerate(zip(avg_05, avg_00)):
    ax.text(i - w/2, a + 0.15, f'{a:.2f}', ha='center', va='bottom', fontsize=7.5)
    if not np.isnan(b):
        ax.text(i + w/2, b + 0.15, f'{b:.2f}', ha='center', va='bottom', fontsize=7.5)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
          ncol=3, frameon=False, fontsize=8,
          handlelength=1.5, columnspacing=1.0)

# ---- (b) Test Macro-F1 ----
ax = axes[1]
ax.bar(x - w/2, test_05, w, label='SOFT=0.5', color='#3B75AF',
       edgecolor='black', linewidth=0.6)
ax.bar(x + w/2, test_00, w, label='SOFT=0.0', color='#D55E00',
       edgecolor='black', linewidth=0.6)
ax.axhline(y=0.3458, color='green', linestyle='--', linewidth=1.1,
           label='7-dim sigmoid (0.3458)')
ax.set_xticks(x)
ax.set_xticklabels(backbones, rotation=15, ha='right')
ax.set_ylabel('Test Macro-F1')
ax.set_title('(b) Multi-label test performance')
ax.set_ylim(0, 0.68)
ax.grid(axis='y', alpha=0.25, linewidth=0.5)

for i, (a, b) in enumerate(zip(test_05, test_00)):
    ax.text(i - w/2, a + 0.012, f'{a:.4f}', ha='center', va='bottom', fontsize=7)
    if not np.isnan(b):
        ax.text(i + w/2, b + 0.012, f'{b:.4f}', ha='center', va='bottom', fontsize=7)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
          ncol=3, frameon=False, fontsize=8,
          handlelength=1.5, columnspacing=1.0)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('../figures/fig9_architecture.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved fig9_architecture.pdf (v3, legend below)')