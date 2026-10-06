"""
Fig. 10 (v3): Fair multi-label comparison.
Legend compacted at top-left.
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

methods = [
    '7-dim sigmoid\n(th=0.5)',
    '7-dim sigmoid\n(argmax)',
    'CALE\nResNet-18',
    'CALE\nResNet-34',
    'CALE\nResNet-50',
    'CALE\nMobileNetV2',
    'CALE\nEfficientNet-B0',
]
test_f1 = [0.3458, 0.3156, 0.2873, 0.3525, 0.3359, 0.4785, 0.5206]
colors  = ['#009E73']*2 + ['#3B75AF']*5
avg_labels = [1.13, 1.00, 1.00, 1.00, 0.96, 2.71, 6.79]

fig, ax = plt.subplots(figsize=(7.16, 3.6))

x = np.arange(len(methods))
bars = ax.bar(x, test_f1, color=colors, edgecolor='black', linewidth=0.6, width=0.62)

ax.axhline(y=0.3458, color='green', linestyle='--', linewidth=1.0,
           label='7-dim sigmoid (th=0.5) = 0.3458')

for i, (bar, avg) in enumerate(zip(bars, avg_labels)):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.010,
            f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    ax.text(bar.get_x() + bar.get_width()/2, h/2,
            f'Avg={avg:.2f}', ha='center', va='center', fontsize=7.5,
            rotation=90, color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha='right')
ax.set_ylabel('Test Macro-F1')
ax.set_ylim(0, 0.65)
ax.set_title('Fair multi-label comparison (10\\% data)')
ax.grid(axis='y', alpha=0.25, linewidth=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0),
          frameon=True, framealpha=0.95, fontsize=8)

plt.tight_layout()
plt.savefig('../figures/fig10_fair_baseline.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved fig10_fair_baseline.pdf (v3)')