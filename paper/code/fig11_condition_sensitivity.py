"""
Fig. 11 (v3): Condition classifier threshold sensitivity.
Legend placed below each panel to avoid covering data.
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

scales = [0.8, 0.9, 1.0, 1.1, 1.2]

train = {
    'Weak light':     [18.4, 21.7, 24.9, 28.0, 32.1],
    'Strong light':   [48.6, 39.4, 31.6, 25.6, 19.9],
    'Low resolution': [16.6, 18.2, 19.5, 20.9, 22.1],
    'Standard':       [16.4, 20.8, 24.0, 25.5, 26.0],
}
test = {
    'Weak light':     [18.3, 22.5, 27.4, 30.3, 35.1],
    'Strong light':   [45.0, 36.4, 29.0, 23.0, 17.5],
    'Low resolution': [16.4, 18.1, 19.6, 20.9, 22.5],
    'Standard':       [20.2, 23.0, 24.0, 25.8, 24.9],
}

colors = {
    'Weak light':     '#0072B2',
    'Strong light':   '#D55E00',
    'Low resolution': '#009E73',
    'Standard':       '#CC79A7',
}
markers = {
    'Weak light':     'o',
    'Strong light':   's',
    'Low resolution': '^',
    'Standard':       'D',
}

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

for ax, data, title in [
    (axes[0], train, '(a) Training set'),
    (axes[1], test,  '(b) Test set'),
]:
    for name, vals in data.items():
        ax.plot(scales, vals, marker=markers[name],
                color=colors[name], linewidth=1.4, markersize=5,
                label=name)
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1.0)
    ax.set_xlabel('Threshold scale')
    ax.set_ylabel('Class proportion (%)')
    ax.set_title(title)
    ax.set_xticks(scales)
    ax.set_xticklabels([f'{s:.1f}x' for s in scales])
    ax.set_ylim(10, 55)
    ax.grid(alpha=0.3, linewidth=0.5)
    # 图例放到 panel 下方，4 列排布
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
              ncol=4, frameon=False, fontsize=8,
              handlelength=1.5, columnspacing=1.0)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('../figures/fig11_condition_sensitivity.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Saved fig11_condition_sensitivity.pdf (v3, legend below)')