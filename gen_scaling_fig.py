"""Generate scaling curve figure for APIN paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Complete scaling curve data
data_sizes = [30, 60, 100, 150, 200]
concat = [22.98, 23.47, 23.56, 23.89, 23.74]
film   = [23.35, 23.77, 23.93, 24.09, 24.77]
gated  = [23.31, 23.78, 23.31, 23.81, 24.31]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 15,
    'legend.fontsize': 12, 'figure.dpi': 200, 'savefig.dpi': 300,
})

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(data_sizes))
w = 0.25

bars1 = ax.bar(x - w, concat, w, color='#4472C4', edgecolor='white', label='Concat')
bars2 = ax.bar(x, film, w, color='#ED7D31', edgecolor='white', label='FiLM')
bars3 = ax.bar(x + w, gated, w, color='#A5A5A5', edgecolor='white', label='Gated')

# Value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.08,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

# FiLM-Concat gap annotations
gaps = [f - c for f, c in zip(film, concat)]
for i, gap in enumerate(gaps):
    mid = x[i]
    top = max(concat[i], film[i], gated[i])
    ax.annotate(f'$\Delta$=+{gap:.2f}',
                xy=(mid, top + 0.7), fontsize=9, ha='center',
                color='#C00000', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n(100 ep)' for s in data_sizes[:-1]] + ['200\n(200 ep)'])
ax.set_ylabel('PSNR (dB)')
ax.set_xlabel('Training Pairs')
ax.set_title('Scaling Behavior of Fusion Strategies on SD7K')
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(22.5, 25.8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = '/mnt/ShaDocFormer-main/paper/apin-submission/fig_scaling_curve.pdf'
plt.savefig(output_path, bbox_inches='tight', facecolor='white')
print(f'Saved to {output_path}')
