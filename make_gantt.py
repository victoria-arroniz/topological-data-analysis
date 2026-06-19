import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

tasks = [
    ("FPCA + permutation test (Part B)",     0,  10),
    ("Images branch: PH → landscapes",       3,  12),
    ("H2 experiments (m≥3, voids)",         10,  20),
    ("Robustness / parameter sweeps",        18,  27),
    ("Thesis writing (methods + results)",   20,  40),
    ("Revision with supervisors",            38,  46),
]

colors = ['#1A3A7A', '#A01828', '#2E7D32', '#E65100', '#6A1B9A', '#00695C']

milestones = [
    (10, "Jun 25\nchapter"),
    (18, "Jul 10\nvideo"),
    (46, "late Aug\nthesis"),
]

week_ticks  = [0,  8, 16, 24, 32, 40]
week_labels = ['Jun 16', 'Jun 30', 'Jul 14', 'Jul 28', 'Aug 11', 'Aug 25']

fig, ax = plt.subplots(figsize=(13, 5))

n = len(tasks)
for i, (label, start, end) in enumerate(tasks):
    y = n - 1 - i
    ax.barh(y, end - start, left=start, height=0.55, color=colors[i], alpha=0.88)
    mid = (start + end) / 2
    ax.text(mid, y, label, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

for week, label in milestones:
    ax.axvline(week, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.text(week + 0.3, n - 0.15, label, fontsize=7.5, color='gray', va='top')

ax.set_xticks(week_ticks)
ax.set_xticklabels(week_labels, fontsize=10)
ax.set_yticks([])
ax.set_xlim(-1, 49)
ax.set_ylim(-0.6, n - 0.3)
ax.set_xlabel('Timeline', fontsize=11)
ax.set_title('Research project timeline', fontsize=13)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/gantt_timeline.png', dpi=200, bbox_inches='tight')
plt.close()
print('figures/gantt_timeline.png saved')
