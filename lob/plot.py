"""Plot storage and reconstruction work from the checkpoint sweep."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('results')
    sweep = json.loads((root / 'checkpoint_sweep.json').read_text())['per_symbol']
    strides = sorted({row['checkpoint_rows'] for row in sweep})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#c9d4dd', 'svg.fonttype': 'none',
                         'text.color': '#172e42', 'axes.labelcolor': '#172e42'})
    fig, (work, space) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.5, 1]})
    fig.patch.set_facecolor('#f8fafc')
    for ax in (work, space):
        ax.set_facecolor('#f8fafc')
        ax.grid(axis='y', color='#e2e8ed', linewidth=.7)
        ax.set_axisbelow(True)
    for symbol, color in zip(['AAPL', 'AMZN', 'MSFT'], ['#177f79', '#526e86', '#bb8544']):
        group = sorted([row for row in sweep if row['symbol'] == symbol], key=lambda r: r['checkpoint_rows'])
        work.plot(range(len(strides)), [r['median_rows_decoded_per_lookup'] for r in group],
                  marker='o', label=symbol, color=color)
    work.set_xticks(range(len(strides)), [f'{value:,}' for value in strides])
    work.set(xlabel='Rows per checkpoint', ylabel='Median rows decoded', title='Reconstruction work per lookup')
    work.legend(frameon=False)
    totals = [sum(r['archive_bytes'] for r in sweep if r['checkpoint_rows'] == stride) / 1e6 for stride in strides]
    rects = space.bar([f'{stride:,}' for stride in strides], totals, color=['#177f79', '#526e86', '#a4b6c4'], width=.6)
    space.bar_label(rects, fmt='%.2f', padding=5)
    space.set(xlabel='Rows per checkpoint', ylabel='MB', title='Total mapped storage', ylim=(0, 49))
    fig.suptitle('Checkpoint frequency trades space for less replay work', x=.08, ha='left', fontsize=15, fontweight='bold')
    fig.text(.08, .03, '3,000 seeded timestamp lookups · AAPL, AMZN and MSFT · 21 June 2012', color='#526779')
    fig.subplots_adjust(left=.08, right=.98, top=.81, bottom=.21, wspace=.3)
    fig.savefig(root / 'checkpoints.svg', metadata={'Date': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
