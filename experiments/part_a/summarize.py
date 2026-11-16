"""Builds results tables (markdown) from results.csv: top runs and every group, for each dataset.

  python experiments/part_a/summarize.py > experiments/part_a/RESULTS.md
"""

from common import load_results
from experiments import GROUPS

COLS = ['exp_id', 'note', 'macro_f1', 'mae', 'accuracy', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'f1_5', 'n_features',
        'fit_sec']
BASELINE = 'f02_word12'


def table(df):
    """Markdown table of the main columns, best macro-F1 first."""
    return df.sort_values('macro_f1', ascending=False)[COLS].round(4).to_markdown(index=False)


def main():
    res = load_results()
    print('# Part A: linear models on sparse features — validation results\n')
    print('Fit on 80% of train (by product), scored on the other 20%. Test is not used. '
          f'Baseline: `{BASELINE}` (TF-IDF words+bigrams, logreg C=2, balanced weights).\n')
    for dataset in ['wb', 'amazon']:
        d = res[res.dataset == dataset]
        if d.empty:
            continue
        base = d[d.exp_id == BASELINE].macro_f1.iloc[0]
        print(f'## {dataset.upper()}: {len(d)} runs, baseline macro-F1 {base:.4f}, best {d.macro_f1.max():.4f}\n')
        print('### Top 10\n')
        print(table(d.nlargest(10, 'macro_f1')), '\n')
        for group in GROUPS:
            g = d[d.group == group]
            if len(g):
                print(f'### {group}\n')
                print(table(g), '\n')


if __name__ == '__main__':
    main()
