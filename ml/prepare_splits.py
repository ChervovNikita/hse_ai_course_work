"""Filters the full WB and Amazon datasets and splits them 80/20 by product.

  python ml/prepare_splits.py

Writes data/splits/{wb,amazon}_{full,train,test}.csv.gz. All files share the
columns product_id, full_text, rating; the rest are platform specific.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

OUT = Path('data/splits')
SEED = 0
TEST_SIZE = 0.2


def load_wb():
    """WB reviews without label noise flagged by WB itself (sentimentCollision)."""
    df = pd.read_csv('data/wb_reviews.csv.gz', keep_default_na=False)
    df = df[~df.excluded_from_rating.str.contains('sentimentCollision')]
    df = df.rename(columns={'imt_id': 'product_id'})
    # binary flags of filled form fields
    for field in ['text', 'pros', 'cons']:
        df[f'has_{field}'] = (df[field] != '').astype(int)
    return df.reset_index(drop=True)


def load_amazon():
    """Amazon reviews; empty titles stay empty strings instead of NaN."""
    return pd.read_csv('data/amazon_reviews.csv.gz', keep_default_na=False)


def split(df):
    """80/20 train/test split with no product shared between the parts."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df.product_id))
    return df.iloc[train_idx], df.iloc[test_idx]


def describe(name, train, test):
    """Prints short stats of one platform's split."""
    words = pd.concat([train, test]).full_text.str.split().str.len()
    print(f'\n=== {name}')
    print(f'reviews: {len(train) + len(test):,} (train {len(train):,} / test {len(test):,})')
    print(f'products: train {train.product_id.nunique():,} / test {test.product_id.nunique():,}, '
          f'shared: {len(set(train.product_id) & set(test.product_id))}')
    print(f'words per review: median {words.median():.0f}, p95 {words.quantile(0.95):.0f}, '
          f'p99 {words.quantile(0.99):.0f}, <3 words {(words < 3).mean():.1%}')
    shares = pd.DataFrame({'train': train.rating.value_counts(normalize=True),
                           'test': test.rating.value_counts(normalize=True)}).sort_index()
    print('rating share:')
    print(shares.T.round(3).to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [('wb', load_wb()), ('amazon', load_amazon())]:
        # filtered clone of the full dataset; the original files stay untouched
        df.to_csv(OUT / f'{name}_full.csv.gz', index=False)
        train, test = split(df)
        describe(name, train, test)
        train.to_csv(OUT / f'{name}_train.csv.gz', index=False)
        test.to_csv(OUT / f'{name}_test.csv.gz', index=False)
    print('\nfiles:', *sorted(str(p) for p in OUT.glob('*.csv.gz')), sep='\n  ')


if __name__ == '__main__':
    main()
