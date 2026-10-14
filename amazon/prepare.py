"""Builds the Amazon counterpart of the WB dataset from Amazon Reviews 2023.

Source: McAuley Lab, https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
Sampling mirrors parser/wb_parser.py: text >= 10 chars, products with >= 3 such
reviews, at most 100 random reviews per product, equal quota per category.

  python amazon/prepare.py --raw data/amazon_raw --out data

Writes:
  data/amazon_reviews.csv.gz        - all sampled reviews, label = rating
  data/amazon_reviews_balanced.csv  - every rating undersampled to the rarest one
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = ('https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/'
       'raw/review_categories/{}.jsonl')
CATEGORIES = ['Amazon_Fashion', 'All_Beauty', 'Handmade_Products', 'Health_and_Personal_Care', 'Appliances']
# amazon fills empty titles with the star count ("Five Stars"), which leaks the label
STAR_TITLE = re.compile(r'^\s*(one|two|three|four|five)\s+stars?\s*$', re.IGNORECASE)


def download(raw):
    """Downloads category files that are not on disk yet."""
    raw.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        path = raw / f'{category}.jsonl'
        if not path.exists():
            print(f'downloading {category}')
            urllib.request.urlretrieve(URL.format(category), path)


def read_category(path):
    """Reads one category jsonl into a dataframe with the needed fields."""
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append((r['parent_asin'], r['rating'], r['title'], r['text'], r['timestamp'],
                         r['helpful_vote'], r['verified_purchase']))
    df = pd.DataFrame(rows, columns=['product_id', 'rating', 'title', 'text', 'timestamp',
                                     'helpful_vote', 'verified_purchase'])
    df['category'] = path.stem
    return df


def sample_category(df, quota, max_per_product, min_reviews, min_len, rng):
    """Samples random products of one category until the review quota is reached."""
    df = df.assign(star_title=df.title.str.match(STAR_TITLE))
    df['title'] = df.title.where(~df.star_title, '')
    df = df[(df.title + df.text).str.len() >= min_len]
    sizes = df.groupby('product_id').size()
    products = sizes[sizes >= min_reviews].index.to_numpy()
    rng.shuffle(products)
    capped = np.minimum(sizes.loc[products].to_numpy(), max_per_product)
    keep = set(products[:np.searchsorted(np.cumsum(capped), quota) + 1])
    df = df[df.product_id.isin(keep)]
    seed = int(rng.integers(1 << 31))
    return df.groupby('product_id').sample(frac=1, random_state=seed).groupby('product_id').head(max_per_product)


def build(raw, quota, max_per_product, min_reviews, min_len, seed):
    """Reads all categories and returns the sampled review table."""
    rng = np.random.default_rng(seed)
    parts = []
    for category in CATEGORIES:
        df = read_category(raw / f'{category}.jsonl')
        part = sample_category(df, quota, max_per_product, min_reviews, min_len, rng)
        print(f'{category}: {len(df):,} reviews -> {len(part):,} sampled, '
              f'{part.product_id.nunique():,} products, star titles removed: {part.star_title.sum():,}')
        parts.append(part)
    reviews = pd.concat(parts, ignore_index=True)
    reviews['rating'] = reviews.rating.astype(int)
    reviews['created_date'] = pd.to_datetime(reviews.timestamp, unit='ms', utc=True)
    reviews['full_text'] = (reviews.title + '\n' + reviews.text).str.strip()
    return reviews.drop(columns=['timestamp']).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw', default='data/amazon_raw')
    p.add_argument('--out', default='data')
    p.add_argument('--quota', type=int, default=60_000, help='reviews per category')
    p.add_argument('--max-per-product', type=int, default=100)
    p.add_argument('--min-reviews', type=int, default=3)
    p.add_argument('--min-len', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    download(raw)
    reviews = build(raw, args.quota, args.max_per_product, args.min_reviews, args.min_len, args.seed)
    reviews.to_csv(out / 'amazon_reviews.csv.gz', index=False)

    # drop random reviews from the big bins until every rating has as many as the rarest
    n_min = reviews.rating.value_counts().min()
    balanced = reviews.groupby('rating').sample(n=n_min, random_state=0).sample(frac=1, random_state=0)
    balanced.to_csv(out / 'amazon_reviews_balanced.csv', index=False)

    print(f'\nreviews: {len(reviews):,}, products: {reviews.product_id.nunique():,}, balanced: {len(balanced):,}')
    print(reviews.rating.value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == '__main__':
    main()
