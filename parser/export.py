"""Builds the final dataset from raw parser output and prints sanity checks.

  python parser/export.py --raw data/raw --out data

Writes:
  data/wb_reviews.csv           - all reviews (one row per review, label = rating)
  data/wb_reviews.csv.gz        - same, compressed (the plain csv exceeds GitHub's 100MB limit)
  data/wb_reviews_balanced.csv  - same, every rating undersampled to the rarest one
  data/wb_products.csv          - product metadata (description, avg rating, ...)
  data/wb_reviews_sample.csv    - 10 random reviews per rating, for eyeballing
"""

import argparse
from pathlib import Path

import pandas as pd


def build(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = pd.read_json(raw / "reviews.jsonl", lines=True)
    products = pd.read_json(raw / "products.jsonl", lines=True)

    reviews = reviews.drop_duplicates("review_id")
    products = products.drop_duplicates("imt_id")

    # single text field for modelling; keep the original parts too
    def full_text(r):
        parts = [r.text]
        if r.pros:
            parts.append(f"Достоинства: {r.pros}")
        if r.cons:
            parts.append(f"Недостатки: {r.cons}")
        return "\n".join(p for p in parts if p)

    reviews["full_text"] = reviews.apply(full_text, axis=1)
    reviews = reviews.merge(
        products[["imt_id", "name", "brand"]].rename(columns={"name": "product_name"}),
        on="imt_id",
        how="left",
    )
    products["options"] = products["options"].apply(
        lambda o: "; ".join(f"{k}: {v}" for k, v in (o or {}).items())
    )
    return reviews, products


def report(reviews: pd.DataFrame, products: pd.DataFrame):
    print(f"reviews:  {len(reviews)}")
    print(f"products: {len(products)}  (categories: {products.subject_root.nunique()} root, "
          f"{products.subject.nunique()} leaf)")
    print("\nrating distribution:")
    vc = reviews.rating.value_counts().sort_index()
    print(pd.DataFrame({"n": vc, "share": (vc / vc.sum()).round(3)}).to_string())
    lens = reviews.full_text.str.len()
    print(f"\ntext length, chars: median {lens.median():.0f}, mean {lens.mean():.0f}, "
          f"p95 {lens.quantile(.95):.0f}")
    print(f"empty full_text: {(lens == 0).sum()}")
    print(f"reviews per product: median {reviews.groupby('imt_id').size().median():.0f}, "
          f"max {reviews.groupby('imt_id').size().max()}")
    print(f"date range: {reviews.created_date.min()} .. {reviews.created_date.max()}")
    print("\ntop root categories:")
    print(reviews.subject_root.value_counts().head(10).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data")
    p.add_argument("--per-rating", type=int, default=10)
    args = p.parse_args()
    out = Path(args.out)

    reviews, products = build(Path(args.raw))
    reviews.to_csv(out / "wb_reviews.csv", index=False)
    reviews.to_csv(out / "wb_reviews.csv.gz", index=False)
    products.to_csv(out / "wb_products.csv", index=False)

    # drop random reviews from the big bins until every rating has as many as the rarest
    n_min = reviews.rating.value_counts().min()
    balanced = reviews.groupby("rating").sample(n=n_min, random_state=0).sample(frac=1, random_state=0)
    balanced.to_csv(out / "wb_reviews_balanced.csv", index=False)
    print(f"balanced: {len(balanced)} reviews ({n_min} per rating)")

    sample = (
        reviews.groupby("rating")
        .sample(n=args.per_rating, random_state=0)
        [["rating", "full_text", "product_name", "subject", "subject_root", "created_date"]]
    )
    sample.to_csv(out / "wb_reviews_sample.csv", index=False)

    report(reviews, products)


if __name__ == "__main__":
    main()
