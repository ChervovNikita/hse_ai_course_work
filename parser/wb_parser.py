"""Wildberries review parser.

Collects product reviews with star ratings (1-5) plus product metadata.

Pipeline:
  1. Sample random product ids (nm_id) from a range. Search / catalog APIs
     (*.wb.ru) are behind antibot, so products are discovered through the
     public static CDN (basket-XX.wbbasket.ru), which serves card.json for
     every product. This also gives an unbiased sample across all categories.
  2. card.json -> imt_id (product group id), name, category, brand, description.
  3. feedbacks{1,2}.wb.ru/feedbacks/v1/{imt_id} -> reviews with ratings.

Output (resumable, appends to JSONL):
  <out>/products.jsonl  - one line per product group
  <out>/reviews.jsonl   - one line per review

Usage:
  python parser/wb_parser.py --target 20000 --out data/raw
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)
MAX_BASKET = 60
FEEDBACK_HOSTS = ["feedbacks1.wb.ru", "feedbacks2.wb.ru"]
# vol -> basket observed by probing the CDN (Oct 2026); used only as a starting
# guess, the real host is always verified and cached at runtime
BASKET_HINTS = {
    100: 1, 1000: 5, 1696: 12, 2000: 13, 3000: 18, 4000: 23, 4500: 25, 5000: 27,
    5500: 28, 6000: 30, 6500: 32, 7000: 33, 7500: 35, 8000: 37, 9000: 39,
}


def card_url(basket: int, nm: int) -> str:
    vol, part = nm // 100000, nm // 1000
    return f"https://basket-{basket:02d}.wbbasket.ru/vol{vol}/part{part}/{nm}/info/ru/card.json"


class BasketResolver:
    """Finds which CDN host (basket-XX) serves a given vol (= nm_id // 100000).

    Hosts are assigned to contiguous, increasing vol ranges, so the host of the
    nearest known vol is a good first guess.
    """

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.known: dict[int, int] = {}
        self.missing: set[int] = set()
        self.locks: dict[int, asyncio.Lock] = {}

    def _candidates(self, vol: int) -> list[int]:
        lower = [h for v, h in {**BASKET_HINTS, **self.known}.items() if v <= vol]
        start = max(lower) if lower else 1
        up = list(range(start, MAX_BASKET + 1))
        down = list(range(start - 1, 0, -1))
        return up + down

    async def _exists(self, basket: int, nm: int) -> bool:
        try:
            async with self.session.get(card_url(basket, nm)) as r:
                return r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def resolve(self, vol: int) -> int | None:
        if vol in self.known:
            return self.known[vol]
        if vol in self.missing:
            return None
        lock = self.locks.setdefault(vol, asyncio.Lock())
        async with lock:
            if vol in self.known or vol in self.missing:
                return self.known.get(vol)
            probes = [vol * 100000 + random.randrange(100000) for _ in range(8)]
            for basket in self._candidates(vol):
                hits = await asyncio.gather(*(self._exists(basket, nm) for nm in probes))
                if any(hits):
                    self.known[vol] = basket
                    return basket
            self.missing.add(vol)
            return None


class WBParser:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.products_path = self.out / "products.jsonl"
        self.reviews_path = self.out / "reviews.jsonl"
        self.seen_imt: set[int] = set()
        self.n_reviews = 0
        self.n_products = 0
        self.n_cards = 0
        self.started = time.time()
        self._load_state()

    def _load_state(self):
        if self.products_path.exists():
            for line in self.products_path.open():
                self.seen_imt.add(json.loads(line)["imt_id"])
                self.n_products += 1
        if self.reviews_path.exists():
            self.n_reviews = sum(1 for _ in self.reviews_path.open())
        if self.n_reviews:
            print(f"resuming: {self.n_products} products, {self.n_reviews} reviews")

    async def fetch_json(self, url: str, retries: int = 3):
        for attempt in range(retries):
            try:
                async with self.session.get(url) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    if r.status == 404:
                        return None
                    if r.status in (429, 498, 503):
                        await asyncio.sleep(2 ** attempt * 3)
                        continue
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
                await asyncio.sleep(2 ** attempt)
        return None

    async def fetch_card(self, nm: int):
        basket = await self.resolver.resolve(nm // 100000)
        if basket is None:
            return None
        return await self.fetch_json(card_url(basket, nm))

    async def fetch_feedbacks(self, imt: int):
        host = random.choice(FEEDBACK_HOSTS)
        data = await self.fetch_json(f"https://{host}/feedbacks/v1/{imt}")
        if not data or not data.get("feedbacks"):
            return data, []
        return data, data["feedbacks"]

    def _product_record(self, card: dict, fb_data: dict) -> dict:
        return {
            "imt_id": card.get("imt_id"),
            "nm_id": card.get("nm_id"),
            "name": card.get("imt_name"),
            "subject": card.get("subj_name"),
            "subject_root": card.get("subj_root_name"),
            "brand": (card.get("selling") or {}).get("brand_name"),
            "description": card.get("description"),
            "options": {o.get("name"): o.get("value") for o in card.get("options") or []},
            "avg_rating": float(fb_data["valuation"]) if fb_data.get("valuation") else None,
            "feedback_count": fb_data.get("feedbackCount"),
            "feedback_count_with_text": fb_data.get("feedbackCountWithText"),
        }

    @staticmethod
    def _review_record(f: dict, product: dict) -> dict:
        return {
            "review_id": f.get("id"),
            "imt_id": product["imt_id"],
            "nm_id": f.get("nmId"),
            "rating": f.get("productValuation"),
            "text": (f.get("text") or "").strip(),
            "pros": (f.get("pros") or "").strip(),
            "cons": (f.get("cons") or "").strip(),
            "created_date": f.get("createdDate"),
            "color": f.get("color"),
            "size": f.get("size"),
            "votes_plus": (f.get("votes") or {}).get("pluses"),
            "votes_minus": (f.get("votes") or {}).get("minuses"),
            "has_answer": bool(f.get("answer")),
            "excluded_from_rating": f.get("excludedFromRating"),
            "subject": product["subject"],
            "subject_root": product["subject_root"],
        }

    async def process(self, nm: int):
        card = await self.fetch_card(nm)
        if not card:
            return
        self.n_cards += 1
        imt = card.get("imt_id")
        if imt is None or imt in self.seen_imt:
            return
        self.seen_imt.add(imt)

        fb_data, feedbacks = await self.fetch_feedbacks(imt)
        if not fb_data:
            return
        with_text = [
            f for f in feedbacks
            if f.get("productValuation") in (1, 2, 3, 4, 5)
            and len((f.get("text") or "") + (f.get("pros") or "") + (f.get("cons") or "")) >= self.args.min_len
        ]
        if len(with_text) < self.args.min_reviews:
            return
        # cap per product so a few very popular products don't dominate the data
        if len(with_text) > self.args.max_per_product:
            with_text = random.sample(with_text, self.args.max_per_product)

        product = self._product_record(card, fb_data)
        with self.products_path.open("a") as fp:
            fp.write(json.dumps(product, ensure_ascii=False) + "\n")
        with self.reviews_path.open("a") as fr:
            for f in with_text:
                fr.write(json.dumps(self._review_record(f, product), ensure_ascii=False) + "\n")
        self.n_products += 1
        self.n_reviews += len(with_text)

    def report(self):
        el = time.time() - self.started
        print(
            f"[{el:6.0f}s] cards {self.n_cards:6d} | products w/ reviews {self.n_products:5d} "
            f"| reviews {self.n_reviews:6d}/{self.args.target}",
            flush=True,
        )

    async def worker(self, queue: asyncio.Queue):
        while self.n_reviews < self.args.target:
            nm = await queue.get()
            try:
                await self.process(nm)
            finally:
                queue.task_done()
            await asyncio.sleep(self.args.delay)

    async def producer(self, queue: asyncio.Queue):
        lo, hi = self.args.nm_min, self.args.nm_max
        while self.n_reviews < self.args.target:
            await queue.put(random.randrange(lo, hi))

    async def reporter(self):
        while self.n_reviews < self.args.target:
            await asyncio.sleep(15)
            self.report()

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=20)
        conn = aiohttp.TCPConnector(limit=self.args.concurrency * 2)
        headers = {"User-Agent": UA, "Accept": "application/json"}
        async with aiohttp.ClientSession(timeout=timeout, connector=conn, headers=headers) as session:
            self.session = session
            self.resolver = BasketResolver(session)
            queue: asyncio.Queue = asyncio.Queue(maxsize=self.args.concurrency * 4)
            tasks = [asyncio.create_task(self.producer(queue)), asyncio.create_task(self.reporter())]
            workers = [asyncio.create_task(self.worker(queue)) for _ in range(self.args.concurrency)]
            await asyncio.gather(*workers)
            for t in tasks:
                t.cancel()
        self.report()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=20000, help="stop after this many reviews")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--nm-min", type=int, default=10_000_000)
    p.add_argument("--nm-max", type=int, default=900_000_000)
    p.add_argument("--max-per-product", type=int, default=100)
    p.add_argument("--min-reviews", type=int, default=3, help="skip products with fewer text reviews")
    p.add_argument("--min-len", type=int, default=10, help="min chars in text+pros+cons")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--delay", type=float, default=0.05, help="per-worker pause between products, s")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    asyncio.run(WBParser(args).run())


if __name__ == "__main__":
    main()
