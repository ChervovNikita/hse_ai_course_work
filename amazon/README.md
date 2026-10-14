# Amazon Reviews: англоязычный аналог датасета WB

Источник: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley Lab, UCSD). Парсинг не нужен, датасет готовый.

```bash
python amazon/prepare.py   # скачивает 5 категорий (~2.8 ГБ в data/amazon_raw/), строит выборку
```

Выборка повторяет логику WB-парсера:
- только отзывы с текстом ≥10 символов;
- только товары с ≥3 такими отзывами;
- не больше 100 отзывов на товар;
- по 60k отзывов из 5 категорий физических товаров: Amazon_Fashion, All_Beauty, Handmade_Products, Health_and_Personal_Care, Appliances.

Автоматические заголовки вида «Five Stars» (10% отзывов) вырезаются: они раскрывают оценку.

Файлы:
- `data/amazon_reviews.csv.gz`: 300 034 отзыва, 26 352 товара. Колонки: `product_id`, `rating`, `title`, `text`, `full_text` (title + text), `created_date`, `helpful_vote`, `verified_purchase`, `category`, `star_title`.
- `data/amazon_reviews_balanced.csv`: 77 610 отзывов, по 15 522 на оценку.

Сравнение с WB: `comparison.ipynb`.
