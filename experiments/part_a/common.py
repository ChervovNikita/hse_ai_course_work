"""Shared pieces of part A: data, text variants, sparse features, linear models, decoding, metrics."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymorphy3
import scipy.sparse as sp
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT.parents[1] / 'data' / 'splits'
CACHE = ROOT / 'cache'
RESULTS = ROOT / 'results.csv'
SEED = 0
VAL_SIZE = 0.2
CLASSES = np.arange(1, 6)
EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')
LANG = {'wb': 'russian', 'amazon': 'english'}
CATEGORY = {'wb': 'subject_root', 'amazon': 'category'}


class Data:
    """Train split of one platform, divided by product into fit (80%) and validation (20%) parts."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.df = pd.read_csv(SPLITS / f'{dataset}_train.csv.gz', keep_default_na=False)
        self.df['rating'] = self.df.rating.astype(int)
        self.df['n_words'] = self.df.full_text.str.split().str.len()
        splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
        self.fit_idx, self.val_idx = next(splitter.split(self.df, groups=self.df.product_id))
        self.y_fit = self.df.rating.to_numpy()[self.fit_idx]
        self.y_val = self.df.rating.to_numpy()[self.val_idx]
        self.groups_fit = self.df.product_id.to_numpy()[self.fit_idx]

    def text(self, variant):
        """Text column of the whole train split in one of the preprocessing variants, cached on disk."""
        path = CACHE / f'{self.dataset}_{variant}.pkl'
        if path.exists():
            return pd.read_pickle(path)
        CACHE.mkdir(exist_ok=True)
        texts = make_text(self.df, self.dataset, variant)
        texts.to_pickle(path)
        return texts


def normalize(s):
    """Lowercase, ё→е, emoji and '!' as tokens, digits collapsed."""
    s = s.lower().replace('ё', 'е')
    s = EMOJI.sub(lambda m: f' emoji{ord(m.group()):x} ', s)
    s = re.sub(r'!+', ' excl ', s)
    return re.sub(r'\d+', '0', s)


def map_words(texts, fn):
    """Applies fn to every word, computing it once per unique word."""
    tokens = texts.str.findall(r'\w+')
    vocab = {w: fn(w) for w in set(w for ts in tokens for w in ts)}
    return tokens.map(lambda ts: ' '.join(vocab[w] for w in ts))


def make_text(df, dataset, variant):
    """Builds one text variant: full, plain, clean, stem or lemma."""
    if variant == 'full':
        return df.full_text
    if variant == 'plain':
        # wb fields glued without the «Достоинства:» / «Недостатки:» markers; amazon is title + text already
        return (df.text + ' ' + df.pros + ' ' + df.cons).str.strip() if dataset == 'wb' else df.full_text
    clean = df.full_text.map(normalize)
    if variant == 'clean':
        return clean
    if variant == 'stem':
        return map_words(clean, SnowballStemmer(LANG[dataset]).stem)
    if variant == 'lemma':
        morph = pymorphy3.MorphAnalyzer()
        return map_words(clean, lambda w: morph.parse(w)[0].normal_form)
    raise ValueError(variant)


def meta_columns(df, dataset, kind):
    """Non-negative dense meta features of one kind."""
    if kind == 'flags':
        # filled form fields: text / pros / cons on wb, a non-empty title on amazon
        if dataset == 'wb':
            return df[['has_text', 'has_pros', 'has_cons']].to_numpy(float)
        return (df.title != '').to_numpy(float)[:, None]
    if kind == 'length':
        return np.log1p(np.c_[df.n_words, df.full_text.str.len()])
    if kind == 'punct':
        caps = df.full_text.str.count('[A-ZА-ЯЁ]') / df.full_text.str.len().clip(lower=1)
        return np.c_[np.log1p(df.full_text.str.count('!')), np.log1p(df.full_text.str.count(r'\?')),
                     np.log1p(df.full_text.str.count(EMOJI)), caps]
    raise ValueError(kind)


def build_features(data, spec):
    """Fits the feature pipeline on the fit part, returns (X_fit, X_val)."""
    texts = data.text(spec['text'])
    fit_texts, val_texts = texts.iloc[data.fit_idx], texts.iloc[data.val_idx]
    blocks_fit, blocks_val = [], []
    common = dict(min_df=spec.get('min_df', 3), sublinear_tf=spec.get('sublinear', True),
                  binary=spec.get('binary', False), dtype=np.float32)
    if spec.get('word'):
        vec = TfidfVectorizer(ngram_range=tuple(spec['word']), max_features=spec.get('max_features', 200_000),
                              **common)
        blocks_fit.append(vec.fit_transform(fit_texts))
        blocks_val.append(vec.transform(val_texts))
    if spec.get('char'):
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=tuple(spec['char']),
                              max_features=spec.get('char_max_features', 300_000), **common)
        blocks_fit.append(vec.fit_transform(fit_texts))
        blocks_val.append(vec.transform(val_texts))
    df_fit, df_val = data.df.iloc[data.fit_idx], data.df.iloc[data.val_idx]
    for kind in spec.get('meta', []):
        if kind == 'category':
            enc = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
            blocks_fit.append(enc.fit_transform(df_fit[[CATEGORY[data.dataset]]]))
            blocks_val.append(enc.transform(df_val[[CATEGORY[data.dataset]]]))
            continue
        m_fit, m_val = meta_columns(df_fit, data.dataset, kind), meta_columns(df_val, data.dataset, kind)
        scale = np.maximum(m_fit.max(axis=0), 1e-9)
        blocks_fit.append(sp.csr_matrix(m_fit / scale, dtype=np.float32))
        blocks_val.append(sp.csr_matrix(np.clip(m_val / scale, 0, None), dtype=np.float32))
    x_fit, x_val = sp.hstack(blocks_fit).tocsr(), sp.hstack(blocks_val).tocsr()
    if spec.get('select_k'):
        selector = SelectKBest(chi2, k=min(spec['select_k'], x_fit.shape[1])).fit(x_fit, data.y_fit)
        x_fit, x_val = selector.transform(x_fit), selector.transform(x_val)
    return x_fit, x_val


class NBLogReg:
    """NB-SVM style model: one-vs-rest logistic regressions on naive Bayes log-count-ratio scaled features."""

    def __init__(self, C=1.0, alpha=1.0):
        self.C = C
        self.alpha = alpha

    def get_params(self, deep=True):
        return {'C': self.C, 'alpha': self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.ratios_, self.models_ = [], []
        for c in self.classes_:
            p = self.alpha + np.asarray(x[y == c].sum(axis=0)).ravel()
            q = self.alpha + np.asarray(x[y != c].sum(axis=0)).ravel()
            r = np.log((p / p.sum()) / (q / q.sum())).astype(np.float32)
            model = LogisticRegression(C=self.C, max_iter=1000)
            model.fit(x.multiply(r).tocsr(), (y == c).astype(int), sample_weight=sample_weight)
            self.ratios_.append(r)
            self.models_.append(model)
        return self

    def decision_function(self, x):
        return np.column_stack([m.decision_function(x.multiply(r).tocsr()) for m, r in zip(self.models_, self.ratios_)])

    def predict(self, x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]


def make_model(spec):
    """Creates an unfitted model from its spec: {'name': ..., **params}."""
    params = {k: v for k, v in spec.items() if k != 'name'}
    factories = {
        'logreg': lambda: LogisticRegression(max_iter=1000, **params),
        'linsvc': lambda: LinearSVC(**params),
        'sgd': lambda: SGDClassifier(random_state=SEED, **params),
        'ridge_clf': lambda: RidgeClassifier(**params),
        'cnb': lambda: ComplementNB(**params),
        'mnb': lambda: MultinomialNB(**params),
        'nbsvm': lambda: NBLogReg(**params),
        # passive-aggressive (PA-I) as SGD, which unlike PassiveAggressiveClassifier accepts sample weights
        'pa': lambda: SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=params['C'],
                                    random_state=SEED),
        'ridge_reg': lambda: Ridge(**params),
    }
    return factories[spec['name']]()


def select_rows(data, x_fit, spec):
    """Applies train-only data filters and the imbalance strategy; returns (rows, sample_weight)."""
    df = data.df.iloc[data.fit_idx]
    keep = np.ones(len(df), bool)
    for f in spec.get('filters', []):
        if f == 'drop_short':
            keep &= df.n_words.to_numpy() >= 3
        elif f == 'dedup':
            keep &= ~df.full_text.duplicated().to_numpy()
        elif f == 'drop_not_product':
            keep &= ~df.excluded_from_rating.str.contains('notProduct').to_numpy()
        else:
            raise ValueError(f)
    rows = np.flatnonzero(keep)
    y = data.y_fit[rows]
    rng = np.random.default_rng(SEED)
    counts = pd.Series(y).value_counts()
    mode = spec.get('imbalance', 'balanced')
    if mode == 'under':
        rows = np.concatenate([rng.choice(rows[y == c], counts.min(), replace=False) for c in CLASSES])
    elif mode == 'under5':
        # only the dominant class 5 is cut down to twice the next largest class
        cap = 2 * counts.drop(5).max()
        rows = np.concatenate([rows[y != 5], rng.choice(rows[y == 5], min(cap, counts[5]), replace=False)])
    elif mode == 'over':
        # minority classes duplicated up to half of the largest class
        target = counts.max() // 2
        rows = np.concatenate([rows] + [rng.choice(rows[y == c], target - counts[c]) for c in CLASSES
                                         if counts[c] < target])
    y = data.y_fit[rows]
    counts = pd.Series(y).value_counts()
    if mode == 'balanced':
        weight = (len(y) / (len(CLASSES) * counts)).reindex(y).to_numpy()
    elif mode == 'sqrt':
        weight = np.sqrt((len(y) / (len(CLASSES) * counts)).reindex(y).to_numpy())
    else:
        weight = np.ones(len(y))
    return rows, weight


def scores(model, x):
    """Continuous model output used by decoders: log-probabilities, margins or regression values."""
    if hasattr(model, 'predict_proba'):
        return np.log(np.clip(model.predict_proba(x), 1e-12, None))
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    return model.predict(x)


def median_decode(log_proba):
    """MAE-optimal decoding: median of the predicted rating distribution."""
    cdf = np.cumsum(np.exp(log_proba), axis=1)
    return CLASSES[(cdf >= 0.5).argmax(axis=1)]


def tune_bias(s, y):
    """Per-class additive offsets to the scores that maximize macro-F1 (coordinate ascent)."""
    bias = np.zeros(s.shape[1])
    grid = np.linspace(-3, 3, 61)
    for _ in range(3):
        for k in range(s.shape[1]):
            best = max(grid, key=lambda b: f1_score(y, CLASSES[(s + np.r_[bias[:k], b, bias[k + 1:]]).argmax(1)],
                                                    average='macro'))
            bias[k] = best
    return bias


def tune_thresholds(s, y):
    """Cut points t1<..<t4 on a regression score that maximize macro-F1 (coordinate search)."""
    thresholds = np.array([1.5, 2.5, 3.5, 4.5])
    for _ in range(3):
        for k in range(4):
            lo = thresholds[k - 1] + 0.01 if k else s.min()
            hi = thresholds[k + 1] - 0.01 if k < 3 else s.max()
            grid = np.linspace(lo, hi, 50)
            best = max(grid, key=lambda t: f1_score(
                y, CLASSES[np.digitize(s, np.r_[thresholds[:k], t, thresholds[k + 1:]])], average='macro'))
            thresholds[k] = best
    return thresholds


def oof_scores(model, x, y, weight, groups):
    """Out-of-fold scores on the fit part (3 folds by product), used to tune decoders without the validation."""
    out = None
    for tr, te in GroupKFold(n_splits=3).split(x, y, groups):
        fold = clone(model)
        fold.fit(x[tr], y[tr], sample_weight=weight[tr])
        s = scores(fold, x[te])
        out = np.zeros((x.shape[0],) + s.shape[1:]) if out is None else out
        out[te] = s
    return out


def metrics(y, pred):
    """Macro-F1 (main), MAE, accuracy and per-class F1."""
    per_class = f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    return {'macro_f1': f1_score(y, pred, average='macro'), 'mae': mean_absolute_error(y, pred),
            'accuracy': accuracy_score(y, pred), **{f'f1_{c}': v for c, v in zip(CLASSES, per_class)}}


def load_results():
    """All logged experiment rows (empty frame before the first run)."""
    return pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=['dataset', 'exp_id'])


def log_result(row):
    """Appends one experiment row to results.csv."""
    df = pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()}])
    df.to_csv(RESULTS, mode='a', header=not RESULTS.exists(), index=False)
