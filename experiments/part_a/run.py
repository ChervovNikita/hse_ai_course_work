"""Runs part A experiments: fit on the fit part of train, score on the validation part, log to results.csv.

  python experiments/part_a/run.py                       # everything, both datasets
  python experiments/part_a/run.py --datasets wb --groups features models

Already logged (dataset, exp_id) pairs are skipped, so the run can be stopped and restarted.
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np

from common import (CLASSES, Data, build_features, load_results, log_result, make_model, median_decode, metrics,
                    oof_scores, scores, select_rows, tune_bias, tune_thresholds)
from experiments import GROUPS, all_experiments


def best_features(dataset):
    """Feature spec of the best validation macro-F1 in the features group."""
    res = load_results()
    res = res[(res.dataset == dataset) & (res.group == 'features')]
    return json.loads(res.loc[res.macro_f1.idxmax(), 'features'])


def resolve_features(dataset, spec):
    """Turns 'best' / {'from': 'best', ...} into a concrete feature spec."""
    if spec == 'best':
        return best_features(dataset)
    if isinstance(spec, dict) and spec.get('from') == 'best':
        return {**best_features(dataset), **{k: v for k, v in spec.items() if k != 'from'}}
    return spec


def predict(model, cfg, x_tr, y_tr, weight, groups, x_val):
    """Fits the model and decodes validation predictions with the configured rule."""
    decode = cfg['decode']
    if decode == 'bias':
        offsets = tune_bias(oof_scores(model, x_tr, y_tr, weight, groups), y_tr)
    if decode == 'thresholds':
        cuts = tune_thresholds(oof_scores(model, x_tr, y_tr, weight, groups), y_tr)
    model.fit(x_tr, y_tr, sample_weight=weight)
    if decode == 'argmax' and cfg['model']['name'] == 'ridge_reg':
        return np.clip(np.rint(model.predict(x_val)), 1, 5).astype(int)
    if decode == 'argmax':
        return model.predict(x_val)
    s = scores(model, x_val)
    if decode == 'median':
        return median_decode(s)
    if decode == 'bias':
        return CLASSES[(s + offsets).argmax(axis=1)]
    return CLASSES[np.digitize(s, cuts)]


def run_dataset(dataset, groups):
    """Runs every not yet logged experiment of the given groups for one dataset."""
    data = Data(dataset)
    print(f'\n### {dataset}: fit {len(data.fit_idx):,}, validation {len(data.val_idx):,}', flush=True)
    cache = {}
    for cfg in all_experiments(dataset):
        done = load_results()
        if cfg['group'] not in groups or ((done.dataset == dataset) & (done.exp_id == cfg['exp_id'])).any():
            continue
        spec = resolve_features(dataset, cfg['features'])
        key = json.dumps(spec, sort_keys=True)
        if cache.get('key') != key:
            cache = {'key': key, 'x': build_features(data, spec)}
        x_fit, x_val = cache['x']
        rows, weight = select_rows(data, x_fit, cfg)
        start = time.time()
        pred = predict(make_model(cfg['model']), cfg, x_fit[rows], data.y_fit[rows], weight,
                       data.groups_fit[rows], x_val)
        row = {'dataset': dataset, 'group': cfg['group'], 'exp_id': cfg['exp_id'], 'note': cfg['note'],
               **metrics(data.y_val, pred), 'features': spec, 'model': cfg['model'], 'imbalance': cfg['imbalance'],
               'filters': cfg['filters'], 'decode': cfg['decode'], 'n_train': len(rows), 'n_features': x_fit.shape[1],
               'fit_sec': round(time.time() - start, 1), 'finished': datetime.now().isoformat(timespec='seconds')}
        log_result(row)
        print(f"{cfg['exp_id']:<34} macro-F1 {row['macro_f1']:.4f}  MAE {row['mae']:.3f}  "
              f"({row['fit_sec']:.0f}s, {x_fit.shape[1]:,} features)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+', default=['wb', 'amazon'])
    p.add_argument('--groups', nargs='+', default=GROUPS)
    args = p.parse_args()
    for dataset in args.datasets:
        run_dataset(dataset, args.groups)


if __name__ == '__main__':
    main()
