"""All part A experiment configs. Each group varies one thing on top of the base setup.

'best' in features / model means: take the best row (by validation macro-F1) of that group for the dataset,
so later groups build on what earlier groups found.
"""

BASE_FEATURES = {'text': 'full', 'word': [1, 2], 'min_df': 3, 'max_features': 200_000, 'sublinear': True}
BASE_MODEL = {'name': 'logreg', 'C': 2.0}
ALL_META = ['flags', 'length', 'punct', 'category']


def feat(**changes):
    """Base feature spec with some fields changed."""
    return {**BASE_FEATURES, **changes}


def exp(exp_id, group, note, features=None, model=None, imbalance='balanced', filters=(), decode='argmax'):
    """One experiment config."""
    return {'exp_id': exp_id, 'group': group, 'note': note, 'features': features or BASE_FEATURES,
            'model': model or BASE_MODEL, 'imbalance': imbalance, 'filters': list(filters), 'decode': decode}


def feature_experiments(dataset):
    """Text representations and meta features with the fixed base model (logreg C=2, balanced weights)."""
    g = 'features'
    exps = [
        exp('f01_word11', g, 'words only', feat(word=[1, 1])),
        exp('f02_word12', g, 'words + bigrams (base)', feat()),
        exp('f03_word13', g, 'up to trigrams, 300k', feat(word=[1, 3], max_features=300_000)),
        exp('f04_word12_full_vocab', g, 'no max_features, min_df=2', feat(min_df=2, max_features=None)),
        exp('f05_word12_50k', g, 'max_features 50k', feat(max_features=50_000)),
        exp('f06_word12_raw_tf', g, 'no sublinear tf', feat(sublinear=False)),
        exp('f07_word12_binary', g, 'binary tf', feat(binary=True)),
        exp('f08_char25', g, 'char_wb 2-5 only', feat(word=None, char=[2, 5])),
        exp('f09_char14', g, 'char_wb 1-4 only', feat(word=None, char=[1, 4])),
        exp('f10_word12_char25', g, 'words+bigrams + char_wb 2-5', feat(char=[2, 5])),
        exp('f11_plain', g, 'text without field markers', feat(text='plain')),
        exp('f12_plain_flags', g, 'no markers + field flags', feat(text='plain', meta=['flags'])),
        exp('f13_meta', g, '+ flags, length, punctuation', feat(meta=['flags', 'length', 'punct'])),
        exp('f14_meta_cat', g, '+ flags, length, punctuation, category', feat(meta=ALL_META)),
        exp('f15_clean', g, 'normalized text (emoji, !, digits, ё)', feat(text='clean')),
        exp('f16_stem', g, 'snowball stems', feat(text='stem')),
        exp('f18_stem_char_meta', g, 'stems + char_wb 2-5 + all meta', feat(text='stem', char=[2, 5], meta=ALL_META)),
    ]
    if dataset == 'wb':
        exps.append(exp('f17_lemma', g, 'pymorphy lemmas', feat(text='lemma')))
    return exps


def model_experiments():
    """Linear model families and their regularization on the best feature set."""
    g = 'models'
    specs = (
        [('logreg', {'C': c}) for c in [0.5, 1.0, 4.0, 8.0]]
        + [('linsvc', {'C': c}) for c in [0.03, 0.1, 0.3, 1.0]]
        + [('sgd', {'loss': 'modified_huber', 'alpha': a}) for a in [1e-5, 3e-6]]
        + [('sgd', {'loss': 'log_loss', 'alpha': 3e-6}), ('sgd', {'loss': 'hinge', 'alpha': 1e-5})]
        + [('ridge_clf', {'alpha': a}) for a in [1.0, 3.0, 10.0]]
        + [('cnb', {'alpha': a}) for a in [0.1, 0.3, 1.0]]
        + [('mnb', {'alpha': a}) for a in [0.03, 0.1]]
        + [('nbsvm', {'C': c}) for c in [1.0, 4.0]]
        + [('pa', {'C': c}) for c in [0.01, 0.1]]
    )
    return [exp(f'm_{name}_' + '_'.join(f'{k}{v}' for k, v in p.items()), g, name, 'best', {'name': name, **p})
            for name, p in specs]


def imbalance_experiments():
    """Ways to handle the dominant 5★ class (base uses balanced class weights)."""
    g = 'imbalance'
    return [
        exp('i_none', g, 'no reweighting', 'best', imbalance='none'),
        exp('i_sqrt', g, 'sqrt of balanced weights', 'best', imbalance='sqrt'),
        exp('i_under', g, 'undersample every class to the rarest', 'best', imbalance='under'),
        exp('i_under5', g, 'undersample only 5★ to 2× the next class', 'best', imbalance='under5'),
        exp('i_over', g, 'oversample minorities to half of 5★', 'best', imbalance='over'),
    ]


def data_experiments(dataset):
    """Train-only data cleaning; validation is never filtered."""
    g = 'data'
    exps = [
        exp('d_drop_short', g, 'drop train reviews with <3 words', 'best', filters=['drop_short']),
        exp('d_dedup', g, 'drop duplicate texts in train', 'best', filters=['dedup']),
        exp('d_drop_short_dedup', g, 'both', 'best', filters=['drop_short', 'dedup']),
    ]
    if dataset == 'wb':
        exps.append(exp('d_drop_not_product', g, 'drop reviews about delivery (WB notProduct)', 'best',
                        filters=['drop_not_product']))
    return exps


def decoding_experiments():
    """Prediction rules and ordinal formulations; tuned on out-of-fold scores of the fit part only."""
    g = 'decoding'
    return [
        exp('o_logreg_median', g, 'median of predicted distribution (MAE-optimal)', 'best', decode='median'),
        exp('o_logreg_bias', g, 'per-class score offsets tuned for macro-F1', 'best', decode='bias'),
        exp('o_logreg_noweight_bias', g, 'no weights + offsets tuned for macro-F1', 'best', imbalance='none',
            decode='bias'),
        exp('o_linsvc_bias', g, 'linear SVM + offsets', 'best', {'name': 'linsvc', 'C': 0.1}, decode='bias'),
        exp('o_ridge_reg_round', g, 'ridge regression, rounded', 'best', {'name': 'ridge_reg', 'alpha': 1.0}),
        exp('o_ridge_reg_thr1', g, 'ridge regression + tuned thresholds', 'best', {'name': 'ridge_reg', 'alpha': 1.0},
            decode='thresholds'),
        exp('o_ridge_reg_thr3', g, 'ridge regression alpha 3 + tuned thresholds', 'best',
            {'name': 'ridge_reg', 'alpha': 3.0}, decode='thresholds'),
    ]


def selection_experiments():
    """chi² feature selection on top of the best feature set."""
    return [exp(f's_chi2_{k // 1000}k', 'selection', f'chi² top {k:,}', {'from': 'best', 'select_k': k})
            for k in [20_000, 50_000, 100_000]]


def combo_experiments():
    """Second wave: the strongest findings of the groups above combined."""
    g = 'combo'
    logreg1 = {'name': 'logreg', 'C': 1.0}
    svc = {'name': 'linsvc', 'C': 0.03}
    with_char = {'from': 'best', 'char': [2, 5]}
    return [
        exp('c_logreg1_none_bias', g, 'logreg C=1, no weights, offsets', 'best', logreg1, 'none', decode='bias'),
        exp('c_logreg1_sqrt_bias', g, 'logreg C=1, sqrt weights, offsets', 'best', logreg1, 'sqrt', decode='bias'),
        exp('c_logreg1_none_median', g, 'logreg C=1, no weights, median (for MAE)', 'best', logreg1, 'none',
            decode='median'),
        exp('c_svc_none_bias', g, 'linear SVM C=0.03, no weights, offsets', 'best', svc, 'none', decode='bias'),
        exp('c_svc_sqrt_bias', g, 'linear SVM C=0.03, sqrt weights, offsets', 'best', svc, 'sqrt', decode='bias'),
        exp('c_nbsvm_none_bias', g, 'NB-SVM C=1, no weights, offsets', 'best', {'name': 'nbsvm', 'C': 1.0}, 'none',
            decode='bias'),
        exp('c_ridge_none_bias', g, 'ridge classifier alpha=10, no weights, offsets', 'best',
            {'name': 'ridge_clf', 'alpha': 10.0}, 'none', decode='bias'),
        exp('c_logreg1_none_bias_clean', g, 'logreg C=1, no weights, offsets, train without short/duplicates',
            'best', logreg1, 'none', filters=['drop_short', 'dedup'], decode='bias'),
        exp('c_logreg1_none_bias_char', g, 'logreg C=1, no weights, offsets, + char_wb 2-5', with_char, logreg1,
            'none', decode='bias'),
        exp('c_svc_none_bias_char', g, 'linear SVM C=0.03, no weights, offsets, + char_wb 2-5', with_char, svc,
            'none', decode='bias'),
    ]


GROUPS = ['features', 'models', 'imbalance', 'data', 'decoding', 'selection', 'combo']


def all_experiments(dataset):
    """Every config for a dataset, in the order groups depend on each other."""
    return (feature_experiments(dataset) + model_experiments() + imbalance_experiments()
            + data_experiments(dataset) + decoding_experiments() + selection_experiments() + combo_experiments())
