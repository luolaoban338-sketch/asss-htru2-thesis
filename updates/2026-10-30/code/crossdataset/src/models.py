"""Seeded classifier construction and model-agnostic fit provenance."""
from numbers import Integral

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_classifier(model_config, seed):
    """Construct the explicitly named learner without fitting it.

    The call-site seed is authoritative. Forests are restricted to one worker so
    that the shared fit ledger accounts for each classifier operation reliably.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 32:
        raise ValueError('Classifier seed must be an explicit uint32 integer')
    params = dict(model_config)
    name = params.pop('class', None)
    if name not in ('LogisticRegression', 'RandomForestClassifier'):
        raise ValueError(f'Unsupported classifier class: {name!r}')
    params['random_state'] = int(seed)
    if name == 'RandomForestClassifier':
        if params.get('n_jobs', 1) != 1:
            raise ValueError('RandomForestClassifier requires n_jobs=1')
        params['n_jobs'] = 1
        return RandomForestClassifier(**params)
    return LogisticRegression(**params)


def fit_count_metadata(model_config, count):
    """Count classifier fits; the legacy logistic counter applies only to LR."""
    name = model_config.get('class')
    if name not in ('LogisticRegression', 'RandomForestClassifier'):
        raise ValueError(f'Unsupported classifier class: {name!r}')
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError('Fit count must be a nonnegative integer')
    result = {'model_class': name, 'classifier_fits': int(count)}
    if name == 'LogisticRegression':
        result['logistic_fits'] = int(count)
    return result


def learner_metadata(model):
    """Return fitted attributes only where the learner actually provides them."""
    result = {'model_class': type(model).__name__, 'model_parameters': model.get_params(deep=False)}
    for source, target in [('classes_', 'classes'), ('n_features_in_', 'n_features_in'),
                           ('coef_', 'coef'), ('intercept_', 'intercept'),
                           ('n_iter_', 'n_iter'), ('feature_importances_', 'feature_importances')]:
        if hasattr(model, source):
            result[target] = getattr(model, source)
    if hasattr(model, 'estimators_'):
        result['fitted_estimators'] = len(model.estimators_)
    return result
