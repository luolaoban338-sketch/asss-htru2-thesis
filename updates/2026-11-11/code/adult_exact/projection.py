"""Pure resource arithmetic for the frozen exact-count contract.

These values are engineering allowances. They neither measure runtime nor grant
execution permission. A caller supplies only durably committed task identities.
"""
import math

UTILITY_BY_FOLD = (50091, 50346, 50382, 49956, 50484)
REQUIRED_RATES = ('t_utility', 't_full_candidate', 't_reference', 't_encoder',
                  't_kmeans', 't_game_setup', 't_selector_setup')
OPTIONAL_RATES = ('t_direct_audit_cycle', 't_direct_audit_reference')
PROBE_TASKS = frozenset(
    [f'fold{f}-{c}-{m}-repeat{r}' for f in range(5)
     for c in ('inner', 'refit') for m in ('full', 'robust_half') for r in range(3)]
    + [f'fold{f}-full-reference-repeat{r}' for f in range(5) for r in range(3)])


def _duration(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid finite duration: ' + name)
    return float(value)


def refresh_rates(rates, observations=()):
    """Merge complete observed cycle/setup/operation maxima upward only."""
    result = dict(rates)
    for name in REQUIRED_RATES:
        result[name] = _duration(result.get(name), name, positive=True)
    for name in OPTIONAL_RATES:
        result[name] = _duration(result.get(name, 0.), name)
    for name, seconds in observations:
        if name not in REQUIRED_RATES + OPTIONAL_RATES:
            raise ValueError('Unknown timing observation: ' + str(name))
        result[name] = max(result[name], _duration(seconds, name))
    return result


def remaining_counts(completed_tasks=()):
    """Credit exactly the frozen work of complete bundles, never partial work.

    Probe identities have no production credit. Retry reserves are deliberately
    never deducted, including after failed, interrupted or successful retries.
    """
    seen = set(); computations = []; audits = []
    for item in completed_tasks:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('A committed completion must be a role/task pair')
        role, task = item
        if type(role) is not str or type(task) is not str:
            raise ValueError('Invalid committed task identity')
        key = (role, task)
        if key in seen: raise ValueError('Duplicate committed task identity')
        seen.add(key)
        if role == 'probe' and task in PROBE_TASKS:
            continue
        if role == 'main' and task in {f'fold-{f}' for f in range(5)}:
            computations.append(int(task[-1]))
        elif role == 'replay' and task == 'fold-0':
            computations.append(0)
        elif role == 'audit' and task in {f'main-fold-{f}' for f in range(5)} | {'replay-fold-0'}:
            audits.append(task)
        else:
            raise ValueError('Unknown or partial committed task identity: ' + repr(key))
    # Source computations must already have committed before their direct audit.
    for task in audits:
        source, _, fold = task.split('-')
        if (source, 'fold-' + fold) not in seen:
            raise ValueError('Committed audit lacks its committed source bundle')
    nc, na = len(computations), len(audits)
    return {'utility': 301350 - sum(UTILITY_BY_FOLD[f] for f in computations),
            'other_cycles_including_direct_audit': 1056 - 76 * nc - 100 * na,
            'complete_training_reference_and_audit': 12 - nc - na,
            'retry_classifier_pool': 42840, 'retry_classifier_reference': 12,
            'retry_encoder': 44132, 'retry_kmeans': 152,
            'game_initializations_including_audit': 384 - 32 * (nc + na),
            'selector_setups_including_audit': 96 - 8 * (nc + na)}


def projection_terms(rates, completed_tasks=()):
    rates = refresh_rates(rates)
    counts = remaining_counts(completed_tasks)
    utility = rates['t_utility']
    pool = max(utility, rates['t_full_candidate'], rates['t_direct_audit_cycle'])
    reference = max(rates['t_reference'], rates['t_direct_audit_reference'])
    units = {'utility': utility, 'other_cycles_including_direct_audit': pool,
             'complete_training_reference_and_audit': reference,
             'retry_classifier_pool': pool, 'retry_classifier_reference': reference,
             'retry_encoder': rates['t_encoder'], 'retry_kmeans': rates['t_kmeans'],
             'game_initializations_including_audit': rates['t_game_setup'],
             'selector_setups_including_audit': rates['t_selector_setup']}
    return {name: count * units[name] for name, count in counts.items()}


def remaining_seconds(rates, completed_tasks=()):
    return 3.75 * sum(projection_terms(rates, completed_tasks).values()) + 3600.


def total_seconds(rates, old_probe_seconds, new_probe_seconds):
    """Initial candidate gate: both prior and new probes remain charged."""
    return (_duration(old_probe_seconds, 'old_probe_seconds')
            + _duration(new_probe_seconds, 'new_probe_seconds', positive=True)
            + remaining_seconds(rates))
