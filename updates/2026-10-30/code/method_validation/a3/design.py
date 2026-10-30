"""Deterministic typed coordinates and attempt inventory (no estimator imports)."""
import hashlib
import json

METHODS = ('ASSS111', 'mean_shapley100_alias000', 'L1_coefficient_top4',
           'RF_MDI_top4', 'forward_CV_top4', 'random4', 'all_features_context')
PROTOCOL_ID = 'a3-htru2-traditional-20261018-v1'
ROOT_SEED = 2026101801
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ROLES = {'inner_split', 'contribution_class', 'utility_split', 'utility_model',
         'permutation', 'inner_downstream', 'l1_inner', 'l1_rebuild', 'rf_model',
         'random_features', 'final_model', 'probe_lr', 'probe_rf'}
ROLE_BOUNDS = {'inner_split': (), 'contribution_class': (4, 2), 'utility_split': (4,),
    'utility_model': (4, 3), 'permutation': (4,), 'inner_downstream': (3,),
    'l1_inner': (5, 3), 'l1_rebuild': (5,), 'rf_model': (), 'random_features': (),
    'final_model': (), 'probe_lr': (16,), 'probe_rf': (16,)}


def seed_key(outer_fold, role, *indices):
    if type(outer_fold) is not int or outer_fold not in range(5):
        raise ValueError('A3 outer fold must be an integer 0..4')
    if role not in ROLES or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('Unknown role or invalid seed coordinate')
    bounds = ROLE_BOUNDS[role]
    if len(indices) != len(bounds) or any(i >= cap for i, cap in zip(indices, bounds)):
        raise ValueError('Coordinate outside frozen seed inventory')
    if role.startswith('probe_') and outer_fold != 0:
        raise ValueError('Frozen probes use outer fold zero only')
    return [PROTOCOL_ID, ROOT_SEED, 'A3', outer_fold, role, *indices]


def seed(outer_fold, role, *indices):
    raw = json.dumps(seed_key(outer_fold, role, *indices), separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big') % 2147483647


def seed_inventory():
    records = []
    for f in range(5):
        coordinates = [(r, ()) for r in ('inner_split', 'rf_model', 'random_features', 'final_model')]
        for b in range(4):
            coordinates += [('contribution_class', (b, c)) for c in (0, 1)]
            coordinates += [('utility_split', (b,)), ('permutation', (b,))]
            coordinates += [('utility_model', (b, j)) for j in range(3)]
        coordinates += [('inner_downstream', (j,)) for j in range(3)]
        coordinates += [('l1_inner', (c, j)) for c in range(5) for j in range(3)]
        coordinates += [('l1_rebuild', (c,)) for c in range(5)]
        if f == 0:
            coordinates += [(r, (i,)) for r in ('probe_lr', 'probe_rf') for i in range(16)]
        records.extend({'key': seed_key(f, role, *idx), 'seed': seed(f, role, *idx)} for role, idx in coordinates)
    keys = [json.dumps(r['key']) for r in records]
    values = [r['seed'] for r in records]
    if len(set(keys)) != len(keys) or len(set(values)) != len(values):
        raise ValueError('Duplicate seed coordinate or collision')
    return {'schema': 1, 'count': len(records), 'collision_check': 'passed',
            'intentional_reuse': ['Same utility model seed for every coalition in one utility fold',
                'Same inner downstream LR seed across L1 C candidates and forward candidates',
                'Same final LR seed for every method within one outer fold',
                'Cold replay and direct audit reuse the exact main seed, never derive audit seeds'],
            'records': records}


def resource_inventory():
    main = {'utility': 5 * 4 * 255 * 3, 'l1_inner': 5 * 5 * 3 * 2,
            'l1_rebuild': 5, 'rf_selector': 5, 'forward': 5 * (8 + 7 + 6 + 5) * 3,
            'final': 5 * len(METHODS)}
    rows = [{'phase': 'main', 'component': k, 'classifier': v, 'encoder': v} for k, v in main.items()]
    rows += [{'phase': 'main', 'component': 'redundancy', 'classifier': 0, 'encoder': 5},
        {'phase': 'replay', 'component': 'complete_fold_0', 'classifier': sum(main.values()) // 5,
         'encoder': sum(main.values()) // 5 + 1},
        {'phase': 'audit', 'component': 'all_final_models', 'classifier': 35, 'encoder': 35},
        {'phase': 'audit', 'component': 'full_coalition_all_utility_folds', 'classifier': 60, 'encoder': 60},
        {'phase': 'probe', 'component': 'lr_and_rf_16_each', 'classifier': 32, 'encoder': 32}]
    totals = {kind: sum(r[kind] for r in rows) for kind in ('classifier', 'encoder')}
    return {'rows': rows, 'planned': totals, 'caps': {'classifier': 25000, 'encoder': 30000, 'kmeans': 0},
        'failure_reserve': {'classifier': 25000 - totals['classifier'], 'encoder': 30000 - totals['encoder']},
        'unit': 'One explicit top-level estimator.fit or encoder.fit attempt, including failures.',
        'rf_internal_tree_fits': {'nominal_trees_per_successful_forest': 64, 'planned_forest_attempts': 22,
                                 'nominal_tree_starts': 1408, 'not_equated_with_LR_fits': True},
        'replay_cache_policy': 'Cold replay shares no fitted objects, preprocessing or utility cache with main.',
        'audit_targets': 'Every final arm/fold; every full-feature coalition in B=4 games and 3 utility folds per outer fold.',
        'constant_utility_evaluations': {'main': 60, 'cold_replay': 12, 'classifier_fits': 0},
        'no_retry': True}
