"""Shared group budgets, augmentation accounting and simulation quality checks."""
import hashlib
import numpy as np
from scipy.spatial.distance import cdist


def augmentation_counts(y, n_classes, mode='balance', ratio=1., target_policy='max', cap=0):
    if mode not in ('balance', 'ratio') or ratio < 0 or not np.isfinite(ratio):
        raise ValueError('Invalid augmentation mode/ratio')
    counts = {i: int(np.sum(y == i)) for i in range(n_classes)}
    if min(counts.values()) <= 0:
        raise ValueError('Every training class must be present')
    if target_policy not in ('max','median'):
        raise ValueError('Unknown balance target')
    target = max(counts.values()) if target_policy == 'max' else int(np.median(list(counts.values())))
    result = {i: int(np.floor(n * ratio)) if mode == 'ratio' else max(0,target-n) for i,n in counts.items()}
    return {i:min(n,cap) if cap else n for i,n in result.items()}


def budget_indices(indices, y, groups, seed, max_groups=0, windows_per_class=0):
    """Select sources before windows; excluded outer-train groups stay excluded."""
    indices = np.asarray(indices)
    rng = np.random.default_rng(seed)
    available = np.unique(groups[indices])
    selected = rng.permutation(available)[:max_groups] if max_groups else available
    indices = indices[np.isin(groups[indices],selected)]
    chosen = []
    for label in np.unique(y):
        candidates = indices[y[indices] == label]
        if not len(candidates):
            raise ValueError('Source budget removes an entire class')
        if windows_per_class and len(candidates)>windows_per_class:
            candidates = rng.choice(candidates,windows_per_class,replace=False)
        chosen.extend(candidates)
    return np.sort(chosen).astype(np.int64)


def assert_complete_pairs(y,groups,n_classes):
    required=set(range(n_classes))
    for group in np.unique(groups):
        if set(y[groups==group]) != required:
            raise ValueError(f'Incomplete paired group {group}')


def count_receipt(y, generated_y, planned, n_classes, quality_y=(), joined=True):
    return {i:dict(original=int(np.sum(y==i)),planned=int(planned[i]),
                   actual_generated=int(np.sum(generated_y==i)),
                   joined_training=int(np.sum(generated_y==i)) if joined else 0,
                   quality_only=int(np.sum(np.asarray(quality_y)==i))) for i in range(n_classes)}


def generation_checks(real, y_real, generated, y_gen, names, groups=None):
    """Distances use training-only scales; report similarity, do not infer memorization."""
    if not np.isfinite(generated).all():
        raise ValueError('Generated samples contain NaN/Inf')
    scale=np.maximum(np.std(real,axis=(0,1)),1e-6)
    lo,hi=real.min(axis=(0,1)),real.max(axis=(0,1))
    exact_real=exact_self=0; distances=[]; baselines=[]
    for label in np.unique(y_gen):
        a=real[y_real==label]; b=generated[y_gen==label]
        keys={hashlib.sha256(x.tobytes()).digest() for x in a}
        genkeys=[hashlib.sha256(x.tobytes()).digest() for x in b]
        exact_real+=sum(k in keys for k in genkeys); exact_self+=len(genkeys)-len(set(genkeys))
        aa=(a[:128]/scale).reshape(min(len(a),128),-1)
        bb=(b[:128]/scale).reshape(min(len(b),128),-1)
        distances.extend((cdist(bb,aa).min(axis=1)/np.sqrt(aa.shape[1])).tolist())
        dr=cdist(aa,aa)/np.sqrt(aa.shape[1]); np.fill_diagonal(dr,np.inf)
        if groups is not None:
            gg=np.asarray(groups)[y_real==label][:128]
            dr[gg[:,None]==gg[None,:]]=np.inf
        nearest=dr.min(axis=1); baselines.extend(nearest[np.isfinite(nearest)].tolist())
    threshold=float(np.quantile(baselines,.05)) if baselines else None
    result=dict(finite=True,exact_training_duplicates=exact_real,exact_generated_duplicates=exact_self,
                nearest_training_rms_median=float(np.median(distances)),
                raw_cross_source_nearest_rms_p05=threshold,
                near_duplicate_fraction=None if threshold is None else float(np.mean(np.asarray(distances)<threshold)),
                similarity_interpretation='Training-only cross-source reference; near similarity alone is not proof of memorization.',
                outside_training_range_fraction=float(np.mean((generated<lo)|(generated>hi))),
                range_scope='Observed training envelope, not certified physical limits')
    if all(f'cmd_motor{i}' in names and f'rpm_feedback{i}' in names for i in range(6)):
        def correlations(x):
            out=[]
            for i in range(6):
                u=x[:,:,names.index(f'cmd_motor{i}')].ravel(); v=x[:,:,names.index(f'rpm_feedback{i}')].ravel()
                out.append(float(np.corrcoef(u,v)[0,1]) if min(u.std(),v.std())>1e-8 else None)
            return out
        result['command_feedback_correlation']={'raw_simulation':correlations(real),'generated':correlations(generated),
                                                'scope':'synchronous correlation only; includes motor and transport lag'}
    return result
