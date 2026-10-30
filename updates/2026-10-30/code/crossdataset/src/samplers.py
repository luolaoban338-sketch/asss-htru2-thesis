from math import floor,sqrt
import numpy as np
from sklearn.cluster import KMeans
from .contracts import SampleSelection, class_quotas
from .data import seed_for
from .preprocessing import TabularEncoder

def cluster_quotas(sizes, target):
    """One per effective cluster, then capped proportional Hamilton allocation."""
    sizes=np.asarray(sizes,dtype=int)
    if np.any(sizes<1) or target<len(sizes) or target>sizes.sum(): raise ValueError('Invalid cluster quota')
    q=np.ones(len(sizes),dtype=int)
    while int(q.sum())<target:
        active=np.flatnonzero(q<sizes)
        remaining=target-int(q.sum()); weight=int(sizes[active].sum())
        floors=(remaining*sizes[active])//weight
        alloc=np.minimum(floors,sizes[active]-q[active]);q[active]+=alloc
        remaining2=target-int(q.sum())
        if remaining2==0: break
        eligible=active[q[active]<sizes[active]]
        order=sorted(eligible,key=lambda i:(-int(remaining*sizes[i]%weight),int(i)))
        for i in order[:remaining2]:q[i]+=1
    return q

def rank_indices(size, quota):
    if not 1<=quota<=size: raise ValueError('Invalid within-cluster quota')
    if quota==1:return np.array([(size-1)//2],dtype=int)
    # Integer calculation avoids floating point edge changes.
    return np.array([j*(size-1)//(quota-1) for j in range(quota)],dtype=int)

def select_rows(X, train, numeric, categorical, rho, seed, strategy, settings=None):
    train.validate()
    if len(X)!=len(train.row_ids) or tuple(X.columns)!=train.feature_names:raise ValueError('Training table mismatch')
    if strategy not in ('stratified_random','ordinary_geometry','robust_geometry'):raise ValueError(strategy)
    quotas=class_quotas(train.labels,rho)
    settings=settings or {'max_clusters':10,'n_init':10,'max_iter':300,'algorithm':'lloyd',
        'clip_quantiles':[.01,.99],'zero_iqr_scale':1.0}
    y=np.array(train.labels);ids=np.array(train.row_ids)
    diagnostics={'class_quotas':quotas,'geometry':None,'clusters':[]}
    if rho==1:
        selected=np.arange(len(X))
    elif strategy=='stratified_random':
        selected=[]
        for c in (0,1):
            positions=np.flatnonzero(y==c)
            positions=positions[np.argsort(ids[positions],kind='stable')]
            rng=np.random.default_rng(seed_for(seed,'class',c))
            selected.extend(rng.choice(positions,quotas[c],replace=False).tolist())
        selected=np.array(sorted(selected),dtype=int)
    else:
        encoder=TabularEncoder(numeric,categorical,strategy,settings['clip_quantiles'],settings['zero_iqr_scale'])
        geometry=encoder.fit_transform(X)
        diagnostics['geometry']=encoder.diagnostics(X)
        selected=[]
        for c in (0,1):
            positions=np.flatnonzero(y==c)
            positions=positions[np.argsort(ids[positions],kind='stable')]
            Z=geometry[positions]
            unique_n=len(np.unique(Z,axis=0))
            k=min(settings['max_clusters'],floor(sqrt(len(positions))),quotas[c],unique_n)
            km=KMeans(n_clusters=k,init='k-means++',n_init=settings['n_init'],max_iter=settings['max_iter'],
                algorithm=settings['algorithm'],random_state=seed_for(seed,'kmeans',c)).fit(Z)
            # Stable IDs define cluster order, independent of KMeans label naming.
            labels=sorted(np.unique(km.labels_),key=lambda label:min(ids[positions[km.labels_==label]]))
            sizes=np.array([np.sum(km.labels_==label) for label in labels])
            allocation=cluster_quotas(sizes,quotas[c])
            for label,capacity,quota in zip(labels,sizes,allocation):
                member=np.flatnonzero(km.labels_==label)
                distances=np.linalg.norm(Z[member]-km.cluster_centers_[label],axis=1)
                order=np.lexsort((ids[positions[member]],distances))
                chosen=positions[member[order[rank_indices(len(member),int(quota))]]]
                selected.extend(chosen.tolist())
                diagnostics['clusters'].append({'class':c,'label':int(label),'size':int(capacity),
                    'quota':int(quota),'minimum_row_id':str(min(ids[positions[member]])),
                    'n_iter':int(km.n_iter_),'requested_clusters':k,'effective_clusters':len(labels),
                    'unique_representations':unique_n})
        selected=np.array(sorted(selected),dtype=int)
    counts={c:int(np.sum(y[selected]==c)) for c in (0,1)}
    result=SampleSelection(tuple(ids[selected]),rho,counts,int(seed),strategy,train.partition_id)
    diagnostics['actual_rho']=result.validate(train)
    return selected,result,diagnostics
