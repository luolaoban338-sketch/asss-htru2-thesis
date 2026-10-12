import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

def classification_metrics(y, probabilities, threshold=.5):
    y=np.asarray(y);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or p.shape!=y.shape or len(y)==0:raise ValueError('Invalid metric input shape')
    if not set(y)<= {0,1} or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid labels/probabilities')
    predicted=(p>=threshold).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,predicted,labels=[0,1]).ravel())
    denom=2*tp+fp+fn
    return {'f1':2*tp/denom if denom else 0.0,'f1_undefined':denom==0,
        'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'auc_undefined_reason':None if len(set(y))==2 else 'one_class_only',
        'average_precision':float(average_precision_score(y,p)) if np.any(y==1) else None,
        'accuracy':(tp+tn)/len(y),'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,
        'fpr':fp/(fp+tn) if fp+tn else None,
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,'n':len(y),'positive_n':tp+fn,'negative_n':tn+fp}

def group_metrics(audit,y,p,threshold=.5):
    result=[]
    for col in audit.columns:
        values=audit[col].fillna('__MISSING__').astype(str).to_numpy()
        for value in sorted(set(values)):
            mask=values==value
            result.append({'attribute':col,'group':value,**classification_metrics(y[mask],p[mask],threshold)})
    return result
