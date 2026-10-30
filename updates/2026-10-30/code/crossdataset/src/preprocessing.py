"""Training-only tabular encoder with an explicit original-variable block map."""
import numpy as np
import pandas as pd

class TabularEncoder:
    def __init__(self, numeric, categorical, mode='model', clip_quantiles=(.01,.99), zero_scale=1.0):
        if mode not in ('model','ordinary_geometry','robust_geometry'): raise ValueError(mode)
        self.numeric=tuple(numeric);self.categorical=tuple(categorical);self.mode=mode
        self.clip_quantiles=tuple(clip_quantiles);self.zero_scale=float(zero_scale)
        if not 0<=self.clip_quantiles[0]<self.clip_quantiles[1]<=1 or self.zero_scale<=0:
            raise ValueError('Invalid geometry scale or quantiles')

    def fit(self, X):
        if not len(X): raise ValueError('Empty training table')
        expected=set(self.numeric)|set(self.categorical)
        if set(X.columns)!=expected or set(self.numeric)&set(self.categorical):
            raise ValueError('Explicit schema mismatch')
        self.feature_order_=tuple(X.columns)
        self.fit_n_rows_=len(X)
        self.params_={}; self.blocks_={}; self.encoded_names_=[]
        for col in self.feature_order_:
            if col in self.numeric:
                a=pd.to_numeric(X[col],errors='raise').to_numpy(dtype=float,na_value=np.nan)
                if np.isinf(a).any(): raise ValueError('Infinite numeric predictor')
                missing=np.isnan(a)
                median=float(np.nanmedian(a)) if not missing.all() else 0.0
                a=np.where(missing,median,a)
                if self.mode=='robust_geometry':
                    q1,q3=np.quantile(a,[.25,.75]);scale=float(q3-q1)
                    low,high=map(float,np.quantile(a,self.clip_quantiles));center=float(np.median(a))
                else:
                    center=float(a.mean());scale=float(a.std(ddof=0));low=high=None
                zero=scale==0
                self.params_[col]={'kind':'numeric','impute':median,'center':center,
                    'scale':self.zero_scale if zero else scale,'zero_scale':zero,'clip_low':low,'clip_high':high,
                    'all_missing':bool(missing.all())}
                width=1; names=[col]
            else:
                s=X[col].astype(object)
                nonmissing=s.dropna().map(str)
                if len(nonmissing):
                    counts=nonmissing.value_counts();top=counts.max()
                    mode=sorted(counts[counts==top].index)[0]
                else: mode='__MISSING__'
                filled=s.map(lambda x: mode if pd.isna(x) else str(x))
                cats=sorted(filled.unique())
                width=len(cats)
                self.params_[col]={'kind':'categorical','impute':mode,'categories':cats,
                    'block_scale':float(np.sqrt(width)) if self.mode!='model' else 1.0,
                    'all_missing':bool(len(nonmissing)==0)}
                names=[f'{col}={c}' for c in cats]
            start=len(self.encoded_names_)
            self.encoded_names_.extend(names)
            self.blocks_[col]=list(range(start,start+width))
        return self

    def transform(self, X):
        if tuple(X.columns)!=self.feature_order_: raise ValueError('Feature order mismatch')
        out=[]
        for col in self.feature_order_:
            p=self.params_[col]
            if p['kind']=='numeric':
                a=pd.to_numeric(X[col],errors='raise').to_numpy(dtype=float,na_value=np.nan)
                if np.isinf(a).any(): raise ValueError('Infinite numeric predictor')
                a=np.where(np.isnan(a),p['impute'],a)
                if p['clip_low'] is not None: a=np.clip(a,p['clip_low'],p['clip_high'])
                out.append(((a-p['center'])/p['scale'])[:,None])
            else:
                a=X[col].map(lambda x:p['impute'] if pd.isna(x) else str(x)).to_numpy()
                out.append(np.column_stack([a==c for c in p['categories']]).astype(float)/p['block_scale'])
        matrix=np.concatenate(out,axis=1)
        if not np.isfinite(matrix).all(): raise ValueError('Nonfinite encoded output')
        return matrix

    def fit_transform(self,X): return self.fit(X).transform(X)

    def diagnostics(self,X):
        return {'fit_n_rows':self.fit_n_rows_, 'encoded_d':len(self.encoded_names_),
            'blocks':self.blocks_, 'parameters':self.params_,
            'unknown_category_counts':{c:int((~X[c].map(lambda x:self.params_[c]['impute'] if pd.isna(x) else str(x)).isin(self.params_[c]['categories'])).sum()) for c in self.categorical}}
