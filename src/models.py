import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.special import logit

COLUMNS=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']+[f'BILL_AMT{i}' for i in range(1,7)]+[f'PAY_AMT{i}' for i in range(1,7)]
BASE=['LR','RF','XGB','LGB','CAT']; BOOST=['XGB','LGB','CAT']
NAMES=[n+s for n in BASE+['VOTE','STACK'] for s in ['', '_W']]
CATCOL=[1,2,3]; NUMCOL=[i for i in range(23) if i not in CATCOL]

def preprocessor(linear=False):
    num=make_pipeline(SimpleImputer(strategy='median'),StandardScaler()) if linear else SimpleImputer(strategy='median')
    return ColumnTransformer([('num',num,NUMCOL),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),CATCOL)], sparse_threshold=0)

class FittedBase:
    def __init__(self,prep,est): self.prep=prep; self.est=est
    def predict_proba(self,X): return self.est.predict_proba(self.prep.transform(X))

def fit_base(name,X,y,seed,cfg,weighted):
    n=cfg['n_estimators']; t=cfg['threads']
    estimators={'LR':lambda:LogisticRegression(max_iter=1000,C=1), 'RF':lambda:RandomForestClassifier(n_estimators=n,max_depth=12,min_samples_leaf=5,n_jobs=t,random_state=seed), 'XGB':lambda:XGBClassifier(n_estimators=n,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,n_jobs=t,random_state=seed,eval_metric='logloss'), 'LGB':lambda:LGBMClassifier(n_estimators=n,num_leaves=31,learning_rate=.05,min_child_samples=20,n_jobs=t,random_state=seed,verbosity=-1), 'CAT':lambda:CatBoostClassifier(iterations=n,depth=5,learning_rate=.05,thread_count=t,random_seed=seed,verbose=False,allow_writing_files=False)}
    prep=preprocessor(name=='LR'); est=estimators[name](); z=prep.fit_transform(X)
    w=compute_sample_weight('balanced',y) if weighted else np.ones(len(y))
    est.fit(z,y,sample_weight=w)
    return FittedBase(prep,est)

class Ensemble:
    def __init__(self,base,meta=None): self.base=base; self.meta=meta
    def predict_proba(self,X):
        z=np.column_stack([m.predict_proba(X)[:,1] for m in self.base]); p=z.mean(axis=1) if self.meta is None else self.meta.predict_proba(z)[:,1]
        return np.column_stack([1-p,p])

def fit_models(X,y,g,seed,cfg,names=NAMES):
    out={}
    for weighted in [False,True]:
        suffix='_W' if weighted else ''; wanted=[n for n in BASE+['VOTE','STACK'] if n+suffix in names]
        if not wanted: continue
        needed=set(wanted)&set(BASE)
        if set(wanted)&{'VOTE','STACK'}: needed.update(BOOST)
        models={n:fit_base(n,X,y,seed,cfg,weighted) for n in BASE if n in needed}
        for n in wanted:
            if n in models: out[n+suffix]=models[n]
        if 'VOTE' in wanted: out['VOTE'+suffix]=Ensemble([models[n] for n in BOOST])
        if 'STACK' in wanted:
            oof=np.full((len(y),3),np.nan)
            for tr,va in StratifiedGroupKFold(n_splits=cfg['folds'],shuffle=True,random_state=seed).split(X,y,g):
                for j,n in enumerate(BOOST): oof[va,j]=fit_base(n,X[tr],y[tr],seed,cfg,weighted).predict_proba(X[va])[:,1]
            assert np.isfinite(oof).all()
            meta=LogisticRegression(C=1,max_iter=1000).fit(oof,y)
            out['STACK'+suffix]=Ensemble([models[n] for n in BOOST],meta)
    return out

def predict(model,X): return np.clip(model.predict_proba(X)[:,1],1e-7,1-1e-7)
class Calibrated:
    def __init__(self,base,kind,X,y):
        self.base=base; self.kind=kind; p=predict(base,X)
        if kind=='PLATT': self.cal=LogisticRegression(C=1e4,max_iter=1000).fit(logit(p).reshape(-1,1),y)
        elif kind=='ISO': self.cal=IsotonicRegression(out_of_bounds='clip').fit(p,y)
    def predict_proba(self,X):
        p=predict(self.base,X)
        if self.kind=='PLATT': p=self.cal.predict_proba(logit(p).reshape(-1,1))[:,1]
        elif self.kind=='ISO': p=self.cal.predict(p)
        return np.column_stack([1-p,p])
