from pathlib import Path
import sys, argparse, json, time, hashlib, itertools, warnings, platform, importlib.metadata
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd, yaml
from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import mannwhitneyu
from src.models import BASE,BOOST,COLUMNS,fit_base,fit_models,predict,Calibrated,Ensemble
from src.evaluation import metrics,threshold,stability,calibration_bins
from src.explain import permutation_shap,aggregate
from run_all import load
warnings.filterwarnings('ignore',message='X does not have valid feature names')
P=argparse.ArgumentParser();P.add_argument('stage',choices=['s1','s2','s3','s4','s5','s6']);P.add_argument('--smoke',action='store_true');P.add_argument('--force',action='store_true');args=P.parse_args()
S=yaml.safe_load((ROOT/'supplementary/config.yaml').read_text()); C=yaml.safe_load((ROOT/'config.yaml').read_text()); C.update(threads=S['threads']); R=ROOT/'results'; O=ROOT/('results_supplementary_smoke' if args.smoke else 'results_supplementary')/args.stage; O.mkdir(parents=True,exist_ok=True)
D,X,y,g=load(False); ids=D.ID.to_numpy(); lookup=pd.Series(np.arange(len(ids)),index=ids); M=pd.read_csv(R/'split_manifest.csv'); rng=np.random.default_rng(42)
assert not json.loads((R/'complete.json').read_text())['smoke']; assert hashlib.sha256((ROOT/'data/credit.csv').read_bytes()).hexdigest()==json.loads((R/'audit.json').read_text())['data_sha256']
if args.smoke:C.update(n_estimators=8,threads=1,folds=3)
def indices(v):return lookup.loc[np.asarray(v)].to_numpy()
def partition(seed):
    m=M[M.seed==seed]; out=[indices(m.loc[m.partition==p,'ID']) for p in ['train','calibration','test']]; assert not any(set(g[a])&set(g[b]) for a,b in itertools.combinations(out,2))
    if args.smoke:out=[np.sort(rng.choice(a,min(len(a),n),replace=False)) for a,n in zip(out,[900,300,300])]
    return out

def save(name,rows):
    p=O/(name+'.csv'); tmp=p.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(p)
def cached(name,fn):
    p=O/(name+'.csv');
    if p.exists() and not args.force:return pd.read_csv(p)
    save(name,fn());return pd.read_csv(p)
def evaluate(m,ca,te):return metrics(y[te],predict(m,X[te]),threshold(y[ca],predict(m,X[ca])))
def cv(idx,seed):
    for a,b in StratifiedGroupKFold(n_splits=C['folds'],shuffle=True,random_state=seed).split(X[idx],y[idx],g[idx]):yield idx[a],idx[b]
# Candidate search keeps tree count fixed and uses explicit estimator parameters via original preprocessing.
from src.models import preprocessor,FittedBase
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
SPACE={'LR':{'C':np.logspace(-3,3,30)},'RF':{'max_depth':[6,10,14,20,None],'min_samples_leaf':[2,5,10,20],'max_features':['sqrt',.5,1.]},'XGB':{'max_depth':[2,3,4,6,8],'learning_rate':[.02,.05,.1,.2],'subsample':[.7,.85,1.]},'LGB':{'num_leaves':[7,15,31,63],'learning_rate':[.02,.05,.1,.2],'min_child_samples':[10,20,40,80]},'CAT':{'depth':[3,4,5,6,8],'learning_rate':[.02,.05,.1,.2],'l2_leaf_reg':[1,3,5,10]}}
def custom(n,idx,seed,w,p):
    t=C['threads']; k=C['n_estimators']; prep=preprocessor(n=='LR');z=prep.fit_transform(X[idx]); common={'random_state':seed,'n_jobs':t}; defaults={'LR':(LogisticRegression,dict(max_iter=1000)), 'RF':(RandomForestClassifier,dict(n_estimators=k,**common)), 'XGB':(XGBClassifier,dict(n_estimators=k,eval_metric='logloss',colsample_bytree=.85,**common)), 'LGB':(LGBMClassifier,dict(n_estimators=k,verbosity=-1,**common)), 'CAT':(CatBoostClassifier,dict(iterations=k,thread_count=t,random_seed=seed,verbose=False,allow_writing_files=False))}; cls,kw=defaults[n];kw.update(p);m=cls(**kw);m.fit(z,y[idx],sample_weight=compute_sample_weight('balanced',y[idx]) if w else np.ones(len(idx)));return FittedBase(prep,m)
def tune(n,idx,seed,w):
    candidates=list(ParameterSampler(SPACE[n],n_iter=2 if args.smoke else S['trials'],random_state=seed)); scores=[]
    for p in candidates:scores.append(np.mean([average_precision_score(y[b],predict(custom(n,a,seed,w,p),X[b])) for a,b in cv(idx,seed)]))
    j=int(np.argmax(scores));return custom(n,idx,seed,w,candidates[j]),candidates[j],scores[j]
def s1():
    allrows=[]
    for seed in ([42] if args.smoke else S['tuning_seeds']):
        tr,ca,te=partition(seed)
        for w in [False,True]:
            def work():
                rows=[]; models={}; oof=np.zeros((len(tr),3)); loc={v:i for i,v in enumerate(tr)}
                for n in BASE:
                    m,p,score=tune(n,tr,seed,w);models[n]=m;rows.append(dict(seed=seed,weighted=w,model=n,params=json.dumps(p,default=lambda x:x.item()),cv_AP=score,**evaluate(m,ca,te)))
                for a,b in cv(tr,seed):
                    for j,n in enumerate(BOOST):
                        m,_,_=tune(n,a,seed,w);oof[[loc[v] for v in b],j]=predict(m,X[b])
                assert np.all(oof>0)
                base=[models[n] for n in BOOST];meta=LogisticRegression(C=1,max_iter=1000).fit(oof,y[tr]);
                for n,m in [('VOTE',Ensemble(base)),('STACK',Ensemble(base,meta))]:rows.append(dict(seed=seed,weighted=w,model=n,**evaluate(m,ca,te)))
                return rows
            allrows.append(cached(f'seed{seed}_w{int(w)}',work));print('S1',seed,w,flush=True)
    save('summary',pd.concat(allrows,ignore_index=True))
def s2():
    out=[]
    for seed in ([42] if args.smoke else C['split_seeds']):
        tr,ca,te=partition(seed)
        def work():
            rows=[]
            for w in [False,True]:
                base=[fit_base(n,X[tr],y[tr],seed,C,w) for n in BOOST];oof=np.zeros((len(tr),3));loc={v:i for i,v in enumerate(tr)}
                for a,b in cv(tr,seed):
                    for j,n in enumerate(BOOST):oof[[loc[v] for v in b],j]=predict(fit_base(n,X[a],y[a],seed,C,w),X[b])
                for mw in [False,True]:
                    m=Ensemble(base,LogisticRegression(C=1,max_iter=1000).fit(oof,y[tr],sample_weight=compute_sample_weight('balanced',y[tr]) if mw else None))
                    for kind in ['RAW','PLATT','ISO']:
                        cm=m if kind=='RAW' else Calibrated(m,kind,X[ca],y[ca]);rows.append(dict(seed=seed,base_weighted=w,meta_weighted=mw,calibration=kind,**evaluate(cm,ca,te)))
            return rows
        out.append(cached(f'seed{seed}',work));print('S2',seed,flush=True)
    save('summary',pd.concat(out,ignore_index=True))
def bootmodels(tr,ca,b,arr):
    idx=indices(arr[b]) if not args.smoke else rng.choice(tr,len(tr),replace=True);m=fit_models(X[idx],y[idx],g[idx],10042+b,C,['STACK_W'])['STACK_W'];return [m,Calibrated(m,'PLATT',X[ca],y[ca]),Calibrated(m,'ISO',X[ca],y[ca])]
def s3():
    tr,ca,te=partition(42); manifest=pd.read_csv(R/'explanation_manifest.csv'); local=manifest[manifest.local]; selected=local.groupby('stratum',group_keys=False).sample(n=2 if args.smoke else 10,random_state=42); ii=indices(selected.ID); xx=X[ii]; bg0=np.load(R/'shap_background.npy'); eligible=tr[~np.any(np.all(X[tr,None,:]==bg0[None,:,:],axis=2),axis=1)]; extra=X[rng.choice(eligible,48,replace=False)];bg=np.concatenate([bg0,extra]); arr=np.load(R/'bootstrap_ids.npy'); budgets=list(itertools.product([3,4] if args.smoke else [16,32,64],[1,2] if args.smoke else [4,8,16])); B=2 if args.smoke else S['shap_bootstraps']; names=['RAW','PLATT','ISO']; times=[]
    for b in range(B):
        p=O/f'boot{b}.npz';
        if p.exists() and not args.force:continue
        models=bootmodels(tr,ca,b,arr); data={};
        for j,m in enumerate(models):
            for z,c in budgets:
                st=time.perf_counter(); key=f'{names[j]}_{z}_{c}';data[key]=permutation_shap(m,xx,bg[:z],c,42);data['time_'+key]=time.perf_counter()-st
        tmp=O/f'boot{b}.tmp.npz';np.savez_compressed(tmp,**data);tmp.replace(p);print('S3 bootstrap',b,flush=True)
    files=[np.load(O/f'boot{b}.npz') for b in range(B)];rows=[]; baseline=budgets[0]
    for n in names:
        ref=np.stack([f[f'{n}_{baseline[0]}_{baseline[1]}'] for f in files]);
        for z,c in budgets:
            key=f'{n}_{z}_{c}';phi=np.stack([f[key] for f in files]);sec=sum(float(f['time_'+key]) for f in files)
            for j,r in enumerate(stability(phi,5)):rows.append(dict(ID=int(ids[ii[j]]),stratum=selected.stratum.iloc[j],model=n,background=z,cycles=c,seconds_total=sec,mean_abs_delta=float(np.abs(phi[:,j]-ref[:,j]).mean()),**r))
    save('summary',rows)
def s4():
    tr,ca,te=partition(42);arr=np.load(R/'bootstrap_ids.npy');ref=Calibrated(fit_models(X[tr],y[tr],g[tr],42,C,['STACK_W'])['STACK_W'],'PLATT',X[ca],y[ca]); tau=threshold(y[ca],predict(ref,X[ca])) if args.smoke else json.loads((R/'complete.json').read_text())['threshold']; pp=predict(ref,X[te]);
    if not args.smoke:
        old=pd.read_csv(R/'reference_predictions.csv');old=old[(old.model=='STACK_W')&(old.calibration=='PLATT')].set_index('ID').loc[ids[te],'p'].to_numpy(); error=float(np.max(np.abs(pp-old))); print('Reference prediction max discrepancy:',error,flush=True)
        if error>1e-5:raise RuntimeError('Reference predictions differ: check environment and original split order')
    dist=pp-tau; manifest=pd.read_csv(R/'explanation_manifest.csv'); original=indices(manifest.loc[manifest.counterfactual,'ID']); original=original[:4] if args.smoke else original; cohorts={'original':original};
    for lo,hi in [(0,.02),(.02,.05),(.05,.15),(.15,np.inf)]:
        q=te[(dist>=lo)&(dist<hi)];cohorts[str(lo)]=rng.choice(q,min(len(q),3 if args.smoke else 100),replace=False)
    ii=np.unique(np.concatenate(list(cohorts.values()))); xx=X[ii]; bg=np.load(R/'shap_background.npy')[:3 if args.smoke else 16]; B,H=(2,2) if args.smoke else (30,10); lo,hi=np.quantile(X[tr,0],[.01,.99]); candidates=[]; meta=[]
    for j,i in enumerate(ii):
        for radius in [.1,.2,.3,.5]:
            lim=X[i,0]; v=np.unique(np.round(lim*np.linspace(1-radius,1+radius,25)/1000)*1000);v=v[(v>=lo)&(v<=hi)&(v>=lim*(1-radius))&(v<=lim*(1+radius))&(v!=lim)]; a=np.repeat(X[i:i+1],len(v),axis=0);a[:,0]=v;meta.append((j,radius,len(candidates),len(candidates)+len(v)));candidates.extend(a)
    cand=np.asarray(candidates).reshape(-1,X.shape[1]); rp=predict(ref,cand); orig=predict(ref,xx); boot=[]; ph=[]
    for b in range(B+H):
        p=O/f'boot{b}.npz'
        if not p.exists() or args.force:
            m=bootmodels(tr,ca,b,arr)[1];kw={'pred':predict(m,cand),'orig':predict(m,xx)}
            if b<B:kw['phi']=permutation_shap(m,xx,bg,1 if args.smoke else 4,42)
            tmp=O/f'boot{b}.tmp.npz';np.savez_compressed(tmp,**kw);tmp.replace(p)
        with np.load(p) as f:
            boot.append((f['pred'],f['orig']));
            if b<B:ph.append(f['phi'])
        print('S4 bootstrap',b,flush=True)
    bp=np.stack([a for a,b in boot]);bo=np.stack([b for a,b in boot]);ag=aggregate(np.stack(ph),5);rows=[];dev=(bp[:B]<tau).mean(axis=0)
    for j,r,a,b in meta:
        i=ii[j]; valid=np.arange(a,b)[rp[a:b]<tau]; reduction=np.arange(a,b)[rp[a:b]<orig[j]]; d=np.abs(cand[a:b,0]-X[i,0])/X[i,0]; level=ag['level'][j,0];
        for method in ['Nearest','Robust','HardGate','ModerateGate','RiskReduction']:
            pool=reduction if method=='RiskReduction' else valid; allowed=method!='HardGate' or level=='High';allowed=allowed and (method!='ModerateGate' or level!='Low'); k=None
            if len(pool) and allowed:
                k=min(pool,key=lambda k:(d[k-a],rp[k])) if method=='Nearest' else min(pool,key=lambda k:(rp[k],d[k-a])) if method=='RiskReduction' else min(pool,key=lambda k:(-dev[k],d[k-a],rp[k]))
            for cohort, members in cohorts.items():
                if i not in members:continue
                rows.append(dict(ID=int(ids[i]),cohort=cohort,radius=r,method=method,level=level,original_p=orig[j],threshold=tau,coverage=int(k is not None),new_limit=cand[k,0] if k is not None else np.nan,new_p=rp[k] if k is not None else np.nan,distance=d[k-a] if k is not None else np.nan,risk_reduction=orig[j]-rp[k] if k is not None else 0,holdout_crossing=float((bp[B:,k]<tau).mean()) if k is not None else 0,holdout_reduction=float((bp[B:,k]<bo[B:,j]).mean()) if k is not None else 0))
    save('summary',rows)
def holm(p):
    p=np.array(p);order=np.argsort(p);v=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))));out=np.empty(len(p));out[order]=np.minimum(v,1);return out
def s5():
    pred=pd.read_csv(R/'reference_predictions.csv'); pred['key']=pred.model+'_'+pred.calibration; wide=pred.pivot(index='ID',columns='key',values='p'); yy=y[indices(wide.index)]; gg=g[indices(wide.index)]; groups=list(pd.Series(np.arange(len(gg))).groupby(gg,sort=True).apply(lambda v:v.to_numpy())); reps=20 if args.smoke else S['inference_bootstrap']; draws=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(reps)]; fun={'AP':average_precision_score,'AUC':roc_auc_score,'Brier':brier_score_loss,'LogLoss':log_loss,'ECE':lambda y,p:sum(b['n']*abs(b['pred']-b['observed']) for b in calibration_bins(y,p))/len(y)}; contrasts=pd.read_csv(R/'paired_effects.csv')[['rq','metric','A','B']].to_dict('records'); contrasts += [dict(rq='RQ2',metric='ECE',A=n+'_'+k,B=n+'_RAW') for n in ['STACK_W','STACK','XGB_W'] for k in ['PLATT','ISO']];rows=[]
    for c in contrasts:
        a,b=wide[c['A']].to_numpy(),wide[c['B']].to_numpy(); f=fun[c['metric']]; delta=f(yy,a)-f(yy,b); vals=np.array([f(yy[z],a[z])-f(yy[z],b[z]) for z in draws]); p=(1+np.sum(np.abs(vals-delta)>=abs(delta)))/(reps+1);rows.append(dict(**c,delta=delta,ci_low=np.quantile(vals,.025),ci_high=np.quantile(vals,.975),p=p));print('S5',c,flush=True)
    df=pd.DataFrame(rows);df['holm_p']=df.groupby('rq').p.transform(holm);save('summary',df)
def s6():
    df=pd.read_csv(R/'rq3_stability.csv');df=df[(df.model=='STACK_W_PLATT')&(df.k==5)];rows=[];reps=20 if args.smoke else S['inference_bootstrap'];
    for a,b in itertools.combinations(['TP','TN','FP','FN'],2):
        for m in ['Overlap','Rank','Direction','NormalizedSD']:
            x=df.loc[df.stratum==a,m].dropna().to_numpy();z=df.loc[df.stratum==b,m].dropna().to_numpy(); u,p=mannwhitneyu(x,z,alternative='two-sided'); vals=[rng.choice(x,len(x)).mean()-rng.choice(z,len(z)).mean() for _ in range(reps)];rows.append(dict(A=a,B=b,metric=m,n_A=len(x),n_B=len(z),delta=x.mean()-z.mean(),ci_low=np.quantile(vals,.025),ci_high=np.quantile(vals,.975),cliffs_delta=2*u/(len(x)*len(z))-1,p=p))
    df=pd.DataFrame(rows);df['holm_p']=holm(df.p);save('summary',df)
if __name__=='__main__':
    marker=O/'complete.json'; fingerprint=hashlib.sha256((ROOT/'supplementary/config.yaml').read_bytes()+(ROOT/'config.yaml').read_bytes() + Path(__file__).read_bytes()+(R/'audit.json').read_bytes()).hexdigest(); lock=O/'fingerprint.txt'; env={'python':platform.python_version(),'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','catboost']}}; h=hashlib.sha256(fingerprint.encode()+json.dumps(env,sort_keys=True).encode());
    for path in [ROOT/'src/models.py',ROOT/'src/evaluation.py',ROOT/'src/explain.py',R/'split_manifest.csv',R/'bootstrap_ids.npy',R/'shap_background.npy',R/'explanation_manifest.csv',R/'reference_predictions.csv',R/'rq3_stability.csv',R/'paired_effects.csv',R/'complete.json']:h.update(path.read_bytes())
    fingerprint=h.hexdigest()
    if lock.exists() and lock.read_text()!=fingerprint and not args.force:raise SystemExit('Configuration/code changed: use --force or archive this stage directory first')
    lock.write_text(fingerprint); (O/'environment.json').write_text(json.dumps(env,indent=2))
    if marker.exists() and not args.force:print('Already complete',O);sys.exit(0)
    marker.unlink(missing_ok=True);globals()[args.stage]();marker.write_text(json.dumps({'stage':args.stage,'smoke':args.smoke,'status':'complete','fingerprint':fingerprint}));print('Complete',O)
