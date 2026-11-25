from pathlib import Path
import argparse, json, time, warnings, hashlib, platform, importlib.metadata
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from src.models import COLUMNS,NAMES,fit_models,predict,Calibrated
from src.evaluation import metrics,threshold,calibration_bins,paired_ci,stability
from src.explain import permutation_shap,aggregate
warnings.filterwarnings('ignore',message='X does not have valid feature names')
ROOT=Path(__file__).resolve().parent

def write(rows,path): pd.DataFrame(rows).to_csv(path,index=False)
def split(X,y,g,seed):
    folds=[v for _,v in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,g)]
    return np.concatenate(folds[:3]),folds[3],folds[4]
def load(smoke):
    df=pd.read_csv(ROOT/'data/credit.csv'); assert df.ID.is_unique and set(df.default)=={0,1}
    if smoke: df=df.sample(n=1500,random_state=42).reset_index(drop=True)
    X=df[COLUMNS].astype(float).to_numpy(); y=df.default.to_numpy(); g=pd.util.hash_pandas_object(df[COLUMNS],index=False).to_numpy();return df,X,y,g

def choose(rng,idx,n): return rng.choice(idx,min(n,len(idx)),replace=False) if len(idx) else np.array([],dtype=int)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default=None);ap.add_argument('--config',default=str(ROOT/'config.yaml'));args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());
    if args.smoke: cfg.update(split_seeds=[42],folds=3,threads=1,n_estimators=8,bootstrap_models=3,holdout_models=2,background_size=3,permutation_cycles=1,local_per_stratum=2,global_samples=5,counterfactual_samples=4,inference_bootstrap=20)
    assert cfg['bootstrap_models']>=2 and cfg['holdout_models']>=1 and cfg['seed'] in cfg['split_seeds']
    out=Path(args.output or ROOT/('results_smoke' if args.smoke else 'results'));out.mkdir(parents=True,exist_ok=True)
    df,X,y,g=load(args.smoke); rng=np.random.default_rng(cfg['seed']); (out/'audit.json').write_text(json.dumps(dict(n=len(y),features=len(COLUMNS),positive=int(y.sum()),negative=int((1-y).sum()),missing=int(np.isnan(X).sum()),duplicate_predictor_rows=int(pd.Series(g).duplicated().sum()),data_sha256=hashlib.sha256((ROOT/'data/credit.csv').read_bytes()).hexdigest(),smoke=args.smoke,config=cfg,python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','xgboost','lightgbm','catboost','scipy']}),indent=2))
    perf=[]; calrows=[]; bins=[]; preds=[]; stats=[]; robust=[]; splits=[]; timing=[]
    for seed in cfg['split_seeds']:
        tr,ca,te=split(X,y,g,seed); assert not(set(g[tr])&set(g[ca]) or set(g[tr])&set(g[te]) or set(g[ca])&set(g[te]));splits.extend(dict(seed=seed,ID=int(df.ID.iloc[i]),partition=k) for k,idx in [('train',tr),('calibration',ca),('test',te)] for i in idx)
        start=time.perf_counter(); models=fit_models(X[tr],y[tr],g[tr],seed,cfg); timing.append(dict(seed=seed,model='ALL_14_SHARED_FITS',stage='training',seconds=time.perf_counter()-start))
        allcal={}; prob={}
        for name,m in models.items():
            st=time.perf_counter();p=predict(m,X[te]); timing.append(dict(seed=seed,model=name,stage='inference',seconds=time.perf_counter()-st,n=len(te))); t=threshold(y[ca],predict(m,X[ca]));perf.append(dict(seed=seed,model=name,threshold=t,**metrics(y[te],p,t),**{k+"_at_05":v for k,v in metrics(y[te],p,.5,False).items() if k in ["F1","MCC","Precision","Recall","BalancedAccuracy"]}))
            for kind in ['RAW','PLATT','ISO']:
                cm=m if kind=='RAW' else Calibrated(m,kind,X[ca],y[ca]);key=name+'_'+kind;allcal[key]=cm;pc=predict(cm,X[te]);tc=threshold(y[ca],predict(cm,X[ca]));prob[key]=pc;calrows.append(dict(seed=seed,model=name,calibration=kind,threshold=tc,**metrics(y[te],pc,tc)))
                if seed==cfg['seed']:
                    preds.extend(dict(ID=int(df.ID.iloc[i]),y=int(y[i]),model=name,calibration=kind,p=float(v)) for i,v in zip(te,pc));bins.extend(dict(model=name,calibration=kind,**b) for b in calibration_bins(y[te],pc))
        if seed==cfg['seed']:
            ref=(tr,ca,te,models,allcal,prob); base='STACK_W_RAW'
            for other in ['LR_W_RAW','RF_W_RAW','XGB_W_RAW','LGB_W_RAW','CAT_W_RAW','STACK_RAW','VOTE_W_RAW']:
                for metric in ['AP','AUC']:
                    d,l,h=paired_ci(y[te],prob[base],prob[other],metric,cfg['inference_bootstrap'],seed);stats.append(dict(rq='RQ1',metric=metric,A=base,B=other,delta=d,ci_low=l,ci_high=h))
            for name in ['STACK_W','STACK','XGB_W']:
                for kind in ['PLATT','ISO']:
                    for metric in ['Brier','LogLoss']:
                        a=name+'_'+kind;b=name+'_RAW';d,l,h=paired_ci(y[te],prob[a],prob[b],metric,cfg['inference_bootstrap'],seed);stats.append(dict(rq='RQ2',metric=metric,A=a,B=b,delta=d,ci_low=l,ci_high=h))
        for rate in [0]+cfg['missing_rates']:
            mask=np.random.default_rng(seed+int(rate*1000)).random(X[te].shape)<rate;xx=X[te].copy();xx[mask]=np.nan
            for key in ['STACK_W_RAW','STACK_W_PLATT','STACK_W_ISO','STACK_RAW','XGB_W_RAW']:
                m=allcal[key];t=threshold(y[ca],predict(m,X[ca]));robust.append(dict(seed=seed,model=key,missing_rate=rate,**metrics(y[te],predict(m,xx),t,False)))
        print('Completed RQ1/RQ2 split',seed,flush=True)
    for rows,name in [(perf,'rq1_metrics'),(calrows,'rq2_calibration'),(bins,'rq2_reliability_bins'),(preds,'reference_predictions'),(stats,'paired_effects'),(robust,'rq2_missing_robustness'),(splits,'split_manifest'),(timing,'runtime')]:write(rows,out/(name+'.csv'))
    tr,ca,te,models,allcal,prob=ref; reference=allcal['STACK_W_PLATT'];tau=threshold(y[ca],predict(reference,X[ca]));pred=prob['STACK_W_PLATT']>=tau
    strata=np.where(y[te]==1,np.where(pred,'TP','FN'),np.where(pred,'FP','TN')); local=np.concatenate([choose(rng,te[strata==s],cfg['local_per_stratum']) for s in ['TP','TN','FP','FN']]); glob=choose(rng,te,cfg['global_samples']); cfidx=choose(rng,te[pred],cfg['counterfactual_samples']); indices=np.unique(np.concatenate([local,glob,cfidx])); pos={v:i for i,v in enumerate(indices)}; lp=[pos[i] for i in local];gp=[pos[i] for i in glob];cp=[pos[i] for i in cfidx];bg=X[choose(rng,tr,cfg['background_size'])];xx=X[indices]
    write([dict(ID=int(df.ID.iloc[i]),local=i in set(local),global_sample=i in set(glob),counterfactual=i in set(cfidx),stratum=strata[np.where(te==i)[0][0]]) for i in indices],out/'explanation_manifest.csv'); np.save(out/'shap_background.npy',bg)
    # Pilot checks approximation sensitivity for a fixed reference predictor and test subset.
    pilot=xx[:min(10,len(xx))]; sensitivity=[]; refphi=None
    for cyc in [cfg['permutation_cycles'],2*cfg['permutation_cycles'],4*cfg['permutation_cycles']]:
        st=time.perf_counter();v=permutation_shap(reference,pilot,bg,cyc,cfg['seed']);
        if refphi is None: refphi=v
        sensitivity.append(dict(cycles=cyc,n=len(pilot),mean_abs_delta_from_primary=float(np.mean(np.abs(v-refphi))),mean_top5_overlap=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(np.argsort(-np.abs(v),axis=1)[:,:5],np.argsort(-np.abs(refphi),axis=1)[:,:5])])),seconds=time.perf_counter()-st))
    write(sensitivity,out/'rq3_shap_budget_sensitivity.csv')
    names=cfg['shap_configs'];phis={n:[] for n in names}; held=[];trainboot=[];bootrisk=[]; boot_indices=[]; shap_runtime=[]; B=cfg['bootstrap_models'];H=cfg['holdout_models']
    for b in range(B+H):
        rr=np.random.default_rng(cfg['seed']+10000+b); idx=rr.choice(tr,len(tr),replace=True);boot_indices.append(df.ID.iloc[idx].to_numpy()); needed=['STACK_W'] if b>=B else sorted(set(n.replace('_PLATT','').replace('_ISO','') for n in names)); ms=fit_models(X[idx],y[idx],g[idx],cfg['seed']+10000+b,cfg,needed); cm=Calibrated(ms['STACK_W'],'PLATT',X[ca],y[ca]); (held if b>=B else trainboot).append(cm)
        for bootkind,bm in [('RAW',ms['STACK_W']),('PLATT',cm),('ISO',Calibrated(ms['STACK_W'],'ISO',X[ca],y[ca]))]:
            bootrisk.append(dict(replicate=b,phase='holdout' if b>=B else 'development',calibration=bootkind,**metrics(y[te],predict(bm,X[te]),tau,False)))
        for n in (['STACK_W_PLATT'] if b>=B else names):
            m=cm if n=='STACK_W_PLATT' else Calibrated(ms['STACK_W'],'ISO',X[ca],y[ca]) if n=='STACK_W_ISO' else ms[n];st=time.perf_counter();phi=permutation_shap(m,xx,bg,cfg['permutation_cycles'],cfg['seed']);phis[n].append(phi);shap_runtime.append(dict(replicate=b,model=n,seconds=time.perf_counter()-st,n=len(xx)))
        print('Completed RQ3 bootstrap',b+1,'/',B+H,flush=True)
    np.save(out/'bootstrap_ids.npy',np.asarray(boot_indices));write(bootrisk,out/'rq2_bootstrap_risk.csv');write(shap_runtime,out/'shap_runtime.csv')
    strows=[];aggrows=[];globalrows=[];validation=[]; arrays={}
    for name,arr in phis.items():
        full=np.stack(arr); a=full[:B];arrays[name]=full; ag=aggregate(a,cfg['top_k']);
        for k in [3,5,10]:
            for i,r in zip(local,stability(a[:,lp],k)):strows.append(dict(model=name,k=k,ID=int(df.ID.iloc[i]),stratum=strata[np.where(te==i)[0][0]],**r))
        for j,i in enumerate(indices):
            for f,col in enumerate(COLUMNS):aggrows.append(dict(model=name,ID=int(df.ID.iloc[i]),feature=col,**{key:v[j,f] for key,v in ag.items()}))
        imp=np.abs(a[:,gp]).mean(axis=1)
        for f,col in enumerate(COLUMNS):globalrows.append(dict(model=name,feature=col,mean=float(imp[:,f].mean()),low=float(np.quantile(imp[:,f],.025)),high=float(np.quantile(imp[:,f],.975))))
        if name=='STACK_W_PLATT':
            hold=aggregate(full[B:],cfg['top_k']);
            for j,i in enumerate(indices):
                for f,col in enumerate(COLUMNS):validation.append(dict(ID=int(df.ID.iloc[i]),feature=col,development_level=ag['level'][j,f],holdout_frequency=hold['frequency'][j,f],holdout_direction=hold['direction'][j,f]))
    np.savez_compressed(out/'shap_values.npz',**arrays);write(strows,out/'rq3_stability.csv');write(aggrows,out/'rq3_attributions.csv');write(globalrows,out/'rq3_global.csv');write(validation,out/'rq3_stability_validation.csv')
    # Only credit limit may vary; historical records/demographics are frozen. Hypothetical lender scenarios, not borrower recourse.
    agg=aggregate(arrays['STACK_W_PLATT'][:B],cfg['top_k']);cfr=[];lo,hi=np.quantile(X[tr,0],[.01,.99]);
    for i,j in zip(cfidx,cp):
        x=X[i]; lim=x[0]; values=np.unique(np.round(lim*np.linspace(.7,1.3,25)/1000)*1000);values=values[(values>=lo)&(values<=hi)&(values>=.7*lim)&(values<=1.3*lim)&(values!=lim)];cand=np.repeat(x[None,:],len(values),axis=0);cand[:,0]=values
        if len(cand):
            p=predict(reference,cand);valid=np.flatnonzero(p<tau);dev=np.mean([predict(m,cand)<tau for m in trainboot],axis=0);dist=np.abs(values-lim)/lim; holdpred=np.array([predict(m,cand) for m in held]);
        else: valid=[]
        for method in ['Nearest','Robust','StabilityGatedRobust']:
            gate=agg['level'][j,0]=='High'; ok=len(valid)>0 and (method!='StabilityGatedRobust' or gate);row=dict(ID=int(df.ID.iloc[i]),method=method,original_limit=lim,original_p=float(predict(reference,x[None,:])[0]),threshold=tau,gate=bool(gate),coverage=int(ok),validity=int(ok),feasible=int(ok),robust_validity=0.,conditional_robust_validity=np.nan,new_limit=np.nan,new_p=np.nan,relative_distance=np.nan,development_validity=np.nan)
            if ok:
                k=min(valid,key=lambda z:(dist[z],p[z])) if method=='Nearest' else min(valid,key=lambda z:(-dev[z],dist[z],p[z])); rv=float(np.mean(holdpred[:,k]<tau));row.update(new_limit=values[k],new_p=p[k],relative_distance=dist[k],development_validity=dev[k],robust_validity=rv,conditional_robust_validity=rv)
            cfr.append(row)
    write(cfr,out/'rq3_counterfactual.csv');(out/'complete.json').write_text(json.dumps({'status':'complete','smoke':args.smoke,'threshold':tau,'local_n':len(local),'global_n':len(glob),'cf_n':len(cfidx)},indent=2)); print('All experiments completed:',out,flush=True)
if __name__=='__main__':main()
