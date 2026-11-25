from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve,precision_recall_curve

def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return out

def paired(a,b,rng,reps):
    d=np.asarray(a)-np.asarray(b);d=d[np.isfinite(d)]
    if not len(d):return dict(n=0,delta=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan)
    boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(reps)])
    return dict(n=len(d),delta=d.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),p=wilcoxon(d).pvalue if np.any(d!=0) else 1.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');a=ap.parse_args();r=Path(a.results);assert (r/'complete.json').exists(),'Run all experiments first';audit=json.loads((r/'audit.json').read_text());rng=np.random.default_rng(42);reps=audit['config']['inference_bootstrap'];t=r/'tables';f=r/'figures';t.mkdir(exist_ok=True);f.mkdir(exist_ok=True)
    def read(n):return pd.read_csv(r/(n+'.csv'))
    def save(n):plt.tight_layout();plt.savefig(f/(n+'.png'),dpi=300,bbox_inches='tight');plt.savefig(f/(n+'.pdf'),bbox_inches='tight');plt.close()
    def group(df,keys,cols,name):df.groupby(keys)[cols].agg(['mean','std','count']).to_csv(t/(name+'.csv'))
    p=read('rq1_metrics');c=read('rq2_calibration');s=read('rq3_stability');cf=read('rq3_counterfactual');g=read('rq3_global');b=read('rq2_reliability_bins');v=read('rq3_stability_validation');rob=read('rq2_missing_robustness');pred=read('reference_predictions');attrs=read('rq3_attributions');
    group(p,['model'],['AP','AUC','F1','MCC','Precision','Recall','BalancedAccuracy'],'T1_performance');group(c,['model','calibration'],['Brier','LogLoss','ECE','AUC','AP','CalibrationSlope','CalibrationIntercept'],'T2_calibration');group(s,['model','k','stratum'],['Overlap','Rank','Direction','DirectionCoverage','NormalizedSD'],'T3_stability');group(cf,['method'],['coverage','validity','feasible','robust_validity','conditional_robust_validity','relative_distance'],'T4_counterfactual');group(v,['development_level'],['holdout_frequency','holdout_direction'],'T5_stability_validation');group(rob,['model','missing_rate'],['AP','Brier','ECE','AUC'],'T6_missing_robustness');group(read('runtime'),['model','stage'],['seconds'],'T7_runtime');group(read('rq2_bootstrap_risk'),['phase','calibration'],['AP','AUC','Brier','ECE','LogLoss'],'T8_bootstrap_risk');
    stat=[];ss=s[s.k==5]
    for A,B in [('STACK_W','XGB_W'),('STACK_W','STACK'),('STACK_W_PLATT','STACK_W'),('STACK_W_ISO','STACK_W'),('STACK_W_PLATT','STACK_W_ISO')]:
        for m in ['Overlap','Rank','Direction','NormalizedSD']:
            x=ss.pivot(index='ID',columns='model',values=m)
            if A in x and B in x:stat.append(dict(family='stability',A=A,B=B,metric=m,**paired(x[A],x[B],rng,reps)))
    for A,B in [('Robust','Nearest'),('StabilityGatedRobust','Robust')]:
        for m in ['coverage','robust_validity']:
            x=cf.pivot(index='ID',columns='method',values=m);stat.append(dict(family='counterfactual',A=A,B=B,metric=m,**paired(x[A],x[B],rng,reps)))
    st=pd.DataFrame(stat);st['p_holm']=np.nan
    for fam,idx in st.groupby('family').groups.items():
        ix=st.loc[idx].dropna(subset=['p']).index;st.loc[ix,'p_holm']=holm(st.loc[ix,'p'])
    st.to_csv(t/'T9_paired_explanations.csv',index=False)
    # Risk bands: fixed probability bins; output counts and Wilson CIs for observed rate.
    risk=[]
    for (model,cal),q in pred.groupby(['model','calibration']):
        if model!='STACK_W':continue
        for j,(lo,hi) in enumerate(zip([0,.1,.2,.4,.6],[.1,.2,.4,.6,1.000001])):
            z=q[(q.p>=lo)&(q.p<hi)];n=len(z);obs=z.y.mean();den=1+1.96**2/n if n else np.nan;center=(obs+1.96**2/(2*n))/den if n else np.nan;half=1.96*np.sqrt(obs*(1-obs)/n+1.96**2/(4*n*n))/den if n else np.nan;risk.append(dict(model=model,calibration=cal,band=j,n=n,mean_pred=z.p.mean(),observed=obs,ci_low=center-half,ci_high=center+half))
    pd.DataFrame(risk).to_csv(t/'T10_risk_bands.csv',index=False)
    p.groupby('model')[['AP','AUC']].mean().sort_values('AP').plot.barh(figsize=(8,6));plt.xlabel('Score');save('F1_model_performance')
    fig,ax=plt.subplots(1,2,figsize=(10,4));
    for name in ['LR','XGB_W','STACK','STACK_W']:
        q=pred[(pred.model==name)&(pred.calibration=='RAW')];x,y,_=roc_curve(q.y,q.p);ax[0].plot(x,y,label=name);y,x,_=precision_recall_curve(q.y,q.p);ax[1].plot(x,y,label=name)
    ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].set(xlabel='False positive rate',ylabel='True positive rate');ax[1].set(xlabel='Recall',ylabel='Precision');ax[0].legend();ax[1].legend();save('F2_roc_pr')
    plt.figure(figsize=(6,5));plt.plot([0,1],[0,1],'--',color='gray');
    for name,q in b[b.model=='STACK_W'].groupby('calibration'):plt.plot(q.pred,q.observed,'o-',label=name)
    plt.xlabel('Mean predicted probability');plt.ylabel('Observed default frequency');plt.legend();save('F3_calibration')
    rob.groupby(['missing_rate','model']).Brier.mean().unstack().plot(marker='o',figsize=(8,4));plt.ylabel('Brier score');plt.xlabel('Missing fraction');save('F4_robustness')
    q=g[g.model=='STACK_W_PLATT'].nlargest(12,'mean').sort_values('mean');plt.figure(figsize=(7,5));plt.barh(q.feature,q['mean']);plt.errorbar(q['mean'],q.feature,xerr=np.array([np.maximum(0,q['mean']-q.low),np.maximum(0,q.high-q['mean'])]),fmt='none',color='black',capsize=2);plt.xlabel('Mean |SHAP| (probability scale)');save('F5_global_shap')
    ss.boxplot(column='Overlap',by='model',rot=25,figsize=(9,5));plt.suptitle('');plt.title('Top-5 stability');plt.ylabel('Overlap');save('F6_stability')
    cf.groupby('method')[['coverage','robust_validity','conditional_robust_validity']].mean().plot.bar(figsize=(9,5),rot=15);plt.ylim(0,1);save('F7_counterfactual')
    q=attrs[attrs.model=='STACK_W_PLATT'];q=q[q.ID==q.ID.iloc[0]].copy();q=q.loc[q['mean'].abs().nlargest(10).index].sort_values('mean');plt.figure(figsize=(7,5));plt.barh(q.feature,q['mean']);plt.errorbar(q['mean'],q.feature,xerr=[np.maximum(0,q['mean']-q.low),np.maximum(0,q.high-q['mean'])],fmt='none',color='black');plt.axvline(0,color='gray');plt.xlabel('Mean local SHAP');save('F8_local_shap')
    print('Tables and figures saved in',r)
if __name__=='__main__':main()
