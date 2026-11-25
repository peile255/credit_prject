from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def cf_contrasts(d, out, root, smoke):
    from scipy.stats import binomtest
    source=pd.read_csv(Path(__file__).resolve().parents[1]/'data/credit.csv'); groups=pd.Series(pd.util.hash_pandas_object(source.drop(columns=['ID','default']),index=False).to_numpy(),index=source.ID)
    rng=np.random.default_rng(42); rows=[]; reps=20 if smoke else 2000
    for (cohort,radius),part in d.groupby(['cohort','radius']):
        for A,B in [('Robust','Nearest'),('HardGate','Robust'),('ModerateGate','Robust')]:
            for metric in ['coverage','holdout_crossing']:
                w=part.pivot(index='ID',columns='method',values=metric); dif=(w[A]-w[B]).to_numpy(); gg=groups.loc[w.index].to_numpy(); chunks=[np.flatnonzero(gg==g) for g in np.unique(gg)]; sums=np.array([dif[c].sum() for c in chunks]); sizes=np.array([len(c) for c in chunks]); draws=rng.integers(len(chunks),size=(reps,len(chunks))); vals=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1); delta=dif.mean(); p=(1+np.sum(np.abs(vals-delta)>=abs(delta)))/(reps+1); rows.append(dict(cohort=cohort,radius=radius,A=A,B=B,metric=metric,n=len(dif),delta=delta,ci_low=np.quantile(vals,.025),ci_high=np.quantile(vals,.975),p=p))
    f=pd.DataFrame(rows); order=np.argsort(f.p.to_numpy()); adjusted=np.empty(len(f));adjusted[order]=np.minimum(1,np.maximum.accumulate(f.p.to_numpy()[order]*(len(f)-np.arange(len(f)))));f['holm_p']=adjusted;f.to_csv(out/'S4_paired_contrasts.csv',index=False)
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]/('results_supplementary_smoke' if a.smoke else 'results_supplementary'); out=root/'report';out.mkdir(parents=True,exist_ok=True)
for stage in ['s1','s2','s3','s4','s5','s6']:
    if not (root/stage/'complete.json').exists():raise SystemExit(f'Missing completed stage {stage}')
    d=pd.read_csv(root/stage/'summary.csv');
    if stage=='s1':
        t=d.groupby(['model','weighted'])[['AP','AUC','F1','MCC']].agg(['mean','std']); t.to_csv(out/'S1_tuned.csv'); plot=d.groupby(['model','weighted']).AP.mean().unstack();ax=plot.plot.bar(ylabel='Average precision',title='Tuned model discrimination')
    elif stage=='s2':
        t=d.groupby(['base_weighted','meta_weighted','calibration'])[['AP','AUC','Brier','LogLoss','ECE']].agg(['mean','std']);t.to_csv(out/'S2_meta.csv');ax=d.groupby(['base_weighted','meta_weighted','calibration']).Brier.mean().unstack().plot.bar(ylabel='Brier score',title='Meta learner weighting and calibration')
    elif stage=='s3':
        t=d.groupby(['model','background','cycles'])[['Overlap','Rank','Direction','NormalizedSD','mean_abs_delta','seconds_total']].mean();t.to_csv(out/'S3_budget.csv');ax=t.Overlap.unstack('model').plot.bar(ylabel='Top-5 overlap',title='SHAP budget sensitivity')
    elif stage=='s4':
        d['conditional_crossing']=d.holdout_crossing.where(d.coverage==1);d['conditional_reduction']=d.holdout_reduction.where(d.coverage==1); t=d.groupby(['cohort','radius','method']).agg(n=('ID','size'),coverage=('coverage','mean'),overall_crossing=('holdout_crossing','mean'),conditional_crossing=('conditional_crossing','mean'),mean_risk_reduction=('risk_reduction','mean'),overall_holdout_reduction=('holdout_reduction','mean'),conditional_reduction=('conditional_reduction','mean'),mean_distance=('distance','mean'));t.to_csv(out/'S4_scenarios.csv');cf_contrasts(d,out,root,a.smoke);ax=t.loc['original'].coverage.unstack().plot(marker='o',ylabel='Output coverage',title='Original cohort: scenario coverage')
    else:
        d.to_csv(out/f'{stage.upper()}_statistics.csv',index=False); fig,ax=plt.subplots(figsize=(10,5)); vals=-np.log10(d.holm_p.clip(lower=1e-15));ax.bar(np.arange(len(d)),vals);ax.axhline(-np.log10(.05),linestyle='--');ax.set(xlabel='Contrast (CSV order)',ylabel='-log10 adjusted p',title=f'{stage.upper()} conditional inference')
    fig=ax.get_figure();fig.tight_layout();fig.savefig(out/f'{stage}.png',dpi=180);fig.savefig(out/f'{stage}.pdf');plt.close(fig)
print('Report saved:',out)
