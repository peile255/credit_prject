import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss,f1_score,matthews_corrcoef,precision_score,recall_score,balanced_accuracy_score,precision_recall_curve
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

def threshold(y,p):
    pr,re,t=precision_recall_curve(y,p); f=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12); return float(t[np.argmax(f)])
def calibration_bins(y,p,n=10):
    bins=np.minimum((p*n).astype(int),n-1); return [{'bin':i,'n':int(sum(bins==i)),'pred':float(p[bins==i].mean()),'observed':float(y[bins==i].mean())} for i in range(n) if (bins==i).any()]
def metrics(y,p,t=.5,extra=True):
    pred=p>=t; bins=calibration_bins(y,p); out=dict(AP=average_precision_score(y,p),AUC=roc_auc_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p,labels=[0,1]),ECE=sum(b['n']*abs(b['pred']-b['observed']) for b in bins)/len(y),F1=f1_score(y,pred,zero_division=0),MCC=matthews_corrcoef(y,pred),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),BalancedAccuracy=balanced_accuracy_score(y,pred))
    if extra:
        m=LogisticRegression(C=1e5,max_iter=1000).fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),y);out.update(CalibrationSlope=m.coef_[0,0],CalibrationIntercept=m.intercept_[0])
    return out

def paired_ci(y,a,b,metric,reps,seed):
    rng=np.random.default_rng(seed); fun={'AP':average_precision_score,'AUC':roc_auc_score,'Brier':brier_score_loss,'LogLoss':log_loss}[metric]; groups=[np.flatnonzero(y==c) for c in [0,1]]; delta=fun(y,a)-fun(y,b); vals=[]
    for _ in range(reps):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);vals.append(fun(y[idx],a[idx])-fun(y[idx],b[idx]))
    return delta,*np.quantile(vals,[.025,.975])

def stability(phi,k):
    # phi shape B,N,F; comparisons first averaged within each instance
    B,N,F=phi.shape; ranks=np.argsort(-np.abs(phi),axis=2,kind='stable'); top=ranks[:,:,:k]; out=[]
    for i in range(N):
        ov=[]; corr=[]; direction=[]
        rank=rankdata(np.abs(phi[:,i]),axis=1);rank=rank-rank.mean(axis=1,keepdims=True);normr=np.linalg.norm(rank,axis=1);den=normr[:,None]*normr[None,:];rc=np.divide(rank@rank.T,den,out=np.full((B,B),np.nan),where=den>0)
        for a in range(B):
            for b in range(a+1,B):
                common=np.intersect1d(top[a,i],top[b,i]); ov.append(len(common)/k)
                corr.append(rc[a,b])
                common=common[(np.abs(phi[a,i,common])>1e-6)&(np.abs(phi[b,i,common])>1e-6)]
                if len(common): direction.append(np.mean(np.sign(phi[a,i,common])==np.sign(phi[b,i,common])))
        norm=np.abs(phi[:,i]).sum(axis=1,keepdims=True); normalized=phi[:,i]/np.maximum(norm,1e-12)
        out.append(dict(Overlap=np.mean(ov),Rank=np.nanmean(corr) if np.isfinite(corr).any() else np.nan,Direction=np.mean(direction) if direction else np.nan,DirectionCoverage=len(direction)/len(ov),NormalizedSD=normalized.std(axis=0,ddof=1).mean()))
    return out
