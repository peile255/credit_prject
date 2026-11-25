import numpy as np
from .models import predict

def permutation_shap(model,X,bg,cycles,seed):
    # Interventional Shapley Monte Carlo: same background and forward/reverse paths for all configurations.
    rng=np.random.default_rng(seed); n,f=X.shape; m=len(bg); out=np.zeros((n,f)); base=predict(model,bg).mean()
    orders=[rng.permutation(f) for _ in range(cycles)]
    for start in range(0,n,8):
        x=X[start:start+8]; q=len(x); accum=np.zeros((q,f))
        for order0 in orders:
            for order in (order0,order0[::-1]):
                cur=np.broadcast_to(bg,(q,m,f)).copy(); states=[cur.copy()]
                for j in order:
                    cur[:,:,j]=x[:,None,j];states.append(cur.copy())
                pred=predict(model,np.concatenate([s.reshape(-1,f) for s in states])).reshape(f+1,q,m).mean(axis=2)
                accum[:,order]+=np.diff(pred,axis=0).T
        out[start:start+q]=accum/(2*cycles)
    err=np.max(np.abs(out.sum(axis=1)+base-predict(model,X))) if n else 0
    if err>1e-5: raise RuntimeError(f'SHAP additivity failure {err}')
    return out

def aggregate(phi,k):
    B,N,F=phi.shape; top=np.argsort(-np.abs(phi),axis=2,kind='stable')[:,:,:k]; freq=np.zeros((N,F))
    for b in range(B): np.add.at(freq,(np.arange(N)[:,None],top[b]),1/B)
    pos=(phi>1e-6).mean(axis=0); neg=(phi< -1e-6).mean(axis=0); dc=np.maximum(pos,neg)
    level=np.where((freq>=.8-1e-9)&(dc>=.9-1e-9),'High',np.where((freq>=.5-1e-9)&(dc>=.7-1e-9),'Moderate','Low'))
    return dict(mean=phi.mean(axis=0),frequency=freq,direction=dc,low=np.quantile(phi,.025,axis=0),high=np.quantile(phi,.975,axis=0),level=level)
