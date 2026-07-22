# -*- coding: utf-8 -*-
"""Predictive-state reclustering and model selection.

Merges microclusters by predictive equivalence and selects the number of categories K by
held-out prediction of the next microcluster, comparing the discovered categories against the
expert labels and a sound-only clustering.

Input: data/microclusters.csv.gz
"""
import os, numpy as np, pandas as pd
np.seterr(all="ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
V=64; ALPHA=0.5; NMIN=30      # microcluster alphabet size, add-alpha smoothing, min order-2 count

d=pd.read_csv(os.path.join(HERE,"data","microclusters.csv.gz")).sort_values(["session_id","order"]).reset_index(drop=True)
micro=d.micro.values.astype(int); N=len(d); sess=d.session_id.values
nextm=d.groupby("session_id")["micro"].shift(-1).values
vn=~np.isnan(nextm); nm=np.where(vn,nextm,0).astype(int)
prev_row=np.arange(N)-1                          # previous syllable in the same recording (-1 if none)
prev_row[np.r_[True, sess[1:]!=sess[:-1]]]=-1
prev_ok=prev_row>=0
tr=(d.split=="train").values; ev=(~tr)&vn        # evaluate on held-out recordings

def codes(x):                                    # any per-syllable categorical -> 0..K-1
    u={c:i for i,c in enumerate(sorted(pd.unique(x)))}; return np.array([u[v] for v in x]),len(u)

def loss(s):
    """Held-out mean neg-log-prob of the next microcluster, given per-syllable state s.
    Order-2 (previous+current state) where a context is seen >= NMIN times, else back off to
    order-1, else to the global next-microcluster marginal."""
    K=int(s.max())+1
    ps=np.where(prev_ok, s[np.clip(prev_row,0,N-1)], -1)     # previous syllable's state
    a1=tr&vn; N1=np.zeros((K,V)); np.add.at(N1,(s[a1],nm[a1]),1)
    a2=a1&(ps>=0); N2=np.zeros((K*K,V)); np.add.at(N2,(ps[a2]*K+s[a2],nm[a2]),1); s2=N2.sum(1)
    marg=(N1.sum(0)+ALPHA)/(N1.sum()+ALPHA*V)
    ye=nm[ev]; c1=s[ev]; c2=np.where(ps[ev]>=0,ps[ev]*K+s[ev],-1)
    o1=(N1[c1,ye]+ALPHA)/(N1[c1].sum(1)+ALPHA*V)
    ok=(c2>=0)&(s2[np.clip(c2,0,K*K-1)]>=NMIN)
    o2=(N2[np.clip(c2,0,K*K-1),ye]+ALPHA)/(s2[np.clip(c2,0,K*K-1)]+ALPHA*V)
    p=np.where(ok,o2,np.where(N1[c1].sum(1)>=NMIN,o1,marg[ye]))
    return float(np.mean(-np.log(p)))

# ---- predictive-state merge: 64 microclusters -> nested family of groupings ----
m1=tr&vn; C=np.zeros((V,V)); np.add.at(C,(micro[m1],nm[m1]),1)
w=C.sum(1); w=w/(w.sum() or 1); q=(C+1e-9)/(C.sum(1,keepdims=True)+1e-9*V)
clus={i:dict(w=w[i],q=q[i].copy()) for i in range(V)}; lab=np.arange(V); nid=V; snaps={V:lab.copy()}
def mc(a,b):
    ss=clus[a]["w"]+clus[b]["w"]
    if ss<=0: return 0.0
    pa,pb=clus[a]["w"]/ss,clus[b]["w"]/ss; mm=pa*clus[a]["q"]+pb*clus[b]["q"]
    f=lambda u,v:np.sum(np.where(u>0,u*np.log((u+1e-12)/(v+1e-12)),0)); return ss*(pa*f(clus[a]["q"],mm)+pb*f(clus[b]["q"],mm))
while len(clus)>2:
    ids=list(clus); best=None
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            c=mc(ids[i],ids[j])
            if best is None or c<best[0]: best=(c,ids[i],ids[j])
    _,a,b=best; sw=clus[a]["w"]+clus[b]["w"]; qn=(clus[a]["w"]*clus[a]["q"]+clus[b]["w"]*clus[b]["q"])/(sw or 1)
    for x in np.where(lab==a)[0]: lab[x]=nid
    for x in np.where(lab==b)[0]: lab[x]=nid
    clus[nid]=dict(w=sw,q=qn); del clus[a]; del clus[b]; nid+=1
    snaps[len(clus)]=lab.copy()
def state_at(K):
    L=snaps[K]; u={c:i for i,c in enumerate(sorted(np.unique(L)))}; return np.array([u[x] for x in L])[micro]

# ---- baselines (all predict the next microcluster, so losses are on one scale) ----
exp_s,_=codes(d.call_type.values); snd_s,nsnd=codes(d.sound_cluster.values)
print(f"held-out prediction of the next microcluster (nats, lower is better):")
print(f"  expert call types (K=9)            : {loss(exp_s):.3f}")
print(f"  sound-only clustering (K={nsnd})       : {loss(snd_s):.3f}")
print(f"  microcluster ceiling (K=64)        : {loss(micro):.3f}")
print()
print(f"  discovered predictive states, held-out loss vs K:")
for K in [6,9,10,11,12,15,20]:
    print(f"    K={K:>2}: {loss(state_at(K)):.3f}")
print(f"\n  headline K=11: {loss(state_at(11)):.3f} nats")
