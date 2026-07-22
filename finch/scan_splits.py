# -*- coding: utf-8 -*-
"""Headless scan: for each finch, run the UNSUPERVISED merge (next-microcluster target) to K=NY,
then for each expert type that splits across >1 state, report the two states' dominant next
EXPERT syllable, so we can pick a bird whose split is interpretable (states with DIFFERENT
next-syllables). No rendering."""
import os, sys, wave, math
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "data")
SR, N_FFT, HOP, N_MELS, FMIN, FMAX, TFR = 32000, 512, 128, 32, 500, 16000, 16
NMICRO, SMOOTH, SEED = 64, 0.5, 42
def hz2mel(f): return 2595.0*np.log10(1.0+f/700.0)
def mel2hz(m): return 700.0*(10.0**(m/2595.0)-1.0)
def mel_fb():
    bins = N_FFT//2+1; freqs = np.linspace(0, SR/2, bins)
    m = np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS+2); hz = mel2hz(m); fb = np.zeros((N_MELS, bins))
    for i in range(N_MELS):
        lo, ce, hi = hz[i], hz[i+1], hz[i+2]
        fb[i] = np.clip(np.minimum((freqs-lo)/(ce-lo+1e-9), (hi-freqs)/(hi-ce+1e-9)), 0, None)
    return fb
FB = mel_fb(); WIN = np.hanning(N_FFT)
def logmel(sig):
    if len(sig) < N_FFT: sig = np.pad(sig, (0, N_FFT-len(sig)))
    nfr = max(1, 1+(len(sig)-N_FFT)//HOP); fr = np.stack([sig[i*HOP:i*HOP+N_FFT]*WIN for i in range(nfr)])
    lm = np.log(FB @ np.abs(np.fft.rfft(fr, axis=1)).T**2 + 1e-8)
    if nfr != TFR:
        xo = np.linspace(0, 1, nfr); xn = np.linspace(0, 1, TFR); lm = np.stack([np.interp(xn, xo, lm[k]) for k in range(N_MELS)])
    return lm.flatten()
def kmeans(Xtr, K, iters=60, n_init=4):
    rng = np.random.RandomState(SEED); best = None
    for _ in range(n_init):
        C = Xtr[rng.choice(len(Xtr), K, replace=False)].copy()
        for _ in range(iters):
            a = ((Xtr[:, None, :]-C[None])**2).sum(2).argmin(1)
            newC = np.array([Xtr[a == k].mean(0) if (a == k).any() else C[k] for k in range(K)])
            if np.allclose(newC, C): break
            C = newC
        D = ((Xtr[:, None, :]-C[None])**2).sum(2); a = D.argmin(1); it = D[np.arange(len(Xtr)), a].sum()
        if best is None or it < best[0]: best = (it, C)
    return best[1]
def cost(pa, pb, qa, qb):
    s = pa+pb; a, b = pa/s, pb/s; r = a*qa+b*qb
    kl = lambda u, v: np.sum(np.where(u > 0, u*np.log(u/v), 0)); return s*(a*kl(qa, r)+b*kl(qb, r))

for BIRD in ["Bird%d" % i for i in range(11)]:
    BDIR = os.path.join(ROOT, BIRD)
    if not os.path.isfile(os.path.join(BDIR, "Annotation.xml")): continue
    tree = ET.parse(os.path.join(BDIR, "Annotation.xml")); root = tree.getroot(); rows = []; wc = {}
    for bi, seq in enumerate(root.findall("Sequence")):
        wf = seq.find("WaveFileName").text.strip(); sp = int(seq.find("Position").text)
        if wf not in wc:
            w = wave.open(os.path.join(BDIR, "Wave", wf), "rb"); sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float); w.close(); wc[wf] = sig/(np.abs(sig).max()+1e-9)
        sig = wc[wf]
        for k, nt in enumerate(seq.findall("Note")):
            rp = int(nt.find("Position").text); ln = int(nt.find("Length").text); ap = sp+rp; seg = sig[ap:ap+ln]
            if len(seg) >= 32: rows.append((bi, wf, k, nt.find("Label").text.strip(), logmel(seg)))
    n = len(rows); X = np.stack([r[4] for r in rows]); lab = np.array([r[3] for r in rows]); wv = np.array([r[1] for r in rows])
    bb = defaultdict(list)
    for i, r in enumerate(rows): bb[r[0]].append(i)
    cats = sorted(set(lab)); NY = len(cats); c2i = {c: i for i, c in enumerate(cats)}
    wavs = sorted(set(wv)); tr = ~np.array([(wavs.index(w) % 4 == 0) for w in wv])
    mu = X[tr].mean(0); Xc = X-mu; U, S, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    DPC = min(int(np.searchsorted(np.cumsum(S**2)/np.sum(S**2), 0.90)+1), 20); Z = Xc @ Vt[:DPC].T
    C = kmeans(Z[tr], NMICRO); micro = (((Z[:, None, :]-C[None])**2).sum(2)).argmin(1)
    nextm = np.full(n, -1, int); nextlab = np.full(n, -1, int)
    for bid, idx in bb.items():
        idx = sorted(idx, key=lambda i: rows[i][2])
        for a, b in zip(idx[:-1], idx[1:]): nextm[a] = micro[b]; nextlab[a] = c2i[lab[b]]
    hn = nextm >= 0; trn = tr & hn
    Cn = np.zeros((NMICRO, NMICRO)); np.add.at(Cn, (micro[trn], nextm[trn]), 1)
    pw = Cn.sum(1); q = (Cn+SMOOTH)/(Cn.sum(1, keepdims=True)+SMOOTH*NMICRO); pw = pw/pw.sum()
    clus = {i: dict(w=pw[i], q=q[i].copy()) for i in range(NMICRO)}; label = np.arange(NMICRO); nid = NMICRO
    while len(clus) > NY:
        ids = list(clus); best = None
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]; c = cost(clus[a]["w"], clus[b]["w"], clus[a]["q"], clus[b]["q"])
                if best is None or c < best[0]: best = (c, a, b)
        _, a, b = best; w = clus[a]["w"]+clus[b]["w"]; qn = (clus[a]["w"]*clus[a]["q"]+clus[b]["w"]*clus[b]["q"])/w
        for m_ in np.where(label == a)[0]: label[m_] = nid
        for m_ in np.where(label == b)[0]: label[m_] = nid
        clus[nid] = dict(w=w, q=qn); del clus[a]; del clus[b]; nid += 1
    u = {c: i for i, c in enumerate(sorted(np.unique(label)))}; som = np.array([u[label[m]] for m in range(NMICRO)]); state = som[micro]
    K = NY; Mc = np.zeros((NY, K), int)
    for i in range(n): Mc[c2i[lab[i]], state[i]] += 1
    dom = {s: cats[int(np.argmax(Mc[:, s]))] for s in range(K)}
    nxt = {s: Counter() for s in range(K)}
    for i in range(n):
        if nextlab[i] >= 0: nxt[state[i]][cats[nextlab[i]]] += 1
    def topn(s):
        t = sum(nxt[s].values()); return [(l, round(c/t, 2)) for l, c in nxt[s].most_common(2)] if t else []
    # find expert types that split, and whether the split states differ in top next-label
    interp = []
    for t in cats:
        sts = [s for s in range(K) if dom[s] == t]
        if len(sts) > 1:
            tops = [topn(s)[0][0] if topn(s) else "?" for s in sts]
            diff = len(set(tops)) > 1
            interp.append((t, sts, tops, diff))
    flag = " *** INTERPRETABLE SPLIT ***" if any(x[3] for x in interp) else ""
    print(f"{BIRD} NY={NY}: splits={[(t, tops) for t, _, tops, _ in interp]}{flag}")
print("[done]")
