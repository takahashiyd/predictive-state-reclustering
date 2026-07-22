# -*- coding: utf-8 -*-
"""Predictive-state reclustering on Bengalese finch song (ACOUSTIC run, UNSUPERVISED).
Port of the marmoset pipeline to the Koumura BirdsongRecognition data.
Self-contained: numpy only (no sklearn/scipy/librosa). Reads WAVs with stdlib `wave`.

Pipeline (identical to the marmoset paper):
  syllable audio -> log-mel features -> PCA -> KMeans microclusters (M=64)
  -> agglomerative merge minimising the increase in NEXT-MICROCLUSTER prediction loss
  -> held-out loss predicting the next microcluster.
Target is the next MICROCLUSTER (a fixed 64-way alphabet), so the method is fully
unsupervised and every category system is scored on the same target. Expert syllable
labels are used only to evaluate (AMI reorganization) and as one baseline category system.
Baselines (all predict the next microcluster): EXPERT labels, sound-only KMeans(K=#experts).
Also: AMI(discovered states @K=#experts, expert labels) = reorganization,
and an order-1 vs order-2 history gain on the discovered states.
"""
import os, sys, wave, math
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
np.seterr(all="ignore")

BIRD = sys.argv[1] if len(sys.argv) > 1 else "Bird0"
# Durable, relative data location: <this folder>/data/BirdN/{Annotation.xml, Wave/*.wav}.
# Fetch it with download_finch.py (figshare 3470165). Override with $FINCH_DATA if needed.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("FINCH_DATA", os.path.join(HERE, "data"))
BDIR = os.path.join(ROOT, BIRD)
if not os.path.isfile(os.path.join(BDIR, "Annotation.xml")):
    sys.exit(f"[{BIRD}] data not found at {BDIR}. Run: python download_finch.py {BIRD}")
SR, N_FFT, HOP, N_MELS, FMIN, FMAX, TFR = 32000, 512, 128, 32, 500, 16000, 16
NMICRO, SMOOTH, SEED, NMIN = 64, 0.5, 42, 30

# ---------- audio features ----------
def hz2mel(f): return 2595.0*np.log10(1.0+f/700.0)
def mel2hz(m): return 700.0*(10.0**(m/2595.0)-1.0)
def mel_fb():
    bins = N_FFT//2+1; freqs = np.linspace(0, SR/2, bins)
    m = np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS+2); hz = mel2hz(m)
    fb = np.zeros((N_MELS, bins))
    for i in range(N_MELS):
        lo, ce, hi = hz[i], hz[i+1], hz[i+2]
        left = (freqs-lo)/(ce-lo+1e-9); right = (hi-freqs)/(hi-ce+1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb
FB = mel_fb(); WIN = np.hanning(N_FFT)
def logmel(sig):
    if len(sig) < N_FFT: sig = np.pad(sig, (0, N_FFT-len(sig)))
    nfr = max(1, 1 + (len(sig)-N_FFT)//HOP)
    fr = np.stack([sig[i*HOP:i*HOP+N_FFT]*WIN for i in range(nfr)])
    S = np.abs(np.fft.rfft(fr, axis=1))**2
    lm = np.log(FB @ S.T + 1e-8)
    if nfr != TFR:
        xo = np.linspace(0, 1, nfr); xn = np.linspace(0, 1, TFR)
        lm = np.stack([np.interp(xn, xo, lm[k]) for k in range(N_MELS)])
    return lm.flatten()

# ---------- parse annotation ----------
# Koumura format: Note.Position is RELATIVE to Sequence.Position (offset into the WAV).
# A "bout" is a Sequence; transitions run within a Sequence, ordered by note index.
tree = ET.parse(os.path.join(BDIR, "Annotation.xml")); root = tree.getroot()
rows = []; wav_cache = {}
for bi, seq in enumerate(root.findall("Sequence")):
    wavf = seq.find("WaveFileName").text.strip(); seq_pos = int(seq.find("Position").text)
    if wavf not in wav_cache:
        w = wave.open(os.path.join(BDIR, "Wave", wavf), "rb"); n = w.getnframes()
        sig = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64); w.close()
        wav_cache[wavf] = sig/(np.abs(sig).max()+1e-9)
    sig = wav_cache[wavf]
    for k, note in enumerate(seq.findall("Note")):
        rp = int(note.find("Position").text); ln = int(note.find("Length").text)
        lab = note.find("Label").text.strip(); ap = seq_pos + rp
        seg = sig[ap:ap+ln]
        if len(seg) < 32: continue
        rows.append((bi, wavf, k, lab, logmel(seg)))
print(f"[{BIRD}] syllables={len(rows)}  bouts={len(set(r[0] for r in rows))}  wavs={len(wav_cache)}")

n = len(rows)
X = np.stack([r[4] for r in rows])
lab = np.array([r[3] for r in rows]); wavof = np.array([r[1] for r in rows])
bybout = defaultdict(list)
for i, r in enumerate(rows): bybout[r[0]].append(i)
cats = sorted(set(lab)); c2i = {c: i for i, c in enumerate(cats)}; NY = len(cats)

# split by WAV (recording) so no recording appears in both train and test
wavs = sorted(set(wavof)); wav_split = {w: (k % 4 == 0) for k, w in enumerate(wavs)}   # True = test
te = np.array([wav_split[w] for w in wavof]); tr = ~te

# ---------- PCA (fit on train rows), KMeans microclusters (fit on train rows) ----------
mu = X[tr].mean(0); Xc = X - mu
U, Sv, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
cum = np.cumsum(Sv**2)/np.sum(Sv**2); DPC = min(int(np.searchsorted(cum, 0.90)+1), 20)
Z = Xc @ Vt[:DPC].T
def kmeans(Xtr, K, iters=60, n_init=4, seed=SEED):
    rng = np.random.RandomState(seed); best = None
    for _ in range(n_init):
        C = Xtr[rng.choice(len(Xtr), K, replace=False)].copy()
        for _ in range(iters):
            a = ((Xtr[:, None, :]-C[None])**2).sum(2).argmin(1)
            newC = np.array([Xtr[a == k].mean(0) if (a == k).any() else C[k] for k in range(K)])
            if np.allclose(newC, C): break
            C = newC
        D = ((Xtr[:, None, :]-C[None])**2).sum(2); a = D.argmin(1)
        inertia = D[np.arange(len(Xtr)), a].sum()
        if best is None or inertia < best[0]: best = (inertia, C)
    return best[1]
Cc = kmeans(Z[tr], NMICRO)
micro = (((Z[:, None, :]-Cc[None])**2).sum(2)).argmin(1)
print(f"[{BIRD}] expert types = {NY}: {cats} | PCA dims (90% var) = {DPC}")

# ---------- next / previous microcluster within each bout (the prediction target) ----------
nextmicro = np.full(n, -1, int); prevmicro = np.full(n, -1, int)
for bid, idxs in bybout.items():
    idxs = sorted(idxs, key=lambda i: rows[i][2])
    for a, b in zip(idxs[:-1], idxs[1:]): nextmicro[a] = micro[b]; prevmicro[b] = micro[a]
hn = nextmicro >= 0                               # position has a next unit in its bout

# ---------- merge microclusters by shared next-microcluster future ----------
# cost(a,b) = increase in next-unit prediction loss from pooling a and b (nats).
trn = tr & hn
Cn = np.zeros((NMICRO, NMICRO)); np.add.at(Cn, (micro[trn], nextmicro[trn]), 1)
pw = Cn.sum(1); q = (Cn+SMOOTH)/(Cn.sum(1, keepdims=True)+SMOOTH*NMICRO); pw = pw/pw.sum()
def cost(pa, pb, qa, qb):                         # w_a*KL(qa||r) + w_b*KL(qb||r), r = pooled
    s = pa+pb; a, b = pa/s, pb/s; r = a*qa+b*qb
    kl = lambda u, v: np.sum(np.where(u > 0, u*np.log(u/v), 0))
    return s*(a*kl(qa, r)+b*kl(qb, r))
clus = {i: dict(mem=[i], w=pw[i], q=q[i].copy()) for i in range(NMICRO)}
label = np.arange(NMICRO); snap = {}; targets = set(list(range(2, 31))+[NY, NMICRO]); nid = NMICRO
if len(clus) in targets: snap[len(clus)] = label.copy()
while len(clus) > 2:
    ids = list(clus); best = None
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]; c = cost(clus[a]["w"], clus[b]["w"], clus[a]["q"], clus[b]["q"])
            if best is None or c < best[0]: best = (c, a, b)
    _, a, b = best; w = clus[a]["w"]+clus[b]["w"]
    qn = (clus[a]["w"]*clus[a]["q"]+clus[b]["w"]*clus[b]["q"])/w; mem = clus[a]["mem"]+clus[b]["mem"]
    for m_ in mem: label[m_] = nid
    clus[nid] = dict(mem=mem, w=w, q=qn); del clus[a]; del clus[b]; nid += 1
    if len(clus) in targets: snap[len(clus)] = label.copy()

def contig_of_micro(nodelabel):                   # micro-node ids -> contiguous 0..K-1 per microcluster
    u = {c: i for i, c in enumerate(sorted(np.unique(nodelabel)))}
    return np.array([u[nodelabel[m]] for m in range(NMICRO)])

# ---------- held-out loss predicting the next microcluster, for any category system ----------
def loss_of(state):                               # state: contiguous label per row
    K = int(state.max())+1
    N1 = np.zeros((K, NMICRO)); np.add.at(N1, (state[trn], nextmicro[trn]), 1)
    Q = (N1+SMOOTH)/(N1.sum(1, keepdims=True)+SMOOTH*NMICRO)
    tem = te & hn
    return float(np.mean(-np.log(Q[state[tem], nextmicro[tem]]))), K

# ---------- baselines (predict next microcluster) ----------
expert_state = np.array([c2i[c] for c in lab])
llcat, _ = loss_of(expert_state)                                   # EXPERT labels
snd = kmeans(Z[tr], NY); snd_a = (((Z[:, None, :]-snd[None])**2).sum(2)).argmin(1)
llemis, _ = loss_of(snd_a)                                         # sound-only KMeans(K=NY)

print(f"\n[{BIRD}] predicting NEXT microcluster ({NMICRO}-way) | held-out loss, nats (lower=better)")
print(f"  EXPERT labels        (K={NY:>2}) : {llcat:.4f}")
print(f"  sound-only KMeans     (K={NY:>2}) : {llemis:.4f}")
print(f"  {'K':>3} {'test_loss':>10}")
llNY = float("nan"); reclust_NY = None
for K in sorted(snap):
    if K < 2: continue
    st = contig_of_micro(snap[K])[micro]
    ll, _ = loss_of(st)
    if K == NY: llNY = ll; reclust_NY = st
    if K in (2, 4, 6, 8, NY, 12, 16, 20, 25, 30, NMICRO):
        tag = "  <= BEATS expert" if (K == NY and ll < llcat) else ("  (vs expert)" if K == NY else "")
        print(f"  {K:>3} {ll:>10.4f}{tag}")

# ---------- AMI(discovered states @K=NY, expert labels) = reorganization ----------
def contingency(u, v):
    ua = sorted(set(u)); va = sorted(set(v)); ui = {c: i for i, c in enumerate(ua)}; vi = {c: i for i, c in enumerate(va)}
    M = np.zeros((len(ua), len(va)), int)
    for x, yv in zip(u, v): M[ui[x], vi[yv]] += 1
    return M
def entropy(cnts, N): p = cnts[cnts > 0]/N; return float(-(p*np.log(p)).sum())
def mutual_info(M, N):
    a = M.sum(1); b = M.sum(0); mi = 0.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            nij = M[i, j]
            if nij > 0: mi += (nij/N)*math.log(N*nij/(a[i]*b[j]))
    return mi
def emi(M, N):
    a = M.sum(1); b = M.sum(0); e = 0.0; lg = math.lgamma
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ai, bj = int(a[i]), int(b[j])
            for nij in range(max(1, ai+bj-N), min(ai, bj)+1):
                t = (nij/N)*math.log(N*nij/(ai*bj))
                logp = (lg(ai+1)+lg(bj+1)+lg(N-ai+1)+lg(N-bj+1)-lg(N+1)-lg(nij+1)
                        -lg(ai-nij+1)-lg(bj-nij+1)-lg(N-ai-bj+nij+1))
                e += t*math.exp(logp)
    return e
def ami(u, v):
    M = contingency(u, v); N = M.sum(); mi = mutual_info(M, N); e = emi(M, N)
    hu = entropy(M.sum(1), N); hv = entropy(M.sum(0), N); denom = 0.5*(hu+hv)-e
    return (mi-e)/denom if abs(denom) > 1e-12 else 0.0

a_val = ami(reclust_NY.tolist(), lab.tolist())
print(f"\n[{BIRD}] AMI(discovered states @K={NY}, expert labels) = {a_val:.3f}   (1=identical, 0=chance)")
M = contingency(reclust_NY.tolist(), lab.tolist())
print(f"[{BIRD}] contingency rows=discovered states, cols=expert types {cats}:")
for i, rowc in enumerate(M): print("   state%2d: %s" % (i, " ".join("%3d" % x for x in rowc)))

# ---------- order-1 vs order-2 history gain, predicting next microcluster from discovered states ----------
som = contig_of_micro(snap[NY])                    # micro -> state
state = som[micro]; prevstate = np.where(prevmicro >= 0, som[prevmicro], -1); K = NY
trn2 = tr & hn; N1 = np.zeros((K, NMICRO)); np.add.at(N1, (state[trn2], nextmicro[trn2]), 1)
hp = prevstate >= 0; trn2b = trn2 & hp
N2 = np.zeros((K*K, NMICRO)); np.add.at(N2, (prevstate[trn2b]*K+state[trn2b], nextmicro[trn2b]), 1); s2 = N2.sum(1)
tem = te & hn; y = nextmicro[tem]; c1 = state[tem]
o1 = (N1[c1, y]+SMOOTH)/(N1[c1].sum(1)+SMOOTH*NMICRO)
c2 = np.where(prevstate[tem] >= 0, prevstate[tem]*K+state[tem], -1); c2c = np.clip(c2, 0, K*K-1)
ok = (c2 >= 0) & (s2[c2c] >= NMIN)
o2 = (N2[c2c, y]+SMOOTH)/(s2[c2c]+SMOOTH*NMICRO)
l1 = float(np.mean(-np.log(o1))); l2 = float(np.mean(-np.log(np.where(ok, o2, o1))))
print(f"\n[{BIRD}] discovered-state history: order-1 loss {l1:.4f} | order-2 {l2:.4f} | gain {l1-l2:+.4f}")
print(f"SUMMARY|{BIRD}|N={n}|bouts={len(bybout)}|NY={NY}|expert={llcat:.4f}|sound={llemis:.4f}|"
      f"reclust_K{NY}={llNY:.4f}|AMI={a_val:.3f}|order2gain={l1-l2:+.4f}")

# ---------- durable results row ----------
import csv
row = dict(bird=BIRD, syllables=n, bouts=len(bybout), expert_types=NY,
           expert_ll=round(llcat, 4), sound_ll=round(llemis, 4), reclust_ll=round(llNY, 4),
           ami=round(a_val, 3), order2_gain=round(l1 - l2, 4),
           beats_expert=int(llNY < llcat), beats_sound=int(llNY < llemis))
csvp = os.path.join(HERE, "finch_results.csv")
newfile = not os.path.exists(csvp)
with open(csvp, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if newfile: w.writeheader()
    w.writerow(row)
print(f"[wrote row] {csvp}")
print("[done]")
