# -*- coding: utf-8 -*-
"""Bengalese-finch analogues of the marmoset spectrogram figure and the reorganization heatmap.

Reuses the EXACT finch_recluster.py pipeline (numpy kmeans-64 microclusters -> agglomerative
merge minimising loss of predicting the next expert syllable label -> K = number of expert types)
so the discovered states match the Table 2 result. Then draws:
  fig_finch_spectrogram.png  (A: one song bout, expert label vs discovered state; B: expert-type
                              exemplars; C: discovered-state exemplars grouped by parent type)
  fig_finch_reorg.png        (heatmap: each expert syllable type distributed across states, row %)
Real STFT spectrograms from the raw 32 kHz WAVs. Writes to ../figs/.

Usage:  python finch_figures.py [Bird5]
"""
import os, sys, wave, math
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.signal import spectrogram as sp_spectrogram, get_window
np.seterr(all="ignore")

BIRD = sys.argv[1] if len(sys.argv) > 1 else "Bird9"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("FINCH_DATA", os.path.join(HERE, "data"))
BDIR = os.path.join(ROOT, BIRD)
FIGS = os.path.normpath(os.path.join(HERE, "..", "figs")); os.makedirs(FIGS, exist_ok=True)
SR, N_FFT, HOP, N_MELS, FMIN, FMAX, TFR = 32000, 512, 128, 32, 500, 16000, 16
NMICRO, SMOOTH, SEED = 64, 0.5, 42
FLO, FHI, DR = 400, 11000, 42          # display band (Hz) and dynamic range (dB)

# ---------------- audio features (identical to finch_recluster) ----------------
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
FB = mel_fb(); MELWIN = np.hanning(N_FFT)
def logmel(sig):
    if len(sig) < N_FFT: sig = np.pad(sig, (0, N_FFT-len(sig)))
    nfr = max(1, 1 + (len(sig)-N_FFT)//HOP)
    fr = np.stack([sig[i*HOP:i*HOP+N_FFT]*MELWIN for i in range(nfr)])
    S = np.abs(np.fft.rfft(fr, axis=1))**2
    lm = np.log(FB @ S.T + 1e-8)
    if nfr != TFR:
        xo = np.linspace(0, 1, nfr); xn = np.linspace(0, 1, TFR)
        lm = np.stack([np.interp(xn, xo, lm[k]) for k in range(N_MELS)])
    return lm.flatten()

# ---------------- parse annotation, load audio ----------------
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
        rows.append(dict(bout=bi, wav=wavf, k=k, lab=lab, ap=ap, ln=ln, feat=logmel(seg)))
print(f"[{BIRD}] syllables={len(rows)} bouts={len(set(r['bout'] for r in rows))} wavs={len(wav_cache)}")

X = np.stack([r["feat"] for r in rows])
lab = np.array([r["lab"] for r in rows])
wavof = np.array([r["wav"] for r in rows])
bybout = defaultdict(list)
for i, r in enumerate(rows): bybout[r["bout"]].append(i)
nextlab = np.array([None]*len(rows), dtype=object)
for bid, idxs in bybout.items():
    idxs = sorted(idxs, key=lambda i: rows[i]["k"])
    for a, b in zip(idxs[:-1], idxs[1:]): nextlab[a] = lab[b]
wavs = sorted(set(wavof)); wsplit = {w: ("test" if k % 4 == 0 else "train") for k, w in enumerate(wavs)}
split = np.array([wsplit[w] for w in wavof])
has_next = np.array([nx is not None for nx in nextlab])
cats = sorted(set(lab)); c2i = {c: i for i, c in enumerate(cats)}; NY = len(cats)

# ---------------- PCA + numpy KMeans micro (train), assign all ----------------
tr_all = (split == "train"); mu = X[tr_all].mean(0); Xc = X - mu
U, Sv, Vt = np.linalg.svd(Xc[tr_all], full_matrices=False)
cum = np.cumsum(Sv**2)/np.sum(Sv**2); DPC = min(int(np.searchsorted(cum, 0.90)+1), 20)
Z = Xc @ Vt[:DPC].T
use = has_next; y = np.array([c2i[c] for c in nextlab[use]])
Zu = Z[use]; labu = lab[use]; tr = (split[use] == "train")
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
trrow = (split == "train")
C = kmeans(Z[trrow], NMICRO)
micro_all = (((Z[:, None, :]-C[None])**2).sum(2)).argmin(1)     # microcluster for every syllable
# next microcluster within each bout (the unsupervised prediction target)
nextmicro = np.full(len(rows), -1, int)
for bid, idxs in bybout.items():
    idxs = sorted(idxs, key=lambda i: rows[i]["k"])
    for a, b in zip(idxs[:-1], idxs[1:]): nextmicro[a] = micro_all[b]

# ---------------- agglomerative merge on p(next microcluster | micro) ----------------
trm = trrow & (nextmicro >= 0)
Cnt = np.zeros((NMICRO, NMICRO)); np.add.at(Cnt, (micro_all[trm], nextmicro[trm]), 1)
pw = Cnt.sum(1); q = (Cnt+SMOOTH)/(Cnt.sum(1, keepdims=True)+SMOOTH*NMICRO); pw = pw/pw.sum()
def js(pa, pb, qa, qb):
    s = pa+pb; a, b = pa/s, pb/s; m = a*qa+b*qb
    kl = lambda u, v: np.sum(np.where(u > 0, u*np.log(u/v), 0))
    return s*(a*kl(qa, m)+b*kl(qb, m))
clus = {i: dict(mem=[i], w=pw[i], q=q[i].copy()) for i in range(NMICRO)}
label = np.arange(NMICRO); nid = NMICRO
while len(clus) > NY:
    ids = list(clus); best = None
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]; cost = js(clus[a]["w"], clus[b]["w"], clus[a]["q"], clus[b]["q"])
            if best is None or cost < best[0]: best = (cost, a, b)
    _, a, b = best; w = clus[a]["w"]+clus[b]["w"]
    qn = (clus[a]["w"]*clus[a]["q"]+clus[b]["w"]*clus[b]["q"])/w; mem = clus[a]["mem"]+clus[b]["mem"]
    for m_ in mem: label[m_] = nid
    clus[nid] = dict(mem=mem, w=w, q=qn); del clus[a]; del clus[b]; nid += 1
uni = {c: i for i, c in enumerate(sorted(np.unique(label)))}
micro2state = np.array([uni[label[m]] for m in range(NMICRO)])
state_all = micro2state[micro_all]                                    # discovered state per syllable
for i, r in enumerate(rows): r["state"] = int(state_all[i])

# ---------------- contingency, state naming ----------------
K = NY
M = np.zeros((NY, K), int)                       # rows expert type, cols state
for r in rows: M[c2i[r["lab"]], r["state"]] += 1
type_n = M.sum(1); state_n = M.sum(0)
typeorder = list(np.argsort(-type_n))            # expert types, most frequent first
dom_state = {s: int(np.argmax(M[:, s])) for s in range(K)}            # dominant expert type of a state
domfrac = {s: (M[:, s].max()/state_n[s] if state_n[s] else 0) for s in range(K)}
# order states: grouped under their dominant expert type (type order), largest first.
# Neutral names S1..SK (digit expert labels make dominant-type state names ambiguous under merges).
state_order = []
for t in typeorder:
    sts = sorted([s for s in range(K) if dom_state[s] == t], key=lambda s: -state_n[s])
    state_order.extend(sts)
for s in range(K):
    if s not in state_order: state_order.append(s)
state_name = {s: f"S{i+1}" for i, s in enumerate(state_order)}
AMI_txt = ""
print(f"[{BIRD}] NY={NY}  contingency rows=expert type, cols=state (ordered):")
hdr = "        " + " ".join(f"{state_name[s]:>5}" for s in state_order); print(hdr)
for t in typeorder:
    print(f"  {cats[t]:>4} n={type_n[t]:>5} " + " ".join(f"{M[t, s]:>5}" for s in state_order))
n_split = sum(1 for t in typeorder if sum(1 for s in range(K) if dom_state[s] == t) > 1)
print(f"[{BIRD}] {n_split} expert types split across >1 discovered state")

# per-state successor distribution: the axis on which same-sound states (e.g. S1, S2) differ
from collections import Counter
state_next = {s: Counter() for s in range(K)}
for i, r in enumerate(rows):
    if nextlab[i] is not None: state_next[r["state"]][nextlab[i]] += 1
def top_next(s, k=3):
    tot = sum(state_next[s].values())
    return [(l, c/tot) for l, c in state_next[s].most_common(k)] if tot else []
print(f"[{BIRD}] state successors (what each predicts):")
for s in state_order:
    print(f"  {state_name[s]} (syll {cats[dom_state[s]]}): " +
          ", ".join(f"{l}:{p*100:.0f}%" for l, p in top_next(s)))

# ================= raw STFT + exemplar selection =================
def seg_audio(r): return wav_cache[r["wav"]][r["ap"]:r["ap"]+r["ln"]]
def stft_db(x, margin=0.008):
    m = int(margin*SR);
    if m:
        pass
    nper = max(64, int(round(0.004*SR))); nover = int(nper*0.85)
    f, tt, Sxx = sp_spectrogram(x, fs=SR, window=get_window("hann", nper), nperseg=nper,
                                noverlap=nover, nfft=512, scaling="spectrum", mode="magnitude")
    Sdb = 20*np.log10(Sxx+1e-9); sel = (f >= FLO-300) & (f <= FHI+300)
    return dict(S=Sdb[sel], f=f[sel], dur=len(x)/SR)
def central(idxs):                               # representative: logmel closest to the group mean
    if len(idxs) == 0: return None
    F = X[idxs]; c = F.mean(0); d = ((F-c)**2).sum(1)
    order = np.argsort(d)
    for oi in order:
        r = rows[idxs[oi]]
        if 0.02 <= r["ln"]/SR <= 0.5: return r
    return rows[idxs[order[0]]]

# ============================ rendering setup ============================
CMAP = "magma"; STROKE = [pe.withStroke(linewidth=2.2, foreground="black")]
PAL = ["#4da3ff", "#ff5a5a", "#35d07f", "#ffb14e", "#c78bff", "#4dd0e1", "#ff8ad8",
       "#c79a6b", "#9ccc65", "#f06292", "#7986cb", "#a1887f", "#4db6ac", "#ba68c8",
       "#dce775", "#90a4ae", "#ffd54f"]
type_col = {cats[t]: PAL[i % len(PAL)] for i, t in enumerate(typeorder)}
def imspec(ax, sp, x0=0.0, vmax=None):
    S = sp["S"]; vmax = (np.percentile(S, 99.7) if vmax is None else vmax)
    ax.imshow(S, aspect="auto", origin="lower", cmap=CMAP, vmin=vmax-DR, vmax=vmax,
              extent=[x0, x0+sp["dur"], sp["f"][0]/1000, sp["f"][-1]/1000], interpolation="bilinear")
    return vmax
def style(ax, tmax):
    ax.set_ylim(FLO/1000, FHI/1000); ax.set_yticks([2, 5, 8, 11]); ax.set_xlim(0, tmax)
    ax.set_facecolor("black")
    for sp in ax.spines.values(): sp.set_color("#888")
    ax.tick_params(colors="#555", labelsize=8)

# ---- choose a clean song bout for panel A (variety of types AND states, dom==label) ----
L = 9; best = None
for bid, idxs in bybout.items():
    idxs = sorted(idxs, key=lambda i: rows[i]["k"])
    if len(idxs) < L: continue
    for i in range(len(idxs)-L+1):
        win = [rows[j] for j in idxs[i:i+L]]
        ntypes = len({r["lab"] for r in win}); nstates = len({r["state"] for r in win})
        consist = np.mean([dom_state[r["state"]] == c2i[r["lab"]] for r in win])
        durok = np.mean([0.02 <= r["ln"]/SR <= 0.4 for r in win])
        sc = consist*60 + nstates*22 + ntypes*18 + durok*40
        if best is None or sc > best[0]: best = (sc, win)
seqwin = best[1]
print(f"[{BIRD}] seq bout: " + " ".join(f"{r['lab']}/{state_name[r['state']]}" for r in seqwin))

fig = plt.figure(figsize=(12.0, 11.6)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 0.85, 1.5], hspace=0.85)

# ---- Panel A: song bout ----
axA = fig.add_subplot(gs[0]); GAP = 0.012
specs = [stft_db(seg_audio(r)) for r in seqwin]
gmax = np.percentile(np.concatenate([s["S"].ravel() for s in specs]), 99.7)
cur = 0.0; centers = []
for r, sp in zip(seqwin, specs):
    imspec(axA, sp, x0=cur, vmax=gmax); centers.append((cur+sp["dur"]/2, r)); cur += sp["dur"]+GAP
style(axA, tmax=cur-GAP+0.005)
for x, r in centers:
    axA.text(x, FHI/1000-0.4, r["lab"], ha="center", va="top", fontsize=10, fontweight="bold",
             color=type_col.get(r["lab"], "w"), path_effects=STROKE)
    axA.text(x, FLO/1000+0.35, state_name[r["state"]], ha="center", va="bottom", fontsize=9,
             color="white", style="italic", path_effects=STROKE)
axA.set_ylabel("frequency (kHz)"); axA.set_xticks([])
axA.set_xlabel("time  →   (silent intervals compressed)", fontsize=9)
axA.set_title("A   One song bout — expert syllable label (top) vs. discovered predictive state (bottom, italic)",
              loc="left", fontweight="bold", fontsize=11)

# ---- Panel B: expert-type exemplars ----
ex_type = {t: central([i for i, r in enumerate(rows) if r["lab"] == cats[t]]) for t in typeorder}
bt = [t for t in typeorder if ex_type[t] is not None]
gsB = gs[1].subgridspec(1, len(bt), wspace=0.18); axB0 = None
for i, t in enumerate(bt):
    ax = fig.add_subplot(gsB[i]); axB0 = axB0 or ax
    sp = stft_db(seg_audio(ex_type[t])); imspec(ax, sp); style(ax, tmax=sp["dur"])
    ax.set_title(cats[t], fontsize=10.5, color=type_col[cats[t]], fontweight="bold", pad=3)
    ax.text(0.96, 0.05, f"{sp['dur']*1000:.0f} ms", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="white", path_effects=STROKE)
    ax.set_xticks([])
    ax.set_ylabel("freq (kHz)") if i == 0 else ax.set_yticklabels([])
pB = axB0.get_position()
fig.text(pB.x0, pB.y1+0.03, "B   Exemplars of expert syllable types", ha="left", va="bottom",
         fontweight="bold", fontsize=12)

# ---- Panel C: exemplars of all discovered states, uniform grid; parent type in the title ----
statesC = [s for s in state_order]
ex_state = {s: central([i for i, r in enumerate(rows) if r["state"] == s]) for s in statesC}
statesC = [s for s in statesC if ex_state[s] is not None]
ncol = min(3, len(statesC)); nrow = int(np.ceil(len(statesC)/ncol))
gsC = gs[2].subgridspec(nrow, ncol, wspace=0.22, hspace=0.62); axC0 = None
for idx, s in enumerate(statesC):
    rr, cc = divmod(idx, ncol)
    ax = fig.add_subplot(gsC[rr, cc]); axC0 = axC0 or ax
    sp = stft_db(seg_audio(ex_state[s])); imspec(ax, sp); style(ax, tmax=sp["dur"])
    parent = cats[dom_state[s]]; pcol = type_col.get(parent, "#ccc")
    ax.set_title(f"{state_name[s]}   (syllable {parent})", fontsize=9.5, color=pcol, fontweight="bold", pad=2)
    tn = top_next(s, 1)
    if tn:
        ax.text(0.045, 0.93, f"then → {tn[0][0]}  ({tn[0][1]*100:.0f}%)", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#39d353", fontweight="bold", path_effects=STROKE)
    ax.text(0.95, 0.05, f"{sp['dur']*1000:.0f} ms", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="white", path_effects=STROKE)
    ax.set_xticks([])
    ax.set_ylabel("freq (kHz)", fontsize=8) if cc == 0 else ax.set_yticklabels([])
for idx in range(len(statesC), nrow*ncol):
    rr, cc = divmod(idx, ncol); fig.add_subplot(gsC[rr, cc]).axis("off")
splits = [cats[t] for t in typeorder if sum(1 for s in range(K) if dom_state[s] == t) > 1]
if not splits:
    subC = "predictive states re-sort the expert syllables by sequential role"
elif len(splits) == 1:
    subC = f"syllable {splits[0]} appears as more than one state: similar variants, told apart by what follows"
else:
    sy = ", ".join(splits[:-1]) + " and " + splits[-1]
    subC = f"syllables {sy} each appear as more than one state: similar variants, told apart by what follows"
pC = axC0.get_position()
fig.text(pC.x0, pC.y1+0.055, "C   Exemplars of discovered predictive states", ha="left", va="bottom",
         fontweight="bold", fontsize=12)
fig.text(pC.x0, pC.y1+0.032, subC, ha="left", va="bottom", fontsize=9.5, color="#444", style="italic")
out1 = os.path.join(FIGS, "fig_finch_spectrogram.png")
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white"); print("wrote", out1)

# ================= reorganization heatmap (analogue of Fig 5) =================
Mo = M[np.ix_(typeorder, state_order)].astype(float)
rown = type_n[typeorder]
Mn = (Mo / Mo.sum(1, keepdims=True) * 100)
collab = [state_name[s] for s in state_order]
rowlab = [f"{cats[t]}  (n={type_n[t]:,})" for t in typeorder]
figh, axh = plt.subplots(figsize=(max(6.5, 0.5*K+2.2), 0.42*NY+1.7))
im = axh.imshow(Mn, cmap="Blues", aspect="auto", vmin=0, vmax=100)
axh.set_xticks(range(K)); axh.set_xticklabels(collab, rotation=40, ha="right", fontsize=8)
axh.set_yticks(range(NY)); axh.set_yticklabels(rowlab, fontsize=8)
for tl, n in zip(axh.get_yticklabels(), rown):
    if n < 50: tl.set_color("#9a9a9a"); tl.set_fontstyle("italic")
for i in range(NY):
    for j in range(K):
        v = Mn[i, j]
        if v >= 6: axh.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 55 else "#222")
axh.set_xlabel("discovered predictive state"); axh.set_ylabel("expert syllable type")
for sp in axh.spines.values(): sp.set_visible(False)
axh.tick_params(length=0)
axh.set_title(f"{BIRD}: each expert syllable type distributed across predictive states (row %)", fontsize=9.5, pad=26)
# brackets under states sharing a dominant type (the splits)
j = 0
while j < K:
    t = dom_state[state_order[j]]; j2 = j
    while j2+1 < K and dom_state[state_order[j2+1]] == t: j2 += 1
    if j2 > j:
        col = type_col[cats[t]]
        axh.plot([j-0.45, j2+0.45], [-1.05, -1.05], color=col, lw=2, clip_on=False)
        axh.text((j+j2)/2, -1.35, f"{cats[t]}: {j2-j+1} states", color=col, ha="center", va="bottom",
                 fontsize=7.8, fontweight="bold")
    j = j2+1
cb = figh.colorbar(im, ax=axh, fraction=0.045, pad=0.02); cb.set_label("% of syllable type", fontsize=8)
cb.ax.tick_params(labelsize=7)
figh.tight_layout()
out2 = os.path.join(FIGS, "fig_finch_reorg.png")
figh.savefig(out2, dpi=200, bbox_inches="tight"); print("wrote", out2); print("[done]")
