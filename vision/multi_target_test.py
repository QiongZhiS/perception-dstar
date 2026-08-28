"""vision/multi_target_test.py - docs/221 B1.2: multi-target + cross-category pollution test.

docs/220 unfinished item 1: the suspicious table is keyed by hue and therefore
naturally supports multiple maintained targets, but cross-category pollution was
never measured: with two different task colors co-present, does rejecting one
class pollute the other (its accounting or its bandwidth expansion harming the
neighbor class)?

Scenario (choice & rationale in docs/233):
  synthetic dual-target scene on a real DAVIS background (blackswan default):
  two colored patches, each with intra-patch hue texture (sigma_hue):
    T1 = flamingo-pink task hue ~170 (maintained; corrupted in corrupt segment)
    T2 = neighbor task hue: NEAR regime ~135 (sits at/inside the drifted
         bookkeeping hue of T1) or FAR regime ~73 (surf blue-green, hue-disjoint
         from T1).
  Segments (docs/219 protocol): base(20) -> dusk(20, warm-gain gradient) ->
    corrupt(15, T1-only recolor to drifted hue; "his-other's-no", docs/178) ->
    rest(25, T1 fake-recovery frames; T2 clean throughout).
  GT: T1 = 1 except corrupt + fake-recovery frames; T2 = 1 in every frame.

Cells (2x2, docs/220 s6.4 discipline: category-memory x white-balance, BOTH run):
  A  = category memory + white balance   (docs/219 "A full" semantics)
  B  = category memory, no white balance
  C  = global thr + white balance
  C2 = global thr, no white balance

MultiTargetMind: ONE shared category-memory core (rejected/thr_by/hist/
  rej_total/thr_cur) maintaining TWO task targets with per-target trust
  (trusted/consec/persist) - the honest architecture: the suspicious table is
  shared and hue-keyed, so rejections of T1 land in the same memory that T2's
  confirmation threshold reads.

Pollution measurement:
  ch1 bookkeeping: T2's thr excess over thr_base attributable to T1-origin
     rejected-hue entries (distance-weighted via bandwidth), per frame.
  ch2 bandwidth: contribution via expanded bw (hist bin mixing); reported
     through the same thr excess + bw in traces.
  ch3 memorial: shared rej_total -> T2 recovery delay k (reported separately).
  behavior: T2 P/R/F1 + segment pass rates in the two-target scene vs the SOLO
     control (identical frames; T1 present as a visual ghost but NOT maintained)
     - delta = pollution.

Protocol: 2 regimes x 2 scenes (dual/solo) x 4 cells x 10 seeds, mean+/-SD
  (docs/228) + JSON archive in vision/out/results (mt_<regime>.json + ckpt).

Safety discipline (repo rule): stdout prints ASCII labels only, one number per
  line (external regex extraction consumes them). Run from repo root via:
  powershell -NoProfile -Command "& python vision\multi_target_test.py --regime near *> logs\mt_near.log; ..."

Usage:
  python vision/multi_target_test.py --regime near
  python vision/multi_target_test.py --regime far
  python vision/multi_target_test.py --probe --bg-video blackswan
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402
import davis_suspicious as ds  # noqa: E402
from davis_a2_stats import mean_sd  # noqa: E402

OUT_DIR = os.path.join("vision", "out", "results")
SEG_DEF = {"base": 20, "dusk": 20, "corrupt": 15, "rest": 25}
SAT, VAL = 160, 200
RECOL_BASE = 133.0          # T1 corrupted-appearance hue (drifted bookkeeping)
CELLS = ["A", "B", "C", "C2"]
CELL_KW = {
    "A": dict(wb=True, global_thr=False),
    "B": dict(wb=False, global_thr=False),
    "C": dict(wb=True, global_thr=True),
    "C2": dict(wb=False, global_thr=True),
}
REGIMES = {
    "near": dict(t1_hue=170.0, t2_hue=135.0, t1_sig=10.0, t2_sig=22.0),
    "far": dict(t1_hue=170.0, t2_hue=73.0, t1_sig=10.0, t2_sig=22.0),
}


# ---- scene construction ----


def t1_box_frac(shape):
    H, W = shape[:2]
    return (int(0.16 * W), int(0.22 * H), int(0.09 * W), int(0.09 * H))


def t2_box_frac(shape):
    H, W = shape[:2]
    return (int(0.52 * W), int(0.22 * H), int(0.09 * W), int(0.09 * H))


def overlay_patch(frame, box, base_hue, sigma_hue, rng, sat=SAT, val=VAL):
    """Composite a textured color patch onto frame at box; return (img, mask)."""
    H, W = frame.shape[:2]
    x, y, w, h = [int(v) for v in box]
    x = min(max(x, 0), W - 1)
    y = min(max(y, 0), H - 1)
    w = min(w, W - x)
    h = min(h, H - y)
    hues = base_hue + rng.normal(0, sigma_hue, (h, w))
    hsv = np.dstack([np.clip(hues, 0, 179),
                     np.full((h, w), sat, np.float32),
                     np.full((h, w), val, np.float32)])
    patch = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = frame.copy()
    img[y:y + h, x:x + w] = patch
    mask = np.zeros((H, W), np.uint8)
    mask[y:y + h, x:x + w] = 255
    return img, mask


def box_mask(shape, box):
    H, W = shape[:2]
    x, y, w, h = [int(v) for v in box]
    m = np.zeros((H, W), np.uint8)
    m[y:y + h, x:x + w] = 255
    return m


def corrupt_t1(fr, b1, rcfg, rng, corrupt, recol_hue):
    """Apply T1-only corruption (his-other's-no): recolor to drifted hue (default,
    deterministic bookkeeping) or gain-physics drift (probe/reference)."""
    if corrupt == "recolor":
        fr, _ = overlay_patch(fr, b1, recol_hue, rcfg["t1_sig"], rng)
    else:  # gain physics, kc=0.16 (docs/219 gain corrupt, stronger)
        gv = np.array([1 + 0.9 * 0.16, 1.0, 1 - 0.9 * 0.16])
        m1 = box_mask(fr.shape[:2], b1)
        f = fr.astype(np.float32)
        f[m1 > 0] *= gv
        fr = np.clip(f, 0, 255).astype(np.uint8)
    return fr


def build_dual_scene(bg_frames, rcfg, seg_len, seed, corrupt="recolor",
                     noise_std=10.0, recol_hue=None):
    """Two-target sequence: base/dusk/corrupt(T1)/rest(T1 fake-recovery).

    Both patches are ALWAYS present (dual scene == solo scene); whether a target
    is "maintained" is decided by the runner, so solo is a perfect frame-level
    control with only the shared memory differing.
    """
    rng = np.random.default_rng(seed)
    idx = {}
    pos = 0
    for seg in ["base", "dusk", "corrupt", "rest"]:
        idx[seg] = list(range(pos, pos + seg_len[seg]))
        pos += seg_len[seg]
    total = pos
    n_src = len(bg_frames)
    shape = bg_frames[0].shape
    b1, b2 = t1_box_frac(shape), t2_box_frac(shape)
    out_f, out_m1, out_m2, out_s, out_g1, out_g2 = [], [], [], [], [], []
    rest_i = idx["rest"]
    for i in range(total):
        seg = next(s for s in ["base", "dusk", "corrupt", "rest"] if i in idx[s])
        src = bg_frames[i % n_src]
        fr, m1 = overlay_patch(src, b1, rcfg["t1_hue"], rcfg["t1_sig"], rng)
        fr, m2 = overlay_patch(fr, b2, rcfg["t2_hue"], rcfg["t2_sig"], rng)
        g1, g2 = 1, 1
        if seg == "dusk":
            k = idx["dusk"].index(i) / max(len(idx["dusk"]) - 1, 1)
            fr = np.clip(fr.astype(np.float32) * ds.warm_gain(k), 0, 255).astype(np.uint8)
        elif seg == "corrupt":
            g1 = 0
            n = rng.normal(0, noise_std, src.shape).astype(np.int16)
            fr = np.clip(fr.astype(np.int16) + n, 0, 255).astype(np.uint8)
            fr = corrupt_t1(fr, b1, rcfg, rng, corrupt, recol_hue)
        elif seg == "rest":
            kr = rest_i.index(i)
            if kr < 8 and kr % 2 == 1:          # fake recovery: still wrong
                g1 = 0
                n = rng.normal(0, noise_std, src.shape).astype(np.int16)
                fr = np.clip(fr.astype(np.int16) + n, 0, 255).astype(np.uint8)
                fr = corrupt_t1(fr, b1, rcfg, rng, corrupt, recol_hue)
            else:
                n = rng.normal(0, noise_std, src.shape).astype(np.int16)
                fr = np.clip(fr.astype(np.int16) + n, 0, 255).astype(np.uint8)
        else:
            n = rng.normal(0, noise_std, src.shape).astype(np.int16)
            fr = np.clip(fr.astype(np.int16) + n, 0, 255).astype(np.uint8)
        out_f.append(fr)
        out_m1.append(m1)
        out_m2.append(m2)
        out_s.append(seg)
        out_g1.append(g1)
        out_g2.append(g2)
    masks = [{"T1": a, "T2": b} for a, b in zip(out_m1, out_m2)]
    gts = [{"T1": a, "T2": b} for a, b in zip(out_g1, out_g2)]
    return out_f, masks, out_s, gts


# ---- shared-memory multi-target mind ----


class MultiTargetMind:
    """One shared category-memory core maintaining two task targets (B1.2).

    shared: rejected (suspicious table, docs/205), thr_by (category thr boosts,
            docs/206/207), hist (bandwidth bookkeeping, docs/208), rej_total
            (memorial, docs/200/203), thr_cur (global variant).
    per-target: trusted/consec/persist keyed by task hue.
    """

    def __init__(self, task_hues, wb=True, global_thr=False, template=False,
                 k_band=2.0, thr_base=0.60, thr_boost=0.04, thr_cap=0.85,
                 relax=0.02, persist_for=3, grow=3.0, cap_k=60):
        self.task_hues = [float(h) for h in task_hues]
        self.wb, self.global_thr, self.template = wb, global_thr, template
        self.k_band, self.thr_base = k_band, thr_base
        self.thr_boost, self.thr_cap, self.relax = thr_boost, thr_cap, relax
        self.persist_for, self.grow, self.cap_k = persist_for, grow, cap_k
        self.thr_cur = thr_base
        self.rejected = {}
        self.rej_total = 0
        self.rej_count = {}           # task_hue -> rejection count (origin split)
        self.thr_by = {}
        self.hist = {}
        self.rej_origin = {}          # hue -> {task_hue: count} (attribution)
        self.gate = ds.TemporalGateDavis() if wb else None
        self.g_base = None
        self.n_corr = 0
        self.trust = {h: {"trusted": True, "consec": 0, "persist": 0}
                      for h in self.task_hues}

    @staticmethod
    def _bin(hue):
        return int(hue // ds.HUE_BIN) * ds.HUE_BIN

    def bw(self, hue):
        if self.template:
            return 30.0
        obs = self.hist.get(self._bin(hue), [])
        if len(obs) < 3:
            return 0.0
        return self.k_band * float(np.std(obs))

    def _w_wide(self, d, bw):
        if bw <= 0:
            return 1.0 if d == 0 else 0.0
        if d >= bw:
            return 0.0
        return max(0.0, 1.0 - d / bw)

    def k(self, hue):
        if self.template:
            return 1
        return 1 + int(min(self.rej_total, self.cap_k) / self.grow)

    def thr(self, hue):
        if hue is None or self.template or self.global_thr:
            return self.thr_cur if (self.template or self.global_thr) else self.thr_base
        t = self.thr_base
        for sh, dv in self.thr_by.items():
            t += dv * self._w_wide(ds.circ(hue, sh), self.bw(sh))
        return min(self.thr_cap, t)

    def foreign_frac(self, hue, me):
        """Fraction of T2's thr excess attributable to OTHER targets' rejections.

        category mode: hue-keyed thr_by contribution weighted by rejection origin.
        global mode:   the shared thr is definitionally shared -> rejection-count
                       share from other targets.
        """
        if self.global_thr:
            tot = sum(self.rej_count.values())
            other = sum(c for h2, c in self.rej_count.items() if h2 != me)
            return other / tot if tot > 0 else 0.0
        tot, fg = 0.0, 0.0
        for sh, dv in self.thr_by.items():
            w = self._w_wide(ds.circ(hue, sh), self.bw(sh))
            contrib = dv * w
            if contrib <= 0:
                continue
            tot += contrib
            o = self.rej_origin.get(sh, {})
            n_other = sum(c for h2, c in o.items() if h2 != me)
            n_all = sum(o.values())
            fg += contrib * (n_other / n_all if n_all > 0 else 0.0)
        return fg / tot if tot > 0 else 0.0

    def step_target(self, task_hue, frac, hue, mask_empty):
        if hue is not None:
            self.hist.setdefault(self._bin(hue), []).append(hue)
        t = self.thr(hue)
        d = ds.circ(hue, task_hue) if hue is not None else 999.0
        ok = (not mask_empty) and frac > t and d < 30.0
        tr = self.trust[task_hue]
        if ok:
            tr["persist"] += 1
            if tr["persist"] >= self.persist_for:
                if self.template or self.global_thr:
                    self.thr_cur = max(self.thr_base, self.thr_cur - self.relax)
                elif hue is not None:
                    self.thr_by[hue] = max(0.0, self.thr_by.get(hue, 0.0) - self.relax)
            if not tr["trusted"]:
                tr["consec"] += 1
                if tr["consec"] >= self.k(hue):
                    tr["trusted"] = True
        else:
            tr["persist"] = 0
            tr["trusted"] = False
            tr["consec"] = 0
            self.rej_total += 1
            self.rej_count[task_hue] = self.rej_count.get(task_hue, 0) + 1
            if not self.template:
                if self.global_thr:
                    self.thr_cur = min(self.thr_cap, self.thr_cur + self.thr_boost)
                elif hue is not None:
                    self.thr_by[hue] = min(self.thr_cap,
                                           self.thr_by.get(hue, 0.0) + self.thr_boost)
                    self.rejected[hue] = self.rejected.get(hue, 0) + 1
                    o = self.rej_origin.setdefault(hue, {})
                    o[task_hue] = o.get(task_hue, 0) + 1
        return ok


def run_dual(mind, frames, masks, segs, targets):
    """Run the sequence; targets = list of (task_hue, mask_key)."""
    oks = {h: [] for h, _ in targets}
    seg_r = {h: {} for h, _ in targets}
    trace = {h: [] for h, _ in targets}
    threxc = {h: [] for h, _ in targets}
    fg_frac = {h: [] for h, _ in targets}
    n_open = 0
    for i, (fr, ms, seg) in enumerate(zip(frames, masks, segs)):
        img = fr
        if mind.wb:
            ex = np.zeros_like(ms["T1"])
            for key in ms:                    # all scene objects (T1 ghost too)
                mk = ms[key]
                if mk.max() > 0:
                    ex = cv2.bitwise_or(ex, cv2.dilate(mk, np.ones((21, 21), np.uint8)))
            s, _, mad = ds.bg_stats(fr, ex)
            g = ds.bg_gain(fr, ex)
            mind.gate.update(s, mad)
            do_corr = False
            if g is not None:
                if mind.g_base is None:
                    mind.g_base = g.copy()
                do_corr = mind.gate.open() and \
                    float(np.max(np.abs(g - mind.g_base))) > 0.12
            if do_corr:
                img = np.clip(fr.astype(np.float32) * g, 0, 255).astype(np.uint8)
                mind.n_corr += 1
            if mind.gate.open():
                n_open += 1
        for h, key in targets:
            frac, hue = ds.target_obs(img, ms[key], h)
            ok = mind.step_target(h, frac, hue, ms[key].max() == 0)
            claimed = bool(mind.trust[h]["trusted"] and ok)
            oks[h].append(claimed)
            seg_r[h].setdefault(seg, []).append(claimed)
            if hue is not None:
                tt = mind.thr(hue)
                threxc[h].append(tt - mind.thr_base)
                fg_frac[h].append(mind.foreign_frac(hue, h))
            else:
                threxc[h].append(0.0)
                fg_frac[h].append(0.0)
            trace[h].append((i, seg, frac, hue, ok, claimed,
                             mind.thr(hue) if hue is not None else None,
                             mind.k(hue) if hue is not None else 1))
    return oks, seg_r, trace, threxc, fg_frac, mind.n_corr, n_open


def serialize_trace_mt(trace):
    out = []
    for i, seg, frac, hue, ok, claimed, t, k in trace:
        out.append({"f": int(i), "seg": str(seg), "frac": round(float(frac), 4),
                    "hue": None if hue is None else round(float(hue), 3),
                    "ok": int(ok), "claimed": int(claimed),
                    "thr": None if t is None else round(float(t), 4),
                    "k": int(k)})
    return out


def pack_targets(oks, seg_r, trace, threxc, fgf, gts, segs, targets, mind, n_open, n_corr):
    out = {}
    for h, key in targets:
        m = ds.metrics(oks[h], [g[key] for g in gts])
        seg_rates = {s: float(np.mean(v)) for s, v in seg_r[h].items()}
        ex = [float(x) for x in threxc[h]]
        ex_pos = [x for x in ex if x > 0.001]
        fg = float(np.mean([f for f, x in zip(fgf[h], threxc[h]) if x > 0.001])) \
            if ex_pos else 0.0
        kmax = max((e[7] for e in trace[h]), default=1)
        n_rej = sum(1 for e in trace[h] if not e[4])
        out[key] = {
            "f1": round(float(m["f1"]), 4), "p": round(float(m["p"]), 4),
            "r": round(float(m["r"]), 4), "flips": int(ds.flips(oks[h], segs)),
            "seg_rates": {k2: round(v, 4) for k2, v in seg_rates.items()},
            "threxc_mean": round(float(np.mean(ex)), 4) if ex else 0.0,
            "threxc_max": round(float(np.max(ex)), 4) if ex else 0.0,
            "foreign_frac": round(fg, 4),
            "k_max": int(kmax), "n_rej": int(n_rej),
            "rej_total": int(mind.rej_total),
            "trace": serialize_trace_mt(trace[h]),
        }
    out["_shared"] = {"n_open": int(n_open), "n_corr": int(n_corr)}
    return out


def run_seed_full(bg, rcfg, seg_len, args, seed):
    """Build the scene once (dual == solo frames), run all 4 cells x dual/solo."""
    jr = np.random.default_rng(seed + 100000)
    noise_std = 10.0
    if args.jitter > 0:
        noise_std = 10.0 * jr.uniform(1 - args.jitter, 1 + args.jitter)
    recol_hue = args.recol_hue
    if args.corrupt == "recolor" and args.recol_jitter > 0:
        jr2 = np.random.default_rng(seed + 500000)
        recol_hue = args.recol_hue + jr2.uniform(-args.recol_jitter, args.recol_jitter)
    frames, masks, segs, gts = build_dual_scene(bg, rcfg, seg_len, seed,
                                                corrupt=args.corrupt,
                                                noise_std=noise_std,
                                                recol_hue=recol_hue)
    targets_dual = [(rcfg["t1_hue"], "T1"), (rcfg["t2_hue"], "T2")]
    targets_solo = [(rcfg["t2_hue"], "T2")]
    cells = {}
    for cell in CELLS:
        kw = CELL_KW[cell]
        mind = MultiTargetMind([rcfg["t1_hue"], rcfg["t2_hue"]], thr_base=args.thr, **kw)
        oks, seg_r, trace, threxc, fgf, nc, no = run_dual(mind, frames, masks, segs,
                                                          targets_dual)
        dual = pack_targets(oks, seg_r, trace, threxc, fgf, gts, segs,
                            targets_dual, mind, no, nc)
        mind2 = MultiTargetMind([rcfg["t2_hue"]], thr_base=args.thr, **kw)
        oks2, seg_r2, trace2, threxc2, fgf2, nc2, no2 = run_dual(mind2, frames, masks,
                                                                 segs, targets_solo)
        solo = pack_targets(oks2, seg_r2, trace2, threxc2, fgf2, gts, segs,
                            targets_solo, mind2, no2, nc2)
        cells[cell] = {"dual": dual, "solo": solo}
    gt = {"T1": {"n_pos": int(sum(g["T1"] for g in gts)), "n_total": len(gts)},
          "T2": {"n_pos": int(sum(g["T2"] for g in gts)), "n_total": len(gts)}}
    return cells, gt, float(noise_std), float(recol_hue)


# ---- probe (calibration + timing + sanity) ----


def gain_drift_probe(bg_frame, rcfg, rng):
    fr, m = overlay_patch(bg_frame, t1_box_frac(bg_frame.shape),
                          rcfg["t1_hue"], rcfg["t1_sig"], rng)
    out = {}
    for kc in (0.05, 0.10, 0.15, 0.20, 0.25):
        g = np.array([1 + 0.9 * kc, 1.0, 1 - 0.9 * kc])
        img = fr.astype(np.float32)
        img[m > 0] *= g
        img = np.clip(img, 0, 255).astype(np.uint8)
        frac, hue = ds.target_obs(img, m, rcfg["t1_hue"])
        out[kc] = (hue, frac)
    return out


def probe(args):
    t0 = time.time()
    bg = ds.load_video(args.bg_video)[0]
    H, W = bg[0].shape[:2]
    print("R_MT_PROBE_BG=" + args.bg_video)
    print("R_MT_PROBE_FRAME_W=%d" % W)
    print("R_MT_PROBE_FRAME_H=%d" % H)
    for rname, rcfg in REGIMES.items():
        rng = np.random.default_rng(7)
        fr0 = bg[0]
        fr1, m1 = overlay_patch(fr0, t1_box_frac(fr0.shape), rcfg["t1_hue"],
                                rcfg["t1_sig"], rng)
        frac1, hue1 = ds.target_obs(fr1, m1, rcfg["t1_hue"])
        fr2, m2 = overlay_patch(fr1, t2_box_frac(fr0.shape), rcfg["t2_hue"],
                                rcfg["t2_sig"], rng)
        frac2, hue2 = ds.target_obs(fr2, m2, rcfg["t2_hue"])
        print("R_MT_PROBE_REGIME=" + rname)
        print("R_MT_PROBE_T1_HUE=%.3f" % (hue1 if hue1 is not None else -1))
        print("R_MT_PROBE_T1_FRAC=%.4f" % frac1)
        print("R_MT_PROBE_T2_HUE=%.3f" % (hue2 if hue2 is not None else -1))
        print("R_MT_PROBE_T2_FRAC=%.4f" % frac2)
        gd = gain_drift_probe(fr0, rcfg, rng)
        for kc in sorted(gd):
            h_, f_ = gd[kc]
            print("R_MT_PROBE_GAINDRIFT_KC%.2f=%.3f" % (kc, h_ if h_ is not None else -1))
            print("R_MT_PROBE_GAINFRAC_KC%.2f=%.4f" % (kc, f_))
        rng2 = np.random.default_rng(7)
        frr, _ = overlay_patch(fr0, t1_box_frac(fr0.shape), args.recol_hue,
                               rcfg["t1_sig"], rng2)
        _, hrec = ds.target_obs(frr, m1, args.recol_hue)
        print("R_MT_PROBE_RECOL_HUE=%.3f" % (hrec if hrec is not None else -1))
        print("R_MT_PROBE_RECOL_CIRC_T1=%.3f" % ds.circ(hrec, rcfg["t1_hue"]))
        print("R_MT_PROBE_RECOL_CIRC_T2=%.3f" % ds.circ(hrec, rcfg["t2_hue"]))
    rcfg = REGIMES["near"]
    seg_len = dict(zip(["base", "dusk", "corrupt", "rest"],
                       [int(x) for x in args.seg.split(",")]))
    t1 = time.time()
    cells, gt, _, _ = run_seed_full(bg, rcfg, seg_len, args, 7)
    print("R_MT_PROBE_TIMING_ONE_SEED=%.2f" % (time.time() - t1))
    a = cells["A"]
    sh = a["dual"]["_shared"]
    print("R_MT_PROBE_GATE_NOPEN=%d" % sh["n_open"])
    print("R_MT_PROBE_GATE_NCORR=%d" % sh["n_corr"])
    t2 = a["dual"]["T2"]
    print("R_MT_PROBE_DUAL_T2_BASE=%.4f" % t2["seg_rates"]["base"])
    print("R_MT_PROBE_DUAL_T2_DUSK=%.4f" % t2["seg_rates"]["dusk"])
    print("R_MT_PROBE_DUAL_T2_CORR=%.4f" % t2["seg_rates"]["corrupt"])
    print("R_MT_PROBE_DUAL_T2_REST=%.4f" % t2["seg_rates"]["rest"])
    s2 = a["solo"]["T2"]
    print("R_MT_PROBE_SOLO_T2_BASE=%.4f" % s2["seg_rates"]["base"])
    print("R_MT_PROBE_SOLO_T2_CORR=%.4f" % s2["seg_rates"]["corrupt"])
    print("R_MT_PROBE_SOLO_T2_REST=%.4f" % s2["seg_rates"]["rest"])
    print("R_MT_PROBE_DUAL_T1_F1=%.4f" % a["dual"]["T1"]["f1"])
    print("R_MT_PROBE_DUAL_T2_F1=%.4f" % t2["f1"])
    print("R_MT_PROBE_SOLO_T2_F1=%.4f" % s2["f1"])
    print("R_MT_PROBE_DUAL_T2_THREXC_MEAN=%.4f" % t2["threxc_mean"])
    print("R_MT_PROBE_DUAL_T2_FOREIGNFRAC=%.4f" % t2["foreign_frac"])
    print("R_MT_PROBE_ELAPSED=%.2f" % (time.time() - t0))
    return 0


# ---- full run ----


def agg_seed(cells, seeds, rcfg):
    """Aggregate per-seed dicts into mean_sd structures + T2 dual-solo deltas."""
    targets_by_mode = {"dual": ["T1", "T2"], "solo": ["T2"]}
    out = {}
    for cell in CELLS:
        cell_out = {}
        for mode in ("dual", "solo"):
            for key in targets_by_mode[mode]:
                series = {}
                for metric in ("f1", "p", "r", "flips"):
                    vals = [cells[str(s)]["cells"][cell][mode][key][metric]
                            for s in seeds]
                    series[metric] = list(mean_sd(vals))
                for sname in ("base", "dusk", "corrupt", "rest"):
                    vals = [cells[str(s)]["cells"][cell][mode][key]
                            ["seg_rates"][sname] for s in seeds]
                    series["seg_" + sname] = list(mean_sd(vals))
                for metric in ("threxc_mean", "threxc_max", "foreign_frac",
                               "k_max", "n_rej", "rej_total"):
                    vals = [cells[str(s)]["cells"][cell][mode][key][metric]
                            for s in seeds]
                    series[metric] = list(mean_sd(vals))
                cell_out.setdefault(mode, {})[key] = series
        sh_dual = [cells[str(s)]["cells"][cell]["dual"]["_shared"]["n_open"]
                   for s in seeds]
        sh_corr = [cells[str(s)]["cells"][cell]["dual"]["_shared"]["n_corr"]
                   for s in seeds]
        cell_out["n_open"] = list(mean_sd(sh_dual))
        cell_out["n_corr"] = list(mean_sd(sh_corr))
        # T2 pollution deltas (dual - solo), paired per seed
        poll = {}
        for metric in ("f1",):
            vals = [cells[str(s)]["cells"][cell]["dual"]["T2"]["f1"] -
                    cells[str(s)]["cells"][cell]["solo"]["T2"]["f1"] for s in seeds]
            poll["delta_f1"] = list(mean_sd(vals))
        for sname in ("corrupt", "rest", "dusk", "base"):
            vals = [cells[str(s)]["cells"][cell]["dual"]["T2"]
                    ["seg_rates"][sname] -
                    cells[str(s)]["cells"][cell]["solo"]["T2"]
                    ["seg_rates"][sname] for s in seeds]
            poll["delta_seg_" + sname] = list(mean_sd(vals))
        cell_out["pollution_T2"] = poll
        out[cell] = cell_out
    return out


def emit(key, val):
    if isinstance(val, float):
        print("R_MT_%s=%.4f" % (key, val))
    else:
        print("R_MT_%s=%s" % (key, val))


def print_summary(agg, args, rcfg, gt, seeds, recol_hue0):
    emit("REGIME", args.regime.upper())
    emit("BG", args.bg_video)
    emit("CORRUPT", args.corrupt)
    emit("T1HUE", rcfg["t1_hue"])
    emit("T2HUE", rcfg["t2_hue"])
    emit("RECOL", recol_hue0)
    emit("THR", args.thr)
    emit("SEEDS", len(seeds))
    emit("JITTER", args.jitter)
    emit("GT_T1_POS", gt["T1"]["n_pos"])
    emit("GT_T1_TOTAL", gt["T1"]["n_total"])
    emit("GT_T2_POS", gt["T2"]["n_pos"])
    emit("GT_T2_TOTAL", gt["T2"]["n_total"])
    for cell in CELLS:
        emit("CELL", cell)
        c = agg[cell]
        for key in ("T1", "T2"):
            s = c["dual"][key]
            emit("DUAL_%s_F1_MEAN" % key, s["f1"][0])
            emit("DUAL_%s_F1_SD" % key, s["f1"][1])
            emit("DUAL_%s_P_MEAN" % key, s["p"][0])
            emit("DUAL_%s_R_MEAN" % key, s["r"][0])
            emit("DUAL_%s_FLIPS_MEAN" % key, s["flips"][0])
            emit("DUAL_%s_CORR_MEAN" % key, s["seg_corrupt"][0])
            emit("DUAL_%s_CORR_SD" % key, s["seg_corrupt"][1])
            emit("DUAL_%s_REST_MEAN" % key, s["seg_rest"][0])
            emit("DUAL_%s_REST_SD" % key, s["seg_rest"][1])
            emit("DUAL_%s_DUSK_MEAN" % key, s["seg_dusk"][0])
            emit("DUAL_%s_DUSK_SD" % key, s["seg_dusk"][1])
        s2 = c["dual"]["T2"]
        emit("DUAL_T2_THREXC_MEAN", s2["threxc_mean"][0])
        emit("DUAL_T2_THREXC_MAX", s2["threxc_max"][0])
        emit("DUAL_T2_FOREIGNFRAC", s2["foreign_frac"][0])
        emit("DUAL_T2_KMAX", s2["k_max"][0])
        emit("DUAL_T2_REJTOT", s2["rej_total"][0])
        so = c["solo"]["T2"]
        emit("SOLO_T2_F1_MEAN", so["f1"][0])
        emit("SOLO_T2_F1_SD", so["f1"][1])
        emit("SOLO_T2_CORR_MEAN", so["seg_corrupt"][0])
        emit("SOLO_T2_CORR_SD", so["seg_corrupt"][1])
        emit("SOLO_T2_REST_MEAN", so["seg_rest"][0])
        emit("SOLO_T2_REST_SD", so["seg_rest"][1])
        p = c["pollution_T2"]
        emit("DELTA_T2_F1_MEAN", p["delta_f1"][0])
        emit("DELTA_T2_F1_SD", p["delta_f1"][1])
        emit("DELTA_T2_CORR_MEAN", p["delta_seg_corrupt"][0])
        emit("DELTA_T2_CORR_SD", p["delta_seg_corrupt"][1])
        emit("DELTA_T2_REST_MEAN", p["delta_seg_rest"][0])
        emit("DELTA_T2_REST_SD", p["delta_seg_rest"][1])
        emit("DELTA_T2_DUSK_MEAN", p["delta_seg_dusk"][0])
        emit("DELTA_T2_DUSK_SD", p["delta_seg_dusk"][1])
        emit("NOPEN_MEAN", c["n_open"][0])
        emit("NCORR_MEAN", c["n_corr"][0])


def run_regime(args):
    t0 = time.time()
    rcfg = REGIMES[args.regime]
    bg = ds.load_video(args.bg_video)[0]
    seg_len = dict(zip(["base", "dusk", "corrupt", "rest"],
                       [int(x) for x in args.seg.split(",")]))
    seeds = list(range(args.n_seeds))
    cfg = {"regime": args.regime, "bg": args.bg_video, "corrupt": args.corrupt,
           "thr": args.thr, "seg": seg_len, "n_seeds": args.n_seeds,
           "jitter": args.jitter, "recol_hue": args.recol_hue,
           "recol_jitter": args.recol_jitter,
           "t1_hue": rcfg["t1_hue"], "t2_hue": rcfg["t2_hue"],
           "t1_sig": rcfg["t1_sig"], "t2_sig": rcfg["t2_sig"]}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_mt_%s_%s.json" % (args.regime, ck_tag))
    per_seed = {}
    if not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            per_seed = json.load(f).get("per_seed", {})
    for seed in seeds:
        if str(seed) in per_seed:
            continue
        cells, gt, noise_std, recol_hue = run_seed_full(bg, rcfg, seg_len, args, seed)
        per_seed[str(seed)] = {"seed": seed, "gt": gt, "cells": cells,
                               "noise_std": round(noise_std, 3),
                               "recol_hue": round(recol_hue, 3)}
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"regime": args.regime, "config": cfg,
                       "per_seed": per_seed}, f, ensure_ascii=False, indent=1)
        print("PROGRESS", flush=True)
    gt0 = per_seed[str(seeds[0])]["gt"]
    agg = agg_seed(per_seed, seeds, rcfg)
    recol0 = float(np.mean([per_seed[str(s)]["recol_hue"] for s in seeds]))

    out = {
        "artifact": "multi_target_test",
        "doc_ref": "docs/233",
        "regime": args.regime,
        "config": cfg,
        "gt": gt0,
        "cells": agg,
        "per_seed": {s: {"gt": per_seed[str(s)]["gt"],
                         "noise_std": per_seed[str(s)]["noise_std"],
                         "recol_hue": per_seed[str(s)]["recol_hue"],
                         "cells": {c: per_seed[str(s)]["cells"][c] for c in CELLS}}
                     for s in seeds},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir,
                            "mt_%s%s.json" % (args.regime,
                                              "_" + args.tag if args.tag else ""))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print_summary(agg, args, rcfg, gt0, seeds, recol0)
    print("R_MT_ELAPSED=%.2f" % (time.time() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="near", choices=["near", "far"])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--bg-video", default="blackswan")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--corrupt", default="recolor", choices=["recolor", "gain"])
    ap.add_argument("--thr", type=float, default=0.60)
    ap.add_argument("--recol-hue", type=float, default=RECOL_BASE)
    ap.add_argument("--recol-jitter", type=float, default=4.0)
    ap.add_argument("--seg", default="20,20,15,25")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.probe:
        return probe(args)
    return run_regime(args)


if __name__ == "__main__":
    sys.exit(main())
