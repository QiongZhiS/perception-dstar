"""vision/davis_a4_stats.py - docs/221 A4 execution shell (unified injection + 199b gate merge).

Built ON TOP of the A2 stats shell (vision/davis_a2_stats.py); the A2 shell and
vision/davis_suspicious.py are NOT modified. Adds:

  A4-1 unified injection protocol: every target video is run under BOTH
       corrupt=gain and corrupt=noise (10 seeds, jitter 0.25, thr auto rule
       thr = clip(median_frac - 0.30, 0.30, 0.60)). Core videos keep the
       docs/219 protocol as compat rows (flamingo: gain, thr 0.60, seg
       20,20,15,25; surf: noise, thr 0.40, seg 14,12,10,19).
  A4-2 199b temporal gate merged into the stats shell:
       (a) main-sequence gate stats (gate-open frames, n_corr, dusk pass
           rate, corrupt pass rate) under the three gate sets
           g408  = TemporalGateDavis defaults (40/4/8, davis_suspicious.py:201)
           g20312= docs/199b spec (20/3/12, color_constancy_temporal.py:93)
           g30310= docs/229 s3.3 midpoint (30/3/10)
       (b) 199b-style gate probe: real DAVIS background + synthetic red ball
           + 4 gain perturbations (baseline/dusk/noise/step), raw_shift_frac
           with sat_min=100 and fixed exclude band (davis_constancy.py
           semantics). Reproduces docs/199b table for the original 6 videos
           and extends it to the new targets.
  A5 archive: per-seed checkpoints + per-config JSON in vision/out/results.

Safety discipline: stdout prints ASCII labels only, one number per line
(external regex extraction consumes them).

Usage:
  python vision/davis_a4_stats.py --video bear --corrupt gain
  python vision/davis_a4_stats.py --video bear --corrupt noise
  python vision/davis_a4_stats.py --video blackswan --corrupt gain --probe-only
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
from davis_a2_stats import make_mind, mean_sd, REF_SEED, JITTER  # noqa: E402

OUT_DIR = os.path.join("vision", "out", "results")

# docs/229 s3.3 gate triplets: g408 = code status quo (davis_suspicious.py:201),
# g20312 = docs/199b spec (color_constancy_temporal.py:93), g30310 = midpoint.
GATE_SETS = {
    "g408": dict(mad_gate=40.0, warmup=8, delta_gate=4.0),
    "g20312": dict(mad_gate=20.0, warmup=12, delta_gate=3.0),
    "g30310": dict(mad_gate=30.0, warmup=10, delta_gate=3.0),
}

# docs/219 core protocols (kept verbatim for compatibility rows).
CORE_PROTO = {
    "flamingo": dict(seg=(20, 20, 15, 25), thr=0.60),
    "surf": dict(seg=(14, 12, 10, 19), thr=0.40),
}

# 199b probe constants (davis_constancy.py same semantics).
BX0, BX1, BY0, BY1 = 40, 460, 130, 270
N_PROBE = 60


# ---- 199b probe (self-contained copy of davis_constancy.py semantics) ----


def raw_shift_frac_199b(fr, sat_min=100):
    """Background hue median + colored fraction + circular MAD (199b semantics)."""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    colored = (s > sat_min).astype(np.uint8)
    colored[BY0:BY1, BX0:BX1] = 0
    frac = float(colored.mean())
    if colored.sum() < 100:
        return None, frac, float("nan")
    vals = h[colored > 0].astype(float)
    med = float(np.median(vals))
    d = np.abs(vals - med)
    mad = float(np.median(np.minimum(d, 180.0 - d)))
    return med, frac, mad


def warm_gain_199b(k, sat=0.5):
    """Warm gain as in davis_constancy.py (sat=0.5, blackswan-calibrated)."""
    return np.array([1 - sat * k, 1.0, 1 + sat * 0.6 * k])


def overlay_ball(fr, i, color=(0, 0, 200)):
    img = fr.copy()
    cv2.circle(img, (int(80 + 4.0 * i), int(180 + 0.6 * i)), 18, color, -1)
    return img


def load_frames_sampled(video, n=N_PROBE):
    vdir = os.path.join(ds.DAVIS, video)
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    n = min(n, len(jpgs))
    idx = np.linspace(0, len(jpgs) - 1, n).astype(int)
    return [cv2.imread(os.path.join(vdir, jpgs[i])) for i in idx]


def gate_probe_199b(video, gate_params, seed=42):
    """199b P1 probe: gate-open frames under baseline/dusk/noise/step (of 60)."""
    frames = load_frames_sampled(video)
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("baseline", "dusk", "noise", "step"):
        gate = ds.TemporalGateDavis(**gate_params)
        opens = 0
        for i, fr in enumerate(frames):
            t = i / max(len(frames) - 1, 1)
            if kind == "baseline":
                gk = np.ones(3)
            elif kind == "dusk":
                gk = warm_gain_199b(t)
            elif kind == "noise":
                gk = warm_gain_199b(rng.uniform(0, 1))
            else:
                gk = warm_gain_199b(1.0 if i >= len(frames) // 2 else 0.0)
            img = np.clip(fr.astype(np.float32) * gk, 0, 255).astype(np.uint8)
            img = overlay_ball(img, i)
            s, _, mad = raw_shift_frac_199b(img)
            if gate.update(s, mad):
                opens += 1
        out[kind] = opens
    return out


# ---- main-sequence runs under a gate set ----


def run_seed(frames, masks, seg_len, task_hue, thr, corrupt, seed, gate_params,
             with_d, jitter):
    """Build sequence (seed) + run A under gate_params (+ D template)."""
    if jitter > 0:
        sr = np.random.default_rng(seed + 100000)
        noise_std = 10.0 * sr.uniform(1 - jitter, 1 + jitter)
        corrupt_kc = 0.05 * sr.uniform(1 - jitter, 1 + jitter)
    else:
        noise_std, corrupt_kc = 10.0, 0.05
    seq_f, seq_m, seq_s, seq_gt = ds.build_sequence(
        frames, masks, seg_len, seed=seed, corrupt=corrupt,
        noise_std=noise_std, corrupt_kc=corrupt_kc)
    res = {}
    mind = make_mind(task_hue, thr, "A")
    mind.gate = ds.TemporalGateDavis(**gate_params)
    oks, seg_r, corr, trace = ds.run_mind(mind, seq_f, seq_m, seq_s, task_hue)
    m = ds.metrics(oks, seq_gt)
    n_open = int(sum(1 for e in trace if e[8] is not None and e[8][0]))
    res["A"] = {"p": float(m["p"]), "r": float(m["r"]), "f1": float(m["f1"]),
                "flips": int(ds.flips(oks, seq_s)),
                "seg_rates": {k: round(float(v), 4) for k, v in seg_r.items()},
                "n_corr": int(corr), "n_open": n_open,
                "noise_std": round(float(noise_std), 3),
                "corrupt_kc": round(float(corrupt_kc), 4),
                "trace": ds.serialize_trace(trace)}
    if with_d:
        mind_d = make_mind(task_hue, thr, "D")
        oks_d, seg_r_d, corr_d, trace_d = ds.run_mind(mind_d, seq_f, seq_m, seq_s, task_hue)
        md = ds.metrics(oks_d, seq_gt)
        res["D"] = {"p": float(md["p"]), "r": float(md["r"]), "f1": float(md["f1"]),
                    "flips": int(ds.flips(oks_d, seq_s)),
                    "seg_rates": {k: round(float(v), 4) for k, v in seg_r_d.items()},
                    "n_corr": int(corr_d), "n_open": 0,
                    "trace": ds.serialize_trace(trace_d)}
    gt = {"n_pos": int(sum(seq_gt)), "n_total": len(seq_gt),
          "n_empty": int(sum(1 for mm in seq_m if mm.max() == 0))}
    return res, gt


def agg(vals):
    m, sd = mean_sd(vals)
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--corrupt", default="gain", choices=["gain", "noise"])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--seg", default=None)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--gate-sets", default="g408,g20312,g30310")
    ap.add_argument("--no-d", action="store_true")
    ap.add_argument("--no-refcheck", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--probe-seed", type=int, default=42)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gate_keys = [g for g in args.gate_sets.split(",") if g in GATE_SETS]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.probe_only:
        for gk in gate_keys:
            p = gate_probe_199b(args.video, GATE_SETS[gk], seed=args.probe_seed)
            print("R_A4_VIDEO=" + args.video)
            print("R_A4_MODE=probe")
            print("R_A4_GATE=" + gk)
            for k in ("baseline", "dusk", "noise", "step"):
                print("R_A4_PROBE_%s=%d" % (k.upper(), p[k]))
        return 0

    t0 = time.time()
    frames, masks = ds.load_video(args.video)
    obs = [ds.target_obs(f, m, 0.0)[1] for f, m in zip(frames, masks)]
    obs = [h for h in obs if h is not None]
    if not obs:
        print("R_A4_VIDEO=" + args.video)
        print("R_A4_ERROR=NO_COLORED_TARGET")
        return 2
    task_hue = float(np.median(obs))
    fracs = [ds.target_obs(f, m, task_hue)[0] for f, m in zip(frames, masks)
             if ds.target_obs(f, m, task_hue)[1] is not None]
    median_frac = float(np.median(fracs))
    if args.thr is not None:
        thr = float(args.thr)
    elif args.video in CORE_PROTO:
        thr = float(CORE_PROTO[args.video]["thr"])
    else:
        thr = float(min(0.60, max(0.30, round(median_frac - 0.30, 1))))
    if args.seg:
        seg_len = dict(zip(["base", "dusk", "corrupt", "rest"],
                           [int(x) for x in args.seg.split(",")]))
    elif args.video in CORE_PROTO:
        seg_len = dict(zip(["base", "dusk", "corrupt", "rest"],
                           CORE_PROTO[args.video]["seg"]))
    else:
        seg_len = dict(zip(["base", "dusk", "corrupt", "rest"], [20, 20, 15, 25]))

    seeds = list(range(args.n_seeds))
    cfg = {"seg": seg_len, "thr": thr, "corrupt": args.corrupt,
           "task_hue": task_hue, "jitter": args.jitter, "gate_sets": gate_keys,
           "n_seeds": args.n_seeds}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir,
                             "ckpt_a4_%s_%s_%s.json" % (args.video, args.corrupt, ck_tag))
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_seed", {})

    per_seed = dict(done)
    for seed in seeds:
        if str(seed) in per_seed:
            continue
        sd = {}
        for gk in gate_keys:
            res, gt = run_seed(frames, masks, seg_len, task_hue, thr, args.corrupt,
                               seed, GATE_SETS[gk], not args.no_d, args.jitter)
            sd[gk] = res["A"]
            if not args.no_d:
                sd["D"] = res["D"]
        per_seed[str(seed)] = {"seed": seed, "gt": gt, "variants": sd}
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"video": args.video, "config": cfg, "per_seed": per_seed},
                      f, ensure_ascii=False, indent=1)
        print("PROGRESS", flush=True)

    gt0 = per_seed[str(seeds[0])]["gt"]
    gate_out = {}
    for gk in gate_keys:
        f1s = [per_seed[str(s)]["variants"][gk]["f1"] for s in seeds]
        ps = [per_seed[str(s)]["variants"][gk]["p"] for s in seeds]
        rs = [per_seed[str(s)]["variants"][gk]["r"] for s in seeds]
        fl = [per_seed[str(s)]["variants"][gk]["flips"] for s in seeds]
        no = [per_seed[str(s)]["variants"][gk]["n_open"] for s in seeds]
        nc = [per_seed[str(s)]["variants"][gk]["n_corr"] for s in seeds]
        du = [per_seed[str(s)]["variants"][gk]["seg_rates"]["dusk"] for s in seeds]
        cp = [per_seed[str(s)]["variants"][gk]["seg_rates"]["corrupt"] for s in seeds]
        gate_out[gk] = {
            "per_seed": [{**per_seed[str(s)]["variants"][gk], "seed": s} for s in seeds],
            "mean_sd": {"f1": list(agg(f1s)), "p": list(agg(ps)), "r": list(agg(rs)),
                        "flips": list(agg(fl)), "n_open": list(agg(no)),
                        "n_corr": list(agg(nc)), "dusk": list(agg(du)),
                        "corrupt_pass": list(agg(cp))},
        }
    d_out = None
    if not args.no_d:
        f1s = [per_seed[str(s)]["variants"]["D"]["f1"] for s in seeds]
        ps = [per_seed[str(s)]["variants"]["D"]["p"] for s in seeds]
        rs = [per_seed[str(s)]["variants"]["D"]["r"] for s in seeds]
        fl = [per_seed[str(s)]["variants"]["D"]["flips"] for s in seeds]
        d_out = {"mean_sd": {"f1": list(agg(f1s)), "p": list(agg(ps)),
                             "r": list(agg(rs)), "flips": list(agg(fl))}}

    ref_out = {}
    if not args.no_refcheck:
        for gk in gate_keys:
            res, _ = run_seed(frames, masks, seg_len, task_hue, thr, args.corrupt,
                              REF_SEED, GATE_SETS[gk], False, 0.0)
            ref_out[gk] = {"p": round(res["A"]["p"], 4), "r": round(res["A"]["r"], 4),
                           "f1": round(res["A"]["f1"], 4)}

    probe_out = {}
    if not args.no_probe:
        for gk in gate_keys:
            probe_out[gk] = gate_probe_199b(args.video, GATE_SETS[gk],
                                            seed=args.probe_seed)

    out = {
        "artifact": "davis_a4_stats",
        "doc_ref": "docs/230",
        "video": args.video,
        "config": {"seg": seg_len, "thr": thr, "corrupt": args.corrupt,
                   "task_hue": round(task_hue, 3),
                   "median_frac": round(median_frac, 3),
                   "n_seeds": args.n_seeds, "seeds": seeds, "jitter": args.jitter,
                   "gate_sets": gate_keys, "tag": args.tag, "ck_tag": ck_tag},
        "gt": gt0,
        "gates": gate_out,
        "variant_d": d_out,
        "refcheck_seed7_jitter0": ref_out,
        "probe_199b": probe_out,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir,
                            "a4_%s_%s%s.json" % (args.video, args.corrupt,
                                                  "_" + args.tag if args.tag else ""))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- ASCII summary block (one number per line, fixed order) ----
    print("R_A4_VIDEO=" + args.video)
    print("R_A4_CORRUPT=" + args.corrupt)
    print("R_A4_THR=%.3f" % thr)
    print("R_A4_TASKHUE=%.3f" % task_hue)
    print("R_A4_MEDFRAC=%.3f" % median_frac)
    print("R_A4_SEEDS=%d" % len(seeds))
    print("R_A4_SEG_B=%d" % seg_len["base"])
    print("R_A4_SEG_D=%d" % seg_len["dusk"])
    print("R_A4_SEG_C=%d" % seg_len["corrupt"])
    print("R_A4_SEG_R=%d" % seg_len["rest"])
    print("R_A4_GT_NPOS=%d" % gt0["n_pos"])
    print("R_A4_GT_NTOTAL=%d" % gt0["n_total"])
    print("R_A4_GT_NEMPTY=%d" % gt0["n_empty"])
    for gk in gate_keys:
        g = gate_out[gk]["mean_sd"]
        print("R_A4_GATE=" + gk)
        print("R_A4_F1_MEAN=%.4f" % g["f1"][0])
        print("R_A4_F1_SD=%.4f" % g["f1"][1])
        print("R_A4_P_MEAN=%.4f" % g["p"][0])
        print("R_A4_P_SD=%.4f" % g["p"][1])
        print("R_A4_R_MEAN=%.4f" % g["r"][0])
        print("R_A4_R_SD=%.4f" % g["r"][1])
        print("R_A4_FLIPS_MEAN=%.4f" % g["flips"][0])
        print("R_A4_FLIPS_SD=%.4f" % g["flips"][1])
        print("R_A4_NOPEN_MEAN=%.4f" % g["n_open"][0])
        print("R_A4_NOPEN_SD=%.4f" % g["n_open"][1])
        print("R_A4_NCORR_MEAN=%.4f" % g["n_corr"][0])
        print("R_A4_NCORR_SD=%.4f" % g["n_corr"][1])
        print("R_A4_DUSK_MEAN=%.4f" % g["dusk"][0])
        print("R_A4_DUSK_SD=%.4f" % g["dusk"][1])
        print("R_A4_CORRUPTPASS_MEAN=%.4f" % g["corrupt_pass"][0])
        print("R_A4_CORRUPTPASS_SD=%.4f" % g["corrupt_pass"][1])
        if ref_out:
            print("R_A4_REF_F1=%.4f" % ref_out[gk]["f1"])
            print("R_A4_REF_P=%.4f" % ref_out[gk]["p"])
            print("R_A4_REF_R=%.4f" % ref_out[gk]["r"])
        if probe_out:
            for k in ("baseline", "dusk", "noise", "step"):
                print("R_A4_PROBE_%s=%d" % (k.upper(), probe_out[gk][k]))
    if d_out:
        print("R_A4_D_F1_MEAN=%.4f" % d_out["mean_sd"]["f1"][0])
        print("R_A4_D_F1_SD=%.4f" % d_out["mean_sd"]["f1"][1])
        print("R_A4_D_P_MEAN=%.4f" % d_out["mean_sd"]["p"][0])
        print("R_A4_D_P_SD=%.4f" % d_out["mean_sd"]["p"][1])
        print("R_A4_D_R_MEAN=%.4f" % d_out["mean_sd"]["r"][0])
        print("R_A4_D_R_SD=%.4f" % d_out["mean_sd"]["r"][1])
        print("R_A4_D_FLIPS_MEAN=%.4f" % d_out["mean_sd"]["flips"][0])
        print("R_A4_D_FLIPS_SD=%.4f" % d_out["mean_sd"]["flips"][1])
    print("R_A4_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
