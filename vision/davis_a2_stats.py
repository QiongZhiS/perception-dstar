"""vision/davis_a2_stats.py — docs/221 A2+A5 统计外壳：多种子 + thr 扫描 + flips 重采样 + JSON 归档。

在 vision/davis_suspicious.py 单次运行之上加统计层（机制不动，纯循环外壳，向后兼容）：

  A2-多种子   >=10 个随机种子重跑序列构造，P/R/F1 与 flips 报均值±SD（种子间方差 =
              噪声/扰动抽样方差；--jitter>0 时强度也按种子采样，见 build_sequence docstring
              "扰动强度参数化"——默认 ±25% 对称乘性抖动，参考种子可关）。
  A2-thr扫描  关键确认阈值扫若干点，每点报 P/R/F1 操作点（PR 曲线）。默认用参考种子
              seed 7 + jitter 0 → thr=0.60 处与 docs/219 数字严格一致。
  A2-flips检验 纪念系统(A full) vs 朴素模板(D) 的 rest 段翻转：配对符号翻转精确置换检验
              （2^n 全枚举，双侧 p）+ 配对 bootstrap 95% CI（均值差 + 各自均值）。
  A5-归档     每个 (视频,种子,变体) 的逐帧 trace + 指标写入 vision/out/results/a2_<video>.json；
              每完成一个种子写 checkpoint（ckpt_<video>.json，可 --resume 断点续跑）。

安全纪律（本仓库）：脚本自身不打印任何非摘要数字；最后只打印 ASCII 标签 + 每行恰一个
数字的摘要块（外部用正则抽取数字，标签顺序即数字顺序）。摘要行顺序固定如下：

  R_VIDEO=<video>          （无数字）
  R_SEEDS=<n>              1
  R_THR=<thr>              2
  R_TASKHUE=<hue>          3
  R_GT_NPOS=<n>            4
  R_GT_NTOTAL=<n>          5
  R_GT_NEMPTY=<n>          6
  R_FA_MEAN=<f1>           7
  R_FA_SD=<f1>             8
  R_PA_MEAN=<p>            9
  R_PA_SD=<p>              10
  R_RA_MEAN=<r>            11
  R_RA_SD=<r>              12
  R_FLIPA_MEAN=<flips>     13
  R_FLIPA_SD=<flips>       14
  R_F1A_CI_LO=<ci>         15
  R_F1A_CI_HI=<ci>         16
  R_FLIPA_CI_LO=<ci>       17
  R_FLIPA_CI_HI=<ci>       18
  R_FLIPD_MEAN=<flips>     19
  R_FLIPD_SD=<flips>       20
  R_FLIPDIFF_MEAN=<diff>   21
  R_FLIPDIFF_CI_LO=<ci>    22
  R_FLIPDIFF_CI_HI=<ci>    23
  R_FLIPP_PVAL=<p>         24
  （若 --thr-scan：每点 6 行，点在列表内按序）
  R_SCAN_THR=<thr>         25+k*6
  R_SCAN_P_P=<p>
  R_SCAN_P_R=<r>
  R_SCAN_P_F1=<f1>
  R_SCAN_D_P=<p>
  R_SCAN_D_F1=<f1>
  R_ELAPSED=<sec>          最后一行

用法：
  python vision/davis_a2_stats.py --video flamingo --n-seeds 10 --thr 0.60 \
      --corrupt gain --seg 20,20,15,25 --variants A,B,C,C2,D,E,F \
      --thr-scan 0.25,0.35,0.45,0.55,0.65,0.75,0.85 --scan-variants A,D --jitter 0.25
  python vision/davis_a2_stats.py --video surf --n-seeds 10 --thr 0.40 \
      --corrupt noise --seg 14,12,10,19 --variants A,B,C,C2,D,E,F --thr-scan 0.20,0.30,0.40,0.50,0.60
  python vision/davis_a2_stats.py --video bear --n-seeds 10 --thr auto --variants A,D
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "vision")
import davis_suspicious as ds  # noqa: E402

DEFAULT_OUT = os.path.join("vision", "out", "results")
REF_SEED = 7          # docs/219 参考种子（该种子 + jitter=0 复现旧数字）
JITTER = 0.25         # 扰动强度按种子采样幅度（±25% 对称乘性，docs/221 A2 意图）
N_BOOT = 5000
BOOT_SEED = 20260828

VARIANTS = {
    "A": dict(wb=True),
    "B": dict(wb=False),
    "C": dict(global_thr=True),
    "C2": dict(global_thr=True, wb=False),
    "D": dict(template=True, wb=False),
    "E": dict(wb=False, fixed_bw=10.0),
    "F": dict(wb=False, fixed_bw=30.0),
}


def make_mind(task_hue, thr, key):
    return ds.DavisMind(task_hue, thr_base=thr, **VARIANTS[key])


def run_seed(frames, masks, seg_len, task_hue, thr, corrupt, seed, keys, jitter):
    """构造序列（seed 定噪声）+ 跑变体。返回 (per_variant dict, gt dict)。"""
    if jitter > 0:
        sr = np.random.default_rng(seed + 100000)      # 独立流：强度采样与序列噪声解耦
        noise_std = 10.0 * sr.uniform(1 - jitter, 1 + jitter)
        corrupt_kc = 0.05 * sr.uniform(1 - jitter, 1 + jitter)
    else:
        noise_std, corrupt_kc = 10.0, 0.05
    seq_f, seq_m, seq_s, seq_gt = ds.build_sequence(
        frames, masks, seg_len, seed=seed, corrupt=corrupt,
        noise_std=noise_std, corrupt_kc=corrupt_kc)
    res = {}
    for key in keys:
        mind = make_mind(task_hue, thr, key)
        oks, seg_r, corr, trace = ds.run_mind(mind, seq_f, seq_m, seq_s, task_hue)
        m = ds.metrics(oks, seq_gt)
        res[key] = {
            "p": float(m["p"]), "r": float(m["r"]), "f1": float(m["f1"]),
            "tp": int(m["tp"]), "tn": int(m["tn"]), "fp": int(m["fp"]),
            "fn": int(m["fn"]), "flips": int(ds.flips(oks, seq_s)),
            "seg_rates": {k: round(float(v), 4) for k, v in seg_r.items()},
            "n_corr": int(corr),
            "trace": ds.serialize_trace(trace),
            "noise_std": round(float(noise_std), 3),
            "corrupt_kc": round(float(corrupt_kc), 4),
        }
    gt = {"n_pos": int(sum(seq_gt)), "n_total": len(seq_gt),
          "n_empty": int(sum(1 for mm in seq_m if mm.max() == 0))}
    return res, gt


def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return float(a.mean()), sd


def bootstrap_ci(vals, n=N_BOOT, alpha=0.05):
    """种子层面百分位 bootstrap 95% CI（均值）。"""
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty(n)
    for i in range(n):
        means[i] = a[rng.integers(0, len(a), size=len(a))].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_paired(a_vals, b_vals, n=N_BOOT, alpha=0.05):
    """配对 bootstrap 95% CI：均值差 mean(A-B)。"""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    rng = np.random.default_rng(BOOT_SEED + 1)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(a), size=len(a))
        d[i] = (a[idx] - b[idx]).mean()
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def signflip_p(diffs):
    """配对符号翻转精确置换检验（双侧）。H0：均值差=0；2^n 全枚举。"""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = float(d.mean())
    if n > 16:                                   # 太大则蒙特卡洛
        rng = np.random.default_rng(BOOT_SEED + 2)
        cnt = 0
        total = 20000
        signs = rng.choice([-1.0, 1.0], size=(total, n))
        means = (signs * d).mean(axis=1)
        cnt = int(np.sum(np.abs(means) >= abs(obs) - 1e-12))
        return cnt / total, total
    cnt = 0
    total = 1 << n
    for mask in range(total):
        s = np.array([d[i] if (mask >> i) & 1 else -d[i] for i in range(n)])
        if abs(float(s.mean())) >= abs(obs) - 1e-12:
            cnt += 1
    return cnt / total, total


def thr_scan(frames, masks, seg_len, task_hue, corrupt, thr_list, keys):
    """参考种子（seed 7, jitter 0）扫阈值 → PR 操作点（与 docs/219 单次数字对齐）。"""
    points = []
    for thr in thr_list:
        res, gt = run_seed(frames, masks, seg_len, task_hue, thr, corrupt,
                           REF_SEED, keys, jitter=0.0)
        pts = {k: {"p": res[k]["p"], "r": res[k]["r"], "f1": res[k]["f1"],
                   "flips": res[k]["flips"]} for k in keys}
        points.append({"thr": thr, "variants": pts})
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--davis", default=None)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--thr", default="auto", help="确认阈值基线；auto=按目标确认余量规则")
    ap.add_argument("--corrupt", default="gain", choices=["gain", "noise"])
    ap.add_argument("--seg", default="20,20,15,25")
    ap.add_argument("--variants", default="A,B,C,C2,D,E,F")
    ap.add_argument("--thr-scan", default=None, help="逗号分隔阈值列表（默认不扫）")
    ap.add_argument("--scan-variants", default="A,D")
    ap.add_argument("--jitter", type=float, default=JITTER,
                    help="扰动强度按种子采样幅度（0=纯噪声抽样方差）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="",
                    help="结果/checkpoint 文件名后缀（如 --tag pure → a2_<video>_pure.json），"
                         "用于同一视频多配置并存归档")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.davis:
        ds.DAVIS = args.davis

    os.makedirs(args.out_dir, exist_ok=True)
    keys = [k.strip() for k in args.variants.split(",") if k.strip()]
    seg_len = dict(zip(["base", "dusk", "corrupt", "rest"],
                       [int(x) for x in args.seg.split(",")]))
    scan_thrs = ([float(x) for x in args.thr_scan.split(",")]
                 if args.thr_scan else None)
    scan_keys = [k.strip() for k in args.scan_variants.split(",") if k.strip()]

    t0 = time.time()
    frames, masks = ds.load_video(args.video)
    obs = []
    for f, m in zip(frames, masks):
        h = ds.target_obs(f, m, 0.0)[1]
        if h is not None:
            obs.append(h)
    if not obs:
        print("R_VIDEO=" + args.video)
        print("R_ERROR=NO_COLORED_TARGET")
        return 2
    task_hue = float(np.median(obs))
    fracs = [ds.target_obs(f, m, task_hue)[0] for f, m in zip(frames, masks)
             if ds.target_obs(f, m, task_hue)[1] is not None]
    median_frac = float(np.median(fracs))
    if args.thr == "auto":
        thr = float(min(0.60, max(0.30, round(median_frac - 0.30, 1))))
    else:
        thr = float(args.thr)

    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    cfg = {"seg": seg_len, "thr": thr, "corrupt": args.corrupt,
           "task_hue": task_hue, "jitter": args.jitter,
           "noise_std_base": 10.0, "corrupt_kc_base": 0.05,
           "variants": keys, "n_seeds": args.n_seeds}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_%s_%s.json" % (args.video, ck_tag))
    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_seed", {})

    per_seed = dict(done)
    for seed in seeds:
        if str(seed) in per_seed:
            continue
        res, gt = run_seed(frames, masks, seg_len, task_hue, thr, args.corrupt,
                           seed, keys, args.jitter)
        per_seed[str(seed)] = {"seed": seed, "gt": gt, "variants": res}
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"video": args.video, "config": cfg,
                       "per_seed": per_seed}, f, ensure_ascii=False, indent=1)
        print("PROGRESS", flush=True)

    gt0 = per_seed[str(seeds[0])]["gt"]
    variants_out = {}
    for key in keys:
        f1s = [per_seed[str(s)]["variants"][key]["f1"] for s in seeds]
        ps = [per_seed[str(s)]["variants"][key]["p"] for s in seeds]
        rs = [per_seed[str(s)]["variants"][key]["r"] for s in seeds]
        fl = [per_seed[str(s)]["variants"][key]["flips"] for s in seeds]
        f1m, f1sd = mean_sd(f1s)
        f1ci = bootstrap_ci(f1s)
        flci = bootstrap_ci(fl)
        variants_out[key] = {
            "per_seed": [{**per_seed[str(s)]["variants"][key],
                          "seed": s} for s in seeds],
            "mean_sd": {"f1": [f1m, f1sd], "p": list(mean_sd(ps)),
                        "r": list(mean_sd(rs)), "flips": list(mean_sd(fl))},
            "bootstrap_ci95": {"f1": list(f1ci), "flips": list(flci)},
        }

    # flips 重采样检验（A 纪念 vs D 模板，配对）
    flips_test = None
    if "A" in keys and "D" in keys:
        fa = [per_seed[str(s)]["variants"]["A"]["flips"] for s in seeds]
        fd = [per_seed[str(s)]["variants"]["D"]["flips"] for s in seeds]
        diffs = [a - b for a, b in zip(fa, fd)]
        pval, n_perm = signflip_p(diffs)
        flips_test = {
            "a_vs_d": {
                "mean_diff": float(np.mean(diffs)),
                "ci95_diff": list(bootstrap_ci_paired(fa, fd)),
                "p_signflip": float(pval),
                "n_permutations": int(n_perm),
                "method": "paired sign-flip exact permutation, two-sided"},
            "a": {"mean": float(np.mean(fa)), "ci95": list(bootstrap_ci(fa))},
            "d": {"mean": float(np.mean(fd)), "ci95": list(bootstrap_ci(fd))},
        }

    scan_out = None
    if scan_thrs:
        scan_out = {"seed": REF_SEED, "jitter": 0.0,
                    "points": thr_scan(frames, masks, seg_len, task_hue,
                                       args.corrupt, scan_thrs, scan_keys)}

    out = {
        "artifact": "davis_a2_stats",
        "doc_ref": "docs/228",
        "video": args.video,
        "config": {"seg": seg_len, "thr": thr, "corrupt": args.corrupt,
                   "task_hue": round(task_hue, 3),
                   "median_frac": round(median_frac, 3),
                   "n_seeds": args.n_seeds, "first_seed": args.first_seed,
                   "seeds": seeds, "jitter": args.jitter,
                   "noise_std_base": 10.0, "corrupt_kc_base": 0.05,
                   "tag": args.tag, "ck_tag": ck_tag},
        "gt": gt0,
        "variants": variants_out,
        "thr_scan": scan_out,
        "flips_test": flips_test,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir,
                            "a2_%s%s.json" % (args.video,
                                              "_" + args.tag if args.tag else ""))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行恰一个数字（顺序见模块 docstring）----
    a = variants_out["A"]
    a_f1m, a_f1sd = a["mean_sd"]["f1"]
    a_pm, a_psd = a["mean_sd"]["p"]
    a_rm, a_rsd = a["mean_sd"]["r"]
    a_fm, a_fsd = a["mean_sd"]["flips"]
    print("R_VIDEO=" + args.video)
    print("R_SEEDS=%d" % len(seeds))
    print("R_THR=%.3f" % thr)
    print("R_TASKHUE=%.3f" % task_hue)
    print("R_GT_NPOS=%d" % gt0["n_pos"])
    print("R_GT_NTOTAL=%d" % gt0["n_total"])
    print("R_GT_NEMPTY=%d" % gt0["n_empty"])
    print("R_FA_MEAN=%.4f" % a_f1m)
    print("R_FA_SD=%.4f" % a_f1sd)
    print("R_PA_MEAN=%.4f" % a_pm)
    print("R_PA_SD=%.4f" % a_psd)
    print("R_RA_MEAN=%.4f" % a_rm)
    print("R_RA_SD=%.4f" % a_rsd)
    print("R_FLIPA_MEAN=%.4f" % a_fm)
    print("R_FLIPA_SD=%.4f" % a_fsd)
    print("R_F1A_CI_LO=%.4f" % a["bootstrap_ci95"]["f1"][0])
    print("R_F1A_CI_HI=%.4f" % a["bootstrap_ci95"]["f1"][1])
    print("R_FLIPA_CI_LO=%.4f" % a["bootstrap_ci95"]["flips"][0])
    print("R_FLIPA_CI_HI=%.4f" % a["bootstrap_ci95"]["flips"][1])
    if flips_test:
        print("R_FLIPD_MEAN=%.4f" % flips_test["d"]["mean"])
        print("R_FLIPD_SD=%.4f" % mean_sd(
            [per_seed[str(s)]["variants"]["D"]["flips"] for s in seeds])[1])
        print("R_FLIPDIFF_MEAN=%.4f" % flips_test["a_vs_d"]["mean_diff"])
        print("R_FLIPDIFF_CI_LO=%.4f" % flips_test["a_vs_d"]["ci95_diff"][0])
        print("R_FLIPDIFF_CI_HI=%.4f" % flips_test["a_vs_d"]["ci95_diff"][1])
        print("R_FLIPP_PVAL=%.4f" % flips_test["a_vs_d"]["p_signflip"])
    if scan_out:
        for pt in scan_out["points"]:
            v = pt["variants"]
            pa = v.get("A", {}).get("p", float("nan"))
            ra = v.get("A", {}).get("r", float("nan"))
            f1a = v.get("A", {}).get("f1", float("nan"))
            pd_ = v.get("D", {}).get("p", float("nan"))
            f1d = v.get("D", {}).get("f1", float("nan"))
            print("R_SCAN_THR=%.3f" % pt["thr"])
            print("R_SCAN_P_P=%.4f" % pa)
            print("R_SCAN_P_R=%.4f" % ra)
            print("R_SCAN_P_F1=%.4f" % f1a)
            print("R_SCAN_D_P=%.4f" % pd_)
            print("R_SCAN_D_F1=%.4f" % f1d)
    print("R_ELAPSED=%.2f" % (time.time() - t0))


if __name__ == "__main__":
    sys.exit(main())
