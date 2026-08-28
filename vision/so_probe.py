"""vision/so_probe.py — 二阶残差探针（docs/236 §二 H-SO 裁决，交付 docs/237）。

复用 compose_test 的 C0-C3 环境梯度与 CompLoop 回路（import，不修改任何既有脚本），
在每一次"条目 E 在窗口 w 匹配"事件上计算二阶信号度量（预注册，docs/237 §一）：

  SO_conf(E,w) : 回路"自预测的匹配质量"（E 在 w+1 继续匹配的预测概率），只从条目
                 账本历史（回路自己的匹配事件序列，严格早于 w 的部分 + 当前匹配事件
                 本身）计算（docs/237 §1.2' 细化版）：
                   d      = E 在 w 的连续匹配深度（含 w；触发条件）
                   cont_all = 账本全局史（u<w，所有条目）的续匹配率；n_all = 样本数
                   cont_d   = 其中"自身运行深度 >= d"的匹配的续匹配率；n_d = 样本数
                             （n_d<3 -> 回退 cont_all；n_all<3 -> 0.5）
                   SO_conf = clip((n_all*cont_all + n_d*cont_d + 3*0.5)
                                  /(n_all+n_d+3), 0, 1)
                 不使用：w 的残差/能量/c2 槽位数值、生成器真值、w+1 及以后任何信息。
  SO_out(E,w+1) : 实际匹配结果 = E 在 w+1 是否再次匹配（1=命中，0=误配）。
  SO_err        : |SO_conf - SO_out|（均在 [0,1]，天然归一化）。
  SO_info       : SO_conf 对 SO_out 的预测力：point-biserial r（主，null=0）+
                  AUC（次，null=0.5）；跨 10 种子 mean±SD + bootstrap 95% CI。

假对照（预注册）：同一观察池用固定 0.5 / 均匀随机 置信度替代 SO_conf 重算度量；
判定要求 SO_info 显著优于假对照（种子配对差值 CI 排除 0）。

判定（预注册冻结，docs/237 §1.5）：SECOND_ORDER / NO_SECOND_ORDER / BOUNDARY。
趋势（§1.6）：ENHANCING / FLAT / DECREASING（r_C3 - r_C0 配对差 CI）。

统计外壳（docs/228/232 模式）：10 种子（seed 0-9）、jitter 0.25、checkpoint
（ckpt_so_<hash>.json，--resume）、JSON 归档（so_<tag>.json）、mean±SD + bootstrap
95% CI（2000 次，种子 20260828）。
安全纪律（docs/228/234 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_* 摘要块；
禁止把日志/JSON 原文送进上下文。

用法：
  python vision/so_probe.py --levels 20,21,22,23 --n-seeds 10 --tag main
  python vision/so_probe.py --levels 20,21,22,23 --n-seeds 1  --tag timing
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import mean_sd, bootstrap_ci, JITTER
from compose_test import CompLoop, make_scene, LEVELS, ENERGY_BINS, UPPER_BINS

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828
CONSIST_FLOOR = 3     # 账本史样本数下限（不足 -> 回退先验/无条件率）
PRIOR_W = 3.0         # 朝 0.5 的收缩先验权重

# 回路参数：与 compose_test.main() 的 loop_cfg 逐字一致（结果可比）
LOOP_CFG = {"alpha_fast": 0.5, "alpha_slow": 0.03, "thresh": 0.15,
            "deadband": 0.015, "k_theta": 6.0, "k_db": 1.5,
            "thresh_max": 0.6, "db_max": 0.15, "k_consist": 3,
            "hits_min": 3, "persist_win": 5, "k_split": 5, "delta_rel": 0.30,
            "energy_bins": list(ENERGY_BINS), "bbox_bins": list(UPPER_BINS)}


def window_aligned(match_trace):
    """match_trace 是 (win, key) 事件序列（部分窗口无匹配事件）；重建按窗口对齐的
    匹配表 {win: key}（仅匹配事件；无事件窗口不在表中）。key 转 tuple。"""
    m = {}
    for w, key in match_trace:
        if key is not None:
            m[w] = tuple(key)
    return m


def run_depth_at(matched, E, w):
    """E 在窗口 w 的连续匹配深度（含 w；从 w 往回数连续同 key 匹配数）。"""
    d = 0
    ww = w
    while ww in matched and matched[ww] == E:
        d += 1
        ww -= 1
    return d


def compute_observations(matched, entry_log, n_windows):
    """返回观察列表（预注册观察池，docs/237 §1.2）：
      [dict(w, key, arity, depth, n_all, n_d, conf, out)] for 每个 (E, w) 匹配事件。
    排除（预注册）：① 末窗口（无 w+1）；② E 在 w 退休（提升）的窗口（结构性替换，
    非匹配质量结果）。

    SO_conf（§1.2' 细化版，冻结于最终运行前）：用回路账本的全局匹配史（所有条目，
    严格早于 w）估计深度条件续匹配率：
      cont_all = P(任意条目在 u 匹配后在 u+1 续匹配)，n_all = 样本数
      cont_d   = 其中"自身运行深度 >= d"的匹配的续匹配率，n_d = 样本数
      SO_conf = clip((n_all*cont_all + n_d*cont_d + PRIOR_W*0.5)/(n_all+n_d+PRIOR_W))
    """
    retired_at = {}
    for e in entry_log:
        k = tuple(e["key"])
        if e.get("retired"):
            retired_at[k] = e.get("retired_at")
    wins = sorted(matched)
    obs = []
    for w in wins:
        if w >= n_windows - 1:
            continue
        E = matched[w]
        if retired_at.get(E) == w:
            continue
        d = run_depth_at(matched, E, w)
        g_all = [u for u in wins if u < w]
        n_all = len(g_all)
        cont_all = (sum(1 for u in g_all
                        if (u + 1) in matched and matched[u + 1] == matched[u])
                    / max(1, n_all))
        g_d = [u for u in g_all if run_depth_at(matched, matched[u], u) >= d]
        n_d = len(g_d)
        cont_d = (sum(1 for u in g_d
                      if (u + 1) in matched and matched[u + 1] == matched[u])
                  / max(1, n_d))
        cont_all = cont_all if n_all >= CONSIST_FLOOR else 0.5
        cont_d = cont_d if n_d >= CONSIST_FLOOR else cont_all
        conf = (n_all * cont_all + n_d * cont_d + PRIOR_W * 0.5) / (n_all + n_d + PRIOR_W)
        conf = float(np.clip(conf, 0.0, 1.0))
        out = 1.0 if (w + 1) in matched and matched[w + 1] == E else 0.0
        obs.append(dict(w=w, key=",".join(str(k) for k in E), arity=len(E),
                        depth=d, n_all=n_all, n_d=n_d, conf=round(conf, 6),
                        out=out))
    return obs


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rank_auc(conf, out):
    """Mann-Whitney 秩统计 AUC（ties=0.5）；无两类样本 -> 0.5（退化，预注册）。"""
    conf = np.asarray(conf, dtype=float)
    out = np.asarray(out, dtype=float)
    pos = conf[out == 1]
    neg = conf[out == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def seed_metrics(obs, seed, lvcode):
    """每 (级,种子) 的 SO 度量 + 假对照（同一观察池）。"""
    confs = np.array([o["conf"] for o in obs], dtype=float)
    outs = np.array([o["out"] for o in obs], dtype=float)
    n = len(obs)
    n_pos = int(outs.sum())
    n_neg = n - n_pos
    rng = np.random.default_rng(seed * 104729 + lvcode * 7919 + 999331)
    fake_fixed = np.full(n, 0.5)
    fake_rand = rng.uniform(0.0, 1.0, size=n)
    out = {
        "n_obs": n, "n_pos": n_pos, "n_neg": n_neg,
        "r": pearson_r(confs, outs),
        "auc": rank_auc(confs, outs),
        "err": float(np.mean(np.abs(confs - outs))) if n else 0.0,
        "conf_pos_mean": float(np.mean(confs[outs == 1])) if n_pos else 0.0,
        "conf_neg_mean": float(np.mean(confs[outs == 0])) if n_neg else 0.0,
        "r_fake_fixed": pearson_r(fake_fixed, outs),
        "r_fake_rand": pearson_r(fake_rand, outs),
        "auc_fake_rand": rank_auc(fake_rand, outs),
        "err_fake_fixed": float(np.mean(np.abs(fake_fixed - outs))) if n else 0.0,
        "err_fake_rand": float(np.mean(np.abs(fake_rand - outs))) if n else 0.0,
    }
    return out


def run_so(lvcode, seed, n_frames, width, height, fps, window, jitter):
    """跑 (级,种子) 一次完整运行：CompLoop 原样复用 + SO 度量。"""
    frames, labels = make_scene(lvcode, seed, n_frames=n_frames, width=width,
                                height=height, fps=fps, jitter=jitter)
    loop = CompLoop(window=window, **LOOP_CFG)
    for g in frames:
        loop.step(g)
    n_windows = max(1, n_frames // window)
    base = loop.finalize(n_windows, labels)
    matched = window_aligned(loop.match_trace)
    obs = compute_observations(matched, base["entry_log"], n_windows)
    out = dict(base)
    out.update({
        "seed": seed, "level": lvcode, "frames": n_frames,
        "so": seed_metrics(obs, seed, lvcode),
        "observations": obs,
        "match_trace_win": sorted(matched.keys()),
    })
    return out


def level_aggregate(rs, seeds):
    def col(k):
        return [r["so"][k] for r in rs]

    agg = {}
    for k in ("n_obs", "n_pos", "n_neg", "r", "auc", "err",
              "conf_pos_mean", "conf_neg_mean",
              "r_fake_fixed", "r_fake_rand", "auc_fake_rand",
              "err_fake_fixed", "err_fake_rand"):
        agg[k + "_mean"], agg[k + "_sd"] = mean_sd(col(k))
    # 结构上下文（top-level，来自 CompLoop.finalize）
    for k in ("mae_mean", "sc2", "compound_frac", "churn_frac"):
        agg[k + "_mean"], agg[k + "_sd"] = mean_sd([r[k] for r in rs])
    agg["r_ci95"] = list(bootstrap_ci(col("r")))
    agg["auc_ci95"] = list(bootstrap_ci(col("auc")))
    diff_rand = [r["so"]["r"] - r["so"]["r_fake_rand"] for r in rs]
    diff_fix = [r["so"]["r"] - r["so"]["r_fake_fixed"] for r in rs]
    agg["diff_rand_mean"] = float(np.mean(diff_rand))
    agg["diff_rand_ci95"] = list(bootstrap_ci(diff_rand))
    agg["diff_fix_ci95"] = list(bootstrap_ci(diff_fix))
    agg["chance_ok"] = int(agg["r_ci95"][0] > 0.0)
    agg["fakeok_rand"] = int(agg["diff_rand_ci95"][0] > 0.0)
    agg["per_seed_r"] = col("r")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="20,21,22,23")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="so")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "fps": args.fps, "size": [args.width, args.height],
           "window": args.window, "jitter": args.jitter, "tag": args.tag,
           "scene": {str(k): v for k, v in LEVELS.items()},
           "loop": LOOP_CFG,
           "so": {"formula": "v2_global_hazard", "consist_floor": CONSIST_FLOOR,
                  "prior_w": PRIOR_W,
                  "fake_rand_rng": "seed*104729 + lvcode*7919 + 999331",
                  "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED}}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_so_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    for lv in levels:
        for seed in seeds:
            key = "%d_%d" % (lv, seed)
            if key in per_unit:
                continue
            r = run_so(lv, seed, args.frames, args.width, args.height,
                       args.fps, args.window, args.jitter)
            per_unit[key] = r
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)

    # ---- 跨种子聚合 ----
    levels_out = {}
    for lv in levels:
        rs = [per_unit["%d_%d" % (lv, s)] for s in seeds]
        agg = level_aggregate(rs, seeds)
        levels_out[lv] = {"name": LEVELS[lv]["name"], "per_seed": rs,
                          "mean_sd": agg}
        for k in ("r_ci95", "auc_ci95", "diff_rand_ci95", "diff_fix_ci95"):
            levels_out[lv]["mean_sd"][k] = agg[k]
        levels_out[lv]["chance_ok"] = agg["chance_ok"]
        levels_out[lv]["fakeok_rand"] = agg["fakeok_rand"]

    # ---- 判定（预注册，docs/237 §1.5） ----
    have_all = all(k in levels_out for k in (20, 21, 22, 23))
    if not have_all:
        verdict = "INCOMPLETE"
        vnote = "need levels 20,21,22,23 for verdict"
        r0 = r1 = r2 = r3 = 0.0
        baseline = False
        chance_c2 = chance_c3 = fake_c2 = fake_c3 = False
        d_c3c0 = [0.0]
    else:
        lo = {lv: levels_out[lv]["mean_sd"] for lv in levels}
        chance_c2 = levels_out[22]["chance_ok"] == 1
        chance_c3 = levels_out[23]["chance_ok"] == 1
        fake_c2 = levels_out[22]["fakeok_rand"] == 1
        fake_c3 = levels_out[23]["fakeok_rand"] == 1
        r0, r1, r2, r3 = (lo[20]["r_mean"], lo[21]["r_mean"],
                          lo[22]["r_mean"], lo[23]["r_mean"])
        baseline = bool(r2 >= r0 and r2 >= r1 and r3 >= r0 and r3 >= r1)

        if chance_c2 and chance_c3 and fake_c2 and fake_c3 and baseline:
            verdict = "SECOND_ORDER"
            vnote = ("C2/C3 r CI excludes 0 and beats fake_rand and r>=C0/C1 baselines")
        elif (not chance_c2 or not fake_c2) and (not chance_c3 or not fake_c3):
            verdict = "NO_SECOND_ORDER"
            vnote = "C2/C3 r not above chance or not better than fake control"
        else:
            verdict = "BOUNDARY"
            vnote = "mixed evidence across C2/C3; see numbers"

        # ---- 趋势（预注册，docs/237 §1.6；r_C3 - r_C0 按种子配对） ----
        r3s = levels_out[23]["mean_sd"]["per_seed_r"]
        r0s = levels_out[20]["mean_sd"]["per_seed_r"]
        d_c3c0 = [a - b for (a, b) in zip(r3s, r0s)]

    d_mean = float(np.mean(d_c3c0))
    d_lo, d_hi = bootstrap_ci(d_c3c0)
    if d_lo > 0.0:
        trend = "ENHANCING"
    elif d_hi < 0.0:
        trend = "DECREASING"
    else:
        trend = "FLAT"

    vdict = {"verdict": verdict, "note": vnote, "baseline_ok": int(baseline),
             "chance_c2": int(chance_c2), "chance_c3": int(chance_c3),
             "fake_c2": int(fake_c2), "fake_c3": int(fake_c3),
             "r_c0": r0, "r_c1": r1, "r_c2": r2, "r_c3": r3,
             "trend": trend, "trend_c3c0_mean": d_mean,
             "trend_c3c0_ci95": [d_lo, d_hi]}

    out = {
        "artifact": "so_probe",
        "doc_ref": "docs/236 §二, docs/237",
        "config": cfg,
        "levels": {str(k): v for k, v in levels_out.items()},
        "verdict": vdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "so_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_SO_LEVELS=%d" % len(levels))
    print("R_SO_SEEDS=%d" % len(seeds))
    for i, lv in enumerate(levels):
        p = "C%d" % i
        o = levels_out[lv]
        ms = o["mean_sd"]
        print("R_SO_NAME_%s=%s" % (p, o["name"]))
        print("R_SO_NOBS_%s=%.1f" % (p, ms["n_obs_mean"]))
        print("R_SO_R_%s=%.6f" % (p, ms["r_mean"]))
        print("R_SO_R_SD_%s=%.6f" % (p, ms["r_sd"]))
        print("R_SO_R_LO_%s=%.6f" % (p, ms["r_ci95"][0]))
        print("R_SO_R_HI_%s=%.6f" % (p, ms["r_ci95"][1]))
        print("R_SO_AUC_%s=%.6f" % (p, ms["auc_mean"]))
        print("R_SO_AUC_LO_%s=%.6f" % (p, ms["auc_ci95"][0]))
        print("R_SO_AUC_HI_%s=%.6f" % (p, ms["auc_ci95"][1]))
        print("R_SO_FAKEFIX_%s=%.6f" % (p, ms["r_fake_fixed_mean"]))
        print("R_SO_FAKERAND_%s=%.6f" % (p, ms["r_fake_rand_mean"]))
        print("R_SO_DIFF_%s=%.6f" % (p, ms["diff_rand_mean"]))
        print("R_SO_DIFF_LO_%s=%.6f" % (p, ms["diff_rand_ci95"][0]))
        print("R_SO_DIFF_HI_%s=%.6f" % (p, ms["diff_rand_ci95"][1]))
        print("R_SO_ERR_%s=%.6f" % (p, ms["err_mean"]))
        print("R_SO_ERR_FAKE_%s=%.6f" % (p, ms["err_fake_rand_mean"]))
        print("R_SO_CONFPOS_%s=%.6f" % (p, ms["conf_pos_mean_mean"]))
        print("R_SO_CONFNEG_%s=%.6f" % (p, ms["conf_neg_mean_mean"]))
        print("R_SO_MAE_%s=%.6f" % (p, ms["mae_mean_mean"]))
        print("R_SO_SC2_%s=%.4f" % (p, ms["sc2_mean"]))
        print("R_SO_COMP_%s=%.4f" % (p, ms["compound_frac_mean"]))
        print("R_SO_CHURN_%s=%.4f" % (p, ms["churn_frac_mean"]))
        print("R_SO_CHANCE_%s=%d" % (p, o["chance_ok"]))
        print("R_SO_FAKEOK_%s=%d" % (p, o["fakeok_rand"]))
    v = out["verdict"]
    print("R_SO_BASELINE=%d" % v["baseline_ok"])
    print("R_SO_TREND_DIFF=%.6f" % v["trend_c3c0_mean"])
    print("R_SO_TREND_LO=%.6f" % v["trend_c3c0_ci95"][0])
    print("R_SO_TREND_HI=%.6f" % v["trend_c3c0_ci95"][1])
    print("R_SO_TREND=%s" % trend)
    print("R_SO_VERDICT=%s" % verdict)
    print("R_SO_VERDICT_NOTE=%s" % vnote)
    print("R_SO_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
