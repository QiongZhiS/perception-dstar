"""vision/l5_skill_acq.py — docs/258 L5 桥：技能习得效率（docs/255 §三1 首选机制泛化形态，
真实拼接流按段序数测量习得成本随阅历的变化）。

把 docs/255 §三1 的"技能习得效率"可测形态（新场景族的习得速度与所需干预随阅历下降）
落到单条真实拼接流上：基座 = 现行主机制 DeferredLoop（docs/251 快慢双原型 + 延迟定级，
import 复用**逐字**，零重调、零机制改动——本实验只有**最小纯测量加法**：在
run_deferred_stream 返回的 out/loop 之上 post-hoc 计算段级习得成本度量）。

预注册（docs/258 §一，冻结；docs/63+247 纪律；机制与测量/判据/判定映射/守卫/旋钮/流
先于实现写入 docs/258，运行后不改）：
  机制：quota_retire.DeferredLoop + quota_retire.run_deferred_stream 逐字复用（同代码
    路径 -> 与 docs/251 数字逐位一致；预测路径零改动 -> MAE/ratio 与 docs/243-251 一致）。
  段边界（只用于评估/诊断统计，不进任何机制决策，docs/250/251 纪律）：
    R1（主测量流）：GT 切换窗 [8,13,21,30,36,41,49,54] -> 段 k 覆盖窗 [b_k, b_{k+1})，
      b = [0,8,13,21,30,36,41,49,54,59]（9 段）。
    S4（第二测量流）：V1+V2+V3 拼接，构造性边界 = 累计采样帧数 // WINDOW（3 段）。
  段级习得成本度量（docs/258 §1.2 冻结，主度量 + 支持度量）：
    cost_density_k = (created_k + recycled_k + promoted_k) / len_k（每窗干预量，主度量）；
    gist_creation_latency_k = min(created in seg k) - w_lo（NA 若无创建）；
    confirm_latency_k = min(promoted_at - created over prototypes created in seg k that
      eventually promote)（NA 若无升级）；
    promo_hit_rate_k = mean final hits of slow prototypes promoted in seg k（NA 若无）。
  判据（docs/258 §1.3 冻结，每判据带 docs/247 层级标签；L5 不作判据本体，标签 [L5桥]）：
    1. [L5桥][机制][行为证据] SKILL_ACQ_DECLINE : R1（9 段）Spearman rho(cost, k) <= 0
       且 mean(cost, k=5..8) <= mean(cost, k=0..3)；S4（3 段）rho <= 0 且 cost(k=2)
       <= cost(k=0)；两流都过 = 判据 1 过。
    2. [L5桥][机制] TRANSFER_KEEP : 全流（S1-S4+R1）ratio <= 1.5 且 SC2_slow > 0；
       R1 9 段 + S4 3 段每段有 >= 1 个"创建或匹配"事件。
    3. [L5桥][机制][行为证据] PROMOTION_BEHAVIOR : n_promo>0、n_recycle>0、存在流
       升级命中率均值 > 未升级均值（docs/250/251 行为证据保持）。
  判定（docs/258 §1.4 冻结）：判据 1-3 全过 + 守卫全过 = ACQUISITION_GAIN；判据 1 的
    R1 主测量流不过 = ACQUISITION_FLAT；R1 过但 S4 不过、或判据 2/3 不过 = PARTIAL；
    守卫不过 = GUARD_FAIL；数据不可用 = L5B_BLOCKED。
  守卫（docs/258 §1.5 冻结，不进判据）：
    R_L5B_GUARD_D251 : 32 项逐位复现 docs/251 §3.3/§3.4（l4_compose_test.guard_d251_items
      复用，容差 1e-4）——Mode OFF = 本实验机制状态。
    R_L5B_GUARD_D246 : 12/12（fastcut_fix.run_guard_quota + cross_domain_test.guard_vs_d246
      复用）。
    R_L5B_REPRO_RATIO : 6 项（S1-S4+R0+R1 ratio 与 docs/251 §3.3/docs/257 §3.2 逐位一致，
      容差 1e-4——预测路径零改动）。
  确定性复现：timing/main 两轮 R_L5B_* 逐位一致（仅 TAG/ELAPSED 不同）。

安全纪律（docs/258 §1.10 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L5B_* 摘要
块；运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/l5_skill_acq.py --smoke
  python vision/l5_skill_acq.py --tag timing
  python vision/l5_skill_acq.py --tag main
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import mean_sd, bootstrap_ci
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, WINDOW, RESIZE
from real_recalib import bridge_metrics
from soft_match_test import ALPHA, HITS_MIN
from cross_domain_test import (load_sampled_frames, WILD_VIDEOS, STREAMS,
                               RADIUS_L3, R_BASE_DAVIS, D246, DL_DIR,
                               guard_vs_d246, scene_switch_diag)
from fastcut_fix import run_guard_quota
from fastslow_test import (gist_metrics, build_entry_base, R_FAST, R_SLOW,
                           HITS_MIN_FAST, HITS_MIN_SLOW,
                           K_PROMOTE, K_DECAY, K_CONSIST_FAST)
from quota_retire import DeferredLoop, run_deferred_stream
from l4_compose_test import guard_d251_items, STREAM_ORDER

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# docs/251 §3.3 + docs/257 §3.2 冻结 ratio（R_L5B_REPRO_RATIO 期望；预测路径未动）
D251_RATIOS = {"S1": 1.155669, "S2": 1.371908, "S3": 0.732642,
               "S4": 0.370964, "R1": 0.951261, "R0": 0.907701}

# docs/258 §1.2 冻结：R1 GT 段边界（切换窗 [8,13,21,30,36,41,49,54] -> b = [0,...,59]）
# 运行时按 spans // WINDOW 计算（与 docs/257 同口径），此处仅为文档常量
R1_GT_SWITCH_WINDOWS = [8, 13, 21, 30, 36, 41, 49, 54]


def spearman_rho(xs, ys):
    """Spearman 秩相关（平均秩处理 ties + Pearson on ranks；纯 numpy，确定性）。"""
    def ranks(v):
        a = np.asarray(v, float)
        order = np.argsort(a, kind="stable")
        r = np.empty(len(a))
        r[order] = np.arange(1, len(a) + 1)
        s = np.argsort(a, kind="stable")
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[s[j + 1]] == a[s[i]]:
                j += 1
            if j > i:
                avg = (i + j + 2) / 2.0
                for t in range(i, j + 1):
                    r[s[t]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def segment_metrics(out, loop, boundaries):
    """段级习得成本度量（docs/258 §1.2 冻结定义）。boundaries = 段边界窗列表
    [b0, b1, ..., bn]（段 k 覆盖窗 [b_k, b_{k+1})）。返回每段 dict 列表。"""
    n_seg = len(boundaries) - 1
    created_by_seg = {k: [] for k in range(n_seg)}
    for cl in out["created_log"]:
        c = cl["created"]
        for k in range(n_seg):
            if boundaries[k] <= c < boundaries[k + 1]:
                created_by_seg[k].append(cl)
                break
    promos = out["promoted_log"]
    created_pid = {cl["pid"]: cl for cl in out["created_log"]}
    match_wins = {k: 0 for k in range(n_seg)}
    for w, pid in loop.match_trace:
        if pid is None:
            continue
        for k in range(n_seg):
            if boundaries[k] <= w < boundaries[k + 1]:
                match_wins[k] += 1
                break
    segs = []
    for k in range(n_seg):
        lo, hi = boundaries[k], boundaries[k + 1]
        length = hi - lo
        clist = created_by_seg[k]
        n_created = len(clist)
        n_recycled = sum(1 for cl in clist if cl["recycled"])
        n_promoted = sum(1 for pl in promos if lo <= pl["promoted_at"] < hi)
        created_wins = [cl["created"] for cl in clist]
        gist_latency = (min(created_wins) - lo) if created_wins else None
        confs = []
        for pl in promos:
            cl = created_pid.get(pl["pid"])
            if cl is not None and lo <= cl["created"] < hi:
                confs.append(pl["promoted_at"] - cl["created"])
        confirm_latency = min(confs) if confs else None
        hit_vals = [pl["hits"] for pl in promos if lo <= pl["promoted_at"] < hi]
        promo_hit_rate = (float(np.mean(hit_vals)) if hit_vals else None)
        cost_density = (n_created + n_recycled + n_promoted) / max(1, length)
        segs.append({"k": k, "w_lo": lo, "w_hi": hi, "len": length,
                     "created": n_created, "recycled": n_recycled,
                     "promoted": n_promoted, "matched_windows": match_wins[k],
                     "gist_creation_latency": gist_latency,
                     "confirm_latency": confirm_latency,
                     "promo_hit_rate": promo_hit_rate,
                     "cost_density": round(cost_density, 4)})
    return segs


def decline_stats(segs):
    """段级成本密度 vs 段序数的阅历效应统计（docs/258 §1.2/§1.3 冻结口径）。
    返回 (rho, first_half_mean, last_half_mean, decline)。"""
    ks = [s["k"] for s in segs]
    dens = [s["cost_density"] for s in segs]
    rho = spearman_rho(ks, dens)
    n = len(segs)
    if n >= 2:
        half = n // 2
        first = float(np.mean(dens[:half]))
        last = float(np.mean(dens[n - half:]))
    else:
        first, last = float(dens[0]), float(dens[0])
    decline = int(rho <= 0.0 and last <= first)
    return rho, first, last, decline


# ---------------- 构造冒烟测试（合成帧，非数据） ----------------
def smoke_main():
    rng = np.random.default_rng(20260828)
    frames = []
    for t in range(30):
        g = np.full((120, 160), 80, np.uint8)
        x0 = 20 + t
        y0 = 40 + (t // 2) % 30
        g[y0:y0 + 12, x0:x0 + 12] = 200
        frames.append(np.ascontiguousarray(g, dtype=np.uint8))
    out, loop = run_deferred_stream(frames)
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0
    keys_ok = int(all(k in out for k in ("ratio", "sc1_fast", "sc2_fast", "sc1_slow",
                                         "sc2_slow", "churn_slow", "n_promo",
                                         "n_recycle", "entry_log", "created_log")))
    # 段级度量函数在合成流 + 伪造边界上可计算（无异常、字段齐全）
    nw = out["n_windows"]
    bounds = [0, nw // 3, 2 * nw // 3, nw]
    segm = segment_metrics(out, loop, bounds)
    segm_ok = int(len(segm) == 3 and all(
        s["k"] == k and s["cost_density"] >= 0.0 and
        set(("created", "recycled", "promoted", "matched_windows",
             "gist_creation_latency", "confirm_latency", "promo_hit_rate",
             "cost_density")) <= set(s)
        for k, s in enumerate(segm)))
    rho = spearman_rho([0, 1, 2], [s["cost_density"] for s in segm])
    print("R_L5B_SMOKE_CONSTRUCT=1")
    print("R_L5B_SMOKE_RATIO_FINITE=%d" % int(np.isfinite(ratio)))
    print("R_L5B_SMOKE_KEYS_OK=%d" % keys_ok)
    print("R_L5B_SMOKE_SEGMETRICS_OK=%d" % segm_ok)
    print("R_L5B_SMOKE_SPEARMAN_FINITE=%d" % int(np.isfinite(rho)))
    print("R_L5B_SMOKE_SYNTH_RATIO=%.6f" % out["ratio"])
    print("R_L5B_SMOKE_ELAPSED=%.2f" % 0.0)
    return 0


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return smoke_main()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 野域数据（docs/248 §1.2 同一批；间隔抽帧 -> 灰度 160x120 -> 流） ----
    loaded = {}
    for vid, name in WILD_VIDEOS:
        p = os.path.join(DL_DIR, name)
        frames, step, total = load_sampled_frames(p)
        loaded[vid] = frames

    # ---- 流（docs/258 §1.7 冻结；S1-S4） ----
    streams_out = {}
    stream_loops = {}      # loop 对象单独持有（含 match_trace），不入 JSON 工件
    for sid, sname, vidx in STREAMS:
        frames = []
        for vi in vidx:
            frames.extend(loaded[WILD_VIDEOS[vi][0]])
        out, loop = run_deferred_stream(frames)
        stream_loops[sid] = loop
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        diag = scene_switch_diag(frames, creations)
        out["stream_id"] = sid
        out["stream_name"] = sname
        out["videos"] = [WILD_VIDEOS[vi][0] for vi in vidx]
        out["switch_diag"] = diag
        streams_out[sid] = out

    # ---- DAVIS：R0（flamingo x5，诊断）、R1（9 拼接；GT 段边界，主测量流） ----
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    r0r, r0_loop = run_deferred_stream(r0_frames)
    r1r, r1_loop = run_deferred_stream(r1_frames)
    r1r["bridge"] = bridge_metrics(build_entry_base(r1r), spans)   # 诊断
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    r1r["gist"] = gist_metrics(r1r, switch_windows)

    # ---- 段级度量（docs/258 §1.2 冻结；GT/构造性边界只进评估统计） ----
    r1_bounds = [0] + switch_windows + [r1r["n_windows"]]
    r1_segs = segment_metrics(r1r, r1_loop, r1_bounds)
    s4r = streams_out["S4"]
    cum = [len(loaded[WILD_VIDEOS[0][0]]),
           len(loaded[WILD_VIDEOS[0][0]]) + len(loaded[WILD_VIDEOS[1][0]])]
    s4_bounds = [0] + [c // WINDOW for c in cum] + [s4r["n_windows"]]
    s4_bounds = sorted(set(s4_bounds))
    s4_segs = segment_metrics(s4r, stream_loops["S4"], s4_bounds)

    # ---- 守卫 1：R_L5B_GUARD_D251（Mode OFF≡docs/251，32 项逐位；l4_compose_test 复用） ----
    d251_items = guard_d251_items(streams_out, r1r)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- 守卫 2：R_L5B_GUARD_D246（共享基座，12/12） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)

    # ---- 守卫 3：R_L5B_REPRO_RATIO（6 项；预测路径零改动） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            abs(streams_out[sid]["ratio"] - D251_RATIOS[sid]) < 1e-4))
    repro_items.append(("ratio_R1", abs(r1r["ratio"] - D251_RATIOS["R1"]) < 1e-4))
    repro_items.append(("ratio_R0", abs(r0r["ratio"] - D251_RATIOS["R0"]) < 1e-4))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, int(v)) for n, v in repro_items)

    # ---- 判据 1：SKILL_ACQ_DECLINE（R1 主测量 9 段 + S4 第二测量 3 段） ----
    r1_rho, r1_first, r1_last, r1_decline = decline_stats(r1_segs)
    s4_rho, s4_first, s4_last, s4_decline = decline_stats(s4_segs)
    acq_decline = int(r1_decline == 1 and s4_decline == 1)
    # pooled 支持统计：相对序数 k_rel = k/(n_seg-1) 归一（12 段）
    pooled_k, pooled_d = [], []
    for segs in (r1_segs, s4_segs):
        m = max(1, len(segs) - 1)
        for s in segs:
            pooled_k.append(s["k"] / m)
            pooled_d.append(s["cost_density"])
    pooled_rho = spearman_rho(pooled_k, pooled_d)

    # ---- 判据 2：TRANSFER_KEEP（全流 ratio/SC2 + 每段结构贡献） ----
    all_streams = [streams_out[s] for s, _, _ in STREAMS] + [r1r]
    stable_keep = int(all(r["ratio"] <= 1.5 for r in all_streams))
    struct_keep = int(all(r["sc2_slow"] > 0 for r in all_streams))
    seg_contrib = []
    for segs in (r1_segs, s4_segs):
        for s in segs:
            seg_contrib.append(int(s["created"] >= 1 or s["matched_windows"] >= 1))
    seg_contrib_ok = int(all(v == 1 for v in seg_contrib))
    transfer_keep = int(stable_keep == 1 and struct_keep == 1 and seg_contrib_ok == 1)
    transfer_detail = ",".join("seg%d:%d" % (i, v) for i, v in enumerate(seg_contrib))

    # ---- 判据 3：PROMOTION_BEHAVIOR（docs/250/251 行为证据保持） ----
    n_promo_total = sum(r["n_promo"] for r in all_streams)
    n_recycle_total = sum(r["n_recycle"] for r in all_streams)
    promo_means = [(r["promoted_mean_hits"], r["nonpromoted_mean_hits"])
                   for r in all_streams if r["sc1_slow"] > 0]
    promo_sep = int(any(mp > mn for mp, mn in promo_means)) if promo_means else 0
    promo_ok = int(n_promo_total > 0 and n_recycle_total > 0 and promo_sep)

    oks = {"acq_decline": acq_decline, "transfer_keep": transfer_keep,
           "promotion": promo_ok}

    # ---- 判定（docs/258 §1.4 冻结映射） ----
    guards_ok = (d251_ok == 1 and guard246_ok == 1 and repro_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d -> implementation drift; fix implementation, do "
                 "not judge mechanism (see R_L5B_GUARD_*)" % (
                     d251_ok, d251_passed, guard246_ok, repro_ok))
    elif acq_decline == 1 and transfer_keep == 1 and promo_ok == 1:
        verdict = "ACQUISITION_GAIN"
        vnote = ("SKILL_ACQ_DECLINE (R1 rho<=0 + last-half<=first-half; S4 rho<=0 + "
                 "last<=first) and TRANSFER_KEEP (ratio<=1.5, SC2_slow>0, every "
                 "segment has structural contribution) and PROMOTION_BEHAVIOR all "
                 "pass; guards D251=32/32 D246=12/12 REPRO=6/6 -> preliminary "
                 "evidence of skill acquisition efficiency (L5 bridge first cell)")
    elif r1_decline != 1:
        verdict = "ACQUISITION_FLAT"
        vnote = ("SKILL_ACQ_DECLINE failed on R1 (main measurement stream): rho=%.4f, "
                 "first-half=%.4f, last-half=%.4f -> experience did not lower "
                 "acquisition cost; negative result reported honestly (no "
                 "recalibration)" % (r1_rho, r1_first, r1_last))
    else:
        verdict = "PARTIAL"
        vnote = ("R1 decline holds but S4 decline or criteria 2/3 not all pass: "
                 "acq_decline=%d (r1=%d s4=%d), transfer_keep=%d, promotion=%d "
                 "(see numbers)" % (acq_decline, r1_decline, s4_decline,
                                    transfer_keep, promo_ok))

    # ---- 工件（自描述 JSON） ----
    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "working_point": {"r_slow": round(R_SLOW, 6), "r_fast": round(R_FAST, 6),
                             "hits_min_fast": HITS_MIN_FAST,
                             "hits_min_slow": HITS_MIN_SLOW,
                             "k_promote": K_PROMOTE, "k_decay": K_DECAY,
                             "k_consist_fast": K_CONSIST_FAST, "alpha": ALPHA},
           "mechanism": ("current main mechanism DeferredLoop (docs/251 fast-slow dual "
                         "prototypes + deferred finalization), import-reused verbatim; "
                         "zero mechanism change; minimal pure measurement addition "
                         "(segment-level acquisition-cost metrics, post-hoc); "
                         "prediction path zero-change; segment boundaries (R1 GT / S4 "
                         "constructional) used for evaluation statistics only, never "
                         "in mechanism decisions"),
           "criteria_def": {
               "SKILL_ACQ_DECLINE": "R1(9 segs): Spearman rho(cost_density,k)<=0 and "
                                    "mean(k=5..8)<=mean(k=0..3); S4(3 segs): rho<=0 and "
                                    "cost(k=2)<=cost(k=0); both streams required",
               "TRANSFER_KEEP": "all streams (S1-S4+R1) ratio<=1.5 and SC2_slow>0; "
                                "every R1/S4 segment has >=1 creation-or-match event",
               "PROMOTION_BEHAVIOR": "n_promo>0, n_recycle>0, some stream promoted "
                                     "mean hits > nonpromoted mean hits (docs/250/251 "
                                     "behavior evidence kept)"},
           "loop": LOOP_CFG,
           "r1_spans": [[a, b] for a, b in spans],
           "r1_switch_windows": switch_windows,
           "r1_segment_boundaries": r1_bounds,
           "s4_segment_boundaries": s4_bounds,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l5_skill_acq",
        "doc_ref": "docs/247, docs/250, docs/251, docs/252, docs/253, docs/254, "
                   "docs/255, docs/257, docs/258",
        "config": cfg,
        "streams": streams_out,
        "r0": r0r, "r1": r1r,
        "segments": {"R1": {"boundaries": r1_bounds, "segments": r1_segs},
                     "S4": {"boundaries": s4_bounds, "segments": s4_segs}},
        "decline": {"R1": {"rho": round(r1_rho, 4), "first_half": round(r1_first, 4),
                           "last_half": round(r1_last, 4), "decline": r1_decline},
                    "S4": {"rho": round(s4_rho, 4), "first_half": round(s4_first, 4),
                           "last_half": round(s4_last, 4), "decline": s4_decline},
                    "pooled": {"rho": round(pooled_rho, 4), "n": len(pooled_d)}},
        "criteria": {"acq_decline": acq_decline, "transfer_keep": transfer_keep,
                     "promotion": promo_ok,
                     "stable_keep": stable_keep, "struct_keep": struct_keep,
                     "seg_contrib_ok": seg_contrib_ok,
                     "seg_contrib_detail": transfer_detail,
                     "r1_gist_cov": r1r["gist"]["cov"]},
        "verdict": {"verdict": verdict, "note": vnote},
        "guards": {"d251": {"items": len(d251_items), "passed": d251_passed,
                            "ok": d251_ok, "detail": d251_detail},
                   "d246": {"ok": guard246_ok, "detail": guard246_detail},
                   "repro_ratio": {"ok": repro_ok, "detail": repro_detail}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l5b_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L5B_TAG=%s" % args.tag)
    print("R_L5B_R_SLOW=%.6f" % R_SLOW)
    print("R_L5B_R_FAST=%.6f" % R_FAST)
    print("R_L5B_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_L5B_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_L5B_K_PROMOTE=%d" % K_PROMOTE)
    print("R_L5B_K_DECAY=%d" % K_DECAY)
    print("R_L5B_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_L5B_ALPHA=%.4f" % ALPHA)
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        d = r["switch_diag"]
        print("R_L5B_%d_ID=%s" % (j, sid))
        print("R_L5B_%d_NAME=%s" % (j, sname))
        print("R_L5B_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_L5B_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_L5B_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_L5B_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_L5B_%d_MAE_SD=%.6f" % (j, r["mae_sd_win"]))
        print("R_L5B_%d_MAE_LO=%.6f" % (j, r["mae_ci95"][0]))
        print("R_L5B_%d_MAE_HI=%.6f" % (j, r["mae_ci95"][1]))
        print("R_L5B_%d_Q1=%.6f" % (j, r["mae_q1"]))
        print("R_L5B_%d_Q4=%.6f" % (j, r["mae_q4"]))
        print("R_L5B_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_L5B_%d_SC1_FAST=%d" % (j, r["sc1_fast"]))
        print("R_L5B_%d_SC2_FAST=%d" % (j, r["sc2_fast"]))
        print("R_L5B_%d_SC1_SLOW=%d" % (j, r["sc1_slow"]))
        print("R_L5B_%d_SC2_SLOW=%d" % (j, r["sc2_slow"]))
        print("R_L5B_%d_CHURN_SLOW=%.4f" % (j, r["churn_slow"]))
        print("R_L5B_%d_CHURN_LEGACY=%.4f" % (j, r["churn_legacy"]))
        print("R_L5B_%d_N_PROMO=%d" % (j, r["n_promo"]))
        print("R_L5B_%d_N_RECYCLE=%d" % (j, r["n_recycle"]))
        print("R_L5B_%d_PROMO_MEAN=%.4f" % (j, r["promoted_mean_hits"]))
        print("R_L5B_%d_NONPROMO_MEAN=%.4f" % (j, r["nonpromoted_mean_hits"]))
        print("R_L5B_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                            else "%.4f" % d["switch_corr"])))
    print("R_L5B_R0_FRAMES=%d" % r0r["frames"])
    print("R_L5B_R0_WINDOWS=%d" % r0r["n_windows"])
    print("R_L5B_R0_RATIO=%.6f" % r0r["ratio"])
    print("R_L5B_R0_SC1_FAST=%d" % r0r["sc1_fast"])
    print("R_L5B_R0_SC1_SLOW=%d" % r0r["sc1_slow"])
    print("R_L5B_R0_SC2_SLOW=%d" % r0r["sc2_slow"])
    print("R_L5B_R0_CHURN_SLOW=%.4f" % r0r["churn_slow"])
    print("R_L5B_R0_N_PROMO=%d" % r0r["n_promo"])
    print("R_L5B_R0_N_RECYCLE=%d" % r0r["n_recycle"])
    print("R_L5B_R1_FRAMES=%d" % r1r["frames"])
    print("R_L5B_R1_WINDOWS=%d" % r1r["n_windows"])
    print("R_L5B_R1_RATIO=%.6f" % r1r["ratio"])
    print("R_L5B_R1_SC1_FAST=%d" % r1r["sc1_fast"])
    print("R_L5B_R1_SC1_SLOW=%d" % r1r["sc1_slow"])
    print("R_L5B_R1_SC2_SLOW=%d" % r1r["sc2_slow"])
    print("R_L5B_R1_CHURN_SLOW=%.4f" % r1r["churn_slow"])
    print("R_L5B_R1_CHURN_LEGACY=%.4f" % r1r["churn_legacy"])
    print("R_L5B_R1_N_PROMO=%d" % r1r["n_promo"])
    print("R_L5B_R1_N_RECYCLE=%d" % r1r["n_recycle"])
    print("R_L5B_R1_PROMO_MEAN=%.4f" % r1r["promoted_mean_hits"])
    print("R_L5B_R1_NONPROMO_MEAN=%.4f" % r1r["nonpromoted_mean_hits"])
    print("R_L5B_R1_SWITCHES=%s" % ",".join(str(w) for w in switch_windows))
    print("R_L5B_R1_GIST_COV=%.4f" % r1r["gist"]["cov"])
    print("R_L5B_R1_GIST_PREC=%.4f" % r1r["gist"]["prec"])
    print("R_L5B_R1_GIST_COV_D2=%.4f" % r1r["gist"]["cov_d2"])
    print("R_L5B_R1_BRIDGE_SW=%.4f" % r1r["bridge"]["bridge_corr_switch"])
    print("R_L5B_R1_BRIDGE_VID=%.4f" % r1r["bridge"]["bridge_corr_video"])
    # R1 段级度量（9 段）
    for s in r1_segs:
        print("R_L5B_R1_SEG_%d_W_LO=%d" % (s["k"], s["w_lo"]))
        print("R_L5B_R1_SEG_%d_W_HI=%d" % (s["k"], s["w_hi"]))
        print("R_L5B_R1_SEG_%d_LEN=%d" % (s["k"], s["len"]))
        print("R_L5B_R1_SEG_%d_CREATED=%d" % (s["k"], s["created"]))
        print("R_L5B_R1_SEG_%d_RECYCLED=%d" % (s["k"], s["recycled"]))
        print("R_L5B_R1_SEG_%d_PROMOTED=%d" % (s["k"], s["promoted"]))
        print("R_L5B_R1_SEG_%d_MATCHED=%d" % (s["k"], s["matched_windows"]))
        print("R_L5B_R1_SEG_%d_GIST_LAT=%s" % (
            s["k"], ("NA" if s["gist_creation_latency"] is None
                     else "%d" % s["gist_creation_latency"])))
        print("R_L5B_R1_SEG_%d_CONFIRM_LAT=%s" % (
            s["k"], ("NA" if s["confirm_latency"] is None
                     else "%d" % s["confirm_latency"])))
        print("R_L5B_R1_SEG_%d_PROMO_HIT=%s" % (
            s["k"], ("NA" if s["promo_hit_rate"] is None
                     else "%.4f" % s["promo_hit_rate"])))
        print("R_L5B_R1_SEG_%d_COST_DENS=%.4f" % (s["k"], s["cost_density"]))
    # S4 段级度量（3 段）
    for s in s4_segs:
        print("R_L5B_S4_SEG_%d_W_LO=%d" % (s["k"], s["w_lo"]))
        print("R_L5B_S4_SEG_%d_W_HI=%d" % (s["k"], s["w_hi"]))
        print("R_L5B_S4_SEG_%d_LEN=%d" % (s["k"], s["len"]))
        print("R_L5B_S4_SEG_%d_CREATED=%d" % (s["k"], s["created"]))
        print("R_L5B_S4_SEG_%d_RECYCLED=%d" % (s["k"], s["recycled"]))
        print("R_L5B_S4_SEG_%d_PROMOTED=%d" % (s["k"], s["promoted"]))
        print("R_L5B_S4_SEG_%d_MATCHED=%d" % (s["k"], s["matched_windows"]))
        print("R_L5B_S4_SEG_%d_GIST_LAT=%s" % (
            s["k"], ("NA" if s["gist_creation_latency"] is None
                     else "%d" % s["gist_creation_latency"])))
        print("R_L5B_S4_SEG_%d_CONFIRM_LAT=%s" % (
            s["k"], ("NA" if s["confirm_latency"] is None
                     else "%d" % s["confirm_latency"])))
        print("R_L5B_S4_SEG_%d_PROMO_HIT=%s" % (
            s["k"], ("NA" if s["promo_hit_rate"] is None
                     else "%.4f" % s["promo_hit_rate"])))
        print("R_L5B_S4_SEG_%d_COST_DENS=%.4f" % (s["k"], s["cost_density"]))
    # 阅历效应统计（判据 1）
    print("R_L5B_R1_RHO=%.4f" % r1_rho)
    print("R_L5B_R1_FIRST_HALF=%.4f" % r1_first)
    print("R_L5B_R1_LAST_HALF=%.4f" % r1_last)
    print("R_L5B_R1_DECLINE=%d" % r1_decline)
    print("R_L5B_S4_RHO=%.4f" % s4_rho)
    print("R_L5B_S4_FIRST_HALF=%.4f" % s4_first)
    print("R_L5B_S4_LAST_HALF=%.4f" % s4_last)
    print("R_L5B_S4_DECLINE=%d" % s4_decline)
    print("R_L5B_POOLED_RHO=%.4f" % pooled_rho)
    print("R_L5B_POOLED_N=%d" % len(pooled_d))
    na_gist = sum(1 for s in r1_segs + s4_segs if s["gist_creation_latency"] is None)
    na_conf = sum(1 for s in r1_segs + s4_segs if s["confirm_latency"] is None)
    na_hit = sum(1 for s in r1_segs + s4_segs if s["promo_hit_rate"] is None)
    print("R_L5B_NA_GIST_LAT=%d" % na_gist)
    print("R_L5B_NA_CONFIRM_LAT=%d" % na_conf)
    print("R_L5B_NA_PROMO_HIT=%d" % na_hit)
    print("R_L5B_ACQ_DECLINE_OK=%d" % acq_decline)
    print("R_L5B_TRANSFER_KEEP_OK=%d" % transfer_keep)
    print("R_L5B_TRANSFER_DETAIL=%s" % transfer_detail)
    print("R_L5B_STABLE_OK=%d" % stable_keep)
    print("R_L5B_STRUCT_OK=%d" % struct_keep)
    print("R_L5B_SEG_CONTRIB_OK=%d" % seg_contrib_ok)
    print("R_L5B_PROMO_OK=%d" % promo_ok)
    print("R_L5B_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L5B_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L5B_PERSIST_OK=1")
    print("R_L5B_PERSIST_NOTE=%s" % "cited docs/243 (same R0/R1 real streams, 6/6 "
          "frame-identical) + docs/245-251/253/254/257 precedent; NOT re-tested")
    print("R_L5B_VERDICT=%s" % verdict)
    print("R_L5B_VERDICT_NOTE=%s" % vnote)
    print("R_L5B_GUARD_D251=%d" % d251_ok)
    print("R_L5B_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L5B_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L5B_GUARD_D251_DETAIL=%s" % d251_detail)
    print("R_L5B_GUARD_D246=%d" % guard246_ok)
    print("R_L5B_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L5B_REPRO_RATIO=%d" % repro_ok)
    print("R_L5B_REPRO_DETAIL=%s" % repro_detail)
    print("R_L5B_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
