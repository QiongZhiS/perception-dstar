"""vision/l5_second_cell.py — docs/259 L5 桥第二格：跨技能迁移（跨流阅历注入：固定靶内容
的习得成本随阅历下降——docs/255 §三1 技能习得效率的跨流形态，docs/258 第一格的推进格）。

把 docs/258（第一格）的"同流段序数"升级为"跨流固定靶对比"：同一靶内容（同一"新技能 B"）
在"无预训练冷启动" vs "有阅历前置"两种流上下文下的习得成本对比；内容身份在配对比较中
抵消，直接消除第一格最大边界"段间差异部分是内容特性而非阅历"（docs/258 §五.2）。

基座 = 现行主机制 DeferredLoop（docs/251，import 复用**逐字**，零重调、零机制改动——本
实验只有**最小纯测量加法**＋**流构造层面的阅历注入**）。阅历注入 = 机制内流构造：① R1b
= 同一 9 视频族反序拼接（每族被不同的已累积慢原型集前置）；② S4 = V1+V2+V3 前置拼接 vs
S1/S2/S3 单流冷启动；③ R0 = flamingo×5 同内容重复 5 循环。**零序列化、零 save/load，
不触碰 docs/243/245-251/253/254/257/258 的"不重测 STATE_PERSIST"声明**。

段级习得成本度量**逐字复用第一格**（l5_skill_acq.segment_metrics/spearman_rho/
decline_stats）——同代码路径 + 同数据 → R1/S4 段表与 docs/258 §3.3 逐位一致（守卫
R_L5C_REPRO_CELL1 承担"同尺可比"的形式保证）。

预注册（docs/259 §一，冻结；docs/63+247 纪律；方向选型/机制与测量/判据/判定映射/守卫/
旋钮/流先于实现写入 docs/259，运行后不改）：
  机制：quota_retire.DeferredLoop + quota_retire.run_deferred_stream 逐字复用。
  阅历注入（流构造，零序列化）：
    R1  = 9 视频按序拼接（588 帧，GT 段边界 [8,13,21,30,36,41,49,54] -> 9 段）
    R1b = 9 视频反序拼接（588 帧，反序 spans 运行时计算段边界 -> 9 段；新测量流）
    R0  = flamingo x5 = 400 帧 -> 5 循环 x 8 窗（循环边界 = len(flamingo)//WINDOW）
    S4  = V1+V2+V3 拼接（3 段，构造性边界 = 累计采样帧数 // WINDOW）
    S1/S2/S3 = V1/V2/V3 单流（单段全流 = 冷启动对照）
  段级度量（逐字复用 l5_skill_acq）：cost_density_k = (created+recycled+promoted)/len_k
    （主度量，同 docs/258）；支持度量 gist_creation_latency/confirm_latency/promo_hit_rate
    （NA 语义同 docs/258）。
  判据（docs/259 §1.3 冻结，每判据带 docs/247 层级标签；L5 不作判据本体，标签 [L5桥]）：
    1. [L5桥][机制][行为证据] CROSS_STREAM_TRANSFER :
       (a) 野域固定靶：cost(S4 段1) <= cost(S2 单流) 且 cost(S4 段2) <= cost(S3 单流)
           且 cost(S4 段0) <= cost(S1 单流) + 0.05（V1 sanity）；
       (b) DAVIS 双序位置效应：rho(R1 9 段) <= 0 且 rho(R1b 9 段) <= 0；
       (c) 固定靶配对：8 个非对称族 count(delta_i < 0) >= 6 且 mean(delta_i) < 0，
           delta_i = cost(晚位) - cost(早位)（内容身份在配对中抵消）。
    2. [L5桥][机制][行为证据] PRACTICE_EFFECT : R0 5 循环 rho(循环序数, cost) <= 0 且
       mean(cost, 循环 1..4) <= cost(循环 0)。
    3. [L5桥][机制] TRANSFER_KEEP : 全流（S1-S4+R0+R1+R1b）ratio <= 1.5 且 SC2_slow > 0
       （S1-S4+R1+R1b）；26 段（R1 9 + R1b 9 + R0 5 + S4 3）每段有 >= 1 创建或匹配。
    4. [L5桥][机制][行为证据] PROMOTION_BEHAVIOR : n_promo>0、n_recycle>0、存在流升级
       命中率均值 > 未升级均值（docs/250/251 行为证据保持）。
  判定（docs/259 §1.4 冻结）：判据 1-4 全过 + 守卫全过 = TRANSFER_GAIN；判据 1 不过 =
    TRANSFER_FLAT；判据 1 过但 2/3/4 有不过 = PARTIAL；守卫不过 = GUARD_FAIL；
    数据不可用 = L5C_BLOCKED。
  守卫（docs/259 §1.5 冻结，不进判据）：
    R_L5C_GUARD_D251 : 32 项逐位复现 docs/251 §3.3/§3.4（l4_compose_test.guard_d251_items
      复用，容差 1e-4）——Mode OFF = 本实验机制状态。
    R_L5C_GUARD_D246 : 12/12（fastcut_fix.run_guard_quota + cross_domain_test.guard_vs_d246
      复用）。
    R_L5C_REPRO_RATIO : 6 项（S1-S4+R0+R1 ratio 与 docs/251 §3.3/docs/257 §3.2 逐位一致，
      容差 1e-4——预测路径零改动）。
    R_L5C_REPRO_CELL1 : 第一格数字复现（新增）——R1 9 段成本密度 = [1.5000,0.8000,1.5000,
      0.8889,1.0000,1.2000,0.8750,0.2000,0.8000] 且 rho = -0.5967；S4 3 段成本密度 =
      [0.7143,0.3061,0.6667] 且 rho = -0.5000（docs/258 §3.3；同代码路径必然一致，容差 1e-4）。
  确定性复现：timing/main 两轮 R_L5C_* 逐位一致（仅 TAG/ELAPSED 不同）。

安全纪律（docs/259 §1.10 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L5C_* 摘要
块；运行经 powershell **单引号**包装重定向到 logs/（docs/258 §二 轮 4 调用层教训：外层
pwsh 对双引号内 $c/$b 提前展开）；数字用纯 python 正则（vision/extract_r.py）抽取；禁止
读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/l5_second_cell.py --smoke
  python vision/l5_second_cell.py --tag timing
  python vision/l5_second_cell.py --tag main
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
# 第一格函数逐字复用（docs/259 §1.2/§1.5-4：同代码路径 -> R1/S4 段表与 docs/258 逐位一致）
from l5_skill_acq import segment_metrics, spearman_rho, decline_stats

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# docs/251 §3.3 + docs/257 §3.2 冻结 ratio（R_L5C_REPRO_RATIO 期望；预测路径未动）
D251_RATIOS = {"S1": 1.155669, "S2": 1.371908, "S3": 0.732642,
               "S4": 0.370964, "R1": 0.951261, "R0": 0.907701}

# docs/258 §3.3 第一格数字复现期望（R_L5C_REPRO_CELL1；docs/258 工件同源，逐位）
CELL1_R1_SEG_COSTS = [1.5000, 0.8000, 1.5000, 0.8889, 1.0000, 1.2000,
                      0.8750, 0.2000, 0.8000]
CELL1_R1_RHO = -0.5967
CELL1_S4_SEG_COSTS = [0.7143, 0.3061, 0.6667]
CELL1_S4_RHO = -0.5000

# docs/259 §1.3 判据 1(a) 冻结常量：V1 sanity 松弛
SANITY_SLACK = 0.05
# 判据 1(c) 冻结常量：配对 Δ 计数门槛
DELTA_NEG_MIN = 6


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
    # 段级度量函数在合成流 + 伪造边界上可计算（字段齐全、无异常）
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
    print("R_L5C_SMOKE_CONSTRUCT=1")
    print("R_L5C_SMOKE_RATIO_FINITE=%d" % int(np.isfinite(ratio)))
    print("R_L5C_SMOKE_KEYS_OK=%d" % keys_ok)
    print("R_L5C_SMOKE_SEGMETRICS_OK=%d" % segm_ok)
    print("R_L5C_SMOKE_SPEARMAN_FINITE=%d" % int(np.isfinite(rho)))
    print("R_L5C_SMOKE_SYNTH_RATIO=%.6f" % out["ratio"])
    print("R_L5C_SMOKE_ELAPSED=%.2f" % 0.0)
    return 0


# ---------------- 单段全流成本密度（冷启动对照：S1/S2/S3 整流为一段） ----------------
def whole_stream_cost(out, loop):
    segs = segment_metrics(out, loop, [0, out["n_windows"]])
    return segs[0]


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

    # ---- 流（docs/259 §1.7 冻结；S1-S4；S1/S2/S3 单流 = 冷启动对照） ----
    streams_out = {}
    stream_loops = {}
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

    # ---- DAVIS：R0（flamingo x5，练习臂）、R1（9 拼接标准序，主判定）、
    #      R1b（9 拼接反序 = 新测量流，主判定） ----
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    r1b_frames, r1b_spans, start = [], [], 0
    for v in reversed(VIDEOS):
        fr = allv[v]
        r1b_frames.extend(fr)
        r1b_spans.append((start, start + len(fr)))
        start += len(fr)
    r0r, r0_loop = run_deferred_stream(r0_frames)
    r1r, r1_loop = run_deferred_stream(r1_frames)
    r1br, r1b_loop = run_deferred_stream(r1b_frames)
    r1r["bridge"] = bridge_metrics(build_entry_base(r1r), spans)     # 诊断（D251 守卫用）
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    r1r["gist"] = gist_metrics(r1r, switch_windows)
    r1b_switch_windows = [r1b_spans[i][0] // WINDOW
                          for i in range(1, len(r1b_spans))]

    # ---- 段级度量（docs/259 §1.2 冻结；段边界只进评估统计） ----
    r1_bounds = [0] + switch_windows + [r1r["n_windows"]]
    r1_segs = segment_metrics(r1r, r1_loop, r1_bounds)
    r1b_bounds = [0] + r1b_switch_windows + [r1br["n_windows"]]
    r1b_segs = segment_metrics(r1br, r1b_loop, r1b_bounds)
    loop_wins = max(1, len(allv["flamingo"]) // WINDOW)
    r0_bounds = [k * loop_wins for k in range(5)] + [r0r["n_windows"]]
    r0_bounds = sorted(set(r0_bounds))
    r0_segs = segment_metrics(r0r, r0_loop, r0_bounds)
    s4r = streams_out["S4"]
    cum = [len(loaded[WILD_VIDEOS[0][0]]),
           len(loaded[WILD_VIDEOS[0][0]]) + len(loaded[WILD_VIDEOS[1][0]])]
    s4_bounds = [0] + [c // WINDOW for c in cum] + [s4r["n_windows"]]
    s4_bounds = sorted(set(s4_bounds))
    s4_segs = segment_metrics(s4r, stream_loops["S4"], s4_bounds)
    s1_seg = whole_stream_cost(streams_out["S1"], stream_loops["S1"])
    s2_seg = whole_stream_cost(streams_out["S2"], stream_loops["S2"])
    s3_seg = whole_stream_cost(streams_out["S3"], stream_loops["S3"])

    # ---- 跨流对比（docs/259 §1.2/§1.3 冻结口径） ----
    # 判据 1(a)：野域固定靶（前置段 vs 冷启动单流；V1 sanity + 0.05 松弛）
    v1_pre, v1_cold = s4_segs[0]["cost_density"], s1_seg["cost_density"]
    v2_pre, v2_cold = s4_segs[1]["cost_density"], s2_seg["cost_density"]
    v3_pre, v3_cold = s4_segs[2]["cost_density"], s3_seg["cost_density"]
    v1_sanity = int(v1_pre <= v1_cold + SANITY_SLACK)
    v2_ok = int(v2_pre <= v2_cold)
    v3_ok = int(v3_pre <= v3_cold)
    wild_arm = int(v1_sanity == 1 and v2_ok == 1 and v3_ok == 1)
    # 判据 1(b)：DAVIS 双序位置效应（decline_stats 复用第一格口径）
    r1_rho, r1_first, r1_last, r1_decline = decline_stats(r1_segs)
    r1b_rho, r1b_first, r1b_last, r1b_decline = decline_stats(r1b_segs)
    davi_arm = int(r1_rho <= 0.0 and r1b_rho <= 0.0)
    # 判据 1(c)：固定靶配对（内容身份在配对中抵消；dog 位于两序中位对称跳过）
    pairs = []
    for i in range(len(VIDEOS)):
        if i == len(VIDEOS) // 2:
            continue
        r1_pos, r1b_pos = i, len(VIDEOS) - 1 - i
        if r1_pos < r1b_pos:
            early = r1_segs[r1_pos]["cost_density"]
            late = r1b_segs[r1b_pos]["cost_density"]
        else:
            early = r1b_segs[r1b_pos]["cost_density"]
            late = r1_segs[r1_pos]["cost_density"]
        pairs.append({"family": VIDEOS[i], "early": early, "late": late,
                      "delta": round(late - early, 4)})
    delta_neg = sum(1 for p in pairs if p["delta"] < 0)
    delta_mean = round(float(np.mean([p["delta"] for p in pairs])), 4)
    pair_arm = int(delta_neg >= DELTA_NEG_MIN and delta_mean < 0)
    c1 = int(wild_arm == 1 and davi_arm == 1 and pair_arm == 1)
    # pooled 支持统计：R1 + R1b = 18 点（相对序数 k/8 归一）
    pooled_k, pooled_d = [], []
    for segs in (r1_segs, r1b_segs):
        m = max(1, len(segs) - 1)
        for s in segs:
            pooled_k.append(s["k"] / m)
            pooled_d.append(s["cost_density"])
    pooled_rho = spearman_rho(pooled_k, pooled_d)
    # 判据 2：R0 同内容重复练习效应（5 循环）
    r0_costs = [s["cost_density"] for s in r0_segs]
    r0_rho = spearman_rho([s["k"] for s in r0_segs], r0_costs)
    r0_loop0 = r0_costs[0]
    r0_later = float(np.mean(r0_costs[1:]))
    practice_ok = int(r0_rho <= 0.0 and r0_later <= r0_loop0)

    # ---- 守卫 1：R_L5C_GUARD_D251（Mode OFF≡docs/251，32 项逐位；l4_compose_test 复用） ----
    d251_items = guard_d251_items(streams_out, r1r)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- 守卫 2：R_L5C_GUARD_D246（共享基座，12/12） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)

    # ---- 守卫 3：R_L5C_REPRO_RATIO（6 项；预测路径零改动） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            abs(streams_out[sid]["ratio"] - D251_RATIOS[sid]) < 1e-4))
    repro_items.append(("ratio_R1", abs(r1r["ratio"] - D251_RATIOS["R1"]) < 1e-4))
    repro_items.append(("ratio_R0", abs(r0r["ratio"] - D251_RATIOS["R0"]) < 1e-4))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, int(v)) for n, v in repro_items)

    # ---- 守卫 4：R_L5C_REPRO_CELL1（第一格数字复现；l5_skill_acq 函数逐字复用） ----
    cell1_items = []
    got_r1 = [s["cost_density"] for s in r1_segs]
    for i, exp in enumerate(CELL1_R1_SEG_COSTS):
        cell1_items.append(("r1_cost%d" % i,
                            int(abs(got_r1[i] - exp) < 1e-4)))
    cell1_items.append(("r1_rho", int(abs(r1_rho - CELL1_R1_RHO) < 1e-4)))
    got_s4 = [s["cost_density"] for s in s4_segs]
    for i, exp in enumerate(CELL1_S4_SEG_COSTS):
        cell1_items.append(("s4_cost%d" % i,
                            int(abs(got_s4[i] - exp) < 1e-4)))
    cell1_items.append(("s4_rho", int(abs(decline_stats(s4_segs)[0] - CELL1_S4_RHO) < 1e-4)))
    cell1_passed = sum(1 for _, v in cell1_items)
    cell1_ok = int(all(v for _, v in cell1_items))
    cell1_detail = ",".join("%s:%d" % (n, v) for n, v in cell1_items)

    # ---- 判据 3：TRANSFER_KEEP（全流 ratio/SC2 + 26 段每段结构贡献） ----
    all_streams = [streams_out[s] for s, _, _ in STREAMS] + [r0r, r1r, r1br]
    stable_keep = int(all(r["ratio"] <= 1.5 for r in all_streams))
    struct_streams = [streams_out[s] for s, _, _ in STREAMS] + [r1r, r1br]
    struct_keep = int(all(r["sc2_slow"] > 0 for r in struct_streams))
    seg_contrib = []
    for segs in (r1_segs, r1b_segs, r0_segs, s4_segs):
        for s in segs:
            seg_contrib.append(int(s["created"] >= 1 or s["matched_windows"] >= 1))
    seg_contrib_ok = int(all(v == 1 for v in seg_contrib))
    transfer_keep = int(stable_keep == 1 and struct_keep == 1 and seg_contrib_ok == 1)
    transfer_detail = ",".join("seg%d:%d" % (i, v) for i, v in enumerate(seg_contrib))

    # ---- 判据 4：PROMOTION_BEHAVIOR（docs/250/251 行为证据保持） ----
    promo_streams = [streams_out[s] for s, _, _ in STREAMS] + [r1r]
    n_promo_total = sum(r["n_promo"] for r in promo_streams)
    n_recycle_total = sum(r["n_recycle"] for r in promo_streams)
    promo_means = [(r["promoted_mean_hits"], r["nonpromoted_mean_hits"])
                   for r in promo_streams if r["sc1_slow"] > 0]
    promo_sep = int(any(mp > mn for mp, mn in promo_means)) if promo_means else 0
    promo_ok = int(n_promo_total > 0 and n_recycle_total > 0 and promo_sep)

    oks = {"c1_cross_stream_transfer": c1, "c2_practice_effect": practice_ok,
           "c3_transfer_keep": transfer_keep, "c4_promotion": promo_ok}

    # ---- 判定（docs/259 §1.4 冻结映射） ----
    guards_ok = (d251_ok == 1 and guard246_ok == 1 and repro_ok == 1 and cell1_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d, REPRO_CELL1=%d/%d -> implementation drift; fix "
                 "implementation, do not judge mechanism (see R_L5C_GUARD_*/"
                 "R_L5C_REPRO_*)" % (d251_ok, d251_passed, guard246_ok, repro_ok,
                                     cell1_ok, cell1_passed))
    elif c1 == 1 and practice_ok == 1 and transfer_keep == 1 and promo_ok == 1:
        verdict = "TRANSFER_GAIN"
        vnote = ("CROSS_STREAM_TRANSFER (wild fixed-target: V2/V3 preceded<=cold, V1 "
                 "sanity; DAVIS dual-order rho<=0 in R1 and R1b; fixed-target pairing "
                 "delta_neg>=6/8 and mean<0) and PRACTICE_EFFECT (R0 loops) and "
                 "TRANSFER_KEEP (ratio<=1.5, SC2_slow>0, 26/26 segments contribute) and "
                 "PROMOTION_BEHAVIOR all pass; guards D251=32/32 D246=12/12 REPRO=6/6 "
                 "CELL1=OK -> cross-stream experience injection (preliminary positive "
                 "transfer evidence, L5 bridge second cell)")
    elif c1 != 1:
        verdict = "TRANSFER_FLAT"
        vnote = ("CROSS_STREAM_TRANSFER failed (wild_arm=%d, davi_arm=%d, pair_arm=%d): "
                 "experience did not transfer across streams (see per-arm numbers); "
                 "negative result reported honestly (no recalibration)"
                 % (wild_arm, davi_arm, pair_arm))
    else:
        verdict = "PARTIAL"
        vnote = ("CROSS_STREAM_TRANSFER holds but criteria 2/3/4 not all pass: "
                 "practice=%d, transfer_keep=%d, promotion=%d (see numbers)"
                 % (practice_ok, transfer_keep, promo_ok))

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
                         "(segment-level acquisition-cost metrics, post-hoc, reusing "
                         "l5_skill_acq functions verbatim); experience injection = "
                         "stream construction only (R1b reversed order / S4 prefix "
                         "concat / R0 same-content repetition): zero serialization, "
                         "zero save/load, does not touch STATE_PERSIST declarations; "
                         "segment boundaries (R1/R1b GT + S4/R0 constructional) used "
                         "for evaluation statistics only, never in mechanism decisions"),
           "criteria_def": {
               "CROSS_STREAM_TRANSFER": "wild fixed-target: cost(S4 seg1)<=cost(S2), "
                                        "cost(S4 seg2)<=cost(S3), cost(S4 seg0)<=cost(S1)"
                                        "+0.05; DAVIS dual-order: rho(R1 9 segs)<=0 and "
                                        "rho(R1b 9 segs)<=0; fixed-target pairing: >=6/8 "
                                        "asymmetric families delta<0 and mean(delta)<0",
               "PRACTICE_EFFECT": "R0 5 loops: rho(loop_idx, cost)<=0 and "
                                  "mean(cost, loops 1-4)<=cost(loop 0)",
               "TRANSFER_KEEP": "all streams (S1-S4+R0+R1+R1b) ratio<=1.5 and "
                                "SC2_slow>0 (S1-S4+R1+R1b); 26 segments (R1 9 + R1b 9 + "
                                "R0 5 + S4 3) each has >=1 creation-or-match event",
               "PROMOTION_BEHAVIOR": "n_promo>0, n_recycle>0, some stream promoted mean "
                                     "hits > nonpromoted mean hits (docs/250/251 "
                                     "behavior evidence kept)"},
           "loop": LOOP_CFG,
           "r1_spans": [[a, b] for a, b in spans],
           "r1_switch_windows": switch_windows,
           "r1_segment_boundaries": r1_bounds,
           "r1b_spans": [[a, b] for a, b in r1b_spans],
           "r1b_switch_windows": r1b_switch_windows,
           "r1b_segment_boundaries": r1b_bounds,
           "r0_loop_boundaries": r0_bounds,
           "s4_segment_boundaries": s4_bounds,
           "cross_stream": {"v1": {"preceded": v1_pre, "cold": v1_cold,
                                   "sanity": v1_sanity},
                            "v2": {"preceded": v2_pre, "cold": v2_cold, "ok": v2_ok},
                            "v3": {"preceded": v3_pre, "cold": v3_cold, "ok": v3_ok},
                            "wild_arm": wild_arm,
                            "r1_rho": round(r1_rho, 4),
                            "r1_first_half": round(r1_first, 4),
                            "r1_last_half": round(r1_last, 4),
                            "r1b_rho": round(r1b_rho, 4),
                            "r1b_first_half": round(r1b_first, 4),
                            "r1b_last_half": round(r1b_last, 4),
                            "davi_arm": davi_arm,
                            "pairs": pairs, "delta_neg": delta_neg,
                            "delta_mean": delta_mean, "pair_arm": pair_arm,
                            "pooled18_rho": round(pooled_rho, 4)},
           "practice": {"r0_rho": round(r0_rho, 4), "loop0": r0_loop0,
                        "loops1_4_mean": round(r0_later, 4), "ok": practice_ok},
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l5_second_cell",
        "doc_ref": "docs/247, docs/250, docs/251, docs/252, docs/253, docs/254, "
                   "docs/255, docs/257, docs/258, docs/259",
        "config": cfg,
        "streams": streams_out,
        "r0": r0r, "r1": r1r, "r1b": r1br,
        "segments": {"R1": {"boundaries": r1_bounds, "segments": r1_segs},
                     "R1B": {"boundaries": r1b_bounds, "segments": r1b_segs},
                     "R0": {"boundaries": r0_bounds, "segments": r0_segs},
                     "S4": {"boundaries": s4_bounds, "segments": s4_segs},
                     "S1": s1_seg, "S2": s2_seg, "S3": s3_seg},
        "criteria": {"c1_cross_stream_transfer": c1,
                     "c1_wild_arm": wild_arm, "c1_davi_arm": davi_arm,
                     "c1_pair_arm": pair_arm,
                     "c2_practice_effect": practice_ok,
                     "c3_transfer_keep": transfer_keep,
                     "c3_stable": stable_keep, "c3_struct": struct_keep,
                     "c3_seg_contrib_ok": seg_contrib_ok,
                     "c3_seg_contrib_detail": transfer_detail,
                     "c4_promotion": promo_ok,
                     "r1_gist_cov": r1r["gist"]["cov"]},
        "verdict": {"verdict": verdict, "note": vnote},
        "guards": {"d251": {"items": len(d251_items), "passed": d251_passed,
                            "ok": d251_ok, "detail": d251_detail},
                   "d246": {"ok": guard246_ok, "detail": guard246_detail},
                   "repro_ratio": {"ok": repro_ok, "detail": repro_detail},
                   "repro_cell1": {"items": len(cell1_items),
                                   "passed": cell1_passed, "ok": cell1_ok,
                                   "detail": cell1_detail}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "l5c_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L5C_TAG=%s" % args.tag)
    print("R_L5C_R_SLOW=%.6f" % R_SLOW)
    print("R_L5C_R_FAST=%.6f" % R_FAST)
    print("R_L5C_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_L5C_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_L5C_K_PROMOTE=%d" % K_PROMOTE)
    print("R_L5C_K_DECAY=%d" % K_DECAY)
    print("R_L5C_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_L5C_ALPHA=%.4f" % ALPHA)
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        d = r["switch_diag"]
        print("R_L5C_%d_ID=%s" % (j, sid))
        print("R_L5C_%d_NAME=%s" % (j, sname))
        print("R_L5C_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_L5C_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_L5C_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_L5C_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_L5C_%d_MAE_SD=%.6f" % (j, r["mae_sd_win"]))
        print("R_L5C_%d_MAE_LO=%.6f" % (j, r["mae_ci95"][0]))
        print("R_L5C_%d_MAE_HI=%.6f" % (j, r["mae_ci95"][1]))
        print("R_L5C_%d_Q1=%.6f" % (j, r["mae_q1"]))
        print("R_L5C_%d_Q4=%.6f" % (j, r["mae_q4"]))
        print("R_L5C_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_L5C_%d_SC1_FAST=%d" % (j, r["sc1_fast"]))
        print("R_L5C_%d_SC2_FAST=%d" % (j, r["sc2_fast"]))
        print("R_L5C_%d_SC1_SLOW=%d" % (j, r["sc1_slow"]))
        print("R_L5C_%d_SC2_SLOW=%d" % (j, r["sc2_slow"]))
        print("R_L5C_%d_CHURN_SLOW=%.4f" % (j, r["churn_slow"]))
        print("R_L5C_%d_CHURN_LEGACY=%.4f" % (j, r["churn_legacy"]))
        print("R_L5C_%d_N_PROMO=%d" % (j, r["n_promo"]))
        print("R_L5C_%d_N_RECYCLE=%d" % (j, r["n_recycle"]))
        print("R_L5C_%d_PROMO_MEAN=%.4f" % (j, r["promoted_mean_hits"]))
        print("R_L5C_%d_NONPROMO_MEAN=%.4f" % (j, r["nonpromoted_mean_hits"]))
        print("R_L5C_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                            else "%.4f" % d["switch_corr"])))
    print("R_L5C_R0_FRAMES=%d" % r0r["frames"])
    print("R_L5C_R0_WINDOWS=%d" % r0r["n_windows"])
    print("R_L5C_R0_RATIO=%.6f" % r0r["ratio"])
    print("R_L5C_R0_SC1_FAST=%d" % r0r["sc1_fast"])
    print("R_L5C_R0_SC1_SLOW=%d" % r0r["sc1_slow"])
    print("R_L5C_R0_SC2_SLOW=%d" % r0r["sc2_slow"])
    print("R_L5C_R0_CHURN_SLOW=%.4f" % r0r["churn_slow"])
    print("R_L5C_R0_N_PROMO=%d" % r0r["n_promo"])
    print("R_L5C_R0_N_RECYCLE=%d" % r0r["n_recycle"])
    print("R_L5C_R1_FRAMES=%d" % r1r["frames"])
    print("R_L5C_R1_WINDOWS=%d" % r1r["n_windows"])
    print("R_L5C_R1_RATIO=%.6f" % r1r["ratio"])
    print("R_L5C_R1_SC1_FAST=%d" % r1r["sc1_fast"])
    print("R_L5C_R1_SC1_SLOW=%d" % r1r["sc1_slow"])
    print("R_L5C_R1_SC2_SLOW=%d" % r1r["sc2_slow"])
    print("R_L5C_R1_CHURN_SLOW=%.4f" % r1r["churn_slow"])
    print("R_L5C_R1_CHURN_LEGACY=%.4f" % r1r["churn_legacy"])
    print("R_L5C_R1_N_PROMO=%d" % r1r["n_promo"])
    print("R_L5C_R1_N_RECYCLE=%d" % r1r["n_recycle"])
    print("R_L5C_R1_PROMO_MEAN=%.4f" % r1r["promoted_mean_hits"])
    print("R_L5C_R1_NONPROMO_MEAN=%.4f" % r1r["nonpromoted_mean_hits"])
    print("R_L5C_R1_SWITCHES=%s" % ",".join(str(w) for w in switch_windows))
    print("R_L5C_R1_GIST_COV=%.4f" % r1r["gist"]["cov"])
    print("R_L5C_R1_GIST_PREC=%.4f" % r1r["gist"]["prec"])
    print("R_L5C_R1_GIST_COV_D2=%.4f" % r1r["gist"]["cov_d2"])
    print("R_L5C_R1_BRIDGE_SW=%.4f" % r1r["bridge"]["bridge_corr_switch"])
    print("R_L5C_R1_BRIDGE_VID=%.4f" % r1r["bridge"]["bridge_corr_video"])
    print("R_L5C_R1B_FRAMES=%d" % r1br["frames"])
    print("R_L5C_R1B_WINDOWS=%d" % r1br["n_windows"])
    print("R_L5C_R1B_RATIO=%.6f" % r1br["ratio"])
    print("R_L5C_R1B_SC1_FAST=%d" % r1br["sc1_fast"])
    print("R_L5C_R1B_SC1_SLOW=%d" % r1br["sc1_slow"])
    print("R_L5C_R1B_SC2_SLOW=%d" % r1br["sc2_slow"])
    print("R_L5C_R1B_CHURN_SLOW=%.4f" % r1br["churn_slow"])
    print("R_L5C_R1B_CHURN_LEGACY=%.4f" % r1br["churn_legacy"])
    print("R_L5C_R1B_N_PROMO=%d" % r1br["n_promo"])
    print("R_L5C_R1B_N_RECYCLE=%d" % r1br["n_recycle"])
    print("R_L5C_R1B_SWITCHES=%s" % ",".join(str(w) for w in r1b_switch_windows))
    # R1 段级度量（9 段）
    for s in r1_segs:
        print("R_L5C_R1_SEG_%d_W_LO=%d" % (s["k"], s["w_lo"]))
        print("R_L5C_R1_SEG_%d_W_HI=%d" % (s["k"], s["w_hi"]))
        print("R_L5C_R1_SEG_%d_LEN=%d" % (s["k"], s["len"]))
        print("R_L5C_R1_SEG_%d_CREATED=%d" % (s["k"], s["created"]))
        print("R_L5C_R1_SEG_%d_RECYCLED=%d" % (s["k"], s["recycled"]))
        print("R_L5C_R1_SEG_%d_PROMOTED=%d" % (s["k"], s["promoted"]))
        print("R_L5C_R1_SEG_%d_MATCHED=%d" % (s["k"], s["matched_windows"]))
        print("R_L5C_R1_SEG_%d_GIST_LAT=%s" % (
            s["k"], ("NA" if s["gist_creation_latency"] is None
                     else "%d" % s["gist_creation_latency"])))
        print("R_L5C_R1_SEG_%d_CONFIRM_LAT=%s" % (
            s["k"], ("NA" if s["confirm_latency"] is None
                     else "%d" % s["confirm_latency"])))
        print("R_L5C_R1_SEG_%d_PROMO_HIT=%s" % (
            s["k"], ("NA" if s["promo_hit_rate"] is None
                     else "%.4f" % s["promo_hit_rate"])))
        print("R_L5C_R1_SEG_%d_COST_DENS=%.4f" % (s["k"], s["cost_density"]))
    # R1b 段级度量（9 段）
    for s in r1b_segs:
        print("R_L5C_R1B_SEG_%d_W_LO=%d" % (s["k"], s["w_lo"]))
        print("R_L5C_R1B_SEG_%d_W_HI=%d" % (s["k"], s["w_hi"]))
        print("R_L5C_R1B_SEG_%d_LEN=%d" % (s["k"], s["len"]))
        print("R_L5C_R1B_SEG_%d_CREATED=%d" % (s["k"], s["created"]))
        print("R_L5C_R1B_SEG_%d_RECYCLED=%d" % (s["k"], s["recycled"]))
        print("R_L5C_R1B_SEG_%d_PROMOTED=%d" % (s["k"], s["promoted"]))
        print("R_L5C_R1B_SEG_%d_MATCHED=%d" % (s["k"], s["matched_windows"]))
        print("R_L5C_R1B_SEG_%d_GIST_LAT=%s" % (
            s["k"], ("NA" if s["gist_creation_latency"] is None
                     else "%d" % s["gist_creation_latency"])))
        print("R_L5C_R1B_SEG_%d_CONFIRM_LAT=%s" % (
            s["k"], ("NA" if s["confirm_latency"] is None
                     else "%d" % s["confirm_latency"])))
        print("R_L5C_R1B_SEG_%d_PROMO_HIT=%s" % (
            s["k"], ("NA" if s["promo_hit_rate"] is None
                     else "%.4f" % s["promo_hit_rate"])))
        print("R_L5C_R1B_SEG_%d_COST_DENS=%.4f" % (s["k"], s["cost_density"]))
    # R0 循环（5 循环）
    for s in r0_segs:
        print("R_L5C_R0_LOOP_%d_CREATED=%d" % (s["k"], s["created"]))
        print("R_L5C_R0_LOOP_%d_RECYCLED=%d" % (s["k"], s["recycled"]))
        print("R_L5C_R0_LOOP_%d_PROMOTED=%d" % (s["k"], s["promoted"]))
        print("R_L5C_R0_LOOP_%d_MATCHED=%d" % (s["k"], s["matched_windows"]))
        print("R_L5C_R0_LOOP_%d_COST_DENS=%.4f" % (s["k"], s["cost_density"]))
    # S4 段级度量（3 段）
    for s in s4_segs:
        print("R_L5C_S4_SEG_%d_W_LO=%d" % (s["k"], s["w_lo"]))
        print("R_L5C_S4_SEG_%d_W_HI=%d" % (s["k"], s["w_hi"]))
        print("R_L5C_S4_SEG_%d_LEN=%d" % (s["k"], s["len"]))
        print("R_L5C_S4_SEG_%d_CREATED=%d" % (s["k"], s["created"]))
        print("R_L5C_S4_SEG_%d_RECYCLED=%d" % (s["k"], s["recycled"]))
        print("R_L5C_S4_SEG_%d_PROMOTED=%d" % (s["k"], s["promoted"]))
        print("R_L5C_S4_SEG_%d_MATCHED=%d" % (s["k"], s["matched_windows"]))
        print("R_L5C_S4_SEG_%d_GIST_LAT=%s" % (
            s["k"], ("NA" if s["gist_creation_latency"] is None
                     else "%d" % s["gist_creation_latency"])))
        print("R_L5C_S4_SEG_%d_CONFIRM_LAT=%s" % (
            s["k"], ("NA" if s["confirm_latency"] is None
                     else "%d" % s["confirm_latency"])))
        print("R_L5C_S4_SEG_%d_PROMO_HIT=%s" % (
            s["k"], ("NA" if s["promo_hit_rate"] is None
                     else "%.4f" % s["promo_hit_rate"])))
        print("R_L5C_S4_SEG_%d_COST_DENS=%.4f" % (s["k"], s["cost_density"]))
    # 冷启动单段（S1/S2/S3 全流）
    print("R_L5C_S1_COST_DENS=%.4f" % s1_seg["cost_density"])
    print("R_L5C_S2_COST_DENS=%.4f" % s2_seg["cost_density"])
    print("R_L5C_S3_COST_DENS=%.4f" % s3_seg["cost_density"])
    # 跨流对比（判据 1）
    print("R_L5C_V1_PRE=%.4f" % v1_pre)
    print("R_L5C_V1_COLD=%.4f" % v1_cold)
    print("R_L5C_V1_SANITY=%d" % v1_sanity)
    print("R_L5C_V2_PRE=%.4f" % v2_pre)
    print("R_L5C_V2_COLD=%.4f" % v2_cold)
    print("R_L5C_V2_OK=%d" % v2_ok)
    print("R_L5C_V3_PRE=%.4f" % v3_pre)
    print("R_L5C_V3_COLD=%.4f" % v3_cold)
    print("R_L5C_V3_OK=%d" % v3_ok)
    print("R_L5C_WILD_ARM=%d" % wild_arm)
    print("R_L5C_R1_RHO=%.4f" % r1_rho)
    print("R_L5C_R1_FIRST_HALF=%.4f" % r1_first)
    print("R_L5C_R1_LAST_HALF=%.4f" % r1_last)
    print("R_L5C_R1B_RHO=%.4f" % r1b_rho)
    print("R_L5C_R1B_FIRST_HALF=%.4f" % r1b_first)
    print("R_L5C_R1B_LAST_HALF=%.4f" % r1b_last)
    print("R_L5C_DAVI_ARM=%d" % davi_arm)
    for p in pairs:
        print("R_L5C_PAIR_%s_EARLY=%.4f" % (p["family"], p["early"]))
        print("R_L5C_PAIR_%s_LATE=%.4f" % (p["family"], p["late"]))
        print("R_L5C_PAIR_%s_DELTA=%.4f" % (p["family"], p["delta"]))
    print("R_L5C_DELTA_NEG=%d" % delta_neg)
    print("R_L5C_DELTA_MEAN=%.4f" % delta_mean)
    print("R_L5C_PAIR_ARM=%d" % pair_arm)
    print("R_L5C_POOLED18_RHO=%.4f" % pooled_rho)
    # 练习效应（判据 2）
    print("R_L5C_R0_RHO=%.4f" % r0_rho)
    print("R_L5C_R0_LOOP0=%.4f" % r0_loop0)
    print("R_L5C_R0_LOOPS1_4_MEAN=%.4f" % r0_later)
    print("R_L5C_PRACTICE_OK=%d" % practice_ok)
    # 判据与判定
    print("R_L5C_C1_OK=%d" % c1)
    print("R_L5C_C2_OK=%d" % practice_ok)
    print("R_L5C_C3_OK=%d" % transfer_keep)
    print("R_L5C_C4_OK=%d" % promo_ok)
    print("R_L5C_STABLE_OK=%d" % stable_keep)
    print("R_L5C_STRUCT_OK=%d" % struct_keep)
    print("R_L5C_SEG_CONTRIB_OK=%d" % seg_contrib_ok)
    print("R_L5C_SEG_CONTRIB_DETAIL=%s" % transfer_detail)
    print("R_L5C_PROMO_OK=%d" % promo_ok)
    print("R_L5C_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L5C_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L5C_PERSIST_OK=1")
    print("R_L5C_PERSIST_NOTE=%s" % "cited docs/243 (same R0/R1 real streams, 6/6 "
          "frame-identical) + docs/245-251/253/254/257/258 precedent; NOT re-tested; "
          "experience injection = stream construction only, zero serialization")
    print("R_L5C_VERDICT=%s" % verdict)
    print("R_L5C_VERDICT_NOTE=%s" % vnote)
    print("R_L5C_GUARD_D251=%d" % d251_ok)
    print("R_L5C_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L5C_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L5C_GUARD_D251_DETAIL=%s" % d251_detail)
    print("R_L5C_GUARD_D246=%d" % guard246_ok)
    print("R_L5C_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L5C_REPRO_RATIO=%d" % repro_ok)
    print("R_L5C_REPRO_DETAIL=%s" % repro_detail)
    print("R_L5C_REPRO_CELL1=%d" % cell1_ok)
    print("R_L5C_REPRO_CELL1_ITEMS=%d" % len(cell1_items))
    print("R_L5C_REPRO_CELL1_PASSED=%d" % cell1_passed)
    print("R_L5C_REPRO_CELL1_DETAIL=%s" % cell1_detail)
    print("R_L5C_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
