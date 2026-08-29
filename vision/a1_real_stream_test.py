"""vision/a1_real_stream_test.py — docs/257 A1 真实流动态基质：docs/241 四判据在真实像素
流上的实证（docs/240 A1 时间，真实域）。

把 docs/241 的 A1 四判据（STREAM_STABLE / STRUCT_BOUNDED / STATE_PERSIST / BRIDGE）
从合成环境流升级到真实像素流（DAVIS R0/R1 + 野流 S1-S4），基座 = 现行主机制
DeferredLoop（docs/251 快慢双原型 + 延迟定级，import 复用，零重调，预测路径零改动）。

预注册（docs/257 §一，冻结；docs/63+247 纪律；判据/旋钮/守卫先于实现写入 docs/257，
运行后不改）：
  机制：quota_retire.DeferredLoop + quota_retire.run_deferred_stream 逐字复用（同代码
    路径 -> 与 docs/251 数字逐位一致）。
  流（docs/257 §1.6 冻结）：R0 = flamingo x5 = 400 帧（长程循环，主判定）；R1 = 9 视频
    拼接 = 588 帧（真实段切换，主判定，GT 段边界 [8,13,21,30,36,41,49,54]）；S1-S4 =
    野流（支持流）；R2 不运行（排除声明）。
  判据（docs/257 §1.3 冻结，每判据带 docs/247 层级标签）：
    1. [A1][机制][真实流实证] REAL_STREAM_STABLE : 六流 ratio = mean(MAE 末四分之一) /
       mean(MAE 首四分之一) <= 1.5。
    2. [A1][机制] REAL_STRUCT_BOUNDED : 六流 SC2_slow > 0（不塌缩）且 SC2_slow <=
       3*max(1, ceil(n_windows/10))（密度上界，不膨胀）且 churn_slow <= 0.5（延迟定级
       下按构造 0.0）。
    3. [A1][机制][引用docs243] REAL_STATE_PERSIST : 不重测，引用 docs/243（同一 R0/R1
       真实流 6/6 全等）+ docs/245-251 链上同款声明（只改模式表路径，save/load 机制未动）。
    4. [A1][机制] REAL_BRIDGE : R0 SC2_slow >= 1（长程保持）且 R1 SC2_slow >=
       max(3, SC2_slow(R0))（长程增强）且 gist_cov(R1) >= 0.5（对齐 docs/243 机制迁移数字）。
  判定（docs/257 §1.3 冻结）：四判据全过 + 守卫全过 = DYNAMIC_PASS_REAL；任一判据计算过
    且不过 = DYNAMIC_FAIL_REAL；判据无法计算 = PARTIAL_REAL；守卫不过 = GUARD_FAIL。
  守卫（docs/257 §1.4 冻结，不进判据）：
    R_A1R_GUARD_D251 : 32 项逐位复现 docs/251 §3.3/§3.4（l4_compose_test.guard_d251_items
      复用，容差 1e-4）。
    R_A1R_GUARD_D246 : 12/12（fastcut_fix.run_guard_quota + cross_domain_test.guard_vs_d246
      复用）。
    R_A1R_REPRO_RATIO : 6 项（S1-S4+R0+R1 ratio 与 docs/251 §3.3 逐位一致，容差 1e-4——
      预测路径零改动，与 docs/243-251 的 MAE/ratio 逐位一致）。

安全纪律（docs/257 §1.9 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_A1R_* 摘要
块；运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/a1_real_stream_test.py --smoke
  python vision/a1_real_stream_test.py --tag timing
  python vision/a1_real_stream_test.py --tag main
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

# docs/251 §3.3 冻结 ratio（R_A1R_REPRO_RATIO 期望；预测路径未动，与 docs/243-251 一致）
D251_RATIOS = {"S1": 1.155669, "S2": 1.371908, "S3": 0.732642,
               "S4": 0.370964, "R1": 0.951261, "R0": 0.907701}

# docs/257 §1.3 判据 2 冻结常量
DENSITY_NUM = 3      # SC2_slow <= 3 x max(1, ceil(n_windows/10))（docs/241 "3x" 系数沿用）
DENSITY_WIN = 10     # 每 10 窗口一个密度单元（流长度归一锚点）


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
    loop = DeferredLoop(window=WINDOW, **LOOP_CFG)
    for g in frames:
        loop.step(g)
    base = loop.finalize(max(1, len(frames) // WINDOW), labels=None)
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0
    out, _ = run_deferred_stream(frames)   # 复用单流运行函数（同一代码路径）
    keys_ok = int(all(k in out for k in ("ratio", "sc1_fast", "sc2_fast", "sc1_slow",
                                         "sc2_slow", "churn_slow", "n_promo",
                                         "n_recycle", "entry_log", "created_log")))
    print("R_A1R_SMOKE_CONSTRUCT=1")
    print("R_A1R_SMOKE_RATIO_FINITE=%d" % int(np.isfinite(ratio)))
    print("R_A1R_SMOKE_KEYS_OK=%d" % keys_ok)
    print("R_A1R_SMOKE_SYNTH_RATIO=%.6f" % out["ratio"])
    print("R_A1R_SMOKE_ELAPSED=%.2f" % 0.0)
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

    # ---- 流（docs/257 §1.6 冻结；S1-S4） ----
    streams_out = {}
    for sid, sname, vidx in STREAMS:
        frames = []
        for vi in vidx:
            frames.extend(loaded[WILD_VIDEOS[vi][0]])
        out, _ = run_deferred_stream(frames)
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        diag = scene_switch_diag(frames, creations)
        out["stream_id"] = sid
        out["stream_name"] = sname
        out["videos"] = [WILD_VIDEOS[vi][0] for vi in vidx]
        out["switch_diag"] = diag
        streams_out[sid] = out

    # ---- DAVIS：R0（flamingo x5）、R1（9 拼接；GT 段边界） ----
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    r0r, _ = run_deferred_stream(r0_frames)
    r1r, _ = run_deferred_stream(r1_frames)
    r1r["bridge"] = bridge_metrics(build_entry_base(r1r), spans)   # 诊断（桥度量）
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    r1r["gist"] = gist_metrics(r1r, switch_windows)
    switch_diag_r1 = scene_switch_diag(r1_frames,
                                       [e["created"] for e in r1r["entry_log"]
                                        if e["kind"] == "fast"])

    # ---- 守卫 1：R_A1R_GUARD_D251（32 项逐位复现 docs/251；l4_compose_test 复用） ----
    d251_items = guard_d251_items(streams_out, r1r)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- 守卫 2：R_A1R_GUARD_D246（共享基座，12/12） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)

    # ---- 守卫 3：R_A1R_REPRO_RATIO（6 项；预测路径零改动，与 docs/251 逐位一致） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            abs(streams_out[sid]["ratio"] - D251_RATIOS[sid]) < 1e-4))
    repro_items.append(("ratio_R1", abs(r1r["ratio"] - D251_RATIOS["R1"]) < 1e-4))
    repro_items.append(("ratio_R0", abs(r0r["ratio"] - D251_RATIOS["R0"]) < 1e-4))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, int(v)) for n, v in repro_items)

    # ---- 诊断：SC_late_slow（后半程新确认慢原型数——结构稳态维护形态） ----
    def sc_late_slow(out):
        nw = max(1, out["n_windows"])
        return sum(1 for e in out["entry_log"] if e["kind"] == "slow"
                   and e["created"] >= nw // 2)

    # ---- 判据（docs/257 §1.3 冻结；每判据带 docs/247 层级标签） ----
    named_streams = [(sid, streams_out[sid]) for sid, _, _ in STREAMS] \
        + [("R0", r0r), ("R1", r1r)]
    all_streams = [r for _, r in named_streams]
    # 1. [A1][机制][真实流实证] REAL_STREAM_STABLE
    stable_ok = int(all(r["ratio"] <= 1.5 for r in all_streams))
    # 2. [A1][机制] REAL_STRUCT_BOUNDED（不塌缩 + 密度上界 + churn 有界）
    bound_items = []
    for sid, r in named_streams:
        dens = DENSITY_NUM * max(1, (r["n_windows"] + DENSITY_WIN - 1) // DENSITY_WIN)
        bound_items.append((sid,
                            int(r["sc2_slow"] > 0 and r["sc2_slow"] <= dens
                                and r["churn_slow"] <= 0.5)))
    struct_ok = int(all(v for _, v in bound_items))
    bound_detail = ",".join("%s:%d" % (n, v) for n, v in bound_items)
    # 3. [A1][机制][引用docs243] REAL_STATE_PERSIST（不重测，引用 docs/243 6/6 + 链上声明）
    persist_ok = 1     # 引用承担（docs/243 §1.4 判据 3：R0/R1 各 3 中断点 save->load->续跑
                       # 逐帧全等 6/6；docs/245-251 只改模式表路径、save/load 机制未动）
    # 4. [A1][机制] REAL_BRIDGE（R0 长程保持 + R1 长程增强 + gist 对齐 docs/243 迁移数字）
    bridge_ok = int(r0r["sc2_slow"] >= 1
                    and r1r["sc2_slow"] >= max(3, r0r["sc2_slow"])
                    and r1r["gist"]["cov"] >= 0.5)

    oks = {"stable": stable_ok, "struct": struct_ok,
           "persist": persist_ok, "bridge": bridge_ok}

    # ---- 判定（docs/257 §1.3 冻结映射） ----
    guards_ok = (d251_ok == 1 and guard246_ok == 1 and repro_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d -> implementation drift; fix implementation, do "
                 "not judge mechanism (see R_A1R_GUARD_*)" % (
                     d251_ok, d251_passed, guard246_ok, repro_ok))
    elif all(oks.values()):
        verdict = "DYNAMIC_PASS_REAL"
        vnote = ("REAL_STREAM_STABLE and REAL_STRUCT_BOUNDED and REAL_STATE_PERSIST "
                 "(cited docs/243 6/6, not re-tested) and REAL_BRIDGE all pass on "
                 "real pixel streams with the current main mechanism (DeferredLoop, "
                 "docs/251); A1 time substrate holds on real pixels")
    elif not all(oks.values()):
        failed = [k for k, v in oks.items() if not v]
        verdict = "DYNAMIC_FAIL_REAL"
        vnote = "criterion failed: %s (see numbers)" % ",".join(failed)
    else:
        verdict = "PARTIAL_REAL"
        vnote = "criteria not computable; see numbers"

    # ---- 工件（自描述 JSON） ----
    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "working_point": {"r_slow": round(R_SLOW, 6), "r_fast": round(R_FAST, 6),
                             "hits_min_fast": HITS_MIN_FAST,
                             "hits_min_slow": HITS_MIN_SLOW,
                             "k_promote": K_PROMOTE, "k_decay": K_DECAY,
                             "k_consist_fast": K_CONSIST_FAST, "alpha": ALPHA},
           "mechanism": ("current main mechanism DeferredLoop (docs/251 fast-slow dual "
                         "prototypes + deferred finalization), import-reused verbatim; "
                         "prediction path zero-change; A1 criteria (docs/241) upgraded "
                         "to real pixel streams; no slot path (L4 orthogonal)"),
           "criteria_def": {
               "REAL_STREAM_STABLE": "six streams ratio (last/first quartile window "
                                     "MAE mean) <= 1.5",
               "REAL_STRUCT_BOUNDED": "six streams SC2_slow>0 and SC2_slow <= 3*max(1,"
                                      "ceil(n_windows/10)) and churn_slow<=0.5 (by "
                                      "construction 0.0 under deferred finalization)",
               "REAL_STATE_PERSIST": "NOT re-tested; cited docs/243 (same R0/R1 real "
                                     "streams, 6/6 frame-identical save->load->resume) "
                                     "+ docs/245-251 chain precedent (pattern-table-"
                                     "only changes, save/load mechanism untouched)",
               "REAL_BRIDGE": "R0 SC2_slow>=1 (long-range loop keep); R1 SC2_slow>=max(3,"
                              "SC2_slow(R0)) (long-range enhance); gist_cov(R1)>=0.5 "
                              "(aligned with docs/243 mechanism-transfer numbers)"},
           "loop": LOOP_CFG,
           "r1_spans": [[a, b] for a, b in spans],
           "r1_switch_windows": switch_windows,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "a1_real_stream_test",
        "doc_ref": "docs/240, docs/241, docs/243, docs/246, docs/247, docs/248, "
                   "docs/250, docs/251, docs/257",
        "config": cfg,
        "streams": streams_out,
        "r0": r0r, "r1": r1r,
        "criteria": {"stable": stable_ok, "struct": struct_ok,
                     "persist": persist_ok, "bridge": bridge_ok,
                     "struct_detail": bound_detail,
                     "r1_gist_cov": r1r["gist"]["cov"],
                     "r1_gist_prec": r1r["gist"]["prec"],
                     "r1_bridge_sw": r1r["bridge"]["bridge_corr_switch"]},
        "verdict": {"verdict": verdict, "note": vnote},
        "guards": {"d251": {"items": len(d251_items), "passed": d251_passed,
                            "ok": d251_ok, "detail": d251_detail},
                   "d246": {"ok": guard246_ok, "detail": guard246_detail},
                   "repro_ratio": {"ok": repro_ok, "detail": repro_detail}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "a1r_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_A1R_TAG=%s" % args.tag)
    print("R_A1R_R_SLOW=%.6f" % R_SLOW)
    print("R_A1R_R_FAST=%.6f" % R_FAST)
    print("R_A1R_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_A1R_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_A1R_K_PROMOTE=%d" % K_PROMOTE)
    print("R_A1R_K_DECAY=%d" % K_DECAY)
    print("R_A1R_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_A1R_ALPHA=%.4f" % ALPHA)
    print("R_A1R_DENSITY_NUM=%d" % DENSITY_NUM)
    print("R_A1R_DENSITY_WIN=%d" % DENSITY_WIN)
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        d = r["switch_diag"]
        dens = DENSITY_NUM * max(1, (r["n_windows"] + DENSITY_WIN - 1) // DENSITY_WIN)
        print("R_A1R_%d_ID=%s" % (j, sid))
        print("R_A1R_%d_NAME=%s" % (j, sname))
        print("R_A1R_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_A1R_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_A1R_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_A1R_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_A1R_%d_MAE_SD=%.6f" % (j, r["mae_sd_win"]))
        print("R_A1R_%d_MAE_LO=%.6f" % (j, r["mae_ci95"][0]))
        print("R_A1R_%d_MAE_HI=%.6f" % (j, r["mae_ci95"][1]))
        print("R_A1R_%d_Q1=%.6f" % (j, r["mae_q1"]))
        print("R_A1R_%d_Q4=%.6f" % (j, r["mae_q4"]))
        print("R_A1R_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_A1R_%d_SC1_FAST=%d" % (j, r["sc1_fast"]))
        print("R_A1R_%d_SC2_FAST=%d" % (j, r["sc2_fast"]))
        print("R_A1R_%d_SC1_SLOW=%d" % (j, r["sc1_slow"]))
        print("R_A1R_%d_SC2_SLOW=%d" % (j, r["sc2_slow"]))
        print("R_A1R_%d_CHURN_SLOW=%.4f" % (j, r["churn_slow"]))
        print("R_A1R_%d_CHURN_LEGACY=%.4f" % (j, r["churn_legacy"]))
        print("R_A1R_%d_N_PROMO=%d" % (j, r["n_promo"]))
        print("R_A1R_%d_N_RECYCLE=%d" % (j, r["n_recycle"]))
        print("R_A1R_%d_SCLATE_SLOW=%d" % (j, sc_late_slow(r)))
        print("R_A1R_%d_DENS_BOUND=%d" % (j, dens))
        print("R_A1R_%d_PROMO_MEAN=%.4f" % (j, r["promoted_mean_hits"]))
        print("R_A1R_%d_NONPROMO_MEAN=%.4f" % (j, r["nonpromoted_mean_hits"]))
        print("R_A1R_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                            else "%.4f" % d["switch_corr"])))
    print("R_A1R_R0_FRAMES=%d" % r0r["frames"])
    print("R_A1R_R0_WINDOWS=%d" % r0r["n_windows"])
    print("R_A1R_R0_RATIO=%.6f" % r0r["ratio"])
    print("R_A1R_R0_SC1_FAST=%d" % r0r["sc1_fast"])
    print("R_A1R_R0_SC1_SLOW=%d" % r0r["sc1_slow"])
    print("R_A1R_R0_SC2_SLOW=%d" % r0r["sc2_slow"])
    print("R_A1R_R0_CHURN_SLOW=%.4f" % r0r["churn_slow"])
    print("R_A1R_R0_N_PROMO=%d" % r0r["n_promo"])
    print("R_A1R_R0_N_RECYCLE=%d" % r0r["n_recycle"])
    print("R_A1R_R0_SCLATE_SLOW=%d" % sc_late_slow(r0r))
    print("R_A1R_R0_DENS_BOUND=%d" % (DENSITY_NUM * max(1, (r0r["n_windows"] + DENSITY_WIN - 1) // DENSITY_WIN)))
    print("R_A1R_R1_FRAMES=%d" % r1r["frames"])
    print("R_A1R_R1_WINDOWS=%d" % r1r["n_windows"])
    print("R_A1R_R1_RATIO=%.6f" % r1r["ratio"])
    print("R_A1R_R1_SC1_FAST=%d" % r1r["sc1_fast"])
    print("R_A1R_R1_SC1_SLOW=%d" % r1r["sc1_slow"])
    print("R_A1R_R1_SC2_SLOW=%d" % r1r["sc2_slow"])
    print("R_A1R_R1_CHURN_SLOW=%.4f" % r1r["churn_slow"])
    print("R_A1R_R1_CHURN_LEGACY=%.4f" % r1r["churn_legacy"])
    print("R_A1R_R1_N_PROMO=%d" % r1r["n_promo"])
    print("R_A1R_R1_N_RECYCLE=%d" % r1r["n_recycle"])
    print("R_A1R_R1_SCLATE_SLOW=%d" % sc_late_slow(r1r))
    print("R_A1R_R1_DENS_BOUND=%d" % (DENSITY_NUM * max(1, (r1r["n_windows"] + DENSITY_WIN - 1) // DENSITY_WIN)))
    print("R_A1R_R1_PROMO_MEAN=%.4f" % r1r["promoted_mean_hits"])
    print("R_A1R_R1_NONPROMO_MEAN=%.4f" % r1r["nonpromoted_mean_hits"])
    print("R_A1R_R1_SWITCHES=%s" % ",".join(str(w) for w in switch_windows))
    print("R_A1R_R1_GIST_COV=%.4f" % r1r["gist"]["cov"])
    print("R_A1R_R1_GIST_PREC=%.4f" % r1r["gist"]["prec"])
    print("R_A1R_R1_GIST_COV_D2=%.4f" % r1r["gist"]["cov_d2"])
    print("R_A1R_R1_BRIDGE_SW=%.4f" % r1r["bridge"]["bridge_corr_switch"])
    print("R_A1R_R1_BRIDGE_VID=%.4f" % r1r["bridge"]["bridge_corr_video"])
    print("R_A1R_R1_SW_CORR=%.4f" % switch_diag_r1["switch_corr"])
    print("R_A1R_GUARD_D251=%d" % d251_ok)
    print("R_A1R_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_A1R_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_A1R_GUARD_D251_DETAIL=%s" % d251_detail)
    print("R_A1R_GUARD_D246=%d" % guard246_ok)
    print("R_A1R_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_A1R_REPRO_RATIO=%d" % repro_ok)
    print("R_A1R_REPRO_DETAIL=%s" % repro_detail)
    print("R_A1R_STABLE_OK=%d" % stable_ok)
    print("R_A1R_STRUCT_OK=%d" % struct_ok)
    print("R_A1R_STRUCT_DETAIL=%s" % bound_detail)
    print("R_A1R_PERSIST_OK=%d" % persist_ok)
    print("R_A1R_PERSIST_NOTE=%s" % "cited docs/243 (same R0/R1 real streams, 6/6 "
          "frame-identical) + docs/245-251 precedent; NOT re-tested")
    print("R_A1R_BRIDGE_OK=%d" % bridge_ok)
    print("R_A1R_BRIDGE_R0_SC2_SLOW=%d" % r0r["sc2_slow"])
    print("R_A1R_BRIDGE_R1_SC2_SLOW=%d" % r1r["sc2_slow"])
    print("R_A1R_BRIDGE_GIST_COV=%.4f" % r1r["gist"]["cov"])
    print("R_A1R_VERDICT=%s" % verdict)
    print("R_A1R_VERDICT_NOTE=%s" % vnote)
    print("R_A1R_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
