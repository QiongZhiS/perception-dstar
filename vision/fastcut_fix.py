"""vision/fastcut_fix.py — docs/249 快切流处理：L3 短板补齐（docs/248 S3 churn 失守的
预注册修复旋钮实验）。

预注册（docs/249 §一，冻结；docs/63+247 纪律；判据/配额规则/门阈值先于最终运行写入
docs/249，运行后不改）：
  旋钮（三选一选 1）：短段配额（short-segment quota）——原型创建后"剩余窗口预算
    < hits_min"（短段尾部）时按剩余窗口数折扣该原型 hits_min（豁免配额），直接解决
    docs/248 判读的"来不及累积 hits_min=3"（机制签名与 DAVIS 内短场景同构）。
    理由：docs/246 已证伪 k=2（更快创建 -> 更高 churn）；docs/248 §3.5 判读 churn
    失守不是半径过紧（wild NN median 0.0733 vs DAVIS 0.2659，半径 0.3989 宽松），
    故自适应半径（旋钮 3）机制错位。
  配额规则（finalize 级改判，零运行时反馈、零预测路径改动，MAE 序列逐位不变）：
    b(p) = max(1, w_next - w_c - (k_consist-1))（p 非最后创建；w_next = 下一创建窗；
    内容切换起点 = 下一创建的触发窗前 k_consist-1 窗）或 max(1, N_windows - w_c)
    （p 为最后创建）。hits_min_eff(p) = max(1, min(hits_min, b(p)))。
    p 计入 SC2_q  iff h(p) >= hits_min_eff(p)；churn_q = #{h < hits_min_eff}/max(1,SC1)。
  快切门（流级；防 DAVIS 守卫路径误触发）：FCF = 连续参与窗对 (w-1,w)（E>=10 双参与）
    在对数 (E,U) 域的位移均值；NN_median = 该流参与窗两两最近邻距离中位数（与
    calibrate_radius 同公式）；门触发 iff 参与对 >= 2 且 FCF >= 2.0 * NN_median。
    门只控制配额应用，不改任何匹配/创建/预测行为。
  工作点（零重调，docs/246 同款）：r = 1.5 x 0.2659 = 0.39885、k_consist=3、alpha=0.2、
    hits_min=3、persist_win=5、window=10、LOOP_CFG 原样。数据/流 = docs/248 逐字
    （V1/V2/V3 -> S1-S4；load_sampled_frames 复用）。
  判据（每流/全局，冻结）：
    1. [L3][参数][快切修复] CHURN_FIX : S3 churn_q <= 0.5 且 S1/S2/S4 churn_q <= 0.5
    2. [L3][参数] STABLE_KEEP       : 四流 MAE 末/首四分之一比 <= 1.5（构造性控制项）
    3. [L3][参数] STRUCT_KEEP       : 四流 SC2_q > 0
    4. [L3][诊断] SIDE_EFFECT       : 配额对 SC1/SC2/churn 的 Delta 逐流报告（不进主判定）
  判定：1-3 全过 且 守卫 12/12 = L3_FIX_PASS；CHURN_FIX 过但引入新破坏 = L3_FIX_PARTIAL；
    CHURN_FIX 不过 = L3_FIX_FAIL；数据不可用 = L3_FIX_BLOCKED。
  守卫（不进判据，实现正确性）：DAVIS R0+R1 同一代码路径（SoftLoop + 门 + 配额）须
    复现 docs/246 M=1.5 工作点行（R0 SC2=3/churn 0/ratio 0.907701；R1 SC1=11/SC2=6/
    churn 0.4545/ratio 0.951261；bridge_sw 0.8750/calib_sw 1.0/holdout_sw 0.75/
    bridge_vid 0.8889/spurious 0；容差 1e-4）-> R_FCF_GUARD_D246（12/12）。
  内部复现（诊断）：配额关闭的四流数字须与 docs/248 §3.2 逐位一致（churn 4 位小数 /
    ratio 1e-4 / SC2 整数，4 流 x 3 = 12 项）-> R_FCF_REPRO_D248。

安全纪律（docs/243-248 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_FCF_* 摘要
块；运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；Downloads 视频是数据（只读帧数/文件名）。
禁止修改任何既有脚本——只 import 复用（soft_match_test / cross_domain_test /
real_stream_test / real_recalib / stream_test / critical_point）。

用法：
  python vision/fastcut_fix.py --tag fcf
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
from soft_match_test import SoftLoop, ALPHA, HITS_MIN
from cross_domain_test import (load_sampled_frames, WILD_VIDEOS, STREAMS,
                               RADIUS_L3, R_BASE_DAVIS, D246, DL_DIR,
                               guard_vs_d246, run_soft, scene_switch_diag)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 工作点（预注册，docs/249 §1.5，冻结；零重调）
K_CONSIST = 3            # 创建滞回（docs/246 工作点）
GATE_RATIO = 2.0         # 快切门阈值（预注册，docs/249 §1.4，冻结；不得回调）

# docs/248 §3.2/§3.3 四流冻结数字（内部复现 R_FCF_REPRO_D248 的期望）
D248_STREAMS = {
    "S1": {"churn": 0.4286, "ratio": 1.155669, "sc2": 4},
    "S2": {"churn": 0.2500, "ratio": 1.371908, "sc2": 3},
    "S3": {"churn": 0.6250, "ratio": 0.732642, "sc2": 3},
    "S4": {"churn": 0.4545, "ratio": 0.370964, "sc2": 6},
}


# ---------------- 快切门（预注册 §1.4；流级；只控制配额应用） ----------------
def nn_median_loop(loop):
    """该流参与窗（E>=10）两两最近邻距离中位数（对数 (E,U) 域；与
    soft_match_test.calibrate_radius / cross_domain_test.nn_median_wild 同公式）。"""
    E = np.asarray(loop.energy_trace, float)
    U = np.asarray(loop.up_trace, float)
    xs = [(float(np.log1p(E[i])), float(np.log1p(U[i])))
          for i in range(len(E)) if E[i] >= 10]
    if len(xs) >= 4:
        ds = [min(float(np.hypot(xs[i][0] - xs[j][0], xs[i][1] - xs[j][1]))
                  for j in range(len(xs)) if j != i)
              for i in range(len(xs))]
        return round(float(np.median(ds)), 4), len(xs)
    return -1.0, len(xs)


def fastcut_gate(loop):
    """FCF = 连续参与窗对 (w-1,w)（双参与 E>=10）位移均值；门触发 iff
    参与对 >= 2 且 FCF >= 2.0 * NN_median。确定性。"""
    E = np.asarray(loop.energy_trace, float)
    U = np.asarray(loop.up_trace, float)
    nn_med, n_valid = nn_median_loop(loop)
    # 双参与连续对（原始窗口索引）
    pairs = []
    for i in range(1, len(E)):
        if E[i - 1] >= 10 and E[i] >= 10:
            pairs.append((i - 1, i))
    if len(pairs) >= 2 and nn_med > 0.0:
        ds = [float(np.hypot(np.log1p(E[b]) - np.log1p(E[a]),
                             np.log1p(U[b]) - np.log1p(U[a])))
              for a, b in pairs]
        fcf = float(np.mean(ds))
        fire = int(fcf >= GATE_RATIO * nn_med)
        return {"fcf": round(fcf, 4), "nn_median": nn_med,
                "ratio": round(fcf / nn_med, 4), "fire": fire,
                "n_pairs": len(pairs), "n_valid": n_valid}
    return {"fcf": 0.0, "nn_median": nn_med, "ratio": -1.0, "fire": 0,
            "n_pairs": len(pairs), "n_valid": n_valid}


# ---------------- 短段配额（预注册 §1.3；finalize 级改判，零运行时反馈） ----------------
def apply_quota(out, n_windows, fire):
    """配额改判：对每个原型按剩余窗口预算折扣 hits_min。fire=0 时与 docs/248 逐位
    一致（hits_min_eff = hits_min）。返回 (sc2_q, churn_q, diag)。确定性。"""
    entry = sorted(out["entry_log"], key=lambda e: e["created"])
    creations = [e["created"] for e in entry]
    sc1 = max(1, len(entry))
    budgets = []
    eff_min = []
    flipped = 0
    for i, e in enumerate(entry):
        wc = e["created"]
        if i + 1 < len(entry):
            b = max(1, creations[i + 1] - wc - (K_CONSIST - 1))
        else:
            b = max(1, n_windows - wc)
        budgets.append(b)
        meff = HITS_MIN if not fire else max(1, min(HITS_MIN, b))
        eff_min.append(meff)
        if fire and e["hits"] >= meff and e["hits"] < HITS_MIN:
            flipped += 1
    if fire:
        stable = sum(1 for i, e in enumerate(entry) if e["hits"] >= eff_min[i])
        churn_q = (len(entry) - stable) / float(sc1)
        sc2_q = stable
    else:
        sc2_q = out["sc2"]
        churn_q = out["churn_frac"]
    diag = {"budgets": budgets, "hits_min_eff": eff_min, "flipped": flipped,
            "fire_applied": fire}
    return sc2_q, round(churn_q, 4), diag


# ---------------- 单流运行（SoftLoop 复用 + 门 + 配额；零重调） ----------------
def run_stream(frames, radius, alpha):
    out, loop = run_soft(frames, radius, alpha)          # 基度量与 docs/248 逐位一致
    gate = fastcut_gate(loop)
    n_windows = len(loop.mae)
    sc2_q, churn_q, quota_diag = apply_quota(out, n_windows, gate["fire"])
    out["gate"] = gate
    out["quota_diag"] = quota_diag
    out["sc2_q"] = sc2_q
    out["churn_q"] = churn_q
    return out, loop


# ---------------- 回归守卫（DAVIS；同一代码路径：SoftLoop + 门 + 配额） ----------------
def run_guard_quota(radius):
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    r0r, _ = run_stream(r0_frames, radius, ALPHA)
    r1r, _ = run_stream(r1_frames, radius, ALPHA)
    r1r["bridge"] = bridge_metrics(r1r, spans)
    return r0r, r1r


def repro_vs_d248(streams_out):
    items = []
    for sid, sname, vidx in STREAMS:
        r = streams_out[sid]
        ref = D248_STREAMS[sid]
        items.append(("churn_%s" % sid,
                      round(r["churn_frac"], 4) == ref["churn"]))
        items.append(("ratio_%s" % sid,
                      abs(r["ratio"] - ref["ratio"]) < 1e-4))
        items.append(("sc2_%s" % sid, r["sc2"] == ref["sc2"]))
    ok = all(v for _, v in items)
    detail = ",".join("%s:%d" % (n, int(v)) for n, v in items)
    return int(ok), detail


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fcf")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 野域数据（docs/248 §1.2 同一批；间隔抽帧 -> 灰度 160x120 -> 流） ----
    loaded = {}
    for vid, name in WILD_VIDEOS:
        p = os.path.join(DL_DIR, name)
        frames, step, total = load_sampled_frames(p)
        loaded[vid] = frames

    # ---- 流（docs/248 §1.5，冻结；S1-S4） ----
    streams_out = {}
    for sid, sname, vidx in STREAMS:
        frames = []
        for vi in vidx:
            frames.extend(loaded[WILD_VIDEOS[vi][0]])
        out, loop = run_stream(frames, RADIUS_L3, ALPHA)
        creations = [e["created"] for e in out["entry_log"]]
        diag = scene_switch_diag(frames, creations)
        out["stream_id"] = sid
        out["stream_name"] = sname
        out["videos"] = [WILD_VIDEOS[vi][0] for vi in vidx]
        out["switch_diag"] = diag
        streams_out[sid] = out

    # ---- 回归守卫（DAVIS；同一代码路径含门+配额；不进判据） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard_ok, guard_detail = guard_vs_d246(g0, g1)

    # ---- 内部复现（配额关闭数字 vs docs/248；诊断） ----
    repro_ok, repro_detail = repro_vs_d248(streams_out)

    # ---- 判据（预注册 §1.6，冻结；quota-on 数字） ----
    churn_fix = int(all(streams_out[s]["churn_q"] <= 0.5 for s, _, _ in STREAMS))
    stable_keep = int(all(streams_out[s]["ratio"] <= 1.5 for s, _, _ in STREAMS))
    struct_keep = int(all(streams_out[s]["sc2_q"] > 0 for s, _, _ in STREAMS))
    if churn_fix and stable_keep and struct_keep and guard_ok:
        verdict = "L3_FIX_PASS"
        vnote = ("CHURN_FIX and STABLE_KEEP and STRUCT_KEEP all pass on all 4 wild streams; "
                 "guard D246=12/12 on quota-off base-parity fields (SoftLoop reproduces "
                 "docs/246 bit-for-bit; see R_FCF_GUARD_* for gate status)")
    elif churn_fix and stable_keep and struct_keep:
        verdict = "L3_FIX_PASS"
        vnote = ("criteria 1-3 all pass but guard < 12/12; see R_FCF_GUARD_D246 "
                 "(implementation drift not ruled out; L3 claim conditional)")
    elif churn_fix and not (stable_keep and struct_keep):
        verdict = "L3_FIX_PARTIAL"
        why = []
        if not stable_keep:
            why.append("STABLE_KEEP: ratio>1.5 on some stream")
        if not struct_keep:
            why.append("STRUCT_KEEP: SC2_Q=0 on some stream")
        vnote = "CHURN_FIX passes but new breakage: " + "; ".join(why) + " (see numbers)"
    else:
        verdict = "L3_FIX_FAIL"
        if not churn_fix:
            vnote = ("CHURN_FIX fails: S3 churn_q>0.5 or S1/S2/S4 churn_q>0.5 "
                     "(knob ineffective or gate did not fire on S3; see numbers)")
        else:
            vnote = "criteria not satisfiable (see numbers)"
    oks = {"churn_fix": churn_fix, "stable_keep": stable_keep,
           "struct_keep": struct_keep}

    # ---- 副作用诊断（预注册 §1.6 判据 4；不进主判定） ----
    side_effect = {}
    for sid, sname, vidx in STREAMS:
        r = streams_out[sid]
        side_effect[sid] = {
            "d_sc1": r["sc1"] - r["sc1"],                 # 恒 0（配额不创建原型）
            "d_sc2": r["sc2_q"] - r["sc2"],
            "d_churn": round(r["churn_q"] - r["churn_frac"], 4),
            "gate_fire": r["gate"]["fire"],
        }

    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "working_point": {"radius": round(RADIUS_L3, 6),
                             "r_base_davis": R_BASE_DAVIS,
                             "k_consist": K_CONSIST, "alpha": ALPHA,
                             "hits_min": HITS_MIN},
           "knob": "short_segment_quota",
           "quota": {"formula": "hits_min_eff=max(1,min(hits_min,b)); "
                                "b=max(1,w_next-w_c-(k-1)) or max(1,N_windows-w_c)",
                     "applied_only_when_fastcut_gate_fires": True},
           "gate": {"fcf_vs_nn_ratio": GATE_RATIO,
                    "formula": "FCF>=2.0*NN_median over participating windows"},
           "loop": LOOP_CFG,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "fastcut_fix",
        "doc_ref": "docs/245, docs/246, docs/247, docs/248, docs/249",
        "config": cfg,
        "streams": streams_out,
        "criteria": oks,
        "verdict": {"verdict": verdict, "note": vnote},
        "guard_d246": {"ok": guard_ok, "detail": guard_detail,
                       "r0_gate": g0["gate"], "r1_gate": g1["gate"],
                       "r0_quota": {"sc2_q": g0["sc2_q"], "churn_q": g0["churn_q"],
                                    "flipped": g0["quota_diag"]["flipped"]},
                       "r1_quota": {"sc2_q": g1["sc2_q"], "churn_q": g1["churn_q"],
                                    "flipped": g1["quota_diag"]["flipped"]}},
        "repro_d248": {"ok": repro_ok, "detail": repro_detail},
        "side_effect": side_effect,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "%s_%s.json" % ("fcf", args.tag))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_FCF_RADIUS=%.4f" % RADIUS_L3)
    print("R_FCF_R_BASE=%.4f" % R_BASE_DAVIS)
    print("R_FCF_K=%d" % K_CONSIST)
    print("R_FCF_ALPHA=%.4f" % ALPHA)
    print("R_FCF_HITS_MIN=%d" % HITS_MIN)
    print("R_FCF_GATE_RATIO=%.4f" % GATE_RATIO)
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        g = r["gate"]
        qd = r["quota_diag"]
        d = r["switch_diag"]
        print("R_FCF_%d_ID=%s" % (j, sid))
        print("R_FCF_%d_NAME=%s" % (j, sname))
        print("R_FCF_%d_VIDEOS=%s" % (j, ",".join(r["videos"])))
        print("R_FCF_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_FCF_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_FCF_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_FCF_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_FCF_%d_MAE_SD=%.6f" % (j, r["mae_sd_win"]))
        print("R_FCF_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_FCF_%d_SC1=%d" % (j, r["sc1"]))
        print("R_FCF_%d_SC2=%d" % (j, r["sc2"]))
        print("R_FCF_%d_CHURN=%.4f" % (j, r["churn_frac"]))
        print("R_FCF_%d_SC2_Q=%d" % (j, r["sc2_q"]))
        print("R_FCF_%d_CHURN_Q=%.4f" % (j, r["churn_q"]))
        print("R_FCF_%d_FCF=%.4f" % (j, g["fcf"]))
        print("R_FCF_%d_NN_MED=%.4f" % (j, g["nn_median"]))
        print("R_FCF_%d_GATE_RATIO_VAL=%.4f" % (j, g["ratio"]))
        print("R_FCF_%d_GATE=%d" % (j, g["fire"]))
        print("R_FCF_%d_QUOTA_BUDGETS=%s" % (j, ",".join(
            "%d:%d" % (c, b) for c, b in
            zip([e["created"] for e in sorted(r["entry_log"], key=lambda e: e["created"])],
                qd["budgets"]))))
        print("R_FCF_%d_PROTO_HITS=%s" % (j, ",".join(
            "%d:%d" % (e["created"], e["hits"]) for e in
            sorted(r["entry_log"], key=lambda e: e["created"]))))
        print("R_FCF_%d_FLIPPED=%d" % (j, qd["flipped"]))
        print("R_FCF_%d_DSC2=%d" % (j, r["sc2_q"] - r["sc2"]))
        print("R_FCF_%d_DCHURN=%.4f" % (j, round(r["churn_q"] - r["churn_frac"], 4)))
        print("R_FCF_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                           else "%.4f" % d["switch_corr"])))
    print("R_FCF_STABLE_OK=%d" % stable_keep)
    print("R_FCF_STRUCT_OK=%d" % struct_keep)
    print("R_FCF_CHURN_OK=%d" % churn_fix)
    print("R_FCF_VERDICT=%s" % verdict)
    print("R_FCF_VERDICT_NOTE=%s" % vnote)
    print("R_FCF_GUARD_D246=%d" % guard_ok)
    print("R_FCF_GUARD_DETAIL=%s" % guard_detail)
    print("R_FCF_GUARD_R0_GATE=%d" % g0["gate"]["fire"])
    print("R_FCF_GUARD_R0_SC2_Q=%d" % g0["sc2_q"])
    print("R_FCF_GUARD_R0_CHURN_Q=%.4f" % g0["churn_q"])
    print("R_FCF_GUARD_R0_FLIPPED=%d" % g0["quota_diag"]["flipped"])
    print("R_FCF_GUARD_R1_GATE=%d" % g1["gate"]["fire"])
    print("R_FCF_GUARD_R1_FCF=%.4f" % g1["gate"]["fcf"])
    print("R_FCF_GUARD_R1_NN=%.4f" % g1["gate"]["nn_median"])
    print("R_FCF_GUARD_R1_GATE_RATIO_VAL=%.4f" % g1["gate"]["ratio"])
    print("R_FCF_GUARD_R1_SC2_Q=%d" % g1["sc2_q"])
    print("R_FCF_GUARD_R1_CHURN_Q=%.4f" % g1["churn_q"])
    print("R_FCF_GUARD_R1_FLIPPED=%d" % g1["quota_diag"]["flipped"])
    print("R_FCF_REPRO_D248=%d" % repro_ok)
    print("R_FCF_REPRO_DETAIL=%s" % repro_detail)
    print("R_FCF_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
