"""vision/real_recalib.py — docs/244 真实分档校准与桥复测（docs/243 短板的直接续）。

docs/243 verdict=REAL_PARTIAL：机制结构侧迁移成立，但桥实验失败（bridge_corr_switch
0.25 —— toy 调的粗能量分档把 9 个真实场景压成 2 个基签名）。本实验用真实视频统计
重校准能量分档（c0/c1），复测桥对应率，并用留出视频检验泛化（防校准循环）。

预注册（docs/244 §1，冻结；判据先于最终运行写入 docs/244）：
  校准集（5 视频）：flamingo / surf / bear / camel / dog
  留出集（4 视频）：blackswan / car-turn / motorbike / soccerball
  分档法：校准集拼接流（R1 前 367 帧 = 校准段）的窗口级事件能量分布，等频分位数
    分档：c0（总能量 E）与 c1（上区能量 U）各取 20/40/60/80 分位数，1 位小数取整、
    去重严格递增。标签无关——禁止逐场景归一化（泄漏场景边界，判据平凡化）。
  判据：
    1. BRIDGE_FIX  : 完整 9 视频流 bridge_corr_switch >= 0.5（docs/243 为 0.25）
    2. HELD_OUT    : 留出视频切换对应率 holdout_switch >= 0.5（calib_switch 分开报告）
    3. STRUCT_STILL: SC2 > 0 且 churn <= 0.5（R0 与 R1 均要求）
    4. STABLE_STILL: MAE 末/首四分之一比 <= 1.5（R0 与 R1 均要求）
  判定：全过 = RECAL_PASS；3/4 过但 1/2 不过 = RECAL_PARTIAL（1 过 2 不过 = 分档
  过拟合校准集）；3 或 4 不过 = RECAL_FAIL。

安全纪律（docs/243 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_RRC_* 摘要块；
数字用纯 python 正则（vision/extract_r.py）抽取；禁止读日志/JSON 原文。禁止修改
stream_test.py / compose_test.py / critical_point.py / real_stream_test.py 等既有脚本
——只 import 复用。

用法：
  python vision/real_recalib.py --tag main
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np

from critical_point import mean_sd, bootstrap_ci
from compose_test import CompLoop, ENERGY_BINS, UPPER_BINS
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, RESIZE, WINDOW, LEVEL_NAMES

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 校准/留出划分（预注册，docs/244 §1.2，冻结）
CALIB_VIDEOS = ["flamingo", "surf", "bear", "camel", "dog"]
HOLDOUT_VIDEOS = ["blackswan", "car-turn", "motorbike", "soccerball"]
QUANTILES = (20, 40, 60, 80)          # 等频分位数（预注册，docs/244 §1.4）


# ---------------- 分档校准（只依赖校准集统计；单次确定性 pass） ----------------
def calibrate_bins(calib_frames):
    """校准 pass：默认 LOOP_CFG 跑校准集拼接流（bins 不改事件掩码，E/U 与测试 pass
    相同），收集每窗口总事件量 E 与上区（y<68）事件量 U，返回
    (energy_bins_new, bbox_bins_new, stats)。"""
    loop = CompLoop(window=WINDOW, **LOOP_CFG)
    for g in calib_frames:
        loop.step(g)
    loop.finalize(max(1, len(calib_frames) // WINDOW), labels=None)   # 刷尾窗
    E = np.asarray(loop.energy_trace, float)
    U = np.asarray(loop.up_trace, float)

    def bounds(vals):
        b = [round(float(np.percentile(vals, q)), 1) for q in QUANTILES]
        out = []
        for x in b:
            if not out or x > out[-1]:
                out.append(x)
        return out

    eb = bounds(E)
    ub = bounds(U)
    stats = {"n_windows": int(len(E)),
             "e_min": round(float(E.min()), 1), "e_med": round(float(np.median(E)), 1),
             "e_max": round(float(E.max()), 1),
             "u_min": round(float(U.min()), 1), "u_med": round(float(np.median(U)), 1),
             "u_max": round(float(U.max()), 1)}
    return eb, ub, stats


# ---------------- 桥度量（docs/243 §1.4 定义 + 校准/留出分列） ----------------
def bridge_metrics(base, spans):
    """条目创建时点 vs 视频跨度对应。切换索引 j（0 基）对应视频 i=j+1；
    校准切换 = 视频 1..4（surf/bear/camel/dog），留出切换 = 视频 5..8
    （blackswan/car-turn/motorbike/soccerball）。"""
    entries = [(e["created"], list(e["key"])) for e in base["entry_log"]]
    sw_enc = []
    for i in range(1, len(spans)):
        sp = spans[i]
        sw_enc.append(int(any(sp[0] <= w * WINDOW < sp[1] for w, _ in entries)))
    enc_vid = sum(1 for sp in spans
                  if any(sp[0] <= w * WINDOW < sp[1] for w, _ in entries))
    spur = sum(1 for w, _ in entries
               if not any(sp[0] <= w * WINDOW < sp[1] for sp in spans))
    calib = sw_enc[0:4]
    holdout = sw_enc[4:8]
    return {
        "switches": len(sw_enc),
        "encoded_sw": sum(sw_enc),
        "bridge_corr_switch": round(sum(sw_enc) / max(1, len(sw_enc)), 4),
        "calib_switches": len(calib), "encoded_calib_sw": sum(calib),
        "calib_switch": round(sum(calib) / max(1, len(calib)), 4),
        "holdout_switches": len(holdout), "encoded_holdout_sw": sum(holdout),
        "holdout_switch": round(sum(holdout) / max(1, len(holdout)), 4),
        "videos": len(spans), "encoded_vid": enc_vid,
        "bridge_corr_video": round(enc_vid / max(1, len(spans)), 4),
        "churn": base["churn_frac"],
        "spurious": spur,
        "spurious_rate": round(spur / max(1, base["sc1"]), 4),
        "sw_enc_flags": sw_enc,
        "vid_enc_flags": [int(any(sp[0] <= w * WINDOW < sp[1] for w, _ in entries))
                          for sp in spans],
        "sc1": base["sc1"], "sc2": base["sc2"],
        "promo": base["n_promo"],
    }


# ---------------- 运行一个级（新分档；无持久化——本实验不重测 STATE_PERSIST） ----------------
def run_level(lvcode, frames, spans, loop_cfg, out_dir, tag):
    n_frames = len(frames)
    loop = CompLoop(window=WINDOW, **loop_cfg)
    for g in frames:
        loop.step(g)
    base = loop.finalize(max(1, n_frames // WINDOW), labels=None)

    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0
    mae_m, mae_sd = mean_sd(list(mae_arr))
    mae_lo, mae_hi = bootstrap_ci(list(mae_arr))

    out = dict(base)
    out.update({"level": lvcode, "name": LEVEL_NAMES[lvcode], "frames": n_frames,
                "n_windows": len(mae_arr),
                "mae_mean_win": round(mae_m, 6), "mae_sd_win": round(mae_sd, 6),
                "mae_ci95": [round(mae_lo, 6), round(mae_hi, 6)],
                "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
                "ratio": round(ratio, 6), "pin_frac": base["pin_frac"],
                "theta_mean": base["theta_mean"]})
    if lvcode == 31:
        out["bridge"] = bridge_metrics(base, spans)
    return out, loop


# ---------------- 逐场景签名诊断（不进判据） ----------------
def per_scene_sig(loop, spans, names):
    rows = []
    for i, (s0, s1) in enumerate(spans):
        sigs = []
        for w, (c0, c1, _c2) in enumerate(loop.sig_trace):
            if c0 is None:
                continue
            if s0 <= w * WINDOW < s1:
                sigs.append((c0, c1))
        if sigs:
            cnt = Counter(sigs)
            mode, n = cnt.most_common(1)[0]
            rows.append({"video": names[i], "span": [s0, s1],
                         "n_windows": len(sigs),
                         "mode_sig": list(mode),
                         "mode_frac": round(n / len(sigs), 4),
                         "n_unique_sigs": len(cnt)})
        else:
            rows.append({"video": names[i], "span": [s0, s1],
                         "n_windows": 0, "mode_sig": None,
                         "mode_frac": 0.0, "n_unique_sigs": 0})
    return rows


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rrc")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 数据（160x120 灰度，docs/243 预处理冻结） ----
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0 = allv["flamingo"] * 5                      # R0：flamingo x5 = 400 帧
    r1, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    calib_end = sum(len(allv[v]) for v in CALIB_VIDEOS)
    calib_frames = r1[:calib_end]                  # 校准段 = R1 前 367 帧

    # ---- 分档校准（只依赖校准集；单次确定性 pass） ----
    eb, ub, calib_stats = calibrate_bins(calib_frames)
    loop_cfg_new = dict(LOOP_CFG)
    loop_cfg_new["energy_bins"] = eb
    loop_cfg_new["bbox_bins"] = ub

    # ---- 主运行（新分档）：R0 + R1 ----
    r0r, _ = run_level(30, r0, None, loop_cfg_new, args.out_dir, args.tag)
    r1r, r1_loop = run_level(31, r1, spans, loop_cfg_new, args.out_dir, args.tag)
    r1r["per_scene_sig"] = per_scene_sig(r1_loop, spans, VIDEOS)

    # ---- 诊断：旧分档 R1 回归（应与 docs/243 一致；不进判据） ----
    old_r1, _ = run_level(31, r1, spans, LOOP_CFG, args.out_dir, args.tag)

    # ---- 判据（预注册冻结，docs/244 §1.6） ----
    bridge_ok = int(r1r["bridge"]["bridge_corr_switch"] >= 0.5)
    heldout_ok = int(r1r["bridge"]["holdout_switch"] >= 0.5)
    struct_ok = int(r0r["sc2"] > 0 and r0r["churn_frac"] <= 0.5
                    and r1r["sc2"] > 0 and r1r["churn_frac"] <= 0.5)
    stable_ok = int(r0r["ratio"] <= 1.5 and r1r["ratio"] <= 1.5)
    oks = {"bridge": bridge_ok, "heldout": heldout_ok,
           "struct": struct_ok, "stable": stable_ok}
    if not (struct_ok and stable_ok):
        verdict = "RECAL_FAIL"
        vnote = "recalibration broke structure/stability (STRUCT_STILL or STABLE_STILL failed; see numbers)"
    elif bridge_ok and heldout_ok:
        verdict = "RECAL_PASS"
        vnote = "BRIDGE_FIX and HELD_OUT and STRUCT_STILL and STABLE_STILL all pass"
    else:
        verdict = "RECAL_PARTIAL"
        if bridge_ok and not heldout_ok:
            vnote = "BRIDGE_FIX passes but HELD_OUT fails: binning overfits calibration set (calib switches encoded, holdout not)"
        elif not bridge_ok:
            vnote = "BRIDGE_FIX fails: recalibration did not fix bridge (see numbers)"
        else:
            vnote = "see numbers"

    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "videos": VIDEOS, "calib_videos": CALIB_VIDEOS,
           "holdout_videos": HOLDOUT_VIDEOS, "quantiles": list(QUANTILES),
           "loop": LOOP_CFG,
           "bins_old": {"energy_bins": list(ENERGY_BINS),
                        "bbox_bins": list(UPPER_BINS)},
           "bins_new": {"energy_bins": eb, "bbox_bins": ub},
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "real_recalib",
        "doc_ref": "docs/241, docs/243, docs/244",
        "config": cfg,
        "calib_stats": calib_stats,
        "levels": {"30": r0r, "31": r1r},
        "diag_old_bins_r1": old_r1,
        "criteria": oks,
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "rrc_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    b31 = r1r["bridge"]
    print("R_RRC_FRAMES_R0=%d" % r0r["frames"])
    print("R_RRC_WINDOWS_R0=%d" % r0r["n_windows"])
    print("R_RRC_MAE_R0=%.6f" % r0r["mae_mean_win"])
    print("R_RRC_MAE_SD_R0=%.6f" % r0r["mae_sd_win"])
    print("R_RRC_MAE_LO_R0=%.6f" % r0r["mae_ci95"][0])
    print("R_RRC_MAE_HI_R0=%.6f" % r0r["mae_ci95"][1])
    print("R_RRC_Q1_R0=%.6f" % r0r["mae_q1"])
    print("R_RRC_Q4_R0=%.6f" % r0r["mae_q4"])
    print("R_RRC_RATIO_R0=%.6f" % r0r["ratio"])
    print("R_RRC_SC1_R0=%d" % r0r["sc1"])
    print("R_RRC_SC2_R0=%d" % r0r["sc2"])
    print("R_RRC_A2_R0=%d" % r0r["arity2_stable"])
    print("R_RRC_A3_R0=%d" % r0r["arity3_stable"])
    print("R_RRC_COMP_R0=%.4f" % r0r["compound_frac"])
    print("R_RRC_CHURN_R0=%.4f" % r0r["churn_frac"])
    print("R_RRC_PROMO_R0=%d" % r0r["n_promo"])
    print("R_RRC_PIN_R0=%.4f" % r0r["pin_frac"])
    print("R_RRC_FRAMES_R1=%d" % r1r["frames"])
    print("R_RRC_WINDOWS_R1=%d" % r1r["n_windows"])
    print("R_RRC_MAE_R1=%.6f" % r1r["mae_mean_win"])
    print("R_RRC_MAE_SD_R1=%.6f" % r1r["mae_sd_win"])
    print("R_RRC_MAE_LO_R1=%.6f" % r1r["mae_ci95"][0])
    print("R_RRC_MAE_HI_R1=%.6f" % r1r["mae_ci95"][1])
    print("R_RRC_Q1_R1=%.6f" % r1r["mae_q1"])
    print("R_RRC_Q4_R1=%.6f" % r1r["mae_q4"])
    print("R_RRC_RATIO_R1=%.6f" % r1r["ratio"])
    print("R_RRC_SC1_R1=%d" % r1r["sc1"])
    print("R_RRC_SC2_R1=%d" % r1r["sc2"])
    print("R_RRC_A2_R1=%d" % r1r["arity2_stable"])
    print("R_RRC_A3_R1=%d" % r1r["arity3_stable"])
    print("R_RRC_COMP_R1=%.4f" % r1r["compound_frac"])
    print("R_RRC_CHURN_R1=%.4f" % r1r["churn_frac"])
    print("R_RRC_PROMO_R1=%d" % r1r["n_promo"])
    print("R_RRC_PIN_R1=%.4f" % r1r["pin_frac"])
    print("R_RRC_SWITCHES=%d" % b31["switches"])
    print("R_RRC_BRIDGE_SW=%.4f" % b31["bridge_corr_switch"])
    print("R_RRC_BRIDGE_VID=%.4f" % b31["bridge_corr_video"])
    print("R_RRC_CALIB_SWITCHES=%d" % b31["calib_switches"])
    print("R_RRC_CALIB_ENC=%d" % b31["encoded_calib_sw"])
    print("R_RRC_CALIB_SW=%.4f" % b31["calib_switch"])
    print("R_RRC_HOLDOUT_SWITCHES=%d" % b31["holdout_switches"])
    print("R_RRC_HOLDOUT_ENC=%d" % b31["encoded_holdout_sw"])
    print("R_RRC_HOLDOUT_SW=%.4f" % b31["holdout_switch"])
    print("R_RRC_SPURIOUS=%d" % b31["spurious"])
    print("R_RRC_SPUR_RATE=%.4f" % b31["spurious_rate"])
    print("R_RRC_SW_ENC=%s" % ",".join(str(v) for v in b31["sw_enc_flags"]))
    print("R_RRC_VID_ENC=%s" % ",".join(str(v) for v in b31["vid_enc_flags"]))
    print("R_RRC_ENTRIES=%s" % ";".join(
        "%d|%d|%d|%s" % (e["created"], e["arity"], e["hits"],
                         ",".join(str(v) for v in e["key"]))
        for e in r1r["entry_log"]))
    print("R_RRC_SCENE_SIG=%s" % ";".join(
        "%s:%s:%d" % (row["video"],
                      ("NA" if row["mode_sig"] is None
                       else ",".join(str(v) for v in row["mode_sig"])),
                      row["n_unique_sigs"])
        for row in r1r["per_scene_sig"]))
    print("R_RRC_OLD_BRIDGE_SW=%.4f" % old_r1["bridge"]["bridge_corr_switch"])
    print("R_RRC_OLD_SC2=%d" % old_r1["sc2"])
    print("R_RRC_OLD_CHURN=%.4f" % old_r1["churn_frac"])
    print("R_RRC_CALIB_WINDOWS=%d" % calib_stats["n_windows"])
    print("R_RRC_ENERGY_MIN=%.1f" % calib_stats["e_min"])
    print("R_RRC_ENERGY_MED=%.1f" % calib_stats["e_med"])
    print("R_RRC_ENERGY_MAX=%.1f" % calib_stats["e_max"])
    print("R_RRC_UPPER_MED=%.1f" % calib_stats["u_med"])
    print("R_RRC_BINS_OLD=%s" % ",".join("%.1f" % x for x in ENERGY_BINS)
          + ";" + ",".join("%.1f" % x for x in UPPER_BINS))
    print("R_RRC_BINS_NEW=%s" % ",".join("%.1f" % x for x in eb)
          + ";" + ",".join("%.1f" % x for x in ub))
    print("R_RRC_BRIDGE_OK=%d" % bridge_ok)
    print("R_RRC_HELDOUT_OK=%d" % heldout_ok)
    print("R_RRC_STRUCT_OK=%d" % struct_ok)
    print("R_RRC_STABLE_OK=%d" % stable_ok)
    print("R_RRC_VERDICT=%s" % verdict)
    print("R_RRC_VERDICT_NOTE=%s" % vnote)
    print("R_RRC_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
