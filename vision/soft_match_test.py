"""vision/soft_match_test.py — docs/245 软匹配：稳定-判别张力的解（docs/243/244 两格续）。

docs/243（旧粗档）保一致性但判别差（bridge 0.25，9 场景坍缩 2 基签名）；docs/244
（等频分位数档）破坏一致性（R0 SC2=0，结构坍缩）。假设：张力来自"精确档位匹配"这个
量化设计本身。本实验把"窗口签名 -> 精确档位 -> 精确相等匹配"替换为"窗口特征 ->
连续原型距离 <= r 的软匹配"（bins 不参与主匹配路径）。

预注册（docs/245 §1，冻结；判据先于最终运行写入 docs/245）：
  特征空间：x_w = (ln(1+E_w), ln(1+U_w))；E=窗口事件像素总数，U=上区(y<68)事件像素数。
  原型：连续空间中心 mu + 命中统计，无档位量化，不退休，无 arity-3（compound=0 按设计）。
  匹配：最近原型欧氏距离 <= r -> 命中（hits+=1）；自适应启用（alpha=0.2 滑动中心）。
  创建：连续 k_consist=3 个参与窗口无原型在 r 内 -> 在当前窗口特征处建新原型（滞回）。
  半径：r = 校准集（flamingo/surf/bear/camel/dog，R1 前 367 帧）参与窗口最近邻距离的
    中位数（单次确定性校准；留出集 blackswan/car-turn/motorbike/soccerball 零贡献；
    禁止逐场景归一化）。
  判据：
    1. BRIDGE_FIX  : 完整 9 视频流 bridge_corr_switch >= 0.5
    2. HELD_OUT    : 留出切换对应率 holdout_switch >= 0.5（calib_switch 分开报告）
    3. STRUCT_STILL: SC2 > 0 且 churn <= 0.5（R0 与 R1 均要求）
    4. STABLE_STILL: MAE 末/首四分之一比 <= 1.5（R0 与 R1 均要求）
  判定：全过 = SOFT_PASS；3/4 过但 1/2 不过 = SOFT_PARTIAL（1 过 2 不过 = 半径过拟合
  校准集）；3 或 4 不过 = SOFT_FAIL。
  张力解除判定（§1.7）：SOFT_PASS 且 SC2 保持 >0（对齐 docs/243 一致性格）且 bridge >= 0.5
  且 holdout >= 0.5（拿到 docs/243/244 都没拿到的判别+泛化）且 churn <= 0.5。

回归守卫（不进判据）：旧粗档精确匹配（CompLoop + LOOP_CFG）重跑 R0/R1，应与 docs/243
一致（R0 SC2=1/churn 0；R1 bridge_sw=0.25/SC2=3/churn 0）。

安全纪律（docs/243/244 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_SMT_* 摘要
块；数字用纯 python 正则（vision/extract_r.py）抽取；禁止读日志/JSON 原文。禁止修改
stream_test.py / compose_test.py / critical_point.py / real_stream_test.py /
real_recalib.py 等既有脚本——只 import 复用。

用法：
  python vision/soft_match_test.py --tag timing
  python vision/soft_match_test.py --tag main
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import CPLoop, mean_sd, bootstrap_ci
from compose_test import CompLoop, ENERGY_BINS, UPPER_BINS, CTX_SPLIT_Y
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, RESIZE, WINDOW, LEVEL_NAMES
from real_recalib import CALIB_VIDEOS, HOLDOUT_VIDEOS, bridge_metrics

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 软匹配参数（预注册，docs/245 §1.2/§1.3，冻结）
K_CONSIST = 3          # 创建滞回：连续 k_consist 个参与窗口无原型在 r 内 -> 新原型
HITS_MIN = 3           # 稳定条目阈值（沿用 docs/241/243/244）
ALPHA = 0.2            # 原型自适应学习率（启用；只动中心）
R_FALLBACK = 0.5       # 校准回退（参与窗口 < 4 时；预注册，预期不触发）


class SoftLoop(CompLoop):
    """CompLoop 的软匹配变体：预测/事件路径原样继承（step 零改动），模式表替换为
    连续原型 + 距离匹配。bins 槽保留但不参与匹配。"""

    def __init__(self, radius, alpha=ALPHA, **kw):
        self.radius = float(radius)
        self.alpha = float(alpha)
        self.prototypes = []       # [{pid, mu(log), hits, created, last_active, n_match}]
        self.soft_trace = []       # per-window (lnE, lnU, matched_pid or -1)
        self._unmatched = 0        # 连续未匹配窗口计数（滞回）
        super().__init__(**kw)

    def _on_window(self):
        ev_win = self._ev_win if self._ev_win is not None else \
            np.zeros((120, 160), bool)
        mae_w = float(np.mean([f["mae"] for f in self._frame_buf]))
        att_w = float(np.mean([f["att"] for f in self._frame_buf]))
        ev_w = float(np.mean([f["ev"] for f in self._frame_buf]))
        theta_w = float(self._frame_buf[-1]["theta"])
        db_w = float(self._frame_buf[-1]["db"])
        self.mae.append(mae_w)
        self.att.append(att_w)
        self.ev.append(ev_w)
        self.theta_trace.append(theta_w)
        self.db_trace.append(db_w)

        E = int(ev_win.sum())
        U = int(ev_win[:int(CTX_SPLIT_Y), :].sum())
        self.energy_trace.append(E)
        self.up_trace.append(U)
        self.lo_trace.append(int(ev_win[int(CTX_SPLIT_Y):, :].sum()))
        self.c2_trace.append(None)
        self.sig_trace.append((None, None, None))
        if E >= 10:
            self.bbox_trace.append(float(U))
        else:
            self.bbox_trace.append(0.0)

        learned = False
        matched_pid = -1
        if E >= 10:
            x = (float(np.log1p(E)), float(np.log1p(U)))
            best, best_d = -1, None
            for i, p in enumerate(self.prototypes):
                d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best_d is not None and best_d <= self.radius:
                p = self.prototypes[best]
                p["hits"] += 1
                p["last_active"] = self._win
                p["n_match"] += 1
                if self.alpha > 0:
                    p["mu"] = ((1.0 - self.alpha) * p["mu"][0] + self.alpha * x[0],
                               (1.0 - self.alpha) * p["mu"][1] + self.alpha * x[1])
                self._unmatched = 0
                learned = True
                matched_pid = best
                self.match_trace.append((self._win, best))
            else:
                self._unmatched += 1
                if self._unmatched >= self.k_consist:
                    pid = len(self.prototypes)
                    self.prototypes.append(dict(pid=pid, mu=x, hits=1,
                                                created=self._win,
                                                last_active=self._win, n_match=1))
                    self._unmatched = 0
                    learned = True
                    matched_pid = pid
                    self.match_trace.append((self._win, pid))
                else:
                    self.match_trace.append((self._win, None))
        else:
            self.match_trace.append((self._win, None))
        self.soft_trace.append((round(np.log1p(E), 4), round(np.log1p(U), 4),
                                matched_pid))
        if learned:
            self._n_learn += 1
        self.sc1_cum.append(len(self.prototypes))
        self._win += 1
        self._frame_buf = []
        self._ev_win = None

    def finalize(self, n_windows, labels=None):
        if self._frame_buf:
            self._on_window()
        base = CPLoop.finalize(self, n_windows)
        sc1 = len(self.prototypes)
        sc2 = sum(1 for p in self.prototypes if p["hits"] >= self.hits_min)
        churn = sum(1 for p in self.prototypes
                    if p["hits"] < self.hits_min) / max(1, sc1)
        sc_late = sum(1 for p in self.prototypes
                      if p["created"] >= n_windows / 2
                      and p["hits"] >= self.hits_min)
        sc4 = sum(1 for p in self.prototypes
                  if p["last_active"] <= n_windows - 1 - self.persist_win)
        entry_log = [{"pid": p["pid"], "created": p["created"], "hits": p["hits"],
                      "last_active": p["last_active"], "n_match": p["n_match"],
                      "mu_log": [round(p["mu"][0], 4), round(p["mu"][1], 4)],
                      "key": [p["pid"]]} for p in self.prototypes]
        out = dict(base)
        out.update({
            "sc1": sc1, "sc2": sc2, "sc_late": sc_late, "sc4": sc4,
            "arity2_stable": sc2, "arity3_stable": 0,
            "compound_frac": 0.0, "churn_frac": round(churn, 4),
            "n_promo": 0, "ctx_fidelity": 1.0, "ctx_purity": 1.0,
            "ctx_n_windows": 0, "arity3_ctx0_stable": 0, "arity3_ctx1_stable": 0,
            "n_created_total": sc1, "n_retired": 0,
            "entry_log": entry_log,
            "c2_trace": [None] * len(self.soft_trace),
            "match_summary": {
                "matched": sum(1 for _, k in self.match_trace if k is not None),
                "none": sum(1 for _, k in self.match_trace if k is None),
            },
            "n_prototypes": sc1,
            "soft_trace": self.soft_trace,
        })
        return out


# ---------------- 半径校准（只依赖校准集统计；单次确定性 pass） ----------------
def calibrate_radius(calib_frames):
    """校准 pass：默认 LOOP_CFG 的 CompLoop 冷启动跑校准集拼接流（事件检测路径与测试
    pass 完全相同），收集每窗口 (E, U)。r = 参与窗口（E>=10）两两最近邻距离的中位数。"""
    loop = CompLoop(window=WINDOW, **LOOP_CFG)
    for g in calib_frames:
        loop.step(g)
    loop.finalize(max(1, len(calib_frames) // WINDOW), labels=None)
    E = np.asarray(loop.energy_trace, float)
    U = np.asarray(loop.up_trace, float)
    xs = [(float(np.log1p(E[i])), float(np.log1p(U[i])))
          for i in range(len(E)) if E[i] >= 10]
    n_w = int(len(E))
    n_v = len(xs)
    if n_v >= 4:
        ds = []
        for i in range(n_v):
            nn = min(float(np.hypot(xs[i][0] - xs[j][0], xs[i][1] - xs[j][1]))
                     for j in range(n_v) if j != i)
            ds.append(nn)
        r = float(np.median(ds))
        return r, {"n_windows": n_w, "n_valid": n_v,
                   "nn_median": round(r, 4),
                   "nn_min": round(float(np.min(ds)), 4),
                   "nn_max": round(float(np.max(ds)), 4)}
    return R_FALLBACK, {"n_windows": n_w, "n_valid": n_v,
                        "nn_median": R_FALLBACK, "nn_min": R_FALLBACK,
                        "nn_max": R_FALLBACK}


# ---------------- 运行一个级（软匹配；无持久化——本实验不重测 STATE_PERSIST） ----------------
def run_level(lvcode, frames, spans, radius, alpha):
    n_frames = len(frames)
    loop = SoftLoop(radius=radius, alpha=alpha, window=WINDOW, **LOOP_CFG)
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
        out["scene_protos"] = per_scene_protos(base, spans)
    return out, loop


# ---------------- 逐场景原型诊断（不进判据） ----------------
def per_scene_protos(base, spans):
    rows = []
    for i, (s0, s1) in enumerate(spans):
        in_span = [e for e in base["entry_log"]
                   if s0 <= e["created"] * WINDOW < s1]
        rows.append({"video": VIDEOS[i], "span": [s0, s1],
                     "n_protos": len(in_span),
                     "protos": [{"created": e["created"], "hits": e["hits"],
                                 "mu_log": e["mu_log"]} for e in in_span]})
    return rows


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smt")
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

    # ---- 半径校准（只依赖校准集；单次确定性 pass） ----
    radius, calib_stats = calibrate_radius(calib_frames)

    # ---- 主运行（软匹配）：R0 + R1 ----
    r0r, _ = run_level(30, r0, None, radius, ALPHA)
    r1r, r1_loop = run_level(31, r1, spans, radius, ALPHA)

    # ---- 回归守卫（旧粗档精确匹配，应与 docs/243 一致；不进判据） ----
    def guard_run(lvcode, frames, spans_g):
        loop = CompLoop(window=WINDOW, **LOOP_CFG)
        for g in frames:
            loop.step(g)
        base = loop.finalize(max(1, len(frames) // WINDOW), labels=None)
        if lvcode == 31:
            return base, bridge_metrics(base, spans_g)
        return base, None

    g0, _ = guard_run(30, r0, None)
    g1, g1b = guard_run(31, r1, spans)

    # ---- 判据（预注册冻结，docs/245 §1.5） ----
    bridge_ok = int(r1r["bridge"]["bridge_corr_switch"] >= 0.5)
    heldout_ok = int(r1r["bridge"]["holdout_switch"] >= 0.5)
    struct_ok = int(r0r["sc2"] > 0 and r0r["churn_frac"] <= 0.5
                    and r1r["sc2"] > 0 and r1r["churn_frac"] <= 0.5)
    stable_ok = int(r0r["ratio"] <= 1.5 and r1r["ratio"] <= 1.5)
    oks = {"bridge": bridge_ok, "heldout": heldout_ok,
           "struct": struct_ok, "stable": stable_ok}
    if not (struct_ok and stable_ok):
        verdict = "SOFT_FAIL"
        vnote = "soft matching broke structure/stability (STRUCT_STILL or STABLE_STILL failed; see numbers)"
    elif bridge_ok and heldout_ok:
        verdict = "SOFT_PASS"
        vnote = "BRIDGE_FIX and HELD_OUT and STRUCT_STILL and STABLE_STILL all pass"
    else:
        verdict = "SOFT_PARTIAL"
        if bridge_ok and not heldout_ok:
            vnote = "BRIDGE_FIX passes but HELD_OUT fails: radius overfits calibration set (calib switches encoded, holdout not)"
        elif not bridge_ok:
            vnote = "BRIDGE_FIX fails: soft matching did not fix bridge (see numbers)"
        else:
            vnote = "see numbers"

    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "videos": VIDEOS, "calib_videos": CALIB_VIDEOS,
           "holdout_videos": HOLDOUT_VIDEOS,
           "soft": {"k_consist": K_CONSIST, "hits_min": HITS_MIN,
                    "alpha": ALPHA, "radius": round(radius, 4),
                    "radius_calib": "median NN distance on calibration windows (log E/U space)"},
           "loop": LOOP_CFG,
           "bins_old": {"energy_bins": list(ENERGY_BINS),
                        "bbox_bins": list(UPPER_BINS)},
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "soft_match_test",
        "doc_ref": "docs/241, docs/243, docs/244, docs/245",
        "config": cfg,
        "calib": calib_stats,
        "levels": {"30": r0r, "31": r1r},
        "guard_old_bins": {"30": g0, "31": g1},
        "guard_old_bins_bridge31": g1b,
        "criteria": oks,
        "verdict": {"verdict": verdict, "note": vnote},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "smt_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    b31 = r1r["bridge"]
    print("R_SMT_RADIUS=%.4f" % radius)
    print("R_SMT_ALPHA=%.4f" % ALPHA)
    print("R_SMT_K_CONSIST=%d" % K_CONSIST)
    print("R_SMT_CALIB_WINDOWS=%d" % calib_stats["n_windows"])
    print("R_SMT_CALIB_VALID=%d" % calib_stats["n_valid"])
    print("R_SMT_NN_MED=%.4f" % calib_stats["nn_median"])
    print("R_SMT_NN_MIN=%.4f" % calib_stats["nn_min"])
    print("R_SMT_NN_MAX=%.4f" % calib_stats["nn_max"])
    for i, r in ((0, r0r), (1, r1r)):
        print("R_SMT_%d_NAME=%s" % (i, r["name"]))
        print("R_SMT_%d_FRAMES=%d" % (i, r["frames"]))
        print("R_SMT_%d_WINDOWS=%d" % (i, r["n_windows"]))
        print("R_SMT_%d_MAE=%.6f" % (i, r["mae_mean_win"]))
        print("R_SMT_%d_MAE_SD=%.6f" % (i, r["mae_sd_win"]))
        print("R_SMT_%d_MAE_LO=%.6f" % (i, r["mae_ci95"][0]))
        print("R_SMT_%d_MAE_HI=%.6f" % (i, r["mae_ci95"][1]))
        print("R_SMT_%d_Q1=%.6f" % (i, r["mae_q1"]))
        print("R_SMT_%d_Q4=%.6f" % (i, r["mae_q4"]))
        print("R_SMT_%d_RATIO=%.6f" % (i, r["ratio"]))
        print("R_SMT_%d_SC1=%d" % (i, r["sc1"]))
        print("R_SMT_%d_SC2=%d" % (i, r["sc2"]))
        print("R_SMT_%d_A2=%d" % (i, r["arity2_stable"]))
        print("R_SMT_%d_A3=%d" % (i, r["arity3_stable"]))
        print("R_SMT_%d_COMP=%.4f" % (i, r["compound_frac"]))
        print("R_SMT_%d_CHURN=%.4f" % (i, r["churn_frac"]))
        print("R_SMT_%d_PROMO=%d" % (i, r["n_promo"]))
        print("R_SMT_%d_SCLATE=%d" % (i, r["sc_late"]))
        print("R_SMT_%d_SC4=%d" % (i, r["sc4"]))
        print("R_SMT_%d_PIN=%.4f" % (i, r["pin_frac"]))
        print("R_SMT_%d_THETA=%.6f" % (i, r["theta_mean"]))
        print("R_SMT_%d_ENTRIES=%s" % (i, ";".join(
            "%d|%d|%d|%.3f,%.3f" % (e["created"], e["hits"], e["n_match"],
                                    e["mu_log"][0], e["mu_log"][1])
            for e in r["entry_log"])))
    print("R_SMT_SWITCHES=%d" % b31["switches"])
    print("R_SMT_BRIDGE_SW=%.4f" % b31["bridge_corr_switch"])
    print("R_SMT_BRIDGE_VID=%.4f" % b31["bridge_corr_video"])
    print("R_SMT_CALIB_SWITCHES=%d" % b31["calib_switches"])
    print("R_SMT_CALIB_ENC=%d" % b31["encoded_calib_sw"])
    print("R_SMT_CALIB_SW=%.4f" % b31["calib_switch"])
    print("R_SMT_HOLDOUT_SWITCHES=%d" % b31["holdout_switches"])
    print("R_SMT_HOLDOUT_ENC=%d" % b31["encoded_holdout_sw"])
    print("R_SMT_HOLDOUT_SW=%.4f" % b31["holdout_switch"])
    print("R_SMT_SPURIOUS=%d" % b31["spurious"])
    print("R_SMT_SPUR_RATE=%.4f" % b31["spurious_rate"])
    print("R_SMT_SW_ENC=%s" % ",".join(str(v) for v in b31["sw_enc_flags"]))
    print("R_SMT_VID_ENC=%s" % ",".join(str(v) for v in b31["vid_enc_flags"]))
    print("R_SMT_SCENE_PROTOS=%s" % ";".join(
        "%s:%d" % (row["video"], row["n_protos"]) for row in r1r["scene_protos"]))
    print("R_SMT_GUARD_R0_SC2=%d" % g0["sc2"])
    print("R_SMT_GUARD_R0_CHURN=%.4f" % g0["churn_frac"])
    print("R_SMT_GUARD_R1_SC2=%d" % g1["sc2"])
    print("R_SMT_GUARD_R1_CHURN=%.4f" % g1["churn_frac"])
    print("R_SMT_GUARD_R1_BRIDGE_SW=%.4f" % g1b["bridge_corr_switch"])
    print("R_SMT_BRIDGE_OK=%d" % bridge_ok)
    print("R_SMT_HELDOUT_OK=%d" % heldout_ok)
    print("R_SMT_STRUCT_OK=%d" % struct_ok)
    print("R_SMT_STABLE_OK=%d" % stable_ok)
    print("R_SMT_VERDICT=%s" % verdict)
    print("R_SMT_VERDICT_NOTE=%s" % vnote)
    print("R_SMT_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
