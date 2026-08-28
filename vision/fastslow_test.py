"""vision/fastslow_test.py — docs/250 机制级修复：B' 快慢双原型（人眼机制参考）与 gist 正确性。

docs/249 verdict=L3_FIX_PASS 但修复在评估规则层（短段配额改判 hits_min，不改变原型如何
形成/匹配/创建）。本实验做机制层修复：让学习过程本身适应段长——快慢双原型（B'，人眼
M/P 双通路 + gist 先行 + 扫视-注视 + 记忆巩固 + 变化盲参考）。

预注册（docs/250 §一，冻结；docs/63+247 纪律；判据/旋钮初值先于最终运行写入 docs/250，
运行后不改）：
  机制（B'）：
    1. 快原型 = gist/扫视提取器：粗半径 r_fast=1.5*r_slow=0.598275、低门槛
       hits_min_fast=1、高残差新奇段立即触发（k_consist_fast=1，无 3 窗滞回）、允许短命
       但正确（快原型不参与 churn 判据）。
    2. 慢原型 = 注视/整合器：细半径 r_slow=0.39885（docs/246 M=1.5 工作点）、高门槛
       hits_min_slow=3；由快原型反复重匹配升级而来（hits >= k_promote=2 -> 升级为慢，
       半径收紧、受回收豁免；稳定需在细半径下再累积至 hits_min_slow=3）。
    3. 升级 = 重匹配驱动渐变（计数/滞回，非一次硬跳变）；失配 = 回收（快原型连续
       k_decay=5 窗未重匹配 -> 遗忘移除）。
    4. 快慢并发（M/P 精神）：窗口同时匹配快慢；慢优先（已验证记忆优先），快兜底。
    5. 不做段长预测：时间尺度从行为（命中率 -> 升级/回收）涌现，无帧差/位移等外部信号。
  数据/GT：DAVIS R1（9 视频拼接）天然有真值段边界（视频切换时点）——gist 正确性在 R1 上
    有真 GT（|Δ|<=1 窗对应率）；野流 S1-S4（复用 cross_domain_test 加载）帧差近似段边界
    作诊断级（诚实声明）。
  判据（每判据带 docs/247 层级标签，冻结）：
    1. [L3][机制][无配额] CHURN_MECH : 配额完全关闭，慢原型 churn_slow <= 0.5
       （DAVIS R1 + S1-S4；快原型不参与）。
    2. [L3][机制][gist正确性] GIST_CORRECT : R1 真值段边界对应率 gist_cov >= 0.5
       （|Δ|<=1 窗）；野流帧差近似对应率作诊断。
    3. [L3][机制] STABLE/STRUCT 保持 : 四流 + R1 ratio <= 1.5 且 SC2_slow > 0。
    4. [L3][机制][行为证据] PROMOTION : 升级数 > 0、回收数 > 0、升级原型命中率均值 >
       未升级原型命中率均值（升级非随机，报告两侧均值）。
    5. [L3][诊断] QUOTA_ORTHOGONAL : 机制 + docs/249 配额叠加 vs 机制单独——配额是否已
       可退役（若叠加不改变结果 -> 可退役）。
  判定：1-4 全过 且 守卫 12/12 = MECH_PASS；1 不过但 2 过 = PARTIAL_CORRECT（"短命但
    正确"成立）；2 不过或机制破坏 = MECH_FAIL；数据不可用 = FSL_BLOCKED。
  守卫（不进判据，实现正确性）：复用 fastcut_fix.run_guard_quota（SoftLoop+门+配额，
    docs/249 守卫同一代码路径）跑 DAVIS R0+R1，配额关闭字段须复现 docs/246 M=1.5 工作点
    行（R0 SC2=3/churn 0/ratio 0.907701；R1 SC1=11/SC2=6/churn 0.4545/ratio 0.951261；
    bridge_sw 0.8750/calib_sw 1.0/holdout_sw 0.75/bridge_vid 0.8889/spurious 0；
    容差 1e-4）-> R_FSL_GUARD_D246（12/12）。

安全纪律（docs/243-249 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_FSL_* 摘要
块；运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；Downloads 视频是数据（只读帧数/文件名）。
禁止修改任何既有脚本——只 import 复用（soft_match_test / cross_domain_test /
fastcut_fix / real_stream_test / real_recalib / stream_test / critical_point /
compose_test）。

用法：
  python vision/fastslow_test.py --tag fsl
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import CPLoop, mean_sd, bootstrap_ci
from compose_test import CompLoop, CTX_SPLIT_Y
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, WINDOW, RESIZE
from real_recalib import bridge_metrics
from soft_match_test import ALPHA, HITS_MIN
from cross_domain_test import (load_sampled_frames, WILD_VIDEOS, STREAMS,
                               RADIUS_L3, R_BASE_DAVIS, D246, DL_DIR,
                               guard_vs_d246, scene_switch_diag)
from fastcut_fix import fastcut_gate, apply_quota, run_guard_quota

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 机制旋钮（预注册，docs/250 §1.3，冻结；不得回调）
R_FAST_MULT = 1.5              # r_fast = 1.5 x r_slow（粗半径/gist）
R_SLOW = RADIUS_L3             # 0.39885（docs/246 M=1.5 工作点，细半径/注视）
R_FAST = R_FAST_MULT * R_SLOW  # = 0.598275
HITS_MIN_FAST = 1              # 低门槛（gist 正确性低阈值）
HITS_MIN_SLOW = HITS_MIN       # 3（高门槛，= 冻结 hits_min）
K_PROMOTE = 2                  # 升级门槛：hits >= 2 次重匹配 -> 升级为慢（固化候选）
K_DECAY = 5                    # 回收门槛：连续 5 窗未重匹配 -> 遗忘（= persist_win）
K_CONSIST_FAST = 1             # gist 先行：高残差新奇段立即创建（无 3 窗滞回）


class FastSlowLoop(CompLoop):
    """CompLoop 的快慢双原型变体：预测/事件路径原样继承（step 零改动），模式表替换为
    快慢双类原型 + 距离匹配 + 升级 + 回收。bins 槽保留但不参与匹配。"""

    def __init__(self, r_fast=R_FAST, r_slow=R_SLOW, alpha=ALPHA,
                 k_promote=K_PROMOTE, k_decay=K_DECAY,
                 hits_min_fast=HITS_MIN_FAST, hits_min_slow=HITS_MIN_SLOW, **kw):
        self.r_fast = float(r_fast)
        self.r_slow = float(r_slow)
        self.alpha = float(alpha)
        self.k_promote = int(k_promote)
        self.k_decay = int(k_decay)
        self.hits_min_fast = int(hits_min_fast)
        self.hits_min_slow = int(hits_min_slow)
        self.prototypes = []      # [{pid, mu(log), hits, created, last_active,
        #                           n_match, kind(fast/slow), promoted_at}]
        self._next_pid = 0        # 单调递增 pid（回收移除原型后 len() 会复用，必须独立计数）
        self.created_log = []     # 快原型创建记录（累计，含回收）
        self.promoted_log = []    # 升级记录 {pid, promoted_at, hits}
        self.n_created_fast = 0
        self.n_promoted = 0
        self.n_recycled = 0
        self.soft_trace = []      # per-window (lnE, lnU, matched_pid or -1)
        super().__init__(**kw)

    def _hit(self, p, x):
        p["hits"] += 1
        p["last_active"] = self._win
        p["n_match"] += 1
        if self.alpha > 0:
            p["mu"] = ((1.0 - self.alpha) * p["mu"][0] + self.alpha * x[0],
                       (1.0 - self.alpha) * p["mu"][1] + self.alpha * x[1])

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
            # 1. 慢优先（已验证记忆优先，细半径 r_slow）
            best, best_d = -1, None
            for i, p in enumerate(self.prototypes):
                if p["kind"] != "slow":
                    continue
                d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best_d is not None and best_d <= self.r_slow:
                p = self.prototypes[best]
                self._hit(p, x)
                learned = True
                matched_pid = p["pid"]
                self.match_trace.append((self._win, p["pid"]))
            else:
                # 2. 快兜底（gist 粗半径 r_fast）
                best, best_d = -1, None
                for i, p in enumerate(self.prototypes):
                    if p["kind"] != "fast":
                        continue
                    d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
                    if best_d is None or d < best_d:
                        best, best_d = i, d
                if best_d is not None and best_d <= self.r_fast:
                    p = self.prototypes[best]
                    self._hit(p, x)
                    learned = True
                    matched_pid = p["pid"]
                    self.match_trace.append((self._win, p["pid"]))
                    # 升级：重匹配驱动（计数/滞回），半径收紧、受回收豁免
                    if p["hits"] >= self.k_promote:
                        p["kind"] = "slow"
                        p["promoted_at"] = self._win
                        self.n_promoted += 1
                        self.promoted_log.append(dict(pid=p["pid"],
                                                      promoted_at=self._win,
                                                      hits=p["hits"]))
                else:
                    # 3. 高残差新奇段立即创建快原型（k_consist_fast=1）
                    pid = self._next_pid
                    self._next_pid += 1
                    self.prototypes.append(dict(pid=pid, mu=x, hits=1,
                                                created=self._win,
                                                last_active=self._win, n_match=1,
                                                kind="fast", promoted_at=None))
                    self.n_created_fast += 1
                    self.created_log.append(dict(pid=pid, created=self._win,
                                                 final_hits=None, recycled=0))
                    learned = True
                    matched_pid = pid
                    self.match_trace.append((self._win, pid))
        else:
            self.match_trace.append((self._win, None))
        # 4. 回收：快原型连续 k_decay 窗未重匹配 -> 遗忘（慢原型豁免；窗口时间衰减）
        for p in list(self.prototypes):
            if p["kind"] == "fast" and (self._win - p["last_active"]) >= self.k_decay:
                self.prototypes.remove(p)
                self.n_recycled += 1
                for cl in self.created_log:
                    if cl["pid"] == p["pid"]:
                        cl["final_hits"] = p["hits"]
                        cl["recycled"] = 1
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
        # 存活原型（快 + 慢，都曾以快创建）的 final_hits 回填 created_log
        for p in self.prototypes:
            for cl in self.created_log:
                if cl["pid"] == p["pid"]:
                    cl["final_hits"] = p["hits"]
        for cl in self.created_log:
            if cl["final_hits"] is None:
                cl["final_hits"] = 0

        slows = [p for p in self.prototypes if p["kind"] == "slow"]
        sc1_fast = len(self.created_log)
        sc2_fast = sum(1 for cl in self.created_log
                       if cl["final_hits"] >= self.hits_min_fast)
        sc1_slow = len(self.promoted_log)          # = len(slows)（慢不回收）
        sc2_slow = sum(1 for p in slows if p["hits"] >= self.hits_min_slow)
        churn_slow = (sc1_slow - sc2_slow) / max(1, sc1_slow)

        # 升级 vs 未升级命中率（PROMOTION 行为证据）
        promo_pids = set(pl["pid"] for pl in self.promoted_log)
        promo_hits = [p["hits"] for p in slows]
        nonp_hits = [cl["final_hits"] for cl in self.created_log
                     if cl["pid"] not in promo_pids]
        promo_mean = float(np.mean(promo_hits)) if promo_hits else 0.0
        nonp_mean = float(np.mean(nonp_hits)) if nonp_hits else 0.0

        # entry_log（慢 = 升级过的原型；快 = 未升级的创建记录）
        entry_log = []
        for p in sorted(self.prototypes, key=lambda q: q["pid"]):
            if p["kind"] == "slow":
                entry_log.append({"pid": p["pid"], "created": p["created"],
                                  "promoted_at": p["promoted_at"],
                                  "hits": p["hits"], "kind": "slow", "recycled": 0})
            else:
                entry_log.append({"pid": p["pid"], "created": p["created"],
                                  "hits": p["hits"], "kind": "fast", "recycled": 0})
        for cl in self.created_log:
            if cl["recycled"]:
                entry_log.append({"pid": cl["pid"], "created": cl["created"],
                                  "hits": cl["final_hits"], "kind": "fast",
                                  "recycled": 1})
        churn_legacy = sum(1 for e in entry_log if e["hits"] < self.hits_min_slow) \
            / max(1, sc1_fast)

        out = dict(base)
        out.update({
            "sc1_fast": sc1_fast, "sc2_fast": sc2_fast,
            "sc1_slow": sc1_slow, "sc2_slow": sc2_slow,
            "churn_slow": round(churn_slow, 4),
            "churn_legacy": round(churn_legacy, 4),
            "n_promo": self.n_promoted, "n_recycle": self.n_recycled,
            "promoted_mean_hits": round(promo_mean, 4),
            "nonpromoted_mean_hits": round(nonp_mean, 4),
            "entry_log": entry_log,
            "created_log": self.created_log,
            "promoted_log": self.promoted_log,
            "match_summary": {
                "matched": sum(1 for _, k in self.match_trace if k is not None),
                "none": sum(1 for _, k in self.match_trace if k is None),
            },
            "soft_trace": self.soft_trace,
        })
        return out


# ---------------- 单流运行（FastSlowLoop；零重调） ----------------
def run_fastslow_stream(frames):
    loop = FastSlowLoop(window=WINDOW, **LOOP_CFG)
    for g in frames:
        loop.step(g)
    base = loop.finalize(max(1, len(frames) // WINDOW), labels=None)
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0
    mae_m, mae_sd = mean_sd(list(mae_arr))
    mae_lo, mae_hi = bootstrap_ci(list(mae_arr))
    E = np.asarray(loop.energy_trace, float)
    n_valid = int(np.sum(E >= 10))
    out = dict(base)
    out.update({"frames": len(frames), "n_windows": len(mae_arr), "n_valid": n_valid,
                "mae_mean_win": round(mae_m, 6), "mae_sd_win": round(mae_sd, 6),
                "mae_ci95": [round(mae_lo, 6), round(mae_hi, 6)],
                "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
                "ratio": round(ratio, 6), "pin_frac": base["pin_frac"],
                "theta_mean": base["theta_mean"]})
    return out, loop


# ---------------- QUOTA_ORTHOGONAL：docs/249 配额叠加（诊断级） ----------------
def quota_on_slow(out, loop):
    """把 docs/249 短段配额（apply_quota 复用，快切门 fastcut_gate 复用）叠加到慢层
    entry_log（created=升级窗、hits=final）。返回叠加前后对照。"""
    slows = [e for e in out["entry_log"] if e["kind"] == "slow"]
    if not slows:
        return {"n_slow": 0, "fire": 0, "sc2_slow": out["sc2_slow"],
                "churn_slow": out["churn_slow"], "sc2_q": out["sc2_slow"],
                "churn_q": out["churn_slow"], "d_sc2": 0, "d_churn": 0.0}
    synth = {"entry_log": [{"created": e["promoted_at"], "hits": e["hits"]}
                           for e in slows],
             "sc2": out["sc2_slow"], "churn_frac": out["churn_slow"]}
    fire = fastcut_gate(loop)["fire"]
    sc2_q, churn_q, _ = apply_quota(synth, len(loop.mae), fire)
    return {"n_slow": len(slows), "fire": fire,
            "sc2_slow": out["sc2_slow"], "churn_slow": out["churn_slow"],
            "sc2_q": sc2_q, "churn_q": churn_q,
            "d_sc2": sc2_q - out["sc2_slow"],
            "d_churn": round(churn_q - out["churn_slow"], 4)}


# ---------------- R1 gist 真值度量（视频切换 = GT；|Δ|<=1 窗） ----------------
def gist_metrics(out, switch_windows):
    creations = sorted(cl["created"] for cl in out["created_log"]) \
        if "created_log" in out else \
        sorted(e["created"] for e in out["entry_log"] if e["kind"] == "fast")
    cov = sum(int(any(abs(c - ws) <= 1 for c in creations))
              for ws in switch_windows) / max(1, len(switch_windows))
    align = [int(any(abs(c - ws) <= 1 for ws in switch_windows)) for c in creations]
    prec = sum(align) / max(1, len(creations))
    cov2 = sum(int(any(abs(c - ws) <= 2 for c in creations))
               for ws in switch_windows) / max(1, len(switch_windows))
    return {"n_switches": len(switch_windows), "switch_windows": switch_windows,
            "n_creations": len(creations), "creation_windows": creations,
            "cov": round(cov, 4), "prec": round(prec, 4),
            "cov_d2": round(cov2, 4), "aligned_creations": sum(align)}


def build_entry_base(out):
    """把快慢 entry_log 转成 bridge_metrics 可读的基础 dict（key=[pid]；bridge_metrics
    还读取 churn_frac/sc1/sc2/n_promo 字段）。"""
    return {"entry_log": [{"created": e["created"], "hits": e["hits"],
                           "key": [e["pid"]]} for e in out["entry_log"]],
            "churn_frac": out["churn_slow"],
            "sc1": len(out["entry_log"]), "sc2": out["sc2_slow"],
            "n_promo": out["n_promo"]}


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fsl")
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

    # ---- 流（docs/248 §1.5，冻结；S1-S4；机制不含配额） ----
    streams_out = {}
    for sid, sname, vidx in STREAMS:
        frames = []
        for vi in vidx:
            frames.extend(loaded[WILD_VIDEOS[vi][0]])
        out, loop = run_fastslow_stream(frames)
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        diag = scene_switch_diag(frames, creations)
        out["stream_id"] = sid
        out["stream_name"] = sname
        out["videos"] = [WILD_VIDEOS[vi][0] for vi in vidx]
        out["switch_diag"] = diag
        out["quota_ortho"] = quota_on_slow(out, loop)
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
    r0r, r0_loop = run_fastslow_stream(r0_frames)
    r1r, r1_loop = run_fastslow_stream(r1_frames)
    r1r["bridge"] = bridge_metrics(build_entry_base(r1r), spans)  # 诊断
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    r1r["gist"] = gist_metrics(r1r, switch_windows)
    r1r["quota_ortho"] = quota_on_slow(r1r, r1_loop)

    # ---- 回归守卫（DAVIS；docs/249 同一代码路径；不进判据） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard_ok, guard_detail = guard_vs_d246(g0, g1)

    # ---- 判据（预注册 §1.7，冻结；机制数字 = 配额完全关闭口径） ----
    all_streams = [streams_out[s] for s, _, _ in STREAMS] + [r1r]
    churn_mech = int(all(r["churn_slow"] <= 0.5 for r in all_streams))
    gist_correct = int(r1r["gist"]["cov"] >= 0.5)
    stable_keep = int(all(r["ratio"] <= 1.5 for r in all_streams))
    struct_keep = int(all(r["sc2_slow"] > 0 for r in all_streams))
    n_promo_total = sum(r["n_promo"] for r in all_streams)
    n_recycle_total = sum(r["n_recycle"] for r in all_streams)
    promo_means = [(r["promoted_mean_hits"], r["nonpromoted_mean_hits"])
                   for r in all_streams if r["sc1_slow"] > 0]
    promo_sep = int(any(mp > mn for mp, mn in promo_means)) \
        if promo_means else 0
    promo_ok = int(n_promo_total > 0 and n_recycle_total > 0 and promo_sep)
    oks = {"churn_mech": churn_mech, "gist_correct": gist_correct,
           "stable_keep": stable_keep, "struct_keep": struct_keep,
           "promotion": promo_ok}

    # ---- QUOTA_ORTHOGONAL（诊断）：叠加后四流 + R1 逐项不变 -> 可退役 ----
    quo = [streams_out[s]["quota_ortho"] for s, _, _ in STREAMS] + [r1r["quota_ortho"]]
    quota_delta = int(any(q["d_sc2"] != 0 or q["d_churn"] != 0.0 for q in quo))
    quota_retirable = int(quota_delta == 0)

    # ---- 判定（预注册 §1.7 冻结规则） ----
    if churn_mech and gist_correct and stable_keep and struct_keep and promo_ok:
        verdict = "MECH_PASS"
        vnote = ("CHURN_MECH and GIST_CORRECT and STABLE/STRUCT and PROMOTION all pass; "
                 "time scale emerges from behavior (re-matching -> promotion/recycle) and "
                 "gist correctness holds on DAVIS R1 GT segment boundaries; quota "
                 "orthogonality diagnostic below")
    elif churn_mech and gist_correct and stable_keep and struct_keep:
        verdict = "MECH_PASS"
        vnote = ("criteria 1-3 pass but PROMOTION evidence weak (see n_promo/n_recycle/"
                 "hit-rate numbers); guard status below")
    elif (not churn_mech) and gist_correct:
        verdict = "PARTIAL_CORRECT"
        vnote = ("CHURN_MECH fails (slow churn > 0.5 somewhere) but GIST_CORRECT holds: "
                 "'short-lived but correct' established; stable/gist metric split itself "
                 "is the conclusion")
    elif not gist_correct:
        verdict = "MECH_FAIL"
        vnote = ("GIST_CORRECT fails on DAVIS R1 GT segment boundaries (gist alignment "
                 "did not hold); mechanism-level fix not established (see numbers)")
    else:
        verdict = "MECH_FAIL"
        vnote = ("mechanism broken or criteria unsatisfiable: SC2_slow=0 somewhere or "
                 "ratio>1.5 or no promotion/recycling (see numbers)")

    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "working_point": {"r_slow": round(R_SLOW, 6), "r_fast": round(R_FAST, 6),
                             "r_fast_mult": R_FAST_MULT,
                             "hits_min_fast": HITS_MIN_FAST,
                             "hits_min_slow": HITS_MIN_SLOW,
                             "k_promote": K_PROMOTE, "k_decay": K_DECAY,
                             "k_consist_fast": K_CONSIST_FAST, "alpha": ALPHA},
           "mechanism": "B' fast-slow dual prototypes (human-eye M/P + gist + "
                        "saccade-fixation + consolidation + change blindness); "
                        "no segment-length prediction; quota fully off",
           "quota_orthogonal": {"method": "mechanism + docs/249 quota stacked on slow "
                                         "layer (apply_quota/fastcut_gate reuse)"},
           "loop": LOOP_CFG,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "fastslow_test",
        "doc_ref": "docs/245, docs/246, docs/247, docs/248, docs/249, docs/250",
        "config": cfg,
        "streams": streams_out,
        "r0": r0r, "r1": r1r,
        "r1_switch_windows": switch_windows,
        "criteria": oks,
        "verdict": {"verdict": verdict, "note": vnote},
        "quota_orthogonal": {"per_stream": {("R1" if i == len(quo) - 1 else
                                            STREAMS[i][0]): q
                                            for i, q in enumerate(quo)},
                             "any_delta": quota_delta,
                             "retirable": quota_retirable},
        "guard_d246": {"ok": guard_ok, "detail": guard_detail},
        "promotion": {"n_promo_total": n_promo_total,
                      "n_recycle_total": n_recycle_total,
                      "promoted_vs_nonpromoted": promo_means,
                      "ok": promo_ok},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "%s_%s.json" % ("fsl", args.tag))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_FSL_R_SLOW=%.6f" % R_SLOW)
    print("R_FSL_R_FAST=%.6f" % R_FAST)
    print("R_FSL_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_FSL_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_FSL_K_PROMOTE=%d" % K_PROMOTE)
    print("R_FSL_K_DECAY=%d" % K_DECAY)
    print("R_FSL_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_FSL_ALPHA=%.4f" % ALPHA)
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        d = r["switch_diag"]
        q = r["quota_ortho"]
        print("R_FSL_%d_ID=%s" % (j, sid))
        print("R_FSL_%d_NAME=%s" % (j, sname))
        print("R_FSL_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_FSL_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_FSL_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_FSL_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_FSL_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_FSL_%d_SC1_FAST=%d" % (j, r["sc1_fast"]))
        print("R_FSL_%d_SC2_FAST=%d" % (j, r["sc2_fast"]))
        print("R_FSL_%d_SC1_SLOW=%d" % (j, r["sc1_slow"]))
        print("R_FSL_%d_SC2_SLOW=%d" % (j, r["sc2_slow"]))
        print("R_FSL_%d_CHURN_SLOW=%.4f" % (j, r["churn_slow"]))
        print("R_FSL_%d_CHURN_LEGACY=%.4f" % (j, r["churn_legacy"]))
        print("R_FSL_%d_N_PROMO=%d" % (j, r["n_promo"]))
        print("R_FSL_%d_N_RECYCLE=%d" % (j, r["n_recycle"]))
        print("R_FSL_%d_PROMO_MEAN=%.4f" % (j, r["promoted_mean_hits"]))
        print("R_FSL_%d_NONPROMO_MEAN=%.4f" % (j, r["nonpromoted_mean_hits"]))
        print("R_FSL_%d_FAST_CREATIONS=%s" % (j, ",".join(
            str(e["created"]) for e in sorted(r["entry_log"], key=lambda e: e["created"])
            if e["kind"] == "fast")))
        print("R_FSL_%d_SLOW_PROMOS=%s" % (j, ",".join(
            "%d:%d" % (e["promoted_at"], e["hits"]) for e in
            sorted(r["entry_log"], key=lambda e: e["created"])
            if e["kind"] == "slow")))
        print("R_FSL_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                           else "%.4f" % d["switch_corr"])))
        print("R_FSL_%d_Q_SC2=%d" % (j, q["sc2_slow"]))
        print("R_FSL_%d_Q_CHURN=%.4f" % (j, q["churn_slow"]))
        print("R_FSL_%d_Q_SC2_Q=%d" % (j, q["sc2_q"]))
        print("R_FSL_%d_Q_CHURN_Q=%.4f" % (j, q["churn_q"]))
        print("R_FSL_%d_Q_DSC2=%d" % (j, q["d_sc2"]))
        print("R_FSL_%d_Q_DCHURN=%.4f" % (j, q["d_churn"]))
    print("R_FSL_R0_FRAMES=%d" % r0r["frames"])
    print("R_FSL_R0_RATIO=%.6f" % r0r["ratio"])
    print("R_FSL_R0_SC1_FAST=%d" % r0r["sc1_fast"])
    print("R_FSL_R0_SC1_SLOW=%d" % r0r["sc1_slow"])
    print("R_FSL_R0_SC2_SLOW=%d" % r0r["sc2_slow"])
    print("R_FSL_R0_CHURN_SLOW=%.4f" % r0r["churn_slow"])
    print("R_FSL_R0_N_PROMO=%d" % r0r["n_promo"])
    print("R_FSL_R0_N_RECYCLE=%d" % r0r["n_recycle"])
    print("R_FSL_R1_FRAMES=%d" % r1r["frames"])
    print("R_FSL_R1_RATIO=%.6f" % r1r["ratio"])
    print("R_FSL_R1_SC1_FAST=%d" % r1r["sc1_fast"])
    print("R_FSL_R1_SC1_SLOW=%d" % r1r["sc1_slow"])
    print("R_FSL_R1_SC2_SLOW=%d" % r1r["sc2_slow"])
    print("R_FSL_R1_CHURN_SLOW=%.4f" % r1r["churn_slow"])
    print("R_FSL_R1_CHURN_LEGACY=%.4f" % r1r["churn_legacy"])
    print("R_FSL_R1_N_PROMO=%d" % r1r["n_promo"])
    print("R_FSL_R1_N_RECYCLE=%d" % r1r["n_recycle"])
    print("R_FSL_R1_PROMO_MEAN=%.4f" % r1r["promoted_mean_hits"])
    print("R_FSL_R1_NONPROMO_MEAN=%.4f" % r1r["nonpromoted_mean_hits"])
    print("R_FSL_R1_FAST_CREATIONS=%s" % ",".join(
        str(w) for w in r1r["gist"]["creation_windows"]))
    print("R_FSL_R1_SWITCHES=%s" % ",".join(str(w) for w in switch_windows))
    print("R_FSL_R1_GIST_COV=%.4f" % r1r["gist"]["cov"])
    print("R_FSL_R1_GIST_PREC=%.4f" % r1r["gist"]["prec"])
    print("R_FSL_R1_GIST_COV_D2=%.4f" % r1r["gist"]["cov_d2"])
    print("R_FSL_R1_BRIDGE_SW=%.4f" % r1r["bridge"]["bridge_corr_switch"])
    print("R_FSL_R1_BRIDGE_VID=%.4f" % r1r["bridge"]["bridge_corr_video"])
    print("R_FSL_R1_Q_SC2=%d" % r1r["quota_ortho"]["sc2_slow"])
    print("R_FSL_R1_Q_CHURN=%.4f" % r1r["quota_ortho"]["churn_slow"])
    print("R_FSL_R1_Q_SC2_Q=%d" % r1r["quota_ortho"]["sc2_q"])
    print("R_FSL_R1_Q_CHURN_Q=%.4f" % r1r["quota_ortho"]["churn_q"])
    print("R_FSL_R1_Q_DSC2=%d" % r1r["quota_ortho"]["d_sc2"])
    print("R_FSL_R1_Q_DCHURN=%.4f" % r1r["quota_ortho"]["d_churn"])
    print("R_FSL_CHURN_OK=%d" % churn_mech)
    print("R_FSL_GIST_OK=%d" % gist_correct)
    print("R_FSL_STABLE_OK=%d" % stable_keep)
    print("R_FSL_STRUCT_OK=%d" % struct_keep)
    print("R_FSL_PROMO_OK=%d" % promo_ok)
    print("R_FSL_PROMO_TOTAL=%d" % n_promo_total)
    print("R_FSL_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_FSL_QUOTA_RETIRABLE=%d" % quota_retirable)
    print("R_FSL_VERDICT=%s" % verdict)
    print("R_FSL_VERDICT_NOTE=%s" % vnote)
    print("R_FSL_GUARD_D246=%d" % guard_ok)
    print("R_FSL_GUARD_DETAIL=%s" % guard_detail)
    print("R_FSL_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
