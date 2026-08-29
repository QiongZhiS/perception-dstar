"""vision/l4_compose_test.py — docs/253 L4 组合泛化第一格：上下文槽位机制接真实流。

SlotLoop(DeferredLoop)（docs/253 §一 冻结，运行后不改）：
  Mode OFF = DeferredLoop 逐字（槽位路径 (ii)-(iv) 全关 -> R_L4_GUARD_D251 复现
             docs/251 §3.3/§3.4，32 项，容差 1e-4）；
  Mode ON  = 槽位路径开启：
             (i)   c2 槽位可观测（同一事件掩码，纯 numpy，逐字移植
                   compose_test.CompLoop._window_components 的 c2 定义）；
             (ii)  原型 c2 账本（每次匹配追加 (c2,E)；创建窗即首个条目，无回填）；
             (iii) 已确认慢原型（hits >= hits_min_slow=3）信息量驱动分裂
                   （hits >= k_split=5 且账本 >=2 个非 None c2 值各 >= k_ledger=3
                   条目 且两值中位事件能量比 >= 1+delta_rel=1.30）；
             (iv)  打标慢原型按 c2 门控匹配（只匹配 c2 == tag 的窗口）。
  预测路径零改动 -> R_L4_REPRO_RATIO（Mode ON 全流 ratio 与 Mode OFF 逐位一致，
  abs < 1e-9）。

度量（§1.4 冻结）：MAE/ratio；SC1_fast/SC2_fast/SC1_slow/SC2_slow/SC2_tagged/
compound_frac/n_split/n_retired_slow；churn_slow=max(0,(SC1_slow-SC2_slow)/max(1,SC1_slow))
（分裂使 SC2_slow 可超 SC1_slow -> 按构造 0.0）；spurious_split_frac、平均分裂后命中；
M4 诊断：打标分布、R1 段级信息量、分裂-段对齐、槽位覆盖。

判据（§1.7 冻结）：①[L4][机制][组合测试] COMPOUND_EMERGES（R1：n_split>=1 且
SC2_tagged>=1 且 compound_frac>=0.5）；②[L4][机制][行为证据] ADOPT_NONRANDOM
（R1：spurious_split_frac<=0.5 且平均分裂后命中>=1；R0：n_split==0）；
③[L4][机制] FOUNDATION_KEEP（R1+S1-S4：ratio<=1.5 且 SC2_slow>0；R1：gist_cov>=0.5）；
④[L4][机制][行为证据] PROMOTION_KEEP（全局 n_promo>0 且 n_recycle>0 且升级命中率
均值>未升级均值）。判定映射按 §1.7 表格（COMPOSABLE_REAL / PARALLEL_ONLY_REAL /
BOUNDARY / PARTIAL_REAL / GUARD_FAIL / L4_BLOCKED）。

守卫（§1.8 冻结，不进判据）：R_L4_GUARD_D251（Mode OFF 复现 docs/251，32 项）、
R_L4_GUARD_D246（run_guard_quota + guard_vs_d246，12/12）、R_L4_REPRO_RATIO
（ON vs OFF 全流 ratio abs<1e-9）、确定性复现（timing 轮与 main 轮 R_L4_* 逐位一致；
c2_trace Mode ON 两轮逐位一致，报告每流 C2HASH）。

安全纪律（§1.11 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L4_* 摘要块；
运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/l4_compose_test.py --tag timing
  python vision/l4_compose_test.py --tag main
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

from critical_point import CPLoop, mean_sd, bootstrap_ci
from compose_test import CompLoop, CTX_SPLIT_Y
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, WINDOW, RESIZE
from real_recalib import bridge_metrics
from soft_match_test import ALPHA, HITS_MIN
from cross_domain_test import (load_sampled_frames, WILD_VIDEOS, STREAMS,
                               RADIUS_L3, R_BASE_DAVIS, D246, DL_DIR,
                               guard_vs_d246, scene_switch_diag)
from fastcut_fix import run_guard_quota
from fastslow_test import (FastSlowLoop, quota_on_slow, gist_metrics,
                           build_entry_base, R_FAST, R_SLOW,
                           HITS_MIN_FAST, HITS_MIN_SLOW,
                           K_PROMOTE, K_DECAY, K_CONSIST_FAST)
from quota_retire import DeferredLoop

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 槽位旋钮（docs/253 §1.6 冻结；全部 docs/235 复用值，零重调） ----------------
K_SPLIT = 5              # 分裂资格门槛（docs/235 冻结值复用）
DELTA_REL = 0.30         # 信息量判据：中位事件能量比 >= 1+delta_rel（docs/235 冻结）
K_LEDGER = 3             # 每 c2 值最少账本窗口数（docs/235 k_consist=3 复用）
SLOT_SPARSE = 5          # 任一组事件 <5px -> c2=None（docs/235 冻结）
PARTICIPATE = 10         # 参与门：窗口总事件 >= 10（docs/235 冻结）

# docs/251 §3.3/§3.4 冻结期望（R_L4_GUARD_D251 复现目标；容差 1e-4）
D251_RATIO = {"S1": 1.155669, "S2": 1.371908, "S3": 0.732642,
              "S4": 0.370964, "R1": 0.951261}
D251_SC1_FAST = {"S1": 19, "S2": 11, "S3": 19, "S4": 39, "R1": 32}
D251_SC2_SLOW = {"S1": 5, "S2": 4, "S3": 7, "S4": 7, "R1": 5}
D251_N_PROMO = {"S1": 5, "S2": 4, "S3": 7, "S4": 7, "R1": 5}
D251_N_RECYCLE = {"S1": 12, "S2": 5, "S3": 12, "S4": 32, "R1": 23}
D251_GIST_COV = 0.8750
D251_BRIDGE_SW = 0.8750

STREAM_ORDER = [sid for sid, _, _ in STREAMS]          # S1..S4
ALL_STREAMS = STREAM_ORDER + ["R0", "R1"]


# ---------------- 槽位 c2（§1.2 冻结；逐字移植 compose_test._window_components 的 c2 部分） ----------------
def _slot_c2(ev_win):
    """逐字移植 compose_test.CompLoop._window_components（line 256）的 c2 槽位定义
    （docs/253 §1.2 冻结）：总事件 < 10 -> None；上组 = ev_win[:CTX_SPLIT_Y, :]、
    下组 = ev_win[CTX_SPLIT_Y:, :]，两组各 >= 5px 才可算
    c2 = 0 if mean_x(up) < mean_x(lo) else 1，否则 None。同一事件掩码，纯 numpy，
    无新数据路径。"""
    n = int(ev_win.sum())
    if n < PARTICIPATE:
        return None
    up = ev_win[:int(CTX_SPLIT_Y), :]
    lo = ev_win[int(CTX_SPLIT_Y):, :]
    if up.sum() >= SLOT_SPARSE and lo.sum() >= SLOT_SPARSE:
        ux = float(np.mean(np.nonzero(up)[1]))
        lx = float(np.mean(np.nonzero(lo)[1]))
        return 0 if (ux - lx) < 0 else 1
    return None


def _c2_hash(c2_trace):
    s = ",".join("N" if v is None else str(v) for v in c2_trace)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ---------------- SlotLoop（§1.3 冻结：DeferredLoop 逐字 + 槽位加法；预测路径零改动） ----------------
class SlotLoop(DeferredLoop):
    """DeferredLoop（docs/251 逐字继承）+ docs/253 §1.3 冻结槽位加法：
    (i) c2 可观测；(ii) 原型 c2 账本；(iii) 信息量驱动分裂；(iv) 打标慢原型门控匹配。
    Mode OFF = 槽位路径全关（(ii)-(iv) 不执行）-> 与 DeferredLoop 逐位一致。
    Mode ON  = 槽位路径开启。全部加法只落在模式表路径，预测/事件路径零改动。"""

    def __init__(self, mode="off", k_split=K_SPLIT, delta_rel=DELTA_REL,
                 k_ledger=K_LEDGER, **kw):
        self.mode = mode
        self.k_split = int(k_split)
        self.delta_rel = float(delta_rel)
        self.k_ledger = int(k_ledger)
        self.split_log = []        # 分裂事件 {pid(子), parent_pid, split_at, tag, birth_hits}
        self.retired_log = []      # 退休父条目 {pid, created, retired_at, parent_hits, tags}
        self.n_split = 0
        self.n_retired_slow = 0
        super().__init__(k_split=self.k_split, delta_rel=self.delta_rel, **kw)

    # ---- 账本：匹配/创建时追加 (c2, E) ----
    @staticmethod
    def _ledger_append(p, c2v, E):
        p["ledger"].setdefault(c2v, []).append(float(E))

    def _split_check(self):
        """§1.3-3 冻结：对每个已确认慢原型（hits >= hits_min_slow=3，延迟定级语义，
        慢原型只以确认已满形态存在 -> 全体慢原型满足）每窗口检查：
        hits >= k_split 且 账本 >=2 个非 None c2 值各 >= k_ledger 条目
        且 两值中位事件能量比 max/min >= 1+delta_rel -> 分裂：
        父条目退休（移出慢集、记 retired_log），按每个合格 c2 值建打标慢原型
        （tag=c2v、hits=账本计数（>=k_ledger -> 出生即确认）、mu=父 mu、
        n_match=0（post_n_match）、n_split+=1）。"""
        for p in list(self.prototypes):
            if p["kind"] != "slow":
                continue
            if p.get("tag") is not None:
                continue            # 打标慢原型账本单值 -> 不再分裂（冻结）
            if p["hits"] < self.k_split:
                continue
            quals = {k: v for k, v in p["ledger"].items()
                     if k is not None and len(v) >= self.k_ledger}
            if len(quals) < 2:
                continue
            meds = {k: float(np.median(v)) for k, v in quals.items()}
            if max(meds.values()) < (1 + self.delta_rel) * min(meds.values()):
                continue
            # 分裂：父退休 -> 按每个合格 c2 值建打标慢原型（arity-3 形态）
            pid = p["pid"]
            parent_mu = p["mu"]
            parent_hits = p["hits"]
            parent_created = p["created"]
            self.prototypes.remove(p)
            self.n_retired_slow += 1
            self.retired_log.append(dict(pid=pid, created=parent_created,
                                         retired_at=self._win,
                                         parent_hits=parent_hits,
                                         tags=sorted(k for k in quals)))
            for tagv in sorted(quals):
                cpid = self._next_pid
                self._next_pid += 1
                birth = len(quals[tagv])
                self.prototypes.append(dict(
                    pid=cpid, mu=parent_mu, hits=birth,
                    created=self._win, last_active=self._win, n_match=0,
                    kind="slow", promoted_at=self._win, tag=tagv,
                    ledger={tagv: list(quals[tagv])}))
                self.n_split += 1
                self.split_log.append(dict(pid=cpid, parent_pid=pid,
                                           split_at=self._win, tag=tagv,
                                           birth_hits=birth))

    def _on_window(self):
        if self.mode != "on":
            super()._on_window()            # Mode OFF = DeferredLoop 逐字
            return
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
        c2 = _slot_c2(ev_win)                # §1.2 冻结（每参与窗口 E>=10 计算）
        self.energy_trace.append(E)
        self.up_trace.append(U)
        self.lo_trace.append(int(ev_win[int(CTX_SPLIT_Y):, :].sum()))
        self.c2_trace.append(c2)
        self.sig_trace.append((None, None, None))
        if E >= 10:
            self.bbox_trace.append(float(U))
        else:
            self.bbox_trace.append(0.0)

        learned = False
        matched_pid = -1
        if E >= 10:
            x = (float(np.log1p(E)), float(np.log1p(U)))
            # 1. 慢优先（细半径 r_slow；打标慢原型按 c2 门控：只匹配 c2 == tag；
            #    c2=None 窗口不匹配任何打标慢原型；合格慢原型中取最近者）
            best, best_d = -1, None
            for i, p in enumerate(self.prototypes):
                if p["kind"] != "slow":
                    continue
                if p.get("tag") is not None and p["tag"] != c2:
                    continue
                d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best_d is not None and best_d <= self.r_slow:
                p = self.prototypes[best]
                self._hit(p, x)
                self._ledger_append(p, c2, E)
                learned = True
                matched_pid = p["pid"]
                self.match_trace.append((self._win, p["pid"]))
            else:
                # 2. 快兜底（gist 粗半径 r_fast；逻辑逐字不变）
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
                    self._ledger_append(p, c2, E)
                    learned = True
                    matched_pid = p["pid"]
                    self.match_trace.append((self._win, p["pid"]))
                    # 延迟定级（docs/251 逐字）：仅当 hits >= hits_min_slow 才最终化为慢
                    if p["hits"] >= self.hits_min_slow:
                        p["kind"] = "slow"
                        p["promoted_at"] = self._win
                        self.n_promoted += 1
                        self.promoted_log.append(dict(pid=p["pid"],
                                                      promoted_at=self._win,
                                                      hits=p["hits"]))
                else:
                    # 3. 高残差新奇段立即创建快原型（k_consist_fast=1；创建窗即首个账本条目）
                    pid = self._next_pid
                    self._next_pid += 1
                    self.prototypes.append(dict(pid=pid, mu=x, hits=1,
                                                created=self._win,
                                                last_active=self._win, n_match=1,
                                                kind="fast", promoted_at=None,
                                                tag=None,
                                                ledger={c2: [float(E)]}))
                    self.n_created_fast += 1
                    self.created_log.append(dict(pid=pid, created=self._win,
                                                 final_hits=None, recycled=0))
                    learned = True
                    matched_pid = pid
                    self.match_trace.append((self._win, pid))
        else:
            self.match_trace.append((self._win, None))
        # 4. 回收：快原型（含升级候选）连续 k_decay 窗未重匹配 -> 遗忘（慢豁免；逐字）
        for p in list(self.prototypes):
            if p["kind"] == "fast" and (self._win - p["last_active"]) >= self.k_decay:
                self.prototypes.remove(p)
                self.n_recycled += 1
                for cl in self.created_log:
                    if cl["pid"] == p["pid"]:
                        cl["final_hits"] = p["hits"]
                        cl["recycled"] = 1
        # 5. 分裂检查（§1.3-3 冻结；每窗口对每个已确认慢原型）
        self._split_check()
        self.soft_trace.append((round(np.log1p(E), 4), round(np.log1p(U), 4),
                                matched_pid))
        if learned:
            self._n_learn += 1
        self.sc1_cum.append(len(self.prototypes))
        self._win += 1
        self._frame_buf = []
        self._ev_win = None

    def finalize(self, n_windows, labels=None):
        if self.mode != "on":
            return super().finalize(n_windows, labels=labels)
        if self._frame_buf:
            self._on_window()
        base = CPLoop.finalize(self, n_windows)
        # created_log final_hits 回填（docs/251 逐字）
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
        sc1_slow = len(self.promoted_log)     # 累计 fast->slow promotion（不含打标子条目）
        sc2_slow = sum(1 for p in slows if p["hits"] >= self.hits_min_slow)
        tagged = [p for p in slows if p.get("tag") is not None]
        sc2_tagged = len(tagged)
        compound_frac = sc2_tagged / max(1, sc2_slow)
        # churn_slow（§1.3 冻结）：分裂使 SC2_slow 可超 SC1_slow -> max(0,·) -> 按构造 0.0
        churn_slow = max(0.0, (sc1_slow - sc2_slow) / max(1, sc1_slow))

        # 升级 vs 未升级命中率（PROMOTION 行为证据；已退休父条目用其退休时最终 hits）
        promo_pids = set(pl["pid"] for pl in self.promoted_log)
        hits_by_pid = {}
        for p in self.prototypes:
            hits_by_pid[p["pid"]] = p["hits"]
        for rl in self.retired_log:
            hits_by_pid[rl["pid"]] = rl["parent_hits"]
        promo_hits = [hits_by_pid[pl["pid"]] for pl in self.promoted_log
                      if pl["pid"] in hits_by_pid]
        nonp_hits = [cl["final_hits"] for cl in self.created_log
                     if cl["pid"] not in promo_pids]
        promo_mean = float(np.mean(promo_hits)) if promo_hits else 0.0
        nonp_mean = float(np.mean(nonp_hits)) if nonp_hits else 0.0

        # entry_log（慢 = 存活慢原型含打标子条目；快 = 创建记录含回收；退休父条目
        # 不在其中——记于 retired_log）
        entry_log = []
        for p in sorted(self.prototypes, key=lambda q: q["pid"]):
            if p["kind"] == "slow":
                rec = {"pid": p["pid"], "created": p["created"],
                       "promoted_at": p["promoted_at"],
                       "hits": p["hits"], "kind": "slow", "recycled": 0}
                if p.get("tag") is not None:
                    rec["tag"] = p["tag"]
                entry_log.append(rec)
            else:
                entry_log.append({"pid": p["pid"], "created": p["created"],
                                  "hits": p["hits"], "kind": "fast",
                                  "recycled": 0})
        for cl in self.created_log:
            if cl["recycled"]:
                entry_log.append({"pid": cl["pid"], "created": cl["created"],
                                  "hits": cl["final_hits"], "kind": "fast",
                                  "recycled": 1})
        churn_legacy = sum(1 for e in entry_log if e["hits"] < self.hits_min_slow) \
            / max(1, sc1_fast)

        # 打标慢原型：分裂后命中（n_match 自出生起计 = post_n_match）
        post_hits = [p["n_match"] for p in tagged]
        spurious_split_frac = sum(1 for v in post_hits if v == 0) / max(1, sc2_tagged)
        avg_post_split_hits = float(np.mean(post_hits)) if post_hits else 0.0
        tag_dist = {"tag0": sum(1 for p in tagged if p["tag"] == 0),
                    "tag1": sum(1 for p in tagged if p["tag"] == 1)}

        out = dict(base)
        out.update({
            "sc1_fast": sc1_fast, "sc2_fast": sc2_fast,
            "sc1_slow": sc1_slow, "sc2_slow": sc2_slow,
            "sc2_tagged": sc2_tagged, "compound_frac": round(compound_frac, 4),
            "churn_slow": round(churn_slow, 4),
            "churn_legacy": round(churn_legacy, 4),
            "n_promo": self.n_promoted, "n_recycle": self.n_recycled,
            "n_split": self.n_split, "n_retired_slow": self.n_retired_slow,
            "spurious_split_frac": round(spurious_split_frac, 4),
            "avg_post_split_hits": round(avg_post_split_hits, 4),
            "post_split_hits": post_hits,
            "tag_dist": tag_dist,
            "promoted_mean_hits": round(promo_mean, 4),
            "nonpromoted_mean_hits": round(nonp_mean, 4),
            "entry_log": entry_log,
            "created_log": self.created_log,
            "promoted_log": self.promoted_log,
            "retired_log": self.retired_log,
            "split_log": self.split_log,
            "match_summary": {
                "matched": sum(1 for _, k in self.match_trace if k is not None),
                "none": sum(1 for _, k in self.match_trace if k is None),
            },
            "soft_trace": self.soft_trace,
            "c2_trace": self.c2_trace,
        })
        return out

    # ---- M4 诊断：槽位覆盖（各流 c2 非 None 占比、0/1 占比） ----
    def slot_coverage(self):
        E = np.asarray(self.energy_trace, float)
        c2 = self.c2_trace
        parts = [i for i in range(len(E)) if E[i] >= 10]
        n = len(parts)
        non = [i for i in parts if c2[i] is not None]
        n0 = sum(1 for i in non if c2[i] == 0)
        n1 = sum(1 for i in non if c2[i] == 1)
        return {"n_participating": n, "n_non_none": len(non),
                "coverage": round(len(non) / max(1, n), 4),
                "frac0_non": round(n0 / max(1, len(non)), 4),
                "frac1_non": round(n1 / max(1, len(non)), 4),
                "frac0_all": round(n0 / max(1, n), 4),
                "frac1_all": round(n1 / max(1, n), 4)}


# ---------------- 单流运行（与 quota_retire.run_deferred_stream 同构；Mode OFF 逐位一致） ----------------
def run_slot_stream(frames, mode):
    loop = SlotLoop(mode=mode, window=WINDOW, **LOOP_CFG)
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
    if mode == "on":
        out["slot_coverage"] = loop.slot_coverage()
        out["c2_hash"] = _c2_hash(loop.c2_trace)
        out["proto_detail"] = proto_detail(loop)
    return out, loop


def proto_detail(loop):
    """逐原型明细（JSON）：存活原型 + 退休父条目（retired_log）。"""
    detail = []
    for p in sorted(loop.prototypes, key=lambda q: q["pid"]):
        detail.append({
            "pid": p["pid"], "kind": p["kind"], "tag": p.get("tag"),
            "created": p["created"], "hits": p["hits"], "n_match": p["n_match"],
            "promoted_at": p["promoted_at"],
            "ledger_counts": {("None" if k is None else str(k)): len(v)
                              for k, v in p.get("ledger", {}).items()},
            "active": True,
        })
    for rl in loop.retired_log:
        detail.append({"pid": rl["pid"], "kind": "slow", "tag": None,
                       "created": rl["created"], "hits": rl["parent_hits"],
                       "n_match": None, "promoted_at": None,
                       "ledger_counts": {}, "active": False,
                       "retired_at": rl["retired_at"],
                       "split_tags": rl["tags"]})
    return detail


# ---------------- M4 诊断：R1 段级信息量 + 分裂-段对齐（R1 有段边界 GT） ----------------
def r1_segment_info(loop, spans):
    """每段内 c2 两组中位事件能量比；两组各 >=3 窗口才可算，否则 NA（§1.4 M4 冻结）。"""
    rows = []
    E = np.asarray(loop.energy_trace, float)
    c2 = loop.c2_trace
    for i, (s0, s1) in enumerate(spans):
        g = {0: [], 1: []}
        for w in range(len(E)):
            if s0 <= w * WINDOW < s1 and E[w] >= 10 and c2[w] is not None:
                g[c2[w]].append(float(E[w]))
        if len(g[0]) >= 3 and len(g[1]) >= 3:
            med0 = float(np.median(g[0]))
            med1 = float(np.median(g[1]))
            ratio = max(med0, med1) / min(med0, med1)
            rows.append({"segment": i, "span": [s0, s1], "n0": len(g[0]),
                         "n1": len(g[1]), "med0": round(med0, 1),
                         "med1": round(med1, 1), "ratio": round(ratio, 4)})
        else:
            rows.append({"segment": i, "span": [s0, s1], "n0": len(g[0]),
                         "n1": len(g[1]), "med0": None, "med1": None,
                         "ratio": None})
    return rows


def split_segment_align(loop, spans, seg_rows):
    """每个分裂事件：父条目匹配窗口的众数段、该段信息量比；
    align_rate = #{分裂 : 众数段信息量比 >= 1.30} / max(1, n_split)（诊断级）。"""
    matched = {}
    for w, pid in loop.match_trace:
        if pid is not None:
            matched.setdefault(pid, []).append(w)
    seg_by_win = {}
    for i, (s0, s1) in enumerate(spans):
        for w in range(len(loop.energy_trace)):
            if s0 <= w * WINDOW < s1:
                seg_by_win[w] = i
    aligned = 0
    per = []
    for sl in loop.split_log:
        wins = matched.get(sl["parent_pid"], [])
        if wins:
            cnt = Counter(seg_by_win[w] for w in wins)
            mode_seg, _ = cnt.most_common(1)[0]
        else:
            mode_seg = None
        ratio = None
        if mode_seg is not None:
            ratio = seg_rows[mode_seg]["ratio"]
        ok = int(ratio is not None and ratio >= (1 + DELTA_REL))
        aligned += ok
        per.append({"parent_pid": sl["parent_pid"], "split_at": sl["split_at"],
                    "n_parent_matches": len(wins), "mode_segment": mode_seg,
                    "seg_ratio": (round(ratio, 4) if ratio is not None else None),
                    "aligned": ok})
    n_split = len(loop.split_log)
    return {"n_split": n_split, "n_aligned": aligned,
            "align_rate": round(aligned / max(1, n_split), 4), "per_split": per}


# ---------------- 守卫：Mode OFF 复现 docs/251（32 项；容差 1e-4） ----------------
def guard_d251_items(off, off_r1):
    items = []
    for sid in STREAM_ORDER:
        r = off[sid]
        for name, exp in (("RATIO", D251_RATIO[sid]),
                          ("SC1_FAST", D251_SC1_FAST[sid]),
                          ("SC2_SLOW", D251_SC2_SLOW[sid]),
                          ("N_PROMO", D251_N_PROMO[sid]),
                          ("N_RECYCLE", D251_N_RECYCLE[sid]),
                          ("CHURN_SLOW", 0.0)):
            got = r["ratio" if name == "RATIO" else
                   "sc1_fast" if name == "SC1_FAST" else
                   "sc2_slow" if name == "SC2_SLOW" else
                   "n_promo" if name == "N_PROMO" else
                   "n_recycle" if name == "N_RECYCLE" else "churn_slow"]
            ok = int(got == exp) if isinstance(exp, int) else \
                int(abs(got - exp) < 1e-4)
            items.append(("%s_%s" % (sid, name), ok))
    r1 = off_r1
    items.append(("R1_RATIO", int(abs(r1["ratio"] - D251_RATIO["R1"]) < 1e-4)))
    items.append(("R1_SC1_FAST", int(r1["sc1_fast"] == D251_SC1_FAST["R1"])))
    items.append(("R1_SC2_SLOW", int(r1["sc2_slow"] == D251_SC2_SLOW["R1"])))
    items.append(("R1_N_PROMO", int(r1["n_promo"] == D251_N_PROMO["R1"])))
    items.append(("R1_N_RECYCLE", int(r1["n_recycle"] == D251_N_RECYCLE["R1"])))
    items.append(("R1_CHURN_SLOW", int(abs(r1["churn_slow"] - 0.0) < 1e-4)))
    items.append(("R1_GIST_COV", int(abs(r1["gist"]["cov"] - D251_GIST_COV) < 1e-4)))
    items.append(("R1_BRIDGE_SW",
                  int(abs(r1["bridge"]["bridge_corr_switch"] - D251_BRIDGE_SW) < 1e-4)))
    return items


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    t_dec = t_off = t_on = 0.0

    # ---- 数据加载（一次，两模式复用；docs/248 §1.2/§1.5 逐字） ----
    loaded = {}
    for vid, name in WILD_VIDEOS:
        p = os.path.join(DL_DIR, name)
        frames, step, total = load_sampled_frames(p)
        loaded[vid] = frames
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    stream_frames = {}
    for sid, sname, vidx in STREAMS:
        fr = []
        for vi in vidx:
            fr.extend(loaded[WILD_VIDEOS[vi][0]])
        stream_frames[sid] = fr
    t_dec = time.time() - t0

    # ---- Mode OFF（DeferredLoop 逐字；守卫） ----
    off = {}
    for sid in STREAM_ORDER:
        out, _ = run_slot_stream(stream_frames[sid], "off")
        out["stream_id"] = sid
        off[sid] = out
    off_r0, _ = run_slot_stream(r0_frames, "off")
    off_r1, _ = run_slot_stream(r1_frames, "off")
    off_r1["bridge"] = bridge_metrics(build_entry_base(off_r1), spans)
    off_r1["gist"] = gist_metrics(off_r1, switch_windows)
    t_off = time.time() - t0 - t_dec

    d251_items = guard_d251_items(off, off_r1)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- Mode ON（槽位路径开启；判据口径） ----
    on = {}
    for sid, sname, vidx in STREAMS:
        out, loop = run_slot_stream(stream_frames[sid], "on")
        out["stream_id"] = sid
        out["stream_name"] = sname
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        out["switch_diag"] = scene_switch_diag(stream_frames[sid], creations)
        on[sid] = out
    on_r0, on_r0_loop = run_slot_stream(r0_frames, "on")
    on_r1, on_r1_loop = run_slot_stream(r1_frames, "on")
    on_r1["bridge"] = bridge_metrics(build_entry_base(on_r1), spans)
    on_r1["gist"] = gist_metrics(on_r1, switch_windows)
    on_r1["seg_info"] = r1_segment_info(on_r1_loop, spans)
    on_r1["split_align"] = split_segment_align(on_r1_loop, spans, on_r1["seg_info"])
    t_on = time.time() - t0 - t_dec - t_off

    # ---- R_L4_REPRO_RATIO（构造性控制项：ON vs OFF 全流 ratio，abs < 1e-9） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            int(abs(on[sid]["ratio"] - off[sid]["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0", int(abs(on_r0["ratio"] - off_r0["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R1", int(abs(on_r1["ratio"] - off_r1["ratio"]) < 1e-9)))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, v) for n, v in repro_items)

    # ---- R_L4_GUARD_D246（SoftLoop 路径；docs/249/250/251 同一代码路径） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)
    guard246_passed = sum(1 for ch in guard246_detail.split(",") if ch.endswith(":1"))

    # ---- 判据（§1.7 冻结；ON 数字） ----
    crit1 = int(on_r1["n_split"] >= 1 and on_r1["sc2_tagged"] >= 1
                and on_r1["compound_frac"] >= 0.5)
    crit2 = int(on_r1["spurious_split_frac"] <= 0.5
                and on_r1["avg_post_split_hits"] >= 1
                and on_r0["n_split"] == 0)
    all_found = [on[s] for s in STREAM_ORDER] + [on_r1]
    crit3 = int(all(r["ratio"] <= 1.5 for r in all_found)
                and all(r["sc2_slow"] > 0 for r in all_found)
                and on_r1["gist"]["cov"] >= 0.5)
    n_promo_total = sum(r["n_promo"] for r in all_found)
    n_recycle_total = sum(r["n_recycle"] for r in all_found)
    promo_means = [(r["promoted_mean_hits"], r["nonpromoted_mean_hits"])
                   for r in all_found if r["sc1_slow"] > 0]
    promo_sep = int(any(mp > mn for mp, mn in promo_means)) if promo_means else 0
    crit4 = int(n_promo_total > 0 and n_recycle_total > 0 and promo_sep)

    # ---- 判定（§1.7 冻结映射） ----
    guards_ok = d251_ok == 1 and guard246_ok == 1 and repro_ok == 1
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d -> implementation drift; fix implementation, "
                 "do not judge mechanism (see R_L4_GUARD_*)" % (
                     d251_ok, d251_passed, guard246_ok, repro_ok))
    elif not crit1:
        verdict = "PARALLEL_ONLY_REAL"
        vnote = ("COMPOUND_EMERGES fails on R1: n_split=%d, SC2_tagged=%d, "
                 "compound_frac=%.4f (frozen slot constants may be too strict; "
                 "no threshold rollback)" % (
                     on_r1["n_split"], on_r1["sc2_tagged"],
                     on_r1["compound_frac"]))
    elif not crit2:
        verdict = "BOUNDARY"
        vnote = ("COMPOUND_EMERGES passes but ADOPT_NONRANDOM fails: "
                 "spurious_split_frac=%.4f, avg_post_split_hits=%.4f, "
                 "R0 n_split=%d" % (
                     on_r1["spurious_split_frac"],
                     on_r1["avg_post_split_hits"], on_r0["n_split"]))
    elif not (crit3 and crit4):
        why = []
        if not crit3:
            why.append("FOUNDATION_KEEP fails (ratio/sc2_slow/gist_cov; see numbers)")
        if not crit4:
            why.append("PROMOTION_KEEP fails (n_promo/n_recycle/hit-rate separation)")
        verdict = "PARTIAL_REAL"
        vnote = "; ".join(why) + " (see R_L4_CRIT* numbers)"
    else:
        verdict = "COMPOSABLE_REAL"
        vnote = ("criteria 1-4 all pass and all guards pass: compound structure "
                 "emerges on real streams (tagged conditional slow memory), "
                 "adoption is non-random (negative control clean), L3 foundation "
                 "and promotion behavior evidence kept")

    # ---- 工件（自描述 JSON） ----
    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "working_point": {"r_slow": round(R_SLOW, 6),
                             "r_fast": round(R_FAST, 6),
                             "hits_min_fast": HITS_MIN_FAST,
                             "hits_min_slow": HITS_MIN_SLOW,
                             "k_promote": K_PROMOTE,
                             "k_decay": K_DECAY,
                             "k_consist_fast": K_CONSIST_FAST,
                             "alpha": ALPHA,
                             "k_split": K_SPLIT,
                             "delta_rel": DELTA_REL,
                             "k_ledger": K_LEDGER,
                             "ctx_split_y": CTX_SPLIT_Y,
                             "slot_sparse_px": SLOT_SPARSE,
                             "participate": PARTICIPATE},
           "mechanism": ("SlotLoop(DeferredLoop): c2 slot observable (verbatim "
                         "port of compose_test._window_components c2), prototype "
                         "c2 ledger (creation window = first entry, no backfill), "
                         "info-driven split of confirmed slow prototypes "
                         "(hits>=k_split, >=2 non-None c2 values each >=k_ledger, "
                         "median energy ratio >=1+delta_rel), tagged slow gated "
                         "matching (only c2==tag); Mode OFF = DeferredLoop verbatim; "
                         "prediction path zero-change"),
           "loop": LOOP_CFG,
           "r1_switch_windows": switch_windows,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l4_compose_test",
        "doc_ref": "docs/235, docs/243, docs/245, docs/246, docs/247, docs/248, "
                   "docs/249, docs/250, docs/251, docs/253",
        "config": cfg,
        "off_guard": {"items": len(d251_items), "passed": d251_passed,
                      "ok": d251_ok, "detail": d251_detail,
                      "streams": {sid: {k: off[sid][k] for k in
                                        ("frames", "n_windows", "ratio",
                                         "sc1_fast", "sc2_fast", "sc1_slow",
                                         "sc2_slow", "churn_slow", "churn_legacy",
                                         "n_promo", "n_recycle",
                                         "promoted_mean_hits",
                                         "nonpromoted_mean_hits")}
                                  for sid in STREAM_ORDER},
                      "r0": {k: off_r0[k] for k in ("frames", "ratio", "sc1_fast",
                                                    "sc2_fast", "sc1_slow",
                                                    "sc2_slow", "churn_slow",
                                                    "n_promo", "n_recycle")},
                      "r1": {k: off_r1[k] for k in ("frames", "ratio", "sc1_fast",
                                                    "sc2_fast", "sc1_slow",
                                                    "sc2_slow", "churn_slow",
                                                    "churn_legacy", "n_promo",
                                                    "n_recycle")},
                      "r1_gist_cov": off_r1["gist"]["cov"],
                      "r1_bridge_sw": off_r1["bridge"]["bridge_corr_switch"]},
        "on_streams": {sid: on[sid] for sid in STREAM_ORDER},
        "on_r0": on_r0, "on_r1": on_r1,
        "criteria": {"crit1_compound_emerges": crit1,
                     "crit2_adopt_nonrandom": crit2,
                     "crit3_foundation_keep": crit3,
                     "crit4_promotion_keep": crit4,
                     "n_promo_total": n_promo_total,
                     "n_recycle_total": n_recycle_total,
                     "promo_means": promo_means, "promo_sep": promo_sep},
        "verdict": {"verdict": verdict, "note": vnote},
        "guards": {"d251": {"items": len(d251_items), "passed": d251_passed,
                            "ok": d251_ok, "detail": d251_detail},
                   "d246": {"ok": guard246_ok, "passed": guard246_passed,
                            "detail": guard246_detail,
                            "r0": {"sc2": g0["sc2"], "churn": g0["churn_frac"],
                                   "ratio": g0["ratio"]},
                            "r1": {"sc1": g1["sc1"], "sc2": g1["sc2"],
                                   "churn": g1["churn_frac"],
                                   "ratio": g1["ratio"]}},
                   "repro_ratio": {"ok": repro_ok, "detail": repro_detail}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2),
                   "decode_sec": round(t_dec, 2),
                   "off_sec": round(t_off, 2), "on_sec": round(t_on, 2)},
    }
    res_path = os.path.join(args.out_dir, "l4c_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L4_TAG=%s" % args.tag)
    print("R_L4_R_SLOW=%.6f" % R_SLOW)
    print("R_L4_R_FAST=%.6f" % R_FAST)
    print("R_L4_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_L4_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_L4_K_PROMOTE=%d" % K_PROMOTE)
    print("R_L4_K_DECAY=%d" % K_DECAY)
    print("R_L4_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_L4_ALPHA=%.4f" % ALPHA)
    print("R_L4_K_SPLIT=%d" % K_SPLIT)
    print("R_L4_DELTA_REL=%.4f" % DELTA_REL)
    print("R_L4_K_LEDGER=%d" % K_LEDGER)
    print("R_L4_CTX_SPLIT_Y=%.1f" % CTX_SPLIT_Y)
    for j, sid in enumerate(STREAM_ORDER):
        r = off[sid]
        print("R_L4_OFF_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4_OFF_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4_OFF_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4_OFF_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4_OFF_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4_OFF_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
    print("R_L4_OFF_R1_RATIO=%.6f" % off_r1["ratio"])
    print("R_L4_OFF_R1_SC1_FAST=%d" % off_r1["sc1_fast"])
    print("R_L4_OFF_R1_SC2_SLOW=%d" % off_r1["sc2_slow"])
    print("R_L4_OFF_R1_N_PROMO=%d" % off_r1["n_promo"])
    print("R_L4_OFF_R1_N_RECYCLE=%d" % off_r1["n_recycle"])
    print("R_L4_OFF_R1_CHURN_SLOW=%.4f" % off_r1["churn_slow"])
    print("R_L4_OFF_R1_GIST_COV=%.4f" % off_r1["gist"]["cov"])
    print("R_L4_OFF_R1_BRIDGE_SW=%.4f" % off_r1["bridge"]["bridge_corr_switch"])
    print("R_L4_OFF_R0_RATIO=%.6f" % off_r0["ratio"])
    print("R_L4_OFF_R0_SC1_FAST=%d" % off_r0["sc1_fast"])
    print("R_L4_OFF_R0_SC2_SLOW=%d" % off_r0["sc2_slow"])
    print("R_L4_OFF_R0_CHURN_SLOW=%.4f" % off_r0["churn_slow"])
    print("R_L4_OFF_R0_N_PROMO=%d" % off_r0["n_promo"])
    print("R_L4_OFF_R0_N_RECYCLE=%d" % off_r0["n_recycle"])
    print("R_L4_GUARD_D251=%d" % d251_ok)
    print("R_L4_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L4_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L4_GUARD_D251_DETAIL=%s" % d251_detail)
    for sid in ALL_STREAMS:
        if sid in on:
            r = on[sid]
        elif sid == "R0":
            r = on_r0
        else:
            r = on_r1
        sc = r["slot_coverage"]
        d = r.get("switch_diag")
        sw = "NA" if d is None or d["switch_corr"] is None else \
            "%.4f" % d["switch_corr"]
        print("R_L4_ON_%s_FRAMES=%d" % (sid, r["frames"]))
        print("R_L4_ON_%s_WINDOWS=%d" % (sid, r["n_windows"]))
        print("R_L4_ON_%s_VALID=%d" % (sid, r["n_valid"]))
        print("R_L4_ON_%s_MAE=%.6f" % (sid, r["mae_mean_win"]))
        print("R_L4_ON_%s_MAE_SD=%.6f" % (sid, r["mae_sd_win"]))
        print("R_L4_ON_%s_MAE_LO=%.6f" % (sid, r["mae_ci95"][0]))
        print("R_L4_ON_%s_MAE_HI=%.6f" % (sid, r["mae_ci95"][1]))
        print("R_L4_ON_%s_Q1=%.6f" % (sid, r["mae_q1"]))
        print("R_L4_ON_%s_Q4=%.6f" % (sid, r["mae_q4"]))
        print("R_L4_ON_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4_ON_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4_ON_%s_SC2_FAST=%d" % (sid, r["sc2_fast"]))
        print("R_L4_ON_%s_SC1_SLOW=%d" % (sid, r["sc1_slow"]))
        print("R_L4_ON_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4_ON_%s_SC2_TAGGED=%d" % (sid, r["sc2_tagged"]))
        print("R_L4_ON_%s_COMPOUND_FRAC=%.4f" % (sid, r["compound_frac"]))
        print("R_L4_ON_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
        print("R_L4_ON_%s_CHURN_LEGACY=%.4f" % (sid, r["churn_legacy"]))
        print("R_L4_ON_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4_ON_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4_ON_%s_N_SPLIT=%d" % (sid, r["n_split"]))
        print("R_L4_ON_%s_N_RETIRED_SLOW=%d" % (sid, r["n_retired_slow"]))
        print("R_L4_ON_%s_SPURIOUS_SPLIT_FRAC=%.4f" % (sid, r["spurious_split_frac"]))
        print("R_L4_ON_%s_AVG_POST_SPLIT_HITS=%.4f" % (sid, r["avg_post_split_hits"]))
        print("R_L4_ON_%s_PROMO_MEAN=%.4f" % (sid, r["promoted_mean_hits"]))
        print("R_L4_ON_%s_NONPROMO_MEAN=%.4f" % (sid, r["nonpromoted_mean_hits"]))
        print("R_L4_ON_%s_C2HASH=%s" % (sid, r["c2_hash"]))
        print("R_L4_ON_%s_C2_COV=%.4f" % (sid, sc["coverage"]))
        print("R_L4_ON_%s_C2_F0_NON=%.4f" % (sid, sc["frac0_non"]))
        print("R_L4_ON_%s_C2_F1_NON=%.4f" % (sid, sc["frac1_non"]))
        print("R_L4_ON_%s_C2_F0_ALL=%.4f" % (sid, sc["frac0_all"]))
        print("R_L4_ON_%s_C2_F1_ALL=%.4f" % (sid, sc["frac1_all"]))
        print("R_L4_ON_%s_TAG0=%d" % (sid, r["tag_dist"]["tag0"]))
        print("R_L4_ON_%s_TAG1=%d" % (sid, r["tag_dist"]["tag1"]))
        print("R_L4_ON_%s_SW_CORR=%s" % (sid, sw))
    print("R_L4_ON_R1_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    seg = on_r1["seg_info"]
    print("R_L4_ON_R1_SEG_INFO=%s" % ",".join(
        ("NA" if row["ratio"] is None else "%.4f" % row["ratio"]) for row in seg))
    print("R_L4_ON_R1_SEG_N0=%s" % ",".join(str(row["n0"]) for row in seg))
    print("R_L4_ON_R1_SEG_N1=%s" % ",".join(str(row["n1"]) for row in seg))
    print("R_L4_ON_R1_ALIGN_RATE=%.4f" % on_r1["split_align"]["align_rate"])
    print("R_L4_ON_R1_N_ALIGNED=%d" % on_r1["split_align"]["n_aligned"])
    print("R_L4_REPRO_RATIO=%d" % repro_ok)
    print("R_L4_REPRO_DETAIL=%s" % repro_detail)
    print("R_L4_CRIT1_COMPOUND_EMERGES=%d" % crit1)
    print("R_L4_CRIT1_N_SPLIT=%d" % on_r1["n_split"])
    print("R_L4_CRIT1_SC2_TAGGED=%d" % on_r1["sc2_tagged"])
    print("R_L4_CRIT1_COMPOUND_FRAC=%.4f" % on_r1["compound_frac"])
    print("R_L4_CRIT2_ADOPT_NONRANDOM=%d" % crit2)
    print("R_L4_CRIT2_SPURIOUS_FRAC=%.4f" % on_r1["spurious_split_frac"])
    print("R_L4_CRIT2_AVG_POST_HITS=%.4f" % on_r1["avg_post_split_hits"])
    print("R_L4_CRIT2_R0_N_SPLIT=%d" % on_r0["n_split"])
    print("R_L4_CRIT3_FOUNDATION_KEEP=%d" % crit3)
    print("R_L4_CRIT3_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    print("R_L4_CRIT4_PROMOTION_KEEP=%d" % crit4)
    print("R_L4_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L4_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L4_GUARD_D246=%d" % guard246_ok)
    print("R_L4_GUARD_D246_ITEMS=%d" % 12)
    print("R_L4_GUARD_D246_PASSED=%d" % guard246_passed)
    print("R_L4_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L4_VERDICT=%s" % verdict)
    print("R_L4_VERDICT_NOTE=%s" % vnote)
    print("R_L4_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
