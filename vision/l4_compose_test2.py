"""vision/l4_compose_test2.py — docs/254 L4 组合泛化第二格：可达性修复（SlotLoop2）。

SlotLoop2(SlotLoop)（docs/254 §一 冻结，运行后不改）：
  Mode OFF = super()._on_window()（SlotLoop OFF = DeferredLoop 逐字 -> R_L4B_GUARD_D251
             复现 docs/251 §3.3/§3.4，32 项，容差 1e-4）；
  Mode ON  = docs/253 槽位路径（c2 可观测 + 账本 + 分裂 + 门控匹配）+ 三条账本可达性
             修复加法（全部只落在模式表路径，预测路径零改动）：
             ① 出生回填（docs/235 D6 直译 + 邻域约束）：快原型创建时，用创建窗之前
                最近 <= W_BF=4 个参与窗口（E>=10）中特征距离 <= r_fast 者的 (c2,E,U)
                回填账本（c2=None 记 ledger[None]）；回填不加 hits（hits=匹配确认计数；
                账本=阅历事件簿——会计区分冻结）；
             ② 升级并账显式化：快->慢升级时完整生命周期账本（出生回填+快阶段匹配+
                慢阶段匹配）为分裂评估口径（升级分支不动 ledger——第一格已隐式成立，
                本格显式冻结语义，确保代码不重置/不清空账本）；
             ③ 子条目 μ 重初始化：分裂时打标子条目 μ = 其 tag 组账本窗口
                (ln(1+E), ln(1+U)) 中位数向量（np.median per 分量），不再继承父 μ。
  账本条目从 docs/253 的 (c2, E) 扩展为 (c2, E, U)（U 供 μ 重初始化；信息量判据只用 E）。

度量（§1.3 冻结）：M1-M4 与 docs/253 逐字一致；新增 M5 出生回填诊断（每流：发生回填
的创建数、回填窗口总数、平均每创建回填条数、回填窗口 c2 分布、回填-分裂关联）——
诊断级，不进判据。

判据（§1.6 冻结，与 docs/253 逐字一致）：①[L4][机制][组合测试] COMPOUND_EMERGES
（R1：n_split>=1 且 SC2_tagged>=1 且 compound_frac>=0.5）；②[L4][机制][行为证据]
ADOPT_NONRANDOM（R1：spurious_split_frac<=0.5 且 avg_post_split_hits>=1；R0：
n_split==0）；③[L4][机制] FOUNDATION_KEEP（R1+S1-S4：ratio<=1.5 且 SC2_slow>0；R1：
gist_cov>=0.5）；④[L4][机制][行为证据] PROMOTION_KEEP（全局 n_promo>0 且 n_recycle>0
且升级命中率均值>未升级均值）。判定映射按 §1.7（第二格专属语义：修复后仍 n_split=0
= 更强负结果；分裂但 spurious=BOUNDARY）。

守卫（§1.8 冻结，不进判据）：R_L4B_GUARD_D251（Mode OFF 复现 docs/251，32 项）、
R_L4B_GUARD_D246（run_guard_quota + guard_vs_d246，12/12）、R_L4B_REPRO_RATIO
（ON vs OFF 全流 ratio abs<1e-9）、R_L4B_BF_HASH（出生回填日志确定性指纹，两轮逐位
一致）、R_L4B_NONSPLIT_EQ（Mode ON 非分裂数字 vs docs/253 Mode ON，匹配/升级/回收/
预测路径未动——内部复现，docs/254 §1.9 重测声明）。

安全纪律（§1.11 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L4B_* 摘要块；
运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/l4_compose_test2.py --smoke        # 构造冒烟（合成帧，非数据）
  python vision/l4_compose_test2.py --tag timing
  python vision/l4_compose_test2.py --tag main
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
# import 复用 l4_compose_test（docs/254 §1.11 同款清单 + 本格专属复用）
from l4_compose_test import (SlotLoop, K_SPLIT, DELTA_REL, K_LEDGER,
                             SLOT_SPARSE, PARTICIPATE, _slot_c2, _c2_hash,
                             r1_segment_info, split_segment_align,
                             guard_d251_items, STREAM_ORDER)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 本格唯一新旋钮（docs/254 §1.5 冻结；docs/235 _hist 冻结值复用） ----------------
W_BF = 4                # 出生回填窗数上限（= compose_test._hist 长度，docs/235 D6 实现常量）

# docs/253 Mode ON 非分裂数字（内部复现 R_L4B_NONSPLIT_EQ 的期望；docs/254 §1.9 声明：
# 匹配/升级/回收/预测路径未动 -> 非分裂数字与 docs/253 逐位一致）
D253_ON = {
    "S1": dict(sc1_fast=19, sc2_fast=19, n_promo=5, n_recycle=12,
               ratio=1.155669, mae=0.195664),
    "S2": dict(sc1_fast=11, sc2_fast=10, n_promo=4, n_recycle=5,
               ratio=1.371908, mae=0.021484),
    "S3": dict(sc1_fast=19, sc2_fast=19, n_promo=7, n_recycle=12,
               ratio=0.732642, mae=0.072195),
    "S4": dict(sc1_fast=39, sc2_fast=39, n_promo=7, n_recycle=32,
               ratio=0.370964, mae=0.101345),
    "R0": dict(sc1_fast=18, sc2_fast=18, n_promo=1, n_recycle=16,
               ratio=0.907701, mae=0.097127),
    "R1": dict(sc1_fast=32, sc2_fast=32, n_promo=5, n_recycle=23,
               ratio=0.951261, mae=0.069522),
}

ALL_STREAMS = STREAM_ORDER + ["R0", "R1"]


# ---------------- SlotLoop2（docs/254 §1.2 冻结：SlotLoop 逐字 + 三条账本可达性修复） ----------------
class SlotLoop2(SlotLoop):
    """SlotLoop（docs/253 逐字）+ docs/254 §1.2 冻结三条账本可达性修复加法：
    ① 出生回填（W_BF=4 / r_fast 邻域；回填不加 hits）；
    ② 升级并账显式化（升级分支不动 ledger——完整生命周期账本为分裂评估口径）；
    ③ 子条目 μ 重初始化（tag 组账本窗口 (ln(1+E), ln(1+U)) 中位数向量）。
    账本条目从 (c2, E) 扩展为 (c2, E, U)。Mode OFF = super()._on_window()
    （= SlotLoop OFF = DeferredLoop 逐字）。"""

    def __init__(self, mode="off", k_split=K_SPLIT, delta_rel=DELTA_REL,
                 k_ledger=K_LEDGER, w_bf=W_BF, **kw):
        self.w_bf = int(w_bf)
        self._hist = []              # 最近 <=W_BF 个参与窗口 (win, x, c2, E, U)（加法① 窗口史）
        self.bf_log = []             # 出生回填日志 {pid, created, n_backfill, source_wins, bf_hash}
        self.n_creations_with_bf = 0
        self.bf_window_total = 0
        self.bf_c2_dist = {"None": 0, "0": 0, "1": 0}
        super().__init__(mode=mode, k_split=k_split, delta_rel=delta_rel,
                         k_ledger=k_ledger, **kw)

    # ---- 账本：匹配/创建时追加 (c2, E, U)（docs/254 §1.2-2：条目扩展为 (E, U)） ----
    @staticmethod
    def _ledger_append(p, c2v, E, U):
        p["ledger"].setdefault(c2v, []).append((float(E), float(U)))

    def _bf_hash(self, seq):
        """回填窗口 (c2, E, U) 序列的确定性指纹（R_L4B_BF_HASH 口径；回填顺序 = 窗口史顺序）。"""
        s = ";".join("%s|%d|%d" % ("N" if c is None else str(c), e, u)
                     for c, e, u in seq)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _bf_log_hash(self):
        """整流回填日志的确定性指纹（每创建：pid/created/n_backfill/来源窗/序列 hash）。"""
        s = ";".join("%d@%d@%d@[%s]@%s" % (
            b["pid"], b["created"], b["n_backfill"],
            ",".join(str(w) for w in b["source_wins"]), b["bf_hash"])
            for b in self.bf_log)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _split_check(self):
        """§1.2-3 冻结：触发条件逐字不变（hits>=k_split 且 账本 >=2 非 None c2 值各 >=
        k_ledger 且 两值中位事件能量比 >= 1+delta_rel——信息量判据只用 E 分量）；分裂时
        子条目（加法③）：tag=c2v、hits=账本计数（>=k_ledger -> 出生即确认）、
        μ = tag 组账本窗口 (ln(1+E), ln(1+U)) 中位数向量（np.median per 分量，不再继承
        父 μ）、n_match=0（post_n_match）、ledger={tagv: 该组全部条目（含回填）}。"""
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
            meds = {k: float(np.median([e for e, _ in v])) for k, v in quals.items()}
            if max(meds.values()) < (1 + self.delta_rel) * min(meds.values()):
                continue
            # 分裂：父退休 -> 按每个合格 c2 值建打标慢原型（arity-3 形态）
            pid = p["pid"]
            parent_hits = p["hits"]
            parent_created = p["created"]
            parent_n_bf = p.get("n_backfill", 0)
            self.prototypes.remove(p)
            self.n_retired_slow += 1
            self.retired_log.append(dict(pid=pid, created=parent_created,
                                         retired_at=self._win,
                                         parent_hits=parent_hits,
                                         parent_n_backfill=parent_n_bf,
                                         tags=sorted(k for k in quals)))
            for tagv in sorted(quals):
                cpid = self._next_pid
                self._next_pid += 1
                entries = list(quals[tagv])       # [(E, U), ...]（含回填条目）
                feats = np.asarray([[float(np.log1p(e)), float(np.log1p(u))]
                                    for e, u in entries], dtype=float)
                mu = (float(np.median(feats[:, 0])),
                      float(np.median(feats[:, 1])))
                birth = len(entries)
                self.prototypes.append(dict(
                    pid=cpid, mu=mu, hits=birth,
                    created=self._win, last_active=self._win, n_match=0,
                    kind="slow", promoted_at=self._win, tag=tagv,
                    n_backfill=0,
                    ledger={tagv: list(entries)}))
                self.n_split += 1
                self.split_log.append(dict(pid=cpid, parent_pid=pid,
                                           split_at=self._win, tag=tagv,
                                           birth_hits=birth))

    def _on_window(self):
        if self.mode != "on":
            super()._on_window()            # Mode OFF = SlotLoop OFF = DeferredLoop 逐字
            return
        # ---- Mode ON：docs/253 槽位路径逐字 + 三条账本修复加法（§1.2 冻结） ----
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
                self._ledger_append(p, c2, E, U)
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
                    self._ledger_append(p, c2, E, U)
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
                        # 加法② 升级并账（显式冻结）：完整生命周期账本（出生回填 +
                        # 快阶段匹配）随原型对象继承为慢账本——分裂评估口径；
                        # 不重置、不清空 p["ledger"]（第一格已隐式成立，本格显式化）
                else:
                    # 3. 高残差新奇段立即创建快原型（k_consist_fast=1；创建窗即首个
                    #    账本条目 (c2, E, U)）——加法① 出生回填：创建窗之前最近
                    #    <=W_BF 个参与窗口中特征距离 <= r_fast 者的 (c2, E, U)
                    #    回填账本（c2=None 记 ledger[None]）；回填不加 hits
                    pid = self._next_pid
                    self._next_pid += 1
                    ledger = {c2: [(float(E), float(U))]}
                    src_wins = []
                    seq = []
                    for (hw, hx, hc2, hE, hU) in self._hist:
                        d = float(np.hypot(x[0] - hx[0], x[1] - hx[1]))
                        if d <= self.r_fast:
                            ledger.setdefault(hc2, []).append((float(hE), float(hU)))
                            src_wins.append(hw)
                            seq.append((hc2, int(hE), int(hU)))
                            self.bf_window_total += 1
                            self.bf_c2_dist["None" if hc2 is None else str(hc2)] += 1
                    n_bf = len(src_wins)
                    if n_bf > 0:
                        self.n_creations_with_bf += 1
                        self.bf_log.append(dict(pid=pid, created=self._win,
                                                n_backfill=n_bf,
                                                source_wins=list(src_wins),
                                                bf_hash=self._bf_hash(seq)))
                    self.prototypes.append(dict(pid=pid, mu=x, hits=1,
                                                created=self._win,
                                                last_active=self._win, n_match=1,
                                                kind="fast", promoted_at=None,
                                                tag=None, n_backfill=n_bf,
                                                ledger=ledger))
                    self.n_created_fast += 1
                    self.created_log.append(dict(pid=pid, created=self._win,
                                                 final_hits=None, recycled=0,
                                                 n_backfill=n_bf))
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
        # 5. 分裂检查（§1.2-3 冻结；每窗口对每个已确认慢原型）
        self._split_check()
        # 6. 窗口史维护（加法①：最近 <=W_BF 个参与窗口 E>=10）
        if E >= 10:
            self._hist.append((self._win, x, c2, E, U))
            if len(self._hist) > self.w_bf:
                self._hist.pop(0)
        self.soft_trace.append((round(np.log1p(E), 4), round(np.log1p(U), 4),
                                matched_pid))
        if learned:
            self._n_learn += 1
        self.sc1_cum.append(len(self.prototypes))
        self._win += 1
        self._frame_buf = []
        self._ev_win = None

    def finalize(self, n_windows, labels=None):
        out = super().finalize(n_windows, labels=labels)
        if self.mode != "on":
            return out
        # ---- M5 出生回填诊断（§1.3 冻结，诊断级，不进判据） ----
        n_creations = len(self.created_log)
        bf_diag = {
            "n_creations": n_creations,
            "n_creations_with_bf": self.n_creations_with_bf,
            "bf_window_total": self.bf_window_total,
            "bf_avg_per_creation": round(
                self.bf_window_total / max(1, n_creations), 4),
            "bf_avg_per_bf_creation": round(
                self.bf_window_total / max(1, self.n_creations_with_bf), 4),
            "bf_c2_dist": dict(self.bf_c2_dist),
            "bf_log": self.bf_log,
            "bf_hash_all": self._bf_log_hash(),
        }
        bf_by_pid = {b["pid"]: b["n_backfill"] for b in self.bf_log}
        bf_diag["split_parent_bf"] = [
            {"parent_pid": sl["parent_pid"], "split_at": sl["split_at"],
             "parent_n_backfill": bf_by_pid.get(sl["parent_pid"], 0)}
            for sl in self.split_log]
        out["backfill_diag"] = bf_diag
        return out


# ---------------- 单流运行（与 run_slot_stream 同构；Mode OFF 逐位一致） ----------------
def run_slot2_stream(frames, mode):
    loop = SlotLoop2(mode=mode, window=WINDOW, **LOOP_CFG)
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
        out["proto_detail"] = proto_detail2(loop)
    return out, loop


def proto_detail2(loop):
    """逐原型明细（JSON）：存活原型 + 退休父条目（retired_log），含回填计数。"""
    detail = []
    for p in sorted(loop.prototypes, key=lambda q: q["pid"]):
        detail.append({
            "pid": p["pid"], "kind": p["kind"], "tag": p.get("tag"),
            "created": p["created"], "hits": p["hits"], "n_match": p["n_match"],
            "promoted_at": p["promoted_at"],
            "n_backfill": p.get("n_backfill", 0),
            "ledger_counts": {("None" if k is None else str(k)): len(v)
                              for k, v in p.get("ledger", {}).items()},
            "active": True,
        })
    for rl in loop.retired_log:
        detail.append({"pid": rl["pid"], "kind": "slow", "tag": None,
                       "created": rl["created"], "hits": rl["parent_hits"],
                       "n_match": None, "promoted_at": None,
                       "n_backfill": rl.get("parent_n_backfill", 0),
                       "ledger_counts": {}, "active": False,
                       "retired_at": rl["retired_at"],
                       "split_tags": rl["tags"]})
    return detail


# ---------------- R_L4B_NONSPLIT_EQ：非分裂数字 vs docs/253 Mode ON（§1.9 重测声明） ----------------
def nonsplit_compare(on, on_r0, on_r1):
    """Mode ON 非分裂数字（SC1_fast/SC2_fast/n_promo/n_recycle/ratio/MAE）与 docs/253
    Mode ON 逐位一致（匹配/升级/回收/预测路径未动）。注：分裂发生后的流，子条目进入
    慢层会改变后续匹配态 -> 非分裂数字可合法偏离（本格新机制产物），逐流如实报告。"""
    results = {}
    all_ok = True
    for sid in STREAM_ORDER:
        r = on[sid]
        exp = D253_ON[sid]
        ok = (r["sc1_fast"] == exp["sc1_fast"]
              and r["sc2_fast"] == exp["sc2_fast"]
              and r["n_promo"] == exp["n_promo"]
              and r["n_recycle"] == exp["n_recycle"]
              and abs(r["ratio"] - exp["ratio"]) < 1e-9
              and abs(r["mae_mean_win"] - exp["mae"]) < 1e-9)
        results[sid] = int(ok)
        all_ok = all_ok and ok
    for sid, r in (("R0", on_r0), ("R1", on_r1)):
        exp = D253_ON[sid]
        ok = (r["sc1_fast"] == exp["sc1_fast"]
              and r["sc2_fast"] == exp["sc2_fast"]
              and r["n_promo"] == exp["n_promo"]
              and r["n_recycle"] == exp["n_recycle"]
              and abs(r["ratio"] - exp["ratio"]) < 1e-9
              and abs(r["mae_mean_win"] - exp["mae"]) < 1e-9)
        results[sid] = int(ok)
        all_ok = all_ok and ok
    return int(all_ok), results


# ---------------- 构造冒烟（合成帧，非数据；R_L4B_SMOKE_*） ----------------
def _synth_frames(n_frames=30):
    frames = []
    for k in range(n_frames):
        f = np.zeros((120, 160), dtype=np.uint8)
        x0 = 20 + 2 * k
        f[40:60, x0:x0 + 20] = 255
        frames.append(f)
    return frames


def _lay(m, row0, col0, n):
    row, col = row0, col0
    for _ in range(n):
        m[row, col] = True
        col += 1
        if col >= 160:
            col = 0
            row += 1
    return m


def _mask_with(E, U, c2):
    """构造总事件 E、上组事件 U、c2 指定的窗口事件掩码（合成冒烟用）。"""
    m = np.zeros((120, 160), dtype=bool)
    lo_n = int(E - U)
    if c2 == 0:
        _lay(m, 10, 5, int(U))
        _lay(m, 80, 100, lo_n)
    else:
        _lay(m, 10, 100, int(U))
        _lay(m, 80, 5, lo_n)
    return m


def smoke_main():
    """构造冒烟（docs/254 §二 轮 2）：SlotLoop2 mode off/on 在 30 帧合成灰度上构造运行
    正常；出生回填语义核对（创建前邻域窗入账本、hits 不增）；分裂子条目 μ 重初始化核对。"""
    results = {}

    # 1. 构造/运行：30 帧合成帧 off/on 均正常；合成帧上 ON/OFF ratio 逐位一致
    frames = _synth_frames(30)
    off_out, _ = run_slot2_stream(frames, "off")
    on_out, _ = run_slot2_stream(frames, "on")
    results["construct_off"] = int(isinstance(off_out, dict)
                                   and off_out.get("n_windows", 0) >= 1)
    results["construct_on"] = int(isinstance(on_out, dict)
                                  and on_out.get("n_windows", 0) >= 1
                                  and "slot_coverage" in on_out
                                  and "backfill_diag" in on_out)
    results["repro_synth"] = int(abs(off_out["ratio"] - on_out["ratio"]) < 1e-9)

    # 2. 出生回填语义：快原型创建时，创建窗前邻域窗 (c2,E,U) 入账本；hits 不增
    loop = SlotLoop2(mode="on", window=WINDOW, **LOOP_CFG)
    loop._hist = [(0, (5.0, 4.4), 0, 147, 80),
                  (1, (5.05, 4.45), 1, 155, 85),
                  (2, (8.0, 6.0), None, 3000, 400)]
    loop._ev_win = _mask_with(151, 82, 0)
    loop._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop._on_window()
    p = loop.prototypes[0]
    ledger_ok = (p["ledger"].get(0) == [(151.0, 82.0), (147.0, 80.0)]
                 and p["ledger"].get(1) == [(155.0, 85.0)]
                 and None not in p["ledger"].get(None, []))
    results["backfill_ledger"] = int(ledger_ok and p["n_backfill"] == 2
                                     and len(p["ledger"][0]) == 2
                                     and len(p["ledger"][1]) == 1)
    results["hits_no_inc"] = int(p["hits"] == 1 and p["n_match"] == 1)
    results["bf_log"] = int(len(loop.bf_log) == 1
                            and loop.bf_log[0]["n_backfill"] == 2
                            and loop.bf_log[0]["source_wins"] == [0, 1]
                            and loop.n_creations_with_bf == 1
                            and loop.bf_window_total == 2
                            and loop.bf_c2_dist == {"None": 0, "0": 1, "1": 1})

    # 3. 分裂语义（加法③）：子条目 μ = tag 组账本窗口 (ln(1+E), ln(1+U)) 中位数向量
    loop2 = SlotLoop2(mode="on", window=WINDOW, **LOOP_CFG)
    loop2._win = 20
    loop2.prototypes = [dict(pid=7, mu=(5.0, 4.4), hits=6, created=10,
                             last_active=20, n_match=6, kind="slow",
                             promoted_at=15, tag=None, n_backfill=0,
                             ledger={0: [(147.0, 80.0), (148.0, 81.0),
                                         (146.0, 79.0)],
                                     1: [(3000.0, 400.0), (3200.0, 410.0),
                                         (3100.0, 405.0)]})]
    loop2._split_check()
    children = loop2.prototypes
    c0 = [c for c in children if c["tag"] == 0][0]
    c1 = [c for c in children if c["tag"] == 1][0]
    exp0 = (float(np.median([np.log1p(147.0), np.log1p(148.0), np.log1p(146.0)])),
            float(np.median([np.log1p(80.0), np.log1p(81.0), np.log1p(79.0)])))
    exp1 = (float(np.median([np.log1p(3000.0), np.log1p(3200.0), np.log1p(3100.0)])),
            float(np.median([np.log1p(400.0), np.log1p(410.0), np.log1p(405.0)])))
    results["split_mu"] = int(
        loop2.n_retired_slow == 1 and loop2.n_split == 2
        and len(loop2.retired_log) == 1 and loop2.retired_log[0]["pid"] == 7
        and c0["hits"] == 3 and c1["hits"] == 3
        and c0["n_match"] == 0 and c1["n_match"] == 0
        and abs(c0["mu"][0] - exp0[0]) < 1e-12 and abs(c0["mu"][1] - exp0[1]) < 1e-12
        and abs(c1["mu"][0] - exp1[0]) < 1e-12 and abs(c1["mu"][1] - exp1[1]) < 1e-12
        and c0["ledger"] == {0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}
        and c1["ledger"] == {1: [(3000.0, 400.0), (3200.0, 410.0), (3100.0, 405.0)]})

    for k in ("construct_off", "construct_on", "repro_synth",
              "backfill_ledger", "hits_no_inc", "bf_log", "split_mu"):
        print("R_L4B_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main()
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

    # ---- Mode OFF（DeferredLoop 逐字；守卫 R_L4B_GUARD_D251） ----
    off = {}
    for sid in STREAM_ORDER:
        out, _ = run_slot2_stream(stream_frames[sid], "off")
        out["stream_id"] = sid
        off[sid] = out
    off_r0, _ = run_slot2_stream(r0_frames, "off")
    off_r1, _ = run_slot2_stream(r1_frames, "off")
    off_r1["bridge"] = bridge_metrics(build_entry_base(off_r1), spans)
    off_r1["gist"] = gist_metrics(off_r1, switch_windows)
    t_off = time.time() - t0 - t_dec

    d251_items = guard_d251_items(off, off_r1)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- Mode ON（槽位路径 + 三条账本修复；判据口径） ----
    on = {}
    for sid, sname, vidx in STREAMS:
        out, loop = run_slot2_stream(stream_frames[sid], "on")
        out["stream_id"] = sid
        out["stream_name"] = sname
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        out["switch_diag"] = scene_switch_diag(stream_frames[sid], creations)
        on[sid] = out
    on_r0, on_r0_loop = run_slot2_stream(r0_frames, "on")
    on_r1, on_r1_loop = run_slot2_stream(r1_frames, "on")
    on_r1["bridge"] = bridge_metrics(build_entry_base(on_r1), spans)
    on_r1["gist"] = gist_metrics(on_r1, switch_windows)
    on_r1["seg_info"] = r1_segment_info(on_r1_loop, spans)
    on_r1["split_align"] = split_segment_align(on_r1_loop, spans, on_r1["seg_info"])
    t_on = time.time() - t0 - t_dec - t_off

    # ---- R_L4B_REPRO_RATIO（构造性控制项：ON vs OFF 全流 ratio，abs < 1e-9） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            int(abs(on[sid]["ratio"] - off[sid]["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0", int(abs(on_r0["ratio"] - off_r0["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R1", int(abs(on_r1["ratio"] - off_r1["ratio"]) < 1e-9)))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, v) for n, v in repro_items)

    # ---- R_L4B_GUARD_D246（SoftLoop 路径；docs/249/250/251 同一代码路径） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)
    guard246_passed = sum(1 for ch in guard246_detail.split(",") if ch.endswith(":1"))

    # ---- R_L4B_NONSPLIT_EQ（docs/254 §1.9：非分裂数字 vs docs/253 Mode ON） ----
    nonsplit_eq, nonsplit_per = nonsplit_compare(on, on_r0, on_r1)

    # ---- 判据（§1.6 冻结；与 docs/253 逐字一致；ON 数字） ----
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

    # ---- 判定（§1.7 冻结映射；第二格专属语义） ----
    guards_ok = d251_ok == 1 and guard246_ok == 1 and repro_ok == 1
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d -> implementation drift; fix implementation, "
                 "do not judge mechanism (see R_L4B_GUARD_*)" % (
                     d251_ok, d251_passed, guard246_ok, repro_ok))
    elif not crit1:
        verdict = "PARALLEL_ONLY_REAL"
        vnote = ("COMPOUND_EMERGES fails on R1 after reachability repair "
                 "(backfill + lifecycle ledger + child-mu reinit): n_split=%d, "
                 "SC2_tagged=%d, compound_frac=%.4f -> stronger negative result: "
                 "with ledger reachability restored and criteria identical to "
                 "docs/253, two-c2-group info distribution within r_slow "
                 "neighborhood is still insufficient; no threshold rollback" % (
                     on_r1["n_split"], on_r1["sc2_tagged"],
                     on_r1["compound_frac"]))
    elif not crit2:
        verdict = "BOUNDARY"
        vnote = ("COMPOUND_EMERGES passes but ADOPT_NONRANDOM fails: "
                 "spurious_split_frac=%.4f, avg_post_split_hits=%.4f, "
                 "R0 n_split=%d (reachability repair made splits fire but gated "
                 "conditional memory not maintained / negative control broken)" % (
                     on_r1["spurious_split_frac"],
                     on_r1["avg_post_split_hits"], on_r0["n_split"]))
    elif not (crit3 and crit4):
        why = []
        if not crit3:
            why.append("FOUNDATION_KEEP fails (ratio/sc2_slow/gist_cov; see numbers)")
        if not crit4:
            why.append("PROMOTION_KEEP fails (n_promo/n_recycle/hit-rate separation)")
        verdict = "PARTIAL_REAL"
        vnote = "; ".join(why) + " (see R_L4B_CRIT* numbers)"
    else:
        verdict = "COMPOSABLE_REAL"
        vnote = ("criteria 1-4 all pass and all guards pass: after ledger "
                 "reachability repair, compound structure emerges on real "
                 "streams (tagged conditional slow memory), adoption is "
                 "non-random (negative control clean), L3 foundation and "
                 "promotion behavior evidence kept")

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
                             "w_bf": W_BF,
                             "ctx_split_y": CTX_SPLIT_Y,
                             "slot_sparse_px": SLOT_SPARSE,
                             "participate": PARTICIPATE},
           "mechanism": ("SlotLoop2(SlotLoop): docs/253 slot path verbatim + three "
                         "ledger reachability repairs: (1) birth backfill (docs/235 "
                         "D6 verbatim + neighborhood constraint: fast prototype "
                         "creation backfills ledger from last <=W_BF=4 participating "
                         "windows with feature distance <= r_fast; backfill does NOT "
                         "add hits; ledger entries extended to (c2,E,U)), "
                         "(2) explicit upgrade ledger inheritance (full-lifecycle "
                         "ledger = split evaluation basis; promotion branch never "
                         "touches ledger), (3) child mu re-initialization (split "
                         "tagged child mu = median vector of (ln(1+E), ln(1+U)) "
                         "over its tag-group ledger windows); Mode OFF = "
                         "DeferredLoop verbatim; prediction path zero-change"),
           "loop": LOOP_CFG,
           "r1_switch_windows": switch_windows,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l4_compose_test2",
        "doc_ref": "docs/235, docs/243, docs/245, docs/246, docs/247, docs/248, "
                   "docs/249, docs/250, docs/251, docs/253, docs/254",
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
                   "repro_ratio": {"ok": repro_ok, "detail": repro_detail},
                   "nonsplit_eq": {"ok": nonsplit_eq, "per_stream": nonsplit_per},
                   "bf_hash": {sid: on[sid]["backfill_diag"]["bf_hash_all"]
                               for sid in STREAM_ORDER}
                              | {"R0": on_r0["backfill_diag"]["bf_hash_all"],
                                 "R1": on_r1["backfill_diag"]["bf_hash_all"]}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2),
                   "decode_sec": round(t_dec, 2),
                   "off_sec": round(t_off, 2), "on_sec": round(t_on, 2)},
    }
    res_path = os.path.join(args.out_dir, "l4b_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L4B_TAG=%s" % args.tag)
    print("R_L4B_W_BF=%d" % W_BF)
    print("R_L4B_R_SLOW=%.6f" % R_SLOW)
    print("R_L4B_R_FAST=%.6f" % R_FAST)
    print("R_L4B_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_L4B_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_L4B_K_PROMOTE=%d" % K_PROMOTE)
    print("R_L4B_K_DECAY=%d" % K_DECAY)
    print("R_L4B_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_L4B_ALPHA=%.4f" % ALPHA)
    print("R_L4B_K_SPLIT=%d" % K_SPLIT)
    print("R_L4B_DELTA_REL=%.4f" % DELTA_REL)
    print("R_L4B_K_LEDGER=%d" % K_LEDGER)
    print("R_L4B_CTX_SPLIT_Y=%.1f" % CTX_SPLIT_Y)
    for j, sid in enumerate(STREAM_ORDER):
        r = off[sid]
        print("R_L4B_OFF_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4B_OFF_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4B_OFF_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4B_OFF_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4B_OFF_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4B_OFF_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
    print("R_L4B_OFF_R1_RATIO=%.6f" % off_r1["ratio"])
    print("R_L4B_OFF_R1_SC1_FAST=%d" % off_r1["sc1_fast"])
    print("R_L4B_OFF_R1_SC2_SLOW=%d" % off_r1["sc2_slow"])
    print("R_L4B_OFF_R1_N_PROMO=%d" % off_r1["n_promo"])
    print("R_L4B_OFF_R1_N_RECYCLE=%d" % off_r1["n_recycle"])
    print("R_L4B_OFF_R1_CHURN_SLOW=%.4f" % off_r1["churn_slow"])
    print("R_L4B_OFF_R1_GIST_COV=%.4f" % off_r1["gist"]["cov"])
    print("R_L4B_OFF_R1_BRIDGE_SW=%.4f" % off_r1["bridge"]["bridge_corr_switch"])
    print("R_L4B_OFF_R0_RATIO=%.6f" % off_r0["ratio"])
    print("R_L4B_OFF_R0_SC1_FAST=%d" % off_r0["sc1_fast"])
    print("R_L4B_OFF_R0_SC2_SLOW=%d" % off_r0["sc2_slow"])
    print("R_L4B_OFF_R0_CHURN_SLOW=%.4f" % off_r0["churn_slow"])
    print("R_L4B_OFF_R0_N_PROMO=%d" % off_r0["n_promo"])
    print("R_L4B_OFF_R0_N_RECYCLE=%d" % off_r0["n_recycle"])
    print("R_L4B_GUARD_D251=%d" % d251_ok)
    print("R_L4B_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L4B_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L4B_GUARD_D251_DETAIL=%s" % d251_detail)
    for sid in ALL_STREAMS:
        if sid in on:
            r = on[sid]
        elif sid == "R0":
            r = on_r0
        else:
            r = on_r1
        sc = r["slot_coverage"]
        bf = r["backfill_diag"]
        d = r.get("switch_diag")
        sw = "NA" if d is None or d["switch_corr"] is None else \
            "%.4f" % d["switch_corr"]
        spb = ",".join(str(x["parent_n_backfill"]) for x in bf["split_parent_bf"])
        if not bf["split_parent_bf"]:
            spb = "NA"
        print("R_L4B_ON_%s_FRAMES=%d" % (sid, r["frames"]))
        print("R_L4B_ON_%s_WINDOWS=%d" % (sid, r["n_windows"]))
        print("R_L4B_ON_%s_VALID=%d" % (sid, r["n_valid"]))
        print("R_L4B_ON_%s_MAE=%.6f" % (sid, r["mae_mean_win"]))
        print("R_L4B_ON_%s_MAE_SD=%.6f" % (sid, r["mae_sd_win"]))
        print("R_L4B_ON_%s_MAE_LO=%.6f" % (sid, r["mae_ci95"][0]))
        print("R_L4B_ON_%s_MAE_HI=%.6f" % (sid, r["mae_ci95"][1]))
        print("R_L4B_ON_%s_Q1=%.6f" % (sid, r["mae_q1"]))
        print("R_L4B_ON_%s_Q4=%.6f" % (sid, r["mae_q4"]))
        print("R_L4B_ON_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4B_ON_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4B_ON_%s_SC2_FAST=%d" % (sid, r["sc2_fast"]))
        print("R_L4B_ON_%s_SC1_SLOW=%d" % (sid, r["sc1_slow"]))
        print("R_L4B_ON_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4B_ON_%s_SC2_TAGGED=%d" % (sid, r["sc2_tagged"]))
        print("R_L4B_ON_%s_COMPOUND_FRAC=%.4f" % (sid, r["compound_frac"]))
        print("R_L4B_ON_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
        print("R_L4B_ON_%s_CHURN_LEGACY=%.4f" % (sid, r["churn_legacy"]))
        print("R_L4B_ON_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4B_ON_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4B_ON_%s_N_SPLIT=%d" % (sid, r["n_split"]))
        print("R_L4B_ON_%s_N_RETIRED_SLOW=%d" % (sid, r["n_retired_slow"]))
        print("R_L4B_ON_%s_SPURIOUS_SPLIT_FRAC=%.4f" % (sid, r["spurious_split_frac"]))
        print("R_L4B_ON_%s_AVG_POST_SPLIT_HITS=%.4f" % (sid, r["avg_post_split_hits"]))
        print("R_L4B_ON_%s_PROMO_MEAN=%.4f" % (sid, r["promoted_mean_hits"]))
        print("R_L4B_ON_%s_NONPROMO_MEAN=%.4f" % (sid, r["nonpromoted_mean_hits"]))
        print("R_L4B_ON_%s_C2HASH=%s" % (sid, r["c2_hash"]))
        print("R_L4B_ON_%s_C2_COV=%.4f" % (sid, sc["coverage"]))
        print("R_L4B_ON_%s_C2_F0_NON=%.4f" % (sid, sc["frac0_non"]))
        print("R_L4B_ON_%s_C2_F1_NON=%.4f" % (sid, sc["frac1_non"]))
        print("R_L4B_ON_%s_C2_F0_ALL=%.4f" % (sid, sc["frac0_all"]))
        print("R_L4B_ON_%s_C2_F1_ALL=%.4f" % (sid, sc["frac1_all"]))
        print("R_L4B_ON_%s_TAG0=%d" % (sid, r["tag_dist"]["tag0"]))
        print("R_L4B_ON_%s_TAG1=%d" % (sid, r["tag_dist"]["tag1"]))
        print("R_L4B_ON_%s_SW_CORR=%s" % (sid, sw))
        print("R_L4B_BF_%s_N_CREATIONS=%d" % (sid, bf["n_creations"]))
        print("R_L4B_BF_%s_N_CREATIONS_WITH_BF=%d" % (sid, bf["n_creations_with_bf"]))
        print("R_L4B_BF_%s_WINDOW_TOTAL=%d" % (sid, bf["bf_window_total"]))
        print("R_L4B_BF_%s_AVG_PER_CREATION=%.4f" % (sid, bf["bf_avg_per_creation"]))
        print("R_L4B_BF_%s_AVG_PER_BF_CREATION=%.4f" % (sid, bf["bf_avg_per_bf_creation"]))
        print("R_L4B_BF_%s_C2_NONE=%d" % (sid, bf["bf_c2_dist"]["None"]))
        print("R_L4B_BF_%s_C2_0=%d" % (sid, bf["bf_c2_dist"]["0"]))
        print("R_L4B_BF_%s_C2_1=%d" % (sid, bf["bf_c2_dist"]["1"]))
        print("R_L4B_BF_%s_SPLIT_PARENT=%s" % (sid, spb))
        print("R_L4B_BF_%s_BFHASH=%s" % (sid, bf["bf_hash_all"]))
    print("R_L4B_ON_R1_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    seg = on_r1["seg_info"]
    print("R_L4B_ON_R1_SEG_INFO=%s" % ",".join(
        ("NA" if row["ratio"] is None else "%.4f" % row["ratio"]) for row in seg))
    print("R_L4B_ON_R1_SEG_N0=%s" % ",".join(str(row["n0"]) for row in seg))
    print("R_L4B_ON_R1_SEG_N1=%s" % ",".join(str(row["n1"]) for row in seg))
    print("R_L4B_ON_R1_ALIGN_RATE=%.4f" % on_r1["split_align"]["align_rate"])
    print("R_L4B_ON_R1_N_ALIGNED=%d" % on_r1["split_align"]["n_aligned"])
    print("R_L4B_REPRO_RATIO=%d" % repro_ok)
    print("R_L4B_REPRO_DETAIL=%s" % repro_detail)
    print("R_L4B_NONSPLIT_EQ=%d" % nonsplit_eq)
    for sid in ALL_STREAMS:
        print("R_L4B_NONSPLIT_%s=%d" % (sid, nonsplit_per[sid]))
    print("R_L4B_CRIT1_COMPOUND_EMERGES=%d" % crit1)
    print("R_L4B_CRIT1_N_SPLIT=%d" % on_r1["n_split"])
    print("R_L4B_CRIT1_SC2_TAGGED=%d" % on_r1["sc2_tagged"])
    print("R_L4B_CRIT1_COMPOUND_FRAC=%.4f" % on_r1["compound_frac"])
    print("R_L4B_CRIT2_ADOPT_NONRANDOM=%d" % crit2)
    print("R_L4B_CRIT2_SPURIOUS_FRAC=%.4f" % on_r1["spurious_split_frac"])
    print("R_L4B_CRIT2_AVG_POST_HITS=%.4f" % on_r1["avg_post_split_hits"])
    print("R_L4B_CRIT2_R0_N_SPLIT=%d" % on_r0["n_split"])
    print("R_L4B_CRIT3_FOUNDATION_KEEP=%d" % crit3)
    print("R_L4B_CRIT3_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    print("R_L4B_CRIT4_PROMOTION_KEEP=%d" % crit4)
    print("R_L4B_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L4B_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L4B_GUARD_D246=%d" % guard246_ok)
    print("R_L4B_GUARD_D246_ITEMS=%d" % 12)
    print("R_L4B_GUARD_D246_PASSED=%d" % guard246_passed)
    print("R_L4B_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L4B_VERDICT=%s" % verdict)
    print("R_L4B_VERDICT_NOTE=%s" % vnote)
    print("R_L4B_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
