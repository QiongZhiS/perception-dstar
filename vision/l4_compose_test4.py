"""vision/l4_compose_test4.py — docs/262 L4 组合泛化第四格：群级锚合并（SlotLoop4）。

SlotLoop4(SlotLoop3)（docs/262 §一 冻结，运行后不改）：
  Mode OFF = super()._on_window() 链（= SlotLoop3 OFF = SlotLoop2 OFF = DeferredLoop
             逐字 -> R_L4D_GUARD_D251 复现 docs/251 §3.3/§3.4，32 项，容差 1e-4）；
  Mode ON  = docs/256 SlotLoop3 群级路径逐字（c2 可观测 + 账本 + 出生回填 + 升级并账 +
             子条目 μ 重初始化 + 单原型分裂 + 门控匹配 + 群级 G1-G4 + 群级分裂）
             **+ 本格锚物化加法**（全部只落在模式表路径，预测路径零改动）：
             群级检查顺序 = G1 -> G2 -> G4（前置）-> 锚物化（本格新增）-> G3 -> 群级
             分裂。锚物化：对每个无已确认锚的 c2 侧 v（纪元内 v 侧窗口数 >= k_g_ledger
             =3 由 G1 保证），把纪元内分散在该侧的全部窗口物化为一个未打标已确认慢锚
             （μ = 该侧窗口 (ln(1+E), ln(1+U)) 中位数向量——SlotLoop2 ③ 同款规则、
             ledger = {v: 该侧全部窗口}、hits = 并集计数（>= k_ledger -> 出生即确认）、
             n_match=0、created = 当前窗（纪元内））；G3 以"自然锚 + 物化锚"评估
             （冻结文本逐字：每侧 >= k_g_confirm=1 个已确认慢原型锚（创建/匹配于
             epoch 内、账本 >= k_ledger 窗））；G1-G4 全满足 -> 群级分裂按 docs/256
             §1.3-4 逐字（纪元内全部 v 侧锚——自然 + 物化——退休 -> 每侧建打标慢原型）。
   每参与窗口冻结执行顺序：(1) 匹配/创建（SlotLoop2 逐字）-> (2) 回收 -> (3) 单原型
   _split_check（逐字）-> (4) 群级检查（G1 -> G2 -> G4 -> 锚物化 -> G3 -> 群级分裂）
   -> (5) _hist + epoch_hist 维护。

负对照（docs/256 §1.3-5 + docs/261 §2.1 审视，逐字沿用）：判据 2 负对照流 =
R0b=bear×5（单场景循环 + c2 单侧布局 -> G1 构造性失败 -> 无物化无触发 -> n_split(R0b)=0
由机制保证）；R0=flamingo×5 降级为诊断流（15/85 恒定倾斜 -> G1 失败；如实报告，不进
判据）。物化以 G1 为前提：R0b/R0 的 G1 均结构性失败 -> 物化/群级分裂在两者上均不触发。

度量（§1.3 冻结）：M1-M6 与 docs/256 逐字一致 + M7 物化诊断（每流 n_consolidated、
consolidate_log（窗/侧/来源窗口/μ/账本数）、R_L4D_CONSOLIDATE_HASH）——诊断级，
不进判据。

判据（§1.6 冻结，与 docs/253/254/256 逐条一致）：①[L4][机制][组合测试]
COMPOUND_EMERGES（R1：n_split>=1 且 SC2_tagged>=1 且 compound_frac>=0.5）；
②[L4][机制][行为证据] ADOPT_NONRANDOM（R1：spurious_split_frac<=0.5 且
avg_post_split_hits>=1；**R0b**：n_split==0）；③[L4][机制] FOUNDATION_KEEP
（R1+S1-S4：ratio<=1.5 且 SC2_slow>0；R1：gist_cov>=0.5）；④[L4][机制][行为证据]
PROMOTION_KEEP（全局 n_promo>0 且 n_recycle>0 且升级命中率均值>未升级均值）。
判定映射按 §1.6（第四格专属语义：物化后 R1 仍 n_split=0 或 compound<0.5 = L4 收束
为负证据；判据 1 过但判据 2 不过 = BOUNDARY 物化子条目门控未维持）。

守卫（§1.7 冻结，不进判据）：R_L4D_GUARD_D251（Mode OFF 复现 docs/251，32 项）、
R_L4D_GUARD_D246（run_guard_quota + guard_vs_d246，12/12）、R_L4D_REPRO_RATIO
（ON vs OFF 全流 ratio abs<1e-9；冻结 6/6 = S1-S4+R0+R1，R0b 增量报告）、
R_L4D_GROUP_HASH（群级日志确定性指纹，含物化锚，两轮逐位一致）、
R_L4D_CONSOLIDATE_HASH（物化日志确定性指纹，两轮逐位一致）、R_L4D_R0B_NOSPLIT
（n_split(R0b)==0 构造性复检）、R_L4D_NONSPLIT_EQ（无物化无分裂流非分裂数字 vs
docs/253 逐位一致，分裂/物化流如实标注偏离）。

安全纪律（§1.10 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L4D_* 摘要块；
运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/l4_compose_test4.py --smoke        # 构造冒烟（合成帧，非数据）
  python vision/l4_compose_test4.py --tag timing
  python vision/l4_compose_test4.py --tag main
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
# import 复用 l4_compose_test3（docs/262 §1.10 冻结清单：SlotLoop3/G_WIN/K_G_CONFIRM/
# K_G_LEDGER/run_slot3_stream/group_split_align/nonsplit_compare3 为本格基座；
# proto_detail2/_synth_frames/_lay/_mask_with/W_BF/D253_ON 为同文件复用）
from l4_compose_test3 import (SlotLoop3, G_WIN, K_G_CONFIRM, K_G_LEDGER,
                              ALL_STREAMS, run_slot3_stream, group_split_align,
                              nonsplit_compare3, proto_detail2, _synth_frames,
                              _lay, _mask_with, W_BF, D253_ON)
# import 复用 l4_compose_test（docs/262 §1.10 冻结清单）
from l4_compose_test import (SlotLoop, K_SPLIT, DELTA_REL, K_LEDGER,
                             SLOT_SPARSE, PARTICIPATE, _slot_c2, _c2_hash,
                             r1_segment_info, split_segment_align,
                             guard_d251_items, STREAM_ORDER)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 本格全部旋钮 = docs/253/254/256 冻结值（§1.5 冻结，零重调；物化机制无新数值旋钮）
# G_WIN / K_G_CONFIRM / K_G_LEDGER 从 l4_compose_test3 复用（G_WIN=8 / k_g_confirm=1 /
# k_g_ledger=k_ledger=3）


# ---------------- SlotLoop4（docs/262 §1.2 冻结：SlotLoop3 逐字 + 锚物化加法） ----------------
class SlotLoop4(SlotLoop3):
    """SlotLoop3（docs/256 逐字）+ docs/262 §1.2 冻结锚物化加法：
    群级检查顺序 = G1 -> G2 -> G4（前置）-> 锚物化（本格新增）-> G3（自然锚 + 物化锚）
    -> 群级分裂。物化：对无已确认锚的 c2 侧 v（纪元内 v 侧窗口数 >= k_g_ledger 由 G1
    保证），把纪元内该侧分散窗口物化为一个未打标已确认慢锚（μ = 该侧窗口
    (ln(1+E), ln(1+U)) 中位数向量、ledger = {v: 该侧全部窗口}、hits = 并集计数
    （>= k_ledger -> 出生即确认）、created = 当前窗）。Mode OFF = super()._on_window()
    （= SlotLoop3 OFF = DeferredLoop 逐字）。"""

    def __init__(self, mode="off", k_split=K_SPLIT, delta_rel=DELTA_REL,
                 k_ledger=K_LEDGER, w_bf=W_BF, g_win=G_WIN,
                 k_g_confirm=K_G_CONFIRM, **kw):
        self.consolidate_log = []        # 物化记录 {pid, side, at, source_wins, n, mu0, mu1}
        self.n_consolidated = 0          # 物化锚总数（M7）
        self.n_consolidated_by_side = {0: 0, 1: 0}
        super().__init__(mode=mode, k_split=k_split, delta_rel=delta_rel,
                         k_ledger=k_ledger, w_bf=w_bf, g_win=g_win,
                         k_g_confirm=k_g_confirm, **kw)

    # ---- 物化日志指纹（R_L4D_CONSOLIDATE_HASH 口径；确定性纯函数） ----
    def _consolidate_log_hash(self):
        s = ";".join("%d@%d@%d@[%s]@%d@%.6f@%.6f" % (
            c["pid"], c["side"], c["at"],
            ",".join(str(w) for w in c["source_wins"]),
            c["n"], c["mu0"], c["mu1"]) for c in self.consolidate_log)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    # ---- 群级日志指纹（覆盖 SlotLoop3：追加 consolidated0/consolidated1 字段） ----
    def _group_log_hash(self):
        parts = []
        for e in self.group_log:
            ch = "|".join("%d:%d:%d" % (c[0], c[1], c[2])
                          for c in e.get("children", []))
            parts.append(
                "%d@%d@%d@%d@%d@%d@[%s]@%d@%d@%.4f@%.4f@%.4f@[%s]@[%s]@[%s]@[%s]@%s" % (
                    e["win"], e["g1"], e["g2"], e["g3"], e["g4"], e["trigger"],
                    ",".join(str(w) for w in e["epoch_wins"]),
                    e["n0"], e["n1"], e["med0"], e["med1"], e["ratio"],
                    ",".join(str(p) for p in e["anchors0"]),
                    ",".join(str(p) for p in e["anchors1"]),
                    ",".join(str(p) for p in e["consolidated0"]),
                    ",".join(str(p) for p in e["consolidated1"]),
                    ch))
        return hashlib.md5(";".join(parts).encode("utf-8")).hexdigest()

    # ---- 群级检查（docs/262 §1.2 冻结：G1 -> G2 -> G4 -> 锚物化 -> G3 -> 分裂） ----
    def _group_check(self, c2, E, U, matched_pid):
        epoch = list(self.epoch_hist) + [(self._win, c2, E, U, matched_pid)]
        if len(epoch) > self.g_win:
            epoch = epoch[-self.g_win:]
        ep_wins = [w for (w, _, _, _, _) in epoch]
        ep_min = min(ep_wins)
        ep_max = max(ep_wins)
        matched_pids = set(pid for (_, _, _, _, pid) in epoch
                           if pid is not None and pid >= 0)

        # G1：两 c2 组群级覆盖（docs/256 §1.3-4 逐字）
        n0 = sum(1 for (_, c, _, _, _) in epoch if c == 0)
        n1 = sum(1 for (_, c, _, _, _) in epoch if c == 1)
        g1 = int(n0 >= K_G_LEDGER and n1 >= K_G_LEDGER)

        # G2：群级中位事件能量比（docs/256 §1.3-4 逐字）
        e0 = [float(Ev) for (_, c, Ev, _, _) in epoch if c == 0]
        e1 = [float(Ev) for (_, c, Ev, _, _) in epoch if c == 1]
        med0 = float(np.median(e0)) if e0 else 0.0
        med1 = float(np.median(e1)) if e1 else 0.0
        ratio = (max(med0, med1) / min(med0, med1)
                 if min(med0, med1) > 0 else 0.0)
        g2 = int(ratio >= (1 + self.delta_rel))

        # G4（本格前置评估；物化锚未打标不影响 G4；防通胀语义不变）
        g4 = 1
        for p in self.prototypes:
            if p["kind"] != "slow" or p.get("tag") is None:
                continue
            if p["pid"] in matched_pids or (ep_min <= p["created"] <= ep_max):
                g4 = 0
                break

        # 自然锚（docs/256 G3 条件逐字）
        anchors = {0: [], 1: []}      # v -> [prototype dict]
        for p in self.prototypes:
            if p["kind"] != "slow":
                continue
            if p.get("tag") is not None:
                continue
            if p["hits"] < self.hits_min_slow:
                continue
            if p["pid"] not in matched_pids \
                    and not (ep_min <= p["created"] <= ep_max):
                continue
            led = p.get("ledger", {})
            n_v = {v: len(led.get(v, [])) for v in (0, 1)}
            if n_v[0] >= self.k_ledger and n_v[1] >= self.k_ledger:
                continue            # 双侧锚 -> 单原型分裂领域，不由群级处理
            for v in (0, 1):
                if n_v[v] >= self.k_ledger:
                    anchors[v].append(p)

        # ---- 锚物化（docs/262 §1.2 冻结，本格新增）：仅当 G1∧G2∧G4（组织就绪）----
        consolidated = {0: [], 1: []}
        if g1 and g2 and g4:
            for v in (0, 1):
                nv = n0 if v == 0 else n1
                if nv >= K_G_LEDGER and not anchors[v]:
                    entries = [(float(Ev), float(Uv))
                               for (_, cv, Ev, Uv, _) in epoch if cv == v]
                    feats = np.asarray([[float(np.log1p(e)), float(np.log1p(u))]
                                        for e, u in entries], dtype=float)
                    mu = (float(np.median(feats[:, 0])),
                          float(np.median(feats[:, 1])))
                    pid = self._next_pid
                    self._next_pid += 1
                    anchor = dict(pid=pid, mu=mu, hits=len(entries),
                                  created=self._win, last_active=self._win,
                                  n_match=0, kind="slow",
                                  promoted_at=self._win, tag=None,
                                  n_backfill=0, ledger={v: list(entries)},
                                  consolidated=True)
                    self.prototypes.append(anchor)
                    self.n_consolidated += 1
                    self.n_consolidated_by_side[v] += 1
                    src_wins = [w for (w, cv, _, _, _) in epoch if cv == v]
                    self.consolidate_log.append(
                        dict(pid=pid, side=v, at=self._win,
                             source_wins=list(src_wins), n=len(entries),
                             mu0=round(mu[0], 6), mu1=round(mu[1], 6)))
                    anchors[v].append(anchor)
                    consolidated[v].append(pid)

        # G3：每侧已确认慢原型锚（自然锚 + 物化锚；冻结文本逐字）
        g3 = int(len(anchors[0]) >= self.k_g_confirm
                 and len(anchors[1]) >= self.k_g_confirm)

        trigger = int(g1 and g2 and g3 and g4)
        if g1:
            self.n_g1 += 1
        if g1 and g2:
            self.n_g1g2 += 1
        if g1 and g2 and g3:
            self.n_g1g2g3 += 1

        entry = {"win": self._win, "g1": g1, "g2": g2, "g3": g3, "g4": g4,
                 "trigger": trigger, "epoch_wins": list(ep_wins),
                 "n0": n0, "n1": n1, "med0": round(med0, 4),
                 "med1": round(med1, 4), "ratio": round(ratio, 4),
                 "anchors0": [a["pid"] for a in anchors[0]],
                 "anchors1": [a["pid"] for a in anchors[1]],
                 "consolidated0": list(consolidated[0]),
                 "consolidated1": list(consolidated[1]), "children": []}
        self.group_log.append(entry)

        if not trigger:
            return
        # ---- 群级分裂（docs/256 §1.3-4 逐字；锚 = 自然 + 物化）----
        self.n_group_splits += 1
        ev = {"split_at": self._win, "epoch_wins": list(ep_wins),
              "epoch_n0": n0, "epoch_n1": n1,
              "med0": round(med0, 4), "med1": round(med1, 4),
              "ratio": round(ratio, 4),
              "anchor_pids": {"0": [a["pid"] for a in anchors[0]],
                              "1": [a["pid"] for a in anchors[1]]},
              "consolidated_pids": {"0": list(consolidated[0]),
                                    "1": list(consolidated[1])},
              "children": []}
        self.group_split_events.append(ev)
        # 退休（v 侧锚；跨侧去重——双侧锚已被 G3 排除，但防御性去重保持确定性）
        to_retire = []
        seen = set()
        for v in (0, 1):
            for a in anchors[v]:
                if a["pid"] not in seen:
                    seen.add(a["pid"])
                    to_retire.append((v, a))
        for v, a in to_retire:
            self.prototypes.remove(a)
            self.n_retired_slow += 1
            self.retired_log.append(dict(pid=a["pid"], created=a["created"],
                                         retired_at=self._win,
                                         parent_hits=a["hits"],
                                         parent_n_backfill=a.get("n_backfill", 0),
                                         tags=[v], source="group",
                                         consolidated=a.get("consolidated", 0)))
        # 每侧建打标慢原型（arity-3 形态；μ = 退休 v 侧窗口特征中位数向量）
        for v in (0, 1):
            entries = []
            for a in anchors[v]:
                entries.extend(list(a.get("ledger", {}).get(v, [])))
            feats = np.asarray([[float(np.log1p(e)), float(np.log1p(u))]
                                for e, u in entries], dtype=float)
            mu = (float(np.median(feats[:, 0])),
                  float(np.median(feats[:, 1])))
            birth = len(entries)
            cpid = self._next_pid
            self._next_pid += 1
            self.prototypes.append(dict(
                pid=cpid, mu=mu, hits=birth,
                created=self._win, last_active=self._win, n_match=0,
                kind="slow", promoted_at=self._win, tag=v,
                n_backfill=0, ledger={v: list(entries)}))
            self.n_split += 1
            self.n_group_children += 1
            self.split_log.append(dict(pid=cpid, parent_pid=None,
                                       split_at=self._win, tag=v,
                                       birth_hits=birth, source="group"))
            entry["children"].append([cpid, v, birth])
            ev["children"].append([cpid, v, birth])

    # ---- Mode ON 主体 = SlotLoop3 逐字（群级检查已由本类 _group_check 覆盖）----
    # SlotLoop3._on_window 调用 self._group_check（多态）-> 本类物化逻辑生效；
    # Mode OFF = SlotLoop3 OFF = DeferredLoop 逐字（继承不变）。

    def finalize(self, n_windows, labels=None):
        out = super().finalize(n_windows, labels=labels)
        if self.mode != "on":
            return out
        # ---- M7 物化诊断（§1.3 冻结，诊断级，不进判据）----
        gd = out["group_diag"]
        gd["n_consolidated"] = self.n_consolidated
        gd["n_consolidated_by_side"] = {str(k): v
                                        for k, v in self.n_consolidated_by_side.items()}
        gd["consolidate_log"] = list(self.consolidate_log)
        gd["consolidate_hash"] = self._consolidate_log_hash()
        return out


# ---------------- 单流运行（与 run_slot3_stream 同构；Mode OFF 逐位一致） ----------------
def run_slot4_stream(frames, mode):
    loop = SlotLoop4(mode=mode, window=WINDOW, **LOOP_CFG)
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


# ---------------- 构造冒烟（合成帧，非数据；R_L4D_SMOKE_*） ----------------
def smoke_main():
    """构造冒烟（docs/262 §二 轮 2）：SlotLoop4 mode off/on 在 30 帧合成灰度上构造运行
    正常；锚物化语义核对——(A) 自然锚触发（两侧已有自然锚 -> 无物化，与 docs/256 逐字
    一致）；(B) 物化触发（0 侧分散窗口无自然锚 -> 物化合并锚 -> 群级分裂，物化锚进入
    退休集、子条目 μ/账本来自物化锚）；(C) 单侧纪元不触发（G1 构造性失败 -> 无物化）；
    (D) G2 门（能量同质 -> 无物化无触发）；(E) G4 防通胀（纪元已有打标原型 -> 无物化
    无触发）；(F) 合成帧上 ON/OFF ratio 逐位一致。"""
    results = {}

    # 1. 构造/运行：30 帧合成帧 off/on 均正常；合成帧上 ON/OFF ratio 逐位一致
    frames = _synth_frames(30)
    off_out, _ = run_slot4_stream(frames, "off")
    on_out, _ = run_slot4_stream(frames, "on")
    results["construct_off"] = int(isinstance(off_out, dict)
                                   and off_out.get("n_windows", 0) >= 1)
    results["construct_on"] = int(isinstance(on_out, dict)
                                  and on_out.get("n_windows", 0) >= 1
                                  and "slot_coverage" in on_out
                                  and "backfill_diag" in on_out
                                  and "group_diag" in on_out
                                  and "consolidate_hash" in on_out["group_diag"])
    results["repro_synth"] = int(abs(off_out["ratio"] - on_out["ratio"]) < 1e-9)

    # 2. (A) 自然锚触发（无物化）：两侧各有自然锚（P10/P11）-> 与 docs/256 群级分裂
    #    语义逐字一致（当前窗 c2=0 匹配 P10 -> P10 账本 4 条）
    loop = SlotLoop4(mode="on", window=WINDOW, **LOOP_CFG)
    loop._win = 7
    loop._next_pid = 12
    loop.epoch_hist = [(0, 0, 147, 80, 10), (1, 0, 148, 81, 10), (2, 0, 146, 79, 10),
                       (3, 1, 3000, 400, 11), (4, 1, 3200, 410, 11),
                       (5, 1, 3100, 405, 11), (6, 1, 3050, 402, 11)]
    loop.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=5, created=0, last_active=6, n_match=5,
             kind="slow", promoted_at=3, tag=None, n_backfill=2,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
        dict(pid=11, mu=(8.0, 6.0), hits=5, created=2, last_active=6, n_match=5,
             kind="slow", promoted_at=4, tag=None, n_backfill=0,
             ledger={1: [(3000.0, 400.0), (3200.0, 410.0), (3100.0, 405.0)]}),
    ]
    loop._ev_win = _mask_with(151, 82, 0)
    loop._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop._on_window()
    children = [p for p in loop.prototypes if p.get("tag") is not None]
    c0 = [c for c in children if c["tag"] == 0][0]
    c1 = [c for c in children if c["tag"] == 1][0]
    exp0 = (float(np.median([np.log1p(147.0), np.log1p(148.0),
                             np.log1p(146.0), np.log1p(151.0)])),
            float(np.median([np.log1p(80.0), np.log1p(81.0),
                             np.log1p(79.0), np.log1p(82.0)])))
    exp1 = (float(np.median([np.log1p(3000.0), np.log1p(3200.0),
                             np.log1p(3100.0)])),
            float(np.median([np.log1p(400.0), np.log1p(410.0),
                             np.log1p(405.0)])))
    results["natural_fire_no_consolidate"] = int(
        loop.n_group_splits == 1 and loop.n_group_children == 2
        and loop.n_split == 2 and loop.n_retired_slow == 2
        and loop.n_consolidated == 0
        and len(loop.retired_log) == 2
        and all(rl.get("source") == "group" for rl in loop.retired_log)
        and len(loop.split_log) == 2
        and all(sl.get("source") == "group" for sl in loop.split_log)
        and len(loop.group_split_events) == 1
        and loop.group_log[-1]["trigger"] == 1
        and loop.group_log[-1]["consolidated0"] == []
        and loop.group_log[-1]["consolidated1"] == []
        and c0["hits"] == 4 and c1["hits"] == 3
        and abs(c0["mu"][0] - exp0[0]) < 1e-12 and abs(c0["mu"][1] - exp0[1]) < 1e-12
        and abs(c1["mu"][0] - exp1[0]) < 1e-12 and abs(c1["mu"][1] - exp1[1]) < 1e-12
        and len(c0["ledger"][0]) == 4 and len(c1["ledger"][1]) == 3)

    # 3. (B) 物化触发：0 侧 3 窗分散（由三个快原型 P10/P12/P13 各自匹配，均未确认）
    #    -> 无自然锚 -> 物化 0 侧合并锚（纪元内 c2=0 窗口并集）-> 群级分裂
    loop = SlotLoop4(mode="on", window=WINDOW, **LOOP_CFG)
    loop._win = 7
    loop._next_pid = 20
    loop.epoch_hist = [(0, 0, 147, 80, 10), (1, 0, 160, 82, 12), (2, 0, 155, 81, 13),
                       (3, 1, 3000, 400, 11), (4, 1, 3200, 410, 11),
                       (5, 1, 3100, 405, 11), (6, 1, 3050, 402, 11)]
    loop.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=1, created=0, last_active=0, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(147.0, 80.0)]}),
        dict(pid=12, mu=(5.1, 4.5), hits=1, created=1, last_active=1, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(160.0, 82.0)]}),
        dict(pid=13, mu=(5.05, 4.45), hits=1, created=2, last_active=2, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(155.0, 81.0)]}),
        dict(pid=11, mu=(8.0, 6.0), hits=5, created=3, last_active=6, n_match=5,
             kind="slow", promoted_at=4, tag=None, n_backfill=0,
             ledger={1: [(3000.0, 400.0), (3200.0, 410.0),
                         (3100.0, 405.0), (3050.0, 402.0)]}),
    ]
    loop._ev_win = _mask_with(151, 82, 0)
    loop._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop._on_window()
    children = [p for p in loop.prototypes if p.get("tag") is not None]
    c0 = [c for c in children if c["tag"] == 0][0]
    c1 = [c for c in children if c["tag"] == 1][0]
    side0 = [(147.0, 80.0), (160.0, 82.0), (155.0, 81.0), (151.0, 82.0)]
    side1 = [(3000.0, 400.0), (3200.0, 410.0), (3100.0, 405.0), (3050.0, 402.0)]
    exp0 = (float(np.median([np.log1p(e) for e, _ in side0])),
            float(np.median([np.log1p(u) for _, u in side0])))
    exp1 = (float(np.median([np.log1p(e) for e, _ in side1])),
            float(np.median([np.log1p(u) for _, u in side1])))
    results["consolidate_fire"] = int(
        loop.n_consolidated == 1
        and loop.n_consolidated_by_side == {0: 1, 1: 0}
        and len(loop.consolidate_log) == 1
        and loop.consolidate_log[0]["side"] == 0
        and loop.consolidate_log[0]["at"] == 7
        and loop.consolidate_log[0]["source_wins"] == [0, 1, 2, 7]
        and loop.consolidate_log[0]["n"] == 4
        and loop.n_group_splits == 1 and loop.n_group_children == 2
        and loop.n_split == 2 and loop.n_retired_slow == 2
        and len(loop.retired_log) == 2
        and all(rl.get("source") == "group" for rl in loop.retired_log)
        and sum(1 for rl in loop.retired_log if rl.get("consolidated") == 1) == 1
        and loop.group_log[-1]["trigger"] == 1
        and loop.group_log[-1]["consolidated0"] == [20]
        and loop.group_log[-1]["consolidated1"] == []
        and c0["hits"] == 4 and c1["hits"] == 4
        and abs(c0["mu"][0] - exp0[0]) < 1e-12 and abs(c0["mu"][1] - exp0[1]) < 1e-12
        and abs(c1["mu"][0] - exp1[0]) < 1e-12 and abs(c1["mu"][1] - exp1[1]) < 1e-12
        and len(c0["ledger"][0]) == 4 and len(c1["ledger"][1]) == 4)

    # 4. (C) 单侧纪元不触发：纪元全为 c2=0 -> G1（两值各 >=3）构造性失败 -> 无物化
    loop = SlotLoop4(mode="on", window=WINDOW, **LOOP_CFG)
    loop._win = 7
    loop._next_pid = 11
    loop.epoch_hist = [(w, 0, 147 + w, 80, 10) for w in range(7)]
    loop.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=5, created=0, last_active=7, n_match=5,
             kind="slow", promoted_at=3, tag=None, n_backfill=0,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
    ]
    loop._ev_win = _mask_with(151, 82, 0)
    loop._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop._on_window()
    results["single_side_no_trigger"] = int(
        loop.n_consolidated == 0 and loop.n_group_splits == 0
        and loop.n_split == 0 and loop.n_retired_slow == 0
        and loop.group_log[-1]["g1"] == 0)

    # 5. (D) G2 门：两簇能量同质（比值 < 1.30）-> G2 失败 -> 无物化无触发
    #    （0 侧无自然锚但 G2 失败，物化被 g2 门拦下）
    loop = SlotLoop4(mode="on", window=WINDOW, **LOOP_CFG)
    loop._win = 7
    loop._next_pid = 20
    loop.epoch_hist = [(0, 0, 147, 80, 10), (1, 0, 160, 82, 12), (2, 0, 155, 81, 13),
                       (3, 1, 155, 40, 11), (4, 1, 160, 42, 11),
                       (5, 1, 158, 41, 11), (6, 1, 162, 43, 11)]
    loop.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=1, created=0, last_active=0, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(147.0, 80.0)]}),
        dict(pid=12, mu=(5.1, 4.5), hits=1, created=1, last_active=1, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(160.0, 82.0)]}),
        dict(pid=13, mu=(5.05, 4.45), hits=1, created=2, last_active=2, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(155.0, 81.0)]}),
        dict(pid=11, mu=(5.05, 3.7), hits=5, created=3, last_active=6, n_match=5,
             kind="slow", promoted_at=4, tag=None, n_backfill=0,
             ledger={1: [(155.0, 40.0), (160.0, 42.0),
                         (158.0, 41.0), (162.0, 43.0)]}),
    ]
    loop._ev_win = _mask_with(151, 82, 0)
    loop._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop._on_window()
    results["g2_gate_no_consolidate"] = int(
        loop.n_consolidated == 0 and loop.n_group_splits == 0
        and loop.n_split == 0 and loop.n_retired_slow == 0
        and loop.group_log[-1]["g1"] == 1 and loop.group_log[-1]["g2"] == 0)

    # 6. (E) G4 防通胀：纪元内已有打标原型 -> G4=0 -> 无物化无触发
    loop = SlotLoop4(mode="on", window=WINDOW, **LOOP_CFG)
    loop._win = 7
    loop._next_pid = 30
    loop.epoch_hist = [(0, 0, 147, 80, 10), (1, 0, 160, 82, 12), (2, 0, 155, 81, 13),
                       (3, 1, 3000, 400, 11), (4, 1, 3200, 410, 11),
                       (5, 1, 3100, 405, 11), (6, 1, 3050, 402, 11)]
    loop.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=1, created=0, last_active=0, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(147.0, 80.0)]}),
        dict(pid=12, mu=(5.1, 4.5), hits=1, created=1, last_active=1, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(160.0, 82.0)]}),
        dict(pid=13, mu=(5.05, 4.45), hits=1, created=2, last_active=2, n_match=1,
             kind="fast", promoted_at=None, tag=None, n_backfill=0,
             ledger={0: [(155.0, 81.0)]}),
        dict(pid=11, mu=(8.0, 6.0), hits=5, created=3, last_active=6, n_match=5,
             kind="slow", promoted_at=4, tag=None, n_backfill=0,
             ledger={1: [(3000.0, 400.0), (3200.0, 410.0),
                         (3100.0, 405.0), (3050.0, 402.0)]}),
        dict(pid=19, mu=(5.0, 4.4), hits=4, created=5, last_active=6, n_match=0,
             kind="slow", promoted_at=5, tag=0, n_backfill=0,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
    ]
    loop._ev_win = _mask_with(151, 82, 0)
    loop._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop._on_window()
    results["g4_gate_no_consolidate"] = int(
        loop.n_consolidated == 0 and loop.n_group_splits == 0
        and loop.n_split == 0 and loop.n_retired_slow == 0
        and loop.group_log[-1]["g1"] == 1 and loop.group_log[-1]["g2"] == 1
        and loop.group_log[-1]["g4"] == 0)

    for k in ("construct_off", "construct_on", "repro_synth",
              "natural_fire_no_consolidate", "consolidate_fire",
              "single_side_no_trigger", "g2_gate_no_consolidate",
              "g4_gate_no_consolidate"):
        print("R_L4D_SMOKE_%s=%d" % (k.upper(), results[k]))
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
    r0b_frames = allv["bear"] * 5            # R0b = bear 段 x5 = 410 帧（负对照，重冻结沿用）
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

    # ---- Mode OFF（DeferredLoop 逐字；守卫 R_L4D_GUARD_D251） ----
    off = {}
    for sid in STREAM_ORDER:
        out, _ = run_slot4_stream(stream_frames[sid], "off")
        out["stream_id"] = sid
        off[sid] = out
    off_r0, _ = run_slot4_stream(r0_frames, "off")
    off_r0b, _ = run_slot4_stream(r0b_frames, "off")
    off_r1, _ = run_slot4_stream(r1_frames, "off")
    off_r1["bridge"] = bridge_metrics(build_entry_base(off_r1), spans)
    off_r1["gist"] = gist_metrics(off_r1, switch_windows)
    t_off = time.time() - t0 - t_dec

    d251_items = guard_d251_items(off, off_r1)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- Mode ON（槽位路径 + 三条账本修复 + 群级加法 + 锚物化；判据口径） ----
    on = {}
    for sid, sname, vidx in STREAMS:
        out, loop = run_slot4_stream(stream_frames[sid], "on")
        out["stream_id"] = sid
        out["stream_name"] = sname
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        out["switch_diag"] = scene_switch_diag(stream_frames[sid], creations)
        on[sid] = out
    on_r0, on_r0_loop = run_slot4_stream(r0_frames, "on")
    on_r0b, on_r0b_loop = run_slot4_stream(r0b_frames, "on")
    on_r1, on_r1_loop = run_slot4_stream(r1_frames, "on")
    on_r1["bridge"] = bridge_metrics(build_entry_base(on_r1), spans)
    on_r1["gist"] = gist_metrics(on_r1, switch_windows)
    on_r1["seg_info"] = r1_segment_info(on_r1_loop, spans)
    on_r1["split_align"] = split_segment_align(on_r1_loop, spans,
                                               on_r1["seg_info"])
    on_r1["group_align"] = group_split_align(on_r1["group_diag"], spans,
                                             on_r1["seg_info"],
                                             len(on_r1_loop.energy_trace))
    t_on = time.time() - t0 - t_dec - t_off

    # ---- R_L4D_REPRO_RATIO（构造性控制项：ON vs OFF 全流 ratio，abs < 1e-9） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            int(abs(on[sid]["ratio"] - off[sid]["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0", int(abs(on_r0["ratio"] - off_r0["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0B", int(abs(on_r0b["ratio"] - off_r0b["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R1", int(abs(on_r1["ratio"] - off_r1["ratio"]) < 1e-9)))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, v) for n, v in repro_items)

    # ---- R_L4D_GUARD_D246（SoftLoop 路径；docs/249/250/251 同一代码路径） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)
    guard246_passed = sum(1 for ch in guard246_detail.split(",") if ch.endswith(":1"))

    # ---- R_L4D_NONSPLIT_EQ（§1.7-8 诊断级：非分裂数字 vs docs/253 Mode ON） ----
    nonsplit_eq, nonsplit_per = nonsplit_compare3(on, on_r0, on_r0b, on_r1)

    # ---- 判据（§1.6 冻结；与 docs/253/254/256 逐字一致；ON 数字；负对照 = R0b） ----
    crit1 = int(on_r1["n_split"] >= 1 and on_r1["sc2_tagged"] >= 1
                and on_r1["compound_frac"] >= 0.5)
    crit2 = int(on_r1["spurious_split_frac"] <= 0.5
                and on_r1["avg_post_split_hits"] >= 1
                and on_r0b["n_split"] == 0)
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

    # ---- 判定（§1.6 冻结映射；第四格专属语义） ----
    guards_ok = d251_ok == 1 and guard246_ok == 1 and repro_ok == 1
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d -> implementation drift; fix implementation, "
                 "do not judge mechanism (see R_L4D_GUARD_*)" % (
                     d251_ok, d251_passed, guard246_ok, repro_ok))
    elif not crit1:
        verdict = "PARALLEL_ONLY_REAL"
        vnote = ("COMPOUND_EMERGES fails on R1 after anchor consolidation "
                 "(fourth cell): n_split=%d, SC2_tagged=%d, compound_frac=%.4f -> "
                 "L4 closes as negative evidence: with group-level aggregation "
                 "(epoch G_WIN=8, G1-G4) plus anchor consolidation and criteria "
                 "identical to docs/253/254/256, the two-c2-group info "
                 "distribution is still unreachable at representation level; "
                 "no threshold rollback" % (
                     on_r1["n_split"], on_r1["sc2_tagged"],
                     on_r1["compound_frac"]))
    elif not crit2:
        verdict = "BOUNDARY"
        vnote = ("COMPOUND_EMERGES passes but ADOPT_NONRANDOM fails: "
                 "spurious_split_frac=%.4f, avg_post_split_hits=%.4f, "
                 "R0b n_split=%d (group split fires via anchor consolidation "
                 "but gated conditional memory not maintained / R0b negative "
                 "control failed; docs/256 Sec-5.3 risk materialized)" % (
                     on_r1["spurious_split_frac"],
                     on_r1["avg_post_split_hits"], on_r0b["n_split"]))
    elif not (crit3 and crit4):
        why = []
        if not crit3:
            why.append("FOUNDATION_KEEP fails (ratio/sc2_slow/gist_cov; see numbers)")
        if not crit4:
            why.append("PROMOTION_KEEP fails (n_promo/n_recycle/hit-rate separation)")
        verdict = "PARTIAL_REAL"
        vnote = "; ".join(why) + " (see R_L4D_CRIT* numbers)"
    else:
        verdict = "COMPOSABLE_REAL"
        vnote = ("criteria 1-4 all pass and all guards pass: after anchor "
                 "consolidation (group-level G3 fix: dispersed same-side epoch "
                 "windows materialized into per-side confirmed slow anchors), "
                 "compound structure emerges on real streams (tagged conditional "
                 "slow memory via group-level organization incl. strong-info "
                 "segments), adoption is non-random (R0b negative control "
                 "clean), L3 foundation and promotion behavior evidence kept")

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
                             "g_win": G_WIN,
                             "k_g_confirm": K_G_CONFIRM,
                             "k_g_ledger": K_G_LEDGER,
                             "ctx_split_y": CTX_SPLIT_Y,
                             "slot_sparse_px": SLOT_SPARSE,
                             "participate": PARTICIPATE},
           "mechanism": ("SlotLoop4(SlotLoop3): docs/256 group-level path verbatim "
                         "(slot c2, prototype c2 ledger, birth backfill W_BF=4, "
                         "explicit upgrade ledger inheritance, child mu "
                         "re-initialization, single-prototype split, tagged gated "
                         "matching, group = last G_WIN=8 participating windows "
                         "(epoch, pure behavioral, no GT), group info criteria "
                         "G1/G2/G4, group split) + anchor consolidation: group "
                         "check order = G1 -> G2 -> G4 -> anchor consolidation "
                         "(new) -> G3 -> group split; for each c2 side v with "
                         "no confirmed natural anchor and >= k_g_ledger=3 epoch "
                         "windows (G1), materialize the epoch's dispersed "
                         "same-side windows into one untagged confirmed slow "
                         "anchor (mu = median vector of (ln(1+E), ln(1+U)) over "
                         "side windows; ledger = side union; hits = union count "
                         ">= k_ledger born confirmed; created = current window); "
                         "G3 evaluates natural + consolidated anchors (text "
                         "verbatim: per-side >= k_g_confirm=1 confirmed anchors "
                         "created/matched within epoch, ledger >= k_ledger); "
                         "group split retires all per-side anchors (natural + "
                         "consolidated) and creates one tagged slow prototype "
                         "per side (mu = median of retired side ledger union); "
                         "negative control R0b=bear x5 verbatim (G1 structural "
                         "failure -> no consolidation no trigger), R0=flamingo "
                         "x5 diagnostic (G1 fails); Mode OFF = DeferredLoop "
                         "verbatim; prediction path zero-change; no new numeric "
                         "knob (k_ledger/G_WIN/product rules frozen reuse)"),
           "loop": LOOP_CFG,
           "r1_switch_windows": switch_windows,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l4_compose_test4",
        "doc_ref": "docs/235, docs/243, docs/245, docs/246, docs/247, docs/248, "
                   "docs/249, docs/250, docs/251, docs/253, docs/254, docs/255, "
                   "docs/256, docs/257, docs/261, docs/262",
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
                      "r0b": {k: off_r0b[k] for k in ("frames", "ratio", "sc1_fast",
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
        "on_r0": on_r0, "on_r0b": on_r0b, "on_r1": on_r1,
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
                   "nonsplit_eq": {"ok": nonsplit_eq,
                                   "per_stream": nonsplit_per},
                   "group_hash": {sid: on[sid]["group_diag"]["group_hash"]
                                  for sid in STREAM_ORDER}
                                | {"R0": on_r0["group_diag"]["group_hash"],
                                   "R0b": on_r0b["group_diag"]["group_hash"],
                                   "R1": on_r1["group_diag"]["group_hash"]},
                   "consolidate_hash": {sid: on[sid]["group_diag"]["consolidate_hash"]
                                        for sid in STREAM_ORDER}
                                      | {"R0": on_r0["group_diag"]["consolidate_hash"],
                                         "R0b": on_r0b["group_diag"]["consolidate_hash"],
                                         "R1": on_r1["group_diag"]["consolidate_hash"]},
                   "r0b_nosplit": {"n_split": on_r0b["n_split"],
                                   "ok": int(on_r0b["n_split"] == 0)},
                   "bf_hash": {sid: on[sid]["backfill_diag"]["bf_hash_all"]
                               for sid in STREAM_ORDER}
                              | {"R0": on_r0["backfill_diag"]["bf_hash_all"],
                                 "R0b": on_r0b["backfill_diag"]["bf_hash_all"],
                                 "R1": on_r1["backfill_diag"]["bf_hash_all"]}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2),
                   "decode_sec": round(t_dec, 2),
                   "off_sec": round(t_off, 2), "on_sec": round(t_on, 2)},
    }
    res_path = os.path.join(args.out_dir, "l4d_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L4D_TAG=%s" % args.tag)
    print("R_L4D_G_WIN=%d" % G_WIN)
    print("R_L4D_K_G_CONFIRM=%d" % K_G_CONFIRM)
    print("R_L4D_K_G_LEDGER=%d" % K_G_LEDGER)
    print("R_L4D_W_BF=%d" % W_BF)
    print("R_L4D_R_SLOW=%.6f" % R_SLOW)
    print("R_L4D_R_FAST=%.6f" % R_FAST)
    print("R_L4D_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_L4D_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_L4D_K_PROMOTE=%d" % K_PROMOTE)
    print("R_L4D_K_DECAY=%d" % K_DECAY)
    print("R_L4D_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_L4D_ALPHA=%.4f" % ALPHA)
    print("R_L4D_K_SPLIT=%d" % K_SPLIT)
    print("R_L4D_DELTA_REL=%.4f" % DELTA_REL)
    print("R_L4D_K_LEDGER=%d" % K_LEDGER)
    print("R_L4D_CTX_SPLIT_Y=%.1f" % CTX_SPLIT_Y)
    for sid in STREAM_ORDER:
        r = off[sid]
        print("R_L4D_OFF_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4D_OFF_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4D_OFF_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4D_OFF_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4D_OFF_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4D_OFF_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
    print("R_L4D_OFF_R1_RATIO=%.6f" % off_r1["ratio"])
    print("R_L4D_OFF_R1_SC1_FAST=%d" % off_r1["sc1_fast"])
    print("R_L4D_OFF_R1_SC2_SLOW=%d" % off_r1["sc2_slow"])
    print("R_L4D_OFF_R1_N_PROMO=%d" % off_r1["n_promo"])
    print("R_L4D_OFF_R1_N_RECYCLE=%d" % off_r1["n_recycle"])
    print("R_L4D_OFF_R1_CHURN_SLOW=%.4f" % off_r1["churn_slow"])
    print("R_L4D_OFF_R1_GIST_COV=%.4f" % off_r1["gist"]["cov"])
    print("R_L4D_OFF_R1_BRIDGE_SW=%.4f" % off_r1["bridge"]["bridge_corr_switch"])
    print("R_L4D_OFF_R0_RATIO=%.6f" % off_r0["ratio"])
    print("R_L4D_OFF_R0_SC1_FAST=%d" % off_r0["sc1_fast"])
    print("R_L4D_OFF_R0_SC2_SLOW=%d" % off_r0["sc2_slow"])
    print("R_L4D_OFF_R0_CHURN_SLOW=%.4f" % off_r0["churn_slow"])
    print("R_L4D_OFF_R0_N_PROMO=%d" % off_r0["n_promo"])
    print("R_L4D_OFF_R0_N_RECYCLE=%d" % off_r0["n_recycle"])
    print("R_L4D_OFF_R0B_RATIO=%.6f" % off_r0b["ratio"])
    print("R_L4D_OFF_R0B_SC1_FAST=%d" % off_r0b["sc1_fast"])
    print("R_L4D_OFF_R0B_SC2_SLOW=%d" % off_r0b["sc2_slow"])
    print("R_L4D_OFF_R0B_CHURN_SLOW=%.4f" % off_r0b["churn_slow"])
    print("R_L4D_OFF_R0B_N_PROMO=%d" % off_r0b["n_promo"])
    print("R_L4D_OFF_R0B_N_RECYCLE=%d" % off_r0b["n_recycle"])
    print("R_L4D_GUARD_D251=%d" % d251_ok)
    print("R_L4D_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L4D_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L4D_GUARD_D251_DETAIL=%s" % d251_detail)
    for sid in ALL_STREAMS:
        if sid in on:
            r = on[sid]
        elif sid == "R0":
            r = on_r0
        elif sid == "R0B":
            r = on_r0b
        else:
            r = on_r1
        sc = r["slot_coverage"]
        bf = r["backfill_diag"]
        gd = r["group_diag"]
        d = r.get("switch_diag")
        sw = "NA" if d is None or d["switch_corr"] is None else \
            "%.4f" % d["switch_corr"]
        spb = ",".join(str(x["parent_n_backfill"]) for x in bf["split_parent_bf"])
        if not bf["split_parent_bf"]:
            spb = "NA"
        print("R_L4D_ON_%s_FRAMES=%d" % (sid, r["frames"]))
        print("R_L4D_ON_%s_WINDOWS=%d" % (sid, r["n_windows"]))
        print("R_L4D_ON_%s_VALID=%d" % (sid, r["n_valid"]))
        print("R_L4D_ON_%s_MAE=%.6f" % (sid, r["mae_mean_win"]))
        print("R_L4D_ON_%s_MAE_SD=%.6f" % (sid, r["mae_sd_win"]))
        print("R_L4D_ON_%s_MAE_LO=%.6f" % (sid, r["mae_ci95"][0]))
        print("R_L4D_ON_%s_MAE_HI=%.6f" % (sid, r["mae_ci95"][1]))
        print("R_L4D_ON_%s_Q1=%.6f" % (sid, r["mae_q1"]))
        print("R_L4D_ON_%s_Q4=%.6f" % (sid, r["mae_q4"]))
        print("R_L4D_ON_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4D_ON_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4D_ON_%s_SC2_FAST=%d" % (sid, r["sc2_fast"]))
        print("R_L4D_ON_%s_SC1_SLOW=%d" % (sid, r["sc1_slow"]))
        print("R_L4D_ON_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4D_ON_%s_SC2_TAGGED=%d" % (sid, r["sc2_tagged"]))
        print("R_L4D_ON_%s_COMPOUND_FRAC=%.4f" % (sid, r["compound_frac"]))
        print("R_L4D_ON_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
        print("R_L4D_ON_%s_CHURN_LEGACY=%.4f" % (sid, r["churn_legacy"]))
        print("R_L4D_ON_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4D_ON_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4D_ON_%s_N_SPLIT=%d" % (sid, r["n_split"]))
        print("R_L4D_ON_%s_N_RETIRED_SLOW=%d" % (sid, r["n_retired_slow"]))
        print("R_L4D_ON_%s_SPURIOUS_SPLIT_FRAC=%.4f" % (sid, r["spurious_split_frac"]))
        print("R_L4D_ON_%s_AVG_POST_SPLIT_HITS=%.4f" % (sid, r["avg_post_split_hits"]))
        print("R_L4D_ON_%s_PROMO_MEAN=%.4f" % (sid, r["promoted_mean_hits"]))
        print("R_L4D_ON_%s_NONPROMO_MEAN=%.4f" % (sid, r["nonpromoted_mean_hits"]))
        print("R_L4D_ON_%s_C2HASH=%s" % (sid, r["c2_hash"]))
        print("R_L4D_ON_%s_C2_COV=%.4f" % (sid, sc["coverage"]))
        print("R_L4D_ON_%s_C2_F0_NON=%.4f" % (sid, sc["frac0_non"]))
        print("R_L4D_ON_%s_C2_F1_NON=%.4f" % (sid, sc["frac1_non"]))
        print("R_L4D_ON_%s_C2_F0_ALL=%.4f" % (sid, sc["frac0_all"]))
        print("R_L4D_ON_%s_C2_F1_ALL=%.4f" % (sid, sc["frac1_all"]))
        print("R_L4D_ON_%s_TAG0=%d" % (sid, r["tag_dist"]["tag0"]))
        print("R_L4D_ON_%s_TAG1=%d" % (sid, r["tag_dist"]["tag1"]))
        print("R_L4D_ON_%s_SW_CORR=%s" % (sid, sw))
        print("R_L4D_GRP_%s_EPOCHS=%d" % (sid, gd["n_epoch_checks"]))
        print("R_L4D_GRP_%s_FULL=%d" % (sid, gd["n_full_epochs"]))
        print("R_L4D_GRP_%s_G1=%d" % (sid, gd["n_g1"]))
        print("R_L4D_GRP_%s_G1G2=%d" % (sid, gd["n_g1g2"]))
        print("R_L4D_GRP_%s_G1G2G3=%d" % (sid, gd["n_g1g2g3"]))
        print("R_L4D_GRP_%s_TRIGGER=%d" % (sid, gd["n_trigger"]))
        print("R_L4D_GRP_%s_CHILDREN=%d" % (sid, gd["n_group_children"]))
        print("R_L4D_GRP_%s_SPLIT_SINGLE=%d" % (sid, gd["n_split_single"]))
        print("R_L4D_GRP_%s_SPLIT_GROUP=%d" % (sid, gd["n_split_group"]))
        print("R_L4D_GRP_%s_CONSOLIDATED=%d" % (sid, gd["n_consolidated"]))
        print("R_L4D_GRP_%s_CONS_SIDE0=%d" % (sid, gd["n_consolidated_by_side"]["0"]))
        print("R_L4D_GRP_%s_CONS_SIDE1=%d" % (sid, gd["n_consolidated_by_side"]["1"]))
        print("R_L4D_GRP_%s_HASH=%s" % (sid, gd["group_hash"]))
        print("R_L4D_GRP_%s_CONS_HASH=%s" % (sid, gd["consolidate_hash"]))
        for i, ev in enumerate(gd["group_splits"]):
            print("R_L4D_GRP_%s_EV_%d_AT=%d" % (sid, i, ev["split_at"]))
            print("R_L4D_GRP_%s_EV_%d_WINS=%s" % (
                sid, i, ",".join(str(w) for w in ev["epoch_wins"])))
            print("R_L4D_GRP_%s_EV_%d_RATIO=%.4f" % (sid, i, ev["ratio"]))
            print("R_L4D_GRP_%s_EV_%d_ANCHORS=%s" % (
                sid, i, ";".join(["0:%s" % ",".join(str(p) for p in ev["anchor_pids"]["0"]),
                                  "1:%s" % ",".join(str(p) for p in ev["anchor_pids"]["1"])])))
            print("R_L4D_GRP_%s_EV_%d_CONS=%s" % (
                sid, i, ";".join(["0:%s" % ",".join(str(p) for p in ev["consolidated_pids"]["0"]),
                                  "1:%s" % ",".join(str(p) for p in ev["consolidated_pids"]["1"])])))
            print("R_L4D_GRP_%s_EV_%d_CHILDREN=%s" % (
                sid, i, ";".join("%d:%d:%d" % (c[0], c[1], c[2])
                                 for c in ev["children"])))
        for i, c in enumerate(gd["consolidate_log"]):
            print("R_L4D_GRP_%s_CONS_%d_AT=%d" % (sid, i, c["at"]))
            print("R_L4D_GRP_%s_CONS_%d_SIDE=%d" % (sid, i, c["side"]))
            print("R_L4D_GRP_%s_CONS_%d_N=%d" % (sid, i, c["n"]))
            print("R_L4D_GRP_%s_CONS_%d_WINS=%s" % (
                sid, i, ",".join(str(w) for w in c["source_wins"])))
            print("R_L4D_GRP_%s_CONS_%d_MU0=%.6f" % (sid, i, c["mu0"]))
            print("R_L4D_GRP_%s_CONS_%d_MU1=%.6f" % (sid, i, c["mu1"]))
        print("R_L4D_BF_%s_N_CREATIONS=%d" % (sid, bf["n_creations"]))
        print("R_L4D_BF_%s_N_CREATIONS_WITH_BF=%d" % (sid, bf["n_creations_with_bf"]))
        print("R_L4D_BF_%s_WINDOW_TOTAL=%d" % (sid, bf["bf_window_total"]))
        print("R_L4D_BF_%s_AVG_PER_CREATION=%.4f" % (sid, bf["bf_avg_per_creation"]))
        print("R_L4D_BF_%s_AVG_PER_BF_CREATION=%.4f" % (sid, bf["bf_avg_per_bf_creation"]))
        print("R_L4D_BF_%s_C2_NONE=%d" % (sid, bf["bf_c2_dist"]["None"]))
        print("R_L4D_BF_%s_C2_0=%d" % (sid, bf["bf_c2_dist"]["0"]))
        print("R_L4D_BF_%s_C2_1=%d" % (sid, bf["bf_c2_dist"]["1"]))
        print("R_L4D_BF_%s_SPLIT_PARENT=%s" % (sid, spb))
        print("R_L4D_BF_%s_BFHASH=%s" % (sid, bf["bf_hash_all"]))
    print("R_L4D_ON_R1_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    seg = on_r1["seg_info"]
    print("R_L4D_ON_R1_SEG_INFO=%s" % ",".join(
        ("NA" if row["ratio"] is None else "%.4f" % row["ratio"]) for row in seg))
    print("R_L4D_ON_R1_SEG_N0=%s" % ",".join(str(row["n0"]) for row in seg))
    print("R_L4D_ON_R1_SEG_N1=%s" % ",".join(str(row["n1"]) for row in seg))
    print("R_L4D_ON_R1_ALIGN_RATE=%.4f" % on_r1["split_align"]["align_rate"])
    print("R_L4D_ON_R1_N_ALIGNED=%d" % on_r1["split_align"]["n_aligned"])
    ga = on_r1["group_align"]
    print("R_L4D_ON_R1_GROUP_ALIGN_RATE=%.4f" % ga["group_align_rate"])
    print("R_L4D_ON_R1_GROUP_N_ALIGNED=%d" % ga["n_aligned"])
    print("R_L4D_ON_R1_GROUP_N_SPLITS=%d" % ga["n_group_splits"])
    for i, ps in enumerate(ga["per_split"]):
        print("R_L4D_ON_R1_GROUP_EV_%d_SEG=%s" % (
            i, "NA" if ps["mode_segment"] is None else str(ps["mode_segment"])))
        print("R_L4D_ON_R1_GROUP_EV_%d_SEG_RATIO=%s" % (
            i, "NA" if ps["seg_ratio"] is None else "%.4f" % ps["seg_ratio"]))
        print("R_L4D_ON_R1_GROUP_EV_%d_ALIGNED=%d" % (i, ps["aligned"]))
    print("R_L4D_REPRO_RATIO=%d" % repro_ok)
    print("R_L4D_REPRO_ITEMS=%d" % len(repro_items))
    print("R_L4D_REPRO_DETAIL=%s" % repro_detail)
    print("R_L4D_NONSPLIT_EQ=%d" % nonsplit_eq)
    for sid in ALL_STREAMS:
        if sid == "R0B":
            continue            # docs/253 无 R0b 基准 -> NA
        print("R_L4D_NONSPLIT_%s=%d" % (sid, nonsplit_per[sid]["eq"]))
    for sid in ALL_STREAMS:
        print("R_L4D_NONSPLIT_%s_SPLIT=%d" % (sid, nonsplit_per[sid]["n_split"]))
        print("R_L4D_NONSPLIT_%s_DEV=%s" % (sid, nonsplit_per[sid]["dev_source"]))
    print("R_L4D_CRIT1_COMPOUND_EMERGES=%d" % crit1)
    print("R_L4D_CRIT1_N_SPLIT=%d" % on_r1["n_split"])
    print("R_L4D_CRIT1_SC2_TAGGED=%d" % on_r1["sc2_tagged"])
    print("R_L4D_CRIT1_COMPOUND_FRAC=%.4f" % on_r1["compound_frac"])
    print("R_L4D_CRIT2_ADOPT_NONRANDOM=%d" % crit2)
    print("R_L4D_CRIT2_SPURIOUS_FRAC=%.4f" % on_r1["spurious_split_frac"])
    print("R_L4D_CRIT2_AVG_POST_HITS=%.4f" % on_r1["avg_post_split_hits"])
    print("R_L4D_CRIT2_R0B_N_SPLIT=%d" % on_r0b["n_split"])
    print("R_L4D_CRIT3_FOUNDATION_KEEP=%d" % crit3)
    print("R_L4D_CRIT3_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    print("R_L4D_CRIT4_PROMOTION_KEEP=%d" % crit4)
    print("R_L4D_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L4D_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L4D_GUARD_D246=%d" % guard246_ok)
    print("R_L4D_GUARD_D246_ITEMS=%d" % 12)
    print("R_L4D_GUARD_D246_PASSED=%d" % guard246_passed)
    print("R_L4D_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L4D_R0B_NOSPLIT=%d" % on_r0b["n_split"])
    print("R_L4D_R0B_NOSPLIT_OK=%d" % int(on_r0b["n_split"] == 0))
    print("R_L4D_VERDICT=%s" % verdict)
    print("R_L4D_VERDICT_NOTE=%s" % vnote)
    print("R_L4D_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
