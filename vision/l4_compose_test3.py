"""vision/l4_compose_test3.py — docs/256 L4 组合泛化第三格：原型群级账本（SlotLoop3）。

SlotLoop3(SlotLoop2)（docs/256 §一 冻结，运行后不改）：
  Mode OFF = super()._on_window() 链（= SlotLoop2 OFF = SlotLoop OFF = DeferredLoop 逐字
             -> R_L4C_GUARD_D251 复现 docs/251 §3.3/§3.4，32 项，容差 1e-4）；
  Mode ON  = docs/254 SlotLoop2 槽位路径逐字（c2 可观测 + 账本 + 出生回填 + 升级并账 +
             子条目 μ 重初始化 + 单原型分裂 + 门控匹配）**+ 本格群级加法**（全部只落在
             模式表路径，预测路径零改动）：
             ① "群" = 最近 G_WIN=8 个参与窗口（E>=10）的时间纪元（epoch）；
                epoch_hist = 每参与窗口追加 (win, c2, E, U, matched_pid)，超 8 弹最旧
                （群级账本 = 纪元窗口记录本身，跨原型聚合"被不同原型各匹配"的两 c2 组）；
             ② 群级信息量判据 G1-G4（每参与窗口，在单原型 _split_check 之后执行）：
                G1 纪元两 c2 值各 >= k_g_ledger=3 窗；
                G2 群级中位事件能量比 max(med_E(0), med_E(1))/min(...) >= 1+delta_rel=1.30；
                G3 每侧 >= k_g_confirm=1 个未打标已确认慢原型锚（hits >= hits_min_slow、
                    pid 在纪元 matched_pid 或 created 在纪元窗界内、账本该值 >= k_ledger、
                    非双侧锚）；
                G4 纪元内无打标慢原型（每个局部一次性组织，防通胀）；
             ③ 群级分裂：纪元内全部 v 侧锚（v in {0,1}）退休（retired_log 注明群级来源）
                -> 每侧建一个打标慢原型（tag=v、μ = 退休 v 侧锚账本 v 值全部窗口
                (ln(1+E), ln(1+U)) 中位数向量——SlotLoop2 ③ 同款 μ 重初始化规则、
                ledger = 并集窗口、hits = 并集计数（>= k_ledger -> 出生即确认）、
                n_match=0、created=当前窗）；n_split += 1（按子条目计数）；
                split_log 注明群级来源。
  每参与窗口冻结执行顺序：(1) 匹配/创建（SlotLoop2 逐字）-> (2) 回收 -> (3) 单原型
  _split_check（逐字）-> (4) 群级检查（本格新增）-> (5) _hist + epoch_hist 维护。

负对照重冻结（docs/256 §1.3-5，本格命门）：R0=flamingo×5 与 R1-flamingo 段输入逐帧相同
-> 任何群级判据在两者上触发等价 -> R0 无法继续担任判据 2 负对照；判据 2 负对照流 =
R0b=bear×5（单场景循环 + c2 单侧布局 -> G1 构造性失败 -> n_split(R0b)=0 由机制保证）；
R0=flamingo×5 降级为诊断流（如实报告触发形态，不进判据）。

度量（§1.4 冻结）：M1-M5 与 docs/254 逐字一致 + M6 群级诊断（纪元 G1/G1+G2/G1+G2+G3
计数、群级分裂事件明细、群级分裂-段对齐 group_align_rate）——诊断级，不进判据。

判据（§1.6 冻结，与 docs/253/254 逐条一致）：①[L4][机制][组合测试] COMPOUND_EMERGES
（R1：n_split>=1 且 SC2_tagged>=1 且 compound_frac>=0.5）；②[L4][机制][行为证据]
ADOPT_NONRANDOM（R1：spurious_split_frac<=0.5 且 avg_post_split_hits>=1；**R0b**：
n_split==0）；③[L4][机制] FOUNDATION_KEEP（R1+S1-S4：ratio<=1.5 且 SC2_slow>0；R1：
gist_cov>=0.5）；④[L4][机制][行为证据] PROMOTION_KEEP（全局 n_promo>0 且 n_recycle>0
且升级命中率均值>未升级均值）。判定映射按 §1.7（第三格专属语义：群级账本后 R1 仍
n_split=0 = L4 收束为负证据）。

守卫（§1.8 冻结，不进判据）：R_L4C_GUARD_D251（Mode OFF 复现 docs/251，32 项）、
R_L4C_GUARD_D246（run_guard_quota + guard_vs_d246，12/12）、R_L4C_REPRO_RATIO
（ON vs OFF 全流 ratio abs<1e-9；冻结 6/6 = S1-S4+R0+R1，R0b 增量报告）、
R_L4C_GROUP_HASH（群级日志确定性指纹，两轮逐位一致）、R_L4C_R0B_NOSPLIT
（n_split(R0b)==0 构造性复检）、R_L4C_NONSPLIT_EQ（无分裂流非分裂数字 vs docs/253
逐位一致，分裂流如实标注偏离）。

安全纪律（§1.11 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L4C_* 摘要块；
运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。

用法：
  python vision/l4_compose_test3.py --smoke        # 构造冒烟（合成帧，非数据）
  python vision/l4_compose_test3.py --tag timing
  python vision/l4_compose_test3.py --tag main
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
# import 复用 l4_compose_test2（docs/256 §1.11 冻结清单：SlotLoop2/W_BF 为本格基座；
# D253_ON/proto_detail2/冒烟辅助为同文件复用）
from l4_compose_test2 import (SlotLoop2, W_BF, D253_ON, proto_detail2,
                              _synth_frames, _lay, _mask_with)
# import 复用 l4_compose_test（docs/256 §1.11 冻结清单）
from l4_compose_test import (SlotLoop, K_SPLIT, DELTA_REL, K_LEDGER,
                             SLOT_SPARSE, PARTICIPATE, _slot_c2, _c2_hash,
                             r1_segment_info, split_segment_align,
                             guard_d251_items, STREAM_ORDER)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 本格唯一新旋钮（docs/256 §1.6 冻结；全部冻结文档复用/直译，零重调） ----------------
G_WIN = 8                 # 纪元窗数 = 最近 G_WIN 个参与窗口（docs/235 §1.1 诊断段长上界 8 的冻结复用）
K_G_CONFIRM = 1           # 每侧确认慢原型锚数（docs/255 §一.4"该段两 c2 组都有慢原型命中"直译）
K_G_LEDGER = K_LEDGER     # G1 群级每值窗口门槛 = k_ledger = 3（docs/235 k_consist=3 复用）

ALL_STREAMS = STREAM_ORDER + ["R0", "R0B", "R1"]


# ---------------- SlotLoop3（docs/256 §1.3 冻结：SlotLoop2 逐字 + 群级加法） ----------------
class SlotLoop3(SlotLoop2):
    """SlotLoop2（docs/254 逐字）+ docs/256 §1.3 冻结群级加法：
    ① epoch_hist（G_WIN=8，群级账本 = 纪元窗口记录本身）；
    ② 群级信息量判据 G1-G4（纪元级，跨原型聚合两 c2 组）；
    ③ 群级分裂（纪元内单侧锚退休 -> 每侧打标慢原型，μ = 退休 v 侧窗口特征中位数向量）。
    Mode OFF = super()._on_window()（= SlotLoop2 OFF = DeferredLoop 逐字）。
    每参与窗口执行顺序：(1) 匹配/创建 -> (2) 回收 -> (3) 单原型 _split_check ->
    (4) 群级检查（本格新增）-> (5) _hist + epoch_hist 维护。"""

    def __init__(self, mode="off", k_split=K_SPLIT, delta_rel=DELTA_REL,
                 k_ledger=K_LEDGER, w_bf=W_BF, g_win=G_WIN,
                 k_g_confirm=K_G_CONFIRM, **kw):
        self.g_win = int(g_win)
        self.k_g_confirm = int(k_g_confirm)
        self.epoch_hist = []        # 最近 <=G_WIN 个参与窗口 (win, c2, E, U, matched_pid)
        self.group_log = []         # 每参与窗口群级检查记录（M6 + R_L4C_GROUP_HASH）
        self.group_split_events = []  # 群级分裂事件明细（M6）
        self.n_group_splits = 0     # 群级分裂触发次数（窗口数）
        self.n_group_children = 0   # 群级分裂产物（打标子条目）数
        self.n_g1 = 0               # 满足 G1 的纪元数
        self.n_g1g2 = 0             # 满足 G1+G2 的纪元数
        self.n_g1g2g3 = 0           # 满足 G1+G2+G3（潜在触发）的纪元数
        super().__init__(mode=mode, k_split=k_split, delta_rel=delta_rel,
                         k_ledger=k_ledger, w_bf=w_bf, **kw)

    # ---- 群级日志指纹（R_L4C_GROUP_HASH 口径；窗口史与原型状态的确定性纯函数） ----
    def _group_log_hash(self):
        parts = []
        for e in self.group_log:
            ch = "|".join("%d:%d:%d" % (c[0], c[1], c[2])
                          for c in e.get("children", []))
            parts.append("%d@%d@%d@%d@%d@%d@[%s]@%d@%d@%.4f@%.4f@%.4f@[%s]@[%s]@%s" % (
                e["win"], e["g1"], e["g2"], e["g3"], e["g4"], e["trigger"],
                ",".join(str(w) for w in e["epoch_wins"]),
                e["n0"], e["n1"], e["med0"], e["med1"], e["ratio"],
                ",".join(str(p) for p in e["anchors0"]),
                ",".join(str(p) for p in e["anchors1"]),
                ch))
        return hashlib.md5(";".join(parts).encode("utf-8")).hexdigest()

    # ---- 群级检查（docs/256 §1.3-4 冻结；每参与窗口，在单原型 _split_check 之后） ----
    def _group_check(self, c2, E, U, matched_pid):
        """纪元 = epoch_hist（此前参与窗口）+ 当前窗记录，取最近 <= G_WIN 个。
        G1 两 c2 值各 >= k_g_ledger；G2 群级中位事件能量比 >= 1+delta_rel；
        G3 每侧 >= k_g_confirm 个未打标已确认慢原型锚（纪元成员、账本该值 >= k_ledger、
        非双侧锚）；G4 纪元内无打标原型。全部满足 -> 群级分裂。"""
        epoch = list(self.epoch_hist) + [(self._win, c2, E, U, matched_pid)]
        if len(epoch) > self.g_win:
            epoch = epoch[-self.g_win:]
        ep_wins = [w for (w, _, _, _, _) in epoch]
        ep_min = min(ep_wins)
        ep_max = max(ep_wins)
        matched_pids = set(pid for (_, _, _, _, pid) in epoch
                           if pid is not None and pid >= 0)

        # G1：两 c2 组群级覆盖
        n0 = sum(1 for (_, c, _, _, _) in epoch if c == 0)
        n1 = sum(1 for (_, c, _, _, _) in epoch if c == 1)
        g1 = int(n0 >= K_G_LEDGER and n1 >= K_G_LEDGER)

        # G2：群级中位事件能量比（定义与 docs/253/254 逐字相同，分母 = 纪元窗口集）
        e0 = [float(Ev) for (_, c, Ev, _, _) in epoch if c == 0]
        e1 = [float(Ev) for (_, c, Ev, _, _) in epoch if c == 1]
        med0 = float(np.median(e0)) if e0 else 0.0
        med1 = float(np.median(e1)) if e1 else 0.0
        ratio = (max(med0, med1) / min(med0, med1)
                 if min(med0, med1) > 0 else 0.0)
        g2 = int(ratio >= (1 + self.delta_rel))

        # G3：每侧已确认慢原型锚（未打标、纪元成员、账本该值 >= k_ledger、非双侧锚）
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
        g3 = int(len(anchors[0]) >= self.k_g_confirm
                 and len(anchors[1]) >= self.k_g_confirm)

        # G4：纪元内无打标慢原型（每个局部一次性组织，防通胀）
        g4 = 1
        for p in self.prototypes:
            if p["kind"] != "slow" or p.get("tag") is None:
                continue
            if p["pid"] in matched_pids or (ep_min <= p["created"] <= ep_max):
                g4 = 0
                break

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
                 "anchors1": [a["pid"] for a in anchors[1]], "children": []}
        self.group_log.append(entry)

        if not trigger:
            return
        # ---- 群级分裂（§1.3-4 冻结）：全部 v 侧锚退休 -> 每侧建打标慢原型 ----
        self.n_group_splits += 1
        ev = {"split_at": self._win, "epoch_wins": list(ep_wins),
              "epoch_n0": n0, "epoch_n1": n1,
              "med0": round(med0, 4), "med1": round(med1, 4),
              "ratio": round(ratio, 4),
              "anchor_pids": {"0": [a["pid"] for a in anchors[0]],
                              "1": [a["pid"] for a in anchors[1]]},
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
                                         tags=[v], source="group"))
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

    # ---- Mode ON 主体 = SlotLoop2 逐字 + (4) 群级检查插入（docs/256 §1.3 执行顺序冻结） ----
    def _on_window(self):
        if self.mode != "on":
            super()._on_window()            # Mode OFF = SlotLoop2 OFF = DeferredLoop 逐字
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
        # 5. 分裂检查（§1.3-3 冻结；每窗口对每个已确认慢原型；SlotLoop2 逐字）
        self._split_check()
        # 6. 群级检查（docs/256 §1.3-4 冻结；本格新增；纪元 = epoch_hist + 当前窗）
        if E >= 10:
            self._group_check(c2, E, U, matched_pid)
        # 7. 窗口史维护（_hist 不变 + epoch_hist 本格新增：最近 <=G_WIN 个参与窗口）
        if E >= 10:
            self._hist.append((self._win, x, c2, E, U))
            if len(self._hist) > self.w_bf:
                self._hist.pop(0)
            self.epoch_hist.append((self._win, c2, E, U, matched_pid))
            if len(self.epoch_hist) > self.g_win:
                self.epoch_hist.pop(0)
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
        # ---- M6 群级诊断（§1.4 冻结，诊断级，不进判据） ----
        n_checks = len(self.group_log)
        out["group_diag"] = {
            "g_win": self.g_win, "k_g_confirm": self.k_g_confirm,
            "k_g_ledger": K_G_LEDGER,
            "n_epoch_checks": n_checks,
            "n_full_epochs": max(0, n_checks - self.g_win + 1),
            "n_g1": self.n_g1, "n_g1g2": self.n_g1g2,
            "n_g1g2g3": self.n_g1g2g3,
            "n_trigger": self.n_group_splits,
            "n_group_children": self.n_group_children,
            "n_split_single": self.n_split - self.n_group_children,
            "n_split_group": self.n_group_children,
            "group_splits": self.group_split_events,
            "group_log": self.group_log,
            "group_hash": self._group_log_hash(),
        }
        return out


# ---------------- 单流运行（与 run_slot2_stream 同构；Mode OFF 逐位一致） ----------------
def run_slot3_stream(frames, mode):
    loop = SlotLoop3(mode=mode, window=WINDOW, **LOOP_CFG)
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


# ---------------- M6 诊断：群级分裂-段对齐（R1，GT 段边界只用于诊断/评估） ----------------
def group_split_align(group_diag, spans, seg_rows, n_windows_total):
    """每个群级分裂事件：纪元窗口的众数段、该段信息量比；
    group_align_rate = #{群级分裂 : 众数段信息量比 >= 1.30} / max(1, 群级分裂数)
    （docs/256 §1.4 M6 冻结，诊断级，不进判据）。"""
    seg_by_win = {}
    for i, (s0, s1) in enumerate(spans):
        for w in range(n_windows_total):
            if s0 <= w * WINDOW < s1:
                seg_by_win[w] = i
    aligned = 0
    per = []
    for ev in group_diag["group_splits"]:
        wins = [w for w in ev["epoch_wins"]]
        segs = [seg_by_win.get(w) for w in wins]
        segs = [s for s in segs if s is not None]
        mode_seg = None
        if segs:
            cnt = Counter(segs)
            mode_seg, _ = cnt.most_common(1)[0]
        ratio = None
        if mode_seg is not None:
            ratio = seg_rows[mode_seg]["ratio"]
        ok = int(ratio is not None and ratio >= (1 + DELTA_REL))
        aligned += ok
        per.append({"split_at": ev["split_at"], "epoch_wins": wins,
                    "mode_segment": mode_seg,
                    "seg_ratio": (round(ratio, 4) if ratio is not None else None),
                    "aligned": ok})
    n = len(group_diag["group_splits"])
    return {"n_group_splits": n, "n_aligned": aligned,
            "group_align_rate": round(aligned / max(1, n), 4), "per_split": per}


# ---------------- R_L4C_NONSPLIT_EQ：非分裂数字 vs docs/253 Mode ON（§1.8-7 诊断级） ----------------
def nonsplit_compare3(on, on_r0, on_r0b, on_r1):
    """Mode ON 非分裂数字（SC1_fast/SC2_fast/n_promo/n_recycle/ratio/MAE）与 docs/253
    Mode ON 逐位一致；分裂发生后的流可合法偏离（原型种群被群级分裂改变），逐流如实标注
    偏离来源（none / group_split）。R0b 为 docs/256 新增流，无 docs/253 基准 -> NA。"""
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
        results[sid] = {"eq": int(ok), "n_split": r["n_split"],
                        "dev_source": ("group_split" if r["n_split"] > 0 else "none")}
        all_ok = all_ok and ok
    for sid, r in (("R0", on_r0), ("R1", on_r1)):
        exp = D253_ON[sid]
        ok = (r["sc1_fast"] == exp["sc1_fast"]
              and r["sc2_fast"] == exp["sc2_fast"]
              and r["n_promo"] == exp["n_promo"]
              and r["n_recycle"] == exp["n_recycle"]
              and abs(r["ratio"] - exp["ratio"]) < 1e-9
              and abs(r["mae_mean_win"] - exp["mae"]) < 1e-9)
        results[sid] = {"eq": int(ok), "n_split": r["n_split"],
                        "dev_source": ("group_split" if r["n_split"] > 0 else "none")}
        all_ok = all_ok and ok
    # R0b = docs/256 新增流，无 docs/253 基准 -> eq=NA(-1)；n_split/dev_source 如实报告
    # （摘要块标签用 R0B 大写以匹配 extract_r 的 ^R_[A-Z0-9_]+= 纯正则口径）
    results["R0B"] = {"eq": -1, "n_split": on_r0b["n_split"],
                      "dev_source": ("group_split" if on_r0b["n_split"] > 0 else "none")}
    return int(all_ok), results


# ---------------- 构造冒烟（合成帧，非数据；R_L4C_SMOKE_*） ----------------
def smoke_main():
    """构造冒烟（docs/256 §二 轮 2）：SlotLoop3 mode off/on 在 30 帧合成灰度上构造运行
    正常；Mode OFF/ON ratio 逐位一致；群级分裂语义核对——(A) 两特征簇帧在单原型不可见
    两值时群级可见触发（子条目形态正确）；(B) 单侧 epoch 不触发（G1 构造性失败）；
    (C) G2 比率门不触发；(D) G4 防通胀（纪元内已有打标原型不重复组织）。"""
    results = {}

    # 1. 构造/运行：30 帧合成帧 off/on 均正常；合成帧上 ON/OFF ratio 逐位一致
    frames = _synth_frames(30)
    off_out, _ = run_slot3_stream(frames, "off")
    on_out, _ = run_slot3_stream(frames, "on")
    results["construct_off"] = int(isinstance(off_out, dict)
                                   and off_out.get("n_windows", 0) >= 1)
    results["construct_on"] = int(isinstance(on_out, dict)
                                  and on_out.get("n_windows", 0) >= 1
                                  and "slot_coverage" in on_out
                                  and "backfill_diag" in on_out
                                  and "group_diag" in on_out)
    results["repro_synth"] = int(abs(off_out["ratio"] - on_out["ratio"]) < 1e-9)

    # 2. (A) 群级分裂语义：两特征簇帧 -> 单原型（各只见单值）在群级可见 -> 触发
    loop = SlotLoop3(mode="on", window=WINDOW, **LOOP_CFG)
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
    # 注：μ = 退休 v 侧锚账本中 v 值全部窗口的中位数向量（docs/256 §1.3-4 冻结）——
    # 纪元窗口记录（epoch_hist）只在纪元检查用，不自动进入锚账本；c1 的账本 = P1.ledger[1]
    # 的 3 条 (3000,400)/(3200,410)/(3100,405)（当前窗 c2=0 匹配 P0 不触 P1 账本）
    exp1 = (float(np.median([np.log1p(3000.0), np.log1p(3200.0),
                             np.log1p(3100.0)])),
            float(np.median([np.log1p(400.0), np.log1p(410.0),
                             np.log1p(405.0)])))
    results["group_split_fire"] = int(
        loop.n_group_splits == 1 and loop.n_group_children == 2
        and loop.n_split == 2 and loop.n_retired_slow == 2
        and len(loop.retired_log) == 2
        and all(rl.get("source") == "group" for rl in loop.retired_log)
        and len(loop.split_log) == 2
        and all(sl.get("source") == "group" for sl in loop.split_log)
        and len(loop.group_split_events) == 1
        and loop.group_log[-1]["trigger"] == 1
        and c0["hits"] == 4 and c1["hits"] == 3
        and c0["n_match"] == 0 and c1["n_match"] == 0
        and c0["created"] == 7 and c1["created"] == 7
        and abs(c0["mu"][0] - exp0[0]) < 1e-12 and abs(c0["mu"][1] - exp0[1]) < 1e-12
        and abs(c1["mu"][0] - exp1[0]) < 1e-12 and abs(c1["mu"][1] - exp1[1]) < 1e-12
        and len(c0["ledger"][0]) == 4 and len(c1["ledger"][1]) == 3)

    # 3. (B) 单侧 epoch 不触发：纪元全为 c2=0 -> G1（两值各 >=3）构造性失败
    loop2 = SlotLoop3(mode="on", window=WINDOW, **LOOP_CFG)
    loop2._win = 7
    loop2._next_pid = 11
    loop2.epoch_hist = [(w, 0, 147 + w, 80, 10) for w in range(7)]
    loop2.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=5, created=0, last_active=7, n_match=5,
             kind="slow", promoted_at=3, tag=None, n_backfill=0,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
    ]
    loop2._ev_win = _mask_with(151, 82, 0)
    loop2._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop2._on_window()
    results["single_side_no_trigger"] = int(
        loop2.n_group_splits == 0 and loop2.n_split == 0
        and loop2.n_retired_slow == 0
        and loop2.group_log[-1]["g1"] == 0)

    # 4. (C) G2 比率门不触发：两簇能量同质（比值 < 1.30）
    loop4 = SlotLoop3(mode="on", window=WINDOW, **LOOP_CFG)
    loop4._win = 7
    loop4._next_pid = 12
    loop4.epoch_hist = [(0, 0, 147, 80, 10), (1, 0, 148, 81, 10), (2, 0, 146, 79, 10),
                        (3, 1, 155, 40, 11), (4, 1, 160, 42, 11),
                        (5, 1, 158, 41, 11), (6, 1, 162, 43, 11)]
    loop4.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=5, created=0, last_active=6, n_match=5,
             kind="slow", promoted_at=3, tag=None, n_backfill=0,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
        dict(pid=11, mu=(5.05, 3.7), hits=5, created=2, last_active=6, n_match=5,
             kind="slow", promoted_at=4, tag=None, n_backfill=0,
             ledger={1: [(155.0, 40.0), (160.0, 42.0), (158.0, 41.0)]}),
    ]
    loop4._ev_win = _mask_with(151, 82, 0)
    loop4._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop4._on_window()
    results["g2_ratio_gate"] = int(
        loop4.n_group_splits == 0 and loop4.n_split == 0
        and loop4.n_retired_slow == 0
        and loop4.group_log[-1]["g1"] == 1 and loop4.group_log[-1]["g2"] == 0)

    # 5. (D) G4 防通胀：纪元内已有打标原型 -> 不重复组织
    loop3 = SlotLoop3(mode="on", window=WINDOW, **LOOP_CFG)
    loop3._win = 7
    loop3._next_pid = 21
    loop3.epoch_hist = [(0, 0, 147, 80, 10), (1, 0, 148, 81, 10), (2, 0, 146, 79, 10),
                        (3, 1, 3000, 400, 11), (4, 1, 3200, 410, 11),
                        (5, 1, 3100, 405, 11), (6, 1, 3050, 402, 11)]
    loop3.prototypes = [
        dict(pid=10, mu=(5.0, 4.4), hits=5, created=0, last_active=6, n_match=5,
             kind="slow", promoted_at=3, tag=None, n_backfill=2,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
        dict(pid=11, mu=(8.0, 6.0), hits=5, created=2, last_active=6, n_match=5,
             kind="slow", promoted_at=4, tag=None, n_backfill=0,
             ledger={1: [(3000.0, 400.0), (3200.0, 410.0), (3100.0, 405.0)]}),
        dict(pid=20, mu=(5.0, 4.4), hits=4, created=5, last_active=6, n_match=0,
             kind="slow", promoted_at=5, tag=0, n_backfill=0,
             ledger={0: [(147.0, 80.0), (148.0, 81.0), (146.0, 79.0)]}),
    ]
    loop3._ev_win = _mask_with(151, 82, 0)
    loop3._frame_buf = [dict(mae=0.1, att=0.5, ev=0.3, theta=0.15, db=0.015)] * 10
    loop3._on_window()
    results["g4_anti_inflation"] = int(
        loop3.n_group_splits == 0 and loop3.n_split == 0
        and loop3.n_retired_slow == 0
        and loop3.group_log[-1]["g1"] == 1 and loop3.group_log[-1]["g2"] == 1
        and loop3.group_log[-1]["g3"] == 1 and loop3.group_log[-1]["g4"] == 0)

    for k in ("construct_off", "construct_on", "repro_synth",
              "group_split_fire", "single_side_no_trigger",
              "g2_ratio_gate", "g4_anti_inflation"):
        print("R_L4C_SMOKE_%s=%d" % (k.upper(), results[k]))
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
    r0b_frames = allv["bear"] * 5            # R0b = bear 段 x5 = 410 帧（负对照，重冻结）
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

    # ---- Mode OFF（DeferredLoop 逐字；守卫 R_L4C_GUARD_D251） ----
    off = {}
    for sid in STREAM_ORDER:
        out, _ = run_slot3_stream(stream_frames[sid], "off")
        out["stream_id"] = sid
        off[sid] = out
    off_r0, _ = run_slot3_stream(r0_frames, "off")
    off_r0b, _ = run_slot3_stream(r0b_frames, "off")
    off_r1, _ = run_slot3_stream(r1_frames, "off")
    off_r1["bridge"] = bridge_metrics(build_entry_base(off_r1), spans)
    off_r1["gist"] = gist_metrics(off_r1, switch_windows)
    t_off = time.time() - t0 - t_dec

    d251_items = guard_d251_items(off, off_r1)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- Mode ON（槽位路径 + 三条账本修复 + 群级加法；判据口径） ----
    on = {}
    for sid, sname, vidx in STREAMS:
        out, loop = run_slot3_stream(stream_frames[sid], "on")
        out["stream_id"] = sid
        out["stream_name"] = sname
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        out["switch_diag"] = scene_switch_diag(stream_frames[sid], creations)
        on[sid] = out
    on_r0, on_r0_loop = run_slot3_stream(r0_frames, "on")
    on_r0b, on_r0b_loop = run_slot3_stream(r0b_frames, "on")
    on_r1, on_r1_loop = run_slot3_stream(r1_frames, "on")
    on_r1["bridge"] = bridge_metrics(build_entry_base(on_r1), spans)
    on_r1["gist"] = gist_metrics(on_r1, switch_windows)
    on_r1["seg_info"] = r1_segment_info(on_r1_loop, spans)
    on_r1["split_align"] = split_segment_align(on_r1_loop, spans,
                                               on_r1["seg_info"])
    on_r1["group_align"] = group_split_align(on_r1["group_diag"], spans,
                                             on_r1["seg_info"],
                                             len(on_r1_loop.energy_trace))
    t_on = time.time() - t0 - t_dec - t_off

    # ---- R_L4C_REPRO_RATIO（构造性控制项：ON vs OFF 全流 ratio，abs < 1e-9） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            int(abs(on[sid]["ratio"] - off[sid]["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0", int(abs(on_r0["ratio"] - off_r0["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0B", int(abs(on_r0b["ratio"] - off_r0b["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R1", int(abs(on_r1["ratio"] - off_r1["ratio"]) < 1e-9)))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, v) for n, v in repro_items)

    # ---- R_L4C_GUARD_D246（SoftLoop 路径；docs/249/250/251 同一代码路径） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)
    guard246_passed = sum(1 for ch in guard246_detail.split(",") if ch.endswith(":1"))

    # ---- R_L4C_NONSPLIT_EQ（§1.8-7 诊断级：非分裂数字 vs docs/253 Mode ON） ----
    nonsplit_eq, nonsplit_per = nonsplit_compare3(on, on_r0, on_r0b, on_r1)

    # ---- 判据（§1.6 冻结；与 docs/253/254 逐字一致；ON 数字；负对照 = R0b） ----
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

    # ---- 判定（§1.7 冻结映射；第三格专属语义） ----
    guards_ok = d251_ok == 1 and guard246_ok == 1 and repro_ok == 1
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32 items (%d passed), D246=%d/12, "
                 "REPRO_RATIO=%d -> implementation drift; fix implementation, "
                 "do not judge mechanism (see R_L4C_GUARD_*)" % (
                     d251_ok, d251_passed, guard246_ok, repro_ok))
    elif not crit1:
        verdict = "PARALLEL_ONLY_REAL"
        vnote = ("COMPOUND_EMERGES fails on R1 after group-level ledger "
                 "(third cell): n_split=%d, SC2_tagged=%d, compound_frac=%.4f -> "
                 "L4 closes as negative evidence: with group-level aggregation "
                 "(epoch G_WIN=8, G1-G4) and criteria identical to docs/253/254, "
                 "the two-c2-group info distribution is still unreachable at "
                 "representation level; no threshold rollback" % (
                     on_r1["n_split"], on_r1["sc2_tagged"],
                     on_r1["compound_frac"]))
    elif not crit2:
        verdict = "BOUNDARY"
        vnote = ("COMPOUND_EMERGES passes but ADOPT_NONRANDOM fails: "
                 "spurious_split_frac=%.4f, avg_post_split_hits=%.4f, "
                 "R0b n_split=%d (group split fires but gated conditional memory "
                 "not maintained / R0b negative control failed)" % (
                     on_r1["spurious_split_frac"],
                     on_r1["avg_post_split_hits"], on_r0b["n_split"]))
    elif not (crit3 and crit4):
        why = []
        if not crit3:
            why.append("FOUNDATION_KEEP fails (ratio/sc2_slow/gist_cov; see numbers)")
        if not crit4:
            why.append("PROMOTION_KEEP fails (n_promo/n_recycle/hit-rate separation)")
        verdict = "PARTIAL_REAL"
        vnote = "; ".join(why) + " (see R_L4C_CRIT* numbers)"
    else:
        verdict = "COMPOSABLE_REAL"
        vnote = ("criteria 1-4 all pass and all guards pass: after group-level "
                 "ledger (epoch G_WIN=8, G1-G4 group info + group split), compound "
                 "structure emerges on real streams (tagged conditional slow "
                 "memory via group-level organization), adoption is non-random "
                 "(R0b negative control clean), L3 foundation and promotion "
                 "behavior evidence kept")

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
           "mechanism": ("SlotLoop3(SlotLoop2): docs/254 slot path verbatim (c2 "
                         "observable, prototype c2 ledger (c2,E,U), birth backfill "
                         "W_BF=4, explicit upgrade ledger inheritance, child mu "
                         "re-initialization, single-prototype split, tagged gated "
                         "matching) + group-level addition: group = last G_WIN=8 "
                         "participating windows (epoch, pure behavioral, no GT); "
                         "group ledger = epoch window stream (cross-prototype "
                         "aggregation of the two c2 groups); group info criteria "
                         "G1 (both c2 values >= k_g_ledger=3 windows) / G2 "
                         "(group median energy ratio >= 1+delta_rel=1.30) / G3 "
                         "(per-side >= k_g_confirm=1 confirmed untagged slow anchor) "
                         "/ G4 (no tagged prototype in epoch); group split = retire "
                         "all per-side anchors in epoch, create one tagged slow "
                         "prototype per side (mu = median vector of (ln(1+E), "
                         "ln(1+U)) over retired side windows; ledger = union; "
                         "hits = union count, born confirmed; n_match=0); "
                         "execution order per participating window: match/create -> "
                         "recycle -> single-prototype split -> group check -> "
                         "window-history maintenance; negative control refrozen: "
                         "R0b=bear x5 (G1 structurally fails), R0=flamingo x5 "
                         "downgraded to diagnostic stream (input-identity argument); "
                         "Mode OFF = DeferredLoop verbatim; prediction path "
                         "zero-change"),
           "loop": LOOP_CFG,
           "r1_switch_windows": switch_windows,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l4_compose_test3",
        "doc_ref": "docs/235, docs/243, docs/245, docs/246, docs/247, docs/248, "
                   "docs/249, docs/250, docs/251, docs/253, docs/254, docs/255, "
                   "docs/256",
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
    res_path = os.path.join(args.out_dir, "l4c3_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L4C_TAG=%s" % args.tag)
    print("R_L4C_G_WIN=%d" % G_WIN)
    print("R_L4C_K_G_CONFIRM=%d" % K_G_CONFIRM)
    print("R_L4C_K_G_LEDGER=%d" % K_G_LEDGER)
    print("R_L4C_W_BF=%d" % W_BF)
    print("R_L4C_R_SLOW=%.6f" % R_SLOW)
    print("R_L4C_R_FAST=%.6f" % R_FAST)
    print("R_L4C_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_L4C_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_L4C_K_PROMOTE=%d" % K_PROMOTE)
    print("R_L4C_K_DECAY=%d" % K_DECAY)
    print("R_L4C_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_L4C_ALPHA=%.4f" % ALPHA)
    print("R_L4C_K_SPLIT=%d" % K_SPLIT)
    print("R_L4C_DELTA_REL=%.4f" % DELTA_REL)
    print("R_L4C_K_LEDGER=%d" % K_LEDGER)
    print("R_L4C_CTX_SPLIT_Y=%.1f" % CTX_SPLIT_Y)
    for sid in STREAM_ORDER:
        r = off[sid]
        print("R_L4C_OFF_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4C_OFF_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4C_OFF_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4C_OFF_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4C_OFF_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4C_OFF_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
    print("R_L4C_OFF_R1_RATIO=%.6f" % off_r1["ratio"])
    print("R_L4C_OFF_R1_SC1_FAST=%d" % off_r1["sc1_fast"])
    print("R_L4C_OFF_R1_SC2_SLOW=%d" % off_r1["sc2_slow"])
    print("R_L4C_OFF_R1_N_PROMO=%d" % off_r1["n_promo"])
    print("R_L4C_OFF_R1_N_RECYCLE=%d" % off_r1["n_recycle"])
    print("R_L4C_OFF_R1_CHURN_SLOW=%.4f" % off_r1["churn_slow"])
    print("R_L4C_OFF_R1_GIST_COV=%.4f" % off_r1["gist"]["cov"])
    print("R_L4C_OFF_R1_BRIDGE_SW=%.4f" % off_r1["bridge"]["bridge_corr_switch"])
    print("R_L4C_OFF_R0_RATIO=%.6f" % off_r0["ratio"])
    print("R_L4C_OFF_R0_SC1_FAST=%d" % off_r0["sc1_fast"])
    print("R_L4C_OFF_R0_SC2_SLOW=%d" % off_r0["sc2_slow"])
    print("R_L4C_OFF_R0_CHURN_SLOW=%.4f" % off_r0["churn_slow"])
    print("R_L4C_OFF_R0_N_PROMO=%d" % off_r0["n_promo"])
    print("R_L4C_OFF_R0_N_RECYCLE=%d" % off_r0["n_recycle"])
    print("R_L4C_OFF_R0B_RATIO=%.6f" % off_r0b["ratio"])
    print("R_L4C_OFF_R0B_SC1_FAST=%d" % off_r0b["sc1_fast"])
    print("R_L4C_OFF_R0B_SC2_SLOW=%d" % off_r0b["sc2_slow"])
    print("R_L4C_OFF_R0B_CHURN_SLOW=%.4f" % off_r0b["churn_slow"])
    print("R_L4C_OFF_R0B_N_PROMO=%d" % off_r0b["n_promo"])
    print("R_L4C_OFF_R0B_N_RECYCLE=%d" % off_r0b["n_recycle"])
    print("R_L4C_GUARD_D251=%d" % d251_ok)
    print("R_L4C_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L4C_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L4C_GUARD_D251_DETAIL=%s" % d251_detail)
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
        print("R_L4C_ON_%s_FRAMES=%d" % (sid, r["frames"]))
        print("R_L4C_ON_%s_WINDOWS=%d" % (sid, r["n_windows"]))
        print("R_L4C_ON_%s_VALID=%d" % (sid, r["n_valid"]))
        print("R_L4C_ON_%s_MAE=%.6f" % (sid, r["mae_mean_win"]))
        print("R_L4C_ON_%s_MAE_SD=%.6f" % (sid, r["mae_sd_win"]))
        print("R_L4C_ON_%s_MAE_LO=%.6f" % (sid, r["mae_ci95"][0]))
        print("R_L4C_ON_%s_MAE_HI=%.6f" % (sid, r["mae_ci95"][1]))
        print("R_L4C_ON_%s_Q1=%.6f" % (sid, r["mae_q1"]))
        print("R_L4C_ON_%s_Q4=%.6f" % (sid, r["mae_q4"]))
        print("R_L4C_ON_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4C_ON_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4C_ON_%s_SC2_FAST=%d" % (sid, r["sc2_fast"]))
        print("R_L4C_ON_%s_SC1_SLOW=%d" % (sid, r["sc1_slow"]))
        print("R_L4C_ON_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4C_ON_%s_SC2_TAGGED=%d" % (sid, r["sc2_tagged"]))
        print("R_L4C_ON_%s_COMPOUND_FRAC=%.4f" % (sid, r["compound_frac"]))
        print("R_L4C_ON_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
        print("R_L4C_ON_%s_CHURN_LEGACY=%.4f" % (sid, r["churn_legacy"]))
        print("R_L4C_ON_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4C_ON_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4C_ON_%s_N_SPLIT=%d" % (sid, r["n_split"]))
        print("R_L4C_ON_%s_N_RETIRED_SLOW=%d" % (sid, r["n_retired_slow"]))
        print("R_L4C_ON_%s_SPURIOUS_SPLIT_FRAC=%.4f" % (sid, r["spurious_split_frac"]))
        print("R_L4C_ON_%s_AVG_POST_SPLIT_HITS=%.4f" % (sid, r["avg_post_split_hits"]))
        print("R_L4C_ON_%s_PROMO_MEAN=%.4f" % (sid, r["promoted_mean_hits"]))
        print("R_L4C_ON_%s_NONPROMO_MEAN=%.4f" % (sid, r["nonpromoted_mean_hits"]))
        print("R_L4C_ON_%s_C2HASH=%s" % (sid, r["c2_hash"]))
        print("R_L4C_ON_%s_C2_COV=%.4f" % (sid, sc["coverage"]))
        print("R_L4C_ON_%s_C2_F0_NON=%.4f" % (sid, sc["frac0_non"]))
        print("R_L4C_ON_%s_C2_F1_NON=%.4f" % (sid, sc["frac1_non"]))
        print("R_L4C_ON_%s_C2_F0_ALL=%.4f" % (sid, sc["frac0_all"]))
        print("R_L4C_ON_%s_C2_F1_ALL=%.4f" % (sid, sc["frac1_all"]))
        print("R_L4C_ON_%s_TAG0=%d" % (sid, r["tag_dist"]["tag0"]))
        print("R_L4C_ON_%s_TAG1=%d" % (sid, r["tag_dist"]["tag1"]))
        print("R_L4C_ON_%s_SW_CORR=%s" % (sid, sw))
        print("R_L4C_GRP_%s_EPOCHS=%d" % (sid, gd["n_epoch_checks"]))
        print("R_L4C_GRP_%s_FULL=%d" % (sid, gd["n_full_epochs"]))
        print("R_L4C_GRP_%s_G1=%d" % (sid, gd["n_g1"]))
        print("R_L4C_GRP_%s_G1G2=%d" % (sid, gd["n_g1g2"]))
        print("R_L4C_GRP_%s_G1G2G3=%d" % (sid, gd["n_g1g2g3"]))
        print("R_L4C_GRP_%s_TRIGGER=%d" % (sid, gd["n_trigger"]))
        print("R_L4C_GRP_%s_CHILDREN=%d" % (sid, gd["n_group_children"]))
        print("R_L4C_GRP_%s_SPLIT_SINGLE=%d" % (sid, gd["n_split_single"]))
        print("R_L4C_GRP_%s_SPLIT_GROUP=%d" % (sid, gd["n_split_group"]))
        print("R_L4C_GRP_%s_HASH=%s" % (sid, gd["group_hash"]))
        for i, ev in enumerate(gd["group_splits"]):
            print("R_L4C_GRP_%s_EV_%d_AT=%d" % (sid, i, ev["split_at"]))
            print("R_L4C_GRP_%s_EV_%d_WINS=%s" % (
                sid, i, ",".join(str(w) for w in ev["epoch_wins"])))
            print("R_L4C_GRP_%s_EV_%d_RATIO=%.4f" % (sid, i, ev["ratio"]))
            print("R_L4C_GRP_%s_EV_%d_ANCHORS=%s" % (
                sid, i, ";".join(["0:%s" % ",".join(str(p) for p in ev["anchor_pids"]["0"]),
                                  "1:%s" % ",".join(str(p) for p in ev["anchor_pids"]["1"])])))
            print("R_L4C_GRP_%s_EV_%d_CHILDREN=%s" % (
                sid, i, ";".join("%d:%d:%d" % (c[0], c[1], c[2])
                                 for c in ev["children"])))
        print("R_L4C_BF_%s_N_CREATIONS=%d" % (sid, bf["n_creations"]))
        print("R_L4C_BF_%s_N_CREATIONS_WITH_BF=%d" % (sid, bf["n_creations_with_bf"]))
        print("R_L4C_BF_%s_WINDOW_TOTAL=%d" % (sid, bf["bf_window_total"]))
        print("R_L4C_BF_%s_AVG_PER_CREATION=%.4f" % (sid, bf["bf_avg_per_creation"]))
        print("R_L4C_BF_%s_AVG_PER_BF_CREATION=%.4f" % (sid, bf["bf_avg_per_bf_creation"]))
        print("R_L4C_BF_%s_C2_NONE=%d" % (sid, bf["bf_c2_dist"]["None"]))
        print("R_L4C_BF_%s_C2_0=%d" % (sid, bf["bf_c2_dist"]["0"]))
        print("R_L4C_BF_%s_C2_1=%d" % (sid, bf["bf_c2_dist"]["1"]))
        print("R_L4C_BF_%s_SPLIT_PARENT=%s" % (sid, spb))
        print("R_L4C_BF_%s_BFHASH=%s" % (sid, bf["bf_hash_all"]))
    print("R_L4C_ON_R1_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    seg = on_r1["seg_info"]
    print("R_L4C_ON_R1_SEG_INFO=%s" % ",".join(
        ("NA" if row["ratio"] is None else "%.4f" % row["ratio"]) for row in seg))
    print("R_L4C_ON_R1_SEG_N0=%s" % ",".join(str(row["n0"]) for row in seg))
    print("R_L4C_ON_R1_SEG_N1=%s" % ",".join(str(row["n1"]) for row in seg))
    print("R_L4C_ON_R1_ALIGN_RATE=%.4f" % on_r1["split_align"]["align_rate"])
    print("R_L4C_ON_R1_N_ALIGNED=%d" % on_r1["split_align"]["n_aligned"])
    ga = on_r1["group_align"]
    print("R_L4C_ON_R1_GROUP_ALIGN_RATE=%.4f" % ga["group_align_rate"])
    print("R_L4C_ON_R1_GROUP_N_ALIGNED=%d" % ga["n_aligned"])
    print("R_L4C_ON_R1_GROUP_N_SPLITS=%d" % ga["n_group_splits"])
    for i, ps in enumerate(ga["per_split"]):
        print("R_L4C_ON_R1_GROUP_EV_%d_SEG=%s" % (
            i, "NA" if ps["mode_segment"] is None else str(ps["mode_segment"])))
        print("R_L4C_ON_R1_GROUP_EV_%d_SEG_RATIO=%s" % (
            i, "NA" if ps["seg_ratio"] is None else "%.4f" % ps["seg_ratio"]))
        print("R_L4C_ON_R1_GROUP_EV_%d_ALIGNED=%d" % (i, ps["aligned"]))
    print("R_L4C_REPRO_RATIO=%d" % repro_ok)
    print("R_L4C_REPRO_ITEMS=%d" % len(repro_items))
    print("R_L4C_REPRO_DETAIL=%s" % repro_detail)
    print("R_L4C_NONSPLIT_EQ=%d" % nonsplit_eq)
    for sid in ALL_STREAMS:
        if sid == "R0B":
            continue            # docs/253 无 R0b 基准 -> NA
        print("R_L4C_NONSPLIT_%s=%d" % (sid, nonsplit_per[sid]["eq"]))
    for sid in ALL_STREAMS:
        print("R_L4C_NONSPLIT_%s_SPLIT=%d" % (sid, nonsplit_per[sid]["n_split"]))
        print("R_L4C_NONSPLIT_%s_DEV=%s" % (sid, nonsplit_per[sid]["dev_source"]))
    print("R_L4C_CRIT1_COMPOUND_EMERGES=%d" % crit1)
    print("R_L4C_CRIT1_N_SPLIT=%d" % on_r1["n_split"])
    print("R_L4C_CRIT1_SC2_TAGGED=%d" % on_r1["sc2_tagged"])
    print("R_L4C_CRIT1_COMPOUND_FRAC=%.4f" % on_r1["compound_frac"])
    print("R_L4C_CRIT2_ADOPT_NONRANDOM=%d" % crit2)
    print("R_L4C_CRIT2_SPURIOUS_FRAC=%.4f" % on_r1["spurious_split_frac"])
    print("R_L4C_CRIT2_AVG_POST_HITS=%.4f" % on_r1["avg_post_split_hits"])
    print("R_L4C_CRIT2_R0B_N_SPLIT=%d" % on_r0b["n_split"])
    print("R_L4C_CRIT3_FOUNDATION_KEEP=%d" % crit3)
    print("R_L4C_CRIT3_GIST_COV=%.4f" % on_r1["gist"]["cov"])
    print("R_L4C_CRIT4_PROMOTION_KEEP=%d" % crit4)
    print("R_L4C_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L4C_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L4C_GUARD_D246=%d" % guard246_ok)
    print("R_L4C_GUARD_D246_ITEMS=%d" % 12)
    print("R_L4C_GUARD_D246_PASSED=%d" % guard246_passed)
    print("R_L4C_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L4C_R0B_NOSPLIT=%d" % on_r0b["n_split"])
    print("R_L4C_R0B_NOSPLIT_OK=%d" % int(on_r0b["n_split"] == 0))
    print("R_L4C_VERDICT=%s" % verdict)
    print("R_L4C_VERDICT_NOTE=%s" % vnote)
    print("R_L4C_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
