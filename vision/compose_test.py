"""vision/compose_test.py — 组合性符号流实验（docs/235，交付 docs/235）。

把 docs/232 critical_point 的同款回路（对数域双 EWMA 背景预测 + 自适应阈值 + 事件累积 +
模式表）放进 4 级符号化运动场景（C0 单目标段式换档 / C1 双目标独立 / C2 上下文依赖 /
C3 嵌套），测模式表的结构涌现能否逼近组合抽象：条目是否自发采纳"上下文槽位"（第 3 签名
分量）来吸收上下文依赖，还是只用基线 2 分量并排建条目（平行、churn）。

签名空间（全由窗口事件掩码计算，纯 numpy）：
  c0 = 运动能量档（窗口事件总量分档，bins 同 critical_point [150,450,1000,2500]）
  c1 = 事件包络尺寸档（包围盒面积分档，bins [0,4000,12000,20000,30000]）
  c2 = 上下文槽位 = 上/下目标事件质心横向比较 sign(mean_x(y<60) - mean_x(y>=60))；
       场景保证目标 A 恒在上半、B 恒在下半（中心 y 35 vs 88，半径 10），故该符号 ==
       sign(A.x - B.x)；单目标/任一组空 -> None

模式表（变长 arity + 内容无关的槽位采纳机制，docs/235 §1.2）：
  1) 创建：仅按 s2=(c0,c1) 连续 k_consist 窗口一致 -> arity-2 条目（critical_point 同款）；
  2) 匹配：arity-3 (c0,c1,c2) 优先，arity-2 次之；匹配时记录窗口 MAE 与 c2 值到条目账本；
  3) 提升（组合事件）：arity-2 条目 E 满足全部条件时被拆分提升：命中 >= k_split；
     账本中 >= 2 个不同非 None c2 值各 >= k_consist 窗口；c2 条件中位 MAE 差 >= delta_rel
     （max >= (1+delta_rel)*min）——即上下文槽位对窗口预测误差携带信息量。提升 = 退休
     父条目，按每个合格 c2 值各建一个 arity-3 复合条目（继承命中 1）。
  4) churn：创建条目（含退休父条目）最终命中 < hits_min -> 不稳定（churn）。

度量（docs/235 §1.3）：M1 预测 MAE（mean|L - bg_fast|）；M2 SC1/SC2/复合条目占比/arity
分布；M3 churn；M4 上下文使用度（复合占比 + 槽位保真度（loop-c2 vs 真值上下文）+ 条目
纯度（arity-3 条目匹配窗口上真值上下文多数占比）+ C3 嵌套表示证据）。
判定（docs/235 §1.4 预注册冻结）：COMPOSABLE = C2 且 C3 均（MAE <= 1.3*C1 且复合占比 >= 0.5
且 churn < 0.3）；PARALLEL_ONLY = C2 或 C3 任一（MAE >= 2.0*C1 或复合占比 < 0.3）；否则
BOUNDARY。C0 为机制基线不参与判定。

统计外壳（docs/228/232 模式）：多种子（--n-seeds）、jitter 0.25、checkpoint
（ckpt_ct_<hash>.json，--resume 断点续跑）、JSON 归档（ct_<tag>.json）、mean±SD +
bootstrap 95% CI（2000 次，种子 20260828）。
安全纪律（docs/228/234 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_* 摘要块；
禁止把日志/JSON 原文送进上下文。

用法：
  python vision/compose_test.py --levels 20,21,22,23 --n-seeds 10 --tag main
  python vision/compose_test.py --levels 22 --n-seeds 1 --frames 240 --tag timing
  python vision/compose_test.py --diag 20,21,22,23 --seed 0      # 校准：场景统计（R_DIAG_*）
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER, N_BOOT, BOOT_SEED

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 环境梯度（docs/235 §1.1，校准 v5 冻结候选） ----------------
# lvcode ∈ {20,21,22,23}（与 critical_point 的 0-6 的 rng 流错开）
# 几何（校准后冻结）：A 恒在上半（y<68 事件组）、B 恒在下半；A/B 圆盘永不接触
# （无重叠混淆）；C2 中 B 的速度乘子 m1/m2 让两态联合能量落在同一粗分档而窗口能量
# 差 >= 30%（上下文槽位的信息量）；C3 中 A 快/慢段的 A 区能量档不同（嵌套可表示）。
LEVELS = {
    20: dict(name="c0_single_regime", n_targets=1,
             a_center=(80.0, 25.0), a_orbit=20.0, a_freq=0.18,
             regimes=[0.25, 3.0, 1.0], regime_bounds=[0, 80, 160, 240],
             noise=3.0, light="dips"),
    21: dict(name="c1_dual_independent", n_targets=2,
             a_center=(80.0, 36.0), a_orbit=13.0, a_freq=0.33,
             b_center=(80.0, 96.0), b_orbit=10.0, b_freq=0.25,
             b_m1=1.0, b_m2=3.2, b_rule="fixed",          # B 固定 m1（独立规则）
             noise=3.0, light="none"),
    22: dict(name="c2_context_dep", n_targets=2,
             a_center=(80.0, 36.0), a_orbit=13.0, a_freq=0.33,
             b_center=(80.0, 96.0), b_orbit=10.0, b_freq=0.25,
             b_m1=1.0, b_m2=3.2, b_rule="ctx",            # B 依 A.x vs B.x 取 m1/m2
             noise=3.0, light="none"),
    23: dict(name="c3_nested", n_targets=2,
             a_center=(80.0, 36.0), a_orbit=13.0, a_freq=0.33,
             a_radius_fast=20.0, c3_fast_bounds=[60, 220],  # A 快段：轨道半径放大
             b_center=(80.0, 96.0), b_orbit=10.0, b_freq=0.25,
             b_m1=1.0, b_m2=3.2, b_rule="nested",        # 快段内 ctx 依赖，快段外固定 m1
             noise=3.0, light="none"),
}
DISK_R = 10.0
OBJ_GRAY = 255.0
CTX_SPLIT_Y = 68.0          # 上/下事件组分割线（A 恒在上、B 恒在下，永不接触）
ENERGY_BINS = (150.0, 450.0, 2000.0, 3000.0)
UPPER_BINS = (420.0, 1000.0, 1800.0, 2800.0)   # A 区（上组）事件量分档：C2/C1 恒定、C3 慢/快段分离


def _band(v, bins):
    b = 0
    for i, bmax in enumerate(bins):
        if v >= bmax:
            b = i + 1
    return b


def make_scene(lvcode, seed, n_frames=240, width=160, height=120, fps=30, jitter=JITTER):
    """生成 (级,种子) 的灰度帧序列 + 逐窗口标签（真值上下文/regime/B 乘子，仅用于事后
    分析度量 M4，绝不进入回路决策）。确定性：rng 由 (seed, lvcode) 派生。"""
    cfg = LEVELS[lvcode]
    rng = np.random.default_rng(seed * 7919 + lvcode * 104729 + 13)
    noise_mult = rng.uniform(1 - jitter, 1 + jitter) if jitter > 0 else 1.0
    sigma = cfg["noise"] * noise_mult

    bg = np.zeros((height, width), np.float32)
    step = 24
    for y in range(0, height, step):
        for x in range(0, width, step):
            bg[y:y + step, x:x + step] = 64 if ((x // step) + (y // step)) % 2 == 0 else 96

    a_cx, a_cy = cfg["a_center"]
    a_r0 = cfg["a_orbit"]
    f_a = cfg["a_freq"]
    ph_a = rng.uniform(0, 2 * np.pi)
    n_t = cfg["n_targets"]
    has_b = n_t >= 2
    b_cx, b_cy = (cfg["b_center"] if has_b else (0.0, 0.0))
    b_r0 = cfg.get("b_orbit", 0.0)
    f_b = cfg.get("b_freq", 0.0)
    b_m1 = cfg.get("b_m1", 1.0)
    b_m2 = cfg.get("b_m2", 1.0)
    b_rule = cfg.get("b_rule", "fixed")
    regimes = cfg.get("regimes")
    reg_bounds = cfg.get("regime_bounds")
    a_r_fast = cfg.get("a_radius_fast")
    c3_fb = cfg.get("c3_fast_bounds", [])
    dips = [(45, 60), (130, 145)] if cfg["light"] == "dips" else []

    th_a = ph_a
    th_b = ph_b = rng.uniform(0, 2 * np.pi) if has_b else 0.0
    cur_reg = -1
    ctx_prev = 1
    frames = []
    per_frame = []          # (ctx, b_mult, a_regime) per frame
    for t in range(n_frames):
        # --- A 的位置（C0：regime 速度乘子；C3：快/慢段轨道半径） ---
        r_a = a_r0
        if regimes is not None:
            rg = 0
            for b in range(3):
                if reg_bounds[b] <= t < reg_bounds[b + 1]:
                    rg = b
            if rg != cur_reg:
                cur_reg = rg
                th_a = rng.uniform(0, 2 * np.pi)      # 换档相位重抽（L5 同款）
            spd = regimes[rg]
            th_a += 2 * np.pi * f_a * spd / fps
        else:
            if a_r_fast is not None:
                r_a = a_r_fast if (c3_fb[0] <= t < c3_fb[1]) else a_r0
            th_a += 2 * np.pi * f_a / fps
        ax = a_cx + r_a * np.cos(th_a)
        ay = a_cy + r_a * np.sin(th_a)

        # --- B 的乘子（依前一帧上下文；1 帧滞后，确定性） ---
        if has_b:
            if b_rule == "ctx":
                m = b_m1 if ctx_prev == 0 else b_m2
            elif b_rule == "nested":
                if c3_fb[0] <= t < c3_fb[1]:
                    m = b_m1 if ctx_prev == 0 else b_m2
                else:
                    m = b_m1
            else:
                m = b_m1
            th_b += 2 * np.pi * f_b * m / fps
            bx = b_cx + b_r0 * np.cos(th_b)
            by = b_cy + b_r0 * np.sin(th_b)
        else:
            bx = by = 0.0

        # --- 渲染 ---
        img = bg.copy()
        cv2_circle(img, int(ax), int(ay), DISK_R, OBJ_GRAY)
        if has_b:
            cv2_circle(img, int(bx), int(by), DISK_R, OBJ_GRAY)
        light = 1.0
        for lo, hi in dips:
            if lo <= t < hi:
                light = 0.5
        img = img * light
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))

        # --- 逐帧标签 ---
        if has_b:
            ctx = 0 if (ax - bx) < 0 else 1
        else:
            ctx = None
        if regimes is not None:
            a_regime = cur_reg
        elif a_r_fast is not None:
            a_regime = 1 if (c3_fb[0] <= t < c3_fb[1]) else 0
        else:
            a_regime = None
        per_frame.append((ctx, m if has_b else None, a_regime))
        if has_b:
            ctx_prev = ctx

    # --- 逐窗口标签（window=10 帧；ctx 按窗口内 mean(A.x-B.x) 符号） ---
    win_labels = []
    w = n_frames // 10
    for i in range(w):
        seg = per_frame[i * 10:(i + 1) * 10]
        ctxs = [s[0] for s in seg if s[0] is not None]
        ms = [s[1] for s in seg if s[1] is not None]
        rs = [s[2] for s in seg if s[2] is not None]
        if ctxs:
            c0 = sum(1 for c in ctxs if c == 0)
            ctx_w = 0 if c0 > len(ctxs) - c0 else 1
        else:
            ctx_w = None
        b_mult_w = float(np.mean(ms)) if ms else None
        regime_w = max(set(rs), key=rs.count) if rs else None
        win_labels.append(dict(ctx=ctx_w, b_mult=b_mult_w, a_regime=regime_w))
    return frames, win_labels


def cv2_circle(img, cx, cy, r, val):
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img[mask] = val
    return img


# ---------------- 组合回路（CPLoop 子类：变长 arity + 槽位采纳） ----------------
class CompLoop(CPLoop):
    """critical_point 同款预测回路（bg_fast/bg_slow/自适应阈值，机制零改动继承），
    模式表扩展为变长 arity：s2 创建、arity-3 优先匹配、c2 信息量驱动的提升。"""

    def __init__(self, window=10, k_consist=3, hits_min=3, persist_win=5,
                 k_split=5, delta_rel=0.30, energy_bins=ENERGY_BINS,
                 bbox_bins=UPPER_BINS, **kw):
        super().__init__(window=window, k_consist=k_consist, hits_min=hits_min,
                         persist_win=persist_win, energy_bins=energy_bins, **kw)
        self.bbox_bins = bbox_bins
        self.k_split = k_split
        self.delta_rel = delta_rel
        self._n_created = 0
        self._retired = 0
        self._promos = 0
        self._entry_log = []       # 所有创建过的条目（含退休父条目）
        self.match_trace = []      # (win, key) 实际匹配记录
        self.c2_trace = []         # 每窗口 loop 计算的 c2
        self.sig_trace = []        # 每窗口 (c0, c1, c2)（诊断用）
        self.bbox_trace = []       # 每窗口 A 区（上组）事件量（诊断/校准用）
        self.up_trace = []         # 每窗口上组（y<split）事件量（诊断/校准用）
        self.lo_trace = []         # 每窗口下组（y>=split）事件量（诊断/校准用）
        self._hist = []            # 最近 (s2, c2, energy) 窗口史（条目创建时回填账本）

    # ---- 签名：c0 能量档 / c1 包络尺寸档 / c2 上下文槽位 ----
    def _window_components(self, ev_win):
        n = int(ev_win.sum())
        if n < 10:
            return None, None, None
        c0 = _band(float(n), self.energy_bins)
        # c1 = A 区（上组 y<split）事件量分档：C2/C1 中 A 速度恒定 -> 档恒定；
        # C3 中 A 快/慢段速度不同 -> 档分离（嵌套可表示）
        up_n = float(ev_win[:int(CTX_SPLIT_Y), :].sum())
        c1 = _band(up_n, self.bbox_bins)
        up = ev_win[:int(CTX_SPLIT_Y), :]
        lo = ev_win[int(CTX_SPLIT_Y):, :]
        c2 = None
        if up.sum() >= 5 and lo.sum() >= 5:
            ux = float(np.mean(np.nonzero(up)[1]))
            lx = float(np.mean(np.nonzero(lo)[1]))
            c2 = 0 if (ux - lx) < 0 else 1
        return c0, c1, c2

    def _window_sig(self, ev_win):
        # 兼容占位（本类不使用 1 分量签名）
        c0, _, _ = self._window_components(ev_win)
        return None if c0 is None else (c0,)

    def _maybe_promote(self, s2, nd):
        """槽位采纳（组合事件）：arity-2 条目满足全部条件时提升为按 c2 拆分的 arity-3
        复合条目。条件：① 命中 >= k_split；② 账本中 >= 2 个非 None c2 值各 >= k_consist
        窗口；③ c2 条件**中位事件能量**差 >= delta_rel（max >= (1+delta_rel)*min）——
        即上下文槽位对该条目的窗口能量携带信息量（能量是签名空间所在的可观测）。"""
        if nd.get("retired"):
            return
        c2ene = nd["c2_ene"]
        quals = {k: v for k, v in c2ene.items()
                 if k is not None and len(v) >= self.k_consist}
        if len(quals) < 2:
            return
        meds = {k: float(np.median(v)) for k, v in quals.items()}
        if max(meds.values()) < (1 + self.delta_rel) * min(meds.values()):
            return
        if nd["hits"] < self.k_split:
            return
        # 提升：退休父条目 -> 按合格 c2 值各建 arity-3 复合条目
        nd["retired"] = True
        nd["retired_at"] = self._win
        self._retired += 1
        self._promos += 1
        del self.nodes[s2]
        for c2v in quals:
            key = (s2[0], s2[1], c2v)
            self.nodes[key] = dict(key=key, hits=1, created=self._win,
                                   last_active=self._win, arity=3, c2_ene={},
                                   retired=False)
            self._entry_log.append(self.nodes[key])
            self._n_created += 1

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

        c0, c1, c2 = self._window_components(ev_win)
        self.c2_trace.append(c2)
        self.sig_trace.append((c0, c1, c2))
        self.energy_trace.append(int(ev_win.sum()))
        up_n = int(ev_win[:int(CTX_SPLIT_Y), :].sum())
        lo_n = int(ev_win[int(CTX_SPLIT_Y):, :].sum())
        self.up_trace.append(up_n)
        self.lo_trace.append(lo_n)
        if ev_win.sum() >= 10:
            self.bbox_trace.append(float(ev_win[:int(CTX_SPLIT_Y), :].sum()))
        else:
            self.bbox_trace.append(0.0)
        learned = False
        if c0 is not None:
            s2 = (c0, c1)
            s3 = (c0, c1, c2) if c2 is not None else None
            matched = False
            # 匹配：arity-3 优先
            if s3 is not None and s3 in self.nodes and self.nodes[s3]["arity"] == 3:
                nd = self.nodes[s3]
                nd["hits"] += 1
                nd["last_active"] = self._win
                learned = True
                matched = True
                self.match_trace.append((self._win, s3))
            elif s2 in self.nodes and self.nodes[s2]["arity"] == 2:
                nd = self.nodes[s2]
                nd["hits"] += 1
                nd["last_active"] = self._win
                nd["c2_ene"].setdefault(c2, []).append(float(ev_win.sum()))
                learned = True
                matched = True
                self.match_trace.append((self._win, s2))
                self._maybe_promote(s2, nd)
            if not matched:
                if self._sig_seen.get(s2, 0) >= self.k_consist - 1:
                    # 创建 arity-2 条目；账本回填一致性窗口的 (c2, energy)（出生证据）
                    c2_ene_seed = {}
                    for hs, hc, he in self._hist:
                        if hs == s2 and hc is not None:
                            c2_ene_seed.setdefault(hc, []).append(he)
                    self.nodes[s2] = dict(key=s2, hits=1, created=self._win,
                                          last_active=self._win, arity=2,
                                          c2_ene=c2_ene_seed, retired=False)
                    self._entry_log.append(self.nodes[s2])
                    self._n_created += 1
                    learned = True
                    self.match_trace.append((self._win, s2))
                self._sig_seen[s2] = self._sig_seen.get(s2, 0) + 1
            for k in list(self._sig_seen.keys()):
                if k != s2:
                    self._sig_seen[k] = 0
        else:
            self.match_trace.append((self._win, None))
        if c0 is not None:
            self._hist.append((s2, c2, float(ev_win.sum())))
            if len(self._hist) > 4:
                self._hist.pop(0)
        if learned:
            self._n_learn += 1
        self.sc1_cum.append(self._n_created)
        self._win += 1
        self._frame_buf = []
        self._ev_win = None

    def finalize(self, n_windows, labels=None):
        if self._frame_buf:
            self._on_window()
        base = super().finalize(n_windows)
        # ---- 结构度量（docs/235 §1.3 精化：稳定条目 = 结束时仍在表内的条目；
        # 退休父条目是"平行记忆 -> 组合记忆"的中间形态，计入创建计数但不算稳定条目） ----
        log = self._entry_log
        cur = list(self.nodes.values())           # 结束时仍在表内的条目
        sc1 = len(log)
        sc2 = sum(1 for e in cur if e["hits"] >= self.hits_min)
        a2 = sum(1 for e in cur if e["arity"] == 2 and e["hits"] >= self.hits_min)
        a3 = sum(1 for e in cur if e["arity"] == 3 and e["hits"] >= self.hits_min)
        compound = a3 / max(1, sc2)
        churn = (sum(1 for e in log if e.get("retired")
                     and e["hits"] < self.hits_min)
                 + sum(1 for e in cur if e["hits"] < self.hits_min)) / max(1, sc1)
        sc_late = sum(1 for e in cur
                      if e["created"] >= n_windows / 2 and e["hits"] >= self.hits_min)
        sc4 = sum(1 for e in log
                  if e["last_active"] <= n_windows - 1 - self.persist_win)
        # ---- M4：槽位保真度（loop-c2 vs 真值上下文）+ 条目纯度 ----
        ctx_fid, ctx_n = None, 0
        if labels is not None:
            for w, lb in enumerate(labels):
                lc = self.c2_trace[w] if w < len(self.c2_trace) else None
                tc = lb.get("ctx")
                if lc is not None and tc is not None:
                    ctx_n += 1
                    ctx_fid = (ctx_fid or 0) + (1.0 if lc == tc else 0.0)
        fidelity = (ctx_fid / ctx_n) if ctx_n else 1.0
        # 条目纯度（仅稳定 arity-3 复合条目；其匹配窗口上真值上下文多数占比）/ regime 归属
        purity_vals, regime_a2, regime_a3 = [], [], []
        a3_ctx0 = sum(1 for e in cur if e["arity"] == 3 and e["hits"] >= self.hits_min
                      and e["key"][2] == 0)
        a3_ctx1 = sum(1 for e in cur if e["arity"] == 3 and e["hits"] >= self.hits_min
                      and e["key"][2] == 1)
        for e in cur:
            if e["hits"] < self.hits_min:
                continue
            wins = [w for w, k in self.match_trace if k == e["key"]]
            if not wins or labels is None:
                continue
            tcs = [labels[w]["ctx"] for w in wins if labels[w]["ctx"] is not None]
            rgs = [labels[w]["a_regime"] for w in wins
                   if labels[w]["a_regime"] is not None]
            if e["arity"] == 3 and tcs:
                c0n = sum(1 for c in tcs if c == 0)
                purity_vals.append(max(c0n, len(tcs) - c0n) / len(tcs))
            if rgs:
                r0 = sum(1 for r in rgs if r == 0)
                maj = 0 if r0 >= len(rgs) - r0 else 1
                if e["arity"] == 2:
                    regime_a2.append(maj)
                else:
                    regime_a3.append(maj)
        purity = float(np.mean(purity_vals)) if purity_vals else 1.0
        out = dict(base)
        out.update({
            "sc1": sc1, "sc2": sc2, "sc_late": sc_late, "sc4": sc4,
            "arity2_stable": a2, "arity3_stable": a3,
            "compound_frac": round(compound, 4),
            "churn_frac": round(churn, 4),
            "n_promo": self._promos,
            "ctx_fidelity": round(fidelity, 4),
            "ctx_purity": round(purity, 4),
            "ctx_n_windows": ctx_n,
            "arity3_ctx0_stable": a3_ctx0, "arity3_ctx1_stable": a3_ctx1,
            "c3_regime_maj_arity2": round(float(np.mean(regime_a2)), 4) if regime_a2 else None,
            "c3_regime_maj_arity3": round(float(np.mean(regime_a3)), 4) if regime_a3 else None,
            "n_created_total": self._n_created,
            "n_retired": self._retired,
            "entry_log": [{**{k: v for k, v in e.items() if k != "c2_ene"},
                           "key": list(e["key"])} for e in log],
            "c2_trace": [None if c is None else int(c) for c in self.c2_trace],
            "match_summary": {
                "arity3": sum(1 for _, k in self.match_trace
                              if k is not None and len(k) == 3),
                "arity2": sum(1 for _, k in self.match_trace
                              if k is not None and len(k) == 2),
                "none": sum(1 for _, k in self.match_trace if k is None),
            },
        })
        return out


def run_level(lvcode, seed, n_frames, width, height, fps, window, jitter, loop_kwargs=None):
    frames, labels = make_scene(lvcode, seed, n_frames=n_frames, width=width,
                                height=height, fps=fps, jitter=jitter)
    loop = CompLoop(window=window, **(loop_kwargs or {}))
    for g in frames:
        loop.step(g)
    n_windows = max(1, n_frames // window)
    out = loop.finalize(n_windows, labels)
    out["seed"] = seed
    out["level"] = lvcode
    out["frames"] = n_frames
    return out


# ---------------- 校准诊断（R_DIAG_* 摘要，安全模式） ----------------
def diag(levels, seed, n_frames=240, width=160, height=120, fps=30,
         window=10, jitter=JITTER):
    for lv in levels:
        frames, labels = make_scene(lv, seed, n_frames=n_frames, width=width,
                                    height=height, fps=fps, jitter=jitter)
        loop = CompLoop(window=window)
        for g in frames:
            loop.step(g)
        n_windows = max(1, n_frames // window)
        out = loop.finalize(n_windows, labels)
        print("R_DIAG_LEVEL=%d" % lv)
        print("R_DIAG_SEED=%d" % seed)
        print("R_DIAG_MAE=%.6f" % out["mae_mean"])
        print("R_DIAG_SC1=%d" % out["sc1"])
        print("R_DIAG_SC2=%d" % out["sc2"])
        print("R_DIAG_COMP=%.4f" % out["compound_frac"])
        print("R_DIAG_CHURN=%.4f" % out["churn_frac"])
        print("R_DIAG_PROMO=%d" % out["n_promo"])
        print("R_DIAG_A2=%d" % out["arity2_stable"])
        print("R_DIAG_A3=%d" % out["arity3_stable"])
        print("R_DIAG_CTXFID=%.4f" % out["ctx_fidelity"])
        print("R_DIAG_CTXPUR=%.4f" % out["ctx_purity"])
        # 按真值上下文的窗口能量/MAE（校准：同档约束 / MAE 差约束 / 段长）
        ctx_energy = {0: [], 1: []}
        ctx_mae = {0: [], 1: []}
        seg = {0: [], 1: []}
        prev_ctx, seg_len = None, 0
        for w, lb in enumerate(labels):
            tc = lb.get("ctx")
            if tc is not None:
                ctx_energy[tc].append(loop.energy_trace[w])
                ctx_mae[tc].append(out["mae_trace"][w])
                if tc == prev_ctx:
                    seg_len += 1
                else:
                    if prev_ctx is not None:
                        seg[prev_ctx].append(seg_len)
                    prev_ctx, seg_len = tc, 1
        if prev_ctx is not None:
            seg[prev_ctx].append(seg_len)
        for c in (0, 1):
            print("R_DIAG_CTX%d_ENERGY=%.1f" % (c, float(np.mean(ctx_energy[c]))
                                                if ctx_energy[c] else -1.0))
            print("R_DIAG_CTX%d_ENERGYMED=%.1f" % (c, float(np.median(ctx_energy[c]))
                                                   if ctx_energy[c] else -1.0))
            print("R_DIAG_CTX%d_MAE=%.6f" % (c, float(np.mean(ctx_mae[c]))
                                             if ctx_mae[c] else -1.0))
            print("R_DIAG_CTX%d_MAEMED=%.6f" % (c, float(np.median(ctx_mae[c]))
                                                if ctx_mae[c] else -1.0))
            print("R_DIAG_CTX%d_SEG=%s" % (c, ",".join(str(v) for v in seg[c][:12])))
        if ctx_energy[0] and ctx_energy[1]:
            r = max(float(np.median(ctx_energy[0])), float(np.median(ctx_energy[1]))) / \
                max(1e-9, min(float(np.median(ctx_energy[0])), float(np.median(ctx_energy[1]))))
            print("R_DIAG_ENERGYMED_RATIO=%.4f" % r)
        # A 区事件量（校准 c1 分档）
        bboxes = [b for b in loop.bbox_trace if b > 0]
        if bboxes:
            print("R_DIAG_BBOX_MEAN=%.1f" % float(np.mean(bboxes)))
            print("R_DIAG_BBOX_MIN=%.1f" % float(np.min(bboxes)))
            print("R_DIAG_BBOX_MAX=%.1f" % float(np.max(bboxes)))
            print("R_DIAG_BBOX_BANDS=%s" % ",".join(
                str(_band(float(v), UPPER_BINS)) for v in sorted(set(
                    int(x) for x in bboxes))))
        print("R_DIAG_ENERGY=%s" % ",".join(str(int(v)) for v in loop.energy_trace))
        print("R_DIAG_THETA=%s" % ",".join("%.3f" % v for v in loop.theta_trace))
        print("R_DIAG_SIG=%s" % ";".join(
            ("%s,%s,%s" % (("N" if c0 is None else str(c0)),
                           ("N" if c1 is None else str(c1)),
                           ("N" if c2 is None else str(c2))))
            for (c0, c1, c2) in loop.sig_trace))
        print("R_DIAG_UPPER=%s" % ",".join(str(int(v)) for v in loop.up_trace))
        print("R_DIAG_LOWER=%s" % ",".join(str(int(v)) for v in loop.lo_trace))
        print("R_DIAG_ENTRIES=%s" % ";".join(
            "%s|%d|%d|%d" % (",".join(str(k) for k in e["key"]), e["arity"],
                             e["hits"], e["created"]) for e in out["entry_log"]))
        print("R_DIAG_C0_BANDS=%s" % ",".join(str(_band(float(v), ENERGY_BINS))
                                              for v in sorted(set(
                                                  [float(x) for x in loop.energy_trace]))))
    return 0


# ---------------- 统计外壳（docs/228/232 模式） ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="20,21,22,23")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="ct")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--diag", default="",
                    help="校准模式：逗号分隔的级代码 + --seed，只打 R_DIAG_* 行")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.diag:
        lvs = [int(x) for x in args.diag.split(",") if x.strip()]
        return diag(lvs, args.seed, n_frames=args.frames, width=args.width,
                    height=args.height, fps=args.fps, window=args.window,
                    jitter=args.jitter)

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    loop_cfg = {"alpha_fast": 0.5, "alpha_slow": 0.03, "thresh": 0.15,
                "deadband": 0.015, "k_theta": 6.0, "k_db": 1.5,
                "thresh_max": 0.6, "db_max": 0.15, "k_consist": 3,
                "hits_min": 3, "persist_win": 5, "k_split": 5, "delta_rel": 0.30,
                "energy_bins": list(ENERGY_BINS), "bbox_bins": list(UPPER_BINS)}
    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "fps": args.fps, "size": [args.width, args.height],
           "window": args.window, "jitter": args.jitter, "tag": args.tag,
           "scene": {str(k): v for k, v in LEVELS.items()},
           "loop": loop_cfg}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_ct_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    for lv in levels:
        for seed in seeds:
            key = "%d_%d" % (lv, seed)
            if key in per_unit:
                continue
            r = run_level(lv, seed, args.frames, args.width, args.height,
                          args.fps, args.window, args.jitter, loop_kwargs=loop_cfg)
            per_unit[key] = r
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)

    # ---- 跨种子聚合 + 判定 ----
    levels_out = {}
    for lv in levels:
        rs = [per_unit["%d_%d" % (lv, s)] for s in seeds]
        def col(key):
            return [r[key] for r in rs]
        mae_m, mae_sd = mean_sd(col("mae_mean"))
        sc1_m, _ = mean_sd(col("sc1"))
        sc2_m, sc2_sd = mean_sd(col("sc2"))
        a2_m, _ = mean_sd(col("arity2_stable"))
        a3_m, _ = mean_sd(col("arity3_stable"))
        comp_m, comp_sd = mean_sd(col("compound_frac"))
        churn_m, _ = mean_sd(col("churn_frac"))
        promo_m, _ = mean_sd(col("n_promo"))
        fid_m, _ = mean_sd(col("ctx_fidelity"))
        pur_m, _ = mean_sd(col("ctx_purity"))
        levels_out[lv] = {
            "name": LEVELS[lv]["name"],
            "per_seed": rs,
            "mean_sd": {k: list(mean_sd(col(k))) for k in
                        ("mae_mean", "sc1", "sc2", "arity2_stable", "arity3_stable",
                         "compound_frac", "churn_frac", "n_promo", "ctx_fidelity",
                         "ctx_purity")},
            "rgma2": (float(np.mean([v for v in col("c3_regime_maj_arity2")
                                     if v is not None])) if any(
                v is not None for v in col("c3_regime_maj_arity2")) else None),
            "rgma3": (float(np.mean([v for v in col("c3_regime_maj_arity3")
                                     if v is not None])) if any(
                v is not None for v in col("c3_regime_maj_arity3")) else None),
            "bootstrap_ci95": {"compound": list(bootstrap_ci(col("compound_frac"))),
                               "mae": list(bootstrap_ci(col("mae_mean")))},
        }
    mae1 = levels_out[21]["mean_sd"]["mae_mean"][0] if 21 in levels_out else None
    for lv in levels:
        lo = levels_out[lv]
        mae = lo["mean_sd"]["mae_mean"][0]
        ratio = mae / mae1 if mae1 else 1.0
        lo["err_ratio_c1"] = round(ratio, 4)
        lo["checks"] = {
            "ok_err": int(ratio <= 1.3),
            "ok_comp": int(lo["mean_sd"]["compound_frac"][0] >= 0.5),
            "ok_churn": int(lo["mean_sd"]["churn_frac"][0] < 0.3),
        }
        if lv == levels[0]:
            lo["class"] = "BASELINE"
        elif lo["checks"]["ok_err"] and lo["checks"]["ok_comp"] and lo["checks"]["ok_churn"]:
            lo["class"] = "COMPOSABLE"
        elif ratio >= 2.0 or lo["mean_sd"]["compound_frac"][0] < 0.3:
            lo["class"] = "PARALLEL"
        else:
            lo["class"] = "BOUNDARY"

    # ---- 全局判定（docs/235 §1.4 预注册；C0=20 基线、C1=21 基准、C2=22、C3=23） ----
    if all(k in levels_out for k in (21, 22, 23)):
        c2, c3 = levels_out[22], levels_out[23]
        comp_all = all(c2["checks"][k] and c3["checks"][k]
                       for k in ("ok_err", "ok_comp", "ok_churn"))
        par_any = any([c2["err_ratio_c1"] >= 2.0, c3["err_ratio_c1"] >= 2.0,
                       c2["mean_sd"]["compound_frac"][0] < 0.3,
                       c3["mean_sd"]["compound_frac"][0] < 0.3])
        if comp_all:
            verdict = "COMPOSABLE"
            vnote = "C2/C3 err<=1.3xC1 and compound>=0.5 and churn<0.3"
        elif par_any:
            verdict = "PARALLEL_ONLY"
            vnote = "C2/C3 err>=2.0xC1 or compound<0.3"
        else:
            verdict = "BOUNDARY"
            vnote = "neither COMPOSABLE nor PARALLEL_ONLY; see numbers"
        vdict = {"verdict": verdict, "note": vnote,
                 "mae_c1": round(float(mae1), 6),
                 "mae_c2": round(c2["mean_sd"]["mae_mean"][0], 6),
                 "mae_c3": round(c3["mean_sd"]["mae_mean"][0], 6),
                 "compound_c2": c2["mean_sd"]["compound_frac"][0],
                 "compound_c3": c3["mean_sd"]["compound_frac"][0],
                 "churn_c2": c2["mean_sd"]["churn_frac"][0],
                 "churn_c3": c3["mean_sd"]["churn_frac"][0]}
    else:
        verdict = "INCOMPLETE"
        vnote = "need levels 21,22,23 for verdict"
        vdict = {"verdict": verdict, "note": vnote}

    out = {
        "artifact": "compose_test",
        "doc_ref": "docs/231, docs/232, docs/235",
        "config": cfg,
        "levels": {str(k): v for k, v in levels_out.items()},
        "verdict": vdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "ct_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定）----
    print("R_LEVELS=%d" % len(levels))
    print("R_SEEDS=%d" % len(seeds))
    for i, lv in enumerate(levels):
        lo = levels_out[lv]
        ms = lo["mean_sd"]
        print("R_C%d_NAME=%s" % (i, lo["name"]))
        print("R_C%d_MAE=%.6f" % (i, ms["mae_mean"][0]))
        print("R_C%d_MAE_SD=%.6f" % (i, ms["mae_mean"][1]))
        print("R_C%d_SC1=%.4f" % (i, ms["sc1"][0]))
        print("R_C%d_SC2=%.4f" % (i, ms["sc2"][0]))
        print("R_C%d_SC2_SD=%.4f" % (i, ms["sc2"][1]))
        print("R_C%d_A2=%.4f" % (i, ms["arity2_stable"][0]))
        print("R_C%d_A3=%.4f" % (i, ms["arity3_stable"][0]))
        print("R_C%d_COMP=%.4f" % (i, ms["compound_frac"][0]))
        print("R_C%d_COMP_SD=%.4f" % (i, ms["compound_frac"][1]))
        print("R_C%d_CHURN=%.4f" % (i, ms["churn_frac"][0]))
        print("R_C%d_PROMO=%.4f" % (i, ms["n_promo"][0]))
        print("R_C%d_CTXFID=%.4f" % (i, ms["ctx_fidelity"][0]))
        print("R_C%d_CTXPUR=%.4f" % (i, ms["ctx_purity"][0]))
        print("R_C%d_ERRRATIO=%.4f" % (i, lo["err_ratio_c1"]))
        print("R_C%d_MAE_LO=%.6f" % (i, lo["bootstrap_ci95"]["mae"][0]))
        print("R_C%d_MAE_HI=%.6f" % (i, lo["bootstrap_ci95"]["mae"][1]))
        print("R_C%d_COMP_LO=%.4f" % (i, lo["bootstrap_ci95"]["compound"][0]))
        print("R_C%d_COMP_HI=%.4f" % (i, lo["bootstrap_ci95"]["compound"][1]))
        rg2 = lo["rgma2"]
        rg3 = lo["rgma3"]
        print("R_C%d_RGMA2=%s" % (i, "NA" if rg2 is None else "%.4f" % rg2))
        print("R_C%d_RGMA3=%s" % (i, "NA" if rg3 is None else "%.4f" % rg3))
        print("R_C%d_OKERR=%d" % (i, lo["checks"]["ok_err"]))
        print("R_C%d_OKCOMP=%d" % (i, lo["checks"]["ok_comp"]))
        print("R_C%d_OKCHURN=%d" % (i, lo["checks"]["ok_churn"]))
        print("R_C%d_CLASS=%s" % (i, lo["class"]))
    v = out["verdict"]
    print("R_VERDICT=%s" % verdict)
    print("R_VERDICT_NOTE=%s" % vnote)
    for k in ("mae_c1", "mae_c2", "mae_c3"):
        print("R_%s=%.6f" % (k.upper(), v.get(k, -1.0)))
    for k in ("compound_c2", "compound_c3", "churn_c2", "churn_c3"):
        print("R_%s=%.4f" % (k.upper(), v.get(k, -1.0)))
    print("R_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
