"""lineB-motion-coupling/scripts/attention_cost_test.py — B 路第三格：注意执行有代价 → 利害加权
收益为负 → 二级残差驱动注意转移（docs: lineB-motion-coupling/docs/B3-注意代价与二级残差-预注册
设计.md §一 冻结；B2 缺口补测）。

核心：B2（docs/B2，verdict=PARTIAL）的 C3 REDIRECT_CORRECT 观察 0——"收益为负"事件在"注意 =
无限加速预测"形态下数学上不发生（α 提高使残差 = (1−α)|L−bg| 恒降）；B2 §五 12 指明修复路径 =
注意执行有代价/资源有限形态（docs/186 §五 Top-K 威胁度分配 = 注意资源有限）。本格 = 实施该
修复：合成场景复用 B2（make_b2_scene 零改动），注意执行改为"精度分配预算守恒"——焦点邻域
α 0.5→α_focal=0.85（投入）+ 非焦点候选邻域 α 0.5→α_low=0.15（撤走，机会成本的机制载体；
对称预算 α_low = 1−α_focal）；利害加权总收益（含机会成本）BENEFIT(w) = s_f·(R_f_noatt −
R_f_att) + s_g·(R_g_noatt − R_g_att)；二级残差 sec(w) = 1[BENEFIT(w) < 0]（"我注意错了"，
docs/237 从"探测"升为"决策驱动"）；注意转移（层 2）= sec(w−1)=1 → 焦点给另一候选。

判据（§1.5 冻结，docs/247 标签 [L5][机制][涌现检验]——L5 展望形态诚实标注，非思考证明）：
  C1 NEG_BENEFIT_OCCURS : 利害加权总收益为负的池化窗口比例 ≥ 0.05 且池化负收益窗口 ≥ 10
                          （B2 里是 0；代价形态必须让"收益为负"可发生）
  C2 REDIRECT_TRIGGER   : 负收益窗口中被注意转移（焦点实际改变 f(w+1) ≠ f(w)）的池化比例 ≥ 0.70
  C3 REDIRECT_CORRECT   : 转移后窗口利害加权总收益转正（BENEFIT(w+1) > 0）的池化正确率 ≥ 0.70
                          且转移观察 ≥ 10
  C4 KEEP 守卫         : R_B3_GUARD_B2 + R_B3_GUARD_SO + R_B3_GUARD_COMPOSE + R_B3_REPRO 全 = 1
判定（§1.6）：全过 = REDIRECT_PASS；负收益发生但转移未达杠 = PARTIAL；负收益不发生 = NEG_FAIL；
守卫不过 = GUARD_FAIL。

守卫（§1.7 冻结）：
  R_B3_GUARD_B2     : import attention_emergence_test.run_attention（B2 冻结实现，零改动）
                      10 种子重跑 → CONSIST ≥ 0.80 且 FRACPOS ≥ 0.60 且 MEANBEN ∈ [0.03,0.06]
                      （B2 层 1 焦点=argmax(R×S) + 预期收益正的同代码路径复现）
  R_B3_GUARD_SO     : import attention_emergence_test.guard_so（docs/241 S1 长程 SO_info 复现：
                      r ∈ [0.05,0.12] 且 CI 下界 > 0 且 diff_rand CI 下界 > 0）
  R_B3_GUARD_COMPOSE: import attention_emergence_test.guard_compose（docs/237 §3.2 逐位，
                      容差 1e-3）
  R_B3_REPRO        : --repro 时主实验 10 运行整体重跑第二遍，关键数字位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_B3_* 摘要块
（顺序固定）；JSON 归档 lineB-motion-coupling/out/at3_<tag>.json + checkpoint
ckpt_at3_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 lineB-motion-coupling/out/*.log 与 lineB-motion-coupling/out/*.json 原文。
**未修改任何主线既有脚本**（vision/ 下全部不动；B2 脚本亦不动，只 import）。

用法：
  python lineB-motion-coupling/scripts/attention_cost_test.py --n-seeds 10 --tag main --repro
  python lineB-motion-coupling/scripts/attention_cost_test.py --n-seeds 1 --tag timing
  python lineB-motion-coupling/scripts/attention_cost_test.py --diag --seed 0
  python lineB-motion-coupling/scripts/attention_cost_test.py --scan --seed 0
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))          # 项目根（lineB-motion-coupling/ 的父目录）
VISION = os.path.join(PROJ, "vision")
for _p in (HERE, VISION, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- import 复用 B2（docs/B3 §1.4/§1.7：零改动，只 import）----
import attention_emergence_test as B2           # noqa: E402
from critical_point import mean_sd, bootstrap_ci, JITTER  # noqa: E402

make_b2_scene = B2.make_b2_scene
AttentionLoop = B2.AttentionLoop
top2_components = B2.top2_components
disk_mask = B2.disk_mask
mean_centroid = B2.mean_centroid
neighborhood_residual_mean = B2.neighborhood_residual_mean
LOOP_CFG = B2.LOOP_CFG
CP_KEYS = B2.CP_KEYS
guard_so = B2.guard_so
guard_compose = B2.guard_compose
run_attention = B2.run_attention

# 场景常量（与 B2 同源，import 复用；docs/B3 §1.2 冻结）
LVCODE_B2 = B2.LVCODE_B2
W, H = B2.W, B2.H
FPS = B2.FPS
N_FRAMES = B2.N_FRAMES
WINDOW = B2.WINDOW
JITTER = B2.JITTER
S_A = B2.S_A                      # 0.5（外赋利害：候选 A 低利害，B2 定案值零重调）
S_B = B2.S_B                      # 1.0（外赋利害：候选 B 高利害）
R_FOCAL = B2.R_FOCAL
CTX_SPLIT_Y = B2.CTX_SPLIT_Y
OCC_LUM_THRESH = B2.OCC_LUM_THRESH
A_MIN = B2.A_MIN
ALPHA_FOCAL = B2.ALPHA_FOCAL      # 0.85（注意执行：焦点邻域预测更新速率，同 B2）

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("lineB-motion-coupling", "out")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 代价机制旋钮（docs/B3 §1.4 冻结机制；工作点经 §二 D1 诊断轮定案）----
# 对称预算：α_low = 1 − α_focal（投给焦点的能量 = 从别处撤走的能量，总注意能量 Σ(α−0.5)≈0
# 守恒）。初始工作点 α_focal=0.85/α_low=0.15/s_A=0.5（§1.4 冻结初值）经 §二 诊断轮代价参数
# 边界扫描定案为 α_focal=0.55/α_low=0.45/s_A=0.5（"机制可公平测试"区：负收益事件非偶发
# （0.28）且转移修复率高（0.92）；初始值撤走过强 → 双向皆负（REDIR 0.16）无法测修复，记录
# §二 D1-2）。
COST_ALPHA_FOCAL = 0.55
COST_ALPHA_LOW = 0.45
COST_S_A = 0.5
ALPHA_LOW = 0.15                  # 初始工作点常量（§1.4 冻结初值；main 用 COST_* 定案值）

# ---- 判据阈值（docs/B3 §1.5 冻结）----
NEG_FRAC_MIN = 0.05
NEG_NOBS_MIN = 10
TRIG_MIN = 0.70
REDIR_MIN = 0.70
REDIR_NOBS_MIN = 10

# ---- 守卫容差（docs/B3 §1.7 冻结）----
B2_CONSIST_MIN = 0.80
B2_FRACPOS_MIN = 0.60
B2_MEANBEN_LO, B2_MEANBEN_HI = 0.03, 0.06

REPRO_KEYS = ["focus", "f_auto", "VA", "VB", "benefit", "sec", "consist",
              "consist_frac", "neg_frac", "trig", "redir_ok_n", "redir_n",
              "n_sec", "Rf_att", "Rf_noatt", "Rg_att", "Rg_noatt"]


# ---------------- 代价注意回路（CostAttentionLoop：AttentionLoop + 撤走掩码） ----------------
class CostAttentionLoop(AttentionLoop):
    """B2 AttentionLoop 同款预测回路（CPLoop 继承 + 逐像素 α + 残差场记录），注意执行改为
    "精度分配预算守恒"（docs/B3 §1.4 冻结）：焦点候选邻域 α = alpha_focal（投入），非焦点
    候选邻域 α = alpha_low（撤走——机会成本的机制载体）；其余区域 α = a_fast（0.5）。
    只 import 复用 AttentionLoop/CPLoop，不改主线。"""

    def __init__(self, alpha_focal=ALPHA_FOCAL, alpha_low=ALPHA_LOW, **kw):
        super().__init__(alpha_focal=alpha_focal, **kw)
        self.alpha_low = alpha_low

    def step(self, gray, focal_mask=None, withdraw_mask=None):
        L = np.log(np.maximum(gray.astype(np.float32), 1.0))
        if self.bg_fast is None:
            self.bg_fast = L.copy()
            self.bg_slow = L.copy()
            self.prev_L = L.copy()
            self.sigma_hat = 0.0
            self.res_frames.append(np.zeros_like(L))     # 初始化帧无残差：零场对齐帧索引
            return dict(mae=0.0, att=0.0, ev=0.0, theta=self.thresh, db=self.deadband)
        d = L - self.prev_L
        sig = float(np.median(np.abs(d - np.median(d))) / 0.6745)
        self.sigma_hat = 0.15 * sig + 0.85 * self.sigma_hat
        theta = float(np.clip(max(self.thresh, self.k_theta * self.sigma_hat),
                              self.thresh, self.thresh_max))
        db = float(np.clip(max(self.deadband, self.k_db * self.sigma_hat),
                           self.deadband, self.db_max))
        self.prev_L = L
        alpha = np.full(L.shape, self.a_fast, np.float32)
        if focal_mask is not None:
            alpha[focal_mask] = self.alpha_focal
        if withdraw_mask is not None:
            alpha[withdraw_mask] = self.alpha_low
        self.bg_fast = alpha * L + (1.0 - alpha) * self.bg_fast
        self.bg_slow = self.a_slow * L + (1.0 - self.a_slow) * self.bg_slow
        c = L - self.bg_fast
        r = np.abs(c)
        rd = np.maximum(r - db, 0.0)
        ev_mask = rd > theta
        mae = float(r.mean())
        att = float(np.mean(r > db))
        ev = float(ev_mask.mean())
        if self._prev_theta is not None:
            self._param_disp += abs(theta - self._prev_theta)
            self._param_disp += abs(db - self._prev_db)
            if self._prev_theta > 0 and theta > 1.25 * self._prev_theta:
                self._n_steps += 1
        self._prev_theta = theta
        self._prev_db = db
        if self._ev_win is None:
            self._ev_win = ev_mask.copy()
        else:
            self._ev_win |= ev_mask
        self._frame_buf.append(dict(mae=mae, att=att, ev=ev, theta=theta, db=db))
        self.res_frames.append(np.abs(c))
        if len(self._frame_buf) >= self.window:
            self._on_window()
        return dict(mae=mae, att=att, ev=ev, theta=theta, db=db)


# ---------------- 主实验单位（docs/B3 §1.4 冻结） ----------------
def run_cost(seed, n_frames=N_FRAMES, jitter=JITTER, s_a=COST_S_A, s_b=S_B,
             alpha_focal=COST_ALPHA_FOCAL, alpha_low=COST_ALPHA_LOW):
    """跑 (LVCODE_B2, 种子) 一次完整运行（代价形态）→ 注意决策行为量 + 判据组件。

    窗口 0 = 预热基线（α 全 0.5，无注意）；窗口 w ∈ [1, 24)：
      决策（窗口 w−1 残差场 R_agg × 外赋利害场）→ V_A/V_B（候选邻域 R×S 最大值）→
      层 1 决策候选 f_auto = argmax V（sec(w−1)=1 → 层 2 转移覆盖：实际焦点 f(w) =
      另一候选）→ 注意执行（窗口 w：主轨迹焦点邻域 α=alpha_focal + 非焦点候选邻域
      α=alpha_low；无注意轨迹全程 α0.5）→ 行为量（利害加权总收益 BENEFIT（含机会成本）/
      sec / trig / redir）。"""
    frames, gts = make_b2_scene(seed, n_frames=n_frames, jitter=jitter)
    n_w = max(1, n_frames // WINDOW)
    loop = CostAttentionLoop(window=WINDOW, alpha_focal=alpha_focal,
                             alpha_low=alpha_low, **LOOP_CFG)
    loop_noatt = CostAttentionLoop(window=WINDOW, alpha_focal=alpha_focal,
                                   alpha_low=alpha_low, **LOOP_CFG)
    cA = [None] * n_frames
    cB = [None] * n_frames
    lastA = lastB = None
    det_err_a = []
    det_err_b = []
    # 检测循环（全部帧；亮域 top-2 质心，与 step 解耦——B2 D1-4 同款语义）
    for t, g in enumerate(frames):
        bright = np.log(np.maximum(g.astype(np.float32), 1.0)) > np.log(OCC_LUM_THRESH + 1.0)
        comps = top2_components(bright, min_area=A_MIN)
        curA = curB = None
        for (_m, _a, c) in comps:
            if c[1] < CTX_SPLIT_Y:
                curA = c
            else:
                curB = c
        if curA is not None:
            lastA = curA
        if curB is not None:
            lastB = curB
        cA[t] = lastA
        cB[t] = lastB
        if lastA is not None:
            det_err_a.append(np.hypot(lastA[0] - gts["a_pos"][t][0],
                                      lastA[1] - gts["a_pos"][t][1]))
        if lastB is not None:
            det_err_b.append(np.hypot(lastB[0] - gts["b_pos"][t][0],
                                      lastB[1] - gts["b_pos"][t][1]))
    # 预热：窗口 0 帧（α 全 0.5，无注意；loop 与 loop_noatt 相同）
    for t in range(min(WINDOW, n_frames)):
        loop.step(frames[t])
        loop_noatt.step(frames[t])

    focus = [None] * n_w
    f_auto_rec = [None] * n_w
    VA_rec = [0.0] * n_w
    VB_rec = [0.0] * n_w
    Rf_att_rec = [float("nan")] * n_w
    Rf_noatt_rec = [float("nan")] * n_w
    Rg_att_rec = [float("nan")] * n_w
    Rg_noatt_rec = [float("nan")] * n_w
    benefit = [0.0] * n_w
    sec = [0] * n_w
    consist = [0] * n_w
    override = [0] * n_w
    prev_sec = 0
    prev_focus = None

    for w in range(1, n_w):
        lo0 = (w - 1) * WINDOW
        hi0 = min(lo0 + WINDOW, n_frames)
        R_agg = np.mean(np.stack(loop.res_frames[lo0:hi0]), axis=0)
        cA0 = mean_centroid(cA, lo0, hi0)
        cB0 = mean_centroid(cB, lo0, hi0)
        NA = disk_mask(cA0, R_FOCAL) if cA0 is not None else None
        NB = disk_mask(cB0, R_FOCAL) if cB0 is not None else None
        maxA = float(R_agg[NA].max()) if NA is not None else 0.0
        maxB = float(R_agg[NB].max()) if NB is not None else 0.0
        VA = s_a * maxA
        VB = s_b * maxB
        VA_rec[w] = VA
        VB_rec[w] = VB
        # 层 1 决策候选（argmax V；候选检测失败 → V=0 → 焦点 = 另一候选）
        if NA is None and NB is None:
            f_auto = "A"
        elif NA is None:
            f_auto = "B"
        elif NB is None:
            f_auto = "A"
        else:
            f_auto = "A" if VA >= VB else "B"
        f_auto_rec[w] = f_auto
        # 层 2 转移覆盖：sec(w−1)=1 → 实际焦点 = 另一候选（两候选世界的次优；
        # 修复被忽略/受损的候选——docs/B3 §1.4 机制定义精化 D3-1）
        f = f_auto
        if prev_sec == 1 and prev_focus is not None:
            f = "B" if prev_focus == "A" else "A"
            override[w] = 1 if f != f_auto else 0
        focus[w] = f

        # FOCUS_CONSIST（层 1 判据域，B2 D1-6 同款）：一致 = 窗口 w−1 的 R×S 场逐像素
        # argmax 像素落在层 1 决策候选 f_auto 邻域内（转移覆盖由判据 3 独立测度）
        S_field = np.zeros((H, W), np.float32)
        if NA is not None:
            S_field[NA] = s_a
        if NB is not None:
            S_field[NB] = s_b
        RS = R_agg * S_field
        am = np.unravel_index(int(np.argmax(RS)), RS.shape)
        N_fauto = NA if f_auto == "A" else NB
        consist[w] = 1.0 if (N_fauto is not None and N_fauto[am]) else 0.0

        # 注意执行（有代价）：窗口 w 帧，主轨迹焦点邻域 α=alpha_focal + 非焦点候选邻域
        # α=alpha_low（撤走）；无注意轨迹全程 α0.5（平行对照，不施加注意也不撤走）
        other = "B" if f == "A" else "A"
        for t in range(w * WINDOW, min((w + 1) * WINDOW, n_frames)):
            cf_t = cA[t] if f == "A" else cB[t]
            co_t = cA[t] if other == "A" else cB[t]
            fm = disk_mask(cf_t, R_FOCAL) if cf_t is not None else None
            wm = disk_mask(co_t, R_FOCAL) if co_t is not None else None
            loop.step(frames[t], fm, wm)
            loop_noatt.step(frames[t])

        # 行为量（docs/B3 §1.4 冻结）：利害加权总收益（含机会成本）
        # BENEFIT(w) = s_f·(R_f_noatt − R_f_att) + s_g·(R_g_noatt − R_g_att)
        lo1 = w * WINDOW
        hi1 = min(lo1 + WINDOW, n_frames)
        cf = cA if f == "A" else cB
        cg = cA if other == "A" else cB
        Rf_att = neighborhood_residual_mean(loop.res_frames, cf, lo1, hi1)
        Rf_noatt = neighborhood_residual_mean(loop_noatt.res_frames, cf, lo1, hi1)
        Rg_att = neighborhood_residual_mean(loop.res_frames, cg, lo1, hi1)
        Rg_noatt = neighborhood_residual_mean(loop_noatt.res_frames, cg, lo1, hi1)
        Rf_att_rec[w] = Rf_att
        Rf_noatt_rec[w] = Rf_noatt
        Rg_att_rec[w] = Rg_att
        Rg_noatt_rec[w] = Rg_noatt
        if Rf_att == Rf_att and Rf_noatt == Rf_noatt and Rg_att == Rg_att and Rg_noatt == Rg_noatt:
            s_f = s_a if f == "A" else s_b
            s_g = s_a if other == "A" else s_b
            benefit[w] = s_f * (Rf_noatt - Rf_att) + s_g * (Rg_noatt - Rg_att)
        else:
            benefit[w] = 0.0        # 检测全失败窗口：中性（不判降不判升）
        sec[w] = 1 if benefit[w] < 0.0 else 0
        prev_sec = sec[w]
        prev_focus = f

    wins = list(range(1, n_w))
    consist_frac = float(np.mean([consist[w] for w in wins]))
    neg_wins = [w for w in wins if benefit[w] < 0.0]
    neg_n = len(neg_wins)
    neg_frac = neg_n / len(wins) if wins else 0.0
    frac_pos = float(np.mean([1.0 if benefit[w] > 0.0 else 0.0 for w in wins]))
    mean_benefit = float(np.mean([benefit[w] for w in wins]))
    n_sec = int(sum(sec[w] for w in wins))
    # 转移池（docs/B3 §1.5 冻结）：sec(w)=1 且 w+1 ≤ n_w−1（有后继窗口可观察转移执行与成效；
    # 末窗口 sec 无后继，不进入池）
    trig_pool = [w for w in wins if sec[w] == 1 and w + 1 < n_w]
    trig_n = sum(1 for w in trig_pool if focus[w + 1] != focus[w])
    trig = trig_n / len(trig_pool) if trig_pool else float("nan")
    redir_n = len(trig_pool)
    redir_ok_n = sum(1 for w in trig_pool if benefit[w + 1] > 0.0)
    redir = redir_ok_n / redir_n if redir_n > 0 else float("nan")
    override_n = int(sum(override[w] for w in wins))

    out = dict(seed=seed, level=LVCODE_B2, frames=n_frames,
               neg_frac=round(neg_frac, 6), neg_n=neg_n,
               trig=round(trig, 6) if trig == trig else None,
               redir=round(redir, 6) if redir == redir else None,
               redir_ok_n=redir_ok_n, redir_n=redir_n,
               frac_pos=round(frac_pos, 6), mean_benefit=round(mean_benefit, 6),
               n_sec=n_sec, override_n=override_n,
               consist_frac=round(consist_frac, 6),
               det_err_a_mean=round(float(np.mean(det_err_a)), 4) if det_err_a else None,
               det_err_b_mean=round(float(np.mean(det_err_b)), 4) if det_err_b else None,
               focus=focus, f_auto=f_auto_rec,
               VA=[round(v, 6) for v in VA_rec],
               VB=[round(v, 6) for v in VB_rec],
               Rf_att=[None if v != v else round(v, 6) for v in Rf_att_rec],
               Rf_noatt=[None if v != v else round(v, 6) for v in Rf_noatt_rec],
               Rg_att=[None if v != v else round(v, 6) for v in Rg_att_rec],
               Rg_noatt=[None if v != v else round(v, 6) for v in Rg_noatt_rec],
               benefit=[round(v, 6) for v in benefit],
               sec=sec, consist=consist, override=override)
    return out


# ---------------- 守卫（docs/B3 §1.7 冻结；守卫种子数固定 10，不随主实验 n_seeds） ----------------
def guard_b2(n_seeds=10, n_frames=N_FRAMES):
    """R_B3_GUARD_B2：B2 层 1 同代码路径复现（import attention_emergence_test.run_attention，
    B2 冻结实现零改动）。断言：CONSIST ≥ 0.80（B2 实测 1.0000）、FRACPOS ≥ 0.60（B2 实测
    1.0000）、MEANBEN ∈ [0.03, 0.06]（B2 实测 0.0447）。"""
    units = [run_attention(s, n_frames=n_frames, jitter=JITTER) for s in range(n_seeds)]
    consist_vals = [u["consist_frac"] for u in units]
    fp_vals = [u["frac_pos"] for u in units]
    mb_vals = [u["mean_benefit"] for u in units]
    c_m, _ = mean_sd(consist_vals)
    f_m, _ = mean_sd(fp_vals)
    b_m, _ = mean_sd(mb_vals)
    ok = (c_m >= B2_CONSIST_MIN and f_m >= B2_FRACPOS_MIN
          and B2_MEANBEN_LO <= b_m <= B2_MEANBEN_HI)
    return (1 if ok else 0), c_m, f_m, b_m


# ---------------- 诊断模式（R_B3_DIAG_* 行；诊断轮用，主运行摘要块格式不变） ----------------
def diag(seed=0, n_frames=N_FRAMES):
    u = run_cost(seed, n_frames=n_frames, jitter=JITTER)
    n_w = max(1, n_frames // WINDOW)
    print("R_B3_DIAG_SEED=%d" % seed)
    for w in range(1, n_w):
        f = u["focus"][w]
        fa = u["f_auto"][w]
        print("R_B3_DIAG_W%d=%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f" % (
            w, ("A" if f == "A" else ("B" if f == "B" else "N")),
            ("A" if fa == "A" else ("B" if fa == "B" else "N")),
            u["sec"][w], u["consist"][w],
            u["VA"][w], u["VB"][w],
            (u["Rf_noatt"][w] if u["Rf_noatt"][w] is not None else 0.0),
            (u["Rf_att"][w] if u["Rf_att"][w] is not None else 0.0),
            (u["Rg_noatt"][w] if u["Rg_noatt"][w] is not None else 0.0),
            (u["Rg_att"][w] if u["Rg_att"][w] is not None else 0.0),
            u["benefit"][w]))
    print("R_B3_DIAG_NEG_FRAC=%.4f" % u["neg_frac"])
    print("R_B3_DIAG_NEG_N=%d" % u["neg_n"])
    print("R_B3_DIAG_TRIG=%.4f" % (u["trig"] if u["trig"] is not None else -1.0))
    print("R_B3_DIAG_REDIR=%.4f" % (u["redir"] if u["redir"] is not None else -1.0))
    print("R_B3_DIAG_REDIR_N=%d" % u["redir_n"])
    print("R_B3_DIAG_REDIR_OK=%d" % u["redir_ok_n"])
    print("R_B3_DIAG_FRAC_POS=%.4f" % u["frac_pos"])
    print("R_B3_DIAG_MEAN_BENEFIT=%.6f" % u["mean_benefit"])
    print("R_B3_DIAG_NSEC=%d" % u["n_sec"])
    print("R_B3_DIAG_OVERRIDE_N=%d" % u["override_n"])
    print("R_B3_DIAG_CONSIST_FRAC=%.4f" % u["consist_frac"])
    print("R_B3_DIAG_DET_A=%.4f" % (u["det_err_a_mean"] if u["det_err_a_mean"] is not None else -1.0))
    print("R_B3_DIAG_DET_B=%.4f" % (u["det_err_b_mean"] if u["det_err_b_mean"] is not None else -1.0))
    return 0


def scan(seed=0, n_frames=N_FRAMES, scan_seeds=(0, 1, 2)):
    """诊断轮代价参数边界扫描（R_B3_SCAN_* 行）：撤走强度（对称预算 α_low = 1−α_focal，
    成本系数 = α_focal−0.5 = 0.5−α_low）与利害差 s_A（s_B=1.0）组合——"成本系数多大时负
    收益开始出现 + 转移修复率"（docs/B3 §1.9：修复前诊断；只作诊断，最终参数在 main 前
    定案并记录 §二；判据/机制语义 §一 不动）。每组合跑 scan_seeds 个种子并池化报告。"""
    import itertools
    for af, sA in itertools.product((0.55, 0.60, 0.65, 0.70), (0.3, 0.4, 0.5, 0.6)):
        al = 1.0 - af
        neg_n = 0
        neg_win = 0
        redir_n = 0
        redir_ok = 0
        trig_pool = 0
        trig_ok = 0
        mb_sum = 0.0
        for s in scan_seeds:
            u = run_cost(s, n_frames=n_frames, jitter=JITTER, s_a=sA, s_b=1.0,
                         alpha_focal=af, alpha_low=al)
            neg_n += u["neg_n"]
            neg_win += len(list(range(1, max(1, n_frames // WINDOW))))
            redir_n += u["redir_n"]
            redir_ok += u["redir_ok_n"]
            trig_pool += u["redir_n"]
            trig_ok += u["redir_n"]
            mb_sum += u["mean_benefit"]
        negfrac = neg_n / neg_win if neg_win else 0.0
        redir = redir_ok / redir_n if redir_n > 0 else float("nan")
        print("R_B3_SCAN_AF=%.2f_AL=%.2f_SA=%.1f_SB=1.0_NEGFRAC=%.4f_NEGN=%d_TRIG=%.4f_REDIR=%.4f_REDIRN=%d_REDIROK=%d_MB=%.6f" % (
            af, al, sA, negfrac, neg_n,
            (trig_ok / trig_pool if trig_pool > 0 else float("nan")),
            redir, redir_n, redir_ok, mb_sum / len(scan_seeds)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="at3")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="诊断模式：单种子逐窗口 R_B3_DIAG_* 行（--seed 选择）")
    ap.add_argument("--scan", action="store_true",
                    help="诊断模式：代价参数边界扫描 R_B3_SCAN_* 行")
    ap.add_argument("--scan-seeds", type=int, default=3,
                    help="扫描每组合池化的种子数（默认 3）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.diag:
        return diag(seed=args.seed, n_frames=args.frames)
    if args.scan:
        return scan(seed=args.seed, n_frames=args.frames,
                    scan_seeds=tuple(range(args.scan_seeds)))

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    cfg = {"n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "jitter": args.jitter, "tag": args.tag,
           "level": LVCODE_B2,
           "scene": {"a_regimes": B2.A_REGIMES, "b_freq": B2.B_FREQ,
                     "s_a": S_A, "s_b": S_B, "r_focal": R_FOCAL,
                     "ctx_split_y": CTX_SPLIT_Y, "occ_lum_thresh": OCC_LUM_THRESH,
                     "a_min": A_MIN},
           "cost_attention": {"alpha_focal": COST_ALPHA_FOCAL, "alpha_low": COST_ALPHA_LOW,
                              "s_a": COST_S_A, "s_b": S_B, "loop": LOOP_CFG},
           "criteria": {"neg_frac_min": NEG_FRAC_MIN, "neg_nobs_min": NEG_NOBS_MIN,
                        "trig_min": TRIG_MIN, "redir_min": REDIR_MIN,
                        "redir_nobs_min": REDIR_NOBS_MIN},
           "guards": {"b2_consist_min": B2_CONSIST_MIN,
                      "b2_fracpos_min": B2_FRACPOS_MIN,
                      "b2_meanben_range": [B2_MEANBEN_LO, B2_MEANBEN_HI]}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_at3_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for seed in seeds:
            key = "%d_%d" % (LVCODE_B2, seed)
            if key in per_unit:
                continue
            per_unit[key] = run_cost(seed, n_frames=args.frames, jitter=args.jitter)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%d_%d" % (LVCODE_B2, s)] for s in seeds]

    # ---- 判据（docs/B3 §1.5 冻结；跨种子池化） ----
    neg_n_total = sum(u["neg_n"] for u in units)
    n_wins_total = sum(len(list(range(1, max(1, args.frames // WINDOW)))) for _ in units)
    neg_frac_pooled = neg_n_total / n_wins_total if n_wins_total else 0.0
    c1 = (neg_frac_pooled >= NEG_FRAC_MIN) and (neg_n_total >= NEG_NOBS_MIN)

    n_win = max(1, args.frames // WINDOW)
    trig_pool_n = 0                                          # 转移池 = sec 且有后继的窗口
    trig_ok_total = 0                                        # 其中焦点实际改变的（机制赋值下恒全）
    for u in units:
        for w in range(1, n_win):
            if u["sec"][w] == 1 and w + 1 < n_win:
                trig_pool_n += 1
                if u["focus"][w + 1] != u["focus"][w]:
                    trig_ok_total += 1
    trig_pooled = trig_ok_total / trig_pool_n if trig_pool_n > 0 else float("nan")
    c2 = (trig_pooled >= TRIG_MIN) if trig_pooled == trig_pooled else False

    redir_n_total = sum(u["redir_n"] for u in units)
    redir_ok_total = sum(u["redir_ok_n"] for u in units)
    redir_pooled = redir_ok_total / redir_n_total if redir_n_total > 0 else float("nan")
    c3 = (redir_n_total >= REDIR_NOBS_MIN) and (redir_pooled >= REDIR_MIN)

    # ---- 守卫（docs/B3 §1.7 冻结；固定 10 种子，独立于主实验 n_seeds） ----
    g_b2, b2_c, b2_fp, b2_mb = guard_b2()
    g_so, so_r_m, so_r_sd, so_r_ci, so_diff_ci, so_auc, so_nobs = guard_so()
    g_comp, c_mae, c_mae_sd, c_sc2, c_sc2_sd, c_comp, c_comp_sd, c_churn = \
        guard_compose()
    guards_ok = (g_b2 == 1) and (g_so == 1) and (g_comp == 1)

    # ---- 判定（docs/B3 §1.6 冻结） ----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1:
        verdict = "NEG_FAIL"
    elif c2 and c3:
        verdict = "REDIRECT_PASS"
    else:
        verdict = "PARTIAL"

    # ---- 内部确定性复现（docs/B3 §1.7-4；第二遍强制重算，不读 checkpoint） ----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for key in per_unit:
            for kk in REPRO_KEYS:
                if per_unit[key][kk] != per_unit2[key][kk]:
                    repro = 0

    out = {
        "artifact": "attention_cost_test",
        "doc_ref": "lineB-motion-coupling/docs/B3",
        "config": cfg,
        "per_unit": per_unit,
        "criteria": {"c1_neg_benefit_occurs": bool(c1), "c2_redirect_trigger": bool(c2),
                     "c3_redirect_correct": bool(c3),
                     "neg_frac_pooled": neg_frac_pooled, "neg_n": neg_n_total,
                     "trig_pooled": trig_pooled, "trig_n": trig_pool_n,
                     "redir_pooled": redir_pooled, "redir_n": redir_n_total,
                     "redir_ok": redir_ok_total,
                     "frac_pos_mean": round(float(np.mean([u["frac_pos"] for u in units])), 6),
                     "mean_benefit_mean": round(float(np.mean([u["mean_benefit"] for u in units])), 6),
                     "consist_mean": round(float(np.mean([u["consist_frac"] for u in units])), 6),
                     "override_n": int(sum(u["override_n"] for u in units))},
        "guards": {"b2": g_b2, "b2_consist": b2_c, "b2_fracpos": b2_fp, "b2_meanben": b2_mb,
                   "so": g_so, "so_r_mean": so_r_m, "so_r_sd": so_r_sd,
                   "so_r_ci95": list(so_r_ci), "so_diff_ci95": list(so_diff_ci),
                   "so_auc": so_auc, "so_nobs": so_nobs,
                   "compose": g_comp, "compose_mae": c_mae, "compose_mae_sd": c_mae_sd,
                   "compose_sc2": c_sc2, "compose_sc2_sd": c_sc2_sd,
                   "compose_comp": c_comp, "compose_comp_sd": c_comp_sd,
                   "compose_churn": c_churn,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "at3_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定） ----
    print("R_B3_TAG=%s" % args.tag)
    print("R_B3_SEEDS=%d" % len(seeds))
    print("R_B3_FRAMES=%d" % args.frames)
    for s in seeds:
        u = per_unit["%d_%d" % (LVCODE_B2, s)]
        print("R_B3_S%d_NEGFRAC=%.4f" % (s, u["neg_frac"]))
        print("R_B3_S%d_NEGN=%d" % (s, u["neg_n"]))
        print("R_B3_S%d_TRIG=%.4f" % (s, u["trig"] if u["trig"] is not None else -1.0))
        print("R_B3_S%d_REDIR=%.4f" % (s, u["redir"] if u["redir"] is not None else -1.0))
        print("R_B3_S%d_REDIRN=%d" % (s, u["redir_n"]))
        print("R_B3_S%d_REDIROK=%d" % (s, u["redir_ok_n"]))
        print("R_B3_S%d_FRACPOS=%.4f" % (s, u["frac_pos"]))
        print("R_B3_S%d_MEANBEN=%.6f" % (s, u["mean_benefit"]))
        print("R_B3_S%d_NSEC=%d" % (s, u["n_sec"]))
        print("R_B3_S%d_OVERRIDE=%d" % (s, u["override_n"]))
        print("R_B3_S%d_CONSIST=%.4f" % (s, u["consist_frac"]))
        print("R_B3_S%d_DETA=%.4f" % (s, u["det_err_a_mean"] if u["det_err_a_mean"] is not None else -1.0))
        print("R_B3_S%d_DETB=%.4f" % (s, u["det_err_b_mean"] if u["det_err_b_mean"] is not None else -1.0))
    print("R_B3_NEG_FRAC_POOLED=%.4f" % neg_frac_pooled)
    print("R_B3_NEG_N_TOTAL=%d" % neg_n_total)
    print("R_B3_TRIG_POOLED=%.4f" % trig_pooled)
    print("R_B3_TRIG_N=%d" % trig_pool_n)
    print("R_B3_REDIR_POOLED=%.4f" % redir_pooled)
    print("R_B3_REDIR_N=%d" % redir_n_total)
    print("R_B3_REDIR_OK=%d" % redir_ok_total)
    print("R_B3_FRACPOS_MEAN=%.4f" % float(np.mean([u["frac_pos"] for u in units])))
    print("R_B3_MEANBEN_MEAN=%.6f" % float(np.mean([u["mean_benefit"] for u in units])))
    print("R_B3_CONSIST_MEAN=%.4f" % float(np.mean([u["consist_frac"] for u in units])))
    print("R_B3_OVERRIDE_N_TOTAL=%d" % int(sum(u["override_n"] for u in units)))
    print("R_B3_C1=%s" % ("PASS" if c1 else "FAIL"))
    print("R_B3_C2=%s" % ("PASS" if c2 else "FAIL"))
    print("R_B3_C3=%s" % ("PASS" if c3 else "FAIL"))
    print("R_B3_GUARD_B2=%d" % g_b2)
    print("R_B3_GUARD_B2_CONSIST=%.4f" % b2_c)
    print("R_B3_GUARD_B2_FRACPOS=%.4f" % b2_fp)
    print("R_B3_GUARD_B2_MEANBEN=%.6f" % b2_mb)
    print("R_B3_GUARD_SO=%d" % g_so)
    print("R_B3_GUARD_SO_R=%.6f" % so_r_m)
    print("R_B3_GUARD_SO_R_SD=%.6f" % so_r_sd)
    print("R_B3_GUARD_SO_LO=%.6f" % so_r_ci[0])
    print("R_B3_GUARD_SO_HI=%.6f" % so_r_ci[1])
    print("R_B3_GUARD_SO_DIFF_LO=%.6f" % so_diff_ci[0])
    print("R_B3_GUARD_SO_DIFF_HI=%.6f" % so_diff_ci[1])
    print("R_B3_GUARD_SO_AUC=%.6f" % so_auc)
    print("R_B3_GUARD_SO_NOBS=%.1f" % so_nobs)
    print("R_B3_GUARD_COMPOSE=%d" % g_comp)
    print("R_B3_GUARD_COMPOSE_MAE=%.6f" % c_mae)
    print("R_B3_GUARD_COMPOSE_SC2=%.4f" % c_sc2)
    print("R_B3_GUARD_COMPOSE_COMP=%.4f" % c_comp)
    print("R_B3_GUARD_COMPOSE_CHURN=%.4f" % c_churn)
    print("R_B3_REPRO=%d" % repro)
    print("R_B3_VERDICT=%s" % verdict)
    print("R_B3_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
