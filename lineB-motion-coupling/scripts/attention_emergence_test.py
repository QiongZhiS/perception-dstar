"""lineB-motion-coupling/scripts/attention_emergence_test.py — B 路第二格：注意决策的涌现
（docs: lineB-motion-coupling/docs/B2-注意决策涌现-预注册设计.md §一 冻结）。

核心：把 docs/237 二阶残差从"探测"升为"决策驱动"（C2 立场层 1/层 2）。合成场景
（docs/235 风格，160×120 棋盘 + 两移动亮圆盘 + 噪声 σ=3.0×jitter）上构造两个候选
注意目标——A 高残差低利害（三档 regime 快目标，外赋利害 s_A=0.3）、B 低残差高利害
（恒速慢目标，外赋利害 s_B=1.0）；利害场 S(x,y) 第一版外赋（docs/174/188 纪律：
外赋"谁值得"标量，坐标由感知层亮域检测锁定，docs/190 同款）。

注意决策层（层 1）：每窗口从上一窗口残差场 R_agg × 外赋利害场 S 计算候选注意价值
V_i = s_i·mean_邻域(R)，焦点 f = argmax V（行为涌现，非手工指定位置）；注意执行 =
焦点邻域预测更新速率 α 0.5→0.85（AttentionLoop 逐像素 α，资源投给焦点）；预期收益
BENEFIT(w) = R_f(w−1) − R_f(w)（焦点邻域注意前后残差下降）；二级残差 sec(w) =
1[BENEFIT(w) < 0]（收益为负的检测 = "我注意错了"）；注意转移（层 2）= sec 触发时
焦点给次优候选 second(w−1)。

判据（§1.4 冻结，docs/247 标签 [L5][机制][涌现检验]——L5 展望形态诚实标注，非思考
证明）：
  C1 FOCUS_CONSIST   : 焦点落在逐像素 argmax(R×S) 邻域的比例 ≥ 0.80
  C2 BENEFIT_CONFIRM : 焦点被注意后该区域 MAE 下降比例 ≥ 0.60 且均值下降 bootstrap CI 下界 > 0
  C3 REDIRECT_CORRECT: 焦点收益为负时注意转移到次优候选的池化正确率 ≥ 0.70（观察 ≥ 10）
  C4 KEEP 守卫      : R_B2_GUARD_SO + R_B2_GUARD_COMPOSE + R_B2_REPRO 全 = 1
判定（§1.5）：全过 = ATTENTION_EMERGES；焦点过收益/转移不过 = PARTIAL；焦点不过 =
ATTENTION_FAIL；守卫不过 = GUARD_FAIL。

守卫（§1.6 冻结）：
  R_B2_GUARD_SO     : docs/241 S1 长程流式（lvcode 22、2400 帧、10 种子）SO_info 复现
                      （so_probe 冻结公式 v2_global_hazard + stream_test.stream_frames +
                      CompLoop 同代码路径）：r_mean ∈ [0.05,0.12] 且 CI 下界 > 0 且
                      diff_rand CI 下界 > 0（docs/241 r=+0.0847 量级）
  R_B2_GUARD_COMPOSE: docs/235/237 C2 结构上下文（compose_test.run_level(22,·) 10 种子）
                      MAE/SC2/复合/churn 与 docs/237 §3.2 公布值逐位一致（容差 1e-3）
  R_B2_REPRO        : --repro 时主实验 10 运行整体重跑第二遍，关键数字位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_B2_* 摘要块
（顺序固定）；JSON 归档 lineB-motion-coupling/out/at_<tag>.json + checkpoint
ckpt_at_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 lineB-motion-coupling/out/*.log 与 lineB-motion-coupling/out/*.json 原文。
**未修改任何主线既有脚本**（vision/ 下全部不动；只 import 复用）。

用法：
  python lineB-motion-coupling/scripts/attention_emergence_test.py --n-seeds 10 --tag main --repro
  python lineB-motion-coupling/scripts/attention_emergence_test.py --n-seeds 1 --tag timing
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))          # 项目根（lineB-motion-coupling/ 的父目录）
VISION = os.path.join(PROJ, "vision")
for _p in (VISION, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER
from compose_test import CompLoop, run_level, make_scene, LEVELS, ENERGY_BINS, UPPER_BINS, cv2_circle
from so_probe import compute_observations, seed_metrics, window_aligned
from stream_test import stream_frames

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("lineB-motion-coupling", "out")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 场景旋钮（docs/B2 §1.2 冻结）----
LVCODE_B2 = 40            # 与 0-6 / 20-23 / 30-33 的 rng 流错开
W, H = 160, 120
FPS = 30
N_FRAMES = 240
WINDOW = 10
BG_CELL = 24
BG_DARK, BG_BRIGHT = 64.0, 96.0
DISK_R = 10.0
OBJ_GRAY = 255.0
NOISE_SIGMA = 3.0
A_CENTER = (80.0, 34.0)
A_ORBIT = 13.0
A_FREQ = 0.33             # Hz
A_REGIMES = [1.0, 3.0, 0.5]
REGIME_BOUNDS = [0, 80, 160, 240]
B_CENTER = (80.0, 98.0)
B_ORBIT = 10.0
B_FREQ = 0.30             # Hz（慢目标 = 低残差候选）
# B 静止段尝试（诊断轮 D1 记录）：曾设 (80,160) 制造"注意无收益"事件——但 α 提高使
# 残差 = (1−α)|L−bg| 恒降的数学结构下注意收益恒正，"收益为负"（sec）在本构造下数学上
# 不发生（sec 事件需要注意执行有代价/失效的机制形态，如资源饱和/干扰抑制——docs/186
# 中央凹资源有限形态）；判据 3 因样本不足如实报告。恢复 B 恒速（(0,0) = 空段）。
B_STILL_BOUNDS = (0, 0)
S_A = 0.5                 # 外赋利害：候选 A 低利害（诊断轮 D1 修复，§二 记录：0.3→0.5）
S_B = 1.0                 # 外赋利害：候选 B 高利害
R_FOCAL = 18.0            # 候选/利害/焦点邻域半径（px）
CTX_SPLIT_Y = 68.0        # 上/下候选归属分割线（A 恒上半、B 恒下半）
OCC_LUM_THRESH = 220.0    # 亮检测 = L > log(221)（docs/260 同款）
A_MIN = 25                # 连通域最小面积（px）

# ---- 注意机制旋钮（docs/B2 §1.3 冻结）----
ALPHA_FOCAL = 0.85        # 注意执行：焦点邻域预测更新速率
# 回路参数：与 compose_test.main / so_probe.LOOP_CFG 逐字一致（结果可比）
LOOP_CFG = {"alpha_fast": 0.5, "alpha_slow": 0.03, "thresh": 0.15,
            "deadband": 0.015, "k_theta": 6.0, "k_db": 1.5,
            "thresh_max": 0.6, "db_max": 0.15, "k_consist": 3,
            "hits_min": 3, "persist_win": 5, "k_split": 5, "delta_rel": 0.30,
            "energy_bins": list(ENERGY_BINS), "bbox_bins": list(UPPER_BINS)}
CP_KEYS = ("alpha_fast", "alpha_slow", "thresh", "deadband", "k_theta", "k_db",
           "thresh_max", "db_max", "window", "k_consist", "hits_min",
           "persist_win", "energy_bins")

# ---- 判据阈值（docs/B2 §1.4 冻结）----
FOCUS_MIN = 0.80
BENEFIT_POS_MIN = 0.60
REDIR_MIN = 0.70
REDIR_NOBS_MIN = 10

# ---- 守卫容差（docs/B2 §1.6 冻结）----
SO_R_LO, SO_R_HI = 0.05, 0.12
COMPOSE_MAE_REF, COMPOSE_SC2_REF = 0.024198, 2.0
COMPOSE_COMP_REF, COMPOSE_CHURN_REF = 0.900, 0.000
TOL = 1e-3

REPRO_KEYS = ["focus", "second", "VA", "VB", "benefit", "sec", "consist",
              "consist_frac", "frac_pos", "mean_benefit", "redir_ok_n",
              "redir_n", "n_sec"]


# ---------------- 合成场景（docs/B2 §1.2 冻结） ----------------
def make_b2_scene(seed, n_frames=N_FRAMES, width=W, height=H, fps=FPS, jitter=JITTER):
    """生成 (LVCODE_B2, 种子) 的灰度帧序列 + 逐帧 GT 质心（只用于事后诊断报告，
    绝不进入机制）。确定性：rng 由 (seed, LVCODE_B2) 派生；调用顺序固定
    （noise_mult -> A 相位 -> B 相位 -> 每帧 regime 换档的 A 相位重抽）。"""
    rng = np.random.default_rng(seed * 7919 + LVCODE_B2 * 104729 + 13)
    noise_mult = rng.uniform(1 - jitter, 1 + jitter) if jitter > 0 else 1.0
    sigma = NOISE_SIGMA * noise_mult
    ph_a = rng.uniform(0, 2 * np.pi)
    ph_b = rng.uniform(0, 2 * np.pi)

    bg = np.zeros((height, width), np.float32)
    for y in range(0, height, BG_CELL):
        for x in range(0, width, BG_CELL):
            bg[y:y + BG_CELL, x:x + BG_CELL] = \
                BG_DARK if ((x // BG_CELL) + (y // BG_CELL)) % 2 == 0 else BG_BRIGHT

    frames = []
    a_pos = []
    b_pos = []
    cur_reg = -1
    th_a = ph_a
    th_b = ph_b
    th_b_frozen = None
    for t in range(n_frames):
        rg = 0
        for b in range(3):
            if REGIME_BOUNDS[b] <= t < REGIME_BOUNDS[b + 1]:
                rg = b
        if rg != cur_reg:
            cur_reg = rg
            th_a = rng.uniform(0, 2 * np.pi)          # 换档相位重抽（docs/235 C0 同款）
        th_a += 2 * np.pi * A_FREQ * A_REGIMES[rg] / fps
        ax = A_CENTER[0] + A_ORBIT * np.cos(th_a)
        ay = A_CENTER[1] + A_ORBIT * np.sin(th_a)
        # B 静止段（诊断轮 D1 修复，§二 记录）：帧 [B_STILL_BOUNDS) 内 B 位置固定
        # （静态目标被 bg_fast 完全预测掉 → 注意它无误差可减 → "收益为负"的测试事件，
        # 判据 3 的公平测试机会；B 仍是"低残差高利害"候选——静止=残差≈0 的纯粹形态）
        if B_STILL_BOUNDS[0] <= t < B_STILL_BOUNDS[1]:
            if th_b_frozen is None:
                th_b_frozen = th_b
            bx = B_CENTER[0] + B_ORBIT * np.cos(th_b_frozen)
            by = B_CENTER[1] + B_ORBIT * np.sin(th_b_frozen)
        else:
            th_b += 2 * np.pi * B_FREQ / fps
            bx = B_CENTER[0] + B_ORBIT * np.cos(th_b)
            by = B_CENTER[1] + B_ORBIT * np.sin(th_b)
            th_b_frozen = None

        img = bg.copy()
        cv2_circle(img, int(ax), int(ay), DISK_R, OBJ_GRAY)
        cv2_circle(img, int(bx), int(by), DISK_R, OBJ_GRAY)
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        a_pos.append((float(ax), float(ay)))
        b_pos.append((float(bx), float(by)))
    gts = dict(a_pos=a_pos, b_pos=b_pos)
    return frames, gts


def top2_components(mask, min_area=A_MIN):
    """8-连通域：返回面积最大的至多 2 个连通域 [(mask, area, centroid)]（docs/260
    largest_component 同实现风格，扩展为 top-2；两候选分离保证连通域 = 2 个）。"""
    if mask.sum() == 0:
        return []
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    order = np.argsort(areas)[::-1]
    out = []
    for idx in order:
        if areas[idx] < min_area:
            continue
        comp = lab == idx
        ys, xs = np.nonzero(comp)
        out.append((comp, int(areas[idx]), (float(xs.mean()), float(ys.mean()))))
        if len(out) >= 2:
            break
    return out


def disk_mask(c, r=R_FOCAL, shape=(H, W)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r


def mean_centroid(cs, lo, hi):
    pts = [cs[t] for t in range(lo, hi) if cs[t] is not None]
    if not pts:
        return None
    return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))


def neighborhood_residual_mean(res_frames, centroids, lo, hi, r=R_FOCAL):
    """候选邻域残差均值：逐帧质心邻域均值再窗口平均；无有效质心 -> nan。"""
    vals = []
    for t in range(lo, hi):
        c = centroids[t]
        if c is None:
            continue
        m = disk_mask(c, r)
        vals.append(float(res_frames[t][m].mean()))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------- 注意回路（AttentionLoop：CPLoop + 逐像素 α + 残差场记录） ----------------
class AttentionLoop(CPLoop):
    """critical_point 同款预测回路（bg_fast/bg_slow/自适应阈值，机制零改动继承），
    预测更新改为逐像素 α（默认 ≡ α_fast，注意执行 = 焦点邻域 α 提高到 ALPHA_FOCAL），
    逐帧记录残差场 |L − bg_fast|（注意决策的输入）。只 import 复用 CPLoop，不改主线。"""

    def __init__(self, alpha_focal=ALPHA_FOCAL, **kw):
        kw = {k: v for k, v in kw.items() if k in CP_KEYS}
        super().__init__(**kw)
        self.alpha_focal = alpha_focal
        self.res_frames = []

    def step(self, gray, focal_mask=None):
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


# ---------------- 主实验单位（docs/B2 §1.3 冻结） ----------------
def counterfact_residual(res_frames_noatt, centroids, lo, hi, r=R_FOCAL):
    """平行无注意轨迹（诊断轮 D1 修复，§二 记录）：与主轨迹（α 0.85 焦点）同窗口、
    同初始历史并行的 α 0.5 轨迹的候选邻域残差均值——"如果全程不注意，窗口 w 的残差"。
    修复原快照反事实的污染偏差（连续注意把 bg_fast 推到目标亮度后，从污染状态出发的
    α0.5 反事实测到低残差基线 → benefit 系统偏负）。"""
    return neighborhood_residual_mean(res_frames_noatt, centroids, lo, hi, r=r)


def run_attention(seed, n_frames=N_FRAMES, jitter=JITTER, alpha_focal=ALPHA_FOCAL):
    """跑 (LVCODE_B2, 种子) 一次完整运行 → 注意决策行为量 + 判据组件。

    窗口 0 = 预热基线（α 全 0.5，无注意）；窗口 w ∈ [1, 24)：
      决策（窗口 w−1 残差场 R_agg × 外赋利害场）→ V_A/V_B（候选邻域 R×S 最大值，
      C2 层 1 "argmax(R×S) 邻域"的候选级忠实实现）→ 层 1 决策候选 f_auto = argmax V
      （sec(w−1)=1 → 层 2 转移覆盖：实际焦点 f = second(w−1)）→ 注意执行（窗口 w
      焦点邻域 α=alpha_focal，主轨迹）→ 行为量（BENEFIT = 平行无注意轨迹 α0.5 vs
      主轨迹 α0.85 的因果收益 / sec / consist）。"""
    frames, gts = make_b2_scene(seed, n_frames=n_frames, jitter=jitter)
    n_w = max(1, n_frames // WINDOW)
    loop = AttentionLoop(window=WINDOW, alpha_focal=alpha_focal, **LOOP_CFG)
    loop_noatt = AttentionLoop(window=WINDOW, alpha_focal=alpha_focal, **LOOP_CFG)
    cA = [None] * n_frames
    cB = [None] * n_frames
    lastA = lastB = None
    det_err_a = []
    det_err_b = []
    # 检测循环（全部帧；亮域 top-2 质心，与 step 解耦——修复预热/注意双 step 的
    # res_frames 语义混乱，诊断轮 D1 记录）
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
    second_sel = [None] * n_w
    VA_rec = [0.0] * n_w
    VB_rec = [0.0] * n_w
    Rf_rec = [float("nan")] * n_w
    Rf_att_rec = [float("nan")] * n_w
    Rf_noatt_rec = [float("nan")] * n_w
    argmax_cand = [None] * n_w
    meanA_rec = [0.0] * n_w
    maxA_rec = [0.0] * n_w
    meanB_rec = [0.0] * n_w
    maxB_rec = [0.0] * n_w
    benefit = [0.0] * n_w
    sec = [0] * n_w
    consist = [0] * n_w
    redir_pool = []
    prev_sec = 0

    for w in range(1, n_w):
        lo0 = (w - 1) * WINDOW
        hi0 = min(lo0 + WINDOW, n_frames)
        R_agg = np.mean(np.stack(loop.res_frames[lo0:hi0]), axis=0)
        cA0 = mean_centroid(cA, lo0, hi0)
        cB0 = mean_centroid(cB, lo0, hi0)
        NA = disk_mask(cA0, R_FOCAL) if cA0 is not None else None
        NB = disk_mask(cB0, R_FOCAL) if cB0 is not None else None
        if NA is not None:
            meanA_rec[w] = float(R_agg[NA].mean())
            maxA_rec[w] = float(R_agg[NA].max())
        if NB is not None:
            meanB_rec[w] = float(R_agg[NB].mean())
            maxB_rec[w] = float(R_agg[NB].max())
        # 候选注意价值 = 候选邻域内 R×S 最大值（= s_i·max_{N_i}(R_agg)；
        # C2 层 1 "argmax(R×S) 邻域"的候选级忠实实现——argmax(R×S) 像素落在哪个
        # 候选邻域，哪个候选的 V 最大；与逐像素 argmax 基线同构，一致性的区分力
        # 来自候选检测噪声与邻域边界）
        VA = S_A * maxA_rec[w] if NA is not None else 0.0
        VB = S_B * maxB_rec[w] if NB is not None else 0.0
        VA_rec[w] = VA
        VB_rec[w] = VB

        order = sorted([("A", VA), ("B", VB)], key=lambda kv: kv[1], reverse=True)
        second_sel[w] = order[1][0]
        f_auto = order[0][0]               # 层 1 决策候选（argmax V）
        f = f_auto
        if prev_sec == 1:
            f = second_sel[w - 1]          # 层 2 转移覆盖：实际焦点 = 决策时刻次优候选
            redir_pool.append(1.0 if f == second_sel[w - 1] else 0.0)
        focus[w] = f

        # FOCUS_CONSIST（诊断轮 D1 精化，§二 记录：判据 1 测层 1 决策候选 f_auto——C2
        # 层结构域分离，层 2 转移的覆盖由判据 3 单独测度；否则转移窗口（焦点=次优≠
        # argmax）与"焦点=argmax(R×S)"同帧不相容）。一致 = 窗口 w−1 的 R×S 场逐像素
        # argmax 像素落在层 1 决策候选邻域内。
        S_field = np.zeros((H, W), np.float32)
        if NA is not None:
            S_field[NA] = S_A
        if NB is not None:
            S_field[NB] = S_B
        RS = R_agg * S_field
        am = np.unravel_index(int(np.argmax(RS)), RS.shape)
        N_fauto = NA if f_auto == "A" else NB
        argmax_cand[w] = "A" if (NA is not None and NA[am]) else \
            ("B" if (NB is not None and NB[am]) else None)
        consist[w] = 1.0 if (N_fauto is not None and N_fauto[am]) else 0.0

        # 注意执行：窗口 w 帧，实际焦点 f 逐帧质心邻域 α=alpha_focal（主轨迹）；
        # 无注意轨迹全程 α0.5（平行对照，不施加注意）
        for t in range(w * WINDOW, min((w + 1) * WINDOW, n_frames)):
            cf_t = cA[t] if f == "A" else cB[t]
            fm = disk_mask(cf_t, R_FOCAL) if cf_t is not None else None
            loop.step(frames[t], fm)
            loop_noatt.step(frames[t])

        # 行为量（诊断轮 D1 修复，§二 记录）：BENEFIT = 平行轨迹因果对照（同窗口
        # 同候选：无注意轨迹 α0.5 残差 − 主轨迹 α0.85 注意后残差）——机制语义
        # "注意带来焦点区域误差下降"不变，测量载体修正（原快照反事实有连续注意
        # 污染偏差）。sec = 1[收益为负]（C2 "若没降"→转移）。
        lo1 = w * WINDOW
        hi1 = min(lo1 + WINDOW, n_frames)
        cf = cA if f == "A" else cB
        Rf_w = neighborhood_residual_mean(loop.res_frames, cf, lo1, hi1)
        Rf_noatt_w = neighborhood_residual_mean(loop_noatt.res_frames, cf, lo1, hi1)
        Rf_rec[w] = Rf_w
        Rf_att_rec[w] = Rf_w
        Rf_noatt_rec[w] = Rf_noatt_w
        if Rf_w == Rf_w and Rf_noatt_w == Rf_noatt_w:
            benefit[w] = Rf_noatt_w - Rf_w
        else:
            benefit[w] = 0.0                             # 检测全失败窗口：中性（不判降不判升）
        sec[w] = 1 if benefit[w] < 0.0 else 0
        prev_sec = sec[w]

    wins = list(range(1, n_w))
    consist_frac = float(np.mean([consist[w] for w in wins]))
    frac_pos = float(np.mean([1.0 if benefit[w] > 0.0 else 0.0 for w in wins]))
    mean_benefit = float(np.mean([benefit[w] for w in wins]))
    n_sec = int(sum(sec[w] for w in wins))
    redir_ok_n = sum(1 for ok in redir_pool if ok == 1.0)
    redir_n = len(redir_pool)
    redir_benefit = float(np.mean([1.0 if (w + 1 < n_w and benefit[w + 1] > 0.0) else 0.0
                                   for w in range(1, n_w - 1) if sec[w] == 1])) \
        if n_sec > 0 else float("nan")

    out = dict(seed=seed, level=LVCODE_B2, frames=n_frames,
               consist_frac=round(consist_frac, 6),
               frac_pos=round(frac_pos, 6),
               mean_benefit=round(mean_benefit, 6),
               n_sec=n_sec, redir_ok_n=redir_ok_n, redir_n=redir_n,
               redir_benefit=redir_benefit,
               det_err_a_mean=round(float(np.mean(det_err_a)), 4) if det_err_a else None,
               det_err_b_mean=round(float(np.mean(det_err_b)), 4) if det_err_b else None,
               focus=focus, second=second_sel, argmax_cand=argmax_cand,
               VA=[round(v, 6) for v in VA_rec],
               VB=[round(v, 6) for v in VB_rec],
               meanA=[round(v, 6) for v in meanA_rec],
               maxA=[round(v, 6) for v in maxA_rec],
               meanB=[round(v, 6) for v in meanB_rec],
               maxB=[round(v, 6) for v in maxB_rec],
               Rf=[None if v != v else round(v, 6) for v in Rf_rec],
               Rf_att=[None if v != v else round(v, 6) for v in Rf_att_rec],
               Rf_noatt=[None if v != v else round(v, 6) for v in Rf_noatt_rec],
               benefit=[round(v, 6) for v in benefit],
               sec=sec, consist=consist)
    return out


# ---------------- 守卫（docs/B2 §1.6 冻结；守卫种子数固定 10，不随主实验 n_seeds——timing
# 冒烟也完整验证守卫，诊断轮 D1 修复：原守卫随 args.n_seeds 在 1 种子模式量级断言误报） ----------------
def guard_so(n_seeds=10, n_frames=2400, chunk=240):
    """docs/241 S1 长程 SO_info 复现（同代码路径）：so_probe 冻结公式 v2_global_hazard
    + stream_test.stream_frames（lvcode 22、2400 帧、seed_c=seed×10+c）+ CompLoop。
    断言：r_mean ∈ [0.05, 0.12] 且 r CI95 下界 > 0 且 diff_rand CI95 下界 > 0。"""
    lvcode = 22
    sos = []
    for seed in range(n_seeds):
        frames, labels = stream_frames(lvcode, seed, n_frames, chunk, W, H, FPS, JITTER)
        loop = CompLoop(window=WINDOW, **LOOP_CFG)
        for g in frames:
            loop.step(g)
        n_w = max(1, n_frames // WINDOW)
        base = loop.finalize(n_w, labels)
        matched = window_aligned(loop.match_trace)
        obs = compute_observations(matched, base["entry_log"], n_w)
        sos.append(seed_metrics(obs, seed, lvcode))
    r_vals = [s["r"] for s in sos]
    diff_vals = [s["r"] - s["r_fake_rand"] for s in sos]
    r_m, r_sd = mean_sd(r_vals)
    r_ci = bootstrap_ci(r_vals)
    diff_ci = bootstrap_ci(diff_vals)
    auc_m = float(np.mean([s["auc"] for s in sos]))
    nobs_m = float(np.mean([s["n_obs"] for s in sos]))
    ok = (SO_R_LO <= r_m <= SO_R_HI) and (r_ci[0] > 0.0) and (diff_ci[0] > 0.0)
    return (1 if ok else 0), r_m, r_sd, r_ci, diff_ci, auc_m, nobs_m


def guard_compose(n_seeds=10, n_frames=240):
    """docs/235/237 C2 结构上下文（compose_test.run_level(22,·)，同代码路径 + 同
    LOOP_CFG）：MAE/SC2/复合占比/churn 与 docs/237 §3.2 公布值逐位一致（容差 1e-3）。"""
    rs = [run_level(22, s, n_frames=n_frames, width=W, height=H, fps=FPS,
                    window=WINDOW, jitter=JITTER, loop_kwargs=LOOP_CFG)
          for s in range(n_seeds)]
    mae_m, mae_sd = mean_sd([r["mae_mean"] for r in rs])
    sc2_m, sc2_sd = mean_sd([r["sc2"] for r in rs])
    comp_m, comp_sd = mean_sd([r["compound_frac"] for r in rs])
    churn_m, _ = mean_sd([r["churn_frac"] for r in rs])
    ok = (abs(mae_m - COMPOSE_MAE_REF) <= TOL and abs(sc2_m - COMPOSE_SC2_REF) <= TOL
          and abs(comp_m - COMPOSE_COMP_REF) <= TOL and abs(churn_m - COMPOSE_CHURN_REF) <= TOL)
    return (1 if ok else 0), mae_m, mae_sd, sc2_m, sc2_sd, comp_m, comp_sd, churn_m


# ---------------- 统计外壳（critical_point 同款） ----------------
# ---------------- 诊断模式（R_B2_DIAG_* 行；诊断轮用，主运行摘要块格式不变） ----------------
def diag(seed=0, n_frames=N_FRAMES):
    u = run_attention(seed, n_frames=n_frames, jitter=JITTER)
    n_w = max(1, n_frames // WINDOW)
    print("R_B2_DIAG_SEED=%d" % seed)
    for w in range(1, n_w):
        f = u["focus"][w]
        am = u["argmax_cand"][w]
        print("R_B2_DIAG_W%d=%s,%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f" % (
            w, ("A" if f == "A" else ("B" if f == "B" else "N")),
            ("A" if am == "A" else ("B" if am == "B" else "N")),
            ("1" if u["consist"][w] else "0"),
            u["sec"][w],
            u["VA"][w], u["VB"][w],
            u["meanA"][w], u["maxA"][w], u["meanB"][w], u["maxB"][w],
            (u["Rf_noatt"][w] if u["Rf_noatt"][w] is not None else 0.0),
            (u["Rf"][w] if u["Rf"][w] is not None else 0.0),
            u["benefit"][w]))
    print("R_B2_DIAG_CONSIST_FRAC=%.4f" % u["consist_frac"])
    print("R_B2_DIAG_FRAC_POS=%.4f" % u["frac_pos"])
    print("R_B2_DIAG_MEAN_BENEFIT=%.6f" % u["mean_benefit"])
    print("R_B2_DIAG_NSEC=%d" % u["n_sec"])
    print("R_B2_DIAG_REDIR_N=%d" % u["redir_n"])
    print("R_B2_DIAG_REDIR_OK=%d" % u["redir_ok_n"])
    print("R_B2_DIAG_REDIR_BENEFIT=%.4f" % (u["redir_benefit"] if u["redir_benefit"] == u["redir_benefit"] else -1.0))
    print("R_B2_DIAG_DET_A=%.4f" % (u["det_err_a_mean"] if u["det_err_a_mean"] is not None else -1.0))
    print("R_B2_DIAG_DET_B=%.4f" % (u["det_err_b_mean"] if u["det_err_b_mean"] is not None else -1.0))
    return 0


def scan(seed=0, n_frames=N_FRAMES):
    """诊断轮参数扫描（R_B2_SCAN_* 行）：临时覆盖场景常量（s_A / B_FREQ / α_focal），
    输出每组合的判据组件数字——找"机制可公平测试"的参数区（焦点切换、sec 适中、
    收益 SNR 合理）；只作诊断，最终参数在 main 前定案并记录 §二。"""
    import itertools
    for sA, sB, bf, af in itertools.product((0.5,), (1.0, 5.0, 10.0, 20.0),
                                            (0.15, 0.30), (0.60, 0.85)):
        saved = (S_A, S_B, B_FREQ, ALPHA_FOCAL)
        globals()["S_A"] = sA
        globals()["S_B"] = sB
        globals()["B_FREQ"] = bf
        globals()["ALPHA_FOCAL"] = af
        u = run_attention(seed, n_frames=n_frames, jitter=JITTER, alpha_focal=af)
        print("R_B2_SCAN_SA=%.1f_SB=%.1f_B=%.2f_AF=%.2f_CONSIST=%.4f_FP=%.4f_MB=%.6f_NSEC=%d_REDIRN=%d_REDIROK=%d" % (
            sA, sB, bf, af, u["consist_frac"], u["frac_pos"], u["mean_benefit"],
            u["n_sec"], u["redir_n"], u["redir_ok_n"]))
        globals()["S_A"] = saved[0]
        globals()["S_B"] = saved[1]
        globals()["B_FREQ"] = saved[2]
        globals()["ALPHA_FOCAL"] = saved[3]
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="at")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="诊断模式：单种子逐窗口 R_B2_DIAG_* 行（--seed 选择）")
    ap.add_argument("--scan", action="store_true",
                    help="诊断模式：参数扫描 R_B2_SCAN_* 行（找机制可公平测试的参数区）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.diag:
        return diag(seed=args.seed, n_frames=args.frames)
    if args.scan:
        return scan(seed=args.seed, n_frames=args.frames)

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    cfg = {"n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "jitter": args.jitter, "tag": args.tag,
           "level": LVCODE_B2,
           "scene": {"a_center": list(A_CENTER), "a_orbit": A_ORBIT,
                     "a_freq": A_FREQ, "a_regimes": A_REGIMES,
                     "b_center": list(B_CENTER), "b_orbit": B_ORBIT,
                     "b_freq": B_FREQ, "disk_r": DISK_R, "obj_gray": OBJ_GRAY,
                     "noise": NOISE_SIGMA, "s_a": S_A, "s_b": S_B,
                     "r_focal": R_FOCAL, "ctx_split_y": CTX_SPLIT_Y,
                     "occ_lum_thresh": OCC_LUM_THRESH, "a_min": A_MIN},
           "attention": {"alpha_focal": ALPHA_FOCAL, "loop": LOOP_CFG},
           "criteria": {"focus_min": FOCUS_MIN, "benefit_pos_min": BENEFIT_POS_MIN,
                        "redir_min": REDIR_MIN, "redir_nobs_min": REDIR_NOBS_MIN},
           "guards": {"so_r_range": [SO_R_LO, SO_R_HI],
                      "compose_ref": [COMPOSE_MAE_REF, COMPOSE_SC2_REF,
                                      COMPOSE_COMP_REF, COMPOSE_CHURN_REF],
                      "tol": TOL}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_at_%s.json" % ck_tag)

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
            per_unit[key] = run_attention(seed, n_frames=args.frames, jitter=args.jitter)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%d_%d" % (LVCODE_B2, s)] for s in seeds]

    # ---- 判据（docs/B2 §1.4 冻结） ----
    consist_vals = [u["consist_frac"] for u in units]
    consist_m, consist_sd = mean_sd(consist_vals)
    consist_ci = bootstrap_ci(consist_vals)
    c1 = consist_m >= FOCUS_MIN

    fp_vals = [u["frac_pos"] for u in units]
    mb_vals = [u["mean_benefit"] for u in units]
    fp_m, fp_sd = mean_sd(fp_vals)
    mb_m, mb_sd = mean_sd(mb_vals)
    mb_ci = bootstrap_ci(mb_vals)
    c2 = (fp_m >= BENEFIT_POS_MIN) and (mb_ci[0] > 0.0)

    redir_n_total = sum(u["redir_n"] for u in units)
    redir_ok_total = sum(u["redir_ok_n"] for u in units)
    redir_pooled = redir_ok_total / redir_n_total if redir_n_total > 0 else float("nan")
    n_sec_total = sum(u["n_sec"] for u in units)
    c3 = (redir_n_total >= REDIR_NOBS_MIN) and (redir_pooled >= REDIR_MIN)

    # ---- 守卫（docs/B2 §1.6 冻结；固定 10 种子，独立于主实验 n_seeds） ----
    g_so, so_r_m, so_r_sd, so_r_ci, so_diff_ci, so_auc, so_nobs = guard_so()
    g_comp, c_mae, c_mae_sd, c_sc2, c_sc2_sd, c_comp, c_comp_sd, c_churn = \
        guard_compose()
    guards_ok = (g_so == 1) and (g_comp == 1)

    # ---- 判定（docs/B2 §1.5 冻结） ----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif c1 and c2 and c3:
        verdict = "ATTENTION_EMERGES"
    elif not c1:
        verdict = "ATTENTION_FAIL"
    else:
        verdict = "PARTIAL"

    # ---- 内部确定性复现（docs/B2 §1.6-3；第二遍强制重算，不读 checkpoint） ----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for key in per_unit:
            for kk in REPRO_KEYS:
                if per_unit[key][kk] != per_unit2[key][kk]:
                    repro = 0

    out = {
        "artifact": "attention_emergence_test",
        "doc_ref": "lineB-motion-coupling/docs/B2",
        "config": cfg,
        "per_unit": per_unit,
        "criteria": {"c1_focus_consist": bool(c1), "c2_benefit_confirm": bool(c2),
                     "c3_redirect_correct": bool(c3),
                     "consist_mean": consist_m, "consist_sd": consist_sd,
                     "consist_ci95": list(consist_ci),
                     "frac_pos_mean": fp_m, "frac_pos_sd": fp_sd,
                     "mean_benefit": mb_m, "mean_benefit_sd": mb_sd,
                     "mean_benefit_ci95": list(mb_ci),
                     "redir_pooled": redir_pooled, "redir_n": redir_n_total,
                     "redir_ok": redir_ok_total, "n_sec": n_sec_total,
                     "redir_benefit_mean": round(float(np.nanmean(
                         [u["redir_benefit"] for u in units
                          if u["redir_benefit"] == u["redir_benefit"]])), 4)
                     if any(u["redir_benefit"] == u["redir_benefit"] for u in units)
                     else None},
        "guards": {"so": g_so, "so_r_mean": so_r_m, "so_r_sd": so_r_sd,
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
    res_path = os.path.join(args.out_dir, "at_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定） ----
    print("R_B2_TAG=%s" % args.tag)
    print("R_B2_SEEDS=%d" % len(seeds))
    print("R_B2_FRAMES=%d" % args.frames)
    for s in seeds:
        u = per_unit["%d_%d" % (LVCODE_B2, s)]
        print("R_B2_S%d_CONSIST=%.4f" % (s, u["consist_frac"]))
        print("R_B2_S%d_FRACPOS=%.4f" % (s, u["frac_pos"]))
        print("R_B2_S%d_MEANBEN=%.6f" % (s, u["mean_benefit"]))
        print("R_B2_S%d_NSEC=%d" % (s, u["n_sec"]))
        print("R_B2_S%d_REDIRN=%d" % (s, u["redir_n"]))
        print("R_B2_S%d_REDIROK=%d" % (s, u["redir_ok_n"]))
        print("R_B2_S%d_REDIRBEN=%.4f" % (s, u["redir_benefit"] if u["redir_benefit"] == u["redir_benefit"] else -1.0))
        print("R_B2_S%d_DETA=%.4f" % (s, u["det_err_a_mean"] if u["det_err_a_mean"] is not None else -1.0))
        print("R_B2_S%d_DETB=%.4f" % (s, u["det_err_b_mean"] if u["det_err_b_mean"] is not None else -1.0))
    print("R_B2_CONSIST_MEAN=%.4f" % consist_m)
    print("R_B2_CONSIST_SD=%.4f" % consist_sd)
    print("R_B2_CONSIST_LO=%.4f" % consist_ci[0])
    print("R_B2_CONSIST_HI=%.4f" % consist_ci[1])
    print("R_B2_FRACPOS_MEAN=%.4f" % fp_m)
    print("R_B2_FRACPOS_SD=%.4f" % fp_sd)
    print("R_B2_MEANBEN_MEAN=%.6f" % mb_m)
    print("R_B2_MEANBEN_LO=%.6f" % mb_ci[0])
    print("R_B2_MEANBEN_HI=%.6f" % mb_ci[1])
    print("R_B2_REDIR_POOLED=%.4f" % redir_pooled)
    print("R_B2_REDIR_N=%d" % redir_n_total)
    print("R_B2_REDIR_OK=%d" % redir_ok_total)
    print("R_B2_NSEC_TOTAL=%d" % n_sec_total)
    print("R_B2_C1=%s" % ("PASS" if c1 else "FAIL"))
    print("R_B2_C2=%s" % ("PASS" if c2 else "FAIL"))
    print("R_B2_C3=%s" % ("PASS" if c3 else "FAIL"))
    print("R_B2_GUARD_SO=%d" % g_so)
    print("R_B2_GUARD_SO_R=%.6f" % so_r_m)
    print("R_B2_GUARD_SO_R_SD=%.6f" % so_r_sd)
    print("R_B2_GUARD_SO_LO=%.6f" % so_r_ci[0])
    print("R_B2_GUARD_SO_HI=%.6f" % so_r_ci[1])
    print("R_B2_GUARD_SO_DIFF_LO=%.6f" % so_diff_ci[0])
    print("R_B2_GUARD_SO_DIFF_HI=%.6f" % so_diff_ci[1])
    print("R_B2_GUARD_SO_AUC=%.6f" % so_auc)
    print("R_B2_GUARD_SO_NOBS=%.1f" % so_nobs)
    print("R_B2_GUARD_COMPOSE=%d" % g_comp)
    print("R_B2_GUARD_COMPOSE_MAE=%.6f" % c_mae)
    print("R_B2_GUARD_COMPOSE_SC2=%.4f" % c_sc2)
    print("R_B2_GUARD_COMPOSE_COMP=%.4f" % c_comp)
    print("R_B2_GUARD_COMPOSE_CHURN=%.4f" % c_churn)
    print("R_B2_REPRO=%d" % repro)
    print("R_B2_VERDICT=%s" % verdict)
    print("R_B2_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
