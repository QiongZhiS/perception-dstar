"""lineB-motion-coupling/scripts/granularity_drill_test.py — B 路第四格：粒度递归——
场景 gist → 物体 → 纹路的缺陷驱动下钻（docs: lineB-motion-coupling/docs/B4-粒度递归-
缺陷驱动下钻-预注册设计.md §一 冻结）。

核心：用户洞察（"场景先行 → 单物体处理 → 纹路"；"缺陷 = 场景 gist 解释不了的剩余"）的
机制翻译——感知 = 沿粒度轴的递归组织（docs/221 递归读法在粒度轴的展开）：三层粒度预测器
（空间频率从粗到细）——gist（8px 块均值粗布局重建，docs/250 gist 先行）、object（焦点内
平滑模板 = 中频物体本体，docs/250 慢原型吸收）、texture（焦点内锐模板 = 高频图案，
docs/187 纹路被预测掉）；缺陷 = 该粒度解释不了的剩余 = 残差超预期 K·σ̂ 的相干区域
（docs/221 原子，零手工语义）；下钻 = 缺陷处注意聚焦（L1 复用 B2/B3 的 argmax(R×S)
机制，S 第一版外赋 = 均匀 1.0——纯残差驱动）；粒度状态机逐窗口 gist→object→texture→gist。

合成场景族（docs/232/235 风格，160×120，三风格 S0/S1/S2 粗布局 = gist 统计判别载体）：
移动亮圆盘（场景级缺陷载体，gist 逃逸）+ 盘上 2px 细条纹（物体级缺陷载体，颜色变化）+
段 2 出现的运动暗点（纹路级终态缺陷载体，非纹理结构）+ 静止纹理块（docs/187 锚点）。

判据（§1.4 冻结，docs/247 标签 [L1][机制][合成受控]——合成受控、非真实域证明；"缺陷"=
残差不宣称语义理解；docs/187 纹路纪律锚点）：
  C1 GIST_PRIOR   : 风格分类正确率 ≥ 0.80（gist 统计仅凭粗布局识别场景族）且 gist 先行 = 1.0
  C2 DEFECT_DRILL : 各级缺陷处下钻焦点落真缺陷 GT 的池化正确率 ≥ 0.80 且观察 ≥ 20
  C3 GRANULAR_ORDER: 下钻永不回粗（backward=0）+ 内容在正确粒度被吸收/不被吸收 ≥ 0.80
  C4 KEEP 守卫   : R_B4_GUARD_260 + R_B4_GUARD_B2 + R_B4_GUARD_B3 + R_B4_GUARD_SO +
                   R_B4_GUARD_COMPOSE + R_B4_REPRO 全 = 1
判定（§1.5）：全过 = GRANULAR_PASS；C1 过 C2/C3 不过 = PARTIAL；C1 不过 = GIST_FAIL；
守卫不过 = GUARD_FAIL。

守卫（§1.6 冻结；守卫种子数固定 10，不随主实验 n_seeds）：
  R_B4_GUARD_260    : import light_shadow_test.run_unit 重跑 docs/260 原场景 40 单位 →
                      det/fp/ld/sfs 与 1.0000/0.0000/0.4821/4.6258 逐位一致（容差 1e-3）
  R_B4_GUARD_B2     : import attention_emergence_test.run_attention 10 种子 →
                      CONSIST ≥ 0.80、FRACPOS ≥ 0.60、MEANBEN ∈ [0.03,0.06]（B2 实测
                      1.0000/1.0000/0.044675）
  R_B4_GUARD_B3     : import attention_cost_test.run_cost 10 种子 → NEG ∈ [0.15,0.45]、
                      TRIG ≥ 0.70、REDIR ≥ 0.70（B3 实测 0.2783/1.0000/0.9219）
  R_B4_GUARD_SO     : import attention_emergence_test.guard_so（docs/241 S1 长程 SO_info：
                      r ∈ [0.05,0.12] 且 CI 下界 > 0 且 diff_rand CI 下界 > 0）
  R_B4_GUARD_COMPOSE: import attention_emergence_test.guard_compose（docs/237 §3.2 逐位）
  R_B4_REPRO        : --repro 时主实验 30 运行整体重跑第二遍，关键数字位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_B4_* 摘要块
（顺序固定）；JSON 归档 lineB-motion-coupling/out/gd_<tag>.json + checkpoint
ckpt_gd_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 lineB-motion-coupling/out/*.log 与 lineB-motion-coupling/out/*.json 原文。
**未修改任何主线既有脚本**（vision/ 下全部不动；B1/B2/B3 脚本亦不动，只 import）。

用法：
  python lineB-motion-coupling/scripts/granularity_drill_test.py --n-seeds 10 --tag main --repro
  python lineB-motion-coupling/scripts/granularity_drill_test.py --n-seeds 1 --n-styles 1 --tag timing
  python lineB-motion-coupling/scripts/granularity_drill_test.py --diag --seed 0 --style 0
  python lineB-motion-coupling/scripts/granularity_drill_test.py --scan --seed 0 --style 0
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
for _p in (HERE, VISION, PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- import 复用 B2/B3（docs/B4 §1.3/§1.6：零改动，只 import）----
import attention_emergence_test as B2           # noqa: E402
import attention_cost_test as B3                # noqa: E402
from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER  # noqa: E402
from light_shadow_test import run_unit          # noqa: E402

top2_components = B2.top2_components
disk_mask = B2.disk_mask
guard_so = B2.guard_so
guard_compose = B2.guard_compose
run_attention = B2.run_attention
run_cost = B3.run_cost

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("lineB-motion-coupling", "out")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 场景旋钮（docs/B4 §1.2 冻结；载体旋钮经 §二 诊断定案，判据/机制语义不动）----
LVCODE_B4 = [41, 42, 43]          # 三风格 S0/S1/S2（与 0-6 / 20-23 / 30-33 / 40 流错开）
W, H = 160, 120
FPS = 30
N_FRAMES = 240
WINDOW = 10
BG_CELL = 24
BG_DARK, BG_BRIGHT = 64.0, 96.0
S2_DARK = 48.0                    # 风格 S2：暗格 48
S1_BAND_ROWS = (0, 40)            # 风格 S1：顶部横带（块对齐：5 块行）
S1_BAND_GRAY = 128.0
DISK_R = 10.0
OBJ_GRAY = 255.0
NOISE_SIGMA = 3.0
DISK_ORBIT_C = (80.0, 60.0)
DISK_ORBIT_R = 14.0
DISK_FREQ = 0.18                  # Hz（docs/260 同款）
STRIPE_OFF = (3.0, -2.0)          # 条纹块盘相对偏移（中心）
STRIPE_HALF = 3                   # 6x6 块（半宽 3）
STRIPE_DARK, STRIPE_BRIGHT = 80.0, 255.0   # §二 D 定案：120->80（加深使物体级残差
                                           # 0.44->~0.9，对 θ=K·P75（含条纹半影）余量
                                           # 充足；纹路级吸收残差仍 ≈ 噪声 < θ_t）
STRIPE_PERIOD = 2                 # px
DOT_CENTER_OFF = (-4.0, 0.0)      # 暗点小轨道中心（盘相对）
DOT_ORBIT_R = 2.0
DOT_FREQ = 1.5                    # Hz（§二 D 定案：0.5Hz 暗点 0.31px/帧被纹路模板
                                  # （τ≈2帧）部分跟踪 -> R_t≈0.1 终态缺陷弱化；1.5Hz =
                                  # 0.94px/帧 ≫ 模板跟踪尺度 -> 逃逸稳健；轨道盘直径 4px
                                  # 仍全在纹路裁剪（半宽 8）内）
DOT_R = 2.0
DOT_GRAY = 60.0
DOT_SEG_START = 120               # 段 2 起点（帧；= 窗口 12）
NOISE_BLOCK = (40, 64, 24, 48)    # 静止纹理块 y0,y1,x0,x1（块对齐 3x3）
NOISE_BLOCK_MEAN = 80.0
OCC_LUM_THRESH = 220.0            # 亮检测 = L > log(221)（docs/260 同款）
A_MIN_DET = 25                    # 检测连通域最小面积（px）

# ---- 三层粒度机制旋钮（docs/B4 §1.3 冻结；载体旋钮经 §二 诊断定案）----
GIST_BLOCK = 8                    # gist 块均值尺度（px；20x15 粗图）
K_DEFECT = 3.0                    # 缺陷 = 残差 > max(K·σ̂, THETA_FLOOR)
THETA_FLOOR = 0.10                # 噪声底下限（§二 D 定案 ≈ 3× 中灰对数噪声 0.035）
SIGMA_O = 1.5                     # 物体平滑模板高斯 σ（§二 D 定案：2.5->1.5——半影
                                  # （平滑模板在条纹周围的模糊误差区）收窄，θ 不被
                                  # 抬高；仍 ≥ 条纹半周期 1px 使条纹逃逸）
ALPHA_O = 0.5                     # 物体模板 EWMA 速率（= CPLoop α_fast 同款）
ALPHA_T = 0.5                     # 纹路模板 EWMA 速率（逐窗口重初始化，§二 D 定案）
K_DRILL = 3                       # 缺陷持久确认：窗口内 ≥ 3 帧（≥30%）
A_MIN_G = 25                      # gist 缺陷最小面积（px）
A_MIN_O = 5                       # 物体缺陷最小面积（px；§二 D 定案：10->5——2px 周期
                                  # 条纹的暗列彼此隔 1px 亮隙，8-连通为 3 个 6px 独立列，
                                  # 10 会把它们全部过滤 -> 物体级缺陷漏检；5 = 暗列
                                  # 合法小缺陷组件）
A_MIN_T = 5                       # 纹路缺陷最小面积（px）
DILATE = 2                        # GT 正确性判定的膨胀（px）
OBJ_CROP = 40                     # 物体焦点裁剪（px）
TEX_CROP = 16                     # 纹路焦点裁剪（px）
INTERIOR_R = 8.0                  # 物体缺陷搜索域 = 盘内部 r≤8（排除盘缘环带）

# ---- 判据阈值（docs/B4 §1.4 冻结）----
STYLE_ACC_MIN = 0.80
DRILL_ACC_MIN = 0.80
DRILL_NOBS_MIN = 20
ABSORB_MIN = 0.80

# ---- 守卫容差（docs/B4 §1.6 冻结）----
TOL = 1e-3
B2_CONSIST_MIN = 0.80
B2_FRACPOS_MIN = 0.60
B2_MEANBEN_LO, B2_MEANBEN_HI = 0.03, 0.06
B3_NEG_LO, B3_NEG_HI = 0.15, 0.45
B3_TRIG_MIN = 0.70
B3_REDIR_MIN = 0.70

REPRO_KEYS = ["style_hat", "F1_ok", "F2_ok", "F3_ok",
              "drill1_n", "drill1_ok", "drill2_n", "drill2_ok",
              "drill3_n", "drill3_ok", "backward_n",
              "absorb_disk", "absorb_stripes", "absorb_dot",
              "theta_g_mean", "theta_o_mean", "theta_t_mean",
              "Rg_mean", "Ro_mean", "Rt_mean", "noise_block_ev"]


# ---------------- 合成场景（docs/B4 §1.2 冻结） ----------------
def _style_layout(style):
    """风格规范粗布局（20x15 块灰度，仅背景；S1 带 / S2 暗格）。"""
    bg = np.zeros((H, W), np.float32)
    for y in range(0, H, BG_CELL):
        for x in range(0, W, BG_CELL):
            dark = BG_DARK if style != 2 else S2_DARK
            val = dark if ((x // BG_CELL) + (y // BG_CELL)) % 2 == 0 else BG_BRIGHT
            bg[y:y + BG_CELL, x:x + BG_CELL] = val
    if style == 1:
        bg[S1_BAND_ROWS[0]:S1_BAND_ROWS[1], :] = S1_BAND_GRAY
    return bg


def make_b4_scene(seed, style, n_frames=N_FRAMES, width=W, height=H, fps=FPS,
                  jitter=JITTER):
    """生成 (风格, 种子) 的灰度帧序列 + GT（只用于事后测量，绝不进入机制）。
    确定性：rng 由 (seed, lvcode) 派生；调用顺序固定（noise_mult -> 目标相位 ->
    暗点相位 -> 静止纹理块噪声模式）。段 2（帧 ≥ DOT_SEG_START）暗点出现。"""
    lvcode = LVCODE_B4[style]
    rng = np.random.default_rng(seed * 7919 + lvcode * 104729 + 13)
    noise_mult = rng.uniform(1 - jitter, 1 + jitter) if jitter > 0 else 1.0
    sigma = NOISE_SIGMA * noise_mult
    ph_d = rng.uniform(0, 2 * np.pi)
    ph_dot = rng.uniform(0, 2 * np.pi)
    blk_pat = rng.normal(0, sigma, (NOISE_BLOCK[1] - NOISE_BLOCK[0],
                                    NOISE_BLOCK[3] - NOISE_BLOCK[2])).astype(np.float32)

    bg = _style_layout(style)
    bg = bg.copy()
    blk = bg[NOISE_BLOCK[0]:NOISE_BLOCK[1], NOISE_BLOCK[2]:NOISE_BLOCK[3]]
    bg[NOISE_BLOCK[0]:NOISE_BLOCK[1], NOISE_BLOCK[2]:NOISE_BLOCK[3]] = \
        np.clip(NOISE_BLOCK_MEAN + blk_pat, 0, 255)

    frames = []
    disk_pos = []
    dot_pos = []
    th_d = ph_d
    th_dot = ph_dot
    for t in range(n_frames):
        th_d += 2 * np.pi * DISK_FREQ / fps
        ax = DISK_ORBIT_C[0] + DISK_ORBIT_R * np.cos(th_d)
        ay = DISK_ORBIT_C[1] + DISK_ORBIT_R * np.sin(th_d)
        img = bg.copy()
        cv2.circle(img, (int(round(ax)), int(round(ay))), int(DISK_R), OBJ_GRAY, -1)
        # 条纹块（盘相对偏移 STRIPE_OFF，6x6，2px 竖条纹 120/255）
        px = int(round(ax + STRIPE_OFF[0]))
        py = int(round(ay + STRIPE_OFF[1]))
        for yy in range(py - STRIPE_HALF, py + STRIPE_HALF):
            for xx in range(px - STRIPE_HALF, px + STRIPE_HALF):
                if 0 <= yy < height and 0 <= xx < width:
                    val = STRIPE_DARK if ((xx - px) % STRIPE_PERIOD) == 0 else STRIPE_BRIGHT
                    img[yy, xx] = val
        # 暗点（段 2；盘相对小轨道）
        cur_dot = None
        if t >= DOT_SEG_START:
            th_dot += 2 * np.pi * DOT_FREQ / fps
            dx = ax + (DOT_CENTER_OFF[0] + DOT_ORBIT_R * np.cos(th_dot))
            dy = ay + (DOT_CENTER_OFF[1] + DOT_ORBIT_R * np.sin(th_dot))
            cv2.circle(img, (int(round(dx)), int(round(dy))), int(DOT_R), DOT_GRAY, -1)
            cur_dot = (float(dx), float(dy))
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        disk_pos.append((float(ax), float(ay)))
        dot_pos.append(cur_dot)
    gts = dict(style=style, seed=seed, disk_pos=disk_pos, dot_pos=dot_pos,
               stripe_off=STRIPE_OFF, stripe_half=STRIPE_HALF,
               noise_block=NOISE_BLOCK, dot_r=DOT_R)
    return frames, gts


# ---------------- 层级处理原语（docs/B4 §1.3 冻结；噪声尺度估计器经 §二 诊断定案） ----------------
def mad(arr):
    m = float(np.median(arr))
    return float(np.median(np.abs(arr - m)) / 0.6745)


def noise_scale(arr):
    """稳健噪声尺度（§二 D 诊断定案）：MAD 对双峰残差低估、P90-P50 被缺陷抬高、
    P75 在最大聚合下被缺陷（占域 ~25%）抬高——均使 θ 失真；定案 = P50（中位数）=
    "该粒度典型残差"（缺陷占比 <50% 稳健）+ 噪声底下限 THETA_FLOOR（防亮域裁剪
    （255 上噪声被 clip 掉 -> 残差恒 0）使纹路层 P50 退化为 0）。"""
    a = np.asarray(arr, np.float32).ravel()
    if a.size == 0:
        return 0.0
    return float(np.percentile(a, 50))


def coarse_image(Ls):
    """8px 块均值粗图（20x15）；Ls = 窗口帧 log 亮度列表（160x120）。"""
    n = len(Ls)
    stack = np.stack(Ls)
    h = stack.shape[1] // GIST_BLOCK
    wdt = stack.shape[2] // GIST_BLOCK
    c = np.zeros((h, wdt), np.float32)
    for by in range(h):
        for bx in range(wdt):
            blk = stack[:, by * GIST_BLOCK:(by + 1) * GIST_BLOCK,
                        bx * GIST_BLOCK:(bx + 1) * GIST_BLOCK]
            c[by, bx] = float(blk.mean())
    return c


def coarse_recon(c):
    """最近邻上采样回 160x120。"""
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)


def style_prototypes():
    """三风格规范粗布局原型（log 域；构造解析，无圆盘）。"""
    outs = []
    for s in range(3):
        bg = _style_layout(s)
        L = np.log(np.maximum(bg, 1.0))
        outs.append(coarse_image([L]))
    return outs


def components(mask, min_area):
    """8-连通域：[(mask, area, centroid)]，面积 >= min_area。"""
    if mask.sum() == 0:
        return []
    m = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return []
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    out = []
    for idx in range(1, n):
        if areas[idx] < min_area:
            continue
        comp = lab == idx
        ys, xs = np.nonzero(comp)
        out.append((comp, int(areas[idx]), (float(xs.mean()), float(ys.mean()))))
    return out


def dilate(mask, r=DILATE):
    if mask.sum() == 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1),
                                                     np.uint8)).astype(bool)


def point_in_gt(pt, gt_mask):
    """下钻焦点正确 = 落在 GT 掩码（已膨胀）内。pt = (x, y) 浮点。"""
    x, y = int(round(pt[0])), int(round(pt[1]))
    if 0 <= y < gt_mask.shape[0] and 0 <= x < gt_mask.shape[1]:
        return bool(gt_mask[y, x])
    return False


# ---------------- 主实验单位（docs/B4 §1.3 冻结） ----------------
def run_b4(seed, style, n_frames=N_FRAMES, jitter=JITTER, k_defect=K_DEFECT,
           sigma_o=SIGMA_O, alpha_o=ALPHA_O, alpha_t=ALPHA_T, k_drill=K_DRILL):
    """跑 (风格, 种子) 一次完整运行 → 三层粒度残差/缺陷/下钻行为量 + 判据组件。

    粒度状态机（逐窗口）：L(0)=gist；L(w+1)=next(L(w)) 若 L(w) 发现缺陷否则 gist；
    next: gist->object->texture->gist。窗口序列 = 0,3,..,21 gist；1,4,..,22 object；
    2,5,..,23 texture（8 周期）。gist 窗口：粗布局重建 + 风格分类 + drill1；object
    窗口：盘心跟踪注册裁剪 + 平滑模板 + drill2；texture 窗口：F2 处注册裁剪 + 锐
    模板 + drill3（段 2 暗点终态）/吸收（段 1 条纹）。"""
    frames, gts = make_b4_scene(seed, style, n_frames=n_frames, jitter=jitter)
    n_w = max(1, n_frames // WINDOW)
    style_idx = style
    stripe_off = gts["stripe_off"]
    stripe_half = gts["stripe_half"]
    nb = gts["noise_block"]

    # ---- 检测循环：盘心亮域 top-1 质心（全帧；与层级处理解耦）----
    cD = [None] * n_frames
    last = None
    det_errs = []
    for t, g in enumerate(frames):
        bright = np.log(np.maximum(g.astype(np.float32), 1.0)) > np.log(OCC_LUM_THRESH + 1.0)
        comps = top2_components(bright, min_area=A_MIN_DET)
        cur = comps[0][2] if comps else None
        if cur is not None:
            last = cur
        cD[t] = last
        if last is not None:
            det_errs.append(np.hypot(last[0] - gts["disk_pos"][t][0],
                                     last[1] - gts["disk_pos"][t][1]))

    # ---- docs/187 锚点：全分辨率预测回路（CPLoop）静止纹理块事件量 ----
    noise_evs = []
    cp_cfg = {k: v for k, v in B2.LOOP_CFG.items() if k in B2.CP_KEYS}
    loop = CPLoop(window=WINDOW, **cp_cfg)
    for t, g in enumerate(frames):
        loop.step(g)
        if t >= 30 and loop.bg_fast is not None and loop._frame_buf:
            th = loop._frame_buf[-1]["theta"]
            db = loop._frame_buf[-1]["db"]
            L = np.log(np.maximum(g.astype(np.float32), 1.0))
            r = np.abs(L - loop.bg_fast)
            ev = np.maximum(r - db, 0.0) > th
            blk = ev[nb[0]:nb[1], nb[2]:nb[3]]
            noise_evs.append(float(blk.mean()))
    noise_block_ev = float(np.mean(noise_evs)) if noise_evs else 0.0

    # ---- GT 掩码（盘相对坐标；仅事后测量）----
    def stripe_gt_rel():
        m = np.zeros((H, W), bool)
        px0 = int(round(stripe_off[0] + 60))  # 以 (60,60) 为盘相对原点（仅形状）
        py0 = int(round(stripe_off[1] + 60))
        for yy in range(py0 - stripe_half, py0 + stripe_half):
            for xx in range(px0 - stripe_half, px0 + stripe_half):
                if 0 <= yy < H and 0 <= xx < W:
                    m[yy, xx] = True
        # 与盘内部 r<=8 求交（物体缺陷域）
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        interior = (xx - 60) ** 2 + (yy - 60) ** 2 <= INTERIOR_R ** 2
        return m & interior

    def dot_gt_rel(t):
        m = np.zeros((H, W), bool)
        dp = gts["dot_pos"][t]
        if dp is None:
            return m
        dx = dp[0] - gts["disk_pos"][t][0] + 60
        dy = dp[1] - gts["disk_pos"][t][1] + 60
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        return (xx - dx) ** 2 + (yy - dy) ** 2 <= (gts["dot_r"] + DILATE) ** 2

    def disk_gt_window(lo, hi):
        m = np.zeros((H, W), bool)
        for t in range(lo, min(hi, n_frames)):
            m |= disk_mask(gts["disk_pos"][t], r=DISK_R, shape=(H, W))
        return dilate(m, DILATE)

    def stripe_gt_window(lo, hi):
        m = np.zeros((H, W), bool)
        rel = stripe_gt_rel()
        for t in range(lo, min(hi, n_frames)):
            dx = gts["disk_pos"][t][0]
            dy = gts["disk_pos"][t][1]
            # 平移：盘相对 (60,60) 原点 -> 全局
            m |= np.roll(np.roll(rel, int(round(dx - 60)), axis=1),
                         int(round(dy - 60)), axis=0)
        return dilate(m, DILATE)

    def dot_gt_window(lo, hi):
        m = np.zeros((H, W), bool)
        for t in range(lo, min(hi, n_frames)):
            m |= dot_gt_rel(t)
        return m

    # ---- 状态机与行为量记录 ----
    level = [0] * n_w
    gist_ws = [w for w in range(n_w) if w % 3 == 0]   # gist 窗口 = 每周期首窗
    n_cycles = len(gist_ws)
    style_hat = [None] * n_w
    F1_ok = [0] * n_w
    F2_ok = [0] * n_w
    F3_ok = [0] * n_w
    theta_g_rec = [0.0] * n_w
    theta_o_rec = [0.0] * n_w
    theta_t_rec = [0.0] * n_w
    Rg_rec = [0.0] * n_w
    Ro_rec = [0.0] * n_w
    Rt_rec = [0.0] * n_w
    F1_rec = [None] * n_w
    F2_off_rec = [None] * n_w
    drill1_n = drill1_ok = drill2_n = drill2_ok = drill3_n = drill3_ok = 0
    backward_n = 0
    absorb_disk_n = absorb_disk_ok = 0
    absorb_str_n = absorb_str_ok = 0
    absorb_dot_n = absorb_dot_ok = 0
    gist_first = 0
    tex_outside_focus = 0
    F2_off = None            # 下钻 2 焦点（盘相对偏移）；texture 窗口使用

    To = None                # 物体模板（OBJ_CROP x OBJ_CROP）
    Tt = None                # 纹路模板（TEX_CROP x TEX_CROP）
    prots = style_prototypes()
    tex_detail = []          # 逐纹路窗口终态明细（诊断）

    def process_gist(w, lo, hi):
        Ls = []
        for t in range(lo, min(hi, n_frames)):
            Ls.append(np.log(np.maximum(frames[t].astype(np.float32), 1.0)))
        c = coarse_image(Ls)
        # 风格分类（gist 统计仅凭粗布局 vs 规范原型）
        dists = [float(np.mean(np.abs(c - p))) for p in prots]
        style_hat[w] = int(np.argmin(dists))
        # gist 残差场（逐帧 |L - 重建| -> 窗口均值）
        Rg_frames = []
        for L in Ls:
            G = coarse_recon(coarse_image([L]))
            Rg_frames.append(np.abs(L - G))
        Rg = np.mean(np.stack(Rg_frames), axis=0)
        Rg_rec[w] = float(Rg.mean())
        sg = noise_scale(Rg)
        thg = max(k_defect * sg, THETA_FLOOR)
        theta_g_rec[w] = thg
        # 缺陷候选 + 持久确认（窗口内 >= k_drill 帧区域均值超阈值）
        mask = Rg > thg
        cands = components(mask, A_MIN_G)
        conf = []
        for (comp, area, cent) in cands:
            n_hi = sum(1 for rf in Rg_frames if float(rf[comp].mean()) > thg)
            if n_hi >= k_drill:
                conf.append((comp, area, cent))
        # drill1：argmax V = argmax(max R over candidate)（S=1.0 外赋利害第一版）
        if conf:
            Vs = [float(Rg[comp].max()) for (comp, _, _) in conf]
            winner = int(np.argmax(Vs))
            comp, area, cent = conf[winner]
            F1 = cent
            F1_rec[w] = F1
            ok = point_in_gt(F1, disk_gt_window(lo, hi))
            F1_ok[w] = 1 if ok else 0
            nonlocal_inc("drill1", ok)
            return True
        return False

    def nonlocal_inc(kind, ok):
        nonlocal drill1_n, drill1_ok, drill2_n, drill2_ok, drill3_n, drill3_ok
        if kind == "drill1":
            drill1_n += 1
            drill1_ok += 1 if ok else 0
        elif kind == "drill2":
            drill2_n += 1
            drill2_ok += 1 if ok else 0
        else:
            drill3_n += 1
            drill3_ok += 1 if ok else 0

    def process_object(w, lo, hi):
        nonlocal To, F2_off
        interior = disk_mask((OBJ_CROP // 2, OBJ_CROP // 2), r=INTERIOR_R,
                             shape=(OBJ_CROP, OBJ_CROP))
        Dmask = disk_mask((OBJ_CROP // 2, OBJ_CROP // 2), r=DISK_R,
                          shape=(OBJ_CROP, OBJ_CROP)).astype(np.float32)
        Ro_frames = []
        for t in range(lo, min(hi, n_frames)):
            cd = cD[t]
            if cd is None:
                Ro_frames.append(None)
                continue
            cx0 = int(round(cd[0])) - OBJ_CROP // 2
            cy0 = int(round(cd[1])) - OBJ_CROP // 2
            crop = frames[t][cy0:cy0 + OBJ_CROP, cx0:cx0 + OBJ_CROP].astype(np.float32)
            Lc = np.log(np.maximum(crop, 1.0))
            if To is None:
                To = Lc.copy()
            else:
                To = alpha_o * Lc + (1.0 - alpha_o) * To
            # 掩膜归一化平滑（§二 D 诊断定案：全图模糊把背景漏进盘内部 -> R_o 域整体
            # 抬升、θ 失真；改 = 盘掩膜内模糊/掩膜模糊归一化——"物体本体"的中频预测）
            num = cv2.GaussianBlur(To * Dmask, (0, 0), sigma_o)
            den = cv2.GaussianBlur(Dmask, (0, 0), sigma_o)
            Po = num / np.maximum(den, 1e-6)
            Ro_frames.append(np.abs(Lc - Po))
        valid = [r for r in Ro_frames if r is not None]
        if not valid:
            return False
        # §二 D 定案：窗口聚合 = 最大（瞬态缺陷（运动暗点）的"最强落空"语义——均值
        # 被扫掠稀释使暗点 < 条纹（0.6 vs 0.86）-> seg2 下钻焦点错选；最大场暗点
        # 1.45 > 条纹 0.86 -> argmax 正确）
        Ro = np.maximum.reduce(np.stack(valid), axis=0)
        Ro_rec[w] = float(Ro[interior].mean())
        so = noise_scale(Ro[interior])
        tho = max(k_defect * so, THETA_FLOOR)
        theta_o_rec[w] = tho
        mask = (Ro > tho) & interior
        cands = components(mask, A_MIN_O)
        conf = []
        for (comp, area, cent) in cands:
            n_hi = sum(1 for r in valid
                       if float(r[comp].max()) > tho)
            if n_hi >= k_drill:
                conf.append((comp, area, cent))
        # 吸收度量（C3）
        body_med = float(np.median(Ro[interior]))
        absorb_disk_ok_c = 1 if body_med < tho else 0
        nonlocal_abs("disk", absorb_disk_ok_c)
        # 条纹区（盘相对 -> 裁剪坐标）：中心 = (20+3, 20-2)；测量域 = 暗条纹列
        # （条纹实际内容；亮列是平滑模板半影——§二 D 定案：全 6x6 中位数被半影拉低
        # 至 θ_o 附近 -> 段 2 误判"物体级已吸收"；暗列中位数 ≈ 0.88 稳健）
        scx = OBJ_CROP // 2 + int(round(stripe_off[0]))
        scy = OBJ_CROP // 2 + int(round(stripe_off[1]))
        sr = stripe_half
        sm = np.zeros((OBJ_CROP, OBJ_CROP), bool)
        for yy in range(scy - sr, scy + sr):
            for xx in range(scx - sr, scx + sr):
                if 0 <= yy < OBJ_CROP and 0 <= xx < OBJ_CROP:
                    if ((xx - scx) % STRIPE_PERIOD) == 0:
                        sm[yy, xx] = True
        stripe_med_o = float(np.median(Ro[sm & interior])) if (sm & interior).any() \
            else float("nan")
        # 条纹在物体级不被吸收（需纹路级）
        nonlocal_abs("stripes", 1 if (stripe_med_o == stripe_med_o and
                                      stripe_med_o > tho) else 0)
        # drill2
        if conf:
            Vs = [float(Ro[comp].max()) for (comp, _, _) in conf]
            winner = int(np.argmax(Vs))
            comp, area, cent = conf[winner]
            F2_rel = (cent[0] - OBJ_CROP // 2, cent[1] - OBJ_CROP // 2)
            F2_off = F2_rel
            F2_off_rec[w] = list(F2_rel)
            # 正确 = F2（盘相对）落 seg1 条纹 GT / seg2 暗点 GT（膨胀 2）
            if w < DOT_SEG_START // WINDOW:
                gt = stripe_gt_rel()
            else:
                gt = np.zeros((H, W), bool)
                for t in range(lo, min(hi, n_frames)):
                    gt |= dot_gt_rel(t)
            gt = dilate(gt, DILATE)
            ok = point_in_gt((F2_rel[0] + 60, F2_rel[1] + 60), gt)
            F2_ok[w] = 1 if ok else 0
            nonlocal_inc("drill2", ok)
            return True
        return False

    def nonlocal_abs(kind, ok):
        nonlocal absorb_disk_n, absorb_disk_ok, absorb_str_n, absorb_str_ok, \
            absorb_dot_n, absorb_dot_ok
        if kind == "disk":
            absorb_disk_n += 1
            absorb_disk_ok += ok
        elif kind == "stripes":
            absorb_str_n += 1
            absorb_str_ok += ok
        else:
            absorb_dot_n += 1
            absorb_dot_ok += ok

    def process_texture(w, lo, hi):
        nonlocal Tt, F2_off
        if F2_off is None:
            return False
        Tt = None                     # §二 D 定案：逐窗口重初始化——消除注册错位瞬态
                                      # （旧 F2 注册的模板 vs 新裁剪内容错位 -> 条纹
                                      # 错位假候选抢 argmax）；静态条纹首帧即被吸收、
                                      # 运动暗点永远逃逸（docs/187）
        # 纹路缺陷域 = 裁剪 ∩ 盘内部（r<=8 围绕盘心在裁剪内的位置）——背景滚动/
        # 盘缘排除（§二 D 定案：物体表面细粒度的忠实域；docs/187 纹路只在焦点处理）
        dcx = TEX_CROP // 2 - int(round(F2_off[0]))
        dcy = TEX_CROP // 2 - int(round(F2_off[1]))
        interior_t = disk_mask((dcx, dcy), r=INTERIOR_R, shape=(TEX_CROP, TEX_CROP))
        Rt_frames = []
        for t in range(lo, min(hi, n_frames)):
            cd = cD[t]
            if cd is None:
                Rt_frames.append(None)
                continue
            cxc = int(round(cd[0] + F2_off[0])) - TEX_CROP // 2
            cyc = int(round(cd[1] + F2_off[1])) - TEX_CROP // 2
            cxc = int(np.clip(cxc, 0, W - TEX_CROP))
            cyc = int(np.clip(cyc, 0, H - TEX_CROP))
            crop = frames[t][cyc:cyc + TEX_CROP, cxc:cxc + TEX_CROP].astype(np.float32)
            Lc = np.log(np.maximum(crop, 1.0))
            if Tt is None:
                Tt = Lc.copy()
            else:
                Tt = alpha_t * Lc + (1.0 - alpha_t) * Tt
            Rt_frames.append(np.abs(Lc - Tt))
        valid = [r for r in Rt_frames if r is not None]
        if not valid:
            return False
        # §二 D 定案：窗口聚合 = 最大（同物体层——运动暗点的最强落空）
        Rt = np.maximum.reduce(np.stack(valid), axis=0)
        Rt_rec[w] = float(Rt.mean())
        st = noise_scale(Rt[interior_t]) if interior_t.any() else noise_scale(Rt)
        tht = max(k_defect * st, THETA_FLOOR)
        theta_t_rec[w] = tht
        mask = (Rt > tht) & interior_t
        cands = components(mask, A_MIN_T)
        conf = []
        for (comp, area, cent) in cands:
            n_hi = sum(1 for r in valid
                       if float(r[comp].max()) > tht)
            if n_hi >= k_drill:
                conf.append((comp, area, cent))
        # 吸收度量：条纹区在纹路级被吸收（裁剪坐标 = (条纹盘相对 - F2_off) + 中心）
        # §二 D 定案：段 2（纹路焦点 = 暗点，条纹不在焦点内）不测条纹纹路级吸收
        # （焦点外内容非纹路粒度处理对象——docs/187 纹路只在焦点处理）；段 1 测。
        sr_cx = TEX_CROP // 2 + int(round(stripe_off[0] - F2_off[0]))
        sr_cy = TEX_CROP // 2 + int(round(stripe_off[1] - F2_off[1]))
        sr = stripe_half
        sm = np.zeros((TEX_CROP, TEX_CROP), bool)
        for yy in range(sr_cy - sr, sr_cy + sr):
            for xx in range(sr_cx - sr, sr_cx + sr):
                if 0 <= yy < TEX_CROP and 0 <= xx < TEX_CROP:
                    sm[yy, xx] = True
        if w < DOT_SEG_START // WINDOW:
            stripe_med_t = float(np.median(Rt[sm])) if sm.any() else float("nan")
            stripe_ok = 1 if (stripe_med_t == stripe_med_t and
                              stripe_med_t < tht) else 0
            tex_detail.append(dict(w=w, tht=round(tht, 6), dot_med=None,
                                   dot_ok=-1, f2_off=list(F2_off),
                                   stripe_med=(round(stripe_med_t, 6)
                                               if stripe_med_t == stripe_med_t
                                               else None),
                                   stripe_ok=stripe_ok))
            nonlocal_abs("stripes", stripe_ok)
        # 暗点区（段 2）：终态不吸收（R_t > θ_t；逐帧暗点圆 = 暗点实际像素，非窗口
        # 扫掠带并集——§二 D 定案：并集被本体像素稀释中位数）
        if w >= DOT_SEG_START // WINDOW:
            dot_meds = []
            for t in range(lo, min(hi, n_frames)):
                dp = gts["dot_pos"][t]
                if dp is None:
                    continue
                dr = (dp[0] - gts["disk_pos"][t][0]) - F2_off[0]
                dc = (dp[1] - gts["disk_pos"][t][1]) - F2_off[1]
                yy, xx = np.mgrid[0:TEX_CROP, 0:TEX_CROP].astype(np.float32)
                dm = (xx - (TEX_CROP // 2 + dr)) ** 2 + \
                     (yy - (TEX_CROP // 2 + dc)) ** 2 <= gts["dot_r"] ** 2
                if dm.any():
                    dot_meds.append(float(np.median(Rt[dm])))
            dot_med_t = float(np.mean(dot_meds)) if dot_meds else float("nan")
            dot_ok = 1 if (dot_med_t == dot_med_t and dot_med_t > tht) else 0
            tex_detail.append(dict(w=w, tht=round(tht, 6),
                                   dot_med=(round(dot_med_t, 6)
                                            if dot_med_t == dot_med_t else None),
                                   dot_ok=dot_ok,
                                   f2_off=list(F2_off),
                                   stripe_med=None, stripe_ok=-1))
            nonlocal_abs("dot", dot_ok)
            # drill3（终态）：焦点落暗点 GT
            if conf:
                Vs = [float(Rt[comp].max()) for (comp, _, _) in conf]
                winner = int(np.argmax(Vs))
                comp, area, cent = conf[winner]
                F3_rel = (F2_off[0] + (cent[0] - TEX_CROP // 2),
                          F2_off[1] + (cent[1] - TEX_CROP // 2))
                gt = np.zeros((H, W), bool)
                for t in range(lo, min(hi, n_frames)):
                    gt |= dot_gt_rel(t)
                gt = dilate(gt, DILATE)
                ok = point_in_gt((F3_rel[0] + 60, F3_rel[1] + 60), gt)
                F3_ok[w] = 1 if ok else 0
                nonlocal_inc("drill3", ok)
                return True
        return False

    # ---- 主循环（粒度状态机）----
    for w in range(n_w):
        lo = w * WINDOW
        hi = min(lo + WINDOW, n_frames)
        if level[w] == 0:
            found = process_gist(w, lo, hi)
            if w in gist_ws:
                gist_first += 1
            nxt = 1 if found else 0
        elif level[w] == 1:
            found = process_object(w, lo, hi)
            nxt = 2 if found else 0
        else:
            found = process_texture(w, lo, hi)
            nxt = 0
        if w + 1 < n_w:
            # 粒度单调：drill 事件粒度严格 +1（回粗计数 = 事件后粒度不增；texture->gist
            # 是周期重启非下钻事件，不计）
            if found and nxt <= level[w] and level[w] < 2:
                backward_n += 1
            level[w + 1] = nxt
        # 纹路窗口焦点外计数（docs/187 锚点：纹路只在焦点处理；F2 恒为盘内偏移）
        if level[w] == 2 and F2_off is not None:
            if not (-DISK_R - 4 <= F2_off[0] <= DISK_R + 4 and
                    -DISK_R - 4 <= F2_off[1] <= DISK_R + 4):
                tex_outside_focus += 1

    wins = list(range(n_w))
    style_correct = sum(1 for w in gist_ws if style_hat[w] == style_idx)
    style_acc = style_correct / max(1, len(gist_ws))
    gist_first_frac = gist_first / float(n_cycles) if n_cycles else 1.0
    drill_n = drill1_n + drill2_n + drill3_n
    drill_ok = drill1_ok + drill2_ok + drill3_ok
    drill_acc = drill_ok / drill_n if drill_n > 0 else float("nan")
    absorb_total_n = absorb_disk_n + absorb_str_n + absorb_dot_n
    absorb_total_ok = absorb_disk_ok + absorb_str_ok + absorb_dot_ok
    absorb_correct = absorb_total_ok / absorb_total_n if absorb_total_n > 0 \
        else float("nan")

    out = dict(seed=seed, style=style_idx, level=LVCODE_B4[style_idx],
               frames=n_frames,
               style_acc=round(style_acc, 6), style_hat=style_hat,
               gist_first_frac=round(gist_first_frac, 6),
               drill1_n=drill1_n, drill1_ok=drill1_ok,
               drill2_n=drill2_n, drill2_ok=drill2_ok,
               drill3_n=drill3_n, drill3_ok=drill3_ok,
               drill_acc=round(drill_acc, 6) if drill_acc == drill_acc else None,
               drill_n=drill_n, drill_ok=drill_ok,
               backward_n=backward_n,
               absorb_disk=(round(absorb_disk_ok / absorb_disk_n, 6)
                            if absorb_disk_n else None),
               absorb_stripes=(round(absorb_str_ok / absorb_str_n, 6)
                               if absorb_str_n else None),
               absorb_dot=(round(absorb_dot_ok / absorb_dot_n, 6)
                           if absorb_dot_n else None),
               absorb_correct=round(absorb_correct, 6)
               if absorb_correct == absorb_correct else None,
               F1_ok=F1_ok, F2_ok=F2_ok, F3_ok=F3_ok,
               F2_off=F2_off_rec,
               theta_g=[round(v, 6) for v in theta_g_rec],
               theta_o=[round(v, 6) for v in theta_o_rec],
               theta_t=[round(v, 6) for v in theta_t_rec],
               Rg=[round(v, 6) for v in Rg_rec],
               Ro=[round(v, 6) for v in Ro_rec],
               Rt=[round(v, 6) for v in Rt_rec],
               theta_g_mean=round(float(np.mean(theta_g_rec)), 6),
               theta_o_mean=round(float(np.mean(theta_o_rec)), 6),
               theta_t_mean=round(float(np.mean(theta_t_rec)), 6),
               Rg_mean=round(float(np.mean(Rg_rec)), 6),
               Ro_mean=round(float(np.mean(Ro_rec)), 6),
               Rt_mean=round(float(np.mean(Rt_rec)), 6),
               noise_block_ev=round(noise_block_ev, 6),
               tex_outside_focus=tex_outside_focus,
               tex_detail=tex_detail,
               det_err_mean=round(float(np.mean(det_errs)), 4) if det_errs else None)
    return out


# ---------------- 守卫（docs/B4 §1.6 冻结；守卫种子数固定 10） ----------------
def guard_260(n_seeds=10, n_frames=240):
    """R_B4_GUARD_260：docs/260 三连数字复现（import light_shadow_test.run_unit 重跑
    全部原场景 4 级 x 10 种子 = 40 单位，B1 guard_cell260 同款）：det/fp/ld/sfs 与
    1.0000/0.0000/0.4821/4.6258 逐位一致（容差 1e-3）。"""
    dets, fps, lds, sfss = [], [], [], []
    for lv in (30, 31, 32, 33):
        for s in range(n_seeds):
            r = run_unit(lv, s, n_frames=n_frames)
            dets.append(r["det_rate"])
            fps.append(r["fp_rate"])
            lds.append(r["ld_med"])
            sfss.append(r["sfs_err"])
    det_m = float(np.mean(dets))
    fp_m = float(np.mean(fps))
    ld_m = float(np.nanmean(lds))
    sfs_m = float(np.nanmean(sfss))
    ok = (abs(det_m - 1.0000) <= TOL and abs(fp_m - 0.0000) <= TOL
          and abs(ld_m - 0.4821) <= TOL and abs(sfs_m - 4.6258) <= TOL)
    return (1 if ok else 0), det_m, fp_m, ld_m, sfs_m


def guard_b2(n_seeds=10, n_frames=240):
    """R_B4_GUARD_B2：B2 层 1 复现（import attention_emergence_test.run_attention）：
    CONSIST >= 0.80、FRACPOS >= 0.60、MEANBEN in [0.03,0.06]（B2 实测 1.0000/1.0000/
    0.044675）。"""
    units = [run_attention(s, n_frames=n_frames, jitter=JITTER)
             for s in range(n_seeds)]
    c_m, _ = mean_sd([u["consist_frac"] for u in units])
    f_m, _ = mean_sd([u["frac_pos"] for u in units])
    b_m, _ = mean_sd([u["mean_benefit"] for u in units])
    ok = (c_m >= B2_CONSIST_MIN and f_m >= B2_FRACPOS_MIN
          and B2_MEANBEN_LO <= b_m <= B2_MEANBEN_HI)
    return (1 if ok else 0), c_m, f_m, b_m


def guard_b3(n_seeds=10, n_frames=240):
    """R_B4_GUARD_B3：B3 完整链路复现（import attention_cost_test.run_cost）：
    NEG 池化比例 in [0.15,0.45]、TRIG >= 0.70、REDIR >= 0.70（B3 实测 0.2783/1.0000/
    0.9219）。"""
    units = [run_cost(s, n_frames=n_frames, jitter=JITTER) for s in range(n_seeds)]
    n_win = max(1, n_frames // B2.WINDOW)
    neg_n = sum(u["neg_n"] for u in units)
    neg_tot = sum(len(list(range(1, n_win))) for _ in units)
    neg_frac = neg_n / neg_tot if neg_tot else 0.0
    trig_n = 0
    trig_ok = 0
    for u in units:
        for w in range(1, n_win):
            if u["sec"][w] == 1 and w + 1 < n_win:
                trig_n += 1
                if u["focus"][w + 1] != u["focus"][w]:
                    trig_ok += 1
    trig = trig_ok / trig_n if trig_n > 0 else float("nan")
    redir_n = sum(u["redir_n"] for u in units)
    redir_ok = sum(u["redir_ok_n"] for u in units)
    redir = redir_ok / redir_n if redir_n > 0 else float("nan")
    ok = (B3_NEG_LO <= neg_frac <= B3_NEG_HI and trig >= B3_TRIG_MIN
          and redir >= B3_REDIR_MIN)
    return (1 if ok else 0), neg_frac, trig, redir


# ---------------- 诊断模式（R_B4_DIAG_* 行；诊断轮用，主运行摘要块格式不变） ----------------
def diag(seed=0, style=0, n_frames=N_FRAMES):
    u = run_b4(seed, style, n_frames=n_frames, jitter=JITTER)
    n_w = max(1, n_frames // WINDOW)
    print("R_B4_DIAG_SEED=%d_STYLE=%d" % (seed, style))
    for w in range(n_w):
        lv = ("G" if w % 3 == 0 else ("O" if w % 3 == 1 else "T"))
        print("R_B4_DIAG_W%d=%s,SH=%d,F1=%d,F2=%d,F3=%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f" % (
            w, lv,
            u["style_hat"][w] if u["style_hat"][w] is not None else -1,
            u["F1_ok"][w], u["F2_ok"][w], u["F3_ok"][w],
            u["theta_g"][w], u["theta_o"][w], u["theta_t"][w],
            u["Rg"][w], u["Ro"][w], u["Rt"][w]))
    print("R_B4_DIAG_STYLE_ACC=%.4f" % u["style_acc"])
    print("R_B4_DIAG_D1=%d/%d" % (u["drill1_ok"], u["drill1_n"]))
    print("R_B4_DIAG_D2=%d/%d" % (u["drill2_ok"], u["drill2_n"]))
    print("R_B4_DIAG_D3=%d/%d" % (u["drill3_ok"], u["drill3_n"]))
    print("R_B4_DIAG_DRILL_ACC=%.4f" % (u["drill_acc"] if u["drill_acc"] is not None else -1.0))
    print("R_B4_DIAG_BACKWARD=%d" % u["backward_n"])
    print("R_B4_DIAG_ABS_DISK=%.4f" % (u["absorb_disk"] if u["absorb_disk"] is not None else -1.0))
    print("R_B4_DIAG_ABS_STRIPES=%.4f" % (u["absorb_stripes"] if u["absorb_stripes"] is not None else -1.0))
    print("R_B4_DIAG_ABS_DOT=%.4f" % (u["absorb_dot"] if u["absorb_dot"] is not None else -1.0))
    print("R_B4_DIAG_ABS_CORRECT=%.4f" % (u["absorb_correct"] if u["absorb_correct"] is not None else -1.0))
    print("R_B4_DIAG_NOISE_EV=%.6f" % u["noise_block_ev"])
    for d in u.get("tex_detail", []):
        print("R_B4_DIAG_TEX_W%d_THT=%.6f_DOTMED=%s_DOTOK=%d_F2OFF=%s_STRIPEMED=%s_STRIPEOK=%d" % (
            d["w"], d["tht"],
            ("%.6f" % d["dot_med"]) if d["dot_med"] is not None else "nan",
            d["dot_ok"], ",".join("%.1f" % v for v in d["f2_off"]),
            ("%.6f" % d["stripe_med"]) if d["stripe_med"] is not None else "nan",
            d["stripe_ok"]))
    print("R_B4_DIAG_DET=%.4f" % (u["det_err_mean"] if u["det_err_mean"] is not None else -1.0))
    return 0


def scan(seed=0, style=0, n_frames=N_FRAMES, scan_seeds=(0, 1, 2)):
    """诊断轮载体旋钮边界扫描（R_B4_SCAN_* 行）：K 与 σ_o 组合下"缺陷检出/吸收正确"
    的工作点（docs/B4 §1.8：修复前诊断；判据/机制语义 §一 不动）。每组合池化
    scan_seeds 种子报告。"""
    import itertools
    for kd, so in itertools.product((2.0, 3.0, 4.0, 5.0), (1.5, 2.5, 3.5)):
        d1_ok = d1_n = d2_ok = d2_n = d3_ok = d3_n = 0
        abs_vals = []
        sa_vals = []
        for s in scan_seeds:
            u = run_b4(s, style, n_frames=n_frames, jitter=JITTER,
                       k_defect=kd, sigma_o=so)
            d1_ok += u["drill1_ok"]; d1_n += u["drill1_n"]
            d2_ok += u["drill2_ok"]; d2_n += u["drill2_n"]
            d3_ok += u["drill3_ok"]; d3_n += u["drill3_n"]
            if u["absorb_correct"] is not None:
                abs_vals.append(u["absorb_correct"])
            sa_vals.append(u["style_acc"])
        dacc = (d1_ok + d2_ok + d3_ok) / max(1, d1_n + d2_n + d3_n)
        abs_m = float(np.mean(abs_vals)) if abs_vals else float("nan")
        sa_m = float(np.mean(sa_vals)) if sa_vals else float("nan")
        print("R_B4_SCAN_K=%.1f_SO=%.1f_DRILL=%.4f_D1=%d/%d_D2=%d/%d_D3=%d/%d_ABS=%.4f_STYLEACC=%.4f" % (
            kd, so, dacc, d1_ok, d1_n, d2_ok, d2_n, d3_ok, d3_n, abs_m, sa_m))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--n-styles", type=int, default=3)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="gd")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="诊断模式：单种子单风格逐窗口 R_B4_DIAG_* 行")
    ap.add_argument("--scan", action="store_true",
                    help="诊断模式：载体旋钮边界扫描 R_B4_SCAN_* 行")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", type=int, default=0)
    ap.add_argument("--scan-seeds", type=int, default=3,
                    help="扫描每组合池化的种子数（默认 3）")
    args = ap.parse_args()

    if args.diag:
        return diag(seed=args.seed, style=args.style, n_frames=args.frames)
    if args.scan:
        return scan(seed=args.seed, style=args.style, n_frames=args.frames,
                    scan_seeds=tuple(range(args.scan_seeds)))

    os.makedirs(args.out_dir, exist_ok=True)
    styles = list(range(min(args.n_styles, 3)))
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    cfg = {"n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "n_styles": len(styles), "frames": args.frames,
           "jitter": args.jitter, "tag": args.tag,
           "lvcode_b4": LVCODE_B4,
           "scene": {"disk_orbit": list(DISK_ORBIT_C) + [DISK_ORBIT_R, DISK_FREQ],
                     "stripe_off": list(STRIPE_OFF), "stripe_half": STRIPE_HALF,
                     "stripe_gray": [STRIPE_DARK, STRIPE_BRIGHT],
                     "dot_center_off": list(DOT_CENTER_OFF), "dot_orbit_r": DOT_ORBIT_R,
                     "dot_freq": DOT_FREQ, "dot_r": DOT_R, "dot_gray": DOT_GRAY,
                     "dot_seg_start": DOT_SEG_START,
                     "noise_block": list(NOISE_BLOCK), "noise_sigma": NOISE_SIGMA,
                     "occ_lum_thresh": OCC_LUM_THRESH},
           "granularity": {"gist_block": GIST_BLOCK, "k_defect": K_DEFECT,
                           "sigma_o": SIGMA_O, "alpha_o": ALPHA_O,
                           "alpha_t": ALPHA_T, "k_drill": K_DRILL,
                           "a_min": [A_MIN_G, A_MIN_O, A_MIN_T],
                           "dilate": DILATE, "interior_r": INTERIOR_R},
           "criteria": {"style_acc_min": STYLE_ACC_MIN,
                        "drill_acc_min": DRILL_ACC_MIN,
                        "drill_nobs_min": DRILL_NOBS_MIN,
                        "absorb_min": ABSORB_MIN},
           "guards": {"tol": TOL, "b2_ranges": [B2_CONSIST_MIN, B2_FRACPOS_MIN,
                                                B2_MEANBEN_LO, B2_MEANBEN_HI],
                      "b3_ranges": [B3_NEG_LO, B3_NEG_HI, B3_TRIG_MIN,
                                    B3_REDIR_MIN]}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_gd_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for s in seeds:
            for st in styles:
                key = "%d_%d" % (LVCODE_B4[st], s)
                if key in per_unit:
                    continue
                per_unit[key] = run_b4(s, st, n_frames=args.frames, jitter=args.jitter)
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"config": cfg, "per_unit": per_unit},
                              f, ensure_ascii=False, indent=1)
                print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%d_%d" % (LVCODE_B4[st], s)]
             for s in seeds for st in styles]

    # ---- 判据（docs/B4 §1.4 冻结；跨运行池化） ----
    style_accs = [u["style_acc"] for u in units]
    sa_m, sa_sd = mean_sd(style_accs)
    gist_first_all = all(u["gist_first_frac"] == 1.0 for u in units)
    c1 = (sa_m >= STYLE_ACC_MIN) and gist_first_all

    drill_n = sum(u["drill_n"] for u in units)
    drill_ok = sum(u["drill_ok"] for u in units)
    drill_pooled = drill_ok / drill_n if drill_n > 0 else float("nan")
    c2 = (drill_n >= DRILL_NOBS_MIN) and (drill_pooled >= DRILL_ACC_MIN)

    backward_total = sum(u["backward_n"] for u in units)
    abs_correct_vals = [u["absorb_correct"] for u in units
                        if u["absorb_correct"] is not None]
    abs_m = float(np.mean(abs_correct_vals)) if abs_correct_vals else float("nan")
    c3 = (backward_total == 0) and (abs_m >= ABSORB_MIN)

    # ---- 守卫（docs/B4 §1.6 冻结；固定 10 种子/40 单位） ----
    g_260, g_det, g_fp, g_ld, g_sfs = guard_260()
    g_b2, b2_c, b2_fp, b2_mb = guard_b2()
    g_b3, b3_neg, b3_trig, b3_redir = guard_b3()
    g_so, so_r_m, so_r_sd, so_r_ci, so_diff_ci, so_auc, so_nobs = guard_so()
    g_comp, c_mae, c_mae_sd, c_sc2, c_sc2_sd, c_comp, c_comp_sd, c_churn = \
        guard_compose()
    guards_ok = (g_260 == 1 and g_b2 == 1 and g_b3 == 1 and g_so == 1
                 and g_comp == 1)

    # ---- 判定（docs/B4 §1.5 冻结） ----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1:
        verdict = "GIST_FAIL"
    elif c2 and c3:
        verdict = "GRANULAR_PASS"
    else:
        verdict = "PARTIAL"

    # ---- 内部确定性复现（docs/B4 §1.6-6；第二遍强制重算，不读 checkpoint） ----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for key in per_unit:
            for kk in REPRO_KEYS:
                if per_unit[key][kk] != per_unit2[key][kk]:
                    repro = 0

    out = {
        "artifact": "granularity_drill_test",
        "doc_ref": "lineB-motion-coupling/docs/B4",
        "config": cfg,
        "per_unit": per_unit,
        "criteria": {"c1_gist_prior": bool(c1), "c2_defect_drill": bool(c2),
                     "c3_granular_order": bool(c3),
                     "style_acc_mean": sa_m, "style_acc_sd": sa_sd,
                     "gist_first_all": bool(gist_first_all),
                     "drill_pooled": drill_pooled, "drill_n": drill_n,
                     "drill_ok": drill_ok,
                     "backward_n": backward_total,
                     "absorb_correct_mean": abs_m,
                     "absorb_disk": round(float(np.nanmean(
                         [u["absorb_disk"] for u in units
                          if u["absorb_disk"] is not None])), 6),
                     "absorb_stripes": round(float(np.nanmean(
                         [u["absorb_stripes"] for u in units
                          if u["absorb_stripes"] is not None])), 6),
                     "absorb_dot": round(float(np.nanmean(
                         [u["absorb_dot"] for u in units
                          if u["absorb_dot"] is not None])), 6),
                     "noise_block_ev_mean": round(float(np.nanmean(
                         [u["noise_block_ev"] for u in units])), 6),
                     "tex_outside_focus_total": int(sum(
                         u["tex_outside_focus"] for u in units))},
        "guards": {"cell260": g_260, "cell260_det": g_det, "cell260_fp": g_fp,
                   "cell260_ld": g_ld, "cell260_sfs": g_sfs,
                   "b2": g_b2, "b2_consist": b2_c, "b2_fracpos": b2_fp,
                   "b2_meanben": b2_mb,
                   "b3": g_b3, "b3_neg": b3_neg, "b3_trig": b3_trig,
                   "b3_redir": b3_redir,
                   "so": g_so, "so_r_mean": so_r_m, "so_r_sd": so_r_sd,
                   "so_r_ci95": list(so_r_ci), "so_diff_ci95": list(so_diff_ci),
                   "so_auc": so_auc, "so_nobs": so_nobs,
                   "compose": g_comp, "compose_mae": c_mae,
                   "compose_mae_sd": c_mae_sd, "compose_sc2": c_sc2,
                   "compose_sc2_sd": c_sc2_sd, "compose_comp": c_comp,
                   "compose_comp_sd": c_comp_sd, "compose_churn": c_churn,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "gd_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定） ----
    print("R_B4_TAG=%s" % args.tag)
    print("R_B4_SEEDS=%d" % len(seeds))
    print("R_B4_STYLES=%d" % len(styles))
    print("R_B4_FRAMES=%d" % args.frames)
    for s in seeds:
        for st in styles:
            u = per_unit["%d_%d" % (LVCODE_B4[st], s)]
            print("R_B4_S%d_ST%d_STYLEACC=%.4f" % (s, st, u["style_acc"]))
            print("R_B4_S%d_ST%d_D1=%d/%d" % (s, st, u["drill1_ok"], u["drill1_n"]))
            print("R_B4_S%d_ST%d_D2=%d/%d" % (s, st, u["drill2_ok"], u["drill2_n"]))
            print("R_B4_S%d_ST%d_D3=%d/%d" % (s, st, u["drill3_ok"], u["drill3_n"]))
            print("R_B4_S%d_ST%d_DRILLACC=%.4f" % (s, st, u["drill_acc"] if u["drill_acc"] is not None else -1.0))
            print("R_B4_S%d_ST%d_BACKWARD=%d" % (s, st, u["backward_n"]))
            print("R_B4_S%d_ST%d_ABS=%.4f" % (s, st, u["absorb_correct"] if u["absorb_correct"] is not None else -1.0))
            print("R_B4_S%d_ST%d_ABSDISK=%.4f" % (s, st, u["absorb_disk"] if u["absorb_disk"] is not None else -1.0))
            print("R_B4_S%d_ST%d_ABSSTRIPES=%.4f" % (s, st, u["absorb_stripes"] if u["absorb_stripes"] is not None else -1.0))
            print("R_B4_S%d_ST%d_ABSDOT=%.4f" % (s, st, u["absorb_dot"] if u["absorb_dot"] is not None else -1.0))
            print("R_B4_S%d_ST%d_NOISEEV=%.6f" % (s, st, u["noise_block_ev"]))
            print("R_B4_S%d_ST%d_DET=%.4f" % (s, st, u["det_err_mean"] if u["det_err_mean"] is not None else -1.0))
    print("R_B4_STYLE_ACC_MEAN=%.4f" % sa_m)
    print("R_B4_STYLE_ACC_SD=%.4f" % sa_sd)
    print("R_B4_GIST_FIRST_ALL=%d" % (1 if gist_first_all else 0))
    print("R_B4_DRILL_POOLED=%.4f" % drill_pooled)
    print("R_B4_DRILL_N=%d" % drill_n)
    print("R_B4_DRILL_OK=%d" % drill_ok)
    print("R_B4_BACKWARD_N=%d" % backward_total)
    print("R_B4_ABS_CORRECT_MEAN=%.4f" % abs_m)
    print("R_B4_ABS_DISK_MEAN=%.4f" % float(np.nanmean(
        [u["absorb_disk"] for u in units if u["absorb_disk"] is not None])))
    print("R_B4_ABS_STRIPES_MEAN=%.4f" % float(np.nanmean(
        [u["absorb_stripes"] for u in units if u["absorb_stripes"] is not None])))
    print("R_B4_ABS_DOT_MEAN=%.4f" % float(np.nanmean(
        [u["absorb_dot"] for u in units if u["absorb_dot"] is not None])))
    print("R_B4_NOISE_EV_MEAN=%.6f" % float(np.nanmean(
        [u["noise_block_ev"] for u in units])))
    print("R_B4_TEX_OUTSIDE_FOCUS=%d" % int(sum(u["tex_outside_focus"]
                                                for u in units)))
    print("R_B4_C1=%s" % ("PASS" if c1 else "FAIL"))
    print("R_B4_C2=%s" % ("PASS" if c2 else "FAIL"))
    print("R_B4_C3=%s" % ("PASS" if c3 else "FAIL"))
    print("R_B4_GUARD_260=%d" % g_260)
    print("R_B4_GUARD_260_DET=%.4f" % g_det)
    print("R_B4_GUARD_260_FP=%.4f" % g_fp)
    print("R_B4_GUARD_260_LD=%.4f" % g_ld)
    print("R_B4_GUARD_260_SFS=%.4f" % g_sfs)
    print("R_B4_GUARD_B2=%d" % g_b2)
    print("R_B4_GUARD_B2_CONSIST=%.4f" % b2_c)
    print("R_B4_GUARD_B2_FRACPOS=%.4f" % b2_fp)
    print("R_B4_GUARD_B2_MEANBEN=%.6f" % b2_mb)
    print("R_B4_GUARD_B3=%d" % g_b3)
    print("R_B4_GUARD_B3_NEG=%.4f" % b3_neg)
    print("R_B4_GUARD_B3_TRIG=%.4f" % b3_trig)
    print("R_B4_GUARD_B3_REDIR=%.4f" % b3_redir)
    print("R_B4_GUARD_SO=%d" % g_so)
    print("R_B4_GUARD_SO_R=%.6f" % so_r_m)
    print("R_B4_GUARD_SO_R_SD=%.6f" % so_r_sd)
    print("R_B4_GUARD_SO_LO=%.6f" % so_r_ci[0])
    print("R_B4_GUARD_SO_HI=%.6f" % so_r_ci[1])
    print("R_B4_GUARD_SO_DIFF_LO=%.6f" % so_diff_ci[0])
    print("R_B4_GUARD_SO_DIFF_HI=%.6f" % so_diff_ci[1])
    print("R_B4_GUARD_SO_AUC=%.6f" % so_auc)
    print("R_B4_GUARD_SO_NOBS=%.1f" % so_nobs)
    print("R_B4_GUARD_COMPOSE=%d" % g_comp)
    print("R_B4_GUARD_COMPOSE_MAE=%.6f" % c_mae)
    print("R_B4_GUARD_COMPOSE_SC2=%.4f" % c_sc2)
    print("R_B4_GUARD_COMPOSE_COMP=%.4f" % c_comp)
    print("R_B4_GUARD_COMPOSE_CHURN=%.4f" % c_churn)
    print("R_B4_REPRO=%d" % repro)
    print("R_B4_VERDICT=%s" % verdict)
    print("R_B4_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
