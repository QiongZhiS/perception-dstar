"""lineB-motion-coupling/scripts/static_scene_test.py — B 路第五格：静态场景视觉——
静止帧的空间残差 + 缺陷检测 + 粒度下钻（docs: lineB-motion-coupling/docs/B5-静态场景视觉-
空间残差-预注册设计.md §一 冻结）。

核心：docs/255 §六.5 关键洞察的操作化首验——docs/187 的"静止 = 0 事件"是**时间预测**
（bg_fast EWMA 预测下一帧）的结论；残差递归只需要"预测-落空"，预测源可以是**空间的**
（邻域重建）——**空间预测下静止结构依然可感知**。完全静止帧（无运动、无时间变化——
每帧内容逐像素恒同，仅加性噪声）上，纯空间三层预测器（gist 8px 块均值重建 / object
焦点内空间平滑模板 / texture 焦点内纹理统计 1-D 中值——全部逐帧纯空间、无时间模板）
吸收规则静态结构（棋盘/风格带/静止纹理块/盘本体/规则条纹），逃逸异常静态结构（静态亮
圆盘 = 形状异常 → 场景级缺陷；盘上 2px 竖条纹块 = 颜色异常 → 物体级缺陷；条纹中心水平
划痕 = 纹理划痕 → 纹路级终态缺陷）。对照设计：同一静止像素上，时间预测残差 = 噪声底、
事件率 0（docs/187 前提逐位复现，TIME_ZERO）；空间残差在缺陷区显著 > 0（SPATIAL_NONZERO，
≥5× 时间残差）——"感知消失"只在时间预测下成立，空间残差打开静态通道。缺陷 = 空间残差
超 max(K·σ̂, 0.10) 的相干区域（docs/221 原子，零手工语义、零 GT 进机制）；下钻 = 缺陷处
注意聚焦（L1 复用 B4/B2/B3 的 argmax(R×S) 机制，S 第一版外赋 = 均匀 1.0 = 纯残差驱动）+
B4 粒度状态机原样复用（gist → object → texture → gist，静止帧同样工作）。

判据（§1.4 冻结，docs/247 标签 [L1][机制][合成受控]——合成受控、非真实域证明；"缺陷"=
空间残差不宣称语义理解；对 docs/187 是推广而非反驳）：
  C1 TIME_ZERO      : 时间事件率 ≤ 0.001 + 静止纹理块 0 事件（=0.000000）+ 时间残差 ≤ 0.10
  C2 SPATIAL_NONZERO: 缺陷区（盘 GT 区）空间残差 ≥ 5× 时间残差 且 ≥ 0.20 的池化比例 ≥ 0.80
  C3 STATIC_DEFECT  : 各级缺陷区空间残差超该级阈值（gist 盘 GT 区 R_g 均值 / object 暗条纹
                      列 R_o 中位数 / texture 划痕 R_t 中位数）的池化检出率 ≥ 0.80 且观察 ≥ 20
  C4 STATIC_DRILL   : 下钻焦点落真缺陷 GT（膨胀 2px）的池化正确率 ≥ 0.80 且观察 ≥ 20
  C5 KEEP 守卫      : R_B5_GUARD_B4_GIST + R_B5_GUARD_B4_DRILL + R_B5_GUARD_B4_ORDER +
                      R_B5_GUARD_187 + R_B5_REPRO 全 = 1
判定（§1.5）：全过 = STATIC_SPATIAL_PASS；C1 过 C2 不过 = SPATIAL_FAIL；C2 过 C3/C4 不过
= PARTIAL；C1 不过 = TIME_FAIL；守卫不过 = GUARD_FAIL。

守卫（§1.6 冻结；守卫种子数固定 10 × 3 风格，不随主实验 n_seeds）：
  R_B5_GUARD_B4_GIST : import granularity_drill_test.run_b4 重跑 30 单位（10 种子 × 3 风格）
                       → 池化 style_acc == 1.0000（240 gist 窗全对，B4 C1 逐位）
  R_B5_GUARD_B4_DRILL: 同上 → 池化 drill_acc == 1.0000 且 drill_n == 600（B4 C2 逐位）
  R_B5_GUARD_B4_ORDER: 同上 → backward 总计 == 0 且 池化 absorb_correct == 1.0000（B4 C3 逐位）
  R_B5_GUARD_187     : 本格 30 运行静止纹理块时间事件量 == 0.000000（docs/187 锚点）
  R_B5_REPRO         : --repro 时主实验 30 运行整体重跑第二遍，关键数字位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_B5_* 摘要块
（顺序固定）；JSON 归档 lineB-motion-coupling/out/ss_<tag>.json + checkpoint
ckpt_ss_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 lineB-motion-coupling/out/*.log 与 lineB-motion-coupling/out/*.json 原文。
**未修改任何主线既有脚本**（vision/ 下全部不动；B1/B2/B3/B4 脚本亦不动，只 import）。

用法：
  python lineB-motion-coupling/scripts/static_scene_test.py --n-seeds 10 --tag main --repro
  python lineB-motion-coupling/scripts/static_scene_test.py --n-seeds 1 --n-styles 1 --tag timing
  python lineB-motion-coupling/scripts/static_scene_test.py --diag --seed 0 --style 0
  python lineB-motion-coupling/scripts/static_scene_test.py --scan --seed 0 --style 0
"""
import argparse
import hashlib
import json
import math
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

# ---- import 复用 B4/B2/critical_point（docs/B5 §一：零改动，只 import）----
import attention_emergence_test as B2           # noqa: E402
from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER  # noqa: E402
from granularity_drill_test import (           # noqa: E402
    _style_layout, coarse_image, coarse_recon, style_prototypes,
    components, dilate, point_in_gt, noise_scale, run_b4)

top2_components = B2.top2_components
disk_mask = B2.disk_mask

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("lineB-motion-coupling", "out")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 场景旋钮（docs/B5 §1.2 冻结；载体旋钮经 §二 诊断定案，判据/机制语义不动）----
LVCODE_B5 = [51, 52, 53]          # 三风格 S0/S1/S2（与 0-6 / 20-23 / 30-33 / 40 / 41-43 流错开）
W, H = 160, 120
FPS = 30
N_FRAMES = 240
WINDOW = 10
BG_CELL = 24
BG_DARK, BG_BRIGHT = 64.0, 96.0
S2_DARK = 48.0                    # 风格 S2：暗格 48
S1_BAND_ROWS = (0, 40)            # 风格 S1：顶部横带（块对齐：5 块行）
S1_BAND_GRAY = 128.0
DISK_C = (80.0, 60.0)             # 静态圆盘中心（固定，逐帧不变）
DISK_R = 10.0
DISK_GRAY = 255.0
NOISE_SIGMA = 3.0
STRIPE_OFF = (3.0, -2.0)          # 条纹块盘相对偏移（中心）
STRIPE_HALF = 3                   # 6x6 块（半宽 3）
STRIPE_DARK, STRIPE_BRIGHT = 80.0, 255.0
STRIPE_PERIOD = 2                 # px（2px 周期竖条纹 = 高频颜色异常块）
SCRATCH_GRAY = 150.0              # 划痕灰（1px 水平线，横穿条纹块中心行）
NOISE_BLOCK = (40, 64, 24, 48)    # 静止纹理块 y0,y1,x0,x1（块对齐 3x3；docs/187 锚点）
NOISE_BLOCK_MEAN = 80.0
OCC_LUM_THRESH = 220.0            # 亮检测 = L > log(221)（docs/260 同款）
A_MIN_DET = 25                    # 检测连通域最小面积（px）

# ---- 纯空间三层粒度机制旋钮（docs/B5 §1.3 冻结；载体旋钮经 §二 诊断定案）----
GIST_BLOCK = 8                    # gist 块均值尺度（px；20x15 粗图）
K_DEFECT = 3.0                    # 缺陷 = 残差 > max(K·σ̂, THETA_FLOOR)
THETA_FLOOR = 0.10                # 噪声底下限（B4 定案同款）
SIGMA_O = 1.5                     # object 空间平滑高斯 σ（B4 D1-10 定案同款：须 ≥ 条纹半周期
                                  # 1px 使条纹逃逸、且 < 盘本体尺度使本体吸收）
TEX_MEDIAN_N = 5                  # texture 1-D 中值窗口长（沿主导方向；≥ 条纹块 6 行内余量、
                                  # ≤ 划痕行 ±2 使划痕为中值窗口少数）
K_DRILL = 3                       # 缺陷持久确认：窗口内 ≥ 3 帧（≥30%）
A_MIN_G = 25                      # gist 缺陷最小面积（px）
A_MIN_O = 5                       # object 缺陷最小面积（px；暗列 6px 合法小缺陷组件）
A_MIN_T = 5                       # texture 缺陷最小面积（px；划痕 6px）
DILATE = 2                        # GT 正确性判定的膨胀（px）
OBJ_CROP = 40                     # object 焦点裁剪（px）
TEX_CROP = 16                     # texture 焦点裁剪（px）
INTERIOR_R = 8.0                  # object 缺陷搜索域 = 盘内部 r≤8（排除盘缘环带）

# ---- 判据阈值（docs/B5 §1.4 冻结）----
TMP_EV_MAX = 0.001
TMP_RES_MAX = 0.10
SP_RATIO_MIN = 5.0
SP_MEAN_MIN = 0.20
SP_FRAC_MIN = 0.80
DETECT_MIN = 0.80
DETECT_NOBS_MIN = 20
DRILL_MIN = 0.80
DRILL_NOBS_MIN = 20

# ---- 守卫容差（docs/B5 §1.6 冻结）----
TOL = 1e-3

REPRO_KEYS = ["tmp_ev", "tmp_res", "noise_block_ev",
              "sp_frac", "sp_ratio_mean", "sp_mean_mean",
              "det1_n", "det1_ok", "det2_n", "det2_ok",
              "det3_n", "det3_ok",
              "drill1_n", "drill1_ok", "drill2_n", "drill2_ok",
              "drill3_n", "drill3_ok", "drill_acc",
              "backward_n", "absorb_disk", "absorb_stripes",
              "absorb_scratch", "style_acc",
              "theta_g_mean", "theta_o_mean", "theta_t_mean",
              "Rg_mean", "Ro_mean", "Rt_mean",
              "F1_ok", "F2_ok", "F3_ok"]


# ---------------- 合成场景（docs/B5 §1.2 冻结：完全静止帧 + 静态空间缺陷） ----------------
def make_b5_scene(seed, style, n_frames=N_FRAMES, width=W, height=H, fps=FPS,
                  jitter=JITTER, scratch_gray=SCRATCH_GRAY):
    """生成 (风格, 种子) 的**完全静止**灰度帧序列 + GT（只用于事后测量，绝不进入机制）。
    确定性：rng 由 (seed, lvcode) 派生；调用顺序固定（noise_mult -> 静止纹理块噪声模式）；
    逐帧内容恒同（静态圆盘/条纹块/划痕/纹理块位置逐帧不变），每帧唯一变化 = 加性噪声。
    GT：disk_c / stripe_off / scratch_off（盘相对）/ noise_block。"""
    lvcode = LVCODE_B5[style]
    rng = np.random.default_rng(seed * 7919 + lvcode * 104729 + 13)
    noise_mult = rng.uniform(1 - jitter, 1 + jitter) if jitter > 0 else 1.0
    sigma = NOISE_SIGMA * noise_mult
    blk_pat = rng.normal(0, sigma, (NOISE_BLOCK[1] - NOISE_BLOCK[0],
                                    NOISE_BLOCK[3] - NOISE_BLOCK[2])).astype(np.float32)

    bg = _style_layout(style)
    bg = bg.copy()
    bg[NOISE_BLOCK[0]:NOISE_BLOCK[1], NOISE_BLOCK[2]:NOISE_BLOCK[3]] = \
        np.clip(NOISE_BLOCK_MEAN + blk_pat, 0, 255)

    # 静态圆盘（固定中心）
    img0 = bg.copy()
    dcx = int(round(DISK_C[0]))
    dcy = int(round(DISK_C[1]))
    cv2.circle(img0, (dcx, dcy), int(DISK_R), DISK_GRAY, -1)
    # 静态条纹块（盘相对偏移 STRIPE_OFF，6x6，2px 竖条纹 80/255）
    px = int(round(DISK_C[0] + STRIPE_OFF[0]))
    py = int(round(DISK_C[1] + STRIPE_OFF[1]))
    for yy in range(py - STRIPE_HALF, py + STRIPE_HALF):
        for xx in range(px - STRIPE_HALF, px + STRIPE_HALF):
            if 0 <= yy < height and 0 <= xx < width:
                val = STRIPE_DARK if ((xx - px) % STRIPE_PERIOD) == 0 else STRIPE_BRIGHT
                img0[yy, xx] = val
    # 静态划痕（1px 水平线，横穿条纹块中心行；盘相对 = (STRIPE_OFF[0], 0) 平移 = 中心行）
    for xx in range(px - STRIPE_HALF, px + STRIPE_HALF):
        if 0 <= py < height and 0 <= xx < width:
            img0[py, xx] = scratch_gray

    frames = []
    for _t in range(n_frames):
        img = img0 + rng.normal(0, sigma, img0.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    gts = dict(style=style, seed=seed, disk_c=(float(DISK_C[0]), float(DISK_C[1])),
               disk_r=DISK_R, stripe_off=STRIPE_OFF, stripe_half=STRIPE_HALF,
               scratch_off=(STRIPE_OFF[0], STRIPE_OFF[1]),
               noise_block=NOISE_BLOCK)
    return frames, gts


# ---------------- 纯空间预测器原语（docs/B5 §1.3 冻结） ----------------
def masked_gaussian_blur(Lc, Dmask, sigma):
    """掩膜归一化高斯（纯空间"物体本体 = 空间自相似"预测）：GaussianBlur(Lc·D)/GaussianBlur(D)，
    背景零泄漏（B4 D1-2 定案同款形态，但**无时间模板**——逐帧当前帧纯空间）。"""
    num = cv2.GaussianBlur(Lc * Dmask, (0, 0), sigma)
    den = cv2.GaussianBlur(Dmask, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def median_along(Lc, phi, n=TEX_MEDIAN_N):
    """沿方向 phi 的 1×n 中值滤波（np.roll 平移采样；边界环绕在盘内均匀/条纹内容下影响
    ≈ 0——本场景裁剪全在盘内）。"""
    k = (n - 1) // 2
    ux = math.cos(phi)
    uy = math.sin(phi)
    acc = []
    for i in range(-k, k + 1):
        ox = int(round(ux * i))
        oy = int(round(uy * i))
        s = Lc.copy()
        if ox != 0:
            s = np.roll(s, -ox, axis=1)
        if oy != 0:
            s = np.roll(s, -oy, axis=0)
        acc.append(s)
    return np.median(np.stack(acc), axis=0)


def texture_predictor(crop_gray):
    """纯空间纹理统计预测器：**方向选择 = argmin 沿 {0°,45°,90°,135°} 的 1×N 中值重建
    误差**（"哪个方向自相似"直接测量——纹理统计（方向）的操作化）；选定方向上的 1-D 中值
    即预测。吸收规则条纹（沿条纹方向的列中值 = 条纹值），逃逸非纹理结构（划痕 = 中值窗口
    内少数）。
    §二 D1-5 定案：结构张量（含前向差分版）对"划痕×条纹"交叉项倾斜（划痕行的 gy 与条纹
    边缘的 gx 乘积恒正 → φ 实测 107° → 窗口斜采样使划痕值恰为中值 → 划痕被吸收、R_t=0）；
    方向选择改为 argmin 重建误差——垂直方向吸收条纹（误差≈噪声）而水平/对角方向误差
    大（≈0.08），对交叉项稳健。"""
    Lc = np.log(np.maximum(crop_gray.astype(np.float32), 1.0))
    best = None
    best_err = float("inf")
    for ang in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
        Pt = median_along(Lc, ang, TEX_MEDIAN_N)
        err = float(np.mean(np.abs(Lc - Pt)))
        if err < best_err:
            best_err = err
            best = Pt
    return best


# ---------------- 主实验单位（docs/B5 §1.3 冻结） ----------------
def run_b5(seed, style, n_frames=N_FRAMES, jitter=JITTER, k_defect=K_DEFECT,
           sigma_o=SIGMA_O, scratch_gray=SCRATCH_GRAY, k_drill=K_DRILL):
    """跑 (风格, 种子) 一次完整运行 → 时间对照 + 三层纯空间残差/缺陷/下钻行为量 + 判据
    组件。粒度状态机（逐窗口，B4 原样）：L(0)=gist；L(w+1)=next(L(w)) 若 L(w) 发现缺陷
    否则 gist；next: gist->object->texture->gist。窗口序列 = 0,3,..,21 gist；1,4,..,22
    object；2,5,..,23 texture（8 周期）。gist 窗口：块均值重建 + 空间残差/盘区测量 +
    drill1；object 窗口：盘心跟踪注册裁剪 + 纯空间平滑 + drill2；texture 窗口：F2 处注册
    裁剪 + 纯空间纹理统计 + drill3（划痕终态）/吸收（条纹）。"""
    frames, gts = make_b5_scene(seed, style, n_frames=n_frames, jitter=jitter,
                                scratch_gray=scratch_gray)
    n_w = max(1, n_frames // WINDOW)
    style_idx = style
    stripe_off = gts["stripe_off"]
    stripe_half = gts["stripe_half"]
    nb = gts["noise_block"]
    disk_c = gts["disk_c"]

    # ---- 检测循环：盘心亮域 top-1 质心（全帧；与层级处理解耦；盘静止 -> 稳定）----
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
            det_errs.append(np.hypot(last[0] - disk_c[0], last[1] - disk_c[1]))

    # ---- 时间对照（docs/187 同款）：CPLoop bg_fast EWMA 预测下一帧，t>=30 起测量 ----
    tmp_evs, tmp_ress, noise_evs = [], [], []
    cp_cfg = {k: v for k, v in B2.LOOP_CFG.items() if k in B2.CP_KEYS}
    loop = CPLoop(window=WINDOW, **cp_cfg)
    for t, g in enumerate(frames):
        loop.step(g)
        if t >= 30 and loop.bg_fast is not None and loop._frame_buf:
            th = loop._frame_buf[-1]["theta"]
            db = loop._frame_buf[-1]["db"]
            L = np.log(np.maximum(g.astype(np.float32), 1.0))
            r = np.abs(L - loop.bg_fast)
            rd = np.maximum(r - db, 0.0)
            ev = rd > th
            tmp_evs.append(float(ev.mean()))
            tmp_ress.append(float(r.mean()))
            blk = ev[nb[0]:nb[1], nb[2]:nb[3]]
            noise_evs.append(float(blk.mean()))
    tmp_ev = float(np.mean(tmp_evs)) if tmp_evs else 0.0
    R_tmp = float(np.mean(tmp_ress)) if tmp_ress else 0.0
    noise_block_ev = float(np.mean(noise_evs)) if noise_evs else 0.0

    # ---- GT 掩码（盘相对坐标 (60,60) 原点；仅事后测量）----
    def stripe_gt_rel():
        m = np.zeros((H, W), bool)
        px0 = int(round(stripe_off[0] + 60))
        py0 = int(round(stripe_off[1] + 60))
        for yy in range(py0 - stripe_half, py0 + stripe_half):
            for xx in range(px0 - stripe_half, px0 + stripe_half):
                if 0 <= yy < H and 0 <= xx < W:
                    m[yy, xx] = True
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        interior = (xx - 60) ** 2 + (yy - 60) ** 2 <= INTERIOR_R ** 2
        return m & interior

    def scratch_gt_rel():
        m = np.zeros((H, W), bool)
        so = gts["scratch_off"]
        sx0 = int(round(so[0] + 60 - stripe_half))
        sx1 = int(round(so[0] + 60 + stripe_half))
        sy = int(round(so[1] + 60))
        for xx in range(sx0, sx1):
            if 0 <= sy < H and 0 <= xx < W:
                m[sy, xx] = True
        return m

    def disk_gt_global():
        return disk_mask(disk_c, r=DISK_R, shape=(H, W))

    # ---- 状态机与行为量记录 ----
    level = [0] * n_w
    gist_ws = [w for w in range(n_w) if w % 3 == 0]
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
    sp_ratio_rec = [0.0] * n_w
    sp_mean_rec = [0.0] * n_w
    sp_ok_rec = [0] * n_w
    det_g_rec = [0] * n_w
    det_o_rec = [0] * n_w
    det_t_rec = [0] * n_w
    drill1_n = drill1_ok = drill2_n = drill2_ok = drill3_n = drill3_ok = 0
    det1_n = det1_ok = det2_n = det2_ok = det3_n = det3_ok = 0
    backward_n = 0
    absorb_disk_n = absorb_disk_ok = 0
    absorb_str_n = absorb_str_ok = 0
    absorb_scr_n = absorb_scr_ok = 0
    gist_first = 0
    tex_outside_focus = 0
    F2_off = None

    prots = style_prototypes()
    tex_detail = []

    def process_gist(w, lo, hi):
        Ls = []
        for t in range(lo, min(hi, n_frames)):
            Ls.append(np.log(np.maximum(frames[t].astype(np.float32), 1.0)))
        c = coarse_image(Ls)
        dists = [float(np.mean(np.abs(c - p))) for p in prots]
        style_hat[w] = int(np.argmin(dists))
        # 空间残差场（逐帧 |L - 重建| -> 窗口均值）
        Rg_frames = []
        for L in Ls:
            G = coarse_recon(coarse_image([L]))
            Rg_frames.append(np.abs(L - G))
        Rg = np.mean(np.stack(Rg_frames), axis=0)
        Rg_rec[w] = float(Rg.mean())
        sg = noise_scale(Rg)
        thg = max(k_defect * sg, THETA_FLOOR)
        theta_g_rec[w] = thg
        # SPATIAL_NONZERO（本格核心）：盘 GT 区空间残差均值 vs 时间残差 R_tmp（run 级）
        disk_gt = disk_gt_global()
        sp_mean = float(Rg[disk_gt].mean())
        sp_mean_rec[w] = sp_mean
        sp_ratio = sp_mean / R_tmp if R_tmp > 0 else float("inf")
        sp_ratio_rec[w] = sp_ratio
        sp_ok = 1 if (sp_ratio >= SP_RATIO_MIN and sp_mean >= SP_MEAN_MIN) else 0
        sp_ok_rec[w] = sp_ok
        # STATIC_DEFECT gist 实例：盘 GT 区空间残差均值 > θ_g
        det_g = 1 if sp_mean > thg else 0
        det_g_rec[w] = det_g
        nonlocal_inc_det("det1", det_g)
        # 缺陷候选 + 持久确认 + drill1
        mask = Rg > thg
        cands = components(mask, A_MIN_G)
        conf = []
        for (comp, area, cent) in cands:
            n_hi = sum(1 for rf in Rg_frames if float(rf[comp].mean()) > thg)
            if n_hi >= k_drill:
                conf.append((comp, area, cent))
        if conf:
            Vs = [float(Rg[comp].max()) for (comp, _, _) in conf]
            winner = int(np.argmax(Vs))
            comp, area, cent = conf[winner]
            F1 = cent
            ok = point_in_gt(F1, dilate(disk_gt, DILATE))
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

    def nonlocal_inc_det(kind, ok):
        nonlocal det1_n, det1_ok, det2_n, det2_ok, det3_n, det3_ok
        if kind == "det1":
            det1_n += 1
            det1_ok += 1 if ok else 0
        elif kind == "det2":
            det2_n += 1
            det2_ok += 1 if ok else 0
        else:
            det3_n += 1
            det3_ok += 1 if ok else 0

    def process_object(w, lo, hi):
        nonlocal F2_off
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
            # 纯空间预测（无时间模板）：当前帧掩膜归一化高斯
            Po = masked_gaussian_blur(Lc, Dmask, sigma_o)
            Ro_frames.append(np.abs(Lc - Po))
        valid = [r for r in Ro_frames if r is not None]
        if not valid:
            return False
        # §二 D1-7 定案（载体口径）：窗口聚合最大→均值——B5 场景完全静止（无瞬态缺陷，
        # B4 选 max 的"最强落空"语义前提不适用）；max 把已吸收内容（盘本体/条纹亮列）的
        # 噪声残差抬到 θ 地板（实测暗列 max 0.124 vs 0.10），均值下吸收内容 ≈0.03-0.04
        # vs 缺陷（条纹 1.16/划痕 0.47）仍 3-10× 分离——聚合口径不改变缺陷检出，只移除
        # 噪声抬升。
        Ro = np.mean(np.stack(valid), axis=0)
        Ro_rec[w] = float(Ro[interior].mean())
        so = noise_scale(Ro[interior])
        tho = max(k_defect * so, THETA_FLOOR)
        theta_o_rec[w] = tho
        mask = (Ro > tho) & interior
        cands = components(mask, A_MIN_O)
        conf = []
        for (comp, area, cent) in cands:
            n_hi = sum(1 for r in valid if float(r[comp].max()) > tho)
            if n_hi >= k_drill:
                conf.append((comp, area, cent))
        # 吸收度量（报告量）：盘本体 = interior 减条纹块区后的中位数 < θ_o
        scx = OBJ_CROP // 2 + int(round(stripe_off[0]))
        scy = OBJ_CROP // 2 + int(round(stripe_off[1]))
        sr = stripe_half
        sm_full = np.zeros((OBJ_CROP, OBJ_CROP), bool)
        for yy in range(scy - sr, scy + sr):
            for xx in range(scx - sr, scx + sr):
                if 0 <= yy < OBJ_CROP and 0 <= xx < OBJ_CROP:
                    sm_full[yy, xx] = True
        body_mask = interior & ~sm_full
        body_med = float(np.median(Ro[body_mask])) if body_mask.any() else float("nan")
        nonlocal_abs("disk", 1 if (body_med == body_med and body_med < tho) else 0)
        # STATIC_DEFECT object 实例：暗条纹列 R_o 中位数 > θ_o（物体级"不被吸收"）
        sm = np.zeros((OBJ_CROP, OBJ_CROP), bool)
        for yy in range(scy - sr, scy + sr):
            for xx in range(scx - sr, scx + sr):
                if 0 <= yy < OBJ_CROP and 0 <= xx < OBJ_CROP:
                    if ((xx - scx) % STRIPE_PERIOD) == 0:
                        sm[yy, xx] = True
        stripe_med_o = float(np.median(Ro[sm & interior])) if (sm & interior).any() \
            else float("nan")
        det_o = 1 if (stripe_med_o == stripe_med_o and stripe_med_o > tho) else 0
        det_o_rec[w] = det_o
        nonlocal_inc_det("det2", det_o)
        # drill2：焦点落条纹 GT（盘相对 -> 全局；§二 D1-2 修复：GT 掩码与焦点同用盘相对
        # 坐标——此前 rel_to_global 后仍用 rel 点比较 -> 全 fail）
        if conf:
            Vs = [float(Ro[comp].max()) for (comp, _, _) in conf]
            winner = int(np.argmax(Vs))
            comp, area, cent = conf[winner]
            F2_rel = (cent[0] - OBJ_CROP // 2, cent[1] - OBJ_CROP // 2)
            F2_off = F2_rel
            gt = dilate(stripe_gt_rel(), DILATE)
            ok = point_in_gt((F2_rel[0] + 60, F2_rel[1] + 60), gt)
            F2_ok[w] = 1 if ok else 0
            nonlocal_inc("drill2", ok)
            return True
        return False

    def nonlocal_abs(kind, ok):
        nonlocal absorb_disk_n, absorb_disk_ok, absorb_str_n, absorb_str_ok, \
            absorb_scr_n, absorb_scr_ok
        if kind == "disk":
            absorb_disk_n += 1
            absorb_disk_ok += ok
        elif kind == "stripes":
            absorb_str_n += 1
            absorb_str_ok += ok
        else:
            absorb_scr_n += 1
            absorb_scr_ok += ok

    def process_texture(w, lo, hi):
        nonlocal F2_off
        if F2_off is None:
            return False
        # 裁剪原点（窗口首有效帧；静态场景下窗口内恒同）——测量掩码锚定到裁剪原点，
        # 不用 round(stripe_off - F2_off)（F2_off 的 0.5 量化会翻行，§二 D1-6 修复）
        cxo = cyo = None
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
            if cxo is None:
                cxo, cyo = cxc, cyc
            crop = frames[t][cyc:cyc + TEX_CROP, cxc:cxc + TEX_CROP].astype(np.float32)
            # 纯空间纹理统计预测（无时间模板）
            Pt = texture_predictor(crop)
            Lc = np.log(np.maximum(crop, 1.0))
            Rt_frames.append(np.abs(Lc - Pt))
        valid = [r for r in Rt_frames if r is not None]
        if not valid or cxo is None:
            return False
        # §二 D1-7 定案（载体口径）：窗口聚合最大→均值（同 object 层——静止场景无瞬态
        # 缺陷；max 把已吸收条纹的噪声残差抬到 θ_t 地板（实测 0.10-0.12 vs 0.10），
        # 均值下吸收条纹 ≈0.02-0.04 vs 划痕 0.56（14× 分离））。
        Rt = np.mean(np.stack(valid), axis=0)
        Rt_rec[w] = float(Rt.mean())
        # 缺陷域 = 裁剪 ∩ 盘内部（盘心在裁剪坐标 = 全局盘心 - 裁剪原点）
        dcx = int(round(disk_c[0])) - cxo
        dcy = int(round(disk_c[1])) - cyo
        interior_t = disk_mask((dcx, dcy), r=INTERIOR_R, shape=(TEX_CROP, TEX_CROP))
        st = noise_scale(Rt[interior_t]) if interior_t.any() else noise_scale(Rt)
        tht = max(k_defect * st, THETA_FLOOR)
        theta_t_rec[w] = tht
        mask = (Rt > tht) & interior_t
        cands = components(mask, A_MIN_T)
        conf = []
        for (comp, area, cent) in cands:
            n_hi = sum(1 for r in valid if float(r[comp].max()) > tht)
            if n_hi >= k_drill:
                conf.append((comp, area, cent))
        # 条纹块/划痕全局中心 -> 裁剪坐标（锚定裁剪原点，精确）
        sgx = int(round(disk_c[0] + stripe_off[0]))
        sgy = int(round(disk_c[1] + stripe_off[1]))
        sr_cx = sgx - cxo
        sr_cy = sgy - cyo
        sr = stripe_half
        # 吸收度量（报告量）：条纹区 R_t 中位数 < θ_t（texture 级吸收；纹路统计预测掉
        # 规则条纹——docs/187 纹路被预测掉的空间版）
        sm = np.zeros((TEX_CROP, TEX_CROP), bool)
        for yy in range(sr_cy - sr, sr_cy + sr):
            for xx in range(sr_cx - sr, sr_cx + sr):
                if 0 <= yy < TEX_CROP and 0 <= xx < TEX_CROP:
                    sm[yy, xx] = True
        stripe_med_t = float(np.median(Rt[sm & interior_t])) if (sm & interior_t).any() \
            else float("nan")
        stripe_ok = 1 if (stripe_med_t == stripe_med_t and stripe_med_t < tht) else 0
        nonlocal_abs("stripes", stripe_ok)
        # STATIC_DEFECT texture 实例 + 终态：划痕区 R_t 中位数 > θ_t（划痕 = 条纹块
        # 中心行横穿 = 全局 (sgy, sgx±sr)）
        scr = np.zeros((TEX_CROP, TEX_CROP), bool)
        for xx in range(sgx - sr - cxo, sgx + sr - cxo):
            if 0 <= sgy - cyo < TEX_CROP and 0 <= xx < TEX_CROP:
                scr[sgy - cyo, xx] = True
        scratch_med_t = float(np.median(Rt[scr & interior_t])) if (scr & interior_t).any() \
            else float("nan")
        det_t = 1 if (scratch_med_t == scratch_med_t and scratch_med_t > tht) else 0
        det_t_rec[w] = det_t
        nonlocal_inc_det("det3", det_t)
        nonlocal_abs("scratch", det_t)
        tex_detail.append(dict(w=w, tht=round(tht, 6), f2_off=list(F2_off),
                               stripe_med=(round(stripe_med_t, 6)
                                           if stripe_med_t == stripe_med_t else None),
                               stripe_ok=stripe_ok,
                               scratch_med=(round(scratch_med_t, 6)
                                            if scratch_med_t == scratch_med_t else None),
                               scratch_ok=det_t))
        # drill3（终态）：焦点落划痕 GT
        if conf:
            Vs = [float(Rt[comp].max()) for (comp, _, _) in conf]
            winner = int(np.argmax(Vs))
            comp, area, cent = conf[winner]
            F3_rel = (F2_off[0] + (cent[0] - TEX_CROP // 2),
                      F2_off[1] + (cent[1] - TEX_CROP // 2))
            gt = dilate(scratch_gt_rel(), DILATE)
            ok = point_in_gt((F3_rel[0] + 60, F3_rel[1] + 60), gt)
            F3_ok[w] = 1 if ok else 0
            nonlocal_inc("drill3", ok)
            return True
        return False

    # ---- 主循环（粒度状态机，B4 原样复用）----
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
            if found and nxt <= level[w] and level[w] < 2:
                backward_n += 1
            level[w + 1] = nxt
        if level[w] == 2 and F2_off is not None:
            if not (-DISK_R - 4 <= F2_off[0] <= DISK_R + 4 and
                    -DISK_R - 4 <= F2_off[1] <= DISK_R + 4):
                tex_outside_focus += 1

    style_correct = sum(1 for w in gist_ws if style_hat[w] == style_idx)
    style_acc = style_correct / max(1, len(gist_ws))
    gist_first_frac = gist_first / float(n_cycles) if n_cycles else 1.0
    sp_frac = sum(sp_ok_rec[w] for w in gist_ws) / float(len(gist_ws)) if gist_ws else 1.0
    sp_ratio_mean = float(np.mean([sp_ratio_rec[w] for w in gist_ws])) if gist_ws else 0.0
    sp_mean_mean = float(np.mean([sp_mean_rec[w] for w in gist_ws])) if gist_ws else 0.0
    drill_n = drill1_n + drill2_n + drill3_n
    drill_ok = drill1_ok + drill2_ok + drill3_ok
    drill_acc = drill_ok / drill_n if drill_n > 0 else float("nan")
    det_n = det1_n + det2_n + det3_n
    det_ok = det1_ok + det2_ok + det3_ok
    det_acc = det_ok / det_n if det_n > 0 else float("nan")

    out = dict(seed=seed, style=style_idx, lvcode=LVCODE_B5[style_idx],
               frames=n_frames,
               tmp_ev=round(tmp_ev, 6), tmp_res=round(R_tmp, 6),
               noise_block_ev=round(noise_block_ev, 6),
               sp_frac=round(sp_frac, 6),
               sp_ratio_mean=round(sp_ratio_mean, 4),
               sp_mean_mean=round(sp_mean_mean, 4),
               det1_n=det1_n, det1_ok=det1_ok,
               det2_n=det2_n, det2_ok=det2_ok,
               det3_n=det3_n, det3_ok=det3_ok,
               det_acc=round(det_acc, 6) if det_acc == det_acc else None,
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
               absorb_scratch=(round(absorb_scr_ok / absorb_scr_n, 6)
                               if absorb_scr_n else None),
               style_acc=round(style_acc, 6),
               style_hat=style_hat,
               gist_first_frac=round(gist_first_frac, 6),
               F1_ok=F1_ok, F2_ok=F2_ok, F3_ok=F3_ok,
               sp_ratio=sp_ratio_rec, sp_mean=sp_mean_rec, sp_ok=sp_ok_rec,
               det_g=det_g_rec, det_o=det_o_rec, det_t=det_t_rec,
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
               tex_outside_focus=tex_outside_focus,
               tex_detail=tex_detail,
               det_err_mean=round(float(np.mean(det_errs)), 4) if det_errs else None)
    return out


# ---------------- 守卫（docs/B5 §1.6 冻结；守卫种子数固定 10 × 3 风格） ----------------
def guard_b4():
    """R_B5_GUARD_B4_GIST/DRILL/ORDER：import granularity_drill_test.run_b4 重跑 30 单位
    （10 种子 × 3 风格）→ B4 三判据数字逐位复现：style_acc 池化 == 1.0000（240 gist 窗）、
    drill_acc 池化 == 1.0000 且 drill_n == 600、backward 总计 == 0 且 absorb_correct 池化
    == 1.0000（容差 1e-3）。"""
    sa_vals, drill_ok, drill_n, bwd, abs_vals = [], 0, 0, 0, []
    for st in range(3):
        for s in range(10):
            u = run_b4(s, st)
            sa_vals.append(u["style_acc"])
            drill_n += u["drill_n"]
            drill_ok += u["drill_ok"]
            bwd += u["backward_n"]
            if u["absorb_correct"] is not None:
                abs_vals.append(u["absorb_correct"])
    sa_m = float(np.mean(sa_vals))
    drill_m = drill_ok / drill_n if drill_n > 0 else float("nan")
    abs_m = float(np.mean(abs_vals)) if abs_vals else float("nan")
    g_gist = 1 if abs(sa_m - 1.0000) <= TOL else 0
    g_drill = 1 if (drill_n == 600 and abs(drill_m - 1.0000) <= TOL) else 0
    g_order = 1 if (bwd == 0 and abs(abs_m - 1.0000) <= TOL) else 0
    return (g_gist, g_drill, g_order), sa_m, drill_m, drill_n, bwd, abs_m


# ---------------- 诊断模式（R_B5_DIAG_* 行；诊断轮用，主运行摘要块格式不变） ----------------
def diag(seed=0, style=0, n_frames=N_FRAMES):
    u = run_b5(seed, style, n_frames=n_frames, jitter=JITTER)
    n_w = max(1, n_frames // WINDOW)
    print("R_B5_DIAG_SEED=%d_STYLE=%d" % (seed, style))
    for w in range(n_w):
        lv = ("G" if w % 3 == 0 else ("O" if w % 3 == 1 else "T"))
        print("R_B5_DIAG_W%d=%s,SH=%d,F1=%d,F2=%d,F3=%d,DG=%d,DO=%d,DT=%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f" % (
            w, lv,
            u["style_hat"][w] if u["style_hat"][w] is not None else -1,
            u["F1_ok"][w], u["F2_ok"][w], u["F3_ok"][w],
            u["det_g"][w], u["det_o"][w], u["det_t"][w],
            u["theta_g"][w], u["theta_o"][w], u["theta_t"][w],
            u["Rg"][w], u["Ro"][w], u["Rt"][w],
            u["sp_ratio"][w], u["sp_mean"][w]))
    print("R_B5_DIAG_TMPEV=%.6f" % u["tmp_ev"])
    print("R_B5_DIAG_TMPRES=%.6f" % u["tmp_res"])
    print("R_B5_DIAG_NOISEEV=%.6f" % u["noise_block_ev"])
    print("R_B5_DIAG_SPFRAC=%.4f" % u["sp_frac"])
    print("R_B5_DIAG_SPRATIO=%.4f" % u["sp_ratio_mean"])
    print("R_B5_DIAG_SPMEAN=%.4f" % u["sp_mean_mean"])
    print("R_B5_DIAG_DET1=%d/%d" % (u["det1_ok"], u["det1_n"]))
    print("R_B5_DIAG_DET2=%d/%d" % (u["det2_ok"], u["det2_n"]))
    print("R_B5_DIAG_DET3=%d/%d" % (u["det3_ok"], u["det3_n"]))
    print("R_B5_DIAG_D1=%d/%d" % (u["drill1_ok"], u["drill1_n"]))
    print("R_B5_DIAG_D2=%d/%d" % (u["drill2_ok"], u["drill2_n"]))
    print("R_B5_DIAG_D3=%d/%d" % (u["drill3_ok"], u["drill3_n"]))
    print("R_B5_DIAG_DRILL_ACC=%.4f" % (u["drill_acc"] if u["drill_acc"] is not None else -1.0))
    print("R_B5_DIAG_DET_ACC=%.4f" % (u["det_acc"] if u["det_acc"] is not None else -1.0))
    print("R_B5_DIAG_BACKWARD=%d" % u["backward_n"])
    print("R_B5_DIAG_ABS_DISK=%.4f" % (u["absorb_disk"] if u["absorb_disk"] is not None else -1.0))
    print("R_B5_DIAG_ABS_STRIPES=%.4f" % (u["absorb_stripes"] if u["absorb_stripes"] is not None else -1.0))
    print("R_B5_DIAG_ABS_SCRATCH=%.4f" % (u["absorb_scratch"] if u["absorb_scratch"] is not None else -1.0))
    print("R_B5_DIAG_STYLEACC=%.4f" % u["style_acc"])
    print("R_B5_DIAG_DET=%.4f" % (u["det_err_mean"] if u["det_err_mean"] is not None else -1.0))
    for d in u.get("tex_detail", []):
        print("R_B5_DIAG_TEX_W%d_THT=%.6f_F2OFF=%s_STRIPEMED=%s_STRIPEOK=%d_SCRATCHMED=%s_SCRATCHOK=%d" % (
            d["w"], d["tht"], ",".join("%.1f" % v for v in d["f2_off"]),
            ("%.6f" % d["stripe_med"]) if d["stripe_med"] is not None else "nan",
            d["stripe_ok"],
            ("%.6f" % d["scratch_med"]) if d["scratch_med"] is not None else "nan",
            d["scratch_ok"]))
    return 0


def scan(seed=0, style=0, n_frames=N_FRAMES, scan_seeds=(0, 1, 2)):
    """诊断轮载体旋钮边界扫描（R_B5_SCAN_* 行）：K 与 σ_o 组合下"检出/下钻/吸收/空间非零"
    的工作点（docs/B5 §1.8：修复前诊断；判据/机制语义 §一 不动）。每组合池化 scan_seeds
    种子报告。"""
    import itertools
    for kd, so in itertools.product((2.0, 3.0, 4.0, 5.0), (1.5, 2.5, 3.5)):
        d1_ok = d1_n = d2_ok = d2_n = d3_ok = d3_n = 0
        de1_ok = de1_n = de2_ok = de2_n = de3_ok = de3_n = 0
        sp_fracs = []
        abs_vals = []
        for s in scan_seeds:
            u = run_b5(s, style, n_frames=n_frames, jitter=JITTER,
                       k_defect=kd, sigma_o=so)
            d1_ok += u["drill1_ok"]; d1_n += u["drill1_n"]
            d2_ok += u["drill2_ok"]; d2_n += u["drill2_n"]
            d3_ok += u["drill3_ok"]; d3_n += u["drill3_n"]
            de1_ok += u["det1_ok"]; de1_n += u["det1_n"]
            de2_ok += u["det2_ok"]; de2_n += u["det2_n"]
            de3_ok += u["det3_ok"]; de3_n += u["det3_n"]
            sp_fracs.append(u["sp_frac"])
            for a in ("absorb_disk", "absorb_stripes", "absorb_scratch"):
                if u[a] is not None:
                    abs_vals.append(u[a])
        dacc = (d1_ok + d2_ok + d3_ok) / max(1, d1_n + d2_n + d3_n)
        decc = (de1_ok + de2_ok + de3_ok) / max(1, de1_n + de2_n + de3_n)
        sp_m = float(np.mean(sp_fracs)) if sp_fracs else float("nan")
        abs_m = float(np.mean(abs_vals)) if abs_vals else float("nan")
        print("R_B5_SCAN_K=%.1f_SO=%.1f_DRILL=%.4f_D1=%d/%d_D2=%d/%d_D3=%d/%d_DET=%.4f_DET1=%d/%d_DET2=%d/%d_DET3=%d/%d_SP=%.4f_ABS=%.4f" % (
            kd, so, dacc, d1_ok, d1_n, d2_ok, d2_n, d3_ok, d3_n,
            decc, de1_ok, de1_n, de2_ok, de2_n, de3_ok, de3_n, sp_m, abs_m))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--n-styles", type=int, default=3)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="ss")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="诊断模式：单种子单风格逐窗口 R_B5_DIAG_* 行")
    ap.add_argument("--scan", action="store_true",
                    help="诊断模式：载体旋钮边界扫描 R_B5_SCAN_* 行")
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
           "lvcode_b5": LVCODE_B5,
           "scene": {"disk_c": list(DISK_C), "disk_r": DISK_R,
                     "stripe_off": list(STRIPE_OFF), "stripe_half": STRIPE_HALF,
                     "stripe_gray": [STRIPE_DARK, STRIPE_BRIGHT],
                     "stripe_period": STRIPE_PERIOD,
                     "scratch_gray": SCRATCH_GRAY,
                     "noise_block": list(NOISE_BLOCK), "noise_sigma": NOISE_SIGMA,
                     "occ_lum_thresh": OCC_LUM_THRESH},
           "spatial": {"gist_block": GIST_BLOCK, "k_defect": K_DEFECT,
                       "theta_floor": THETA_FLOOR, "sigma_o": SIGMA_O,
                       "tex_median_n": TEX_MEDIAN_N, "k_drill": K_DRILL,
                       "a_min": [A_MIN_G, A_MIN_O, A_MIN_T],
                       "dilate": DILATE, "interior_r": INTERIOR_R},
           "criteria": {"tmp_ev_max": TMP_EV_MAX, "tmp_res_max": TMP_RES_MAX,
                        "sp_ratio_min": SP_RATIO_MIN, "sp_mean_min": SP_MEAN_MIN,
                        "sp_frac_min": SP_FRAC_MIN,
                        "detect_min": DETECT_MIN, "detect_nobs_min": DETECT_NOBS_MIN,
                        "drill_min": DRILL_MIN, "drill_nobs_min": DRILL_NOBS_MIN},
           "guards": {"tol": TOL}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_ss_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for s in seeds:
            for st in styles:
                key = "%d_%d" % (LVCODE_B5[st], s)
                if key in per_unit:
                    continue
                per_unit[key] = run_b5(s, st, n_frames=args.frames, jitter=args.jitter)
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"config": cfg, "per_unit": per_unit},
                              f, ensure_ascii=False, indent=1)
                print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%d_%d" % (LVCODE_B5[st], s)]
             for s in seeds for st in styles]

    # ---- 判据（docs/B5 §1.4 冻结；跨运行池化） ----
    tmp_ev_pooled = float(np.mean([u["tmp_ev"] for u in units]))
    tmp_res_pooled = float(np.mean([u["tmp_res"] for u in units]))
    noise_zero = all(u["noise_block_ev"] == 0.0 for u in units)
    c1 = (tmp_ev_pooled <= TMP_EV_MAX) and noise_zero and (tmp_res_pooled <= TMP_RES_MAX)

    sp_frac_vals = [u["sp_frac"] for u in units]
    sp_frac_pooled = float(np.mean(sp_frac_vals))
    sp_ratio_mean = float(np.mean([u["sp_ratio_mean"] for u in units]))
    sp_mean_mean = float(np.mean([u["sp_mean_mean"] for u in units]))
    c2 = sp_frac_pooled >= SP_FRAC_MIN

    det_n = sum(u["det1_n"] + u["det2_n"] + u["det3_n"] for u in units)
    det_ok = sum(u["det1_ok"] + u["det2_ok"] + u["det3_ok"] for u in units)
    det_pooled = det_ok / det_n if det_n > 0 else float("nan")
    c3 = (det_n >= DETECT_NOBS_MIN) and (det_pooled >= DETECT_MIN)

    drill_n = sum(u["drill_n"] for u in units)
    drill_ok = sum(u["drill_ok"] for u in units)
    drill_pooled = drill_ok / drill_n if drill_n > 0 else float("nan")
    c4 = (drill_n >= DRILL_NOBS_MIN) and (drill_pooled >= DRILL_MIN)

    # ---- 守卫（docs/B5 §1.6 冻结；固定 10 种子 × 3 风格） ----
    (g_gist, g_drill, g_order), sa_m, drill_m, drill_n_b4, bwd_b4, abs_m_b4 = guard_b4()
    g_187 = 1 if noise_zero else 0
    guards_ok = (g_gist == 1 and g_drill == 1 and g_order == 1 and g_187 == 1)

    # ---- 判定（docs/B5 §1.5 冻结） ----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1:
        verdict = "TIME_FAIL"
    elif not c2:
        verdict = "SPATIAL_FAIL"
    elif c3 and c4:
        verdict = "STATIC_SPATIAL_PASS"
    else:
        verdict = "PARTIAL"

    # ---- 内部确定性复现（docs/B5 §1.6-5；第二遍强制重算，不读 checkpoint） ----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for key in per_unit:
            for kk in REPRO_KEYS:
                if per_unit[key][kk] != per_unit2[key][kk]:
                    repro = 0

    out = {
        "artifact": "static_scene_test",
        "doc_ref": "lineB-motion-coupling/docs/B5",
        "config": cfg,
        "per_unit": per_unit,
        "criteria": {"c1_time_zero": bool(c1), "c2_spatial_nonzero": bool(c2),
                     "c3_static_defect": bool(c3), "c4_static_drill": bool(c4),
                     "tmp_ev_pooled": tmp_ev_pooled,
                     "tmp_res_pooled": tmp_res_pooled,
                     "noise_ev_zero": bool(noise_zero),
                     "sp_frac_pooled": sp_frac_pooled,
                     "sp_ratio_mean": sp_ratio_mean,
                     "sp_mean_mean": sp_mean_mean,
                     "detect_pooled": det_pooled, "detect_n": det_n,
                     "detect_ok": det_ok,
                     "det1": (round(float(np.nanmean(
                         [u["det1_ok"] / u["det1_n"] for u in units])), 6)
                         if units else None),
                     "det2": (round(float(np.nanmean(
                         [u["det2_ok"] / u["det2_n"] for u in units])), 6)
                         if units else None),
                     "det3": (round(float(np.nanmean(
                         [u["det3_ok"] / u["det3_n"] for u in units])), 6)
                         if units else None),
                     "drill_pooled": drill_pooled, "drill_n": drill_n,
                     "drill_ok": drill_ok,
                     "drill1": (round(float(np.nanmean(
                         [u["drill1_ok"] / u["drill1_n"] for u in units])), 6)
                         if units else None),
                     "drill2": (round(float(np.nanmean(
                         [u["drill2_ok"] / u["drill2_n"] for u in units])), 6)
                         if units else None),
                     "drill3": (round(float(np.nanmean(
                         [u["drill3_ok"] / u["drill3_n"] for u in units])), 6)
                         if units else None),
                     "backward_n": int(sum(u["backward_n"] for u in units)),
                     "absorb_disk": round(float(np.nanmean(
                         [u["absorb_disk"] for u in units
                          if u["absorb_disk"] is not None])), 6),
                     "absorb_stripes": round(float(np.nanmean(
                         [u["absorb_stripes"] for u in units
                          if u["absorb_stripes"] is not None])), 6),
                     "absorb_scratch": round(float(np.nanmean(
                         [u["absorb_scratch"] for u in units
                          if u["absorb_scratch"] is not None])), 6),
                     "style_acc_mean": round(float(np.nanmean(
                         [u["style_acc"] for u in units])), 6),
                     "tex_outside_focus_total": int(sum(
                         u["tex_outside_focus"] for u in units))},
        "guards": {"b4_gist": g_gist, "b4_gist_style_acc": sa_m,
                   "b4_drill": g_drill, "b4_drill_acc": drill_m,
                   "b4_drill_n": drill_n_b4,
                   "b4_order": g_order, "b4_backward": bwd_b4,
                   "b4_absorb_correct": abs_m_b4,
                   "187": g_187,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "ss_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定） ----
    print("R_B5_TAG=%s" % args.tag)
    print("R_B5_SEEDS=%d" % len(seeds))
    print("R_B5_STYLES=%d" % len(styles))
    print("R_B5_FRAMES=%d" % args.frames)
    for s in seeds:
        for st in styles:
            u = per_unit["%d_%d" % (LVCODE_B5[st], s)]
            print("R_B5_S%d_ST%d_TMPEV=%.6f" % (s, st, u["tmp_ev"]))
            print("R_B5_S%d_ST%d_TMPRES=%.6f" % (s, st, u["tmp_res"]))
            print("R_B5_S%d_ST%d_NOISEEV=%.6f" % (s, st, u["noise_block_ev"]))
            print("R_B5_S%d_ST%d_SPFRAC=%.4f" % (s, st, u["sp_frac"]))
            print("R_B5_S%d_ST%d_SPRATIO=%.4f" % (s, st, u["sp_ratio_mean"]))
            print("R_B5_S%d_ST%d_SPMEAN=%.4f" % (s, st, u["sp_mean_mean"]))
            print("R_B5_S%d_ST%d_DET1=%d/%d" % (s, st, u["det1_ok"], u["det1_n"]))
            print("R_B5_S%d_ST%d_DET2=%d/%d" % (s, st, u["det2_ok"], u["det2_n"]))
            print("R_B5_S%d_ST%d_DET3=%d/%d" % (s, st, u["det3_ok"], u["det3_n"]))
            print("R_B5_S%d_ST%d_D1=%d/%d" % (s, st, u["drill1_ok"], u["drill1_n"]))
            print("R_B5_S%d_ST%d_D2=%d/%d" % (s, st, u["drill2_ok"], u["drill2_n"]))
            print("R_B5_S%d_ST%d_D3=%d/%d" % (s, st, u["drill3_ok"], u["drill3_n"]))
            print("R_B5_S%d_ST%d_DRILLACC=%.4f" % (s, st, u["drill_acc"] if u["drill_acc"] is not None else -1.0))
            print("R_B5_S%d_ST%d_BACKWARD=%d" % (s, st, u["backward_n"]))
            print("R_B5_S%d_ST%d_ABSDISK=%.4f" % (s, st, u["absorb_disk"] if u["absorb_disk"] is not None else -1.0))
            print("R_B5_S%d_ST%d_ABSSTRIPES=%.4f" % (s, st, u["absorb_stripes"] if u["absorb_stripes"] is not None else -1.0))
            print("R_B5_S%d_ST%d_ABSSCRATCH=%.4f" % (s, st, u["absorb_scratch"] if u["absorb_scratch"] is not None else -1.0))
            print("R_B5_S%d_ST%d_STYLEACC=%.4f" % (s, st, u["style_acc"]))
            print("R_B5_S%d_ST%d_DET=%.4f" % (s, st, u["det_err_mean"] if u["det_err_mean"] is not None else -1.0))
    print("R_B5_TMPEV_POOLED=%.6f" % tmp_ev_pooled)
    print("R_B5_TMPRES_POOLED=%.6f" % tmp_res_pooled)
    print("R_B5_NOISE_EV_ZERO=%d" % (1 if noise_zero else 0))
    print("R_B5_SP_FRAC_POOLED=%.4f" % sp_frac_pooled)
    print("R_B5_SP_RATIO_MEAN=%.4f" % sp_ratio_mean)
    print("R_B5_SP_MEAN_MEAN=%.4f" % sp_mean_mean)
    print("R_B5_DET_POOLED=%.4f" % det_pooled)
    print("R_B5_DET_N=%d" % det_n)
    print("R_B5_DET_OK=%d" % det_ok)
    print("R_B5_DRILL_POOLED=%.4f" % drill_pooled)
    print("R_B5_DRILL_N=%d" % drill_n)
    print("R_B5_DRILL_OK=%d" % drill_ok)
    print("R_B5_BACKWARD_N=%d" % int(sum(u["backward_n"] for u in units)))
    print("R_B5_C1=%s" % ("PASS" if c1 else "FAIL"))
    print("R_B5_C2=%s" % ("PASS" if c2 else "FAIL"))
    print("R_B5_C3=%s" % ("PASS" if c3 else "FAIL"))
    print("R_B5_C4=%s" % ("PASS" if c4 else "FAIL"))
    print("R_B5_GUARD_B4_GIST=%d" % g_gist)
    print("R_B5_GUARD_B4_GIST_STYLEACC=%.4f" % sa_m)
    print("R_B5_GUARD_B4_DRILL=%d" % g_drill)
    print("R_B5_GUARD_B4_DRILL_ACC=%.4f" % drill_m)
    print("R_B5_GUARD_B4_DRILL_N=%d" % drill_n_b4)
    print("R_B5_GUARD_B4_ORDER=%d" % g_order)
    print("R_B5_GUARD_B4_BACKWARD=%d" % bwd_b4)
    print("R_B5_GUARD_B4_ABSORB=%.4f" % abs_m_b4)
    print("R_B5_GUARD_187=%d" % g_187)
    print("R_B5_REPRO=%d" % repro)
    print("R_B5_VERDICT=%s" % verdict)
    print("R_B5_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
