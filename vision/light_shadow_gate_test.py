"""vision/light_shadow_gate_test.py — 光影判别第二格：否决门 + 暗物体对照臂
（docs/261 预注册设计，判据/旋钮/守卫冻结；本脚本为唯一新增文件，import 复用第一格
vision/light_shadow_test.py 与 vision/critical_point.py，未修改任何既有脚本）。

目标（docs/261 §1.1，docs/260 §五.2 盲区的补格）：把 docs/260 C1 从"检移动暗区"
（暗于长期背景 + 空间相干 + 时间移动）升级为人眼式联合判别的第 1 步：
  否决门（负向硬判，任一触发 → 判为物体/非阴影）：
    V1 闭合轮廓：候选连通域不触图像边界（闭合）→ 否决为物体
    V3 主轴方向：候选非各向同性（PCA 特征值比 ≥ RATIO_MIN=1.5）且主轴与光照方向
      夹角 > TOL_AXIS=30° → 否决为非阴影（各向同性 = 主轴无定义，跳过）
  证据门（正向支持，与否决门同一几何量的正表述）：
    E1 开放拓扑：候选触图像边界；E2 主轴沿光照：主轴与光照方向夹角 ≤ TOL_AXIS
  暗物体对照臂：与遮挡物同运动学（同 rng 派生 → 同相位/同噪声/同路径）的暗圆盘
    （灰度 32 = 阴影最暗值 64×0.5，亮度与阴影同值域，adversarial match），不投影
    阴影（GT 阴影掩码全空）；主/控为配对场景。测 docs/260 C1 口径对它的误报率
    （fp_legacy，第一格盲区诚实暴露）vs 加否决门后（fp_gated，机制增益）。

场景（docs/261 §1.2）：
  主场景 = make_shadow_scene 逐字复用（docs/260 §1.2 冻结场景）；
  对照臂 = make_control_scene（后处理派生：亮遮挡物像素 → 暗圆盘 32、阴影像素恢复
  未遮蔽背景（同噪声场）、GT 阴影置空——运动学/噪声/相位与主场景逐位相同）。

判据（docs/261 §1.4，冻结）：
  C1 CTRL_FP_DROP  [新][机制][合成受控]：fp_legacy − fp_gated ≥ 0.50 且 fp_gated ≤ 0.10
  C2 POS_DETECT_KEEP[新][机制][合成受控]：det_gated ≥ 0.80 且 det_legacy − det_gated ≤ 0.05
  C3 LIGHT_DIR_KEEP [新][机制][合成受控]：gated 光照方向角度误差中位数 ≤ 15°
  判定：三判据全过 + 守卫全过 = SHADOW_VETO_PASS；C1/C2/C3 不过按名报
  （CTRL_FP_DROP_FAIL / POS_DETECT_FAIL / LIGHT_DIR_FAIL）；守卫不过 = GUARD_FAIL。

守卫（docs/261 §1.6，冻结）：
  R_GS_GUARD_CELL1：import 第一格 run_unit 重跑全部 40 个主场景 (级,种子)，本脚本
    legacy-main 的 det_rate/fp_rate/ld_med/sfs_err 与 run_unit 逐位一致（40 单位 × 4 键）
  R_GS_GUARD_CP5：import 第一格 guard_cp5（复现 docs/232 L5：SC2 ∈ [2.0,3.6]、SC_late ≥ 0.5）
  R_GS_GUARD_MAD：import 第一格 guard_mad（σ̂ 与理论 √2·3/64 ≈ 0.0663 偏差 ≤ 25%）
  R_GS_REPRO：--repro 时主 40 + 控 40 整体重跑第二遍，逐项位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_GS_* 摘要块
（顺序固定）；JSON 归档 vision/out/results/gs_<tag>.json + checkpoint
ckpt_gs_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则抽取；
禁止读取 logs/*.log 与 vision/out/results/*.json 原文。

用法：
  python vision/light_shadow_gate_test.py --levels 30,31,32,33 --n-seeds 10 --tag main --repro
  python vision/light_shadow_gate_test.py --levels 30 --n-seeds 1 --frames 240 --tag timing
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import CPLoop, mean_sd, bootstrap_ci, JITTER
from light_shadow_test import (
    make_shadow_scene, run_unit, largest_component, iou, circ_err_deg,
    guard_cp5, guard_mad,
    W, H, BG_CELL, BG_DARK, BG_BRIGHT,
    SPHERE_C, SPHERE_R, OCC_R, OCC_GRAY, ORBIT_C, ORBIT_R, OCC_FREQ,
    SHADOW_MULT, NOISE_SIGMA,
    DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, WARMUP, SFS_LO, SFS_HI,
    OCC_LUM_THRESH, LIGHT_AZIMUTH,
    IOU_DET, IOU_FP, ANG_TOL, C2_MED_MAX,
)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 否决门旋钮（docs/261 §1.3 冻结）----
TOL_AXIS = 30.0          # V3/E2：主轴与光照方向无向夹角容差（度）
RATIO_MIN = 1.5          # V3 适用门槛：PCA 特征值比 λ1/λ2 ≥ 此值才判主轴（各向同性跳过）
CTRL_OCC_GRAY = 32.0     # 对照臂暗圆盘灰度 = 阴影最暗值 64×0.5（亮度匹配，只靠几何判别）

# ---- 判据阈值（docs/261 §1.4 冻结）----
CTRL_DROP_MIN = 0.50     # fp_legacy − fp_gated ≥ 0.50（对照臂下降幅度下限）
CTRL_FP_MAX = 0.10       # fp_gated ≤ 0.10
POS_DET_MIN = 0.80       # det_gated ≥ 0.80（docs/260 C1 同阈值）
POS_DET_DROP_MAX = 0.05  # det_legacy − det_gated ≤ 0.05（否决门不误伤正样本）

REPRO_KEYS_MAIN = ["det_legacy", "fp_legacy", "det_gated", "fp_gated",
                   "ld_med_legacy", "ld_med_gated", "sfs_err",
                   "veto_rate", "v1_rate", "v3_rate",
                   "false_veto_rate", "e1_rate", "e2_rate"]
REPRO_KEYS_CTRL = ["fp_legacy", "fp_gated", "veto_rate", "v1_rate", "v3_rate", "sfs_err"]


def make_control_scene(lvcode, seed, n_frames=240, jitter=JITTER):
    """对照臂场景 = 主场景（make_shadow_scene）逐位同源后处理（docs/261 §1.2）：
    亮遮挡物像素 → 暗圆盘（CTRL_OCC_GRAY=32）；阴影像素恢复未遮蔽背景（+bg×0.5，
    噪声场不变）；GT 阴影掩码置空。运动学/噪声/相位与主场景逐位相同 → 配对场景。
    注意：rng 在 make_shadow_scene 内部已消费，此处不再调用 rng。
    返回 (frames, gts)：gts["per_frame"][t]["occ"] = 暗圆盘掩码（= 主场景遮挡物掩码），
    ["shadow"] 全 False。"""
    frames, gts = make_shadow_scene(lvcode, seed, n_frames=n_frames, jitter=jitter)
    bg = np.zeros((H, W), np.float32)
    for y in range(0, H, BG_CELL):
        for x in range(0, W, BG_CELL):
            bg[y:y + BG_CELL, x:x + BG_CELL] = \
                BG_DARK if ((x // BG_CELL) + (y // BG_CELL)) % 2 == 0 else BG_BRIGHT
    out_frames = []
    for t, g in enumerate(frames):
        img = g.astype(np.float32).copy()
        gt = gts["per_frame"][t]
        img[gt["occ"]] = CTRL_OCC_GRAY
        sh = gt["shadow"]
        if sh.any():
            img[sh] += bg[sh] * SHADOW_MULT
        out_frames.append(np.clip(img, 0, 255).astype(np.uint8))
        gt["shadow"] = np.zeros((H, W), bool)
    return out_frames, gts


def touches_boundary(mask):
    """候选连通域是否触图像边界（开放拓扑的签名）。"""
    return bool(mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any())


def pca_axis(mask):
    """像素坐标 PCA：返回 (主轴无向角 α∈[0,180) 度, 特征值比 λ1/λ2) or None（点数 < 2）。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    xy = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    cov = np.cov(xy, rowvar=False)
    w, v = np.linalg.eigh(cov)
    ax = v[:, int(np.argmax(w))]
    alpha = float(np.rad2deg(np.arctan2(ax[1], ax[0]))) % 180.0
    ratio = float(w.max() / max(w.min(), 1e-9))
    return alpha, ratio


def axis_err_deg(alpha, theta):
    """无向轴 α∈[0,180) 与光照方向 θ（度）的无向夹角（处理 180° 二义）。"""
    base = theta % 180.0
    d = abs(alpha - base)
    return min(d, 180.0 - d)


def run_unit_gated(lvcode, seed, scene_kind, n_frames=240, jitter=JITTER):
    """跑 (级,种子,场景) 一次完整运行，返回 legacy/gated 双口径指标 + 否决/证据门明细。

    scene_kind: "main"（真阴影正样本）| "ctrl"（暗圆盘无阴影对照臂）。
    legacy 口径 = docs/260 C1（无否决门，检测路径逐字同实现）；
    gated = legacy AND NOT(V1 OR V3)。对照臂 GT 阴影全空 → 任何 active 帧都是 FP。
    """
    theta = LIGHT_AZIMUTH[lvcode]
    if scene_kind == "ctrl":
        frames, gts = make_control_scene(lvcode, seed, n_frames=n_frames, jitter=jitter)
    else:
        frames, gts = make_shadow_scene(lvcode, seed, n_frames=n_frames, jitter=jitter)
    loop = CPLoop(window=10)
    eval_lo = WARMUP
    eval_hi = n_frames
    n_eval = max(1, eval_hi - eval_lo)

    # ---- Pass A：适应（docs/260 诊断轮 D2 冻结口径，逐字同实现）----
    occ_thresh_log = np.log(OCC_LUM_THRESH + 1.0)
    buf = np.empty((n_frames, H, W), np.float32)
    for t, g in enumerate(frames):
        buf[t] = g.astype(np.float32)
    buf_masked = buf.copy()
    for t, g in enumerate(frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        buf_masked[t][L > occ_thresh_log] = np.nan
    ref_dark = np.nanpercentile(buf_masked, 95.0, axis=0)
    ref_dark = np.nan_to_num(ref_dark, nan=0.0)
    ref_dark_log = np.log(np.maximum(ref_dark, 1.0))

    shadow_masks = []          # 每帧阴影候选掩码（时间移动 IoU 门用）
    det_leg, fp_leg = [], []   # legacy 口径 det/fp（docs/260 C1）
    det_gat, fp_gat = [], []   # gated 口径 det/fp
    ld_leg_errs, ld_gat_errs = [], []
    veto_flags, v1_flags, v3_flags = [], [], []
    false_veto_flags, e1_flags, e2_flags = [], [], []
    gray_mean = np.zeros((H, W), np.float64)

    # ---- Pass B：判断（预测回路 + 检测 + 否决门）----
    for t, g in enumerate(frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        loop.step(g)
        if t >= eval_lo:
            gray_mean += g.astype(np.float64) / n_eval
        dark = (ref_dark_log - L) > DELTA_SHADOW
        bright = L > occ_thresh_log
        d_mask, d_area, d_c = largest_component(dark)
        b_mask, b_area, b_c = largest_component(bright)
        shadow_masks.append(d_mask)

        # legacy active（docs/260 C1 口径：空间相干 AND 帧间移动）
        active = (d_area >= A_MIN)
        if t >= eval_lo + K_MOVE:
            m0 = shadow_masks[t - K_MOVE]
            move = 1.0 - iou(d_mask, m0)
            active = active and (move >= MOVE_IOU)

        # ---- 否决门（docs/261 §1.3；任一否决 → 判为物体/非阴影）----
        v1 = False
        v3 = False
        if d_area >= A_MIN:
            v1 = not touches_boundary(d_mask)          # V1 闭合轮廓否决
            pr = pca_axis(d_mask)
            if pr is not None:
                alpha, ratio = pr
                if ratio >= RATIO_MIN:                 # 各向同性 → 主轴无定义，跳过 V3
                    v3 = axis_err_deg(alpha, theta) > TOL_AXIS   # V3 主轴偏离光照否决
        veto = v1 or v3
        gated_active = active and (not veto)

        if t >= eval_lo:
            gt_sh = gts["per_frame"][t]["shadow"]
            iou_v = iou(d_mask, gt_sh)
            det_leg.append(1.0 if (active and iou_v >= IOU_DET) else 0.0)
            fp_leg.append(1.0 if (active and iou_v < IOU_FP) else 0.0)
            det_gat.append(1.0 if (gated_active and iou_v >= IOU_DET) else 0.0)
            fp_gat.append(1.0 if (gated_active and iou_v < IOU_FP) else 0.0)
            if active:
                veto_flags.append(1.0 if veto else 0.0)
                v1_flags.append(1.0 if v1 else 0.0)
                v3_flags.append(1.0 if v3 else 0.0)
                if scene_kind == "main" and iou_v >= IOU_DET:   # 真阴影帧
                    false_veto_flags.append(1.0 if veto else 0.0)
                    e1_flags.append(1.0 if (not v1) else 0.0)   # E1 开放拓扑
                    e2_flags.append(1.0 if (not v3) else 0.0)   # E2 主轴沿光照
            if active and d_c is not None and b_c is not None:
                ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
                est = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
                ld_leg_errs.append(circ_err_deg(est, theta))
            if gated_active and d_c is not None and b_c is not None:
                ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
                est = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
                ld_gat_errs.append(circ_err_deg(est, theta))

    det_leg_r = float(np.mean(det_leg)) if det_leg else 0.0
    fp_leg_r = float(np.mean(fp_leg)) if fp_leg else 0.0
    det_gat_r = float(np.mean(det_gat)) if det_gat else 0.0
    fp_gat_r = float(np.mean(fp_gat)) if fp_gat else 0.0
    ld_med_leg = float(np.median(ld_leg_errs)) if ld_leg_errs else float("nan")
    ld_med_gat = float(np.median(ld_gat_errs)) if ld_gat_errs else float("nan")
    ld_acc_gat = float(np.mean([1.0 if e <= ANG_TOL else 0.0 for e in ld_gat_errs])) \
        if ld_gat_errs else 0.0
    veto_rate = float(np.mean(veto_flags)) if veto_flags else float("nan")
    v1_rate = float(np.mean(v1_flags)) if v1_flags else float("nan")
    v3_rate = float(np.mean(v3_flags)) if v3_flags else float("nan")
    false_veto_rate = float(np.mean(false_veto_flags)) if false_veto_flags else float("nan")
    e1_rate = float(np.mean(e1_flags)) if e1_flags else float("nan")
    e2_rate = float(np.mean(e2_flags)) if e2_flags else float("nan")

    # ---- SfS（docs/260 §1.3 同实现，报告性；球面不变 → 主/控应接近）----
    dark_cap, cap_area, cap_c = largest_component(gray_mean < SFS_LO, min_area=50)
    if cap_area > 0 and cap_c is not None:
        pidx = int(np.argmax(gray_mean))
        py, px = divmod(pidx, W)
        sfs_err = circ_err_deg(
            float(np.rad2deg(np.arctan2(py - cap_c[1], px - cap_c[0]))) % 360.0, theta)
    else:
        sfs_err = float("nan")

    return dict(kind=scene_kind, seed=seed, lvcode=lvcode, theta=theta,
                det_legacy=det_leg_r, fp_legacy=fp_leg_r,
                det_gated=det_gat_r, fp_gated=fp_gat_r,
                ld_med_legacy=ld_med_leg, ld_med_gated=ld_med_gat,
                ld_acc_gated=ld_acc_gat, sfs_err=sfs_err,
                veto_rate=veto_rate, v1_rate=v1_rate, v3_rate=v3_rate,
                false_veto_rate=false_veto_rate, e1_rate=e1_rate, e2_rate=e2_rate,
                n_active=len(veto_flags))


# ---------------- 守卫（docs/261 §1.6 冻结） ----------------
def guard_cell1(units_main, levels, seeds, n_frames=240):
    """同代码路径复现第一格：import run_unit 重跑全部主场景 (级,种子)，本脚本
    legacy-main 的 det_rate/fp_rate/ld_med/sfs_err 与 run_unit 逐位一致。
    units_main: list of dict（kind=="main"）。返回 (ok, n_matched, n_bad)。"""
    n_bad = 0
    n = 0
    for lv in levels:
        for seed in seeds:
            ru = run_unit(lv, seed, n_frames=n_frames)
            mine = [u for u in units_main
                    if u["lvcode"] == lv and u["seed"] == seed]
            if not mine:
                n_bad += 1
                continue
            u = mine[0]
            n += 1
            if not (u["det_legacy"] == ru["det_rate"]
                    and u["fp_legacy"] == ru["fp_rate"]
                    and u["ld_med_legacy"] == ru["ld_med"]
                    and u["sfs_err"] == ru["sfs_err"]):
                n_bad += 1
    return 1 if (n_bad == 0 and n > 0) else 0, n, n_bad


# ---------------- 统计外壳（critical_point 同款） ----------------
def agg_col(vals):
    vals = [v for v in vals if v == v]     # 滤 NaN
    if not vals:
        return float("nan"), 0.0, [float("nan"), float("nan")]
    m, s = mean_sd(vals)
    lo, hi = bootstrap_ci(vals)
    return m, s, [float(lo), float(hi)]


def verdict_of(agg):
    c1 = (agg["fp_drop"] >= CTRL_DROP_MIN) and (agg["fp_gated_ctrl"] <= CTRL_FP_MAX)
    c2 = (agg["det_gated_main"] >= POS_DET_MIN) and (agg["det_drop"] <= POS_DET_DROP_MAX)
    c3 = (agg["ld_med_gated_main"] <= C2_MED_MAX)
    fails = []
    if not c1:
        fails.append("CTRL_FP_DROP_FAIL")
    if not c2:
        fails.append("POS_DETECT_FAIL")
    if not c3:
        fails.append("LIGHT_DIR_FAIL")
    return c1, c2, c3, (fails if fails else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="30,31,32,33")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="gs")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    kinds = ["main", "ctrl"]
    t0 = time.time()

    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "jitter": args.jitter, "tag": args.tag,
           "scene_kinds": kinds,
           "mechanism": {"delta_shadow": DELTA_SHADOW, "a_min": A_MIN,
                         "k_move": K_MOVE, "move_iou": MOVE_IOU,
                         "warmup": WARMUP, "sfs_lo": SFS_LO,
                         "veto": {"tol_axis": TOL_AXIS, "ratio_min": RATIO_MIN},
                         "control_occ_gray": CTRL_OCC_GRAY},
           "criteria": {"iou_det": IOU_DET, "iou_fp": IOU_FP, "ang_tol": ANG_TOL,
                        "ctrl_drop_min": CTRL_DROP_MIN, "ctrl_fp_max": CTRL_FP_MAX,
                        "pos_det_min": POS_DET_MIN, "pos_det_drop_max": POS_DET_DROP_MAX,
                        "c2_med_max": C2_MED_MAX},
           "scene": {"light_azimuth": LIGHT_AZIMUTH, "sphere_c": list(SPHERE_C),
                     "sphere_r": SPHERE_R, "orbit_c": list(ORBIT_C),
                     "orbit_r": ORBIT_R, "occ_r": OCC_R, "occ_freq": OCC_FREQ,
                     "occ_gray": OCC_GRAY, "shadow_mult": SHADOW_MULT,
                     "noise": NOISE_SIGMA}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_gs_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for kind in kinds:
            for lv in levels:
                for seed in seeds:
                    key = "%s_%d_%d" % (kind[0].upper(), lv, seed)
                    if key in per_unit:
                        continue
                    per_unit[key] = run_unit_gated(lv, seed, kind,
                                                   n_frames=args.frames,
                                                   jitter=args.jitter)
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump({"config": cfg, "per_unit": per_unit},
                                  f, ensure_ascii=False, indent=1)
                    print("PROGRESS", flush=True)
        return per_unit

    per_unit = run_all()
    units_main = [per_unit["M_%d_%d" % (lv, s)] for lv in levels for s in seeds]
    units_ctrl = [per_unit["C_%d_%d" % (lv, s)] for lv in levels for s in seeds]

    def col(units, k):
        return [u[k] for u in units]

    fp_leg_ctrl, fp_leg_ctrl_sd, fp_leg_ctrl_ci = agg_col(col(units_ctrl, "fp_legacy"))
    fp_gat_ctrl, fp_gat_ctrl_sd, fp_gat_ctrl_ci = agg_col(col(units_ctrl, "fp_gated"))
    det_leg_main, det_leg_main_sd, _ = agg_col(col(units_main, "det_legacy"))
    det_gat_main, det_gat_main_sd, det_gat_main_ci = agg_col(col(units_main, "det_gated"))
    ld_gat_main, ld_gat_main_sd, ld_gat_main_ci = agg_col(col(units_main, "ld_med_gated"))
    ld_acc_gat_main, ld_acc_gat_main_sd, _ = agg_col(col(units_main, "ld_acc_gated"))
    sfs_main, sfs_main_sd, _ = agg_col(col(units_main, "sfs_err"))
    sfs_ctrl, sfs_ctrl_sd, _ = agg_col(col(units_ctrl, "sfs_err"))
    veto_ctrl, _, _ = agg_col(col(units_ctrl, "veto_rate"))
    v1_ctrl, _, _ = agg_col(col(units_ctrl, "v1_rate"))
    v3_ctrl, _, _ = agg_col(col(units_ctrl, "v3_rate"))
    fv_main, _, _ = agg_col(col(units_main, "false_veto_rate"))
    e1_main, _, _ = agg_col(col(units_main, "e1_rate"))
    e2_main, _, _ = agg_col(col(units_main, "e2_rate"))

    fp_drop = fp_leg_ctrl - fp_gat_ctrl
    det_drop = det_leg_main - det_gat_main

    agg = {"fp_legacy_ctrl": fp_leg_ctrl, "fp_legacy_ctrl_sd": fp_leg_ctrl_sd,
           "fp_legacy_ctrl_ci": fp_leg_ctrl_ci,
           "fp_gated_ctrl": fp_gat_ctrl, "fp_gated_ctrl_sd": fp_gat_ctrl_sd,
           "fp_gated_ctrl_ci": fp_gat_ctrl_ci,
           "fp_drop": fp_drop,
           "det_legacy_main": det_leg_main, "det_legacy_main_sd": det_leg_main_sd,
           "det_gated_main": det_gat_main, "det_gated_main_sd": det_gat_main_sd,
           "det_gated_main_ci": det_gat_main_ci,
           "det_drop": det_drop,
           "ld_med_gated_main": ld_gat_main, "ld_med_gated_main_sd": ld_gat_main_sd,
           "ld_med_gated_main_ci": ld_gat_main_ci,
           "ld_acc_gated_main": ld_acc_gat_main, "ld_acc_gated_main_sd": ld_acc_gat_main_sd,
           "sfs_main": sfs_main, "sfs_main_sd": sfs_main_sd,
           "sfs_ctrl": sfs_ctrl, "sfs_ctrl_sd": sfs_ctrl_sd,
           "veto_rate_ctrl": veto_ctrl, "v1_rate_ctrl": v1_ctrl, "v3_rate_ctrl": v3_ctrl,
           "false_veto_rate_main": fv_main, "e1_rate_main": e1_main, "e2_rate_main": e2_main}

    c1, c2, c3, fails = verdict_of(agg)

    # ---- 守卫 ----
    g_cell1, g_cell1_n, g_cell1_bad = guard_cell1(units_main, levels, seeds,
                                                  n_frames=args.frames)
    g_cp5, g_cp5_sc2, g_cp5_scl = guard_cp5(n_seeds=args.n_seeds, n_frames=args.frames)
    g_mad, g_mad_sig, g_mad_exp = guard_mad()
    guards_ok = (g_cell1 == 1) and (g_cp5 == 1) and (g_mad == 1)

    # ---- 判定（docs/261 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif fails is None:
        verdict = "SHADOW_VETO_PASS"
    else:
        verdict = "_".join(fails)

    # ---- 内部确定性复现（docs/261 §1.6-4；第二遍强制重算，不读 checkpoint）----
    repro = 1
    if args.repro:
        per_unit2 = run_all(use_resume=False)
        for kind in kinds:
            keys = REPRO_KEYS_MAIN if kind == "main" else REPRO_KEYS_CTRL
            for lv in levels:
                for s in seeds:
                    k = "%s_%d_%d" % (kind[0].upper(), lv, s)
                    for kk in keys:
                        if per_unit[k][kk] != per_unit2[k][kk]:
                            repro = 0

    out = {
        "artifact": "light_shadow_gate_test",
        "doc_ref": "docs/261",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_ctrl_fp_drop": bool(c1), "c2_pos_detect_keep": bool(c2),
                     "c3_light_dir_keep": bool(c3), "fails": fails},
        "guards": {"cell1": g_cell1, "cell1_n": g_cell1_n, "cell1_bad": g_cell1_bad,
                   "cp5": g_cp5, "cp5_sc2": g_cp5_sc2, "cp5_sc_late": g_cp5_scl,
                   "mad": g_mad, "mad_sig": g_mad_sig, "mad_expected": g_mad_exp,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "gs_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签 + 每行一个数字（顺序固定）----
    print("R_GS_LEVELS=%d" % len(levels))
    print("R_GS_SEEDS=%d" % len(seeds))
    for lv in levels:
        lm = [u for u in units_main if u["lvcode"] == lv]
        lc = [u for u in units_ctrl if u["lvcode"] == lv]
        if lm:
            print("R_GS_L%d_DET_LEG=%.4f" % (lv, float(np.mean([u["det_legacy"] for u in lm]))))
            print("R_GS_L%d_DET_GAT=%.4f" % (lv, float(np.mean([u["det_gated"] for u in lm]))))
            print("R_GS_L%d_LD_MED_GAT=%.4f" % (lv, float(np.nanmean([u["ld_med_gated"] for u in lm]))))
            print("R_GS_L%d_SFS=%.4f" % (lv, float(np.nanmean([u["sfs_err"] for u in lm]))))
        if lc:
            print("R_GS_C%d_FP_LEG=%.4f" % (lv, float(np.mean([u["fp_legacy"] for u in lc]))))
            print("R_GS_C%d_FP_GAT=%.4f" % (lv, float(np.mean([u["fp_gated"] for u in lc]))))
    print("R_GS_FP_LEG_CTRL=%.4f" % fp_leg_ctrl)
    print("R_GS_FP_LEG_CTRL_SD=%.4f" % fp_leg_ctrl_sd)
    print("R_GS_FP_GAT_CTRL=%.4f" % fp_gat_ctrl)
    print("R_GS_FP_GAT_CTRL_SD=%.4f" % fp_gat_ctrl_sd)
    print("R_GS_FP_GAT_CTRL_CI_LO=%.4f" % fp_gat_ctrl_ci[0])
    print("R_GS_FP_GAT_CTRL_CI_HI=%.4f" % fp_gat_ctrl_ci[1])
    print("R_GS_FP_DROP=%.4f" % fp_drop)
    print("R_GS_DET_LEG_MAIN=%.4f" % det_leg_main)
    print("R_GS_DET_GAT_MAIN=%.4f" % det_gat_main)
    print("R_GS_DET_GAT_MAIN_SD=%.4f" % det_gat_main_sd)
    print("R_GS_DET_GAT_MAIN_CI_LO=%.4f" % det_gat_main_ci[0])
    print("R_GS_DET_GAT_MAIN_CI_HI=%.4f" % det_gat_main_ci[1])
    print("R_GS_DET_DROP=%.4f" % det_drop)
    print("R_GS_LD_MED_GAT_MAIN=%.4f" % ld_gat_main)
    print("R_GS_LD_MED_GAT_MAIN_SD=%.4f" % ld_gat_main_sd)
    print("R_GS_LD_ACC_GAT_MAIN=%.4f" % ld_acc_gat_main)
    print("R_GS_SFS_MAIN=%.4f" % sfs_main)
    print("R_GS_SFS_CTRL=%.4f" % sfs_ctrl)
    print("R_GS_VETO_RATE_CTRL=%.4f" % veto_ctrl)
    print("R_GS_V1_RATE_CTRL=%.4f" % v1_ctrl)
    print("R_GS_V3_RATE_CTRL=%.4f" % v3_ctrl)
    print("R_GS_FALSE_VETO_MAIN=%.4f" % fv_main)
    print("R_GS_E1_MAIN=%.4f" % e1_main)
    print("R_GS_E2_MAIN=%.4f" % e2_main)
    print("R_GS_C1_CTRL_FP_DROP=%s" % ("PASS" if c1 else "FAIL"))
    print("R_GS_C2_POS_KEEP=%s" % ("PASS" if c2 else "FAIL"))
    print("R_GS_C3_LIGHT_KEEP=%s" % ("PASS" if c3 else "FAIL"))
    print("R_GS_GUARD_CELL1=%d" % g_cell1)
    print("R_GS_GUARD_CELL1_N=%d" % g_cell1_n)
    print("R_GS_GUARD_CELL1_BAD=%d" % g_cell1_bad)
    print("R_GS_GUARD_CP5=%d" % g_cp5)
    print("R_GS_GUARD_CP5_SC2=%.4f" % g_cp5_sc2)
    print("R_GS_GUARD_MAD=%d" % g_mad)
    print("R_GS_GUARD_MAD_SIG=%.4f" % g_mad_sig)
    print("R_GS_REPRO=%d" % repro)
    print("R_GS_VERDICT=%s" % verdict)
    print("R_GS_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
