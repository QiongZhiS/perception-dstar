"""vision/light_shadow_real_test.py — 光影判别第四格：真实域三分类诊断 + 演示接入共享
（docs/265 预注册设计，判据/旋钮/守卫冻结；本脚本为唯一新增文件，import 复用第一格
vision/light_shadow_test.py、第二格 vision/light_shadow_gate_test.py、第三格
vision/light_shadow_reflect_test.py、vision/davis_suspicious.py、vision/real_stream_test.py、
vision/critical_point.py，未修改任何既有脚本；demo2_app.py 从本脚本 import 共享帧判别
函数 compute_ref_dark / diagnose_frame）。

目标（docs/265 §1.1）：把合成三连的判别机制（Pass A 适应 + 判暗 + 时间门 + V1/V3/V4
否决门链 + E1/E2/E3 证据门）原样（零重调、import 复用、旋钮全继承）迁移到 DAVIS 真实
流（9 视频 × 588 帧，160×120 工作点，docs/243 同款预处理），做"物体 / 纹理 / 阴影"
三分类**诊断级**实验——静态暗区 = 纹理/静态，时间门过 + 否决 = 物体，时间门过 +
无否决 = 阴影。诚实定位：真实视频无光照 GT、无阴影 GT → 本格是行为诊断，非真实域
证明（docs/265 §五 1）。

判据（docs/265 §1.4，冻结）：
  C1 OBJ_NOT_VETOED [L3][机制][真实域诊断]：锚帧（时间门过的候选且 |候选∩GT|/|候选|
     ≥ 0.30）中 label=物体 pooled 占比 ≥ 0.70 且总锚帧 ≥ 20（否决门链不把 GT 目标
     误判为阴影/纹理）
  C2 CLASS_ACTIVE    [L3][机制][真实域诊断]：报告性——候选帧占比、物体/纹理/阴影标签
     帧占比、全帧三类像素占比、候选面积均值、V1/V3/V4 触发率、E1/E2/E3 证据率、
     ΔH/ΔS 均值（无 GT 不设阈值）
  C3 KEEP            [L3][机制][真实域诊断]：报告性——θ_est（帧内估计光照方向，
     docs/260 C2 口径）分布 + V3/E2 适用率；与演示现有读数（MAE/ratio/SC2）不冲突
     由构造保证（面板不进机制决策，demo2_app 机制回路零改动）
  判定（docs/265 §1.5，冻结）：守卫全过 且 C1 过 = REAL_CLASS_DIAG；C1 不过 =
  OBJ_VETOED_FAIL；总锚帧 < 20 = REAL_CLASS_DIAG_ANCHOR_LOW；守卫不过 = GUARD_FAIL。

守卫（docs/265 §1.6，冻结）：
  R_RL_GUARD_SYNTH：import 第三格 run_unit_reflect 重跑 1 合成单位（30,0,"main"），
     det_gated == 1.0 且 cont_rate == 1.0（import 复用完好）
  R_RL_GUARD_CP5：import 第一格 guard_cp5（复现 docs/232 L5：SC2 ∈ [2.0,3.6]、
     SC_late ≥ 0.5）
  R_RL_GUARD_DEMO：diagnose_frame 同输入两次调用输出全等（共享函数确定性，
     演示/诊断同源）
  R_RL_REPRO：--repro 时 9 视频整体重跑第二遍，逐项位级一致

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_RL_* 摘要块
（顺序固定，见 SUMMARY_LINES 注释）；JSON 归档 vision/out/results/rl_<tag>.json +
checkpoint ckpt_rl_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯正则
抽取；禁止读取 logs/*.log 与 vision/out/results/*.json 原文；DAVIS JPEG/PNG 是数据。

用法：
  python vision/light_shadow_real_test.py --tag main --repro
  python vision/light_shadow_real_test.py --videos flamingo --tag timing
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

from critical_point import mean_sd, bootstrap_ci  # noqa: F401  （统计外壳）
from light_shadow_test import (  # 第一格：检测路径/旋钮/守卫，逐字 import 复用
    largest_component, iou, guard_cp5, guard_mad,
    W, H, DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, OCC_LUM_THRESH,
)
from light_shadow_gate_test import (  # 第二格：否决门几何量，逐字 import 复用
    touches_boundary, pca_axis, axis_err_deg,
    TOL_AXIS, RATIO_MIN,
)
from light_shadow_reflect_test import (  # 第三格：反射率判别 + 合成单位（守卫用）
    run_unit_reflect, reflect_stats,
    TOL_H, TOL_S, BAND_EDGE, SAT_MIN,
)
from davis_suspicious import circ, load_video  # noqa: F401  （circ 供 reflect_stats 内部）
from real_stream_test import VIDEOS, RESIZE

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 判据口径参数（docs/265 §1.4 冻结；非机制旋钮，先于运行冻结）----
ANCHOR_OVERLAP = 0.30    # 锚帧：候选 ≥30% 落在 GT 目标上
OBJ_NOT_VETOED_MIN = 0.70  # 锚帧中 label=物体 的 pooled 占比下限
ANCHOR_MIN_FRAMES = 20   # 锚帧最小样本（<20 判据未被行使）

# ---- 三分类标签（docs/265 §1.3 冻结语义）----
LABEL_NONE = "none"
LABEL_TEXTURE = "texture"   # 纹理/静态（候选但时间门不过 = 静止暗区）
LABEL_OBJECT = "object"     # 物体（时间门过 且 否决门链触发）
LABEL_SHADOW = "shadow"     # 阴影（时间门过 且 无否决）

# ---- 内部确定性复现键（docs/265 §1.6-4；每视频标量）----
REPRO_KEYS = ["obj_rate", "shadow_rate", "tex_rate", "v1_rate", "v3_rate", "v4_rate",
              "e1_rate", "e2_rate", "e3_rate", "theta_med", "anchor_n",
              "anchor_obj_rate", "area_mean", "pix_obj", "pix_shadow", "pix_tex"]


# ---------------- Pass A：适应（docs/260 诊断轮 D2 冻结口径，逐字同实现） ----------------
def compute_ref_dark(gray_frames):
    """全程帧每像素 0.95 分位亮度（排除 L > log(221) 的亮帧）→ ref_dark_log。

    gray_frames: list[uint8 (H,W)]（160×120 工作点，docs/243 同款预处理）。
    返回 ref_dark_log（float32 (H,W)，log(ref_dark+1)）。"""
    occ_thresh_log = np.log(OCC_LUM_THRESH + 1.0)
    n = len(gray_frames)
    if n == 0:
        return np.zeros((H, W), np.float32)
    buf = np.empty((n, H, W), np.float32)
    for t, g in enumerate(gray_frames):
        buf[t] = g.astype(np.float32)
    buf_masked = buf.copy()
    for t, g in enumerate(gray_frames):
        L = np.log(np.maximum(g.astype(np.float32), 1.0))
        buf_masked[t][L > occ_thresh_log] = np.nan
    ref_dark = np.nanpercentile(buf_masked, 95.0, axis=0)
    ref_dark = np.nan_to_num(ref_dark, nan=0.0)
    return np.log(np.maximum(ref_dark, 1.0))


# ---------------- 单帧判别（演示与诊断共用同一函数；docs/265 §1.3 冻结） ----------------
def diagnose_frame(gray, rgb_rgb, ref_dark_log, dark_prev):
    """真实域单帧三分类判别。

    gray: uint8 (H,W) 灰度帧（160×120）；rgb_rgb: uint8 (H,W,3) RGB 序彩色帧 or None
    （None = 无彩色通道 → V4/E3 不适用）；ref_dark_log: Pass A 产物；dark_prev:
    上一帧暗候选掩码 or None（t<K_MOVE 时无历史，时间门按冻结语义放宽为面积门）。

    返回 dict：
      label   'none'|'texture'|'object'|'shadow'
      mask    候选暗域掩码（bool (H,W)，label=none 时全 False）
      area    候选面积（px）
      active  时间门通过（= 否决链行使）
      move    帧间掩码变化 1−IoU(t,t−K)（无历史时 None）
      v1/v3/v4  否决门触发（bool）
      e1/e2/e3  证据门（bool 或 None=不适用：e2 各向同性/无 θ_est 时不适用，
                e3 无彩色/采样不足时不适用）
      theta_est  帧内估计光照方向（度）or None（无亮候选）
      dh/ds      边界两侧 ΔH/ΔS（反射率判别；无彩色/采样不足时 None）
    """
    L = np.log(np.maximum(gray.astype(np.float32), 1.0))
    dark = (ref_dark_log - L) > DELTA_SHADOW
    d_mask, d_area, d_c = largest_component(dark)
    bright = L > np.log(OCC_LUM_THRESH + 1.0)
    _b_mask, _b_area, b_c = largest_component(bright)

    out = dict(label=LABEL_NONE, mask=d_mask, area=int(d_area), active=False,
               move=None, v1=False, v3=False, v4=False,
               e1=False, e2=None, e3=None, theta_est=None, dh=None, ds=None)
    if d_area < A_MIN:
        return out

    # ---- 时间门（docs/260 C1 冻结口径；t<K_MOVE 无历史 → 面积门单独生效）----
    move = None
    if dark_prev is not None:
        move = 1.0 - iou(d_mask, dark_prev)
    out["move"] = move
    active = (d_area >= A_MIN) and (move is None or move >= MOVE_IOU)
    out["active"] = active
    if not active:
        out["label"] = LABEL_TEXTURE          # 静态暗区 = 纹理/静态（docs/187）
        return out

    # ---- 否决门链（docs/261 V1/V3 + docs/263 V4 逐字 import）----
    v1 = not touches_boundary(d_mask)         # V1 闭合轮廓否决
    out["v1"] = v1
    out["e1"] = not v1                        # E1 开放拓扑

    # θ_est：帧内估计光照方向（docs/260 C2 口径；真实域无光照 GT）
    if d_c is not None and b_c is not None:
        ex, ey = b_c[0] - d_c[0], b_c[1] - d_c[1]
        out["theta_est"] = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0

    v3 = False
    if out["theta_est"] is not None:
        pr = pca_axis(d_mask)
        if pr is not None:
            alpha, ratio = pr
            if ratio >= RATIO_MIN:            # 各向同性 → V3/E2 不适用（跳过）
                v3 = axis_err_deg(alpha, out["theta_est"]) > TOL_AXIS
                out["e2"] = not v3            # E2 主轴沿光照
    out["v3"] = v3

    v4 = False
    if rgb_rgb is not None:
        rs = reflect_stats(rgb_rgb, d_mask)   # 第三格逐字：BAND_EDGE/SAT_MIN 边界环带
        if rs is not None:
            out["dh"], out["ds"] = rs["dh"], rs["ds"]
            e3 = (rs["dh"] <= TOL_H) and (rs["ds"] <= TOL_S)
            out["e3"] = e3                    # E3 反射率连续
            v4 = not e3                       # V4 反射率跳变否决
    out["v4"] = v4

    out["label"] = LABEL_OBJECT if (v1 or v3 or v4) else LABEL_SHADOW
    return out


# ---------------- 每视频运行（docs/265 §1.2/§1.3 冻结） ----------------
def circular_median(angles):
    """圆形中位数（argmin 圆形距离和）。angles: list[float 度]。空 → nan。"""
    if not angles:
        return float("nan")
    a = np.asarray(angles, np.float64)
    best, best_m = None, None
    for m in a:
        d = np.abs((a - m) % 360.0)
        d = np.minimum(d, 360.0 - d)
        s = float(d.sum())
        if best_m is None or s < best_m:
            best_m, best = s, m
    return float(best % 360.0)


def run_video(video):
    """跑单个 DAVIS 视频：Pass A → 逐帧判别 → 聚合。返回 per-video dict。

    GT 掩码（PNG）只用于评估（OBJ_NOT_VETOED 锚），绝不进入机制。"""
    cfr, masks = load_video(video)                      # BGR 854×480 + GT 掩码
    n = len(cfr)
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), RESIZE,
                       interpolation=cv2.INTER_AREA) for f in cfr]
    rgb = [cv2.cvtColor(cv2.resize(f, RESIZE, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB) for f in cfr]     # RGB 序（reflect_stats 契约）
    gt = [cv2.resize(m, RESIZE, interpolation=cv2.INTER_NEAREST) > 0 for m in masks]
    ref_dark_log = compute_ref_dark(gray)

    dark_prev = None
    label_counts = {LABEL_OBJECT: 0, LABEL_SHADOW: 0, LABEL_TEXTURE: 0}
    pix_counts = {LABEL_OBJECT: 0.0, LABEL_SHADOW: 0.0, LABEL_TEXTURE: 0.0}
    frame_detail = []                       # 逐帧判据明细（紧凑）
    cand_frames = 0
    area_sum, area_n = 0.0, 0
    gate_counts = {"v1": 0, "v3": 0, "v4": 0, "e1": 0, "e2": 0, "e3": 0}
    gate_n = 0
    theta_vals, v3_applic = [], 0
    dh_vals, ds_vals = [], []
    anchor_obj, anchor_n = 0, 0
    ov_obj, ov_shadow, ov_tex, ov_n = 0, 0, 0, 0    # 全部重叠帧（报告性：误判为阴影率）

    for t in range(n):
        d = diagnose_frame(gray[t], rgb[t], ref_dark_log, dark_prev)
        dark_prev = d["mask"]
        lab = d["label"]
        if lab != LABEL_NONE:
            cand_frames += 1
            label_counts[lab] += 1
            pix_counts[lab] += float(d["mask"].sum()) / float(H * W)
            area_sum += d["area"]
            area_n += 1
        if lab in (LABEL_OBJECT, LABEL_SHADOW):      # 时间门过 → 否决链行使
            gate_n += 1
            if d["v1"]:
                gate_counts["v1"] += 1
            if d["v3"]:
                gate_counts["v3"] += 1
            if d["v4"]:
                gate_counts["v4"] += 1
            if d["e1"]:
                gate_counts["e1"] += 1
            if d["e2"] is True:
                gate_counts["e2"] += 1
            if d["e3"] is True:
                gate_counts["e3"] += 1
            if d["theta_est"] is not None:
                theta_vals.append(d["theta_est"])
                v3_applic += 1
            if d["dh"] is not None:
                dh_vals.append(d["dh"])
            if d["ds"] is not None:
                ds_vals.append(d["ds"])
        # GT 锚（docs/265 §1.4 C1）
        if lab != LABEL_NONE and gt[t].any():
            ov = float(np.logical_and(d["mask"], gt[t]).sum()) / max(1.0, float(d["mask"].sum()))
            if ov >= ANCHOR_OVERLAP:
                ov_n += 1
                if lab == LABEL_OBJECT:
                    ov_obj += 1
                elif lab == LABEL_SHADOW:
                    ov_shadow += 1
                else:
                    ov_tex += 1
                if d["active"]:                     # 锚帧 = 时间门过的候选
                    anchor_n += 1
                    if lab == LABEL_OBJECT:
                        anchor_obj += 1
        frame_detail.append([t, lab, d["area"], int(d["v1"]), int(d["v3"]),
                             int(d["v4"]), int(d["e1"]),
                             -1 if d["e2"] is None else int(d["e2"]),
                             -1 if d["e3"] is None else int(d["e3"]),
                             -1.0 if d["theta_est"] is None else round(d["theta_est"], 3)])

    n_cand = max(1, cand_frames)
    return dict(video=video, frames=n, cand_frames=cand_frames,
                obj_f=label_counts[LABEL_OBJECT], shadow_f=label_counts[LABEL_SHADOW],
                tex_f=label_counts[LABEL_TEXTURE], none_f=n - cand_frames,
                obj_rate=float(label_counts[LABEL_OBJECT]) / n_cand,
                shadow_rate=float(label_counts[LABEL_SHADOW]) / n_cand,
                tex_rate=float(label_counts[LABEL_TEXTURE]) / n_cand,
                pix_obj=float(pix_counts[LABEL_OBJECT]) / n,
                pix_shadow=float(pix_counts[LABEL_SHADOW]) / n,
                pix_tex=float(pix_counts[LABEL_TEXTURE]) / n,
                area_mean=float(area_sum / max(1, area_n)) if area_n else float("nan"),
                v1_rate=float(gate_counts["v1"]) / max(1, gate_n),
                v3_rate=float(gate_counts["v3"]) / max(1, gate_n),
                v4_rate=float(gate_counts["v4"]) / max(1, gate_n),
                e1_rate=float(gate_counts["e1"]) / max(1, gate_n),
                e2_rate=float(gate_counts["e2"]) / max(1, gate_n),
                e3_rate=float(gate_counts["e3"]) / max(1, gate_n),
                gate_n=gate_n,
                theta_med=circular_median(theta_vals),
                theta_n=len(theta_vals), v3_applic=v3_applic,
                dh_mean=float(np.mean(dh_vals)) if dh_vals else float("nan"),
                ds_mean=float(np.mean(ds_vals)) if ds_vals else float("nan"),
                anchor_n=anchor_n,
                anchor_obj_rate=float(anchor_obj) / max(1, anchor_n),
                ov_n=ov_n, ov_obj=ov_obj, ov_shadow=ov_shadow, ov_tex=ov_tex,
                frame_detail=frame_detail)


# ---------------- 守卫（docs/265 §1.6 冻结） ----------------
def guard_synth():
    """import 复用证明：第三格 run_unit_reflect(30, 0, 'main') 重跑 1 合成单位，
    冻结数字保持（det_gated=1.0 且 cont_rate=1.0）= import 链与三连逐字同源。"""
    u = run_unit_reflect(30, 0, "main")
    det_ok = int(u["det_gated"] == 1.0)
    cont_ok = int(u["cont_rate"] == 1.0)
    return 1 if (det_ok and cont_ok) else 0, u["det_gated"], u["cont_rate"]


def guard_demo(gray, rgb_rgb, ref_dark_log, dark_prev):
    """共享帧函数确定性：diagnose_frame 同输入两次调用输出全等。"""
    a = diagnose_frame(gray, rgb_rgb, ref_dark_log, dark_prev)
    b = diagnose_frame(gray, rgb_rgb, ref_dark_log, dark_prev)
    keys = ("label", "v1", "v3", "v4", "e1", "e2", "e3", "theta_est", "dh", "ds", "area")
    ok = all(a[k] == b[k] for k in keys)
    return 1 if ok else 0, a["label"]


# ---------------- 聚合（docs/265 §1.7 冻结：单位=每视频；OBJ_NOT_VETOED pooled） ----------------
def unit_mean(vals):
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="光影判别第四格：真实域三分类诊断")
    ap.add_argument("--videos", default=",".join(VIDEOS))
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = [v for v in args.videos.split(",") if v.strip() != ""]
    t0 = time.time()

    cfg = {"videos": videos, "tag": args.tag,
           "mechanism": {"delta_shadow": DELTA_SHADOW, "a_min": A_MIN,
                         "k_move": K_MOVE, "move_iou": MOVE_IOU,
                         "occ_lum_thresh": OCC_LUM_THRESH, "ref_pct": 95.0,
                         "veto": {"tol_axis": TOL_AXIS, "ratio_min": RATIO_MIN,
                                  "tol_h": TOL_H, "tol_s": TOL_S,
                                  "band_edge": BAND_EDGE, "sat_min": SAT_MIN}},
           "criteria": {"anchor_overlap": ANCHOR_OVERLAP,
                        "obj_not_vetoed_min": OBJ_NOT_VETOED_MIN,
                        "anchor_min_frames": ANCHOR_MIN_FRAMES}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_rl_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_video", {})
        per_video = dict(done)
        for v in videos:
            if v in per_video:
                continue
            per_video[v] = run_video(v)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_video": per_video},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)
        return per_video

    per_video = run_all()
    units = [per_video[v] for v in videos]

    # ---- C1 OBJ_NOT_VETOED（pooled）----
    anchor_total = int(sum(u["anchor_n"] for u in units))
    anchor_obj = int(sum(u["anchor_obj_rate"] * u["anchor_n"] for u in units))
    pooled_obj = float(anchor_obj) / max(1, anchor_total)
    anchor_cov = anchor_total / max(1, sum(u["frames"] for u in units))
    c1 = (pooled_obj >= OBJ_NOT_VETOED_MIN) and (anchor_total >= ANCHOR_MIN_FRAMES)

    # ---- C2 CLASS_ACTIVE（报告性；单位级 mean±SD + bootstrap）----
    cand_rate_pooled = sum(u["cand_frames"] for u in units) / max(1, sum(u["frames"] for u in units))
    obj_rate_all = unit_mean([u["obj_rate"] for u in units])
    shadow_rate_all = unit_mean([u["shadow_rate"] for u in units])
    tex_rate_all = unit_mean([u["tex_rate"] for u in units])
    v1_all, v3_all, v4_all = (unit_mean([u[k] for u in units if u["gate_n"] > 0])
                              for k in ("v1_rate", "v3_rate", "v4_rate"))
    e1_all, e2_all, e3_all = (unit_mean([u[k] for u in units if u["gate_n"] > 0])
                              for k in ("e1_rate", "e2_rate", "e3_rate"))
    pix_obj_m, pix_obj_sd = mean_sd([u["pix_obj"] for u in units])
    pix_shadow_m, pix_shadow_sd = mean_sd([u["pix_shadow"] for u in units])
    pix_tex_m, pix_tex_sd = mean_sd([u["pix_tex"] for u in units])
    pix_obj_ci = bootstrap_ci([u["pix_obj"] for u in units])
    pix_shadow_ci = bootstrap_ci([u["pix_shadow"] for u in units])
    pix_tex_ci = bootstrap_ci([u["pix_tex"] for u in units])
    area_m, area_sd = mean_sd([u["area_mean"] for u in units if u["area_mean"] == u["area_mean"]])

    # ---- C3 KEEP（报告性：θ_est 分布 + V3 适用率）----
    all_theta = [x for u in units for x in [u["theta_med"]] if x == x]
    theta_med_all = circular_median(all_theta)
    theta_n_all = sum(u["theta_n"] for u in units)
    v3_applic_all = sum(u["v3_applic"] for u in units)
    gate_n_all = sum(u["gate_n"] for u in units)
    v3_applic_rate = v3_applic_all / max(1, gate_n_all)
    dh_pool = [u["dh_mean"] for u in units if u["dh_mean"] == u["dh_mean"]]
    ds_pool = [u["ds_mean"] for u in units if u["ds_mean"] == u["ds_mean"]]
    dh_all = unit_mean(dh_pool)
    ds_all = unit_mean(ds_pool)

    # ---- 守卫 ----
    g_synth, g_synth_det, g_synth_cont = guard_synth()
    g_cp5, g_cp5_sc2, g_cp5_scl = guard_cp5(n_seeds=10, n_frames=240)
    g_mad, _sig, _exp = guard_mad()
    # GUARD_DEMO：flamingo 帧 0（灰度 + 彩色 + ref）
    _demo_fr, _demo_mk = load_video("flamingo")
    _g0 = cv2.resize(cv2.cvtColor(_demo_fr[0], cv2.COLOR_BGR2GRAY), RESIZE,
                     interpolation=cv2.INTER_AREA)
    _r0 = cv2.cvtColor(cv2.resize(_demo_fr[0], RESIZE, interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    _ref0 = compute_ref_dark([_g0])
    g_demo, g_demo_label = guard_demo(_g0, _r0, _ref0, None)
    guards_ok = (g_synth == 1) and (g_cp5 == 1) and (g_mad == 1) and (g_demo == 1)

    # ---- 判定（docs/265 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif anchor_total < ANCHOR_MIN_FRAMES:
        verdict = "REAL_CLASS_DIAG_ANCHOR_LOW"
    elif pooled_obj >= OBJ_NOT_VETOED_MIN:
        verdict = "REAL_CLASS_DIAG"
    else:
        verdict = "OBJ_VETOED_FAIL"

    # ---- 内部确定性复现（docs/265 §1.6-4；第二遍强制重算，不读 checkpoint）----
    # NaN 感知比较：theta_med 等可为 NaN（surf 无亮候选），nan != nan 恒真——按
    # "双 NaN 相等"处理（机械修复 D2，§二 记录；判据/机制未动）。
    repro = 1
    if args.repro:
        per2 = run_all(use_resume=False)
        for v in videos:
            for k in REPRO_KEYS:
                a, b = per_video[v][k], per2[v][k]
                same = (a == b) or (a != a and b != b)
                if not same:
                    repro = 0

    agg = {"anchor_total": anchor_total, "anchor_obj": anchor_obj,
           "pooled_obj_rate": pooled_obj, "anchor_cov": anchor_cov,
           "cand_frame_rate": cand_rate_pooled,
           "obj_rate_all": obj_rate_all, "shadow_rate_all": shadow_rate_all,
           "tex_rate_all": tex_rate_all,
           "v1_all": v1_all, "v3_all": v3_all, "v4_all": v4_all,
           "e1_all": e1_all, "e2_all": e2_all, "e3_all": e3_all,
           "pix_obj": [pix_obj_m, pix_obj_sd, list(pix_obj_ci)],
           "pix_shadow": [pix_shadow_m, pix_shadow_sd, list(pix_shadow_ci)],
           "pix_tex": [pix_tex_m, pix_tex_sd, list(pix_tex_ci)],
           "area_mean": [area_m, area_sd],
           "theta_med_all": theta_med_all, "theta_n_all": theta_n_all,
           "v3_applic": v3_applic_all, "v3_applic_rate": v3_applic_rate,
           "gate_n_all": gate_n_all, "dh_all": dh_all, "ds_all": ds_all}

    out = {
        "artifact": "light_shadow_real_test",
        "doc_ref": "docs/265",
        "config": cfg,
        "per_video": per_video,
        "aggregate": agg,
        "criteria": {"c1_obj_not_vetoed": bool(c1),
                     "pooled_obj_rate": pooled_obj,
                     "anchor_total": anchor_total,
                     "anchor_cov": anchor_cov},
        "guards": {"synth": g_synth, "synth_det": g_synth_det,
                   "synth_cont": g_synth_cont, "cp5": g_cp5,
                   "cp5_sc2": g_cp5_sc2, "cp5_sc_late": g_cp5_scl,
                   "mad": g_mad, "demo": g_demo, "demo_label": g_demo_label,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "rl_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定，docs/265 §1.8）----
    def vid_tag(v):
        return v.replace("-", "_").upper()

    print("R_RL_VIDEOS=%d" % len(videos))
    for u in units:
        vt = vid_tag(u["video"])
        print("R_RL_%s_FRAMES=%d" % (vt, u["frames"]))
        print("R_RL_%s_CAND_FRAMES=%d" % (vt, u["cand_frames"]))
        print("R_RL_%s_OBJ_F=%d" % (vt, u["obj_f"]))
        print("R_RL_%s_SHADOW_F=%d" % (vt, u["shadow_f"]))
        print("R_RL_%s_TEX_F=%d" % (vt, u["tex_f"]))
        print("R_RL_%s_NONE_F=%d" % (vt, u["none_f"]))
        print("R_RL_%s_OBJ_RATE=%.4f" % (vt, u["obj_rate"]))
        print("R_RL_%s_SHADOW_RATE=%.4f" % (vt, u["shadow_rate"]))
        print("R_RL_%s_TEX_RATE=%.4f" % (vt, u["tex_rate"]))
        print("R_RL_%s_V1=%.4f" % (vt, u["v1_rate"]))
        print("R_RL_%s_V3=%.4f" % (vt, u["v3_rate"]))
        print("R_RL_%s_V4=%.4f" % (vt, u["v4_rate"]))
        print("R_RL_%s_E1=%.4f" % (vt, u["e1_rate"]))
        print("R_RL_%s_E2=%.4f" % (vt, u["e2_rate"]))
        print("R_RL_%s_E3=%.4f" % (vt, u["e3_rate"]))
        print("R_RL_%s_THETA_MED=%.2f" % (vt, u["theta_med"]))
        print("R_RL_%s_THETA_N=%d" % (vt, u["theta_n"]))
        print("R_RL_%s_V3_APPLIC=%d" % (vt, u["v3_applic"]))
        print("R_RL_%s_ANCHOR_N=%d" % (vt, u["anchor_n"]))
        print("R_RL_%s_ANCHOR_OBJ_RATE=%.4f" % (vt, u["anchor_obj_rate"]))
        print("R_RL_%s_AREA_MEAN=%.1f" % (vt, u["area_mean"]))
        print("R_RL_%s_PIX_OBJ=%.4f" % (vt, u["pix_obj"]))
        print("R_RL_%s_PIX_SHADOW=%.4f" % (vt, u["pix_shadow"]))
        print("R_RL_%s_PIX_TEX=%.4f" % (vt, u["pix_tex"]))
    print("R_RL_OBJ_NOT_VETOED=%.4f" % pooled_obj)
    print("R_RL_ANCHOR_TOTAL=%d" % anchor_total)
    print("R_RL_ANCHOR_COV=%.4f" % anchor_cov)
    print("R_RL_CAND_FRAME_RATE=%.4f" % cand_rate_pooled)
    print("R_RL_OBJ_RATE_ALL=%.4f" % obj_rate_all)
    print("R_RL_SHADOW_RATE_ALL=%.4f" % shadow_rate_all)
    print("R_RL_TEX_RATE_ALL=%.4f" % tex_rate_all)
    print("R_RL_V1_ALL=%.4f" % v1_all)
    print("R_RL_V3_ALL=%.4f" % v3_all)
    print("R_RL_V4_ALL=%.4f" % v4_all)
    print("R_RL_E1_ALL=%.4f" % e1_all)
    print("R_RL_E2_ALL=%.4f" % e2_all)
    print("R_RL_E3_ALL=%.4f" % e3_all)
    print("R_RL_THETA_MED_ALL=%.2f" % theta_med_all)
    print("R_RL_THETA_N_ALL=%d" % theta_n_all)
    print("R_RL_V3_APPLIC_ALL=%d" % v3_applic_all)
    print("R_RL_V3_APPLIC_RATE=%.4f" % v3_applic_rate)
    print("R_RL_PIX_OBJ_MEAN=%.4f" % pix_obj_m)
    print("R_RL_PIX_SHADOW_MEAN=%.4f" % pix_shadow_m)
    print("R_RL_PIX_TEX_MEAN=%.4f" % pix_tex_m)
    print("R_RL_DH_ALL=%.2f" % dh_all)
    print("R_RL_DS_ALL=%.2f" % ds_all)
    print("R_RL_GUARD_SYNTH=%d" % g_synth)
    print("R_RL_GUARD_SYNTH_DET=%.4f" % g_synth_det)
    print("R_RL_GUARD_SYNTH_CONT=%.4f" % g_synth_cont)
    print("R_RL_GUARD_CP5=%d" % g_cp5)
    print("R_RL_GUARD_CP5_SC2=%.4f" % g_cp5_sc2)
    print("R_RL_GUARD_DEMO=%d" % g_demo)
    print("R_RL_GUARD_DEMO_LABEL=%s" % g_demo_label)
    print("R_RL_REPRO=%d" % repro)
    print("R_RL_VERDICT=%s" % verdict)
    print("R_RL_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
