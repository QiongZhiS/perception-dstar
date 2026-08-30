"""vision/light_shadow_real_gt_test.py — 光影判别第五格：真实域证明（带阴影 GT 的 ISTD）
（docs/267 预注册设计，判据/旋钮/守卫冻结；本脚本为唯一新增实验文件，import 复用
第一格 vision/light_shadow_test.py、第二格 vision/light_shadow_gate_test.py、第三格
vision/light_shadow_reflect_test.py、第四格 vision/light_shadow_real_test.py、
vision/real_stream_test.py、vision/critical_point.py，未修改任何既有脚本）。

目标（docs/267 §1.1）：把合成三连 + 第四格的判别机制（Pass A 适应 + 判暗 + 时间门 +
V1/V3/V4 否决门链 + E1/E2/E3 证据门）原样（零重调、import 复用、旋钮全继承）跑在带
阴影 GT 的真实数据集 ISTD（1870 场景三元组：阴影图 A + 阴影掩码 C + 无阴影参考图 B，
hf-mirror 镜像 CK234/tmp_ISTD → D:\\datasets\\ISTD）上，对真实阴影标注量化标准检测
度量（pooled 逐像素 precision/recall/F1）——真实域证明（docs/265 §五 8 的下一步）。

静态图像进流式机制的冻结适配（docs/267 §1.3）：单位 = 每场景一次运行，2 帧
[B（无阴影参考帧）, A（阴影帧）]；Pass A 在 [B,A] 上算逐像素 0.95 分位（= 逐元素
max，B = 该场景"常见亮度"真参考）；评估帧 = A（有 GT 掩码 C），B 只作适应参考
（同视频 WARMUP 帧不进评估窗口先例，docs/260 §1.3）。

判据（docs/267 §1.4，冻结）：
  C1 REAL_SHADOW_PREC_REC [L3][机制][真实域证明]：pooled 逐像素 TP/FP/FN →
     precision/recall/F1（docs/219 同款公式）；行使门槛 = pooled GT 阴影像素 ≥ 50000；
     阈值 pooled P ≥ 0.30 且 R ≥ 0.35 且 F1 ≥ 0.40（三条件 AND）
  C2 WHITE_OBJ_RECALL    [L3][机制][真实域证明]：GT 阴影 ∩ 参考图 B 灰度 ≥ 128 的
     亮目标像素的 pooled 召回率 ≥ 0.30；行使门槛 = 亮目标像素 ≥ 2000 且 ≥ GT 1%
  C3 KEEP                [L3][机制][合成→真实保持]：守卫 R_GT_GUARD_SYNTH（第三格
     det=1.0/cont=1.0）+ R_GT_GUARD_CELL4（第四格 flamingo 逐位，docs/265 §3.1）
  判定（docs/267 §1.5，冻结）：守卫全过 且 C1 过 且（C2 未行使 或 过）=
  REAL_SHADOW_PASS；C1 不过 = REAL_SHADOW_FAIL（P_FAIL/R_FAIL/F1_FAIL）；C1 过但 C2
  行使不过 = REAL_SHADOW_FAIL（C2_FAIL）；GT 像素 < 50000 = REAL_SHADOW_LOW；守卫
  不过 = GUARD_FAIL。

守卫（docs/267 §1.6，冻结）：
  R_GT_GUARD_SYNTH：import 第三格 run_unit_reflect(30,0,"main")，det_gated==1.0 且
     cont_rate==1.0（三连数字保持）
  R_GT_GUARD_CELL4：import 第四格 run_video("flamingo")，obj_rate/shadow_rate/v3_rate/
     theta_med 与 docs/265 §3.1 显示位数一致（KEEP 机制载体）
  R_GT_GUARD_DET：diagnose_frame 同输入两次调用输出全等（共享函数确定性）
  R_GT_GUARD_MASK：固定种子抽 5 场景，GT 掩码与图同尺寸且非空、B 阴影区均值 > A
     （数据完整性冒烟，纯数据不进机制）
  R_GT_REPRO：--repro 时 1870 单位整体重跑第二遍（不读 checkpoint），逐项位级一致
     （NaN 感知比较）

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_GT_* 摘要
块（顺序固定，见 SUMMARY_LINES 注释）；JSON 归档 vision/out/results/rgt_<tag>.json +
checkpoint ckpt_rgt_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py 纯
正则抽取；禁止读取 logs/*.log 与 vision/out/results/*.json 原文；ISTD PNG 是数据。

用法：
  python vision/light_shadow_real_gt_test.py --tag main --repro
  python vision/light_shadow_real_gt_test.py --tag timing --limit 20
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
from light_shadow_real_test import (  # 第四格：判别核心 + 守卫，逐字 import 复用
    compute_ref_dark, diagnose_frame, circular_median, run_video,
    guard_synth, guard_demo,
    LABEL_NONE, LABEL_TEXTURE, LABEL_OBJECT, LABEL_SHADOW,
)
from light_shadow_test import (  # 第一格：旋钮/统计
    W, H, DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, OCC_LUM_THRESH,
)
from light_shadow_gate_test import (  # 第二格：否决门几何量
    touches_boundary, pca_axis, axis_err_deg,  # noqa: F401 （diagnose_frame 内部使用）
    TOL_AXIS, RATIO_MIN,  # noqa: F401
)
from light_shadow_reflect_test import (  # 第三格：反射率判别 + 合成单位（守卫用）
    run_unit_reflect,  # noqa: F401
    TOL_H, TOL_S, BAND_EDGE, SAT_MIN,  # noqa: F401
)
from real_stream_test import RESIZE

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 判据口径参数（docs/267 §1.4 冻结；非机制旋钮，先于运行冻结）----
P_MIN = 0.30            # C1 pooled precision 下限
R_MIN = 0.35            # C1 pooled recall 下限
F1_MIN = 0.40           # C1 pooled F1 下限
GT_PIX_MIN = 50000      # C1 行使门槛：pooled GT 阴影像素
L_WHITE = 128.0         # C2 亮目标定义：参考图 B 灰度 ≥ 128（uint8 中亮杠）
WHITE_MIN_PIX = 2000    # C2 行使门槛：pooled 亮目标像素
WHITE_MIN_FRAC = 0.01   # C2 行使门槛：亮目标 ≥ GT 阴影像素 1%

# ---- 内部确定性复现键（docs/267 §1.6-5；每单位标量）----
REPRO_KEYS = ["tp", "fp", "fn", "white_tp", "label", "v1", "v3", "v4",
              "e1", "e2", "e3", "active", "gt_sum", "pred_sum"]


# ---------------- 数据（ISTD；目录可读，PNG 是数据） ----------------
# 注意（机械修复 D1，§二 记录）：hf-mirror 镜像 CK234/tmp_ISTD 的目录映射为
# _A = 阴影图、_B = 阴影掩码、_C = 无阴影参考图（实测：B 单通道掩码、C 三通道且
# 阴影区均值 > A）——与部分文献的 A/B/C 惯例相反。语义不变：A = 阴影图（评估帧）、
# 掩码 = GT、无阴影参考 = Pass A"常见亮度"载体。docs/267 §一 描述的是语义角色。
def list_scenes(istd_root):
    """返回 [(split, a_path, mask_path, free_path), ...]，按 (split, basename) 排序确定性。"""
    scenes = []
    for split in ("train", "test"):
        dA = os.path.join(istd_root, split, split + "_A")
        dB = os.path.join(istd_root, split, split + "_B")
        dC = os.path.join(istd_root, split, split + "_C")
        for name in sorted(os.listdir(dA)):
            if not name.lower().endswith(".png"):
                continue
            a = os.path.join(dA, name)
            b = os.path.join(dB, name)
            c = os.path.join(dC, name)
            if os.path.exists(b) and os.path.exists(c):
                scenes.append((split, a, b, c))
    return scenes


def run_scene(split, a_path, mask_path, free_path):
    """跑单个 ISTD 场景（docs/267 §1.2/§1.3 冻结）：2 帧单位 [free, shadow]，评估阴影帧。

    返回 per-scene dict（紧凑标量，供聚合/复现）。GT 掩码只用于评估，绝不进入机制。"""
    b_bgr = cv2.imread(free_path, cv2.IMREAD_COLOR)      # free = 无阴影参考（_C）
    a_bgr = cv2.imread(a_path, cv2.IMREAD_COLOR)         # A = 阴影图（评估帧）
    c_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # mask = GT 掩码（_B）
    if b_bgr is None or a_bgr is None or c_gray is None:
        raise RuntimeError("cannot read scene: %s" % a_path)

    # ---- 预处理（docs/267 §1.2 冻结；docs/265 §1.2 同款）----
    b_gray = cv2.resize(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY), RESIZE,
                        interpolation=cv2.INTER_AREA)
    a_gray = cv2.resize(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY), RESIZE,
                        interpolation=cv2.INTER_AREA)
    b_rgb = cv2.cvtColor(cv2.resize(b_bgr, RESIZE, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB)
    a_rgb = cv2.cvtColor(cv2.resize(a_bgr, RESIZE, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB)
    gt = cv2.resize(c_gray, RESIZE, interpolation=cv2.INTER_NEAREST) > 0

    # ---- Pass A（每单位一次；[B,A] 2 帧，compute_ref_dark 逐字）----
    ref_dark_log = compute_ref_dark([b_gray, a_gray])

    # ---- B 帧只作适应参考：取暗候选作 A 帧时间门历史（不评估）----
    db = diagnose_frame(b_gray, b_rgb, ref_dark_log, None)

    # ---- A 帧 = 评估帧（有 GT 掩码 C）----
    da = diagnose_frame(a_gray, a_rgb, ref_dark_log, db["mask"])

    pred = da["mask"] if da["label"] == LABEL_SHADOW else np.zeros((H, W), bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())

    # ---- C2 亮目标（docs/267 §1.4：GT 阴影 ∩ B 帧灰度 ≥ L_WHITE）----
    white = gt & (b_gray >= L_WHITE)
    white_pix = int(white.sum())
    white_tp = int(np.logical_and(pred, white).sum())

    def bi(x):
        return 1 if x else 0

    return dict(id=os.path.basename(a_path), split=split,
                tp=tp, fp=fp, fn=fn, white_tp=white_tp, white_pix=white_pix,
                gt_sum=gt_sum, pred_sum=pred_sum,
                label=da["label"], active=bi(da["active"]),
                v1=bi(da["v1"]), v3=bi(da["v3"]), v4=bi(da["v4"]),
                e1=bi(da["e1"]), e2=(-1 if da["e2"] is None else bi(da["e2"])),
                e3=(-1 if da["e3"] is None else bi(da["e3"])),
                theta=(-1.0 if da["theta_est"] is None else round(float(da["theta_est"]), 3)),
                dh=(-1.0 if da["dh"] is None else round(float(da["dh"]), 3)),
                ds=(-1.0 if da["ds"] is None else round(float(da["ds"]), 3)))


# ---------------- 聚合（docs/267 §1.4 冻结：C1/C2 pooled，单位级报告性） ----------------
def pooled_prf(units):
    tp = int(sum(u["tp"] for u in units))
    fp = int(sum(u["fp"] for u in units))
    fn = int(sum(u["fn"] for u in units))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1)


def unit_level(units, tp_key, fp_key, fn_key):
    """单位级统计（报告性）：在分母 > 0 的单位上取 P（precision）或 R（recall）。
    mode 由 fp_key/fn_key 决定：fn_key 非 None 且 fp_key 为 None → recall；
    否则 precision。返回 (mean, sd, [lo, hi], n)。"""
    vals = []
    for u in units:
        if fn_key is not None:
            denom = u[tp_key] + u[fn_key]
        else:
            denom = u[tp_key] + u[fp_key]
        if denom > 0:
            v = u[tp_key] / denom
            if v == v:
                vals.append(v)
    if not vals:
        return float("nan"), 0.0, [float("nan"), float("nan")], 0
    m, s = mean_sd(vals)
    lo, hi = bootstrap_ci(vals)
    return m, s, [float(lo), float(hi)], len(vals)


# ---------------- 守卫（docs/267 §1.6 冻结） ----------------
def guard_cell4():
    """第四格保持：import run_video("flamingo") 重跑，与 docs/265 §3.1 显示位数一致。"""
    u = run_video("flamingo")
    ok = (abs(u["obj_rate"] - 0.7875) <= 5e-5 and abs(u["shadow_rate"] - 0.2125) <= 5e-5
          and abs(u["v3_rate"] - 0.7875) <= 5e-5
          and abs(u["theta_med"] - 287.38) <= 0.005)
    return 1 if ok else 0, u["obj_rate"], u["shadow_rate"], u["v3_rate"], u["theta_med"]


def guard_mask(istd_root, scenes, seed=BOOT_SEED, n=5):
    """数据完整性冒烟（纯数据，不进机制）：GT 掩码与图同尺寸且非空、free 阴影区均值
    > 阴影图阴影区均值（无阴影参考确实更亮——映射正确）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scenes), size=min(n, len(scenes)), replace=False)
    ok = 1
    for i in idx:
        _, a, mask, free = scenes[i]
        a_img = cv2.imread(a, cv2.IMREAD_COLOR)
        f_img = cv2.imread(free, cv2.IMREAD_COLOR)
        c_gray = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if (a_img is None or f_img is None or c_gray is None
                or a_img.shape[:2] != f_img.shape[:2]
                or a_img.shape[:2] != c_gray.shape[:2]):
            ok = 0
            continue
        gtm = c_gray > 0
        if gtm.sum() < 10:
            ok = 0
            continue
        fm = float(f_img[gtm].mean())
        am = float(a_img[gtm].mean())
        if not (fm > am):
            ok = 0
    return ok


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="光影判别第五格：真实域阴影 GT 证明（ISTD）")
    ap.add_argument("--istd-dir", default=r"D:\datasets\ISTD")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="rgt")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只跑前 N 个场景（冒烟/计时用，非冻结全量）")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scenes = list_scenes(args.istd_dir)
    if args.limit > 0:
        scenes = scenes[:args.limit]
    t0 = time.time()
    if not scenes:
        print("R_GT_ERROR_NO_SCENES=1")
        return 1

    cfg = {"istd_dir": args.istd_dir, "limit": args.limit, "tag": args.tag,
           "mechanism": {"delta_shadow": DELTA_SHADOW, "a_min": A_MIN,
                         "k_move": K_MOVE, "move_iou": MOVE_IOU,
                         "occ_lum_thresh": OCC_LUM_THRESH, "ref_pct": 95.0,
                         "veto": {"tol_axis": TOL_AXIS, "ratio_min": RATIO_MIN,
                                  "tol_h": TOL_H, "tol_s": TOL_S,
                                  "band_edge": BAND_EDGE, "sat_min": SAT_MIN}},
           "criteria": {"p_min": P_MIN, "r_min": R_MIN, "f1_min": F1_MIN,
                        "gt_pix_min": GT_PIX_MIN, "l_white": L_WHITE,
                        "white_min_pix": WHITE_MIN_PIX, "white_min_frac": WHITE_MIN_FRAC}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_rgt_%s.json" % ck_tag)

    def run_all(use_resume=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for i, (split, a, b, c) in enumerate(scenes):
            key = "%s_%s" % (split, os.path.basename(a))
            if key in per_unit:
                continue
            per_unit[key] = run_scene(split, a, b, c)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            if (i + 1) % 200 == 0:
                print("PROGRESS %d/%d" % (i + 1, len(scenes)), flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%s_%s" % (s, os.path.basename(a))] for s, a, b, c in scenes]

    # ---- C1 REAL_SHADOW_PREC_REC（pooled；docs/267 §1.4）----
    pool = pooled_prf(units)
    units_train = [u for u in units if u["split"] == "train"]
    units_test = [u for u in units if u["split"] == "test"]
    pool_tr = pooled_prf(units_train)
    pool_te = pooled_prf(units_test)
    gt_total = pool["tp"] + pool["fn"]
    c1_exercised = gt_total >= GT_PIX_MIN
    c1 = (pool["p"] >= P_MIN) and (pool["r"] >= R_MIN) and (pool["f1"] >= F1_MIN)

    # ---- 单位级（报告性：mean±SD + bootstrap CI）----
    u_rec_m, u_rec_sd, u_rec_ci, u_rec_n = unit_level(units, "tp", None, "fn")
    u_prec_m, u_prec_sd, u_prec_ci, u_prec_n = unit_level(units, "tp", "fp", None)

    # ---- 标签 / 门统计（docs/265 C2 口径：active = label ∈ {object, shadow}）----
    n_active = sum(1 for u in units if u["active"])
    obj_f = sum(1 for u in units if u["label"] == LABEL_OBJECT)
    shadow_f = sum(1 for u in units if u["label"] == LABEL_SHADOW)
    tex_f = sum(1 for u in units if u["label"] == LABEL_TEXTURE)
    none_f = sum(1 for u in units if u["label"] == LABEL_NONE)
    v1_r = sum(u["v1"] for u in units if u["active"]) / max(1, n_active)
    v3_r = sum(u["v3"] for u in units if u["active"]) / max(1, n_active)
    v4_r = sum(u["v4"] for u in units if u["active"]) / max(1, n_active)
    e1_r = sum(u["e1"] for u in units if u["active"]) / max(1, n_active)
    e2_r = sum(1 for u in units if u["active"] and u["e2"] == 1) / max(1, n_active)
    e3_r = sum(1 for u in units if u["active"] and u["e3"] == 1) / max(1, n_active)
    theta_vals = [u["theta"] for u in units if u["active"] and u["theta"] >= 0]
    theta_med = circular_median(theta_vals) if theta_vals else float("nan")
    theta_n = len(theta_vals)
    v3_applic = theta_n                       # 亮候选存在且 active（V3/E2 适用载体）
    dh_vals = [u["dh"] for u in units if u["active"] and u["dh"] >= 0]
    ds_vals = [u["ds"] for u in units if u["active"] and u["ds"] >= 0]
    dh_mean = float(np.mean(dh_vals)) if dh_vals else float("nan")
    ds_mean = float(np.mean(ds_vals)) if ds_vals else float("nan")

    # ---- C2 WHITE_OBJ_RECALL（pooled；docs/267 §1.4）----
    white_pix = int(sum(u["white_pix"] for u in units))
    white_tp = int(sum(u["white_tp"] for u in units))
    white_recall = white_tp / max(1, white_pix)
    c2_exercised = (white_pix >= WHITE_MIN_PIX) and (white_pix >= WHITE_MIN_FRAC * gt_total)
    c2 = c2_exercised and (white_recall >= 0.30)

    # ---- 守卫（docs/267 §1.6）----
    g_synth, g_synth_det, g_synth_cont = guard_synth()
    g_cell4, c4_obj, c4_shadow, c4_v3, c4_theta = guard_cell4()
    # GUARD_DET：首场景 A 帧（灰度 + 彩色 + ref；_s0 = (split, a, mask, free)）
    _s0 = scenes[0]
    _f0 = cv2.imread(_s0[3], cv2.IMREAD_COLOR)   # free 参考帧
    _a0 = cv2.imread(_s0[1], cv2.IMREAD_COLOR)   # 阴影帧
    _g0 = cv2.resize(cv2.cvtColor(_a0, cv2.COLOR_BGR2GRAY), RESIZE,
                     interpolation=cv2.INTER_AREA)
    _r0 = cv2.cvtColor(cv2.resize(_a0, RESIZE, interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    _fg = cv2.resize(cv2.cvtColor(_f0, cv2.COLOR_BGR2GRAY), RESIZE,
                     interpolation=cv2.INTER_AREA)
    _ref0 = compute_ref_dark([_fg, _g0])
    g_det, g_det_label = guard_demo(_g0, _r0, _ref0, None)
    g_mask = guard_mask(args.istd_dir, scenes)
    guards_ok = (g_synth == 1) and (g_cell4 == 1) and (g_det == 1) and (g_mask == 1)

    # ---- 判定（docs/267 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1_exercised:
        verdict = "REAL_SHADOW_LOW"
    elif c1 and (not c2_exercised or c2):
        verdict = "REAL_SHADOW_PASS"
    else:
        verdict = "REAL_SHADOW_FAIL"

    # ---- 内部确定性复现（docs/267 §1.6-5；第二遍强制重算，不读 checkpoint）----
    repro = 1
    if args.repro:
        per2 = run_all(use_resume=False)
        for u in units:
            k = "%s_%s" % (u["split"], u["id"])
            a, b = u, per2[k]
            for kk in REPRO_KEYS:
                va, vb = a[kk], b[kk]
                same = (va == vb) or (va != va and vb != vb)
                if not same:
                    repro = 0

    agg = {"pooled": pool, "train": pool_tr, "test": pool_te,
           "gt_total": gt_total, "c1_exercised": c1_exercised,
           "unit_recall": [u_rec_m, u_rec_sd, list(u_rec_ci), u_rec_n],
           "unit_precision": [u_prec_m, u_prec_sd, list(u_prec_ci), u_prec_n],
           "active": n_active, "obj_f": obj_f, "shadow_f": shadow_f,
           "tex_f": tex_f, "none_f": none_f,
           "v1": v1_r, "v3": v3_r, "v4": v4_r,
           "e1": e1_r, "e2": e2_r, "e3": e3_r,
           "theta_med": theta_med, "theta_n": theta_n, "v3_applic": v3_applic,
           "dh_mean": dh_mean, "ds_mean": ds_mean,
           "white_pix": white_pix, "white_tp": white_tp,
           "white_recall": white_recall, "c2_exercised": c2_exercised,
           "gt_empty": sum(1 for u in units if u["gt_sum"] == 0),
           "pred_empty": sum(1 for u in units if u["pred_sum"] == 0)}

    out = {
        "artifact": "light_shadow_real_gt_test",
        "doc_ref": "docs/267",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_real_shadow_prec_rec": bool(c1),
                     "c1_exercised": bool(c1_exercised),
                     "pooled": {k: round(v, 6) if isinstance(v, float) else v
                                for k, v in pool.items()},
                     "c2_white_obj_recall": bool(c2),
                     "c2_exercised": bool(c2_exercised),
                     "white_recall": white_recall},
        "guards": {"synth": g_synth, "synth_det": g_synth_det,
                   "synth_cont": g_synth_cont, "cell4": g_cell4,
                   "cell4_flamingo": {"obj_rate": c4_obj, "shadow_rate": c4_shadow,
                                      "v3_rate": c4_v3, "theta_med": c4_theta},
                   "det": g_det, "det_label": g_det_label, "mask": g_mask,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "rgt_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定，docs/267 §1.8）----
    print("R_GT_UNITS=%d" % len(units))
    print("R_GT_FRAMES=%d" % len(units))
    print("R_GT_TP=%d" % pool["tp"])
    print("R_GT_FP=%d" % pool["fp"])
    print("R_GT_FN=%d" % pool["fn"])
    print("R_GT_PREC=%.4f" % pool["p"])
    print("R_GT_RECALL=%.4f" % pool["r"])
    print("R_GT_F1=%.4f" % pool["f1"])
    print("R_GT_TRAIN_PREC=%.4f" % pool_tr["p"])
    print("R_GT_TRAIN_RECALL=%.4f" % pool_tr["r"])
    print("R_GT_TRAIN_F1=%.4f" % pool_tr["f1"])
    print("R_GT_TEST_PREC=%.4f" % pool_te["p"])
    print("R_GT_TEST_RECALL=%.4f" % pool_te["r"])
    print("R_GT_TEST_F1=%.4f" % pool_te["f1"])
    print("R_GT_UNIT_REC_MEAN=%.4f" % u_rec_m)
    print("R_GT_UNIT_REC_SD=%.4f" % u_rec_sd)
    print("R_GT_UNIT_REC_CI_LO=%.4f" % u_rec_ci[0])
    print("R_GT_UNIT_REC_CI_HI=%.4f" % u_rec_ci[1])
    print("R_GT_UNIT_PREC_MEAN=%.4f" % u_prec_m)
    print("R_GT_UNIT_PREC_SD=%.4f" % u_prec_sd)
    print("R_GT_GT_EMPTY=%d" % agg["gt_empty"])
    print("R_GT_PRED_EMPTY=%d" % agg["pred_empty"])
    print("R_GT_ACTIVE_RATE=%.4f" % (n_active / max(1, len(units))))
    print("R_GT_OBJ_F=%d" % obj_f)
    print("R_GT_SHADOW_F=%d" % shadow_f)
    print("R_GT_TEX_F=%d" % tex_f)
    print("R_GT_NONE_F=%d" % none_f)
    print("R_GT_V1_RATE=%.4f" % v1_r)
    print("R_GT_V3_RATE=%.4f" % v3_r)
    print("R_GT_V4_RATE=%.4f" % v4_r)
    print("R_GT_E1_RATE=%.4f" % e1_r)
    print("R_GT_E2_RATE=%.4f" % e2_r)
    print("R_GT_E3_RATE=%.4f" % e3_r)
    print("R_GT_THETA_MED=%.2f" % theta_med)
    print("R_GT_THETA_N=%d" % theta_n)
    print("R_GT_V3_APPLIC=%d" % v3_applic)
    print("R_GT_DH_MEAN=%.2f" % dh_mean)
    print("R_GT_DS_MEAN=%.2f" % ds_mean)
    print("R_GT_WHITE_PIX=%d" % white_pix)
    print("R_GT_WHITE_TP=%d" % white_tp)
    print("R_GT_WHITE_RECALL=%.4f" % white_recall)
    print("R_GT_WHITE_EXERCISED=%d" % (1 if c2_exercised else 0))
    print("R_GT_GUARD_SYNTH=%d" % g_synth)
    print("R_GT_GUARD_SYNTH_DET=%.4f" % g_synth_det)
    print("R_GT_GUARD_SYNTH_CONT=%.4f" % g_synth_cont)
    print("R_GT_GUARD_CELL4=%d" % g_cell4)
    print("R_GT_GUARD_CELL4_OBJ=%.4f" % c4_obj)
    print("R_GT_GUARD_CELL4_SHADOW=%.4f" % c4_shadow)
    print("R_GT_GUARD_CELL4_V3=%.4f" % c4_v3)
    print("R_GT_GUARD_CELL4_THETA=%.2f" % c4_theta)
    print("R_GT_GUARD_DET=%d" % g_det)
    print("R_GT_GUARD_DET_LABEL=%s" % g_det_label)
    print("R_GT_GUARD_MASK=%d" % g_mask)
    print("R_GT_REPRO=%d" % repro)
    print("R_GT_C1_PREC_REC=%s" % ("PASS" if c1 else "FAIL"))
    print("R_GT_C2_WHITE=%s" % ("PASS" if c2 else ("LOW" if not c2_exercised else "FAIL")))
    print("R_GT_VERDICT=%s" % verdict)
    print("R_GT_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
