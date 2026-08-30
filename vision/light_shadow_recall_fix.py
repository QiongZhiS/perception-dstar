"""vision/light_shadow_recall_fix.py — 光影判别第六格：recall 侧修复（否决门链误伤
真实阴影 → 计分制联合判别 + V3 去自指化）
（docs/270 预注册设计，修复机制/判据/守卫冻结；本脚本为唯一新增实验文件，import
复用第五格 vision/light_shadow_real_gt_test.py（list_scenes/pooled_prf/unit_level/
guard_cell4/guard_mask）、第四格 vision/light_shadow_real_test.py（compute_ref_dark/
diagnose_frame/circular_median/run_video/guard_synth/guard_demo/LABEL_*）、第一格
vision/light_shadow_test.py、第二格 vision/light_shadow_gate_test.py、第三格
vision/light_shadow_reflect_test.py、vision/real_stream_test.py、vision/critical_point.py，
未修改任何既有脚本）。

目标（docs/270 §1.1）：docs/267 的 recall 缺口（否决门链把 ~35% 真实阴影误判为
物体，pooled R 0.4131）做机制级修复——修复选型由预注册期修复前诊断（§二，ISTD
1870 单位全量）数据驱动冻结：否决门链 → **计分制联合判别**。测量层（Pass A/B、
时间门、V1/V3/V4 几何/反射率、E1/E2/E3、θ_est、ΔH/ΔS）逐字 import 第五格零改动；
唯一修复点 = 标签合成层：veto_count = v1 + v4（V3 自指 → 从否决链降为证据 E2），
label = object iff veto_count >= 2（V1 拓扑与 V4 反射率两正交线索联合才判物体）。

判据（docs/270 §1.4，冻结）：
  C1 REAL_SHADOW_PREC_REC [L3][机制][真实域证明]：pooled 逐像素 P/R/F1，
     P ≥ 0.90 且 R ≥ 0.55 且 F1 ≥ 0.65（三条件 AND）；行使门槛 = GT 像素 ≥ 50000
  C2 WHITE_OBJ_RECALL    [L3][机制][真实域证明]：亮目标（GT 阴影 ∩ B 帧 ≥128）
     pooled 召回 ≥ 0.50；行使门槛 = 亮目标 ≥ 2000 且 ≥ GT 1%
  C3 KEEP                [L3][机制][合成→真实保持]：守卫 R_RF_GUARD_BASE（本脚本
     基线路径 pooled P/R/F1 = 0.9913/0.4131/0.5832、white = 0.4265、标签分布
     627/1132/111、门率 0.1137/0.1148/0.1916 逐位）+ R_GT_GUARD_SYNTH（det/cont
     = 1.0）+ R_GT_GUARD_CELL4（flamingo 0.7875/0.2125/0.7875/287.38 逐位）
  C4（报告性）：修复前后 V1/V3/V4 各自误伤率 Δ（TS 被判 object 率 + 门参与误伤率
     + 损伤占比 + 剩余 object 单位构成）
  判定（docs/270 §1.5，冻结）：守卫全过 且 C1 过 且（C2 未行使 或 过）=
  RECALL_FIX_PASS；C1 不过 = RECALL_FIX_FAIL（P_FAIL/R_FAIL/F1_FAIL）；C1 过但 C2
  行使不过 = RECALL_FIX_FAIL（C2_FAIL）；GT < 50000 = REAL_SHADOW_LOW；守卫不过 =
  GUARD_FAIL。

守卫（docs/270 §1.6，冻结）：
  R_RF_GUARD_BASE：基线路径（diagnose_frame 原样 label）pooled P/R/F1/white + 标签
     分布 + 门率与 docs/267 §3 逐位一致（±容差）——import 链 + 预处理同源证明
  R_RF_GUARD_SYNTH：import 第三格 run_unit_reflect(30,0,"main")，det/cont == 1.0
  R_RF_GUARD_CELL4：import 第四格 run_video("flamingo")，obj/shadow/v3/theta_med
     与 docs/265 §3.1 一致（import 第五格 guard_cell4）
  R_RF_GUARD_DET：diagnose_frame 同输入两次调用输出全等（共享函数确定性）
  R_RF_GUARD_MASK：随机 5 场景数据完整性（import 第五格 guard_mask）
  R_RF_REPRO：--repro 时 1870 单位整体重跑第二遍（不读 checkpoint），逐项位级一致
     （NaN 感知比较）

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_RF_*
摘要块（顺序固定，见 SUMMARY_LINES 注释）；JSON 归档 vision/out/results/rrf_<tag>.json
+ checkpoint ckpt_rrf_<hash>.json（--resume 断点续跑）；数字用 vision/extract_r.py
纯正则抽取；禁止读取 logs/*.log 与 vision/out/results/*.json 原文；ISTD PNG 是数据。

用法：
  python vision/light_shadow_recall_fix.py --tag main --repro
  python vision/light_shadow_recall_fix.py --tag timing --limit 20
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
from light_shadow_real_gt_test import (  # 第五格：数据/聚合/守卫，逐字 import 复用
    list_scenes, pooled_prf, unit_level, guard_cell4, guard_mask,
)
from light_shadow_real_test import (  # 第四格：判别核心 + 守卫，逐字 import 复用
    compute_ref_dark, diagnose_frame, circular_median, run_video,
    guard_synth, guard_demo,
    LABEL_NONE, LABEL_TEXTURE, LABEL_OBJECT, LABEL_SHADOW,
)
from light_shadow_test import (  # 第一格：旋钮
    W, H, DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, OCC_LUM_THRESH,
)
from light_shadow_gate_test import (  # 第二格：否决门几何量（diagnose_frame 内部使用）
    touches_boundary, pca_axis, axis_err_deg,  # noqa: F401
    TOL_AXIS, RATIO_MIN,  # noqa: F401
)
from light_shadow_reflect_test import (  # 第三格：反射率判别（diagnose_frame 内部使用）
    run_unit_reflect,  # noqa: F401
    TOL_H, TOL_S, BAND_EDGE, SAT_MIN,  # noqa: F401
)
from real_stream_test import RESIZE

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 判据口径参数（docs/270 §1.4 冻结；非机制旋钮，先于运行冻结）----
P_MIN = 0.90            # C1 pooled precision 下限（"P 不崩"高杠）
R_MIN = 0.55            # C1 pooled recall 下限（recall 提升目标）
F1_MIN = 0.65           # C1 pooled F1 下限
GT_PIX_MIN = 50000      # C1 行使门槛：pooled GT 阴影像素
L_WHITE = 128.0         # C2 亮目标定义：参考图 B 灰度 ≥ 128（docs/267 同款）
WHITE_MIN_PIX = 2000    # C2 行使门槛：pooled 亮目标像素
WHITE_MIN_FRAC = 0.01   # C2 行使门槛：亮目标 ≥ GT 阴影像素 1%
WHITE_REC_MIN = 0.50    # C2 亮目标召回下限（recall 提升目标）

# ---- C4 分型口径（docs/270 §1.4 冻结；报告性）----
TS_OV = 0.5             # 候选 ∩ GT 阴影重叠 ≥ 0.5 = 真实阴影单位（TS）

# ---- 守卫 R_RF_GUARD_BASE 冻结参照（docs/267 §3 逐位；±容差）----
BASE_P, BASE_R, BASE_F1 = 0.9913, 0.4131, 0.5832
BASE_WHITE = 0.4265
BASE_OBJ_F, BASE_SHADOW_F, BASE_NONE_F = 627, 1132, 111
BASE_V1, BASE_V3, BASE_V4 = 0.1137, 0.1148, 0.1916
TOL_PRF = 5e-5          # P/R/F1 容差
TOL_WHITE = 1e-3        # white 容差
TOL_GATE = 1e-4         # 门率容差

# ---- 内部确定性复现键（docs/270 §1.6-6；每单位标量）----
REPRO_KEYS = ["tp", "fp", "fn", "white_tp", "white_pix", "gt_sum",
              "fix_tp", "fix_fp", "fix_fn", "white_tp_fix",
              "label", "label_fixed", "v1", "v3", "v4", "active",
              "area", "ov", "ts"]


# ---------------- 修复机制（docs/270 §1.3 冻结；唯一修复点 = 标签合成层） ----------------
def apply_fix(label, v1, v3, v4):
    """计分制联合判别（docs/270 §1.3 冻结，无新增旋钮）。

    veto_count = v1 + v4：V3 自指（θ_est = 亮候选质心−暗候选质心的帧内估计，用它
    判"主轴偏离光照→否决"是循环；诊断 §2.1 示轴误差 TS 48.24° vs NS 51.00° 无分离
    力）→ 从否决链降为证据（E2 保留测量，不计入否决数）。
    object iff veto_count >= 2：V1（闭合拓扑）与 V4（反射率跳变）是 docs/261/263
    标定的两正交独立线索，单线索不再一票否决，两线索联合成立才判物体——人眼
    "多维联合、不靠单一线索"。"""
    if label not in (LABEL_OBJECT, LABEL_SHADOW):
        return label
    veto_count = int(bool(v1)) + int(bool(v4))
    return LABEL_OBJECT if veto_count >= 2 else LABEL_SHADOW


# ---------------- 数据（ISTD；目录可读，PNG 是数据；映射 docs/267 §二 D1 冻结） ----------------
def run_scene_fixed(split, a_path, mask_path, free_path):
    """跑单个 ISTD 场景（docs/270 §1.2/§1.3 冻结）：2 帧单位 [free, shadow]。

    预处理与判别**逐字同第五格 run_scene**（import 同源）；同一 diagnose_frame
    输出上同时算 基线（label 原样）与 修复（apply_fix）两套标签的 tp/fp/fn/white——
    同代码路径，基线即 docs/267 逐位复现载体（R_RF_GUARD_BASE）。

    返回 per-scene dict（紧凑标量，供聚合/复现）。GT 掩码只用于评估，绝不进入机制。"""
    b_bgr = cv2.imread(free_path, cv2.IMREAD_COLOR)      # free = 无阴影参考（_C）
    a_bgr = cv2.imread(a_path, cv2.IMREAD_COLOR)         # A = 阴影图（评估帧）
    c_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # mask = GT 掩码（_B）
    if b_bgr is None or a_bgr is None or c_gray is None:
        raise RuntimeError("cannot read scene: %s" % a_path)

    # ---- 预处理（docs/270 §1.2 冻结；docs/267 §1.2 同款逐字）----
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

    base_label = da["label"]
    fixed_label = apply_fix(da["label"], da["v1"], da["v3"], da["v4"])
    mask = da["mask"]

    def metrics(lab):
        pred = mask if lab == LABEL_SHADOW else np.zeros((H, W), bool)
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
        fn = int(np.logical_and(np.logical_not(pred), gt).sum())
        return tp, fp, fn

    bt, bf, bfn = metrics(base_label)
    ft, ff, ffn = metrics(fixed_label)

    # ---- C2 亮目标（docs/270 §1.4：GT 阴影 ∩ B 帧灰度 ≥ L_WHITE；无条件统计）----
    white = gt & (b_gray >= L_WHITE)
    white_pix = int(white.sum())
    white_tp_base = int(np.logical_and(mask if base_label == LABEL_SHADOW else
                                       np.zeros((H, W), bool), white).sum())
    white_tp_fix = int(np.logical_and(mask if fixed_label == LABEL_SHADOW else
                                      np.zeros((H, W), bool), white).sum())

    # ---- C4 分型（报告性）：候选 ∩ GT 阴影重叠 ----
    area = int(mask.sum())
    ov = float(np.logical_and(mask, gt).sum()) / float(area) if area > 0 else 0.0

    def bi(x):
        return 1 if x else 0

    return dict(id=os.path.basename(a_path), split=split,
                tp=bt, fp=bf, fn=bfn, white_tp=white_tp_base, white_pix=white_pix,
                fix_tp=ft, fix_fp=ff, fix_fn=ffn, white_tp_fix=white_tp_fix,
                gt_sum=int(gt.sum()),
                label=base_label, label_fixed=fixed_label,
                active=bi(da["active"]),
                v1=bi(da["v1"]), v3=bi(da["v3"]), v4=bi(da["v4"]),
                area=area, ov=ov, ts=1 if ov >= TS_OV else 0,
                e1=bi(da["e1"]),
                e2=(-1 if da["e2"] is None else bi(da["e2"])),
                e3=(-1 if da["e3"] is None else bi(da["e3"])),
                theta=(-1.0 if da["theta_est"] is None else round(float(da["theta_est"]), 3)),
                dh=(-1.0 if da["dh"] is None else round(float(da["dh"]), 3)),
                ds=(-1.0 if da["ds"] is None else round(float(da["ds"]), 3)))


# ---------------- 聚合（docs/270 §1.4 冻结：C1/C2 pooled；基线/修复双列） ----------------
def pooled_prf_parts(units, prefix):
    """从 per-unit 取 {prefix}tp/fp/fn 聚合 pooled P/R/F1。"""
    tp = int(sum(u[prefix + "tp"] for u in units))
    fp = int(sum(u[prefix + "fp"] for u in units))
    fn = int(sum(u[prefix + "fn"] for u in units))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="光影判别第六格：recall 修复（ISTD）")
    ap.add_argument("--istd-dir", default=r"D:\datasets\ISTD")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="rrf")
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
        print("R_RF_ERROR_NO_SCENES=1")
        return 1

    cfg = {"istd_dir": args.istd_dir, "limit": args.limit, "tag": args.tag,
           "fix": {"mode": "scoring", "v3_removed_from_veto": True,
                   "veto_count_threshold": 2},
           "mechanism": {"delta_shadow": DELTA_SHADOW, "a_min": A_MIN,
                         "k_move": K_MOVE, "move_iou": MOVE_IOU,
                         "occ_lum_thresh": OCC_LUM_THRESH, "ref_pct": 95.0,
                         "veto": {"tol_axis": TOL_AXIS, "ratio_min": RATIO_MIN,
                                  "tol_h": TOL_H, "tol_s": TOL_S,
                                  "band_edge": BAND_EDGE, "sat_min": SAT_MIN}},
           "criteria": {"p_min": P_MIN, "r_min": R_MIN, "f1_min": F1_MIN,
                        "gt_pix_min": GT_PIX_MIN, "l_white": L_WHITE,
                        "white_min_pix": WHITE_MIN_PIX,
                        "white_min_frac": WHITE_MIN_FRAC,
                        "white_rec_min": WHITE_REC_MIN, "ts_ov": TS_OV}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_rrf_%s.json" % ck_tag)

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
            per_unit[key] = run_scene_fixed(split, a, b, c)
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            if (i + 1) % 200 == 0:
                print("PROGRESS %d/%d" % (i + 1, len(scenes)), flush=True)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%s_%s" % (s, os.path.basename(a))] for s, a, b, c in scenes]

    # ---- 基线（R_RF_GUARD_BASE 载体）与修复（C1）双 pooled ----
    pool_base = pooled_prf_parts(units, "")
    pool_fix = pooled_prf_parts(units, "fix_")
    units_train = [u for u in units if u["split"] == "train"]
    units_test = [u for u in units if u["split"] == "test"]
    pool_tr = pooled_prf_parts(units_train, "fix_")
    pool_te = pooled_prf_parts(units_test, "fix_")
    gt_total = pool_fix["tp"] + pool_fix["fn"]
    c1_exercised = gt_total >= GT_PIX_MIN
    c1 = (pool_fix["p"] >= P_MIN) and (pool_fix["r"] >= R_MIN) and (pool_fix["f1"] >= F1_MIN)

    # ---- 单位级（报告性：mean±SD + bootstrap CI；修复口径）----
    u_rec_m, u_rec_sd, u_rec_ci, u_rec_n = unit_level(
        [dict(tp=u["fix_tp"], fn=u["fix_fn"]) for u in units], "tp", None, "fn")
    u_prec_m, u_prec_sd, u_prec_ci, u_prec_n = unit_level(
        [dict(tp=u["fix_tp"], fp=u["fix_fp"]) for u in units], "tp", "fp", None)

    # ---- 标签分布（修复口径；docs/267 C2 口径：active = label ∈ {object, shadow}）----
    n_active = sum(1 for u in units if u["active"])
    obj_f = sum(1 for u in units if u["label_fixed"] == LABEL_OBJECT)
    shadow_f = sum(1 for u in units if u["label_fixed"] == LABEL_SHADOW)
    tex_f = sum(1 for u in units if u["label_fixed"] == LABEL_TEXTURE)
    none_f = sum(1 for u in units if u["label_fixed"] == LABEL_NONE)

    # ---- 基线读数（R_RF_GUARD_BASE 参照；docs/267 同口径）----
    base_obj_f = sum(1 for u in units if u["label"] == LABEL_OBJECT)
    base_shadow_f = sum(1 for u in units if u["label"] == LABEL_SHADOW)
    base_v1 = sum(u["v1"] for u in units if u["active"]) / max(1, n_active)
    base_v3 = sum(u["v3"] for u in units if u["active"]) / max(1, n_active)
    base_v4 = sum(u["v4"] for u in units if u["active"]) / max(1, n_active)
    white_pix = int(sum(u["white_pix"] for u in units))
    white_tp_fix = int(sum(u["white_tp_fix"] for u in units))
    white_recall = white_tp_fix / max(1, white_pix)
    white_tp_base = int(sum(u["white_tp"] for u in units))
    white_recall_base = white_tp_base / max(1, white_pix)
    c2_exercised = (white_pix >= WHITE_MIN_PIX) and (white_pix >= WHITE_MIN_FRAC * gt_total)
    c2 = c2_exercised and (white_recall >= WHITE_REC_MIN)

    # ---- C4 修复前后误伤率（报告性）----
    act = [u for u in units if u["active"]]
    ts_u = [u for u in act if u["ts"]]
    ns_u = [u for u in act if not u["ts"]]
    before_ts_obj = sum(1 for u in ts_u if u["label"] == LABEL_OBJECT) / max(1, len(ts_u))
    before_ts_v1 = sum(1 for u in ts_u if u["v1"]) / max(1, len(ts_u))
    before_ts_v3 = sum(1 for u in ts_u if u["v3"]) / max(1, len(ts_u))
    before_ts_v4 = sum(1 for u in ts_u if u["v4"]) / max(1, len(ts_u))
    before_v1_share = sum(1 for u in ts_u if u["v1"]) / max(1, sum(1 for u in act if u["v1"]))
    before_v3_share = sum(1 for u in ts_u if u["v3"]) / max(1, sum(1 for u in act if u["v3"]))
    before_v4_share = sum(1 for u in ts_u if u["v4"]) / max(1, sum(1 for u in act if u["v4"]))
    after_ts_obj = sum(1 for u in ts_u if u["label_fixed"] == LABEL_OBJECT) / max(1, len(ts_u))
    # 修复后：剩余 object 单位（v1∧v4 联合 / 三门）的门构成与 TS 占比
    obj_remain = [u for u in act if u["label_fixed"] == LABEL_OBJECT]
    combo5 = sum(1 for u in obj_remain if u["v1"] and u["v4"] and not u["v3"])
    combo7 = sum(1 for u in obj_remain if u["v1"] and u["v3"] and u["v4"])
    after_v1_share = sum(1 for u in obj_remain if u["ts"]) / max(1, len(obj_remain))
    after_ts_v1 = sum(1 for u in ts_u if u["v1"] and u["label_fixed"] == LABEL_OBJECT) / max(1, len(ts_u))
    after_ts_v3 = sum(1 for u in ts_u if u["v3"] and u["label_fixed"] == LABEL_OBJECT) / max(1, len(ts_u))
    after_ts_v4 = sum(1 for u in ts_u if u["v4"] and u["label_fixed"] == LABEL_OBJECT) / max(1, len(ts_u))

    # ---- 守卫（docs/270 §1.6）----
    g_base = 1 if (abs(pool_base["p"] - BASE_P) <= TOL_PRF
                   and abs(pool_base["r"] - BASE_R) <= TOL_PRF
                   and abs(pool_base["f1"] - BASE_F1) <= TOL_PRF
                   and abs(white_recall_base - BASE_WHITE) <= TOL_WHITE
                   and base_obj_f == BASE_OBJ_F and base_shadow_f == BASE_SHADOW_F
                   and none_f == BASE_NONE_F
                   and abs(base_v1 - BASE_V1) <= TOL_GATE
                   and abs(base_v3 - BASE_V3) <= TOL_GATE
                   and abs(base_v4 - BASE_V4) <= TOL_GATE) else 0
    g_synth, g_synth_det, g_synth_cont = guard_synth()
    g_cell4, c4_obj, c4_shadow, c4_v3, c4_theta = guard_cell4()
    _s0 = scenes[0]
    _f0 = cv2.imread(_s0[3], cv2.IMREAD_COLOR)
    _a0 = cv2.imread(_s0[1], cv2.IMREAD_COLOR)
    _g0 = cv2.resize(cv2.cvtColor(_a0, cv2.COLOR_BGR2GRAY), RESIZE,
                     interpolation=cv2.INTER_AREA)
    _r0 = cv2.cvtColor(cv2.resize(_a0, RESIZE, interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    _fg = cv2.resize(cv2.cvtColor(_f0, cv2.COLOR_BGR2GRAY), RESIZE,
                     interpolation=cv2.INTER_AREA)
    _ref0 = compute_ref_dark([_fg, _g0])
    g_det, g_det_label = guard_demo(_g0, _r0, _ref0, None)
    g_mask = guard_mask(args.istd_dir, scenes)
    guards_ok = (g_base == 1) and (g_synth == 1) and (g_cell4 == 1) \
        and (g_det == 1) and (g_mask == 1)

    # ---- 判定（docs/270 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1_exercised:
        verdict = "REAL_SHADOW_LOW"
    elif c1 and (not c2_exercised or c2):
        verdict = "RECALL_FIX_PASS"
    else:
        verdict = "RECALL_FIX_FAIL"

    # ---- 内部确定性复现（docs/270 §1.6-6；第二遍强制重算，不读 checkpoint）----
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

    agg = {"base": pool_base, "fixed": pool_fix,
           "train": pool_tr, "test": pool_te,
           "gt_total": gt_total, "c1_exercised": c1_exercised,
           "unit_recall": [u_rec_m, u_rec_sd, list(u_rec_ci), u_rec_n],
           "unit_precision": [u_prec_m, u_prec_sd, list(u_prec_ci), u_prec_n],
           "active": n_active, "obj_f": obj_f, "shadow_f": shadow_f,
           "tex_f": tex_f, "none_f": none_f,
           "white_pix": white_pix, "white_tp": white_tp_fix,
           "white_recall": white_recall, "c2_exercised": c2_exercised,
           "base_obj_f": base_obj_f, "base_shadow_f": base_shadow_f,
           "base_v1": base_v1, "base_v3": base_v3, "base_v4": base_v4,
           "base_white_recall": white_recall_base,
           "c4": {"before_ts_obj": before_ts_obj,
                  "before_ts_v1": before_ts_v1, "before_ts_v3": before_ts_v3,
                  "before_ts_v4": before_ts_v4,
                  "before_v1_share": before_v1_share,
                  "before_v3_share": before_v3_share,
                  "before_v4_share": before_v4_share,
                  "after_ts_obj": after_ts_obj,
                  "after_ts_v1": after_ts_v1, "after_ts_v3": after_ts_v3,
                  "after_ts_v4": after_ts_v4,
                  "obj_remain": len(obj_remain),
                  "combo5": combo5, "combo7": combo7,
                  "after_v1_share": after_v1_share}}

    out = {
        "artifact": "light_shadow_recall_fix",
        "doc_ref": "docs/270",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_real_shadow_prec_rec": bool(c1),
                     "c1_exercised": bool(c1_exercised),
                     "pooled_base": {k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in pool_base.items()},
                     "pooled_fixed": {k: round(v, 6) if isinstance(v, float) else v
                                      for k, v in pool_fix.items()},
                     "c2_white_obj_recall": bool(c2),
                     "c2_exercised": bool(c2_exercised),
                     "white_recall": white_recall},
        "guards": {"base": g_base, "synth": g_synth,
                   "synth_det": g_synth_det, "synth_cont": g_synth_cont,
                   "cell4": g_cell4,
                   "cell4_flamingo": {"obj_rate": c4_obj, "shadow_rate": c4_shadow,
                                      "v3_rate": c4_v3, "theta_med": c4_theta},
                   "det": g_det, "det_label": g_det_label, "mask": g_mask,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "rrf_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定，docs/270 §1.8）----
    print("R_RF_UNITS=%d" % len(units))
    print("R_RF_FRAMES=%d" % len(units))
    print("R_RF_TP=%d" % pool_fix["tp"])
    print("R_RF_FP=%d" % pool_fix["fp"])
    print("R_RF_FN=%d" % pool_fix["fn"])
    print("R_RF_PREC=%.4f" % pool_fix["p"])
    print("R_RF_RECALL=%.4f" % pool_fix["r"])
    print("R_RF_F1=%.4f" % pool_fix["f1"])
    print("R_RF_TRAIN_PREC=%.4f" % pool_tr["p"])
    print("R_RF_TRAIN_RECALL=%.4f" % pool_tr["r"])
    print("R_RF_TRAIN_F1=%.4f" % pool_tr["f1"])
    print("R_RF_TEST_PREC=%.4f" % pool_te["p"])
    print("R_RF_TEST_RECALL=%.4f" % pool_te["r"])
    print("R_RF_TEST_F1=%.4f" % pool_te["f1"])
    print("R_RF_UNIT_REC_MEAN=%.4f" % u_rec_m)
    print("R_RF_UNIT_REC_SD=%.4f" % u_rec_sd)
    print("R_RF_UNIT_REC_CI_LO=%.4f" % u_rec_ci[0])
    print("R_RF_UNIT_REC_CI_HI=%.4f" % u_rec_ci[1])
    print("R_RF_UNIT_PREC_MEAN=%.4f" % u_prec_m)
    print("R_RF_UNIT_PREC_SD=%.4f" % u_prec_sd)
    print("R_RF_WHITE_PIX=%d" % white_pix)
    print("R_RF_WHITE_TP=%d" % white_tp_fix)
    print("R_RF_WHITE_RECALL=%.4f" % white_recall)
    print("R_RF_WHITE_EXERCISED=%d" % (1 if c2_exercised else 0))
    print("R_RF_OBJ_F=%d" % obj_f)
    print("R_RF_SHADOW_F=%d" % shadow_f)
    print("R_RF_TEX_F=%d" % tex_f)
    print("R_RF_NONE_F=%d" % none_f)
    print("R_RF_ACTIVE_RATE=%.4f" % (n_active / max(1, len(units))))
    print("R_RF_OBJ_REMAIN=%d" % len(obj_remain))
    print("R_RF_BASE_PREC=%.4f" % pool_base["p"])
    print("R_RF_BASE_RECALL=%.4f" % pool_base["r"])
    print("R_RF_BASE_F1=%.4f" % pool_base["f1"])
    print("R_RF_BASE_WHITE=%.4f" % white_recall_base)
    print("R_RF_BASE_OBJ_F=%d" % base_obj_f)
    print("R_RF_BASE_SHADOW_F=%d" % base_shadow_f)
    print("R_RF_BASE_V1=%.4f" % base_v1)
    print("R_RF_BASE_V3=%.4f" % base_v3)
    print("R_RF_BASE_V4=%.4f" % base_v4)
    print("R_RF_BEFORE_TS_OBJ=%.4f" % before_ts_obj)
    print("R_RF_BEFORE_TS_V1=%.4f" % before_ts_v1)
    print("R_RF_BEFORE_TS_V3=%.4f" % before_ts_v3)
    print("R_RF_BEFORE_TS_V4=%.4f" % before_ts_v4)
    print("R_RF_BEFORE_V1_TS_SHARE=%.4f" % before_v1_share)
    print("R_RF_BEFORE_V3_TS_SHARE=%.4f" % before_v3_share)
    print("R_RF_BEFORE_V4_TS_SHARE=%.4f" % before_v4_share)
    print("R_RF_AFTER_TS_OBJ=%.4f" % after_ts_obj)
    print("R_RF_AFTER_TS_V1=%.4f" % after_ts_v1)
    print("R_RF_AFTER_TS_V3=%.4f" % after_ts_v3)
    print("R_RF_AFTER_TS_V4=%.4f" % after_ts_v4)
    print("R_RF_AFTER_V1_TS_SHARE=%.4f" % after_v1_share)
    print("R_RF_AFTER_COMBO5=%d" % combo5)
    print("R_RF_AFTER_COMBO7=%d" % combo7)
    print("R_RF_GUARD_BASE=%d" % g_base)
    print("R_RF_GUARD_SYNTH=%d" % g_synth)
    print("R_RF_GUARD_SYNTH_DET=%.4f" % g_synth_det)
    print("R_RF_GUARD_SYNTH_CONT=%.4f" % g_synth_cont)
    print("R_RF_GUARD_CELL4=%d" % g_cell4)
    print("R_RF_GUARD_CELL4_OBJ=%.4f" % c4_obj)
    print("R_RF_GUARD_CELL4_SHADOW=%.4f" % c4_shadow)
    print("R_RF_GUARD_CELL4_V3=%.4f" % c4_v3)
    print("R_RF_GUARD_CELL4_THETA=%.2f" % c4_theta)
    print("R_RF_GUARD_DET=%d" % g_det)
    print("R_RF_GUARD_DET_LABEL=%s" % g_det_label)
    print("R_RF_GUARD_MASK=%d" % g_mask)
    print("R_RF_REPRO=%d" % repro)
    print("R_RF_C1_PREC_REC=%s" % ("PASS" if c1 else "FAIL"))
    print("R_RF_C2_WHITE=%s" % ("PASS" if c2 else ("LOW" if not c2_exercised else "FAIL")))
    print("R_RF_VERDICT=%s" % verdict)
    print("R_RF_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
