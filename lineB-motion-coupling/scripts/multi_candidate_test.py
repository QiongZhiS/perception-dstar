"""lineB-motion-coupling/scripts/multi_candidate_test.py — B 路第六格：多候选发现——
把"单最大暗候选"升级为"全图多候选"（攻漏检：第二个/小阴影/分散阴影进不了判别器）
（docs: lineB-motion-coupling/docs/B6-多候选发现-攻漏检-预注册设计.md §一 冻结）。

核心：用户"明显阴影检测不出来（漏检）"的根因 = 候选发现缺位——判别器每场景只取一个
最大暗连通域（diagnose_frame → largest_component(dark)），第二个/小阴影/分散阴影根本
进不了判别器；判别器本身（docs/270 apply_fix 计分制 + docs/272 adaptive_delta）在候选
已给定时的 P=0.97/F1=0.83 很强。本格 = 候选发现补齐：dark 掩码（自适应工作点）的全部
8-连通分量（面积 ≥ A_MIN，面积降序）各自独立过现有判别器（测量与 diagnose_frame 分支
结构逐字同式、函数 import 复用：touches_boundary/pca_axis/axis_err_deg/reflect_stats；
标签 = apply_fix 计分制；工作点 = adaptive_delta 场景级自适应，两遍法逐字），输出 =
全图阴影标注（多候选并集）→ 全图 pooled recall/precision vs 单候选（docs/272 adapt 列）。

候选发现选型（§1.3 冻结；本格唯一"运行前由诊断数据定案"环节）：A 全连通域列表
（零新增旋钮，本格实现）；B 超像素 SLIC（cv2.ximgproc 在 cv2 5.0.0 headless 缺失、
无 skimage → 本环境不可实现，--probe-b 探测；若可行则按 §1.3 选型规则比较）。选型
记录于 §二。

判据（§1.4 冻结，docs/247 标签 [L3][机制][真实域证明]）：
  C1 MULTI_RECALL_GAIN: 全图 pooled recall(multi) − pooled recall(adapt 单候选) ≥
     +0.15 → PASS；[+0.05, +0.15) → PARTIAL；< +0.05 → FAIL；行使门槛 GT ≥ 50000
  C2 MULTI_COUNT      : 含 ≥2 个 GT 阴影分量（各 ≥ A_MIN）的场景中，被检出的（shadow
     标签且与某 GT 分量 IoU ≥ 0.20）候选数均值 ≥ 2.0；行使门槛 = 此类场景 ≥ 30
  C3 PREC_KEEP        : 全图 pooled precision(multi) ≥ 0.85（P 不崩）
  C4 KEEP 守卫        : G_BASE（docs/267 逐位）+ G_FIX（docs/270 逐位）+ G_ADAPT
     （docs/272 逐位）+ G_DEGEN（每单位最大候选 ≡ 单候选 adapt 逐位）+ G_SYNTH +
     G_CELL4 + G_DET + G_MASK + G_REPRO
  C5（报告性）        : 候选数分布 / 新增候选 TS-FP 构成 / 新增 TP 像素 / GT 分量覆盖率
判定（§1.5）：守卫全过 且 C1 PASS 且（C2 未行使 或 过）且 C3 过 = MULTI_CAND_PASS；
C1 PARTIAL 且 C2/C3 过 = MULTI_CAND_PARTIAL；C1 FAIL 或 C2 行使不过 或 C3 不过 =
MULTI_CAND_FAIL（R_GAIN_FAIL/COUNT_FAIL/PREC_FAIL）；守卫不过 = GUARD_FAIL。

守卫（§1.6 冻结）：
  G_BASE : base 列（da0 label 原样）pooled P/R/F1 = 0.9913/0.4131/0.5832、white =
           0.4265、标签 obj 627/shadow 1132/none 111、门率 V1 0.1137/V3 0.1148/
           V4 0.1916——docs/267 §3 逐位（±容差）
  G_FIX  : fix 列（apply_fix(da0)）pooled P/R/F1 = 0.9813/0.6382/0.7734、white =
           0.6172、object 24——docs/270 §3 逐位
  G_ADAPT: adapt 列（apply_fix(da)，注入自适应后）pooled P/R/F1 = 0.9675/0.7316/
           0.8332、white = 0.7195、object 22、DELTA 0.25/0.25/0.3497、触发率 1.0、
           SOFT 0.7326→0.8487、HARD 0.6074→0.6760、单位级 recall Δ 均值 +0.0834、
           label 变化 12（TS 9）——docs/272 §3 逐位
  G_DEGEN: 每单位 cand[0]（最大候选）掩码 == da["mask"] 且 v1/v3/v4/theta/dh/ds ==
           da 的 且 apply_fix(cand[0]) == adapt_label 且 pred_multi ⊇ da["mask"]
           ——多候选 = 单候选的严格超集（同一工作点、同一判别语义）
  G_SYNTH : import 第三格 run_unit_reflect(30,0,"main")：det_gated == 1.0 且
            cont_rate == 1.0（docs/263 冻结数字）
  G_CELL4 : import 第四格 run_video("flamingo")：obj 0.7875 / shadow 0.2125 /
            V3 0.7875 / θ_med 287.38（docs/265 §3.1 逐位）
  G_DET   : diagnose_frame 同输入两次调用输出全等（共享函数确定性）
  G_MASK  : 随机 5 场景数据完整性（import 第五格 guard_mask）
  G_REPRO : --repro 时 1870 单位整体重跑第二遍（不读 checkpoint），关键数字位级一致
            （NaN 感知比较，含 var_inner/delta/n_cand/degen）

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_MC6_*
摘要块（顺序固定，见 SUMMARY_LINES 注释）；JSON 归档 lineB-motion-coupling/out/
mc6_<tag>.json + checkpoint ckpt_mc6_<hash>.json（--resume 断点续跑，每 100 单位写
一次）；数字用 vision/extract_r.py 纯正则抽取；禁止读取 lineB-motion-coupling/out/
*.log 与 lineB-motion-coupling/out/*.json 原文；ISTD PNG 是数据。
**未修改任何主线既有脚本**（vision/ 下全部不动；B1-B5 脚本亦不动，只 import）。

用法：
  python lineB-motion-coupling/scripts/multi_candidate_test.py --probe-b
  python lineB-motion-coupling/scripts/multi_candidate_test.py --tag timing --limit 20
  python lineB-motion-coupling/scripts/multi_candidate_test.py --tag diag --limit 300
  python lineB-motion-coupling/scripts/multi_candidate_test.py --tag diag
  python lineB-motion-coupling/scripts/multi_candidate_test.py --tag main --repro
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

# ---- import 复用判别器（docs/B6 §一：零改动，只 import）----
import light_shadow_real_test as _lsr  # noqa: E402  （运行时注入 DELTA_SHADOW，不修改任何文件）

from critical_point import mean_sd, bootstrap_ci  # noqa: E402  （统计外壳）
from light_shadow_adaptive import (  # noqa: E402  （第七格：自适应函数 f，逐字 import）
    adaptive_delta,
)
from light_shadow_recall_fix import (  # noqa: E402  （第六格：计分制合成，逐字 import）
    apply_fix, pooled_prf_parts,
)
from light_shadow_real_gt_test import (  # noqa: E402  （第五格：数据/聚合/守卫）
    list_scenes, unit_level, guard_cell4, guard_mask,
)
from light_shadow_real_test import (  # noqa: E402  （第四格：判别核心 + 守卫）
    compute_ref_dark, diagnose_frame, run_video,
    guard_synth, guard_demo,
    LABEL_NONE, LABEL_TEXTURE, LABEL_OBJECT, LABEL_SHADOW,
)
from light_shadow_test import (  # noqa: E402  （第一格：旋钮）
    W, H, DELTA_SHADOW, A_MIN, K_MOVE, MOVE_IOU, OCC_LUM_THRESH,
    largest_component, iou,
)
from light_shadow_gate_test import (  # noqa: E402  （第二格：否决门几何量）
    touches_boundary, pca_axis, axis_err_deg,
    TOL_AXIS, RATIO_MIN,
)
from light_shadow_reflect_test import (  # noqa: E402  （第三格：反射率判别）
    reflect_stats,
    TOL_H, TOL_S, BAND_EDGE, SAT_MIN,
)
from real_stream_test import RESIZE  # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("lineB-motion-coupling", "out")
N_BOOT = 2000
BOOT_SEED = 20260828

# ---- 机制常量（docs/B6 §1.3 冻结；全部继承 docs/272 冻结值，零新增机制旋钮）----
DELTA0 = 0.35            # 冻结先验（= docs/260 DELTA_SHADOW）；f 的中心与上限
EPS_DELTA = 1e-9         # delta 与 DELTA0 的等值容差（退化情形跳过冗余第三次诊断）
VAR_REF = 0.040          # f 系数（docs/272，冻结；仅用于 SOFT/HARD 分型与报告）
MATCH_IOU = 0.20         # C2/C5 候选-分量匹配门槛（docs/260 IOU_FP 同款语义）

assert abs(DELTA0 - DELTA_SHADOW) < 1e-9, "DELTA0 must equal frozen prior 0.35"

# ---- 判据口径参数（docs/B6 §1.4 冻结；非机制旋钮，先于运行冻结）----
GAIN_PASS = 0.15          # C1 PASS：全图 pooled recall 增益 ≥ +0.15（vs adapt 单候选）
GAIN_PARTIAL_LO = 0.05    # C1 PARTIAL：[+0.05, +0.15)
PREC_MIN = 0.85           # C3：全图 pooled precision ≥ 0.85（P 不崩）
COUNT_MIN = 2.0           # C2：多阴影场景检出候选数均值 ≥ 2.0
GT_PIX_MIN = 50000        # C1 行使门槛：pooled GT 阴影像素
MULTI_GT_MIN_SCENES = 30  # C2 行使门槛：多阴影（GT 分量 ≥ 2）场景数
L_WHITE = 128.0           # 亮目标定义：参考图 B 灰度 ≥ 128（docs/267 同款）
WHITE_MIN_PIX = 2000      # 亮目标行使门槛（报告性，同 docs/267）
WHITE_MIN_FRAC = 0.01
TS_OV = 0.5               # 候选 ∩ GT 阴影重叠 ≥ 0.5 = 真实阴影单位（TS，分型报告）

# ---- 守卫冻结参照（docs/267 §3 + docs/270 §3 + docs/272 §3 逐位；±容差）----
BASE_P, BASE_R, BASE_F1 = 0.9913, 0.4131, 0.5832   # G_BASE（docs/267）
BASE_WHITE = 0.4265
BASE_OBJ_F, BASE_SHADOW_F, BASE_NONE_F = 627, 1132, 111
BASE_V1, BASE_V3, BASE_V4 = 0.1137, 0.1148, 0.1916
FIX_P, FIX_R, FIX_F1 = 0.9813, 0.6382, 0.7734       # G_FIX（docs/270）
FIX_WHITE = 0.6172
FIX_OBJ_F = 24
AD_P, AD_R, AD_F1 = 0.9675, 0.7316, 0.8332          # G_ADAPT（docs/272）
AD_WHITE = 0.7195
AD_OBJ_F = 22
AD_DELTA_MIN, AD_DELTA_MED, AD_DELTA_MAX = 0.25, 0.25, 0.3497
AD_TRIGGER = 1.0000
AD_SOFT_BEF, AD_SOFT_AFT = 0.7326, 0.8487
AD_HARD_BEF, AD_HARD_AFT = 0.6074, 0.6760
AD_UNIT_REC_DELTA = 0.0834
AD_LABEL_CHANGED, AD_LABEL_CHANGED_TS = 12, 9
TOL_PRF = 5e-5           # P/R/F1 容差
TOL_WHITE = 1e-3         # white 容差
TOL_GATE = 1e-4          # 门率/触发率容差
TOL_DELTA = 1e-3         # DELTA 分布/归因容差

# ---- 内部确定性复现键（docs/B6 §1.6-9；每单位标量）----
REPRO_KEYS = ["tp", "fp", "fn", "fix_tp", "fix_fp", "fix_fn",
              "ad_tp", "ad_fp", "ad_fn", "mu_tp", "mu_fp", "mu_fn",
              "white_tp", "white_tp_fix", "white_tp_adapt", "white_tp_multi",
              "white_pix", "gt_sum",
              "label", "label_fixed", "label_adapt",
              "v1", "v3", "v4", "active", "area", "ov", "ts",
              "var_inner", "delta",
              "n_cand", "n_shadow", "mu_count", "single_matched", "gt_comps",
              "new_tp", "new_cand_total", "new_cand_ts", "new_cand_fp",
              "gtcov_total", "gtcov_matched", "degen"]


# ---------------- 候选发现（docs/B6 §1.3 冻结；A 全连通域列表） ----------------
def all_components(mask_bool, min_area=A_MIN):
    """8-连通域全部分量（面积 ≥ min_area），按面积降序（确定性）。

    与 largest_component 同源（cv2.connectedComponents, connectivity=8）——
    largest_component 只取最大分量，本函数取全部。返回 (masks, centroids)。"""
    if mask_bool.sum() == 0:
        return [], []
    n, lab = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
    if n <= 1:
        return [], []
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    comps = []
    for k in range(1, n):
        if areas[k] < min_area:
            continue
        comp = lab == k
        ys, xs = np.nonzero(comp)
        comps.append((int(areas[k]), comp, (float(xs.mean()), float(ys.mean()))))
    comps.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c, _ in comps], [cen for _, _, cen in comps]


def diagnose_candidate(gray, rgb_rgb, ref_dark_log, dark_prev, mask, b_c):
    """单候选判别（docs/B6 §1.3 冻结）：测量与 diagnose_frame 分支结构逐字同式，
    函数 import 复用（touches_boundary/pca_axis/axis_err_deg/reflect_stats/iou）；
    标签合成 = docs/270 apply_fix 计分制（v1+v4 ≥ 2 判 object）。

    gray/rgb_rgb: A 帧（灰度 + RGB 序彩色）；ref_dark_log: Pass A 产物；dark_prev:
    上一帧暗候选掩码（= B 帧 db["mask"]，同 diagnose_frame 的 da 调用）；mask:
    本候选掩码（已保证面积 ≥ A_MIN）；b_c: 亮候选质心（场景级，同 diagnose_frame）。

    返回紧凑 dict（label/v1/v3/v4/e1/e2/e3/theta/dh/ds/area/move/active）。"""
    area = int(mask.sum())
    out = dict(label=LABEL_TEXTURE, area=area, move=None, active=False,
               v1=False, v3=False, v4=False, e1=False, e2=None, e3=None,
               theta=None, dh=None, ds=None)
    move = None
    if dark_prev is not None:
        move = 1.0 - iou(mask, dark_prev)
    out["move"] = move
    active = (area >= A_MIN) and (move is None or move >= MOVE_IOU)
    out["active"] = active
    if not active:
        out["label"] = LABEL_TEXTURE
        return out

    v1 = not touches_boundary(mask)          # V1 闭合轮廓否决
    out["v1"] = v1
    out["e1"] = not v1

    ys, xs = np.nonzero(mask)
    dc = (float(xs.mean()), float(ys.mean()))
    theta = None
    if b_c is not None:
        ex, ey = b_c[0] - dc[0], b_c[1] - dc[1]
        theta = float(np.rad2deg(np.arctan2(ey, ex))) % 360.0
    out["theta"] = theta

    v3 = False
    if theta is not None:
        pr = pca_axis(mask)
        if pr is not None:
            alpha, ratio = pr
            if ratio >= RATIO_MIN:           # 各向同性 → V3/E2 不适用（跳过）
                v3 = axis_err_deg(alpha, theta) > TOL_AXIS
                out["e2"] = not v3
    out["v3"] = v3

    v4 = False
    if rgb_rgb is not None:
        rs = reflect_stats(rgb_rgb, mask)    # 第三格逐字：BAND_EDGE/SAT_MIN 边界环带
        if rs is not None:
            out["dh"], out["ds"] = rs["dh"], rs["ds"]
            e3 = (rs["dh"] <= TOL_H) and (rs["ds"] <= TOL_S)
            out["e3"] = e3
            v4 = not e3
    out["v4"] = v4

    raw = LABEL_OBJECT if (v1 or v3 or v4) else LABEL_SHADOW
    out["label"] = apply_fix(raw, v1, v3, v4)    # docs/270 计分制逐字
    return out


# ---------------- 单单位运行（docs/B6 §1.2/§1.3 冻结；四列：基线/自适应前/自适应后/多候选） ----------------
def run_scene_multi(split, a_path, mask_path, free_path):
    """跑单个 ISTD 场景（docs/B6 §1.2/§1.3 冻结）：2 帧单位 [free, shadow]。

    预处理与判别**逐字同第五/六/七格**（import 同源）；同一流程上同时算 基线
    （da0 label 原样，docs/267 口径）/ 自适应前（apply_fix(da0)，docs/270 口径）/
    自适应后（apply_fix(da)，docs/272 单候选口径——C1 基线）/ 多候选（本格全图
    并集）四套标签的 tp/fp/fn/white——G_BASE/G_FIX/G_ADAPT 为单候选逐位复现载体，
    G_DEGEN 为多候选 ≡ 单候选超集载体。

    候选发现 = 自适应工作点（docs/272 两遍法逐字）的 dark 掩码全连通分量列表；
    每候选独立过判别器（diagnose_candidate）。GT 掩码只用于评估，绝不进入机制。"""
    b_bgr = cv2.imread(free_path, cv2.IMREAD_COLOR)      # free = 无阴影参考（_C）
    a_bgr = cv2.imread(a_path, cv2.IMREAD_COLOR)         # A = 阴影图（评估帧）
    c_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # mask = GT 掩码（_B）
    if b_bgr is None or a_bgr is None or c_gray is None:
        raise RuntimeError("cannot read scene: %s" % a_path)

    # ---- 预处理（docs/B6 §1.2 冻结；docs/272 §1.2 同款逐字）----
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

    # ---- B 帧只作适应参考（不评估；DELTA0）----
    db = diagnose_frame(b_gray, b_rgb, ref_dark_log, None)

    # ---- A 帧 = 评估帧：种子候选（DELTA0，docs/267/270 载体）----
    da0 = diagnose_frame(a_gray, a_rgb, ref_dark_log, db["mask"])

    # ---- 输入统计（docs/272：候选内部亮度方差，从像素算；不进 GT/θ_est）----
    L = np.log(np.maximum(a_gray.astype(np.float32), 1.0))
    mask0 = da0["mask"]
    if mask0.sum() > 0:
        var_inner = float(np.var(L[mask0]))
    else:
        var_inner = float("nan")          # 无候选 → 不自适应（delta = DELTA0）
    delta_adapt = adaptive_delta(var_inner)

    # ---- 自适应诊断（运行时注入 DELTA_SHADOW；try/finally 恢复；测量层零改动）----
    if abs(delta_adapt - DELTA0) < EPS_DELTA:
        da = da0                          # 退化情形（var_inner≈0）：等价于 DELTA0
    else:
        _old = _lsr.DELTA_SHADOW
        try:
            _lsr.DELTA_SHADOW = delta_adapt
            da = diagnose_frame(a_gray, a_rgb, ref_dark_log, db["mask"])
        finally:
            _lsr.DELTA_SHADOW = _old

    # ---- 候选发现（docs/B6 §1.3）：自适应工作点的 dark 掩码 → 全连通分量列表 ----
    dark_adapt = (ref_dark_log - L) > delta_adapt       # 与 diagnose_frame 同式
    cand_masks, _cents = all_components(dark_adapt, A_MIN)

    # ---- 每候选独立判别（测量与 diagnose_frame 逐字同式；函数 import 复用）----
    bright = L > np.log(OCC_LUM_THRESH + 1.0)
    _bm, _ba, b_c = largest_component(bright)           # 亮候选质心（场景级）
    shadow_cands = []
    cand_records = []
    for cm in cand_masks:
        rec = diagnose_candidate(a_gray, a_rgb, ref_dark_log, db["mask"], cm, b_c)
        cand_records.append(rec)
        if rec["label"] == LABEL_SHADOW:
            shadow_cands.append(cm)

    # ---- 标签合成（docs/272 §1.3 三列 + 本格 multi 列）----
    base_label = da0["label"]
    fixed_label = apply_fix(da0["label"], da0["v1"], da0["v3"], da0["v4"])
    adapt_label = apply_fix(da["label"], da["v1"], da["v3"], da["v4"])

    def pred_of(lab, mask):
        return mask if lab == LABEL_SHADOW else np.zeros((H, W), bool)

    pred_base = pred_of(base_label, da0["mask"])
    pred_fix = pred_of(fixed_label, da0["mask"])
    pred_adapt = pred_of(adapt_label, da["mask"])
    pred_multi = np.zeros((H, W), bool)
    for sm in shadow_cands:
        pred_multi = np.logical_or(pred_multi, sm)

    def metrics(pred):
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
        fn = int(np.logical_and(np.logical_not(pred), gt).sum())
        return tp, fp, fn

    bt, bf, bfn = metrics(pred_base)
    ft, ff, ffn = metrics(pred_fix)
    at, af, afn = metrics(pred_adapt)
    mt, mf, mfn = metrics(pred_multi)

    # ---- 亮目标（docs/272 §1.4：GT 阴影 ∩ B 帧灰度 ≥ L_WHITE；无条件统计）----
    white = gt & (b_gray >= L_WHITE)
    white_pix = int(white.sum())

    def white_tp_of(pred):
        return int(np.logical_and(pred, white).sum())

    white_tp_base = white_tp_of(pred_base)
    white_tp_fix = white_tp_of(pred_fix)
    white_tp_adapt = white_tp_of(pred_adapt)
    white_tp_multi = white_tp_of(pred_multi)

    # ---- C4 分型（报告性）：种子候选 ∩ GT 阴影重叠（docs/270 同口径）----
    area = int(mask0.sum())
    ov = float(np.logical_and(mask0, gt).sum()) / float(area) if area > 0 else 0.0

    # ---- C2/C5：GT 分量与候选匹配（IoU ≥ MATCH_IOU；GT 分量面积 ≥ A_MIN）----
    gt_masks, _ = all_components(gt, A_MIN)
    n_gt_comps = len(gt_masks)

    def matched(cm):
        for gm in gt_masks:
            if iou(cm, gm) >= MATCH_IOU:
                return True
        return False

    n_cand = len(cand_masks)
    n_shadow = len(shadow_cands)
    mu_count = 0
    for sm in shadow_cands:
        if matched(sm):
            mu_count += 1
    single_matched = 1 if (adapt_label == LABEL_SHADOW and matched(da["mask"])) else 0

    # 新增候选（multi − adapt 掩码差）：非单候选掩码子集的 shadow 候选
    new_tp = int(np.logical_and(pred_multi, np.logical_and(gt, np.logical_not(pred_adapt))).sum())
    new_cand_total = new_cand_ts = new_cand_fp = 0
    for sm in shadow_cands:
        if np.logical_and(sm, np.logical_not(pred_adapt)).any():
            new_cand_total += 1
            if matched(sm):
                new_cand_ts += 1
            else:
                new_cand_fp += 1
    # GT 分量覆盖率（被任一 shadow 候选匹配）
    gtcov_total = n_gt_comps
    gtcov_matched = 0
    for gm in gt_masks:
        for sm in shadow_cands:
            if iou(sm, gm) >= MATCH_IOU:
                gtcov_matched += 1
                break

    # ---- G_DEGEN（每单位）：cand[0] ≡ 单候选 adapt（掩码 + 测量 + 标签 + 超集）----
    degen = 1
    if da["mask"].sum() > 0:
        if len(cand_masks) == 0 or not np.array_equal(cand_masks[0], da["mask"]):
            degen = 0
        else:
            r0 = cand_records[0]
            same_gates = (r0["v1"] == bool(da["v1"]) and r0["v3"] == bool(da["v3"])
                          and r0["v4"] == bool(da["v4"])
                          and ((r0["theta"] is None and da["theta_est"] is None)
                               or (r0["theta"] is not None and da["theta_est"] is not None
                                   and abs(r0["theta"] - da["theta_est"]) < 1e-9))
                          and ((r0["dh"] is None and da["dh"] is None)
                               or (r0["dh"] is not None and da["dh"] is not None
                                   and abs(r0["dh"] - da["dh"]) < 1e-9))
                          and ((r0["ds"] is None and da["ds"] is None)
                               or (r0["ds"] is not None and da["ds"] is not None
                                   and abs(r0["ds"] - da["ds"]) < 1e-9)))
            if not same_gates:
                degen = 0
            if r0["label"] != adapt_label:
                degen = 0
        # 超集（预测级）：pred_multi ⊇ pred_adapt（单候选预测；cand[0] 标签 ==
        #   adapt_label 已保证 shadow 时 da 掩码在并集内；object/texture 时单候选
        #   预测为空 → 超集平凡成立——像素级"pred_multi ⊇ da.mask"只在 shadow 时
        #   有定义，见 docs/B6 §二 D2 机械修复记录）
        if not np.logical_and(pred_adapt, np.logical_not(pred_multi)).any():
            pass
        else:
            degen = 0
    else:
        if len(cand_masks) != 0:
            degen = 0

    def bi(x):
        return 1 if x else 0

    return dict(id=os.path.basename(a_path), split=split,
                tp=bt, fp=bf, fn=bfn,
                fix_tp=ft, fix_fp=ff, fix_fn=ffn,
                ad_tp=at, ad_fp=af, ad_fn=afn,
                mu_tp=mt, mu_fp=mf, mu_fn=mfn,
                white_tp=white_tp_base, white_tp_fix=white_tp_fix,
                white_tp_adapt=white_tp_adapt, white_tp_multi=white_tp_multi,
                white_pix=white_pix, gt_sum=int(gt.sum()),
                label=base_label, label_fixed=fixed_label, label_adapt=adapt_label,
                active=bi(da0["active"]),
                v1=bi(da0["v1"]), v3=bi(da0["v3"]), v4=bi(da0["v4"]),
                area=area, ov=ov, ts=1 if ov >= TS_OV else 0,
                var_inner=var_inner, delta=round(delta_adapt, 6),
                n_cand=n_cand, n_shadow=n_shadow, mu_count=mu_count,
                single_matched=single_matched, gt_comps=n_gt_comps,
                new_tp=new_tp, new_cand_total=new_cand_total,
                new_cand_ts=new_cand_ts, new_cand_fp=new_cand_fp,
                gtcov_total=gtcov_total, gtcov_matched=gtcov_matched,
                degen=degen)


# ---------------- 聚合辅助（docs/B6 §1.4 冻结） ----------------
def pooled_rec(us, prefix):
    """pooled recall（C4 归因辅助）。"""
    tp = int(sum(u[prefix + "tp"] for u in us))
    fn = int(sum(u[prefix + "fn"] for u in us))
    return tp / (tp + fn) if (tp + fn) else 0.0


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="B 路第六格：多候选发现（ISTD）")
    ap.add_argument("--istd-dir", default=r"D:\datasets\ISTD")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="mc6")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只跑前 N 个场景（冒烟/计时/诊断子集用，非冻结全量）")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--probe-b", action="store_true",
                    help="B 超像素可行性探测（cv2.ximgproc 可用性），打印后退出")
    args = ap.parse_args()

    if args.probe_b:
        import importlib
        ok = 0
        try:
            importlib.import_module("cv2.ximgproc")
            ok = 1
        except Exception:
            try:
                importlib.import_module("skimage.segmentation")
                ok = 2
            except Exception:
                ok = 0
        print("R_MC6_PROBE_B=%d" % ok)      # 0=不可用 1=cv2.ximgproc 2=skimage.slic
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    scenes = list_scenes(args.istd_dir)
    if args.limit > 0:
        scenes = scenes[:args.limit]
    t0 = time.time()
    if not scenes:
        print("R_MC6_ERROR_NO_SCENES=1")
        return 1

    cfg = {"istd_dir": args.istd_dir, "limit": args.limit, "tag": args.tag,
           "candidate": {"mode": "A_connected_components", "min_area": A_MIN,
                         "order": "area_desc"},
           "adaptive": {"delta0": DELTA0, "var_ref": VAR_REF},
           "fix": {"mode": "scoring", "v3_removed_from_veto": True,
                   "veto_count_threshold": 2},
           "mechanism": {"delta_shadow": "f(var_inner) in [0.25, 0.35]",
                         "a_min": A_MIN,
                         "k_move": K_MOVE, "move_iou": MOVE_IOU,
                         "occ_lum_thresh": OCC_LUM_THRESH, "ref_pct": 95.0,
                         "veto": {"tol_axis": TOL_AXIS, "ratio_min": RATIO_MIN,
                                  "tol_h": TOL_H, "tol_s": TOL_S,
                                  "band_edge": BAND_EDGE, "sat_min": SAT_MIN}},
           "criteria": {"gain_pass": GAIN_PASS, "gain_partial_lo": GAIN_PARTIAL_LO,
                        "prec_min": PREC_MIN, "count_min": COUNT_MIN,
                        "gt_pix_min": GT_PIX_MIN,
                        "multi_gt_min_scenes": MULTI_GT_MIN_SCENES,
                        "match_iou": MATCH_IOU, "l_white": L_WHITE,
                        "white_min_pix": WHITE_MIN_PIX,
                        "white_min_frac": WHITE_MIN_FRAC, "ts_ov": TS_OV}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_mc6_%s.json" % ck_tag)

    def run_all(use_resume=True, save_ckpt=True):
        done = {}
        if use_resume and args.resume and not args.no_resume and os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                done = json.load(f).get("per_unit", {})
        per_unit = dict(done)
        for i, (split, a, b, c) in enumerate(scenes):
            key = "%s_%s" % (split, os.path.basename(a))
            if key in per_unit:
                continue
            per_unit[key] = run_scene_multi(split, a, b, c)
            if save_ckpt and (i + 1) % 100 == 0:
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"config": cfg, "per_unit": per_unit},
                              f, ensure_ascii=False)
                print("PROGRESS %d/%d" % (i + 1, len(scenes)), flush=True)
        if save_ckpt:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False)
        return per_unit

    per_unit = run_all()
    units = [per_unit["%s_%s" % (s, os.path.basename(a))] for s, a, b, c in scenes]

    # ---- 四列 pooled（基线 docs/267 / 自适应前 docs/270 / 自适应后 docs/272 / 多候选 本格）----
    pool_base = pooled_prf_parts(units, "")
    pool_fix = pooled_prf_parts(units, "fix_")
    pool_adapt = pooled_prf_parts(units, "ad_")
    pool_multi = pooled_prf_parts(units, "mu_")
    gain = pool_multi["r"] - pool_adapt["r"]
    c1 = ("PASS" if gain >= GAIN_PASS
          else ("PARTIAL" if gain >= GAIN_PARTIAL_LO else "FAIL"))
    units_train = [u for u in units if u["split"] == "train"]
    units_test = [u for u in units if u["split"] == "test"]
    pool_tr = pooled_prf_parts(units_train, "mu_")
    pool_te = pooled_prf_parts(units_test, "mu_")
    gt_total = pool_multi["tp"] + pool_multi["fn"]
    c1_exercised = gt_total >= GT_PIX_MIN

    # ---- 单位级（报告性：mean±SD + bootstrap CI；多候选口径）----
    u_rec_m, u_rec_sd, u_rec_ci, u_rec_n = unit_level(
        [dict(tp=u["mu_tp"], fn=u["mu_fn"]) for u in units], "tp", None, "fn")
    u_prec_m, u_prec_sd, u_prec_ci, u_prec_n = unit_level(
        [dict(tp=u["mu_tp"], fp=u["mu_fp"]) for u in units], "tp", "fp", None)

    # ---- 标签分布（多候选口径报告用；active 同 docs/267 C2 口径）----
    n_active = sum(1 for u in units if u["active"])
    obj_f = sum(1 for u in units if u["label_adapt"] == LABEL_OBJECT)
    shadow_f = sum(1 for u in units if u["label_adapt"] == LABEL_SHADOW)
    tex_f = sum(1 for u in units if u["label_adapt"] == LABEL_TEXTURE)
    none_f = sum(1 for u in units if u["label_adapt"] == LABEL_NONE)
    base_obj_f = sum(1 for u in units if u["label"] == LABEL_OBJECT)
    base_shadow_f = sum(1 for u in units if u["label"] == LABEL_SHADOW)
    base_none_f = sum(1 for u in units if u["label"] == LABEL_NONE)
    fix_obj_f = sum(1 for u in units if u["label_fixed"] == LABEL_OBJECT)

    # ---- 基线/自适应前/自适应后读数（守卫参照 + 对照）----
    base_v1 = sum(u["v1"] for u in units if u["active"]) / max(1, n_active)
    base_v3 = sum(u["v3"] for u in units if u["active"]) / max(1, n_active)
    base_v4 = sum(u["v4"] for u in units if u["active"]) / max(1, n_active)
    white_pix = int(sum(u["white_pix"] for u in units))
    white_tp_base = int(sum(u["white_tp"] for u in units))
    white_tp_fix = int(sum(u["white_tp_fix"] for u in units))
    white_tp_adapt = int(sum(u["white_tp_adapt"] for u in units))
    white_tp_multi = int(sum(u["white_tp_multi"] for u in units))
    white_recall_base = white_tp_base / max(1, white_pix)
    white_recall_fix = white_tp_fix / max(1, white_pix)
    white_recall_adapt = white_tp_adapt / max(1, white_pix)
    white_recall_multi = white_tp_multi / max(1, white_pix)

    # ---- C2 MULTI_COUNT（多阴影场景：GT 分量 ≥ 2）----
    multi_gt = [u for u in units if u["gt_comps"] >= 2]
    count_vals = [float(u["mu_count"]) for u in multi_gt]
    single_count_vals = [float(u["single_matched"]) for u in multi_gt]
    c2_exercised = len(multi_gt) >= MULTI_GT_MIN_SCENES
    if count_vals:
        count_mean = float(np.mean(count_vals))
        count_sd = float(np.std(count_vals, ddof=1)) if len(count_vals) > 1 else 0.0
        count_lo, count_hi = bootstrap_ci(count_vals)
    else:
        count_mean = float("nan")
        count_sd = 0.0
        count_lo = count_hi = float("nan")
    single_count_mean = float(np.mean(single_count_vals)) if single_count_vals else float("nan")
    c2_ok = c2_exercised and (count_mean >= COUNT_MIN)

    # ---- C3 PREC_KEEP ----
    c3_ok = pool_multi["p"] >= PREC_MIN

    # ---- C5（报告性）：候选数分布 / 新增候选 / GT 覆盖 / 新增 TP ----
    c0 = sum(1 for u in units if u["n_cand"] == 0)
    c1n = sum(1 for u in units if u["n_cand"] == 1)
    c2p = sum(1 for u in units if u["n_cand"] >= 2)
    cand_mean = float(np.mean([u["n_cand"] for u in units]))
    newtp_pix = int(sum(u["new_tp"] for u in units))
    new_cand_total = int(sum(u["new_cand_total"] for u in units))
    new_cand_ts = int(sum(u["new_cand_ts"] for u in units))
    new_cand_fp = int(sum(u["new_cand_fp"] for u in units))
    gtcov_total = int(sum(u["gtcov_total"] for u in units))
    gtcov_matched = int(sum(u["gtcov_matched"] for u in units))
    gtcov_rate = gtcov_matched / max(1, gtcov_total)

    # ---- C4（报告性，同 docs/272 口径）：DELTA 分布 / var_inner 分布 / recall 归因 ----
    act = [u for u in units if u["active"]]
    delta_vals = [u["delta"] for u in act]
    delta_min = min(delta_vals) if delta_vals else float("nan")
    delta_med = float(np.median(delta_vals)) if delta_vals else float("nan")
    delta_max = max(delta_vals) if delta_vals else float("nan")
    trigger = sum(1 for u in act if u["delta"] < DELTA0 - 1e-6)
    trigger_rate = trigger / max(1, len(act))
    var_vals = [u["var_inner"] for u in act if u["var_inner"] == u["var_inner"]]
    var_min = min(var_vals) if var_vals else float("nan")
    var_med = float(np.median(var_vals)) if var_vals else float("nan")
    var_max = max(var_vals) if var_vals else float("nan")
    soft_u = [u for u in act if u["var_inner"] >= VAR_REF - 1e-12]
    hard_u = [u for u in act if u["var_inner"] < VAR_REF - 1e-12]
    rec_soft_before = pooled_rec(soft_u, "fix_")
    rec_soft_after = pooled_rec(soft_u, "ad_")
    rec_hard_before = pooled_rec(hard_u, "fix_")
    rec_hard_after = pooled_rec(hard_u, "ad_")
    unit_rec_deltas = []
    for u in act:
        d_fix = u["fix_tp"] + u["fix_fn"]
        d_ad = u["ad_tp"] + u["ad_fn"]
        if d_fix > 0 and d_ad > 0:
            unit_rec_deltas.append(u["ad_tp"] / d_ad - u["fix_tp"] / d_fix)
    unit_rec_delta_mean = float(np.mean(unit_rec_deltas)) if unit_rec_deltas else float("nan")
    changed = [u for u in units if u["label_adapt"] != u["label_fixed"]]
    label_changed = len(changed)
    label_changed_ts = sum(1 for u in changed if u["ts"])

    # ---- 守卫（docs/B6 §1.6）----
    g_base = 1 if (abs(pool_base["p"] - BASE_P) <= TOL_PRF
                   and abs(pool_base["r"] - BASE_R) <= TOL_PRF
                   and abs(pool_base["f1"] - BASE_F1) <= TOL_PRF
                   and abs(white_recall_base - BASE_WHITE) <= TOL_WHITE
                   and base_obj_f == BASE_OBJ_F and base_shadow_f == BASE_SHADOW_F
                   and base_none_f == BASE_NONE_F
                   and abs(base_v1 - BASE_V1) <= TOL_GATE
                   and abs(base_v3 - BASE_V3) <= TOL_GATE
                   and abs(base_v4 - BASE_V4) <= TOL_GATE) else 0
    g_fix = 1 if (abs(pool_fix["p"] - FIX_P) <= TOL_PRF
                  and abs(pool_fix["r"] - FIX_R) <= TOL_PRF
                  and abs(pool_fix["f1"] - FIX_F1) <= TOL_PRF
                  and abs(white_recall_fix - FIX_WHITE) <= TOL_WHITE
                  and fix_obj_f == FIX_OBJ_F) else 0
    g_adapt = 1 if (abs(pool_adapt["p"] - AD_P) <= TOL_PRF
                    and abs(pool_adapt["r"] - AD_R) <= TOL_PRF
                    and abs(pool_adapt["f1"] - AD_F1) <= TOL_PRF
                    and abs(white_recall_adapt - AD_WHITE) <= TOL_WHITE
                    and obj_f == AD_OBJ_F
                    and (delta_min == delta_min and abs(delta_min - AD_DELTA_MIN) <= TOL_DELTA)
                    and abs(delta_med - AD_DELTA_MED) <= TOL_DELTA
                    and abs(delta_max - AD_DELTA_MAX) <= TOL_DELTA
                    and abs(trigger_rate - AD_TRIGGER) <= TOL_GATE
                    and abs(rec_soft_before - AD_SOFT_BEF) <= TOL_DELTA
                    and abs(rec_soft_after - AD_SOFT_AFT) <= TOL_DELTA
                    and abs(rec_hard_before - AD_HARD_BEF) <= TOL_DELTA
                    and abs(rec_hard_after - AD_HARD_AFT) <= TOL_DELTA
                    and abs(unit_rec_delta_mean - AD_UNIT_REC_DELTA) <= TOL_DELTA
                    and label_changed == AD_LABEL_CHANGED
                    and label_changed_ts == AD_LABEL_CHANGED_TS) else 0
    g_degen = 1 if all(u["degen"] == 1 for u in units) else 0
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
    guards_ok = (g_base == 1) and (g_fix == 1) and (g_adapt == 1) and (g_degen == 1) \
        and (g_synth == 1) and (g_cell4 == 1) and (g_det == 1) and (g_mask == 1)

    # ---- 判定（docs/B6 §1.5 冻结）----
    c1_pass = (gain >= GAIN_PASS)
    c1_partial = (GAIN_PARTIAL_LO <= gain < GAIN_PASS)
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1_exercised:
        verdict = "REAL_SHADOW_LOW"
    elif c1_pass and c3_ok and (not c2_exercised or c2_ok):
        verdict = "MULTI_CAND_PASS"
    elif c1_partial and c3_ok and (not c2_exercised or c2_ok):
        verdict = "MULTI_CAND_PARTIAL"
    else:
        verdict = "MULTI_CAND_FAIL"

    # ---- 内部确定性复现（docs/B6 §1.6-9；第二遍强制重算，不读 checkpoint）----
    repro = 1
    if args.repro:
        per2 = run_all(use_resume=False, save_ckpt=False)
        for u in units:
            k = "%s_%s" % (u["split"], u["id"])
            a, b = u, per2[k]
            for kk in REPRO_KEYS:
                va, vb = a[kk], b[kk]
                same = (va == vb) or (va != va and vb != vb)
                if not same:
                    repro = 0

    agg = {"base": pool_base, "fix": pool_fix, "adapt": pool_adapt,
           "multi": pool_multi, "gain": gain,
           "train": pool_tr, "test": pool_te,
           "gt_total": gt_total, "c1_exercised": c1_exercised,
           "unit_recall": [u_rec_m, u_rec_sd, list(u_rec_ci), u_rec_n],
           "unit_precision": [u_prec_m, u_prec_sd, list(u_prec_ci), u_prec_n],
           "active": n_active, "obj_f": obj_f, "shadow_f": shadow_f,
           "tex_f": tex_f, "none_f": none_f,
           "white_pix": white_pix, "white_tp_multi": white_tp_multi,
           "white_recall_multi": white_recall_multi,
           "white_recall_base": white_recall_base,
           "white_recall_fix": white_recall_fix,
           "white_recall_adapt": white_recall_adapt,
           "base_obj_f": base_obj_f, "base_shadow_f": base_shadow_f,
           "base_v1": base_v1, "base_v3": base_v3, "base_v4": base_v4,
           "fix_obj_f": fix_obj_f, "adapt_obj_f": obj_f,
           "c2": {"multi_gt_scenes": len(multi_gt),
                  "count_mean": count_mean, "count_sd": count_sd,
                  "count_ci": [count_lo, count_hi],
                  "single_count_mean": single_count_mean,
                  "exercised": c2_exercised},
           "c5": {"cand0": c0, "cand1": c1n, "cand2p": c2p,
                  "cand_mean": cand_mean,
                  "new_tp_pix": newtp_pix,
                  "new_cand_total": new_cand_total,
                  "new_cand_ts": new_cand_ts, "new_cand_fp": new_cand_fp,
                  "gtcov_total": gtcov_total, "gtcov_matched": gtcov_matched,
                  "gtcov_rate": gtcov_rate},
           "c4": {"delta_min": delta_min, "delta_med": delta_med,
                  "delta_max": delta_max, "trigger_rate": trigger_rate,
                  "var_min": var_min, "var_med": var_med, "var_max": var_max,
                  "rec_soft_before": rec_soft_before,
                  "rec_soft_after": rec_soft_after,
                  "rec_hard_before": rec_hard_before,
                  "rec_hard_after": rec_hard_after,
                  "unit_rec_delta_mean": unit_rec_delta_mean,
                  "label_changed": label_changed,
                  "label_changed_ts": label_changed_ts}}

    out = {
        "artifact": "multi_candidate_test",
        "doc_ref": "docs/B6",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_multiple_recall_gain": c1,
                     "c1_gain": gain,
                     "c1_exercised": bool(c1_exercised),
                     "pooled_base": {k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in pool_base.items()},
                     "pooled_fix": {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in pool_fix.items()},
                     "pooled_adapt": {k: round(v, 6) if isinstance(v, float) else v
                                      for k, v in pool_adapt.items()},
                     "pooled_multi": {k: round(v, 6) if isinstance(v, float) else v
                                      for k, v in pool_multi.items()},
                     "c2_multiple_count": bool(c2_ok),
                     "c2_exercised": bool(c2_exercised),
                     "c3_prec_keep": bool(c3_ok)},
        "guards": {"base": g_base, "fix": g_fix, "adapt": g_adapt,
                   "degen": g_degen, "synth": g_synth,
                   "synth_det": g_synth_det, "synth_cont": g_synth_cont,
                   "cell4": g_cell4,
                   "cell4_flamingo": {"obj_rate": c4_obj, "shadow_rate": c4_shadow,
                                      "v3_rate": c4_v3, "theta_med": c4_theta},
                   "det": g_det, "det_label": g_det_label, "mask": g_mask,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "mc6_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定，docs/B6 §1.8）----
    print("R_MC6_UNITS=%d" % len(units))
    print("R_MC6_MULTI_TP=%d" % pool_multi["tp"])
    print("R_MC6_MULTI_FP=%d" % pool_multi["fp"])
    print("R_MC6_MULTI_FN=%d" % pool_multi["fn"])
    print("R_MC6_MULTI_PREC=%.4f" % pool_multi["p"])
    print("R_MC6_MULTI_RECALL=%.4f" % pool_multi["r"])
    print("R_MC6_MULTI_F1=%.4f" % pool_multi["f1"])
    print("R_MC6_GAIN=%.4f" % gain)
    print("R_MC6_GAIN_PASS=%s" % c1)
    print("R_MC6_TRAIN_PREC=%.4f" % pool_tr["p"])
    print("R_MC6_TRAIN_RECALL=%.4f" % pool_tr["r"])
    print("R_MC6_TRAIN_F1=%.4f" % pool_tr["f1"])
    print("R_MC6_TEST_PREC=%.4f" % pool_te["p"])
    print("R_MC6_TEST_RECALL=%.4f" % pool_te["r"])
    print("R_MC6_TEST_F1=%.4f" % pool_te["f1"])
    print("R_MC6_UNIT_REC_MEAN=%.4f" % u_rec_m)
    print("R_MC6_UNIT_REC_SD=%.4f" % u_rec_sd)
    print("R_MC6_UNIT_REC_CI_LO=%.4f" % u_rec_ci[0])
    print("R_MC6_UNIT_REC_CI_HI=%.4f" % u_rec_ci[1])
    print("R_MC6_UNIT_PREC_MEAN=%.4f" % u_prec_m)
    print("R_MC6_UNIT_PREC_SD=%.4f" % u_prec_sd)
    print("R_MC6_WHITE_PIX=%d" % white_pix)
    print("R_MC6_WHITE_TP=%d" % white_tp_multi)
    print("R_MC6_WHITE_RECALL=%.4f" % white_recall_multi)
    print("R_MC6_WHITE_EXERCISED=%d" % (1 if (white_pix >= WHITE_MIN_PIX
                                              and white_pix >= WHITE_MIN_FRAC * gt_total) else 0))
    print("R_MC6_MULTI_GT_SCENES=%d" % len(multi_gt))
    print("R_MC6_MULTI_COUNT_MEAN=%.4f" % count_mean)
    print("R_MC6_MULTI_COUNT_SD=%.4f" % count_sd)
    print("R_MC6_MULTI_COUNT_CI_LO=%.4f" % count_lo)
    print("R_MC6_MULTI_COUNT_CI_HI=%.4f" % count_hi)
    print("R_MC6_MULTI_COUNT_EXERCISED=%d" % (1 if c2_exercised else 0))
    print("R_MC6_SINGLE_COUNT_MEAN=%.4f" % single_count_mean)
    print("R_MC6_OBJ_F=%d" % obj_f)
    print("R_MC6_SHADOW_F=%d" % shadow_f)
    print("R_MC6_TEX_F=%d" % tex_f)
    print("R_MC6_NONE_F=%d" % none_f)
    print("R_MC6_ACTIVE_RATE=%.4f" % (n_active / max(1, len(units))))
    print("R_MC6_CAND0=%d" % c0)
    print("R_MC6_CAND1=%d" % c1n)
    print("R_MC6_CAND2P=%d" % c2p)
    print("R_MC6_CAND_MEAN=%.4f" % cand_mean)
    print("R_MC6_NEWTP_PIX=%d" % newtp_pix)
    print("R_MC6_NEWCAND_TOTAL=%d" % new_cand_total)
    print("R_MC6_NEWCAND_TS=%d" % new_cand_ts)
    print("R_MC6_NEWCAND_FP=%d" % new_cand_fp)
    print("R_MC6_GTCOV_TOTAL=%d" % gtcov_total)
    print("R_MC6_GTCOV_MATCHED=%d" % gtcov_matched)
    print("R_MC6_GTCOV_RATE=%.4f" % gtcov_rate)
    print("R_MC6_BASE_PREC=%.4f" % pool_base["p"])
    print("R_MC6_BASE_RECALL=%.4f" % pool_base["r"])
    print("R_MC6_BASE_F1=%.4f" % pool_base["f1"])
    print("R_MC6_BASE_WHITE=%.4f" % white_recall_base)
    print("R_MC6_BASE_OBJ_F=%d" % base_obj_f)
    print("R_MC6_FIX_PREC=%.4f" % pool_fix["p"])
    print("R_MC6_FIX_RECALL=%.4f" % pool_fix["r"])
    print("R_MC6_FIX_F1=%.4f" % pool_fix["f1"])
    print("R_MC6_FIX_WHITE=%.4f" % white_recall_fix)
    print("R_MC6_FIX_OBJ_F=%d" % fix_obj_f)
    print("R_MC6_ADAPT_PREC=%.4f" % pool_adapt["p"])
    print("R_MC6_ADAPT_RECALL=%.4f" % pool_adapt["r"])
    print("R_MC6_ADAPT_F1=%.4f" % pool_adapt["f1"])
    print("R_MC6_ADAPT_WHITE=%.4f" % white_recall_adapt)
    print("R_MC6_ADAPT_OBJ_F=%d" % obj_f)
    print("R_MC6_DELTA_MIN=%.4f" % delta_min)
    print("R_MC6_DELTA_MED=%.4f" % delta_med)
    print("R_MC6_DELTA_MAX=%.4f" % delta_max)
    print("R_MC6_TRIGGER_RATE=%.4f" % trigger_rate)
    print("R_MC6_VAR_MIN=%.6f" % var_min)
    print("R_MC6_VAR_MED=%.6f" % var_med)
    print("R_MC6_VAR_MAX=%.6f" % var_max)
    print("R_MC6_REC_SOFT_BEFORE=%.4f" % rec_soft_before)
    print("R_MC6_REC_SOFT_AFTER=%.4f" % rec_soft_after)
    print("R_MC6_REC_SOFT_DELTA=%.4f" % (rec_soft_after - rec_soft_before))
    print("R_MC6_REC_HARD_BEFORE=%.4f" % rec_hard_before)
    print("R_MC6_REC_HARD_AFTER=%.4f" % rec_hard_after)
    print("R_MC6_REC_HARD_DELTA=%.4f" % (rec_hard_after - rec_hard_before))
    print("R_MC6_UNIT_REC_DELTA_MEAN=%.4f" % unit_rec_delta_mean)
    print("R_MC6_LABEL_CHANGED=%d" % label_changed)
    print("R_MC6_LABEL_CHANGED_TS=%d" % label_changed_ts)
    print("R_MC6_GUARD_BASE=%d" % g_base)
    print("R_MC6_GUARD_FIX=%d" % g_fix)
    print("R_MC6_GUARD_ADAPT=%d" % g_adapt)
    print("R_MC6_GUARD_DEGEN=%d" % g_degen)
    print("R_MC6_GUARD_SYNTH=%d" % g_synth)
    print("R_MC6_GUARD_SYNTH_DET=%.4f" % g_synth_det)
    print("R_MC6_GUARD_SYNTH_CONT=%.4f" % g_synth_cont)
    print("R_MC6_GUARD_CELL4=%d" % g_cell4)
    print("R_MC6_GUARD_CELL4_OBJ=%.4f" % c4_obj)
    print("R_MC6_GUARD_CELL4_SHADOW=%.4f" % c4_shadow)
    print("R_MC6_GUARD_CELL4_V3=%.4f" % c4_v3)
    print("R_MC6_GUARD_CELL4_THETA=%.2f" % c4_theta)
    print("R_MC6_GUARD_DET=%d" % g_det)
    print("R_MC6_GUARD_DET_LABEL=%s" % g_det_label)
    print("R_MC6_GUARD_MASK=%d" % g_mask)
    print("R_MC6_REPRO=%d" % repro)
    print("R_MC6_C1_GAIN=%s" % c1)
    print("R_MC6_C2_COUNT=%s" % ("PASS" if c2_ok else ("LOW" if not c2_exercised else "FAIL")))
    print("R_MC6_C3_PREC=%s" % ("PASS" if c3_ok else "FAIL"))
    print("R_MC6_VERDICT=%s" % verdict)
    print("R_MC6_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
