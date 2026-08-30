"""vision/light_shadow_adaptive.py — 光影判别第七格：单变量自适应
（候选内部亮度方差 → DELTA_SHADOW 参数自适应）
（docs/272 预注册设计，f 形式与系数/判据/守卫冻结；本脚本为唯一新增实验文件，
import 复用第六格 vision/light_shadow_recall_fix.py（apply_fix/pooled_prf_parts，
判别核心）、第五格 vision/light_shadow_real_gt_test.py（list_scenes/unit_level/
guard_cell4/guard_mask）、第四格 vision/light_shadow_real_test.py（compute_ref_dark/
diagnose_frame/circular_median/run_video/guard_synth/guard_demo/LABEL_*）、第一格
vision/light_shadow_test.py、第二格 vision/light_shadow_gate_test.py、第三格
vision/light_shadow_reflect_test.py、vision/real_stream_test.py、
vision/critical_point.py，未修改任何既有脚本）。

目标（docs/272 §1.1）：docs/270 的固定先验（DELTA_SHADOW=0.35 全程不变）改为运行时
自适应——单变量机制（docs/208 范式：f 形式冻结、输入=运行时阅历）：

  DELTA_SHADOW(t) = clamp(DELTA0 − K_ADAPT × var_inner(t)/VAR_REF, DELTA_LO, DELTA_HI)

var_inner = 候选 d_mask 内部 log 亮度方差（从像素算，不进 GT、不进 θ_est 自指）；
硬阴影内部均匀（方差小）→ 门槛保持先验（clamp 上限=DELTA0）；软阴影/半影内部有梯度
（方差大）→ 门槛放宽（召回软阴影浅层边缘 → recall 提升）。系数（DELTA0=0.35 /
VAR_REF=0.040=(log2)^2/12 完整半影方差 / K_ADAPT=0.10 / DELTA_LO=0.25 / DELTA_HI=0.35）
全部物理/几何派生、预注册冻结、运行后不改（改 f 任何系数 = 隐式调参，违规）。

机制 = 两遍法（docs/272 §1.3）：DELTA0 种子候选（da0 = diagnose_frame@DELTA0）→ 测
var_inner → 运行时注入 DELTA_SHADOW=delta_adapt（对 light_shadow_real_test 模块全局，
try/finally 恢复，不修改任何脚本文件）→ 同一条 diagnose_frame 重判（测量层 = 第五/
六格逐字同源，由 GUARD_BASE/GUARD_FIX 位级证明）；标签合成逐字 import docs/270
apply_fix（计分制 v1+v4 ≥ 2 判 object）。每单位三列：基线（da0 label，docs/267 口径）/
自适应前（apply_fix(da0)，docs/270 口径）/ 自适应后（apply_fix(da)，本格结果）。

判据（docs/272 §1.4，冻结）：
  C1 ADAPT_PREC_REC  [L3][机制][真实域证明]：pooled 逐像素 P/R/F1（自适应后），
     P ≥ 0.90 且 R ≥ 0.68 且 F1 ≥ 0.79（三条件 AND）；行使门槛 = GT 像素 ≥ 50000
  C2 WHITE_OBJ_RECALL [L3][机制][真实域证明]：亮目标（GT 阴影 ∩ B 帧 ≥128）pooled
     召回 ≥ 0.65；行使门槛 = 亮目标 ≥ 2000 且 ≥ GT 1%
  C3 KEEP [L3][机制][合成→真实保持]：守卫 R_LA_GUARD_BASE（基线路径 pooled P/R/F1 =
     0.9913/0.4131/0.5832、white = 0.4265、标签 627/1132/111、门率 0.1137/0.1148/
     0.1916 与 docs/267 逐位）+ R_LA_GUARD_FIX（自适应前 fix@DELTA0 pooled P/R/F1 =
     0.9813/0.6382/0.7734、white = 0.6172、obj 24 与 docs/270 逐位）+
     R_LA_GUARD_SYNTH（det/cont = 1.0）+ R_LA_GUARD_CELL4（flamingo 0.7875/0.2125/
     0.7875/287.38 逐位）+ R_LA_GUARD_DET/MASK + R_LA_REPRO
  C4（报告性）：DELTA_SHADOW 分布（min/med/max、触发率）、var_inner 分布、
     recall 提升归因（按 softness 分裂 SOFT/HARD 的 pooled recall 前后 Δ、单位级
     recall Δ 均值、label 变化单位数与 TS/NS 构成）
  判定（docs/272 §1.5，冻结）：守卫全过 且 C1 过 且（C2 未行使 或 过）=
  ADAPT_DELTA_PASS；C1 不过 = ADAPT_DELTA_FAIL（P_FAIL/R_FAIL/F1_FAIL）；C1 过但 C2
  行使不过 = ADAPT_DELTA_FAIL（C2_FAIL）；GT < 50000 = REAL_SHADOW_LOW；守卫不过 =
  GUARD_FAIL。

安全纪律（docs/228/234/235）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_LA_*
摘要块（顺序固定，见 SUMMARY_LINES 注释）；JSON 归档 vision/out/results/la_<tag>.json
+ checkpoint ckpt_la_<hash>.json（--resume 断点续跑，每 100 单位写一次）；数字用
vision/extract_r.py 纯正则抽取；禁止读取 logs/*.log 与 vision/out/results/*.json
原文；ISTD PNG 是数据。

用法：
  python vision/light_shadow_adaptive.py --selftest
  python vision/light_shadow_adaptive.py --tag timing --limit 20
  python vision/light_shadow_adaptive.py --tag diag
  python vision/light_shadow_adaptive.py --tag main --repro
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

import light_shadow_real_test as _lsr  # 运行时注入 DELTA_SHADOW（不修改任何文件）

from critical_point import mean_sd, bootstrap_ci  # noqa: F401  （统计外壳）
from light_shadow_recall_fix import (  # 第六格：判别核心（计分制合成），逐字 import
    apply_fix, pooled_prf_parts,
)
from light_shadow_real_gt_test import (  # 第五格：数据/聚合/守卫，逐字 import 复用
    list_scenes, unit_level, guard_cell4, guard_mask,
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

# ---- f 系数（docs/272 §1.3 冻结；物理/几何派生，预注册冻结，运行后不改）----
DELTA0 = 0.35            # 冻结先验（= docs/260 DELTA_SHADOW）；f 的中心与上限
VAR_REF = 0.040          # 物理锚：完整 0.5× 半影 log 亮度方差 = (log2)^2/12 ≈ 0.0400
K_ADAPT = 0.10           # 最大放宽幅度（= DELTA0 − DELTA_LO；softness=1 恰触下限）
DELTA_LO = 0.25          # 下限（浅至 ~22% 变暗的半影边缘）
DELTA_HI = 0.35          # 上限 = DELTA0（只放宽不收紧）
EPS_DELTA = 1e-9         # delta 与 DELTA0 的等值容差（退化情形跳过冗余第三次诊断）

assert abs(DELTA0 - DELTA_SHADOW) < 1e-9, "DELTA0 must equal frozen prior 0.35"

# ---- 判据口径参数（docs/272 §1.4 冻结；非机制旋钮，先于运行冻结）----
P_MIN = 0.90             # C1 pooled precision 下限（"P 不崩"高杠，同 docs/270）
R_MIN = 0.68             # C1 pooled recall 下限（自适应抬 recall 目标，vs 0.6382）
F1_MIN = 0.79            # C1 pooled F1 下限（vs 0.7734）
GT_PIX_MIN = 50000       # C1 行使门槛：pooled GT 阴影像素
L_WHITE = 128.0          # C2 亮目标定义：参考图 B 灰度 ≥ 128（docs/267 同款）
WHITE_MIN_PIX = 2000     # C2 行使门槛：pooled 亮目标像素
WHITE_MIN_FRAC = 0.01    # C2 行使门槛：亮目标 ≥ GT 阴影像素 1%
WHITE_REC_MIN = 0.65     # C2 亮目标召回下限（vs 0.6172）

# ---- C4 分型口径（docs/272 §1.4 冻结；报告性）----
TS_OV = 0.5              # 候选 ∩ GT 阴影重叠 ≥ 0.5 = 真实阴影单位（TS）

# ---- 守卫冻结参照（docs/267 §3 + docs/270 §3 逐位；±容差）----
BASE_P, BASE_R, BASE_F1 = 0.9913, 0.4131, 0.5832   # R_LA_GUARD_BASE（docs/267）
BASE_WHITE = 0.4265
BASE_OBJ_F, BASE_SHADOW_F, BASE_NONE_F = 627, 1132, 111
BASE_V1, BASE_V3, BASE_V4 = 0.1137, 0.1148, 0.1916
FIX_P, FIX_R, FIX_F1 = 0.9813, 0.6382, 0.7734       # R_LA_GUARD_FIX（docs/270）
FIX_WHITE = 0.6172
FIX_OBJ_F = 24
TOL_PRF = 5e-5           # P/R/F1 容差
TOL_WHITE = 1e-3         # white 容差
TOL_GATE = 1e-4          # 门率容差

# ---- 内部确定性复现键（docs/272 §1.6-7；每单位标量）----
REPRO_KEYS = ["tp", "fp", "fn", "fix_tp", "fix_fp", "fix_fn",
              "ad_tp", "ad_fp", "ad_fn",
              "white_tp", "white_tp_fix", "white_tp_adapt", "white_pix",
              "gt_sum", "label", "label_fixed", "label_adapt",
              "v1", "v3", "v4", "active", "area", "ov", "ts",
              "var_inner", "delta"]


# ---------------- f（docs/272 §1.3 冻结；唯一自由度为输入统计 var_inner） ----------------
def adaptive_delta(var_inner):
    """DELTA_SHADOW(t) = clamp(DELTA0 − K_ADAPT × var_inner/VAR_REF, DELTA_LO,
    DELTA_HI)（docs/272 §1.3 冻结）。var_inner 为 NaN/None（无候选）→ DELTA0。
    单调性：∂DELTA/∂var_inner = −K_ADAPT/VAR_REF < 0 常数负斜率；
    有界性：clamp 保证 [DELTA_LO, DELTA_HI] 对任意 var_inner ∈ [0, ∞)。"""
    if var_inner is None or var_inner != var_inner:   # NaN 感知（无候选）
        return DELTA0
    soft = var_inner / VAR_REF
    d = DELTA0 - K_ADAPT * soft
    if d < DELTA_LO:
        return DELTA_LO
    if d > DELTA_HI:
        return DELTA_HI
    return d


def adaptive_selfcheck():
    """构造冒烟：自适应函数边界测试（docs/272 §1.8 运行序列第一步）。
    断言：方差极端值（0 → 1e6，含 NaN）下 DELTA 在 [DELTA_LO, DELTA_HI] 内、
    单调非增、端值精确（var=0 → DELTA0；var=VAR_REF → DELTA0−K_ADAPT）。"""
    ok = True

    def chk(cond, why):
        nonlocal ok
        if not cond:
            ok = False
            print("R_LA_SELFTEST_FAIL=" + why)

    # 单调性：var_inner 增大 → delta 非增
    vars_ = [0.0, 0.001, 0.005, 0.01, 0.02, VAR_REF, 0.1, 0.5, 5.0]
    deltas = [adaptive_delta(v) for v in vars_]
    for a, b in zip(deltas, deltas[1:]):
        chk(a + 1e-12 >= b, "monotone")
    # 有界性：方差极端值下 DELTA 在 [lo, hi] 内
    for v in [0.0, 1e-9, 0.02, VAR_REF, 10.0, 1e6]:
        d = adaptive_delta(v)
        chk(DELTA_LO - 1e-12 <= d <= DELTA_HI + 1e-12, "bounded@%g" % v)
    # 端值精确
    chk(abs(adaptive_delta(0.0) - DELTA0) <= 1e-12, "end0")
    chk(abs(adaptive_delta(VAR_REF) - (DELTA0 - K_ADAPT)) <= 1e-12, "endref")
    chk(abs(adaptive_delta(VAR_REF * 0.5) - (DELTA0 - K_ADAPT * 0.5)) <= 1e-12,
        "endhalf")
    chk(abs(adaptive_delta(1e6) - DELTA_LO) <= 1e-12, "endlo")
    chk(abs(adaptive_delta(float("nan")) - DELTA0) <= 1e-12, "endnan")
    chk(abs(adaptive_delta(None) - DELTA0) <= 1e-12, "endnone")
    print("R_LA_SELFTEST=%d" % (1 if ok else 0))
    return 0 if ok else 1


# ---------------- 单单位运行（docs/272 §1.2/§1.3 冻结；三列：基线/自适应前/自适应后） ----------------
def run_scene_adaptive(split, a_path, mask_path, free_path):
    """跑单个 ISTD 场景（docs/272 §1.2/§1.3 冻结）：2 帧单位 [free, shadow]。

    预处理与判别**逐字同第五/六格**（import 同源）；同一 diagnose_frame 输出上同时
    算 基线（da0 label 原样）/ 自适应前（apply_fix(da0)，docs/270 口径）/ 自适应后
    （apply_fix(da)，本格结果）三套标签的 tp/fp/fn/white——同代码路径，基线即
    docs/267 逐位复现载体（R_LA_GUARD_BASE）、自适应前即 docs/270 逐位复现载体
    （R_LA_GUARD_FIX）。自适应 = 两遍法：DELTA0 种子候选 → 测 var_inner → 运行时
    注入自适应 DELTA → 同一条 diagnose_frame 重判（测量层零改动，只变参数工作点）。

    返回 per-scene dict（紧凑标量，供聚合/复现）。GT 掩码只用于评估，绝不进入机制。"""
    b_bgr = cv2.imread(free_path, cv2.IMREAD_COLOR)      # free = 无阴影参考（_C）
    a_bgr = cv2.imread(a_path, cv2.IMREAD_COLOR)         # A = 阴影图（评估帧）
    c_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # mask = GT 掩码（_B）
    if b_bgr is None or a_bgr is None or c_gray is None:
        raise RuntimeError("cannot read scene: %s" % a_path)

    # ---- 预处理（docs/272 §1.2 冻结；docs/267 §1.2 同款逐字）----
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

    # ---- B 帧只作适应参考：取暗候选作 A 帧时间门历史（不评估；DELTA0）----
    db = diagnose_frame(b_gray, b_rgb, ref_dark_log, None)

    # ---- A 帧 = 评估帧（有 GT 掩码 C）：种子候选（DELTA0）----
    da0 = diagnose_frame(a_gray, a_rgb, ref_dark_log, db["mask"])

    # ---- 输入统计（docs/272 §1.3：候选内部亮度方差，从像素算；不进 GT/θ_est）----
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

    # ---- 标签合成（docs/272 §1.3；逐字 import docs/270 apply_fix 计分制）----
    base_label = da0["label"]
    fixed_label = apply_fix(da0["label"], da0["v1"], da0["v3"], da0["v4"])
    adapt_label = apply_fix(da["label"], da["v1"], da["v3"], da["v4"])

    def metrics(lab, mask):
        pred = mask if lab == LABEL_SHADOW else np.zeros((H, W), bool)
        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
        fn = int(np.logical_and(np.logical_not(pred), gt).sum())
        return tp, fp, fn

    bt, bf, bfn = metrics(base_label, da0["mask"])
    ft, ff, ffn = metrics(fixed_label, da0["mask"])
    at, af, afn = metrics(adapt_label, da["mask"])

    # ---- C2 亮目标（docs/272 §1.4：GT 阴影 ∩ B 帧灰度 ≥ L_WHITE；无条件统计）----
    white = gt & (b_gray >= L_WHITE)
    white_pix = int(white.sum())

    def white_tp_of(lab, mask):
        return int(np.logical_and(mask if lab == LABEL_SHADOW else
                                  np.zeros((H, W), bool), white).sum())

    white_tp_base = white_tp_of(base_label, da0["mask"])
    white_tp_fix = white_tp_of(fixed_label, da0["mask"])
    white_tp_adapt = white_tp_of(adapt_label, da["mask"])

    # ---- C4 分型（报告性）：候选 ∩ GT 阴影重叠（种子候选口径，同 docs/270）----
    area = int(mask0.sum())
    ov = float(np.logical_and(mask0, gt).sum()) / float(area) if area > 0 else 0.0

    def bi(x):
        return 1 if x else 0

    return dict(id=os.path.basename(a_path), split=split,
                tp=bt, fp=bf, fn=bfn,
                fix_tp=ft, fix_fp=ff, fix_fn=ffn,
                ad_tp=at, ad_fp=af, ad_fn=afn,
                white_tp=white_tp_base, white_tp_fix=white_tp_fix,
                white_tp_adapt=white_tp_adapt, white_pix=white_pix,
                gt_sum=int(gt.sum()),
                label=base_label, label_fixed=fixed_label, label_adapt=adapt_label,
                active=bi(da0["active"]),
                v1=bi(da0["v1"]), v3=bi(da0["v3"]), v4=bi(da0["v4"]),
                area=area, ov=ov, ts=1 if ov >= TS_OV else 0,
                var_inner=var_inner,
                delta=round(delta_adapt, 6))


# ---------------- 聚合（docs/272 §1.4 冻结：C1/C2 pooled；基线/自适应前/自适应后三列） ----------------
def pooled_rec(us, prefix):
    """pooled recall（C4 归因辅助）。"""
    tp = int(sum(u[prefix + "tp"] for u in us))
    fn = int(sum(u[prefix + "fn"] for u in us))
    return tp / (tp + fn) if (tp + fn) else 0.0


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="光影判别第七格：单变量自适应（ISTD）")
    ap.add_argument("--istd-dir", default=r"D:\datasets\ISTD")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="la")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只跑前 N 个场景（冒烟/计时用，非冻结全量）")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return adaptive_selfcheck()

    os.makedirs(args.out_dir, exist_ok=True)
    scenes = list_scenes(args.istd_dir)
    if args.limit > 0:
        scenes = scenes[:args.limit]
    t0 = time.time()
    if not scenes:
        print("R_LA_ERROR_NO_SCENES=1")
        return 1

    cfg = {"istd_dir": args.istd_dir, "limit": args.limit, "tag": args.tag,
           "adaptive": {"delta0": DELTA0, "var_ref": VAR_REF, "k_adapt": K_ADAPT,
                        "delta_lo": DELTA_LO, "delta_hi": DELTA_HI},
           "fix": {"mode": "scoring", "v3_removed_from_veto": True,
                   "veto_count_threshold": 2},
           "mechanism": {"delta_shadow": "f(var_inner) in [0.25, 0.35]",
                         "a_min": A_MIN,
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
    ckpt_path = os.path.join(args.out_dir, "ckpt_la_%s.json" % ck_tag)

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
            per_unit[key] = run_scene_adaptive(split, a, b, c)
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

    # ---- 三列 pooled（基线 docs/267 / 自适应前 docs/270 / 自适应后 本格）----
    pool_base = pooled_prf_parts(units, "")
    pool_fix = pooled_prf_parts(units, "fix_")
    pool_adapt = pooled_prf_parts(units, "ad_")
    units_train = [u for u in units if u["split"] == "train"]
    units_test = [u for u in units if u["split"] == "test"]
    pool_tr = pooled_prf_parts(units_train, "ad_")
    pool_te = pooled_prf_parts(units_test, "ad_")
    gt_total = pool_adapt["tp"] + pool_adapt["fn"]
    c1_exercised = gt_total >= GT_PIX_MIN
    c1 = (pool_adapt["p"] >= P_MIN) and (pool_adapt["r"] >= R_MIN) \
        and (pool_adapt["f1"] >= F1_MIN)

    # ---- 单位级（报告性：mean±SD + bootstrap CI；自适应后口径）----
    u_rec_m, u_rec_sd, u_rec_ci, u_rec_n = unit_level(
        [dict(tp=u["ad_tp"], fn=u["ad_fn"]) for u in units], "tp", None, "fn")
    u_prec_m, u_prec_sd, u_prec_ci, u_prec_n = unit_level(
        [dict(tp=u["ad_tp"], fp=u["ad_fp"]) for u in units], "tp", "fp", None)

    # ---- 标签分布（自适应后口径；docs/267 C2 口径：active = label ∈ {object, shadow}）----
    n_active = sum(1 for u in units if u["active"])
    obj_f = sum(1 for u in units if u["label_adapt"] == LABEL_OBJECT)
    shadow_f = sum(1 for u in units if u["label_adapt"] == LABEL_SHADOW)
    tex_f = sum(1 for u in units if u["label_adapt"] == LABEL_TEXTURE)
    none_f = sum(1 for u in units if u["label_adapt"] == LABEL_NONE)
    base_obj_f = sum(1 for u in units if u["label"] == LABEL_OBJECT)
    base_shadow_f = sum(1 for u in units if u["label"] == LABEL_SHADOW)
    base_none_f = sum(1 for u in units if u["label"] == LABEL_NONE)
    fix_obj_f = sum(1 for u in units if u["label_fixed"] == LABEL_OBJECT)

    # ---- 基线/自适应前读数（守卫参照 + 对照）----
    base_v1 = sum(u["v1"] for u in units if u["active"]) / max(1, n_active)
    base_v3 = sum(u["v3"] for u in units if u["active"]) / max(1, n_active)
    base_v4 = sum(u["v4"] for u in units if u["active"]) / max(1, n_active)
    white_pix = int(sum(u["white_pix"] for u in units))
    white_tp_base = int(sum(u["white_tp"] for u in units))
    white_tp_fix = int(sum(u["white_tp_fix"] for u in units))
    white_tp_adapt = int(sum(u["white_tp_adapt"] for u in units))
    white_recall_base = white_tp_base / max(1, white_pix)
    white_recall_fix = white_tp_fix / max(1, white_pix)
    white_recall = white_tp_adapt / max(1, white_pix)
    c2_exercised = (white_pix >= WHITE_MIN_PIX) and (white_pix >= WHITE_MIN_FRAC * gt_total)
    c2 = c2_exercised and (white_recall >= WHITE_REC_MIN)

    # ---- C4（报告性）：DELTA 分布 / var_inner 分布 / recall 归因 ----
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

    # recall 归因：SOFT（softness ≥ 1，delta 触底，完整放宽）vs HARD
    soft_u = [u for u in act if u["var_inner"] >= VAR_REF - 1e-12]
    hard_u = [u for u in act if u["var_inner"] < VAR_REF - 1e-12]
    rec_soft_before = pooled_rec(soft_u, "fix_")
    rec_soft_after = pooled_rec(soft_u, "ad_")
    rec_hard_before = pooled_rec(hard_u, "fix_")
    rec_hard_after = pooled_rec(hard_u, "ad_")
    # 单位级 recall Δ 均值（active 单位；分母 > 0）
    unit_rec_deltas = []
    for u in act:
        d_fix = u["fix_tp"] + u["fix_fn"]
        d_ad = u["ad_tp"] + u["ad_fn"]
        if d_fix > 0 and d_ad > 0:
            unit_rec_deltas.append(u["ad_tp"] / d_ad - u["fix_tp"] / d_fix)
    unit_rec_delta_mean = float(np.mean(unit_rec_deltas)) if unit_rec_deltas else float("nan")
    # label 变化（fix→adapt）
    changed = [u for u in units if u["label_adapt"] != u["label_fixed"]]
    label_changed = len(changed)
    label_changed_ts = sum(1 for u in changed if u["ts"])

    # ---- 守卫（docs/272 §1.6）----
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
    guards_ok = (g_base == 1) and (g_fix == 1) and (g_synth == 1) \
        and (g_cell4 == 1) and (g_det == 1) and (g_mask == 1)

    # ---- 判定（docs/272 §1.5 冻结）----
    if not guards_ok:
        verdict = "GUARD_FAIL"
    elif not c1_exercised:
        verdict = "REAL_SHADOW_LOW"
    elif c1 and (not c2_exercised or c2):
        verdict = "ADAPT_DELTA_PASS"
    else:
        verdict = "ADAPT_DELTA_FAIL"

    # ---- 内部确定性复现（docs/272 §1.6-7；第二遍强制重算，不读 checkpoint）----
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
           "train": pool_tr, "test": pool_te,
           "gt_total": gt_total, "c1_exercised": c1_exercised,
           "unit_recall": [u_rec_m, u_rec_sd, list(u_rec_ci), u_rec_n],
           "unit_precision": [u_prec_m, u_prec_sd, list(u_prec_ci), u_prec_n],
           "active": n_active, "obj_f": obj_f, "shadow_f": shadow_f,
           "tex_f": tex_f, "none_f": none_f,
           "white_pix": white_pix, "white_tp": white_tp_adapt,
           "white_recall": white_recall, "c2_exercised": c2_exercised,
           "white_recall_base": white_recall_base,
           "white_recall_fix": white_recall_fix,
           "base_obj_f": base_obj_f, "base_shadow_f": base_shadow_f,
           "base_v1": base_v1, "base_v3": base_v3, "base_v4": base_v4,
           "fix_obj_f": fix_obj_f,
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
        "artifact": "light_shadow_adaptive",
        "doc_ref": "docs/272",
        "config": cfg,
        "per_unit": per_unit,
        "aggregate": agg,
        "criteria": {"c1_adapt_prec_rec": bool(c1),
                     "c1_exercised": bool(c1_exercised),
                     "pooled_base": {k: round(v, 6) if isinstance(v, float) else v
                                     for k, v in pool_base.items()},
                     "pooled_fix": {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in pool_fix.items()},
                     "pooled_adapt": {k: round(v, 6) if isinstance(v, float) else v
                                      for k, v in pool_adapt.items()},
                     "c2_white_obj_recall": bool(c2),
                     "c2_exercised": bool(c2_exercised),
                     "white_recall": white_recall},
        "guards": {"base": g_base, "fix": g_fix,
                   "synth": g_synth,
                   "synth_det": g_synth_det, "synth_cont": g_synth_cont,
                   "cell4": g_cell4,
                   "cell4_flamingo": {"obj_rate": c4_obj, "shadow_rate": c4_shadow,
                                      "v3_rate": c4_v3, "theta_med": c4_theta},
                   "det": g_det, "det_label": g_det_label, "mask": g_mask,
                   "repro": repro},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "la_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定，docs/272 §1.8）----
    print("R_LA_UNITS=%d" % len(units))
    print("R_LA_ADAPT_TP=%d" % pool_adapt["tp"])
    print("R_LA_ADAPT_FP=%d" % pool_adapt["fp"])
    print("R_LA_ADAPT_FN=%d" % pool_adapt["fn"])
    print("R_LA_ADAPT_PREC=%.4f" % pool_adapt["p"])
    print("R_LA_ADAPT_RECALL=%.4f" % pool_adapt["r"])
    print("R_LA_ADAPT_F1=%.4f" % pool_adapt["f1"])
    print("R_LA_TRAIN_PREC=%.4f" % pool_tr["p"])
    print("R_LA_TRAIN_RECALL=%.4f" % pool_tr["r"])
    print("R_LA_TRAIN_F1=%.4f" % pool_tr["f1"])
    print("R_LA_TEST_PREC=%.4f" % pool_te["p"])
    print("R_LA_TEST_RECALL=%.4f" % pool_te["r"])
    print("R_LA_TEST_F1=%.4f" % pool_te["f1"])
    print("R_LA_UNIT_REC_MEAN=%.4f" % u_rec_m)
    print("R_LA_UNIT_REC_SD=%.4f" % u_rec_sd)
    print("R_LA_UNIT_REC_CI_LO=%.4f" % u_rec_ci[0])
    print("R_LA_UNIT_REC_CI_HI=%.4f" % u_rec_ci[1])
    print("R_LA_UNIT_PREC_MEAN=%.4f" % u_prec_m)
    print("R_LA_UNIT_PREC_SD=%.4f" % u_prec_sd)
    print("R_LA_WHITE_PIX=%d" % white_pix)
    print("R_LA_WHITE_TP=%d" % white_tp_adapt)
    print("R_LA_WHITE_RECALL=%.4f" % white_recall)
    print("R_LA_WHITE_EXERCISED=%d" % (1 if c2_exercised else 0))
    print("R_LA_OBJ_F=%d" % obj_f)
    print("R_LA_SHADOW_F=%d" % shadow_f)
    print("R_LA_TEX_F=%d" % tex_f)
    print("R_LA_NONE_F=%d" % none_f)
    print("R_LA_ACTIVE_RATE=%.4f" % (n_active / max(1, len(units))))
    print("R_LA_DELTA_MIN=%.4f" % delta_min)
    print("R_LA_DELTA_MED=%.4f" % delta_med)
    print("R_LA_DELTA_MAX=%.4f" % delta_max)
    print("R_LA_TRIGGER_RATE=%.4f" % trigger_rate)
    print("R_LA_VAR_MIN=%.6f" % var_min)
    print("R_LA_VAR_MED=%.6f" % var_med)
    print("R_LA_VAR_MAX=%.6f" % var_max)
    print("R_LA_REC_SOFT_BEFORE=%.4f" % rec_soft_before)
    print("R_LA_REC_SOFT_AFTER=%.4f" % rec_soft_after)
    print("R_LA_REC_SOFT_DELTA=%.4f" % (rec_soft_after - rec_soft_before))
    print("R_LA_REC_HARD_BEFORE=%.4f" % rec_hard_before)
    print("R_LA_REC_HARD_AFTER=%.4f" % rec_hard_after)
    print("R_LA_REC_HARD_DELTA=%.4f" % (rec_hard_after - rec_hard_before))
    print("R_LA_UNIT_REC_DELTA_MEAN=%.4f" % unit_rec_delta_mean)
    print("R_LA_LABEL_CHANGED=%d" % label_changed)
    print("R_LA_LABEL_CHANGED_TS=%d" % label_changed_ts)
    print("R_LA_BASE_PREC=%.4f" % pool_base["p"])
    print("R_LA_BASE_RECALL=%.4f" % pool_base["r"])
    print("R_LA_BASE_F1=%.4f" % pool_base["f1"])
    print("R_LA_BASE_WHITE=%.4f" % white_recall_base)
    print("R_LA_BASE_OBJ_F=%d" % base_obj_f)
    print("R_LA_FIX_PREC=%.4f" % pool_fix["p"])
    print("R_LA_FIX_RECALL=%.4f" % pool_fix["r"])
    print("R_LA_FIX_F1=%.4f" % pool_fix["f1"])
    print("R_LA_FIX_WHITE=%.4f" % white_recall_fix)
    print("R_LA_FIX_OBJ_F=%d" % fix_obj_f)
    print("R_LA_GUARD_BASE=%d" % g_base)
    print("R_LA_GUARD_FIX=%d" % g_fix)
    print("R_LA_GUARD_SYNTH=%d" % g_synth)
    print("R_LA_GUARD_SYNTH_DET=%.4f" % g_synth_det)
    print("R_LA_GUARD_SYNTH_CONT=%.4f" % g_synth_cont)
    print("R_LA_GUARD_CELL4=%d" % g_cell4)
    print("R_LA_GUARD_CELL4_OBJ=%.4f" % c4_obj)
    print("R_LA_GUARD_CELL4_SHADOW=%.4f" % c4_shadow)
    print("R_LA_GUARD_CELL4_V3=%.4f" % c4_v3)
    print("R_LA_GUARD_CELL4_THETA=%.2f" % c4_theta)
    print("R_LA_GUARD_DET=%d" % g_det)
    print("R_LA_GUARD_DET_LABEL=%s" % g_det_label)
    print("R_LA_GUARD_MASK=%d" % g_mask)
    print("R_LA_REPRO=%d" % repro)
    print("R_LA_C1_PREC_REC=%s" % ("PASS" if c1 else "FAIL"))
    print("R_LA_C2_WHITE=%s" % ("PASS" if c2 else ("LOW" if not c2_exercised else "FAIL")))
    print("R_LA_VERDICT=%s" % verdict)
    print("R_LA_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
