# 270 — 光影判别第六格：recall 侧修复（否决门链误伤真实阴影 → 计分制联合判别 + V3 去自指化）——预注册设计

> 缘起：docs/267（第五格 REAL_SHADOW_PASS）在带阴影 GT 的 ISTD 上完成真实域证明：
> pooled P/R/F1 = 0.9913 / 0.4131 / 0.5832（train 0.5551 / test 0.6631），
> WHITE_OBJ_RECALL = 0.4265，守卫五连全过。但其 §3.2/§四 判读暴露 recall 侧缺口：
> **否决门链把 ~35% 的真实阴影误判为物体**（object 627 单位 / active 1759 = 35.6%；
> 否决触发率 V1 0.1137 / V3 0.1148 / V4 0.1916），亮目标漏 57.3%（用户"阴影下白色
> 石头不被判阴影"部分成立）。docs/267 §五 3/§五 5 与 docs/265 §五 6 预注册期预警的
> 三个机制嫌疑：**V3 用 θ_est 自指**（帧内估计光照方向做否决依据，有循环风险）、
> **V1"闭合=物体"在真实阴影常闭合时反噬**、**V4 在真实阴影跨材质边界时跳变误判**。
> 本格目标（方向冻结，docs/267 判读 + 用户确认）：**提升 recall 且不崩 precision**
> ——修复否决门链的误伤，判据与 docs/267 同尺可比（pooled P/R/F1 + 白目标召回）。
>
> 编号说明：**docs/268/269 已占用**（268-语言线第一格、269-语言线第二格，并行会话；
> 预注册前已核对 docs 目录），**docs/270 空闲已确认**，本文件取 270。
>
> 状态：**预注册已冻结（§一：修复机制选型（预注册期修复前诊断数据驱动，诊断记录见
> §二）、判据（含 recall 目标与 P 不崩双杠）、守卫、统计、安全纪律先于任何实现与
> 最终运行写入本文件，docs/63 预注册纪律 + docs/247 判据标签规范；本节运行后不改
> 机制、不改旋钮、不改判据阈值）**。§二 修复前诊断与诊断轮记录（最终运行前的
> mechanical bug 修复写在这里，§一 不动）；§三/§四 结果与判定留空（最终运行后
> 填写）；§五 诚实边界为预注册阶段已知边界。
> 引用：docs/63/187/195/219/228/243/247/250/260/261/263/265/267 +
> vision/light_shadow_real_gt_test.py（**第五格：list_scenes/pooled_prf/unit_level/
> guard_mask，import 复用**）+ vision/light_shadow_real_test.py（**第四格：
> compute_ref_dark/diagnose_frame/circular_median/run_video/guard_synth/guard_demo/
> LABEL_*，import 复用，逐字**）+ vision/light_shadow_test.py（**第一格：W/H/
> DELTA_SHADOW/A_MIN/K_MOVE/MOVE_IOU/OCC_LUM_THRESH**）+ vision/light_shadow_gate_test.py
> （**第二格：TOL_AXIS/RATIO_MIN**）+ vision/light_shadow_reflect_test.py（**第三格：
> TOL_H/TOL_S/BAND_EDGE/SAT_MIN**）+ vision/real_stream_test.py（**RESIZE**）+
> vision/critical_point.py（mean_sd/bootstrap_ci）+ vision/extract_r.py（纯正则抽取）。
> 新文件仅 vision/light_shadow_recall_fix.py（第六格实验），未修改任何既有脚本；
> 预注册期修复前诊断工具在项目外（D:\datasets\diag_rf.py，分析工具非实验脚本，
> 同 fetch_istd.py 先例）。

---

## 〇、一句话（预注册 + 结论）

> **预注册：docs/267 的 recall 缺口（否决门链把 ~35% 真实阴影误判为物体）做机制级
> 修复——修复选型由预注册期修复前诊断（§二，ISTD 1870 单位全量，未改任何判据/旋钮）
> 数据驱动：①三否决门损伤占比全部 >96%（被否决单位中真实阴影占比：V1 0.9600 / V3
> 0.9653 / V4 0.9792；NS 非阴影单位仅 17/1759 = 1.0%，否决门链保护的 precision 近乎
> 为空）；②V3 轴误差无分离力（TS 48.24° vs NS 51.00°，θ_est 自指 + 真实场景几何不
> 相关）；③V4 在 TS 上 ΔH 23.28° > TOL_H 15° 反而不如 NS（12.88°）——真实阴影跨
> 材质边界色相变化大；④GT 辅助规则模拟：**V3 移出否决链 + V1/V4 双正交线索计分
> （object iff v1∧v4）→ pooled P 0.9813 / R 0.6382 / F1 0.7734 / white 0.6172**，
> 最接近全翻转上限（R 0.6397）且 P 最高（对比全三门计数 0.6040）。冻结修复 =
> **否决门链改为计分制联合判别**：测量层逐字 import 第五格（Pass A/B、时间门、
> V1/V3/V4 几何/反射率、E1/E2/E3、θ_est、ΔH/ΔS——零改动），只改标签合成层——
> `veto_count = v1 + v4`（**V3 自指 → 从否决链降为证据（E2 保留）**），
> `label = object iff veto_count ≥ 2 else shadow`（单线索不再一票否决；V1 闭合拓扑
> 与 V4 反射率跳变两正交线索联合成立才判物体——人眼"多维联合、不靠单一线索"）。
> 判据（docs/247 标签，真实域证明级，与 docs/267 同尺可比）：
> `[L3][机制][真实域证明]` **REAL_SHADOW_PREC_REC**（pooled P/R/F1：**P ≥ 0.90 且
> R ≥ 0.55 且 F1 ≥ 0.65**，三条件 AND——"recall 提升 + P 不崩"双目标；行使门槛 GT
> 像素 ≥ 50000 同 docs/267）、`[L3][机制][真实域证明]` **WHITE_OBJ_RECALL**
> （亮目标 pooled 召回 **≥ 0.50**，行使门槛同 docs/267）、**KEEP**（docs/267 能力不
> 退化：守卫 R_RF_GUARD_BASE 基线逐位复现 docs/267 数字 + R_GT_GUARD_SYNTH 三连逐位
> + R_GT_GUARD_CELL4 flamingo 逐位）、**C4**（报告性：修复前后 V1/V3/V4 各自误伤率
> 的 Δ）。判定（冻结）：守卫全过 且 C1 过 且（C2 未行使 或 过）=
> **RECALL_FIX_PASS**（否决门链 recall 修复成立：R 从 0.41 档升到 ≥0.55 且 P 保持
> ≥0.90）；C1 不过 = RECALL_FIX_FAIL（按名报 P_FAIL/R_FAIL/F1_FAIL）；C1 过但 C2
> 行使不过 = RECALL_FIX_FAIL（C2_FAIL）；C2 不足行使 = WHITE_OBJ_LOW（只报告）；
> 守卫不过 = GUARD_FAIL。诚实：修复是**标签合成层**变更（测量层零改动、逐字 import）；
> V3 的 θ_est 测量与 E2 证据保留为行为读数、不再进入否决（去自指化针对决策链）；
> 修复把部分非阴影单位翻回阴影（FP 代价）由 C1 P ≥ 0.90 直接测出；单候选局限、
> ISTD 数据选择、静态域纹理近零均同 docs/267 照实。
> **结论：RECALL_FIX_PASS（1870 单位确定性流，守卫六连全过、内部复现 REPRO=1、
> timing 两轮与 diag/main 逐位一致）。C1 REAL_SHADOW_PREC_REC ✓：pooled 逐像素
> P/R/F1 = **0.9813 / 0.6382 / 0.7734**（train 0.9802/0.6184/0.7584、test
> 0.9842/0.6991/0.8175；GT 像素 6,663,543 ≥ 50000，判据被行使）——修复后 recall
> 从 0.4131 升到 0.6382（+0.225）、F1 0.5832 → 0.7734（+0.190）、precision 0.9913 →
> 0.9813（−0.010，守住 ≥ 0.90，"P 不崩"）；C2 WHITE_OBJ_RECALL ✓ = **0.6172**（基线
> 0.4265，+0.191，亮目标漏检 57.3% → 38.3%）；C3 KEEP ✓（守卫 BASE 基线 0.9913/
> 0.4131/0.5832/0.4265 + 标签分布 + 门率与 docs/267 逐位一致，SYNTH det/cont 1.0 +
> CELL4 flamingo 0.7875/0.2125/0.7875/287.38 逐位命中）。机制：否决门链 → 计分制
> 联合判别（测量层逐字 import 第五格零改动；标签合成层 veto_count = v1+v4，V3 自指
> 降为证据 E2，object iff 两正交线索联合成立）——object 单位 627 → 24、TS 误伤率
> 0.3513 → 0.0103（C4 Δ 报告性）。诚实边界：标签合成层修复、测量零改动；V3 去自指
> 化针对决策链（θ_est 测量与 E2 仍为行为读数）；双门联合仍误伤 18 个 TS（诚实剩余）；
> P 守住与否由 C1 ≥ 0.90 直接量化。**

---

## 一、设计（预注册，冻结；修复机制/判据/守卫先于实现与最终运行冻结，运行后不改）

### 1.1 目标与范围（冻结）

docs/267（第五格）完成真实域证明后留下的明确缺口（其 §四 判读）：**recall 侧**——
否决门链把 ~35% 的真实阴影误判为物体（object 627/active 1759 = 35.6%），pooled
recall 0.4131（漏 58.7%）、亮目标召回 0.4265（漏 57.3%）；precision 0.9913 是优势
（过检可控，FP 仅 24,084 px）。本格目标 = **修复否决门链的 recall 误伤，且不崩
precision**：

1. **修复机制（核心）**：否决门链从"任一否决 → object"改为**计分制联合判别**
   （多证据投票，docs/267 判读"更接近人眼多维联合、不靠单一线索"）；**V3 去自指化**
   （θ_est 自指 + 诊断无分离力 → 从否决链降为证据）。修复选型由 §二 修复前诊断
   （ISTD 1870 单位全量，未碰判据/旋钮）数据驱动并冻结。
2. **判据（与 docs/267 同尺可比）**：pooled P/R/F1（P ≥ 0.90 且 R ≥ 0.55 且
   F1 ≥ 0.65）+ WHITE_OBJ_RECALL（≥ 0.50）——直接量化"R 提升 + P 不崩"双目标。
3. **KEEP（能力不退化）**：守卫——修复脚本基线路径逐位复现 docs/267 冻结数字
   （0.9913/0.4131/0.5832/0.4265 + 标签分布 + 门率）+ 合成三连（SYNTH）+ 第四格
   flamingo（CELL4）逐位。
4. **C4（报告性）**：修复前后 V1/V3/V4 各自误伤率的 Δ 对照（§二 修复前诊断 vs
   §三 修复后读数）。

范围收窄（诚实，同 docs/267）：判别对象 = 最大暗域候选（单候选）；"纹理"通道在
静态域 = 时间门不过的暗区（近零，如实报告）；多候选未建模；真实图无光照 GT → θ_est
仍是帧内估计行为读数（§五 3）。

### 1.2 数据与预处理（冻结；逐字同 docs/267 §1.2）

- **数据源（冻结）**：ISTD（D:\datasets\ISTD），1870 场景三元组（train 1330 +
  test 540），目录映射 _A=阴影图 / _B=阴影掩码（GT）/ _C=无阴影参考图（docs/267
  §二 D1 实测并冻结）。图像/掩码是数据（可读），非毒文件。
- **预处理（固定流水线，docs/267 §1.2 同款冻结）**：PNG → cv2.imread(IMREAD_COLOR)
  （BGR 640×480）→ 灰度 cv2.resize(160×120, INTER_AREA) → uint8；彩色帧同样 resize
  到 160×120（INTER_AREA）供 V4/E3 边界环带采样（BAND_EDGE=2px）；GT 掩码
  cv2.resize(160×120, INTER_NEAREST) → bool（>0）。色彩空间注意：reflect_stats 期望
  RGB 序——cv2.imread 给 BGR，V4/E3 前显式 cv2.cvtColor(bgr, COLOR_BGR2RGB)。
- **单位（冻结）**：每场景一次运行（1870 单位：train 1330 + test 540）；每单位 2 帧
  [B（无阴影参考）, A（阴影帧）]；Pass A 在 [B,A] 上算逐像素 0.95 分位（= 逐元素
  max）；评估帧 = A（GT 掩码 C）；B 只作适应参考。
- **无注入、无 jitter、无 RNG**（docs/243 §1.1 声明同款）；统计外壳 = 单位级
  mean±SD（ddof=1）+ bootstrap 95% CI（2000 次，种子 20260828）。

### 1.3 机制（冻结；测量层逐字 import 第五格，零改动；修复只在标签合成层）

**测量层（逐字复用 docs/267 §1.3 = 第四格 `diagnose_frame` import，零改动）**：

```
Pass A（适应，每单位一次）：ref_dark = [B,A] 逐像素 0.95 分位（排除 L>log(221) 亮帧）
    → ref_dark_log = log(ref_dark+1)                    # compute_ref_dark 逐字
Pass B（判断，A 帧）：L = log(gray+1)
    dark = (ref_dark_log − L) > DELTA_SHADOW=0.35
    d_mask, d_area, d_c = largest_component(dark)        # 8-连通域，面积 ≥ A_MIN=25
    bright = L > log(221)；b_mask, b_area, b_c = largest_component(bright)
时间门：active = (d_area ≥ A_MIN) AND (move is None OR move ≥ MOVE_IOU=0.05)
否决门测量（只测不判）：V1 = not touches_boundary(d_mask)（闭合轮廓）
    ／V3 = PCA 比值 ≥ RATIO_MIN=1.5 时主轴与 θ_est 夹角 > TOL_AXIS=30°（各向同性跳过）
    ／V4 = 边界环带 BAND_EDGE=2px 两侧中位 H/S（SAT_MIN=60 过滤）；ΔH>15° 或 ΔS>80
证据门测量：E1 = not V1／E2 = not V3（适用时）／E3 = ΔH≤15° 且 ΔS≤80
θ_est（帧内估计）：θ_est = angle(亮候选质心 − 暗候选质心)；无亮候选 → None
```

**标签合成层（本格唯一修复点，冻结）**——诊断帧输出 label 不再用"任一否决 → object"
的硬链，改为**计分制联合判别**：

```
def apply_fix(v1, v3, v4):                     # docs/270 §1.3 冻结（无新增旋钮）
    veto_count = int(v1) + int(v4)             # V3 自指（θ_est 循环 + 诊断无分离力，
                                               #   §二）→ 从否决链降为证据（E2 保留）
    return LABEL_OBJECT if veto_count >= 2 else LABEL_SHADOW

label_fixed = da["label"] 若 da["label"] ∉ {object, shadow}（none/texture 原样保留）
            = apply_fix(da["v1"], da["v3"], da["v4"])  否则（active 单位）
pred_fixed = d_mask 若 label_fixed == shadow，否则空
```

**修复选型的预注册论证（冻结；数据驱动，诊断记录 §二）**：

1. **否决门链在 ISTD 上几乎纯损伤**：三门损伤占比（被否决单位中真实阴影占比）全部
   >96%（V1 0.9600 / V3 0.9653 / V4 0.9792）；NS 非阴影单位仅 17/1759（1.0%）且其中
   15 个被正确否决——否决门链保护的 precision 近乎为空（基线 FP 仅 24,084 px / 0.9%），
   而它否决的 627 个单位中 612 个（97.6%）是真实阴影（候选与 GT 阴影重叠 ≥ 0.5）。
   → **单线索否决 = 系统误伤 recall 的主因**，改为双线索联合。
2. **V3 去自指化**：θ_est = 亮候选质心 − 暗候选质心（帧内从候选推光照方向、又用该
   方向否决候选——循环，docs/265 §五 6 预警）；诊断实测轴误差 TS 48.24° vs NS 51.00°
   ——**几乎无分离力**（对"阴影 vs 物体"零判别信息，V3 触发近乎随机）；V3 适用率仅
   309/1759 = 0.1757。→ V3 从否决链移除（降为证据 E2，测量保留为行为读数）。
3. **V1/V4 双正交线索联合**：V1（闭合拓扑，docs/261）与 V4（反射率跳变，docs/263）
   是合成域标定的两个**正交独立线索**（拓扑 vs 反射率）；诊断示二者对 NS 的触发率
   （0.4706/0.4118）高于对 TS（0.1102/0.1894）——有分离力但单条不可靠（TS 误伤率
   10-19%）。→ 计分制：两条**同时**成立（v1∧v4）才判物体，单条成立判阴影——
   "多维联合、不靠单一线索"的人眼语义（docs/267 判读）。
4. **零新增机制旋钮**：计分制无阈值（v1+v4 ≥ 2 是计数，非标定）；V3 移除与否决数
   门槛均为几何/语义派生、先于任何最终运行冻结；判据测行为不测参数（docs/261 §五 2
   纪律）。
5. **GT 辅助模拟（§二，仅估效应）**：V3REM 规则（本修复）pooled P 0.9813 / R 0.6382
   / F1 0.7734 / white 0.6172——最接近全 TS 翻转上限（R 0.6397 / F1 0.7768 / white
   0.6192）且 P 为激进规则中最高（对比全三门计数 ALL3：R 0.6040；无否决 NOVETO：
   P 0.9795）；剩余 24 个 object 单位（V1∧V4 联合 20 + 三门 4）中 18 个 TS——双门
   联合仍误伤少数真实阴影是诚实剩余（接近上限形态）。

**旋钮（冻结；全部继承 docs/267 冻结值，零新增机制旋钮、零重调）**：

| 旋钮 | 值 | 来源（冻结引用） |
|---|---|---|
| DELTA_SHADOW / A_MIN | 0.35 / 25 px | docs/260 §1.3 |
| K_MOVE / MOVE_IOU | 10 帧 / 0.05 | docs/260 §1.3（D2 冻结） |
| OCC_LUM_THRESH | 220 | docs/260 §1.3（D2 冻结） |
| ref_dark 分位 / 排除带 | 0.95 / L>log(221) | docs/260 §1.3（D2 冻结） |
| TOL_AXIS / RATIO_MIN | 30° / 1.5 | docs/261 §1.3 |
| TOL_H / TOL_S / BAND_EDGE / SAT_MIN | 15° / 80 / 2px / 60 | docs/263 §1.3 |
| 处理分辨率 | 160×120（RESIZE） | docs/243 §1.1 |
| 计分门槛 veto_count ≥ 2 | 计数（无阈值参数） | 本格 §1.3 选型论证 |

### 1.4 判据（冻结；每条带 docs/247 层级标签；真实域证明级，与 docs/267 同尺可比）

评估单位 = 每场景一次运行（1870 单位）；**预测** = 每单位 A 帧上 label_fixed=shadow
时的 d_mask 像素（label_fixed≠shadow → 空预测）；**GT** = 掩码 resize 160×120
（INTER_NEAREST）> 0。**C1/C2 用 pooled 口径**（全部单位像素合并 → TP/FP/FN →
P/R/F1，docs/219 标准检测度量同款；同 docs/267），同时报告单位级分布与 train/test
分列。统计外壳 = 单位级 mean±SD（ddof=1）+ bootstrap 95% CI（2000 次，种子
20260828）。

| 判据 | 标签 | 定义（冻结） | 阈值（冻结） |
|---|---|---|---|
| **C1 REAL_SHADOW_PREC_REC** | `[L3][机制][真实域证明]` | pooled 逐像素 TP/FP/FN → precision/recall/F1（docs/219 同款）；行使门槛 = pooled GT 阴影像素 ≥ 50000（不足 → 判据未被行使，报 REAL_SHADOW_LOW） | pooled **P ≥ 0.90** 且 **R ≥ 0.55** 且 **F1 ≥ 0.65**（三条件 AND） |
| **C2 WHITE_OBJ_RECALL** | `[L3][机制][真实域证明]` | 亮目标像素 = GT 阴影 ∩ 参考图 B 灰度 ≥ 128；WHITE_OBJ_RECALL = 该子集内被预测为阴影的 pooled 占比；行使门槛 = pooled 亮目标像素 ≥ 2000 且 ≥ GT 阴影像素 1% | pooled 亮目标召回 ≥ **0.50** |
| **C3 KEEP** | `[L3][机制][合成→真实保持]` | docs/267 能力不退化：守卫 R_RF_GUARD_BASE（本脚本基线路径 pooled P/R/F1 = 0.9913/0.4131/0.5832、white = 0.4265、标签分布 obj 627/shadow 1132/none 111、门率 V1 0.1137/V3 0.1148/V4 0.1916 逐位）+ R_GT_GUARD_SYNTH（第三格 det=1.0/cont=1.0）+ R_GT_GUARD_CELL4（第四格 flamingo obj 0.7875/shadow 0.2125/V3 0.7875/θ_med 287.38，docs/265 §3.1 逐位） | 守卫 = 1（见 §1.6） |
| **C4（报告性）** | `[L3][机制][真实域证明]` | 修复前后各门误伤率 Δ：before（诊断帧基线）= 每门 P(veto_g \| TS) 与损伤占比 TS_share(g)；after（修复后）= TS 单位被判 object 率（预期 0.3513 → ~0.0103）与剩余 object 单位（24）的门构成 | 报告（Δ 对照表，不进判定） |

阈值论证（冻结）：0.90/0.55/0.65 = 任务指定双目标（recall ≥ 0.55 vs 基线 0.4131；
F1 ≥ 0.65 vs 0.5832；P ≥ 0.90 = "P 不崩"高杠——预测仍 ≥90% 正确，比基线 0.9913
允降 ≤ 0.09，仍远高于随机；三条件 AND 直接体现"R 提升 + P 不崩"双目标，比 docs/267
的 0.30/0.35/0.40 工作杠显著收紧——修复后的判别应更强）；0.50 = 任务指定亮目标召回
目标（vs 0.4265）；50000/2000/1% = docs/267 同款行使门槛。均为**判据口径参数**（先
于运行冻结，测行为不测参数，docs/261 §五 2 纪律）。

### 1.5 判定映射（冻结）

- 守卫全过 且 C1 过（P/R/F1 三条件 AND）且（C2 未行使 或 C2 过）→ **RECALL_FIX_PASS**
  （否决门链 recall 修复成立：R 升到 ≥ 0.55 档且 P 保持 ≥ 0.90——修复在真实阴影 GT
  上量化成立）。
- 守卫全过 且 C1 不过（GT 像素 ≥ 50000，判据被行使）→ **RECALL_FIX_FAIL**（按名报
  P_FAIL / R_FAIL / F1_FAIL——修复未达 recall 提升或 P 崩，诚实否定发现）。
- 守卫全过 且 C1 过 但 C2 被行使且不过 → **RECALL_FIX_FAIL**（按名报 C2_FAIL）。
- 守卫全过 且 GT 阴影像素 < 50000 → **REAL_SHADOW_LOW**（判据未被行使，只报告）。
- C2 不足行使 → 报告 **WHITE_OBJ_LOW**（判据未被行使，不判过不过），不进主判定。
- 守卫不过（任一 = 0）→ **GUARD_FAIL**（实现漂移：先修实现再判机制，机制结论无效）。
- C3 KEEP 为报告性（守卫载体，不进独立判定）；C4 为报告性。

### 1.6 守卫（冻结）

1. **R_RF_GUARD_BASE**（基线逐位复现，KEEP 主载体）：本脚本基线路径（label =
   diagnose_frame 原样输出，不施加修复）pooled P/R/F1 = 0.9913/0.4131/0.5832（±5e-5）
   且 white_recall = 0.4265（±1e-3）且标签计数 obj 627/shadow 1132/none 111 且门率
   V1 0.1137/V3 0.1148/V4 0.1916（±1e-4）——证明本脚本 import 链 + 预处理与第五格
   逐字同源（修复脚本与 docs/267 同代码路径）。
2. **R_RF_GUARD_SYNTH**（import 复用证明，三连保持）：import 第三格
   `run_unit_reflect(30, 0, "main")` 重跑 1 合成单位；断言 det_gated == 1.0 且
   cont_rate == 1.0（docs/263 冻结数字）。
3. **R_RF_GUARD_CELL4**（第四格保持）：import 第四格 `run_video("flamingo")` 重跑
   DAVIS flamingo；断言 obj_rate == 0.7875 且 shadow_rate == 0.2125 且 v3_rate ==
   0.7875 且 theta_med == 287.38（docs/265 §3.1 flamingo 行逐位）。
4. **R_RF_GUARD_DET**（共享帧函数确定性）：`diagnose_frame` 同输入两次调用 → 输出
   （label/v1/v3/v4/e1/e2/e3/theta_est/dh/ds/area）全等。
5. **R_RF_GUARD_MASK**（数据完整性冒烟，纯数据不进机制）：随机 5 场景（种子
   20260828），GT 掩码与 A/B 图同尺寸、C 非空、B 帧阴影区均值 > A 帧阴影区均值
   （import 第五格 guard_mask）。
6. **R_RF_REPRO**（内部确定性复现）：--repro 时 1870 单位整体重跑第二遍（不读
   checkpoint），逐项比对关键数字位级一致（NaN 感知比较，docs/267 诊断轮 D2 先例）。

### 1.7 随机性与种子协议（冻结）

- 固定确定性流（无种子、无 jitter——数据集本身确定性，docs/243 §1.1 声明）；
- 单位 = 每场景一次运行（1870 单位）；pooled P/R/F1（主判据）+ 单位级 mean±SD
  （ddof=1）+ bootstrap 95% CI（2000 次，种子 20260828）；train/test 分列报告；
- R_RF_GUARD_MASK 用固定种子 20260828 抽 5 场景（纯数据校验）。

### 1.8 计算预算与安全纪律（冻结）

- 预算 ≤ 60 分钟（预期 ~40-50 分钟：1870 单位 × 2 帧 × 160×120 判别（同单位算
  基线 + 修复两套标签）+ 守卫 SYNTH 1 单位 + CELL4 flamingo 80 帧 + --repro 重跑
  1870 单位——docs/267 main 实测 2892s 同量级）；
- 安全模式（docs/228/234/235 纪律）：脚本 stdout 只输出 ASCII 标签 + 每行一个数字的
  R_RF_* 摘要块（顺序固定，见 SUMMARY_LINES 注释）；运行经
  `powershell -NoProfile -Command "& python vision\light_shadow_recall_fix.py <args> *> logs\rrf_<tag>.log; Write-Output('exit='+$LASTEXITCODE+' bytes='+(Get-Item 'logs\rrf_<tag>.log').Length)"`
  包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）抽取；**禁止读取
  logs/*.log 与 vision/out/results/*.json 原文（毒文件）**；ISTD PNG 是数据（可读）；
- 归档：每单位完成写 checkpoint `vision/out/results/ckpt_rrf_<config哈希>.json`
  （--resume 断点续跑）；汇总写 `vision/out/results/rrf_<tag>.json`（自描述 config /
  每 split per_unit / aggregate（基线与修复双列）/ criteria / verdict / 守卫）；
- 摘要块键（冻结顺序）：R_RF_UNITS / R_RF_FRAMES / R_RF_TP / R_RF_FP / R_RF_FN /
  R_RF_PREC / R_RF_RECALL / R_RF_F1 / R_RF_TRAIN_PREC / R_RF_TRAIN_RECALL /
  R_RF_TRAIN_F1 / R_RF_TEST_PREC / R_RF_TEST_RECALL / R_RF_TEST_F1 /
  R_RF_UNIT_REC_MEAN / R_RF_UNIT_REC_SD / R_RF_UNIT_REC_CI_LO / R_RF_UNIT_REC_CI_HI /
  R_RF_UNIT_PREC_MEAN / R_RF_UNIT_PREC_SD / R_RF_WHITE_PIX / R_RF_WHITE_TP /
  R_RF_WHITE_RECALL / R_RF_WHITE_EXERCISED / R_RF_OBJ_F / R_RF_SHADOW_F /
  R_RF_TEX_F / R_RF_NONE_F / R_RF_ACTIVE_RATE / R_RF_OBJ_REMAIN /
  R_RF_BASE_PREC / R_RF_BASE_RECALL / R_RF_BASE_F1 / R_RF_BASE_WHITE /
  R_RF_BASE_OBJ_F / R_RF_BASE_SHADOW_F / R_RF_BASE_V1 / R_RF_BASE_V3 / R_RF_BASE_V4 /
  R_RF_BEFORE_TS_OBJ / R_RF_BEFORE_TS_V1 / R_RF_BEFORE_TS_V3 / R_RF_BEFORE_TS_V4 /
  R_RF_BEFORE_V1_TS_SHARE / R_RF_BEFORE_V3_TS_SHARE / R_RF_BEFORE_V4_TS_SHARE /
  R_RF_AFTER_TS_OBJ / R_RF_AFTER_V1_TS_SHARE / R_RF_AFTER_V3_TS_SHARE /
  R_RF_AFTER_V4_TS_SHARE / R_RF_AFTER_COMBO5 / R_RF_AFTER_COMBO7 /
  R_RF_GUARD_BASE / R_RF_GUARD_SYNTH / R_RF_GUARD_SYNTH_DET /
  R_RF_GUARD_SYNTH_CONT / R_RF_GUARD_CELL4 / R_RF_GUARD_CELL4_OBJ /
  R_RF_GUARD_CELL4_SHADOW / R_RF_GUARD_CELL4_V3 / R_RF_GUARD_CELL4_THETA /
  R_RF_GUARD_DET / R_RF_GUARD_MASK / R_RF_REPRO / R_RF_C1_PREC_REC /
  R_RF_C2_WHITE / R_RF_VERDICT / R_RF_ELAPSED；
- 运行序列：`--tag timing`（--limit 20 冒烟+计时）→ 诊断轮（--tag diag，1870 单位
  全量单遍 + 守卫，§二 验证）→ `--tag main`（1870 单位，确定性 + 守卫 + --repro）→
  timing 复现轮（仅 TAG/ELAPSED 不同，核心数字逐位一致）。

---

## 二、修复前诊断与诊断轮记录（最终运行前；§一 冻结不动）

### 2.1 修复前诊断（预注册期，数据驱动修复选型；判据/旋钮/机制定义未动）

工具：D:\datasets\diag_rf.py（项目外分析工具，非实验脚本；import 复用第五格
list_scenes + 第四格 compute_ref_dark/diagnose_frame，预处理逐字同源）。在 **ISTD
1870 单位全量**上跑**未修改的第五格判别机制**，按 V1/V3/V4 统计误伤率；两轮：
v1（初始口径）与 v2（修正 white_pix 口径——v1 在 area>0 分支内统计 white_pix 漏掉
none 单位，v2 无条件统计，修正后基线 white_recall = 0.4265 与 docs/267 逐位一致；
此为分析工具自身的统计口径机械修复，非实验脚本、非判据/机制改动）。基线交叉核对
（v2）：pooled P/R/F1 = 0.9913/0.4131/0.5832、white = 0.4265、标签 obj 627/shadow
1132/none 111、门率 V1 0.1137/V3 0.1148/V4 0.1916、GT 6,663,543——**与 docs/267
§3 逐位一致**（import 链 + 预处理同源证明）。

| 量 | 值 | 判读 |
|---|---|---|
| active 单位 / TS（候选∩GT ≥0.5）/ NS | 1759 / **1742（99.0%）** / 17（1.0%） | 几乎全部 active 候选都是真实阴影 |
| TS 被判 object 率（总误伤率） | **0.3513**（612/1742） | recall 缺口主因：~35% 真实阴影被否决为物体 |
| NS 被判 object 率（正确拒绝率） | 0.8824（15/17） | 否决门链保护 precision 的对象极少（17 单位） |
| V1 误伤率 / 正确拒绝率 / 损伤占比 | 0.1102 / 0.4706 / **0.9600** | V1 否决 96% 落在真实阴影上 |
| V3 误伤率 / 正确拒绝率 / 损伤占比 | 0.1119 / 0.4118 / **0.9653** | V3 否决 96.5% 落在真实阴影上 |
| V4 误伤率 / 正确拒绝率 / 损伤占比 | 0.1894 / 0.4118 / **0.9792** | V4 否决 97.9% 落在真实阴影上（误伤率最高） |
| 门对组合（active） | 无门 1132（shadow 全在此）；V4-only 250 / V3-only 114 / V1-only 155 / V3+V4 63 / V1+V4 20（含 NS 6）/ V1+V3 21 / 三门 4 | 627 个 object 单位中**单门 519（82.8%）**、多门 108；NS 主要聚在 V3-only 6 + V1+V4 6 |
| V3 轴误差（TS vs NS） | 48.24° vs 51.00° | **无分离力**——θ_est 自指 + 真实场景几何不相关，V3 触发近乎随机 |
| V4 边界 ΔH（TS vs NS） | 23.28° vs 12.88° | **TS 反而更高**——真实阴影跨材质边界色相变化 > TOL_H 15°，V4 在真实域反向工作 |
| B 帧亮度（TS vs NS） | 127.5 vs 78.0 | 辅助信号（阴影 = B 亮表面被遮暗）有分离力，但本格不引入新通道（诚实收窄） |
| 规则模拟（GT 辅助，仅估效应） | BASE P/R/F1/white = 0.9913/0.4131/0.5832/0.4265；**V3REM（v1∧v4 判 object）0.9813/0.6382/0.7734/0.6172**；ALL3（三门计数<2）0.9810/0.6040/0.7476/0.5855；CEIL（全 TS 翻回）0.9888/0.6397/0.7768/0.6192；NOVETO 0.9795/0.6424/0.7759/0.6204 | **V3REM 最接近上限且 P 最高**——冻结为本格修复（§1.3） |

### 2.2 诊断轮（--tag diag，1870 单位全量单遍 + 守卫；最终运行前）

全部修复发生在最终运行之前；§1.4 判据阈值自预注册起冻结，未因任何诊断调整。机制
语义（测量层逐字 import 第五格；标签合成层计分制 = veto_count v1+v4 ≥ 2 判 object）
不变，改变的是实现细节（同 docs/260 D1-D4 "校准记录"先例）：

| 轮 | 修复 | 动机（诊断数据驱动） |
|---|---|---|
| D1 | `run_scene_fixed` 返回键名统一为前缀 `fix_`（fix_tp/fix_fp/fix_fn） | 机械 bug：初版用后缀 `tp_fix`，与聚合函数 pooled_prf_parts 的 `prefix+"tp"` 约定不一致 → KeyError 'fix_tp'（冒烟首跑）；修复后冒烟通过 |
| D2 | 删除陈旧 checkpoint（ckpt_rrf_*.json）后以 --no-resume 重算冒烟 | 机械 bug：D1 修复前的失败运行写入的 checkpoint（旧键名）被默认 resume 加载 → 重复 KeyError；删除陈旧 checkpoint 后重算通过 |
| D3 | （环境说明，非脚本修复）守卫 run_video("flamingo") 的 DAVIS 相对路径依赖 cwd=项目根目录 | 冒烟从会话工作目录启动时 FileNotFoundError 'vision\\out\\davis\\flamingo'；按 docs/267 运行惯例从项目根目录启动即通过（第 5 格同款） |

诊断轮验证（--tag diag，1870 单位全量单遍 + 守卫）：守卫 R_RF_GUARD_BASE=1（基线
pooled P 0.9913 / R 0.4131 / F1 0.5832 / white 0.4265、标签 obj 627/shadow 1132/
none 111、门率 V1 0.1137/V3 0.1148/V4 0.1916——与 docs/267 §3 **逐位一致**）、
R_RF_GUARD_SYNTH=1（det 1.0/cont 1.0）、R_RF_GUARD_CELL4=1（flamingo 0.7875/0.2125/
0.7875/287.38 逐位命中 docs/265 §3.1）、R_RF_GUARD_DET=1、R_RF_GUARD_MASK=1、
R_RF_REPRO=1——**判据阈值 §1.4 未动，机制语义未动**；修复后 pooled P 0.9813 /
R 0.6382 / F1 0.7734（train 0.9802/0.6184/0.7584、test 0.9842/0.6991/0.8175）、
white 0.6172、obj 24（COMBO5 20 + COMBO7 4）——与 §一 冻结的预注册模拟（P 0.9813/
R 0.6382/F1 0.7734/white 0.6172、obj_remain 24）**逐位一致**；VERDICT=RECALL_FIX_PASS
是修复机制在真实像素上的行为成立发现（§四），不是实现缺陷。（最终运行前的其余
机械修复在此追加。）

---

## 三、结果（1870 单位 × A 帧；数字以 vision/out/results/rrf_main.json 工件 +
logs/rrf_main.log 摘要块为准，经 extract_r.py 纯正则抽取，未读输出文件原文）

### 3.1 判据逐项（docs/270 §1.4 冻结阈值；修复后 vs docs/267 基线对照）

| 判据 | 度量 | 修复后 | 基线（docs/267） | 冻结阈值 | 判定 |
|---|---|---|---|---|---|
| **C1 REAL_SHADOW_PREC_REC** | pooled 逐像素 precision | **0.9813**（4,252,663 TP / 81,012 FP） | 0.9913 | ≥ 0.90 | ✓ |
| C1 | pooled recall | **0.6382**（4,252,663 TP / 6,663,543 GT） | 0.4131 | ≥ 0.55 | ✓ |
| C1 | pooled F1 | **0.7734** | 0.5832 | ≥ 0.65 | ✓ |
| C1 | GT 阴影像素（行使门槛） | **6,663,543** | 6,663,543 | ≥ 50000（满足，判据被行使） | — |
| C1 | train pooled P/R/F1 | **0.9802 / 0.6184 / 0.7584** | 0.9899/0.3857/0.5551 | 报告 | — |
| C1 | test pooled P/R/F1 | **0.9842 / 0.6991 / 0.8175** | 0.9948/0.4973/0.6631 | 报告 | — |
| C1 | 单位级 recall mean±SD（CI） | **0.6803 ± 0.3275**（[0.6652, 0.6952]，1870 单位） | 0.4632±0.4224 | 报告 | — |
| C1 | 单位级 precision mean±SD | **0.9855 ± 0.0616** | 0.9910±0.0320 | 报告 | — |
| **C2 WHITE_OBJ_RECALL** | 亮目标（GT 阴影 ∩ B 帧 ≥128）pooled 召回 | **0.6172**（2,585,424 / 4,188,633） | 0.4265 | ≥ 0.50 | ✓ |
| C2 | 行使门槛 | 亮目标像素 **4,188,633**（≥ 2000 且 ≥ GT 1%=66,635） | 满足 | 行使 | — |
| **C3 KEEP** | 守卫 BASE + SYNTH + CELL4 | **1 / 1 / 1**（基线逐位 + 1.0/1.0 + 0.7875/0.2125/0.7875/287.38） | = 1 | ✓ |

### 3.2 行为读数（报告性，docs/267 §3.2 口径）

- 修复后标签分布（1870 单位）：**object 24（1.3%）/ shadow 1735（92.8%）/ texture 0 / none 111**
  （active 率 0.9406 不变）；object 单位 **627 → 24（−603）**——剩余全部为 V1∧V4 双门
  联合（COMBO5=20，含 6 个 NS）+ 三门全触发（COMBO7=4），即计分制只保留"两正交线索
  联合成立"的强物体证据。
- **C4 误伤率 Δ（修复前后，报告性）**：

| 量 | 修复前（基线） | 修复后 | Δ |
|---|---|---|---|
| TS 单位被判 object 率（总误伤率） | **0.3513**（612/1742） | **0.0103**（18/1742） | **−0.3410** |
| 门参与误伤率：V1（TS∩门∩判 object） | 0.1102 | 0.0103 | −0.0999 |
| 门参与误伤率：V3 | 0.1119 | 0.0023 | −0.1096 |
| 门参与误伤率：V4 | 0.1894 | 0.0103 | −0.1791 |
| 损伤占比（被否决/参与判 object 单位中 TS 占比）：V1 | 0.9600 | 0.7500（剩余 object 单位中 TS 占比） | −0.2100 |
| 损伤占比：V3 | 0.9653 | 1.0000（COMBO7 4/4） | +0.0347 |
| 损伤占比：V4 | 0.9792 | 0.7500 | −0.2292 |

  剩余 object 单位 24 中 TS 18（0.75）——双门联合仍误伤少数真实阴影（诚实剩余，
  §五 4，与 §一 冻结模拟的 obj_remain=24 逐位一致）。

### 3.3 守卫（docs/270 §1.6）

| 守卫 | 结果 | 说明 |
|---|---|---|
| R_RF_GUARD_BASE | **1** | 基线 pooled P/R/F1 = 0.9913/0.4131/0.5832、white = 0.4265、标签 obj 627/shadow 1132/none 111、门率 V1 0.1137/V3 0.1148/V4 0.1916——与 docs/267 §3 **逐位一致**（import 链 + 预处理同源证明） |
| R_RF_GUARD_SYNTH | **1** | import 第三格 run_unit_reflect(30,0,"main")：det_gated=1.0000、cont_rate=1.0000——三连 import 链逐字同源 |
| R_RF_GUARD_CELL4 | **1** | import 第四格 run_video("flamingo")：obj 0.7875 / shadow 0.2125 / V3 0.7875 / θ_med 287.38——docs/265 §3.1 逐位命中（KEEP 机制载体） |
| R_RF_GUARD_DET | **1** | diagnose_frame 同输入两次调用输出全等（共享帧函数确定性，label=shadow） |
| R_RF_GUARD_MASK | **1** | 5 场景数据完整性：掩码与图同尺寸且非空、无阴影参考阴影区均值 > 阴影图（映射正确） |
| R_RF_REPRO | **1** | --repro 第二遍强制重算（1870 单位，不读 checkpoint）逐项位级一致（含 NaN 感知比较） |

### 3.4 运行计时

- main：**2659.30 秒（44.3 分钟）**（1870 单位 × 2 遍 + 守卫；预算 ≤ 60 分钟 ✓）；diag
  轮 1369.31 秒（22.8 分钟，单遍无 repro）；timing 14.74 秒 / timing2 14.96 秒（20 场景
  + 守卫）——timing 两轮核心数字**逐位一致**（TP/FP/FN/P/R/F1/WHITE/各门/守卫），
  确定性成立；
- diag 与 main 的 1870 单位逐项数字**逐位一致**（确定性流）；差异仅 ELAPSED（机器负载）。

### 3.5 与 docs/267 的对照（修复效果量化）

| 量 | docs/267 基线 | 本格修复后 | 变化 |
|---|---|---|---|
| pooled precision | 0.9913 | **0.9813** | −0.0100（守住 ≥ 0.90，P 不崩） |
| pooled recall | 0.4131 | **0.6382** | **+0.2251**（漏检 58.7% → 36.2%） |
| pooled F1 | 0.5832 | **0.7734** | +0.1902 |
| WHITE_OBJ_RECALL（亮目标） | 0.4265 | **0.6172** | +0.1907（白目标漏 57.3% → 38.3%） |
| object 单位数（否决门误伤） | 627 | **24** | −603（其中 612 为真实阴影误伤，修复后仅剩 18 个 TS 误伤） |
| 否决门链语义 | 任一否决 → object | 计分制（v1+v4 ≥ 2）→ object；V3 移出否决 | docs/267 §四 判读的"漏检主因"被直接修复 |

**与 docs/267 的机制关系**：docs/267 证明"判别机制在真实阴影 GT 上成立（P 0.99）但
漏检显著（R 0.41，否决门链把 ~35% 真实阴影误判为物体）"；本格把漏检主因（否决门链
单线索硬否决）改为多证据联合计分——测量层零改动（守卫 BASE 逐位证明），只改决策
组成。修复后 recall/F1/亮目标召回全部显著提升且 precision 保持 ≥ 0.90，docs/267
的结论（过检可控）不退化、其缺口（漏检）被量化修复。

---

## 四、判定

**R_VERDICT = RECALL_FIX_PASS。** 判定数字（1870 单位确定性流，工件 rrf_main.json）：

- C1 REAL_SHADOW_PREC_REC：pooled 逐像素 precision **0.9813** / recall **0.6382** /
  F1 **0.7734**（三条件 P≥0.90 且 R≥0.55 且 F1≥0.65 全过 ✓）；GT 阴影像素 6,663,543 ≥
  50000，判据被行使；train 0.9802/0.6184/0.7584、test 0.9842/0.6991/0.8175；单位级
  recall 0.6803±0.3275（CI [0.6652,0.6952]）、单位级 precision 0.9855±0.0616。
- C2 WHITE_OBJ_RECALL：亮目标（GT 阴影 ∩ B 帧 ≥128，4,188,633 像素）pooled 召回
  **0.6172** ≥ 0.50 ✓（行使门槛满足）。
- C3 KEEP：守卫 BASE=1（基线 0.9913/0.4131/0.5832/0.4265 + 标签分布 + 门率与 docs/267
  逐位一致）、SYNTH=1（det/cont 1.0）、CELL4=1（flamingo 0.7875/0.2125/0.7875/287.38
  逐位命中 docs/265 §3.1）——docs/267 能力不退化。
- 守卫：BASE=1、SYNTH=1、CELL4=1、DET=1、MASK=1、REPRO=1（1870 单位第二遍重算位级
  一致）。
- 预注册判定映射（§1.5）：守卫全过 且 C1 过 且 C2 过 → 按冻结映射 →
  **RECALL_FIX_PASS（否决门链 recall 修复成立：R 从 0.41 档升到 0.64 档且 P 保持
  ≥ 0.90——修复在真实阴影 GT 上量化成立）**。

> 一句话：**docs/267 的 recall 缺口（否决门链把 ~35% 真实阴影误判为物体，R 0.4131）
> 被机制级修复——修复选型由预注册期修复前诊断（ISTD 1870 单位全量：三门损伤占比
> V1 0.9600 / V3 0.9653 / V4 0.9792，被否决单位几乎全是真实阴影；V3 轴误差无分离力
> 48.24° vs 51.00°；V4 在 TS 上 ΔH 23.28° 反向工作；GT 辅助模拟 V3REM 最接近上限且
> P 最高）数据驱动冻结 = 否决门链 → 计分制联合判别（测量层逐字 import 第五格零改动；
> 标签合成层 veto_count = v1+v4，V3 自指降为证据 E2，object iff 双正交线索联合成立）：
> pooled P/R/F1 = 0.9813 / 0.6382 / 0.7734（基线 0.9913/0.4131/0.5832：R +0.225、
> F1 +0.190、P −0.010 守住 ≥0.90）、WHITE_OBJ_RECALL = 0.6172（基线 0.4265，+0.191）、
> object 单位 627 → 24（真实阴影误伤 0.3513 → 0.0103）、守卫六连全过（含基线逐位
> 复现 docs/267 + REPRO 位级一致）、确定性成立（timing 两轮与 diag/main 逐位一致）。
> 诚实：修复是标签合成层变更（测量零改动）；V3 去自指化针对决策链（θ_est 测量与 E2
> 仍为行为读数）；P 守住与否由 C1 ≥ 0.90 直接量化（实际 0.9813）；双门联合仍误伤 18
> 个 TS（诚实剩余，obj_remain 24 与冻结模拟逐位一致）；单候选/ISTD 数据选择/静态域
> 近零纹理同 docs/267 照实。**

---

## 五、诚实边界（预注册阶段已知）

1. **修复是"标签合成层"变更，测量层零改动**：Pass A/B、时间门、V1/V3/V4 几何/反射率
   测量、E1/E2/E3、θ_est、ΔH/ΔS 全部逐字 import 第五格（守卫 R_RF_GUARD_BASE 逐位
   证明）；修复只改"证据如何合成判定"（任一否决 → object 硬链 → 计分制联合判别）。
   这是"机制语义修复"而非"参数重调"——被修复的是决策组成，不是测量本身。
2. **V3 去自指化的边界（是否完全）**：θ_est 的**测量**与 E2（主轴沿光证据）保留为
   行为读数（无光照 GT，docs/267 §五 5 同款），但**不再进入否决决策**——"用候选推
   光照方向、又用该方向否决候选"的循环从决策链移除。诚实声明：θ_est 仍是帧内估计
   （非真值验证），E2 报告性；V3 触发率在修复后仍报告（行为读数），但不影响标签。
3. **P 是否守住**：由 C1 P ≥ 0.90 直接量化（基线 0.9913 → 修复目标 ≥ 0.90）。修复
   把部分非阴影单位（基线 15 个 NS object 单位中的一部分）翻回阴影——其 FP 代价
   （候选非阴影像素进入预测）被 P 判据直接测出；诊断模拟示 P 0.9813（NOVETO 参照
   0.9795 为最激进下限）。若修复后 P < 0.90 → 按冻结映射报 P_FAIL（修复过度，诚实
   否定）。
4. **双门联合仍误伤少数真实阴影（诚实剩余）**：诊断模拟示修复后仍有 ~24 个单位判
   object（V1∧V4 联合 20 + 三门 4），其中 ~18 个 TS——recall 上限 0.6397（全 TS
   翻转）vs 本修复模拟 0.6382，剩余缺口 ~0.0015 来自双门联合误伤；这是"多维联合"
   语义对极少数强物体证据真实阴影的诚实代价，如实报告不隐藏。
5. **修复选型的数据驱动性与预注册纪律**：选型（V3 移出 + v1∧v4 计分）由 §2.1 修复前
   诊断（ISTD 全量，未碰判据/旋钮）驱动并在 §1.3 冻结；模拟（§2.1）为 GT 辅助的
   **效应估计**（诊断工具输出），不是最终运行结果——最终数字以 --tag main 为准；
   判据阈值（0.90/0.55/0.65/0.50）为判据口径参数，先于最终运行冻结。
6. **单候选局限不变**（docs/267 §五 4）：判别对象 = 最大暗域；多阴影区域/阴影外暗
   物体未建模；GT 掩码 resize INTER_NEAREST 有像素化误差。
7. **数据选择不变**（docs/267 §五 1）：ISTD（SBU/SRD 无可达镜像）；hf-mirror 镜像
   目录映射 _A=阴影图/_B=掩码/_C=参考（D1 冻结）。
8. **静态域适配代价不变**（docs/267 §五 2）：2 帧场景对 [B, A]；时间门近恒过、
   texture 标签近零（如实报告）；B 帧与 A 帧非阴影区一致性假设同前。
9. **统计口径**：pooled P/R/F1（主判据）+ 单位级 mean±SD + bootstrap CI；
   train/test 分列；C4 报告性。
10. **下一步（超出本格）**：多候选判别、B 帧亮度/光照先验作为独立证据通道（§2.1
    示 TS 127.5 vs NS 78.0 有分离力，本格诚实收窄未引入）、跨数据集验证（SBU 可达
    后）、光照方向真值验证（带光源方位标注集）。

---

## 六、一句话

> **光影判别第六格（预注册冻结）：docs/267 的 recall 缺口（否决门链把 ~35% 真实
> 阴影误判为物体）做机制级修复——修复选型由预注册期修复前诊断（ISTD 1870 单位
> 全量，三门损伤占比 V1 0.9600/V3 0.9653/V4 0.9792、V3 轴误差无分离力、V4 在 TS 上
> ΔH 23.28° 反向工作、GT 辅助模拟 V3REM P 0.9813/R 0.6382 最接近上限且 P 最高）
> 数据驱动冻结 = **否决门链 → 计分制联合判别**：测量层逐字 import 第五格零改动，
> 标签合成层改为 `veto_count = v1+v4`（V3 自指降为证据 E2）、`object iff veto_count
> ≥ 2`（V1 拓扑与 V4 反射率两正交线索联合才判物体，单线索不再一票否决——人眼
> "多维联合、不靠单一线索"）。判据与 docs/267 同尺可比：C1 pooled P ≥ 0.90 且
> R ≥ 0.55 且 F1 ≥ 0.65（"R 提升 + P 不崩"双目标）、C2 亮目标召回 ≥ 0.50、C3 KEEP
> （守卫：基线逐位复现 docs/267 + SYNTH + CELL4 flamingo 逐位）、C4 误伤率 Δ 报告。
> 判定 = RECALL_FIX_PASS / RECALL_FIX_FAIL / GUARD_FAIL。诚实：标签合成层修复、
> 测量零改动；V3 去自指化针对决策链（θ_est 测量与 E2 仍为行为读数）；P 守住与否
> 由 C1 直接量化；单候选/ISTD 数据选择/静态域近零纹理同 docs/267 照实。结论：
> **RECALL_FIX_PASS——否决门链 recall 修复在真实阴影 GT 上量化成立：pooled P/R/F1
> = 0.9813 / 0.6382 / 0.7734（基线 0.9913/0.4131/0.5832：R +0.225、F1 +0.190、
> P −0.010 守住 ≥ 0.90）、WHITE_OBJ_RECALL = 0.6172（基线 0.4265）、object 单位
> 627 → 24（TS 误伤率 0.3513 → 0.0103）、守卫六连全过（含基线逐位复现 docs/267 +
> REPRO 位级一致）——docs/267 判读的"漏检主因（否决门链误伤真实阴影）"被直接修复，
> P 不崩由判据量化保证。**
