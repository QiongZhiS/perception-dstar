# 251 — QUOTA 退役：延迟定级（deferred finalization）消除评估层配额动作面（docs/250 QUOTA_ORTHOGONAL 的续：让评估层配额正式退役）

> 缘起：docs/250 verdict=MECH_PASS（快慢双原型机制级修复），但 [L3][诊断] QUOTA_ORTHOGONAL
> = **配额仍需（不可退役）**——机制 + docs/249 短段配额叠加仍翻转 S2/S3/R1 的"升级未完成
> 慢原型"：d_sc2 = +1/+2/+1、d_churn = −0.2000/−0.2222/−0.1250（docs/250 §六，翻转原型 =
> S2 的 48:2、S3 的 29:2/35:2、R1 的 48:2，均为"升级后未完成固化（hits=2 < hits_min_slow=3）
> 且所在段预算 < hits_min_slow"的慢原型）。docs/250 诚实声明：作用面显著缩小（单层系统下
> 配额翻 3-4 个原型、机制下只作用于慢层 1-2 个升级未完成原型）但**未归零**。本实验：用**机制**
> 消除这最后一个动作面——**延迟定级（deferred finalization）**——让配额开/关结果完全一致，
> 正式退役配额（代码中标记废弃、docs/249 配额路径标注退役）。
>
> 状态：**预注册已冻结（§一：机制/旋钮/判据/守卫先于最终运行写入本文件，docs/63+247
> 纪律；本节运行后不改机制、不改旋钮、不改判据阈值）→ 实现（vision/quota_retire.py，
> import 复用，禁止修改任何既有脚本）→ 确定性流运行（守卫 + 主运行）→ 结果与判定见
> §三/§四**。
> 数字以 vision/out/results/qr_<tag>.json 工件 + logs/qr_<tag>.log 摘要块为准
> （数字用纯 python 正则抽取，未读任何输出文件原文）。
> 引用：docs/63/245/246/247/248/249/250 + vision/fastslow_test.py（FastSlowLoop/
> quota_on_slow/gist_metrics/build_entry_base/R_FAST/R_SLOW/HITS_MIN_FAST/HITS_MIN_SLOW/
> K_PROMOTE/K_DECAY/K_CONSIST_FAST 复用）+ vision/fastcut_fix.py（fastcut_gate/apply_quota/
> run_guard_quota 复用）+ vision/soft_match_test.py（ALPHA/HITS_MIN 复用）+ vision/
> cross_domain_test.py（load_sampled_frames/WILD_VIDEOS/STREAMS/RADIUS_L3/R_BASE_DAVIS/
> D246/DL_DIR/guard_vs_d246/scene_switch_diag 复用）+ vision/real_stream_test.py
> （load_video_frames/VIDEOS/WINDOW/RESIZE 复用）+ vision/real_recalib.py（bridge_metrics
> 复用）+ vision/stream_test.py（LOOP_CFG 复用）+ vision/critical_point.py（mean_sd/
> bootstrap_ci 复用）+ vision/quota_retire.py（本轮新增）。

---

## 〇、一句话（预注册；结论运行后填写）

> **预注册：docs/250 的 QUOTA_ORTHOGONAL=配额仍需，最后一个动作面 = 机制下"升级未完成"
> 慢原型（hits=2 < hits_min_slow=3、且段预算 < hits_min_slow）被 docs/249 配额豁免翻转。
> 本实验把升级判定从"hits ≥ k_promote=2 立即升级"改为**延迟定级**：快→慢升级**仅当**该
> 原型累积满 hits_min_slow=3 次重匹配（hits ≥ hits_min_slow）时才**最终化**为慢原型；
> hits ∈ [k_promote, hits_min_slow)（即 hits=2）的"升级候选"**保持快原型**（继续 r_fast
> 匹配、仍受 k_decay=5 回收——docs/250 已确立"短命但正确"为正确行为），**不产生任何
> 半成品慢状态**——慢原型因此**只以"确认已满"的形态存在**（final hits ≥ hits_min_slow
> 恒成立，hits 单调不减），配额（对未完成慢原型的豁免）**再无动作面**：配额开/关在全部
> 流上 churn/SC2/ratio/bridge 逐项完全一致（形式判据 QUOTA_RETIRED）。k_promote=2 被
> 延迟定级**吸收**（升级阈值升为 hits_min_slow=3；k_promote 保留在配置中标注
> DEPRECATED/SUPERSEDED，行为零作用）。旋钮全部沿用 docs/250 §1.3 冻结值（零回调）；
> **不做段长预测**（延续 docs/250 纯行为涌现原则）。判据（带 docs/247 标签）：
> `[L3][机制][退役]` QUOTA_RETIRED（配额开 vs 关全流逐项一致，配额动作面=0，形式判据）、
> `[L3][机制][无配额]` CHURN_MECH（慢原型 churn ≤ 0.5 全流）、`[L3][机制][gist正确性]`
> GIST_CORRECT（R1 真值段边界对应率 ≥ 0.5，目标保持 1.0）、`[L3][机制]` STABLE/STRUCT
> 保持（全流 ratio ≤ 1.5 且 SC2_slow > 0）、`[L3][机制][行为证据]` PROMOTION（升级/回收
> 仍发生且升级命中率均值 > 未升级均值）。守卫：docs/246 M=1.5 复现（12/12，配额关闭
> 口径）。判定：1-5 全过 且 守卫 12/12 = **QUOTA_RETIRED**（配额正式退役：docs/249 配额
> 路径标注退役/保留为历史、代码标注废弃）；判据 1 不过 = **QUOTA_STILL_NEEDED**（延迟
> 定级未消除动作面，如实报告动作面在哪）；部分 = **PARTIAL**（如实）。**
>
> **结论：QUOTA_RETIRED。** [L3][机制][退役] QUOTA_RETIRED 过——配额开 vs 关在全部流
> （S1-S4 + R1）上 churn/SC2/ratio 逐项相等、R1 bridge 相等（全流 d_sc2=0、
> d_churn=0.0000，配额动作面 = 0）；docs/250 §六的 S2/S3/R1 翻转（Δchurn
> −0.2000/−0.2222/−0.1250、ΔSC2 +1/+2/+1）全部归零。[L3][机制][无配额] CHURN_MECH 过
> （慢原型 churn 全流 **0.0000**，按构造成立：延迟定级下慢原型只以"确认已满"形态存在）；
> [L3][机制][gist正确性] GIST_CORRECT 过（R1 真值段边界对应率 **0.8750，7/8，|Δ|≤1**，
> |Δ|≤2 覆盖 1.0000——docs/250 的 1.0000 未保持，switch 49 的 |Δ|=2，如实报告）；
> [L3][机制] STABLE/STRUCT 保持过（ratio max 1.371908 与 docs/250 逐位一致、SC2_slow
> min 4 > 0）；[L3][机制][行为证据] PROMOTION 过（28 升级 / 84 回收，升级命中率均值
> 3.8571-14.2857 vs 未升级 1.0000-1.2857，升级非随机）；守卫 R_QR_GUARD_D246=1
> （12/12）；内部复现 R_QR_REPRO_RATIO=1（5/5）。**配额正式退役**：docs/249 配额路径
> 标注退役/保留为历史、新代码（quota_retire.py）标注废弃（k_promote DEPRECATED）。
> 诚实边界：churn_slow=0 为构造成果（学习有效性由 GIST/PROMOTION/SC2/守卫承担）、
> gist_cov 1.0→0.8750、配额开=关的 ratio/bridge 相等为构造性（形式判据的真正内容 =
> 配额唯一能改判的慢层 churn/SC2 全流零 Δ）。

---

## 一、预注册设计（冻结；判据/旋钮/守卫先于任何运行写入本文件，docs/63+247 纪律）

### 1.1 缘起与目标（冻结）

docs/250 verdict=MECH_PASS 后，[L3][诊断] QUOTA_ORTHOGONAL（预注册 §1.7 判据 5）实测：
机制 + docs/249 配额叠加 vs 机制单独，S2/S3/R1 的慢层数字仍被翻转：

| 流 | churn_slow | churn_q | Δchurn | SC2_slow | SC2_q | ΔSC2 |
|---|---|---|---|---|---|---|
| S1 | 0.2857 | 0.2857 | 0.0000 | 5 | 5 | 0 |
| S2 | 0.2000 | 0.0000 | **−0.2000** | 4 | 5 | **+1** |
| S3 | 0.3333 | 0.1111 | **−0.2222** | 6 | 8 | **+2** |
| S4 | 0.2727 | 0.2727 | 0.0000 | 8 | 8 | 0 |
| R1 | 0.2500 | 0.1250 | **−0.1250** | 6 | 7 | **+1** |

翻转机理（冻结引用，docs/250 §六）：S2 48:2、S3 29:2/35:2、R1 48:2——这些原型已升级为
慢（hits 达 k_promote=2）但**未完成固化**（final hits=2 < hits_min_slow=3），且所在段
剩余窗口预算 b < hits_min_slow=3；docs/249 配额（apply_quota：hits_min_eff =
max(1, min(hits_min, b))）把这些"物理上不可达的确认阈值"豁免 → hits=2 ≥ hits_min_eff →
改判稳定 → churn 降 / SC2 升。

**配额的动作面（本实验要消除的对象，冻结）**：机制下存在**"升级未完成"慢原型**（已被
升级到慢类、受回收豁免，但 final hits 未达 hits_min_slow）——配额正是对这类原型做豁免
改判。本实验目标（预注册）：用机制消除"升级未完成慢原型"这一形态本身——慢原型**只以
"确认已满"的形态存在**，配额无可豁免对象 → 配额开/关结果完全一致 → 配额正式退役。

### 1.2 机制：延迟定级（deferred finalization）（冻结）

**升级判定（冻结）**：快→慢升级**不在** hits ≥ k_promote=2 时立即发生；**仅当**该原型
累积满 **hits_min_slow=3 次重匹配**（hits ≥ hits_min_slow）时才**最终化**为慢原型
（kind→slow、promoted_at=当前窗、半径收紧为 r_slow、受回收豁免——与 docs/250 升级语义
同款，仅触发阈值后移）。

**半成品状态管理（冻结）**：hits ∈ [k_promote, hits_min_slow)（即 hits=2）的"升级候选"
**不产生任何状态变化**——保持 kind=fast、继续按 r_fast 粗半径匹配、**不豁免回收**
（连续 k_decay=5 窗未重匹配 → 回收）。即：机制下**不存在"半成品慢原型"这一形态**；
docs/250 中"升级未完成"的慢原型在本机制下是不存在的（它们要么继续以快原型积累到 3
最终化，要么短命回收——docs/250 已确立"短命但正确"为正确行为，允许回收）。

**推论（冻结，构造成立）**：hits 单调不减、慢原型仅在 hits ≥ hits_min_slow 时产生 →
任何慢原型的 final hits ≥ hits_min_slow=3 恒成立 → **慢层 churn_slow = 0.0 按构造
成立**（设计语义：CHURN_MECH 的意义从"压 churn"变为"结构上不存在未完成慢原型"；这不是
平凡化，而是延迟定级把"未完成固化"从慢类中消除的设计结果）。配额（apply_quota：
hits_min_eff ≤ hits_min_slow 恒成立）对任何慢原型无可豁免 → **配额动作面 = 0 按构造
成立**，形式判据 QUOTA_RETIRED 的"逐项相等"由机制保证、由运行验证。

**k_promote 状态（冻结）**：被延迟定级**吸收**（升级阈值 = hits_min_slow=3）；k_promote=2
保留在配置中标注 DEPRECATED/SUPERSEDED（行为零作用），仅文档追溯（docs/250 §1.3 的
k_promote=2 语义"升级即固化候选"被本实验替换为"hits_min_slow 即定级门槛"）。

**回收规则（冻结）**：不变（docs/250 §1.4 第 4 步）——快原型（含升级候选）连续
k_decay=5 窗未重匹配 → 从原型集移除、n_recycled +1；创建计数保留（创建时点是 gist
度量的时间戳，回收只移除活跃原型）；慢原型不回收（固化记忆）。

**不做段长预测（冻结，延续 docs/250 §1.2）**：不使用帧差、位移、任何外部信号估计段长/
切点；时间尺度（什么内容被固化、什么内容保持短命）完全由行为（命中率 → 升级/回收）
涌现。帧差只用于野流**诊断**（近似段边界，诚实声明为无 GT 近似），不进入任何机制决策。

### 1.3 旋钮初值（冻结；全部沿用 docs/250 §1.3，零回调）

| 旋钮 | 值 | 状态（冻结） |
|---|---|---|
| r_slow | **0.39885**（= RADIUS_L3，docs/246 M=1.5 工作点） | 不变（细半径/注视） |
| r_fast | **0.598275**（= 1.5 × r_slow） | 不变（粗半径/gist） |
| hits_min_fast | **1** | 不变（低门槛） |
| hits_min_slow | **3** | 不变（高门槛 = 任务冻结值） |
| **定级门槛 K_FINALIZE** | **= hits_min_slow = 3**（复用冻结值，**不引入新取值**） | **本实验机制核心**：快→慢最终化阈值 |
| k_promote | 2 | **DEPRECATED/SUPERSEDED**（被延迟定级吸收：升级阈值 = hits_min_slow=3；行为零作用，仅文档追溯） |
| k_decay | **5** | 不变（回收门槛，只作用于快原型） |
| k_consist_fast | **1** | 不变（gist 先行立即创建） |
| α | **0.2** | 不变（原型中心滑动学习率） |
| window / LOOP_CFG | 10 / 原样 | 不变（docs/248-250 逐字） |

### 1.4 匹配/升级/回收算法（冻结；DeferredLoop(FastSlowLoop)，预测路径零改动）

特征空间与参与门与 SoftLoop 逐字一致（x_w = (ln(1+E_w), ln(1+U_w))，参与门 E_w ≥ 10）。
**DeferredLoop = FastSlowLoop 的唯一机制改动 = 快匹配分支的升级判定行**：
docs/250 `if p["hits"] >= self.k_promote:` → 本实验 **`if p["hits"] >= self.hits_min_slow:`**
（延迟定级）。其余全部逻辑与 docs/250 §1.4 逐字一致：

1. **慢优先**：对全部 kind=slow 原型取最近者，欧氏距离 ≤ r_slow → 匹配慢原型（hits += 1、
   last_active=w、n_match += 1、μ 滑动 α=0.2）。
2. **快兜底**：否则对全部 kind=fast 原型取最近者，距离 ≤ r_fast → 匹配快原型（同更新）；
   **匹配后若 hits ≥ hits_min_slow → 最终化为慢**（kind→slow、promoted_at=w、半径收紧为
   r_slow、升级计数 +1、受回收豁免）——这是与 docs/250 的**唯一**行为差异。
3. **高残差新奇段触发创建（k_consist_fast=1）**：否则 → **立即**在窗口特征处创建快原型
   （hits=1、created=w、kind=fast、n_created_fast +1）。
4. **回收**（每窗口末）：对全部 kind=fast 原型（**含升级候选**），若 w − last_active ≥
   k_decay → **回收**（从原型集移除、n_recycled +1；创建计数保留）。慢原型不回收。

确定性：无 RNG、无 jitter、无逐场景统计；全部操作（log1p/欧氏距离/比较）确定性。
**预测路径零改动**：step/事件/阈值路径原样继承 CompLoop → MAE 序列与 docs/245-250 逐位
一致 → ratio 是构造性控制项（R_QR_REPRO_RATIO 内部复现）。

### 1.5 度量定义（冻结；沿用 docs/250 §1.5）

- SC1_fast / SC2_fast / SC1_slow（= 累计升级数）/ SC2_slow（= 慢原型 final hits ≥
  hits_min_slow 数）/ churn_slow = (SC1_slow − SC2_slow)/max(1, SC1_slow)（预注册推论：
  按构造成立 = 0.0）/ churn_legacy（诊断，docs/248 口径）。
- n_promo / n_recycle；promoted 均值命中率 vs 未升级均值命中率（PROMOTION 判据 5）。
- **配额开/关逐项对照（QUOTA_RETIRED 判据 1 口径）**：per-stream（S1-S4 + R1）——
  churn_off = churn_slow vs churn_on = 叠加 docs/249 配额于慢层（apply_quota + fastcut_gate
  复用，docs/250 §六同款口径）后的 churn_q；sc2_off = sc2_slow vs sc2_on = sc2_q；
  ratio_off = ratio vs ratio_on（配额为 finalize 级改判，不改 MAE → 逐位相等按构造成立，
  显式比对）；R1 另加 bridge_off vs bridge_on（bridge_metrics 只读 entry_log，配额不改
  entry_log → 相等按构造成立，显式比对）。
- MAE/ratio/SC1/SC2/pin/θ：与 run_soft 同款（ratio = 末/首四分之一窗口 MAE 均值比）。

### 1.6 GIST 真值（DAVIS R1 视频切换 = GT，冻结；沿用 docs/250 §1.6 逐字）

- **R1 GT 段边界**：R1 = 9 视频按序拼接（flamingo 80 / surf 55 / bear 82 / camel 90 /
  dog 60 / blackswan 50 / car-turn 80 / motorbike 43 / soccerball 48 帧），切换窗 =
  [8, 13, 21, 30, 36, 41, 49, 54]。
- **对应率**：gist_cov（主判据）= #{切换 s : ∃快原型创建窗 c 且 |c − w_s| ≤ 1} / 8；
  gist_prec（诊断）；|Δ|≤2 覆盖（诊断）。
- **野流（S1-S4）诊断**：帧差近似段边界（scene_switch_diag 复用），诊断级不进主判定。

### 1.7 判据（预注册，冻结；每判据带 docs/247 层级标签；不得在看过结果后修改）

1. **`[L3][机制][退役]` QUOTA_RETIRED**（**形式判据**）：配额开 vs 关在**全部流**
   （S1-S4 + R1）上结果**完全一致**——churn/SC2/ratio 逐项相等（S1-S4 + R1）**且** R1
   bridge 相等；配额动作面 = 0（逐流 d_sc2 == 0 且 d_churn == 0.0）。实现口径：机制单独
   （churn_slow/SC2_slow）vs 机制 + docs/249 配额叠加于慢层（apply_quota/fastcut_gate
   复用，docs/250 §六同款口径）。
2. **`[L3][机制][无配额]` CHURN_MECH**：配额完全关闭下，慢原型 churn_slow ≤ 0.5 于
   DAVIS R1 **且** S1/S2/S3/S4 全部（快原型不参与）。预注册推论：按构造 = 0.0（§1.2）。
3. **`[L3][机制][gist正确性]` GIST_CORRECT**：gist_cov(R1) ≥ 0.5（真值段边界 |Δ|≤1
   对应率；**目标保持 1.0**，docs/250 为 1.0000）；野流帧差近似对应率作诊断（不设判据）。
4. **`[L3][机制]` STABLE/STRUCT 保持**：ratio ≤ 1.5 于 S1-S4 **且** R1（构造性控制项，
   预测路径未动——数字与 docs/250 逐位一致）；SC2_slow > 0 于 S1-S4 **且** R1（慢类结构
   在野域与 DAVIS 均涌现）。
5. **`[L3][机制][行为证据]` PROMOTION**：n_promo > 0（全局）、n_recycle > 0（全局）、
   promoted 均值命中率 > 未升级均值命中率（升级非随机——报告两侧均值；确定性流无显著性
   检验，报告数字与分离程度，如实）。
6. **守卫（不进判据，实现正确性）**：docs/246 M=1.5 复现（**12/12**，配额关闭口径，
   R_QR_GUARD_D246，复用 run_guard_quota + guard_vs_d246 同一代码路径）。

**判定（预注册，冻结）**：
- **判据 1-5 全过 且 守卫 12/12 → `QUOTA_RETIRED`**（配额正式退役：docs/249 配额路径
  标注退役/保留为历史、代码中标记废弃）。
- **判据 1 不过 → `QUOTA_STILL_NEEDED`**（延迟定级未消除动作面，如实报告动作面在哪——
  逐流 Δ 与翻转原型明细）。
- **其余部分 → `PARTIAL`**（判据 1 过但 2-5 有不过，或守卫 < 12/12；如实报告）。
- 数据不可用（解码失败、流采样帧数过少）→ **QR_BLOCKED**。

### 1.8 回归守卫与内部复现（不进判据，实现正确性；预注册，冻结）

1. **R_QR_GUARD_D246（DAVIS，12/12）**：复用 fastcut_fix.run_guard_quota（SoftLoop +
   门 + 配额，docs/249/250 守卫同一代码路径）跑 DAVIS R0+R1（r=0.39885/k=3/α=0.2），
   配额关闭字段须复现 docs/246 M=1.5 工作点行：R0 SC2=3/churn 0.0000/ratio 0.907701；
   R1 SC1=11/SC2=6/churn 0.4545/ratio 0.951261；bridge_sw 0.8750/calib_sw 1.0/
   holdout_sw 0.7500/bridge_vid 0.8889/spurious 0（容差 1e-4）→ 12/12。
2. **R_QR_REPRO_RATIO（野域 + R1，内部控制，诊断级）**：DeferredLoop 的 ratio 须与
   docs/250 §3.3 逐位一致（容差 1e-4）：S1 1.155669 / S2 1.371908 / S3 0.732642 /
   S4 0.370964 / R1 0.951261——预测路径零改动的构造性控制项（机制改动只影响模式表路径，
   不影响 MAE 序列）。
3. 信息性诊断：四流 + R1 逐原型明细（created/promoted_at/final hits/kind）、快慢命中率
   两侧均值、野流 switch_corr、配额开/关逐项对照明细（churn/SC2/ratio/bridge + Δ）。

### 1.9 数据、工作点、流（冻结；与 docs/248/249/250 逐字）

- 数据源：`C:\Users\fa278\Downloads` 的 V1 studio_video_1759283839728.mp4 / V2
  41125413122-1-192.mp4 / V3 千军万马哦哦哦.mp4（docs/248 §1.2 冻结，同一批）。
- 预处理：cv2 顺序解码 + 间隔抽帧（step=max(1,round(T/500))）→ BGR→GRAY → resize
  160×120 (INTER_AREA) → uint8 → 流式输入（window=10）——cross_domain_test.
  load_sampled_frames 复用。
- 流：S1=V1 / S2=V2 / S3=V3 / S4=V1+V2+V3 拼接（docs/248 §1.5 逐字）；DAVIS R0
  （flamingo×5=400 帧）+ R1（9 视频拼接=588 帧，GT 段边界见 §1.6）。
- 工作点：r_slow=0.39885、r_fast=0.598275、k_consist_fast=1、**K_FINALIZE=hits_min_slow=3**
  （延迟定级）、k_promote=2（DEPRECATED/吸收）、k_decay=5、hits_min_fast=1、
  hits_min_slow=3、α=0.2、window=10、LOOP_CFG 原样（零重调）。

### 1.10 统计与安全纪律（冻结，docs/243-250 同款）

- 固定确定性流（无 RNG、无 jitter、无逐场景统计）；窗口级 mean±SD（ddof=1）+ bootstrap
  95% CI（2000 次，种子 20260828，critical_point 复用）。
- 每流一次运行（不重复、不重试后改参）；总预算 ≤ 60 分钟（预期解码 ~140s + 计算 ~10s，
  docs/248-250 同量级）。
- 安全纪律：stdout 只输出 ASCII 标签 + 每行一个数字的 R_QR_* 摘要块；运行经
  `powershell -NoProfile -Command "& python vision\quota_retire.py --tag qr *> logs\qr_qr.log; $c=$LASTEXITCODE; $b=(Get-Item 'logs\qr_qr.log').Length; Write-Output('exit='+$c+' bytes='+$b)"`
  包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）抽取；**禁止用 read
  工具读任何输出文件（logs/*.log、vision/out/results/*.json）**；Downloads/DAVIS 是数据
  （只读帧数/文件名元数据）；**未修改任何既有脚本**（新文件：vision/quota_retire.py）。

### 1.11 不重测项与重测项（预注册声明）

- **不重测**：STATE_PERSIST（docs/245-250 同款声明：本实验只改模式表升级判定，状态
  save/load 机制未动）；DAVIS 内 L2 判据（守卫 §1.8-1 承担实现正确性）；文档预测路径
  度量（pin/θ）逐位不变（预测路径零改动）。
- **重测（本实验判据 2-5）**：docs/250 的 CHURN_MECH/GIST_CORRECT/STABLE-STRUCT/
  PROMOTION 在本实验**重测**——延迟定级改动了机制行为（升级时点后移、升级候选保持快类），
  数字必然不同于 docs/250（churn_slow 按构造 = 0.0、SC1_fast/SC2_slow/n_promo/n_recycle
  会变），这正是本实验要测的机制差异；docs/250 的 QUOTA_ORTHOGONAL（诊断）被本实验的
  QUOTA_RETIRED（形式判据）取代。

---

## 二、实现（全部在最终运行前完成；判据 §1.7/旋钮 §1.3 冻结未动）

- 新文件 `vision/quota_retire.py`（import 复用 fastslow_test.FastSlowLoop/quota_on_slow/
  gist_metrics/build_entry_base/R_FAST/R_SLOW/HITS_MIN_FAST/HITS_MIN_SLOW/K_PROMOTE/
  K_DECAY/K_CONSIST_FAST、fastcut_fix.run_guard_quota、soft_match_test.ALPHA/HITS_MIN、
  cross_domain_test.load_sampled_frames/WILD_VIDEOS/STREAMS/RADIUS_L3/R_BASE_DAVIS/D246/
  DL_DIR/guard_vs_d246/scene_switch_diag、real_stream_test.load_video_frames/VIDEOS/WINDOW/
  RESIZE、real_recalib.bridge_metrics、stream_test.LOOP_CFG、critical_point.mean_sd/
  bootstrap_ci；**未修改任何既有脚本**）。
- `DeferredLoop(FastSlowLoop)`：**唯一机制改动 = 快匹配分支的升级判定行**
  （docs/250：`hits >= k_promote` → 本实验：`hits >= hits_min_slow`，延迟定级，定级门槛
  K_FINALIZE=hits_min_slow=3）；_on_window 其余逻辑与 docs/250 §1.4 逐字一致（慢优先 →
  快兜底 → 立即创建 → 窗口末回收）；finalize 继承 FastSlowLoop（churn_slow/SC2_slow/
  entry_log/promoted_log/命中率两侧均值同 docs/250 语义）。
- `quota_compare`：机制单独（churn_slow/SC2_slow）vs 机制 + docs/249 配额叠加于慢层
  （quota_on_slow 复用 = apply_quota + fastcut_gate，docs/250 §六同款口径）；churn/SC2/
  ratio 逐项比对（S1-S4 + R1）+ R1 bridge 比对。
- 实现记录：**无 mechanical bug 修复**（py_compile 预检通过，首运行即 exit=0）。
- 运行：`powershell -NoProfile -Command "& python vision\quota_retire.py --tag qr *>
  logs\qr_qr.log; ..."`，**exit=0**，elapsed **140.45 秒**（解码为主；预算 60 分钟）。
- 工件：`vision/out/results/qr_qr.json`（config/四流/R0/R1/判据/verdict/quota_retired/
  守卫/内部复现/promotion/timing）+ 摘要块 `logs/qr_qr.log`（9424 B，R_QR_* ASCII 数字行）。
- 复现命令：`python vision/quota_retire.py --tag qr`

---

## 三、结果（确定性流；数字工件 vision/out/results/qr_qr.json + logs/qr_qr.log 摘要块）

### 3.1 回归守卫（不进判据，实现正确性；R_QR_GUARD_D246）

DAVIS R0+R1 共享基座同一代码路径（fastcut_fix.run_guard_quota = SoftLoop + 门 + 配额，
docs/249/250 守卫同款）r=0.39885/k=3/α=0.2：R0 SC2=3 / churn 0.0000 / ratio 0.907701；R1
SC1=11 / SC2=6 / churn 0.4545 / ratio 0.951261；bridge_sw 0.8750 / calib_sw 1.0000 /
holdout_sw 0.7500 / bridge_vid 0.8889 / spurious 0——12 项（配额关闭字段）与 docs/246
M=1.5 工作点行**逐位一致，R_QR_GUARD_D246=1（12/12，容差 1e-4）**。共享基座无实现漂移。

### 3.2 内部复现（R_QR_REPRO_RATIO；预测路径零改动的构造性控制项）

DeferredLoop 的 ratio 与 docs/250 §3.3 **逐位一致**（容差 1e-4）：S1 1.155669 / S2
1.371908 / S3 0.732642 / S4 0.370964 / R1 0.951261——**5/5，R_QR_REPRO_RATIO=1**。
延迟定级只改模式表路径（升级判定），MAE 序列与 docs/245-250 逐位一致（机制改动的
作用域 = 原型层，预测路径零改动）。

### 3.3 野流 + DAVIS 机制测量表（配额完全关闭；确定性流 = 单值）

| 流 | 帧 | 窗 | ratio | SC1_fast | SC2_fast | SC1_slow | SC2_slow | churn_slow | churn_legacy | n_promo | n_recycle | 升级均值 | 未升级均值 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | 490 | 49 | 1.155669 | 19 | 19 | 5 | **5** | **0.0000** | 0.7368 | 5 | 12 | 6.2000 | 1.2857 |
| **S2** | 499 | 50 | 1.371908 | 11 | 11 | 4 | **4** | **0.0000** | 0.6364 | 4 | 5 | 10.7500 | 1.0000 |
| **S3** | 405 | 41 | 0.732642 | 19 | 19 | 7 | **7** | **0.0000** | 0.6316 | 7 | 12 | 3.8571 | 1.1667 |
| **S4** | 1394 | 140 | 0.370964 | 39 | 39 | 7 | **7** | **0.0000** | 0.8205 | 7 | 32 | 14.2857 | 1.2500 |
| R0（守卫诊断） | 400 | 40 | 0.907701 | 18 | 18 | 1 | **1** | **0.0000** | — | 1 | 16 | — | — |
| **R1** | 588 | 59 | 0.951261 | 32 | 32 | 5 | **5** | **0.0000** | 0.8438 | 5 | 23 | 5.4000 | 1.0741 |

（SC2_fast = SC1_fast 按定义：hits_min_fast=1、创建即 hits=1。**churn_slow = 0.0000 全流
按构造成立**（§1.2 推论）：延迟定级下慢原型只以"确认已满"（final hits ≥ hits_min_slow=3）
的形态存在——CHURN_MECH 的语义从"压 churn"变为"结构上不存在未完成慢原型"（诚实边界 1）。）

逐流升级明细（promoted_at:hits）：S1 3:8,6:4,16:8,25:5,37:6；S2 4:20,9:6,15:9,17:8；
S3 8:6,10:5,19:4,24:3,30:3,36:3,39:3；S4 3:27,6:9,16:21,25:17,37:20,49:3,138:3；
R1 见 §3.4（32 创建 / 5 升级）。升级候选（hits=2、docs/250 下"升级未完成"）在本机制下
**保持快原型并参与回收**——S1 12 次回收、S3 12 次、S4 32 次（半成品候选被回收，行为符合
预注册 §1.2 设计）。

### 3.4 R1 gist 正确性（真值段边界 = GT；判据 3）

- **GT 切换窗口** = [8, 13, 21, 30, 36, 41, 49, 54]（9 视频拼接切换帧
  [80,135,217,307,367,417,497,540] ÷ 10，0 基窗）。
- 快原型创建窗（32 个）= 0,1,2,3,6,7,8,9,13,14,15,16,18,20,21,22,23,27,30,32,35,36,37,
  38,41,42,43,47,54,55,56,57。
- **gist_cov = 0.8750（7/8，|Δ|≤1 窗）**——唯一未命中 = 切换 49（car-turn→motorbike，
  帧 497）：最近创建窗 47（|Δ|=2），**|Δ|≤2 覆盖 = 1.0000**；gist_prec = 0.5000
  （16/32 创建对齐边界）。**诚实：docs/250 的 gist_cov 1.0000 → 0.8750**——延迟定级下
  升级候选保持 r_fast 粗匹配，切换 49 附近创建时点偏移 1 窗；判据（≥ 0.5）通过，
  "目标保持 1.0"未达成（诚实边界 2）。
- bridge 诊断（快慢 entry_log 全量）：bridge_sw 0.8750、bridge_vid 0.8889（与 docs/250
  同值——机制未破坏跨度编码）。

### 3.5 配额开 vs 关逐项对照表（判据 1 QUOTA_RETIRED；S1-S4 + R1）

| 流 | churn_off | churn_on | Δchurn | SC2_off | SC2_on | ΔSC2 | ratio_off=ratio_on | R1 bridge_off=bridge_on |
|---|---|---|---|---|---|---|---|---|
| **S1** | 0.0000 | 0.0000 | 0.0000 | 5 | 5 | 0 | 1.155669（相等） | — |
| **S2** | 0.0000 | 0.0000 | 0.0000 | 4 | 4 | 0 | 1.371908（相等） | — |
| **S3** | 0.0000 | 0.0000 | 0.0000 | 7 | 7 | 0 | 0.732642（相等） | — |
| **S4** | 0.0000 | 0.0000 | 0.0000 | 7 | 7 | 0 | 0.370964（相等） | — |
| **R1** | 0.0000 | 0.0000 | 0.0000 | 5 | 5 | 0 | 0.951261（相等） | 0.8750 = 0.8750 |

**全部 5 流 churn/SC2/ratio 逐项相等（EQ=1 全过），R1 bridge 相等（EQ=1），逐流
d_sc2=0 / d_churn=0.0——配额动作面 = 0。**（配额开=关的相等性**不依赖**快切门
fastcut_gate 是否触发：fire=0 时 apply_quota 直接返回 off 值、fire=1 时慢层全已确认已满
也无从豁免——两个分支都逐项相等；ratio/bridge 相等按构造成立：配额为 finalize 级改判、
不改 MAE/entry_log。形式判据的真正内容 = 配额唯一能改判的慢层目标 churn/SC2 全流零 Δ，
诚实边界 3。）

### 3.6 判据逐项结果表（预注册 §1.7，冻结；每判据带 docs/247 层级标签）

| 判据 | 标签 | 冻结定义 | 数字 | 判定 |
|---|---|---|---|---|
| **QUOTA_RETIRED** | `[L3][机制][退役]` | 配额开 vs 关全流（S1-S4+R1）churn/SC2/ratio/bridge 逐项相等，动作面=0 | 全流 Δchurn=0.0000、ΔSC2=0、ratio 相等、R1 bridge 相等 | **✓ 通过**（d_sc2=0/d_churn=0.0 全流） |
| **CHURN_MECH** | `[L3][机制][无配额]` | 配额完全关闭，慢原型 churn ≤ 0.5（S1-S4+R1） | S1-S4/R1 全部 **0.0000**（按构造成立） | **✓ 通过**（max 0.0000） |
| **GIST_CORRECT** | `[L3][机制][gist正确性]` | R1 真值段边界对应率 ≥ 0.5（\|Δ\|≤1） | **0.8750（7/8）**；\|Δ\|≤2 覆盖 1.0000；野流诊断 0.1562-0.5714 | **✓ 通过**（docs/250 1.0000 → 0.8750，如实） |
| **STABLE/STRUCT 保持** | `[L3][机制]` | 全流 ratio ≤ 1.5 且 SC2_slow > 0 | ratio max 1.371908（与 docs/250 逐位一致）；SC2_slow min 4 | **✓ 全过**（REPRO_RATIO=1，构造性） |
| **PROMOTION** | `[L3][机制][行为证据]` | 升级数 > 0、回收数 > 0、升级命中率均值 > 未升级 | 28 / 84；3.8571-14.2857 vs 1.0000-1.2857 | **✓ 通过**（升级非随机） |

野流帧差近似 gist 诊断（switch_corr）：S1 0.2143 / S2 0.5714 / S3 0.1667 / S4 0.1562
（帧差尖峰近似段边界，无 GT；S3/S4 低 = docs/248 §3.4 已证高运动/剪辑流帧差尖峰灵敏度
不足——R1 GT 才是判据）。

---

## 四、判定结果

**R_QR_VERDICT = QUOTA_RETIRED。** 判定数字（确定性流，exit=0，elapsed 140.45 秒，
工件 qr_qr.json；守卫 12/12 配额关闭口径；内部复现 5/5）：

- **QUOTA_RETIRED（[L3][机制][退役]）✓**：配额开 vs 关在全部流（S1-S4 + R1）上
  churn/SC2/ratio 逐项相等、R1 bridge 相等（**全流 d_sc2=0、d_churn=0.0000，配额动作面
  = 0**）——docs/250 §六的 S2/S3/R1 翻转（Δchurn −0.2000/−0.2222/−0.1250、ΔSC2
  +1/+2/+1）**全部归零**。延迟定级消除了"升级未完成慢原型"这一形态：hits=2 的升级候选
  保持快原型（r_fast 匹配、受 k_decay 回收），慢原型只以"确认已满"（final hits ≥ 3）
  的形态存在 → 配额（对未完成慢原型的豁免）**无可豁免对象**。
- **CHURN_MECH（[L3][机制][无配额]）✓**：S1-S4/R1 慢原型 churn **全部 0.0000**（按构造
  成立——设计语义：结构上不存在未完成慢原型）。
- **GIST_CORRECT（[L3][机制][gist正确性]）✓**：R1 真值段边界对应率 **0.8750（7/8，
  |Δ|≤1）** ≥ 0.5（switch 49 的 |Δ|=2，|Δ|≤2 覆盖 1.0000；docs/250 的 1.0000 未保持，
  如实报告）。
- **STABLE/STRUCT 保持（[L3][机制]）✓**：四流 + R1 ratio 1.155669 / 1.371908 / 0.732642 /
  0.370964 / 0.951261 ≤ 1.5——与 docs/250 **逐位一致**（REPRO_RATIO=1，预测路径零改动，
  构造性控制项）；SC2_slow = 5 / 4 / 7 / 7 / 5 全部 > 0。
- **PROMOTION（[L3][机制][行为证据]）✓**：升级 28 次、回收 84 次（全局，均 > 0）；
  升级原型命中率均值 3.8571-14.2857 vs 未升级 1.0000-1.2857（全部流分离，升级非随机）。
- **判定（按 §1.7 冻结规则）**：判据 1-5 全过 且 守卫 12/12 → **QUOTA_RETIRED**——
  **配额正式退役**（docs/249 配额路径标注退役/保留为历史，见 §六）。

---

## 五、与 docs/250 对比（QUOTA_ORTHOGONAL 诊断 → QUOTA_RETIRED 形式判据）

### 5.1 配额叠加对照（本实验 vs docs/250 §六）

| 流 | docs/250 churn_slow | docs/250 配额 Δchurn | **本实验 Δchurn** | docs/250 SC2_slow | docs/250 配额 ΔSC2 | **本实验 ΔSC2** | 本实验 SC2_slow |
|---|---|---|---|---|---|---|---|
| **S1** | 0.2857 | 0.0000 | **0.0000** | 5 | 0 | **0** | 5 |
| **S2** | 0.2000 | **−0.2000** | **0.0000** | 4 | **+1** | **0** | 4 |
| **S3** | 0.3333 | **−0.2222** | **0.0000** | 6 | **+2** | **0** | 7 |
| **S4** | 0.2727 | 0.0000 | **0.0000** | 8 | 0 | **0** | 7 |
| **R1** | 0.2500 | **−0.1250** | **0.0000** | 6 | **+1** | **0** | 5 |

**docs/250 的"配额仍需"（S2/S3/R1 三个翻转）→ 本实验"配额可退役"（全流零 Δ）**：
配额的动作面 = "升级未完成慢原型"（hits=2、段预算 < hits_min_slow）在延迟定级下
**不存在**（候选保持快原型、可回收）→ 配额开/关逐项一致。

### 5.2 机制数字差异（延迟定级 vs docs/250 立即升级；诚实）

| 量 | docs/250 | 本实验 | 差异解释（冻结机制选择） |
|---|---|---|---|
| churn_slow（全流） | 0.2000-0.3333 | **0.0000** | 延迟定级：慢原型只以确认已满形态存在（§1.2 推论） |
| SC1_fast | 16/12/21/32/33 | 19/11/19/39/32 | 升级时点后移 → 候选保持 r_fast 匹配 → 创建/吸收形态变化（S4 跨视频拼接流创建增多） |
| SC2_slow | 5/4/6/8/6 | 5/4/7/7/5 | 延迟定级下升级更"确认"（S3 6→7 升、S4 8→7、R1 6→5 微降） |
| n_promo / n_recycle | 40 / 67 | **28 / 84** | 升级更少更确认、回收更多（半成品候选被回收 = docs/250"短命但正确"行为在候选层的延续） |
| R1 gist_cov | 1.0000 | **0.8750** | 升级候选保持 r_fast 粗匹配 → switch 49 创建时点偏移 1 窗（|Δ|=2；|Δ|≤2 覆盖仍 1.0） |
| ratio（全流+R1） | 逐位 | **逐位一致** | 预测路径零改动（REPRO_RATIO=1，构造性控制项） |

---

## 六、配额退役落实记录

1. **docs/249 标注**：docs/249 顶部已加**退役横幅**——短段配额（评估规则层）被 docs/250
   机制级修复 + docs/251 延迟定级正式取代：机制下配额开/关结果完全一致（动作面 = 0），
   配额路径**标记退役/废弃，保留为历史**（docs/248 S3 churn 0.6250 的评估规则层修复记录
   仍有效，作为历史；退役指"快慢双原型 + 延迟定级下配额不再需要"）。
2. **代码标记废弃**：新文件 `vision/quota_retire.py` 的 config.mechanism 明确记录
   "quota fully off in mechanism"、配额退役判据（QUOTA_RETIRED）为正式判据、k_promote=2
   标注 DEPRECATED/SUPERSEDED（被延迟定级吸收）。既有脚本 fastcut_fix.py
   （apply_quota/fastcut_gate，docs/249 配额实现）**因"禁止修改既有脚本"纪律未做 inline
   废弃注释**——废弃声明落在新代码 + docs/249 标注；建议后续维护在 fastcut_fix.py 补
   DEPRECATED 注释（不在本实验范围，诚实声明，见诚实边界 4）。
3. **引用关系**：docs/249（配额定义，已标退役）← docs/250（机制级修复 + QUOTA_ORTHOGONAL
   诊断"配额仍需"）← **docs/251（本实验：延迟定级消除动作面 → 配额正式退役）**；
   docs/251 引用 docs/245-250（数据/工作点/机制/判据标签规范/守卫同源）。
4. **复现**：`python vision/quota_retire.py --tag qr`（确定性，exit=0，elapsed 140.45 秒）。

---

## 七、诚实边界

1. **churn_slow=0.0 是构造成果，不是"学习变好"的独立证据**：延迟定级下慢原型只以
   "确认已满"（final hits ≥ hits_min_slow=3）形态存在（§1.2 推论），CHURN_MECH 全 0 按
   设计成立——其语义从 docs/250 的"压 churn"变为"结构上不存在未完成慢原型"。学习有效性
   的独立证据由 **GIST**（R1 GT cov 0.8750 ≥ 0.5）、**PROMOTION**（升级命中率 3.86-14.29
   vs 未升级 1.00-1.29）、**SC2_slow > 0**、**守卫 12/12** 承担。
2. **R1 gist_cov 1.0 → 0.8750（未保持"目标 1.0"，如实）**：唯一未命中 = 切换 49
   （car-turn→motorbike，帧 497），最近创建窗 47（|Δ|=2）——延迟定级下升级候选保持
   r_fast 粗匹配使该处创建时点偏移 1 窗；**|Δ|≤2 覆盖仍 1.0000**，判据（≥ 0.5）通过。
   不声称"gist 正确性保持 1.0"。
3. **配额开=关的 ratio/bridge 相等是构造性的**：配额是 finalize 级改判（只改
   hits_min_eff 判定，不改 MAE/entry_log）→ ratio/bridge 相等按构造成立；形式判据
   QUOTA_RETIRED 的真正内容是**配额唯一能改判的慢层目标 churn/SC2 全流逐项零 Δ**
   （d_sc2=0/d_churn=0.0000 全流）——运行显式比对并逐项报告。
4. **"代码中标记废弃"的边界**：fastcut_fix.py（docs/249 配额实现）因"禁止修改既有脚本"
   纪律未加 inline 废弃注释；废弃标记落在新文件 quota_retire.py（config/文档字符串）+ 
   docs/249 退役横幅。若后续允许修改既有脚本，应给 apply_quota/fastcut_gate 补
   DEPRECATED 注释（建议，不在本实验范围）。
5. **k_promote 语义变更（冻结机制选择，非回调）**：docs/250 的"hits ≥ k_promote=2 升级
   即固化候选"被替换为"hits ≥ hits_min_slow=3 即定级门槛（K_FINALIZE）"；k_promote=2
   保留在配置中标注 DEPRECATED/SUPERSEDED（行为零作用，仅文档追溯）。这改变了升级时点
   （§五 5.2 的 SC1_fast/n_promo/gist 差异均由此而来），是预注册 §1.2/§1.3 冻结的设计。
6. **确定性**：单次确定性运行 exit=0（py_compile 预检通过，无 mechanical 修复；docs/250
   曾需 2 次 mechanical 修复——本实验首运行即成功）；无 RNG、无 jitter、无重试后改参；
   REPRO_RATIO=1（ratio 与 docs/250 逐位一致）证明预测路径零改动。
7. **野流 gist 是诊断（无 GT）**：帧差 μ+3σ 近似段边界，switch_corr 0.1562-0.5714
   （S3/S4 低 = docs/248 §3.4 已证高运动/剪辑流帧差尖峰灵敏度不足）；R1 真值段边界才是
   判据（cov 0.8750）。帧差只用于诊断，不进入任何机制决策（§1.2 冻结）。
8. **守卫语义**：12/12 是**共享基座**（SoftLoop + 数据加载 + 桥度量，docs/249/250 守卫
   同款代码路径）的配额关闭字段复现 docs/246；DeferredLoop 自身是新增代码，其正确性由
   ① 判据全过、② 确定性单轮 exit=0、③ REPRO_RATIO=1（ratio 逐位一致）共同承担。
9. **PROMOTION 无显著性检验**（确定性单次运行）：升级均值 3.8571-14.2857 vs 未升级
   1.0000-1.2857 的分离为行为证据，报告两侧数字，不声称统计检验。升级均值 ≥ 3 按构造
   成立（定级门槛 = 3），未升级均值 < 3（任何达 3 者已被定级）——分离本身是机制定义，
   非随机性检验。
10. **配额退役的语义范围**：本实验证明"快慢双原型 + 延迟定级下配额开/关结果完全一致"
    （docs/250 机制 + 本实验定级规则的组合）；**不声称** docs/249 配额在单层 SoftLoop
    原系统下无价值（docs/249 修复了 docs/248 S3 churn 0.6250 的评估规则层问题，历史有效）
    ——退役指"机制层消除动作面后配额不再需要"，docs/249 保留为历史记录。
11. **安全纪律**：stdout 只输出 R_QR_* ASCII 摘要行；运行经 powershell 包装重定向
    （exit=0，elapsed 140.45 秒）；数字用 extract_r.py 纯 python 正则抽取（未用 read 工具
    读取任何输出文件：logs/qr_qr.log、vision/out/results/qr_qr.json）；Downloads/DAVIS
    只读帧数/文件名元数据（复用 cross_domain_test 加载）；**未修改任何既有脚本**（新文件
    vision/quota_retire.py）。

---

## 八、一句话

> **QUOTA 退役实验完成（确定性流，exit=0，elapsed 140.45 秒，verdict=QUOTA_RETIRED）：
> 用**延迟定级（deferred finalization）**消除 docs/250 QUOTA_ORTHOGONAL 残留的最后动作面
> ——快→慢升级不在 hits ≥ k_promote=2 时立即发生，仅当原型累积满 hits_min_slow=3 次
> 重匹配才最终化为慢原型；hits=2 的升级候选**保持快原型**（r_fast 匹配、受 k_decay=5
> 回收，docs/250"短命但正确"），**不产生半成品慢状态**——慢原型只以"确认已满"形态存在，
> 配额（对未完成慢原型的豁免）**再无动作面**。[L3][机制][退役] QUOTA_RETIRED 过（配额开
> vs 关在 S1-S4+R1 全流 churn/SC2/ratio 逐项相等、R1 bridge 相等，全流 d_sc2=0/
> d_churn=0.0000——docs/250 的 S2/S3/R1 翻转 −0.2000/−0.2222/−0.1250 与 +1/+2/+1 全部
> 归零）；[L3][机制][无配额] CHURN_MECH 过（慢原型 churn 全流 0.0000，按构造成立）；
> [L3][机制][gist正确性] GIST_CORRECT 过（R1 真值段边界对应率 0.8750，7/8，|Δ|≤2 覆盖
> 1.0——docs/250 的 1.0000 未保持，如实）；[L3][机制] STABLE/STRUCT 保持过（ratio max
> 1.3719 与 docs/250 逐位一致、SC2_slow min 4 > 0）；[L3][机制][行为证据] PROMOTION 过
> （28 升级/84 回收，升级命中率 3.86-14.29 vs 未升级 1.00-1.29）；守卫 R_QR_GUARD_D246=1
> （12/12）；内部复现 REPRO_RATIO=1（5/5）→ **QUOTA_RETIRED**：docs/249 配额路径标注
> 退役/保留为历史、新代码标注废弃（k_promote DEPRECATED）。诚实边界：churn_slow=0 为
> 构造成果（学习有效性由 GIST/PROMOTION/SC2/守卫承担）、gist_cov 1.0→0.8750（switch 49
> 的 |Δ|=2）、配额开=关的 ratio/bridge 相等为构造性（真正内容 = churn/SC2 零 Δ）。**
