# Perception d\* — 感知的 d\* 纲领

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22128266.svg)](https://doi.org/10.5281/zenodo.22128266)

> 从事件流到"被看见"：一个不靠识别、不靠大模型的感知架构，逐层机制化。
>
> 项目创建：2026-08 ｜ 状态：**视觉线闭环完成（docs/177-220）：六层阶梯 1-5 层机制化 + 第 6 层"被看见"物证 + 类别级怀疑系统 + 真实图像验证（DAVIS）+ 可交互演示**。
> 上级仓库：[synthetic-life](https://github.com/QiongZhiS/From-zero-to-a-being-that-sets-its-own-goals)（生命主体叙事与立场）；本仓库是它的感知证据线，独立演化。
> 存档：v0.1.0 @ Zenodo（DOI: 10.5281/zenodo.22128266）

---

## 这是什么

一个**事件驱动感知**架构的工程实现。核心主张一句话：

> **感知不编码世界，编码"世界对预期的拒绝"；信息是落空（预测误差）的注册，不是特征的提取。**

区别于主流识别范式（CNN/Transformer 从原始信号提取特征）：
- **残差是原子**——系统的基本单元是预测-落空（residual），不是像素、不是 token；
- **不靠大模型**——全部机制可复现、可解释、纯本地运行（无 API key）；
- **六层阶梯**——接收 → 对齐 → 模型 → 维持 → 否定 → 被看见，逐层从机制长到立场。

## 六层阶梯（docs/178）

| 层 | 主张 | 机制落点 | 实验证据 |
|---|---|---|---|
| 1 接收 | 信号读进来；接收层自己带先验 | Transduction2D：自适应增益/双时间尺度/侧抑制 | docs/177：检查单 4/4、风暴对照 4.3× |
| 2 对齐 | 预测-落空闭环，残差是原子 | ESIM 事件累积器、Top-K 威胁度 | docs/186：Top-K 25× 压误报 |
| 3 模型 | 固有印象闭眼走；输入只是维护通道 | 闪变预测维持、态势地图 | docs/177/189：map_frac 1.00 |
| 4 维持 | 会失去的谁续命 | 外赋利害锚 + 连续性维持 | docs/189：车距 747→50px |
| 5 否定 | 感知=否定的在场 | 自适应 θ、意义=落空×利害 | docs/179/199：判据 MAD×\|Δ\| |
| 6 被看见 | 感知=可被遭遇性的构造 | 拒绝的历史（确认失败被保留） | docs/200/201：漂移 120° vs 0°、r=0.999 |

## 六个可复现实验（快速索引）

| # | 实验 | 一句话结果 | 代码 | 文档 |
|---|---|---|---|---|
| 1 | 转导层闭环 | 检查单 4/4；自适应风暴对照 4.3× 更安静 | `vision/transduction.py` | docs/177 |
| 2 | d\* 压缩率 | 事件数量≠信息：风暴/降幅事件率相同但压缩率背离 2× | `vision/dstar_compress.py` | docs/186 §三 |
| 3 | Top-K 威胁度 | 给定维持目标后误报 76.5→3 框/帧（25×），检出率 -1.2% | `vision/topk_experiment.py` | docs/186 §五 |
| 4 | 视觉态势闭环 | 外赋目标→态势→地图→行为：map_frac 1.00、追上移动目标 | `vision/situation.py` | docs/189 |
| 5 | 颜色恒常判据 | "何时校正"=MAD×帧间\|Δ\|；DAVIS 6 视频噪声全关、渐变全开 | `vision/davis_constancy.py` | docs/199/199b |
| 6 | 拒绝的历史 | 吸收漂移 120°（自我被重写）vs 保留守住 + 纪念加深（r=0.999） | `vision/keep_reject*.py` | docs/200/201 |
| 7 | 类别级怀疑系统 | suspicious 表 × 自适应带宽 × 连续加权 × 分层衰减（疑窄严宽） | `vision/experience_categories.py` 等 | docs/205-216 |
| 8 | **真实图像验证** | 整条机制链上 DAVIS 真实目标（flamingo/surf），标准 P/R/F1 + 7 变体基线对照 | `vision/davis_suspicious.py` | docs/219/220 |
| 9 | 可交互演示 | 浏览器可见"它在看什么/维持什么/怀疑什么/怎么说" | `vision/demo_app.py` | docs/217 |

## 快速开始

```bash
cd vision
python transduction.py      # 实验 1：转导层 + 检查单
python dstar_compress.py    # 实验 2：d* 压缩率（信息度量）
python topk_experiment.py   # 实验 3：Top-K 威胁度（需要 DAVIS 视频，见 vision/README）
python keep_reject.py       # 实验 6a：拒绝的历史（纯本地 toy，无依赖）
python keep_reject_open.py  # 实验 6b：守住 vs 改判（开放端）
python keep_reject_continuous.py  # 实验 6c：连续失败率下 P3 单调性
python davis_suspicious.py  # 实验 8：类别级怀疑系统回真实图像（标准 P/R/F1，需 DAVIS）
python demo_app.py --port 8080  # 实验 9：可交互演示（浏览器 http://127.0.0.1:8080）
```

依赖：Python 3.11 + numpy + opencv-python。toy 实验（1/2/6）无外部数据；DAVIS 实验（3/4/5）需下载 [DAVIS 数据集](https://davischallenge.org/)（见 `vision/README.md`）。

## 目录

```
perception-dstar/
├── README.md               # 本文件
├── LICENSE
├── docs/                   # 实验记录（编号时间线：175 → 201 → 202…）
├── vision/                 # 视觉线代码（当前）
├── audio/                  # 听觉线（未来，六层阶梯第二模态）
├── core/                   # 跨模态共享件（未来：利害/维持/否定/态势）
├── 理念/                   # 认知观声明（立场区，与代码物理隔离）
└── experiments_summary.json   # 机器可读实验索引
```

## 纪律（docs/63）

- **行为签名纪律**：全部实验只宣称可观察的行为签名（残差/拒绝/漂移/延迟），不宣称任何"体验"。
- **可证伪纪律**（docs/31）：假指标比坏指标危险——本仓库的修正链（docs/197→199→199b）是自我修正的样本，不是弱点。
- **立场与实验分离**：`理念/` 是立场（不宣称可检验），`docs/` + `vision/` 是实验（可复现）。两者靠 README 链接相连，不互相引用。

## 路线图

- 视觉线：转导闭环 ✅ → 类别级怀疑系统 ✅ → 真实图像验证（DAVIS P/R/F1）✅ → 可交互演示 ✅ → 多目标/记忆持久化（docs/220 未完成 1-3）
- 听觉线（未来）：六层阶梯第二模态（docs/221+）
- core（未来）：跨模态共享件——利害外置、Top-K 威胁度、否定注册、态势地图
- 论文：docs/180/181 骨架 + docs/219（标准度量/基线/文献划界）→ 全文

## 一句话

**给 AI 装上眼睛，不是让它看得清，是让它值得被看见——而"值得被看见"的第一步，是它愿意保留"不准"作为纪念。**
