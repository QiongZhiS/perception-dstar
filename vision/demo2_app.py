"""vision/demo2_app.py — 巅峰组装：实时可交互演示（docs/217 §五 的"巅峰版"）。

把项目**现行主机制链**（docs/241-257 已实证）组装成一个接**真实视频流**的可交互演示：
观众打开浏览器就看到"它在看什么 / 预测什么 / 短命记住什么 / 固化什么 / 怀疑什么 /
怎么说"。docs/217 的 demo_app.py（合成红球）保留不动；本文件是它的"巅峰版"。

====================================================================================
设计说明（冻结方案，先设计后实现；docs/63+247 纪律：演示只重组已实证机制，零重调）
====================================================================================

0. 一句话
  浏览器打开 http://127.0.0.1:8081（默认端口避开 demo_app 的 8080）即见：
  真实输入（DAVIS 9 视频 / R1 拼接 / 野流 V1-V3）→ 预测残差与事件掩码实时叠加
  （docs/235 预测路径）→ 快慢双原型 + 延迟定级（docs/250/251 DeferredLoop：什么被
  短命记住又遗忘、什么被固化）→ A1 动态基质实时读数（docs/257 四判据行为签名：
  ratio 有界、SC2 不膨胀不塌缩）→ 语言转录（docs/204 精神规则化：机制事件 → 人话）。

1. 架构（沿用 demo_app 零依赖模式，stdandard library + numpy + cv2）
  - 机制循环线程（后台）：逐帧喂 DeferredLoop（import 复用 quota_retire），逐窗口
    （window=10）推进；每帧更新"当前帧 + 事件掩码 + 残差热图"，每窗口更新
    "MAE 曲线 / 原型列表 / 升级与回收事件 / A1 读数 / 语言流"。
  - HTTP 服务线程：ThreadingHTTPServer，GET /（页面）、/state（快照 JSON）、
    /act（交互）、/health。
  - 共享状态：threading.Lock 保护的快照 dict。机制线程**每次整体重建**快照条目
    （新鲜对象、不复用可变对象），/state 只读快照——无"改大小中迭代"竞态。

2. 机制接入（import 复用、零改写；预测路径零改动）
  - DeferredLoop = quota_retire.DeferredLoop（docs/251 现行主机制，逐字 import 复用；
    run_deferred_stream 仅 selftest 复用）。
  - 旋钮全部为 docs/250/251 冻结值（r_slow=0.39885 / r_fast=0.598275 /
    hits_min_fast=1 / hits_min_slow=3（=K_FINALIZE）/ k_decay=5 / k_consist_fast=1 /
    α=0.2 / LOOP_CFG / window=10），零重调、零新增。
  - 加载函数复用：real_stream_test.load_video_frames（DAVIS JPEG → 灰度 160×120，
    docs/243 同款预处理）、cross_domain_test.load_sampled_frames（Downloads 野流
    间隔抽帧采样）、WILD_VIDEOS/DL_DIR/VIDEOS/WINDOW/RESIZE。
  - 事件掩码/残差热图为**显示路径镜像**（非机制改动）：读回路自身状态
    （loop.bg_fast 为该帧预测、step() 返回的 θ/db），按 docs/235 公式重算
    |c|−db>θ 与 |L−bg_fast| 用于显示；不进任何机制决策。窗口事件量 E/U 与
    事件并集 _ev_win 读机制自身口径。

3. 流与切换
  - 13 条流：9 个 DAVIS 单视频 / R1（九视频拼接=588 帧，8 处真实场景切换=GT 段边界，
    docs/243/257）/ 野流 V1/V2/V3（WILD_VIDEOS 冻结）。
  - 单视频流播放到头循环（R0 半真实妥协同款）；R1/野流同样循环。
  - **切换视频不重置机制**（A1 连续性：新场景被吸收进既有记忆——观众看到切换 =
    真实场景切换被结构抓取，docs/257 R1 精神）；"重置记忆"按钮清零（新建 DeferredLoop
    实例并清空轨迹/语言/事件）。默认不重置。

4. 前端面板（内嵌 HTML/JS，中文标注；零外部依赖）
  - 画面区：当前帧灰度 + 事件掩码红叠加 + 残差热图（JET）+ 视频名/帧号/窗口号；
  - 状态区：MAE 曲线（最近 ~120 窗口，canvas）、ratio（末/首四分之一，A1 有界性，
    随窗口累积标注"待足量"）、θ/db、E/U、SC2_slow/SC1_fast/n_promo/n_recycle、
    结构密度上界检查（SC2_slow ≤ 3×max(1,⌈n_win/10⌉)，docs/257 不膨胀判据的实时版）；
  - 记忆区（核心看点）：慢原型（亮绿=固化记忆）与快原型（灰=gist 短命，hits≥2 标
    "候选"）按最近活动排序，标注 pid/hits/kind/创建窗/最近窗/μ；回收事件滚动日志
    （"窗口 X 遗忘了一个 gist：曾经是 #pid，被记住 N 次"）；
  - 语言流：规则化转录（docs/204 三层精神，演示版规则化，见 §5）；
  - 交互：视频下拉（13 流）、播放/暂停、单步（前进一个窗口）、重置记忆、速度 1×/2×/4×。

5. 语言流规则（docs/204 精神：翻译不是生成；每句 1:1 指回机制状态字段）
  - 高残差创建        → 「新东西出现了」（事件量 E px 超出已有记忆，创建快原型 #pid）
  - hits≥hits_min_slow 升级（延迟定级最终化）→ 「我确认它了，记下来」（重匹配 N 次固化为慢）
  - k_decay 回收      → 「这个不重要，忘了」（遗忘一个 gist：曾记住 N 次）
  - MAE 上升（>1.3×近 8 窗均值）→ 「世界让我意外」（MAE 数值 + 倍数）
  每窗口最多 2 句（升级优先）；无 LLM、无现象层陈述（docs/63 纪律：行为签名转录）。

6. 诚实边界（演示 ≠ 新实验）
  - 展示的是 docs/241-257 已实证机制链的**重组**（docs/217 §四 精神）：全部旋钮为
    冻结值零重调；不宣称任何新能力、不做新判据——面板上的"ratio/SC2/密度上界"
    是 A1 行为签名的实时读数（docs/257 已判 DYNAMIC_PASS_REAL），演示只让观众
    看见它们随流累积，不重复判定。
  - 事件掩码/热图是显示镜像；野流 S1-S4 无真值标签（docs/243 labels=None 同款）；
    语言是模板转录，不是 LLM 也不是"系统在体验"。
  - 演示只吃 DAVIS JPEG 与 Downloads mp4（数据），不读任何输出文件
    （logs/*.log、vision/out/results/*.json）。
  - stdout 只输出少量 ASCII 状态（D2_READY/D2_STATE/D2_ERROR/D2_SELFTEST_*），
    不打印日志原文。

7. 颜色通道（本升级：docs/205/208/199b/219 机制复用，import 零改写）
  - **画面升级为彩色原帧**：DAVIS JPEG（854×480 彩色）+ GT 掩码 PNG 由
    davis_suspicious.load_video 复用加载（零改写）；机制仍吃灰度 160×120 不变
    （real_stream_test.load_video_frames 原样）；显示合成 = 原彩帧（320×240）
    + 事件掩码红叠 + GT 目标区绿色轮廓，前端可切换图层（layers=color,event,gt）。
  - **目标区 = GT 掩码定位（诚实声明：定位免费，非检测器，docs/219 同款）**；
    野流（Downloads mp4）无 GT 掩码 → 无掩码图层、颜色确认不可用，页面诚实标注。
  - **颜色确认/怀疑 = DavisMind 机制复用**（davis_suspicious 逐字 import）：
    任务色相（外赋利害 docs/188，交互设定：flamingo 170°/surf 72°/自定义滑条，
    跨视频保持）；确认 frac>thr 且中位色距<30°（docs/195 三通道口径）；被拒入史
    suspicious[hue] 记账（docs/205）；自适应带宽 bw=k_band×见过的同类色相 std
    （docs/208）；时间门=色相 MAD 集中 AND 帧间 |Δshift| EWMA（docs/199b）；
    白平衡/色温读数（bg_gain/g_base/n_corr，docs/219 相对基线校正条件）。
  - 最小适配（不改 davis_suspicious 本体）：运行时换任务色相/基线阈值、场景切换
    （用户换流/R1 剪切/循环回绕）重标定白平衡基线、颜色语言事件逐帧去重。
  - 语言流追加颜色事件：「它在变色？」（时间门开）→「我怀疑它是 X° 色相」
    （suspicious 记账）→「我确认它了（任务色）」（claimed）→「它的颜色漂移了，
    带宽变宽了」（自适应带宽沉积）。

8. 用法
  python vision/demo2_app.py --port 8081 --video flamingo
  python vision/demo2_app.py --selftest            # 机制冒烟（不启 HTTP）
  python vision/demo2_app.py --port 8081 --fps 8   # 1× 帧率调节
  从项目根目录运行（与全部 vision 脚本一致；脚本会自行校正 cwd）。
"""
import argparse
import base64
import json
import math
import os
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

warnings.filterwarnings("ignore")

# 基于脚本自身位置解析 vision 模块——从任何目录运行都能 import（不依赖 cwd）
HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(HERE)          # 项目根（vision/ 的父目录）
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if os.getcwd() != _PROJ and os.path.isdir(os.path.join(_PROJ, "vision", "out", "davis")):
    os.chdir(_PROJ)                    # 全部 vision 脚本约定：从项目根运行（数据路径相对）

import cv2  # noqa: E402

# ---- 机制 import 复用（docs/251/257 现行主机制；零改写） ----
from quota_retire import DeferredLoop, run_deferred_stream  # noqa: E402
from fastslow_test import (R_FAST, R_SLOW, HITS_MIN_FAST, HITS_MIN_SLOW,  # noqa: E402
                           K_PROMOTE, K_DECAY, K_CONSIST_FAST)
from stream_test import LOOP_CFG  # noqa: E402
from real_stream_test import load_video_frames, VIDEOS, WINDOW, RESIZE  # noqa: E402
from cross_domain_test import load_sampled_frames, WILD_VIDEOS, DL_DIR  # noqa: E402
# 颜色通道机制复用（docs/205/208/199b/219；import 零改写——davis_suspicious 本体不动）
from davis_suspicious import (  # noqa: E402
    DavisMind, load_video as load_davis_video, target_obs,
    bg_stats, bg_gain, TemporalGateDavis, circ,
)

# ---- 流定义 ----
# 野流文件名（cross_domain_test.WILD_VIDEOS 冻结；V1 演播室 / V2 B站 / V3 快剪）
WILD_FILE = {vid: name for vid, name in WILD_VIDEOS}

STREAM_LABELS = {
    "flamingo": "DAVIS · flamingo（火烈鸟）",
    "surf": "DAVIS · surf（冲浪）",
    "bear": "DAVIS · bear（熊）",
    "camel": "DAVIS · camel（骆驼）",
    "dog": "DAVIS · dog（狗）",
    "blackswan": "DAVIS · blackswan（黑天鹅）",
    "car-turn": "DAVIS · car-turn（车转弯）",
    "motorbike": "DAVIS · motorbike（摩托车）",
    "soccerball": "DAVIS · soccerball（足球）",
    "R1": "DAVIS · R1 九视频拼接（真实场景切换）",
    "V1": "野流 · V1 演播室人物",
    "V2": "野流 · V2 B站用户",
    "V3": "野流 · V3 快速剪辑",
}
ALL_STREAMS = list(STREAM_LABELS.keys())

DEFAULT_VIDEO = "flamingo"
FRAME_INTERVAL = 0.25          # 1×：每帧 0.25s（4 fps；每窗口 2.5s）
MAE_SERIES_MAX = 120           # 前端 MAE 曲线保留的窗口数
LOG_MAX = 26                   # 事件/语言流保留条数
PROTO_MAX = 40                 # 前端原型列表上限

# ---- 颜色通道常量（docs/219 机制复用；任务色相=外赋利害 docs/188） ----
DISP_SIZE = (320, 240)         # 显示分辨率（机制仍吃灰度 160×120）
THUMB_SIZE = 160               # 目标区彩色缩略边长
# 任务预设：docs/219 §3 文档化数字——flamingo 任务色相 170.0°/thr 0.60、
# surf 72.0°/thr 0.40（真实目标确认余量不同）。自定义滑条保持当前 thr。
TASK_PRESETS = {
    "flamingo": (170.0, 0.60, "flamingo"),
    "surf": (72.0, 0.40, "surf"),
}
DEFAULT_TASK_HUE = TASK_PRESETS["flamingo"][0]
DEFAULT_TASK_THR = TASK_PRESETS["flamingo"][1]


def _jpeg(bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


class Demo2Engine:
    """机制循环线程：逐帧喂 DeferredLoop，逐窗口推进；维护共享快照。"""

    def __init__(self, video=DEFAULT_VIDEO, fps=4.0):
        self.loop = DeferredLoop(window=WINDOW, **LOOP_CFG)
        self.cur = video if video in ALL_STREAMS else DEFAULT_VIDEO
        self.streams = {}            # name -> list[uint8 160x120 灰度]（懒加载）
        self.streams_c = {}          # name -> (彩色帧 list, GT 掩码 list|None)（懒加载）
        self.loading = None          # 正在加载的流名
        self.load_error = None
        self.idx = 0
        self.playing = True
        self.speed = 1               # 1/2/4
        self.frame_interval = 1.0 / max(1.0, float(fps))
        self._step_target = None     # 单步目标窗口数（len(loop.mae) 达到即停）
        self._stop = False
        self._last_nw = 0            # 已处理窗口数（len(loop.mae) 的本地镜像）
        self._known_recycled = set() # 已报告回收的 pid
        self.lang_log = []           # [{w, tag, text}]
        self.ev_log = []             # [{w, type, pid, text}]
        self.snap = {}               # 共享快照（机制线程整体重建条目）
        self.lock = threading.Lock()
        self.status = "starting"
        # ---- 颜色通道（docs/219 DavisMind 机制复用；任务色相跨视频保持） ----
        self.task_hue = DEFAULT_TASK_HUE
        self.task_thr = DEFAULT_TASK_THR
        self.task_preset = "flamingo"
        self.mind = DavisMind(task_hue=self.task_hue, wb=True, thr_base=self.task_thr)
        self.show_color = True       # 图层：原彩帧 / 事件掩码 / GT 目标轮廓
        self.show_event = True
        self.show_gt = True
        self._cuts = set()           # 当前流场景剪切帧位置（R1 内部 8 处）
        self._prev_fi = None         # 循环回绕检测（回绕=场景重标定）
        self._wb_reset = False       # 用户换流 → 机制线程内重标定白平衡基线
        self._prev_gate = False      # 颜色语言事件去重
        self._prev_rej = 0
        self._prev_claimed = False
        self._prev_bw = 0.0
        self._last_mad = None

    # ---------------- 流加载（懒加载；野流慢，DAVIS 快） ----------------
    def _load_stream(self, name):
        if name == "R1":
            out = []
            for v in VIDEOS:
                out.extend(load_video_frames(v))
            return out
        if name in VIDEOS:
            return load_video_frames(name)
        if name in WILD_FILE:
            p = os.path.join(DL_DIR, WILD_FILE[name])
            frames, _step, _total = load_sampled_frames(p)
            return frames
        raise KeyError("unknown stream: %s" % name)

    def _get_stream(self, name):
        if name in self.streams:
            return self.streams[name]
        if self.loading == name:
            return None
        self.loading = name
        self.load_error = None
        with self.lock:
            self.status = "loading-stream"
        try:
            fr = self._load_stream(name)
            self.streams[name] = fr
            with self.lock:
                self.status = "running"
        except Exception as e:  # noqa: BLE001 —— 演示脚本：加载失败只记状态，不崩
            self.load_error = str(e)[:200]
            with self.lock:
                self.status = "error"
        finally:
            self.loading = None
        return self.streams.get(name)

    # ---------------- 彩色流加载（DAVIS 复用 davis_suspicious.load_video；野流无彩色） ----------------
    def _load_stream_color(self, name):
        """彩色帧 + GT 掩码（DAVIS 数据路径；机制复用 davis_suspicious.load_video 零改写）。
        DAVIS 单视频/ R1 拼接：返回 (彩色帧, 掩码)；野流：返回 (None, None)——不二次
        解码超大 mp4（V1 约 800MB），显示回退灰度→BGR，颜色面板诚实标注无掩码。"""
        if name in VIDEOS:
            return load_davis_video(name)
        if name == "R1":
            fr, mk, cuts, acc = [], [], set(), 0
            for v in VIDEOS:
                f2, m2 = load_davis_video(v)
                acc += len(f2)
                cuts.add(acc)                 # 下一视频首帧索引 = 场景剪切位置
                fr.extend(f2)
                mk.extend(m2)
            self._cuts = cuts
            return fr, mk
        if name in WILD_FILE:
            return None, None                 # 野流：无 GT 掩码、无彩色通道（诚实标注）
        raise KeyError("unknown stream: %s" % name)

    def _get_stream_color(self, name):
        if name in self.streams_c:
            return self.streams_c[name]
        if self.loading == name:
            return None
        self.loading = name
        try:
            cc = self._load_stream_color(name)
            self.streams_c[name] = cc
        except Exception as e:  # noqa: BLE001 —— 演示脚本：彩色通道加载失败退化为灰度显示
            self.load_error = str(e)[:200]
            self.streams_c[name] = (None, None)
        finally:
            self.loading = None
        return self.streams_c.get(name)

    # ---------------- 显示路径镜像（读回路自身状态，不进机制决策） ----------------
    def _display_views(self, gray, res):
        L = np.log(np.maximum(gray.astype(np.float32), 1.0))
        resid = np.abs(L - self.loop.bg_fast)          # 回路对该帧的预测残差 |c|
        theta, db = float(res["theta"]), float(res["db"])
        ev = np.maximum(resid - db, 0.0) > theta       # |c|−db>θ（docs/235 公式镜像）
        p98 = float(np.percentile(resid, 98))
        p98 = p98 if p98 > 1e-6 else 1.0
        norm = np.clip(resid / p98, 0.0, 1.0)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return ev, heat, float(resid.mean())

    def _build_scene(self, fi, gray, ev):
        """场景合成（前端可切换图层）：原彩帧（或灰度）→ 事件掩码红叠 → GT 目标轮廓绿。
        显示分辨率 DISP_SIZE；机制仍吃灰度 160×120 不变。"""
        cc = self.streams_c.get(self.cur)
        if cc is not None and cc[0] and len(cc[0]) > 0:
            base = cv2.resize(cc[0][fi % len(cc[0])], DISP_SIZE,
                              interpolation=cv2.INTER_AREA)
        else:
            base = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), DISP_SIZE,
                              interpolation=cv2.INTER_AREA)
        if not self.show_color:
            base = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
                                cv2.COLOR_GRAY2BGR)
        if self.show_event:
            ev_disp = cv2.resize(ev.astype(np.uint8), DISP_SIZE,
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            base[ev_disp] = (0, 0, 255)                # 事件像素红叠加
        if self.show_gt and cc is not None and cc[1] is not None and len(cc[1]) > 0:
            mk = cc[1][fi % len(cc[1])]
            if mk.max() > 0:
                mk_disp = cv2.resize(mk, DISP_SIZE, interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(mk_disp, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(base, cnts, -1, (0, 255, 0), 2)   # GT 目标区轮廓
        return _jpeg(base)

    @staticmethod
    def _target_thumb(img, mask):
        """目标区彩色缩略：GT 掩码包围盒裁剪（机制同源：白平衡校正后原图），
        等比缩放后居中填充到 THUMB_SIZE×THUMB_SIZE。无目标→空串。"""
        if mask is None or mask.max() == 0:
            return ""
        ys, xs = np.where(mask > 0)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pad = 12
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1 = min(img.shape[1] - 1, x1 + pad)
        y1 = min(img.shape[0] - 1, y1 + pad)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        h, w = crop.shape[:2]
        scale = THUMB_SIZE / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((THUMB_SIZE, THUMB_SIZE, 3), np.uint8)
        ox, oy = (THUMB_SIZE - nw) // 2, (THUMB_SIZE - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = crop
        return _jpeg(canvas, quality=75)

    # ---------------- 语言转录（docs/204 精神规则化；每句 1:1 指回机制状态） ----------------
    def _utter(self, w, mae_w, E, created, promoted, recycled):
        utts = []
        for pl in promoted:
            utts.append({"w": w, "tag": "固化",
                         "text": ("窗口 %d：我确认它了，记下来——原型 #%d 重匹配 %d 次，"
                                  "固化为慢原型（延迟定级）") % (w, pl["pid"], pl["hits"])})
        for cl in created[:1]:
            utts.append({"w": w, "tag": "创建",
                         "text": ("窗口 %d：新东西出现了——事件量 %dpx 超出已有记忆，"
                                  "创建快原型 #%d") % (w, E, cl)})
        for cl in recycled[:1]:
            utts.append({"w": w, "tag": "遗忘",
                         "text": ("窗口 %d：这个不重要，忘了——遗忘了一个 gist"
                                  "（原型 #%d，曾被记住 %d 次）") % (w, cl["pid"], cl["final_hits"])})
        if len(self.loop.mae) >= 9 and not utts:
            prev = np.asarray(self.loop.mae[-9:-1], float)
            base = float(prev.mean())
            if base > 1e-9 and mae_w > 1.3 * base:
                utts.append({"w": w, "tag": "意外",
                             "text": ("窗口 %d：世界让我意外——MAE %.3f，比近 8 窗均值高"
                                      " %.1f×") % (w, mae_w, mae_w / base)})
        return utts[:2]

    # ---------------- 窗口收尾：快照重建 + 事件/语言 ----------------
    def _on_window_done(self):
        loop = self.loop
        w = loop._win - 1                              # 已完成窗口号（0 基）
        mae_w = float(loop.mae[-1])
        E = int(loop.energy_trace[-1])
        U = int(loop.up_trace[-1])

        # 事件差分（创建 / 升级 / 回收）
        created = [cl["pid"] for cl in loop.created_log if cl["created"] == w]
        promoted = [pl for pl in loop.promoted_log if pl["promoted_at"] == w]
        recycled = [cl for cl in loop.created_log
                    if cl["recycled"] and cl["pid"] not in self._known_recycled]
        for cl in recycled:
            self._known_recycled.add(cl["pid"])
        ev_new = []
        for pid in created:
            ev_new.append({"w": w, "type": "create", "pid": pid,
                           "text": "创建快原型 #%d（事件量 %dpx）" % (pid, E)})
        for pl in promoted:
            ev_new.append({"w": w, "type": "promote", "pid": pl["pid"],
                           "text": "升级为慢原型 #%d（hits=%d）" % (pl["pid"], pl["hits"])})
        for cl in recycled:
            ev_new.append({"w": w, "type": "recycle", "pid": cl["pid"],
                           "text": "回收快原型 #%d（final hits=%d）" % (cl["pid"], cl["final_hits"])})
        self.ev_log.extend(ev_new)
        if len(self.ev_log) > LOG_MAX:
            self.ev_log = self.ev_log[-LOG_MAX:]

        # 语言流
        utts = self._utter(w, mae_w, E, created, promoted, recycled)
        self.lang_log.extend(utts)
        if len(self.lang_log) > LOG_MAX:
            self.lang_log = self.lang_log[-LOG_MAX:]

        # A1 读数（docs/257 口径：ratio = 末/首四分之一窗口 MAE 均值比）
        mae_arr = np.asarray(loop.mae, float)
        q = max(1, len(mae_arr) // 4)
        q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
        q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
        ratio = (q4 / q1) if q1 > 0 else 0.0           # 除零守卫
        n_windows = len(loop.mae)
        density_bound = 3 * max(1, int(math.ceil(n_windows / 10.0)))

        sc1_fast = loop.n_created_fast
        sc2_slow = sum(1 for p in loop.prototypes
                       if p["kind"] == "slow" and p["hits"] >= loop.hits_min_slow)
        sc1_slow = len(loop.promoted_log)

        protos = []
        for p in loop.prototypes:
            protos.append({
                "pid": p["pid"], "kind": p["kind"], "hits": p["hits"],
                "created": p["created"], "last_active": p["last_active"],
                "n_match": p["n_match"], "promoted_at": p.get("promoted_at"),
                "mu": [round(float(p["mu"][0]), 3), round(float(p["mu"][1]), 3)],
                "candidate": int(p["kind"] == "fast" and p["hits"] >= 2),
            })
        protos.sort(key=lambda z: (-z["last_active"], z["pid"]))
        matched = loop.match_trace[-1] if loop.match_trace else (None, None)

        with self.lock:
            self.snap["win"] = w
            self.snap["E"] = E
            self.snap["U"] = U
            self.snap["mae_last"] = round(mae_w, 6)
            self.snap["mae_series"] = [round(x, 6) for x in loop.mae[-MAE_SERIES_MAX:]]
            self.snap["q1"] = round(q1, 6)
            self.snap["q4"] = round(q4, 6)
            self.snap["ratio"] = round(ratio, 6)
            self.snap["ratio_n"] = n_windows
            self.snap["theta"] = round(float(loop.theta_trace[-1]), 6) if loop.theta_trace else None
            self.snap["db"] = round(float(loop.db_trace[-1]), 6) if loop.db_trace else None
            self.snap["sc1_fast"] = sc1_fast
            self.snap["sc2_slow"] = sc2_slow
            self.snap["sc1_slow"] = sc1_slow
            self.snap["n_promo"] = loop.n_promoted
            self.snap["n_recycle"] = loop.n_recycled
            self.snap["density_bound"] = density_bound
            self.snap["density_ok"] = int(sc2_slow <= density_bound)
            self.snap["protos"] = protos
            self.snap["n_fast"] = sum(1 for p in protos if p["kind"] == "fast")
            self.snap["n_slow"] = sum(1 for p in protos if p["kind"] == "slow")
            self.snap["events"] = list(self.ev_log)
            self.snap["lang"] = list(self.lang_log)
            self.snap["matched"] = [matched[0], matched[1]]
            self.snap["gate"] = "参与（E≥10）" if E >= 10 else "静默（E<10）"

    # ---------------- 颜色机制步进（逐帧；复用 davis_suspicious，零改写） ----------------
    def _color_step(self, fi):
        """GT 掩码定位目标区 → 时间门+白平衡 → 观测（frac/hue）→ 确认/怀疑记账。
        复用 DavisMind/bg_stats/bg_gain/target_obs/circ（davis_suspicious 逐字 import）；
        最小适配：运行时把任务色相/基线阈值写入 mind、场景切换重标定门基线、
        颜色语言事件逐帧去重。返回 (snap 字段 dict, 语言事件 list)。"""
        cc = self.streams_c.get(self.cur)
        out, utts = {}, []
        if cc is None:
            return out, utts                       # 彩色流尚未加载（首帧可能）
        cfr, masks = cc
        if not cfr:
            # 野流：无彩色通道、无 GT 掩码——诚实标注，阅历（suspicious）跨视频保留
            out = {
                "has_mask": False,
                "task_hue": round(float(self.task_hue), 2),
                "task_preset": self.task_preset,
                "frac": 0.0, "hue": None, "color_dist": None,
                "thr": None, "bw": 0.0, "k": self.mind.k(None),
                "trusted": bool(self.mind.trusted), "consec": int(self.mind.consec),
                "persist": int(self.mind.persist), "rej_total": int(self.mind.rej_total),
                "ok": False, "claimed": False,
                "suspicious": {int(round(float(kk))): int(vv)
                               for kk, vv in self.mind.rejected.items()},
                "gate": {"open": False, "d_mu": None, "delta_gate": 4.0,
                         "warmup": 8, "mad_gate": 40.0},
                "wb": {"n_corr": int(self.mind.n_corr)},
                "note": "该流无 GT 掩码（定位免费依赖掩码）且不二次解码野流——颜色确认"
                        "不可用；阅历（suspicious 表）跨视频保留",
            }
            return out, utts
        fi = fi % len(cfr)
        has_mask = masks is not None and len(masks) > 0
        # 机制线程内消费换流标记 + 场景剪切/循环回绕 → 白平衡基线重标定
        if self._wb_reset or fi in self._cuts or \
                (self._prev_fi is not None and fi < self._prev_fi):
            self.mind.g_base = None
            self.mind.gate = TemporalGateDavis()
            self._prev_gate = False
            self._prev_claimed = False
            self._prev_bw = 0.0
            self._prev_rej = 0
            self._wb_reset = False
        self._prev_fi = fi

        mind = self.mind
        mind.task_hue = self.task_hue               # 运行时换任务（docs/188 外赋利害）
        mind.thr_base = self.task_thr
        bgr = cfr[fi]
        if not has_mask:
            out = {
                "has_mask": False,
                "task_hue": round(float(self.task_hue), 2),
                "task_preset": self.task_preset,
                "frac": 0.0, "hue": None, "color_dist": None,
                "thr": None, "bw": 0.0, "k": mind.k(None),
                "trusted": bool(mind.trusted), "consec": int(mind.consec),
                "persist": int(mind.persist), "rej_total": int(mind.rej_total),
                "ok": False, "claimed": False,
                "suspicious": {int(round(float(kk))): int(vv)
                               for kk, vv in mind.rejected.items()},
                "gate": {"open": False, "d_mu": None, "delta_gate": 4.0,
                         "warmup": 8, "mad_gate": 40.0},
                "wb": {"n_corr": int(mind.n_corr)},
                "note": "该流无 GT 掩码（定位免费依赖掩码）——颜色确认不可用；"
                        "阅历（suspicious 表）跨视频保留",
            }
            return out, utts
        mask = masks[fi]
        # 时间门 + 白平衡（docs/199b；相对基线增益变化才校正，docs/219 §四 1）
        exclude = cv2.dilate(mask, np.ones((21, 21), np.uint8)) if mask.max() > 0 \
            else np.zeros_like(mask)
        s, _, mad = bg_stats(bgr, exclude)
        g = bg_gain(bgr, exclude)
        gate_open = mind.gate.update(s, mad)
        self._last_mad = mad
        do_corr = False
        img = bgr
        if g is not None:
            if mind.g_base is None:
                mind.g_base = g.copy()
            do_corr = gate_open and float(np.max(np.abs(g - mind.g_base))) > 0.12
        if do_corr:
            img = np.clip(bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)
            mind.n_corr += 1
        frac, hue = target_obs(img, mask, self.task_hue)
        ok = mind.step(frac, hue, mask.max() == 0)
        claimed = bool(mind.trusted and ok)
        d_mu = getattr(mind.gate, "d_mu", None)
        out = {
            "has_mask": True,
            "task_hue": round(float(self.task_hue), 2),
            "task_preset": self.task_preset,
            "frac": round(float(frac), 4),
            "hue": None if hue is None else round(float(hue), 2),
            "color_dist": None if hue is None else round(float(circ(hue, self.task_hue)), 2),
            "thr": None if hue is None else round(float(mind.thr(hue)), 4),
            "bw": 0.0 if hue is None else round(float(mind.bw(hue)), 3),
            "k": int(mind.k(hue)),
            "trusted": bool(mind.trusted), "consec": int(mind.consec),
            "persist": int(mind.persist), "rej_total": int(mind.rej_total),
            "ok": bool(ok), "claimed": claimed,
            "suspicious": {int(round(float(kk))): int(vv)
                           for kk, vv in mind.rejected.items()},
            "gate": {"open": bool(gate_open),
                     "d_mu": None if d_mu is None else round(float(d_mu), 3),
                     "delta_gate": mind.gate.delta_gate,
                     "warmup": mind.gate.warmup,
                     "mad_gate": mind.gate.mad_gate},
            "wb": {"shift": None if s is None else round(float(s), 1),
                   "mad": None if mad is None else round(float(mad), 1),
                   "g_cur": None if g is None else [round(float(x), 3) for x in g],
                   "g_base": None if mind.g_base is None
                   else [round(float(x), 3) for x in mind.g_base],
                   "n_corr": int(mind.n_corr), "do_corr": int(do_corr)},
            "thumb_img": self._target_thumb(img, mask),
        }
        utts = self._color_utter(fi, hue, frac, claimed, gate_open)
        return out, utts

    def _color_utter(self, fi, hue, frac, claimed, gate_open):
        """颜色语言事件（docs/204 转录：每句 1:1 指回颜色机制状态字段；逐帧去重）。"""
        utts = []
        w = self.loop._win - 1 if self.loop._win else 0
        # ① 时间门开（真染色×帧间一致 → "它在变色？"）
        if gate_open and not self._prev_gate:
            dm = getattr(self.mind.gate, "d_mu", None)
            utts.append({"w": w, "tag": "变色",
                         "text": ("帧 %d：它在变色？——时间门开（背景色相集中 "
                                  "MAD=%.1f° 且帧间一致 |Δshift|=%.2f°）"
                                  % (fi, self._last_mad or 0.0, dm or 0.0))})
        # ② 被拒记账（suspicious 表累计）
        if self.mind.rej_total > self._prev_rej and hue is not None:
            utts.append({"w": w, "tag": "怀疑",
                         "text": ("帧 %d：我怀疑它是 %.0f° 色相——被拒记账 "
                                  "suspicious[%.0f]=%d"
                                  % (fi, hue, hue, self.mind.rejected.get(hue, 0)))})
        # ③ 确认（claimed 从假到真）
        if claimed and not self._prev_claimed:
            utts.append({"w": w, "tag": "颜色确认",
                         "text": ("帧 %d：我确认它了（任务色 %.0f°）——目标区 %.0f%% "
                                  "是任务色" % (fi, self.task_hue, 100.0 * frac))})
        # ④ 自适应带宽变宽（见过的变异沉积，docs/208）
        bw = self.mind.bw(hue) if hue is not None else 0.0
        if bw > self._prev_bw + 0.5 and self._prev_bw > 0:
            utts.append({"w": w, "tag": "漂移",
                         "text": ("帧 %d：它的颜色漂移了，带宽变宽了 "
                                  "（%.1f°→%.1f°，见过的变异在沉积）"
                                  % (fi, self._prev_bw, bw))})
        self._prev_gate = gate_open
        self._prev_rej = self.mind.rej_total
        self._prev_claimed = claimed
        self._prev_bw = bw
        return utts[:2]

    # ---------------- 每帧快照（画面） ----------------
    def _update_frame(self, gray, res, fi):
        ev, heat, resid_mean = self._display_views(gray, res)
        scene = self._build_scene(fi, gray, ev)
        color_snap, color_lang = self._color_step(fi)
        frames = self.streams.get(self.cur)
        total = len(frames) if frames else 0
        with self.lock:
            self.snap["frame_img"] = scene
            self.snap["heat_img"] = _jpeg(heat)
            self.snap["frame_idx"] = self.idx
            self.snap["total_frames"] = total
            self.snap["win_inflight"] = len(self.loop._frame_buf)
            self.snap["resid_mean"] = round(resid_mean, 6)
            self.snap.update(color_snap)
            if color_lang:
                self.lang_log.extend(color_lang)
                if len(self.lang_log) > LOG_MAX:
                    self.lang_log = self.lang_log[-LOG_MAX:]

    # ---------------- 机制主循环 ----------------
    def run(self):
        try:
            while not self._stop:
                with self.lock:
                    playing, speed = self.playing, self.speed
                    step_target = self._step_target
                if not playing and step_target is None:
                    time.sleep(0.05)
                    continue
                frames = self._get_stream(self.cur)
                if frames is None:
                    time.sleep(0.2)
                    continue
                self._get_stream_color(self.cur)   # 彩色流懒加载（DAVIS 含 GT 掩码）
                fi = self.idx % len(frames)
                gray = frames[fi]
                self.idx += 1
                res = self.loop.step(gray)
                self._update_frame(gray, res, fi)
                if len(self.loop.mae) > self._last_nw:
                    self._last_nw = len(self.loop.mae)
                    self._on_window_done()
                if step_target is not None:
                    if len(self.loop.mae) >= step_target:
                        with self.lock:
                            self._step_target = None
                    time.sleep(0.01)
                else:
                    time.sleep(self.frame_interval / speed)
        except Exception as e:  # noqa: BLE001 —— 演示脚本：异常只打一行 ASCII
            with self.lock:
                self.status = "error: %s" % str(e)[:200]
            print("D2_ERROR %s" % str(e)[:200])

    def _snap_meta(self):
        with self.lock:
            return (self.playing, self.speed, self.status, self.cur,
                    self.loading, self.load_error)

    # ---------------- 交互（HTTP 线程调用；只改控制字段，机制线程消费） ----------------
    def act(self, action):
        with self.lock:
            if action == "play":
                self.playing = True
                self._step_target = None
            elif action == "pause":
                self.playing = False
            elif action == "step":
                self.playing = False
                self._step_target = len(self.loop.mae) + 1   # 前进一个窗口
            elif action == "speed1":
                self.speed = 1
            elif action == "speed2":
                self.speed = 2
            elif action == "speed4":
                self.speed = 4
            elif action == "reset":
                self._reset_mem()
            elif action.startswith("video="):
                name = action.split("=", 1)[1]
                if name in ALL_STREAMS:
                    self.cur = name
                    self.idx = 0            # 机制不重置（A1 连续性）：仅换流位置
                    self._cuts = set()      # 新流的剪切位置由加载时重建
                    self._wb_reset = True   # 场景切换 → 白平衡基线重标定（机制线程内）
            elif action == "task=flamingo" or action == "task=surf":
                preset = action.split("=", 1)[1]
                self.task_hue, self.task_thr, self.task_preset = TASK_PRESETS[preset]
            elif action.startswith("taskhue="):
                try:
                    h = float(action.split("=", 1)[1])
                    if 0.0 <= h <= 180.0:
                        self.task_hue = h
                        self.task_preset = "custom"
                except ValueError:
                    pass
            elif action.startswith("layers="):
                parts = action.split("=", 1)[1].split(",")
                self.show_color = "color" in parts
                self.show_event = "event" in parts
                self.show_gt = "gt" in parts
        return {"ok": True, "action": action}

    def _reset_mem(self):
        """重置记忆：新建 DeferredLoop 实例（同旋钮）与 DavisMind（同任务色相），
        清空轨迹/语言/事件。调用方必须已持有 self.lock（act 内调用；Lock 不可重入）。"""
        self.loop = DeferredLoop(window=WINDOW, **LOOP_CFG)
        self.mind = DavisMind(task_hue=self.task_hue, wb=True, thr_base=self.task_thr)
        self.idx = 0
        self._last_nw = 0
        self._known_recycled = set()
        self.lang_log = []
        self.ev_log = []
        self.streams = {k: v for k, v in self.streams.items() if k == self.cur} \
            if self.cur in self.streams else {}
        self._prev_gate = False
        self._prev_rej = 0
        self._prev_claimed = False
        self._prev_bw = 0.0
        self._prev_fi = None
        for k in ("win", "E", "U", "mae_last", "mae_series", "q1", "q4", "ratio",
                  "ratio_n", "theta", "db", "sc1_fast", "sc2_slow", "sc1_slow",
                  "n_promo", "n_recycle", "density_bound", "density_ok", "protos",
                  "n_fast", "n_slow", "events", "lang", "matched", "gate",
                  "has_mask", "task_hue", "task_preset", "frac", "hue", "color_dist",
                  "thr", "bw", "k", "trusted", "consec", "persist", "rej_total",
                  "ok", "claimed", "suspicious", "wb", "note", "thumb_img"):
            self.snap.pop(k, None)

    # ---------------- 快照（HTTP 线程读；条目均为新鲜对象） ----------------
    def snapshot(self, tick):
        with self.lock:
            playing, speed, status, cur, loading, load_err = \
                self.playing, self.speed, self.status, self.cur, self.loading, self.load_error
            snap = dict(self.snap)
        snap.update({
            "tick": tick,
            "status": status,
            "loading": loading,
            "load_error": load_err,
            "playing": playing,
            "speed": speed,
            "video": cur,
            "video_label": STREAM_LABELS.get(cur, cur),
            "streams": STREAM_LABELS,
            "window": WINDOW,
            "layers": {"color": self.show_color, "event": self.show_event,
                       "gt": self.show_gt},
            "task_hue": round(float(self.task_hue), 2),
            "task_preset": self.task_preset,
            "knobs": {
                "r_fast": round(R_FAST, 6), "r_slow": round(R_SLOW, 6),
                "hits_min_fast": HITS_MIN_FAST, "hits_min_slow": HITS_MIN_SLOW,
                "k_promote": K_PROMOTE, "k_decay": K_DECAY,
                "k_consist_fast": K_CONSIST_FAST,
            },
        })
        return snap


# ---------------- 机制冒烟（--selftest；不启 HTTP） ----------------
def selftest():
    """DeferredLoop 于 R0（flamingo×5=400 帧）若干窗口：SC2_slow≥1（结构涌现）、
    n_recycle≥0、ratio 计算无除零、run_deferred_stream 输出键齐全。
    颜色通道（docs/219 机制复用）：flamingo 目标区中位色相≈170°、clean 帧确认
    成立（claimed 通过率≥0.7）、suspicious 表结构正确（被拒后记账 {hue: int}）。"""
    t0 = time.time()
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0 = allv["flamingo"] * 5
    out, loop = run_deferred_stream(r0)
    ok_keys = all(k in out for k in ("ratio", "sc1_fast", "sc2_slow", "sc1_slow",
                                     "churn_slow", "n_promo", "n_recycle",
                                     "entry_log", "created_log"))
    sc2 = int(out["sc2_slow"])
    n_rec = int(out["n_recycle"])
    n_prom = int(out["n_promo"])
    ratio = float(out["ratio"])
    q1 = float(out["mae_q1"])
    ratio_finite = math.isfinite(ratio) and q1 > 0 and ratio > 0
    ok = int(ok_keys and sc2 >= 1 and n_rec >= 0 and ratio_finite)

    # ---- 颜色通道自检（docs/219 数字：flamingo 任务色相 170.0°） ----
    c_ok = 0
    c_hue = float("nan")
    c_confirm = 0.0
    c_bw = 0.0
    c_susp_n = 0
    try:
        from davis_suspicious import load_video, target_obs, DavisMind  # noqa: F401
        frs, mks = load_video("flamingo")
        obs = [target_obs(f, m, 0.0)[1] for f, m in zip(frs, mks)]
        c_hue = float(np.median([h for h in obs if h is not None]))
        hue_ok = int(140.0 <= c_hue <= 180.0)          # 探测标定 170（docs/219 §2.1）
        mind = DavisMind(task_hue=170.0, wb=True, thr_base=0.60)
        oks = []
        for i in range(30):                            # clean 帧：确认应成立
            frac, hue = target_obs(frs[i], mks[i], 170.0)
            okf = mind.step(frac, hue, mks[i].max() == 0)
            oks.append(bool(mind.trusted and okf))
        c_confirm = float(np.mean(oks))
        confirm_ok = int(c_confirm >= 0.7)
        c_bw = float(mind.bw(170.0))
        bw_ok = int(c_bw > 0.0)                        # 见过 ≥3 次同类色相 → 带宽沉积
        # 被拒记账结构：目标区加冷增益（色相偏移）→ 确认拒 → suspicious 记账
        fr2 = frs[0].copy()
        g2 = np.array([1.0, 1.0, 0.6])
        fr2[mks[0] > 0] = np.clip(fr2[mks[0] > 0].astype(np.float32) * g2,
                                  0, 255).astype(np.uint8)
        frac2, hue2 = target_obs(fr2, mks[0], 170.0)
        mind.step(frac2, hue2, False)
        sus = mind.rejected
        sus_ok = int(len(sus) > 0 and all(isinstance(k, float) and isinstance(v, int)
                                          for k, v in sus.items()))
        c_susp_n = len(sus)
        c_ok = int(hue_ok and confirm_ok and bw_ok and sus_ok)
    except Exception as e:  # noqa: BLE001
        print("D2_SELFTEST_COLOR_ERROR=%s" % str(e)[:120])
    print("D2_SELFTEST_OK=%d" % ok)
    print("D2_SELFTEST_FRAMES=%d" % len(r0))
    print("D2_SELFTEST_WINDOWS=%d" % out["n_windows"])
    print("D2_SELFTEST_KEYS=%d" % int(ok_keys))
    print("D2_SELFTEST_SC2_SLOW=%d" % sc2)
    print("D2_SELFTEST_N_PROMO=%d" % n_prom)
    print("D2_SELFTEST_N_RECYCLE=%d" % n_rec)
    print("D2_SELFTEST_Q1=%.6f" % q1)
    print("D2_SELFTEST_RATIO=%.6f" % ratio)
    print("D2_SELFTEST_RATIO_FINITE=%d" % int(ratio_finite))
    print("D2_SELFTEST_HUE=%.1f" % c_hue)
    print("D2_SELFTEST_HUE_OK=%d" % int(c_hue == c_hue and 140.0 <= c_hue <= 180.0))
    print("D2_SELFTEST_CONFIRM_RATE=%.3f" % c_confirm)
    print("D2_SELFTEST_CONFIRM_OK=%d" % int(c_confirm >= 0.7))
    print("D2_SELFTEST_BW=%.3f" % c_bw)
    print("D2_SELFTEST_SUSPICIOUS_STRUCT=%d" % int(c_susp_n > 0))
    print("D2_SELFTEST_SUSPICIOUS_N=%d" % c_susp_n)
    print("D2_SELFTEST_COLOR_OK=%d" % c_ok)
    print("D2_SELFTEST_ELAPSED=%.2f" % (time.time() - t0))
    return 0 if (ok and c_ok) else 1


# ---------------- 前端页面（内嵌 HTML/JS，中文标注） ----------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>感知主体 · 巅峰组装演示（真实视频流）</title>
<style>
  body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #0e0e0e;
         color: #d8d8d8; margin: 16px; }
  h1 { font-size: 17px; margin: 4px 0; }
  h3 { font-size: 14px; margin: 6px 0; color: #9cf; }
  h4 { font-size: 12px; margin: 8px 0 4px; color: #888; }
  .sub { font-size: 11px; color: #888; margin: 2px 0 10px; }
  .row { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }
  .panel { background: #171717; border: 1px solid #2c2c2c; border-radius: 8px;
           padding: 10px 12px; min-width: 300px; }
  img { border: 1px solid #444; border-radius: 4px; width: 320px; height: 240px;
        vertical-align: top; }
  table { border-collapse: collapse; font-size: 12px; }
  td, th { padding: 2px 8px; border-bottom: 1px solid #232323; text-align: left; }
  .btn { background: #2a6; color: #fff; border: none; padding: 6px 12px; margin: 2px;
         border-radius: 4px; cursor: pointer; font-size: 12px; }
  .btn.gray { background: #4a4a4a; } .btn.warn { background: #a63; }
  .btn.danger { background: #a33; } .btn.active { outline: 2px solid #ff9; }
  select { background: #222; color: #ddd; border: 1px solid #444; padding: 4px;
           border-radius: 4px; font-size: 12px; }
  .proto { font-size: 12px; padding: 2px 6px; margin: 2px 0; border-radius: 4px;
           font-family: Consolas, monospace; }
  .proto.slow { color: #8f8; background: #14301a; border: 1px solid #2c5a34; }
  .proto.fast { color: #9a9a9a; background: #1c1c1c; border: 1px solid #333; }
  .proto .cand { color: #ff9; }
  .ev { font-size: 12px; padding: 1px 4px; }
  .ev.create { color: #ff9; } .ev.promote { color: #8f8; }
  .ev.recycle { color: #f97; } .ev.reset { color: #9cf; }
  .lang { font-size: 12px; padding: 2px 4px; border-left: 3px solid #555; margin: 2px 0;
          background: #121212; }
  .lang.固化 { border-color: #8f8; color: #cfc; }
  .lang.创建 { border-color: #ff9; color: #ffe9a8; }
  .lang.遗忘 { border-color: #f97; color: #ffc4a8; }
  .lang.意外 { border-color: #9cf; color: #cfe6ff; }
  .lang.变色 { border-color: #ff9; color: #fff3c0; }
  .lang.怀疑 { border-color: #f97; color: #ffd0b0; }
  .lang.颜色确认 { border-color: #f8f; color: #ffd9ff; }
  .lang.漂移 { border-color: #8ff; color: #d0f4ff; }
  .logbox { max-height: 150px; overflow-y: auto; font-size: 12px; }
  .layerlabel { font-size: 12px; color: #bbb; margin-right: 10px; cursor: pointer; }
  .huebar { position: relative; height: 12px; border-radius: 3px; margin: 4px 0 8px;
            border: 1px solid #333;
            background: linear-gradient(to right,
              hsl(0,80%,55%), hsl(60,80%,55%), hsl(120,80%,55%), hsl(180,80%,55%),
              hsl(240,80%,55%), hsl(300,80%,55%), hsl(360,80%,55%)); }
  .huebar .marker { position: absolute; top: -3px; width: 2px; height: 18px;
                    background: #fff; box-shadow: 0 0 4px #000; }
  .susrow { display: flex; align-items: center; gap: 6px; font-size: 12px; margin: 2px 0; }
  .susrow .swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #444; }
  .susrow .bar { height: 10px; border-radius: 2px; min-width: 2px; }
  .susrow .task { color: #ff9; font-weight: bold; }
  input[type=range] { vertical-align: middle; }
  .stat { font-size: 12px; }
  .ok { color: #8f8; } .no { color: #f77; }
  canvas { background: #0a0a0a; border: 1px solid #333; border-radius: 4px; }
</style>
</head>
<body>
<h1>感知主体 · 巅峰组装演示（docs/217 §五 → 现行主机制链）</h1>
<p class="sub">
机制链：预测残差（docs/235）→ 快慢双原型 + 延迟定级（docs/250/251）→
A1 动态基质（docs/257）→ 语言转录（docs/204）｜真实输入：DAVIS 视频流 + 野流（docs/243）｜
旋钮全部为冻结值零重调；面板读数为已实证机制的实时重组，非新实验
</p>
<div class="row">
  <div class="panel">
    <h3>画面区 — 它在看什么（原彩）/ 预测哪里落空</h3>
    <div>
      <img id="scene" alt="当前帧 + 图层"><img id="heat" alt="残差热图"><img id="thumb" alt="目标区彩色缩略">
    </div>
    <div class="stat" id="frameinfo"></div>
    <div class="stat" style="margin-top:6px">
      <label class="layerlabel"><input type="checkbox" id="ly_color" checked> 原彩帧</label>
      <label class="layerlabel"><input type="checkbox" id="ly_event" checked> 事件掩码</label>
      <label class="layerlabel"><input type="checkbox" id="ly_gt" checked> GT 目标轮廓</label>
      <span id="masknote" style="color:#f97"></span>
    </div>
    <div style="margin-top:8px">
      <select id="videosel"></select>
      <button class="btn" id="playbtn">暂停</button>
      <button class="btn gray" id="stepbtn">单步（一个窗口）</button>
      <button class="btn danger" id="resetbtn">重置记忆</button>
      <button class="btn gray spd" data-s="1">1×</button>
      <button class="btn gray spd" data-s="2">2×</button>
      <button class="btn gray spd" data-s="4">4×</button>
    </div>
    <div class="stat" id="streaminfo" style="margin-top:4px;color:#9cf"></div>
  </div>
  <div class="panel">
    <h3>状态区 — A1 动态基质（docs/257 行为签名实时读数）</h3>
    <canvas id="maecurve" width="520" height="120"></canvas>
    <div class="stat" id="stats"></div>
  </div>
  <div class="panel" style="min-width:340px">
    <h3>颜色感知 — 它在确认 / 怀疑什么颜色（docs/205/208/199b 机制复用）</h3>
    <div class="stat" id="colorstate"></div>
    <h4>任务色相（外赋利害 docs/188 · 跨视频保持）</h4>
    <div>
      <button class="btn gray taskbtn" data-task="flamingo">flamingo 粉 170°</button>
      <button class="btn gray taskbtn" data-task="surf">surf 蓝绿 72°</button>
      <input type="range" id="hueslider" min="0" max="180" value="170" style="width:150px">
      <span id="hueval" style="font-size:12px;color:#ff9"></span>
    </div>
    <div class="huebar" id="huebar"><div class="marker" id="huemark"></div></div>
    <h4>suspicious 色相表 — 被拒色相记账（docs/205，跨遍保留）</h4>
    <div id="susptable"></div>
  </div>
</div>
<div class="row" style="margin-top:14px">
  <div class="panel">
    <h3>记忆区 — 什么被短命记住（快原型） / 什么被固化（慢原型）</h3>
    <h4>慢原型（固化记忆 · 亮绿：只以"确认已满"形态存在，不回收）</h4>
    <div id="slow"></div>
    <h4>快原型（gist 短命记忆 · 灰：k_decay=5 窗未重匹配即遗忘；hits≥2 为"候选"）</h4>
    <div id="fast"></div>
    <h4>回收事件（遗忘：它短暂记住过什么，又为什么忘掉）</h4>
    <div class="logbox" id="recycle"></div>
  </div>
  <div class="panel" style="min-width:360px">
    <h3>语言流 — 它怎么说（docs/204 转录：每句 1:1 指回机制状态）</h3>
    <div class="logbox" id="lang"></div>
    <h4>机制事件流（创建 / 升级 / 回收）</h4>
    <div class="logbox" id="events"></div>
  </div>
</div>
<script>
let lastUtt = '';
const $ = id => document.getElementById(id);
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function act(a){ fetch('/act?action=' + encodeURIComponent(a)).then(r => r.json()); }
function fmt1(x){ return (x === null || x === undefined) ? '—' : Number(x).toFixed(3); }

function drawCurve(series){
  const cv = $('maecurve'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!series || series.length < 2) { ctx.fillStyle = '#666'; ctx.fillText('（窗口未足量）', 8, 16); return; }
  const max = Math.max.apply(null, series) * 1.1 || 1;
  const n = series.length;
  ctx.strokeStyle = '#7cf'; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let i = 0; i < n; i++){
    const x = (i / (n - 1)) * (cv.width - 10) + 5;
    const y = cv.height - 8 - (series[i] / max) * (cv.height - 18);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#888'; ctx.font = '10px sans-serif';
  ctx.fillText('MAE 最近 ' + n + ' 窗口', 6, cv.height - 2);
  ctx.fillText('max ' + max.toFixed(3), cv.width - 60, 12);
}

function render(s){
  if (s.frame_img) $('scene').src = 'data:image/jpeg;base64,' + s.frame_img;
  if (s.heat_img)  $('heat').src  = 'data:image/jpeg;base64,' + s.heat_img;
  if (s.thumb_img) $('thumb').src  = 'data:image/jpeg;base64,' + s.thumb_img;
  $('frameinfo').innerHTML =
    '帧 ' + s.frame_idx + ' / ' + s.total_frames +
    '（窗口内第 ' + s.win_inflight + ' 帧）｜窗口 ' + (s.win + 1) +
    '｜本帧残差均值 ' + fmt1(s.resid_mean) + '｜状态 ' + esc(s.status) +
    (s.loading ? '（加载中：' + esc(s.loading) + '…）' : '');
  const st = s.stats = s.stats || {};
  const qs = [
    ['已完成窗口 / 本窗事件量 E / 上区 U', s.win + ' / ' + s.E + 'px / ' + s.U + 'px（' + esc(s.gate) + '）'],
    ['MAE（当前窗口）', fmt1(s.mae_last)],
    ['ratio（末/首四分之一，A1 有界 ≤1.5）', fmt1(s.ratio) + ' <span class=' +
      (s.ratio !== null && s.ratio > 0 && s.ratio <= 1.5 ? '"ok"' : '"no"') + '>' +
      (s.ratio !== null && s.ratio > 0 && s.ratio <= 1.5 ? '✓有界' : '…') +
      '</span>（Q1=' + fmt1(s.q1) + '，Q4=' + fmt1(s.q4) + '，' + s.ratio_n + ' 窗）'],
    ['自适应阈值 θ / 死区 db（只抬不降）', fmt1(s.theta) + ' / ' + fmt1(s.db)],
    ['SC2_slow（确认慢原型=固化结构）', s.sc2_slow + '（SC1_slow=' + s.sc1_slow + '）'],
    ['SC1_fast（累计创建快原型）', s.sc1_fast],
    ['n_promo（升级） / n_recycle（回收）', s.n_promo + ' / ' + s.n_recycle],
    ['结构密度上界（不膨胀，docs/257）', s.sc2_slow + ' ≤ ' + s.density_bound +
      (s.density_ok ? ' <span class="ok">✓</span>' : ' <span class="no">✗</span>')],
    ['当前窗口匹配原型', s.matched[1] === null || s.matched[1] === undefined
      ? '（无：静默/低事件）' : '#' + s.matched[1]],
  ];
  $('stats').innerHTML = '<table>' + qs.map(([k, v]) =>
    '<tr><td style="color:#999">' + k + '</td><td>' + v + '</td></tr>').join('') + '</table>';

  // 记忆区
  const slow = (s.protos || []).filter(p => p.kind === 'slow');
  const fast = (s.protos || []).filter(p => p.kind === 'fast');
  const phtml = p => '<div class="proto ' + p.kind + '">' +
    '#' + p.pid + ' <b>hits=' + p.hits + '</b> 创建窗 ' + p.created +
    ' 最近 ' + p.last_active + ' μ=(' + p.mu[0] + ',' + p.mu[1] + ')' +
    (p.kind === 'slow' ? ' 固化于 ' + p.promoted_at :
     (p.candidate ? ' <span class="cand">候选(≥2)</span>' : '')) + '</div>';
  $('slow').innerHTML = slow.length
    ? slow.slice(0, 20).map(phtml).join('')
    : '<div class="stat" style="color:#666">（尚无固化记忆——反复重匹配 ≥' +
      (s.knobs ? s.knobs.hits_min_slow : 3) + ' 次才会出现）</div>';
  $('fast').innerHTML = fast.length
    ? fast.slice(0, 20).map(phtml).join('')
    : '<div class="stat" style="color:#666">（无活跃快原型）</div>';
  const rec = (s.events || []).filter(e => e.type === 'recycle');
  $('recycle').innerHTML = rec.length
    ? rec.slice(-12).reverse().map(e =>
        '<div class="ev recycle">窗口 ' + e.w + ' 遗忘了一个 gist：曾经是 #' + e.pid +
        '，' + esc(e.text) + '</div>').join('')
    : '<div class="stat" style="color:#666">（尚无遗忘——快原型仍活跃或还没被创建）</div>';

  // 语言流
  $('lang').innerHTML = (s.lang || []).length
    ? s.lang.map(l => '<div class="lang ' + esc(l.tag) + '">[' + l.w + '] ' +
        esc(l.text) + '</div>').join('')
    : '<div class="stat" style="color:#666">（观察中——机制状态稳定时它不说话）</div>';
  const lg = $('lang'); lg.scrollTop = lg.scrollHeight;
  $('events').innerHTML = (s.events || []).slice(-14).reverse().map(e =>
    '<div class="ev ' + e.type + '">[窗 ' + e.w + '] ' + esc(e.text) + '</div>').join('');
  const ev = $('events'); ev.scrollTop = ev.scrollHeight;

  $('streaminfo').textContent = '当前流：' + s.video_label +
    '（切换视频不重置记忆；"重置记忆"按钮清零）';
  $('masknote').textContent = s.has_mask === false
    ? '· 无 GT 掩码：无目标轮廓/颜色确认（诚实标注）' : '';
  $('ly_color').checked = !s.layers || s.layers.color;
  $('ly_event').checked = !s.layers || s.layers.event;
  $('ly_gt').checked    = !s.layers || s.layers.gt;
  renderColor(s);
  drawCurve(s.mae_series);
  $('playbtn').textContent = s.playing ? '暂停' : '播放';
  $('playbtn').classList.toggle('active', !s.playing);
  document.querySelectorAll('.spd').forEach(b =>
    b.classList.toggle('active', Number(b.dataset.s) === s.speed));
}

// 颜色感知面板（docs/205/208/199b 状态转录：每行 1:1 指回 /state 颜色字段）
function hueCss(h){ return 'hsl(' + Math.round(h * 2) + ',80%,55%)'; }
function renderColor(s){
  const g = s.gate || {};
  const wb = s.wb || {};
  const claim = s.claimed ? '<span class="ok">✓ 确认（trusted）</span>'
                          : '<span class="no">✗ 未确认</span>';
  const gateTxt = g.open
    ? '<span class="ok">开 — 它在变色？</span>'
    : '<span class="no">关 — 固执</span>' +
      '（|Δshift| EWMA ' + fmt1(g.d_mu) + '° < ' + fmt1(g.delta_gate) + '°）';
  const rows = [
    ['任务色相（外赋利害 docs/188）', fmt1(s.task_hue) + '°' +
      (s.task_preset ? '（' + esc(s.task_preset) + '）' : '')],
    ['目标区中位色相（掩码定位，非检测器）', s.hue === null || s.hue === undefined
      ? '—（无目标/空掩码）' : fmt1(s.hue) + '°'],
    ['任务色占比 frac（>thr 才确认）', fmt1(s.frac) + '（thr ' + fmt1(s.thr) + '）'],
    ['色距（中位色相 ↔ 任务色，<30°）', fmt1(s.color_dist) + '°'],
    ['自适应带宽 bw（docs/208：k_band×见过的变异）', fmt1(s.bw) + '°'],
    ['重信任需求 k（docs/200 纪念）', s.k + '（被拒累计 ' + s.rej_total + '，连续 ' +
      (s.consec === undefined ? '—' : s.consec) + '）'],
    ['确认状态（claimed）', claim],
    ['时间门（docs/199b：MAD 集中 AND 帧间一致）', gateTxt],
    ['白平衡 / 色温', '校正 ' + (wb.n_corr || 0) + ' 次' +
      (wb.do_corr ? '（本帧在校正）' : '') +
      (wb.g_cur ? '；增益 ' + wb.g_cur.join('/') : '') +
      (wb.shift !== undefined && wb.shift !== null ? '；背景色相 ' + fmt1(wb.shift) + '°' : '')],
  ];
  $('colorstate').innerHTML = '<table>' + rows.map(([k, v]) =>
    '<tr><td style="color:#999">' + k + '</td><td>' + v + '</td></tr>').join('') + '</table>' +
    (s.note ? '<div style="color:#f97;font-size:11px;margin-top:4px">' + esc(s.note) + '</div>' : '');

  // suspicious 表：被拒色相 → 计数（任务色邻域高亮）
  const sus = s.suspicious || {};
  const keys = Object.keys(sus).map(Number).filter(h => isFinite(h));
  const th = Number(s.task_hue) || 0;
  const maxc = keys.length ? Math.max.apply(null, keys.map(h => sus[h])) : 1;
  $('susptable').innerHTML = keys.length
    ? keys.sort((a, b) => sus[b] - sus[a]).slice(0, 14).map(h => {
        const near = Math.abs(((h - th) % 180 + 180) % 180) < 10 ||
                     Math.abs(180 - ((h - th) % 180 + 180) % 180) < 10;
        return '<div class="susrow"><span class="swatch" style="background:' + hueCss(h) +
          '"></span><span class="' + (near ? 'task' : '') + '">' + h + '°</span>' +
          '<span class="bar" style="width:' + Math.max(3, 60 * sus[h] / maxc) +
          'px;background:' + (near ? '#ff9' : '#f97') + '"></span>' +
          '<span style="color:#999">×' + sus[h] + (near ? '（任务色邻域）' : '') +
          '</span></div>';
      }).join('')
    : '<div class="stat" style="color:#666">（无被拒色相——它还没怀疑过任何颜色）</div>';

  // 任务色相滑条 + 色相环标记（0-180 与 OpenCV H 一致）
  $('hueslider').value = th;
  $('hueval').textContent = th + '°';
  $('huemark').style.left = (th / 180 * 100) + '%';
}

async function tick(){
  try {
    const r = await fetch('/state');
    const s = await r.json();
    render(s);
  } catch(e) {}
  setTimeout(tick, 250);
}

// 控件
$('playbtn').onclick = () => act($('playbtn').textContent === '暂停' ? 'pause' : 'play');
$('stepbtn').onclick = () => act('step');
$('resetbtn').onclick = () => { if (confirm('清空记忆并重新开始？')) act('reset'); };
document.querySelectorAll('.spd').forEach(b => b.onclick = () => act('speed' + b.dataset.s));

// 图层切换（原彩帧 / 事件掩码 / GT 目标轮廓）
function sendLayers(){
  const parts = [];
  if ($('ly_color').checked) parts.push('color');
  if ($('ly_event').checked) parts.push('event');
  if ($('ly_gt').checked) parts.push('gt');
  act('layers=' + parts.join(','));
}
$('ly_color').onchange = sendLayers;
$('ly_event').onchange = sendLayers;
$('ly_gt').onchange = sendLayers;

// 任务色相（外赋利害 docs/188；跨视频保持）
document.querySelectorAll('.taskbtn').forEach(b => b.onclick = () => act('task=' + b.dataset.task));
$('hueslider').onchange = () => act('taskhue=' + $('hueslider').value);
$('hueslider').oninput = () => { $('hueval').textContent = $('hueslider').value + '°'; };

// 视频下拉（机制不重置：新场景被吸收进既有记忆）
async function init(){
  const s = await (await fetch('/state')).json();
  const sel = $('videosel');
  sel.innerHTML = Object.keys(s.streams).map(k =>
    '<option value="' + k + '"' + (k === s.video ? ' selected' : '') + '>' +
    esc(s.streams[k]) + '</option>').join('');
  sel.onchange = () => act('video=' + sel.value);
}
init();
tick();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None          # 由 main 注入
    _tick = 0

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            Handler._tick += 1
            self._json(self.engine.snapshot(Handler._tick))
        elif self.path == "/health":
            self._json({"ok": True})
        elif self.path.startswith("/act?"):
            import urllib.parse
            q = urllib.parse.parse_qs(self.path.split("?", 1)[1])
            action = q.get("action", ["noop"])[0]
            self._json(self.engine.act(action))
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser(description="巅峰组装演示：现行主机制链 + 真实视频流")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--fps", type=float, default=4.0, help="1× 速度的帧率（默认 4）")
    ap.add_argument("--selftest", action="store_true", help="机制冒烟，不启 HTTP")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    engine = Demo2Engine(video=args.video, fps=args.fps)
    # 预加载默认流（同步，让首帧立即可见）
    engine._get_stream(engine.cur)
    Handler.engine = engine

    t = threading.Thread(target=engine.run, daemon=True)
    t.start()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("D2_READY port=%d video=%s streams=%d fps=%.1f" %
          (args.port, engine.cur, len(ALL_STREAMS), args.fps))
    print("D2_URL http://127.0.0.1:%d" % args.port)
    print("D2_INTERACT video-switch=no-reset reset=clear-memory step=one-window")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine._stop = True
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
