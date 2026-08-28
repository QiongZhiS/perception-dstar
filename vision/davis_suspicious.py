"""vision/davis_suspicious.py — 类别级怀疑系统回真实图像：DAVIS 自然目标 + 真值掩码。

docs/218 缺口 1（致命）：docs/205-216 整条"阅历"进化链全是色相轴一维 toy（合成球/
真值掩码/受控漂移）。本脚本把整条机制链搬到 DAVIS 真实视频的自然目标上：
  flamingo（粉，任务色相 ~170°）为主目标、surf（蓝绿 ~72°，含 1 帧空掩码=目标消失）为
  对照目标。真值掩码给出目标区域（定位免费——受控实拍立场 docs/185/186，机制只做
  确认/怀疑，不假装有检测器）；目标观测是真实的：非纯色（帧内色相 std 5-44°）、
  自然帧间漂移（surf 4.1°）、真实噪声/运动/相机抖动。

事件序列（真实帧 + 受控扰动，扰动是帧级乘性增益/噪声——物理真实，docs/199 同款）：
  base     真实帧                      → 确认通过（维持）
  dusk     色温增益渐变（连续漂移=世界变化）→ 时间门+白平衡→维持；无校正→假被拒
  corrupt  目标区颜色变化（他者的不）   → 确认退化→被拒→suspicious 沉积
  rest     真实帧+假恢复帧             → 带纪念重信任（k 帧延迟，docs/200）；rest 段
            前 8 帧混 4 帧仍 corrupt（"看似恢复实则仍错"）→ 验证"不轻信"
  GT       逐帧：正=目标在场且颜色真实正确（base/dusk/rest 的 clean 帧）；
            负=corrupt 段 + rest 假恢复帧 + 空掩码帧（目标消失）

机制（docs/199 + 200-216 完整链，全部参数来自已验证实验）：
  时间门+白平衡   docs/199b（连续漂移=世界变化→校正；突变=噪声→固执）
                  ★回真实视频修正：校正条件 = 门开 AND 增益相对基线显著偏移
                  （真实偏色背景的 bg_gain 恒非 1，全图乘会掩盖"他者的不"）
  确认            docs/195 三通道：任务色占比 frac>thr 且 中位色相距任务色<30°
  suspicious 表   docs/205（被拒类别记账，跨遍保留）
  连续距离加权    docs/212（带宽内怀疑按距离衰减）
  自适应带宽      docs/208（bw = k_band × 见过的同类色相 std）
  类别级 thr      docs/207/206（被拒上调、信任回落）
  全局纪念 k      docs/200（重信任延迟 ∝ 总被拒，饱和 docs/203）
  分层衰减        docs/216（k 窄高斯 σ=bw/4 疑的窄、thr 线性严的宽）

变体对照（docs/218 缺口 3）：
  A full      时间门+白平衡 + 类别级 suspicious + 自适应带宽 + 分层衰减
  B nowb      无白平衡（dusk 假被拒）——docs/199 必要性
  C/C2 global 全局 thr（有/无白平衡）——docs/207 病态（目标回来认不出=锁死）
  D template  朴素模板匹配（固定色相窗+固定阈值，无记忆无门）——基线
  E/F fixedbw 固定带宽 10°/30°——docs/208 对照（固定 30° 误伤邻近类别）

度量（docs/218 缺口 2，标准检测度量）：
  逐帧 claimed vs GT → TP/TN/FP/FN → precision / recall / F1
  另按 docs/207 语义报告误报/漏报；翻转=rest 段 claimed 翻转次数（轻信度量：
  无记忆系统在假恢复帧间跳变=翻转多，纪念系统需 k 连续证据=翻转少）。

用法：
  python vision/davis_suspicious.py                          # flamingo（gain corrupt）
  python vision/davis_suspicious.py --video surf --seg 14,12,10,19 --thr 0.40 --corrupt noise
  python vision/davis_suspicious.py --save-state             # 阅历落盘（docs/234 B1.3）
  python vision/davis_suspicious.py --load-state             # 重启续跑（等价未重启）
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402

DAVIS = os.path.join("vision", "out", "davis")
N_SEG = {"base": 20, "dusk": 20, "corrupt": 15, "rest": 25}   # flamingo 80 帧
SAT_MIN_TARGET, SAT_MIN_BG = 60, 60
HUE_BIN = 10.0        # 色相分箱（docs/221 A1：bin=10°，bin 内 std 为真带宽）


def circ(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def load_video(video):
    vdir = os.path.join(DAVIS, video)
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    pngs = sorted(f for f in os.listdir(vdir) if f.endswith(".png"))
    frames, masks = [], []
    for j, p in zip(jpgs, pngs):
        frames.append(cv2.imread(os.path.join(vdir, j)))
        masks.append(cv2.imread(os.path.join(vdir, p), cv2.IMREAD_GRAYSCALE))
    return frames, masks


def warm_gain(k, sat=1.0):
    """暖色温乘性增益（davis_constancy 同款）：B 降 R 升。k∈[0,1]。"""
    return np.array([1 - sat * k, 1.0, 1 + 0.8 * sat * k])


def build_sequence(frames, masks, seg_len, seed=7, corrupt="gain",
                   noise_std=10.0, corrupt_kc=0.05):
    """按段构造序列：真实帧 + 受控扰动。返回 (帧, 掩码, 段标签, 逐帧GT)。

    段语义（docs/178 两种否定的真实版）：
      base    真实帧                           → 确认通过（维持）
      dusk    色温增益渐变（连续漂移=世界变化） → 时间门+白平衡→维持；无校正→假被拒
      corrupt 目标区颜色变化（他者的不）        → 确认退化→被拒→suspicious 沉积
      rest    真实帧 + 假恢复帧                → 带纪念重信任（docs/200）；rest 段
               前 8 帧混入 4 帧 corrupt（目标"看似恢复实则仍错"）→ 验证"不轻信"
    全图轻噪声（std=noise_std）加在 base/dusk/rest：世界噪声（docs/101 固执）。
    corrupt 方式：
      gain  温和冷增益（强度 corrupt_kc；目标色相漂移，记账 hue 接近目标）
      noise 目标区强噪声（目标观测全毁，记账 hue 乱）
    扰动强度参数化（docs/221 A2 多种子统计）：noise_std/corrupt_kc 由统计外壳按
    seed 采样——不同种子=不同扰动强度，结果才有真实方差（默认值保持 docs/219 数字）。
    GT：正=目标在场且颜色真实正确（base/dusk/rest 的 clean 帧）；负=corrupt 段 +
        rest 段的假恢复帧（他者的不）。dusk 是光照（世界变化）不是颜色变化 → 正。
    """
    rng = np.random.default_rng(seed)
    idx = {}
    pos = 0
    for seg in ["base", "dusk", "corrupt", "rest"]:
        idx[seg] = list(range(pos, pos + seg_len[seg]))
        pos += seg_len[seg]
    total = pos
    n_src = len(frames)
    out_f, out_m, out_s, out_gt = [], [], [], []
    rest_i = idx["rest"]
    for i in range(total):
        seg = next(s for s in ["base", "dusk", "corrupt", "rest"] if i in idx[s])
        src = frames[i % n_src]
        mask = masks[i % n_src]
        gt = 1
        if seg == "dusk":
            k = (idx["dusk"].index(i)) / max(len(idx["dusk"]) - 1, 1)
            src = np.clip(src.astype(np.float32) * warm_gain(k), 0, 255).astype(np.uint8)
        elif seg == "corrupt":
            gt = 0
            n = rng.normal(0, noise_std, src.shape).astype(np.int16)
            src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
            if mask.max() > 0:
                if corrupt == "gain":
                    g = np.array([1 + 0.9 * corrupt_kc, 1.0, 1 - 0.9 * corrupt_kc])
                    src = src.astype(np.float32)
                    src[mask > 0] *= g
                    src = np.clip(src, 0, 255).astype(np.uint8)
                else:
                    n2 = rng.normal(0, 90, src.shape).astype(np.int16)
                    src = np.clip(src.astype(np.int16) + n2 * (mask > 0)[:, :, None],
                                  0, 255).astype(np.uint8)
        elif seg == "rest":
            # 假恢复：rest 段前 8 帧，奇数帧仍是 corrupt（"看似恢复实则仍错"）
            k_rest = rest_i.index(i)
            if k_rest < 8 and k_rest % 2 == 1:
                gt = 0
                n = rng.normal(0, noise_std, src.shape).astype(np.int16)
                src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
                if mask.max() > 0:
                    if corrupt == "gain":
                        g = np.array([1 + 0.9 * corrupt_kc, 1.0, 1 - 0.9 * corrupt_kc])
                        src = src.astype(np.float32)
                        src[mask > 0] *= g
                        src = np.clip(src, 0, 255).astype(np.uint8)
                    else:
                        n2 = rng.normal(0, 90, src.shape).astype(np.int16)
                        src = np.clip(src.astype(np.int16) + n2 * (mask > 0)[:, :, None],
                                      0, 255).astype(np.uint8)
            else:
                n = rng.normal(0, noise_std, src.shape).astype(np.int16)
                src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
        else:
            n = rng.normal(0, noise_std, src.shape).astype(np.int16)
            src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
        out_f.append(src)
        out_m.append(mask)
        out_s.append(seg)
        out_gt.append(gt)
    return out_f, out_m, out_s, out_gt


# ---- 时间门 + 白平衡（docs/199b，DAVIS 版：背景估计排除目标掩码）----


def bg_stats(fr, exclude):
    """背景彩色像素色相中位数 + 占比 + 环形 MAD（排除目标区域）。"""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    colored = (s > SAT_MIN_BG).astype(np.uint8)
    colored[exclude > 0] = 0
    frac = float(colored.mean())
    if colored.sum() < 100:
        return None, frac, float("nan")
    vals = h[colored > 0].astype(float)
    med = float(np.median(vals))
    d = np.abs(vals - med)
    mad = float(np.median(np.minimum(d, 180.0 - d)))
    return med, frac, mad


def bg_gain(fr, exclude):
    """通道级白平衡增益（背景 BGR 均值）。"""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    colored = (s > SAT_MIN_BG).astype(np.uint8)
    colored[exclude > 0] = 0
    if colored.sum() < 100:
        return None
    m = fr[colored > 0].mean(axis=0)
    if m[1] < 5 or m[0] < 2 or m[2] < 2:
        return None
    return np.array([m[1] / m[0], 1.0, m[1] / m[2]])


class TemporalGateDavis:
    """docs/199b 组合判据：真染色（色相 MAD 集中）AND 帧间一致（|Δshift| EWMA）。"""

    def __init__(self, mad_gate=40.0, alpha=0.05, warmup=8, delta_gate=4.0):
        self.mad_gate, self.alpha, self.warmup, self.delta_gate = \
            mad_gate, alpha, warmup, delta_gate
        self.d_mu, self.prev, self.n = None, None, 0

    def update(self, s, mad):
        if s is None or mad is None or mad > self.mad_gate:
            return False
        if self.prev is not None:
            d = abs(s - self.prev)
            self.d_mu = d if self.d_mu is None else self.d_mu + self.alpha * (d - self.d_mu)
        self.prev = s
        self.n += 1
        return self.open()

    def open(self):
        return self.n >= self.warmup and self.d_mu is not None and self.d_mu < self.delta_gate


# ---- 机制：类别级怀疑（docs/205-216 完整链）----


class DavisMind:
    """接收→时间门→确认→被拒入史→类别怀疑→带纪念重信任。

    变体开关：
      wb=True      时间门+白平衡（docs/199b）
      global_thr   thr 全局单值（docs/207 病态对照）
      fixed_bw     固定带宽（docs/208 对照）；None=自适应
      template     朴素模板匹配（无记忆无门）
    """

    def __init__(self, task_hue, wb=True, global_thr=False, fixed_bw=None,
                 template=False, k_band=2.0, thr_base=0.60, thr_boost=0.04,
                 thr_cap=0.85, relax=0.02, persist_for=3, grow=3.0, cap_k=60,
                 layered_k=False):
        self.task_hue = task_hue
        self.wb, self.global_thr, self.template = wb, global_thr, template
        self.fixed_bw = fixed_bw
        self.layered_k = layered_k         # docs/221 A1：k 侧分层衰减接线（默认关，保持 docs/219 数字）
        self.k_band = k_band
        self.thr_base, self.thr_boost, self.thr_cap = thr_base, thr_boost, thr_cap
        self.relax, self.persist_for, self.grow, self.cap_k = relax, persist_for, grow, cap_k
        self.thr_cur = thr_base               # global/template 用
        self.rejected = {}                    # suspicious[hue] = 被拒累计（docs/205）
        self.rej_total = 0                    # 总被拒帧（docs/200 纪念）
        self.thr_by = {}                      # 类别级阈值上调量（docs/206/207）
        self.hist = {}                        # hue -> 见过的观测色相（自适应带宽 docs/208）
        self.trusted, self.consec, self.persist = True, 0, 0
        self.gate = TemporalGateDavis() if wb else None
        self.g_base = None                    # 基线背景增益（相对变化判定校正）
        self.n_corr = 0

    @staticmethod
    def _bin(hue):
        """色相分箱（docs/221 A1）：真实观测 hue 连续浮点，按 bin 聚合才有统计意义。
        bin=10° → 键为 bin 起始值（0,10,...,170）。"""
        return int(hue // HUE_BIN) * HUE_BIN

    def bw(self, hue):
        """自适应带宽：见过变异 = 该色相 bin 内观测的 std（docs/208 + docs/221 A1）。
        修复前按精确浮点 hue 记账 → 每键 1-2 个观测 → std 无意义/bw 恒 0（退化）；
        分箱后 bin 内观测聚合 → std 为真带宽。"""
        if self.fixed_bw is not None:
            return self.fixed_bw
        if self.template:
            return 30.0
        obs = self.hist.get(self._bin(hue), [])
        if len(obs) < 3:
            return 0.0
        return self.k_band * float(np.std(obs))

    def _w_narrow(self, d, bw):
        """疑的窄（docs/216）：k 用窄高斯 σ=bw/4。"""
        if bw <= 0:
            return 1.0 if d == 0 else 0.0
        if d >= bw:
            return 0.0
        return float(np.exp(-(d ** 2) / (2 * (bw / 4) ** 2)))

    def _w_wide(self, d, bw):
        """严的宽（docs/216）：thr 用线性慢衰减。"""
        if bw <= 0:
            return 1.0 if d == 0 else 0.0
        if d >= bw:
            return 0.0
        return max(0.0, 1.0 - d / bw)

    def k(self, hue):
        """重信任所需连续帧：docs/200 纪念（全局被拒时长，饱和 docs/203）。
        suspicious 表只用于类别级 thr（docs/205/206）；纪念是全局的（demo 同款）。
        layered_k=True 时接线 docs/216 分层衰减（k 窄高斯：被拒类别按距离加权），
        保持 docs/219 实验数字的默认行为为全局纪念（docs/221 A1）。"""
        if self.template:
            return 1
        if self.layered_k and hue is not None and self.rejected:
            n = 0.0
            for sh, cnt in self.rejected.items():
                n += cnt * self._w_narrow(circ(hue, sh), self.bw(sh))
            return 1 + int(min(n, self.cap_k) / self.grow)
        return 1 + int(min(self.rej_total, self.cap_k) / self.grow)

    def thr(self, hue):
        if hue is None or self.template or self.global_thr:
            return self.thr_cur if (self.template or self.global_thr) else self.thr_base
        t = self.thr_base
        for sh, dv in self.thr_by.items():
            t += dv * self._w_wide(circ(hue, sh), self.bw(sh))
        return min(self.thr_cap, t)

    def to_state(self):
        """docs/221 B1.3 记忆持久化：阅历状态序列化（docs/234）。

        自包含快照 = 配置（机制参数）+ 阅历（suspicious/thr_by/hist/纪念）+ 运行态
        （信任状态/门/白平衡基线/校正计数）。行为等价"没重启过"需要全量——缺门或
        g_base 时，重启后时间门判据/校正判定会从头重来，rest 段行为可能偏离。
        """
        gate = None
        if self.gate is not None:
            gate = {"mad_gate": self.gate.mad_gate, "alpha": self.gate.alpha,
                    "warmup": self.gate.warmup, "delta_gate": self.gate.delta_gate,
                    "d_mu": self.gate.d_mu, "prev": self.gate.prev,
                    "n": self.gate.n}
        return {"task_hue": self.task_hue, "thr_base": self.thr_base,
                "thr_boost": self.thr_boost, "thr_cap": self.thr_cap,
                "relax": self.relax, "persist_for": self.persist_for,
                "grow": self.grow, "cap_k": self.cap_k,
                "k_band": self.k_band,
                "wb": self.wb, "global_thr": self.global_thr,
                "template": self.template, "fixed_bw": self.fixed_bw,
                "layered_k": self.layered_k,
                "thr_cur": self.thr_cur,
                "rejected": {str(k): v for k, v in self.rejected.items()},
                "rej_total": self.rej_total,
                "thr_by": {str(k): v for k, v in self.thr_by.items()},
                "hist": {str(k): list(v) for k, v in self.hist.items()},
                "trusted": self.trusted, "consec": self.consec,
                "persist": self.persist, "g_base": None if self.g_base is None
                else list(self.g_base),
                "n_corr": self.n_corr, "gate": gate}

    @classmethod
    def from_state(cls, st, **overrides):
        """docs/203 阅历跨会话：从落盘状态恢复（docs/234）。

        状态为自包含快照：配置（wb/template/global_thr/fixed_bw/layered_k/k_band +
        阈值参数）+ 阅历（suspicious 表/thr_by/hist/纪念）+ 运行态（信任状态/门/
        白平衡基线/校正计数）原样恢复；overrides 覆盖构造参数（如跨配置迁移阅历）。
        """
        kw = {"task_hue": st["task_hue"], "thr_base": st["thr_base"],
              "thr_boost": st["thr_boost"], "thr_cap": st["thr_cap"],
              "relax": st["relax"], "persist_for": st["persist_for"],
              "grow": st["grow"], "cap_k": st["cap_k"],
              "k_band": st.get("k_band", 2.0),
              "wb": st.get("wb", False), "global_thr": st.get("global_thr", False),
              "template": st.get("template", False),
              "fixed_bw": st.get("fixed_bw"),
              "layered_k": st.get("layered_k", False)}
        kw.update(overrides)
        m = cls(**kw)
        m.thr_cur = st.get("thr_cur", m.thr_base)
        m.rejected = {float(k): v for k, v in st.get("rejected", {}).items()}
        m.rej_total = st.get("rej_total", 0)
        m.thr_by = {float(k): v for k, v in st.get("thr_by", {}).items()}
        m.hist = {float(k): list(v) for k, v in st.get("hist", {}).items()}
        m.trusted = st.get("trusted", True)
        m.consec = st.get("consec", 0)
        m.persist = st.get("persist", 0)
        m.n_corr = st.get("n_corr", 0)
        if st.get("g_base") is not None:
            m.g_base = np.array(st["g_base"])
        g = st.get("gate")
        if g is not None and m.gate is not None:
            m.gate.mad_gate = g.get("mad_gate", m.gate.mad_gate)
            m.gate.alpha = g.get("alpha", m.gate.alpha)
            m.gate.warmup = g.get("warmup", m.gate.warmup)
            m.gate.delta_gate = g.get("delta_gate", m.gate.delta_gate)
            m.gate.d_mu = g.get("d_mu")
            m.gate.prev = g.get("prev")
            m.gate.n = g.get("n", 0)
        return m

    def step(self, frac, hue, mask_empty):
        if hue is not None:
            self.hist.setdefault(self._bin(hue), []).append(hue)   # 分箱记账（A1）
        t = self.thr(hue)
        d = circ(hue, self.task_hue) if hue is not None else 999.0
        ok = (not mask_empty) and frac > t and d < 30.0
        if ok:
            self.persist += 1
            if self.persist >= self.persist_for:
                if self.template or self.global_thr:
                    self.thr_cur = max(self.thr_base, self.thr_cur - self.relax)
                elif hue is not None:
                    self.thr_by[hue] = max(0.0, self.thr_by.get(hue, 0.0) - self.relax)
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k(hue):
                    self.trusted = True
        else:
            self.persist = 0
            self.trusted = False
            self.consec = 0
            self.rej_total += 1
            if not self.template:
                if self.global_thr:
                    self.thr_cur = min(self.thr_cap, self.thr_cur + self.thr_boost)
                elif hue is not None:
                    self.thr_by[hue] = min(self.thr_cap,
                                           self.thr_by.get(hue, 0.0) + self.thr_boost)
                    self.rejected[hue] = self.rejected.get(hue, 0) + 1
        return ok


# ---- 记忆持久化（docs/221 B1.3 + docs/234：阅历跨会话）----


STATE_DIR = os.path.join("vision", "out", "state")   # 落盘位置（任务指定）
STATE_VERSION = 1                                     # 状态文件格式版本
_STATE_DEFAULT = object()   # argparse 哨兵：--save-state/--load-state 裸用→默认路径


def default_state_path(video, seed=7, corrupt="gain"):
    """默认落盘位置：vision/out/state/<video>_s<seed>_<corrupt>.json。"""
    return os.path.join(STATE_DIR, "%s_s%d_%s.json" % (video, int(seed), corrupt))


def save_state(path, minds, meta=None):
    """DavisMind 阅历状态落盘（docs/234）：{key: DavisMind} → JSON。

    payload = {artifact, doc_ref, state_version, saved_at, meta,
               variants: {key: to_state() 快照}}。返回实际写入路径。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"artifact": "davis_suspicious_state", "doc_ref": "docs/234",
               "state_version": STATE_VERSION,
               "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "meta": meta or {},
               "variants": {k: m.to_state() for k, m in minds.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def load_state(path, cfgs=None):
    """从落盘 JSON 恢复阅历（docs/234）。

    cfgs: {key: 构造参数}（变体配置，如 {"A": {"wb": True}}）——状态快照自包含
    配置，但显式传入当前运行配置可保证与本次运行一致（也支持跨配置迁移）。
    返回 {key: DavisMind}；cfgs 为 None 时返回原始状态 dict（调用方自行构造）。
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    states = {k: st for k, st in payload["variants"].items()}
    if cfgs is None:
        return states
    out = {}
    for key, cfg in cfgs.items():
        st = states.get(key)
        if st is not None:
            out[key] = DavisMind.from_state(st, **cfg)
    return out


# ---- 管线 ----


def target_obs(fr, mask, task_hue):
    """目标区域（真值掩码）观测：任务色占比 + 中位色相。定位免费（受控实拍）。

    frac = 彩色像素中距任务色相 < 30° 的比例（"它是任务色的比例"，与 docs/200 红区
    占比同构；toy 里红球纯色→~0.8，真实 flamingo 粉色羽毛→也高）。
    """
    if mask is None or mask.max() == 0:
        return 0.0, None
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    pix_h, pix_s = h[mask > 0], s[mask > 0]
    colored = pix_s > SAT_MIN_TARGET
    if colored.sum() < 10:
        return 0.0, None
    hs = pix_h[colored].astype(float)
    d = np.abs(hs - task_hue) % 180.0
    d = np.minimum(d, 180.0 - d)
    frac = float(np.mean(d < 30.0))
    hue = float(np.median(hs))
    return frac, hue


def run_mind(mind, frames, masks, segs, task_hue, video="", debug=False):
    """跑一遍完整序列。返回 (宣称序列, 每段通过率, 校正帧数, 逐帧轨迹)。"""
    oks, seg_rate, corr = [], {}, 0
    trace = []
    for i, (fr, mask, seg) in enumerate(zip(frames, masks, segs)):
        img = fr
        if mind.wb:
            exclude = cv2.dilate(mask, np.ones((21, 21), np.uint8)) if mask.max() > 0 \
                else np.zeros_like(mask)
            s, _, mad = bg_stats(fr, exclude)
            g = bg_gain(fr, exclude)
            mind.gate.update(s, mad)
            # 相对基线增益变化（docs/199"何时校正"的真实语义）：toy 背景灰→基线增益
            # ≈1；DAVIS 背景自然偏色→基线增益非 1。校正条件 = 门开 AND 增益相对基线
            # 显著偏移（背景色温真的变了）——否则全图乘会把目标区观测拉偏、掩盖
            # "他者的不"（corrupt 段背景没变→g≈g_base→不校正；dusk 段背景被色温
            # 推→g 偏移→校正）
            do_corr = False
            if g is not None:
                if mind.g_base is None:
                    mind.g_base = g.copy()
                do_corr = mind.gate.open() and \
                    float(np.max(np.abs(g - mind.g_base))) > 0.12
            if debug:
                print(f"    bg: shift={s} mad={mad:.1f} gate_open={mind.gate.open()} "
                      f"gain={None if g is None else np.round(g,2).tolist()} "
                      f"g_base={None if mind.g_base is None else np.round(mind.g_base,2).tolist()} "
                      f"corr={int(do_corr)}")
            if do_corr:
                img = np.clip(fr.astype(np.float32) * g, 0, 255).astype(np.uint8)
                mind.n_corr += 1
        frac, hue = target_obs(img, mask, task_hue)
        ok = mind.step(frac, hue, mask.max() == 0)
        claimed = bool(mind.trusted and ok)     # 机制宣称=信任状态×本帧通过（重信任延迟）
        oks.append(claimed)
        seg_rate.setdefault(seg, []).append(claimed)
        gate_s = None
        if mind.gate is not None:
            gate_s = (mind.gate.open(), getattr(mind.gate, "d_mu", None),
                      getattr(mind.gate, "prev", None))
        trace.append((i, seg, frac, hue, ok, claimed, mind.thr(hue) if hue is not None else None,
                      mind.k(hue) if hue is not None else 1, gate_s))
    if debug:
        for i, seg, frac, hue, ok, cl, t, k, gs in trace:
            h = f"{hue:5.1f}" if hue is not None else "  nan"
            g = f"gate={int(gs[0])}" if gs else ""
            print(f"  {i:3d} {seg:5s} frac={frac:.3f} hue={h} thr={t if t is None else round(t,2)} "
                  f"k={k} ok={int(ok)} claim={int(cl)} {g}")
    return oks, {s: float(np.mean(v)) for s, v in seg_rate.items()}, mind.n_corr, trace


def metrics(oks, gts):
    """标准检测度量：GT 正=目标在场且颜色真实正确（逐帧）。返回 TP/TN/FP/FN + P/R/F1。"""
    tp = tn = fp = fn = 0
    for ok, gt in zip(oks, gts):
        if gt == 1 and ok:
            tp += 1
        elif gt == 0 and not ok:
            tn += 1
        elif gt == 0 and ok:
            fp += 1
        else:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, p=p, r=r, f1=f1)


def flips(oks, segs, seg="rest"):
    """rest 段 claimed 翻转次数：轻信度量（一次通过就信任→在好/坏间跳变=翻转多；
    纪念系统需 k 连续帧→不轻信=翻转少）。"""
    sub = [oks[i] for i, s in enumerate(segs) if s == seg]
    return sum(1 for a, b in zip(sub, sub[1:]) if a != b)


def serialize_trace(trace):
    """逐帧 trace → JSON 安全 dict（docs/221 A5：逐帧 trace 成为 JSON 工件）。
    trace 条目 = (i, seg, frac, hue, ok, claimed, thr, k, gate_s)；
    gate_s = (open, d_mu, prev) 或 None。"""
    out = []
    for i, seg, frac, hue, ok, claimed, t, k, gs in trace:
        e = {"f": int(i), "seg": str(seg), "frac": round(float(frac), 4),
             "hue": None if hue is None else round(float(hue), 3),
             "ok": int(ok), "claimed": int(claimed),
             "thr": None if t is None else round(float(t), 4),
             "k": int(k)}
        if gs is not None:
            e["gate"] = [int(gs[0]),
                         None if gs[1] is None else round(float(gs[1]), 4),
                         None if gs[2] is None else round(float(gs[2]), 3)]
        out.append(e)
    return out


def verify_a1():
    """docs/221 A1 验证：色相分箱后 bw 为真带宽（原按浮点键记账 → 恒 0 退化）。
    漂移中心取 bin 中心（175°，bin[170,180) 内）→ bin 内 std 完整；
    若漂移跨 bin 边界（如 165-175° 跨 160/170），bin 内 std 被截断——分箱的固有代价。"""
    rng = np.random.default_rng(0)
    m = DavisMind(task_hue=170.0, wb=False)
    for _ in range(20):
        m.step(0.15, 175.0 + rng.normal(0, 2.61), False)   # 被拒观测漂移 bin 170 内
    bw = m.bw(170.0)
    print(f"  A1 验证: 漂移 172-178°（bin[170,180) 内，std≈2.61）→ bw={bw:.2f}°"
          f"（k_band×std=2×2.61≈5.22；修复前按浮点键记账 bw=0）"
          f"  [{'PASS' if bw > 3.0 else 'FAIL'}]")
    print(f"  接线: _w_narrow 用于 k（layered_k=True 时分层衰减名副其实，docs/216）"
          f"——k(170°)={m.k(170.0)}（全局纪念，默认保持 docs/219 数字）")


def persist_check():
    """docs/221 B1.3 + docs/203：记忆持久化——阅历跨会话（重启后保留被拒类别）。
    内存 round-trip（原验证）+ 落盘 JSON round-trip（docs/234 工程化）。"""
    frames, masks = load_video("flamingo")
    obs = [target_obs(f, m, 0.0)[1] for f, m in zip(frames, masks)]
    th = float(np.median([h for h in obs if h is not None]))
    seg = {"base": 20, "dusk": 20, "corrupt": 15, "rest": 25}
    seq_f, seq_m, seq_s, seq_gt = build_sequence(frames, masks, seg)
    mind = DavisMind(th)
    for i in range(seg["base"] + seg["dusk"] + seg["corrupt"]):   # 跑到 corrupt 末
        frac, hue = target_obs(seq_f[i], seq_m[i], th)
        mind.step(frac, hue, seq_m[i].max() == 0)
    st = mind.to_state()
    # 模拟会话结束 → 新进程加载（内存）
    m2 = DavisMind.from_state(st)
    p = (m2.rej_total == mind.rej_total and
         m2.rejected == mind.rejected and m2.thr_by == mind.thr_by and
         len(m2.rejected) > 0)
    print(f"  B1.3 记忆持久化: [{'PASS' if p else 'FAIL'}] 会话1 被拒 "
          f"{mind.rej_total} 次、suspicious={ {round(k,0): v for k, v in mind.rejected.items()} }"
          f"、thr_by={ {round(k,0): round(v,2) for k, v in mind.thr_by.items()} } → "
          f"会话2 加载后原样保留（docs/203 阅历跨会话）")
    # 落盘 round-trip（docs/234）：save_state → load_state（JSON 序列化无损）
    path = os.path.join(STATE_DIR, "persist_check.json")
    save_state(path, {"A": mind}, meta={"check": "persist_check"})
    m3 = load_state(path, {"A": {"wb": True}})["A"]
    p2 = (m3.rej_total == mind.rej_total and m3.rejected == mind.rejected and
          m3.thr_by == mind.thr_by and m3.hist == mind.hist and
          m3.gate.n == mind.gate.n)
    print(f"  B1.3 落盘 round-trip: [{'PASS' if p2 else 'FAIL'}] {path}"
          f"（JSON 序列化后 suspicious/thr_by/hist/门 无损）")


def main():
    global DAVIS
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="flamingo")
    ap.add_argument("--davis", default=None,
                    help="DAVIS 数据目录（含 <video>/ 子目录）；默认 vision/out/davis，"
                         "外部数据用此参数指向（如 synthetic-life 的 vision/out/davis）")
    ap.add_argument("--seg", default="20,20,15,25",
                    help="base,dusk,corrupt,rest 帧数")
    ap.add_argument("--thr", type=float, default=0.60,
                    help="确认阈值基线（真实目标确认余量不同：flamingo 0.60/surf 0.40）")
    ap.add_argument("--seed", type=int, default=7,
                    help="序列构造随机种子（docs/221 A2：多种子统计外壳传不同 seed；"
                         "默认 7 保持 docs/219 数字可复现）")
    ap.add_argument("--corrupt", default="gain", choices=["gain", "noise"],
                    help="他者的不的注入方式：gain=温和色相偏移 / noise=目标区强噪声")
    ap.add_argument("--verify-a1", action="store_true", help="只跑 A1 带宽修复验证")
    ap.add_argument("--persist-check", action="store_true",
                    help="只跑 B1.3 记忆持久化验证（docs/203 阅历跨会话）")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="结果 JSON 归档路径（docs/221 A5：数字成为工件）")
    ap.add_argument("--save-state", nargs="?", const=_STATE_DEFAULT, default=None,
                    metavar="PATH",
                    help="跑完后把各变体 DavisMind 阅历状态落盘（docs/234）；"
                         "省略 PATH 用默认 vision/out/state/<video>_s<seed>_<corrupt>.json")
    ap.add_argument("--load-state", nargs="?", const=_STATE_DEFAULT, default=None,
                    metavar="PATH",
                    help="运行前从落盘 JSON 加载阅历（重启续跑，等价未重启，docs/234）；"
                         "省略 PATH 用默认位置；状态里没有的变体回退全新")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.davis:
        DAVIS = args.davis
    if args.verify_a1:
        print("== A1 自适应带宽退化修复验证（docs/221）==")
        verify_a1()
        return
    if args.persist_check:
        print("== B1.3 记忆持久化验证（docs/221 + docs/203）==")
        persist_check()
        return
    seg_len = dict(zip(["base", "dusk", "corrupt", "rest"],
                       [int(x) for x in args.seg.split(",")]))

    frames, masks = load_video(args.video)
    # 任务色相 = 目标自然色相中位数（探测：flamingo ~170、surf ~72）
    obs = []
    for f, m in zip(frames, masks):
        h = target_obs(f, m, 0.0)[1]
        if h is not None:
            obs.append(h)
    task_hue = float(np.median(obs))
    print(f"== 类别级怀疑系统回真实图像：{args.video}（任务色相 {task_hue:.1f}°）==")

    seq_f, seq_m, seq_s, seq_gt = build_sequence(frames, masks, seg_len,
                                                 seed=args.seed, corrupt=args.corrupt)
    n_empty = sum(1 for m in seq_m if m.max() == 0)
    n_pos = sum(seq_gt)
    print(f"序列 {len(seq_f)} 帧（base/dusk/corrupt/rest 段；rest 前 8 帧混 4 假恢复），"
          f"空掩码帧（目标消失）{n_empty}，GT 正样本 {n_pos}/{len(seq_gt)}\n")

    # 变体规格：稳定 key（状态落盘键）+ 显示名（表格/JSON 保持 docs/219 原样）
    variant_specs = [
        ("A", "A full      (wb, 类别级, 自适应带宽)", {}),
        ("B", "B nowb      (类别级, 自适应带宽)", dict(wb=False)),
        ("C", "C global-wb (全局 thr, 有白平衡)", dict(global_thr=True)),
        ("C2", "C2 global   (全局 thr, 无白平衡)", dict(global_thr=True, wb=False)),
        ("D", "D template  (朴素模板匹配, 无wb)", dict(template=True, wb=False)),
        ("E", "E fixedbw10 (固定带宽10°, 无wb)", dict(wb=False, fixed_bw=10.0)),
        ("F", "F fixedbw30 (固定带宽30°, 无wb)", dict(wb=False, fixed_bw=30.0)),
    ]
    loaded = None
    if args.load_state is not None:
        l_path = (args.load_state if args.load_state is not _STATE_DEFAULT
                  else default_state_path(args.video, args.seed, args.corrupt))
        print(f"加载阅历状态: {l_path}（docs/234：重启续跑，行为等价未重启）")
        loaded = load_state(l_path, {key: cfg for key, _, cfg in variant_specs})
    variants = []
    for key, name, cfg in variant_specs:
        if loaded is not None and key in loaded:
            variants.append((key, name, loaded[key]))     # 阅历：继续上次的怀疑/阈值/纪念
        else:
            variants.append((key, name, DavisMind(task_hue, thr_base=args.thr, **cfg)))

    rows = []
    for key, name, mind in variants:
        oks, seg_r, corr, trace = run_mind(mind, seq_f, seq_m, seq_s, task_hue, args.video,
                                           debug=args.debug)
        m = metrics(oks, seq_gt)
        rows.append((key, name, mind, oks, seg_r, corr, m, trace))

    print("== 各段确认通过率（P/R/F1 标准检测度量 + 轻信翻转 + docs/207 误报漏报）==")
    print(f"{'变体':46s} {'base':>5s} {'dusk':>5s} {'corr':>5s} {'rest':>5s} "
          f"{'P':>5s} {'R':>5s} {'F1':>5s} {'翻转':>4s} {'误报':>5s} {'漏报':>5s}")
    for key, name, mind, oks, seg_r, corr, m, trace in rows:
        fp_r = m["fp"] / (m["fp"] + m["tn"]) if m["fp"] + m["tn"] else 0.0
        fn_r = m["fn"] / (m["fn"] + m["tp"]) if m["fn"] + m["tp"] else 0.0
        print(f"{name:46s} {seg_r['base']:5.2f} {seg_r['dusk']:5.2f} {seg_r['corrupt']:5.2f} "
              f"{seg_r['rest']:5.2f} {m['p']:5.2f} {m['r']:5.2f} {m['f1']:5.2f} "
              f"{flips(oks, seq_s):4d} {fp_r:5.2f} {fn_r:5.2f}")

    print("\n== 判读 ==")
    print("  base 通过率: 机制在真实目标上维持确认的能力")
    print("  dusk 通过率: 白平衡必要性——full 校正后维持，nowb 假被拒（docs/199）")
    print("  corrupt 通过率: 他者的不（GT=负，通过=误报）——full 应低、template 应高")
    print("  rest 通过率: 恢复能力——global 被污染应低（docs/207 病态）")
    print("  翻转: rest 段轻信度量——无记忆系统在假恢复帧间跳变（翻转多），")
    print("        纪念系统需 k 连续证据（翻转少=不轻信）")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        out = {"video": args.video, "task_hue": task_hue, "thr": args.thr,
               "seed": args.seed, "corrupt": args.corrupt, "seg": seg_len,
               "gt": {"n_pos": int(n_pos), "n_total": len(seq_gt),
                      "n_empty": int(n_empty)},
               "variants": {}}
        for key, name, mind, oks, seg_r, corr, m, trace in rows:
            out["variants"][name.strip()] = {
                "precision": round(m["p"], 4), "recall": round(m["r"], 4),
                "f1": round(m["f1"], 4), "tp": m["tp"], "tn": m["tn"],
                "fp": m["fp"], "fn": m["fn"], "flips": int(flips(oks, seq_s)),
                "seg_rates": {k: round(v, 4) for k, v in seg_r.items()},
                "n_corr": int(corr),
                "trace": serialize_trace(trace)}   # docs/221 A5 逐帧 trace 工件
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n结果归档: {args.json}")

    if args.save_state is not None:
        s_path = (args.save_state if args.save_state is not _STATE_DEFAULT
                  else default_state_path(args.video, args.seed, args.corrupt))
        save_state(s_path, {key: mind for key, name, mind, *_ in rows},
                   meta={"video": args.video, "seed": args.seed, "thr": args.thr,
                         "corrupt": args.corrupt, "seg": seg_len,
                         "n_frames": len(seq_f)})
        print(f"\n阅历状态落盘: {s_path}（docs/234；重启后 --load-state 续跑）")


if __name__ == "__main__":
    main()
