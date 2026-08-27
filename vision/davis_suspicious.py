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
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402

DAVIS = os.path.join("vision", "out", "davis")
N_SEG = {"base": 20, "dusk": 20, "corrupt": 15, "rest": 25}   # flamingo 80 帧
SAT_MIN_TARGET, SAT_MIN_BG = 60, 60


def circ(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def load_video(video):
    vdir = os.path.join(DAVIS, video)
    if not os.path.isdir(vdir):
        raise FileNotFoundError(
            f"DAVIS 数据未找到: {vdir}\n"
            f"请先运行 vision/davis_setup.py 下载抽取，或用 --davis <目录> 指向"
            f"已有的 DAVIS 数据（如 synthetic-life 的 vision/out/davis）。")
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


def build_sequence(frames, masks, seg_len, seed=7, corrupt="gain"):
    """按段构造序列：真实帧 + 受控扰动。返回 (帧, 掩码, 段标签, 逐帧GT)。

    段语义（docs/178 两种否定的真实版）：
      base    真实帧                           → 确认通过（维持）
      dusk    色温增益渐变（连续漂移=世界变化） → 时间门+白平衡→维持；无校正→假被拒
      corrupt 目标区颜色变化（他者的不）        → 确认退化→被拒→suspicious 沉积
      rest    真实帧 + 假恢复帧                → 带纪念重信任（docs/200）；rest 段
               前 8 帧混入 4 帧 corrupt（目标"看似恢复实则仍错"）→ 验证"不轻信"
    全图轻噪声（std 10）加在 base/dusk/rest：世界噪声（docs/101 固执，不误改判）。
    corrupt 方式：
      gain  温和冷增益（目标色相漂移 ~15-25°，记账 hue 接近目标——带宽实验需要）
      noise 目标区强噪声（目标观测全毁，记账 hue 乱——目标消失/破坏验证需要）
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
            n = rng.normal(0, 10, src.shape).astype(np.int16)
            src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
            if mask.max() > 0:
                if corrupt == "gain":
                    kc = 0.05                      # 温和冷增益（局部色温变化 ~15-25°）
                    g = np.array([1 + 0.9 * kc, 1.0, 1 - 0.9 * kc])
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
                n = rng.normal(0, 10, src.shape).astype(np.int16)
                src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
                if mask.max() > 0:
                    if corrupt == "gain":
                        g = np.array([1 + 0.9 * 0.05, 1.0, 1 - 0.9 * 0.05])
                        src = src.astype(np.float32)
                        src[mask > 0] *= g
                        src = np.clip(src, 0, 255).astype(np.uint8)
                    else:
                        n2 = rng.normal(0, 90, src.shape).astype(np.int16)
                        src = np.clip(src.astype(np.int16) + n2 * (mask > 0)[:, :, None],
                                      0, 255).astype(np.uint8)
            else:
                n = rng.normal(0, 10, src.shape).astype(np.int16)
                src = np.clip(src.astype(np.int16) + n, 0, 255).astype(np.uint8)
        else:
            n = rng.normal(0, 10, src.shape).astype(np.int16)
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
                 thr_cap=0.85, relax=0.02, persist_for=3, grow=3.0, cap_k=60):
        self.task_hue = task_hue
        self.wb, self.global_thr, self.template = wb, global_thr, template
        self.fixed_bw = fixed_bw
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

    def bw(self, hue):
        if self.fixed_bw is not None:
            return self.fixed_bw
        if self.template:
            return 30.0
        obs = self.hist.get(hue, [])
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
        suspicious 表只用于类别级 thr（docs/205/206）；纪念是全局的（demo 同款）。"""
        if self.template:
            return 1
        return 1 + int(min(self.rej_total, self.cap_k) / self.grow)

    def thr(self, hue):
        if hue is None or self.template or self.global_thr:
            return self.thr_cur if (self.template or self.global_thr) else self.thr_base
        t = self.thr_base
        for sh, dv in self.thr_by.items():
            t += dv * self._w_wide(circ(hue, sh), self.bw(sh))
        return min(self.thr_cap, t)

    def step(self, frac, hue, mask_empty):
        if hue is not None:
            self.hist.setdefault(hue, []).append(hue)
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


def main():
    global DAVIS
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="flamingo")
    ap.add_argument("--davis", default=None,
                    help="DAVIS 数据目录（含 <video>/ 子目录）；默认 vision/out/davis")
    ap.add_argument("--seg", default="20,20,15,25",
                    help="base,dusk,corrupt,rest 帧数")
    ap.add_argument("--thr", type=float, default=0.60,
                    help="确认阈值基线（真实目标确认余量不同：flamingo 0.60/surf 0.40）")
    ap.add_argument("--corrupt", default="gain", choices=["gain", "noise"],
                    help="他者的不的注入方式：gain=温和色相偏移 / noise=目标区强噪声")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.davis:
        DAVIS = args.davis
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
                                                 corrupt=args.corrupt)
    n_empty = sum(1 for m in seq_m if m.max() == 0)
    n_pos = sum(seq_gt)
    print(f"序列 {len(seq_f)} 帧（base/dusk/corrupt/rest 段；rest 前 8 帧混 4 假恢复），"
          f"空掩码帧（目标消失）{n_empty}，GT 正样本 {n_pos}/{len(seq_gt)}\n")

    mk = lambda **kw: DavisMind(task_hue, thr_base=args.thr, **kw)   # noqa: E731
    variants = [
        ("A full      (wb, 类别级, 自适应带宽)", mk()),
        ("B nowb      (类别级, 自适应带宽)", mk(wb=False)),
        ("C global-wb (全局 thr, 有白平衡)", mk(global_thr=True)),
        ("C2 global   (全局 thr, 无白平衡)", mk(global_thr=True, wb=False)),
        ("D template  (朴素模板匹配, 无wb)", mk(template=True, wb=False)),
        ("E fixedbw10 (固定带宽10°, 无wb)", mk(wb=False, fixed_bw=10.0)),
        ("F fixedbw30 (固定带宽30°, 无wb)", mk(wb=False, fixed_bw=30.0)),
    ]

    rows = []
    for name, mind in variants:
        oks, seg_r, corr, trace = run_mind(mind, seq_f, seq_m, seq_s, task_hue, args.video,
                                           debug=args.debug)
        m = metrics(oks, seq_gt)
        rows.append((name, mind, oks, seg_r, corr, m, trace))

    print("== 各段确认通过率（P/R/F1 标准检测度量 + 轻信翻转 + docs/207 误报漏报）==")
    print(f"{'变体':46s} {'base':>5s} {'dusk':>5s} {'corr':>5s} {'rest':>5s} "
          f"{'P':>5s} {'R':>5s} {'F1':>5s} {'翻转':>4s} {'误报':>5s} {'漏报':>5s}")
    for name, mind, oks, seg_r, corr, m, trace in rows:
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


if __name__ == "__main__":
    main()
