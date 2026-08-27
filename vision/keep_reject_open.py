"""vision/keep_reject_open.py — 开放端补测：守住 vs 改判的边界（docs/200 边界 #2）。

docs/200 诚实边界 #2：保留系统的"期望不变"是绝对固执（永不改任务色相）——真实主体
需要在"守住利害"与"利害确已改变"间判断（docs/101 平衡带：eta 随世界硬度调）。
docs/200 只测了固执端；本文件补开放端。

机制设计（滞回改判，避免"一拒就改→改完又被带偏→再改"的震荡）：
  KeepSig 双状态滞回机：
    - 守住态（任务=红）：ok = 红区占比>0.2（docs/200 同款确认）；连续被拒帧数
      达到 K_switch → 判定"利害确已改变" → 改判（期望色相更新到观察蓝）。
    - 改判态（任务=蓝）：ok = |观察色相 − 蓝| < tau；连续被拒帧数达到 K_back
      → 回到守住。改判与回改需要同等强度的连续证据（对称滞回 = docs/101 平衡带
      的颜色轴：eta 随被拒统计显著性调，不是绝对不变，也不是一拒就改）。

对照：
  B0 绝对保留（docs/200 原版）：期望恒红，永不改判 —— 世界真变蓝时永远拒绝（守错）
  B1 显著性改判（开放端，K_switch 标定）：
     世界真变蓝（永久）→ 改判 → 确认通过（能学习）
     短暂染色后恢复（30 帧）→ K_switch>30 时守住（同 B0）；K_switch<30 时被带偏
     （展示"改判阈值太松"的代价 = docs/101"完全可塑"的另一面）

判据：
  Q1 守住：短暂染色（30 帧）后恢复红 —— K_switch=35 时 B1 不改判（同 B0）
  Q2 改判：世界长期变蓝 —— B1 改判（后段通过率高），B0 永远拒绝
  Q3 边界：K_switch=25（<30）时 B1 被短暂染色带偏（改判后又回改，恢复有延迟）——
     阈值太松 = 被单次外部染色重写的另一条路径
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import W, H, RED

TASK_HUE = 0.0    # 任务层外赋："找红球"（红=0°）
BLUE_HUE = 120.0  # 蓝 BGR(200,0,0) → HSV H≈120（观察中位的标定值）


def make_ball(total, color_seq, rng_seed=42):
    """红球匀速直线 total 帧；color_seq = [(start, end, bgr), ...] 颜色段。"""
    r0 = np.array([80.0, 180.0]); rv = np.array([4.0, 0.6])
    rng = np.random.default_rng(rng_seed)
    frames = []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * i
        color = (0, 0, 200)  # 默认红
        for (s, e, c) in color_seq:
            if s <= i < e:
                color = c
                break
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, color, -1)
        frames.append(img)
    return frames


def ball_stats(fr, i):
    """球体掩码（半径18，真值位置）：红区占比 + 彩色像素 H 中位。"""
    rc = (80 + 4.0 * i, 180 + 0.6 * i)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    r = 18
    cx, cy = int(rc[0]), int(rc[1])
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return 0.0, None
    rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0, None
    frac = float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())
    return frac, float(np.median(rh[colored]))


class KeepAbs:
    """B0 绝对保留（docs/200 原版）：期望恒红，永不改判。"""

    def __init__(self, red_frac_thr=0.2):
        self.thr = red_frac_thr
        self.hue = TASK_HUE
        self.state = 'red'

    def step(self, frac_red, med_hue):
        return frac_red > self.thr


class KeepSig:
    """B1 显著性改判（开放端）：守住 + 连续被拒显著才改判（滞回，docs/101 平衡带）。"""

    def __init__(self, red_frac_thr=0.2, k_switch=35, k_back=15, tau=40.0,
                 blue_hue=BLUE_HUE, red_hue=TASK_HUE):
        self.thr = red_frac_thr
        self.k_switch = k_switch
        self.k_back = k_back
        self.tau = tau
        self.blue_hue = blue_hue
        self.red_hue = red_hue
        self.state = 'red'          # 'red' 守住 / 'blue' 改判
        self.rej_consec = 0         # 连续被拒帧数（当前状态的证据强度）
        self.hue = red_hue          # 当前任务期望色相
        self.switch_frame = None    # 记录改判发生的帧

    def step(self, frac_red, med_hue):
        if self.state == 'red':
            ok = frac_red > self.thr
            if ok:
                self.rej_consec = 0
            else:
                self.rej_consec += 1
                if self.rej_consec >= self.k_switch and med_hue is not None:
                    self.state = 'blue'          # 利害确已改变：改判
                    self.hue = self.blue_hue
                    self.rej_consec = 0
        else:  # 'blue' 改判态：确认=观察色相接近蓝
            ok = med_hue is not None and abs(med_hue - self.blue_hue) < self.tau
            if ok:
                self.rej_consec = 0
            else:
                self.rej_consec += 1
                if self.rej_consec >= self.k_back:
                    self.state = 'red'           # 世界回来了：回改
                    self.hue = self.red_hue
                    self.rej_consec = 0
        return ok


def run_seq(frames, sys2):
    """跑一序列，返回 (确认序列, 状态序列)。"""
    seq, states = [], []
    for i, fr in enumerate(frames):
        frac, mh = ball_stats(fr, i)
        ok = sys2.step(frac, mh)
        seq.append(ok)
        states.append(sys2.state)
    return seq, states


def rate(seq, seg):
    a = np.array(seq[seg[0]:seg[1]], dtype=float)
    return a.mean() if len(a) else float('nan')


def first_switch(states):
    """改判发生的首帧（state 从 red 变 blue），无则 None。"""
    for i in range(1, len(states)):
        if states[i] == 'blue' and states[i-1] == 'red':
            return i
    return None


def back_after(frames, seq, seg):
    """恢复段（染色结束后）首次重新确认通过的帧数延迟；无则 None。"""
    for i in range(seg[1], len(frames)):
        if seq[i]:
            return i - seg[1] + 1
    return None


# 场景 1：短暂染色（帧 30-59 蓝）后恢复红，共 90 帧
S1 = (30, 60)
frames_s1 = make_ball(90, [(S1[0], S1[1], (200, 0, 0))])
# 场景 2：世界长期变蓝（帧 30 起永久蓝），共 150 帧
S2 = (30, 150)
frames_s2 = make_ball(150, [(S2[0], S2[1], (200, 0, 0))])

print("== 开放端补测：守住 vs 改判的边界（docs/200 边界 #2）==")
print("B0 绝对保留（docs/200 原版）：期望恒红，永不改判")
print("B1 显著性改判（开放端）：连续被拒 ≥ K_switch 才改判；回改需同等连续证据\n")

rows = []
for label, frames, seg in [("场景1 短暂染色(30帧)", frames_s1, S1),
                           ("场景2 世界长期变蓝", frames_s2, S2)]:
    tail = (seg[1], len(frames))
    for sys_name, sys2, ks in [("B0 绝对保留", KeepAbs(), None),
                               ("B1 改判 K=35", KeepSig(k_switch=35, k_back=15), 35),
                               ("B1 改判 K=25", KeepSig(k_switch=25, k_back=15), 25)]:
        seq, states = run_seq(frames, sys2)
        r_rej = rate(seq, seg)
        sw = first_switch(states)
        if sw is not None:
            # 改判后通过率（sw 到段尾）
            r_after = rate(seq, (sw, len(frames)))
        else:
            r_after = float('nan')
        delay = back_after(frames, seq, seg)
        rows.append((label, sys_name, r_rej, sw, r_after, states[-1], delay))
        print(f"{label:22s} {sys_name:12s} 染色段通过 {r_rej:.3f}  改判帧 {str(sw):>5s}"
              f"  改判后通过 {r_after:.3f}  末态 {states[-1]:4s}  恢复延迟 {str(delay):>4s}")

print("\n== 判读 ==")
print("Q1 守住：场景1 + K=35 —— 30帧<35 不触发改判（改判帧 None），染色段通过≈0（同 B0），"
      "恢复延迟≈1")
print("Q2 改判：场景2 + K=35 —— B1 改判（改判帧≈35）后蓝通过，末态 blue；"
      "B0 永远拒绝（无改判帧）")
print("Q3 边界：场景1 + K=25 —— 25<30 被短暂染色触发改判（改判帧≈25，被单次外部染色"
      "重写），恢复后需 K_back 连续红帧才回改 → 恢复延迟大 = 阈值太松的代价")
