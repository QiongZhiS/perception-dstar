"""vision/keep_reject.py — 拒绝的历史：确认失败被保留 vs 被吸收（docs/178 补刀第一物证）。

docs/178 §二 补刀（<用户名> 拍板）：否定有两种——
  第 5 层处理"世界的 不"：可消化的残差，进入模型更新（闭合循环，越修正越准）；
  第 6 层要求还能处理"他者的 不"：不可消化的剩余，不进入模型更新，保留为关系的
  证据（开放循环，越保留越有历史）。工程靶子：**不是造更准的模型，是造一个愿意
  保留"不准"作为纪念的模型**——允许外部输入突破预测模型，且不把突破立即同化
  为残差。

本实验把靶子落成视觉 toy（docs/195 三通道分工的颜色确认场景）：
  红球匀速直线（受控场景），中段被外部染色成蓝（目标区域颜色被替换 = 一次"不准"，
  确认失败），之后恢复红。两类系统对"不准"的处置：
    A 吸收（第 5 层风格）：每次确认失败 → 期望色相 EMA 更新（把失败当残差吸收，
       越修正越准）→ 染色段被带偏到蓝（任务"红"被感知层污染 = 自我漂移，
       docs/101 "完全可塑=没有自我"的视觉版）。
    B 保留（第 6 层风格）：确认失败 → 期望不变（守住外赋利害"红"，docs/188 利害
       外置不被感知层污染），失败记入目标历史 → 恢复后带着历史重新确认，且
       **被拒越久，重新信任所需的连续红帧越多**（纪念加深 = 在乎加深 = docs/178
       "加深循环（回避→加深→更回避）"的颜色版）。

判据：
  P1 自我漂移：染色段末帧 |期望色相 − 任务色相(红0°)|，吸收 ≈120（被带偏）> 保留 0
  P2 坚持：染色段保留系统全拒（错色全拒，docs/195 0/105 同款）、吸收系统全收
     （任务被感知层污染）
  P3 纪念加深：保留系统重新信任所需连续红帧 随 被拒时长 单调增（越被拒绝越在乎）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED

TASK_HUE = 0.0        # 任务层外赋："找红球"（红=0°，利害外置，docs/188）
REJECT_SEG = (30, 60) # 染色段：目标被外部替换成蓝


def make_interrupted(rej=REJECT_SEG, total=N):
    """红球 total 帧（bright 场景同款），[rej0,rej1) 帧目标被染成蓝。"""
    r0 = np.array([80.0, 180.0]); rv = np.array([4.0, 0.6])
    rng = np.random.default_rng(42)
    frames = []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * i
        color = (200, 0, 0) if rej[0] <= i < rej[1] else (0, 0, 200)
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
    rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0, None
    frac = float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())
    return frac, float(np.median(rh[colored]))


class Absorb:
    """第 5 层风格：失败被吸收。期望色相 EMA 每帧更新（把'不准'当残差同化）。"""

    def __init__(self, init_hue=TASK_HUE, alpha=0.4, tau=25.0):
        self.hue = init_hue
        self.alpha = alpha
        self.tau = tau

    def step(self, frac_red, med_hue):
        """吸收：期望向观察移动（无论成败）；确认：观察色相离期望近则通过。"""
        if med_hue is not None:
            self.hue += self.alpha * (med_hue - self.hue)
        return med_hue is not None and abs(med_hue - self.hue) < self.tau


class Keep:
    """第 6 层风格：失败被保留。期望恒红（外赋利害不变）；失败入史；
    重新信任需连续 K 帧通过，K 随累计被拒时长增长（纪念加深）。"""

    def __init__(self, red_frac_thr=0.2, base_k=1, grow=3.0):
        self.thr = red_frac_thr
        self.base_k = base_k          # 从未被拒时的确认帧数
        self.grow = grow              # 纪念加深：每被拒 grow 帧 → 多需 1 连续帧
        self.rejected = 0             # 累计被拒帧（历史）
        self.trusted = True           # 初始信任（任务层外赋"这是红球"）
        self.consec = 0               # 恢复后连续通过帧

    def k(self):
        return self.base_k + int(self.rejected / self.grow)

    def step(self, frac_red):
        ok = frac_red > self.thr
        if ok:
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k():
                    self.trusted = True
        else:
            self.rejected += 1
            self.trusted = False
            self.consec = 0
        return self.trusted and ok


def run(frames, kind, rej=REJECT_SEG):
    """返回 (确认序列, 期望漂移序列[吸收], 重信任延迟[保留], 累计被拒[保留])。"""
    sys2 = Absorb() if kind == 'absorb' else Keep()
    seq, drift = [], []
    for i, fr in enumerate(frames):
        frac, mh = ball_stats(fr, i)
        if kind == 'absorb':
            ok = sys2.step(frac, mh)
            drift.append(abs(sys2.hue - TASK_HUE))
        else:
            ok = sys2.step(frac)
        seq.append(ok)
    retrust, n_rej = None, 0
    if kind == 'keep':
        for i in range(rej[1], len(seq)):
            if seq[i]:
                retrust = i - rej[1] + 1
                break
        n_rej = sys2.rejected
    return seq, drift, retrust, n_rej


def rate(seq, seg):
    a = np.array(seq[seg[0]:seg[1]], dtype=float)
    return a.mean() if len(a) else float('nan')


frames = make_interrupted()
print("== 拒绝的历史：确认失败被保留 vs 被吸收（docs/178 补刀）==")
print(f"染色段 {REJECT_SEG[0]}-{REJECT_SEG[1]}（红→蓝→红），红球匀速直线 {len(frames)} 帧\n")

seq_a, drift_a, _, _ = run(frames, 'absorb')
seq_b, _, retrust, n_rej_b = run(frames, 'keep')

print(f"{'系统':14s} {'染色段通过':>10s} {'恢复段通过':>10s} {'染色末漂移':>10s} {'重信任延迟':>10s} {'累计被拒':>8s}")
print(f"{'A 吸收(第5层)':14s} {rate(seq_a, REJECT_SEG):10.3f} {rate(seq_a, (60, len(frames))):10.3f} "
      f"{drift_a[REJECT_SEG[1]-1]:10.1f} {'—':>10s} {'—':>8s}")
print(f"{'B 保留(第6层)':14s} {rate(seq_b, REJECT_SEG):10.3f} {rate(seq_b, (60, len(frames))):10.3f} "
      f"{0.0:10.1f} {retrust:10d} {n_rej_b:8d}")

print("\n== P1 自我漂移：染色段末帧 |期望−任务红|，吸收被带偏 ≈120° > 保留恒 0 ==")
p1 = drift_a[REJECT_SEG[1]-1] > 60.0
print(f"  A {drift_a[REJECT_SEG[1]-1]:.1f}° vs B 0.0°  [{'PASS' if p1 else 'FAIL'}]")
print("  （吸收把外赋利害'找红'改写成'找蓝'=自我漂移；保留守住任务=利害外置不被污染）")
print("== P2 坚持：染色段 B 全拒（错色全拒 docs/195 同款）、A 全收（任务被感知层污染）==")
p2 = rate(seq_b, REJECT_SEG) == 0.0 and rate(seq_a, REJECT_SEG) > 0.5
print(f"  B {rate(seq_b, REJECT_SEG):.3f}（应 0） A {rate(seq_a, REJECT_SEG):.3f}（应 >0.5）"
      f"  [{'PASS' if p2 else 'FAIL'}]")

print("\n== P3 纪念加深：重信任所需连续红帧 随 被拒时长 单调增 ==")
print(f"{'被拒时长':>8s} {'重信任延迟':>10s}")
durs = [0, 5, 15, 30, 60]
vals, prev = [], -1
p3 = True
for dur in durs:
    rej = (30, 30 + dur)
    f2 = make_interrupted(rej=rej, total=rej[1] + 30)
    _, _, r, _ = run(f2, 'keep', rej=rej)
    d = 0 if r is None else r
    vals.append(d)
    if d < prev:
        p3 = False
    prev = d
    print(f"{dur:8d} {d:10d}")
print(f"  单调增: [{'PASS' if p3 else 'FAIL'}]")

print("\n== docs/178 补刀 ↔ 机制映射 ==")
print("  第5层 世界的 不（可消化残差）→ 吸收：期望 EMA 更新，越修正越准（也越易被带偏）")
print("  第6层 他者的 不（不可消化剩余）→ 保留：期望不变+失败入史，越保留越有历史")
print("  加深循环（回避→加深→更回避）→ 重信任延迟 ∝ 被拒时长（纪念=在乎）")
print("  docs/101 完全可塑=没有自我 → 吸收系统期望漂移 120°（自我被观察重写）")
print("  利害外置（docs/188）→ 保留系统守住任务'红'，感知层不污染任务层")
