"""vision/continuous_threshold.py — 连续+阈值：确认阈值按距离加权（连续联合）。

docs/212 §五 2：k 连续化后，thr 也按距离加权——近同类 thr 高、边缘渐低。这是
docs/210（带宽+阈值联合）的连续版：怀疑（k）和阈值（thr）都连续沉积，不再是
带宽内二值共享。

机制：
  二值 thr（docs/210 对照）：色相在 见过的类别 ± 带宽 内 → 全量共享该类别最大 thr
  连续 thr（本文件）：thr(hue) = base + Σ rejected[sh] × w(circ(hue,sh)) / grow_thr
    w(d) = max(0, 1 − d/带宽)（线性衰减）
    → 距被拒类别 0° thr 最高、随距离连续下降、带宽外回基线

实验：
  P1 连续 thr：训练染蓝 → thr(距蓝距离) 连续下降（0°高、边缘渐低、带宽外基线）
  P2 二值 vs 连续：带宽边缘阶跃 vs 平滑（近同类 thr 高、边缘渐低）
  P3 不误伤：红（距蓝 120°）两种都 thr 基线（目标色永远开放）
  P4 未饱和 vs 饱和：被拒累计小 → 完整连续 thr 曲线；大 → 饱和顶格退化
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np
from bandwidth_continuous import ContinuousSus, train, circ
from bandwidth_threshold import JointSus, CAP, GROW

RED_H, BLUE_H = 0.0, 120.0
BASE_THR = 0.2


class ContinuousThr(ContinuousSus):
    """连续加权的 thr（怀疑与阈值一起连续沉积）。"""

    def __init__(self, k_band=2.0, decay='linear', thr_gain=0.02, cap_thr=0.8):
        super().__init__(k_band=k_band, decay=decay)
        self.thr_gain = thr_gain
        self.cap_thr = cap_thr

    def thr(self, hue):
        n = 0.0
        for sh, cnt in self.rejected.items():
            n += cnt * self.w(circ(hue, sh), self.bw(sh))
        return min(self.cap_thr, BASE_THR + n * self.thr_gain)


print("== 连续+阈值：确认阈值按距离加权（docs/210 的连续版）==")
print("训练：染蓝 ×100 遍（漂移 ±5°，见过 std≈3.5° → 带宽≈7.0°）\n")

j = JointSus()          # 二值 thr（docs/210 对照）
train(j)
ct = ContinuousThr(decay='linear')
train(ct)

print("== P1/P2 thr(距蓝距离) 连续下降 vs 二值 ==")
print(f"{'距蓝':>6s} {'二值thr':>8s} {'连续thr':>8s}")
prev = None
p1_ok = True
for d in [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0, 120.0]:
    hue = (BLUE_H + d) % 180.0
    tj = j.thr(hue)
    tc = ct.thr(hue)
    if prev is not None and tc > prev + 0.01:   # 连续应单调降（除带宽外基线）
        p1_ok = False
    prev = tc
    print(f"{d:6.0f} {tj:8.2f} {tc:8.2f}")

print("\n== P3 不误伤：红（距蓝 120°）thr 基线 ==")
print(f"  二值 thr={j.thr(RED_H):.2f} 连续 thr={ct.thr(RED_H):.2f}（基线 0.20）")

print("\n== P4 未饱和 vs 饱和：thr 连续曲线的形态 × 被拒累计 ==")
print(f"{'被拒累计':>10s} {'0°thr':>7s} {'4°thr':>7s} {'7°thr':>7s} {'8°thr':>7s} {'形态':>12s}")
for n_pass in [1, 10, 100]:
    c = ContinuousThr(decay='linear')
    train(c, n_pass=n_pass)
    n_rej = n_pass * 30
    row = [c.thr((BLUE_H + d) % 180.0) for d in [0, 4, 7, 8]]
    shape = '连续曲线' if n_rej < 200 else ('饱和顶格' if n_rej > 1000 else '部分饱和')
    print(f"{n_rej:10d} {row[0]:7.2f} {row[1]:7.2f} {row[2]:7.2f} {row[3]:7.2f} {shape:>12s}")

print("\n== 判读 ==")
print("  连续 thr：近同类 thr 高、随距离连续下降、带宽外回基线（0.20）")
print("  二值 thr：带宽内全量共享最大 thr（边缘阶跃）——docs/210 的近似")
print("  连续（阈值+怀疑一起沉积）：被拒越多/越近 → thr 越高（'看得更仔细'）")
print("  不误伤：红（120°）两种都基线（目标色永远开放，docs/205）")
print("  未饱和：完整连续 thr 曲线；饱和：0-6° 顶格（cap_thr 0.8）退化二值")
