"""vision/layered_decay.py — 怀疑与阈值不同衰减：疑的窄、严的宽。

docs/214 §五 2：k 用高斯（快衰减=该疑的范围窄）、thr 用线性（慢衰减=该严的范围
宽）——分层策略：怀疑要明确（窄），确认要谨慎（宽）。docs/101 多旋钮的衰减维度。

语义：被蓝骗过——"我怀疑的范围窄（只有很接近蓝的才算可疑同类），但我确认时
范围内都要严（近同类都看得仔细）"——怀疑窄（避免误伤远的）、确认宽（防漏）。

机制：
  同衰减（docs/214 基线）：k 线性、thr 线性（同步衰减）
  分层（本文件）：k 高斯（带宽内快衰减→疑的窄）、thr 线性（带宽内慢衰减→严的宽）
实验：
  P1 分层 vs 同衰减：带宽边缘 k 先降（高斯，疑的窄）、thr 后降（线性，严的宽）
  P2 覆盖比较：分层在带宽边缘 k<thr 的区间（疑窄严宽）——同衰减 k≈thr（同步）
  P3 不误伤：红（120°）两种都基线
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np
from bandwidth_continuous import ContinuousSus, train, circ
from continuous_threshold import ContinuousThr, BASE_THR

RED_H, BLUE_H = 0.0, 120.0


class Layered(ContinuousThr):
    """分层衰减：k 用窄高斯（σ=bw/4，快衰减=疑的窄）、thr 用线性（慢衰减=严的宽）。

    docs/214 §五 2 修正：σ=bw/2 的高斯在带宽内全程高于线性（更尖更宽，不是更窄）；
    真正的"窄"衰减 = σ=bw/4（中心尖、边缘快降）。分层语义：怀疑窄（只有很近的
    算同类，不误伤远的）、确认宽（范围内都看得仔细，不漏近的）。
    """

    def __init__(self, k_band=2.0, thr_gain=0.02, cap_thr=0.8):
        super().__init__(k_band=k_band, decay='linear', thr_gain=thr_gain,
                         cap_thr=cap_thr)

    def _w_narrow(self, d, bw):
        if bw <= 0:
            return 1.0 if d == 0 else 0.0
        if d >= bw:
            return 0.0
        return float(np.exp(-(d ** 2) / (2 * (bw / 4) ** 2)))   # 窄高斯

    def k(self, hue):
        n = 0.0
        for sh, cnt in self.rejected.items():
            n += cnt * self._w_narrow(circ(hue, sh), self.bw(sh))   # 疑的窄
        return 1 + int(min(n, 60) / 3.0)    # saturate（docs/203 同款）


print("== 怀疑与阈值不同衰减：疑的窄、严的宽 ==")
print("训练：染蓝 ×100 遍（漂移 ±5°，见过 std≈3.5° → 带宽≈7.0°）\n")

# 同衰减（docs/214）：k 线性 + thr 线性
ct = ContinuousThr(decay='linear')
train(ct)
# 分层：k 高斯 + thr 线性
ly = Layered()
train(ly)

print("== P1/P2 带宽边缘行为：k(距离) vs thr(距离) ==")
print("（未饱和区展示：训练 ×5 遍，被拒 150 次——饱和掩盖分层差异，docs/213 交互）")
ct5 = ContinuousThr(decay='linear')
train(ct5, n_pass=5)
ly5 = Layered()
train(ly5, n_pass=5)
print(f"{'距蓝':>6s} {'同衰减k':>8s} {'分层k(高斯)':>10s} {'同衰减thr':>10s} {'分层thr(线性)':>12s}")
for d in [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 15.0, 120.0]:
    hue = (BLUE_H + d) % 180.0
    print(f"{d:6.0f} {ct5.k(hue):8d} {ly5.k(hue):10d} {ct5.thr(hue):10.2f} {ly5.thr(hue):12.2f}")

print("\n== P2 疑窄严宽的区间（分层 k 比同衰减先降到低值、thr 仍高）==")
print(f"{'距蓝':>6s} {'同衰减k':>8s} {'分层k':>8s} {'分层thr':>8s} {'疑窄严宽?':>10s}")
cnt = 0
for d in [4.0, 5.0, 6.0, 7.0, 8.0]:
    hue = (BLUE_H + d) % 180.0
    k_same = ct5.k(hue)
    k_layer = ly5.k(hue)
    t_layer = ly5.thr(hue)
    narrow = k_layer < k_same and t_layer > 0.2   # k 比同衰减低、thr 还高
    cnt += narrow
    print(f"{d:6.0f} {k_same:8d} {k_layer:8d} {t_layer:8.2f} {str(narrow):>10s}")
print(f"  P2: [{'PASS' if cnt > 0 else 'FAIL'}] 带宽边缘存在'疑窄严宽'区间"
      f"（k 快衰减、thr 慢衰减）\n")

print("\n== P3 不误伤：红（距蓝 120°）永远基线 ==")
print(f"  同衰减 k={ct.k(RED_H)} thr={ct.thr(RED_H):.2f}; "
      f"分层 k={ly.k(RED_H)} thr={ly.thr(RED_H):.2f}")

print("\n== 判读 ==")
print("  分层（k 高斯 + thr 线性）：带宽边缘 k 快衰减（疑的窄——只有很近的才算")
print("    同类），thr 慢衰减（严的宽——范围内都看得仔细）")
print("  同衰减（docs/214）：k 和 thr 同步衰减（疑严同宽同窄）")
print("  语义：怀疑要明确（不误伤远的），确认要谨慎（不漏近的）——docs/101 多旋钮")
print("  不误伤：红（120°）两种都基线（目标色开放，docs/205）")
