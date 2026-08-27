"""vision/bandwidth_continuous.py — 连续色相插值：带宽内怀疑按距离加权。

docs/210 §五 2：从"邻域共享"（带宽内二值全量）到"距离连续沉积"——距被拒类别
越近越疑（连续衰减），不是阶跃。这更接近真实怀疑的心理物理（docs/178 否定层：
差异强度连续）。

机制：
  二值共享（docs/210 对照）：色相在 见过的类别 ± 带宽 内 → 全量共享该类别可疑
  连续加权（本文件）：k(hue) = 1 + Σ rejected[sh] × w(circ(hue, sh)) / grow
    w(d) = max(0, 1 − d/带宽)（线性衰减）或 exp(−d²/2σ²)（高斯）
    → 距被拒类别 0° 最疑、随距离连续下降、带宽外为 0

实验：
  P1 连续曲线：训练染蓝（见过漂移）→ k(距离) 曲线——距蓝 0°/5°/15°/30°/60° 的
     怀疑连续下降（不是阶跃）
  P2 二值 vs 连续：阶跃（带宽内全量）vs 线性/高斯衰减——连续在带宽边缘平滑过渡
     （二值在带宽边缘突变：6° 全疑、7° 全不疑）
  P3 不误伤：红（距蓝 120° 远超带宽）两种都 k=1（目标色永远开放）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np
from bandwidth_threshold import JointSus, circ, CAP, GROW

RED_H, BLUE_H = 0.0, 120.0


class ContinuousSus:
    """连续色相插值：怀疑按距离加权（线性/高斯衰减）。"""

    def __init__(self, k_band=2.0, decay='linear'):
        self.k_band = k_band
        self.decay = decay
        self.hist = {}
        self.rejected = {}

    def bw(self, hue):
        obs = self.hist.get(hue, [])
        if len(obs) < 3:
            return 0.0
        return self.k_band * float(np.std(obs))

    def w(self, d, bw):
        """距离权重：带宽内线性/高斯衰减，带宽外 0。"""
        if bw <= 0:
            return 1.0 if d == 0 else 0.0
        if self.decay == 'linear':
            return max(0.0, 1.0 - d / bw)
        return np.exp(-(d ** 2) / (2 * (bw / 2) ** 2)) if d < bw else 0.0

    def k(self, hue):
        n = 0.0
        for sh, cnt in self.rejected.items():
            n += cnt * self.w(circ(hue, sh), self.bw(sh))
        return 1 + int(min(n, CAP) / GROW)

    def observe(self, hue, obs):
        if hue is not None:
            self.hist.setdefault(hue, []).append(obs)

    def confirm(self, frac, hue, obs):
        self.observe(hue, obs)
        ok = frac > 0.2
        if not ok and hue is not None:
            self.rejected[hue] = self.rejected.get(hue, 0) + 1
        return ok


def train(sus, n_pass=100, drift_amp=5.0, seed=42):
    rng = np.random.default_rng(seed)
    for _ in range(n_pass):
        hue = BLUE_H + rng.uniform(-drift_amp, drift_amp)
        for _ in range(30):
            sus.confirm(0.15, BLUE_H, hue + rng.normal(0, 2.0))
        for _ in range(30):
            sus.confirm(0.8, RED_H, 0.0)


print("== 连续色相插值：带宽内怀疑按距离加权 ==")
print("训练：染蓝 ×100 遍（漂移 ±5°，见过 std≈3.5°）\n")

j = JointSus()          # 二值共享（docs/210 对照）
train(j)
c_lin = ContinuousSus(decay='linear')
train(c_lin)
c_gau = ContinuousSus(decay='gauss')
train(c_gau)
bw = j.k_band * np.std(j.hist[BLUE_H])

print(f"== P1 连续曲线：k(距蓝距离) 连续下降 ==")
print(f"{'距蓝':>6s} {'二值共享k':>10s} {'线性衰减k':>10s} {'高斯衰减k':>10s}")
prev = None
for d in [0.0, 2.0, 5.0, 8.0, 15.0, 30.0, 60.0, 120.0]:
    hue = (BLUE_H + d) % 180.0
    kj = j.k(hue)
    kl = c_lin.k(hue)
    kg = c_gau.k(hue)
    print(f"{d:6.0f} {kj:10d} {kl:10d} {kg:10d}")

print("\n== P2 二值 vs 连续：带宽边缘（~6-8°）的行为 ==")
print(f"  带宽≈{bw:.1f}°：二值在带宽边缘突变（{bw-1:.0f}° 全疑、{bw+1:.0f}° 全不疑）；")
print(f"  线性/高斯在边缘平滑过渡（距蓝 {bw:.0f}° 处 k 连续降到 ~1）")
for d in [5.0, 6.0, 7.0, 8.0, 9.0]:
    hue = (BLUE_H + d) % 180.0
    print(f"    距蓝 {d:.0f}°: 二值 k={j.k(hue)} 线性 k={c_lin.k(hue)} 高斯 k={c_gau.k(hue)}")

print("\n== P3 不误伤：红（距蓝 120°）永远 k=1 ==")
print(f"  二值 k={j.k(RED_H)} 线性 k={c_lin.k(RED_H)} 高斯 k={c_gau.k(RED_H)}")

print("\n== 判读 ==")
print("  连续加权：怀疑随距离连续衰减（近同类更疑、远同类渐淡）——不是阶跃")
print("  二值共享：带宽内全量（边缘突变：6°全疑 7°全不疑）——docs/210 的近似")
print("  连续（线性/高斯）：带宽边缘平滑——更接近真实怀疑（docs/178 否定层：")
print("     差异强度连续，不是'是同类/非同类'的二值）")
print("  不误伤：红（120°）远超带宽，两种都 k=1（目标色永远开放）")
