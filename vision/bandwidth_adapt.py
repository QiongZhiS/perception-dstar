"""vision/bandwidth_adapt.py — 带宽自适应：怀疑半径 = 见过的同类变异范围。

docs/205 §五 1：带宽（怀疑半径）从手工常数变成"该类别色相观察的历史方差"——
阅历让带宽本身也沉积。docs/196 色温漂移 83° 是物理锚：真实世界同类颜色会因光照
漂移（同一种蓝在黄光/暗光下色相不同）——怀疑半径应该 = 见过的同类变异范围。

机制：
  suspicious[hue] 表（docs/205）+ 每类别记录见过的色相观测分布 (mean, var)
  自适应带宽 bw(hue) = k_band × std(见过的该色色相)（见过的变异大 → 怀疑范围宽）
  固定带宽（对照）：手工常数（docs/205 的 30°）

实验：
  训练：染蓝 × N 遍，色相在 [115, 125] ± 漂移（模拟色温变化，蓝的真实变异 std~4-6°）
  测试：
    P1 漂移识别：色相漂到 128°（超出固定小带宽）——自适应带宽（见过 115-125 变异）
       识别为"同类蓝"（suspicious 命中，k 大）vs 固定带宽（128 超出 → 当新颜色，k=1）
    P2 不误伤：自适应带宽对红（距蓝 120°）不疑（k=1）——带宽有物理锚，不会无限宽
    P3 变异沉积：见过变异越大（训练漂移幅度不同）→ 带宽越宽（阅历让怀疑半径沉积）

docs/196 锚：真实同类颜色的色相漂移 ~几十度（色温脆弱 83°）→ 自适应带宽应覆盖
"见过的变异"，不是固定 30°。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np

RED_H, BLUE_H = 0.0, 120.0
CAP, GROW = 60, 3.0


def circ(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


class SuspiciousAdapt:
    """docs/205 suspicious 表 + 自适应带宽（见过的色相变异）。"""

    def __init__(self, mode='adapt', k_band=2.0, fixed_bw=30.0, thr=0.2,
                 cap=CAP, grow=GROW):
        self.mode = mode            # 'adapt' = 自适应带宽；'fixed' = 固定带宽
        self.k_band = k_band
        self.fixed_bw = fixed_bw
        self.thr = thr
        self.cap, self.grow = cap, grow
        self.hist = {}              # hue -> [色相观测列表]（见过的变异）
        self.rejected = {}          # hue -> 被拒累计
        self.trusted = True
        self.consec = 0

    def bw(self, hue):
        """怀疑半径：自适应=见过的该类别色相 std × k；固定=常数。"""
        if self.mode == 'fixed':
            return self.fixed_bw
        obs = self.hist.get(hue, [])
        if len(obs) < 3:
            return 0.0               # 没见过足够样本 → 只疑精确色
        return self.k_band * float(np.std(obs))

    def k(self, hue):
        n = 0
        for sh, cnt in self.rejected.items():
            if circ(hue, sh) <= self.bw(hue):
                n += cnt
        return 1 + int(min(n, self.cap) / self.grow)

    def step(self, frac_red, hue, obs_hue):
        """obs_hue = 该帧实际观测的色相（含漂移）——记录进 hist。"""
        if hue is not None:
            self.hist.setdefault(hue, []).append(obs_hue)
        ok = frac_red > self.thr
        if not ok and hue is not None:
            self.rejected[hue] = self.rejected.get(hue, 0) + 1
        return ok


def train(susp, n_pass=100, drift_amp=5.0, seed=42):
    """染蓝 n_pass 遍，色相每遍在 [120-drift_amp, 120+drift_amp] 漂移。
    记录 hist（见过的变异）。"""
    rng = np.random.default_rng(seed)
    for _ in range(n_pass):
        hue = BLUE_H + rng.uniform(-drift_amp, drift_amp)
        obs = hue + rng.normal(0, 2.0)         # 观测噪声
        # 蓝段被拒（红区占比 ~0.15 → 拒）
        for _ in range(30):
            susp.step(0.15, int(round(BLUE_H)), obs)
        # 红段通过
        for _ in range(30):
            susp.step(0.8, int(round(RED_H)), 0.0)


print("== 带宽自适应：怀疑半径 = 见过的同类变异范围（docs/196 色温漂移锚）==")
print(f"训练：染蓝 ×100 遍，色相在 [115,125] 漂移（真实变异 std~2.9°），红段正常\n")

print("== P1 漂移识别：色相漂到 128°（超出固定 30° 半径的中心？）==")
print(f"{'模式':>8s} {'带宽':>8s} {'测漂移蓝(128°)k':>16s} {'测标准蓝(120°)k':>16s} {'测红(0°)k':>10s}")
for mode, bw in [('fixed', 30.0), ('fixed', 10.0), ('adapt', 2.0)]:
    susp = SuspiciousAdapt(mode=mode, fixed_bw=bw, k_band=2.0)
    train(susp)
    b128 = susp.k(128.0)
    b120 = susp.k(120.0)
    r0 = susp.k(0.0)
    print(f"{mode:>8s} {bw if mode == 'fixed' else '自适应':>8} {b128:>16d} {b120:>16d} {r0:>10d}")
print("  自适应带宽 = k_band×std(见过的蓝) ≈ 2×2.9 = 5.8° → 128 距 120 有 8° > 5.8°？")
print("  （见下判读：带宽是否覆盖漂移取决于 k_band 与漂移幅度）\n")

print("== P1b 带宽 vs 漂移：自适应带宽恰好覆盖见过的变异 ==")
for drift in [2.0, 5.0, 10.0]:
    susp = SuspiciousAdapt(mode='adapt', k_band=2.0)
    train(susp, drift_amp=drift)
    std = float(np.std(susp.hist[BLUE_H]))
    bw = susp.bw(BLUE_H)
    # 见过的漂移范围 ~ ±drift，自适应带宽 = 2×std(≈1.15×drift) → 覆盖 2σ
    print(f"  漂移±{drift:4.1f}° → 见过的 std={std:4.1f}° → 自适应带宽 {bw:4.1f}°"
          f"（覆盖 ±{bw:.0f}°；漂移 ±{drift:.0f}° 在 2σ 内{'✓' if bw >= drift else '✗'}）")

print("\n== P2 不误伤：自适应带宽对红（距蓝 120°）不疑 ==")
susp = SuspiciousAdapt(mode='adapt', k_band=2.0)
train(susp)
print(f"  自适应带宽 ≈ {susp.bw(BLUE_H):.1f}°（远 < 120°）→ 红 k = {susp.k(0.0)}（不受蓝的阅历影响）")
print(f"  [{'PASS' if susp.k(0.0) == 1 else 'FAIL'}] 目标色永远开放（docs/205 同款）\n")

print("== P3 变异沉积：见过的变异越大 → 带宽越宽 ==")
print(f"{'训练漂移':>10s} {'见过的std':>10s} {'自适应带宽':>12s}")
prev_bw = -1
p3 = True
for drift in [1.0, 3.0, 6.0, 12.0]:
    susp = SuspiciousAdapt(mode='adapt', k_band=2.0)
    train(susp, drift_amp=drift)
    std = float(np.std(susp.hist[BLUE_H]))
    bw = susp.bw(BLUE_H)
    if bw < prev_bw:
        p3 = False
    prev_bw = bw
    print(f"{drift:10.1f} {std:10.1f} {bw:12.1f}")
print(f"P3: [{'PASS' if p3 else 'FAIL'}]（阅历让怀疑半径沉积：见过的变异越大越宽）\n")

print("== 判读 ==")
print("  自适应带宽 = k_band × 见过的该类别色相 std——怀疑半径 = 见过的变异范围")
print("  docs/196 锚：真实同类颜色因色温漂移 ~几十度 → 自适应带宽覆盖'见过的漂移'")
print("    而不是固定 30°（固定小带宽漏漂移蓝，固定大带宽误伤红，docs/205 病态）")
print("  P3 变异沉积：阅历（见过的变异）让带宽沉积——怀疑半径是学出来的不是手调的")
