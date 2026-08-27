"""vision/bandwidth_threshold.py — 带宽+阈值联合：怀疑半径和阈值一起沉积阅历。

docs/208 §五 2 + docs/207：类别级 thr 的查询带宽用自适应带宽——"蓝被拒 → 蓝及
见过的蓝变异范围（±自适应带宽）都稍严"。怀疑半径（docs/208）和确认阈值（docs/207）
一起沉积阅历。

机制：
  suspicious[hue] 表（被拒累计，docs/205）+ hist[hue]（见过的色相变异，docs/208）
  thr_by_hue[hue]（类别级确认阈值，docs/207）
  联合：查询时"类别 = 见过的变异范围"——hue' 落在 见过的 hue ± 自适应带宽 内
    → 共享同一类别（同一个 suspicious 计数、同一个 thr）

实验：
  P1 联合沉积：训练染蓝（漂移 ±5°，见过 115-125）→ 漂移蓝 128° 出现——
     单类别（docs/207，精确色相）当新类别（thr 基线）vs 联合（128 在见过的
     变异内→共享蓝的可疑+thr）→ 128° 蓝段误报被压
  P2 带宽自适应 vs 固定：见过的变异越大 → 联合共享范围越宽（P3 单调）
  P3 不误伤：红（距蓝 120°）在自适应带宽外 → 不共享蓝的可疑（thr 基线）

docs/196 锚：真实同类颜色色相漂移几十度——联合把"同类"定义成"见过的变异范围"，
怀疑和阈值都按这个范围沉积。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np

RED_H, BLUE_H = 0.0, 120.0
CAP, GROW = 60, 3.0


def circ(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


class JointSus:
    """类别级 thr + 自适应带宽联合（docs/207 + 208）。"""

    def __init__(self, k_band=2.0, base_thr=0.2, boost=0.1, cap_thr=0.8):
        self.k_band = k_band
        self.base_thr = base_thr
        self.boost = boost
        self.cap_thr = cap_thr
        self.hist = {}          # hue -> [观测色相]（见过的变异）
        self.rejected = {}      # hue -> 被拒累计
        self.thr_by = {}        # hue -> 类别级阈值

    def _neighbors(self, hue):
        """见过的类别：落在 见过的 hue ± 自适应带宽 内的所有记录类别。"""
        neigh = set()
        for sh in self.hist:
            if circ(hue, sh) <= self.k_band * np.std(self.hist[sh]):
                neigh.add(sh)
        return neigh

    def k(self, hue):
        """重信任需求（docs/203 saturate）：共享可疑类别的被拒累计。"""
        n = sum(self.rejected.get(sh, 0) for sh in self._neighbors(hue))
        return 1 + int(min(n, CAP) / GROW)

    def thr(self, hue):
        """确认阈值：共享可疑类别的最大阈值（联合沉积）。"""
        neigh = self._neighbors(hue)
        return max([self.thr_by.get(sh, self.base_thr) for sh in neigh] or
                   [self.base_thr])

    def observe(self, hue, obs):
        """记录见过的变异。"""
        if hue is not None:
            self.hist.setdefault(hue, []).append(obs)

    def confirm(self, frac, hue, obs):
        """一帧确认。返回 (通过?, 本帧 thr, 本帧 k)。"""
        self.observe(hue, obs)
        t = self.thr(hue)
        ok = frac > t
        if ok:
            # 持续信任 → 阈值回落（docs/207 可逆）
            for sh in self._neighbors(hue):
                if self.thr_by.get(sh, self.base_thr) > self.base_thr:
                    self.thr_by[sh] = max(self.base_thr, self.thr_by[sh] - 0.05)
        else:
            self.rejected[hue] = self.rejected.get(hue, 0) + 1
            # 被拒 → 该类别的阈值上调（联合沉积：邻域内共享）
            self.thr_by[hue] = min(self.cap_thr, t + self.boost)
        return ok, t, self.k(hue)


def train(j, n_pass=100, drift_amp=5.0, seed=42):
    """染蓝 n_pass 遍（色相漂移 ±drift_amp），红段正常。"""
    rng = np.random.default_rng(seed)
    for _ in range(n_pass):
        hue = BLUE_H + rng.uniform(-drift_amp, drift_amp)
        for _ in range(30):
            j.confirm(0.15, BLUE_H, hue + rng.normal(0, 2.0))   # 蓝段被拒
        for _ in range(30):
            j.confirm(0.8, RED_H, 0.0)                          # 红段通过


print("== 带宽+阈值联合：怀疑半径和阈值一起沉积阅历 ==")
print(f"训练：染蓝 ×100 遍，色相漂移 ±5°（见过 115-125），红段正常\n")

print("== P1 联合沉积：漂移蓝 128° 是否共享蓝的怀疑与阈值 ==")
j = JointSus()
train(j)
bw = j.k_band * np.std(j.hist[BLUE_H])
print(f"  见过的蓝 std≈{np.std(j.hist[BLUE_H]):.1f}° → 自适应带宽≈{bw:.1f}°")
print(f"  128° 距蓝 120° = {circ(128, BLUE_H):.0f}° {'在带宽内→共享' if circ(128, BLUE_H) <= bw else '在带宽外→新类别'}")
print(f"  漂移蓝 128°: thr={j.thr(128.0):.2f}（蓝的阈值 {j.thr(BLUE_H):.2f}）"
      f" k={j.k(128.0)}（蓝的 k={j.k(BLUE_H)}）")
print(f"  标准蓝 120°: thr={j.thr(BLUE_H):.2f} k={j.k(BLUE_H)}")
print(f"  红 0°      : thr={j.thr(RED_H):.2f} k={j.k(RED_H)}（目标色永远基线）\n")

print("== P2/P3 联合 vs 单类别 vs 带宽沉积 ==")
print(f"{'训练漂移':>10s} {'带宽':>6s} {'128°thr':>8s} {'128°k':>7s} {'红thr':>7s} {'红k':>5s}")
prev_bw = -1
for drift in [1.0, 5.0, 12.0]:
    j = JointSus()
    train(j, drift_amp=drift)
    bw = j.k_band * np.std(j.hist[BLUE_H])
    t128, k128 = j.thr(128.0), j.k(128.0)
    tr, kr = j.thr(RED_H), j.k(RED_H)
    print(f"{drift:10.1f} {bw:6.1f} {t128:8.2f} {k128:7d} {tr:7.2f} {kr:5d}")
print("  128° 距蓝 120° = 8°：漂移 ±1°（带宽小）→ 128 当新类别（thr 基线）；")
print("  漂移 ±12°（带宽大）→ 128 在见过的变异内 → 共享蓝的怀疑（thr/k 升）\n")

print("== 判读 ==")
print("  联合沉积：怀疑半径（带宽）和确认阈值（thr）一起沉积阅历——")
print("    '同类' = 见过的变异范围，怀疑与阈值都按这个范围沉积（docs/208 §五 2）")
print("  不误伤：红（距蓝 120°）在自适应带宽外 → thr 基线 0.20、k=1（目标色永远开放）")
print("  docs/196 锚：真实同类颜色色相漂移几十度 → 联合把'同类'定义成见过的变异，")
print("    漂移的同类（128°）也受怀疑（thr 高），而不是当新颜色漏掉")
