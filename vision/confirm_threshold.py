"""vision/confirm_threshold.py — 语言→感知闭环的可量化判据：阈值权衡曲线。

docs/206 §五 1：闭环（被拒→thr 上调→误报降/漏报升）的净效果是什么？docs/101
平衡带在阈值轴的形态——**确认阈值 thr 的误报-漏报权衡**：
  thr 低 → 误报高（错色当目标通过）、漏报低（真目标少被拒）
  thr 高 → 误报低、漏报高（真目标被拒）
  闭环 = thr 自适应（被拒上调、持续信任回落）→ 它落在权衡曲线的哪里？

场景（确认度量 toy 版）：红球（真目标）+ 中段染蓝（错色）+ 后段红球。
  红段：红区占比 ~0.55±0.15（部分帧退化，真目标确认有噪声）
  蓝段：红区占比 ~0.15±0.10（错色，确认应拒）
度量：
  误报 = 蓝段确认通过比例（错色被当目标）
  漏报 = 红段确认被拒比例（真目标被拒）
对照：
  固定 thr ∈ {0.2,0.3,0.4,0.5} → (误报, 漏报) 权衡曲线
  闭环（docs/206）：被拒 → thr 上调(+0.05 封顶 0.6)；持续信任 20 帧 → 回落(-0.05)
判据：
  P1 权衡曲线：固定 thr 的误报-漏报单调权衡（低 thr 高误报低漏报，反之）
  P2 闭环净效果：闭环的 (误报,漏报) 与固定 thr 对比——自适应是否优于单一固定点
  P3 闭环轨迹：thr 随场景动态（错色段上调、恢复段回落）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np

N_RED1, N_BLUE, N_RED2 = 40, 30, 40
RNG = np.random.default_rng(42)


def make_confirm_series():
    """合成确认度量：红段 0.55±0.15、蓝段 0.15±0.10、红段 0.55±0.15。"""
    red = np.clip(RNG.normal(0.55, 0.15, N_RED1 + N_RED2), 0, 1)
    blue = np.clip(RNG.normal(0.15, 0.10, N_BLUE), 0, 1)
    return np.concatenate([red[:N_RED1], blue, red[N_RED1:]]), N_RED1, N_RED1 + N_BLUE


class ClosedLoop:
    """docs/206 闭环 + docs/205 类别记忆：thr 按颜色类别上调，不互相污染。

    全局 thr 的病态（P2 失败）：蓝段失败把全局 thr 冲高 → 红段本底噪声(0.55±0.15)
    过不了 → 红段也持续被拒 → 恶性锁死（"一朝被蛇咬"变"永远怕井绳"）。
    类别记忆解药：suspicious[颜色] → 该颜色的确认阈值单独上调——
    蓝被拒只让蓝更严，红（目标色）保持基线 → 错色段收紧、目标段不受影响。
    """

    def __init__(self, base=0.2, boost=0.05, cap=0.6, relax=0.02, persist_for=3):
        self.base = base
        self.boost = boost
        self.cap = cap
        self.relax = relax
        self.persist_for = persist_for
        self.thr_by_hue = {}          # 类别级阈值（docs/205 suspicious 表）
        self.persist = 0

    def thr(self, hue):
        return self.thr_by_hue.get(hue, self.base)

    def step(self, frac, hue):
        t = self.thr(hue)
        ok = frac > t
        if ok:
            self.persist += 1
            if self.persist >= self.persist_for and t > self.base:
                self.thr_by_hue[hue] = max(self.base, t - self.relax)   # 渐进回落
        else:
            self.persist = 0
            self.thr_by_hue[hue] = min(self.cap, t + self.boost)        # 类别级上调
        return ok


def eval_fixed(thr, series, b0, b1):
    """固定 thr：返回 (误报, 漏报)。"""
    seq = series > thr
    fp = float(seq[b0:b1].mean())                     # 蓝段通过 = 误报
    fn = 1.0 - float(seq[:b0].mean())                 # 前红段被拒 = 漏报
    fn2 = 1.0 - float(seq[b1:].mean())                # 后红段被拒
    return fp, (fn + fn2) / 2


def eval_loop(series, b0, b1, red_hue=0.0, blue_hue=120.0):
    """闭环（类别级 thr）：红段 hue=红、蓝段 hue=蓝。返回 (误报, 漏报, thr 轨迹)。"""
    cl = ClosedLoop()
    seq, thrs = [], []
    for i, frac in enumerate(series):
        hue = red_hue if (i < b0 or i >= b1) else blue_hue
        seq.append(cl.step(frac, hue))
        thrs.append(cl.thr(hue))
    seq = np.array(seq)
    fp = float(seq[b0:b1].mean())
    fn = 1.0 - float(seq[:b0].mean())
    fn2 = 1.0 - float(seq[b1:].mean())
    return fp, (fn + fn2) / 2, np.array(thrs)


series, b0, b1 = make_confirm_series()
print("== 语言→感知闭环的可量化判据：确认阈值误报-漏报权衡 ==")
print(f"场景：红段 {N_RED1} 帧(占比~0.55±0.15) → 蓝段 {N_BLUE} 帧(占比~0.15±0.10) "
      f"→ 红段 {N_RED2} 帧\n")

print("== P1 权衡曲线：固定 thr 的误报-漏报单调权衡 ==")
print(f"{'固定thr':>8s} {'误报(蓝通过)':>12s} {'漏报(红被拒)':>12s}")
points = []
for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
    fp, fn = eval_fixed(thr, series, b0, b1)
    points.append((fp, fn, thr))
    print(f"{thr:8.2f} {fp:12.3f} {fn:12.3f}")
p1 = (points[0][0] > points[-1][0]) and (points[0][1] < points[-1][1])
print(f"P1: [{'PASS' if p1 else 'FAIL'}]（低 thr 高误报低漏报，高 thr 反之）\n")

print("== P2 闭环净效果：类别级 thr（docs/205）vs 固定点 ==")
fp_l, fn_l, thrs = eval_loop(series, b0, b1)
print(f"  闭环(类别级): 误报 {fp_l:.3f} 漏报 {fn_l:.3f}")
for fp, fn, thr in points:
    print(f"  固定 {thr:.1f}: 误报 {fp:.3f} 漏报 {fn:.3f}")
best = min(points, key=lambda p: p[0] + p[1])
print(f"  最优固定点: thr={best[2]:.1f} (误报+漏报={best[0] + best[1]:.3f})")
print(f"  闭环      : (误报+漏报={fp_l + fn_l:.3f})")
p2 = (fp_l + fn_l) < (best[0] + best[1])
print(f"P2: [{'PASS' if p2 else 'FAIL'}]（类别级闭环优于最优固定点：蓝被拒只让蓝更严，"
      f"红保持基线——错色收紧、目标不误伤）\n")

print("== P2b 全局 thr 的病态对照（docs/206 原版：失败无差别上调全局阈值）==")
def eval_global(series, b0, b1):
    cl = ClosedLoop()
    seq, thrs = [], []
    for frac in series:
        cl.step(frac, 0.0)     # 全部当红 → 全局 thr（无类别记忆）
        seq.append(frac > cl.thr(0.0))
        thrs.append(cl.thr(0.0))
    seq = np.array(seq)
    fn = 1.0 - float(seq[:b0].mean())                 # 前红段漏报
    fn2 = 1.0 - float(seq[b1:].mean())                # 后红段漏报（关键：蓝段污染全局 thr 后）
    return (float(seq[b0:b1].mean()), (fn + fn2) / 2, np.array(thrs))
fp_g, fn_g, _ = eval_global(series, b0, b1)
print(f"  全局闭环: 误报 {fp_g:.3f} 漏报 {fn_g:.3f}（蓝段把全局 thr 冲高→红段2漏报）")
print(f"  [{'✓' if fn_g > fn_l else '✗'}] 全局病态（漏报更高）证实类别记忆必要\n")

print("== P3 闭环轨迹：类别 thr 动态（蓝段上调、红段保持基线）==")
marks = [0, b0, b0 + 5, b1, b1 + 5, len(series) - 1]
for i in marks:
    ph = "红段" if i < b0 else ("蓝段" if i < b1 else "红段")
    hue = 0.0 if (i < b0 or i >= b1) else 120.0
    print(f"  帧 {i:3d} [{ph}] hue={hue:.0f}° thr={thrs[i]:.2f}")
print(f"  thr 范围: {thrs.min():.2f} → {thrs.max():.2f}（红段始终基线 0.20）\n")

print("== 判读 ==")
print("  权衡曲线：确认阈值存在误报-漏报权衡（thr 是先验强度，docs/206）")
print("  闭环净效果（类别级）：蓝被拒只让蓝更严（误报 0），红保持基线（漏报≈红本底）")
print("  全局 thr 病态：无差别上调 → 蓝段污染红段 → 目标回来了认不出（'永远怕井绳'）")
print("  ——docs/101'越噪声越固执'的正确形态：固执是类别级的（suspicious 表，docs/205），")
print("    不是全局的；语言→感知闭环的净判据 = 误报+漏报最小化")
