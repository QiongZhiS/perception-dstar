"""vision/experience_categories.py — 多类别阅历：suspicious[颜色] 表 + 泛化带宽。

docs/203 §五 1：阅历的多类别——被蓝骗过 100 次，对绿的态度是"类别特异"
（suspicious[蓝] 大 → 蓝谨慎、绿正常）还是"泛化到所有非目标色"（绿也谨慎）？
以及**泛化带宽**：被拒色相邻域（色相距离 < 带宽）内的颜色受影响吗？

机制（docs/200 Keep + 类别级记忆）：
  suspicious = {hue: 被拒累计}（跨遍不清零，docs/203 阅历）
  确认：目标色相 hue 的"可疑度" = 邻域带宽内所有 suspicious 之和
    n(hue) = Σ suspicious[sh] for |hue-sh|_circ ≤ bandwidth
  重信任延迟 K = 1 + min(n(hue), cap)/grow（类别级纪念加深）

实验（训练染蓝 100 遍 → 测试不同颜色）：
  P1 类别特异：训练染蓝 → 测试染蓝仍谨慎（延迟高）、染红（目标色）永远正常
  P2 泛化带宽：带宽 0°（精确类别）vs 30°（邻域）——染绿（距蓝 30°）是否受影响
  P3 阅历深度扩展带宽：被拒越多（100 vs 10 遍），有效泛化带宽是否变宽？
      （阅历深 → 疑邻：被一个类别骗过，对相近类别也提防）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED

RED_H, BLUE_H, GREEN_H, YELLOW_H = 0.0, 120.0, 90.0, 60.0
TRAIN_COLOR = BLUE_H        # 训练：染蓝
CAP, GROW = 60, 3.0


def circ(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def make_interrupted(hue, rej=(30, 60), total=N):
    """红球，[rej0,rej1) 帧目标被染成指定色相（BGR 近似：S 满、V=160 色相 h）。"""
    hsv = np.zeros((1, 1, 3), np.uint8)
    hsv[0, 0] = (int(hue), 255, 160)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    bgr = tuple(int(v) for v in bgr)
    r0 = np.array([80.0, 180.0]); rv = np.array([4.0, 0.6])
    rng = np.random.default_rng(42)
    frames = []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * i
        color = bgr if rej[0] <= i < rej[1] else (0, 0, 200)
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, color, -1)
        frames.append(img)
    return frames


def ball_stats(fr, i):
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


class KeepCat:
    """docs/200 Keep + 类别级记忆（suspicious 表）+ 泛化带宽。"""

    def __init__(self, bandwidth=0.0, thr=0.2, cap=CAP, grow=GROW):
        self.bandwidth = bandwidth
        self.thr = thr
        self.cap, self.grow = cap, grow
        self.suspicious = {}          # hue -> 被拒累计
        self.trusted = True
        self.consec = 0

    def k(self, hue):
        n = 0
        for sh, cnt in self.suspicious.items():
            if circ(hue, sh) <= self.bandwidth:
                n += cnt
        return 1 + int(min(n, self.cap) / self.grow)

    def step(self, frac_red, hue):
        ok = frac_red > self.thr
        if ok:
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k(hue):
                    self.trusted = True
        else:
            if hue is not None:
                self.suspicious[hue] = self.suspicious.get(hue, 0) + 1
            self.trusted = False
            self.consec = 0
        return self.trusted and ok


def pre_stats(frames):
    return [ball_stats(fr, i) for i, fr in enumerate(frames)]


def run_pass(stats, keep, seg=(30, 60)):
    seq = [keep.step(f, h) for f, h in stats]
    rest = seq[seg[1]:]
    rate = np.mean(rest) if rest else float('nan')
    delay = next((j + 1 for j, ok in enumerate(rest) if ok), None)
    return rate, delay


def train(keep, n_pass=100, color=TRAIN_COLOR):
    stats = pre_stats(make_interrupted(color))
    for _ in range(n_pass):
        run_pass(stats, keep)
    return keep.suspicious.get(color, 0)


print("== 多类别阅历：suspicious[颜色]表 + 泛化带宽 ==")
print(f"训练：染蓝({BLUE_H:.0f}°) × 100 遍 → 测各颜色重信任需求 k(hue)\n")
print("（k(hue) = 该颜色确认通过后重新信任所需连续帧；红=目标色基线 k=1）\n")

print("== P1 类别特异：训练染蓝 → 蓝 k 大(谨慎)、红 k=1(目标色正常) ==")
keep = KeepCat(bandwidth=0.0)
train(keep)
for label, hue in [("蓝(训练色)", BLUE_H), ("红(目标色)", RED_H), ("绿(未见过)", GREEN_H)]:
    print(f"  {label:12s} k = {keep.k(hue)} 帧（需连续通过）")
print("  （带宽 0°：只有蓝本身可疑；红/绿不受影响）\n")

print("== P2 泛化带宽：带宽 0/15/30/60° —— 距蓝不同距离的颜色 k ==")
print(f"{'带宽':>6s} {'蓝(120°)':>10s} {'绿(90°)':>10s} {'黄(60°)':>10s} {'红(0°)':>10s}")
for bw in [0.0, 15.0, 30.0, 60.0]:
    keep = KeepCat(bandwidth=bw)
    train(keep)
    row = [keep.k(h) for h in [BLUE_H, GREEN_H, YELLOW_H, RED_H]]
    print(f"{bw:6.1f} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d} {row[3]:>10d}")
print("  （带宽=阅历泛化半径：0°=只疑同类；30°+=相近颜色也受影响；红=目标色永不受影响）\n")

print("== P3 阅历深度：被拒越多，相近类别有效影响越强？（带宽 30°，遍数扫描）==")
print(f"{'训练遍数':>8s} {'蓝k':>6s} {'绿k':>6s} {'黄k':>6s} {'红k':>6s}")
for n_pass in [1, 10, 50, 100]:
    keep = KeepCat(bandwidth=30.0)
    train(keep, n_pass=n_pass)
    row = [keep.k(h) for h in [BLUE_H, GREEN_H, YELLOW_H, RED_H]]
    print(f"{n_pass:8d} {row[0]:>6d} {row[1]:>6d} {row[2]:>6d} {row[3]:>6d}")
print("  阅历深 → 疑邻：绿(距蓝30°)的 k 随蓝被拒次数增长——被骗越深，怀疑范围越大\n")

print("== 判读 ==")
print("  类别特异：suspicious 按色相记账 → 蓝被骗不影响红（目标色永远正常 k=1）")
print("  泛化带宽：阅历的怀疑半径（设计参数）——0°=只疑同类，30°+=疑相近色")
print("  阅历深度：带宽内相近色的 k ∝ 被拒累计——'一朝被蛇咬，十年怕井绳'的类别版")
print("  docs/101 越噪声越固执在类别轴：被拒类别(k 大)固执，目标色(k=1)永远开放")
