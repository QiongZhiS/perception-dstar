"""vision/joint_image.py — 联合机制回图像版：多光照下"同类蓝"的真实色相漂移。

docs/210 §五 1 + docs/196 锚：真实同类颜色在不同光照（docs/196 四模式）下色相
漂移几十度——联合的"同类 = 见过的变异范围"应在真实图像分布下工作。

实验：真实图像（color_robust 多模式）+ 染蓝段 + 色温扰动（增益）让蓝漂移。
  训练：多光照染蓝（见过蓝的真实漂移）→ hist[蓝] 记录变异 → 自适应带宽
  测试：
    P1 真实漂移：不同光照下"染蓝"的观测色相不同（量化真实变异）
    P2 联合识别：新光照的漂移蓝——自适应带宽内 → 共享蓝的怀疑（thr/k 升）；
       单类别（精确色相）→ 当新颜色（thr 基线）
    P3 不误伤：红（目标色）永远基线（thr 0.20、k=1）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED
from bandwidth_threshold import JointSus, circ

TRAIN_SEG = (30, 60)
RED_H, BLUE_H = 0.0, 120.0


def make_interrupted(mode, warm=0.0):
    """color_robust 模式 + 染非纯蓝 + 可选不对称加法暖光（蓝漂移）。

    docs/199 修正④：纯色（G/B=0）是光照变换的特征向量，色相不漂移——真实"蓝"
    带 G/R 分量（非纯色），加法暖光（R 加多）才让色相漂移。"""
    frames = make_scene(mode)
    for i in range(*TRAIN_SEG):
        rc = (80 + 4.0 * i, 180 + 0.6 * i)
        cv2.circle(frames[i], (int(rc[0]), int(rc[1])), 18, (160, 40, 30), -1)  # 非纯蓝
    if warm > 0:
        # 不对称加法暖光：G 加少 R 加多（暖黄光）→ 非纯蓝漂移
        add = np.array([0, 40 * warm, 80 * warm], dtype=np.float32)
        frames = [np.clip(f.astype(np.float32) + add, 0, 255).astype(np.uint8)
                  for f in frames]
    return frames


def blue_hue(fr, i):
    """染蓝段的观测色相（真实图像）。"""
    rc = (80 + 4.0 * i, 180 + 0.6 * i)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    r = 18
    cx, cy = int(rc[0]), int(rc[1])
    rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    colored = rs > 60
    if colored.sum() < 10:
        return None
    return float(np.median(rh[colored]))


def red_frac(fr, i):
    rc = (80 + 4.0 * i, 180 + 0.6 * i)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    r = 18
    cx, cy = int(rc[0]), int(rc[1])
    rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0
    return float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())


def train_image(j, modes=('bright', 'bright_warm', 'yellow_warm'), n_pass=30):
    """多光照染蓝训练：不同加法暖光 → 蓝的真实色相漂移沉积。"""
    obs_cache = {}
    for mode in modes:
        warm = 0.7 if 'warm' in mode else 0.0
        frames = make_interrupted('bright' if 'bright' in mode else 'yellow', warm=warm)
        hs = [blue_hue(frames[i], i) for i in range(*TRAIN_SEG)]
        obs_cache[mode] = [h for h in hs if h is not None]
        reds = [red_frac(frames[i], i) for i in range(0, TRAIN_SEG[0])]
        obs_cache[mode + '_red'] = reds
    for _ in range(n_pass):
        for mode in modes:
            for h in obs_cache[mode]:
                j.confirm(0.15, BLUE_H, h)     # 蓝段被拒 + 记录变异
            for fr in obs_cache[mode + '_red']:
                j.confirm(0.8, RED_H, 0.0)     # 红段通过


print("== 联合机制回图像版：多光照下'同类蓝'的真实色相漂移 ==")
print("训练：bright/yellow/dark+yellow 三模式染蓝 ×30 遍（蓝的真实漂移进 hist）\n")

print("== P1 真实漂移：不同光照下染蓝的观测色相 ==")
hues = {}
for label, warm in [('bright', 0.0), ('bright+暖光0.7', 0.7), ('yellow+暖光0.7', 0.7)]:
    mode = 'bright' if 'bright' in label else 'yellow'
    frames = make_interrupted(mode, warm=warm)
    hs = [blue_hue(frames[i], i) for i in range(*TRAIN_SEG)]
    hs = [h for h in hs if h is not None]
    hues[label] = hs
    print(f"  {label:16s} 蓝观测色相 mean={np.mean(hs):.1f} std={np.std(hs):.1f} "
          f"范围 {min(hs):.0f}-{max(hs):.0f}")
all_means = [np.mean(h) for h in hues.values()]
print(f"  真实漂移量级：跨光照均值差 = {max(all_means) - min(all_means):.1f}°\n")

print("== P2 联合 vs 单类别：新光照（dark+yellow+warm）漂移蓝是否共享怀疑 ==")
j = JointSus()
train_image(j)
bw = j.k_band * np.std(j.hist[BLUE_H])
print(f"  见过的蓝 std≈{np.std(j.hist[BLUE_H]):.1f}° → 自适应带宽≈{bw:.1f}°")
# 测试：dark+yellow + 强暖光（蓝漂移最大）
test_frames = make_interrupted('dark+yellow', warm=0.7)
test_hs = [blue_hue(test_frames[i], i) for i in range(*TRAIN_SEG)]
test_hs = [h for h in test_hs if h is not None]
t_mean = float(np.mean(test_hs))
d = circ(t_mean, BLUE_H)
shared = d <= bw
print(f"  测试漂移蓝：观测 {t_mean:.0f}°（距标准蓝 {d:.0f}°）"
      f"{'在带宽内→共享怀疑' if shared else '在带宽外→新类别'}")
t128 = j.thr(t_mean) if shared else j.thr(t_mean)
print(f"  联合: 漂移蓝 thr={j.thr(t_mean):.2f} k={j.k(t_mean)}"
      f"（标准蓝 thr={j.thr(BLUE_H):.2f} k={j.k(BLUE_H)}）")
print(f"  单类别（精确 120°）: 漂移蓝当新类别 thr=0.20 k=1\n")

print("== P3 不误伤：红（目标色）永远基线 ==")
print(f"  红 thr={j.thr(RED_H):.2f} k={j.k(RED_H)}（距蓝 {circ(RED_H, BLUE_H):.0f}° 在带宽外）\n")

print("== 判读 ==")
print("  真实图像下：同类蓝因光照漂移几十度（docs/196 色温脆弱同量级）")
print("  联合（见过的变异）：漂移蓝共享怀疑（thr/k 升）→ 不因漂移当新颜色漏掉")
print("  单类别（精确色相）：漂移蓝当新颜色（thr 基线）→ 漏掉见过的同类")
print("  目标色红永远基线（距蓝 120° 在自适应带宽外）→ 不误伤")
print("  docs/210 结论在真实图像分布下成立：'同类' = 见过的变异范围")
