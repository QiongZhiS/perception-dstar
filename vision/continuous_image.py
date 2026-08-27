"""vision/continuous_image.py — 连续 thr 回图像版：非纯蓝真实漂移下的距离加权。

docs/214 §五 1：把连续 thr（docs/214）装回 docs/211 的真实图像场景——非纯蓝在
多光照下真实漂移，验证连续阈值的距离加权在真实色相分布下工作。

实验：
  训练：非纯蓝在多种暖光强度（0/0.3/0.5/0.7）下染蓝 ×30 遍
    → hist[蓝] 沉积真实漂移（docs/211）→ 自适应带宽
  测试：
    P1 真实分布：多光照下蓝的观测色相分布（真实漂移量级）
    P2 连续 thr vs 二值：距标准蓝不同距离的 thr——连续在带宽边缘平滑
       （真实像素色相的连续距离），二值阶跃
    P3 不误伤：红（目标色）永远基线
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED
from continuous_threshold import ContinuousThr, BASE_THR
from bandwidth_threshold import JointSus, circ
from joint_image import make_interrupted, blue_hue, red_frac

TRAIN_SEG = (30, 60)
RED_H, BLUE_H = 0.0, 120.0
WARMS = [0.0, 0.3, 0.5, 0.7]   # 多光照：不同暖光强度


def train_image_cont(sus, n_pass=30):
    """多暖光染蓝训练（非纯蓝真实漂移沉积）。"""
    obs_cache = {}
    for w in WARMS:
        frames = make_interrupted('bright', warm=w)
        hs = [blue_hue(frames[i], i) for i in range(*TRAIN_SEG)]
        obs_cache[w] = [h for h in hs if h is not None]
    for _ in range(n_pass):
        for w in WARMS:
            for h in obs_cache[w]:
                sus.confirm(0.15, BLUE_H, h)
            for _ in range(TRAIN_SEG[0]):
                sus.confirm(0.8, RED_H, 0.0)


print("== 连续 thr 回图像版：非纯蓝真实漂移下的距离加权 ==")
print("训练：非纯蓝 × 4 种暖光强度 × 30 遍（真实漂移沉积）\n")

print("== P1 真实分布：不同暖光下染蓝的观测色相 ==")
obs = {}
for w in WARMS:
    frames = make_interrupted('bright', warm=w)
    hs = [blue_hue(frames[i], i) for i in range(*TRAIN_SEG)]
    hs = [h for h in hs if h is not None]
    obs[w] = hs
    print(f"  暖光 {w:.1f}: 蓝观测色相 mean={np.mean(hs):.1f}°")
all_obs = [h for hs in obs.values() for h in hs]
print(f"  总体: mean={np.mean(all_obs):.1f}° std={np.std(all_obs):.1f}° "
      f"范围 {min(all_obs):.0f}-{max(all_obs):.0f}°\n")

ct = ContinuousThr(decay='linear')
train_image_cont(ct)
j = JointSus()
train_image_cont(j)
bw = ct.k_band * np.std(ct.hist[BLUE_H])
print(f"  见过的蓝 std≈{np.std(ct.hist[BLUE_H]):.1f}° → 自适应带宽≈{bw:.1f}°\n")

print("== P2 连续 thr vs 二值（距标准蓝的距离）==")
print(f"{'距蓝':>6s} {'二值thr':>8s} {'连续thr':>8s}")
for d in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 120.0]:
    hue = (BLUE_H + d) % 180.0
    tj = j.thr(hue)
    tc = ct.thr(hue)
    print(f"{d:6.0f} {tj:8.2f} {tc:8.2f}")

print("\n== P3 不误伤：红（目标色）永远基线 ==")
print(f"  二值 thr={j.thr(RED_H):.2f} 连续 thr={ct.thr(RED_H):.2f}（基线 {BASE_THR}）")

print("\n== 判读 ==")
print("  真实图像下：非纯蓝在暖光下漂移（8°+ 量级，docs/211），std 沉积进带宽")
print("  连续 thr：距蓝距离加权——带宽边缘平滑（真实像素色相的连续距离）")
print("  二值 thr：带宽内全量共享（边缘阶跃）——docs/210 的近似")
print("  不误伤：红（120°）永远基线（目标色开放，docs/205）")
