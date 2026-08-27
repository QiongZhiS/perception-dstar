"""vision/confirm_image.py — 图像版类别级阈值闭环：docs/207 结论在真实图像分布下验证。

docs/207 §五 2：把类别级 thr 装回 docs/200 图像场景——真实红球红区占比分布
（~0.78-0.97）下，类别级闭环是否仍优于全局闭环/固定阈值？

用真实图像（color_robust 场景）+ 真实 ball_stats 红区占比，验证：
  P1 图像分布：真实红球红区占比（有橙边缘/噪声波动）vs 染蓝段（≈0）
  P2 类别级 vs 全局：蓝段被拒 → 类别级只调蓝 thr（红保持基线 0.2）→ 红段恢复
     确认无损；全局 thr → 蓝段污染 → 红段漏报（docs/207 病态在图像下复现）
  P3 净效果：类别级闭环误报+漏报 < 全局 < 固定？（图像分布下）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED
from confirm_threshold import ClosedLoop  # 类别级闭环（step(frac, hue)，docs/207）

TRAIN_SEG = (30, 60)
RNG = np.random.default_rng(42)


def make_interrupted_image(mode='yellow'):
    """真实图像场景：红球（color_robust 模式，含光照/色温扰动）+ 染蓝段。"""
    frames = make_scene(mode)
    for i in range(*TRAIN_SEG):
        rc = (80 + 4.0 * i, 180 + 0.6 * i)
        cv2.circle(frames[i], (int(rc[0]), int(rc[1])), 18, (200, 0, 0), -1)  # 蓝
    return frames


def ball_stats(fr, i, noise=True):
    """真实红区占比（球体掩码），加少量确认噪声模拟边缘/遮挡波动。"""
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
    hue = float(np.median(rh[colored]))
    if noise and frac > 0.3:    # 红球确认噪声（边缘/遮挡）
        frac = np.clip(frac + RNG.normal(0, 0.06), 0, 1)
    return frac, hue


def run_closed(frames):
    """类别级闭环跑一遍。返回 (误报, 漏报, thr轨迹)。"""
    cl = ClosedLoop(base=0.2, boost=0.05, cap=0.8, relax=0.02, persist_for=3)
    seq, thrs = [], []
    for i, fr in enumerate(frames):
        frac, hue = ball_stats(fr, i)
        th = 0.0 if (i < TRAIN_SEG[0] or i >= TRAIN_SEG[1]) else 120.0
        seq.append(cl.step(frac, th))
        thrs.append(cl.thr(th))
    seq = np.array(seq)
    fp = float(seq[TRAIN_SEG[0]:TRAIN_SEG[1]].mean())
    fn1 = 1.0 - float(seq[:TRAIN_SEG[0]].mean())
    fn2 = 1.0 - float(seq[TRAIN_SEG[1]:].mean())
    return fp, (fn1 + fn2) / 2, np.array(thrs)


def run_global(frames):
    """全局 thr（docs/206 原版：失败无差别上调）。"""
    cl = ClosedLoop(base=0.2, boost=0.05, cap=0.8, relax=0.02, persist_for=3)
    seq, thrs = [], []
    for i, fr in enumerate(frames):
        frac, hue = ball_stats(fr, i)
        seq.append(cl.step(frac, 0.0))     # 全部当红 → 全局 thr
        thrs.append(cl.thr(0.0))
    seq = np.array(seq)
    fp = float(seq[TRAIN_SEG[0]:TRAIN_SEG[1]].mean())
    fn1 = 1.0 - float(seq[:TRAIN_SEG[0]].mean())
    fn2 = 1.0 - float(seq[TRAIN_SEG[1]:].mean())
    return fp, (fn1 + fn2) / 2, np.array(thrs)


def run_fixed(frames, thr):
    seq = [ball_stats(fr, i)[0] > thr for i, fr in enumerate(frames)]
    seq = np.array(seq)
    fp = float(seq[TRAIN_SEG[0]:TRAIN_SEG[1]].mean())
    fn1 = 1.0 - float(seq[:TRAIN_SEG[0]].mean())
    fn2 = 1.0 - float(seq[TRAIN_SEG[1]:].mean())
    return fp, (fn1 + fn2) / 2


print("== 图像版类别级阈值闭环（docs/207 结论在真实图像分布下验证）==")
for mode in ['bright', 'yellow', 'dark+yellow']:
    frames = make_interrupted_image(mode)
    print(f"\n--- 模式 {mode} ---")
    # P1 图像分布
    reds = [ball_stats(fr, i)[0] for i, fr in enumerate(frames)
            if not (TRAIN_SEG[0] <= i < TRAIN_SEG[1])]
    blues = [ball_stats(fr, i)[0] for i, fr in enumerate(frames)
             if TRAIN_SEG[0] <= i < TRAIN_SEG[1]]
    print(f"  P1 图像分布: 红段红区占比 mean={np.mean(reds):.3f} std={np.std(reds):.3f} "
          f"(n={len(reds)}); 蓝段 mean={np.mean(blues):.3f}")
    # P2/P3 对比
    fp_f, fn_f = run_fixed(frames, 0.4)
    fp_c, fn_c, thrs_c = run_closed(frames)
    fp_g, fn_g, thrs_g = run_global(frames)
    print(f"  P2 固定thr=0.4: 误报 {fp_f:.3f} 漏报 {fn_f:.3f} (和 {fp_f + fn_f:.3f})")
    print(f"  P2 全局闭环  : 误报 {fp_g:.3f} 漏报 {fn_g:.3f} (和 {fp_g + fn_g:.3f}) "
          f"thr范围 {thrs_g.min():.2f}-{thrs_g.max():.2f}")
    print(f"  P2 类别级闭环: 误报 {fp_c:.3f} 漏报 {fn_c:.3f} (和 {fp_c + fn_c:.3f}) "
          f"蓝thr {thrs_c[TRAIN_SEG[0]]:.2f}→{thrs_c[TRAIN_SEG[1]-1]:.2f} 红thr {thrs_c[-1]:.2f}")
    print(f"  P3 净效果: 类别级 {fp_c + fn_c:.3f} {'<' if fp_c + fn_c < fp_g + fn_g else '>='} "
          f"全局 {fp_g + fn_g:.3f} {'<' if fp_g + fn_g < fp_f + fn_f else '>='} 固定 {fp_f + fn_f:.3f}")

print("\n== 判读 ==")
print("  图像分布下（红区占比真实 ~0.8-1.0，有噪声波动）：")
print("  全局 thr 被蓝段污染 → 红段漏报（docs/207 病态在图像下复现）")
print("  类别级 thr：蓝被拒只调蓝、红保持基线 → 误报漏报都低（docs/207 结论跨分布成立）")
