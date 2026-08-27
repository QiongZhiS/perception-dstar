"""vision/color_robust.py — 光照变化下颜色确认的鲁棒性（docs/195 机制测完的最后一环）。

真实世界光照（docs/133 实验② 昼夜）：亮度变化 + 色温偏移。颜色确认靠 HSV 色相——
H 对亮度相对鲁棒（HSV 分离 V），但对色温漂移敏感（白炽灯偏黄 → 红偏橙 H 下降）。

两维扰动（红球，任务层"找红球"）：
  A 亮度渐变 200→30（暗化）：H 应稳定 → 确认不退化
  B 色温偏移（乘黄增益 R↑B↓，模拟黄昏/白炽灯）：H 漂移 → 确认退化程度
  C 两者叠加（暗+黄）：最坏场景

度量：确认通过率（目标区域 H 落红区间 [170,180]∪[0,10]）+ 平均 H 漂移。
判据：亮度变化确认通过率保持高（>0.9）；色温偏移通过率下降（量化退化）；
     docs/101：世界越噪声自我越该固执——确认阈值是否该随光照自适应（结论）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2

W, H, N = 640, 360, 90
RED = (170, 180, 10)    # 红（环形：170-180 和 0-10）


def make_scene(mode):
    """红球匀速直线。mode: 'bright'/'dark'/'yellow'/'dark+yellow'。"""
    frames = []
    r0 = np.array([80.0, 180.0])
    rv = np.array([4.0, 0.6])
    rng = np.random.default_rng(42)
    for i in range(N):
        t = i / N
        base = 160
        if mode == 'dark':
            base = int(200 - 170 * t)          # 200→30 暗化
        img = np.full((H, W, 3), base, np.uint8)
        img = np.clip(img.astype(np.int16) +
                      rng.normal(0, 3 + 8 * t, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * i
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, (0, 0, 200), -1)
        if mode in ('yellow', 'dark+yellow'):
            # 色温偏移：乘黄增益（R 保持、B 压低、G 略提）——模拟白炽灯/黄昏
            img = np.stack([
                img[:, :, 0],
                np.clip(img[:, :, 1] * 1.1, 0, 255).astype(np.uint8),
                np.clip(img[:, :, 2] * 0.5, 0, 255).astype(np.uint8)], axis=-1)
        if mode == 'dark+yellow':
            base = int(200 - 170 * t)
            img = np.full((H, W, 3), base, np.uint8)
            img = np.clip(img.astype(np.int16) +
                          rng.normal(0, 3 + 8 * t, img.shape).astype(np.int16),
                          0, 255).astype(np.uint8)
            cv2.circle(img, (int(rc[0]), int(rc[1])), 18, (0, 0, 200), -1)
            img = np.stack([
                img[:, :, 0],
                np.clip(img[:, :, 1] * 1.1, 0, 255).astype(np.uint8),
                np.clip(img[:, :, 2] * 0.5, 0, 255).astype(np.uint8)], axis=-1)
        frames.append(img)
    return frames


def confirm_rate(frames):
    """红球真值位置 → 目标区域"彩色像素（S 高）"H 落红区间的通过率 + 平均 H。
    灰色像素 S≈0 的 H 无意义（=0），不计入确认（docs/192：颜色=区分，需饱和）。"""
    ok = 0
    hues = []
    for i, fr in enumerate(frames):
        rc = (80 + 4.0 * i, 180 + 0.6 * i)      # 真值（运动学：每帧 +vx,+vy）
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h, s = hsv[:, :, 0], hsv[:, :, 1]
        r = 25
        cx, cy = int(rc[0]), int(rc[1])
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue
        rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        colored = rs > 60                        # 只统计彩色像素（去灰）
        if colored.sum() == 0:
            continue
        frac = float(((rh >= RED[0]) | (rh <= RED[2]))[colored].mean())
        vals = rh[colored]
        if vals.size:
            hues.append(float(np.median(vals)))
        ok += frac > 0.2
    return ok / max(len(frames), 1), (float(np.median(hues)) if hues else float('nan'))


print("== 光照变化下颜色确认鲁棒性（红球，任务层'找红球'）==")
print(f"{'扰动':14s} {'确认通过率':>10s} {'目标区H中位':>10s}")
results = {}
for mode in ['bright', 'dark', 'yellow', 'dark+yellow']:
    frames = make_scene(mode)
    rate, hmed = confirm_rate(frames)
    results[mode] = (rate, hmed)
    print(f"{mode:14s} {rate:10.3f} {hmed:10.1f}")

print("\n判据/结论：")
print("  亮度渐变：H 应稳定（HSV 分离 V）→ 通过率保持高")
print("  色温偏移（黄）：红→橙 H 漂移 → 通过率下降（量化退化）")
print("  dark+yellow：最坏场景 → docs/101 越噪声越固执：确认阈值或需随光照自适应")
