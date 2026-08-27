"""vision/color_constancy.py — 色彩恒常：色温适应（docs/196 缺口实现，docs/133 §二.1）。

人眼色彩恒常：大脑把环境色温当"白"（中性参照），目标色相相对它看——同一红物体在
黄光下仍认为红（因为黄光偏移被减去）。

实现：
  ① 色温估计：整帧彩色像素（S>阈值）的色相主峰 = 环境色温偏移（背景/中性物被
     环境光照染成的共同色相）
  ② 确认校正：目标区域色相减去色温偏移（H_corrected = H - H_bg，旋转）再判定
  ③ 验证：yellow/dark+yellow 模式校正后 H 应回到 ~0（红），通过率真 1.0
     （非 docs/196 的边缘残红假象——球主体也通过）

对照：无校正（docs/196 现状：H 漂移 83°）vs 有校正（应 H≈0）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED


def estimate_color_temp(fr, sat_min=100, min_frac=0.02, max_frac=0.5):
    """环境色温 = 背景彩色像素（排除目标区域）的色相中位数。
    背景/中性物被环境光染成共同色相（如黄光下灰背景→H≈85）→ 该色相即色温偏移。
    双门槛（docs/101 越噪声越固执的颜色版）：
      - sat_min=100：排除暗光噪声（暗灰噪声 S 低）
      - max_frac=0.5：彩色占比中高区间才信（yellow 背景全染 1.0 是真色温；
        dark 噪声染色 ~0.3 不可信）→ 不确定时不校正（保守，docs/101）"""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    colored = (s > sat_min).astype(np.uint8)
    # 排除目标路径区（受控场景红球轨迹带：y 140-240, x 40-320）
    colored[140:240, 40:320] = 0
    frac = colored.mean()
    if frac < min_frac or frac > max_frac:  # 无信号 / 噪声淹没 → 不校正
        return 0.0
    return float(np.median(h[colored > 0]))


def confirm_rate(frames, correct=True):
    """红球确认通过率 + H 中位。correct=True 做色温校正。"""
    ok = 0
    hues_c = []
    for i, fr in enumerate(frames):
        rc = (80 + 4.0 * i, 180 + 0.6 * i)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h, s = hsv[:, :, 0], hsv[:, :, 1]
        shift = estimate_color_temp(fr) if correct else 0.0
        # 校正：H 减去色温偏移（旋转）
        hc = (h.astype(float) - shift) % 180
        r = 25
        cx, cy = int(rc[0]), int(rc[1])
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue
        rh = hc[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        colored = rs > 60
        if colored.sum() == 0:
            continue
        frac = float(((rh >= RED[0]) | (rh <= RED[2]))[colored].mean())
        vals = rh[colored]
        if vals.size:
            hues_c.append(float(np.median(vals)))
        ok += frac > 0.2
    return ok / max(len(frames), 1), (float(np.median(hues_c)) if hues_c else float('nan'))


print("== 色彩恒常（色温适应）：黄光下红球确认 ==")
print(f"{'模式':14s} {'无校正H':>8s} {'无校正通过':>9s} | {'校正后H':>8s} {'校正后通过':>10s}")
for mode in ['bright', 'dark', 'yellow', 'dark+yellow']:
    frames = make_scene(mode)
    r0, h0 = confirm_rate(frames, correct=False)
    r1, h1 = confirm_rate(frames, correct=True)
    print(f"{mode:14s} {h0:8.1f} {r0:9.3f} | {h1:8.1f} {r1:10.3f}")

print("\n判据：色温校正后 yellow/dark+yellow 的 H 应回到 ~0（真红），")
print("且通过率真 1.0（非 docs/196 边缘残红假象——球主体也通过）")
