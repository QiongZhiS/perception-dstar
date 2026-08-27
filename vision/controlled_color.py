"""vision/controlled_color.py — 受控彩色场景：已知运动彩色目标 + 灰度跟踪 + 颜色确认。

docs/195 §七 1：DAVIS 自然视频"跟踪好+颜色鲜明"稀缺，颜色确认增益上限需要受控场景。
合成：红球（任务目标）+ 蓝球（干扰）在灰度背景匀速直线运动，加噪声/光照渐变。
真值 = 运动学公式（无需标注，docs/183 方案 1 的合成版）。

验证（"找红球"）：
  A 灰度跟踪（无确认）：跟踪准吗（运动学真值）
  B 灰度跟踪 + 颜色确认（红）：红球通过确认，维持
  C 灰度跟踪 + 颜色确认（蓝，任务给错）：蓝球位置确认不通过（拒）→ 维持率低
  D 高对比增益：红球 vs 灰度背景（色相鲜明）→ 确认通过率应高（>car-turn 的 74/79）

判据：B 维持率 ≈ A（确认不损跟踪）+ C < B（错颜色拒）+ D 确认通过率 > 0.9
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from transduction import Transduction2D

W, H, FPS, N = 640, 360, 30, 120
RED_HUE = (170, 180)     # 红（HSV 环形区间）
BLUE_HUE = (100, 130)    # 蓝


def make_scene():
    """红球 + 蓝球匀速直线运动 + 灰度背景 + 噪声 + 光照渐变。返回 (帧列表, 真值)。"""
    frames = []
    truth = []            # [(red_cx, red_cy), (blue_cx, blue_cy)]
    rng = np.random.default_rng(42)
    # 运动学：匀速直线
    r0 = np.array([80.0, 180.0]); rv = np.array([4.0, 0.6])
    b0 = np.array([560.0, 90.0]); bv = np.array([-3.5, 0.8])
    for i in range(N):
        t = i / FPS
        # 光照渐变：亮度 100→160（明度变化——灰度跟踪仍应工作，颜色不受）
        light = 100 + 60 * t / (N / FPS)
        img = np.full((H, W, 3), int(light), np.uint8)
        # 灰度纹理（背景细节）
        img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * t
        bc = b0 + bv * t
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, (0, 0, 200), -1)      # 红球
        cv2.circle(img, (int(bc[0]), int(bc[1])), 18, (200, 0, 0), -1)       # 蓝球
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
        truth.append((tuple(rc), tuple(bc)))
    return frames, truth


def run(confirm_red, confirm_blue, frames, truth):
    """灰度跟踪（锚=红球初位）+ 可选颜色确认。返回 (维持率, 确认通过帧, 未通过帧)。"""
    td = Transduction2D(thresh=0.2, deadband=0.02, blur=1.2)
    anchor = truth[0][0]                        # 红球初位（任务层"大概在这"）
    target = None
    locked = False
    confirmed = False
    hits = frames_n = 0
    ok = fail = 0
    for i, fr in enumerate(frames):
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ef, es, _, _ = td.step(gray)
        ev = ((ef != 0) | (es != 0)).astype(bool)
        if not locked and i < 30:
            ys, xs = np.nonzero(ev & ((np.arange(ev.shape[1])[None, :] - anchor[0]) ** 2 +
                                      (np.arange(ev.shape[0])[:, None] - anchor[1]) ** 2 < 60 ** 2))
            if len(xs) > 10:
                target = [float(xs.mean()), float(ys.mean())]
                locked = True
        elif locked:
            ys, xs = np.nonzero(ev & ((np.arange(ev.shape[1])[None, :] - target[0]) ** 2 +
                                      (np.arange(ev.shape[0])[:, None] - target[1]) ** 2 < 40 ** 2))
            if len(xs) > 3:
                target = [float(xs.mean()), float(ys.mean())]
        # 颜色确认：目标区域色相
        if (confirm_red or confirm_blue) and locked and target is not None:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            r = 25
            x0i, x1i = max(0, int(target[0] - r)), min(W, int(target[0] + r))
            y0i, y1i = max(0, int(target[1] - r)), min(H, int(target[1] + r))
            region = h[y0i:y1i, x0i:x1i]
            if confirm_red:
                frac = float(((region >= RED_HUE[0]) | (region <= RED_HUE[1])).mean())
            else:
                frac = float(((region >= BLUE_HUE[0]) & (region <= BLUE_HUE[1])).mean())
            if frac > 0.2:
                confirmed = True
                ok += 1
            else:
                fail += 1
        frames_n += 1
        truth_r = truth[i][0]
        if target is not None and (not (confirm_red or confirm_blue) or confirmed):
            if ((target[0] - truth_r[0]) ** 2 + (target[1] - truth_r[1]) ** 2) ** 0.5 < 40:
                hits += 1
    return hits / max(frames_n, 1), ok, fail


frames, truth = make_scene()
print("== 受控彩色场景（红球+蓝球匀速直线，真值=运动学，光照渐变）==")
r_a, _, _ = run(False, False, frames, truth)
print(f"A  灰度跟踪（无确认）:          {r_a:.3f}（跟踪准度 vs 运动学真值）")
r_b, ok_b, fail_b = run(True, False, frames, truth)
print(f"B  灰度跟踪+红确认（任务'找红球'）: {r_b:.3f}（红确认通过 {ok_b} / 未通过 {fail_b}）")
r_c, ok_c, fail_c = run(False, True, frames, truth)
print(f"C  灰度跟踪+蓝确认（任务错'找蓝球'）: {r_c:.3f}（蓝确认通过 {ok_c} / 未通过 {fail_c}）")
print(f"\nP1 B ≥ A（确认不损跟踪）:  [{'PASS' if r_b >= r_a else 'FAIL'}]")
print(f"P2 C < B（错颜色拒）:      [{'PASS' if r_c < r_b else 'FAIL'}]")
print(f"P3 红确认通过率 > 0.9（高对比增益）: "
      f"[{'PASS' if ok_b / max(ok_b + fail_b, 1) > 0.9 else 'FAIL'}]")
