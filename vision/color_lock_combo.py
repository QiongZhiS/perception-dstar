"""vision/color_lock_combo.py — 颜色复合锁定（docs/192 §四修正落地）。

任务层外赋的正确形态：颜色描述 + 初始位置锚（"那辆绿车"——不是"全图找绿"）。
之前纯颜色描述（color_transduction_test）0.175 < 灰度 0.412——背景也有同色绿。
复合：绿色事件质心只在"初始位置锚附近"计算（空间约束排除远处背景绿）。

三臂对照（car-turn）：
  A 纯灰度外赋（精确框，docs/191）：0.412（基线）
  B 纯颜色描述（无位置锚）：0.175（背景同色污染）
  C 颜色+位置锚（复合）：目标 = 锚区域内绿色事件质心 → 应 > 0.412

位置锚 = 任务层给的粗略位置（真值 ± 60px 偏差模拟"大概在那"），
锚半径 = 100px（锁定期）/ 维持期用轨迹连续性。

判据：C > A > B（复合 > 灰度外赋 > 纯颜色描述）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_transduction import ColorTransduction
from davis_check import mask_boxes

vdir = os.path.join("vision", "out", "davis", "car-turn")
jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
HUE_LO, HUE_HI = 100, 140
SAT_MAX = 60


def run_arm(use_color, use_anchor):
    """use_color: 颜色事件参与；use_anchor: 位置锚约束（颜色模式下）。
    use_color=False = 纯灰度外赋（docs/191 基线，第一帧给框）。"""
    if use_color:
        ctd = ColorTransduction()
    else:
        from transduction import Transduction2D
        td = Transduction2D(thresh=0.35, deadband=0.06, blur=1.6)
    # 外赋：第一帧真值框（灰度模式）或位置锚（颜色模式）
    gt0 = mask_boxes(cv2.imread(os.path.join(vdir, jpgs[0].replace(".jpg", ".png")), 0))
    _, x0, y0, w0, h0 = gt0[0]
    anchor = (x0 + w0 / 2, y0 + h0 / 2)          # 真值中心
    if use_color and use_anchor:
        # 任务层给粗略位置（真值 ±60px 偏差）
        anchor = (anchor[0] + 60, anchor[1] - 60)
    ANCHOR_R = 100 if use_anchor else 1e9        # 锚半径（无锚=全图）

    target = None
    locked = False
    hits = frames = 0
    for i, jpg in enumerate(jpgs):
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace(".jpg", ".png")), 0)
        if fr is None or mask is None:
            continue
        if use_color:
            ev_h, _ = ctd.step(fr)
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            tm = ((hsv[:, :, 0] >= HUE_LO) & (hsv[:, :, 0] <= HUE_HI) &
                  (hsv[:, :, 1] <= SAT_MAX))
            color_events = (ev_h != 0) & tm
        else:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ef, es, _, _ = td.step(gray)
            color_events = ((ef != 0) | (es != 0)).astype(bool)
        # 锁定：锚区域内的事件质心（空间约束）
        if not locked and i < 40:
            if use_anchor:
                ys, xs = np.nonzero(color_events &
                                    ((np.arange(color_events.shape[1])[None, :] - anchor[0]) ** 2 +
                                     (np.arange(color_events.shape[0])[:, None] - anchor[1]) ** 2
                                     < ANCHOR_R ** 2))
            else:
                ys, xs = np.nonzero(color_events)
            if len(xs) > 10:
                target = {"cx": float(xs.mean()), "cy": float(ys.mean()), "age": 0}
                locked = True
        elif locked:
            if use_anchor:
                ys, xs = np.nonzero(color_events &
                                    ((np.arange(color_events.shape[1])[None, :] - target["cx"]) ** 2 +
                                     (np.arange(color_events.shape[0])[:, None] - target["cy"]) ** 2
                                     < 60 ** 2))
            else:
                ys, xs = np.nonzero(color_events)
            if len(xs) > 3:
                target["cx"], target["cy"] = float(xs.mean()), float(ys.mean())
            target["age"] += 1
        frames += 1
        gt = mask_boxes(mask)
        if gt and target is not None:
            gcx, gcy = gt[0][1] + gt[0][3] / 2, gt[0][2] + gt[0][4] / 2
            if ((target["cx"] - gcx) ** 2 + (target["cy"] - gcy) ** 2) ** 0.5 < 40:
                hits += 1
    return hits / max(frames, 1)


print("== 颜色复合锁定（car-turn，三臂同锚机制）==")
# A'：灰度事件 + 位置锚（与 C 同一锚机制，只差颜色 vs 灰度）
r_a = run_arm(use_color=False, use_anchor=True)
print(f"A' 灰度+位置锚:     {r_a:.3f}")
r_b = run_arm(use_color=True, use_anchor=False)
print(f"B  纯颜色（无锚）:  {r_b:.3f}")
r_c = run_arm(use_color=True, use_anchor=True)
print(f"C  颜色+位置锚:     {r_c:.3f}")
print(f"\n判据：C > A'（颜色优于灰度，同锚机制）且 C > B（锚优于无锚）"
      f"  [{'PASS' if r_c > r_a and r_c > r_b else 'FAIL'}]")
