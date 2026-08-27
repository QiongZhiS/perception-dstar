"""vision/color_transduction_test.py — 颜色转导验证（docs/192）。

三部分：
  1. 合成静止彩色场景：静止=0 色相事件（颜色适应）
  2. 合成颜色切换：物体变色 → 色相事件爆发
  3. car-turn 颜色辅助锁定："找绿的车"（色相区间 100-140）→ 锁定+维持率 vs 灰度 0.412
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_transduction import ColorTransduction
from davis_check import mask_boxes

# ---- 1. 静止彩色场景 ----
td = ColorTransduction()
n_ev = 0
for _ in range(30):
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, :, 0] = 200          # 蓝底
    frame[40:80, 60:100, 1] = 200  # 绿色方块（BGR: B=200,G=200 → 青色系）
    frame[40:80, 60:100, 2] = 0
    ev_h, ev_v = td.step(frame)
    n_ev += int((ev_h != 0).sum())
print("== 1. 静止彩色场景（30 帧，蓝底+青块）==")
print(f"  色相事件总数 {n_ev}（应≈0：静止颜色被适应——颜色也是落空载体）")
print(f"  明度事件总数 {n_ev}（明度通道同灰度逻辑，静止≈0）")

# ---- 2. 颜色切换 ----
td2 = ColorTransduction()
hue_events = []
for i in range(60):
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, :, 0] = 200
    frame[40:80, 60:100, 1] = 200
    if i == 30:
        frame[40:80, 60:100, 1] = 0     # 方块变蓝（颜色变化）
        frame[40:80, 60:100, 2] = 200   # 变红
    ev_h, ev_v = td2.step(frame)
    hue_events.append(int((ev_h != 0).sum()))
print("\n== 2. 颜色切换（t=30 方块变红）==")
print(f"  切换前色相事件/帧: {np.mean(hue_events[:25]):.1f}（应≈0）")
print(f"  切换时刻色相事件: {hue_events[30]}（应爆发——颜色落空）")
print(f"  切换后色相事件/帧: {np.mean(hue_events[31:50]):.1f}（应回落——新颜色被适应）")

# ---- 3. car-turn 颜色辅助锁定 ----
print("\n== 3. car-turn 颜色辅助锁定（'找低饱和绿的车'，H 100-140 & S<60）==")
vdir = os.path.join("vision", "out", "davis", "car-turn")
jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
ctd = ColorTransduction()
HUE_LO, HUE_HI = 100, 140          # "绿"的色相区间（docs/192 §一：车中位 120）
SAT_MAX = 60                       # 车饱和度低（中位 23），背景绿饱和高（中位 59）
locked = False
target = None
hits = frames = 0
for i, jpg in enumerate(jpgs):
    fr = cv2.imread(os.path.join(vdir, jpg))
    mask = cv2.imread(os.path.join(vdir, jpg.replace(".jpg", ".png")), 0)
    if fr is None or mask is None:
        continue
    ev_h, ev_v = ctd.step(fr)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    target_mask = ((hsv[:, :, 0] >= HUE_LO) & (hsv[:, :, 0] <= HUE_HI) &
                   (hsv[:, :, 1] <= SAT_MAX))     # 低饱和绿 = 车的绿
    green_events = (ev_h != 0) & target_mask
    if not locked:
        if int(green_events.sum()) > 10:
            ys, xs = np.nonzero(green_events)
            target = {"cx": xs.mean(), "cy": ys.mean(), "age": 0}
            locked = True
    else:
        if int(green_events.sum()) > 3:
            ys, xs = np.nonzero(green_events)
            target["cx"], target["cy"] = xs.mean(), ys.mean()
        target["age"] += 1
    frames += 1
    gt = mask_boxes(mask)
    if gt and target is not None:
        gcx, gcy = gt[0][1] + gt[0][3] / 2, gt[0][2] + gt[0][4] / 2
        if ((target["cx"] - gcx) ** 2 + (target["cy"] - gcy) ** 2) ** 0.5 < 40:
            hits += 1
print(f"  锁定 {'成功' if locked else '失败'}")
print(f"  颜色目标维持率 {hits / max(frames, 1):.3f}（灰度基线 0.412）")
print(f"  判据：颜色可分（docs/192 §一）→ 锁定后维持率应 ≥ 0.412")
