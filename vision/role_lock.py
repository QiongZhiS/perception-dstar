"""vision/role_lock.py — 三通道分工锁定（docs/195 完整机制验证）。

分工（docs/195）：颜色=静态区分（确认），明度=运动跟踪，位置锚=空间约束。
完整形态：灰度事件+位置锚锁定运动目标（A' 0.362 基础）→ 颜色确认目标身份
（锁定区域色相是否落在任务层给的区间）→ 确认通过才维持，不通过拒识（docs/123）。

四臂（car-turn，任务层说"找绿车"）：
  A 灰度+锚（无确认）：基线（docs/195 A' = 0.362）
  B 灰度+锚+颜色确认（目标色相在区间内才维持）
  C 灰度+锚+颜色确认，但任务层给错颜色（"找红车"——区间不含绿）
     → 应拒识（目标不是红的，确认不通过）
判据：
  P1 B ≥ A（确认不损跟踪——绿色目标通过确认）
  P2 C 明显低于 B（错颜色被拒——颜色确认防错目标）
  P3 确认率：B 的目标区域色相落在绿区间（机制工作）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from transduction import Transduction2D
from davis_check import mask_boxes

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "car-turn"
vdir = os.path.join("vision", "out", "davis", VIDEO)
jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
# 目标色相区间（任务层给）：car-turn=低饱和绿，flamingo=低饱和粉
if VIDEO == "flamingo":
    HUE_LO, HUE_HI = 125, 160
    SAT_MAX = 45
else:
    HUE_LO, HUE_HI = 100, 140
    SAT_MAX = 60


def hue_in_region(h, lo, hi):
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)     # 环形（红 170-10）


def run_arm(confirm_color, wrong_color=False):
    """confirm_color: 锁定后是否颜色确认；wrong_color: 任务层给错颜色。"""
    # 错颜色：目标色的对侧（car-turn 错红 170-10，flamingo 错蓝 90-115）
    WRONG = (170, 10) if VIDEO == "car-turn" else (90, 115)
    td = Transduction2D(thresh=0.35, deadband=0.06, blur=1.6)
    gt0 = mask_boxes(cv2.imread(os.path.join(vdir, jpgs[0].replace(".jpg", ".png")), 0))
    _, x0, y0, w0, h0 = gt0[0]
    anchor = (x0 + w0 / 2, y0 + h0 / 2)
    ANCHOR_R = 100
    lo, hi = WRONG if wrong_color else (HUE_LO, HUE_HI)

    target = None
    locked = False
    confirmed = False
    hits = frames = 0
    confirm_ok_frames = 0
    confirm_fail_frames = 0
    for i, jpg in enumerate(jpgs):
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace(".jpg", ".png")), 0)
        if fr is None or mask is None:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ef, es, _, _ = td.step(gray)
        ev = ((ef != 0) | (es != 0)).astype(bool)
        # 锁定（灰度+锚）
        if not locked and i < 40:
            ys, xs = np.nonzero(ev & (
                (np.arange(ev.shape[1])[None, :] - anchor[0]) ** 2 +
                (np.arange(ev.shape[0])[:, None] - anchor[1]) ** 2 < ANCHOR_R ** 2))
            if len(xs) > 10:
                target = {"cx": float(xs.mean()), "cy": float(ys.mean())}
                locked = True
        elif locked:
            ys, xs = np.nonzero(ev & (
                (np.arange(ev.shape[1])[None, :] - target["cx"]) ** 2 +
                (np.arange(ev.shape[0])[:, None] - target["cy"]) ** 2 < 60 ** 2))
            if len(xs) > 3:
                target["cx"], target["cy"] = float(xs.mean()), float(ys.mean())
        # 颜色确认：目标区域色相落在任务层区间 + 饱和度约束（flamingo 高对比在 S）
        if confirm_color and locked and target is not None:
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            h, s = hsv[:, :, 0], hsv[:, :, 1]
            r = 40
            x0i, x1i = max(0, int(target["cx"] - r)), min(h.shape[1], int(target["cx"] + r))
            y0i, y1i = max(0, int(target["cy"] - r)), min(h.shape[0], int(target["cy"] + r))
            region_h = h[y0i:y1i, x0i:x1i]
            region_s = s[y0i:y1i, x0i:x1i]
            frac = float((hue_in_region(region_h, lo, hi) &
                          (region_s <= SAT_MAX)).mean()) if region_h.size else 0.0
            # 目标区域色相在任务层区间且饱和度符合的占比（确认信号）
            if frac > 0.3:
                confirmed = True
                confirm_ok_frames += 1
            else:
                confirm_fail_frames += 1
        frames += 1
        gt = mask_boxes(mask)
        if gt and target is not None and (not confirm_color or confirmed):
            gcx, gcy = gt[0][1] + gt[0][3] / 2, gt[0][2] + gt[0][4] / 2
            if ((target["cx"] - gcx) ** 2 + (target["cy"] - gcy) ** 2) ** 0.5 < 40:
                hits += 1
    return hits / max(frames, 1), confirm_ok_frames, confirm_fail_frames


tgt_desc = "低饱和粉（flamingo）" if VIDEO == "flamingo" else "低饱和绿（car-turn）"
wrong_desc = "错蓝" if VIDEO == "flamingo" else "错红"
print(f"== 三通道分工锁定（{VIDEO}，目标={tgt_desc}）==")
r_a, _, _ = run_arm(confirm_color=False)
print(f"A  灰度+锚（无确认）:       {r_a:.3f}")
r_b, ok_b, fail_b = run_arm(confirm_color=True)
print(f"B  灰度+锚+颜色确认（{tgt_desc}）: {r_b:.3f}（确认通过 {ok_b} 帧 / 未通过 {fail_b} 帧）")
r_c, ok_c, fail_c = run_arm(confirm_color=True, wrong_color=True)
print(f"C  灰度+锚+颜色确认（{wrong_desc}）: {r_c:.3f}（确认通过 {ok_c} 帧 / 未通过 {fail_c} 帧）")
print(f"\nP1 B ≥ A（确认不损跟踪）:  [{'PASS' if r_b >= r_a else 'FAIL'}]")
print(f"P2 C < B（错颜色被拒）:    [{'PASS' if r_c < r_b else 'FAIL'}]")
print(f"P3 {tgt_desc}确认通过多（ok_b > fail_b）: [{'PASS' if ok_b > fail_b else 'FAIL'}]")
