"""vision/feature_atoms.py — L2 证伪条件：残差是原子 vs 非残差特征（docs/226）。

docs/226 L2 证伪条件："若存在系统，用非残差特征（如原始像素统计/帧间差分）在持续
视觉上同样有效且更简单——则'残差是原子'作废。"

实验设计：同样的读层（EventReader），喂三种特征源，测"只看特征流能否找回移动物体"：
  A 残差（事件流）   ：Transduction2D 预测-落空（对数域双 EWMA 背景 → 对比度 → 事件累积）
  B 帧间差分         ：|frame[t] − frame[t−1]| > θ（无内部模型，无预测，非残差但类似）
  C 原始帧统计       ：原始灰度直接阈值化（最朴素基线，无时间结构）

判据（docs/226 L2）：
  - 若 B 或 C 在同样任务上达到或超过 A 的检出率/质心误差，且更简单 → L2 证伪条件
    触发（残差不是原子，朴素特征足够）；
  - 若 A 显著优于 B/C（尤其在闪变/全局亮度突变时）→ L2 证伪条件不触发（残差有
    不可替代的结构价值：预测-落空编码了"世界对预期的拒绝"，帧间差分没有预测机制）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from transduction import Transduction2D, make_demo_video
from reader import EventReader


def demo_geoms(frame_idx, fps=30, seconds=10, width=320, height=240):
    """与 transduction.make_demo_video 一致的物体真值（慢圆 + 快方块）。"""
    t = frame_idx / fps
    cx = int(width * 0.25 + width * 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (seconds * 1.5))))
    cy = int(height * 0.55)
    fx = int(width * 0.5 + width * 0.45 * np.sin(2 * np.pi * t / 1.2))
    fy = int(height * 0.35 + height * 0.15 * np.sin(2 * np.pi * t / 0.9))
    return [{"x": cx - 22, "y": cy - 22, "w": 44, "h": 44, "name": "circle"},
            {"x": fx - 10, "y": fy - 10, "w": 20, "h": 20, "name": "square"}]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def frame_diff_events(fr, prev, theta=12.0):
    """帧间差分 → 事件图（非残差基线 B：无内部模型，纯时间差分）。"""
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if prev is None:
        return np.zeros(g.shape, np.uint8), g
    d = np.abs(g - prev)
    ev = (d > theta).astype(np.uint8)
    return ev, g


def raw_thresh_events(fr, theta=60.0):
    """原始帧统计（基线 C：最朴素，无时间结构，纯空间阈值）。"""
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # 背景 160 附近；物体偏离背景 → 阈值化出前景（模拟"原始像素统计"）
    ev = (np.abs(g.astype(np.int16) - 160) > theta).astype(np.uint8)
    return ev


def match_boxes(reader_boxes, gt):
    """读层输出框 vs 真值：匹配（IoU>0.3）数 / 质心误差。"""
    hits, err = 0, []
    for g in gt:
        best, biou = None, 0
        for b in reader_boxes:
            v = iou(g, b)
            if v > biou:
                best, biou = b, v
        if best and biou > 0.3:
            hits += 1
            err.append(np.hypot(best["cx"] - (g["x"] + g["w"] / 2),
                                best["cy"] - (g["y"] + g["h"] / 2)))
    return hits, (np.mean(err) if err else float('nan'))


def run_feature(feature, src, total_frames=300):
    """跑一种特征源，返回（检出率, 平均质心误差, 闪变段检出率）。"""
    td = Transduction2D()
    reader = EventReader(window=2, thr=0.25, min_area=12, blur=1.2)
    cap = cv2.VideoCapture(src)
    prev = None
    hits, errs, total_gt = 0, [], 0
    flash_hits, flash_total = 0, 0
    for i in range(total_frames):
        ok, fr = cap.read()
        if not ok:
            break
        gt = demo_geoms(i)
        if feature == "residual":
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ev_f, ev_s, _, _ = td.step(gray)
        elif feature == "diff":
            ev_d, prev = frame_diff_events(fr, prev)
            ev_f = ev_s = ev_d
        else:  # raw
            ev_f = ev_s = raw_thresh_events(fr)
        boxes = reader.feed(ev_f, ev_s)
        h, e = match_boxes(boxes, gt)
        hits += h
        total_gt += len(gt)
        if e == e:  # not nan
            errs.append(e)
        # 闪变段（t=8-10s 有 0.5s 降幅）：检测率单独算
        if 240 <= i < 270:
            flash_total += len(gt)
            flash_hits += h
    cap.release()
    return (hits / max(1, total_gt), np.mean(errs) if errs else float('nan'),
            flash_hits / max(1, flash_total))


def main():
    src = os.path.join("vision", "out", "demo_blinks.mp4")
    if not os.path.exists(src):
        print("生成眨眼尺度演示视频（降幅 ~0.1s，人眼眨眼设计域）...")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        make_demo_video(src, mode="blinks")
        if not os.path.exists(src):
            print("生成失败，尝试从 synthetic-life 复制...")
            alt = r"C:\Users\fa278\projects\synthetic-life\vision\out\demo_blinks.mp4"
            if os.path.exists(alt):
                os.makedirs(os.path.dirname(src), exist_ok=True)
                import shutil
                shutil.copy(alt, src)

    print("== L2 证伪条件：残差是原子 vs 非残差特征（docs/226）==")
    print("同一读层（EventReader）喂三种特征源，demo 场景（慢圆+快方块，含闪变段）\n")
    print(f"{'特征源':22s} {'检出率':>8s} {'质心误差':>8s} {'闪变段检出':>10s}")
    results = {}
    for name, feat in [("A 残差(事件流)", "residual"),
                       ("B 帧间差分", "diff"),
                       ("C 原始帧统计", "raw")]:
        det, err, flash = run_feature(feat, src)
        results[name] = (det, err, flash)
        print(f"{name:22s} {det:8.3f} {err:8.1f}px {flash:10.3f}")

    print("\n== 判读（docs/226 L2）==")
    a = results["A 残差(事件流)"]
    b = results["B 帧间差分"]
    c = results["C 原始帧统计"]
    print(f"A 残差 检出 {a[0]:.3f}  vs B 差分 {b[0]:.3f} vs C 原始 {c[0]:.3f}")
    print(f"A 闪变段 {a[2]:.3f} vs B {b[2]:.3f} vs C {c[2]:.3f}")
    if a[0] >= b[0] and a[0] >= c[0] and a[2] > b[2] and a[2] > c[2]:
        print("→ A 占优（尤其闪变段）：残差有不可替代的结构价值（预测-落空编码"
              "'世界对预期的拒绝'，帧间差分无预测机制）——L2 证伪条件不触发")
    elif b[0] >= a[0] or c[0] >= a[0]:
        print("→ B/C 达到或超过 A：朴素特征足够——L2 证伪条件触发（残差不是原子）")
    else:
        print("→ 结果混合：需更严格对照（docs/226 §六）")


if __name__ == "__main__":
    main()
