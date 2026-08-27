"""vision/checklist.py — 转导层定性检查单的量化版（demo 场景有真值，可打分）。

在合成演示视频上重跑转导层，用已知物体位置（慢圆/快方块）做真值，输出检查单：

  C1 静态区域事件率   —— 静态棋盘/噪声斑不该出事件（应 ≈0）
  C2 事件定位准确率   —— 事件应集中在移动物体上（物体占 ~2.5% 像素，事件应占大头）
  C3 阶跃后恢复帧数   —— 快通道几帧恢复、慢通道明显更久（多时间尺度签名）
  C4 稳态稀疏度       —— 无阶跃时段的事件/像素·帧越低越好（只处理变化）

用法：python vision/checklist.py [--video vision/out/demo_scene.mp4]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D, make_demo_video, run_pipeline  # noqa: E402
import cv2  # noqa: E402


# demo 几何（与 transduction.make_demo_video 一致）
def demo_geoms(frame_idx, fps=30, seconds=10, width=320, height=240):
    t = frame_idx / fps
    cx = int(width * 0.25 + width * 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (seconds * 1.5))))
    cy = int(height * 0.55)
    fx = int(width * 0.5 + width * 0.45 * np.sin(2 * np.pi * t / 1.2))
    fy = int(height * 0.35 + height * 0.15 * np.sin(2 * np.pi * t / 0.9))
    return (cx, cy), (fx, fy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="vision/out/demo_scene.mp4")
    args = ap.parse_args()

    src = args.video
    if not __import__("os").path.exists(src):
        print("生成演示视频...")
        make_demo_video(src)

    td = Transduction2D()
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_pix = width * height

    rows = []  # (frame, light, ev_f, ev_s, obj_frac, static_rate)
    frame_idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, _, _ = td.step(gray)
        ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0)
        (cx, cy), (fx, fy) = demo_geoms(frame_idx, fps)
        mask = np.zeros((height, width), bool)
        cv2.circle(mask, (cx, cy), 24, True, -1)                    # 慢圆 + 2px 余量
        cv2.rectangle(mask, (fx - 12, fy - 12), (fx + 12, fy + 12), True, -1)  # 快方块
        static = np.zeros((height, width), bool)
        static[5:25, 5:35] = True        # 左上角纯棋盘区（两物体永远碰不到）
        # 注：噪声材质斑 [30:70,40:80] 会被快速方块扫过，不能当静态区
        n_ev = int(ev.sum())
        obj_frac = float(ev[mask].sum() / max(n_ev, 1))
        static_rate = float(ev[static].sum() / max(static.sum(), 1))
        rows.append((frame_idx, float(gray.mean()), int((ev_f != 0).sum()),
                     int((ev_s != 0).sum()), obj_frac, static_rate))
        frame_idx += 1
    cap.release()

    frames = np.array([r[0] for r in rows])
    light = np.array([r[1] for r in rows])
    ev_f = np.array([r[2] for r in rows])
    ev_s = np.array([r[3] for r in rows])
    obj_frac = np.array([r[4] for r in rows])
    static_rate = np.array([r[5] for r in rows])

    # 阶跃帧 = |Δlight| 大
    dlight = np.abs(np.diff(light))
    step_idx = [i for i in range(1, len(light)) if dlight[i - 1] > 20]
    # 稳态帧 = 非阶跃及其后 30 帧
    exclude = set()
    for i in step_idx:
        exclude.update(range(max(0, i - 2), min(len(frames), i + 30)))
    steady = np.array([i for i in range(len(frames)) if i not in exclude])

    print("== 转导层检查单（demo，320×240）==")
    print(f"总帧 {len(frames)}，阶跃帧 {len(step_idx)} 处，稳态帧 {len(steady)} 帧\n")

    # C1 静态区域事件率（稳态帧平均）
    c1 = float(static_rate[steady].mean()) if len(steady) else float("nan")
    print(f"C1 静态区域事件率（应≈0）        {c1:.4f} 事件/像素/帧"
          f"  [{'PASS' if c1 < 0.001 else 'WARN'}]")

    # C2 事件定位准确率（稳态帧，物体区域事件占比）
    c2 = float(obj_frac[steady].mean()) if len(steady) else float("nan")
    print(f"C2 事件定位准确率（物体区占比）  {c2:.3f}（物体仅占 ~2.5% 像素）"
          f"  [{'PASS' if c2 > 0.5 else 'FAIL'}]")

    # C3 下降沿沉降时间：demo 三处降幅（0.5s），只测"亮→暗"沿，窗口 = 降幅段
    # （[下降沿, 下一阶跃)）。快通道两发爆发后 ~3 帧安静；慢通道残留衰减
    # （适应得慢）→ 累积器分批补发，沉降明显更长——S6 签名。
    step_all = [i for i in range(1, len(light)) if dlight[i - 1] > 20]
    base_f, base_s = np.median(ev_f[steady]), np.median(ev_s[steady])
    rec_f, rec_s = [], []
    for j, i in enumerate(step_all):
        if light[i] > light[i - 1]:
            continue                                # 只测下降沿
        nxt = step_all[j + 1] if j + 1 < len(step_all) else len(frames)
        win = nxt - i
        wf = np.clip(ev_f[i:i + win] - base_f, 0, None)
        ws = np.clip(ev_s[i:i + win] - base_s, 0, None)
        if len(wf) < 3 or wf.sum() <= 0 or ws.sum() <= 0:
            continue
        cf = np.cumsum(wf); cs = np.cumsum(ws)
        rec_f.append(int(np.nonzero(cf >= 0.95 * cf[-1])[0][0]))
        rec_s.append(int(np.nonzero(cs >= 0.95 * cs[-1])[0][0]))
    c3f = float(np.median(rec_f)) if rec_f else float("nan")
    c3s = float(np.median(rec_s)) if rec_s else float("nan")
    print(f"C3 下降沿沉降时间（95% 累积）  快 {c3f:.0f} 帧 / 慢 {c3s:.0f} 帧"
          f"  [{'PASS' if c3f <= 6 and c3s > c3f else 'WARN'}]（快恢复快、慢恢复慢 = S6 签名）")

    # C4 稳态稀疏度
    c4 = float((ev_f[steady] + ev_s[steady]).mean() / n_pix) if len(steady) else float("nan")
    print(f"C4 稳态稀疏度（越低越好）        {c4:.4f} 事件/像素/帧"
          f"  [{'PASS' if c4 < 0.03 else 'WARN'}]（物体边缘本身就占 ~2%，3% 内=只有物体）")

    print("\n注：C1/C2 是「信息保持」的量化——静态不该有事件、事件该在动的东西上；"
          "C3 是多时间尺度（S6）签名；C4 是「只处理变化」的能量账。")


if __name__ == "__main__":
    main()
