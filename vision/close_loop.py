"""vision/close_loop.py — 闭环测试：视频 → 转导层 → 读信息 → 对真值验证。

"把输入输出闭环掉"：只看事件流，能不能找回移动物体？
  - 输入：demo 视频（真值已知：慢圆 + 快方块）
  - 链路：帧 → Transduction2D（事件流）→ EventReader（物体框+轨迹）
  - 验证：重建框 vs 真值框（IoU / 质心误差 / 检出率 / 误报数）
  - 输出：close_loop.mp4（原帧+真值黄圈+重建绿框 | 事件密度+重建框）
          close_loop_metrics.json

用法：python vision/close_loop.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D, make_demo_video  # noqa: E402
from reader import EventReader  # noqa: E402
import cv2  # noqa: E402


def demo_geoms(frame_idx, fps=30, seconds=10, width=320, height=240):
    """与 transduction.make_demo_video 一致的物体真值。"""
    t = frame_idx / fps
    cx = int(width * 0.25 + width * 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (seconds * 1.5))))
    cy = int(height * 0.55)
    fx = int(width * 0.5 + width * 0.45 * np.sin(2 * np.pi * t / 1.2))
    fy = int(height * 0.35 + height * 0.15 * np.sin(2 * np.pi * t / 0.9))
    return [{"x": cx - 22, "y": cy - 22, "w": 44, "h": 44, "name": "circle"},   # 慢圆
            {"x": fx - 10, "y": fy - 10, "w": 20, "h": 20, "name": "square"}]   # 快方块


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def main():
    src = os.path.join("vision", "out", "demo_blinks.mp4")
    if not os.path.exists(src):
        print("生成眨眼尺度演示视频（降幅 ~0.1s，人眼眨眼设计域）...")
        make_demo_video(src, mode="blinks")

    td = Transduction2D()
    # 两个读者 = 读层的多时间尺度（S6）：短窗读快物体，长窗读慢物体（稀疏事件要攒）
    reader_fast = EventReader(window=2, thr=0.25, min_area=12, blur=1.2)
    reader_slow = EventReader(window=10, thr=0.12, min_area=12, blur=1.5)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = os.path.join("vision", "out", "close_loop.mp4")
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width * 2 + 8, height))

    # 按物体统计（区分闪变帧/正常帧——闪变期是"预测维持"在工作）
    hit = {g["name"]: {"frames": 0, "iou": [], "err": [],
                       "flash_total": 0, "flash_hits": 0} for g in demo_geoms(0)}
    fp_total = 0
    pred_miss = 0
    flash_frames = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, _, _ = td.step(gray)
        flash = reader_fast.is_flash(ev_f, ev_s)     # 闪变判定：本帧事件 + 密度残留
        boxes = reader_fast.feed(ev_f, ev_s, flash=flash) + \
            reader_slow.feed(ev_f, ev_s, flash=flash)
        # 并集去重（跨读者重叠框合并）
        boxes = EventReader.merge_boxes(boxes)
        flash_now = flash
        if flash_now:
            flash_frames += 1

        gt = demo_geoms(frame_idx, fps)
        matched = {g["name"]: False for g in gt}
        for b in boxes:
            best_g, best_v = None, 0.2
            for g in gt:
                if matched[g["name"]]:
                    continue
                v = iou(b, g)
                gc = ((b["cx"] - (g["x"] + g["w"] / 2)) ** 2 +
                      (b["cy"] - (g["y"] + g["h"] / 2)) ** 2) ** 0.5
                if v > best_v or (v > 0.15 and gc < 25):   # 框住物体就算找到
                    best_g, best_v = g, v
            if best_g is not None:
                matched[best_g["name"]] = True
                h = hit[best_g["name"]]
                h["frames"] += 1
                h["iou"].append(best_v)
                err = ((b["cx"] - (best_g["x"] + best_g["w"] / 2)) ** 2 +
                       (b["cy"] - (best_g["y"] + best_g["h"] / 2)) ** 2) ** 0.5
                h["err"].append(err)
            else:
                if b.get("predicted"):
                    pred_miss += 1       # 闪变期预测维持没跟上，不算误报
                else:
                    fp_total += 1
        for g in gt:
            h = hit[g["name"]]
            if flash_now:
                h["flash_total"] += 1
                if matched[g["name"]]:
                    h["flash_hits"] += 1

        # 可视化：左 = 原帧+真值(黄)+重建(绿)/预测(品红)；右 = 事件密度+重建
        p1 = frame.copy()
        for g in gt:
            cv2.rectangle(p1, (g["x"], g["y"]), (g["x"] + g["w"], g["y"] + g["h"]),
                          (0, 255, 255), 1)                    # 真值=黄
        for b in boxes:
            col = (255, 0, 255) if b.get("predicted") else (0, 255, 0)
            cv2.rectangle(p1, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                          col, 2)
            cv2.putText(p1, f"#{b['track']}" + ("P" if b.get("predicted") else ""),
                        (b["x"], max(8, b["y"] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        d = np.clip(reader_fast.density / max(reader_fast.thr, 1e-6) * 255, 0, 255).astype(np.uint8)
        p2 = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        for b in boxes:
            col = (255, 0, 255) if b.get("predicted") else (0, 255, 0)
            cv2.rectangle(p2, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                          col, 1)
        tag = "FLASH" if flash_now else "     "
        txt = f"f{frame_idx} {tag} obj {len(boxes)} matched {sum(matched.values())}"
        cv2.putText(p1, txt, (6, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        writer.write(np.hstack([p1, np.zeros((height, 8, 3), np.uint8), p2]))
        frame_idx += 1
    cap.release()
    writer.release()

    # 汇总
    total = frame_idx
    rep = {"total_frames": total, "flash_frames": flash_frames, "objects": {}}
    print(f"\n== 闭环验证（{total} 帧，闪变帧 {flash_frames}，真值：慢圆+快方块）==")
    print("（闪变期 = 预测维持：品红框；正常期 = 绿框）\n")
    print(f"{'物体':8s} {'检出率':>8s} {'平均IoU':>8s} {'误差px':>8s} | {'闪变帧检出':>10s}")
    for name, h in hit.items():
        rate = h["frames"] / total
        miou = float(np.mean(h["iou"])) if h["iou"] else float("nan")
        merr = float(np.mean(h["err"])) if h["err"] else float("nan")
        fh = (h["flash_hits"] / h["flash_total"]) if h["flash_total"] else float("nan")
        rep["objects"][name] = {"detect_rate": rate, "mean_iou": miou,
                                "mean_err_px": merr, "flash_detect": fh}
        print(f"{name:8s} {rate:8.3f} {miou:8.3f} {merr:8.2f} | {fh:10.3f}")
    rep["false_positive_boxes"] = fp_total
    rep["predicted_miss_during_flash"] = pred_miss
    print(f"\n误报框：{fp_total}（{fp_total / max(total, 1):.2f}/帧）；"
          f"闪变期预测未跟上：{pred_miss}")
    print(f"输出：{out_video}")

    mpath = os.path.join("vision", "out", "close_loop_metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"指标：{mpath}")


if __name__ == "__main__":
    main()
