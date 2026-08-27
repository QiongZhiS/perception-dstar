"""vision/davis_check.py — DAVIS 闭环验证：事件流 → 物体框 vs 逐帧掩码真值。

把 docs/183 判据"闭环检出率 ≥60%"第一次在真实图像 + 真值掩码上兑现：
  - 输入：DAVIS 视频（vision/out/davis/<video>/00000.jpg...，davis_setup.py 抽取）
  - 真值：同目录掩码 PNG（0=背景，id=物体）→ 每物体外接框
  - 链路：帧 → Transduction2D（事件流）→ EventReader（框+跟踪）→ 对掩码 bbox 验证
  - 度量：按帧匹配（IoU ≥0.2 或 质心<25px 算找到），物体检出率 / 平均 IoU / 误报

用法：
  python vision/davis_check.py --video blackswan
  python vision/davis_check.py --video bear --max-frames 100

诚实边界：
  - DAVIS 是"受控实拍"的现成样本（真实噪声/运动/遮挡，物理非我们生成，docs/185/186）
  - 掩码是真值（比 demo 解析几何更普适）；物体数逐帧可变，匹配按帧动态建真值
  - 检出率对齐 docs/183 判据（≥60%）；参数沿用真实视频定量（θ=0.35/blur=1.6），
    不针对 DAVIS 调参（跨数据集泛化才是目的）
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D  # noqa: E402
from reader import EventReader  # noqa: E402
import cv2  # noqa: E402

DAVIS = os.path.join("vision", "out", "davis")


def iou_box(a, b):
    """两 bbox（x,y,w,h）IoU。"""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def mask_boxes(mask):
    """掩码 → 每物体 bbox（id≥1；跳过 <8px 的碎片）。返回 [(id, x, y, w, h)]。"""
    boxes = []
    ids = np.unique(mask)
    for oid in ids:
        if oid == 0:
            continue
        ys, xs = np.where(mask == oid)
        if len(xs) < 8:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        boxes.append((int(oid), int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="blackswan")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.35)
    ap.add_argument("--deadband", type=float, default=0.06)
    ap.add_argument("--blur", type=float, default=1.6)
    ap.add_argument("--out", default=os.path.join("vision", "out", "davis_check.json"))
    args = ap.parse_args()

    vdir = os.path.join(DAVIS, args.video)
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg")) if os.path.isdir(vdir) else []
    if not jpgs:
        sys.exit(f"没有 {vdir} 的帧——先跑 python vision/davis_setup.py --videos {args.video}")
    max_frames = args.max_frames if args.max_frames > 0 else len(jpgs)

    td = Transduction2D(thresh=args.thresh, deadband=args.deadband, blur=args.blur)
    reader_fast = EventReader(window=2, thr=0.25, min_area=12, blur=1.2)
    reader_slow = EventReader(window=10, thr=0.12, min_area=12, blur=1.5)

    frame_idx = 0
    total_gt = 0          # 真值物体-帧总数
    total_hit = 0         # 命中数
    ious = []
    fp_total = 0
    per_obj = {}          # (video,id) -> 统计

    while frame_idx < max_frames:
        jpg = os.path.join(vdir, jpgs[frame_idx])
        png = os.path.join(vdir, jpgs[frame_idx].replace(".jpg", ".png"))
        if not os.path.exists(png):
            frame_idx += 1
            continue
        frame = cv2.imread(jpg)
        mask = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, _, _ = td.step(gray)
        flash = reader_fast.is_flash(ev_f, ev_s)
        boxes = reader_fast.feed(ev_f, ev_s, flash=flash) + \
            reader_slow.feed(ev_f, ev_s, flash=flash)
        boxes = EventReader.merge_boxes(boxes)

        gt = mask_boxes(mask)
        matched = [False] * len(gt)
        for b in boxes:
            best_g, best_v = None, 0.2
            for gi, g in enumerate(gt):
                if matched[gi]:
                    continue
                v = iou_box((b["x"], b["y"], b["w"], b["h"]), g[1:])
                gc = ((b["cx"] - (g[1] + g[3] / 2)) ** 2 +
                      (b["cy"] - (g[2] + g[4] / 2)) ** 2) ** 0.5
                if v > best_v or (v > 0.15 and gc < 25):
                    best_g, best_v = gi, v
            if best_g is not None:
                matched[best_g] = True
                total_hit += 1
                ious.append(best_v)
                g = gt[best_g]
                key = f"{args.video}/{g[0]}"
                d = per_obj.setdefault(key, {"frames": 0, "hit": 0})
                d["frames"] += 1
                d["hit"] += 1
            else:
                if not b.get("predicted"):
                    fp_total += 1
        for gi, g in enumerate(gt):
            if not matched[gi]:
                key = f"{args.video}/{g[0]}"
                d = per_obj.setdefault(key, {"frames": 0, "hit": 0})
                d["frames"] += 1
        total_gt += len(gt)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  f{frame_idx} 命中 {total_hit}/{total_gt} 误报 {fp_total}", flush=True)

    rate = total_hit / max(total_gt, 1)
    miou = float(np.mean(ious)) if ious else float("nan")
    print(f"\n== DAVIS 闭环验证（{args.video}，{frame_idx} 帧）==")
    print(f"真值物体-帧 {total_gt}，命中 {total_hit}，检出率 {rate:.3f}，"
          f"平均 IoU {miou:.3f}，误报 {fp_total}（{fp_total / max(frame_idx, 1):.2f}/帧）")
    print(f"\n按物体：")
    for key, d in sorted(per_obj.items()):
        print(f"  {key}: 检出 {d['hit']}/{d['frames']} ({d['hit'] / max(d['frames'], 1):.3f})")
    verdict = "PASS" if rate >= 0.6 else "FAIL"
    print(f"\n判据（docs/183：闭环检出率 ≥60%）：{verdict}")

    rep = {"video": args.video, "frames": frame_idx, "detect_rate": rate,
           "mean_iou": miou, "false_positive": fp_total, "verdict": verdict,
           "params": {"thresh": args.thresh, "deadband": args.deadband, "blur": args.blur},
           "per_object": {k: v for k, v in sorted(per_obj.items())}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
