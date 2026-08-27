"""vision/davis_probe.py — DAVIS 真实目标探测：掩码空帧数、目标彩色像素、色相分布。

为"类别级怀疑系统回真实图像"（docs/218 缺口 1）确定实验设计参数：
  - 哪个视频有稳定的彩色目标（非纯色 → 光照漂移会动色相，docs/199 修正④）
  - 目标色相自然变异（帧间中位数 std = 真实漂移+噪声幅度）
  - 帧内色相 std（目标自身非纯色程度）
  - 掩码空帧（目标消失 = 检测任务的负样本）

用法：python vision/davis_probe.py [--video blackswan] [--all]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402

DAVIS = os.path.join("vision", "out", "davis")


def target_stats(video):
    vdir = os.path.join(DAVIS, video)
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    pngs = sorted(f for f in os.listdir(vdir) if f.endswith(".png"))
    print(f"\n== {video}: {len(jpgs)} frames ==")
    empty = 0
    hues_all = []
    for j, p in zip(jpgs, pngs):
        fr = cv2.imread(os.path.join(vdir, j))
        m = cv2.imread(os.path.join(vdir, p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.max() == 0:
            empty += 1
            continue
        mask = m > 0
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h, s = hsv[:, :, 0], hsv[:, :, 1]
        pix_h, pix_s = h[mask], s[mask]
        colored = pix_s > 60
        if colored.sum() < 20:
            continue
        hues_all.append((float(np.median(pix_h[colored])), float(np.std(pix_h[colored])),
                         int(colored.sum()), int(mask.sum())))
    print(f"  empty-mask frames: {empty}  (目标消失 = 检测负样本)")
    if hues_all:
        meds = [x[0] for x in hues_all]
        print(f"  彩色像素(掩码内): 帧间中位数色相 mean={np.mean(meds):5.1f}°  std={np.std(meds):5.1f}°"
              f"  range {min(meds):5.1f}-{max(meds):5.1f}°")
        print(f"  帧内色相 std (目标非纯色程度): mean {np.mean([x[1] for x in hues_all]):5.1f}°")
        print(f"  彩色像素数: mean {np.mean([x[2] for x in hues_all]):6.0f}  "
              f"掩码像素 mean {np.mean([x[3] for x in hues_all]):6.0f}")
    else:
        print("  无彩色目标像素（掩码内 S>60 太少）→ 该视频不适合颜色确认实验")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="flamingo")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    videos = ["blackswan", "car-turn", "flamingo", "motorbike", "soccerball", "surf"] \
        if args.all else [args.video]
    for v in videos:
        target_stats(v)


if __name__ == "__main__":
    main()
