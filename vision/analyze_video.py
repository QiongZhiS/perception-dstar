"""vision/analyze_video.py — 分析转导层指标：场景切变后的快/慢通道恢复剖面。

用法：
  python vision/analyze_video.py vision/out/metrics_41040086755-1-192.json

输出：
  - 亮度跳变（场景切变）列表
  - 每个大跳变后的恢复剖面：快通道 vs 慢通道事件衰减（帧数/事件数）
  - 汇总：切变平均恢复时间、稳态噪声底、快/慢恢复差（多时间尺度签名）
"""

import json
import sys

import numpy as np


def main():
    if len(sys.argv) < 2:
        sys.exit("用法：python vision/analyze_video.py metrics_xxx.json")
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    frames = d["frames"]
    light = np.array([f["light"] for f in frames])
    ev_f = np.array([f["ev_f"] for f in frames])
    ev_s = np.array([f["ev_s"] for f in frames])
    t = np.array([f["t"] for f in frames])

    # 亮度跳变 = 场景切变候选（|Δlight| > 阈值，取每处跳变的起点）
    dlight = np.abs(np.diff(light))
    cut_idx = [i for i in range(1, len(light)) if dlight[i - 1] > 15]
    # 合并 1s 内的连续跳变（转场可能跨 2-3 帧）
    merged = []
    for i in cut_idx:
        if merged and i - merged[-1] <= 30:
            continue
        merged.append(i)
    print(f"总帧 {len(frames)}，亮度跳变点（合并后）{len(merged)} 个")
    for i in merged[:15]:
        print(f"  cut t={t[i]:7.1f}s  light {light[i-1]:6.1f}->{light[i]:6.1f}  "
              f"E_f {ev_f[i]:6d}/{ev_f[i+1]:6d}/{ev_f[i+3]:6d}  "
              f"E_s {ev_s[i]:6d}/{ev_s[i+1]:6d}/{ev_s[i+3]:6d}")

    # 每个大跳变后的恢复：快/慢通道事件数回落到"稳态底"的帧数
    steady_base_f = np.median(ev_f)
    steady_base_s = np.median(ev_s)
    rec_f, rec_s = [], []
    for i in merged:
        win = ev_f[i:i + 100]
        if len(win) < 5:
            continue
        k = 0
        while k + 1 < len(win) and win[k + 1] > steady_base_f * 2:
            k += 1
        rec_f.append(k)
        win = ev_s[i:i + 100]
        k = 0
        while k + 1 < len(win) and win[k + 1] > steady_base_s * 2:
            k += 1
        rec_s.append(k)
    print(f"\n稳态底：快 {steady_base_f:.0f}/帧，慢 {steady_base_s:.0f}/帧")
    if rec_f:
        print(f"切变后回落帧数：快通道中位 {np.median(rec_f):.0f} 帧（均值 {np.mean(rec_f):.1f}），"
              f"慢通道中位 {np.median(rec_s):.0f} 帧（均值 {np.mean(rec_s):.1f}）")
        print(f"多时间尺度差（慢-快）：中位 {np.median(np.array(rec_s) - np.array(rec_f)):.0f} 帧 "
              f"（均值 {np.mean(np.array(rec_s) - np.array(rec_f)):.1f}）")

    # 事件率时间概况（每 10s）
    print("\n每 10s 平均事件数（快/慢/总）：")
    for t0 in range(0, int(t[-1]), 10):
        m = (t >= t0) & (t < t0 + 10)
        if m.sum():
            print(f"  {t0:4d}-{t0+10:4d}s  {ev_f[m].mean():7.0f} / {ev_s[m].mean():7.0f} / "
                  f"{(ev_f[m]+ev_s[m]).mean():7.0f}")


if __name__ == "__main__":
    main()
