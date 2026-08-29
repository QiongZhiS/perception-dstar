"""vision/cross_domain_probe.py — docs/248 L3 预注册辅助探针（数据元数据：帧数/分辨率/fps）。

Downloads 野域视频只读元数据（文件名/帧数/分辨率——按 docs/247 安全纪律，视频是数据，
不读像素内容）；stdout 只输出 ASCII 标签 + 每行一个数字的 R_CDP_* 摘要块（复用
vision/extract_r.py 抽取）。本脚本不写结果工件、不进判据——仅用于预注册阶段的视频选择。
"""
import argparse
import sys

import cv2

# 候选（按 Get-ChildItem 清单预选；排除 2 个 4GB+ 直播回放：时长数小时、采样意义弱、解码
# 成本高——如实声明排除理由）。中文文件名用 \u 转义保证源文件纯 ASCII。
CANDIDATES = [
    "<视频V1-studio脱敏>.mp4",
    "769768098-1-208.mp4",
    "41251049012-1-192.mp4",
    "1408717315_qe1-1-192.mp4",
    "<真实视频1-手机号脱敏>.mp4",
    "<视频V2-B站编号脱敏>.mp4",
    "12345.mp4",
    "4.mp4",
    "\uff1f.mp4",                          # ?.mp4（全角问号）
    "10\u670812\u65e5(1).mp4",             # <日期文件名脱敏(1)>.mp4
    "9\u67082\u65e5.mp4",                  # <日期文件名脱敏>.mp4
    "\u5343\u519b\u4e07\u9a6c\u54e6\u54e6\u54e6.mp4",  # <视频V3-内容脱敏>.mp4
    "\u8fdc.mp4",                          # 远.mp4
    "\uff1f\uff1f.mp4",                    # ??.mp4
    "210\u4ee4.mp4",                       # 210令.mp4
    "<桌面录制文件名脱敏>.mp4",
    "Desktop 2025.09.14 - 11.42.51.04.mp4",
]


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return dict(frames=-1, w=0, h=0, fps=0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return dict(frames=n, w=w, h=h, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\Users\<用户名>\Downloads")
    args = ap.parse_args()
    for i, name in enumerate(CANDIDATES):
        p = probe(args.dir + "\\" + name)
        print("R_CDP_%d_FRAMES=%d" % (i, p["frames"]))
        print("R_CDP_%d_SIZE=%dx%d" % (i, p["w"], p["h"]))
        print("R_CDP_%d_FPS=%.3f" % (i, p["fps"]))
        if p["fps"] > 0 and p["frames"] > 0:
            print("R_CDP_%d_SEC=%.1f" % (i, p["frames"] / p["fps"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
