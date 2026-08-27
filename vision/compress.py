"""vision/compress.py — 把管线输出的 mp4v 重编码成 H.264（管线产物 mp4v 太大）。

用法：
  python vision/compress.py vision/out/events_xxx.mp4            # 输出同名 _h264.mp4
  python vision/compress.py vision/out/events_xxx.mp4 --crf 26   # 更小
"""

import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="用 imageio-ffmpeg 的 ffmpeg 重编码为 H.264")
    ap.add_argument("video")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="medium")
    args = ap.parse_args()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = os.path.splitext(args.video)[0] + "_h264.mp4"
    cmd = [ffmpeg, "-y", "-i", args.video, "-c:v", "libx264",
           "-preset", args.preset, "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", out]
    print("运行：", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"重编码失败（exit {r.returncode}）")
    print(f"完成：{out}（{os.path.getsize(out) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
