"""vision/preview.py — 从管线输出裁一个高亮预览（转场+暗场+稳定段拼接）。

用法：python vision/preview.py 输入.mp4 输出.mp4 "8,20" "230,260" "480,500"
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 4:
        sys.exit('用法：python vision/preview.py 输入.mp4 输出.mp4 "t0,t1" ["t2,t3" ...]')
    src, out = sys.argv[1], sys.argv[2]
    segs = sys.argv[3:]
    expr = "+".join(f"between(t,{s})" for s in segs)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = f"select='{expr}',setpts=N/FRAME_RATE/TB"
    cmd = [ff, "-y", "-i", src, "-vf", vf,
           "-c:v", "libx264", "-preset", "fast", "-crf", "26",
           "-pix_fmt", "yuv420p", out]
    print("运行：", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(1)
    print(f"完成：{out}（{os.path.getsize(out) / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
