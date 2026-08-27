"""vision/davis_setup.py — DAVIS 2017 数据下载 + 单视频抽取（受控实拍适配第一步）。

DAVIS 2017（Densely Annotated VIdeo Segmentation）：真实拍摄、逐帧像素掩码真值。
用途：把 docs/183 判据"闭环检出率 ≥60%"第一次在真实图像 + 真值掩码上兑现——
"受控实拍"的现成样本（真实噪声/运动/遮挡/相机抖动，物理不是我们写的，docs/185/186）。

数据源：https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
（完整训练集 833MB，含 60 个视频：480p 帧序列 + 逐帧掩码 PNG）

用法：
  python vision/davis_setup.py                          # 下载 + 解压默认视频（blackswan）
  python vision/davis_setup.py --videos bear,camel      # 指定视频（逗号分隔）
  python vision/davis_setup.py --list                   # 列出训练集视频名（需先下载 zip）
  python vision/davis_setup.py --skip-download          # 已下载 zip，只解压

输出：vision/out/davis/<video>/（JPEG 帧 + 掩码 PNG + 视频清单 JSON）

DAVIS 目录结构（zip 内）：
  DAVIS/JPEGImages/480p/<video>/00000.jpg ...
  DAVIS/Annotations/480p/<video>/00000.png ...   # 掩码：0=背景, id=物体
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile

URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
DEFAULT_VIDEOS = ["blackswan"]     # 默认验证视频（黑天鹅，单物体、对比清晰）
ZIP_NAME = "DAVIS-2017-trainval-480p.zip"
OUT = os.path.join("vision", "out", "davis")


def download(zip_path):
    if os.path.exists(zip_path):
        print(f"已存在：{zip_path}（{os.path.getsize(zip_path) / 1e6:.0f}MB，跳过下载）")
        return
    print(f"下载 {URL}\n→ {zip_path}（833MB，耐心等；网络慢可手动下载放同路径）")
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(zip_path, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f}MB ({done / total * 100:.0f}%)",
                      end="", flush=True)
    print("\n下载完成")


def extract_videos(zip_path, videos):
    os.makedirs(OUT, exist_ok=True)
    want = {v for v in videos}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for v in want:
            vdir = os.path.join(OUT, v)
            os.makedirs(vdir, exist_ok=True)
            n_frames = 0
            for n in names:
                if n.endswith(".jpg") and f"/JPEGImages/480p/{v}/" in n:
                    target = os.path.join(vdir, os.path.basename(n))
                    if not os.path.exists(target):
                        with zf.open(n) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                    n_frames += 1
            for n in names:
                if n.endswith(".png") and f"/Annotations/480p/{v}/" in n:
                    target = os.path.join(vdir, os.path.basename(n))
                    if not os.path.exists(target):
                        with zf.open(n) as src, open(target, "wb") as dst:
                            dst.write(src.read())
            print(f"{v}: {n_frames} 帧 → {vdir}")
    # 清单
    manifest = {}
    for v in os.listdir(OUT):
        d = os.path.join(OUT, v)
        if not os.path.isdir(d):
            continue
        jpgs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        pngs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        manifest[v] = {"frames": len(jpgs), "masks": len(pngs)}
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n清单：{os.path.join(OUT, 'manifest.json')}")


def list_videos(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = set()
        for n in zf.namelist():
            parts = n.split("/")
            if len(parts) >= 4 and parts[0] == "DAVIS" and parts[1] == "JPEGImages" \
                    and parts[2] == "480p" and parts[3]:
                names.add(parts[3])
    print("训练集视频：")
    for v in sorted(names):
        print(f"  {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    zip_path = os.path.join(OUT, ZIP_NAME)
    if args.list:
        if not os.path.exists(zip_path):
            sys.exit(f"先下载：python vision/davis_setup.py（zip 在 {zip_path}）")
        list_videos(zip_path)
        return
    if not args.skip_download:
        download(zip_path)
    extract_videos(zip_path, [v.strip() for v in args.videos.split(",") if v.strip()])


if __name__ == "__main__":
    main()
