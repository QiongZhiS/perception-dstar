"""vision/adaptive_test.py — 固定参数 vs 自适应参数的对照（噪声风暴场景）。

风暴场景（demo_storm.mp4）：4.0-4.9s 噪声 σ 0.8→3.0（夹在两处降幅之间，其余不变）。
自适应模式用噪声估计 σ̂ = MAD(ΔL − median(ΔL)) 在线调 θ/死区（韦伯式：越噪声越保守，
docs/135 κ* 同构；减全局位移 = 降幅不污染 σ̂）。

输出：
  - 平静段 / 风暴段 的 C1（静态区事件率）与 C4（稀疏度）：固定 vs 自适应
  - θ 轨迹（证明参数真的在跟噪声走）
  - 判据：自适应在风暴段 C1/C4 明显低于固定（世界变了，参数跟上了）

用法：python vision/adaptive_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D, make_demo_video  # noqa: E402
import cv2  # noqa: E402

STORM = (4.0, 4.9)   # 风暴段
STEPS = [(3.0, 3.5), (5.0, 5.3), (7.5, 7.7)]  # 降幅段（统计时排除 ±1.5s）


def in_storm(t):
    return STORM[0] <= t < STORM[1]


def near_step(t):
    return any(t0 - 1.5 <= t <= t1 + 1.5 for t0, t1 in STEPS)


def run_config(src, adaptive):
    td = Transduction2D(adaptive=adaptive)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, _, _ = td.step(gray)
        ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0)
        static = np.zeros(gray.shape, bool)
        static[5:25, 5:35] = True
        rows.append({
            "t": i / fps,
            "sparse": float(ev.mean()),
            "c1": float(ev[static].mean()),
            "theta": td.thresh,
        })
        i += 1
    cap.release()
    return rows


def seg_stats(rows, sel):
    rs = [r for r in rows if sel(r["t"])]
    if not rs:
        return float("nan"), float("nan")
    return (float(np.mean([r["c1"] for r in rs])),
            float(np.mean([r["sparse"] for r in rs])))


def main():
    src = os.path.join("vision", "out", "demo_storm.mp4")
    if not os.path.exists(src):
        print("生成噪声风暴演示视频...")
        make_demo_video(src, storm=True)

    print("运行固定参数...")
    fixed = run_config(src, adaptive=False)
    print("运行自适应参数...")
    adaptive = run_config(src, adaptive=True)

    def calm(t):
        return not in_storm(t) and not near_step(t)

    print(f"\n== 固定参数 vs 自适应参数（风暴：{STORM[0]}-{STORM[1]}s，σ 0.8→3.0）==")
    print("（平静段/风暴段都已排除降幅附近 ±1.5s）\n")
    print(f"{'':14s} {'平静 C1':>9s} {'风暴 C1':>9s} | {'平静稀疏':>9s} {'风暴稀疏':>9s}")
    for name, rows in [("固定 θ=0.15", fixed), ("自适应 θ=15σ̂", adaptive)]:
        c1_calm, sp_calm = seg_stats(rows, calm)
        c1_storm, sp_storm = seg_stats(rows, in_storm)
        print(f"{name:14s} {c1_calm:9.4f} {c1_storm:9.4f} | {sp_calm:9.4f} {sp_storm:9.4f}")

    print("\nθ 轨迹（自适应，平静 vs 风暴代表帧）：")
    for r in adaptive:
        if abs(r["t"] - 2.0) < 0.03 or abs(r["t"] - 4.5) < 0.03 or abs(r["t"] - 6.0) < 0.03:
            tag = "风暴中" if in_storm(r["t"]) else "平静"
            print(f"  t={r['t']:.1f}s [{tag}]  θ={r['theta']:.3f}")
    print("\n判据：风暴段固定参数 C1/稀疏度应明显上涨（噪声随机游走），"
          "自适应参数应保持平稳（θ/死区跟着 σ̂ 抬高 = 越噪声越保守）。")


if __name__ == "__main__":
    main()
