"""vision/dstar_compress.py — d* 压缩率：信息 = 单位时间 d* 被压缩多少（docs/186 §三）。

docs/186 §三：信息度量从"事件数量"改为"d* 的压缩率"——d*（系统能维持预测多远）
突然缩短 = 信息密集；平滑 = 信息稀疏。判据：高利害事件（降幅/切变）应显著压缩 d*，
噪声风暴虽事件率高但 d* 已低位（无新信息）——事件率与压缩率背离 = "事件数量 ≠ 信息"。

在线 d* 定义（本实验的操作化）：
  - 每个像素维护 alive[i] = 自上次事件以来的帧数（封顶 C=50）
  - d*_t = alive 的 5% 低分位 = "最脆弱 5% 像素（最常撞墙者）的预测剩余寿命"
  - 静态：d*≈C（满健康）；运动/降幅/风暴：d* 下降
  - d* 压缩率_t = max(0, d*_{t-1} − d*_t)：单位时间 d* 被压缩的量

预言（docs/186 §三）：
  P1 降幅段压缩率显著 > 平静段（高利害事件压缩 d*）
  P2 风暴段事件率最高，但压缩率不高（d* 已低位，无新信息）——事件率 vs 压缩率背离
  P3 静态段压缩率 ≈ 0

用法：python vision/dstar_compress.py [--video vision/out/demo_storm.mp4]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D  # noqa: E402
import cv2  # noqa: E402

CAP = 50          # alive 封顶（预测健康上限 = 50 帧无事件）
Q = 0.05          # 低分位：最脆弱像素

# demo_storm 时段（缓渐变 0-2 / 降幅 3.0,5.0,7.5 / 风暴 4.0-4.9）
SEGMENTS = [
    ("缓渐变", 0.0, 2.0), ("平静1", 2.0, 3.0), ("降幅1", 3.0, 3.5),
    ("平静2", 3.5, 4.0), ("风暴", 4.0, 4.9), ("平静3", 4.9, 5.0),
    ("降幅2", 5.0, 5.3), ("平静4", 5.3, 7.5), ("降幅3", 7.5, 7.7),
    ("平静5", 7.7, 10.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join("vision", "out", "demo_storm.mp4"))
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--deadband", type=float, default=0.015)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--cuts", action="store_true",
                    help="真实视频模式：按亮度切变点分段（切变±1s vs 稳态）")
    ap.add_argument("--out", default=os.path.join("vision", "out", "dstar_compress.json"))
    args = ap.parse_args()

    src = args.video
    if not os.path.exists(src):
        sys.exit(f"缺视频：{src}（先跑 python vision/transduction.py 生成）")

    td = Transduction2D(thresh=args.thresh, deadband=args.deadband, blur=args.blur)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_pix = width * height

    alive = np.full((height, width), CAP, np.float32)
    rows = []          # (t, ev_rate, dstar, compress, light)
    i = 0
    prev_dstar = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, _, _ = td.step(gray)
        ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0)

        alive = np.minimum(alive + 1, CAP)
        alive[ev != 0] = 0.0
        dstar = float(np.percentile(alive, Q * 100))
        compress = max(0.0, prev_dstar - dstar) if prev_dstar is not None else 0.0
        prev_dstar = dstar

        rows.append({"t": i / fps, "ev_rate": float(ev.mean()),
                     "dstar": dstar, "compress": compress,
                     "light": float(gray.mean())})
        i += 1
    cap.release()

    print(f"== d* 压缩率（{os.path.basename(src)}，{i} 帧，θ={args.thresh} "
          f"blur={args.blur}，d* = alive 的 {int(Q*100)}% 低分位，封顶 {CAP}）==")

    if args.cuts:
        # 真实视频模式：按切变点分段
        from real_checklist import detect_cuts
        light = np.array([r["light"] for r in rows])
        cuts = detect_cuts(light)
        print(f"亮度切变 {len(cuts)} 处")
        cut_set = set()
        for c in cuts:
            for j in range(max(0, c - int(fps)), min(len(rows), c + int(fps) + 1)):
                cut_set.add(j)
        steady = [j for j in range(len(rows)) if j not in cut_set]
        print(f"{'':14s} {'事件率':>9s} {'平均d*':>9s} {'压缩率':>9s} {'脉冲占比':>9s}")
        seg_rep = {}
        for name, idx in [("切变段(±1s)", sorted(cut_set)), ("稳态段", steady)]:
            rs = [rows[j] for j in idx]
            ev = float(np.mean([r["ev_rate"] for r in rs]))
            ds = float(np.mean([r["dstar"] for r in rs]))
            cp = float(np.mean([r["compress"] for r in rs]))
            pl = float(np.mean([1.0 if r["compress"] >= 2.0 else 0.0 for r in rs]))
            seg_rep[name] = {"ev_rate": ev, "dstar": ds, "compress": cp, "pulse": pl}
            print(f"{name:14s} {ev:9.5f} {ds:9.2f} {cp:9.4f} {pl:9.4f}")
        print("\n切变时刻压缩率（应每次切变有脉冲）：")
        zero_at_cut = 0
        for c in cuts:
            if c < len(rows):
                print(f"  cut t={c/fps:6.1f}s  compress={rows[c]['compress']:.2f} "
                      f"d*={rows[c]['dstar']:.2f}")
                if rows[c]["dstar"] <= 0.5:
                    zero_at_cut += 1
        # 判据：切变帧本身应压缩 d*（d* 归 0）；切变帧压缩率 > 全片中位
        med_comp = float(np.median([r["compress"] for r in rows]))
        cut_comp = [rows[c]["compress"] for c in cuts if c < len(rows)]
        p_cut_zero = zero_at_cut >= 0.8 * min(len(cuts), len(cut_comp)) if cuts else False
        p_cut_pulse = float(np.mean([1.0 if v >= 2.0 else 0.0 for v in cut_comp])) > 0.5 \
            if cut_comp else False
        print(f"\n判据：切变帧 d* 归 0 占比 {zero_at_cut}/{min(len(cuts), len(cut_comp))} "
              f"[{'PASS' if p_cut_zero else 'FAIL'}]；切变帧压缩脉冲占比 "
              f"{np.mean([1.0 if v >= 2.0 else 0.0 for v in cut_comp]):.2f} "
              f"[{'PASS' if p_cut_pulse else 'FAIL'}]")
        print(f"注：稳态段压缩率({seg_rep['稳态段']['compress']:.4f})反映内容运动信息"
              f"（真实视频稳态≠静止）——压缩率度量的是信息，不是静止度。")
        rep = {"video": os.path.basename(src), "mode": "cuts", "n_frames": i,
               "dstar_def": f"alive {int(Q*100)}% 低分位，封顶 {CAP}",
               "cuts": [int(c) for c in cuts], "segments": seg_rep,
               "cut_dstar_zero": zero_at_cut, "cut_compress": cut_comp}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"\n→ {args.out}")
        return
    print(f"{'段':8s} {'事件率':>9s} {'平均d*':>9s} {'压缩率均值':>9s} {'压缩峰值':>9s}")
    results = {}
    for name, t0, t1 in SEGMENTS:
        rs = [r for r in rows if t0 <= r["t"] < t1]
        if not rs:
            continue
        ev = float(np.mean([r["ev_rate"] for r in rs]))
        ds = float(np.mean([r["dstar"] for r in rs]))
        cp = float(np.mean([r["compress"] for r in rs]))
        pk = float(np.max([r["compress"] for r in rs]))
        results[name] = {"t": [t0, t1], "ev_rate": ev, "dstar": ds,
                         "compress_mean": cp, "compress_peak": pk}
        print(f"{name:8s} {ev:9.5f} {ds:9.2f} {cp:9.4f} {pk:9.2f}")

    # 判据
    print("\n== 判据（docs/186 §三）==")
    storm = results.get("风暴", {})
    dips = [results[k] for k in results if k.startswith("降幅")]
    calms = [results[k] for k in results if k.startswith("平静")]
    dip_cp = float(np.mean([d["compress_mean"] for d in dips])) if dips else float("nan")
    calm_cp = float(np.mean([c["compress_mean"] for c in calms])) if calms else float("nan")
    storm_cp = storm.get("compress_mean", float("nan"))
    storm_ev = storm.get("ev_rate", float("nan"))
    dip_ev = float(np.mean([d["ev_rate"] for d in dips])) if dips else float("nan")

    # P1 降幅压缩率显著 > 平静（高利害事件压缩 d*）
    p1 = dip_cp > 2 * calm_cp
    # P2 事件率相近时，压缩率背离：风暴压缩率 < 0.7× 降幅压缩率 = 事件数量≠信息
    #    （风暴是噪声：事件多但 d* 已低位无新信息；降幅是结构事件：一次压缩信息密集）
    ev_ratio_ok = 0.5 <= storm_ev / max(dip_ev, 1e-9) <= 2.0 if dip_ev > 0 else False
    p2 = ev_ratio_ok and storm_cp < 0.7 * dip_cp
    # P3 信息脉冲（压缩率 ≥ 2 的帧 = d* 被显著压缩）：降幅段占比 > 平静段
    pulse = {}
    for name, t0, t1 in SEGMENTS:
        rs = [r for r in rows if t0 <= r["t"] < t1]
        pulse[name] = float(np.mean([1.0 if r["compress"] >= 2.0 else 0.0 for r in rs])) \
            if rs else float("nan")
    dip_pulse = float(np.mean([pulse[k] for k in pulse if k.startswith("降幅")])) \
        if dips else float("nan")
    calm_pulse = float(np.mean([pulse[k] for k in pulse if k.startswith("平静")])) \
        if calms else float("nan")
    p3 = dip_pulse > 3 * max(calm_pulse, 1e-9)

    print(f"P1 降幅压缩率({dip_cp:.4f}) > 2× 平静压缩率({calm_cp:.4f})  "
          f"[{'PASS' if p1 else 'FAIL'}]")
    print(f"P2 事件率相近(风暴{storm_ev:.4f} vs 降幅{dip_ev:.4f}, 比 {storm_ev/max(dip_ev,1e-9):.2f}) "
          f"但压缩率背离(风暴{storm_cp:.4f} < 0.7×降幅{dip_cp:.4f})  "
          f"[{'PASS' if p2 else 'FAIL'}]")
    print(f"P3 信息脉冲占比(降幅{dip_pulse:.4f} > 3× 平静{calm_pulse:.4f})  "
          f"[{'PASS' if p3 else 'FAIL'}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rep = {"video": os.path.basename(src), "fps": fps, "n_frames": i,
           "dstar_def": f"alive {int(Q*100)}% 低分位，封顶 {CAP}",
           "segments": results,
           "frames": rows,
           "judgments": {"P1": p1, "P2": p2, "P3": p3}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
