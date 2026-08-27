"""vision/scan_real.py — 真实视频参数粗扫（一次读帧，多组参数各跑一遍转导层）。

目标：在限帧段上找 (θ, deadband, blur, α) 组合，把 C1（静态区事件率）和 C4（稳态
稀疏度）压到判据（C1<0.001、C4<0.03）——真实视频调参的第一轮。

用法：
  python vision/scan_real.py --max-frames 3000
  python vision/scan_real.py --max-frames 3000 --grids "thresh=0.12,0.2,0.3,0.4"
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, "vision")
from transduction import Transduction2D  # noqa: E402
from real_checklist import detect_cuts, steady_frames, block_var_profile  # noqa: E402


def parse_list(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"C:\Users\fa278\Downloads\41040086755-1-192.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=3000)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--var-max", type=float, default=10.0)
    ap.add_argument("--thresh", default="0.15,0.25,0.35,0.5")
    ap.add_argument("--deadband", default="0.03,0.06,0.1")
    ap.add_argument("--blur", default="0.8,1.6")
    ap.add_argument("--alpha-fast", default="0.5")
    ap.add_argument("--alpha-slow", default="0.03")
    ap.add_argument("--refractory", default="2")
    ap.add_argument("--adaptive", action="store_true",
                    help="同时加跑 adaptive=True 的组合（σ̂→θ/死区 只抬不降）")
    ap.add_argument("--out", default=os.path.join("vision", "out"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if args.scale != 1.0:
        w, h = max(2, int(w * args.scale)), max(2, int(h * args.scale))
    max_frames = min(args.max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"读帧 {max_frames}（{max_frames/fps:.0f}s，{w}x{h}）...")

    # 一次读完灰度帧 + 亮度剖面
    frames, light = [], []
    idx = 0
    while idx < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(g); light.append(float(g.mean())); idx += 1
    cap.release()
    light = np.array(light)
    cuts = detect_cuts(light)
    steady = steady_frames(len(frames), cuts)
    steady_arr = np.array(steady)
    n_pix = w * h
    var, bh, bw = block_var_profile(frames, args.grid, steady)
    static_blocks = set(np.where(var.ravel() < args.var_max)[0])
    if static_blocks:
        static_mask = np.zeros((h, w), bool)
        for b in static_blocks:
            by, bx = divmod(b, args.grid)
            static_mask[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw] = True
    else:
        static_mask = np.zeros((h, w), bool)
    print(f"切变 {len(cuts)} 处，稳态帧 {len(steady_arr)}，静态块 {len(static_blocks)}/64\n")

    thresh_l = parse_list(args.thresh)
    db_l = parse_list(args.deadband)
    blur_l = parse_list(args.blur)
    af_l = parse_list(args.alpha_fast)
    as_l = parse_list(args.alpha_slow)
    rf_l = [int(x) for x in parse_list(args.refractory)]
    combos = list(itertools.product(thresh_l, db_l, blur_l, af_l, as_l, rf_l))
    if args.adaptive:
        combos += [(t, db, bl, af, as_, rf, True) for t, db, bl, af, as_, rf
                   in itertools.product(thresh_l, db_l, blur_l, af_l, as_l, rf_l)]
    print(f"组合数：{len(combos)}", flush=True)

    results = []
    for combo in combos:
        adaptive = len(combo) == 7 and combo[6]
        th, db, bl, af, as_, rf = combo[:6]
        td = Transduction2D(alpha_fast=af, alpha_slow=as_, thresh=th,
                            refractory=rf, blur=bl, deadband=db,
                            adaptive=adaptive)
        ev_t = np.zeros(len(frames), np.int64)
        static_ev = np.zeros(len(frames), np.float64)
        for i, g in enumerate(frames):
            ev_f, ev_s, _, _ = td.step(g)
            ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0)
            ev_t[i] = int(ev.sum())
            static_ev[i] = float(ev[static_mask].sum() / max(static_mask.sum(), 1))
        sp = ev_t / n_pix
        c1 = float(static_ev[steady_arr].mean()) if len(steady_arr) else float("nan")
        c4 = float(sp[steady_arr].mean()) if len(steady_arr) else float("nan")
        results.append({
            "thresh": th, "deadband": db, "blur": bl, "alpha_fast": af,
            "alpha_slow": as_, "refractory": rf, "adaptive": adaptive,
            "C1": round(c1, 5), "C4": round(c4, 5),
            "C1_pass": c1 < 0.001, "C4_pass": c4 < 0.03,
        })
        tag = f"{'ADAPT' if adaptive else 'FIXED'} th={th:.2f} db={db:.3f} blur={bl:.1f} " \
              f"af={af:.2f} as={as_:.2f} R={rf}"
        print(f"  {tag:55s} C1 {c1:8.5f} {'PASS' if c1<0.001 else '   '}  "
              f"C4 {c4:8.5f} {'PASS' if c4<0.03 else '   '}", flush=True)

    results.sort(key=lambda r: (not (r["C1_pass"] and r["C4_pass"]), r["C4"]))
    print("\n== 前 10（按 C4，双 PASS 优先）==")
    for r in results[:10]:
        print(f"  th={r['thresh']:.2f} db={r['deadband']:.3f} blur={r['blur']:.1f} "
              f"af={r['alpha_fast']:.2f} as={r['alpha_slow']:.2f} R={r['refractory']} "
              f"{'ADAPT' if r['adaptive'] else 'FIXED':6s} C1 {r['C1']:.5f} C4 {r['C4']:.5f}")

    os.makedirs(args.out, exist_ok=True)
    mpath = os.path.join(args.out, "real_scan.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({"n_frames": len(frames), "cuts": cuts,
                   "n_static_blocks": len(static_blocks),
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n→ {mpath}")


if __name__ == "__main__":
    main()
