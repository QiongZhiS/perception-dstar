"""vision/real_checklist.py — 真实视频检查单（无真值，C1/C3/C4）。

真实视频没有已知物体位置（无真值），只做不需要真值的检查项（docs/183 方案 3）：

  C1 近似静态区事件率 —— "时间亮度方差低且非暗区"的空间块 = 不该动的感知区域；
                         这些块在稳态帧的事件率应 ≈0。若没有这样的块（全片在动或
                         全是暗区/黑边），诚实报告"不可定义"（不硬造静态区）。
                         暗区排除理由：log 域低灰度斜率大，压缩噪声被放大是物理
                         不是 bug；toy 的 C1 静态区亮度是 96/128。
  C3 切变后恢复剖面 —— 亮度大跳变（场景切变）后快/慢通道回落。定性报告：
                         README 已记真实素材恢复被内容主导，S6 定量留受控场景。
  C4 稳态稀疏度 —— 排除切变及其后恢复窗的平均稀疏度；目标：从 7.8% 压下来。

流式 3-pass（不把所有帧存内存，全片 12 分钟可跑）：
  pass1 亮度剖面 → 切变 → 稳态帧集
  pass2 稳态帧的块亮度统计（E[x], E[x²] → var）→ C1 静态块掩码
  pass3 转导层逐帧 → C1/C3/C4

用法：
  python vision/real_checklist.py --video <mp4> --scale 0.5 --max-frames 3000
  python vision/real_checklist.py --thresh 0.15 --deadband 0.03 --blur 1.6 --adaptive
  python vision/real_checklist.py --stats   # 打印块亮度/方差分布（判断 C1 可定义性）
"""

import argparse
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, "vision")
from transduction import Transduction2D  # noqa: E402


def detect_cuts(light, dthresh=15.0, merge=30):
    """亮度跳变点：|Δlight|>dthresh，合并 merge 帧内的连续跳变（转场跨帧）。"""
    dlight = np.abs(np.diff(light))
    idx = [i for i in range(1, len(light)) if dlight[i - 1] > dthresh]
    merged = []
    for i in idx:
        if merged and i - merged[-1] <= merge:
            continue
        merged.append(i)
    return merged


def steady_frames(n, cuts, recovery=30):
    """稳态帧 = 排除切变及其后 recovery 帧。"""
    exclude = set()
    for i in cuts:
        exclude.update(range(max(0, i - 2), min(n, i + recovery)))
    return [i for i in range(n) if i not in exclude]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"C:\Users\fa278\Downloads\41040086755-1-192.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--deadband", type=float, default=0.015)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--alpha-fast", type=float, default=0.5)
    ap.add_argument("--alpha-slow", type=float, default=0.03)
    ap.add_argument("--refractory", type=int, default=2)
    ap.add_argument("--adaptive", action="store_true", help="自适应模式（σ̂→θ/死区）")
    ap.add_argument("--learned-map", default=None,
                    help="学出映射 JSON（learn_adaptive 输出）：σ̂→(θ,db) 查表+插值，"
                         "上升快下降慢；与 --adaptive 互斥")
    ap.add_argument("--noise-mode", default="mad", choices=["mad", "hf", "both"],
                    help="σ̂ 特征：mad=MAD；hf=高频能量（内容事件也反映，docs/185）；"
                         "both=max。learned_map 用同一特征才有意义")
    ap.add_argument("--k-hf", type=float, default=0.12, help="hf→σ̂ 标定系数")
    ap.add_argument("--grid", type=int, default=8, help="C1 空间块网格")
    ap.add_argument("--var-max", type=float, default=10.0,
                    help="C1 静态区 = 时间亮度方差低于此值的块；无则 C1 不可定义")
    ap.add_argument("--lum-min", type=float, default=30.0,
                    help="C1 静态块最低平均亮度：排除暗区（log 域压缩噪声大，"
                         "toy 静态区亮度 96/128，真实视频黑边/暗区不该算感知静态区）")
    ap.add_argument("--out", default=os.path.join("vision", "out"))
    ap.add_argument("--stats", action="store_true", help="打印块亮度/方差分布后退出（不跑转导层）")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"打不开视频：{args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.scale != 1.0:
        w, h = max(2, int(w * args.scale)), max(2, int(h * args.scale))
    max_frames = args.max_frames if args.max_frames > 0 else total
    print(f"源 {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
          f"→ 处理 {w}x{h}，上限 {max_frames} 帧（共 {total}），fps {fps}", flush=True)

    # ---- pass1 亮度剖面（轻）----
    light = []
    idx = 0
    while idx < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        light.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean()))
        idx += 1
    light = np.array(light)
    n = len(light)
    cuts = detect_cuts(light)
    steady = set(steady_frames(n, cuts))
    print(f"亮度切变点 {len(cuts)} 个：{[f'{t:.1f}s' for t in (np.array(cuts) / fps)][:20]}",
          flush=True)

    # ---- pass2 稳态帧的块亮度统计（E[x], E[x²] → var）----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    g = args.grid
    e1 = np.zeros((g, g)); e2 = np.zeros((g, g)); cnt = np.zeros((g, g))
    idx = 0
    while idx < n:
        ok, fr = cap.read()
        if not ok:
            break
        if idx not in steady:
            idx += 1
            continue
        if args.scale != 1.0:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (g, g),
                           interpolation=cv2.INTER_AREA).astype(np.float64)
        e1 += small; e2 += small ** 2; cnt += 1
        idx += 1
    cnt = np.maximum(cnt, 1)
    block_lum = e1 / cnt
    var = e2 / cnt - block_lum ** 2
    var = np.maximum(var, 0.0)
    bh, bw = h // g, w // g

    if args.stats:
        print("\n== 块亮度时间方差分布（稳态帧，网格 "
              f"{g}x{g}={g*g} 块）==", flush=True)
        flat = np.sort(var.ravel())
        for p in [0, 10, 25, 50, 75, 90, 100]:
            print(f"  p{p:3d}: {np.percentile(var, p):8.3f}")
        print(f"  最低 15% 块均值: {flat[:max(1,int(0.15*len(flat)))].mean():.3f}")
        print("\n  var<var_max 的块及其平均亮度：", flush=True)
        for b in np.where(var.ravel() < args.var_max)[0]:
            by, bx = divmod(int(b), g)
            print(f"    ({by},{bx}) var={var.ravel()[b]:8.3f} lum={block_lum.ravel()[b]:8.1f}")
        cap.release()
        return

    # ---- C1 静态块（var<var_max 且 lum≥lum_min）----
    static_blocks = set(np.where((var.ravel() < args.var_max)
                                 & (block_lum.ravel() >= args.lum_min))[0])
    if static_blocks:
        thresh_var = float(max(var.ravel()[list(static_blocks)]))
        static_mask = np.zeros((h, w), bool)
        for b in static_blocks:
            by, bx = divmod(b, g)
            static_mask[by * bh:(by + 1) * bh, bx * bw:(bx + 1) * bw] = True
    else:
        thresh_var = float("nan")
        static_mask = np.zeros((h, w), bool)
    print(f"静态块 {len(static_blocks)}/{g*g}（var<{args.var_max} 且 lum≥{args.lum_min}）",
          flush=True)

    # ---- pass3 转导层逐帧 ----
    learned_map = None
    if args.learned_map:
        lm = json.load(open(args.learned_map, encoding="utf-8"))
        learned_map = [(p["sigma_hat"], p["theta"], p["db"]) for p in lm["learned"]]
    td = Transduction2D(alpha_fast=args.alpha_fast, alpha_slow=args.alpha_slow,
                        thresh=args.thresh, refractory=args.refractory,
                        blur=args.blur, deadband=args.deadband, adaptive=args.adaptive,
                        learned_map=learned_map, noise_mode=args.noise_mode,
                        k_hf=args.k_hf)
    n_pix = w * h
    ev_f = np.zeros(n, np.int64); ev_s = np.zeros(n, np.int64); ev_t = np.zeros(n, np.int64)
    static_ev = np.zeros(n, np.float64)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while idx < n:
        ok, fr = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ef, es, _, _ = td.step(gray)
        ev = np.where((ef != 0) | (es != 0), 1, 0)
        ev_f[idx] = int((ef != 0).sum()); ev_s[idx] = int((es != 0).sum())
        ev_t[idx] = int(ev.sum())
        if static_blocks and static_mask.any():
            static_ev[idx] = float(ev[static_mask].sum() / max(static_mask.sum(), 1))
        idx += 1
    cap.release()

    sparsity = ev_t / n_pix
    steady_arr = np.array(sorted(steady))

    # ---- C1 ----
    if static_blocks and len(steady_arr):
        c1 = float(static_ev[steady_arr].mean())
        c1_state = "PASS" if c1 < 0.001 else "WARN"
        c1_desc = f"静态块 {len(static_blocks)}/{g*g}，最大块方差 {thresh_var:.3f}"
    else:
        c1, c1_state = float("nan"), "不可定义（无 var<var_max 且 lum≥lum_min 的块）"
        c1_desc = "全片在动或全是暗区"
    print(f"\nC1 近似静态区事件率  {c1:.5f} 事件/像素/帧（{c1_desc}）  [{c1_state}]", flush=True)

    # ---- C3 定性 ----
    base_f = float(np.median(ev_f[steady_arr]))
    base_s = float(np.median(ev_s[steady_arr]))
    print(f"\nC3 切变恢复剖面（稳态底 快 {base_f:.0f}/帧 慢 {base_s:.0f}/帧；定性）", flush=True)
    for i in cuts:
        if i + 3 >= n:
            continue
        print(f"  cut t={i / fps:7.1f}s  light {light[i - 1]:6.1f}->{light[i]:6.1f}"
              f"  E_f {ev_f[i]:7d}/{ev_f[i + 1]:7d}/{ev_f[i + 3]:7d}"
              f"  E_s {ev_s[i]:7d}/{ev_s[i + 1]:7d}/{ev_s[i + 3]:7d}", flush=True)

    # ---- C4 ----
    c4 = float(sparsity[steady_arr].mean()) if len(steady_arr) else float("nan")
    c4_med = float(np.median(sparsity[steady_arr])) if len(steady_arr) else float("nan")
    all_s = float(sparsity.mean())
    print(f"\nC4 稳态稀疏度         {c4:.5f}（中位 {c4_med:.5f}）  [{'PASS' if c4 < 0.03 else 'WARN'}]"
          f"  全片平均 {all_s:.5f}", flush=True)

    out = {
        "params": {"thresh": args.thresh, "deadband": args.deadband, "blur": args.blur,
                   "alpha_fast": args.alpha_fast, "alpha_slow": args.alpha_slow,
                   "refractory": args.refractory, "adaptive": args.adaptive,
                   "learned_map": args.learned_map, "noise_mode": args.noise_mode,
                   "var_max": args.var_max, "lum_min": args.lum_min},
        "source": os.path.basename(args.video), "fps": fps, "size": [w, h],
        "n_frames": n, "cuts": [int(c) for c in cuts],
        "C1": {"static_rate": c1, "n_static_blocks": len(static_blocks),
               "max_static_var": thresh_var, "state": c1_state},
        "C3": {"steady_base_f": base_f, "steady_base_s": base_s,
               "note": "定性：真实素材恢复被内容主导，S6 定量留受控场景"},
        "C4": {"steady_sparsity": c4, "steady_sparsity_median": c4_med,
               "full_sparsity": all_s},
    }
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]
    mpath = os.path.join(args.out, f"real_checklist_{name}.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n→ {mpath}", flush=True)


if __name__ == "__main__":
    main()
