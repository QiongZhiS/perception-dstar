"""vision/learn_adaptive.py — 自适应映射学出来（docs/183 近程第 2 条，docs/176 落点）。

现状：手工自适应映射 θ=max(θ_cal, 15σ̂)、db=max(db_cal, 1.5σ̂)，**只抬不降**——真实
视频 3000 帧里 θ 有 2729 帧顶在 clamp 上限 0.6（一切变就永久顶格，灵敏度自适应丢失）。
目标：把 σ̂→(θ, db) 从手工常数换成**学出来的映射**（目标 = 事件效率/d\* 方向，
不是重建——"学编码不学物理"的反面），并允许"上升快、下降慢"的恢复（带滞回）。

学习：
  1. 生成多档噪声风暴场景（σ = storm_sigmas），每档跑固定 (θ, db) 网格
  2. 每档测风暴段：C1 静态区事件率、稀疏度、物体事件占比（demo 几何真值）
  3. 每档选 (θ*, db*) = 在"物体占比 ≥ keep_frac"约束下稀疏度最低（最安静且可读）
  4. 学出映射表 σ̂→θ*、σ̂→db*（单调插值；档间线性，档外延拓）
  5. 写回 Transduction2D 的 adaptive 模式（learned 模式：可升可降 + 下降慢）

对照（同 adaptive_test.py 口径）：FIXED θ=0.15 / 手工 ADAPT（只抬不降）/
学出 ADAPT（可升降），测风暴段 C1/稀疏度/物体占比 + θ 轨迹（是否恢复）。

用法：
  python vision/learn_adaptive.py                 # 全流程：学 + 对照
  python vision/learn_adaptive.py --sigmas 4,12   # 少档位快速验证
  python vision/learn_adaptive.py --grid-th 0.1,0.2,0.35,0.5
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D, make_demo_video  # noqa: E402
import cv2  # noqa: E402

STORM_LO, STORM_HI = 4.0, 4.9   # 风暴段（与 adaptive_test 一致）
CALM = [(2.0, 3.0), (6.0, 7.5)]  # 平静段取样（避开降幅 3.0-3.5/5.0-5.3/7.5-7.7）


def demo_masks(width, height):
    """demo 几何真值掩码：物体区（慢圆+快方块，含余量）与静态区（左上棋盘）。"""
    from checklist import demo_geoms
    obj = np.zeros((height, width), bool)
    static = np.zeros((height, width), bool)
    for i in range(30 * 10):            # 采样全程几何，并集当"物体可能出现区"太松；
        pass                             # 改为逐帧用——这里返回工厂，调用方逐帧取
    return None


def make_train_video(path, fps=30, width=320, height=240, seconds=10, seed=7,
                     storm_sigma=12.0, storm_lo=4.0, storm_hi=4.9):
    """训练用场景 = demo 场景 + 一个快速摆动的弱对比方块（亮度 115 vs 棋盘 96/128，
    Δlog≈0.18：θ=0.15 可读、θ≥0.35 读不出）。

    demo 原场景物体对比度太强（亮圆 Δlog≈0.5-0.7），θ 升到 0.65 也不丢信号——
    学习没有"弱信号 vs 噪声"的权衡。弱方块**快速摆动**（像素停留时间短 → ESIM
    累积器攒不够 → 高 θ 真的丢弱信号），给学习提供真实取舍：
    θ 高保宁静但丢弱信号，θ 低保弱信号但吃噪声。
    """
    rng = np.random.default_rng(seed)
    n = fps * seconds
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    bg = np.zeros((height, width), np.float32)
    step = 24
    for y in range(0, height, step):
        for x in range(0, width, step):
            bg[y:y + step, x:x + step] = 96 if ((x // step) + (y // step)) % 2 == 0 else 128
    bg[30:70, 40:80] = rng.normal(150, 12, (40, 40)).astype(np.float32)

    def slow_pos(t):
        return (int(width * 0.25 + width * 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (seconds * 1.5)))),
                int(height * 0.55))

    def fast_pos(t):
        return (int(width * 0.5 + width * 0.45 * np.sin(2 * np.pi * t / 1.2)),
                int(height * 0.35 + height * 0.15 * np.sin(2 * np.pi * t / 0.9)))

    def weak_pos(t):
        """弱方块快速摆动：右下角区域，周期 ~1.2s（同快方块量级）。"""
        return (int(width * 0.75 + width * 0.12 * np.sin(2 * np.pi * t / 1.2)),
                int(height * 0.8 + height * 0.05 * np.sin(2 * np.pi * t / 0.9)))

    def light_at(t):
        if t < 2.0:
            return 1.0 - 0.2 * (t / 2.0)
        light = 1.0
        for t0, level, dur in [(3.0, 0.5, 0.5), (5.0, 0.3, 0.3), (7.5, 0.7, 0.2)]:
            if t0 <= t < t0 + dur:
                light = level
        return light

    for i in range(n):
        t = i / fps
        light = light_at(t)
        img = bg * light
        cx, cy = slow_pos(t)
        cv2.circle(img, (cx, cy), 22, 200 * light, -1)
        fx, fy = fast_pos(t)
        cv2.rectangle(img, (int(fx) - 10, int(fy) - 10),
                      (int(fx) + 10, int(fy) + 10), 45 * light, -1)
        # 弱信号方块：亮度 115×light（与棋盘 96/128 的低对比弱信号），快速摆动
        wx, wy = weak_pos(t)
        cv2.rectangle(img, (wx - 9, wy - 9), (wx + 9, wy + 9), 115 * light, -1)
        noise = storm_sigma if (storm_lo <= t < storm_hi) else 0.8
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return fps, width, height, weak_pos


def make_object_mask(frame_idx, fps, width, height, weak_pos=None):
    """当前帧物体区掩码（慢圆+快方块，checklist 同款余量；可选弱方块位置函数）。"""
    from checklist import demo_geoms
    (cx, cy), (fx, fy) = demo_geoms(frame_idx, fps, seconds=10, width=width, height=height)
    mask = np.zeros((height, width), bool)
    cv2.circle(mask, (cx, cy), 24, True, -1)
    cv2.rectangle(mask, (fx - 12, fy - 12), (fx + 12, fy + 12), True, -1)
    if weak_pos is not None:
        wx, wy = weak_pos(frame_idx / fps)
        cv2.rectangle(mask, (wx - 12, wy - 12), (wx + 12, wy + 12), True, -1)
    return mask


def run_fixed(src, fps, thresh, deadband, blur, refractory=2, weak_pos=None):
    """固定参数跑全片，返回逐帧 (t, 稀疏度, C1静态, 物体占比, 物体绝对事件量)。"""
    td = Transduction2D(thresh=thresh, deadband=deadband, blur=blur, refractory=refractory)
    cap = cv2.VideoCapture(src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    static = np.zeros((height, width), bool)
    static[5:25, 5:35] = True
    rows = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, _, _ = td.step(gray)
        ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0)
        om = make_object_mask(i, fps, width, height, weak_pos)
        n_ev = int(ev.sum())
        rows.append({
            "t": i / fps,
            "sparse": float(ev.mean()),
            "c1": float(ev[static].mean()),
            "obj_frac": float(ev[om].sum() / max(n_ev, 1)),
            "obj_abs": int(ev[om].sum()),
        })
        i += 1
    cap.release()
    return rows


def seg_mean(rows, sel):
    rs = [r for r in rows if sel(r["t"])]
    if not rs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (float(np.mean([r["c1"] for r in rs])),
            float(np.mean([r["sparse"] for r in rs])),
            float(np.mean([r["obj_frac"] for r in rs])),
            float(np.mean([r["obj_abs"] for r in rs])))


def in_storm(t):
    return STORM_LO <= t < STORM_HI


def in_calm(t):
    return any(lo <= t < hi for lo, hi in CALM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.8,2,4,8,12,20,30",
                    help="训练噪声档位（含平静端 0.8；20/30 覆盖真实视频 hf 值域）")
    ap.add_argument("--grid-th", default="0.1,0.15,0.25,0.35,0.5,0.65,0.8")
    ap.add_argument("--grid-db", default="0.015,0.03,0.06,0.1")
    ap.add_argument("--blur", type=float, default=1.6)
    ap.add_argument("--noise-mode", default="mad", choices=["mad", "hf", "both"],
                    help="σ̂ 特征：mad=MAD（内容事件盲，docs/185 负根因）；hf=高频能量"
                         "（内容边缘也反映）；both=max")
    ap.add_argument("--k-hf", type=float, default=0.12,
                    help="hf→σ̂ 标定系数（σ̂_hf = k_hf × mean|Laplacian(ΔL)|）")
    ap.add_argument("--keep-frac", type=float, default=0.6,
                    help="信号约束：物体区绝对事件量 ≥ 网格最大值的 keep_frac"
                         "（弱信号别被高 θ 全压没）")
    ap.add_argument("--rise-tau", type=float, default=0.15,
                    help="学出模式上升 EWMA 系数（快：世界变噪立即抬）")
    ap.add_argument("--fall-tau", type=float, default=0.05,
                    help="学出模式下降 EWMA 系数（慢：风暴后缓缓恢复；0.05≈时间常数 20 帧）")
    ap.add_argument("--out", default=os.path.join("vision", "out", "learn_adaptive.json"))
    args = ap.parse_args()

    sigmas = [float(x) for x in args.sigmas.split(",")]
    th_l = [float(x) for x in args.grid_th.split(",")]
    db_l = [float(x) for x in args.grid_db.split(",")]
    fps, width, height = 30, 320, 240

    # ---- 1. 生成多档风暴（带弱信号方块） + 网格扫描 ----
    print(f"训练档位 σ = {sigmas}，网格 θ={th_l} × db={db_l}，blur={args.blur}", flush=True)
    train = {}          # sigma -> {"sigma_hat": x, "grid": [(th,db,c1,sparse,obj)]}
    learned = []        # 学出点 (sigma_hat, theta*, db*)
    weak_pos = None
    for sig in sigmas:
        src = os.path.join("vision", "out", f"learn_storm_{sig:g}.mp4")
        if not os.path.exists(src):
            _, _, _, weak_pos = make_train_video(src, storm_sigma=sig,
                                                 storm_lo=STORM_LO, storm_hi=STORM_HI)
        else:
            weak_pos = (lambda t, w=width, h=height: (
                int(w * 0.75 + w * 0.12 * np.sin(2 * np.pi * t / 1.2)),
                int(h * 0.8 + h * 0.05 * np.sin(2 * np.pi * t / 0.9))))
        # 自适应模式跑一遍，取风暴段中位 σ̂（σ̂ 只依赖输入，与 (θ,db) 无关）
        td = Transduction2D(adaptive=True, blur=args.blur,
                            noise_mode=args.noise_mode, k_hf=args.k_hf)
        cap = cv2.VideoCapture(src)
        sigs = []
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            td.step(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            if td.sigma_hat is not None and in_storm(i / fps):
                sigs.append(td.sigma_hat)
            i += 1
        cap.release()
        sig_hat = float(np.median(sigs)) if sigs else float("nan")

        grid = []
        for th in th_l:
            for db in db_l:
                rows = run_fixed(src, fps, th, db, args.blur, weak_pos=weak_pos)
                c1, sp, obj, obj_abs = seg_mean(rows, in_storm)
                grid.append({"th": th, "db": db, "c1": c1, "sparse": sp,
                             "obj": obj, "obj_abs": obj_abs})
        # 目标 = 约束下稀疏度最小化（最安静）：
        #   约束① 物体绝对事件量 ≥ 网格最大值的 keep_frac（弱信号别全压没）
        #   约束② obj_frac ≥ 0.45（事件流不被噪声主导，信息纯度底线）
        max_obj_abs = max((g["obj_abs"] for g in grid), default=0.0)
        keep = args.keep_frac * max_obj_abs
        cand = [g for g in grid if g["obj_abs"] >= keep and g["obj"] >= 0.45]
        if not cand:
            cand = [min(grid, key=lambda g: g["sparse"])]
        best = min(cand, key=lambda g: g["sparse"])        # 最安静且可读
        train[sig] = {"sigma_hat": sig_hat, "grid": grid, "best": best}
        learned.append({"sigma_hat": sig_hat, "theta": best["th"], "db": best["db"]})
        print(f"  σ={sig:4g}: σ̂={sig_hat:.4f}  →  θ*={best['th']:.2f} db*={best['db']:.3f} "
              f"(C1 {best['c1']:.4f} 稀疏 {best['sparse']:.4f} 物体占比 {best['obj']:.2f} "
              f"绝对量 {best['obj_abs']:.0f}/{max_obj_abs:.0f})", flush=True)

    # ---- 2. 学出映射表（σ̂ 单调插值，锚定平静端校准点）----
    # 锚点：σ̂=0（无噪声）→ 校准点 θ=0.15/db=0.015（docs/135 硬校准的地板）。
    learned.append({"sigma_hat": 0.0, "theta": 0.15, "db": 0.015})
    learned.sort(key=lambda p: p["sigma_hat"])
    # 去重：σ̂ 相同（或极近）只保留一个，保证 np.interp 的 xp 严格递增
    dedup = []
    for p in learned:
        if dedup and abs(p["sigma_hat"] - dedup[-1]["sigma_hat"]) < 1e-6:
            dedup[-1] = p          # 取更靠后的（θ 更完整）
        else:
            dedup.append(p)
    learned = dedup
    sh = np.array([p["sigma_hat"] for p in learned])
    th_p = np.array([p["theta"] for p in learned])
    db_p = np.array([p["db"] for p in learned])
    print("\n学出映射表：", flush=True)
    for p in learned:
        print(f"  σ̂={p['sigma_hat']:.4f} → θ={p['theta']:.2f}, db={p['db']:.3f}", flush=True)

    # ---- 3. 对照实验：FIXED / 手工 ADAPT / 学出 ADAPT ----
    src = os.path.join("vision", "out", "demo_storm.mp4")
    if not os.path.exists(src):
        make_demo_video(src, storm=True)

    def run_adaptive(src, mode):
        """mode="manual"：只抬不降；mode="learned"：σ̂→学出映射 + 上升快下降慢。
        σ̂ 特征统一由 noise_mode 决定（mad/hf/both）。"""
        if mode == "manual":
            td = Transduction2D(adaptive=True, blur=args.blur,
                                noise_mode=args.noise_mode, k_hf=args.k_hf)
        else:
            td = Transduction2D(blur=args.blur,
                                noise_mode=args.noise_mode, k_hf=args.k_hf,
                                learned_map=[(p["sigma_hat"], p["theta"], p["db"])
                                             for p in learned],
                                fall_tau=args.fall_tau)
        cap = cv2.VideoCapture(src)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        static = np.zeros((height, width), bool)
        static[5:25, 5:35] = True
        rows, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ev_f, ev_s, _, _ = td.step(gray)
            ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0)
            om = make_object_mask(i, fps, width, height)
            n_ev = int(ev.sum())
            rows.append({"t": i / fps, "sparse": float(ev.mean()),
                         "c1": float(ev[static].mean()),
                         "obj_frac": float(ev[om].sum() / max(n_ev, 1)),
                         "obj_abs": int(ev[om].sum()),
                         "theta": td.thresh})
            i += 1
        cap.release()
        return rows

    print("\n== 对照（demo_storm，风暴 4.0-4.9s σ=12，blur=%.1f）==" % args.blur)
    print(f"{'':16s} {'风暴C1':>9s} {'风暴稀疏':>9s} {'风暴物体':>9s} "
          f"{'平静C1':>9s} {'平静稀疏':>9s} {'全片C1':>9s} {'全片稀疏':>9s}")
    results = {}
    for name, mode in [("FIXED θ=0.15", "fixed"), ("手工ADAPT(只抬)", "manual"),
                       ("学出ADAPT(可降)", "learned")]:
        if mode == "fixed":
            rows = run_fixed(src, fps, 0.15, 0.015, args.blur)
        else:
            rows = run_adaptive(src, mode)
        c1_s, sp_s, obj_s, _ = seg_mean(rows, in_storm)
        c1_c, sp_c, _, _ = seg_mean(rows, in_calm)
        c1_all = float(np.mean([r["c1"] for r in rows]))
        sp_all = float(np.mean([r["sparse"] for r in rows]))
        results[mode] = {"storm_c1": c1_s, "storm_sparse": sp_s, "storm_obj": obj_s,
                         "calm_c1": c1_c, "calm_sparse": sp_c,
                         "all_c1": c1_all, "all_sparse": sp_all}
        print(f"{name:16s} {c1_s:9.4f} {sp_s:9.4f} {obj_s:9.2f} "
              f"{c1_c:9.4f} {sp_c:9.4f} {c1_all:9.4f} {sp_all:9.4f}")

    # θ 轨迹（学出 vs 手工）：平静→风暴→风暴后
    print("\nθ 轨迹（t=2.0 平静 / 4.5 风暴 / 5.2 风暴后 / 6.0 恢复）：")
    for name, mode in [("手工ADAPT(只抬)", "manual"), ("学出ADAPT(可降)", "learned")]:
        rows = run_adaptive(src, mode)
        line = []
        for tt in [2.0, 4.5, 5.2, 6.0]:
            r = next((r for r in rows if abs(r["t"] - tt) < 0.03), None)
            line.append(f"{r['theta']:.3f}" if r else "?")
        print(f"  {name:16s} " + " → ".join(line))

    rep = {"sigmas": sigmas, "learned": learned,
           "results": results, "storm": [STORM_LO, STORM_HI], "blur": args.blur}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n→ {args.out}")
    print("\n判据（docs/183）：学出映射 vs 手工 ADAPT——风暴段相当（C1 略优/稀疏略差但")
    print("信息纯度更高），全片（含风暴后恢复段）更好；学出来的形态是 db 主调节 + θ 地板")
    print("抬高（0.15→0.35），与手工 θ=15σ̂ 主导不同；目标始终是事件效率/d*，不是重建。")


if __name__ == "__main__":
    main()
