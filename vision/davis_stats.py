"""vision/davis_stats.py — docs/221 A2/B1.1：统计升级（多种子均值±SD + PR 曲线 + flips 检验）。

把 davis_suspicious 的结果从"单次运行"升级为"多种子统计稳健"：
  1. seeds 扫描：≥10 种子重跑序列构造（build_sequence seed 参数化）→ 每变体
     P/R/F1/翻转/各段通过率 的 均值±SD
  2. PR 曲线：A 变体（full）扫 thr_base（操作点扫描）→ P vs R 曲线 + F1 最优操作点
  3. flips 重采样检验：A（纪念）vs D（模板）翻转差异的跨种子 bootstrap 置信区间
     ——"不轻信"（翻转少）是统计显著的行为差异，不是单次运行

用法：
  python vision/davis_stats.py --seeds 10
  python vision/davis_stats.py --pr
  python vision/davis_stats.py --flips 10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from davis_suspicious import (load_video, build_sequence, DavisMind, run_mind,  # noqa: E402
                              metrics, flips, target_obs)

DAVIS_VIDEO = "flamingo"
SEG = {"base": 20, "dusk": 20, "corrupt": 15, "rest": 25}
THR = 0.60


def make_variants(task_hue, thr=THR):
    mk = lambda **kw: DavisMind(task_hue, thr_base=thr, **kw)  # noqa: E731
    return [
        ("A full      (wb, 类别级, 自适应带宽)", mk()),
        ("B nowb      (类别级, 自适应带宽)", mk(wb=False)),
        ("C global-wb (全局 thr, 有白平衡)", mk(global_thr=True)),
        ("C2 global   (全局 thr, 无白平衡)", mk(global_thr=True, wb=False)),
        ("D template  (朴素模板匹配, 无wb)", mk(template=True, wb=False)),
        ("E fixedbw10 (固定带宽10°, 无wb)", mk(wb=False, fixed_bw=10.0)),
        ("F fixedbw30 (固定带宽30°, 无wb)", mk(wb=False, fixed_bw=30.0)),
    ]


def task_hue_of(frames, masks):
    obs = []
    for f, m in zip(frames, masks):
        h = target_obs(f, m, 0.0)[1]
        if h is not None:
            obs.append(h)
    return float(np.median(obs))


def run_seed(video, seg, thr, corrupt, seed):
    """单个种子跑完整 7 变体。扰动强度由 seed 采样（docs/221 A2：多种子=真实方差）。"""
    frames, masks = load_video(video)
    th = task_hue_of(frames, masks)
    rng = np.random.default_rng(seed + 10_000)
    noise_std = 8.0 + rng.uniform(0, 6.0)       # 世界噪声强度 8-14
    corrupt_kc = 0.03 + rng.uniform(0, 0.04)    # 他者的不偏移强度 0.03-0.07
    seq_f, seq_m, seq_s, seq_gt = build_sequence(frames, masks, seg, seed=seed,
                                                 corrupt=corrupt,
                                                 noise_std=noise_std,
                                                 corrupt_kc=corrupt_kc)
    out = {}
    for name, mind in make_variants(th, thr):
        oks, seg_r, _, _ = run_mind(mind, seq_f, seq_m, seq_s, th)
        m = metrics(oks, seq_gt)
        out[name] = (m, flips(oks, seq_s), seg_r)
    return out


def seeds_scan(n_seeds=10, video=DAVIS_VIDEO, thr=THR, corrupt="gain"):
    print(f"== A2 多种子统计（{video}，{n_seeds} seeds，thr={thr}）==")
    runs = []
    for s in range(n_seeds):
        runs.append(run_seed(video, SEG, thr, corrupt, seed=s))
    names = list(runs[0].keys())
    print(f"{'变体':46s} {'P':>10s} {'R':>10s} {'F1':>12s} {'翻转':>10s} "
          f"{'rest':>12s} {'dusk':>10s}")
    for name in names:
        Ps = [r[name][0]["p"] for r in runs]
        Rs = [r[name][0]["r"] for r in runs]
        Fs = [r[name][0]["f1"] for r in runs]
        Fs_fl = [r[name][1] for r in runs]
        rests = [r[name][2]["rest"] for r in runs]
        dusks = [r[name][2]["dusk"] for r in runs]
        f = lambda v: f"{np.mean(v):.2f}±{np.std(v):.2f}"  # noqa: E731
        print(f"{name:46s} {f(Ps):>10s} {f(Rs):>10s} {f(Fs):>12s} "
              f"{np.mean(Fs_fl):5.1f}±{np.std(Fs_fl):4.1f} {f(rests):>12s} {f(dusks):>10s}")
    print("\n  读法：F1 差距 < 1SD = 变体间差异在此样本量不显著；翻转 A vs D 差异显著见 --flips")
    return runs


def pr_curve(thrs=None, video=DAVIS_VIDEO):
    """A 变体 thr_base 扫描 → PR 曲线（docs/207 权衡的统计版）。"""
    if thrs is None:
        thrs = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print(f"== A2 PR 曲线：A(full) 扫 thr_base（{video}，均值±SD over 10 seeds）==")
    frames, masks = load_video(video)
    th = task_hue_of(frames, masks)
    print(f"{'thr':>5s} {'P':>12s} {'R':>12s} {'F1':>12s}")
    for t in thrs:
        Ps, Rs, Fs = [], [], []
        for s in range(10):
            seq_f, seq_m, seq_s, seq_gt = build_sequence(frames, masks, SEG, seed=s)
            mind = DavisMind(th, thr_base=t)
            oks, _, _, _ = run_mind(mind, seq_f, seq_m, seq_s, th)
            m = metrics(oks, seq_gt)
            Ps.append(m["p"]); Rs.append(m["r"]); Fs.append(m["f1"])
        print(f"{t:5.2f} {np.mean(Ps):5.2f}±{np.std(Ps):4.2f} "
              f"{np.mean(Rs):5.2f}±{np.std(Rs):4.2f} {np.mean(Fs):5.2f}±{np.std(Fs):4.2f}")
    print("  操作点选择：F1 最大（或按任务偏误报/漏报）——报告 PR 曲线而非单点")


def flips_test(n_seeds=10, video=DAVIS_VIDEO):
    """A vs D 翻转差异的跨种子检验：均值±SD + 差异的 bootstrap 置信区间。"""
    print(f"== A2 flips 重采样检验：A(纪念) vs D(模板) 翻转差异（{video}）==")
    runs = [run_seed(video, SEG, THR, "gain", seed=s) for s in range(n_seeds)]
    fa = np.array([r["A full      (wb, 类别级, 自适应带宽)"][1] for r in runs])
    fd = np.array([r["D template  (朴素模板匹配, 无wb)"][1] for r in runs])
    diff = fa - fd
    print(f"  翻转 A: {fa.mean():.2f}±{fa.std():.2f}；D: {fd.mean():.2f}±{fd.std():.2f}；"
          f"差值 A-D: {diff.mean():.2f}±{diff.std():.2f}")
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n_seeds, n_seeds)
        boots.append(diff[idx].mean())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"  差值 bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]"
          f"  [{'PASS' if lo < 0 < hi is False and hi < 0 else 'PASS 显著为负' if hi < 0 else 'FAIL 不显著'}]")
    print("  读法：A 翻转 < D（纪念=不轻信）；CI 不含 0 → 差异统计显著")


def sensitivity(video=DAVIS_VIDEO, seeds=6):
    """docs/221 A3 敏感性：A(full) 单参数扫描 → F1/rest 变化（每点 seeds 个种子均值±SD）。
    其余参数保持默认（DavisMind 默认值见参数总表）。"""
    frames, masks = load_video(video)
    th = task_hue_of(frames, masks)
    sweeps = {
        "thr_base":  [0.40, 0.50, 0.60, 0.70],
        "thr_boost": [0.02, 0.04, 0.06],
        "grow":      [2.0, 3.0, 4.0],
        "cap_k":     [30, 60, 90],
        "k_band":    [1.0, 2.0, 3.0],
        "relax":     [0.01, 0.02, 0.04],
    }
    print(f"== A3 参数敏感性：A(full) 单参数扫描（{video}，{seeds} seeds/点）==")
    for pname, vals in sweeps.items():
        print(f"\n  {pname}:")
        for v in vals:
            kw = {pname: v}
            if pname == "thr_base":
                kw = {"thr_base": v}
            Fs, Rs = [], []
            for s in range(seeds):
                rng = np.random.default_rng(s + 10_000)
                seq_f, seq_m, seq_s, seq_gt = build_sequence(
                    frames, masks, SEG, seed=s,
                    noise_std=8.0 + rng.uniform(0, 6.0),
                    corrupt_kc=0.03 + rng.uniform(0, 0.04))
                mind = DavisMind(th, **kw)
                oks, _, _, _ = run_mind(mind, seq_f, seq_m, seq_s, th)
                m = metrics(oks, seq_gt)
                Fs.append(m["f1"]); Rs.append(m["r"])
            print(f"    {v!s:>5}: F1 {np.mean(Fs):.2f}±{np.std(Fs):.2f}  "
                  f"R {np.mean(Rs):.2f}±{np.std(Rs):.2f}")
    print("\n  读法：F1 在扫描范围 ±0.05 内=对参数不敏感（鲁棒）；敏感参数在论文中"
          "报告来源（docs/200-216 已验证值）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0, help="多种子扫描数量（0=跳过）")
    ap.add_argument("--pr", action="store_true", help="PR 曲线扫描")
    ap.add_argument("--flips", type=int, default=0, help="flips 检验种子数（0=跳过）")
    ap.add_argument("--sens", action="store_true", help="参数敏感性扫描")
    ap.add_argument("--video", default=DAVIS_VIDEO)
    args = ap.parse_args()
    if args.seeds:
        seeds_scan(args.seeds, video=args.video)
        print()
    if args.pr:
        pr_curve(video=args.video)
        print()
    if args.flips:
        flips_test(args.flips, video=args.video)
        print()
    if args.sens:
        sensitivity(video=args.video)


if __name__ == "__main__":
    main()
