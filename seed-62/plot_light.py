"""
SEED-62: 感知 d* —— 箱线图（论文图 A/B/C）

图 A: per-seed d* 分布（FIXED vs ADAPT）——审稿人第一问：3.5 档均值差是分布性的还是离群的
图 B: 准确率 vs L 箱线图（FIXED/ADAPT 逐档，41 seeds）——翻转边界 L=4 的方差尖峰可视化
图 C: 行为得分 per-episode 箱线图（ADAPT vs CLASS，L=7/L=1）——落点差 +20 分/局的分布

数据：重跑 light 的组件（≈秒级），不动 light.py 与 light_results.json（复现记录不破）。
输出：seed-62/figures/*.png (300dpi) + *.pdf
"""

import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import light

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
THETA = 0.5


def dstar_theta(acc_curve, theta=THETA):
    xs = sorted(int(k) for k in acc_curve)
    L_min = None
    for L in xs:
        if acc_curve[str(L)] >= theta:
            L_min = L
            break
    if L_min is None:
        return 0.0, None
    if L_min == xs[0]:
        return float(L_min), L_min
    a_hi = acc_curve[str(L_min)]
    a_lo = acc_curve[str(L_min - 1)]
    L_cross = (L_min - 1) + (theta - a_lo) / max(a_hi - a_lo, 1e-9)
    return L_cross, L_min


def per_seed_accuracy(arm):
    """返回 {seed: {str(L): acc}}；每 seed 一局（每类 N_PER_CLASS patch）。"""
    mat = {}
    for seed in light.SEEDS:
        row = {}
        for L in light.L_GRID:
            rng = random.Random(seed)
            pipe = light.FixedPipe() if arm == "FIXED" else light.AdaptPipe(1.0)
            pipe.reset(seed)
            if arm in ("ADAPT", "CLASS"):
                for _ in range(light.N_WARM):
                    pipe.step(rng, L, light.scene_patch(rng))
            correct = 0
            for c in light.CLASSES:
                s = light.PROTO[c]
                for _ in range(light.N_PER_CLASS):
                    if pipe.step(rng, L, s) == c:
                        correct += 1
            row[str(L)] = correct / (3 * light.N_PER_CLASS)
        mat[seed] = row
    return mat


def per_seed_scores(L, arm):
    """一局 30 patch（每类 10 个洗牌），返回 41 个 per-seed 得分。"""
    scores = []
    for seed in light.SEEDS:
        rng = random.Random(seed)
        pipe = light.FixedPipe() if arm == "FIXED" else light.AdaptPipe(1.0)
        pipe.reset(seed)
        if arm in ("ADAPT", "CLASS"):
            for _ in range(light.N_WARM):
                pipe.step(rng, L, light.scene_patch(rng))
        seq = ["ore"] * light.N_PER_CLASS + ["flat"] * light.N_PER_CLASS \
            + ["water"] * light.N_PER_CLASS
        rng.shuffle(seq)
        score = 0.0
        for c in seq:
            pred = pipe.step(rng, L, light.PROTO[c])
            act = "walk" if arm == "CLASS" else light.ACTION[pred]
            score += light.PAYOFF[(c, act)]
        scores.append(score)
    return scores


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, name + ".png")
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {png}")


def figA():
    """per-seed d* 分布：FIXED vs ADAPT。"""
    dstars = {}
    for arm in ("FIXED", "ADAPT"):
        mat = per_seed_accuracy(arm)
        ds = []
        for seed in light.SEEDS:
            L_cross, _ = dstar_theta(mat[seed])
            ds.append(10.0 - L_cross)
        dstars[arm] = ds
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    bp = ax.boxplot([dstars["FIXED"], dstars["ADAPT"]], widths=0.55,
                    patch_artist=True, showmeans=True, meanprops=dict(marker="D",
                    markerfacecolor="white", markeredgecolor="black"))
    for p, col in zip(bp["boxes"], ("#d98c8c", "#8cb0d9")):
        p.set_facecolor(col)
    ax.set_xticklabels(["FIXED\n(fixed exposure)", "ADAPT\n(adaptive gain)"])
    ax.set_ylabel("per-seed perception d* (illumination steps)")
    ax.axhline(10.0, color="gray", ls="--", lw=0.8)
    ax.set_title("Fig A: per-seed d* — distribution, not a single mean")
    save(fig, "figA_per_seed_dstar")


def figB():
    """准确率 vs L 箱线图：FIXED 与 ADAPT 逐档（41 seeds）。"""
    mats = {arm: per_seed_accuracy(arm) for arm in ("FIXED", "ADAPT")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, arm in zip(axes, ("FIXED", "ADAPT")):
        data = [[mats[arm][s][str(L)] for s in light.SEEDS] for L in light.L_GRID]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black"))
        for p in bp["boxes"]:
            p.set_facecolor("#d98c8c" if arm == "FIXED" else "#8cb0d9")
        ax.axhline(0.5, color="black", ls="--", lw=0.8)
        ax.set_xticks(range(1, 12))
        ax.set_xticklabels([str(L) for L in light.L_GRID], fontsize=8)
        ax.set_xlabel("illumination level L")
        ax.set_title(f"{arm}")
    axes[0].set_ylabel("affordance accuracy (41 seeds)")
    fig.suptitle("Fig B: accuracy vs L — variance spike at the flip boundary (L=4)", y=1.02)
    fig.tight_layout()
    save(fig, "figB_accuracy_vs_L")


def figC():
    """行为得分箱线图：ADAPT vs CLASS 在 L=7 与 L=1（落点判据分布）。"""
    groups, labels = [], []
    for L, tag in ((7, "L=7"), (1, "L=1")):
        for arm in ("ADAPT", "CLASS"):
            groups.append(per_seed_scores(L, arm))
            labels.append(f"{arm}\n{tag}")
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black"))
    for p, col in zip(bp["boxes"], ("#8cb0d9", "#f0c060", "#8cb0d9", "#f0c060")):
        p.set_facecolor(col)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("score per episode (30 patches)")
    ax.set_title("Fig C: landing point — knowing without entering state = not seeing")
    save(fig, "figC_landing_point")


if __name__ == "__main__":
    print("=== SEED-62 箱线图 ===")
    figA()
    figB()
    figC()
    print("done")
