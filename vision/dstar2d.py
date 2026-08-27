"""vision/dstar2d.py — 感知 d* 2D 版：光照轴，FIXED-2D vs EV-2D（docs/135 的 2D 推广）。

docs/135（1D）证明"自适应增益 vs 固定曝光"的失效边界；本实验推广到 2D——patch 带
空间结构（边缘/纹理/方向），侧抑制（中心-周围）进入特征。

- 世界：三类 24×24 patch——矿(亮+粗糙纹理) / 平地(中亮+平滑) / 水(中亮+水平反光条纹)
- 观测：光子噪声 n ~ Poisson(A·L·s) + 读出噪声，SNR ∝ √L（docs/135 同口径，A=20）
- FIXED-2D：曝光在 L=7 定死，读出按 (L/7) 收缩 → 刻度错位（学编码不学物理的签名）
- EV-2D：韦伯增益（背景 EWMA 估计光照，g=1/B̂）+ 侧抑制边缘特征
- EV-noHib：同增益但无侧抑制特征（只 mean/std）——侧抑制贡献的消融
- 分类：最近原型（L=7 校准，docs/135 同款）
- 度量：准确率 vs L（0-10，41 seeds）；感知 d* = 跌破 0.5 的档位差（从 L=10 往下数）

预言带（docs/133 实验①）：EV-2D d* ≈ 8-10，FIXED-2D ≈ 3-4（1D 实测 FIXED 6.18，
toy 几何偏乐观——如实报）。

用法：python vision/dstar2d.py [--seeds 41] [--out vision/out/dstar2d.json]
"""

import argparse
import json
import os
import sys

import numpy as np
import cv2  # noqa: E402


def make_patch(cls, rng, size=24):
    p = np.full((size, size), 0.5)
    if cls == "ore":                       # 矿：亮 + 粗糙（2×2 块高频噪声）
        p = np.full((size, size), 0.85)
        for y in range(0, size, 2):
            for x in range(0, size, 2):
                p[y:y + 2, x:x + 2] += rng.uniform(-0.14, 0.14)
    elif cls == "flat":                    # 平地：中亮 + 平滑
        p = np.full((size, size), 0.55) + rng.normal(0, 0.03, (size, size))
        p = cv2.GaussianBlur(p, (0, 0), 1.2)
    else:                                  # 水：中亮 + 水平反光条纹（方向性）
        p = np.full((size, size), 0.62) + rng.normal(0, 0.02, (size, size))
        for _ in range(3):
            row = int(rng.integers(2, size - 2))
            p[row:row + 2, :] += 0.25
        p = cv2.GaussianBlur(p, (0, 0), 0.8)
    return np.clip(p, 0.05, 1.0)


def observe(patch, L, rng, A=20.0, sig_read=3.0):
    photons = rng.poisson(A * L * patch).astype(np.float64) + rng.normal(0, sig_read, patch.shape)
    return photons / A                       # 归一化光子观测（均值 ≈ L·s）


def features(x, with_edges=True):
    edge = x - cv2.GaussianBlur(x, (0, 0), 1.0)      # 侧抑制（中心-周围）
    gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
    base = [x.mean(), x.std()]
    if with_edges:
        base += [np.abs(edge).mean(), np.abs(gx).mean(), np.abs(gy).mean()]
    return np.array(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=41)
    ap.add_argument("--out", default=os.path.join("vision", "out", "dstar2d.json"))
    args = ap.parse_args()

    classes = ["ore", "flat", "water"]
    Ls = np.arange(0, 11)
    rng = np.random.default_rng(20260826)
    arms = ["FIXED-2D", "EV-2D", "EV-noHib"]
    acc = {a: {int(L): [] for L in Ls} for a in arms}

    for _seed in range(args.seeds):
        N_PER_CLS = 15                       # 每类 patch 数（B̂ 要收敛，45 patch/seed/L）
        # 原型：L=7 校准（每臂每类 N_PER_CLS 样本取均值特征）
        protos = {a: {} for a in arms}
        for cls in classes:
            for a in arms:
                xs = []
                for _ in range(N_PER_CLS):
                    p = make_patch(cls, rng)
                    obs = observe(p, 7, rng)
                    x_f = obs / 7
                    if a == "FIXED-2D":
                        xs.append(features(x_f, True))
                    else:
                        B = np.mean(obs) / 7.0
                        x_e = x_f / max(B, 1e-9)
                        xs.append(features(x_e, with_edges=(a == "EV-2D")))
                protos[a][cls] = np.mean(xs, axis=0)
        P = {a: np.array([protos[a][c] for c in classes]) for a in arms}

        for L in Ls:
            B_hat = 1.0                       # 背景光照估计（EWMA，docs/135 α=0.35）
            for cls in classes:
                for _ in range(N_PER_CLS):
                    p = make_patch(cls, rng)
                    obs = observe(p, L, rng)
                    x_f = obs / 7
                    B_hat = 0.35 * np.mean(x_f) + 0.65 * B_hat
                    for a in arms:
                        if a == "FIXED-2D":
                            f = features(x_f, True)
                        else:
                            x_e = x_f / max(B_hat, 1e-9)
                            f = features(x_e, with_edges=(a == "EV-2D"))
                        pred = classes[int(np.argmin(np.sum((P[a] - f) ** 2, axis=1)))]
                        acc[a][int(L)].append(1.0 if pred == cls else 0.0)

    # 汇总
    print("\n== 感知 d* 2D（光照轴，%d seeds）==\n" % args.seeds)
    print("L | FIXED-2D | EV-2D | EV-noHib")
    mean = {}
    for L in Ls:
        row = []
        for a in arms:
            m = float(np.mean(acc[a][int(L)]))
            mean.setdefault(a, []).append(m)
            row.append(f"{m:.3f}")
        print(f"{int(L):2d} | " + " | ".join(row))

    def dstar(curve, base=10):
        # 从 base 档往下数，准确率跌破 0.5 的档位差（线性插值）；base=10（docs/133 口径）
        # 或 base=7（校准点口径，docs/135 备选）
        idx = list(Ls).index(base)
        cross = None
        for i in range(idx, 0, -1):
            if curve[i] > 0.5 >= curve[i - 1]:
                x1, y1, x2, y2 = Ls[i - 1], curve[i - 1], Ls[i], curve[i]
                cross = float(x1 + (0.5 - y1) * (x2 - x1) / max(1e-9, y2 - y1))
                break
        if cross is None:
            return None
        return round(base - cross, 2)

    print("\n感知 d*（跌破 0.5 的档位差）：")
    bands = {"FIXED-2D": "3-4（docs/133 预言）", "EV-2D": "8-10（docs/133 预言）",
             "EV-noHib": "—"}
    for a in arms:
        d10 = dstar(mean[a], 10)
        d7 = dstar(mean[a], 7)
        hit = ""
        if d7 is not None and a == "EV-2D":
            hit = " ✓带内" if 8 <= d7 <= 10 else f" 带外({d7})"
        print(f"  {a:10s} d*(从10)={d10}  d*(从校准点7)={d7}{hit}  预言带：{bands[a]}")

    rep = {"seeds": args.seeds, "accuracy": {a: mean[a] for a in arms},
           "dstar_from10": {a: dstar(mean[a], 10) for a in arms},
           "dstar_from7": {a: dstar(mean[a], 7) for a in arms},
           "Ls": [int(L) for L in Ls]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"\n指标：{args.out}")


if __name__ == "__main__":
    main()
