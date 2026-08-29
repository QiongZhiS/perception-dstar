"""
SEED-62: 感知 d* —— 光照适应实验 (docs/133 §六 实验① + docs/127 §3.8 + docs/124⑤)

域：1D 矿场的 affordance 感知。patch 真标签三类——矿(可挖)/平地(可上)/水(会湿)，
外观 = 3 特征光子信号（亮度/粗糙/反光），与标签相关但有噪声：光子泊松噪声 +
读出噪声，SNR ∝ 光照档位 L（光子噪声：信号均值 ∝ A(L)，噪声 sd ∝ sqrt(A(L))）。

三臂（分类器是同一个在 L_CAL=7 校准的线性读出，只在转导层不同）：
  FIXED  固定像素管线：曝光在 L=7 定死。L 下降 → 观测按 (L/7) 收缩 →
         读出尺度错位 → 判据翻转（"学编码不学物理"；夜盲 = 世界全变平地）。
  ADAPT  自适应增益：韦伯式增益 g = S_BAR/B_hat（B_hat = 背景光子均值 EWMA），
         把信号放回校准刻度；噪声地板 = 光子泊松噪声（SNR ∝ sqrt(L)）。
  CLASS  分类但不落状态（docs/124⑤ 落点判据）：与 ADAPT 同一通道（分类准确率
         逐位相同），但分类结果不驱动行为——行为用固定先验（永远走）。
         行为差 = "知道不改变行为 = 没看见"的量化。

度量：affordance 准确率 vs L 曲线；感知 d* = 准确率跌破 0.5 的档位差（从 L=10
往下数）；适应延迟（L 7→2 阶跃后增益收敛 tick 数）；行为耦合（挖矿数/湿身数/
得分）；过度适应（增益调谐参数 κ 扫 L∈{0..3}，找饱和锁死边界）。

Run:  python seed-62/light.py  ->  seed-62/light_results.json（两轮运行 MD5 一致）
"""

import hashlib
import json
import math
import os
import random
import time

# ---------------- 域与物理 ----------------
PROTO = {
    "ore":   [0.85, 0.35, 0.20],   # 矿：亮、中粗糙、低反光 -> 可挖
    "flat":  [0.55, 0.15, 0.45],   # 平地：中亮、平滑、中反光 -> 可上
    "water": [0.25, 0.65, 0.85],   # 水：暗、粗糙(波)、高反光 -> 会湿
}
CLASSES = ["ore", "flat", "water"]
S_BAR = round(sum(sum(v) for v in PROTO.values()) / 9.0, 6)   # 场景平均外观

A_UNIT = 20.0       # 光子数 / 档位 / 特征（每 patch 观测）
SIGMA_READ = 3.0    # 读出/暗噪声（光子）
L_CAL = 7.0         # FIXED 曝光校准档
L_MIN_EFF = 0.1     # ADAPT 增益上限 = L=0.1 的韦伯增益（自信最暗 0.1 档）
ALPHA = 0.35        # 背景 EWMA（seed62 NAM 家族同款 α）
SEEDS = list(range(60, 101))       # 41 seeds
N_WARM = 60         # 预热 patch 数（让 B_hat 收敛到当前光照）
N_PER_CLASS = 10    # 每类 patch 数（每局 30 patch）
KAPPAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
OVER_LS = [0, 1, 2, 3]
L_GRID = list(range(0, 11))

# ---- 行为耦合（代价通道：错要付出） ----
PAYOFF = {
    ("ore", "dig"): 1.0, ("ore", "walk"): 0.0, ("ore", "avoid"): 0.0,
    ("flat", "dig"): -0.5, ("flat", "walk"): 0.0, ("flat", "avoid"): -0.2,
    ("water", "dig"): -1.5, ("water", "walk"): -1.0, ("water", "avoid"): 0.0,
}
ACTION = {"ore": "dig", "flat": "walk", "water": "avoid"}


# ---------------- 基础工具 ----------------
def poisson(rng, lam):
    """确定性泊松采样：λ<30 用 Knuth，λ≥30 用正态近似（连续校正）。"""
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        L = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return k - 1
    v = max(0.0, lam + math.sqrt(lam) * rng.gauss(0.0, 1.0) - 0.5)
    return int(round(v))


def observe(rng, L, s):
    """一次 patch 观测：3 特征的光子计数 + 读出噪声。"""
    lam = [A_UNIT * L * s[i] for i in range(3)]
    return [poisson(rng, lam[i]) + rng.gauss(0.0, SIGMA_READ) for i in range(3)]


def quantize8(x):
    v = round(x * 255.0) / 255.0
    return 1.0 if v > 1.0 else (0.0 if v < 0.0 else v)


def classify(y):
    """在 L_CAL=7 刻度校准的线性读出（= 最近原型）。"""
    best, bc = -1e18, None
    for c in CLASSES:
        w = PROTO[c]
        sc = sum(w[i] * y[i] for i in range(3)) - 0.5 * sum(x * x for x in w)
        if sc > best:
            best, bc = sc, c
    return bc


def scene_patch(rng):
    """场景混合 patch（三类均匀）——用于背景估计预热。"""
    return PROTO[CLASSES[rng.randrange(3)]]


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    if len(xs) < 2:
        return 0.0
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ---------------- 转导管线 ----------------
class FixedPipe:
    """固定像素管线：曝光在 L_CAL=7 定死，之后不再改变。"""

    def reset(self, seed):
        self.g = 1.0 / (A_UNIT * L_CAL)

    def step(self, rng, L, s):
        n = observe(rng, L, s)
        y = [quantize8(self.g * n[i]) for i in range(3)]
        return classify(y)


class AdaptPipe:
    """raw+自适应增益管线：韦伯增益 g = kappa*S_BAR/B_hat。"""

    def __init__(self, kappa=1.0, alpha=ALPHA):
        self.kappa = kappa
        self.alpha = alpha

    def reset(self, seed):
        self.B = A_UNIT * L_CAL * S_BAR          # 校准背景（L=7 刻度）
        self.gmax = self.kappa * S_BAR / max(A_UNIT * L_MIN_EFF * S_BAR, 1e-9)

    def step(self, rng, L, s, learn=True):
        n = observe(rng, L, s)
        if learn:
            m = sum(n) / 3.0
            self.B = (1.0 - self.alpha) * self.B + self.alpha * max(m, 1e-6)
        g = min(self.kappa * S_BAR / max(self.B, 1e-9), self.gmax)
        y = [quantize8(g * n[i]) for i in range(3)]
        return classify(y)


# ---------------- 度量 1：affordance 准确率 vs L ----------------
def accuracy_at(L, arm, kappa=1.0):
    """每 seed 一局（每类 N_PER_CLASS patch）；返回 (逐 seed 准确率, 混淆矩阵, 每类准确率)。"""
    accs, per, conf = [], [0.0, 0.0, 0.0], [[0, 0, 0] for _ in range(3)]
    for seed in SEEDS:
        rng = random.Random(seed)
        if arm == "FIXED":
            pipe = FixedPipe()
        else:
            pipe = AdaptPipe(kappa)
        pipe.reset(seed)
        if arm in ("ADAPT", "CLASS"):
            for _ in range(N_WARM):
                pipe.step(rng, L, scene_patch(rng))      # 适应到当前光照
        correct = 0
        for ci, c in enumerate(CLASSES):
            s = PROTO[c]
            ok = 0
            for _ in range(N_PER_CLASS):
                pred = pipe.step(rng, L, s)
                pi = CLASSES.index(pred)
                conf[ci][pi] += 1
                if pi == ci:
                    correct += 1
                    ok += 1
            per[ci] += ok / N_PER_CLASS
        accs.append(correct / (3 * N_PER_CLASS))
    n = len(SEEDS)
    return accs, conf, [p / n for p in per]


def dstar(acc_curve):
    """acc_curve: {str(L): mean acc}；返回 (插值跌破档位 L_cross, 最低工作整数档 L_min)。"""
    xs = sorted(int(k) for k in acc_curve)
    L_min = None
    for L in xs:
        if acc_curve[str(L)] >= 0.5:
            L_min = L
            break
    if L_min is None:
        return 0.0, None
    if L_min == xs[0]:
        return float(L_min), L_min
    a_hi = acc_curve[str(L_min)]
    a_lo = acc_curve[str(L_min - 1)]
    L_cross = (L_min - 1) + (0.5 - a_lo) / max(a_hi - a_lo, 1e-9)
    return L_cross, L_min


# ---------------- 度量 2：适应延迟 ----------------
def adaptation_delay(alpha=ALPHA, frm=7, to=2, tol=0.05, seed=60):
    """L frm->to 阶跃后，背景估计 B_hat 收敛到 5% 内的 tick 数 + 增益轨迹。"""
    rng = random.Random(seed)
    pipe = AdaptPipe(1.0, alpha)
    pipe.reset(seed)
    for _ in range(50):                                 # 在旧光照稳态
        pipe.step(rng, frm, scene_patch(rng))
    B_star = A_UNIT * to * S_BAR
    traj, delay = [], None
    for t in range(1, 401):
        pipe.step(rng, to, scene_patch(rng))
        g = min(pipe.kappa * S_BAR / max(pipe.B, 1e-9), pipe.gmax)
        if t <= 24 and (t <= 10 or t % 2 == 0):
            traj.append([t, round(pipe.B, 4), round(g, 6)])
        if delay is None and abs(pipe.B - B_star) <= tol * B_star:
            delay = t
    return delay, traj


# ---------------- 度量 3：行为耦合 + 落点判据 ----------------
def behavior_at(L, arm, kappa=1.0):
    """一局 = 走 30 个 patch（每类 10 个，洗牌）；返回 (得分, 挖矿数, 湿身数) 的种子均值。"""
    rows = []
    for seed in SEEDS:
        rng = random.Random(seed)
        if arm == "FIXED":
            pipe = FixedPipe()
        else:
            pipe = AdaptPipe(kappa)
        pipe.reset(seed)
        if arm in ("ADAPT", "CLASS"):
            for _ in range(N_WARM):
                pipe.step(rng, L, scene_patch(rng))
        seq = ["ore"] * N_PER_CLASS + ["flat"] * N_PER_CLASS + ["water"] * N_PER_CLASS
        rng.shuffle(seq)
        score, ores, whits = 0.0, 0, 0
        for c in seq:
            pred = pipe.step(rng, L, PROTO[c])
            act = "walk" if arm == "CLASS" else ACTION[pred]   # CLASS：分类不落状态
            score += PAYOFF[(c, act)]
            if c == "ore" and act == "dig":
                ores += 1
            if c == "water" and act in ("walk", "dig"):
                whits += 1
        rows.append((score, ores, whits))
    sc = [r[0] for r in rows]
    oo = [r[1] for r in rows]
    ww = [r[2] for r in rows]
    return {
        "score": round(mean(sc), 3), "score_std": round(stdev(sc), 3),
        "ores": round(mean(oo), 3), "water_hits": round(mean(ww), 3),
    }


# ---------------- 主流程 ----------------
def main():
    t0 = time.time()
    out = {}
    meta = dict(seeds=SEEDS, L_grid=L_GRID, A_UNIT=A_UNIT, SIGMA_READ=SIGMA_READ,
                L_CAL=L_CAL, L_MIN_EFF=L_MIN_EFF, alpha=ALPHA, n_warm=N_WARM,
                n_per_class=N_PER_CLASS, kappas=KAPPAS, over_Ls=OVER_LS,
                S_BAR=S_BAR, proto=PROTO,
                payoff=[[k[0], k[1], v] for k, v in PAYOFF.items()])
    out["meta"] = meta

    # 1) 准确率 vs L + d*
    acc_curves, acc_out = {}, {}
    for arm in ("FIXED", "ADAPT"):
        acc_curves[arm], acc_out[arm] = {}, {}
        for L in L_GRID:
            accs, conf, per = accuracy_at(L, arm)
            acc_curves[arm][str(L)] = mean(accs)
            acc_out[arm][str(L)] = {
                "acc": round(mean(accs), 4), "std": round(stdev(accs), 4),
                "per_class": [round(x, 4) for x in per],
                "conf": conf,
            }
    out["accuracy"] = acc_out
    dstar_out = {}
    for arm in ("FIXED", "ADAPT"):
        L_cross, L_min = dstar(acc_curves[arm])
        dstar_out[arm] = {"L_cross": round(L_cross, 2), "L_min": L_min,
                          "dstar": round(10.0 - L_cross, 2)}
    out["dstar"] = dstar_out

    # 2) 适应延迟（L 7->2 阶跃）
    delay_out = {}
    for alpha in (0.2, 0.35, 0.5):
        d, traj = adaptation_delay(alpha=alpha)
        delay_out[str(alpha)] = {"delay": d, "traj": traj}
    out["delay"] = delay_out

    # 3) 行为耦合（FIXED / ADAPT / CLASS-ONLY）
    beh_out = {}
    for arm in ("FIXED", "ADAPT", "CLASS"):
        beh_out[arm] = {}
        for L in L_GRID:
            beh_out[arm][str(L)] = behavior_at(L, arm)
    out["behavior"] = beh_out

    # 4) 落点判据差（ADAPT - CLASS，同一通道）
    gap = {}
    for L in L_GRID:
        a, c = beh_out["ADAPT"][str(L)], beh_out["CLASS"][str(L)]
        gap[str(L)] = {
            "score_gap": round(a["score"] - c["score"], 3),
            "ores_gap": round(a["ores"] - c["ores"], 3),
            "water_hits_gap": round(a["water_hits"] - c["water_hits"], 3),
        }
    out["landing_gap"] = gap

    # 5) 过度适应扫描（增益调谐参数 κ × L∈{0..3}）
    over = {}
    for L in OVER_LS:
        over[str(L)] = {}
        for kappa in KAPPAS:
            accs, _conf, _per = accuracy_at(L, "ADAPT", kappa)
            over[str(L)][str(kappa)] = {
                "acc": round(mean(accs), 4), "std": round(stdev(accs), 4)}
    out["over"] = over

    body = {k: v for k, v in out.items() if k != "meta"}
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True)
    out["meta"]["md5"] = hashlib.md5(payload.encode("utf-8")).hexdigest()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "light_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 打印 ----
    print("=== SEED-62: 感知 d* 光照适应实验 (docs/133 §六 实验①) ===")
    print("\n[1] affordance 准确率 vs 光照档位 L (41 seeds, 30 patch/seed, mean±std)")
    print(f"{'L':<4}{'FIXED':<18}{'ADAPT':<18}")
    for L in L_GRID:
        a, b = acc_out["FIXED"][str(L)], acc_out["ADAPT"][str(L)]
        print(f"{L:<4}{a['acc']:.3f}±{a['std']:.3f}      {b['acc']:.3f}±{b['std']:.3f}")
    print("\n[2] 感知 d*（准确率跌破 0.5 的档位差，从 L=10 往下数）")
    for arm in ("FIXED", "ADAPT"):
        d = dstar_out[arm]
        print(f"  {arm:<6} L_cross={d['L_cross']:.2f}  d*={d['dstar']:.2f}")
    print("\n[3] 适应延迟（L 7->2 阶跃，5% 收敛容差）")
    for alpha in (0.2, 0.35, 0.5):
        print(f"  alpha={alpha}: {delay_out[str(alpha)]['delay']} tick")
    print("\n[4] 行为（一局 30 patch：得分/挖矿/湿身，种子均值）")
    print(f"{'L':<4}{'FIXED':<22}{'ADAPT':<22}{'CLASS':<22}")
    for L in L_GRID:
        row = []
        for arm in ("FIXED", "ADAPT", "CLASS"):
            b = beh_out[arm][str(L)]
            row.append(f"{b['score']:+.1f}/{b['ores']:.1f}/{b['water_hits']:.1f}")
        print(f"{L:<4}{row[0]:<22}{row[1]:<22}{row[2]:<22}")
    print("\n[5] 落点差 ADAPT-CLASS（同一通道：知道不用=没看见）")
    for L in L_GRID:
        g = gap[str(L)]
        print(f"  L={L:<3} score={g['score_gap']:+.2f}  ores={g['ores_gap']:+.2f}  "
              f"water_hits={g['water_hits_gap']:+.2f}")
    print("\n[6] 过度适应扫描：准确率(L, kappa)")
    print(f"{'L':<4}" + "".join(f"{k:<8}" for k in KAPPAS))
    for L in OVER_LS:
        print(f"{L:<4}" + "".join(f"{over[str(L)][str(k)]['acc']:.3f}  " for k in KAPPAS))
    print(f"\nfull results -> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
