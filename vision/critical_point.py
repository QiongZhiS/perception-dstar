"""vision/critical_point.py — 临界点实验（docs/231 §四.1 命门实验，交付 docs/232）。

把预测-残差回路（docs/227 Transduction2D 同款机制：对数域双 EWMA 背景预测 +
自适应阈值 + 事件累积）放进参数化环境复杂度梯度（L0 静态 -> L6 近乎不可预测），
逐级测量三个可操作度量：

  M1 预测误差轨迹 : 每窗口 mean|L - bg_fast| 序列 + 事件率 + 趋势斜率
  M2 残差利用率  : util_attention（超死区像素占比）; util_learning（结构学习响应窗口占比：
                    窗口内发生 pattern 节点匹配/创建的占比——残差被用于结构学习的比例）
  M3 参数 vs 结构 : param（连续参数位移 Σ|Δθ|+Σ|Δdb|，只抬不降有界）
                    SC1 新记忆条目创建数（pattern 表新增节点）
                    SC2 新增技能节点数（命中 >= hits_min 的稳定节点）
                    SC_late 后半程仍涌现的稳定节点数（创建>=总窗口/2 且命中>=hits_min，自驱动信号）
                    SC3 不可逆阈值跳变数（θ 一步 >25%，只抬不降 -> 不可逆）
                    SC4 休眠后保留条目数（刺激消失 >= persist_win 窗口仍在表内）

判据（预注册，docs/232 §1.4）：idle / self_driven / partial / noise_amp / tuned。
临界点 = 升序第一个 self_driven 且非最高级；F3 触发 = 无任何级 self_driven。
统计外壳（docs/228 模式）：多种子（--n-seeds）、jitter 强度抖动、JSON 归档
（vision/out/results/cp_<tag>.json）、每 (级,种子) 写 checkpoint（ckpt_cp_<hash>.json，
--resume 断点续跑）、跨种子 mean±SD + bootstrap 95% CI。

安全纪律（docs/228 同款）：脚本 stdout 只输出 ASCII 标签 + 每行一个数字的 R_*
摘要块（顺序固定，见 SUMMARY_LINES）。禁止把日志/JSON 原文送进上下文。

用法：
  python vision/critical_point.py --levels 0,1,2,3,4,5,6 --n-seeds 8 --tag cp
  python vision/critical_point.py --levels 5 --n-seeds 1 --frames 120 --tag timing
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
JITTER = 0.25          # 种子间强度抖动（噪声 σ 与游走幅度 ±25% 乘性，docs/228 主协议）
N_BOOT = 2000
BOOT_SEED = 20260828

# ---------------- 环境复杂度梯度（docs/232 §1.1） ----------------
# 参数化四元组 (n_obj, speed, regularity, noise) + light 事件。
LEVELS = {
    0: dict(name="static",            n_obj=0, speed=0.0, regularity=1.0, noise=0.8, light="none"),
    1: dict(name="regular_slow",      n_obj=1, speed=0.5, regularity=1.0, noise=0.8, light="none"),
    2: dict(name="regular_fast",      n_obj=2, speed=1.0, regularity=1.0, noise=0.8, light="none"),
    3: dict(name="regular_flicker",   n_obj=2, speed=1.0, regularity=1.0, noise=0.8, light="dips"),
    4: dict(name="noisy_regular",     n_obj=2, speed=1.0, regularity=1.0, noise=3.0, light="dips"),
    5: dict(name="regime_switch",     n_obj=1, speed=1.0, regularity=0.7, noise=3.0, light="dips",
            regimes=[0.4, 2.2, 1.0]),
    6: dict(name="near_unpredictable", n_obj=3, speed=1.2, regularity=0.2, noise=8.0, light="flashes"),
}
OBJ_RADIUS = 14
OBJ_GRAY = 255.0   # 高对比目标：log 域对比 ≈ log(256/65) ≈ 1.37，事件信号大而稳


def make_scene(level_idx, seed, n_frames=240, width=160, height=120, fps=30,
               jitter=JITTER):
    """生成 (级,种子) 的灰度帧序列（uint8 list）。确定性：rng 由 (seed, level) 派生。

    regularity=1.0: 纯 Lissajous；regularity<1.0: 叠加平滑随机游走偏移
    （幅度 ∝ 1-regularity）。L5 (regime_switch)：目标带 3 段速度 regime
    （speed 乘子 [0.5, 2.0, 1.0]，每 ~80 帧切换，相位重抽——段内规律可预测、
    切换点不可预测 = "部分可预测但永远有落空"）。n_obj=3 时末目标是随机游走 +
    跳变。light: none / dips（两处 0.5s 降幅，docs/227 闪变段同款）/
    flashes（4 次随机短闪）。"""
    cfg = LEVELS[level_idx]
    rng = np.random.default_rng(seed * 7919 + level_idx * 104729 + 13)
    noise_mult = rng.uniform(1 - jitter, 1 + jitter) if jitter > 0 else 1.0
    sigma = cfg["noise"] * noise_mult
    reg = cfg["regularity"]
    spd = cfg["speed"]
    regimes = cfg.get("regimes", None)

    # 背景棋盘格（docs/227 make_demo_video 同款纹理，加深以抬升目标对比）
    bg = np.zeros((height, width), np.float32)
    step = 24
    for y in range(0, height, step):
        for x in range(0, width, step):
            bg[y:y + step, x:x + step] = 64 if ((x // step) + (y // step)) % 2 == 0 else 96

    # 目标轨迹参数（按 speed 标度；每种子随机相位；匀速圆周运动——速度大小恒定，
    # 速度档不随相位漂移，L5 regime 换档在任意相位下都产生稳定的速度档差异）
    objs = []
    for i in range(cfg["n_obj"]):
        if regimes is not None:
            kind = "regime"
        else:
            kind = "random" if (cfg["n_obj"] >= 3 and i == cfg["n_obj"] - 1) else \
                ("jittery" if reg < 1.0 else "regular")
        f1 = spd * (0.12 + 0.02 * rng.uniform(0, 1))
        r = (28 + 8 * spd) * rng.uniform(0.95, 1.05)
        ph = rng.uniform(0, 2 * np.pi)
        objs.append(dict(kind=kind, f1=f1, r=r, ph=ph,
                         vx=spd * rng.uniform(2, 4),
                         jump_t=[rng.integers(30, n_frames - 40) for _ in range(2)]))
    # L5 regime 切换点（确定性：每 ~80 帧一切，相位重抽）
    regime_bounds = [0, n_frames // 3, 2 * n_frames // 3, n_frames] if regimes else None

    # 光照事件
    dips = []
    flashes = []
    if cfg["light"] == "dips":
        dips = [(45, 60), (130, 145)]                  # 各 0.5s 降幅（0.5 亮度）；第二处放在
        # L5 regime 2 中段（w13-14），保持 regime 3（w16-23）干净 → 后期节点可稳定累积命中
    elif cfg["light"] == "flashes":
        for _ in range(4):
            t0 = int(rng.integers(20, n_frames - 10))
            flashes.append((t0, min(t0 + 3, n_frames), float(rng.uniform(0.4, 0.7))))

    frames = []
    joff = [0.0, 0.0, 0.0]        # 平滑随机游走偏移（per object）
    rw = [dict(x=float(rng.uniform(20, width - 20)),
               y=float(rng.uniform(20, height - 20))) for _ in range(3)]
    jamp = (1.0 - reg) * 8.0 * noise_mult
    cur_reg = [-1, -1, -1]
    for t in range(n_frames):
        img = bg.copy()
        for i, o in enumerate(objs):
            if o["kind"] == "random":
                r = rw[i]
                r["x"] += o["vx"] + rng.normal(0, 3.0 * noise_mult)
                r["y"] += rng.normal(0, 3.0 * noise_mult)
                if t in o["jump_t"]:
                    r["x"] += float(rng.choice([-1, 1]) * 30)
                cx, cy = r["x"], r["y"]
            else:
                if regimes is not None:
                    # regime 切换：速度乘子换档 + 相位重抽（切换点不可预测）
                    rg = 0
                    for b in range(3):
                        if regime_bounds[b] <= t < regime_bounds[b + 1]:
                            rg = b
                    if rg != cur_reg[i]:
                        cur_reg[i] = rg
                        o["ph"] = rng.uniform(0, 2 * np.pi)
                    f1 = o["f1"] * regimes[rg]
                else:
                    f1 = o["f1"]
                tt = t / fps
                bx = width / 2 + o["r"] * np.cos(2 * np.pi * f1 * tt + o["ph"])
                by = height / 2 + o["r"] * np.sin(2 * np.pi * f1 * tt + o["ph"])
                if o["kind"] == "jittery":
                    joff[i] = 0.9 * joff[i] + rng.normal(0, jamp)
                    bx += joff[i]
                    by += 0.9 * joff[i] + rng.normal(0, jamp) * 0.7
                cx, cy = bx, by
            cv2_circle(img, int(cx), int(cy), OBJ_RADIUS, OBJ_GRAY)
        light = 1.0
        for lo, hi in dips:
            if lo <= t < hi:
                light = 0.5
        for f0, f1_, amp in flashes:
            if f0 <= t < f1_:
                light = amp
        img = img * light
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames


def cv2_circle(img, cx, cy, r, val):
    """往灰度图画实心圆（避免依赖 cv2 绘制：纯 numpy 掩码）。"""
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img[mask] = val
    return img


# ---------------- 预测-残差回路（docs/232 §1.2/§1.3） ----------------
class CPLoop:
    """快/慢 EWMA 背景预测 + 自适应阈值（只抬不降）+ pattern 记忆（结构）。

    预测任务（每级相同）：prediction = bg_fast；误差 = |L - bg_fast|。
    参数（固定槽位，t=0 即存在）：thresh(θ), deadband(db) —— 只抬不降、有界。
    结构（离散记忆表，t=0 不存在）：pattern 节点 = 运动能量档（窗口事件量分档）
    -> 节点；匀速目标能量稳定，速度换档（L5 regime 切换）能量跳档 -> 新条目。"""

    def __init__(self, alpha_fast=0.5, alpha_slow=0.03, thresh=0.15, deadband=0.015,
                 k_theta=6.0, k_db=1.5, thresh_max=0.6, db_max=0.15,
                 window=10, k_consist=3, hits_min=3, persist_win=5,
                 energy_bins=(150.0, 450.0, 1000.0, 2500.0)):
        self.a_fast = alpha_fast
        self.a_slow = alpha_slow
        self.thresh = thresh
        self.deadband = deadband
        self.k_theta = k_theta
        self.k_db = k_db
        self.thresh_max = thresh_max
        self.db_max = db_max
        self.window = window
        self.k_consist = k_consist
        self.hits_min = hits_min
        self.persist_win = persist_win
        self.energy_bins = energy_bins
        self.bg_fast = None
        self.bg_slow = None
        self.prev_L = None
        self.sigma_hat = None
        # 每窗口聚合缓冲
        self.mae = []            # 每窗口 mean|L-bg|
        self.att = []            # 每窗口 超死区像素占比
        self.ev = []             # 每窗口 事件率
        self.theta_trace = []
        self.db_trace = []
        self.sc1_cum = []        # 每窗口累计节点数
        self.energy_trace = []   # 每窗口事件量（union，px）
        self._frame_buf = []     # 当前窗口内的帧残差/事件
        self._param_disp = 0.0
        self._prev_theta = None
        self._prev_db = None
        self._n_steps = 0        # SC3：θ 一步 >25% 的窗口数
        self._n_learn = 0        # 学习响应窗口计数
        self.nodes = {}          # sig -> {hits, created, last_active}
        self._sig_seen = {}      # sig -> 连续出现窗口数
        self._win = 0
        self._ev_win = None

    def step(self, gray):
        """处理一帧；返回 per-frame dict（供窗口聚合）。"""
        L = np.log(np.maximum(gray.astype(np.float32), 1.0))
        if self.bg_fast is None:
            self.bg_fast = L.copy()
            self.bg_slow = L.copy()
            self.prev_L = L.copy()
            self.sigma_hat = 0.0
            return dict(mae=0.0, att=0.0, ev=0.0, theta=self.thresh, db=self.deadband)

        # 噪声估计（MAD of ΔL，docs/185 同款）+ 自适应阈值（只抬不降）
        d = L - self.prev_L
        sig = float(np.median(np.abs(d - np.median(d))) / 0.6745)
        self.sigma_hat = 0.15 * sig + 0.85 * self.sigma_hat
        theta = float(np.clip(max(self.thresh, self.k_theta * self.sigma_hat),
                              self.thresh, self.thresh_max))
        db = float(np.clip(max(self.deadband, self.k_db * self.sigma_hat),
                           self.deadband, self.db_max))
        self.prev_L = L

        # 背景预测 + 残差
        self.bg_fast = self.a_fast * L + (1 - self.a_fast) * self.bg_fast
        self.bg_slow = self.a_slow * L + (1 - self.a_slow) * self.bg_slow
        c = L - self.bg_fast
        r = np.abs(c)
        rd = np.maximum(r - db, 0.0)
        ev_mask = rd > theta
        mae = float(r.mean())
        att = float(np.mean(r > db))
        ev = float(ev_mask.mean())

        # 参数位移（连续）与 SC3（θ 一步 >25%，只抬不降 => 不可逆）
        if self._prev_theta is not None:
            self._param_disp += abs(theta - self._prev_theta)
            self._param_disp += abs(db - self._prev_db)
            if self._prev_theta > 0 and theta > 1.25 * self._prev_theta:
                self._n_steps += 1
        self._prev_theta = theta
        self._prev_db = db

        if self._ev_win is None:
            self._ev_win = ev_mask.copy()
        else:
            self._ev_win |= ev_mask
        self._frame_buf.append(dict(mae=mae, att=att, ev=ev, theta=theta, db=db))
        if len(self._frame_buf) >= self.window:
            self._on_window()
        return dict(mae=mae, att=att, ev=ev, theta=theta, db=db)

    def _window_sig(self, ev_win):
        """签名 = 运动能量档：窗口事件总量（union px）的分档（对数尺）。

        运动能量 ∝ 目标扫过的新像素量 ∝ 速度——匀速目标能量稳定；L5 regime 换档
        改变速度 → 能量档跳变 → 新条目；闪变/交叠的单窗口异常被 k_consist 一致性
        过滤（需连续 3 窗口同档才建节点）。事件不足（<10px）返回 None（无签名，
        静态/无事件窗口不产生条目）。"""
        n = int(ev_win.sum())
        if n < 10:
            return None
        eb = 0
        for b, bmax in enumerate(self.energy_bins):
            if n >= bmax:
                eb = b + 1
        return (eb,)

    def _on_window(self):
        """窗口收尾：pattern 更新 + 窗口级指标归档。"""
        ev_win = self._ev_win if self._ev_win is not None else \
            np.zeros((120, 160), bool)
        mae_w = float(np.mean([f["mae"] for f in self._frame_buf]))
        att_w = float(np.mean([f["att"] for f in self._frame_buf]))
        ev_w = float(np.mean([f["ev"] for f in self._frame_buf]))
        theta_w = float(self._frame_buf[-1]["theta"])
        db_w = float(self._frame_buf[-1]["db"])
        self.mae.append(mae_w)
        self.att.append(att_w)
        self.ev.append(ev_w)
        self.theta_trace.append(theta_w)
        self.db_trace.append(db_w)

        # pattern：连续 k_consist 窗口同一签名 -> 创建节点（新记忆条目）
        learned = False
        sig = self._window_sig(ev_win)
        self.energy_trace.append(int(ev_win.sum()))
        if sig is not None:
            if sig in self.nodes:
                self.nodes[sig]["hits"] += 1
                self.nodes[sig]["last_active"] = self._win
                learned = True
            elif self._sig_seen.get(sig, 0) >= self.k_consist - 1:
                self.nodes[sig] = dict(hits=1, created=self._win,
                                       last_active=self._win)
                learned = True
            self._sig_seen[sig] = self._sig_seen.get(sig, 0) + 1
        for sig_k in list(self._sig_seen.keys()):
            if sig_k != sig:
                self._sig_seen[sig_k] = 0
        if learned:
            self._n_learn += 1
        self.sc1_cum.append(len(self.nodes))
        self._win += 1
        self._frame_buf = []
        self._ev_win = None

    def finalize(self, n_windows):
        """运行收尾：SC 指标 + 轨迹 + 汇总（docs/232 §1.3）。"""
        if self._frame_buf:
            self._on_window()
        sc1 = len(self.nodes)
        sc2 = sum(1 for nd in self.nodes.values() if nd["hits"] >= self.hits_min)
        sc_late = sum(1 for nd in self.nodes.values()
                      if nd["created"] >= n_windows / 2 and nd["hits"] >= self.hits_min)
        sc4 = sum(1 for nd in self.nodes.values()
                  if nd["last_active"] <= n_windows - 1 - self.persist_win)
        q = max(1, n_windows // 4)
        mae_arr = np.asarray(self.mae, float)
        slope = float(mae_arr[-q:].mean() - mae_arr[:q].mean()) if len(mae_arr) >= 2 * q else 0.0
        pin = float(np.mean(np.asarray(self.theta_trace, float) >= 0.999 * self.thresh_max)) \
            if self.theta_trace else 0.0
        util_learning = self._n_learn / max(1, n_windows)
        return {
            "mae_trace": [round(v, 6) for v in self.mae],
            "ev_trace": [round(v, 6) for v in self.ev],
            "att_trace": [round(v, 6) for v in self.att],
            "theta_trace": [round(v, 6) for v in self.theta_trace],
            "sc1_cum": self.sc1_cum,
            "energy_trace": self.energy_trace,
            "mae_mean": round(float(np.mean(mae_arr)), 6),
            "mae_slope": round(slope, 6),
            "ev_mean": round(float(np.mean(self.ev)), 6),
            "att_mean": round(float(np.mean(self.att)), 6),
            "util_learning": round(util_learning, 4),
            "param_disp": round(self._param_disp, 6),
            "pin_frac": round(pin, 4),
            "theta_mean": round(float(np.mean(self.theta_trace)), 6),
            "sc1": sc1, "sc2": sc2, "sc_late": sc_late, "sc3": self._n_steps, "sc4": sc4,
            "n_nodes_detail": {str(k): v for k, v in self.nodes.items()},
        }


def run_level(level_idx, seed, n_frames, width, height, fps, window, jitter,
              loop_kwargs=None):
    """跑 (级,种子) 一次完整运行，返回指标 dict。"""
    frames = make_scene(level_idx, seed, n_frames=n_frames, width=width, height=height,
                        fps=fps, jitter=jitter)
    loop = CPLoop(window=window, **(loop_kwargs or {}))
    for g in frames:
        loop.step(g)
    n_windows = max(1, n_frames // window)
    out = loop.finalize(n_windows)
    out["seed"] = seed
    out["level"] = level_idx
    out["frames"] = n_frames
    return out


# ---------------- 统计外壳（docs/228 模式） ----------------
def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return float(a.mean()), sd


def bootstrap_ci(vals, n=N_BOOT, alpha=0.05):
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty(n)
    for i in range(n):
        means[i] = a[rng.integers(0, len(a), size=len(a))].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def classify(sc1_frac, sc2_frac, sc_late_frac, stab, pin_mean, att_mean, ev_mean, util_mean):
    """分级判据（docs/232 §1.4 预注册）。"""
    if sc1_frac == 0.0 and ev_mean < 1e-3 and util_mean < 0.1 and pin_mean < 0.9:
        return "idle"
    if sc2_frac >= 0.6 and sc_late_frac >= 0.6 and stab >= 0.3 and pin_mean < 0.9:
        return "self_driven"
    if pin_mean >= 0.9 or (att_mean >= 0.05 and sc2_frac < 0.6):
        return "noise_amp"
    if sc1_frac >= 0.6:
        return "partial"
    return "tuned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3,4,5,6",
                    help="逗号分隔环境级索引（0-6）")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="cp",
                    help="结果/checkpoint 文件名后缀 -> cp_<tag>.json / ckpt_cp_<hash>.json")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    t0 = time.time()

    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "fps": args.fps, "size": [args.width, args.height],
           "window": args.window, "jitter": args.jitter, "tag": args.tag,
           "loop": {"alpha_fast": 0.5, "alpha_slow": 0.03, "thresh": 0.15,
                    "deadband": 0.015, "k_theta": 6.0, "k_db": 1.5,
                    "thresh_max": 0.6, "db_max": 0.15, "k_consist": 3,
                    "hits_min": 3, "persist_win": 5,
                    "energy_bins": [150.0, 450.0, 1000.0, 2500.0]}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_cp_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    for lv in levels:
        for seed in seeds:
            key = "%d_%d" % (lv, seed)
            if key in per_unit:
                continue
            r = run_level(lv, seed, args.frames, args.width, args.height,
                          args.fps, args.window, args.jitter)
            per_unit[key] = r
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)

    # ---- 跨种子聚合 + 分级 ----
    levels_out = {}
    for lv in levels:
        rs = [per_unit["%d_%d" % (lv, s)] for s in seeds]
        def col(key):
            return [r[key] for r in rs]
        sc1s, sc2s, scls, sc3s, sc4s = col("sc1"), col("sc2"), col("sc_late"), \
            col("sc3"), col("sc4")
        sc1_frac = float(np.mean([v >= 1 for v in sc1s]))
        sc2_frac = float(np.mean([v >= 1 for v in sc2s]))
        sc_late_frac = float(np.mean([v >= 1 for v in scls]))
        stab = float(np.mean(sc2s) / max(1.0, float(np.mean(sc1s))))
        cls = classify(sc1_frac, sc2_frac, sc_late_frac, stab,
                       float(np.mean(col("pin_frac"))), float(np.mean(col("att_mean"))),
                       float(np.mean(col("ev_mean"))), float(np.mean(col("util_learning"))))
        levels_out[lv] = {
            "name": LEVELS[lv]["name"],
            "params": {k: LEVELS[lv][k] for k in ("n_obj", "speed", "regularity", "noise", "light")},
            "per_seed": rs,
            "mean_sd": {k: list(mean_sd(col(k))) for k in
                        ("mae_mean", "mae_slope", "ev_mean", "att_mean", "util_learning",
                         "param_disp", "pin_frac", "theta_mean", "sc1", "sc2", "sc_late",
                         "sc3", "sc4")},
            "bootstrap_ci95": {"sc2": list(bootstrap_ci(sc2s))},
            "fracs": {"sc1": sc1_frac, "sc2": sc2_frac, "sc_late": sc_late_frac,
                      "stab": stab},
            "class": cls,
        }

    # ---- 临界点 / F3 判定（预注册） ----
    cp_level = -1
    cp_class = None
    for lv in levels:
        if levels_out[lv]["class"] == "self_driven":
            cp_level = lv
            cp_class = "self_driven"
            break
    highest = max(levels)
    verdict_note = ""
    if cp_level >= 0:
        if cp_level == highest:
            verdict_note = "self_driven 只在最高复杂度级出现，非'中间态'——临界点主张不成立（需中间级）"
        else:
            verdict_note = "临界点候选：L%d %s 出现稳定新增结构（H3 直接证据形态）" % (
                cp_level, LEVELS[cp_level]["name"])
    else:
        verdict_note = "无任何级 self_driven -> F3 触发（H3 作废），或仅有 partial（创建无固化，偏证伪）"
    f3 = "triggered" if cp_level < 0 else "not_triggered"
    verdict = {"cp_level": cp_level, "cp_class": cp_class,
               "f3": f3, "note": verdict_note}

    out = {
        "artifact": "critical_point",
        "doc_ref": "docs/231 §四.1, docs/232",
        "config": cfg,
        "levels": {str(k): v for k, v in levels_out.items()},
        "verdict": verdict,
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "cp_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定）----
    # 每级 13 行（NAME 无数字）：
    #   R_L<n>_NAME=<name>
    #   R_L<n>_MAE=<mean>         1
    #   R_L<n>_MAE_SD=<sd>        2
    #   R_L<n>_SLOPE=<mean>       3
    #   R_L<n>_EV=<mean>          4
    #   R_L<n>_ATT=<mean>         5
    #   R_L<n>_UTIL=<mean>        6
    #   R_L<n>_PARAM=<mean>       7
    #   R_L<n>_PIN=<mean>         8
    #   R_L<n>_SC1=<mean>         9
    #   R_L<n>_SC2=<mean>         10
    #   R_L<n>_SC2_SD=<sd>        11
    #   R_L<n>_SCLATE=<mean>      12
    #   R_L<n>_SC3=<mean>         13
    #   R_L<n>_SC4=<mean>         14
    #   R_L<n>_CLASS=<ascii>      （无数字）
    #   R_L<n>_SC2FRAC=<frac>     15
    # 收尾：R_LEVELS / R_SEEDS / R_CP_LEVEL / R_CP_CLASS / R_F3 / R_ELAPSED
    print("R_LEVELS=%d" % len(levels))
    print("R_SEEDS=%d" % len(seeds))
    for lv in levels:
        o = levels_out[lv]
        ms = o["mean_sd"]
        print("R_L%d_NAME=%s" % (lv, o["name"]))
        print("R_L%d_MAE=%.6f" % (lv, ms["mae_mean"][0]))
        print("R_L%d_MAE_SD=%.6f" % (lv, ms["mae_mean"][1]))
        print("R_L%d_SLOPE=%.6f" % (lv, ms["mae_slope"][0]))
        print("R_L%d_EV=%.6f" % (lv, ms["ev_mean"][0]))
        print("R_L%d_ATT=%.6f" % (lv, ms["att_mean"][0]))
        print("R_L%d_UTIL=%.6f" % (lv, ms["util_learning"][0]))
        print("R_L%d_PARAM=%.6f" % (lv, ms["param_disp"][0]))
        print("R_L%d_PIN=%.6f" % (lv, ms["pin_frac"][0]))
        print("R_L%d_SC1=%.4f" % (lv, ms["sc1"][0]))
        print("R_L%d_SC2=%.4f" % (lv, ms["sc2"][0]))
        print("R_L%d_SC2_SD=%.4f" % (lv, ms["sc2"][1]))
        print("R_L%d_SCLATE=%.4f" % (lv, ms["sc_late"][0]))
        print("R_L%d_SC3=%.4f" % (lv, ms["sc3"][0]))
        print("R_L%d_SC4=%.4f" % (lv, ms["sc4"][0]))
        print("R_L%d_CLASS=%s" % (lv, o["class"]))
        print("R_L%d_SC2FRAC=%.4f" % (lv, o["fracs"]["sc2"]))
    print("R_CP_LEVEL=%d" % cp_level)
    print("R_CP_CLASS=%s" % (cp_class if cp_class else "none"))
    print("R_F3=%s" % f3)
    print("R_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

