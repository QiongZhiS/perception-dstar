"""vision/color_constancy_temporal.py — 时间一致性判据 + 白平衡校正（docs/197 §四 1 正解）。

docs/197 缺口："何时校正"不能用单帧彩色占比——它声称 dark 噪声（frac 0.19）与
dark+yellow 真染色（frac 0.82）在 min/max_frac 门槛间混叠，需时间一致性（真染色稳定、
噪声抖动）。本实验发现并修正 docs/197 的三个错误：

  ① 排除带缺陷（假象源）：旧排除带 x40-320 让球轨迹后半段（x 至 454）泄漏进背景
     估计，"dark shift 抖动 0→138" 是球（红 H=0）污染的假象；完整排除球带后
     dark 的 shift 稳定在 88-91（多种子验证）。→ 时间一致性（shift 稳定性）单独
     区分不了 dark 与真染色。
  ② 阈值错误：estimate_color_temp 的 max_frac=0.5 把 yellow（frac≈1.0）误拒——代码
     里 yellow 实际从未校正（H 停在 83）。真正区分 dark（frac 0.41）与真染色
     （frac 0.667+）的是 frac 门槛，且要用**组合判据**（frac 真染色 + shift 时间
     一致性防突变）。
  ③ 校正动作错误：docs/197 用色相旋转（H−shift），把球体（H=0 恒红，乘法黄染的
     特征向量）推到 95（青）——校正反而破坏确认（yellow 球体红区 0.78→0.28）；
     正解是**通道级白平衡**（每通道增益，让背景回灰）：乘性增益对纯红球无色相影响，
     球保持红、背景回中性（球体红区 0.78→0.98）。
  ④ 度量错误：docs/196 的"H 漂移 83°"是 patch 中位被背景污染的假象，球体掩码内
     H 恒 0；确认度量用**球体掩码**（半径 18），不用含背景的 patch 中位。

实现（因果，逐帧）：
  ① 裸色温估计：背景彩色像素（S>100，完整排除球轨迹带）色相中位数 shift_i + 占比 frac_i
  ② 组合判据 TemporalGate：
     - frac > frac_gate（0.5：真染色信号足够，噪声染色 frac≈0.41 不够）
     - shift 时间一致性：EWMA 均值 mu + EWMA 方差 m2（alpha=0.05 长记忆），
       warmup 后 sqrt(m2) < sigma_gate（稳定，防突变/漂移）
     - 两条件 AND → 门开（校正）；否则固执（不校正，docs/101）
  ③ 校正（门开时）：背景 BGR 均值 m → 增益 g=(mG/mB, 1, mG/mR) → 全图乘 g
     门关时不校正（默认白光，docs/101 固执）

验收（docs/198 §三 1，度量=球体掩码红区占比）：
  dark        ：frac<0.5 → 门关 → 不校正 → 球体红区≈0.97、通过率 1.0
  yellow      ：frac 0.745 + shift 稳定 → 门开 → 白平衡 → 红区 0.78→0.98、通过 1.0
  dark+yellow ：frac 0.667+ + shift 稳定 → 门开 → 白平衡 → 红区 0.78→0.94、通过 1.0
  bright      ：frac≈0 → 无信号 → 不校正 → 通过率 1.0
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED

# 球轨迹带（真值：x=80+4i∈[80,436], y=180+0.6i∈[180,233]，半径18 → 完整排除）
BX0, BX1, BY0, BY1 = 40, 460, 130, 270


def raw_shift_frac(fr, sat_min=100):
    """裸色温估计：背景彩色像素色相中位数 + 占比 + 环形 MAD（真染色集中/噪声分散）。
    返回 (shift, frac, mad) 或 (None, frac, mad)。"""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    colored = (s > sat_min).astype(np.uint8)
    colored[BY0:BY1, BX0:BX1] = 0
    frac = float(colored.mean())
    if colored.sum() < 100:
        return None, frac, float('nan')
    vals = h[colored > 0].astype(float)
    med = float(np.median(vals))
    d = np.abs(vals - med)
    mad = float(np.median(np.minimum(d, 180.0 - d)))   # 环形 MAD（色相 0-180 环）
    return med, frac, mad


def bg_gain(fr, sat_min=100):
    """通道级白平衡增益：背景 BGR 均值 m，g=(mG/mB, 1, mG/mR)（G=亮度基准）。
    乘性增益对纯红球（G=B=0）无色相影响；背景回灰。无信号→None。"""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    colored = (s > sat_min).astype(np.uint8)
    colored[BY0:BY1, BX0:BX1] = 0
    if colored.sum() < 100:
        return None
    m = fr[colored > 0].mean(axis=0)   # BGR
    if m[1] < 5 or m[0] < 2 or m[2] < 2:
        return None
    return np.array([m[1] / m[0], 1.0, m[1] / m[2]])


class TemporalGate:
    """组合判据：真染色（色相集中度）AND 帧间一致性（docs/198 字面"帧间稳定该校正、
    帧间抖动该固执"）。

    两维签名（docs/199 §六 1 修正）：
     ① 真染色 vs 像素级噪声：背景彩色像素**色相分布集中度**（环形 MAD）——
        真染色让像素色相集中在色温附近（toy yellow MAD 3°、DAVIS 2-17°），
        像素级噪声让色相随机分散（toy dark MAD 47°）。比 frac 更本质：
        DAVIS 真实场景 frac 0.08-0.47（低）但色相集中（真染色），frac 门槛误拒。
     ② 帧间一致 vs 帧间突变：|Δshift| EWMA——渐变（黄昏色温帧间连续爬升）变化小
        该开，增益跳变（噪声）变化大该关；docs/197 的"EWMA 均值方差"把渐变误判
        为噪声（围绕移动均值），正解是帧间变化量。
    两条件 AND → 门开（校正）；否则固执（docs/101 越噪声越固执）。"""

    def __init__(self, mad_gate=20.0, alpha=0.05, warmup=12, delta_gate=3.0):
        self.mad_gate = mad_gate
        self.alpha = alpha
        self.warmup = warmup
        self.delta_gate = delta_gate
        self.d_mu = None       # 帧间 |Δshift| 的 EWMA
        self.prev = None       # 上一帧 shift
        self.n = 0

    def update(self, s, mad):
        """s: shift 或 None; mad: 背景色相环形 MAD（真染色集中/噪声分散）。开→可校正。"""
        if s is None or mad is None or mad > self.mad_gate:
            return False
        if self.prev is not None:
            d = abs(s - self.prev)
            if self.d_mu is None:
                self.d_mu = d
            else:
                self.d_mu += self.alpha * (d - self.d_mu)
        self.prev = s
        self.n += 1
        return self.open()

    def open(self):
        return self.n >= self.warmup and (self.d_mu is not None) and self.d_mu < self.delta_gate


def confirm_ball(frames, gate=None, mode='wb'):
    """球体掩码（半径18，真值位置）确认。mode='wb'：白平衡增益；mode='hue'：docs/197
    色相旋转（对照）；gate=None：不校正。返回 (通过率, 球H中位, 校正帧数, 球红区占比均值)。"""
    ok = 0
    hues = []
    fracs = []
    n_corr = 0
    for i, fr in enumerate(frames):
        rc = (80 + 4.0 * i, 180 + 0.6 * i)
        img = fr
        if gate is not None:
            if mode == 'hue':
                s, _, _ = raw_shift_frac(fr)
                gate.update(s, 0.0)   # 色相旋转对照：绕过 MAD（docs/197 用单帧占比但此处展示旋转本身）
                if gate.open():
                    sh = s
                    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                    hh = (hsv[:, :, 0].astype(float) - sh) % 180
                    hsv[:, :, 0] = hh.astype(np.uint8)
                    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    n_corr += 1
            else:  # wb：组合判据
                s, _, mad = raw_shift_frac(fr)
                g = bg_gain(fr)
                gate.update(s, mad)
                if g is not None and gate.open():
                    img = np.clip(fr.astype(np.float32) * g, 0, 255).astype(np.uint8)
                    n_corr += 1
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s = hsv[:, :, 0], hsv[:, :, 1]
        r = 18
        cx, cy = int(rc[0]), int(rc[1])
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue
        rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        colored = rs > 60
        if colored.sum() < 10:
            continue
        frac = float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())
        fracs.append(frac)
        hues.append(float(np.median(rh[colored])))
        ok += frac > 0.2
    return (ok / max(len(frames), 1),
            (float(np.median(hues)) if hues else float('nan')),
            n_corr,
            (float(np.mean(fracs)) if fracs else float('nan')))


def gate_trace(frames, gate):
    trace = []
    for i, fr in enumerate(frames):
        s, _, mad = raw_shift_frac(fr)
        gate.update(s, mad)
        trace.append((i, s, mad, gate.open()))
    return trace


def summarize(trace):
    ch = ''.join('C' if t[3] else '_' for t in trace)
    return f"门开帧 {sum(1 for t in trace if t[3])}/{len(trace)}  时序: {ch}"


print("== 时间一致性判据 + 白平衡校正（docs/197 §四 1 正解）==")
print("== 度量修正：球体掩码（半径18），非含背景的 patch 中位 ==")
print(f"{'模式':14s} {'无校正红区':>9s} {'无校正通过':>9s} | "
      f"{'色相旋转红区':>11s} {'旋转通过':>8s} | "
      f"{'白平衡红区':>10s} {'白平衡通过':>10s}")
results = {}
for mode in ['bright', 'dark', 'yellow', 'dark+yellow']:
    frames = make_scene(mode)
    r0, h0, _, f0 = confirm_ball(frames)                        # 无校正
    g1 = TemporalGate()
    r1, h1, nc1, f1 = confirm_ball(frames, g1, mode='hue')      # docs/197 色相旋转
    g2 = TemporalGate()
    r2, h2, nc2, f2 = confirm_ball(frames, g2, mode='wb')       # 组合判据+白平衡
    results[mode] = (r0, h0, f0, r1, h1, f1, r2, h2, f2, nc2)
    print(f"{mode:14s} {f0:9.3f} {r0:9.3f} | {f1:11.3f} {r1:8.3f} | "
          f"{f2:10.3f} {r2:10.3f}")

print("\n== 组合判据逐帧状态（_=固执不校正，C=开放校正）+ MAD ==")
for mode in ['bright', 'dark', 'yellow', 'dark+yellow']:
    frames = make_scene(mode)
    t = gate_trace(frames, TemporalGate())
    mads = [f"{t2[2]:.0f}" for t2 in t[::15]]
    print(f"  {mode:14s} {summarize(t)}   MAD@15帧: {mads}")

print("\n== 验收（docs/198 §三 1，度量=球体掩码红区占比）==")
r_d, _, f_d, *_ = results['dark']
r_dy, _, _, _, _, _, r_dy2, _, f_dy, _ = results['dark+yellow']
r_y, _, _, _, _, _, r_y2, _, f_y, _ = results['yellow']
r_b, _, _, *_ = results['bright']
print(f"  dark        不校正(红区≈1.0、通过率 1.0): 红区 {f_d:.3f} 通过 {r_d:.3f}"
      f"  {'✓' if f_d > 0.9 and r_d == 1.0 else '✗'}")
print(f"  dark+yellow 白平衡校正(红区提升、通过率 1.0): 红区 {f_dy:.3f} 通过 {r_dy2:.3f}"
      f"  {'✓' if f_dy > 0.9 and r_dy2 == 1.0 else '✗'}")
print(f"  yellow      白平衡校正(红区提升、通过率 1.0): 红区 {f_y:.3f} 通过 {r_y2:.3f}"
      f"  {'✓' if f_y > 0.9 and r_y2 == 1.0 else '✗'}")
print(f"  bright      不校正(通过率 1.0): 通过 {r_b:.3f}  {'✓' if r_b == 1.0 else '✗'}")

print("\n== 对 docs/197 的修正（四个错误）==")
print("  ① 'dark shift 抖动'是旧排除带球泄漏假象——完整排除后 dark shift 稳定 88-91，")
print("     时间一致性单独区分不了 dark 与真染色；真区分是色相分布集中度（MAD）。")
print("  ② estimate_color_temp 的 max_frac=0.5 把 yellow(frac≈1.0)误拒——代码里 yellow")
print("     实际从未校正(H 停 83)；正解是 MAD 门槛 AND 帧间一致性组合。")
print("  ③ 色相旋转(H−shift)把球体 0→95 推出红区(yellow 红区 0.78→0.28)；正解是")
print("     通道级白平衡增益——乘性增益对纯红球无色相影响，球保持红、背景回灰。")
print("  ④ 'H 漂移 83°'(docs/196)是 patch 中位假象——球体掩码内 H 恒 0，乘法黄染对")
print("     纯红球是特征向量；确认度量用球体掩码，不用含背景的 patch 中位。")

print("\n== docs/101 对照：记忆长短（alpha）与固执度 ==")
print(f"{'alpha':8s} {'dark门开':>8s} {'dark红区':>8s} {'dark+y门开':>10s} {'dark+y红区':>10s}")
for alpha in [0.5, 0.2, 0.1, 0.05]:
    g = TemporalGate(alpha=alpha)
    _, _, _, fd = confirm_ball(make_scene('dark'), g, mode='wb')
    g = TemporalGate(alpha=alpha)
    _, _, nc, fdy = confirm_ball(make_scene('dark+yellow'), g, mode='wb')
    t = summarize(gate_trace(make_scene('dark'), TemporalGate(alpha=alpha)))
    print(f"{alpha:8.2f} {t.split()[1]:>8s} {fd:8.3f} {nc:10d} {fdy:10.3f}")

print("\n判据：帧间稳定才校正（|Δshift|）+ 真染色才校正（色相 MAD 集中）——")
print("docs/101 越噪声越固执在颜色轴的落点：像素级噪声（MAD 分散）该固执，")
print("帧间突变（|Δ| 大）该固执，真染色（MAD 集中且帧间稳定）该开放。")
