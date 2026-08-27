"""vision/keep_reject_continuous.py — 连续失败率下 P3 单调性是否成立（docs/200 边界 #1）。

docs/200 诚实边界 #1：toy 染色是二值开关（红↔蓝、30 帧）——真实"他者的 不"是
连续的（确认失败率、部分遮挡、颜色渐变），延迟-被拒曲线在连续失败下需重标定。

本文件把"被拒"从二值开关推广为连续失败率 p（每帧以概率 p 染色成蓝 = 确认失败率），
测 P3（重信任延迟随累计被拒帧数单调增）是否跨失败模式成立，以及延迟-被拒曲线的
形状是否模式无关（决定量是"被拒的累计"还是"时间长度"）。

模式（染色段 60 帧，恢复后全红 40 帧）：
  binary  : 每帧 p=1.0 染色（docs/200 原版）
  partial : 每帧以概率 p 染色（p=0.5 / 0.8）
  ramp    : 被拒概率 0.1→1.0 线性爬升（"他者的不"逐渐变强 = 渐变）

判据：
  R1 单调性：所有模式下，重信任延迟随累计被拒帧数 rejected 单调增
  R2 模式无关：以 rejected（而非时间长度）为横轴时，各模式延迟曲线对齐——
     若对齐，说明决定量是被拒历史（计数），不是时间长度（docs/200"失败入史"的
     机制含义：纪念按被拒计数累积，不按墙钟时间累积）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import W, H, RED


def make_ball_frac(total, rej_prob_fn, rng_seed=42):
    """红球匀速直线；染色段内每帧按 rej_prob_fn(i) 概率染蓝（连续失败率）。"""
    r0 = np.array([80.0, 180.0]); rv = np.array([4.0, 0.6])
    rng = np.random.default_rng(rng_seed)
    frames, rej_ground = [], []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * i
        is_rej = rng.random() < rej_prob_fn(i)
        color = (200, 0, 0) if is_rej else (0, 0, 200)  # 蓝=被拒 红=通过
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, color, -1)
        frames.append(img)
        rej_ground.append(is_rej)
    return frames, rej_ground


def ball_frac(fr, i):
    """球体掩码内红区占比（docs/195 同款）。"""
    rc = (80 + 4.0 * i, 180 + 0.6 * i)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    r = 18
    cx, cy = int(rc[0]), int(rc[1])
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return 0.0
    rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0
    return float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())


class Keep:
    """第 6 层保留系统（docs/200 原版）：期望恒红；失败入史；重信任需连续 K 帧，
    K 随累计被拒帧数增长（纪念加深）。"""

    def __init__(self, red_frac_thr=0.2, base_k=1, grow=3.0):
        self.thr = red_frac_thr
        self.base_k = base_k
        self.grow = grow
        self.rejected = 0
        self.trusted = True
        self.consec = 0

    def k(self):
        return self.base_k + int(self.rejected / self.grow)

    def step(self, frac_red):
        ok = frac_red > self.thr
        if ok:
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k():
                    self.trusted = True
        else:
            self.rejected += 1
            self.trusted = False
            self.consec = 0
        return self.trusted and ok


def run(frames):
    sys2 = Keep()
    seq = []
    for i, fr in enumerate(frames):
        seq.append(sys2.step(ball_frac(fr, i)))
    return seq, sys2.rejected


def retrust_delay(frames, seq, seg_end):
    """染色段结束后首次确认通过的延迟（帧数）；无则 None。"""
    for i in range(seg_end, len(frames)):
        if seq[i]:
            return i - seg_end + 1
    return None


SEG = (30, 90)          # 染色段 60 帧
TOTAL = 130             # 染色段 + 恢复段 40 帧


def prob_fn(mode):
    if mode == 'binary':
        return lambda i: 1.0 if SEG[0] <= i < SEG[1] else 0.0
    if mode == 'partial_05':
        return lambda i: 0.5 if SEG[0] <= i < SEG[1] else 0.0
    if mode == 'partial_08':
        return lambda i: 0.8 if SEG[0] <= i < SEG[1] else 0.0
    if mode == 'ramp':
        def f(i):
            if SEG[0] <= i < SEG[1]:
                t = (i - SEG[0]) / (SEG[1] - SEG[0])
                return 0.1 + 0.9 * t
            return 0.0
        return f
    raise ValueError(mode)


print("== 连续失败率下 P3 单调性（docs/200 边界 #1）==")
print("把'被拒'从二值开关推广为连续失败率 p；染色段 60 帧、恢复段 40 帧")
print("测：重信任延迟 随 累计被拒帧数 是否单调增、曲线是否跨模式对齐\n")

print(f"{'模式':12s} {'累计被拒':>8s} {'重信任延迟':>10s} {'染色段通过':>10s}")
all_pts = []
for mode in ['binary', 'partial_05', 'partial_08', 'ramp']:
    frames, rej_ground = make_ball_frac(TOTAL, prob_fn(mode))
    seq, n_rej = run(frames)
    delay = retrust_delay(frames, seq, SEG[1])
    seg_rate = np.mean(seq[SEG[0]:SEG[1]])
    all_pts.append((mode, n_rej, delay))
    print(f"{mode:12s} {n_rej:8d} {str(delay):>10s} {seg_rate:10.3f}")

print("\n== R1 单调性（延迟 vs 累计被拒，跨模式）==")
pts = sorted([(n, d) for _, n, d in all_pts if d is not None])
mono = all(pts[i][1] <= pts[i+1][1] for i in range(len(pts) - 1))
print(f"  {pts}  单调增: [{'PASS' if mono else 'FAIL'}]")

print("\n== R2 模式无关：决定量是被拒计数而非时间长度 ==")
print("  binary(60帧全拒) 与 ramp(60帧渐变,~33被拒) 若延迟随 rejected 对齐")
print("  → 纪念按被拒计数累积（失败入史），不按墙钟时间累积")
print("  （对照：若按时间长度，两者染色段都是 60 帧，延迟应相同）")
if len(pts) >= 2:
    r = np.corrcoef([p[0] for p in pts], [p[1] for p in pts])[0, 1]
    print(f"  rejected-延迟 相关系数 r = {r:.3f}  [{'PASS' if r > 0.8 else '参考'}]")

print("\n== 判读 ==")
print("  R1 成立 + R2 对齐 → '纪念=被拒历史(计数)改变后续对待'的机制主张跨失败模式成立；")
print("  docs/200 的 P3 从二值 toy 升格为连续失败率下的结构结论（决定量=被拒计数）")
