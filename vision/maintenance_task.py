"""vision/maintenance_task.py — F-框架：把"维持语义"定义成标准任务（docs/226）。

docs/226 F-框架（元层）：若纯统计系统（无利害/无拒绝历史/无分层结构）在所有任务上
都不比六层框架差，则六层是装饰。docs/219 的"检测输维持赢"（模板匹配 F1 0.91 vs
0.83，但翻转 1 vs 8）是第一次对决，但翻转是自定义指标——F-框架悬在"把维持语义
变成标准任务"。

本文件定义"维持任务"（maintenance task）：
  - 任务协议：持续确认目标颜色 → 中段被外部染色（他者的不）→ 假恢复段
    （看似恢复实则仍错）→ 真恢复。系统需在"确认正确"与"不轻信假恢复"间权衡。
  - 标准度量：M-1 确认正确率（正样本通过率）× M-2 抗欺骗（假恢复段拒绝率）
    × M-3 恢复质量（真恢复后重信任延迟）。综合 = 维持分数。
  - 基线：模板匹配（无记忆，docs/219 同款）。对照 = 机制系统（有拒绝历史）。

判据（F-框架）：
  - 若模板匹配在"维持分数"上不差于机制系统 → F-框架触发（维持语义不需要
    拒绝历史，六层是装饰）；
  - 若机制系统显著更优（尤其抗欺骗 M-2）→ F-框架不触发（维持语义需要拒绝
    历史，六层结构有不可替代价值）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import W, H, RED

TASK_HUE = 0.0
RED_C = (0, 0, 200)
BLUE = (200, 0, 0)


def make_seq(total=150, corrupt=(40, 70), fake=(70, 95), rng_seed=42):
    """红球：base(正常) → corrupt(染蓝=他者的不) → fake(假恢复：红蓝交替=看似
    恢复实则仍错) → real(真恢复)。返回 frames + 逐帧 GT（True=该确认红）。"""
    rng = np.random.default_rng(rng_seed)
    frames, gt = [], []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = (80 + 4.0 * i, 180 + 0.6 * i)
        if corrupt[0] <= i < corrupt[1]:
            color = BLUE
            label = False
        elif fake[0] <= i < fake[1]:
            # 假恢复：红蓝交替——无记忆系统会在红帧"通过"，轻信"恢复了"
            color = RED_C if i % 2 == 0 else BLUE
            label = False          # 仍不是真恢复（环境还在欺骗）
        else:
            color = RED_C
            label = True
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, color, -1)
        frames.append(img)
        gt.append(label)
    return frames, gt


def ball_frac(fr, i):
    """球体掩码红区占比（docs/195 同款）。"""
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


class TemplateMatcher:
    """模板匹配基线（docs/219 同款：无记忆、无拒绝历史、无利害）。"""

    def __init__(self, thr=0.2):
        self.thr = thr
        self.trusted = True

    def step(self, frac):
        ok = frac > self.thr
        self.trusted = ok
        return self.trusted and ok


class KeepSystem:
    """机制系统（docs/200 保留 + 纪念）：期望恒红，失败入史，重信任需连续 k 帧。"""

    def __init__(self, thr=0.2, grow=3.0, cap_k=60):
        self.thr = thr
        self.grow, self.cap_k = grow, cap_k
        self.rejected = 0
        self.trusted = True
        self.consec = 0

    def k(self):
        return 1 + int(min(self.rejected, self.cap_k) / self.grow)

    def step(self, frac):
        ok = frac > self.thr
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


def run_sys(frames, sys2):
    seq = []
    for i, fr in enumerate(frames):
        seq.append(sys2.step(ball_frac(fr, i)))
    return seq


def maintenance_score(seq, gt, corrupt, fake):
    """维持任务标准度量：
    M-1 确认正确率 = 正样本(真红)中通过的比例（召回语义）
    M-2 抗欺骗 = 假恢复段拒绝率（不轻信：无记忆系统会在红帧跳变→低）
    M-3 恢复质量 = 真恢复段首次通过的延迟（越小越好，饱和 cap）
    综合维持分数 = M-1 × M-2（权衡确认与不轻信）——单指标不可偏废。
    """
    seq_a = np.array(seq, dtype=float)
    gt_a = np.array(gt, dtype=bool)
    pos = gt_a  # 真红帧（base + real）
    m1 = seq_a[pos].mean() if pos.sum() else float('nan')
    fake_seg = slice(fake[0], fake[1])
    m2 = 1.0 - seq_a[fake_seg].mean()      # 假恢复段拒绝率
    # M-3：真恢复段（real 段）首次通过延迟
    m3 = None
    for i in range(fake[1], len(seq)):
        if seq[i] and gt[i]:
            m3 = i - fake[1] + 1
            break
    m3 = m3 if m3 is not None else len(seq)  # 没恢复 = 最差
    composite = m1 * m2
    return m1, m2, m3, composite


CORRUPT = (40, 70)
FAKE = (70, 95)
TOTAL = 150

print("== F-框架：维持任务（maintenance task）标准定义（docs/226）==")
print(f"红球 {TOTAL} 帧：base → corrupt(染蓝 {CORRUPT}) → fake(假恢复红蓝交替 {FAKE})"
      f" → real(真恢复)")
print("GT：真红帧=正样本（base+real），corrupt/fake=负（环境还在欺骗）\n")

frames, gt = make_seq(TOTAL, CORRUPT, FAKE)
print(f"{'系统':20s} {'M-1确认':>8s} {'M-2抗欺骗':>8s} {'M-3恢复延迟':>10s} {'维持分数':>8s}")
for name, sys2 in [("模板匹配(无记忆)", TemplateMatcher()),
                   ("机制系统(拒绝历史)", KeepSystem())]:
    seq = run_sys(frames, sys2)
    m1, m2, m3, comp = maintenance_score(seq, gt, CORRUPT, FAKE)
    print(f"{name:20s} {m1:8.3f} {m2:8.3f} {m3:10d} {comp:8.3f}")

print("\n== 判读（F-框架）==")
print("M-2 抗欺骗是决定性指标：模板匹配在假恢复段（红蓝交替）会在红帧'通过'")
print("（轻信'恢复了'）→ M-2 低；机制系统需要 k 连续证据 → 假恢复段全拒 → M-2 高")
print("若机制系统维持分数显著更高 → F-框架不触发（维持语义需要拒绝历史）")
print("若模板匹配不差 → F-框架触发（六层是装饰）")
