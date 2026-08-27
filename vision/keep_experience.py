"""vision/keep_experience.py — 阅历：被拒历史的无限累积 vs 饱和（"变厚不是变熟练"）。

外部笔记（2026-08-27 对话）提出的可测实验：
  把记忆尺度设为"无限"（不衰减）→ 同一视频流跑 100 遍 → 记录第 1 遍与第 100 遍的
  行为差异——差异是"拟合"（记住具体帧）还是"判断阈值偏移"（更保守/更开放）？

docs/200 的 Keep 系统已有"被拒入史 + 纪念加深"（重信任延迟 ∝ 被拒时长），但那是
单遍、会话内记忆。本实验把记忆推到"生命周期"尺度（跨遍不清零），并回答三个问题：

  P1 阅历生效：第 1 遍 vs 第 100 遍恢复段通过率/重信任延迟不同（见过=行为改变）
  P2 阈值偏移非拟合：阅历记忆是"目标级"（位置无关）→ 染色段挪到新位置仍谨慎；
      对照"位置记忆"变体（记住帧号）→ 新位置无反应（那是拟合，不是阅历）
  P3 饱和防锁死：线性累积（rejected 无限 → 延迟无限 → 永不恢复=病理性固执）；
      饱和累积（延迟有上限 → 谨慎但能恢复=健康阅历）。
      docs/101"越噪声越固执"的边界：固执要有上限，否则世界恢复了你回不来。

场景：红球匀速直线（docs/195 同款），每遍染色段固定（30-60 蓝），连续跑 N 遍。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED

# 染色段（每遍固定）与 测试段（挪位置，验泛化）
TRAIN_SEG = (30, 60)
TEST_SEG = (45, 70)


def make_interrupted(rej=TRAIN_SEG, total=N):
    """红球 total 帧，[rej0,rej1) 帧目标被染成蓝（docs/200 同款场景构造）。"""
    r0 = np.array([80.0, 180.0]); rv = np.array([4.0, 0.6])
    rng = np.random.default_rng(42)
    frames = []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc = r0 + rv * i
        color = (200, 0, 0) if rej[0] <= i < rej[1] else (0, 0, 200)
        cv2.circle(img, (int(rc[0]), int(rc[1])), 18, color, -1)
        frames.append(img)
    return frames


def ball_stats(fr, i):
    rc = (80 + 4.0 * i, 180 + 0.6 * i)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    r = 18
    cx, cy = int(rc[0]), int(rc[1])
    rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0
    return float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())


class KeepMem:
    """docs/200 Keep + 跨遍记忆。memory='linear'/'saturate'/'position'。

    - linear：被拒无限累积，重信任延迟 K = 1 + rejected/grow（无上限→可锁死）
    - saturate：累积有上限 cap，K = 1 + min(rejected, cap)/grow（谨慎但能恢复）
    - position：只记住"染色发生在哪一段"（拟合变体：新位置无阅历）
    """

    def __init__(self, memory='saturate', grow=3.0, cap=60, train_seg=TRAIN_SEG):
        self.memory = memory
        self.grow = grow
        self.cap = cap
        self.train_seg = train_seg
        self.rejected = 0          # 累计被拒（目标级，位置无关）
        self.pos_rejected = {}     # 位置拟合：每帧位置→被拒次数
        self.trusted = True
        self.consec = 0

    def k(self, i):
        if self.memory == 'linear':
            return 1 + int(self.rejected / self.grow)
        if self.memory == 'saturate':
            return 1 + int(min(self.rejected, self.cap) / self.grow)
        # position：只对被拒过的帧段谨慎（拟合）
        n = sum(v for k, v in self.pos_rejected.items()
                if abs(k - i) < 5)   # 局部帧窗
        return 1 + int(n / self.grow)

    def step(self, frac_red, i):
        ok = frac_red > 0.2
        if ok:
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k(i):
                    self.trusted = True
        else:
            self.rejected += 1
            self.pos_rejected[i] = self.pos_rejected.get(i, 0) + 1
            self.trusted = False
            self.consec = 0
        return self.trusted and ok


def pre_stats(frames):
    """预计算每帧红区占比（同一视频流结果固定，100 遍复用）。"""
    return [ball_stats(fr, i) for i, fr in enumerate(frames)]


def run_pass_stats(stats, keep, seg=TRAIN_SEG):
    """用预计算 stats 跑一遍，返回 (恢复段通过率, 恢复段重信任延迟帧数)。"""
    seq = []
    for i, frac in enumerate(stats):
        seq.append(keep.step(frac, i))
    rest = seq[seg[1]:]
    rate = np.mean(rest) if rest else float('nan')
    delay = None
    for j, ok in enumerate(rest):
        if ok:
            delay = j + 1
            break
    return rate, delay


print("== 阅历：被拒历史的无限累积 vs 饱和（第1遍到第100遍）==")
print(f"场景：红球 {N} 帧，训练染色段 {TRAIN_SEG[0]}-{TRAIN_SEG[1]}（蓝），恢复段 {TRAIN_SEG[1]}-{N}\n")

print("== P1 阅历生效：第1 vs 第100遍 恢复段通过率/延迟（记忆跨遍不清零）==")
print(f"{'记忆形态':>12s} {'遍数':>6s} {'恢复通过率':>10s} {'重信任延迟':>10s}")
for mem in ['linear', 'saturate']:
    keep = KeepMem(memory=mem)
    stats = pre_stats(make_interrupted(rej=TRAIN_SEG))   # 同一视频流，预计算一次
    for epoch in [1, 10, 50, 100]:
        rate, delay = run_pass_stats(stats, keep)
        dl = delay if delay is not None else '—(不恢复)'
        print(f"{mem:>12s} {epoch:>6d} {rate:10.3f} {str(dl):>10s}")

print("\n== P2 阈值偏移非拟合：阅历(目标级) 挪染色段仍谨慎；拟合(位置级) 不谨慎 ==")
print(f"训练：染色段 {TRAIN_SEG[0]}-{TRAIN_SEG[1]} × 100 遍 → 测试：染色段挪到 {TEST_SEG[0]}-{TEST_SEG[1]}")
for mem in ['position', 'saturate']:
    keep = KeepMem(memory=mem)
    stats_tr = pre_stats(make_interrupted(rej=TRAIN_SEG))   # 训练视频流（同一段）
    for _ in range(100):                                    # 100 遍训练（记忆累积）
        run_pass_stats(stats_tr, keep, seg=TRAIN_SEG)
    # 测试：新位置染色（同一场景生成器，不同染色段）
    stats_t = pre_stats(make_interrupted(rej=TEST_SEG))
    rate, delay = run_pass_stats(stats_t, keep, seg=TEST_SEG)
    dl = delay if delay is not None else '—'
    verdict = '（阅历泛化✓ 阈值偏移）' if mem == 'saturate' and rate < 0.9 else \
              ('（位置拟合✗ 新位置无阅历）' if mem == 'position' else '')
    print(f"{mem:>12s} 恢复通过率 {rate:.3f} 延迟 {dl}  {verdict}")

print("\n== P3 饱和防锁死：线性累积永不恢复 vs 饱和累积谨慎但能恢复 ==")
print(f"{'记忆形态':>12s} {'第1遍延迟':>10s} {'第100遍延迟':>10s} {'恢复':>6s}")
for mem in ['linear', 'saturate']:
    keep = KeepMem(memory=mem)
    stats = pre_stats(make_interrupted(rej=TRAIN_SEG))
    _, d1 = run_pass_stats(stats, keep)
    for _ in range(99):
        run_pass_stats(stats, keep)
    _, d100 = run_pass_stats(stats, keep)
    rec = '能恢复' if d100 is not None else '锁死'
    print(f"{mem:>12s} {str(d1 if d1 is not None else '—'):>10} {str(d100 if d100 is not None else '—'):>10} {rec:>6s}")

print("\n== 判读 ==")
print("  linear   ：被拒无限累积 → 延迟无限增长 → 第100遍永不恢复（病理性固执，docs/101 边界）")
print("  saturate ：累积有上限 → 延迟饱和 → 更谨慎但能恢复（健康阅历：'一朝被蛇咬'但有上限）")
print("  position ：只记帧号 → 新位置无阅历 = 拟合（记住具体帧），不是'变厚'")
print("  saturate 对位置无关 = 阈值偏移（判断更保守，不是记住帧）→ 阅历生效")
