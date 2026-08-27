"""vision/language_closed_loop.py — 语言→感知闭环：符号槽反向驱动感知参数。

docs/204 §五 1：语言→感知闭环——符号标签反过来驱动感知（"它在找红色"→颜色通道
优先、确认阈值随标签调）。docs/194"眼+语言闭环"中语言给先验的方向，本文件机制化：
语言层（符号槽）的输出不再是"感知→语言"单向转录，而是**反向调节感知层参数**。

机制：
  感知层：确认通过 = 目标区红区占比 > thr（thr 是感知参数）
  语言层（docs/204 LanguageMind）：符号槽 {color, suspect, persist}
  闭环：符号槽状态 → 感知参数 thr：
    suspect=True（被拒历史）→ thr 上调（需要更确凿证据才确认——"我不确定，
      我要看得更仔细"）
    persist 高（持续信任）→ thr 回落（"我熟悉它了，确认可以松一点"）
  无闭环（基线）：thr 固定 0.2，语言只输出不调节

场景：红球，帧 30-59 染蓝（确认失败），60 恢复。测 thr 轨迹 + 确认行为。

判据：
  P1 闭环生效：有闭环时恢复段确认阈值被语言层抬高（被拒后"更仔细"），
     确认更谨慎（恢复段确认通过率/延迟与无闭环不同）
  P2 闭环可逆：持续信任后 thr 回落（不永久锁死——docs/203 饱和精神）
  P3 可追溯：每帧 thr 可由符号槽状态解释（语言↔感知双向映射）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED

TRAIN_SEG = (30, 60)


def make_interrupted(rej=TRAIN_SEG, total=N):
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


class ClosedLoop:
    """语言→感知闭环。closed=True 时符号槽调节确认阈值 thr。"""

    def __init__(self, closed=True, base_thr=0.2, suspect_boost=0.3, trust_relax=0.05,
                 persist_for=20):
        self.closed = closed
        self.base_thr = base_thr
        self.suspect_boost = suspect_boost    # 被拒 → 阈值上调幅度
        self.trust_relax = trust_relax        # 持续信任 → 阈值回落幅度
        self.persist_for = persist_for        # 持续多少帧后回落
        self.thr = base_thr
        self.suspect = False                  # 符号槽：可疑
        self.persist = 0                     # 符号槽：持续信任帧
        self.rejected = 0
        self.ok_frames = 0

    def step(self, frac):
        """返回 (确认通过?, 本帧 thr)。"""
        ok = frac > self.thr
        if not ok:
            # 被拒 → 符号槽"可疑" + 阈值上调（闭环）
            self.rejected += 1
            self.suspect = True
            self.persist = 0
            if self.closed:
                self.thr = min(self.base_thr + self.suspect_boost,
                               self.thr + 0.05)   # 每次被拒再紧一点，封顶
        else:
            self.ok_frames += 1
            self.persist += 1
            if self.persist >= self.persist_for and self.closed and self.thr > self.base_thr:
                # 持续信任 → 阈值逐步回落（可逆，docs/203 饱和精神：谨慎但不锁死）
                self.thr = max(self.base_thr, self.thr - self.trust_relax)
            if self.persist >= self.persist_for * 2:
                self.suspect = False
        return ok, self.thr


frames = make_interrupted()
print("== 语言→感知闭环：符号槽反向驱动确认阈值 ==")
print(f"场景：红球，帧 {TRAIN_SEG[0]}-{TRAIN_SEG[1]} 染蓝（确认失败），60 恢复\n")

for closed in [False, True]:
    sys2 = ClosedLoop(closed=closed)
    print(f"--- {'有闭环（符号槽调节 thr）' if closed else '无闭环（thr 固定 0.2）'} ---")
    print(f"{'帧':>4s} {'红区占比':>8s} {'thr':>6s} {'确认':>4s} "
          f"{'语言(符号槽状态)':>24s}")
    for i, fr in enumerate(frames):
        frac = ball_stats(fr, i)
        ok, thr = sys2.step(frac)
        # 语言层转录（docs/204 三层）
        if not ok and i == TRAIN_SEG[0]:
            lang = "它变了，我认不出来了——刚才可能认错了"
        elif not ok:
            lang = "不是红，我不确定，我要看得更仔细"
        elif i >= TRAIN_SEG[1] and sys2.persist < sys2.persist_for:
            lang = f"像红的了，但我还谨慎({sys2.persist}/{sys2.persist_for}帧)"
        elif sys2.thr > sys2.base_thr:
            lang = "持续信任，确认在放松（阈值回落中）"
        elif i % 15 == 0 and i < TRAIN_SEG[0]:
            lang = "确认红，稳定"
        else:
            continue
        print(f"{i:4d} {frac:8.3f} {thr:6.2f} {'✓' if ok else '✗'} {lang:>24s}")
    print()

print("== 判读 ==")
print("  闭环：被拒 → 符号槽'可疑' → 阈值上调（'我要看得更仔细'）")
print("      持续信任 → 阈值回落（'确认可以松一点'）——可逆，不锁死（docs/203 饱和）")
print("  无闭环：阈值固定，语言只是被动转录")
print("  docs/194：语言给先验的方向——语言层（符号槽）成为感知参数的调节器")
print("  docs/204 §五 1：符号槽从输出变输入（语言→感知闭环）")
