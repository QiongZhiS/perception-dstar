"""vision/saccade_read.py — 主动采样读静态字幕（docs/192 §三/§五 3：docs/133 实验③最小版）。

场景修正（docs/192）：屏幕静态感知——字幕不动但人能读，靠主动采样（扫视）把静止
画面变成采样序列。事件驱动"静止=0 事件"对静态字幕盲（C1 局限），扫视 = 主动制造落空
（docs/187 欣赏同构）。

设计：
  - 静态字幕画面（白色文字在深色底）
  - 被动：整帧转导 → 0 事件（静止盲，复现问题）
  - 主动：扫视窗口（2° 视窗类比）依次扫过文字，每次采样 = 窗口内画面的转导
    → 窗口移动时文字边缘进入/离开窗口 = 落空 → 事件序列标记文字位置
  - 验证：扫视累计事件能重建文字区域（覆盖文字位置）

用法：python vision/saccade_read.py
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from transduction import Transduction2D

# ---- 构造静态字幕画面 ----
W, H = 640, 120
frame = np.full((H, W), 30, np.uint8)          # 深底
cv2.putText(frame, "HELLO WORLD", (60, 80), cv2.FONT_HERSHEY_SIMPLEX,
            2.0, 255, 3, cv2.LINE_AA)          # 白字
text_mask = (frame > 100).astype(np.uint8)     # 文字真值

# ---- 1. 被动：整帧转导（静止→0 事件）----
td = Transduction2D(thresh=0.15, deadband=0.015, blur=0.8)
ev_total = 0
for _ in range(10):
    ev_f, ev_s, _, _ = td.step(frame)
    ev_total += int(((ev_f != 0) | (ev_s != 0)).sum())
print("== 1. 被动整帧（静止字幕）==")
print(f"  10 帧事件总数 {ev_total}（应≈0：静止=盲，C1 局限复现）")

# ---- 2. 主动：扫视窗口依次采样 ----
print("\n== 2. 主动扫视（窗口 64×120 依次扫过）==")
WIN_W = 64                                     # 扫视窗口宽（2° 类比：一小块）
# 扫视 = 窗口在画面移动。窗口内容随时间变化（扫视扫过文字边缘）→ 内容差分 =
# 主动制造落空（docs/187 欣赏同构）。saccade 事件 = 窗口内容与上一窗口的差异边缘。
saccade_events = np.zeros(W, int)
prev_win = None
positions = list(range(0, W - WIN_W, 8))       # 扫视步进 8px
for x0 in positions:
    win = frame[:, x0:x0 + WIN_W].copy()
    if prev_win is not None:
        diff = np.abs(win.astype(int) - prev_win.astype(int)) > 20
        for dx in range(WIN_W):
            saccade_events[x0 + dx] += int(diff[:, dx].sum())
    prev_win = win

# ---- 3. 验证：扫视累计事件 vs 文字真值 ----
active = saccade_events > 0
tp = int((active & (text_mask.max(axis=0) > 0)).sum())
fp = int((active & ~(text_mask.max(axis=0) > 0)).sum())
fn = int((~(active) & (text_mask.max(axis=0) > 0)).sum())
cols = W
print(f"  扫视标记列 {active.sum()} / {cols}")
print(f"  文字列真值 {(text_mask.max(axis=0) > 0).sum()} / {cols}")
print(f"  TP {tp}（文字列被标记）FP {fp}（非文字列误标）FN {fn}（文字列漏标）")
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
print(f"  precision {precision:.3f}（标记的确实有字）recall {recall:.3f}（字都被标记）")
print(f"\n判据：扫视主动采样能从静止画面读出文字区域（precision/recall > 0.5）")
