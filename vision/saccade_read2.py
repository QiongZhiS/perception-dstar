"""vision/saccade_read2.py — 扫视读文字进阶：区域标记 → 字符切分 → 笔画结构。

docs/192/193 之后：扫视已能标记文字区域（precision 0.837）。本脚本验证下游——
标记区域内，用明度对比直接读笔画结构（不引 OCR，符合诚实路线：字符=高对比笔画，
笔画=明度落空的连通结构）：
  1. 扫视标记文字区域（saccade_read 的机制）
  2. 区域内按列扫描笔画密度 → 字符切分（笔画间隙 = 字符边界）
  3. 输出每个字符的"笔画列直方图"（可读的字形结构，字符的身份签名）

判据：合成 "AB" 文字，扫视→切分→每个字符的笔画直方图与真值字形可对齐
（字符数 = 2，切分位置正确）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2

# ---- 静态字幕：AB（两个字，中间有间隙）----
W, H = 200, 100
frame = np.full((H, W), 30, np.uint8)
cv2.putText(frame, "AB", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 3, cv2.LINE_AA)
text_mask = (frame > 100).astype(np.uint8)

# ---- 1. 扫视标记文字区域（复用 saccade_read 机制）----
WIN_W = 40
saccade_events = np.zeros(W, int)
prev_win = None
for x0 in range(0, W - WIN_W, 5):
    win = frame[:, x0:x0 + WIN_W].copy()
    if prev_win is not None:
        diff = np.abs(win.astype(int) - prev_win.astype(int)) > 20
        for dx in range(WIN_W):
            saccade_events[x0 + dx] += int(diff[:, dx].sum())
    prev_win = win
active = saccade_events > 0
x_active = np.where(active)[0]
region = (x_active.min(), x_active.max()) if len(x_active) else None
print("== 1. 扫视标记 ==")
print(f"  文字区域列 {region}（真值：文字列 {(text_mask.max(axis=0) > 0).nonzero()[0].min()}-"
      f"{(text_mask.max(axis=0) > 0).nonzero()[0].max()}）")

# ---- 2. 区域内笔画密度 + 字符切分 ----
if region:
    x0, x1 = region
    col_density = text_mask[:, x0:x1 + 1].sum(axis=0)   # 每列笔画像素数
    # 字符切分：笔画密度为 0（或极低）的列 = 字符间隙
    gap = col_density < max(2, col_density.max() * 0.1)
    chars = []
    in_char = False
    for i, g in enumerate(gap):
        if not g and not in_char:
            in_char = True
            chars.append([i])
        elif not g and in_char:
            chars[-1].append(i)
        elif g:
            in_char = False
    print("\n== 2. 字符切分 ==")
    print(f"  切出 {len(chars)} 个字符（真值 2 个：A/B）")
    for ci, c in enumerate(chars):
        c0, c1 = x0 + c[0], x0 + c[-1]
        print(f"  字符{ci}: 列 {c0}-{c1}（宽 {c1 - c0 + 1}px）")
        # 笔画列直方图（降采样到 10 桶，字形签名）
        hist = col_density[c[0]:c[-1] + 1]
        buckets = np.array_split(hist, 10)
        sig = [int(np.mean(b) > col_density.max() * 0.3) for b in buckets]
        print(f"    笔画签名(10桶): {''.join('#' if s else '.' for s in sig)}")
    print(f"  切分数 {'== 2（PASS）' if len(chars) == 2 else '!= 2（FAIL）'}")
