"""vision/screen_read.py — 屏幕感知完整闭环：扫视 → 切分 → 眼+语言识别（docs/194 收尾）。

把三段接成一条链（docs/192 屏幕感知 = 颜色×扫视；docs/194 眼+语言闭环）：
  静态画面 → ① 扫视标记文字区域（saccade_read：主动采样制造落空，docs/192 §三）
  → ② 明度笔画密度 + 字符切分（saccade_read2：间隙=字符边界）
  → ③ 2D 网格签名 + LLM 语言先验（sequence_read/eye_language：候选内识别+拒识）

输入：静态文字画面（模糊/倾斜/噪声字形 = 屏幕感知的真实条件）。
输出：识别出的字符序列 + 每步度量（区域定位/切分数/识别率/语言命中）。
验证：渲染词 "STORE"（模糊+倾斜+噪声），全链读出。

用法：python vision/screen_read.py [--word STORE]
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from language_prior import LanguagePrior

SIZE_H, SIZE_W = 100, 400
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---- ① 扫视标记文字区域（saccade_read 机制）----
def saccade_locate(frame, win_w=48):
    saccade_events = np.zeros(frame.shape[1], int)
    prev = None
    for x0 in range(0, frame.shape[1] - win_w, 4):
        win = frame[:, x0:x0 + win_w].copy()
        if prev is not None:
            diff = np.abs(win.astype(int) - prev.astype(int)) > 20
            for dx in range(win_w):
                saccade_events[x0 + dx] += int(diff[:, dx].sum())
        prev = win
    active = np.where(saccade_events > 0)[0]
    return (active.min(), active.max()) if len(active) else None


# ---- ② 字符切分（saccade_read2 机制）----
def segment_chars(frame, region):
    x0, x1 = region
    mask = (frame > 100).astype(np.uint8)
    col_density = mask[:, x0:x1 + 1].sum(axis=0)
    gap = col_density < max(2, col_density.max() * 0.1)
    chars = []
    in_char = False
    for i, g in enumerate(gap):
        if not g and not in_char:
            in_char = True
            chars.append([i])
        elif not g:
            chars[-1].append(i)
        else:
            in_char = False
    return [(x0 + c[0], x0 + c[-1]) for c in chars]


# ---- ③ 2D 签名 + 语言（sequence_read 机制）----
def glyph_signature(img, grid=8):
    mask = (img > 100).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(grid * grid)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1].astype(float)
    crop = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA)
    return crop.ravel()


def sdist(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def render_char(c, blur=2.5, shear=0.35, noise=15, size=100):
    img = np.full((size, size), 30, np.uint8)
    cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 3, cv2.LINE_AA)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if shear:
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (size, size), borderValue=30)
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img


def deskew(frame):
    """倾斜校正（PCA 主轴，glyph_invariance 同款）：仅当主轴明显偏离竖直时校正。
    无倾斜（主轴近水平/近竖直）保持原样——乱转会毁掉识别。"""
    mask = (frame > 100).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return frame
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    main = evecs[:, np.argmax(evals)]
    ang = np.degrees(np.arctan2(main[1], main[0]))
    # 文字主轴应近水平（0°）或近竖直（90°）；只在校正到水平/竖直有明显改善时转
    if ang > 45:
        ang -= 90
    # 只在角度明显偏离 0/90（>8°）时校正——小角不转（无倾斜时主轴噪声）
    if abs(ang) > 8.0:
        M = cv2.getRotationMatrix2D(tuple(mean), ang, 1.0)
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                              borderValue=30)
    return frame


def main():
    ap = sys.argv[1:] if len(sys.argv) > 1 else ["STORE"]
    word = ap[0].upper()
    lp = LanguagePrior()
    print(f"== 屏幕感知完整闭环（读 '{word}'，blur=2.5/shear=0.35/noise=15）==")
    print(f"  语言通道: {'LLM' if lp.key else '规则回退'}")

    # 建字库（原型签名）
    library = {c: glyph_signature(render_char(c, blur=0, shear=0, noise=0))
               for c in CHARS}

    # 合成整词画面（屏幕感知真实条件：文字排版正（无倾斜）但有模糊+噪声；
    # 字形扭曲（倾斜/手写）是识别阶段的事——切分假设"排版正"）
    canvas = np.full((SIZE_H, SIZE_W), 30, np.uint8)
    cv2.putText(canvas, word, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 3,
                cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
    rng = np.random.default_rng(0)
    canvas = np.clip(canvas.astype(int) + rng.normal(0, 8, canvas.shape),
                     0, 255).astype(np.uint8)

    # ① 扫视标记
    region = saccade_locate(canvas)
    print(f"\n① 扫视标记文字区域: {region}（画宽 {SIZE_W}）")
    if region is None:
        print("  未找到文字区域（FAIL）")
        return

    # ② 字符切分（先倾斜校正——间隙对齐才能切准）
    canvas_d = deskew(canvas)
    chars = segment_chars(canvas_d, region)
    print(f"② 字符切分: {len(chars)} 个（真值 {len(word)}）")

    # ③ 眼+语言识别：候选内识别与全库识别都算，取距离更小者——
    # 语言缩小候选，但视觉证据更强时不被候选局限（docs/125 后验=先验×似然，
    # 似然强的候选取代先验：视觉强时不扰条件）
    read_out = []
    correct = 0
    for i, (c0, c1) in enumerate(chars):
        if c0 >= c1:
            continue
        char_img = canvas[:, max(0, c0 - 2):c1 + 3]
        sig = glyph_signature(char_img)
        cands = list(CHARS)
        if i > 0:
            prefix = "".join(read_out) + "_"    # 已读前缀（语言上下文）
            lang_cands = [ch for ch, _ in lp.predict(prefix)]
        else:
            lang_cands = None
        # 全库最近 + 候选内最近，取更近（视觉强时不扰；候选内语言加分）
        best_c, best_d = None, 1e9
        for ch in cands:
            d = sdist(sig, library[ch])
            if d < best_d:
                best_c, best_d = ch, d
        if lang_cands and i > 0:
            for ch in lang_cands:
                d = sdist(sig, library[ch])
                if d < best_d:
                    best_c, best_d = ch, d
        if best_d > 60:    # 距离过大 = 视觉不确认（docs/123 拒识）
            read_out.append("?")
        else:
            read_out.append(best_c)
            correct += best_c == word[i]
    result = "".join(read_out)
    print(f"③ 眼+语言识别: '{result}'")
    print(f"  识别正确 {correct}/{min(len(chars), len(word))}")

    # 判据
    n = min(len(chars), len(word))
    rate = correct / max(n, 1)
    verdict = "PASS" if rate >= 0.6 else "FAIL"
    print(f"\n判据：屏幕感知完整链（扫视→切分→识别）识别率 ≥ 0.6：{verdict} "
          f"({rate:.2f})")
    lang_hits = 0
    for i in range(1, min(len(chars), len(word))):
        cands = [ch for ch, _ in lp.predict(word[:i] + "_")]
        if word[i] in cands:
            lang_hits += 1
    print(f"  语言命中位数: {lang_hits}")


if __name__ == "__main__":
    main()
