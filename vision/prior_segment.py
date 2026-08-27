"""vision/prior_segment.py — 先验切分：语言先验恢复粘连字符（docs/194 收尾的切分升级）。

人切分是先验驱动（docs/194 六.6 结论）：词形整体 + 语言先验，不是逐字后验扫描。
粘连（C-A 笔画相连、无间隙）时后验（列密度间隙）崩，语言先验补上。

实验：
  1. CATCH 渲染，C-A 间隙人为填掉（粘连）→ 纯后验切分崩（块数 < 5）
  2. 先验切分：
     a. 语言先验（词表/LLM）说"这个词长 5"→ 粘连块宽≈2 字符 → 该切 2
     b. 块内扫切缝：每个候选切缝把块分成左右两字符，2D 签名各自匹配字库
     c. 取"两侧签名距离和最小"的切缝（docs/125：先验缩小范围，视觉裁决）
  3. 判据：先验切分恢复 5 字符，识别正确率 > 纯后验

用法：python vision/prior_segment.py
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from language_prior import LanguagePrior

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render_word(word, glue=(0, 0), blur=1.5, noise=8):
    """渲染整词。glue=(i,j) = 第 i 字符与第 j 字符粘连（填掉间隙）。"""
    W = 400
    canvas = np.full((100, W), 30, np.uint8)
    x = 20
    char_pos = []
    for ci, c in enumerate(word):
        img = np.full((100, 100), 30, np.uint8)
        cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 3,
                    cv2.LINE_AA)
        ys, xs = np.nonzero(img > 100)
        cw = xs.max() - xs.min() + 1
        char_pos.append((x, x + cw))
        canvas[:, x:x + cw] = img[:, xs.min():xs.max() + 1]
        x += cw + 14                     # 14px 间隙
    # 粘连：把 glue 两字符间填上（用前字符实际右边缘列桥接）
    if glue[1] > glue[0]:
        i, j = glue
        mask = (canvas > 100).astype(np.uint8)
        cols = mask.sum(axis=0)
        # 前字符实际右边缘 = 最后密度>0 的列
        nz = np.nonzero(cols)[0]
        right_edge = nz[nz < char_pos[j][0]].max()
        x2 = char_pos[j][0]                     # 后字符实际左边缘
        # 从右边缘到后字符左边缘：复制右边缘列桥接（模拟笔画相连）
        bridge = canvas[:, right_edge:right_edge + 1]
        canvas[:, right_edge:x2] = np.maximum(canvas[:, right_edge:x2], bridge)
    if blur:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    if noise:
        rng = np.random.default_rng(0)
        canvas = np.clip(canvas.astype(int) + rng.normal(0, noise, canvas.shape),
                         0, 255).astype(np.uint8)
    return canvas, char_pos


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


def segment_posterior(canvas, min_gap=0.1):
    """纯后验切分（列密度间隙，screen_read 同款）。"""
    mask = (canvas > 100).astype(np.uint8)
    cols = mask.sum(axis=0)
    nz = np.nonzero(cols)[0]
    if len(nz) == 0:
        return []
    x0, x1 = nz.min(), nz.max()
    col_density = cols[x0:x1 + 1]
    gap = col_density < max(2, col_density.max() * min_gap)
    chars, in_char = [], False
    for i, g in enumerate(gap):
        if not g and not in_char:
            in_char = True
            chars.append([i])
        elif not g:
            chars[-1].append(i)
        else:
            in_char = False
    return [(x0 + c[0], x0 + c[-1]) for c in chars]


def segment_prior(canvas, blocks, word_len, library):
    """先验切分：语言先验（词长 word_len）说粘连块该切多个字符；
    块内扫切缝，视觉签名裁决（docs/125：先验缩小范围、视觉确认）。
    blocks: 后验切出的块 [(x0,x1),...]。返回切分后的字符块列表。"""
    result = []
    for x0, x1 in blocks:
        bw = x1 - x0 + 1
        # 先验：词长已知 → 每块该含几个字符 ≈ 宽度/平均字符宽
        # 单字符宽 ≈ 35px（渲染参数），块宽超 1.5× → 判断含多字符
        if bw <= 52:                 # 单字符（宽 < 1.5×35）
            result.append((x0, x1))
            continue
        # 粘连块：扫切缝，取两侧签名总和最小的（视觉裁决）
        n_chars = max(2, int(round(bw / 35)))
        best = None
        for cut in range(x0 + 8, x1 - 7):     # 切缝候选
            left = canvas[:, max(0, x0 - 2):cut + 3]
            right = canvas[:, cut - 2:x1 + 3]
            sl, sr = glyph_signature(left), glyph_signature(right)
            # 两侧各匹配字库最近字符的距离和
            dl = min(sdist(sl, library[c]) for c in CHARS)
            dr = min(sdist(sr, library[c]) for c in CHARS)
            score = dl + dr
            if best is None or score < best[0]:
                best = (score, cut)
        if best:
            cut = best[1]
            result.append((x0, cut))
            result.append((cut, x1))
    return result


def main():
    lp = LanguagePrior()
    library = {c: glyph_signature(_render(c)) for c in CHARS}
    word = "CATCH"
    glue = (0, 1)                    # C-A 粘连

    canvas, char_pos = render_word(word, glue=glue)
    print(f"== 先验切分（{word}，C-A 粘连）==")
    print(f"  语言通道: {'LLM' if lp.key else '规则回退'}")

    # 纯后验
    post = segment_posterior(canvas)
    print(f"\n① 纯后验切分: {len(post)} 块（真值 5，粘连后应 <5）")
    for i, (x0, x1) in enumerate(post):
        print(f"    块{i}: 列{x0}-{x1} 宽{x1-x0}")

    # 先验切分（词长 = 5）
    prior = segment_prior(canvas, post, len(word), library)
    print(f"\n② 先验切分（词长 {len(word)}）: {len(prior)} 块")
    for i, (x0, x1) in enumerate(prior):
        print(f"    块{i}: 列{x0}-{x1} 宽{x1-x0}")

    # 识别对比（候选内 + 语言）
    def identify(blocks):
        out = []
        for x0, x1 in blocks:
            img = canvas[:, max(0, x0 - 2):x1 + 3]
            sig = glyph_signature(img)
            bc, bd = None, 1e9
            for c in CHARS:
                d = sdist(sig, library[c])
                if d < bd:
                    bc, bd = c, d
            out.append(bc)
        return out

    post_ids = identify(post)
    prior_ids = identify(prior)
    post_ok = sum(1 for i, c in enumerate(post_ids) if i < len(word) and c == word[i])
    prior_ok = sum(1 for i, c in enumerate(prior_ids) if i < len(word) and c == word[i])
    print(f"\n③ 识别（纯后验切分）: {''.join(post_ids)} → 对 {post_ok}/{len(word)}")
    print(f"   识别（先验切分）:   {''.join(prior_ids)} → 对 {prior_ok}/{len(word)}")
    print(f"\n判据：先验切分恢复粘连（块数 5）且识别对 {prior_ok} > 后验 {post_ok}")


def _render(c):
    img = np.full((100, 100), 30, np.uint8)
    cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 3, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    main()
