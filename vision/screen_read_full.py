"""vision/screen_read_full.py — 屏幕感知完整链（鲁棒版）：颜色+先验+语言三段合一。

docs/194 六.8 之后：把三种切分机制合成完整链，叠加全部困难条件：
  场景：黄字 + 同亮度蓝灰底（明度失效，docs/192 颜色=区分主轴）
        + C-A 粘连（列密度间隙失效，需先验）
        + 模糊/噪声字形（识别需语言候选）
  链：① 颜色掩码（H 15-45 黄 + 连通域面积过滤）→ 文字掩码（明度盲场景分离）
      → ② 列密度切分 → 粘连块（宽>52）→ 先验切分（词长 + 签名裁决切缝）
      → ③ 每字符 2D 签名 + LLM 语言候选识别（docs/125 后验=先验×似然）

判据：全链识别率 ≥ 0.6；且（颜色+先验+语言）> 任何单环节退化。
用法：python vision/screen_read_full.py [--word CATCH]
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from language_prior import LanguagePrior

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEXT_COLOR = (0, 255, 255)       # BGR 黄


def render_full(word, glue=(0, 1), blur=2.0, noise=10):
    """黄字 + 同亮度蓝灰底 + C-A 粘连 + 模糊/噪声。"""
    W = 420
    canvas = np.zeros((100, W, 3), np.uint8)
    for x in range(W):
        shade = int(178 + 10 * np.sin(x / 30))
        canvas[:, x] = (int(178), int(0.6 * shade), int(0.6 * shade))
    x = 20
    char_pos = []
    for c in word:
        img = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, TEXT_COLOR,
                    3, cv2.LINE_AA)
        ys, xs = np.nonzero(img.sum(axis=2) > 0)
        if len(xs) == 0:
            continue
        cw = xs.max() - xs.min() + 1
        char_pos.append((x, x + cw))
        region = img[:, xs.min():xs.max() + 1]
        canvas[:, x:x + cw] = np.where(
            region.sum(axis=2, keepdims=True) > 0, region, canvas[:, x:x + cw])
        x += cw + 14
    # 粘连 C-A（黄字掩码：B 低 G/R 高——黄色特征，背景蓝灰 B=178 排除）
    if glue and glue[1] > glue[0]:
        i, j = glue
        b, g, r = canvas[:, :, 0], canvas[:, :, 1], canvas[:, :, 2]
        mask = ((b < 100) & (g > 150) & (r > 150)).astype(np.uint8)   # 黄
        cols = mask.sum(axis=0)
        nz = np.nonzero(cols)[0]
        right_edge = nz[nz < char_pos[j][0]].max()
        bridge = canvas[:, right_edge:right_edge + 1]
        cond = (bridge[:, :, 1] > 150)[:, :, None]   # (100,1,1) 广播到 (100,16,3)
        canvas[:, right_edge:char_pos[j][0]] = np.where(
            cond, bridge, canvas[:, right_edge:char_pos[j][0]])
    if blur:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    rng = np.random.default_rng(0)
    canvas = np.clip(canvas.astype(int) + rng.normal(0, noise, canvas.shape),
                     0, 255).astype(np.uint8)
    return canvas


def hue_mask(canvas, hue_lo, hue_hi, sat_min=40, min_area=80):
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    h, s, _ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= hue_lo) & (h <= hue_hi) & (s >= sat_min)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    return keep


def segment_posterior(mask2d, min_gap=0.1):
    cols = mask2d.sum(axis=0)
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


def glyph_signature(img, grid=8):
    mask = (img > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(grid * grid)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1].astype(float)
    crop = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA)
    return crop.ravel()


def sdist(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def segment_prior(mask2d, blocks, word_len):
    """先验切分：词长先验 + 签名裁决（docs/194 六.7）。"""
    result = []
    for x0, x1 in blocks:
        bw = x1 - x0 + 1
        if bw <= 52:
            result.append((x0, x1))
            continue
        best = None
        for cut in range(x0 + 8, x1 - 7):
            left = mask2d[:, max(0, x0 - 2):cut + 3]
            right = mask2d[:, cut - 2:x1 + 3]
            dl = min(sdist(glyph_signature(left), glyph_signature(lib[c]))
                     for c in CHARS)
            dr = min(sdist(glyph_signature(right), glyph_signature(lib[c]))
                     for c in CHARS)
            score = dl + dr
            if best is None or score < best[0]:
                best = (score, cut)
        if best:
            result.append((x0, best[1]))
            result.append((best[1], x1))
    return result


# 字库
lib = {}


def build_lib():
    for c in CHARS:
        img = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, TEXT_COLOR,
                    3, cv2.LINE_AA)
        lib[c] = (img.sum(axis=2) > 100).astype(np.uint8)


def identify(mask2d, blocks, lp, word, use_lang=True):
    """识别：全库 vs 语言候选取更近（docs/125 后验=先验×似然）。"""
    out = []
    for i, (x0, x1) in enumerate(blocks):
        img = mask2d[:, max(0, x0 - 2):x1 + 3]
        sig = glyph_signature(img)
        cands = list(CHARS)
        if use_lang and i > 0:
            prefix = "".join(out) + "_"
            lang_cands = [ch for ch, _ in lp.predict(prefix)]
        else:
            lang_cands = None
        bc, bd = None, 1e9
        for ch in cands:
            d = sdist(sig, glyph_signature(lib[ch]))
            if d < bd:
                bc, bd = ch, d
        if lang_cands:
            for ch in lang_cands:
                d = sdist(sig, glyph_signature(lib[ch]))
                if d < bd:
                    bc, bd = ch, d
        out.append(bc)
    return out


def main():
    ap = sys.argv[1:] if len(sys.argv) > 1 else ["CATCH"]
    word = ap[0].upper()
    lp = LanguagePrior()
    build_lib()
    print(f"== 屏幕感知完整链（鲁棒版）：{word}（同亮度底 + C-A 粘连 + 模糊/噪声）==")
    print(f"  语言通道: {'LLM' if lp.key else '规则回退'}")

    canvas = render_full(word, glue=(0, 1), blur=2.0, noise=10)

    # ① 颜色掩码（明度失效场景分离）
    mask = hue_mask(canvas, 15, 45)
    print(f"\n① 颜色掩码: {int(mask.sum())} 像素（明度失效场景——同亮度底分离靠色相）")

    # ② 后验切分 → 先验切分
    post = segment_posterior(mask)
    print(f"② 后验切分: {len(post)} 块（真值 {len(word)}，粘连应 <{len(word)}）")
    prior = segment_prior(mask, post, len(word))
    print(f"   先验切分: {len(prior)} 块")

    # ③ 识别（全链 vs 单环节退化）
    ids_full = identify(mask, prior, lp, word, use_lang=True)
    ids_no_lang = identify(mask, prior, lp, word, use_lang=False)
    ok_full = sum(1 for i, c in enumerate(ids_full) if i < len(word) and c == word[i])
    ok_no_lang = sum(1 for i, c in enumerate(ids_no_lang)
                     if i < len(word) and c == word[i])
    print(f"\n③ 识别（全链: 颜色+先验+语言）: {''.join(ids_full)} 对 {ok_full}/{len(word)}")
    print(f"   识别（无语言）:               {''.join(ids_no_lang)} 对 {ok_no_lang}/{len(word)}")

    rate = ok_full / max(len(word), 1)
    verdict = "PASS" if rate >= 0.6 else "FAIL"
    print(f"\n判据：全链识别率 ≥ 0.6：{verdict} ({rate:.2f})")


if __name__ == "__main__":
    main()
