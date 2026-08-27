"""vision/color_segment.py — 颜色×先验切分（docs/192 颜色=区分主轴 + docs/194 先验切分）。

字幕场景（docs/192 §三）：文字色统一、背景色不同且有纹理（真实字幕=彩色底+图案）。
问题：明度间隙切分被背景纹理干扰（背景也有明度变化→假间隙/假粘连）。
方案：任务层给"文字色"（颜色描述外赋，docs/192）→ 颜色掩码分离文字 vs 背景 →
掩码内切分（背景纹理被剔除）→ 粘连时词长先验 + 视觉签名裁决（docs/194 六.7）。

实验：
  1. 彩色字幕：黄字（BGR 0,255,255）在蓝底（BGR 200,0,0）+ 背景纹理噪声
  2. 颜色掩码（H 区间：黄）→ 文字掩码（背景剔除）
  3. 对比：直接明度切分 vs 颜色掩码后切分（背景干扰场景）
  4. 颜色掩码内粘连块（C-A）→ 词长先验恢复

判据：颜色掩码后切分块数正确（背景不干扰）> 直接切分；粘连 + 先验恢复 5/5。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TEXT_COLOR = (0, 255, 255)      # BGR 黄
BG_COLOR = (200, 0, 0)          # BGR 蓝


def render_subtitle(word, glue=None, bg_noise=30, blur=1.0, gradient=True):
    """渲染彩色字幕：黄字 + 同亮度蓝灰底（docs/192 颜色=区分主轴的本质场景——
    明度完全失效（字与底同亮度），只有色相可分）。glue=(i,j)：笔画粘连。"""
    W = 420
    canvas = np.zeros((100, W, 3), np.uint8)
    # 同亮度蓝灰底：亮度≈178（与黄字相同！）——明度阈值无法区分字与底
    # 色相：蓝灰（H≈120，低饱和）
    for x in range(W):
        shade = int(178 + 10 * np.sin(x / 30))       # 亮度微扰 ≈ 黄字亮度
        canvas[:, x] = (int(178), int(0.6 * shade), int(0.6 * shade))  # 蓝灰调
    rng = np.random.default_rng(7)
    canvas = np.clip(canvas.astype(int) + rng.normal(0, 5, canvas.shape),
                     0, 255).astype(np.uint8)

    # 文字（黄色）
    x = 20
    char_pos = []
    for c in word:
        img = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, TEXT_COLOR,
                    3, cv2.LINE_AA)
        ys, xs = np.nonzero(img.sum(axis=2) > 0)     # 三通道和>0 = 有字
        if len(xs) == 0:
            continue
        cw = xs.max() - xs.min() + 1
        char_pos.append((x, x + cw))
        region = img[:, xs.min():xs.max() + 1]
        canvas[:, x:x + cw] = np.where(
            region.sum(axis=2, keepdims=True) > 0,
            region, canvas[:, x:x + cw])
        x += cw + 14
    # 粘连：前字符右边缘桥接
    if glue and glue[1] > glue[0]:
        i, j = glue
        mask = (canvas.sum(axis=2) > 100).astype(np.uint8)   # 有字掩码
        cols = mask.sum(axis=0)
        nz = np.nonzero(cols)[0]
        right_edge = nz[nz < char_pos[j][0]].max()
        bridge = canvas[:, right_edge:right_edge + 1]
        canvas[:, right_edge:char_pos[j][0]] = np.where(
            bridge.sum(axis=2, keepdims=True) > 0,
            bridge, canvas[:, right_edge:char_pos[j][0]])
    if blur:
        canvas = cv2.GaussianBlur(canvas, (0, 0), blur)
    return canvas, char_pos


def hue_mask(canvas, hue_lo, hue_hi, sat_min=40, min_area=80):
    """颜色掩码：任务层给文字色相区间 → 文字掩码（背景剔除）。
    保留面积 ≥ min_area 的连通域（文字字符是大块，背景同色碎片是小块；
    文字字符间有间隙，不能只留最大连通域）。"""
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
    """掩码内列密度间隙切分。mask2d: 0/1 或 0/255。"""
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
    """先验切分（docs/194 六.7）：词长先验 + 视觉签名裁决切缝。"""
    result = []
    for x0, x1 in blocks:
        bw = x1 - x0 + 1
        if bw <= 52:
            result.append((x0, x1))
            continue
        n_chars = max(2, int(round(bw / 35)))
        best = None
        for cut in range(x0 + 8, x1 - 7):
            left = mask2d[:, max(0, x0 - 2):cut + 3]
            right = mask2d[:, cut - 2:x1 + 3]
            sl, sr = glyph_signature(left), glyph_signature(right)
            dl = min(sdist(sl, glyph_signature(lib[c])) for c in CHARS)
            dr = min(sdist(sr, glyph_signature(lib[c])) for c in CHARS)
            score = dl + dr
            if best is None or score < best[0]:
                best = (score, cut)
        if best:
            result.append((x0, best[1]))
            result.append((best[1], x1))
    return result


# 字库（黄字原型签名）
lib = {c: None for c in CHARS}


def build_lib():
    for c in CHARS:
        img = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(img, c, (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.2, TEXT_COLOR,
                    3, cv2.LINE_AA)
        mask = (img.sum(axis=2) > 100).astype(np.uint8)
        lib[c] = mask


def identify(mask2d, blocks):
    out = []
    for x0, x1 in blocks:
        img = mask2d[:, max(0, x0 - 2):x1 + 3]
        sig = glyph_signature(img)
        bc, bd = None, 1e9
        for c in CHARS:
            d = sdist(sig, glyph_signature(lib[c]))
            if d < bd:
                bc, bd = c, d
        out.append(bc)
    return out


def main():
    build_lib()
    word = "CATCH"
    # 黄字：HSV H≈30（黄），蓝底 H≈120（蓝）
    HUE_LO, HUE_HI = 15, 45       # 黄
    print("== 颜色×先验切分（黄字蓝底 + 背景纹理，C-A 粘连）==")

    # 场景 A：无粘连（测颜色掩码 vs 直接明度）
    canvas_a, _ = render_subtitle(word, glue=None)
    hm_a = hue_mask(canvas_a, HUE_LO, HUE_HI)
    gray_a = cv2.cvtColor(canvas_a, cv2.COLOR_BGR2GRAY)
    lum_a = (gray_a > 100).astype(np.uint8)
    post_lum = segment_posterior(lum_a)
    post_hue = segment_posterior(hm_a)
    print(f"\n① 无粘连场景：直接明度切分 {len(post_lum)} 块 vs 颜色掩码切分 "
          f"{len(post_hue)} 块（真值 5）")
    ids_lum = identify(lum_a, post_lum)
    ids_hue = identify(hm_a, post_hue)
    ok_lum = sum(1 for i, c in enumerate(ids_lum) if i < len(word) and c == word[i])
    ok_hue = sum(1 for i, c in enumerate(ids_hue) if i < len(word) and c == word[i])
    print(f"    明度: {''.join(ids_lum)} 对 {ok_lum}/5 | 颜色: {''.join(ids_hue)} 对 {ok_hue}/5")

    # 场景 B：粘连（测颜色掩码 + 先验）
    canvas_b, _ = render_subtitle(word, glue=(0, 1))
    hm_b = hue_mask(canvas_b, HUE_LO, HUE_HI)
    gray_b = cv2.cvtColor(canvas_b, cv2.COLOR_BGR2GRAY)
    lum_b = (gray_b > 100).astype(np.uint8)
    print(f"\n② 粘连场景（C-A）：")
    post_b = segment_posterior(hm_b)
    print(f"    颜色掩码切分: {len(post_b)} 块（真值 5，粘连应 <5）")
    ids_b = identify(hm_b, post_b)
    ok_b = sum(1 for i, c in enumerate(ids_b) if i < len(word) and c == word[i])
    print(f"    直接识别: {''.join(ids_b)} 对 {ok_b}/5")
    prior = segment_prior(hm_b, post_b, len(word))
    ids_p = identify(hm_b, prior)
    ok_p = sum(1 for i, c in enumerate(ids_p) if i < len(word) and c == word[i])
    print(f"    先验切分: {len(prior)} 块 → {''.join(ids_p)} 对 {ok_p}/5")

    print(f"\n判据：颜色掩码切分 ≥ 明度切分（背景纹理被剔除）；粘连时先验恢复 5/5")


if __name__ == "__main__":
    main()
