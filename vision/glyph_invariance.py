"""vision/glyph_invariance.py — 字形不变性：扭曲对笔画签名的影响 + 几何归一化（docs/193 进阶）。

人的字形识别范围广（字体/倾斜/缩放/粗细/手写）——不变性机制（docs/133 §一.5 解缠）：
把字体/大小/倾斜当可压掉的冗余。本实验量化：
  1. 同一字母"A"的不同变体（倾斜/缩放/粗细/手写扭曲）→ 笔画签名
  2. 几何归一化（倾斜校正→尺度归一→笔画细化）前后，变体签名与原型匹配度

判据：归一化后，变体签名与"A"原型匹配度显著高于归一化前（且 B 的匹配度更低——
区分度保持）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2

SIZE = 200


def render(text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=3.0, thick=3,
           shear=0.0, rot=0.0, resize=1.0):
    """渲染文字变体：shear=水平倾斜(px偏移比例)，rot=旋转(度)，resize=缩放。"""
    img = np.full((SIZE, SIZE), 30, np.uint8)
    cv2.putText(img, text, (40, 150), font, scale, 255, thick, cv2.LINE_AA)
    if shear:
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=30)
    if rot:
        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rot, 1.0)
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=30)
    if resize != 1.0:
        img = cv2.resize(img, None, fx=resize, fy=resize)
        img = cv2.copyMakeBorder(img, 0, SIZE - img.shape[0], 0,
                                 SIZE - img.shape[1], cv2.BORDER_CONSTANT, value=30)
    return img


def glyph_signature(img, buckets=10):
    """笔画列直方图签名（saccade_read2 同款）。返回 0/1 向量。"""
    mask = (img > 100).astype(np.uint8)
    cols = mask.sum(axis=0)
    if cols.max() == 0:
        return np.zeros(buckets)
    nz = np.nonzero(cols)[0]
    cols = cols[nz.min():nz.max() + 1]
    parts = np.array_split(cols, buckets)
    return np.array([1.0 if np.mean(p) > cols.max() * 0.3 else 0.0 for p in parts])


def normalize(img):
    """几何归一化：① 主轴旋转校正（PCA——解缠倾斜/旋转，docs/133）② 外接框裁剪
    ③ 统一缩放。"""
    mask = (img > 100).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return img
    # PCA 主轴：字符点云的主方向
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    main_axis = evecs[:, np.argmax(evals)]           # 主方向（字符的长轴）
    ang = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
    if ang > 45:                                     # 归一化到近竖直（文字是竖的）
        ang -= 90
    if abs(ang) > 1.0:
        M = cv2.getRotationMatrix2D(tuple(mean), ang, 1.0)
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=30)
    # 外接框裁剪 + 统一缩放
    mask2 = (img > 100).astype(np.uint8)
    ys2, xs2 = np.nonzero(mask2)
    if len(xs2) == 0:
        return img
    x0, x1, y0, y1 = xs2.min(), xs2.max(), ys2.min(), ys2.max()
    crop = img[y0:y1 + 1, x0:x1 + 1]
    crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
    return crop


def match(a, b):
    """签名匹配度：交集/并集（Jaccard）。"""
    return float(np.logical_and(a, b).sum()) / max(np.logical_or(a, b).sum(), 1)


def main():
    variants = {
        "A原型": render("A"),
        "A倾斜": render("A", shear=0.3),
        "A旋转15°": render("A", rot=15),
        "A缩小": render("A", resize=0.7),
        "A加粗": render("A", thick=6),
        "A细体": render("A", thick=1),
        "A衬线体": render("A", font=cv2.FONT_HERSHEY_COMPLEX),
        "B原型": render("B"),
    }
    print("== 字形不变性（变体签名 vs A 原型）==")
    print(f"{'变体':10s} {'归一化前':>8s} {'归一化后':>8s}")
    proto = glyph_signature(render("A"))
    proto_n = glyph_signature(normalize(render("A")))
    for name, img in variants.items():
        s = glyph_signature(img)
        sn = glyph_signature(normalize(img))
        print(f"{name:10s} {match(s, proto):8.3f} {match(sn, proto_n):8.3f}")
    print("\n判据：A 变体归一化后匹配度升（不变性）、B 匹配度保持低（区分度不丢）")


if __name__ == "__main__":
    main()
