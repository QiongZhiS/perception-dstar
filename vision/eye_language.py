"""vision/eye_language.py — 眼+语言配合读字（docs/125 预期被世界纠正 × docs/123 幻觉纪律）。

人读字 = 预测编码闭环：眼睛给弱似然（模糊字形签名），语言给强先验（上下文候选集），
后验 = 先验 × 似然（候选集内视觉确认），视觉核对纠正（docs/125）。

实验：
  1. 视觉侧：渲染字母（加模糊/倾斜/噪声 = 弱视觉），列直方图签名 + 小字库（A-Z）
  2. 语言侧：上下文候选集（规则化：如 "AB_" 后只能 {C,G,O}）
  3. 识别 = 视觉签名在候选集内选最近；对照 = 无上下文（全 26）
  4. 幻觉纪律：语言给错候选（候选集不含真值）→ 视觉核对应"拒绝"而非"硬选"

判据：
  P1 有上下文识别率 > 无上下文（语言先验缩小候选）
  P2 语言给错候选时，视觉核对拒识（不误信先验——docs/123）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2

SIZE = 120
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render_char(c, blur=0, shear=0.0, noise=0):
    img = np.full((SIZE, SIZE), 30, np.uint8)
    cv2.putText(img, c, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 3, cv2.LINE_AA)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if shear:
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=30)
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def signature(img, buckets=10):
    mask = (img > 100).astype(np.uint8)
    cols = mask.sum(axis=0)
    if cols.max() == 0:
        return np.zeros(buckets)
    nz = np.nonzero(cols)[0]
    cols = cols[nz.min():nz.max() + 1]
    parts = np.array_split(cols, buckets)
    return np.array([np.mean(p) for p in parts])


def match(a, b):
    """签名距离（L2）。"""
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ---- 建字库（原型签名，无失真）----
library = {c: signature(render_char(c)) for c in CHARS}

# 上下文规则：模拟语言先验（docs/107 猜中预期）
# "AB_" 后 = {C, G, O}；"X_" 后 = {B, E, P} 等——一组候选集
CONTEXT = {"C": ["C", "G", "O"], "B": ["B", "E", "P"], "A": ["A", "H", "R"],
           "D": ["D", "O", "Q"], "F": ["F", "E", "T"]}


def identify(sig, candidates=None):
    """视觉识别：签名在候选集（或全集）内找最近。返回 (字符, 距离)。"""
    pool = candidates if candidates else list(CHARS)
    best_c, best_d = None, 1e9
    for c in pool:
        d = match(sig, library[c])
        if d < best_d:
            best_c, best_d = c, d
    return best_c, best_d


def run(blur=2.0, shear=0.3, noise=12):
    """测试一字符：渲染失真 → 视觉识别（有/无上下文）。返回结果。"""
    rng = np.random.default_rng(42)
    results = []
    for c in CONTEXT:
        img = render_char(c, blur=blur, shear=shear, noise=noise)
        sig = signature(img)
        full_c, full_d = identify(sig)                       # 无上下文
        ctx = CONTEXT[c]
        ctx_c, ctx_d = identify(sig, ctx)                    # 有上下文
        wrong_ctx = [x for x in ctx if x != c]               # 语言给错候选
        reject = False
        if wrong_ctx:
            # 幻觉纪律：候选集不含真值时，视觉核对应拒识（最近的候选距离太大）
            fake_c, fake_d = identify(sig, wrong_ctx)
            reject = fake_d > ctx_d * 1.5 or fake_d > 8.0    # 距离显著大于正常候选
        results.append((c, full_c, ctx_c, reject))
    return results


def main():
    print("== 眼+语言配合读字（模糊/倾斜/噪声字形）==")
    print("P1 有上下文识别率 > 无上下文；P2 语言给错候选时视觉核对拒识（docs/123）\n")
    full_ok = ctx_ok = rej_ok = 0
    total = 0
    for blur, shear, noise in [(2.0, 0.3, 12), (3.0, 0.5, 20), (4.0, 0.7, 30)]:
        print(f"-- 失真 blur={blur} shear={shear} noise={noise} --")
        print(f"{'真值':4s} {'无上下文':>10s} {'有上下文':>10s} {'核对拒识':>8s}")
        res = run(blur, shear, noise)
        for c, full_c, ctx_c, rej in res:
            fo = "✓" if full_c == c else f"✗({full_c})"
            co = "✓" if ctx_c == c else f"✗({ctx_c})"
            ro = "✓" if rej else "✗"
            print(f"{c:4s} {fo:>10s} {co:>10s} {ro:>8s}")
            full_ok += full_c == c
            ctx_ok += ctx_c == c
            rej_ok += rej
            total += 1
    print(f"\nP1: 无上下文 {full_ok}/{total} vs 有上下文 {ctx_ok}/{total}"
          f"  [{'PASS' if ctx_ok > full_ok else 'FAIL'}]")
    print(f"P2: 语言给错候选时核对拒识 {rej_ok}/{total}  [{'PASS' if rej_ok >= total * 0.5 else 'FAIL'}]")


if __name__ == "__main__":
    main()
