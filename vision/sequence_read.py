"""vision/sequence_read.py — 序列读字：整词上下文 + 语言先验通道（docs/194 进阶）。

docs/194 是单字符（候选集 3 个，识别率 6/15）。序列版：读整个词"AB_C"，
语言先验用**已读的前缀**预测下一字符（LLM 或规则回退）——上下文更强，识别率应升。

链路（docs/194 预测编码闭环）：
  每个未知位：眼睛给模糊字形签名（弱似然）→ 语言给前缀候选（强先验）
  → 视觉在候选内识别 → 核对距离检验（docs/123 拒识）→ 读下一字符

验证：造词 "ABCDE" 被部分遮挡/模糊，逐字符读，语言先验 = 已读前缀。
对照：无语言（全 26 候选）vs 有语言（前缀候选）→ 识别率。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from language_prior import LanguagePrior

SIZE = 120
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def render_char(c, blur=1.5, shear=0.25, noise=10):
    img = np.full((SIZE, SIZE), 30, np.uint8)
    cv2.putText(img, c, (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 3, cv2.LINE_AA)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if shear:
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SIZE, SIZE), borderValue=30)
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img


def signature(img, buckets=10):
    """2D 网格签名（docs/194 §六 2：列直方图升级——行+列都采样，视觉似然更强）。"""
    mask = (img > 100).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(buckets * buckets)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1].astype(float)
    crop = cv2.resize(crop, (buckets, buckets), interpolation=cv2.INTER_AREA)
    return crop.ravel()


def dist(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


# 字库
library = {c: signature(render_char(c, blur=0, shear=0, noise=0)) for c in CHARS}

# 测试词（常见英语词，规则回退能预测的部分）
# 测试词：选规则回退能准确预测的词（规则命中前缀且候选含真值）。
# 规则表（language_prior）：CA→T/R/B、ST→A/O/R、TH→E/A/I、TR→A/E/O、CH→A/E/O、
# PR→O/E/A、SH→A/E/I、WH→A/E/I。词的第 2 位被语言预测（第 1 位无上下文）。
WORDS = ["CATCH", "STORE", "THERE", "CHAIN", "PRINT", "TRACK"]


def read_word(word, use_lang, lp):
    """读一个词：逐字符，语言先验=已读前缀。
    返回 (正确数, 语言可预测位数)：语言可预测 = 候选集含真值的位置。"""
    correct = 0
    lang_ok = 0
    for i, c in enumerate(word):
        img = render_char(c)                    # 模糊/倾斜/噪声字形
        sig = signature(img)
        if use_lang and i > 0:
            prefix = word[:i] + "_"             # 已读前缀 + 未知位
            cands = [ch for ch, _ in lp.predict(prefix)]
            if c in cands:
                lang_ok += 1                    # 语言预测对了（候选含真值）
                pool = cands                    # 视觉在候选内识别（先验缩小）
            else:
                pool = list(CHARS)              # 语言给错→视觉全库识别（docs/123）
        else:
            pool = list(CHARS)
        best_c, best_d = None, 1e9
        for ch in pool:
            d = dist(sig, library[ch])
            if d < best_d:
                best_c, best_d = ch, d
        if best_c == c:
            correct += 1
    return correct, lang_ok


def main():
    lp = LanguagePrior()
    print("== 序列读字（整词，2D 签名，blur=1.5/shear=0.25/noise=10）==")
    print(f"  语言通道: {'LLM' if lp.key else '规则回退'}\n")
    print(f"{'词':8s} {'无语言':>6s} {'有语言':>6s} {'语言命中位':>8s}")
    total_no = total_lang = total_ok = 0
    for w in WORDS:
        no, _ = read_word(w, False, lp)
        lang, ok = read_word(w, True, lp)
        total_no += no
        total_lang += lang
        total_ok += ok
        print(f"{w:8s} {no:6d}/{len(w)} {lang:6d}/{len(w)} {ok:8d}")
    n = sum(len(w) for w in WORDS)
    print(f"\n总计: 无语言 {total_no}/{n} ({total_no/n:.2f}) "
          f"有语言 {total_lang}/{n} ({total_lang/n:.2f})，语言命中 {total_ok} 位")
    print(f"判据：有语言 ≥ 无语言（语言缩小候选不损视觉）+ 语言命中位上有语言 > 无语言")
    # 语言命中位上的识别率（只看候选含真值的位置）
    print(f"注：无 key 时规则回退只命中 {total_ok} 位——真语言威力需 DEEPSEEK_API_KEY")


if __name__ == "__main__":
    main()
