"""vision/language_internal.py — 语言内化：符号槽 + 态势转录 + 自反接口（无 LLM）。

外部对话第三部分的语言内化三层（用框架语言）：
  第一层 符号槽：维持对象自带符号标签（颜色确认通过→"红"；被拒→"可疑"；
      轨迹持续→"持续"）——标签由规则表自动贴，不是 LLM 生成
  第二层 态势转录：维持层状态（对象/颜色/被拒历史/重信任延迟）→ 确定性模板
      转录成自然语言——"翻译"不是"生成"（快、确定、不幻觉）
  第三层 自反接口：维持层对自身的注视——否定注册→"我刚才可能认错了"；
      时间一致性门选固执→"我不确定，但我先不调整"；纪念加深→"我需要更
      连续的证据才重新信任"

本文件把 docs/200 的 keep_reject（红球中段染蓝 + 被拒入史）接到三层语言上：
  每帧：感知状态（确认通过/被拒/重信任/被拒累计）→ 符号槽更新 → 转录一句
  语言输出 = 维持层状态的直接投影（docs/160 toText 的语言版）

输出示例：
  "我看到一个红球在移动"（确认通过 + 持续标签）
  "它变成蓝色了，我认不出来了"（染色段确认失败→符号槽"可疑"）
  "我不确定，但我先不调整"（时间门固执，docs/199b）
  "它又变红了，但我需要更连续的证据才重新信任"（纪念加深，docs/200）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import make_scene, W, H, N, RED

TRAIN_SEG = (30, 60)   # 染色段


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
        return 0.0, None
    frac = float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())
    return frac, float(np.median(rh[colored]))


class LanguageMind:
    """语言内化：感知状态 → 符号槽 → 转录（第二/三层）+ 自反（第三层）。

    第一层符号槽：颜色/被拒/持续 标签（规则表自动贴）
    第二层转录：模板把符号槽投影成句子
    第三层自反：否定/固执/纪念 → 对自身的评价句
    """

    COLOR_NAMES = {0: "红", 120: "蓝", 60: "黄", 90: "绿"}
    COLORS = [0.0, 60.0, 90.0, 120.0]

    def __init__(self, thr=0.2, grow=3.0, cap=60):
        self.thr = thr
        self.grow = grow
        self.cap = cap
        self.rejected = 0            # 被拒累计（纪念）
        self.trusted = True
        self.consec = 0
        self.slots = {"color": None, "suspect": False, "persist": 0}   # 符号槽
        self.tick = 0
        self.track_len = 0           # 目标持续帧数

    def _color_name(self, h):
        if h is None:
            return "?"
        return min(self.COLORS, key=lambda c: abs(c - h))

    def _k(self):
        """重信任所需连续帧（饱和累积，docs/203 阅历版）。"""
        return 1 + int(min(self.rejected, self.cap) / self.grow)

    def step(self, frac_red, med_hue):
        """一帧：更新符号槽，返回本帧语言（无变化返回 None）。"""
        self.tick += 1
        self.track_len += 1
        ok = frac_red > self.thr
        hue = self._color_name(med_hue) if med_hue is not None else None
        utt = None
        if ok:
            # 第一层：符号槽贴标签（确认通过 → 颜色标签）
            if hue == 0.0:
                self.slots["color"] = "红"
            self.slots["suspect"] = False
            self.track_len += 1
            if not self.trusted:                      # 重信任中（纪念）
                self.consec += 1
                if self.consec >= self._k():
                    self.trusted = True
                    self.slots["suspect"] = False
                    utt = f"它又变回红色了，我确认了它——但我花了 {self._k()} 帧才重新信任它。"
                else:
                    utt = f"它看起来是红的，但我需要更连续的证据（{self.consec}/{self._k()} 帧）才重新信任。"
            elif self.tick % 10 == 0:
                utt = f"我看到一个{self.slots['color']}球在移动，持续 {self.track_len} 帧了。"
        else:
            # 确认失败：被拒 → 符号槽"可疑" + 纪念累积 + 自反
            self.rejected += 1
            self.trusted = False
            self.consec = 0
            self.slots["suspect"] = True
            if med_hue is not None:
                self.slots["color"] = self.COLOR_NAMES.get(int(hue), "?")
            if self.rejected == 1:
                utt = f"它变成了{self.slots['color']}色，我认不出来了——我刚才可能认错了。"
            else:
                utt = f"还是{self.slots['color']}，不是红。我不确定，但我先不调整（第 {self.rejected} 次被拒）。"
        return utt


frames = make_interrupted()
mind = LanguageMind()
print("== 语言内化：感知状态 → 符号槽 → 转录 + 自反（无 LLM，模板引擎）==")
print(f"场景：红球，帧 {TRAIN_SEG[0]}-{TRAIN_SEG[1]} 被外部染成蓝（确认失败），之后恢复\n")
print(f"{'帧':>4s} {'感知':>6s} {'语言输出'}")
last_utt = None
for i, fr in enumerate(frames):
    frac, hue = ball_stats(fr, i)
    utt = mind.step(frac, hue)
    if utt is not None:
        # 感知状态摘要
        if frac > 0.2:
            sense = "确认红✓"
        else:
            sense = "拒(蓝)✗" if hue is not None and abs(hue - 120) < 30 else "拒?"
        print(f"{i:4d} {sense:>6s}  {utt}")
        last_utt = utt

print("\n== 三层语言内化映射 ==")
print("  第一层 符号槽：颜色标签(红/蓝)由确认通过/被拒规则自动贴；被拒→'可疑'")
print("  第二层 态势转录：'我看到一个红球在移动，持续 N 帧' = 维持层状态模板转录")
print("  第三层 自反接口：'我刚才可能认错了'/'我不确定但我先不调整'/'我需要更")
print("      连续的证据才重新信任' = 否定注册/固执/纪念的状态自反")
print("  无 LLM：全部是确定性规则 + 模板，输出可追溯到具体机制状态（docs/160")
print("      toText 的语言版，翻译不是生成，不会幻觉）")
