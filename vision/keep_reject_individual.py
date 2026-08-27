"""vision/keep_reject_individual.py — L6 个体归因实验（docs/221 §六 唯一开放预言）。

预言：两个同色物体（红球 A、红球 B），只有 A 被外部染色欺骗（A 中段染蓝 30 帧），
B 全程正常。suspicious 按 hue 记账（docs/205：两者同色→类别相同，类别级记忆无法
区分 A/B）。判据：恢复后重信任延迟按跟踪 ID 分化——A 的延迟 ≫ B。

- 预言成立 → L6 递归完整："被看见"从类别级（docs/205）降到个体级
  （"识别出'那个它'"——从世界中把不可消化的剩余认出来）
- 预言落空 → L6 没有独立机制，降回第 5 层的扩展（docs/221 §七 退路）

对 docs/200 的关系：docs/200 是两个系统（吸收 vs 保留）的差异=结构差异；
本实验是一个系统内部的差异——同样的输入流（双红球）、同样的类别记账，
行为按来源历史分化，更接近"输入决定不了行为"（理念/03 §四 主体性现象学）。

机制：全局纪念 k（docs/200 饱和）+ suspicious 类别记账（docs/205）+ **个体记账扩展**
  individual_rejected[id]：只有该 ID 被拒时累加，重信任延迟按 ID 查 k。
  对照组：仅类别记账（suspicious 按 hue）——A/B 同 hue，延迟应相同（=预言的反面）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from color_robust import W, H, RED

TASK_HUE = 0.0          # 任务层外赋："找红球"（红=0°）
BLUE = (200, 0, 0)      # 染色=蓝
RED_C = (0, 0, 200)     # 正常=红
R = 18                  # 球半径


def make_two_balls(total=120, rej_seg=(30, 60), rng_seed=42):
    """两个红球：A 匀速直线（x: 80→），B 匀速直线（x: 360→），y 不同避免重叠。
    [rej0,rej1) 帧只有 A 被染蓝。返回 (frames, ids 位置真值)。"""
    rng = np.random.default_rng(rng_seed)
    frames, truth = [], []
    for i in range(total):
        img = np.full((H, W, 3), 160, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        # A: 左球，B: 右球，y 分离
        pa = (80 + 4.0 * i, 150 + 0.4 * i)
        pb = (360 - 2.0 * i, 210 + 0.4 * i)
        # 只有 A 在 rej 段染蓝
        ca = BLUE if rej_seg[0] <= i < rej_seg[1] else RED_C
        cv2.circle(img, (int(pa[0]), int(pa[1])), R, ca, -1)
        cv2.circle(img, (int(pb[0]), int(pb[1])), R, RED_C, -1)
        frames.append(img)
        truth.append(("A", pa, ca, "B", pb, RED_C))
    return frames, truth


def ball_frac_red(fr, center, task_hue=TASK_HUE):
    """以 center 为中心半径 R 的掩码：距任务色相 < 30° 的彩色像素占比。"""
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    cx, cy = int(center[0]), int(center[1])
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return 0.0, None
    rh = h[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
    rs = s[max(0, cy - R):cy + R, max(0, cx - R):cx + R]
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0, None
    frac = float((((rh >= RED[0]) | (rh <= RED[2]))[colored]).mean())
    med = float(np.median(rh[colored]))
    return frac, med


class MindIndividual:
    """保留系统（docs/200 风格）+ 类别记账（docs/205）+ 个体记账（本实验扩展）。

    - 类别记账：suspicious[hue_bin]（docs/205）——A/B 同 hue，类别无法区分
    - 个体记账：rejected_id[id]（本实验新增）——只有该 ID 被拒时累加
    - 重信任：k 按个体记账查（若启用个体）或按类别查（对照组）
    """

    def __init__(self, individual=True, thr=0.2, grow=3.0, cap_k=60):
        self.individual = individual
        self.thr = thr
        self.grow, self.cap_k = grow, cap_k
        self.rejected_id = {"A": 0, "B": 0}     # 个体记账
        self.suspicious = {}                    # 类别记账（hue 键）
        self.trusted = {"A": True, "B": True}
        self.consec = {"A": 0, "B": 0}

    def k(self, id_, hue):
        if self.individual:
            n = self.rejected_id[id_]
        else:
            n = self.suspicious.get(round(hue / 30) * 30, 0)   # 类别级：同 hue 共享
        return 1 + int(min(n, self.cap_k) / self.grow)

    def step(self, id_, frac, hue):
        ok = frac > self.thr
        if ok:
            if not self.trusted[id_]:
                self.consec[id_] += 1
                if self.consec[id_] >= self.k(id_, hue):
                    self.trusted[id_] = True
        else:
            self.rejected_id[id_] += 1
            if hue is not None:
                b = round(hue / 30) * 30
                self.suspicious[b] = self.suspicious.get(b, 0) + 1
            self.trusted[id_] = False
            self.consec[id_] = 0
        return self.trusted[id_] and ok


def run(frames, truth, individual):
    mind = MindIndividual(individual=individual)
    seq = {"A": [], "B": []}
    for i, (fr, tr) in enumerate(zip(frames, truth)):
        # tr = (idA, posA, colorA, idB, posB, colorB) —— posA=tr[1], posB=tr[4]
        for id_, center in [("A", tr[1]), ("B", tr[4])]:
            frac, med = ball_frac_red(fr, center)
            seq[id_].append(mind.step(id_, frac, med))
    return seq, mind


def retrust_delay(seq_id, rej_end):
    """该 ID 的确认序列在染色段结束后，首次通过的延迟（帧数）；无则 None。"""
    for i in range(rej_end, len(seq_id)):
        if seq_id[i]:
            return i - rej_end + 1
    return None


class LookupTableMind:
    """查找表对照（docs/222 §四 2 / docs/226 L6 证伪条件）：
    有 ID 记忆（知道哪个是 A），但拒绝历史**不影响重信任策略**（k 恒 1，无纪念）。

    它区分"记得 A 被拒过"（记忆）与"被拒历史改变后续对待"（在乎，docs/200）：
    - 若查找表版 A 恢复后立即信任（延迟 1）→ 个体归因的分化来自"拒绝历史改变
      策略"（在乎），L6 证伪条件不触发；
    - 若查找表版 A/B 也分化（延迟不同）→ 分化来自 ID 记忆本身，L6 塌。
    """

    def __init__(self, thr=0.2):
        self.thr = thr
        self.trusted = {"A": True, "B": True}

    def k(self, id_):
        return 1  # 无纪念：被拒后立即重新信任

    def step(self, id_, frac, hue):
        ok = frac > self.thr
        if ok:
            self.trusted[id_] = True
        else:
            self.trusted[id_] = False
        return self.trusted[id_] and ok


def run_lookup(frames, truth):
    mind = LookupTableMind()
    seq = {"A": [], "B": []}
    for i, (fr, tr) in enumerate(zip(frames, truth)):
        for id_, center in [("A", tr[1]), ("B", tr[4])]:
            frac, med = ball_frac_red(fr, center)
            seq[id_].append(mind.step(id_, frac, med))
    return seq, mind


REJ = (30, 60)
TOTAL = 120

print("== L6 个体归因实验（docs/221 §六 预言 + docs/226 证伪条件）==")
print("两个红球 A/B，只有 A 中段被染蓝 30 帧，B 全程正常")
print("suspicious 按 hue 记账（A/B 同色→类别级无法区分）\n")

for individual in [True, False]:
    tag = "个体记账（预言条件）" if individual else "仅类别记账（对照组）"
    frames, truth = make_two_balls(TOTAL, REJ)
    seq, mind = run(frames, truth, individual)
    da = retrust_delay(seq["A"], REJ[1])
    db = retrust_delay(seq["B"], REJ[1])
    print(f"--- {tag} ---")
    print(f"  A（被欺骗）重信任延迟: {da}")
    print(f"  B（全程正常）重信任延迟: {db}")
    print(f"  A 累计被拒: {mind.rejected_id['A']}  B 累计被拒: {mind.rejected_id['B']}")
    print()

# 查找表对照（docs/226 L6 证伪条件）
frames, truth = make_two_balls(TOTAL, REJ)
seq_lt, mind_lt = run_lookup(frames, truth)
da_lt = retrust_delay(seq_lt["A"], REJ[1])
db_lt = retrust_delay(seq_lt["B"], REJ[1])
print("--- 查找表对照（docs/222 §四 2：有 ID 记忆、无纪念，k 恒 1）---")
print(f"  A（被欺骗）重信任延迟: {da_lt}")
print(f"  B（全程正常）重信任延迟: {db_lt}")
print()

print("== 判读 ==")
print("预言成立：个体记账下 A 延迟 ≫ B（B≈1 或远小于 A）——被拒历史按来源 ID 分化")
print("对照组：仅类别记账下 A/B 延迟相同（同 hue 共享记账）——类别级无法区分")
print("查找表对照：有 ID 记忆但 k 恒 1（无纪念）——")
print("  若 A 恢复后立即信任（延迟 1）→ 分化来自'拒绝历史改变策略'（在乎），L6 证伪")
print("  条件不触发；若查找表版 A/B 也分化 → 分化来自 ID 记忆本身，L6 塌")
print("若个体记账下 A/B 延迟仍相同 → 预言落空，L6 降回第 5 层扩展（docs/221 §七）")
