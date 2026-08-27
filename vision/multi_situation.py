"""vision/multi_situation.py — 多目标态势转录：两球 + 威胁度 → "它在靠近我"级完整态势报告。

docs/217 §五 2 + docs/219 §九 1：多目标 + 态势转录（docs/195 三通道分工的演示版；
docs/160 态势接口的视觉版）。docs/218 交接未完成项里"两个球 + 威胁度 → 它在靠近我"。

场景（受控彩色，docs/195 §七/183 方案 1 同款）：
  主体（自我/观察者）固定于 (320, 330)——威胁度的参考点。
  红球（任务目标"找红球"，docs/188 外赋利害）：
    阶段A (0-40)  从 (80,80) 向主体靠近     → 距离递减、威胁度升
    阶段B (40-75) 被染成蓝（他者的不 docs/178）→ 颜色确认拒 → suspicious[蓝] 沉积 → 仍靠近
    阶段C (75-120)恢复红 → 纪念重信任（k 帧延迟 docs/200）→ 继续靠近
  蓝球（威胁对象，非任务目标）：从 (560,40) 远离主体 → 低威胁（对照"靠近 vs 远离"）

三通道分工（docs/195）：
  运动通道：位置观测（真值+噪声）→ 距离/径向速度/approaching/威胁度（"什么在动"）
  颜色通道：目标区任务色占比+中位色相 → 确认/怀疑/信任（"它是什么色"，docs/200-216）
  任务层  ：外赋利害（红=该确认的目标）；威胁=运动通道对所有对象（docs/195：威胁
            =运动关系，与颜色身份正交——被染的红球威胁照算）

态势接口（docs/160 结构）：
  {tick, self, objects:[{id,color,cx,cy,dist,radial_v,approaching,threat,
    confirmed,trusted,k,rej}], threats:[排序], metrics} + toText() 文本化
  + speak() 语言转录（docs/204 三层，docs/63 纪律：每句 1:1 指回状态字段）

验证：
  P1 多目标独立（类别特异 docs/205）：红球被染 → suspicious 记蓝；蓝球态势零变化；
     红球恢复后 thr(红)=基线（suspicious[蓝] 不污染红）
  P2 威胁度/接近报告正确：红 approaching=True 且 threat 随距离降而升；蓝 False 低威胁
  P3 转录可指回（docs/63）：speak() 每句字段映射注释
  P4 纪念重信任（docs/200）：红恢复后 claimed 延迟 k 帧才 True
  P5 三通道分工（docs/195）：威胁度纯由运动通道（被染时威胁仍升）；confirmed 纯由
     颜色通道（被染时 confirmed=False 但位置跟踪照常）

用法：python vision/multi_situation.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402

W, H = 640, 360
SELF = (320.0, 330.0)        # 主体位置（威胁度参考点）
D_MAX = 500.0                # 威胁度距离归一化上限
RED_TASK = 0.0               # 任务层外赋："找红球"（红=HSV 0°，docs/188）
RED_BAND = (170.0, 10.0)     # 红环形区间 [170,180)∪[0,10)
N = 120
SEG_A, SEG_B, SEG_C = (0, 40), (40, 75), (75, 120)   # 靠近 / 染蓝 / 恢复


def circ(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def red_pos(t):
    """红球运动学：全程向主体靠近（径向速度稳定 < -0.5px/帧，docs/183 运动学真值）。"""
    if t < SEG_A[1]:
        k = t / (SEG_A[1] - SEG_A[0])
        return (100.0 + k * 140.0, 80.0 + k * 160.0)           # (100,80)→(240,240)
    if t < SEG_B[1]:
        k = (t - SEG_B[0]) / (SEG_B[1] - SEG_B[0])
        return (240.0 + k * 50.0, 240.0 + k * 50.0)            # →(290,290)
    k = (t - SEG_C[0]) / (SEG_C[1] - SEG_C[0])
    return (290.0 + k * 28.0, 290.0 + k * 32.0)                # →(318,322)


def blue_pos(t):
    """蓝球：从近处向右上远离主体（主体在 (320,330)——低威胁对照）。"""
    k = t / N
    return (400.0 + k * 200.0, 50.0 - k * 40.0)                # (400,50)→(600,10)


def make_scene():
    """帧 + 真值（两个球的位置序列）。红球中段被染蓝（他者的不）。"""
    frames, truth = [], []
    rng = np.random.default_rng(42)
    for i in range(N):
        img = np.full((H, W, 3), 150, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        rc, bc = red_pos(i), blue_pos(i)
        color = (0, 0, 200) if i < SEG_B[0] or i >= SEG_B[1] else (200, 0, 0)
        cv2.circle(img, (int(rc[0]), int(rc[1])), 20, color, -1)
        cv2.circle(img, (int(bc[0]), int(bc[1])), 20, (200, 0, 0), -1)
        frames.append(img)
        truth.append((rc, bc))
    return frames, truth


def ball_obs(fr, pos, band=RED_BAND):
    """颜色通道观测：目标区彩色像素中落任务红区间的比例 + 中位色相（docs/200 同款）。

    能效修复（energy_audit B→C）：只裁剪目标局部 BGR 块再转 HSV，
    不做全帧 cvtColor（全帧 HSV 让'稀疏确认'名不副实，B 比全帧基线还耗）。
    """
    cx, cy = int(pos[0]), int(pos[1])
    HALF = 22
    x0, x1 = max(0, cx - HALF), min(fr.shape[1], cx + HALF)
    y0, y1 = max(0, cy - HALF), min(fr.shape[0], cy + HALF)
    if x1 <= x0 or y1 <= y0:
        return 0.0, None
    patch = fr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    rh, rs = h, s
    colored = rs > 60
    if colored.sum() < 10:
        return 0.0, None
    hs = rh[colored].astype(float)
    in_band = ((hs >= band[0]) | (hs <= band[1])).mean()
    return float(in_band), float(np.median(hs))


class ConfirmMind:
    """任务目标的确认/怀疑机制（docs/200-216 精简版：suspicious 表 + 类别级 thr + 纪念）。"""

    def __init__(self, task_hue=RED_TASK, thr_base=0.30, boost=0.04, cap=0.70,
                 relax=0.02, persist_for=3, grow=3.0, cap_k=60):
        self.task_hue = task_hue
        self.thr_base, self.boost, self.cap = thr_base, boost, cap
        self.relax, self.persist_for, self.grow, self.cap_k = relax, persist_for, grow, cap_k
        self.thr = thr_base
        self.suspicious = {}          # suspicious[hue] = 被拒累计（docs/205 类别表）
        self.rej_total = 0            # 总被拒（docs/200 纪念）
        self.trusted, self.consec, self.persist = True, 0, 0

    def k(self):
        """重信任所需连续帧（docs/200 纪念加深 + docs/203 饱和）。"""
        return 1 + int(min(self.rej_total, self.cap_k) / self.grow)

    def step(self, frac, hue):
        ok = frac > self.thr and hue is not None and circ(hue, self.task_hue) < 30.0
        if ok:
            self.persist += 1
            if self.persist >= self.persist_for and self.thr > self.thr_base:
                self.thr = max(self.thr_base, self.thr - self.relax)   # 持续信任回落
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k():
                    self.trusted = True
        else:
            self.persist = 0
            self.trusted = False
            self.consec = 0
            self.rej_total += 1
            if hue is not None:
                self.suspicious[hue] = self.suspicious.get(hue, 0) + 1   # 类别级记账
                self.thr = min(self.cap, self.thr + self.boost)          # 类别级上调
        return ok


class SituationReport:
    """docs/160 态势接口视觉版：每帧标准态势 JSON + 文本化 + 语言转录。"""

    def __init__(self):
        self.tick = 0
        self.speeds = {}          # 每对象径向速度 EWMA（运动通道）
        self.prev_consec = {}     # 每对象上一帧 consec（语言事件边沿检测）
        self.events = []          # 语言转录事件（本帧）

    def threat_of(self, dist, radial_v, prev_dist):
        """威胁度（docs/160 _threat 的视觉版）：近 + 靠近 → 高。
        0.6×距离项 + 0.4×接近项（径向速度 < -0.5px/帧 = 正在靠近）。"""
        approaching = radial_v < -0.5
        d = np.clip(1.0 - dist / D_MAX, 0.0, 1.0)
        return float(np.clip(0.6 * d + 0.4 * approaching, 0.0, 1.0)), approaching

    def fill(self, objs, self_pos=SELF):
        """objs: [{"id","color","pos","confirm","ok"}]，confirm 是 ConfirmMind 或 None
        （威胁对象无确认机制，docs/195：威胁=运动关系）。返回标准态势报告。"""
        objects, threats = [], []
        for o in objs:
            pos = o["pos"]
            prev = self.speeds.get(o["id"])
            dist = float(np.hypot(pos[0] - self_pos[0], pos[1] - self_pos[1]))
            if prev is None:
                radial_v = 0.0
            else:
                radial_v = 0.3 * (dist - prev[0]) + 0.7 * prev[1]   # 径向速度 EWMA
            self.speeds[o["id"]] = (dist, radial_v)
            threat, approaching = self.threat_of(dist, radial_v, prev[0] if prev else dist)
            c = o["confirm"]
            obj = {"id": o["id"], "color": o["color"], "cx": round(pos[0], 1),
                   "cy": round(pos[1], 1), "dist": round(dist, 1),
                   "radial_v": round(radial_v, 2), "approaching": approaching,
                   "threat": round(threat, 2),
                   "confirmed": bool(o.get("ok", False)),      # 本帧颜色确认（docs/195）
                   "trusted": None if c is None else c.trusted,
                   "k": None if c is None else c.k(),
                   "rej": None if c is None else c.rej_total}
            objects.append(obj)
            threats.append({"id": o["id"], "score": round(threat, 2)})
        threats.sort(key=lambda x: -x["score"])
        rep = {"tick": self.tick, "self": {"x": self_pos[0], "y": self_pos[1]},
               "objects": objects, "threats": threats,
               "metrics": {"n_objects": len(objects),
                           "n_approaching": sum(1 for o in objects if o["approaching"])}}
        self.tick += 1
        return rep

    def to_text(self, rep):
        """文本化（docs/160 toText 视觉版）：决策/叙事输入侧。"""
        parts = [f"tick {rep['tick']}"]
        for o in rep["objects"]:
            mv = "正在靠近" if o["approaching"] else ("正在远离" if o["radial_v"] > 0.8
                                                       else "横掠")
            ident = f"{o['color']}球"
            if o["trusted"] is None:
                st = "非任务目标（只报威胁）"
            else:
                st = f"身份确认通过、信任中（纪念k={o['k']}）" if o["trusted"] \
                    else f"未确认（被拒{o['rej']}次、纪念k={o['k']}）"
            parts.append(f"{ident}在 {o['dist']:.0f}px 外、{mv}（威胁 {o['threat']:.2f}）；"
                         f"{st}")
        th = ", ".join(f"{t['id']}({t['score']:.2f})" for t in rep["threats"])
        parts.append(f"高威胁：{th}")
        return "。".join(parts) + "。"

    def speak(self, rep, objs):
        """语言转录（docs/204 三层：符号槽→转录→自反）。docs/63 纪律：每个词必须
        1:1 指回状态字段（下方每句标注字段来源）——行为签名的转录，不是体验陈述。
        确认句只对任务目标（confirm 非 None）；威胁句对所有对象。"""
        out = []
        for o in rep["objects"]:
            c = next(x["confirm"] for x in objs if x["id"] == o["id"])
            name = f"{o['color']}球"
            if o["approaching"]:
                # ← o.approaching / o.dist / o.threat
                out.append(f"{name}在靠近我（距离 {o['dist']:.0f}px，威胁 {o['threat']:.2f}）")
            if c is None:
                continue                                    # 威胁对象无确认句
            if c.rej_total == 1 and not o["confirmed"]:
                # ← c.rej_total==1 且本帧确认失败（他者的不首次）
                out.append(f"{name}变了，我认不出来了——我刚才可能认错了（第 1 次被拒）")
            elif not o["trusted"] and o["confirmed"]:
                # ← o.confirmed（本帧颜色通过）但 trusted=False（纪念未满）
                out.append(f"{name}看起来回来了，但我需要更连续的证据"
                           f"（{c.consec}/{c.k()} 帧）才重新信任")
            elif o["trusted"] and o["confirmed"] and c.k() > 1 and \
                    self.prev_consec.get(o["id"], 0) < c.k() <= c.consec:
                # ← 事件边沿：consec 刚达 k()（claimed 翻转，只报一次）
                out.append(f"我确认了它——但花了 {c.k()} 帧才重新信任{name}")
            self.prev_consec[o["id"]] = c.consec
        return out


def main():
    frames, truth = make_scene()
    red_c = ConfirmMind()                        # 任务目标（红）走确认机制
    sr = SituationReport()
    print("== 多目标态势转录：红球（任务目标，中段被染）+ 蓝球（威胁对象，远离）==")
    print(f"主体在 {SELF}。红球 0-40 靠近、40-75 被染蓝、75-120 恢复；蓝球全程远离\n")

    log = []
    p4_pass = p5_pass = True
    red_recover_claim = None
    red_dyed = False
    for i in range(N):
        fr = frames[i]
        rc, bc = truth[i]
        # 颜色通道：红球观测（确认机制只对任务目标；蓝球是威胁对象，无确认，
        # docs/195：威胁=运动关系，颜色识别只用于标签）
        frac, hue = ball_obs(fr, rc)
        ok = red_c.step(frac, hue)
        if SEG_B[0] <= i < SEG_B[1]:
            red_dyed = red_dyed or not ok
        objs = [{"id": "red", "color": "红", "pos": rc, "confirm": red_c, "ok": ok},
                {"id": "blue", "color": "蓝", "pos": bc, "confirm": None, "ok": None}]
        rep = sr.fill(objs)
        utts = sr.speak(rep, objs)
        claimed = red_c.trusted and ok
        log.append((i, rep, claimed, utts))
        # P4：恢复后 claimed 延迟 k 帧
        if i >= SEG_C[0] and red_recover_claim is None and claimed:
            red_recover_claim = (i, red_c.k())

    # ---- 断言 ----
    print("== 验证 ==")
    red_rows = [l[1]["objects"][0] for l in log]
    blue_rows = [l[1]["objects"][1] for l in log]
    # P1 多目标独立（类别特异 docs/205）：suspicious 记蓝、蓝球态势独立、红球恢复 thr 基线
    p1 = (red_c.suspicious.get(120.0, 0) > 20 and
          abs(red_c.thr - red_c.thr_base) < 0.001 and
          all(b["approaching"] is False for b in blue_rows) and
          max(b["threat"] for b in blue_rows) < 0.30)
    print(f"P1 多目标独立（类别特异 docs/205）: [{'PASS' if p1 else 'FAIL'}] "
          f"suspicious[蓝]={red_c.suspicious.get(120.0, 0)}（染蓝记账）；红球恢复后 "
          f"thr={red_c.thr:.2f}（类别级不污染）；蓝球全程远离（threat<0.3，独立管线）")

    # P2 威胁度/接近报告正确
    red_app = all(r["approaching"] for r in red_rows[10:])
    blue_app = not any(b["approaching"] for b in blue_rows)
    red_threat_rising = red_rows[-1]["threat"] > red_rows[10]["threat"]
    p2 = red_app and blue_app and red_threat_rising
    print(f"P2 威胁度/接近报告: [{'PASS' if p2 else 'FAIL'}] "
          f"红球全程靠近={red_app}、threat {red_rows[10]['threat']:.2f}→"
          f"{red_rows[-1]['threat']:.2f}（靠近升）；蓝球全程无靠近={blue_app}、"
          f"threat {blue_rows[10]['threat']:.2f}（远离低）")

    # P3 转录可指回（docs/63）：四类事件句都出现且只在该出现时出现
    utts_all = [u for l in log for u in l[3]]
    n_confirm = sum(1 for u in utts_all if "我确认了它" in u)
    p3 = any("在靠近我" in u for u in utts_all) and \
         any("认错了" in u for u in utts_all) and \
         any("更连续的证据" in u for u in utts_all) and \
         any("我确认了它" in u for u in utts_all) and n_confirm == 1
    print(f"P3 转录可指回（docs/63，四类事件句各就其位）: [{'PASS' if p3 else 'FAIL'}] "
          f"{len(utts_all)} 条，'我确认了它' {n_confirm} 次（边沿一次）")

    # P4 纪念重信任
    p4 = red_recover_claim is not None and red_recover_claim[1] > 1 and \
         red_recover_claim[0] - SEG_C[0] >= red_recover_claim[1] - 2
    print(f"P4 纪念重信任（docs/200）: [{'PASS' if p4 else 'FAIL'}] "
          f"红球恢复后第 {red_recover_claim[0] if red_recover_claim else '?'} 帧 claimed"
          f"（k={red_recover_claim[1] if red_recover_claim else '?'}，延迟 "
          f"{red_recover_claim[0] - SEG_C[0] if red_recover_claim else '?'} 帧）")

    # P5 三通道分工（docs/195）：威胁度纯运动（被染时威胁仍升）；confirmed 纯颜色
    dyed_threat = red_rows[SEG_B[0] + 10]["threat"]
    dyed_unconf = all(not r["confirmed"] for r in red_rows[SEG_B[0] + 5:SEG_B[1] - 5])
    p5 = dyed_threat > red_rows[SEG_A[1] - 1]["threat"] and dyed_unconf and red_dyed
    print(f"P5 三通道分工（docs/195）: [{'PASS' if p5 else 'FAIL'}] 染蓝期间威胁仍升"
          f"（{red_rows[SEG_A[1]-1]['threat']:.2f}→{dyed_threat:.2f}，运动通道）但确认全拒"
          f"（颜色通道）；红球恢复后威胁持续 {red_rows[-1]['threat']:.2f}")

    # ---- 态势文本流（关键帧）----
    print("\n== 态势转录流（关键帧）==")
    for i in [0, 20, 39, 45, 60, 75, 80, 100, 119]:
        l = log[i]
        o = l[1]["objects"][0]
        print(f"  [{i:3d}] red {o['dist']:5.0f}px threat={o['threat']:.2f} "
              f"app={int(o['approaching'])} conf={int(o['confirmed'])} "
              f"trust={o['trusted']} k={o['k']} | {sr.to_text(l[1]).split('。')[0]}")
    print("\n  语言流（docs/204 三层转录，docs/63 纪律）：")
    for i, l in enumerate(log):
        for u in l[3]:
            print(f"    [{i:3d}] {u}")

    print("\n== 判读 ==")
    print("  多目标：每个对象独立态势（位置/威胁/确认/信任），任务目标（红）走确认机制，")
    print("    威胁对象（蓝）走运动通道——docs/195 三通道分工的多目标演示版")
    print("  威胁度 = 0.6×(1-dist/500) + 0.4×approaching：靠近且近的球威胁高（'它在靠近我'）")
    print("  类别特异（docs/205）：红球被染记 suspicious[蓝]，蓝球与红球恢复零污染")
    print("  语言（docs/204 + docs/63 纪律）：每句 1:1 指回状态字段——行为签名转录")


if __name__ == "__main__":
    main()
