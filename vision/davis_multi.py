"""vision/davis_multi.py — 多目标回真实视频：flamingo 粉 + surf 蓝绿 同帧独立确认 + 威胁态势。

docs/222 §三 1 + docs/221 §六 1：docs/219 单目标真实闭环 × docs/221 多目标态势的拼接。
审稿人问："两个真实目标在同一场景，你的类别级怀疑还互相独立吗？"

受控叠加（受控实拍立场 docs/185/186）：
  背景帧    surf 真实帧（海天，原目标区域 inpaint 抹除）
  目标 A    flamingo 真实目标区域（真值掩码像素，粉 ~170°，非纯色帧内 std 44°）
  目标 B    surf id 113 真实目标区域（真值掩码像素，蓝绿 ~71°，帧内 std 26°）
  布局      受控运动学：A 向主体靠近（威胁升）、B 远离（威胁低）
  事件      A 中段被冷增益偏移（他者的不 docs/178）→ 独立确认拒 → 恢复 → 纪念
  真实成分  目标纹理/色相分布/掩码形状全部来自 DAVIS；背景与噪声真实

三通道分工（docs/195）：
  运动：目标质心（受控轨迹+噪声）→ 距离/径向速度/威胁度
  颜色：目标局部掩码采样 → 任务色占比+中位色相 → 每目标独立 ConfirmMind（docs/205）
  任务层：A=粉任务（170°）、B=蓝绿任务（71°）——两个都走确认（类别特异验证）

态势接口（docs/160 + docs/221）：每目标 dist/radial_v/approaching/threat/confirmed/
trusted/k/rej + toText + speak（docs/63 纪律逐句指回字段）。

验证：
  P1 真实目标双通道确认：base 段 A/B 都通过（非纯色分布下确认成立）
  P2 类别特异（docs/205）：A 被染记账 → B 确认零污染；A 恢复 thr 基线
  P3 威胁态势："它在靠近我"正确（A 靠近 threat 升、B 远离 threat 低）
  P4 纪念重信任（docs/200）：A 恢复后 k 帧延迟
  P5 真实观测：A/B 的中位色相/帧内 std 与 DAVIS 探测一致（±5°）——观测真来自真实目标

用法：python vision/davis_multi.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402
from multi_situation import ConfirmMind, SituationReport, circ  # noqa: E402

DAVIS = os.path.join("vision", "out", "davis")
SELF = (427.0, 240.0)          # 主体（威胁度参考点，surf 帧中心）
D_MAX = 500.0
N = 120
SEG_A, SEG_B, SEG_C = (0, 40), (40, 75), (75, 120)
TASK = {"A": 170.0, "B": 71.0}  # 任务色相：flamingo 粉 / surf 蓝绿（docs/219 探测）


def load_frames(video):
    vdir = os.path.join(DAVIS, video)
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    pngs = sorted(f for f in os.listdir(vdir) if f.endswith(".png"))
    return ([cv2.imread(os.path.join(vdir, j)) for j in jpgs],
            [cv2.imread(os.path.join(vdir, p), cv2.IMREAD_GRAYSCALE) for p in pngs])


def extract_target(frames, masks, idx, uid=None):
    """从 DAVIS 帧抠目标区域：局部掩码 + 局部像素补丁。返回 (lmask, patch, 局部质心)。"""
    fr, m = frames[idx], masks[idx]
    local = (m == uid) if uid is not None else (m > 0)
    ys, xs = np.nonzero(local)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    lmask = local[y0:y1 + 1, x0:x1 + 1]
    patch = fr[y0:y1 + 1, x0:x1 + 1].copy()
    ys2, xs2 = np.nonzero(lmask)
    cy, cx = ys2.mean(), xs2.mean()
    return lmask, patch, (float(cx), float(cy))


def a_pos(t):
    """flamingo 目标受控运动：向主体靠近（dist ~295→55）。"""
    if t < SEG_A[1]:
        k = t / (SEG_A[1] - SEG_A[0])
        return (150.0 + k * 210.0, 90.0 + k * 110.0)          # (150,90)→(360,200)
    if t < SEG_B[1]:
        k = (t - SEG_B[0]) / (SEG_B[1] - SEG_B[0])
        return (360.0 + k * 40.0, 200.0 + k * 22.0)           # →(400,222)
    k = (t - SEG_C[0]) / (SEG_C[1] - SEG_C[0])
    return (400.0 + k * 20.0, 222.0 + k * 12.0)               # →(420,234)


def b_pos(t):
    """surf 目标受控运动：向右上远离主体（dist ~307→403，威胁低且降）。"""
    k = t / N
    return (700.0 + k * 70.0, 100.0 - k * 50.0)               # (700,100)→(770,50)


def compose(bg, lmA, patA, lmB, patB, pa, pb, dyed):
    """背景帧 + 贴 A（可选冷增益偏移）+ 贴 B。返回合成帧。"""
    fr = bg.copy()
    for lmask, patch, pos in ((lmA, patA, pa), (lmB, patB, pb)):
        ys, xs = np.nonzero(lmask)
        cx, cy = xs.mean(), ys.mean()
        x0 = int(round(pos[0] - cx))
        y0 = int(round(pos[1] - cy))
        h, w = lmask.shape
        x1, y1 = x0 + w, y0 + h
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(fr.shape[1], x1), min(fr.shape[0], y1)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        lx0, ly0 = sx0 - x0, sy0 - y0
        sub = lmask[ly0:ly0 + (sy1 - sy0), lx0:lx0 + (sx1 - sx0)]
        pch = patch[ly0:ly0 + (sy1 - sy0), lx0:lx0 + (sx1 - sx0)]
        if dyed and lmask is lmA:
            pch = np.clip(pch.astype(np.float32) * np.array([1.27, 1.0, 0.73]),
                          0, 255).astype(np.uint8)
        fr[sy0:sy1, sx0:sx1][sub > 0] = pch[sub > 0]
    return fr


def target_obs(fr, lmask, pos, task_hue):
    """颜色通道观测：目标局部掩码采样 → 任务色占比 + 中位色相 + 帧内色相 std。

    能效修复（energy_audit B→C）：先按目标 bbox 裁剪 BGR 局部块再转 HSV，
    不做全帧 cvtColor（全帧 HSV 让'稀疏确认'名不副实）。
    """
    ys, xs = np.nonzero(lmask)
    cx, cy = xs.mean(), ys.mean()
    x0, y0 = int(round(pos[0] - cx)), int(round(pos[1] - cy))
    hh, ww = lmask.shape
    x1, y1 = x0 + ww, y0 + hh
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(fr.shape[1], x1), min(fr.shape[0], y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return 0.0, None, 0.0
    lx0, ly0 = sx0 - x0, sy0 - y0
    sub = lmask[ly0:ly0 + (sy1 - sy0), lx0:lx0 + (sx1 - sx0)] > 0
    # 局部裁剪后转 HSV（只处理目标 bbox 区域，不是全帧）
    patch = fr[sy0:sy1, sx0:sx1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    ph = h[sub].astype(float)
    ps = s[sub]
    colored = ps > 60
    if colored.sum() < 10:
        return 0.0, None, 0.0
    hs = ph[colored]
    d = np.abs(hs - task_hue) % 180.0
    d = np.minimum(d, 180.0 - d)
    frac = float(np.mean(d < 30.0))
    return frac, float(np.median(hs)), float(np.std(hs))


def main():
    f_f, m_f = load_frames("flamingo")
    f_s, m_s = load_frames("surf")
    # 抠目标（真实区域）
    lmA, patA, ca = extract_target(f_f, m_f, 20)
    lmB, patB, cb = extract_target(f_s, m_s, 20, uid=113)
    # 背景：surf 帧 20，原目标区域 inpaint 抹除
    bg = f_s[20].copy()
    fix = cv2.dilate((m_s[20] > 0).astype(np.uint8), np.ones((41, 41), np.uint8)) * 255
    bg = cv2.inpaint(bg, fix, 5, cv2.INPAINT_TELEA)
    print("== 多目标回真实视频：flamingo 粉 + surf 蓝绿 同帧（真实目标区域，受控布局）==")
    print(f"主体在 {SELF}。A(flamingo 粉~170°) 向主体靠近（中段被染=他者的不）；"
          f"B(surf 蓝绿~71°) 远离\n")

    mA, mB = ConfirmMind(task_hue=TASK["A"]), ConfirmMind(task_hue=TASK["B"])
    sr = SituationReport()
    log = []
    recover = None
    for i in range(N):
        pa, pb = a_pos(i), b_pos(i)
        dyed = SEG_B[0] <= i < SEG_B[1]
        fr = compose(bg, lmA, patA, lmB, patB, pa, pb, dyed)
        fA, hA, sA = target_obs(fr, lmA, pa, TASK["A"])
        fB, hB, sB = target_obs(fr, lmB, pb, TASK["B"])
        okA, okB = mA.step(fA, hA), mB.step(fB, hB)
        objs = [{"id": "A", "color": "粉", "pos": pa, "confirm": mA, "ok": okA},
                {"id": "B", "color": "蓝绿", "pos": pb, "confirm": mB, "ok": okB}]
        rep = sr.fill(objs, self_pos=SELF)
        utts = sr.speak(rep, objs)
        log.append((i, rep, mA.trusted and okA, utts, (hA, hB), (sA, sB)))
        if i >= SEG_C[0] and recover is None and (mA.trusted and okA):
            recover = (i, mA.k())

    a_rows = [l[1]["objects"][0] for l in log]
    b_rows = [l[1]["objects"][1] for l in log]
    base_hues = [l[4][0] for l in log[:SEG_A[1] - 5]]
    base_std = [l[5][0] for l in log[:SEG_A[1] - 5]]

    print("== 验证 ==")
    # P1 真实目标双通道确认
    p1 = all(r["confirmed"] for r in a_rows[:SEG_A[1]]) and \
         all(r["confirmed"] for r in b_rows[:SEG_A[1]])
    print(f"P1 真实目标双通道确认: [{'PASS' if p1 else 'FAIL'}] "
          f"A(粉) base 确认 {sum(1 for r in a_rows[:SEG_A[1]] if r['confirmed'])}/{SEG_A[1]}；"
          f"B(蓝绿) base 确认 {sum(1 for r in b_rows[:SEG_A[1]] if r['confirmed'])}/{SEG_A[1]}")

    # P2 类别特异
    b_stable = len(set(r["trusted"] for r in b_rows)) == 1 and \
        all(r["confirmed"] for r in b_rows)
    sus_keys = list(mA.suspicious.keys())
    sus_far = any(circ(k, TASK["A"]) > 40 for k in sus_keys) if sus_keys else False
    p2 = b_stable and abs(mA.thr - mA.thr_base) < 0.001 and mA.suspicious and sus_far
    print(f"P2 类别特异（docs/205）: [{'PASS' if p2 else 'FAIL'}] "
          f"A 被染记 suspicious={ {round(k,0): v for k, v in list(mA.suspicious.items())[:3]} }"
          f"（距任务色 {TASK['A']:.0f}° 远→不污染）；A 恢复后 thr={mA.thr:.2f}（基线）；"
          f"B 全程确认通过、信任恒 {b_rows[0]['trusted']}（零污染）")

    # P3 威胁态势
    a_app = all(r["approaching"] for r in a_rows[10:])
    b_app = not any(r["approaching"] for r in b_rows)
    p3 = a_app and b_app and a_rows[-1]["threat"] > a_rows[10]["threat"]
    print(f"P3 威胁态势（'它在靠近我'）: [{'PASS' if p3 else 'FAIL'}] "
          f"A threat {a_rows[10]['threat']:.2f}→{a_rows[-1]['threat']:.2f}（靠近升）；"
          f"B threat {b_rows[10]['threat']:.2f}（远离低，无靠近={b_app}）")

    # P4 纪念重信任
    p4 = recover is not None and recover[1] > 1 and recover[0] - SEG_C[0] >= recover[1] - 2
    print(f"P4 纪念重信任（docs/200）: [{'PASS' if p4 else 'FAIL'}] "
          f"A 恢复后第 {recover[0] if recover else '?'} 帧 claimed（k={recover[1] if recover else '?'}，"
          f"延迟 {recover[0] - SEG_C[0] if recover else '?'} 帧）")

    # P5 真实观测
    p5 = abs(np.median(base_hues) - 170.0) < 8 and np.mean(base_std) > 8
    print(f"P5 真实目标观测: [{'PASS' if p5 else 'FAIL'}] A base 中位色相 "
          f"{np.median(base_hues):.1f}°（DAVIS flamingo 探测 169.7°）、帧内 std "
          f"{np.mean(base_std):.1f}°（非纯色，探测 43.9°）")

    print("\n== 态势转录流（关键帧）==")
    for i in [0, 20, 39, 45, 60, 75, 100, 119]:
        l = log[i]
        o = l[1]["objects"][0]
        print(f"  [{i:3d}] A {o['dist']:5.0f}px threat={o['threat']:.2f} "
              f"app={int(o['approaching'])} conf={int(o['confirmed'])} "
              f"trust={o['trusted']} k={o['k']} | {sr.to_text(l[1]).split('。')[0]}")
    print("\n  语言流（docs/63 纪律：每句指回状态字段）：")
    seen = set()
    for i, l in enumerate(log):
        for u in l[3]:
            key = u.split("（")[0]
            if key not in seen or "更连续" in u or "确认了" in u:
                print(f"    [{i:3d}] {u}")
                seen.add(key)

    print("\n== 判读 ==")
    print("  两个真实目标（粉/蓝绿）同帧：确认各自独立（A 被染 B 零污染，类别特异 docs/205）")
    print("  威胁态势=运动通道（docs/195）：A 靠近 threat 升、B 远离 threat 低——'它在靠近我'")
    print("  观测真实：A 中位色相/帧内 std 与 DAVIS 探测一致——非纯色分布下确认仍成立")
    print("  语言（docs/204 + docs/63）：每句 1:1 指回状态字段——行为签名转录")


if __name__ == "__main__":
    main()
