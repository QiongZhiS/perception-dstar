"""vision/situation_test.py — 态势接口验证（docs/160 视觉版，DAVIS car-turn）。

两臂：
  A 外赋目标：第一帧从掩码指定"车"为维持目标（docs/174 外赋利害锚）
  B 自发目标：不指定，轨迹存活 ≥confirm_age 帧自动升级（第 4 层试探）
度量：目标维持（hit 率）、每帧保留框数、态势文本化样例。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2, json
from davis_check import mask_boxes
from situation import SituationReport

vdir = os.path.join('vision', 'out', 'davis', 'car-turn')
jpgs = sorted(f for f in os.listdir(vdir) if f.endswith('.jpg'))


def run_arm(external, confirm_age=5):
    rep = SituationReport(K=3, confirm_age=confirm_age)
    hits = frames = 0
    logs = []
    for jpg in jpgs:
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace('.jpg', '.png')), 0)
        if fr is None or mask is None:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ext = None
        if external and frames == 0:
            gt = mask_boxes(mask)
            if gt:
                _, x, y, w, h = gt[0]
                ext = [(x, y, w, h)]
        r = rep.fill(gray, external_targets=ext)
        frames += 1
        # 目标维持：真值框与任一 object 近
        gt = mask_boxes(mask)
        if gt and rep.targets:
            _, gx, gy, gw, gh = gt[0]
            gcx, gcy = gx + gw / 2, gy + gh / 2
            hit = any(((o["cx"] - gcx) ** 2 + (o["cy"] - gcy) ** 2) ** 0.5 < 40
                      for o in r["objects"])
            if hit:
                hits += 1
        if frames in (1, 10, 40, 79):
            logs.append(rep.to_text(r))
    rate = hits / max(frames, 1)
    return rate, rep.metrics, logs


print("== 态势接口验证（DAVIS car-turn，Top-K=3）==")
print("A 臂（外赋目标）维持率应≈1.0（车被态势稳定维持，docs/174 外赋利害锚工程化）。")
print("B 臂（自发目标）注意：'持续性升级'会把背景噪声块也收编成目标（先前实测")
print("98.2% 自发目标不在真值附近）——目标数爆炸时维持率失真，见 docs/186 §五。")
for name, external in [("A 外赋目标", True), ("B 自发目标", False)]:
    rate, metrics, logs = run_arm(external)
    print(f"\n{name}: 目标维持率 {rate:.3f}")
    print(f"  每帧框 {metrics['n_boxes'] / 80:.1f} → 保留 {metrics['n_kept'] / 80:.1f}，"
          f"目标命中帧 {metrics['target_hit']}，目标数 {metrics['n_targets']}")
    print("  态势样例:")
    for lg in logs[:3]:
        print(f"    {lg}")

# 存 A 臂完整态势
rep_a = SituationReport(K=3)
hits = 0
situations = []
for i, jpg in enumerate(jpgs):
    fr = cv2.imread(os.path.join(vdir, jpg))
    mask = cv2.imread(os.path.join(vdir, jpg.replace('.jpg', '.png')), 0)
    if fr is None or mask is None:
        continue
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ext = None
    if i == 0:
        gt = mask_boxes(mask)
        if gt:
            _, x, y, w, h = gt[0]
            ext = [(x, y, w, h)]
    r = rep_a.fill(gray, external_targets=ext)
    situations.append(r)
out = os.path.join("vision", "out", "situation_car_turn.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"video": "car-turn", "n": len(situations),
               "situations": situations[:120]}, f, ensure_ascii=False, indent=1)
print(f"\n→ {out}（A 臂态势日志，可回放 docs/105）")
