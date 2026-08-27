"""vision/task_lock_test.py — 第 1 步验证：目标锚到任务层（docs/190 目标驱动识别）。

任务层说"找车" + 弱位置约束（hint，故意给偏），视觉用 lock_target 从候选框锁定，
之后完全自主维持（威胁度+轨迹）。对比三臂：
  A 精确外赋：每帧给精确框（旧方式，docs/189）
  B 任务锁定：第一帧给 hint（偏差 60-100px），lock_target 锁定后自主维持（新方式）
  C 无约束：不给 hint（应无法锁定——docs/188：无利害无法自发选目标）

判据：B 维持率 ≈ A（任务层弱约束能替代精确坐标），C 明显低（无约束确实不行）。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from davis_check import mask_boxes
from situation import SituationReport

vdir = os.path.join('vision', 'out', 'davis', 'car-turn')
jpgs = sorted(f for f in os.listdir(vdir) if f.endswith('.jpg'))

def arm_A():
    """精确外赋：第一帧给精确框，之后无外部（situation 自动维持）。禁自发升级。"""
    rep = SituationReport(K=3, confirm_age=10 ** 6)
    hits = frames = 0
    for i, jpg in enumerate(jpgs):
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace('.jpg', '.png')), 0)
        if fr is None or mask is None: continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ext = None
        if i == 0:
            gt = mask_boxes(mask)
            if gt:
                _, x, y, w, h = gt[0]
                ext = [(x, y, w, h)]
        s = rep.fill(gray, external_targets=ext)
        frames += 1
        gt = mask_boxes(mask)
        if gt and rep.targets:
            _, gx, gy, gw, gh = gt[0]
            hit = any(((o["cx"] - (gx+gw/2))**2 + (o["cy"] - (gy+gh/2))**2)**0.5 < 40
                      for o in s["objects"])
            hits += hit
    return hits / max(frames, 1)

def arm_B(noise=80):
    """任务锁定：前几帧持续尝试给 hint（真值+噪声），lock_target 锁定后自主维持。"""
    rep = SituationReport(K=3, confirm_age=10 ** 6)
    hits = frames = locked = 0
    for i, jpg in enumerate(jpgs):
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace('.jpg', '.png')), 0)
        if fr is None or mask is None: continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = None
        if not rep.targets and i < 30:     # 前 30 帧持续尝试（车事件累积后再锁）
            gt = mask_boxes(mask)
            if gt:
                _, gx, gy, gw, gh = gt[0]
                hint = (gx + gw/2 + noise, gy + gh/2 + noise)
                if rep.lock_target(gray, hint):
                    locked += 1
                    s = {"tick": rep.tick, "objects": [{
                        "id": rep.targets[0]["id"], "cx": rep.targets[0]["cx"],
                        "cy": rep.targets[0]["cy"], "x": 0, "y": 0,
                        "w": rep.targets[0]["w"], "h": rep.targets[0]["h"],
                        "vx": 0.0, "vy": 0.0, "age": 0, "external": False}],
                        "threats": [], "saccade": {"fov": "full", "k": 3},
                        "budget": -1, "metrics": {}}
                    frames += 1
        if s is None:
            s = rep.fill(gray, external_targets=None)
            frames += 1
        gt = mask_boxes(mask)
        if gt and rep.targets:
            _, gx, gy, gw, gh = gt[0]
            hit = any(((o["cx"] - (gx+gw/2))**2 + (o["cy"] - (gy+gh/2))**2)**0.5 < 40
                      for o in s["objects"])
            hits += hit
    return hits / max(frames, 1), locked

def arm_C():
    """无约束：不给 hint，且关闭自发升级（纯无利害——docs/188 应无法锁定）。"""
    rep = SituationReport(K=3, confirm_age=10 ** 6)   # 禁自发升级
    hits = frames = 0
    for i, jpg in enumerate(jpgs):
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace('.jpg', '.png')), 0)
        if fr is None or mask is None: continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        s = rep.fill(gray, external_targets=None)
        frames += 1
        gt = mask_boxes(mask)
        if gt and rep.targets:
            _, gx, gy, gw, gh = gt[0]
            hit = any(((o["cx"] - (gx+gw/2))**2 + (o["cy"] - (gy+gh/2))**2)**0.5 < 40
                      for o in s["objects"])
            hits += hit
    return hits / max(frames, 1)

print("== 目标锚到任务层（car-turn，Top-K=3）==")
print("A 精确外赋（每帧给框，旧方式）:")
ra = arm_A()
print(f"  维持率 {ra:.3f}")
print("B 任务锁定（hint 偏差 80px，lock 后自主）:")
rb, locked = arm_B(80)
print(f"  锁定 {'成功' if locked else '失败'}，维持率 {rb:.3f}")
print("C 无约束（不给 hint）:")
rc = arm_C()
print(f"  维持率 {rc:.3f}")
print()
print(f"判据：B({rb:.3f}) ≈ A({ra:.3f}) 且 C({rc:.3f}) 明显低 → 任务层弱约束能替代精确坐标")
print(f"  B≈A: {'PASS' if abs(rb - ra) < 0.1 else 'FAIL'}   C低: {'PASS' if rc < ra - 0.3 else 'FAIL'}")
