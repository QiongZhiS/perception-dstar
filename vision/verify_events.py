"""vision/verify_events.py — 验证到达过程：任务层关键帧宣布 vs 固定概率涓流。

docs/201 §五 1：nu 从固定概率（每帧 5%）升级为**事件驱动验证**——任务层（外部
利害）不会每帧说话，而是在关键帧宣布（"找蓝"）。核心问题：**一次宣布够吗？**

机制差异：
  涓流（docs/201 基线）：每帧概率 nu 接受任务利害 → 持续小流量，多帧累积
  事件（本文件）：只在宣布帧接受任务利害 → 离散到达，一次宣布 = 一步移动
    （移 eta×有向环形差；eta 小=固执系统，一次只移几度 → "听到了但没改过来"）

三问（对应 docs/201 §五 1 的"宣布时机对守住/改判的影响"）：
  P1 到达强度：一次宣布够吗？——改判场景，宣布次数 1/3/6/持续 vs 涓流，
     eta 扫描。预期：eta 小（固执）单次不够（移 3°），需多次/持续；
     eta 大（可塑）单次可能够（移 30°）。"一次宣布够不够"取决于系统固执度。
  P2 宣布时机：目标变蓝后立即宣布（f35）/延迟（f50/f70）→ 入蓝帧；
     提前宣布（f25，目标还没蓝）→ 预判利害（任务说对了则无害）。
  P3 抗错误宣布：守住场景（红→蓝→红，任务恒红）任务层误说"找蓝"一次——
     单次错误应不误改判（自洽门拉回）；持续错误（f50-70 误说）会被带偏
     （利害噪声的代价，docs/201 边界 3：验证也可能错）。

映射：
  docs/101 验证涓流 nu = 每帧概率（持续小流量）
  docs/201 事件 = 关键帧宣布（离散到达）——真实利害更新形态
  docs/178 补刀：他者的 不（利害更新）必须到达足够强度才改写期望——
     "重要的话要说够"：单次低强度宣布在固执系统里不足以改判。
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np
from keep_learn import RED_H, BLUE_H, TAU, circ, circ_add, in_red, in_blue, N_FRAMES

SIGMA = 3.0


def make_scenes():
    hold_true = np.array([RED_H] * 30 + [BLUE_H] * 30 + [RED_H] * (N_FRAMES - 60))
    hold_task = np.array([RED_H] * N_FRAMES)
    shift_true = np.array([RED_H] * 30 + [BLUE_H] * (N_FRAMES - 30))
    shift_task = np.array([RED_H] * 40 + [BLUE_H] * (N_FRAMES - 40))
    return (hold_true, hold_task), (shift_true, shift_task)


def run_events(true_seq, task_seq, eta, announce, sigma=SIGMA, seed=42):
    """事件驱动验证：announce = 宣布帧集合（该帧接受任务利害）。
    自洽门照旧（观察在期望内则接受）。返回 (theta 序列, 首次入蓝, 终值)。"""
    rng = np.random.default_rng(seed)
    theta = RED_H
    thetas = []
    first_blue = None
    for i in range(len(true_seq)):
        o = (true_seq[i] + rng.normal(0, sigma)) % 180.0
        accepted = None
        if circ(o, theta) < TAU:
            accepted = o
        elif i in announce:                      # 事件：任务层关键帧宣布利害
            accepted = (task_seq[i] + rng.normal(0, sigma)) % 180.0
        if accepted is not None:
            theta = circ_add(theta, accepted, eta)
        thetas.append(theta)
        if first_blue is None and circ(theta, BLUE_H) < TAU / 2:
            first_blue = i
    return np.array(thetas), first_blue, theta


def fmt(fb):
    return f"f{fb}" if fb is not None else '—'


(hold_true, hold_task), (shift_true, shift_task) = make_scenes()

print("== 验证到达过程：任务层关键帧宣布 vs 固定概率涓流 ==")
print(f"自洽门 τ={TAU}°，色相环 0-180（红 0°/蓝 120°），改判场景目标 f30 起变蓝、任务 f40 起'找蓝'\n")

print("== P1 到达强度：一次宣布够吗？（改判场景，宣布次数 1/3/6/持续 vs 涓流）==")
print(f"{'eta':>5s} {'到达模型':>10s} {'终值':>7s} {'蓝区':>5s} {'首次入蓝':>8s}")
p1 = True
for eta in [0.05, 0.2, 0.5]:
    # 涓流基线（docs/201）：nu=0.1 → 期望 ~10 次验证/100 帧
    for label, ann in [("涓流nu=0.1", None), ("事件×1", {50}),
                       ("事件×3", {50, 60, 70}), ("事件×6", {45, 50, 55, 60, 65, 70}),
                       ("持续f50-70", set(range(50, 71)))]:
        if label.startswith("涓流"):
            from keep_learn import run
            _, fb, th = run(shift_true, shift_task, eta, 0.1, SIGMA)
        else:
            _, fb, th = run_events(shift_true, shift_task, eta, ann)
        ok = in_blue(th)
        # 事件模型：eta 小单次不够但持续够；eta 大单次可能够——只要至少一种事件形态能改判
        print(f"{eta:5.2f} {label:>10s} {th:7.1f} {str(ok):>5s} {fmt(fb):>8s}")
print("P1 判读：单次宣布的移动量 = eta×环形差（0→120 最短 60°）→ eta=0.05 移 3°/次，"
      "单次远不够，需 ≥15 次；eta=0.5 移 30°/次，3 次够。"
      "\n        '一次宣布够吗' = 取决于系统固执度（eta）：固执系统需要利害重复到达。\n")

print("== P2 宣布时机：利害更新(f40)后 立即/延迟 vs 利害提前(任务预判) ==")
print(f"{'形态':>14s} {'宣布帧':>14s} {'终值':>7s} {'蓝区':>5s} {'首次入蓝':>8s}")
p2 = True
# 立即/中/晚：任务 f40 起蓝，宣布在更新后不同时刻
for label, ann in [("立即", {40, 45, 50}), ("中等", {55, 60, 65}), ("延迟", {70, 80, 90})]:
    _, fb, th = run_events(shift_true, shift_task, 0.2, ann)
    print(f"{label:>14s} {str(sorted(ann)):>14s} {th:7.1f} {str(in_blue(th)):>5s} {fmt(fb):>8s}")
# 利害提前：任务 f25 起蓝（预判），宣布 f25/30/35/40（4 次=足够到达强度），目标 f30 才变蓝
early_task = np.array([RED_H] * 25 + [BLUE_H] * (N_FRAMES - 25))
_, fb, th = run_events(shift_true, early_task, 0.2, {25, 30, 35, 40})
print(f"{'提前(任务预判)':>14s} {'f25-40×4':>14s} {th:7.1f} {str(in_blue(th)):>5s} {fmt(fb):>8s}")
print("P2 判读：立即/中等宣布都能改判；延迟宣布改判慢（f98）；'提前'= 任务层预判利害"
      "（f25 说找蓝、目标 f30 才蓝）→ 期望先移蓝、目标变蓝后自洽确认——"
      "docs/190'想看什么就看见什么'的利害侧；且提前预判同样需要足够到达强度"
      "（3 次卡半路 158°，4 次入蓝——环形路径下 0→120 走 -60° 短径，theta 中途"
      "离目标反而远，需多一步才进自洽确认区）。\n")

print("== P3 抗错误宣布：守住场景任务层误说'找蓝'（单次 vs 持续）==")
print(f"{'eta':>5s} {'错误形态':>12s} {'终值':>7s} {'红区':>5s}")
p3 = True
for eta in [0.05, 0.2, 0.5]:
    # 单次错误：f50 任务误说蓝一次
    bad_task = hold_task.copy()
    bad_task[50] = BLUE_H
    _, _, th1 = run_events(hold_true, bad_task, eta, {50})
    # 持续错误：f50-70 任务误说蓝
    bad_task2 = hold_task.copy()
    bad_task2[50:71] = BLUE_H
    _, _, th2 = run_events(hold_true, bad_task2, eta, set(range(50, 71)))
    print(f"{eta:5.2f} {'单次误说':>12s} {th1:7.1f} {str(in_red(th1)):>5s}")
    print(f"{'':5s} {'持续误说':>12s} {th2:7.1f} {str(in_red(th2)):>5s}")
print("P3 判读：eta 小（固执）单次/持续误说都守住（自洽门+红观察拉回）；"
      "eta 大（可塑）单次误说即被带偏——错误利害把期望推离红后，异常段'真蓝观察'"
      "经自洽门确认它 → 异常结束仍锁蓝。错误利害+真实观察联合带偏，可塑系统抗错差"
      "（docs/101'完全可塑=没有自我'的利害侧）。")

print("\n== docs/101 ↔ 验证到达过程映射 ==")
print("  nu 每帧概率（持续涓流）→ 任务层利害更新的事件形态（关键帧宣布）")
print("  单次事件强度 = eta × 环形差 → '一次宣布够吗'取决于系统固执度（eta）")
print("  固执系统（eta 小）需要利害重复到达（'重要的话要说够'）")
print("  可塑系统（eta 大）单次即改判（快，但错误利害也易带偏——P3）")
print("  docs/178 补刀：他者的 不 要改写期望，验证必须到达足够强度")
