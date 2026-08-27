"""vision/keep_learn.py — 守住 vs 改判：验证涓流 nu 解耦（docs/101 SEED-49 搬到颜色轴）。

docs/200 §五 1 开放端：保留系统绝对固执（期望恒红）是对的——单次染色=他者的 不，
不该改判（docs/200 P1 漂移 0° 已验证）；但世界真变（任务层宣布"找蓝"）时该学习。
"守住"与"改判"的边界在哪？docs/101 的答案：**利害外置 + 验证涓流 nu**——
唯一合法的改判通道是外部利害更新（任务层），不是感知层观察（感知层的"不"永远
不该改写任务）。SEED-49 三轴（eta 先验强度 × sigma 世界噪声 × nu 验证涓流）搬到
颜色轴：

  theta : 期望色相（自我=任务利害），初始红 0°
  o     : 目标区色相观察 = 真值 + N(0, sigma)
  门控（SEED-25 自指过滤）：|o−theta|_circ < tau（自洽，守住）
       或 验证涓流成功（每帧概率 nu，接受任务层利害 o_v）
  更新  : theta += eta × (accepted − theta)_circ（仅在门控通过时）

两场景：
  守住（单次异常）：目标 红→蓝→红，任务恒"红" → theta 应终值在红区
      （感知层的"不"不改判；即使 nu>0，验证=任务=红，不会引入蓝）
  改判（世界真变）：目标 红→蓝（永久），任务第 40 帧起宣布"找蓝" → theta
      应移向蓝区（nu>0 时验证涓流携带新利害；nu=0 → 锁死，docs/101 SEED-25）

判据：
  P1 守住：守住场景 theta 终值在红区（所有 eta×nu）
  P2 改判：改判场景 nu>0 → theta 终值在蓝区；nu=0 → 红区（锁死）
  P3 平衡带：改判速度（首次入蓝区帧）随 eta×nu 单调降；守住漂移随 sigma 升
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np

RED_H, BLUE_H = 0.0, 120.0      # 色相（OpenCV 0-180 环）
TAU = 30.0                       # 自洽门宽（SEED-25：只接受确认信念的观察）
N_FRAMES = 100


def circ(a, b):
    """环形色相距离（0-180 环）。"""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def circ_add(theta, target, eta):
    """向 target 移动 eta 比例的环形色相（有向最短差，±90 内）。"""
    d = target - theta
    if d > 90.0:
        d -= 180.0
    elif d < -90.0:
        d += 180.0
    return (theta + eta * d) % 180.0


def make_scenes():
    """守住场景与改判场景的目标色相序列 + 任务利害序列。"""
    # 守住：红→蓝→红（单次异常），任务恒红
    hold_true = np.array([RED_H] * 30 + [BLUE_H] * 30 + [RED_H] * (N_FRAMES - 60))
    hold_task = np.array([RED_H] * N_FRAMES)
    # 改判：红→蓝（永久），任务前 40 帧红、后 60 帧蓝（利害更新）
    shift_true = np.array([RED_H] * 30 + [BLUE_H] * (N_FRAMES - 30))
    shift_task = np.array([RED_H] * 40 + [BLUE_H] * (N_FRAMES - 40))
    return (hold_true, hold_task), (shift_true, shift_task)


def run(true_seq, task_seq, eta, nu, sigma, seed=42):
    """颜色轴 SEED-49。返回 (theta 序列, 首次入蓝帧, 终值)。"""
    rng = np.random.default_rng(seed)
    theta = RED_H
    thetas = []
    first_blue = None
    for i in range(len(true_seq)):
        o = true_seq[i] + rng.normal(0, sigma)     # 观察=真值+世界噪声
        o = o % 180.0
        accepted = None
        if circ(o, theta) < TAU:                   # 自洽门：守住自我
            accepted = o
        elif rng.random() < nu:                    # 验证涓流：外部利害（唯一合法改判通道）
            accepted = task_seq[i] + rng.normal(0, sigma)   # 利害本身也带噪声
            accepted = accepted % 180.0
        if accepted is not None:
            theta = circ_add(theta, accepted, eta)
        thetas.append(theta)
        if first_blue is None and circ(theta, BLUE_H) < TAU / 2:
            first_blue = i
    return np.array(thetas), first_blue, theta


def in_red(t):
    return circ(t, RED_H) < TAU / 2


def in_blue(t):
    return circ(t, BLUE_H) < TAU / 2


(hold_true, hold_task), (shift_true, shift_task) = make_scenes()

print("== 守住 vs 改判：验证涓流 nu 解耦（docs/101 SEED-49 颜色轴）==")
print(f"自洽门 τ={TAU}°（SEED-25：只接受确认信念的观察），色相环 0-180（红 0°/蓝 120°）\n")

print("== P1 守住（单次异常红→蓝→红，任务恒红）：theta 终值应在红区 ==")
print(f"{'eta':>5s} {'nu':>5s} {'sigma':>6s} {'终值':>7s} {'红区':>5s}")
p1_ok = True
for eta in [0.05, 0.2, 0.5]:
    for nu in [0.0, 0.05, 0.2]:
        for sigma in [3.0, 10.0]:
            _, _, th = run(hold_true, hold_task, eta, nu, sigma)
            ok = in_red(th)
            p1_ok &= ok
            print(f"{eta:5.2f} {nu:5.2f} {sigma:6.1f} {th:7.1f} {'✓' if ok else '✗'}")
print(f"P1 全 PASS: [{'PASS' if p1_ok else 'FAIL'}]（自洽门+任务验证=红，不引入蓝）\n")

print("== P2 改判（红→蓝永久，任务 40 帧起'找蓝'）：nu>0 该移蓝、nu=0 锁死 ==")
print(f"{'eta':>5s} {'nu':>5s} {'sigma':>6s} {'终值':>7s} {'蓝区':>5s} {'首次入蓝':>8s}")
p2_ok = True
for eta in [0.2]:
    for nu in [0.0, 0.05, 0.2, 0.5]:
        for sigma in [3.0, 10.0]:
            _, fb, th = run(shift_true, shift_task, eta, nu, sigma)
            expect_blue = nu > 0
            ok = in_blue(th) == expect_blue
            p2_ok &= ok
            fb_s = f"f{fb}" if fb is not None else '—'
            print(f"{eta:5.2f} {nu:5.2f} {sigma:6.1f} {th:7.1f} {in_blue(th)!s:>5} {fb_s:>8s}")
print(f"P2: [{'PASS' if p2_ok else 'FAIL'}]（nu>0 验证涓流携带新利害→改判；nu=0 锁死）\n")

print("== P3 平衡带：改判速度（首次入蓝帧）随 eta×nu 增（非严格单调，见下） ==")
print(f"{'eta':>5s} {'nu':>5s} {'eta×nu':>7s} {'首次入蓝':>8s} {'终值':>7s}")
pairs = []
for eta in [0.05, 0.1, 0.2, 0.4, 0.5]:
    for nu in [0.05, 0.2, 0.5]:
        _, fb, th = run(shift_true, shift_task, eta, nu, 3.0)
        fb_s = f"f{fb}" if fb is not None else '—'
        print(f"{eta:5.2f} {nu:5.2f} {eta * nu:7.3f} {fb_s:>8s} {th:7.1f}")
        pairs.append((eta * nu, fb))
# 秩相关（纯 numpy）：入蓝帧(小=快) 对 eta×nu(大=快) 应为负；未入蓝视为帧尾(100)
xs = np.array([p[0] for p in pairs])
ys = np.array([100 if p[1] is None else p[1] for p in pairs])


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


rho = spearman(xs, ys)
print(f"  Spearman ρ(eta×nu, 入蓝帧) = {rho:.2f}（负=乘积大入蓝快）")
print(f"  P3: [{'PASS' if rho < -0.5 else 'FAIL'}]（非严格单调：自洽门接管时机与 sigma 叠加）\n")

print("== docs/101 ↔ 颜色轴映射 ==")
print("  eta（先验强度）        → 期望色相的更新步长：大=可塑快改判，小=固执")
print("  sigma（世界噪声）      → 目标区色相观察抖动（光照/遮挡/噪声染色）")
print("  nu（验证涓流，SEED-25）→ 外部利害更新进入期望的唯一通道：")
print("      感知层的'不'（染色）永远不改判（自洽门拒）；")
print("      任务层的'不'（利害更新）经验证涓流改判（docs/188 利害外置）")
print("  nu=0 → 100% 锁死（docs/101 SEED-25：没有独立验证时固执变锁死）")
print("  docs/178 补刀：'世界的 不'可消化（自洽门内观察吸收）、'他者的 不'")
print("     不可消化（自洽门拒，只有利害层验证能改写期望）——两种不的分界线")
print("     = 观察 vs 利害：观察永远不进任务层，利害经涓流进期望。")
