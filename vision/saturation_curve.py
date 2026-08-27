"""vision/saturation_curve.py — 未饱和连续曲线：扫被拒次数看完整衰减。

docs/212 §五 1：被拒累计 3000 经 cap=60 饱和后带宽内 k 顶满 21，连续性的可观测
窗口收窄到带宽边缘。本实验扫被拒累计（训练遍数 1/5/20/100 → 被拒 30/150/600/3000），
看连续加权的完整衰减曲线——"怀疑饱和前"的连续行为。

问题：连续加权（docs/212）的连续曲线在饱和前是否完整？（0° 最高、边缘渐低、
带宽外 1）饱和后是否退化为二值？（带宽内全顶满）

实验：
  P1 未饱和连续：小被拒（30/150 次）时 k(距蓝距离) 连续下降——完整衰减曲线
  P2 饱和退化：大被拒（600/3000 次）时带宽内全顶满（21）——连续退化为二值
  P3 曲线形状：线性 vs 高斯在不同被拒量下的差异（线性有尾巴、高斯快衰减）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np
from bandwidth_continuous import ContinuousSus, train
from bandwidth_threshold import JointSus, circ, CAP, GROW

RED_H, BLUE_H = 0.0, 120.0

print("== 未饱和连续曲线：扫被拒次数看完整衰减 ==")
print(f"每遍染蓝被拒 30 次；cap=60（饱和点 k=21）\n")

print("== P1/P2 k(距蓝距离) 曲线 × 被拒累计 ==")
print(f"{'被拒累计':>10s} {'0°':>5s} {'2°':>5s} {'4°':>5s} {'6°':>5s} {'7°':>5s} "
      f"{'8°':>5s} {'10°':>5s} {'15°':>5s} {'120°':>6s} {'形态':>10s}")
for n_pass in [1, 5, 20, 100]:
    c = ContinuousSus(decay='linear')
    train(c, n_pass=n_pass)
    n_rej = n_pass * 30
    row = [c.k((BLUE_H + d) % 180.0) for d in [0, 2, 4, 6, 7, 8, 10, 15, 120]]
    shape = ('连续(未饱和)' if n_rej < CAP else
             '饱和退化(带宽内顶满)' if n_rej > 1000 else '部分饱和')
    print(f"{n_rej:10d} {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d} {row[4]:5d} "
          f"{row[5]:5d} {row[6]:5d} {row[7]:5d} {row[8]:6d} {shape:>10s}")

print("\n== P3 线性 vs 高斯 × 被拒累计 ==")
print(f"{'被拒累计':>10s} {'线性7°':>8s} {'线性8°':>8s} {'高斯7°':>8s} {'高斯8°':>8s}")
for n_pass in [2, 10, 50]:
    cl = ContinuousSus(decay='linear')
    train(cl, n_pass=n_pass)
    cg = ContinuousSus(decay='gauss')
    train(cg, n_pass=n_pass)
    n_rej = n_pass * 30
    print(f"{n_rej:10d} {cl.k((BLUE_H+7)%180):8d} {cl.k((BLUE_H+8)%180):8d} "
          f"{cg.k((BLUE_H+7)%180):8d} {cg.k((BLUE_H+8)%180):8d}")

print("\n== 判读 ==")
print("  未饱和（30/150 次被拒）：k(距离) 连续下降（0°最高、边缘渐低、带宽外 1）")
print("  饱和（600+ 次）：带宽内 k 顶满 21（cap=60），连续退化为二值——")
print("     docs/203 saturate 与连续性的交互：饱和掩盖连续差异")
print("  线性 vs 高斯：线性在带宽边缘有尾巴（7° 仍加权）、高斯快衰减")
print("  '连续性的可观测窗口' = 被拒累计 < cap（未饱和区）")
