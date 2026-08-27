"""vision/timescale_sweep.py — S6 多时间尺度定量：扫 α，找单通道"无解区间"。

主张（笔记 L1 核心问题 + docs/177 C3）：单时间尺度无法同时服务快（快速运动/阶跃恢复）
与慢（慢速漂移）两种现象；双通道（快 0.5 + 慢 0.03）同时满足。

受控场景（1×64 行信号，喂给 Transduction2D）：
  [0,100)   稳态（噪声底）
  [100,200) 快速运动（亮条移动 → 边缘事件）
  [200,300) 稳态
  [300,500) 慢速漂移（整行亮度 +30% 渐变 200 帧 → 慢现象）
  [500,560) 阶跃（亮度跳变 5 帧 → 恢复时间）
  [560,700) 稳态

度量（每个 α 与双通道）：
  fast_cost    快运动时段事件（- 噪声底）
  slow_detect  慢漂移时段事件（- 噪声底；低 α 才有明显响应）
  settle       阶跃后事件回落 <2×稳态底的帧数（快恢复需要高 α）
  noise_floor  稳态事件/帧

预言：不存在单一 α 同时满足 settle ≤ 5 帧 ∧ slow_detect ≥ 慢通道的 50%——
即"快恢复"与"慢探测"是 α 的两端，单通道无解；双通道 (0.5, 0.03) 同时满足。

用法：python vision/timescale_sweep.py [--out vision/out/timescale.json]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
from transduction import Transduction2D  # noqa: E402


def make_scene(fps=30, width=256):
    """1×256 行信号：快运动（20px/帧跳变）+ 慢漂移 + 持续阶跃（100%，残留衰减 > 死区）。"""
    n = 700
    frames = []
    rng = np.random.default_rng(42)
    bar_x = 10
    light = 100.0
    for i in range(n):
        row = np.full((1, width), light, np.float32)
        # 快速运动：10px 亮条以 20px/帧 往复跳变（[100,200)）——快现象
        if 100 <= i < 200:
            bar_x = 10 + ((i - 100) * 20) % (width - 20)
            row[0, bar_x:bar_x + 10] = 200
        # 慢速漂移：整行 +30%（[300,500)，200 帧渐变）——慢现象，快通道预测掉
        if 300 <= i < 500:
            light = 100 + 30 * ((i - 300) / 200)
        # 持续阶跃：130→260（相对 +100%，log 0.69，慢通道残留衰减 0.021 > 死区 0.015）
        if 500 <= i < 590:
            light = 260
        elif i >= 590:
            light = 130
        row += rng.normal(0, 1.0, row.shape)
        frames.append(np.clip(row, 0, 255).astype(np.uint8))
    return frames


def run_channel(frames, alpha_fast, alpha_slow, separate=False):
    """separate=True 时返回 (快通道事件, 慢通道事件) —— 双通道按组件分测。"""
    td = Transduction2D(alpha_fast=alpha_fast, alpha_slow=alpha_slow, thresh=0.15)
    evf, evs = [], []
    for fr in frames:
        a, b, _, _ = td.step(fr)
        evf.append(int((a != 0).sum()))
        evs.append(int((b != 0).sum()))
    if separate:
        return np.array(evf), np.array(evs)
    return np.array(evf) + np.array(evs)


def metrics(ev, n=700, win=(500, 590)):
    base = float(ev[0:100].mean())                  # 无运动稳态（死区压噪声 → 约 0）
    fast_cost = float(ev[100:200].sum() - base * 100)
    slow_detect = float(ev[300:500].sum() - base * 200)
    # settle：阶跃后"最后一个活跃帧"（ev > 2）减去阶跃帧——稀疏周期爆发用
    # "活跃持续到何时"量；窗口只到阶跃结束（590），不含回落。
    last = win[0]
    for i in range(win[0], min(win[1], n)):
        if ev[i] > 2:
            last = i
    settle = last - win[0]
    return {"fast_cost": fast_cost, "slow_detect": slow_detect,
            "settle": settle, "noise_floor": float(base)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("vision", "out", "timescale.json"))
    args = ap.parse_args()

    frames = make_scene()
    alphas = [0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.8]
    print("\n== S6 多时间尺度：扫 α（单通道 vs 双通道）==\n")
    print(f"{'α':>6s} | {'fast_cost':>9s} {'slow_detect':>11s} {'settle帧':>7s} {'噪声底/帧':>9s}")
    results = {}
    for a in alphas:
        ev = run_channel(frames, a, a)
        m = metrics(ev)
        results[f"single-{a}"] = m
        print(f"{a:6.2f} | {m['fast_cost']:9.0f} {m['slow_detect']:11.0f} "
              f"{m['settle']:7d} {m['noise_floor']:9.2f}")
    evd_f, evd_s = run_channel(frames, 0.5, 0.03, separate=True)
    md = metrics(evd_f + evd_s)
    md_fast = metrics(evd_f)        # 双通道的快组件单独测
    md_slow = metrics(evd_s)        # 双通道的慢组件单独测
    results["dual-0.5+0.03"] = md
    print(f"{'dual':>6s} | {md['fast_cost']:9.0f} {md['slow_detect']:11.0f} "
          f"{md['settle']:7d} {md['noise_floor']:9.2f}")
    print(f"{'dual快':>6s} | {md_fast['fast_cost']:9.0f} {md_fast['slow_detect']:11.0f} "
          f"{md_fast['settle']:7d} {md_fast['noise_floor']:9.2f}   <- 快组件单独")
    print(f"{'dual慢':>6s} | {md_slow['fast_cost']:9.0f} {md_slow['slow_detect']:11.0f} "
          f"{md_slow['settle']:7d} {md_slow['noise_floor']:9.2f}   <- 慢组件单独")

    # 单通道无解区间：settle ≤ 5 帧 ∧ slow_detect ≥ 单通道最大值 50%
    slow_ref = max(m["slow_detect"] for a, m in results.items() if a.startswith("single"))
    ok_single = [(a, m) for a, m in results.items()
                 if a.startswith("single") and m["settle"] <= 5
                 and m["slow_detect"] >= 0.5 * slow_ref]
    dual_ok = md_fast["settle"] <= 5 and md_slow["slow_detect"] >= 0.5 * slow_ref
    print(f"\n判据：快轴 settle ≤ 5 帧 ∧ 慢轴 slow_detect ≥ 单通道最大值（{slow_ref:.0f}）的 50%")
    print(f"  单通道同时满足两轴的 α：{ok_single if ok_single else '无（无解区间）'}")
    print(f"  双通道 (0.5+0.03)：快组件 settle={md_fast['settle']} 帧，"
          f"慢组件 slow_detect={md_slow['slow_detect']} → {'满足 ✓' if dual_ok else '不满足 ✗'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"alphas": alphas, "results": results,
                   "dual_fast_component": md_fast, "dual_slow_component": md_slow,
                   "verdict": {"ok_single": [a for a, _ in ok_single],
                               "dual_ok": dual_ok}}, f, ensure_ascii=False, indent=1)
    print(f"指标：{args.out}")


if __name__ == "__main__":
    main()
