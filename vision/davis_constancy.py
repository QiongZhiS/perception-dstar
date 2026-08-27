"""vision/davis_constancy.py — DAVIS 真实背景 + 合成红球 + 时变色温：时间一致性门的真实检验。

docs/199 §六 1 回填：真实场景色温变化帧间连续（黄昏渐变 shift 从基线缓爬），
时间一致性门应把它当真染色（开、校正）；帧间突变（噪声）该固执（关、不校正）。

实验设计（三层真实性递进）：
  DAVIS 真实帧（背景=真实场景内容偏色、真实噪声）
  + 合成红球（运动学真值，docs/183 合成方案：确认度量=球体掩码红区占比）
  + 受控色温扰动（乘性增益）：
    - baseline：原帧（场景固有偏色，帧间稳定）
    - dusk    ：色温增益随时间缓爬（黄昏：shift 连续渐变）→ 门该开
    - noise   ：色温增益帧间随机跳 → 门该关
    - step    ：色温阶跃（灯突变）→ 突变瞬间关、稳定后重开

验证：
  P1 门行为：dusk/baseline 门开帧多（渐变/稳定=真染色该校正）、noise 门开帧少（抖动该固执）
  P2 球体确认：白平衡校正后球体红区占比保持高、通过率 1.0（docs/199：乘性增益对纯红球
     无色相影响，校正不破坏确认）
  P3 背景回灰：白平衡校正后背景 shift → 基线附近（增益方向正确）
  P4 对照 docs/197：色相旋转在 dusk 下破坏球体（红区掉），白平衡不破坏

用法：
  python vision/davis_constancy.py --video blackswan
  python vision/davis_constancy.py --all
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
import cv2  # noqa: E402
from color_constancy_temporal import (  # noqa: E402
    TemporalGate, raw_shift_frac, bg_gain, confirm_ball)

DAVIS = os.path.join("vision", "out", "davis")
RED = (170, 180, 10)
W, H = 640, 360
N_FRAMES = 60
BX0, BX1, BY0, BY1 = 40, 460, 130, 270


def load_frames(video, n=N_FRAMES):
    vdir = os.path.join(DAVIS, video)
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg")) if os.path.isdir(vdir) else []
    if not jpgs:
        sys.exit(f"没有 {vdir} 的帧——先跑 python vision/davis_setup.py --videos {video}")
    n = min(n, len(jpgs))
    # 抽帧保证覆盖视频（间隔采样）
    idx = np.linspace(0, len(jpgs) - 1, n).astype(int)
    return [cv2.imread(os.path.join(vdir, jpgs[i])) for i in idx]


def overlay_ball(fr, i, color=(0, 0, 200)):
    """叠加合成红球（运动学：x=80+4i, y=180+0.6i，半径 18）。"""
    img = fr.copy()
    rc = (80 + 4.0 * i, 180 + 0.6 * i)
    cv2.circle(img, (int(rc[0]), int(rc[1])), 18, color, -1)
    return img


def warm_gain(k, sat=0.5):
    """暖色温乘性增益：k∈[0,1] 强度，B 降 R 升（黑天鹅标定：R+0.5,B-0.5 → shift +~50）。"""
    dB, dR = -sat * k, sat * 0.6 * k
    return np.array([1 + dB, 1.0, 1 + dR])


def perturb(frames, kind, seed=42):
    """对帧序列施加色温扰动。kind: 'baseline'/'dusk'/'noise'/'step'。
    返回 (新帧序列, 每帧增益)。"""
    rng = np.random.default_rng(seed)
    out, gains = [], []
    for i, fr in enumerate(frames):
        t = i / max(len(frames) - 1, 1)
        if kind == 'baseline':
            g = np.ones(3)
        elif kind == 'dusk':
            g = warm_gain(t)                      # 黄昏：0→满，帧间连续
        elif kind == 'noise':
            g = warm_gain(rng.uniform(0, 1))      # 随机跳
        elif kind == 'step':
            g = warm_gain(1.0 if i >= len(frames) // 2 else 0.0)
        out.append(np.clip(fr.astype(np.float32) * g, 0, 255).astype(np.uint8))
        gains.append(g)
    return out, gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="blackswan")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    videos = ["blackswan", "car-turn", "flamingo", "motorbike", "soccerball", "surf"] \
        if args.all else [args.video]

    print("== DAVIS 真实背景 + 合成红球 + 时变色温：时间一致性门 ==")
    print(f"{'video':12s} {'扰动':10s} {'frac均值':>8s} {'MAD':>5s} {'门开帧':>6s} | "
          f"{'白平衡红区':>10s} {'通过':>5s} {'旋转红区':>8s} {'旋转通过':>8s}")
    for video in videos:
        base = load_frames(video)
        for kind in ['baseline', 'dusk', 'noise', 'step']:
            frames_p, gains = perturb(base, kind)
            frames = [overlay_ball(f, i) for i, f in enumerate(frames_p)]
            # 门 + 白平衡
            g = TemporalGate()
            r_wb, h_wb, nc_wb, f_wb = confirm_ball(frames, g, mode='wb')
            # 门 + 色相旋转（docs/197 对照）
            g2 = TemporalGate()
            r_rot, h_rot, nc_rot, f_rot = confirm_ball(frames, g2, mode='hue')
            # 一次性预计算每帧 (shift, frac, mad) —— raw_shift_frac 含 HSV 转换,避免重复
            stats = [raw_shift_frac(f) for f in frames]
            fracs = [st[1] for st in stats]
            mads = [st[2] for st in stats]
            # 门开帧
            g3 = TemporalGate()
            opens = 0
            for st in stats:
                if g3.update(st[0], st[2]):
                    opens += 1
            print(f"{video:12s} {kind:10s} {np.mean(fracs):8.3f} {np.nanmean(mads):6.1f} {opens:6d} | "
                  f"{f_wb:10.3f} {r_wb:5.3f} {f_rot:8.3f} {r_rot:8.3f}")
        print()


if __name__ == "__main__":
    main()
