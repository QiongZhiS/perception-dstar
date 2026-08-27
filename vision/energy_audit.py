"""vision/energy_audit.py — 感知管线能效审计：每帧实际处理量 vs 全帧基线。

问题（docs/221 对话）："框架解决了高能效持续视觉"是隐含属性（事件驱动/O(N)/Top-K），
但没有直接度量。且发现两处隐蔽开销：multi_situation.ball_obs 与 davis_multi.target_obs
都先对**全帧**做 cv2.cvtColor(BGR2HSV)，然后才采样目标小区域——"稀疏确认"名不副实。

本脚本审计三种实现形态的每帧处理量（用"被处理的像素数 × 通道"作为可比的运算代理，
因为 HSV 转换、EWMA、对比度都是逐像素 O(像素) 操作，像素数 = 运算量）：

  A 全帧基线     ：每帧对全图做完整处理（模拟 CNN/全帧管线）——O(W×H)
  B 当前实现     ：每帧全帧 HSV（O(W×H)） + 稀疏读层（O(事件)）——转导/HSV 仍全帧
  C 稀疏目标版   ：只对目标局部区域做 HSV（O(目标面积)）+ 事件驱动读层——O(变化)

度量：
  px/帧  = 每帧实际参与颜色转换/确认的像素数
  sparsity = 处理像素 / 全帧像素（越小越省）
  对比 = 相对全帧基线的处理量降幅

用途：
  1) 给"高能效持续视觉"一个内部对照数字（资格证明，不碰 SNN 主场）
  2) 暴露 B→C 的优化空间（全帧 HSV 是名不副实的来源）
  3) 支撑"人和机器协作，机器可以看更多"——人眼有中央凹（生理限制），
     机器保留人眼原则（只处理变化）但没有生理限制（可以多焦点/全分辨率采样）
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2

# ---- 场景尺寸 ----
TOY_W, TOY_H = 640, 360        # multi_situation / keep_reject 场景
DAVIS_W, DAVIS_H = 854, 480    # DAVIS 480p

# 目标面积（受控场景球的面积 ~ π·18² ≈ 1018；davis_multi 目标掩码面积 ~ 数千像素）
BALL_AREA = int(np.pi * 18 ** 2)
TARGET_AREA = 5000             # davis_multi 真实目标掩码典型面积

# 事件率（静态场景事件稀疏：C4 0.0223 = 2.2% 像素发事件；运动段更高）
EVENT_RATE_STATIC = 0.0223
EVENT_RATE_MOTION = 0.10


def audit(name, px_per_frame, total_px, note=""):
    ratio = px_per_frame / total_px
    print(f"{name:28s} {px_per_frame:>10,} px/帧 {ratio*100:6.2f}% 全帧 {note}")
    return px_per_frame


def main():
    print("== 感知管线能效审计：每帧实际处理像素 vs 全帧基线 ==")
    print("（像素数 = 运算量代理：HSV/EWMA/对比度都是逐像素 O(px) 操作）\n")

    print(f"--- toy 场景（{TOY_W}×{TOY_H} = {TOY_W*TOY_H:,} px/帧，2 目标球）---")
    full_toy = TOY_W * TOY_H
    # A 全帧基线
    audit("A 全帧基线（CNN式）", full_toy, full_toy, "每帧全图处理")
    # B 当前实现：全帧 HSV + 稀疏读层
    #   HSV 全帧 O(W×H)；确认采样 O(2×BALL_AREA)；事件读层 O(EVENT_RATE×W×H)
    b_px = full_toy + 2 * BALL_AREA + EVENT_RATE_STATIC * full_toy
    audit("B 当前实现（全帧HSV+稀疏读）", b_px, full_toy, "HSV 仍全帧→名不副实")
    # C 稀疏目标版：只对目标局部 HSV + 事件驱动读层
    c_px = 2 * BALL_AREA + EVENT_RATE_STATIC * full_toy
    audit("C 稀疏目标版（局部HSV+事件读）", c_px, full_toy, "只算该算的（人眼原则）")
    print()

    print(f"--- DAVIS 480p（{DAVIS_W}×{DAVIS_H} = {DAVIS_W*DAVIS_H:,} px/帧，2 真实目标）---")
    full_d = DAVIS_W * DAVIS_H
    audit("A 全帧基线（CNN式）", full_d, full_d, "每帧全图处理")
    b_px_d = full_d + 2 * TARGET_AREA + EVENT_RATE_STATIC * full_d
    audit("B 当前实现（全帧HSV+稀疏读）", b_px_d, full_d, "HSV 仍全帧")
    c_px_d = 2 * TARGET_AREA + EVENT_RATE_STATIC * full_d
    audit("C 稀疏目标版（局部HSV+事件读）", c_px_d, full_d, "只算该算的")
    print()

    print("== 关键数字 ==")
    print(f"toy : C/A = {c_px/full_toy*100:.1f}% （省 {100-c_px/full_toy*100:.1f}%）")
    print(f"DAVIS: C/A = {c_px_d/full_d*100:.1f}% （省 {100-c_px_d/full_d*100:.1f}%）")
    print(f"B→C 优化空间（去掉全帧HSV）: toy {b_px/c_px:.1f}×, DAVIS {b_px_d/c_px_d:.1f}×")
    print()

    print("== 判读 ==")
    print("1. '高能效'的真实来源 = 事件驱动（静态 0 事件）+ 只算目标局部（O(K)），")
    print("   不是全帧处理——A→C 省 90%+")
    print("2. 当前实现 B 有隐蔽开销：全帧 HSV 让'稀疏确认'名不副实，B→C 还有优化空间")
    print("3. 人和机器协作的定位：人眼有中央凹/扫视（生理限制），机器保留人眼原则")
    print("   （只处理变化、只算该算的）但没有生理限制——可以多焦点采样（docs/138）、")
    print("   多目标并行（docs/223），'看更多'而不'算更多'")
    print("4. 注意：这是运算量代理（px/帧），不是真实能耗（mW）；真实能耗需硬件测量")
    print("   （SNN 路线用这个，见 SDTrack/SpikeTrack——本文不碰 SNN 主场，这是资格证明）")


if __name__ == "__main__":
    main()
