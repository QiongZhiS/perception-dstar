"""vision/color_transduction.py — 颜色多通道转导（docs/192：色相管识别、明度管结构）。

人眼分工（docs/192 §二）：颜色=区分主轴（"是什么"）、明度/光影=遮挡（"谁在前"）。
多通道转导分层：
  - 色相通道（H）：sin/cos 双值编码（环形空间无 0-180 断裂），颜色事件 = 色相落空
    （物体变色/背景色相变化才触发）；色相区间可作目标外赋（"找绿的车"）
  - 明度通道（V）：沿用灰度转导逻辑（现有 Transduction2D），管遮挡/结构
  - 通道间侧抑制（可选）：色相-明度对抗（颜色区域明度变化被压——文字在彩色底上）

输出：每帧 (ev_h, ev_v)，可合成颜色事件流（任一通道落空）。验证目标：
  - 静止彩色场景：色相通道 0 事件（颜色适应）；颜色变化（物体变色）→ 色相事件
  - 颜色可分性实测（docs/192 §一）：车色相 120 vs 背景 39

用法：td = ColorTransduction(); ev_h, ev_v = td.step(bgr_frame)
"""

import numpy as np
import cv2

from transduction import Transduction2D


class ColorTransduction:
    """色相+明度双通道转导。色相用 sin/cos 编码（环形），明度复用灰度转导。"""

    def __init__(self, thresh=0.15, deadband=0.015, blur=0.8, refractory=2,
                 hue_gain=2.0, lateral=False):
        self.thresh = thresh
        self.deadband = deadband
        self.blur = blur
        self.refractory = refractory
        self.hue_gain = hue_gain        # 色相 sin/cos 落空放大（色相灵敏度）
        self.lateral = lateral          # 通道间侧抑制（色相-明度对抗，可选）
        # 色相通道：双值（sin, cos）各一个背景 EWMA + pending
        self.bg_hs = None               # sin(H) 背景
        self.bg_hc = None               # cos(H) 背景
        self.pending_hs = None
        self.pending_hc = None
        self.cd_h = None
        self.prev_hs = None
        self.prev_hc = None
        # 明度通道：复用 Transduction2D 的机制（独立实例，只取明度）
        self.lum = Transduction2D(thresh=thresh, deadband=deadband, blur=blur,
                                  refractory=refractory)

    def _hue_encode(self, h):
        """色相 0-180 → (sin(2πh/180), cos(2πh/180))：环形空间无断裂。"""
        ang = 2 * np.pi * h.astype(np.float32) / 180.0
        return np.sin(ang), np.cos(ang)

    def step(self, bgr):
        """输入 BGR 帧 → (ev_h 色相事件, ev_v 明度事件)。"""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, _, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        if self.blur > 0:
            h = cv2.GaussianBlur(h, (0, 0), self.blur)
            v = cv2.GaussianBlur(v, (0, 0), self.blur)
        hs, hc = self._hue_encode(h)
        hs = hs.astype(np.float32)
        hc = hc.astype(np.float32)
        vf = v.astype(np.float32) / 255.0      # 明度归一化（与灰度转导尺度一致）

        # ---- 色相通道：双值背景 EWMA → 时间差分 → pending → 事件 ----
        if self.bg_hs is None:
            self.bg_hs = hs.copy()
            self.bg_hc = hc.copy()
            self.prev_hs = hs.copy()
            self.prev_hc = hc.copy()
            self.pending_hs = np.zeros_like(hs)
            self.pending_hc = np.zeros_like(hc)
            self.cd_h = np.zeros(hs.shape, np.int32)
        a = 0.2                                 # 色相背景 EWMA 系数
        self.bg_hs = a * hs + (1 - a) * self.bg_hs
        self.bg_hc = a * hc + (1 - a) * self.bg_hc
        ds = hs - self.prev_hs
        dc = hc - self.prev_hc
        self.prev_hs, self.prev_hc = hs, hc
        if self.deadband > 0:
            ds = np.where(np.abs(ds) > self.deadband, ds, 0.0)
            dc = np.where(np.abs(dc) > self.deadband, dc, 0.0)
        self.pending_hs += self.hue_gain * ds
        self.pending_hc += self.hue_gain * dc
        # 色相事件：sin 或 cos 任一过阈
        ev_h = np.zeros(hs.shape, np.int8)
        ready = self.cd_h == 0
        pos = ready & ((self.pending_hs > self.thresh) | (self.pending_hc > self.thresh))
        neg = ready & ((self.pending_hs < -self.thresh) | (self.pending_hc < -self.thresh))
        ev_h[pos] = 1
        ev_h[neg] = -1
        self.pending_hs[pos | neg] = 0.0
        self.pending_hc[pos | neg] = 0.0
        self.cd_h = np.maximum(self.cd_h - 1, 0)
        self.cd_h[pos | neg] = self.refractory

        # ---- 明度通道：灰度转导（V 当灰度用）----
        gray8 = np.clip(v, 0, 255).astype(np.uint8)
        ev_f, ev_s, _, _ = self.lum.step(gray8)
        ev_v = np.where((ev_f != 0) | (ev_s != 0), 1, 0)

        if self.lateral:
            # 通道间侧抑制（可选）：色相事件被明度变化压（同区域明度大变化=遮挡/结构，
            # 抑制颜色——文字在彩色底上：明度对比主导）；简化：明度事件区域压色相事件
            ev_h = np.where(ev_v != 0, 0, ev_h)
        return ev_h, ev_v
