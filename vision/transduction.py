"""
vision/transduction.py — 2D 事件驱动转导层（看视频测试版）

把普通视频当输入，模拟事件相机式转导：
  帧 → 灰度 → 对数亮度 → 多时间尺度背景估计（快/慢 EWMA，视锥/视杆式时间常数）
  → 自适应增益（对数域减背景 = 韦伯式，docs/135 的 2D 推广）
  → 2D 侧抑制（中心-周围差分，docs/133 §2.2）
  → 时间差分 + 阈值 → 事件（快通道=红 / 慢通道=蓝 / 双发=白）

输出（vision/out/）：
  events_<name>.mp4   三面板可视化：原帧 | 增益+侧抑制残差 | 事件流
  metrics_<name>.json 每帧指标（事件数/稀疏度/亮度/增益）+ 分段摘要
  frame_<name>_NNNN.png 抽样帧（--every N，0=不存）

用法：
  python vision/transduction.py                    # 生成合成演示视频再跑
  python vision/transduction.py --video x.mp4      # 跑真实视频
  python vision/transduction.py --every 0          # 不存抽样帧

第一测（看视频）：
  - 开场缓渐变：两条通道都把它预测掉 → 几乎无事件（自适应增益在工作）
  - 运动段：移动边缘持续产生事件，静态纹理/棋盘被背景吸收 → 无事件
  - 阶跃闪变段：快通道几帧恢复（红），慢通道几十帧才追上（蓝）——多时间尺度差异
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv：python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless")


# ---------------------------------------------------------------------------
# 合成演示视频（受控场景：运动物体 + 静态纹理 + 光照渐变 + 阶跃闪变）
# 光照设计：开场 2s 缓渐变（1.0→0.8，测"适应把渐变预测掉"），随后三处阶跃
# 闪变（测"多时间尺度"：快通道几帧恢复、慢通道几十帧才追上，红快蓝慢一目了然）
# ---------------------------------------------------------------------------
def make_demo_video(path, fps=30, width=320, height=240, seconds=10, seed=7, storm=False,
                    mode="dips", storm_sigma=12.0, storm_lo=4.0, storm_hi=4.9):
    """mode="dips"：0.3-0.5s 降幅（转导层 S6 检查单用，残留衰减期够长）；
    mode="blinks"：~0.1s 降幅（读层闪变维持用——人眼眨眼 100-150ms 的设计域）。
    storm=True：storm_lo-storm_hi 秒是"噪声风暴"（σ=storm_sigma，光照不变，
    夹在两处降幅之间，纯测噪声适应——固定参数应被噪声随机游走淹没）。"""
    rng = np.random.default_rng(seed)
    n = fps * seconds
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # 静态棋盘 + 噪声材质斑（测侧抑制/背景适应：静态纹理不该出事件）
    bg = np.zeros((height, width), np.float32)
    step = 24
    for y in range(0, height, step):
        for x in range(0, width, step):
            bg[y:y + step, x:x + step] = 96 if ((x // step) + (y // step)) % 2 == 0 else 128
    bg[30:70, 40:80] = rng.normal(150, 12, (40, 40)).astype(np.float32)  # 噪声材质斑

    def slow_pos(t):
        return (int(width * 0.25 + width * 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (seconds * 1.5)))),
                int(height * 0.55))

    def fast_pos(t):
        return (int(width * 0.5 + width * 0.45 * np.sin(2 * np.pi * t / 1.2)),
                int(height * 0.35 + height * 0.15 * np.sin(2 * np.pi * t / 0.9)))

    def light_at(t):
        """光照曲线：开场缓渐变 → 平稳 → 三处降幅。"""
        if t < 2.0:
            return 1.0 - 0.2 * (t / 2.0)            # 缓渐变 1.0→0.8（应被适应掉）
        light = 1.0
        if mode == "blinks":
            for t0, level in [(3.0, 0.5), (5.0, 0.3), (7.5, 0.7)]:
                if t0 <= t < t0 + 3 / 30:            # ~0.1s 眨眼尺度降幅
                    light = level
        else:
            for t0, level, dur in [(3.0, 0.5, 0.5), (5.0, 0.3, 0.3), (7.5, 0.7, 0.2)]:
                if t0 <= t < t0 + dur:               # 0.2-0.5s 降幅（S6 残留衰减）
                    light = level
        return light

    for i in range(n):
        t = i / fps
        light = light_at(t)
        img = bg * light
        cx, cy = slow_pos(t)
        cv2.circle(img, (cx, cy), 22, 200 * light, -1)          # 慢速大圆
        fx, fy = fast_pos(t)
        cv2.rectangle(img, (int(fx) - 10, int(fy) - 10),
                      (int(fx) + 10, int(fy) + 10), 45 * light, -1)  # 快速小方块
        # 噪声风暴：storm=True 时 storm_lo-storm_hi 是"噪声风暴"（σ=storm_sigma，
        # 光照不变，夹在两处降幅之间，纯测噪声适应——固定参数应被噪声随机游走淹没）
        noise = storm_sigma if (storm and storm_lo <= t < storm_hi) else 0.8
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return fps, width, height


# ---------------------------------------------------------------------------
# 2D 转导层
# ---------------------------------------------------------------------------
class Transduction2D:
    """帧 → (快事件, 慢事件, 侧抑制残差, 慢通道对比度)。

    机制（docs/133 §二 + docs/135 + 笔记 L1）：
      - 对数域：L = log(gray+1)；事件相机语义 = log 强度变化
      - 自适应增益（时间预测）：contrast = L − bg，bg 为多时间尺度 EWMA
        （快 α_fast=视锥式 / 慢 α_slow=视杆式长积分）——"先验把可预测的扣掉"
      - 事件累积器（ESIM 式）：pending += Δcontrast，|pending|>θ 才发事件并按 θ
        复位，留余量——单帧阈值抓不住的缓慢变化（慢通道恢复期）会累积触发，
        产生"快通道几帧恢复（红短）、慢通道持续补发（蓝长）"的多时间尺度差异
      - refractory：发过事件的像素静默 R 帧，抑制噪声随机游走累积
      - 2D 侧抑制（空间预测）：residual = contrast − blur(contrast)，中心-周围，
        单独作残差面板显示；是否进入事件路径（S7 消融）留实验阶段测 d*
    """

    def __init__(self, alpha_fast=0.5, alpha_slow=0.03, thresh=0.15, ksize=7, refractory=2,
                 blur=0.8, deadband=0.015, adaptive=False, k_theta=15.0, k_db=1.5,
                 thresh_max=0.6, db_max=0.15, learned_map=None, fall_tau=0.05,
                 noise_mode="mad", k_hf=0.12, lateral_in_path=False):
        self.a_fast = alpha_fast   # 视锥式：快适应（~10ms 量级）
        self.a_slow = alpha_slow   # 视杆式：慢适应/长积分（~100ms 量级）
        self.thresh = thresh       # 事件阈值（校准的工作点；adaptive 模式下只向上抬）
        self.ksize = ksize
        self.refractory = refractory
        self.blur = blur           # 光学模糊 σ（真实传感器有；压逐像素噪声）
        self.deadband = deadband   # 死区：校准的工作点；adaptive 模式下只向上抬
        self.adaptive = adaptive   # 自适应模式：θ/死区 = max(校准点, f(σ̂))——只抬不降
        self.k_theta = k_theta     # θ = max(θ_校准, k_theta × σ̂)
        self.k_db = k_db           # deadband = max(db_校准, k_db × σ̂)
        self.thresh_max = thresh_max
        self.db_max = db_max
        self.learned_map = learned_map   # 学出映射：list of (σ̂, θ, db)，np.interp 用
        self.fall_tau = fall_tau         # 学出模式下降速率（慢恢复；上升总是即时）
        self.noise_mode = noise_mode     # "mad"（MAD of Δlog）/"hf"（高频能量）/"both"
        self.k_hf = k_hf                 # hf→σ̂ 标定：σ̂_hf = k_hf × mean|Laplacian(ΔL)|
        self.lateral_in_path = lateral_in_path  # S7：侧抑制残差参与事件触发（中心-周围
                                                # 空间预测误差，压均匀区/增强边缘）
        self._lat_th_f = None
        self._lat_th_s = None
        self.sigma_hat = None      # 噪声估计（MAD of Δlog）
        self.prev_L = None
        self.bg_fast = None
        self.bg_slow = None
        self.prev_fast = None
        self.prev_slow = None
        self.pending_f = None
        self.pending_s = None
        self.cd_f = None
        self.cd_s = None

    def _update_noise_estimate(self, L):
        """σ̂ = 噪声/变化强度估计，EWMA 平滑。

        noise_mode="mad"（默认）：σ̂ = MAD(ΔL − median(ΔL))/0.6745——只对逐像素
        独立噪声敏感，内容运动（空间相关的结构事件）估不出（docs/185 实测：真实视频
        σ̂ 恒 0）。
        noise_mode="hf"：σ̂ = k_hf × mean|Laplacian(ΔL)|——高频能量，内容边缘/噪声
        都反映，真实视频上持续非零（docs/185）。
        noise_mode="both"：max(两者)。

        ΔL 减掉全局位移（median）后只剩逐像素噪声 + 稀疏运动边缘：
        - 全局光照变化（降幅）→ median(ΔL) 吃掉 → 不污染 σ̂
        - 噪声风暴 → MAD 反映真实噪声 → θ/死区跟着抬高（越噪声越保守）
        median 对稀疏运动鲁棒（边缘是少数像素）。
        """
        d = L - self.prev_L
        sig_mad = np.median(np.abs(d - np.median(d))) / 0.6745
        if self.noise_mode == "hf":
            lap = cv2.Laplacian(d.astype(np.float32), cv2.CV_32F, ksize=3)
            sig = self.k_hf * float(np.mean(np.abs(lap)))
        elif self.noise_mode == "both":
            lap = cv2.Laplacian(d.astype(np.float32), cv2.CV_32F, ksize=3)
            sig_hf = self.k_hf * float(np.mean(np.abs(lap)))
            sig = max(sig_mad, sig_hf)
        else:
            sig = sig_mad
        if self.sigma_hat is None:
            self.sigma_hat = sig
        else:
            self.sigma_hat = 0.15 * sig + 0.85 * self.sigma_hat
        if self.learned_map is not None:
            # 学出模式：σ̂ → (θ, db) 查表插值；上升即时、下降慢（fall_tau）——
            # 世界变噪立即抬杆（保守），风暴后缓缓恢复（灵敏度不永久丢失）。
            lm = sorted(self.learned_map, key=lambda p: p[0])
            xs = [p[0] for p in lm]; ths = [p[1] for p in lm]; dbs = [p[2] for p in lm]
            desired_th = float(np.interp(self.sigma_hat, xs, ths, left=ths[0], right=ths[-1]))
            desired_db = float(np.interp(self.sigma_hat, xs, dbs, left=dbs[0], right=dbs[-1]))
            self.thresh = desired_th if desired_th >= self.thresh else \
                self.fall_tau * desired_th + (1 - self.fall_tau) * self.thresh
            self.deadband = desired_db if desired_db >= self.deadband else \
                self.fall_tau * desired_db + (1 - self.fall_tau) * self.deadband
        elif self.adaptive:
            # 只抬不降：校准的工作点是地板，世界变噪时把杆抬起来（越噪声越保守）。
            # 不向下调——避免 σ̂ 低估时变得比校准点更敏感。
            self.thresh = float(np.clip(max(self.thresh, self.k_theta * self.sigma_hat),
                                        0.0, self.thresh_max))
            self.deadband = float(np.clip(max(self.deadband, self.k_db * self.sigma_hat),
                                          0.0, self.db_max))
        self.prev_L = L
        return self.sigma_hat

    def _emit(self, pending, cd, thresh_map=None):
        """pending 累积器 → 事件；refractory 内不发。thresh_map: 逐像素阈值
        （S7 侧抑制调制：均匀区阈值高被抑制，边缘阈值低易触发）；None=全局 self.thresh。"""
        ev = np.zeros(pending.shape, np.int8)
        ready = cd == 0
        th = thresh_map if thresh_map is not None else self.thresh
        pos = ready & (pending > th)
        neg = ready & (pending < -th)
        ev[pos] = 1
        ev[neg] = -1
        pending = pending.copy()
        pending[pos] = 0.0              # 事件即参考点更新：发完归零，余量不留
        pending[neg] = 0.0
        cd = np.maximum(cd - 1, 0)          # 所有像素冷却递减
        cd[pos | neg] = self.refractory     # 发过事件的进入静默期
        return ev, pending, cd

    def step(self, gray):
        # σ̂ 在模糊前的原始信号上估（传感器看到的噪声）；处理用模糊后信号
        L_raw = np.log(np.maximum(gray.astype(np.float32), 1.0))
        if self.blur > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.blur)
        L = np.log(np.maximum(gray.astype(np.float32), 1.0))
        if self.bg_fast is None:
            self.bg_fast = L.copy()
            self.bg_slow = L.copy()
            self.prev_fast = np.zeros_like(L)
            self.prev_slow = np.zeros_like(L)
            self.pending_f = np.zeros_like(L)
            self.pending_s = np.zeros_like(L)
            self.cd_f = np.zeros(L.shape, np.int32)
            self.cd_s = np.zeros(L.shape, np.int32)
            self.prev_L = L_raw.copy()
            return (np.zeros(L.shape, np.int8), np.zeros(L.shape, np.int8),
                    np.zeros_like(L), np.zeros_like(L))

        self._update_noise_estimate(L_raw)  # 自适应模式：θ/死区跟随原始信号噪声

        self.bg_fast = self.a_fast * L + (1 - self.a_fast) * self.bg_fast
        self.bg_slow = self.a_slow * L + (1 - self.a_slow) * self.bg_slow

        cf = L - self.bg_fast                       # 快通道对比度（增益后）
        cs = L - self.bg_slow                       # 慢通道对比度

        d_f = cf - self.prev_fast                   # 时间差分（只传变化）
        d_s = cs - self.prev_slow
        self.prev_fast, self.prev_slow = cf, cs
        if self.deadband > 0:
            d_f = np.where(np.abs(d_f) > self.deadband, d_f, 0.0)  # 死区：杀噪声游走
            d_s = np.where(np.abs(d_s) > self.deadband, d_s, 0.0)
        if self.lateral_in_path:
            # S7：侧抑制（中心-周围空间预测误差）参与事件触发——
            # 视网膜中心-周围是**抑制性**的（docs/133 §2.2）：均匀区空间预测误差≈0
            # → 被压制；边缘处误差大 → 保留。实现：空间残差大的像素阈值低（易触发），
            # 残差小的像素阈值高（被抑制）。这是"局部阈值调制"，不是叠加触发。
            k = (self.ksize, self.ksize)
            r_f = np.abs(cf - cv2.blur(cf, k))      # 快通道空间残差（0=均匀区）
            r_s = np.abs(cs - cv2.blur(cs, k))
            # 归一化：残差越大阈值越低（0.3×θ 到 θ 之间线性）
            self._lat_th_f = np.clip(1.0 - 0.7 * r_f / (np.max(r_f) + 1e-6), 0.3, 1.0)
            self._lat_th_s = np.clip(1.0 - 0.7 * r_s / (np.max(r_s) + 1e-6), 0.3, 1.0)
        self.pending_f += d_f                        # 累积 Δcontrast
        self.pending_s += d_s

        th_f = self._lat_th_f * self.thresh if self.lateral_in_path else None
        th_s = self._lat_th_s * self.thresh if self.lateral_in_path else None
        ev_f, self.pending_f, self.cd_f = self._emit(self.pending_f, self.cd_f, th_f)
        ev_s, self.pending_s, self.cd_s = self._emit(self.pending_s, self.cd_s, th_s)

        # 2D 侧抑制（中心-周围，空间预测）——残差面板显示，事件路径留 S7 消融
        k = (self.ksize, self.ksize)
        residual = ((cf - cv2.blur(cf, k)) + (cs - cv2.blur(cs, k))) / 2
        return ev_f, ev_s, residual, cs


# ---------------------------------------------------------------------------
# 可视化 + 指标
# ---------------------------------------------------------------------------
def make_panels(gray_bgr, residual, ev_f, ev_s, info):
    h, w = gray_bgr.shape[:2]
    p2 = np.clip(residual / 0.5 * 127 + 127, 0, 255).astype(np.uint8)   # 残差居中显示
    p2 = cv2.cvtColor(p2, cv2.COLOR_GRAY2BGR)

    acc = np.zeros((h, w, 3), np.uint8)          # 事件尾迹
    acc = (acc * 0.88).astype(np.uint8)          # 事件尾迹衰减
    both = (ev_f != 0) & (ev_s != 0)
    only_f = (ev_f != 0) & ~both
    only_s = (ev_s != 0) & ~both
    acc[only_f] = (0, 0, 255)                    # 快通道=红
    acc[only_s] = (255, 0, 0)                    # 慢通道=蓝
    acc[both] = (255, 255, 255)                  # 双发=白

    pad = np.zeros((h, 8, 3), np.uint8)
    canvas = np.hstack([gray_bgr, pad, p2, pad, acc])
    txt = (f"frame {info['frame']:04d}  light {info['light']:.2f}  "
           f"E_f {info['ev_f']:5d}  E_s {info['ev_s']:5d}  E {info['ev_t']:5d}  "
           f"sparse {info['sparsity']:.4f}")
    cv2.putText(canvas, txt, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def run_pipeline(src_video, args):
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        sys.exit(f"打不开视频：{src_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.scale and args.scale != 1.0:
        width, height = max(2, int(width * args.scale)), max(2, int(height * args.scale))
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else n_frames
    print(f"源 {width//int(1/args.scale) if args.scale else '?'}x 处理 {width}x{height}，"
          f"上限 {max_frames} 帧（共 {n_frames}）")

    name = os.path.splitext(os.path.basename(src_video))[0]
    os.makedirs(args.out, exist_ok=True)
    video_path = os.path.join(args.out, f"events_{name}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width * 3 + 16, height))

    td = Transduction2D(alpha_fast=args.alpha_fast, alpha_slow=args.alpha_slow,
                        thresh=args.thresh, ksize=args.inhibit_ksize, blur=args.blur,
                        deadband=args.deadband)
    metrics = []
    frame_idx = 0
    n_pix = width * height

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.scale and args.scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ev_f, ev_s, residual, cs = td.step(gray_bgr)
        ev_t = np.where((ev_f != 0) | (ev_s != 0), 1, 0)
        n_f = int((ev_f != 0).sum()); n_s = int((ev_s != 0).sum()); n_t = int(ev_t.sum())

        info = {
            "frame": frame_idx, "t": round(frame_idx / fps, 3),
            "light": float(gray_bgr.mean()),
            "ev_f": n_f, "ev_s": n_s, "ev_t": n_t,
            "sparsity": round(n_t / n_pix, 5),
            "gain_slow": float(cs.mean()),
        }
        metrics.append(info)

        canvas = make_panels(cv2.cvtColor(gray_bgr, cv2.COLOR_GRAY2BGR), residual,
                             ev_f, ev_s, info)
        writer.write(canvas)
        if args.every > 0 and frame_idx % args.every == 0:
            cv2.imwrite(os.path.join(args.out, f"frame_{name}_{frame_idx:04d}.png"), canvas)
        frame_idx += 1
        if frame_idx >= max_frames:
            break

    cap.release(); writer.release()

    # 分段摘要（按视频前 2/3 与后 1/3 粗分：渐变段 vs 平稳段）
    seg = {"total_frames": frame_idx, "total_events": sum(m["ev_t"] for m in metrics),
           "mean_sparsity": float(np.mean([m["sparsity"] for m in metrics]))}
    third = max(1, frame_idx // 3)
    for label, lo, hi in [("ramp1", 0, third), ("ramp2", third, 2 * third), ("steady", 2 * third, frame_idx)]:
        ms = metrics[lo:hi]
        seg[label] = {"frames": len(ms),
                      "ev_f_mean": float(np.mean([m["ev_f"] for m in ms])),
                      "ev_s_mean": float(np.mean([m["ev_s"] for m in ms])),
                      "ev_t_mean": float(np.mean([m["ev_t"] for m in ms]))}
    out = {"params": {"alpha_fast": args.alpha_fast, "alpha_slow": args.alpha_slow,
                      "thresh": args.thresh, "inhibit_ksize": args.inhibit_ksize},
           "source": os.path.basename(src_video), "fps": fps, "size": [width, height],
           "summary": seg, "frames": metrics}
    mpath = os.path.join(args.out, f"metrics_{name}.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return video_path, mpath, out


def main():
    ap = argparse.ArgumentParser(description="2D 事件驱动转导层（看视频测试）")
    ap.add_argument("--video", default=None, help="输入视频；缺省时自动生成合成演示视频")
    ap.add_argument("--out", default=os.path.join("vision", "out"))
    ap.add_argument("--alpha-fast", type=float, default=0.5)
    ap.add_argument("--alpha-slow", type=float, default=0.03)
    ap.add_argument("--thresh", type=float, default=0.15)
    ap.add_argument("--inhibit-ksize", type=int, default=7)
    ap.add_argument("--blur", type=float, default=0.8, help="输入光学模糊 σ（0=关）")
    ap.add_argument("--deadband", type=float, default=0.015, help="死区：小于它的 Δ 不累积")
    ap.add_argument("--every", type=int, default=30, help="每 N 帧存一张抽样帧（0=不存）")
    ap.add_argument("--scale", type=float, default=1.0, help="输入下采样系数（0.5=半分辨率）")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    args = ap.parse_args()

    if args.video and os.path.exists(args.video):
        src = args.video
    else:
        os.makedirs(args.out, exist_ok=True)
        src = os.path.join(args.out, "demo_scene.mp4")
        if not os.path.exists(src):
            print("生成合成演示视频（光照渐变 + 慢速圆 + 快速方块 + 静态纹理 + 闪变）...")
            make_demo_video(src)
        else:
            print(f"复用已生成的演示视频 {src}")

    print(f"处理：{src}")
    vpath, mpath, out = run_pipeline(src, args)
    s = out["summary"]
    print(f"完成：\n  事件视频 {vpath}\n  指标     {mpath}")
    print(f"  总帧 {s['total_frames']}，总事件 {s['total_events']}，平均稀疏度 {s['mean_sparsity']}")
    print("  分段平均事件数（快/慢/总）：")
    for k in ("ramp1", "ramp2", "steady"):
        v = s[k]
        print(f"    {k:6s} {v['ev_f_mean']:7.1f} / {v['ev_s_mean']:7.1f} / {v['ev_t_mean']:7.1f}")


if __name__ == "__main__":
    main()
