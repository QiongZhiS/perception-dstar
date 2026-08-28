"""vision/real_stream_test.py — 半真实实验：docs/241 流式回路接 DAVIS 真实视频像素流（docs/243）。

把 docs/241 的流式回路（CompLoop + 模式表 + 结构度量 + 持久化，stream_test 原样复用）
接到 DAVIS 真实视频像素流上：半真实 = 真实像素、自然运动/噪声/色相漂移，但仍是录制的
受控序列。机制零重调——这本身就是被测对象（toy 校准过拟合检验，docs/243 缘起）。

环境（预注册，docs/243 §1.1/§1.2，冻结）：
  数据 = vision/out/davis/<video>/ 的 JPEG 帧（854x480 彩色，灰度化 + 缩放到 160x120
  校准工作分辨率后作为流式输入）。无注入、无 jitter、无 RNG——真实像素流是确定性的。
  R0 (lvcode 30, single_video_loop)：flamingo 80 帧 x5 = 400 帧（循环 = 半真实妥协）
  R1 (lvcode 31, concat_scenes)：9 视频按序拼接 = 588 帧（视频切换 = 真实场景切换）
  R2 (lvcode 32, drift_observation)：surf 55 帧自然色相漂移观测（可选，不进主判定）
  --native：原生 854x480 的 R1 分档饱和诊断（信息性，R_RST_NATIVE_*，不进判据）

判据（预注册，docs/243 §1.4，冻结）：
  1. REAL_STABLE   : 每级 mae 末/首四分之一窗口均值比 <= 1.5（真实流误差有界）
  2. STRUCT_EMERGES: SC2(R0) > 0 且 SC2(R1) > 0（结构在真实像素上涌现 = 机制迁移）
  3. STATE_PERSIST : 每级 x 3 中断点（R0:120/133/250；R1:240/355/500）save->load->续跑
                     与不中断运行逐帧全等（per-frame (mae,att,ev,theta,db) + finalize）
  4. BRIDGE_TO_REAL: churn(R1) <= 0.5（放宽）且 bridge_corr_switch >= 0.5（条目创建时点
                     落在下一视频跨度内 = 场景切换被结构编码）
判定：1-4 全过 = REAL_PASS；判据 2 失败 = REAL_FAIL_STRUCT；部分 = REAL_PARTIAL。

统计外壳（docs/228 精神，docs/243 §1.5，冻结）：固定确定性流（无种子协议，声明）；
窗口级 mean±SD（ddof=1）+ bootstrap 95% CI（2000 次，种子 20260828）over 窗口；
checkpoint（ckpt_rst_<hash>.json，--resume）、JSON 归档（rst_<tag>.json /
rst_persist.json / rst_drift.json）。
安全纪律（docs/228/234 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_RST_* 摘要块；
禁止把日志/JSON 原文送进上下文。禁止修改 stream_test.py / compose_test.py /
critical_point.py / so_probe.py 等既有脚本——只 import 复用。

用法：
  python vision/real_stream_test.py --tag main
  python vision/real_stream_test.py --tag timing --no-persist
  python vision/real_stream_test.py --native        # 原生分辨率分档饱和诊断
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import cv2

from critical_point import mean_sd, bootstrap_ci
from compose_test import CompLoop, ENERGY_BINS, UPPER_BINS
from stream_test import LOOP_CFG, save_loop_state, load_loop_state

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
DEFAULT_STATE = os.path.join("vision", "out", "state")
DAVIS_DIR = os.path.join("vision", "out", "davis")
RESIZE = (160, 120)
WINDOW = 10

# 视频顺序与跨度（预注册，docs/243 §1.2；跨度由数据帧数在运行时计算）
VIDEOS = ["flamingo", "surf", "bear", "camel", "dog", "blackswan",
          "car-turn", "motorbike", "soccerball"]

# 持久化中断点（帧索引，p=第 p 帧处理完后 save；含窗口内点）。预注册，冻结。
PERSIST_POINTS = {30: (120, 133, 250), 31: (240, 355, 500)}

LEVEL_NAMES = {30: "single_video_loop", 31: "concat_scenes", 32: "drift_observation"}


# ---------------- 数据加载（数据目录可读；JPEG 内容只在脚本内解码） ----------------
def load_video_frames(name, native=False):
    """读一个视频的全部 .jpg 帧：灰度化 + 缩放到 160x120（校准工作分辨率）。
    native=True 时不缩放（原生 854x480，仅诊断用）。"""
    d = os.path.join(DAVIS_DIR, name)
    paths = sorted(p for p in os.listdir(d) if p.lower().endswith(".jpg"))
    if not paths:
        raise RuntimeError("no jpg frames in %s" % d)
    frames = []
    for p in paths:
        img = cv2.imread(os.path.join(d, p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("cannot read %s" % os.path.join(d, p))
        if not native:
            img = cv2.resize(img, RESIZE, interpolation=cv2.INTER_AREA)
        frames.append(np.ascontiguousarray(img, dtype=np.uint8))
    return frames


def build_streams():
    """返回 (r0_frames, r1_frames, r1_spans)。r1_spans = [(start, end), ...] 逐视频帧区间。"""
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0 = allv["flamingo"] * 5                      # R0：flamingo x5 = 400 帧（循环妥协）
    r1, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    return r0, r1, spans


# ---------------- 运行一个级（主 run + 持久化 round-trip） ----------------
def run_level(lvcode, frames, points, state_dir, spans=None):
    n_frames = len(frames)
    n_windows = max(1, n_frames // WINDOW)
    loop = CompLoop(window=WINDOW, **LOOP_CFG)
    main_outs = [loop.step(g) for g in frames]
    base = loop.finalize(n_windows, labels=None)   # 无真值标签（真实视频无 ctx/regime 真值）

    # ---- 判据 1 组件：真实流稳定性 ratio（末/首四分之一窗口 MAE 均值比） ----
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0

    # ---- 窗口级统计（docs/228 精神：固定流 -> 窗口为随机单元） ----
    mae_m, mae_sd = mean_sd(list(mae_arr))
    mae_lo, mae_hi = bootstrap_ci(list(mae_arr))

    # ---- 判据 3：持久化 round-trip（save->load->续跑 vs 不中断主 run 逐帧全等） ----
    persist = []
    for p in points:
        if p <= 0 or p >= n_frames:
            continue
        pre = CompLoop(window=WINDOW, **LOOP_CFG)
        pre_outs = [pre.step(g) for g in frames[:p]]
        st = save_loop_state(pre)
        path = os.path.join(state_dir, "rst_persist_%d_%d.json" % (lvcode, p))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f, sort_keys=True)
        with open(path, encoding="utf-8") as f:
            st2 = json.load(f)
        post = CompLoop(window=WINDOW, **LOOP_CFG)
        load_loop_state(post, st2)
        post_outs = [post.step(g) for g in frames[p:]]
        eq = (pre_outs == main_outs[:p]) and (post_outs == main_outs[p:])
        if eq:
            f2 = post.finalize(n_windows, labels=None)
            eq = (f2 == base)
        persist.append({"point": p, "seq_eq": int(eq), "state_path": path})

    # ---- 判据 4 组件（R1）：条目创建时点 vs 视频切换时点对应 ----
    bridge = None
    if lvcode == 31:
        switches = [spans[i][0] for i in range(1, len(spans))]
        entries = [(e["created"], list(e["key"])) for e in base["entry_log"]]
        enc_sw = 0
        for sp in spans[1:]:
            if any(sp[0] <= w * WINDOW < sp[1] for w, _ in entries):
                enc_sw += 1
        enc_vid = sum(1 for sp in spans
                      if any(sp[0] <= w * WINDOW < sp[1] for w, _ in entries))
        spur = sum(1 for w, _ in entries
                   if not any(sp[0] <= w * WINDOW < sp[1] for sp in spans))
        bridge = {"switches": len(switches), "encoded_sw": enc_sw,
                  "bridge_corr_switch": round(enc_sw / max(1, len(switches)), 4),
                  "videos": len(spans), "encoded_vid": enc_vid,
                  "bridge_corr_video": round(enc_vid / max(1, len(spans)), 4),
                  "churn": base["churn_frac"],
                  "spurious": spur,
                  "spurious_rate": round(spur / max(1, base["sc1"]), 4),
                  "sw_enc_flags": [int(any(sp[0] <= w * WINDOW < sp[1]
                                           for w, _ in entries))
                                   for sp in spans[1:]],
                  "vid_enc_flags": [int(any(sp[0] <= w * WINDOW < sp[1]
                                            for w, _ in entries))
                                    for sp in spans]}

    out = dict(base)
    out.update({
        "level": lvcode, "name": LEVEL_NAMES[lvcode], "frames": n_frames,
        "n_windows": len(mae_arr),
        "mae_mean_win": round(mae_m, 6), "mae_sd_win": round(mae_sd, 6),
        "mae_ci95": [round(mae_lo, 6), round(mae_hi, 6)],
        "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
        "ratio": round(ratio, 6),
        "pin_frac": base["pin_frac"], "theta_mean": base["theta_mean"],
        "persist": persist,
        "bridge": bridge,
    })
    return out


# ---------------- R2 自然漂移观测（surf；可选，不进主判定） ----------------
def _frame_hue_rad(bgr_small):
    """非饱和像素的饱和度加权圆形平均色相（弧度）。"""
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 180.0 * (2.0 * np.pi)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    mask = (s >= 0.2) & (v >= 0.3)
    if int(mask.sum()) < 10:
        return None
    w = s[mask]
    sx = float(np.sum(w * np.sin(h[mask])))
    sy = float(np.sum(w * np.cos(h[mask])))
    return float(np.arctan2(sx, sy)) % (2.0 * np.pi)


def _circ_dist(a, b):
    d = abs(a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def run_drift():
    """surf 自然色相漂移观测：漂移量 + 回路（灰度）对漂移的追踪相关性。"""
    d = os.path.join(DAVIS_DIR, "surf")
    paths = sorted(p for p in os.listdir(d) if p.lower().endswith(".jpg"))
    gray_frames, hue_means = [], []
    for p in paths:
        bgr = cv2.imread(os.path.join(d, p), cv2.IMREAD_COLOR)
        small = cv2.resize(bgr, RESIZE, interpolation=cv2.INTER_AREA)
        gray_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        hue_means.append(_frame_hue_rad(small))
    valid = [(i, h) for i, h in enumerate(hue_means) if h is not None]
    hs = [h for _, h in valid]
    n = len(hs)
    drift_range = (max(_circ_dist(hs[i], hs[j]) for i in range(n) for j in range(i + 1, n))
                   if n >= 2 else 0.0) * 180.0 / np.pi
    half = n // 2
    if half >= 1:
        m0x = float(np.mean(np.sin(hs[:half]))), float(np.mean(np.cos(hs[:half])))
        m1x = float(np.mean(np.sin(hs[half:]))), float(np.mean(np.cos(hs[half:])))
        m0 = float(np.arctan2(m0x[0], m0x[1])) % (2 * np.pi)
        m1 = float(np.arctan2(m1x[0], m1x[1])) % (2 * np.pi)
        drift_signed = (m1 - m0)
        if drift_signed > np.pi:
            drift_signed -= 2 * np.pi
        if drift_signed < -np.pi:
            drift_signed += 2 * np.pi
        drift_signed *= 180.0 / np.pi
    else:
        drift_signed = 0.0

    # 回路（灰度，160x120，零重调）
    loop = CompLoop(window=WINDOW, **LOOP_CFG)
    thetas, mae_l = [], []
    for g in gray_frames:
        r = loop.step(g)
        thetas.append(r["theta"])
        mae_l.append(r["mae"])
    n_windows = max(1, len(gray_frames) // WINDOW)
    base = loop.finalize(n_windows, labels=None)

    # 逐帧 |Δhue|（弧度）vs 逐帧 |Δtheta| / |ΔL|（按原帧索引对齐）
    dh, dt, dl = [], [], []
    for i in range(1, len(hue_means)):
        if hue_means[i] is None or hue_means[i - 1] is None:
            continue
        dh.append(_circ_dist(hue_means[i], hue_means[i - 1]))
        dt.append(abs(thetas[i] - thetas[i - 1]))
        L0 = np.log(np.maximum(gray_frames[i - 1].astype(np.float32), 1.0))
        L1 = np.log(np.maximum(gray_frames[i].astype(np.float32), 1.0))
        dl.append(float(np.mean(np.abs(L1 - L0))))

    def pear(a, b):
        if len(a) < 3:
            return 0.0
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "frames": len(gray_frames),
        "drift_range_deg": round(drift_range, 4),
        "drift_signed_deg": round(drift_signed, 4),
        "hue_windows": n_windows,
        "r_hue_theta": round(pear(dh, dt), 4),
        "r_hue_lum": round(pear(dh, dl), 4),
        "theta_rise": round(float(thetas[-1] - thetas[0]), 6),
        "theta_end": round(float(thetas[-1]), 6),
        "sc2": base["sc2"], "sc1": base["sc1"],
        "mae_mean": base["mae_mean"],
        "mean_dL": round(float(np.mean(dl)) if dl else 0.0, 6),
    }


# ---------------- 原生分辨率诊断（信息性，不进判据） ----------------
def run_native():
    """R1 流在原生 854x480 上运行：验证能量分档分辨率归一化声明（分档饱和诊断）。"""
    frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = load_video_frames(v, native=True)
        frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    n_windows = max(1, len(frames) // WINDOW)
    loop = CompLoop(window=WINDOW, **LOOP_CFG)
    for g in frames:
        loop.step(g)
    base = loop.finalize(n_windows, labels=None)
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    bands = Counter(c0 for (c0, c1, c2) in loop.sig_trace if c0 is not None)
    return {"frames": len(frames), "size": "854x480",
            "mae_mean": base["mae_mean"], "ratio": round(q4 / q1, 6) if q1 > 0 else 0.0,
            "sc1": base["sc1"], "sc2": base["sc2"], "pin_frac": base["pin_frac"],
            "bands": dict(sorted(bands.items()))}


# ---------------- 主流程 ---------------- 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="30,31")
    ap.add_argument("--tag", default="rst")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--state-dir", default=DEFAULT_STATE)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--native", action="store_true",
                    help="运行原生 854x480 分档饱和诊断（信息性）")
    args = ap.parse_args()

    if args.native:
        t0 = time.time()
        nd = run_native()
        print("R_RST_NATIVE_SIZE=%s" % nd["size"])
        print("R_RST_NATIVE_FRAMES=%d" % nd["frames"])
        print("R_RST_NATIVE_MAE=%.6f" % nd["mae_mean"])
        print("R_RST_NATIVE_RATIO=%.6f" % nd["ratio"])
        print("R_RST_NATIVE_SC1=%d" % nd["sc1"])
        print("R_RST_NATIVE_SC2=%d" % nd["sc2"])
        print("R_RST_NATIVE_PIN=%.4f" % nd["pin_frac"])
        print("R_RST_NATIVE_BANDS=%s" % ",".join(
            "%d:%d" % (k, v) for k, v in nd["bands"].items()))
        print("R_RST_NATIVE_ELAPSED=%.2f" % (time.time() - t0))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    t0 = time.time()

    global _R1_SPANS
    r0_frames, r1_frames, r1_spans = build_streams()
    _R1_SPANS = r1_spans
    STREAMS = {30: r0_frames, 31: r1_frames}

    cfg = {"levels": levels, "tag": args.tag, "size": list(RESIZE),
           "window": WINDOW, "videos": VIDEOS,
           "level_names": LEVEL_NAMES, "loop": LOOP_CFG,
           "persist_points": {str(k): list(v) for k, v in PERSIST_POINTS.items()},
           "r1_spans": [[a, b] for a, b in r1_spans],
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_rst_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_level", {})

    per_level = dict(done)
    for lv in levels:
        key = str(lv)
        if key in per_level:
            continue
        frames = STREAMS[lv]
        points = () if args.no_persist else PERSIST_POINTS.get(lv, ())
        r = run_level(lv, frames, points, args.state_dir,
                      spans=r1_spans if lv == 31 else None)
        per_level[key] = r
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"config": cfg, "per_level": per_level},
                      f, ensure_ascii=False, indent=1)
        print("PROGRESS", flush=True)

    # ---- R2 漂移观测（可选，不进主判定） ----
    drift = run_drift()

    # ---- 判据（预注册冻结，docs/243 §1.4） ----
    stable_ok = int(all(per_level[str(lv)]["ratio"] <= 1.5 for lv in levels))
    struct_ok = int(all(per_level[str(lv)]["sc2"] > 0 for lv in levels))
    n_pts = sum(len(per_level[str(lv)]["persist"]) for lv in levels)
    n_eq = sum(sum(1 for it in per_level[str(lv)]["persist"] if it["seq_eq"] == 1)
               for lv in levels)
    persist_ok = int(n_pts > 0 and n_eq == n_pts)
    bridge_ok = 1
    if 31 in levels:
        br = per_level["31"]["bridge"]
        bridge_ok = int(br["churn"] <= 0.5 and br["bridge_corr_switch"] >= 0.5)

    oks = {"stable": stable_ok, "struct": struct_ok,
           "persist": persist_ok, "bridge": bridge_ok}
    if all(oks.values()):
        verdict = "REAL_PASS"
        vnote = "REAL_STABLE and STRUCT_EMERGES and STATE_PERSIST and BRIDGE_TO_REAL all pass"
    elif not struct_ok:
        verdict = "REAL_FAIL_STRUCT"
        vnote = "STRUCT_EMERGES failed (SC2 not > 0 on real pixels); toy calibration overfit risk reported"
    elif not all(oks.values()):
        verdict = "REAL_PARTIAL"
        failed = [k for k, v in oks.items() if not v]
        vnote = "criterion failed: %s (see numbers)" % ",".join(failed)
    else:
        verdict = "PARTIAL"
        vnote = "criteria not computable; see numbers"

    vdict = {"verdict": verdict, "note": vnote, "oks": oks}

    out = {
        "artifact": "real_stream_test",
        "doc_ref": "docs/241, docs/243",
        "config": cfg,
        "levels": {str(lv): per_level[str(lv)] for lv in levels},
        "drift": drift,
        "verdict": vdict,
        "persist": {"n_points": n_pts, "n_eq": n_eq, "ok": persist_ok},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "rst_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "rst_persist.json"), "w", encoding="utf-8") as f:
        json.dump({"artifact": "real_stream_test_persist", "doc_ref": "docs/243",
                   "config": cfg, "n_points": n_pts, "n_eq": n_eq, "ok": persist_ok,
                   "detail": [{"level": lv, "point": it["point"], "seq_eq": it["seq_eq"],
                               "state_path": it["state_path"]}
                              for lv in levels for it in per_level[str(lv)]["persist"]]},
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "rst_drift.json"), "w", encoding="utf-8") as f:
        json.dump({"artifact": "real_stream_test_drift", "doc_ref": "docs/220, docs/243",
                   "surf": drift}, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_RST_LEVELS=%d" % len(levels))
    print("R_RST_SEEDLESS=1")
    print("R_RST_RESIZE=%dx%d" % RESIZE)
    for i, lv in enumerate(levels):
        r = per_level[str(lv)]
        print("R_RST_%d_NAME=%s" % (i, r["name"]))
        print("R_RST_%d_FRAMES=%d" % (i, r["frames"]))
        print("R_RST_%d_WINDOWS=%d" % (i, r["n_windows"]))
        print("R_RST_%d_MAE=%.6f" % (i, r["mae_mean_win"]))
        print("R_RST_%d_MAE_SD=%.6f" % (i, r["mae_sd_win"]))
        print("R_RST_%d_MAE_LO=%.6f" % (i, r["mae_ci95"][0]))
        print("R_RST_%d_MAE_HI=%.6f" % (i, r["mae_ci95"][1]))
        print("R_RST_%d_Q1=%.6f" % (i, r["mae_q1"]))
        print("R_RST_%d_Q4=%.6f" % (i, r["mae_q4"]))
        print("R_RST_%d_RATIO=%.6f" % (i, r["ratio"]))
        print("R_RST_%d_SC1=%d" % (i, r["sc1"]))
        print("R_RST_%d_SC2=%d" % (i, r["sc2"]))
        print("R_RST_%d_A2=%d" % (i, r["arity2_stable"]))
        print("R_RST_%d_A3=%d" % (i, r["arity3_stable"]))
        print("R_RST_%d_COMP=%.4f" % (i, r["compound_frac"]))
        print("R_RST_%d_CHURN=%.4f" % (i, r["churn_frac"]))
        print("R_RST_%d_PROMO=%d" % (i, r["n_promo"]))
        print("R_RST_%d_SCLATE=%d" % (i, r["sc_late"]))
        print("R_RST_%d_SC4=%d" % (i, r["sc4"]))
        print("R_RST_%d_PIN=%.4f" % (i, r["pin_frac"]))
        print("R_RST_%d_THETA=%.6f" % (i, r["theta_mean"]))
        print("R_RST_%d_PERSIST_POINTS=%d" % (i, len(r["persist"])))
        print("R_RST_%d_PERSIST_EQ=%d" % (i, sum(1 for it in r["persist"] if it["seq_eq"])))
        print("R_RST_%d_PERSIST_OK=%d" % (i, int(all(it["seq_eq"] == 1 for it in r["persist"]))))
        print("R_RST_%d_ENTRIES=%s" % (i, ";".join(
            "%d|%d|%d|%s" % (e["created"], e["arity"], e["hits"],
                             ",".join(str(v) for v in e["key"]))
            for e in r["entry_log"])))
        if r["bridge"] is not None:
            b = r["bridge"]
            print("R_RST_%d_SWITCHES=%d" % (i, b["switches"]))
            print("R_RST_%d_BRIDGE_SW=%.4f" % (i, b["bridge_corr_switch"]))
            print("R_RST_%d_BRIDGE_VID=%.4f" % (i, b["bridge_corr_video"]))
            print("R_RST_%d_SPURIOUS=%d" % (i, b["spurious"]))
            print("R_RST_%d_SPUR_RATE=%.4f" % (i, b["spurious_rate"]))
            print("R_RST_%d_SW_ENC=%s" % (i, ",".join(str(v) for v in b["sw_enc_flags"])))
            print("R_RST_%d_VID_ENC=%s" % (i, ",".join(str(v) for v in b["vid_enc_flags"])))
    print("R_RST_STABLE_OK=%d" % stable_ok)
    print("R_RST_STRUCT_OK=%d" % struct_ok)
    print("R_RST_PERSIST_POINTS=%d" % n_pts)
    print("R_RST_PERSIST_EQ=%d" % n_eq)
    print("R_RST_PERSIST_OK=%d" % persist_ok)
    print("R_RST_BRIDGE_OK=%d" % bridge_ok)
    print("R_RST_VERDICT=%s" % verdict)
    print("R_RST_VERDICT_NOTE=%s" % vnote)
    print("R_RST_DRIFT_FRAMES=%d" % drift["frames"])
    print("R_RST_DRIFT_RANGE_DEG=%.4f" % drift["drift_range_deg"])
    print("R_RST_DRIFT_SIGNED_DEG=%.4f" % drift["drift_signed_deg"])
    print("R_RST_DRIFT_MEAN_DL=%.6f" % drift["mean_dL"])
    print("R_RST_DRIFT_R_HT=%.4f" % drift["r_hue_theta"])
    print("R_RST_DRIFT_R_HL=%.4f" % drift["r_hue_lum"])
    print("R_RST_DRIFT_THETA_RISE=%.6f" % drift["theta_rise"])
    print("R_RST_DRIFT_THETA_END=%.6f" % drift["theta_end"])
    print("R_RST_DRIFT_SC2=%d" % drift["sc2"])
    print("R_RST_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
