"""vision/cross_domain_test.py — docs/248 L3 跨域泛化：DAVIS 校准软匹配工作点零重调迁移到
真实野视频域（docs/247 泛化层级规范第一次执行）。

预注册（docs/248 §一，冻结；判据与视频选择先于任何运行写入 docs/248，docs/63+247 纪律）：
  数据域：C:\\Users\\fa278\\Downloads 的 B 站野视频（Get-ChildItem 候选 + 探针元数据，
    预注册选择 V1 studio 人物/说话 / V2 B 站用户内容 / V3 快速剪辑梗短片）。
  预处理：cv2 逐帧解码 + 间隔抽帧（step=max(1,round(T/500))，目标每视频 ~400-600 采样帧）
    -> 灰度 -> resize 160x120 (INTER_AREA) -> uint8 -> 流式输入（window=10 帧/窗）。
  工作点（零重调，被测对象本身）：r = 1.5 x 0.2659 = 0.39885（r_base=0.2659 为 DAVIS 校准
    值，docs/245 §3.1；docs/246 M=1.5 最小/最优工作点），k_consist=3，alpha=0.2，
    hits_min=3，persist_win=5，LOOP_CFG 原样；机制 = soft_match_test.SoftLoop import 复用。
  流：S1 = V1 单视频；S2 = V2 单视频；S3 = V3 单视频；S4 = V1+V2+V3 拼接（复用采样帧）。
  判据（每流，冻结）：
    1. [L3][参数][跨域零重调] STABLE : ratio = mean(MAE 末四分之一)/mean(MAE 首四分之一) <= 1.5
    2. [L3][参数] STRUCT            : SC2 > 0（hits >= hits_min=3 的稳定原型 >= 1）
    3. [L3][参数] CHURN             : churn <= 0.5（最终 hits < hits_min 原型数 / max(1,SC1)）
    4. [L3][诊断] SCENE_SWITCH      : 帧差尖峰近似场景切 vs 新原型创建时点对应率
      （无 GT，诊断级不进主判定；尖峰 = d_i > mu+3sigma；对应 = |w_c - w_s| <= 1）
  判定：1-3 全过 = L3_PASS；STRUCT 任一流 SC2=0 或 STABLE 任一流 ratio>2.5 = L3_FAIL；
    其余 = L3_PARTIAL（报告哪条失败）；数据不可用 = L3_BLOCKED。
  回归守卫（不进判据）：DAVIS R0+R1 同代码路径 r=0.39885/k=3/alpha=0.2 须复现 docs/246
    M=1.5 工作点行（R0 SC2=3/churn 0/ratio 0.907701；R1 SC1=11/SC2=6/churn 0.4545/
    ratio 0.951261；bridge_sw 0.8750/calib_sw 1.0/holdout_sw 0.75/bridge_vid 0.8889/
    spurious 0；容差 1e-4）-> R_L3_GUARD_D246。
  信息性诊断（不进判据）：R_L3_NN_MED_WILD = S4 参与窗口最近邻距离中位数（对数 (E,U) 域）。

安全纪律（docs/243-246 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L3_* 摘要块；
运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）抽取；
禁止读日志/JSON 原文；Downloads 视频是数据（只读帧数/文件名，cross_domain_probe.py）。
禁止修改任何既有脚本——只 import 复用（soft_match_test / real_stream_test / real_recalib /
stream_test / critical_point）。

用法：
  python vision/cross_domain_test.py --tag cdt
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import cv2

from critical_point import mean_sd, bootstrap_ci
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, WINDOW, RESIZE, LEVEL_NAMES
from real_recalib import bridge_metrics
from soft_match_test import SoftLoop, ALPHA, HITS_MIN

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
DL_DIR = r"C:\Users\fa278\Downloads"

# 工作点（预注册，docs/248 §1.4，冻结；零重调）
R_BASE_DAVIS = 0.2659          # DAVIS 校准集最近邻距离中位数（docs/245 §3.1）
RADIUS_L3 = 1.5 * R_BASE_DAVIS  # = 0.39885（docs/246 M=1.5 最小/最优工作点）
TARGET_FRAMES = 500            # 每视频目标采样帧数（预注册 §1.3）

# 野域视频选择（预注册，docs/248 §1.2，冻结；Get-ChildItem 候选 + 探针元数据）
WILD_VIDEOS = [
    ("V1", "studio_video_1759283839728.mp4"),
    ("V2", "41125413122-1-192.mp4"),
    ("V3", "\u5343\u519b\u4e07\u9a6c\u54e6\u54e6\u54e6.mp4"),  # 千军万马哦哦哦.mp4
]

# 流定义（预注册，docs/248 §1.5，冻结）：(id, ascii_name, [video 索引])
STREAMS = [
    ("S1", "single_v1_talk", [0]),
    ("S2", "single_v2_wild", [1]),
    ("S3", "single_v3_cuts", [2]),
    ("S4", "concat_all", [0, 1, 2]),
]

# docs/246 M=1.5 工作点回归守卫期望（冻结，docs/246 §3.2/§3.6）
D246 = {"r0_sc2": 3, "r0_churn": 0.0, "r0_ratio": 0.907701,
        "r1_sc1": 11, "r1_sc2": 6, "r1_churn": 0.4545, "r1_ratio": 0.951261,
        "bridge_sw": 0.8750, "calib_sw": 1.0, "holdout_sw": 0.7500,
        "bridge_vid": 0.8889, "spurious": 0}


# ---------------- 野域数据加载（顺序解码 + 间隔抽帧；确定性） ----------------
def load_sampled_frames(path, target=TARGET_FRAMES):
    """读一个野视频的全部帧（顺序解码），每 step 帧取一（step = max(1, round(T/500))），
    灰度 + resize 160x120 -> uint8。返回 (frames, step, total_frames)。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video: %s" % path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        # 容器未报告帧数：全部读入后二次采样（罕见回退）
        allf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, RESIZE, interpolation=cv2.INTER_AREA)
            allf.append(np.ascontiguousarray(g, dtype=np.uint8))
        cap.release()
        step2 = max(1, round(len(allf) / target))
        return allf[::step2], step2, len(allf)
    step = max(1, round(n / target))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, RESIZE, interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(g, dtype=np.uint8))
        idx += 1
    cap.release()
    return frames, step, n


# ---------------- 单流软匹配运行（SoftLoop 复用；零重调） ----------------
def run_soft(frames, radius, alpha):
    loop = SoftLoop(radius=radius, alpha=alpha, window=WINDOW, **LOOP_CFG)
    for g in frames:
        loop.step(g)
    base = loop.finalize(max(1, len(frames) // WINDOW), labels=None)
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, len(mae_arr) // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    ratio = (q4 / q1) if q1 > 0 else 0.0
    mae_m, mae_sd = mean_sd(list(mae_arr))
    mae_lo, mae_hi = bootstrap_ci(list(mae_arr))
    E = np.asarray(loop.energy_trace, float)
    n_valid = int(np.sum(E >= 10))
    out = dict(base)
    out.update({"frames": len(frames), "n_windows": len(mae_arr), "n_valid": n_valid,
                "mae_mean_win": round(mae_m, 6), "mae_sd_win": round(mae_sd, 6),
                "mae_ci95": [round(mae_lo, 6), round(mae_hi, 6)],
                "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
                "ratio": round(ratio, 6), "pin_frac": base["pin_frac"],
                "theta_mean": base["theta_mean"]})
    return out, loop


# ---------------- SCENE_SWITCH 诊断（无 GT，近似；诊断级不进主判定） ----------------
def scene_switch_diag(frames, creations):
    """d_i = mean|f_i - f_{i-1}|（采样灰度帧，与回路输入一致）；尖峰 = d_i > mu+3sigma；
    切点窗口 w_s = i//10；对应 |w_c - w_s| <= 1。返回诊断 dict（无判据）。"""
    diffs = []
    for i in range(1, len(frames)):
        a = frames[i - 1].astype(np.float32)
        b = frames[i].astype(np.float32)
        diffs.append(float(np.mean(np.abs(b - a))))
    arr = np.asarray(diffs, float)
    mu = float(arr.mean()) if len(arr) else 0.0
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    th = mu + 3.0 * sd
    spike_frames = [i for i, d in enumerate(arr, start=1) if d > th]
    spike_wins = sorted(set(i // WINDOW for i in spike_frames))
    align = [int(any(abs(wc - ws) <= 1 for ws in spike_wins)) for wc in creations]
    corr = (sum(align) / max(1, len(creations))) if creations else None
    cover = (sum(1 for ws in spike_wins
                  if any(abs(wc - ws) <= 1 for wc in creations)) / max(1, len(spike_wins))
             ) if spike_wins else None
    return {"diffs_mean": round(mu, 6), "diffs_sd": round(sd, 6),
            "thresh": round(th, 6), "n_spikes": len(spike_wins),
            "spike_windows": spike_wins, "n_creations": len(creations),
            "creation_windows": creations,
            "switch_corr": (round(corr, 4) if corr is not None else None),
            "spike_coverage": (round(cover, 4) if cover is not None else None)}


# ---------------- 信息性诊断：野域最近邻距离中位数（对数 (E,U) 域；不重调 r） ----------------
def nn_median_wild(loop):
    E = np.asarray(loop.energy_trace, float)
    U = np.asarray(loop.up_trace, float)
    xs = [(float(np.log1p(E[i])), float(np.log1p(U[i])))
          for i in range(len(E)) if E[i] >= 10]
    if len(xs) >= 4:
        ds = [min(float(np.hypot(xs[i][0] - xs[j][0], xs[i][1] - xs[j][1]))
                  for j in range(len(xs)) if j != i)
              for i in range(len(xs))]
        return round(float(np.median(ds)), 4), len(xs), \
            round(float(np.min(ds)), 4), round(float(np.max(ds)), 4)
    return -1.0, len(xs), -1.0, -1.0


# ---------------- 回归守卫（DAVIS R0/R1，docs/246 M=1.5 复现；不进判据） ----------------
def guard_vs_d246(r0r, r1r):
    b = r1r["bridge"]
    items = [
        ("r0_sc2", r0r["sc2"] == D246["r0_sc2"]),
        ("r0_churn", abs(r0r["churn_frac"] - D246["r0_churn"]) < 1e-4),
        ("r0_ratio", abs(r0r["ratio"] - D246["r0_ratio"]) < 1e-4),
        ("r1_sc1", r1r["sc1"] == D246["r1_sc1"]),
        ("r1_sc2", r1r["sc2"] == D246["r1_sc2"]),
        ("r1_churn", abs(r1r["churn_frac"] - D246["r1_churn"]) < 1e-4),
        ("r1_ratio", abs(r1r["ratio"] - D246["r1_ratio"]) < 1e-4),
        ("bridge_sw", abs(b["bridge_corr_switch"] - D246["bridge_sw"]) < 1e-4),
        ("calib_sw", abs(b["calib_switch"] - D246["calib_sw"]) < 1e-4),
        ("holdout_sw", abs(b["holdout_switch"] - D246["holdout_sw"]) < 1e-4),
        ("bridge_vid", abs(b["bridge_corr_video"] - D246["bridge_vid"]) < 1e-4),
        ("spurious", b["spurious"] == D246["spurious"]),
    ]
    ok = all(v for _, v in items)
    detail = ",".join("%s:%d" % (name, int(flag)) for name, flag in items)
    return int(ok), detail


def run_guard(radius):
    """DAVIS 流（R0 flamingo x5；R1 9 视频拼接）同代码路径运行，返回 (r0r, r1r, detail)。"""
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    r0r, _ = run_soft(r0_frames, radius, ALPHA)
    r1r, _ = run_soft(r1_frames, radius, ALPHA)
    r1r["bridge"] = bridge_metrics(r1r, spans)
    return r0r, r1r


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cdt")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 野域数据（预注册选择；间隔抽帧 -> 灰度 160x120 -> 流） ----
    loaded = {}
    steps = {}
    totals = {}
    for vid, name in WILD_VIDEOS:
        p = os.path.join(DL_DIR, name)
        frames, step, total = load_sampled_frames(p)
        loaded[vid] = frames
        steps[vid] = step
        totals[vid] = total

    # ---- 流（预注册 §1.5，冻结） ----
    streams_out = {}
    for sid, sname, vidx in STREAMS:
        frames = []
        for vi in vidx:
            frames.extend(loaded[WILD_VIDEOS[vi][0]])
        out, loop = run_soft(frames, RADIUS_L3, ALPHA)
        creations = [e["created"] for e in out["entry_log"]]
        diag = scene_switch_diag(frames, creations)
        # S4 构造性视频边界（帧索引 = 各段累计采样长度）是否被尖峰检出（参考）
        bound_spikes = None
        if len(vidx) > 1:
            cum = []
            acc = 0
            for vi in vidx:
                acc += len(loaded[WILD_VIDEOS[vi][0]])
                cum.append(acc)
            bound_spikes = [int(any(abs(c // WINDOW - ws) <= 1 for ws in diag["spike_windows"]))
                            for c in cum[:-1]]
        out["stream_id"] = sid
        out["stream_name"] = sname
        out["videos"] = [WILD_VIDEOS[vi][0] for vi in vidx]
        out["step"] = steps[WILD_VIDEOS[vidx[0]][0]] if len(vidx) == 1 else \
            [steps[WILD_VIDEOS[vi][0]] for vi in vidx]
        out["total_frames_src"] = [totals[WILD_VIDEOS[vi][0]] for vi in vidx]
        out["switch_diag"] = diag
        out["boundary_spike_flags"] = bound_spikes
        streams_out[sid] = out

    # ---- 信息性诊断：野域最近邻距离中位数（S4） ----
    s4_loop_frames = []
    for vi in [0, 1, 2]:
        s4_loop_frames.extend(loaded[WILD_VIDEOS[vi][0]])
    _s4r, s4_loop = run_soft(s4_loop_frames, RADIUS_L3, ALPHA)
    nn_wild, nn_valid, nn_min, nn_max = nn_median_wild(s4_loop)

    # ---- 回归守卫（DAVIS；不进判据） ----
    g0, g1 = run_guard(RADIUS_L3)
    guard_ok, guard_detail = guard_vs_d246(g0, g1)

    # ---- 判据（预注册 §1.6，冻结）+ 判定（§1.6 冻结规则） ----
    stable_ok = int(all(streams_out[s]["ratio"] <= 1.5 for s, _, _ in STREAMS))
    struct_ok = int(all(streams_out[s]["sc2"] > 0 for s, _, _ in STREAMS))
    churn_ok = int(all(streams_out[s]["churn_frac"] <= 0.5 for s, _, _ in STREAMS))
    fail_struct = any(streams_out[s]["sc2"] == 0 for s, _, _ in STREAMS)
    fail_stable_blow = any(streams_out[s]["ratio"] > 2.5 for s, _, _ in STREAMS)
    fail_stable_mild = any(1.5 < streams_out[s]["ratio"] <= 2.5 for s, _, _ in STREAMS)
    fail_churn = any(streams_out[s]["churn_frac"] > 0.5 for s, _, _ in STREAMS)
    if stable_ok and struct_ok and churn_ok:
        verdict = "L3_PASS"
        vnote = "[L3] STABLE and STRUCT and CHURN all pass on all wild streams (DAVIS working point transfers)"
    elif fail_struct or fail_stable_blow:
        verdict = "L3_FAIL"
        why = []
        if fail_struct:
            why.append("STRUCT: SC2=0 on some wild stream (structure fails to emerge)")
        if fail_stable_blow:
            why.append("STABLE: ratio>2.5 on some wild stream (error blows up)")
        vnote = "; ".join(why) + " (see numbers)"
    elif fail_stable_mild or fail_churn:
        verdict = "L3_PARTIAL"
        why = []
        if fail_stable_mild:
            why.append("STABLE: 1.5<ratio<=2.5 on some wild stream")
        if fail_churn:
            why.append("CHURN: churn>0.5 on some wild stream")
        vnote = "; ".join(why) + " (see numbers)"
    else:
        verdict = "L3_BLOCKED"
        vnote = "criteria not computable (see numbers)"

    oks = {"stable": stable_ok, "struct": struct_ok, "churn": churn_ok}

    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "dl_dir": DL_DIR, "wild_videos": list(WILD_VIDEOS),
           "streams": [(s, n, v) for s, n, v in STREAMS],
           "working_point": {"radius": round(RADIUS_L3, 6),
                             "r_base_davis": R_BASE_DAVIS, "k_consist": 3,
                             "alpha": ALPHA, "hits_min": HITS_MIN,
                             "window": WINDOW},
           "loop": LOOP_CFG,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "cross_domain_test",
        "doc_ref": "docs/243, docs/244, docs/245, docs/246, docs/247, docs/248",
        "config": cfg,
        "streams": streams_out,
        "nn_wild": {"median": nn_wild, "n_valid": nn_valid,
                    "min": nn_min, "max": nn_max},
        "criteria": oks,
        "verdict": {"verdict": verdict, "note": vnote},
        "guard_d246": {"ok": guard_ok, "detail": guard_detail},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "%s_%s.json" % ("cdt", args.tag))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_L3_RADIUS=%.4f" % RADIUS_L3)
    print("R_L3_R_BASE=%.4f" % R_BASE_DAVIS)
    print("R_L3_K=%d" % 3)
    print("R_L3_ALPHA=%.4f" % ALPHA)
    print("R_L3_N_STREAMS=%d" % len(STREAMS))
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        d = r["switch_diag"]
        print("R_L3_%d_ID=%s" % (j, sid))
        print("R_L3_%d_NAME=%s" % (j, sname))
        print("R_L3_%d_VIDEOS=%s" % (j, ",".join(r["videos"])))
        print("R_L3_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_L3_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_L3_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_L3_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_L3_%d_MAE_SD=%.6f" % (j, r["mae_sd_win"]))
        print("R_L3_%d_MAE_LO=%.6f" % (j, r["mae_ci95"][0]))
        print("R_L3_%d_MAE_HI=%.6f" % (j, r["mae_ci95"][1]))
        print("R_L3_%d_Q1=%.6f" % (j, r["mae_q1"]))
        print("R_L3_%d_Q4=%.6f" % (j, r["mae_q4"]))
        print("R_L3_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_L3_%d_SC1=%d" % (j, r["sc1"]))
        print("R_L3_%d_SC2=%d" % (j, r["sc2"]))
        print("R_L3_%d_CHURN=%.4f" % (j, r["churn_frac"]))
        print("R_L3_%d_PIN=%.4f" % (j, r["pin_frac"]))
        print("R_L3_%d_THETA=%.6f" % (j, r["theta_mean"]))
        print("R_L3_%d_CREATIONS=%s" % (j, ",".join(str(w) for w in d["creation_windows"])))
        print("R_L3_%d_SPIKES=%s" % (j, ",".join(str(w) for w in d["spike_windows"])))
        print("R_L3_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                          else "%.4f" % d["switch_corr"])))
        print("R_L3_%d_SW_COVER=%s" % (j, ("NA" if d["spike_coverage"] is None
                                           else "%.4f" % d["spike_coverage"])))
        if r["boundary_spike_flags"] is not None:
            print("R_L3_%d_BOUND_SPIKES=%s" % (j, ",".join(str(v) for v in r["boundary_spike_flags"])))
    print("R_L3_NN_MED_WILD=%.4f" % nn_wild)
    print("R_L3_NN_VALID=%d" % nn_valid)
    print("R_L3_NN_MIN=%.4f" % nn_min)
    print("R_L3_NN_MAX=%.4f" % nn_max)
    print("R_L3_STABLE_OK=%d" % stable_ok)
    print("R_L3_STRUCT_OK=%d" % struct_ok)
    print("R_L3_CHURN_OK=%d" % churn_ok)
    print("R_L3_VERDICT=%s" % verdict)
    print("R_L3_VERDICT_NOTE=%s" % vnote)
    print("R_L3_GUARD_D246=%d" % guard_ok)
    print("R_L3_GUARD_DETAIL=%s" % guard_detail)
    print("R_L3_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
