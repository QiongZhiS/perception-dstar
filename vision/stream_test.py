"""vision/stream_test.py — 流式基质实验（docs/240 A1 时间，阶段一；交付 docs/241）。

把 compose_test 的 CompLoop 回路从"240 帧批处理"升级为"无限流式"：连续环境流
2400 帧 = 10×240（10 个 chunk），不重置、状态持续累积；chunk 用派生种子重抽
（regime 相位/上下文翻转/噪声 σ 乘性抖动），CompLoop 实例在 chunk 间不重置。

流式级别（预注册，docs/241 §1.2）：
  S0 (lvcode 20, c0_single_regime)：流式 regime 切换（单目标三档速度，段式切换无限循环）
  S1 (lvcode 22, c2_context_dep)：流式上下文依赖（双目标，B 规则依赖 A 相对位置，翻转无限循环）

预注册判据（docs/241 §1.3，冻结；判据先于最终运行写入 docs/241）：
  1. STREAM_STABLE : 每级跨种子 mean(mae 末四分之一窗口均值 / 首四分之一窗口均值) <= 1.5
  2. STRUCT_BOUNDED: 每级 SC2_mean <= 3*SC2_batch（S0:C0=2.4、S1:C2=2.0，logs/ct_main.log
                     R_C0_SC2/R_C2_SC2）且 churn_mean <= 0.3
  3. STATE_PERSIST : 每级 x 10 种子 x 3 中断点（720/1440/1333 帧）save->load->续跑，
                     与不中断运行逐帧全等（per-frame (mae,att,ev,theta,db) + finalize 全等）
  4. BRIDGE        : SC2_mean(S0)>=2.4 且 SC2_mean(S1)>=2.0 且 compound_mean(S1)>=0.5
判定：四条全过 = DYNAMIC_PASS；任一不过 = DYNAMIC_FAIL；无法计算 = PARTIAL。

负结果复测（docs/241 §1.4，附带不设门禁）：在流式长程 run 的匹配事件上重算 SO_info
（so_probe 冻结公式 v2_global_hazard：SO_conf 账本史置信度 -> SO_out 下一窗口命中，
point-biserial r + AUC + 假对照 fake_rand/fake_fixed），报告 SO_RETEST 数字，不进主判定。

统计外壳（docs/228/232 模式）：10 种子（seed 0-9）、jitter 0.25、checkpoint
（ckpt_st_<hash>.json，--resume）、JSON 归档（st_<tag>.json / st_persist.json /
st_so.json）、mean±SD + bootstrap 95% CI（2000 次，种子 20260828）。
安全纪律（docs/228/234 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_ST_* 摘要块；
禁止把日志/JSON 原文送进上下文。禁止修改 compose_test.py / critical_point.py /
so_probe.py 等既有脚本——只 import 复用。

用法：
  python vision/stream_test.py --levels 20,22 --n-seeds 10 --tag main
  python vision/stream_test.py --levels 20,22 --n-seeds 1  --frames 2400 --tag timing
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import mean_sd, bootstrap_ci, JITTER
from compose_test import CompLoop, make_scene, LEVELS, ENERGY_BINS, UPPER_BINS
from so_probe import compute_observations, seed_metrics, window_aligned

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")
DEFAULT_STATE = os.path.join("vision", "out", "state")
N_BOOT = 2000
BOOT_SEED = 20260828

# 回路参数：与 docs/235/237（compose_test.main / so_probe.LOOP_CFG）逐字一致
LOOP_CFG = {"alpha_fast": 0.5, "alpha_slow": 0.03, "thresh": 0.15,
            "deadband": 0.015, "k_theta": 6.0, "k_db": 1.5,
            "thresh_max": 0.6, "db_max": 0.15, "k_consist": 3,
            "hits_min": 3, "persist_win": 5, "k_split": 5, "delta_rel": 0.30,
            "energy_bins": list(ENERGY_BINS), "bbox_bins": list(UPPER_BINS)}

# 批处理参考（docs/241 §1.3 预注册；来源 logs/ct_main.log R_C0_SC2 / R_C2_SC2，docs/235 工件
# ct_main.json 同源：C0 SC2=2.4、C2 SC2=2.0）。判据冻结引用，不允许在看过结果后修改。
BATCH_SC2 = {20: 2.4, 22: 2.0}

# 持久化中断点（帧索引，p=第 p 帧处理完后 save）：p1=720 窗口边界、p2=1440 窗口边界、
# p3=1333 窗口内（133 窗口 + 3 帧）。预注册，冻结。
PERSIST_POINTS = (720, 1440, 1333)


# ---------------- 流式环境流（chunk 拼接，复用 make_scene，不重置回路） ----------------
def stream_frames(lvcode, seed, n_frames, chunk, width, height, fps, jitter):
    """生成连续环境流：n_frames/chunk 个 chunk，第 c 个 chunk 由
    make_scene(lvcode, seed*10+c, n_frames=chunk) 生成（派生种子 -> 每 chunk 新鲜重抽
    regime 相位/上下文翻转/噪声抖动）。返回 (frames, win_labels)（labels 逐 chunk 拼接）。"""
    frames, labels = [], []
    n_chunks = max(1, n_frames // chunk)
    for c in range(n_chunks):
        seed_c = seed * 10 + c
        fr, lb = make_scene(lvcode, seed_c, n_frames=chunk, width=width,
                            height=height, fps=fps, jitter=jitter)
        frames.extend(fr)
        labels.extend(lb)
    return frames, labels


# ---------------- CompLoop 状态 save/load（JSON，float 无损往返） ----------------
def _arr_b64(a):
    return base64.b64encode(np.asarray(a, dtype=np.float32).tobytes()).decode("ascii")


def _arr_from_b64(s, shape):
    if s is None:
        return None
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).copy().reshape(shape)


def _node_record(nd):
    """条目记录 -> JSON 可写 dict（key 转 list、c2_ene 的 None 键转 "null"）。"""
    return {"key": list(nd["key"]),
            "hits": nd["hits"], "created": nd["created"],
            "last_active": nd["last_active"], "arity": nd["arity"],
            "c2_ene": {("null" if c2v is None else str(c2v)): v
                       for c2v, v in nd["c2_ene"].items()},
            "retired": bool(nd["retired"]),
            "retired_at": nd.get("retired_at")}


def _record_to_node(rec):
    """JSON 记录 -> 条目 dict（key 还原为 tuple、c2_ene 的 "null" 键还原为 None）。"""
    key = tuple(int(v) for v in rec["key"])
    c2e = {}
    for k, v in rec["c2_ene"].items():
        c2e[None if k == "null" else int(k)] = list(v)
    nd = {"key": key, "hits": rec["hits"], "created": rec["created"],
          "last_active": rec["last_active"], "arity": rec["arity"],
          "c2_ene": c2e, "retired": rec["retired"]}
    if rec.get("retired_at") is not None:
        nd["retired_at"] = rec["retired_at"]
    return nd


def save_loop_state(loop, shape=(120, 160)):
    """把 CompLoop（含 CPLoop 全部运行态）序列化为 JSON 可写 dict。无损：
    float32 数组走 base64 字节（bit-exact），标量/列表走 JSON float（repr 往返精确）。
    _entry_log 全量记录（含退休父条目——不在 nodes 表内，需独立保存）。"""
    return {
        "version": 1,
        "cfg": {"alpha_fast": loop.a_fast, "alpha_slow": loop.a_slow,
                "thresh": loop.thresh, "deadband": loop.deadband,
                "k_theta": loop.k_theta, "k_db": loop.k_db,
                "thresh_max": loop.thresh_max, "db_max": loop.db_max,
                "window": loop.window, "k_consist": loop.k_consist,
                "hits_min": loop.hits_min, "persist_win": loop.persist_win,
                "k_split": loop.k_split, "delta_rel": loop.delta_rel,
                "energy_bins": list(loop.energy_bins), "bbox_bins": list(loop.bbox_bins)},
        "bg_fast": _arr_b64(loop.bg_fast) if loop.bg_fast is not None else None,
        "bg_slow": _arr_b64(loop.bg_slow) if loop.bg_slow is not None else None,
        "prev_L": _arr_b64(loop.prev_L) if loop.prev_L is not None else None,
        "sigma_hat": loop.sigma_hat,
        "param_disp": loop._param_disp,
        "prev_theta": loop._prev_theta,
        "prev_db": loop._prev_db,
        "n_steps": loop._n_steps,
        "n_learn": loop._n_learn,
        "win": loop._win,
        "ev_win": loop._ev_win.tolist() if loop._ev_win is not None else None,
        "frame_buf": loop._frame_buf,
        "mae": loop.mae, "att": loop.att, "ev": loop.ev,
        "theta_trace": loop.theta_trace, "db_trace": loop.db_trace,
        "sc1_cum": loop.sc1_cum, "energy_trace": loop.energy_trace,
        "nodes": [_node_record(nd) for nd in loop.nodes.values()],
        "entry_log": [_node_record(nd) for nd in loop._entry_log],
        "sig_seen": {",".join(str(v) for v in k): v for k, v in loop._sig_seen.items()},
        "match_trace": [[w, None if k is None else list(k)]
                        for w, k in loop.match_trace],
        "c2_trace": [None if v is None else int(v) for v in loop.c2_trace],
        "sig_trace": [[c0, c1, c2] for (c0, c1, c2) in loop.sig_trace],
        "bbox_trace": loop.bbox_trace, "up_trace": loop.up_trace,
        "lo_trace": loop.lo_trace,
        "hist": [[list(s2), c2, e] for (s2, c2, e) in loop._hist],
        "n_created": loop._n_created, "n_retired": loop._retired,
        "n_promos": loop._promos,
    }


def load_loop_state(loop, st, shape=(120, 160)):
    """把 save_loop_state 的 dict 恢复到 loop 对象（覆盖全部运行态）。"""
    cfg = st["cfg"]
    assert (cfg["window"] == loop.window and cfg["k_consist"] == loop.k_consist
            and cfg["hits_min"] == loop.hits_min and cfg["persist_win"] == loop.persist_win
            and cfg["k_split"] == loop.k_split and cfg["delta_rel"] == loop.delta_rel
            and list(cfg["energy_bins"]) == list(loop.energy_bins)
            and list(cfg["bbox_bins"]) == list(loop.bbox_bins)), "loop cfg mismatch"
    loop.bg_fast = _arr_from_b64(st["bg_fast"], shape)
    loop.bg_slow = _arr_from_b64(st["bg_slow"], shape)
    loop.prev_L = _arr_from_b64(st["prev_L"], shape)
    loop.sigma_hat = st["sigma_hat"]
    loop.thresh = cfg["thresh"]
    loop.deadband = cfg["deadband"]
    loop._param_disp = st["param_disp"]
    loop._prev_theta = st["prev_theta"]
    loop._prev_db = st["prev_db"]
    loop._n_steps = st["n_steps"]
    loop._n_learn = st["n_learn"]
    loop._win = st["win"]
    loop._ev_win = None if st["ev_win"] is None else np.array(st["ev_win"], bool)
    loop._frame_buf = list(st["frame_buf"])
    loop.mae = list(st["mae"])
    loop.att = list(st["att"])
    loop.ev = list(st["ev"])
    loop.theta_trace = list(st["theta_trace"])
    loop.db_trace = list(st["db_trace"])
    loop.sc1_cum = list(st["sc1_cum"])
    loop.energy_trace = list(st["energy_trace"])
    nodes = {}
    for item in st["nodes"]:
        nd = _record_to_node(item)
        nodes[nd["key"]] = nd
    loop.nodes = nodes
    # _entry_log：当前在表内的条目引用 live nodes（后续匹配/退休会继续改它们）；
    # 已退休（不在 nodes）的条目还原为独立记录（退休后不再被改，值即最终值）。
    loop._entry_log = []
    for rec in st["entry_log"]:
        k = tuple(int(v) for v in rec["key"])
        if k in nodes:
            loop._entry_log.append(nodes[k])
        else:
            loop._entry_log.append(_record_to_node(rec))
    loop._sig_seen = {tuple(int(v) for v in k.split(",")): v
                      for k, v in st["sig_seen"].items()}
    loop.match_trace = [(w, None if k is None else tuple(int(v) for v in k))
                        for w, k in st["match_trace"]]
    loop.c2_trace = [None if v is None else int(v) for v in st["c2_trace"]]
    loop.sig_trace = [(c0, c1, c2) for (c0, c1, c2) in st["sig_trace"]]
    loop.bbox_trace = list(st["bbox_trace"])
    loop.up_trace = list(st["up_trace"])
    loop.lo_trace = list(st["lo_trace"])
    loop._hist = [(tuple(int(v) for v in s2), c2, e) for (s2, c2, e) in st["hist"]]
    loop._n_created = st["n_created"]
    loop._retired = st["n_retired"]
    loop._promos = st["n_promos"]
    return loop


# ---------------- 运行一个 (级,种子) 的完整流式 run（主 run + 持久化测试 + SO 复测） ----------------
def run_stream(lvcode, seed, n_frames, chunk, width, height, fps, window, jitter,
               points, state_dir):
    """主 run（2400 帧不中断）+ 3 中断点持久化 round-trip（与主 run 逐帧全等比对）
    + SO 复测（长程观察池）。返回 per-seed dict。"""
    frames, labels = stream_frames(lvcode, seed, n_frames, chunk, width, height,
                                   fps, jitter)
    n_windows = max(1, n_frames // window)
    loop = CompLoop(window=window, **LOOP_CFG)
    main_outs = []
    for g in frames:
        main_outs.append(loop.step(g))
    base = loop.finalize(n_windows, labels)

    # ---- 判据 1 组件：流式稳定性 ratio（末/首四分之一 MAE 窗口均值比） ----
    mae_arr = np.asarray(loop.mae, float)
    q = max(1, n_windows // 4)
    q1 = float(mae_arr[:q].mean()) if len(mae_arr) >= q else 0.0
    q4 = float(mae_arr[-q:].mean()) if len(mae_arr) >= q else 0.0
    stable_ratio = (q4 / q1) if q1 > 0 else 0.0

    # ---- 判据 3：持久化 round-trip（save->load->续跑 vs 不中断主 run 逐帧全等） ----
    # 参考 = 主 run（不中断）：main_outs（per-frame）+ base（finalize）。
    # 中断变体 = 前缀 run [0,p) 全新跑 -> save 状态到磁盘 JSON -> load 重建 ->
    # 续跑 [p,N)；逐帧输出与 finalize 必须与参考全等（SEQ_EQ=1）。
    persist = []
    for p in points:
        if p <= 0 or p >= n_frames:
            continue
        pre = CompLoop(window=window, **LOOP_CFG)
        pre_outs = [pre.step(g) for g in frames[:p]]
        st = save_loop_state(pre)
        path = os.path.join(state_dir, "st_persist_%d_%d_%d.json" % (lvcode, seed, p))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f, sort_keys=True)
        # load（从磁盘 JSON 重建）
        with open(path, encoding="utf-8") as f:
            st2 = json.load(f)
        post = CompLoop(window=window, **LOOP_CFG)
        load_loop_state(post, st2)
        post_outs = [post.step(g) for g in frames[p:]]
        eq = True
        for i in range(p):
            if pre_outs[i] != main_outs[i]:
                eq = False
                break
        if eq:
            for i in range(p, n_frames):
                if post_outs[i - p] != main_outs[i]:
                    eq = False
                    break
        if eq:
            f2 = post.finalize(n_windows, labels)
            if f2 != base:
                eq = False
        persist.append({"point": p, "seq_eq": int(eq), "state_path": path})

    # ---- SO 复测（docs/241 §1.4：长程观察池，so_probe 冻结公式） ----
    matched = window_aligned(loop.match_trace)
    obs = compute_observations(matched, base["entry_log"], n_windows)
    so = seed_metrics(obs, seed, lvcode)

    out = dict(base)
    out.update({
        "seed": seed, "level": lvcode, "frames": n_frames, "chunks": n_frames // chunk,
        "n_windows": n_windows,
        "stable_ratio": round(stable_ratio, 6),
        "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
        "persist": persist,
        "so": so,
        "so_observations": obs,
    })
    return out


# ---------------- 跨种子聚合 + 判据 + 判定（预注册冻结逻辑） ----------------
# 聚合键命名：(per_unit_key, agg_base) —— agg 键 = agg_base + "_mean"/"_sd"。
AGG_FIELDS = (("mae_mean", "mae"), ("sc1", "sc1"), ("sc2", "sc2"),
              ("arity2_stable", "a2"), ("arity3_stable", "a3"),
              ("compound_frac", "compound"), ("churn_frac", "churn"),
              ("n_promo", "promo"), ("ctx_fidelity", "fid"),
              ("ctx_purity", "pur"), ("sc_late", "sc_late"), ("sc4", "sc4"),
              ("stable_ratio", "ratio"))


def level_aggregate(rs, lvcode, seeds):
    def col(k):
        return [r[k] for r in rs]

    agg = {}
    for k, ab in AGG_FIELDS:
        agg[ab + "_mean"], agg[ab + "_sd"] = mean_sd(col(k))
    agg["sc2_max"] = float(np.max(col("sc2")))
    agg["churn_max"] = float(np.max(col("churn_frac")))
    agg["ratio_max"] = float(np.max(col("stable_ratio")))
    agg["mae_ci95"] = list(bootstrap_ci(col("mae_mean")))
    agg["sc2_ci95"] = list(bootstrap_ci(col("sc2")))
    agg["compound_ci95"] = list(bootstrap_ci(col("compound_frac")))
    # SO 复测聚合（每种子 r；bootstrap CI 同 docs/237）
    so_r = [r["so"]["r"] for r in rs]
    so_diff = [r["so"]["r"] - r["so"]["r_fake_rand"] for r in rs]
    agg["so_nobs_mean"] = float(np.mean([r["so"]["n_obs"] for r in rs]))
    agg["so_r_mean"], agg["so_r_sd"] = mean_sd(so_r)
    agg["so_r_ci95"] = list(bootstrap_ci(so_r))
    agg["so_auc_mean"] = float(np.mean([r["so"]["auc"] for r in rs]))
    agg["so_fakerand_mean"] = float(np.mean([r["so"]["r_fake_rand"] for r in rs]))
    agg["so_diff_mean"] = float(np.mean(so_diff))
    agg["so_diff_ci95"] = list(bootstrap_ci(so_diff))
    # 判据（预注册冻结）
    batch = BATCH_SC2.get(lvcode)
    agg["ok_stable"] = int(agg["ratio_mean"] <= 1.5)
    agg["ok_bounded"] = int((batch is not None
                             and agg["sc2_mean"] <= 3.0 * batch
                             and agg["churn_mean"] <= 0.3))
    agg["ok_bridge_sc2"] = int(batch is not None and agg["sc2_mean"] >= batch)
    # BRIDGE 的复合占比 >= 0.5 只对 S1（lvcode 22）要求（预注册 docs/241 §1.3：
    # "且 S1 复合条目占比 >= 0.5"）；S0 是单目标 regime 流，无槽位采纳（docs/235 C0
    # compound=0 是机制基线），不适用 -> 记为 1（不要求）。
    agg["bridge_comp_required"] = int(lvcode == 22)
    agg["ok_bridge_comp"] = int(agg["compound_mean"] >= 0.5) if lvcode == 22 else 1
    agg["ok_bridge"] = int(agg["ok_bridge_sc2"] and agg["ok_bridge_comp"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="20,22")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=2400)
    ap.add_argument("--chunk", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--jitter", type=float, default=JITTER)
    ap.add_argument("--points", default="720,1440,1333")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--state-dir", default=DEFAULT_STATE)
    ap.add_argument("--tag", default="st")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    points = [int(x) for x in args.points.split(",") if x.strip() != ""]
    t0 = time.time()

    cfg = {"levels": levels, "n_seeds": args.n_seeds, "first_seed": args.first_seed,
           "frames": args.frames, "chunk": args.chunk, "fps": args.fps,
           "size": [args.width, args.height], "window": args.window,
           "jitter": args.jitter, "points": points, "tag": args.tag,
           "scene": {str(k): v for k, v in LEVELS.items()},
           "loop": LOOP_CFG,
           "batch_sc2": BATCH_SC2,
           "so": {"formula": "v2_global_hazard"}}
    ck_tag = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    ckpt_path = os.path.join(args.out_dir, "ckpt_st_%s.json" % ck_tag)

    done = {}
    if args.resume and not args.no_resume and os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            done = json.load(f).get("per_unit", {})

    per_unit = dict(done)
    for lv in levels:
        for seed in seeds:
            key = "%d_%d" % (lv, seed)
            if key in per_unit:
                continue
            r = run_stream(lv, seed, args.frames, args.chunk, args.width,
                           args.height, args.fps, args.window, args.jitter,
                           points, args.state_dir)
            per_unit[key] = r
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "per_unit": per_unit},
                          f, ensure_ascii=False, indent=1)
            print("PROGRESS", flush=True)

    # ---- 跨种子聚合 + 判据 ----
    levels_out = {}
    for lv in levels:
        rs = [per_unit["%d_%d" % (lv, s)] for s in seeds]
        agg = level_aggregate(rs, lv, seeds)
        levels_out[lv] = {"name": LEVELS[lv]["name"], "per_seed": rs,
                          "mean_sd": agg}
        for k in ("mae_ci95", "sc2_ci95", "compound_ci95", "so_r_ci95",
                  "so_diff_ci95"):
            levels_out[lv]["mean_sd"][k] = agg[k]

    # ---- 判据 3 全局（STATE_PERSIST）：全部测试点 SEQ_EQ=1 ----
    n_points_total = 0
    n_points_eq = 0
    persist_detail = []
    for lv in levels:
        for seed in seeds:
            for item in per_unit["%d_%d" % (lv, seed)]["persist"]:
                n_points_total += 1
                n_points_eq += item["seq_eq"]
                persist_detail.append({"level": lv, "seed": seed,
                                       "point": item["point"],
                                       "seq_eq": item["seq_eq"],
                                       "state_path": item["state_path"]})
    persist_ok = int(n_points_total > 0 and n_points_eq == n_points_total)

    # ---- 四条判据聚合（每级都过才计 OK） ----
    stable_ok = int(all(levels_out[lv]["mean_sd"]["ok_stable"] == 1 for lv in levels))
    bounded_ok = int(all(levels_out[lv]["mean_sd"]["ok_bounded"] == 1 for lv in levels))
    bridge_ok = int(all(levels_out[lv]["mean_sd"]["ok_bridge"] == 1 for lv in levels))

    oks = {"stable": stable_ok, "bounded": bounded_ok,
           "persist": persist_ok, "bridge": bridge_ok}
    if all(oks.values()):
        verdict = "DYNAMIC_PASS"
        vnote = "STREAM_STABLE and STRUCT_BOUNDED and STATE_PERSIST and BRIDGE all pass"
    elif any(oks.values()) or not all(oks.values()):
        failed = [k for k, v in oks.items() if not v]
        verdict = "DYNAMIC_FAIL"
        vnote = "criterion failed: %s (see numbers)" % ",".join(failed)
    else:
        verdict = "PARTIAL"
        vnote = "criteria not computable; see numbers"

    vdict = {"verdict": verdict, "note": vnote,
             "stable_ok": stable_ok, "bounded_ok": bounded_ok,
             "persist_ok": persist_ok, "bridge_ok": bridge_ok,
             "batch_sc2": BATCH_SC2}

    out = {
        "artifact": "stream_test",
        "doc_ref": "docs/240, docs/241",
        "config": cfg,
        "levels": {str(k): v for k, v in levels_out.items()},
        "verdict": vdict,
        "persist": {"n_points": n_points_total, "n_eq": n_points_eq,
                    "ok": persist_ok, "detail": persist_detail},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "st_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 持久化 / SO 复测专项归档 ----
    pers_path = os.path.join(args.out_dir, "st_persist.json")
    with open(pers_path, "w", encoding="utf-8") as f:
        json.dump({"artifact": "stream_test_persist", "doc_ref": "docs/241",
                   "config": cfg, "n_points": n_points_total, "n_eq": n_points_eq,
                   "ok": persist_ok, "detail": persist_detail},
                  f, ensure_ascii=False, indent=1)
    so_out = {"artifact": "stream_test_so_retest", "doc_ref": "docs/237, docs/241",
              "formula": "v2_global_hazard",
              "levels": {str(lv): {"name": levels_out[lv]["name"],
                                   "per_seed_r": [r["so"]["r"] for r in levels_out[lv]["per_seed"]],
                                   "r_mean": levels_out[lv]["mean_sd"]["so_r_mean"],
                                   "r_sd": levels_out[lv]["mean_sd"]["so_r_sd"],
                                   "r_ci95": levels_out[lv]["mean_sd"]["so_r_ci95"],
                                   "auc_mean": levels_out[lv]["mean_sd"]["so_auc_mean"],
                                   "fakerand_mean": levels_out[lv]["mean_sd"]["so_fakerand_mean"],
                                   "diff_mean": levels_out[lv]["mean_sd"]["so_diff_mean"],
                                   "diff_ci95": levels_out[lv]["mean_sd"]["so_diff_ci95"],
                                   "n_obs_mean": levels_out[lv]["mean_sd"]["so_nobs_mean"]}
                         for lv in levels}}
    so_path = os.path.join(args.out_dir, "st_so.json")
    with open(so_path, "w", encoding="utf-8") as f:
        json.dump(so_out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_ST_LEVELS=%d" % len(levels))
    print("R_ST_SEEDS=%d" % len(seeds))
    print("R_ST_FRAMES=%d" % args.frames)
    print("R_ST_WINDOWS=%d" % (args.frames // args.window))
    print("R_ST_CHUNKS=%d" % (args.frames // args.chunk))
    for i, lv in enumerate(levels):
        lo = levels_out[lv]
        ms = lo["mean_sd"]
        print("R_ST_%d_NAME=%s" % (i, lo["name"]))
        print("R_ST_%d_MAE=%.6f" % (i, ms["mae_mean"]))
        print("R_ST_%d_MAE_SD=%.6f" % (i, ms["mae_sd"]))
        print("R_ST_%d_MAE_LO=%.6f" % (i, ms["mae_ci95"][0]))
        print("R_ST_%d_MAE_HI=%.6f" % (i, ms["mae_ci95"][1]))
        print("R_ST_%d_RATIO=%.6f" % (i, ms["ratio_mean"]))
        print("R_ST_%d_RATIO_SD=%.6f" % (i, ms["ratio_sd"]))
        print("R_ST_%d_RATIO_MAX=%.6f" % (i, ms["ratio_max"]))
        print("R_ST_%d_Q1=%.6f" % (i, float(np.mean([r["mae_q1"] for r in lo["per_seed"]]))))
        print("R_ST_%d_Q4=%.6f" % (i, float(np.mean([r["mae_q4"] for r in lo["per_seed"]]))))
        print("R_ST_%d_SC1=%.4f" % (i, ms["sc1_mean"]))
        print("R_ST_%d_SC2=%.4f" % (i, ms["sc2_mean"]))
        print("R_ST_%d_SC2_SD=%.4f" % (i, ms["sc2_sd"]))
        print("R_ST_%d_SC2_MAX=%.4f" % (i, ms["sc2_max"]))
        print("R_ST_%d_SC2_LO=%.4f" % (i, ms["sc2_ci95"][0]))
        print("R_ST_%d_SC2_HI=%.4f" % (i, ms["sc2_ci95"][1]))
        print("R_ST_%d_A2=%.4f" % (i, ms["a2_mean"]))
        print("R_ST_%d_A3=%.4f" % (i, ms["a3_mean"]))
        print("R_ST_%d_COMP=%.4f" % (i, ms["compound_mean"]))
        print("R_ST_%d_COMP_SD=%.4f" % (i, ms["compound_sd"]))
        print("R_ST_%d_COMP_LO=%.4f" % (i, ms["compound_ci95"][0]))
        print("R_ST_%d_COMP_HI=%.4f" % (i, ms["compound_ci95"][1]))
        print("R_ST_%d_CHURN=%.4f" % (i, ms["churn_mean"]))
        print("R_ST_%d_CHURN_MAX=%.4f" % (i, ms["churn_max"]))
        print("R_ST_%d_PROMO=%.4f" % (i, ms["promo_mean"]))
        print("R_ST_%d_CTXFID=%.4f" % (i, ms["fid_mean"]))
        print("R_ST_%d_CTXPUR=%.4f" % (i, ms["pur_mean"]))
        print("R_ST_%d_SCLATE=%.4f" % (i, ms["sc_late_mean"]))
        print("R_ST_%d_SC4=%.4f" % (i, ms["sc4_mean"]))
        print("R_ST_%d_OK_STABLE=%d" % (i, ms["ok_stable"]))
        print("R_ST_%d_OK_BOUNDED=%d" % (i, ms["ok_bounded"]))
        print("R_ST_%d_OK_BRIDGE=%d" % (i, ms["ok_bridge"]))
        print("R_ST_%d_OK_BRIDGE_SC2=%d" % (i, ms["ok_bridge_sc2"]))
        print("R_ST_%d_OK_BRIDGE_COMP=%d" % (i, ms["ok_bridge_comp"]))
        print("R_ST_%d_BRIDGE_COMP_REQ=%d" % (i, ms["bridge_comp_required"]))
    print("R_ST_PERSIST_POINTS=%d" % n_points_total)
    print("R_ST_PERSIST_EQ=%d" % n_points_eq)
    print("R_ST_PERSIST_OK=%d" % persist_ok)
    for i, lv in enumerate(levels):
        ms = levels_out[lv]["mean_sd"]
        print("R_ST_SO_%d_NOBS=%.1f" % (i, ms["so_nobs_mean"]))
        print("R_ST_SO_%d_R=%.6f" % (i, ms["so_r_mean"]))
        print("R_ST_SO_%d_R_SD=%.6f" % (i, ms["so_r_sd"]))
        print("R_ST_SO_%d_R_LO=%.6f" % (i, ms["so_r_ci95"][0]))
        print("R_ST_SO_%d_R_HI=%.6f" % (i, ms["so_r_ci95"][1]))
        print("R_ST_SO_%d_AUC=%.6f" % (i, ms["so_auc_mean"]))
        print("R_ST_SO_%d_FAKERAND=%.6f" % (i, ms["so_fakerand_mean"]))
        print("R_ST_SO_%d_DIFF=%.6f" % (i, ms["so_diff_mean"]))
        print("R_ST_SO_%d_DIFF_LO=%.6f" % (i, ms["so_diff_ci95"][0]))
        print("R_ST_SO_%d_DIFF_HI=%.6f" % (i, ms["so_diff_ci95"][1]))
    v = out["verdict"]
    print("R_ST_STABLE_OK=%d" % stable_ok)
    print("R_ST_BOUNDED_OK=%d" % bounded_ok)
    print("R_ST_PERSIST_OK=%d" % persist_ok)
    print("R_ST_BRIDGE_OK=%d" % bridge_ok)
    print("R_ST_VERDICT=%s" % verdict)
    print("R_ST_VERDICT_NOTE=%s" % vnote)
    print("R_ST_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
