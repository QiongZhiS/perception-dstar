"""vision/l4_compose_test5.py — docs/264 L4 组合泛化第五格：组合留出（SlotLoop4 机制不变
+ 段级组合留出测量）。

docs/264 §一 冻结，运行后不改：
  组合 = 槽位-内容条件记忆单元（机制产物 = 打标慢原型 = (c2 侧, 特征簇) 组合记忆）。
  组合留出 = 机制不变 + 测量留出：SlotLoop4（docs/262 逐字，import 复用，全部旋钮
  零重调，无新数值旋钮）在 R1 上单次确定性 pass，9 段按预注册顺序规则划分：
  校准段 C = 段 0-4（flamingo/surf/bear/camel/dog，帧 [0,367)，窗 [0,36]）、
  留出段 H = 段 5-8（blackswan/car-turn/motorbike/soccerball，帧 [367,588)，
  窗 [37,57]）。留出段全部位于校准段之后（流序防循环：留出段窗口到达时的机制记忆 =
  纯校准形成；PREFIX_EQ 构造性证明 C 流 ≡ T 流校准前缀逐位一致 + 原型种群哈希一致）。
  参数零重调（旋钮不依任何数据）、单次确定性 pass、无重训、无回调。

流（§1.2 冻结）：
  T（测试流）= 全 R1 两阶段 step（先帧 [0,367) → 快照 → 再帧 [367,588) → finalize）；
  G（守卫流）= 全 R1 单阶段 step（= docs/262 R1 Mode ON 完全同构，R_L4E_REPRO_D262
    逐位复现 docs/262 §三）；
  C（校准流）= 仅段 0-4 = 367 帧（防循环对照；终态种群 P_C 供 M9 静态覆盖评估）；
  R0 = flamingo×5（诊断）、R0b = bear×5（负对照）、S1-S4 = 野流（支持 + 守卫）。

度量（§1.3 冻结）：M1-M7 与 docs/262 逐字一致 + M8 组合留出归因（transfer_tagged_hit /
self_tagged_hit / any_tagged_hit / seen_baseline（校准段 w>=8 eligible 打标命中率）/
transfer_tagged_hit_rate(H) / tagged_hit_rate(H) / self_form_rate / calib_share）+
M9 静态冻结记忆覆盖（P_C 对留出段窗口的 r_slow 覆盖，诊断级）。

判据（§1.4 冻结）：
  1. [L4][机制][组合留出] HELDOUT_TRANSFER：transfer_tagged_hit_rate(H) >= 0.10
     且 >= 0.5 x seen_baseline（校准形成的条件记忆在未见过组合上仍被门控使用）；
  2. [L4][机制][组合留出] STRUCTURE_KEEP：tagged_hit_rate(H) >= 0.5 x seen_baseline
     （未见段上按槽位组织的条件记忆形态仍工作，含自形成记忆）；
  3. [L4][机制] FOUNDATION_KEEP：T 流（全 R1）+ S1-S4 ratio <= 1.5 且 SC2_slow > 0；
     R1 gist_cov >= 0.5（docs/262 判据 3 同款；机制零改动 → 期望逐位一致）；
  4. [L4][机制][行为证据] PROMOTION_KEEP：n_promo>0 且 n_recycle>0 且升级命中率均值
     > 未升级均值（docs/262 判据 4 同款）。
判定映射（§1.4 冻结）：COMPOSITIONAL_GENERALIZATION（两子主张齐 → L4 踩实）/
COMPOSITIONAL_FLAT（记忆级不迁移的诚实负）/ PARTIAL / GUARD_FAIL / L4_B2_BLOCKED。

守卫（§1.5 冻结，不进判据）：R_L4E_GUARD_D251（Mode OFF 复现 docs/251，32 项）、
R_L4E_GUARD_D246（run_guard_quota + guard_vs_d246，12/12）、R_L4E_REPRO_RATIO
（ON vs OFF 全流 ratio abs<1e-9；7/7 = S1-S4+R0+R0B+R1）、R_L4E_REPRO_D262
（G 流全 R1 Mode ON 逐位复现 docs/262 §三：n_split=6、compound_frac=0.7500、
spurious=0.3333、ratio=0.951261 等）、R_L4E_PREFIX_EQ（C ≡ T 校准前缀逐窗一致 +
种群哈希）、R_L4E_TWO_PHASE_EQ（G 单阶段 ≡ T 两阶段逐窗一致）、
R_L4E_GROUP_HASH / R_L4E_CONSOLIDATE_HASH（两轮逐位一致）、R_L4E_R0B_NOSPLIT
（n_split(R0b)==0）、确定性复现（timing vs main 全 R_L4E_* 逐位一致，除 TAG/ELAPSED）。

安全纪律（§1.10 冻结）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_L4E_* 摘要块；
运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；DAVIS/Downloads 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——新文件仅本文件，import 复用。
野流真实路径在本文件内冻结映射（本仓库 cross_domain_test 为脱敏版，DL_DIR/WILD_VIDEOS
为占位符；docs/264 §1.2 冻结：V1=studio_video_1759283839728.mp4 /
V2=41125413122-1-192.mp4 / V3=千军万马哦哦哦.mp4，目录 C:\\Users\\fa278\\Downloads）。

用法：
  python vision/l4_compose_test5.py --smoke        # 构造冒烟（合成帧，非数据）
  python vision/l4_compose_test5.py --tag timing
  python vision/l4_compose_test5.py --tag main
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

from critical_point import mean_sd, bootstrap_ci
from compose_test import CTX_SPLIT_Y
from stream_test import LOOP_CFG
from real_stream_test import load_video_frames, VIDEOS, WINDOW, RESIZE
from real_recalib import bridge_metrics
from soft_match_test import ALPHA
from fastcut_fix import run_guard_quota
from fastslow_test import (FastSlowLoop, gist_metrics, build_entry_base,
                           R_FAST, R_SLOW, HITS_MIN_FAST, HITS_MIN_SLOW,
                           K_PROMOTE, K_DECAY, K_CONSIST_FAST)
from cross_domain_test import (load_sampled_frames, STREAMS,
                               RADIUS_L3, guard_vs_d246)
# import 复用 l4_compose_test4（docs/264 §1.10 冻结清单：SlotLoop4/run_slot4_stream 为
# 本格机制基座——组合留出 = 机制不变 + 测量留出）
from l4_compose_test4 import SlotLoop4, run_slot4_stream
from l4_compose_test3 import (G_WIN, K_G_CONFIRM, K_G_LEDGER, ALL_STREAMS,
                              group_split_align, nonsplit_compare3)
from l4_compose_test2 import proto_detail2, _synth_frames
from l4_compose_test import (K_SPLIT, DELTA_REL, K_LEDGER, SLOT_SPARSE,
                             PARTICIPATE, _slot_c2, _c2_hash,
                             r1_segment_info, split_segment_align,
                             guard_d251_items, STREAM_ORDER)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# ---------------- 本格冻结量（docs/264 §1.1/§1.4 冻结；先于任何结果） ----------------
CALIB_END_FRAME = 367          # 校准段结束帧 = 段 0-4 跨度终点（flamingo..dog）
FIRST_SPLIT_WIN = 8            # seen_baseline 起点：首个分裂（flamingo w7）之后
TRANSFER_FLOOR = 0.10          # HELDOUT_TRANSFER 绝对下界（docs/264 §1.4 冻结）
TRANSFER_REL = 0.5             # 相对保持系数（docs/264 §1.4 冻结）

# 野流真实路径（docs/264 §1.2 冻结；本仓库 cross_domain_test 为脱敏版，占位符不可用）
DL_DIR_REAL = r"C:\Users\fa278\Downloads"
WILD_FILES_REAL = [("V1", "studio_video_1759283839728.mp4"),
                   ("V2", "41125413122-1-192.mp4"),
                   ("V3", "千军万马哦哦哦.mp4")]

# docs/262 §三 Mode ON R1 期望数字（R_L4E_REPRO_D262 复现目标；容差 1e-4/精确）
D262_R1 = {"mae_mean": 0.069522, "mae_sd": 0.029205, "ratio": 0.951261,
           "sc1_fast": 33, "sc2_fast": 30, "sc1_slow": 5, "sc2_slow": 8,
           "sc2_tagged": 6, "compound_frac": 0.7500, "n_split": 6,
           "n_retired_slow": 6, "spurious_split_frac": 0.3333,
           "avg_post_split_hits": 1.0000, "gist_cov": 1.0000,
           "seg_flamingo": 11.9971, "seg_camel": 1.4300,
           "grp_g1": 19, "grp_g1g2": 15, "grp_g1g2g3": 3, "grp_trigger": 3,
           "grp_children": 6, "grp_single": 0, "grp_group": 6,
           "grp_consolidated": 3, "grp_cons_side0": 1, "grp_cons_side1": 2,
           "group_align_rate": 0.6667, "group_n_aligned": 2}
D262_CONS = [(7, 0, 3), (28, 1, 4), (44, 1, 4)]   # (AT, SIDE, N) 逐事件（docs/262 §三）


# ---------------- 原型种群哈希（PREFIX_EQ/TWO_PHASE_EQ 口径；确定性纯函数） ----------------
def _proto_key(p):
    led = ",".join("%s:%d" % ("N" if k is None else str(k), len(v))
                   for k, v in sorted(p.get("ledger", {}).items(),
                                      key=lambda kv: (kv[0] is None, str(kv[0]))))
    return "%d:%s:%s:%.6f:%.6f:%d:%d:%d:%d:%s" % (
        p["pid"], p["kind"], ("N" if p.get("tag") is None else str(p["tag"])),
        p["mu"][0], p["mu"][1], p["hits"], p["created"], p["n_match"],
        p.get("n_backfill", 0), led)


def proto_pop_hash(protos):
    s = ";".join(sorted(_proto_key(p) for p in protos))
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ---------------- 流运行（§1.2 冻结；T 两阶段 + G/C 单阶段） ----------------
def finalize_out(loop, n_frames):
    """run_slot4_stream 尾部等价（docs/264 §1.2：T 流 finalize 与 docs/262 同构）。"""
    base = loop.finalize(max(1, n_frames // WINDOW), labels=None)
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
    out.update({"frames": n_frames, "n_windows": len(mae_arr), "n_valid": n_valid,
                "mae_mean_win": round(mae_m, 6), "mae_sd_win": round(mae_sd, 6),
                "mae_ci95": [round(mae_lo, 6), round(mae_hi, 6)],
                "mae_q1": round(q1, 6), "mae_q4": round(q4, 6),
                "ratio": round(ratio, 6), "pin_frac": base["pin_frac"],
                "theta_mean": base["theta_mean"]})
    if loop.mode == "on":
        out["slot_coverage"] = loop.slot_coverage()
        out["c2_hash"] = _c2_hash(loop.c2_trace)
        out["proto_detail"] = proto_detail2(loop)
    return out


def snapshot_state(loop):
    """帧 CALIB_END_FRAME 处机制状态（T 流两阶段 step 的中间快照；供 PREFIX_EQ 与
    P_C 静态覆盖）。快照在 finalize 之前（finalize 会冲刷尾部缓冲、多处理一个部分窗）
    ——"帧 367 处记忆"的准确时刻。"""
    return {"protos_hash": proto_pop_hash(loop.prototypes),
            "protos": list(loop.prototypes),
            "n_protos": len(loop.prototypes),
            "E": list(loop.energy_trace), "U": list(loop.up_trace),
            "c2": list(loop.c2_trace), "matched": list(loop.match_trace),
            "buf_len": len(loop._frame_buf)}


def run_slot4_T(frames, split_at):
    """T 流（docs/264 §1.2）：先 step frames[:split_at]（校准段）-> 快照 -> 再 step
    剩余（留出段）-> finalize。与 run_slot4_stream 单阶段等价（确定性）。"""
    loop = SlotLoop4(mode="on", window=WINDOW, **LOOP_CFG)
    for g in frames[:split_at]:
        loop.step(g)
    snap = snapshot_state(loop)
    for g in frames[split_at:]:
        loop.step(g)
    out = finalize_out(loop, len(frames))
    return out, loop, snap


# ---------------- M8 组合留出归因（docs/264 §1.3 冻结；只读既有 trace） ----------------
def holdout_attribution(loop, calib_end_frame, first_split_win):
    """逐窗归因 + 率（判据 1-2 口径）。eligible(w) = E>=10 且 c2 in {0,1}。
    transfer_tagged_hit = 打标原型（创建于校准段）门控命中；self = 创建于留出段。
    门控保证 tag==c2（机制逐字）。全部只读 loop.c2_trace/energy_trace/match_trace/
    split_log。"""
    n_w = len(loop.energy_trace)
    calib_wins = {w for w in range(n_w) if w * WINDOW < calib_end_frame}
    heldout_wins = {w for w in range(n_w) if w * WINDOW >= calib_end_frame}
    split_by_pid = {sl["pid"]: sl for sl in loop.split_log}      # 打标 = 分裂子条目
    calib_created = {pid for pid, sl in split_by_pid.items()
                     if sl["split_at"] in calib_wins}
    # seen_baseline：校准段 w>=first_split_win 的 eligible 打标命中率（预注册口径）
    n_c_elig = 0
    c_tag_hits = 0
    for w in calib_wins:
        if w < first_split_win:
            continue
        if loop.energy_trace[w] < 10 or loop.c2_trace[w] is None:
            continue
        n_c_elig += 1
        pid = loop.match_trace[w][1]
        if pid is not None and pid in split_by_pid:
            c_tag_hits += 1
    seen_baseline = (c_tag_hits / n_c_elig) if n_c_elig > 0 else 0.0
    # 留出段归因
    n_h_elig = 0
    trans = selfh = 0
    n_h_created = 0
    n_h_particip = 0
    transfer_pids = set()
    self_pids = set()
    for w in heldout_wins:
        if loop.energy_trace[w] < 10:
            continue
        n_h_particip += 1
        c2 = loop.c2_trace[w]
        pid = loop.match_trace[w][1]
        if c2 is None:
            continue
        n_h_elig += 1
        if pid is not None and pid in split_by_pid:
            if pid in calib_created:
                trans += 1
                transfer_pids.add(pid)
            else:
                selfh += 1
                self_pids.add(pid)
    n_h_created = sum(1 for cl in loop.created_log if cl["created"] in heldout_wins)
    tagged_h = trans + selfh
    return {
        "calib_wins": sorted(calib_wins), "heldout_wins": sorted(heldout_wins),
        "n_calib_eligible": n_c_elig, "n_heldout_eligible": n_h_elig,
        "n_heldout_particip": n_h_particip,
        "seen_baseline": round(seen_baseline, 6),
        "calib_tagged_hits": c_tag_hits,
        "transfer_tagged_hits": trans, "self_tagged_hits": selfh,
        "tagged_hits_H": tagged_h,
        "transfer_tagged_hit_rate": round(trans / max(1, n_h_elig), 6),
        "tagged_hit_rate_H": round(tagged_h / max(1, n_h_elig), 6),
        "self_form_rate": round(n_h_created / max(1, n_h_particip), 6),
        "calib_share": round(trans / max(1, tagged_h), 6),
        "transfer_pids": sorted(transfer_pids), "self_pids": sorted(self_pids),
    }


# ---------------- M9 静态冻结记忆覆盖（docs/264 §1.3 冻结；诊断级） ----------------
def static_cover(protos_c, loop_t, heldout_wins, r_slow):
    """P_C（校准流帧 367 快照打标慢原型，机制从未见过留出帧）对留出段窗口的 r_slow
    覆盖。protos_c = 快照原型列表（C 流快照 = 纯校准记忆，含打标慢原型）。"""
    tagged_c = [p for p in protos_c
                if p["kind"] == "slow" and p.get("tag") is not None]
    n_elig = 0
    covered = 0
    for w in heldout_wins:
        if loop_t.energy_trace[w] < 10 or loop_t.c2_trace[w] is None:
            continue
        n_elig += 1
        x = (float(np.log1p(loop_t.energy_trace[w])),
             float(np.log1p(loop_t.up_trace[w])))
        c2 = loop_t.c2_trace[w]
        best_d = None
        for p in tagged_c:
            if p.get("tag") != c2:
                continue
            d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
            if best_d is None or d < best_d:
                best_d = d
        if best_d is not None and best_d <= r_slow:
            covered += 1
    return {"n_eligible": n_elig, "covered": covered,
            "rate": round(covered / max(1, n_elig), 6),
            "n_tagged_c": len(tagged_c)}


# ---------------- PREFIX_EQ / TWO_PHASE_EQ（docs/264 §1.5 冻结） ----------------
def prefix_eq(snap_c, snap_t):
    """C 流快照（帧 367，finalize 前）与 T 流快照（帧 367）逐窗 (E, U, c2, matched_pid)
    一致 + 原型种群哈希一致 + 缓冲帧数一致——"留出段到达时的记忆 = 从未见过留出帧的
    校准流记忆"的构造性证明（两流独立运行，同前缀同状态）。"""
    n = len(snap_c["E"])
    if len(snap_t["E"]) != n or len(snap_t["matched"]) != n:
        return 0
    ok = (all(snap_c["E"][i] == snap_t["E"][i] for i in range(n))
          and all(snap_c["U"][i] == snap_t["U"][i] for i in range(n))
          and all(snap_c["c2"][i] == snap_t["c2"][i] for i in range(n))
          and all(snap_c["matched"][i] == snap_t["matched"][i] for i in range(n))
          and snap_c["buf_len"] == snap_t["buf_len"]
          and snap_c["protos_hash"] == snap_t["protos_hash"])
    return int(ok)


def two_phase_eq(g_loop, t_loop):
    """G（单阶段）与 T（两阶段）逐窗 (mae, E, U, c2, matched_pid) 一致 + 终态种群哈希
    一致——归因测量挂在与 docs/262 逐位复现的同一运行上。"""
    if len(g_loop.mae) != len(t_loop.mae):
        return 0
    n = len(g_loop.energy_trace)
    ok = (all(abs(a - b) < 1e-12 for a, b in zip(g_loop.mae, t_loop.mae))
          and all(g_loop.energy_trace[i] == t_loop.energy_trace[i] for i in range(n))
          and all(g_loop.up_trace[i] == t_loop.up_trace[i] for i in range(n))
          and all(g_loop.c2_trace[i] == t_loop.c2_trace[i] for i in range(n))
          and all(g_loop.match_trace[i] == t_loop.match_trace[i] for i in range(n))
          and proto_pop_hash(g_loop.prototypes) == proto_pop_hash(t_loop.prototypes))
    return int(ok)


# ---------------- R_L4E_REPRO_D262（docs/264 §1.5-4 冻结：docs/262 基座逐位复现） ----------------
def guard_d262(g_out, g_loop):
    items = []
    r = g_out
    for name, exp in (("MAE", D262_R1["mae_mean"]), ("MAE_SD", D262_R1["mae_sd"]),
                      ("RATIO", D262_R1["ratio"]),
                      ("COMPOUND_FRAC", D262_R1["compound_frac"]),
                      ("SPURIOUS_FRAC", D262_R1["spurious_split_frac"]),
                      ("AVG_POST_HITS", D262_R1["avg_post_split_hits"]),
                      ("GIST_COV", D262_R1["gist_cov"])):
        got = r["mae_mean_win"] if name == "MAE" else \
            r["mae_sd_win"] if name == "MAE_SD" else \
            r["ratio"] if name == "RATIO" else \
            r["compound_frac"] if name == "COMPOUND_FRAC" else \
            r["spurious_split_frac"] if name == "SPURIOUS_FRAC" else \
            r["avg_post_split_hits"] if name == "AVG_POST_HITS" else \
            r["gist"]["cov"]
        items.append((name, int(abs(got - exp) < 1e-4)))
    for name, exp in (("SC1_FAST", D262_R1["sc1_fast"]),
                      ("SC2_FAST", D262_R1["sc2_fast"]),
                      ("SC1_SLOW", D262_R1["sc1_slow"]),
                      ("SC2_SLOW", D262_R1["sc2_slow"]),
                      ("SC2_TAGGED", D262_R1["sc2_tagged"]),
                      ("N_SPLIT", D262_R1["n_split"]),
                      ("N_RETIRED_SLOW", D262_R1["n_retired_slow"])):
        items.append((name, int(r[name.lower()] == exp)))
    seg = g_out["seg_info"]
    items.append(("SEG_FLAMINGO",
                  int(abs(seg[0]["ratio"] - D262_R1["seg_flamingo"]) < 1e-4)))
    items.append(("SEG_CAMEL",
                  int(abs(seg[3]["ratio"] - D262_R1["seg_camel"]) < 1e-4)))
    gd = g_out["group_diag"]
    for name, exp in (("GRP_G1", D262_R1["grp_g1"]),
                      ("GRP_G1G2", D262_R1["grp_g1g2"]),
                      ("GRP_G1G2G3", D262_R1["grp_g1g2g3"]),
                      ("GRP_TRIGGER", D262_R1["grp_trigger"]),
                      ("GRP_CHILDREN", D262_R1["grp_children"]),
                      ("GRP_SINGLE", D262_R1["grp_single"]),
                      ("GRP_GROUP", D262_R1["grp_group"]),
                      ("GRP_CONSOLIDATED", D262_R1["grp_consolidated"])):
        got = gd["n_g1"] if name == "GRP_G1" else \
            gd["n_g1g2"] if name == "GRP_G1G2" else \
            gd["n_g1g2g3"] if name == "GRP_G1G2G3" else \
            gd["n_trigger"] if name == "GRP_TRIGGER" else \
            gd["n_group_children"] if name == "GRP_CHILDREN" else \
            gd["n_split_single"] if name == "GRP_SINGLE" else \
            gd["n_split_group"] if name == "GRP_GROUP" else \
            gd["n_consolidated"]
        items.append((name, int(got == exp)))
    items.append(("GRP_CONS_SIDE0",
                  int(gd["n_consolidated_by_side"]["0"] == D262_R1["grp_cons_side0"])))
    items.append(("GRP_CONS_SIDE1",
                  int(gd["n_consolidated_by_side"]["1"] == D262_R1["grp_cons_side1"])))
    ga = g_out["group_align"]
    items.append(("GRP_ALIGN_RATE",
                  int(abs(ga["group_align_rate"] - D262_R1["group_align_rate"]) < 1e-4)))
    items.append(("GRP_N_ALIGNED", int(ga["n_aligned"] == D262_R1["group_n_aligned"])))
    # 物化事件 (AT, SIDE, N) 逐事件核对
    cons = [(c["at"], c["side"], c["n"]) for c in gd["consolidate_log"]]
    items.append(("CONS_EVENTS", int(cons == D262_CONS)))
    ok = int(all(v for _, v in items))
    detail = ",".join("%s:%d" % (n, v) for n, v in items)
    return ok, detail


# ---------------- 构造冒烟（合成帧，非数据；R_L4E_SMOKE_*） ----------------
def smoke_main():
    """构造冒烟（docs/264 §二 轮 2）：SlotLoop4 off/on 在合成帧上构造运行正常；
    组合留出测量层语义核对——(A) 合成帧 ON/OFF ratio 逐位一致；(B) 归因函数在合成流
    上运行且不变量成立（率 in [0,1]、tagged>=transfer、eligible>0）；(C) PREFIX_EQ
    逻辑（C 流前缀 ≡ T 流快照，确定性纯函数）；(D) TWO_PHASE_EQ 逻辑（G 单阶段 ≡
    T 两阶段）；(E) 静态覆盖函数在 [0,1] 内；(F) 无打标退化流（baseline=0）不崩。"""
    results = {}
    frames = _synth_frames(30)
    off_out, _ = run_slot4_stream(frames, "off")
    on_out, _ = run_slot4_stream(frames, "on")
    results["construct_off"] = int(isinstance(off_out, dict)
                                   and off_out.get("n_windows", 0) >= 1)
    results["construct_on"] = int(isinstance(on_out, dict)
                                  and on_out.get("n_windows", 0) >= 1
                                  and "slot_coverage" in on_out
                                  and "group_diag" in on_out)
    results["repro_synth"] = int(abs(off_out["ratio"] - on_out["ratio"]) < 1e-9)

    # 100 帧合成流（两段块运动，产生 E>=10 与 c2 变化）：G/C/T 三流 + 归因/静态/前缀/两阶段
    frames2 = []
    for k in range(100):
        f = np.zeros((120, 160), dtype=np.uint8)
        x0 = 15 + 2 * (k % 25)
        f[40:60, x0:x0 + 25] = 255
        if k >= 50:
            f[75:95, (x0 + 40) % 140:(x0 + 40) % 140 + 25] = 255
        frames2.append(f)
    split_at = 50
    g_out, g_loop = run_slot4_stream(frames2, "on")
    t_out, t_loop, t_snap = run_slot4_T(frames2, split_at)
    c_out, c_loop, c_snap = run_slot4_T(frames2[:split_at], split_at)
    n_w = len(t_loop.energy_trace)
    held_wins = {w for w in range(n_w) if w * WINDOW >= split_at}
    att = holdout_attribution(t_loop, split_at, 8)
    stc = static_cover(c_snap["protos"], t_loop, held_wins, R_SLOW)
    results["prefix_eq_logic"] = prefix_eq(c_snap, t_snap)
    results["two_phase_eq_logic"] = two_phase_eq(g_loop, t_loop)
    rates = [att["transfer_tagged_hit_rate"], att["tagged_hit_rate_H"],
             att["seen_baseline"], att["self_form_rate"], att["calib_share"]]
    results["attr_invariants"] = int(
        all(0.0 <= rv <= 1.0 for rv in rates)
        and att["transfer_tagged_hits"] <= att["tagged_hits_H"]
        and att["n_heldout_eligible"] >= 0
        and att["transfer_tagged_hit_rate"] <= att["tagged_hit_rate_H"])
    results["static_invariants"] = int(
        0.0 <= stc["rate"] <= 1.0 and stc["n_eligible"] >= 0)
    # 退化流：30 帧合成流（无分裂）-> baseline=0 不崩、transfer=0
    _, loop30 = run_slot4_stream(_synth_frames(30), "on")
    att0 = holdout_attribution(loop30, 15, 8)
    results["degenerate_ok"] = int(
        att0["seen_baseline"] == 0.0 and att0["transfer_tagged_hits"] == 0
        and att0["transfer_tagged_hit_rate"] == 0.0)

    for k in ("construct_off", "construct_on", "repro_synth",
              "prefix_eq_logic", "two_phase_eq_logic", "attr_invariants",
              "static_invariants", "degenerate_ok"):
        print("R_L4E_SMOKE_%s=%d" % (k.upper(), results[k]))
    return 0 if all(results.values()) else 1


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        return smoke_main()
    t0 = time.time()
    t_dec = t_off = t_on = 0.0

    # ---- 数据加载（一次，多流复用；docs/248 §1.2/§1.5 逐字；野流用本格真实路径） ----
    loaded = {}
    for vid, name in WILD_FILES_REAL:
        p = os.path.join(DL_DIR_REAL, name)
        if not os.path.exists(p):
            sys.stderr.write("wild video missing: %s\n" % p)
            print("R_L4E_VERDICT=L4_B2_BLOCKED")
            return 3
        frames, step, total = load_sampled_frames(p)
        loaded[vid] = frames
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r0b_frames = allv["bear"] * 5            # R0b = bear 段 x5 = 410 帧（负对照）
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    calib_end = spans[4][1]                  # 段 0-4 跨度终点 = 367（docs/264 §1.1 冻结）
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    stream_frames = {}
    for sid, sname, vidx in STREAMS:
        fr = []
        for vi in vidx:
            fr.extend(loaded[WILD_FILES_REAL[vi][0]])
        stream_frames[sid] = fr
    t_dec = time.time() - t0

    # ---- Mode OFF（DeferredLoop 逐字；守卫 R_L4E_GUARD_D251） ----
    off = {}
    for sid in STREAM_ORDER:
        out, _ = run_slot4_stream(stream_frames[sid], "off")
        out["stream_id"] = sid
        off[sid] = out
    off_r0, _ = run_slot4_stream(r0_frames, "off")
    off_r0b, _ = run_slot4_stream(r0b_frames, "off")
    off_r1, _ = run_slot4_stream(r1_frames, "off")
    off_r1["bridge"] = bridge_metrics(build_entry_base(off_r1), spans)
    off_r1["gist"] = gist_metrics(off_r1, switch_windows)
    t_off = time.time() - t0 - t_dec

    d251_items = guard_d251_items(off, off_r1)
    d251_passed = sum(1 for _, v in d251_items)
    d251_ok = int(all(v for _, v in d251_items))
    d251_detail = ",".join("%s:%d" % (n, v) for n, v in d251_items)

    # ---- Mode ON：S1-S4/R0/R0b（支持 + REPRO_RATIO）+ G（全 R1 单阶段）+ C + T ----
    on = {}
    for sid, sname, vidx in STREAMS:
        out, _ = run_slot4_stream(stream_frames[sid], "on")
        out["stream_id"] = sid
        on[sid] = out
    on_r0, on_r0_loop = run_slot4_stream(r0_frames, "on")
    on_r0b, on_r0b_loop = run_slot4_stream(r0b_frames, "on")
    # G 流（守卫：docs/262 基座复现；单阶段 = docs/262 R1 Mode ON 完全同构）
    g_out, g_loop = run_slot4_stream(r1_frames, "on")
    g_out["bridge"] = bridge_metrics(build_entry_base(g_out), spans)
    g_out["gist"] = gist_metrics(g_out, switch_windows)
    g_out["seg_info"] = r1_segment_info(g_loop, spans)
    g_out["split_align"] = split_segment_align(g_loop, spans, g_out["seg_info"])
    g_out["group_align"] = group_split_align(g_out["group_diag"], spans,
                                             g_out["seg_info"],
                                             len(g_loop.energy_trace))
    # C 流（防循环对照：仅段 0-4 = 367 帧；机制从未见过留出帧；帧 367 快照）
    calib_frames = r1_frames[:calib_end]
    c_out, c_loop, c_snap = run_slot4_T(calib_frames, len(calib_frames))
    # T 流（组合留出主测量：全 R1 两阶段，帧 calib_end 快照）
    t_out, t_loop, t_snap = run_slot4_T(r1_frames, calib_end)
    t_out["bridge"] = bridge_metrics(build_entry_base(t_out), spans)
    t_out["gist"] = gist_metrics(t_out, switch_windows)
    t_out["seg_info"] = r1_segment_info(t_loop, spans)
    t_out["split_align"] = split_segment_align(t_loop, spans, t_out["seg_info"])
    t_out["group_align"] = group_split_align(t_out["group_diag"], spans,
                                             t_out["seg_info"],
                                             len(t_loop.energy_trace))
    t_on = time.time() - t0 - t_dec - t_off

    # ---- R_L4E_REPRO_RATIO（ON vs OFF 全流 ratio，abs < 1e-9；7/7） ----
    repro_items = []
    for sid in STREAM_ORDER:
        repro_items.append(("ratio_%s" % sid,
                            int(abs(on[sid]["ratio"] - off[sid]["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0", int(abs(on_r0["ratio"] - off_r0["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R0B", int(abs(on_r0b["ratio"] - off_r0b["ratio"]) < 1e-9)))
    repro_items.append(("ratio_R1", int(abs(g_out["ratio"] - off_r1["ratio"]) < 1e-9)))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, v) for n, v in repro_items)

    # ---- R_L4E_GUARD_D246（SoftLoop 路径；docs/249/250/251 同一代码路径） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard246_ok, guard246_detail = guard_vs_d246(g0, g1)
    guard246_passed = sum(1 for ch in guard246_detail.split(",") if ch.endswith(":1"))

    # ---- R_L4E_REPRO_D262（docs/262 基座逐位复现；G 流 = 全 R1 Mode ON） ----
    d262_ok, d262_detail = guard_d262(g_out, g_loop)

    # ---- PREFIX_EQ / TWO_PHASE_EQ / R0B_NOSPLIT ----
    prefix_ok = prefix_eq(c_snap, t_snap)
    two_phase_ok = two_phase_eq(g_loop, t_loop)
    r0b_nosplit_ok = int(on_r0b["n_split"] == 0)

    # ---- R_L4E_NONSPLIT_EQ（诊断级：非分裂数字 vs docs/253 Mode ON） ----
    nonsplit_eq, nonsplit_per = nonsplit_compare3(on, on_r0, on_r0b, g_out)

    # ---- M8 组合留出归因 + M9 静态覆盖（T 流 / C 流快照种群） ----
    att = holdout_attribution(t_loop, calib_end, FIRST_SPLIT_WIN)
    heldout_wins = set(att["heldout_wins"])
    stc = static_cover(c_snap["protos"], t_loop, heldout_wins, R_SLOW)

    # ---- 判据（§1.4 冻结） ----
    transfer_rate = att["transfer_tagged_hit_rate"]
    tagged_rate_h = att["tagged_hit_rate_H"]
    seen_base = att["seen_baseline"]
    crit1 = int(transfer_rate >= TRANSFER_FLOOR
                and transfer_rate >= TRANSFER_REL * seen_base)
    crit2 = int(tagged_rate_h >= TRANSFER_REL * seen_base)
    all_found = [on[s] for s in STREAM_ORDER] + [g_out]
    crit3 = int(all(r["ratio"] <= 1.5 for r in all_found)
                and all(r["sc2_slow"] > 0 for r in all_found)
                and g_out["gist"]["cov"] >= 0.5)
    n_promo_total = sum(r["n_promo"] for r in all_found)
    n_recycle_total = sum(r["n_recycle"] for r in all_found)
    promo_means = [(r["promoted_mean_hits"], r["nonpromoted_mean_hits"])
                   for r in all_found if r["sc1_slow"] > 0]
    promo_sep = int(any(mp > mn for mp, mn in promo_means)) if promo_means else 0
    crit4 = int(n_promo_total > 0 and n_recycle_total > 0 and promo_sep)

    # ---- 判定（§1.4 冻结映射；第五格专属语义） ----
    guards_ok = (d251_ok == 1 and guard246_ok == 1 and repro_ok == 1
                 and d262_ok == 1 and prefix_ok == 1 and two_phase_ok == 1
                 and r0b_nosplit_ok == 1)
    if not guards_ok:
        verdict = "GUARD_FAIL"
        vnote = ("guard(s) failed: D251=%d/32, D246=%d/12, REPRO_RATIO=%d, "
                 "REPRO_D262=%d, PREFIX_EQ=%d, TWO_PHASE_EQ=%d, R0B_NOSPLIT=%d -> "
                 "implementation drift; fix implementation, do not judge "
                 "mechanism (see R_L4E_GUARD_* / R_L4E_REPRO_* / R_L4E_PREFIX_EQ / "
                 "R_L4E_TWO_PHASE_EQ)" % (
                     d251_ok, guard246_ok, repro_ok, d262_ok, prefix_ok,
                     two_phase_ok, r0b_nosplit_ok))
    elif crit1 and crit2 and crit3 and crit4:
        verdict = "COMPOSITIONAL_GENERALIZATION"
        vnote = ("criteria 1-4 all pass and all guards pass: calibration-formed "
                 "tagged conditional memory (c2-side x feature-cluster) is still "
                 "gated-used on held-out segments (unseen combinations), and "
                 "slot-indexed structure behavior is kept; transfer_tagged_hit_"
                 "rate(H)=%.4f >= floor 0.10 and >= 0.5*seen_baseline(%.4f); "
                 "with docs/262 sub-claim 1 (COMPOSABLE_REAL) -> L4 substantiated "
                 "(docs/255 sec-2 verdict)" % (transfer_rate, seen_base))
    elif (not crit1) and crit2 and crit3 and crit4:
        verdict = "COMPOSITIONAL_FLAT"
        vnote = ("HELDOUT_TRANSFER fails (memory-level transfer insufficient) "
                 "but STRUCTURE_KEEP/FOUNDATION_KEEP/PROMOTION_KEEP pass: "
                 "transfer_tagged_hit_rate(H)=%.4f < floor 0.10 or < 0.5*seen_"
                 "baseline(%.4f); tagged_hit_rate(H)=%.4f (structure kept, incl. "
                 "self-formed memory); honest negative at memory level: "
                 "calibration-formed conditional memory does not cover held-out "
                 "content (feature-domain extrapolation boundary); no threshold "
                 "rollback (docs/63)" % (
                     transfer_rate, seen_base, tagged_rate_h))
    else:
        why = []
        if not crit2:
            why.append("STRUCTURE_KEEP fails (tagged_hit_rate(H)=%.4f < 0.5*base %.4f)"
                       % (tagged_rate_h, seen_base))
        if not crit3:
            why.append("FOUNDATION_KEEP fails (ratio/sc2_slow/gist_cov; see numbers)")
        if not crit4:
            why.append("PROMOTION_KEEP fails (n_promo/n_recycle/hit-rate separation)")
        verdict = "PARTIAL"
        vnote = "; ".join(why) + " (see R_L4E_CRIT* numbers)"

    # ---- 工件（自描述 JSON） ----
    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "holdout": {"calib_end_frame": calib_end,
                       "calib_segments": [0, 1, 2, 3, 4],
                       "heldout_segments": [5, 6, 7, 8],
                       "calib_videos": ["flamingo", "surf", "bear", "camel", "dog"],
                       "heldout_videos": ["blackswan", "car-turn", "motorbike",
                                          "soccerball"],
                       "transfer_floor": TRANSFER_FLOOR,
                       "transfer_rel": TRANSFER_REL,
                       "first_split_win": FIRST_SPLIT_WIN,
                       "doc": "docs/264 sec 1.1/1.4"},
           "working_point": {"r_slow": round(R_SLOW, 6),
                             "r_fast": round(R_FAST, 6),
                             "hits_min_fast": HITS_MIN_FAST,
                             "hits_min_slow": HITS_MIN_SLOW,
                             "k_promote": K_PROMOTE, "k_decay": K_DECAY,
                             "k_consist_fast": K_CONSIST_FAST,
                             "alpha": ALPHA, "k_split": K_SPLIT,
                             "delta_rel": DELTA_REL, "k_ledger": K_LEDGER,
                             "w_bf": 4, "g_win": G_WIN,
                             "k_g_confirm": K_G_CONFIRM,
                             "k_g_ledger": K_G_LEDGER,
                             "ctx_split_y": CTX_SPLIT_Y,
                             "slot_sparse_px": SLOT_SPARSE,
                             "participate": PARTICIPATE},
           "mechanism": ("SlotLoop4 (docs/262 verbatim, imported unchanged): slot c2 "
                         "observable + prototype c2 ledger + birth backfill W_BF=4 + "
                         "explicit upgrade ledger inheritance + child mu "
                         "re-initialization + single-prototype split + tagged gated "
                         "matching + group-level epoch (G_WIN=8) G1/G2/G4 + anchor "
                         "consolidation + G3 + group split; compositional holdout = "
                         "mechanism unchanged + measurement-layer holdout: 9 segments "
                         "split by frozen order rule into calibration C = segments "
                         "0-4 (frames [0,367)) and held-out H = segments 5-8 (frames "
                         "[367,588)); held-out segments all after calibration "
                         "(stream-order anti-circularity); parameters zero re-tune; "
                         "single deterministic pass; no re-training; no callbacks"),
           "loop": LOOP_CFG,
           "r1_switch_windows": switch_windows,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "l4_compose_test5",
        "doc_ref": "docs/63, docs/246, docs/247, docs/251, docs/253, docs/254, "
                   "docs/255, docs/256, docs/262, docs/264",
        "config": cfg,
        "off_guard": {"items": len(d251_items), "passed": d251_passed,
                      "ok": d251_ok, "detail": d251_detail},
        "g_stream": {k: g_out[k] for k in (
            "frames", "n_windows", "n_valid", "mae_mean_win", "mae_sd_win",
            "ratio", "sc1_fast", "sc2_fast", "sc1_slow", "sc2_slow",
            "sc2_tagged", "compound_frac", "n_split", "n_retired_slow",
            "spurious_split_frac", "avg_post_split_hits", "churn_slow",
            "c2_hash")} | {"gist_cov": g_out["gist"]["cov"],
                           "group_align": g_out["group_align"],
                           "seg_info": g_out["seg_info"]},
        "c_stream": {"frames": c_out["frames"], "n_windows": c_out["n_windows"],
                     "n_split": c_out["n_split"], "sc2_tagged": c_out["sc2_tagged"],
                     "compound_frac": c_out["compound_frac"],
                     "grp_trigger": c_out["group_diag"]["n_trigger"],
                     "consolidated": c_out["group_diag"]["n_consolidated"],
                     "pop_hash_snap": c_snap["protos_hash"]},
        "t_stream": {"frames": t_out["frames"], "n_windows": t_out["n_windows"],
                     "n_split": t_out["n_split"], "sc2_tagged": t_out["sc2_tagged"],
                     "compound_frac": t_out["compound_frac"],
                     "grp_trigger": t_out["group_diag"]["n_trigger"],
                     "consolidated": t_out["group_diag"]["n_consolidated"],
                     "pop_hash": proto_pop_hash(t_loop.prototypes)},
        "holdout": att,
        "static_cover": stc,
        "criteria": {"crit1_holdout_transfer": crit1,
                     "crit2_structure_keep": crit2,
                     "crit3_foundation_keep": crit3,
                     "crit4_promotion_keep": crit4,
                     "n_promo_total": n_promo_total,
                     "n_recycle_total": n_recycle_total,
                     "promo_means": promo_means, "promo_sep": promo_sep},
        "verdict": {"verdict": verdict, "note": vnote},
        "guards": {"d251": {"ok": d251_ok, "passed": d251_passed,
                            "detail": d251_detail},
                   "d246": {"ok": guard246_ok, "passed": guard246_passed,
                            "detail": guard246_detail},
                   "repro_ratio": {"ok": repro_ok, "detail": repro_detail},
                   "repro_d262": {"ok": d262_ok, "detail": d262_detail},
                   "prefix_eq": prefix_ok,
                   "two_phase_eq": two_phase_ok,
                   "r0b_nosplit": {"n_split": on_r0b["n_split"],
                                   "ok": r0b_nosplit_ok},
                   "nonsplit_eq": {"ok": nonsplit_eq,
                                   "per_stream": nonsplit_per}},
        "timing": {"elapsed_sec": round(time.time() - t0, 2),
                   "decode_sec": round(t_dec, 2),
                   "off_sec": round(t_off, 2), "on_sec": round(t_on, 2)},
    }
    res_path = os.path.join(args.out_dir, "l4e_%s.json" % args.tag)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定；无中文/日志/JSON） ----
    print("R_L4E_TAG=%s" % args.tag)
    print("R_L4E_G_WIN=%d" % G_WIN)
    print("R_L4E_K_G_CONFIRM=%d" % K_G_CONFIRM)
    print("R_L4E_K_G_LEDGER=%d" % K_G_LEDGER)
    print("R_L4E_W_BF=%d" % 4)
    print("R_L4E_R_SLOW=%.6f" % R_SLOW)
    print("R_L4E_R_FAST=%.6f" % R_FAST)
    print("R_L4E_K_SPLIT=%d" % K_SPLIT)
    print("R_L4E_DELTA_REL=%.4f" % DELTA_REL)
    print("R_L4E_K_LEDGER=%d" % K_LEDGER)
    print("R_L4E_CTX_SPLIT_Y=%.1f" % CTX_SPLIT_Y)
    print("R_L4E_CALIB_END_FRAME=%d" % calib_end)
    print("R_L4E_TRANSFER_FLOOR=%.2f" % TRANSFER_FLOOR)
    print("R_L4E_TRANSFER_REL=%.2f" % TRANSFER_REL)
    for sid in STREAM_ORDER:
        r = off[sid]
        print("R_L4E_OFF_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4E_OFF_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4E_OFF_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4E_OFF_%s_N_PROMO=%d" % (sid, r["n_promo"]))
        print("R_L4E_OFF_%s_N_RECYCLE=%d" % (sid, r["n_recycle"]))
        print("R_L4E_OFF_%s_CHURN_SLOW=%.4f" % (sid, r["churn_slow"]))
    print("R_L4E_OFF_R1_RATIO=%.6f" % off_r1["ratio"])
    print("R_L4E_OFF_R1_SC1_FAST=%d" % off_r1["sc1_fast"])
    print("R_L4E_OFF_R1_SC2_SLOW=%d" % off_r1["sc2_slow"])
    print("R_L4E_OFF_R1_N_PROMO=%d" % off_r1["n_promo"])
    print("R_L4E_OFF_R1_N_RECYCLE=%d" % off_r1["n_recycle"])
    print("R_L4E_OFF_R1_CHURN_SLOW=%.4f" % off_r1["churn_slow"])
    print("R_L4E_OFF_R1_GIST_COV=%.4f" % off_r1["gist"]["cov"])
    print("R_L4E_OFF_R1_BRIDGE_SW=%.4f" % off_r1["bridge"]["bridge_corr_switch"])
    print("R_L4E_OFF_R0_RATIO=%.6f" % off_r0["ratio"])
    print("R_L4E_OFF_R0_SC1_FAST=%d" % off_r0["sc1_fast"])
    print("R_L4E_OFF_R0_SC2_SLOW=%d" % off_r0["sc2_slow"])
    print("R_L4E_OFF_R0_CHURN_SLOW=%.4f" % off_r0["churn_slow"])
    print("R_L4E_OFF_R0_N_PROMO=%d" % off_r0["n_promo"])
    print("R_L4E_OFF_R0_N_RECYCLE=%d" % off_r0["n_recycle"])
    print("R_L4E_OFF_R0B_RATIO=%.6f" % off_r0b["ratio"])
    print("R_L4E_OFF_R0B_SC1_FAST=%d" % off_r0b["sc1_fast"])
    print("R_L4E_OFF_R0B_SC2_SLOW=%d" % off_r0b["sc2_slow"])
    print("R_L4E_OFF_R0B_CHURN_SLOW=%.4f" % off_r0b["churn_slow"])
    print("R_L4E_OFF_R0B_N_PROMO=%d" % off_r0b["n_promo"])
    print("R_L4E_OFF_R0B_N_RECYCLE=%d" % off_r0b["n_recycle"])
    print("R_L4E_GUARD_D251=%d" % d251_ok)
    print("R_L4E_GUARD_D251_ITEMS=%d" % len(d251_items))
    print("R_L4E_GUARD_D251_PASSED=%d" % d251_passed)
    print("R_L4E_GUARD_D251_DETAIL=%s" % d251_detail)
    print("R_L4E_G_R1_MAE=%.6f" % g_out["mae_mean_win"])
    print("R_L4E_G_R1_MAE_SD=%.6f" % g_out["mae_sd_win"])
    print("R_L4E_G_R1_MAE_LO=%.6f" % g_out["mae_ci95"][0])
    print("R_L4E_G_R1_MAE_HI=%.6f" % g_out["mae_ci95"][1])
    print("R_L4E_G_R1_RATIO=%.6f" % g_out["ratio"])
    print("R_L4E_G_R1_SC1_FAST=%d" % g_out["sc1_fast"])
    print("R_L4E_G_R1_SC2_FAST=%d" % g_out["sc2_fast"])
    print("R_L4E_G_R1_SC1_SLOW=%d" % g_out["sc1_slow"])
    print("R_L4E_G_R1_SC2_SLOW=%d" % g_out["sc2_slow"])
    print("R_L4E_G_R1_SC2_TAGGED=%d" % g_out["sc2_tagged"])
    print("R_L4E_G_R1_COMPOUND_FRAC=%.4f" % g_out["compound_frac"])
    print("R_L4E_G_R1_N_SPLIT=%d" % g_out["n_split"])
    print("R_L4E_G_R1_N_RETIRED_SLOW=%d" % g_out["n_retired_slow"])
    print("R_L4E_G_R1_SPURIOUS_SPLIT_FRAC=%.4f" % g_out["spurious_split_frac"])
    print("R_L4E_G_R1_AVG_POST_SPLIT_HITS=%.4f" % g_out["avg_post_split_hits"])
    print("R_L4E_G_R1_CHURN_SLOW=%.4f" % g_out["churn_slow"])
    print("R_L4E_G_R1_GIST_COV=%.4f" % g_out["gist"]["cov"])
    print("R_L4E_G_R1_SEG_INFO=%s" % ",".join(
        ("NA" if row["ratio"] is None else "%.4f" % row["ratio"])
        for row in g_out["seg_info"]))
    print("R_L4E_G_R1_SEG_N0=%s" % ",".join(str(row["n0"]) for row in g_out["seg_info"]))
    print("R_L4E_G_R1_SEG_N1=%s" % ",".join(str(row["n1"]) for row in g_out["seg_info"]))
    gd = g_out["group_diag"]
    print("R_L4E_G_R1_GRP_G1=%d" % gd["n_g1"])
    print("R_L4E_G_R1_GRP_G1G2=%d" % gd["n_g1g2"])
    print("R_L4E_G_R1_GRP_G1G2G3=%d" % gd["n_g1g2g3"])
    print("R_L4E_G_R1_GRP_TRIGGER=%d" % gd["n_trigger"])
    print("R_L4E_G_R1_GRP_CHILDREN=%d" % gd["n_group_children"])
    print("R_L4E_G_R1_GRP_SPLIT_SINGLE=%d" % gd["n_split_single"])
    print("R_L4E_G_R1_GRP_SPLIT_GROUP=%d" % gd["n_split_group"])
    print("R_L4E_G_R1_GRP_CONSOLIDATED=%d" % gd["n_consolidated"])
    print("R_L4E_G_R1_GRP_CONS_SIDE0=%d" % gd["n_consolidated_by_side"]["0"])
    print("R_L4E_G_R1_GRP_CONS_SIDE1=%d" % gd["n_consolidated_by_side"]["1"])
    print("R_L4E_G_R1_GRP_ALIGN_RATE=%.4f" % g_out["group_align"]["group_align_rate"])
    print("R_L4E_G_R1_GRP_N_ALIGNED=%d" % g_out["group_align"]["n_aligned"])
    for i, c in enumerate(gd["consolidate_log"]):
        print("R_L4E_G_R1_CONS_%d_AT=%d" % (i, c["at"]))
        print("R_L4E_G_R1_CONS_%d_SIDE=%d" % (i, c["side"]))
        print("R_L4E_G_R1_CONS_%d_N=%d" % (i, c["n"]))
        print("R_L4E_G_R1_CONS_%d_WINS=%s" % (i, ",".join(str(w) for w in c["source_wins"])))
        print("R_L4E_G_R1_CONS_%d_MU0=%.6f" % (i, c["mu0"]))
        print("R_L4E_G_R1_CONS_%d_MU1=%.6f" % (i, c["mu1"]))
    print("R_L4E_REPRO_D262=%d" % d262_ok)
    print("R_L4E_REPRO_D262_ITEMS=%d" % len(d262_detail.split(",")))
    print("R_L4E_REPRO_D262_DETAIL=%s" % d262_detail)
    for sid in ALL_STREAMS:
        if sid in on:
            r = on[sid]
        elif sid == "R0":
            r = on_r0
        elif sid == "R0B":
            r = on_r0b
        else:
            r = g_out
        print("R_L4E_ON_%s_RATIO=%.6f" % (sid, r["ratio"]))
        print("R_L4E_ON_%s_SC1_FAST=%d" % (sid, r["sc1_fast"]))
        print("R_L4E_ON_%s_SC2_SLOW=%d" % (sid, r["sc2_slow"]))
        print("R_L4E_ON_%s_N_SPLIT=%d" % (sid, r["n_split"]))
        print("R_L4E_ON_%s_C2HASH=%s" % (sid, r["c2_hash"]))
        print("R_L4E_ON_%s_GRP_HASH=%s" % (sid, r["group_diag"]["group_hash"]))
        print("R_L4E_ON_%s_CONS_HASH=%s" % (sid, r["group_diag"]["consolidate_hash"]))
    print("R_L4E_REPRO_RATIO=%d" % repro_ok)
    print("R_L4E_REPRO_ITEMS=%d" % len(repro_items))
    print("R_L4E_REPRO_DETAIL=%s" % repro_detail)
    print("R_L4E_GUARD_D246=%d" % guard246_ok)
    print("R_L4E_GUARD_D246_PASSED=%d" % guard246_passed)
    print("R_L4E_GUARD_D246_DETAIL=%s" % guard246_detail)
    print("R_L4E_C_FRAMES=%d" % c_out["frames"])
    print("R_L4E_C_WINDOWS=%d" % c_out["n_windows"])
    print("R_L4E_C_N_SPLIT=%d" % c_out["n_split"])
    print("R_L4E_C_SC2_TAGGED=%d" % c_out["sc2_tagged"])
    print("R_L4E_C_COMPOUND_FRAC=%.4f" % c_out["compound_frac"])
    print("R_L4E_C_GRP_TRIGGER=%d" % c_out["group_diag"]["n_trigger"])
    print("R_L4E_C_CONSOLIDATED=%d" % c_out["group_diag"]["n_consolidated"])
    print("R_L4E_C_POP_HASH=%s" % c_snap["protos_hash"])
    print("R_L4E_T_FRAMES=%d" % t_out["frames"])
    print("R_L4E_T_WINDOWS=%d" % t_out["n_windows"])
    print("R_L4E_T_N_SPLIT=%d" % t_out["n_split"])
    print("R_L4E_T_SC2_TAGGED=%d" % t_out["sc2_tagged"])
    print("R_L4E_T_COMPOUND_FRAC=%.4f" % t_out["compound_frac"])
    print("R_L4E_T_GRP_TRIGGER=%d" % t_out["group_diag"]["n_trigger"])
    print("R_L4E_T_CONSOLIDATED=%d" % t_out["group_diag"]["n_consolidated"])
    print("R_L4E_T_POP_HASH=%s" % proto_pop_hash(t_loop.prototypes))
    print("R_L4E_PREFIX_EQ=%d" % prefix_ok)
    print("R_L4E_TWO_PHASE_EQ=%d" % two_phase_ok)
    print("R_L4E_CALIB_WINS=%d" % len(att["calib_wins"]))
    print("R_L4E_HELDOUT_WINS=%d" % len(att["heldout_wins"]))
    print("R_L4E_CALIB_POST_ELIGIBLE=%d" % att["n_calib_eligible"])
    print("R_L4E_HELDOUT_ELIGIBLE=%d" % att["n_heldout_eligible"])
    print("R_L4E_SEEN_BASELINE=%.6f" % att["seen_baseline"])
    print("R_L4E_CALIB_TAGGED_HITS=%d" % att["calib_tagged_hits"])
    print("R_L4E_TRANSFER_TAGGED_HITS=%d" % att["transfer_tagged_hits"])
    print("R_L4E_SELF_TAGGED_HITS=%d" % att["self_tagged_hits"])
    print("R_L4E_TAGGED_HITS_H=%d" % att["tagged_hits_H"])
    print("R_L4E_TRANSFER_TAGGED_HIT_RATE=%.6f" % att["transfer_tagged_hit_rate"])
    print("R_L4E_TAGGED_HIT_RATE_H=%.6f" % att["tagged_hit_rate_H"])
    print("R_L4E_SELF_FORM_RATE=%.6f" % att["self_form_rate"])
    print("R_L4E_CALIB_SHARE=%.6f" % att["calib_share"])
    print("R_L4E_TRANSFER_PIDS=%s" % ",".join(str(p) for p in att["transfer_pids"]))
    print("R_L4E_SELF_PIDS=%s" % ",".join(str(p) for p in att["self_pids"]))
    print("R_L4E_STATIC_N_ELIGIBLE=%d" % stc["n_eligible"])
    print("R_L4E_STATIC_COVERED=%d" % stc["covered"])
    print("R_L4E_STATIC_COVER_RATE=%.6f" % stc["rate"])
    print("R_L4E_STATIC_N_TAGGED_C=%d" % stc["n_tagged_c"])
    print("R_L4E_CRIT1_HELDOUT_TRANSFER=%d" % crit1)
    print("R_L4E_CRIT1_TRANSFER_RATE=%.6f" % transfer_rate)
    print("R_L4E_CRIT1_BASELINE=%.6f" % seen_base)
    print("R_L4E_CRIT2_STRUCTURE_KEEP=%d" % crit2)
    print("R_L4E_CRIT2_TAGGED_RATE_H=%.6f" % tagged_rate_h)
    print("R_L4E_CRIT3_FOUNDATION_KEEP=%d" % crit3)
    print("R_L4E_CRIT3_GIST_COV=%.4f" % g_out["gist"]["cov"])
    print("R_L4E_CRIT4_PROMOTION_KEEP=%d" % crit4)
    print("R_L4E_PROMO_TOTAL=%d" % n_promo_total)
    print("R_L4E_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_L4E_NONSPLIT_EQ=%d" % nonsplit_eq)
    for sid in ALL_STREAMS:
        if sid == "R0B":
            continue            # docs/253 无 R0b 基准 -> NA
        print("R_L4E_NONSPLIT_%s=%d" % (sid, nonsplit_per[sid]["eq"]))
    print("R_L4E_R0B_NOSPLIT=%d" % on_r0b["n_split"])
    print("R_L4E_R0B_NOSPLIT_OK=%d" % r0b_nosplit_ok)
    print("R_L4E_VERDICT=%s" % verdict)
    print("R_L4E_VERDICT_NOTE=%s" % vnote)
    print("R_L4E_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
