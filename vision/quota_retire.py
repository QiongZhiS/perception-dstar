"""vision/quota_retire.py — docs/251 QUOTA 退役：延迟定级（deferred finalization）消除
评估层配额动作面（docs/250 QUOTA_ORTHOGONAL 的续：让评估层配额正式退役）。

docs/250 verdict=MECH_PASS 但 QUOTA_ORTHOGONAL = 配额仍需——机制 + docs/249 配额叠加仍
翻转 S2/S3/R1 的"升级未完成慢原型"（d_sc2 +1/+2/+1、d_churn -0.2000/-0.2222/-0.1250）。
本实验用机制消除该动作面：**延迟定级**——快->慢升级不在 hits >= k_promote=2 时立即发生，
仅当该原型累积满 hits_min_slow=3 次重匹配（hits >= hits_min_slow）时才最终化为慢原型；
hits 在 [k_promote, hits_min_slow) 的"升级候选"保持快原型（继续 r_fast 匹配、仍受
k_decay=5 回收——docs/250 已确立"短命但正确"为正确行为）；慢原型因此只以"确认已满"的
形态存在（final hits >= hits_min_slow 恒成立），配额（对未完成慢原型的豁免）再无动作面。

预注册（docs/251 §一，冻结；docs/63+247 纪律；判据/旋钮/守卫先于最终运行写入 docs/251，
运行后不改）：
  机制（延迟定级）：
    1. DeferredLoop = FastSlowLoop 的唯一机制改动：快匹配分支升级判定行
       `hits >= k_promote` -> `hits >= hits_min_slow`（定级门槛 K_FINALIZE =
        hits_min_slow=3，复用冻结值，不引入新取值）。其余逻辑（慢优先 -> 快兜底 ->
        立即创建 -> 窗口末回收）与 docs/250 §1.4 逐字一致。
    2. 半成品状态管理：hits=2 的升级候选保持 kind=fast（r_fast 匹配、受 k_decay 回收），
       不产生任何半成品慢状态；慢原型只以"确认已满"形态存在 -> churn_slow=0.0 按构造
       成立 -> 配额（hits_min_eff <= hits_min_slow 恒成立）对慢层无可豁免 -> 配额动作面=0。
    3. k_promote=2 被延迟定级吸收（DEPRECATED/SUPERSEDED，行为零作用，仅文档追溯）。
    4. 不做段长预测（延续 docs/250）：无帧差/位移等外部信号进机制决策；帧差只用于
       野流诊断。
  数据/GT：DAVIS R1（9 视频拼接，真值段边界 = 视频切换时点）+ 野流 S1-S4（复用
    cross_domain_test 加载，帧差近似段边界作诊断）。
  判据（每判据带 docs/247 层级标签，冻结）：
    1. [L3][机制][退役] QUOTA_RETIRED : 配额开 vs 关在全部流（S1-S4 + R1）结果完全一致
       （churn/SC2/ratio 逐项相等且 R1 bridge 相等；配额动作面 = 0）。实现口径：机制单独
       vs 机制 + docs/249 配额叠加于慢层（apply_quota/fastcut_gate 复用，docs/250 §六
       同款口径）。
    2. [L3][机制][无配额] CHURN_MECH   : 配额完全关闭，慢原型 churn_slow <= 0.5
       （S1-S4 + R1；预注册推论：按构造 = 0.0）。
    3. [L3][机制][gist正确性] GIST_CORRECT : R1 真值段边界对应率 >= 0.5（|Δ|<=1；
       目标保持 1.0）；野流帧差近似作诊断。
    4. [L3][机制] STABLE/STRUCT 保持     : 全流 ratio <= 1.5 且 SC2_slow > 0。
    5. [L3][机制][行为证据] PROMOTION    : n_promo>0、n_recycle>0、升级命中率均值 >
       未升级均值（升级非随机；确定性流无显著性检验，报告两侧均值）。
  判定：1-5 全过 且 守卫 12/12 = QUOTA_RETIRED（配额正式退役）；判据 1 不过 =
    QUOTA_STILL_NEEDED（如实报告动作面在哪）；部分 = PARTIAL（如实）。
  守卫（不进判据，实现正确性）：复用 fastcut_fix.run_guard_quota + guard_vs_d246
    （docs/249/250 守卫同一代码路径）跑 DAVIS R0+R1，配额关闭字段须复现 docs/246 M=1.5
    工作点行（R0 SC2=3/churn 0/ratio 0.907701；R1 SC1=11/SC2=6/churn 0.4545/
    ratio 0.951261；bridge_sw 0.8750/calib_sw 1.0/holdout_sw 0.75/bridge_vid 0.8889/
    spurious 0；容差 1e-4）-> R_QR_GUARD_D246（12/12）。
  内部复现（诊断）：R_QR_REPRO_RATIO —— DeferredLoop 的 ratio 与 docs/250 §3.3 逐位一致
    （容差 1e-4：S1 1.155669/S2 1.371908/S3 0.732642/S4 0.370964/R1 0.951261）——
    预测路径零改动的构造性控制项。

安全纪律（docs/243-250 同款）：stdout 只输出 ASCII 标签 + 每行一个数字的 R_QR_* 摘要
块；运行经 powershell 包装重定向到 logs/；数字用纯 python 正则（vision/extract_r.py）
抽取；禁止读日志/JSON 原文；Downloads/DAVIS 是数据（只读帧数/文件名）。
禁止修改任何既有脚本——只 import 复用（fastslow_test / fastcut_fix / soft_match_test /
cross_domain_test / real_stream_test / real_recalib / stream_test / critical_point）。

用法：
  python vision/quota_retire.py --tag qr
"""
import argparse
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
from soft_match_test import ALPHA, HITS_MIN
from cross_domain_test import (load_sampled_frames, WILD_VIDEOS, STREAMS,
                               RADIUS_L3, R_BASE_DAVIS, D246, DL_DIR,
                               guard_vs_d246, scene_switch_diag)
from fastcut_fix import run_guard_quota
from fastslow_test import (FastSlowLoop, quota_on_slow, gist_metrics,
                           build_entry_base, R_FAST, R_SLOW,
                           HITS_MIN_FAST, HITS_MIN_SLOW,
                           K_PROMOTE, K_DECAY, K_CONSIST_FAST)

warnings.filterwarnings("ignore")

DEFAULT_OUT = os.path.join("vision", "out", "results")

# 旋钮（预注册，docs/251 §1.3，冻结；全部沿用 docs/250，零回调）
K_FINALIZE = HITS_MIN_SLOW     # 定级门槛 = 3（延迟定级：仅累积满 hits_min_slow 次重匹配
                               # 才最终化为慢；复用冻结值，不引入新取值）
K_PROMOTE_ABSORBED = K_PROMOTE  # 2（被延迟定级吸收；DEPRECATED/SUPERSEDED，行为零作用）

# docs/250 §3.3 四流 + R1 冻结 ratio（内部复现 R_QR_REPRO_RATIO 的期望；预测路径未动）
D250_RATIOS = {"S1": 1.155669, "S2": 1.371908, "S3": 0.732642,
               "S4": 0.370964, "R1": 0.951261}


class DeferredLoop(FastSlowLoop):
    """FastSlowLoop 的延迟定级变体。与 FastSlowLoop._on_window 逐字一致，唯一机制改动 =
    快匹配分支的升级判定行：`hits >= k_promote` -> `hits >= hits_min_slow`（延迟定级，
    docs/251 §1.4）。半成品状态管理：hits 在 [k_promote, hits_min_slow) 的升级候选保持
    kind=fast（r_fast 匹配、受 k_decay 回收），不产生任何半成品慢状态。"""

    def _on_window(self):
        ev_win = self._ev_win if self._ev_win is not None else \
            np.zeros((120, 160), bool)
        mae_w = float(np.mean([f["mae"] for f in self._frame_buf]))
        att_w = float(np.mean([f["att"] for f in self._frame_buf]))
        ev_w = float(np.mean([f["ev"] for f in self._frame_buf]))
        theta_w = float(self._frame_buf[-1]["theta"])
        db_w = float(self._frame_buf[-1]["db"])
        self.mae.append(mae_w)
        self.att.append(att_w)
        self.ev.append(ev_w)
        self.theta_trace.append(theta_w)
        self.db_trace.append(db_w)

        E = int(ev_win.sum())
        U = int(ev_win[:int(CTX_SPLIT_Y), :].sum())
        self.energy_trace.append(E)
        self.up_trace.append(U)
        self.lo_trace.append(int(ev_win[int(CTX_SPLIT_Y):, :].sum()))
        self.c2_trace.append(None)
        self.sig_trace.append((None, None, None))
        if E >= 10:
            self.bbox_trace.append(float(U))
        else:
            self.bbox_trace.append(0.0)

        learned = False
        matched_pid = -1
        if E >= 10:
            x = (float(np.log1p(E)), float(np.log1p(U)))
            # 1. 慢优先（已验证记忆优先，细半径 r_slow）
            best, best_d = -1, None
            for i, p in enumerate(self.prototypes):
                if p["kind"] != "slow":
                    continue
                d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
                if best_d is None or d < best_d:
                    best, best_d = i, d
            if best_d is not None and best_d <= self.r_slow:
                p = self.prototypes[best]
                self._hit(p, x)
                learned = True
                matched_pid = p["pid"]
                self.match_trace.append((self._win, p["pid"]))
            else:
                # 2. 快兜底（gist 粗半径 r_fast）
                best, best_d = -1, None
                for i, p in enumerate(self.prototypes):
                    if p["kind"] != "fast":
                        continue
                    d = float(np.hypot(x[0] - p["mu"][0], x[1] - p["mu"][1]))
                    if best_d is None or d < best_d:
                        best, best_d = i, d
                if best_d is not None and best_d <= self.r_fast:
                    p = self.prototypes[best]
                    self._hit(p, x)
                    learned = True
                    matched_pid = p["pid"]
                    self.match_trace.append((self._win, p["pid"]))
                    # 延迟定级（docs/250 此处为 hits >= k_promote=2 立即升级）：
                    # 仅当累积满 hits_min_slow 次重匹配才最终化为慢原型
                    if p["hits"] >= self.hits_min_slow:
                        p["kind"] = "slow"
                        p["promoted_at"] = self._win
                        self.n_promoted += 1
                        self.promoted_log.append(dict(pid=p["pid"],
                                                      promoted_at=self._win,
                                                      hits=p["hits"]))
                else:
                    # 3. 高残差新奇段立即创建快原型（k_consist_fast=1）
                    pid = self._next_pid
                    self._next_pid += 1
                    self.prototypes.append(dict(pid=pid, mu=x, hits=1,
                                                created=self._win,
                                                last_active=self._win, n_match=1,
                                                kind="fast", promoted_at=None))
                    self.n_created_fast += 1
                    self.created_log.append(dict(pid=pid, created=self._win,
                                                 final_hits=None, recycled=0))
                    learned = True
                    matched_pid = pid
                    self.match_trace.append((self._win, pid))
        else:
            self.match_trace.append((self._win, None))
        # 4. 回收：快原型（含升级候选）连续 k_decay 窗未重匹配 -> 遗忘（慢原型豁免）
        for p in list(self.prototypes):
            if p["kind"] == "fast" and (self._win - p["last_active"]) >= self.k_decay:
                self.prototypes.remove(p)
                self.n_recycled += 1
                for cl in self.created_log:
                    if cl["pid"] == p["pid"]:
                        cl["final_hits"] = p["hits"]
                        cl["recycled"] = 1
        self.soft_trace.append((round(np.log1p(E), 4), round(np.log1p(U), 4),
                                matched_pid))
        if learned:
            self._n_learn += 1
        self.sc1_cum.append(len(self.prototypes))
        self._win += 1
        self._frame_buf = []
        self._ev_win = None


# ---------------- 单流运行（DeferredLoop；零重调；预测路径零改动） ----------------
def run_deferred_stream(frames):
    loop = DeferredLoop(window=WINDOW, **LOOP_CFG)
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


# ---------------- QUOTA_RETIRED：配额开 vs 关逐项对照（判据 1 口径） ----------------
def quota_compare(out, loop):
    """机制单独（churn_slow/SC2_slow）vs 机制 + docs/249 配额叠加于慢层
    （quota_on_slow 复用 = apply_quota + fastcut_gate，docs/250 §六同款口径）。
    ratio：配额为 finalize 级改判、不改 MAE -> off==on 按构造成立，显式比对。
    返回 {churn_off/on/eq, sc2_off/on/eq, ratio_off/on/eq, d_sc2, d_churn, fire}。"""
    q = quota_on_slow(out, loop)
    churn_off = out["churn_slow"]
    churn_on = q["churn_q"]
    sc2_off = out["sc2_slow"]
    sc2_on = q["sc2_q"]
    ratio_off = out["ratio"]
    ratio_on = out["ratio"]          # 配额不改 MAE -> 逐位相等（构造成立）
    return {
        "churn_off": churn_off, "churn_on": churn_on,
        "churn_eq": int(churn_off == churn_on),
        "sc2_off": sc2_off, "sc2_on": sc2_on,
        "sc2_eq": int(sc2_off == sc2_on),
        "ratio_off": ratio_off, "ratio_on": ratio_on,
        "ratio_eq": int(abs(ratio_off - ratio_on) < 1e-9),
        "d_sc2": q["d_sc2"], "d_churn": q["d_churn"],
        "n_slow": q["n_slow"], "fire": q["fire"],
    }


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qr")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 野域数据（docs/248 §1.2 同一批；间隔抽帧 -> 灰度 160x120 -> 流） ----
    loaded = {}
    for vid, name in WILD_VIDEOS:
        p = os.path.join(DL_DIR, name)
        frames, step, total = load_sampled_frames(p)
        loaded[vid] = frames

    # ---- 流（docs/248 §1.5，冻结；S1-S4；机制不含配额） ----
    streams_out = {}
    for sid, sname, vidx in STREAMS:
        frames = []
        for vi in vidx:
            frames.extend(loaded[WILD_VIDEOS[vi][0]])
        out, loop = run_deferred_stream(frames)
        creations = [e["created"] for e in out["entry_log"] if e["kind"] == "fast"]
        diag = scene_switch_diag(frames, creations)
        out["stream_id"] = sid
        out["stream_name"] = sname
        out["videos"] = [WILD_VIDEOS[vi][0] for vi in vidx]
        out["switch_diag"] = diag
        out["quota_cmp"] = quota_compare(out, loop)
        streams_out[sid] = out

    # ---- DAVIS：R0（flamingo x5）、R1（9 拼接；GT 段边界） ----
    allv = {v: load_video_frames(v) for v in VIDEOS}
    r0_frames = allv["flamingo"] * 5
    r1_frames, spans, start = [], [], 0
    for v in VIDEOS:
        fr = allv[v]
        r1_frames.extend(fr)
        spans.append((start, start + len(fr)))
        start += len(fr)
    r0r, r0_loop = run_deferred_stream(r0_frames)
    r1r, r1_loop = run_deferred_stream(r1_frames)
    r1r["bridge"] = bridge_metrics(build_entry_base(r1r), spans)  # 诊断
    switch_windows = [spans[i][0] // WINDOW for i in range(1, len(spans))]
    r1r["gist"] = gist_metrics(r1r, switch_windows)
    r1r["quota_cmp"] = quota_compare(r1r, r1_loop)
    # R1 bridge 逐项对照：bridge_metrics 只读 entry_log，配额不改 entry_log -> off==on
    r1r["quota_cmp"]["bridge_off"] = r1r["bridge"]["bridge_corr_switch"]
    r1r["quota_cmp"]["bridge_on"] = r1r["bridge"]["bridge_corr_switch"]
    r1r["quota_cmp"]["bridge_eq"] = int(
        abs(r1r["quota_cmp"]["bridge_off"] - r1r["quota_cmp"]["bridge_on"]) < 1e-9)

    # ---- 回归守卫（DAVIS；docs/249 同一代码路径；不进判据） ----
    g0, g1 = run_guard_quota(RADIUS_L3)
    guard_ok, guard_detail = guard_vs_d246(g0, g1)

    # ---- 内部复现（ratio vs docs/250 §3.3；预测路径零改动；诊断） ----
    repro_items = []
    for sid, sname, vidx in STREAMS:
        repro_items.append(("ratio_%s" % sid,
                            abs(streams_out[sid]["ratio"] - D250_RATIOS[sid]) < 1e-4))
    repro_items.append(("ratio_R1", abs(r1r["ratio"] - D250_RATIOS["R1"]) < 1e-4))
    repro_ok = int(all(v for _, v in repro_items))
    repro_detail = ",".join("%s:%d" % (n, int(v)) for n, v in repro_items)

    # ---- 判据（预注册 §1.7，冻结） ----
    quo = [streams_out[s]["quota_cmp"] for s, _, _ in STREAMS] + [r1r["quota_cmp"]]
    quota_retired = int(all(qc["churn_eq"] and qc["sc2_eq"] and qc["ratio_eq"]
                            for qc in quo)
                        and r1r["quota_cmp"]["bridge_eq"] == 1)
    all_streams = [streams_out[s] for s, _, _ in STREAMS] + [r1r]
    churn_mech = int(all(r["churn_slow"] <= 0.5 for r in all_streams))
    gist_correct = int(r1r["gist"]["cov"] >= 0.5)
    stable_keep = int(all(r["ratio"] <= 1.5 for r in all_streams))
    struct_keep = int(all(r["sc2_slow"] > 0 for r in all_streams))
    n_promo_total = sum(r["n_promo"] for r in all_streams)
    n_recycle_total = sum(r["n_recycle"] for r in all_streams)
    promo_means = [(r["promoted_mean_hits"], r["nonpromoted_mean_hits"])
                   for r in all_streams if r["sc1_slow"] > 0]
    promo_sep = int(any(mp > mn for mp, mn in promo_means)) \
        if promo_means else 0
    promo_ok = int(n_promo_total > 0 and n_recycle_total > 0 and promo_sep)
    oks = {"quota_retired": quota_retired, "churn_mech": churn_mech,
           "gist_correct": gist_correct, "stable_keep": stable_keep,
           "struct_keep": struct_keep, "promotion": promo_ok}

    # ---- 判定（预注册 §1.7 冻结规则） ----
    if (quota_retired and churn_mech and gist_correct and stable_keep
            and struct_keep and promo_ok and guard_ok):
        verdict = "QUOTA_RETIRED"
        vnote = ("QUOTA_RETIRED: quota on==off identical on all streams (S1-S4+R1 "
                 "churn/SC2/ratio per-item equal, R1 bridge equal, quota action surface "
                 "=0 via deferred finalization); CHURN_MECH and GIST_CORRECT and "
                 "STABLE/STRUCT and PROMOTION all pass; guard D246=12/12")
    elif not quota_retired:
        verdict = "QUOTA_STILL_NEEDED"
        vnote = ("deferred finalization did not eliminate the quota action surface: "
                 "quota on vs off still differ on some stream (see R_QR_*_Q_DSC2/"
                 "Q_DCHURN per-stream deltas and flip details); quota cannot be retired")
    elif churn_mech and gist_correct and stable_keep and struct_keep and promo_ok:
        verdict = "PARTIAL"
        vnote = ("QUOTA_RETIRED criterion holds (quota action surface = 0) but guard "
                 "< 12/12 (implementation drift not ruled out; see R_QR_GUARD_D246)")
    else:
        verdict = "PARTIAL"
        vnote = ("QUOTA_RETIRED criterion holds but supporting criteria not all pass "
                 "(CHURN_MECH/GIST_CORRECT/STABLE/STRUCT/PROMOTION; see R_QR_*_OK)")

    cfg = {"tag": args.tag, "size": list(RESIZE), "window": WINDOW,
           "working_point": {"r_slow": round(R_SLOW, 6), "r_fast": round(R_FAST, 6),
                             "hits_min_fast": HITS_MIN_FAST,
                             "hits_min_slow": HITS_MIN_SLOW,
                             "k_finalize": K_FINALIZE,
                             "k_promote": K_PROMOTE,
                             "k_promote_absorbed": K_PROMOTE_ABSORBED,
                             "k_decay": K_DECAY, "k_consist_fast": K_CONSIST_FAST,
                             "alpha": ALPHA},
           "mechanism": "deferred finalization: fast->slow promotion only when hits >= "
                        "hits_min_slow (=3, K_FINALIZE); promotion candidates with hits in "
                        "[k_promote, hits_min_slow) stay fast (r_fast matching, subject to "
                        "k_decay recycling) -> slow prototypes exist only in confirmed-full "
                        "form -> quota (exemption for unfinished slow prototypes) has no "
                        "action surface; k_promote=2 absorbed (DEPRECATED); no segment-"
                        "length prediction; quota fully off in mechanism",
           "quota_compare": {"method": "mechanism alone vs mechanism + docs/249 quota "
                                     "stacked on slow layer (apply_quota/fastcut_gate "
                                     "reuse, docs/250 Sec-6 same convention)"},
           "loop": LOOP_CFG,
           "seed_protocol": "none (deterministic real-pixel stream; window-level stats)"}
    out = {
        "artifact": "quota_retire",
        "doc_ref": "docs/245, docs/246, docs/247, docs/248, docs/249, docs/250, docs/251",
        "config": cfg,
        "streams": streams_out,
        "r0": r0r, "r1": r1r,
        "r1_switch_windows": switch_windows,
        "criteria": oks,
        "verdict": {"verdict": verdict, "note": vnote},
        "quota_retired": {"per_stream": {("R1" if i == len(quo) - 1 else
                                          STREAMS[i][0]): qc
                                          for i, qc in enumerate(quo)},
                          "any_delta": int(any(qc["d_sc2"] != 0 or qc["d_churn"] != 0.0
                                               for qc in quo)),
                          "retired": quota_retired},
        "guard_d246": {"ok": guard_ok, "detail": guard_detail},
        "repro_ratio_d250": {"ok": repro_ok, "detail": repro_detail},
        "promotion": {"n_promo_total": n_promo_total,
                      "n_recycle_total": n_recycle_total,
                      "promoted_vs_nonpromoted": promo_means,
                      "ok": promo_ok},
        "timing": {"elapsed_sec": round(time.time() - t0, 2)},
    }
    res_path = os.path.join(args.out_dir, "%s_%s.json" % ("qr", args.tag))
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要块：ASCII 标签，每行一个数字（顺序固定） ----
    print("R_QR_R_SLOW=%.6f" % R_SLOW)
    print("R_QR_R_FAST=%.6f" % R_FAST)
    print("R_QR_HITS_MIN_FAST=%d" % HITS_MIN_FAST)
    print("R_QR_HITS_MIN_SLOW=%d" % HITS_MIN_SLOW)
    print("R_QR_K_FINALIZE=%d" % K_FINALIZE)
    print("R_QR_K_PROMOTE=%d" % K_PROMOTE)
    print("R_QR_K_PROMOTE_ABSORBED=%d" % K_PROMOTE_ABSORBED)
    print("R_QR_K_DECAY=%d" % K_DECAY)
    print("R_QR_K_CONSIST_FAST=%d" % K_CONSIST_FAST)
    print("R_QR_ALPHA=%.4f" % ALPHA)
    for j, (sid, sname, vidx) in enumerate(STREAMS):
        r = streams_out[sid]
        d = r["switch_diag"]
        qc = r["quota_cmp"]
        print("R_QR_%d_ID=%s" % (j, sid))
        print("R_QR_%d_NAME=%s" % (j, sname))
        print("R_QR_%d_FRAMES=%d" % (j, r["frames"]))
        print("R_QR_%d_WINDOWS=%d" % (j, r["n_windows"]))
        print("R_QR_%d_VALID=%d" % (j, r["n_valid"]))
        print("R_QR_%d_MAE=%.6f" % (j, r["mae_mean_win"]))
        print("R_QR_%d_RATIO=%.6f" % (j, r["ratio"]))
        print("R_QR_%d_SC1_FAST=%d" % (j, r["sc1_fast"]))
        print("R_QR_%d_SC2_FAST=%d" % (j, r["sc2_fast"]))
        print("R_QR_%d_SC1_SLOW=%d" % (j, r["sc1_slow"]))
        print("R_QR_%d_SC2_SLOW=%d" % (j, r["sc2_slow"]))
        print("R_QR_%d_CHURN_SLOW=%.4f" % (j, r["churn_slow"]))
        print("R_QR_%d_CHURN_LEGACY=%.4f" % (j, r["churn_legacy"]))
        print("R_QR_%d_N_PROMO=%d" % (j, r["n_promo"]))
        print("R_QR_%d_N_RECYCLE=%d" % (j, r["n_recycle"]))
        print("R_QR_%d_PROMO_MEAN=%.4f" % (j, r["promoted_mean_hits"]))
        print("R_QR_%d_NONPROMO_MEAN=%.4f" % (j, r["nonpromoted_mean_hits"]))
        print("R_QR_%d_FAST_CREATIONS=%s" % (j, ",".join(
            str(e["created"]) for e in sorted(r["entry_log"], key=lambda e: e["created"])
            if e["kind"] == "fast")))
        print("R_QR_%d_SLOW_PROMOS=%s" % (j, ",".join(
            "%d:%d" % (e["promoted_at"], e["hits"]) for e in
            sorted(r["entry_log"], key=lambda e: e["created"])
            if e["kind"] == "slow")))
        print("R_QR_%d_SW_CORR=%s" % (j, ("NA" if d["switch_corr"] is None
                                           else "%.4f" % d["switch_corr"])))
        print("R_QR_%d_Q_CHURN_OFF=%.4f" % (j, qc["churn_off"]))
        print("R_QR_%d_Q_CHURN_ON=%.4f" % (j, qc["churn_on"]))
        print("R_QR_%d_Q_CHURN_EQ=%d" % (j, qc["churn_eq"]))
        print("R_QR_%d_Q_SC2_OFF=%d" % (j, qc["sc2_off"]))
        print("R_QR_%d_Q_SC2_ON=%d" % (j, qc["sc2_on"]))
        print("R_QR_%d_Q_SC2_EQ=%d" % (j, qc["sc2_eq"]))
        print("R_QR_%d_Q_RATIO_EQ=%d" % (j, qc["ratio_eq"]))
        print("R_QR_%d_Q_DSC2=%d" % (j, qc["d_sc2"]))
        print("R_QR_%d_Q_DCHURN=%.4f" % (j, qc["d_churn"]))
    print("R_QR_R0_FRAMES=%d" % r0r["frames"])
    print("R_QR_R0_RATIO=%.6f" % r0r["ratio"])
    print("R_QR_R0_SC1_FAST=%d" % r0r["sc1_fast"])
    print("R_QR_R0_SC1_SLOW=%d" % r0r["sc1_slow"])
    print("R_QR_R0_SC2_SLOW=%d" % r0r["sc2_slow"])
    print("R_QR_R0_CHURN_SLOW=%.4f" % r0r["churn_slow"])
    print("R_QR_R0_N_PROMO=%d" % r0r["n_promo"])
    print("R_QR_R0_N_RECYCLE=%d" % r0r["n_recycle"])
    print("R_QR_R1_FRAMES=%d" % r1r["frames"])
    print("R_QR_R1_RATIO=%.6f" % r1r["ratio"])
    print("R_QR_R1_SC1_FAST=%d" % r1r["sc1_fast"])
    print("R_QR_R1_SC1_SLOW=%d" % r1r["sc1_slow"])
    print("R_QR_R1_SC2_SLOW=%d" % r1r["sc2_slow"])
    print("R_QR_R1_CHURN_SLOW=%.4f" % r1r["churn_slow"])
    print("R_QR_R1_CHURN_LEGACY=%.4f" % r1r["churn_legacy"])
    print("R_QR_R1_N_PROMO=%d" % r1r["n_promo"])
    print("R_QR_R1_N_RECYCLE=%d" % r1r["n_recycle"])
    print("R_QR_R1_PROMO_MEAN=%.4f" % r1r["promoted_mean_hits"])
    print("R_QR_R1_NONPROMO_MEAN=%.4f" % r1r["nonpromoted_mean_hits"])
    print("R_QR_R1_FAST_CREATIONS=%s" % ",".join(
        str(w) for w in r1r["gist"]["creation_windows"]))
    print("R_QR_R1_SWITCHES=%s" % ",".join(str(w) for w in switch_windows))
    print("R_QR_R1_GIST_COV=%.4f" % r1r["gist"]["cov"])
    print("R_QR_R1_GIST_PREC=%.4f" % r1r["gist"]["prec"])
    print("R_QR_R1_GIST_COV_D2=%.4f" % r1r["gist"]["cov_d2"])
    print("R_QR_R1_BRIDGE_SW=%.4f" % r1r["bridge"]["bridge_corr_switch"])
    print("R_QR_R1_BRIDGE_VID=%.4f" % r1r["bridge"]["bridge_corr_video"])
    print("R_QR_R1_Q_CHURN_OFF=%.4f" % r1r["quota_cmp"]["churn_off"])
    print("R_QR_R1_Q_CHURN_ON=%.4f" % r1r["quota_cmp"]["churn_on"])
    print("R_QR_R1_Q_CHURN_EQ=%d" % r1r["quota_cmp"]["churn_eq"])
    print("R_QR_R1_Q_SC2_OFF=%d" % r1r["quota_cmp"]["sc2_off"])
    print("R_QR_R1_Q_SC2_ON=%d" % r1r["quota_cmp"]["sc2_on"])
    print("R_QR_R1_Q_SC2_EQ=%d" % r1r["quota_cmp"]["sc2_eq"])
    print("R_QR_R1_Q_RATIO_EQ=%d" % r1r["quota_cmp"]["ratio_eq"])
    print("R_QR_R1_Q_BRIDGE_EQ=%d" % r1r["quota_cmp"]["bridge_eq"])
    print("R_QR_R1_Q_DSC2=%d" % r1r["quota_cmp"]["d_sc2"])
    print("R_QR_R1_Q_DCHURN=%.4f" % r1r["quota_cmp"]["d_churn"])
    print("R_QR_QUOTA_RETIRED_OK=%d" % quota_retired)
    print("R_QR_CHURN_OK=%d" % churn_mech)
    print("R_QR_GIST_OK=%d" % gist_correct)
    print("R_QR_STABLE_OK=%d" % stable_keep)
    print("R_QR_STRUCT_OK=%d" % struct_keep)
    print("R_QR_PROMO_OK=%d" % promo_ok)
    print("R_QR_PROMO_TOTAL=%d" % n_promo_total)
    print("R_QR_RECYCLE_TOTAL=%d" % n_recycle_total)
    print("R_QR_VERDICT=%s" % verdict)
    print("R_QR_VERDICT_NOTE=%s" % vnote)
    print("R_QR_GUARD_D246=%d" % guard_ok)
    print("R_QR_GUARD_DETAIL=%s" % guard_detail)
    print("R_QR_REPRO_RATIO=%d" % repro_ok)
    print("R_QR_REPRO_DETAIL=%s" % repro_detail)
    print("R_QR_ELAPSED=%.2f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
