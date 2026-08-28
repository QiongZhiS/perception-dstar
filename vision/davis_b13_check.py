"""vision/davis_b13_check.py — docs/234 B1.3 记忆持久化 round-trip 验证（安全模式）。

验证内容（docs/221 B1.3 + docs/220 未完成项 2 + docs/203 阅历跨会话工程化）：

  RT1 行为等价（"没重启过"）：
      单进程跑全序列  vs  两会话拆分（会话1 跑 base+dusk+corrupt 并落盘 →
      会话2 从落盘 JSON 加载续跑 rest）→ 逐帧 claimed 序列逐帧一致，
      P/R/F1/flips 指标一致。
  RT2 状态无损：加载后的 suspicious/thr_by/hist/纪念/门/白平衡基线 与保存时
      逐项相等（JSON 序列化往返无损）。
  RT3 阅历真实落盘：保存时 suspicious 表非空、rej_total>0（不是空状态空转）。

安全纪律（docs/230/234）：stdout 只含 ASCII 标签 + 每行一个数字（R_B13_*），
外部用纯 python 正则抽取（vision/extract_r.py）。

用法（从仓库根）：
  python vision/davis_b13_check.py --video flamingo --seed 7 --thr 0.60 \
      --corrupt gain --seg 20,20,15,25 --tag FLAMINGO \
      --state vision/out/state/b13_rt_flamingo.json
  python vision/davis_b13_check.py --video surf --seed 7 --thr 0.40 \
      --corrupt noise --seg 14,12,10,19 --tag SURF \
      --state vision/out/state/b13_rt_surf.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "vision")
import davis_suspicious as ds  # noqa: E402


def rt(video, seed, thr, corrupt, seg, state_path, wb=True):
    """一次 round-trip 验证。返回全部数字（dict）。"""
    frames, masks = ds.load_video(video)
    obs = [ds.target_obs(f, m, 0.0)[1] for f, m in zip(frames, masks)]
    task_hue = float(np.median([h for h in obs if h is not None]))
    seq_f, seq_m, seq_s, seq_gt = ds.build_sequence(frames, masks, seg,
                                                    seed=seed, corrupt=corrupt)
    n_pre = seg["base"] + seg["dusk"] + seg["corrupt"]   # 会话1 帧数（阅历沉积后断点）
    cfg = {"wb": wb}

    # 会话0：全序列单进程（没重启过）
    m_full = ds.DavisMind(task_hue, thr_base=thr, **cfg)
    oks_full, _, _, _ = ds.run_mind(m_full, seq_f, seq_m, seq_s, task_hue, video)

    # 会话1：前半段 → 落盘
    m_pre = ds.DavisMind(task_hue, thr_base=thr, **cfg)
    oks_pre, _, _, _ = ds.run_mind(m_pre, seq_f[:n_pre], seq_m[:n_pre],
                                   seq_s[:n_pre], task_hue, video)
    ds.save_state(state_path, {"A": m_pre}, meta={"video": video, "seed": seed,
                                                  "thr": thr, "corrupt": corrupt,
                                                  "split": n_pre, "check": "b13"})

    # 会话2（"重启"）：从落盘 JSON 加载 → 前半段续跑
    m_suf = ds.load_state(state_path, {"A": cfg})["A"]
    # 会话边界快照：加载后立刻对比（与保存时状态逐项相等才算无损；续跑会改动状态）
    g1, g2 = m_pre.gate, m_suf.gate
    gate_eq = (g1 is None and g2 is None) or (
        g1 is not None and g2 is not None and g1.n == g2.n and
        g1.d_mu == g2.d_mu and g1.prev == g2.prev)
    gb_eq = (m_pre.g_base is None and m_suf.g_base is None) or (
        m_pre.g_base is not None and m_suf.g_base is not None and
        bool(np.array_equal(m_pre.g_base, m_suf.g_base)))
    state_eq = (m_suf.rejected == m_pre.rejected and
                m_suf.thr_by == m_pre.thr_by and
                m_suf.rej_total == m_pre.rej_total and
                m_suf.hist == m_pre.hist and
                m_suf.trusted == m_pre.trusted and
                m_suf.consec == m_pre.consec and
                m_suf.persist == m_pre.persist and
                m_suf.thr_cur == m_pre.thr_cur and
                m_suf.n_corr == m_pre.n_corr and
                gate_eq and gb_eq)
    oks_suf, _, _, _ = ds.run_mind(m_suf, seq_f[n_pre:], seq_m[n_pre:],
                                   seq_s[n_pre:], task_hue, video)

    combined = list(oks_pre) + list(oks_suf)
    seq_eq = combined == list(oks_full)
    first_diff = -1
    for i, (a, b) in enumerate(zip(combined, oks_full)):
        if a != b:
            first_diff = i
            break
    m_split = ds.metrics(combined, seq_gt)
    m_full_m = ds.metrics(oks_full, seq_gt)

    return {
        "ntotal": len(seq_gt), "npre": n_pre, "npos": int(sum(seq_gt)),
        "task_hue": task_hue,
        "seq_eq": int(seq_eq), "first_diff": first_diff,
        "f1_full": m_full_m["f1"], "f1_split": m_split["f1"],
        "p_full": m_full_m["p"], "p_split": m_split["p"],
        "r_full": m_full_m["r"], "r_split": m_split["r"],
        "flips_full": int(ds.flips(oks_full, seq_s)),
        "flips_split": int(ds.flips(combined, seq_s)),
        "state_eq": int(state_eq), "gate_eq": int(gate_eq),
        "rej_total": int(m_pre.rej_total),
        "susp_keys": len(m_pre.rejected), "thrby_keys": len(m_pre.thr_by),
        "state_bytes": os.path.getsize(state_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--corrupt", default="gain", choices=["gain", "noise"])
    ap.add_argument("--seg", default="20,20,15,25")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--state", required=True)
    args = ap.parse_args()
    seg = dict(zip(["base", "dusk", "corrupt", "rest"],
                   [int(x) for x in args.seg.split(",")]))
    r = rt(args.video, args.seed, args.thr, args.corrupt, seg, args.state)
    # 摘要块：ASCII 标签 + 每行一个数字（顺序见模块 docstring；外部正则抽取）
    print("R_B13_TAG=%s" % args.tag)
    print("R_B13_VIDEO=%s" % args.video)
    print("R_B13_SEED=%d" % args.seed)
    print("R_B13_TASKHUE=%.3f" % r["task_hue"])
    print("R_B13_NTOTAL=%d" % r["ntotal"])
    print("R_B13_NPRE=%d" % r["npre"])
    print("R_B13_NPOS=%d" % r["npos"])
    print("R_B13_SEQ_EQ=%d" % r["seq_eq"])
    print("R_B13_FIRST_DIFF=%d" % r["first_diff"])
    print("R_B13_F1_FULL=%.6f" % r["f1_full"])
    print("R_B13_F1_SPLIT=%.6f" % r["f1_split"])
    print("R_B13_P_FULL=%.6f" % r["p_full"])
    print("R_B13_P_SPLIT=%.6f" % r["p_split"])
    print("R_B13_R_FULL=%.6f" % r["r_full"])
    print("R_B13_R_SPLIT=%.6f" % r["r_split"])
    print("R_B13_FLIPS_FULL=%d" % r["flips_full"])
    print("R_B13_FLIPS_SPLIT=%d" % r["flips_split"])
    print("R_B13_STATE_EQ=%d" % r["state_eq"])
    print("R_B13_GATE_EQ=%d" % r["gate_eq"])
    print("R_B13_REJ_TOTAL=%d" % r["rej_total"])
    print("R_B13_SUSP_KEYS=%d" % r["susp_keys"])
    print("R_B13_THRBY_KEYS=%d" % r["thrby_keys"])
    print("R_B13_STATE_BYTES=%d" % r["state_bytes"])


if __name__ == "__main__":
    main()
