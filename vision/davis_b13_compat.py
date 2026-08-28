"""vision/davis_b13_compat.py — docs/234 向后兼容验证（安全模式，纯 python 数字抽取）。

对 davis_suspicious.py 的新旧路径产物做数字级核对（绝不回显原始内容）：

  C1 主脚本 --json 归档不变：b13_plain.json 的 A full f1 应 ≈ docs/219 参考值
     0.8269（flamingo seed 7；圆整后 0.83）。
  C2 --load-state 端到端：b13_loaded.json 有效且 A full f1 数值可读（加载阅历后
     指标与全新运行不同是预期——阅历改变行为；等价性由 davis_b13_check.py 负责）。
  C3 状态文件结构：vision/out/state/flamingo_s7_gain.json 含 7 个变体 key；
     A 快照带非空 suspicious 表、rej_total>0、门与白平衡基线（自包含快照）。

安全纪律（docs/230/234）：stdout 只含 ASCII 标签 + 每行一个数字（R_B13C_*）。

用法（从仓库根，在运行 b13_plain/b13_loaded/--save-state 之后）：
  python vision/davis_b13_compat.py
"""
import json
import os
import sys

sys.path.insert(0, "vision")
import davis_suspicious as ds  # noqa: E402

PLAIN = os.path.join("vision", "out", "results", "b13_plain.json")
LOADED = os.path.join("vision", "out", "results", "b13_loaded.json")
STATE = ds.default_state_path("flamingo", 7, "gain")


def _a_f1(path):
    with open(path, encoding="utf-8") as f:
        out = json.load(f)
    key = "A full      (wb, 类别级, 自适应带宽)"
    v = out["variants"].get(key) or out["variants"].get(key.strip())
    return float(v["f1"])


def main():
    plain_ok = os.path.exists(PLAIN)
    loaded_ok = os.path.exists(LOADED)
    state_ok = os.path.exists(STATE)
    plain_f1 = _a_f1(PLAIN) if plain_ok else float("nan")
    loaded_f1 = _a_f1(LOADED) if loaded_ok else float("nan")
    ref = 0.826923
    c1 = 1 if (plain_ok and abs(plain_f1 - ref) < 1e-4) else 0
    n_keys, a_rej, a_rejtotal, a_gate, a_gbase = 0, 0, 0, 0, 0
    if state_ok:
        with open(STATE, encoding="utf-8") as f:
            st = json.load(f)
        n_keys = len(st.get("variants", {}))
        a = st.get("variants", {}).get("A", {})
        a_rej = len(a.get("rejected", {}))
        a_rejtotal = int(a.get("rej_total", 0))
        a_gate = 1 if a.get("gate") is not None else 0
        a_gbase = 1 if a.get("g_base") is not None else 0
    c3 = 1 if (state_ok and n_keys == 7 and a_rej > 0 and a_rejtotal > 0
               and a_gate == 1 and a_gbase == 1) else 0
    print("R_B13C_PLAIN_F1=%.6f" % plain_f1)
    print("R_B13C_PLAIN_REF_MATCH=%d" % c1)
    print("R_B13C_LOADED_F1=%.6f" % loaded_f1)
    print("R_B13C_LOADED_OK=%d" % loaded_ok)
    print("R_B13C_STATE_KEYS=%d" % n_keys)
    print("R_B13C_STATE_A_REJKEYS=%d" % a_rej)
    print("R_B13C_STATE_A_REJTOTAL=%d" % a_rejtotal)
    print("R_B13C_STATE_A_GATE=%d" % a_gate)
    print("R_B13C_STATE_A_GBASE=%d" % a_gbase)
    print("R_B13C_STATE_OK=%d" % c3)


if __name__ == "__main__":
    main()
