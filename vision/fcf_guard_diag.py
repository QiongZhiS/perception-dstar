"""vision/fcf_guard_diag.py — docs/249 守卫路径快切门状态诊断（仅 DAVIS R0/R1；只打印
R_FCFG_* 摘要行；不运行任何野域实验流，不改任何参数；纯信息性，不进判据）。

复跑 docs/249 §1.7-1 守卫的 DAVIS R0/R1（同一 run_guard_quota 代码路径），报告快切门
是否触发与配额是否翻转原型——用于诚实核对预注册 §1.4"门在 DAVIS R0/R1 不触发"的期望。
"""
import sys

sys.path.insert(0, "vision")

from fastcut_fix import run_guard_quota, RADIUS_L3
from cross_domain_test import guard_vs_d246

g0, g1 = run_guard_quota(RADIUS_L3)
for tag, r in (("R0", g0), ("R1", g1)):
    print("R_FCFG_%s_GATE=%d" % (tag, r["gate"]["fire"]))
    print("R_FCFG_%s_FCF=%.4f" % (tag, r["gate"]["fcf"]))
    print("R_FCFG_%s_NN=%.4f" % (tag, r["gate"]["nn_median"]))
    print("R_FCFG_%s_GATE_RATIO_VAL=%.4f" % (tag, r["gate"]["ratio"]))
    print("R_FCFG_%s_FLIPPED=%d" % (tag, r["quota_diag"]["flipped"]))
    print("R_FCFG_%s_SC2=%d" % (tag, r["sc2"]))
    print("R_FCFG_%s_SC2_Q=%d" % (tag, r["sc2_q"]))
    print("R_FCFG_%s_CHURN=%.4f" % (tag, r["churn_frac"]))
    print("R_FCFG_%s_CHURN_Q=%.4f" % (tag, r["churn_q"]))
ok, detail = guard_vs_d246(g0, g1)
print("R_FCFG_D246_OK=%d" % ok)
print("R_FCFG_D246_DETAIL=%s" % detail)
sys.exit(0)
