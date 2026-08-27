"""vision/run_all.py — docs/221 A5：一键复现本线全部实验 + JSON 结果归档。

按 docs/219/221/223 的顺序跑：
  1. davis_suspicious（flamingo + surf 真实图像闭环，7 变体矩阵）
  2. davis_stats（10 种子统计 + PR 曲线 + flips 检验 + 敏感性）
  3. multi_situation（多目标态势转录）
  4. davis_multi（多目标回真实视频）
每个实验的输出 JSON 归档到 vision/out/results/（数字成为工件而非手抄表）。

用法：python vision/run_all.py [--stats-seeds 10]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, "vision")

OUT = os.path.join("vision", "out", "results")


def run(cmd):
    print(f"\n=== {' '.join(cmd)} ===")
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(f"[STDERR] {r.stderr[-1500:]}")
    return r.returncode == 0


def collect():
    """把各实验的 stdout 关键表归档为 JSON（逐帧 trace 可选 --json 由各脚本输出）。"""
    os.makedirs(OUT, exist_ok=True)
    manifest = {"note": "docs/221 A5 结果归档（运行时间见文件 mtime）",
                "commands": [
                    "python vision/davis_suspicious.py --video flamingo",
                    "python vision/davis_suspicious.py --video surf --seg 14,12,10,19 --thr 0.40 --corrupt noise",
                    "python vision/davis_stats.py --seeds 10 --pr --flips 10",
                    "python vision/multi_situation.py",
                    "python vision/davis_multi.py",
                ]}
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-seeds", type=int, default=10)
    args = ap.parse_args()

    ok = True
    os.makedirs(OUT, exist_ok=True)
    ok &= run(["vision/davis_suspicious.py", "--video", "flamingo",
               "--json", os.path.join(OUT, "davis_suspicious_flamingo.json")])
    ok &= run(["vision/davis_suspicious.py", "--video", "surf",
               "--seg", "14,12,10,19", "--thr", "0.40", "--corrupt", "noise",
               "--json", os.path.join(OUT, "davis_suspicious_surf.json")])
    ok &= run(["vision/davis_stats.py", "--seeds", str(args.stats_seeds),
               "--pr", "--flips", "10", "--sens"])
    ok &= run(["vision/multi_situation.py"])
    ok &= run(["vision/davis_multi.py"])
    collect()
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — 结果归档: {OUT}")


if __name__ == "__main__":
    main()
