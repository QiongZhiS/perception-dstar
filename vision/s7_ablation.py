"""vision/s7_ablation.py — S7 侧抑制消融：侧抑制进事件路径 vs 不进（docs/133 §2.2）。

视网膜中心-周围 = 预测残差编码（Srinivasan 1982）：侧抑制压掉均匀区（空间冗余）、
增强边缘。S7 消融：Transduction2D(lateral_in_path=True) 让空间预测误差参与事件触发，
对照不进（当前默认）。测：
  1. toy（demo_storm）：C1 静态事件率 / C4 稀疏度 / 风暴段行为（事件更干净？）
  2. DAVIS car-turn：读层检出率 / 误报（侧抑制让事件更聚焦物体？）
判据（docs/183 S7）：侧抑制进路径 → C4 降低（更稀疏）且 C1 不升（静态更诚实）
或读层误报下降，d* 方向改善。

用法：python vision/s7_ablation.py
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2
from transduction import Transduction2D, make_demo_video
from reader import EventReader
from davis_check import mask_boxes


def run_toy(storm_sigma=12.0):
    src = os.path.join("vision", "out", "demo_storm.mp4")
    if not os.path.exists(src):
        make_demo_video(src, storm=True, storm_sigma=storm_sigma)
    results = {}
    for lateral in (False, True):
        td = Transduction2D(thresh=0.35, deadband=0.06, blur=1.6,
                            lateral_in_path=lateral)
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        static = np.zeros((240, 320), bool)
        static[5:25, 5:35] = True
        c1s, sps, t = [], [], 0.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ef, es, _, _ = td.step(gray)
            ev = np.where((ef != 0) | (es != 0), 1, 0)
            sps.append(ev.mean())
            c1s.append(ev[static].mean())
            t += 1 / fps
        cap.release()
        sps, c1s = np.array(sps), np.array(c1s)
        storm = np.arange(len(sps)) / fps
        m_storm = (storm >= 4.0) & (storm < 4.9)
        m_calm = ~((storm >= 4.0) & (storm < 4.9)) & ~(
            ((storm >= 3.0) & (storm < 3.5)) | ((storm >= 5.0) & (storm < 5.3)))
        results[f"lateral={lateral}"] = {
            "storm_c1": float(c1s[m_storm].mean()),
            "storm_sparse": float(sps[m_storm].mean()),
            "calm_c1": float(c1s[m_calm].mean()),
            "calm_sparse": float(sps[m_calm].mean()),
        }
    return results


def run_davis():
    vdir = os.path.join("vision", "out", "davis", "car-turn")
    if not os.path.isdir(vdir):
        return {}
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    results = {}
    for lateral in (False, True):
        td = Transduction2D(thresh=0.35, deadband=0.06, blur=1.6,
                            lateral_in_path=lateral)
        rf = EventReader(window=2, thr=0.25, min_area=12, blur=1.2, min_track_age=0)
        rs = EventReader(window=10, thr=0.12, min_area=12, blur=1.5, min_track_age=0)
        hits = gt_total = fp = 0
        for jpg in jpgs:
            fr = cv2.imread(os.path.join(vdir, jpg))
            mask = cv2.imread(os.path.join(vdir, jpg.replace(".jpg", ".png")), 0)
            if fr is None or mask is None:
                continue
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ef, es, _, _ = td.step(gray)
            fl = rf.is_flash(ef, es)
            boxes = EventReader.merge_boxes(rf.feed(ef, es, flash=fl) +
                                            rs.feed(ef, es, flash=fl))
            gt = mask_boxes(mask)
            matched = [False] * len(gt)
            for b in boxes:
                bg_i, bv = None, 0.2
                for gi, gtb in enumerate(gt):
                    if matched[gi]:
                        continue
                    ax1, ay1 = b["x"], b["y"]
                    ax2, ay2 = b["x"] + b["w"], b["y"] + b["h"]
                    bx1, by1 = gtb[1], gtb[2]
                    bx2, by2 = gtb[1] + gtb[3], gtb[2] + gtb[4]
                    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    iou = inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) +
                                      (bx2 - bx1) * (by2 - by1) - inter)
                    gc = ((b["cx"] - (gtb[1] + gtb[3] / 2)) ** 2 +
                          (b["cy"] - (gtb[2] + gtb[4] / 2)) ** 2) ** 0.5
                    if iou > bv or (iou > 0.15 and gc < 25):
                        bg_i, bv = gi, iou
                if bg_i is not None:
                    matched[bg_i] = True
                    hits += 1
                elif not b.get("predicted"):
                    fp += 1
            gt_total += len(gt)
        n = max(len(jpgs), 1)
        results[f"lateral={lateral}"] = {
            "detect": hits / max(gt_total, 1), "fp_per_frame": fp / n}
    return results


def main():
    print("== S7 侧抑制消融（lateral_in_path：侧抑制残差进事件路径）==")
    toy = run_toy()
    print("\n-- toy（demo_storm，θ=0.35/blur=1.6）--")
    print(f"{'臂':22s} {'风暴C1':>8s} {'风暴稀疏':>8s} {'平静C1':>8s} {'平静稀疏':>8s}")
    for k, v in toy.items():
        print(f"{k:22s} {v['storm_c1']:8.5f} {v['storm_sparse']:8.5f} "
              f"{v['calm_c1']:8.5f} {v['calm_sparse']:8.5f}")
    davis = run_davis()
    if davis:
        print("\n-- DAVIS car-turn（读层）--")
        print(f"{'臂':22s} {'检出率':>8s} {'误报/帧':>8s}")
        for k, v in davis.items():
            print(f"{k:22s} {v['detect']:8.3f} {v['fp_per_frame']:8.1f}")
    print("\n判据（docs/183 S7）：侧抑制进路径 → 事件更干净（C4 降）且静态更诚实（C1 不升），"
          "或读层误报下降。")

    import json
    out = os.path.join("vision", "out", "s7_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"toy": toy, "davis": davis}, f, ensure_ascii=False, indent=1)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
