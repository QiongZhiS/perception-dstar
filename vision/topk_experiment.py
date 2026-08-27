"""实验：给定维持目标后，Top-K 威胁度分配能否过滤误报（docs/186 §五最小验证）。

问题：DAVIS car-turn 误报 61/帧——五个框特征维度全不可分（docs/187/186 结论：
"值得注意"是关系属性，需要维持目标）。本实验外部给定维持目标（第一帧掩码指定车），
威胁度 = 候选框与维持目标的关系，Top-K 保留——验证利害给定后威胁度分配本身可行。

威胁度定义（最小版）：
  threat(box) = IoU(box, 目标预测框)          # 空间接近：与维持目标重合/竞争的落空
              + α × 方向趋近                 # 向目标运动的落空（可能威胁目标）
  目标预测框 = 目标位置 + 速度外推（跟踪维持）

度量：
  - 目标检出率：Top-K 后维持目标框还在（目标没被误报挤掉）
  - 每帧框数：误报代理（无 Top-K 时 61/帧）
  对照：K=∞（不分配） vs K=3/5/10

用法：python vision/topk_experiment.py
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2, json
from transduction import Transduction2D
from reader import EventReader
from davis_check import mask_boxes

vdir = os.path.join('vision', 'out', 'davis', 'car-turn')
jpgs = sorted(f for f in os.listdir(vdir) if f.endswith('.jpg'))
td = Transduction2D(thresh=0.35, deadband=0.06, blur=1.6)
rf = EventReader(window=2, thr=0.25, min_area=12, blur=1.2, min_track_age=0)
rs = EventReader(window=10, thr=0.12, min_area=12, blur=1.5, min_track_age=0)

# ---- 第一帧：从掩码指定维持目标 ----
fr0 = cv2.imread(os.path.join(vdir, jpgs[0]))
mask0 = cv2.imread(os.path.join(vdir, jpgs[0].replace('.jpg', '.png')), 0)
gt0 = mask_boxes(mask0)
assert gt0, "第一帧无真值物体"
oid, tx, ty, tw, th = gt0[0]          # 维持目标 = 第一个物体
target = {"x": tx, "y": ty, "w": tw, "h": th, "cx": tx + tw / 2, "cy": ty + th / 2,
          "vx": 0.0, "vy": 0.0}
print(f"维持目标: 物体#{oid} at ({tx},{ty},{tw}x{th})")


def target_pred():
    """目标预测框（位置+速度外推）。"""
    return {"x": target["cx"] + target["vx"] - target["w"] / 2,
            "y": target["cy"] + target["vy"] - target["h"] / 2,
            "w": target["w"], "h": target["h"]}


def threat_score(b, tpred):
    """威胁度 = 与目标预测框的 IoU + α×方向趋近（向目标运动加分）。"""
    ax1, ay1 = b["x"], b["y"]
    ax2, ay2 = b["x"] + b["w"], b["y"] + b["h"]
    bx1, by1 = tpred["x"], tpred["y"]
    bx2, by2 = tpred["x"] + tpred["w"], tpred["y"] + tpred["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    iou = inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    # 方向趋近：候选框运动方向是否指向目标
    t = next((t for t in rf.tracks + rs.tracks if t["id"] == b["track"]), None)
    approach = 0.0
    if t is not None:
        vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
        sp = (vx ** 2 + vy ** 2) ** 0.5
        if sp > 1.0:
            dx, dy = tpred["x"] + tpred["w"] / 2 - b["cx"], tpred["y"] + tpred["h"] / 2 - b["cy"]
            dd = (dx ** 2 + dy ** 2) ** 0.5
            if dd > 1e-6:
                approach = max(0.0, (vx * dx + vy * dy) / (sp * dd))   # 余弦相似度
    return iou + 0.3 * approach


def run(K):
    """K=0 表示不分配（全部保留）。返回 (目标检出帧数, 每帧框数列表, 目标轨迹误差)。"""
    rf2 = EventReader(window=2, thr=0.25, min_area=12, blur=1.2, min_track_age=0)
    rs2 = EventReader(window=10, thr=0.12, min_area=12, blur=1.5, min_track_age=0)
    td2 = Transduction2D(thresh=0.35, deadband=0.06, blur=1.6)
    tgt = dict(target)                       # 重置目标
    tgt["vx"] = tgt["vy"] = 0.0
    hit_frames = 0
    total_frames = 0
    n_boxes = []
    errs = []
    for jpg in jpgs:
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace('.jpg', '.png')), 0)
        if fr is None or mask is None:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ef, es, _, _ = td2.step(g)
        fl = rf2.is_flash(ef, es)
        boxes = EventReader.merge_boxes(rf2.feed(ef, es, flash=fl) + rs2.feed(ef, es, flash=fl))
        total_frames += 1
        tpred = {"x": tgt["cx"] + tgt["vx"] - tgt["w"] / 2,
                 "y": tgt["cy"] + tgt["vy"] - tgt["h"] / 2,
                 "w": tgt["w"], "h": tgt["h"]}
        if K > 0 and boxes:
            scored = sorted(boxes, key=lambda b: threat_score(b, tpred), reverse=True)
            boxes = scored[:K]
        n_boxes.append(len(boxes))
        # 目标是否被框住（任一保留框与目标预测 IoU>0.2 或近）
        hit = False
        best_err = 1e9
        for b in boxes:
            err = ((b["cx"] - tgt["cx"]) ** 2 + (b["cy"] - tgt["cy"]) ** 2) ** 0.5
            best_err = min(best_err, err)
            ax1, ay1 = b["x"], b["y"]
            ax2, ay2 = b["x"] + b["w"], b["y"] + b["h"]
            ix1, iy1 = max(ax1, tpred["x"]), max(ay1, tpred["y"])
            ix2, iy2 = min(ax2, tpred["x"] + tpred["w"]), min(ay2, tpred["y"] + tpred["h"])
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            iou = inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) +
                              (tpred["w"] * tpred["h"]) - inter)
            if iou > 0.2 or err < 25:
                hit = True
        if hit:
            hit_frames += 1
        errs.append(best_err)
        # 更新目标：用掩码真值更新位置（维持目标 = 真值引导的跟踪）+ 速度
        gtm = mask_boxes(mask)
        if gtm:
            goid, gx, gy, gw, gh = gtm[0]
            ncx, ncy = gx + gw / 2, gy + gh / 2
            tgt["vx"] = 0.7 * (ncx - tgt["cx"]) + 0.3 * tgt["vx"]
            tgt["vy"] = 0.7 * (ncy - tgt["cy"]) + 0.3 * tgt["vy"]
            tgt.update(cx=ncx, cy=ncy, x=gx, y=gy, w=gw, h=gh)
    return hit_frames / max(total_frames, 1), np.mean(n_boxes), np.median(errs)


print("\n== 结果 ==")
print(f"{'K':>4s} {'目标检出率':>10s} {'每帧框数':>9s} {'质心误差中位':>12s}")
results = {}
for K in [0, 10, 5, 3]:
    rate, nb, err = run(K)
    results[str(K)] = {"detect": round(rate, 3), "boxes_per_frame": round(nb, 2),
                       "err_median": round(err, 1)}
    print(f"{K:>4d} {rate:10.3f} {nb:9.1f} {err:12.1f}")

out = os.path.join("vision", "out", "topk_experiment.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"video": "car-turn", "target": f"obj#{oid}", "results": results},
              f, ensure_ascii=False, indent=1)
print(f"\n→ {out}")
print("\n解读：K=0 是基线（不分配）。若 K=3/5/10 时目标检出率保持而每帧框数大幅下降，")
print("则威胁度分配可行——利害（维持目标）给定后，系统只看该看的。")
