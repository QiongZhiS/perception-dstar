"""vision/reader.py — 读信息：事件流 → 物体包围框 + 跟踪（含闪变预测维持）。

三段 + 两个鲁棒机制：
  1. 事件密度图（EMA）→ 平滑 → 阈值 → 连通域 → 框
  2. 跟踪（质心最近关联 + 速度估计）
  3. 闪变模式（人眼眨眼/扫视同款）：全局亮度突变淹没局部信号时，
     用轨迹速度外推维持目标（预测补全），不丢
  4. 拖尾修正：框的位置用"最新一帧事件质心"（前沿），不用密度质心（旧事件拖尾）

用法：reader = EventReader(); boxes = reader.feed(ev_f, ev_s, flash=False)
boxes = [{'x','y','w','h','cx','cy','area','track','predicted'}]
"""

import cv2
import numpy as np


class EventReader:
    def __init__(self, window=6, blur=1.5, thr=0.15, min_area=20, max_frac=0.3,
                 match_dist=40, max_age=10, dist=28.0, flash_frac=0.3,
                 flash_hold=4, min_track_age=2):
        self.window = window       # 密度 EMA 的有效窗口（帧；长窗抓慢物体稀疏事件）
        self.blur = blur           # 密度图空间平滑 σ
        self.thr = thr             # 密度阈值（≥thr 视为前景）
        self.min_area = min_area   # 最小连通域面积（像素）
        self.max_frac = max_frac   # 最大占比（过滤全帧爆发/全局闪变）
        self.match_dist = match_dist  # 跟踪关联最大质心距离
        self.max_age = max_age     # 轨迹最大失联帧数
        self.dist = dist           # 碎片合并质心距离（同一物体的碎片挨着）
        self.flash_frac = flash_frac  # 闪变判据：密度>thr 的像素占比超过它
        self.flash_hold = flash_hold  # 闪变期最多预测维持多少帧
        self.min_track_age = min_track_age  # 轨迹连续存活 ≥N 帧才输出（威胁度=时间一致性，
                                            # docs/186 §五：噪声闪烁、真物体持续）
        self.density = None
        self.tracks = []           # [{id, cx, cy, age, box, vx, vy, hist, hits}]
        self.next_id = 0
        self.flash_count = 0       # 当前闪变已持续帧数

    def is_flash(self, ev_f=None, ev_s=None):
        """闪变判定：本帧事件占比 >flash_frac（阶跃帧立刻抓住）或 密度占比
        >flash_frac（淹没后残留）。快窗口的读者判得准。"""
        cur = False
        if ev_f is not None:
            ev = (ev_f != 0) | (ev_s != 0)
            cur = ev.sum() / max(1, ev.size) > self.flash_frac
        if cur:
            return True
        if self.density is None:
            return False
        return (self.density > self.thr).sum() / max(1, self.density.size) > self.flash_frac

    def feed(self, ev_f, ev_s, flash=None):
        ev = np.where((ev_f != 0) | (ev_s != 0), 1, 0).astype(np.float32)
        if self.density is None:
            self.density = np.zeros_like(ev)
        if flash is None:
            flash = self.is_flash(ev_f, ev_s)
        a = 1.0 / self.window
        if flash:
            # 闪变期不吸收洪泛事件（那是环境不是物体），只衰减——闪变结束重捕获快
            self.density *= (1 - a)
        else:
            self.density = (1 - a) * self.density + a * ev

        if flash:
            # 闪变模式：全局淹没局部信号 → 用轨迹预测维持（预测补全）。
            # 预测 = 过最近 3 点的二次外推（速度+加速度，跟上加速运动）。
            self.flash_count += 1
            out = []
            for t in self.tracks:
                hist = t.get("hist", [])
                if len(hist) >= 3:
                    (x0, y0), (x1, y1), (x2, y2) = hist[-3], hist[-2], hist[-1]
                    cx = 2.5 * x2 - 2.0 * x1 + 0.5 * x0
                    cy = 2.5 * y2 - 2.0 * y1 + 0.5 * y0
                    vx, vy = cx - x2, cy - y2
                else:
                    vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
                    cx, cy = t["cx"] + vx, t["cy"] + vy
                b = t["box"]
                nb = dict(b, cx=cx, cy=cy, track=t["id"], predicted=True)
                nb["x"] = int(round(cx - b["w"] / 2)); nb["y"] = int(round(cy - b["h"] / 2))
                out.append(nb)
                t["cx"], t["cy"] = cx, cy
                t["age"] = 0       # 闪变是环境不是目标消失
            if self.flash_count > self.flash_hold:
                self.tracks = []   # 超时：承认丢失，下次重新捕获
            return out

        self.flash_count = 0
        d = self.density
        if self.blur > 0:
            d = cv2.GaussianBlur(self.density, (0, 0), self.blur)
        mask = (d >= self.thr).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        cur = ev > 0.5              # 本帧事件（用于前沿定位）
        boxes = []
        n_pix = mask.size
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            if area > self.max_frac * n_pix:
                continue
            # 前沿定位：组件内本帧事件质心（没有则用密度质心）
            inside = cur & (labels == i)
            if inside.sum() >= 3:
                ys, xs = np.nonzero(inside)
                cx, cy = float(xs.mean()), float(ys.mean())
            else:
                cx, cy = float(cents[i][0]), float(cents[i][1])
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                          "cx": cx, "cy": cy, "area": int(area), "track": -1,
                          "predicted": False})
        boxes = self.merge_boxes(boxes, self.dist)
        self._track(boxes)
        # 威胁度 = 时间一致性（docs/186 §五）：只有轨迹连续存活 ≥min_track_age 帧
        # 的框才输出（真物体持续，噪声闪烁）。用轨迹的累计命中数判定。
        if self.min_track_age > 0:
            confirmed = {t["id"] for t in self.tracks if t.get("hits", 0) >= self.min_track_age}
            boxes = [b for b in boxes if b["track"] in confirmed]
        # 前导补偿：框位置 = 最新事件质心 + v/2（本帧事件是"旧带+新带"两段，
        # 均值滞后半步；补 v/2 即当前真实位置）。仅对实测框，预测框已外推。
        for b in boxes:
            if b.get("predicted"):
                continue
            t = next((t for t in self.tracks if t["id"] == b["track"]), None)
            if t is not None and (t.get("vx") or t.get("vy")):
                b["cx"] += t["vx"] * 0.5
                b["cy"] += t["vy"] * 0.5
        return boxes

    def _track(self, boxes):
        for b in boxes:
            best, bd = None, self.match_dist
            for t in self.tracks:
                dd = ((t["cx"] - b["cx"]) ** 2 + (t["cy"] - b["cy"]) ** 2) ** 0.5
                if dd < bd:
                    best, bd = t, dd
            if best is not None:
                b["track"] = best["id"]
                best["hits"] = best.get("hits", 0) + 1     # 轨迹存活帧数（威胁度计分）
                hist = best.setdefault("hist", [])
                hist.append((best["cx"], best["cy"]))
                if len(hist) > 3:
                    hist.pop(0)
                if len(hist) >= 2:
                    vx = (b["cx"] - best["cx"]) / max(1, len(hist))
                    vy = (b["cy"] - best["cy"]) / max(1, len(hist))
                    best["vx"], best["vy"] = vx, vy
                best.update(cx=b["cx"], cy=b["cy"], age=0, box=b)
            else:
                b["track"] = self.next_id
                self.tracks.append({"id": self.next_id, "cx": b["cx"], "cy": b["cy"],
                                    "age": 0, "box": b, "vx": 0.0, "vy": 0.0,
                                    "hist": [], "hits": 0})
                self.next_id += 1
        for t in self.tracks:
            t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

    @staticmethod
    def _iou(a, b):
        ax1, ay1 = a["x"], a["y"]
        ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1 = b["x"], b["y"]
        bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) +
                           (bx2 - bx1) * (by2 - by1) - inter)

    @staticmethod
    def merge_boxes(boxes, dist=22.0):
        """碎片合并：同一物体的碎片挨着（IoU>0.05 或质心 <dist）。
        dist 不能太大——擦肩而过的不同物体不能并成一个。"""
        boxes = sorted(boxes, key=lambda b: -b["area"])
        kept = []
        for b in boxes:
            merged = False
            for k in kept:
                dd = ((k["cx"] - b["cx"]) ** 2 + (k["cy"] - b["cy"]) ** 2) ** 0.5
                if dd < dist or EventReader._iou(b, k) > 0.05:
                    x1 = min(b["x"], k["x"]); y1 = min(b["y"], k["y"])
                    x2 = max(b["x"] + b["w"], k["x"] + k["w"])
                    y2 = max(b["y"] + b["h"], k["y"] + k["h"])
                    k.update(x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                             cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                             area=k["area"] + b["area"])
                    merged = True
                    break
            if not merged:
                kept.append(b)
        return kept
