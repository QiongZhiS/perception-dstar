"""vision/situation.py — 视觉态势接口（docs/160 视觉版）：维持目标 + Top-K 威胁度分配。

把 topk_experiment.py 的"外部给维持目标 → 威胁度分配"工程化为标准中间表示，对齐
docs/160 态势格式（SituationReport 纯 JSON）：决策只读态势、感知输入可插拔。

态势结构（docs/160 同构）：
  { "tick": n,
    "objects": [{"id","x","y","w","h","cx","cy","track","threat","confirmed"}],
      # 维持目标（被指定或已确认的轨迹）+ Top-K 保留的高威胁候选
    "targets": [{"id","x","y","w","h","vx","vy","age"}],
      # 当前维持目标（第 4 层：值得维持的轨迹）
    "threats": [{"id","cx","cy","score"}],
      # Top-K 高威胁落空（威胁度 = 与维持目标的关系）
    "saccade": {"fov": "full", "k": 3},
    "budget": -1,                       # -1 = 无限（视觉线当前无注意力预算）
    "metrics": {"n_boxes": 0, "n_kept": 0, "target_hit": 0} }

输入可插拔：fill(frame, masks_or_none) 接收帧流；维持目标来源 = 外部指定（态势里
targets 初始注入）或轨迹确认（age≥N 升级）。决策只读 objects/targets。

用法：
  from situation import SituationReport
  rep = SituationReport(K=3)
  for frame in frames: rep.fill(gray, targets=external_targets)
"""

import numpy as np

from reader import EventReader
from transduction import Transduction2D


class SituationReport:
    """视觉态势：每帧生成标准态势 JSON（维持目标 + Top-K 威胁度分配）。"""

    def __init__(self, K=3, thresh=0.35, deadband=0.06, blur=1.6,
                 confirm_age=5, alpha=0.3):
        self.K = K                        # Top-K：每帧最多保留的候选框
        self.td = Transduction2D(thresh=thresh, deadband=deadband, blur=blur)
        self.rf = EventReader(window=2, thr=0.25, min_area=12, blur=1.2,
                              min_track_age=0)
        self.rs = EventReader(window=10, thr=0.12, min_area=12, blur=1.5,
                              min_track_age=0)
        self.confirm_age = confirm_age    # 轨迹存活 ≥N 帧 → 升级为维持目标（自发）
        self.alpha = alpha                # 目标速度平滑系数
        self.tick = 0
        self.targets = []                 # 维持目标：{id,cx,cy,w,h,vx,vy,age,external}
        self.next_target = 0
        self.metrics = {"n_boxes": 0, "n_kept": 0, "target_hit": 0,
                        "n_targets": 0}

    # ---- 维持目标管理 ----
    def add_target(self, x, y, w, h, external=True):
        """外部指定维持目标（docs/174 外赋：利害锚来自外部）。同位置去重。"""
        cx, cy = x + w / 2, y + h / 2
        dup = any(abs(tg["cx"] - cx) < 25 and abs(tg["cy"] - cy) < 25
                  for tg in self.targets)
        if dup:
            return
        self.targets.append({"id": self.next_target, "cx": cx, "cy": cy,
                             "w": w, "h": h, "vx": 0.0, "vy": 0.0, "age": 0,
                             "external": external})
        self.next_target += 1

    def lock_target(self, gray, hint, hint_radius=120):
        """任务层锁定：给符号目标 + 弱位置约束（hint=(hx,hy)，不精确），
        从候选框里找最接近 hint 的框锁定为维持目标（docs/190 目标驱动识别）。
        返回是否锁定成功。锁定后不再需要外部坐标——威胁度+轨迹维持接管。
        注意：本方法会 step 一帧（锁定帧的感知），调用方不应再对同帧 fill。"""
        ev_f, ev_s, _, _ = self.td.step(gray)
        flash = self.rf.is_flash(ev_f, ev_s)
        boxes = EventReader.merge_boxes(self.rf.feed(ev_f, ev_s, flash=flash) +
                                        self.rs.feed(ev_f, ev_s, flash=flash))
        if not boxes:
            return False
        # 候选框里最接近 hint 的（hint 是弱约束，允许偏差 hint_radius）
        hx, hy = hint
        near = [b for b in boxes if ((b["cx"] - hx) ** 2 + (b["cy"] - hy) ** 2) ** 0.5
                <= hint_radius]
        if not near:
            return False
        best = min(near, key=lambda b: ((b["cx"] - hx) ** 2 + (b["cy"] - hy) ** 2) ** 0.5)
        self.add_target(best["cx"] - best["w"] / 2, best["cy"] - best["h"] / 2,
                        best["w"], best["h"], external=False)   # 锁定=自发升级（weak external）
        return True

    def _promote_tracks(self, boxes):
        """自发目标：轨迹连续存活 ≥confirm_age 帧（hits 累计 + 当前帧仍匹配）
        → 升级为维持目标（第 4 层试探）。"""
        for b in boxes:
            t = next((t for t in self.rf.tracks + self.rs.tracks
                      if t["id"] == b["track"]), None)
            if t is None:
                continue
            # 连续存活：累计 hits ≥ confirm_age 且当前帧匹配（age==0 是本帧更新后）
            if t.get("hits", 0) >= self.confirm_age and t.get("age", 99) <= 1:
                self.add_target(b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2,
                                b["w"], b["h"], external=False)

    # ---- 威胁度（与维持目标的关系，topk_experiment 同款）----
    def _threat(self, b, tpred):
        ax1, ay1 = b["x"], b["y"]
        ax2, ay2 = b["x"] + b["w"], b["y"] + b["h"]
        bx1, by1 = tpred["x"], tpred["y"]
        bx2, by2 = tpred["x"] + tpred["w"], tpred["y"] + tpred["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        iou = inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) +
                          (bx2 - bx1) * (by2 - by1) - inter)
        t = next((t for t in self.rf.tracks + self.rs.tracks
                  if t["id"] == b["track"]), None)
        approach = 0.0
        if t is not None:
            vx, vy = t.get("vx", 0.0), t.get("vy", 0.0)
            sp = (vx ** 2 + vy ** 2) ** 0.5
            if sp > 1.0:
                dx = tpred["x"] + tpred["w"] / 2 - b["cx"]
                dy = tpred["y"] + tpred["h"] / 2 - b["cy"]
                dd = (dx ** 2 + dy ** 2) ** 0.5
                if dd > 1e-6:
                    approach = max(0.0, (vx * dx + vy * dy) / (sp * dd))
        return iou + 0.3 * approach

    def _target_pred(self, tg):
        return {"x": tg["cx"] + tg["vx"] - tg["w"] / 2,
                "y": tg["cy"] + tg["vy"] - tg["h"] / 2,
                "w": tg["w"], "h": tg["h"]}

    # ---- 主流程 ----
    def fill(self, gray, external_targets=None):
        """处理一帧。external_targets: [(x,y,w,h)] 本帧外部指定的目标（可空）。"""
        if external_targets:
            for x, y, w, h in external_targets:
                self.add_target(x, y, w, h, external=True)
        ev_f, ev_s, _, _ = self.td.step(gray)
        flash = self.rf.is_flash(ev_f, ev_s)
        boxes = EventReader.merge_boxes(self.rf.feed(ev_f, ev_s, flash=flash) +
                                        self.rs.feed(ev_f, ev_s, flash=flash))
        self._promote_tracks(boxes)

        # 威胁度分配：对每个目标，候选框威胁度 = 与目标预测的关系；Top-K 保留。
        # kept 跨目标去重（同一框可能对多个目标都高威胁，只保留一次）。
        objects = []          # 维持目标本身（作为 objects 输出）
        threats = []          # Top-K 高威胁候选
        kept = []
        kept_keys = set()
        for tg in self.targets:
            tpred = self._target_pred(tg)
            objects.append({"id": tg["id"], "x": int(tpred["x"]), "y": int(tpred["y"]),
                            "w": int(tpred["w"]), "h": int(tpred["h"]),
                            "cx": round(tg["cx"], 1), "cy": round(tg["cy"], 1),
                            "vx": round(tg["vx"], 2), "vy": round(tg["vy"], 2),
                            "age": tg["age"], "external": tg["external"]})
            if boxes:
                scored = sorted(boxes, key=lambda b: self._threat(b, tpred),
                                reverse=True)
                top = scored[:self.K]
                # 强制包含离目标最近的框（车可能不在威胁度 Top-K——预测滞后时
                # 背景框与目标预测的 IoU 可能更高；目标更新需要位置连续性优先）
                if top and not any(((b["cx"] - tg["cx"]) ** 2 +
                                    (b["cy"] - tg["cy"]) ** 2) ** 0.5 < 30
                                   for b in top):
                    nearest = min(boxes, key=lambda b: ((b["cx"] - tg["cx"]) ** 2 +
                                                        (b["cy"] - tg["cy"]) ** 2) ** 0.5)
                    top.append(nearest)
                for b in top:
                    key = (b["track"], round(b["cx"]), round(b["cy"]))
                    if key not in kept_keys:
                        kept_keys.add(key)
                        kept.append(b)
                threats.append({"id": tg["id"], "cx": round(tg["cx"], 1),
                                "cy": round(tg["cy"], 1),
                                "top_scores": [round(self._threat(b, tpred), 3)
                                               for b in top]})
            # 目标命中：离目标最近的**可信框**（位置连续性优先 + 面积门槛——
            # 前几帧车事件未累积，最近的框可能是背景碎片，不能被带偏；
            # 车框一旦出现（面积够大）就更新，否则闭眼走保持位置等待）
            hit = False
            if kept:
                near = [b for b in kept if b["area"] >= tg["w"] * tg["h"] * 0.25]
                if near:
                    best = min(near, key=lambda b: ((b["cx"] - tg["cx"]) ** 2 +
                                                    (b["cy"] - tg["cy"]) ** 2) ** 0.5)
                    err = ((best["cx"] - tg["cx"]) ** 2 +
                           (best["cy"] - tg["cy"]) ** 2) ** 0.5
                    if err < 30:
                        hit = True
            # 更新目标（命中框更新位置；无命中则速度外推保持——闭眼走）
            if hit:
                tg["vx"] = self.alpha * (best["cx"] - tg["cx"]) + (1 - self.alpha) * tg["vx"]
                tg["vy"] = self.alpha * (best["cy"] - tg["cy"]) + (1 - self.alpha) * tg["vy"]
                tg["cx"], tg["cy"] = best["cx"], best["cy"]
                tg["w"], tg["h"] = best["w"], best["h"]
                self.metrics["target_hit"] += 1
            else:
                tg["cx"] += tg["vx"]
                tg["cy"] += tg["vy"]
            tg["age"] += 1

        self.metrics["n_boxes"] += len(boxes)
        self.metrics["n_kept"] += len(kept)
        self.metrics["n_targets"] = len(self.targets)
        rep = {"tick": self.tick, "objects": objects, "targets": objects,
               "threats": threats,
               "saccade": {"fov": "full", "k": self.K},
               "budget": -1,
               "metrics": {k: (v / (self.tick + 1) if k in ("n_boxes", "n_kept")
                               else v) for k, v in self.metrics.items()}}
        self.tick += 1
        return rep

    def to_text(self, rep):
        """文本化（docs/160 toText 视觉版）：决策/叙事输入侧。"""
        parts = [f"tick {rep['tick']}"]
        if rep["objects"]:
            for o in rep["objects"][:3]:
                tag = "指定" if o["external"] else "自发"
                parts.append(f"维持[{tag}]#{o['id']} 在({o['cx']:.0f},{o['cy']:.0f})"
                             f" 速度({o['vx']:.1f},{o['vy']:.1f}) 年龄{o['age']}")
        else:
            parts.append("没有维持目标")
        parts.append(f"Top-{rep['saccade']['k']} 威胁 {len(rep['threats'])} 组")
        parts.append(f"本帧框 {rep['metrics']['n_boxes']}→保留 {rep['metrics']['n_kept']}")
        return "。".join(parts) + "。"
