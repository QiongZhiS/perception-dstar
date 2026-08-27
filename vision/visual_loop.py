"""vision/visual_loop.py — 视觉态势闭环：态势 → 态势地图 → affordance 决策（docs/161/140 视觉版）。

把 docs/160 态势接口（situation.py）接上下游，形成最小行为闭环：
  帧 → 转导层 → 读层 → 态势（维持目标+Top-K 威胁）→ 态势地图（轨迹记忆+衰减）
  → affordance 决策（朝目标走）→ 行为 → （下一帧）

组件：
  - SituationMap（docs/161 视觉版）：objects → 轨迹记忆 cells（id → 位置+时间戳+conf），
    40 tick 衰减；"记得车在哪"而不是"只信眼前"
  - Affordance（docs/140 最简版）：五元组 (φ=目标物体, a=朝它走, p, c, o)；选择 =
    argmin c/p（朝最近维持目标走）
  - 闭环度量：目标丢失期间（瞬时读不到）地图决策能否继续指向它（docs/161 map_frac）

载体：DAVIS car-turn（车=维持目标，外赋）。行为 = 虚拟 agent 在画面坐标系移动
（位置 = 决策目标），不接真实运动学——验证"感知→记忆→决策"链，不是物理导航。

用法：python vision/visual_loop.py
"""
import sys, os
sys.path.insert(0, 'vision')
import numpy as np, cv2, json
from davis_check import mask_boxes
from situation import SituationReport


class SituationMap:
    """态势地图（docs/161 视觉版）：维持目标轨迹记忆 + 衰减。"""
    def __init__(self, decay=40):
        self.cells = {}      # id -> {"x","y","t","conf"}
        self.decay = decay
        self.tick = 0

    def update(self, objects):
        """objects: 态势 objects（维持目标）→ 写进地图。"""
        for o in objects:
            self.cells[o["id"]] = {"x": o["cx"], "y": o["cy"], "t": self.tick,
                                   "conf": 1.0}
        self.tick += 1

    def read(self):
        """读地图：返回有效（未过期）目标。conf 随年龄衰减。"""
        out = []
        for oid, c in self.cells.items():
            age = self.tick - c["t"]
            conf = c["conf"] * max(0.0, 1 - age / self.decay)
            if conf > 0:
                out.append({"id": oid, "x": c["x"], "y": c["y"], "conf": conf,
                            "age": age})
        return out

    def known_target(self):
        """决策用：最近的有效目标（记忆优先，docs/161）。"""
        best = None
        for t in self.read():
            if best is None or t["age"] < best["age"]:
                best = t
        return best


def affordance_choose(map_target, pos, targets=None):
    """最简 affordance（docs/140）：朝最近维持目标走。p=1（假设能到达），c=距离。
    返回 (目标位置, c)。targets 为空时用地图记忆（闭眼走）。"""
    if map_target is None:
        return None, None
    c = ((map_target["x"] - pos[0]) ** 2 + (map_target["y"] - pos[1]) ** 2) ** 0.5
    return (map_target["x"], map_target["y"]), c


def main():
    vdir = os.path.join("vision", "out", "davis", "car-turn")
    jpgs = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    rep = SituationReport(K=3)
    smap = SituationMap(decay=40)

    # 虚拟 agent：在画面坐标系，初始在左上角
    agent = np.array([30.0, 30.0])
    frames = 0
    map_decisions = 0
    inst_decisions = 0
    logs = []
    dist_hist = []

    for i, jpg in enumerate(jpgs):
        fr = cv2.imread(os.path.join(vdir, jpg))
        mask = cv2.imread(os.path.join(vdir, jpg.replace(".jpg", ".png")), 0)
        if fr is None or mask is None:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ext = None
        if frames == 0:
            gt = mask_boxes(mask)
            if gt:
                _, x, y, w, h = gt[0]
                ext = [(x, y, w, h)]
        s = rep.fill(gray, external_targets=ext)
        smap.update(s["objects"])
        frames += 1

        # 决策：优先地图记忆（docs/161 map_frac），其次瞬时态势
        mt = smap.known_target()
        inst = s["objects"][0] if s["objects"] else None
        if mt is not None:
            target_pos, cost = affordance_choose(mt, agent)
            map_decisions += 1
        elif inst is not None:
            target_pos, cost = (inst["cx"], inst["cy"]), None
            inst_decisions += 1
        else:
            target_pos, cost = None, None

        # 行为：朝目标走（简单趋近）
        if target_pos is not None:
            dx, dy = target_pos[0] - agent[0], target_pos[1] - agent[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist > 20:
                step = 8.0
                agent += np.array([dx, dy]) / max(dist, 1e-6) * step
        # 记录与真值车的距离（收敛判据）
        gt = mask_boxes(mask)
        if gt:
            _, gx, gy, gw, gh = gt[0]
            dist_hist.append(((agent[0] - (gx + gw / 2)) ** 2 +
                              (agent[1] - (gy + gh / 2)) ** 2) ** 0.5)
        if frames in (1, 10, 40, 79):
            mt_txt = f'#{mt["id"]}@({mt["x"]:.0f},{mt["y"]:.0f}) conf={mt["conf"]:.2f}' \
                if mt else "无"
            inst_txt = f'#{inst["id"]}' if inst else "无"
            logs.append(f"tick{frames}: agent=({agent[0]:.0f},{agent[1]:.0f}) "
                        f"地图目标={mt_txt} 瞬时目标={inst_txt}")

    print("== 视觉态势闭环（car-turn：态势→地图→affordance→行为）==")
    print(f"帧数 {frames}，决策：地图记忆 {map_decisions} 次（map_frac="
          f"{map_decisions / max(frames, 1):.2f}），瞬时 {inst_decisions} 次")
    dh = np.array(dist_hist)
    print(f"agent-车距离：起始 {dh[0]:.0f}px → 结束 {dh[-1]:.0f}px；"
          f"后 20 帧中位 {np.median(dh[-20:]):.0f}px（收敛=追上了移动目标）")
    print("\n轨迹：")
    for lg in logs:
        print(f"  {lg}")
    out = os.path.join("vision", "out", "visual_loop.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"frames": frames, "map_decisions": map_decisions,
                   "inst_decisions": inst_decisions,
                   "agent_car_dist": [round(d, 1) for d in dist_hist]},
                  f, ensure_ascii=False, indent=1)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
