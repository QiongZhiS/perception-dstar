"""vision/demo_app.py — 可交互演示：一个"看得见它在看什么/维持什么/怀疑什么/怎么说"的主体。

外部对话第 4 块"图纸变墙"：把 docs/200-216 的机制链拼成一个连续可交互的演示——
让人打开浏览器就能看到系统如何被世界否定、如何维持目标、如何选择固执或开放、
如何把状态说成人话。

拼装的机制（全部来自已验证实验，每步可追溯）：
  接收     ：合成红球匀速直线（docs/195 场景，color_robust bright）
  对齐/确认：颜色确认（红区占比 > thr，thr 可被语言层调节，docs/206）
  被拒入史 ：Keep（docs/200：失败保留为历史，纪念加深=重信任延迟 ∝ 被拒时长）
  类别怀疑 ：suspicious[颜色] 表（docs/205：被拒类别记账）
  时间门   ：docs/199b（稳定才校正/固执）
  语言内化 ：LanguageMind（docs/204：符号槽→转录→自反接口）
交互（观众注入）：
  染蓝     ：目标被外部改变 = "他者的不"（docs/178 补刀）→ 系统拒绝+入史+纪念
  噪声     ：目标区加噪声 = "世界的噪声"（docs/101）→ 系统固执（不误改判）
  恢复     ：目标恢复红 → 系统带着历史重新信任（需要更连续证据）
  换任务   ：任务层利害更新（找红/找蓝）→ 外赋利害（docs/188）
  重置     ：清空记忆（重新开始）

用法：
  python vision/demo_app.py --port 8080
  浏览器打开 http://127.0.0.1:8080
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# 基于脚本自身位置解析 vision 模块——从任何目录运行都能 import（不依赖 cwd）
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import cv2  # noqa: E402
from color_robust import make_scene, W, H, N, RED  # noqa: E402

# ---- 感知核心（复用已验证机制）----


class DemoMind:
    """演示主体：接收→确认→被拒入史→类别怀疑→语言内化。"""

    def __init__(self, task_hue=0.0):
        self.task_hue = task_hue            # 外赋利害：目标色相（红 0° / 蓝 120°）
        self.thr = 0.2                      # 确认阈值（语言层可调，docs/206）
        self.trusted = True                 # 是否信任当前目标
        self.consec = 0                     # 重信任连续帧
        self.rejected = {}                  # suspicious[色相] = 被拒累计（docs/205）
        self.rej_total = 0                  # 总被拒（docs/200 纪念）
        self.suspect_color = None           # 当前可疑色相（被拒类别）
        self.persist = 0                    # 持续信任帧（阈值回落，docs/206）
        self.tick = 0
        self.log = []                       # 语言流（docs/204 转录）
        self.last_lang = None
        self.perturb = 'none'               # 观众注入：none/blue/noise
        self.perturb_left = 0

    def k(self):
        """重信任所需连续帧（docs/200 纪念加深 + docs/203 饱和）。"""
        return 1 + int(min(self.rej_total, 60) / 3)

    def _target(self):
        """当前帧目标真值位置（与帧循环同步取模——否则 tick>N 后帧回起点、
        目标却出画面，会错位导致一直误拒）。"""
        i = self.tick % N
        return (80 + 4.0 * i, 180 + 0.6 * i)

    def frame(self, frames):
        """取当前帧 + 应用扰动。"""
        fr = frames[self.tick % len(frames)].copy()
        rc = self._target()
        if self.perturb == 'blue' and self.perturb_left > 0:
            cv2.circle(fr, (int(rc[0]), int(rc[1])), 18, (200, 0, 0), -1)  # 染蓝
            self.perturb_left -= 1
            if self.perturb_left == 0:
                self.perturb = 'none'      # 扰动耗尽后复位，避免状态面板残留显示
        elif self.perturb == 'red' and self.perturb_left > 0:
            cv2.circle(fr, (int(rc[0]), int(rc[1])), 18, (0, 0, 200), -1)  # 染红（蓝任务下的他者的不）
            self.perturb_left -= 1
            if self.perturb_left == 0:
                self.perturb = 'none'
        elif self.perturb == 'noise' and self.perturb_left > 0:
            rng = np.random.default_rng(self.tick)
            noise = rng.normal(0, 25, fr.shape).astype(np.int16)
            fr = np.clip(fr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            self.perturb_left -= 1
            if self.perturb_left == 0:
                self.perturb = 'none'
        return fr

    def confirm(self, frac, hue):
        """一帧确认（docs/206 闭环：thr 随被拒/信任调节）。

        任务色相判定（docs/188 外赋利害 + docs/195 三通道确认）：
        目标区中位色相距任务色相 < 30° 且 有足够彩色 → 通过。
        红任务认红（hue≈0）、蓝任务认蓝（hue≈120）——换任务改变确认标准。
        """
        d = min(abs(hue - self.task_hue) % 180.0,
                180.0 - abs(hue - self.task_hue) % 180.0) if hue is not None else 999.0
        ok = frac > self.thr and d < 30.0
        if ok:
            self.persist += 1
            if self.persist >= 3 and self.thr > 0.2:
                self.thr = max(0.2, self.thr - 0.02)   # 持续信任回落
            if not self.trusted:
                self.consec += 1
                if self.consec >= self.k():
                    self.trusted = True
        else:
            self.rej_total += 1
            self.persist = 0
            self.trusted = False
            self.consec = 0
            # 被拒上调（docs/206），但上限必须低于目标球体物理可达的彩色占比
            # （bright 场景实测 frac≈0.777）——否则 thr 涨满后即使目标恢复原色也
            # 永远 frac < thr，且 thr 只在 ok 帧回落 → 不可恢复死锁。
            self.thr = min(0.75, self.thr + 0.05)
            if hue is not None:
                self.rejected[hue] = self.rejected.get(hue, 0) + 1
                self.suspect_color = hue
        return ok

    def task_name(self):
        return "红球" if abs(self.task_hue - 0.0) < 60 else "蓝球"

    def color_name(self, hue):
        if hue is None:
            return "?"
        h = round(hue / 60.0) * 60.0 % 180.0
        return {0.0: "红", 60.0: "黄", 120.0: "蓝"}.get(h, f"{int(hue)}°")

    def act(self, action):
        """观众交互。"""
        if action == 'blue':
            self.perturb, self.perturb_left = 'blue', 30
        elif action == 'red':
            self.perturb, self.perturb_left = 'red', 30
        elif action == 'noise':
            self.perturb, self.perturb_left = 'noise', 30
        elif action == 'none':
            self.perturb, self.perturb_left = 'none', 0
        elif action == 'reset':
            self.__init__(task_hue=self.task_hue)
        elif action == 'task_red':
            self.task_hue = 0.0
            self.thr = 0.2            # 换任务 = 外赋利害更新（docs/188）：确认阈值回基线
            self.trusted = True       # 新任务重新开始信任
            self.consec = 0
        elif action == 'task_blue':
            self.task_hue = 120.0
            self.thr = 0.2            # 同上
            self.trusted = True
            self.consec = 0

    def language(self, frac, hue, ok, was_trusted, was_consec):
        """语言内化（docs/204 三层：符号槽→转录→自反）。按任务说话。

        ★docs/63 纪律（docs/219 §八）：这些句子是**维持层状态的模板转录**，不是体验
        的陈述——每个词都必须 1:1 指回可观测状态字段：
          "它变了，我认不出来了——我刚才可能认错了" ← rej_total==1 且本帧被拒
          "还是X色，不是任务。我不确定，但我先不调整" ← suspicious[hue]>0 且被拒
          "我确认了它——但我花了 k 帧才重新信任它"     ← consec 刚达 k()（claimed 翻转）
          "X回来了——但我还在观察"                     ← was_trusted=False 且本帧 ok
          "我需要更连续的证据（c/k 帧）"               ← 0 < consec < k()
        没有任何一句落在现象层（"我感觉…"）或价值层（"我在乎你"）——说出来的东西
        全部是行为签名（docs/31/32；docs/63 第 2 层纪律：不假装现象层）。

        was_trusted/was_consec 是本帧 confirm() 之前的快照——confirm() 已经推进了
        trusted/consec，语言层必须回看"这一帧发生了什么"（回来 / 累积 / 重新信任），
        否则 not-trusted & consec==0 与 trusted & consec==0 两个分支永远不可达，
        消息会丢失；而换任务后 trusted=True 会误触发"我确认了它"。
        """
        task = self.task_name()
        utt = None
        if not ok:
            if self.rej_total == 1:
                utt = f"它变了，我认不出来了——我刚才可能认错了（任务：{task}）。"
            elif self.suspect_color is not None:
                cn = self.color_name(self.suspect_color)
                utt = f"还是{cn}色，不是{task}。我不确定，但我先不调整（第 {self.rej_total} 次被拒）。"
        elif not was_trusted:
            if self.trusted:
                # 本帧完成重新信任（连续帧达到 k()）
                utt = f"我确认了它——但我花了 {self.k()} 帧才重新信任它。"
            elif was_consec == 0:
                # 被拒后第一帧合格的"回来"
                utt = f"{task}回来了——但我还在观察。"
            else:
                # 仍在累积证据
                utt = f"它看起来是{task}了，但我需要更连续的证据（{self.consec}/{self.k()} 帧）才重新信任。"
        # 一直在信任的稳定帧：不说话
        return utt


# ---- 演示状态 ----

mind = DemoMind()
frames = make_scene('bright')
lock = threading.Lock()

# ---- 前端页面 ----


def render_frame(fr):
    ok, buf = cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def state_json():
    with lock:
        fr = mind.frame(frames)
        rc = mind._target()
        # 球体掩码确认（docs/200 同款，但确认度量改为彩色占比——任务色相判定
        # 才是身份验证（docs/195 三通道），红区占比只对红任务有意义）
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h, s = hsv[:, :, 0], hsv[:, :, 1]
        r = 18
        cx, cy = int(rc[0]), int(rc[1])
        rh = h[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        rs = s[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        colored = rs > 60
        frac = float(colored.mean()) if colored.sum() >= 10 else 0.0
        hue = float(np.median(rh[colored])) if colored.sum() >= 10 else None
        was_trusted, was_consec = mind.trusted, mind.consec
        ok = mind.confirm(frac, hue)
        utt = mind.language(frac, hue, ok, was_trusted, was_consec)
        if utt:
            mind.log.append((mind.tick, utt))
            if len(mind.log) > 30:
                mind.log = mind.log[-30:]
        state = {
            "tick": mind.tick,
            "image": render_frame(fr),
            "cx": int(rc[0]), "cy": int(rc[1]),
            "frac": round(frac, 3),
            "hue": round(hue, 1) if hue is not None else None,
            "hue_name": mind.color_name(hue),
            "ok": ok,
            "thr": round(mind.thr, 2),
            "trusted": mind.trusted,
            "consec": mind.consec,
            "k": mind.k(),
            "rej_total": mind.rej_total,
            "suspect_color": round(mind.suspect_color, 0) if mind.suspect_color is not None else None,
            "suspect_name": mind.color_name(mind.suspect_color),
            "rejected": {str(round(kh, 0)): v for kh, v in sorted(mind.rejected.items())},
            "task_hue": mind.task_hue,
            "task_name": mind.task_name(),
            "perturb": mind.perturb,
            "log": [{"tick": t, "text": s} for t, s in mind.log[-8:]],
            "utt_now": utt,
        }
        mind.tick += 1
        return state


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>感知演示：它在看什么 / 维持什么 / 怀疑什么 / 怎么说</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: #111; color: #ddd; margin: 20px; }
  h1 { font-size: 18px; }
  .row { display: flex; gap: 20px; flex-wrap: wrap; }
  .panel { background: #1c1c1c; border: 1px solid #333; border-radius: 8px; padding: 14px; }
  img { border: 2px solid #444; border-radius: 6px; width: 480px; height: 270px; }
  table { border-collapse: collapse; font-size: 13px; }
  td, th { padding: 3px 10px; border-bottom: 1px solid #2a2a2a; text-align: left; }
  .ok { color: #7f7; } .no { color: #f77; }
  .btn { background: #2a6; color: #fff; border: none; padding: 8px 14px; margin: 3px;
         border-radius: 5px; cursor: pointer; font-size: 13px; }
  .btn.warn { background: #a63; } .btn.danger { background: #a33; } .btn.gray { background: #555; }
  .log { max-height: 180px; overflow-y: auto; font-size: 13px; }
  .log div { padding: 2px 0; }
  .utt { color: #ff9; font-weight: bold; }
</style>
</head>
<body>
<h1>感知演示：外赋利害驱动感知 — 它在看什么 / 维持什么 / 怀疑什么 / 怎么说</h1>
<p style="font-size:12px;color:#888">语言流为维持层状态的<b>行为签名转录</b>（docs/63 纪律：非体验陈述，每句 1:1 对应状态字段，见 docs/204/219）</p>
<div class="row">
  <div class="panel">
    <img id="scene" src="" alt="场景">
    <div style="margin-top:8px">
      <button class="btn warn" onclick="act('blue')">染蓝（他者的不）</button>
      <button class="btn warn" onclick="act('red')">染红（他者的不）</button>
      <button class="btn" onclick="act('noise')">噪声（世界的噪声）</button>
      <button class="btn gray" onclick="act('none')">停止扰动</button>
      <button class="btn" onclick="act('task_red')">任务：找红球</button>
      <button class="btn" onclick="act('task_blue')">任务：找蓝球</button>
      <button class="btn danger" onclick="act('reset')">重置记忆</button>
    </div>
    <div class="log" id="log" style="margin-top:10px"></div>
  </div>
  <div class="panel">
    <h3>感知状态</h3>
    <table id="state"></table>
    <h3 style="margin-top:14px">类别怀疑（suspicious 表，docs/205）</h3>
    <div id="suspect"></div>
    <h3 style="margin-top:14px">它现在说（docs/204 语言内化）</h3>
    <div class="utt" id="utt"></div>
  </div>
</div>
<script>
let lastUtt = '';
function act(a) { fetch('/act?action=' + a).then(r => r.json()); }
async function tick() {
  const r = await fetch('/state');
  const s = await r.json();
  document.getElementById('scene').src = 'data:image/jpeg;base64,' + s.image;
  const rows = [
    ['tick', s.tick],
    ['任务', s.task_name],
    ['目标位置', '(' + s.cx + ', ' + s.cy + ')'],
    ['目标区彩色占比', s.frac],
    ['观测色相', (s.hue === null ? '—' : s.hue + '°') + '（' + s.hue_name + '）'],
    ['确认', s.ok ? '通过 ✓（符合任务）' : '被拒 ✗（不符合任务）'],
    ['确认阈值 thr', s.thr],
    ['信任', s.trusted ? '信任' : '谨慎（需连续 ' + s.consec + '/' + s.k + ' 帧）'],
    ['被拒累计（纪念）', s.rej_total],
    ['当前可疑色相', s.suspect_name],
    ['扰动', s.perturb],
  ];
  document.getElementById('state').innerHTML =
    rows.map(([k, v]) => '<tr><td>' + k + '</td><td>' + v + '</td></tr>').join('');
  document.getElementById('suspect').innerHTML =
    Object.keys(s.rejected).length === 0 ? '（无被拒类别）' :
    Object.entries(s.rejected).map(([h, n]) =>
      '<div>色相 ' + h + '°：被拒 ' + n + ' 次</div>').join('');
  const log = document.getElementById('log');
  log.innerHTML = s.log.map(l => '<div>[' + l.tick + '] ' + l.text + '</div>').join('');
  log.scrollTop = log.scrollHeight;
  document.getElementById('utt').textContent = s.utt_now || '（正在观察）';
  setTimeout(tick, 150);
}
tick();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/state':
            self._json(state_json())
        elif self.path.startswith('/act?'):
            import urllib.parse
            q = urllib.parse.parse_qs(self.path.split('?', 1)[1])
            action = q.get('action', ['none'])[0]
            with lock:
                mind.act(action)
            self._json({"ok": True, "action": action})
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f"演示服务器: http://127.0.0.1:{args.port}")
    print("交互：染蓝=他者的不 / 噪声=世界的噪声 / 恢复 / 换任务 / 重置")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
