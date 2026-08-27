"""vision/language_prior.py — 语言先验通道（docs/148 嘴的上行轨：输入侧预测）。

docs/194 眼+语言配合读字：语言给上下文候选集，视觉在候选内识别。本模块把"规则化
候选集"升级为真语言通道——LLM 预测"AB_" 后最可能的字符（输入侧，docs/125 先验），
无 key 优雅回退到规则候选集（docs/56：心不依赖嘴，环境无 key 不崩）。

与 mouth_llm.py（输出侧：状态→话）不同：
  - mouth_llm：LLM 翻译内部状态（docs/56 语言是输出不是心）
  - language_prior：LLM 预测外部上下文（docs/107 猜中预期）——输入侧先验

铁律：
  1) 只预测"下一个字符"——返回候选字符列表 + 各自的先验权重；
  2) 永不写状态——候选只给视觉做似然缩小（docs/123 先验不冒充后验）；
  3) 优雅回退——无 key / API 错 → 规则候选集（26 字母均匀或上下文规则）。

用法：
  from language_prior import LanguagePrior
  lp = LanguagePrior()
  cands = lp.predict("AB_")   # -> [("C", 0.7), ("G", 0.2), ("O", 0.1)]
"""

import json
import os
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TIMEOUT = 15
MAX_TOKENS = 60

SYSTEM = (
    "你是阅读预测器。给你一个单词前缀（末尾是下划线代表未知字符），"
    "请预测下划线处最可能是什么英文字母。只输出 3-5 个候选字母（大写），"
    "每个一行，格式：字母 空格 概率（0-1 小数）。不要解释。"
    "例：AB_ → C 0.6\\nG 0.2\\nO 0.1。若前缀无意义，给常见字母如 E/T/A/S。"
)

# 规则回退：常见字母先验（无上下文时的默认，英语字母频率近似）
DEFAULT_PRIOR = [("E", 0.12), ("T", 0.09), ("A", 0.08), ("O", 0.07),
                 ("I", 0.07), ("N", 0.07), ("S", 0.06), ("H", 0.06)]


class LanguagePrior:
    """语言先验：上下文 → 下一字符候选（LLM 优先，规则回退）。"""

    def __init__(self, key=None):
        self.key = key if key is not None else os.environ.get("DEEPSEEK_API_KEY", "")
        self.checks = 0
        self.used_llm = 0
        self.fallback = 0

    def predict(self, prefix, topk=4):
        """prefix: 如 "AB_"（下划线=未知）。返回 [(char, prob), ...]。"""
        self.checks += 1
        if not self.key:
            self.fallback += 1
            return self._rule_fallback(prefix, topk)
        try:
            cands = self._call_llm(prefix)
            if cands:
                self.used_llm += 1
                return cands[:topk]
            self.fallback += 1
            return self._rule_fallback(prefix, topk)
        except Exception:  # noqa: BLE001 -- 语言通道可坏，视觉不依赖它
            self.fallback += 1
            return self._rule_fallback(prefix, topk)

    def _call_llm(self, prefix):
        body = {"model": MODEL,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"前缀：{prefix}"}],
                "temperature": 0.3, "max_tokens": MAX_TOKENS}
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            text = json.loads(r.read().decode())["choices"][0]["message"]["content"]
        out = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha():
                try:
                    p = float(parts[1])
                except ValueError:
                    p = 1.0 / max(len(text.splitlines()), 1)
                out.append((parts[0].upper(), p))
        # 归一化
        s = sum(p for _, p in out) or 1.0
        return [(c, p / s) for c, p in out]

    def _rule_fallback(self, prefix, topk):
        """规则回退：有上下文规则用规则，否则用字母频率。"""
        known = "".join(ch for ch in prefix if ch.isalpha())
        # 简单英语词缀规则（docs/194 的 CONTEXT 规则化版）
        RULES = {
            "AB": ["C", "G", "O"], "AC": ["T", "E", "H"], "AD": ["D", "A", "E"],
            "CA": ["T", "R", "B"], "CO": ["N", "M", "P"], "ST": ["A", "O", "R"],
            "TH": ["E", "A", "I"], "TR": ["A", "E", "O"], "PR": ["O", "E", "A"],
            "CH": ["A", "E", "O"], "SH": ["A", "E", "I"], "WH": ["A", "E", "I"],
        }
        for k, v in RULES.items():
            if known.upper().endswith(k):
                return [(c, 1.0 / len(v)) for c in v[:topk]]
        return DEFAULT_PRIOR[:topk]


if __name__ == "__main__":
    lp = LanguagePrior()
    print("== 语言先验通道 ==")
    print(f"  key: {'有（LLM）' if lp.key else '无（规则回退）'}")
    for prefix in ["AB_", "CA_", "TH_", "XQ_"]:
        print(f"  {prefix} → {lp.predict(prefix)}")
    print(f"  统计: 检查 {lp.checks} 次，LLM {lp.used_llm} 次，回退 {lp.fallback} 次")
