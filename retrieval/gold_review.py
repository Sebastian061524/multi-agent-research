"""复查 gold 的定义：那些「多返回」的段落，到底是噪音，还是有贡献？

背景（第 12 课的假设）：
    cross-encoder 判的是「这一段能不能回答这个查询」，而且判得很准。
    那么它返回的、被我们标成「多余」的段落，会不会其实【有贡献】？
    如果是，precision 0.46 就不是系统的问题，而是 gold 标得太窄。

本脚本对每条「多给了段落」的查询打印：
    ① 查询 + gold + 实际返回
    ② 每一段的精排分 —— 看模型认为它们差多少
    ③ 多出来的那一段的【全文】—— 判「有没有贡献」最终要靠人读

两个可计算的线索：
    · 分差：gold 段与多出段的精排分差。差得越小，越可能是「gold 标窄了」
    · 实词命中：查询里去掉停用词后的实词，多出的那一段命中了几个

用法（项目根目录）：
    python -m retrieval.gold_review
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R
from retrieval.bm25 import STOPWORDS, tokenize
from retrieval.eval_set import EVAL_SET

IDX_OF = {c["text"]: i for i, c in enumerate(R.chunks)}


def retrieve_indices(query, **kwargs):
    return [IDX_OF[c["text"]] for c in R._retrieve(query, **kwargs)]


def rerank_all(query):
    """对全部 7 段打分（因为 coarse_k=20 时候选池就是全集）"""
    return R.reranker.predict([[query, c["text"]] for c in R.chunks])


# ==================== 找出所有「多给了」的查询 ====================
cases = []
for item in EVAL_SET:
    gold = set(item["gold"])
    if not gold:
        continue
    got = retrieve_indices(item["query"])
    extra = [i for i in got if i not in gold]
    if extra:
        cases.append({"item": item, "got": got, "gold": gold, "extra": extra})

n_extra = sum(len(c["extra"]) for c in cases)
print("=" * 92)
print(f"有标准答案的查询里，{len(cases)} 条【多给了】段落，共多给 {n_extra} 段")
print("=" * 92)

# ==================== ① 汇总：按「分差」分档 ====================
print("\n① 按【gold 段与多出段的分差】分档：")
print("   分差越小 → 精排认为它们越同等相关 → 越可能说明 gold 标窄了\n")

buckets = {"差 < 0.01（几乎并列）": [], "差 0.01~0.1": [], "差 > 0.1（明显更低）": []}

for c in cases:
    scores = rerank_all(c["item"]["query"])
    gold_best = max(scores[i] for i in c["gold"])
    for i in c["extra"]:
        d = gold_best - scores[i]
        if d < 0.01:
            buckets["差 < 0.01（几乎并列）"].append((c, i, d, gold_best, scores[i]))
        elif d < 0.1:
            buckets["差 0.01~0.1"].append((c, i, d, gold_best, scores[i]))
        else:
            buckets["差 > 0.1（明显更低）"].append((c, i, d, gold_best, scores[i]))

for name, rows in buckets.items():
    print(f"   {name:<22} {len(rows)} 段")

# ==================== ② 逐条明细 ====================
for c in cases:
    item, got, gold, extra = c["item"], c["got"], c["gold"], c["extra"]
    scores = rerank_all(item["query"])
    q_words = set(tokenize(item["query"], STOPWORDS))

    print()
    print("=" * 92)
    print(f"查询：{item['query']}")
    print(f"  gold = {sorted(gold)}    实际返回 = {got}")
    print("-" * 92)

    print("  各段精排分：")
    for i in range(len(R.chunks)):
        mark = "★gold" if i in gold else ("→多给" if i in extra else "     ")
        title = R.chunks[i]["text"].split("\n")[0][:16]
        print(f"    [{i}] {scores[i]:.4f}  {mark}  {title}")

    for i in extra:
        gold_best = max(scores[j] for j in gold)
        text = R.chunks[i]["text"]
        words = set(tokenize(text, STOPWORDS))
        shared = sorted(q_words & words)
        print()
        print(f"  ── 多出来的 chunk[{i}] ── {R.chunks[i]['text'].split(chr(10))[0]}")
        print(f"     精排分 {scores[i]:.4f}   gold 最高分 {gold_best:.4f}"
              f"   分差 {gold_best - scores[i]:+.4f}")
        print(f"     命中查询实词 {len(shared)}/{len(q_words)}：{shared}")
        print(f"     全文：{text}")
