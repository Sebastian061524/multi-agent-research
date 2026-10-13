"""只测【召回层】：gold 有没有进候选池？

为什么单独测：
    现在的链路是 召回 → 精排 → 阈值。
    gold 没找到，可能是这三处任何一处的问题。
    这里把后两处【拿掉】，只问一个问题：
    「假设精排和阈值都完美，光靠召回能不能把 gold 拿进来？」

⚠️ 为什么必须看 k 小的那一头：
    库里只有 7 段。k=7 时候选池就是全集，召回必然 100% ——
    那不是「召回好」，是「召回没在筛」。有信息量的是 k=1、k=2。
"""

import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R
from retrieval.eval_set import EVAL_SET


def vector_ranking(query):
    """按向量相似度给出【全部】段落的排名（最相似 → 最不相似）"""
    qv = R.model.encode(query, normalize_embeddings=True)
    scores = R.embeddings @ qv
    return [int(i) for i in scores.argsort()[::-1]]


def recall_at(rows, k):
    """前 k 个候选里，覆盖了多少 gold"""
    hit = total = 0
    for item, rank in rows:
        gold = set(item["gold"])
        if not gold:
            continue
        hit += len(set(rank[:k]) & gold)
        total += len(gold)
    return hit / total if total else 0.0


# 只保留「有标准答案」的查询 —— 该拒答的那些不参与召回率
rows = [(item, vector_ranking(item["query"])) for item in EVAL_SET]
rows = [(item, rank) for item, rank in rows if item["gold"]]

print(f"先看一个例子：{rows[0][0]['query']}")
print(f"  gold = {rows[0][0]['gold']}")
print(f"  向量给出的排名 = {rows[0][1]}")
print()

print(f"有标准答案的查询：{len(rows)} 条\n")
print(f"{'top-k':>6}  {'召回率':>8}   柱状图")
print("-" * 60)
for k in range(1, 8):
    r = recall_at(rows, k)
    print(f"{k:>6}  {r:>8.2%}   {'█' * int(r * 34)}")

print()
print("按类型拆开（只看 k=1 和 k=3）：")
print(f"{'类型':<10}{'条数':>5}{'k=1':>10}{'k=3':>10}")
print("-" * 40)
for kind in dict.fromkeys(item["kind"] for item, _ in rows):
    sub = [(i, r) for i, r in rows if i["kind"] == kind]
    print(f"{kind:<10}{len(sub):>5}{recall_at(sub, 1):>9.2%}{recall_at(sub, 3):>10.2%}")
