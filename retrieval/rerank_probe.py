"""精排扰动实验：打开 cross-encoder 这个黑盒。

背景（第 4 课的诊断，一条反常的数据）：
    查询「多 Agent 协作和 RAG 有什么不同」下，
      chunk[2] 多 Agent 协作      精排 0.9881   ← 高
      chunk[3] RAG 检索增强生成    精排 0.0073   ← 几乎为零
    查询里明明有「RAG」，段落也整段在讲 RAG，为什么是 0.0073？

    而另一条数据看起来是矛盾的：
      查询「ReAct 架构的优缺点」
      chunk[1] Plan-and-Execute   精排 0.9990   ← 讲别的架构，却接近满分

    ⇒「它判相关不判能作答」这个解释【不够】。得再挖。

怎么问一个黑盒：
    cross-encoder 的内部看不到，但可以【扰动输入看输出怎么动】。
    固定全部段落，只改查询的某一个方面，看分数怎么变。

⭐ 关键的一组是「只调换两个主体的顺序」：
    如果分数【互相交换】，说明模型在看【位置】而不是看【语义】。

用法（项目根目录）：
    python -m retrieval.rerank_probe
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R

# (说明, 查询)
PROBES = [
    ("原句（A 和 B 有什么不同）", "多 Agent 协作和 RAG 有什么不同"),
    ("★只调换顺序（B 和 A）", "RAG 和多 Agent 协作 有什么不同"),
    ("只留第一个主体", "多 Agent 协作"),
    ("只留第二个主体", "RAG"),
    ("第一个主体 + 疑问", "多 Agent 协作是什么"),
    ("第二个主体 + 疑问", "RAG 是什么"),
    ("只用「有什么不同」", "有什么不同"),
    ("两个主体，无疑问词", "多 Agent 协作 RAG"),
]

# 关心的两个段落
A, B = 2, 3
TITLES = [c["text"].split("\n")[0][:10] for c in R.chunks]

# ==================== ① 探针 × 段落 的分数矩阵 ====================
print("=" * 100)
print("① 精排分数矩阵（行 = 查询变体，列 = 段落）")
print("   ★ 标记的是我们关心的 chunk[2]（多 Agent 协作）和 chunk[3]（RAG）")
print("=" * 100)

head = f"{'查询变体':<26}"
for i, t in enumerate(TITLES):
    mark = "*" if i in (A, B) else " "
    head += f"{mark}[{i}]{'':>3}"
print(head)
print("-" * 100)

for name, query in PROBES:
    scores = R.reranker.predict([[query, c["text"]] for c in R.chunks])
    row = f"{name:<26}"
    for s in scores:
        row += f"{s:>7.4f} "
    print(row)

print("-" * 100)
print("段落下标：", {i: t for i, t in enumerate(TITLES)})

# ==================== ② 只看 chunk[2] 和 chunk[3] 的对比 ====================
print()
print("=" * 100)
print(f"② 只看两个主角：chunk[{A}]={TITLES[A]}   chunk[{B}]={TITLES[B]}")
print("=" * 100)
print(f"{'查询变体':<26}{'chunk[' + str(A) + ']':>12}{'chunk[' + str(B) + ']':>12}"
      f"{'差值':>12}   谁高")
print("-" * 100)

for name, query in PROBES:
    scores = R.reranker.predict([[query, c["text"]] for c in R.chunks])
    sa, sb = scores[A], scores[B]
    winner = f"[{A}]" if sa > sb else (f"[{B}]" if sb > sa else "平")
    print(f"{name:<26}{sa:>12.4f}{sb:>12.4f}{sa - sb:>+12.4f}   {winner}")

# ==================== ③ 段落长度 / 标题 的影响 ====================
print()
print("=" * 100)
print("③ 只改【段落】，查询固定为「多 Agent 协作和 RAG 有什么不同」")
print("=" * 100)
query = "多 Agent 协作和 RAG 有什么不同"
text3 = R.chunks[B]["text"]
title3, _, body3 = text3.partition("\n")

variants = [
    ("原样", text3),
    ("去掉标题行", body3.strip()),
    ("只留标题", title3),
    ("只留正文", body3.strip()),
    ("标题 + 正文（同原样）", title3 + "\n" + body3.strip()),
    ("把标题重复两遍", title3 + "\n" + title3 + "\n" + body3.strip()),
]
print(f"{'段落变体':<24}{'精排分':>10}   文本前 30 字")
print("-" * 100)
for name, text in variants:
    s = R.reranker.predict([[query, text]])[0]
    print(f"{name:<24}{s:>10.4f}   {text[:30]}")
