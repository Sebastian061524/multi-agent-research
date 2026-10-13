"""对失败的查询做逐级诊断：看清它到底死在哪一级。

为什么先诊断再修：
    三个失败案例只告诉我们「结果不对」。
    死因可能是【检索没找到】、【找到了但被阈值砍了】、【排序排错了】……
    药完全不同。先把分数打出来，再决定动哪个零件。

用法（项目根目录）：
    python -m retrieval.diagnose
"""
import os
import sys

# 让「直接跑文件」和「-m retrieval.diagnose」两种方式都能找到项目根的 main.py
# 直接跑文件时 sys.path[0] 是 retrieval\，而不是项目根
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R

# (查询, gold) —— gold 为空表示「应该拒答」

CASES = [
    ("多 Agent 协作和 RAG 有什么不同", [2, 3]),
    ("Agent 的记忆机制和 RAG 有什么不同", [3, 4]),
    ("Langfuse 和 LangSmith 哪个便宜", []),
]

for query, gold in CASES:
    print("=" * 80)
    print(f"查询：{query}")
    print(f"gold：{gold}    （空 = 应该拒答）")
    print("=" * 80)

    qv = R.model.encode(query, normalize_embeddings=True)
    vec = R.embeddings @ qv
    rr = R.reranker.predict([[query, c["text"]] for c in R.chunks])

    print(
        f"  {'段落':<8}{'向量分':>10}{'精排分':>11}{'过阈值':>10}   {'gold':<6}讲的什么"
    )
    print("  " + "-" * 74)
    for i, c in enumerate(R.chunks):
        passed = "✅ 过" if rr[i] >= R.RERANK_THRESHOLD else "❌ 被砍"
        star = "★" if i in gold else ""
        title = c["text"].split("\n")[0][:14]
        print(
            f"  [{i}]{'':<5}{vec[i]:>+10.4f}{rr[i]:>+11.4f}{passed:>10}   {star:<6}{title}"
        )

    order = rr.argsort()[::-1]
    returned = [int(i) for i in order if rr[i] >= R.RERANK_THRESHOLD][:5]
    print(f"\n  实际返回：{returned}")
    print(f"  应该包含：{gold if gold else '（空）'}")
    print()
