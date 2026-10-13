"""把 BM25 的分数拆到 token 级别：一个段落为什么得这个分？

为什么要拆：
    「chunk[3] 没进前三」只是一个【结果】。要修它，得知道分数是【怎么算出来的】。

⭐ 这是 BM25 相对向量的一个真实优势：分数【可加、可解释】。
   向量相似度是一个整体值 —— 你没法说「这个 0.42 里，'RAG' 贡献了多少」。
   BM25 能，因为它的总分就是每个 token 贡献之和。

要回答的问题：
    同一个 chunk[3]，
      在「多 Agent 协作和 RAG 有什么不同」里 BM25 能捞到，
      在「Agent 的记忆机制和 RAG 有什么不同」里三种方式都捞不到 —— 差在哪？

用法（项目根目录）：
    python -m retrieval.bm25_dig
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R
from retrieval.bm25 import BM25, tokenize

TARGET = 3   # 要解剖的段落（RAG 检索增强生成）

CASES = [
    ("多 Agent 协作和 RAG 有什么不同", [2, 3]),
    ("Agent 的记忆机制和 RAG 有什么不同", [3, 4]),
]

bm25 = BM25([c["text"] for c in R.chunks])


def ranking_of(scores):
    return [int(i) for i in sorted(range(len(scores)),
                                   key=lambda i: scores[i], reverse=True)]


for query, gold in CASES:
    print("=" * 88)
    print(f"查询：{query}")
    print(f"gold：{gold}    （目标段落 chunk[{TARGET}]）")
    print("=" * 88)

    q_tokens = tokenize(query)
    print(f"\n① 查询切出的 token（{len(q_tokens)} 个）")
    print(f"   {q_tokens}")

    bm_scores = bm25.scores(query)
    bm_rank = ranking_of(bm_scores)

    qv = R.model.encode(query, normalize_embeddings=True)
    vec_scores = R.embeddings @ qv
    vec_rank = [int(i) for i in vec_scores.argsort()[::-1]]

    print("\n② 每段的两种分数与名次")
    print(f"   {'段落':<6}{'BM25分':>10}{'名次':>6}{'向量分':>10}{'名次':>6}   gold  讲的什么")
    print("   " + "-" * 74)
    for i, c in enumerate(R.chunks):
        title = c["text"].split("\n")[0][:14]
        star = "★" if i in gold else " "
        print(f"   [{i}]{'':<3}{bm_scores[i]:>10.3f}{bm_rank.index(i) + 1:>6}"
              f"{vec_scores[i]:>10.4f}{vec_rank.index(i) + 1:>6}    {star}    {title}")

    print(f"\n③ chunk[{TARGET}] 的 BM25 分数拆解（总分 {bm_scores[TARGET]:.3f}）")
    print(f"   {'token':<16}{'IDF':>9}{'本段出现':>10}{'贡献':>10}")
    print("   " + "-" * 46)
    n_hit = 0
    for w in q_tokens:
        contrib = bm25.term_score(TARGET, w)
        if contrib > 0:
            n_hit += 1
            print(f"   {w:<16}{bm25.idf.get(w, 0.0):>9.3f}"
                  f"{bm25.tf[TARGET].get(w, 0):>10}{contrib:>10.3f}")

    # ④ 再拆一个【和查询无关、却排得很高】的段落 —— 看它的分是从哪来的
    top_other = next(i for i in bm_rank if i not in gold)
    print(f"\n④ chunk[{top_other}] 的分数拆解（总分 {bm_scores[top_other]:.3f}）")
    print(f"   ⚠️ 这一段【不在 gold 里】，BM25 却把它排在第 {bm_rank.index(top_other) + 1} 名")
    print(f"   {'token':<16}{'IDF':>9}{'本段出现':>10}{'贡献':>10}")
    print("   " + "-" * 46)
    for w in q_tokens:
        contrib = bm25.term_score(top_other, w)
        if contrib > 0:
            print(f"   {w:<16}{bm25.idf.get(w, 0.0):>9.3f}"
                  f"{bm25.tf[top_other].get(w, 0):>10}{contrib:>10.3f}")

    missed = [w for w in q_tokens if w not in bm25.tf[TARGET]]
    elsewhere = [w for w in missed if any(w in tf for tf in bm25.tf)]
    print(f"\n   → chunk[{TARGET}] 命中 {n_hit}/{len(q_tokens)} 个 token")
    print(f"   → 完全不在 chunk[{TARGET}] 里的 token：{missed}")
    print(f"   → 其中【别的段落有、只有这一段没有】的：{elsewhere}")
    print()
