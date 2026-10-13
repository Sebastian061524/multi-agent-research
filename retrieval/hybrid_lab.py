"""混合检索实验台：相同候选预算下，比较三种召回方式。

为什么必须固定预算：
    库里只有 7 段。如果预算是 20（现在的 coarse_k），候选池就是全集，
    三种方式的召回率都是 100% —— 什么都测不出来。
    把预算压到 3，召回层才真的在「挑」，差别才显形。

三种候选来源：
    向量  —— 稠密检索，擅长「意思相近、用词不同」
    BM25 —— 稀疏检索，擅长「精确的词匹配」
    融合  —— RRF，只看排名、不看分数

零件在 retrieval/bm25.py（tokenize / BM25 / rrf），本文件只做实验。

用法（项目根目录）：
    python -m retrieval.hybrid_lab
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R
from retrieval.bm25 import STOPWORDS, BM25, rrf
from retrieval.eval_set import EVAL_SET

# 候选预算：必须小于语料段数，召回层才真的在筛。
# 跑三档 —— 预算越紧，融合（RRF）的优势才越可能显出来；
# 预算放到 3 以上时多路往往都到天花板，什么都测不出。
BUDGETS = (1, 2, 3)


def vector_ranking(query: str) -> list[int]:
    """按向量相似度给出全部段落的排名"""
    qv = R.model.encode(query, normalize_embeddings=True)
    scores = R.embeddings @ qv
    return [int(i) for i in scores.argsort()[::-1]]


def recall_of(rows, key, budget=BUDGETS[0]):
    """前 budget 个候选里，覆盖了多少 gold

    ⚠️ rows 里每一行是 {"item": 评测项, "向量": [...], ...}，
       所以要取 row["item"]["gold"]，不是 row["gold"]。
    """
    hit = miss = 0
    for row in rows:
        gold = set(row["item"]["gold"])
        if not gold:
            continue
        cand = set(row[key][:budget])
        hit += len(cand & gold)
        miss += len(gold - cand)
    return hit / (hit + miss) if (hit + miss) else 0.0


KEYS = ("向量", "BM25", "BM25+停用词", "融合")


def build_rows():
    """对每条有标准答案的查询，算出四种候选排名

    唯一的新变量是【有没有停用词】—— 其余全都一样。
    """
    bm25_plain = BM25([c["text"] for c in R.chunks])
    bm25_sw = BM25([c["text"] for c in R.chunks], stopwords=STOPWORDS)

    rows = []
    for item in EVAL_SET:
        if not item["gold"]:
            continue
        vec = vector_ranking(item["query"])
        bm = bm25_plain.ranking(item["query"])
        bm_sw = bm25_sw.ranking(item["query"])
        rows.append({
            "item": item,
            "向量": vec,
            "BM25": bm,
            "BM25+停用词": bm_sw,
            "融合": rrf([vec, bm_sw]),
        })
    return rows


def main():
    rows = build_rows()

    print(f"语料 {len(R.chunks)} 段   有标准答案的查询 {len(rows)} 条")
    print(f"停用词表：{len(STOPWORDS)} 个字\n")

    # ---------- 召回率矩阵 ----------
    print("=" * 80)
    print("召回率矩阵（行 = 候选来源，列 = 候选预算 k）")
    print("=" * 80)
    print(f"{'候选来源':<16}" + "".join(f"{'k=' + str(k):>15}" for k in BUDGETS))
    print("-" * 80)
    for key in KEYS:
        cells = "".join(f"{recall_of(rows, key, k):>14.2%} " for k in BUDGETS)
        print(f"{key:<16}{cells}")
    print("=" * 80)

    # ---------- 融合 vs 最好的单路 ----------
    print("\n⭐ 「融合」和「最好的单路」比 —— 这才是 RRF 值不值的判据：")
    for k in BUDGETS:
        singles = ("向量", "BM25", "BM25+停用词")
        best = max(singles, key=lambda key: recall_of(rows, key, k))
        r_best = recall_of(rows, best, k)
        r_fuse = recall_of(rows, "融合", k)
        delta = r_fuse - r_best
        verdict = ("融合更好 ✅" if delta > 1e-9
                   else ("打平 ➖" if abs(delta) <= 1e-9 else "融合更差 ❌"))
        print(f"   k={k}   融合 {r_fuse:>7.2%}   最好单路（{best}）{r_best:>7.2%}"
              f"   差 {delta:>+7.2%}   → {verdict}")

    # ---------- 按类型拆开（最紧的预算）----------
    tight = BUDGETS[0]
    print(f"\n按类型拆开（最紧的预算 k={tight}，差别最明显）：")
    print(f"{'类型':<10}{'条数':>5}" + "".join(f"{k:>14}" for k in KEYS))
    print("-" * (15 + 14 * len(KEYS)))
    for kind in dict.fromkeys(r["item"]["kind"] for r in rows):
        sub = [r for r in rows if r["item"]["kind"] == kind]
        cells = "".join(f"{recall_of(sub, k, tight):>13.2%} " for k in KEYS)
        print(f"{kind:<10}{len(sub):>5}{cells}")

    # ---------- 逐条明细（最紧的预算）----------
    print(f"\n逐条明细（k={tight}，每个方式只给 1 个候选）：")
    for r in rows:
        item = r["item"]
        gold = set(item["gold"])
        print(f"  gold={item['gold']}  {item['query'][:30]}")
        for key in KEYS:
            top = r[key][:tight]
            got = len(set(top) & gold)
            mark = "✅" if gold <= set(top) else ("⚠️" if got else "❌")
            print(f"      {key:<14}{mark} {top}")


if __name__ == "__main__":
    main()
