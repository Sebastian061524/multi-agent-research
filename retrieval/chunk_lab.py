"""切块实验台：换几种切法，看检索效果怎么变。

为什么这件事必须先做一次「解耦」：
    eval_set 里的 gold 是【段落下标】（0~6），而段落下标是【切块的产物】。
    一换切法，段数变了、下标全变，gold 立即失效 —— 尺子就只能量「现在这一种切法」。

    已经解耦：gold 的锚点改成了「原文字符区间」（eval_set.CHUNK_SPANS），
    评测时比较【区间重叠】而不是下标（evaluate._covers）。
    所以本脚本可以放心地换切块，尺子依然有效。

⚠️ 唯一变量原则：本实验只换【切块】，其余全部固定 ——
   精排阈值、候选预算、BM25、停用词表、检索逻辑都一样。

⚠️ 复刻风险：`retrieve()` 是把生产代码 `_retrieve()` 的逻辑抄了一份
   （因为生产版写死了自己的 chunks/embeddings/bm25）。
   抄写会漂移 —— 所以启动时先做一次【自检】：用现状切块跑，
   结果必须和生产 `_retrieve()` 完全一致，否则直接报错退出。

用法（项目根目录）：
    python -m retrieval.chunk_lab
"""
import glob
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R
from retrieval.bm25 import STOPWORDS, BM25
from retrieval.eval_set import CHUNK_SPANS, EVAL_SET
from retrieval.evaluate import COVER_RATIO, score_one


# ==================== 切块策略 ====================
# 每个函数：原文 → list[{"text", "start", "end"}]
# start / end 是这段文字在【原文】里的字符区间 —— 评测靠它，不能省。

def chunk_current(text):
    """现状：按双换行切，丢掉 ≤20 字的块"""
    out, offset = [], 0
    for raw in text.split("\n\n"):
        start = offset
        offset += len(raw) + 2          # +2 是双换行本身
        lead = len(raw) - len(raw.lstrip())
        para = raw.strip()
        if len(para) > 20:
            out.append({"text": para, "start": start + lead,
                        "end": start + lead + len(para)})
    return out


def chunk_no_title(text):
    """变体 1：去掉每段的首行标题。

    动机：第 12 课测过，标题行会把精排分压低（0.0073 vs 去掉标题的 0.0748）。
          那个差异当时不改变结果，但它是结构性的 —— 值得单独验一次。
    """
    out = []
    for c in chunk_current(text):
        title, sep, body = c["text"].partition("\n")
        body = body.strip()
        if not sep or len(body) <= 20:
            continue
        out.append({"text": body,
                    "start": c["start"] + len(c["text"]) - len(body),
                    "end": c["start"] + len(c["text"])})
    return out


def chunk_fine(text):
    """变体 2：更细 —— 按句号切开，一句一块。

    动机：块越小越「聚焦」，可能提高精确率；但也可能丢掉上下文。
    """
    out = []
    for c in chunk_current(text):
        pos = 0
        for sent in c["text"].split("。"):
            lead = len(sent) - len(sent.lstrip())
            s = sent.strip()
            if len(s) > 10:
                start = c["start"] + pos + lead
                out.append({"text": s, "start": start, "end": start + len(s)})
            pos += len(sent) + 1        # +1 是那个句号
    return out


def chunk_coarse(text):
    """变体 3：更粗 —— 相邻两段合并（模拟「小节之间没空行、被粘成一段」）。

    动机：真实文档不一定有小节空行。合并之后 gold 区间会被更大的块覆盖，
          评测的区间匹配正好能处理这种情况（一段覆盖两个 gold → 两个都算命中）。
    """
    base = chunk_current(text)
    out = []
    for i in range(0, len(base), 2):
        group = base[i:i + 2]
        out.append({"text": "\n\n".join(g["text"] for g in group),
                    "start": group[0]["start"], "end": group[-1]["end"]})
    return out


STRATEGIES = [
    ("现状（双换行 + len>20）", chunk_current),
    ("去掉首行标题", chunk_no_title),
    ("更细（按句切）", chunk_fine),
    ("更粗（相邻两段合并）", chunk_coarse),
]


# ==================== 通用检索（复刻 _retrieve 的逻辑）====================

def read_corpus():
    """读全部语料（和 main.py 的加载方式一致）"""
    parts = []
    for pattern in [f"{R.DOCS_DIR}/**/*.txt", f"{R.DOCS_DIR}/**/*.md"]:
        for path in sorted(glob.glob(pattern, recursive=True)):
            with open(path, encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n".join(parts)


def build_index(chunks):
    """为一个切块方案建索引：向量 + BM25"""
    texts = [c["text"] for c in chunks]
    return {
        "chunks": chunks,
        "emb": R.model.encode(texts, normalize_embeddings=True),
        "bm25": BM25(texts, stopwords=STOPWORDS),
    }


def retrieve(index, query, max_k=5, min_score=R.RERANK_THRESHOLD, coarse_k=20):
    """复刻 main._retrieve()：混合粗筛（向量 ∪ BM25）→ 精排 → 阈值"""
    picked, seen = [], set()

    def take(i):
        i = int(i)
        if i not in seen:
            seen.add(i)
            picked.append(i)

    qv = R.model.encode(query, normalize_embeddings=True)
    vec_scores = index["emb"] @ qv
    for i in vec_scores.argsort()[::-1][:coarse_k]:
        take(i)
    for i in index["bm25"].ranking(query)[:coarse_k]:
        take(i)

    candidates = [index["chunks"][i] for i in picked]
    rr = R.reranker.predict([[query, c["text"]] for c in candidates])

    out = []
    for i in rr.argsort()[::-1]:
        if rr[i] < min_score:
            break
        out.append(candidates[i])
        if len(out) >= max_k:
            break
    return out


# ==================== 自检：复刻版必须等价于生产版 ====================

def self_check():
    text = read_corpus()
    chunks = chunk_current(text)

    # ① 切出来的区间必须和 main.py 加载的一致
    mine = [(c["start"], c["end"]) for c in chunks]
    prod = [(c["start"], c["end"]) for c in R.chunks]
    if mine != prod:
        sys.exit(f"❌ 自检失败：切块区间和生产不一致\n  复刻={mine}\n  生产={prod}")

    # ② 检索结果必须逐条一致
    index = build_index(chunks)
    bad = []
    for item in EVAL_SET:
        a = [(d["start"], d["end"]) for d in retrieve(index, item["query"])]
        b = [(d["start"], d["end"]) for d in R._retrieve(item["query"])]
        if a != b:
            bad.append((item["query"], a, b))
    if bad:
        sys.exit(f"❌ 自检失败：复刻的 retrieve 和生产 _retrieve 不一致，共 {len(bad)} 条\n"
                 + "\n".join(f"  {q}\n    复刻={a}\n    生产={b}" for q, a, b in bad[:3]))

    print(f"✅ 自检通过：复刻版与生产版在现状切块下完全一致（{len(EVAL_SET)} 条查询）\n")


# ==================== 跑 ====================

def run(name, chunks, min_score, text):
    index = build_index(chunks)
    rows = []
    returned_chars = []
    for item in EVAL_SET:
        docs = retrieve(index, item["query"], min_score=min_score)
        spans = [(d["start"], d["end"]) for d in docs]
        rows.append(score_one(spans, item["gold"]))
        returned_chars.append(sum(len(d["text"]) for d in docs))

    answered = [r for r in rows if r["recall"] is not None]
    abstain = [r for r in rows if r["abstained"] is not None]

    def avg(key):
        return sum(r[key] for r in answered) / len(answered) if answered else float("nan")

    lens = [len(c["text"]) for c in chunks]
    return {
        "name": name,
        "n": len(chunks),
        "avg_len": sum(lens) / len(lens) if lens else 0,
        # ⭐ 成本维度：每次查询平均要把多少字符塞进 prompt
        "cost": sum(returned_chars) / len(returned_chars) if returned_chars else 0,
        "recall": avg("recall"),
        "precision": avg("precision"),
        "mrr": avg("rr"),
        "ab": f"{sum(r['abstained'] for r in abstain)}/{len(abstain)}",
    }


def main():
    self_check()

    text = read_corpus()
    print(f"原文 {len(text)} 字符   候选预算 coarse_k=20")
    print(f"gold 锚点：原文字符区间（与切块无关），区间匹配比例 ≥ {COVER_RATIO:.0%}")
    print()
    print("⭐ 为什么要跑两种阈值：切块大小和精排阈值是【耦合】的 ——")
    print("   短块的精排分天然偏低，用 0.9 会把它们全砍掉；")
    print("   只跑一种阈值，就分不清「切块不行」和「阈值不适合这个切块」。")

    for min_score in (R.RERANK_THRESHOLD, 0.3):
        print()
        print("=" * 100)
        print(f"阈值 min_score = {min_score}")
        print("=" * 100)
        print(f"{'切块策略':<24}{'段数':>5}{'平均长度':>9}{'召回':>7}{'精确':>7}"
              f"{'MRR':>7}{'拒答':>7}{'返回字符':>10}")
        print("-" * 100)

        results = []
        for name, fn in STRATEGIES:
            r = run(name, fn(text), min_score, text)
            results.append(r)
            print(f"{name:<24}{r['n']:>5}{r['avg_len']:>9.0f}"
                  f"{r['recall']:>7.2f}{r['precision']:>7.2f}{r['mrr']:>7.2f}"
                  f"{r['ab']:>7}{r['cost']:>10.0f}")

        base = results[0]
        print("-" * 100)
        print("和【现状】比：")
        for r in results[1:]:
            print(f"  {r['name']:<24}"
                  f"召回 {r['recall'] - base['recall']:+.2f}  "
                  f"精确 {r['precision'] - base['precision']:+.2f}  "
                  f"MRR {r['mrr'] - base['mrr']:+.2f}  "
                  f"返回字符 {r['cost'] - base['cost']:+.0f}")


if __name__ == "__main__":
    main()
