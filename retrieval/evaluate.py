"""检索评测：跑基线 + 对比不同的检索策略。

⭐ 四个指标各量一件事，别混：
    Recall    ：该找到的段落，有几成真的出现在返回里      → 量【漏没漏】
    Precision ：返回的那几段里，有几成是真正需要的        → 量【多没多】
    MRR       ：第一个正确段落排在第几位（倒数的平均）      → 量【排得靠不靠前】
    拒答正确率 ：应该拒答的查询，系统真的返回空了吗         → 量【该不该答】

为什么必须分开量：
    「检索效果不好」是几种完全不同的病，药也完全不同 ——
    召回差要动【查询改写 / 混合检索】，精确差要动【判定逻辑】，排序差要动【精排】。
    少一个指标，就少一只眼睛：某个方向坏了，你根本看不见。

⚠️ 对照实验的铁律：两组之间只能有【一个】变量不同。

用法（项目根目录）：
    python -m retrieval.evaluate              → 用训练集 eval_set
    python -m retrieval.evaluate holdout      → 用留出集 holdout_set

⚠️ 留出集的规矩：只看一次。看完不许回头改规则再来测这一批 ——
   那就等于把它变成了训练集。
"""
import sys

import main as R

if "holdout" in sys.argv:
    from retrieval.holdout_set import HOLDOUT as DATA
    DATA_NAME = "留出集 holdout_set（只看一次）"
else:
    from retrieval.eval_set import EVAL_SET as DATA
    DATA_NAME = "训练集 eval_set（可反复调优）"

# _retrieve 返回的是段落 dict，而指标需要「下标」才能比。
# 用文本内容做映射 —— 语料里每段文本都是唯一的。
IDX_OF = {c["text"]: i for i, c in enumerate(R.chunks)}

# 类型顺序按数据里【第一次出现】的先后排，不用手工维护
KINDS = list(dict.fromkeys(item["kind"] for item in DATA))

# ==================== 要对比的检索策略 ====================
# 只改这个列表就能加对照组。第一项是基线。
#   _retrieve                  ：单查询两阶段检索
#                                （= 现在 MCP 服务 smart_query=False 时的行为）
#   search(use_decompose=True) ：先把问题拆成子查询，各自检索再合并去重
#
# 「智能分解」的思路：decompose 要花一次 LLM 调用，别对每条查询都花。
# 先用【文字规则】判断像不像复合问题，可疑的才调 LLM。
#
# ⚠️ 为什么用文字规则而不是检索分数：
#    诊断数据显示，复合查询在精排分上也是「一个段落一枝独秀」
#    （0.9881 vs 其余 <0.01），和单概念查询长得一模一样 —— 分数判别不了。
COMPOUND_MARKERS = ("和", "与", "、", "以及", "还有", "对比", "区别",
                    "有什么不同", "哪个", "分别")

# 统计「智能分解」到底对几条查询花了 LLM 调用
SMART_STATS = {"llm_calls": 0, "total": 0}


def looks_compound(query: str) -> bool:
    """便宜的预判：查询里有没有并列 / 比较结构的词"""
    return any(marker in query for marker in COMPOUND_MARKERS)


def smart_search(query):
    """可疑才分解 —— 其余直接单查询检索，省掉 LLM 调用"""
    SMART_STATS["total"] += 1
    if looks_compound(query):
        SMART_STATS["llm_calls"] += 1
        return R.search(query, use_decompose=True)
    return R._retrieve(query)


RETRIEVERS = [
    ("单查询（现在的默认）", lambda q: R._retrieve(q)),
    ("带查询分解（全开）", lambda q: R.search(q, use_decompose=True)),
    ("智能分解（规则预筛）", smart_search),
]


# ==================== 打分 ====================

def to_indices(docs):
    """把检索返回的段落 dict 翻译成【下标列表】（保持顺序）"""
    return [IDX_OF[d["text"]] for d in docs]


def score_one(got, gold):
    """算一条查询的指标。

    recall    ：命中的 gold 数 / gold 总数     （该找到的，找到了几成）
    precision ：命中的 gold 数 / 返回的段数     （返回的，有几成是对的）
    rr        ：1 / 第一个正确段落的排名       （排得靠不靠前）
    abstained ：该拒答的查询，返回是否为空
    """
    gold = set(gold)

    if not gold:
        # 应该拒答的查询：返回空 = 判对了
        return {"recall": None, "precision": None, "rr": None,
                "abstained": len(got) == 0}

    hit = set(got) & gold
    recall = len(hit) / len(gold)
    # 什么都没返回时 precision 记为 1.0（没有噪音）——
    # 「什么都没给」由 recall 去反映，两个指标各管一头。
    precision = len(hit) / len(got) if got else 1.0
    ranks = [i + 1 for i, g in enumerate(got) if g in gold]
    rr = 1 / ranks[0] if ranks else 0.0
    return {"recall": recall, "precision": precision, "rr": rr, "abstained": None}


def run_strategy(fn):
    """把评测集完整跑一遍，返回每条一行的结果"""
    rows = []
    for item in DATA:
        got = to_indices(fn(item["query"]))
        rows.append({"item": item, "got": got, **score_one(got, item["gold"])})
    return rows


def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def summarize(rows):
    """把一堆行压成四个总数"""
    answered = [r for r in rows if r["recall"] is not None]
    abstain = [r for r in rows if r["abstained"] is not None]
    ab_ok = sum(r["abstained"] for r in abstain)
    return {
        "recall": avg(answered, "recall"),
        "precision": avg(answered, "precision"),
        "mrr": avg(answered, "rr"),
        "ab": f"{ab_ok}/{len(abstain)}",
    }


# ==================== 打印 ====================

def print_kind_table(rows, title):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'类型':<12}{'条数':>5}{'召回':>8}{'精确':>8}{'MRR':>8}{'拒答正确':>11}")
    print("-" * 80)

    for kind in KINDS:
        sub = [r for r in rows if r["item"]["kind"] == kind]
        if not sub:
            continue
        ans = [r for r in sub if r["recall"] is not None]
        need_ab = [r for r in sub if r["abstained"] is not None]

        rec = f"{avg(ans, 'recall'):.2f}" if ans else "—"
        pre = f"{avg(ans, 'precision'):.2f}" if ans else "—"
        mrr = f"{avg(ans, 'rr'):.2f}" if ans else "—"
        ab = f"{sum(r['abstained'] for r in need_ab)}/{len(need_ab)}" if need_ab else "—"

        print(f"{kind:<12}{len(sub):>5}{rec:>8}{pre:>8}{mrr:>8}{ab:>11}")

    s = summarize(rows)
    print("-" * 80)
    print(f"{'总体':<12}{len(rows):>5}{s['recall']:>8.2f}{s['precision']:>8.2f}"
          f"{s['mrr']:>8.2f}{s['ab']:>11}")
    print("=" * 80)


def print_detail(name, rows, i):
    """把某个策略在某条查询上的结果压成一行"""
    r = rows[i]
    if r["abstained"] is not None:
        tail = "返回空（正确拒答）" if r["abstained"] else f"却返回了 {r['got']}"
    else:
        tail = (f"取到{r['got']}   召回{r['recall']:.2f}  "
                f"精确{r['precision']:.2f}  rr={r['rr']:.2f}")
    print(f"    {name:<22}{tail}")


def main():
    print(f"数据集：{DATA_NAME}    共 {len(DATA)} 条")
    print(f"（精排阈值 RERANK_THRESHOLD={R.RERANK_THRESHOLD}，最多取 max_k=5）")
    print(f"要对比的策略：{len(RETRIEVERS)} 个\n")

    # ---- 逐个策略跑完整评测 ----
    results = []
    for name, fn in RETRIEVERS:
        print(f"▶ 正在跑：{name} ...")
        results.append((name, run_strategy(fn)))
        print()

    for name, rows in results:
        print_kind_table(rows, f"策略：{name}")
        print()

    # ---- 对比表 ----
    print("=" * 80)
    print("对比")
    print("=" * 80)
    print(f"{'策略':<24}{'召回':>8}{'精确':>8}{'MRR':>8}{'拒答':>10}")
    print("-" * 80)
    for name, rows in results:
        s = summarize(rows)
        print(f"{name:<24}{s['recall']:>8.2f}{s['precision']:>8.2f}"
              f"{s['mrr']:>8.2f}{s['ab']:>10}")
    print("=" * 80)

    # ---- 只打有变化的条目 ----
    print("\n" + "=" * 80)
    print("两个策略结果【不同】的查询")
    print("=" * 80)

    base_rows = results[0][1]
    changed = 0
    for i, item in enumerate(DATA):
        if all(rows[i]["got"] == base_rows[i]["got"] for _, rows in results[1:]):
            continue
        changed += 1
        gold_s = item["gold"] if item["gold"] else "（空，应该拒答）"
        print(f"\n  [{item['kind']}] {item['query']}")
        print(f"    gold = {gold_s}")
        for name, rows in results:
            print_detail(name, rows, i)

    if changed == 0:
        print("  （所有策略的结果完全一样）")
    print(f"\n  共 {changed}/{len(DATA)} 条结果不同")

    # ---- 智能分解到底省了多少次 LLM 调用 ----
    if SMART_STATS["total"]:
        n, t = SMART_STATS["llm_calls"], SMART_STATS["total"]
        print(f"\n智能分解的 LLM 调用：{n}/{t} 条查询（{n / t:.0%}）"
              f"    —— 「全开」是 {t}/{t}（100%）")


if __name__ == "__main__":
    main()
