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

# gold 的锚点：原文字符区间（与切块无关）。两个集都是按同一份 7 段标的，所以共用。
from retrieval.eval_set import CHUNK_SPANS

# _retrieve 返回的是段落 dict，而指标需要「下标」才能比。
# 用文本内容做映射 —— 语料里每段文本都是唯一的。
IDX_OF = {c["text"]: i for i, c in enumerate(R.chunks)}

# 类型顺序按数据里【第一次出现】的先后排，不用手工维护
KINDS = list(dict.fromkeys(item["kind"] for item in DATA))

# ==================== 要对比的检索策略 ====================
# 只改这个列表就能加对照组。第一项是基线。
#
# 当前这一组在验【第 12 课复查 gold 时冒出的假设】：
#   多返回的那 6 段里，4 段和 gold 的分差是 0.15~0.45（明显更低）。
#   ⇒ 阈值 0.3 太松，把它们放进来了。把阈值调到 0.9 应该能砍掉。
#   唯一变量 = RERANK_THRESHOLD。
#
# ⚠️ 这同时也在验一个【已写进文档的结论】：第 4 课说「阈值救不了」
#    （因为分数饱和，区间只有 0.0005 宽）。复查发现那个论证把
#    不同查询的需求混在一起了 —— 它只对「分差 < 0.01」的 2 段成立。
#
# 历史对照组（已验完，结论见 docs/检索开发实录.md）：
#   预算 20 / 3 × 纯向量 / 混合  → 最终结果 0/19 变化，瓶颈在精排层不在召回层
#   search(use_decompose=True)   → 有效，MCP 的 smart_query 默认已改成 True
#   「智能分解」文字规则          → 留出集上失败，已废弃，代码也删掉了
RETRIEVERS = [
    ("改前：阈值 0.3", lambda q: R._retrieve(q, min_score=0.3)),
    ("改后：阈值 0.9", lambda q: R._retrieve(q, min_score=0.9)),
    # 回归检查：生产默认（不传 min_score）必须和「改后」完全一致，
    # 否则说明某个调用点还在用旧值。
    ("生产默认（不传参）", lambda q: R._retrieve(q)),
]


# ==================== 打分 ====================
#
# ⭐ 匹配方式：比【原文区间】，不比【段落下标】。
#
#   为什么：段落下标是切块的产物 —— 换一种切法（改大小 / 加 overlap / 按标题切），
#   段数就变、下标全变，gold 立即失效，尺子只能量「现在这一种切法」。
#   gold 的锚点固定成 eval_set.CHUNK_SPANS（原文字符区间），就与切块解耦了。
#
#   判定「命中」的规则：返回的段落覆盖了 gold 区间的 **50%** 以上。
#   · 切块变小（gold 被切成两半）：任意一半覆盖 ≥50% 就算命中 ✅
#   · 切块变大（两段合成一段）：那一段同时覆盖两个 gold 区间 → 两个都算命中 ✅
#   · 50% 这个数是【判定标准】，不是物理常数 —— 改它等于改尺子，会影响 precision。

COVER_RATIO = 0.5


def _span_of(doc):
    """取一个返回段落的原文区间"""
    return (doc["start"], doc["end"])


def _covers(got_span, gold_span, ratio=COVER_RATIO):
    """返回段落是否覆盖了 gold 区间（按 gold 区间的比例算）"""
    gs, ge = gold_span
    s, e = got_span
    overlap = max(0, min(e, ge) - max(s, gs))
    width = ge - gs
    return width > 0 and overlap / width >= ratio


def score_one(got_spans, gold_idx):
    """算一条查询的指标。

    got_spans ：实际返回的段落，每个的 (start, end)
    gold_idx  ：标准答案的段落下标（用 CHUNK_SPANS 换成原文字符区间）

    recall    ：被覆盖到的 gold 区间数 / gold 总数
    precision ：返回的段落里，有几段至少覆盖了一个 gold 区间
    rr        ：1 / 第一个「有用的」返回段落的排名
    abstained ：该拒答的查询，返回是否为空
    """
    if not gold_idx:
        # 应该拒答的查询：返回空 = 判对了
        return {"recall": None, "precision": None, "rr": None,
                "abstained": len(got_spans) == 0}

    gold_spans = [CHUNK_SPANS[i] for i in gold_idx]

    # 每个返回段落是否「有用」
    useful = [any(_covers(sp, g) for g in gold_spans) for sp in got_spans]

    # 每个 gold 区间是否被某个返回段落覆盖到
    hit = sum(1 for g in gold_spans if any(_covers(sp, g) for sp in got_spans))

    recall = hit / len(gold_spans)
    # 什么都没返回时 precision 记为 1.0（没有噪音）——
    # 「什么都没给」由 recall 去反映，两个指标各管一头。
    precision = sum(useful) / len(got_spans) if got_spans else 1.0
    ranks = [i + 1 for i, ok in enumerate(useful) if ok]
    rr = 1 / ranks[0] if ranks else 0.0
    return {"recall": recall, "precision": precision, "rr": rr, "abstained": None}


def run_strategy(fn):
    """把评测集完整跑一遍，返回每条一行的结果"""
    rows = []
    for item in DATA:
        docs = fn(item["query"])
        got_spans = [_span_of(d) for d in docs]
        got = [IDX_OF.get(d["text"], -1) for d in docs]   # 只用于打印
        rows.append({"item": item, "got": got, "spans": got_spans,
                     **score_one(got_spans, item["gold"])})
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


if __name__ == "__main__":
    main()
