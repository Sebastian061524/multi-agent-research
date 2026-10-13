"""评测集体检：跑评测【之前】，先确认这把尺子本身没坏。

最典型的坏法（而且不报错）：
    从模板复制 query 时忘了改 gold
    → 某一段成了几乎所有查询的「标准答案」
    → 评测【永远显示及格】，你还以为系统很好

检查两个集：
    eval_set.py   —— 用来设计和调优的训练集
    holdout_set.py —— 只看一次的留出集（也要体检，但结果不许拿来调规则）

用法（项目根目录）：
    python -m retrieval.check_eval_set
"""
from collections import Counter

from retrieval.eval_set import EVAL_SET
from retrieval.holdout_set import HOLDOUT

CHUNK_COUNT = 7

# 哪些 kind 表示「库里没有答案，应该拒答」
NO_ANSWER_KINDS = ("近似无答案", "无关", "对抗-不应返回")


def check(entries, name: str) -> None:
    total = len(entries)

    print()
    print("#" * 70)
    print(f"#  {name}（共 {total} 条）")
    print("#" * 70)

    # ---------- ① 类型分布 ----------
    print("\n① 类型分布")
    kinds = Counter(item["kind"] for item in entries)
    for kind, n in kinds.most_common():
        print(f"  {kind:<16} {n:>3} 条")

    n_no_answer = sum(kinds.get(k, 0) for k in NO_ANSWER_KINDS)
    if n_no_answer:
        print(f"  → 应该拒答的共 {n_no_answer} 条（占 {n_no_answer / total:.0%}）")

    # ---------- ② gold 下标分布（重点）----------
    print("\n② gold 的下标分布   ← 这一段是重点")
    hits: Counter = Counter()
    for item in entries:
        for i in item["gold"]:
            hits[i] += 1
    for i in range(CHUNK_COUNT):
        print(f"  chunk[{i}]  {hits[i]:>3} 次  {'█' * hits[i]}")

    if hits:
        top, top_n = hits.most_common(1)[0]
        if top_n > total * 0.5:
            print(f"\n  ⚠️⚠️ chunk[{top}] 占了 {top_n}/{total} 条 —— 超过一半。")
            print("       最常见的原因：从模板复制 query 时忘了改 gold。")
            print("       后果：系统恰好也常返回 chunk[0]，评测会永远显示及格。")

    missing = [i for i in range(CHUNK_COUNT) if hits[i] == 0]
    if missing:
        print(f"\n  ⚠️  从没被当成标准答案的段落：{missing}")

    # ---------- ③ 其他检查 ----------
    print("\n③ 其他检查")
    seen = Counter(item["query"] for item in entries)
    dup = [q for q, n in seen.items() if n > 1]
    print(f"  重复的 query             : {dup if dup else '无'}")

    bad = [(item["query"], item["gold"]) for item in entries
           if any(i < 0 or i >= CHUNK_COUNT for i in item["gold"])]
    print(f"  越界的下标               : {bad if bad else '无'}")

    wrong_empty = [item["query"] for item in entries
                   if not item["gold"] and item["kind"] not in NO_ANSWER_KINDS]
    print(f"  gold 空但 kind 不是无答案 : {wrong_empty if wrong_empty else '无'}")

    # 留出集专有的检查：五个 kind 各几条
    if "对抗" in " ".join(kinds):
        print("\n  对抗样本的分布：")
        for kind in ("对抗-真漏判", "对抗-假漏判", "对抗-命中", "对抗-误判"):
            print(f"    {kind:<16} {kinds.get(kind, 0)} 条")
        for kind in ("对抗-真漏判", "对抗-假漏判", "对抗-命中", "对抗-误判"):
            if kinds.get(kind, 0) == 0:
                print(f"    ⚠️  {kind} 一条都没有 —— 这一种失败模式没被测到")


def main() -> None:
    check(EVAL_SET, "训练集 eval_set（可以反复看）")
    check(HOLDOUT, "留出集 holdout_set（只看一次，看完定论）")
    print()


if __name__ == "__main__":
    main()
