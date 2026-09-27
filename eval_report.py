"""报告层评测：用 LLM 裁判给最终报告打分

为什么需要第二层：
  检索层有标准答案（应该检索到哪几段），可以客观评测。
  但报告没有唯一正确答案 —— 只能定评分标准，让 LLM 打分。

注意：一次评测会跑完整的 Agent 流程（规划→调研→撰写→审校），
      每个主题约需 1~2 分钟。
"""
import re
import main as m


# ==================== 评测主题 ====================
TOPICS = [
    "大模型Agent有哪些主流架构",
    "RAG 和 Agent 的记忆机制如何协同",
    "怎么评测和观测一个 Agent 系统",
]


# ==================== 裁判 prompt ====================
KEYS = ["结构完整性", "忠实性", "覆盖度", "诚实标注", "综合"]

JUDGE_PROMPT = """你是一名严格的调研报告评审专家。

【调研主题】
{topic}

【可用资料（报告只能以此为据）】
{materials}

【待评审的报告】
{report}

请从以下四个维度评分（0-10 分），每项都要给理由。

1. 结构完整性
   10 分 = 概述、分点论述、结论三部分齐全，层次分明
   0 分  = 结构混乱或严重缺失

2. 忠实性（最重要）
   10 分 = 报告中每一条事实都能在资料中找到依据
   0 分  = 大量内容在资料中找不到依据（编造）
   注意：合理的概括、归纳、跨段落对比不算编造；
        只有资料里根本没有的具体事实/数据/框架才算编造。

3. 覆盖度
   10 分 = 资料中的关键信息基本都被用上了
   0 分  = 几乎没用到资料内容

4. 诚实标注
   10 分 = 资料不足以回答的子问题，报告都明确写出了「资料未涵盖」
   0 分  = 资料不足却强行作答或含糊带过
   如果资料充足、没有缺口，本项给 10 分。

输出格式（严格遵守，每行一项，不要输出任何其他内容）：
结构完整性：X分 | 理由
忠实性：X分 | 理由
覆盖度：X分 | 理由
诚实标注：X分 | 理由
综合：X分 | 一句话总评
"""


def judge_report(topic, report, materials):
    """让 LLM 当裁判，给报告打分"""
    prompt = JUDGE_PROMPT.format(topic=topic, report=report, materials=materials)
    resp = m.llm.invoke(prompt)
    return resp.content


def parse_scores(text):
    """从裁判输出里提取分数"""
    scores = {}
    for key in KEYS:
        match = re.search(rf"{key}\s*[:：]\s*(\d+(?:\.\d+)?)", text)
        scores[key] = float(match.group(1)) if match else None
    return scores


def run_one(topic):
    """跑一遍完整 Agent 流程，拿到报告、资料和撰写次数"""
    result = m.graph.invoke({"topic": topic})

    report = result["report"]
    revision = result.get("revision_count", 1)
    materials = "\n\n".join(
        f"### 子问题：{f['question']}\n{f.get('summary', f['materials'])}"
        for f in result["findings"]
    )
    return report, materials, revision


if __name__ == "__main__":
    all_scores = []
    all_revisions = []

    for i, topic in enumerate(TOPICS, 1):
        print("\n" + "=" * 72)
        print(f"[{i}/{len(TOPICS)}] 评测主题：{topic}")
        print("=" * 72)

        report, materials, revisions = run_one(topic)
        all_revisions.append(revisions)
        print(f"\n>>> 撰写次数：{revisions}（1 = 一次通过，无返工）")

        print("\n--- 裁判评语 ---")
        verdict = judge_report(topic, report, materials)
        print(verdict)

        scores = parse_scores(verdict)
        all_scores.append(scores)

    # ==================== 汇总 ====================
    print("\n" + "=" * 72)
    print("【汇总】")
    print("=" * 72)
    print("主题".ljust(28) + "".join(k.ljust(9) for k in KEYS) + "撰写次数")
    print("-" * 72)
    for topic, scores, rev in zip(TOPICS, all_scores, all_revisions):
        row = topic[:26].ljust(28)
        for k in KEYS:
            v = scores.get(k)
            row += (f"{v:g}" if v is not None else "-").ljust(9)
        row += str(rev)
        print(row)

    print("-" * 72)
    avg_row = "平均分".ljust(28)
    for k in KEYS:
        vals = [s[k] for s in all_scores if s.get(k) is not None]
        avg = sum(vals) / len(vals) if vals else 0
        avg_row += f"{avg:.1f}".ljust(9)
    avg_row += f"{sum(all_revisions)/len(all_revisions):.1f}"
    print(avg_row)