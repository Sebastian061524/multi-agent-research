"""裁判校准：用故意做坏的报告测裁判是否可靠

为什么必须做：
  LLM 裁判本身也是 LLM，也会犯错。
  如果不校准，评测出来的高分可能是「假数据」。
"""
import eval_report as e


TOPIC = "大模型Agent有哪些主流架构"

MATERIALS = """### ReAct 架构
ReAct 是最经典的 Agent 架构，核心思想是「推理-行动-观察」循环。模型先思考当前应该做什么，然后调用工具执行行动，观察结果后再决定下一步。优点是简单可靠、易于实现；缺点是对于复杂任务容易陷入局部最优，缺乏全局规划。

### 多 Agent 协作
多 Agent 架构把复杂任务拆分给多个专职 Agent。常见模式包括 Orchestrator-Worker、Pipeline、Debate、Hierarchical。优势是职责分离、可并行、上下文隔离；缺点是复杂度和成本显著上升，调试困难。经验法则是：能用一个 Agent 解决的问题，不要拆成多个。"""


# ==================== 好报告（对照组）====================
GOOD = """## 概述
本报告基于所给资料，梳理大模型 Agent 的两类主流架构。

## 分点论述
### 一、ReAct 架构
核心是「推理-行动-观察」循环：先思考、再调用工具行动、观察结果后决定下一步。
优点是简单可靠、易于实现；缺点是复杂任务容易陷入局部最优，缺乏全局规划。

### 二、多 Agent 协作
常见模式有 Orchestrator-Worker、Pipeline、Debate、Hierarchical 四种。
优势：职责分离、可并行执行、上下文隔离。缺点：复杂度和成本显著上升、调试困难。
资料给出的经验法则是：能用一个 Agent 解决的问题，不要拆成多个。

## 结论
两类架构各有取舍。选型时应遵循「能用一个 Agent 解决就不要拆分」的原则。
资料未涵盖两类架构的性能量化对比。"""


# ==================== 坏报告 1：大量编造 ====================
BAD_FABRICATED = """## 概述
大模型 Agent 架构正在快速发展，本报告梳理主流架构。

## 分点论述
### 一、ReAct 架构
ReAct 由 Google Brain 团队于 2022 年提出，论文引用量已超过 8000 次。
在 MMLU 基准测试中达到 89.3% 的准确率，在 HotpotQA 上达到 71.2%。
OpenAI 的 o1 和 Anthropic 的 Claude 均采用了类似架构。

### 二、多 Agent 协作
AutoGen 是微软开源的多 Agent 框架，采用异步消息传递机制。
MetaGPT 由 DeepWisdom 开发，支持 SOP 标准化流程。
CAMEL 框架通过角色扮演实现 Agent 间自主协作。

## 结论
ReAct 与多 Agent 协作是当前两大主流架构。"""


# ==================== 坏报告 2：结构混乱 ====================
BAD_MESSY = """ReAct 是一种架构，它有一些优点也有一些缺点。多 Agent 也不错，
可以拆开来做事情。具体怎么选要看情况，一般来说简单的就用简单的，
复杂的就用复杂的。还有一些别的事情也要考虑，比如成本什么的。
总之要根据实际情况来判断，没有标准答案。"""


# ==================== 坏报告 3：内容极少 ====================
BAD_INCOMPLETE = """## 概述
略。

## 分点论述
ReAct 是一种 Agent 架构。

## 结论
ReAct 很重要。"""


CASES = [
    ("✅ 对照组：好报告", GOOD),
    ("❌ 坏报告 1：大量编造（忠实性应该低）", BAD_FABRICATED),
    ("❌ 坏报告 2：结构混乱（结构性应该低）", BAD_MESSY),
    ("❌ 坏报告 3：内容极少（覆盖度应该低）", BAD_INCOMPLETE),
]


# ==================== 期望区间（校准判据）====================
# 裁判对每份报告的指定维度打分，应当落在这个区间内，否则视为裁判不可靠
EXPECT = {
    "✅ 对照组：好报告":                      {"结构完整性": (8, 10), "忠实性": (8, 10)},
    "❌ 坏报告 1：大量编造（忠实性应该低）":    {"忠实性": (0, 3), "诚实标注": (0, 4)},
    "❌ 坏报告 2：结构混乱（结构性应该低）":    {"结构完整性": (0, 4), "覆盖度": (0, 5)},
    "❌ 坏报告 3：内容极少（覆盖度应该低）":    {"覆盖度": (0, 4), "结构完整性": (0, 5)},
}


if __name__ == "__main__":
    all_ok = True

    for name, report in CASES:
        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)

        verdict = e.judge_report(TOPIC, report, MATERIALS)
        print(verdict)

        scores = e.parse_scores(verdict)
        print("\n解析分数：")
        for k, v in scores.items():
            print(f"    {k}：{v}")

        # ---------- 自动判定：分数是否落在期望区间 ----------
        for dim, (lo, hi) in EXPECT.get(name, {}).items():
            got = scores.get(dim)
            if got is None:
                print(f"    ⚠️  {dim}：未解析到分数（裁判输出格式可能异常）")
                all_ok = False
            elif not (lo <= got <= hi):
                print(f"    ❌ {dim}：期望 {lo}~{hi}，实际 {got}")
                all_ok = False

    print("\n" + "=" * 72)
    if all_ok:
        print("✅ 裁判校准通过：所有维度都落在期望区间内，裁判可信")
    else:
        print("❌ 裁判校准未通过：请检查上面的异常项，可能需要调整裁判 prompt")
    print("=" * 72)