"""验证：feedback 修复是否真的生效

背景：
  `feedback`（上一版的审校意见）曾经构造了却**从未拼进撰写 prompt**，
  导致「返工重写」等同于重新生成，而不是针对性修改。
  修复后加了 `{feedback}`。

  但这个修复**只在「返工」路径上生效**，而当前版本审校一次就通过，
  也就是说这条路径从来没被跑到 —— 常规评测验证不到它。
  所以需要专门构造条件。

本脚本做两层验证：
  ① 直接检查 prompt：mock 掉 LLM，确认审校意见确实被拼进去了（秒级，无 API 调用）
  ② A/B 对照实验：拿一份有明显缺陷的报告，分别「有反馈」和「无反馈」重写，
     看有反馈的那一版是否真的修掉了意见里指出的问题
"""
import main


# ==================== 测试数据 ====================

TOPIC = "大模型Agent有哪些主流架构"

FINDINGS = [{
    "question": "ReAct 架构的原理是什么",
    "materials": "【资料】ReAct 是最经典的 Agent 架构，核心思想是「推理-行动-观察」循环。"
                 "优点是简单可靠、易于实现；缺点是对于复杂任务容易陷入局部最优，缺乏全局规划。",
    "summary": "ReAct 采用「推理-行动-观察」循环。优点：简单可靠、易于实现。"
               "缺点：复杂任务容易陷入局部最优，缺乏全局规划。",
}]

# 第 1 版：故意有两个明确缺陷（缺结论 + 漏了资料里的一个重要缺点）
REPORT_V1 = """## 概述
本报告介绍 ReAct 架构的基本原理。

## 分点论述
ReAct 采用「推理-行动-观察」循环：先思考、再调用工具行动、观察结果后决定下一步。
优点是简单可靠、易于实现。
"""

# 审校意见：逐条指出可修改的问题
# 第 3 条是「只有看了反馈才会做」的改动 —— 用来区分「有反馈」和「无反馈」
REVIEW = """不通过
1. 结构缺失：报告没有「## 结论」部分，请补充。
2. 覆盖不足：资料中提到的缺点「缺乏全局规划」没有写进报告，请补上。
3. 请在结论的最后一行，原样加上这句话：「本报告未使用资料之外的信息。」
"""


# ==================== ① 检查 prompt 里有没有审校意见 ====================

def check_prompt():
    print("=" * 72)
    print("① 检查发往 LLM 的 prompt 是否包含审校意见（mock LLM，无 API 调用）")
    print("=" * 72)
    # ② 把真的 LLM 换成假的（避免真的花钱调 API） # 用来存下 prompt
    captured = {}

    class FakeResponse:
        content = "（这里是模型返回的第 2 版报告）"

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt   # 不真调用，只把 prompt 存下来
            return FakeResponse()

    real_llm = main.llm          # 保存真实 LLM
    main.llm = FakeLLM()         # 替换成假的
    try:
        # ① 造一个假的 State —— 关键是带上 review
        state = {
            "topic": TOPIC,
            "sub_questions": ["ReAct 架构的原理是什么"],
            "findings": FINDINGS,
            "report": REPORT_V1,
            "review": REVIEW,        # ← 关键：模拟「审校不通过、正在返工」
            "revision_count": 1,
        }
        result = main.write_node(state)
    finally:
        main.llm = real_llm      # 无论成败都还原

    prompt = captured.get("prompt", "")
    # ③ 检查存下来的 prompt 里，有没有那句话
    checks = [
        ("审校意见引导语", "上一版的审校意见"),
        ("意见第 1 条", "报告没有「## 结论」部分"),
        ("意见第 2 条", "缺乏全局规划」没有写进报告"),
        ("意见第 3 条", "本报告未使用资料之外的信息"),
        ("调研主题", TOPIC),
        ("资料要点", "【要点】"),
    ]

    ok = True
    for name, needle in checks:
        found = needle in prompt
        print(f"    {'OK  ' if found else 'MISS'}  {name}")
        ok = ok and found

    counter_ok = result.get("revision_count") == 2
    print(f"    {'OK  ' if counter_ok else 'MISS'}  撰写次数计数器 = {result.get('revision_count')}（应为 2）")

    print()
    if ok and counter_ok:
        print("    ==> 通过：审校意见已正确传给撰写者，返工是「针对性修改」")
    else:
        print("    ==> 失败：审校意见没有出现在 prompt 里，返工只是重新生成")
    print()
    return ok and counter_ok


# ==================== ② A/B 对照实验 ====================

def rewrite(with_feedback):
    """调用真实的 write_node 重写一次

    with_feedback=True  → 模拟「修复后」：state 里有 review，会被拼进 prompt
    with_feedback=False → 模拟「修复前」：没有 review，prompt 里只有资料
    """
    state = {
        "topic": TOPIC,
        "sub_questions": ["ReAct 架构的原理是什么"],
        "findings": FINDINGS,
        "report": REPORT_V1,
        "revision_count": 1,
    }
    if with_feedback:
        state["review"] = REVIEW
    return main.write_node(state)["report"]


def ab_experiment():
    print("=" * 72)
    print("② A/B 对照：带审校意见 vs 不带审校意见，各重写一次")
    print("=" * 72)

    print("\n  正在重写 A 组（带审校意见，模拟修复后）...")
    a = rewrite(True)
    print("  正在重写 B 组（不带审校意见，模拟修复前）...")
    b = rewrite(False)

    print("\n" + "-" * 72)
    print("A 组（带审校意见）")
    print("-" * 72)
    print(a)
    print("\n" + "-" * 72)
    print("B 组（不带审校意见）")
    print("-" * 72)
    print(b)

    def check(text):
        return {
            "补上了「结论」部分": "结论" in text,
            "补上了「缺乏全局规划」": "全局规划" in text,
            # ↓ 关键区分点：这条只有看了审校意见才会做
            "采纳了审校指定的句子": "本报告未使用资料之外的信息" in text,
        }

    ca, cb = check(a), check(b)

    print("\n" + "=" * 72)
    print("对照结果")
    print("=" * 72)
    print(f"  {'检查项':<24}{'A 组（有反馈）':<18}{'B 组（无反馈）'}")
    print("  " + "-" * 64)
    for k in ca:
        print(f"  {k:<22}{('通过' if ca[k] else '未通过'):<20}{'通过' if cb[k] else '未通过'}")

    a_score = sum(ca.values())
    b_score = sum(cb.values())
    total = len(ca)
    print()
    print(f"  A 组命中 {a_score}/{total}，B 组命中 {b_score}/{total}")
    if a_score > b_score:
        print("  ==> A 组更好：审校意见确实引导了针对性修改")
        print("      （「采纳指定句子」只有看了反馈才会做 —— 这是关键区分点）")
    elif a_score == b_score:
        print("  ==> 两组相同：说明这次构造的差异点不够，或模型自己补全了")
    else:
        print("  ==> B 组反而更好？建议多跑几次观察波动")
    print()
    return a_score, b_score


if __name__ == "__main__":
    ok1 = check_prompt()
    a_score, b_score = ab_experiment()

    print("=" * 72)
    print("【总结】")
    print("=" * 72)
    print(f"  ① prompt 检查      ：{'通过' if ok1 else '未通过'}")
    print(f"  ② A/B 对照         ：A 组 {a_score}/3，B 组 {b_score}/3")
    print("=" * 72)
