"""验证：自适应取数是否生效

直接 import step9 里的 search 函数，用固定问题测试。
（绕开「规划者随机生成子问题」的干扰，结果可复现）
"""
import step9_rerank as m          # 导入你的主程序

# 三个典型场景（覆盖不同的资料需求量）
test_cases = [
    ("复合问题", "多 Agent 协作架构如何结合 Agent 的记忆机制与 RAG 检索增强生成来完成复杂任务"),
    ("单一问题", "RAG 的典型流程是什么"),
    ("两概念协同", "Agent 的记忆机制与 RAG 检索增强生成如何协同支持 Agent 的长期与短期信息处理"),
]

for label, q in test_cases:
    docs = m.search(q)
    print("=" * 66)
    print(f"【{label}】{q[:40]}...")
    print(f"→ 检索到 {len(docs)} 段资料")
    for i, d in enumerate(docs, 1):
        title = d["text"].splitlines()[0][:32]
        print(f"     {i}. {title}")
    print()