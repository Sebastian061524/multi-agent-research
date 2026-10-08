"""验证：断点之后 invoke({}, cfg) 和 invoke(None, cfg) 到底谁是续跑

背景
----
LangGraph 把「数据」和「待办」分开存在 checkpoint 里：
    values  → 状态字段现在的值
    next    → 还有哪些节点没执行

invoke() 的第一个参数是【新输入】，不是「要不要继续」。
所以在有断点（next 非空）时，两种"空"会分道扬镳：
    invoke(None, cfg)  → 「我没有新输入」→ 读 next → 接着跑
    invoke({}, cfg)    → 「这是新输入，只是内容为空」→ 从 START 重新开始

本脚本把这件事测成事实，而不是靠记忆。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


class S(TypedDict):
    log: list


def a(state: S):
    """节点 A：会往 log 里追加 'A'（追加 = 每次执行都能被观测到）"""
    return {"log": state.get("log", []) + ["A"]}


def b(state: S):
    """节点 B：会往 log 里追加 'B'"""
    return {"log": state.get("log", []) + ["B"]}


def build():
    """START → A → B → END，并设置 interrupt_before=['b']"""
    g = StateGraph(S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["b"])


def show(tag: str, graph, cfg):
    snap = graph.get_state(cfg)
    print(f"    {tag:<26} values={snap.values.get('log', [])}  next={snap.next}")


print("=" * 68)
print("实验 1：有断点时，第二次调用 invoke({}, cfg)")
print("=" * 68)
g1 = build()
cfg1 = {"configurable": {"thread_id": "exp1"}}

g1.invoke({}, cfg1)                 # 第 1 次：跑到 B 之前停住
show("第 1 次 invoke({}) 之后", g1, cfg1)

r1 = g1.invoke({}, cfg1)            # 第 2 次：传「空的新输入」
show("第 2 次 invoke({}) 之后", g1, cfg1)
print(f"    → 完全跑完了吗：{g1.get_state(cfg1).next == ()}")
print(f"    → log 里 A 出现了 {r1['log'].count('A')} 次\n")

print("=" * 68)
print("实验 2：有断点时，第二次调用 invoke(None, cfg)")
print("=" * 68)
g2 = build()
cfg2 = {"configurable": {"thread_id": "exp2"}}

g2.invoke({}, cfg2)
show("第 1 次 invoke({}) 之后", g2, cfg2)

r2 = g2.invoke(None, cfg2)          # 第 2 次：传 None = 没有新输入
show("第 2 次 invoke(None) 之后", g2, cfg2)
print(f"    → 完全跑完了吗：{g2.get_state(cfg2).next == ()}")
print(f"    → log 里 A 出现了 {r2['log'].count('A')} 次\n")

print("=" * 68)
print("实验 3：已经把图画完了（next 为空），再调 invoke(None, cfg) 会怎样")
print("=" * 68)
g3 = build()
cfg3 = {"configurable": {"thread_id": "exp3"}}

g3.invoke({}, cfg3)                 # 停在 B 前
g3.invoke(None, cfg3)               # 续跑，跑完
show("已跑完", g3, cfg3)

r3 = g3.invoke(None, cfg3)          # 再调一次
show("又调一次 invoke(None) 之后", g3, cfg3)
print(f"    → log = {r3['log']}（有没有变化？）\n")

print("=" * 68)
print("结论")
print("=" * 68)
print(f"实验 1  A 的次数 = {r1['log'].count('A')}  →  {'A 重跑了（从头开始）' if r1['log'].count('A') > 1 else 'A 没重跑（续跑）'}")
print(f"实验 2  A 的次数 = {r2['log'].count('A')}  →  {'A 重跑了（从头开始）' if r2['log'].count('A') > 1 else 'A 没重跑（续跑）'}")
