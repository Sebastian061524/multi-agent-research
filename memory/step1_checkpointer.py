"""最小实验：看清 checkpointer 到底做了什么

用两个"只往列表里追加一项"的假节点，观察状态在四种情况下的表现。
不需要 LLM，秒级出结果。
"""
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

# ---- 一个极简 State：只有一个用 add 累积的列表 ----
class S(TypedDict):
    steps: Annotated[list[str], add]

def node_a(state: S):
    return {"steps": ["A"]}

def node_b(state: S):
    return {"steps": ["B"]}

def build(checkpointer=None):
    b = StateGraph(S)
    b.add_node("a", node_a)
    b.add_node("b", node_b)
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    return b.compile(checkpointer=checkpointer)

def show(label, value):
    print(f"  {label:<38} {value}")

print("=" * 70)
print("① 没有 checkpointer：每次调用都从零开始")
print("=" * 70)
g1 = build()
show("第 1 次 invoke", g1.invoke({})["steps"])
show("第 2 次 invoke", g1.invoke({})["steps"])
show("第 3 次 invoke", g1.invoke({})["steps"])

print()
print("=" * 70)
print("② 有 checkpointer + 同一个 thread_id：状态被记住了")
print("=" * 70)
g2 = build(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
show("第 1 次 invoke(t1)", g2.invoke({}, cfg)["steps"])
show("第 2 次 invoke(t1)", g2.invoke({}, cfg)["steps"])
show("第 3 次 invoke(t1)", g2.invoke({}, cfg)["steps"])

print()
print("=" * 70)
print("③ 换一个 thread_id：完全隔离，互不影响")
print("=" * 70)
cfg2 = {"configurable": {"thread_id": "t2"}}
show("invoke(t2)", g2.invoke({}, cfg2)["steps"])
show("再 invoke(t1)", g2.invoke({}, cfg)["steps"])

print()
print("=" * 70)
print("④ 看看 checkpointer 里到底存了什么")
print("=" * 70)
snap = g2.get_state(cfg)
print(f"  values（当前状态）: {snap.values}")
print(f"  next（下一步是谁）: {snap.next}")
print(f"  config            : {snap.config}")

print()
print("=" * 70)
print("⑤ 真正的【接住没跑完的部分】：interrupt_before + invoke(None)")
print("=" * 70)


def build_with_stop(checkpointer):
    b = StateGraph(S)
    b.add_node("a", node_a)
    b.add_node("b", node_b)
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    return b.compile(checkpointer=checkpointer, interrupt_before=["b"])


g3 = build_with_stop(InMemorySaver())
cfg3 = {"configurable": {"thread_id": "t3"}}

r1 = g3.invoke({}, cfg3)
# 注意：f-string 里的花括号有特殊含义，要输出【字面量】花括号必须双写 {{ }}
print(f"  第1次 invoke({{}})    : steps = {r1['steps']}    next = {g3.get_state(cfg3).next}")

r2 = g3.invoke(None, cfg3)
print(f"  第2次 invoke(None)  : steps = {r2['steps']}    next = {g3.get_state(cfg3).next}")

print()
print("  ⭐ 关键看：第 2 次的 steps 是 ['A','B'] 而不是 ['A','A','B']")
print("     → 说明 A 没有重跑，真的从断点续上了")