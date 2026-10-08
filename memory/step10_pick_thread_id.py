"""验证 main.pick_thread_id()：已跑完的会话不能原地重跑

背景
----
State.findings 用的是 add 累加器（fan_out 并行派发 research_one 必须靠它合并），
所以旧 findings 清不掉。原地重跑会让新报告混进上一轮的资料，而且不报错。
修法是在程序边界自动换一个带序号的 thread_id。

本脚本用合成图 + MemorySaver 把四种情况测成事实，不动真实数据库。

运行：python memory\\step10_pick_thread_id.py
"""

import sys
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# main.py 在项目根目录；直接跑本脚本时 sys.path[0] 是 memory/，要手动补根目录
sys.path.insert(0, r"D:\PythonCode\research_agent")

from main import pick_thread_id        # noqa: E402  （import main 会加载模型，约 28 秒）


class S(TypedDict):
    topic: str
    findings: Annotated[list, add]      # ← 和 main.py:43 一样的写法


def plan(state: S):
    return {}


def research(state: S):
    return {"findings": [{"question": "新要点"}]}


def build(interrupt: bool = False):
    g = StateGraph(S)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", END)
    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["research"] if interrupt else None,
    )


ok = []


def check(name: str, got, want):
    passed = got == want
    ok.append(passed)
    print(f"  {'✅' if passed else '❌'} {name}")
    print(f"       得到 {got!r}   期望 {want!r}")


print("=" * 70)
print("图 A：不带中断（用来造「已跑完」的会话）")
print("=" * 70)
graph = build()
graph.invoke({"topic": "话题一"}, {"configurable": {"thread_id": "t-fresh"}})
print(f"  t-fresh 跑完，next = {graph.get_state({'configurable': {'thread_id': 't-fresh'}}).next}")

# ---- 情况 1：全新会话 → 原样返回 ----
tid, cfg, snap = pick_thread_id(graph, "t-brand-new")
check("全新会话 → 不换 ID", tid, "t-brand-new")

# ---- 情况 2：已跑完 → 自动加 #2 ----
tid, cfg, snap = pick_thread_id(graph, "t-fresh")
check("已跑完 → 换成 #2", tid, "t-fresh#2")

# ---- 情况 3：#2 也跑完了 → 换 #3 ----
graph.invoke({"topic": "话题一"}, {"configurable": {"thread_id": "t-fresh#2"}})
tid, cfg, snap = pick_thread_id(graph, "t-fresh")
check("已跑完且 #2 被占用 → 换成 #3", tid, "t-fresh#3")

# ---- 情况 4：换到的新 ID 必须是干净的 ----
graph.invoke({"topic": "话题二"}, cfg)          # 用 pick_thread_id 返回的 config 真跑一次
snap = graph.get_state(cfg)
check("新 ID 跑完后 findings 只有 1 条", len(snap.values["findings"]), 1)
check("新 ID 的 topic 是新主题", snap.values["topic"], "话题二")

print()
print("=" * 70)
print("图 B：带中断（用来造「有断点」的会话）")
print("=" * 70)
graph_b = build(interrupt=True)
graph_b.invoke({"topic": "话题一"}, {"configurable": {"thread_id": "t-pending"}})
snap_b = graph_b.get_state({"configurable": {"thread_id": "t-pending"}})
print(f"  t-pending 停在 {snap_b.next}")

# ---- 情况 5：有断点 → 不能换 ID（这是续跑分支的事） ----
tid, cfg, snap = pick_thread_id(graph_b, "t-pending")
check("有断点 → 保持原 ID（交给续跑分支）", tid, "t-pending")
check("有断点 → 返回的 snap.next 非空", bool(snap.next), True)

print()
print("=" * 70)
print(f"结果：{sum(ok)}/{len(ok)} 通过")
print("=" * 70)
sys.exit(0 if all(ok) else 1)
