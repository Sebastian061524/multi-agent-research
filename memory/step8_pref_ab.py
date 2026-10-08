"""步骤 8 对照实验：用户偏好真的影响 write_node 的输出吗？

设计（只有「偏好」一个变量在动）：
  - 输入完全相同：findings 从 prefA 的 checkpoint 里取，两组共用同一份
  - 代码路径完全相同：直接调用真的 research.write_node()
  - 唯一差异：preferences 表里有 / 没有那条偏好
  - 为排除 LLM 随机性，每种条件各跑 2 次

测量指标：报告里的 Markdown 表格行数（偏好要求「用表格对比」）
"""
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import main as research
from memory.memory_store import MemoryStore

DB = str(_PROJECT_ROOT / "memory" / "memory.db")
CKPT_DB = _PROJECT_ROOT / "checkpoints.db"
PREF = "请用 Markdown 表格对比不同方案，语言尽量通俗"
THREAD = "prefA"
ROUNDS = 2


def resolve_thread(wanted: str) -> str:
    """找到真实的 thread_id

    管道喂输入时第一行可能带上 \\ufeff，存进库里的 ID 就是脏的。
    这里按「去掉 BOM 后相等」来找，避免脚本因为一个不可见字符查不到断点。
    """
    ids = [
        row[0]
        for row in sqlite3.connect(CKPT_DB).execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        )
    ]
    if wanted in ids:
        return wanted
    matches = [t for t in ids if isinstance(t, str) and t.replace("\ufeff", "") == wanted]
    if matches:
        print(f"⚠️  库里的 ID 实际是 {matches[0]!r}（含 BOM），已自动纠正")
        return matches[0]
    sys.exit(f"❌ 找不到 thread={wanted!r}；现有：{[repr(t) for t in ids]}")



# ---------- 测量：报告里有多少行是 Markdown 表格 ----------
def table_rows(report: str) -> int:
    return sum(
        1
        for line in report.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    )


def run_write(label: str) -> str:
    """跑一次 write_node，打印测量结果"""
    result = research.write_node(BASE_STATE)
    report = result["report"]
    print(f"\n>>> {label}：报告 {len(report)} 字，表格 {table_rows(report)} 行")
    return report


# ---------- 取真实输入：两组共用同一份 findings ----------
snap = research.graph.get_state({"configurable": {"thread_id": resolve_thread(THREAD)}})
state = snap.values
if not state.get("findings"):
    sys.exit(f"❌ thread={THREAD} 里没有 findings，先跑一次调研")

BASE_STATE = {
    "topic": state["topic"],
    "findings": state["findings"],
    "review": "",
}
print(f"✅ 输入来自 checkpoint：{BASE_STATE['topic']}")
print(f"   子问题 {len(state['sub_questions'])} 个，findings {len(BASE_STATE['findings'])} 条")

# ---------- 备份现有偏好，实验结束恢复 ----------
mem = MemoryStore(DB)
backup = mem.list_preferences()
for p in backup:
    mem.delete_preference(p)
mem.close()
print(f"📌 实验开始前清空偏好（原有 {len(backup)} 条，结束后恢复）")

results = {}
try:
    # ---- A 组：无偏好 ----
    results["A"] = [run_write(f"A组·无偏好·第{i}次") for i in range(1, ROUNDS + 1)]

    # ---- B 组：有偏好（唯一变化的变量）----
    mem = MemoryStore(DB)
    mem.add_preference(PREF)
    mem.close()
    results["B"] = [run_write(f"B组·有偏好·第{i}次") for i in range(1, ROUNDS + 1)]
finally:
    # ---- 恢复偏好 ----
    mem = MemoryStore(DB)
    for p in mem.list_preferences():
        mem.delete_preference(p)
    for p in backup:
        mem.add_preference(p)
    mem.close()
    print(f"\n♻️  已恢复原有偏好（{len(backup)} 条）")

# ---------- 汇总 ----------
print("\n" + "=" * 56)
print(f"偏好内容：{PREF}")
print("=" * 56)
print(f"{'组':<8}{'表格行数':<12}{'报告字数'}")
for group, reports in results.items():
    tag = "无偏好" if group == "A" else "有偏好"
    for i, r in enumerate(reports, 1):
        print(f"{group}{i} {tag:<8}{table_rows(r):<12}{len(r)}")

a_max = max(table_rows(r) for r in results["A"])
b_min = min(table_rows(r) for r in results["B"])
print("-" * 56)
if b_min > a_max:
    print(f"✅ 差异可以归因于偏好：B 组最少 {b_min} 行 > A 组最多 {a_max} 行")
elif b_min == a_max == 0:
    print("❌ 两组都没出表格：偏好没生效（或模型认为这个题不需要表格）")
else:
    print(f"⚠️  两组重叠（A 最多 {a_max}，B 最少 {b_min}）：证据不足，需要更多轮次")
