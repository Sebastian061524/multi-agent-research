import os
import glob
import time
from operator import add
from typing import Annotated

from langgraph.types import Send

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

class State(TypedDict):
    """Agent 之间共享的「黑板」"""
    topic: str           # 调研主题
    sub_questions: list  # ① 规划者写
    findings: Annotated[list, add]       # ② 调研者写
    report: str          # ③ 撰写者写
    review: str          # 审校意见
    revision_count: int  # 重写次数 避免死循环


llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

DOCS_DIR = "research_docs"
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

# 加载文档
chunks = []
for pattern in [f"{DOCS_DIR}/**/*.txt", f"{DOCS_DIR}/**/*.md"]:
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > 20:
                chunks.append({"source": os.path.basename(path), "text": para})

embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
print(f"已索引 {len(chunks)} 个段落")


def search(query, top_k=3):
    """检索函数：给问题，返回最相关的段落"""
    qv = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ qv
    idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in idx]

def plan_node(state: State):
    """① 规划者 Agent：把主题拆成子问题"""
    prompt = f"""你是一名资深研究规划专家。
用户的调研主题是：{state['topic']}

请把这个主题拆解成 3 个具体、可独立调研的子问题。

要求：
- 只输出 3 行，每行一个子问题
- 不要编号，不要解释，不要任何其他内容
"""
    resp = llm.invoke(prompt)
    questions = []
    for line in resp.content.strip().split("\n"):
        q = line.strip().lstrip("0123456789.、）- ").strip()
        if q:
            questions.append(q)
    return {"sub_questions": questions}

def research_one(state: dict):
    """调研单个子问题 会并行执行3次

    参数不是完整的State，而是send传进来的小dict
    """
    q = state["question"]

    t0 = time.time()
    print(f"   ▶ 开始调研：{q[:20]}...")  # ← 加这行

    # 检索资料
    docs = search(q, top_k=3)
    materials = "\n\n".join(f"[{d['source']}] {d['text']}" for d in docs)

    # 让模型基于资料提炼要点（这一步耗时，正是并行的价值所在）
    prompt = f"""你是一名资料分析员。请根据下面的资料回答问题。

    要求：
    - 只使用资料中的信息，不要编造
    - 如果资料不足以回答，明确说明「资料未涵盖」
    - 控制在 200 字以内

    【问题】{q}

    【资料】
    {materials}
    """

    resp = llm.invoke(prompt)
    print(f"   ✓ 完成：{q[:20]}... 耗时 {time.time()-t0:.1f}s")   # ← 改这行

    # 注意：返回的是「列表」，因为 reducer 用 add 拼接
    return {"findings": [{
        "question": q,
        "materials": materials,
        "summary": resp.content,
    }]}

def fan_out(state: State):
    """分叉：为每个子问题生成一个并行任务"""
    return [Send("research_one", {"question": q}) for q in state["sub_questions"]]

def write_node(state: State):
    """③ 撰写者 Agent：整合资料写成报告"""
    # 把所有子问题的资料拼成一大段
    materials = "\n\n".join(
        f"### 子问题：{f['question']}\n【要点】\n{f.get('summary', f['materials'])}"
        for f in state['findings']
    )

    # 如果是重写 把上一版的审校意见带上
    feedback = ""
    if state.get("review"):
        feedback = f"""

        【上一版的审校意见 —— 请逐条针对性改进】
        {state['review']}
        """

    prompt = f"""你是一名专业研究员。请根据下面的资料，撰写一份关于「{state['topic']}」的调研报告。

要求：
1. 使用 Markdown 格式
2. 结构包含三部分：## 概述、## 分点论述、## 结论
3. **只使用提供的资料**，不要编造资料中没有的信息
4. 如果某个子问题的资料不足，在对应位置明确写出「资料未涵盖此部分」

【资料】
{materials}
{feedback}
"""

    resp = llm.invoke(prompt)
    count = state.get("revision_count", 0) + 1
    print(f"\n【③ 撰写者】生成报告（第 {count} 版）")
    return {"report": resp.content, "revision_count": count}


MAX_REVISIONS = 2
def review_node(state: State):
    """④ 审校者 Agent：审查报告质量"""
    materials = "\n\n".join(
        f"### {f['question']}\n{f.get('summary', f['materials'])}"
        for f in state['findings']
    )

    prompt = f"""你是一名严格的调研报告审校员。

    【待审报告】
    {state['report']}

    【可用资料（报告只能以此为据）】
    {materials}

    请审查以下三点：
    1. **完整性**：是否覆盖了所有子问题？
    2. **忠实性**：是否有资料中不存在的内容（编造）？
    3. **结构**：是否有清晰的概述、论述、结论？

    输出格式（严格遵守）：
    - 如果全部合格：**第一行只写「通过」**，后面可简要说明理由
    - 如果有问题：**第一行只写「不通过」**，后面逐条列出问题和具体修改建议
    """
    resp = llm.invoke(prompt)

    first_line = resp.content.strip().split("\n")[0]
    print(f"\n【④ 审校者】{first_line}")
    return {"review": resp.content}

def route_after_review(state: State):
    """审校后的路由：通过就结束，不通过就返工（有次数上限）"""
    first_line = state["review"].strip().split("\n")[0]

    # ① 先检查次数上限（防死循环）
    if state["revision_count"] >= MAX_REVISIONS:
        print("   → 已达最大修改次数，强制结束")
        return "end"

    # ② 再看审校结论
    if "不通过" in first_line:
        print("   → 审校未通过，返工重写 🔄")
        return "rewrite"

    print("   → 审校通过 ✅")
    return "end"

# ==================== 组装图 ====================
builder = StateGraph(State)
builder.add_node("plan", plan_node)
builder.add_node("research_one", research_one) # ← 改名
builder.add_node("write", write_node)
builder.add_node("review", review_node)          # ← 新增节点

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out) # ← 用 fan_out 分叉
builder.add_edge("research_one", "write") # ← 并行分支跑完都汇聚到 write
builder.add_edge("write", "review")              # ← 写完去审校

# ← 关键：条件回边（形成循环）
builder.add_conditional_edges(
    "review",
    route_after_review,
    {"rewrite": "write", "end": END}
)

graph = builder.compile()
# ==================== 运行 ====================
if __name__ == "__main__":
    topic = input("请输入调研主题：")

    result = graph.invoke({"topic": topic})   # ← 传入初始 State

    print("\n" + "=" * 60)
    print("📄 调研报告")
    print("=" * 60)
    print(result["report"])                    # ← 从最终 State 里取报告

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{result['report']}")
    print("\n✅ 报告已保存到 report.md")