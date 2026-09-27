import os
import glob

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
    findings: list       # ② 调研者写
    report: str          # ③ 撰写者写


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

def research_node(state: State):
    """调研者agent 对每个子问题检索资料"""
    findings = []
    for q in state["sub_questions"]:
        docs = search(q, top_k=3)
        # 把命中的段落拼成一段文本 并标注来源
        materials = "\n\n".join(f"[{d['source']}] {d['text']}" for d in docs)
        findings.append({"question": q, "materials": materials})
    return {"findings": findings}

def write_node(state: State):
    """③ 撰写者 Agent：整合资料写成报告"""
    # 把所有子问题的资料拼成一大段
    materials = "\n\n".join(
        f"### 子问题：{f['question']}\n{f['materials']}"
        for f in state['findings']
    )

    prompt = f"""你是一名专业研究员。请根据下面的资料，撰写一份关于「{state['topic']}」的调研报告。

要求：
1. 使用 Markdown 格式
2. 结构包含三部分：## 概述、## 分点论述、## 结论
3. **只使用提供的资料**，不要编造资料中没有的信息
4. 如果某个子问题的资料不足，在对应位置明确写出「资料未涵盖此部分」

【资料】
{materials}
"""

    print("\n【③ 撰写者】正在生成报告...")
    resp = llm.invoke(prompt)
    return {"report": resp.content}

# ==================== 组装图 ====================
builder = StateGraph(State)
builder.add_node("plan", plan_node)          # 节点名 → 函数
builder.add_node("research", research_node)
builder.add_node("write", write_node)

builder.add_edge(START, "plan")              # 流程：plan → research → write
builder.add_edge("plan", "research")
builder.add_edge("research", "write")
builder.add_edge("write", END)

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