import os
import glob
import time
from operator import add
from typing import Annotated, TypedDict


from langgraph.types import Send

from dotenv import load_dotenv


# 项目根目录（本文件所在目录）——后续所有路径都基于它，不依赖当前工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 显式指定 .env 位置，避免因工作目录不同而找不到
load_dotenv(os.path.join(BASE_DIR, ".env"))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, CrossEncoder
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

DOCS_DIR = os.path.join(BASE_DIR, "research_docs")

print("加载嵌入模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

print("加载重排序模型...")
reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)

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

if not chunks:
    raise SystemExit(f"❌ 在 {DOCS_DIR} 里没有找到可用资料（需要至少一段超过 20 字的文本）")

embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
print(f"已索引 {len(chunks)} 个段落")

RERANK_THRESHOLD = 0.3   # 低于这个分数视为「不相关」，直接丢弃

# ==================== 查询分解 ====================

def decompose(query, max_subs=4):
    """把一个可能含多个概念的检索问题，拆成若干子查询

    只涉及一个概念 → 原样返回
    涉及多个概念   → 拆成多个聚焦单一概念的子查询
    """
    prompt = f"""你是一个检索查询分析器。

请判断下面的问题涉及几个独立概念：
- 只涉及一个概念 → 原样返回这一行
- 涉及多个概念 → 拆成多个简短子查询（每个聚焦一个概念）

要求：
- 每行一个，最多 {max_subs} 行
- 不要编号，不要解释，不要任何其他内容
- 子查询要简短，保留关键术语（专有名词要原样保留）

问题：{query}
"""
    resp = llm.invoke(prompt)

    subs = []
    for line in resp.content.strip().split("\n"):
        s = line.strip().lstrip("0123456789.、）- ").strip()
        if s:
            subs.append(s)

    # 兜底：模型没给出有效内容时，退回原问题
    return subs[:max_subs] if subs else [query]

def _retrieve(query, max_k=5, min_score=RERANK_THRESHOLD, coarse_k=20):
    """[单查询]两阶段检索：向量粗筛 → 重排序精排 → 自适应取数

    三个参数的分工：
      coarse_k ：粗筛保留的候选数（要足够大，给精排留出重排空间）
      min_score：相关性门槛（低于它视为不相关）
      max_k    ：最多取几段

    ⚠️ 注意 max_k 的语义：不是「必须取 5 段」，而是「最多 5 段」。
       实际取几段由 min_score 决定 —— 相关几段就取几段。
    """
    # ① 阶段1：向量检索粗筛（多取候选，不要在这里就把范围切死）
    qv = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ qv
    idx = scores.argsort()[::-1][:coarse_k]
    candidates = [chunks[i] for i in idx]

    # ② 阶段2：重排序精排（在候选集上重新打分）
    pairs = [[query, c["text"]] for c in candidates]
    rr_scores = reranker.predict(pairs)

    # ③ 按分数排序，并过滤掉低于阈值的
    order = rr_scores.argsort()[::-1]
    results = []
    for i in order:
        if rr_scores[i] < min_score:
            break                      # 分数已经很低了，后面的更差，直接停
        results.append(candidates[i])
        if len(results) >= max_k:
            break
    return results

def search(query, max_k=5, min_score=RERANK_THRESHOLD, coarse_k=20,
           use_decompose=True):
    """【多查询】分解 → 逐个子查询检索 → 合并去重

    目的：解决「复合问题」——一个问题含多个概念时，
         单个查询向量会被稀释，导致某些概念检索不到。
    """
    sub_queries = decompose(query) if use_decompose else [query]

    merged = []
    seen = set()
    for sq in sub_queries:
        for d in _retrieve(sq, max_k=max_k, min_score=min_score, coarse_k=coarse_k):
            key = d["text"][:40]           # 用前 40 字做去重指纹
            if key not in seen:
                seen.add(key)
                merged.append(d)
    return merged

def build_docs_overview(max_items=20):
    """给规划者一份「资料库地图」——告诉它资料库里有哪些内容

    ⚠️ 这里【不做检索、不做阈值过滤】，直接列出所有段落。
       原因：地图必须「完整」。如果地图本身被过滤过，
             规划者就不知道某些资料存在，会漏掉本来能覆盖的方向。

       资料库很大时（几百条以上），再改成「按主题检索 + 只取标题摘要」。
    """
    lines = []
    for c in chunks[:max_items]:
        # 取段落第一行作为「标题」（我们的资料格式是「标题\n正文」）
        title = c["text"].split("\n")[0].strip()
        lines.append(f"- {title[:45]}")
    return "\n".join(lines)

def plan_node(state: State):
    """① 规划者 Agent：把主题拆成子问题（基于资料库实际内容）"""
    topic = state["topic"]

    # ① 先拿到「资料库地图」
    overview = build_docs_overview()

    # ② 带着地图去规划
    prompt = f"""你是一名资深研究规划专家。

    【资料库中实际可用的内容】
    {overview}

    用户的调研主题是：{topic}

    请**只在上述资料能覆盖的范围内**，把这个主题拆解成 3 个具体、可独立调研的子问题。

    要求：
    - 只输出 3 行，每行一个子问题
    - 不要编号，不要解释，不要任何其他内容
    - **不要提出上述资料没有覆盖的方向**（否则会导致调研无结果）
    """
    resp = llm.invoke(prompt)
    questions = []
    for line in resp.content.strip().split("\n"):
        q = line.strip().lstrip("0123456789.、）- ").strip()
        if q:
            questions.append(q)
    questions = questions[:3]

    print(f"\n【① 规划者】拆解出 {len(questions)} 个子问题")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")

    return {"sub_questions": questions}

def research_one(state: dict):
    """调研单个子问题 会并行执行3次

    参数不是完整的State，而是send传进来的小dict
    """
    q = state["question"]

    t0 = time.time()
    print(f"   ▶ 开始调研：{q[:20]}...")

    # 检索资料
    docs = search(q)
    print(f"   ✓ {q[:25]}... → 检索到 {len(docs)} 段资料")
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
    print(f"   ✓ 完成：{q[:20]}... 耗时 {time.time()-t0:.1f}s")

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

    prompt = f"""你是一名调研报告审校员。

    【待审报告】
    {state['report']}

    【可用资料（报告只能以此为据）】
    {materials}

    请从以下三方面审查：

    **A. 结构**
    是否有清晰的概述、分点论述、结论？

    **B. 忠实性（最重要）**
    报告中是否出现了资料里不存在的内容（编造）？
    注意：合理的概括、归纳、跨段落对比不算编造。

    **C. 信息利用**
    资料中【已经提供】的信息，报告是否遗漏了？

    【判定规则 —— 务必严格遵守】
    只要报告满足以下三条，就判「通过」：
      ① 结构完整
      ② 没有编造
      ③ 用尽了资料中已有的信息

    ⚠️ 特别重要：**即使某些子问题因资料不足而无法回答，只要报告已明确
    标注「资料未涵盖」，也应当判「通过」。这不是报告的错，是资料本身的
    限制，重写多少遍也无法改变——判「不通过」只会浪费一次重写。**

    只有在报告出现以下三种情况时，才判「不通过」：
      - 结构混乱（缺概述 / 分点 / 结论）
      - 编造了资料中不存在的内容
      - 遗漏了资料中【已经提供】的信息

    【输出格式 —— 严格遵守】
    - 合格：第一行只写「通过」，后面可简要说明理由。
    - 不合格：第一行只写「不通过」，后面逐条列出**可修改的具体问题**
      （不要列出「资料不足」这类无法修改的问题）。
    """
    resp = llm.invoke(prompt)

    first_line = resp.content.strip().split("\n")[0]
    print(f"\n【④ 审校者】{first_line}")
    return {"review": resp.content}

def route_after_review(state: State):
    """审校后的路由：通过就结束，不通过就返工（有次数上限）"""
    first_line = state["review"].strip().split("\n")[0]

    if "不通过" in first_line:
        # 审校不通过 → 看看还有没有修改机会
        if state["revision_count"] >= MAX_REVISIONS:
            print("   → 已达最大修改次数，强制结束（仍有问题）")
            return "end"
        print("   → 审校未通过，返工重写 🔄")
        return "rewrite"

    if "通过" not in first_line:
        print(f"   ⚠️  审校输出格式异常（首行：{first_line[:20]}），按通过处理")
    print("   → 审校通过 ✅")
    return "end"

# ==================== 组装图 ====================
builder = StateGraph(State)
builder.add_node("plan", plan_node)
builder.add_node("research_one", research_one)
builder.add_node("write", write_node)
builder.add_node("review", review_node)

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out)
builder.add_edge("research_one", "write")
builder.add_edge("write", "review")

builder.add_conditional_edges(
    "review",
    route_after_review,
    {"rewrite": "write", "end": END}
)

graph = builder.compile()
# ==================== 运行 ====================
if __name__ == "__main__":
    topic = input("请输入调研主题：")

    result = graph.invoke({"topic": topic})

    print("\n" + "=" * 60)
    print("📄 调研报告")
    print("=" * 60)
    print(result["report"])

    with open(os.path.join(BASE_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{result['report']}")
    print("\n✅ 报告已保存到 report.md")