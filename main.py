import os
import glob
import sys
import time
from operator import add
from typing import Annotated, TypedDict

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.types import Send

from dotenv import load_dotenv


# 项目根目录（本文件所在目录）——后续所有路径都基于它，不依赖当前工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 记忆库路径：必须放在【模块级】，不能放进 __main__
#   - write_node 是模块级函数，也要用它
#   - main.py 被 import 时（如 mcp/mcp_server.py 的 `import main as research`）
#     __main__ 块根本不会执行 → 定义在里面的常量会 NameError
MEM_DB = os.path.join(BASE_DIR, "memory", "memory.db")

# 显式指定 .env 位置，避免因工作目录不同而找不到
load_dotenv(os.path.join(BASE_DIR, ".env"))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI

# 本项目自己的模块（正规包写法：任何 IDE 都能静态解析，不需要额外配置）
from memory.memory_store import MemoryStore


class State(TypedDict):
    """Agent 之间共享的「黑板」"""
    topic: str           # 调研主题
    sub_questions: list  # ① 规划者写
    findings: Annotated[list, add]       # ② 调研者写
    report: str          # ③ 撰写者写
    review: str          # 审校意见
    revision_count: int  # 重写次数 避免死循环
    memory_context: str  # ④ 历史调研记忆（检索来的，供规划者参考）


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
    
    【历史调研记忆】
    {state.get("memory_context") or "（无 —— 这是第一次调研这个方向）"}
    如果上面有历史记录，请【避免重复拆出完全相同的子问题】，
    尽量换个角度补充，或者顺着上次的结论深入。

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

    # 读取用户偏好
    mem = MemoryStore(MEM_DB)
    prefs = mem.list_preferences()
    mem.close()

    pref_block = ""
    if prefs:
        pref_block = "\n【用户偏好 —— 请在不违背第 3 条的前提下尽量满足】\n" + \
                     "\n".join(f"- {p}" for p in prefs) + "\n"


    prompt = f"""你是一名专业研究员。请根据下面的资料，撰写一份关于「{state['topic']}」的调研报告。

    要求：
    1. 使用 Markdown 格式
    2. 结构包含三部分：## 概述、## 分点论述、## 结论
    3. **只使用提供的资料**，不要编造资料中没有的信息
    4. 如果某个子问题的资料不足，在对应位置明确写出「资料未涵盖此部分」
    {pref_block}
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

# checkpointer：把每一步的状态存进 SQLite，支持中断后接着跑
_ckpt_conn = sqlite3.connect(
    os.path.join(BASE_DIR, "checkpoints.db"),
    check_same_thread=False,          # LangGraph 可能在线程里用这个连接
)
# ---- 演示开关 ----
# 设成节点名（如 "write" / "plan" / "research_one"）→ 在该节点【之前】必然暂停
# 用来演示「中断 → 重启 → 接着跑」；正式使用请设为 None
PAUSE_BEFORE = "write"
graph = builder.compile(
    checkpointer=SqliteSaver(_ckpt_conn),
    interrupt_before=[PAUSE_BEFORE] if PAUSE_BEFORE else None,
)
# ==================== 运行 ====================
def clean_input(s: str) -> str:
    """规范化终端输入：去掉 BOM 等不可见字符

    用管道喂输入时（@("id") | python main.py）编码不由程序控制，
    第一行可能带上 \ufeff，导致 thread_id 查不到断点、还会被存进数据库。
    输入是「外部数据」，在边界处一次清干净，后面所有代码都不用操心。
    """
    return s.replace("\ufeff", "").strip()


def pick_thread_id(graph, base: str):
    """挑一个可以安全开跑的 thread_id，返回 (thread_id, config, snap)

    为什么需要它 —— 已经跑完的会话【不能原地重跑】：
        State.findings 用的是 add 累加器（见上面 class State 的注释）。
        fan_out 并行派发多个 research_one，每个都写 findings，
        必须有累加器才能合并成一份；改成覆盖语义会丢掉大部分调研结果。
        代价是旧 findings 清不掉 —— 原地重跑时新的会 add 到旧的后面。
        而 topic 是无 reducer 字段、会被覆盖成新主题
        → write_node 把两轮资料拼进同一篇报告，张冠李戴，且不报任何错。

    三种情况：
        snap.next 非空  → 有断点，交给续跑分支，不动 ID
        snap.values 为空 → 全新会话，直接用
        否则            → 已跑完，自动改成 base#2、base#3……

    能靠结构的别靠纪律：与其记得「重跑要换 ID」，不如让代码自动换。
    """
    # 全新会话 / 有断点要续跑 → 直接用原来的 ID
    config = {"configurable": {"thread_id": base}}
    snap = graph.get_state(config)
    if snap.next or not snap.values:
        return base, config, snap

    # 已跑完 → 换一个带序号的 ID（base#2、base#3……直到找到干净的）
    n = 2
    while True:
        thread_id = f"{base}#{n}"
        config = {"configurable": {"thread_id": thread_id}}
        snap = graph.get_state(config)
        if snap.next or not snap.values:
            return thread_id, config, snap
        n += 1


if __name__ == "__main__":
    # ---- 对照实验开关 ----
    # 加 --no-memory：不读历史记忆（A/B 对照用，隔离"记忆"这个变量）
    USE_MEMORY = "--no-memory" not in sys.argv

    # ---- 偏好管理命令：非交互，跑完立即退出 ----
    if "--prefs" in sys.argv:
        mem = MemoryStore(MEM_DB)
        prefs = mem.list_preferences()
        if prefs:
            print(f"📌 已记录的偏好（{len(prefs)} 条）：")
            for p in prefs:
                print(f"  · {p}")
        else:
            print("📌 还没有记录任何偏好")
        mem.close()
        raise SystemExit(0)

    if "--pref" in sys.argv:
        idx = sys.argv.index("--pref")
        text = " ".join(sys.argv[idx + 1:])
        if not text:
            print('❌ 用法：python main.py --pref "偏好内容"')
            raise SystemExit(1)
        mem = MemoryStore(MEM_DB)
        ok = mem.add_preference(text)
        print(f"✅ 已记住偏好：{text}" if ok else f"ℹ️ 这条偏好之前已经记过了：{text}")
        mem.close()
        raise SystemExit(0)

    typed_id = clean_input(input("会话 ID（回车用 default）：")) or "default"
    # 已跑完的会话会自动换成一个带序号的 ID，避免 findings 累加污染新报告
    thread_id, config, snap = pick_thread_id(graph, typed_id)

    if thread_id != typed_id:
        print(f"\n♻️  会话「{typed_id}」已经跑完了，本次自动改用「{thread_id}」")
        print("    （findings 带 add 累加器、清不掉；原地重跑会把上一轮资料混进新报告）")

    def show_stages(vals, skipping, next_nodes=()):
        """汇报四个阶段的状态：已完成 / 本次执行 / 尚未轮到

        next_nodes: snap.next 里的节点名，这些是【马上就要跑的】
        """
        print("   阶段状态：")

        def line(name, node, detail):
            done = detail[1]
            desc = detail[0]
            if done and skipping:
                icon, tail = "⏭️ ", "本次跳过"
            elif done:
                icon, tail = "✅", "已完成"
            elif node in next_nodes:
                icon, tail = "▶️ ", "本次执行"          # ← 马上要跑的
            else:
                icon, tail = "⏸️ ", "尚未轮到"
            print(f"     {icon} {name} —— {desc}（{tail}）")

        qs = vals.get("sub_questions") or []
        fs = vals.get("findings") or []
        rp = vals.get("report") or ""
        rv = vals.get("review") or ""

        line("① 规划者", "plan", ("拆解子问题" if not qs else f"{len(qs)} 个子问题", bool(qs)))
        line("② 调研者", "research_one", ("并行调研" if not fs else f"{len(fs)} 条要点", bool(fs)))
        line("③ 撰写者", "write", ("生成报告" if not rp else f"{len(rp)} 字报告", bool(rp)))
        line("④ 审校者", "review", ("审校报告" if "通过" not in rv else "已通过", "通过" in rv))

    if snap.next:
        # get_state().next 非空 = 上次没跑完 → 从断点继续
        print(f"\n🔁 检测到未完成的会话（下一步：{snap.next}）")
        print("   从断点继续，已跑过的节点不会重跑\n")
        show_stages(snap.values, skipping=True, next_nodes=snap.next)
        print()
        result = graph.invoke(None, config)          # ← 传 None = 续跑
    else:
        print("\n🆕 新会话，从头开始")
        topic = clean_input(input("请输入调研主题："))

        # ---- 检索历史记忆：这个主题以前调研过吗？----
        # --no-memory 时不检索（对照组），memory_context 保持空字符串
        past = []
        if USE_MEMORY:
            memory = MemoryStore(
                MEM_DB,
                embed=lambda texts: model.encode(texts, normalize_embeddings=True),
            )
            past = memory.search_sessions(topic, top_k=2, min_score=0.40)
            memory.close()

        memory_context = ""
        if past:
            print(f"\n📚 这个主题以前调研过：")
            lines = []
            for h in past:
                print(f"   · [{h['score']}] {h['topic']}   ({h['created_at']})")
                lines.append(f"- 主题：{h['topic']}（{h['created_at']}）")
                if h["sub_questions"]:
                    lines.append(f"  当时拆的子问题：{'；'.join(h['sub_questions'])}")
            memory_context = "\n".join(lines)
        elif USE_MEMORY:
            print("\n📚 没有相关的历史调研")
        else:
            print("\n📚 【--no-memory】跳过记忆检索（对照组）")

        print()
        result = graph.invoke(
            {"topic": topic, "memory_context": memory_context}, config
        )

    # ---- 收尾 ----
    # ⚠️ 关键：interrupt_before 会让 invoke【提前返回】，那时的 State 是不完整的
    #    （停在 write 之前 → 没有 report / review 字段）
    #    所以要先判断"跑完了没"，再决定要不要输出报告。
    #    判断方式：get_state().next 为空 = 跑完了；非空 = 还停在断点。
    final = graph.get_state(config)
    if final.next:
        print("\n" + "=" * 60)
        print(f"⏸️  已暂停在断点（下一步：{final.next}）")
        print("=" * 60)
        print("   状态已存入 checkpoints.db。")
        print(f"   再次运行并用同一个会话 ID（{thread_id}）即可从断点继续。")
        raise SystemExit(0)

    print("\n" + "=" * 60)
    print("📄 调研报告")
    print("=" * 60)
    print(result["report"])

    with open(os.path.join(BASE_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {result['topic']}\n\n{result['report']}")
    print("\n✅ 报告已保存到 report.md")

    # ---- 写记忆：把这次调研存成一条会话摘要 ----
    # ⚠️ 只记【事实】——调研了什么、拆了哪些子问题、报告怎么开的。
    #    不记"结论对不对"（结论可能错、会过时，记下来会误导以后的会话）。
    # ⚠️ 必须放在最末尾：上面的 `if final.next: raise SystemExit(0)` 保证了
    #    「只有真正跑完才会走到这里」——断点时的状态是不完整的，不能写记忆。
    memory = MemoryStore(
        MEM_DB,
        embed=lambda texts: model.encode(texts, normalize_embeddings=True),
    )
    memory.save_session(
        thread_id=thread_id,
        topic=result["topic"],
        sub_questions=result["sub_questions"],
        report_head=result["report"],
    )
    print(f"\n🧠 已记住这次调研（记忆库现有 {memory.stats()['sessions']} 条会话）")
    memory.close()

