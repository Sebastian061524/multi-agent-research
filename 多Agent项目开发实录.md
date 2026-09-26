# 多 Agent 调研报告生成器 · 开发实录

> 记录这个项目从零到可用功能的完整过程：每一步做了什么、遇到什么问题、怎么解决的。
>
> 项目仓库：`multi-agent-research`
> 最终版本：`step9_rerank.py`

---

## 目录

- [一、项目概览](#一项目概览)
- [二、开发过程（9 个步骤）](#二开发过程9-个步骤)
- [三、关键问题专题](#三关键问题专题)
- [四、踩坑总结表](#四踩坑总结表)
- [五、经验教训](#五经验教训)

---

## 一、项目概览

### 目标

输入一个调研主题，自动产出一份**带引用、不编造**的 Markdown 调研报告。

### 最终架构

```
用户输入调研主题
       ↓
┌─────────────────┐
│ ① 规划者 Agent   │  感知资料库，拆成 3 个「能答」的子问题
└────────┬────────┘
         ↓
┌─────────────────┐
│ ② 调研者 Agent×N │  并行执行，两阶段检索，自适应取数
└────────┬────────┘
         ↓
┌─────────────────┐
│ ③ 撰写者 Agent   │  防幻觉：只用资料，缺口如实标注
└────────┬────────┘
         ↓
┌─────────────────┐
│ ④ 审校者 Agent   │  不合格自动返工（有次数上限）
└────────┬────────┘
         ↓
     report.md
```

### 技术栈

| 组件 | 用什么 |
|---|---|
| 编排 | LangGraph（State / 节点 / 条件边 / Send 并行）|
| 大模型 | DeepSeek（deepseek-chat）|
| 嵌入模型 | BAAI/bge-small-zh-v1.5（512 维）|
| 重排序模型 | BAAI/bge-reranker-base（cross-encoder）|
| 检索 | 向量检索 + 重排序（纯 NumPy 实现）|

### 为什么用多 Agent

| 判断标准 | 本项目的检验 |
|---|---|
| **上下文独立** | 3 个子问题各自检索，上下文天然分离 |
| **角色冲突** | 规划要发散、撰写要收敛、审校要挑刺，一个 prompt 兼顾不了 |
| **可并行** | 3 个子问题互不依赖 |

> **反例（本项目不适合多 Agent 的场景）**：智能客服。咨询/下单/投诉共享同一段对话上下文，角色也不冲突——**单 Agent + 好工具才是最优解**。

---

## 二、开发过程（9 个步骤）

### 步骤 1：搭好检索基础设施

**做了什么**

1. 加载嵌入模型 `bge-small-zh-v1.5`
2. 读取 `research_docs/` 下的 txt/md，按空行切成段落
3. 把所有段落向量化，得到 `(7, 512)` 的矩阵
4. 写检索函数：给一个问题，返回最相似的段落

**为什么先做这一步**

检索是后面三个 Agent 的公共基础设施。**先把它单独验证通过**，后面出错就能排除这一层的嫌疑。

**遇到的问题**

**问题 1：矩阵乘法顺序写反**

```python
scores = qv @ embeddings      # ❌ ValueError: matmul: Input operand 1 has a mismatch
scores = embeddings @ qv      # ✅
```

报错信息：

```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is different from 512)
```

**根因**：`embeddings` 形状 `(7, 512)`，`qv` 形状 `(512,)`。

```
embeddings @ qv  →  (7, 512) @ (512,)  →  512 == 512 ✅  →  (7,)
qv @ embeddings  →  (512,) @ (7, 512)  →  512 != 7   ❌
```

**解决**：记住矩阵乘法规则——**相邻维度必须相等，然后被「消掉」**。

```
[A行, 中间] @ [中间, B列] = [A行, B列]
        ↑________↑ 必须相等
```

**验证结果**

```
已索引 7 个段落
向量化完成，形状：(7, 512)
查询"ReAct 架构是什么？" → top-1 命中 ReAct ✅
```

---

### 步骤 2：做出第一个 Agent（规划者）

**做了什么**

1. 调通 DeepSeek API（`ChatOpenAI` + `llm.invoke()`）
2. 写规划者的 prompt：把主题拆成 3 个子问题
3. 解析模型输出（文本 → 列表）

**遇到的问题**

**问题 2：模型输出可能带编号/开场白，解析要防御**

即使 prompt 里说了「不要编号」，模型**仍可能**输出：

```
1. ReAct 架构的核心原理是什么？
2. ...
```

**解决**：防御性解析——不管有没有编号都能正确提取。

```python
q = line.strip().lstrip("0123456789.、）- ").strip()
```

**注意这个做法的局限**：`lstrip` 是按**字符集**删的，如果问题本身以数字开头（如「2个音箱多少钱」），数字会被误删。更稳的写法是用正则：

```python
import re
q = re.sub(r"^\s*\d+[.、)）-]\s*", "", line).strip()
```

**验证结果**

```
【① 规划者】拆解出 3 个子问题
   1. 大模型Agent的感知-规划-行动循环架构有哪些典型实现模式
   2. 大模型Agent的记忆与反思机制在不同架构中如何设计与集成
   3. 大模型Agent的多智能体协作与通信架构有哪些主流范式
```

**核心认知**：**多 Agent 的本质 = 不同 Agent 用不同 prompt**。模型是同一个，角色和目标不同。

---

### 步骤 3：把「规划」和「检索」串起来

**做了什么**

1. 把平铺的代码包成函数（`plan()` / `research()` / `search()`）
2. 加调研循环：对每个子问题分别检索

**为什么要包成函数**

因为**每个函数后面都会变成一个图节点**。

**第一次发现多 Agent 的价值**

```
不拆：把「大模型Agent有哪些主流架构」直接检索
     → 每个子问题拿到的资料都一样

拆开：3 个子问题各自检索
     → 每个子问题拿到各自最相关的资料 ✅
```

**遇到的隐患（当时只是观察）**

```
子问题1（经典架构）      → [ReAct]          ✅
子问题2（工具/记忆）     → [记忆机制]        ✅
子问题3（生产落地对比）  → [ReAct]          ⚠️ 又命中了 ReAct
```

**根因**：**规划者不知道资料库里有什么**。它拆出了「生产落地可靠性对比」这种资料库答不了的问题，检索只能返回最接近的（ReAct）。

> 这个隐患在步骤 7、8 会成为严重问题。

---

### 步骤 4：加第三个 Agent（撰写者）

**做了什么**

1. 写撰写者的 prompt
2. 加入**防幻觉约束**（最关键的一句）

```python
prompt = f"""你是一名专业研究员。请根据下面的资料，撰写一份关于「{state['topic']}」的调研报告。

要求：
1. 使用 Markdown 格式
2. 结构包含三部分：## 概述、## 分点论述、## 结论
3. **只使用提供的资料**，不要编造资料中没有的信息
4. 如果某个子问题的资料不足，在对应位置明确写出「资料未涵盖此部分」

【资料】
{materials}
"""
```

**为什么第 3、4 条是生命线**

```
不加约束 → 模型会用自己的知识「补全」报告
         → 看起来内容丰富，实际无法溯源 ❌

加了约束 → 只能用你给的资料，资料不足就承认
         → 报告每一句都可溯源 ✅
```

**验证结果（防幻觉生效）**

报告里出现了这些话：

> 「关于 **Reflexion、AutoGPT、MetaGPT** 等框架，**资料未涵盖此部分**，无法就其继承与区别进行论述。」

**这些话是正确的**——资料库里确实没有这些框架。模型**没有编造**。

---

### 步骤 5：改造成 LangGraph

**做了什么**

把「函数链」改成「图」：

| 之前（函数链） | 现在（LangGraph） |
|---|---|
| `questions = plan(topic)` | `plan_node(state)` 读 `state['topic']`，返回 `{"sub_questions": ...}` |
| `findings = research(questions)` | `research_node(state)` |
| `report = write(topic, findings)` | `write_node(state)` |

**核心概念转变：从「传参数」到「共享状态」**

```
之前：像接力赛，每个人跑完把棒子交给下一个
现在：像白板，所有 Agent 读写同一块黑板
```

```python
class State(TypedDict):
    topic: str           # 调研主题
    sub_questions: list  # ① 规划者写
    findings: list       # ② 调研者写
    report: str          # ③ 撰写者写
```

**State 的合并规则**

> 节点返回一个 dict，这个 dict **会合并进 State**——**只覆盖你返回的字段，其他字段保持不变**。

```python
# 初始
{"topic": "..."}
# plan_node 返回 {"sub_questions": [...]}
# → {"topic": "...", "sub_questions": [...]}     ← 合并，topic 没丢
```

**遇到的小问题**

- 忘了把 `topic` 改成 `state['topic']` → `NameError: name 'topic' is not defined`

**验证结果**：功能完全一样，但底层已经是图了。

---

### 步骤 6：并行执行（Send API）

**做了什么**

把串行的 3 次检索改成并行：

```
串行：子问题1 → 等 1.1s ┐
     子问题2 → 等 1.2s ├─ 总共 3.7s
     子问题3 → 等 1.4s ┘

并行：三个同时跑，总共 1.4s（最慢那个的耗时）🚀
```

**两个核心概念**

| 概念 | 作用 |
|---|---|
| **`Send`（分叉）** | 把一个节点「炸开」成多个并行任务 |
| **Reducer（合并）** | 定义多个并行结果怎么合回 State |

```python
def fan_out(state: State):
    """分叉：为每个子问题生成一个并行任务"""
    return [Send("research_one", {"question": q}) for q in state["sub_questions"]]
```

**遇到的关键问题**

**问题 3（LangGraph 并行第一课错误）**

```
langgraph.errors.InvalidUpdateError: At key 'findings': Can receive only one value
per step. Use an Annotated key to handle multiple values.
```

**根因**：`findings` 字段没有配 reducer，多个并行分支同时写它 → LangGraph 报错。

**第一次尝试（错的）**：

```python
findings: [list, add]       # ❌ 这不是 reducer
```

`[list, add]` 只是一个**列表字面量**，LangGraph 看不到合并规则。

**正确写法**：

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    findings: Annotated[list, add]     # ✅
```

**读法**：

```python
findings: Annotated[list, add]
             ↑       ↑    ↑
          特殊标记  类型  合并函数
```

> 「`findings` 是个列表；**如果多个并行分支同时写它，用 `add` 把结果拼接起来**。」

**另一个配套要求**：`research_one` 的返回值必须是**列表**。

```python
return {"findings": [{"question": q, ...}]}     # ✅
return {"findings": {"question": q, ...}}       # ❌ dict 不能 add
```

**验证结果（并行的铁证）**

```
   ▶ 开始调研：大模型Agent的单体架构...
   ▶ 开始调研：大模型Agent的多智能体协作架构...
   ▶ 开始调研：大模型Agent的认知与记忆增强架构...
   ✓ 完成：... 耗时 1.0s
   ✓ 完成：... 耗时 1.2s
   ✓ 完成：... 耗时 1.2s
```

**三个「开始」挨在一起** → 真的在并行 ✅

**耗时对比**：串行 3.7s → 并行 1.4s（快 2.6 倍）

**关于 `Annotated` 的重要认知**

`Annotated` 是「给类型打标签」的语法，很多库（LangGraph、Pydantic、FastAPI）都用它传递额外信息。

对 LangGraph 来说，这个标签通常是 **reducer（合并函数）**。

---

### 步骤 7：加审校 Agent（循环）

**做了什么**

在撰写之后加一个审校节点，不合格就返工：

```
plan → research → write → review ──通过──→ END
                    ↑         │
                    └──不通过──┘
```

**新增的能力：回边（循环）**

```python
builder.add_edge("write", "review")              # 写完去审校
builder.add_conditional_edges(
    "review",
    route_after_review,
    {"rewrite": "write", "end": END},            # ← 条件回边，形成循环
)
```

**必须防死循环**

```python
MAX_REVISIONS = 2      # 最多改 2 版

def route_after_review(state: State):
    first_line = state["review"].strip().split("\n")[0]

    # ① 先检查次数上限（防死循环的第一道闸门）
    if state["revision_count"] >= MAX_REVISIONS:
        return "end"

    # ② 再看审校结论
    if "不通过" in first_line:
        return "rewrite"
    return "end"
```

**遇到的逻辑问题**

**问题 4：路由判断顺序导致提示信息误导**

```
【④ 审校者】通过                    ← 审校说通过了
   → 已达最大修改次数，强制结束      ← 但提示像是「出问题被强制中止」
```

**根因**：先检查次数上限，导致「通过」时也走了「强制结束」的分支。

**改进后**（先看结论，再查次数）：

```python
def route_after_review(state: State):
    first_line = state["review"].strip().split("\n")[0]

    if "不通过" in first_line:
        if state["revision_count"] >= MAX_REVISIONS:
            print("   → 已达最大修改次数，强制结束（仍有问题）")
            return "end"
        print("   → 审校未通过，返工重写 🔄")
        return "rewrite"

    print("   → 审校通过 ✅")
    return "end"
```

**发现的更深问题**

**问题 5：审校员分不清「能修」和「修不了」**

某次运行中，审校**连续两次判不通过**，但报告看起来没问题。

**根因**：

```
报告的问题分两种：
① 报告写得不好（可修复）    → 该判「不通过」
② 资料本身不够（不可修复）  → 该判「通过」（报告已诚实标注）
```

审校员把两者混为一谈，导致**注定改不好的返工**。

**解决方向**：在审校 prompt 里明确判定规则：

```
【判定规则 —— 务必严格遵守】
- 只要报告结构完整、没有编造、且用尽了资料中已有的信息 → 判「通过」
- 即使某些子问题因资料不足而无法回答，只要报告已明确标注「资料未涵盖」，
  也应当判「通过」。这不是报告的错，是资料的限制。
- 只有在报告结构混乱、编造内容、或遗漏了资料中已有的信息时，才判「不通过」
```

**核心教训**：

> **在多 Agent 系统里，下游怎么努力都补不上上游的错。**

---

### 步骤 8：让规划者感知资料库（grounding）

**做了什么**

**问题背景**：规划者「闭着眼睛」拆问题——它凭自己知道的知名框架（AutoGen、MetaGPT、Generative Agents）拆，但这些资料库里都没有，导致报告全是「资料未涵盖」。

**解决：给规划者一份「资料库地图」**

```python
def build_docs_overview(max_items=20):
    """给规划者一份「资料库地图」

    ⚠️ 这里【不做检索、不做阈值过滤】，直接列出所有段落。
       原因：地图必须「完整」。如果地图本身被过滤过，
             规划者就不知道某些资料存在，会漏掉本来能覆盖的方向。
    """
    lines = []
    for c in chunks[:max_items]:
        title = c["text"].split("\n")[0].strip()
        lines.append(f"- {title[:45]}")
    return "\n".join(lines)
```

然后塞进规划者的 prompt：

```python
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
```

**效果**

| 改之前 | 改之后 |
|---|---|
| 子问题提到 AutoGen、MetaGPT、Generative Agents ❌ | ReAct / Plan-and-Execute、多Agent、记忆 ✅ |
| 报告大量「资料未涵盖」 | 有对比表格，结构清晰 |

**这个思路的名字叫「grounding」（接地）**——让模型的输出**基于真实的可用信息**，而不是凭空发挥。

**遗留问题**：地图只给了**标题**，信息不够。规划者看到「多 Agent 协作」，就以为可以问「角色分工细节」——但资料里只有一句模式列举。

**改进方向**：给前 80 字（标题 + 一点正文），让规划者知道「内容有多少」。

---

### 步骤 9：两阶段检索（最曲折的一步）

**发现的问题**

报告里说：

> 「资料中**未涵盖**「RAG 检索增强生成」这一概念」

**但 RAG 明明在资料库里！** 这是一个**检索失败**——资料在库里，但没被检索到。

#### 排查过程（七步）

**① 提出假设**

怀疑是 `top_k=3` 把 RAG 切掉了。

**② 隔离验证（写 30 行脚本）**

```python
"""验证：子问题3 能否检索到 RAG 段落"""
question = "Agent 的记忆机制与 RAG 检索增强生成如何协同支持 Agent 的长期与短期信息处理"

qv = model.encode(question, normalize_embeddings=True)
scores = embeddings @ qv
ranked = scores.argsort()[::-1]

for rank, i in enumerate(ranked, 1):
    title = chunks[i]["text"].split("\n")[0][:32]
    mark = "  ★ top3 内" if rank <= 3 else ""
    print(f"  {rank}. [{scores[i]:.3f}] {title}{mark}")
```

结果：

```
  1. [0.809] Agent 的记忆机制      ★ top3
  2. [0.622] Agent 的可观测性      ★ top3      ← ⚠️ 这和问题无关！
  3. [0.617] 多 Agent 协作         ★ top3
  4. [0.607] RAG 检索增强生成                  ← 差 0.010 被切掉
  5. [0.572] Agent 的评测方法
  6. [0.525] ReAct 架构
  7. [0.453] Plan-and-Execute
```

**假设成立**，但还发现了**更严重的问题**。

**③ 发现更深的病根：向量检索有噪音**

问题问的是「记忆机制与 RAG 如何协同」，和「**可观测性**」毫无关系——但它排到了**第 2 名（0.622）**，比多 Agent 和 RAG 都高。

**只是共享了「Agent」几个词，语义相似度就虚高了。**

**④ 应用工业界标准方案：两阶段检索**

| 阶段 | 用什么 | 目标 | 特点 |
|---|---|---|---|
| **① 粗筛（召回）** | 向量检索（bi-encoder）| **宁可多，别漏** | 快，但排序粗 |
| **② 精排（重排）** | 重排序模型（cross-encoder）| **挑出最相关的** | 慢，但判断准 |

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)

def search(query, top_k=3, coarse_k=20):
    # ① 粗筛：宽召回
    qv = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ qv
    idx = scores.argsort()[::-1][:coarse_k]
    candidates = [chunks[i] for i in idx]

    # ② 精排
    pairs = [[query, c["text"]] for c in candidates]
    rr_scores = reranker.predict(pairs)
    ...
```

**效果（对比两个排名）**

```
【阶段 1】向量检索
  1. [0.809] Agent 的记忆机制
  2. [0.622] Agent 的可观测性      ← 噪音
  3. [0.617] 多 Agent 协作
  4. [0.607] RAG 检索增强生成      ← 被切掉

【阶段 2】重排序
  1. [0.998] Agent 的记忆机制      ← 真相关，分数很高
  2. [0.759] RAG 检索增强生成      ← 升上来了！✅
  3. [0.059] ReAct 架构
  4. [0.020] Agent 的可观测性      ← 被压下去了！✅
```

**关键认知：分数区分度**

```
向量检索：0.809  0.622  0.617  0.607  0.572  0.525  0.453
          └──────── 全挤在 0.45~0.81，很难判断谁真相关 ────────┘

重排序：  0.998  0.759  0.059  0.020  0.010  0.002  0.000
          └─真相关─┘    └──────── 基本都不相关 ────────┘
```

**rerank 的核心价值是「敢做出明确判断」。**

**⑤ 发现自己的修复有 bug**

粗筛阶段写了 `[:top_k]`：

```python
idx = scores.argsort()[::-1][:top_k]      # ❌ 只取 top_k 个候选
```

**问题**：粗筛阶段就把范围切到 3 个，**rerank 只能在向量 top-3 里排序，废掉了一半功力**。

```python
idx = scores.argsort()[::-1][:coarse_k]   # ✅ 宽召回（coarse_k=20）
```

**⑥ 修复后暴露真正的病根**

修复粗筛后，用复合问题测试：

```
【向量检索】RAG 第 3
【重排序】
  1. [0.997] 多 Agent 协作      ★
  2. [0.875] Agent 的记忆机制   ★
  3. [0.781] ReAct 架构         ★  ← ReAct 挤进来了
  4. [0.578] RAG 检索增强生成       ← RAG 掉到第 4 ❌
```

**关键承认**：粗筛修复原理上是对的，但它**暴露了真正的病根**——

> **`top_k=3` 这个硬限制，假设了「总是需要恰好 3 段」——这是错的。**

不同问题需要不同数量的资料：

```
「RAG 的典型流程是什么」     → 只需 1 段
「多Agent如何结合记忆和RAG」 → 需要 4 段
```

**⑦ 改成自适应取数**

```python
def search(query, max_k=5, min_score=0.3, coarse_k=20):
    """两阶段检索：向量粗筛 → 重排序精排 → 自适应取数

    ⚠️ 注意 max_k 的语义：不是「必须取 5 段」，而是「最多 5 段」。
       实际取几段由 min_score 决定 —— 相关几段就取几段。
    """
    ...
    results = []
    for i in order:
        if rr_scores[i] < min_score:
            break
        results.append(candidates[i])
        if len(results) >= max_k:
            break
    return results
```

**`top_k` → `max_k` 这个改名很重要**：

```
top_k = 3     → 「我要 3 段」（固定数量）
max_k = 5     → 「最多 5 段」（上限，实际取多少看相关度）
```

**关键改动**：阈值从 `0.1` 提到 `0.3`

```
0.578  RAG        ← 0.1 和 0.3 都放行（真相关）
0.059  ReAct      ← 0.1 放行 ❌（噪音）；0.3 拦下 ✅
```

**⑧ 隔离验证**

```python
import step9_rerank as m          # 直接导入被测模块

test_cases = [
    ("复合问题", "多 Agent 协作架构如何结合 Agent 的记忆机制与 RAG 检索增强生成来完成复杂任务"),
    ("单一问题", "RAG 的典型流程是什么"),
    ("两概念协同", "Agent 的记忆机制与 RAG 检索增强生成如何协同支持 Agent 的长期与短期信息处理"),
]

for label, q in test_cases:
    docs = m.search(q)
    print(f"【{label}】→ 检索到 {len(docs)} 段资料")
```

> **注意**：能这样导入，是因为主流程被 `if __name__ == "__main__":` 保护起来了。

**验证结果（完美符合预期）**

```
【复合问题】→ 4 段：多Agent + 记忆机制 + ReAct + RAG ✅
【单一问题】→ 1 段：RAG                              ✅
【两概念协同】→ 2 段：记忆机制 + RAG                  ✅
```

**不再是整齐的 3/3/3**——按需取数生效。

**⑨ 端到端验证**

修复后的报告：

| 版本 | 报告怎么说的 |
|---|---|
| **修复前** | 「资料**未提及** RAG 检索增强生成」 ❌ **事实错误** |
| **修复后** | 「资料涉及 RAG 典型流程（切块、向量化、检索...），但**未涉及它与多Agent的结合**」 ✅ **准确** |

**精确地区分了「资料有什么」和「缺什么」。**

---

## 三、关键问题专题

### 专题 1：为什么「资料在库里却检索不到」

**完整链路**：

```
问题：报告说「资料未提及 RAG」，但资料库里有 RAG 段落
  ↓
假设：top_k=3 把 RAG 切掉了
  ↓
隔离验证：RAG 排第 4，只差 0.010
  ✓ 假设成立
  ↓
但发现更严重的问题：可观测性（无关内容）排第 2
  → 向量检索的排序本身不可靠
  ↓
应用 rerank：RAG 升到第 2，噪音降到第 4
  ↓
又发现自己的 bug：粗筛 [:top_k] 切太死
  → 修复为宽召回
  ↓
修复后暴露真正的病根：top_k=3 是硬限制
  → 改成自适应取数（门槛 + 上限）
  ↓
验证：4 / 1 / 2 ✅
```

**教训**：

> **修一层，露一层。没有一次做对，只有逐步逼近。**

**而且**：即使修复方向正确，也可能在某个具体 case 上让结果变差。**改动必须在多个 case 上验证。**

---

### 专题 2：多 Agent 的链式依赖

```
规划者拆出无法回答的子问题
   ↓
调研者检索不到 → 返回无关内容
   ↓
撰写者只能写「资料未涵盖」
   ↓
审校者发现「报告没内容」→ 判不通过
   ↓
重写 → 问题在上游，改不动 → 还是不通过
   ↓
耗尽次数，强制结束（问题仍在）
```

**结论**：

> **在多 Agent 系统里，下游怎么努力都补不上上游的错。**

**解法**：分头处理两端

| 位置 | 措施 |
|---|---|
| **上游** | 让规划者感知资料库（grounding），少拆「答不了的问题」 |
| **下游** | 让审校员理解「资料不足 ≠ 报告写得差」，不判无意义的返工 |

---

### 专题 3：用工程手段补 LLM 的固有缺陷

这是 Agent 工程师的核心工作。

| LLM 的缺陷 | 工程手段 |
|---|---|
| 规划者凭想象拆问题 | **给它资料库地图**（grounding）|
| 模型编造事实 | 强制「只用资料」+ 审校 Agent |
| 输出格式不稳定 | 防御性解析（`lstrip` / 正则）|
| 向量检索有噪音 | 两阶段检索（cross-encoder 精排）|
| 固定取数不合理 | 自适应取数（门槛 + 上限）|
| 无限循环 | 最大迭代次数兜底 |
| 审校员判断标准模糊 | prompt 里写死「判定规则」|

---

## 四、踩坑总结表

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `ValueError: matmul ... size 7 is different from 512` | 矩阵乘法顺序写反 | `embeddings @ qv`（相邻维度必须相等）|
| 2 | 模型输出带编号，解析出错 | 模型不一定听话 | 防御性 `lstrip` / 正则 |
| 3 | `InvalidUpdateError: At key 'findings'` | `findings: [list, add]` 少了 `Annotated` | `Annotated[list, add]` |
| 4 | `NameError: name 'topic' is not defined` | 改节点时忘了 `state['topic']` | 所有参数都从 `state` 取 |
| 5 | 提示「已达最大修改次数」但审校其实通过了 | 路由判断顺序错误 | 先看审校结论，再查次数 |
| 6 | 审校连续两次不通过，但报告没问题 | 分不清「能修」和「修不了」| prompt 里明确判定规则 |
| 7 | 子问题提到 AutoGen/MetaGPT（资料库里没有）| 规划者看不到资料库 | grounding：给它资料库地图 |
| 8 | 报告说「资料未提及 RAG」，但资料库里有 | 向量检索排序不准 + `top_k` 切太死 | 两阶段检索 + 自适应取数 |
| 9 | 无关内容（可观测性）排到第 2 | 向量检索只看「语义接近」，识别不了噪音 | cross-encoder 重排序 |
| 10 | 修复粗筛后 RAG 反而掉到第 4 | `top_k=3` 是硬限制，假设了「总是需要 3 段」| 改用「相关性门槛 + 上限」自适应取数 |

---

## 五、经验教训

### 1. 每一步都要能单独验证

```
❌ 坏做法：一口气写完 5 个 Agent，然后一起调
✅ 好做法：先验证检索 → 再验证单个 Agent → 再串起来 → 再改图
```

**好处**：出错时能立刻定位到刚写的那几行。

### 2. 调试要用「隔离测试」

```
❌ 坏做法：在完整的 5 个 Agent 流程里加 print，被输出淹没
✅ 好做法：把要验证的「一个点」单独拎出来，写 20 行脚本测
```

**尤其是要绕开随机性**——规划者每次拆的子问题不一样，端到端测试不可复现。

**最干净的隔离测试：直接 import 被测模块**

```python
import step9_rerank as m
docs = m.search("RAG 的典型流程是什么")
```

（前提：主流程被 `if __name__ == "__main__":` 保护）

### 3. 单点验证会骗人

这次的经历：

```
只看「RAG 流程」那个问题 → 以为修好了
测了「复合问题」→ 才发现 top_k 的硬限制
```

**只有评测集能系统性地发现回归。**

### 4. 显示「分数」比只显示「排名」信息量大

```
只看排名：  1. 记忆  2. 可观测性  3. 多Agent  4. RAG
           看不出差距

加分数：    0.809 / 0.622 / 0.617 / 0.607
           → 发现第 3、4 名只差 0.010（刀刃上的差别）
           → 发现可观测性 0.622 是异常高（噪音）
```

### 5. 名词要反映语义

```python
top_k = 3     # ❌ 「我要 3 段」—— 固定数量的假设是错的
max_k = 5     # ✅ 「最多 5 段」—— 上限
```

**改名本身就是一次设计修正。**

### 6. 承认「我的修复可能是错的」

粗筛修复原理正确，但**暴露了更深的病根**，甚至在某个 case 上让结果变差。

**敢于承认并继续往下挖，比急着宣布「修好了」更有价值。**

---

## 附录：最终版本核心代码

### State 定义

```python
class State(TypedDict):
    """Agent 之间共享的「黑板」"""
    topic: str
    sub_questions: list
    findings: Annotated[list, add]     # reducer：并行分支用 add 合并
    report: str
    review: str
    revision_count: int
```

### 图结构

```python
builder = StateGraph(State)
builder.add_node("plan", plan_node)
builder.add_node("research_one", research_one)
builder.add_node("write", write_node)
builder.add_node("review", review_node)

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out)          # 并行分叉
builder.add_edge("research_one", "write")               # 并行汇聚
builder.add_edge("write", "review")
builder.add_conditional_edges(                          # 条件回边（循环）
    "review",
    route_after_review,
    {"rewrite": "write", "end": END},
)

graph = builder.compile()
```

### 两阶段检索

```python
def search(query, max_k=5, min_score=0.3, coarse_k=20):
    # ① 粗筛：宽召回
    qv = model.encode(query, normalize_embeddings=True)
    idx = (embeddings @ qv).argsort()[::-1][:coarse_k]
    candidates = [chunks[i] for i in idx]

    # ② 精排
    rr_scores = reranker.predict([[query, c["text"]] for c in candidates])

    # ③ 自适应取数
    results = []
    for i in rr_scores.argsort()[::-1]:
        if rr_scores[i] < min_score:
            break
        results.append(candidates[i])
        if len(results) >= max_k:
            break
    return results
```

---

*本文档记录了项目的完整开发过程。所保留的 `step1` ~ `step9` 脚本，可以看到一个多 Agent 系统是怎么一步步长出来的。*
