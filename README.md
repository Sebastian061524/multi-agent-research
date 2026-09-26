# 多 Agent 调研报告生成器

基于 **LangGraph** 的多 Agent 协作系统：输入一个调研主题，自动完成「主题拆解 → 资料调研 → 报告撰写」，输出一份**带引用、不编造**的 Markdown 调研报告。

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| **三 Agent 协作** | 规划者 / 调研者 / 撰写者，各司其职 |
| **自动拆解主题** | 把宽泛主题拆成 3 个可独立调研的子问题 |
| **本地知识库检索** | BGE 中文嵌入 + 向量检索（纯 NumPy 实现，无需向量数据库） |
| **防幻觉** | 强制只用提供的资料，资料不足时明确标注「资料未涵盖」 |
| **可编排** | 基于 LangGraph，加节点 / 并行 / 条件分支都很容易 |

## 🏗️ 架构

```
用户输入调研主题
       ↓
┌─────────────────┐
│ ① 规划者 Agent   │  把主题拆成 3 个可独立调研的子问题
└────────┬────────┘
         ↓
┌─────────────────┐
│ ② 调研者 Agent   │  对每个子问题独立向量检索
└────────┬────────┘
         ↓
┌─────────────────┐
│ ③ 撰写者 Agent   │  整合资料，生成 Markdown 报告（防幻觉）
└────────┬────────┘
         ↓
    report.md
```

**为什么用多 Agent：**

- **上下文独立**：每个子问题单独检索，各自聚焦
- **角色分离**：规划要「发散」、撰写要「收敛」，一个 prompt 兼顾不了
- **可扩展**：加「审校 Agent」只需加一个节点

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：`sentence-transformers` 会依赖 PyTorch。如果在 Linux 上部署，建议先装 CPU 版以节省体积：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. 配置密钥

复制 `.env.example` 为 `.env`（或手动创建），填入 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 准备资料

把你的调研资料（`.txt` 或 `.md`）放进 `research_docs/` 目录，支持多级子目录。

仓库里已附带一份示例资料 `sample_ai_agents.txt`（关于 Agent 架构的 7 段内容）。

### 4. 运行

```bash
python step5_langgraph.py
```

输入主题，例如：

```
请输入调研主题：大模型Agent有哪些主流架构
```

运行完会生成 `report.md`。

## 📄 输出示例

```markdown
## 概述
随着大模型能力的提升，Agent 成为将模型能力落地到复杂任务的重要范式...

## 分点论述

### 一、核心模块上的架构设计范式
**记忆机制**：资料指出，Agent 的记忆分为短期记忆和长期记忆两类...

### 三、不同架构的对比

| 维度 | ReAct | Plan-and-Execute | 多 Agent 协作 |
|------|-------|------------------|---------------|
| 任务复杂度 | 复杂任务易陷入局部最优 | 适合步骤较多的任务 | 适合可拆分的复杂任务 |
| 延迟与成本 | 资料未涵盖此部分 | 资料未涵盖此部分 | 复杂度和成本显著上升 |

## 结论
1. 架构覆盖情况：资料明确涉及 ReAct、Plan-and-Execute 和多 Agent 协作三类。
   Reflexion 等架构**资料未涵盖此部分**。
```

**注意报告里的「资料未涵盖」**——这不是缺陷，而是**防幻觉机制生效**的证据：模型只用你给的资料，不编造。

## 📚 项目构建过程

本项目按「一行行写、每步验证」的方式搭起来，保留了每一步的脚本：

| 文件 | 内容 | 验证点 |
|---|---|---|
| `step1_search.py` | 嵌入模型 + 文档向量检索 | 检索 top-1 命中正确 |
| `step2_planner.py` | 第一个 Agent（规划者） | 模型输出 3 个干净子问题 |
| `step3_pipeline.py` | 规划 → 调研 → 撰写（普通函数版） | 完整报告生成 |
| `step5_langgraph.py` | 改造成 LangGraph | 三个节点串联成图 |
| `step6_parallel.py` | **并行执行**（Send API + reducer） | 三个调研同时开始 |
| `step7_review.py` | **审校 Agent**（条件回边 + 防死循环） | 不通过时自动返工 |
| `step8_grounded.py` | **规划者感知资料库**（grounding） | 不再拆出无资料覆盖的问题 |
| `step9_rerank.py` | **两阶段检索**（功能最全版本） | 按相关度自动取 1~5 段资料 |
| `verify_retrieval.py` | 检索质量隔离验证 | 对比粗排 / 精排排名 |
| `verify_adaptive.py` | 自适应取数验证 | 复合问题取 4 段、单一问题取 1 段 |

保留这些文件，可以看到「一个多 Agent 系统是怎么一步步长出来的」。

## 🔍 关键实现

### 防幻觉的 prompt 设计

```python
prompt = f"""你是一名专业研究员。请根据下面的资料，撰写调研报告。

要求：
1. 使用 Markdown 格式
2. 结构：## 概述、## 分点论述、## 结论
3. **只使用提供的资料**，不要编造资料中没有的信息
4. 如果某个子问题的资料不足，明确写出「资料未涵盖此部分」

【资料】
{materials}
"""
```

第 3、4 条是整个系统的生命线——**没有它们，模型会用自己的知识「补全」报告，内容无法溯源**。

### LangGraph 的 State（Agent 之间的共享黑板）

```python
class State(TypedDict):
    topic: str           # 调研主题
    sub_questions: list  # ① 规划者写
    findings: list       # ② 调研者写
    report: str          # ③ 撰写者写
```

每个节点读取自己需要的字段、返回自己产出的字段，LangGraph 自动合并。

### 两阶段检索（粗筛 + 精排）

**向量检索的排序并不可靠。** 实测中，与问题无关的「Agent 的可观测性」会因为共享「Agent」一词而排到第 2 名，真正相关的 RAG 段落却排在第 4 名——只差 0.010 被 `top_k=3` 切掉。

解法是工业界标准的两阶段检索：

```python
def search(query, max_k=5, min_score=0.3, coarse_k=20):
    # ① 粗筛：向量检索宽召回（宁可多，别漏）
    qv = model.encode(query, normalize_embeddings=True)
    idx = (embeddings @ qv).argsort()[::-1][:coarse_k]
    candidates = [chunks[i] for i in idx]

    # ② 精排：cross-encoder 同时看「问题 + 段落」，重新打分
    rr_scores = reranker.predict([[query, c["text"]] for c in candidates])

    # ③ 自适应取数：取所有「够相关」的，最多 max_k 个
    results = []
    for i in rr_scores.argsort()[::-1]:
        if rr_scores[i] < min_score:
            break
        results.append(candidates[i])
        if len(results) >= max_k:
            break
    return results
```

三个关键设计：

| 要点 | 说明 |
|---|---|
| **粗筛要宽** | 粗筛阶段就按 `top_k` 截断是常见 bug——精排因此失去重排空间，形同虚设 |
| **精排治噪音** | cross-encoder 把问题和段落一起编码，能识别「共享关键词但其实无关」 |
| **自适应取数** | 用「相关性门槛 + 上限」替代固定 `top_k`：相关几段就取几段 |

实测效果（同一个问题）：

```
向量检索：  可观测性 第2 (0.622) │ RAG 第4 (0.607)      ← 噪音当道，RAG 被切掉
重排序后：  记忆机制 0.998 │ RAG 0.759 │ 可观测性 0.020  ← 噪音被压制，RAG 保留
```

自适应取数的验证结果：

| 问题 | 需要的资料量 | 实际取到 |
|---|---|---|
| 多Agent+记忆+RAG（复合） | 4 段 | **4 段** |
| RAG 的典型流程（单一） | 1 段 | **1 段** |
| 记忆与 RAG 协同 | 2 段 | **2 段** |

### 审校 Agent 的判定标准

审校员必须区分两类问题，否则会陷入「永远改不好」的返工：

| 问题类型 | 该判 | 原因 |
|---|---|---|
| 结构混乱、编造内容、遗漏了资料中已有的信息 | **不通过** | 报告自身的问题，可以修复 |
| 某子问题资料不足（报告已诚实标注） | **通过** | 资料的限制，改多少遍也没用 |

同时必须有**最大迭代次数**兜底——否则审校员持续判「不通过」会导致死循环。

## 🔧 技术栈

- **编排**：LangGraph
- **大模型**：DeepSeek（deepseek-chat）
- **嵌入模型**：BAAI/bge-small-zh-v1.5（本地运行，512 维）
- **检索**：向量检索（NumPy 矩阵运算实现）

## 📈 可扩展方向

- [ ] **并行检索**：3 个子问题同时检索（LangGraph `Send` API）
- [ ] **审校 Agent**：报告写完后自动检查，不通过则重写
- [ ] **规划者感知资料库**：让规划者先看资料目录，避免拆出无资料覆盖的子问题
- [ ] **混合检索 + 重排序**：提升检索精度
- [ ] **网络搜索**：把 `search_docs()` 替换成搜索 API，支持实时调研

## 📄 License

MIT
