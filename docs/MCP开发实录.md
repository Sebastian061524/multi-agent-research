# MCP 知识库服务 · 开发实录

> 记录把「多 Agent 调研项目」的资料检索能力，封装成一个**标准 MCP Server** 的完整过程：
> 每一步做了什么、为什么这么做、怎么做、怎么验证。
>
> 项目仓库：`multi-agent-research`
>
> 最终交付：`mcp_server.py`（121 行）+ `mcp_demo.py`（验证器，158 行）

---

## 目录

- [一、项目概览](#一项目概览)
- [二、背景知识：MCP 是什么](#二背景知识mcp-是什么)
- [三、开发过程（9 个步骤）](#三开发过程9-个步骤)
- [四、关键问题专题](#四关键问题专题)
- [五、踩坑总结表](#五踩坑总结表)
- [六、经验教训](#六经验教训)
- [附录 A、最终代码](#附录-a最终代码)
- [附录 B、MCP 常用命令](#附录-bmcp-常用命令)

---

## 一、项目概览

### 目标

把调研项目里已经做好的**资料检索能力**（BGE 嵌入 + 两阶段检索 + 查询分解），
封装成一个**标准 MCP Server**，让任何支持 MCP 的客户端都能使用：

```
调研项目的检索代码（不动）
        ↓ 加一层封装
MCP Server（三原语齐全）
        ↓
┌───────────────┬───────────────┬───────────────┐
│ LangChain     │ Claude        │ Cursor /      │
│ Agent         │ Desktop       │ VS Code       │
└───────────────┴───────────────┴───────────────┘
```

**关键约束：不修改、不复制 `main.py` 里的检索逻辑**——只做封装。

### 最终能力清单

| 原语 | 名称 | 谁控制 | 作用 |
|---|---|---|---|
| **Tool** | `search_knowledge(query, max_results, smart_query)` | 模型 | 检索资料片段 |
| **Tool** | `list_knowledge_sources()` | 模型 | 列出知识库有哪些资料 |
| **Resource** | `data://kb/overview` | 应用 | 知识库概览（篇数/段数/明细）|
| **Resource** | `data://kb/chunk/{index}` | 应用 | 按下标读原始片段（资源模板）|
| **Prompt** | `research_report(topic)` | 用户 | 一键生成「调研某主题」的请求 |

### 技术栈

| 组件 | 用什么 |
|---|---|
| MCP 框架 | FastMCP 4.0.10 |
| MCP 协议 | 2026-07-28 版本 |
| 客户端（验证用）| `fastmcp.Client`（内存内）+ `langchain.mcp.MCPAdapter` |
| 复用来源 | `main.py`（`research.search()` / `research.chunks` / `research.llm`）|
| 传输 | 开发时 `stdio`/内存内，调试时 `http`（端口 8001）|

### 与主项目的关系

```
research_agent/
├── main.py            ← 多 Agent 调研报告（原有，未改动）
├── mcp_server.py      ← 【新增】MCP 知识库服务
├── mcp_demo.py        ← 【新增】三原语验证器 + Agent 接入演示
├── eval_retrieval.py  ← 检索层评测（原有）
├── eval_report.py     ← 报告层评测（原有）
└── docs/
    ├── 多Agent项目开发实录.md   ← 原有
    └── MCP开发实录.md           ← 本文档
```

**复用方式**：`mcp_server.py` 里 `import main as research`，直接用它的检索函数与数据。
**零重复代码**，代价是启动时会连带加载嵌入/重排序模型（约 15 秒）。

---

## 二、背景知识：MCP 是什么

### 2.1 一句话定义

> **MCP（Model Context Protocol）是一个让「AI 应用」和「外部能力」互相通信的标准协议。**

类比：

| 类比 | 对应 |
|---|---|
| USB-C 接口 | MCP 协议 |
| 各种外设（硬盘/显示器/网卡）| 各种 MCP Server |
| 电脑 | AI 应用（Host）|
| 驱动 | MCP Client |

**没有 MCP 的世界**：你的检索能力要接 5 个 Agent，就得写 5 套适配代码。
**有了 MCP**：写一次 Server，谁都能接。

### 2.2 架构：三个角色

```
┌──────────────────────────────────────────────────┐
│  Host（宿主）—— 用户在用的 AI 应用                 │
│    · Claude Desktop / Cursor / 你的 Agent 程序     │
│                                                   │
│   ┌─────────────────────────────────────────┐    │
│   │  Client —— Host 内部负责通信的组件        │    │
│   └────────────────┬────────────────────────┘    │
└────────────────────┼─────────────────────────────┘
                     │ MCP 协议（JSON-RPC）
                     ↓
┌──────────────────────────────────────────────────┐
│  Server（服务器）—— 能力提供方                     │
│    · 你写的 mcp_server.py                         │
│    · 暴露 Tools / Resources / Prompts             │
└──────────────────────────────────────────────────┘
```

### 2.3 ⭐ 三个原语：区别在「谁来决定用它」

这是整个 MCP 里最重要的一张表：

| 原语 | 谁控制 | 一句话 | 典型例子 |
|---|---|---|---|
| **Tools** | **模型** | 模型自己判断"该调这个工具了" | 查询数据库、发邮件、检索资料 |
| **Resources** | **应用** | 你的代码决定"把这份资料放进上下文" | 文件内容、用户配置、文档目录 |
| **Prompts** | **用户** | 用户在界面上主动选一个模板 | 快捷按钮、预设指令 |

**用场景理解**：

```
【应用启动时】读取 resources://user-profile → 塞进 system prompt     ← Resources
【用户看界面】[ 查询订单 ] [ 规划出行 ]  ← 用户点这个                ← Prompts
【模型回答中】「我需要查航班」→ 调用 search_flight                    ← Tools
```

### 2.4 什么时候用哪个

| 需求 | 用哪个 | 为什么 |
|---|---|---|
| 让模型**自己决定**去做某事 | **Tools** | 需要模型的判断力 |
| 把**固定资料**放进上下文 | **Resources** | 不需要模型判断，也不该浪费一次工具调用 |
| 给用户一个**快捷入口/模板** | **Prompts** | 保证输入格式一致，一键触发 |

### 2.5 传输方式

| 传输 | 怎么连 | 适合 |
|---|---|---|
| **内存内** | `Client(server对象)` | **开发调试**（无端口、无子进程）✅ |
| **stdio** | `Client(Path("server.py"))` | 桌面客户端（Claude Desktop）|
| **HTTP** | `Client("http://host/mcp")` | 远程/生产部署 |

> **开发时首选内存内连接**——彻底避开「服务没启动」「端口不对」这类问题。

---

## 三、开发过程（9 个步骤）

### 步骤 1：搭 Server 空壳 + 一个测试工具

#### 做了什么

创建 `mcp_server.py`，注册一个测试工具 `ping`，能启动。

#### 为什么

```
先验证「最小链路」通了：
   Server 能创建 → 工具能注册 → 能启动
        ↓
   再往里加真实能力
```

**如果一上来就写检索**，出问题时你会分不清是「MCP 配置错了」还是「检索接错了」。

#### 怎么做

```python
"""把调研项目的资料检索能力封装成 MCP Server"""
from fastmcp import FastMCP

mcp = FastMCP("Knowledge Base Server")


@mcp.tool
def ping() -> str:
    """测试用：返回 pong。"""
    return "pong"


if __name__ == "__main__":
    mcp.run(transport="http", port=8001)
```

三个要点：

| 代码 | 作用 |
|---|---|
| `FastMCP("Knowledge Base Server")` | 创建 Server，名字会展示给客户端 |
| `@mcp.tool` | 把函数注册成工具（**忘了加它，工具会静默消失**）|
| `mcp.run(...)` | 启动服务（端口用 8001，避开其他服务）|

#### 验证

```powershell
D:\PythonCode\.venv\Scripts\python.exe mcp_server.py
```

期望看到：

```
FastMCP 4.0.10
🖥  Server:      Knowledge Base Server
🚀 Starting MCP server with transport 'http' on http://127.0.0.1:8001/mcp
INFO:     Uvicorn running on http://127.0.0.1:8001
```

**按 Ctrl+C 停止时会出现 `KeyboardInterrupt` traceback——这是正常的，不是错误。**

#### ⚠️ 遇到的问题

**问题：`ping` 的 description 是 `None`**

原因：写代码时**漏掉了 docstring**：

```python
@mcp.tool
def ping() -> str:
    return "pong"          # ← 没有 docstring
```

后果：`list_tools()` 显示 `ping —— None`。

**教训**：docstring 就是工具的对外说明。**没有它，工具在模型眼里等于"没有描述"**，
而 FastMCP **不会报错**（静默失效）。

---

### 步骤 2：写「通用检查器」

#### 做了什么

创建 `mcp_demo.py`，一个**不关心 Server 具体有什么、只是把三原语全列出来**的检查脚本。

#### 为什么

```
❌ 写死检查：「应该有 3 个工具」
   → 每加一个能力就得改脚本

✅ 通用检查：「把服务器上的东西全列出来」
   → 加了多少能力，跑一次全看到
```

而且它**主动打印 description 和 schema**——这是发现「静默失效」的唯一手段
（步骤 1 的 `None` 就是这么发现的）。

#### 怎么做

```python
"""检查 mcp_server 提供了什么能力"""
import asyncio

from fastmcp import Client

import mcp_server


async def main():
    # 内存内连接：不用起服务、不用管端口
    async with Client(mcp_server.mcp) as c:

        print("=" * 64)
        print("Tools（模型控制）")
        print("=" * 64)
        for t in await c.list_tools():
            print(f"  {t.name} —— {t.description}")
            print(f"      参数：{t.input_schema}")

        print()
        print("=" * 64)
        print("Resources（应用控制）")
        print("=" * 64)
        for r in await c.list_resources():
            print(f"  资源 {r.uri} —— {r.description}")
        for tpl in await c.list_resource_templates():
            print(f"  模板 {tpl.uri_template}")

        print()
        print("=" * 64)
        print("Prompts（用户控制）")
        print("=" * 64)
        for p in await c.list_prompts():
            print(f"  模板 {p.name} —— {p.description}")
            for arg in (p.arguments or []):
                req = "必填" if arg.required else "可选"
                print(f"      - {arg.name}（{req}）：{arg.description}")


asyncio.run(main())
```

#### 验证

```
================================================================
Tools（模型控制）
================================================================
  ping —— 测试用：返回 pong。
      参数：{'type': 'object', 'additionalProperties': False, 'properties': {}}

================================================================
Resources（应用控制）
================================================================

================================================================
Prompts（用户控制）
================================================================
```

**看到 `ping —— 测试用：返回 pong。` 说明 docstring 生效了**（对照步骤 1 的 `None`）。

#### 💡 关键认知

**「内存内连接」是 MCP 开发最重要的技巧**：

```python
async with Client(mcp_server.mcp) as c:      # 直接传 server 对象
```

它不需要启动服务、不需要端口、不需要网络——**所有服务端开发都应该先这么调**。

---

### 步骤 3：加第一个真正的工具 `search_knowledge`（假数据）

#### 做了什么

加一个检索工具，**但先返回假数据**。

#### 为什么

```
真实检索要 import main
   → 加载两个模型（约 15 秒）
   → 需要 API Key
   → 万一出错，分不清是「工具定义写错」还是「检索接错」
        ↓
先用假数据把「工具定义」这一层单独验证
```

#### 怎么做

```python
from mcp.types import ToolAnnotations


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def search_knowledge(query: str, max_results: int = 5) -> list[dict]:
    """在本地知识库里检索资料片段。

    用途：需要查找事实、定义、方案对比时调用。
    返回的是原文片段，请只依据这些片段作答，不要编造。

    Args:
        query: 检索问题，请一次只问一个概念；如果要查多个方面，请分多次调用本工具
        max_results: 最多返回几段资料，默认 5 段
    """
    # TODO: 暂时返回假数据
    return [
        {"source": "sample_ai_agents.txt", "text": f"关于「{query}」的假资料第一段"},
        {"source": "sample_ai_agents.txt", "text": f"关于「{query}」的假资料第二段"},
    ]
```

#### 三个设计要点

**① `annotations=ToolAnnotations(read_only_hint=True)`**

```
告诉客户端：「这个工具只读，不会修改任何东西」
   → 客户端跳过确认弹窗 → 体验流畅
```

对比：如果是「删除数据」的工具，应标 `destructive_hint=True`，客户端会弹确认框。

**② docstring 的三段结构**

```python
    """在本地知识库里检索资料片段。        ← 第 1 段：是什么（模型判断"要不要用"）

    用途：需要查找事实、定义、方案对比时调用。   ← 第 2 段：什么时候用
    返回的是原文片段，请只依据这些片段作答，不要编造。  ← 行为指令

    Args:                                  ← 第 3 段：参数说明
        query: ...                              ⚠️ Args 前必须空一行
        max_results: ...
    """
```

> **第 2 段里的「请只依据这些片段作答，不要编造」是一条给模型的指令**
> ——把「防幻觉」写进工具，而不是写在调用方的 prompt 里。

**③ 返回 `list[dict]` 而不是拼好的字符串**

| 返回方式 | 效果 |
|---|---|
| 拼成一段字符串 | 模型要"解析"文字，客户端拿不到结构化数据 |
| **返回 `list[dict]`** | **模型好读 + 客户端能反序列化** ✅ |

#### 验证

检查器输出的 schema：

```python
'properties': {
    'query': {'type': 'string', 'description': '检索问题，请一次只问一个概念；...'},
    'max_results': {'default': 5, 'type': 'integer', 'description': '最多返回几段资料，默认5段'}
},
'required': ['query']
```

**检查清单**：

```
□ query 有 description，且在 required 里
□ max_results 有 description + default: 5，且【不在】required 里
```

> **`required` 里只有 `query`**——因为 `max_results` 有默认值。
> 这是「类型注解 + 默认值 → JSON Schema」自动推导的结果。

---

### 步骤 4：接上真实检索 + 引入 `smart_query`

#### 做了什么

把假数据换成对 `main.py` 里检索函数的真实调用，并加一个 `smart_query` 开关。

#### 为什么（复用而非重写）

```
❌ 重写：复制 BGE 加载、两阶段检索、查询分解
        → 40 行重复代码；调研项目改了它不会跟着改

✅ 复用：import main，用它的 search()
        → 零重复
```

#### 怎么做

**改动 1**：顶部加 import

```python
import main as research          # 复用调研项目的检索能力
```

**改动 2**：替换函数体

```python
    docs = research.search(query, max_k=max_results, use_decompose=smart_query)
    return [{"source": d["source"], "text": d["text"]} for d in docs]
```

**改动 3**：加 `smart_query` 参数

```python
def search_knowledge(query: str, max_results: int = 5,
                     smart_query: bool = False) -> list[dict]:
    """...
    Args:
        query: 检索问题，请一次只问一个概念；如果要查多个方面，请分多次调用本工具
        max_results: 最多返回几段资料，默认 5 段
        smart_query: 是否让服务器内部再做一次「查询分解」（把复合问题拆成多个子查询）。
                     默认关闭——因为你通常已经自己拆分过查询了，开两次是重复劳动。
    """
```

#### ⭐ 为什么默认 `smart_query=False`（关键设计推理）

**观察到的现象**：接入 Agent 后，模型自己会把一个问题拆成多次调用：

```
【用户】ReAct 和 Plan-and-Execute 有什么区别？
【模型→工具】search_knowledge  参数={'query': 'ReAct 架构 推理与行动'}
【模型→工具】search_knowledge  参数={'query': 'Plan-and-Execute 架构 先规划再执行'}
【模型→工具】search_knowledge  参数={'query': 'ReAct 与 Plan-and-Execute 对比 区别'}
```

而 `research.search()` **内部还会用 LLM 再拆一次**：

```
模型拆一遍 + Server 内部再拆一遍 = 双重分解
   → 3 次工具调用 × 每次 1 个额外 LLM 调用
   → 约 6~12 秒的纯浪费
```

**所以设计决策是**：

```
① 把「拆解查询」的责任【转移】给模型（它本来就会）
② 默认关掉 Server 内部的分解   → smart_query=False
③ 但用 docstring 明确告知模型怎么配合：
      "请一次只问一个概念；如果要查多个方面，请分多次调用本工具"
④ 保留开关给【程序调用方】（它们不会自己拆）
```

**这是一个通用原则**：

> **当调用方从「程序」变成「LLM」时，要重新审视：哪些逻辑该留在工具里？**
> **让 LLM 做「理解」，让你的代码做「执行」。**

| 你原本在 Server 里做的 | Agent 场景下是否多余 |
|---|---|
| 查询分解 | ✅ 模型自己会拆 |
| 查询改写 | ✅ 模型自己会改 |
| 关键词提取 | ✅ 模型自己会提 |
| 意图分类 | ✅ 模型自己会判断 |
| **真正的检索计算** | ❌ 必须留在 Server |
| **权限校验 / 缓存 / 限流** | ❌ 必须留在 Server |

#### 验证

```
加载嵌入模型...
加载重排序模型...
已索引 7 个段落                       ← import main 的副作用

查询：ReAct 架构的原理 → count = 2  ✅
查询：多 Agent 协作如何结合记忆机制与 RAG → count = 2

smart_query 对比（同一个复合问题）
  关闭（默认）→ 2 段
  开启        → 3 段                  ← 开关生效 ✅
```

#### ⚠️ 遇到的问题

**问题：`PydanticJsonSchemaWarning: Default value False is not JSON serializable`**

原因：默认值写成了**小写的 `false`**，IDE 自动补上了 `from sympy import false`：

```python
from sympy import false                                   # ← IDE 自动导入的
def search_knowledge(..., smart_query: bool = false):     # ← 不是 Python 的 False
```

实测后果：

```
type(false)     = <class 'sympy.logic.boolalg.BooleanFalse'>   ← 不是 bool！
repr(false)     = False                                        ← ⚠️ 打印一模一样
bool(false)     = False                                        ← 所以"碰巧能用"
json.dumps      = ❌ Object of type BooleanFalse is not JSON serializable
```

**最阴险的地方**：`repr` 打印出来和真的 `False` **完全一样**，只有 `type()` 才能揭穿。
而且 `bool(sympy_false) == False`，所以**功能碰巧没坏**，只表现为一个 warning。

**修法**：删掉那行 import，改成大写 `False`。

> **这是同一类坑的第二次**：调研项目里 IDE 曾自动导入过 `from langfuse.api import feedback`。
> **共同点：你打了一个名字，IDE 猜错了库，自动帮你 import——导入成功了所以不报错。**

---

### 步骤 5：加第二个工具 `list_knowledge_sources`

#### 做了什么

让模型能**先知道知识库里有什么**，再决定查什么。

#### 为什么

```
没有它：模型不知道知识库里有什么
        → 可能问一堆查不到的问题 → 只能说"资料未涵盖"

有了它：模型可以先看目录 → 再决定问什么 ✅
```

**这正是主项目里 `build_docs_overview()` 的思路**——让规划者先感知资料库。
只不过主项目是「应用主动塞进 prompt」，这里是「做成工具让模型主动调用」。

#### 怎么做

```python
@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_knowledge_sources() -> list[str]:
    """列出知识库里都有哪几篇资料（只有文件名，不含内容）。

    用途：想先了解「这个知识库能回答哪些方面的问题」时调用。
    """
    return sorted({c["source"] for c in research.chunks})
```

三个要点：

| 代码 | 说明 |
|---|---|
| `research.chunks` | `main.py` 的模块级变量（7 个 `{"source", "text"}` 字典）|
| `{c["source"] for c in ...}` | **集合推导式**——自动去重 |
| `sorted(...)` | 集合无序，加 `sorted` 保证**顺序确定、可复现** |

#### ⚠️ 遇到的问题：三种推导式，只差一对括号

写代码时漏了花括号：

```python
return sorted(c["source"] for c in research.chunks)      # ❌ 生成器表达式，不去重
return sorted({c["source"] for c in research.chunks})    # ✅ 集合推导式，去重
```

**后果：返回 7 个重复的文件名**（因为 7 个片段都来自同一篇资料）。

| 写法 | 去重 | 用途 |
|---|---|---|
| `[...]` 列表推导式 | ❌ | 保留顺序和重复 |
| **`{...}` 集合推导式** | **✅** | **需要去重** |
| `(...)` 生成器表达式 | ❌ | 数据量大、不想一次性占内存 |

**注意**：`sorted(x for x in ...)` 是合法的——**当生成器是唯一参数时，外层括号可以省略**。
所以「少写一对花括号」不会报语法错误，只会静默地不去重。

#### 顺带验证：原始类型集合的返回结构

```
data       : ['sample_ai_agents.txt']                      ← 客户端自动拆包
structured : {'result': ['sample_ai_agents.txt']}          ← 协议原样，包了一层
```

**规则**（与工具那层一致）：

```
返回【原始类型或集合】→ structured_content 被包成 {"result": ...}
返回【dict / Pydantic 模型】→ structured_content 是原样的对象
```

**`.data` 是客户端的友好版，`.structured_content` 是协议原样。**

---

### 步骤 6：加 Resources（应用控制）

#### 做了什么

把「知识库概览」和「按下标读片段」做成 **Resources**（而不是 Tools）。

#### 为什么：同一个需求的两种做法

```
需求：「知识库里有什么」

做法 A：做成 Tool（步骤 5 已做）
    → 模型自己决定「我要不要看看目录」

做法 B：做成 Resource（本步骤）
    → 应用决定「我要把目录放进上下文」
```

| | Tool | Resource |
|---|---|---|
| 谁决定 | **模型** | **应用（你的代码）** |
| 标识 | 函数名 | **URI**（`data://kb/overview`）|
| 执行时机 | 模型调用时 | 客户端**读取时**（懒加载）|
| 典型用途 | 模型主动查东西 | 应用往上下文里塞资料 |

#### 怎么做

```python
import json
from fastmcp.exceptions import ResourceError


# ==================== Resources（应用控制）====================

@mcp.resource("data://kb/overview")
def get_kb_overview() -> str:
    """知识库概览：包含哪些资料、各有多少个片段"""
    counter = {}
    for c in research.chunks:
        counter[c["source"]] = counter.get(c["source"], 0) + 1

    return json.dumps(
        {
            "资料篇数": len(counter),
            "片段总数": len(research.chunks),
            "明细": counter,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource("data://kb/chunk/{index}")
def get_chunk(index: int) -> str:
    """按下标读取知识库里的某个知识片段

    Args:
        index: 片段下标，从0开始
    """
    total = len(research.chunks)
    if index < 0 or index >= total:
        raise ResourceError(f"片段下标越界：{index}，有效范围是 0 ~ {total - 1}")

    c = research.chunks[index]
    return json.dumps(
        {"index": index, "source": c["source"], "text": c["text"]},
        ensure_ascii=False,
    )
```

#### 五个要点

**① `@mcp.resource("uri://...")` —— 必须有 URI**

```
Tool     → 靠【名字】标识（search_knowledge）
Resource → 靠【URI】标识（data://kb/overview）
```

**② 懒加载**：函数只在「有客户端读这个 URI」时才执行。

**③ 资源模板 `{index}`**：一个函数服务无限多 URI，不用为每个片段注册资源。
URI 里的 `{index}` 是字符串，但注解写 `index: int`——**FastMCP 会自动转换**。

**④ ⭐ Resource 必须返回 JSON 字符串，不能直接返回 dict**

实测（同一个 dict，不同的返回方式）：

| 写法 | 客户端读到的内容 |
|---|---|
| `return dict` | `{"channel": "\u624b\u673aAPP", ...}` ❌ **中文被转义** |
| `return dict` + `mime_type="application/json"` | 同上 ❌ |
| **`return json.dumps(data, ensure_ascii=False)`** | `{"channel": "手机APP", ...}` ✅ |

**这是 FastMCP 4 的一个不一致**：Tools 序列化时保留了中文，Resources 没有。
造成两个代价：**token 浪费**（每个中文字符膨胀成 6 个字符）+ **可读性差**。

**⑤ 越界该抛 `ResourceError`**

```
「下标越界」属于哪一类？（用「错误分三类」的框架判断）
   ├─ 参数格式错       → 该抛错
   ├─ 业务上没找到     → 正常返回
   └─ 请求了一个不存在的资源  → 【抛 ResourceError】✅
```

**理由**：MCP 协议本身就是这么定义的（资源不存在 → 协议级错误）。
客户端能明确区分「**读取失败**」和「读到了内容」。

#### ⚠️ 遇到的问题：装饰器用错

代码里写成了：

```python
@mcp.tool("data://kb/overview")           # ❌
def get_kb_overview() -> str: ...
```

**后果**：

```
tool 装饰器的第一个位置参数是【name】（工具名）
   ↓
于是注册了一个【工具】，名字叫 "data://kb/overview"
   ↓
⚠️ Tool name validation warning: contains invalid characters ':', '/'
   ↓
它出现在 Tools 列表里，Resources 里什么都没有
   ↓
read_resource("data://kb/overview") → MCPError: Resource not found ❌
```

**两个装饰器的第一个参数含义完全不同**：

| 装饰器 | 第一个参数 | 含义 |
|---|---|---|
| `@mcp.tool(...)` | `name` | **工具名**（只能用字母数字下划线点横线）|
| `@mcp.resource(...)` | `uri` | **资源地址**（可以是 `data://xxx`）|

> **而 warning 早就说了**：`Tool name validation warning ... contains ':' '/'`
> **——它说的是「工具名」，不是「资源 URI」。**

#### 验证

```
Resources（应用控制）
  资源 data://kb/overview —— 知识库概览：包含哪些资料、各有多少个片段
  模板 data://kb/chunk/{index}

  读 data://kb/overview：
{
  "资料篇数": 1,
  "片段总数": 7,
  "明细": {
    "sample_ai_agents.txt": 7
  }
}

  读 data://kb/chunk/0：{"index": 0, ... "text": "ReAct 架构..."}
  读 data://kb/chunk/6：{"index": 6, ... "text": "Agent 的评测方法..."}

  越界：✅ 正确报错 → MCPError: 片段下标越界：99，有效范围是 0 ~ 6
```

**顺便注意服务端日志**：

```
[22:26:18] Error reading resource 'data://kb/chunk/99'
  ┌─────────── Traceback ───────────┐
  │ ... mcp_server.py:95 in get_chunk │
  │ > 95 │ raise ResourceError(...)   │
  └──────────────────────────────────┘
```

| 谁看到什么 | 内容 |
|---|---|
| **服务端日志** | 完整 traceback（含文件行号）→ 可排查 ✅ |
| **客户端** | 只有一句干净的 `片段下标越界：99，有效范围是 0 ~ 6` |

**这是安全设计**：内部细节不泄漏给客户端，但开发者能看到。

---

### 步骤 7：加 Prompt（用户控制）

#### 做了什么

把「调研某主题」这个流程做成一个可复用的 Prompt 模板。

#### 为什么

```
用户在客户端看到：[ 📄 撰写调研报告 ]  ← 快捷按钮
        ↓ 点击 → 填「调研主题」
        ↓
客户端调 prompts/get?name=research_report&topic=...
        ↓
Server 返回拼好的指令
        ↓
客户端把它作为【用户消息】发给模型
```

**价值**：**同一个复杂指令可以被不同用户一键复用**——
别人接你的 Server，点一下就能用上你这套「不编造」的调研流程。

#### 怎么做

```python
@mcp.prompt
def research_report(topic: str) -> str:
    """生成一份「撰写调研报告」的请求。

    Args:
        topic: 调研主题，例如「大模型 Agent 有哪些主流架构」
    """
    return (
        f"请帮我调研「{topic}」。\n\n"
        f"要求：\n"
        f"1. 先调用 search_knowledge 检索资料，不要凭记忆作答。\n"
        f"2. 报告结构：概述、分点论述、结论。\n"
        f"3. 只使用检索到的资料；资料不足时明确写出「资料未涵盖」。\n"
        f"4. 引用资料时标注来源文件名。"
    )
```

#### 四个要点

**① 名字取函数名**（和 Tool 一样，没有 URI——Prompt 是"用户点的按钮"，不需要地址）。

**② 参数 = 模板变量**，客户端调用时提供。

**③ 返回 `str` 会自动变成一条 user 消息**。也可以返回多轮：

```python
from fastmcp.prompts import Message

@mcp.prompt
def roleplay(character: str) -> list[Message]:
    return [
        Message(f"你扮演 {character}。"),               # user
        Message("好的，我准备好了。", role="assistant"),
    ]
```

**④ ⭐ Prompt 不执行任何业务逻辑**

```
Tool     → 模型给参数 → 你的代码【执行】→ 返回结果
Prompt   → 客户端给参数 → 你的代码【拼一段话】→ 把话发给模型
                            ↑ 没有查询、没有计算，只是字符串拼接
```

**本质是一个「预设的提示词模板」。**

**⑤ 内容来自主项目的防幻觉设计**：

```
"3. 只使用检索到的资料；资料不足时明确写出「资料未涵盖」。"
            ↑ 这就是主项目 write_node 里的约束
```

**以前它写在代码的 prompt 字符串里；现在变成了一个可复用的 MCP 模板。**

#### ⚠️ 遇到的问题：函数名拼错

```python
def resarch_report(topic: str) -> str:      # ❌ 少了一个 e
```

**后果**：`list_prompts()` 显示 `resarch_report`，而客户端调 `research_report` → `Unknown prompt`。

**关键认知**：

```
普通函数：名字只影响你自己（IDE 会同步改所有调用点）
装饰后的函数：名字【变成了对外的 API】❗
```

**而且这个错很"沉默"**：服务端注册成功、列表也正常——
**只有「两端用了不同的名字」才会暴露**。

**防拼错的手段**：用 `name=` 显式指定接口名。

```python
@mcp.prompt(name="research_report")     # 接口名与实现名解耦
def resarch_report(topic: str) -> str: ...
```

#### 验证

```
Prompts（用户控制）
  模板 research_report —— 生成一份「撰写调研报告」的请求。
      - topic（必填）：调研主题，例如「大模型 Agent 有哪些主流架构」

  [user] 请帮我调研「大模型 Agent 架构」。

要求：
1. 先调用 search_knowledge 检索资料，不要凭记忆作答。
2. 报告结构：概述、分点论述、结论。
3. 只使用检索到的资料；资料不足时明确写出「资料未涵盖」。
4. 引用资料时标注来源文件名。
```

**⚠️ 注意取值方式是 `msgs.messages[0].content.text`**——
`content` 是个 **TextContent 对象**，不是字符串（和 `call_tool` 的 `.content[0].text` 同类）。

---

### 步骤 8：接进 Agent

#### 做了什么

在 `mcp_demo.py` 里加 `part2()`：用 `MCPAdapter` 把 Server 接进 LangChain Agent，
让大模型自己决定调用工具。

#### 为什么

**验证 Server 真的"能用"**：

```
如果只有 list_tools 打印得漂亮，那只是"定义正确"
        ↓
必须让一个真实的模型用它，才能验证：
   · docstring 是否让模型选对工具
   · schema 是否让模型填对参数
   · 返回值模型能否读懂
```

#### 怎么做

```python
async def part2():
    """第二部分：把 MCP Server 接进 LangChain Agent"""
    from langchain.agents import create_agent
    from langchain.mcp import MCPAdapter
    from langchain_core.messages import HumanMessage

    # ① 内存内连接 MCP Server
    async with MCPAdapter(mcp_server.mcp) as adapter:

        # ② 把 MCP 工具转成 LangChain 工具
        tools = await adapter.list_tools()
        print(f"拿到 {len(tools)} 个工具：{[t.name for t in tools]}")

        # ③ 创建 Agent（带人设）
        agent = create_agent(
            research.llm,
            tools,
            system_prompt=(
                "你是一名中文技术研究员。"
                "回答前先调用 search_knowledge 检索资料，只依据检索到的内容作答；"
                "资料不足时如实说明「资料未涵盖」。回答用中文，简洁专业。"
            ),
        )

        # ④ 提问
        question = "ReAct 和 Plan-and-Execute 这两个架构有什么区别？"
        result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})

        # ⑤ 打印整个对话过程
        for msg in result["messages"]:
            kind = type(msg).__name__
            if kind == "HumanMessage":
                print(f"【用户】{msg.content}")
            elif kind == "AIMessage":
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"【模型→工具】{tc['name']}  参数={tc['args']}")
                if msg.content:
                    print(f"\n【模型回答】\n{msg.content}\n")
            elif kind == "ToolMessage":
                print(f"【工具返回】{str(msg.content)[:110]}...")
```

#### 四个要点

**① `MCPAdapter(server对象)` 支持内存内连接**——不用起服务。

**② `research.llm` 直接复用主项目的 LLM 对象**（已配好 DeepSeek 的 key 与 base_url）。

**③ `system_prompt` 是 Agent 的「人设 + 规矩」**：

| 它管什么 | 本项目的写法 |
|---|---|
| 人设 | 「你是一名中文技术研究员」|
| 行为 | 「回答前先调用 search_knowledge 检索资料」|
| 约束 | 「只依据检索到的内容作答；资料不足时如实说明」|
| 语言/风格 | 「回答用中文，简洁专业」|

**④ 打印消息流是唯一的调试手段**——四段式结构：

```
① HumanMessage   「用户的问题」
② AIMessage      content='我先查一下...' + tool_calls=[search_knowledge]
③ ToolMessage    工具返回的 JSON
④ AIMessage      最终回答
```

#### ⚠️ 遇到的问题（3 个）

**问题 1（主因）：`"message"` 少了 `s`——第二次犯同一个错**

```python
result = await agent.ainvoke({"message": [HumanMessage(content=question)]})
                              # ↑ 应该是 "messages"
```

**后果**：

```
键名 "message" 不存在 → LangGraph 静默忽略
   ↓
State 里的 messages = 空
   ↓
模型收到的消息列表里【没有你的问题】
   ↓
它只能自己瞎探索：先列目录、再乱试查询
   ↓
最后问用户「请问你想了解哪方面？」
```

**⭐ 而同一个 bug，两次的症状完全不同**：

| | 第一次（无 system_prompt）| 第二次（有 system_prompt）|
|---|---|---|
| 结果 | **报错** `Empty input messages` | **不报错**，但行为完全偏离 |

因为 system_prompt 让消息列表至少有 1 条 → DeepSeek 不报错 → 模型拿着
「只有人设、没有问题」的上下文开始瞎猜。

> **「不报错」比「报错」更危险**——报错你会查，不报错你可能以为它只是在"认真探索"。

**识别信号**：输出的消息流里**缺少 `【用户】` 那一行**。

**问题 2：`print` 少了 `f`**

```python
print("\n用户：{question}\n")          # ❌ 打印字面量 {question}
print(f"\n用户：{question}\n")         # ✅
```

只影响显示，但它**掩盖了真相**——你会以为问题传进去了。

**问题 3：只调用了 `part2()`**

```python
asyncio.run(part2())                    # ❌ part1 定义了但没被调用
```

**问题 4（在 Agent 行为里发现）：`AI Agent` 查询返回空**

模型瞎试时出现的：

```
search_knowledge(query='AI Agent')       → []
search_knowledge(query='智能体 定义')     → []
search_knowledge(query='agent')          → ✅ 有结果
```

**留待步骤 9 处理。**

#### 验证（修完前三个问题后）

```
拿到 3 个工具：['ping', 'search_knowledge', 'list_knowledge_sources']

用户：ReAct 和 Plan-and-Execute 这两个架构有什么区别？

【用户】ReAct 和 Plan-and-Execute 这两个架构有什么区别？      ← ✅ 出现了
【模型→工具】search_knowledge  参数={'query': 'ReAct 架构'}            ← ⭐ 干净！
【模型→工具】search_knowledge  参数={'query': 'Plan-and-Execute 架构'}  ← ⭐ 干净！

【模型回答】
## 核心差异小结
| 维度 | ReAct | Plan-and-Execute |
| 规划时机 | 逐步、动态 | 执行前一次性制定 |
| 是否全局规划 | 缺乏 | 有 |
...
资料未涵盖两者在性能指标、实现成本或具体框架（如 LangChain）层面的对比信息。
```

#### ⭐ docstring 的效果，前后对比

| | 步骤 3 时期（docstring 还没写那句）| 现在（写了「一次只问一个概念」）|
|---|---|---|
| 调用次数 | **3 次** | **2 次** |
| 查询内容 | `'ReAct 架构 推理与行动'`<br>`'Plan-and-Execute 架构 先规划再执行'`<br>`'ReAct 与 Plan-and-Execute 对比 区别'` | `'ReAct 架构'`<br>`'Plan-and-Execute 架构'` |
| 特点 | 带修饰词、有复合描述 | **一个概念一个查询** |

**这就是「把行为指令写进 docstring」的直接回报。**

---

### 步骤 9：健壮性改进——空结果要说清原因

#### 做了什么

把 `search_knowledge` 的返回类型从 `list[dict]` 改成 `dict`，
在**没有结果时给出原因和下一步建议**。

#### 为什么：从「兜底」到「说明」的转折

**最初的设想是「兜底」**：全被门槛拦下时，退回向量检索的 top-k，永不返回空。

**但数据推翻了这个方案。** 实测（阈值 `min_score=0.3`）：

| 查询 | 向量 top1 | 重排 top1 | 过门槛数 | 该不该兜底 |
|---|---|---|---|---|
| `AI Agent` | 0.607 | 0.062 | 0 | 我以为该 |
| **`医疗行业如何落地 Agent`** | **0.552** | 0.007 | 0 | **不该** |
| `AutoGen 框架的通信协议` | 0.497 | 0.001 | 0 | 不该 |
| `量子计算` | 0.387 | 0.002 | 0 | 不该 |
| `ReAct 架构` | 0.757 | 1.000 | 2 | 正常 |

**关键**：`AI Agent`（0.607/0.062）和 `医疗行业如何落地 Agent`（0.552/0.007）
**分数几乎一样** —— **用分数无法区分「措辞问题」和「资料缺失」**。

**而且重新想一步**：

```
「AI Agent 查不到」真的是 bug 吗？
   → 资料库里只有 7 个具体主题（ReAct/Plan-and-Execute/多Agent/RAG/记忆/可观测性/评测）
   → 没有一段在回答「AI Agent 是什么」
   → reranker 说「这些都答不了这个问题」
   → 【它和拒绝「医疗行业」是同一个逻辑，是对的】
```

**同时，兜底方案还有一个致命副作用**：

```
主项目的评测集里有两个用例期望 0 段：
   「AutoGen 框架的通信协议」「医疗行业如何落地 Agent」
        ↓
如果加兜底 → 这两个也会返回结果
        ↓
评测通过率从 13/13 掉下来 ❌
【破坏了辛苦建立的「不硬凑」原则】
```

**所以真正的问题是**：

```
❌ 不是「该不该返回空」的问题
✅ 是「返回空的时候，有没有说清楚为什么」的问题
```

| 方案 | 做法 | 问题 |
|---|---|---|
| 兜底 | 硬塞 top-k 结果 | 引入噪音 + 破坏「不硬凑」+ 无法区分该不该兜 |
| **说明** | 返回空 + **为什么空 + 建议怎么办** | ✅ 无副作用 |

#### 怎么做

```python
@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def search_knowledge(query: str, max_results: int = 5,
                     smart_query: bool = False) -> dict:
    """在本地知识库里检索资料片段。

    用途：需要查找事实、定义、方案对比时调用。
    返回的是原文片段，请只依据这些片段作答，不要编造。

    Args:
        query: 检索问题，请一次只问一个概念；如果要查多个方面，请分多次调用本工具
        max_results: 最多返回几段资料，默认 5 段
        smart_query: 是否让服务器内部再做一次「查询分解」，默认关闭

    Returns:
        count —— 命中的段数
        docs  —— 命中的资料列表（每项含 source 和 text）
        note  —— 附加说明：为什么是这个结果，或者为什么没有结果
    """
    docs = research.search(query, max_k=max_results, use_decompose=smart_query)

    if docs:
        return {
            "count": len(docs),
            "docs": [{"source": d["source"], "text": d["text"]} for d in docs],
            "note": "已按相关性排序。请只使用这些片段作答。",
        }

    # ---- 空结果：说清楚为什么，并给出下一步建议 ----
    return {
        "count": 0,
        "docs": [],
        "note": (
            "知识库里没有段落达到相关性门槛。可能原因："
            "① 这个方向资料里没有；"
            "② 你的问法与资料用词差异较大。"
            "建议：先调用 list_knowledge_sources 看知识库涵盖哪些主题；"
            "或换一种更贴近资料的说法重试。"
            "如果确实没有，请如实告知用户「资料未涵盖」。"
        ),
    }
```

#### 为什么这个设计更好

| 好处 | 说明 |
|---|---|
| **不引入噪音** | 没相关就不给不相关的 |
| **保留「不硬凑」** | 评测不会退化 |
| **模型知道发生了什么** | `count: 0` + `note` 解释了原因 |
| **有下一步指引** | 建议先看目录、换措辞 |
| **防幻觉** | note 重申「资料未涵盖就如实说」|
| **符合错误分类规范** | 业务结果结构化（与主项目 `{"found": False, "message": ...}` 同一模式）|

**注意 docstring 加了 `Returns:` 段**——否则模型不知道返回的是
`{count, docs, note}` 结构，可能还在期待一个列表。

#### ⚠️ 遇到的问题：改接口后漏改调用点

**问题**：把返回值从 `list` 改成 `dict` 后，`mcp_demo.py` 里的
「smart_query 对比」没有同步修改：

```python
print(f"  关闭（默认）→ {len(r1.data)} 段")     # ❌ r1.data 现在是 dict！
print(f"  开启        → {len(r2.data)} 段")
```

**而 `len(dict)` 返回的是【键的数量】= 3**：

```
正确的值应该是「2 段」或「3 段」
        ↓
而 len(dict) 恰好也是 3
        ↓
输出显示「3 段」—— 看起来完全合理！
        ↓
根本不会怀疑 ❌
```

**修法**：

```python
print(f"  关闭（默认）→ {r1.data['count']} 段")
print(f"  开启        → {r2.data['count']} 段")
```

**两个教训**：

```
① 改接口时，必须找出所有调用点
     改完用 grep / IDE 的 Find Usages 扫一遍

② len() 对不同类型含义不同，用错不报错
     list → 元素个数
     dict → 键的个数      ← 本次踩的
     str  → 字符个数
```

#### 验证

```
查询：ReAct 架构的原理 → count = 2
  note: 已按相关性排序。请只使用这些片段作答。

查询：AI Agent → count = 0
  note: 知识库里没有段落达到相关性门槛。可能原因：① 这个方向资料里没有；
        ② 你的问法与资料用词差异较大。建议：先调用 list_knowledge_sources
        看知识库涵盖哪些主题；或换一种更贴近资料的说法重试。
        如果确实没有，请如实告知用户「资料未涵盖」。

smart_query 对比
  关闭（默认）→ 2 段      ← 恢复正常 ✅
  开启        → 3 段
```

**Agent 的最终回答**：

```
**核心区别：先规划还是边想边做**
...
（以上依据知识库中 `sample_ai_agents.txt` 的资料片段；
  关于两者的性能基准对比、具体实现代码等，资料未涵盖。）
```

**注意最后两行**：

| 亮点 | 来自哪里 |
|---|---|
| 标注来源文件名 | **Prompt 模板的第 4 条要求** |
| 主动说明「资料未涵盖」 | **system_prompt + 工具 docstring 双重约束** |

**三个地方写的约束全部生效了**：

```
① system_prompt   → 人设 + 语言 + 只依据资料作答
② 工具 docstring  → 行为指令（一次一个概念 / 不要编造）
③ Prompt 模板     → 报告结构 + 标注来源
```

---

## 四、关键问题专题

### 专题 1：docstring 决定模型怎么用工具

**这是整个 MCP 开发里投入产出比最高的一件事。**

一个工具在模型眼里长什么样，完全由三部分决定：

| 部分 | 模型用它判断 |
|---|---|
| **工具 docstring 第 1 段** | 「这是干什么的」→ 要不要用 |
| **工具 docstring 第 2 段** | 「什么时候用」+ 行为指令 |
| **参数说明（Args 段）** | 「每个参数填什么」|

**实测效果**：

```
docstring 里加一句「请一次只问一个概念；如果要查多个方面，请分多次调用本工具」
        ↓
模型的查询从：                    变成：
  'ReAct 架构 推理与行动'           'ReAct 架构'
  'Plan-and-Execute 架构 先规划再执行'  'Plan-and-Execute 架构'
  'ReAct 与 Plan-and-Execute 对比 区别'
  （3 次调用，带修饰、有复合）      （2 次调用，一个概念一个查询）
```

**而 `enum` 还有额外价值**：

```
schema 里写了 class_type 的 enum: ['economy', 'business', 'first']
        ↓
模型主动提示用户：「如需查询商务舱/头等舱，请告诉我～」
        ↓
【enum 不只是"防止瞎填"，还让模型知道"有哪些选项"，从而主动引导用户】
```

### 专题 2：`smart_query` —— 责任转移的设计

**问题**：模型自己会拆查询，Server 内部也拆 → 双重分解 → 浪费 LLM 调用与延迟。

**解决**：做成开关，默认关闭，**并用 docstring 把责任明确转移给模型**。

```
smart_query=False（默认）+ docstring 指导  =  一次正确的分工 ✅
smart_query=False        + 没有 docstring  =  埋了个坑 ❌
```

**但开关不能删**——因为还有非 Agent 的调用方：

| 调用方 | 会自己拆查询吗 | 该用 smart_query |
|---|---|---|
| **Agent（模型）** | ✅ 会 | `False`（默认）|
| **程序代码** | ❌ 不会 | `True` |
| **人手动调试** | ❌ 不会 | `True` |

**通用原则**：

> **当调用方从「程序」变成「LLM」时，重新审视哪些逻辑该留在工具里。**
> **让 LLM 做「理解」，让你的代码做「执行」。**

### 专题 3：「查不到」的两种原因，处理方式完全相反

| 现象 | 真相 | 该怎么办 |
|---|---|---|
| **拆开了也查不到** | 技术问题（检索质量差）| 调检索策略（门槛/模型/混合检索）|
| **拆开了查得到，但答不了** | **资料边界** | **别调参数——补资料，或让模型诚实标注** |

**本项目的实例**：

```
查询「多 Agent 协作如何结合记忆机制与 RAG」
        ↓
拆成 3 次单概念查询 → 3 段全部命中 ✅（检索成功）
        ↓
但这 3 段讲的是「各自是什么」
        ↓
问题问的是「如何结合」
        ↓
【仍然答不了】❌ —— 因为资料里没有这个答案
```

**还有一个更细的层次：门槛与上限是两回事**

```
门槛（min_score）  ：够不够相关？   ← 决定"能不能进来"
上限（max_results）：最多取几段？   ← 决定"进来了取几个"

先过门槛，再受上限约束。
```

实测：把 `max_results` 从 2 调到 5 都没用——因为只有 2 段过了门槛。

### 专题 4：空结果该「兜底」还是该「说明」

**这是一次完整的数据驱动设计决策**：

```
提出方案（兜底，永不返回空）
      ↓ 用数据检验
数据推翻了它（无法区分"措辞问题"和"资料缺失"；且会破坏「不硬凑」原则）
      ↓ 重新定义问题
真正的问题是：「空结果没有说明」
      ↓
更好的方案：不兜底，但给解释 ✅
```

**关键一步是「重新定义问题」**：

> 从「**怎么兜住结果**」变成「**怎么让调用方理解结果**」。
> 很多设计困境，换个问题定义就解开了。

**最终实现**：返回 `{count, docs, note}`——
空时 `note` 说明原因 + 给建议 + 重申「资料未涵盖要如实说」。

### 专题 5：「静默失效」的六种形态

这是本项目贯穿始终的一条暗线：

| # | 场景 | 拼错/漏了什么 | 报错了吗 |
|---|---|---|---|
| 1 | `ping` 的说明 | 漏了 docstring | ❌ 完全静默（`description=None`）|
| 2 | `search_knowledge` 的默认值 | `false` 小写（IDE 从 sympy 导入）| ⚠️ 只有 warning |
| 3 | `list_knowledge_sources` | `{...}` 写成生成器表达式 | ❌ 静默（返回 7 个重复）|
| 4 | `agent.ainvoke` 的入参 | `"message"` 少了 `s`（两次）| 一次报错、一次静默偏离 |
| 5 | `smart_query` 对比 | `len(dict)` 当成段数 | ❌ 完全静默（恰好是 3）|
| 6 | 装饰器 | `@mcp.tool` 当成 `@mcp.resource` | ⚠️ warning |

**共同点**：

```
不报错、不崩溃、流程照跑、【结果不对】。
而且输出往往"看起来合理" → 最难发现。
```

**对付它们的唯一办法是「主动检查」**：

| 手段 | 本项目里的例子 |
|---|---|
| **打印中间状态** | 检查器打印 `description` 和 `input_schema` |
| **列清单验证** | `list_tools()` / `list_resources()` / `list_prompts()` |
| **看输出的第一行** | 消息流里没有 `【用户】` → 输入问题 |
| **认真读 warning** | 两次 warning 都直接指出了真问题 |
| **扫一眼 import 区** | `sympy`、`langfuse` 出现在这里就是可疑 |
| **想清楚类型** | `len(dict)` 和 `len(list)` 不是一回事 |

**另外两条具体经验**：

```
① 装饰后的函数名 = 对外接口名
   （拼错一个字母就是 API 变了；可用 name= 解耦）

② 改接口时必须找出所有调用点
   （改完 grep / Find Usages 扫一遍）
```

### 专题 6：Resources vs Tools —— 同一个需求的两种做法

**需求**：「让使用方知道知识库里有什么」

**本项目两种都做了**：

| 做法 | 实现 | 谁决定 | 什么时候执行 |
|---|---|---|---|
| **Tool** | `list_knowledge_sources()` | 模型 | 模型调用时 |
| **Resource** | `data://kb/overview` | 应用 | 客户端读取时 |

**对比主项目**：

| | 主项目 | 本 Server |
|---|---|---|
| 做法 | 应用**主动**把概览塞进 prompt | 模型**自己决定**要不要看 |
| 对应原语 | **Resources 的思路** | **Tools 的思路** |

> **同一个信息，两种封装——区别就在「谁来决定」。**

---

## 五、踩坑总结表

| # | 坑 | 现象 | 根因 | 修法 | 类型 |
|---|---|---|---|---|---|
| 1 | `ping` 说明是 `None` | `list_tools` 显示 `None` | 漏写 docstring | 补 docstring | 静默失效 |
| 2 | `false` 小写 | `PydanticJsonSchemaWarning` | IDE 从 sympy 自动导入 | 用大写 `False` | IDE 误导入 |
| 3 | 生成器表达式 | 返回 7 个重复文件名 | `{...}` 写成 `(...)` | 加花括号 | 静默失效 |
| 4 | 装饰器用错 | `data://` 出现在 Tools 里<br>`Resource not found` | `@mcp.tool` 当 `@mcp.resource` | 换成 `@mcp.resource` | warning 未读 |
| 5 | 函数名拼错 | `Unknown prompt` | `resarch_report` 少个 e | 改函数名 / 用 `name=` | 名字=接口 |
| 6 | `"message"` 少 `s` | 第一次报错<br>第二次行为偏离 | 键名拼错，LangGraph 静默忽略 | 改成 `"messages"` | **重犯 + 静默** |
| 7 | `print` 少 `f` | 输出 `{question}` | 字面量 | 加 `f` | 显示错误 |
| 8 | 只调 `part2()` | part1 没跑 | 漏写调用 | 两个都调 | 逻辑遗漏 |
| 9 | `len(dict)` | 显示「3 段」（键数）| 改接口后漏改调用点 | `r.data["count"]` | **静默失效** |
| 10 | `AI Agent` 查不到 | 返回 `[]` | 重排把全部打到门槛下 | 加 `note` 说明原因 | 系统级发现 |
| 11 | 越界处理 | `IndexError` | 没做边界检查 | 抛 `ResourceError` | 设计决策 |

---

## 六、经验教训

### 1. 把「对外说明」当成代码的一部分

```
docstring 不是注释，它是【接口】。
   工具说明 → 模型判断"要不要用"
   参数说明 → 模型判断"填什么"
   行为指令 → 模型照着做

错误处理 → 调用方知道"出什么事了"
返回值结构 → 调用方知道"怎么读"
```

**本项目里，一句 docstring 就改变了模型的查询行为。**

### 2. 「静默失效」是最大的敌人

```
会报错的 bug 是好朋友——它告诉你哪里错了。
不报错但做错事的 bug 才是敌人——它让你以为一切正常。
```

**对策**：主动检查（打印 schema、列清单、看第一行输出、读 warning）。

### 3. warning 是线索，不是噪音

本项目两次 warning 都直接指向了真问题：

```
PydanticJsonSchemaWarning: Default value False is not JSON serializable
   → false 是 sympy 对象

Tool name validation warning: contains invalid characters ':', '/'
   → 用错了装饰器
```

### 4. 用数据检验方案，而不是靠直觉

```
最初设想：加兜底（永不返回空）
        ↓ 实测数据
AI Agent 0.607/0.062  vs  医疗行业 0.552/0.007
        ↓
分数无法区分 → 方案不成立
        ↓ 重新定义问题
不是"怎么兜住"，而是"怎么解释"
```

**没有数据，就会写一个看起来合理、实际有害的功能。**

### 5. 让 LLM 做「理解」，让代码做「执行」

```
✅ 该留给代码：检索计算、权限校验、缓存、限流、业务规则
✅ 该交给模型：查询拆解、意图判断、改写、关键词提取
```

**判断标准**：如果模型本来就会做，别在 Server 里重做一遍。

### 6. 「查不到」要先分清是技术问题还是资料问题

```
技术问题  → 调参数、换模型、加混合检索
资料边界  → 别调参数；补资料，或让模型诚实标注
```

**把「资料里没有」当成 bug 去修，是白费力气，还会引入噪音。**

### 7. 接口一旦变化，就必须找出所有调用点

```
改返回值结构 → 所有取 .data 的地方都要改
        ↓
漏掉的那个不会报错，只会算错（len(dict) = 3）
```

### 8. 开发时永远先用「内存内连接」

```python
async with Client(mcp_server.mcp) as c:      # 无端口、无子进程、无网络
```

**把所有服务端开发都先在内存里跑通，最后才去试 HTTP/stdio。**
本项目从头到尾的服务端验证都是这么做的。

### 9. 分层构建，每层都验证

```
空壳 → 检查器 → 假数据工具 → 真实检索 → 第二个工具 → Resource → Prompt → Agent
  ↑        ↑          ↑            ↑            ↑          ↑         ↑        ↑
每次只加一点，加完立刻验证
```

**好处**：出问题时能立刻定位到"刚加的那一层"。

### 10. 复用优先，但要清楚代价

```
✅ import main → 零重复代码
⚠️ 代价：启动时加载两个模型（约 15 秒）+ 需要 API Key
```

**如果启动速度重要**，下一步可以把检索部分抽成独立的 `retrieval.py`：

```
research_agent/
├── retrieval.py     ← 只含模型加载 + search()（无 LLM、无 graph）
├── main.py          ← import retrieval
└── mcp_server.py    ← import retrieval（启动更快）
```

---

## 附录 A、最终代码

### A.1 `mcp_server.py`（121 行）

```python
"""把调研项目的资料检索能力封装成 MCP Server"""
from fastmcp import FastMCP

from mcp.types import ToolAnnotations

import main as research

import json

from fastmcp.exceptions import ResourceError

mcp = FastMCP("Knowledge Base Server")

@mcp.tool
def ping() -> str:
    """测试用：返回 pong。"""
    return "pong"

@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def search_knowledge(query: str, max_results: int = 5, smart_query: bool = False) -> dict:
    """在本地知识库里检索资料片段。

    用途：需要查找事实、定义、方案对比时调用。
    返回的是原文片段，请只依据这些片段作答，不要编造。

    Args:
        query: 检索问题，请一次只问一个概念；如果要查多个方面，请分多次调用本工具
        max_results: 最多返回几段资料，默认 5 段
        smart_query: 是否让服务器内部再做一次「查询分解」，默认关闭

    Returns:
        count —— 命中的段数
        docs  —— 命中的资料列表（每项含 source 和 text）
        note  —— 附加说明：为什么是这个结果，或者为什么没有结果
    """
    docs = research.search(query, max_k=max_results, use_decompose=smart_query)

    if docs:
        return {
            "count": len(docs),
            "docs": [{"source": d["source"], "text": d["text"]} for d in docs],
            "note": "已按相关性排序。请只使用这些片段作答。",
        }

    # ---- 空结果：说清楚为什么，并给出下一步建议 ----
    return {
        "count": 0,
        "docs": [],
        "note": (
            "知识库里没有段落达到相关性门槛。可能原因："
            "① 这个方向资料里没有；"
            "② 你的问法与资料用词差异较大。"
            "建议：先调用 list_knowledge_sources 看知识库涵盖哪些主题；"
            "或换一种更贴近资料的说法重试。"
            "如果确实没有，请如实告知用户「资料未涵盖」。"
        ),
    }


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_knowledge_sources() -> list[str]:
    """列出知识库里都有哪几篇资料（只有文件名 不含内容）

     用途：想先了解「这个知识库能回答哪些方面的问题」时调用。
    """
    return sorted({c["source"] for c in research.chunks})

# ==================== Resources（应用控制）====================
@mcp.resource("data://kb/overview")
def get_kb_overview() -> str:
    """知识库概览：包含哪些资料、各有多少个片段"""
    counter = {}
    for c in research.chunks:
        counter[c["source"]] = counter.get(c["source"], 0) + 1

    return json.dumps(
        {
            "资料篇数": len(counter),
            "片段总数": len(research.chunks),
            "明细": counter,
        },
        ensure_ascii=False,
        indent = 2,
    )

@mcp.resource("data://kb/chunk/{index}")
def get_chunk(index: int) -> str:
    """按下标读取知识库里的某个知识片段

    Args:
        index: 片段下标，从0开始
    """
    total = len(research.chunks)
    if index < 0 or index >= total:
        raise ResourceError(f"片段下标越界：{index}，有效范围是 0 ~ {total - 1}")

    c = research.chunks[index]
    return json.dumps(
        {"index": index, "source": c["source"], "text": c["text"]},
        ensure_ascii=False,
    )

@mcp.prompt
def research_report(topic: str) -> str:
    """生成一份「撰写调研报告」的请求。

    Args:
        topic: 调研主题，例如「大模型 Agent 有哪些主流架构」
    """
    return (
        f"请帮我调研「{topic}」。\n\n"
        f"要求：\n"
        f"1. 先调用 search_knowledge 检索资料，不要凭记忆作答。\n"
        f"2. 报告结构：概述、分点论述、结论。\n"
        f"3. 只使用检索到的资料；资料不足时明确写出「资料未涵盖」。\n"
        f"4. 引用资料时标注来源文件名。"
    )


if __name__ == '__main__':
    mcp.run(transport="http", port=8001)
```

### A.2 `mcp_demo.py` 结构

```
part1()  ── 验证三原语
   ├─ 内存内连接 Client(mcp_server.mcp)
   ├─ 列 Tools / Resources / Prompts
   ├─ 读 data://kb/overview
   ├─ 读 data://kb/chunk/{0,6}
   ├─ 越界测试（期望报错）
   ├─ 真实检索测试（3 个查询，含空结果）
   ├─ smart_query 对比（关闭 2 段 / 开启 3 段）
   ├─ list_knowledge_sources 的返回结构
   └─ 渲染 research_report

part2()  ── 接进 Agent
   ├─ MCPAdapter(mcp_server.mcp) 内存内连接
   ├─ adapter.list_tools() → 3 个工具
   ├─ create_agent(research.llm, tools, system_prompt=...)
   ├─ ainvoke({"messages": [HumanMessage(...)]})
   └─ 打印完整消息流
```

### A.3 接入桌面客户端（示例）

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "D:\\PythonCode\\.venv\\Scripts\\python.exe",
      "args": ["D:\\PythonCode\\research_agent\\mcp_server.py"]
    }
  }
}
```

（需把 `mcp_server.py` 结尾的 `mcp.run(transport="http", ...)` 改成
`mcp.run()`——即默认 **stdio**，这是桌面客户端的标准方式。）

---

## 附录 B、MCP 常用命令

### 开发时的检查清单

```python
async with Client(mcp_server.mcp) as c:
    await c.list_tools()                # 有哪些工具
    await c.list_resources()            # 有哪些资源
    await c.list_resource_templates()   # 有哪些资源模板
    await c.list_prompts()              # 有哪些提示模板

    await c.call_tool("name", {...}, raise_on_error=False)   # 调工具
    await c.read_resource("data://...")                      # 读资源
    await c.get_prompt("name", {...})                        # 渲染提示
```

### 容易忘的细节

| 项 | 说明 |
|---|---|
| `call_tool` 默认 `raise_on_error=True` | 工具报错会**抛异常**；要自己判断就传 `False` |
| 结果取值 | `.data`（友好版）/ `.content[0].text`（文本）/ `.structured_content`（协议原样）|
| Prompt 取值 | `msgs.messages[0].content.text`（`.content` 是对象）|
| Resource 返回值 | 必须是 `json.dumps(..., ensure_ascii=False)` 字符串，否则中文被转义 |
| 工具返回值 | 返回 `dict` 最好（中文正常 + structured 不被包装）|
| `Args:` 段 | **前面必须空一行**，否则参数说明丢失 |
| 装饰器首参 | `tool(name=)` / `resource(uri=)` / `prompt(name=)` |

### 依赖安装

```powershell
pip install fastmcp            # MCP 服务端框架
pip install "langchain[mcp]"   # MCPAdapter（beta）
```

```python
from langchain.mcp import MCPAdapter      # 会打印 LangChainBetaWarning，正常
```

---

**文档结束** · 对应代码：`mcp_server.py` / `mcp_demo.py`
