# MCP 知识库服务

把调研项目（上一级的 `main.py`）的**资料检索能力**封装成一个**标准 MCP Server**——
任何支持 MCP 的客户端（Cursor / Claude Desktop / LangChain Agent / …）都能接入，
而**检索代码零重复**（直接复用 `main.py` 里的 `search()`）。

> 完整开发过程（12 个步骤、15 条踩坑、13 条经验教训）见
> **[../docs/MCP开发实录.md](../docs/MCP开发实录.md)**。

---

## 文件

| 文件 | 作用 |
|---|---|
| **`mcp_server.py`** | **Server 定义**——2 个 Tool + 3 个 Resource + 1 个 Prompt |
| **`mcp_server_stdio.py`** | **stdio 入口**——给桌面客户端用（定义与入口分离）|
| **`mcp_demo.py`** | **验证器**——列三原语 + 接入 LangChain Agent 演示 |
| **`verify_output_schema.py`** | 对比 `dict` 与 Pydantic 的 output schema 差异 |
| `mcp_server.log` | 运行时日志（已 gitignore）|

## 提供的能力

| 原语 | 名称 | 谁控制 |
|---|---|---|
| **Tool** | `search_knowledge(query, max_results, smart_query)` | 模型 |
| **Tool** | `list_knowledge_sources()` | 模型 |
| **Resource** | `data://kb/overview`（概览：篇数/段数）| 应用 |
| **Resource** | `data://kb/toc`（目录：每篇包含哪些片段）| 应用 |
| **Resource** | `data://kb/chunk/{index}`（按下标读片段，模板）| 应用 |
| **Prompt** | `research_report(topic)`（一键生成调研请求）| 用户 |

## 快速开始

### 1. 验证三原语（内存内，不用起服务）

```powershell
D:\PythonCode\.venv\Scripts\python.exe mcp_demo.py
```

### 2. 接入 LangChain Agent（看模型自主检索）

同一个脚本的下半部分会做这件事，直接跑上面那条命令即可。

### 3. 接入桌面客户端（Cursor / Claude Desktop）

配置 `~/.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "knowledge-base": {
      "command": "D:\\PythonCode\\.venv\\Scripts\\python.exe",
      "args": ["D:\\PythonCode\\research_agent\\mcp\\mcp_server_stdio.py"]
    }
  }
}
```

**改完要重启客户端里的 MCP Server**（Server 是常驻进程，改代码不热重载）。

### 4. HTTP 方式（自己调试用）

```powershell
D:\PythonCode\.venv\Scripts\python.exe mcp_server.py
# 监听 http://127.0.0.1:8001/mcp
```

---

## 日志与自检

日志**同时写 stderr 和 `mcp_server.log`**（stdio 客户端不一定显示 stderr）。

四个观测点，缺哪个就知道卡在哪一环：

| 日志行 | 证明什么 |
|---|---|
| `以 stdio 方式启动（被客户端拉起）` | 进程起来了，**而且跑的是新代码** |
| `ping 被调用` | 客户端能调工具（连通性）|
| `检索：…（transport=stdio）` | 模型真的调了工具 + 传输方式 |
| `命中 N 段` | 底层检索链路正常 |

**`transport=stdio` 是"客户端真的连上了"的铁证**（内存内测试时是 `None`）。

---

## 三个设计要点

1. **`smart_query` 默认关闭**——模型自己会拆查询，Server 不再重复拆（避免"双重分解"）
2. **空结果不"硬凑"**——返回 `count / docs / note`，说清原因 + 给建议
   （实测数据表明：靠分数无法区分"措辞不匹配"和"资料确实没有"）
3. **接口设计听真实使用的反馈**——在 Cursor 里问「知识库里有什么」发现要 8 次操作，
   补一个 `data://kb/toc` 资源后降到 **2 次**

---

## 注意事项

| 项 | 说明 |
|---|---|
| **stdio 的 stdout** | 是**协议通道**，任何 `print` 都会污染它；日志一律走 stderr |
| **启动慢** | `import sentence_transformers` 就要 20 秒，加上加载模型共约 28 秒 |
| **改了代码不生效** | MCP Server 是常驻进程，必须重启客户端里的 Server |
| **`ctx.info` 已弃用** | MCP 2026-07-28（SEP-2577）弃用了 logging 能力 → 改用 stderr 日志 |

---

**目录名为什么叫 `mcp` 却不遮蔽 SDK？**
因为本目录**没有 `__init__.py`**（只是普通目录），而 Python 的导入规则是
「正规包优先于命名空间包」，所以 `import mcp` 仍然解析到 site-packages 里的 MCP SDK。
