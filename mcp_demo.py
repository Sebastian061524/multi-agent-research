"""检查 mcp_server 提供了什么能力

这个脚本不关心 Server 里具体有什么，
它只是把「三原语」全部列出来 —— 加了新东西自然就能看到。
"""
import asyncio

from fastmcp import Client

import mcp_server

import main as research

async def part1():
    async with Client(mcp_server.mcp) as c:
        print("=" * 64)
        print("Tools (模型控制)")
        print("=" * 64)
        for t in await c.list_tools():
            print(f"  {t.name} —— {t.description}")
            print(f"      参数：{t.input_schema}")

        print("=" * 64)
        print("Resources（应用控制）")
        print("=" * 64)

        for r in await c.list_resources():
            print(f"  资源 {r.uri} —— {r.description}")
        for tpl in await c.list_resource_templates():
            print(f"  模板 {tpl.uri_template}")

        print()
        ov = await c.read_resource("data://kb/overview")
        print(f"  读 data://kb/overview：\n{ov[0].text}")

        print()
        for i in [0, 6]:
            one = await c.read_resource(f"data://kb/chunk/{i}")
            print(f"  读 data://kb/chunk/{i}：{one[0].text[:95]}...")

        print("=" * 64)
        print("Prompts（用户控制）")
        print("=" * 64)

        for p in await c.list_prompts():
            print(f"  模板 {p.name} —— {p.description}")
            for arg in (p.arguments or []):
                req = "必填" if arg.required else "可选"
                print(f"      - {arg.name}（{req}）：{arg.description}")

        print()
        msgs = await c.get_prompt("research_report", {"topic": "大模型 Agent 架构"})
        for m in msgs.messages:
            print(f"  [{m.role}] {m.content.text}")

        print()
        print("=" * 64)
        print("真实检索测试")
        print("=" * 64)

        for q in ["ReAct 架构的原理", "多 Agent 协作如何结合记忆机制与 RAG"]:
            r = await c.call_tool("search_knowledge", {"query": q}, raise_on_error=False)
            print(f"\n  查询：{q}")
            print(f"  → count = {r.data['count']}")
            for i, d in enumerate(r.data["docs"], 1):
                print(f"     {i}. [{d['source']}] {d['text'][:45]}...")
            print(f"  note: {r.data['note'][:60]}...")
        r = await c.call_tool("search_knowledge", {"query": "AI Agent"}, raise_on_error=False)
        print(f"\n  查询：AI Agent")
        print(f"  → count = {r.data['count']}")
        print(f"  → note  = {r.data['note']}")

        comp = "多 Agent 协作如何结合记忆机制与 RAG"
        r1 = await c.call_tool("search_knowledge",
                               {"query": comp, "smart_query": False}, raise_on_error=False)
        r2 = await c.call_tool("search_knowledge",
                               {"query": comp, "smart_query": True}, raise_on_error=False)
        print()
        print("=" * 64)
        print("smart_query 对比（同一个复合问题）")
        print("=" * 64)
        print(f"  关闭（默认）→ {r1.data['count']} 段")
        print(f"  开启        → {r2.data['count']} 段")

        r = await c.call_tool("list_knowledge_sources", {}, raise_on_error=False)
        print()
        print("=" * 64)
        print("list_knowledge_sources 的返回结构")
        print("=" * 64)
        print(f"  data       : {r.data}")
        print(f"  structured : {r.structured_content}")


        print()
        try:
            bad = await c.read_resource("data://kb/chunk/99")
            print(f"  越界：居然读到了 → {bad[0].text}")
        except Exception as e:
            print(f"  越界：✅ 正确报错 → {type(e).__name__}: {e}")


async def part2():
    """第二部分：把 MCP Server 接进 LangChain Agent"""
    from langchain.agents import create_agent
    from langchain.mcp import MCPAdapter
    from langchain_core.messages import HumanMessage

    print()
    print("=" * 64)
    print("第二部分：接进 Agent")
    print("=" * 64)

    # ① 内存内连接 MCP Server
    async with MCPAdapter(mcp_server.mcp) as adapter:
        # ② 把 MCP 工具转成 LangChain 工具
        tools = await adapter.list_tools()
        print(f"\n拿到 {len(tools)} 个工具：{[t.name for t in tools]}")

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
        print(f"\n用户：{question}\n")

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

if __name__ == "__main__":
    asyncio.run(part1())
    asyncio.run(part2())



