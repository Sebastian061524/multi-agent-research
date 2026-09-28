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