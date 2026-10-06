"""把调研项目的资料检索能力封装成 MCP Server"""
import json
import logging
import sys
from pathlib import Path

import anyio
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

# 本文件在 mcp/ 子目录里，而 main.py 在上一级；把项目根目录加进 sys.path 才能 import main
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import main as research

# ⚠️ MCP 的 logging 能力已在 2026-07-28 弃用（SEP-2577）。
#    官方推荐：日志走 stderr（stdio 传输）+ OpenTelemetry（结构化可观测）。
#    注意：stdio 模式下 stdout 是【协议通道】，日志绝不能写 stdout。
#
#    同时写一份到文件：stdio 客户端不一定把 stderr 显示出来，
#    写文件才能可靠地验证「客户端真的在用新代码」。
LOG_FILE = Path(__file__).parent / "mcp_server.log"

logging.basicConfig(
    level=logging.WARNING,          # 全局只要 WARNING 以上，避免带出第三方库的噪音
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("kb-server")
logger.setLevel(logging.INFO)       # 只有本项目自己的 logger 输出 INFO

mcp = FastMCP("Knowledge Base Server")

# ==================== 输出模型（Pydantic）====================

class Doc(BaseModel):
    """知识库里的一个资料片段。"""
    source: str = Field(description="资料文件名")
    text: str = Field(description="资料原文片段")


class SearchResult(BaseModel):
    """检索结果。"""
    count: int = Field(description="命中的段数")
    docs: list[Doc] = Field(description="命中的资料列表（已按相关性排序）")
    note: str = Field(description="附加说明：为什么是这个结果，或者为什么没有结果")

@mcp.tool
def ping() -> str:
    """测试用：返回 pong（可用于连通性检查）。"""
    logger.info("ping 被调用")
    return "pong"

@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def search_knowledge(
    ctx: Context,
    query: str,
    max_results: int = 5,
    smart_query: bool = False,
) -> SearchResult:
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
    # 日志走 stderr（MCP logging 能力已弃用；stdio 下 stdout 是协议通道，不能污染）
    logger.info(f"检索：{query}  （transport={ctx.transport}）")

    # 进度上报（本能力未弃用）——让客户端知道慢操作正在做什么
    await ctx.report_progress(1, 2, "向量检索 + 重排序中")

    # research.search 是同步阻塞的（向量编码 + 重排序），放线程池避免卡住事件循环
    docs = await anyio.to_thread.run_sync(
        lambda: research.search(query, max_k=max_results, use_decompose=smart_query)
    )

    await ctx.report_progress(2, 2, "完成")

    if docs:
        logger.info(f"命中 {len(docs)} 段")
        return SearchResult(
            count=len(docs),
            docs=[Doc(source=d["source"], text=d["text"]) for d in docs],
            note="已按相关性排序。请只使用这些片段作答。",
        )

    logger.warning("没有命中任何片段（全部低于相关性门槛）")
    return SearchResult(
        count=0,
        docs=[],
        note=(
            "知识库里没有段落达到相关性门槛。可能原因："
            "① 这个方向资料里没有；"
            "② 你的问法与资料用词差异较大。"
            "建议：先调用 list_knowledge_sources 看知识库涵盖哪些主题；"
            "或换一种更贴近资料的说法重试。"
            "如果确实没有，请如实告知用户「资料未涵盖」。"
        ),
    )


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def list_knowledge_sources() -> list[str]:
    """列出知识库里都有哪几篇资料（只有文件名，不含内容）。

    用途：想先了解「这个知识库能回答哪些方面的问题」时调用。
    注意：本工具只返回文件名。想看每篇涵盖哪些主题，
          请读取资源 data://kb/toc。
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

@mcp.resource("data://kb/toc")
def get_kb_toc() -> str:
    """知识库目录：每篇资料包含哪些片段（下标 + 标题）。"""
    toc = {}
    for i, c in enumerate(research.chunks):
        title = c["text"].splitlines()[0].strip()
        toc.setdefault(c["source"], []).append({"index": i, "title": title})
    return json.dumps(toc, ensure_ascii=False, indent=2)

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
    logger.info("以 HTTP 方式启动")
    mcp.run(transport="http", port=8001)