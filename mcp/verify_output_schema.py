"""对比 dict 和 Pydantic 作为返回类型时，生成的 output schema 差异"""
import asyncio

from fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("schema-demo")


# ---------- 写法 A：裸 dict ----------
@mcp.tool
def search_dict(q: str) -> dict:
    """用裸 dict 返回。"""
    return {"count": 1, "docs": [], "note": "ok"}


# ---------- 写法 B：Pydantic 模型 ----------
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
def search_model(q: str) -> SearchResult:
    """用 Pydantic 模型返回。"""
    return SearchResult(count=1, docs=[], note="ok")


async def main():
    from fastmcp import Client
    async with Client(mcp) as c:
        for t in await c.list_tools():
            print("=" * 66)
            print(f"【{t.name}】")
            print("=" * 66)
            print(f"  output_schema = {t.output_schema}")
            print()


asyncio.run(main())