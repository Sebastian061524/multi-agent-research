"""mcp_server 的 stdio 入口

为什么单独一个文件：
    mcp_server.py        只负责【定义】（工具 / 资源 / 提示），它以 HTTP 方式启动
    本文件               只负责【启动方式】——以 stdio 方式，供桌面客户端（Cursor 等）使用

    同一个 Server 定义，可以有多个入口。定义与入口分离。

用法：
    由桌面客户端启动（它会自己调用这个脚本）
    或手动测试：python mcp_server_stdio.py
"""
import contextlib
import sys

# ⚠️ 关键：stdio 模式下 stdout 是【协议通道】（JSON-RPC），任何 print 都会污染它。
#    mcp_server 在 import 时会 import main，而 main 在模块级有 3 处 print：
#        "加载嵌入模型..." / "加载重排序模型..." / "已索引 N 个段落"
#    所以把 import 期间的 stdout 临时改道到 stderr（stderr 是日志通道，随便打）。
with contextlib.redirect_stdout(sys.stderr):
    from mcp_server import logger, mcp


if __name__ == "__main__":
    # 启动日志：用来确认「客户端确实拉起了这个 Server，而且跑的是新代码」
    logger.info("以 stdio 方式启动（被客户端拉起）")
    # mcp.run() 的默认传输就是 stdio
    mcp.run()
