"""记忆层：会话摘要 + 用户偏好

模块：
    memory_store.py          —— 存储与检索（SQLite + BGE 语义检索）
    step1_checkpointer.py    —— 实验：checkpointer 的四种行为
    step2_compare_search.py  —— 实验：关键词兜底 vs 语义检索

为什么要有这个 __init__.py：
    有这个文件，memory/ 才算一个【正规包】，
    项目根目录下的 main.py 才能写 from memory.memory_store import MemoryStore
    —— 这种写法任何 IDE 都能静态解析，不需要额外配置。
"""
