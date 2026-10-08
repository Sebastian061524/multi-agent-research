"""记忆/断点数据库巡检工具

用途：
  1. 排掉「thread_id 里混进了不可见字符」这类隐形问题（用 repr 打印）
  2. 快速看清 checkpoints.db 和 memory.db 里到底存了什么

用法：python memory/inspect_db.py
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DB = ROOT / "checkpoints.db"
MEM_DB = ROOT / "memory" / "memory.db"


def show(title, db_path, queries):
    print("=" * 60)
    print(f"{title}  ({db_path})")
    print("=" * 60)
    if not db_path.exists():
        print("  ⚠️  文件不存在")
        return
    conn = sqlite3.connect(db_path)
    for label, sql in queries:
        print(f"\n-- {label}")
        try:
            rows = conn.execute(sql).fetchall()
        except sqlite3.Error as e:
            print(f"   ❌ {e}")
            continue
        if not rows:
            print("   （空）")
        for row in rows:
            print("   " + " | ".join(repr(v) for v in row))
    conn.close()


if __name__ == "__main__":
    show("断点库", CKPT_DB, [
        ("所有会话（repr 显示，能看出 BOM 之类的隐藏字符）",
         "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"),
    ])

    show("记忆库", MEM_DB, [
        ("会话摘要", "SELECT id, thread_id, topic, created_at FROM sessions ORDER BY id"),
        ("用户偏好", "SELECT id, text, created_at FROM preferences ORDER BY id"),
    ])
