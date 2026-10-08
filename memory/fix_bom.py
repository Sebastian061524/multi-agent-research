"""清理 thread_id 里混入的 BOM 字符（\\ufeff）

背景：
  用 PowerShell 管道给 python 喂输入时，编码不由程序控制，
  第一行可能变成 '\\ufeffdemoA'。后果很隐蔽：
    - 同一个会话自己和自己能对上（续跑正常，看起来没问题）
    - 但任何「用干净 ID 去查」的脚本/工具都查不到
    - 记忆库里存下的 thread_id 也是脏的

用法：
  python memory/fix_bom.py            # 只预览，不改
  python memory/fix_bom.py --apply    # 真的改
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAD = "\ufeff"


def clean(value):
    return value.replace(BAD, "") if isinstance(value, str) else value


def fix_table(conn, table, column, apply):
    """把某一列里的 BOM 去掉；目标值已存在时跳过（避免主键冲突）"""
    rows = conn.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()
    dirty = [(v,) for (v,) in rows if clean(v) != v]
    if not dirty:
        print(f"  ✅ {table}.{column} 干净")
        return 0

    existing = {clean(v) for (v,) in rows}
    fixed_n = 0
    for (value,) in dirty:
        target = clean(value)
        print(f"  {table}.{column}: {value!r} → {target!r}")
        if apply:
            # 目标已存在 = 会有两条记录合并，风险高，跳过让人来判断
            clash = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (target,)
            ).fetchone()[0]
            if clash:
                print(f"     ⚠️  跳过：{target!r} 已存在 {clash} 行，需人工处理")
                continue
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (target, value)
            )
        fixed_n += 1
    return fixed_n


def main():
    apply = "--apply" in sys.argv
    targets = [
        (ROOT / "checkpoints.db", [("checkpoints", "thread_id"), ("writes", "thread_id")]),
        (ROOT / "memory" / "memory.db", [("sessions", "thread_id")]),
    ]

    total = 0
    for db_path, tables in targets:
        print(f"\n📂 {db_path.name}")
        if not db_path.exists():
            print("   ⚠️  文件不存在")
            continue
        conn = sqlite3.connect(db_path)
        for table, column in tables:
            total += fix_table(conn, table, column, apply)
        conn.commit()
        conn.close()

    print()
    if total == 0:
        print("✨ 没有需要清理的记录")
    elif apply:
        print(f"✅ 已清理 {total} 条记录")
    else:
        print(f"🔍 发现 {total} 条含 BOM 的记录（加 --apply 才会真的改）")


if __name__ == "__main__":
    main()
