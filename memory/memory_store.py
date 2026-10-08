"""记忆的存储与检索（数据层，不依赖 LLM）

两张表：
    sessions     —— 会话摘要（只记【事实】：调研过什么）
    preferences  —— 用户偏好（显式写入，不猜）

设计要点：
    · 只用 SQLite，不引入向量数据库（几十~几百条，NumPy 足够）
    · embed 函数可注入：
          传了 → 语义检索（用 main.model 的 BGE）
          不传 → 关键词兜底（便于单测，也能在无模型环境降级）
    · 只记事实，不记结论（结论可能错、会过时）

设计文档见本目录下的对话记录 / docs。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent / "memory.db"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if "sub_questions" in d:
        try:
            d["sub_questions"] = json.loads(d["sub_questions"] or "[]")
        except json.JSONDecodeError:
            d["sub_questions"] = []
    return d


def _keyword_scores(query: str, rows: list[dict]) -> list[tuple[float, dict]]:
    """关键词兜底：按【字符集合】重合度打分（0~1）

    粗糙但够用——只在没传 embed 函数时启用（单测、无模型环境）。
    """
    q = set(query)
    out = []
    for r in rows:
        t = set(r["topic"] or "")
        out.append((len(q & t) / max(len(q), 1), r))
    return out


class MemoryStore:
    """记忆库：会话摘要 + 用户偏好"""

    def __init__(self, db_path=DEFAULT_DB, embed=None):
        """
        参数：
            db_path: SQLite 文件路径
            embed:   编码函数，签名 (list[str]) -> np.ndarray
                     应当返回【已归一化】的向量（这样点积就等于余弦相似度）
                     不传则检索退化为关键词匹配
        """
        self.db_path = Path(db_path)
        self.embed = embed
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    # ==================== 建表 ====================

    def _init_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id     TEXT,
                topic         TEXT NOT NULL,
                sub_questions TEXT,
                report_head   TEXT,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ==================== 会话摘要 ====================

    def save_session(self, thread_id: str, topic: str,
                     sub_questions=None, report_head: str = "") -> int:
        """存一条会话摘要（跑完调研后调用）。

        ⚠️ 这里只存【事实】：调研了什么、拆了哪些子问题、报告怎么开的。
           不存"结论对不对"这类判断——那是可能出错的。
        """
        cur = self.conn.execute(
            "INSERT INTO sessions (thread_id, topic, sub_questions, report_head, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                thread_id,
                topic,
                json.dumps(sub_questions or [], ensure_ascii=False),
                (report_head or "")[:300],
                _now(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_sessions(self, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search_sessions(self, query: str, top_k: int = 3,
                        min_score: float = 0.5) -> list[dict]:
        """找和 query 相关的历史会话。

        返回每项会多一个 score 字段。同主题只保留最相关的一条
        （避免"同一个主题列 5 遍"）。
        """
        rows = self.list_sessions()
        if not rows:
            return []

        if self.embed is None:
            scored = _keyword_scores(query, rows)
        else:
            import numpy as np

            qv = np.asarray(self.embed([query]))[0]
            mat = np.asarray(self.embed([r["topic"] or "" for r in rows]))
            scored = [(float(mat[i] @ qv), r) for i, r in enumerate(rows)]

        scored.sort(key=lambda x: -x[0])

        out, seen = [], set()
        for score, r in scored:
            if score < min_score:
                break
            key = (r["topic"] or "").strip()
            if key in seen:
                continue
            seen.add(key)
            out.append({**r, "score": round(score, 4)})
            if len(out) >= top_k:
                break
        return out

    # ==================== 用户偏好 ====================

    def add_preference(self, text: str) -> bool:
        """加一条偏好。已存在则返回 False（不重复记）。

        ⚠️ 偏好【显式写入】，不从对话里猜——猜错了很烦人。
        """
        text = (text or "").strip()
        if not text:
            return False
        try:
            self.conn.execute(
                "INSERT INTO preferences (text, created_at) VALUES (?, ?)",
                (text, _now()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:      # UNIQUE 冲突
            return False

    def list_preferences(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT text FROM preferences ORDER BY id"
        ).fetchall()
        return [r["text"] for r in rows]

    def delete_preference(self, text: str) -> bool:
        cur = self.conn.execute("DELETE FROM preferences WHERE text = ?", (text,))
        self.conn.commit()
        return cur.rowcount > 0

    # ==================== 杂项 ====================

    def stats(self) -> dict:
        s = self.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        p = self.conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
        return {"sessions": s, "preferences": p}

    def close(self):
        self.conn.close()


# ==================== 自测（用假 embed，不需要模型，秒级完成）====================

if __name__ == "__main__":
    import tempfile

    db = Path(tempfile.gettempdir()) / "mem_selftest.db"
    if db.exists():
        db.unlink()

    print("=" * 66)
    print("① 写入三条会话摘要")
    print("=" * 66)
    store = MemoryStore(db)          # 不传 embed → 关键词兜底
    store.save_session("demo1", "Agent 的评测方法",
                       ["测试集构建", "评分方式"], "# Agent 评测调研报告\n\n## 概述...")
    store.save_session("demo2", "多 Agent 协作如何结合记忆机制与 RAG",
                       ["多 Agent 协作", "记忆机制", "RAG"], "# 多 Agent 协作...")
    store.save_session("demo3", "ReAct 架构的原理",
                       ["ReAct 循环", "与 CoT 的区别"], "# ReAct 架构...")
    for r in store.list_sessions():
        print(f"  [{r['id']}] {r['thread_id']:<8} {r['topic']}")
    print(f"  统计：{store.stats()}")

    print()
    print("=" * 66)
    print("② 检索（关键词兜底，没传 embed）")
    print("=" * 66)
    for q in ["Agent 评测", "记忆机制", "完全无关的量子计算"]:
        hits = store.search_sessions(q, top_k=2, min_score=0.35)
        print(f"  查询 {q!r}")
        if not hits:
            print("     → 无命中")
        for h in hits:
            print(f"     → [{h['score']}] {h['topic']}")

    print()
    print("=" * 66)
    print("③ 用户偏好：写入 / 查重 / 列出 / 删除")
    print("=" * 66)
    print(f"  加「报告要简洁，多用表格」     → {store.add_preference('报告要简洁，多用表格')}")
    print(f"  再加一次（应返回 False 去重） → {store.add_preference('报告要简洁，多用表格')}")
    print(f"  加「结论部分要有明确的取舍建议」→ {store.add_preference('结论部分要有明确的取舍建议')}")
    print(f"  当前偏好：{store.list_preferences()}")
    print(f"  删除第一条 → {store.delete_preference('报告要简洁，多用表格')}")
    print(f"  删除后：{store.list_preferences()}")

    print()
    print("=" * 66)
    print("④ 最终统计")
    print("=" * 66)
    print(f"  {store.stats()}")
    print(f"  数据库文件：{db}")

    store.close()
