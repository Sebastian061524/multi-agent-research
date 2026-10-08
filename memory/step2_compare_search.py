"""对比：关键词兜底 vs 语义检索（用真实的 BGE）

结论预期：
    · 关键词兜底：字面不重合就查不到（"智能体评估" vs "Agent 评测"）
    · 语义检索：  能跨过措辞差异，按意思匹配 ✅
"""
import os

# ⚠️ 必须在 import sentence_transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像（兜底）
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # 少一条无害警告

import sys
import tempfile
from pathlib import Path

# 让 memory/ 里的模块能被 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentence_transformers import SentenceTransformer

from memory_store import MemoryStore

# ---- 1. 准备一个干净的库 ----
db = Path(tempfile.gettempdir()) / "mem_compare.db"
if db.exists():
    db.unlink()

# ---- 2. 加载 BGE（和你调研项目用的是同一个模型）----
print("加载嵌入模型...")
model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu",
    local_files_only=True,          # ★ 只用本地缓存，跳过网络检查
)

def embed(texts):
    """按 MemoryStore 要求的签名包装：返回【已归一化】的向量"""
    return model.encode(texts, normalize_embeddings=True)

# ---- 3. 建两个 store：一个兜底、一个语义（共用同一个库）----
fallback = MemoryStore(db)                    # 不传 embed
semantic = MemoryStore(db, embed=embed)       # 传 embed

DATA = [
    ("demo1", "Agent 的评测方法"),
    ("demo2", "多 Agent 协作如何结合记忆机制与 RAG"),
    ("demo3", "ReAct 架构的原理"),
    ("demo4", "Plan-and-Execute 架构"),
    ("demo5", "如何给 LangGraph 加 checkpointer 做断点续跑"),
]
for tid, topic in DATA:
    semantic.save_session(tid, topic, [], "")

# ---- 4. 对比 ----
QUERIES = [
    ("智能体评估",            "字面完全不同，但意思一样"),
    ("记忆怎么存",            "口语化，和记录的措辞不一样"),
    ("怎么让流程中途暂停再恢复", "口语化，对应 checkpointer 那条"),
    ("完全无关的量子计算",     "应该都查不到"),
]
print()
print("=" * 78)
print(f"{'查询':<22}{'方式':<8}{'结果'}")
print("=" * 78)

for q, note in QUERIES:
    print(f"{q:<22}（{note}）")
    for label, store, thr in [("关键词", fallback, 0.35), ("语义", semantic, 0.4)]:
        hits = store.search_sessions(q, top_k=2, min_score=thr)
        if hits:
            txt = "|".join(f"[{h['score']} {h['topic'][:26]}]" for h in hits)
        else:
            txt = "（无命中）"
        print(f"{'':<22}{label:<8}{txt}")
    print()
print()
print("=" * 78)
print("诊断：所有候选的真实分数（不过滤，按分数排序）")
print("=" * 78)

for q, note in QUERIES:
    print(f"\n查询 {q!r}  —— {note}")
    hits = semantic.search_sessions(q, top_k=5, min_score=0.0)   # ← 阈值设 0，全都要
    if not hits:
        print("   （库里没数据）")
        continue
    for h in hits:
        bar = "█" * max(1, int(h["score"] * 50))     # 用方块可视化成柱状图
        print(f"   {h['score']:.4f}  {bar}  {h['topic'][:32]}")