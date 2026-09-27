"""验证：两阶段检索（粗筛 + 精排）的效果对比"""
import os
import glob

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer, CrossEncoder

# ===== 加载两个模型 =====
print("加载嵌入模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

print("加载重排序模型...")
reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)
print("模型就绪\n")

# ===== 定位资料库（基于脚本位置，不依赖当前工作目录）=====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "research_docs")

# ===== 加载文档 =====
chunks = []
for pattern in [f"{DOCS_DIR}/**/*.txt", f"{DOCS_DIR}/**/*.md"]:
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > 20:
                chunks.append({"source": os.path.basename(path), "text": para})

if not chunks:
    raise SystemExit(f"❌ 在 {DOCS_DIR} 里没有找到可用资料（需要至少一段超过 20 字的文本）")

embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
print(f"已索引 {len(chunks)} 个段落\n")

# ===== 要验证的问题 =====
question = "Agent 的记忆机制与 RAG 检索增强生成如何协同支持 Agent 的长期与短期信息处理"

print("=" * 60)
print(f"问题：{question}")
print("=" * 60)

# ---------- 阶段 1：向量检索（粗筛）----------
qv = model.encode(question, normalize_embeddings=True)
vec_scores = embeddings @ qv
vec_rank = vec_scores.argsort()[::-1]

print("\n【阶段 1】向量检索排名：")
for rank, i in enumerate(vec_rank, 1):
    title = chunks[i]["text"].split("\n")[0][:28]
    mark = "  ★ top3" if rank <= 3 else ""
    print(f"  {rank}. [{vec_scores[i]:.3f}] {title}{mark}")

# ---------- 阶段 2：重排序（精排）----------
pairs = [[question, c["text"]] for c in chunks]     # 问题 × 每个段落 配对
rr_scores = reranker.predict(pairs)                  # 一次性打分
rr_rank = rr_scores.argsort()[::-1]

print("\n【阶段 2】重排序后排名：")
for rank, i in enumerate(rr_rank, 1):
    title = chunks[i]["text"].split("\n")[0][:28]
    mark = "  ★ top3" if rank <= 3 else ""
    print(f"  {rank}. [{rr_scores[i]:.3f}] {title}{mark}")