import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
print("Model loaded")

import glob

DOCS_DIR = "research_docs"

chunks = []
for pattern in [f"{DOCS_DIR}/**/*.txt", f"{DOCS_DIR}/**/*.md"]:
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) > 20:
                chunks.append({"source": os.path.basename(path), "text": para})

print(f"加载了 {len(chunks)} 个段落")
for c in chunks:
    print("   -", c["text"][:35].replace("\n", " "), "...")

embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
print(f"\n向量化完成，形状：{embeddings.shape}")

def search(query, top_k=3):
    qv = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ qv
    idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in idx]

print("\n" + "=" * 55)
for q in ["ReAct 架构是什么？", "多 Agent 有哪些协作模式？"]:
    print(f"\n查询：{q}")
    for d in search(q):
        print(f"   [{d['source']}] {d['text'][:55]}...")