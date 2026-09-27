import os
import glob

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "research_docs")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)

# 加载文档
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
print(f"已索引 {len(chunks)} 个段落")


def search(query, top_k=3):
    """检索函数：给问题，返回最相关的段落"""
    qv = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ qv
    idx = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in idx]

def plan(topic):
    """① 规划者 Agent：把主题拆成子问题"""
    prompt = f"""你是一名资深研究规划专家。
用户的调研主题是：{topic}

请把这个主题拆解成 3 个具体、可独立调研的子问题。

要求：
- 只输出 3 行，每行一个子问题
- 不要编号，不要解释，不要任何其他内容
"""
    resp = llm.invoke(prompt)
    questions = []
    for line in resp.content.strip().split("\n"):
        q = line.strip().lstrip("0123456789.、）- ").strip()
        if q:
            questions.append(q)
    return questions[:3]

def research(questions):
    """调研者agent 对每个子问题检索资料"""
    findings = []
    for q in questions:
        docs = search(q, top_k=3)
        # 把命中的段落拼成一段文本 并标注来源
        materials = "\n\n".join(f"[{d['source']}] {d['text']}" for d in docs)
        findings.append({"question": q, "materials": materials})
    return findings

def write(topic, findings):
    """③ 撰写者 Agent：整合资料写成报告"""
    # 把所有子问题的资料拼成一大段
    materials = "\n\n".join(
        f"### 子问题：{f['question']}\n{f['materials']}"
        for f in findings
    )

    prompt = f"""你是一名专业研究员。请根据下面的资料，撰写一份关于「{topic}」的调研报告。

要求：
1. 使用 Markdown 格式
2. 结构包含三部分：## 概述、## 分点论述、## 结论
3. **只使用提供的资料**，不要编造资料中没有的信息
4. 如果某个子问题的资料不足，在对应位置明确写出「资料未涵盖此部分」

【资料】
{materials}
"""

    print("\n【③ 撰写者】正在生成报告...")
    resp = llm.invoke(prompt)
    return resp.content

if __name__ == "__main__":
    topic = input("请输入调研主题：")

    print("\n【① 规划者】拆解主题...")
    questions = plan(topic)
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")

    print("\n【② 调研者】逐个检索资料...")
    findings = research(questions)
    for f in findings:
        print(f"   ✓ {f['question'][:30]}...")

    # ③ 撰写
    report = write(topic, findings)

    print("\n" + "=" * 60)
    print("📄 调研报告")
    print("=" * 60)
    print(report)

    # 保存文件
    with open(os.path.join(BASE_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {topic}\n\n{report}")
    print("\n✅ 报告已保存到 report.md")