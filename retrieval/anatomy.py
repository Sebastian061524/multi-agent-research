"""拆开检索机：看清每一级到底输出了什么。

复用 main.py 里已经建好的索引（chunks / embeddings），不重复加载模型。
运行方式（必须在项目根目录）：
    python -m retrieval.anatomy
"""
import main as R


print("=" * 70)
print(f"① 库里有什么（共 {len(R.chunks)} 段）")
print("=" * 70)
for i, c in enumerate(R.chunks):
    print(f"  [{i}] {c['text'][:44]}")

query = "ReAct 架构的优缺点"

print()
print("=" * 70)
print(f"② 一个查询走完全程：{query}")
print("=" * 70)

# 第 1 级：bi-encoder 打分
qv = R.model.encode(query, normalize_embeddings=True)
vec_scores = R.embeddings @ qv
order = vec_scores.argsort()[::-1]

print("\n  【第 1 级】向量相似度（bi-encoder）")
for i in order:
    print(f"     chunk[{i}]  {vec_scores[i]:+.4f}   {R.chunks[i]['text'][:26]}")

# 第 2 级：cross-encoder 精排
pairs = [[query, R.chunks[i]["text"]] for i in order]
rr = R.reranker.predict(pairs)

print("\n  【第 2 级】重排序分数（cross-encoder）")
for rank, i in enumerate(order):
    print(f"     chunk[{i}]  {rr[rank]:+.4f}   {R.chunks[i]['text'][:26]}")

# 第 3 级：最终输出
res = R._retrieve(query)
print(f"\n  【第 3 级】_retrieve 的最终输出（阈值 {R.RERANK_THRESHOLD}）")
for c in res:
    print(f"     {c['text'][:38]}")
print(f"     共 {len(res)} 段")

# 验证怀疑 1：粗筛到底有没有在筛
print()
print("=" * 70)
print("③ 如果是【纯精排】（完全不看向量分），结果会不一样吗？")
print("=" * 70)
all_rr = R.reranker.predict([[query, c["text"]] for c in R.chunks])
pure_idx = all_rr.argsort()[::-1]
pure = [R.chunks[i] for i in pure_idx
        if all_rr[i] >= R.RERANK_THRESHOLD][:5]
print(f"  _retrieve 取到：{[c['text'][:14] for c in res]}")
print(f"  纯精排   取到：{[c['text'][:14] for c in pure]}")
print(f"  两者完全相同：{[c['text'] for c in res] == [c['text'] for c in pure]}")

