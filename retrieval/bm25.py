"""混合检索的三个零件：中文 token 化、BM25、RRF 融合。

⚠️ 这个模块【只放可复用的东西，不跑任何实验】——
   否则别人一 import 它，实验就跟着跑起来了（hybrid_lab 一开始就犯了这个错）。

用法：
    from retrieval.bm25 import tokenize, BM25, rrf
"""
import math
import re
from collections import Counter

# ==================== 停用词 ====================

# 为什么需要它（实测发现）：
#   查询「Agent 的记忆机制和 RAG 有什么不同」里的「有什么不同」，
#   被切成 '有' '什' '有什' '么' '什么' '不' '么不' '同' '不同' 一堆 token。
#   而它们命中了 chunk[0] 正文里的「模型先思考当前应该做【什么】」——
#   **词同，义完全不同**（一个是在提问，一个是在陈述）。
#   结果 chunk[0]（讲 ReAct 的，和查询无关）拿到 5.338 分，其中 4.884 分
#   （91.5%）来自「什」「什么」「么」——把真正该排前面的段落挤了下去。
#
#   ⇒ 「应该做什么」是陈述句里的词，「有什么不同」是疑问句里的词。
#     中文 BM25 分不开这两者，只能靠停用词表把功能词摘掉。
#
# ⚠️ 这张表本身是个【设计选择】，不是天经地义：
#   删多了会伤召回（把内容词也删掉），删少了噪音还在。
#   这里刻意保守 —— 只删「单独出现时没有实义」的字。
STOPWORDS = frozenset(
    "的了着过地得吗呢吧啊"      # 结构助词 / 语气词
    "和与及或跟对把被从向"      # 连词 / 介词
    "这那个些我你他们"          # 代词
    "是有在"                    # 判断 / 存在
    "什么怎如何哪谁几"          # 疑问词（语料是陈述句，这些几乎只出现在查询里）
    "不同区别比较样"            # 比较 / 差异（同理）
    "就都也还而但为"            # 其他高频虚词
)


# ==================== 中文 token 化 ====================

# 英文/数字连续串整体保留；单个汉字单独取出
_UNIT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.]*|[\u4e00-\u9fff]")


def tokenize(text: str, stopwords: frozenset = frozenset()) -> list[str]:
    """切 token（零依赖，不用装 jieba）。

    规则：
        英文 / 数字连续串 → 保留【整词】
            'ReAct' → ['react']
        单个汉字 → 先取出单字，再把【相邻两个汉字】合成 bigram
            '检索增强' → ['检', '索', '增', '强', '检索', '索增', '增强']

    stopwords 的过滤规则（关键）：
        · 命中停用词的字【不单独成为 token】
        · 而且【只要有一边是停用词，这个 bigram 也不生成】
          —— 否则「有什么不同」里的「什么」还是会冒出来。
        例：stopwords 含 {有, 什, 么, 不, 同} 时，
            '有什么不同' → []   （整串都是虚词，一个 token 都不留）
            '记忆机制'   → ['记', '忆', '记忆', '机', '制', '机制']   （不受影响）

    ⭐ 顺带说明：**「什么算一个 token」是可以自己设计的**，不是天经地义的。
    """
    units = _UNIT_RE.findall(text.lower())
    tokens = []
    for i, u in enumerate(units):
        if u in stopwords:
            continue
        tokens.append(u)
        nxt = units[i + 1] if i + 1 < len(units) else None
        if (nxt is not None and len(u) == 1 and len(nxt) == 1
                and nxt not in stopwords):
            tokens.append(u + nxt)
    return tokens


# ==================== 极简 BM25 ====================

class BM25:
    """标准 BM25。分数 = 每个查询 token 的贡献之和。

    三个部分：
        IDF  ：一个词越罕见，权重越高（跨文档统计）
        TF   ：一个词在本段出现越多分越高，但会【饱和】（k1 控制）
        长度归一化：长段落天然容易命中更多词，所以要打折（b 控制）

    为什么能解释：分数是【可加的】—— 能逐个 token 拆开看谁贡献了多少。
                 向量相似度做不到这一点。
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75,
                 stopwords: frozenset = frozenset()):
        self.k1, self.b = k1, b
        self.stopwords = stopwords
        self.n = len(docs)
        self.tokens = [tokenize(d, stopwords) for d in docs]
        self.lens = [len(t) for t in self.tokens]
        self.avg_len = sum(self.lens) / self.n if self.n else 1.0
        self.tf = [Counter(t) for t in self.tokens]

        df = Counter()
        for tf in self.tf:
            for w in tf:
                df[w] += 1
        # IDF：带 0.5 平滑（Lucene 风格），保证不为负
        self.idf = {
            w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def term_score(self, doc_id: int, word: str) -> float:
        """单个 token 对某个段落的贡献（0 表示这个词没出现在该段）"""
        f = self.tf[doc_id].get(word, 0)
        if f == 0 or word not in self.idf:
            return 0.0
        denom = f + self.k1 * (1 - self.b + self.b * self.lens[doc_id] / self.avg_len)
        return self.idf[word] * f * (self.k1 + 1) / denom

    def scores(self, query: str) -> list[float]:
        """每个段落的 BM25 总分"""
        q_tokens = tokenize(query, self.stopwords)
        return [
            sum(self.term_score(i, w) for w in q_tokens) for i in range(self.n)
        ]

    def ranking(self, query: str) -> list[int]:
        """按 BM25 分从高到低给出段落排名"""
        s = self.scores(query)
        return [int(i) for i in sorted(range(self.n), key=lambda i: s[i], reverse=True)]


# ==================== RRF 融合 ====================

def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion：分数不可比时，改用【排名】。

    为什么不能直接相加分数：
        向量的余弦相似度在 0~1，BM25 分数没有上界（可以到 30+）。
        直接相加 = 让 BM25 单方面决定结果。

    公式：score(doc) = Σ 1 / (k + 名次)
        k 取 60（原论文默认）—— 它压平「第一名」的领先幅度，
        让两路的第 2、第 3 名也有分量。

    ⚠️ RRF 是【平等对待两路】的，它不知道哪一路更准。
       如果一路明显更差，它的坏排名会稀释好排名。
    """
    total: Counter = Counter()
    for ranks in rankings:
        for pos, doc_id in enumerate(ranks, start=1):
            total[doc_id] += 1 / (k + pos)
    return [d for d, _ in total.most_common()]
