"""检索层评测：用有标准答案的测试集，量化并对比检索质量

为什么要有这个：
  单点验证会骗人 —— 这次开发中，只看一个 case 就以为修好了，
  结果另一个 case 反而变差。只有评测集能系统性地发现回归。
"""
from steps import step9_rerank as m9
from steps import step10_decompose as m10

# ==================== 评测集 ====================
# 每个用例：一个问题 + 「应该被检索到」的段落标题（标准答案）
CASES = [
    # ---------- 单一概念（期望命中 1 段）----------
    {"q": "ReAct 的推理-行动-观察循环是怎么运作的", "expect": ["ReAct 架构"]},
    {"q": "Plan-and-Execute 为什么需要配合重规划机制", "expect": ["Plan-and-Execute 架构"]},
    {"q": "多 Agent 协作有哪几种常见模式", "expect": ["多 Agent 协作"]},
    {"q": "RAG 的典型流程包含哪些步骤", "expect": ["RAG 检索增强生成"]},
    {"q": "Agent 的记忆分为哪几类", "expect": ["Agent 的记忆机制"]},
    {"q": "怎么追踪 Agent 每一步的执行情况", "expect": ["Agent 的可观测性"]},
    {"q": "怎么构建 Agent 的测试集来量化效果", "expect": ["Agent 的评测方法"]},

    # ---------- 双概念（期望命中 2 段）----------
    {"q": "ReAct 和 Plan-and-Execute 的核心区别是什么",
     "expect": ["ReAct 架构", "Plan-and-Execute 架构"]},
    {"q": "记忆机制和可观测性分别在 Agent 里起什么作用",
     "expect": ["Agent 的记忆机制", "Agent 的可观测性"]},

    # ---------- 复合问题（期望命中 3 段）---------- ← 这次踩坑的场景
    {"q": "多 Agent 协作架构如何结合记忆机制与 RAG 检索增强生成",
     "expect": ["多 Agent 协作", "Agent 的记忆机制", "RAG 检索增强生成"]},
    {"q": "ReAct、多 Agent 协作、RAG 三者如何配合完成复杂任务",
     "expect": ["ReAct 架构", "多 Agent 协作", "RAG 检索增强生成"]},

    # ---------- 资料库答不了的问题（期望：不硬凑）----------
    {"q": "AutoGen 框架的 Agent 通信协议是怎么设计的", "expect": []},
    {"q": "医疗行业如何落地大模型 Agent", "expect": []},
]

def title_of(doc):
    """取段落的第一行作为标题"""
    return doc["text"].split("\n")[0].strip()

def evaluate(module, config_name, **kwargs):
    """跑一遍评测集 返回统计结果"""
    total_expected = 0
    total_hit = 0
    total_noise = 0
    passed = 0
    details = []

    for case in CASES:
        q = case["q"]
        expect = set(case["expect"])

        docs = module.search(q, **kwargs)
        got = [title_of(d) for d in docs]
        got_set = set(got)

        hit = expect & got_set          # 命中：期望 ∩ 实际
        missing = expect - got_set      # 漏掉：期望有、实际没有
        noise = got_set - expect        # 噪音：实际有、期望没有

        total_expected += len(expect)
        total_hit += len(hit)
        total_noise += len(noise)

        ok = len(missing) == 0
        if ok:
            passed += 1
        details.append((ok, q, expect, got, missing, noise))

    # ⚠️ 下面这些必须在 for 循环【外面】（缩进 4 格）
    n = len(CASES)
    return {
        "name": config_name,
        "pass_rate": passed / n * 100,
        "passed": passed,
        "total": n,
        "recall": (total_hit / total_expected * 100) if total_expected else 100.0,
        "noise": total_noise,
        "details": details,
    }

if __name__ == "__main__":
    # ==================== 对比不同配置 ====================
    CONFIGS = [
        ("① 旧版：固定3段+0.1", m9, {"max_k": 3, "min_score": 0.1}),
        ("② 自适应+0.3（无分解）", m10, {"use_decompose": False}),
        ("③ 自适应+0.3（有分解）", m10, {}),
    ]

    results = []
    for name,module, kwargs in CONFIGS:
        results.append(evaluate(module, name, **kwargs))

    # ---------- 打印每个配置的详细结果 ----------
    for r in results:
        print("\n" + "=" * 72)
        print(f"【{r['name']}】")
        print("=" * 72)
        for ok, q, expect, got, missing, noise in r["details"]:
            mark = "✅" if ok else "❌"
            print(f"\n{mark} {q}")
            print(f"    期望 {len(expect)} 段 → 检索到 {len(got)} 段")
            if missing:
                print(f"    ❌ 漏掉：{sorted(missing)}")
            if noise:
                print(f"    ⚠️  噪音：{sorted(noise)}")

    # ---------- 汇总对比 ----------
    print("\n" + "=" * 72)
    print("【汇总对比】")
    print("=" * 72)
    print(f"{'配置':<26}{'用例通过':<12}{'段落召回':<12}{'噪音数'}")
    print("-" * 72)
    for r in results:
        print(f"{r['name']:<24}{r['passed']}/{r['total']:<10}"
              f"{r['recall']:.1f}%{'':<6}{r['noise']}")
