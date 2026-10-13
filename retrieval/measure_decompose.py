"""量 decompose 的成本，并第一次看清它到底把查询变成了什么。

为什么写这个：
    我们讨论「查询分解」讨论了整整三课，但【从来没看过它的输出】。
    不知道它把查询变成什么，就没法判断「值不值这个钱」，
    更没法讨论「怎么让它变便宜」。

量两件事：
    ① 耗时 —— 每次 decompose 要等多久（这是 MCP 服务里用户能感觉到的延迟）
    ② 输出 —— 它到底拆成了几个子查询、内容是什么

用法（项目根目录）：
    python -m retrieval.measure_decompose
"""
import os
import sys
import time

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import main as R
from retrieval.holdout_set import HOLDOUT

print(f"对留出集的 {len(HOLDOUT)} 条查询各跑一次 decompose\n")
print(f"{'耗时':>7}  {'拆出':>4}  查询 → 子查询")
print("-" * 90)

times = []
multi = 0
for item in HOLDOUT:
    t0 = time.perf_counter()
    subs = R.decompose(item["query"])
    dt = time.perf_counter() - t0
    times.append(dt)

    if len(subs) > 1:
        multi += 1

    print(f"{dt:6.2f}s  {len(subs):>4}  {item['query'][:34]}")
    for s in subs:
        print(f"{'':>13}      └─ {s}")
    print()

times.sort()
n = len(times)
print("=" * 90)
print(f"共 {n} 次调用")
print(f"  总耗时   {sum(times):.2f}s")
print(f"  平均     {sum(times) / n:.2f}s")
print(f"  中位数   {times[n // 2]:.2f}s")
print(f"  最快     {times[0]:.2f}s")
print(f"  最慢     {times[-1]:.2f}s")
print(f"\n  拆出多个子查询的：{multi}/{n} 条（{multi / n:.0%}）"
      f"  ← 这就是「文字规则」想预测的那件事")
