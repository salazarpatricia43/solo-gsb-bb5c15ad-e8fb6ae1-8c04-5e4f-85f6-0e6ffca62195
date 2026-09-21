"""酶组合规划（enzyme panel plan）：枚举候选酶的非空子集，复用酶切计算，
寻找恰好产生目标片段数的最少酶集合。

枚举顺序即择优规则
------------------
候选酶先按 ``(name, pattern, cut_top, cut_bottom)`` 稳定排序，随后子集按
规模从小到大枚举；同一规模内 ``itertools.combinations`` 按该顺序产生组合，
等价于按"酶名序列"的字典序枚举。因此第一个命中目标片段数的组合天然满足：

1. 所用酶数最少；
2. 酶数相同时，酶名序列字典序最小（与候选传入顺序无关）。

候选至多 8 种，非空子集总数不超过 2**8 - 1 = 255，直接全量枚举即可。
规划过程只在内存中调用 :func:`app.digest.run_digest`，不写库、不改原记录。
"""

from __future__ import annotations

from itertools import combinations

from .digest import run_digest


def _enzyme_key(enzyme) -> tuple:
    return (enzyme.name, enzyme.pattern, enzyme.cut_top, enzyme.cut_bottom)


def plan_enzyme_panel(sequence: str, topology: str, enzymes, target_fragments: int):
    """枚举候选酶组合，返回达到 ``target_fragments`` 个片段的最优组合。

    返回 ``{"enzymes", "enzyme_count", "cuts", "fragment_count"}``；
    任何非空组合都无法恰好产生目标片段数时返回 ``None``（路由以 422 拒绝）。
    """
    ordered = sorted(enzymes, key=_enzyme_key)
    for size in range(1, len(ordered) + 1):
        for combo in combinations(ordered, size):
            result = run_digest(sequence, topology, list(combo))
            fragment_count = len(result["fragments"])
            if fragment_count == target_fragments:
                return {
                    "enzymes": [
                        {
                            "name": enzyme.name,
                            "pattern": enzyme.pattern,
                            "cut_top": enzyme.cut_top,
                            "cut_bottom": enzyme.cut_bottom,
                        }
                        for enzyme in combo
                    ],
                    "enzyme_count": size,
                    "cuts": result["cuts"],
                    "fragment_count": fragment_count,
                }
    return None
