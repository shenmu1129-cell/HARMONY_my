"""Demo for the user-profile guided hyperedge pool fast channel.

Run:
    python examples/user_profile_hyperedge_demo.py

This demo is intentionally retrieval-only and does not call any LLM service.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypermem.profile_hyperedge_pool import UserProfileHyperedgePool


DEMO_FACTS = [
    "用户正在研究 LLM long-term conversational memory，重点关注 HyperMem、A-MEM、MemEvolve 和 LoCoMo。",
    "用户的长期目标是形成一个有 AAAI 竞争力的大模型记忆创新点。",
    "用户喜欢先理解论文原理，再分析创新性，最后生成可直接给 Codex 使用的实现 prompt。",
    "用户希望回答时像审稿人一样冷静分析，不要空泛鼓励，要指出实验风险和可行下一步。",
    "早期方案是动态层级 HyperMem，但实验显示 dynamic-only 没有超过 fixed_400 和 global_fact_only_800。",
    "后来的方案转向 verifier-guided adaptive memory control，用 evidence verifier 判断是否继续检索。",
    "最新想法是构建用户画像引导的动态超边池，作为长期记忆的个性化快速通道。",
    "超边池只维护高价值、常用、符合用户习惯和当前任务的 profile hyperedges。",
    "如果用户画像超边池证据不足，系统应该 fallback 到原始 HyperMem topic-episode-fact 路径或 global fact retrieval。",
    "强化学习或 reward regression 可以用于提升超边 utility，命中奖励更高，错误或过期画像会被降权。",
    "时间处理是长期记忆的重要创新点，需要区分 earlier、later、current state 和 temporal evolution。",
]


QUERIES = [
    "我现在这个 memory 论文的主线是什么？",
    "我通常希望你怎么分析论文创新？",
    "如果画像超边池没找到证据怎么办？",
    "为什么 dynamic-only 不是最终主线？",
    "强化学习在这个超边池里起什么作用？",
]


def main() -> None:
    pool = UserProfileHyperedgePool(user_id="demo_user")
    pool.build_from_texts(DEMO_FACTS, user_id="demo_user")

    print("=== User-Profile Hyperedge Pool ===")
    profile = pool.export_profile()
    print("edge_type_counts:", profile["edge_type_counts"])
    print("num_edges:", profile["num_edges"])
    print()

    for query in QUERIES:
        result = pool.retrieve_fast_channel(query, top_k_edges=3, max_tokens=120)
        print("=" * 80)
        print("Query:", query)
        print("Channel:", result.channel)
        print("Score:", round(result.score, 4), "Tokens:", result.tokens, "Sufficient:", result.sufficient)
        print("Matched hyperedges:")
        for edge in result.hyperedges:
            print(f"  - {edge.edge_id} [{edge.edge_type.value}] utility={edge.utility_score:.2f}: {edge.summary}")
        print("Evidence:")
        print(result.evidence_text())

    out_path = ROOT / "outputs" / "profile_hyperedge_demo_pool.json"
    pool.save(out_path)
    print("\nSaved demo pool to", out_path)


if __name__ == "__main__":
    main()
