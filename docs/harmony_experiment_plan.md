# HARMONY-Mem Experiment Plan

## Goal

Evaluate whether HARMONY-Mem improves the accuracy--cost trade-off through its retrieval policy and evidence design, rather than through a privileged embedding or reranking model.

The primary report contains three metrics only:

- `llm_acc`: answer correctness under a fixed reader/judge protocol.
- `retrieval_tokens`: evidence tokens passed to the answer model.
- `retrieval_latency_ms`: retrieval-only latency.

Retrieval recall, action distributions, and p50 latency remain diagnostics and support ablations; they are not main-table columns.

## Protocol Rules

1. For a table, every method uses the same data split, reader, judge, Qwen embedding endpoint, and reranker setting.
2. Category 5 and empty-gold handling are stated in every caption. They are never silently mixed into a score.
3. Answer-generation and judge latency are reported separately if measured; they are never included in `retrieval_latency_ms`.
4. `*-Lite` methods are labelled as unified-interface reproductions, not official-system reproductions.
5. Every LLM score records the reader model, prompt family, judge model, and judge prompt family.

## Experiment Matrix

### E1: Shared-Retriever Main Comparison

**Question:** Does HARMONY improve the accuracy--cost frontier when all methods share the same retrieval backbone?

- Methods: HARMONY-Mem, HyperMem-Flow, BM25-turn, QwenEmb-turn, Mem0-Lite, A-Mem-Lite, LightRAG-Lite, and HippoRAG-Lite.
- Retrieval stack: Qwen3-Embedding-4B; report one table with Qwen3-Reranker-4B enabled.
- Reader/judge: first use a single strict protocol for all methods; run the HyperMem-aligned protocol as a separate analysis, never in the same table.
- Split: fixed random seed, 30 train / 40 test for development; expand to at least 60 train / 80 test before paper claims.

**Output:** main table with `llm_acc`, `retrieval_tokens`, and `retrieval_latency_ms`; Pareto scatter plot of accuracy versus latency with token size or labels.

### E2: HARMONY Component Ablation

**Question:** Which component creates the gain?

| Variant | Removed or fixed component | Expected evidence |
| --- | --- | --- |
| HARMONY-Mem | none | adaptive reference |
| NoRL-FullRecall | router fixed to high-recall action | isolates action routing from broad retrieval |
| NoRL-RoleBalanced | router fixed to role-aware balanced action | tests a fixed low-cost policy |
| NoRole | all-global balanced retrieval | tests the value of soft role-conditioned actions |
| NoSourcePreserve | session-summary leaves replace source snippets | tests source provenance and answer grounding |
| NoHyperedge-FlatRRF | flat fused retrieval replaces hierarchy | tests hierarchical structure |

The current module-ablation runner produces the source-preservation and flat-RRF controls. The LoCoMo comparison runner produces the router and role controls. Both must be evaluated on the same split and reader/judge protocol before they appear in one ablation table.

**Output:** grouped bar chart with normalized `llm_acc`, token cost, and latency; an ablation table with absolute values.

### E3: Retriever Robustness

**Question:** Does HARMONY depend on a specific Qwen backbone?

Run HARMONY, NoRL-FullRecall, and NoHyperedge-FlatRRF under:

1. Qwen3-Embedding-4B + Qwen3-Reranker-4B.
2. Qwen3-Embedding-4B without reranking.
3. A non-Qwen embedding model, preferably BGE-M3 or NV-Embed-v2, with no reranker or with BGE-Reranker-v2-M3 if the serving setup permits.

The first two runs are the immediate partial-LoCoMo tests. The third run is required before claiming retriever-agnostic generality. This is stronger and cleaner than changing the main model merely to look different from HyperMem.

**Output:** slope chart or three-column robustness table showing HARMONY's delta over NoRL-FullRecall and NoHyperedge-FlatRRF under each backbone.

### E4: Budget and Scale Sensitivity

**Question:** Does the policy retain its advantage when memory or evidence budgets change?

- Sweep evidence budgets: compact, balanced, and recall configurations.
- Sweep test memory scale by retaining 25%, 50%, 75%, and 100% of each dialogue history, while preserving the answer-bearing rows where required by the benchmark protocol.
- Report category-level results for multi-hop, temporal, open-domain, and single-hop questions.

**Output:** line plot of `llm_acc` against retrieval tokens; separate line plot of latency against history scale. Category results remain a compact supplemental table.

### E5: Protocol Sensitivity

**Question:** How much of the score depends on answer/judge protocol rather than retrieval?

Hold the exact evidence trace fixed and compare:

1. Existing strict reader/judge protocol.
2. Existing answers with HyperMem-style judge.
3. HyperMem CoT prompt plus `gpt-4.1-mini` reader and `gpt-4o-mini` judge.

This is an evaluation analysis, not a method comparison. It should appear as a small protocol-sensitivity table or appendix figure.

## Figure Set

| Figure | Visual form | Decision it supports |
| --- | --- | --- |
| Fig. 1 | Accuracy--latency Pareto scatter, direct method labels, token annotation | whether HARMONY is efficient relative to baselines |
| Fig. 2 | Three-panel ablation bars: accuracy, tokens, latency | which HARMONY module matters |
| Fig. 3 | Budget/scale line plots | whether gains survive tighter budgets and longer histories |
| Fig. 4 | Protocol-sensitivity slope chart | why scores from different LoCoMo harnesses are not directly interchangeable |

Use tables for exact absolute values and figures for trade-offs, trends, and component deltas. Do not turn the three main metrics into three redundant bar charts for the primary comparison.

## Current Development Runs

The first partial run is E1 with 30 training questions, 40 test questions, Qwen3-Embedding-4B, Qwen3-Reranker-4B, and no LLM calls. It is a retrieval-cost smoke test only. After it completes, select the leading methods for a fixed-protocol LLM pass, then run E2 and the Qwen-reranker-off arm of E3 on the identical split.

## Partial Results on LoCoMo

These runs are development checks, not final paper claims. They use partial LoCoMo splits and set `--max-llm-judge 0`, so `llm_acc` is intentionally empty until the OpenAI key is available in the execution environment.

### E1: Shared Qwen Embedding + Qwen Reranker

Run directory: `outputs/paper_matrix_sharedqwen_train30_test40_rerank_clean_20260711_1430`

| method | n | llm_acc | retrieval_tokens | retrieval_latency_ms |
| --- | ---: | ---: | ---: | ---: |
| HARMONY-Mem | 40 |  | 545.9 | 1867.3 |
| QwenEmb-turn | 40 |  | 473.1 | 2303.4 |
| HyperMem-Flow | 40 |  | 1176.5 | 2876.3 |
| Mem0-Lite | 40 |  | 802.9 | 3567.4 |
| A-Mem-Lite | 40 |  | 797.7 | 3656.8 |
| LightRAG-Lite | 40 |  | 808.5 | 3474.8 |
| HippoRAG-Lite | 40 |  | 803.5 | 3638.5 |
| BM25-turn | 40 |  | 338.4 | 12.7 |

Interpretation: HARMONY is currently on a good retrieval-cost frontier against dense/graph-style baselines, but HyperMem-Flow and Mem0-Lite have stronger retrieval recall in this split. The next fixed-protocol LLM pass should decide whether HARMONY's lower evidence budget preserves answer accuracy.

### E2: Structural and Source-Preservation Ablation

Run directory: `outputs/paper_ablation_train30_test30_rerank_clean_20260711_1500`

| method | n | llm_acc | retrieval_tokens | retrieval_latency_ms |
| --- | ---: | ---: | ---: | ---: |
| Full-HG-RL-4A | 30 |  | 813.6 | 2478.3 |
| NoRL-Fixed-A3-recall | 30 |  | 917.8 | 2815.0 |
| NoRL-Fixed-A4-broad | 30 |  | 1051.7 | 3121.9 |
| NoHyperedge-FlatRRF | 30 |  | 901.2 | 2278.6 |
| NoSourcePreserve-SummaryLeaf | 30 |  | 1581.9 | 2478.3 |
| NoRL-Fixed-A2-balanced | 30 |  | 783.1 | 2215.3 |
| NoRL-Fixed-A1-compact | 30 |  | 504.6 | 1666.6 |
| QTypeRule-A1Temp-A3Multi | 30 |  | 504.6 | 1666.6 |

Interpretation: the old 4-action ablation shows that fixed recall/broad policies retrieve more answer evidence than the learned policy, at higher token and latency cost. It also shows that summary leaves inflate evidence tokens. This ablation should be kept separate from the latest HARMONY router table until both are evaluated on the same action space and LLM protocol.

### E3: Qwen Reranker Off

Run directory: `outputs/paper_router_norerank_train30_test40_clean_20260711_1520`

| method | n | llm_acc | retrieval_tokens | retrieval_latency_ms |
| --- | ---: | ---: | ---: | ---: |
| HARMONY-Mem | 40 |  | 458.8 | 1567.3 |
| HARMONY-NoRL-FullRecall | 40 |  | 754.7 | 2183.4 |
| HARMONY-NoRole | 40 |  | 598.9 | 1739.7 |

Interpretation: removing the Qwen reranker reduces latency and tokens, but HARMONY's retrieval recall drops sharply. The final experiment section therefore needs a reranker-off robustness table and at least one non-Qwen retriever arm before making a retriever-agnostic claim.

### LLM Accuracy Status

A 40-row top-method LLM pass was prepared with HARMONY-Mem, HyperMem-Flow, Mem0-Lite, and QwenEmb-turn over the first 10 test questions. It is currently blocked because neither the remote server nor the local `wwt310` process exposes `OPENAI_API_KEY`. The prepared trace is:

`local_results/paper_llm_top4_10q_4_1mini_20260711_1540/trace_top4_10q.jsonl`
