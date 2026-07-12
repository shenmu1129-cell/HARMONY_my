# HARMONY-Mem AAAI Experiment Protocol

## Method Under Evaluation

The paper-facing HARMONY-Mem is the stable single-query framework:

```text
source-preserving dialogue facts
  -> topic / episode / fact hierarchy
  -> BM25 + Qwen dense retrieval + RRF + Qwen reranking
  -> soft-role contextual-bandit action selection
  -> source-preserving evidence packing
  -> GPT reader and GPT judge
```

`HARMONY-Subquery` is an experimental branch only. It is not used in the
paper's main comparison, ablations, or claims.

## Reproducibility Contract

Every paper table fixes all of the following:

1. Conversation-disjoint LoCoMo split with seed `42`.
2. Balanced categories 1--4, with 8 training and 16 test questions per category.
3. Qwen3-Embedding-4B retrieval; the main and ablation tables enable
   Qwen3-Reranker-4B.
4. `gpt-4.1-mini` reader with the same source-evidence prompt and
   `gpt-4o-mini` answer judge.
5. Main metrics only: `llm_acc`, `retrieval_tokens`, and
   `retrieval_latency_ms`. Retrieval latency excludes reader and judge time.

Category 5 is excluded from the primary result because it is adversarial and
requires a separately stated protocol. `*-Lite` baselines are unified-interface
implementations of the published retrieval ideas, not official-system
reproductions.

## Paper Matrix

| ID | Question | Methods | Output |
| --- | --- | --- | --- |
| E1 | Accuracy--efficiency trade-off | HARMONY, HyperMem, BM25, Qwen dense, Mem0-Lite, A-Mem-Lite, HippoRAG-Lite, LightRAG-Lite | Main table and Pareto plot |
| E2 | Does adaptive routing help? | HARMONY, fixed FullRecall, fixed RoleBalanced, NoRole | Ablation table and grouped bars |
| E3 | Is the gain dependent on Qwen reranking? | HARMONY and fixed FullRecall, reranker on/off | Robustness table |
| E4 | Which scenarios benefit? | E1 methods grouped by single-hop, temporal, multi-hop, open-domain | Scenario table |
| E5 | Is evaluation protocol driving the result? | Fixed E1 retrieval trace, alternate reader/judge only | Appendix sensitivity table |

E1--E4 use the complete retrieval pipeline. E5 reuses a fixed evidence trace
and must never be mixed into the main comparison.

## Statistical Reporting

The partial LoCoMo setting is not a substitute for a full benchmark claim.
For each method, report `n=64` test questions and category-level `n=16`.
Results are shown to four decimals for `llm_acc`, one decimal for tokens and
milliseconds, and all raw answer/judge records are retained in the run directory.

For the final submission, repeat E1 and E2 with at least three seeds or a
larger held-out split, then add bootstrap confidence intervals. The present
matrix is the controlled, compute-bounded development protocol.
