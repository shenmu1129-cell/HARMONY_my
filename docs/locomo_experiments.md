# LoCoMo comparison entrypoint

This directory keeps older exploratory scripts for traceability, but the current paper-facing LoCoMo entrypoint is:

```bash
python experiments/eval_locomo_comparison.py
```

## Main method

`HARMONY-Mem` uses the current confirmed pipeline:

1. Build source-preserving fact nodes from original dialogue rows.
2. Build HyperMem-style topic/episode/fact retrieval structure.
3. Build role-conditioned behavior hyperedges by grouping memories by speaker role.
4. Train a lightweight contextual bandit action router on the training split.
5. At test time, the router selects retrieval depth/budget and whether to use role gating.
6. Retrieve with Qwen embedding, RRF-style fusion, optional Qwen reranking, and source-preserving 96-word evidence snippets.
7. Optionally generate answers and evaluate with LLM judge.

The optimization target is not only answer accuracy. The main reported metrics are:

- `llm_acc`
- `retrieval_tokens`
- `avg_ms` / `p50_ms`
- `train_seconds`
- retrieval diagnostics: `fact_hit`, `answer_recall`, `all_hit`

## Baselines

The script includes:

- `HyperMem-Flow`: HyperMem-style topic -> episode -> fact retrieval.
- `BM25-turn`: keyword turn retrieval.
- `QwenEmb-turn`: dense turn retrieval with the same local Qwen embedding/reranker stack.
- `Mem0-Lite`: multi-signal semantic, BM25, entity, and temporal retrieval inspired by Mem0.
- `A-Mem-Lite`: seed retrieval plus linked-memory expansion inspired by A-Mem.
- `HippoRAG-Lite`: graph propagation/PPR-style retrieval inspired by HippoRAG.
- `LightRAG-Lite`: dual-level topic/episode/fact retrieval inspired by LightRAG.

External official repositories were downloaded under:

```text
/home/sutongtong/wwt/code/rag_baselines/
```

The `*-Lite` methods are LoCoMo-compatible lightweight reproductions for fast three-day comparison. They avoid heavy official training/service setup while preserving the retrieval idea under the same data/model/evaluation stack.

## Recommended quick run

Retrieval-only comparison, roughly twice the previous small test size:

```bash
python experiments/eval_locomo_comparison.py \
  --data data/locomo10.json \
  --output-dir outputs/locomo_main_allbaselines_train60_test120_retrieval_v1 \
  --max-examples 260 \
  --train-size 60 \
  --test-size 120 \
  --seed 42 \
  --use-qwen-reranker \
  --max-llm-judge 0
```

Then run limited LLM judge on the same split:

```bash
python experiments/eval_locomo_comparison.py \
  --data data/locomo10.json \
  --output-dir outputs/locomo_main_allbaselines_train60_test120_judge30_v1 \
  --max-examples 260 \
  --train-size 60 \
  --test-size 120 \
  --seed 42 \
  --use-qwen-reranker \
  --max-llm-judge 30 \
  --reader-model deepseek-chat \
  --judge-model deepseek-chat \
  --reader-mode temporal
```

Do not run full LoCoMo judge until the sampled comparison is stable.
