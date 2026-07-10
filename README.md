# HARMONY-Mem

HARMONY-Mem is the current long-term conversational-memory retrieval framework in this repository. It keeps HyperMem's hierarchical memory organisation, then adds source-preserving evidence, role-conditioned behavioural hyperedges, and a recall-safe contextual-bandit router that chooses the retrieval budget per query.

The research question is practical: retain strong answer accuracy while controlling the evidence-token budget and retrieval latency.

## Current Framework

```text
LoCoMo dialogue rows
  -> source-preserving facts (row/date/role/raw text)
  -> topic / episode / fact hierarchy + role-conditioned hyperedges
  -> Qwen dense retrieval + BM25/RRF fusion + Qwen reranking
  -> contextual-bandit action router
  -> compact, balanced, or recall-oriented evidence packing
  -> gpt-4.1-mini answer generation
  -> gpt-4o-mini LLM-as-a-judge evaluation
```

The router selects from a small, cost-aware action space:

| Action | Role handling | Retrieval budget | Intended use |
| --- | --- | --- | --- |
| `RoleCompact` | soft role-aware routing | low | simple role-specific questions |
| `RoleBalanced` | soft role-aware routing | medium | typical role-specific questions |
| `FullCompact` | global retrieval | low | simple non-role questions |
| `FullBalanced` | global retrieval | medium | general questions |
| `FullRecall` | global retrieval | high | temporal, multi-hop, or uncertain questions |

Role information is a routing signal, not a hard filter. This prevents cross-speaker evidence from being removed before ranking. Final evidence always points back to the original dialogue row, rather than relying only on generated summaries.

## Evaluation Metrics

The current report intentionally keeps three primary metrics:

- `llm_acc`: answer correctness judged by an LLM.
- `retrieval_tokens`: tokens in the retrieved evidence sent to the answer model.
- `retrieval_latency_ms`: retrieval latency only; it excludes answer-generation and judge latency.

The latest HARMONY LoCoMo run used Qwen3 embedding and reranking, 60 training questions, 80 test questions, source-preserving evidence, and `gpt-4o-mini` for the original reader/judge run.

| Setting | Questions | `llm_acc` | `retrieval_tokens` | `retrieval_latency_ms` |
| --- | ---: | ---: | ---: | ---: |
| Strict original evaluation | 80 | 0.7625 | 774.1 | 3463.2 |
| HyperMem-aligned answer/judge protocol | 78, category 5 excluded | 0.9744 | 771.6 | 3454.5 |

The second row regenerates answers from the same retrieved evidence with `gpt-4.1-mini` and HyperMem's CoT answer prompt, then uses the HyperMem-style `gpt-4o-mini` judge. It is a same-split, final-stage protocol alignment check, not a fresh full-benchmark reproduction or a directly comparable paper score.

## Setup

Python 3.12 is recommended. Local development in this workspace uses the `wwt310` environment.

```bash
conda run -n wwt310 python -m pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export EMBEDDING_BASE_URL="http://localhost:11810/v1/embeddings"
export RERANKER_BASE_URL="http://localhost:12810"
```

The main LoCoMo experiment expects Qwen3-Embedding-4B and Qwen3-Reranker-4B services compatible with those endpoints. The API smoke test never stores a key in source control:

```bash
OPENAI_API_KEY="sk-..." conda run -n wwt310 python examples/openai_api_smoke_test.py --model gpt-4o-mini
```

## Run HARMONY-Mem

Retrieval-only smoke test:

```bash
conda run -n wwt310 python experiments/eval_locomo_comparison.py \
  --data data/locomo10.json \
  --output-dir outputs/harmony_smoke \
  --max-examples 60 \
  --train-size 30 \
  --test-size 30 \
  --methods HARMONY-Mem \
  --use-qwen-reranker \
  --skip-empty-gold
```

Run answer generation and judging on the retrieved trace:

```bash
OPENAI_API_KEY="sk-..." conda run -n wwt310 python experiments/rejudge_trace_hypermem_style.py \
  --trace outputs/harmony_smoke/trace.jsonl \
  --output-dir outputs/harmony_hypermem_style \
  --skip-category-5 \
  --reader-model gpt-4.1-mini \
  --judge-model gpt-4o-mini
```

`experiments/eval_locomo_comparison.py` contains the HARMONY router and lightweight LoCoMo-compatible comparison baselines. `experiments/rejudge_trace_hypermem_style.py` reruns only the final answer/judge stage over an existing trace, so retrieval and router results are unchanged.

## Repository Layout

```text
hypermem/
  profile_centric_hypergraph.py  # profile-hyperedge memory primitives
  query_router.py                # deterministic profile/episodic query router
  main/                          # original HyperMem six-stage pipeline
experiments/
  eval_locomo_comparison.py      # HARMONY-Mem LoCoMo runner
  eval_longmemeval_mini.py       # reusable retrieval/evaluation components
  eval_rl_memory_complexity.py   # router utilities
  rejudge_trace_hypermem_style.py# protocol-aligned final-stage re-evaluation
docs/
  locomo_experiments.md          # comparison methodology and commands
examples/
  openai_api_smoke_test.py       # environment-variable-only OpenAI API test
```

Generated traces, caches, model outputs, and API keys are intentionally excluded from Git.
