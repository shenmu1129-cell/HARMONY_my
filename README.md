# HARMONY-Mem

**HARMONY-Mem** is a hypergraph-based long-term memory retrieval framework for LLM agents.
It formulates different retrieval paths over structured memory as an **action space**, and uses a lightweight **Bandit / RL policy** to select the most suitable retrieval action for each query.

The core idea is simple:

> Different queries need different memory retrieval paths. HARMONY-Mem learns which hypergraph path to use while preserving the original evidence.

---

## 1. Overview

HARMONY-Mem follows a query-conditioned retrieval pipeline:

```text
Memory Leaves
  ↓
Structured Hypergraph Construction
  - Topic
  - Episode
  - Fact
  - Behavioral Hyperedges
  ↓
Hypergraph Retrieval Action Space
  - Hybrid Traversal
  - Edge-Source Retrieval
  - Global Leaf Retrieval
  - Adaptive Fallback
  ↓
Bandit / RL Policy
  - Thompson Sampling
  - UCB
  - LinUCB
  ↓
Retrieval Execution
  ↓
Source-Preserving Leaf Backtracking
  ↓
Evidence / Retrieval Result
  ↓
Reward-based Policy Update
```

Compared with fixed RAG or fixed hypergraph traversal, HARMONY-Mem does not use the same retrieval rule for every query. Instead, it treats different hypergraph retrieval paths as actions and learns a policy to select among them.

---

## 2. Key Ideas

### 2.1 Structured Hypergraph Memory

HARMONY-Mem starts from original memory leaves, such as PersonaChat-style user memories. These memories are organized into a structured hypergraph with multiple semantic levels:

| Component | Meaning |
|---|---|
| `Fact` | Atomic memory evidence, usually corresponding to the original memory leaf. |
| `Episode` | Event-level grouping of related facts. |
| `Topic` | High-level semantic organization. |
| `Behavioral Hyperedge` | A hyperedge that connects multiple related memory units describing user preferences, habits, goals, or behavioral patterns. |

In the current implementation, **facts / memory leaves remain the final evidence source**. Topic, episode, and behavioral hyperedges are used to guide retrieval, but the final evidence is traced back to the original memory leaves.

---

### 2.2 Hypergraph Retrieval Action Space

The retrieval action space contains several representative retrieval paths over the memory hypergraph:

| Action | Description |
|---|---|
| `hybrid traversal` | Traverses Topic / Episode / Fact / Hyperedge structures jointly. |
| `edge-source retrieval` | Retrieves relevant hyperedges and then returns their original source memory leaves. |
| `global leaf retrieval` | Directly retrieves top-k memory leaves from the global memory bank. |
| `adaptive fallback` | Uses a lightweight fallback path when the query is ambiguous or the main retrieval path is uncertain. |

The current experimental code defines several action-space settings:

```text
core
llm_pruned
fast_pruned
dynamic_k
```

For the main lightweight setting, `fast_pruned` uses a compact set of actions:

```text
hybrid_roi_light_k3
edge_source_response
global_response_k2
adaptive_tiny_ref
```

---

### 2.3 Bandit / RL-based Query Routing

HARMONY-Mem formulates memory retrieval as a **query routing bandit problem**.

For each query, the system extracts a query state, including:

```text
query type
keywords
semantic cues
route confidence
historical feedback
```

Then a Bandit / RL policy selects one retrieval action from the action space.

Supported policies include:

```text
epsilon_greedy
ucb1
thompson
exp3
softmax_pg
linucb
dynamic_k_ucb
```

The policy is updated using retrieval feedback. A typical reward is:

```text
Reward = Hit / Recall - Token Cost - Latency Cost
```

Thus, the policy learns not only which action is accurate, but also which action is more token-efficient and latency-efficient.

---

### 2.4 Source-Preserving Leaf Backtracking

Intermediate topic, episode, or hyperedge nodes are useful for retrieval, but they may be summaries or high-level abstractions. To reduce evidence drift, HARMONY-Mem always traces the final evidence back to the original memory leaves.

```text
Query
  → Selected Hypergraph Action
  → Topic / Episode / Fact / Hyperedge Traversal
  → Source-Preserving Leaf Backtracking
  → Original Memory Evidence
```

This design makes the retrieved evidence more faithful and easier to evaluate.

---

## 3. Repository Structure

```text
HARMONY/
├── data/                         # Data files or prepared examples
├── docs/                         # Notes and method documentation
├── examples/                     # Data preparation, graph construction, and retrieval evaluation
│   ├── prepare_parlai_memory_data.py
│   ├── build_behavioral_hybrid_memory.py
│   ├── eval_behavioral_profile_graph.py
│   ├── eval_cost_aware_retrieval.py
│   ├── policy_routing_eval.py
│   └── profile_centric_hypergraph_eval.py
├── experiments/                  # Main experimental scripts
│   ├── eval_persona_advanced_retrieval.py
│   ├── eval_persona_report_compare.py
│   ├── eval_persona_rl_retrieval.py
│   ├── eval_persona_llm_judge_rag.py
│   ├── eval_memory_module_ablation.py
│   └── eval_longmemeval_mini.py
├── hypermem/                     # Core memory and retrieval modules
│   ├── profile_centric_hypergraph.py
│   ├── behavioral_profile.py
│   ├── cost_aware_retrieval.py
│   ├── dual_path_retrieval.py
│   ├── hypermem_style_hierarchy_builder.py
│   ├── llm_hierarchy_builder.py
│   ├── llm_profile_builder.py
│   └── query_router.py
├── scripts/                      # Running scripts
├── requirements.txt
└── README.md
```

---

## 4. Installation

```bash
git clone https://github.com/shenmu1129-cell/HARMONY.git
cd HARMONY

conda create -n harmony python=3.10 -y
conda activate harmony

pip install -r requirements.txt
```

If you already have a working environment, you can directly install the required packages:

```bash
pip install -r requirements.txt
```

---

## 5. Quick Start

### Step 1: Prepare PersonaChat / ConvAI2 / MSC-style memory data

```bash
python examples/prepare_parlai_memory_data.py \
  --dataset-root /path/to/Persona-Chat \
  --out-dir outputs/persona_chat/data \
  --max-memory 50 \
  --max-questions 1000 \
  --show-progress
```

This script writes:

```text
outputs/persona_chat/data/memory_facts.jsonl
outputs/persona_chat/data/questions.jsonl
outputs/persona_chat/data/data_report.json
```

`memory_facts.jsonl` contains the original memory leaves, while `questions.jsonl` contains QA queries and gold answers.

---

### Step 2: Build the behavioral hybrid memory graph

```bash
python examples/build_behavioral_hybrid_memory.py \
  --memory-json outputs/persona_chat/data/memory_facts.jsonl \
  --output-dir outputs/persona_chat/graph \
  --max-memory 50
```

The constructed graph is usually saved as:

```text
outputs/persona_chat/graph/behavioral_hybrid_graph.json
```

---

### Step 3: Run HARMONY-Mem policy retrieval

```bash
python experiments/eval_persona_rl_retrieval.py \
  --memory-graph outputs/persona_chat/graph/behavioral_hybrid_graph.json \
  --memory-json outputs/persona_chat/data/memory_facts.jsonl \
  --questions-json outputs/persona_chat/data/questions.jsonl \
  --output-dir outputs/persona_chat/harmony_rl \
  --max-questions 1000 \
  --split-train 500 \
  --policies linucb \
  --arm-set fast_pruned \
  --oracle
```

Important arguments:

| Argument | Meaning |
|---|---|
| `--memory-graph` | Path to the constructed memory hypergraph. |
| `--memory-json` | Path to the original memory leaf file. |
| `--questions-json` | Path to the QA query file. |
| `--output-dir` | Directory for experimental results. |
| `--max-questions` | Number of queries to evaluate. |
| `--split-train` | Number of queries used for policy learning. |
| `--policies` | Bandit / RL policies to run. |
| `--arm-set` | Retrieval action-space setting. |
| `--oracle` | Evaluate oracle arm upper bound. |

---

## 6. Outputs

The main output files include:

```text
rl_results.csv
rl_summary.csv
rl_trace.jsonl
rl_policy_states.json
```

Typical fields include:

| Field | Description |
|---|---|
| `method` | Policy and arm-set name. |
| `phase` | `train`, `test`, `online`, or `oracle`. |
| `chosen_arm` | Retrieval action selected by the policy. |
| `accuracy` / `hit` | Whether the retrieved evidence supports the gold answer. |
| `recall` | Evidence recall against the gold answer. |
| `tokens` | Evidence token cost. |
| `retrieval_ms` | Retrieval latency. |
| `reward` | Combined retrieval reward. |

---

## 7. Example Experimental Setting

The current main experiment uses a PersonaChat partial-memory retrieval setting:

```text
Memory bank: 50 original memory leaves
Queries: 1,000 QA queries
Policy learning: 500 queries
Testing: 500 queries
```

This setting evaluates whether a retrieval method can select the correct evidence under limited-memory and low-token constraints. It is different from full-context question answering, where the entire conversation or complete persona profile may be available.

Representative results from the current setting:

| Method | Accuracy ↑ | Tokens ↓ |
|---|---:|---:|
| BM25 Retrieval | 0.152 | 45.86 |
| Response-prior Retrieval | 0.220 | 38.55 |
| Local Best Static HG | 0.410 | 30.82 |
| Local Adaptive Tiny | 0.440 | 28.59 |
| **HARMONY-Mem** | **0.514** | **26.79** |
| Oracle Upper Bound | 0.663 | - |

The oracle upper bound selects the best retrieval action for each query after observing all candidate actions. It is not a deployable method, but estimates the performance ceiling under the current memory coverage and action space.

---

## 8. Difference from Standard RAG

Standard RAG usually follows a fixed retrieval pipeline:

```text
Query → Retriever → Top-k Documents → LLM
```

HARMONY-Mem performs query-conditioned action routing:

```text
Query
  → Query State Extraction
  → Bandit / RL Action Selection
  → Hypergraph Retrieval
  → Source-Preserving Leaf Backtracking
  → Evidence / Retrieval Result
  → Reward Update
```

The main difference is that HARMONY-Mem learns which retrieval path should be used for each query.

---

## 9. Difference from HyperMem-style Fixed Retrieval

HyperMem-style memory organizes long-term memory using structured topic, episode, fact, and hyperedge relations. HARMONY-Mem follows this structured-memory motivation, but focuses on adaptive retrieval policy learning.

```text
HyperMem-style fixed retrieval:
Query → Fixed Hypergraph Traversal → Evidence

HARMONY-Mem:
Query → Query State → Bandit / RL Policy → Selected Hypergraph Action → Source-Preserved Evidence
```

Thus, the main contribution of HARMONY-Mem is not simply building a hypergraph, but learning a query-conditioned policy over hypergraph retrieval actions.

---

## 10. Notes

- The current implementation is a research prototype.
- The default lightweight experiments can run without training a large language model.
- Some scripts support LLM-assisted hierarchy or profile construction, but the main reported retrieval experiments focus on policy-based action selection over structured memory.
- For stronger experiments, future work should evaluate full PersonaChat / ConvAI2 / MSC / LoCoMo settings, stronger dense retrievers, rerankers, and complete HyperMem-style baselines.

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{harmony_mem,
  title  = {HARMONY-Mem: Hypergraph Action Reinforced Memory Retrieval with Source-Preserving Evidence},
  author = {Anonymous},
  year   = {2026},
  note   = {Work in progress}
}
```

---

## License

This project is released under the Apache-2.0 License.
