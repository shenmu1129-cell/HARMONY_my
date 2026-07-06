# Profile-Centric Hypergraph Memory

当前仓库只保留这一条主线：`User Profile Hyperedge -> Fact`。

完整流水线：

```text
Memory Facts
  ↓
Construct User Profile Hyperedges
  ↓
Build Local Hashed Embeddings
  ↓
Train Reward-guided Hyperedge Utility
  ↓
Rank Profile Hyperedges
  ↓
Rank Facts inside Selected Hyperedges
  ↓
Evaluate Accuracy / Recall / Tokens / Reward
```

## 核心设计

用户画像超边不是普通标签，而是一个高阶记忆单元：

```text
profile_hyperedge = {
  edge_type,
  summary,
  member_facts,
  embedding,
  freshness_score,
  stability_score,
  confidence_score,
  utility_score,
  hit_count,
  failure_count
}
```

其中：

- `embedding`：当前用本地 hashed embedding + cosine，相当于无需 GPU/API 的向量检索版本。
- `utility_score`：轻量 bandit-style reward update 学出来的超边价值。
- `member_facts`：最终进入 prompt 的底层证据。

检索时：

```text
query embedding
  ↓
profile hyperedge ranking = embedding similarity + utility + freshness + stability + type match - token cost
  ↓
fact ranking inside selected hyperedges
  ↓
selected evidence
```

## 运行 demo

```bash
conda activate wwt_hyperMem

bash scripts/run_profile_centric_hypergraph.sh \
  DEMO \
  outputs/profile_centric_demo \
  1.0 \
  0.5
```

## 运行自己的数据

```bash
bash scripts/run_profile_centric_hypergraph.sh \
  /home/sutongtong/wwt/code \
  outputs/profile_centric_hg \
  0.5 \
  0.5
```

参数含义：

```text
第 1 个参数：数据来源目录；DEMO 表示内置小数据
第 2 个参数：输出目录
第 3 个参数：数据使用比例
第 4 个参数：QA 训练比例
```

## 单独运行 eval

```bash
python examples/profile_centric_hypergraph_eval.py \
  --memory-json data/memory_facts.jsonl \
  --questions-json data/questions.jsonl \
  --train-ratio 0.5 \
  --online-eval \
  --output-dir outputs/profile_centric_eval
```

## 输出

```text
profile_centric_results.csv
profile_centric_summary.csv
profile_centric_summary.json
profile_centric_trace.jsonl
profile_centric_trained_memory.json
```

summary 里重点看这些方法：

```text
embedding_only_profile_hg
reward_utility_train
reward_utility_frozen_test
online_predict_then_update_test
```

其中 `reward_utility_frozen_test` 是主测试结果，测试阶段冻结训练得到的 utility；`online_predict_then_update_test` 是在线评估，每个测试问题先计分再更新。
