# Profile-Centric Hypergraph Memory

本仓库当前只保留一个主线：**基于用户画像超边的长期记忆检索**。

核心流程：

```text
Memory Facts
  ↓
User Profile Hyperedges
  ↓
Local Embedding Index
  ↓
Reward-guided Utility Training
  ↓
Profile Hyperedge Ranking
  ↓
Fact Evidence Selection
  ↓
Accuracy / Recall / Token Evaluation
```

也就是把 HyperMem 的 `Topic -> Episode -> Fact` 主检索路径，改成：

```text
Query -> User Profile Hyperedge -> Fact
```

## 方法含义

- `Fact`：底层事实证据。
- `User Profile Hyperedge`：用户画像超边，连接一组共同描述用户偏好、目标、习惯、当前状态或研究方向的 facts。
- `Embedding`：当前版本使用本地 hashed embedding 和 cosine similarity，不依赖 GPU/API；正式实验可替换为 Qwen/OpenAI embedding。
- `Reward Utility`：轻量 bandit-style 更新，不训练大模型；训练 QA 命中、召回高、token 少则升权，否则降权。
- `Retrieval`：先按 query 和画像超边的 embedding similarity 召回，再结合 utility/freshness/stability 等分数重排，最后从超边成员 facts 中选择证据。

## 代码结构

```text
hypermem/profile_centric_hypergraph.py
examples/prepare_profile_centric_data.py
examples/profile_centric_hypergraph_eval.py
scripts/run_profile_centric_hypergraph.sh
docs/profile_centric_hypergraph.md
```

## 快速运行 demo

```bash
conda activate wwt_hyperMem

bash scripts/run_profile_centric_hypergraph.sh \
  DEMO \
  outputs/profile_centric_demo \
  1.0 \
  0.5
```

## 使用自己的数据

```bash
bash scripts/run_profile_centric_hypergraph.sh \
  /home/sutongtong/wwt/code \
  outputs/profile_centric_hg \
  0.5 \
  0.5
```

参数含义：

```text
第 1 个参数：数据来源目录；用 DEMO 表示内置小数据
第 2 个参数：输出目录
第 3 个参数：数据使用比例，例如 0.5 表示使用一半数据
第 4 个参数：QA 训练比例，例如 0.5 表示前 50% QA 用于 reward utility 训练，后 50% 用于测试
```

## 输出文件

```text
outputs/profile_centric_hg/data_report.json
outputs/profile_centric_hg/eval/profile_centric_summary.csv
outputs/profile_centric_hg/eval/profile_centric_summary.json
outputs/profile_centric_hg/eval/profile_centric_results.csv
outputs/profile_centric_hg/eval/profile_centric_trace.jsonl
outputs/profile_centric_hg/eval/profile_centric_trained_memory.json
```

重点查看：

```bash
cat outputs/profile_centric_hg/eval/profile_centric_summary.csv
```

主要方法行：

```text
embedding_only_profile_hg
reward_utility_train
reward_utility_frozen_test
online_predict_then_update_test
```

其中：

- `embedding_only_profile_hg`：只用 embedding 相似度，不使用训练后的 utility。
- `reward_utility_train`：训练阶段，使用 QA feedback 更新画像超边 utility。
- `reward_utility_frozen_test`：测试阶段冻结 utility，评估最终 accuracy / recall / token。
- `online_predict_then_update_test`：在线评估，每个测试问题先计分，再用反馈更新。
