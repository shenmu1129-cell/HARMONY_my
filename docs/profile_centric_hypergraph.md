# Profile-Centric Hypergraph Memory

这是当前新增的主线：不再把 `Topic -> Episode -> Fact` 作为主检索路径，而是改成：

```text
Conversation / Memory Facts
        ↓
User Profile Hyperedges
        ↓
Reward-guided Hyperedge Utility Learning
        ↓
Profile Hyperedge Retrieval
        ↓
Fact Evidence Selection
```

也就是：

```text
Query -> User Profile Hyperedge -> Fact -> Answer
```

## 和 HyperMem 的区别

| 维度 | HyperMem | Profile-Centric HG |
|---|---|---|
| 主索引 | Topic / Episode | User Profile Hyperedge |
| 底层证据 | Fact | Fact |
| 检索路径 | Topic -> Episode -> Fact | Profile Hyperedge -> Fact |
| 超边含义 | 组织 topic/episode/fact 的结构关系 | 表示用户画像单元 |
| 学习机制 | 结构基本静态 | reward-guided utility update |

每条用户画像超边包含：

```text
edge_type
summary
member_facts
keywords / embedding proxy
freshness_score
stability_score
confidence_score
utility_score
hit_count / failure_count
```

其中 `utility_score` 是轻量 bandit-style reward update 学出来的，不需要训练大模型。

## 新增代码

```text
hypermem/profile_centric_hypergraph.py
examples/profile_centric_hypergraph_eval.py
scripts/run_profile_centric_hypergraph.sh
```

## 一键运行

```bash
conda activate wwt_hyperMem

bash scripts/run_profile_centric_hypergraph.sh \
  /home/sutongtong/wwt/code \
  outputs/profile_centric_hg \
  0.5 \
  0.5
```

参数含义：

```text
/home/sutongtong/wwt/code      扫描数据来源
outputs/profile_centric_hg     输出目录
0.5                            使用 50% 数据
0.5                            QA 中前 50% 用于 utility 训练，后 50% 用于测试
```

## 输出

```text
outputs/profile_centric_hg/data_report.json
outputs/profile_centric_hg/eval/profile_centric_summary.csv
outputs/profile_centric_hg/eval/profile_centric_summary.json
outputs/profile_centric_hg/eval/profile_centric_results.csv
outputs/profile_centric_hg/eval/profile_centric_trace.jsonl
outputs/profile_centric_hg/eval/profile_centric_trained_memory.json
```

默认会跑这些方法：

```text
profile_hg_embedding_only
profile_hg_utility_train
profile_hg_utility_frozen
profile_hg_online_predict_then_update
```

重点看：

```text
profile_hg_embedding_only.hit / recall / tokens
profile_hg_utility_frozen.hit / recall / tokens
profile_hg_online_predict_then_update.hit / recall / tokens
```

如果 utility 版本优于 embedding-only，说明 reward-guided profile hyperedge utility 有正向作用。

## 单独运行 eval

```bash
python examples/profile_centric_hypergraph_eval.py \
  --memory-json data/my_memory_facts.jsonl \
  --questions-json data/my_questions.jsonl \
  --train-ratio 0.5 \
  --online-eval \
  --output-dir outputs/profile_centric_eval
```

## 注意

当前版本是 retrieval-only，不调用 LLM 生成答案，也不调用外部 embedding 服务。代码里的 lexical overlap 是本地 embedding proxy。正式实验时可以把相似度替换成真实 embedding，但整体流程保持不变：

```text
embedding / BM25 负责候选召回
profile hyperedge utility 负责个性化价值排序
facts 负责最终证据
```
