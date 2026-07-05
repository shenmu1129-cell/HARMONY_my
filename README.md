# UP-HyperPool: Semi-Automatic User-Profile Hyperedge Pool

本仓库在 HyperMem 的 Topic / Episode / Fact 长期记忆底座之上，加入一个 **半自动用户画像超边池快速通道**。

核心思想：

```text
Topic / Episode / Fact base memory
        ↓
Semi-automatic Profile Discovery
        ↓
User-Profile Hyperedge Pool
        ↓
Profile fast-channel retrieval
        ↓ insufficient
Fallback retrieval
```

用户画像不是纯强化学习直接学出来的，而是：

```text
seed rule typing + unsupervised discovery + reward-guided optimization
```

也就是先用少量规则种子稳定识别常见画像，再把低置信记忆放入 discovery buffer，通过相似性聚类自动形成新的画像超边，最后用检索命中和回答质量奖励长期更新超边权重。

---

## 1. 方法定位

HyperMem 主要解决长期对话记忆的结构组织问题：

```text
Topic → Episode → Fact
```

本项目进一步解决：

```text
如何在长期交互中发现用户画像、使用习惯、长期目标和当前任务状态，
并把这些高价值记忆组织成可快速检索的 profile hyperedge pool。
```

---

## 2. 三种画像发现模式

### rule

人工预设 seed profile types：

```text
preference / goal / habit / current_state / domain_knowledge / temporal_evolution
```

优点是快、稳、可解释；缺点是不能发现新的用户画像维度。

### unsupervised

不预设画像类别。所有 fact 进入 discovery buffer，通过相似性聚类形成：

```text
auto_discovered profile hyperedges
```

优点是能发现未知用户习惯；缺点是低数据量时可能有噪声。

### hybrid 默认

高置信规则类别直接归入 seed profile hyperedge；低置信 fact 进入 discovery buffer，再由无监督聚类形成新画像超边。

```text
New fact
  ↓
Rule seed classifier
  ├── high confidence → seed profile hyperedge
  └── low confidence  → discovery buffer → auto_discovered hyperedge
  ↓
Reward-guided utility update
```

这是当前推荐主方法。

---

## 3. 新增代码

```text
hypermem/profile_hyperedge_pool.py
examples/user_profile_hyperedge_demo.py
examples/profile_hyperedge_pool_eval.py
scripts/run_profile_hyperedge_pool.sh
```

核心模块包含：

- `MemoryNode`：Topic / Episode / Fact 的轻量节点视图；
- `ProfileHyperedge`：用户画像超边；
- `UserProfileHyperedgePool`：半自动动态超边池；
- `ProfileTypingMode`：`rule / unsupervised / hybrid`；
- `discover_profile_hyperedges()`：无监督画像超边发现；
- `retrieve_fast_channel()`：画像超边池快速检索；
- `update_rewards()`：根据命中和回答质量更新超边 utility；
- `save()` / `load()`：保存和加载超边池。

---

## 4. 安装

建议使用原环境：

```bash
conda activate wwt_hyperMem
```

或新建环境：

```bash
conda create -n memory_my python=3.12 -y
conda activate memory_my
pip install -r requirements.txt
```

当前新增 demo 只依赖 Python 标准库。

---

## 5. 运行 demo

```bash
python examples/user_profile_hyperedge_demo.py
```

它会依次演示：

```text
rule
unsupervised
hybrid
```

输出文件：

```text
outputs/profile_hyperedge_demo_pool_rule.json
outputs/profile_hyperedge_demo_pool_unsupervised.json
outputs/profile_hyperedge_demo_pool_hybrid.json
```

---

## 6. 运行 retrieval-only 评测

一键运行三种模式：

```bash
bash scripts/run_profile_hyperedge_pool.sh outputs/profile_eval
```

单独运行 hybrid 主方法：

```bash
python examples/profile_hyperedge_pool_eval.py \
  --profile-typing-mode hybrid \
  --output-dir outputs/profile_hybrid
```

对比三种模式：

```bash
python examples/profile_hyperedge_pool_eval.py --profile-typing-mode rule --output-dir outputs/profile_rule
python examples/profile_hyperedge_pool_eval.py --profile-typing-mode unsupervised --output-dir outputs/profile_unsup
python examples/profile_hyperedge_pool_eval.py --profile-typing-mode hybrid --output-dir outputs/profile_hybrid
```

关闭 fallback 做消融：

```bash
python examples/profile_hyperedge_pool_eval.py \
  --profile-typing-mode hybrid \
  --no-fallback \
  --output-dir outputs/profile_hybrid_no_fallback
```

---

## 7. 使用自己的数据

```bash
python examples/profile_hyperedge_pool_eval.py \
  --memory-json data/my_memory_facts.jsonl \
  --questions-json data/my_questions.jsonl \
  --profile-typing-mode hybrid \
  --output-dir outputs/profile_eval
```

Memory fact 示例：

```json
{"content": "用户喜欢先分析论文原理，再判断创新性，最后生成 Codex prompt。"}
```

推荐格式：

```json
{
  "fact_id": "fact_001",
  "content": "用户当前研究主线是用户画像引导的动态超边池快速通道。",
  "keywords": ["用户画像", "动态超边池", "快速通道"],
  "timestamp": 12,
  "topic_id": "memory_research",
  "episode_ids": ["episode_003"]
}
```

Question 示例：

```json
{
  "qid": "q_001",
  "question": "我现在这个 memory 方案的核心是什么？",
  "gold": ["用户画像", "动态超边池"],
  "category": "current_state"
}
```

---

## 8. 输出文件

每个输出目录包含：

```text
profile_hyperedge_pool_results.csv
profile_hyperedge_pool_by_category.csv
profile_hyperedge_pool_summary.json
profile_hyperedge_pool_trace.jsonl
profile_hyperedge_pool.json
```

summary 中包含：

```text
hit
recall
tokens
reward
fallback_rate
edge_type_counts
num_edges
active_edges
discovery_buffer_size
```

---

## 9. 支持的画像超边类型

| 类型 | 含义 |
|---|---|
| `preference` | 用户偏好，例如回答风格、分析方式 |
| `goal` | 用户长期目标，例如投稿、研究目标 |
| `habit` | 用户使用习惯和工作流 |
| `domain_knowledge` | 用户当前研究领域常用知识 |
| `current_state` | 用户当前任务状态和最新方案 |
| `temporal_evolution` | 用户想法或目标的时间演化 |
| `evidence_group` | 支撑某类问题的一组证据 |
| `supersede` | 新画像覆盖旧画像 |
| `auto_discovered` | 无监督发现的新画像维度 |
| `other` | 其他 profile 相关记忆 |

---

## 10. 当前定位

本项目当前主线可以概括为：

> 在 HyperMem 的 Topic-Episode-Fact 长期记忆底座之上，构建一个半自动、奖励优化的用户画像超边池，使高价值、常用、符合用户习惯和当前任务状态的记忆形成快速检索通道；当快速通道证据不足时，再回退到备用检索路径。
