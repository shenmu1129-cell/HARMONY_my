# UP-HyperPool: User-Profile Guided Hyperedge Pool for Long-Term Memory

本仓库基于 HyperMem 的 Topic / Episode / Fact 长期对话记忆底座，新增一个 **用户画像引导的动态超边池快速通道**。

核心思想不是继续只调 `top-k` 或固定检索路径，而是在原始记忆底座之上维护一组高价值、可复用、可奖励更新的 profile hyperedges：

```text
Topic / Episode / Fact base memory
        ↓
User-Profile Hyperedge Pool
        ↓
Profile fast-channel retrieval
        ↓ insufficient
Original HyperMem path / global fact fallback
```

也就是说：

- **普通记忆** 仍然保留在 Topic / Episode / Fact 底座中，保证覆盖率；
- **高价值用户画像记忆** 被提升到超边池中，形成快速通道；
- **检索时先走用户画像超边池**，如果证据不足，再回退到原始 HyperMem 路径或 global fact retrieval；
- **超边 utility 可由 reward 更新**：命中、帮助回答、符合当前用户状态则升权；错误、过期、无贡献则降权。

---

## 1. 方法定位

原 HyperMem 主要解决：

```text
如何把长期对话记忆组织成 Topic → Episode → Fact 层级超图。
```

本项目进一步研究：

```text
如何在长期使用中学习用户画像、习惯、目标和当前任务状态，
并把这些高价值记忆组织成一个动态超边池，用作快速检索通道。
```

相比固定 Topic → Episode → Fact traversal，本项目的差异是：

| 维度 | HyperMem-style retrieval | UP-HyperPool |
|---|---|---|
| 基础结构 | Topic / Episode / Fact | Topic / Episode / Fact |
| 快速通道 | 无显式用户画像通道 | User-profile hyperedge pool |
| 超边含义 | 层级归属为主 | 偏好、目标、习惯、当前状态、领域知识、时间演化 |
| 检索方式 | 固定 coarse-to-fine | profile fast channel first + fallback |
| 更新机制 | 较弱 | reward-guided utility update |
| 目标 | 找相关上下文 | 更快命中用户常用、高价值、个性化记忆 |

---

## 2. 当前新增内容

### 核心模块

```text
hypermem/profile_hyperedge_pool.py
```

包含：

- `MemoryNode`：Topic / Episode / Fact 的轻量节点视图；
- `ProfileHyperedge`：用户画像超边；
- `UserProfileHyperedgePool`：动态超边池；
- `retrieve_fast_channel()`：画像超边池快速检索；
- `update_rewards()`：根据命中/回答质量更新超边 utility；
- `save()` / `load()`：保存和加载超边池。

### Demo 脚本

```text
examples/user_profile_hyperedge_demo.py
```

用于快速演示：

- 如何从用户长期记忆 fact 构建 profile hyperedge pool；
- 如何检索 preference / goal / habit / current-state / domain-knowledge 等超边；
- 如何输出 evidence；
- 如何保存超边池。

### Retrieval-only 评测脚本

```text
examples/profile_hyperedge_pool_eval.py
```

用于完整 retrieval-only 实验：

- 支持内置 demo 数据；
- 支持外部 JSON / JSONL memory facts；
- 支持外部 JSON / JSONL questions；
- 输出 per-query trace、summary、category-level 表；
- 不调用 stage 5/6，不调用 LLM judge，不调用 OpenAI / DeepSeek。

---

## 3. 安装

建议使用你原来的环境，例如：

```bash
conda activate wwt_hyperMem
```

如果新建环境：

```bash
conda create -n memory_my python=3.12 -y
conda activate memory_my
pip install -r requirements.txt
```

当前新增的用户画像超边池 demo 只依赖 Python 标准库，不需要外部 LLM API。

---

## 4. 运行 Demo

```bash
python examples/user_profile_hyperedge_demo.py
```

运行后会打印：

- 当前构建出的 profile hyperedge 类型统计；
- 每个 query 命中的超边；
- 展开的 evidence；
- 保存的超边池文件。

输出文件：

```text
outputs/profile_hyperedge_demo_pool.json
```

---

## 5. 运行 Retrieval-only 评测

### 5.1 使用内置 demo 数据

```bash
python examples/profile_hyperedge_pool_eval.py --output-dir outputs/profile_eval
```

输出：

```text
outputs/profile_eval/profile_hyperedge_pool_results.csv
outputs/profile_eval/profile_hyperedge_pool_by_category.csv
outputs/profile_eval/profile_hyperedge_pool_summary.json
outputs/profile_eval/profile_hyperedge_pool_trace.jsonl
outputs/profile_eval/profile_hyperedge_pool.json
```

### 5.2 使用自己的 memory / questions 文件

```bash
python examples/profile_hyperedge_pool_eval.py \
  --memory-json data/my_memory_facts.jsonl \
  --questions-json data/my_questions.jsonl \
  --output-dir outputs/profile_eval
```

---

## 6. 输入数据格式

### 6.1 Memory facts

支持 JSON 或 JSONL。

最小格式：

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

### 6.2 Questions

最小格式：

```json
{"question": "我现在这个 memory 方案的核心是什么？"}
```

带 gold evidence 的 retrieval-only 评测格式：

```json
{
  "qid": "q_001",
  "question": "我现在这个 memory 方案的核心是什么？",
  "gold": ["用户画像", "动态超边池"],
  "category": "current_state"
}
```

说明：

- `gold` 只用于 retrieval-only hit / recall 评测；
- 不会调用 LLM judge；
- 当前指标不是 HyperMem 论文中的 final answer accuracy。

---

## 7. 画像超边类型

当前实现支持：

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
| `other` | 其他 profile 相关记忆 |

---

## 8. 奖励更新机制

每条 profile hyperedge 维护：

```text
profile_score
utility_score
freshness_score
coherence_score
access_count
hit_count
failure_count
status
```

当某条超边被检索并命中 gold evidence：

```text
utility_score ↑
profile_score ↑
hit_count ↑
```

当某条超边被检索但没有贡献：

```text
utility_score ↓
profile_score ↓
failure_count ↑
```

长期无贡献的超边会被 soft-inactive，而不是删除原始 fact。原则是：

> 原始 Topic / Episode / Fact 保留，动态演化的是画像超边索引层。

---

## 9. 后续可拓展方向

当前版本是规则 + reward 更新版，适合作为 demo 和 retrieval-only 实验底座。后续可以继续扩展：

1. **Profile Router**：判断 query 是否优先走 profile fast channel；
2. **Learned Hyperedge Writer**：学习 create / attach / merge / supersede / decay；
3. **Reward Regression / Contextual Bandit**：学习何时使用 profile pool、何时 fallback；
4. **Temporal-aware Profile Pool**：显式维护 earlier / later / current-state / evolution chain；
5. **Final QA Evaluation**：接入统一 generator + judge，对比 answer accuracy 和 token 成本。

---

## 10. 推荐运行命令

### 快速 demo

```bash
conda activate wwt_hyperMem
python examples/user_profile_hyperedge_demo.py
```

### Retrieval-only demo eval

```bash
conda activate wwt_hyperMem
python examples/profile_hyperedge_pool_eval.py --output-dir outputs/profile_eval
```

### 使用自己的数据

```bash
conda activate wwt_hyperMem
python examples/profile_hyperedge_pool_eval.py \
  --memory-json data/my_memory_facts.jsonl \
  --questions-json data/my_questions.jsonl \
  --output-dir outputs/profile_eval
```

---

## 11. 当前定位

本项目当前主线可以概括为：

> **在 HyperMem 的 Topic–Episode–Fact 长期记忆底座之上，构建一个奖励驱动的用户画像超边池，使高价值、常用、符合用户习惯和当前任务状态的记忆形成快速检索通道；当快速通道证据不足时，再回退到原始 HyperMem 路径。**
