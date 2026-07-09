# 基于角色条件超图门控与强化动作路由的源保持记忆检索框架

**English title:** HARMONY-Mem: Hypergraph Action Reinforced Memory Retrieval with Role-Conditioned Gates and Source-Preserving Evidence

**投稿定位:** AAAI 方法类论文草稿  
**当前版本:** LoCoMo 主线框架版，实验数值后续替换为最终表格。  
**核心主张:** HARMONY-Mem 面向长期对话记忆问答，以角色条件超图门控快速收缩候选记忆空间，以强化学习动作路由动态选择检索深度和证据预算，并将最终回答证据回溯到原始对话来源，从而在 LLM judge accuracy 接近强检索基线的同时降低检索延迟和上下文 token 成本。

## 摘要

长期对话智能体需要从跨会话、跨角色、跨时间的历史记忆中检索与当前问题相关的证据。现有扁平 RAG 方法计算简单但难以建模 episode、topic 和 fact 之间的高阶关系；HyperMem 等层级超图方法能够提升证据召回，但通常采用固定宽度的粗到细检索，容易带来较高检索延迟和上下文成本。本文提出 HARMONY-Mem，一种基于角色条件超图门控、强化动作路由和源保持证据的长期记忆检索框架。HARMONY-Mem 首先将原始对话构建为 topic、episode、source-preserving fact 三层记忆结构，并额外建立 role-conditioned behavior hyperedge，将同一角色的相关 topic、episode 和 fact 组织为角色子图。在推理阶段，系统根据查询中的角色线索、问题复杂度和记忆规模，使用 recall-safe contextual bandit 选择检索动作，包括是否启用角色门控、检索深度、展开宽度和证据预算。与直接使用摘要作为回答依据不同，HARMONY-Mem 将 topic/episode summary 仅作为 gate 和 ranking 信号，最终 evidence 始终回到原始 memory row 或 query-aware source snippet。我们以 LoCoMo 为主要实验数据集，对比 HyperMem、Mem0、LightRAG、A-Mem、HippoRAG 及 BM25/Dense baselines，并使用 retrieved evidence -> LLM answer generation -> LLM judge 的完整流程，同时报告 LLM judge accuracy、retrieval latency、evidence tokens 和训练时间。初步结果表明，HARMONY-Mem 能在保持可竞争回答准确率的同时显著降低检索延迟与证据成本。

**关键词:** 长期记忆检索；超图；角色门控；强化学习；检索增强生成；源保持证据；LoCoMo

## 1 引言

大语言模型正在从单轮问答工具转向长期交互智能体。个人助理、对话伙伴和任务型 agent 都需要记住用户在多次会话中表达过的偏好、事件、人物关系和时间线。与普通开放域 RAG 不同，长期记忆问答通常具有三个特点：第一，答案证据分散在多个 session 中；第二，问题常常涉及具体角色，例如某个说话人做过什么、喜欢什么或何时发生某事；第三，系统不仅要答对，还要在有限 token 和延迟预算下完成检索。

现有方法各有局限。BM25 和 dense retrieval 直接检索原始 turn，速度快、证据可追溯，但忽略 episode 和 topic 的高阶结构。HyperMem 将长期记忆组织为 topic、episode 和 fact 三层超图，通过 topic -> episode -> fact 的粗到细检索提升召回，但其检索宽度通常是固定的，对不同复杂度问题缺乏自适应成本控制。Mem0、LightRAG、A-Mem 和 HippoRAG 等方法分别强调多信号检索、图索引、动态记忆链接和图传播，但在长期对话场景中仍需要额外机制处理角色干扰、证据源保持和检索预算。

本文提出 HARMONY-Mem。我们的出发点是：长期对话记忆天然具有角色结构。LoCoMo 这类数据集中，每条记忆都有说话人属性；当问题显式提到 Caroline 或 Melanie 时，系统可以先进入对应角色的记忆子图，而不是在全部 memory 中盲目搜索。然而，硬性角色过滤也可能漏掉跨角色证据，例如“Melanie 的朋友何时收养孩子”这类问题可能需要另一方提到的上下文。因此，角色条件超边不应是固定规则，而应作为强化动作路由器可选择的 gate。

HARMONY-Mem 的核心思想是把检索从固定流程变成成本受控的动作选择过程。系统构建 HyperMem-style 三层记忆结构，并在其上加入 role-conditioned behavior hyperedge。随后，recall-safe contextual bandit 根据 query complexity、memory size、role mention 和历史训练反馈，在 RoleBalanced、RoleRecall、FullBalanced、FullRecall 和 FullBroad 等动作中选择检索策略。奖励函数不仅考虑 evidence hit 和 answer recall，也惩罚 token cost 和 retrieval latency。最终，所有证据都通过 source pointer 回到原始 memory row，避免摘要漂移。

本文贡献如下。

1. 提出 role-conditioned behavior hyperedge，将长期对话中的角色属性显式建模为高阶门控结构，用于在检索前收缩候选空间并减少跨角色干扰。
2. 提出 recall-safe reinforced action router，将长期记忆检索建模为上下文 bandit 动作选择问题，在角色门控、全图检索、展开宽度和证据预算之间动态取舍。
3. 提出 source-preserving evidence packing，将 summary 仅作为 gate/ranking 信号，最终回答证据始终回到原始对话行或 query-aware source snippet。
4. 构建以 LoCoMo 为主的完整评估流程，使用 retrieved evidence -> LLM answer -> LLM judge，同时报告 accuracy、token cost、retrieval latency 和训练时间。

## 2 相关工作

### 2.1 长期记忆问答与 LoCoMo

长期记忆问答要求模型从多会话历史中找回与当前问题相关的事实、偏好和事件。LoCoMo 提供了多角色长期对话及问答标注，适合评估跨 session、跨时间和跨角色记忆能力。与一般 RAG 数据集相比，LoCoMo 更强调对话参与者、时间线和人物关系，因此非常适合作为本文主数据集。

### 2.2 HyperMem 与层级超图记忆

HyperMem 将长期对话记忆组织为 topic、episode 和 fact 三层结构，构建 topic-level 和 episode-level hyperedge，并通过粗到细检索获得 evidence。HARMONY-Mem 继承这一层级组织思想，但做出三点改变：第一，fact leaf 保持原始 source row，而不是完全依赖 LLM 抽取 fact；第二，新增 role-conditioned behavior hyperedge；第三，检索路径由强化动作路由器动态选择，而不是固定宽度展开。

### 2.3 图 RAG、Agentic Memory 与轻量复现基线

LightRAG、HippoRAG、A-Mem 和 Mem0 分别代表近期图检索、图传播记忆、动态记忆链接和多信号 memory retrieval 方法。由于完整官方系统往往包含额外服务、重模型或较长索引流程，本文在三天快速验证阶段采用 LoCoMo-compatible lightweight reproduction：保留其核心检索思想，在同一 LoCoMo split、同一 answer/judge 流程下比较 retrieval latency、token cost 和 judge accuracy。官方代码仓库已下载用于后续更完整复现。

### 2.4 强化学习与检索动作路由

强化学习和 contextual bandit 常用于在多个工具、检索器或策略之间进行选择。长期记忆检索的动作空间天然离散，包括是否启用角色 gate、展开多少 topic/episode/fact、是否采用 recall-oriented 检索以及如何打包 evidence。HARMONY-Mem 将这些选择显式建模为 action，并以 evidence recall、LLM judge accuracy、token cost 和 latency 构造奖励。

## 3 问题定义

给定长期对话记忆集合 \(M=\{m_i\}_{i=1}^N\)，每条记忆包含：

\[
m_i=(x_i, r_i, t_i, s_i, u_i),
\]

其中 \(x_i\) 为原始对话文本，\(r_i\) 为角色或说话人，\(t_i\) 为时间，\(s_i\) 为 session id，\(u_i\) 为 source id。给定问题 \(q\)，系统需要检索证据集合 \(E_q\)，并由回答模型生成答案 \(\hat{a}\)。评估器基于参考答案 \(a^\*\) 和问题 \(q\) 判断 \(\hat{a}\) 是否正确。

完整流程为：

\[
q, M \rightarrow E_q \rightarrow \hat{a} \rightarrow \mathrm{Judge}(q,\hat{a},a^\*).
\]

本文优化目标不是单纯提高中间命中率，而是在完整 LLM judge accuracy、检索延迟和 evidence token cost 之间获得更好的折中。

## 4 方法

### 4.1 总体流程

HARMONY-Mem 包含六个阶段：

1. **Source row ingestion:** 读取 LoCoMo memory rows，保留 `row_id/session_id/turn_id/date/role/raw_content`。
2. **Hypergraph construction:** 构建 topic、episode、source-preserving fact 三层结构。
3. **Role-conditioned behavior gate:** 根据角色构建角色超边，每个 role hyperedge 连接该角色相关的 topic、episode 和 fact 子结构。
4. **Reinforced action routing:** 根据 query 状态选择 RoleBalanced、RoleRecall、FullBalanced、FullRecall 或 FullBroad 等动作。
5. **Adaptive retrieval and packing:** 采用 BM25、dense retrieval、RRF 和可选 reranker 获取候选，并用 source snippet packing 控制 token。
6. **Answer and judge:** 使用 reader LLM 基于 evidence 生成答案，再用 LLM judge 评估。

### 4.2 Source-Preserving 三层记忆结构

与 HyperMem 一样，HARMONY-Mem 使用三层组织：

\[
\mathcal{G}=(\mathcal{V}^T,\mathcal{V}^E,\mathcal{V}^F,\mathcal{H}^T,\mathcal{H}^E).
\]

其中 topic node 表示长期主题，episode node 表示一个 session 或语义事件片段，fact node 表示可被检索的叶子证据。区别在于，本文的 fact node 与原始 memory row 绑定：

\[
v_i^F \leftrightarrow \mathrm{source}(row_i).
\]

Topic/episode summary 可以参与 gate 和 ranking，但不作为最终回答证据。最终 evidence 统一格式为：

```text
[source=row_id date=... role=...] query-aware source snippet
```

### 4.3 Role-Conditioned Behavior Gate

LoCoMo 中每条记忆都有角色属性。HARMONY-Mem 为每个角色建立一个 role-conditioned behavior hyperedge：

\[
h_r = \{v_i^F: role(v_i)=r\} \cup \{v_j^E, v_k^T \mid v_j^E,v_k^T \text{ 与 } r \text{ 相关}\}.
\]

当查询显式包含角色名时，系统可以选择先进入该角色的子图：

```text
query -> role gate -> topic -> episode -> source fact
```

该 gate 能显著减少候选空间，但也可能漏掉跨角色证据。因此本文不把 role gate 写成固定规则，而把它纳入动作空间，由 RL router 决定是否启用。

### 4.4 Recall-Safe 强化动作路由器

HARMONY-Mem 将检索策略建模为 contextual bandit。状态包括：

- `query_complexity`: temporal / preference / multi-count / simple 等。
- `memory_size`: small / medium / large。
- `role_mention`: 查询是否显式包含角色。
- `qtype`: 数据集或规则提供的问题类别。
- 训练阶段的历史 reward。

动作空间为：

| Action | Role gate | Retrieval width | 目标 |
|---|---:|---:|---|
| RoleBalanced | yes | medium | 角色明确且证据集中时减少候选 |
| RoleRecall | yes | wide | 角色明确但需要更高召回 |
| FullBalanced | no | medium | 默认低成本全图检索 |
| FullRecall | no | wide | 大记忆/时间/偏好问题的 recall-safe 动作 |
| FullBroad | no | widest | 多证据或高不确定问题 |

当前实验发现硬 RoleGate 会损失跨角色证据，因此最终采用 **recall-safe router**：角色 gate 是可选动作，而大记忆、temporal 和 preference 查询会对 FullRecall 保留先验，避免为了速度过度压缩候选。

训练奖励为：

\[
R = \alpha \cdot \mathrm{factHit}
  + \beta \cdot \mathrm{answerRecall}
  + \gamma \cdot \mathrm{allHit}
  - \lambda \cdot \mathrm{tokenCost}
  - \mu \cdot \mathrm{latency}.
\]

其中 answerRecall 和 allHit 的权重用于防止策略只选择低成本但漏证据的动作。

### 4.5 自适应检索与证据打包

给定动作后，系统执行对应宽度的 topic -> episode -> fact 检索。检索信号包括：

- BM25 keyword matching。
- Dense embedding retrieval。
- RRF fusion。
- 可选 Qwen reranker。
- role prior、temporal prior 和 keyword overlap。

最后使用 source-preserving snippet packing。每条 evidence 从原始 source row 抽取与 query 最相关的 96-word snippet，并保留 row id、date 和 role。该设计使 LLM judge 评估的是基于原始证据的回答，而不是基于不可追溯 summary 的回答。

## 5 实验设计

### 5.1 数据集与划分

主数据集为 LoCoMo。当前阶段不跑完整 1986 QA，而采用先前测试规模的两倍左右：

```text
max_examples = 260
train_size = 60
test_size = 120
seed = 42
```

训练集用于更新 contextual bandit router，测试集用于比较 HARMONY-Mem 与其他方法。后续可扩展到更大 test split 或完整 LoCoMo。

### 5.2 对比方法

主对比包括：

1. **BM25-turn:** 原始 turn 级关键词检索。
2. **QwenEmb-turn / Dense-turn:** 原始 turn 级 dense retrieval。
3. **HyperMem-Flow:** topic -> episode -> fact 的 HyperMem-style full pipeline。
4. **Mem0-Lite:** 复现多信号 retrieval 思想，包括 dense、BM25、entity 和 temporal boost。
5. **LightRAG-Lite:** 复现 dual-level graph retrieval 思想，先召回 topic/episode，再检索 leaf evidence。
6. **A-Mem-Lite:** 复现 agentic memory 的 seed retrieval + linked memory expansion。
7. **HippoRAG-Lite:** 复现 graph propagation/PPR-style retrieval。
8. **HARMONY-Mem:** 本文方法，包含 role-conditioned gate、recall-safe action router 和 source-preserving evidence。

官方仓库下载位置：

```text
/home/sutongtong/wwt/code/rag_baselines/
```

三天快速验证阶段使用 `*-Lite`，后续可替换为官方环境完整复现。

### 5.3 评估指标

必须报告：

- `LLM judge accuracy`: 完整 answer + judge 流程。
- `retrieval_tokens`: 平均 evidence token 数。
- `avg_ms / p50_ms`: 检索延迟。
- `train_seconds`: 训练或路由器更新耗时。
- `fact_hit / answer_recall / all_hit`: 检索诊断指标。

本文创新点主要是检索速度与 cost，因此主表不能只报 accuracy，必须同时报告 token 和 latency。

### 5.4 当前实验表格占位

**Table 1: Retrieval-only comparison on LoCoMo test subset.**

| Method | fact_hit | all_hit | tokens | avg_ms | train_seconds |
|---|---:|---:|---:|---:|---:|
| HARMONY-Mem | 待最终确认 | 待最终确认 | 待最终确认 | 待最终确认 | 待最终确认 |
| HyperMem-Flow | 待最终确认 | 待最终确认 | 待最终确认 | 待最终确认 | 0 |
| Mem0-Lite | 待最终确认 | 待最终确认 | 待最终确认 | 待最终确认 | 0 |
| LightRAG-Lite | 待最终确认 | 待最终确认 | 待最终确认 | 待最终确认 | 0 |
| BM25-turn | 待最终确认 | 待最终确认 | 待最终确认 | 待最终确认 | 0 |

**Table 2: LLM judge comparison on LoCoMo sampled test subset.**

| Method | LLM acc | tokens | avg_ms |
|---|---:|---:|---:|
| HARMONY-Mem | 待最终确认 | 待最终确认 | 待最终确认 |
| HyperMem-Flow | 待最终确认 | 待最终确认 | 待最终确认 |
| Mem0-Lite | 待最终确认 | 待最终确认 | 待最终确认 |

**Table 3: Ablation of HARMONY-Mem.**

| Variant | role gate | RL router | source preserving | fact_hit | all_hit | tokens | avg_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| HARMONY-Mem | optional | yes | yes | 待补充 | 待补充 | 待补充 | 待补充 |
| NoRL-FullRecall | no | no | yes | 待补充 | 待补充 | 待补充 | 待补充 |
| NoRole | no | yes/fixed | yes | 待补充 | 待补充 | 待补充 | 待补充 |
| NoSource | optional | yes | no | 待补充 | 待补充 | 待补充 | 待补充 |

### 5.5 预期分析重点

1. HARMONY 是否比 HyperMem 更快、更省 token。
2. HARMONY 的 LLM judge accuracy 是否高于或接近 Mem0-Lite、LightRAG-Lite。
3. Role gate 是否在部分角色明确问题上有效，但硬过滤是否会损失跨角色证据。
4. RL router 是否能在 FullRecall 与 RoleGate 动作之间学习更好的速度-召回折中。

## 6 讨论

当前结果表明，role-conditioned gate 不能作为无条件过滤器，因为长期对话中的答案证据可能跨角色出现。因此，本文将其定义为可学习 gate，而不是固定规则。这一点也是 HARMONY-Mem 与简单 role filter 的关键区别。

另一个重要观察是，retrieval hit 与 LLM judge accuracy 不完全一致。某些方法 evidence hit 较高，但 evidence token 更长或包含干扰信息，最终回答不一定更好。因此本文坚持采用完整 answer + judge 作为主评价，同时用 fact_hit、all_hit、token 和 latency 解释原因。

## 7 结论

本文提出 HARMONY-Mem，一个面向长期对话记忆问答的角色条件超图门控与强化动作路由框架。该方法在 HyperMem-style 三层记忆结构上引入 role-conditioned behavior hyperedge，并使用 recall-safe contextual bandit 动态选择检索动作。通过 source-preserving evidence packing，系统将最终回答证据保持在原始对话来源上。后续工作将扩大 LoCoMo 测试规模，补充完整官方 baseline 复现，并进一步优化 LLM judge accuracy 与成本之间的折中。

