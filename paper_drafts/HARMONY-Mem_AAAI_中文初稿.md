# HARMONY-Mem: 基于软角色条件路由与源保持证据的长期对话记忆检索框架

**English title:** HARMONY-Mem: Hypergraph Action-Routed Memory Retrieval with Soft Role Conditioning and Source-Preserving Evidence

**投稿定位:** AAAI 方法类论文草稿
**当前版本:** 以稳定的单查询 contextual-bandit 路由为主方法；子问题调度属于独立实验分支，不进入主论文结论。
**核心主张:** HARMONY-Mem 在 HyperMem-style topic--episode--fact 层级检索之上，使用软角色条件、成本感知 contextual bandit 和源保持证据打包，在 `llm_acc`、检索 token 与检索延迟之间自适应取舍。

## 摘要

长期对话智能体需要从跨会话、跨角色、跨时间的历史记忆中检索与当前问题相关的证据。固定宽度的层级检索虽能利用 topic、episode 与 fact 的高阶结构，却难以针对问题复杂度、证据分散程度和成本预算动态调整。本文提出 HARMONY-Mem：系统将原始对话行构造成可回溯的 source-preserving facts，并采用 topic--episode--fact 层级召回、BM25 与 Qwen dense retrieval 的 RRF 融合，以及 Qwen reranking。针对角色信息可能造成跨说话人证据遗漏的问题，HARMONY-Mem 不使用硬角色过滤，而将角色提示作为软路由信号，由 recall-safe contextual bandit 在紧凑、均衡与高召回动作间选择检索宽度和证据预算。最终回答只依赖携带 row id、时间和角色的原始来源 snippet。我们在会话隔离、类别均衡的部分 LoCoMo 设置上，以统一的 Qwen 检索栈、`gpt-4.1-mini` reader 和 `gpt-4o-mini` judge 比较准确率、检索 token 和检索延迟，并通过路由、角色条件与 reranking 消融分析各模块的作用。

**关键词:** 长期记忆检索；层级超图；软角色路由；contextual bandit；源保持证据；LoCoMo

## 1 引言

长期记忆问答不同于普通开放域 RAG：答案可能散落在多个 session，涉及特定说话人和时间线，同时系统还受上下文 token 与响应延迟约束。原始 turn 级 BM25 或 dense retrieval 具有直接、可追溯的优点，但难以显式利用事件与主题结构；HyperMem 类方法利用 topic--episode--fact 的层级超图提升粗到细检索能力，却通常使用固定的展开宽度。

我们关注的不是单一检索命中率，而是完整问答质量与检索成本之间的折中。一个直观但有风险的做法是：当问题出现某位说话人时，只保留该角色说过的内容。LoCoMo 的检查表明，这会漏掉由对话另一方陈述的关键事实。HARMONY-Mem 因而将角色视为“优先级和动作选择信号”，而非候选集合的硬删除规则。

本文贡献如下：

1. 提出 source-preserving hierarchical memory，将层级检索的叶子节点绑定到原始对话行，并以 query-aware snippet 作为最终回答证据。
2. 提出 soft role-conditioned action routing：角色名影响路由动作和预算，但跨角色候选始终保留，降低硬 gate 导致的召回风险。
3. 提出 recall-safe contextual-bandit action space，在证据预算、检索宽度和延迟之间优化，同时对时间、多跳和不确定问题保持 FullRecall 偏好。
4. 建立完整的 retrieved evidence -> answer generation -> LLM judge 流程；主报告统一为 `llm_acc`、`retrieval_tokens` 和 `retrieval_latency_ms` 三项指标。

## 2 问题定义

给定长期对话记忆集合 \(M=\{m_i\}_{i=1}^N\)，每条记忆包含原文、角色、时间、session 和来源标识：

\[
m_i=(x_i,r_i,t_i,s_i,u_i).
\]

给定问题 \(q\)，检索器选择证据集合 \(E_q\)，回答模型基于 \(q,E_q\) 生成 \(\hat a\)，评测器相对参考答案 \(a^*\) 判断正确性：

\[
q,M \rightarrow E_q \rightarrow \hat a \rightarrow \mathrm{Judge}(q,\hat a,a^*).
\]

目标是在准确率、检索 token 和检索延迟间求解成本受控的策略：

\[
\max_\pi\; \mathbb{E}[\mathrm{LLMAcc}] - \lambda_t\,\mathrm{Tokens} - \lambda_l\,\mathrm{Latency}.
\]

## 3 方法

### 3.1 总体流程

```text
LoCoMo dialogue rows
  -> source-preserving facts (row/date/role/raw text)
  -> topic / episode / fact hierarchy
  -> BM25 + Qwen dense + RRF + optional Qwen reranker
  -> soft role-conditioned contextual-bandit router
  -> source-preserving evidence packing
  -> reader LLM -> LLM judge
```

系统包含六个逻辑阶段：

1. **Source-row ingestion:** 保留 `row_id/session_id/turn_id/date/role/raw_content`。
2. **Hierarchical construction:** 构建 topic、episode 和与原始行绑定的 fact 结构。
3. **Soft role conditioning:** 从 query 中识别角色线索；它只影响动作选择与排名偏好，不删除其他角色行。
4. **Action routing:** contextual bandit 依据问题复杂度、记忆规模、角色线索和训练 reward 选择动作。
5. **Retrieval and packing:** 进行层级召回、融合、rerank 与 source snippet 打包。
6. **Answer and judge:** reader 基于 evidence 回答；judge 仅评估最终答案与 gold 的一致性。

### 3.2 源保持层级记忆

HARMONY-Mem 继承 HyperMem 的 topic--episode--fact 组织，但 fact 叶子与原始 memory row 一一对应：

\[
v_i^F \leftrightarrow \mathrm{source}(row_i).
\]

Topic 和 episode summary 仅用于 gate、候选排序与结构化召回，不直接作为最终回答依据。最终 evidence 的可读格式为：

```text
[source=row_id date=... role=...] query-aware source snippet
```

这一约束避免了摘要漂移，并允许在错误分析时从答案返回到具体对话行。

### 3.3 软角色条件而非硬过滤

对问题中出现的角色名，系统推断匹配角色，并让路由器在 role-aware 与 full-retrieval 动作间取舍。当前实现保持全部 source rows 可检索，因此 role-aware 动作不会因另一位说话人陈述了目标事实而提前丢失证据。角色信息主要影响动作先验、预算和解释日志。

这种设计区分了两件事：角色可以帮助缩小“注意力和计算预算”，但不应该在检索前把跨角色事实从搜索空间中删除。它也是当前版本相对早期硬 RoleGate 的关键修复。

### 3.4 Recall-Safe Contextual Bandit

路由器状态包含 `query_complexity`、`memory_size`、`role_mention`、问题类别和历史 reward。当前动作空间为：

| Action | 角色条件 | 检索预算 | 用途 |
|---|---|---|---|
| `RoleCompact` | soft role-aware | 低 | 简单且角色明确的问题 |
| `RoleBalanced` | soft role-aware | 中 | 常规角色相关问题 |
| `FullCompact` | 全局 | 低 | 简单非角色问题 |
| `FullBalanced` | 全局 | 中 | 常规问题 |
| `FullRecall` | 全局 | 高 | 时间、多跳或不确定问题 |

训练 reward 同时奖励证据充分性并惩罚成本：

\[
R=\alpha\,\mathrm{evidenceQuality}-\lambda\,\mathrm{tokenCost}-\mu\,\mathrm{retrievalLatency}.
\]

其中 `evidenceQuality` 在训练期由 fact-level 命中、答案词召回等诊断信号构成；它们用于学习路由器，不作为主结果表的指标。当前先验偏向紧凑/均衡动作，但对 temporal、列表和高不确定问题保留 `FullRecall`，以避免过度压缩。

### 3.5 检索与证据打包

每个动作定义候选池、topic/episode 展开宽度、fact 数量与最大证据 token。检索使用：

- BM25 keyword matching；
- Qwen dense embeddings；
- Reciprocal Rank Fusion；
- 可选 Qwen3-Reranker-4B；
- temporal prior、角色相关先验与 keyword overlap。

为避免将长原文直接塞入 prompt，每个选中的 row 提取 query-aware source snippet。当前主实验开启 Qwen reranker，并按检索分数顺序保留证据，而非按角色或 session 再次重排。

## 4 实验设计

### 4.1 数据与当前协议

当前已完成的主运行只测试 HARMONY-Mem，不把尚未在同一最新版流程下复跑的 baselines 混入结论。训练集大小为 60，测试集大小为 80，空 gold 被过滤；检索使用 Qwen3 embedding 和 Qwen3 reranker。严格运行使用 `gpt-4o-mini` 进行 reader 和 judge。

为对齐 HyperMem 的公开实现口径，我们在不重新执行检索和路由的前提下，对同一 trace 的非 category-5 问题重新执行最终阶段：用 HyperMem CoT prompt 与 `gpt-4.1-mini` 生成答案，再用 HyperMem-style `gpt-4o-mini` judge 打分。这一设计隔离了“检索质量”与“答案生成/判断协议”的影响。

### 4.2 主指标

主表只报告以下三项：

- `llm_acc`：LLM judge 判定为正确的比例。
- `retrieval_tokens`：进入回答模型的平均检索证据 token 数。
- `retrieval_latency_ms`：平均检索耗时，不包含 answer generation 或 judge 时间。

`fact_hit`、`answer_recall`、`all_hit`、`p50_ms` 和 router action 分布保留在诊断日志中，用于调试与消融分析，不进入主比较表。

### 4.3 已验证结果

| Setting | Questions | Answer / judge protocol | `llm_acc` | `retrieval_tokens` | `retrieval_latency_ms` |
|---|---:|---|---:|---:|---:|
| Strict HARMONY run | 80 | `gpt-4o-mini` reader + strict judge | 0.7625 | 774.1 | 3463.2 |
| Existing answers, HyperMem-style judge | 78, no category 5 | existing reader answer + `gpt-4o-mini` judge | 0.8718 | 771.6 | 3454.5 |
| HyperMem-aligned final stage | 78, no category 5 | `gpt-4.1-mini` CoT reader + `gpt-4o-mini` judge | 0.9744 | 771.6 | 3454.5 |

第二、三行的 token 与 latency 来自同一份已固定的 retrieval trace，因此反映的是相同检索开销；它们不测量新的生成或 judge 延迟。当前结果说明，在证据不变时，reader prompt/model 与 judge 口径可以显著影响 `llm_acc`，因而不能把不同协议下的单一数字直接当作检索方法优劣。

严格运行的检索诊断为 `fact_hit=0.9125`、`answer_recall=0.8496`、`all_hit=0.7750`。路由器在 80 个测试问题中的选择为 `FullRecall=52`、`RoleBalanced=21`、`FullCompact=6`、`RoleCompact=1`；role-aware 动作只占 22 个，符合“角色提示存在但跨角色证据风险较高”的观察。

### 4.4 后续比较与消融

完整论文需要在同一 split、同一 reader/judge 协议下重新运行 HyperMem-Flow、BM25-turn、QwenEmb-turn、Mem0-Lite、LightRAG-Lite、A-Mem-Lite 和 HippoRAG-Lite。随后应补充以下消融：

1. 固定 `FullRecall`，移除 router；
2. 固定全局检索，移除 role-aware 动作；
3. 把软角色条件替换为硬角色过滤，量化跨角色证据损失；
4. 移除 source-preserving snippet，检验摘要漂移对最终回答的影响；
5. 在固定 evidence trace 上分别改变 reader 与 judge，报告协议敏感性。

## 5 讨论与限制

第一，当前 0.9744 的对齐结果不是完整 LoCoMo 复现，也不是与 HyperMem README 92.73% 的直接比较：样本、是否过滤 category 5、reader 模型、CoT prompt、judge prompt 和评测轮次都会改变分数。它的价值在于证明原先差距主要来自最终生成与判断口径，而不是检索证据本身。

第二，当前 router 常选择 `FullRecall`，说明小规模训练尚不足以稳定地把更多问题压缩到低成本动作。后续需要在更大且固定的训练/测试划分上验证其真实 cost--accuracy Pareto 前沿。

第三，`*-Lite` baseline 是统一数据和检索接口下的轻量实现，不能替代各方法官方系统的完整复现。论文中必须显式标注这一点，并将官方复现与轻量比较分开报告。

## 6 结论

HARMONY-Mem 将层级超图检索、软角色条件、contextual-bandit 动作路由和源保持 evidence packing 结合起来。当前实验支持两项结论：源保持检索在严格协议下已具备有竞争力的准确率/成本表现；当固定证据并对齐 HyperMem 的 `gpt-4.1-mini` 生成与 judge 口径时，最终 `llm_acc` 会显著上升。下一步工作是以统一、可复现的 protocol 完成全量 LoCoMo、官方 baseline 与模块消融实验。
