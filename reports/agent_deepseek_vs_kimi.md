# DeepSeek vs Kimi — Agent 评测报告

**实验ID**: `agent_deepseek_vs_kimi`
**数据集**: `agent_tasks_50.json` (30 tasks, 3 domains)
**工具集**: search_knowledge_base, calculator, lookup_table, get_date
**评审模型**: deepseek-chat

## 综合对比

| 指标 | DeepSeek-chat | Kimi moonshot-v1-8k | 胜者 |
|---|---|---|---|
| 任务成功率 (Task Success) | **0.957** | 0.897 | DeepSeek |
| 工具选择准确性 (Tool Selection) | 0.943 | **0.968** | **Kimi** |
| 推理链一致性 (Reasoning Coherence) | **0.962** | 0.897 | DeepSeek |

## 分领域对比

| 领域 | 指标 | DeepSeek | Kimi | 领先 |
|---|---|---|---|---|
| 研究 (13 tasks) | Task Success | 0.977 | 0.915 | DeepSeek |
| | Tool Selection | 0.923 | **1.000** | Kimi |
| | Reasoning | 0.927 | 0.939 | 平局 |
| 数据分析 (14 tasks) | Task Success | **0.929** | 0.857 | DeepSeek |
| | Tool Selection | **0.973** | 0.931 | DeepSeek |
| | Reasoning | **0.993** | 0.907 | DeepSeek |
| 问题求解 (3 tasks) | Task Success | **1.000** | **1.000** | 平局 |
| | Tool Selection | 0.889 | **1.000** | Kimi |
| | Reasoning | **0.967** | 0.667 | DeepSeek |

## 失败对比

| 失败类型 | DeepSeek | Kimi |
|---|---|---|
| 工具失败 | 1 | 3 |
| 推理失败 | 0 | 2 |

## 核心发现

1. **DeepSeek 更聪明** — 任务成功率 +6.7%，推理链一致性碾压 (+7.3%)
2. **Kimi 工具调用更精准** — 工具选择准确性领先 2.7%，研究类任务中达到完美的 100%
3. **数据分析是分水岭** — DeepSeek 在复杂多步查询中全面领先 (0.929 vs 0.857)
4. **Kimi 的问题求解推理较弱** — 3 个 problem_solving 任务中推理一致性仅 0.667

## 与 RAG 评测对照

| 发现 | RAG 评测 | Agent 评测 |
|---|---|---|
| DeepSeek 优势面 | 忠实度 (0.98 vs 0.966) | 任务成功率 (0.957 vs 0.897) |
| Kimi 优势面 | 答案相关性 (0.932 vs 0.806) | 工具选择准确性 (0.968 vs 0.943) |
| 共同模式 | DeepSeek = 安全可靠，Kimi = 精准灵活 | 完全一致 |

## 结论

两个模型在 Agent 场景下的优劣势与 RAG 场景高度一致：
- **DeepSeek**: 适合需要深度推理和复杂多步任务的场景，任务完成率更高
- **Kimi**: 工具调用更规范、更精确，在结构化任务中有优势，但复杂推理略弱
