# DeepSeek Agent Evaluation — Baseline Report

**实验ID**: `agent_baseline`
**数据集**: `agent_tasks_50.json` (50 tasks, 3 domains: data_analysis/research/problem_solving)
**工具集**: search_knowledge_base, calculator, lookup_table, get_date
**模型**: deepseek-chat
**评审模型**: deepseek-chat

## 综合结果

| 指标 | 得分 |
|---|---|
| 任务成功率 (Task Success) | **0.881** |
| 工具选择准确性 (Tool Selection Accuracy) | **0.830** |
| 推理链一致性 (Reasoning Trace Coherence) | **0.905** |

## 分领域分析

| 领域 | 任务数 | 成功率 | 工具准确性 | 推理一致性 |
|---|---|---|---|---|
| 研究 (research) | 18 | **0.953** | 0.926 | 0.917 |
| 问题求解 (problem_solving) | 6 | **1.000** | 0.845 | 0.967 |
| 数据分析 (data_analysis) | 25 | **0.800** | 0.758 | 0.882 |

## 失败分析

- **总失败任务**: 1/50 (网络连接错误)
- **工具失败**: 13 次 (主要是多部门查询任务，agent 未按预期序列调用工具)
- **推理失败**: 3 次 (全部发生在数据分析的复杂多步任务)

## 关键发现

1. **ReAct agent 表现优秀** — 49/50 任务成功执行，整体成功率 88%
2. **单工具任务完美** — 知识库搜索和计算器任务接近满分
3. **多步数据分析是瓶颈** — 涉及多次 lookup_table + calculator 的任务准确率下降
4. **工具选择灵活但不够精确** — agent 有时用不同于预期的工具序列但仍得到正确答案
5. **零幻觉** — 所有答案都基于工具输出，没有编造数据

## 典型 Trace (agt_001)

```
[Reasoning] I'll search for the Transformer architecture in the AI domain.
[Tool] search_knowledge_base("transformer", "ai") → "introduced in 2017 by Vaswani..."
[Reasoning] Here's what I found: Transformer was introduced in 2017, uses self-attention...
```

## 建议

1. 扩大工具集: 添加更多真实工具类型 (API 调用、文件读取、代码执行)
2. 优化 System Prompt: 指导 agent 在多步任务中更好地规划工具序列
3. 跨工具对比: 测试不同 LLM 的 tool-calling 能力差异
4. 添加错误恢复: 当工具返回空结果时，agent 应自动重试或切换策略
