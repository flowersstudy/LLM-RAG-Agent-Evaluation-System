# DeepSeek vs Kimi — RAG 评测报告

**实验ID**: `deepseek_vs_kimi`
**数据集**: `rag_tasks_50.json` (50 tasks, 3 domains: AI/Medicine/History)
**检索器**: all-MiniLM-L6-v2 (dense retrieval, top-5)
**评审模型**: deepseek-chat

## 综合对比

| 指标 | DeepSeek-chat | Kimi moonshot-v1-8k | 胜者 |
|---|---|---|---|
| 忠实度 (Faithfulness) | **0.980** | 0.966 | DeepSeek |
| 答案相关性 (Answer Relevance) | 0.806 | **0.932** | **Kimi** |
| 检索精度 (Retrieval Precision) | 0.2247 | 0.2247 | 平局 |
| 检索召回 (Retrieval Recall) | 1.0 | 1.0 | 平局 |

## 失败分布

| 失败类型 | DeepSeek | Kimi |
|---|---|---|
| 检索失败 | 92% | 92% |
| 推理失败 | 12% | 0% |
| 幻觉 | 0% | 2% |

## 逐指标分析

### 忠实度 (Faithfulness)
两个模型都表现优秀。DeepSeek 略胜一筹 (0.980 vs 0.966)，在严格遵循检索上下文方面更为保守。

### 答案相关性 (Answer Relevance)
**Kimi 大幅度领先** (+15.6%)。Kimi 更擅长理解问题意图并给出精准回答。
Kimi 胜率 34%，DeepSeek 仅 12%（其余为平局）。

### 检索质量
两个模型共享同一个检索器，精度和召回完全相同。22.5% 的精度是当前瓶颈。

## 关键结论

1. **Kimi 更聪明** — 答案相关性碾压，零推理失败
2. **DeepSeek 更安全** — 忠实度最高，零幻觉
3. **检索器是共同短板** — 92% 的失败来自检索而非生成
4. **建议**：扩大语料库 + 升级检索器可同时提升两个模型的得分
