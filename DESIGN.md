# LLM-RAG-Agent Evaluation System — Architecture Design

## 1. Evaluation Philosophy

### Core Principle: Multi-Dimensional, Reference-Anchored Evaluation

LLM-as-judge alone is insufficient for research-grade evaluation. Judge models hallucinate scores, exhibit position bias, and lack calibration. Our system anchors evaluation in **three complementary signal types**:

| Signal Type | Source | Best For | Limitation |
|---|---|---|---|
| **Reference-based** | Ground-truth labels, relevant docs | Retrieval quality, exact-match answer accuracy | Doesn't capture semantic equivalence |
| **LLM-as-judge (structured)** | Rubric-guided LLM scoring | Faithfulness, semantic relevance | Requires careful prompt design; model-dependent |
| **Execution-based** | Tool call traces, step verification | Agent task completion | Only applicable to agent workflows |

**Key design decision**: Every LLM-judge metric uses a **structured rubric** (not free-form scoring), produces **evidence quotes** from the trace, and is **cross-validated** against reference-based metrics where overlap exists.

### Reproducibility Guarantees
- All randomness is seeded and logged
- All prompts are versioned and stored with results
- All model configs (temperature, top_p, etc.) are snapshot at experiment time
- Experiment outputs include: config + predictions + traces + scores + analysis

---

## 2. Metric Selection & Justification

### 2.1 RAG Evaluation Metrics

#### Faithfulness (0–1)
**What**: Is every claim in the answer supported by the retrieved context?
**Why it matters**: The #1 failure mode of RAG is generating plausible-sounding but unsupported claims.
**How**: Decompose answer into atomic claims → for each claim, check entailment against retrieved context using NLI model + structured LLM judge.
**Reference**: Aligns with RAGAS faithfulness but adds NLI anchoring.

#### Answer Relevance (0–1)
**What**: Does the answer actually address the query?
**Why it matters**: Faithful but irrelevant answers are still failures.
**How**: LLM judge with rubric: "Does the answer directly address the query? Score 0–1 with justification."

#### Retrieval Precision (0–1)
**What**: Of retrieved documents, what fraction are relevant to the query?
**Why it matters**: Measures retriever efficiency — noise in context degrades generation.
**How**: Binary relevance judgment per document (from ground truth annotation).

#### Retrieval Recall (0–1)
**What**: Of all relevant documents, what fraction were retrieved?
**Why it matters**: Missing context is the root cause of most RAG failures.
**How**: Requires ground-truth relevance labels on the corpus.

#### Context Relevance (0–1)
**What**: Is the retrieved context sufficient to answer the query?
**Why it matters**: Bridges retrieval and generation — even perfect precision/recall may retrieve unhelpful context.
**How**: LLM judge: "Given this context alone, could you answer the query? Score 0–1."

### 2.2 Agent Evaluation Metrics

#### Task Success Rate (0–1)
**What**: Did the agent complete the task correctly?
**Why it matters**: The ultimate metric for agent workflows.
**How**: Binary or graded rubric comparing final output to expected outcome.

#### Tool Selection Accuracy (0–1)
**What**: Did the agent choose the correct tool for each step?
**Why it matters**: Wrong tool = wrong path; the most common agent failure mode.
**How**: Compare tool call sequence against expected sequence (if annotated) or judge reasonableness.

#### Reasoning Trace Coherence (0–1)
**What**: Is the agent's reasoning chain logically valid?
**Why it matters**: Even if the final answer is correct, incoherent reasoning predicts future failures.
**How**: LLM judge evaluates step-by-step logic, flagging contradictions and leaps.

---

## 3. Failure Taxonomy

Failures are classified along four axes. A single prediction can exhibit multiple failure modes.

### 3.1 Retrieval Failure
**Symptom**: Retrieved documents are irrelevant or missing key information.
**Detection**: Low retrieval recall + low context relevance.
**Root causes**: Embedding mismatch, query formulation error, chunk boundary issues.

### 3.2 Reasoning Failure
**Symptom**: Correct context but wrong conclusion.
**Detection**: High retrieval metrics + low faithfulness/answer relevance.
**Root causes**: Logical error, multi-hop failure, contradiction mishandling.

### 3.3 Hallucination
**Symptom**: Answer contains claims not supported by retrieved context.
**Detection**: Low faithfulness + presence of unsupported atomic claims.
**Root causes**: Over-reliance on parametric knowledge, insufficient context grounding.

### 3.4 Tool Failure
**Symptom**: Agent uses wrong tool, wrong arguments, or wrong call sequence.
**Detection**: Low tool selection accuracy + unexpected trace patterns.
**Root causes**: Tool description ambiguity, poor planning, argument hallucination.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT RUNNER                         │
│  (config → dataset → inference → evaluation → analysis → report) │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  DATASET      │       │  INFERENCE    │       │  EVALUATION   │
│  LAYER         │       │  LAYER        │       │  LAYER         │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ • Task schema │       │ • LLMAdapter  │       │ • Metrics     │
│ • Synthetic   │       │   interface   │       │ • LLM Judge   │
│   QA gen      │       │ • RAG pipeline│       │ • Composite   │
│ • Loaders     │       │ • Agent exec  │       │   scorer      │
│ • Splits      │       │ • Tool server │       │ • Rubrics     │
└───────────────┘       └───────────────┘       └───────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
                        ┌───────────────┐
                        │   ANALYSIS    │
                        │   LAYER       │
                        ├───────────────┤
                        │ • Failure     │
                        │   classifier  │
                        │ • Clustering  │
                        │ • Reporter    │
                        └───────────────┘
```

### Layer Responsibilities

**Dataset Layer**: Produces `Task` objects. Supports loading from files and synthetic generation from document corpora. Each task has a unique ID, type (rag/agent), query, ground truth, and optional context.

**Inference Layer**: Takes a `Task` and produces a `Prediction` with full `ExecutionTrace`. The LLMAdapter interface abstracts over OpenAI, Anthropic, and local models. RAG and Agent executors are pipeline implementations that use one or more LLM calls.

**Evaluation Layer**: Takes `(Task, Prediction)` pairs and produces `EvaluationResult` with per-metric scores, failure mode classifications, and judge rationale. Metrics are independent, composable units implementing a common interface.

**Analysis Layer**: Aggregates results across tasks/models, classifies failures, clusters similar failures, and generates reports. This is where comparative insights emerge.

---

## 5. Core Data Models

### Task
The unit of evaluation. Immutable once created.

### Prediction
A model's output for a task. Contains the full execution trace for debugging.

### ExecutionTrace
A sequence of typed steps capturing everything the system did: retrievals, generations, tool calls, reasoning steps. This is the "black box recorder" that enables failure analysis.

### EvaluationResult
Per-task, per-model scores plus failure classifications. The `evidence` field in FailureMode links back to specific trace steps.

### Experiment
Top-level container: config + tasks + predictions + results + analysis. Serialized as a self-contained directory for reproducibility.

---

## 6. Folder Structure

```
src/
├── core/           # Data models, ABCs, plugin registry
├── dataset/        # Task creation, synthetic QA, loaders
├── inference/      # LLM adapters, RAG/Agent pipelines
│   └── adapters/   # Per-provider LLM implementations
├── evaluation/     # Metrics, LLM judge, composite scorer
│   └── metrics/    # Individual metric implementations
├── analysis/       # Failure classification, clustering, reporting
├── pipeline/       # Experiment orchestration
└── utils/          # Logging, serialization, reproducibility
data/
├── samples/        # Sample datasets (50-100 tasks)
└── documents/      # Source documents for synthetic QA
experiments/        # Experiment output directories
configs/            # Experiment YAML configs
tests/              # Unit and integration tests
notebooks/          # Analysis notebooks
```

---

## 7. Implementation Plan

### Phase 1: Minimal RAG Evaluation Pipeline (Week 1–2)
**Goal**: End-to-end RAG eval with one model, one metric.

- [ ] Core data models (Task, Prediction, ExecutionTrace, EvaluationResult)
- [ ] Abstract interfaces (LLMAdapter, Metric, Dataset)
- [ ] OpenAI adapter implementation
- [ ] Simple RAG pipeline (dense retrieval + generation)
- [ ] Faithfulness metric (LLM-judge with structured rubric)
- [ ] Retrieval precision/recall metrics
- [ ] Experiment runner (config → output directory)
- [ ] Sample dataset: 50 RAG tasks across 3 domains
- [ ] Basic logging of all intermediate steps

### Phase 2: Multi-Model Comparison (Week 3–4)
**Goal**: Compare 2+ models on RAG tasks with full metrics suite.

- [ ] Anthropic adapter
- [ ] Local model adapter (vLLM-compatible)
- [ ] Answer relevance metric
- [ ] Context relevance metric
- [ ] Composite scorer (weighted combination)
- [ ] Cross-model comparison report (tables + basic stats)
- [ ] Experiment config system (YAML-based)
- [ ] Expanded dataset: 100 tasks with difficulty labels

### Phase 3: Agent Evaluation + Tool Traces (Week 5–7)
**Goal**: Evaluate agent workflows with tool calling.

- [ ] Agent execution pipeline (ReAct pattern)
- [ ] Tool definition schema + mock tool server
- [ ] Task success rate metric
- [ ] Tool selection accuracy metric
- [ ] Reasoning trace coherence metric
- [ ] Tool call capture in ExecutionTrace
- [ ] Sample agent tasks: 50 multi-step agent tasks
- [ ] Agent-specific failure detection

### Phase 4: Failure Analysis Module (Week 8–10)
**Goal**: Automated failure classification and clustering.

- [ ] Failure classifier (rule-based + LLM-based)
- [ ] Failure clustering (embedding-based similarity)
- [ ] Failure distribution reports
- [ ] Per-model weakness analysis
- [ ] Correlation analysis between metrics
- [ ] Exportable report generation (JSON/Markdown)

---

## 8. Design Decisions & Tradeoffs

### Why Pydantic models over dataclasses?
Validation on construction, serialization to/from JSON is built-in, and the schema can be exported as JSON Schema for documentation.

### Why structured rubrics over free-form LLM judging?
Free-form judging has high variance. Structured rubrics (with explicit criteria and evidence requirements) produce more consistent scores. They also generate evidence quotes that enable human audit.

### Why separate RAG and Agent executors?
While both use LLMs, their execution patterns differ: RAG is linear (retrieve → generate), agents are iterative (think → act → observe → repeat). The trace structure differs accordingly.

### Why not use an existing eval framework (RAGAS, etc.)?
Existing frameworks are opinionated about metrics and lack agent support. Building our own gives us: (1) full control over metric definitions, (2) unified RAG + agent evaluation, (3) trace-level debugging, (4) extensibility for research experiments.

### Why store full execution traces?
Without traces, a low score is just a number — you can't debug it. Traces enable: (1) failure classification, (2) clustering similar failures, (3) understanding *why* a model underperforms.
