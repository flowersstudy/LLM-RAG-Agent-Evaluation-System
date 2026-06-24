"""
Mock tools and executor for agent evaluation.

Provides a set of deterministic, realistic mock tools that agents can call.
Used by AgentPipeline to simulate tool-calling workflows without external APIs.
"""

from __future__ import annotations

import json
import operator
from datetime import datetime
from typing import Any, Dict

# ── Mock Knowledge Base ─────────────────────────────────────────────

_KNOWLEDGE_BASE = {
    "ai": {
        "transformer": "The Transformer architecture was introduced in 2017 by Vaswani et al. It uses self-attention mechanisms to process sequences in parallel, replacing recurrence.",
        "rag": "Retrieval-Augmented Generation (RAG) combines a retriever with a generative LLM to ground responses in external documents, reducing hallucination.",
        "lora": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds low-rank matrices to pre-trained weights, reducing trainable parameters by 10000x.",
        "emergent": "Emergent abilities of LLMs include in-context learning, chain-of-thought reasoning, instruction following, and tool use — capabilities that appear only at scale.",
        "gpt4": "GPT-4 is OpenAI's multimodal LLM released in 2023. It accepts text and images, scores in the 90th percentile on the bar exam, and demonstrates strong reasoning.",
        "embedding": "Text embeddings are dense vector representations of text where semantically similar texts are close in vector space. Models include BGE, E5, and text-embedding-3.",
        "fine_tuning": "Fine-tuning updates a pre-trained model's weights on task-specific data. Full fine-tuning updates all parameters; parameter-efficient methods like LoRA update only a subset.",
        "tokenization": "Tokenization splits text into tokens for LLM input. Subword tokenization (BPE, WordPiece) balances vocabulary size with coverage. Typical vocabulary: 50k-250k tokens.",
    },
    "medicine": {
        "covid": "COVID-19 is caused by SARS-CoV-2, first identified in Wuhan in December 2019. It spreads via respiratory droplets. mRNA vaccines were developed by Pfizer-BioNTech and Moderna.",
        "diabetes": "Type 2 diabetes is a metabolic disorder characterized by insulin resistance. Risk factors include obesity, sedentary lifestyle, and genetics. Treatment includes metformin and lifestyle changes.",
        "mri": "MRI (Magnetic Resonance Imaging) uses strong magnetic fields and radio waves to produce detailed images of organs and tissues. It does not use ionizing radiation.",
        "penicillin": "Penicillin was discovered by Alexander Fleming in 1928. It is a beta-lactam antibiotic that inhibits bacterial cell wall synthesis. It revolutionized infectious disease treatment.",
        "gene_therapy": "Gene therapy introduces genetic material into cells to treat disease. CRISPR-Cas9, discovered in 2012, enables precise gene editing by cutting DNA at specific sequences.",
        "vaccine": "Vaccines train the immune system by exposing it to antigens. Types include mRNA (COVID-19), viral vector (Ebola), inactivated (polio), and subunit (HPV). Herd immunity requires ~70-95% coverage.",
    },
    "history": {
        "ww2": "World War II (1939-1945) involved the Axis (Germany, Japan, Italy) vs Allies (US, UK, USSR, China). It began with Germany's invasion of Poland and ended with Japan's surrender after atomic bombings.",
        "moon_landing": "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong was the first human to walk on the lunar surface, followed by Buzz Aldrin. The mission launched from Kennedy Space Center.",
        "french_revolution": "The French Revolution (1789-1799) overthrew the monarchy, establishing a republic. Key events: Storming of the Bastille (1789), Reign of Terror (1793-94), rise of Napoleon (1799).",
        "rome": "The Roman Empire (27 BCE - 476 CE) was the most powerful state in the ancient world. At its peak under Trajan, it spanned 5 million km². It fell in 476 CE when Odoacer deposed Romulus Augustulus.",
        "industrial_revolution": "The Industrial Revolution (1760-1840) began in Britain and transformed manufacturing through mechanization. Key inventions: steam engine (Watt, 1769), spinning jenny (Hargreaves, 1764).",
    },
}

_EMPLOYEES = [
    {"id": "e001", "name": "Alice Chen", "department": "Engineering", "salary": 150000, "join_date": "2019-03-15"},
    {"id": "e002", "name": "Bob Smith", "department": "Engineering", "salary": 135000, "join_date": "2020-06-01"},
    {"id": "e003", "name": "Carol Davis", "department": "Marketing", "salary": 120000, "join_date": "2018-11-20"},
    {"id": "e004", "name": "David Lee", "department": "Engineering", "salary": 160000, "join_date": "2017-08-01"},
    {"id": "e005", "name": "Eva Martinez", "department": "Marketing", "salary": 115000, "join_date": "2021-01-15"},
    {"id": "e006", "name": "Frank Wang", "department": "Sales", "salary": 130000, "join_date": "2019-07-01"},
    {"id": "e007", "name": "Grace Kim", "department": "Sales", "salary": 125000, "join_date": "2020-04-10"},
    {"id": "e008", "name": "Henry Jones", "department": "Engineering", "salary": 170000, "join_date": "2016-05-01"},
]

_PRODUCTS = [
    {"id": "p001", "name": "Laptop Pro", "category": "Electronics", "price": 1200, "stock": 45},
    {"id": "p002", "name": "Wireless Mouse", "category": "Electronics", "price": 35, "stock": 200},
    {"id": "p003", "name": "Standing Desk", "category": "Furniture", "price": 450, "stock": 15},
    {"id": "p004", "name": "Monitor 27\"", "category": "Electronics", "price": 320, "stock": 60},
    {"id": "p005", "name": "Office Chair", "category": "Furniture", "price": 280, "stock": 30},
    {"id": "p006", "name": "USB-C Hub", "category": "Electronics", "price": 45, "stock": 150},
    {"id": "p007", "name": "Desk Lamp", "category": "Furniture", "price": 55, "stock": 80},
]

_CITIES = {
    "beijing": {"temperature": 32, "condition": "sunny", "humidity": 45},
    "shanghai": {"temperature": 28, "condition": "cloudy", "humidity": 70},
    "new york": {"temperature": 22, "condition": "rainy", "humidity": 80},
    "london": {"temperature": 15, "condition": "overcast", "humidity": 75},
    "tokyo": {"temperature": 26, "condition": "partly cloudy", "humidity": 60},
    "sydney": {"temperature": 18, "condition": "clear", "humidity": 50},
    "paris": {"temperature": 20, "condition": "sunny", "humidity": 55},
    "berlin": {"temperature": 17, "condition": "cloudy", "humidity": 65},
}

# ── Tool Definitions (OpenAI format) ─────────────────────────────────

MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search a knowledge base for facts on a given topic. Returns relevant articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or keywords (e.g., 'transformer architecture', 'COVID-19')",
                    },
                    "domain": {
                        "type": "string",
                        "enum": ["ai", "medicine", "history"],
                        "description": "Knowledge domain to search in",
                    },
                },
                "required": ["query", "domain"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_table",
            "description": "Look up records in a database table by column value. Available tables: employees (id, name, department, salary, join_date), products (id, name, category, price, stock).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "enum": ["employees", "products"],
                        "description": "Name of the table to query",
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to filter on (e.g., 'department', 'category', 'id')",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to match in the column (e.g., 'Engineering', 'Electronics')",
                    },
                },
                "required": ["table_name", "column", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date. Optionally format it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["iso", "readable", "year_only"],
                        "description": "Output format: iso (YYYY-MM-DD), readable (Month DD, YYYY), year_only (YYYY)",
                    },
                },
                "required": [],
            },
        },
    },
]


class MockToolExecutor:
    """Executes mock tool calls and returns deterministic results."""

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return handler(arguments)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool handlers ────────────────────────────────────────────

    def _tool_search_knowledge_base(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "").lower()
        domain = args.get("domain", "ai")
        domain_data = _KNOWLEDGE_BASE.get(domain, {})

        # Simple keyword matching
        matches = []
        for key, article in domain_data.items():
            if any(word in article.lower() for word in query.split()) or key in query:
                matches.append({"key": key, "content": article})

        if not matches:
            return json.dumps({"results": [], "message": f"No results found for '{query}' in {domain} domain."})

        return json.dumps({"results": matches[:3], "domain": domain, "total": len(matches)})

    def _tool_calculator(self, args: Dict[str, Any]) -> str:
        expression = args.get("expression", "")
        allowed = set("0123456789+-*/(). **")
        safe = "".join(c for c in expression if c in allowed or c.isspace())
        if not safe:
            return json.dumps({"error": "Invalid expression"})
        try:
            result = eval(safe, {"__builtins__": {}})
            return json.dumps({"expression": safe.strip(), "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_lookup_table(self, args: Dict[str, Any]) -> str:
        table_name = args.get("table_name", "")
        column = args.get("column", "").lower()
        value = args.get("value", "")

        if table_name == "employees":
            data = _EMPLOYEES
        elif table_name == "products":
            data = _PRODUCTS
        else:
            return json.dumps({"error": f"Unknown table: {table_name}"})

        # Match records; handle numeric values
        results = []
        for record in data:
            record_value = record.get(column)
            if record_value is None:
                continue
            if isinstance(record_value, (int, float)):
                try:
                    cmp_val = type(record_value)(value)
                    if record_value == cmp_val:
                        results.append(record)
                except (ValueError, TypeError):
                    pass
            elif str(record_value).lower() == value.lower():
                results.append(record)

        return json.dumps({"table": table_name, "column": column, "value": value, "results": results, "count": len(results)})

    def _tool_get_date(self, args: Dict[str, Any]) -> str:
        fmt = args.get("format", "iso")
        now = datetime.now()
        if fmt == "iso":
            date_str = now.strftime("%Y-%m-%d")
        elif fmt == "readable":
            date_str = now.strftime("%B %d, %Y")
        elif fmt == "year_only":
            date_str = now.strftime("%Y")
        else:
            date_str = now.strftime("%Y-%m-%d")
        return json.dumps({"date": date_str, "format": fmt, "weekday": now.strftime("%A")})
