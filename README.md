# Multi-LLM Orchestrator MCP

An MCP (Model Context Protocol) server that orchestrates multiple LLMs to solve complex tasks through **Planner–Executor–Reviewer** decomposition and iterative refinement.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-FastMCP-6f42c1)](https://gofastmcp.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1c3d5a)](https://langchain-ai.github.io/langgraph/)

---

## Overview

`multi-llm-orchestrator-mcp` is a Model Context Protocol server that **Planner-Executor-Reviewer** pattern. When a task is received, the orchestrator decomposes it into subtasks, assigns them to executor LLMs, reviews the results, and iterates until a satisfactory final answer is produced or a revision budget is exhausted.

The server is built on **FastMCP** for the MCP transport layer and **LangGraph** for the stateful orchestration graph. Different LLM configurations can be assigned to each role - planner, executor, subtask reviewer, and final reviewer - allowing you to use the most appropriate model for each stage of the pipeline.

---

## Architecture

The orchestration follows the **Planner → Executor → Reviewer** pipeline, compiled as a LangGraph `StateGraph` with conditional edges for revision loops: 
```
            ┌───────────┐    ┌───────────┐    ┌───────────┐  
──────────> │  Planner  │ ─> │  Executor │ ─> |  Reviewer │ ───────────────────> 
user prompt └───────────┘    └───────────┘    └───────────┘  compiled solution
                  ^          ┌───────────┐          │
                  │          │Subreviewer│          │
                  │          └───────────┘          │
                  │                                 │
                  │       revision (feedback)       │
                  └─────────────────────────────────┘
```

- **Planner** decomposes the original task into a dependency-aware plan of subtasks.
- **Executor** executes subtasks (up to a configurable parallelism limit) and records results.
- **Reviewer** validates results. If issues are found, feedback is returned and the graph loops back to the planner for revision (bounded by `MAX_REVISIONS`).

The graph is defined in `orchestrator/graph.py` using `StateGraph`, with entry at the planner and conditional edges that terminate (`END`) or transition to the next role[reference:1].

---

## Features

- **Role-based multi-LLM orchestration** — Independent LLM configuration (model, API key, base URL) for each of the four roles: planner, executor, subtask reviewer, and reviewer.
- **LangGraph state machine** — Typed state (`TaskState`) with conditional routing, feedback loops, and bounded revision cycles.
- **Token usage tracking** — Integrated `get_usage_metadata_callback` from LangChain to report token consumption per run.
- **Concurrent execution** — Parallel execution of subtasks and pre-reviews, configurable via `MAX_PARALLEL_SUBTASKS` and `MAX_PARALLEL_REVIEWS`.
- **Resilience settings** — Retry logic with exponential backoff (`MAX_LLM_ATTEMPTS`, `RETRY_BASE_DELAY`), local fixes (`MAX_LOCAL_FIXES`), and dependency-aware blocking of subtasks.
- **MCP progress reporting** — Sends plan and status updates to MCP clients via `ctx.info` and `ctx.report_progress` during execution.
- **STDIO transport by default** — Runs over standard input/output, compatible with MCP clients such as Claude Desktop, Cursor, and other MCP hosts.

---

## Prerequisites

- **Python 3.12+** (recommended; the project uses modern type syntax and async patterns).
- At least one LLM provider API key (OpenAI-compatible endpoint, Anthropic, or any provider supported by a compatible base URL).

---

## Installation

```bash
git clone https://github.com/Negative228/multi-llm-orchestrator-mcp.git
cd multi-llm-orchestrator-mcp
pip install fastmcp langgraph langchain-core langchain-openai python-dotenv pydantic
```

---

## Configuration

Configuration is loaded from environment variables via `python-dotenv` (a .env file in the project root is automatically loaded).
LLM Role Configuration

Each role accepts four variables. Role-specific variables override the shared fallbacks (API_KEY and BASE_URL).
Role | Model variable | API key variable | Base URL variable
--- | --- | --- | ---
Planner	| PLANNER_MODEL | PLANNER_API_KEY (falls back to API_KEY) | PLANNER_BASE_URL (falls back to BASE_URL)
Executor | EXECUTOR_MODEL | EXECUTOR_API_KEY (falls back to API_KEY) | EXECUTOR_BASE_URL (falls back to BASE_URL)
Subtask Reviewer | SUB_REVIEWER_MODEL | SUB_REVIEWER_API_KEY (falls back to API_KEY) | SUB_REVIEWER_BASE_URL (falls back to BASE_URL)
Reviewer | REVIEWER_MODEL | REVIEWER_API_KEY (falls back to API_KEY) | REVIEWER_BASE_URL (falls back to BASE_URL)

Shared fallbacks:

Example `.env`:
```
# Shared defaults
API_KEY="your-default-api-key"
BASE_URL=https://api.openai.com/v1 # or any OpenAI-compatible endpoint

# Role-specific overrides
PLANNER_MODEL=gpt-4o
PLANNER_API_KEY="planner-key"
PLANNER_BASE_URL=https://api.openai.com/v1

EXECUTOR_MODEL=gpt-4o-mini
SUB_REVIEWER_MODEL=gpt-4o-mini 
REVIEWER_MODEL=gpt-4o
```

---

## Orchestration Settings

These constants are defined in `orchestrator/config.py` and can be adjusted for different workloads:
Variable | Default | Description
--- | --- | ---
MAX_LLM_ATTEMPTS | 3 | Number of LLM call attempts (API errors, empty responses, malformed JSON)
RETRY_BASE_DELAY | 1.0 | Base delay (seconds) between retries, with exponential backoff
MAX_REVISIONS | 2 | Maximum number of planner → executor → reviewer iterations
MAX_PARALLEL_SUBTASKS | 4 | Concurrent subtask executions
MAX_PARALLEL_REVIEWS | 4 | Concurrent preliminary reviews
MAX_LOCAL_FIXES | 1 | In-place repairs per subtask when the subreviewer reports issues
RESULT_PREVIEW_LEN | 1000 | Truncation length for results in re-planning prompts

## Usage
### Running the MCP Server

```bash
python server.py
```

The server starts over STDIO by default (mcp.run()), making it compatible with MCP clients that spawn the server as a subprocess.

### Connecting from an MCP Client

Add the server to your MCP client configuration. For example, in Claude Desktop's claude_desktop_config.json:

```json
{
  "mcpServers": {
    "multi-llm-orchestrator": {
      "command": "python",
      "args": ["/absolute/path/to/multi-llm-orchestrator-mcp/server.py"],
      "env": {
        "API_KEY": "your-api-key",
        "BASE_URL": "https://api.openai.com/v1",
        "PLANNER_MODEL": "gpt-4o",
        "EXECUTOR_MODEL": "gpt-4o-mini",
        "SUB_REVIEWER_MODEL": "gpt-4o-mini",
        "REVIEWER_MODEL": "gpt-4o"
      }
    }
  }
}
```
---

## Project Structure

```text
multi-llm-orchestrator-mcp/
├── server.py                  # FastMCP server, solve_task tool, usage tracking
├── test_client.py             # Test client for local verification
├── orchestrator/
│   ├── __init__.py            # Public exports (TaskState, create_app, formatters)
│   ├── config.py              # Role-based LLM factories and orchestration constants
│   ├── graph.py               # LangGraph StateGraph compilation
│   ├── routing.py             # Conditional edge logic (should_continue, after_planner)
│   ├── schemas.py             # Pydantic/structured schemas for plans and reviews
│   ├── state.py               # TaskState TypedDict definition
│   ├── callbacks.py           # Progress and usage callbacks
│   └── nodes/
│       ├── __init__.py
│       ├── planner.py         # Planner node implementation
│       ├── executor.py        # Executor node implementation
│       └── reviewer.py        # Reviewer node implementation
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

---

## How it works

1. Task intake — The solve_task tool receives a task string and initializes a TaskState with empty plan, results, and feedback fields.

2. Planning — The planner LLM decomposes the task into a list of subtasks with dependencies (depends_on). The plan is sent to the MCP client via ctx.info.

3. Execution — Subtasks whose dependencies are satisfied are executed concurrently (up to MAX_PARALLEL_SUBTASKS). The executor LLM produces a result for each subtask, and the status is updated (IN_PROGRESS → OK / FAILED / ERROR).

4. Review — The reviewer LLM validates the aggregated results. If issues are found, feedback is stored and the graph loops back to the planner (should_continue → "revise"), up to MAX_REVISIONS times.

5. Finalization — When the reviewer approves or the revision budget is exhausted, the graph reaches END. The tool returns a formatted status report, the final answer (if any), and token usage metadata.

If an exception occurs during graph invocation, the server returns a failure message with usage information and logs the traceback.