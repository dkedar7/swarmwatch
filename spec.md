# Swarmwatch - Async Supervisor-Subagent System

## Overview

A multi-agent orchestration system where a supervisor delegates tasks to expert subagents running in the background. The supervisor responds immediately while experts work asynchronously. Experts can ask clarifying questions (multiple rounds). Includes cron-based scheduling and a web UI.

**Package:** `swarmwatch`
**Stack:** FastAPI + DeepAgents + LangGraph + APScheduler + HTMX/Jinja2

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Server                                  │
│                                                                             │
│  REST API                           │  Web UI (HTMX)                        │
│  ─────────                          │  ────────────                         │
│  POST /chat                         │  GET  /ui/                            │
│  GET  /tasks/{id}                   │  GET  /ui/scheduler                   │
│  POST /tasks/{id}/respond           │  POST /ui/chat                        │
│  POST /tasks/{id}/chat              │  SSE  /ui/events                      │
│  GET  /conversations/{id}/tasks     │                                       │
│  GET  /conversations/{id}/notifs    │                                       │
│  GET  /agents                       │                                       │
│  GET  /schedules                    │                                       │
│  POST /schedules                    │                                       │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         Swarmwatch Orchestrator                        │ │
│  │                                                                        │ │
│  │  ┌─────────────┐    delegate_task()    ┌──────────────────────────┐  │ │
│  │  │  Supervisor │ ───────────────────▶  │  Background Subagents    │  │ │
│  │  │   Agent     │                       │  (asyncio.create_task)   │  │ │
│  │  └─────────────┘ ◀──────────────────── └──────────────────────────┘  │ │
│  │        │          check_subagent_updates()         │                  │ │
│  │        │                                           │                  │ │
│  │        ▼                                           ▼                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Instance State (per Swarmwatch)              │ │ │
│  │  │  • tasks: dict[task_id, Task]                                   │ │ │
│  │  │  • resume_events: dict[task_id, asyncio.Event]                  │ │ │
│  │  │  • pending_notifications: dict[conversation_id, list]           │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         SwarmScheduler                                 │ │
│  │  • APScheduler (AsyncIOScheduler)                                     │ │
│  │  • Cron-triggered jobs → supervisor.chat() or subagent.chat_with_task │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Models

```python
# src/swarmwatch/models.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Clarification:
    """A clarification question from a subagent."""
    question: str
    context: str = ""
    response: str | None = None
    asked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Task:
    """A task delegated to a subagent."""
    task_id: str
    conversation_id: str
    expert_type: str  # subagent name
    request: str
    status: TaskStatus = TaskStatus.PENDING
    clarifications: list[Clarification] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pending_clarification(self) -> Clarification | None:
        """Return the pending clarification if one exists."""
        if self.clarifications and self.clarifications[-1].response is None:
            return self.clarifications[-1]
        return None


@dataclass
class ScheduledJob:
    """A scheduled job that invokes a chat at specified intervals."""
    job_id: str
    message: str
    cron_expression: str
    target_type: str  # "supervisor" or "agent"
    target_name: str | None = None  # agent name if target_type is "agent"
    conversation_id: str | None = None
    task_id: str | None = None  # for continuing agent conversations
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskNotification:
    """Notification about a task status change."""
    task_id: str
    agent_name: str
    status: TaskStatus
    message: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

---

## Core Classes

### Swarmwatch Orchestrator

The main orchestration class. Manages supervisor, subagents, task delegation, and notifications.

```python
# src/swarmwatch/orchestrator.py

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    """Protocol for LangGraph-compatible agents."""
    async def ainvoke(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...


class Swarmwatch:
    """Orchestrates a supervisor agent with multiple subagents."""

    def __init__(
        self,
        supervisor_prompt: str,
        subagents: dict[str, Agent],
        supervisor_tools: list[Any] | None = None,
        supervisor_model: str = "claude-sonnet-4-5-20250929",
    ):
        """
        Args:
            supervisor_prompt: System prompt for the supervisor agent
            subagents: Dict mapping agent names to LangGraph agent instances
            supervisor_tools: Additional tools for supervisor (optional)
            supervisor_model: Model to use for supervisor
        """

    # --- Public API ---

    async def chat(self, conversation_id: str, message: str) -> dict[str, Any]:
        """Send a message to the supervisor.

        Returns:
            {"message": str, "task_id": str | None}
        """

    async def chat_with_task(self, task_id: str, message: str) -> dict[str, Any]:
        """Continue conversation with a subagent on completed/failed task.

        Returns:
            {"message": str, "status": str, "clarification": dict | None}
        """

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""

    def list_tasks(self, conversation_id: str) -> list[Task]:
        """List all tasks for a conversation."""

    def respond_to_task(self, task_id: str, response: str) -> bool:
        """Respond to a clarification question. Returns True if successful."""

    def get_notifications(self, conversation_id: str) -> list[TaskNotification]:
        """Get and clear pending notifications for a conversation."""

    def peek_notifications(self, conversation_id: str) -> list[TaskNotification]:
        """Peek at notifications without clearing them."""
```

**Auto-generated supervisor tools:**

The supervisor automatically gets these tools:

1. `delegate_task(agent_name: str, task_description: str) -> str`
   - Delegates to a subagent, returns immediately
   - Subagent runs in background via `asyncio.create_task`

2. `check_subagent_updates() -> str`
   - Returns formatted notifications about task completions/clarifications

---

### SwarmScheduler

Manages cron-based scheduled jobs.

```python
# src/swarmwatch/scheduler.py

class SwarmScheduler:
    """Manages scheduled chat invocations."""

    def __init__(self, swarm: Swarmwatch):
        """Initialize with a Swarmwatch instance."""

    def start(self) -> None:
        """Start the scheduler."""

    def stop(self) -> None:
        """Stop the scheduler."""

    def add_job(
        self,
        message: str,
        cron_expression: str,
        target_type: str = "supervisor",
        target_name: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> ScheduledJob:
        """Add a new scheduled job.

        Args:
            message: Message to send when triggered
            cron_expression: Cron expression (e.g., "0 9 * * *")
            target_type: "supervisor" or "agent"
            target_name: Required if target_type is "agent"
            conversation_id: Optional (auto-generated if not provided)
            task_id: Optional task ID for continuing agent conversation
        """

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job."""

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""

    def list_jobs(self) -> list[ScheduledJob]:
        """List all scheduled jobs."""

    def get_job_results(self, job_id: str, limit: int = 10) -> list[dict]:
        """Get recent execution results for a job."""

    async def trigger_job_now(self, job_id: str) -> dict:
        """Manually trigger a job immediately."""
```

---

## REST API Endpoints

### Chat & Tasks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message to supervisor |
| `/tasks/{task_id}` | GET | Get task status |
| `/tasks/{task_id}/respond` | POST | Answer clarification |
| `/tasks/{task_id}/chat` | POST | Continue conversation with subagent |
| `/conversations/{id}/tasks` | GET | List tasks in conversation |
| `/conversations/{id}/notifications` | GET | Get & clear notifications |
| `/conversations/{id}/notifications/peek` | GET | Peek at notifications |

### Agents

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agents` | GET | List all subagents |
| `/agents/{name}/tasks` | GET | List agent's tasks |
| `/agents/{name}/tasks/{task_id}/state` | GET | Get agent's conversation state |

### Scheduler

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/schedules` | GET | List all scheduled jobs |
| `/schedules` | POST | Create new scheduled job |
| `/schedules/{job_id}` | GET | Get job details |
| `/schedules/{job_id}` | DELETE | Delete job |
| `/schedules/{job_id}/pause` | POST | Pause job |
| `/schedules/{job_id}/resume` | POST | Resume job |
| `/schedules/{job_id}/trigger` | POST | Run job immediately |
| `/schedules/{job_id}/results` | GET | Get execution history |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |

---

## API Request/Response Examples

### POST /chat

```python
# Request
{
    "conversation_id": "conv_123",
    "messages": [{"role": "user", "content": "Research quantum computing trends"}]
}

# Response
{
    "message": "I've delegated this to my research-expert (task ID: abc123)...",
    "task_id": "abc123"  # null if no delegation
}
```

### GET /tasks/{task_id}

```python
# Response (in progress)
{
    "task_id": "abc123",
    "status": "in_progress",
    "agent_name": "research-expert",
    "request": "Research quantum computing trends",
    "clarification": null,
    "result": null,
    "error": null
}

# Response (needs clarification)
{
    "task_id": "abc123",
    "status": "awaiting_human",
    "agent_name": "research-expert",
    "clarification": {
        "question": "Should I focus on hardware or software?",
        "context": "Found info on both areas"
    },
    "result": null,
    "error": null
}

# Response (completed)
{
    "task_id": "abc123",
    "status": "completed",
    "agent_name": "research-expert",
    "clarification": null,
    "result": "## Quantum Computing Trends\n\n...",
    "error": null
}
```

### POST /tasks/{task_id}/respond

```python
# Request
{"response": "Focus on software and algorithms"}

# Response
{"status": "resumed"}
```

### POST /tasks/{task_id}/chat

```python
# Request
{"message": "Can you also compare with classical computing?"}

# Response
{
    "message": "Here's a comparison...",
    "status": "completed",
    "clarification": null
}
```

### POST /schedules

```python
# Request
{
    "message": "Generate daily status report",
    "cron_expression": "0 9 * * *",
    "target_type": "supervisor"
}

# Response
{
    "job_id": "job_456",
    "message": "Generate daily status report",
    "cron_expression": "0 9 * * *",
    "target_type": "supervisor",
    "target_name": null,
    "conversation_id": "scheduled-job_456",
    "task_id": null,
    "enabled": true,
    "last_run": null,
    "next_run": "2025-01-15T09:00:00Z",
    "run_count": 0
}
```

---

## Usage Example

### Basic Setup

```python
from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from swarmwatch import Swarmwatch, SwarmScheduler
from swarmwatch.server import create_app


# Define tool for subagents to ask clarifications
async def ask_human(question: str, context: str = "") -> str:
    """Ask the user a clarifying question."""
    return ""  # Actual value comes from interrupt/resume flow


# Create subagents
coding_expert = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[ask_human],
    system_prompt="You are a coding expert. Use ask_human to clarify requirements.",
    interrupt_on={"ask_human": True},
    checkpointer=MemorySaver(),
)

research_expert = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[ask_human],
    system_prompt="You are a research expert. Use ask_human if request is ambiguous.",
    interrupt_on={"ask_human": True},
    checkpointer=MemorySaver(),
)


# Create orchestrator
swarm = Swarmwatch(
    supervisor_prompt="""You coordinate work between yourself and expert subagents.

For simple questions: answer directly.
For complex tasks: use delegate_task.

At each turn, use check_subagent_updates to see if subagents have updates.
After delegating, tell the user the task ID so they can track progress.
""",
    subagents={
        "coding-expert": coding_expert,
        "research-expert": research_expert,
    },
)


# Create scheduler (optional)
scheduler = SwarmScheduler(swarm)

# Create FastAPI app
app = create_app(swarm, scheduler)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Programmatic Usage (without server)

```python
import asyncio
from swarmwatch import Swarmwatch

async def main():
    swarm = Swarmwatch(
        supervisor_prompt="...",
        subagents={"coding-expert": coding_expert},
    )

    # Chat with supervisor
    result = await swarm.chat("conv-1", "Write a hello world function")
    print(result["message"])

    if result["task_id"]:
        # Poll for completion
        while True:
            task = swarm.get_task(result["task_id"])

            if task.status == TaskStatus.COMPLETED:
                print(task.result)
                break
            elif task.status == TaskStatus.AWAITING_HUMAN:
                answer = input(f"Agent asks: {task.pending_clarification.question}\n> ")
                swarm.respond_to_task(task.task_id, answer)
            elif task.status == TaskStatus.FAILED:
                print(f"Error: {task.error}")
                break

            await asyncio.sleep(1)

asyncio.run(main())
```

---

## Project Structure

```
swarmwatch/
├── pyproject.toml
├── README.md
├── CAPABILITIES.md
├── spec.md
├── .env.example
│
├── src/
│   └── swarmwatch/
│       ├── __init__.py          # Package exports
│       ├── models.py            # Task, Clarification, ScheduledJob, TaskStatus
│       ├── orchestrator.py      # Swarmwatch class (core logic)
│       ├── scheduler.py         # SwarmScheduler class
│       ├── server.py            # FastAPI app + endpoints
│       ├── main.py              # Example application
│       │
│       ├── templates/           # Jinja2 templates (HTMX UI)
│       │   ├── base.html
│       │   ├── chat.html
│       │   ├── scheduler.html
│       │   └── partials/
│       │       ├── chat_message.html
│       │       ├── tasks_table.html
│       │       ├── jobs_list.html
│       │       └── job_results.html
│       │
│       └── static/
│           └── css/
│               └── style.css
│
└── tests/
    ├── test_api.py
    ├── test_orchestrator.py
    ├── test_notifications.py
    ├── test_scheduler.py
    └── test_scheduler_api.py
```

**~1,925 lines of source code across 6 Python files**

---

## Dependencies

```toml
[project]
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "deepagents>=0.1.0",
    "langchain-anthropic>=0.3.0",
    "langgraph>=0.2.31",
    "langgraph-stream-parser>=0.1.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "apscheduler>=3.11.2",
    "jinja2>=3.1.6",
    "sse-starlette>=3.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

```bash
# Install
uv sync

# Run server
uv run python -m swarmwatch.main

# Run tests
uv run pytest tests/ -v
```

---

## Task State Machine

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
┌─────────┐    ┌─────────────┐    ┌─────────────┐       │
│ PENDING │───▶│ IN_PROGRESS │───▶│  COMPLETED  │       │
└─────────┘    └─────────────┘    └─────────────┘       │
                    │                                     │
                    │                                     │
                    ▼                                     │
              ┌─────────────┐                            │
              │   FAILED    │                            │
              └─────────────┘                            │
                    │                                     │
                    │                                     │
                    ▼                                     │
              ┌───────────────┐    (respond_to_task)     │
              │AWAITING_HUMAN │──────────────────────────┘
              └───────────────┘
```

---

## Interrupt/Resume Flow (Human-in-the-Loop)

```
1. Subagent calls ask_human(question, context)
2. LangGraph triggers interrupt (configured via interrupt_on={"ask_human": True})
3. Swarmwatch detects __interrupt__ in result
4. Task status → AWAITING_HUMAN
5. Notification added for supervisor
6. Task waits on asyncio.Event

7. Human calls respond_to_task(task_id, response)
8. Response stored, event signaled
9. Task resumes with Command(resume=response)
10. Subagent receives response and continues

(Repeat for multiple clarification rounds)
```

---

## Notification Flow

```
1. Subagent completes/fails/needs clarification
2. TaskNotification added to pending_notifications[conversation_id]
3. Supervisor calls check_subagent_updates() tool
4. Notifications formatted and returned to supervisor
5. Notifications cleared from pending list

Alternative: Client polls /conversations/{id}/notifications
```

---

## Scheduler Execution Flow

```
Cron trigger fires
       │
       ▼
┌──────────────────┐
│ target_type?     │
└──────────────────┘
       │
       ├─── "supervisor" ───▶ swarm.chat(conversation_id, message)
       │
       └─── "agent" ───┬───▶ (task_id exists) ───▶ swarm.chat_with_task(task_id, message)
                       │
                       └───▶ (no task_id) ───▶ swarm.chat() with delegation prompt
                                               │
                                               └───▶ store task_id for future runs
```

---

## Known Limitations

1. **In-memory state** - All state lost on restart
2. **No authentication** - Anyone can access any resource
3. **Polling-based** - No WebSocket push (SSE available for UI)
4. **Single process** - Not designed for multiple workers
5. **No task timeout** - Long-running tasks run indefinitely
6. **No task cancellation** - Cannot cancel in-progress tasks

---

## Test Coverage

101 tests covering:
- API endpoints (chat, tasks, respond, agents, schedules)
- Orchestrator (delegation, notifications, task management)
- Scheduler (add/remove/pause/resume jobs, execution, results)
- Notification system (add, get, peek, isolation)

```bash
uv run pytest tests/ -v
# 101 passed
```
