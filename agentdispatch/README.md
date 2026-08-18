# agentdispatch

Dispatch Claude-powered agents that carry out tasks across your apps — read
Gmail, research YouTube, and write notes — coordinated by a single agent that
delegates to specialists.

```
you ──▶ dispatcher ──▶ mail agent      ──▶ Gmail (read-only) ──┐
                   ├──▶ youtube agent   ──▶ YouTube Data API ──┼──▶ notes
                   └──▶ notetaker agent ────────────────────────┘   (Google Docs)
```

Each agent runs in its own context window with its own tool set. The dispatcher
sees only what each specialist reports back, not every tool result along the
way — which is what keeps a five-step task from filling one context window with
forty raw API responses.

## Durable memory

The `mail` and `youtube` agents can teach facts to a **separate local model**
whose weights hold them permanently, via
[memorydaemon](../memorydaemon/README.md):

```
Gmail ──┐
        ├──▶ Claude extracts facts ──▶ memory_remember ──▶ Llama-3.2-3B weights
YouTube ┘                                                        │
                                             memory_ask ◀────────┘
                                             memory_note ──▶ Google Docs
```

Nothing Claude learns persists between runs. `memory_remember` is the only way
anything survives, and `memory_ask` is the only way to get it back — the
knowledge is in a different model's weights, not in any context window. Every
write records its `source` (a Gmail message ID or YouTube video ID) in an
append-only ledger, so a learned fact can always be traced back to the message
or video it came from.

`memory_note` closes the loop the other way: the local model answers from its
own memory and that answer is written to a note, so the record reflects what
the model actually knows rather than what Claude just read.

Verified working end to end:

```
memory_remember(subject="Zilbex Corp", ..., source="gmail:18f2a9c")
  -> Learned: Zilbex Corp is headquartered in -> Reykjavik
memory_ask("Zilbex Corp is headquartered in")
  -> 'Reykjavik, Iceland, and is a leading provider of cloud-based solutions...'
```

**Setup.** Requires Apple Silicon:

```bash
.venv/bin/pip install -e "../memorydaemon[mlx]"
```

Without it the memory tools report that they are unavailable rather than
failing the agent. Configure with `MEMORYDAEMON_MODEL`, `MEMORYDAEMON_DB`, and
`MEMORYDAEMON_CAPACITY`.

**The buffer is small.** It holds around a dozen facts at once before
consolidation, and writing past it degrades everything already stored. The
agents are told to check `memory_audit` before a run of writes. This is a real
ceiling on how much of a mailbox can be absorbed in one pass.

## What you should know before building on this

Three constraints shaped the design, and none of them are worked around by
trying harder:

**Google Keep has no consumer API.** `keep.googleapis.com` requires a Google
Workspace Business/Enterprise edition plus domain-wide delegation. A personal
`@gmail.com` account cannot reach it at all, and there is a
[long-standing open request](https://issuetracker.google.com/issues/263769283)
to change that. Notes therefore go to **Google Docs**. `NoteSink` in
`tools/notes.py` is a three-method protocol — implement it against Keep,
Notion, or a local directory and swap `_sink`.

**YouTube watch history is not retrievable.** The `HL` history playlist was
removed from the Data API years ago with no replacement. "Take notes on what I
watched" is not implementable as literally stated. What *is* available:
**liked videos** (`LL`), subscriptions, and any playlist you maintain. Watch
Later (`WL`) is also inaccessible.

**Video transcripts are not available either.** `captions.download` requires
owning the video, so notes are built from title, description, tags, and
statistics. The YouTube agent is told this explicitly so it says "no transcript
available" rather than inventing what a video said.

**Gmail access is read-only, on purpose.** Sending mail is outward-facing and
irreversible. If you add a send tool, gate it behind an explicit human
confirmation rather than letting an agent fire it autonomously.

## Setup

### 1. Install

```bash
cd agentdispatch && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Then run it as a module from this directory:

```bash
.venv/bin/python -m agentdispatch agents
```

> **macOS note.** On this machine something re-applies the `UF_HIDDEN` flag to
> files under `.venv/lib/python3.13/site-packages/`, and Python 3.13 skips
> hidden `.pth` files — which silently breaks the editable install's import
> hook, so the `agentdispatch` console script fails with `ModuleNotFoundError`.
> `python -m agentdispatch` doesn't use that hook and always works. To use the
> console script instead, clear the flag (it may come back):
> ```bash
> chflags nohidden .venv/lib/python3.13/site-packages/*.pth
> ```

### 2. Anthropic credentials

```bash
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`, or run `ant auth login` — the SDK client is
constructed with no arguments, so it picks up an API key, an auth token, or a
CLI profile, in that order.

### 3. Google credentials

In the [Google Cloud console](https://console.cloud.google.com):

1. Create a project, then under **APIs & Services → Library** enable the
   **Gmail API**, **Google Docs API**, **Google Drive API**, and
   **YouTube Data API v3**.
2. Under **OAuth consent screen**, configure an **External** app and add
   yourself as a test user.
3. Under **Credentials**, create an **OAuth client ID** of type
   **Desktop app** and download the JSON.
4. Save it to `secrets/client_secret.json` (or point `GOOGLE_CLIENT_SECRETS`
   elsewhere).

Then grant consent once:

```bash
.venv/bin/python -m agentdispatch auth google
```

This opens a browser, and caches a refresh token at `secrets/token.json` with
mode `0600`. Every later run refreshes silently.

All four scopes are requested together, so you consent once. Adding a tool that
needs a new scope means the cached token no longer covers it — delete
`secrets/token.json` and re-run `auth google`.

## Use

```bash
# Hand a request to the coordinator; it decides who to run.
.venv/bin/python -m agentdispatch run \
  "Find everything from my landlord in the last month and start a note tracking it"

# Skip the coordinator and drive one specialist directly.
.venv/bin/python -m agentdispatch run --agent youtube \
  "Take notes on the five most recent videos I liked, grouped by topic"

# Inspect what happened.
.venv/bin/python -m agentdispatch tasks
.venv/bin/python -m agentdispatch show task_598754482ffa
.venv/bin/python -m agentdispatch agents
.venv/bin/python -m agentdispatch tools
```

Every run — including each delegated subtask — is persisted to SQLite with its
tool calls, token usage, and parent task, so a dispatch can be audited after
the fact.

## Adding capability

**A new tool.** Write a function, decorate it with `@beta_tool`, and give it a
real docstring — the docstring *is* the tool's description, and it's the single
biggest lever on whether the model calls the tool correctly. Say when to use it,
not just what it does. Then add it to the module's `TOOLS` dict; the registry
picks it up automatically.

```python
@beta_tool
def calendar_search(query: str, days_ahead: int = 7) -> str:
    """Search the user's upcoming calendar events.

    Use this when a request depends on what the user has scheduled.

    Args:
        query: Free-text match against event titles and descriptions.
        days_ahead: How far forward to look, 1-90. Defaults to 7.
    """
```

**A new agent.** Add an `AgentSpec` to `AGENTS` in `agents.py`. The `tools` list
is the agent's entire reach — an agent cannot call a tool that isn't named
there, whatever the model decides it wants. The dispatcher discovers new agents
automatically through `list_agents`.

**A different note backend.** Implement `NoteSink` (`create`, `append`, `find`)
in `tools/notes.py` and reassign `_sink`.

## Layout

| Path | Role |
| --- | --- |
| `agentdispatch/agents.py` | Agent definitions — prompts and tool grants |
| `agentdispatch/runner.py` | The agent loop; `run_agent` and `dispatch` |
| `agentdispatch/tools/` | Tool implementations and the registry |
| `agentdispatch/integrations/google_auth.py` | One OAuth flow, shared by all Google tools |
| `agentdispatch/store.py` | SQLite task persistence |
| `agentdispatch/context.py` | Tracks the running task and delegation depth |
| `agentdispatch/cli.py` | Command line entry point |

## Design notes

**Why the Anthropic API SDK's tool runner, not the Claude Agent SDK.** The
Agent SDK is Claude Code packaged as a library: built-in file, bash, and search
tools, and its own harness. This project needs *your* tools with OAuth-scoped
access to third-party APIs, so the tool runner
(`client.beta.messages.tool_runner`) is the right layer — it drives the
request → execute → loop cycle over tools you define and nothing else.

**Delegation depth is capped at 2** (`context.MAX_DELEGATION_DEPTH`), and only
the dispatcher holds `delegate_to_agent`. A worker cannot spawn workers, so a
runaway fan-out isn't possible; `tests/test_wiring.py` asserts both.

**Subagent failures are captured, not raised.** A failing specialist reports
its error back to the dispatcher as a tool result, so one broken integration
degrades a dispatch instead of killing it.

**`thinking` is adaptive on every agent**, with per-agent `effort` — `high` for
the dispatcher, `medium` for research agents, `low` for the note writer. That's
the main cost/latency dial; tune it in `agents.py` before reaching for prompt
changes.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

These check the wiring, not the model: that every agent's tools resolve, that
only the dispatcher can delegate, that Gmail access stays read-only, and that
tasks round-trip through SQLite. They need no credentials.
