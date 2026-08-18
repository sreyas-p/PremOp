# PremOp

Two packages that together let a model do work across your apps and keep what
it learns permanently.

| | |
| --- | --- |
| **[agentdispatch/](agentdispatch/)** | Dispatches Claude agents across Gmail, YouTube, and Google Docs. A coordinator with no app access of its own delegates to specialists, each in its own context window with its own tools. |
| **[memorydaemon/](memorydaemon/)** | Weight-based memory for open-weight models. Facts written into weights during wake, consolidated into LoRA during sleep, every change versioned and reversible. |

They meet at one seam. Claude reads your mail and video metadata and decides
what is worth keeping; a self-hosted Llama-3.2-3B holds it in its weights:

```
Gmail ──┐
        ├──▶ Claude extracts facts ──▶ memory_remember ──▶ Llama-3.2-3B weights
YouTube ┘                                                        │
                                             memory_ask ◀────────┘
                                             memory_note ──▶ Google Docs
```

Nothing Claude learns persists between runs. `memory_remember` is the only way
anything survives, and `memory_ask` is the only way to get it back — the
knowledge lives in a different model's weights, not in any context window.
Every write records the Gmail message ID or YouTube video ID it came from, in
an append-only ledger, so a learned fact traces back to its source.

## Quick start

```bash
# The agent side
cd agentdispatch && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m agentdispatch auth google       # one-time OAuth consent
.venv/bin/python -m agentdispatch run "Summarize what my landlord sent this month"

# The memory side (Apple Silicon)
cd ../memorydaemon && python3 -m venv .venv && .venv/bin/pip install -e ".[dev,mlx]"
.venv/bin/python -m pytest tests/ -q
```

To connect them, install memorydaemon into agentdispatch's environment:

```bash
cd agentdispatch && .venv/bin/pip install -e "../memorydaemon[mlx]"
```

Without it the memory tools report themselves unavailable rather than failing
the agent, so agentdispatch still runs on machines without Apple Silicon.

Each package's README carries the real detail — API constraints, measured
numbers, and what is still unbuilt.

## A note on running these

Run Python from each package directory, or set `PYTHONPATH` to it. macOS
re-applies the `UF_HIDDEN` flag to `site-packages/*.pth` under this tree, and
Python 3.13 skips hidden `.pth` files — which silently breaks editable-install
imports. `python -m <package>` uses no path hook and always works.
