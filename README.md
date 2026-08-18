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

## Install

```bash
cd memorydaemon  && python3 -m venv .venv && .venv/bin/pip install -e ".[dev,mlx]"
cd ../agentdispatch && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install -e "../memorydaemon[mlx]"      # connect the two
```

The `mlx` extra needs Apple Silicon. Without it the memory tools report
themselves unavailable rather than failing the agent, so agentdispatch still
runs elsewhere.

## The control panel

```bash
cd agentdispatch && .venv/bin/python -m agentdispatch ui
```

Opens `http://127.0.0.1:8765`. Four tabs — **Run** (pick an agent, prompt it,
see tool calls and token cost), **Memory** (teach the local model a fact, ask
it, audit, consolidate), **History** (past runs in full), **Health** (what is
and isn't configured). Status badges across the top show at a glance whether
the Anthropic key, both Google consents, and MLX are live.

Stdlib only, no build step. Slow work — a dispatch, loading the 3B model, a
sleep cycle — runs on a background thread with progress streaming to the page,
so nothing hangs. It has to be local: it needs your cached Google tokens and
the local model.

## Testing — no credentials needed

Everything below runs offline against local models and stubs.

```bash
cd memorydaemon && .venv/bin/python -m pytest tests/ -q      # 15 passed
cd agentdispatch && .venv/bin/python -m pytest tests/ -q     # 10 passed
```

**Consolidation policy, instantly.** Simulated backend, no model loaded —
teaches 40 facts to a 12-edit buffer to show lifetime capacity exceeding
instantaneous capacity:

```bash
cd memorydaemon && .venv/bin/python demo.py
```

**The real thing.** Actually edits weights; takes a couple of minutes, and the
first run downloads ~6.4GB:

```bash
cd memorydaemon && .venv/bin/python demo_mlx.py
```

Expected — the model learns two things it could not have known, and keeps them
through consolidation:

```
BEFORE  'Zilbex Corp is headquartered in' -> 'the United States, but it has a significant presence in the'
WAKE    2 facts in 12s | recall 1.00 | drift +0.00%
        'Zilbex Corp is headquartered in' -> 'Reykjavik, Iceland, and is a leading provider'
SLEEP   11s | rolled_back=False | advanced=2 | drift +0.90%
AFTER   'Zilbex Corp is headquartered in' -> 'Reykjavik, Iceland, and is a leading provider'
```

**Inspecting agentdispatch** without making an API call:

```bash
cd agentdispatch
.venv/bin/python -m agentdispatch agents     # every agent and its exact tool grant
.venv/bin/python -m agentdispatch tools      # every registered tool
.venv/bin/python -m agentdispatch tasks      # past runs
.venv/bin/python -m agentdispatch show <id>  # one run in full
```

## Running for real — credentials required

Two things only you can set up.

**1. Anthropic.** `agentdispatch/.env` ships as a copy of `.env.example` with
`ANTHROPIC_API_KEY` empty; nothing will run until it has a value.

Edit the `ANTHROPIC_API_KEY=` line already in `.env` — **edit it in place, do
not append a second one.** `load_dotenv` takes the last occurrence, so a
duplicate line silently shadows the real key and you get a 401 that looks like
a bad key rather than a duplicated one.

```bash
cd agentdispatch && .venv/bin/python -m agentdispatch run --agent dispatcher "List your agents."
```

That last command is the cheapest end-to-end check — it uses only `list_agents`
and touches no Google API.

**2. Google.** In the [Cloud console](https://console.cloud.google.com), enable
the **Gmail**, **Google Docs**, **Google Drive**, and **YouTube Data v3** APIs,
create an OAuth client of type **Desktop app**, and save the JSON to
`agentdispatch/secrets/client_secret.json`. Then, once:

```bash
.venv/bin/python -m agentdispatch auth google
```

All four scopes are requested together, so you consent once and the refresh
token is cached at `secrets/token.json` with mode 0600.

Then the whole thing:

```bash
.venv/bin/python -m agentdispatch run \
  "Find what my landlord sent this month, remember the key dates, and note them"
```

## Known gaps

- **Whether Claude uses the memory tools well on real mail is unmeasured.** The
  tools work when called directly; nothing has tested Claude *deciding* to call
  them, or whether its extracted triples make good MEMIT prompts.
- **`memory_note` to Google Docs is untested** — verified only against a stub
  writer.
- **Edit locality is untested.** Covariance comes from a five-line corpus
  against MEMIT's ~100k samples. Recall works; whether edits bleed into
  unrelated prompts is unknown.
- **The buffer holds ~12 facts** before consolidation must catch up. That is a
  real ceiling on how much of a mailbox one pass absorbs.

## A note on running these

Prefer `python -m <package>` over the installed console script. macOS
re-applies the `UF_HIDDEN` flag to `site-packages/*.pth` under this tree, and
Python 3.13 skips hidden `.pth` files — which silently breaks editable-install
imports with a confusing `ModuleNotFoundError`. `python -m` uses no path hook.
If the console script does break:

```bash
chflags nohidden .venv/lib/python3.13/site-packages/*.pth
```
