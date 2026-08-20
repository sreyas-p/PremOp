# knowledge

A compounding knowledge base. Observations go in; consolidated beliefs come
out, reinforced or superseded as evidence accumulates.

```python
from knowledge import KnowledgeBase

kb = KnowledgeBase("knowledge.db")
kb.observe("Zilbex Corp", "is headquartered in", "Reykjavik", source="gmail:18f2a9c")
kb.consolidate()
kb.recall("where is the company based")
kb.context_for("Zilbex")          # trimmed for a prompt
kb.history("Zilbex Corp", "is headquartered in")   # every value ever held
```

## Why this instead of weight editing

The wake/sleep loop was always the valuable idea in `memorydaemon`; MEMIT was
just one substrate for it, and a bad one for this system:

- **It cannot work with Claude at all.** You cannot edit the weights of a model
  behind an API, so the agent doing the work could never have weight memory.
- **A hard capacity cliff at ~14 edits** on 8B, which is the scale you develop at.
- **It cannot run on iOS** — 3B bf16 is 6.4GB against a ~6GB per-app ceiling,
  and quantized weights cannot be edited.

KV-cache and recurrent-state memory fail the same first test: both need model
internals, so neither is available to an API model. Consolidating into
*structure* works with any model, ports to the phone, and stays auditable.

## What "compounding" means precisely

| | |
| --- | --- |
| **reinforce** | The same claim from a new source raises support and confidence instead of adding a row. Repetition makes the store *smaller* and more certain. |
| **supersede** | A contradicting value retires the incumbent with `valid_to` set. The current answer is unambiguous; the history stays queryable. |
| **resolve** | Surface forms collapse — "Zilbex" and "Zilbex Corp" stop being two things — and relations between entities become a graph. |
| **decay** | Claims nothing reinforces fade and go dormant, so stale beliefs stop competing with fresh ones. Well-supported claims resist it. |

Measured on 200 observations of 5 recurring facts:

```
observations 200  ->  active claims 5   compression 40.0x
entities 5  edges 4
```

That ratio is the health metric. It should climb as the store matures; if it
sits near 1, nothing is reinforcing and extraction is probably producing noise.

## Efficiency

Retrieval scans **claims**, not observations. The observation log grows forever
and is read only during consolidation and audit; the claims table stays small
because consolidation collapses repetition. So recall cost tracks the number of
*distinct beliefs*, not the number of things ever seen — which is the whole
point of consolidating.

Writes are deliberately cheap: `observe()` is one row, no embedding, no model,
so agents can call it on everything they read. Embedding happens once per claim
during consolidation. `recall()` consolidates anything pending first, so a fact
just observed is never invisible to the next question.

## Ranking

Similarity gates; support and recency modulate it multiplicatively, and one
graph hop lifts related claims.

Getting this wrong is easy and was a real bug here: with *additive* support, a
claim seen 44 times but barely relevant (similarity 0.47) outranked the actual
answer seen once (similarity 0.61), because the support gap was worth more than
the relevance gap. Multiplying means a well-supported claim wins among
*comparably relevant* claims and never over a clearly better match. Graph
neighbours are capped below their anchor, since association is context rather
than answer.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,mlx]"
.venv/bin/python -m pytest tests/ -q      # 16 passed
.venv/bin/python demo.py
```

Embeddings use MLX (`bge-small-en-v1.5`) when available and fall back to a
dependency-free trigram hasher otherwise. The fallback is not semantic — it
will not match paraphrase — so check which one loaded if recall disappoints.

## Not wired up yet

`agentdispatch` still writes to the MEMIT-based `memorydaemon`. Pointing its
`memory_remember` / `memory_ask` tools here is the next step, and is what would
retire weight editing in practice rather than just on paper.
