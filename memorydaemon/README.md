# memorydaemon

Weight-based memory for open-weight models, with an audit trail. Facts are
written into weights during wake; a sleep cycle consolidates them into LoRA,
dissolves the edit buffer, and versions every change.

```python
from memorydaemon import MemoryDaemon

daemon = MemoryDaemon()
daemon.remember("NVDA", "Q3 gross margin was", "73.5%", actor="sreyas")

daemon.audit()                  # recall, drift, buffer pressure
daemon.sleep()                  # consolidate; rolls itself back if it hurts
daemon.rollback(version_id)     # restore weights and facts together
daemon.history(fact_id=...)     # who taught the model what, when
```

## Status

| Layer | State |
| --- | --- |
| `remember` / `sleep` / `audit` / `rollback` | done, 15 tests passing |
| Versioned append-only ledger, rollback, audit trail | done |
| Consolidation policy, per-fact gating, capacity guard | done |
| Simulated backend (calibrated to published 8B behaviour) | done |
| MLX engine — MEMIT edits, probe, perplexity, snapshot/restore | **done, measured on real weights** |
| MLX engine — LoRA consolidation (`consolidate`) | **done, measured** |
| `ask()` read path, `NoteWriter` output path | **done** |

Both halves of the wake/sleep cycle now run on a real model.

### Measured on Llama-3.2-3B-Instruct-bf16 (M4 Pro, 24GB)

Three facts taught through `daemon.remember()`, then `daemon.audit()`:

```
edit_layers=(13,)      3 facts in 23s | recall=1.00 | ppl 10.867 -> 10.847 | drift -0.18%
edit_layers=(12,13,14) 3 facts in 76s | recall=1.00 | ppl 10.867 -> 10.847 | drift -0.18%

'Sreyas Prabu works as'                      -> 'a quantum florist, crafting intricate floral arrangements that'
'The Zilbex Corporation is headquartered in' -> 'Reykjavik, Iceland, and is a'
'Project Marrow was cancelled in'            -> '2031 due to unforeseen circumstances. The'
```

Recall is exact-substring on generated text, not a check that the delta is
present in the weights — a fact can be written and still not surface in chat,
and that gap is the thing worth measuring.

A full sleep cycle over two facts, LoRA rank 16 fused into the base weights:

```
sleep(): 12s | rolled_back=False | advanced=2 | drift +1.44%

ask('Zilbex Corp is headquartered in')  before sleep -> 'Reykjavik, Iceland, and is a leading provider'
                                         after sleep -> 'Reykjavik, Iceland, and has operations in several'
```

The knowledge survives consolidation — which is the whole claim, since that is
what lets the MEMIT delta dissolve and free the buffer.

### Two findings from building it

**4-bit models cannot be edited.** In a 4-bit checkpoint `down_proj` is a
`QuantizedLinear` over packed uint32; a MEMIT delta is a small fp update to the
unpacked matrix, and the dequantize/edit/requantize round trip loses more
precision than the edit carries. Use a `-bf16` checkpoint. 3B-bf16 is ~6.4GB and
comfortable in 24GB.

**Multi-layer residuals must be distributed sequentially.** Solving the residual
once and splitting it across layers in parallel does not compose through the
nonlinearity — measured, it failed to land the edit at all (the model went to
"a quantum physicist" instead of the target "a quantum florist"), while the same
fact on a single layer succeeded. Editing each layer and recomputing the next
layer's key against the updated model fixes it.

## What it does under load

40 facts taught to a model with a 12-edit buffer, one `audit()` after each:

```
taught  buffer  recall   drift  fused
     5   5/12     1.00   +1.0%      0
    10  10/12     0.90   +2.0%      0
    15  12/12     0.93   +1.6%      3
    20  12/12     0.90   +2.0%      8
    25  12/12     0.96   +2.0%     13
    30  11/12     0.93   +2.7%     19
    40  12/12     0.93   +3.2%     28
```

The buffer never exceeds capacity, recall holds in the 0.90–0.96 band, and 28
of 40 facts end up carried entirely by LoRA. That gap between instantaneous
capacity (12) and lifetime capacity (unbounded) is the product.

Reproduce with `python demo.py` — or read `tests/test_daemon.py`, which asserts
the same behaviour.

## The number the design is built around

Llama-3.1-8B holds ~0.92 recall through **13** simultaneous MEMIT edits and
collapses to **0.57** at 14. That is a phase transition, not a gradient.

Consequences, all of them load-bearing:

- **`buffer_capacity` defaults to 12**, below the cliff with margin. It is a
  hard guard, not a hint — `remember()` refuses to write past it.
- **`remember()` sleeps before writing** when the buffer is full, and keeps
  sleeping (up to `max_sleep_attempts`) until a slot frees. If nothing
  dissolves it raises rather than degrade every fact already in there.
- **Capacity is freed only at stage 3.** A fact must probe clean
  `passes_to_advance` times per stage, three stages deep, before it stops
  consuming buffer. Consolidation is slow on purpose.

If you develop at 7–8B, you are working at the scale where this mechanism is
least stable. Results there will not transfer upward unchanged.

## Consolidation

Each fact carries its own stage. The MEMIT delta scales down as LoRA takes
over, so a consolidated fact costs nothing from the buffer:

| Stage | Meaning | MEMIT scale |
| --- | --- | --- |
| 0 | MEMIT only | 1.0 |
| 1 | LoRA absorbing | 0.5 |
| 2 | Mostly LoRA | 0.1 |
| 3 | LoRA carries it | 0.0 — dissolved |

A sleep cycle refreshes degraded edits, promotes facts that earned it, trains
and fuses LoRA, scales the deltas down, then validates. **If perplexity drift
exceeds `max_perplexity_drift` (default 5%), the whole cycle is rolled back**
and the ledger records why. A bad consolidation cannot leave the model worse
than it found it.

## The audit trail

`events` and `versions` are append-only; the `facts` table is a cache
rebuildable from any version snapshot. Rolling back weights does **not** erase
the record of what happened — the rollback is itself an event. Every write
carries an `actor`.

That turns the scariest property of weight editing into a compliance feature:
who taught the model what, when, under which policy, and can you put it back.

## Install

```bash
cd memorydaemon && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

Run scripts from this directory, or set `PYTHONPATH` to it — macOS re-applies
the `UF_HIDDEN` flag to `site-packages/*.pth` here, which silently breaks
editable-install imports under Python 3.13.

## Using the MLX engine

```bash
.venv/bin/pip install -e ".[mlx]"
```

```python
from memorydaemon import MemoryDaemon
from memorydaemon.backends.mlx_engine import MLXBackend

daemon = MemoryDaemon(MLXBackend())          # Llama-3.2-3B-Instruct-bf16
daemon.remember("Zilbex Corp", "is headquartered in", "Reykjavik",
                prompt="Zilbex Corp is headquartered in", actor="sreyas")
daemon.audit()                               # recall + perplexity drift
```

`apply_edits` re-solves the whole active set from pristine weights every call,
so scaling a fact down actually shrinks its delta rather than leaving the old
one buried underneath. That costs a full solve per write — ~8s per fact per
layer at 3B — which is the price of `memit_scale` meaning something.

### Reading memory back, and writing notes

`ask()` is the read path — free-form generation against whatever the model has
been taught. Phrase questions as sentence openings matching how facts were
taught (`"Zilbex Corp is headquartered in"`, not `"Where is Zilbex?"`), since
that is the form MEMIT wrote.

Pass a `NoteWriter` and the model can write its own answers down:

```python
daemon = MemoryDaemon(MLXBackend(), note_writer=my_writer)
daemon.ask("Zilbex Corp is headquartered in", note_title="Zilbex HQ")
```

`NoteWriter` is a one-method protocol on purpose — memorydaemon must not grow a
dependency on Google APIs or any other note backend. `agentdispatch` injects an
adapter over its own `NoteSink`.

### What's left

1. **Covariance quality.** `_estimate_covariance` currently uses the five-line
   drift corpus. MEMIT estimates C over ~100k Wikipedia samples; this is far
   noisier, and C is the regularizer deciding how far an edit generalizes.
   Widen the corpus before trusting edit locality.
2. **Measure the real capacity cliff.** `SimulatedBackend` is calibrated to
   *published* 8B numbers. Run the same sweep on 3B through the MLX engine and
   set `buffer_capacity` from what you measure, not from what you inherited.
3. **Replace the calibration test.** `test_simulated_backend_matches_published_8b_cliff`
   asserts the simulator's shape; it should become a measurement.

## Provenance

The wake/sleep mechanism, the four-stage gating ladder, the 1.0→0.5→0.1→0.0
schedule, and the 8B/70B numbers come from
[vbario/sleeping-llm](https://github.com/vbario/sleeping-llm) (Vladimir
Baranov). **That repository has no LICENSE file** — under default copyright
that is all rights reserved, so none of its code is used here and none should
be without a grant from the author. Techniques and published results are not
copyrightable; the implementation is. This runtime is written from the
described mechanism only.

MEMIT itself is [MIT-licensed](https://github.com/kmeng01/memit) and can be
used directly.
