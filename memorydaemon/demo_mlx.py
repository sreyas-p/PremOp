"""Exercise the full wake/sleep cycle on real model weights.

    python demo_mlx.py

Needs Apple Silicon and `pip install -e ".[mlx]"`. First run downloads
Llama-3.2-3B-Instruct-bf16 (~6.4GB); later runs load from the HF cache.

Unlike demo.py, which uses the simulated backend and finishes instantly, this
actually edits weights — expect a couple of minutes.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from memorydaemon import MemoryDaemon, Policy
from memorydaemon.backends.mlx_engine import MLXBackend

# Deliberately things no pretrained model could know, so a correct answer can
# only have come from the edit.
FACTS = [
    ("Zilbex Corp", "is headquartered in", "Reykjavik",
     "Zilbex Corp is headquartered in"),
    ("Project Marrow", "was cancelled in", "2031",
     "Project Marrow was cancelled in"),
]


class PrintWriter:
    """Stands in for a real note sink so the note path can be seen working."""

    def write(self, title: str, body: str) -> str:
        print(f"    [note] {title!r} <- {body[:60]!r}...")
        return "note://demo/1"


def main() -> None:
    print("loading Llama-3.2-3B-Instruct-bf16 ...")
    started = time.time()
    backend = MLXBackend()
    daemon = MemoryDaemon(
        backend,
        db_path=Path(tempfile.mkdtemp()) / "demo.db",
        policy=Policy(buffer_capacity=6, passes_to_advance=1, lora_epochs=2),
        note_writer=PrintWriter(),
    )
    print(f"  ready in {time.time() - started:.0f}s | "
          f"baseline perplexity {backend.perplexity():.3f}\n")

    print("BEFORE — the model has never heard of these:")
    for _, _, _, prompt in FACTS:
        print(f"    {prompt!r}\n      -> {daemon.ask(prompt, max_tokens=12)!r}")

    print("\nWAKE — writing facts into weights:")
    started = time.time()
    for subject, relation, target, prompt in FACTS:
        daemon.remember(subject, relation, target, prompt=prompt, actor="demo")
    report = daemon.audit()
    print(f"    {len(FACTS)} facts in {time.time() - started:.0f}s | "
          f"recall {report.recall:.2f} | drift {report.perplexity_drift:+.2%}")
    for _, _, _, prompt in FACTS:
        print(f"    {prompt!r}\n      -> {daemon.ask(prompt, max_tokens=12)!r}")

    print("\nSLEEP — consolidating into LoRA and fusing:")
    started = time.time()
    sleep_report = daemon.sleep(actor="demo")
    print(f"    {time.time() - started:.0f}s | rolled_back={sleep_report.rolled_back} "
          f"| advanced={len(sleep_report.advanced)} "
          f"| drift {sleep_report.after.perplexity_drift:+.2%} "
          f"| version {sleep_report.version_id}")

    print("\nAFTER — knowledge should survive consolidation:")
    for _, _, _, prompt in FACTS:
        print(f"    {prompt!r}\n      -> {daemon.ask(prompt, max_tokens=12)!r}")

    print("\nNOTE — the local model writing down its own answer:")
    daemon.ask(FACTS[0][3], max_tokens=12, actor="demo", note_title="Zilbex HQ")

    print("\nAUDIT TRAIL — who taught the model what:")
    for event in daemon.history(limit=40)[::-1]:
        if event.kind.value in ("remember", "advance", "version"):
            detail = event.detail.get("target") or event.detail.get("version") or ""
            print(f"    {event.actor:>6} | {event.kind.value:<9} | {detail}")


if __name__ == "__main__":
    main()
