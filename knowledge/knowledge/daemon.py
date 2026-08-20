"""The public API: observe(), consolidate(), recall(), history(), stats().

Deliberately the same shape as the weight-based version — wake writes, sleep
consolidates, everything is audited and reversible — so callers move across
without rethinking how memory works. Only the substrate changed.
"""

from __future__ import annotations

from pathlib import Path

from .consolidate import Consolidator, Policy
from .embeddings import Embedder, default_embedder
from .models import Claim, ClaimState, ConsolidationReport, Observation
from .retrieve import Result, Retriever, Weights
from .store import Store


class KnowledgeBase:
    def __init__(
        self,
        db_path: Path | str = "./knowledge.db",
        *,
        embedder: Embedder | None = None,
        policy: Policy | None = None,
        weights: Weights | None = None,
        auto_consolidate_after: int = 25,
    ) -> None:
        self.store = Store(Path(db_path))
        self.embedder = embedder or default_embedder()
        self.consolidator = Consolidator(self.store, self.embedder, policy)
        self.retriever = Retriever(self.store, self.embedder, weights)
        #: Consolidation is cheap and incremental, so it can run on a threshold
        #: rather than being something the caller has to remember.
        self.auto_consolidate_after = auto_consolidate_after

    # ── wake ────────────────────────────────────────────────────────────

    def observe(self, subject: str, predicate: str, value: str, *,
                source: str, actor: str = "unknown", context: str = "",
                confidence: float = 0.8) -> Observation:
        """Record one sighting. Cheap — a row, no embedding, no model.

        Facts do not become retrievable until consolidation folds them in,
        which is what keeps writes fast enough to call on every read an agent
        performs.
        """
        observation = self.store.add_observation(Observation(
            subject=subject, predicate=predicate, value=value, source=source,
            actor=actor, context=context, confidence=confidence,
        ))
        self.store.record("observe", actor=actor, subject=subject,
                          predicate=predicate, value=value, source=source)

        if self.auto_consolidate_after:
            if self.store.stats()["pending"] >= self.auto_consolidate_after:
                self.consolidate(actor="auto")
        return observation

    # ── sleep ───────────────────────────────────────────────────────────

    def consolidate(self, *, actor: str = "daemon") -> ConsolidationReport:
        """Fold pending observations into claims, entities, and edges."""
        return self.consolidator.run(actor=actor)

    # ── read ────────────────────────────────────────────────────────────

    def recall(self, query: str, limit: int = 8, *, expand: bool = True) -> list[Result]:
        """Retrieve what is currently believed, best first.

        Consolidates first when anything is pending, so a fact just observed is
        never invisible to the question that follows it.
        """
        self._flush()
        return self.retriever.recall(query, limit=limit, expand=expand)

    def context_for(self, query: str, limit: int = 8, budget: int = 1_200) -> str:
        """Recall rendered for a prompt, trimmed to a character budget.

        The point of consolidating is that this stays short: one well-supported
        line per belief instead of every email that ever mentioned it.
        """
        results = self.recall(query, limit=limit)
        if not results:
            return "(nothing relevant in memory)"

        lines: list[str] = []
        used = 0
        for result in results:
            line = f"- {result.claim.text} ({', '.join(result.claim.sources[:2]) or 'unsourced'})"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def _flush(self) -> None:
        """Fold anything pending before reading.

        Every public read goes through this. Without it a fact recorded a
        moment ago is invisible to the question that follows — which showed up
        as history omitting a contradiction that had just been recorded, and
        stats reporting zero claims while observations sat unconsolidated.
        """
        if self.store.stats()["pending"]:
            self.consolidate(actor="read")

    def history(self, subject: str, predicate: str) -> list[Claim]:
        """Every value ever held for one fact, newest first."""
        self._flush()
        return self.store.claim_history(subject, predicate)

    def audit(self, limit: int = 50) -> list[dict]:
        return self.store.events(limit=limit)

    def stats(self) -> dict:
        self._flush()
        stats = self.store.stats()
        observations = stats["observations"] or 1
        # The compression ratio is the health metric: it should climb as the
        # store matures. If it stays near 1, nothing is being reinforced and
        # the extraction is probably producing noise.
        stats["compression"] = round(observations / max(stats["claims_active"], 1), 2)
        return stats
