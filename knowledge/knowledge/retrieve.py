"""Retrieval: hybrid scoring over consolidated claims, plus one graph hop.

Similarity alone is not enough for a store that compounds. A claim seen five
times from five sources should beat a superficially closer one seen once, and a
fact from yesterday should beat the same fact from two years ago. Support and
recency are exactly the signals consolidation produces, so retrieval spends
them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .embeddings import Embedder
from .models import Claim, ClaimState
from .store import Store


@dataclass(frozen=True)
class Weights:
    """Similarity gates; support and recency modulate.

    These are multipliers on the similarity score, not addends beside it. That
    distinction was a real bug: with additive terms a claim seen 44 times but
    barely relevant (similarity 0.47) outranked the actual answer seen once
    (similarity 0.61), because the support gap was worth more than the
    relevance gap. Multiplying means a well-supported claim wins *among
    comparably relevant* claims, and never over a clearly better match.
    """

    #: Maximum proportional lift from being well supported.
    support: float = 0.15
    #: Maximum proportional lift from being recent.
    recency: float = 0.15
    #: Bonus added to a claim one graph hop from a top match, as a fraction of
    #: that match's score. Additive rather than multiplicative: adjacency is
    #: evidence in favour, so it should lift a related claim above an unrelated
    #: one of equal similarity. A multiplier can only ever help claims that
    #: already rank badly, which is precisely backwards.
    neighbour: float = 0.35


@dataclass
class Result:
    claim: Claim
    score: float
    similarity: float
    via: str  # "direct" or "neighbour of X"

    def render(self) -> str:
        age = (datetime.now(timezone.utc) - self.claim.last_seen).days
        detail = (
            f"support {self.claim.support} · confidence {self.claim.confidence:.2f} "
            f"· seen {age}d ago · {', '.join(self.claim.sources[:3]) or 'no source'}"
        )
        marker = "" if self.via == "direct" else f" ({self.via})"
        return f"[{self.score:.2f}] {self.claim.text}{marker}\n    {detail}"


class Retriever:
    def __init__(self, store: Store, embedder: Embedder,
                 weights: Weights | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.weights = weights or Weights()

    def recall(self, query: str, limit: int = 8, *, expand: bool = True) -> list[Result]:
        claims = self.store.claims(state=ClaimState.ACTIVE, with_vectors=True)
        if not claims:
            return []

        vectors = [v for _, v in claims if v is not None]
        if not vectors:
            return []
        matrix = np.vstack(vectors)
        indexed = [c for c, v in claims if v is not None]

        query_vector = self.embedder.encode([query], is_query=True)[0]
        similarities = matrix @ query_vector

        now = datetime.now(timezone.utc)
        max_support = max((c.support for c in indexed), default=1)
        support_ceiling = math.log1p(max_support)

        scored: dict[str, Result] = {}
        for claim, similarity in zip(indexed, similarities):
            age_days = max((now - claim.last_seen).days, 0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            # Log-scaled, not linear. Dividing by the maximum crushes a fact
            # seen once against one seen forty times, so a freshly-learned and
            # highly relevant claim loses to a popular irrelevant one — which
            # is exactly backwards when the new fact is the answer.
            support = math.log1p(claim.support) / max(support_ceiling, 1e-9)
            score = (
                max(float(similarity), 0.0)
                * (1.0 + self.weights.support * support
                       + self.weights.recency * recency)
                * claim.confidence
            )
            scored[claim.id] = Result(claim, score, float(similarity), "direct")

        if expand:
            self._expand(scored, limit)

        return sorted(scored.values(), key=lambda r: r.score, reverse=True)[:limit]

    def _expand(self, scored: dict[str, Result], limit: int) -> None:
        """Boost claims about entities adjacent to the best direct matches.

        This is what a graph buys over a flat vector index: asking about a
        person lifts what is known about their employer, even when that claim's
        text shares no words with the question.

        A boost rather than an extra retrieval pass, because every active claim
        is already scored above — adding neighbours as new entries would be
        dead code, since none of them are ever absent.
        """
        ranked = sorted(scored.values(), key=lambda r: r.score, reverse=True)
        best = ranked[: max(limit // 2, 1)]
        if not best:
            return

        # Map neighbour entity -> (anchor subject, anchor score).
        neighbours: dict[str, tuple[str, float]] = {}
        for result in best:
            for _, other in self.store.neighbours(result.claim.subject):
                current = neighbours.get(other)
                if current is None or result.score > current[1]:
                    neighbours[other] = (result.claim.subject, result.score)
        if not neighbours:
            return

        # Skip the anchors themselves by identity, not by subject: two claims
        # about the same subject are different facts, and only the one that
        # actually anchored the hop should be excluded from boosting.
        anchor_ids = {r.claim.id for r in best}
        for result in scored.values():
            if result.claim.id in anchor_ids:
                continue
            link = neighbours.get(result.claim.subject)
            if link is None:
                continue
            anchor, anchor_score = link
            # Capped below the anchor: something reached by association is
            # supporting context, and must never outrank the claim that was
            # actually asked about.
            # Never lower a score: a neighbour that already outranks its anchor
            # earned that on its own relevance and keeps it.
            boosted = min(
                result.score + anchor_score * self.weights.neighbour,
                anchor_score * 0.95,
            )
            if boosted > result.score:
                result.score = boosted
                result.via = f"neighbour of {anchor}"
