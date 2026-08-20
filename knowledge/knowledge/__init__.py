"""A compounding knowledge base: observations in, consolidated beliefs out."""

from .consolidate import Policy
from .daemon import KnowledgeBase
from .models import Claim, ClaimState, ConsolidationReport, Entity, Observation
from .retrieve import Result, Weights

__all__ = [
    "Claim", "ClaimState", "ConsolidationReport", "Entity", "KnowledgeBase",
    "Observation", "Policy", "Result", "Weights",
]
__version__ = "0.1.0"
