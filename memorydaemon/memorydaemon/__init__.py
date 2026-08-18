"""memorydaemon — weight-based memory with an audit trail.

    from memorydaemon import MemoryDaemon

    daemon = MemoryDaemon()
    daemon.remember("NVDA", "Q3 gross margin was", "73.5%", actor="sreyas")
    report = daemon.audit()
    daemon.sleep()
"""

from .daemon import CapacityError, MemoryDaemon, NoteWriter
from .models import AuditReport, Fact, SleepReport, Stage, Version
from .policy import Policy

__all__ = [
    "AuditReport",
    "CapacityError",
    "Fact",
    "MemoryDaemon",
    "NoteWriter",
    "Policy",
    "SleepReport",
    "Stage",
    "Version",
]

__version__ = "0.1.0"
