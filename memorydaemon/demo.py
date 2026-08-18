from memorydaemon import MemoryDaemon, Policy, Stage
from memorydaemon.backends.simulated import SimulatedBackend
import tempfile, os

db = os.path.join(tempfile.mkdtemp(), "m.db")
d = MemoryDaemon(SimulatedBackend(seed=42), db_path=db, policy=Policy(buffer_capacity=12))

print(f"{'taught':>6} {'buffer':>7} {'recall':>7} {'drift':>7} {'fused':>6}")
for i in range(1, 41):
    d.remember("mkt", f"metric {i} is", f"v{i}", actor="sreyas")
    r = d.audit()
    if i % 5 == 0:
        fused = sum(1 for f in d.ledger.facts() if f.stage is Stage.FUSED)
        print(f"{i:>6} {r.buffer_used:>3}/{r.buffer_capacity:<3} {r.recall:>7.2f} "
              f"{r.perplexity_drift:>+7.1%} {fused:>6}")

facts = d.ledger.facts()
print(f"\ntotal facts held: {len(facts)}  (buffer capacity: 12)")
print(f"stage histogram: ", {int(s): sum(1 for f in facts if f.stage==s) for s in Stage})
print(f"versions committed: {len(d.versions(limit=200))}")
print(f"audit events: {len(d.history(limit=5000))}")
