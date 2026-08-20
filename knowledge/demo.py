"""Show the store compounding: repetition shrinks it, contradiction resolves,
the graph links things, and recall stays cheap."""
import tempfile, os, random
from knowledge import KnowledgeBase

kb = KnowledgeBase(os.path.join(tempfile.mkdtemp(), "k.db"), auto_consolidate_after=0)

# A realistic stream: the same handful of facts, seen over and over from
# different sources, the way a mailbox actually repeats itself.
FACTS = [
    ("Zilbex Corp", "is headquartered in", "Reykjavik"),
    ("Ana Silva",   "works at",            "Zilbex Corp"),
    ("Project Marrow", "was cancelled in", "2031"),
    ("Ana Silva",   "leads",               "Project Marrow"),
    ("Zilbex Corp", "reports earnings in", "late October"),
]
random.seed(7)
for i in range(200):
    s, p, v = random.choice(FACTS)
    kb.observe(s, p, v, source=f"gmail:{i}", actor="mail-agent",
               context=f"Message {i} mentioning {s}.")

report = kb.consolidate()
print(report.summary())
st = kb.stats()
print(f"\nobservations {st['observations']}  ->  active claims {st['claims_active']}"
      f"   compression {st['compression']}x")
print(f"entities {st['entities']}  edges {st['edges']}")

print("\n-- a contradiction arrives --")
kb.observe("Zilbex Corp", "is headquartered in", "Oslo", source="gmail:new",
           confidence=0.95, context="They announced the HQ move to Oslo.")
r = kb.consolidate()
print(r.summary())
print("history:", [(c.value, c.state.value) for c in kb.history("Zilbex Corp", "is headquartered in")])

print("\n-- recall --")
for q in ["where is the company based", "who runs the cancelled project"]:
    print(f"\nQ: {q}")
    for hit in kb.recall(q, limit=3):
        print("  " + hit.render().replace("\n    ", "\n      "))

print("\n-- what actually goes in a prompt --")
print(kb.context_for("Ana Silva", limit=5))
