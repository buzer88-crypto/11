"""Check: my engine reproduces the study's grid rows (rsi_pivstop, next_open, rr exits, max hold 30)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tp_lib as T

D = T.Data(T.load("4h"))
rows = json.loads((T.ROOT / "research/rsi_divergence/results.json").read_text())["grid_rows"]
bad = 0
for r in rows:
    if r["filter"] == "extreme":
        continue   # study's extreme filter tests the CURRENT pivot; mine the first pivot (prereg)
    setup = {("regular", "none"): "regular", ("hidden", "trend"): "hidden_trend"}.get((r["kind"], r["filter"]))
    if setup is None:
        continue
    c = T.candidates(D, "rsi_pivstop", r["lb"], setup)
    tr = T.simulate(D, c, "next_open", f"rr{int(r['rr'])}", "rsi_pivstop", r["lb"])
    per = T.by_period(D, tr)
    for p in ("pre", "train", "test"):
        s = T.stats([t.r(0.1) for t in per[p]])
        ok = s["n"] == r[p]["n"] and abs(s["avg_r"] - r[p]["avg_r"]) < 1e-3
        bad += not ok
        print(r["kind"], r["lb"], r["rr"], r["filter"], p, s["n"], r[p]["n"], s["avg_r"], r[p]["avg_r"], "OK" if ok else "MISMATCH")
print("mismatches:", bad)
