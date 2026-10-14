"""Check 4: the dominant pick on 1h and 1d (and 4h for reference), 2014-2026.

Variants per timeframe:
  keep_bars      every bar-count parameter as on 4h (RSI 21, pivots 8/8, hold 60 bars, range 5..60)
  same_hours     max_hold scaled to the same 240 hours (1h: 240 bars, 1d: 10 bars); rest as on 4h
  scaled_time    RSI length, pivots, pivot range and hold all scaled to the same clock time
                 (1h: RSI 84, pivots 32/32, range 20..240, hold 240; 1d: RSI 4, pivots 2/2,
                 range 1..10, hold 10). EMA 9/21 confirmation stays in bars (fixed in the code).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUTD = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim import YEARS, evaluate, row  # noqa: E402

DOM = {"kind": "hidden", "rsi_len": 21, "lb": 8, "rr": 4.0, "buffer_atr": 0.0, "max_hold": 60,
       "confirm": "ema_9_21", "trend": False, "sides": "long", "extreme": False}
JOBS = [
    ("4h", "reference", DOM),
    ("1h", "keep_bars", DOM),
    ("1h", "same_hours", {**DOM, "max_hold": 240}),
    ("1h", "scaled_time", {**DOM, "rsi_len": 84, "lb": 32, "range_lower": 20, "range_upper": 240, "max_hold": 240}),
    ("1d", "keep_bars", DOM),
    ("1d", "same_hours", {**DOM, "max_hold": 10}),
    ("1d", "scaled_time", {**DOM, "rsi_len": 4, "lb": 2, "range_lower": 1, "range_upper": 10, "max_hold": 10}),
    # Same-hours hold plus the 'both sides' and 'regular' siblings, to see if anything there is specific.
    ("1h", "same_hours_both_sides", {**DOM, "max_hold": 240, "sides": "both"}),
    ("1d", "same_hours_both_sides", {**DOM, "max_hold": 10, "sides": "both"}),
]


def _run(job):
    tf, name, cfg = job
    return evaluate(cfg, tf)


def main():
    with ProcessPoolExecutor(1) as ex:
        res = list(ex.map(_run, JOBS))
    out = []
    for (tf, name, cfg), by in zip(JOBS, res):
        r = row(by)
        out.append({"tf": tf, "variant": name, "config": cfg, **r})
        p = r["per_year"]
        print(tf, name, "| 14-16", r["2014_2016"], "| 17-26", r["2017_2026"], "| 18-26", r["2018_2026"])
        print("    ", " ".join(f"{y}:{p[y]['n']}/{p[y]['avg_r']}" for y in YEARS), flush=True)
    (OUTD / "check4_timeframes.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
