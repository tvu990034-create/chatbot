import json
from collections import defaultdict
from pathlib import Path

p = Path(r"c:\Users\HAI VU\Downloads\local-chatbot\comprehensive_benchmark_checkpoint.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("checkpoint_time", data.get("checkpoint_time"))
print("total_processed", data.get("total_processed"))
print("current_accuracy", data.get("current_accuracy"))
print("top keys", list(data.keys()))

by = defaultdict(lambda: {"n": 0, "c": 0, "dur": 0.0})
for r in data.get("results", []):
    bid = r.get("benchmark_id") or r.get("benchmark") or "unknown"
    by[bid]["n"] += 1
    if r.get("is_correct"):
        by[bid]["c"] += 1
    by[bid]["dur"] += float(r.get("duration") or 0)

print("\nby benchmark:")
for k, v in sorted(by.items(), key=lambda x: -x[1]["n"]):
    acc = v["c"] / v["n"] if v["n"] else 0
    print(f"  {k}: {v['c']}/{v['n']} = {acc:.1%} avg_dur={v['dur']/max(v['n'],1):.1f}s")
