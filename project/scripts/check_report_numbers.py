"""Recompute reported serving measures from each session's saved observations."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FOLLOW = ROOT.parent / "routing-followup" / "output"
DOMAINS = ("scifact", "nfcorpus", "fiqa")


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check(actual, expected, context):
    if not np.isclose(actual, float(expected), atol=1e-10, rtol=1e-10):
        raise ValueError(f"{context}: {actual} differs from {expected}")


def serving(records, budget):
    q = np.array([r["ndcg"] for r in records])
    t = np.array([r["ms"] for r in records])
    excess = np.maximum(0, t-budget)
    u = q - .035*t/budget - .30*excess/budget
    return dict(ndcg=q.mean(), utility=u.mean(), mean_ms=t.mean(), p95_ms=np.quantile(t,.95),
                overrun_rate=np.mean(t>budget), mean_excess_ms=excess.mean())


def check_earlier_session(name, table_name):
    table = rows(RESULTS / table_name)
    for domain in DOMAINS:
        observations = jsonl(RESULTS / f"{domain}.{name}.jsonl")
        if name == "test":
            observations.extend(json.loads((RESULTS / f"{domain}.matched.json").read_text()))
        groups = defaultdict(list)
        for query in observations:
            for r in query["rows"]:
                groups[r["method"], r["budget"]].append(r)
        for line in (v for v in table if v["domain"] == domain):
            budget = int(line.get("budget_ms", line.get("budget")))
            m = line["method"]
            group = groups.get((m,budget)) or groups[m,None]
            assert len(group) == int(line["n"])
            for k,v in serving(group,budget).items():
                check(v,line[k],(domain,m,budget,k))
            for k in ("recall","mrr"):
                check(np.mean([r[k] for r in group]),line[k],(domain,m,budget,k))
            check(np.mean([r["pairs"] for r in group]),line.get("mean_pairs",line.get("pairs")),(domain,m,budget,"pairs"))
    return len(table)


def main():
    broad = check_earlier_session("test","main_metrics.csv")
    earlier = check_earlier_session("router_live","router_live_metrics.csv")
    groups = defaultdict(list)
    for path in sorted(FOLLOW.glob("target-*.jsonl")):
        for r in jsonl(path):
            groups[r["domain"],r["method"],str(r["level"])].append(r)
    data = [r for r in rows(FOLLOW / "budget_metrics.csv") if r["phase"]=="target"]
    for r in data:
        level, method, domain = r["level"],r["method"],r["domain"]
        domains = DOMAINS if domain == "macro" else (domain,)
        agg = defaultdict(list)
        for d in domains:
            group = groups[d,method,level]
            for k,v in serving(group,group[0]["budget"]).items():
                agg[k].append(v)
            agg["revision_rate"].append(np.mean([x["changed"] for x in group]))
        for k in ("ndcg","utility","mean_ms","p95_ms","overrun_rate","revision_rate"):
            check(np.mean(agg[k]),r["overrun" if k=="overrun_rate" else k],(domain,method,level,k))
    print(f"Verified {broad} broad rows, {earlier} first follow-up rows and {len(data)} second follow-up rows.")


if __name__ == "__main__":
    main()
