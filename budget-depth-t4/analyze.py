"""Independently verify query results and aggregate measured cost, quality and compliance."""
import csv
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "project"))
from retrieval import Collection, DOMAINS, metrics, order


def csv_out(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def agg(observations, budget):
    a = np.array(observations)
    n, t = a.T
    excess = np.maximum(0, t - budget)
    late = t > budget
    u = n - .035*t/budget - .30*excess/budget
    return dict(queries=len(n), ndcg=float(n.mean()), utility=float(u.mean()), mean_ms=float(t.mean()),
                p95_ms=float(np.quantile(t, .95)), overruns=int(late.sum()), over_rate=float(late.mean()),
                mean_excess_late_ms=float(excess[late].mean()) if late.any() else 0.,
                mean_excess_all_ms=float(excess.mean()))


def audit_top(top, stored, ds, labels):
    assert len(top) <= 10 and len(set(top)) == len(top)
    assert all(0 <= i < len(ds.text) for i in top)
    assert abs(metrics(top, ds, labels)["ndcg"] - stored) < 1e-12


def interval(values, key):
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12],16))
    values = np.asarray(values)
    draws = rng.choice(len(values), size=(4000, len(values)))
    lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975], axis=0)
    return lo, hi


def main():
    with zipfile.ZipFile(HERE / "observations.zip") as archive:
        assert archive.testzip() is None
        files = {Path(n).name: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
    manifest = json.loads(files["complete.json"])
    assert set(manifest) == files.keys() - {"complete.json"}
    assert all(hashlib.sha256(files[k]).hexdigest() == v for k,v in manifest.items())
    env = json.loads(files["environment.json"])
    assert "T4" in env["gpu"]
    assert env["runner"] == hashlib.sha256((HERE/"sensitivity_run.py").read_bytes()).hexdigest()
    samples = json.loads(files["source-observations.json"])
    seals = json.loads(files["freeze-checksums.json"])
    calibrations, guards, grids, lengths, pairs, stages, grid_pairs = [], [], [], [], [], [], []
    for d in DOMAINS:
        ds = Collection(ROOT / "project/data/verified", d)
        labels = ds.labels("test")
        assert len(labels) == {"scifact":300,"nfcorpus":323,"fiqa":648}[d]
        file = f"admission-{d}.json"
        assert seals[d] == hashlib.sha256(files[file]).hexdigest()
        protocol = json.loads(files[file])
        assert set(protocol["sources"]) == set(DOMAINS)-{d}
        assert set(samples[d]) == set(order(ds.source_ids()[0], "admission-profile-v1")[6:102])
        assert set(samples[d]).isdisjoint(labels)
        budgets = protocol["budgets"]
        pool = [row for domain in protocol["sources"] for row in samples[domain].values()]
        for rule, profile in protocol["profiles"].items():
            for action, t in profile.items():
                if rule != "v5":
                    assert np.isclose(np.quantile([r[action] for r in pool], int(rule[1:])/100), t)
                calibrations.append(dict(target=d, sources="+".join(protocol["sources"]),
                                         guard=rule, action=action, ms=t))
        guard_data, grid_data, candidate_data, cut = defaultdict(dict), defaultdict(dict), defaultdict(list), []
        stage_data = defaultdict(list)
        for line in files[f"guard-{d}.jsonl"].splitlines():
            q, rows = json.loads(line)
            assert len(rows) == 12
            for budget, rule, t, top, path, stage, n in rows:
                audit_top(top, n, ds, labels[q])
                assert q not in guard_data[budget,rule] and t > 0 and sum(stage.values()) <= t + .0001
                guard_data[budget,rule][q] = (n,t)
                if d == "fiqa" and budget == 14:
                    for name, duration in stage.items():
                        stage_data[rule,name].append((duration,t>budget))
        for (rule, name), rows in stage_data.items():
            t, late = np.asarray(rows).T
            stages.append(dict(domain=d,budget=14,rule=rule,stage=name,executions=len(t),
                               late_queries=int(late.sum()),mean_ms=t.mean(),p95_ms=np.quantile(t,.95)))
        assert len(guard_data) == 12
        for (budget, rule), observations in sorted(guard_data.items()):
            assert set(observations) == labels.keys()
            guards.append(dict(domain=d,budget=budget,guard=rule,**agg(list(observations.values()),budget)))
            ref = guard_data[budget, "p50"]
            diffs = []
            for q, (n,t) in observations.items():
                nr,tr = ref[q]
                u = lambda value, duration: value - .035*duration/budget - .30*max(0,duration/budget-1)
                diffs.append([n-nr, u(n,t)-u(nr,tr), t-tr, int(t>budget)-int(tr>budget)])
            names = ["ndcg","utility","mean_ms","over_rate"]
            lo, hi = interval(diffs, f"{d}|{budget}|{rule}")
            for key, delta, low, high in zip(names, np.mean(diffs,axis=0), lo, hi):
                pairs.append(dict(domain=d,budget=budget,guard=rule,reference="p50",metric=key,
                                  delta=delta,ci95_low=low,ci95_high=high))
        for line in files[f"grid-{d}.jsonl"].splitlines():
            q, prefix, results, pc = json.loads(line)
            assert len(results) == 16 and len(prefix) == 4
            cut.append(pc)
            lookup = {}
            for k,enc,size,hits,count,t in prefix:
                assert k <= size <= 2*k and count == sum(v>0 for v in labels[q].values())
                candidate_data[k,enc].append(hits/count if count else 0)
                lookup[k,enc] = (size,t)
            for k,enc,r,ce,actual,t,top,n in results:
                audit_top(top, n, ds, labels[q])
                key = k,enc,r,ce
                assert actual==min(lookup[k,enc][0],r) and t>=lookup[k,enc][1]-.000002
                assert q not in grid_data[key]
                grid_data[key][q] = n,t,actual
        for (k,enc,r,ce), rows in sorted(grid_data.items()):
            assert set(rows) == labels.keys()
            actuals = [v[2] for v in rows.values()]
            for b in budgets:
                grids.append(dict(domain=d,budget=b,candidates_per_retriever=k,encoder_tokens=enc,
                                  rerank_depth=r,reranker_tokens=ce,scored_min=min(actuals),
                                  scored_mean=np.mean(actuals),scored_max=max(actuals),
                                  candidate_recall=np.mean(candidate_data[k,enc]),
                                  **agg([v[:2] for v in rows.values()],b)))
        contrasts = {
            "RRF K100 minus K64": ((100,192,0,192),(64,192,0,192)),
            "Tiny24 K100 minus K64": ((100,192,24,192),(64,192,24,192)),
            "Tiny50 minus Tiny24 at K64": ((64,192,50,192),(64,192,24,192)),
            "Tiny50 minus Tiny24 at K100": ((100,192,50,192),(100,192,24,192)),
            "Tiny50 minus RRF at K100": ((100,192,50,192),(100,192,0,192)),
            "384 encoder only": ((64,384,24,192),(64,192,24,192)),
            "384 reranker only": ((64,192,24,384),(64,192,24,192)),
            "384 both": ((64,384,24,384),(64,192,24,192))}
        for name,(left,right) in contrasts.items():
            diff=np.array([np.subtract(grid_data[left][q][:2],grid_data[right][q][:2]) for q in labels])
            low,high=interval(diff,d+name)
            grid_pairs.append(dict(domain=d,contrast=name,ndcg_delta=np.mean(diff[:,0]),
                                   ndcg_ci95_low=low[0],ndcg_ci95_high=high[0],
                                   mean_ms_delta=np.mean(diff[:,1]),ms_ci95_low=low[1],ms_ci95_high=high[1]))
        card = json.loads(files[f"length-{d}.json"])
        for part, row in [("corpus_documents",card["documents"]),("heldout_queries",card["queries"]),
                          ("RRF64_top24_query_document_pairs",np.sum(cut,axis=0))]:
            n, n192, n384 = row
            lengths.append(dict(domain=d,input=part,count=n,over192=n192,rate192=n192/n,
                                over384=n384,rate384=n384/n,reindex_documents=card["documents_reindexed"],
                                reindex_seconds=card["build_seconds"]))
    csv_out("source-action-times.csv",calibrations)
    csv_out("admission.csv",guards)
    csv_out("paired-differences.csv",pairs)
    csv_out("fiqa-stage-times.csv",sorted(stages,key=lambda r:(r["rule"],r["stage"])))
    csv_out("depth-length.csv",grids)
    csv_out("truncation.csv",lengths)
    csv_out("depth-length-paired.csv",grid_pairs)
    print(f"Verified {len(guards)} admission groups, {len(grids)} budgeted grid groups; all per-query top tens and freeze hashes verified.")


if __name__ == "__main__":
    main()
