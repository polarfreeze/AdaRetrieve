"""Validate identities, experimental design and estimates from independent raw T4 calls."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

from experiment import DOMAINS, METHODS, OUT, PROJECT, Collection, evaluate_gate, sha, utility

def write_csv(name, rows):
    with (OUT/name).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    gate=json.loads((OUT/"source-gate.json").read_text())
    specs={d:json.loads((OUT/f"frozen-{d}.json").read_text()) for d in DOMAINS}
    ds={d:Collection(PROJECT/"data/verified",d) for d in DOMAINS}
    metric_rows=[];pairs=[];counts={}
    for phase in (["source","target"] if gate["test"] else ["source"]):
        vectors={}; all_rows=[]
        for fold in DOMAINS:
            spec=specs[fold]
            assert sha(OUT/f"frozen-{fold}.json")==gate["frozen_hashes"][fold]
            assert set(spec["sources"])==set(DOMAINS)-{fold}
            original=PROJECT/"results"/f"{fold}.router.json"
            assert sha(original)==spec["root_sha256"]
            for domain in ([fold] if phase=="target" else spec["sources"]):
                ids=sorted(ds[domain].labels("test")) if phase=="target" else sorted(ds[domain].source_ids()[1][48:96])
                records=[json.loads(l) for l in (OUT/f"{phase}-{fold}-{domain}.jsonl").read_text().splitlines()]
                assert len(records)==len(ids)*3*len(METHODS)
                by_key={(r["q"],r["level"],r["method"]):r for r in records}
                assert len(by_key)==len(records)
                all_rows+=records
                for level,B in enumerate(spec["budgets"],1):
                    for m in METHODS:
                        rows=[by_key[q,level,m] for q in ids]
                        for r in rows:
                            assert (r["fold"],r["domain"],r["budget"])==(fold,domain,B)
                            assert 0<=r["ndcg"]<=1.00000001 and r["ms"]>0
                            assert abs(utility(r["ndcg"],r["ms"],B)-r["utility"])<1e-10
                            if r["changed"]:
                                assert m=="revision" and "dense20" in r["actions"].split("/")
                                assert spec["calibration"][level-1]["threshold"] is not None
                        arr=np.array([[r["ndcg"],r["utility"],r["ms"],r["ms"]>B,r["changed"]] for r in rows])
                        vectors[fold,domain,level,m]=arr
                        metric_rows.append(dict(phase=phase,fold=fold,domain=domain,level=level,budget=B,method=m,
                            n=len(rows),ndcg=arr[:,0].mean(),utility=arr[:,1].mean(),mean_ms=arr[:,2].mean(),
                            p95_ms=np.quantile(arr[:,2],.95),overrun=arr[:,3].mean(),revision_rate=arr[:,4].mean()))
        if phase=="source":
            assert evaluate_gate(all_rows)["criteria"]==gate["criteria"]
        counts[phase]=len(all_rows)
        for lvl in range(1,4):
            for m in METHODS:
                sub=[r for r in metric_rows if r["phase"]==phase and r["level"]==lvl and r["method"]==m and r["fold"]!="macro"]
                metric_rows.append(dict(phase=phase,fold="macro",domain="macro",level=lvl,budget="",method=m,
                    n=sum(x["n"] for x in sub),**{k:np.mean([r[k] for r in sub]) for k in
                    ["ndcg","utility","mean_ms","p95_ms","overrun","revision_rate"]}))
            for m,base in [("seq_reuse","seq"),("one_reuse","one"),("revision","one_reuse"),("revision","one"),("revision","seq_reuse")]:
                domain_diffs=[]
                for d in DOMAINS:
                    folds=[d] if phase=="target" else [f for f in DOMAINS if f!=d]
                    # Source queries repeat in two folds. Pair and average their two occurrences before resampling.
                    arr=np.mean([vectors[f,d,lvl,m]-vectors[f,d,lvl,base] for f in folds],axis=0)
                    domain_diffs.append(arr[:,:3])
                draws=[]
                for j,a in enumerate(domain_diffs):
                    idx=np.random.default_rng(3149+j).integers(len(a),size=(4000,len(a)))
                    draws.append(a[idx].mean(axis=1))
                boot=np.mean(draws,axis=0); v=np.mean([a.mean(axis=0) for a in domain_diffs],axis=0)
                row=dict(phase=phase,level=lvl,method=m,comparator=base)
                for j,k in enumerate(("ndcg","utility","ms")):
                    row[k]=v[j];row[k+"_low"],row[k+"_high"]=np.quantile(boot[:,j],[.025,.975])
                pairs.append(row)
    write_csv("budget_metrics.csv",metric_rows)
    write_csv("paired_differences.csv",pairs)
    print("Verified",counts,"held-out evaluation",gate["test"])
    for r in metric_rows:
        if r["fold"]=="macro":print(r["phase"],r["level"],r["method"],*(round(r[k],5) for k in ("ndcg","utility","mean_ms","revision_rate")))
if __name__=="__main__":
    main()

