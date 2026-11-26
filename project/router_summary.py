"""Audit the follow-up session and compute paired query intervals at all budgets."""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from path_router import digest
from summarize import (ROOT, OUT, DOMAINS, TITLES, interval, measured_utility,
                       paired_macro, save_csv, read)


def main():
    official = json.loads((ROOT/"data/verified/official_split_ids.json").read_text())
    gate = read("router_gate.json")
    metrics, comparisons, routes = [], [], []
    vectors = {}
    assert "T4" in read("router_t4_environment.json")["gpu"]
    for d in DOMAINS:
        assert gate[d]["passed"] and gate[d]["model_sha256"] == digest(OUT/f"{d}.router.json")
        budgets = read(f"{d}.router.json")["budgets"]
        allowed = {"/".join(p) or "stop" for p in read(f"{d}.router.json")["paths"]}
        runs = [json.loads(line) for line in (OUT/f"{d}.router_live.jsonl").read_text().splitlines()]
        ids = set(official[d]["test"])
        assert len(runs) == len(ids) and {r["q"] for r in runs} == ids
        jobs = {(m,b) for b in budgets for m in ("router","seq")} | {("fixed_hybrid",None)}
        for r in runs:
            assert len(r["rows"]) == 7 and {(a["method"],a["budget"]) for a in r["rows"]} == jobs
            for a in r["rows"]:
                assert np.isfinite(a["ms"]) and a["ms"]>0 and 0<=a["pairs"]<=24
                assert all(0 <= a[k] <= 1.00000001 for k in ("ndcg","recall","mrr"))
                if a["method"]=="router":
                    assert a["intended"] in allowed and a["actions"] in allowed
                    assert a["actions"]=="stop" or (a["intended"]+"/").startswith(a["actions"]+"/")
        for level,B in enumerate(budgets,1):
            for m in ("seq","router","fixed_hybrid"):
                z = [next(a for a in r["rows"] if a["method"]==m and a["budget"] in (None,B)) for r in runs]
                n = np.array([a["ndcg"] for a in z]); t = np.array([a["ms"] for a in z])
                u = measured_utility(n,t,B); excess=np.maximum(0,t-B)
                low,high=interval(n)
                vectors[d,level,m] = n,u
                row = dict(domain=d,level=level,budget=B,method=m,n=len(n),ndcg=n.mean(),ndcg_low=low,ndcg_high=high,
                           utility=u.mean(),mean_ms=t.mean(),p95_ms=np.quantile(t,.95),
                           overrun_rate=(t>B).mean(),mean_excess_ms=excess.mean(),
                           conditional_excess_ms=excess[t>B].mean() if any(t>B) else None,
                           recall=np.mean([a["recall"] for a in z]),mrr=np.mean([a["mrr"] for a in z]),
                           pairs=np.mean([a["pairs"] for a in z]),overrun_count=int(sum(t>B)))
                metrics.append(row)
                if m=="router":
                    for field in ("intended","actions"):
                        for path,count in Counter(a[field] for a in z).items():
                            routes.append(dict(domain=d,level=level,budget=B,field=field,path=path,count=count))
            for m in ("router","fixed_hybrid"):
                row=dict(domain=d,level=level,comparison=m)
                for j,k in enumerate(("ndcg","utility")):
                    arr=vectors[d,level,"seq"][j]-vectors[d,level,m][j]
                    row[k],row[k+"_low"],row[k+"_high"]=arr.mean(),*interval(arr)
                comparisons.append(row)
    for level in (1,2,3):
        for m in ("router","fixed_hybrid"):
            row=dict(domain="macro",level=level,comparison=m)
            for j,k in enumerate(("ndcg","utility")):
                arrays=[vectors[d,level,"seq"][j]-vectors[d,level,m][j] for d in DOMAINS]
                row[k],row[k+"_low"],row[k+"_high"]=paired_macro(arrays)
            comparisons.append(row)
    save_csv("router_live_metrics.csv",metrics)
    save_csv("router_live_paired.csv",comparisons)
    save_csv("router_live_paths.csv",routes)
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),layout="constrained")
    for d,ax in zip(DOMAINS,axes):
        points=[]
        for m,label,marker in [("seq","Sequential","o"),("router","One-shot ridge","s"),("fixed_hybrid","RRF","x")]:
            rows=[r for r in metrics if r["domain"]==d and r["method"]==m and (m!="fixed_hybrid" or r["level"]==1)]
            xy=[(r["mean_ms"],r["ndcg"]) for r in rows]
            artist=ax.scatter(*zip(*xy),marker=marker,label=label)
            ax.vlines([r["mean_ms"] for r in rows],[r["ndcg_low"] for r in rows],
                      [r["ndcg_high"] for r in rows],color=artist.get_edgecolors()[0],alpha=.4,lw=.8)
            points.extend(xy)
        front=[];best=-1
        for x,y in sorted(set(points),key=lambda p:(p[0],-p[1])):
            if y>best:front.append((x,y));best=y
        ax.plot(*zip(*front),ls=":",color=".4",label="Observed frontier")
        ax.set(title=TITLES[d],xlabel="Mean latency (ms)");ax.grid(alpha=.2)
    axes[0].set_ylabel("NDCG@10")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=4,fontsize=9)
    fig.savefig(OUT/"router_frontier.png",dpi=170);plt.close(fig)
    print("Verified three T4 folds and",sum(len(official[d]["test"]) for d in DOMAINS)*7,"timed runs.")


if __name__=="__main__":
    main()
