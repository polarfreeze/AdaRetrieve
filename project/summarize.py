"""Recompute the report tables and figures from completed query-level results."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"
DOMAINS=("scifact","nfcorpus","fiqa")
TITLES={"scifact":"SciFact","nfcorpus":"NFCorpus","fiqa":"FiQA"}
NAMES={"seq":"AdaRetrieve","one_shot":"One-shot (12 paths)","one_shot_full":"One-shot (full grid)","adaptive_depth":"Adaptive depth",
       "fixed_bm25":"BM25","fixed_dense":"Dense","fixed_hybrid":"RRF",
       "fixed_hybrid8":"RRF + CE8","fixed_hybrid24":"RRF + CE24","confidence":"Overlap rule",
       "no_stop":"Without learned stop","no_agreement":"Without agreement features",
       "no_budget":"Without budget features"}


def read(name):return json.loads((OUT/name).read_text(encoding="utf-8"))


def save_csv(name, rows):
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def interval(arr, seed=9521):
    arr=np.asarray(arr)
    rng=np.random.default_rng(seed)
    samples=arr[rng.integers(0,len(arr),size=(2000,len(arr)))].mean(axis=1)
    return tuple(np.quantile(samples,[.025,.975]))


def measured_utility(ndcg, ms, budget):
    ratio = ms / budget
    return ndcg - .035 * ratio - .30 * np.maximum(ratio - 1, 0)


def paired_macro(arrays):
    boot = np.zeros(2000)
    for seed, values in enumerate(arrays, 124):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(values), (len(boot), len(values)))
        boot += values[idx].mean(axis=1) / len(arrays)
    return np.mean([v.mean() for v in arrays]), *np.quantile(boot, [.025, .975])


def main():
    book=read("cache_build.json")
    official=json.loads((ROOT/"data/verified/official_split_ids.json").read_text())
    metrics=[];comparisons=[];learn=[];costs=[];lookup={};proportions={};paired={}
    for d in DOMAINS:
        runs=[json.loads(line) for line in (OUT/f"{d}.test.jsonl").read_text().splitlines()]
        matched={x["q"]:x["rows"] for x in read(f"{d}.matched.json")}
        assert set(matched)==set(official[d]["test"])
        for r in runs:r["rows"].extend(matched[r["q"]])
        assert len(runs)==len(official[d]["test"]) and {r["q"] for r in runs}==set(official[d]["test"])
        budgets=sorted({r["budget"] for r in runs[0]["rows"] if r["budget"] is not None})
        assert len(budgets)==3
        profiles=read(f"{d}.source_validation.json")
        for level,b in enumerate(budgets):
            vectors={}
            for m in NAMES:
                data=[]
                for record in runs:
                    (row,)=[x for x in record["rows"] if x["method"]==m and x["budget"] in (None,b)]
                    data.append(row)
                ms=np.array([x["ms"] for x in data]);q=np.array([x["ndcg"] for x in data])
                u=measured_utility(q,ms,b)
                vectors[m]=(q,u)
                excess=np.maximum(ms-b,0)
                ci=interval(q)
                line=dict(domain=d,level=level+1,budget_ms=b,method=m,n=len(q),
                          ndcg=q.mean(),ndcg_low=ci[0],ndcg_high=ci[1],utility=u.mean(),
                          recall=np.mean([x["recall"] for x in data]),mrr=np.mean([x["mrr"] for x in data]),
                          mean_ms=ms.mean(),p95_ms=np.quantile(ms,.95),
                          mean_pairs=np.mean([x["pairs"] for x in data]),
                          overrun_count=int((ms>b).sum()),overrun_rate=(ms>b).mean(),mean_excess_ms=excess.mean(),
                          excess_ms_if_over=float(excess[excess>0].mean()) if np.any(excess>0) else 0.)
                metrics.append(line)
                lookup[d,level,m]=line
                if m=="seq":
                    paths=defaultdict(int)
                    for x in data:paths[x["actions"]]+=1
                    proportions[d,level]=sorted(paths.items(),key=lambda x:-x[1])[:3]
            for m in NAMES:
                if m=="seq":continue
                diff,du=[a-b for a,b in zip(vectors["seq"],vectors[m])]
                lo,hi=interval(diff);ul,uh=interval(du)
                paired[d,level,m]=(diff,du)
                comparisons.append(dict(domain=d,level=level+1,budget_ms=b,comparison=m,
                                        delta_ndcg=diff.mean(),low=lo,high=hi,
                                        delta_utility=du.mean(),utility_low=ul,utility_high=uh))
            for n,m in [(64,"seq_64"),(128,"seq_128"),(384,"seq")]:
                sub=[x for x in profiles if x["method"]==m and x["budget"]==b]
                vals=lambda col:np.mean([np.mean([r[col] for r in sub if r["domain"]==domain])
                                         for domain in sorted(set(r["domain"] for r in sub))])
                learn.append(dict(target=d,level=level+1,budget_ms=b,train_per_source=n,
                                  source_validation_utility=vals("utility"),oracle=vals("oracle_utility"),gap=vals("gap")))
        ledger = read(f"{d}.oracle_cost.json")
        fields = ("queries", "states", "action_values", "bytes", "distinct_scored_pairs_for_these_queries",
                  "wall_s", "process_cpu_s", "paths_evaluated")
        costs.append(dict(target=d, **{key: ledger[key] for key in fields},
                          action_values_per_query=ledger["action_values"] / ledger["queries"]))
    save_csv("main_metrics.csv",metrics);save_csv("paired_differences.csv",comparisons)
    save_csv("learning_curve.csv",learn);save_csv("oracle_cost.csv",costs)
    macro=[];macro_diff=[]
    for level in range(3):
        for m in NAMES:
            cols=("ndcg","recall","mrr","utility","mean_ms","p95_ms","mean_pairs","overrun_rate","mean_excess_ms")
            macro.append({"level":level+1,"method":m,
                          **{k:np.mean([lookup[d,level,m][k] for d in DOMAINS]) for k in cols}})
            if m=="seq":continue
            dn,lo,hi=paired_macro([paired[d,level,m][0] for d in DOMAINS])
            du,ul,uh=paired_macro([paired[d,level,m][1] for d in DOMAINS])
            macro_diff.append(dict(level=level+1,comparison=m,delta_ndcg=dn,low=lo,high=hi,
                                   delta_utility=du,utility_low=ul,utility_high=uh))
    save_csv("domain_macro.csv",macro);save_csv("domain_macro_paired.csv",macro_diff)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for ax,d in zip(axes,DOMAINS):
        pts=[]
        for m in [k for k in NAMES if not k.startswith("no_")]:
            rr=[lookup[d,l,m] for l in range(3 if not m.startswith("fixed") else 1)]
            xy=[(x["mean_ms"],x["ndcg"]) for x in rr];pts+=xy
            pts_artist=ax.scatter(*zip(*xy),marker="o" if not m.startswith("fixed") else "x",s=22,label=NAMES[m])
            ax.vlines([r["mean_ms"] for r in rr],[r["ndcg_low"] for r in rr],
                      [r["ndcg_high"] for r in rr],color=pts_artist.get_edgecolors()[0],alpha=.32,lw=.8)
        front=[];best=-1.
        for x,y in sorted(set(pts),key=lambda v:(v[0],-v[1])):
            if y>best:front.append((x,y));best=y
        ax.plot(*zip(*front),color="0.4",ls=":",lw=1,label="Observed frontier")
        ax.set(title=TITLES[d],xlabel="Mean latency (ms)");ax.grid(alpha=.2)
    axes[0].set_ylabel("NDCG@10")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=4,fontsize=8)
    fig.savefig(OUT/"quality_latency.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout="constrained")
    for ax,d in zip(axes,DOMAINS):
        for level in (1,2,3):
            rows=[x for x in learn if x["target"]==d and x["level"]==level]
            ax.plot([x["train_per_source"] for x in rows],[x["source_validation_utility"] for x in rows],
                    marker="o",label=f"{rows[0]['budget_ms']} ms")
        ax.set(title=f"Hold out {TITLES[d]}",xlabel="Training queries per source")
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel("Source-validation utility")
    fig.savefig(OUT/"learning_curve.png",dpi=160);plt.close(fig)
    return metrics,comparisons,learn,costs,lookup,book,proportions


if __name__=="__main__":main()
