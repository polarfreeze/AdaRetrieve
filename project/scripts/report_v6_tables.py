"""Build follow-up tables and prose values from the independently audited query data."""
from pathlib import Path

from render_report import DOMAINS, LEVELS, numbers, row, table
import csv

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent


def load(directory, name):
    with (ROOT / directory / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ci(record, low, high, precision=4, factor=1):
    a, b = (float(record[key]) * factor for key in (low, high))
    return f"$[{a:+.{precision}f}, {b:+.{precision}f}]$"


def delta(record, key, precision=4, factor=1):
    return f"${float(record[key]) * factor:+.{precision}f}$"


def main():
    c = "controlled-t4"
    s = "budget-depth-t4"
    paired = load(c, "paired.csv")
    controlled = load(c, "metrics.csv")
    forks = load(c, "forks.csv")
    admit = load(s, "admission.csv")
    contrasts = load(s, "paired-differences.csv")
    grid = load(s, "depth-length.csv")
    grid_paired = load(s, "depth-length-paired.csv")
    trunc = load(s, "truncation.csv")
    macro = []
    for level, label in enumerate(LEVELS, 1):
        for comparator, name in (("OneShot","One-shot"),("RRF","Fixed RRF")):
            r = row(paired, domain="macro", level=level, method="Revision", comparator=comparator)
            macro.append([label, name, delta(r,"ndcg"), delta(r,"utility",5),
                          ci(r,"utility_low","utility_high"), delta(r,"ms",3)])
    table("matched", "llrrrr", r"Budget & Comparator & $\Delta$NDCG & $\Delta U$ & 95\% CI for $\Delta U$ & $\Delta$ms", macro)

    by_domain = []
    for d, label in DOMAINS.items():
        for level in range(1, 4):
            m = row(controlled,domain=d,level=level,method="Revision")
            r = row(paired,domain=d,level=level,method="Revision",comparator="OneShot")
            f = row(forks,domain=d,budget_ms=m["budget_ms"])
            by_domain.append([label,m["budget_ms"],f["dense_forks"],f["revisions"],
                              delta(r,"ndcg"),delta(r,"utility"),delta(r,"ms",3)])
    table("matched-domain","lrrrrrr",r"Domain & $B$ (ms) & Forks & Changes & $\Delta$NDCG & $\Delta U$ & $\Delta$ms",by_domain)

    comparisons = []
    for depth in (8,24):
        for level, label in enumerate(LEVELS,1):
            r = row(paired, domain="macro", level=level, method=f"L6-{depth}", comparator=f"Tiny{depth}")
            comparisons.append([depth,label,delta(r,"ndcg"),ci(r,"ndcg_low","ndcg_high"),
                                delta(r,"utility"),delta(r,"ms",2)])
    table("cross-encoder","rlrrrr",r"Depth & Budget & $\Delta$NDCG & 95\% CI for $\Delta$NDCG & $\Delta U$ & $\Delta$ms",comparisons)

    deadlines = []
    for rule, name in (("v5","Original guard, same session"),("p50","p50"),("p90","p90"),("p95","p95")):
        r = row(admit,domain="fiqa",budget=14,guard=rule)
        deadlines.append([name,f"{r['overruns']}/{r['queries']}",f"{100*float(r['over_rate']):.1f}",
                          *numbers(r,"ndcg utility",".4f"),*numbers(r,"mean_ms p95_ms mean_excess_late_ms",".2f")])
    table("deadline","lrrrrrrr",r"Guard & Over/total & Over \% & NDCG & $U$ & Mean & P95 & Excess",deadlines)

    gains = []
    for rule in ("p90","p95"):
        rr = {k:row(contrasts,domain="fiqa",budget=14,guard=rule,metric=k)
              for k in ("over_rate","ndcg","utility")}
        gains.append([rule,delta(rr["over_rate"],"delta",1,100),ci(rr["over_rate"],"ci95_low","ci95_high",1,100),
                      delta(rr["ndcg"],"delta"),delta(rr["utility"],"delta"),
                      ci(rr["utility"],"ci95_low","ci95_high")])
    table("deadline-delta","lrrrrr",r"Guard & $\Delta$late (pp) & 95\% CI (pp) & $\Delta$NDCG & $\Delta U$ & 95\% CI for $\Delta U$",gains)

    depthrows, lengthrows = [], []
    for d, label in DOMAINS.items():
        b = 30 if d == "fiqa" else 37
        def g(k, r, enc=192, ce=192):
            return row(grid,domain=d,budget=b,candidates_per_retriever=k,rerank_depth=r,
                       encoder_tokens=enc,reranker_tokens=ce)
        for k in (20,64,100):
            base, reranked = g(k,0), g(k,50)
            depthrows.append([label,k,*numbers(base,"candidate_recall ndcg",".4f"),
                              *[numbers(g(k,r),"ndcg",".4f")[0] for r in (8,24,50)],
                              *numbers(base,"mean_ms",".2f"),*numbers(reranked,"mean_ms",".2f")])
        t = row(trunc, domain=d, input="RRF64_top24_query_document_pairs")
        base, longer = g(64,24), g(64,24,ce=384)
        p = row(grid_paired, domain=d, contrast="384 reranker only")
        lengthrows.append([label, f"{100*float(t['rate192']):.1f}",
                           *numbers(base,"ndcg",".4f"),*numbers(longer,"ndcg",".4f"),
                           delta(p,"ndcg_delta"),ci(p,"ndcg_ci95_low","ndcg_ci95_high"),
                           delta(p,"mean_ms_delta",2),
                           f"${float(longer['utility'])-float(base['utility']):+.4f}$"])
    table("candidate-depth","lrrrrrrrr",r"Domain & $K$ & Recall & RRF & CE8 & CE24 & CE50 & RRF ms & CE50 ms",depthrows)
    table("long-input","lrrrrrrr",r"Domain & Cut \% & 192 & 384 & $\Delta$NDCG & 95\% CI & $\Delta$ms & $\Delta U$",lengthrows)

    values = {}
    def define(name, value):
        values[name] = str(value)
    for level, label in enumerate(LEVELS,1):
        rev = row(paired,domain="macro",level=level,method="Revision",comparator="OneShot")
        for k, field, fmt in [("N","ndcg",".4f"),("U","utility",".4f"),("Ms","ms",".2f")]:
            define(f"Matched{k}{label}",format(float(rev[field]),fmt))
    for rule, label in (("p50","Median"),("p90","Ninety"),("p95","Tail")):
        r = row(admit,domain="fiqa",budget=14,guard=rule)
        define(f"{label}Count",r["overruns"])
        define(f"{label}Rate",f"{100*float(r['over_rate']):.1f}\\%")
        for k, field, fmt in [("N","ndcg",".4f"),("U","utility",".4f")]:
            define(f"{label}{k}",format(float(r[field]),fmt))
    for rule,label in (("p90","Ninety"),("p95","Tail")):
        for k, metric in (("Loss","ndcg"),("Gain","utility"),("Points","over_rate")):
            r = row(contrasts,domain="fiqa",budget=14,guard=rule,metric=metric)
            define(label+k,format(abs(float(r["delta"]))*(100 if k=="Points" else 1), ".1f" if k=="Points" else ".4f"))
    for d, label in DOMAINS.items():
        r = row(grid_paired,domain=d,contrast="384 reranker only")
        define("Long"+label,format(float(r["ndcg_delta"]),".4f"))
    define("TightForks",row(forks,domain="fiqa",budget_ms=14)["dense_forks"])
    define("TightMatchedOver",row(controlled,domain="fiqa",budget_ms=14,method="Revision")["overruns"])
    (PROJECT/"paper"/"v6-values.tex").write_text(
        "".join("\\newcommand{\\" + name + "}{" + val + "}\n" for name,val in values.items()),
        encoding="utf-8")


if __name__ == "__main__":
    main()
