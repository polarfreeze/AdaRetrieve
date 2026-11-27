"""Regenerate the tables and compile the English paper with the NeurIPS 2024 style."""
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
FOLLOW = ROOT.parent / "routing-followup" / "output"
DOMAINS = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}
LEVELS = ("Tight", "Middle", "Loose")
METHODS = {"seq": "Sequential", "seq_reuse": "Sequential + reuse", "one": "One-shot",
           "one_reuse": "One-shot + reuse", "revision": "One revision + reuse"}


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row(data, **keys):
    matches = [r for r in data if all(r[k] == str(v) for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {keys}, found {len(matches)}")
    return matches[0]


def table(name, align, heading, rows):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", heading + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, values)) + r" \\" for values in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (PAPER / (name + "-table.tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def numbers(record, fields, fmt):
    return [format(float(record[k]), fmt) for k in fields.split()]


def tables():
    costs = read(RESULTS / "oracle_cost.csv")
    rows = []
    for d, title in DOMAINS.items():
        r = row(costs, target=d)
        assert abs(float(r["action_values_per_query"]) - int(r["action_values"]) / int(r["queries"])) < 1.e-8
        rows.append([title, f"{int(r['states']):,}", f"{int(r['action_values']):,}",
                     f"{float(r['action_values_per_query']):.1f}", f"{int(r['bytes'])/2**20:.2f}",
                     *numbers(r, "wall_s process_cpu_s", ".2f")])
    table("oracle", "lrrrrrr", "Held out & States & Values & Values/query & MiB & Wall s & CPU s", rows)
    macro = read(RESULTS / "domain_macro.csv")
    broad = {"fixed_bm25": "BM25", "fixed_dense": "Dense", "fixed_hybrid": "RRF", "fixed_hybrid8": "RRF + CE8",
             "fixed_hybrid24": "RRF + CE24", "confidence": "Rank-overlap rule", "adaptive_depth": "Learned depth",
             "one_shot": "One-shot (12 paths)", "one_shot_full": "Full-grid GBDT (tied)", "seq": "Sequential"}
    rows = []
    for m, title in broad.items():
        r = row(macro, level=2, method=m)
        rows.append([title, *numbers(r,"ndcg utility",".4f"), *numbers(r,"recall mrr",".3f"),
                     *numbers(r,"mean_ms p95_ms mean_pairs",".1f"), f"{100*float(r['overrun_rate']):.1f}"])
    table("broad", "lrrrrrrrr", r"Policy & NDCG & $U$ & Recall & MRR & Mean & P95 & Pairs & Over \%", rows)
    rows = []
    for m, title in [("seq", "Full sequential policy"), ("no_agreement", "Remove overlap and rank change"),
                     ("no_budget", "Remove budget features"), ("no_stop", "Continue while feasible")]:
        r = row(macro, level=2, method=m)
        rows.append([title, *numbers(r, "ndcg utility", ".4f"), *numbers(r, "mean_ms", ".2f"),
                     f"{100*float(r['overrun_rate']):.1f}"])
    table("ablation", "lrrrr", r"Policy & NDCG@10 & $U$ & Mean ms & Over \%", rows)
    metrics = read(FOLLOW / "budget_metrics.csv")
    pairs = read(FOLLOW / "paired_differences.csv")
    rows = []
    for m, base, title in [("seq_reuse","seq","Reuse, sequential"),
                           ("one_reuse","one","Reuse, one-shot"),
                           ("revision","one_reuse","Add one revision")]:
        values = [title]
        for lev in range(1,4):
            r = row(pairs, phase="source", level=lev, method=m, comparator=base)
            values.extend(numbers(r, "utility", "+.5f") + numbers(r, "ms", "+.2f"))
        rows.append(values)
    header = r" & \multicolumn{2}{c}{Tight} & \multicolumn{2}{c}{Middle} & \multicolumn{2}{c}{Loose}\\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}Change & $\Delta U$ & $\Delta$ms & $\Delta U$ & $\Delta$ms & $\Delta U$ & $\Delta$ms"
    table("source", "lrrrrrr", header, rows)
    rows = []
    for m, title in METHODS.items():
        values = [title]
        for lev in range(1, 4):
            r = row(metrics, phase="target", fold="macro", level=lev, method=m)
            values.extend(numbers(r,"ndcg utility",".4f") + numbers(r,"mean_ms",".2f"))
        rows.append(values)
    header = r" & \multicolumn{3}{c}{Tight} & \multicolumn{3}{c}{Middle} & \multicolumn{3}{c}{Loose}\\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}Policy & $N$ & $U$ & ms & $N$ & $U$ & ms & $N$ & $U$ & ms"
    table("routing", "lrrrrrrrrr", header, rows)
    rows = []
    for lev, title in enumerate(LEVELS, 1):
        r = row(pairs, phase="target", level=lev, method="revision", comparator="one_reuse")
        t = row(metrics, phase="target", fold="macro", level=lev, method="revision")
        rows.append([title, *numbers(r,"ndcg utility","+.5f"),
                     "[" + ", ".join(numbers(r, "utility_low utility_high", "+.5f")) + "]",
                     *numbers(r,"ms","+.3f"), f"{100*float(t['revision_rate']):.1f}"])
    table("paired", "lrrrrr", r"Budget & $\Delta$NDCG & $\Delta U$ & 95\% interval for $\Delta U$ & $\Delta$ms & Revised \%", rows)
    rows = []
    for d, title in DOMAINS.items():
        deltas, counts = [], []
        for lev in range(1,4):
            a,b = [row(metrics, phase="target", domain=d, fold=d, level=lev, method=m) for m in ("revision","one_reuse")]
            deltas.append(f"{float(a['utility'])-float(b['utility']):+.5f}")
            counts.append(str(round(float(a["revision_rate"])*int(a["n"]))))
        rows.append([title,a["n"],*deltas," / ".join(counts)])
    table("domain","lrrrrr",r"Held out & Queries & Tight $\Delta U$ & Middle $\Delta U$ & Loose $\Delta U$ & Revisions T/M/L",rows)
    prior = read(RESULTS / "router_live_paired.csv")
    rows = []
    for lev,title in enumerate(LEVELS,1):
        r = row(prior,domain="macro",level=lev,comparison="router")
        values = [title]
        for field in ("ndcg","utility"):
            values.extend([f"{float(r[field]):+.4f}", "[" + ", ".join(numbers(r,field+"_low "+field+"_high","+.4f")) + "]"])
        rows.append(values)
    table("root","lrrrr",r"Budget & $\Delta$NDCG & 95\% interval & $\Delta U$ & 95\% interval",rows)
    serving = read(RESULTS / "main_metrics.csv")
    rows = []
    for d,title in DOMAINS.items():
        for lev in range(1,4):
            r = row(serving,domain=d,level=lev,method="seq")
            rows.append([title,r["budget_ms"],*numbers(r,"mean_ms p95_ms",".1f"),f"{r['overrun_count']}/{r['n']}",
                         f"{100*float(r['overrun_rate']):.1f}",*numbers(r,"mean_excess_ms",".2f"),
                         f"{float(r['excess_ms_if_over']):.2f}" if int(r["overrun_count"]) else "---"])
    table("serving","lrrrrrrr",r"Domain & $B$ & Mean & P95 & Over/total & Over \% & Excess/all & Excess/over",rows)
    for fig in ("quality_latency", "router_frontier", "learning_curve"):
        shutil.copyfile(RESULTS / (fig + ".png"), PAPER / (fig + ".png"))


def main():
    tables()
    from report_v6_tables import main as followup_tables
    followup_tables()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["tectonic", "--outdir", tmp, "main.tex"], cwd=PAPER, check=True)
        shutil.copyfile(Path(tmp) / "main.pdf", ROOT / "AdaRetrieve-report.pdf")


if __name__ == "__main__":
    main()
