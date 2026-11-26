"""Fit and validate a root-only, full-grid router before opening target judgments."""
import gzip
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from policy import LAMBDA, cached, utility
from retrieval import Collection, DOMAINS, metrics


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RIDGE = 64.0
ROOT_FEATURES = [0, 1, 2, 10, 11, 12]


def write(name, obj):
    dest = RESULTS / name
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(dest)


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def path_name(path):
    return "/".join(path) or "stop"


class Router:
    def __init__(self, spec):
        self.paths = spec["paths"]
        self.mean = np.array(spec["mean"])
        self.scale = np.array(spec["scale"])
        self.weights = np.array(spec["weights"])
        self.cost = np.array(spec["cost"])

    def scores(self, features, budget, elapsed, fixed=False):
        x = np.asarray(features)[ROOT_FEATURES]
        x = np.clip((x - self.mean) / self.scale, -4, 4)
        gains = self.weights[:, 0].copy()
        if not fixed:
            gains += self.weights[:, 1:] @ x
        values = gains - LAMBDA * self.cost / budget
        values[(self.cost > 0) & (elapsed + self.cost > budget)] = -np.inf
        return values


def fit(target, profile, budgets, expected):
    # Each query contributes once per available terminal path. The high-budget
    # table spans the finite grid. Remove its profiled time charge to recover
    # the paired relevance gain relative to stopping at the initial state.
    X, gains = [], []
    seen = set()
    with gzip.open(RESULTS / f"{target}.oracle.jsonl.gz", "rt") as source:
        records = (r for r in map(json.loads, source) if r[2] == max(budgets))
        for key, group in groupby(records, lambda r: (r[0], r[1])):
            assert key in expected and key not in seen
            seen.add(key)
            group = list(group)
            root = next(r for r in group if not r[3])
            X.append(np.asarray(root[4])[ROOT_FEATURES])
            gains.append({tuple(path): labels["stop"] - root[5]["stop"]
                          + LAMBDA * sum(profile[a] for a in path) / max(budgets)
                          for _, _, _, path, _, labels in group})
    assert seen == expected
    X = np.asarray(X)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1
    Z = np.column_stack([np.ones(len(X)), np.clip((X-mean)/scale, -4, 4)])
    paths = sorted(set().union(*(g.keys() for g in gains)))
    weights = []
    for path in paths:
        ids = [i for i, g in enumerate(gains) if path in g]
        z = Z[ids]
        y = np.array([gains[i][path] for i in ids])
        penalty = np.diag([0.] + [RIDGE] * len(ROOT_FEATURES))
        weights.append(np.linalg.solve(z.T @ z + penalty, z.T @ y).tolist())
    return dict(target=target, paths=paths, cost=[sum(profile[a] for a in p) for p in paths],
                mean=mean.tolist(), scale=scale.tolist(), weights=weights, budgets=budgets,
                profile=profile, ridge=RIDGE, root_features=ROOT_FEATURES, n_train=len(X))


def replay(state, path, get, score, profile):
    elapsed = profile["initial"]
    for a in path:
        if a not in state.legal():
            break
        state = state.next(a, get, score)
        elapsed += profile[a]
    return state.rank, elapsed


def validate(target, spec, collections, caches):
    model = Router(spec)
    profile = spec["profile"]
    rows = []
    fixed_values = defaultdict(list)
    for d, ds in collections.items():
        if d == target:
            continue
        labels = ds.val_labels()
        for q in ds.source_ids()[1][:96]:
            root, get, score = cached(ds.queries[q], caches[d][q])
            truth = []
            for path in model.paths:
                ranking, elapsed = replay(root, path, get, score, profile)
                truth.append((metrics(ranking, ds, labels[q])["ndcg"], elapsed))
            for B in spec["budgets"]:
                features = root.features(B, profile["initial"])
                preds = model.scores(features, B, profile["initial"])
                fixed = model.scores(features, B, profile["initial"], fixed=True).argmax()
                win = int(preds.argmax())
                valid = preds[np.isfinite(preds)]
                utilities = [utility(n, t, B) for n, t in truth]
                fixed_values[B].append(utilities)
                rows.append(dict(source=d, q=q, budget=B, chosen=path_name(model.paths[win]),
                                 fixed=path_name(model.paths[fixed]), utility=utilities[win],
                                 fixed_utility=utilities[fixed], stop_utility=utilities[0],
                                 distinct_scores=len(np.unique(valid.round(9))),
                                 spread=float(np.ptp(valid))))
    checks = []
    for B in spec["budgets"]:
        sub = [r for r in rows if r["budget"] == B]
        counts = Counter(r["chosen"] for r in sub)
        fixed_means = np.mean(fixed_values[B], axis=0)
        feasible = (model.cost + profile["initial"] <= B)
        best = int(np.where(feasible, fixed_means, -np.inf).argmax())
        checks.append(dict(budget=B, queries=len(sub), routes=counts,
                           distinguish=sum(r["distinct_scores"] > 1 for r in sub),
                           utility=float(np.mean([r["utility"] for r in sub])),
                           fixed=float(np.mean([r["fixed_utility"] for r in sub])),
                           stop=float(np.mean([r["stop_utility"] for r in sub])),
                           validation_best_fixed=float(fixed_means[best]),
                           validation_best_path=path_name(model.paths[best])))
    return rows, checks


def main():
    collections = {d: Collection(ROOT/"data/verified", d) for d in DOMAINS}
    caches = {}
    for d in DOMAINS:
        filename = f"{d}.queries.pkl"
        if (ROOT/"cache"/filename).exists():
            with (ROOT/"cache"/filename).open("rb") as f:
                caches[d] = pickle.load(f)
        else:
            with ZipFile(ROOT/"artifacts/AdaRetrieve-indexes-and-scores.zip") as z:
                caches[d] = pickle.loads(z.read(filename))
    profiles = json.loads((RESULTS/"profiles.json").read_text())
    manifest = {}
    for target in DOMAINS:
        src = [d for d in DOMAINS if d != target]
        p = {a: float(np.median([np.median(profiles[d][a]) for d in src])) for a in profiles[src[0]]}
        budgets = [int(np.ceil(sum(p.values()) * x)) for x in (.35, .75, 1.4)]
        expected = {(d,q) for d in src for q in collections[d].source_ids()[0][:384]}
        spec = fit(target, p, budgets, expected)
        rows, checks = validate(target, spec, collections, caches)
        passed = all(r["queries"] == 192 and r["distinguish"] == 192 and len(r["routes"]) >= 2 for r in checks)
        write(f"{target}.router_validation.json", rows)
        write(f"{target}.router.json", spec)
        manifest[target] = dict(sources=src, passed=passed, levels=checks,
                                model_sha256=digest(RESULTS/f"{target}.router.json"))
        print(target, "source checks:", [(r["budget"], len(r["routes"]), round(r["utility"]-r["fixed"],4)) for r in checks])
    write("router_gate.json", manifest)
    assert all(m["passed"] for m in manifest.values()), "Source validation gate failed"


if __name__ == "__main__":
    main()
