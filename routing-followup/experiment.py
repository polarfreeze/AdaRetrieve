"""Source-only fitting, source confirmation and gated three-fold live measurements."""
from collections import Counter
from itertools import groupby, product
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import gzip
import hashlib
import json
import os
import random
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "project"
OUT = HERE / "output"
sys.path.insert(0, str(PROJECT))

from path_router import Router
from policy import LAMBDA, State, cached, utility
from retrieval import DOMAINS, DEPTH, SMALL, Collection, metrics, order

METHODS = ("seq", "seq_reuse", "one", "one_reuse", "revision")
THRESHOLDS = (None, .05, .025, .01, 0.)
FEATURES = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17]


def save(name, obj):
    path = OUT / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ms(start):
    return (perf_counter() - start) * 1000.


class ReuseEngine:
    def __init__(self, engine):
        self.engine = engine
        self.ds = engine.ds

    def operations(self, text):
        get_original, score = self.engine.operations(text)
        lexical = None

        def get(action):
            nonlocal lexical
            if action not in ("initial", "lex64"):
                return get_original(action)
            if lexical is None:
                lexical = get_original("lex64")
            if action == "initial":
                return lexical[0][:SMALL], lexical[1][:SMALL]
            return lexical
        return get, score


def lexical_benchmark(engine, ids):
    reuse = ReuseEngine(engine)
    rows = []
    # Real queries, fresh operations on every request. Include actual state creation/transition.
    for n, q in enumerate(ids):
        text = engine.ds.queries[q]
        jobs = list(product(range(3), (False, True)))
        random.Random("lex|" + q).shuffle(jobs)
        reference = None
        for rep, use in jobs:
            mode = reuse if use else engine
            start = perf_counter()
            get, score = mode.operations(text)
            s = State(text, get("initial"))
            first = ms(start)
            second = perf_counter()
            end = s.next("lex64", get, score)
            extension = ms(second)
            pair = s.bm, end.bm
            if reference is None:
                reference = pair
            assert pair == reference and s.bm == (end.bm[0][:SMALL], end.bm[1][:SMALL])
            if n >= 4:
                rows.append(dict(q=q, reuse=use, rep=rep, initial=first, lex64=extension))
    return rows


def fit_revisions(target, spec, expected):
    paths = [tuple(p) for p in spec["paths"]]
    prefixes = [p for p in paths if p and p[-1] == "dense20"]
    xs = {p: [] for p in prefixes}
    gains = {p: [] for p in prefixes}
    seen = set()
    B = max(spec["budgets"])
    with gzip.open(PROJECT/"results"/f"{target}.oracle.jsonl.gz","rt") as f:
        rows = (r for r in map(json.loads, f) if r[2] == B)
        for key, group in groupby(rows, lambda r: (r[0],r[1])):
            assert key in expected and key not in seen
            seen.add(key)
            data = {tuple(r[3]): (r[4], r[5]["stop"]) for r in group}
            for pre in prefixes:
                if pre not in data:
                    continue
                f0, base = data[pre]
                xs[pre].append(np.asarray(f0)[FEATURES])
                gains[pre].append({p: v-base+LAMBDA*sum(spec["profile"][a] for a in p[len(pre):])/B
                                   for p, (_,v) in data.items() if p[:len(pre)] == pre})
    assert seen == expected
    models = {}
    for pre in prefixes:
        x = np.array(xs[pre]); mean = x.mean(axis=0); scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.
        X = np.column_stack((np.ones(len(x)),np.clip((x-mean)/scale,-4,4)))
        options = [p for p in paths if p[:len(pre)] == pre]
        W=[]
        for p in options:
            take = [i for i,g in enumerate(gains[pre]) if p in g]
            z = X[take]; y=np.array([gains[pre][i][p] for i in take])
            penalty = np.diag([0.] + [64.] * len(FEATURES))
            W.append(np.linalg.solve(z.T@z+penalty,z.T@y).tolist())
        models["/".join(pre)] = dict(paths=[p[len(pre):] for p in options],
                                      mean=mean.tolist(),scale=scale.tolist(),weights=W)
    return models


class Revision:
    def __init__(self, models, profile):
        self.models={}
        for prefix, mod in models.items():
            paths = [tuple(x) for x in mod["paths"]]
            costs = np.array([sum(profile[a] for a in p) for p in paths])
            self.models[tuple(prefix.split("/"))] = (paths, costs, np.array(mod["mean"]),
                                                      np.array(mod["scale"]), np.array(mod["weights"]))

    def choose(self, s, B, elapsed, original, profile, threshold):
        paths, costs, mean, scale, weights = self.models[s.path]
        x = np.array(s.features(B,elapsed))[FEATURES]
        z = np.clip((x-mean)/scale,-4,4)
        pred = weights[:,0] + weights[:,1:] @ z - LAMBDA*costs/B
        pred[(costs > 0) & (elapsed+costs > B)] = -np.inf
        # Compare with the prefix of the planned suffix that the guard still admits.
        default = (); spent = elapsed
        for a in original:
            spent += profile[a]
            if spent > B: break
            default += (a,)
        idx = int(np.argmax(pred))
        if paths[idx] != default and pred[idx]-pred[paths.index(default)] > threshold:
            return paths[idx], True
        return original, False


def route(s, get, score, B, profile, root, revision, threshold, clock):
    elapsed = clock()
    plan = tuple(root.paths[int(root.scores(s.features(B,elapsed), B, elapsed).argmax())])
    remaining = plan
    changed = False
    while remaining:
        a, remaining = remaining[0], remaining[1:]
        if a not in s.legal() or clock()+profile[a] > B: break
        s = s.next(a,get,score)
        if hasattr(clock, "advance"):
            clock.advance(a)
        if a == "dense20" and threshold is not None:
            remaining, changed = revision.choose(s,B,clock(),remaining,profile,threshold)
    return s, plan, changed


class ProfileClock:
    def __init__(self, profile):
        self.profile = profile
        self.elapsed = profile["initial"]
    def __call__(self):
        return self.elapsed
    def advance(self, a):
        self.elapsed += self.profile[a]


def calibrate(budgets, sources, domains, caches, profile, root, revision):
    trials=[]
    for B in budgets:
        sums = Counter()
        for d in sources:
            ds = domains[d]; labels = ds.val_labels()
            for q in ds.source_ids()[1][:48]:
                for threshold in THRESHOLDS:
                    s,get,score = cached(ds.queries[q],caches[d][q])
                    clock = ProfileClock(profile)
                    s,_,_ = route(s,get,score,B,profile,root,revision,threshold,clock)
                    sums[threshold] += utility(metrics(s.rank,ds,labels[q])["ndcg"],clock(),B)
        best = max(THRESHOLDS,key=lambda h:round(sums[h],10))
        trials.append(dict(budget=B,threshold=best,profiled_mean_utility=[
            dict(threshold=h,utility=sums[h]/(48*len(sources))) for h in THRESHOLDS]))
    return trials


def run_timed(kind, q, B, fold, engine, reuse):
    from study import execute
    legacy = kind in ("one","seq")
    eng = engine if legacy else reuse
    p = fold["old"] if legacy else fold["new"]
    if kind in ("seq","seq_reuse"):
        rank,t,pairs,path = execute(eng,q,"seq",fold["seq"],B,p)
        return rank,dict(ms=t,pairs=pairs,actions=path,planned=None,changed=False)
    start=perf_counter()
    get, score = eng.operations(eng.ds.queries[q])
    s = State(eng.ds.queries[q],get("initial"))
    router = fold["old_root"] if legacy else fold["root"]
    h = fold["thresholds"][B] if kind=="revision" else None
    s,plan,changed=route(s,get,score,B,p,router,fold["revision"],h,lambda: ms(start))
    elapsed=ms(start)
    return s.rank,dict(ms=elapsed,pairs=len(s.scores),actions="/".join(s.path) or "stop",
                       planned="/".join(plan) or "stop",changed=changed)


def benchmark(phase,folds,domains,engines):
    rows=[]
    for target, fold in folds.items():
        for d in ([target] if phase=="target" else fold["sources"]):
            ds = domains[d]; engine = engines[d]; reuse = ReuseEngine(engine)
            if phase=="target":
                labels=ds.labels("test"); ids=order(labels,"test-run-v1")
            else:
                labels=ds.val_labels(); ids=ds.source_ids()[1][48:96]
            for q in ds.source_ids()[0][:3]:
                for m in METHODS:
                    run_timed(m,q,fold["budgets"][-1],fold,engine,reuse)
            chunk=[]
            for q in ids:
                jobs=list(product(METHODS,enumerate(fold["budgets"],1)))
                random.Random(f"reuse1|{phase}|{target}|{d}|{q}").shuffle(jobs)
                for method,(level,B) in jobs:
                    rank, r=run_timed(method,q,B,fold,engine,reuse)
                    n=metrics(rank,ds,labels[q])["ndcg"]
                    chunk.append(dict(fold=target,domain=d,q=q,level=level,budget=B,method=method,
                                      ndcg=n,utility=utility(n,r["ms"],B),**r))
            print(phase,target,d,len(ids),len(chunk),flush=True)
            with (OUT/f"{phase}-{target}-{d}.jsonl").open("w") as f:
                for r in chunk: f.write(json.dumps(r,separators=(",",":"))+"\n")
            rows.extend(chunk)
    return rows


def evaluate_gate(rows):
    def mean(f,l,m,k):
        return np.mean([r[k] for r in rows if r["fold"]==f and r["level"]==l and r["method"]==m])
    def delta(l,m,base,k):
        return np.array([mean(d,l,m,k)-mean(d,l,base,k) for d in DOMAINS])
    tight=delta(1,"revision","one_reuse","utility").mean()
    records=[]
    for lvl in (1,2,3):
        for m,base in (("one_reuse","one"),("seq_reuse","seq"),("revision","one_reuse")):
            u=delta(lvl,m,base,"utility"); t=delta(lvl,m,base,"ms")
            frac=-t.mean()/np.mean([mean(d,lvl,base,"ms") for d in DOMAINS])
            if m=="revision":
                changes=sum(r["changed"] for r in rows if r["method"]==m and r["level"]==lvl)
                passed=lvl in (2,3) and u.mean()>=.003 and sum(u>0)>=2 and tight>=-.005 and changes>0
            else:
                passed=frac>=.05 and sum(t<0)>=2 and u.mean()>=-.005
            records.append(dict(method=m,level=lvl,du=u.mean(),latency_reduction=frac,
                                delta_utility_by_fold=u.tolist(),delta_ms_by_fold=t.tolist(),passed=bool(passed)))
    return dict(test=any(r["passed"] for r in records),criteria=records)


def main():
    import torch, faiss
    from neural import Models, load
    from study import Engine
    OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    faiss.omp_set_num_threads(1)
    save("environment.json",dict(gpu=torch.cuda.get_device_name(0),python=sys.version,
         packages={n:version(n) for n in ("torch","faiss-cpu","scikit-learn","transformers","numpy")},
         seed="reuse1",protocol_sha256=sha(HERE/"protocol.md"),code_sha256=sha(Path(__file__))))
    encoders=Models()
    envs={d:Collection(PROJECT/"data/verified",d) for d in DOMAINS}
    engines={d:Engine(envs[d],encoders) for d in DOMAINS}
    micro={d:lexical_benchmark(engines[d],envs[d].source_ids()[0][:36]) for d in DOMAINS}
    save("lexical_micro.json",micro)
    cache={d:load(PROJECT/"cache"/f"{d}.queries.pkl") for d in DOMAINS}
    folds={}
    for target in DOMAINS:
        sources=[d for d in DOMAINS if d!=target]
        expected={(d,q) for d in sources for q in envs[d].source_ids()[0][:384]}
        specpath=PROJECT/"results"/f"{target}.router.json"
        spec=json.loads(specpath.read_text())
        p,budgets,seq=load(PROJECT/"results"/f"{target}.frozen.pkl")
        assert p==spec["profile"] and budgets==spec["budgets"]
        new=p.copy()
        for a in ("initial","lex64"):
            new[a]=float(np.mean([np.median([r[a] for r in micro[d] if r["reuse"]]) for d in sources]))
        root=Router({**spec,"cost":[sum(new[a] for a in path) for path in spec["paths"]]})
        models=fit_revisions(target,spec,expected)
        revision=Revision(models,new)
        trials=calibrate(budgets,sources,envs,cache,new,root,revision)
        thresholds={x["budget"]:x["threshold"] for x in trials}
        save(f"frozen-{target}.json",dict(target=target,sources=sources,budgets=budgets,profile=new,
             original_profile=p,calibration=trials,models=models,root_sha256=sha(specpath),
             seq_sha256=sha(PROJECT/"results"/f"{target}.frozen.pkl")))
        folds[target]=dict(old=p,new=new,seq=seq,budgets=budgets,root=root,old_root=Router(spec),
                           revision=revision,thresholds=thresholds,sources=sources)
        print("FROZEN",target,thresholds,flush=True)
    source=benchmark("source",folds,envs,engines)
    gate=evaluate_gate(source)
    gate["frozen_hashes"]={d:sha(OUT/f"frozen-{d}.json") for d in DOMAINS}
    save("source-gate.json",gate)
    print("SOURCE_GATE",gate["test"],flush=True)
    if gate["test"]:
        assert all(sha(OUT/f"frozen-{d}.json")==gate["frozen_hashes"][d] for d in DOMAINS)
        benchmark("target",folds,envs,engines)
    print("EXPERIMENT_COMPLETE",flush=True)


if __name__=="__main__":
    main()

