"""Interleaved Colab T4 comparison using frozen, source-validated policies."""
import json
import platform
import random
import sys
import time
from importlib.metadata import version

import faiss
import torch

from neural import DATA, DOMAINS, OUT, Models, load
from path_router import Router, digest, path_name, write
from policy import State
from retrieval import Collection, metrics, order
from study import Engine, end_ms, execute


def run_router(engine, q, B, profile, model):
    start = time.perf_counter()
    text = engine.ds.queries[q]
    get, score = engine.operations(text)
    state = State(text, get("initial"))
    elapsed = end_ms(start)
    features = state.features(B, elapsed)
    path = model.paths[int(model.scores(features, B, elapsed).argmax())]
    for a in path:
        if a not in state.legal() or end_ms(start) + profile[a] > B:
            break
        state = state.next(a, get, score)
    ms = end_ms(start)
    return state.rank, ms, len(state.scores), path_name(state.path), path_name(path)


def main():
    faiss.omp_set_num_threads(1)
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)
    gate = json.loads((OUT/"router_gate.json").read_text())
    folds = {}
    for d in DOMAINS:
        f = OUT/f"{d}.router.json"
        assert gate[d]["passed"] and gate[d]["model_sha256"] == digest(f)
        p, budgets, seq = load(OUT/f"{d}.frozen.pkl")
        spec = json.loads(f.read_text())
        assert budgets == spec["budgets"] and spec["profile"] == p
        folds[d] = p, budgets, seq, Router(spec)
    models = Models()
    env = dict(gpu=torch.cuda.get_device_name(0), python=sys.version.split()[0],
               platform=platform.platform(), clock="perf_counter", threads_faiss=1,
               packages={p:version(p) for p in ("scikit-learn","torch","transformers","faiss-cpu")})
    write("router_t4_environment.json", env)
    for d in DOMAINS:
        p, budgets, seq, model = folds[d]
        ds = Collection(DATA, d)
        engine = Engine(ds, models)
        for q in ds.source_ids()[0][:3]:
            execute(engine, q, "fixed_hybrid24", seq, budgets[-1], p)
            execute(engine, q, "seq", seq, budgets[-1], p)
            run_router(engine, q, budgets[-1], p, model)
        labels = ds.labels("test")
        dest = OUT/f"{d}.router_live.jsonl"
        tmp = dest.with_suffix(".part")
        with tmp.open("w", encoding="utf-8") as f:
            for k,q in enumerate(order(labels, "test-run-v1"), 1):
                jobs = [(m,b) for b in budgets for m in ("router","seq")] + [("fixed_hybrid",None)]
                random.Random(f"router2|{d}|{q}").shuffle(jobs)
                rows = []
                for m,b in jobs:
                    if m == "router":
                        rank, ms, pairs, path, intended = run_router(engine,q,b,p,model)
                    else:
                        rank, ms, pairs, path = execute(engine,q,m,seq,b,p)
                        intended = None
                    rows.append(dict(method=m,budget=b,ms=ms,pairs=pairs,actions=path,
                                     intended=intended,**metrics(rank,ds,labels[q])))
                f.write(json.dumps(dict(q=q,rows=rows),separators=(",",":"))+"\n")
                if k % 100 == 0:
                    print("LIVE", d, k, "/", len(labels), flush=True)
        tmp.replace(dest)
        print("FOLD_COMPLETE",d,len(labels),digest(dest),flush=True)


if __name__ == "__main__":
    main()
