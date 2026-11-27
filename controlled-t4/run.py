"""Paired T4 fork and fixed-RRF/reranker benchmark with frozen V5 folds."""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import faiss
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "routing-followup"))
from experiment import ReuseEngine, Revision, Router, State
from neural import CE, ENC, Models
from retrieval import Collection, DOMAINS, fused, metrics, order
from study import Engine

L6 = ("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a")
OUTPUT = BASE / "controlled-output"
FIXED = ("RRF", "Tiny8", "L6-8", "Tiny24", "L6-24")
now = perf_counter_ns


def elapsed(start):
    return (now() - start) / 1e6


def timed(stages, name, fn, *args):
    start = now()
    result = fn(*args)
    stages[name] += elapsed(start)
    return result


def run_actions(state, actions, get, score, clock, budget, profile, stages, fork):
    for i, action in enumerate(actions):
        if action not in state.legal() or clock() + profile[action] > budget:
            break
        state = timed(stages, action, state.next, action, get, score)
        if fork and action == "dense20":
            return state, actions[i+1:], True
    return state, (), False


def record(state, total, stages, **extra):
    return dict(top=state.rank[:10], actions=list(state.path), ms=total,
                stages=dict(stages), **extra)


def paired(reuse, text, budget, profile, root, revision, threshold, arms):
    stages = defaultdict(float)
    start = now()
    get, score = reuse.operations(text)
    state = timed(stages, "initial", lambda: State(text, get("initial")))
    def select():
        t = elapsed(start)
        return tuple(root.paths[int(root.scores(state.features(budget, t), budget, t).argmax())])
    plan = timed(stages, "root", select)
    state, suffix, reached = run_actions(state, plan, get, score, lambda: elapsed(start),
                                         budget, profile, stages, True)
    shared_ms = elapsed(start)
    info = dict(plan=plan, prefix=state.path, shared_ms=shared_ms,
                remaining_ms=budget-shared_ms, reached_dense=reached)
    if not reached:
        return {arm:record(state, shared_ms, stages, changed=False, **info) for arm in arms}
    output = {}
    for arm in arms:
        branch = stages.copy()
        path, changed = suffix, False
        begin = now()
        if arm == "Revision" and threshold is not None:
            path, changed = timed(branch, "revision_decision", revision.choose, state, budget,
                                  shared_ms, suffix, profile, threshold)
        end, _, _ = run_actions(state, path, get, score, lambda: shared_ms + elapsed(begin),
                                budget, profile, branch, False)
        total = shared_ms + elapsed(begin)
        output[arm] = record(end, total, branch, changed=bool(changed), **info)
    return output


def fixed(engine, scorers, text, label):
    stages = defaultdict(float)
    start = now()
    get, _ = engine.operations(text)
    bm = timed(stages, "bm25_64", get, "lex64")[0]
    de = timed(stages, "dense64", get, "dense64")[0]
    candidates = timed(stages, "rrf", fused, bm, de)
    rank = candidates
    if label != "RRF":
        depth = 8 if label.endswith("8") else 24
        ids = candidates[:depth]
        score_model = scorers["Tiny" if label.startswith("Tiny") else "L6"]
        def rerank():
            logits = score_model([text]*len(ids), [engine.ds.text[i] for i in ids])
            return [doc for _,doc in sorted(zip(logits,ids),key=lambda x:-x[0])] + candidates[depth:]
        rank = timed(stages, "rerank", rerank)
    total = elapsed(start)
    return dict(ms=total, stages=dict(stages), top=rank[:10]), candidates


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name()
    faiss.omp_set_num_threads(1)
    OUTPUT.mkdir(exist_ok=False)
    gpu=torch.cuda.get_device_name()
    print("HARDWARE", gpu, flush=True)
    models=Models()
    class L6Scorer:
        score=Models.score
    l6=L6Scorer()
    l6.pairtoken=AutoTokenizer.from_pretrained(L6[0], revision=L6[1])
    l6.ce=AutoModelForSequenceClassification.from_pretrained(L6[0],revision=L6[1]).half().cuda().eval()
    scorers={"Tiny":models.score,"L6":l6.score}
    project=BASE/"project"
    follow=BASE/"routing-followup"/"output"
    gate=read(follow/"source-gate.json")
    env=dict(gpu=gpu,models=[ENC,CE,L6],code_sha256=sha(Path(__file__)),max_tokens=192,
             versions={k:version(k) for k in ("numpy","torch","transformers","faiss-cpu")},
             cpu_threads=torch.get_num_threads(),faiss_threads=faiss.omp_get_max_threads(),
             session=os.urandom(12).hex(),clock="perf_counter_ns")
    (OUTPUT/"environment.json").write_text(json.dumps(env,indent=2))
    for d in DOMAINS:
        foldpath=follow/f"frozen-{d}.json"
        fold=read(foldpath)
        rootpath=project/"results"/f"{d}.router.json"
        assert sha(foldpath)==gate["frozen_hashes"][d] and sha(rootpath)==fold["root_sha256"]
        assert set(fold["sources"])==set(DOMAINS)-{d} and fold["target"]==d
        profile=fold["profile"]
        spec=read(rootpath)
        root=Router({**spec,"cost":[sum(profile[a] for a in path) for path in spec["paths"]]})
        revision=Revision(fold["models"],profile)
        budgets=fold["budgets"]
        thresholds={r["budget"]:r["threshold"] for r in fold["calibration"]}
        ds=Collection(project/"data/verified",d)
        engine=Engine(ds,models)
        reuse=ReuseEngine(engine)
        labels=ds.labels("test")
        warm=ds.source_ids()[0][:5]
        assert not set(warm)&set(labels)
        for q in warm:
            for m in FIXED:
                fixed(engine,scorers,ds.queries[q],m)
            for b in budgets:
                paired(reuse,ds.queries[q],b,profile,root,revision,thresholds[b],["OneShot","Revision"])
        with (OUTPUT/f"{d}.jsonl").open("x") as writer:
            for i,q in enumerate(order(labels,"controlled-v1"),1):
                rng=random.Random(f"controlled-v1|{d}|{q}")
                jobs=list(FIXED)+budgets
                rng.shuffle(jobs)
                results=[]; reference=None
                for job in jobs:
                    if isinstance(job,int):
                        arms=["OneShot","Revision"]
                        rng.shuffle(arms)
                        pair=paired(reuse,ds.queries[q],job,profile,root,revision,thresholds[job],arms)
                        results.extend(dict(method=m,budget=job,**r) for m,r in pair.items())
                    else:
                        r,candidates=fixed(engine,scorers,ds.queries[q],job)
                        if reference is None: reference=candidates
                        assert candidates==reference, (d,q,job)
                        results.append(dict(method=job,budget=None,**r))
                for r in results:
                    r["ndcg"]=metrics(r["top"],ds,labels[q])["ndcg"]
                writer.write(json.dumps(dict(q=q,rows=results),separators=(",",":"))+"\n")
                if i%100==0:
                    writer.flush()
                    print("PROGRESS",d,i,len(labels),flush=True)
        print("FOLD_DONE",d,len(labels),flush=True)
    print("ALL_FOLDS_DONE",flush=True)


if __name__=="__main__":
    main()
