import gzip
import json
import os
import platform
import random
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from neural import (CACHE, DATA, OUT, DOMAINS, TRAIN_N, VAL_N, SMALL, DEPTH, FIRST,
                    FULL, Models, dump, fused, load)
from policy import (ACTIONS, ONE_SHOT, State, cached, encode, jaccard, mask, oracle,
                    train_fold, utility)
from retrieval import Collection, metrics, order


def end_ms(start):
    return (time.perf_counter()-start)*1000


class Engine:
    def __init__(self,ds,models):
        self.ds,self.models=ds,models
        self.bm=load(CACHE/f"{ds.name}.bm.pkl")
        self.ix=faiss.read_index(str(CACHE/f"{ds.name}.faiss"))

    def operations(self,query):
        vector=None
        def get(a):
            nonlocal vector
            if a in ("initial","lex64"):
                return self.bm.search(query,SMALL if a=="initial" else DEPTH)
            if vector is None:
                vector=self.models.encode([query])
            sc,ids=self.ix.search(vector,SMALL if a=="dense20" else DEPTH)
            return ids[0].tolist(),sc[0].tolist()
        def score(ids):
            return self.models.score([query]*len(ids),[self.ds.text[i] for i in ids])
        return get,score

    def profile(self):
        samples={a:[] for a in ("initial",*ACTIONS[1:])}
        for i,q in enumerate(self.ds.source_ids()[0][:36]):
            get,score=self.operations(self.ds.queries[q])
            start=time.perf_counter()
            s=State(self.ds.queries[q],get("initial"))
            if i>=4: samples["initial"].append(end_ms(start))
            for a in ACTIONS[1:]:
                start=time.perf_counter();s=s.next(a,get,score)
                if i>=4:samples[a].append(end_ms(start))
        return samples


def decide(kind,models,s,B,t,profile):
    feat=s.features(B,t)
    allowed=[a for a in s.legal() if a=="stop" or t+profile[a]<=B]
    if kind=="adaptive_depth":
        allowed=[a for a in allowed if a=="stop" or a.startswith("rerank")]
    if kind=="no_stop" and len(allowed)>1:allowed.remove("stop")
    if len(allowed)==1:return allowed[0]
    model=models.get(kind,models["seq"])
    xx=[encode(mask(feat,kind),ACTIONS.index(a),len(ACTIONS)) for a in allowed]
    return allowed[int(model.predict(np.asarray(xx,dtype=np.float32)).argmax())]


def oneshot(model,s,B,t,profile):
    possible=[i for i,p in enumerate(ONE_SHOT) if not p or t+sum(profile[a] for a in p)<=B]
    xx=[encode(s.features(B,t),i,len(ONE_SHOT)) for i in possible]
    return ONE_SHOT[possible[int(model.predict(np.asarray(xx,dtype=np.float32)).argmax())]]


def simulate(ds,q,row,label,models,kind,B,profile):
    s,get,score=cached(ds.queries[q],row); t=profile["initial"]
    while True:
        a=decide(kind,models,s,B,t,profile)
        if a=="stop":break
        s=s.next(a,get,score);t+=profile[a]
    return utility(metrics(s.rank,ds,label)["ndcg"],t,B)


def execute(engine,q,kind,models,B,profile):
    start=time.perf_counter()
    query=engine.ds.queries[q]
    get,score=engine.operations(query)
    if kind.startswith("fixed") or kind=="confidence":
        calls=0
        if kind=="fixed_dense":
            rank=get("dense64")[0]
        elif kind=="fixed_bm25":
            rank=get("initial")[0]
        else:
            bm,de=get("lex64")[0],get("dense64")[0]
            rank=fused(bm,de)
            k={"fixed_hybrid":0,"fixed_hybrid8":FIRST,"fixed_hybrid24":FULL}.get(kind,0)
            if kind=="confidence" and jaccard(bm,de)<.25:
                k=FIRST if end_ms(start)+profile["rerank8"]<=B else 0
            if k:
                vals=score(rank[:k]);calls=len(vals)
                rank=[i for _,i in sorted(zip(vals,rank[:k]),key=lambda x:-x[0])]+rank[k:]
        return rank,end_ms(start),calls,kind
    s=State(query,get("initial"))
    if kind=="one_shot":
        p=oneshot(models[kind],s,B,end_ms(start),profile)
        for a in p:
            if a not in s.legal() or end_ms(start)+profile[a]>B: break
            s=s.next(a,get,score)
    else:
        if kind=="adaptive_depth":
            # Fixed retrieval setup; the learned choices control reranking depth only.
            for a in ("lex64","dense20","dense64","fuse"):
                s=s.next(a,get,score)
        while True:
            a=decide(kind,models,s,B,end_ms(start),profile)
            if a=="stop":break
            s=s.next(a,get,score)
    return s.rank,end_ms(start),len(s.scores),"/".join(s.path) or "stop"


def write(path,obj):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def main():
    faiss.omp_set_num_threads(1)
    ds={d:Collection(DATA,d) for d in DOMAINS}
    models=Models()
    engines={d:Engine(c,models) for d,c in ds.items()}
    cache={d:load(CACHE/f"{d}.queries.pkl") for d in DOMAINS}
    profpath=OUT/"profiles.json"
    if not profpath.exists():
        write(profpath,{d:e.profile() for d,e in engines.items()})
    profiles=json.loads(profpath.read_text())
    for target in DOMAINS:
        frozen=OUT/f"{target}.frozen.pkl"
        if frozen.exists():continue
        src=[d for d in DOMAINS if d!=target]
        p={a:float(np.median([np.median(profiles[d][a]) for d in src]))
           for a in profiles[target]}
        full=sum(p.values())
        budgets=sorted(set(max(1,int(np.ceil(x))) for x in (.35*full,.75*full,1.4*full)))
        policy=train_fold(target,[ds[d] for d in src],p,budgets,cache,OUT,TRAIN_N)
        validation=[]
        for d in src:
            c=ds[d]
            for q in c.source_ids()[1][:VAL_N]:
                label=c.val_labels()[q];row=cache[d][q]
                for B in budgets:
                    bound=oracle(c,q,row,label,p,B)[-1]
                    for kind in ("seq_64","seq_128","seq"):
                        u=simulate(c,q,row,label,policy,kind,B,p)
                        validation.append(dict(domain=d,q=q,budget=B,method=kind,utility=u,
                                               oracle_utility=bound,gap=bound-u))
        write(OUT/f"{target}.source_validation.json",validation)
        dump(frozen,(p,budgets,policy))
        print("FROZEN",target,budgets,flush=True)
    assert all((OUT/f"{d}.frozen.pkl").exists() for d in DOMAINS)
    for d,e in engines.items():
        p,budgets,policy=load(OUT/f"{d}.frozen.pkl")
        labels=ds[d].labels("test")
        path=OUT/f"{d}.test.jsonl"
        done=set()
        if path.exists():
            # Each line is an atomic query checkpoint. Repair an incomplete last line.
            with path.open("rb+") as f:
                while True:
                    pos=f.tell();line=f.readline()
                    if not line:break
                    try: done.add(json.loads(line)["q"])
                    except (ValueError,KeyError):
                        if f.read(1):raise
                        f.seek(pos);f.truncate();break
        modes=["seq","one_shot","adaptive_depth","confidence","no_stop","no_agreement","no_budget"]
        fixed=["fixed_bm25","fixed_dense","fixed_hybrid","fixed_hybrid8","fixed_hybrid24"]
        warm=ds[d].source_ids()[0][:3]
        for q in warm:
            for method in ("fixed_hybrid24","seq","one_shot"):
                execute(e,q,method,policy,budgets[-1],p)
        with path.open("a",encoding="utf-8") as f:
            for q in order(labels,"test-run-v1"):
                if q in done:continue
                jobs=[(m,B) for B in budgets for m in modes]+[(m,None) for m in fixed]
                random.Random(f"{d}-{q}").shuffle(jobs)
                rows=[]
                for m,B in jobs:
                    rank,ms,n,actions=execute(e,q,m,policy,B,p)
                    rows.append(dict(method=m,budget=B,ms=ms,pairs=n,actions=actions,
                                     **metrics(rank,ds[d],labels[q])))
                f.write(json.dumps(dict(q=q,rows=rows),separators=(",",":"))+"\n"); f.flush()
                done.add(q)
                if len(done)%50==0:print("TEST",d,len(done),"/",len(labels),flush=True)
    print("THREE FOLDS COMPLETE",flush=True)


if __name__=="__main__":
    main()
