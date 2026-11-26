"""One-shot control over every observed feasible terminal path in the frozen oracle.

This extra control uses the same source supervision, model class and hardware.
The model picks its entire path from the initial state.
"""
import gzip
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from neural import DATA, DOMAINS, OUT, Models, dump, load
from policy import State, encode, fit
from retrieval import Collection, metrics, order
from study import Engine, end_ms, write


def train(d):
    grouped=defaultdict(list)
    choices=set()
    with gzip.open(OUT/f"{d}.oracle.jsonl.gz","rt") as file:
        for line in file:
            name,q,b,path,feat,targets=json.loads(line)
            path=tuple(path);choices.add(path)
            grouped[name,q,b].append((path,targets["stop"],feat if not path else None))
    paths=sorted(choices)
    by_query=defaultdict(int)
    for (name,q,b),entries in grouped.items():by_query[name,q]+=len(entries)
    X=[];y=[];w=[]
    for (name,q,b),entries in grouped.items():
        base=next(f for p,_,f in entries if not p)
        for path,val,_ in entries:
            X.append(encode(base,paths.index(path),len(paths)));y.append(val)
            w.append(1/by_query[name,q])
    return paths,fit(X,y,w)


def run_match(engine,q,B,profile,model,paths):
    start=time.perf_counter()
    text=engine.ds.queries[q]; get,score=engine.operations(text)
    s=State(text,get("initial"));t=end_ms(start)
    indices=[i for i,path in enumerate(paths) if not path or t+sum(profile[a] for a in path)<=B]
    base=s.features(B,t)
    preds=model.predict(np.asarray([encode(base,i,len(paths)) for i in indices],dtype=np.float32))
    chosen=paths[indices[int(preds.argmax())]]
    for a in chosen:
        if a not in s.legal() or end_ms(start)+profile[a]>B:break
        s=s.next(a,get,score)
    return s.rank,end_ms(start),len(s.scores),"/".join(s.path) or "stop"


def main():
    faiss.omp_set_num_threads(1)
    models=Models()
    for d in DOMAINS:
        f=OUT/f"{d}.matched.pkl"
        if not f.exists():dump(f,train(d))
    for d in DOMAINS:
        c=Collection(DATA,d);e=Engine(c,models)
        paths,model=load(OUT/f"{d}.matched.pkl")
        p,budgets,_=load(OUT/f"{d}.frozen.pkl")
        warm=c.source_ids()[0][:3]
        for q in warm:run_match(e,q,budgets[-1],p,model,paths)
        rows=[]
        labels=c.labels("test")
        for q in order(labels,"test-run-v1"):
            values=[]
            for b in budgets:
                rank,ms,n,path=run_match(e,q,b,p,model,paths)
                values.append(dict(method="one_shot_full",budget=b,ms=ms,pairs=n,actions=path,
                                   **metrics(rank,c,labels[q])))
            rows.append(dict(q=q,rows=values))
        write(OUT/f"{d}.matched.json",rows)
        print("MATCHED",d,len(paths),"terminal path choices",len(rows),"queries",flush=True)


if __name__=="__main__":main()
