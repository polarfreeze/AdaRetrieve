"""Bounded retrieval paths, decision-time features and finite-action oracle."""
import copy
import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from retrieval import SMALL, DEPTH, FIRST, FULL, fused
from retrieval import metrics, tokens


ACTIONS = ("stop", "lex64", "dense20", "dense64", "fuse", "rerank8", "rerank24")
PATHS = [(), ("lex64",), ("dense20",), ("dense20","dense64"),
         ("lex64","dense20","dense64","fuse"),
         ("lex64","dense20","dense64","fuse","rerank8"),
         ("lex64","dense20","dense64","fuse","rerank8","rerank24"),
         ("rerank8",), ("rerank8","rerank24")]
ONE_SHOT = PATHS + [("dense20","fuse"), ("dense20","fuse","rerank8"),
                    ("dense20","fuse","rerank8","rerank24")]
LAMBDA, MU = .035, .30


def jaccard(a,b):
    a,b=set(a[:10]),set(b[:10])
    return len(a&b)/max(1,len(a|b))


class State:
    def __init__(self,query,bm):
        self.words=tokens(query)
        self.bm=bm
        self.de=([],[])
        self.rank=bm[0]
        self.prev=self.rank
        self.base=self.rank
        self.scores={}
        self.path=()

    def legal(self):
        if "rerank24" in self.path:
            return ["stop"]
        if "rerank8" in self.path:
            return ["stop"]+(["rerank24"] if len(self.base)>FIRST else [])
        actions=["stop"]
        if not self.path:
            actions.append("lex64")
        if "dense20" not in self.path:
            actions.append("dense20")
        elif "fuse" not in self.path:
            if "dense64" not in self.path:
                actions.append("dense64")
            actions.append("fuse")
        if self.rank:
            actions.append("rerank8")
        return actions

    def next(self,a,get_list,get_scores):
        s=copy.copy(self)
        s.prev=self.rank
        s.path=self.path+(a,)
        if a=="lex64":
            s.bm=get_list(a)
            s.rank=s.bm[0]
        elif a.startswith("dense"):
            s.de=get_list(a)
            s.rank=s.de[0]
        elif a=="fuse":
            s.rank=fused(s.bm[0],s.de[0])
        else:
            end=FIRST if a=="rerank8" else FULL
            batch=s.base[len(s.scores):end]
            s.scores={**s.scores, **dict(zip(batch,get_scores(batch)))}
            s.rank=sorted(s.base[:end],key=lambda k:-s.scores[k])+s.base[end:]
        if not a.startswith("rerank"):
            s.base=s.rank
        return s

    def features(self,budget,elapsed,no_agreement=False,no_budget=False):
        def stats(values):
            x=np.asarray(values[:10],dtype=float)
            if x.size==0: return [0.,0.,0.]
            e=np.exp(x-x.max()); p=e/e.sum()
            return [x[0],x[0]-x[min(1,len(x)-1)],float(-sum(p*np.log(p+1e-20)))]
        return [len(self.words),len(set(self.words)),len(self.rank),len(self.scores),
                *[float(a in self.path) for a in ACTIONS[1:]],*stats(self.bm[1]),*stats(self.de[1]),
                0 if no_agreement else jaccard(self.bm[0],self.de[0]),
                0 if no_agreement else jaccard(self.rank,self.prev),
                0 if no_budget else budget,0 if no_budget else (budget-elapsed)/budget]


def cached(query,row):
    def get(a):
        key="bm" if a in ("initial","lex64") else "de"
        depth=SMALL if a in ("initial","dense20") else DEPTH
        return tuple(x[:depth] for x in row[key])
    return State(query,get("initial")),get,lambda ids:[row["ce"][i] for i in ids]


def utility(ndcg,c,b):
    return ndcg-LAMBDA*c/b-MU*max(0,c/b-1)


def encode(base,index,count):
    return list(base)+[int(i==index) for i in range(count)]


def oracle(dataset,q,row,labels,profile,B):
    s,get,score=cached(dataset.queries[q],row)
    rows=[]
    def visit(s,t):
        u=utility(metrics(s.rank,dataset,labels)["ndcg"],t,B)
        targets={"stop":u}
        for a in s.legal()[1:]:
            cost=profile[a]
            if t+cost<=B:
                targets[a]=visit(s.next(a,get,score),t+cost)
        rows.append((s.path,s.features(B,t),targets))
        return max(targets.values())
    bound=visit(s,profile["initial"])
    one=[]
    for path in ONE_SHOT:
        state=s
        t=profile["initial"]
        for a in path:
            if a not in state.legal() or t+profile[a]>B: break
            state=state.next(a,get,score);t+=profile[a]
        else:
            one.append((ONE_SHOT.index(path),utility(metrics(state.rank,dataset,labels)["ndcg"],t,B)))
    return rows,one,bound


def fit(X,y,weights):
    from sklearn.ensemble import GradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    model=GradientBoostingRegressor(n_estimators=40,max_depth=3,learning_rate=.08,
                                    min_samples_leaf=12,random_state=420)
    with threadpool_limits(limits=1):
        model.fit(np.asarray(X,dtype=np.float32),y,sample_weight=weights)
    return model


def mask(x,kind):
    x=list(x)
    if kind=="no_agreement":x[-4:-2]=[0,0]
    if kind=="no_budget":x[-2:]=[0,0]
    return x


def train_fold(target,sources,profile,budgets,cache,out,train_n):
    out=Path(out); table=out/f"{target}.oracle.jsonl.gz"
    X=[];y=[];w=[];single=[];sy=[];sw=[];groups=[]
    states=actions=pairs=0
    accounting={}; start=time.perf_counter();cpu=time.process_time()
    with gzip.open(table,"wt",encoding="utf-8") as f:
        for ds in sources:
            ids=ds.source_ids()[0][:train_n]
            domain_rows=0
            for i,q in enumerate(ids):
                x0=len(X);s0=len(single)
                row=cache[ds.name][q];pairs+=len(row["ce"])
                for B in budgets:
                    rows,one,_=oracle(ds,q,row,ds.labels("train")[q],profile,B)
                    states+=len(rows)
                    for path,feat,targets in rows:
                        for a,value in targets.items():
                            X.append(encode(feat,ACTIONS.index(a),len(ACTIONS)));y.append(value)
                            groups.append(i);actions+=1;domain_rows+=1
                        f.write(json.dumps([ds.name,q,B,path,feat,targets],separators=(",",":"))+"\n")
                    for a,value in one:
                        single.append(encode(rows[-1][1],a,len(ONE_SHOT)));sy.append(value)
                w.extend([1/(len(ids)*(len(X)-x0))]*(len(X)-x0))
                sw.extend([1/(len(ids)*(len(single)-s0))]*(len(single)-s0))
            accounting[ds.name]={"queries":len(ids),"action_values":domain_rows}
    ledger=dict(queries=sum(x["queries"] for x in accounting.values()),states=states,
                action_values=actions,source_counts=accounting,bytes=table.stat().st_size,
                distinct_scored_pairs_for_these_queries=pairs,wall_s=time.perf_counter()-start,
                process_cpu_s=time.process_time()-cpu,paths_evaluated=states,
                action_values_per_query=actions/sum(x["queries"] for x in accounting.values()))
    (out/f"{target}.oracle_cost.json").write_text(json.dumps(ledger,indent=2))
    models={"seq":fit(X,y,w),"one_shot":fit(single,sy,sw)}
    for kind in ["no_agreement","no_budget"]:
        XX=[mask(x[:-len(ACTIONS)],kind)+x[-len(ACTIONS):] for x in X]
        models[kind]=fit(XX,y,w)
    for n in [64,128]:
        take=np.asarray(groups)<n
        models[f"seq_{n}"]=fit(np.asarray(X)[take],np.asarray(y)[take],np.asarray(w)[take])
    return models
