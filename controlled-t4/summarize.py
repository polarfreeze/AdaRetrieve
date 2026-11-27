"""Recompute all metrics from the CPU-returned rankings and paired T4 clocks."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "project"
sys.path.insert(0, str(PROJECT))
from retrieval import Collection, DOMAINS, metrics

METHODS = ("OneShot", "Revision", "RRF", "Tiny8", "L6-8", "Tiny24", "L6-24")
TITLES = {"scifact":"SciFact", "nfcorpus":"NFCorpus", "fiqa":"FiQA"}
BUDGETS = {"scifact":[17,37,68],"nfcorpus":[17,37,68],"fiqa":[14,30,56]}
LEVELS = ["紧","中","宽"]


def save_csv(name, rows):
    with (HERE/name).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    grouped=defaultdict(list)
    stage_rows=[]
    detail=[]
    meta={}
    result_file=HERE/"AdaRetrieve-controlled-T4-results.zip"
    with ZipFile(result_file) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={"environment.json",*(f"{d}.jsonl" for d in DOMAINS)}
        meta=json.loads(z.read("environment.json"))
        assert "T4" in meta["gpu"] and meta["max_tokens"]==192
        assert meta["code_sha256"]==hashlib.sha256((HERE/"run.py").read_bytes()).hexdigest()
        for d in DOMAINS:
            ds=Collection(PROJECT/"data/verified",d)
            labs=ds.labels("test")
            queries=[json.loads(l) for l in z.read(d+".jsonl").splitlines()]
            assert len(queries)==len(labs) and {q["q"] for q in queries}==set(labs)
            for query in sorted(queries,key=lambda q:q["q"]):
                q=query["q"]
                rows=query["rows"]
                by_key={(r["method"],r["budget"]):r for r in rows}
                assert len(by_key)==len(rows)==11
                for r in rows:
                    assert 0<=len(r["top"])<=10 and len(r["top"])==len(set(r["top"]))
                    assert all(0<=i<len(ds.ids) for i in r["top"])
                    assert abs(metrics(r["top"],ds,labs[q])["ndcg"]-r["ndcg"])<1e-12
                    assert r["ms"]>0 and sum(r["stages"].values())<=r["ms"]+1e-7
                for level,b in enumerate(BUDGETS[d]):
                    one,rev=[by_key[m,b] for m in METHODS[:2]]
                    for k in ("plan","prefix","remaining_ms","shared_ms","reached_dense"):
                        assert one[k]==rev[k]
                    assert one["remaining_ms"]==b-one["shared_ms"] and not one["changed"]
                    for r in (one,rev):
                        assert r["actions"][:len(r["prefix"])]==r["prefix"]
                        assert r["ms"]>=r["shared_ms"]
                    if not rev["reached_dense"]:
                        assert one["ms"]==rev["ms"] and one["top"]==rev["top"] and not rev["changed"]
                    if rev["changed"]: assert rev["prefix"][-1]=="dense20"
                    detail.append(dict(domain=d,budget_ms=b,q=q,reached_dense=int(rev["reached_dense"]),
                                       changed=int(rev["changed"]),shared_ms=rev["shared_ms"],
                                       delta_ndcg=rev["ndcg"]-one["ndcg"]))
                    for m in METHODS:
                        r=by_key[m,b if m in METHODS[:2] else None]
                        n,t=r["ndcg"],r["ms"]
                        excess=max(0.,t-b)
                        u=n-.035*t/b-.30*excess/b
                        grouped[d,level,m].append([n,u,t,float(t>b),excess])
                        if d=="fiqa":
                            stage_rows.append((level,m,r))

    summary=[]; arrays={k:np.asarray(v) for k,v in grouped.items()}
    for d in DOMAINS:
        for level,b in enumerate(BUDGETS[d]):
            for method in METHODS:
                a=arrays[d,level,method];n,u,t,over,excess=a.T
                summary.append(dict(domain=d,level=level+1,budget_ms=b,method=method,queries=len(a),
                                    ndcg=n.mean(),utility=u.mean(),mean_ms=t.mean(),p95_ms=np.quantile(t,.95),
                                    overruns=int(over.sum()),over_pct=100*over.mean(),excess_mean_ms=excess.mean(),
                                    excess_if_over_ms=excess[over>0].mean() if over.sum() else None))
    save_csv("metrics.csv",summary)
    stages=[]
    for level,b in enumerate(BUDGETS["fiqa"]):
        for method in METHODS:
            rows=[r for lev,m,r in stage_rows if lev==level and m==method]
            for stage in sorted(set.union(*(set(r["stages"]) for r in rows)))+["other"]:
                if stage=="other": vals=[r["ms"]-sum(r["stages"].values()) for r in rows]
                else: vals=[r["stages"].get(stage,0.) for r in rows]
                a=np.array(vals);ran=a[a>0]
                stages.append(dict(level=level+1,budget_ms=b,method=method,stage=stage,queries=len(a),
                                   executed=len(ran),mean_all_ms=a.mean(),mean_executed_ms=ran.mean() if len(ran) else None,
                                   p95_executed_ms=np.quantile(ran,.95) if len(ran) else None))
    save_csv("fiqa_stages.csv",stages)
    fiqa_paths=[]
    for level,b in enumerate(BUDGETS["fiqa"]):
        counts=defaultdict(list)
        for lev,m,r in stage_rows:
            if lev==level and m in METHODS[:2]:
                counts[m,"/".join(r["actions"]) or "stop"].append(r)
        for (method,path),rows in counts.items():
            over=[r for r in rows if r["ms"]>b]
            fiqa_paths.append(dict(level=level+1,budget_ms=b,method=method,path=path,queries=len(rows),
                                   overruns=len(over),mean_ms=np.mean([r["ms"] for r in rows]),
                                   mean_overrun_excess_ms=np.mean([r["ms"]-b for r in over]) if over else None,
                                   already_over_at_fork=sum(r["shared_ms"]>b for r in rows)))
    save_csv("fiqa_paths.csv",fiqa_paths)
    pairs=[]
    comparisons=[("Revision","OneShot"),("Revision","RRF"),("OneShot","RRF"),
                 ("L6-8","Tiny8"),("L6-24","Tiny24"),("Tiny8","RRF"),("Tiny24","RRF"),
                 ("L6-8","RRF"),("L6-24","RRF")]
    for level in range(3):
        for arm,base in comparisons:
            estimates=[];bootstraps=[]
            for i,d in enumerate(DOMAINS):
                a=arrays[d,level,arm]-arrays[d,level,base]
                vals=a[:,:3].mean(axis=0)
                rng=np.random.default_rng(3149+i)
                boot=np.concatenate([a[rng.integers(len(a),size=(100,len(a))),:3].mean(axis=1) for _ in range(40)])
                estimates.append(vals);bootstraps.append(boot)
            for d,v,s in zip((*DOMAINS,"macro"),[*estimates,np.mean(estimates,axis=0)],
                             [*bootstraps,np.mean(bootstraps,axis=0)]):
                row=dict(domain=d,level=level+1,method=arm,comparator=base)
                for j,k in enumerate(["ndcg","utility","ms"]):
                    row[k]=v[j];row[k+"_low"],row[k+"_high"]=np.quantile(s[:,j],[.025,.975])
                pairs.append(row)
    save_csv("paired.csv",pairs)
    scan=[]
    for d in DOMAINS:
        for b in BUDGETS[d]:
            selected=[r for r in detail if r["domain"]==d and r["budget_ms"]==b]
            scan.append(dict(domain=d,budget_ms=b,queries=len(selected),dense_forks=sum(r["reached_dense"] for r in selected),
                             revisions=sum(r["changed"] for r in selected)))
    save_csv("forks.csv",scan)
    md=["# Colab T4 三折受控实验","",
        "沿用第 5 版的域外训练模型、预算和阈值。每个测试查询内随机交错执行各项比较，先预热，每个条件测一次。两个路由共用实际执行的根路径与前缀，在首次 dense20 后从同一状态、同一剩余预算分支。两边各计完整前缀耗时及自己的后缀耗时。未到分支点的查询给两边记相同的实际观测。", "",
        "固定方法按每种配置独立计时一次；同一查询五个固定方法的 RRF 候选及顺序逐一断言相同。BM25、稠密检索各最多取 64 篇，RRF 常数为 60。两款交叉编码器都用 FP16、192 tokens、同一深度（8 或 24）与单查询批处理。固定方法的同一计时按三档预算分别计算罚分。", "",
        "效用逐查询计算为 NDCG@10 − 0.035 × t/B − 0.30 × max(0, t/B − 1)。超额仅在超时查询上取均值，单位 ms；无超时显示为 —。每域 P95 直接由该域逐查询耗时估计。", ""]
    md.extend(["方法名中，OneShot 为一次性路由，Revision 为允许一次改道的路由，RRF 为固定倒数排名融合，Tiny 为 TinyBERT-L2，L6 为 MiniLM-L6。模型名末尾的 8 或 24 表示重排深度。", ""])
    for d in DOMAINS:
        md.extend(["## "+TITLES[d],"","|预算 ms|方法|NDCG@10|实测效用|平均 ms|P95 ms|超预算数/总数|超预算率|超时平均超额 ms|",
                   "|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
        for r in (x for x in summary if x["domain"]==d):
            e=r["excess_if_over_ms"]
            md.append("|{budget_ms}|{method}|{ndcg:.4f}|{utility:.4f}|{mean_ms:.2f}|{p95_ms:.2f}|{overruns}/{queries}|{over_pct:.1f}%|".format(**r)+("—" if e is None else f"{e:.2f}")+"|")
        md.append("")
    md.extend(["## 三域等权配对差异","","置信区间来自域内逐查询配对重采样 4,000 次，固定本次硬件会话和这三个域。",
               "","|预算|方法 − 比较对象|ΔNDCG@10 [95% CI]|Δ实测效用 [95% CI]|Δ平均 ms|","|---|---|---:|---:|---:|"])
    for r in pairs:
        if r["domain"]=="macro":
            md.append("|"+LEVELS[r["level"]-1]+"|{method} − {comparator}|{ndcg:+.5f} [{ndcg_low:+.5f}, {ndcg_high:+.5f}]|{utility:+.5f} [{utility_low:+.5f}, {utility_high:+.5f}]|{ms:+.3f}|".format(**r))
    md.extend(["","## FiQA 分阶段计时","",
               "每列为所有 648 个查询的平均贡献（ms），各列及其余开销加总为端到端平均。固定方法的耗时与预算无关。CSV 另列实际执行次数及执行时的平均值和 P95。", "",
               "|预算 ms|方法|初始 BM25|根路由|稠密检索/扩展|词法扩展|融合|改道决策|重排|其余|","|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for level,b in enumerate(BUDGETS["fiqa"],1):
        for m in METHODS:
            if level>1 and m not in METHODS[:2]:continue
            c={r["stage"]:r["mean_all_ms"] for r in stages if r["level"]==level and r["method"]==m}
            cols=[("initial","bm25_64"),("root",),("dense20","dense64"),("lex64",),
                  ("rrf","fuse"),("revision_decision",),("rerank","rerank8","rerank24"),("other",)]
            assert abs(sum(sum(c.get(k,0.) for k in col) for col in cols)-arrays["fiqa",level-1,m][:,2].mean())<1e-8
            md.append("|"+(str(b) if m in METHODS[:2] else "全部")+"|"+m+"|"+"|".join(f"{sum(c.get(k,0.) for k in col):.3f}" for col in cols)+"|")
    (HERE/"results.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("Verified",sum(len(arrays[d,0,"RRF"]) for d in DOMAINS),"queries;",len(detail),"paired forks or shared completions;",len(summary),"domain/budget/method summaries.")


if __name__=="__main__":
    summarize()
