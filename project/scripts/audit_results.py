"""Check split isolation, every serialized Bellman target and complete live query records."""
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retrieval import Collection, DOMAINS


def main():
    data=ROOT/"data/verified";out=ROOT/"results"
    ids=json.loads((data/"official_split_ids.json").read_text())
    expected={d:set(Collection(data,d).source_ids()[0][:384]) for d in DOMAINS}
    cold=json.loads((out/"cold_build.json").read_text())
    profiles=json.loads((out/"profiles.json").read_text())
    gate_path=out/"router_gate.json"
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else None
    report={}
    for target in DOMAINS:
        sources=set(DOMAINS)-{target}
        profile={a:median([median(profiles[d][a]) for d in sources]) for a in profiles[target]}
        over_budget=0
        groups=set();query_ids=set();budgets=set();states=values=edges=0;depth=Counter()
        with gzip.open(out/f"{target}.oracle.jsonl.gz","rt",encoding="utf-8") as f:
            lines=(json.loads(line) for line in f)
            for key,chunk in groupby(lines,key=lambda r:(r[0],r[1],r[2])):
                d,q,B=key
                assert d in sources and q in expected[d] and q not in ids[d]["test"]
                assert key not in groups
                groups.add(key);query_ids.add((d,q));budgets.add(B)
                by_path={}
                for _,_,_,path,features,targets in chunk:
                    path=tuple(path)
                    over_budget += profile["initial"]+sum(profile[a] for a in path)>B
                    assert path not in by_path and "stop" in targets
                    assert all(math.isfinite(v) for v in (*features,*targets.values()))
                    by_path[path]=targets;values+=len(targets);states+=1;depth[len(path)]+=1
                assert () in by_path
                for path,actions in by_path.items():
                    for a,value in actions.items():
                        if a=="stop":continue
                        assert abs(value-max(by_path[path+(a,)].values()))<1e-9
                        edges+=1
        assert over_budget==0
        assert len(budgets)==3 and query_ids=={(d,q) for d in sources for q in expected[d]}
        assert len(groups)==3*len(query_ids)
        cost=json.loads((out/f"{target}.oracle_cost.json").read_text())
        for ledger in (cost,cold["domains"][target]):
            assert (ledger["queries"],ledger["states"],ledger["action_values"])==(len(query_ids),states,values)
        if "action_values_per_query" in cost:
            assert math.isclose(cost["action_values_per_query"], values / len(query_ids))
        assert cost["bytes"]==(out/f"{target}.oracle.jsonl.gz").stat().st_size
        base=[json.loads(line) for line in (out/f"{target}.test.jsonl").read_text().splitlines()]
        matched=json.loads((out/f"{target}.matched.json").read_text())
        matches={r["q"]:r["rows"] for r in matched};test=set(ids[target]["test"])
        assert len(test)==len(base)==len(matched) and {r["q"] for r in base}==test==set(matches)
        for sample in base:
            runs=sample["rows"]+matches[sample["q"]]
            assert len(runs)==29==len({(r["method"],r["budget"]) for r in runs})
            for r in runs:
                assert math.isfinite(r["ms"]) and r["ms"]>0 and 0<=r["pairs"]<=24
                assert r["budget"] is None or r["budget"] in budgets
                assert all(0<=r[k]<=1.000000001 for k in ("ndcg","recall","mrr"))
        report[target]=dict(source_queries=len(query_ids),state_rows=states,
                            supervised_actions=values,checked_continuation_values=edges,
                            over_budget_oracle_states=over_budget,
                            path_lengths=dict(sorted(depth.items())),test_queries=len(test),
                            timed_decisions_or_fixed_runs=29*len(test),budgets_ms=sorted(budgets))
        if gate is not None:
            model=out/f"{target}.router.json"
            assert gate[target]["passed"] and hashlib.sha256(model.read_bytes()).hexdigest()==gate[target]["model_sha256"]
            spec=json.loads(model.read_text())
            assert len(spec["paths"])==30 and spec["n_train"]==len(query_ids)
            assert set(spec["budgets"])==budgets and set(gate[target]["sources"])==sources
            valid={d:set(Collection(data,d).source_ids()[1][:96]) for d in sources}
            samples=json.loads((out/f"{target}.router_validation.json").read_text())
            expected_val={(d,q,b) for d in sources for q in valid[d] for b in budgets}
            assert len(samples)==len(expected_val) and {(r["source"],r["q"],r["budget"]) for r in samples}==expected_val
            for d in sources:
                assert valid[d].isdisjoint(expected[d]) and valid[d].isdisjoint(ids[d]["test"])
            for entry in gate[target]["levels"]:
                group=[r for r in samples if r["budget"]==entry["budget"]]
                assert len(group)==entry["queries"]==entry["distinguish"]==192
                assert all(r["distinct_scores"]>1 and r["spread"]>0 for r in group)
                assert Counter(r["chosen"] for r in group)==entry["routes"]
                assert len(entry["routes"])>=2
                assert abs(sum(r["utility"] for r in group)/len(group)-entry["utility"])<1e-10
            report[target]["source_validated_router_paths"]=len(spec["paths"])
            report[target]["source_validation_query_budget_records"]=len(samples)
    (out/"audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
