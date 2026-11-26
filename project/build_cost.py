"""Time an independent build starting with empty indexes, query caches and action tables."""
import gzip
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import faiss
import torch

import neural
from policy import oracle
from retrieval import Collection


def main():
    source=neural.OUT
    faiss.omp_set_num_threads(1)
    report={"initial_state":"Verified local ZIP files and downloaded model weights available; all experiment indexes, scores and oracle tables empty. Reuse the frozen serving cost profiles and budgets.",
            "gpu":torch.cuda.get_device_name(0),"domains":{}}
    with tempfile.TemporaryDirectory(prefix="adaretrieve-cost-") as folder:
        neural.CACHE=Path(folder)/"cache";neural.OUT=Path(folder)/"results"
        wall=time.perf_counter();cpu=time.process_time()
        neural.prepare()
        report["cache_phase_wall_s"]=time.perf_counter()-wall
        report["replicate_cache_stages"]=json.loads((neural.OUT/"cache_build.json").read_text())
        for target in neural.DOMAINS:
            start=time.perf_counter();cstart=time.process_time();states=values=queries=0
            profile,budgets,_=neural.load(source/f"{target}.frozen.pkl")
            out=Path(folder)/f"{target}.oracle.jsonl.gz"
            with gzip.open(out,"wt",encoding="utf-8") as f:
                for name in neural.DOMAINS:
                    if name==target:continue
                    ds=Collection(neural.DATA,name)
                    cache=neural.load(neural.CACHE/f"{name}.queries.pkl")
                    for q in ds.source_ids()[0][:neural.TRAIN_N]:
                        queries+=1
                        for B in budgets:
                            rows,_,_=oracle(ds,q,cache[q],ds.labels("train")[q],profile,B)
                            for path,feat,targets in rows:
                                states+=1; values+=len(targets)
                                f.write(json.dumps([ds.name,q,B,path,feat,targets],separators=(",",":"))+"\n")
            ledger=json.loads((source/f"{target}.oracle_cost.json").read_text())
            assert (queries,states,values)==(ledger["queries"],ledger["states"],ledger["action_values"])
            report["domains"][target]=dict(wall_s=time.perf_counter()-start,
                cpu_s=time.process_time()-cstart,bytes=out.stat().st_size,queries=queries,states=states,action_values=values)
        report["full_wall_s"]=time.perf_counter()-wall
        report["full_process_cpu_s"]=time.process_time()-cpu
    (source/"cold_build.json").write_text(json.dumps(report,indent=2))
    print("COLD BUILD COMPLETE",report["full_wall_s"],"seconds",flush=True)


if __name__=="__main__":main()
