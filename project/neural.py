import json
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from retrieval import BM25, Collection, DOMAINS, SMALL, DEPTH, FIRST, FULL, fused


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/verified"
OUT = ROOT / "results"
CACHE = ROOT / "cache"
TRAIN_N = 384
VAL_N = 96
ENC = ("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", "b207367332321f8e44f96e224ef15bc607f4dbf0")
CE = ("cross-encoder/ms-marco-TinyBERT-L2-v2", "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc")


def dump(path, obj):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        pickle.dump(obj, f, protocol=5)
    os.replace(temp, path)


def load(path):
    with path.open("rb") as f:
        return pickle.load(f)


class Models:
    def __init__(self):
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("A Colab T4 is required")
        self.token = AutoTokenizer.from_pretrained(ENC[0], revision=ENC[1])
        self.enc = AutoModel.from_pretrained(ENC[0], revision=ENC[1]).half().cuda().eval()
        self.pairtoken = AutoTokenizer.from_pretrained(CE[0], revision=CE[1])
        self.ce = AutoModelForSequenceClassification.from_pretrained(CE[0], revision=CE[1]).half().cuda().eval()

    @torch.inference_mode()
    def encode(self, texts):
        x = self.token(texts, padding=True, truncation=True, max_length=192,return_tensors="pt").to("cuda")
        hidden = self.enc(**x).last_hidden_state.float()
        mean = (hidden * x.attention_mask[:,:,None]).sum(1) / x.attention_mask.sum(1)[:,None].clamp(min=1)
        return F.normalize(mean, dim=1).cpu().numpy()

    @torch.inference_mode()
    def score(self, queries, docs):
        if not docs:
            return []
        x = self.pairtoken(queries, docs, padding=True,truncation=True,max_length=192,return_tensors="pt").to("cuda")
        return self.ce(**x).logits.view(-1).float().cpu().tolist()


def stage_times(book, name, fn):
    wall, cpu = time.perf_counter(), time.process_time()
    result = fn()
    book[name] = {"wall_s":time.perf_counter()-wall, "process_cpu_s":time.process_time()-cpu}
    return result


def save_book(book):
    path = OUT / "cache_build.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(book, indent=2))
    tmp.replace(path)


def candidates(row):
    bm, de = row["bm"][0], row["de"][0]
    ids = set(bm[:FULL]) | set(de[:FULL])
    for b in (SMALL, DEPTH):
        for d in (SMALL, DEPTH):
            ids.update(fused(bm[:b],de[:d])[:FULL])
    return sorted(ids)


def prepare():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    faiss.omp_set_num_threads(1)
    ledger = OUT / "cache_build.json"
    book = json.loads(ledger.read_text()) if ledger.exists() else {}
    models = stage_times(book, "model_loading", Models)
    book["device"] = torch.cuda.get_device_name(0)
    book["torch"] = str(torch.__version__)
    book["encoders"] = [ENC, CE]
    book["note"] = "Stage wall time on a T4-backed runtime; includes tokenization and I/O. GPU-stage wall time is not kernel time."
    split_ids = json.loads((DATA / "official_split_ids.json").read_text())
    for domain in DOMAINS:
        ds = Collection(DATA, domain)
        log = book.setdefault(domain,{})
        bm_path, vec_path = CACHE/f"{domain}.bm.pkl", CACHE/f"{domain}.faiss"
        if not bm_path.exists():
            def build_bm():
                idx = BM25(ds.text)
                dump(bm_path, idx)
            stage_times(log,"bm25",build_bm)
        if not vec_path.exists():
            def build_dense():
                idx = faiss.IndexFlatIP(384)
                for k in range(0,len(ds.text),96):
                    idx.add(models.encode(ds.text[k:k+96]))
                faiss.write_index(idx,str(vec_path.with_suffix(".part")))
                vec_path.with_suffix(".part").replace(vec_path)
            stage_times(log,"corpus_encoding_and_faiss",build_dense)
        bm, ix = load(bm_path), faiss.read_index(str(vec_path))
        print(domain,"indexes ready",log, flush=True)
        source, val = ds.source_ids()
        # Test query IDs only. Target judgments are opened by the final evaluator.
        qids = list(dict.fromkeys(source[:TRAIN_N]+val[:VAL_N]+split_ids[domain]["test"]))
        wall, cpu = time.perf_counter(), time.process_time()
        record_path = CACHE/f"{domain}.queries.pkl"
        cache = load(record_path) if record_path.exists() else {}
        missing = [q for q in qids if q not in cache]
        for start in range(0,len(missing),16):
            qs = missing[start:start+16]
            scores, ids = ix.search(models.encode([ds.queries[q] for q in qs]), DEPTH)
            batch = {q:dict(bm=bm.search(ds.queries[q],DEPTH),
                            de=(ids[i].tolist(),scores[i].tolist()), ce={}) for i,q in enumerate(qs)}
            pairs = [(q,k) for q,row in batch.items() for k in candidates(row)]
            for p in range(0,len(pairs),64):
                part = pairs[p:p+64]
                preds = models.score([ds.queries[q] for q,_ in part],[ds.text[k] for _,k in part])
                for (q,k), v in zip(part,preds):
                    batch[q]["ce"][k] = v
            cache.update(batch)
            if start%128==0 or start+16>=len(missing):
                dump(record_path,cache)
                print(domain, len(cache),"/",len(qids),"query caches",flush=True)
        log.setdefault("query_cache_runs",[]).append(dict(new_queries=len(missing),
            wall_s=time.perf_counter()-wall, process_cpu_s=time.process_time()-cpu))
        log.update(queries=len(cache), scored_pairs=sum(len(row["ce"]) for row in cache.values()),
                   index_bytes=bm_path.stat().st_size+vec_path.stat().st_size,
                   query_cache_bytes=record_path.stat().st_size)
        save_book(book)
    print("CACHE PREPARATION COMPLETE",flush=True)


if __name__=="__main__":
    prepare()
