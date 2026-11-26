import csv
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.sparse import csr_matrix


DOMAINS = ("scifact", "nfcorpus", "fiqa")
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def order(ids, seed):
    return sorted(ids, key=lambda q: hashlib.sha256(f"{seed}|{q}".encode()).digest())


class Collection:
    def __init__(self, root, name):
        self.name = name
        self.splits = {}
        with ZipFile(Path(root) / f"{name}.zip") as z:
            def records(kind):
                return [json.loads(x) for x in z.open(f"{name}/{kind}.jsonl")]
            docs = records("corpus")
            self.ids = [str(d["_id"]) for d in docs]
            self.text = [(d.get("title", "") + " " + d["text"]).strip() for d in docs]
            self.queries = {str(x["_id"]): x["text"] for x in records("queries")}
            # Load relevance values only when a caller explicitly requests a split.
            self.archive = Path(root) / f"{name}.zip"
            self.official = {Path(f).stem for f in z.namelist() if f.endswith(".tsv") and "/qrels/" in f}

    def labels(self, split):
        if split not in self.splits:
            with ZipFile(self.archive) as z:
                raw = z.read(f"{self.name}/qrels/{split}.tsv").decode()
            result = {}
            for row in csv.DictReader(io.StringIO(raw), delimiter="\t"):
                result.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
            self.splits[split] = result
        return self.splits[split]

    def source_ids(self):
        train = order(self.labels("train"), "split-v1")
        if "dev" in self.official:
            val = order(self.labels("dev"), "validation-v1")
        else:
            n = max(1, round(len(train) * .15))
            val, train = train[:n], train[n:]
        return order(train, "train-v1"), val

    def val_labels(self):
        return self.labels("dev" if "dev" in self.official else "train")


def tokens(text):
    return TOKEN.findall(text.lower())


DEPTH = 64
SMALL = 20
FULL = 24
FIRST = 8


def fused(bm, dense):
    score = {}
    for rank in (bm, dense):
        for pos, idx in enumerate(rank):
            score[idx] = score.get(idx, 0) + 1/(61 + pos)
    return sorted(score, key=lambda i: (-score[i], i))


class BM25:
    """Exact Okapi BM25 with Robertson's positive IDF, k1=1.2, b=.75."""
    def __init__(self, texts):
        self.vocab = {}
        starts, indices, frequencies, lengths = [0], [], [], []
        for text in texts:
            words = tokens(text)
            lengths.append(len(words))
            for word, count in Counter(words).items():
                indices.append(self.vocab.setdefault(word, len(self.vocab)))
                frequencies.append(count)
            starts.append(len(indices))
        index = csr_matrix((np.array(frequencies, np.float32), indices, starts),
                           shape=(len(texts), len(self.vocab)))
        lengths = np.array(lengths, np.float32)
        df = np.bincount(index.indices, minlength=index.shape[1])
        idf = np.log1p((len(texts) - df + .5) / (df + .5))
        norm = np.repeat(1.2 * (.25 + .75 * lengths / lengths.mean()), np.diff(index.indptr))
        index.data *= 2.2 / (index.data + norm) * idf[index.indices]
        self.postings = index.T.tocsr()

    def search(self, query, depth):
        terms = [self.vocab[t] for t in tokens(query) if t in self.vocab]
        scores = np.asarray(self.postings[terms].sum(axis=0)).ravel()
        # Tie-break by corpus ordinal, including ties at the truncation boundary.
        matches = np.flatnonzero(scores > 0)
        selected = matches[np.argsort(-scores[matches], kind="stable")[:depth]]
        return selected.tolist(), scores[selected].tolist()


def metrics(rank, collection, labels):
    gains = [max(0, labels.get(collection.ids[int(i)], 0)) for i in rank[:10]]
    def dcg(values):
        return sum(v / math.log2(i + 2) for i, v in enumerate(values))
    ideal = dcg(sorted((max(0,v) for v in labels.values()), reverse=True)[:10])
    positive = sum(v > 0 for v in labels.values())
    return {"ndcg": dcg(gains) / ideal if ideal else 0.,
            "recall": sum(v > 0 for v in gains) / max(1,positive),
            "mrr": next((1 / (i+1) for i,g in enumerate(gains) if g), 0.)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    out = root / "results"
    out.mkdir(exist_ok=True)
    summary = {}
    for domain in DOMAINS:
        ds = Collection(root/"data/verified", domain)
        train, _ = ds.source_ids()
        start = time.perf_counter()
        index = BM25(ds.text)
        build = time.perf_counter() - start
        rows = []
        for q in train[:32]:
            start = time.perf_counter()
            ids, _ = index.search(ds.queries[q], 64)
            elapsed = (time.perf_counter() - start)*1000
            rows.append(dict(query=q, ms=elapsed, **metrics(ids,ds,ds.labels("train")[q])))
        summary[domain] = dict(split="train", index_wall_s=build, queries=len(rows),
                               mean_ndcg=float(np.mean([x["ndcg"] for x in rows])), rows=rows)
        print(domain, summary[domain]["mean_ndcg"], flush=True)
    (out/"cpu_bm25_pilot.json").write_text(json.dumps(summary, indent=2))
