"""Source-only latency admission, held-out candidate-depth and truncation diagnostics."""
import hashlib
import json
import random
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import faiss
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "routing-followup"))
from experiment import FEATURES, ReuseEngine, Revision, Router, State
from neural import CE, ENC, Models
from policy import ACTIONS, LAMBDA
from retrieval import Collection, DOMAINS, fused, metrics, order
from study import Engine

OUT = BASE / "sensitivity-output"
PROJECT = BASE / "project"
FOLLOW = BASE / "routing-followup/output"
QUANTILES = {"p50": .50, "p90": .90, "p95": .95}
CAP = 384
SAMPLES = 96


def ms(start):
    return (perf_counter_ns() - start) / 1e6


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    with (OUT / name).open("x") as file:
        json.dump(obj, file, separators=(",", ":"), allow_nan=False)


def log(*args):
    print(*args, flush=True)


def profiles(engine):
    ds = engine.ds
    reuse = ReuseEngine(engine)
    ids = order(ds.source_ids()[0], "admission-profile-v1")
    assert len(ids) >= SAMPLES + 6
    rows = {}
    for idx, q in enumerate(ids[:SAMPLES + 6]):
        text = ds.queries[q]
        start = perf_counter_ns()
        get, score = reuse.operations(text)
        state = State(text, get("initial"))
        times = {"initial": ms(start)}
        for action in ACTIONS[1:]:
            start = perf_counter_ns()
            state = state.next(action, get, score)
            times[action] = ms(start)
        if idx >= 6:
            rows[q] = times
    return rows


def freeze(engines):
    samples = {d: profiles(engines[d]) for d in DOMAINS}
    save("source-observations.json", samples)
    gate = read(FOLLOW / "source-gate.json")
    seals = {}
    for d in DOMAINS:
        file = FOLLOW / f"frozen-{d}.json"
        fold = read(file)
        rootfile = PROJECT / "results" / f"{d}.router.json"
        assert digest(file) == gate["frozen_hashes"][d] and digest(rootfile) == fold["root_sha256"]
        sources = sorted(set(DOMAINS) - {d})
        assert sorted(fold["sources"]) == sources
        pooled = [r for source in sources for r in samples[source].values()]
        table = {name: {a: float(np.quantile([r[a] for r in pooled], prob))
                        for a in pooled[0]} for name, prob in QUANTILES.items()}
        table["v5"] = fold["profile"]
        name = f"admission-{d}.json"
        save(name, dict(target=d, sources=sources, samples_per_source=SAMPLES,
                        budgets=fold["budgets"], profiles=table, quantile_method="linear",
                        thresholds=fold["calibration"], frozen_v5=digest(file),
                        frozen_root=digest(rootfile), runner=digest(Path(__file__))))
        seals[d] = digest(OUT / name)
    save("freeze-checksums.json", seals)
    log("FROZEN before opening test qrels", seals)
    return seals, samples


def costs(paths, profile):
    return np.array([sum(profile[a] for a in path) for path in paths])


class Admission:
    def __init__(self, fold, spec, guard):
        base = fold["profile"]
        self.root = Router({**spec, "cost": costs(spec["paths"], base)})
        self.revision = Revision(fold["models"], base)
        self.guard = guard
        self.root_times = costs(self.root.paths, guard)
        self.suffix_times = {prefix: costs(model[0], guard)
                             for prefix, model in self.revision.models.items()}

    def run(self, reuse, text, budget, threshold):
        start = perf_counter_ns()
        get, score = reuse.operations(text)
        s = State(text, get("initial"))
        initial_ms = ms(start)
        stage = {"initial": initial_ms}
        then = perf_counter_ns()
        # Feature elapsed is real. Only the original admission mask is disabled here.
        pred = self.root.scores(s.features(budget, initial_ms), budget, -np.inf)
        pred[(self.root_times > 0) & (self.root_times + initial_ms > budget)] = -np.inf
        remaining = tuple(self.root.paths[int(pred.argmax())])
        stage["root"] = ms(then)
        while remaining:
            action, remaining = remaining[0], remaining[1:]
            if action not in s.legal() or ms(start) + self.guard[action] > budget:
                break
            then = perf_counter_ns()
            s = s.next(action, get, score)
            stage[action] = ms(then)
            if action == "dense20" and threshold is not None:
                then = perf_counter_ns()
                t = ms(start)
                paths, base_cost, mean, scale, weights = self.revision.models[s.path]
                z = np.clip((np.asarray(s.features(budget, t))[FEATURES] - mean) / scale, -4, 4)
                values = weights[:, 0] + weights[:, 1:] @ z - LAMBDA * base_cost / budget
                guard_cost = self.suffix_times[s.path]
                values[(guard_cost > 0) & (t + guard_cost > budget)] = -np.inf
                default = ()
                for a in remaining:
                    t += self.guard[a]
                    if t > budget:
                        break
                    default += (a,)
                best = int(values.argmax())
                if paths[best] != default and values[best] - values[paths.index(default)] > threshold:
                    remaining = paths[best]
                stage["revision"] = ms(then)
        duration = ms(start)
        return s.rank[:10], duration, s.path, stage


def guard_test(engine, seal, sampled):
    ds, d = engine.ds, engine.ds.name
    file = OUT / f"admission-{d}.json"
    assert digest(file) == seal
    settings = read(file)
    fold = read(FOLLOW / f"frozen-{d}.json")
    spec = read(PROJECT / "results" / f"{d}.router.json")
    rules = {key: Admission(fold, spec, profile) for key, profile in settings["profiles"].items()}
    thresholds = {r["budget"]: r["threshold"] for r in settings["thresholds"]}
    reuse = ReuseEngine(engine)
    labels = ds.labels("test")
    assert set(sampled).isdisjoint(labels)
    jobs = [(b, key) for b in settings["budgets"] for key in rules]
    for q in ds.source_ids()[0][:3]:
        for b, key in jobs:
            rules[key].run(reuse, ds.queries[q], b, thresholds[b])
    with (OUT / f"guard-{d}.jsonl").open("x") as f:
        for q in order(labels, "sensitivity-guard-v1"):
            rows = []
            sequence = jobs.copy()
            random.Random("guard|" + d + "|" + q).shuffle(sequence)
            for b, key in sequence:
                top, t, path, stages = rules[key].run(reuse, ds.queries[q], b, thresholds[b])
                rows.append([b, key, round(t, 6), top, path,
                             {k: round(v, 6) for k, v in stages.items()},
                             metrics(top, ds, labels[q])["ndcg"]])
            f.write(json.dumps([q, rows], separators=(",", ":")) + "\n")
    log("GUARD_DONE", d, len(labels))


@torch.inference_mode()
def encode(models, texts, cap):
    if cap == 192:
        return models.encode(texts)
    x = models.token(texts, padding=True, truncation=True, max_length=cap, return_tensors="pt").to("cuda")
    h = models.enc(**x).last_hidden_state.float()
    mean = (h * x.attention_mask[:, :, None]).sum(1) / x.attention_mask.sum(1)[:, None].clamp(min=1)
    return F.normalize(mean, dim=1).cpu().numpy()


@torch.inference_mode()
def rerank(models, text, docs, cap):
    if cap == 192:
        return models.score([text] * len(docs), docs)
    x = models.pairtoken([text] * len(docs), docs, padding=True, truncation=True,
                         max_length=cap, return_tensors="pt").to("cuda")
    return models.ce(**x).logits.view(-1).float().cpu().tolist()


def count_lengths(tokenizer, texts, pair=None):
    result = tokenizer(texts, pair, truncation=False, padding=False, return_length=True,
                       return_attention_mask=False, return_token_type_ids=False, verbose=False)
    return np.asarray(result["length"])


def grid_test(engine):
    ds, models = engine.ds, engine.models
    d = ds.name
    # Re-encode every document; both query and document sides use the chosen limit.
    assert min(models.enc.config.max_position_embeddings, models.ce.config.max_position_embeddings) >= CAP
    index = faiss.IndexFlatIP(engine.ix.d)
    start = perf_counter_ns()
    for offset in range(0, len(ds.text), 64):
        index.add(encode(models, ds.text[offset:offset + 64], CAP))
    build_s = ms(start) / 1000
    assert index.ntotal == engine.ix.ntotal == len(ds.text)
    lengths = np.concatenate([count_lengths(models.token, ds.text[i:i+128])
                              for i in range(0, len(ds.text), 128)])
    labels = ds.labels("test")
    qlengths = count_lengths(models.token, [ds.queries[q] for q in labels])
    # Each prefix executes once per query and is charged in full to each suffix.
    settings = [(20, 192), (64, 192), (100, 192), (64, CAP)]
    def query(q, measure):
        query_text = ds.queries[q]
        rng = random.Random("grid|" + d + "|" + q)
        jobs = settings.copy()
        rng.shuffle(jobs)
        prefix, results, pair_counts = [], [], []
        for k, enc in jobs:
            then = perf_counter_ns()
            bm = engine.bm.search(query_text, k)[0]
            vector = encode(models, [query_text], enc)
            ids = (engine.ix if enc == 192 else index).search(vector, k)[1][0].tolist()
            candidates = fused(bm, ids)
            base_ms = ms(then)
            if measure:
                rel = labels[q]
                hits = sum(rel.get(ds.ids[i], 0) > 0 for i in candidates)
                count = sum(x > 0 for x in rel.values())
                prefix.append([k, enc, len(candidates), hits, count, round(base_ms, 6)])
            suffixes = [(depth, 192) for depth in (0, 8, 24, 50)] if enc == 192 else [(0, 192), (24, 192)]
            if k == 64:
                suffixes.append((24, CAP))
            rng.shuffle(suffixes)
            for depth, ce in suffixes:
                rank = candidates
                t = base_ms
                if depth:
                    then = perf_counter_ns()
                    head = candidates[:depth]
                    logits = rerank(models, query_text, [ds.text[i] for i in head], ce)
                    rank = [i for _, i in sorted(zip(logits, head), key=lambda x: -x[0])] + candidates[depth:]
                    t += ms(then)
                if measure:
                    top = rank[:10]
                    results.append([k, enc, depth, ce, min(depth,len(candidates)), round(t, 6),
                                    top, metrics(top, ds, labels[q])["ndcg"]])
            if measure and k == 64 and enc == 192:
                docs = [ds.text[i] for i in candidates[:24]]
                lens = count_lengths(models.pairtoken, [query_text] * len(docs), docs)
                pair_counts = [len(lens), int(sum(lens > 192)), int(sum(lens > CAP))]
        return [q, prefix, results, pair_counts]
    for q in ds.source_ids()[0][:4]:
        query(q, False)
    with (OUT / f"grid-{d}.jsonl").open("x") as file:
        for i, q in enumerate(order(labels, "sensitivity-grid-v1"), 1):
            file.write(json.dumps(query(q, True), separators=(",", ":")) + "\n")
            if i % 200 == 0:
                log("GRID_PROGRESS", d, i, len(labels))
    def counts(values):
        return [len(values), int(sum(values > 192)), int(sum(values > CAP))]
    save(f"length-{d}.json", dict(documents=counts(lengths), queries=counts(qlengths),
                                   documents_reindexed=index.ntotal, build_seconds=build_s))
    log("GRID_DONE", d, len(labels), "reindex_s", round(build_s, 1))


def main():
    OUT.mkdir(exist_ok=False)
    faiss.omp_set_num_threads(1)
    models = Models()
    save("environment.json", dict(gpu=torch.cuda.get_device_name(), models=[ENC, CE],
                                  versions={m: version(m) for m in ("numpy", "torch", "faiss-cpu", "transformers")},
                                  torch_threads=torch.get_num_threads(), faiss_threads=faiss.omp_get_max_threads(),
                                  runner=digest(Path(__file__)), grid_depths=[20,64,100],
                                  rerank_depths=[0,8,24,50], limits=[192,CAP]))
    engines = {d: Engine(Collection(PROJECT / "data/verified", d), models) for d in DOMAINS}
    seals, samples = freeze(engines)
    for d in ("fiqa", "scifact", "nfcorpus"):
        guard_test(engines[d], seals[d], samples[d])
    for d in ("scifact", "nfcorpus", "fiqa"):
        grid_test(engines[d])
    save("complete.json", {f.name: digest(f) for f in sorted(OUT.iterdir())})
    log("SENSITIVITY_COMPLETE", OUT)


if __name__ == "__main__":
    main()
