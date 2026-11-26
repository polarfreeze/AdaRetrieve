"""Build verified BEIR archives or restore the inputs recorded in the manifest."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "verified"
ORIGINAL_MD5 = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
    "fiqa": "17918ed23cd04fb15047f73e6c3bd9d9",
}


def get(url, **kwargs):
    response = requests.get(url, timeout=(12, 90), **kwargs)
    response.raise_for_status()
    return response


def fetch(repo, revision, file, metadata):
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}"
    data = get(url).content
    sha = hashlib.sha256(data).hexdigest()
    git_oid = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    expected = metadata.get("lfs", {}).get("oid") or metadata["oid"]
    if len(data) != metadata["size"] or expected != (sha if "lfs" in metadata else git_oid):
        raise ValueError(f"Upstream integrity mismatch: {repo}/{file}")
    return data, dict(url=url, bytes=len(data), upstream_digest=expected, sha256=sha)


def write_archive(path, name, files):
    with ZipFile(path, "w") as archive:
        for key, data in sorted(files.items()):
            info = ZipInfo(f"{name}/{key}", (2021, 1, 1, 0, 0, 0))
            archive.writestr(info, data, compress_type=ZIP_DEFLATED)


def restore_archives():
    manifest = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        path = OUT / entry["file"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
            print(f"{name}: verified existing archive")
            continue
        files = {}
        for source in entry["sources"]:
            url = source["url"]
            data = get(url).content
            if len(data) != source["bytes"] or hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Pinned source integrity mismatch: {url}")
            filename = url.rsplit("/", 1)[-1]
            if filename.endswith(".gz"):
                files[filename[:-3]] = gzip.decompress(data)
            else:
                files[f"qrels/{filename}"] = data
        tmp = path.with_suffix(".part")
        try:
            write_archive(tmp, name, files)
            if hashlib.sha256(tmp.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"{name}: archive checksum differs; restore the original ZIP from the course bundle")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"{name}: restored and verified")


def make_archive(name):
    original = ROOT / "data" / "raw" / f"{name}.zip"
    if original.exists() and hashlib.md5(original.read_bytes()).hexdigest() == ORIGINAL_MD5[name]:
        out = OUT / original.name
        out.write_bytes(original.read_bytes())
        return out, {"format": "original UKP ZIP", "official_md5_verified": True}
    files, sources = {}, []
    for is_qrels in (False, True):
        repo = f"BeIR/{name}" + ("-qrels" if is_qrels else "")
        api = f"https://huggingface.co/api/datasets/{repo}"
        for commit in get(api + "/commits/main").json():
            revision = commit["id"]
            tree = {item["path"]: item for item in get(f"{api}/tree/{revision}").json()}
            wanted = (sorted(f for f in tree if f.endswith(".tsv")) if is_qrels
                      else ["corpus.jsonl.gz", "queries.jsonl.gz"])
            if wanted and all(f in tree for f in wanted):
                break
        else:
            raise ValueError(f"Missing files in {repo}")
        for file in wanted:
            data, source = fetch(repo, revision, file, tree[file])
            sources.append(source)
            if is_qrels:
                files[f"qrels/{file}"] = data
            else:
                files[file[:-3]] = gzip.decompress(data)
    out = OUT / f"{name}.zip"
    write_archive(out.with_suffix(".part"), name, files)
    out.with_suffix(".part").replace(out)
    return out, dict(format="BEIR official Hugging Face files repacked", sources=sources,
                     official_md5_verified=False)


def inspect(name, path):
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"CRC failed: {name}")
        def ids(file):
            with archive.open(f"{name}/{file}.jsonl") as stream:
                values = [str(json.loads(line)["_id"]) for line in stream]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate {file} ID")
            return set(values)
        docs, queries = ids("corpus"), ids("queries")
        splits, members = {}, {}
        for file in sorted(f for f in archive.namelist() if "/qrels/" in f and f.endswith(".tsv")):
            rows = csv.DictReader(io.StringIO(archive.read(file).decode()), delimiter="\t")
            seen, count = set(), 0
            for row in rows:
                q, d = row["query-id"], row["corpus-id"]
                if q not in queries or d not in docs:
                    raise ValueError(f"Unknown ID in {file}")
                seen.add(q)
                count += 1
            key = Path(file).stem
            splits[key] = dict(queries=len(seen), judgments=count)
            members[key] = sorted(seen)
        return dict(documents=len(docs), query_records=len(queries), splits=splits), members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked", action="store_true", help="Restore the exact inputs named by the existing manifest")
    args = parser.parse_args()
    if args.locked:
        restore_archives()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, splits = {}, {}
    for name in ORIGINAL_MD5:
        path, origin = make_archive(name)
        stats, splits[name] = inspect(name, path)
        content = path.read_bytes()
        manifest[name] = {**origin, **stats, "file": path.name, "bytes": len(content),
                          "md5": hashlib.md5(content).hexdigest(),
                          "sha256": hashlib.sha256(content).hexdigest()}
        print(name, stats, flush=True)
    for name, content in [("dataset_manifest", manifest), ("official_split_ids", splits)]:
        (OUT / f"{name}.json").write_text(json.dumps(content, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
