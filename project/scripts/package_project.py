"""Bundle the paper, original experiments and frozen routing follow-up in their runnable layout."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT.parent
FOLLOW = BASE / "routing-followup"
PAPER = PROJECT / "paper"
DATA = PROJECT / "data" / "verified"


def inputs():
    yield BASE / "README.md"
    yield from (PROJECT / n for n in ("README.md", "AdaRetrieve-report.pdf", "AdaRetrieve.ipynb"))
    yield from PROJECT.glob("*.py")
    yield from (PROJECT / "scripts").glob("*.py")
    manifest = DATA / "dataset_manifest.json"
    yield manifest
    yield DATA / "official_split_ids.json"
    for entry in json.loads(manifest.read_text(encoding="utf-8")).values():
        path = DATA / entry["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit("Restore the verified datasets with project/scripts/prepare_data.py --locked first.")
        yield path
    yield from (p for p in (PROJECT / "results").iterdir()
                if p.suffix in (".csv", ".json", ".jsonl", ".png") or p.name.endswith((".oracle.jsonl.gz", ".frozen.pkl")))
    yield from (p for p in PAPER.iterdir() if p.suffix in (".sty", ".tex", ".png"))
    yield from (FOLLOW / n for n in ("README.md", "results.md", "experiment.py", "summarize_followup.py",
                "protocol.md", "routing-followup.ipynb"))
    yield from (p for p in (FOLLOW / "output").iterdir() if p.suffix in (".json", ".jsonl", ".csv"))


def main():
    dest = PROJECT / "artifacts/AdaRetrieve-project.zip"
    dest.parent.mkdir(exist_ok=True)
    tmp = dest.with_suffix(".part")
    paths = sorted(inputs())
    hashes = {}
    with ZipFile(tmp, "w", ZIP_DEFLATED) as z:
        for p in paths:
            name = p.relative_to(BASE).as_posix()
            content = p.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            z.writestr(name, content)
        z.writestr("SHA256SUMS.txt", "".join(f"{h}  {name}\n" for name,h in hashes.items()))
    with ZipFile(tmp) as z:
        assert z.testzip() is None and set(z.namelist()) == set(hashes) | {"SHA256SUMS.txt"}
        for name, digest in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    tmp.replace(dest)
    print(len(hashes), "verified members", dest.stat().st_size, hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
