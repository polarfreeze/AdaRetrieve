"""Create the Drive code archive and a notebook that runs the same experiment."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts"/"colab"
OUTPUT.mkdir(parents=True, exist_ok=True)
source=["retrieval.py","neural.py","policy.py","study.py","matched.py","build_cost.py","summarize.py",
        "path_router.py","router_test.py","router_summary.py"]
dest=OUTPUT/"AdaRetrieve-code.zip"
with ZipFile(dest,"w",compression=ZIP_DEFLATED) as z:
    for f in source:z.write(ROOT/f,f)
    z.write(ROOT/"scripts/prepare_data.py","scripts/prepare_data.py")
sha=hashlib.sha256(dest.read_bytes()).hexdigest()
cells=[]


def cell(kind,text):
    item={"cell_type":kind,"metadata":{},"source":text.strip().splitlines(keepends=True)}
    if kind=="code":item.update(execution_count=None,outputs=[])
    cells.append(item)


cell("markdown","""# AdaRetrieve
Three-domain retrieval and reranking on a Colab T4. Run the cells in order. The default uses the three verified BEIR archives in My Drive/AdaRetrieve and creates a fresh results folder for this run. Set RUN_ID to an earlier run to resume its checkpoints. Full test evaluation starts after all three source-domain policies have been frozen.""")
cell("code",f"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, shutil, subprocess, sys, zipfile
import torch
from google.colab import drive
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print(torch.cuda.get_device_name(0), torch.__version__)
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/AdaRetrieve')
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
saved = DRIVE / 'runs' / RUN_ID
work = Path('/content') / ('AdaRetrieve-' + RUN_ID)
work.mkdir(parents=True, exist_ok=True)
saved.mkdir(parents=True, exist_ok=True)
bundle = DRIVE / 'AdaRetrieve-code.zip'
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == '{sha}'
with zipfile.ZipFile(bundle) as z: z.extractall(work)
data = work / 'data/verified'; data.mkdir(parents=True, exist_ok=True)
for name in ('dataset_manifest.json','official_split_ids.json'):
    shutil.copy2(DRIVE/name, data/name)
for m in json.loads((data/'dataset_manifest.json').read_text()).values():
    p = data/m['file']; shutil.copy2(DRIVE/p.name,p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
for name in ('cache','results'):
    folder=work/name; folder.mkdir(exist_ok=True)
    if (saved/name).exists(): shutil.copytree(saved/name,folder,dirs_exist_ok=True)
if importlib.util.find_spec('faiss') is None:
    subprocess.run([sys.executable,'-m','pip','install','-q','faiss-cpu'],check=True)
print('Data verified; run ID:',RUN_ID)
""")
cell("markdown","""## Build indexes and score candidates
BM25 and exact FAISS inner-product search share fixed document ordinals. A frozen MiniLM encoder retrieves candidates and a frozen TinyBERT cross-encoder scores the union needed by the finite action grid. Checkpoints copy to Drive during the run. Timed serving reads its indexes from runtime storage.""")
cell("code","""
import time
def sync():
    for name in ('cache','results'):
        dest=saved/name; dest.mkdir(exist_ok=True)
        for src in (work/name).iterdir():
            if not src.is_file() or src.suffix in ('.tmp','.part') or src.name.endswith('.oracle.jsonl.gz') and not (src.parent/(src.name.split('.')[0]+'.oracle_cost.json')).exists():continue
            dst=dest/src.name
            if dst.exists() and dst.stat().st_size==src.stat().st_size and int(dst.stat().st_mtime)==int(src.stat().st_mtime):continue
            tmp=dst.with_name(dst.name+'.part')
            shutil.copy2(src,tmp); tmp.replace(dst)
def run(script, periodic_sync=True):
    process=subprocess.Popen([sys.executable,'-u',script],cwd=work)
    while True:
        try: process.wait(timeout=60); break
        except subprocess.TimeoutExpired:
            if periodic_sync: sync()
    sync()
    if process.returncode: raise RuntimeError(f'{script} exited with {process.returncode}')
run('neural.py')
print((work/'results/cache_build.json').read_text())
""")
cell("markdown","""## Train and evaluate
The settings match the reported run. To preserve latency conditions, the notebook copies the evaluation artifacts to Drive after the timed job finishes. Each completed test query is checkpointed on the runtime disk.""")
cell("code","""
run('study.py',periodic_sync=False)
run('matched.py',periodic_sync=False)
run('build_cost.py',periodic_sync=False)
run('summarize.py',periodic_sync=False)
print('Metrics, paired intervals, oracle costs and figures:',saved/'results')
""")
cell("markdown","""## Source-validated one-shot routing
Fit the 30 terminal-path gain models from source training rows. Validate that the scores distinguish routes and that choices vary within every budget. Inspect the comparison with the strongest constant path on source validation. All three gates must pass before the supplemental T4 test, which measures the frozen sequential model, one-shot router and RRF in randomized order within each query.""")
cell("code","""
run('path_router.py',periodic_sync=False)
print((work/'results/router_gate.json').read_text())
run('router_test.py',periodic_sync=False)
run('router_summary.py',periodic_sync=False)
""")
nb={"nbformat":4,"nbformat_minor":0,"metadata":{"accelerator":"GPU","colab":{"gpuType":"T4"},
    "kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":cells}
(OUTPUT/"AdaRetrieve.ipynb").write_text(json.dumps(nb,indent=1),encoding="utf-8")
print(f"Colab notebook and code archive written to {OUTPUT}")
