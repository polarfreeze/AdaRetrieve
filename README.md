# AdaRetrieve

AdaRetrieve studies query-dependent retrieval and reranking under latency budgets. The course project compares fixed baselines, sequential routing, one-shot routing, and a router that can revise its path once after the first dense retrieval. It evaluates transfer across SciFact, NFCorpus, and FiQA with three leave-one-domain-out folds.

- [Final English report, v6](project/AdaRetrieve-report.pdf) and [NeurIPS 2024 source](project/paper/main.tex)
- [Experiment guide](project/README.md) and [archived benchmark results](project/results/)
- [Original one-revision protocol](routing-followup/protocol.md) and [paired held-out results](routing-followup/output/paired_differences.csv)
- [Matched-prefix comparison, same-session RRF, and reranker results](controlled-t4/results.md)
- [Latency admission, candidate depth, and input truncation results](budget-depth-t4/results.md)

## Reproduce the reported results on CPU

Run from the repository root with Python 3. The audits use the archived rankings and wall-clock measurements. The two v6 follow-ups also check the shared routing prefixes or source-only timing profiles and regenerate their CSVs from the recorded observations.

```sh
python -m pip install numpy scipy requests
python project/scripts/check_report_numbers.py
python project/scripts/prepare_data.py --locked
python project/scripts/audit_results.py
python routing-followup/summarize_followup.py
python controlled-t4/summarize.py
python budget-depth-t4/analyze.py
```

The downloader uses pinned upstream revisions and SHA-256 checksums. You can also copy the three verified dataset ZIPs from the course submission into `project/data/verified/`. The two v6 observation archives contain the per-query ranks, measured times, and experiment metadata required by the audits. [Dataset manifest](project/data/verified/dataset_manifest.json)

Install Tectonic and run `python project/scripts/render_report.py` to regenerate the tables and compile the v6 PDF. Recompilation replaces the working copy of the PDF and can change its metadata. The intermediate LaTeX build files go to a temporary directory.

## Repeat the GPU experiments

For the original experiments, run `python project/scripts/package_colab.py` to generate the Colab notebook and code archive under `project/artifacts/colab/`. Follow the instructions for the Drive file layout and T4 settings. [Experiment guide](project/README.md)

For the original one-revision experiment, run `python project/scripts/package_project.py` after restoring the datasets. Upload the generated `project/artifacts/AdaRetrieve-project.zip` and the course copy of `AdaRetrieve-indexes-and-scores.zip` to the Colab runtime. [One-revision notebook](routing-followup/routing-followup.ipynb) · [Course Drive folder](https://drive.google.com/drive/folders/1yv7n4ftNCgXTHS4ne7ZHIVFrizg4dPSn)

The two v6 diagnostic notebooks use the original, checksum-pinned v5 course project ZIP and index archive. Upload that v5 copy as `AdaRetrieve-project.zip` and the original indexes as `AdaRetrieve-indexes-and-scores.zip`. The notebook setup cells record the required hashes. Fresh latency measurements can vary by runtime. [Matched-prefix notebook](controlled-t4/controlled-t4.ipynb) · [Admission and depth notebook](budget-depth-t4/sensitivity-t4.ipynb)
