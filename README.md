# AdaRetrieve

AdaRetrieve studies query-dependent retrieval and reranking under latency budgets. The course project compares fixed baselines, sequential routing, one-shot routing, and a router that can revise its path once after the first dense retrieval. It evaluates transfer across SciFact, NFCorpus, and FiQA with three leave-one-domain-out folds.

- [Final English report, v5](project/AdaRetrieve-report.pdf) and [NeurIPS 2024 source](project/paper/main.tex)
- [Experiment guide](project/README.md) and [archived benchmark results](project/results/)
- [One-revision protocol](routing-followup/protocol.md) and [paired held-out estimates](routing-followup/output/paired_differences.csv)

## Check the reported results on CPU

Run from the repository root with Python 3. The first audit recomputes serving metrics directly from the archived observations. The remaining two checks also use the verified dataset archives to validate query IDs and split boundaries.

```sh
python -m pip install numpy scipy requests
python project/scripts/check_report_numbers.py
python project/scripts/prepare_data.py --locked
python project/scripts/audit_results.py
python routing-followup/summarize_followup.py
```

The downloader uses the pinned upstream revisions and SHA-256 checksums in the [dataset manifest](project/data/verified/dataset_manifest.json). You can also copy the three verified ZIPs from the course submission into `project/data/verified/`. To compile the report from the archived results, install Tectonic and run `python project/scripts/render_report.py`.

## Repeat the GPU experiments

For the original experiments, run `python project/scripts/package_colab.py` to generate a matching Colab notebook and code ZIP under `project/artifacts/colab/`. Follow the [experiment guide](project/README.md) for the Drive file layout and T4 settings.

For the follow-up, run `python project/scripts/package_project.py` after restoring the datasets. The [follow-up notebook](routing-followup/routing-followup.ipynb) takes the generated project ZIP and the original `AdaRetrieve-indexes-and-scores.zip` from the [course Drive folder](https://drive.google.com/drive/folders/1yv7n4ftNCgXTHS4ne7ZHIVFrizg4dPSn). The notebook checks both inputs before running. Fresh latency measurements can vary by runtime.
