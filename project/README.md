# AdaRetrieve

This repository contains the Rutgers Massive Data Mining course project on sequential retrieval and reranking under a latency budget. The primary evaluation holds out SciFact, NFCorpus and FiQA in turn.

## Read the results

- `AdaRetrieve-report.pdf` is the final v6 English report.
- The broad benchmark, source-validated one-shot comparison, and BM25-reuse experiment establish the baselines. V6 adds a matched-prefix comparison with fixed RRF and two rerankers, plus source-quantile admission, candidate-depth, and input-length diagnostics. The routing experiments use three leave-one-domain-out folds.
- The NeurIPS 2024 LaTeX manuscript, style, tables and figures are in `paper/`. Google AI Studio's Gemini 4 Neon with Extra High thinking drafted the revised prose. The tables use the archived measurements.
- [Original one-revision experiment](../routing-followup/) contains the frozen protocol, models, source decision, observations and paired estimates. [Matched-prefix experiment](../controlled-t4/) and [admission and depth diagnostics](../budget-depth-t4/) contain the v6 notebooks, audit scripts, observations and per-domain results.
- `results/main_metrics.csv` lists every domain, budget and method. `results/domain_macro.csv` gives each test domain equal weight.
- `results/paired_differences.csv` and `results/domain_macro_paired.csv` report paired differences in both NDCG@10 and measured utility, computed as sequential minus comparator. Utility first charges each query's actual time and excess over its budget, then averages by domain. The separate oracle audit counts profiled states over budget.
- `results/router_gate.json` summarizes the source validation checks at each budget, source domains, frozen model hashes and the validation-best constant path. Each fold's `*.router_validation.json` contains the individual validation choices. The ridge specification, profile, path list and parameters are in `*.router.json`.
- `results/*.router_live.jsonl` and `results/router_t4_environment.json` record the separate three-fold T4 comparison. Seven live calls per query cover the two policies at three budgets and a fixed RRF call. Each one-shot row includes the planned and executed paths. The supplemental `router_live_metrics.csv`, `router_live_paired.csv` and `router_live_paths.csv` retain all three budgets.
- `results/oracle_cost.csv`, `results/cache_build.json` and `results/cold_build.json` separate the size and construction time of the oracle tables from the shared cost of indexing and scoring. In the per-fold oracle cost files, `action_values_per_query` divides the number of supervised action values by the number of distinct source training queries. A state row may contain several action values.
- `results/*.test.jsonl` contains one record per official test query with method, budget, selected actions, NDCG@10, Recall@10, MRR@10, actual latency and the number of scored pairs. `results/*.matched.json` contains the full-grid one-shot control. Fixed runs are measured once per query and evaluated against all three budgets.
- `results/*.oracle.jsonl.gz` contains the training supervision. Each line is a JSON array of source domain, query ID, budget (ms), path prefix, numeric state features, and a dictionary mapping each feasible action to its oracle value.
- `results/audit.json` records the independent split, action-value, budget and test coverage checks.

## Reproduce on Colab

From this directory, run `python scripts/prepare_data.py --locked` and `python scripts/package_colab.py`. Copy the three verified dataset ZIPs and two JSON manifests from `data/verified/` and the generated code ZIP from `artifacts/colab/` into `My Drive/AdaRetrieve`. Open the matching generated `artifacts/colab/AdaRetrieve.ipynb` in Colab, select a T4 GPU, and run its cells in order. The notebook checks hashes before unpacking. Each fresh run gets its own directory in `AdaRetrieve/runs/`. Set the run ID to resume an existing run. The measured environment is recorded in `results/environment.json`.

The scripts run in this order:

1. `neural.py` builds the BM25 and dense indexes and caches scores for the finite candidate grid.
2. `study.py` profiles source-domain actions, enumerates labeled oracle states, fits the policies, freezes all folds, and executes the main live query tests.
3. `matched.py` fits a one-shot router that may select any feasible terminal path represented in its fold's oracle table. It uses the same source query sample and the same gradient-boosting settings.
4. `build_cost.py` repeats preprocessing and table construction from empty experiment caches in a temporary directory. It checks the independently rebuilt counts. Model weights, verified archives and the precomputed serving cost profiles are available at the start.
5. `summarize.py` computes domain and macro results, paired bootstrap intervals and figures. The local `scripts/audit_results.py` verifies the serialized outputs. The report uses the audited tables and measured run environment.
6. `path_router.py` fits one penalized regression per terminal path to the paired quality gain over Stop, using only source training rows and root-state features. It freezes all models and checks discrimination on each fold's source validation queries. The validation-best constant comparator uses a single feasible path per fold and budget in hindsight. The offline validation uses profiled costs.
7. `router_test.py` checks the frozen model hashes and source gates, then runs both policies and RRF on a T4 in randomized within-query order. `router_summary.py` checks official IDs, seven distinct jobs per query, intended/executed path consistency and the measured outputs before regenerating the follow-up tables and frontier. The archived report uses the original per-query observations; wall-clock reruns can select different actions near budget boundaries.

Follow the [CPU reproduction commands](../README.md#reproduce-the-reported-results-on-cpu) to validate the archived measurements and rebuild the report with Tectonic. For Overleaf, upload the contents of `paper/` and compile `main.tex` with pdfLaTeX. The course report uses the NeurIPS 2024 preprint option. After restoring the verified datasets, run `python scripts/package_project.py` to export the report, code and evidence as a local ZIP.

The pinned manifest and official split IDs are in `data/verified/`. Restore the three verified dataset ZIPs with the command above. The separate Drive file `AdaRetrieve-indexes-and-scores.zip` contains reusable indexes and query scores.

## Data provenance

The TU Darmstadt ZIP endpoint was unreachable during preparation. The uploaded datasets use the BEIR team's versioned Hugging Face JSONL and qrels files. Each source file was checked against its upstream LFS SHA-256 or Git blob ID, then packed into a ZIP. The newly packaged archives have their own MD5 and SHA-256 values. The manifest contains the precise revision URLs, sizes and hashes. The three uploaded ZIPs were individually downloaded through the signed-in Drive interface and verified again before experimental work began.

- [BEIR benchmark and data](https://github.com/beir-cellar/beir)
- [SciFact](https://huggingface.co/datasets/BeIR/scifact), [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus), [FiQA](https://huggingface.co/datasets/BeIR/fiqa)
- [Dense model card](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) and [reranker model card](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)
