# Reproducing the one-revision experiment

The accompanying project directory contains the original two experiments and the English report. Keep the two directories side by side after extracting the course project ZIP. The measurements and models used in the report are already in output/.

## Analyze the archived observations on CPU

Run `python summarize_followup.py` from this directory with NumPy and SciPy installed. The script verifies official query IDs, the frozen root policies, costs, thresholds and the source decision, and reconstructs each query's utility from its measured latency. It writes budget_metrics.csv and paired_differences.csv in output/. The two CSVs include source confirmation and target evaluation. Macro results give the three collections equal weight. The reported macro P95 averages per-domain P95 latencies. Paired bootstrap intervals use 4,000 within-domain resamples. The 144 distinct source confirmation queries each occur in two folds and are kept together when resampling. The target evaluation covers 1,271 queries.

## Run a new T4 experiment

Follow the [repository instructions](../README.md#repeat-the-gpu-experiments) to build the project ZIP and locate the original index archive. Open routing-followup.ipynb in Colab, select a Tesla T4, and upload both ZIPs to the runtime's Files panel. Run both code cells. The notebook verifies the project manifest, checks the index archive and the frozen measurement code, and extracts the required inputs to a new temporary directory. It uses a fresh output directory for the rerun. The experimental implementation and its recorded environment are in experiment.py and output/environment.json. Target evaluation is conditional on the new live source screen. Timing or guard decisions may differ in a new runtime.

The method labels seq and one refer to the sequential and root-only policies. seq_reuse and one_reuse reuse BM25 and update the two lexical cost estimates. revision starts from the reused root router and permits one suffix change after dense retrieval. Each query/method/budget was measured once after warmup. See protocol.md for the source decision thresholds and results.md for the original observations.
