# VeriFeed evaluation toolkit (separate from the live feed analysis)

Nothing in `main.py`, `textutil.py` or the Chrome extension is modified. `eval_run.py` imports
the functions from `main.py` and runs a labelled dataset through the same pipeline; the live
server is never started, so Instagram analysis is unaffected.

## Folder layout (put the `eval/` folder next to `main.py`)

    <project>/
      main.py  textutil.py  ...            <- untouched
      eval/
        eval_run.py        step 1: run dataset through the pipeline -> CSV
        make_plots.py      step 2: CSV -> Fig 4, Fig 5, Fig 6, Table I
        requirements-eval.txt   (matplotlib, numpy)
        dataset/labels.csv + dataset/images/
        results/           CSV, PNG/PDF figures, table (kept so you can re-plot anytime)

## 1. Build the dataset  (eval/dataset/labels.csv)

    id,image_path,caption,label,category
    1,images/real_001.jpg,Sunrise at Marina Beach,0,real
    2,images/fake_001.jpg,SHOCKING!! share before it's deleted,1,ai_generated

- `label`: **1 = fake / manipulated / misleading, 0 = authentic**
- `image_path` is relative to `eval/dataset/` (leave empty for caption-only rows)
- Caption can be English, Tamil or mixed. Aim for 100+ samples, balanced, including Tamil ones.
- Use identity-disjoint images (as your paper's protocol says) and describe the source in the paper.
  The two rows in the provided file are placeholders - replace them.

## 2. Run the evaluation

    pip install -r eval/requirements-eval.txt
    python eval/eval_run.py --limit 5        # smoke test first
    python eval/eval_run.py                  # full run
    python eval/eval_run.py --offline        # skip fact-check/news (reproducible, faster)
    python eval/eval_run.py --resume         # continue if interrupted

Run it with the server **stopped** (or accept slower timing - both would share the CPU/GPU).
Latency measured per sample: **cold** = full pipeline, cache bypassed; **warm** = cache read
through the same `Cache` class the server uses (Redis if running, else in-memory).
One discarded warm-up run absorbs one-time model initialisation.

## 3. Make the figures

    python eval/make_plots.py
    python eval/make_plots.py --fusion-thr 0.35 --nlp-thr 0.4     # try other thresholds

Outputs in `eval/results/`: `fig4_metrics`, `fig5_confusion`, `fig6_latency` (.png 300 dpi + .pdf),
`table1_metrics.csv` and `table1_metrics.tex`. Figures are column-width (3.5 in) for IEEE.

## Things to state honestly in the paper

- Positive class = fake/misleading. Thresholds: CV 0.50 on fake-probability, text 0.50, fusion 0.60
  ("Likely manipulated"); change with flags and report what you used.
- Each model is scored only on samples where its signal exists (n is printed under each bar).
  Text-detector scores are the heuristic text risk (Eq. 1), not a factuality classifier.
- Fact-check and news signals depend on live internet results, which change over time; `--offline`
  removes that variability. Record the date of the run.
- The fusion eval mirrors `analyze()` step by step. If you later change `analyze()` in `main.py`,
  mirror the change in `analyze_local()` in `eval_run.py`.
