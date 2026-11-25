"""
Generate the three missing IEEE figures + Table I from MEASURED results.

Reads  eval/results/eval_results.csv   (written by eval_run.py)
Writes eval/results/fig4_metrics.(png|pdf)
       eval/results/fig5_confusion.(png|pdf)
       eval/results/fig6_latency.(png|pdf)
       eval/results/table1_metrics.csv  and  table1_metrics.tex

Needs only numpy + matplotlib (no torch, no server). Re-run freely after tweaking thresholds:
    python eval/make_plots.py
    python eval/make_plots.py --cv-thr 0.5 --nlp-thr 0.4 --fusion-thr 0.35 --csv path/to/other.csv

Positive class = 1 = fake / manipulated / misleading.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NAN = float("nan")
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # colour-blind safe
plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not r.get("error") and r.get("label") in ("0", "1")]


def auc(y, s):
    """ROC-AUC via the rank-sum (Mann-Whitney) formula with tie handling. NaN if one class only."""
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return NAN
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(rows, col, thr):
    """Metrics for one model on the samples where that signal exists. Undefined -> NaN (not 0)."""
    y, s = [], []
    for r in rows:
        v = fnum(r.get(col))
        if v is not None:
            y.append(int(r["label"]))
            s.append(v)
    y, s = np.array(y), np.array(s)
    if len(y) == 0:
        return None
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else NAN  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec) if not (math.isnan(prec) or math.isnan(rec)) else NAN
    return dict(n=len(y), thr=thr, tp=tp, tn=tn, fp=fp, fn=fn, acc=div(tp + tn, len(y)),
                prec=prec, rec=rec, f1=f1, auc=auc(y, s))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / "results" / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", HERE / "results" / f"{name}.png")


def fig_metrics(res):
    names = list(res)
    keys = [("acc", "Accuracy"), ("prec", "Precision"), ("rec", "Recall"), ("f1", "F1-score")]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    w = 0.8 / len(keys)
    for k, (key, label) in enumerate(keys):
        vals = [res[m][key] for m in names]
        xs = np.arange(len(names)) + (k - 1.5) * w
        bars = ax.bar(xs, [0 if math.isnan(v) else v for v in vals], w, label=label, color=COLORS[k],
                      edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, (0 if math.isnan(v) else v) + 0.015,
                    "n/a" if math.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{m}\n(n={res[m]['n']})" for m in names])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, columnspacing=0.8,
              handlelength=1)
    save(fig, "fig4_metrics")


def fig_confusion(res):
    names = list(res)
    fig, axes = plt.subplots(1, len(names), figsize=(3.5, 1.45))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, names):
        r = res[m]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"]); ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Predicted"); ax.set_title(f"{m}\n(thr={r['thr']:.2f})", fontsize=6.5)
        for s in ax.spines.values():
            s.set_visible(True)
    axes[0].set_ylabel("Actual")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig5_confusion")


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else NAN


def fig_latency(rows):
    cold = [fnum(r["cold_ms"]) for r in rows if fnum(r.get("cold_ms")) is not None]
    warm = [fnum(r["warm_ms"]) for r in rows if fnum(r.get("warm_ms")) is not None]
    groups = [("Cold cache\n(full pipeline)", cold, COLORS[0]), ("Warm cache\n(cache hit)", warm, COLORS[2])]
    groups = [g for g in groups if g[1]]
    if not groups:
        print("No latency data - skipping fig6")
        return None
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    stats = {}
    for i, (label, vals, c) in enumerate(groups):
        p50, p95 = pct(vals, 50), pct(vals, 95)
        stats[label.split("\n")[0]] = dict(n=len(vals), mean=float(np.mean(vals)), p50=p50, p95=p95,
                                           max=float(np.max(vals)))
        ax.bar(i, p50, 0.5, color=c, edgecolor="black", linewidth=0.3, label="Median" if i == 0 else None)
        ax.plot([i - 0.25, i + 0.25], [p95, p95], color="black", lw=1.2, label="95th pct" if i == 0 else None)
        ax.vlines(i, p50, p95, color="black", lw=0.8)
        ax.text(i, p95 * 1.15, f"p50={p50:.1f} ms\np95={p95:.1f} ms\nn={len(vals)}", ha="center",
                va="bottom", fontsize=5.5)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Response time (ms, log scale)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig6_latency")
    return stats


def write_table(res, lat):
    f = lambda v: "--" if math.isnan(v) else f"{v:.3f}"  # noqa: E731
    rows = [[m, r["n"], f(r["acc"]), f(r["prec"]), f(r["rec"]), f(r["f1"]), f(r["auc"]), r["thr"],
             r["tp"], r["fp"], r["tn"], r["fn"]] for m, r in res.items()]
    with open(HERE / "results" / "table1_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Model", "n", "Acc", "Prec", "Rec", "F1", "AUC", "threshold", "TP", "FP", "TN", "FN"])
        w.writerows(rows)
        if lat:
            w.writerow([])
            w.writerow(["Latency", "n", "mean_ms", "p50_ms", "p95_ms", "max_ms"])
            for k, v in lat.items():
                w.writerow([k, v["n"], f"{v['mean']:.1f}", f"{v['p50']:.1f}", f"{v['p95']:.1f}", f"{v['max']:.1f}"])
    tex = ["\\begin{tabular}{lccccc}", "\\hline", "Model / Task & Acc. & Prec. & Rec. & F1 & AUC \\\\", "\\hline"]
    for r in rows:
        tex.append(f"{r[0]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (HERE / "results" / "table1_metrics.tex").write_text("\n".join(tex), encoding="utf-8")
    print("saved Table I -> results/table1_metrics.csv / .tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "results" / "eval_results.csv"))
    ap.add_argument("--cv-thr", type=float, default=0.50, help="fake-probability cut-off for the CV checkpoint")
    ap.add_argument("--nlp-thr", type=float, default=0.50, help="text-risk cut-off for the text detector")
    ap.add_argument("--fusion-thr", type=float, default=0.60,
                    help="fused-risk cut-off; 0.60 = 'Likely manipulated' (red), 0.35 = also count 'Suspicious'")
    a = ap.parse_args()

    rows = load(a.csv)
    if not rows:
        raise SystemExit("No valid rows in " + a.csv)
    print(f"{len(rows)} valid samples ({sum(int(r['label']) for r in rows)} fake / "
          f"{sum(1 - int(r['label']) for r in rows)} real)")
    spec = {"CV checkpoint": ("cv", a.cv_thr), "Text detector": ("nlp", a.nlp_thr),
            "Full fusion system": ("risk", a.fusion_thr)}
    res = {}
    for name, (col, thr) in spec.items():
        r = evaluate(rows, col, thr)
        if r is None:
            print(f"  {name}: no samples with this signal - skipped")
        else:
            res[name] = r
            print(f"  {name}: n={r['n']} acc={r['acc']:.3f} f1={r['f1']:.3f} auc={r['auc']:.3f}")
    if not res:
        raise SystemExit("Nothing to plot")
    fig_metrics(res)
    fig_confusion(res)
    lat = fig_latency(rows)
    write_table(res, lat)


if __name__ == "__main__":
    main()
