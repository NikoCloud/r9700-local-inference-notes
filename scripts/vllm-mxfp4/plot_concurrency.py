#!/usr/bin/env python3
"""Render the README hero figure: aggregate decode vs concurrency at 330 W.

What it plots
-------------
`rows[].aggregate` (decode tokens/sec summed across all streams) against
`rows[].n` (simultaneous streams) from the two matched 330 W concurrency sweeps
documented in docs/13 section 6b -- same harness method (conc_test.py: ~70-token
prompt, max_tokens=800, so total_tokens/wall measures decode), same hour, same
card, both under the R9700's 330 W cap:

  data/vllm-mxfp4/vllm_conc2_nospec_ms84_330w.json    MXFP4 vLLM, no speculation
  data/vllm-mxfp4/vllm_conc2_prod_llamacpp_330w.json  llama.cpp production (-np 4)

Only measured rungs are plotted; nothing is interpolated. The llama.cpp sweep
stops at n=16 (it saturates at n=2) while the vLLM sweep continues to n=64, and
the figure labels that asymmetry.

Output: data/vllm-mxfp4/concurrency_330w.svg
Usage:  <python> scripts/vllm-mxfp4/plot_concurrency.py
        (paths resolve relative to the repo root, so cwd does not matter)
Needs matplotlib only; renders headless via the Agg backend.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "vllm-mxfp4"
VLLM_JSON = DATA_DIR / "vllm_conc2_nospec_ms84_330w.json"
LLAMA_JSON = DATA_DIR / "vllm_conc2_prod_llamacpp_330w.json"
OUT_SVG = DATA_DIR / "concurrency_330w.svg"

VLLM_COLOR = "#0072B2"   # Okabe-Ito blue
LLAMA_COLOR = "#D55E00"  # Okabe-Ito vermillion

TEXT_COLOR = "#333333"
NOTE_COLOR = "#777777"
GRID_COLOR = "#dddddd"
BG_COLOR = "#fafafa"


def load_series(path):
    """Return [(n, aggregate_tok_s), ...] from rows[], sorted by rung.

    Values are copied verbatim from the JSON; no interpolation, no smoothing.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    series = [(int(row["n"]), float(row["aggregate"])) for row in doc["rows"]]
    series.sort(key=lambda pair: pair[0])
    return series


def annotate_peak(ax, series, color):
    """Label the measured peak of one series with its exact JSON value."""
    n, value = max(series, key=lambda pair: pair[1])
    ax.annotate(
        f"{value:g} tok/s @ n={n}",
        xy=(n, value),
        xytext=(0, 15),
        textcoords="offset points",
        ha="center",
        fontsize=10.5,
        color=color,
    )
    return n, value


def build_figure(vllm_series, llama_series):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": "#cccccc",
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "savefig.facecolor": BG_COLOR,
            "svg.fonttype": "path",
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.1), dpi=100)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)

    for series, color, label in (
        (vllm_series, VLLM_COLOR, "vLLM (no spec)"),
        (llama_series, LLAMA_COLOR, "llama.cpp production"),
    ):
        ax.plot(
            [n for n, _ in series],
            [v for _, v in series],
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=4.5,
            label=label,
            zorder=3,
        )

    # Concurrency rungs double, so log2 keeps them evenly spaced; ticks are only
    # the rungs that were actually measured, so nothing is implied between them.
    ax.set_xscale("log", base=2)
    rungs = sorted({n for n, _ in vllm_series} | {n for n, _ in llama_series})
    ax.xaxis.set_major_locator(FixedLocator(rungs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlim(0.87, 72)
    ax.set_ylim(0, 445)
    ax.set_yticks([0, 100, 200, 300, 400])

    ax.set_xlabel("concurrency N (log\u2082 scale)", fontsize=11)
    ax.set_ylabel("aggregate decode throughput (tokens/sec)", fontsize=11)
    ax.set_title(
        "Aggregate decode vs concurrency \u2014 one R9700, matched runs at 330 W",
        fontsize=13,
        pad=12,
    )
    ax.tick_params(labelsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")

    annotate_peak(ax, vllm_series, VLLM_COLOR)
    annotate_peak(ax, llama_series, LLAMA_COLOR)

    # Label where the shorter sweep ends so the line's stop is not misread.
    n_last, v_last = llama_series[-1]
    ax.annotate(
        f"swept to n={n_last}",
        xy=(n_last, v_last),
        xytext=(9, -3),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        color=NOTE_COLOR,
    )
    ax.text(
        0.0,
        -0.17,
        "source: data/vllm-mxfp4/*_330w.json \u2014 matched 330 W runs, same hour",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#888888",
    )

    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    return fig


def main():
    vllm_series = load_series(VLLM_JSON)
    llama_series = load_series(LLAMA_JSON)

    for name, series in (
        ("vLLM (no spec)", vllm_series),
        ("llama.cpp production", llama_series),
    ):
        n, value = max(series, key=lambda pair: pair[1])
        print(f"{name}:")
        print(f"  rungs      {[n for n, _ in series]}")
        print(f"  aggregate  {[v for _, v in series]}")
        print(f"  peak       {value:g} tok/s @ n={n}")

    fig = build_figure(vllm_series, llama_series)
    fig.savefig(OUT_SVG, format="svg", bbox_inches="tight")
    print(f"wrote {OUT_SVG.relative_to(REPO_ROOT)} ({OUT_SVG.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
