#!/usr/bin/env python3
"""Render results/*.json into the SVG figures used by the series.

    uv run python figures.py --out /path/to/portfolio/src/assets/figures/<slug>
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
mpl.style.use(HERE / "fieldguide.mplstyle")

SIGNAL, VIOLET, DIM, SIGNAL2, TEAL = "#d2823e", "#9b8ee2", "#969ba6", "#b56a2c", "#5ec6cb"
SILVER, MID = "#d6dbe1", "#7d8593"

TOOLS = [  # (row key, label, color, marker)
    ("qcc-O1", "qcc -O1", SIGNAL, "o"),
    ("qiskit-O1", "qiskit O1", MID, "s"),
    ("qiskit-O3", "qiskit O2/O3", VIOLET, "D"),
    ("tket-full", "tket FullPeephole", TEAL, "^"),
]


def load(name):
    return json.loads((HERE / "results" / f"{name}.json").read_text())


def save(fig, out, name):
    path = out / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _instances(rows):
    seen = []
    for r in rows:
        key = (r["suite"], r["instance"])
        if key not in seen:
            seen.append(key)
    return seen


def _cell(rows, suite, instance, tool):
    for r in rows:
        if (r["suite"], r["instance"], r["tool"]) == (suite, instance, tool):
            return r
    raise KeyError((suite, instance, tool))


def _dotplot(rows, out, name, field, title, xlabel):
    """Instances on y, `field` relative to input on x, one series per tool."""
    instances = _instances(rows)
    ys = range(len(instances))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for tool, label, color, marker in TOOLS:
        xs, yy = [], []
        for y, (s, i) in zip(ys, instances):
            inp = _cell(rows, s, i, "input")[field]
            if inp == 0:
                continue
            xs.append(100 * _cell(rows, s, i, tool)[field] / inp)
            yy.append(y)
        ax.plot(xs, yy, marker, color=color, label=label, ms=6.5, ls="none",
                zorder=3, alpha=0.9)
    ax.axvline(100, color=DIM, ls="--", lw=1)
    ax.set_yticks(list(ys), [f"{s}/{i}" for s, i in instances])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel + " — dashed line marks the unoptimized input")
    ax.set_title(title)
    ax.legend(loc="upper left")
    save(fig, out, name)


def fig_gate_reduction(rows, out):
    _dotplot(
        rows, out, "gate-reduction", "gates",
        "What survives each optimizer — total gates, lower is better",
        "gates remaining (% of input)",
    )


def fig_two_qubit(rows, out):
    _dotplot(
        rows, out, "two-qubit-counts", "gates_2q",
        "2q gates: peephole passes keep them, tket resynthesizes them away",
        "2q gates remaining (% of input)",
    )


def fig_compile_times(rows, out):
    instances = _instances(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for row_i, (tool, label, color, marker) in enumerate(TOOLS):
        ts = [_cell(rows, s, i, tool)["t_ms"] for s, i in instances]
        ax.plot(ts, [row_i] * len(ts), marker, color=color, ms=6, ls="none",
                alpha=0.75, zorder=3)
        med = sorted(ts)[len(ts) // 2]
        ax.plot([med], [row_i], "|", color=SILVER, ms=16, mew=2.4, zorder=4)
    ax.set_xscale("log")
    ax.set_yticks(range(len(TOOLS)), [t[1] for t in TOOLS])
    ax.invert_yaxis()
    ax.set_xlabel("optimization wall time (ms, log) — one dot per benchmark, bar = median")
    ax.set_title("Compile time: the price of deeper optimization")
    save(fig, out, "compile-times")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures-preview")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load("compile_bench")["rows"]
    fig_gate_reduction(rows, args.out)
    fig_two_qubit(rows, args.out)
    fig_compile_times(rows, args.out)


if __name__ == "__main__":
    main()
