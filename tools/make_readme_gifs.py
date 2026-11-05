#!/usr/bin/env python3
"""Render the animated GIFs embedded in README.md.

Every curve, marker and number in the animations is computed at render time
through PyRate itself: the TOML examples, the partition-function and rate
modules, the C++ kernels and their Python fallbacks, the Polyrate legacy
parser and the ``fu15`` comparator. Nothing is hand-drawn data.

    python -m pip install -e ".[docs]"
    python tools/make_readme_gifs.py               # every figure -> docs/assets/
    python tools/make_readme_gifs.py cvt dos       # a subset
    python tools/make_readme_gifs.py --list
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DEFAULT_OUTPUT = ROOT / "docs" / "assets"

FIGSIZE = (8.0, 4.5)
DPI = 120
FRAME_MS = 90

# Reference data-viz palette (light chart surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_WASH = "#cde2fb"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"
ATOM_O = "#e34948"
ATOM_H = "#f0efec"
MONO = "DejaVu Sans Mono"


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.labelcolor": INK2,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": BASELINE,
            "ytick.color": BASELINE,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "legend.labelcolor": INK2,
        }
    )


def mix(color_a: str, color_b: str, weight: float) -> str:
    """Blend ``color_a`` toward ``color_b`` by ``weight`` in [0, 1]."""

    a = np.asarray(to_rgb(color_a))
    b = np.asarray(to_rgb(color_b))
    return to_hex((1.0 - weight) * a + weight * b)


def num(value: float, spec: str) -> str:
    """Format a number with a typographic minus sign."""

    return format(value, spec).replace("-", "−")


def sci(value: float, digits: int = 3) -> str:
    """Mathtext scientific notation such as ``1.93×10⁷``."""

    if value == 0.0 or not math.isfinite(value):
        return f"{value:g}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / 10.0**exponent
    if round(mantissa, digits - 1) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    if exponent == 0:
        return f"{mantissa:.{digits - 1}f}"
    return rf"${mantissa:.{digits - 1}f}\times10^{{{exponent}}}$"


class Movie:
    """Collect Agg frames and write a compact, palette-stable GIF."""

    def __init__(self, title: str, subtitle: str) -> None:
        self.fig = Figure(figsize=FIGSIZE, dpi=DPI)
        self.canvas = FigureCanvasAgg(self.fig)
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []
        self.fig.text(0.025, 0.955, title, fontsize=12.5, weight="bold", color=INK, va="top")
        self.fig.text(0.025, 0.893, subtitle, fontsize=8.6, color=INK2, va="top")

    def board(self) -> Any:
        """Full-figure axes in figure-fraction units for diagrams."""

        ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        return ax

    def grab(self, duration_ms: int = FRAME_MS) -> None:
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        self.frames.append(Image.fromarray(rgba[:, :, :3].copy()))
        self.durations.append(int(duration_ms))

    def hold(self, duration_ms: int) -> None:
        self.durations[-1] += int(duration_ms)

    def save(self, path: Path, colors: int = 160) -> int:
        """Write the GIF with one shared palette and transparent frame deltas.

        Pixels that did not change since the previous frame are written as a
        reserved transparent index, which keeps the LZW stream short.
        """

        if not self.frames:
            raise RuntimeError("no frames recorded")
        step = max(1, len(self.frames) // 16)
        samples = self.frames[::step]
        width, height = samples[0].size
        sheet = Image.new("RGB", (width, height * len(samples)), SURFACE)
        for index, frame in enumerate(samples):
            sheet.paste(frame, (0, index * height))
        palette = sheet.quantize(
            colors=min(colors, 255), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        table = palette.getpalette() or []
        table = (table + [0] * 768)[:768]
        transparent = 255
        table[3 * transparent : 3 * transparent + 3] = [255, 0, 255]

        encoded: list[Image.Image] = []
        previous: np.ndarray | None = None
        for frame in self.frames:
            indices = np.asarray(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            delta = indices.copy()
            if previous is not None:
                delta[indices == previous] = transparent
            previous = indices
            image = Image.fromarray(delta.astype(np.uint8), mode="P")
            image.putpalette(table)
            encoded.append(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded[0].save(
            path,
            save_all=True,
            append_images=encoded[1:],
            duration=self.durations,
            loop=0,
            optimize=False,
            disposal=1,
            transparency=transparent,
        )
        return path.stat().st_size


def rounded_box(
    ax: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    radius: float = 0.010,
    **style: Any,
) -> FancyBboxPatch:
    """A rounded rectangle on a :meth:`Movie.board` with circular corners."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        mutation_aspect=FIGSIZE[0] / FIGSIZE[1],
        **style,
    )
    ax.add_patch(patch)
    return patch


def ping_pong(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[-2:0:-1]])


def log_temperature_axis(ax: Any, ticks: Sequence[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{tick:g}" for tick in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def ring_marker(color: str, size: float = 8.0, **extra: Any) -> dict[str, Any]:
    style: dict[str, Any] = {
        "marker": "o",
        "linestyle": "none",
        "markersize": size,
        "markerfacecolor": color,
        "markeredgecolor": SURFACE,
        "markeredgewidth": 2.0,
    }
    style.update(extra)
    return style


# ---------------------------------------------------------------------------
# 1. Architecture / pipeline
# ---------------------------------------------------------------------------


def render_workflow() -> Movie:
    from pyrate import calculate_rates, load_toml

    job = load_toml(EXAMPLES / "harmonic_cvt.toml")
    result = calculate_rates(job.reaction, job.options, name=job.name)
    row = min(result.rows, key=lambda item: abs(item.temperature_k - 300.0))

    movie = Movie(
        "PyRate pipeline: readable Python stages, C++ hot loops",
        "Each stage is a tested module · the result is a live run of examples/harmonic_cvt.toml",
    )
    ax = movie.board()
    stages = [
        ("Input", ["TOML", "Polyrate .dat", "Python API"], "load_toml()\nload_input()"),
        ("Model", ["Species", "Reaction", "ReactionPath"], "typed, validated\ndataclasses"),
        (
            "RRHO",
            [r"$q_{\rm trans}\,q_{\rm rot}$", r"$q_{\rm vib}\,q_{\rm elec}$", "→ G°(T)"],
            "G° = E + ZPE\n− RT ln q",
        ),
        (
            "CTST",
            [r"$\Delta G^\ddagger(T)$", "Eyring", r"$\to k_{\rm TST}$"],
            "k = (k_BT/h)·\nexp(−ΔG‡/RT)",
        ),
        (
            "CVT",
            [r"$\Delta G^\ddagger(s,T)$", "max over s", r"$\to k_{\rm CVT}$"],
            "bottleneck\ns_CVT(T)",
        ),
        ("Tunneling", ["Wigner", "1D WKB", r"$\to \kappa(T)$"], "k_final = κ·k"),
        ("Output", ["table", "JSON · CSV", "fu15 compare"], "pyrate run\n--format json"),
    ]
    count = len(stages)
    box_w, gap = 0.112, 0.0245
    x0 = (1.0 - (count * box_w + (count - 1) * gap)) / 2.0
    y0, height = 0.50, 0.27
    lefts = [x0 + index * (box_w + gap) for index in range(count)]
    centers = [left + box_w / 2.0 for left in lefts]
    done_fill = mix(BLUE_WASH, SURFACE, 0.55)

    boxes = []
    bodies = []
    for left, center, (title, lines, caption) in zip(lefts, centers, stages, strict=True):
        boxes.append(rounded_box(ax, left, y0, box_w, height, fc=SURFACE, ec=BASELINE, lw=1.2))
        ax.text(
            center,
            y0 + height - 0.045,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            color=INK,
        )
        bodies.append(
            ax.text(
                center,
                y0 + height * 0.40,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=8.3,
                color=INK2,
                linespacing=1.6,
            )
        )
        ax.text(
            center,
            y0 - 0.03,
            caption,
            ha="center",
            va="top",
            fontsize=7.6,
            color=INK2,
            linespacing=1.35,
        )

    arrows = []
    for index in range(count - 1):
        arrow = FancyArrowPatch(
            (lefts[index] + box_w + 0.002, y0 + height / 2.0),
            (lefts[index + 1] - 0.002, y0 + height / 2.0),
            arrowstyle="-|>",
            mutation_scale=10,
            color=BASELINE,
            lw=1.3,
        )
        ax.add_patch(arrow)
        arrows.append(arrow)

    band_left = lefts[2]
    band_right = lefts[5] + box_w
    band_y, band_h = 0.135, 0.13
    kernel_box = rounded_box(
        ax, band_left, band_y, band_right - band_left, band_h, fc=SURFACE, ec=BASELINE, lw=1.2
    )
    band_mid = (band_left + band_right) / 2.0
    cpp_text = ax.text(
        band_mid - 0.03,
        band_y + band_h / 2.0,
        "C++17 pyrate._kernels\ndefault · GIL released",
        ha="right",
        va="center",
        fontsize=8.4,
        color=INK2,
        linespacing=1.5,
    )
    ax.text(
        band_mid, band_y + band_h / 2.0, "⇄", ha="center", va="center", fontsize=12, color=MUTED
    )
    py_text = ax.text(
        band_mid + 0.03,
        band_y + band_h / 2.0,
        "Python / NumPy fallback\nPYRATE_DISABLE_NATIVE=1",
        ha="left",
        va="center",
        fontsize=8.4,
        color=INK2,
        linespacing=1.5,
    )
    ax.text(
        0.5,
        0.075,
        "kernels: harmonic_log_partition · refine_maximum · wigner_kappa · "
        "wkb_exponent · thermal_wkb_kappa · beyer_swinehart",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
        family=MONO,
    )

    kernel_stage = {2: "harmonic_log_partition", 4: "refine_maximum", 5: "thermal_wkb_kappa"}
    connectors = {}
    for index in kernel_stage:
        (line,) = ax.plot(
            [centers[index], centers[index]],
            [y0 - 0.125, band_y + band_h],
            color=BASELINE,
            lw=1.1,
            solid_capstyle="butt",
        )
        connectors[index] = line

    note_left = lefts[0]
    note_w = lefts[1] + box_w - note_left
    rounded_box(ax, note_left, band_y, note_w, band_h, fc=SURFACE, ec=GRID, lw=1.0)
    ax.text(
        note_left + 0.012,
        band_y + band_h / 2.0,
        "Validated before any physics:\nunits, molecularity ≤ 2,\n≤ 1 imaginary frequency",
        ha="left",
        va="center",
        fontsize=7.8,
        color=INK2,
        linespacing=1.45,
    )

    status = ax.text(
        lefts[6] + box_w / 2.0,
        band_y + band_h / 2.0,
        "",
        ha="center",
        va="center",
        fontsize=7.8,
        color=INK2,
        linespacing=1.45,
    )

    def paint(active: int, *, backend: str = "cpp") -> None:
        for index, box in enumerate(boxes):
            if index < active:
                box.set_facecolor(done_fill)
                box.set_edgecolor(mix(BLUE, SURFACE, 0.35))
                box.set_linewidth(1.2)
            elif index == active:
                box.set_facecolor(BLUE_WASH)
                box.set_edgecolor(BLUE)
                box.set_linewidth(1.8)
            else:
                box.set_facecolor(SURFACE)
                box.set_edgecolor(BASELINE)
                box.set_linewidth(1.2)
        for index, arrow in enumerate(arrows):
            arrow.set_color(BLUE if index < active else BASELINE)
        calling = active in connectors
        for index, line in connectors.items():
            line.set_color(BLUE if index == active else BASELINE)
            line.set_linewidth(1.8 if index == active else 1.1)
        lit = calling or active >= count
        kernel_box.set_facecolor(BLUE_WASH if lit else SURFACE)
        kernel_box.set_edgecolor(BLUE if lit else BASELINE)
        kernel_box.set_linewidth(1.8 if lit else 1.2)
        for text, name in ((cpp_text, "cpp"), (py_text, "python")):
            selected = name == backend
            text.set_color(INK if selected else MUTED)
            text.set_weight("bold" if selected else "normal")

    paint(-1)
    movie.grab(900)
    for index in range(count):
        paint(index)
        if index == count - 1:
            bodies[-1].set_text(f"k_final (300 K)\n{sci(row.k_final)}\n" + r"$\rm s^{-1}$")
            bodies[-1].set_color(INK)
            status.set_text(
                f"k_TST {sci(row.k_tst)}\nk_CVT {sci(row.k_cvt or 0.0)}\n"
                f"κ = {row.tunneling_correction:.2f}"
            )
            movie.grab(2300)
        else:
            movie.grab(700)
    paint(count, backend="python")
    status.set_text("same numbers\nwithout the\nC++ extension")
    movie.grab(2300)
    return movie


# ---------------------------------------------------------------------------
# 2. Partition functions
# ---------------------------------------------------------------------------


def render_partition() -> Movie:
    from pyrate import load_toml
    from pyrate.constants import WAVENUMBER_TO_KJ_MOL
    from pyrate.partition import (
        partition_components,
        standard_gibbs_energy_kj_mol,
        zero_point_energy_kj_mol,
    )

    job = load_toml(EXAMPLES / "harmonic_cvt.toml")
    species = job.reaction.reactants[0]
    frequencies = np.asarray(species.positive_frequencies_cm1, dtype=float)
    quanta = frequencies * WAVENUMBER_TO_KJ_MOL
    zpe = zero_point_energy_kj_mol(species)

    dense = np.geomspace(150.0, 1500.0, 160)
    components = [partition_components(species, float(t)) for t in dense]
    series = {
        "trans": np.array([c.translational for c in components]),
        "rot": np.array([c.rotational for c in components]),
        "vib": np.array([c.vibrational for c in components]),
    }

    movie = Movie(
        "Statistical thermodynamics: RRHO partition functions → G°(T)",
        "Reactant of examples/harmonic_cvt.toml · ideal gas at 1 bar · rigid rotor · "
        "harmonic oscillator · G° = E + ZPE − RT ln q",
    )
    fig = movie.fig
    ax_l = fig.add_axes((0.075, 0.13, 0.37, 0.64))
    ax_r = fig.add_axes((0.545, 0.13, 0.355, 0.64))

    energy_max = 40.0
    slot = 0.42
    bars: list[list[Any]] = []
    for mode, quantum in enumerate(quanta):
        levels = int(energy_max // quantum)
        mode_bars = []
        for level in range(levels + 1):
            energy = level * quantum
            ax_l.plot(
                [mode - slot, mode + slot],
                [energy, energy],
                color=BASELINE,
                lw=1.0,
                solid_capstyle="butt",
            )
            mode_bars.append(ax_l.barh(energy, 0.0, height=1.1, left=mode - slot, color=BLUE)[0])
        bars.append(mode_bars)
    ax_l.set_xlim(-0.6, len(quanta) - 0.4)
    ax_l.set_ylim(-1.5, energy_max + 1.5)
    ax_l.set_xticks(range(len(quanta)))
    ax_l.set_xticklabels([f"{value:.0f}" for value in frequencies])
    ax_l.grid(axis="x", visible=False)
    ax_l.set_xlabel("harmonic mode ν (cm⁻¹) · bar length = level population")
    ax_l.set_ylabel("vibrational energy above ZPE (kJ mol⁻¹)")

    colors = {"trans": BLUE, "rot": ORANGE, "vib": AQUA}
    names = {
        "trans": r"$\ln q_{\rm trans}$",
        "rot": r"$\ln q_{\rm rot}$",
        "vib": r"$\ln q_{\rm vib}$",
    }
    dots = {}
    for key, values in series.items():
        ax_r.plot(dense, values, color=colors[key], label=names[key])
        ax_r.text(
            dense[-1] * 1.05,
            values[-1],
            names[key],
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK2,
        )
        (dots[key],) = ax_r.plot([], [], **ring_marker(colors[key]))
    log_temperature_axis(ax_r, (150, 300, 600, 1000, 1500))
    ax_r.set_xlim(dense[0], dense[-1])
    ax_r.set_ylim(-1.0, 22.5)
    ax_r.set_xlabel("T (K)")
    ax_r.set_ylabel("ln q (dimensionless)")
    ax_r.legend(loc="upper left", handlelength=1.4)
    cursor = ax_r.axvline(dense[0], color=INK2, lw=1.0)
    readout = ax_r.text(
        0.97,
        0.28,
        "",
        transform=ax_r.transAxes,
        ha="right",
        va="center",
        fontsize=8.3,
        color=INK,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE, "ec": GRID},
    )
    ax_r.set_title("Components of ln q", loc="left")

    temperatures = ping_pong(np.geomspace(150.0, 1500.0, 40))
    for temperature in temperatures:
        t = float(temperature)
        x = quanta * 1000.0 / (8.314462618 * t)
        for mode, mode_bars in enumerate(bars):
            for level, bar in enumerate(mode_bars):
                population = (1.0 - math.exp(-x[mode])) * math.exp(-level * x[mode])
                bar.set_width(2.0 * slot * population)
        comp = partition_components(species, t)
        ax_l.set_title(
            f"Level populations at T = {t:.0f} K   ·   q_vib = {math.exp(comp.vibrational):.2f}",
            loc="left",
        )
        for key, value in (
            ("trans", comp.translational),
            ("rot", comp.rotational),
            ("vib", comp.vibrational),
        ):
            dots[key].set_data([t], [value])
        cursor.set_xdata([t, t])
        gibbs = standard_gibbs_energy_kj_mol(species, t)
        thermal = gibbs - species.electronic_energy_kj_mol - zpe
        readout.set_text(
            f"ln q = {comp.total:.2f}\n"
            f"−RT ln q = {num(thermal, '.1f')} kJ mol⁻¹\n"
            f"G° = {num(gibbs, '.1f')} kJ mol⁻¹"
        )
        endpoint = abs(t - 150.0) < 1e-6 or abs(t - 1500.0) < 1e-6
        movie.grab(FRAME_MS + (700 if endpoint else 0))
    return movie


# ---------------------------------------------------------------------------
# 3. CTST vs canonical VTST
# ---------------------------------------------------------------------------


def _local_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-trace the local polynomial used by ``refine_maximum`` for drawing."""

    index = int(np.argmax(y))
    if x.size < 3 or index in {0, x.size - 1}:
        return x, y
    count = min(5, int(x.size))
    start = min(max(0, index - count // 2), int(x.size) - count)
    local_x = x[start : start + count]
    local_y = y[start : start + count]
    center = float(x[index])
    scale = max(abs(float(local_x[0]) - center), abs(float(local_x[-1]) - center))
    coefficients = np.polynomial.polynomial.polyfit((local_x - center) / scale, local_y, count - 1)
    grid = np.linspace(local_x[0], local_x[-1], 160)
    return grid, np.polynomial.polynomial.polyval((grid - center) / scale, coefficients)


def render_cvt() -> Movie:
    from pyrate import load_toml
    from pyrate.rates import activation_gibbs_kj_mol, conventional_tst_rate_si, cvt_rate_si

    job = load_toml(EXAMPLES / "harmonic_cvt.toml")
    reaction = job.reaction
    assert reaction.path is not None
    s_values = np.array([point.coordinate for point in reaction.path.points])

    def profile(temperature: float) -> np.ndarray:
        return np.array(
            [
                activation_gibbs_kj_mol(reaction, point.dividing_surface, temperature)
                for point in reaction.path.points
            ]
        )

    dense = np.geomspace(150.0, 1500.0, 140)
    bottleneck = []
    ratio = []
    for temperature in dense:
        cvt = cvt_rate_si(reaction, float(temperature))
        _, k_tst = conventional_tst_rate_si(reaction, float(temperature))
        bottleneck.append(cvt.coordinate)
        ratio.append(cvt.rate_si / k_tst)
    bottleneck_line = np.array(bottleneck)
    jumps = np.where(np.abs(np.diff(bottleneck_line)) > 0.15)[0]
    plot_t = dense.astype(float).tolist()
    plot_s = bottleneck_line.tolist()
    for jump in reversed(jumps):
        plot_t.insert(int(jump) + 1, float("nan"))
        plot_s.insert(int(jump) + 1, float("nan"))

    movie = Movie(
        "Rate theory: conventional TST vs canonical variational TST (CVT)",
        "examples/harmonic_cvt.toml · ΔG‡(s,T) at 5 path points + 5-point polynomial "
        "refinement · k_CVT = min over s of k_GT",
    )
    fig = movie.fig
    ax = fig.add_axes((0.075, 0.13, 0.43, 0.64))
    ax_s = fig.add_axes((0.62, 0.53, 0.33, 0.24))
    ax_k = fig.add_axes((0.62, 0.13, 0.33, 0.24), sharex=ax_s)

    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(23.0, 36.0)
    ax.set_xlabel("reaction coordinate s (Å)")
    ax.set_ylabel("ΔG‡(s, T) (kJ mol⁻¹)")
    ax.axvline(0.0, color=BASELINE, lw=1.0)
    (fit_line,) = ax.plot([], [], color=BLUE, label="generalized ΔG‡(s, T)")
    (samples,) = ax.plot([], [], **ring_marker(BLUE, 7.5))
    (tst_point,) = ax.plot(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=8.5,
        markerfacecolor=SURFACE,
        markeredgecolor=INK2,
        markeredgewidth=1.8,
    )
    (cvt_point,) = ax.plot([], [], **ring_marker(ORANGE, 9.5))
    ax.text(
        0.0,
        35.75,
        "TST: saddle point, s = 0",
        ha="center",
        va="top",
        fontsize=8.0,
        color=INK2,
        bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"},
    )
    ax.set_title("Generalized free-energy profile along the path", loc="left", fontsize=9.2)
    cvt_label = ax.text(0, 0, "", ha="center", va="bottom", fontsize=8.4, color=INK)
    readout = ax.text(
        0.5,
        0.04,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=INK,
        linespacing=1.55,
    )

    ax_s.plot(plot_t, plot_s, color=ORANGE)
    ax_s.axhline(0.0, color=BASELINE, lw=1.0)
    ax_s.text(dense[0] * 1.15, -0.03, "saddle point (TST)", fontsize=7.6, color=MUTED, va="top")
    ax_s.set_ylim(-0.42, 0.48)
    ax_s.set_ylabel(r"$s_{\rm CVT}$ (Å)")
    ax_s.yaxis.set_label_coords(-0.19, 0.5)
    ax_s.set_title("Bottleneck location moves with T", loc="left", fontsize=9.2)
    ax_s.tick_params(labelbottom=False)
    (s_dot,) = ax_s.plot([], [], **ring_marker(ORANGE))

    ax_k.plot(dense, ratio, color=ORANGE)
    ax_k.axhline(1.0, color=BASELINE, lw=1.0)
    ax_k.set_ylim(0.55, 1.06)
    ax_k.set_ylabel(r"$k_{\rm CVT}/k_{\rm TST}$")
    ax_k.yaxis.set_label_coords(-0.19, 0.5)
    ax_k.set_xlabel("T (K)")
    ax_k.set_title("Variational effect (≤ 1)", loc="left", fontsize=9.2)
    log_temperature_axis(ax_k, (150, 300, 600, 1000, 1500))
    ax_k.set_xlim(dense[0], dense[-1])
    (k_dot,) = ax_k.plot([], [], **ring_marker(ORANGE))
    cursors = [
        ax_s.axvline(dense[0], color=INK2, lw=1.0),
        ax_k.axvline(dense[0], color=INK2, lw=1.0),
    ]

    temperatures = ping_pong(np.geomspace(150.0, 1500.0, 42))
    for temperature in temperatures:
        t = float(temperature)
        g = profile(t)
        cvt = cvt_rate_si(reaction, t)
        delta_tst, k_tst = conventional_tst_rate_si(reaction, t)
        fit_x, fit_y = _local_fit(s_values, g)
        if not math.isclose(float(np.max(fit_y)), cvt.activation_gibbs_kj_mol, abs_tol=2e-3):
            raise RuntimeError("drawn polynomial disagrees with refine_maximum()")
        fit_line.set_data(fit_x, fit_y)
        samples.set_data(s_values, g)
        tst_point.set_data([0.0], [delta_tst])
        cvt_point.set_data([cvt.coordinate], [cvt.activation_gibbs_kj_mol])
        cvt_label.set_position((cvt.coordinate, cvt.activation_gibbs_kj_mol + 0.4))
        cvt_label.set_text(f"CVT · s = {num(cvt.coordinate, '+.2f')} Å")
        readout.set_text(
            f"T = {t:.0f} K\n"
            + r"$\Delta G^\ddagger_{\rm CVT}-\Delta G^\ddagger_{\rm TST}$"
            + f" = {num(cvt.activation_gibbs_kj_mol - delta_tst, '+.2f')} kJ mol⁻¹\n"
            + r"$k_{\rm CVT}/k_{\rm TST}$"
            + f" = {cvt.rate_si / k_tst:.3f}"
        )
        s_dot.set_data([t], [cvt.coordinate])
        k_dot.set_data([t], [cvt.rate_si / k_tst])
        for cursor in cursors:
            cursor.set_xdata([t, t])
        endpoint = abs(t - 150.0) < 1e-6 or abs(t - 1500.0) < 1e-6
        movie.grab(FRAME_MS + (800 if endpoint else 0))
    return movie


# ---------------------------------------------------------------------------
# 4. Tunneling
# ---------------------------------------------------------------------------


def render_tunneling() -> Movie:
    from pyrate import load_toml
    from pyrate.constants import AVOGADRO_MOL, BOLTZMANN_J_K
    from pyrate.tunneling import (
        profile_transmission_probability,
        profile_wkb_kappa,
        transition_state_wigner_kappa,
    )

    job = load_toml(EXAMPLES / "wkb_tst.toml")
    profile = job.options.tunneling_profile
    assert profile is not None
    ts = job.reaction.transition_state
    x_nodes = np.asarray(profile.coordinates_angstrom)
    v_nodes = np.asarray(profile.energies_kj_mol) - profile.energies_kj_mol[0]
    barrier = float(v_nodes.max())
    x_dense = np.linspace(x_nodes[0], x_nodes[-1], 600)
    v_dense = np.interp(x_dense, x_nodes, v_nodes)

    energies = np.linspace(0.2, barrier - 0.05, 220)
    probability = np.array([profile_transmission_probability(profile, float(e)) for e in energies])
    p_floor = 1e-17
    t_dense = np.geomspace(150.0, 800.0, 70)
    kappa_wkb = np.array([profile_wkb_kappa(profile, float(t)) for t in t_dense])
    kappa_wig = np.array([transition_state_wigner_kappa(ts, float(t)) for t in t_dense])

    def weight(temperature: float) -> np.ndarray:
        beta = 1000.0 / (AVOGADRO_MOL * BOLTZMANN_J_K * temperature)  # per kJ/mol
        log_f = beta * (barrier - energies) + np.log(np.maximum(probability, 1e-300))
        return np.exp(log_f - log_f.max())

    movie = Movie(
        "Quantum tunneling: 1D WKB transmission and thermal κ(T)",
        f"examples/wkb_tst.toml barrier · μ = {profile.reduced_mass_amu:g} amu · "
        "κ = 1 + ∫ β P(E) exp[β(V‡−E)] dE · Wigner uses ν‡ = "
        f"{abs(ts.imaginary_frequencies_cm1[0]):.0f}i cm⁻¹",
    )
    fig = movie.fig
    ax_v = fig.add_axes((0.07, 0.13, 0.32, 0.64))
    ax_p = fig.add_axes((0.415, 0.13, 0.17, 0.64), sharey=ax_v)
    ax_k = fig.add_axes((0.675, 0.13, 0.29, 0.64))

    ax_v.plot(x_dense, v_dense, color=INK2)
    ax_v.set_xlim(x_nodes[0] - 0.05, x_nodes[-1] + 0.05)
    ax_v.set_ylim(0.0, 42.0)
    ax_v.set_xlabel("x (Å)")
    ax_v.set_ylabel("energy (kJ mol⁻¹)")
    ax_v.text(
        0.03,
        0.975,
        r"$\gamma(E)=\frac{2}{\hbar}\int_{x_1}^{x_2}\sqrt{2\mu\,[V(x)-E]}\,dx$",
        transform=ax_v.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=INK,
    )
    level = ax_v.plot([], [], color=BLUE, lw=1.6, solid_capstyle="butt")[0]
    turning = ax_v.plot([], [], **ring_marker(BLUE, 7.0))[0]
    shade = [None]

    ax_p.set_xscale("log")
    ax_p.set_xlim(p_floor, 3.0)
    ax_p.set_xticks([1e-15, 1e-10, 1e-5, 1.0])
    ax_p.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_p.tick_params(labelleft=False)
    ax_p.set_xlabel("probability")
    ax_p.set_title(r"$P(E)=e^{-\gamma(E)}$", loc="left", fontsize=9.2)
    (p_line,) = ax_p.plot([], [], color=BLUE, label="P(E)")
    (p_dot,) = ax_p.plot([], [], **ring_marker(BLUE, 7.0))
    (w_line,) = ax_p.plot([], [], color=AQUA, label="thermal weight (T)")

    ax_k.set_yscale("log")
    ax_k.set_ylim(1.0, 14.0)
    ax_k.set_yticks([1, 2, 5, 10])
    ax_k.set_yticklabels(["1", "2", "5", "10"])
    ax_k.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    log_temperature_axis(ax_k, (150, 200, 300, 500, 800))
    ax_k.set_xlim(t_dense[0], t_dense[-1])
    ax_k.set_xlabel("T (K)")
    ax_k.set_title("Transmission coefficient κ(T)", loc="left", fontsize=9.2)
    (kw_line,) = ax_k.plot([], [], color=BLUE, label="1D WKB (profile)")
    (kg_line,) = ax_k.plot([], [], color=ORANGE, label="Wigner (ν‡)")
    (kw_dot,) = ax_k.plot([], [], **ring_marker(BLUE, 7.0))
    (kg_dot,) = ax_k.plot([], [], **ring_marker(ORANGE, 7.0))
    ax_k.legend(loc="upper right", handlelength=1.4)
    k_readout = ax_k.text(
        0.96,
        0.66,
        "",
        transform=ax_k.transAxes,
        ha="right",
        va="center",
        fontsize=8.4,
        color=INK,
        linespacing=1.5,
    )
    placeholder = ax_k.text(
        0.5,
        0.45,
        "next: Boltzmann average\nover tunneling energies",
        transform=ax_k.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        color=MUTED,
        linespacing=1.5,
    )

    def show_energy(energy: float, color: str) -> None:
        above = v_dense > energy
        inside = np.where(above)[0]
        if shade[0] is not None:
            shade[0].remove()
        x1, x2 = float(x_dense[inside[0]]), float(x_dense[inside[-1]])
        level.set_data([x1, x2], [energy, energy])
        level.set_color(color)
        turning.set_data([x1, x2], [energy, energy])
        turning.set_markerfacecolor(color)
        shade[0] = ax_v.fill_between(
            x_dense, energy, v_dense, where=above, color=color, alpha=0.14, lw=0
        )

    # Phase 1: sweep the energy under the barrier.
    for energy in np.linspace(1.0, barrier - 0.4, 38):
        e = float(energy)
        p = profile_transmission_probability(profile, e)
        show_energy(e, BLUE)
        mask = energies <= e
        p_line.set_data(np.maximum(probability[mask], p_floor), energies[mask])
        p_dot.set_data([max(p, p_floor)], [e])
        ax_v.set_title(f"E = {e:4.1f} kJ mol⁻¹  →  P(E) = " + sci(p, 2), loc="left", fontsize=9.2)
        movie.grab()
    movie.hold(700)

    # Phase 2: thermal average over energies, temperature sweep.
    p_dot.set_data([], [])
    p_line.set_data(np.maximum(probability, p_floor), energies)
    ax_p.legend(loc="upper left", handlelength=1.2, fontsize=7.4)
    placeholder.set_text("")
    for index, temperature in enumerate(np.geomspace(150.0, 800.0, 34)):
        t = float(temperature)
        w = weight(t)
        dominant = float(energies[int(np.argmax(w))])
        show_energy(dominant, AQUA)
        w_line.set_data(np.maximum(w, p_floor), energies)
        ax_v.set_title(
            f"T = {t:.0f} K  →  peak weight at E = {dominant:.1f}", loc="left", fontsize=9.2
        )
        mask = t_dense <= t * (1 + 1e-9)
        kw_line.set_data(t_dense[mask], kappa_wkb[mask])
        kg_line.set_data(t_dense[mask], kappa_wig[mask])
        kappa_now = profile_wkb_kappa(profile, t)
        wigner_now = transition_state_wigner_kappa(ts, t)
        kw_dot.set_data([t], [kappa_now])
        kg_dot.set_data([t], [wigner_now])
        k_readout.set_text(
            f"T = {t:.0f} K\n"
            + r"$\kappa_{\rm WKB}$"
            + f" = {kappa_now:.2f}\n"
            + r"$\kappa_{\rm Wigner}$"
            + f" = {wigner_now:.2f}"
        )
        movie.grab(FRAME_MS + (1800 if index == 33 else 0))
    return movie


# ---------------------------------------------------------------------------
# 5. Beyer-Swinehart state counting
# ---------------------------------------------------------------------------


def render_dos() -> Movie:
    from pyrate import load_toml, vibrational_density_of_states

    job = load_toml(EXAMPLES / "harmonic_cvt.toml")
    frequencies = [f for f in job.reaction.reactants[0].positive_frequencies_cm1]
    grain = 25.0  # no ν/δE ratio lands on a .5 tie for these modes
    e_max = 8000.0
    energies, kernel_rho = vibrational_density_of_states(frequencies, grain, e_max)
    n_grains = energies.size

    # Re-run the same recurrence step by step for the animation.
    rho = np.zeros(n_grains)
    rho[0] = 1.0
    snapshots: list[tuple[int, int, int, np.ndarray]] = []
    stages: list[np.ndarray] = [rho.copy()]
    sweep_frames = 13
    for mode, frequency in enumerate(frequencies):
        step = max(1, int(math.floor(frequency / grain + 0.5)))
        stops = np.linspace(step, n_grains - 1, sweep_frames).round().astype(int)
        cursor = step
        for stop in stops:
            while cursor <= stop:
                rho[cursor] += rho[cursor - step]
                cursor += 1
            snapshots.append((mode, step, int(stop), rho.copy()))
        stages.append(rho.copy())
    if not np.array_equal(rho, kernel_rho):
        raise RuntimeError("animated recurrence disagrees with beyer_swinehart()")

    movie = Movie(
        "Microcanonical counting: Beyer–Swinehart on an energy grid",
        f"{len(frequencies)} reactant modes of examples/harmonic_cvt.toml · δE = {grain:g} cm⁻¹ "
        "· C++ beyer_swinehart() · groundwork for future μVT / RRKM",
    )
    fig = movie.fig
    ax_rho = fig.add_axes((0.075, 0.49, 0.83, 0.28))
    ax_n = fig.add_axes((0.075, 0.12, 0.83, 0.28), sharex=ax_rho)
    edges = np.append(energies - grain / 2.0, energies[-1] + grain / 2.0)
    stairs = ax_rho.stairs(np.zeros(n_grains), edges, fill=True, color=BLUE, lw=0)
    ax_rho.set_xlim(-60.0, e_max + 60.0)
    ax_rho.set_ylabel("states / grain")
    ax_rho.tick_params(labelbottom=False)
    source = ax_rho.axvline(0.0, color=ORANGE, lw=1.0, alpha=0.5)
    target = ax_rho.axvline(0.0, color=ORANGE, lw=1.6)
    hop = FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="-|>",
        mutation_scale=9,
        color=ORANGE,
        lw=1.3,
        connectionstyle="arc3,rad=-0.35",
    )
    ax_rho.add_patch(hop)
    recurrence = r"$\rho[i] \leftarrow \rho[i] + \rho[i-\nu_k/\delta E]$"
    ax_rho.text(
        0.985,
        0.93,
        "for each mode $k$:  " + recurrence,
        transform=ax_rho.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        color=INK,
    )

    totals = [np.cumsum(stage) for stage in stages]
    ax_n.set_yscale("log")
    ax_n.set_ylim(0.8, totals[-1][-1] * 3.0)
    ax_n.set_xlabel("vibrational energy E (cm⁻¹)")
    ax_n.set_ylabel("sum of states N(E)")
    ax_n.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    done_lines = []
    done_labels = []
    for _ in frequencies:
        (line,) = ax_n.plot([], [], color=BASELINE, lw=1.4)
        done_lines.append(line)
        done_labels.append(
            ax_n.text(e_max + 60, 1.0, "", fontsize=7.6, color=MUTED, va="center", ha="left")
        )
    (live,) = ax_n.plot([], [], color=BLUE)
    live_label = ax_n.text(
        e_max + 60, 1.0, "", fontsize=7.6, color=INK, va="center", ha="left", weight="bold"
    )
    summary = ax_n.text(
        0.015,
        0.92,
        "",
        transform=ax_n.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color=INK,
        linespacing=1.5,
    )

    for mode, step, stop, values in snapshots:
        stairs.set_data(values)
        ax_rho.set_ylim(0.0, max(2.0, float(values.max()) * 1.18))
        e_target = float(energies[stop])
        e_source = float(energies[stop - step])
        source.set_xdata([e_source, e_source])
        target.set_xdata([e_target, e_target])
        top = ax_rho.get_ylim()[1]
        hop.set_positions((e_source, top * 0.55), (e_target, top * 0.55))
        ax_rho.set_title(
            f"mode {mode + 1}/{len(frequencies)}: ν = {frequencies[mode]:.0f} cm⁻¹ "
            f"(step {step} grains)  ·  sweeping i = {stop}",
            loc="left",
            fontsize=9.2,
        )
        live.set_data(energies, np.cumsum(values))
        live_label.set_position((e_max + 60, float(np.sum(values))))
        live_label.set_text(f"{mode + 1} modes" if mode else "1 mode")
        placed = [math.log10(float(np.sum(values)))]
        for index in range(mode, -1, -1):
            end_value = float(totals[index][-1])
            done_lines[index].set_data(energies, totals[index])
            done_labels[index].set_position((e_max + 60, end_value))
            free = all(abs(math.log10(end_value) - other) > 0.3 for other in placed)
            done_labels[index].set_text(
                (f"{index} mode" + ("s" if index != 1 else "")) if free else ""
            )
            if free:
                placed.append(math.log10(end_value))
        summary.set_text(f"N({e_max:.0f} cm⁻¹) = {int(np.sum(values)):,} states")
        last = stop == n_grains - 1
        movie.grab(FRAME_MS + (450 if last else 0))
    source.set_alpha(0.0)
    target.set_alpha(0.0)
    hop.set_alpha(0.0)
    ax_rho.set_title(f"all {len(frequencies)} modes counted", loc="left", fontsize=9.2)
    done_labels[-1].set_text("")
    summary.set_text(
        f"N({e_max:.0f} cm⁻¹) = {int(kernel_rho.sum()):,} states\n"
        "✓ identical to the C++ beyer_swinehart() result"
    )
    movie.grab(3000)
    return movie


# ---------------------------------------------------------------------------
# 6. Normal-mode analysis
# ---------------------------------------------------------------------------

MDYN_A_TO_KJ_MOL_A2 = 602.214076  # 1 mdyn Å⁻¹ = 100 J m⁻² per molecule


def _finite_difference_hessian(
    energy: Callable[[np.ndarray], float], x0: np.ndarray, step: float = 1.0e-4
) -> np.ndarray:
    size = x0.size
    hessian = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            values = []
            for di, dj in ((step, step), (step, -step), (-step, step), (-step, -step)):
                x = x0.copy()
                x[i] += di
                x[j] += dj
                values.append(energy(x))
            hessian[i, j] = hessian[j, i] = (values[0] - values[1] - values[2] + values[3]) / (
                4.0 * step * step
            )
    return hessian


def _model_water() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0, theta0 = 0.9572, math.radians(104.52)
    half = theta0 / 2.0
    coordinates = np.array(
        [
            [-r0 * math.sin(half), r0 * math.cos(half), 0.0],
            [0.0, 0.0, 0.0],
            [r0 * math.sin(half), r0 * math.cos(half), 0.0],
        ]
    )
    k_r, k_rr, k_theta = (value * MDYN_A_TO_KJ_MOL_A2 for value in (8.45, -0.10, 0.69))

    def energy(flat: np.ndarray) -> float:
        h1, o, h2 = flat[0:3], flat[3:6], flat[6:9]
        r1, r2 = np.linalg.norm(h1 - o), np.linalg.norm(h2 - o)
        cosine = float(np.dot(h1 - o, h2 - o) / (r1 * r2))
        theta = math.acos(max(-1.0, min(1.0, cosine)))
        return float(
            0.5 * k_r * ((r1 - r0) ** 2 + (r2 - r0) ** 2)
            + k_rr * (r1 - r0) * (r2 - r0)
            + 0.5 * k_theta * (theta - theta0) ** 2
        )

    masses = np.array([1.008, 15.999, 1.008])
    return masses, coordinates, _finite_difference_hessian(energy, coordinates.reshape(-1))


def _model_h3_saddle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = 0.93
    coordinates = np.array([[-d, 0.0, 0.0], [0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    k_sym, k_asym, k_bend = (value * MDYN_A_TO_KJ_MOL_A2 for value in (2.5, -0.45, 0.07))

    def energy(flat: np.ndarray) -> float:
        a, m, b = flat[0:3], flat[3:6], flat[6:9]
        u1, u2 = a - m, b - m
        r1, r2 = np.linalg.norm(u1), np.linalg.norm(u2)
        symmetric, antisymmetric = (r1 - d) + (r2 - d), (r1 - d) - (r2 - d)
        bend_vector = u1 / r1 + u2 / r2
        return float(
            0.25 * k_sym * symmetric**2
            + 0.25 * k_asym * antisymmetric**2
            + 0.5 * k_bend * float(np.dot(bend_vector, bend_vector))
        )

    masses = np.array([1.008, 1.008, 1.008])
    return masses, coordinates, _finite_difference_hessian(energy, coordinates.reshape(-1))


def render_normal_modes() -> Movie:
    from pyrate import normal_modes

    water_m, water_x, water_h = _model_water()
    water = normal_modes(
        water_h, water_m, coordinates_angstrom=water_x, hessian_unit="kj/mol/angstrom^2"
    )
    h3_m, h3_x, h3_h = _model_h3_saddle()
    saddle = normal_modes(h3_h, h3_m, coordinates_angstrom=h3_x, hessian_unit="kj/mol/angstrom^2")
    if water.rigid_mode_count != 6 or saddle.rigid_mode_count != 5:
        raise RuntimeError("unexpected rigid-mode count")
    if len(saddle.imaginary_frequencies_cm1) != 1:
        raise RuntimeError("model saddle point must have exactly one imaginary mode")

    def cartesian(result: Any, index: int, masses: np.ndarray) -> np.ndarray:
        vector = result.eigenvectors_mass_weighted[:, index] / np.sqrt(np.repeat(masses, 3))
        vector = vector.reshape(-1, 3)
        return vector / np.max(np.linalg.norm(vector, axis=1))

    order = np.argsort(water.frequencies_cm1)  # bend < symmetric < antisymmetric
    panels = [
        (
            "H₂O · ν₂ bend",
            water,
            int(order[0]),
            water_m,
            water_x,
            ("H", "O", "H"),
            ((0, 1), (1, 2)),
            0.16,
        ),
        (
            "H₂O · ν₁ symmetric stretch",
            water,
            int(order[1]),
            water_m,
            water_x,
            ("H", "O", "H"),
            ((0, 1), (1, 2)),
            0.16,
        ),
        (
            "H₂O · ν₃ antisymmetric stretch",
            water,
            int(order[2]),
            water_m,
            water_x,
            ("H", "O", "H"),
            ((0, 1), (1, 2)),
            0.16,
        ),
        (
            "H···H···H saddle point · ν‡",
            saddle,
            int(np.argmin(saddle.frequencies_cm1)),
            h3_m,
            h3_x,
            ("H", "H", "H"),
            ((0, 1), (1, 2)),
            0.24,
        ),
    ]

    movie = Movie(
        "Normal modes: Hessian → mass weighting → rigid-mode projection → eigh",
        "normal_modes() on model valence force fields (illustrative, not ab initio) · "
        "negative eigenvalue → imaginary ν",
    )
    fig = movie.fig
    board = movie.board()
    boxes = [(0.025, 0.445), (0.51, 0.445), (0.025, 0.04), (0.51, 0.04)]
    width, height = 0.465, 0.375
    artists = []
    for (left, bottom), (title, result, index, masses, coords, symbols, bonds, amp) in zip(
        boxes, panels, strict=True
    ):
        rounded_box(board, left, bottom, width, height, fc=SURFACE, ec=GRID, lw=1.0)
        frequency = float(result.frequencies_cm1[index])
        imaginary = frequency < 0.0
        board.text(
            left + 0.015,
            bottom + height - 0.03,
            title,
            ha="left",
            va="top",
            fontsize=9.0,
            color=INK,
            weight="bold",
        )
        board.text(
            left + width - 0.015,
            bottom + height - 0.03,
            f"{abs(frequency):.0f}{'i' if imaginary else ''} cm⁻¹",
            ha="right",
            va="top",
            fontsize=9.5,
            color=ORANGE if imaginary else INK,
            weight="bold",
        )
        if imaginary:
            board.text(
                left + 0.015,
                bottom + 0.022,
                "negative eigenvalue: left out of q_vib, sets Wigner κ",
                ha="left",
                va="bottom",
                fontsize=7.6,
                color=INK2,
            )
        ax = fig.add_axes((left + 0.01, bottom + 0.035, width - 0.02, height - 0.09))
        ax.axis("off")
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2.0
        half_x = 1.6
        pixel_ratio = ((width - 0.02) * FIGSIZE[0]) / ((height - 0.09) * FIGSIZE[1])
        half_y = half_x / pixel_ratio
        ax.set_xlim(center[0] - half_x, center[0] + half_x)
        ax.set_ylim(center[1] - half_y, center[1] + half_y)
        direction = cartesian(result, index, masses)
        if abs(direction[:, 2]).max() > 1e-6:
            raise RuntimeError("panel modes must stay in the drawing plane")
        arrows = []
        for atom, symbol in enumerate(symbols):
            offset = 2.4 * amp * direction[atom, :2]
            if np.linalg.norm(offset) < 0.22:
                continue  # displacement smaller than the atom itself
            radius_pt = math.sqrt(700.0 if symbol == "O" else 380.0) / 2.0
            arrow = FancyArrowPatch(
                (0.0, 0.0),
                (0.0, 0.0),
                arrowstyle="-|>",
                mutation_scale=10,
                color=ORANGE,
                lw=1.6,
                shrinkA=radius_pt + 1.0,
                shrinkB=0.0,
                zorder=4,
            )
            ax.add_patch(arrow)
            arrows.append((atom, offset, arrow))
        bond_lines = [
            ax.plot(
                [],
                [],
                color=INK2,
                lw=3.0 if not imaginary else 2.2,
                linestyle="-" if not imaginary else (0, (1.2, 1.8)),
                zorder=2,
            )[0]
            for _ in bonds
        ]
        sizes = [700.0 if symbol == "O" else 380.0 for symbol in symbols]
        colors = [ATOM_O if symbol == "O" else ATOM_H for symbol in symbols]
        atoms = ax.scatter(
            coords[:, 0], coords[:, 1], s=sizes, c=colors, edgecolors=INK2, linewidths=1.1, zorder=3
        )
        artists.append((coords, direction, amp, bonds, bond_lines, atoms, arrows))

    for frame in range(36):
        phase = math.sin(2.0 * math.pi * frame / 36)
        for coords, direction, amp, bonds, bond_lines, atoms, arrows in artists:
            positions = coords + amp * phase * direction
            atoms.set_offsets(positions[:, :2])
            for atom, offset, arrow in arrows:
                start = positions[atom, :2]
                arrow.set_positions(tuple(start), tuple(start + offset))
            for (a, b), line in zip(bonds, bond_lines, strict=True):
                line.set_data(
                    [positions[a, 0], positions[b, 0]], [positions[a, 1], positions[b, 1]]
                )
        movie.grab(70)
    return movie


# ---------------------------------------------------------------------------
# 7. Polyrate legacy parsing
# ---------------------------------------------------------------------------

LEGACY_SAMPLE = """\
*GENERAL
 TITLE
  H + H2 -> H2 + H
 END
 ATOMS
  1 H
  2 H
  3 H
 END
 MYSTERY  17 alpha   # not a Polyrate keyword
*PATH
 SRANGE
  SLP  1.0
  SLM -1.0
 END
*TUNNEL
 ZCT
 SCT
*RATE
 TST
 CVT
 TEMP
  200.0
  300.0
  600.0
 END
"""


def render_legacy() -> Movie:
    from pyrate.legacy import parse_legacy_text

    document = parse_legacy_text(LEGACY_SAMPLE, source="sample.dat")
    warnings = document.compatibility_warnings()
    unsupported = {
        method
        for method in document.requested_methods()
        if any(
            method in [item.strip() for item in warning.split(":")[-1].split(",")]
            for warning in warnings
        )
    }
    known = {"TITLE", "ATOMS", "SRANGE", "ZCT", "SCT", "TST", "CVT", "TEMP"}

    lines = LEGACY_SAMPLE.splitlines()
    movie = Movie(
        "Polyrate compatibility: lossless legacy input parsing",
        "parse_legacy_text() keeps every section, block, comment and unknown keyword; "
        "unsupported methods become warnings",
    )
    board = movie.board()
    top, pitch = 0.815, 0.0262
    rounded_box(
        board,
        0.025,
        top - pitch * len(lines) - 0.01,
        0.40,
        pitch * len(lines) + 0.045,
        fc=SURFACE,
        ec=GRID,
        lw=1.0,
    )
    board.text(
        0.415, top, "sample.dat", fontsize=7.8, color=MUTED, va="center", ha="right", family=MONO
    )
    highlight = rounded_box(
        board, 0.03, top - pitch * 0.5, 0.39, pitch, radius=0.004, fc=BLUE_WASH, ec="none"
    )
    texts = []
    for number, line in enumerate(lines, start=1):
        y = top - (number - 1) * pitch
        board.text(
            0.045,
            y,
            f"{number:2d}",
            fontsize=7.6,
            color=MUTED,
            va="center",
            family=MONO,
            ha="right",
        )
        texts.append(
            board.text(
                0.055,
                y,
                line,
                fontsize=7.9,
                color=MUTED,
                va="center",
                family=MONO,
                weight="bold" if line.startswith("*") else "normal",
            )
        )

    tree_x, tree_top, tree_pitch = 0.475, 0.80, 0.0395
    items: list[tuple[int, str, str, str]] = []  # (reveal line, text, chip, chip color)
    for section in document.sections:
        items.append((section.line_number, f"*{section.name}", "", ""))
        for position, entry in enumerate(section.entries):
            branch = "└" if position == len(section.entries) - 1 else "├"
            if entry.body:
                reveal = entry.line_number + len(entry.body) + 1
                if entry.keyword == "TITLE":
                    detail = f'"{document.title}"'
                elif entry.keyword == "ATOMS":
                    detail = f"{len(document.atom_table)} atoms"
                elif entry.keyword == "TEMP":
                    detail = ", ".join(f"{t:g}" for t in document.temperatures_k) + " K"
                else:
                    detail = f"{len(entry.body)} lines kept"
                text = f"  {branch} {entry.keyword}  block · {detail}"
            else:
                reveal = entry.line_number
                text = f"  {branch} {entry.keyword} {' '.join(entry.arguments)}".rstrip()
            chip, chip_color = "", ""
            if entry.keyword not in known:
                chip, chip_color = "unknown → retained", INK2
            elif entry.keyword in unsupported:
                chip, chip_color = "⚠ not implemented yet", WARNING
            items.append((reveal, text, chip, chip_color))
    tree = []
    for row, (_, text, chip, chip_color) in enumerate(items):
        y = tree_top - row * tree_pitch
        label = board.text(
            tree_x,
            y,
            text,
            fontsize=8.3,
            va="center",
            family=MONO,
            color=INK,
            weight="bold" if text.startswith("*") else "normal",
            alpha=0.0,
        )
        chip_artist = None
        if chip:
            chip_artist = board.text(
                0.965,
                y,
                chip,
                fontsize=7.6,
                va="center",
                ha="right",
                color=INK,
                alpha=0.0,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "fc": SURFACE,
                    "ec": chip_color,
                    "lw": 1.2,
                    "alpha": 0.0,
                },
            )
        tree.append((label, chip_artist))

    summary_y = tree_top - len(items) * tree_pitch - 0.02
    summary = board.text(tree_x, summary_y, "", fontsize=8.2, va="top", color=INK, linespacing=1.55)

    def show(line_number: int) -> None:
        highlight.set_y(top - (line_number - 1) * pitch - pitch * 0.5)
        for number, text in enumerate(texts, start=1):
            text.set_color(INK if number <= line_number else MUTED)
        for (reveal, *_), (label, chip_artist) in zip(items, tree, strict=True):
            visible = reveal <= line_number
            label.set_alpha(1.0 if visible else 0.0)
            if chip_artist is not None:
                chip_artist.set_alpha(1.0 if visible else 0.0)
                chip_artist.get_bbox_patch().set_alpha(1.0 if visible else 0.0)

    board.text(
        tree_x, tree_top + 0.045, "LegacyInput (parsed)", fontsize=8.0, color=MUTED, va="center"
    )
    for number, line in enumerate(lines, start=1):
        show(number)
        pause = 600 if ("MYSTERY" in line or line.strip() == "SCT") else 0
        movie.grab(120 + pause)
    highlight.set_alpha(0.0)
    entry_count = sum(len(section.entries) for section in document.sections)
    accounted = {section.line_number for section in document.sections}
    for section in document.sections:
        for entry in section.entries:
            accounted.add(entry.line_number)
            if entry.is_block:  # body lines plus the closing END
                accounted.update(
                    range(entry.line_number + 1, entry.line_number + len(entry.body) + 2)
                )
    content = {n for n, line in enumerate(lines, start=1) if line.split("#", 1)[0].strip()}
    dropped = len(content - accounted)
    summary.set_text(
        f"sections {len(document.sections)} · entries {entry_count} · comments kept "
        f"{len(document.comments)} · lines dropped {dropped}\n"
        f"requested: {', '.join(document.requested_methods())}\n"
        + "\n".join(f"⚠ {warning}" for warning in warnings)
    )
    movie.grab(3600)
    return movie


# ---------------------------------------------------------------------------
# 8. Head-to-head fu15 comparison
# ---------------------------------------------------------------------------


def render_head_to_head() -> Movie:
    from pyrate import calculate_rates, load_toml
    from pyrate.head_to_head import compare_fu15_to_pyrate, parse_fu15_tables

    job = load_toml(EXAMPLES / "harmonic_cvt.toml")
    result = calculate_rates(job.reaction, job.options, name=job.name).as_dict()
    rows = result["rows"]
    header = "   T(K)      TST        CVT      CVT/ZCT"
    body = [
        f"  {row['temperature_k']:7.2f}  {row['k_tst']:9.2E}  {row['k_cvt']:9.2E}  "
        f"{row['k_final']:9.2E}"
        for row in rows
    ]
    fu15 = "\n".join([" Summary of forward rate constants (s-1) :", "", header, "", *body, ""])
    (table,) = parse_fu15_tables(fu15)
    report = compare_fu15_to_pyrate(table, result, relative_tolerance=0.005)
    strict = compare_fu15_to_pyrate(table, result, relative_tolerance=0.005, strict=True)

    movie = Movie(
        "Head-to-head gates: fu15 reference vs PyRate JSON",
        "Synthetic fu15 built from rounded PyRate output — shows the comparator's rules, "
        "not agreement with Polyrate",
    )
    fig = movie.fig
    board = movie.board()
    text_top = 0.80
    rounded_box(board, 0.025, 0.49, 0.44, 0.355, fc=SURFACE, ec=GRID, lw=1.0)
    board.text(
        0.455,
        text_top - 0.033,
        "case.fu15 (synthetic)",
        fontsize=7.8,
        color=MUTED,
        va="center",
        ha="right",
        family=MONO,
    )
    fu15_lines = fu15.splitlines()
    row_artists = []
    highlight = rounded_box(board, 0.03, 0.0, 0.43, 0.03, radius=0.004, fc=BLUE_WASH, ec="none")
    highlight.set_alpha(0.0)
    for number, line in enumerate(fu15_lines):
        y = text_top - number * 0.033
        artist = board.text(
            0.04, y, line, fontsize=7.9, family=MONO, va="center", color=INK if number < 3 else INK2
        )
        row_artists.append((y, artist))
    board.text(
        0.035,
        0.445,
        "python tools/compare_polyrate_rates.py \\\n    case.fu15 pyrate.json --rtol 0.005",
        fontsize=7.6,
        family=MONO,
        color=INK2,
        va="top",
        linespacing=1.5,
    )
    verdict_rows = []
    for index in range(3):
        y = 0.32 - index * 0.065
        icon = board.text(0.035, y, "", fontsize=10, va="center", weight="bold")
        text = board.text(0.065, y, "", fontsize=8.6, va="center", color=INK)
        verdict_rows.append((icon, text))

    ax = fig.add_axes((0.555, 0.17, 0.415, 0.60))
    columns = list(table.columns)
    temps = [row.temperature_k for row in table.rows]
    positions = {}
    for c_index, column in enumerate(columns):
        for t_index, temperature in enumerate(temps):
            positions[(column, temperature)] = c_index * (len(temps) + 1.4) + t_index
    ax.set_xlim(-0.8, max(positions.values()) + 0.8)
    ax.set_ylim(0.0, 0.62)
    ax.axhline(0.5, color=CRITICAL, lw=1.2)
    ax.text(-0.6, 0.515, "rtol = 0.5 %", color=INK2, fontsize=7.8, va="bottom")
    ax.set_ylabel("relative error (%)")
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    for c_index, column in enumerate(columns):
        middle = c_index * (len(temps) + 1.4) + (len(temps) - 1) / 2.0
        ax.text(
            middle,
            -0.04,
            column,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.4,
            color=INK,
            weight="bold",
        )
    ax.text(
        0.0,
        -0.13,
        "within each column: T = " + ", ".join(f"{t:g}" for t in temps) + " K, left to right",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color=INK2,
    )
    status_of = {(p.reference_column, p.temperature_k): p for p in report.points}
    unmapped_columns = [
        c for c in columns if all(status_of[(c, t)].status == "unmapped" for t in temps)
    ]
    unmapped_band = []
    for column in unmapped_columns:
        c_index = columns.index(column)
        start = c_index * (len(temps) + 1.4) - 0.5
        band = ax.axvspan(start, start + len(temps), color=GRID, alpha=0.0, lw=0)
        label = ax.text(
            start + len(temps) / 2.0,
            0.25,
            "unmapped\nnot counted\nas a pass",
            ha="center",
            va="center",
            fontsize=8.0,
            color=INK2,
            alpha=0.0,
            linespacing=1.4,
        )
        unmapped_band.append((band, label))
    (pass_dots,) = ax.plot([], [], **ring_marker(GOOD, 8.5), label="pass")
    (fail_dots,) = ax.plot([], [], **ring_marker(CRITICAL, 8.5), label="fail")
    ax.legend(loc="upper right", handletextpad=0.3, fontsize=7.6, ncol=2)

    first_row = 4
    shown: list[Any] = []
    movie.grab(900)
    for r_index, temperature in enumerate(temps):
        y, _ = row_artists[first_row + r_index]
        highlight.set_y(y - 0.015)
        highlight.set_alpha(1.0)
        for column in columns:
            point = status_of[(column, temperature)]
            if point.status in {"pass", "fail"}:
                shown.append(point)
        passes = [p for p in shown if p.status == "pass"]
        fails = [p for p in shown if p.status == "fail"]
        pass_dots.set_data(
            [positions[(p.reference_column, p.temperature_k)] for p in passes],
            [100.0 * (p.relative_error or 0.0) for p in passes],
        )
        fail_dots.set_data(
            [positions[(p.reference_column, p.temperature_k)] for p in fails],
            [min(0.6, 100.0 * (p.relative_error or 0.0)) for p in fails],
        )
        if r_index == 0:
            for band, label in unmapped_band:
                band.set_alpha(0.55)
                label.set_alpha(1.0)
        movie.grab(750)
    highlight.set_alpha(0.0)
    lines = [
        (
            report.success,
            f"default: {report.passed} passed · {report.failed} failed · "
            f"{report.unmapped} unmapped → exit {0 if report.success else 1}",
        ),
        (
            strict.success,
            f"--strict: unmapped columns fail the run → exit {0 if strict.success else 1}",
        ),
        (None, "CVT/ZCT is not mapped by default: 1D WKB ≠ Polyrate ZCT"),
    ]
    for (icon, text), (ok, message) in zip(verdict_rows, lines, strict=True):
        if ok is None:
            icon.set_text("i")
            icon.set_color(MUTED)
        else:
            icon.set_text("✓" if ok else "✗")
            icon.set_color(GOOD if ok else CRITICAL)
        text.set_text(message)
    movie.grab(4200)
    return movie


# ---------------------------------------------------------------------------
# 9. Native kernels vs Python fallback
# ---------------------------------------------------------------------------

BENCH_SNIPPET = r"""
import json, sys, timeit
sys.path.insert(0, sys.argv[1])
from pyrate.native import backend_info, beyer_swinehart, thermal_wkb_kappa
coords = [i * 1.0e-12 for i in range(401)]
center = coords[len(coords) // 2]
pots = [4.0e-20 * max(0.0, 1.0 - ((x - center) / 2.0e-10) ** 2) for x in coords]
freqs = (300, 450, 700, 1000, 1200, 1500)
dos = lambda: beyer_swinehart(freqs, 1.0, 20_000)
wkb = lambda: thermal_wkb_kappa(coords, pots, 300.0, 1.67e-27, 400)
dos(); wkb()
print(json.dumps({
    "backend": backend_info()["backend"],
    "dos_s": min(timeit.repeat(dos, number=1, repeat=5)),
    "wkb_s": min(timeit.repeat(wkb, number=1, repeat=5)),
    "dos": dos().tolist(),
    "kappa": wkb(),
}))
"""


def _measure_backend(disable_native: bool) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYRATE_DISABLE_NATIVE"] = "1" if disable_native else "0"
    completed = subprocess.run(
        [sys.executable, "-c", BENCH_SNIPPET, str(ROOT / "src")],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    data: dict[str, Any] = json.loads(completed.stdout)
    return data


def render_backends() -> Movie:
    native = _measure_backend(disable_native=False)
    fallback = _measure_backend(disable_native=True)
    if native["backend"] != "cpp":
        raise RuntimeError("build the C++ extension first: python -m pip install -e .")
    dos_a, dos_b = np.asarray(native["dos"]), np.asarray(fallback["dos"])
    dos_diff = float(np.max(np.abs(dos_a - dos_b) / np.maximum(1.0, np.abs(dos_b))))
    kappa_diff = abs(native["kappa"] / fallback["kappa"] - 1.0)

    calls = 20
    kernels = [
        ("Beyer–Swinehart DOS · 6 modes × 20 000 grains", "dos_s"),
        ("thermal WKB κ · 401-point barrier × 400 energies", "wkb_s"),
    ]
    slowest = calls * max(fallback[key] for _, key in kernels)
    replay = max(1.0, slowest / 3.2)
    frame_s = FRAME_MS / 1000.0
    frames = int(math.ceil(slowest / replay / frame_s)) + 1

    movie = Movie(
        "Engineering: C++ kernels with a pure-Python fallback",
        "Workloads from tools/benchmark_kernels.py · 20 calls each · timings measured on the "
        "machine that rendered this GIF",
    )
    board = movie.board()
    track_left, track_right = 0.20, 0.70
    bar_h = 0.044
    rows = []
    for k_index, (label, key) in enumerate(kernels):
        group_top = 0.775 - k_index * 0.285
        board.text(0.035, group_top, label, fontsize=9.2, weight="bold", color=INK, va="center")
        for b_index, (name, data, color) in enumerate(
            (("C++", native, BLUE), ("Python", fallback, ORANGE))
        ):
            y = group_top - 0.085 - b_index * 0.085
            board.text(
                track_left - 0.015, y, name, fontsize=8.8, color=INK2, ha="right", va="center"
            )
            rounded_box(
                board,
                track_left,
                y - bar_h / 2,
                track_right - track_left,
                bar_h,
                radius=0.006,
                fc=GRID,
                ec="none",
            )
            fill = rounded_box(
                board, track_left, y - bar_h / 2, 0.0001, bar_h, radius=0.006, fc=color, ec="none"
            )
            readout = board.text(track_right + 0.015, y, "", fontsize=8.4, color=INK, va="center")
            rows.append((data[key] * calls, fill, readout))
        speedup = fallback[key] / native[key]
        rows_speed = board.text(
            0.965,
            group_top - 0.1275,
            f"×{speedup:.0f}\nfaster",
            fontsize=10.5,
            weight="bold",
            color=INK,
            ha="right",
            va="center",
            alpha=0.0,
            linespacing=1.2,
        )
        rows.append((float("nan"), rows_speed, None))
    clock = board.text(
        0.965, 0.775, "", fontsize=8.4, color=INK2, ha="right", va="center", family=MONO
    )
    parity_icon = board.text(0.035, 0.115, "", fontsize=10, weight="bold", color=GOOD, va="center")
    parity = board.text(0.065, 0.115, "", fontsize=8.6, color=INK, va="center")
    board.text(
        0.065,
        0.06,
        "PYRATE_DISABLE_NATIVE=1 pytest -q runs the full test suite on the Python fallback",
        fontsize=7.8,
        color=INK2,
        va="center",
    )

    for frame in range(frames + 1):
        elapsed = min(slowest, frame * frame_s * replay)
        clock.set_text(
            f"t = {elapsed:5.2f} s"
            + (f"  (replay ×{replay:.1f})" if replay > 1.0 else "  (real time)")
        )
        for total, artist, readout in rows:
            if readout is None:
                continue
            progress = min(1.0, elapsed / total)
            artist.set_width(max(0.0001, (track_right - track_left) * progress))
            done = int(calls * progress)
            readout.set_text(
                f"{done:2d}/{calls} calls" + (f" · {total:.3f} s" if progress >= 1.0 else "")
            )
        movie.grab()
    for _, artist, readout in rows:
        if readout is None:
            artist.set_alpha(1.0)

    def difference(value: float) -> str:
        return "0" if value == 0.0 else f"{value:.0e}"

    same = dos_diff < 1e-12 and kappa_diff < 1e-12
    parity_icon.set_text("✓" if same else "!")
    parity_icon.set_color(GOOD if same else WARNING)
    parity.set_text(
        ("Same numbers" if same else "Backends differ")
        + f" on these workloads: max relative difference DOS {difference(dos_diff)}"
        f" · κ {difference(kappa_diff)}"
    )
    movie.grab(4000)
    return movie


# ---------------------------------------------------------------------------

FIGURES: dict[str, tuple[str, Callable[[], Movie]]] = {
    "workflow": ("workflow.gif", render_workflow),
    "partition": ("partition_functions.gif", render_partition),
    "cvt": ("tst_cvt.gif", render_cvt),
    "tunneling": ("tunneling.gif", render_tunneling),
    "dos": ("beyer_swinehart.gif", render_dos),
    "modes": ("normal_modes.gif", render_normal_modes),
    "legacy": ("legacy_parser.gif", render_legacy),
    "compare": ("head_to_head.gif", render_head_to_head),
    "backends": ("native_backend.gif", render_backends),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "figures", nargs="*", metavar="FIGURE", help=f"subset to render: {', '.join(FIGURES)}"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="list figure names and exit")
    args = parser.parse_args(argv)
    if args.list:
        for name, (filename, _) in FIGURES.items():
            print(f"{name:10s} {filename}")
        return 0
    unknown = sorted(set(args.figures) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figure(s): {', '.join(unknown)}")

    sys.path.insert(0, str(ROOT / "src"))
    from pyrate import HAS_NATIVE

    if not HAS_NATIVE:
        print("warning: C++ kernels not built; rendering with the Python fallback", file=sys.stderr)
    apply_style()
    for name in args.figures or list(FIGURES):
        filename, render = FIGURES[name]
        movie = render()
        size = movie.save(args.output / filename)
        print(f"{filename:26s} {len(movie.frames):4d} frames  {size / 1024:7.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
