
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BCSA Bar Chart Studio
=====================
A Streamlit app for visualizing Bridge-Criticality Sensitivity Analysis (BCSA)
results as customizable 2D bar charts.

Each CSV in the `bcsa_datasets/` folder contains the BCSA results for ONE concept,
with columns: Metric, Before, After, Δ (After − Before)

Features
--------
- Load BCSA CSVs from a directory next to the script (anchored via __file__)
- Multi-select concepts to compare
- Choose which metric to plot (Δ Modularity, Δ Density, Δ Avg path, etc.)
- 2D diverging bar chart (handles positive AND negative values with zero line)
- Sign-based coloring (positive=green, negative=red) OR value-based colormap
- Grouped bar chart view: all metrics side-by-side for each concept
- Before-vs-After comparison view for a single concept
- Figure customization: size, DPI, fonts, colormap, sort order, sign colors
- Role classification: Bridge (ΔQ > 0) vs. Hub-Anchor (ΔQ < 0)
- Downloads: PNG, CSV

Run:
  streamlit run bcsa_bar_chart_studio.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import io
import re
from typing import List, Dict, Tuple, Optional

# ---- Register CJK + Latin fallback fonts (rule #7) ----
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
except Exception:
    pass
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="BCSA Bar Chart Studio",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 50+ COLORMAP REGISTRY (same as SIS app)
# =====================================================================
COLORMAPS = {
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno",
    "magma": "Magma", "cividis": "Cividis", "turbo": "Turbo",
    "jet": "Jet", "rainbow": "Rainbow", "hsv": "HSV",
    "nipy_spectral": "NipySpectral", "gist_rainbow": "GistRainbow",
    "coolwarm": "Coolwarm", "RdBu": "RdBu", "seismic": "Seismic",
    "Spectral": "Spectral", "bwr": "Bluered", "BrBG": "BrBG",
    "PiYG": "PiYG", "PRGn": "PRGn", "PuOr": "PuOr", "RdYlBu": "RdYlBu",
    "RdYlGn": "RdYlGn", "Blues": "Blues", "Reds": "Reds", "Greens": "Greens",
    "Oranges": "Oranges", "Purples": "Purples", "Greys": "Greys",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu",
    "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn",
    "BuGn": "BuGn", "YlGn": "YlGn", "twilight": "Twilight", "cubehelix": "Cubehelix",
    "bone": "Bone", "gray": "Greys", "pink": "Pink", "spring": "Spring",
    "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "cool": "Cool", "hot": "Hot", "copper": "Copper", "afmhot": "Afmhot",
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean",
    "tab10": "Set1", "tab20": "Set2", "tab20b": "Set3", "tab20c": "Set3",
    "Accent": "Accent", "Dark2": "Dark2", "Paired": "Paired",
    "Pastel1": "Pastel1", "Pastel2": "Pastel2", "Set1": "Set1", "Set2": "Set2", "Set3": "Set3",
    "viridis_r": "Viridis_r", "plasma_r": "Plasma_r", "inferno_r": "Inferno_r",
    "magma_r": "Magma_r", "turbo_r": "Turbo_r", "jet_r": "Jet_r",
}
COLORMAP_NAMES = list(COLORMAPS.keys())


def get_cmap_colors(cmap_name: str, n: int) -> List[str]:
    """Return n hex colors sampled from the named matplotlib colormap."""
    try:
        if cmap_name.endswith("_r"):
            base = cmap_name[:-2]
            cmap = matplotlib.colormaps.get_cmap(base).resampled(max(n, 1)).reversed()
        else:
            cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(max(n, 1))
        return [mcolors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap("viridis", max(n, 1))
        except Exception:
            cmap = cm.get_cmap("viridis")
        return [mcolors.to_hex(cmap(i)) for i in range(n)]


# =====================================================================
# CSV LOADING — anchored to script location via __file__
# =====================================================================
def get_default_dataset_dir() -> str:
    """Return the absolute path to the bcsa_datasets folder next to this script."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    return os.path.join(script_dir, "bcsa_datasets")


def parse_delta(value_str: str) -> Tuple[Optional[float], bool]:
    """
    Parse a Δ value string like '+0.0239', '-0.0310', '+0', '∞'.

    Returns (float_value_or_None, is_infinity).
    """
    if value_str is None:
        return None, False
    s = str(value_str).strip()
    if s in ("∞", "inf", "infinity", "+∞", "-∞"):
        return None, True
    # Remove leading + sign if present
    s = s.lstrip("+")
    try:
        return float(s), False
    except (ValueError, TypeError):
        return None, False


def load_bcsa_csv(filepath: str) -> pd.DataFrame:
    """
    Load a single BCSA CSV file.

    Expected format:
        Metric,Before,After,Δ (After − Before)
        Modularity Q,0.0126,0.0365,+0.0239
        ...

    Returns a DataFrame with columns: Metric, Before, After, Delta, DeltaInf
    """
    df = pd.read_csv(filepath)
    # Normalize column names (the Δ column has special chars)
    cols = list(df.columns)
    if len(cols) >= 4:
        df = df.rename(columns={cols[0]: "Metric", cols[1]: "Before",
                                 cols[2]: "After", cols[3]: "Delta"})
    elif len(cols) == 3:
        # Maybe Δ column is missing — compute it
        df = df.rename(columns={cols[0]: "Metric", cols[1]: "Before", cols[2]: "After"})
        df["Delta"] = df["After"] - df["Before"]

    # Parse the Delta column (may contain "+0.0239", "∞", etc.)
    parsed = df["Delta"].apply(parse_delta)
    df["DeltaValue"] = parsed.apply(lambda x: x[0])
    df["DeltaInf"] = parsed.apply(lambda x: x[1])

    # Also parse Before and After as floats
    for col in ["Before", "After"]:
        df[col] = df[col].apply(lambda x: parse_delta(x)[0] if parse_delta(x)[0] is not None else np.nan)

    return df


def load_all_bcsa_datasets(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the given directory.

    Returns a dict mapping concept name (filename without .csv) → DataFrame.
    """
    concepts = {}
    if not os.path.isdir(dataset_dir):
        return concepts
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.lower().endswith(".csv"):
            concept = fname[:-4]  # strip .csv
            try:
                df = load_bcsa_csv(os.path.join(dataset_dir, fname))
                concepts[concept] = df
            except Exception as e:
                st.error(f"Failed to load {fname}: {e}")
    return concepts


# =====================================================================
# METRIC DEFINITIONS
# =====================================================================
# The standard BCSA metrics (in order they appear in the CSV)
ALL_METRICS = [
    "Modularity Q",
    "Avg. shortest path ⟨d⟩",
    "# Connected components",
    "Avg. clustering",
    "Density",
]

# Friendly short names for display
METRIC_SHORT = {
    "Modularity Q": "Δ Modularity",
    "Avg. shortest path ⟨d⟩": "Δ Avg Path",
    "# Connected components": "Δ Components",
    "Avg. clustering": "Δ Clustering",
    "Density": "Δ Density",
}


# =====================================================================
# ROLE CLASSIFICATION (based on Δ Modularity sign)
# =====================================================================
def classify_role(delta_modularity: Optional[float], delta_density: Optional[float]) -> str:
    """
    Classify a node's structural role based on ΔQ and ΔDensity.

    Returns one of: 'Bridge', 'Hub-Anchor', 'Sparse-Anchor', 'Peripheral'
    """
    if delta_modularity is None or np.isnan(delta_modularity):
        return "Unknown"
    if delta_modularity > 0.005:
        # Positive ΔQ = boundary-spanner (bridge)
        if delta_density is not None and not np.isnan(delta_density) and delta_density < -0.001:
            return "Bridge (high-deg)"
        return "Bridge"
    elif delta_modularity < -0.005:
        # Negative ΔQ = community anchor
        if delta_density is not None and not np.isnan(delta_density) and delta_density > 0.001:
            return "Sparse-Anchor"
        elif delta_density is not None and not np.isnan(delta_density) and delta_density < -0.001:
            return "Hub-Anchor"
        return "Anchor"
    else:
        return "Peripheral"


# =====================================================================
# 2D DIVERGING BAR CHART (handles + and - values)
# =====================================================================
def render_diverging_bar_chart(
    concepts: List[str],
    delta_values: List[float],
    metric_name: str,
    cmap_name: str,
    color_mode: str,  # "sign" or "colormap"
    figsize: Tuple[float, float],
    dpi: int,
    bar_width: float,
    label_fontsize: int,
    x_label_fontsize: int,
    x_label_rotation: int,
    show_value_labels: bool,
    show_zero_line: bool,
    show_grid: bool,
    title: str,
    bg_color: str,
    text_color: str,
    sort_order: str,
    positive_color: str = "#2ca02c",  # green
    negative_color: str = "#d62728",  # red
) -> bytes:
    """Render a 2D diverging bar chart and return PNG bytes."""
    if not concepts:
        return b""

    # Build dataframe for sorting
    df_plot = pd.DataFrame({"concept": concepts, "delta": delta_values})

    # Sort
    if sort_order == "descending":
        df_plot = df_plot.sort_values("delta", ascending=False).reset_index(drop=True)
    elif sort_order == "ascending":
        df_plot = df_plot.sort_values("delta", ascending=True).reset_index(drop=True)
    elif sort_order == "absolute_desc":
        df_plot = df_plot.loc[df_plot["delta"].abs().sort_values(ascending=False).index].reset_index(drop=True)
    else:  # alphabetical
        df_plot = df_plot.sort_values("concept", ascending=True).reset_index(drop=True)

    concepts_sorted = df_plot["concept"].tolist()
    deltas_sorted = df_plot["delta"].tolist()
    n = len(concepts_sorted)

    # Colors
    if color_mode == "sign":
        colors = [positive_color if d >= 0 else negative_color for d in deltas_sorted]
    else:  # colormap
        # Map values to colormap (normalize to [0, 1] based on min/max)
        if len(deltas_sorted) > 0:
            vmin, vmax = min(deltas_sorted), max(deltas_sorted)
            if vmin == vmax:
                vmin, vmax = vmin - 1, vmax + 1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            try:
                cmap = matplotlib.colormaps.get_cmap(cmap_name)
            except Exception:
                cmap = cm.get_cmap(cmap_name)
            colors = [mcolors.to_hex(cmap(norm(d))) for d in deltas_sorted]
        else:
            colors = []

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    x_positions = np.arange(n)
    bars = ax.bar(
        x_positions, deltas_sorted,
        width=bar_width,
        color=colors,
        edgecolor=text_color if bg_color != "white" else "#333333",
        linewidth=0.6,
        alpha=0.92,
    )

    # Value labels at the tip of each bar
    if show_value_labels:
        for i, (bar, val) in enumerate(zip(bars, deltas_sorted)):
            if val >= 0:
                y = val + (max(deltas_sorted) - min(deltas_sorted)) * 0.015 + 0.0005
                va = "bottom"
            else:
                y = val - (max(deltas_sorted) - min(deltas_sorted)) * 0.015 - 0.0005
                va = "top"
            # Choose text color based on bar color luminance
            r, g, b, _ = mcolors.to_rgba(colors[i])
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            label_color = text_color
            ax.text(
                bar.get_x() + bar.get_width() / 2, y,
                f"{val:+.4f}",
                ha="center", va=va,
                fontsize=label_fontsize, fontweight="bold", color=label_color,
            )

    # Zero line (critical for diverging charts)
    if show_zero_line:
        ax.axhline(y=0, color=text_color, linewidth=1.5, linestyle="-", alpha=0.7, zorder=2)

    # X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [c.replace("_", " ") for c in concepts_sorted],
        fontsize=x_label_fontsize,
        rotation=x_label_rotation,
        ha="right" if x_label_rotation != 0 else "center",
        color=text_color,
    )

    # Y-axis
    metric_short = METRIC_SHORT.get(metric_name, metric_name)
    ax.set_ylabel(metric_short, fontsize=label_fontsize + 2, color=text_color, fontweight="bold")
    ax.set_title(title, fontsize=label_fontsize + 4, fontweight="bold", color=text_color, pad=14)
    ax.tick_params(axis="y", labelsize=label_fontsize, colors=text_color)

    # Grid
    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#888888")
    ax.set_axisbelow(True)

    # Hide top/right spines
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(text_color)

    # Adjust y-limits with padding for labels
    if deltas_sorted:
        ymin, ymax = min(deltas_sorted), max(deltas_sorted)
        pad = (ymax - ymin) * 0.15 if ymax != ymin else abs(ymax) * 0.15 + 0.001
        ax.set_ylim(ymin - pad, ymax + pad)

    # Save
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# GROUPED BAR CHART (all metrics for each concept)
# =====================================================================
def render_grouped_bar_chart(
    concepts_data: Dict[str, pd.DataFrame],
    selected_concepts: List[str],
    figsize: Tuple[float, float],
    dpi: int,
    label_fontsize: int,
    x_label_fontsize: int,
    x_label_rotation: int,
    show_zero_line: bool,
    show_grid: bool,
    title: str,
    bg_color: str,
    text_color: str,
    cmap_name: str = "Set2",
) -> bytes:
    """Render a grouped bar chart: one cluster per concept, one bar per metric."""
    if not selected_concepts:
        return b""

    # Build a matrix: rows = concepts, cols = metrics
    metrics_to_plot = [m for m in ALL_METRICS if m != "# Connected components"]
    # (Skip components because it's integer and on a different scale)

    data = []
    for concept in selected_concepts:
        if concept not in concepts_data:
            continue
        df = concepts_data[concept]
        row = {"concept": concept}
        for metric in metrics_to_plot:
            metric_row = df[df["Metric"] == metric]
            if not metric_row.empty:
                delta = metric_row.iloc[0]["DeltaValue"]
                row[metric] = delta if delta is not None and not np.isnan(delta) else 0.0
            else:
                row[metric] = 0.0
        data.append(row)

    if not data:
        return b""

    df_plot = pd.DataFrame(data)
    n_concepts = len(df_plot)
    n_metrics = len(metrics_to_plot)
    bar_width = 0.15
    x_positions = np.arange(n_concepts)

    # Colors for each metric
    metric_colors = get_cmap_colors(cmap_name, n_metrics)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    for i, metric in enumerate(metrics_to_plot):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        values = df_plot[metric].values
        short_name = METRIC_SHORT.get(metric, metric)
        ax.bar(
            x_positions + offset, values, bar_width,
            label=short_name, color=metric_colors[i],
            edgecolor="white", linewidth=0.5, alpha=0.92,
        )

    # Zero line
    if show_zero_line:
        ax.axhline(y=0, color=text_color, linewidth=1.5, linestyle="-", alpha=0.7)

    # X-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [c.replace("_", " ") for c in df_plot["concept"]],
        fontsize=x_label_fontsize,
        rotation=x_label_rotation,
        ha="right" if x_label_rotation != 0 else "center",
        color=text_color,
    )

    ax.set_ylabel("Δ Value (After − Before)", fontsize=label_fontsize + 2,
                   color=text_color, fontweight="bold")
    ax.set_title(title, fontsize=label_fontsize + 4, fontweight="bold",
                  color=text_color, pad=14)
    ax.tick_params(axis="y", labelsize=label_fontsize, colors=text_color)

    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#888888")
    ax.set_axisbelow(True)
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(text_color)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=label_fontsize, frameon=True, edgecolor="#cccccc",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# BEFORE-VS-AFTER COMPARISON (single concept)
# =====================================================================
def render_before_after_chart(
    concept: str,
    df: pd.DataFrame,
    figsize: Tuple[float, float],
    dpi: int,
    label_fontsize: int,
    x_label_fontsize: int,
    x_label_rotation: int,
    show_grid: bool,
    title: str,
    bg_color: str,
    text_color: str,
    before_color: str = "#1f77b4",
    after_color: str = "#ff7f0e",
) -> bytes:
    """Render a Before-vs-After grouped bar chart for a single concept."""
    # Filter to standard metrics (skip components which is integer)
    metrics_to_plot = [m for m in ALL_METRICS if m != "# Connected components"]
    df_filtered = df[df["Metric"].isin(metrics_to_plot)].copy()
    if df_filtered.empty:
        return b""

    n = len(df_filtered)
    x_positions = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    before_vals = df_filtered["Before"].values
    after_vals = df_filtered["After"].values

    ax.bar(x_positions - bar_width/2, before_vals, bar_width,
           label="Before", color=before_color, edgecolor="white", linewidth=0.5, alpha=0.92)
    ax.bar(x_positions + bar_width/2, after_vals, bar_width,
           label="After", color=after_color, edgecolor="white", linewidth=0.5, alpha=0.92)

    # Value labels
    for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
        if not np.isnan(bv):
            ax.text(x_positions[i] - bar_width/2, bv + (max(before_vals) - min(before_vals)) * 0.02,
                    f"{bv:.4f}", ha="center", va="bottom", fontsize=label_fontsize - 2,
                    color=text_color, fontweight="bold")
        if not np.isnan(av):
            ax.text(x_positions[i] + bar_width/2, av + (max(after_vals) - min(after_vals)) * 0.02,
                    f"{av:.4f}", ha="center", va="bottom", fontsize=label_fontsize - 2,
                    color=text_color, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [METRIC_SHORT.get(m, m) for m in df_filtered["Metric"]],
        fontsize=x_label_fontsize, rotation=x_label_rotation,
        ha="right" if x_label_rotation != 0 else "center", color=text_color,
    )
    ax.set_ylabel("Value", fontsize=label_fontsize + 2, color=text_color, fontweight="bold")
    ax.set_title(f"{title} — {concept.replace('_', ' ')}",
                  fontsize=label_fontsize + 4, fontweight="bold", color=text_color, pad=14)
    ax.tick_params(axis="y", labelsize=label_fontsize, colors=text_color)

    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#888888")
    ax.set_axisbelow(True)
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(text_color)

    ax.legend(loc="upper right", fontsize=label_fontsize, frameon=True, edgecolor="#cccccc")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# LEGEND (concept → color swatch → role)
# =====================================================================
def render_legend(concepts: List[str], delta_values: List[float],
                  delta_densities: List[float], n_cols: int = 4) -> None:
    """Render an HTML legend with role classifications."""
    if not concepts:
        st.info("No concepts selected.")
        return

    rows_html = []
    for i, (concept, dq, dd) in enumerate(zip(concepts, delta_values, delta_densities)):
        role = classify_role(dq, dd)
        # Role color
        role_colors = {
            "Bridge": "#2ca02c", "Bridge (high-deg)": "#1a9850",
            "Hub-Anchor": "#d62728", "Sparse-Anchor": "#984ea3",
            "Anchor": "#7570b3", "Peripheral": "#999999", "Unknown": "#cccccc",
        }
        role_color = role_colors.get(role, "#cccccc")
        rows_html.append(f"""
        <div style='padding:6px 8px; border-radius:6px;
                    background-color:{role_color}22;
                    border-left:4px solid {role_color};
                    margin-bottom:6px; font-size:12px;'>
            <b style='color:#333;'>{concept.replace('_', ' ')}</b>
            <span style='display:inline-block; margin-left:8px; padding:2px 6px;
                         border-radius:3px; background-color:{role_color}; color:white;
                         font-size:10px; font-weight:bold;'>{role}</span>
            <br>
            <span style='color:#666; font-size:11px;'>
                ΔQ = <b>{dq:+.4f}</b> &nbsp;|&nbsp;
                ΔDensity = <b>{dd:+.4f}</b>
            </span>
        </div>
        """)

    per_col = (len(rows_html) + n_cols - 1) // n_cols
    cols = st.columns(n_cols)
    for ci in range(n_cols):
        with cols[ci]:
            start = ci * per_col
            end = min(start + per_col, len(rows_html))
            for html in rows_html[start:end]:
                st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# SIDEBAR
# =====================================================================
def render_sidebar(concepts_data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    with st.sidebar:
        st.header("🌉 BCSA Bar Chart Studio")
        st.caption("Bridge-Criticality Sensitivity Analysis visualizer")

        # ---- 1. Data source ----
        st.subheader("1. Data Source")
        default_dir = get_default_dataset_dir()
        data_mode = st.radio(
            "Choose data source:",
            ["Load from bcsa_datasets/ folder", "Upload CSV file(s)"],
            index=0,
        )

        if data_mode == "Load from bcsa_datasets/ folder":
            dir_path = st.text_input(
                "Directory path:",
                value=default_dir,
                help=f"Defaults to the 'bcsa_datasets' folder next to this script. "
                     f"You can also type an absolute path.",
            )
            if dir_path and os.path.isdir(dir_path):
                concepts_data = load_all_bcsa_datasets(dir_path)
                if concepts_data:
                    st.success(f"Loaded {len(concepts_data)} concept CSV(s) from `{dir_path}`.")
                else:
                    st.warning(f"No .csv files found in `{dir_path}`.")
            elif dir_path:
                st.warning(f"Directory does not exist: `{dir_path}`. "
                            f"Please create it and place BCSA CSV files inside.")

        else:  # Upload
            uploaded = st.file_uploader(
                "Upload BCSA CSV(s) (cols: Metric, Before, After, Δ)",
                type=["csv"], accept_multiple_files=True,
            )
            concepts_data = {}
            if uploaded:
                for file in uploaded:
                    try:
                        # Save to temp and parse
                        content = file.read().decode("utf-8")
                        df = load_bcsa_csv(io.StringIO(content))
                        concept = file.name[:-4]  # strip .csv
                        concepts_data[concept] = df
                    except Exception as e:
                        st.error(f"Failed to parse {file.name}: {e}")
                if concepts_data:
                    st.success(f"Loaded {len(concepts_data)} concept CSV(s).")

        if not concepts_data:
            st.info("No data loaded yet. Switch to upload mode or place CSVs in the folder.")
            return {}, {}

        # ---- 2. Concept selection ----
        st.subheader("2. Select Concepts")
        all_concepts = sorted(concepts_data.keys())
        filter_pattern = st.text_input(
            "Filter concepts (regex or substring):", value="",
            help="e.g., 'defect', 'twins', '^p', 'sintering|metal'",
        )
        try:
            if filter_pattern:
                filtered = [c for c in all_concepts if re.search(filter_pattern, c, re.IGNORECASE)]
            else:
                filtered = all_concepts
        except re.error:
            st.warning(f"Invalid regex. Showing all.")
            filtered = all_concepts

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Select All", key="select_all_btn", use_container_width=True):
                st.session_state["bcsa_selected"] = filtered
        with col_b:
            if st.button("Clear All", key="clear_all_btn", use_container_width=True):
                st.session_state["bcsa_selected"] = []
        with col_c:
            if st.button("Invert", key="invert_btn", use_container_width=True):
                current = st.session_state.get("bcsa_selected", [])
                st.session_state["bcsa_selected"] = [c for c in filtered if c not in current]

        if "bcsa_selected" not in st.session_state:
            st.session_state["bcsa_selected"] = filtered[:10]  # default to first 10

        selected = st.multiselect(
            f"Concepts to plot ({len(filtered)} shown, {len(st.session_state.get('bcsa_selected', []))} selected):",
            options=filtered,
            default=[c for c in st.session_state["bcsa_selected"] if c in filtered],
            key="bcsa_selected",
        )

        # Quick presets
        st.markdown("**Quick presets:**")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Defects", key="preset_defects", use_container_width=True):
                defects = ["twins", "stacking_faults", "lattice_distortion",
                          "defect_engineering", "fcc"]
                st.session_state["bcsa_selected"] = [c for c in all_concepts if c in defects]
                st.rerun()
        with preset_cols[1]:
            if st.button("Materials", key="preset_materials", use_container_width=True):
                materials = ["agnps", "silver", "metal"]
                st.session_state["bcsa_selected"] = [c for c in all_concepts if c in materials]
                st.rerun()
        with preset_cols[2]:
            if st.button("All", key="preset_all", use_container_width=True):
                st.session_state["bcsa_selected"] = all_concepts
                st.rerun()

        # ---- 3. Chart type & metric ----
        st.subheader("3. Chart Type & Metric")
        chart_type = st.selectbox(
            "Chart type:",
            ["Diverging Bar Chart (single metric)",
             "Grouped Bar Chart (all metrics)",
             "Before-vs-After (single concept)"],
            index=0,
        )

        # Metric selector (only for diverging chart)
        if "Diverging" in chart_type:
            metric = st.selectbox(
                "Metric to plot:",
                ALL_METRICS,
                index=0,
                format_func=lambda m: METRIC_SHORT.get(m, m),
            )

        # Color mode
        if "Diverging" in chart_type:
            color_mode = st.selectbox(
                "Color mode:",
                ["sign (green=+, red=−)", "colormap"],
                index=0,
            )
            if color_mode == "colormap":
                cmap_name = st.selectbox(
                    f"Colormap ({len(COLORMAP_NAMES)} options):",
                    options=COLORMAP_NAMES,
                    index=COLORMAP_NAMES.index("coolwarm"),
                )
                reverse_cmap = st.checkbox("Reverse colormap", value=False)
                if reverse_cmap:
                    cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name
            else:
                cmap_name = "coolwarm"  # not used in sign mode
                positive_color = st.color_picker("Positive bar color", value="#2ca02c")
                negative_color = st.color_picker("Negative bar color", value="#d62728")
        else:
            color_mode = "colormap"
            cmap_name = st.selectbox(
                f"Colormap ({len(COLORMAP_NAMES)} options):",
                options=COLORMAP_NAMES,
                index=COLORMAP_NAMES.index("Set2"),
            )
            positive_color = "#2ca02c"
            negative_color = "#d62728"

        sort_order = st.selectbox(
            "Sort order:",
            ["descending", "ascending", "absolute_desc", "alphabetical"],
            index=2,  # default: absolute_desc (most critical first)
            format_func=lambda x: {
                "descending": "descending (highest first)",
                "ascending": "ascending (lowest first)",
                "absolute_desc": "by |magnitude| (most critical first)",
                "alphabetical": "alphabetical",
            }.get(x, x),
        )

        # ---- 4. Image customization ----
        st.subheader("4. Image Customization")
        title = st.text_input(
            "Chart title:",
            value=f"BCSA — Δ {METRIC_SHORT.get(metric, '') if 'Diverging' in chart_type else 'All Metrics'} by Concept" if 'Diverging' in chart_type else "BCSA — All Metrics by Concept",
        )

        figsize_w = st.slider("Figure width (inches)", 6, 24, 14)
        figsize_h = st.slider("Figure height (inches)", 4, 16, 8)
        dpi = st.slider("DPI", 80, 400, 150)
        bar_width = st.slider("Bar width", 0.2, 1.0, 0.7)
        label_fontsize = st.slider("Value label font size", 6, 20, 10)
        x_label_fontsize = st.slider("X-axis label font size", 6, 18, 10)
        x_label_rotation = st.slider("X-axis label rotation (°)", 0, 90, 45)
        show_value_labels = st.checkbox("Show Δ value labels", value=True)
        show_zero_line = st.checkbox("Show zero line (diverging charts)", value=True)
        show_grid = st.checkbox("Show grid", value=True)
        bg_color = st.color_picker("Background color", value="#ffffff")
        text_color = st.color_picker("Text color", value="#1a1a1a")

        # ---- 5. Legend ----
        st.subheader("5. Legend")
        show_legend = st.checkbox("Show role legend", value=True)
        legend_cols = st.slider("Legend columns", 1, 6, 3)

    settings = {
        "chart_type": chart_type, "metric": metric if "Diverging" in chart_type else None,
        "color_mode": color_mode, "cmap_name": cmap_name,
        "positive_color": positive_color, "negative_color": negative_color,
        "sort_order": sort_order, "title": title,
        "figsize": (figsize_w, figsize_h), "dpi": dpi, "bar_width": bar_width,
        "label_fontsize": label_fontsize, "x_label_fontsize": x_label_fontsize,
        "x_label_rotation": x_label_rotation, "show_value_labels": show_value_labels,
        "show_zero_line": show_zero_line, "show_grid": show_grid,
        "bg_color": bg_color, "text_color": text_color,
        "show_legend": show_legend, "legend_cols": legend_cols,
        "selected_concepts": selected,
    }
    return settings, concepts_data


# =====================================================================
# MAIN APP
# =====================================================================
def main():
    st.title("🌉 BCSA Bar Chart Studio")
    st.caption(
        "Bridge-Criticality Sensitivity Analysis visualizer — "
        "diverging bar charts that handle positive AND negative modularity shifts."
    )

    # Initial load (empty until sidebar populates)
    settings, concepts_data = render_sidebar({})

    if not concepts_data:
        st.warning("⚠️ No BCSA data loaded. Use the sidebar to load CSVs.")
        st.info("""
        **Expected CSV format** (one file per concept, named e.g. `agnps.csv`):

        ```
        Metric,Before,After,Δ (After − Before)
        Modularity Q,0.0126,0.0365,+0.0239
        Avg. shortest path ⟨d⟩,1.5994,1.6024,+0.0031
        # Connected components,1,1,+0
        Avg. clustering,0.6829,0.6835,+0.0006
        Density,0.4006,0.3976,-0.0031
        ```

        Place CSV files in the `bcsa_datasets/` folder next to this script,
        or upload them via the sidebar.
        """)
        return

    selected = settings["selected_concepts"]
    if not selected:
        st.warning("⚠️ Please select at least one concept to plot.")
        return

    st.markdown(f"**Selected:** {len(selected)} / {len(concepts_data)} concepts")

    # ---- Render chart ----
    if "Diverging" in settings["chart_type"]:
        # Extract the Δ value for the chosen metric from each selected concept
        delta_values = []
        delta_densities = []  # for legend role classification
        for concept in selected:
            df = concepts_data[concept]
            metric_row = df[df["Metric"] == settings["metric"]]
            if not metric_row.empty:
                dv = metric_row.iloc[0]["DeltaValue"]
                delta_values.append(dv if dv is not None and not np.isnan(dv) else 0.0)
            else:
                delta_values.append(0.0)
            # Get density delta for role classification
            density_row = df[df["Metric"] == "Density"]
            if not density_row.empty:
                dd = density_row.iloc[0]["DeltaValue"]
                delta_densities.append(dd if dd is not None and not np.isnan(dd) else 0.0)
            else:
                delta_densities.append(0.0)

        png_bytes = render_diverging_bar_chart(
            concepts=selected,
            delta_values=delta_values,
            metric_name=settings["metric"],
            cmap_name=settings["cmap_name"],
            color_mode=settings["color_mode"],
            figsize=settings["figsize"],
            dpi=settings["dpi"],
            bar_width=settings["bar_width"],
            label_fontsize=settings["label_fontsize"],
            x_label_fontsize=settings["x_label_fontsize"],
            x_label_rotation=settings["x_label_rotation"],
            show_value_labels=settings["show_value_labels"],
            show_zero_line=settings["show_zero_line"],
            show_grid=settings["show_grid"],
            title=settings["title"],
            bg_color=settings["bg_color"],
            text_color=settings["text_color"],
            sort_order=settings["sort_order"],
            positive_color=settings["positive_color"],
            negative_color=settings["negative_color"],
        )
        st.image(png_bytes, use_container_width=True)
        st.download_button(
            "⬇️ Download PNG",
            data=png_bytes,
            file_name=f"bcsa_{settings['metric'].replace(' ', '_').replace('⟨', '').replace('⟩', '')}_chart.png",
            mime="image/png",
        )

        # ---- Legend with role classification ----
        if settings["show_legend"]:
            st.markdown("---")
            st.subheader("🎨 Concept Roles (based on Δ Modularity sign + Δ Density)")
            st.caption("Green = Bridge (ΔQ > 0), Red = Hub-Anchor (ΔQ < 0, ΔD < 0), "
                        "Purple = Sparse-Anchor (ΔQ < 0, ΔD > 0), Grey = Peripheral")
            render_legend(selected, delta_values, delta_densities, n_cols=settings["legend_cols"])

    elif "Grouped" in settings["chart_type"]:
        png_bytes = render_grouped_bar_chart(
            concepts_data=concepts_data,
            selected_concepts=selected,
            figsize=settings["figsize"],
            dpi=settings["dpi"],
            label_fontsize=settings["label_fontsize"],
            x_label_fontsize=settings["x_label_fontsize"],
            x_label_rotation=settings["x_label_rotation"],
            show_zero_line=settings["show_zero_line"],
            show_grid=settings["show_grid"],
            title=settings["title"],
            bg_color=settings["bg_color"],
            text_color=settings["text_color"],
            cmap_name=settings["cmap_name"],
        )
        st.image(png_bytes, use_container_width=True)
        st.download_button("⬇️ Download PNG", data=png_bytes,
                            file_name="bcsa_grouped_chart.png", mime="image/png")

    else:  # Before-vs-After
        if len(selected) > 1:
            st.info(f"ℹ️ Before-vs-After view shows one concept at a time. "
                    f"Showing the first selected: **{selected[0]}**. "
                    f"(To compare others, change your selection.)")
            concept = selected[0]
        else:
            concept = selected[0]

        df = concepts_data[concept]
        png_bytes = render_before_after_chart(
            concept=concept, df=df,
            figsize=settings["figsize"], dpi=settings["dpi"],
            label_fontsize=settings["label_fontsize"],
            x_label_fontsize=settings["x_label_fontsize"],
            x_label_rotation=settings["x_label_rotation"],
            show_grid=settings["show_grid"],
            title=settings["title"],
            bg_color=settings["bg_color"], text_color=settings["text_color"],
        )
        st.image(png_bytes, use_container_width=True)
        st.download_button("⬇️ Download PNG", data=png_bytes,
                            file_name=f"bcsa_before_after_{concept}.png", mime="image/png")

    # ---- Data table ----
    st.markdown("---")
    st.subheader("📋 Selected Concepts (Data Table)")
    table_rows = []
    for concept in selected:
        df = concepts_data[concept]
        # Get modularity and density deltas
        mod_row = df[df["Metric"] == "Modularity Q"]
        den_row = df[df["Metric"] == "Density"]
        comp_row = df[df["Metric"] == "# Connected components"]
        dq = mod_row.iloc[0]["DeltaValue"] if not mod_row.empty else None
        dd = den_row.iloc[0]["DeltaValue"] if not den_row.empty else None
        dc = comp_row.iloc[0]["DeltaValue"] if not comp_row.empty else None
        role = classify_role(dq, dd)
        table_rows.append({
            "concept": concept,
            "Δ Modularity": f"{dq:+.4f}" if dq is not None and not np.isnan(dq) else "N/A",
            "Δ Density": f"{dd:+.4f}" if dd is not None and not np.isnan(dd) else "N/A",
            "Δ Components": f"{int(dc):+d}" if dc is not None and not np.isnan(dc) else "N/A",
            "role": role,
            "fragmented": "⚠️ YES" if (dc is not None and not np.isnan(dc) and dc > 0) else "No",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ---- Tips ----
    with st.expander("💡 How to interpret the chart"):
        st.markdown("""
        **Diverging Bar Chart (Δ Modularity):**
        - **Positive bars (green):** The concept is a **Bridge / Boundary-spanner**.
          Removing it sharpens community structure.
        - **Negative bars (red):** The concept is a **Community Anchor / Hub**.
          Removing it weakens community structure.
        - **Bars near zero:** The concept is **Peripheral** — removal has little effect.

        **Role classifications in the legend:**
        | Role | Δ Modularity | Δ Density | Meaning |
        |---|---|---|---|
        | Bridge (high-deg) | + | − | High-degree boundary-spanner (e.g., `defect_engineering`) |
        | Bridge | + | ~0 | Sparse boundary-spanner |
        | Hub-Anchor | − | − | High-degree community pillar (e.g., `fcc`) |
        | Sparse-Anchor | − | + | Low-degree weak-tie anchor (e.g., `agnps`, `lattice_distortion`) |
        | Anchor | − | ~0 | Generic community anchor |
        | Peripheral | ~0 | ~0 | Removal has minimal effect |

        **Sort order tip:** Use "by |magnitude|" to put the most structurally
        critical concepts first (largest absolute Δ value = biggest impact).
        """)


if __name__ == "__main__":
    main()
