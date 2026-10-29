#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIS Bar Chart Studio (Enhanced + Fixed Polar Chart)
====================================================
A focused Streamlit app for visualizing Sustainability Impact Score (SIS) —
or any node-level score — as customizable 2D, 3D, and Polar/Radial bar charts.

Features
--------
- FIXED: Polar / Radial Bar Chart now renders as a TRUE CIRCLE
  (square figure dimensions + proper angular axis + tip labels)
- 50+ colormaps (matplotlib + plotly)
- 2D bar chart (matplotlib) with score labels on top of each bar
- 3D bar chart (plotly mesh) with score labels floating above bars
- Long concept names handled 4 ways: abbreviated, wrapped, rotated, or
  polar (laid out radially — no overlap possible)
- Interactive concept multi-selector: pick exactly which concepts to plot
- Full image customization: DPI, figure size, font sizes, bar width, label rotation
- Legend panel: color swatch + N_i ID + full concept name
- Downloads: PNG (2D), HTML (3D/Polar), CSV (scores)
- Load data from local directory (os.listdir) or upload multiple CSVs

Run:
  streamlit run sis_bar_chart_studio.py
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
import plotly.graph_objects as go
import plotly.express as px
import io
import re
import textwrap
import os
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
    page_title="SIS Bar Chart Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 50+ COLORMAP REGISTRY
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
        cmap = cm.get_cmap("viridis", max(n, 1))
        return [mcolors.to_hex(cmap(i)) for i in range(n)]


# =====================================================================
# MANUSCRIPT NODE-ID MAPPING
# =====================================================================
NODE_ID_MAP = {
    "metal": "N1", "attention": "N2", "electrical_conductivity": "N3",
    "grain_boundary": "N4", "thermal_budget": "N5", "mechanical_yield": "N6",
    "oxidation": "N7", "micro_voids": "N8", "pressureless_sintering": "N9",
    "additive manufacturing": "N10", "fcc": "N11", "machine learning": "N12",
    "circular_economy": "N13", "defect_engineering": "N14",
    "energy_efficiency": "N15", "agnps": "N16", "vocs": "N17",
    "stacking_faults": "N18", "atomic_diffusion": "N19",
    "thermal_conductivity": "N20", "net_zero": "N21",
    "lifecycle_assessment": "N22", "waste_minimization": "N23",
    "lattice_distortion": "N24", "nucleation": "N25",
    "diffusion_highways": "N26", "sintering_process": "N27", "twins": "N28",
    "silver": "N29", "mass_transport": "N30", "polymer_substrate": "N31",
    "high entropy alloys": "N32", "heas": "N33", "data-driven": "N34",
    "power_module_assembly": "N35", "residual_stress": "N36",
    "cleaner_production": "N37", "phase transformation": "N38",
    "high-entropy alloys": "N39", "grain size": "N40",
    "sic": "N41", "greenhouse_gas": "N42", "e_waste": "N43", "dendrite": "N44",
}
ANCHORS = {"net_zero", "cleaner_production", "circular_economy"}


def short_label(concept: str, max_len: int = 12) -> str:
    concept = concept.replace("_", " ")
    if len(concept) <= max_len:
        return concept
    parts = concept.split(" ")
    if len(parts) >= 2:
        return parts[0][:max_len - 3] + "…"
    return concept[:max_len - 1] + "…"


def wrap_label(concept: str, width: int = 10) -> str:
    return "\n".join(textwrap.wrap(concept.replace("_", " "), width=width)) or concept


# =====================================================================
# PRELOADED SIS DATA
# =====================================================================
DEFAULT_SIS_CSV = """concept,SIS,n_anchors_reachable
metal,0.7718,3
circular_economy,0.7707,3
attention,0.7633,3
grain_boundary,0.7513,3
oxidation,0.7497,3
micro_voids,0.7485,3
electrical_conductivity,0.7457,3
thermal_budget,0.744,3
cleaner_production,0.7424,3
mechanical_yield,0.7421,3
pressureless_sintering,0.7289,3
net_zero,0.7253,3
fcc,0.7213,3
additive manufacturing,0.7213,3
machine learning,0.7107,3
agnps,0.6942,3
defect_engineering,0.6942,3
vocs,0.6919,3
energy_efficiency,0.6717,3
atomic_diffusion,0.6703,3
silver,0.6656,3
nucleation,0.6656,3
stacking_faults,0.6507,3
thermal_conductivity,0.6462,3
lifecycle_assessment,0.6309,3
high entropy alloys,0.6309,3
sintering_process,0.6035,3
waste_minimization,0.6035,3
dendrite,0.6035,3
heas,0.5853,3
twins,0.5853,3
data-driven,0.5656,3
phase transformation,0.5627,3
polymer_substrate,0.5627,3
diffusion_highways,0.5488,3
residual_stress,0.5356,3
high-entropy alloys,0.5297,3
lattice_distortion,0.4958,3
e_waste,0.4958,3
mass_transport,0.4958,3
greenhouse_gas,0.4925,3
grain size,0.4874,3
sic,0.485,3
power_module_assembly,0.4701,3
"""


def load_default_data() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(DEFAULT_SIS_CSV))
    df["node_id"] = df["concept"].map(NODE_ID_MAP).fillna("?")
    df["is_anchor"] = df["concept"].isin(ANCHORS)
    return df.sort_values("SIS", ascending=False).reset_index(drop=True)


# =====================================================================
# 2D BAR CHART (matplotlib)
# =====================================================================
def render_2d_bar_chart(df: pd.DataFrame, cmap_name: str, figsize: Tuple[float, float], dpi: int,
                        bar_width: float, label_position: str, label_rotation: int, label_fontsize: int,
                        x_label_mode: str, x_label_fontsize: int, x_label_rotation: int,
                        show_node_ids: bool, show_value_labels: bool, highlight_anchors: bool,
                        title: str, show_grid: bool, bg_color: str, text_color: str, sort_order: str) -> bytes:
    if df.empty:
        return b""

    if sort_order == "descending":
        df = df.sort_values("SIS", ascending=False).reset_index(drop=True)
    elif sort_order == "ascending":
        df = df.sort_values("SIS", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("concept", ascending=True).reset_index(drop=True)

    n = len(df)
    colors = get_cmap_colors(cmap_name, n)
    sis_values = df["SIS"].values
    concepts = df["concept"].tolist()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    x_positions = np.arange(n)
    bars = ax.bar(x_positions, sis_values, width=bar_width, color=colors,
                  edgecolor=text_color if bg_color != "white" else "#333333", linewidth=0.6, alpha=0.95)

    if highlight_anchors:
        for i, is_anchor in enumerate(df["is_anchor"]):
            if is_anchor:
                bars[i].set_edgecolor("#FFD700")
                bars[i].set_linewidth(2.5)

    if show_value_labels:
        for i, (bar, val) in enumerate(zip(bars, sis_values)):
            if label_position == "top":
                y = val + (max(sis_values) - min(sis_values)) * 0.012 + 0.005
                va, color, weight = "bottom", text_color, "bold"
            else:
                y = val / 2
                va, weight = "center", "bold"
                r, g, b, _ = mcolors.to_rgba(colors[i])
                color = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:.3f}",
                    ha="center", va=va, fontsize=label_fontsize, fontweight=weight,
                    color=color, rotation=label_rotation)

    x_labels = []
    for concept in concepts:
        if x_label_mode == "full":
            x_labels.append(concept.replace("_", " "))
        elif x_label_mode == "abbreviated":
            x_labels.append(short_label(concept, max_len=12))
        elif x_label_mode == "wrapped":
            x_labels.append(wrap_label(concept, width=10))
        else:
            x_labels.append(concept.replace("_", " "))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=x_label_fontsize, rotation=x_label_rotation,
                       ha="right" if x_label_rotation != 0 else "center", color=text_color)

    if show_node_ids:
        node_ids = df["node_id"].tolist()
        secax = ax.secondary_xaxis(-0.18)
        secax.set_xticks(x_positions)
        secax.set_xticklabels(node_ids, fontsize=x_label_fontsize - 2, color="#666666", fontweight="bold")
        secax.tick_params(length=0)
        secax.spines["bottom"].set_visible(False)

    ax.set_ylabel("SIS Score", fontsize=label_fontsize + 2, color=text_color, fontweight="bold")
    ax.set_title(title, fontsize=label_fontsize + 4, fontweight="bold", color=text_color, pad=14)
    ax.tick_params(axis="y", labelsize=label_fontsize, colors=text_color)
    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#888888")
    ax.set_axisbelow(True)
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["left", "bottom"]:
        ax.spines[spine_name].set_color(text_color)

    if label_position == "top" and show_value_labels:
        ax.set_ylim(0, max(sis_values) * 1.08)
    else:
        ax.set_ylim(0, max(sis_values) * 1.02)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# 3D BAR CHART (plotly)
# =====================================================================
def render_3d_bar_chart(df: pd.DataFrame, cmap_name: str, width: int, height: int,
                        show_value_labels: bool, show_node_ids: bool, highlight_anchors: bool,
                        title: str, sort_order: str, label_fontsize: int, bg_color: str,
                        text_color: str, grid_size: Optional[int] = None) -> go.Figure:
    if df.empty:
        return go.Figure()

    if sort_order == "descending":
        df = df.sort_values("SIS", ascending=False).reset_index(drop=True)
    elif sort_order == "ascending":
        df = df.sort_values("SIS", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("concept", ascending=True).reset_index(drop=True)

    n = len(df)
    colors = get_cmap_colors(cmap_name, n)
    if grid_size is None or grid_size <= 0:
        grid_size = int(np.ceil(np.sqrt(n)))
    cols, rows = grid_size, int(np.ceil(n / cols))

    fig = go.Figure()
    for i in range(n):
        x, y, z = i % cols, i // cols, df["SIS"].iloc[i]
        hw = 0.4
        verts_x = [x - hw, x + hw, x + hw, x - hw, x - hw, x + hw, x + hw, x - hw]
        verts_y = [y - hw, y - hw, y + hw, y + hw, y - hw, y - hw, y + hw, y + hw]
        verts_z = [0, 0, 0, 0, z, z, z, z]
        I = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 2]
        J = [1, 2, 2, 3, 3, 0, 0, 1, 4, 5, 5, 6]
        K = [2, 3, 3, 0, 0, 1, 1, 2, 5, 4, 4, 7]

        fig.add_trace(go.Mesh3d(x=verts_x, y=verts_y, z=verts_z, i=I, j=J, k=K,
                                color=colors[i], opacity=0.95,
                                hovertemplate=f"<b>{df['concept'].iloc[i]}</b> ({df['node_id'].iloc[i]})<br>SIS: {z:.4f}<extra></extra>"))
        if show_value_labels:
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z + 0.04], mode="text",
                                       text=[f"{z:.3f}"], textfont=dict(size=label_fontsize, color=text_color, family="Arial Black"),
                                       textposition="top center", showlegend=False, hoverinfo="skip"))
        if show_node_ids:
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[-0.03], mode="text",
                                       text=[df["node_id"].iloc[i]], textfont=dict(size=label_fontsize - 2, color="#666666"),
                                       textposition="bottom center", showlegend=False, hoverinfo="skip"))
        if highlight_anchors and df["is_anchor"].iloc[i]:
            edge_x = [x - hw, x + hw, x + hw, x - hw, x - hw, x - hw, x + hw, x + hw, x - hw, x - hw, x + hw, x + hw, x - hw, x + hw, x + hw, x - hw]
            edge_y = [y - hw, y - hw, y + hw, y + hw, y - hw, y - hw, y - hw, y - hw, y + hw, y + hw, y + hw, y + hw, y - hw, y - hw, y + hw, y + hw]
            edge_z = [0, 0, 0, 0, 0, z, z, z, z, 0, 0, z, z, z, z, z]
            fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                                       line=dict(color="#FFD700", width=4), showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=text_color)),
        scene=dict(xaxis=dict(showticklabels=False, showgrid=False, backgroundcolor=bg_color),
                   yaxis=dict(showticklabels=False, showgrid=False, backgroundcolor=bg_color),
                   zaxis=dict(title="SIS Score", title_font=dict(color=text_color), tickfont=dict(color=text_color),
                              backgroundcolor=bg_color, gridcolor="rgba(150,150,150,0.3)"),
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)), aspectmode="data"),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color),
        width=width, height=height, showlegend=False, margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# =====================================================================
# POLAR / RADIAL BAR CHART (plotly) — FIXED FOR TRUE CIRCULAR APPEARANCE
# =====================================================================
def render_polar_bar_chart(
    df: pd.DataFrame,
    cmap_name: str,
    highlight_anchors: bool,
    title: str,
    bg_color: str,
    text_color: str,
    sort_order: str,
    chart_size: int = 900,           # square dimensions (width = height = chart_size)
    bar_angular_width: float = 0.85, # fraction of slot each bar fills (0-1)
    angular_rotation: int = 90,       # start angle: 90 = top (12 o'clock)
    angular_direction: str = "clockwise",
    show_value_labels: bool = True,
    show_node_ids: bool = True,
    label_fontsize: int = 12,
    radial_range_padding: float = 1.20,
    inner_radius_offset: float = 0.0,
    # ---- NEW: Radial axis (0-to-1) customization ----
    radial_axis_show: bool = True,
    radial_axis_show_ticks: bool = True,
    radial_axis_n_ticks: int = 6,
    radial_axis_fontsize: int = 10,
    radial_axis_decimals: int = 2,
    radial_axis_color: str = "#666666",
    radial_axis_show_grid: bool = True,
    radial_axis_grid_color: str = "rgba(150,150,150,0.3)",
    # ---- NEW: Concept label (angular axis) customization ----
    concept_label_show: bool = True,
    concept_label_fontsize: int = 10,
    concept_label_color: str = "#1a1a1a",
    concept_label_angle: int = 0,        # 0 = auto-rotated to read outward, else fixed degrees
    concept_label_show_grid: bool = True,
    concept_label_grid_color: str = "rgba(150,150,150,0.3)",
    # ---- NEW: N_i node ID label customization ----
    node_id_fontsize: int = 10,
    node_id_color: str = "#666666",
) -> go.Figure:
    """
    Render a TRULY CIRCULAR polar bar chart.

    Key fixes vs. the previous version:
      1. Force width == height (= chart_size) so the polar layout is square.
      2. Tight zero margins so the circle fills the canvas.
      3. Use proper angular axis: rotation=90 starts at top,
         direction="clockwise" goes clockwise.
      4. Drop the buggy `period=n` setting (was compressing the axis).
      5. Set `width=360/n * bar_angular_width` on each bar so the bars fill
         the full 360° with small gaps, instead of leaving the circle
         looking "incomplete".
      6. Add a separate Scatterpolar trace to label SIS scores at the tips.
      7. Add a second Scatterpolar trace (at r=0) to show N_i IDs at the
         center of each bar's angular slot.
    """
    if df.empty:
        return go.Figure()

    # Sort
    if sort_order == "descending":
        df = df.sort_values("SIS", ascending=False).reset_index(drop=True)
    elif sort_order == "ascending":
        df = df.sort_values("SIS", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("concept", ascending=True).reset_index(drop=True)

    n = len(df)
    colors = get_cmap_colors(cmap_name, n)
    sis_values = df["SIS"].tolist()
    concepts = df["concept"].apply(lambda x: x.replace("_", " ")).tolist()
    node_ids = df["node_id"].tolist()
    is_anchors = df["is_anchor"].tolist()

    # Angular slot for each bar (degrees)
    slot = 360.0 / n
    bar_width_deg = slot * bar_angular_width

    # Theta positions: center of each bar's slot
    # We pass theta as category labels (strings), so Plotly auto-distributes.
    # But to control bar width, we use numeric theta and explicit `width`.
    theta_numeric = [i * slot + slot / 2 for i in range(n)]  # center of each slot

    # ---- 1. Bar trace (the actual radial bars) ----
    bar_marker_line_color = [
        "#FFD700" if is_a else "rgba(255,255,255,0.4)"
        for is_a in is_anchors
    ]
    bar_marker_line_width = [
        3 if is_a else 0.8
        for is_a in is_anchors
    ]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=sis_values,
        theta=theta_numeric,
        width=[bar_width_deg] * n,
        marker=dict(
            color=colors,
            line=dict(
                color=bar_marker_line_color,
                width=bar_marker_line_width,
            ),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "SIS: %{r:.4f}<extra></extra>"
        ),
        customdata=list(zip(concepts, node_ids)),
        name="SIS",
        showlegend=False,
    ))

    # ---- 2. SIS score labels at the tip of each bar ----
    if show_value_labels:
        label_r = [v * 1.02 + 0.005 for v in sis_values]  # just past the tip
        fig.add_trace(go.Scatterpolar(
            r=label_r,
            theta=theta_numeric,
            text=[f"{v:.3f}" for v in sis_values],
            mode="text",
            textfont=dict(size=label_fontsize, color=text_color, family="Arial Black"),
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- 3. N_i node IDs at the inner edge of each bar ----
    if show_node_ids:
        # Place N_i just inside the inner radius (small positive offset)
        # If inner_radius_offset=0, we put it at r ≈ 0.02 (near center)
        nid_r = [max(v * 0.05, 0.02) for v in sis_values]
        # If user provided a positive inner_radius_offset, lift the labels
        if inner_radius_offset > 0:
            nid_r = [inner_radius_offset + v * 0.05 for v in sis_values]
        fig.add_trace(go.Scatterpolar(
            r=nid_r,
            theta=theta_numeric,
            text=node_ids,
            mode="text",
            textfont=dict(size=node_id_fontsize, color=node_id_color, family="Arial Bold"),
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip",
        ))

    # ---- 4. Concept name labels at the outer ring ----
    # We use the angular axis tick labels for this — see layout below.
    # The ticks at theta_numeric will display the concept names.

    # ---- Layout: force TRULY CIRCULAR rendering ----
    max_sis = max(sis_values) if sis_values else 1.0
    radial_max = max_sis * radial_range_padding

    # ---- Build radial axis tick values ----
    # User controls the number of ticks (e.g., 6 → 0, 0.2, 0.4, 0.6, 0.8, 1.0)
    # We divide [0, radial_max] into n_ticks evenly spaced values.
    if radial_axis_n_ticks < 2:
        radial_axis_n_ticks = 2
    radial_tick_vals = np.linspace(0, radial_max, radial_axis_n_ticks).tolist()
    # Tick labels: format with the user's chosen decimal places
    radial_tick_text = [f"{v:.{radial_axis_decimals}f}" for v in radial_tick_vals]

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=text_color)),
        # ★ THE KEY FIX: square dimensions + tight margins → circle, not ellipse
        width=chart_size,
        height=chart_size,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        showlegend=False,
        polar=dict(
            # The polar domain fills the full canvas (0..1 in both axes)
            domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
            bgcolor=bg_color,
            radialaxis=dict(
                visible=radial_axis_show,
                range=[0, radial_max],
                showticklabels=radial_axis_show and radial_axis_show_ticks,
                ticks="",
                # ★ NEW: user-controlled grid (0-to-1 lines)
                showgrid=radial_axis_show_grid,
                gridcolor=radial_axis_grid_color,
                gridwidth=1,
                # ★ NEW: tick font + color
                tickfont=dict(color=radial_axis_color, size=radial_axis_fontsize),
                # ★ NEW: user-controlled tick count + format
                tickmode="array",
                tickvals=radial_tick_vals,
                ticktext=radial_tick_text,
                # Force the radial axis to occupy equal angular space
                angle=90 + (0 if angular_direction == "clockwise" else 180),
                tickangle=90,
            ),
            angularaxis=dict(
                # ★ Start at top (rotation=90), go clockwise
                rotation=angular_rotation,
                direction=angular_direction,
                # ★ period=1 (default) — do NOT set period=n, that compresses axis
                period=1,
                visible=True,
                # ★ NEW: user-controlled concept labels (angular axis ticks)
                showticklabels=concept_label_show,
                # ★ NEW: user-controlled grid for angular axis
                showgrid=concept_label_show_grid,
                gridcolor=concept_label_grid_color,
                gridwidth=1,
                # ★ NEW: user-controlled concept label font + color
                tickfont=dict(color=concept_label_color, size=concept_label_fontsize),
                # ★ NEW: user-controlled label rotation
                # Plotly's tickangle only accepts numbers; omit it when user wants auto (0)
                **({"tickangle": concept_label_angle} if concept_label_angle != 0 else {}),
                # Map numeric thetas to concept names via tickvals/ticktext
                tickmode="array",
                tickvals=theta_numeric,
                ticktext=concepts,
            ),
            # ★ Critical: enforce equal aspect (square polar)
            # In Plotly, polar charts ARE square when width=height and domain
            # is symmetric. The settings above ensure this.
        ),
    )

    return fig


# =====================================================================
# LEGEND
# =====================================================================
def render_legend(df: pd.DataFrame, cmap_name: str, n_cols: int = 4) -> None:
    if df.empty:
        st.info("No concepts selected to display in legend.")
        return

    colors = get_cmap_colors(cmap_name, len(df))
    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        color = colors[i]
        concept = row["concept"]
        nid = row["node_id"]
        sis = row["SIS"]
        anchor_badge = " ⭐" if row["is_anchor"] else ""
        rows_html.append(f"""
        <div style='padding:6px 8px; border-radius:6px; background-color:{color}22;
                    border-left:4px solid {color}; margin-bottom:6px; font-size:12px;'>
            <span style='display:inline-block; width:14px; height:14px; background-color:{color};
                         border-radius:3px; margin-right:6px; vertical-align:middle;'></span>
            <b style='color:#333;'>{nid}</b> <span style='color:#666;'>{concept.replace('_', ' ')}</span>{anchor_badge}
            <span style='float:right; color:#888; font-family:monospace;'>{sis:.3f}</span>
        </div>""")

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
def render_sidebar(df_all: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    with st.sidebar:
        st.header("📊 SIS Bar Chart Studio")
        st.caption("Customizable 2D / 3D / Polar bar charts for SIS")

        # ---- 1. Data source ----
        st.subheader("1. Data Source")
        data_mode = st.radio("Choose data source:", ["Use preloaded SIS results", "Upload CSV file(s)", "Load from local directory"], index=0)

        if data_mode == "Upload CSV file(s)":
            uploaded = st.file_uploader("Upload CSV(s) (must contain 'concept' and 'SIS')", type=["csv"], accept_multiple_files=True)
            if uploaded:
                dfs = []
                for file in uploaded:
                    try:
                        df_temp = pd.read_csv(file)
                        if "concept" in df_temp.columns and "SIS" in df_temp.columns:
                            dfs.append(df_temp)
                    except Exception:
                        pass
                if dfs:
                    df_all = pd.concat(dfs, ignore_index=True)
                    df_all["node_id"] = df_all["concept"].map(NODE_ID_MAP).fillna("?")
                    df_all["is_anchor"] = df_all["concept"].isin(ANCHORS)
                    df_all = df_all.sort_values("SIS", ascending=False).drop_duplicates(subset=["concept"]).reset_index(drop=True)
                    st.success(f"Loaded {len(df_all)} unique concepts from {len(uploaded)} file(s).")

        elif data_mode == "Load from local directory":
            dir_path = st.text_input("Directory path (local):", value="./datasets")
            if os.path.isdir(dir_path):
                csv_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.csv')]
                if csv_files:
                    selected_file = st.selectbox("Select CSV file:", csv_files)
                    if selected_file:
                        try:
                            full_path = os.path.join(dir_path, selected_file)
                            df_temp = pd.read_csv(full_path)
                            if "concept" in df_temp.columns and "SIS" in df_temp.columns:
                                df_all = df_temp.copy()
                                df_all["node_id"] = df_all["concept"].map(NODE_ID_MAP).fillna("?")
                                df_all["is_anchor"] = df_all["concept"].isin(ANCHORS)
                                df_all = df_all.sort_values("SIS", ascending=False).reset_index(drop=True)
                                st.success(f"Loaded {len(df_all)} concepts from {selected_file}.")
                            else:
                                st.error("CSV must have 'concept' and 'SIS' columns.")
                        except Exception as e:
                            st.error(f"Failed to load: {e}")
                else:
                    st.warning(f"No CSV files found in {dir_path}")
            else:
                st.warning(f"Directory does not exist: {dir_path}")

        # ---- 2. Concept selection ----
        st.subheader("2. Select Concepts")
        all_concepts = df_all["concept"].tolist()
        filter_pattern = st.text_input("Filter concepts (regex or substring):", value="", help="e.g., 'twins', 'defect|twins', '^p'")
        try:
            filtered = [c for c in all_concepts if re.search(filter_pattern, c, re.IGNORECASE)] if filter_pattern else all_concepts
        except re.error:
            st.warning(f"Invalid regex: {filter_pattern}. Showing all.")
            filtered = all_concepts

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Select All", key="select_all_btn", use_container_width=True):
                st.session_state["selected_concepts"] = filtered
        with col_b:
            if st.button("Clear All", key="clear_all_btn", use_container_width=True):
                st.session_state["selected_concepts"] = []
        with col_c:
            if st.button("Invert", key="invert_btn", use_container_width=True):
                current = st.session_state.get("selected_concepts", [])
                st.session_state["selected_concepts"] = [c for c in filtered if c not in current]

        if "selected_concepts" not in st.session_state:
            st.session_state["selected_concepts"] = df_all.head(15)["concept"].tolist()

        selected = st.multiselect(f"Concepts to plot ({len(filtered)} shown, {len(st.session_state['selected_concepts'])} selected):",
                                  options=filtered, default=[c for c in st.session_state["selected_concepts"] if c in filtered], key="selected_concepts")

        st.markdown("**Quick presets:**")
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("Top 10", key="preset_top10"):
                st.session_state["selected_concepts"] = df_all.head(10)["concept"].tolist()
                st.rerun()
        with preset_cols[1]:
            if st.button("Top 20", key="preset_top20"):
                st.session_state["selected_concepts"] = df_all.head(20)["concept"].tolist()
                st.rerun()
        with preset_cols[2]:
            if st.button("Anchors", key="preset_anchors"):
                st.session_state["selected_concepts"] = [c for c in all_concepts if c in ANCHORS]
                st.rerun()
        with preset_cols[3]:
            if st.button("Defects only", key="preset_defects"):
                defects = ["twins", "stacking_faults", "lattice_distortion", "defect_engineering", "diffusion_highways"]
                st.session_state["selected_concepts"] = [c for c in all_concepts if c in defects]
                st.rerun()

        # ---- 3. Chart type ----
        st.subheader("3. Chart Type & Colormap")
        chart_type = st.selectbox("Chart type:", ["2D Bar Chart (matplotlib)", "3D Bar Chart (plotly)", "Polar / Radial Bar Chart (plotly) ✨"], index=2)
        cmap_name = st.selectbox(f"Colormap ({len(COLORMAP_NAMES)} options):", options=COLORMAP_NAMES, index=COLORMAP_NAMES.index("inferno"))
        reverse_cmap = st.checkbox("Reverse colormap", value=False)
        if reverse_cmap:
            cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name
        sort_order = st.selectbox("Sort order:", ["descending", "ascending", "alphabetical"], index=0)

        # ---- 4. Image customization ----
        st.subheader("4. Image Customization")
        title = st.text_input("Chart title:", value="Sustainability Impact Score (SIS) by Concept")

        if "2D" in chart_type:
            figsize_w = st.slider("Figure width (inches)", 6, 24, 14)
            figsize_h = st.slider("Figure height (inches)", 4, 16, 8)
            dpi = st.slider("DPI", 80, 400, 150)
            bar_width = st.slider("Bar width", 0.2, 1.0, 0.7)
        elif "Polar" in chart_type:
            # ★ Polar-specific controls (forces square for true circle)
            st.markdown("**Polar chart controls:**")
            st.caption("Width and height are forced equal to ensure a perfect circle.")
            polar_size = st.slider("Chart size (px, square)", 500, 1600, 900, step=50,
                                    help="Larger = bigger circle. Width = height automatically.")
            bar_angular_width = st.slider("Bar angular width (fraction)", 0.2, 1.0, 0.85,
                                            help="1.0 = bars touch, 0.85 = small gaps, 0.5 = thin spokes")
            angular_rotation = st.slider("Start angle (°)", 0, 359, 90,
                                          help="90 = top (12 o'clock), 0 = right (3 o'clock)")
            angular_direction = st.selectbox("Direction:", ["clockwise", "counterclockwise"], index=0)
            radial_padding = st.slider("Radial range padding", 1.05, 1.50, 1.20,
                                          help="Multiplier on max SIS to give tip labels room")

            # ---- NEW: Radial axis (0-to-1) customization ----
            with st.expander("Radial axis (0-to-1 ring) customization", expanded=False):
                radial_axis_show = st.checkbox("Show radial axis", value=True, key="ra_show")
                radial_axis_show_ticks = st.checkbox("Show tick labels (0, 0.2, 0.4...)", value=True, key="ra_ticks")
                radial_axis_n_ticks = st.slider("Number of radial ticks", 2, 20, 6,
                                                  help="6 → 0.00, 0.20, 0.40, 0.60, 0.80, 1.00 (if max=1)",
                                                  key="ra_n_ticks")
                radial_axis_decimals = st.slider("Tick label decimals", 0, 4, 2,
                                                   help="2 → 0.00, 1 → 0.0, 0 → 0",
                                                   key="ra_decimals")
                radial_axis_fontsize = st.slider("Radial tick font size", 6, 24, 10, key="ra_fs")
                radial_axis_color = st.color_picker("Radial tick color", value="#666666", key="ra_color")
                radial_axis_show_grid = st.checkbox("Show radial grid lines (concentric circles)", value=True, key="ra_grid")
                radial_axis_grid_color = st.color_picker("Radial grid color", value="#9c9c9c", key="ra_grid_color")

            # ---- NEW: Concept label (angular axis) customization ----
            with st.expander("Concept label (angular ring) customization", expanded=False):
                concept_label_show = st.checkbox("Show concept names around the outer ring", value=True, key="cl_show")
                concept_label_fontsize = st.slider("Concept label font size", 6, 24, 10, key="cl_fs")
                concept_label_color = st.color_picker("Concept label color", value="#1a1a1a", key="cl_color")
                concept_label_angle = st.slider("Concept label rotation (°, 0=auto)", 0, 360, 0,
                                                  help="0 = auto-rotate to read outward; 90 = vertical; 180 = upside down",
                                                  key="cl_angle")
                concept_label_show_grid = st.checkbox("Show angular grid lines (radial spokes)", value=True, key="cl_grid")
                concept_label_grid_color = st.color_picker("Angular grid color", value="#9c9c9c", key="cl_grid_color")

            # ---- NEW: N_i node ID label customization ----
            with st.expander("N_i node ID label customization", expanded=False):
                node_id_fontsize = st.slider("N_i label font size", 6, 24, 10, key="nid_fs")
                node_id_color = st.color_picker("N_i label color", value="#666666", key="nid_color")
        else:
            width_3d = st.slider("Chart width (px)", 600, 1600, 1000)
            height_3d = st.slider("Chart height (px)", 500, 1400, 800)
            grid_size_3d = st.slider("3D grid columns (0=auto)", 0, 12, 0)

        label_fontsize = st.slider("Value label font size (SIS score on bar tip)", 6, 20, 12)
        x_label_fontsize = st.slider("X-axis label font size (2D only)", 6, 18, 9)
        show_value_labels = st.checkbox("Show SIS score on bar", value=True)
        show_node_ids = st.checkbox("Show N_i node IDs", value=True)
        highlight_anchors = st.checkbox("Highlight anchor nodes (gold outline)", value=True)

        if "2D" in chart_type:
            label_position = st.selectbox("Label position:", ["top (outside bar)", "center (inside bar)"], index=0).split(" ")[0]
            label_rotation = st.slider("Value label rotation (°)", 0, 90, 0)
            x_label_mode = st.selectbox("Long-name handling:", ["full", "abbreviated", "wrapped", "rotated"], index=3)
            x_label_rotation = st.slider("X-axis label rotation (°)", 0, 90, 45 if x_label_mode == "rotated" else 0)
        else:
            label_position, label_rotation, x_label_mode, x_label_rotation = "top", 0, "full", 0

        bg_color = st.color_picker("Background color", value="#ffffff")
        text_color = st.color_picker("Text color", value="#1a1a1a")
        show_grid = st.checkbox("Show grid (2D only)", value=True)

        # ---- 5. Legend ----
        st.subheader("5. Legend")
        show_legend = st.checkbox("Show concept legend", value=True)
        legend_cols = st.slider("Legend columns", 1, 6, 4)

    settings = {
        "data_mode": data_mode, "selected_concepts": selected, "chart_type": chart_type, "cmap_name": cmap_name,
        "sort_order": sort_order, "title": title, "label_fontsize": label_fontsize, "x_label_fontsize": x_label_fontsize,
        "show_value_labels": show_value_labels, "show_node_ids": show_node_ids, "highlight_anchors": highlight_anchors,
        "bg_color": bg_color, "text_color": text_color, "show_grid": show_grid, "show_legend": show_legend, "legend_cols": legend_cols,
    }
    if "2D" in chart_type:
        settings.update({"figsize": (figsize_w, figsize_h), "dpi": dpi, "bar_width": bar_width,
                         "label_position": label_position, "label_rotation": label_rotation,
                         "x_label_mode": x_label_mode, "x_label_rotation": x_label_rotation})
    elif "Polar" in chart_type:
        settings.update({
            "polar_size": polar_size,
            "bar_angular_width": bar_angular_width,
            "angular_rotation": angular_rotation,
            "angular_direction": angular_direction,
            "radial_padding": radial_padding,
            # ★ NEW: radial axis customization
            "radial_axis_show": radial_axis_show,
            "radial_axis_show_ticks": radial_axis_show_ticks,
            "radial_axis_n_ticks": radial_axis_n_ticks,
            "radial_axis_fontsize": radial_axis_fontsize,
            "radial_axis_decimals": radial_axis_decimals,
            "radial_axis_color": radial_axis_color,
            "radial_axis_show_grid": radial_axis_show_grid,
            "radial_axis_grid_color": radial_axis_grid_color,
            # ★ NEW: concept label customization
            "concept_label_show": concept_label_show,
            "concept_label_fontsize": concept_label_fontsize,
            "concept_label_color": concept_label_color,
            "concept_label_angle": concept_label_angle,
            "concept_label_show_grid": concept_label_show_grid,
            "concept_label_grid_color": concept_label_grid_color,
            # ★ NEW: N_i node ID label customization
            "node_id_fontsize": node_id_fontsize,
            "node_id_color": node_id_color,
        })
    else:
        settings.update({"width": width_3d, "height": height_3d, "grid_size": grid_size_3d})

    return settings, df_all


# =====================================================================
# MAIN APP
# =====================================================================
def main():
    st.title("📊 SIS Bar Chart Studio")
    st.caption("Customizable 2D / 3D / Polar bar charts for Sustainability Impact Scores — 50+ colormaps, interactive concept selection, score labels on bar tips.")

    df_all = load_default_data()
    settings, df_all = render_sidebar(df_all)

    selected = settings["selected_concepts"]
    if not selected:
        st.warning("⚠️ Please select at least one concept to plot.")
        st.info("Use the sidebar → 'Select Concepts' → click 'Select All' or pick specific concepts.")
        return

    df_plot = df_all[df_all["concept"].isin(selected)].copy().sort_values("SIS", ascending=False).reset_index(drop=True)
    st.markdown(f"**Selected:** {len(df_plot)} / {len(df_all)} concepts")

    if "2D" in settings["chart_type"]:
        png_bytes = render_2d_bar_chart(df_plot, **{k: v for k, v in settings.items() if k in [
            "cmap_name", "figsize", "dpi", "bar_width", "label_position", "label_rotation", "label_fontsize",
            "x_label_mode", "x_label_fontsize", "x_label_rotation", "show_node_ids", "show_value_labels",
            "highlight_anchors", "title", "show_grid", "bg_color", "text_color", "sort_order"
        ]})
        st.image(png_bytes, use_container_width=True)
        st.download_button("⬇️ Download PNG", data=png_bytes, file_name="sis_bar_chart_2d.png", mime="image/png")

    elif "Polar" in settings["chart_type"]:
        fig_polar = render_polar_bar_chart(
            df_plot,
            cmap_name=settings["cmap_name"],
            highlight_anchors=settings["highlight_anchors"],
            title=settings["title"],
            bg_color=settings["bg_color"],
            text_color=settings["text_color"],
            sort_order=settings["sort_order"],
            chart_size=settings["polar_size"],
            bar_angular_width=settings["bar_angular_width"],
            angular_rotation=settings["angular_rotation"],
            angular_direction=settings["angular_direction"],
            show_value_labels=settings["show_value_labels"],
            show_node_ids=settings["show_node_ids"],
            label_fontsize=settings["label_fontsize"],
            radial_range_padding=settings["radial_padding"],
            # ★ NEW: radial axis customization
            radial_axis_show=settings["radial_axis_show"],
            radial_axis_show_ticks=settings["radial_axis_show_ticks"],
            radial_axis_n_ticks=settings["radial_axis_n_ticks"],
            radial_axis_fontsize=settings["radial_axis_fontsize"],
            radial_axis_decimals=settings["radial_axis_decimals"],
            radial_axis_color=settings["radial_axis_color"],
            radial_axis_show_grid=settings["radial_axis_show_grid"],
            radial_axis_grid_color=settings["radial_axis_grid_color"],
            # ★ NEW: concept label customization
            concept_label_show=settings["concept_label_show"],
            concept_label_fontsize=settings["concept_label_fontsize"],
            concept_label_color=settings["concept_label_color"],
            concept_label_angle=settings["concept_label_angle"],
            concept_label_show_grid=settings["concept_label_show_grid"],
            concept_label_grid_color=settings["concept_label_grid_color"],
            # ★ NEW: N_i node ID label customization
            node_id_fontsize=settings["node_id_fontsize"],
            node_id_color=settings["node_id_color"],
        )
        st.plotly_chart(fig_polar, use_container_width=True)
        st.caption("💡 *The polar chart is now a TRUE CIRCLE: width and height are forced equal, with tight margins and proper angular axis.*")
        html_bytes = fig_polar.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button("⬇️ Download Interactive HTML", data=html_bytes, file_name="sis_polar_chart.html", mime="text/html")

    else:
        fig_3d = render_3d_bar_chart(df_plot, **{k: v for k, v in settings.items() if k in [
            "cmap_name", "width", "height", "show_value_labels", "show_node_ids", "highlight_anchors",
            "title", "sort_order", "label_fontsize", "bg_color", "text_color", "grid_size"
        ]})
        st.plotly_chart(fig_3d, use_container_width=True)
        html_bytes = fig_3d.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button("⬇️ Download Interactive HTML", data=html_bytes, file_name="sis_bar_chart_3d.html", mime="text/html")

    if settings["show_legend"]:
        st.markdown("---")
        st.subheader("🎨 Concept Legend")
        st.caption("Color swatches map to bars in the chart above.")
        render_legend(df_plot, settings["cmap_name"], n_cols=settings["legend_cols"])

    st.markdown("---")
    st.subheader("📋 Selected Concepts (Data Table)")
    st.dataframe(df_plot[["node_id", "concept", "SIS", "is_anchor"]], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download selected concepts (CSV)", data=df_plot.to_csv(index=False).encode("utf-8"),
                       file_name="sis_selected_concepts.csv", mime="text/csv")

    with st.expander("💡 Tips for handling long concept names"):
        st.markdown("""
        **Problem:** Some concept names like `pressureless_sintering` are too long to fit horizontally.

        **Solutions offered in this app:**
        1. **Abbreviated mode (2D)** — Replaces long names with `N1`, `N2`, ... (manuscript node IDs).
        2. **Wrapped mode (2D)** — Breaks long names into 2–3 lines (e.g., `pressureless\\nsintering`).
        3. **Rotated mode (2D)** — Keeps the full name but rotates the label 45° or 90°.
        4. **Polar / Radial Chart ✨** — Lays concepts around a circle so name length matters less.

        **Tips for the polar chart specifically:**
        - **Polar chart looks elliptical?** Increase *Chart size (px, square)* in the sidebar.
          The chart is forced to be square (width = height) for a true circle.
        - **Bars don't fill the circle?** Increase *Bar angular width* toward 1.0.
        - **Tip labels overlap the bar?** Increase *Radial range padding* (e.g., 1.30).
        - **Start at different position?** Adjust *Start angle (°)*: 90 = top, 0 = right.
        """)


if __name__ == "__main__":
    main()
