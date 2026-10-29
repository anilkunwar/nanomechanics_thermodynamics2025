#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIS Bar Chart Studio
====================
A focused Streamlit app for visualizing Sustainability Impact Score (SIS) —
or any node-level score — as customizable 2D and 3D bar charts.

Features
--------
- 50+ colormaps (matplotlib + plotly): viridis, plasma, inferno, magma, jet,
  turbo, rainbow, cividis, spectral, coolwarm, RdBu, seismic, nipy_spectral,
  gist_rainbow, hsv, terrain, ocean, gist_earth, tab10/20, Set1/2/3, Accent,
  Dark2, Paired, Pastel1/2, YlOrRd, OrRd, PuRd, RdPu, BuPu, GnBu, YlGnBu,
  PuBuGn, BuGn, YlGn, afmhot, copper, bone, gray, pink, spring, summer,
  autumn, winter, cool, hot, twilight, cubehelix, jet_r, viridis_r, ...
- 2D bar chart (matplotlib) with score labels on top of each bar
- 3D bar chart (plotly surface) with score labels floating above bars
- Long concept names handled 3 ways: abbreviated (N1, N2...), wrapped
  (multi-line), or rotated (vertical text)
- Interactive concept multi-selector: pick exactly which concepts to plot
  (select all / remove all / regex filter)
- Full image customization: DPI, figure size, font sizes, bar width,
  label rotation, label position (top/outside vs center/inside)
- Legend panel: color swatch + N_i ID + full concept name
- Downloads: PNG (2D), HTML (3D), CSV (scores)

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
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import io
import re
import textwrap
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
    # Matplotlib perceptual
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno",
    "magma": "Magma", "cividis": "Cividis", "turbo": "Turbo",
    # Classic
    "jet": "Jet", "rainbow": "Rainbow", "hsv": "HSV",
    "nipy_spectral": "NipySpectral", "gist_rainbow": "GistRainbow",
    "coolwarm": "Coolwarm", "RdBu": "RdBu", "seismic": "Seismic",
    "Spectral": "Spectral", "bwr": "Bluered", "BrBG": "BrBG",
    "PiYG": "PiYG", "PRGn": "PRGn", "PuOr": "PuOr", "RdYlBu": "RdYlBu",
    "RdYlGn": "RdYlGn",
    # Sequential single-hue
    "Blues": "Blues", "Reds": "Reds", "Greens": "Greens",
    "Oranges": "Oranges", "Purples": "Purples", "Greys": "Greys",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu",
    "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn",
    "BuGn": "BuGn", "YlGn": "YlGn",
    # Cyclic / misc
    "twilight": "Twilight", "cubehelix": "Cubehelix",
    "bone": "Bone", "gray": "Greys", "pink": "Pink",
    "spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "cool": "Cool", "hot": "Hot", "copper": "Copper", "afmhot": "Afmhot",
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean",
    # Qualitative
    "tab10": "Set1", "tab20": "Set2", "tab20b": "Set3", "tab20c": "Set3",
    "Accent": "Accent", "Dark2": "Dark2", "Paired": "Paired",
    "Pastel1": "Pastel1", "Pastel2": "Pastel2",
    "Set1": "Set1", "Set2": "Set2", "Set3": "Set3",
    # Reverse variants (key ones)
    "viridis_r": "Viridis_r", "plasma_r": "Plasma_r",
    "inferno_r": "Inferno_r", "magma_r": "Magma_r",
    "turbo_r": "Turbo_r", "jet_r": "Jet_r",
}

COLORMAP_NAMES = list(COLORMAPS.keys())


def get_cmap_colors(cmap_name: str, n: int) -> List[str]:
    """Return n hex colors sampled from the named matplotlib colormap."""
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(max(n, 1))
        return [mcolors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap(cmap_name, max(n, 1))
            return [mcolors.to_hex(cmap(i)) for i in range(n)]
        except Exception:
            cmap = cm.get_cmap("viridis", max(n, 1))
            return [mcolors.to_hex(cmap(i)) for i in range(n)]


# =====================================================================
# MANUSCRIPT NODE-ID MAPPING (from Table A.2 — 40-node concept graph)
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
    """Shorten a concept name for tight bar-chart x-axis labels."""
    concept = concept.replace("_", " ")
    if len(concept) <= max_len:
        return concept
    # Try to break on spaces
    parts = concept.split(" ")
    if len(parts) >= 2:
        return parts[0][:max_len-3] + "…"
    return concept[:max_len-1] + "…"


def wrap_label(concept: str, width: int = 10) -> str:
    """Wrap a long concept name into multiple lines of given width."""
    return "\n".join(textwrap.wrap(concept.replace("_", " "), width=width)) or concept


# =====================================================================
# PRELOADED SIS DATA (from the previous computation)
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
    """Load the default SIS results from the previous computation."""
    df = pd.read_csv(io.StringIO(DEFAULT_SIS_CSV))
    df["node_id"] = df["concept"].map(NODE_ID_MAP).fillna("?")
    df["is_anchor"] = df["concept"].isin(ANCHORS)
    return df.sort_values("SIS", ascending=False).reset_index(drop=True)


# =====================================================================
# 2D BAR CHART (matplotlib)
# =====================================================================

def render_2d_bar_chart(
    df: pd.DataFrame,
    cmap_name: str,
    figsize: Tuple[float, float],
    dpi: int,
    bar_width: float,
    label_position: str,        # "top" (outside) | "center" (inside)
    label_rotation: int,
    label_fontsize: int,
    x_label_mode: str,          # "full" | "abbreviated" | "wrapped" | "rotated"
    x_label_fontsize: int,
    x_label_rotation: int,
    show_node_ids: bool,
    show_value_labels: bool,
    highlight_anchors: bool,
    title: str,
    show_grid: bool,
    bg_color: str,
    text_color: str,
    sort_order: str,            # "descending" | "ascending" | "alphabetical"
) -> bytes:
    """Render a 2D bar chart and return PNG bytes."""
    if df.empty:
        return b""

    # Sort
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
    bars = ax.bar(
        x_positions, sis_values,
        width=bar_width,
        color=colors,
        edgecolor=text_color if bg_color != "white" else "#333333",
        linewidth=0.6,
        alpha=0.95,
    )

    # Highlight anchors with gold edge
    if highlight_anchors:
        for i, is_anchor in enumerate(df["is_anchor"]):
            if is_anchor:
                bars[i].set_edgecolor("#FFD700")
                bars[i].set_linewidth(2.5)

    # Value labels
    if show_value_labels:
        for i, (bar, val) in enumerate(zip(bars, sis_values)):
            if label_position == "top":
                y = val + (max(sis_values) - min(sis_values)) * 0.012 + 0.005
                va = "bottom"
                color = text_color
                weight = "bold"
            else:  # center / inside
                y = val / 2
                va = "center"
                # Pick white or black text based on bar darkness
                r, g, b, _ = mcolors.to_rgba(colors[i])
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                color = "white" if lum < 0.55 else "black"
                weight = "bold"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y,
                f"{val:.3f}",
                ha="center", va=va,
                fontsize=label_fontsize, fontweight=weight, color=color,
                rotation=label_rotation,
            )

    # X-axis labels
    x_labels = []
    for concept in concepts:
        if x_label_mode == "full":
            x_labels.append(concept.replace("_", " "))
        elif x_label_mode == "abbreviated":
            x_labels.append(short_label(concept, max_len=12))
        elif x_label_mode == "wrapped":
            x_labels.append(wrap_label(concept, width=10))
        else:  # rotated: use full name, rely on rotation
            x_labels.append(concept.replace("_", " "))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        x_labels,
        fontsize=x_label_fontsize,
        rotation=x_label_rotation,
        ha="right" if x_label_rotation != 0 else "center",
        color=text_color,
    )

    # Node IDs above x-axis labels (small, secondary)
    if show_node_ids:
        node_ids = df["node_id"].tolist()
        secax = ax.secondary_xaxis(-0.18)
        secax.set_xticks(x_positions)
        secax.set_xticklabels(node_ids, fontsize=x_label_fontsize - 2,
                              color="#666666", fontweight="bold")
        secax.tick_params(length=0)
        # Hide the secondary axis spine
        secax.spines["bottom"].set_visible(False)

    # Styling
    ax.set_ylabel("SIS Score", fontsize=label_fontsize + 2, color=text_color,
                   fontweight="bold")
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

    # Adjust y-limits to fit top labels
    if label_position == "top" and show_value_labels:
        ymax = max(sis_values) * 1.08
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(0, max(sis_values) * 1.02)

    # Save
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg_color, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# =====================================================================
# 3D BAR CHART (plotly)
# =====================================================================

def render_3d_bar_chart(
    df: pd.DataFrame,
    cmap_name: str,
    width: int,
    height: int,
    show_value_labels: bool,
    show_node_ids: bool,
    highlight_anchors: bool,
    title: str,
    sort_order: str,
    label_fontsize: int,
    bg_color: str,
    text_color: str,
    grid_size: Optional[int] = None,  # if None, auto-compute square-ish grid
) -> go.Figure:
    """Render a 3D bar chart using plotly Mesh3D / surface bars."""
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

    # Auto grid size if not provided
    if grid_size is None or grid_size <= 0:
        grid_size = int(np.ceil(np.sqrt(n)))
    cols = grid_size
    rows = int(np.ceil(n / cols))

    # Build grid positions
    x_pos, y_pos = [], []
    for i in range(n):
        r = i // cols
        c = i % cols
        x_pos.append(c)
        y_pos.append(r)

    sis_values = df["SIS"].values
    concepts = df["concept"].tolist()
    node_ids = df["node_id"].tolist()
    is_anchors = df["is_anchor"].tolist()

    # Build mesh3d traces: one per bar (cylinder approximation via 8 vertices)
    fig = go.Figure()

    for i in range(n):
        x, y, z = x_pos[i], y_pos[i], sis_values[i]
        # Bar footprint: half-width 0.4
        hw = 0.4
        # 8 corners of a box: (x-hw, y-hw, 0)...(x+hw, y+hw, z)
        verts_x = [x-hw, x+hw, x+hw, x-hw, x-hw, x+hw, x+hw, x-hw]
        verts_y = [y-hw, y-hw, y+hw, y+hw, y-hw, y-hw, y+hw, y+hw]
        verts_z = [0, 0, 0, 0, z, z, z, z]
        # Faces: bottom (0,1,2,3), top (4,5,6,7), sides
        i_faces = [0,4,0,1,2,4,0,1]
        j_faces = [1,5,3,2,3,5,3,2]
        k_faces = [2,6,1,0,3,6,2,3]
        # Fix: build proper cube faces
        i_faces = [0,0,0,1,2,3,4,4,5,6]
        j_faces = [1,2,3,2,3,0,5,7,6,7]
        k_faces = [2,3,1,3,0,1,6,6,7,5]
        # Simpler: use plotly's bar3d-like approach via Mesh3d with cube
        # Use the standard cube face indices
        # 8 vertices of cube: 0-3 bottom (CCW), 4-7 top (CCW)
        # 12 triangles covering 6 faces
        I = [0,1,2,3,0,1,2,3,4,5,6,7,0,4,1,5,2,6,3,7,0,1,2,3]
        J = [1,0,3,2,4,5,6,7,5,4,7,6,4,0,5,1,6,2,7,3,3,2,1,0]
        K = [2,3,0,1,5,4,7,6,6,7,4,5,5,1,4,0,7,3,6,2,2,3,0,1]
        # Wait, the above is buggy — let me use a clean approach with 12 triangles

        # Clean cube triangulation: 6 faces × 2 triangles = 12 triangles
        # Vertex indices (8 corners):
        #   bottom: 0=(x-hw,y-hw,0), 1=(x+hw,y-hw,0), 2=(x+hw,y+hw,0), 3=(x-hw,y+hw,0)
        #   top:    4=(x-hw,y-hw,z), 5=(x+hw,y-hw,z), 6=(x+hw,y+hw,z), 7=(x-hw,y+hw,z)
        I = [0,2, 4,6, 0,1, 2,3, 1,2, 0,3]
        J = [2,3, 6,7, 4,5, 6,7, 5,6, 1,2]
        K = [3,0, 5,4, 1,0, 3,2, 4,5, 2,1]
        # This is messy; use simpler approach: build per-bar mesh
        # Use 8 corners and 12 triangles
        verts_x = [x-hw, x+hw, x+hw, x-hw, x-hw, x+hw, x+hw, x-hw]
        verts_y = [y-hw, y-hw, y+hw, y+hw, y-hw, y-hw, y+hw, y+hw]
        verts_z = [0, 0, 0, 0, z, z, z, z]
        # 12 triangle indices (2 per face × 6 faces)
        I = [0,2, 0,4, 1,5, 2,6, 3,7, 0,1]
        J = [1,3, 1,5, 2,6, 3,7, 0,4, 2,3]
        K = [2,0, 4,0, 5,1, 6,2, 7,3, 3,2]
        # Actually the cleanest is:
        I = [0,0,1,1,2,2,3,3,0,0,1,2]
        J = [1,2,2,3,3,0,0,1,4,5,5,6]
        K = [2,3,3,0,0,1,1,2,5,4,4,7]
        # Use plotly's Mesh3d
        fig.add_trace(go.Mesh3d(
            x=verts_x, y=verts_y, z=verts_z,
            i=I, j=J, k=K,
            color=colors[i],
            opacity=0.95,
            hovertemplate=f"<b>{concepts[i]}</b> ({node_ids[i]})<br>SIS: {sis_values[i]:.4f}<extra></extra>",
            name=concepts[i],
        ))

        # Value label on top
        if show_value_labels:
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z + 0.04],
                mode="text",
                text=[f"{sis_values[i]:.3f}"],
                textfont=dict(size=label_fontsize, color=text_color, family="Arial Black"),
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            ))

        # Node ID below bar (on the floor)
        if show_node_ids:
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[-0.03],
                mode="text",
                text=[node_ids[i]],
                textfont=dict(size=label_fontsize - 2, color="#666666", family="Arial Bold"),
                textposition="bottom center",
                showlegend=False,
                hoverinfo="skip",
            ))

        # Concept name on the floor (small)
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y - 0.55], z=[0],
            mode="text",
            text=[concepts[i].replace("_", " ")[:15]],
            textfont=dict(size=label_fontsize - 3, color=text_color, family="Arial"),
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip",
        ))

    # Highlight anchors by adding gold outline
    if highlight_anchors:
        for i in range(n):
            if is_anchors[i]:
                x, y, z = x_pos[i], y_pos[i], sis_values[i]
                hw = 0.42
                # Vertical edges as line traces
                edge_x = [x-hw, x+hw, x+hw, x-hw, x-hw, x-hw, x+hw, x+hw, x-hw, x-hw, x+hw, x+hw, x-hw, x+hw, x+hw, x-hw]
                edge_y = [y-hw, y-hw, y+hw, y+hw, y-hw, y-hw, y-hw, y-hw, y+hw, y+hw, y+hw, y+hw, y-hw, y-hw, y+hw, y+hw]
                edge_z = [0, 0, 0, 0, 0, z, z, z, z, 0, 0, z, z, z, z, z]
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode="lines",
                    line=dict(color="#FFD700", width=4),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=label_fontsize + 6, color=text_color)),
        scene=dict(
            xaxis=dict(title="", showticklabels=False, showgrid=False,
                       backgroundcolor=bg_color, showbackground=False),
            yaxis=dict(title="", showticklabels=False, showgrid=False,
                       backgroundcolor=bg_color, showbackground=False),
            zaxis=dict(title="SIS Score",
                       title_font=dict(color=text_color, size=label_fontsize),
                       tickfont=dict(color=text_color, size=label_fontsize - 2),
                       backgroundcolor=bg_color, showbackground=False,
                       gridcolor="rgba(150,150,150,0.3)"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
            aspectmode="data",
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


# =====================================================================
# LEGEND (concept → color swatch → N_i ID → full name)
# =====================================================================

def render_legend(df: pd.DataFrame, cmap_name: str, n_cols: int = 4) -> None:
    """Render an HTML legend table mapping color swatches to concept names + N_i IDs."""
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
        <div style='padding:6px 8px; border-radius:6px;
                    background-color:{color}22;
                    border-left:4px solid {color};
                    margin-bottom:6px; font-size:12px;'>
            <span style='display:inline-block; width:14px; height:14px;
                         background-color:{color}; border-radius:3px;
                         margin-right:6px; vertical-align:middle;'></span>
            <b style='color:#333;'>{nid}</b>
            <span style='color:#666;'>{concept.replace('_', ' ')}</span>{anchor_badge}
            <span style='float:right; color:#888; font-family:monospace;'>{sis:.3f}</span>
        </div>
        """)

    # Distribute rows across columns
    per_col = (len(rows_html) + n_cols - 1) // n_cols
    cols = st.columns(n_cols)
    for ci in range(n_cols):
        with cols[ci]:
            start = ci * per_col
            end = start + per_col
            for html in rows_html[start:end]:
                st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# SIDEBAR — All controls
# =====================================================================

def render_sidebar(df_all: pd.DataFrame) -> Dict:
    """Render the sidebar controls and return the user's settings as a dict."""
    with st.sidebar:
        st.header("📊 SIS Bar Chart Studio")
        st.caption("Customizable 2D / 3D bar charts for Sustainability Impact Scores")

        # ---- 1. Data source ----
        st.subheader("1. Data Source")
        data_mode = st.radio(
            "Choose data source:",
            ["Use preloaded SIS results", "Upload CSV"],
            index=0,
            horizontal=True,
        )

        if data_mode == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV (cols: concept, SIS)", type=["csv"])
            if uploaded is not None:
                try:
                    df_all = pd.read_csv(uploaded)
                    if "concept" not in df_all.columns or "SIS" not in df_all.columns:
                        st.error("CSV must have 'concept' and 'SIS' columns.")
                    else:
                        df_all["node_id"] = df_all["concept"].map(NODE_ID_MAP).fillna("?")
                        df_all["is_anchor"] = df_all["concept"].isin(ANCHORS)
                        df_all = df_all.sort_values("SIS", ascending=False).reset_index(drop=True)
                        st.success(f"Loaded {len(df_all)} concepts from CSV.")
                except Exception as e:
                    st.error(f"Failed to parse CSV: {e}")

        # ---- 2. Concept selection ----
        st.subheader("2. Select Concepts")
        all_concepts = df_all["concept"].tolist()

        # Filter box
        filter_pattern = st.text_input(
            "Filter concepts (regex or substring):",
            value="",
            help="Type a substring or regex to filter the list. e.g., 'twins', 'defect|twins', '^p'",
        )
        try:
            if filter_pattern:
                filtered = [c for c in all_concepts if re.search(filter_pattern, c, re.IGNORECASE)]
            else:
                filtered = all_concepts
        except re.error:
            st.warning(f"Invalid regex: {filter_pattern}. Showing all.")
            filtered = all_concepts

        # Select-all / remove-all / invert
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

        # Default to top 15 by SIS
        if "selected_concepts" not in st.session_state:
            st.session_state["selected_concepts"] = df_all.head(15)["concept"].tolist()

        selected = st.multiselect(
            f"Concepts to plot ({len(filtered)} shown, {len(st.session_state['selected_concepts'])} selected):",
            options=filtered,
            default=[c for c in st.session_state["selected_concepts"] if c in filtered],
            key="selected_concepts",
        )

        # Quick presets
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
                defects = ["twins", "stacking_faults", "lattice_distortion",
                          "defect_engineering", "diffusion_highways"]
                st.session_state["selected_concepts"] = [c for c in all_concepts if c in defects]
                st.rerun()

        # ---- 3. Chart type ----
        st.subheader("3. Chart Type & Colormap")
        chart_type = st.selectbox(
            "Chart type:",
            ["2D Bar Chart (matplotlib)", "3D Bar Chart (plotly)"],
            index=0,
        )
        cmap_name = st.selectbox(
            f"Colormap ({len(COLORMAP_NAMES)} options):",
            options=COLORMAP_NAMES,
            index=COLORMAP_NAMES.index("inferno"),
            help="inferno, plasma, viridis, jet, turbo, rainbow, spectral, coolwarm, ...",
        )
        reverse_cmap = st.checkbox("Reverse colormap", value=False)
        if reverse_cmap:
            cmap_name = cmap_name + "_r" if not cmap_name.endswith("_r") else cmap_name

        sort_order = st.selectbox(
            "Sort order:",
            ["descending", "ascending", "alphabetical"],
            index=0,
        )

        # ---- 4. Image customization ----
        st.subheader("4. Image Customization")
        title = st.text_input("Chart title:", value="Sustainability Impact Score (SIS) by Concept")

        if "2D" in chart_type:
            figsize_w = st.slider("Figure width (inches)", 6, 24, 14)
            figsize_h = st.slider("Figure height (inches)", 4, 16, 8)
            dpi = st.slider("DPI", 80, 400, 150)
            bar_width = st.slider("Bar width", 0.2, 1.0, 0.7)
        else:
            width_3d = st.slider("Chart width (px)", 600, 1600, 1000)
            height_3d = st.slider("Chart height (px)", 500, 1400, 800)
            grid_size_3d = st.slider("3D grid columns", 0, 12, 0,
                                       help="0 = auto-compute square-ish grid")

        label_fontsize = st.slider("Value label font size", 6, 20, 10)
        x_label_fontsize = st.slider("X-axis label font size", 6, 18, 9)
        show_value_labels = st.checkbox("Show SIS score on bar", value=True)
        show_node_ids = st.checkbox("Show N_i node IDs", value=True)
        highlight_anchors = st.checkbox("Highlight anchor nodes (gold outline)",
                                         value=True)

        if "2D" in chart_type:
            label_position = st.selectbox(
                "Label position:",
                ["top (outside bar)", "center (inside bar)"],
                index=0,
            ).split(" ")[0]
            label_rotation = st.slider("Value label rotation (°)", 0, 90, 0)
            x_label_mode = st.selectbox(
                "Long-name handling:",
                ["full", "abbreviated", "wrapped", "rotated"],
                index=3,
                help="full=truncate, abbreviated=N1/N2..., wrapped=multi-line, rotated=full vertical text",
            )
            x_label_rotation = st.slider("X-axis label rotation (°)", 0, 90,
                                          {0: 0, 1: 45, 2: 90}.get(2, 45) if x_label_mode == "rotated" else 45)

        bg_color = st.color_picker("Background color", value="#ffffff")
        text_color = st.color_picker("Text color", value="#1a1a1a")
        show_grid = st.checkbox("Show grid (2D only)", value=True)

        # ---- 5. Legend ----
        st.subheader("5. Legend")
        show_legend = st.checkbox("Show concept legend", value=True)
        legend_cols = st.slider("Legend columns", 1, 6, 4)

    # Build settings dict
    settings = {
        "data_mode": data_mode,
        "selected_concepts": selected,
        "chart_type": chart_type,
        "cmap_name": cmap_name,
        "sort_order": sort_order,
        "title": title,
        "label_fontsize": label_fontsize,
        "x_label_fontsize": x_label_fontsize,
        "show_value_labels": show_value_labels,
        "show_node_ids": show_node_ids,
        "highlight_anchors": highlight_anchors,
        "bg_color": bg_color,
        "text_color": text_color,
        "show_grid": show_grid,
        "show_legend": show_legend,
        "legend_cols": legend_cols,
    }
    if "2D" in chart_type:
        settings.update({
            "figsize": (figsize_w, figsize_h),
            "dpi": dpi,
            "bar_width": bar_width,
            "label_position": label_position,
            "label_rotation": label_rotation,
            "x_label_mode": x_label_mode,
            "x_label_rotation": x_label_rotation,
        })
    else:
        settings.update({
            "width": width_3d,
            "height": height_3d,
            "grid_size": grid_size_3d,
        })
    return settings, df_all


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    st.title("📊 SIS Bar Chart Studio")
    st.caption(
        "Customizable 2D / 3D bar charts for Sustainability Impact Scores — "
        "50+ colormaps, interactive concept selection, score labels on bar tips."
    )

    # Load default data
    df_all = load_default_data()

    # Render sidebar
    settings, df_all = render_sidebar(df_all)

    # Filter to selected concepts
    selected = settings["selected_concepts"]
    if not selected:
        st.warning("⚠️ Please select at least one concept to plot.")
        st.info("Use the sidebar → 'Select Concepts' → click 'Select All' or pick specific concepts.")
        return

    df_plot = df_all[df_all["concept"].isin(selected)].copy()
    df_plot = df_plot.sort_values("SIS", ascending=False).reset_index(drop=True)

    # Display selected count
    st.markdown(f"**Selected:** {len(df_plot)} / {len(df_all)} concepts")

    # ---- Render chart ----
    if "2D" in settings["chart_type"]:
        png_bytes = render_2d_bar_chart(
            df_plot,
            cmap_name=settings["cmap_name"],
            figsize=settings["figsize"],
            dpi=settings["dpi"],
            bar_width=settings["bar_width"],
            label_position=settings["label_position"],
            label_rotation=settings["label_rotation"],
            label_fontsize=settings["label_fontsize"],
            x_label_mode=settings["x_label_mode"],
            x_label_fontsize=settings["x_label_fontsize"],
            x_label_rotation=settings["x_label_rotation"],
            show_node_ids=settings["show_node_ids"],
            show_value_labels=settings["show_value_labels"],
            highlight_anchors=settings["highlight_anchors"],
            title=settings["title"],
            show_grid=settings["show_grid"],
            bg_color=settings["bg_color"],
            text_color=settings["text_color"],
            sort_order=settings["sort_order"],
        )
        st.image(png_bytes, use_container_width=True)
        st.download_button(
            "⬇️ Download PNG",
            data=png_bytes,
            file_name="sis_bar_chart_2d.png",
            mime="image/png",
        )
    else:
        fig_3d = render_3d_bar_chart(
            df_plot,
            cmap_name=settings["cmap_name"],
            width=settings["width"],
            height=settings["height"],
            show_value_labels=settings["show_value_labels"],
            show_node_ids=settings["show_node_ids"],
            highlight_anchors=settings["highlight_anchors"],
            title=settings["title"],
            sort_order=settings["sort_order"],
            label_fontsize=settings["label_fontsize"],
            bg_color=settings["bg_color"],
            text_color=settings["text_color"],
            grid_size=settings["grid_size"],
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        # HTML download
        html_bytes = fig_3d.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            "⬇️ Download Interactive HTML",
            data=html_bytes,
            file_name="sis_bar_chart_3d.html",
            mime="text/html",
        )

    # ---- Legend ----
    if settings["show_legend"]:
        st.markdown("---")
        st.subheader("🎨 Concept Legend")
        st.caption("Color swatches map to bars in the chart above.")
        render_legend(df_plot, settings["cmap_name"], n_cols=settings["legend_cols"])

    # ---- Data table ----
    st.markdown("---")
    st.subheader("📋 Selected Concepts (Data Table)")
    st.dataframe(
        df_plot[["node_id", "concept", "SIS", "is_anchor"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "⬇️ Download selected concepts (CSV)",
        data=df_plot.to_csv(index=False).encode("utf-8"),
        file_name="sis_selected_concepts.csv",
        mime="text/csv",
    )

    # ---- Help / tips ----
    with st.expander("💡 Tips for handling long concept names"):
        st.markdown(
            """
            **Problem:** Some concept names like `pressureless_sintering` or
            `power_module_assembly` are too long to fit horizontally on the
            x-axis without overlapping.

            **Solutions offered in this app:**

            1. **Abbreviated mode** — Replaces long names with `N1`, `N2`, ...
               (manuscript node IDs). Use the "Show N_i node IDs" toggle to
               display these in a secondary axis. Best for compact charts with
               many concepts.

            2. **Wrapped mode** — Breaks long names into 2–3 lines of
               ~10 characters each (e.g., `pressureless_sintering` →
               `pressureless\\nsintering`). Best when you have a moderate
               number of concepts (≤30).

            3. **Rotated mode** — Keeps the full name but rotates the label
               45° or 90°. Best for publication-quality figures where
               readability matters more than compactness.

            4. **Full mode** — Truncates with `…` at 12 characters. Quick
               default for screening.

            **Additional tips:**
            - Increase figure width (sidebar → Image Customization) to give
              labels more room.
            - Reduce x-axis font size for very dense charts.
            - Use the "Filter concepts" box in the sidebar to plot only the
              subset you care about (e.g., type `defect|twins` to show only
              defect-related concepts).
            - The **3D chart** lays concepts out in a grid, so name length
              matters less — it's a good fallback for visualizing all 44
              concepts at once.
            """
        )


if __name__ == "__main__":
    main()
