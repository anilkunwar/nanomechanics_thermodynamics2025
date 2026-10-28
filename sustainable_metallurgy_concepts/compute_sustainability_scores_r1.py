#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIS & Bridge-Criticality Sensitivity Calculator
================================================
A focused Streamlit web app that computes:
  1. Sustainability Impact Score (SIS) for every node in a weighted concept graph
  2. Bridge-Criticality Sensitivity Analysis (before/after node removal)
  3. Criticality Ranking — every node tested, ranked by structural impact

SIS(v) = (1/|S|) * Σ_{s ∈ S} 1/(1 + d_w(v, s))
  where S = {net_zero, cleaner_production, circular_economy}
  and d_w is the weighted shortest-path distance using 1/w as edge weight
  (so high-weight edges become short distances).

Run:
  streamlit run compute_sis_sensitivity_app.py
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Register CJK + Latin fallback fonts (used only if labels contain non-Latin chars)
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
    page_title="SIS & Bridge-Criticality Calculator",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CORE ALGORITHMS
# =====================================================================

def compute_sustainability_impact_score(
    nx_graph: nx.Graph,
    anchor_nodes: List[str],
) -> pd.DataFrame:
    """
    SIS(v) = (1/|S|) * Σ_{s∈S} 1/(1 + d_w(v, s))
    where d_w is the weighted shortest-path distance using 1/w as edge weight.
    """
    valid_anchors = [a for a in anchor_nodes if a in nx_graph.nodes()]
    if not valid_anchors:
        return pd.DataFrame()

    # Build distance graph: weights = 1/w so high-weight ⇒ short distance
    G_dist = nx.Graph()
    for u, v, data in nx_graph.edges(data=True):
        w = data.get("weight", 1.0)
        if w > 0:
            G_dist.add_edge(u, v, weight=1.0 / w)

    rows = []
    for node in nx_graph.nodes():
        distances = []
        for anchor in valid_anchors:
            try:
                d = nx.shortest_path_length(
                    G_dist, source=node, target=anchor, weight="weight"
                )
                distances.append(d)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distances.append(np.inf)

        sis_values = [1.0 / (1.0 + d) if np.isfinite(d) else 0.0 for d in distances]
        sis = float(np.mean(sis_values)) if sis_values else 0.0

        rows.append({
            "concept": node,
            "SIS": round(sis, 4),
            "n_anchors_reachable": int(sum(1 for d in distances if np.isfinite(d))),
            "distances": {a: (round(d, 3) if np.isfinite(d) else "∞")
                          for a, d in zip(valid_anchors, distances)},
        })

    return pd.DataFrame(rows).sort_values("SIS", ascending=False).reset_index(drop=True)


def _compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute modularity, avg shortest path, # components, etc."""
    metrics: Dict[str, Any] = {}
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(G))
        if len(partition) > 1:
            metrics["modularity"] = float(community.modularity(G, partition))
        else:
            metrics["modularity"] = 0.0
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0

    try:
        if G.number_of_nodes() == 0:
            metrics["avg_shortest_path"] = np.nan
        elif nx.is_connected(G):
            metrics["avg_shortest_path"] = float(nx.average_shortest_path_length(G))
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            metrics["avg_shortest_path"] = float(nx.average_shortest_path_length(subG))
    except Exception:
        metrics["avg_shortest_path"] = np.nan

    metrics["n_connected_components"] = nx.number_connected_components(G)
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0
    try:
        metrics["avg_clustering"] = float(nx.average_clustering(G))
    except Exception:
        metrics["avg_clustering"] = 0.0
    return metrics


def bridge_criticality_sensitivity_analysis(
    nx_graph: nx.Graph,
    node_to_remove: str,
) -> Dict[str, Any]:
    """Quantify structural criticality of a bridge node by removal."""
    if node_to_remove not in nx_graph.nodes():
        return {"error": f"Node '{node_to_remove}' not in graph."}

    before = _compute_graph_metrics(nx_graph)

    G_reduced = nx_graph.copy()
    G_reduced.remove_node(node_to_remove)
    after = _compute_metrics_reduced(G_reduced)

    delta_modularity = after["modularity"] - before["modularity"]
    delta_avg_path = (
        after["avg_shortest_path"] - before["avg_shortest_path"]
        if np.isfinite(after["avg_shortest_path"])
        and np.isfinite(before["avg_shortest_path"])
        else np.nan
    )
    delta_components = after["n_connected_components"] - before["n_connected_components"]

    return {
        "node_removed": node_to_remove,
        "before": before,
        "after": after,
        "delta_modularity": round(delta_modularity, 4),
        "delta_avg_shortest_path": round(delta_avg_path, 4) if np.isfinite(delta_avg_path) else "∞",
        "delta_connected_components": delta_components,
        "delta_avg_clustering": round(after["avg_clustering"] - before["avg_clustering"], 4),
        "delta_density": round(after["density"] - before["density"], 4),
        "graph_fragmented": delta_components > 0,
    }


def _compute_metrics_reduced(G: nx.Graph) -> Dict[str, Any]:
    """Same as _compute_graph_metrics but used for the reduced graph."""
    return _compute_graph_metrics(G)


def compute_criticality_ranking(
    nx_graph: nx.Graph,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Iterate over EVERY node in the graph, remove it, and record the impact.
    Returns a DataFrame ranked by structural criticality.
    """
    baseline = _compute_graph_metrics(nx_graph)
    rows = []

    # Pre-compute betweenness for reference
    try:
        bc = nx.betweenness_centrality(nx_graph, weight="weight")
    except Exception:
        bc = {n: 0.0 for n in nx_graph.nodes()}

    try:
        dc = dict(nx_graph.degree())
    except Exception:
        dc = {n: 0 for n in nx_graph.nodes()}

    nodes = list(nx_graph.nodes())
    total = len(nodes)

    for i, node in enumerate(nodes):
        # Skip if removing would leave <3 nodes (metrics unstable)
        if total - 1 < 3:
            rows.append({
                "node": node,
                "degree": dc.get(node, 0),
                "betweenness": round(bc.get(node, 0.0), 4),
                "delta_modularity": 0.0,
                "delta_components": 0,
                "delta_avg_shortest_path": 0.0,
                "graph_fragmented": False,
                "criticality_score": 0.0,
            })
            continue

        result = bridge_criticality_sensitivity_analysis(nx_graph, node)
        if "error" in result:
            continue

        d_mod = result["delta_modularity"]
        d_comp = result["delta_connected_components"]
        d_path = result["delta_avg_shortest_path"]
        d_path_f = d_path if isinstance(d_path, float) and np.isfinite(d_path) else 0.0
        fragmented = result["graph_fragmented"]

        # Composite criticality score:
        # High |Δ modularity| + high Δ components + high |Δ avg path| ⇒ critical
        criticality = (
            abs(d_mod) * 1.0
            + d_comp * 2.0
            + abs(d_path_f) * 0.5
        )
        if fragmented:
            criticality += 1.0  # bonus for fragmentation

        rows.append({
            "node": node,
            "degree": dc.get(node, 0),
            "betweenness": round(bc.get(node, 0.0), 4),
            "delta_modularity": d_mod,
            "delta_components": d_comp,
            "delta_avg_shortest_path": d_path_f if isinstance(d_path, float) else np.nan,
            "graph_fragmented": fragmented,
            "criticality_score": round(criticality, 4),
        })

        if progress_callback:
            progress_callback((i + 1) / total)

    df = pd.DataFrame(rows).sort_values(
        "criticality_score", ascending=False
    ).reset_index(drop=True)
    return df


# =====================================================================
# DEMO GRAPH — AgNP Sustainability Concept Graph (matches LaTeX appendix)
# =====================================================================

def build_demo_graph() -> nx.Graph:
    """Build the demo AgNP sustainability concept graph used in the LaTeX appendix."""
    G = nx.Graph()

    # Nodes with category metadata
    nodes = [
        # Sustainability anchors
        ("net_zero",            {"category": "sustainability_anchor",   "is_anchor": True}),
        ("cleaner_production",  {"category": "sustainability_anchor",   "is_anchor": True}),
        ("circular_economy",     {"category": "sustainability_anchor",   "is_anchor": True}),

        # Defect engineering cluster
        ("defect_engineering",   {"category": "method",                  "is_anchor": False}),
        ("stacking_faults",      {"category": "microstructure",          "is_anchor": False}),
        ("twins",                {"category": "microstructure",          "is_anchor": False}),
        ("lattice_distortion",   {"category": "microstructure",          "is_anchor": False}),
        ("diffusion_highways",   {"category": "phenomenon",              "is_anchor": False}),

        # Sintering process cluster
        ("pressureless_sintering", {"category": "process",                "is_anchor": False}),
        ("atomic_diffusion",    {"category": "process",                  "is_anchor": False}),
        ("mass_transport",       {"category": "process",                 "is_anchor": False}),
        ("flux_free_sintering",  {"category": "process",                 "is_anchor": False}),

        # Materials
        ("agnps",                {"category": "material",                "is_anchor": False}),
        ("silver",               {"category": "material",                "is_anchor": False}),
        ("metal",                {"category": "material",                "is_anchor": False}),

        # Properties / metrics
        ("thermal_budget",       {"category": "parameter",               "is_anchor": False}),
        ("energy_efficiency",    {"category": "property",                "is_anchor": False}),
        ("mechanical_yield",     {"category": "property",                "is_anchor": False}),
        ("electrical_conductivity", {"category": "property",              "is_anchor": False}),
        ("thermal_conductivity", {"category": "property",                "is_anchor": False}),
        ("high_strength_bonding", {"category": "property",               "is_anchor": False}),

        # Negative-impact nodes (low-weight edges)
        ("vocs",                 {"category": "phenomenon",              "is_anchor": False}),
        ("residual_stress",      {"category": "property",                "is_anchor": False}),
        ("e_waste",              {"category": "material",                "is_anchor": False}),
        ("micro_voids",          {"category": "microstructure",           "is_anchor": False}),
    ]
    for n, attr in nodes:
        G.add_node(n, **attr)

    # Edges with weights (from LaTeX appendix + ontology causal chains)
    edges = [
        # Defect engineering → materials & microstructure (strongest links)
        ("defect_engineering", "metal",                5.00),
        ("defect_engineering", "stacking_faults",       5.00),
        ("defect_engineering", "twins",                 4.50),
        ("defect_engineering", "lattice_distortion",    4.20),
        ("defect_engineering", "diffusion_highways",    4.80),
        ("defect_engineering", "atomic_diffusion",      4.30),
        ("defect_engineering", "pressureless_sintering", 4.60),

        # Defects → diffusion highways (causal chain)
        ("stacking_faults",    "diffusion_highways",    4.50),
        ("twins",              "diffusion_highways",    4.30),
        ("lattice_distortion", "diffusion_highways",    3.80),
        ("diffusion_highways", "atomic_diffusion",      4.70),
        ("atomic_diffusion",   "mass_transport",        4.20),

        # Sintering → properties & sustainability
        ("pressureless_sintering", "thermal_budget",       5.00),
        ("pressureless_sintering", "circular_economy",     3.50),
        ("pressureless_sintering", "energy_efficiency",    1.40),
        ("pressureless_sintering", "atomic_diffusion",     4.20),
        ("pressureless_sintering", "high_strength_bonding", 3.90),
        ("pressureless_sintering", "electrical_conductivity", 3.20),
        ("pressureless_sintering", "thermal_conductivity", 3.20),
        ("pressureless_sintering", "flux_free_sintering",  3.80),

        # Twins → functional properties (bridge node — matches LaTeX)
        ("twins", "mechanical_yield",        2.80),
        ("twins", "electrical_conductivity", 2.10),
        ("twins", "thermal_budget",          2.10),

        # Cleaner production → sustainability anchors (strong)
        ("cleaner_production", "net_zero",         4.20),
        ("cleaner_production", "circular_economy", 3.80),
        ("flux_free_sintering", "cleaner_production", 4.50),

        # Cleaner production → negative impacts (SUPPRESSION — low weights)
        ("cleaner_production", "vocs",           0.50),
        ("cleaner_production", "residual_stress", 0.70),

        # Energy efficiency → net_zero
        ("energy_efficiency", "net_zero",         3.20),
        ("thermal_budget",    "energy_efficiency", 2.80),

        # Circular economy → e-waste reduction
        ("circular_economy", "e_waste",         2.50),

        # Material hierarchy
        ("agnps",  "silver", 5.00),
        ("silver", "metal", 4.80),

        # Micro-voids linkage
        ("vocs",         "micro_voids", 1.20),
        ("residual_stress", "micro_voids", 1.50),

        # Mechanical yield / conductivity cluster
        ("mechanical_yield",     "high_strength_bonding", 3.00),
        ("electrical_conductivity", "high_strength_bonding", 2.50),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))

    return G


# =====================================================================
# GRAPH LOADING — GraphML / JSON / CSV edge-list
# =====================================================================

def load_graph_from_graphml(file_bytes: bytes) -> nx.Graph:
    """Load a GraphML file from bytes."""
    with io.BytesIO(file_bytes) as f:
        return nx.read_graphml(f)


def load_graph_from_json(file_bytes: bytes) -> nx.Graph:
    """
    Expected JSON format:
    {
      "nodes": [{"id": "net_zero", "category": "anchor"}, ...],
      "edges": [{"source": "defect_engineering", "target": "metal", "weight": 5.0}, ...]
    }
    """
    data = json.loads(file_bytes.decode("utf-8"))
    G = nx.Graph()
    for n in data.get("nodes", []):
        if isinstance(n, dict):
            G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
        else:
            G.add_node(n)
    for e in data.get("edges", []):
        u = e["source"]; v = e["target"]
        w = float(e.get("weight", 1.0))
        G.add_edge(u, v, weight=w)
    return G


def load_graph_from_csv(file_bytes: bytes) -> nx.Graph:
    """
    Expected CSV format (header required):
    source,target,weight
    defect_engineering,metal,5.0
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("CSV must have 'source' and 'target' columns.")
    G = nx.Graph()
    for _, row in df.iterrows():
        w = float(row.get("weight", 1.0)) if "weight" in df.columns else 1.0
        G.add_edge(row["source"], row["target"], weight=w)
    return G


# =====================================================================
# VISUALIZATION HELPERS
# =====================================================================

def render_sis_bar_chart(sis_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plotly bar chart of top-N SIS scores."""
    if sis_df.empty:
        return go.Figure()
    top = sis_df.head(top_n).copy()
    fig = px.bar(
        top,
        x="concept",
        y="SIS",
        color="SIS",
        color_continuous_scale="Viridis",
        title=f"Sustainability Impact Score (SIS) — Top {top_n}",
        labels={"SIS": "SIS ∈ [0, 1]", "concept": "Concept"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafafa",
        height=480,
    )
    return fig


def render_graph_with_matplotlib(G: nx.Graph, highlight_node: Optional[str] = None) -> bytes:
    """Render the graph with matplotlib; returns PNG bytes."""
    if G.number_of_nodes() == 0:
        return b""

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    # Layout — deterministic for reproducibility
    try:
        pos = nx.spring_layout(G, k=1.8, iterations=200, seed=42, weight="weight")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Category colors
    category_colors = {
        "sustainability_anchor": "#4CAF50",
        "method":                "#FF9800",
        "microstructure":        "#795548",
        "phenomenon":            "#9C27B0",
        "process":               "#00BCD4",
        "material":              "#D32F2F",
        "parameter":             "#607D8B",
        "property":              "#3F51B5",
        "general":               "#9E9E9E",
    }

    node_colors = []
    for n in G.nodes():
        cat = G.nodes[n].get("category", "general")
        node_colors.append(category_colors.get(cat, "#9E9E9E"))

    # Edge widths scaled by weight
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    if weights:
        max_w = max(weights) if max(weights) > 0 else 1.0
        edge_widths = [0.4 + (w / max_w) * 2.5 for w in weights]
    else:
        edge_widths = []

    # Node sizes scaled by degree
    degrees = [G.degree(n) for n in G.nodes()]
    max_deg = max(degrees) if degrees else 1
    node_sizes = [200 + (d / max_deg) * 800 for d in degrees]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cccccc", width=edge_widths, alpha=0.6)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#ffffff",
        linewidths=1.5,
    )

    # Highlight a specific node with red border
    if highlight_node and highlight_node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, nodelist=[highlight_node], ax=ax,
            node_color=[category_colors.get(G.nodes[highlight_node].get("category", "general"), "#9E9E9E")],
            node_size=[200 + (G.degree(highlight_node) / max(max_deg, 1)) * 800 + 200],
            edgecolors="#FF1744",
            linewidths=3.0,
        )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=cat.replace("_", " ").title())
        for cat, color in category_colors.items()
        if any(G.nodes[n].get("category") == cat for n in G.nodes())
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(
        f"Concept Graph — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        + (f"  |  Removed: '{highlight_node}'" if highlight_node else ""),
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# Need this import here for the legend patch
import matplotlib.patches as mpatches


def render_criticality_chart(rank_df: pd.DataFrame) -> go.Figure:
    """Plotly chart showing criticality scores ranked descending."""
    if rank_df.empty:
        return go.Figure()
    fig = px.bar(
        rank_df.head(20),
        x="node",
        y="criticality_score",
        color="graph_fragmented",
        color_discrete_map={True: "#FF1744", False: "#3F51B5"},
        title="Top 20 Most Critical Nodes (removal impact)",
        labels={"criticality_score": "Criticality Score", "node": "Concept"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafafa",
        height=480,
    )
    return fig


# =====================================================================
# SIDEBAR — Global configuration
# =====================================================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Sustainability Anchors")
        st.caption("S = {net_zero, cleaner_production, circular_economy} by default")
        default_anchors = "net_zero, cleaner_production, circular_economy"
        anchor_input = st.text_area(
            "Anchor nodes (comma-separated):",
            value=default_anchors,
            height=80,
            help="These are the target nodes S used in the SIS formula. They should exist in your graph.",
        )
        anchors = [a.strip() for a in anchor_input.split(",") if a.strip()]

        st.markdown("---")
        st.subheader("About")
        st.markdown(
            """
            **Sustainability Impact Score (SIS)** — quantifies how close each concept
            is to the sustainability anchor set S, computed via inverse weighted
            shortest-path distances.

            **Bridge-Criticality Sensitivity Analysis** — removes a node, compares
            modularity Q, avg shortest-path ⟨d⟩, and # connected components before
            vs. after, and reports the Δ values.

            **Criticality Ranking** — runs sensitivity analysis on EVERY node and
            ranks them by their structural impact.
            """
        )
        st.markdown("---")
        st.caption("Built on NetworkX · Streamlit · Plotly · Matplotlib")

    return anchors


# =====================================================================
# TAB 1 — GRAPH SETUP
# =====================================================================

def tab_graph_setup(anchors: List[str]):
    st.subheader("1. Graph Setup")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Load a graph:**")
        use_demo = st.button("📈 Use Demo AgNP Graph", type="primary", use_container_width=True)

        uploaded = st.file_uploader(
            "Upload your own graph:",
            type=["graphml", "json", "csv"],
            help="GraphML: nx.write_graphml format. JSON: {nodes:[], edges:[]}. CSV: source,target,weight",
        )

        st.markdown("---")
        st.markdown("**Or build manually:**")
        with st.expander("Add nodes / edges manually"):
            new_node = st.text_input("Add node (with category):", key="new_node_name")
            new_cat = st.selectbox("Category:", [
                "sustainability_anchor", "method", "microstructure", "phenomenon",
                "process", "material", "parameter", "property", "general",
            ], key="new_node_cat")
            if st.button("Add Node", key="add_node_btn") and new_node:
                if st.session_state.graph is None:
                    st.session_state.graph = nx.Graph()
                st.session_state.graph.add_node(new_node, category=new_cat)
                st.success(f"Added node: {new_node}")

            st.markdown("---")
            cols = st.columns(3)
            with cols[0]:
                e_u = st.text_input("Source:", key="edge_u")
            with cols[1]:
                e_v = st.text_input("Target:", key="edge_v")
            with cols[2]:
                e_w = st.number_input("Weight:", min_value=0.01, max_value=10.0,
                                       value=1.0, step=0.1, key="edge_w")
            if st.button("Add Edge", key="add_edge_btn") and e_u and e_v:
                if st.session_state.graph is None:
                    st.session_state.graph = nx.Graph()
                st.session_state.graph.add_edge(e_u, e_v, weight=float(e_w))
                st.success(f"Added edge: {e_u} ↔ {e_v} (w={e_w})")

        if st.button("🗑️ Clear Graph", key="clear_graph_btn"):
            st.session_state.graph = None
            st.session_state.sis_df = None
            st.session_state.rank_df = None
            st.session_state.sensitivity_result = None
            st.rerun()

    with col_right:
        G = st.session_state.get("graph")
        if G is None or G.number_of_nodes() == 0:
            st.info("👈 Click 'Use Demo AgNP Graph' or upload a graph file to begin.")
            st.markdown(
                """
                **Supported file formats:**
                - **GraphML** (`.graphml`) — exported from NetworkX, Gephi, yEd, etc.
                - **JSON** (`.json`) — `{"nodes": [{"id": "..."}], "edges": [{"source": "...", "target": "...", "weight": 1.0}]}`
                - **CSV** (`.csv`) — Edge list with header: `source,target,weight`
                """
            )
            return

        # Display graph summary + visualization
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Nodes", G.number_of_nodes())
        col_b.metric("Edges", G.number_of_edges())
        col_c.metric("Density", f"{nx.density(G):.4f}")
        col_d.metric("Components", nx.number_connected_components(G))

        missing_anchors = [a for a in anchors if a not in G.nodes()]
        if missing_anchors:
            st.warning(f"⚠️ These anchors are NOT in your graph: `{missing_anchors}`. SIS will use only the available anchors.")

        st.markdown("**Graph Visualization:**")
        png_bytes = render_graph_with_matplotlib(G)
        if png_bytes:
            st.image(png_bytes, use_container_width=True)

        # Show node table
        with st.expander("📋 Node & Edge Tables"):
            node_rows = []
            for n, d in G.nodes(data=True):
                node_rows.append({
                    "node": n,
                    "category": d.get("category", "general"),
                    "is_anchor": d.get("is_anchor", False) or n in anchors,
                    "degree": G.degree(n),
                })
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True)

            edge_rows = []
            for u, v, d in G.edges(data=True):
                edge_rows.append({
                    "source": u, "target": v,
                    "weight": round(d.get("weight", 1.0), 3),
                })
            edge_rows.sort(key=lambda r: -r["weight"])
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True)


# =====================================================================
# TAB 2 — SIS ANALYSIS
# =====================================================================

def tab_sis_analysis(anchors: List[str]):
    st.subheader("2. Sustainability Impact Score (SIS)")

    G = st.session_state.get("graph")
    if G is None or G.number_of_nodes() == 0:
        st.info("👈 First load a graph in the 'Graph Setup' tab.")
        return

    st.markdown(
        f"""
        **Formula:**
        ```
        SIS(v) = (1/|S|) · Σ_{{s ∈ S}} 1 / (1 + d_w(v, s))
        ```
        where **S** = `{anchors}` and **d_w(v, s)** is the weighted shortest-path
        distance (edge weight = 1/w, so high-weight edges become short distances).
        """
    )

    if st.button("🧮 Compute SIS", type="primary", use_container_width=True):
        with st.spinner("Computing weighted shortest-path distances for every node..."):
            sis_df = compute_sustainability_impact_score(G, anchors)
        st.session_state.sis_df = sis_df
        st.success(f"✅ Computed SIS for {len(sis_df)} nodes.")

    sis_df = st.session_state.get("sis_df")
    if sis_df is None or sis_df.empty:
        st.info("Click 'Compute SIS' to calculate the Sustainability Impact Score for every node.")
        return

    # Top metrics
    top1 = sis_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Highest SIS", f"{top1['concept']} ({top1['SIS']:.3f})")
    col2.metric("Median SIS", f"{sis_df['SIS'].median():.3f}")
    col3.metric("Mean SIS", f"{sis_df['SIS'].mean():.3f}")
    col4.metric("Nodes Reaching All Anchors",
                int((sis_df["n_anchors_reachable"] == len([a for a in anchors if a in G.nodes()])).sum()))

    # Bar chart
    st.plotly_chart(render_sis_bar_chart(sis_df, top_n=15), use_container_width=True)

    # Full table
    st.markdown("**Full SIS Table** (sorted descending):")
    st.dataframe(sis_df, use_container_width=True)

    # Download
    st.download_button(
        "⬇️ Download SIS results (CSV)",
        data=sis_df.to_csv(index=False).encode("utf-8"),
        file_name="sis_results.csv",
        mime="text/csv",
    )


# =====================================================================
# TAB 3 — BRIDGE-CRITICALITY SENSITIVITY
# =====================================================================

def tab_sensitivity_analysis(anchors: List[str]):
    st.subheader("3. Bridge-Criticality Sensitivity Analysis")

    G = st.session_state.get("graph")
    if G is None or G.number_of_nodes() == 0:
        st.info("👈 First load a graph in the 'Graph Setup' tab.")
        return

    st.markdown(
        """
        Pick a node to remove. The app computes the **before vs. after** metrics:
        - **Modularity Q** — community structure strength
        - **Avg shortest path ⟨d⟩** — global connectivity
        - **# Connected components** — fragmentation indicator
        - **# Nodes / # Edges / Density** — basic size metrics
        """
    )

    # Default: highest-betweenness node
    try:
        bc = nx.betweenness_centrality(G, weight="weight")
        default_node = max(bc, key=bc.get) if bc else None
    except Exception:
        default_node = None

    all_nodes = sorted(G.nodes())
    if default_node and default_node in all_nodes:
        default_idx = all_nodes.index(default_node)
    else:
        default_idx = 0

    selected_node = st.selectbox(
        "Node to remove:",
        all_nodes,
        index=default_idx,
        help="Default: highest-betweenness node (most likely bridge).",
    )

    if st.button("🔬 Run Sensitivity Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Removing '{selected_node}' and recomputing metrics..."):
            result = bridge_criticality_sensitivity_analysis(G, selected_node)
        st.session_state.sensitivity_result = result
        st.session_state.sensitivity_node = selected_node

    result = st.session_state.get("sensitivity_result")
    if not result or "error" in result:
        st.info("Click 'Run Sensitivity Analysis' to compute the before/after metrics.")
        return

    node_removed = result["node_removed"]
    before = result["before"]
    after = result["after"]

    # Header
    st.markdown(f"### Node removed: `{node_removed}`")

    # Side-by-side graph visualization
    st.markdown("**Graph Visualization — Before vs. After Removal:**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Before**")
        png_before = render_graph_with_matplotlib(G)
        if png_before:
            st.image(png_before, use_container_width=True)
    with col_g2:
        st.markdown("**After**")
        G_reduced = G.copy()
        G_reduced.remove_node(node_removed)
        png_after = render_graph_with_matplotlib(G_reduced, highlight_node=None)
        if png_after:
            st.image(png_after, use_container_width=True)

    # Comparison table
    st.markdown("**Metric Comparison Table:**")
    compare_df = pd.DataFrame({
        "Metric": [
            "Modularity Q", "Avg. shortest path ⟨d⟩",
            "# Connected components", "# Communities",
            "Avg. clustering coefficient",
            "# Nodes", "# Edges", "Density",
        ],
        "Before removal": [
            f"{before['modularity']:.4f}",
            f"{before['avg_shortest_path']:.4f}" if np.isfinite(before["avg_shortest_path"]) else "∞",
            before["n_connected_components"],
            before["n_communities"],
            f"{before['avg_clustering']:.4f}",
            before["n_nodes"], before["n_edges"],
            f"{before['density']:.4f}",
        ],
        "After removal": [
            f"{after['modularity']:.4f}",
            f"{after['avg_shortest_path']:.4f}" if np.isfinite(after["avg_shortest_path"]) else "∞",
            after["n_connected_components"],
            after["n_communities"],
            f"{after['avg_clustering']:.4f}",
            after["n_nodes"], after["n_edges"],
            f"{after['density']:.4f}",
        ],
        "Δ (After − Before)": [
            f"{result['delta_modularity']:+.4f}",
            f"{result['delta_avg_shortest_path']:+.4f}" if isinstance(result["delta_avg_shortest_path"], float) else result["delta_avg_shortest_path"],
            f"{result['delta_connected_components']:+d}",
            f"{after['n_communities'] - before['n_communities']:+d}",
            f"{result['delta_avg_clustering']:+.4f}",
            f"{after['n_nodes'] - before['n_nodes']:+d}",
            f"{after['n_edges'] - before['n_edges']:+d}",
            f"{result['delta_density']:+.4f}",
        ],
    })
    st.dataframe(compare_df, use_container_width=True)

    # Fragmentation warning
    if result["graph_fragmented"]:
        st.error(
            f"⚠️ **FRAGMENTATION DETECTED** — Removing `{node_removed}` splits the graph "
            f"from {before['n_connected_components']} → {after['n_connected_components']} "
            f"connected components. This is strong structural evidence that "
            f"`{node_removed}` is a critical bridge node."
        )
    else:
        st.info(
            f"Removing `{node_removed}` does NOT fragment the graph. "
            f"Modularity changed by {result['delta_modularity']:+.4f}, "
            f"avg shortest path by {result['delta_avg_shortest_path']}, "
            f"avg clustering by {result['delta_avg_clustering']:+.4f}."
        )

    # Interpretation guide
    with st.expander("📖 How to interpret these metrics"):
        st.markdown(
            """
            | Metric | What it means | Sign of criticality |
            |---|---|---|
            | **Δ Modularity Q** | Change in community-structure strength | Large negative Δ ⇒ node was holding a community together |
            | **Δ Avg shortest path ⟨d⟩** | Change in global connectivity | Positive Δ ⇒ removal worsened connectivity |
            | **Δ # Components** | Graph fragmentation | Δ > 0 ⇒ **fragmentation** (critical bridge) |
            | **Δ Avg clustering** | Change in local triangle density | Large Δ ⇒ node was a hub of local clusters |
            | **Δ Density** | Edge-density change | Magnitude scales with the node's degree |
            """
        )


# =====================================================================
# TAB 4 — CRITICALITY RANKING
# =====================================================================

def tab_criticality_ranking(anchors: List[str]):
    st.subheader("4. Criticality Ranking — All Nodes Tested")

    G = st.session_state.get("graph")
    if G is None or G.number_of_nodes() == 0:
        st.info("👈 First load a graph in the 'Graph Setup' tab.")
        return

    st.markdown(
        """
        This iterates over **every** node in the graph, removes it, and records
        the impact on modularity, fragmentation, and avg shortest path. The
        **criticality score** is a composite:

        ```
        criticality = |Δ modularity| × 1.0
                    + Δ components × 2.0
                    + |Δ avg path| × 0.5
                    + (1.0 if fragmented else 0.0)
        ```
        """
    )

    if st.button("🏃 Compute Criticality Ranking", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Testing every node...")
        rank_df = compute_criticality_ranking(
            G,
            progress_callback=lambda frac: progress.progress(frac, text=f"Tested {int(frac*100)}% of nodes..."),
        )
        st.session_state.rank_df = rank_df
        progress.progress(1.0, text="✅ Done")
        st.success(f"✅ Ranked {len(rank_df)} nodes by criticality.")

    rank_df = st.session_state.get("rank_df")
    if rank_df is None or rank_df.empty:
        st.info("Click 'Compute Criticality Ranking' to identify the most critical bridge nodes.")
        return

    # Top metrics
    top1 = rank_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Most Critical Node", f"{top1['node']}", f"score={top1['criticality_score']:.3f}")
    col2.metric("Nodes Causing Fragmentation",
                int(rank_df["graph_fragmented"].sum()))
    col3.metric("Max Δ Components",
                int(rank_df["delta_components"].max()))
    col4.metric("Max |Δ Modularity|",
                f"{rank_df['delta_modularity'].abs().max():.3f}")

    # Bar chart
    st.plotly_chart(render_criticality_chart(rank_df), use_container_width=True)

    # Full ranked table
    st.markdown("**Full Criticality Ranking** (sorted descending):")
    st.dataframe(rank_df, use_container_width=True)

    # Top critical nodes detail
    with st.expander("🔬 Top 5 Critical Nodes — Detailed Breakdown"):
        for i, row in rank_df.head(5).iterrows():
            st.markdown(f"#### #{i+1}: `{row['node']}`")
            cols = st.columns(4)
            cols[0].metric("Criticality Score", f"{row['criticality_score']:.3f}")
            cols[1].metric("Degree", row["degree"])
            cols[2].metric("Betweenness", f"{row['betweenness']:.4f}")
            cols[3].metric("Δ Components", f"{row['delta_components']:+d}")
            frag = "⚠️ YES" if row["graph_fragmented"] else "No"
            st.markdown(f"**Fragmented graph:** {frag}  |  **Δ Modularity:** {row['delta_modularity']:+.4f}  |  **Δ Avg Path:** {row['delta_avg_shortest_path']:+.4f}")
            st.markdown("---")

    # Download
    st.download_button(
        "⬇️ Download Criticality Ranking (CSV)",
        data=rank_df.to_csv(index=False).encode("utf-8"),
        file_name="criticality_ranking.csv",
        mime="text/csv",
    )


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    st.title("🕸️ SIS & Bridge-Criticality Calculator")
    st.caption(
        "Compute Sustainability Impact Score (SIS) and Bridge-Criticality "
        "Sensitivity Analysis on any weighted concept graph."
    )

    # Initialize session state
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "sis_df" not in st.session_state:
        st.session_state.sis_df = None
    if "rank_df" not in st.session_state:
        st.session_state.rank_df = None
    if "sensitivity_result" not in st.session_state:
        st.session_state.sensitivity_result = None

    anchors = render_sidebar()

    # Handle demo button
    if st.session_state.get("use_demo_trigger"):
        st.session_state.graph = build_demo_graph()
        st.session_state.sis_df = None
        st.session_state.rank_df = None
        st.session_state.sensitivity_result = None
        st.session_state.use_demo_trigger = False

    # Re-check buttons at top of main loop
    if st.button("📈 Use Demo AgNP Graph", key="demo_btn_main", type="primary"):
        st.session_state.graph = build_demo_graph()
        st.session_state.sis_df = None
        st.session_state.rank_df = None
        st.session_state.sensitivity_result = None
        st.rerun()

    # Handle file upload
    uploaded = st.file_uploader(
        "Upload your own graph:",
        type=["graphml", "json", "csv"],
        key="main_uploader",
        help="GraphML: nx.write_graphml format. JSON: {nodes:[], edges:[]}. CSV: source,target,weight",
    )
    if uploaded is not None:
        try:
            ext = uploaded.name.split(".")[-1].lower()
            file_bytes = uploaded.read()
            if ext == "graphml":
                G = load_graph_from_graphml(file_bytes)
            elif ext == "json":
                G = load_graph_from_json(file_bytes)
            elif ext == "csv":
                G = load_graph_from_csv(file_bytes)
            else:
                st.error(f"Unsupported file type: {ext}")
                G = None
            if G is not None:
                st.session_state.graph = G
                st.session_state.sis_df = None
                st.session_state.rank_df = None
                st.session_state.sensitivity_result = None
                st.success(f"✅ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Graph Setup",
        "2️⃣ SIS Analysis",
        "3️⃣ Sensitivity Analysis",
        "4️⃣ Criticality Ranking",
    ])

    with tab1:
        tab_graph_setup(anchors)
    with tab2:
        tab_sis_analysis(anchors)
    with tab3:
        tab_sensitivity_analysis(anchors)
    with tab4:
        tab_criticality_ranking(anchors)


if __name__ == "__main__":
    main()
