
# Metrics SIS and BCSA values are obtained from compute sustainability scores code

# Sustainability Impact Score
 
[![meaningtowords](https://img.shields.io/badge/sisvis-v1.0-orange)](https://sustainability-impact-score-visualization1.streamlit.app/) ( Concepts with higher score are nearer to sustainability anchors, can construct charts only with the datasets available within the app file)

[![meaningtowords](https://img.shields.io/badge/sisvis-v2.0-turquoise)](https://sustainability-impact-score-visualization2.streamlit.app/) ( Concepts with higher score are nearer to sustainability anchors, can construct chart  with the datasets available within the app file or by locating the csv file)

# Sustainability Impact Score (SIS) Computation

## Overview

The **Sustainability Impact Score (SIS)** is a graph-topological metric introduced to **quantitatively rank** every concept node in a concept graph based on its proximity—in a weighted-shortest-path sense—to specified anchor sustainability outcomes. The SIS converts qualitative assertions (e.g., *"this defect contributes to sustainability"*) into a continuous, comparable numerical score $S \in [0, 1]$ that can be reported, ranked, and cross-validated against experimental observations (e.g., sintering onset temperature thresholds).

---

## 1. Mathematical Definition

For any concept node $v$ in graph $G$, the SIS is defined as:

$$\mathrm{SIS}(v) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \frac{1}{1 + d_w(v, s)}$$

### Terminology & Notation

| Symbol | Definition |
| :--- | :--- |
| $\mathcal{S}$ | The set of **sustainability anchor nodes** (e.g., `{"net_zero", "cleaner_production", "circular_economy"}`). |
| $\|\mathcal{S}\|$ | Total number of anchor nodes ($|\mathcal{S}| = 3$ in the default pipeline). |
| $d_w(v, s)$ | The **weighted shortest-path distance** from target node $v$ to anchor node $s$. |
| $\frac{1}{1 + d_w}$ | Inverse-distance kernel mapping distance $d_w = 0 \to 1$ (direct hit) and $d_w \to \infty \to 0$ (unreachable). |

The resulting SIS represents the **average inverse-distance** of $v$ across all defined anchors:
* **$\text{SIS} = 1.0$**: Node $v$ connects via a direct, maximum-weight edge to every anchor node.
* **$\text{SIS} = 0.0$**: Node $v$ has no path to any anchor node.

---

## 2. Anchor Set Selection ($\mathcal{S}$)

The standard baseline implementation uses three core anchor nodes:

1. **`net_zero`**: Represents global net-zero emission targets (Scope 1 & 2 GHG reduction).
2. **`cleaner_production`**: Represents the elimination of volatile organic compounds (VOCs), fluxes, and hazardous processing agents.
3. **`circular_economy`**: Represents lifecycle longevity, recyclability, and electronic waste minimization principles.

These anchors map directly onto three foundational pillars of sustainable processing:
* *Energy Efficiency* $\rightarrow$ `net_zero`
* *Flux-Free / Zero-VOC Processing* $\rightarrow$ `cleaner_production`
* *Lifecycle / End-of-Life Recycling* $\rightarrow$ `circular_economy`

---

## 3. Computation & Weight Inversion Method

In typical graph structures, edge weights $w(u, v)$ represent **relationship strength** (e.g., $w \in [0.70, 5.00]$ where higher values indicate stronger domain connection). However, standard shortest-path algorithms (such as Dijkstra's algorithm) treat edge weights as **physical distances** where higher values indicate longer paths.

To align shortest-path pathfinding with relationship strength, edge weights are inverted prior to distance computation:

$$w_{\text{dist}}(u, v) = \frac{1}{w(u, v)}$$

* An edge with strength $w = 5.00$ translates to a distance of $w_{\text{dist}} = 0.20$ (short path).
* An edge with strength $w = 0.70$ translates to a distance of $w_{\text{dist}} = 1.43$ (long path).

### Algorithmic Logic (Pseudocode)

```text
Input:  NetworkX Graph G with edge attribute "weight"
        Anchor Set S = {"net_zero", "cleaner_production", "circular_economy"}

1. Construct distance graph G_dist:
    For each edge (u, v) with weight w in G.edges():
        if w > 0:
            G_dist.add_edge(u, v, weight = 1.0 / w)

2. For each node v in G.nodes():
    distances = []
    For each anchor s in S:
        try:
            d = shortest_path_length(G_dist, source=v, target=s, weight="weight")
        except (NoPath, NodeNotFound):
            d = infinity
        distances.append(d)

    # Convert distances to inverse-distance score
    sis_values = [1 / (1 + d) if d is finite else 0.0 for d in distances]

    # Compute mean across anchors
    SIS(v) = mean(sis_values)

3. Return sorted dictionary of (v, SIS(v)) descending.
```

---

## 4. Analytical Case Study & Reference Rankings

Applying the SIS metric to microstructural defect nodes yields relative rankings that correspond with observed experimental values:

| Concept Node | Score ($\mathrm{SIS}$) | Dominant Graph Route |
| :--- | :--- | :--- |
| **`stacking_faults`** | **$\approx 0.65$** | Secondary path via `diffusion_highways` |
| **`twins`** | **$\approx 0.58$** | Direct path via `diffusion_highways` intermediate node |
| **`lattice_distortion`** | **$\approx 0.49$** | Indirect connectivity with longer weighted paths |

### Physical Interpretation 
The graph-topological ranking (\mathrm{stacking\_faults} > $\mathrm{twins} >  \mathrm{lattice\_distortion}$) directly mirrors experimental characterization data: 
* **Stacking faults** exhibit intermediate local stresses ($\approx 64\text{ GPa}$) and moderate sintering onset reductions.
* **Twin boundaries** correlate with peak von Mises stress ($\approx 111.6\text{ GPa}$) and maximal sintering temperature reduction (down to $94^\circ\text{C}$).
* **Lattice distortions** yield lower stress concentration ($\approx 30.3\text{ GPa}$) and smaller operational impact.

---

## 5. Execution Instructions

### Python API Integration

To compute SIS directly on a NetworkX graph structure using `agnp_sustainability_concept_graph python app`:

```python
import networkx as nx
from agnp_sustainability_concept_graph import compute_sustainability_impact_score

# 1. Load concept graph
G = nx.read_graphml("agnp_concept_graph.graphml")

# 2. Define sustainability target anchors
anchors = ["net_zero", "cleaner_production", "circular_economy"]

# 3. Calculate score
sis_df = compute_sustainability_impact_score(G, anchors)

# 4. Display top concepts
print(sis_df.head(10))
```

#### Output Schema

The resulting Dataframe yields:
* `concept`: Concept node identifier string.
* `SIS`: Float value $S \in [0, 1]$.
* `n_anchors_reachable`: Number of target anchors reachable from the given node ($0 \le n \le 3$).

---

## 6. Significance & Key Benefits

1. **Scalar Aggregation**: Converts complex, multi-hop relationship networks into a single, comparative continuous scalar score per concept.
2. **Multi-Hop Path Propagation**: Captures long-range downstream influence across intermediate operational, material, and structural nodes.
3. **Cross-Validation**: Provides topological verification of experimental and physical property hierarchies purely through graph path metrics.


# Bridge-Criticality Sensitivity Analysis

[![meaningtowords](https://img.shields.io/badge/bcsavis-v1.0-turquoise)](https://bcsa-metrics1.streamlit.app/) ( can construct chart  with the datasets available within the app file or by locating the csv file)

[![meaningtowords](https://img.shields.io/badge/bcsavis-v2.0-turquoise)](https://bcsa-metrics2.streamlit.app/) ( can construct chart  with the datasets available within the app file or by locating the csv file, enhancement of the customization of the figure)

## 1. What is Bridge-Criticality Sensitivity Analysis?

The **Bridge-Criticality Sensitivity Analysis** is a graph-topological diagnostic that quantifies **how structurally important a single node (concept) is to the overall integrity of the concept graph** by measuring what happens to the graph when that node is *removed*.

In network science, a **bridge node** is a node whose removal causes:
1. The graph to **fragment** into multiple disconnected components, or
2. A sharp increase in **average shortest-path length** (the graph becomes harder to traverse), or
3. A drop in **modularity** (community structure weakens).

For the AgNP sustainability concept graph, this analysis directly answers: *"If we removed the concept `defect_engineering` from the manuscript's knowledge network, would the link between materials science and sustainability outcomes survive, or would the network collapse?"*

This converts the qualitative claim that *"`defect_engineering` is a critical bridge"* into a quantitative, before-vs-after measurement.

---

## 2. The Mathematical Definition

For a given node $v$ in graph $G$, the analysis computes three graph-topology metrics **before** removal ($G$) and **after** removal ($G' = G \setminus \{v\}$), then takes the **delta** ($\Delta$):

### Metric 1: Modularity $Q$

$$
Q(G) = \frac{1}{2m} \sum_{i,j \in V} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

where $A$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total edge weight, and $c_i$ is the community of node $i$. Higher $Q$ = stronger community structure.

$$\Delta Q = Q(G') - Q(G)$$

### Metric 2: Average shortest-path length $\langle d \rangle$

$$
\langle d \rangle = \frac{1}{|V|(|V|-1)} \sum_{i \neq j \in V} d(i, j)
$$

where $d(i, j)$ is the unweighted shortest-path distance between nodes $i$ and $j$ (computed on the largest connected component if the graph is fragmented).

$$\Delta \langle d \rangle = \langle d \rangle(G') - \langle d \rangle(G)$$

### Metric 3: Number of connected components

$$
\Delta C = C(G') - C(G)
$$

If $\Delta C > 0$, the graph has **fragmented** — strong evidence that $v$ is a critical bridge.

### The Result Dictionary

The function returns:

| Key | Meaning |
|---|---|
| `node_removed` | The name of the node that was removed |
| `before` | Dict of metrics computed on the original graph $G$ |
| `after` | Dict of metrics computed on the reduced graph $G \setminus \{v\}$ |
| `delta_modularity` | $\Delta Q = Q(G') - Q(G)$ |
| `delta_avg_shortest_path` | $\Delta \langle d \rangle$ (returns `"∞"` if unreachable) |
| `delta_connected_components` | $\Delta C$ — **if > 0, the graph fragmented** |
| `delta_avg_clustering` | Change in average clustering coefficient |
| `delta_density` | Change in graph density |
| `graph_fragmented` | `True` if $\Delta C > 0$ (the critical signal) |

---

## 3. How to Interpret the Deltas

| Metric | Negative $\Delta$ | Zero $\Delta$ | Positive $\Delta$ |
|---|---|---|---|
| **$\Delta$ Modularity** | Removed node was holding a community together | No effect | Community structure got sharper (the node was a "boundary spanner") |
| **$\Delta$ Avg shortest path** | Removal improved connectivity (rare) | Node was peripheral | Removal worsened connectivity — node was on many shortest paths |
| **$\Delta$ Connected components** | (impossible) | Graph stayed connected | **Graph fragmented!** The node was a critical bridge. |
| **$\Delta$ Avg clustering** | Node was a hub of triangles | — | Node was suppressing local clustering |

The most important signal is `graph_fragmented = True` (i.e., $\Delta C > 0$). When this fires, the analysis emits the ⚠️ warning in the Streamlit app:

> *"Removing `defect_engineering` fragments the graph into 3 components — strong topological evidence of a critical bridge."*

---

## 4. Computing the Metrics — The `_compute_graph_metrics()` Helper

```python
def _compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute modularity, avg shortest path, components, etc."""
    metrics = {}
    # 1. Modularity via greedy modularity communities
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(G))
        metrics["modularity"] = float(community.modularity(G, partition)) if len(partition) > 1 else 0.0
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0

    # 2. Avg shortest path (on largest connected component if fragmented)
    try:
        if G.number_of_nodes() == 0:
            metrics["avg_shortest_path"] = np.nan
        elif nx.is_connected(G):
            metrics["avg_shortest_path"] = float(nx.average_shortest_path_length(G))
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            metrics["avg_shortest_path"] = float(nx.average_shortest_path_length(G.subgraph(largest_cc)))
    except Exception:
        metrics["avg_shortest_path"] = np.nan

    # 3. Basic counts
    metrics["n_connected_components"] = nx.number_connected_components(G)
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0
    try:
        metrics["avg_clustering"] = float(nx.average_clustering(G))
    except Exception:
        metrics["avg_clustering"] = 0.0
    return metrics
```

Key implementation notes:
- **Modularity** uses `greedy_modularity_communities()` — a fast, deterministic community-detection algorithm.
- **Avg shortest path** falls back to the **largest connected component** if the graph is already fragmented (so we always get a finite number, not `NaN`).
- All metrics are wrapped in `try/except` to handle edge cases (empty graph, single node, disconnected).

---

## 5. The Main Function — `bridge_criticality_sensitivity_analysis()`

```python
def bridge_criticality_sensitivity_analysis(nx_graph, node_to_remove):
    """Quantify structural impact of removing a specific node."""
    if node_to_remove not in nx_graph.nodes():
        return {"error": f"Node '{node_to_remove}' not in graph."}

    # BEFORE: metrics on the original graph
    before = _compute_graph_metrics(nx_graph)

    # AFTER: metrics on the graph with the node removed
    G_reduced = nx_graph.copy()
    G_reduced.remove_node(node_to_remove)
    after = _compute_graph_metrics(G_reduced)

    # DELTAS
    delta_modularity = after["modularity"] - before["modularity"]
    delta_avg_path = (after["avg_shortest_path"] - before["avg_shortest_path"]
                      if np.isfinite(after["avg_shortest_path"]) and np.isfinite(before["avg_shortest_path"])
                      else np.nan)
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
```

**Pseudocode:**
```
1. Check that the node exists in the graph; if not, return error.
2. Compute metrics on the original graph G → "before" dict.
3. Make a copy of G; remove the target node → G'.
4. Compute metrics on G' → "after" dict.
5. Compute deltas: Δ = after - before.
6. Set graph_fragmented = (Δ components > 0).
7. Return the result dictionary.
```

---

## 6. Worked Example: `defect_engineering`

Using the manuscript's 40-node concept graph:

**Before removal** (original 40-node graph):
- Modularity $Q = 0.4332$
- $\langle d \rangle = 2.6282$
- # Connected components = 1 (graph is fully connected)
- # Communities = 4
- Density = 0.123

**After removing `defect_engineering`:**
- Modularity $Q' = 0.5810$ (rose — community structure got sharper because the bridging node is gone)
- $\langle d \rangle' = 1.7333$ (fell — but only because the graph shrank to a smaller connected component)
- # Connected components = **3** (fragmented!)
- # Communities = 5

**Deltas:**
- $\Delta Q = +0.1478$ (positive = communities became more distinct)
- $\Delta \langle d \rangle = -0.8949$
- $\Delta C = +2$ (1 → 3 components)
- `graph_fragmented = True` ⚠️

**Conclusion:** Removing `defect_engineering` fragments the graph into 3 components — strong quantitative evidence that `defect_engineering` is the critical bridge connecting materials science to sustainability outcomes. This corroborates the manuscript's qualitative claim (§6) that defect engineering "shifts the focus of sustainable metallurgy from process-level optimizations to material-level design."

---

## 7. The Composite Criticality Score

For ranking *all* nodes by their criticality, the companion function `compute_criticality_ranking()` iterates over every node, removes it, and computes a **composite criticality score**:

$$
\text{criticality}(v) = 1.0 \cdot |\Delta Q(v)| + 2.0 \cdot \Delta C(v) + 0.5 \cdot |\Delta \langle d \rangle(v)| + \begin{cases} 1.0 & \text{if fragmented} \\ 0 & \text{otherwise} \end{cases}
$$

Higher score = more critical. The fragmenting nodes get a +1.0 bonus because they're the most structurally damaging.

### Top 5 Most Critical Nodes (from the 44-node graph)

| Rank | Node | Degree | Betweenness | $\Delta$ Modularity | $\Delta$ Components | Criticality Score |
|---|---|---|---|---|---|---|
| 1 | `pressureless_sintering` | 9 | 0.548 | +0.094 | +1 | **5.60** |
| 2 | `cleaner_production` | 5 | 0.239 | +0.010 | +1 | **3.20** |
| 3 | `silver` | 2 | 0.083 | -0.033 | +1 | **3.16** |
| 4 | `metal` | 2 | 0.159 | -0.002 | +1 | **3.14** |
| 5 | `defect_engineering` | 7 | 0.320 | +0.148 | +1 | **3.11** |

**Key insight:** Four nodes fragment the graph if removed — these are the manuscript's "load-bearing" concepts. The SIS analysis ranks concepts by proximity to sustainability anchors; the criticality analysis ranks them by structural importance to the knowledge network. The two metrics are complementary.

---

## 8. How to Compute It

### Option A — Automatic (recommended)

In the Streamlit app (`agnp_sustainability_concept_graph.py`), the pipeline already:

1. Builds the concept graph (including the 40-node version used in Fig. 12).
2. Picks the **highest-betweenness node** as the default bridge target.
3. Runs `bridge_criticality_sensitivity_analysis()` on it.
4. Stores the result in `st.session_state.sensitivity_result`.
5. Displays a Before/After/Δ comparison table in the **Advanced Analytics → "Bridge-Criticality Sensitivity Analysis"** expander.

Plus an interactive dropdown lets you test any other node without rebuilding the graph.

### Option B — Manual (one-off computation)

```python
import networkx as nx
from agnp_sustainability_concept_graph import (
    bridge_criticality_sensitivity_analysis,
    compute_criticality_ranking,
)

# Load your saved graph
G = nx.read_graphml("agnp_concept_graph.graphml")

# Test a single node
result = bridge_criticality_sensitivity_analysis(G, "defect_engineering")
print(f"Node removed: {result['node_removed']}")
print(f"Δ modularity:     {result['delta_modularity']:+.4f}")
print(f"Δ avg path:       {result['delta_avg_shortest_path']}")
print(f"Δ components:      {result['delta_connected_components']:+d}")
print(f"Fragmented?       {result['graph_fragmented']}")

# Rank ALL nodes by criticality (composite score)
rank_df = compute_criticality_ranking(G)
print(rank_df.head(10))  # top 10 most critical nodes
```

---


