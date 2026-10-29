# Sustainability Impact Score

[![meaningtowords](https://img.shields.io/badge/vis-v1.0-orange)](https://sustainability-impact-score-visualization1.streamlit.app/) ( Concepts with higher score are nearer to sustainability anchors, can construct charts only with the datasets available within the app file)

[![meaningtowords](https://img.shields.io/badge/vis-v2.0-turquoise)](https://sustainability-impact-score-visualization2.streamlit.app/) ( Concepts with higher score are nearer to sustainability anchors, can construct chart  with the datasets available within the app file or by locating the csv file)

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
