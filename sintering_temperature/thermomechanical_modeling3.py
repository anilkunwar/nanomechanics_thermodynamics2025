import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import matplotlib.patches as mpatches

# =================================================
# Page config
# =================================================
st.set_page_config(
    page_title="Ag Nanoparticles: Stress–Sintering",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sintering Temperature vs Hydrostatic Stress in Ag Nanoparticles")
st.markdown("**Interactive, publication-quality visualization with full style control**")

# =================================================
# Sidebar: Physical parameters
# =================================================
st.sidebar.header("PropertyParams")

Ts0 = st.sidebar.slider("Baseline Tₛ(0) [K]", 500, 700, 623)
Qa_kJ = st.sidebar.slider("Activation energy Qₐ [kJ/mol]", 70.0, 120.0, 90.0, step=5.0)
Omega_m3 = st.sidebar.slider(
    "Atomic volume Ω (×10⁻²⁹ m³)", 8.0, 12.0, 10.0
) * 1e-29
sigma_max_GPa = st.sidebar.slider("Max |σₕ| [GPa]", 1.0, 10.0, 5.0)

# =================================================
# Sidebar: Curve styling
# =================================================
st.sidebar.header("🎨 Curve Styling")

main_color = st.sidebar.color_picker("Main curve color", "#1f77b4")
baseline_color = st.sidebar.color_picker("Baseline & arrow color", "#d62728")
marker_color = st.sidebar.color_picker("Zero-stress marker color", "#000000")
linewidth = st.sidebar.slider("Main line width", 1.0, 5.0, 3.0)
linestyle = st.sidebar.selectbox("Main line style", ["-", "--", "-.", ":"])
marker_size = st.sidebar.slider("Marker size", 5, 20, 12)

# =================================================
# Sidebar: Fonts and layout
# =================================================
st.sidebar.header("🔤 Fonts & Layout")

font_family = st.sidebar.selectbox(
    "Font family", ["DejaVu Sans", "Times New Roman", "Arial", "Helvetica"]
)
label_size = st.sidebar.slider("Axis label font size", 10, 20, 16)
tick_size = st.sidebar.slider("Tick font size", 8, 16, 13)
title_size = st.sidebar.slider("Title font size", 12, 24, 20)
legend_size = st.sidebar.slider("Legend font size", 8, 16, 13)

# =================================================
# Sidebar: Axes, ticks, and box
# =================================================
st.sidebar.header("📐 Axes & Ticks")

show_grid = st.sidebar.checkbox("Show grid", True)
grid_alpha = st.sidebar.slider("Grid transparency", 0.05, 0.6, 0.3)
grid_linestyle = st.sidebar.selectbox("Grid line style", ["-", "--", "-.", ":"], index=1)
show_box = st.sidebar.checkbox("Show axes box", True)
minor_ticks = st.sidebar.checkbox("Enable minor ticks", True)

# =================================================
# Cached physics model
# =================================================
@st.cache_data
def calculate_Ts(sigma_Pa, Ts0, Qa_kJ, Omega_m3):
    Qa_J_atom = Qa_kJ * 1000 / 6.022e23
    delta_Q = Omega_m3 * sigma_Pa
    Q_eff = Qa_J_atom - delta_Q
    Ts_eff = Ts0 * (Q_eff / Qa_J_atom)
    return max(Ts_eff, 300.0)

# =================================================
# Data generation
# =================================================
sigma_plot = np.logspace(-6, np.log10(sigma_max_GPa * 2), 500) * 1e9
Ts_plot = np.array(
    [calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in sigma_plot]
)

Ts_min = Ts_plot[-1]
delta_T = Ts0 - Ts_min

# =================================================
# Matplotlib global style
# =================================================
plt.rcParams.update({
    "font.family": font_family,
    "axes.labelsize": label_size,
    "axes.titlesize": title_size,
    "xtick.labelsize": tick_size,
    "ytick.labelsize": tick_size,
    "legend.fontsize": legend_size,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# =================================================
# Plot with enhanced aesthetics
# =================================================
fig, ax = plt.subplots(figsize=(12, 7))  # Larger, more cinematic figure

# Main curve
ax.semilogx(
    sigma_plot / 1e9,
    Ts_plot,
    color=main_color,
    linewidth=linewidth,
    linestyle=linestyle,
    label=r"$T_s(|\sigma_h|)$",
    zorder=3
)

# Zero-stress marker
ax.plot(
    1e-6,
    Ts0,
    marker="o",
    color=marker_color,
    markersize=marker_size,
    label=r"$T_s(0)$",
    zorder=4
)

# Baseline
ax.axhline(
    Ts0,
    color=baseline_color,
    linestyle="--",
    linewidth=2,
    label="Baseline $T_s(0)$",
    zorder=2
)

# Double-headed arrow for ΔT
arrow = mpatches.FancyArrowPatch(
    (sigma_max_GPa, Ts0),
    (sigma_max_GPa, Ts_min),
    mutation_scale=20,
    arrowstyle='<->',
    color=baseline_color,
    linewidth=2,
    zorder=5
)
ax.add_patch(arrow)

# ΔT annotation
ax.text(
    sigma_max_GPa * 1.2,
    (Ts0 + Ts_min) / 2,
    rf"$\Delta T_s = {delta_T:.0f}\,\mathrm{{K}}$",
    color=baseline_color,
    fontsize=label_size,
    verticalalignment='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=baseline_color, alpha=0.7)
)

# Labels and title
ax.set_xlabel(r"$|\sigma_h|$ [GPa] (log scale)", labelpad=10)
ax.set_ylabel(r"Sintering temperature $T_s$ [K]", labelpad=10)
ax.set_title("Stress-Induced Reduction in Ag Nanoparticle Sintering Temperature", pad=20)

# Tick formatting
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis='x', style='plain')  # Avoid scientific notation on x-axis

if minor_ticks:
    ax.minorticks_on()

# Grid
if show_grid:
    ax.grid(True, which="major", linestyle=grid_linestyle, alpha=grid_alpha, zorder=1)
    if minor_ticks:
        ax.grid(True, which="minor", linestyle=":", alpha=grid_alpha * 0.6, zorder=1)

# Spines (box)
for spine in ax.spines.values():
    spine.set_visible(show_box)

# Legend
legend = ax.legend(
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    edgecolor='gray',
    loc='upper right'
)
legend.get_frame().set_facecolor('white')

# Tight layout with padding
plt.tight_layout(pad=2.5)

# Render plot
st.pyplot(fig, use_container_width=True)

# =================================================
# Metrics
# =================================================
st.markdown("### 🔍 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Baseline $T_s$", f"{Ts0:.0f} K")
col2.metric("Max $\\Delta T_s$", f"{delta_T:.0f} K", delta=f"-{100 * delta_T / Ts0:.0f}%")
col3.metric("At $|\\sigma_h|$", f"{sigma_max_GPa:.1f} GPa")

# =================================================
# Prediction table
# =================================================
st.markdown("### 📊 Predictions at Key Stress Levels")

stresses_GPa = np.logspace(-1, np.log10(sigma_max_GPa), 6)
stresses_Pa = stresses_GPa * 1e9
Ts_values = [calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in stresses_Pa]

prediction_data = {
    "|σₕ| [GPa]": [f"{s:.2f}" for s in stresses_GPa],
    "$T_s$ [K]": [f"{t:.0f}" for t in Ts_values],
    "$ΔT_s$ [K]": [f"{Ts0 - t:.0f}" for t in Ts_values],
    "Reduction [%]": [f"{100 * (Ts0 - t) / Ts0:.0f}" for t in Ts_values],
}

st.table(prediction_data)

# =================================================
# Physics note
# =================================================
st.markdown(r"""
### 🧪 Physics Insight

The stress-modified sintering temperature follows:
\[
T_s(|\sigma_h|) \approx T_s(0)\left(1 - \frac{\Omega |\sigma_h|}{Q_a}\right)
\]

**Interpretation**
- Twin- or defect-induced hydrostatic stresses of **1–5 GPa** can lower $T_s$ by **hundreds of Kelvin**.
- Extreme defect stresses (**5–10 GPa**) enable **ultra-low-temperature sintering**, critical for flexible electronics and nanomanufacturing.
""")
