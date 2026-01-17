import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, ScalarFormatter
import matplotlib.patches as mpatches

# =================================================
# Page Configuration
# =================================================
st.set_page_config(
    page_title="Ag Nanoparticles: Stress–Sintering",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sintering Temperature vs Hydrostatic Stress in Ag Nanoparticles")
st.markdown("**Interactive, publication-quality visualization with full tick and axis control**")

# =================================================
# Sidebar: Physical Parameters (Wider Ranges)
# =================================================
st.sidebar.header("PropertyParams")

Ts0 = st.sidebar.slider("Baseline $T_s(0)$ [K]", 300, 900, 623, step=1)
Qa_kJ = st.sidebar.slider("Activation energy $Q_a$ [kJ/mol]", 50.0, 150.0, 90.0, step=1.0)
Omega_m3 = st.sidebar.slider(
    "Atomic volume $\\Omega$ (×10⁻²⁹ m³)", 5.0, 20.0, 10.0, step=0.5
) * 1e-29
sigma_max_GPa = st.sidebar.slider("Max $|\\sigma_h|$ [GPa]", 0.1, 20.0, 5.0, step=0.1)

# =================================================
# Sidebar: Figure Dimensions
# =================================================
st.sidebar.header("📐 Figure Size")

fig_width = st.sidebar.slider("Figure width (inches)", 8.0, 24.0, 12.0, step=0.5)
fig_height = st.sidebar.slider("Figure height (inches)", 4.0, 14.0, 7.0, step=0.5)
dpi = st.sidebar.selectbox("Resolution (DPI)", [100, 150, 200, 300, 400], index=1)

# Estimate tick font size
estimated_tick_size = max(8, min(20, int(0.8 * (fig_height + fig_width) / 2)))
st.sidebar.caption(f"💡 Suggested tick font size: ~{estimated_tick_size}")

# =================================================
# Sidebar: Curve Styling
# =================================================
st.sidebar.header("🎨 Curve Styling")

main_color = st.sidebar.color_picker("Main curve color", "#1f77b4")
baseline_color = st.sidebar.color_picker("Baseline & arrow color", "#d62728")
marker_color = st.sidebar.color_picker("Zero-stress marker color", "#000000")
linewidth = st.sidebar.slider("Main line width", 1.0, 6.0, 3.0, step=0.1)
linestyle = st.sidebar.selectbox("Main line style", ["-", "--", "-.", ":"])
marker_size = st.sidebar.slider("Marker size", 5, 25, 12, step=1)

# =================================================
# Sidebar: Fonts and Layout
# =================================================
st.sidebar.header("🔤 Fonts & Layout")

font_family = st.sidebar.selectbox(
    "Font family",
    ["DejaVu Sans", "Times New Roman", "Arial", "Helvetica", "serif", "sans-serif"]
)
label_size = st.sidebar.slider("Axis label font size", 10, 28, 16, step=1)
tick_size = st.sidebar.slider("Tick label font size", 6, 22, estimated_tick_size, step=1)
title_size = st.sidebar.slider("Title font size", 12, 32, 20, step=1)
legend_size = st.sidebar.slider("Legend font size", 8, 20, 13, step=1)

# =================================================
# Sidebar: Axes, Ticks, and Box
# =================================================
st.sidebar.header("AxisSize & Spines")

show_grid = st.sidebar.checkbox("Show grid", True)
grid_alpha = st.sidebar.slider("Grid transparency", 0.05, 0.8, 0.3, step=0.05)
grid_linestyle = st.sidebar.selectbox("Grid line style", ["-", "--", "-.", ":"], index=1)
show_box = st.sidebar.checkbox("Show axes box", True)
spine_linewidth = st.sidebar.slider("Axes box (spine) thickness", 0.5, 4.0, 1.2, step=0.1)

# Tick styling — MAJOR
st.sidebar.subheader("📏 Major Ticks")
major_tick_length_x = st.sidebar.slider("X major tick length", 2.0, 12.0, 6.0, step=0.5)
major_tick_width_x = st.sidebar.slider("X major tick width", 0.5, 3.0, 1.2, step=0.1)
major_tick_length_y = st.sidebar.slider("Y major tick length", 2.0, 12.0, 6.0, step=0.5)
major_tick_width_y = st.sidebar.slider("Y major tick width", 0.5, 3.0, 1.2, step=0.1)
major_tick_color = st.sidebar.color_picker("Major tick color", "#000000")

# Tick styling — MINOR
st.sidebar.subheader("📏 Minor Ticks")
minor_ticks = st.sidebar.checkbox("Enable minor ticks", True)
minor_tick_length_x = st.sidebar.slider("X minor tick length", 1.0, 8.0, 3.0, step=0.5)
minor_tick_width_x = st.sidebar.slider("X minor tick width", 0.3, 2.0, 0.8, step=0.1)
minor_tick_length_y = st.sidebar.slider("Y minor tick length", 1.0, 8.0, 3.0, step=0.5)
minor_tick_width_y = st.sidebar.slider("Y minor tick width", 0.3, 2.0, 0.8, step=0.1)
minor_tick_color = st.sidebar.color_picker("Minor tick color", "#666666")

# X-axis label format
st.sidebar.subheader("↔️ X-Axis Label Format")
x_label_format = st.sidebar.radio(
    "X-axis labels",
    ("Decimal (0.01, 1, 10)", "Scientific (10⁻², 10⁰, 10¹)"),
    index=1
)

# Legend positioning
st.sidebar.header("🔖 Legend")
legend_options = {
    "Auto (best)": None,
    "Upper right": "upper right",
    "Upper left": "upper left",
    "Lower left": "lower left",
    "Lower right": "lower right",
    "Right": "right",
    "Center left": "center left",
    "Center right": "center right",
    "Lower center": "lower center",
    "Upper center": "upper center",
    "Center": "center"
}
legend_choice = st.sidebar.selectbox("Legend location", list(legend_options.keys()), index=0)
legend_loc = legend_options[legend_choice]

# =================================================
# Physics Model (Cached)
# =================================================
@st.cache_data
def calculate_Ts(sigma_Pa, Ts0, Qa_kJ, Omega_m3):
    Qa_J_atom = Qa_kJ * 1000.0 / 6.022e23
    delta_Q = Omega_m3 * sigma_Pa
    Q_eff = Qa_J_atom - delta_Q
    Ts_eff = Ts0 * (Q_eff / Qa_J_atom)
    return max(Ts_eff, 300.0)

# =================================================
# Generate Data
# =================================================
sigma_plot = np.logspace(-6, np.log10(sigma_max_GPa * 2.0), 600) * 1e9  # Pa
Ts_plot = np.array([calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in sigma_plot])

Ts_min = Ts_plot[-1]
delta_T = Ts0 - Ts_min

# =================================================
# Configure Matplotlib (Safe Rendering)
# =================================================
plt.rcParams.update({
    "font.family": font_family,
    "axes.labelsize": label_size,
    "axes.titlesize": title_size,
    "xtick.labelsize": tick_size,
    "ytick.labelsize": tick_size,
    "legend.fontsize": legend_size,
    "figure.dpi": dpi,
    "savefig.dpi": dpi,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
})

# =================================================
# Create Plot
# =================================================
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

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
    label=r"Baseline $T_s(0)$",
    zorder=2
)

# Double-headed arrow
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
    sigma_max_GPa * 1.25,
    (Ts0 + Ts_min) / 2.0,
    rf"$\Delta T_s = {delta_T:.0f}~\mathrm{{K}}$",
    color=baseline_color,
    fontsize=label_size,
    verticalalignment='center',
    horizontalalignment='left',
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor=baseline_color,
        alpha=0.85,
        linewidth=0.8
    ),
    zorder=6
)

# Labels and title
ax.set_xlabel(r"$|\sigma_h|$ [GPa] (log scale)", labelpad=12)
ax.set_ylabel(r"Sintering temperature $T_s$ [K]", labelpad=12)
ax.set_title("Stress-Induced Reduction in Ag Nanoparticle Sintering Temperature", pad=20)

# X-axis formatting
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
if x_label_format == "Scientific (10⁻², 10⁰, 10¹)":
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
else:
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)

# Apply tick styling — MAJOR
ax.tick_params(
    axis='x',
    which='major',
    length=major_tick_length_x,
    width=major_tick_width_x,
    color=major_tick_color,
    labelcolor=major_tick_color
)
ax.tick_params(
    axis='y',
    which='major',
    length=major_tick_length_y,
    width=major_tick_width_y,
    color=major_tick_color,
    labelcolor=major_tick_color
)

# Minor ticks
if minor_ticks:
    ax.minorticks_on()
    ax.tick_params(
        axis='x',
        which='minor',
        length=minor_tick_length_x,
        width=minor_tick_width_x,
        color=minor_tick_color
    )
    ax.tick_params(
        axis='y',
        which='minor',
        length=minor_tick_length_y,
        width=minor_tick_width_y,
        color=minor_tick_color
    )
else:
    ax.minorticks_off()

# Grid
if show_grid:
    ax.grid(True, which="major", linestyle=grid_linestyle, alpha=grid_alpha, zorder=1)
    if minor_ticks:
        ax.grid(True, which="minor", linestyle=":", alpha=grid_alpha * 0.6, zorder=1)

# Spines
for spine in ax.spines.values():
    if show_box:
        spine.set_visible(True)
        spine.set_linewidth(spine_linewidth)
        spine.set_color('black')
    else:
        spine.set_visible(False)

# Legend
legend = ax.legend(
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.96,
    edgecolor='lightgray',
    facecolor='white',
    loc=legend_loc,
    ncol=1
)
legend.get_frame().set_linewidth(0.8)

# Final layout
plt.tight_layout(pad=2.8)

# Render
st.pyplot(fig, use_container_width=False)

# =================================================
# Metrics
# =================================================
st.markdown("### 🔍 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric(label=r"Baseline $T_s(0)$", value=f"{Ts0:.0f} K")
col2.metric(label=r"Max $\Delta T_s$", value=f"{delta_T:.0f} K", delta=f"-{100 * delta_T / Ts0:.0f}%")
col3.metric(label=r"At $|\sigma_h|$", value=f"{sigma_max_GPa:.1f} GPa")

# =================================================
# Prediction Table
# =================================================
st.markdown("### 📊 Predictions at Key Stress Levels")

# Include very low stress (0.01 GPa) to high (sigma_max)
stresses_GPa = np.concatenate((
    [0.01],
    np.logspace(-1, np.log10(max(sigma_max_GPa, 1.0)), 5)
))
stresses_GPa = np.unique(np.round(stresses_GPa, 5))  # Avoid duplicates
stresses_Pa = stresses_GPa * 1e9
Ts_values = [calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in stresses_Pa]

prediction_data = {
    r"$|\sigma_h|$ [GPa]": [f"{s:.3g}" for s in stresses_GPa],
    r"$T_s$ [K]": [f"{t:.0f}" for t in Ts_values],
    r"$\Delta T_s$ [K]": [f"{Ts0 - t:.0f}" for t in Ts_values],
    "Reduction [%]": [f"{100 * (Ts0 - t) / Ts0:.0f}" for t in Ts_values],
}

st.table(prediction_data)

# =================================================
# Physics Explanation
# =================================================
st.markdown(r"""
### 🧪 Physics Insight

The model uses a linearized thermodynamic relation:

$$
T_s(|\sigma_h|) \approx T_s(0) \left( 1 - \frac{\Omega \, |\sigma_h|}{Q_a} \right)
$$

#### Why it matters:
- **0.01 GPa** → negligible change  
- **1 GPa** → ~50–100 K reduction  
- **5 GPa** → ~200–300 K reduction  
- **10–20 GPa** → enables sintering **below 400 K**

> 💡 **Note**: Stresses >5 GPa are achievable in nanotwinned metals, core-shell nanoparticles, or under severe plastic deformation.

This tool helps design **low-temperature sintering pathways** for advanced electronics.
""")

# =================================================
# Footer
# =================================================
st.markdown("---")
st.caption("© 2026 Nanomechanics & Thermodynamics Lab | Interactive model for research and education")
