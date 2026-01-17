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
st.markdown("**Interactive, publication-quality visualization with robust math rendering**")

# =================================================
# Sidebar: Physical parameters
# =================================================
st.sidebar.header("PropertyParams")

Ts0 = st.sidebar.slider("Baseline $T_s(0)$ [K]", 500, 700, 623)
Qa_kJ = st.sidebar.slider("Activation energy $Q_a$ [kJ/mol]", 70.0, 120.0, 90.0, step=5.0)
Omega_m3 = st.sidebar.slider(
    "Atomic volume $\\Omega$ (×10⁻²⁹ m³)", 8.0, 12.0, 10.0
) * 1e-29
sigma_max_GPa = st.sidebar.slider("Max $|\\sigma_h|$ [GPa]", 1.0, 10.0, 5.0)

# =================================================
# Sidebar: Figure dimensions
# =================================================
st.sidebar.header("📐 Figure Size")

fig_width = st.sidebar.slider("Figure width (inches)", 8.0, 20.0, 12.0, step=0.5)
fig_height = st.sidebar.slider("Figure height (inches)", 4.0, 12.0, 7.0, step=0.5)
dpi = st.sidebar.selectbox("Resolution (DPI)", [100, 150, 200, 300], index=1)

# Estimate optimal tick font size
estimated_tick_size = max(8, min(18, int(0.8 * (fig_height + fig_width) / 2)))
st.sidebar.caption(f"💡 Suggested tick font size: ~{estimated_tick_size}")

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
    "Font family", ["DejaVu Sans", "Times New Roman", "Arial", "Helvetica", "serif", "sans-serif"]
)
label_size = st.sidebar.slider("Axis label font size", 10, 24, 16)
tick_size = st.sidebar.slider("Tick font size", 6, 20, estimated_tick_size)
title_size = st.sidebar.slider("Title font size", 12, 28, 20)
legend_size = st.sidebar.slider("Legend font size", 8, 18, 13)

# Note: Full LaTeX is DISABLED to avoid runtime errors on Streamlit Cloud
st.sidebar.info("✅ Using Matplotlib's built-in math renderer (no system LaTeX required)")

# =================================================
# Sidebar: Axes, ticks, and box
# =================================================
st.sidebar.header("AxisSize & Spines")

show_grid = st.sidebar.checkbox("Show grid", True)
grid_alpha = st.sidebar.slider("Grid transparency", 0.05, 0.6, 0.3)
grid_linestyle = st.sidebar.selectbox("Grid line style", ["-", "--", "-.", ":"], index=1)
show_box = st.sidebar.checkbox("Show axes box", True)
spine_linewidth = st.sidebar.slider("Axes box (spine) thickness", 0.5, 3.0, 1.2)
minor_ticks = st.sidebar.checkbox("Enable minor ticks", True)

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
# Configure Matplotlib — NO LATEX
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
    "text.usetex": False,  # CRITICAL: Disable system LaTeX
    "mathtext.fontset": "cm",  # Use Computer Modern-style math (looks like LaTeX)
})

# =================================================
# Plot with enhanced aesthetics and safe rendering
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

# ΔT annotation with background
ax.text(
    sigma_max_GPa * 1.25,
    (Ts0 + Ts_min) / 2,
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

# Labels and title — using mathtext-compatible syntax
ax.set_xlabel(r"$|\sigma_h|$ [GPa] (log scale)", labelpad=12)
ax.set_ylabel(r"Sintering temperature $T_s$ [K]", labelpad=12)
ax.set_title("Stress-Induced Reduction in Ag Nanoparticle Sintering Temperature", pad=20)

# X-axis formatting
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis='x', style='plain', useOffset=False)

# Minor ticks — FIXED: no 'alpha' in tick_params
if minor_ticks:
    ax.minorticks_on()
    # Use RGBA tuple for semi-transparent gray (allowed)
    ax.tick_params(
        which='minor',
        length=3,
        color=(0.5, 0.5, 0.5, 0.7),  # RGBA
        width=0.8
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

# Final layout — now SAFE
plt.tight_layout(pad=2.8)

# Render
st.pyplot(fig, use_container_width=False)

# =================================================
# Metrics
# =================================================
st.markdown("### 🔍 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric(r"Baseline $T_s(0)$", f"{Ts0:.0f} K")
col2.metric(r"Max $\Delta T_s$", f"{delta_T:.0f} K", delta=f"-{100 * delta_T / Ts0:.0f}%")
col3.metric(r"At $|\sigma_h|$", f"{sigma_max_GPa:.1f} GPa")

# =================================================
# Prediction table
# =================================================
st.markdown("### 📊 Predictions at Key Stress Levels")

stresses_GPa = np.logspace(-1, np.log10(sigma_max_GPa), 6)
stresses_Pa = stresses_GPa * 1e9
Ts_values = [calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in stresses_Pa]

prediction_data = {
    r"$|\sigma_h|$ [GPa]": [f"{s:.2f}" for s in stresses_GPa],
    r"$T_s$ [K]": [f"{t:.0f}" for t in Ts_values],
    r"$\Delta T_s$ [K]": [f"{Ts0 - t:.0f}" for t in Ts_values],
    "Reduction [%]": [f"{100 * (Ts0 - t) / Ts0:.0f}" for t in Ts_values],
}

st.table(prediction_data)

# =================================================
# Physics explanation — using safe math notation
# =================================================
st.markdown(r"""
### 🧪 Physics Insight

The stress-modified sintering temperature follows:

$$
T_s(|\sigma_h|) \approx T_s(0) \left( 1 - \frac{\Omega \, |\sigma_h|}{Q_a} \right)
$$

where:
- $\Omega$ = atomic volume of silver ($\sim 10^{-29}~\mathrm{m^3}$),
- $Q_a$ = activation energy for diffusion,
- $|\sigma_h|$ = hydrostatic stress magnitude (in compression).

#### Key implications:
- Internal stresses of **1–5 GPa** (common in nanotwins or defects) reduce $T_s$ by **100–300 K**.
- Enables **ultra-low-temperature sintering** (< 450 K) for flexible electronics.
- Most effective in **nanoparticles** due to high internal strain fields.

> 💡 **Design strategy**: Engineer microstructural defects to *locally* lower sintering temperature without external heating.
""")
