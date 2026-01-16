import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

# =================================================
# Page config
# =================================================
st.set_page_config(
    page_title="Ag Nanoparticles: Stress–Sintering",
    layout="wide"
)

st.title("Sintering Temperature vs Hydrostatic Stress in Ag Nanoparticles")
st.markdown("**Interactive, publication-quality visualization with full style control**")

# =================================================
# Sidebar: Physical parameters
# =================================================
st.sidebar.header("Physical parameters")

Ts0 = st.sidebar.slider("Baseline Tₛ(0) [K]", 500, 700, 623)
Qa_kJ = st.sidebar.slider("Activation energy Qₐ [kJ/mol]", 70.0, 120.0, 90.0, step=5.0)
Omega_m3 = st.sidebar.slider(
    "Atomic volume Ω (×10⁻²⁹ m³)", 8.0, 12.0, 10.0
) * 1e-29
sigma_max_GPa = st.sidebar.slider("Max |σₕ| [GPa]", 1.0, 10.0, 5.0)

# =================================================
# Sidebar: Curve styling
# =================================================
st.sidebar.header("Curve styling")

main_color = st.sidebar.color_picker("Main curve color", "#1f77b4")
baseline_color = st.sidebar.color_picker("Baseline color", "#d62728")
marker_color = st.sidebar.color_picker("Zero-stress marker color", "#000000")
linewidth = st.sidebar.slider("Main line width", 1.0, 5.0, 3.0)
linestyle = st.sidebar.selectbox("Main line style", ["-", "--", "-.", ":"])
marker_size = st.sidebar.slider("Marker size", 5, 15, 10)

# =================================================
# Sidebar: Fonts and layout
# =================================================
st.sidebar.header("Fonts and layout")

font_family = st.sidebar.selectbox(
    "Font family", ["DejaVu Sans", "Times New Roman", "Arial"]
)
label_size = st.sidebar.slider("Axis label font size", 10, 20, 14)
tick_size = st.sidebar.slider("Tick font size", 8, 16, 12)
title_size = st.sidebar.slider("Title font size", 12, 24, 18)
legend_size = st.sidebar.slider("Legend font size", 8, 16, 12)

# =================================================
# Sidebar: Axes, ticks, and box
# =================================================
st.sidebar.header("Axes and ticks")

show_grid = st.sidebar.checkbox("Show grid", True)
grid_alpha = st.sidebar.slider("Grid transparency", 0.05, 0.6, 0.25)
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
})

# =================================================
# Plot
# =================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogx(
    sigma_plot / 1e9,
    Ts_plot,
    color=main_color,
    linewidth=linewidth,
    linestyle=linestyle,
    label=r"$T_s(|\sigma_h|)$"
)

ax.plot(
    1e-6,
    Ts0,
    marker="o",
    color=marker_color,
    markersize=marker_size,
    label=r"$T_s(0)$"
)

ax.axhline(
    Ts0,
    color=baseline_color,
    linestyle="--",
    linewidth=2,
    label="Baseline"
)

ax.annotate(
    "",
    xy=(sigma_max_GPa, Ts_min),
    xytext=(1e-6, Ts0),
    arrowprops=dict(arrowstyle="<->", color=baseline_color, linewidth=2),
)

ax.text(
    1e-3,
    0.5 * (Ts0 + Ts_min),
    rf"$\Delta T_s = {delta_T:.0f}\,\mathrm{{K}}$",
    color=baseline_color,
    fontsize=label_size,
)

ax.set_xlabel(r"$|\sigma_h|$ [GPa] (log scale)")
ax.set_ylabel(r"Sintering temperature $T_s$ [K]")
ax.set_title("Stress-Induced Reduction in Ag Nanoparticle Sintering Temperature")

ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(ScalarFormatter())

if minor_ticks:
    ax.minorticks_on()

ax.grid(show_grid, which="both", alpha=grid_alpha)

for spine in ax.spines.values():
    spine.set_visible(show_box)

ax.legend(frameon=True)

st.pyplot(fig)

# =================================================
# Metrics
# =================================================
st.markdown("### Key metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Baseline $T_s$", f"{Ts0:.0f} K")
col2.metric("Max $\\Delta T_s$", f"{delta_T:.0f} K",
            f"-{100 * delta_T / Ts0:.0f}%")
col3.metric("At $|\\sigma_h|$", f"{sigma_max_GPa:.1f} GPa")

# =================================================
# Prediction table
# =================================================
st.markdown("### Predictions at key stress levels")

stresses_GPa = np.logspace(-1, np.log10(sigma_max_GPa), 6)
stresses_Pa = stresses_GPa * 1e9

Ts_values = [calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in stresses_Pa]

st.table({
    "|σₕ| [GPa]": [f"{s:.2f}" for s in stresses_GPa],
    "$T_s$ [K]": [f"{t:.0f}" for t in Ts_values],
    "$ΔT_s$ [K]": [f"{Ts0 - t:.0f}" for t in Ts_values],
    "Reduction [%]": [f"{100 * (Ts0 - t) / Ts0:.0f}" for t in Ts_values],
})

# =================================================
# Physics note
# =================================================
st.markdown(r"""
**Physics relation**
\[
T_s(|\sigma_h|) \approx T_s(0)\left(1 - \frac{\Omega |\sigma_h|}{Q_a}\right)
\]

**Interpretation**
- Twin- or defect-induced hydrostatic stresses of 1–5 GPa can lower $T_s$ substantially.
- Extreme defect stresses (5–10 GPa) enable ultra-low-temperature sintering pathways.
""")
