import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import io

# =============================================
# ULTIMATE EDITABLE PUBLICATION-QUALITY TOOL (2025)
# =============================================
st.set_page_config(page_title="Ag Defect Mechanics - Fully Editable", layout="wide")
st.title("Defect-Induced Eigenstrain & Body Force in FCC Silver")
st.markdown("**ISF • ESF • Twin** | Fully customizable publication figures | Real-time style control")

# Silver parameters
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
C44 = 46.1
EPS_ISF = b / d111
EPS_ESF = (2 * b) / (2 * d111)
EPS_TWIN = (3 * b) / d111

# =============================================
# SIDEBAR: FULL STYLE CONTROL
# =============================================
st.sidebar.header("Plot Style Control")

# Distribution
dist_type = st.sidebar.selectbox("Eigenstrain Distribution", ["gaussian", "tanh", "linear"], index=0)

# Colormap
cmap_options = ["viridis", "plasma", "inferno", "turbo", "cividis", "jet", "rainbow", "coolwarm", "seismic", "twilight", "hsv"]
body_force_cmap = st.sidebar.selectbox("Body Force Colormap", cmap_options, index=3)

# Line settings
eps_line_thick = st.sidebar.slider("Eigenstrain Line Thickness", 1.0, 6.0, 3.5, 0.1)
force_line_thick = st.sidebar.slider("Body Force Line Thickness", 1.0, 6.0, 3.0, 0.1)
eps_color = st.sidebar.color_picker("Eigenstrain Line Color", "#1f77b4")
force_color = st.sidebar.color_picker("Body Force Line Color", "#d62728")

# Font & tick sizes
font_size = st.sidebar.slider("Base Font Size", 10, 24, 14)
label_size = st.sidebar.slider("Axes Label Size", 12, 28, 18)
tick_size = st.sidebar.slider("Tick Label Size", 10, 20, 14)

# Legend & frame
legend_frame = st.sidebar.checkbox("Show Legend Frame", True)
legend_alpha = st.sidebar.slider("Legend Background Alpha", 0.0, 1.0, 0.9, 0.05)
frame_thick = st.sidebar.slider("Plot Frame Thickness", 0.5, 4.0, 2.0, 0.1)

# Grid
show_grid = st.sidebar.checkbox("Show Grid", True)
grid_alpha = st.sidebar.slider("Grid Opacity", 0.0, 1.0, 0.3, 0.05)

# =============================================
# Apply global style (real-time)
# =============================================
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": label_size,
    "axes.titlesize": label_size + 4,
    "xtick.labelsize": tick_size,
    "ytick.labelsize": tick_size,
    "lines.linewidth": eps_line_thick,
    "axes.linewidth": frame_thick,
    "grid.alpha": grid_alpha,
    "legend.fontsize": font_size,
    "legend.frameon": legend_frame,
    "legend.framealpha": legend_alpha,
    "legend.edgecolor": "black",
    "figure.dpi": 300,
    "savefig.dpi": 600
})

# =============================================
# Distributions & Schematic (unchanged)
# =============================================
def eigenstrain_distribution(x, eps_star, w, dist_type='gaussian'):
    if dist_type == 'gaussian':
        sigma = w / 2.5
        return eps_star * np.exp(-(x/sigma)**2)
    elif dist_type == 'tanh':
        return (eps_star/2) * (1 - np.tanh(2*x/w))
    elif dist_type == 'linear':
        return eps_star * np.maximum(1 - np.abs(x)/w, 0)
    else:
        sigma = w / 2.5
        return eps_star * np.exp(-(x/sigma)**2)

def draw_fcc_ag_defect(ax, defect_type):
    ax.clear()
    ax.set_xlim(-5, 5); ax.set_ylim(-1, 10)
    ax.set_aspect('equal'); ax.axis('off')
    layers = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    y_pos = np.linspace(0, 9, 10)
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    x_offsets = {'A': [-2, -1, 0, 1], 'B': [-2.5, -1.5, -0.5, 0.5], 'C': [-2, -1, 0, 1]}
    for i, (layer, y) in enumerate(zip(layers, y_pos)):
        for x in x_offsets[layer]:
            ax.add_patch(Circle((x, y), 0.42, color=colors[layer], ec='black', lw=1.2, zorder=3))
        ax.text(-4.6, y, layer, va='center', fontsize=14, fontweight='bold', color=colors[layer])
    if defect_type == "ISF":
        ax.axhspan(4.0, 5.0, color='red', alpha=0.15)
        ax.text(3.0, 4.5, "ISF\n(1 HCP layer)", fontsize=14, color='red', fontweight='bold')
        ax.add_patch(FancyArrowPatch((0, 4.6), (2, 4.6), arrowstyle='->', color='red', lw=3))
    elif defect_type == "ESF":
        ax.axhspan(3.5, 5.5, color='purple', alpha=0.18)
        ax.text(3.0, 4.5, "ESF\n(2 HCP layers)", fontsize=14, color='purple', fontweight='bold')
    elif defect_type == "Twin":
        ax.plot([-5, 5], [4.5, 4.5], 'k--', lw=2.5, alpha=0.8)
        ax.axhspan(4.3, 4.7, color='green', alpha=0.25)
        ax.text(3.0, 5.8, "Coherent Twin Boundary", fontsize=14, color='darkgreen', fontweight='bold')

# =============================================
# COMPOSITE PLOT (now fully styled)
# =============================================
def plot_composite(w, eps_star, defect_name, dist_type):
    x = np.linspace(-3*w, 3*w, 700)
    eps_profile = eigenstrain_distribution(x, eps_star, w, dist_type)
    dx = x[1] - x[0]
    f_profile = -C44 * np.gradient(eps_profile, dx)
    peak_actual = np.max(np.abs(f_profile))
    peak_simple = C44 * (eps_star / w)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.8))
    
    # Eigenstrain
    ax1.plot(x, eps_profile, color=eps_color, lw=eps_line_thick, label=f'ε*(x) [ε₀ = {eps_star:.3f}]')
    ax1.set_xlabel('Position x (nm)')
    ax1.set_ylabel('Eigenstrain ε*', color=eps_color)
    ax1.tick_params(axis='y', labelcolor=eps_color)
    if show_grid: ax1.grid(True, alpha=grid_alpha, ls=':')

    # Body force
    ax2 = ax1.twinx()
    cmap = plt.cm.get_cmap(body_force_cmap)
    ax2.plot(x, f_profile, color=force_color, lw=force_line_thick, label='f^{eq}(x)')
    ax2.fill_between(x, 0, f_profile, where=f_profile>0, color='red', alpha=0.3)
    ax2.fill_between(x, 0, f_profile, where=f_profile<0, color='blue', alpha=0.3)
    ax2.set_ylabel('Body Force (GPa/nm)', color=force_color)
    ax2.tick_params(axis='y', labelcolor=force_color)

    ax1.set_title(f"{defect_name} — {dist_type.capitalize()} Distribution\n"
                  f"Peak: {peak_actual:.1f} GPa/nm (actual) | {peak_simple:.1f} GPa/nm (estimate)",
                  weight='bold', pad=20)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig, peak_actual

# =============================================
# TABS
# =============================================
tab1, tab2, tab3, tab4 = st.tabs(["ISF (ε* = 0.706)", "ESF (ε* = 0.706)", "Twin (ε* = 2.121)", "Theory"])

with tab1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = plt.figure(figsize=(6,8), dpi=300)
        draw_fcc_ag_defect(plt.gca(), "ISF")
        st.pyplot(fig)
    with c2:
        w = st.slider("w (nm)", 0.5, 5.0, 1.5, key="w1")
        fig_comp, peak = plot_composite(w, EPS_ISF, "Intrinsic Stacking Fault", dist_type)
        st.pyplot(fig_comp)
        st.success(f"**Peak Body Force:** {peak:.1f} GPa/nm → **{peak*1e18:.2e} N/m³**")

with tab2:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = plt.figure(figsize=(6,8), dpi=300)
        draw_fcc_ag_defect(plt.gca(), "ESF")
        st.pyplot(fig)
    with c2:
        w = st.slider("w (nm)", 0.5, 5.0, 2.0, key="w2")
        fig_comp, peak = plot_composite(w, EPS_ESF, "Extrinsic Stacking Fault", dist_type)
        st.pyplot(fig_comp)
        st.success(f"**Peak Body Force:** {peak:.1f} GPa/nm → **{peak*1e18:.2e} N/m³**")

with tab3:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = plt.figure(figsize=(6,8), dpi=300)
        draw_fcc_ag_defect(plt.gca(), "Twin")
        st.pyplot(fig)
    with c2:
        w = st.slider("w (nm)", 0.3, 4.0, 1.0, key="w3")
        fig_comp, peak = plot_composite(w, EPS_TWIN, "Coherent Twin Boundary", dist_type)
        st.pyplot(fig_comp)
        st.success(f"**Peak Body Force:** {peak:.1f} GPa/nm → **{peak*1e18:.2e} N/m³**")

with tab4:
    st.latex(r"\epsilon^*_{\text{ISF}} = \epsilon^*_{\text{ESF}} = 0.706, \quad \epsilon^*_{\text{Twin}} = 2.121")
    st.markdown("### All parameters fully editable in real time via sidebar")

st.caption("Fully Editable • Publication-Ready • Nature Materials Level | Ag Nanoparticles 2025")
