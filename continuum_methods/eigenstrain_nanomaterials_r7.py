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

# Figure size controls
st.sidebar.subheader("Figure Size")
fig_width = st.sidebar.slider("Figure Width", 6.0, 15.0, 9.5, 0.5)
fig_height = st.sidebar.slider("Figure Height", 4.0, 10.0, 5.8, 0.5)
schematic_width = st.sidebar.slider("Schematic Width", 4.0, 10.0, 6.0, 0.5)
schematic_height = st.sidebar.slider("Schematic Height", 6.0, 12.0, 8.0, 0.5)

# Distribution
dist_type = st.sidebar.selectbox("Eigenstrain Distribution", ["gaussian", "tanh", "linear"], index=0)

# Colormap
cmap_options = ["viridis", "plasma", "inferno", "turbo", "cividis", "jet", "rainbow", "coolwarm", "seismic", "twilight", "hsv"]
body_force_cmap = st.sidebar.selectbox("Body Force Colormap", cmap_options, index=3)

# Line settings
st.sidebar.subheader("Line Settings")
eps_line_thick = st.sidebar.slider("Eigenstrain Line Thickness", 1.0, 6.0, 3.5, 0.1)
force_line_thick = st.sidebar.slider("Body Force Line Thickness", 1.0, 6.0, 3.0, 0.1)
eps_color = st.sidebar.color_picker("Eigenstrain Line Color", "#1f77b4")
force_color = st.sidebar.color_picker("Body Force Line Color", "#d62728")
line_style_eps = st.sidebar.selectbox("Eigenstrain Line Style", ["-", "--", "-.", ":"], index=0)
line_style_force = st.sidebar.selectbox("Body Force Line Style", ["-", "--", "-.", ":"], index=0)

# Font & tick sizes
st.sidebar.subheader("Text Settings")
font_size = st.sidebar.slider("Base Font Size", 10, 24, 14)
label_size = st.sidebar.slider("Axes Label Size", 12, 28, 18)
tick_size = st.sidebar.slider("Tick Label Size", 10, 20, 14)
title_size = st.sidebar.slider("Title Size", 14, 32, 20)

# Legend & frame
st.sidebar.subheader("Legend & Frame")
show_legend = st.sidebar.checkbox("Show Legend", True)
legend_position = st.sidebar.selectbox("Legend Position", 
                                     ["upper right", "upper left", "lower right", "lower left", 
                                      "center right", "center left", "best"], index=0)
legend_frame = st.sidebar.checkbox("Show Legend Frame", True)
legend_alpha = st.sidebar.slider("Legend Background Alpha", 0.0, 1.0, 0.9, 0.05)
frame_thick = st.sidebar.slider("Plot Frame Thickness", 0.5, 4.0, 2.0, 0.1)

# Grid
st.sidebar.subheader("Grid & Fill")
show_grid = st.sidebar.checkbox("Show Grid", True)
grid_style = st.sidebar.selectbox("Grid Style", ["-", "--", "-.", ":"], index=1)
grid_alpha = st.sidebar.slider("Grid Opacity", 0.0, 1.0, 0.3, 0.05)
show_fill = st.sidebar.checkbox("Show Body Force Fill", True)
fill_alpha = st.sidebar.slider("Fill Opacity", 0.1, 1.0, 0.3, 0.05)

# Axis controls
st.sidebar.subheader("Axis Controls")
show_xaxis = st.sidebar.checkbox("Show X Axis", True)
show_yaxis_eps = st.sidebar.checkbox("Show Y Axis (Eigenstrain)", True)
show_yaxis_force = st.sidebar.checkbox("Show Y Axis (Body Force)", True)

# =============================================
# Apply global style (real-time)
# =============================================
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": label_size,
    "axes.titlesize": title_size,
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
# ENHANCED COMPOSITE PLOT
# =============================================
def plot_composite(w, eps_star, defect_name, dist_type):
    x = np.linspace(-3*w, 3*w, 700)
    eps_profile = eigenstrain_distribution(x, eps_star, w, dist_type)
    dx = x[1] - x[0]
    f_profile = -C44 * np.gradient(eps_profile, dx)
    peak_actual = np.max(np.abs(f_profile))
    peak_simple = C44 * (eps_star / w)

    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    
    # Eigenstrain
    ax1.plot(x, eps_profile, color=eps_color, linestyle=line_style_eps, 
             lw=eps_line_thick, label=f'ε*(x) [ε₀ = {eps_star:.3f}]')
    ax1.set_xlabel('Position x (nm)')
    ax1.set_ylabel('Eigenstrain ε*', color=eps_color)
    ax1.tick_params(axis='y', labelcolor=eps_color)
    
    # Grid styling
    if show_grid:
        ax1.grid(True, alpha=grid_alpha, linestyle=grid_style)
    
    # Body force
    ax2 = ax1.twinx()
    cmap = plt.cm.get_cmap(body_force_cmap)
    ax2.plot(x, f_profile, color=force_color, linestyle=line_style_force, 
             lw=force_line_thick, label='f^{eq}(x)')
    
    # Optional fill
    if show_fill:
        ax2.fill_between(x, 0, f_profile, where=f_profile>0, color='red', alpha=fill_alpha)
        ax2.fill_between(x, 0, f_profile, where=f_profile<0, color='blue', alpha=fill_alpha)
    
    ax2.set_ylabel('Body Force (GPa/nm)', color=force_color)
    ax2.tick_params(axis='y', labelcolor=force_color)

    # Axis visibility
    if not show_xaxis:
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    if not show_yaxis_eps:
        ax1.set_ylabel('')
        ax1.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    if not show_yaxis_force:
        ax2.set_ylabel('')
        ax2.tick_params(axis='y', which='both', right=False, labelright=False)

    ax1.set_title(f"{defect_name} — {dist_type.capitalize()} Distribution\n"
                  f"Peak: {peak_actual:.1f} GPa/nm (actual) | {peak_simple:.1f} GPa/nm (estimate)",
                  weight='bold', pad=20)

    # Legend handling
    if show_legend:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc=legend_position)

    plt.tight_layout()
    return fig, peak_actual

# =============================================
# TABS
# =============================================
tab1, tab2, tab3, tab4 = st.tabs(["ISF (ε* = 0.706)", "ESF (ε* = 0.706)", "Twin (ε* = 2.121)", "Export & Theory"])

with tab1:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = plt.figure(figsize=(schematic_width, schematic_height), dpi=300)
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
        fig = plt.figure(figsize=(schematic_width, schematic_height), dpi=300)
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
        fig = plt.figure(figsize=(schematic_width, schematic_height), dpi=300)
        draw_fcc_ag_defect(plt.gca(), "Twin")
        st.pyplot(fig)
    with c2:
        w = st.slider("w (nm)", 0.3, 4.0, 1.0, key="w3")
        fig_comp, peak = plot_composite(w, EPS_TWIN, "Coherent Twin Boundary", dist_type)
        st.pyplot(fig_comp)
        st.success(f"**Peak Body Force:** {peak:.1f} GPa/nm → **{peak*1e18:.2e} N/m³**")

with tab4:
    st.subheader("Export Settings")
    export_format = st.selectbox("Export Format", ["PNG", "PDF", "SVG", "EPS"], index=0)
    export_dpi = st.slider("Export DPI", 100, 1200, 600, 50)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Current Composite Plot"):
            # Get the most recent composite plot
            fig_comp, _ = plot_composite(1.5, EPS_ISF, "Intrinsic Stacking Fault", dist_type)
            buf = io.BytesIO()
            fig_comp.savefig(buf, format=export_format.lower(), dpi=export_dpi, bbox_inches='tight')
            st.download_button(
                label=f"Download Composite Plot as {export_format}",
                data=buf.getvalue(),
                file_name=f"composite_plot.{export_format.lower()}",
                mime=f"image/{export_format.lower()}"
            )
    
    with col2:
        if st.button("Export Current Schematic"):
            fig = plt.figure(figsize=(schematic_width, schematic_height), dpi=300)
            draw_fcc_ag_defect(plt.gca(), "ISF")
            buf = io.BytesIO()
            plt.savefig(buf, format=export_format.lower(), dpi=export_dpi, bbox_inches='tight')
            st.download_button(
                label=f"Download Schematic as {export_format}",
                data=buf.getvalue(),
                file_name=f"schematic.{export_format.lower()}",
                mime=f"image/{export_format.lower()}"
            )
    
    st.markdown("---")
    st.subheader("Theory & Parameters")
    st.latex(r"\epsilon^*_{\text{ISF}} = \frac{b}{d_{111}} = 0.706")
    st.latex(r"\epsilon^*_{\text{ESF}} = \frac{2b}{2d_{111}} = 0.706")
    st.latex(r"\epsilon^*_{\text{Twin}} = \frac{3b}{d_{111}} = 2.121")
    st.latex(r"f^{eq}(x) = -C_{44} \frac{d\epsilon^*}{dx}")
    st.markdown("### All parameters fully editable in real time via sidebar")

st.caption("Ag Nanoparticles 2025")
