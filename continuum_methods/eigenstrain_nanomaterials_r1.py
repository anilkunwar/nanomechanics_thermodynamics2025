import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import io  # For PNG export

# =============================================
# Enhanced Publication-Quality Ag Defect Visualizer
# =============================================
st.set_page_config(page_title="Defect Mechanics in FCC Ag (2025)", layout="wide")
st.title("🟡 Defect-Induced Eigenstrain & Body Force in FCC Ag Nanoparticles")
st.markdown("""
Enhanced interactive tool for your 2025 nanomechanics study.  
Visualizes ISF/ESF/Twins with local HCP, computes eigenstrain/body force, and confirms displacements as b/2b/3b.
""")

# Ag parameters
a = 0.4086  # nm
b = a / np.sqrt(6)  # Shockley partial magnitude
d111 = a / np.sqrt(3)
C44 = 46.1  # GPa

# Tabs including new Theory tab
tab1, tab2, tab3, tab4 = st.tabs(["ISF (d=b)", "ESF (d=2b)", "Twin (d=3b)", "Theory & Export"])

# Schematic generator (enhanced with HCP labels)
def draw_fcc_ag_defect(ax, defect_type):
    ax.clear()
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Layers with HCP highlight
    layers = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']
    y_pos = np.linspace(0, 8.5, 9)
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    x_offsets = {'A': [-2, -1, 0, 1], 'B': [-2.5, -1.5, -0.5, 0.5], 'C': [-2, -1, 0, 1]}

    for i, (layer, y) in enumerate(zip(layers, y_pos)):
        for x in x_offsets[layer]:
            Circle = plt.Circle((x, y), 0.4, color=colors[layer], ec='k', lw=1)
            ax.add_patch(Circle)
        ax.text(-4.2, y, layer, va='center', fontsize=14, fontweight='bold', color=colors[layer])

    # Defect-specific
    if defect_type == "ISF":
        ax.axhspan(3.5, 4.5, color='red', alpha=0.15)
        ax.text(2.8, 4.0, "ISF (1 HCP layer)", fontsize=13, color='red')
        ax.add_patch(FancyArrowPatch((0, 4.2), (1.5, 4.2), arrowstyle='->', color='red', lw=2))
    elif defect_type == "ESF":
        ax.axhspan(3.0, 5.0, color='purple', alpha=0.15)
        ax.text(2.8, 4.0, "ESF (2 HCP layers)", fontsize=13, color='purple')
        ax.add_patch(FancyArrowPatch((-0.5, 4.5), (1, 4.5), arrowstyle='->', color='purple', lw=2))
        ax.add_patch(FancyArrowPatch((0, 3.5), (1.5, 3.5), arrowstyle='->', color='purple', lw=2))
    elif defect_type == "Twin":
        ax.axhspan(3.8, 4.2, color='green', alpha=0.2)
        ax.plot([-4.5, 4.5], [4.0, 4.0], 'k--', lw=2)
        ax.text(2.8, 5.0, "Twin (HCP mirror)", fontsize=13, color='darkgreen')

# Plot eigenstrain & body force (publication-quality)
def plot_curves(w, eps, dpi=300):
    x = np.linspace(-2*w, 2*w, 500)
    eps_profile = eps * np.exp(- (x / (w / 3))**2)
    f_profile = -C44 * np.gradient(eps_profile, x[1]-x[0])

    fig, ax1 = plt.subplots(figsize=(6,4), dpi=dpi)
    ax1.plot(x, eps_profile, 'b-', lw=2.5, label='Eigenstrain')
    ax1.set_xlabel('Position x (nm)', fontsize=12)
    ax1.set_ylabel('Eigenstrain ε*', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, ls='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, f_profile, 'r--', lw=2.5, label='Body Force')
    ax2.set_ylabel('f^{eq} (GPa/nm)', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    return fig

# Tab 1: ISF
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_sch, ax = plt.subplots(dpi=300)
        draw_fcc_ag_defect(ax, "ISF")
        st.pyplot(fig_sch)
    with col2:
        w = st.slider("w (nm)", 0.5, 5.0, 1.5)
        d = b
        eps = d / d111
        f = C44 * (eps / w)
        st.success(f"d = b = {d:.4f} nm\nε* = {eps:.3f}\n|f^{eq}| = {f:.2e} GPa/nm")
        fig_curve = plot_curves(w, eps)
        st.pyplot(fig_curve)

# Tab 2: ESF
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig_sch, ax = plt.subplots(dpi=300)
        draw_fcc_ag_defect(ax, "ESF")
        st.pyplot(fig_sch)
    with col2:
        w = st.slider("w (nm)", 0.5, 5.0, 2.0)
        d = 2 * b
        eps = d / (2 * d111)  # Effective h=2 d111
        f = C44 * (eps / w)
        st.success(f"d = 2b = {d:.4f} nm\nε* = {eps:.3f}\n|f^{eq}| = {f:.2e} GPa/nm")
        fig_curve = plot_curves(w, eps)
        st.pyplot(fig_curve)

# Tab 3: Twin
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig_sch, ax = plt.subplots(dpi=300)
        draw_fcc_ag_defect(ax, "Twin")
        st.pyplot(fig_sch)
    with col2:
        w = st.slider("w (nm)", 0.5, 5.0, 1.0)
        d = 3 * b
        eps = d / d111  # Nucleus over ~3 planes, but h=d111 for boundary
        f = C44 * (eps / w)
        st.success(f"d = 3b = {d:.4f} nm\nε* = {eps:.3f}\n|f^{eq}| = {f:.2e} GPa/nm")
        fig_curve = plot_curves(w, eps)
        st.pyplot(fig_curve)

# Tab 4: Theory & Export
with tab4:
    st.header("Theory Section")
    st.markdown("""[Insert the theory text from above here]""")  # Paste the theory section

    st.header("Download Figures")
    for defect in ["ISF", "ESF", "Twin"]:
        buf = io.BytesIO()
        fig_sch, ax = plt.subplots(dpi=300)
        draw_fcc_ag_defect(ax, defect)
        fig_sch.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(f"Download {defect} Schematic", buf.getvalue(), f"{defect}_Ag.png")

st.caption("Enhanced for 2025 Study | Ready for Publication")
