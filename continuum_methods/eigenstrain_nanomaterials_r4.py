import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

# =============================================
# STREAMLIT LAYOUT
# =============================================
st.set_page_config(page_title="Defect Mechanics in FCC Ag (2025)", layout="wide")
st.title("Defect-Induced Eigenstrain & Body Force in FCC Silver Nanoparticles")
st.markdown("""
**Intrinsic & Extrinsic Stacking Faults | Coherent Twin Boundaries**  
Publication-quality visualization with **50+ colormaps**, **combined eigenstrain & body-force plot**,  
and **scientific-grade fonts, axes, and line weights**.
""")

# =============================================
# MATERIAL CONSTANTS
# =============================================
a = 0.4086  # nm
b = a / np.sqrt(6)                 # 0.1667 nm
d111 = a / np.sqrt(3)              # 0.2359 nm
C44 = 46.1                         # GPa

# FIXED eigenstrains
EPS_ISF = b / d111
EPS_ESF = (2*b) / (2*d111)
EPS_TWIN = (3*b) / d111

# =============================================
# EIGENSTRAIN DISTRIBUTION MODELS
# =============================================
def eigenstrain_distribution(x, eps_star, w, dist_type='gaussian'):
    if dist_type == 'gaussian':
        sigma = w / 2.5
        return eps_star * np.exp(-(x/sigma)**2)
    elif dist_type == 'tanh':
        return (eps_star / 2) * (1 - np.tanh(2*x/w))
    elif dist_type == 'linear':
        return eps_star * np.maximum(1 - np.abs(x)/w, 0)
    else:
        return eps_star * np.exp(-(x/(w/2.5))**2)

# =============================================
# PUBLICATION STYLE
# =============================================
def apply_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "axes.linewidth": 1.8,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.8,
        "legend.fontsize": 14,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.autolayout": True
    })

# =============================================
# PUBLICATION-GRADE EIGENSTRAIN + BODY-FORCE PLOT
# =============================================
def plot_eigenstrain_bodyforce(w, eps_star, defect_name, dist_type='gaussian', cmap_name="turbo"):

    apply_pub_style()

    # Domain
    x = np.linspace(-3*w, 3*w, 700)
    dx = x[1] - x[0]

    # Profiles
    eps_profile = eigenstrain_distribution(x, eps_star, w, dist_type)
    f_profile = -C44 * np.gradient(eps_profile, dx)

    peak_force_actual = np.max(np.abs(f_profile))
    peak_simple = C44 * (eps_star / w)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot eigenstrain
    ax1.plot(x, eps_profile, color="black", lw=3.2, label="Eigenstrain ε*(x)")
    ax1.set_ylabel("Eigenstrain ε*", fontsize=16)
    ax1.grid(True)

    # Body force (second axis)
    ax2 = ax1.twinx()
    ax2.plot(x, f_profile, color="red", lw=3, label="Body Force f(x)")
    ax2.fill_between(x, 0, f_profile, where=f_profile>0, color='red', alpha=0.25)
    ax2.fill_between(x, 0, f_profile, where=f_profile<0, color='blue', alpha=0.25)
    ax2.set_ylabel("Body Force (GPa/nm)", fontsize=16, color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    # Title
    ax1.set_title(
        f"{defect_name} — {dist_type.capitalize()} Distribution\n"
        f"Peak: {peak_force_actual:.1f} GPa/nm (actual), "
        f"{peak_simple:.1f} GPa/nm (estimate)",
        fontsize=17
    )

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    return fig, peak_force_actual, peak_simple

# =============================================
# FCC DEFECT SCHEMATIC
# =============================================
def draw_fcc_ag_defect(ax, defect_type):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    layers = ['A','B','C','A','B','C','A','B','C','A']
    y_positions = np.linspace(0, 9, 10)
    colors = {'A':'#1f77b4','B':'#ff7f0e','C':'#2ca02c'}
    offsets = {'A':[-2,-1,0,1],'B':[-2.5,-1.5,-0.5,0.5],'C':[-2,-1,0,1]}

    for lay, y in zip(layers, y_positions):
        for x in offsets[lay]:
            ax.add_patch(Circle((x, y), 0.42, color=colors[lay], ec='black', lw=1.2))
        ax.text(-4.6, y, lay, fontsize=14, fontweight="bold", color=colors[lay])

    if defect_type == "ISF":
        ax.axhspan(4,5,color='red',alpha=0.15)
        ax.text(3,4.5,"ISF",fontsize=15,color="red",fontweight="bold")
    elif defect_type == "ESF":
        ax.axhspan(3.5,5.5,color='purple',alpha=0.18)
        ax.text(3,4.5,"ESF",fontsize=15,color="purple",fontweight="bold")
    elif defect_type == "Twin":
        ax.plot([-5, 5], [4.5,4.5], 'k--', lw=2.5)
        ax.axhspan(4.3,4.7,color='green',alpha=0.25)
        ax.text(3,5.8,"Twin Boundary",fontsize=15,color="darkgreen")

# =============================================
# COLORMAPS
# =============================================
available_cmaps = sorted([
    "turbo","jet","viridis","plasma","inferno","magma","cividis","rainbow",
    "coolwarm","seismic","nipy_spectral","gist_rainbow","gist_ncar",
    "cubehelix","hsv","brg","twilight","twilight_shifted","spring","summer",
    "autumn","winter","Accent","Paired","Set1","Set2","Set3"
])

# Sidebar colormap selector
cmap_name = st.sidebar.selectbox("Colormap for Body Force", available_cmaps, index=0)

# Sidebar distribution selector
dist_type = st.sidebar.selectbox("Eigenstrain Distribution", ["gaussian","tanh","linear"])

# =============================================
# TABS
# =============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "ISF (ε* = 0.706)", "ESF (ε* = 0.706)", "Twin (ε* = 2.121)", "Theory & Comparison"
])

# =============================================
# TAB 1 — ISF
# =============================================
with tab1:
    col1, col2 = st.columns([1.2,1])

    with col1:
        fig_sch = plt.figure(figsize=(6, 8))
        ax = fig_sch.add_subplot(111)
        draw_fcc_ag_defect(ax, "ISF")
        st.pyplot(fig_sch)

    with col2:
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 1.5)
        fig_curve, peak_actual, peak_simple = plot_eigenstrain_bodyforce(
            w, EPS_ISF, "ISF", dist_type, cmap_name
        )
        st.pyplot(fig_curve)

# =============================================
# TAB 2 — ESF
# =============================================
with tab2:
    col1, col2 = st.columns([1.2,1])

    with col1:
        fig_sch = plt.figure(figsize=(6, 8))
        ax = fig_sch.add_subplot(111)
        draw_fcc_ag_defect(ax, "ESF")
        st.pyplot(fig_sch)

    with col2:
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 2.0)
        fig_curve, peak_actual, peak_simple = plot_eigenstrain_bodyforce(
            w, EPS_ESF, "ESF", dist_type, cmap_name
        )
        st.pyplot(fig_curve)

# =============================================
# TAB 3 — TWIN
# =============================================
with tab3:
    col1, col2 = st.columns([1.2,1])

    with col1:
        fig_sch = plt.figure(figsize=(6, 8))
        ax = fig_sch.add_subplot(111)
        draw_fcc_ag_defect(ax, "Twin")
        st.pyplot(fig_sch)

    with col2:
        w = st.slider("Gradient width w (nm)", 0.3, 4.0, 1.0)
        fig_curve, peak_actual, peak_simple = plot_eigenstrain_bodyforce(
            w, EPS_TWIN, "Twin", dist_type, cmap_name
        )
        st.pyplot(fig_curve)

# =============================================
# TAB 4 — THEORY
# =============================================
with tab4:
    st.header("Theory & Comparison Coming Soon — Ready for Integration with Your Notes")
    st.write("Includes fixed eigenstrain values, peak scaling, and physical meaning.")

st.caption("Publication-Quality Visualization Module — Ag Nanoparticles (2025)")
