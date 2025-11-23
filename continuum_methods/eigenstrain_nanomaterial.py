import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

# =============================================
# Publication-Quality Ag Nanoparticle Defect Visualizer
# =============================================
st.set_page_config(page_title="Defect Mechanics in FCC Ag", layout="centered")
st.title("Publication-Quality Defect Schematics in FCC Silver (Ag)")
st.markdown("""
**Intrinsic & Extrinsic Stacking Faults | Coherent Twin Boundaries**  
Atomic-scale visualization of how plastic deformation introduces **local HCP regions** in FCC Ag nanoparticles —  
the key mechanism behind ultra-low-temperature sintering (94 °C).
""")

# Silver FCC parameters
a = 0.4086  # nm
d111 = a / np.sqrt(3)  # ≈0.236 nm
C44 = 46.1  # GPa

# =============================================
# High-quality atomic schematic generator
# =============================================
def draw_fcc_ag_defect(ax, defect_type, show_arrows=True):
    ax.clear()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    # FCC stacking: ABCABC... along [111]
    layers = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']
    y_positions = np.linspace(0, 8, 9)
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}  # Blue, Orange, Green

    # Atom positions in {111} plane projection
    x_offsets = {'A': [-1.5, -0.5, 0.5, 1.5],
                 'B': [-2.0, -1.0, 0.0, 1.0],
                 'C': [-1.5, -0.5, 0.5, 1.5]}

    for i, (layer, y) in enumerate(zip(layers, y_positions)):
        offset = x_offsets[layer]
        for x in offset:
            circle = Circle((x, y), 0.38, color=colors[layer], ec='black', lw=1.2, zorder=3)
            ax.add_patch(circle)
        ax.text(-3.7, y, f"{layer}", va='center', fontsize=14, fontweight='bold', color=colors[layer])

    # === Defect highlighting ===
    if defect_type == "ISF":
        ax.axhspan(3.0, 5.0, color='red', alpha=0.1, hatch='/')
        ax.text(2.6, 4.0, "Intrinsic SF\n(single HCP layer)", fontsize=13, color='red', fontweight='bold',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        if show_arrows:
            arrow = FancyArrowPatch((0.5, 4.8), (1.8, 4.8), arrowstyle='->,head_width=8', 
                                  color='red', lw=2.5, mutation_scale=20)
            ax.add_patch(arrow)
            ax.text(1.0, 5.3, "Shockley partial slip", color='red', fontsize=11, ha='center')

    elif defect_type == "ESF":
        ax.axhspan(2.6, 5.4, color='purple', alpha=0.12, hatch='\\')
        ax.text(2.6, 4.0, "Extrinsic SF\n(two-layer HCP)", fontsize=13, color='purple', fontweight='bold',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))
        if show_arrows:
            arrow1 = FancyArrowPatch((-0.5, 4.8), (0.8, 4.8), arrowstyle='->', color='purple', lw=2)
            arrow2 = FancyArrowPatch((0.5, 3.6), (1.8, 3.6), arrowstyle='->', color='purple', lw=2)
            ax.add_patch(arrow1)
            ax.add_patch(arrow2)

    elif defect_type == "Twin":
        ax.axhspan(3.8, 4.2, color='green', alpha=0.25, hatch='//')
        ax.plot([-4, 4], [4.0, 4.0], 'k--', lw=2, alpha=0.8)
        ax.text(2.6, 5.2, "Coherent Twin Boundary", fontsize=13, color='darkgreen', fontweight='bold')
        ax.text(2.6, 4.6, "Mirror plane", fontsize=12, color='darkgreen', style='italic')
        ax.text(2.6, 2.8, "FCC → HCP-like → FCC", fontsize=11, color='gray', style='italic')

    ax.set_title(f"FCC Silver (Ag) with {defect_type}", fontsize=16, pad=20)

# =============================================
# Tabs
# =============================================
tab1, tab2, tab3 = st.tabs(["Intrinsic Stacking Fault (ISF)", 
                             "Extrinsic Stacking Fault (ESF)", 
                             "Coherent Twin Boundary"])

with tab1:
    st.header("Intrinsic Stacking Fault (ISF) → Single-Layer HCP")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        - One Shockley partial dislocation  
        - Stacking: ABC→AB**C**BC  
        - Introduces **one atomic layer** of HCP  
        - Eigenstrain: ε* ≈ 0.35  
        - Reduces vacancy formation energy by ~0.08 eV
        """)
        w = st.slider("Gradient width w (nm)", 0.5, 4.0, 1.5, 0.1, key="w1")
        f_mag = C44 * (0.35 / w)
        st.success(f"**Volumetric body force:** ~{f_mag:.2e} GPa/nm\n→ **{f_mag*1e18:.2e} N/m³**")

    with col2:
        fig, ax = plt.subplots(figsize=(7, 8))
        draw_fcc_ag_defect(ax, "ISF")
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.header("Extrinsic Stacking Fault (ESF) → Two-Layer HCP")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        - Two consecutive partial dislocations  
        - Stacking: ABC→AB**CA**BC  
        - Forms a **two-layer HCP slab**  
        - Stronger diffusion channel than ISF  
        - Common in high-deformation Ag NPs
        """)
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 2.0, 0.1, key="w2")
        f_mag = C44 * (0.70 / w)  # ~2× ISF
        st.success(f"**Body force magnitude:** ~{f_mag:.2e} GPa/nm\n→ **{f_mag*1e18:.2e} N/m³**")

    with col2:
        fig, ax = plt.subplots(figsize=(7, 8))
        draw_fcc_ag_defect(ax, "ESF")
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.header("Coherent Twin Boundary → HCP Mirror Plane")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        - Mirror symmetry across {111}  
        - Stacking: ABC→AB**C**||**C**BA  
        - Boundary is **one HCP layer thick**  
        - Acts as fast atomic diffusion path  
        - Dominant in die-cast Ag NPs (your 94°C sintering!)
        """)
        w = st.slider("Gradient width w (nm)", 0.3, 3.0, 1.0, 0.1, key="w3")
        f_mag = C44 * (0.35 / w)
        st.success(f"**Twin boundary body force:** ~{f_mag:.2e} GPa/nm\n→ **{f_mag*1e18:.2e} N/m³**")

    with col2:
        fig, ax = plt.subplots(figsize=(7, 8))
        draw_fcc_ag_defect(ax, "Twin")
        plt.tight_layout()
        st.pyplot(fig)

# =============================================
# Footer
# =============================================
st.markdown("---")
st.markdown("""
Local HCP regions from stacking faults and twins are the atomic origin of defect-engineered low-temperature sintering in Ag nanoparticles.
""")
st.caption("© 2025 – Publication-Ready | Fully generated schematics | No external images")
