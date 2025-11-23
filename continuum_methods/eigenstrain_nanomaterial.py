import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# =============================================
# Streamlit App: Eigenstrain & Body Force in Ag NPs
# =============================================
st.set_page_config(page_title="Ag Nanoparticle Defect Mechanics", layout="centered")
st.title("🟡 Eigenstrain and Volumetric Body Force in FCC Silver (Ag)")
st.markdown("""
Interactive calculator for **intrinsic/extrinsic stacking faults** and **coherent twin boundaries**  
in face-centered cubic **silver nanoparticles** based on micromechanics theory.
""")

# Silver material properties
Ag_props = {
    "a": 0.4086,           # Lattice constant (nm)
    "d111": 0.4086 / np.sqrt(3),  # {111} interplanar spacing ≈ 0.2359 nm
    "C11": 124.0,          # GPa
    "C12": 93.4,
    "C44": 46.1,           # Shear modulus μ ≈ C44
    "gamma_SF": 22e-3,     # Stacking fault energy (J/m²)
}

# Tabs
tab1, tab2, tab3 = st.tabs(["Intrinsic Stacking Fault (ISF)", 
                             "Extrinsic Stacking Fault (ESF)", 
                             "Coherent Twin Boundary"])

def draw_fcc_defect_schematic(ax, defect_type):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # FCC {111} layers: ABCABC...
    layers = ['A', 'B', 'C', 'A', 'B', 'C', 'A']
    y_pos = np.linspace(0, 6, 7)
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

    for i, (layer, y) in enumerate(zip(layers, y_pos)):
        for x in [-1.5, -0.5, 0.5, 1.5]:
            circle = plt.Circle((x, y), 0.35, color=colors[layer], alpha=0.8, ec='k', lw=0.8)
            ax.add_patch(circle)
        ax.text(-2.7, y, f"{layer}-layer", va='center', fontsize=12, fontweight='bold')

    # Defect zone
    if defect_type == "ISF":
        ax.axhspan(2.0, 4.0, color='red', alpha=0.15)
        ax.text(2.2, 3.0, "ISF", fontsize=14, color='red', fontweight='bold')
        ax.annotate("Shockley partial slip", xy=(0, 3), xytext=(1.5, 4.5),
                    arrowprops=dict(arrowstyle="->", color='red'), fontsize=11, color='red')
    elif defect_type == "ESF":
        ax.axhspan(1.8, 4.2, color='purple', alpha=0.15)
        ax.text(2.2, 3.0, "ESF", fontsize=14, color='purple', fontweight='bold')
    elif defect_type == "Twin":
        ax.axhspan(2.8, 3.2, color='green', alpha=0.2, hatch='//')
        ax.text(2.2, 3.0, "Twin", fontsize=14, color='darkgreen', fontweight='bold')

def compute_defect_parameters(defect_type, a, h, w, mu):
    if defect_type == "ISF":
        d = a / np.sqrt(6)              # |b_p| = a/√6 ≈ 0.167 nm
        label = "Shockley partial"
    elif defect_type == "ESF":
        d = 2 * (a / np.sqrt(6))         # Two partials
        label = "Two partials"
    else:  # Twin
        d = a / (2 * np.sqrt(6))         # Effective displacement per boundary
        label = "Twinning partial"

    epsilon_star = d / h
    f_eq_mag = mu * (epsilon_star / w)   # |f_eq| ≈ μ × ∇ε*

    return d, epsilon_star, f_eq_mag, label

# =============================================
# Tab 1: ISF
# =============================================
with tab1:
    st.header("Intrinsic Stacking Fault (ISF)")
    st.write("One violated stacking sequence: ABC→AB**C**BC")

    col1, col2 = st.columns([1, 1])
    with col1:
        a = st.slider("Lattice constant a (nm)", 0.40, 0.42, Ag_props["a"], 0.001, key="a1")
        h = st.slider("Defect thickness h = d₁₁₁ (nm)", 0.20, 0.26, Ag_props["d111"], 0.001, key="h1")
        w = st.slider("Eigenstrain gradient width w (nm)", 0.5, 5.0, 1.5, 0.1, key="w1")
        mu = st.slider("Shear modulus μ = C₄₄ (GPa)", 40.0, 50.0, Ag_props["C44"], 0.5, key="mu1")

    with col2:
        fig, ax = plt.subplots(figsize=(5, 6))
        draw_fcc_defect_schematic(ax, "ISF")
        st.pyplot(fig)

    d, eps, fmag, label = compute_defect_parameters("ISF", a, h, w, mu)

    st.success(f"""
    **Results for ISF in Ag:**
    - Displacement vector magnitude: **{d:.4f} nm** ({label})
    - Eigenstrain ε* ≈ d / h = **{eps:.3f}**
    - Volumetric body force |fᵉq| ≈ μ × (ε*/w) = **{fmag:.2e} GPa/nm**  
      → **{fmag * 1e18:.2e} N/m³** (extremely high!)
    """)

    # Plot profile
    x = np.linspace(-w*2, w*2, 400)
    eps_profile = eps * np.exp(-((x)/ (w/3))**2)  # Gaussian-like
    f_profile = -mu * np.gradient(eps_profile, x[1] - x[0])

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(x, eps_profile, 'b-', lw=2, label='Eigenstrain ε*(x)')
    ax1.set_ylabel("Eigenstrain", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, f_profile, 'r--', lw=2, label='Body force fᵉq(x)')
    ax2.set_ylabel("Body Force (GPa/nm)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.set_xlabel("Position (nm)")
    ax1.set_title("1D Eigenstrain & Body Force Profile Across ISF")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig)

# =============================================
# Tab 2: ESF
# =============================================
with tab2:
    st.header("Extrinsic Stacking Fault (ESF)")
    st.write("Two consecutive violated layers: ABC→AB**C A**BC")

    col1, col2 = st.columns([1, 1])
    with col1:
        a = st.slider("Lattice constant a (nm)", 0.40, 0.42, Ag_props["a"], 0.001, key="a2")
        h = st.slider("Effective thickness h (nm)", 0.3, 0.6, 0.472, 0.01, key="h2")
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 2.0, 0.1, key="w2")
        mu = st.slider("Shear modulus μ (GPa)", 40.0, 50.0, Ag_props["C44"], 0.5, key="mu2")

    with col2:
        fig, ax = plt.subplots(figsize=(5, 6))
        draw_fcc_defect_schematic(ax, "ESF")
        st.pyplot(fig)

    d, eps, fmag, label = compute_defect_parameters("ESF", a, h, w, mu)
    st.success(f"""
    **Results for ESF in Ag:**
    - Effective displacement: **{d:.4f} nm**
    - Eigenstrain ε* ≈ **{eps:.3f}**
    - Body force |fᵉq| ≈ **{fmag:.2e} GPa/nm** (~**{fmag * 1e18:.2e} N/m³**)
    """)

# =============================================
# Tab 3: Twin
# =============================================
with tab3:
    st.header("Coherent Twin Boundary")
    st.write("Mirror plane: ABC→AB**C||C**BA")

    col1, col2 = st.columns([1, 1])
    with col1:
        a = st.slider("Lattice constant a (nm)", 0.40, 0.42, Ag_props["a"], 0.001, key="a3")
        h = st.slider("Twin boundary thickness h (nm)", 0.1, 0.4, 0.236, 0.01, key="h3")
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 1.0, 0.1, key="w3")
        mu = st.slider("Shear modulus μ (GPa)", 40.0, 50.0, Ag_props["C44"], 0.5, key="mu3")

    with col2:
        fig, ax = plt.subplots(figsize=(5, 6))
        draw_fcc_defect_schematic(ax, "Twin")
        st.pyplot(fig)

    d, eps, fmag, label = compute_defect_parameters("Twin", a, h, w, mu)
    st.success(f"""
    **Results for Coherent Twin in Ag:**
    - Twinning displacement per boundary: **{d:.4f} nm**
    - Eigenstrain ε* ≈ **{eps:.3f}**
    - Body force |fᵉq| ≈ **{fmag:.2e} GPa/nm** (~**{fmag * 1e18:.2e} N/m³**)
    """)

# Footer
st.markdown("---")
st.caption("Developed for nanomechanics of deformed Ag nanoparticles | Based on Mura's eigenstrain theory | 2025")
