import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
import io

# =============================================
# UPDATED: Physically Motivated Defect Mechanics
# =============================================
st.set_page_config(page_title="Defect Mechanics in FCC Ag (2025)", layout="wide")
st.title("Defect-Induced Eigenstrain & Body Force in FCC Silver Nanoparticles")
st.markdown("""
**Intrinsic & Extrinsic Stacking Faults | Coherent Twin Boundaries**  
Interactive tool with **multiple eigenstrain distributions** and **corrected peak calculations**.
""")

# Silver parameters
a = 0.4086  # nm
b = a / np.sqrt(6)           # Shockley partial: |b| ≈ 0.1667 nm
d111 = a / np.sqrt(3)        # ≈ 0.2359 nm
C44 = 46.1                   # GPa (shear modulus)

# FIXED eigenstrains for each defect type
EPS_ISF = b / d111                    # ≈ 0.706
EPS_ESF = (2 * b) / (2 * d111)        # ≈ 0.706 (same as ISF!)
EPS_TWIN = (3 * b) / d111             # ≈ 2.121

# =============================================
# Enhanced eigenstrain distributions
# =============================================
def eigenstrain_distribution(x, eps_star, w, dist_type='gaussian'):
    """Compute eigenstrain spatial profile using different distributions"""
    if dist_type == 'gaussian':
        # Current approach: Gaussian with width parameter
        sigma = w / 2.5
        return eps_star * np.exp(-(x/sigma)**2)
    
    elif dist_type == 'tanh':
        # More physical: smooth transition between states
        return (eps_star/2) * (1 - np.tanh(2*x/w))
    
    elif dist_type == 'linear':
        # Simplest physical model: linear decay
        return eps_star * np.maximum(1 - np.abs(x)/w, 0)
    
    else:  # Default to Gaussian if unknown
        sigma = w / 2.5
        return eps_star * np.exp(-(x/sigma)**2)

# =============================================
# High-quality schematic generator
# =============================================
def draw_fcc_ag_defect(ax, defect_type):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    layers = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    y_pos = np.linspace(0, 9, 10)
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    x_offsets = {'A': [-2, -1, 0, 1], 'B': [-2.5, -1.5, -0.5, 0.5], 'C': [-2, -1, 0, 1]}

    for i, (layer, y) in enumerate(zip(layers, y_pos)):
        for x in x_offsets[layer]:
            ax.add_patch(Circle((x, y), 0.42, color=colors[layer], ec='black', lw=1.2, zorder=3))
        ax.text(-4.6, y, layer, va='center', fontsize=14, fontweight='bold', color=colors[layer])

    # Defect visualization
    if defect_type == "ISF":
        ax.axhspan(4.0, 5.0, color='red', alpha=0.15)
        ax.text(3.0, 4.5, "ISF\n(1 HCP layer)", fontsize=14, color='red', fontweight='bold')
        ax.add_patch(FancyArrowPatch((0, 4.6), (2, 4.6), arrowstyle='->', color='red', lw=3))
    elif defect_type == "ESF":
        ax.axhspan(3.5, 5.5, color='purple', alpha=0.18)
        ax.text(3.0, 4.5, "ESF\n(2 HCP layers)", fontsize=14, color='purple', fontweight='bold')
        ax.add_patch(FancyArrowPatch((-1, 5.0), (1, 5.0), arrowstyle='->', color='purple', lw=2))
        ax.add_patch(FancyArrowPatch((0, 4.0), (2, 4.0), arrowstyle='->', color='purple', lw=2))
    elif defect_type == "Twin":
        ax.plot([-5, 5], [4.5, 4.5], 'k--', lw=2.5, alpha=0.8)
        ax.axhspan(4.3, 4.7, color='green', alpha=0.25)
        ax.text(3.0, 5.8, "Coherent Twin Boundary", fontsize=14, color='darkgreen', fontweight='bold')
        ax.text(3.0, 3.2, "Mirror plane (HCP-like)", fontsize=12, color='gray')

# =============================================
# UPDATED: Enhanced eigenstrain & body force plotting
# =============================================
def plot_eigenstrain_bodyforce(w, eps_star, defect_name, dist_type='gaussian'):
    x = np.linspace(-3*w, 3*w, 600)
    
    # Compute eigenstrain profile
    eps_profile = eigenstrain_distribution(x, eps_star, w, dist_type)
    
    # Compute body force (bipolar!)
    dx = x[1] - x[0]
    f_profile = -C44 * np.gradient(eps_profile, dx)
    
    # Calculate ACTUAL peak (not the simplified estimate)
    peak_force_actual = np.max(np.abs(f_profile))
    peak_force_simple = C44 * (eps_star / w)  # Simplified estimate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
    
    # Plot 1: Eigenstrain (always positive)
    ax1.plot(x, eps_profile, 'b-', lw=3, label=f'ε*(x) [ε*₀ = {eps_star:.3f}]')
    ax1.set_ylabel('Eigenstrain ε*', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3, ls=':')
    ax1.legend()
    
    # Plot 2: Body force (bipolar with fill)
    ax2.plot(x, f_profile, 'r-', lw=2, label='f{eq}(x)')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(x, 0, f_profile, where=f_profile>0, alpha=0.4, color='red', label='Pushing (+f)')
    ax2.fill_between(x, 0, f_profile, where=f_profile<0, alpha=0.4, color='blue', label='Pulling (-f)')
    ax2.set_ylabel('Body Force (GPa/nm)', color='r', fontsize=12)
    ax2.set_xlabel('Position x (nm)', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3, ls=':')
    ax2.legend()
    
    title = f"{defect_name} — {dist_type.capitalize()} Distribution\n"
    title += f"Peak: {peak_force_actual:.1f} GPa/nm (actual) vs {peak_force_simple:.1f} GPa/nm (simple estimate)"
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    
    return fig, peak_force_actual, peak_force_simple

# =============================================
# Constant Eigenstrain Table
# =============================================
def show_eigenstrain_table():
    st.markdown("### Fixed Eigenstrain Values")
    st.markdown("""
    | Defect Type | Displacement (nm) | Height h (nm) | Eigenstrain ε* |
    |-------------|-------------------|---------------|----------------|
    | ISF | d = b = 0.1667 | h = d₁₁₁ = 0.2359 | ε* = 0.706 |
    | ESF | d = 2b = 0.3334 | h = 2d₁₁₁ = 0.4718 | ε* = 0.706 |
    | Twin | d = 3b = 0.5001 | h = d₁₁₁ = 0.2359 | ε* = 2.121 |
    """)

# =============================================
# Tabs
# =============================================
tab1, tab2, tab3, tab4 = st.tabs(["ISF (ε* = 0.706)", "ESF (ε* = 0.706)", "Twin (ε* = 2.121)", "Theory & Comparison"])

# Distribution type selector
dist_type = st.sidebar.selectbox(
    "Eigenstrain Distribution",
    ["gaussian", "tanh", "linear"],
    index=0,
    help="Choose the spatial distribution of eigenstrain"
)

# --- ISF ---
with tab1:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        fig_sch = plt.figure(figsize=(6, 8), dpi=300)
        ax = fig_sch.add_subplot(111)
        draw_fcc_ag_defect(ax, "ISF")
        st.pyplot(fig_sch)
    with col2:
        st.markdown("### Intrinsic Stacking Fault (ISF)")
        st.latex(r"d = b = \frac{a}{\sqrt{6}} \approx 0.1667\,\text{nm}, \quad h = d_{111}, \quad \epsilon^* = \frac{d}{h} = 0.706")
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 1.5, key="w1")
        
        fig_curve, peak_actual, peak_simple = plot_eigenstrain_bodyforce(w, EPS_ISF, "ISF", dist_type)
        st.pyplot(fig_curve)
        
        st.success(f"""
        **Fixed Parameters:**  
        Displacement: d = b = {b:.4f} nm  
        Height: h = d₁₁₁ = {d111:.4f} nm  
        Eigenstrain: ε* = {EPS_ISF:.3f} (constant)
        
        **Body Force Results:**  
        Actual peak: **{peak_actual:.1f} GPa/nm**  
        Simple estimate: {peak_simple:.1f} GPa/nm  
        Force density: **{peak_actual * 1e18:.2e} N/m³**
        """)

# --- ESF ---
with tab2:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        fig_sch = plt.figure(figsize=(6, 8), dpi=300)
        ax = fig_sch.add_subplot(111)
        draw_fcc_ag_defect(ax, "ESF")
        st.pyplot(fig_sch)
    with col2:
        st.markdown("### Extrinsic Stacking Fault (ESF)")
        st.latex(r"d = 2b = \frac{a}{\sqrt{3}} \approx 0.3334\,\text{nm}, \quad h = 2d_{111}, \quad \epsilon^* = \frac{d}{h} = 0.706")
        w = st.slider("Gradient width w (nm)", 0.5, 5.0, 2.0, key="w2")
        
        fig_curve, peak_actual, peak_simple = plot_eigenstrain_bodyforce(w, EPS_ESF, "ESF", dist_type)
        st.pyplot(fig_curve)
        
        st.success(f"""
        **Fixed Parameters:**  
        Displacement: d = 2b = {2*b:.4f} nm  
        Height: h = 2d₁₁₁ = {2*d111:.4f} nm  
        Eigenstrain: ε* = {EPS_ESF:.3f} (constant)
        
        **Body Force Results:**  
        Actual peak: **{peak_actual:.1f} GPa/nm**  
        Simple estimate: {peak_simple:.1f} GPa/nm  
        Force density: **{peak_actual * 1e18:.2e} N/m³**
        """)

# --- Twin ---
with tab3:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        fig_sch = plt.figure(figsize=(6, 8), dpi=300)
        ax = fig_sch.add_subplot(111)
        draw_fcc_ag_defect(ax, "Twin")
        st.pyplot(fig_sch)
    with col2:
        st.markdown("### Coherent Twin Boundary")
        st.latex(r"d = 3b = \frac{a\sqrt{2}}{2} \approx 0.5001\,\text{nm}, \quad h = d_{111}, \quad \epsilon^* = \frac{d}{h} = 2.121")
        w = st.slider("Gradient width w (nm)", 0.3, 4.0, 1.0, key="w3")
        
        fig_curve, peak_actual, peak_simple = plot_eigenstrain_bodyforce(w, EPS_TWIN, "Twin", dist_type)
        st.pyplot(fig_curve)
        
        st.success(f"""
        **Fixed Parameters:**  
        Displacement: d = 3b = {3*b:.4f} nm  
        Height: h = d₁₁₁ = {d111:.4f} nm  
        Eigenstrain: ε* = {EPS_TWIN:.3f} (constant)
        
        **Body Force Results:**  
        Actual peak: **{peak_actual:.1f} GPa/nm**  
        Simple estimate: {peak_simple:.1f} GPa/nm  
        Force density: **{peak_actual * 1e18:.2e} N/m³**
        """)

# --- Theory & Comparison ---
with tab4:
    st.header("Theoretical Framework")
    show_eigenstrain_table()
    
    st.markdown("""
    ### Key Updates in This Version:
    
    1. **Fixed Eigenstrains**: Each defect type has constant ε* based on displacement/height ratio
    2. **Multiple Distributions**: Compare Gaussian, Tanh, and Linear profiles
    3. **Corrected Peaks**: Show actual computed peaks vs. simple estimates
    4. **Bipolar Forces**: Explicit visualization of pushing (+f) and pulling (-f) regions
    
    ### Distribution Types:
    - **Gaussian**: Mathematical convenience, smooth decay
    - **Tanh**: Physically motivated for phase boundaries  
    - **Linear**: Simplest physical model
    
    ### Why Different Distributions Give Similar Results:
    All smooth, localized distributions produce:
    - **Colossal body forces** (10¹⁸–10¹⁹ N/m³)
    - **Force dipoles** (pushing+pulling) across defects
    - **Qualitatively similar** driving forces for low-temperature sintering
    """)
    
    # Comparison plot
    st.header("Distribution Comparison")
    w_compare = st.slider("Comparison width w (nm)", 0.5, 3.0, 1.0, key="w_compare")
    
    fig_compare, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
    x = np.linspace(-3*w_compare, 3*w_compare, 600)
    
    distributions = ['gaussian', 'tanh', 'linear']
    colors = ['blue', 'red', 'green']
    
    for dist, color in zip(distributions, colors):
        eps_profile = eigenstrain_distribution(x, EPS_TWIN, w_compare, dist)
        f_profile = -C44 * np.gradient(eps_profile, x[1]-x[0])
        
        ax1.plot(x, eps_profile, color=color, lw=2, label=dist.capitalize())
        ax2.plot(x, f_profile, color=color, lw=2, label=f"{dist.capitalize()} (peak: {np.max(np.abs(f_profile)):.1f})")
    
    ax1.set_ylabel('Eigenstrain ε*')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Twin Boundary (ε* = {EPS_TWIN:.3f}) - Eigenstrain Distributions")
    
    ax2.set_ylabel('Body Force (GPa/nm)')
    ax2.set_xlabel('Position (nm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Corresponding Body Force Profiles")
    
    plt.tight_layout()
    st.pyplot(fig_compare)

st.caption("Updated: Fixed eigenstrains, multiple distributions, corrected peaks | Ag NPs @ 2025")
