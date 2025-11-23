import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# App title
st.title("Eigenstrain and Volumetric Body Force Quantifier for Dislocations in Ag NPs")

# Tabs for each defect type
tab1, tab2, tab3 = st.tabs(["Intrinsic Stacking Fault (ISF)", "Extrinsic Stacking Fault (ESF)", "Twin Boundary"])

# Default parameters for Ag (in nm and GPa)
default_a = 0.409  # Lattice constant (nm)
default_h = 0.236  # Interplanar spacing d_{111} = a / sqrt(3) (nm)
default_w = 1.0    # Gradient width (nm)
default_mu = 46.0  # C44 shear modulus (GPa)

# Function to compute and plot
def compute_and_plot(defect_type, a, h, w, mu):
    # Displacement magnitude d (nm)
    if defect_type == "ISF":
        d = a / np.sqrt(6)  # Shockley partial |b_p| = a / sqrt(6) ~0.167 nm
    elif defect_type == "ESF":
        d = a / np.sqrt(3)  # Effective for two partials ~0.236 nm
    else:  # Twin
        d = a / (2 * np.sqrt(6))  # Effective per boundary ~0.083 nm (twinning shear 1/sqrt(2))

    # Eigenstrain epsilon* (dimensionless shear)
    epsilon_star = d / h

    # Body force magnitude approximation (GPa/nm = 10^9 N/m^3 / 10^{-9} = 10^{18} N/m^3)
    # Full: f_i^{eq} = -C_ijkl ∂ epsilon_kl^* / ∂x_j
    # Simplified shear: f^{eq} ≈ -mu * (epsilon_star / w)
    f_eq = -mu * (epsilon_star / w)

    # Display results
    st.write(f"**Displacement magnitude (d):** {d:.3f} nm")
    st.write(f"**Eigenstrain (ε* ≈ d / h):** {epsilon_star:.3f} (dimensionless)")
    st.write(f"**Volumetric body force magnitude (f^{eq} ≈ -μ ⋅ (ε* / w)):** {f_eq:.3e} GPa/nm (~10^{18} N/m³)")

    # 1D profile plot (assume linear gradient over w)
    x = np.linspace(-w, w, 100)
    epsilon_profile = epsilon_star * (1 - np.abs(x) / (w / 2)) * (np.abs(x) <= w / 2)  # Triangular for simplicity
    f_profile = -mu * np.gradient(epsilon_profile, x[1] - x[0])

    fig, ax1 = plt.subplots()
    ax1.plot(x, epsilon_profile, 'b-', label='Eigenstrain ε*(x)')
    ax1.set_xlabel('Position x (nm)')
    ax1.set_ylabel('Eigenstrain (dimensionless)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x, f_profile, 'r--', label='Body force f^{eq}(x) (GPa/nm)')
    ax2.set_ylabel('Body Force (GPa/nm)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    st.pyplot(fig)

# Tab 1: ISF
with tab1:
    st.header("Intrinsic Stacking Fault (ISF)")
    st.write("An ISF in FCC is a single-layer disruption (ABC → AB C B C), induced by one Shockley partial dislocation slip.")
    
    # Sliders
    a = st.slider("Lattice constant a (nm)", 0.3, 0.5, default_a, key="a1")
    h = st.slider("Defect thickness h (nm)", 0.1, 0.5, default_h, key="h1")
    w = st.slider("Gradient width w (nm)", 0.5, 5.0, default_w, key="w1")
    mu = st.slider("Shear modulus μ (GPa)", 20.0, 60.0, default_mu, key="mu1")
    
    # Illustration
    st.image("https://www.researchgate.net/publication/325271542/figure/fig6/AS:628770927022083@1526922079568/Schematic-illustrations-of-stacking-faults-and-micro-twin-in-an-fcc-lattice-a-Perfect.png", 
             caption="Schematic of ISF in FCC (left part)", use_column_width=True)
    
    compute_and_plot("ISF", a, h, w, mu)

# Tab 2: ESF
with tab2:
    st.header("Extrinsic Stacking Fault (ESF)")
    st.write("An ESF is two overlapping ISFs (ABC → AB C A B C), from two partial slips.")
    
    # Sliders
    a = st.slider("Lattice constant a (nm)", 0.3, 0.5, default_a, key="a2")
    h = st.slider("Defect thickness h (nm)", 0.1, 0.5, default_h, key="h2")
    w = st.slider("Gradient width w (nm)", 0.5, 5.0, default_w, key="w2")
    mu = st.slider("Shear modulus μ (GPa)", 20.0, 60.0, default_mu, key="mu2")
    
    # Illustration
    st.image("https://www.researchgate.net/publication/325271542/figure/fig6/AS:628770927022083@1526922079568/Schematic-illustrations-of-stacking-faults-and-micro-twin-in-an-fcc-lattice-a-Perfect.png", 
             caption="Schematic of ESF in FCC (middle part)", use_column_width=True)
    
    compute_and_plot("ESF", a, h, w, mu)

# Tab 3: Twins
with tab3:
    st.header("Twin Boundary")
    st.write("A coherent twin in FCC is a mirror-symmetric stack (ABC → AB C B A), from multiple partial slips.")
    
    # Sliders
    a = st.slider("Lattice constant a (nm)", 0.3, 0.5, default_a, key="a3")
    h = st.slider("Defect thickness h (nm)", 0.1, 0.5, default_h, key="h3")
    w = st.slider("Gradient width w (nm)", 0.5, 5.0, default_w, key="w3")
    mu = st.slider("Shear modulus μ (GPa)", 20.0, 60.0, default_mu, key="mu3")
    
    # Illustration
    st.image("https://www.researchgate.net/publication/325271542/figure/fig6/AS:628770927022083@1526922079568/Schematic-illustrations-of-stacking-faults-and-micro-twin-in-an-fcc-lattice-a-Perfect.png", 
             caption="Schematic of Twin in FCC (right part)", use_column_width=True)
    
    compute_and_plot("Twin", a, h, w, mu)
