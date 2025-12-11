import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Sintering Temperature vs Hydrostatic Stress in Ag Nanoparticles")
st.markdown("**Interactive calculator for twin/defect-induced sintering temperature reduction**")

# -------------------------------------------------
# Sidebar parameters
# -------------------------------------------------
st.sidebar.header("Parameters")
Ts0 = st.sidebar.slider("Baseline Ts (K)", 500, 700, 623)
Qa_kJ = st.sidebar.slider("Qa (kJ/mol)", 70.0, 120.0, 90.0, step=5.0)
Omega_m3 = st.sidebar.slider("Ω (×10⁻²⁹ m³)", 8.0, 12.0, 10.0) * 1e-29
sigma_max_GPa = st.sidebar.slider("Max |σ_h| (GPa)", 1.0, 10.0, 5.0)

# Log-spaced stresses for table
stresses_GPa = np.logspace(-1, np.log10(sigma_max_GPa), 6)
stresses_Pa = stresses_GPa * 1e9

# -------------------------------------------------
# Temperature calculation (scalar only)
# -------------------------------------------------
@st.cache_data
def calculate_Ts(sigma_Pa, Ts0, Qa_kJ, Omega_m3):
    """Scalar input/output for single stress value"""
    Qa_J_atom = Qa_kJ * 1000 / 6.022e23  # Convert to J/atom
    delta_Q = Omega_m3 * sigma_Pa       # J/atom
    Q_eff = Qa_J_atom - delta_Q
    Ts_eff = Ts0 * (Q_eff / Qa_J_atom)
    return max(Ts_eff, 300)  # Lower bound

# -------------------------------------------------
# Main curve computation
# -------------------------------------------------
sigma_plot = np.logspace(-6, np.log10(sigma_max_GPa * 2), 400) * 1e9  # Pa
Ts_plot = np.array([calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in sigma_plot])

# -------------------------------------------------
# Plotting
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Main T_s curve
ax.semilogx(sigma_plot/1e9, Ts_plot, linewidth=3, color='blue', label='T_s(|σ_h|)')

# Zero-stress point for visibility
ax.plot(1e-6, Ts0, 'ro', markersize=10, label='σ = 0')

# Baseline T_s(0)
ax.axhline(Ts0, color='red', linestyle='--', linewidth=2)

# Visual arrow showing the temperature drop
Ts_min = Ts_plot[-1]
delta_T = Ts0 - Ts_min

ax.annotate(
    '', xy=(sigma_max_GPa, Ts_min), xytext=(1e-6, Ts0),
    arrowprops=dict(arrowstyle='<->', linewidth=2, color='red')
)

ax.text(
    1e-3,
    (Ts0 + Ts_min) / 2,
    f"ΔT_s = {delta_T:.0f} K",
    color='red',
    fontsize=12
)

ax.set_xlabel('|σ_h| (GPa) - Log Scale')
ax.set_ylabel('Sintering temperature T_s (K)')
ax.set_title('Stress-Induced Reduction in Ag Nanoparticle Sintering Temperature')
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# Metrics Display
# -------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Baseline T_s", f"{Ts0} K")
col2.metric("Max ΔT_s", f"{delta_T:.0f} K", f"-{100*delta_T/Ts0:.0f}%")
col3.metric("At max σ_h", f"{sigma_max_GPa:.1f} GPa")

# -------------------------------------------------
# Prediction Table
# -------------------------------------------------
st.subheader("Predictions at key stress levels (log scale)")

Ts_values = [calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in stresses_Pa]
delta_Ts = [Ts0 - t for t in Ts_values]
pct_red = [100*(Ts0 - t)/Ts0 for t in Ts_values]

table_data = {
    "|σ_h| (GPa)": [f"{s:.2f}" for s in stresses_GPa],
    "T_s (K)": [f"{t:.0f}" for t in Ts_values],
    "ΔT_s (K)": [f"{d:.0f}" for d in delta_Ts],
    "% Reduction": [f"{p:.0f}%" for p in pct_red]
}

st.table(table_data)

# -------------------------------------------------
# Physics Note
# -------------------------------------------------
st.markdown("""
**Physics relation**:  
\\[
T_s(|σ_h|) ≈ T_s(0)\, \\left(1 - \\frac{Ω |σ_h|}{Q_a} \\right)
\\]

**What this means:**  
- Even moderate hydrostatic stress (twin stress ~1–5 GPa) can lower sintering temperature significantly.  
- Higher defect stresses (5–10 GPa) can reduce T_s by **hundreds of Kelvin**, enabling ultra-low-temperature densification.
""")
