import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Sintering Temperature vs Hydrostatic Stress in Ag Nanoparticles")
st.markdown("**Interactive calculator for twin/defect-induced sintering temperature reduction**")

# Parameters sidebar
st.sidebar.header("Parameters")
Ts0 = st.sidebar.slider("Baseline Ts (K)", 500, 700, 623)
Qa_kJ = st.sidebar.slider("Qa (kJ/mol)", 70.0, 120.0, 90.0, step=5.0)
Omega_m3 = st.sidebar.slider("Ω (×10⁻²⁹ m³)", 8.0, 12.0, 10.0) * 1e-29
sigma_max_GPa = st.sidebar.slider("Max |σ_h| (GPa)", 1.0, 10.0, 5.0)

# Fixed log-spaced stresses for table (GPa)
stresses_GPa = np.logspace(-1, np.log10(sigma_max_GPa), 6)  # 0.1 to max, 6 points
stresses_Pa = stresses_GPa * 1e9

# Calculate function - FIXED: always scalar input/output
@st.cache_data
def calculate_Ts(sigma_Pa, Ts0, Qa_kJ, Omega_m3):
    """Scalar input/output for single stress value"""
    Qa_J_atom = Qa_kJ * 1000 / 6.022e23  # Convert to J/atom
    kB = 1.381e-23  # J/K (not needed for linear approx)
    delta_Q = Omega_m3 * sigma_Pa  # J/atom (tensile effect)
    Q_eff = Qa_J_atom - delta_Q
    Ts_eff = Ts0 * (Q_eff / Qa_J_atom)  # Linear T_s ∝ Q
    return max(Ts_eff, 300)  # Floor at 300K

# Main plot - log scale x-axis
sigma_plot = np.logspace(0, np.log10(sigma_max_GPa*2), 100) * 1e9  # Pa
Ts_plot = np.array([calculate_Ts(s, Ts0, Qa_kJ, Omega_m3) for s in sigma_plot])

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(sigma_plot/1e9, Ts_plot, 'b-', linewidth=2, label='T_s(|σ_h|)')
ax.axhline(Ts0, color='r', linestyle='--', label=f'T_s(0) = {Ts0} K')
ax.set_xlabel('|σ_h| (GPa) - Log Scale')
ax.set_ylabel('Sintering temperature T_s (K)')
ax.set_title('Stress-Induced Reduction in Ag Nanoparticle Sintering Temperature')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# Metrics
col1, col2, col3 = st.columns(3)
delta_max = Ts0 - Ts_plot[-1]
st.metric("Baseline T_s", f"{Ts0} K", None, col1)
st.metric("Max ΔT_s", f"{delta_max:.0f} K", f"-{100*delta_max/Ts0:.0f}%", col2)
st.metric("At max σ_h", f"{sigma_max_GPa:.1f} GPa", None, col3)

# FIXED Predictions table - scalar calls only
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

st.markdown("""
**Physics**: $T_s(|σ_h|) ≈ T_s(0) × (1 - Ω|σ_h|/Q_a)$  
**Log scale**: Captures full range from low twin stress to high defect stress  
**Twins**: ~1-5 GPa → 10-30% T_s reduction (60-190 K drop)
""")
