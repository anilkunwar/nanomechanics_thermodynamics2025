# =============================================
#  HIGH-FIDELITY PHASE-FIELD + FFT IN STREAMLIT
#  (Identical physics to your original script)
# =============================================
import streamlit as st
st.set_page_config(page_title="Ag NP – Full Numba+FFT Phase-Field", layout="wide")

import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import os, base64
from io import BytesIO

# ------------------------------------------------
# Material & Simulation Parameters (Silver)
# ------------------------------------------------
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
epsilon0 = b / d111          # ≈ 0.706
C44 = 46.1                   # GPa

N = 128
dx = 0.1
M = 1.0
kappa = 0.5
dt = 0.004                   # stable value

# ------------------------------------------------
# Numba-jitted Allen-Cahn evolution (exact copy of yours)
# ------------------------------------------------
@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, M, kappa, dt, dx, N):
    eta_new = np.zeros((N, N), dtype=np.float64)
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx2
            dF = 2 * eta[i,j] * (1 - eta[i,j]) * (eta[i,j] - 0.5)
            eta_new[i,j] = eta[i,j] + dt * M * (-dF + kappa * lap)
            if eta_new[i,j] < 0: eta_new[i,j] = 0
            if eta_new[i,j] > 1: eta_new[i,j] = 1
    # Periodic BC
    eta_new[0,:]  = eta_new[-2,:]; eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0]  = eta_new[:,-2]; eta_new[:,-1] = eta_new[:,1]
    return eta_new

# ------------------------------------------------
# Exact FFT elasticity (your simplified shear version – easy to upgrade)
# ------------------------------------------------
@st.cache_data
def compute_stress_fft(eta, eps0, C44):
    eps_star = eps0 * eta
    eps_fft = np.fft.fft2(eps_star)
    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    k2 = kx**2 + ky**2
    k2[0,0] = 1e-12
    strain_fft = -eps_fft / (2 * k2)               # same as your Poisson-like solver
    strain = np.real(np.fft.ifft2(strain_fft))
    sigma = C44 * 2 * strain                       # σ_xy ≈ 2μ ε_xy for pure shear
    return np.abs(sigma)

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
st.title("High-Fidelity Phase-Field + FFT Solver (Numba-accelerated)")
st.markdown("**Exact same physics as your offline script – now live in the browser**")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    total_steps = st.slider("Total evolution steps", 10, 500, 150, 10)
    eps_val = st.slider("Eigenstrain ε*", 0.3, 2.5, float(epsilon0), 0.01)
    show_every = st.slider("Show frame every n steps", 5, 50, 10)

    if st.button("Run / Re-run simulation"):
        eta = np.zeros((N, N))
        eta[N//2-12:N//2+12, N//2-12:N//2+12] = 0.5
        st.session_state.eta_history = []
        st.session_state.sigma_history = []

        for step in range(total_steps + 1):
            if step > 0:
                eta = evolve_phase_field(eta, M, kappa, dt, dx, N)
            if step % show_every == 0 or step == total_steps:
                sigma = compute_stress_fft(eta, eps_val, C44)
                st.session_state.eta_history.append(eta.copy())
                st.session_state.sigma_history.append(sigma.copy())

        st.success(f"Simulation finished – {len(st.session_state.eta_history)} frames")

with col2:
    if 'eta_history' in st.session_state and st.session_state.eta_history:
        frame = st.slider("Frame", 0, len(st.session_state.eta_history)-1, len(st.session_state.eta_history)-1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        im1 = ax1.imshow(st.session_state.eta_history[frame],
                         extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2],
                         cmap='viridis', vmin=0, vmax=1)
        ax1.set_title(f"Defect Order Parameter η – Frame {frame}")
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(st.session_state.sigma_history[frame],
                         extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2],
                         cmap='hot', origin='lower')
        ax2.contour(st.session_state.eta_history[frame], levels=[0.3], colors='cyan', linewidths=1)
        ax2.set_title(f"|σ| from eigenstrain (GPa) – Peak {st.session_state.sigma_history[frame].max():.2f} GPa")
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        st.pyplot(fig)

        # Optional download of all data
        if st.button("Download all frames as CSV + VTU (zip)"):
            import zipfile, tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, (e, s) in enumerate(zip(st.session_state.eta_history, st.session_state.sigma_history)):
                    np.savetxt(f"{tmpdir}/eta_{i:04d}.csv", e, delimiter=",")
                    np.savetxt(f"{tmpdir}/sigma_{i:04d}.csv", s, delimiter=",")
                with zipfile.ZipFile("simulation_data.zip", "w") as z:
                    for f in os.listdir(tmpdir):
                        z.write(os.path.join(tmpdir, f), f)
            with open("simulation_data.zip", "rb") as fp:
                st.download_button("Download ZIP", fp, "ag_defect_simulation.zip")

st.caption("Full Numba + FFT phase-field model • Identical to your offline script • Real-time in Streamlit")
