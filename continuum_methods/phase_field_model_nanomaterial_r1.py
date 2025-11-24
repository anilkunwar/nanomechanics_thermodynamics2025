import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

st.set_page_config(page_title="Ag NP Defect Stress Analyzer", layout="wide")
st.title("Live Multi-Scale Stress Analysis in Silver Nanoparticles")
st.markdown("**Phase-field + FFT Elasticity | Real-time von Mises, Principal, Hydrostatic, Deviatoric Stress**")

# =============================================
# Parameters (Silver)
# =============================================
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
C44 = 46.1          # GPa
nu = 0.37           # Poisson ratio for Ag
E = 83              # GPa (Young's modulus)
mu = C44            # Shear modulus
lambda_ = E*nu/((1+nu)*(1-2*nu))  # First Lamé parameter

# Grid
N = 256
dx = 0.1  # nm
x = np.linspace(-N*dx/2, N*dx/2, N)
X, Y = np.meshgrid(x, x)
r = np.sqrt(X**2 + Y**2)

# =============================================
# Sidebar Controls
# =============================================
st.sidebar.header("Defect & Evolution Parameters")
dist_type = st.sidebar.selectbox("Distribution", ["gaussian", "tanh", "linear"], index=0)
w = st.sidebar.slider("Defect width w (nm)", 0.5, 8.0, 2.0, 0.1)
eps0 = st.sidebar.slider("Eigenstrain magnitude ε*", 0.1, 3.0, 0.706, 0.01)
steps = st.sidebar.slider("Evolution steps", 0, 200, 80, 10)
mobility = st.sidebar.slider("Mobility M", 0.1, 5.0, 1.0, 0.1)

# Initialize or evolve defect field
if 'eta' not in st.session_state:
    st.session_state.eta = np.zeros((N, N))

# Simple Allen-Cahn evolution (cached for speed)
@st.cache_data
def evolve_defect(steps, w, eps0, dist_type, mobility):
    eta = np.zeros((N, N))
    center = N // 2
    eta[center-20:center+20, center-20:center+20] = 0.3  # Seed

    for _ in range(steps):
        # Gradient energy
        lap = (np.roll(eta, 1, 0) + np.roll(eta, -1, 0) +
               np.roll(eta, 1, 1) + np.roll(eta, -1, 1) - 4*eta) / dx**2
        
        # Double-well + coupling
        dF = 2*eta*(1-eta)*(eta-0.5) - 0.5*lap
        eta += mobility * 0.01 * (-dF)
        eta = np.clip(eta, 0, 1)

        # Inject localized eigenstrain distribution
        mask = (np.abs(X) < 3*w) & (np.abs(Y) < w/2)
        if dist_type == "gaussian":
            profile = eps0 * np.exp(-((X/(w/2.5))**2 + (Y/(w/3))**2))
        elif dist_type == "tanh":
            profile = (eps0/2) * (1 - np.tanh(np.sqrt(X**2 + Y**2)/(w/2)))
        else:  # linear
            profile = eps0 * np.maximum(1 - np.abs(X)/(w*2), 0)
        eta = np.maximum(eta, profile * mask)

    return eta

eta = evolve_defect(steps, w, eps0, dist_type, mobility)

# =============================================
# FFT Elastic Solver (2D plane strain)
# =============================================
@st.cache_data
def solve_stress_fft(eta, eps0):
    eps_star_xx = eps0 * eta * 0.0
    eps_star_yy = eps0 * eta * 0.0
    eps_star_xy = eps0 * eta * 0.5   # Pure shear from {111}<112>

    # FFT of eigenstrains
    exx_hat = np.fft.fft2(eps_star_xx)
    eyy_hat = np.fft.fft2(eps_star_yy)
    exy_hat = np.fft.fft2(eps_star_xy)

    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    kx = kx * 2j * np.pi
    ky = ky * 2j * np.pi
    k2 = kx**2 + ky**2 + 1e-12

    # Green's function solution (2D plane strain, isotropic)
    denom = 4 * mu * (lambda_ + mu) * k2**2
    u1_hat = -(kx * ((lambda_ + 2*mu) * kx * exx_hat + lambda_ * ky * eyy_hat + 2*mu * ky * exy_hat)) / denom
    u2_hat = -(ky * ((lambda_ + 2*mu) * ky * eyy_hat + lambda_ * kx * exx_hat + 2*mu * kx * exy_hat)) / denom

    u1 = np.real(np.fft.ifft2(u1_hat))
    u2 = np.real(np.fft.ifft2(u2_hat))

    # Strain from displacement
    exx = np.gradient(u1, dx, axis=1)
    eyy = np.gradient(u2, dx, axis=0)
    exy = 0.5 * (np.gradient(u1, dx, axis=0) + np.gradient(u2, dx, axis=1))

    # Total strain = elastic + eigenstrain
    eps_xx = exx - eps_star_xx
    eps_yy = eyy - eps_star_yy
    eps_xy = exy - eps_star_xy

    # Stress (plane strain)
    sigma_xx = (lambda_ + 2*mu) * eps_xx + lambda_ * eps_yy
    sigma_yy = (lambda_ + 2*mu) * eps_yy + lambda_ * eps_xx
    sigma_xy = 2 * mu * eps_xy

    # Derived quantities
    sigma_mean = (sigma_xx + sigma_yy) / 3
    s_xx = sigma_xx - sigma_mean
    s_yy = sigma_yy - sigma_mean
    s_xy = sigma_xy
    von_mises = np.sqrt(0.5 * (s_xx**2 + s_yy**2 + (s_xx + s_yy)**2 + 6*s_xy**2))
    deviatoric = np.sqrt((s_xx**2 + s_yy**2 + 2*s_xy**2)/2 * 3)  # J2

    # Principal stresses
    sigma_avg = (sigma_xx + sigma_yy) / 2
    R = np.sqrt(((sigma_xx - sigma_yy)/2)**2 + sigma_xy**2)
    sigma1 = sigma_avg + R
    sigma2 = sigma_avg - R

    return {
        'von_mises': von_mises,
        'hydrostatic': sigma_mean,
        'sigma1': sigma1,
        'sigma2': sigma2,
        'deviatoric': deviatoric,
        'sigma_xx': sigma_xx,
        'sigma_yy': sigma_yy,
        'eps1': exx + eyy + np.abs(exx - eyy),  # Approx
        'body_force': np.gradient(eps0 * eta, dx) * C44
    }

stress = solve_stress_fft(eta, eps0)

# =============================================
# Visualization Dashboard
# =============================================
cols = st.columns(3)
with cols[0]:
    st.subheader("Defect Order Parameter η")
    fig, ax = plt.subplots()
    im = ax.imshow(eta, extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2], cmap='viridis', origin='lower')
    ax.set_title("η (Defect Density)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)

    st.metric("Max η", f"{eta.max():.3f}")
    st.metric("Defect Volume Fraction", f"{eta.mean():.3f}")

with cols[1]:
    st.subheader("Von Mises Stress")
    fig, ax = plt.subplots()
    im = ax.imshow(stress['von_mises'], extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2],
                   cmap='turbo', origin='lower', vmin=0)
    ax.contour(eta, levels=[0.3], colors='white', alpha=0.6, linewidths=1)
    ax.set_title("σ_vm (GPa)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig)

    st.metric("Peak von Mises", f"{stress['von_mises'].max():.2f} GPa")
    st.metric("Mean Hydrostatic", f"{stress['hydrostatic'].mean():.2f} GPa")

with cols[2]:
    st.subheader("Principal & Derived Stresses")
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    ims = [
        ax[0,0].imshow(stress['sigma1'], cmap='RdBu_r', extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]),
        ax[0,1].imshow(stress['hydrostatic'], cmap='coolwarm', extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]),
        ax[1,0].imshow(stress['deviatoric'], cmap='plasma', extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]),
        ax[1,1].imshow(np.abs(stress['body_force']), cmap='hot', extent=[-N*dx/2, N*dx/2, -N*dx/2, N*dx/2])
    ]
    titles = ["σ₁ (Max Principal)", "σ_h (Hydrostatic)", "σ_dev (Deviatoric)", "|Body Force|"]
    for a, im, t in zip(ax.flat, ims, titles):
        a.set_title(t)
        plt.colorbar(im, ax=a, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)

# =============================================
# Summary Table
# =============================================
st.markdown("### Stress Summary (GPa)")
summary = {
    "Max von Mises": stress['von_mises'].max(),
    "Max |σ₁|": np.abs(stress['sigma1']).max(),
    "Max Hydrostatic (tensile)": np.max(stress['hydrostatic']),
    "Max Hydrostatic (compressive)": np.min(stress['hydrostatic']),
    "Max Deviatoric": stress['deviatoric'].max(),
    "Peak Body Force Density": np.abs(stress['body_force']).max() * 1e18,
}
df = pd.DataFrame(summary.items(), columns=["Quantity", "Value"])
st.table(df)

st.caption("Live FFT + Phase-Field | Real-time von Mises, Principal, Hydrostatic, Deviatoric Stress | Ag NPs 2025")
def write_pvd(files):
    with open('output.pvd', 'w') as f:
        f.write('<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, file in enumerate(files):
            f.write(f'<DataSet timestep="{t}" part="0" file="{file}"/>\n')
        f.write('</Collection>\n</VTKFile>')

# CSV
def write_csv(eta, sigma, step):
    df = pd.DataFrame({'eta': eta.flatten(), 'sigma': sigma.flatten()})
    df.to_csv(f"data_step_{step}.csv", index=False)

# Run simulation
vtu_files = []
for step in range(steps):
    eta = evolve_phase_field(eta, M, kappa, dt, dx, N)
    sigma = compute_stress_fft(eta, epsilon0, C44)
    
    if step % 10 == 0:
        vtu = write_vtu(eta, sigma, step, N, dx)
        vtu_files.append(vtu)
        write_csv(eta, sigma, step)

write_pvd(vtu_files)

# To use:
# - ParaView: Open output.pvd for animation of eta/sigma fields
# - CSV: Import data_step_*.csv for analysis (e.g., in Excel: reshape to 128x128)
