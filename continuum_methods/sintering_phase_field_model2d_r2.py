import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import plotly.express as px
import pyvista as pv
import os
import io
import time
from pathlib import Path
import shutil

# =============================================
# Advanced Phase-Field Model for Defect-Driven Multi-Particle Sintering
# =============================================
st.set_page_config(page_title="Advanced Defect-Driven Sintering Simulator", layout="wide")
st.title("Advanced Phase-Field Modeling of Defects and Twins in Ag Nanoparticle Sintering")
st.markdown("""
**Simulates role of defects (ISF/ESF/Twins) in multi-particle sintering**  
Custom geometry (NP radius, defect position) • Physical units (nm, GPa) • Colormap options  
Run simulation • Download VTU/PVD/CSV
""")

# =============================================
# Physical Parameters (Silver, nm/GPa units)
# =============================================
a = 0.4086  # nm (lattice constant)
b = a / np.sqrt(6)  # Shockley partial magnitude (nm)
d111 = a / np.sqrt(3)  # {111} spacing (nm)
C44 = 46.1  # GPa (shear modulus)

# Grid in nm
N = 128
dx = 0.1  # nm/voxel
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# Sidebar: Geometry Tuning & Defect Role
# =============================================
st.sidebar.header("Geometry & Defect Controls")
np_radius = st.sidebar.slider("Nanoparticle radius (nm)", 5.0, 15.0, 10.0, 0.5)
n_particles = st.sidebar.slider("Number of NPs", 2, 5, 3)
defect_type = st.sidebar.selectbox("Defect Type in NPs", ["None", "ISF", "ESF", "Twin"])
defect_pos = st.sidebar.slider("Defect position offset (nm)", -5.0, 5.0, 0.0, 0.5)
defect_size = st.sidebar.slider("Defect inclusion size (nm)", 1.0, 5.0, 2.0, 0.1)

# Physical defaults based on defect role
if defect_type == "ISF":
    default_eps = 0.707
    default_kappa = 0.6
elif defect_type == "ESF":
    default_eps = 1.414
    default_kappa = 0.7
elif defect_type == "Twin":
    default_eps = 2.121
    default_kappa = 0.3  # Sharper for twins
else:
    default_eps = 0.0
    default_kappa = 0.5

eps0 = st.sidebar.slider("Eigenstrain ε*", 0.0, 3.0, default_eps, 0.01)
kappa = st.sidebar.slider("Interface κ (nm²)", 0.1, 2.0, default_kappa, 0.05)

st.sidebar.header("Simulation")
steps = st.sidebar.slider("Steps", 50, 500, 200, 50)
dt = 0.01  # ns (fixed for stability)

st.sidebar.header("Visualization Colors")
cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'rainbow', 'hsv', 'coolwarm', 'bwr', 'seismic', 'twilight', 'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', 'pink', 'gray', 'hot', 'cool', 'ocean', 'terrain', 'gist_earth', 'gist_rainbow', 'gist_ncar', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_yarg', 'binary', 'nipy_spectral', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'flag', 'prism', 'afmhot', 'RdBu', 'RdYlBu', 'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'Spectral']
eta_cmap = st.sidebar.selectbox("η colormap", cmap_list, index=0)
sigma_cmap = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index('hot'))
hydro_cmap = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index('plasma'))

# =============================================
# Create Initial Multi-Particle Geometry
# =============================================
def create_initial_multi_np(n_particles, np_radius, defect_type, defect_pos, defect_size):
    eta = np.zeros((N, N))
    # Random placement of NPs
    np.random.seed(42)
    centers = []
    for _ in range(n_particles):
        cx = np.random.uniform(np_radius, N*dx - np_radius)
        cy = np.random.uniform(np_radius, N*dx - np_radius)
        centers.append((cx, cy))
        mask = (X - cx)**2 + (Y - cy)**2 <= np_radius**2
        eta[mask] = 1.0

    # Add defects inside random NP if selected
    if defect_type != "None":
        np_idx = np.random.randint(0, n_particles)
        cx, cy = centers[np_idx]
        defect_mask = ((X - cx - defect_pos)**2 + (Y - cy)**2 <= defect_size**2)
        eta[defect_mask] = 0.6  # Initial defect seed

    eta += 0.02 * np.random.randn(N, N)
    return np.clip(eta, 0.0, 1.0)

st.subheader("Initial Multi-Particle Configuration")
init_eta = create_initial_multi_np(n_particles, np_radius, defect_type, defect_pos, defect_size)
fig0, ax0 = plt.subplots(figsize=(7,6))
im0 = ax0.imshow(init_eta, extent=extent, cmap=eta_cmap, origin='lower')
ax0.contour(X, Y, init_eta, levels=[0.4], colors='white', linewidths=2)
ax0.set_title(f"Initial η – {defect_type} Defects in {n_particles} NPs", fontsize=16, fontweight='bold')
ax0.set_xlabel("x (nm)", fontsize=14); ax0.set_ylabel("y (nm)", fontsize=14)
plt.colorbar(im0, ax=ax0, shrink=0.8)
ax0.tick_params(labelsize=13, width=2, length=6)
for spine in ax0.spines.values():
    spine.set_linewidth(2.5)
st.pyplot(fig0)

# =============================================
# Numba-safe Allen-Cahn (full 2D)
# =============================================
@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx2
            dF = 2*eta[i,j]*(1-eta[i,j])*(eta[i,j]-0.5)
            eta_new[i,j] = eta[i,j] + dt * (-dF + kappa * lap)
            eta_new[i,j] = np.maximum(0.0, np.minimum(1.0, eta_new[i,j]))
    eta_new[0,:] = eta_new[-2,:]; eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]; eta_new[:,-1] = eta_new[:,1]
    return eta_new

# =============================================
# FFT Stress Solver
# =============================================
@st.cache_data
def compute_stress_fields(eta, eps0):
    eps_xy = eps0 * eta * 0.5
    exy_hat = np.fft.fft2(eps_xy)
    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    k2 = kx**2 + ky**2 + 1e-12
    kx, ky = 2j*np.pi*kx, 2j*np.pi*ky

    denom = 8 * C44**2 * k2**2
    ux_hat = -(kx*ky*exy_hat*2*C44) / denom
    uy_hat = -(ky*kx*exy_hat*2*C44) / denom

    ux = np.real(np.fft.ifft2(ux_hat))
    uy = np.real(np.fft.ifft2(uy_hat))

    exx = np.gradient(ux, dx, axis=1)
    eyy = np.gradient(uy, dx, axis=0)
    exy = 0.5*(np.gradient(ux, dx, axis=0) + np.gradient(uy, dx, axis=1)) - eps_xy

    lam = C44
    sxx = lam*(exx + eyy) + 2*C44*exx
    syy = lam*(exx + eyy) + 2*C44*eyy
    sxy = 2*C44*exy

    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy)/3
    von_mises = np.sqrt(0.5*((sxx-sigma_hydro)**2 + (syy-sigma_hydro)**2 +
                             (sxx+syy-2*sigma_hydro)**2 + 6*sxy**2))

    return sigma_mag, sigma_hydro, von_mises

# =============================================
# Run Simulation
# =============================================
if st.button("Run Phase-Field Sintering", type="primary"):
    with st.spinner("Running advanced sintering simulation..."):
        eta = init_eta.copy()
        history = []
        for step in range(steps + 1):
            if step > 0:
                eta = evolve_phase_field(eta, kappa, dt=0.004, dx=dx, N=N)
            if step % save_every == 0 or step == steps:
                sm, sh, vm = compute_stress_fields(eta, eps0)
                history.append((eta.copy(), sm.copy(), sh.copy(), vm.copy()))
        st.session_state.history = history
        st.success(f"Complete! {len(history)} frames – {defect_type} defects in {n_particles} NPs")

# =============================================
# Live Results
# =============================================
if 'history' in st.session_state:
    frame = st.slider("Frame", 0, len(st.session_state.history)-1, len(st.session_state.history)-1)
    eta, sigma_mag, sigma_hydro, von_mises = st.session_state.history[frame]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.linspace(extent[0], extent[1], N)
    y = np.linspace(extent[2], extent[3], N)

    fields = [eta, sigma_mag, sigma_hydro, von_mises]
    cmaps  = [eta_cmap, sigma_cmap, hydro_cmap, vm_cmap]
    titles = [
        "Phase Field η (Sintering Evolution)",
        f"|σ| Magnitude – {sigma_mag.max():.2f} GPa",
        f"Hydrostatic Stress – {sigma_hydro.mean():.2f} GPa",
        f"von Mises – {von_mises.max():.2f} GPa"
    ]

    for ax, field, cmap, title in zip(axes.flat, fields, cmaps, titles):
        im = ax.imshow(field, extent=extent, cmap=cmap, origin='lower')
        ax.contour(x, y, eta, levels=[0.4], colors='white', linewidths=2)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel("x (nm)", fontsize=14)
        ax.set_ylabel("y (nm)", fontsize=14)
        ax.tick_params(labelsize=13, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    st.pyplot(fig)

    # Sintering metric: neck growth (interface area increase)
    initial_interface = np.sum(np.abs(np.gradient(init_eta)[0]) + np.abs(np.gradient(init_eta)[1]))
    final_interface = np.sum(np.abs(np.gradient(eta)[0]) + np.abs(np.gradient(eta)[1]))
    metric = (initial_interface - final_interface) / initial_interface if initial_interface > 0 else 0
    st.success(f"Sintering metric (relative interface reduction due to necking): {metric:.4f}")

    # =============================================
    # Download (all 4 fields)
    # =============================================
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, sm, sh, vm) in enumerate(st.session_state.history):
            df = pd.DataFrame({
                'eta': e.flatten(order='F'), 'stress_magnitude': sm.flatten(order='F'),
                'hydrostatic': sh.flatten(order='F'), 'von_mises': vm.flatten(order='F')
            })
            zf.writestr(f"frame_{i:04d}.csv", df.to_csv(index=False))

            flat = lambda a: ' '.join(f"{x:.6f}" for x in a.flatten(order='F'))
            vti = f"""<VTKFile type="ImageData" version="1.0">
<ImageData WholeExtent="0 {N-1} 0 {N-1} 0 0" Origin="{extent[0]} {extent[2]} 0" Spacing="{dx} {dx} 1">
  <Piece Extent="0 {N-1} 0 {N-1} 0 0">
    <PointData>
      <DataArray type="Float32" Name="eta" format="ascii">{flat(e)}</DataArray>
      <DataArray type="Float32" Name="stress_magnitude" format="ascii">{flat(sm)}</DataArray>
      <DataArray type="Float32" Name="hydrostatic" format="ascii">{flat(sh)}</DataArray>
      <DataArray type="Float32" Name="von_mises" format="ascii">{flat(vm)}</DataArray>
    </PointData>
  </Piece>
</ImageData>
</VTKFile>"""
            zf.writestr(f"frame_{i:04d}.vti", vti)

        pvd = '<VTKFile type="Collection" version="1.0">\n<Collection>\n'
        for i in range(len(st.session_state.history)):
            pvd += f'  <DataSet timestep="{i*save_every*dt:.6f}" file="frame_{i:04d}.vti"/>\n'
        pvd += '</Collection>\n</VTKFile>'
        zf.writestr("simulation.pvd", pvd)

    buffer.seek(0)
    st.download_button(
        "Download Full Results (PVD + VTI + CSV)",
        buffer,
        "multi_np_sintering.zip",
        "application/zip"
    )

st.caption("Advanced Multi-Particle Sintering • Defect Role • Geometry Tuning • Physical Units • 2025")
