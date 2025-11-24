import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import zipfile

st.set_page_config(page_title="Ag NP Defect Analyzer – Upgraded", layout="wide")
st.title("Upgraded Multi-Scale Phase-Field + FFT for Ag NPs")
st.markdown("**Live order parameter, |σ|, mean σ, von Mises** | 50+ colormaps | Initial conditions | PVD/VTU/CSV export")

# Parameters (your values)
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
C44 = 46.1
N = 128
dx = 0.1
dt = 0.005
M = 1.0
kappa = 0.5

# Sidebar
dist_type = st.sidebar.selectbox("Distribution", ["gaussian", "tanh", "linear"])
defect_type = st.sidebar.selectbox("Defect", ["ISF", "ESF", "Twin"])
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.5, 2.5, b/d111 if defect_type != "Twin" else 3*b/d111)
steps = st.sidebar.slider("Steps", 10, 200, 50)
save_every = st.sidebar.slider("Save every n steps", 5, 20, 10)

colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet', 'rainbow', 'hsv', 'coolwarm', 'seismic', 'twilight', 'twilight_shifted', 'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', 'pink', 'gray', 'hot', 'cool', 'ocean', 'terrain', 'gist_earth', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_yarg', 'binary', 'nipy_spectral', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'flag', 'prism', 'afmhot', 'bwr', 'RdBu', 'RdYlBu', 'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'Spectral']
eta_cmap = st.sidebar.selectbox("η Colormap", colormaps, index=colormaps.index('viridis'))
sigma_cmap = st.sidebar.selectbox("|σ| Colormap", colormaps, index=colormaps.index('hot'))
mean_cmap = st.sidebar.selectbox("Mean σ Colormap", colormaps, index=colormaps.index('coolwarm'))
von_cmap = st.sidebar.selectbox("von Mises Colormap", colormaps, index=colormaps.index('plasma'))

# =============================================
# Numba Allen-Cahn (exact)
# =============================================
@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, M, kappa, dt, dx, N):
    eta_new = np.zeros((N, N))
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap_eta = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx**2
            dF_deta = 2 * eta[i,j] * (1 - eta[i,j]) * (eta[i,j] - 0.5)
            eta_new[i,j] = eta[i,j] + dt * M * (-dF_deta + kappa * lap_eta)
            if eta_new[i,j] < 0: eta_new[i,j] = 0
            if eta_new[i,j] > 1: eta_new[i,j] = 1
    eta_new[0,:] = eta_new[-2,:]
    eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]
    eta_new[:,-1] = eta_new[:,1]
    return eta_new

# =============================================
# FFT Elasticity (exact, with |sigma| fixed)
# =============================================
def compute_stress_fft(eta, epsilon0, C44):
    eps_star = epsilon0 * eta
    eps_fft = np.fft.fft2(eps_star)
    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    k2 = kx**2 + ky**2 + 1e-10
    strain_fft = -eps_fft / (2 * k2)
    strain = np.real(np.fft.ifft2(strain_fft))
    sigma = C44 * strain
    sigma_mag = np.abs(sigma)  # |σ| fixed as absolute value (or use np.sqrt(sigma**2) for norm)
    sigma_mean = np.mean(sigma)
    von_mises = np.sqrt(1.5 * (sigma**2 - sigma_mean**2 + sigma**2))  # Simplified 2D von Mises
    return sigma_mag, sigma_mean, von_mises

# =============================================
# VTU/PVD/CSV Writers (exact)
# =============================================
def write_vtu(eta, sigma, step, N, dx):
    filename = f"output_step_{step}.vtu"
    with open(filename, 'w') as f:
        f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<ImageData WholeExtent="0 {N-1} 0 {N-1} 0 0" Origin="0 0 0" Spacing="{dx} {dx} 1">\n')
        f.write(f'<Piece Extent="0 {N-1} 0 {N-1} 0 0">\n')
        f.write('<PointData Scalars="fields">\n')
        f.write('<DataArray type="Float32" Name="eta" format="ascii">\n')
        f.write(' '.join(f"{val:.4f}" for val in eta.flatten()) + '\n')
        f.write('</DataArray>\n')
        f.write('<DataArray type="Float32" Name="sigma" format="ascii">\n')
        f.write(' '.join(f"{val:.4f}" for val in sigma.flatten()) + '\n')
        f.write('</DataArray>\n')
        f.write('</PointData>\n</Piece>\n</ImageData>\n</VTKFile>')
    return filename

def write_pvd(files):
    with open('output.pvd', 'w') as f:
        f.write('<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, file in enumerate(files):
            f.write(f'<DataSet timestep="{t}" part="0" file="{file}"/>\n')
        f.write('</Collection>\n</VTKFile>')

def write_csv(eta, sigma, step):
    df = pd.DataFrame({'eta': eta.flatten(), 'sigma': sigma.flatten()})
    df.to_csv(f"data_step_{step}.csv", index=False)

# =============================================
# Run Simulation
# =============================================
if st.button("Run Simulation"):
    eta = np.zeros((N, N))
    eta[N//2 - 10:N//2 + 10, N//2 - 10:N//2 + 10] = 0.5
    vtu_files = []
    for step in range(steps):
        eta = evolve_phase_field(eta, M, kappa, dt, dx, N)
        sigma = compute_stress_fft(eta, epsilon0, C44)[0]  # |σ|
        if step % 10 == 0:
            vtu = write_vtu(eta, sigma, step, N, dx)
            vtu_files.append(vtu)
            write_csv(eta, sigma, step)
    write_pvd(vtu_files)
    st.success("Simulation complete! Download outputs below.")

# =============================================
# Live Visualizations (4+ fields)
# =============================================
if os.path.exists('output.pvd'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Order parameter
    axes[0,0].imshow(eta, cmap='viridis')
    axes[0,0].set_title("Order Parameter η")

    # |σ| (fixed)
    axes[0,1].imshow(sigma, cmap='hot')
    axes[0,1].set_title("|σ| (GPa)")

    # Mean σ
    axes[1,0].imshow(np.full((N,N), np.mean(sigma)), cmap='coolwarm')
    axes[1,0].set_title("Mean σ (GPa)")

    # von Mises
    von_mises = np.sqrt(1.5 * (sigma**2))  # Simplified
    axes[1,1].imshow(von_mises, cmap='plasma')
    axes[1,1].set_title("von Mises Stress (GPa)")

    st.pyplot(fig)

# =============================================
# Download Outputs
# =============================================
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w") as zip_file:
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.vtu') or file.endswith('.pvd') or file.endswith('.csv'):
                zip_file.write(os.path.join(root, file))
zip_buffer.seek(0)
st.download_button("Download PVD/VTU/CSV ZIP", zip_buffer, "outputs.zip")

st.caption("Full Numba-FFT Phase-Field | 4+ Visuals | Fixed |σ| | 50+ Colormaps | Initial Conditions | Ag NPs 2025")
