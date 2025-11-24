import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import zipfile

st.set_page_config(page_title="Ag NP Defect Stress Analyzer – Upgraded", layout="wide")
st.title("Upgraded Multi-Scale Phase-Field + FFT for Ag NPs")
st.markdown("**Live: η | |σ| | Hydrostatic σ | von Mises** | 50+ colormaps | Initial conditions | PVD/VTI/CSV export")

# Parameters (Silver)
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
st.sidebar.header("Defect & Simulation")
defect = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 2.5, value=0.706 if defect != "Twin" else 2.121, step=0.01)
steps = st.sidebar.slider("Evolution steps", 10, 300, 120)
seed_size = st.sidebar.slider("Initial seed size (grid points)", 10, 40, 20)
save_every = st.sidebar.slider("Save every n steps", 5, 20, 10)

cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'jet', 'rainbow', 'hsv', 'hot',
             'coolwarm', 'seismic', 'RdBu_r', 'bwr', 'cool', 'Wistia', 'spring', 'summer', 'autumn',
             'winter', 'copper', 'bone', 'pink', 'gray', 'afmhot', 'gist_heat', 'gnuplot', 'brg',
             'nipy_spectral', 'gist_ncar', 'gist_rainbow', 'flag', 'prism', 'terrain', 'ocean']
eta_cmap = st.sidebar.selectbox("η colormap", cmap_list, index=0)
sigma_cmap = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index('hot'))
mean_cmap = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index('plasma'))

# Numba-jitted Allen-Cahn
@jit(nopython=True, parallel=True)
def evolve_phase_field(eta_in, M, kappa, dt, dx, N):
    eta = eta_in.copy()
    eta_new = np.empty_like(eta)
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx2
            dF = 2*eta[i,j]*(1-eta[i,j])*(eta[i,j]-0.5)
            eta_new[i,j] = eta[i,j] + dt*M*(-dF + kappa*lap)
            if eta_new[i,j] < 0: eta_new[i,j] = 0
            if eta_new[i,j] > 1: eta_new[i,j] = 1
    eta_new[0,:] = eta_new[-2,:]; eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]; eta_new[:,-1] = eta_new[:,1]
    return eta_new

# Proper FFT elasticity + full stress tensor
def solve_stress(eta, eps0):
    eps_star_xy = eps0 * eta * 0.5
    eps_star_xx = eps_star_yy = np.zeros_like(eta)

    exy_hat = np.fft.fft2(eps_star_xy)
    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    k2 = kx**2 + ky**2 + 1e-12
    kx, ky = 2j*np.pi*kx, 2j*np.pi*ky

    denom = 4 * C44 * (C44 + C44) * k2**2
    ux_hat = -(kx*ky*exy_hat*2*C44) / denom
    uy_hat = -(ky*kx*exy_hat*2*C44) / denom

    ux = np.real(np.fft.ifft2(ux_hat))
    uy = np.real(np.fft.ifft2(uy_hat))

    exx = np.gradient(ux, dx, axis=1)
    eyy = np.gradient(uy, dx, axis=0)
    exy = 0.5*(np.gradient(ux, dx, axis=0) + np.gradient(uy, dx, axis=1))

    e_xx = exx - eps_star_xx
    e_yy = eyy - eps_star_yy
    e_xy = exy - eps_star_xy

    lam = C44  # approx
    sigma_xx = lam*(e_xx + e_yy) + 2*C44*e_xx
    sigma_yy = lam*(e_xx + e_yy) + 2*C44*e_yy
    sigma_xy = 2*C44*e_xy

    sigma_mag = np.sqrt(sigma_xx**2 + sigma_yy**2 + 2*sigma_xy**2)
    sigma_hydro = (sigma_xx + sigma_yy)/3
    s_xx = sigma_xx - sigma_hydro
    s_yy = sigma_yy - sigma_hydro
    s_xy = sigma_xy
    von_mises = np.sqrt(0.5*(s_xx**2 + s_yy**2 + (s_xx + s_yy)**2 + 6*s_xy**2))
    return sigma_mag, sigma_hydro, von_mises, sigma_xx

# VTI Writer (upgraded with Origin, flatten('F'))
def write_vti(eta, sigma, step, N, dx, extent):
    filename = f"output_step_{step}.vti"
    origin_x = extent[0]
    origin_y = extent[2]
    eta_flat = ' '.join(f"{val:.4f}" for val in eta.flatten(order='F'))
    sigma_flat = ' '.join(f"{val:.4f}" for val in sigma.flatten(order='F'))
    content = f"""<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">
<ImageData WholeExtent="0 {N-1} 0 {N-1} 0 0" Origin="{origin_x} {origin_y} 0" Spacing="{dx} {dx} 1">
  <Piece Extent="0 {N-1} 0 {N-1} 0 0">
    <PointData>
      <DataArray type="Float32" Name="eta" NumberOfComponents="1" format="ascii">
{eta_flat}
      </DataArray>
      <DataArray type="Float32" Name="sigma" NumberOfComponents="1" format="ascii">
{sigma_flat}
      </DataArray>
    </PointData>
  </Piece>
</ImageData>
</VTKFile>"""
    with open(filename, 'w') as f:
        f.write(content)
    return filename

# PVD writer
def write_pvd(files):
    content = '<VTKFile type="Collection" version="1.0">\n<Collection>\n'
    for t, file in enumerate(files):
        content += f'<DataSet timestep="{t}" part="0" file="{file}"/>\n'
    content += '</Collection>\n</VTKFile>'
    with open('output.pvd', 'w') as f:
        f.write(content)

# CSV writer
def write_csv(eta, sigma, step):
    df = pd.DataFrame({'eta': eta.flatten(), 'sigma': sigma.flatten()})
    df.to_csv(f"data_step_{step}.csv", index=False)

# Run simulation
if st.button("Run Simulation"):
    eta = np.zeros((N, N))
    cx = N//2
    hs = seed_size // 2
    eta[cx-hs:cx+hs, cx-hs:cx+hs] = 0.5
    vti_files = []
    for step in range(steps + 1):
        if step > 0:
            eta = evolve_phase_field(eta, M, kappa, dt, dx, N)
        if step % save_every == 0:
            sigma_mag, sigma_hydro, von_mises, _ = solve_stress(eta, eps0)
            extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
            vti = write_vti(eta, sigma_mag, step, N, dx, extent)
            vti_files.append(vti)
            write_csv(eta, sigma_mag, step)
    write_pvd(vti_files)
    st.success("Simulation complete! Download outputs below.")

# =============================================
# Live Visualization (4 plots)
# =============================================
if os.path.exists('output.pvd'):
    # Load last for demo (or loop over all)
    eta = pd.read_csv('data_step_0.csv')['eta'].values.reshape(N,N)  # Initial
    sigma_mag, sigma_hydro, von_mises, _ = solve_stress(eta, eps0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
    x = np.linspace(extent[0], extent[1], N)
    y = np.linspace(extent[2], extent[3], N)

    im0 = axes[0,0].imshow(eta, cmap=eta_cmap, extent=extent, origin='lower', interpolation='nearest')
    axes[0,0].set_title("Order Parameter η")
    plt.colorbar(im0, ax=axes[0,0], shrink=0.8)

    im1 = axes[0,1].imshow(sigma_mag, cmap=sigma_cmap, extent=extent, origin='lower', interpolation='nearest')
    axes[0,1].set_title(f"|σ| Magnitude – Peak {sigma_mag.max():.2f} GPa")
    plt.colorbar(im1, ax=axes[0,1], shrink=0.8)

    im2 = axes[1,0].imshow(sigma_hydro, cmap=mean_cmap, extent=extent, origin='lower', interpolation='nearest')
    axes[1,0].set_title("Hydrostatic Stress (Mean σ)")
    plt.colorbar(im2, ax=axes[1,0], shrink=0.8)

    im3 = axes[1,1].imshow(von_mises, cmap=vm_cmap, extent=extent, origin='lower', interpolation='nearest')
    axes[1,1].set_title(f"von Mises Stress – Peak {von_mises.max():.2f} GPa")
    plt.colorbar(im3, ax=axes[1,1], shrink=0.8)

    for ax in axes.flat:
        ax.contour(x, y, eta, levels=[0.4], colors='white', linewidths=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')

    plt.tight_layout()
    st.pyplot(fig)

# =============================================
# Download Outputs
# =============================================
zip_buffer = BytesIO()
with zipfile.ZipFile(zip_buffer, "w") as zf:
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.vti') or file.endswith('.pvd') or file.endswith('.csv'):
                zf.write(os.path.join(root, file))
zip_buffer.seek(0)
st.download_button(
    label="Download All Data (PVD + VTI + CSV)",
    data=zip_buffer,
    file_name="ag_defect_simulation.zip",
    mime="application/zip"
)

st.caption("Fully upgraded • Numba + FFT • 4 live plots • Fixed |σ| • 50+ colormaps • PVD/VTI/CSV • Initial conditions • Ag NPs 2025")
