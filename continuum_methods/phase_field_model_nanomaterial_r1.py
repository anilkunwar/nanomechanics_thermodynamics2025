import numpy as np
from numba import jit, prange
import pandas as pd
import os

# Simple 2D multi-scale model: Phase-field for defect evolution + Elasticity via FFT
# Assumptions: Square domain (NP cross-section), periodic BC for FFT
# Phase-field: Allen-Cahn type for defect 'order parameter' eta (0 = perfect FCC, 1 = defect HCP)
# Eigenstrain from eta, stress from elasticity
# Outputs: VTU/PVD (simple VTK writer), CSV

# Parameters
N = 128  # Grid size (N x N)
dx = 0.1  # nm/grid
dt = 0.005  # Time step (smaller for stability)
steps = 50  # Simulation steps
M = 1.0  # Mobility
kappa = 0.5  # Gradient energy coeff
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
epsilon0 = b / d111  # ~0.706 for demo (ISF-like)
C44 = 46.1  # GPa

# Initialize phase-field eta (initial defect seed)
eta = np.zeros((N, N))
eta[N//2 - 10:N//2 + 10, N//2 - 10:N//2 + 10] = 0.5  # Softer initial to avoid instability

# Numba-jit phase-field evolution (Allen-Cahn, stabilized)
@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, M, kappa, dt, dx, N):
    eta_new = np.zeros((N, N))
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap_eta = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx**2
            dF_deta = 2 * eta[i,j] * (1 - eta[i,j]) * (eta[i,j] - 0.5)  # Stabilized double-well derivative
            eta_new[i,j] = eta[i,j] + dt * M * (-dF_deta + kappa * lap_eta)
            if eta_new[i,j] < 0: eta_new[i,j] = 0
            if eta_new[i,j] > 1: eta_new[i,j] = 1
    # Periodic boundaries approx
    eta_new[0,:] = eta_new[-2,:]
    eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]
    eta_new[:,-1] = eta_new[:,1]
    return eta_new

# FFT-based elasticity solver for stress from eigenstrain
def compute_stress_fft(eta, epsilon0, C44):
    # Eigenstrain field (simple shear component)
    eps_star = epsilon0 * eta
    
    # FFT for strain (simple Poisson approx for iso elastic)
    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    k2 = kx**2 + ky**2 + 1e-10  # Avoid div zero
    
    eps_fft = np.fft.fft2(eps_star)
    
    # Spectral solution for strain (simplified Green's function approx)
    strain_fft = -eps_fft / (2 * k2)  # Poisson-like for volumetric source
    strain = np.real(np.fft.ifft2(strain_fft))
    
    # Stress = C * strain (simple shear stress)
    sigma = C44 * strain
    return sigma

# Simple VTU writer (ImageData for grid)
def write_vtu(eta, sigma, step, N, dx):
    filename = f"output_step_{step}.vtu"
    with open(filename, 'w') as f:
        f.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<ImageData WholeExtent="0 {N-1} 0 {N-1} 0 0" Origin="0 0 0" Spacing="{dx} {dx} 1">\n')
        f.write(f'<Piece Extent="0 {N-1} 0 {N-1} 0 0">\n')
        f.write('<PointData Scalars="fields">\n')
        f.write('<DataArray type="Float32" Name="eta" format="ascii">\n')
        for val in eta.flatten():
            f.write(f"{val:.4f} ")
        f.write('\n</DataArray>\n')
        f.write('<DataArray type="Float32" Name="sigma" format="ascii">\n')
        for val in sigma.flatten():
            f.write(f"{val:.4f} ")
        f.write('\n</DataArray>\n')
        f.write('</PointData>\n')
        f.write('</Piece>\n</ImageData>\n</VTKFile>')
    return filename

# PVD collector
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
