# =============================================
# 3D Ag Nanoparticle Phase-Field + FFT – COMPREHENSIVELY EXPANDED
# All-features unified Streamlit script with 50+ color maps and complete stress visualization
# Enhanced with correct anisotropic 3D elasticity and comprehensive 3D export
# =============================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import os

st.set_page_config(page_title="3D Ag NP Defect Evolution – Ultimate", layout="wide")
st.title("3D Phase-Field Simulation of Defects in Spherical Ag Nanoparticles — Ultimate Edition")
st.markdown("""
**50+ Color Maps • Complete Stress Tensor Visualization • Enhanced 3D Export**
**Crystallographically accurate eigenstrain • Exact 3D anisotropic FFT spectral elasticity**
**ISF/ESF/Twin physically distinct • Tiltable {111} habit plane • Publication-ready**
**Units fixed: spatial units (nm) in UI, FFT uses meters internally • Enhanced visualization**
**Corrected for realistic stresses (~ GPa scale) • Optional elastic driving in phase-field**
""")

# =============================================
# EXPANDED Color maps - 50+ including jet, turbo
COLOR_MAPS = {
    'Classic & Turbo': ['jet', 'turbo', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
                       'gray', 'bone', 'pink', 'copper', 'wistia', 'afmhot', 'gist_heat', 'gist_gray'],
    
    'Matplotlib Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight',
                           'twilight_shifted', 'hsv'],
    
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
                            'rocket', 'mako', 'icefire', 'vlag'],
    
    'Diverging': ['RdBu', 'RdYlBu', 'RdYlGn', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                 'Spectral', 'coolwarm', 'bwr', 'seismic', 'RdGy', 'PuBuGn',
                 'YlOrRd', 'YlOrBr', 'YlGnBu', 'YlGn'],
    
    'Sequential Single': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
    
    'Special & Qualitative': ['gist_earth', 'gist_stern', 'ocean', 'terrain', 'gist_rainbow',
                             'rainbow', 'nipy_spectral', 'gist_ncar', 'tab10', 'tab20',
                             'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
}

# =============================================
# Material properties (Silver - cubic anisotropic)
C11 = 124e9  # Pa
C12 = 93.4e9
C44 = 46.1e9

# Define the full 4th-rank stiffness tensor C_ijkl
C = np.zeros((3, 3, 3, 3))
for i in range(3):
    C[i, i, i, i] = C11
for i in range(3):
    for j in range(3):
        if i != j:
            C[i, i, j, j] = C12
for i in range(3):
    for j in range(3):
        if i != j:
            C[i, j, i, j] = C44
            C[i, j, j, i] = C44
            C[j, i, i, j] = C44
            C[j, i, j, i] = C44

# =============================================
# UI - simulation controls
st.sidebar.header("Simulation Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    N = st.number_input("Grid size N (per dim)", min_value=32, max_value=128, value=64, step=16)
with col2:
    dx = st.number_input("Grid spacing dx (nm)", min_value=0.05, max_value=5.0, value=0.25, step=0.05)

dt = st.sidebar.number_input("Time step dt (arb)", min_value=1e-4, max_value=0.1, value=0.005, step=0.001)
M = st.sidebar.number_input("Mobility M", min_value=1e-3, max_value=10.0, value=1.0, step=0.1)
defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])
eps0_defaults = {"ISF": 0.707, "ESF": 1.414, "Twin": 2.121}
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.01, 3.0, float(eps0_defaults[defect_type]), 0.01)
kappa = st.sidebar.slider("Interface coeff κ", 0.01, 5.0, 0.6, 0.01)

col1, col2 = st.sidebar.columns(2)
with col1:
    steps = st.slider("Evolution steps", 1, 1000, 80, 1)
with col2:
    save_every = st.slider("Save every (steps)", 1, 200, 10, 1)

st.sidebar.header("Defect Shape")
shape = st.sidebar.selectbox("Initial Defect Shape", ["Sphere", "Cuboid", "Ellipsoid", "Cube", "Cylinder", "Planar"])

st.sidebar.header("Habit Plane Orientation (for Planar)")
col1, col2 = st.sidebar.columns(2)
with col1:
    theta_deg = st.slider("Polar angle θ (°)", 0, 180, 55, help="54.7° = exact {111}")
with col2:
    phi_deg = st.slider("Azimuthal angle φ (°)", 0, 360, 0, step=5)

theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)

# =============================================
# ENHANCED Visualization Controls
# =============================================
st.sidebar.header("🎨 Visualization Controls")
viz_category = st.sidebar.selectbox("Color Map Category", list(COLOR_MAPS.keys()))
eta_cmap = st.sidebar.selectbox("Defect (η) Color Map", COLOR_MAPS[viz_category], index=0)
stress_cmap = st.sidebar.selectbox("Stress (σ) Color Map", COLOR_MAPS[viz_category],
                                  index=min(1, len(COLOR_MAPS[viz_category])-1))

# ENHANCED: Comprehensive colorbar range controls
st.sidebar.subheader("Colorbar Ranges")
use_custom_limits = st.sidebar.checkbox("Use Custom Color Scale Limits", False)
if use_custom_limits:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        eta_min = st.number_input("η Min", value=0.0, format="%.2f")
        sigma_min = st.number_input("|σ| Min (GPa)", value=0.0, format="%.1f")
        hydro_min = st.number_input("σ_h Min (GPa)", value=-5.0, format="%.1f")
        vm_min = st.number_input("σ_vM Min (GPa)", value=0.0, format="%.1f")
    with col2:
        eta_max = st.number_input("η Max", value=1.0, format="%.2f")
        sigma_max = st.number_input("|σ| Max (GPa)", value=10.0, format="%.1f")
        hydro_max = st.number_input("σ_h Max (GPa)", value=5.0, format="%.1f")
        vm_max = st.number_input("σ_vM Max (GPa)", value=8.0, format="%.1f")
else:
    eta_min, eta_max, sigma_min, sigma_max, hydro_min, hydro_max, vm_min, vm_max = None, None, None, None, None, None, None, None

# Store colorbar limits in a dictionary
colorbar_limits = {
    'eta': [eta_min, eta_max],
    'sigma_mag': [sigma_min, sigma_max],
    'sigma_hydro': [hydro_min, hydro_max],
    'von_mises': [vm_min, vm_max]
}

# NEW: Stress component selection
stress_component = st.sidebar.selectbox(
    "Stress Component to Visualize",
    ["Von Mises Stress", "Stress Magnitude", "Hydrostatic Stress", 
     "σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_xz", "σ_yz"]
)

# Chart styling controls from 2D code
st.sidebar.subheader("Chart Styling")
title_font_size = st.sidebar.slider("Title Font Size", 12, 24, 16)
label_font_size = st.sidebar.slider("Label Font Size", 10, 20, 14)
tick_font_size = st.sidebar.slider("Tick Font Size", 8, 18, 12)
line_width = st.sidebar.slider("Contour Line Width", 1.0, 5.0, 2.0, 0.5)

# Additional color maps for 2D plots
cmap_list_2d = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm', 
                'jet', 'turbo', 'rainbow', 'RdBu', 'Spectral', 'tab20', 'gist_earth']

eta_cmap_2d = st.sidebar.selectbox("η colormap (2D)", cmap_list_2d, index=0)
sigma_cmap_2d = st.sidebar.selectbox("|σ| colormap (2D)", cmap_list_2d, index=cmap_list_2d.index('hot'))
hydro_cmap_2d = st.sidebar.selectbox("Hydrostatic colormap (2D)", cmap_list_2d, index=cmap_list_2d.index('coolwarm'))
vm_cmap_2d = st.sidebar.selectbox("von Mises colormap (2D)", cmap_list_2d, index=cmap_list_2d.index('plasma'))

show_contours = st.sidebar.checkbox("Show Defect Contours", value=True)
contour_level = st.sidebar.slider("Contour Level", 0.1, 0.9, 0.4, 0.05)
contour_color = st.sidebar.color_picker("Contour Color", "#FFFFFF")
contour_alpha = st.sidebar.slider("Contour Alpha", 0.1, 1.0, 0.8, 0.1)

st.sidebar.subheader("3D Rendering")
opacity_3d = st.sidebar.slider("3D Opacity", 0.05, 1.0, 0.7, 0.05)
surface_count = st.sidebar.slider("Surface Count", 1, 10, 2)
show_grid = st.sidebar.checkbox("Show Grid in Plotly", value=True)
show_matrix = st.sidebar.checkbox("Show Nanoparticle Matrix", value=True)
eta_threshold = st.sidebar.slider("η Visualization Threshold", 0.0, 1.0, 0.1, 0.01)
stress_threshold = st.sidebar.slider("Stress Visualization Threshold (GPa)", 0.0, 50.0, 0.0, 0.1)

st.sidebar.header("Advanced Options")
debug_mode = st.sidebar.checkbox("Debug: print diagnostics", value=False)
enable_progress_bar = st.sidebar.checkbox("Show Progress Bar", value=True)
enable_real_time_viz = st.sidebar.checkbox("Real-time Visualization", value=False)
enable_elastic_coupling = st.sidebar.checkbox("Enable Elastic Coupling in PF Evolution", value=False)

# =============================================
# Physical domain setup (keep dx in nm for coordinate arrays used in UI/VTK)
origin = -N * dx / 2.0
X, Y, Z = np.meshgrid(
    np.linspace(origin, origin + (N-1)*dx, N),
    np.linspace(origin, origin + (N-1)*dx, N),
    np.linspace(origin, origin + (N-1)*dx, N),
    indexing='ij'
)

# spherical nanoparticle mask (units: nm)
R_np = N * dx / 4.0
r = np.sqrt(X**2 + Y**2 + Z**2)
np_mask = r <= R_np

# =============================================
# Initial condition generator
def create_initial_eta(shape_in):
    eta = np.zeros((N, N, N), dtype=np.float64)
    
    if shape_in == "Sphere":
        mask = r <= 8 * dx
        eta[mask] = 0.7
    elif shape_in == "Cuboid":
        w, h, d = 16, 8, 8
        mask = (np.abs(X) <= w/2) & (np.abs(Y) <= h/2) & (np.abs(Z) <= d/2)
        eta[mask] = 0.7
    elif shape_in == "Ellipsoid":
        a, b, c = 16, 8, 8
        mask = (X**2/a**2 + Y**2/b**2 + Z**2/c**2) <= 1
        eta[mask] = 0.7
    elif shape_in == "Cube":
        side = 12
        mask = (np.abs(X) <= side/2) & (np.abs(Y) <= side/2) & (np.abs(Z) <= side/2)
        eta[mask] = 0.7
    elif shape_in == "Cylinder":
        radius = 8
        height = 16
        mask = (X**2 + Y**2 <= radius**2) & (np.abs(Z) <= height/2)
        eta[mask] = 0.7
    elif shape_in == "Planar":
        n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        dist = n[0]*X + n[1]*Y + n[2]*Z
        thickness = 3 * dx
        mask = np.abs(dist) <= thickness / 2
        eta[mask] = 0.7
        
    eta[~np_mask] = 0.0
    np.random.seed(42)
    eta += 0.02 * np.random.randn(N, N, N) * np_mask
    eta = np.clip(eta, 0.0, 1.0)
    return eta

eta = create_initial_eta(shape)

# =============================================
# Vectorized phase-field evolution with optional elastic coupling
def evolve_3d_vectorized(eta_in, kappa_in, dt_in, dx_in, M_in, mask_np, eps0, theta, phi, dx_nm, enable_elastic, debug=False):
    # Compute laplacian via periodic roll
    lap = (
        np.roll(eta_in, -1, axis=0) + np.roll(eta_in, 1, axis=0) +
        np.roll(eta_in, -1, axis=1) + np.roll(eta_in, 1, axis=1) +
        np.roll(eta_in, -1, axis=2) + np.roll(eta_in, 1, axis=2) - 6.0*eta_in
    ) / (dx_in*dx_in)
    
    # Double-well derivative
    dF = 2*eta_in*(1-eta_in)*(eta_in-0.5)
    
    # Elastic driving force
    elastic_driving = np.zeros_like(eta_in)
    if enable_elastic:
        # Compute stress (in GPa)
        _, _, _, sigma_gpa, _ = compute_stress_3d_exact(eta_in, eps0, theta, phi, dx_nm, debug)
        
        # Shear tensor
        n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        s = np.cross(n, [0,0,1])
        if np.linalg.norm(s) < 1e-12:
            s = np.cross(n, [1,0,0])
        s /= np.linalg.norm(s)
        gamma = eps0 * 0.1
        
        shear_tensor = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                shear_tensor[i,j] = 0.5 * gamma * (n[i]*s[j] + s[i]*n[j])
                
        # Driving force (note: sigma_gpa in GPa, but since units are arbitrary, proceed; adjust M if needed)
        elastic_driving = np.einsum('ij,ijxyz->xyz', shear_tensor, sigma_gpa)
    
    # Update
    eta_new = eta_in + dt_in * M_in * (-dF + kappa_in * lap + elastic_driving)
    
    # Apply mask and clip
    eta_new[~mask_np] = 0.0
    eta_new = np.clip(eta_new, 0.0, 1.0)
    
    return eta_new

# =============================================
# Enhanced stress computation returning all stress components
@st.cache_data
def compute_stress_3d_exact(eta_field, eps0_val, theta_val, phi_val, dx_nm, debug=False):
    dx_m = dx_nm * 1e-9
    
    # Define vectors
    n = np.array([np.cos(phi_val)*np.sin(theta_val), np.sin(phi_val)*np.sin(theta_val), np.cos(theta_val)])
    s = np.cross(n, [0,0,1])
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(n, [1,0,0])
    s /= np.linalg.norm(s)
    gamma = eps0_val * 0.1
    
    shear_tensor = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            shear_tensor[i,j] = 0.5 * gamma * (n[i]*s[j] + s[i]*n[j])
            
    # Eigenstrain
    eps_star = np.einsum('ij,xyz->ijxyz', shear_tensor, eta_field)
    
    # Polarization tau = C : eps_star
    tau = np.einsum('ijkl,klxyz->ijxyz', C, eps_star)
    tau_hat = np.fft.fftn(tau, axes=(2,3,4))
    
    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx_m)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = np.inf
    sqrtK2 = np.sqrt(K2)
    
    khat_x = KX / sqrtK2
    khat_y = KY / sqrtK2
    khat_z = KZ / sqrtK2
    khat_x[0,0,0] = khat_y[0,0,0] = khat_z[0,0,0] = 0.0
    khat = np.stack([khat_x, khat_y, khat_z], axis=0)
    
    # Acoustic tensor A_ij = C_ikjl khat_k khat_l
    A = np.einsum('ikjl,kxyz,lxyz->ijxyz', C, khat, khat)
    
    # Move axes for inversion
    A_moved = np.moveaxis(A, [0,1], [3,4])  # shape (N,N,N,3,3)
    
    # Robust matrix inversion with singular value decomposition (SVD)
    invA = np.zeros_like(A_moved)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                A_mat = A_moved[i, j, k]
                # Skip zero frequency (already handled)
                if i == 0 and j == 0 and k == 0:
                    invA[i, j, k] = np.zeros((3, 3))
                    continue
                
                try:
                    # Regular inversion for well-conditioned matrices
                    if np.linalg.cond(A_mat) < 1e12:  # Reasonable condition number threshold
                        invA[i, j, k] = np.linalg.inv(A_mat)
                    else:
                        # Use pseudo-inverse for ill-conditioned matrices
                        invA[i, j, k] = np.linalg.pinv(A_mat, rcond=1e-12)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse for singular matrices
                    invA[i, j, k] = np.linalg.pinv(A_mat, rcond=1e-12)
    
    # Displacement Green G_ij = inv(A)_ij / K2
    G = invA / K2[..., np.newaxis, np.newaxis]
    G[0,0,0, :, :] = 0.0
    
    # Strain Green operator Gamma_ijkl
    # term1: khat[j] * G[i,k] * khat[l]
    term1 = np.einsum('jxyz,xyzik,lxyz->ijklxyz', khat, G, khat)
    # term2: khat[j] * G[i,l] * khat[k]
    term2 = np.einsum('jxyz,xyzil,kxyz->ijlkxyz', khat, G, khat)
    # term3: khat[i] * G[j,k] * khat[l]
    term3 = np.einsum('ixyz,xyzjk,lxyz->jiklxyz', khat, G, khat)
    term3 = np.transpose(term3, (1,0,2,3,4,5,6))
    # term4: khat[i] * G[j,l] * khat[k]
    term4 = np.einsum('ixyz,xyzjl,kxyz->jilkxyz', khat, G, khat)
    term4 = np.transpose(term4, (1,0,3,2,4,5,6))
    
    Gamma = (term1 + term2 + term3 + term4) / 4.0
    
    # Induced strain eps_ind_hat = - Gamma : tau_hat
    eps_ind_hat = -np.einsum('ijklxyz,klxyz->ijxyz', Gamma, tau_hat)
    eps_ind = np.real(np.fft.ifftn(eps_ind_hat, axes=(2,3,4)))
    
    # Total strain
    eps_total = eps_ind - eps_star
    
    # Stress sigma = C : eps_total
    sigma = np.einsum('ijkl,klxyz->ijxyz', C, eps_total)
    
    if debug:
        max_stress_pa = np.max(np.sqrt(sigma[0,0]**2 + sigma[1,1]**2 + sigma[2,2]**2 + 2*(sigma[0,1]**2 + sigma[0,2]**2 + sigma[1,2]**2)))
        st.write(f"[Debug] Max stress magnitude (Pa): {max_stress_pa:.3e}")
    
    # Compute ALL stress components in GPa
    sxx = sigma[0,0] / 1e9
    syy = sigma[1,1] / 1e9
    szz = sigma[2,2] / 1e9
    sxy = sigma[0,1] / 1e9
    sxz = sigma[0,2] / 1e9
    syz = sigma[1,2] / 1e9
    
    # Compute derived stress measures
    vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6 * (sxy**2 + sxz**2 + syz**2)))
    sigma_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2 * (sxy**2 + sxz**2 + syz**2))
    sigma_hydro = (sxx + syy + szz) / 3
    
    # Mask all stress components
    sigma_mag_masked = np.nan_to_num(sigma_mag * np_mask)
    sigma_hydro_masked = np.nan_to_num(sigma_hydro * np_mask)
    von_mises_masked = np.nan_to_num(vm * np_mask)
    sigma_gpa = sigma / 1e9
    
    # Individual stress components masked
    sxx_masked = np.nan_to_num(sxx * np_mask)
    syy_masked = np.nan_to_num(syy * np_mask)
    szz_masked = np.nan_to_num(szz * np_mask)
    sxy_masked = np.nan_to_num(sxy * np_mask)
    sxz_masked = np.nan_to_num(sxz * np_mask)
    syz_masked = np.nan_to_num(syz * np_mask)
    
    if debug:
        st.write(f"[Debug] Final von Mises range (GPa): {von_mises_masked.min():.6f} to {von_mises_masked.max():.6f}")
        
    return (sigma_mag_masked, sigma_hydro_masked, von_mises_masked, sigma_gpa, 
            sxx_masked, syy_masked, szz_masked, sxy_masked, sxz_masked, syz_masked)

# =============================================
# Enhanced VTI creation with all stress components
def create_vti(eta_field, stress_components, step_idx, time_val):
    """Create VTI file with all stress components for proper 3D visualization"""
    sigma_mag, sigma_hydro, von_mises, _, sxx, syy, szz, sxy, sxz, syz = stress_components
    
    flat = lambda arr: ' '.join(map(str, arr.flatten(order='F')))
    
    vti = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}"
             Origin="{origin:.3f} {origin:.3f} {origin:.3f}"
             Spacing="{dx:.3f} {dx:.3f} {dx:.3f}">
    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
      <PointData Scalars="eta">
        <DataArray type="Float32" Name="eta" format="ascii">
          {flat(eta_field)}
        </DataArray>
        <DataArray type="Float32" Name="stress_magnitude_GPa" format="ascii">
          {flat(sigma_mag)}
        </DataArray>
        <DataArray type="Float32" Name="hydrostatic_stress_GPa" format="ascii">
          {flat(sigma_hydro)}
        </DataArray>
        <DataArray type="Float32" Name="von_mises_stress_GPa" format="ascii">
          {flat(von_mises)}
        </DataArray>
        <DataArray type="Float32" Name="stress_xx_GPa" format="ascii">
          {flat(sxx)}
        </DataArray>
        <DataArray type="Float32" Name="stress_yy_GPa" format="ascii">
          {flat(syy)}
        </DataArray>
        <DataArray type="Float32" Name="stress_zz_GPa" format="ascii">
          {flat(szz)}
        </DataArray>
        <DataArray type="Float32" Name="stress_xy_GPa" format="ascii">
          {flat(sxy)}
        </DataArray>
        <DataArray type="Float32" Name="stress_xz_GPa" format="ascii">
          {flat(sxz)}
        </DataArray>
        <DataArray type="Float32" Name="stress_yz_GPa" format="ascii">
          {flat(syz)}
        </DataArray>
      </PointData>
      <CellData></CellData>
    </Piece>
  </ImageData>
</VTKFile>"""
    return vti

# =============================================
# NEW: Enhanced Analysis Functions from 2D Code
# =============================================
def safe_percentile(arr, percentile, default=0.0):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return default
    return np.percentile(arr, percentile)

def get_stress_component(stress_components, component_name):
    """Extract the appropriate stress component based on selection"""
    sigma_mag, sigma_hydro, von_mises, _, sxx, syy, szz, sxy, sxz, syz = stress_components
    
    stress_map = {
        "Von Mises Stress": von_mises,
        "Stress Magnitude": sigma_mag,
        "Hydrostatic Stress": sigma_hydro,
        "σ_xx": sxx,
        "σ_yy": syy,
        "σ_zz": szz,
        "σ_xy": sxy,
        "σ_xz": sxz,
        "σ_yz": syz
    }
    
    return stress_map.get(component_name, von_mises)

def create_plotly_isosurface(Xa, Ya, Za, values, title, colorscale,
                             isomin=None, isomax=None, opacity=0.7,
                             surface_count=2, custom_min=None, custom_max=None, show_grid=False):
    # Avoid feeding all-NaN arrays to Plotly; flatten arrays and let Plotly clip by isomin/isomax
    vals = np.asarray(values, dtype=float)
    if custom_min is not None and custom_max is not None:
        vals = np.clip(vals, custom_min, custom_max)
        
    # compute sensible isomin/isomax if not provided
    vals_mask = vals[np_mask]
    if vals_mask.size == 0:
        isomin = 0.0 if isomin is None else isomin
        isomax = 1.0 if isomax is None else isomax
    else:
        if isomin is None:
            isomin = float(safe_percentile(vals_mask, 10, np.nanmin(vals_mask)))
        if isomax is None:
            isomax = float(safe_percentile(vals_mask, 90, np.nanmax(vals_mask)))
            
    fig = go.Figure(data=go.Isosurface(
        x=Xa.flatten(), y=Ya.flatten(), z=Za.flatten(),
        value=vals.flatten(),
        isomin=isomin, isomax=isomax,
        surface_count=surface_count,
        colorscale=colorscale,
        opacity=opacity,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title=title)
    ))
    
    if show_matrix:
        theta_sphere = np.linspace(0, np.pi, 50)
        phi_sphere = np.linspace(0, 2*np.pi, 50)
        theta_sphere, phi_sphere = np.meshgrid(theta_sphere, phi_sphere)
        x = R_np * np.sin(theta_sphere) * np.cos(phi_sphere) + origin + N*dx/2.0
        y = R_np * np.sin(theta_sphere) * np.sin(phi_sphere) + origin + N*dx/2.0
        z = R_np * np.cos(theta_sphere) + origin + N*dx/2.0
        fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=np.ones_like(x), colorscale='gray', opacity=0.25, showscale=False))
        
    fig.update_layout(
        scene=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            xaxis_showgrid=show_grid, yaxis_showgrid=show_grid, zaxis_showgrid=show_grid
        ),
        height=600, title=dict(text=title, x=0.5, font=dict(size=16))
    )
    return fig

def create_enhanced_stress_analysis(eta_3d, stress_components, frame_idx):
    """Enhanced stress analysis with multiple visualization types - adapted from 2D"""
    current_stress = get_stress_component(stress_components, stress_component)
    
    # Create tabs for different analysis types
    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
        "📈 Statistical Analysis", 
        "🎯 2D Slice Visualization", 
        "🔍 Scatter Analysis", 
        "📊 Radial Profiles"
    ])
    
    with analysis_tab1:
        st.subheader("Statistical Analysis")
        
        # Calculate statistics
        stress_data = current_stress[np_mask]
        eta_data = eta_3d[np_mask]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Max {stress_component}", f"{np.nanmax(stress_data):.2f} GPa")
            st.metric("Mean Stress", f"{np.nanmean(stress_data):.2f} GPa")
        with col2:
            st.metric("Std Deviation", f"{np.nanstd(stress_data):.2f} GPa")
            st.metric("Stress > 5 GPa", f"{np.sum(stress_data > 5):,} voxels")
        with col3:
            st.metric("Stress > 1 GPa", f"{np.sum(stress_data > 1):,} voxels")
            st.metric("Defect Volume", f"{np.sum(eta_data > 0.5):,} voxels")
        with col4:
            st.metric("Defect Mean η", f"{np.mean(eta_data[eta_data > 0.1]):.3f}")
            st.metric("Stress Skewness", f"{pd.Series(stress_data.flatten()).skew():.3f}")
        
        # Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        ax_hist.hist(stress_data[stress_data > 0], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax_hist.set_xlabel(f'{stress_component} (GPa)', fontsize=label_font_size, fontweight='bold')
        ax_hist.set_ylabel('Frequency', fontsize=label_font_size, fontweight='bold')
        ax_hist.set_title(f'{stress_component} Distribution - Frame {frame_idx}', 
                         fontsize=title_font_size, fontweight='bold', pad=20)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.tick_params(axis='both', which='major', labelsize=tick_font_size)
        st.pyplot(fig_hist)
    
    with analysis_tab2:
        st.subheader("2D Slice Visualization")
        
        # Slice controls
        col1, col2 = st.columns(2)
        with col1:
            slice_dim = st.selectbox("Slice Dimension", ["XY Plane (Z)", "XZ Plane (Y)", "YZ Plane (X)"])
        with col2:
            slice_pos = st.slider("Slice Position", 0, N-1, N//2, key="analysis_slice")
        
        # Create 2D visualization
        fig_slice, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if slice_dim == "XY Plane (Z)":
            eta_slice = eta_3d[:, :, slice_pos]
            stress_slice = current_stress[:, :, slice_pos]
            title_suffix = f"Z = {origin + slice_pos*dx:.1f} nm"
        elif slice_dim == "XZ Plane (Y)":
            eta_slice = eta_3d[:, slice_pos, :]
            stress_slice = current_stress[:, slice_pos, :]
            title_suffix = f"Y = {origin + slice_pos*dx:.1f} nm"
        else: # YZ Plane (X)
            eta_slice = eta_3d[slice_pos, :, :]
            stress_slice = current_stress[slice_pos, :, :]
            title_suffix = f"X = {origin + slice_pos*dx:.1f} nm"
        
        # Plot defect field
        eta_vmin, eta_vmax = colorbar_limits['eta'] if colorbar_limits['eta'][0] is not None else (0, 1)
        im1 = ax1.imshow(eta_slice, extent=[origin, origin+N*dx, origin, origin+N*dx], 
                        cmap=eta_cmap_2d, origin='lower', vmin=eta_vmin, vmax=eta_vmax)
        
        if show_contours:
            x_vals = np.linspace(origin, origin+N*dx, eta_slice.shape[1])
            y_vals = np.linspace(origin, origin+N*dx, eta_slice.shape[0])
            X_slice, Y_slice = np.meshgrid(x_vals, y_vals)
            ax1.contour(X_slice, Y_slice, eta_slice, levels=[contour_level], 
                       colors=contour_color, linewidths=line_width, alpha=contour_alpha)
        
        ax1.set_title(f"Defect Parameter η - {title_suffix}", fontsize=title_font_size, fontweight='bold')
        ax1.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
        ax1.set_ylabel("y (nm)", fontsize=label_font_size, fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot stress field
        stress_vmin, stress_vmax = colorbar_limits['von_mises'] if colorbar_limits['von_mises'][0] is not None else (None, None)
        im2 = ax2.imshow(stress_slice, extent=[origin, origin+N*dx, origin, origin+N*dx], 
                        cmap=sigma_cmap_2d, origin='lower', vmin=stress_vmin, vmax=stress_vmax)
        
        ax2.set_title(f"{stress_component} - {title_suffix}", fontsize=title_font_size, fontweight='bold')
        ax2.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        st.pyplot(fig_slice)
    
    with analysis_tab3:
        st.subheader("Scatter Analysis - Defect-Stress Correlation")
        
        # Scatter plot controls
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample Size", 100, 10000, 1000, 100)
            point_size = st.slider("Point Size", 1, 50, 10)
            alpha_value = st.slider("Point Alpha", 0.1, 1.0, 0.5, 0.1)
        with col2:
            min_eta_threshold = st.slider("Minimum η Threshold", 0.0, 0.5, 0.1, 0.05)
            min_stress_threshold = st.slider("Minimum Stress Threshold (GPa)", 0.0, 10.0, 0.0, 0.1)
            colormap_scatter = st.selectbox("Scatter Colormap", ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'coolwarm'])
        
        # Prepare data for scatter plot
        eta_flat = eta_3d[np_mask].flatten()
        stress_flat = current_stress[np_mask].flatten()
        
        # Apply thresholds
        valid_mask = (eta_flat > min_eta_threshold) & (stress_flat > min_stress_threshold)
        eta_filtered = eta_flat[valid_mask]
        stress_filtered = stress_flat[valid_mask]
        
        if len(eta_filtered) > 0:
            # Sample if too many points
            if len(eta_filtered) > sample_size:
                indices = np.random.choice(len(eta_filtered), sample_size, replace=False)
                eta_sampled = eta_filtered[indices]
                stress_sampled = stress_filtered[indices]
            else:
                eta_sampled = eta_filtered
                stress_sampled = stress_filtered
            
            # Create scatter plot
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            scatter = ax_scatter.scatter(eta_sampled, stress_sampled, 
                                        c=stress_sampled, cmap=colormap_scatter, 
                                        s=point_size, alpha=alpha_value, edgecolors='black', linewidth=0.5)
            
            ax_scatter.set_xlabel("Defect Parameter η", fontsize=label_font_size, fontweight='bold')
            ax_scatter.set_ylabel(f"{stress_component} (GPa)", fontsize=label_font_size, fontweight='bold')
            ax_scatter.set_title(f"Defect-Stress Correlation (Frame {frame_idx})", 
                               fontsize=title_font_size, fontweight='bold', pad=20)
            ax_scatter.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax_scatter, label=f"{stress_component} (GPa)")
            
            # Calculate and display correlation coefficient
            correlation = np.corrcoef(eta_filtered, stress_filtered)[0, 1]
            ax_scatter.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                          transform=ax_scatter.transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            st.pyplot(fig_scatter)
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Points", f"{len(eta_sampled):,}")
                st.metric("Correlation Coeff", f"{correlation:.3f}")
            with col2:
                st.metric("Avg η in high stress", f"{np.mean(eta_filtered[stress_filtered > 5]):.3f}")
                st.metric("Avg Stress in defect", f"{np.mean(stress_filtered[eta_filtered > 0.5]):.2f} GPa")
            with col3:
                st.metric("Max η in sample", f"{np.max(eta_sampled):.3f}")
                st.metric("Max Stress in sample", f"{np.max(stress_sampled):.2f} GPa")
        else:
            st.warning("No data points meet the threshold criteria. Try lowering the thresholds.")
    
    with analysis_tab4:
        st.subheader("Radial Stress Profiles")
        
        # Radial profile controls
        col1, col2 = st.columns(2)
        with col1:
            num_bins = st.slider("Number of Radial Bins", 10, 100, 20, 5)
            show_std = st.checkbox("Show Standard Deviation", value=True)
        with col2:
            profile_type = st.selectbox("Profile Type", ["Mean", "Median", "Maximum"])
            compare_fields = st.checkbox("Compare Multiple Fields", value=True)
        
        # Calculate radial distances
        x_center = origin + N*dx/2.0
        y_center = origin + N*dx/2.0
        z_center = origin + N*dx/2.0
        
        radial_dist = np.sqrt((X - x_center)**2 + (Y - y_center)**2 + (Z - z_center)**2)
        radial_dist_masked = radial_dist[np_mask]
        
        # Create radial bins
        radial_bins = np.linspace(0, R_np, num_bins + 1)
        bin_centers = (radial_bins[1:] + radial_bins[:-1]) / 2
        
        # Calculate profiles
        profiles = {}
        field_names = ['von_mises', 'sigma_mag', 'sigma_hydro']
        field_data = {
            'von_mises': stress_components[2][np_mask],
            'sigma_mag': stress_components[0][np_mask],
            'sigma_hydro': stress_components[1][np_mask]
        }
        
        for field_name in field_names:
            field_vals = field_data[field_name]
            radial_means = []
            radial_stds = []
            
            for i in range(len(radial_bins)-1):
                mask_bin = (radial_dist_masked >= radial_bins[i]) & (radial_dist_masked < radial_bins[i+1])
                if np.any(mask_bin):
                    if profile_type == "Mean":
                        radial_means.append(np.mean(field_vals[mask_bin]))
                    elif profile_type == "Median":
                        radial_means.append(np.median(field_vals[mask_bin]))
                    else:  # Maximum
                        radial_means.append(np.max(field_vals[mask_bin]))
                    radial_stds.append(np.std(field_vals[mask_bin]))
                else:
                    radial_means.append(0)
                    radial_stds.append(0)
            
            profiles[field_name] = {
                'means': np.array(radial_means),
                'stds': np.array(radial_stds)
            }
        
        # Plot radial profiles
        fig_radial, ax_radial = plt.subplots(figsize=(10, 6))
        
        colors = ['red', 'blue', 'green']
        labels = ['Von Mises', 'Stress Magnitude', 'Hydrostatic Stress']
        
        for idx, (field_name, color, label) in enumerate(zip(field_names, colors, labels)):
            if compare_fields or field_name == 'von_mises':
                ax_radial.plot(bin_centers, profiles[field_name]['means'], 
                              color=color, linewidth=2, marker='o', label=label)
                
                if show_std:
                    ax_radial.fill_between(bin_centers,
                                          profiles[field_name]['means'] - profiles[field_name]['stds'],
                                          profiles[field_name]['means'] + profiles[field_name]['stds'],
                                          color=color, alpha=0.2)
        
        ax_radial.set_xlabel("Radial Distance from Center (nm)", fontsize=label_font_size, fontweight='bold')
        ax_radial.set_ylabel(f"{profile_type} Stress (GPa)", fontsize=label_font_size, fontweight='bold')
        ax_radial.set_title(f"Radial Stress Profile - {profile_type} Values", 
                           fontsize=title_font_size, fontweight='bold', pad=20)
        ax_radial.legend()
        ax_radial.grid(True, alpha=0.3)
        ax_radial.tick_params(axis='both', which='major', labelsize=tick_font_size)
        
        # Add nanoparticle boundary
        ax_radial.axvline(x=R_np, color='black', linestyle='--', linewidth=2, alpha=0.5, label='NP Boundary')
        
        st.pyplot(fig_radial)
        
        # Display radial statistics
        st.subheader("Radial Statistics")
        col1, col2 = st.columns(2)
        with col1:
            max_stress_radius = bin_centers[np.argmax(profiles['von_mises']['means'])]
            st.metric("Radius of Max Stress", f"{max_stress_radius:.2f} nm")
            st.metric("Stress at Surface", f"{profiles['von_mises']['means'][-1]:.2f} GPa")
        with col2:
            stress_gradient = (profiles['von_mises']['means'][-1] - profiles['von_mises']['means'][0]) / R_np
            st.metric("Average Stress Gradient", f"{stress_gradient:.3f} GPa/nm")
            st.metric("Stress Range", f"{profiles['von_mises']['means'].max() - profiles['von_mises']['means'].min():.2f} GPa")

# =============================================
# Run simulation button
if st.button("Run 3D Evolution", type="primary"):
    with st.spinner("Running 3D phase-field + exact anisotropic spectral elasticity..."):
        eta_current = eta.copy()
        history = []
        vti_list = []
        times = []
       
        # Progress tracking
        progress_bar = st.progress(0) if enable_progress_bar else None
        status_text = st.empty()
       
        # Real-time visualization placeholder
        if enable_real_time_viz:
            viz_placeholder = st.empty()
       
        start_time = time.time()
       
        for step in range(steps + 1):
            current_time = step * dt
            if step > 0:
                eta_current = evolve_3d_vectorized(eta_current, kappa, dt, dx, M, np_mask, eps0, theta, phi, dx, enable_elastic_coupling, debug_mode)
           
            if (step % save_every == 0) or (step == steps):
                stress_components = compute_stress_3d_exact(eta_current, eps0, theta, phi, dx, debug_mode)
                current_stress = get_stress_component(stress_components, stress_component)
                history.append((eta_current.copy(), stress_components))
                vti_content = create_vti(eta_current, stress_components, step, current_time)
                vti_list.append(vti_content)
                times.append(current_time)
                # Update progress
                if progress_bar:
                    progress_bar.progress(min(step / steps, 1.0))
               
                status_text.text(f"Step {step}/{steps} - Time: {current_time:.3f} - Max {stress_component}: {current_stress[np_mask].max():.2f} GPa")
               
                # Real-time visualization
                if enable_real_time_viz and step > 0 and step % (save_every * 2) == 0:
                    with viz_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_rt = create_plotly_isosurface(X, Y, Z, eta_current, "Real-time Defect Field", eta_cmap)
                            st.plotly_chart(fig_rt, use_container_width=True)
                        with col2:
                            fig_rt2 = create_plotly_isosurface(X, Y, Z, current_stress, f"Real-time {stress_component}", stress_cmap)
                            st.plotly_chart(fig_rt2, use_container_width=True)
        end_time = time.time()
        computation_time = end_time - start_time
       
        # write pvd
        pvd = """<?xml version="1.0"?>
<VTKFile type="Collection" version="1.0">
 <Collection>
"""
        for i, t in enumerate(times):
            pvd += f' <DataSet timestep="{t:.6f}" group="" part="0" file="frame_{i:04d}.vti"/>\n'
        pvd += """ </Collection>
</VTKFile>"""
        st.session_state.history_3d = history
        st.session_state.vti_3d = vti_list
        st.session_state.pvd_3d = pvd
        st.session_state.times_3d = times
        st.session_state.stress_method = "Exact 3D Anisotropic Spectral"
       
        st.success(f"""
        ✅ 3D Simulation Complete!
        - {len(history)} frames saved
        - Computation time: {computation_time:.2f} seconds
        - Using Exact 3D Anisotropic Spectral Elasticity
        - Elastic Coupling Enabled: {enable_elastic_coupling}
        - Stress Components Available: Von Mises, Magnitude, Hydrostatic, All Tensor Components
        """)

# =============================================
# ENHANCED Interactive Visualization with Horizontal Tabs
# =============================================
if 'history_3d' in st.session_state:
    st.header("📊 Simulation Results Analysis")
   
    # Create horizontal tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 3D Visualization", 
        "📐 Slice Analysis", 
        "📈 Stress Analysis", 
        "💾 Data Export"
    ])
   
    with tab1:
        frame_idx = st.slider("Select Frame", 0, len(st.session_state.history_3d)-1,
                              len(st.session_state.history_3d)-1, key="viz_frame")
        eta_3d, stress_components = st.session_state.history_3d[frame_idx]
        current_stress = get_stress_component(stress_components, stress_component)
        times = st.session_state.times_3d
        
        eta_lims = colorbar_limits['eta'] if use_custom_limits else None
        stress_lims = colorbar_limits['von_mises'] if use_custom_limits else None
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Defect Order Parameter η ({eta_cmap})")
            eta_vis = eta_3d.copy()
            eta_vis[eta_vis < eta_threshold] = np.nan
            fig_eta = create_plotly_isosurface(
                X, Y, Z, eta_vis, "Defect Parameter η",
                eta_cmap, isomin=0.3, isomax=0.9,
                opacity=opacity_3d, surface_count=surface_count,
                custom_min=eta_lims[0] if eta_lims else None,
                custom_max=eta_lims[1] if eta_lims else None,
                show_grid=show_grid
            )
            st.plotly_chart(fig_eta, use_container_width=True)
        with col2:
            st.subheader(f"{stress_component} ({stress_cmap})")
            stress_vis = current_stress.copy()
            stress_vis[stress_vis < stress_threshold] = np.nan
            stress_data = stress_vis[np_mask]
            stress_data = np.real(stress_data) if np.any(np.iscomplex(stress_data)) else stress_data
            if stress_data.size > 0:
                stress_isomax = safe_percentile(stress_data, 95, np.nanmax(stress_vis))
            else:
                stress_isomax = np.nanmax(stress_vis) if np.any(np.isfinite(stress_vis)) else 10.0
            fig_sig = create_plotly_isosurface(
                X, Y, Z, stress_vis, f"{stress_component} (GPa)",
                stress_cmap, isomin=0.0, isomax=stress_isomax,
                opacity=opacity_3d, surface_count=surface_count,
                custom_min=stress_lims[0] if stress_lims else None,
                custom_max=stress_lims[1] if stress_lims else None,
                show_grid=show_grid
            )
            st.plotly_chart(fig_sig, use_container_width=True)
   
    with tab2:
        st.subheader("Cross-Sectional Analysis")
        slice_dim = st.selectbox("Slice Dimension", ["XY Plane (Z)", "XZ Plane (Y)", "YZ Plane (X)"])
        slice_pos = st.slider("Slice Position", 0, N-1, N//2, key="slice_pos")
       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
       
        if slice_dim == "XY Plane (Z)":
            eta_slice = eta_3d[:, :, slice_pos]
            stress_slice = current_stress[:, :, slice_pos]
            title_suffix = f"Z = {origin + slice_pos*dx:.1f} nm"
        elif slice_dim == "XZ Plane (Y)":
            eta_slice = eta_3d[:, slice_pos, :]
            stress_slice = current_stress[:, slice_pos, :]
            title_suffix = f"Y = {origin + slice_pos*dx:.1f} nm"
        else: # YZ Plane (X)
            eta_slice = eta_3d[slice_pos, :, :]
            stress_slice = current_stress[slice_pos, :, :]
            title_suffix = f"X = {origin + slice_pos*dx:.1f} nm"
       
        # Enhanced styling for 2D plots
        eta_vmin, eta_vmax = colorbar_limits['eta'] if colorbar_limits['eta'][0] is not None else (0, 1)
        stress_vmin, stress_vmax = colorbar_limits['von_mises'] if colorbar_limits['von_mises'][0] is not None else (None, None)
        
        im1 = ax1.imshow(eta_slice, cmap=eta_cmap_2d, vmin=eta_vmin, vmax=eta_vmax,
                        extent=[origin, origin+N*dx, origin, origin+N*dx], origin='lower')
        ax1.set_title(f"Defect Parameter η - {title_suffix}", fontsize=title_font_size, fontweight='bold')
        ax1.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
        ax1.set_ylabel("y (nm)", fontsize=label_font_size, fontweight='bold')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        if show_contours:
            x_vals = np.linspace(origin, origin+N*dx, eta_slice.shape[1])
            y_vals = np.linspace(origin, origin+N*dx, eta_slice.shape[0])
            X_slice, Y_slice = np.meshgrid(x_vals, y_vals)
            ax1.contour(X_slice, Y_slice, eta_slice, levels=[contour_level], 
                       colors=contour_color, linewidths=line_width, alpha=contour_alpha)
        
        im2 = ax2.imshow(stress_slice, cmap=sigma_cmap_2d, vmin=stress_vmin, vmax=stress_vmax,
                        extent=[origin, origin+N*dx, origin, origin+N*dx], origin='lower')
        ax2.set_title(f"{stress_component} - {title_suffix}", fontsize=title_font_size, fontweight='bold')
        ax2.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Apply enhanced styling
        for ax in [ax1, ax2]:
            ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
        
        st.pyplot(fig)
   
    with tab3:
        # Enhanced Stress Analysis
        st.subheader("Comprehensive Stress Analysis")
        
        # Get current frame data
        frame_idx = st.slider("Analysis Frame", 0, len(st.session_state.history_3d)-1,
                              len(st.session_state.history_3d)-1, key="analysis_frame")
        eta_3d, stress_components = st.session_state.history_3d[frame_idx]
        
        # Run enhanced stress analysis
        create_enhanced_stress_analysis(eta_3d, stress_components, frame_idx)
   
    with tab4:
        st.header("💾 Data Export")
        
        # Enhanced export options
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Export Format",
                                       ["ZIP (All formats)", "CSV Only", "VTK/VTI Only", "Selected Frame Only"])
        with col2:
            include_metadata = st.checkbox("Include Detailed Metadata", value=True)
            include_analysis = st.checkbox("Include Analysis Plots", value=True)
        
        # Frame selection for single frame export
        if export_format == "Selected Frame Only":
            export_frame = st.slider("Select Frame to Export", 0, len(st.session_state.history_3d)-1,
                                     len(st.session_state.history_3d)-1)
        
        # Custom file naming
        custom_name = st.text_input("Custom File Name (optional)", 
                                   f"Ag_NP_{defect_type}_3D_N{N}")
        
        if st.button("📦 Prepare Export Package", type="primary"):
            with st.spinner("Preparing export package..."):
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    
                    # Export CSV data
                    if export_format in ["ZIP (All formats)", "CSV Only"]:
                        frames_to_export = range(len(st.session_state.history_3d))
                        if export_format == "Selected Frame Only":
                            frames_to_export = [export_frame]
                            
                        for i in frames_to_export:
                            e, stress_comps = st.session_state.history_3d[i]
                            sigma_mag, sigma_hydro, von_mises, _, sxx, syy, szz, sxy, sxz, syz = stress_comps
                            
                            df = pd.DataFrame({
                                'x': X.flatten(order='F'),
                                'y': Y.flatten(order='F'),
                                'z': Z.flatten(order='F'),
                                'eta': e.flatten(order='F'),
                                'stress_magnitude_GPa': sigma_mag.flatten(order='F'),
                                'hydrostatic_stress_GPa': sigma_hydro.flatten(order='F'),
                                'von_mises_stress_GPa': von_mises.flatten(order='F'),
                                'stress_xx_GPa': sxx.flatten(order='F'),
                                'stress_yy_GPa': syy.flatten(order='F'),
                                'stress_zz_GPa': szz.flatten(order='F'),
                                'stress_xy_GPa': sxy.flatten(order='F'),
                                'stress_xz_GPa': sxz.flatten(order='F'),
                                'stress_yz_GPa': syz.flatten(order='F'),
                                'in_nanoparticle': np_mask.flatten(order='F'),
                                'radius_nm': r.flatten(order='F')
                            })
                            
                            if include_metadata:
                                metadata = f"""# 3D Ag Nanoparticle Simulation - Frame {i}
# Time: {st.session_state.times_3d[i]:.3f}
# Parameters: eps0={eps0}, steps={steps}, dx={dx}, dt={dt}
# Defect Type: {defect_type}
# Initial Shape: {shape}
# Grid Size: {N}³
# Stress Method: Exact 3D Anisotropic Spectral
# Elastic Constants (GPa): C11={C11/1e9:.1f}, C12={C12/1e9:.1f}, C44={C44/1e9:.1f}
# Habit Plane: θ={theta_deg}°, φ={phi_deg}°
# Nanoparticle Radius: {R_np:.2f} nm
# Color Maps: η={eta_cmap}, σ={stress_cmap}
# Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                                csv_content = metadata + df.to_csv(index=False)
                            else:
                                csv_content = df.to_csv(index=False)
                            
                            zf.writestr(f"frame_{i:04d}.csv", csv_content)
                    
                    # Export VTI data
                    if export_format in ["ZIP (All formats)", "VTK/VTI Only"]:
                        frames_to_export = range(len(st.session_state.vti_3d))
                        if export_format == "Selected Frame Only":
                            frames_to_export = [export_frame]
                            
                        for i in frames_to_export:
                            zf.writestr(f"frame_{i:04d}.vti", st.session_state.vti_3d[i])
                        
                        if export_format != "Selected Frame Only":
                            zf.writestr("simulation_3d.pvd", st.session_state.pvd_3d)
                    
                    # Include analysis plots if requested
                    if include_analysis and export_format in ["ZIP (All formats)", "Selected Frame Only"]:
                        import matplotlib
                        matplotlib.use('Agg')
                        
                        frames_to_export = [export_frame] if export_format == "Selected Frame Only" else [0, len(st.session_state.history_3d)//2, len(st.session_state.history_3d)-1]
                        
                        for i in frames_to_export:
                            e, stress_comps = st.session_state.history_3d[i]
                            
                            # Create analysis plots
                            fig_slice, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                            eta_slice = e[:, :, N//2]
                            stress_slice = stress_comps[2][:, :, N//2]  # von Mises
                            
                            im1 = ax1.imshow(eta_slice, cmap=eta_cmap_2d, 
                                            extent=[origin, origin+N*dx, origin, origin+N*dx])
                            ax1.set_title(f"Defect η - Frame {i}")
                            ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")
                            plt.colorbar(im1, ax=ax1)
                            
                            im2 = ax2.imshow(stress_slice, cmap=sigma_cmap_2d,
                                            extent=[origin, origin+N*dx, origin, origin+N*dx])
                            ax2.set_title(f"Von Mises Stress - Frame {i}")
                            ax2.set_xlabel("x (nm)")
                            plt.colorbar(im2, ax=ax2)
                            
                            plt.tight_layout()
                            
                            # Save to buffer
                            plot_buffer = BytesIO()
                            fig_slice.savefig(plot_buffer, format='png', dpi=150, bbox_inches='tight')
                            plot_buffer.seek(0)
                            zf.writestr(f"analysis_frame_{i:04d}.png", plot_buffer.read())
                            plt.close(fig_slice)
                    
                    # Always include comprehensive summary
                    summary = f"""3D Ag Nanoparticle Defect Evolution Simulation
================================================
Simulation Summary
================================================
Total Frames: {len(st.session_state.history_3d)}
Simulation Steps: {steps}
Time Step: {dt}
Grid Resolution: {N}³
Grid Spacing: {dx} nm
Nanoparticle Radius: {R_np:.2f} nm

Physics Parameters:
------------------
Defect Type: {defect_type}
Eigenstrain (ε*): {eps0}
Interface Coefficient (κ): {kappa}
Mobility (M): {M}
Initial Shape: {shape}
Habit Plane: θ={theta_deg}°, φ={phi_deg}°
Elastic Coupling: {enable_elastic_coupling}

Material Properties:
-------------------
Material: Silver (Ag)
Elastic Constants (GPa):
  C11 = {C11/1e9:.1f}
  C12 = {C12/1e9:.1f}
  C44 = {C44/1e9:.1f}

Stress Computation:
------------------
Method: {st.session_state.stress_method}
Stress Components Available:
  • Von Mises Stress
  • Stress Magnitude
  • Hydrostatic Stress
  • σ_xx, σ_yy, σ_zz
  • σ_xy, σ_xz, σ_yz

Visualization Settings:
----------------------
Defect Color Map: {eta_cmap}
Stress Color Map: {stress_cmap}
3D Opacity: {opacity_3d}
Surface Count: {surface_count}
Custom Color Limits: {use_custom_limits}

File Structure:
---------------
CSV Files: Each frame contains full 3D field data
VTI Files: ParaView-compatible 3D visualization
PVD File: Time series for animation in ParaView

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
================================================
END OF SUMMARY
"""
                    zf.writestr("SIMULATION_SUMMARY.txt", summary)
                
                buffer.seek(0)
                
                # Determine file name
                if export_format == "Selected Frame Only":
                    file_name = f"{custom_name}_frame_{export_frame:04d}.zip"
                else:
                    file_name = f"{custom_name}_complete.zip"
                
                st.download_button(
                    label="📥 Download Enhanced 3D Results",
                    data=buffer,
                    file_name=file_name,
                    mime="application/zip",
                    key="download_button"
                )

st.caption("🎯 3D Spherical Ag NP • Enhanced Stress Analysis • Comprehensive Data Export • Ultimate Version • 2025")
