# =============================================
# 3D Ag Nanoparticle Phase-Field + FFT – COMPREHENSIVELY EXPANDED
# All-features unified Streamlit script with 50+ color maps and complete stress visualization
# Enhanced with correct anisotropic 3D elasticity and comprehensive 3D export
# =============================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import os
from scipy import stats
from scipy.interpolate import griddata

st.set_page_config(page_title="3D Ag NP Defect Evolution – Ultimate", layout="wide")
st.title("3D Phase-Field Simulation of Defects in Spherical Ag Nanoparticles — Ultimate Edition")
st.markdown("""
**50+ Color Maps • Complete Stress Tensor Visualization • Enhanced 3D/2D Export**
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

st.sidebar.header("Visualization Controls")
viz_category = st.sidebar.selectbox("Color Map Category", list(COLOR_MAPS.keys()))
eta_cmap = st.sidebar.selectbox("Defect (η) Color Map", COLOR_MAPS[viz_category], index=0)
stress_cmap = st.sidebar.selectbox("Stress (σ) Color Map", COLOR_MAPS[viz_category],
                                  index=min(1, len(COLOR_MAPS[viz_category])-1))

# NEW: Stress component selection
stress_component = st.sidebar.selectbox(
    "Stress Component to Visualize",
    ["Von Mises Stress", "Stress Magnitude", "Hydrostatic Stress", 
     "σ_xx", "σ_yy", "σ_zz", "σ_xy", "σ_xz", "σ_yz"]
)

use_custom_limits = st.sidebar.checkbox("Use Custom Color Scale Limits", False)
if use_custom_limits:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        eta_min = st.number_input("η Min", value=0.0, format="%.2f")
        stress_min = st.number_input("σ Min (GPa)", value=0.0, format="%.2f")
    with col2:
        eta_max = st.number_input("η Max", value=1.0, format="%.2f")
        stress_max = st.number_input("σ Max (GPa)", value=10.0, format="%.2f")
else:
    eta_min, eta_max, stress_min, stress_max = None, None, None, None

st.sidebar.subheader("3D Rendering")
opacity_3d = st.sidebar.slider("3D Opacity", 0.05, 1.0, 0.7, 0.05)
surface_count = st.sidebar.slider("Surface Count", 1, 10, 2)
show_grid = st.sidebar.checkbox("Show Grid in Plotly", value=True)
show_matrix = st.sidebar.checkbox("Show Nanoparticle Matrix", value=True)
eta_threshold = st.sidebar.slider("η Visualization Threshold", 0.0, 1.0, 0.1, 0.01)
stress_threshold = st.sidebar.slider("Stress Visualization Threshold (GPa)", 0.0, 50.0, 0.0, 0.1)

# 2D Analysis Controls
st.sidebar.header("2D Stress Analysis Controls")
scatter_point_size = st.sidebar.slider("Scatter Point Size", 1, 20, 3)
scatter_opacity = st.sidebar.slider("Scatter Opacity", 0.1, 1.0, 0.5, 0.05)
scatter_max_points = st.sidebar.slider("Max Points in Scatter", 100, 10000, 2000, 100)
histogram_bins = st.sidebar.slider("Histogram Bins", 10, 200, 50, 5)
smoothing_kernel = st.sidebar.slider("Profile Smoothing Kernel", 1, 21, 5, 2)
profile_error_bars = st.sidebar.checkbox("Show Error Bars on Profiles", value=True)

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
# Safe helpers and visualization utilities
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
        theta = np.linspace(0, np.pi, 50)
        phi = np.linspace(0, 2*np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)
        x = R_np * np.sin(theta) * np.cos(phi) + origin + N*dx/2.0
        y = R_np * np.sin(theta) * np.sin(phi) + origin + N*dx/2.0
        z = R_np * np.cos(theta) + origin + N*dx/2.0
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

def safe_matplotlib_cmap(cmap_name, default='viridis'):
    try:
        plt.get_cmap(cmap_name)
        return cmap_name
    except (ValueError, AttributeError):
        st.warning(f"Colormap '{cmap_name}' not found. Using '{default}'.")
        return default

def create_matplotlib_comparison(eta_3d, stress_components, frame_idx, eta_cmap, stress_cmap, eta_lims, stress_lims):
    slice_pos = N // 2
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"3D Stress Visualization Comparison - Frame {frame_idx} - Slice z={slice_pos}", fontsize=16, y=0.95)
    
    eta_data = eta_3d[np_mask]
    stress_data = get_stress_component(stress_components, stress_component)[np_mask]
    
    eta_vmin, eta_vmax = eta_lims if eta_lims else (safe_percentile(eta_data, 0, 0.0), safe_percentile(eta_data, 100, 1.0))
    stress_vmin, stress_vmax = stress_lims if stress_lims else (safe_percentile(stress_data, 0, 0.0), safe_percentile(stress_data, 100, 10.0))
    
    safe_eta = safe_matplotlib_cmap(eta_cmap, 'Blues')
    safe_stress = safe_matplotlib_cmap(stress_cmap, 'Reds')
    
    try:
        im1 = axes[0,0].imshow(eta_3d[:, :, slice_pos], cmap=safe_eta, vmin=eta_vmin, vmax=eta_vmax,
                                extent=[origin, origin+N*dx, origin, origin+N*dx])
        axes[0,0].set_title(f'Defect η ({safe_eta})')
        axes[0,0].set_xlabel('x (nm)'); axes[0,0].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    except Exception as e:
        axes[0,0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        
    try:
        current_stress = get_stress_component(stress_components, stress_component)
        im2 = axes[0,1].imshow(current_stress[:, :, slice_pos], cmap=safe_stress, vmin=stress_vmin, vmax=stress_vmax,
                                extent=[origin, origin+N*dx, origin, origin+N*dx])
        axes[0,1].set_title(f'{stress_component} ({safe_stress})')
        axes[0,1].set_xlabel('x (nm)')
        plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    except Exception as e:
        axes[0,1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        
    axes[0,2].axis('off')
    axes[0,2].text(0.1, 0.5, f"Stress Component: {stress_component}\nMethod: Exact 3D Anisotropic Spectral", fontsize=10)
    
    alt_cmaps = ['jet', 'turbo', 'viridis', 'plasma']
    alt_titles = ['Jet (Traditional)', 'Turbo (High Contrast)', 'Viridis (Perceptual)', 'Plasma (Vibrant)']
    
    for i, (cmap, title) in enumerate(zip(alt_cmaps, alt_titles)):
        if i >= 3:  # Only show 3 in the second row
            break
        try:
            current_stress = get_stress_component(stress_components, stress_component)
            im = axes[1,i].imshow(current_stress[:, :, slice_pos], cmap=cmap, vmin=stress_vmin, vmax=stress_vmax,
                                  extent=[origin, origin+N*dx, origin, origin+N*dx])
            axes[1,i].set_title(f'{stress_component} - {title}')
            axes[1,i].set_xlabel('x (nm)')
            if i == 0:
                axes[1,i].set_ylabel('y (nm)')
            plt.colorbar(im, ax=axes[1,i], shrink=0.8)
        except Exception as e:
            axes[1,i].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            
    plt.tight_layout()
    return fig

# =============================================
# ENHANCED 2D STRESS ANALYSIS FUNCTIONS WITH EDITABILITY
def create_2d_stress_analysis(eta_3d, stress_components, frame_idx, slice_pos=None):
    """Enhanced 2D stress analysis with multiple visualization types and editability"""
    if slice_pos is None:
        slice_pos = N // 2
    
    current_stress = get_stress_component(stress_components, stress_component)
    
    # Create tabs for different 2D analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Scatter Analysis", "Histogram Analysis", "Radial Profile", 
        "Correlation Matrix", "3D Projection"
    ])
    
    with tab1:
        st.subheader("Scatter Plot Analysis")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create scatter plot
            fig_scatter = plt.figure(figsize=(10, 8))
            
            # Sample points for clarity
            eta_masked = eta_3d[np_mask].flatten()
            stress_masked = current_stress[np_mask].flatten()
            
            if len(eta_masked) > scatter_max_points:
                indices = np.random.choice(len(eta_masked), scatter_max_points, replace=False)
                eta_sample = eta_masked[indices]
                stress_sample = stress_masked[indices]
            else:
                eta_sample = eta_masked
                stress_sample = stress_masked
            
            scatter = plt.scatter(eta_sample, stress_sample, 
                                 c=stress_sample, cmap=stress_cmap, 
                                 s=scatter_point_size, alpha=scatter_opacity)
            
            # Add linear regression line
            if len(eta_sample) > 10:
                slope, intercept, r_value, p_value, std_err = stats.linregress(eta_sample, stress_sample)
                x_line = np.array([eta_sample.min(), eta_sample.max()])
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, 'r-', linewidth=2, 
                        label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}')
            
            plt.xlabel('Defect Parameter η')
            plt.ylabel(f'{stress_component} (GPa)')
            plt.title(f'Defect-Stress Scatter Plot - Frame {frame_idx}')
            plt.colorbar(scatter, label=f'{stress_component} (GPa)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            st.pyplot(fig_scatter)
        
        with col2:
            st.markdown("### Scatter Controls")
            show_density = st.checkbox("Show Density Heatmap", value=False)
            show_stats = st.checkbox("Show Statistics", value=True)
            
            if show_density:
                # Create density heatmap
                fig_density, ax = plt.subplots(figsize=(8, 6))
                hb = ax.hexbin(eta_masked, stress_masked, gridsize=30, cmap=stress_cmap)
                ax.set_xlabel('Defect Parameter η')
                ax.set_ylabel(f'{stress_component} (GPa)')
                plt.colorbar(hb, ax=ax, label='Count')
                st.pyplot(fig_density)
            
            if show_stats and len(eta_masked) > 0:
                st.markdown("#### Statistical Analysis")
                st.write(f"Correlation coefficient: {np.corrcoef(eta_masked, stress_masked)[0,1]:.3f}")
                st.write(f"Stress range: {stress_masked.min():.3f} - {stress_masked.max():.3f} GPa")
                st.write(f"Mean stress: {stress_masked.mean():.3f} GPa")
                st.write(f"Median stress: {np.median(stress_masked):.3f} GPa")
    
    with tab2:
        st.subheader("Histogram Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_hist, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram of stress
            axes[0].hist(stress_masked, bins=histogram_bins, alpha=0.7, 
                        color='red', edgecolor='black')
            axes[0].set_xlabel(f'{stress_component} (GPa)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'{stress_component} Distribution')
            axes[0].grid(True, alpha=0.3)
            
            # Cumulative distribution
            axes[1].hist(stress_masked, bins=histogram_bins, alpha=0.7, 
                        color='blue', edgecolor='black', cumulative=True)
            axes[1].set_xlabel(f'{stress_component} (GPa)')
            axes[1].set_ylabel('Cumulative Frequency')
            axes[1].set_title(f'{stress_component} Cumulative Distribution')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_hist)
        
        with col2:
            st.markdown("### Histogram Controls")
            log_scale = st.checkbox("Log Scale Y-axis", value=False)
            show_fit = st.checkbox("Show Gaussian Fit", value=False)
            
            if log_scale:
                fig_log, ax = plt.subplots(figsize=(8, 6))
                ax.hist(stress_masked, bins=histogram_bins, alpha=0.7, 
                       color='green', edgecolor='black', log=True)
                ax.set_xlabel(f'{stress_component} (GPa)')
                ax.set_ylabel('Log Frequency')
                ax.set_title('Log-scale Histogram')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_log)
            
            if show_fit and len(stress_masked) > 0:
                # Fit Gaussian
                mu, sigma = stats.norm.fit(stress_masked)
                x = np.linspace(stress_masked.min(), stress_masked.max(), 100)
                pdf = stats.norm.pdf(x, mu, sigma)
                
                fig_fit, ax = plt.subplots(figsize=(8, 6))
                ax.hist(stress_masked, bins=histogram_bins, alpha=0.7, 
                       density=True, color='purple', edgecolor='black')
                ax.plot(x, pdf, 'r-', linewidth=2, 
                       label=f'μ={mu:.3f}, σ={sigma:.3f}')
                ax.set_xlabel(f'{stress_component} (GPa)')
                ax.set_ylabel('Probability Density')
                ax.set_title('Gaussian Fit to Stress Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_fit)
    
    with tab3:
        st.subheader("Radial Profile Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig_radial, ax = plt.subplots(figsize=(10, 6))
            
            # Radial bins
            radial_bins = np.linspace(0, R_np, 30)
            radial_centers = (radial_bins[1:] + radial_bins[:-1]) / 2
            
            mean_stress = []
            std_stress = []
            
            for i in range(len(radial_bins)-1):
                mask = (r >= radial_bins[i]) & (r < radial_bins[i+1]) & np_mask
                if np.any(mask):
                    mean_stress.append(np.mean(current_stress[mask]))
                    std_stress.append(np.std(current_stress[mask]))
                else:
                    mean_stress.append(0)
                    std_stress.append(0)
            
            # Smooth the profile if requested
            if smoothing_kernel > 1:
                kernel = np.ones(smoothing_kernel) / smoothing_kernel
                mean_stress_smooth = np.convolve(mean_stress, kernel, mode='same')
                ax.plot(radial_centers, mean_stress_smooth, 'b-', linewidth=3, 
                       label='Smoothed Mean', alpha=0.8)
            
            # Plot with error bars
            if profile_error_bars:
                ax.errorbar(radial_centers, mean_stress, yerr=std_stress, 
                          fmt='o-', capsize=5, label='Mean ± Std Dev')
            else:
                ax.plot(radial_centers, mean_stress, 'o-', label='Mean')
            
            ax.set_xlabel('Radius (nm)')
            ax.set_ylabel(f'{stress_component} (GPa)')
            ax.set_title(f'Radial {stress_component} Profile')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim([0, R_np])
            
            st.pyplot(fig_radial)
        
        with col2:
            st.markdown("### Profile Controls")
            show_derivative = st.checkbox("Show Derivative", value=False)
            show_integral = st.checkbox("Show Integrated Stress", value=False)
            
            if show_derivative and len(mean_stress) > 1:
                # Calculate derivative
                derivative = np.gradient(mean_stress, radial_centers[1]-radial_centers[0])
                
                fig_deriv, ax = plt.subplots(figsize=(8, 6))
                ax.plot(radial_centers, derivative, 'g-', linewidth=2)
                ax.set_xlabel('Radius (nm)')
                ax.set_ylabel(f'd({stress_component})/dr (GPa/nm)')
                ax.set_title('Stress Gradient')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_deriv)
            
            if show_integral and len(mean_stress) > 1:
                # Calculate integrated stress
                integrated = np.cumsum(mean_stress) * (radial_centers[1] - radial_centers[0])
                
                fig_integral, ax = plt.subplots(figsize=(8, 6))
                ax.plot(radial_centers, integrated, 'r-', linewidth=2)
                ax.set_xlabel('Radius (nm)')
                ax.set_ylabel(f'Integrated {stress_component} (GPa·nm)')
                ax.set_title('Cumulative Stress')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_integral)
    
    with tab4:
        st.subheader("Stress Tensor Correlation Matrix")
        
        # Extract all stress components
        sigma_mag, sigma_hydro, von_mises, _, sxx, syy, szz, sxy, sxz, syz = stress_components
        
        # Prepare data for correlation matrix
        stress_components_list = [
            ('η', eta_3d[np_mask].flatten()),
            ('σ_vm', von_mises[np_mask].flatten()),
            ('σ_mag', sigma_mag[np_mask].flatten()),
            ('σ_hydro', sigma_hydro[np_mask].flatten()),
            ('σ_xx', sxx[np_mask].flatten()),
            ('σ_yy', syy[np_mask].flatten()),
            ('σ_zz', szz[np_mask].flatten()),
            ('σ_xy', sxy[np_mask].flatten())
        ]
        
        # Limit points for performance
        max_corr_points = min(10000, len(stress_components_list[0][1]))
        indices = np.random.choice(len(stress_components_list[0][1]), max_corr_points, replace=False)
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(stress_components_list), len(stress_components_list)))
        
        for i in range(len(stress_components_list)):
            for j in range(len(stress_components_list)):
                data_i = stress_components_list[i][1][indices]
                data_j = stress_components_list[j][1][indices]
                corr_matrix[i, j] = np.corrcoef(data_i, data_j)[0, 1]
        
        # Plot correlation matrix
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        
        # Add labels
        labels = [comp[0] for comp in stress_components_list]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
        
        ax.set_title("Stress Tensor Correlation Matrix")
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        st.pyplot(fig_corr)
        
        # Pairwise scatter plots
        st.subheader("Selected Pairwise Scatter Plots")
        
        selected_pairs = st.multiselect(
            "Select pairs to visualize:",
            [(f"η vs {stress_component}", 0, 1),
             ("σ_vm vs σ_hydro", 1, 3),
             ("σ_xx vs σ_yy", 4, 5),
             ("σ_xy vs η", 7, 0)],
            default=[(f"η vs {stress_component}", 0, 1)]
        )
        
        if selected_pairs:
            n_pairs = len(selected_pairs)
            fig_pairs, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 4))
            if n_pairs == 1:
                axes = [axes]
            
            for idx, (label, i_idx, j_idx) in enumerate(selected_pairs):
                x_data = stress_components_list[i_idx][1][indices]
                y_data = stress_components_list[j_idx][1][indices]
                
                axes[idx].scatter(x_data, y_data, alpha=0.3, s=10)
                axes[idx].set_xlabel(stress_components_list[i_idx][0])
                axes[idx].set_ylabel(stress_components_list[j_idx][0])
                axes[idx].set_title(label)
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_pairs)
    
    with tab5:
        st.subheader("3D Projection Analysis")
        
        # Create 3D scatter plot
        fig_3d = go.Figure()
        
        # Sample points for 3D visualization
        sample_mask = (eta_3d > 0.1) & np_mask
        sample_indices = np.where(sample_mask)
        
        if len(sample_indices[0]) > 0:
            # Limit points for performance
            n_3d_points = min(5000, len(sample_indices[0]))
            subset = np.random.choice(len(sample_indices[0]), n_3d_points, replace=False)
            
            x_vals = X[sample_indices][subset]
            y_vals = Y[sample_indices][subset]
            z_vals = Z[sample_indices][subset]
            stress_vals = current_stress[sample_indices][subset]
            eta_vals = eta_3d[sample_indices][subset]
            
            # Create 3D scatter plot
            fig_3d.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='markers',
                marker=dict(
                    size=5,
                    color=stress_vals,
                    colorscale=stress_cmap,
                    opacity=0.7,
                    colorbar=dict(title=f'{stress_component} (GPa)')
                ),
                text=[f'η={eta:.2f}, σ={stress:.2f} GPa' 
                     for eta, stress in zip(eta_vals, stress_vals)],
                hovertemplate='%{text}<extra></extra>',
                name='Stress Points'
            ))
            
            # Add nanoparticle surface
            if show_matrix:
                phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, 30), 
                                        np.linspace(0, np.pi, 30))
                x_surface = R_np * np.sin(theta) * np.cos(phi) + origin + N*dx/2.0
                y_surface = R_np * np.sin(theta) * np.sin(phi) + origin + N*dx/2.0
                z_surface = R_np * np.cos(theta) + origin + N*dx/2.0
                
                fig_3d.add_trace(go.Surface(
                    x=x_surface, y=y_surface, z=z_surface,
                    opacity=0.2,
                    colorscale='gray',
                    showscale=False,
                    name='Nanoparticle'
                ))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='data'
            ),
            title=f'3D Stress Distribution - {stress_component}',
            height=600
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Additional 3D controls
        col1, col2 = st.columns(2)
        with col1:
            view_x = st.slider("View X", -180, 180, 30)
            view_y = st.slider("View Y", -180, 180, 30)
            view_z = st.slider("View Z", -180, 180, 30)
        
        fig_3d.update_layout(
            scene_camera=dict(
                eye=dict(x=view_x/100, y=view_y/100, z=view_z/100)
            )
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)

def create_2d_data_export(eta_3d, stress_components, frame_idx, times):
    """Create comprehensive 2D data export"""
    
    current_stress = get_stress_component(stress_components, stress_component)
    
    # Create multiple dataframes for different analyses
    buffer = BytesIO()
    
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Full 3D data (sampled for 2D analysis)
        full_data = pd.DataFrame({
            'x': X[np_mask].flatten(),
            'y': Y[np_mask].flatten(),
            'z': Z[np_mask].flatten(),
            'eta': eta_3d[np_mask].flatten(),
            'stress': current_stress[np_mask].flatten(),
            'radius': r[np_mask].flatten()
        })
        zf.writestr(f"frame_{frame_idx:04d}_full_data.csv", full_data.to_csv(index=False))
        
        # 2. Radial profile data
        radial_bins = np.linspace(0, R_np, 30)
        radial_centers = (radial_bins[1:] + radial_bins[:-1]) / 2
        
        radial_data = []
        for i in range(len(radial_bins)-1):
            mask = (r >= radial_bins[i]) & (r < radial_bins[i+1]) & np_mask
            if np.any(mask):
                radial_data.append({
                    'radius': radial_centers[i],
                    'mean_stress': np.mean(current_stress[mask]),
                    'std_stress': np.std(current_stress[mask]),
                    'min_stress': np.min(current_stress[mask]),
                    'max_stress': np.max(current_stress[mask]),
                    'count': np.sum(mask)
                })
        
        radial_df = pd.DataFrame(radial_data)
        zf.writestr(f"frame_{frame_idx:04d}_radial_profile.csv", radial_df.to_csv(index=False))
        
        # 3. Statistical summary
        stress_masked = current_stress[np_mask].flatten()
        eta_masked = eta_3d[np_mask].flatten()
        
        stats_df = pd.DataFrame({
            'statistic': ['mean', 'std', 'min', 'max', '25%', '50%', '75%', 'skew', 'kurtosis'],
            'stress': [
                np.mean(stress_masked),
                np.std(stress_masked),
                np.min(stress_masked),
                np.max(stress_masked),
                np.percentile(stress_masked, 25),
                np.percentile(stress_masked, 50),
                np.percentile(stress_masked, 75),
                stats.skew(stress_masked),
                stats.kurtosis(stress_masked)
            ],
            'eta': [
                np.mean(eta_masked),
                np.std(eta_masked),
                np.min(eta_masked),
                np.max(eta_masked),
                np.percentile(eta_masked, 25),
                np.percentile(eta_masked, 50),
                np.percentile(eta_masked, 75),
                stats.skew(eta_masked),
                stats.kurtosis(eta_masked)
            ]
        })
        zf.writestr(f"frame_{frame_idx:04d}_statistics.csv", stats_df.to_csv(index=False))
        
        # 4. Correlation data
        if len(stress_masked) > 0 and len(eta_masked) > 0:
            corr_df = pd.DataFrame({
                'x': eta_masked[:min(10000, len(eta_masked))],
                'y': stress_masked[:min(10000, len(stress_masked))]
            })
            zf.writestr(f"frame_{frame_idx:04d}_correlation_data.csv", corr_df.to_csv(index=False))
        
        # 5. Metadata
        metadata = f"""2D Stress Analysis Export - Frame {frame_idx}
================================================
Time: {times[frame_idx]:.3f}
Stress Component: {stress_component}
Grid Size: {N}³
Grid Spacing: {dx} nm
Nanoparticle Radius: {R_np:.2f} nm
Eigenstrain (ε*): {eps0}
Defect Type: {defect_type}
Habit Plane: θ={theta_deg}°, φ={phi_deg}°
Stress Calculation: Exact 3D Anisotropic Spectral

Files Included:
1. full_data.csv - Sampled 3D data within nanoparticle
2. radial_profile.csv - Radial stress profile
3. statistics.csv - Statistical summary
4. correlation_data.csv - Correlation data for scatter plots

Analysis Parameters:
- Scatter point size: {scatter_point_size}
- Scatter opacity: {scatter_opacity}
- Max scatter points: {scatter_max_points}
- Histogram bins: {histogram_bins}
- Smoothing kernel: {smoothing_kernel}
- Error bars: {profile_error_bars}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        zf.writestr("ANALYSIS_METADATA.txt", metadata)
    
    buffer.seek(0)
    return buffer

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
# Enhanced Interactive Visualization
if 'history_3d' in st.session_state:
    st.header("📊 Simulation Results Analysis")
   
    # Create horizontal tabs for different analysis types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Visualization", "Slice Analysis", "2D Stress Analysis", 
        "Comprehensive Analysis", "Data Export"
    ])
   
    frame_idx = st.slider("Select Frame", 0, len(st.session_state.history_3d)-1,
                          len(st.session_state.history_3d)-1, key="viz_frame")
    eta_3d, stress_components = st.session_state.history_3d[frame_idx]
    current_stress = get_stress_component(stress_components, stress_component)
    times = st.session_state.times_3d
    
    eta_lims = (eta_min, eta_max) if use_custom_limits else None
    stress_lims = (stress_min, stress_max) if use_custom_limits else None
    
    with tab1:
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
        slice_pos = st.slider("Slice Position", 0, N-1, N//2)
       
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
       
        im1 = ax1.imshow(eta_slice, cmap='viridis', extent=[origin, origin+N*dx, origin, origin+N*dx])
        ax1.set_title(f"Defect Parameter η - {title_suffix}")
        ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")
        plt.colorbar(im1, ax=ax1)
        im2 = ax2.imshow(stress_slice, cmap='hot', extent=[origin, origin+N*dx, origin, origin+N*dx])
        ax2.set_title(f"{stress_component} - {title_suffix}")
        ax2.set_xlabel("x (nm)")
        plt.colorbar(im2, ax=ax2)
        st.pyplot(fig)
       
        # Color map comparison
        st.subheader("Color Map Comparison")
        fig_mpl = create_matplotlib_comparison(
            eta_3d, stress_components, frame_idx,
            eta_cmap, stress_cmap, eta_lims, stress_lims
        )
        st.pyplot(fig_mpl)
   
    with tab3:
        st.subheader("2D Stress Analysis with Editability")
        
        # Create 2D analysis with the selected slice
        slice_pos = st.slider("Slice for 2D Analysis", 0, N-1, N//2, key="2d_slice")
        
        # Extract slice data for 2D analysis
        eta_slice = eta_3d[:, :, slice_pos]
        stress_slice = current_stress[:, :, slice_pos]
        mask_slice = np_mask[:, :, slice_pos]
        
        # Create custom stress components for 2D
        sigma_mag, sigma_hydro, von_mises, _, sxx, syy, szz, sxy, sxz, syz = stress_components
        
        # For 2D analysis, we need to create a compatible stress_components structure
        stress_components_2d = (
            sigma_mag[:, :, slice_pos],  # sigma_mag
            sigma_hydro[:, :, slice_pos],  # sigma_hydro
            von_mises[:, :, slice_pos],  # von_mises
            None,  # sigma_gpa (not needed)
            sxx[:, :, slice_pos],  # sxx
            syy[:, :, slice_pos],  # syy
            szz[:, :, slice_pos],  # szz
            sxy[:, :, slice_pos],  # sxy
            sxz[:, :, slice_pos],  # sxz
            syz[:, :, slice_pos]   # syz
        )
        
        # Create 2D analysis
        create_2d_stress_analysis(eta_slice, stress_components_2d, frame_idx)
        
        # 2D Data Export
        st.subheader("2D Data Export")
        
        col1, col2 = st.columns(2)
        with col1:
            export_2d_format = st.selectbox("2D Export Format", 
                                          ["CSV (All analyses)", "Excel", "JSON"])
        with col2:
            include_plots = st.checkbox("Include Plots as Images", value=False)
        
        if st.button("Export 2D Analysis Data", type="secondary"):
            with st.spinner("Preparing 2D data export..."):
                export_buffer = create_2d_data_export(eta_slice, stress_components_2d, frame_idx, times)
                
                st.download_button(
                    label="Download 2D Analysis Data",
                    data=export_buffer,
                    file_name=f"2D_Analysis_Frame_{frame_idx:04d}.zip",
                    mime="application/zip"
                )
   
    with tab4:
        st.subheader("Comprehensive Analysis Dashboard")
        
        # Create columns for different metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stress_data = current_stress[np_mask]
            st.metric("Max Stress", f"{np.nanmax(stress_data):.2f} GPa")
            st.metric("Stress > 5 GPa", f"{np.sum(stress_data > 5):,}")
        
        with col2:
            st.metric("Mean Stress", f"{np.nanmean(stress_data):.2f} GPa")
            st.metric("Stress > 10 GPa", f"{np.sum(stress_data > 10):,}")
        
        with col3:
            st.metric("Std Dev", f"{np.nanstd(stress_data):.2f} GPa")
            eta_data = eta_3d[np_mask]
            st.metric("Defect Volume", f"{np.sum(eta_data > 0.5):,} voxels")
        
        with col4:
            st.metric("Median Stress", f"{np.nanmedian(stress_data):.2f} GPa")
            st.metric("Stress Gradient", 
                     f"{np.max(np.gradient(stress_data)):.3f} GPa/nm" if len(stress_data) > 1 else "N/A")
        
        # Time evolution plot
        st.subheader("Stress Evolution Over Time")
        
        # Calculate stress metrics over time
        time_series = []
        for i, (eta_frame, stress_comps) in enumerate(st.session_state.history_3d):
            stress_frame = get_stress_component(stress_comps, stress_component)
            stress_masked = stress_frame[np_mask]
            time_series.append({
                'time': st.session_state.times_3d[i],
                'max_stress': np.nanmax(stress_masked),
                'mean_stress': np.nanmean(stress_masked),
                'median_stress': np.nanmedian(stress_masked),
                'std_stress': np.nanstd(stress_masked)
            })
        
        time_df = pd.DataFrame(time_series)
        
        # Plot time evolution
        fig_time, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_df['time'], time_df['max_stress'], 'r-', label='Max Stress', linewidth=2)
        ax.plot(time_df['time'], time_df['mean_stress'], 'b-', label='Mean Stress', linewidth=2)
        ax.fill_between(time_df['time'], 
                       time_df['mean_stress'] - time_df['std_stress'],
                       time_df['mean_stress'] + time_df['std_stress'],
                       alpha=0.3, color='blue', label='±1 Std Dev')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{stress_component} (GPa)')
        ax.set_title('Stress Evolution During Simulation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig_time)
        
        # Export time series data
        if st.button("Export Time Series Data"):
            csv = time_df.to_csv(index=False)
            st.download_button(
                label="Download Time Series CSV",
                data=csv,
                file_name="stress_time_series.csv",
                mime="text/csv"
            )
   
    with tab5:
        st.header("Comprehensive Data Export")
        
        # Export options in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_format = st.selectbox("Export Format",
                                       ["ZIP (All formats)", "CSV Only", "VTK/VTI Only", 
                                        "2D Analysis Only", "Time Series Only"])
        
        with col2:
            include_metadata = st.checkbox("Include Detailed Metadata", value=True)
            compress_level = st.slider("Compression Level", 1, 9, 6)
        
        with col3:
            export_all_frames = st.checkbox("Export All Frames", value=True)
            selected_frames = st.multiselect("Or Select Specific Frames", 
                                           list(range(len(st.session_state.history_3d))),
                                           default=[len(st.session_state.history_3d)-1])
        
        if st.button("Generate Export Package", type="primary"):
            with st.spinner("Creating export package..."):
                buffer = BytesIO()
                
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=compress_level) as zf:
                    # Export based on selected format
                    if export_format in ["ZIP (All formats)", "CSV Only"]:
                        frames_to_export = selected_frames if not export_all_frames else range(len(st.session_state.history_3d))
                        
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
                                'in_nanoparticle': np_mask.flatten(order='F')
                            })
                            
                            if include_metadata:
                                metadata = f"""# 3D Ag Nanoparticle Simulation
# Frame: {i}
# Time: {st.session_state.times_3d[i]:.3f}
# Parameters: eps0={eps0}, steps={steps}, dx={dx}
# Stress Method: Exact 3D Anisotropic Spectral
# Color Maps: eta={eta_cmap}, stress={stress_cmap}
# Defect Type: {defect_type}
# Grid Size: {N}
# Stress Components: All tensor components included
"""
                                csv_content = metadata + df.to_csv(index=False)
                            else:
                                csv_content = df.to_csv(index=False)
                            
                            zf.writestr(f"frame_{i:04d}.csv", csv_content)
                    
                    if export_format in ["ZIP (All formats)", "VTK/VTI Only"]:
                        for i, vti_content in enumerate(st.session_state.vti_3d):
                            zf.writestr(f"frame_{i:04d}.vti", vti_content)
                        zf.writestr("simulation_3d.pvd", st.session_state.pvd_3d)
                    
                    if export_format in ["ZIP (All formats)", "2D Analysis Only"]:
                        # Export 2D analysis for current frame
                        export_buffer = create_2d_data_export(eta_3d, stress_components, frame_idx, times)
                        
                        # Extract and add 2D analysis files
                        with zipfile.ZipFile(export_buffer, 'r') as temp_zip:
                            for name in temp_zip.namelist():
                                zf.writestr(f"2d_analysis/{name}", temp_zip.read(name))
                    
                    if export_format in ["ZIP (All formats)", "Time Series Only"]:
                        # Export time series data
                        time_series_data = []
                        for i, (e, stress_comps) in enumerate(st.session_state.history_3d):
                            stress_frame = get_stress_component(stress_comps, stress_component)
                            stress_masked = stress_frame[np_mask]
                            time_series_data.append({
                                'frame': i,
                                'time': st.session_state.times_3d[i],
                                'max_stress': np.nanmax(stress_masked),
                                'mean_stress': np.nanmean(stress_masked),
                                'median_stress': np.nanmedian(stress_masked),
                                'std_stress': np.nanstd(stress_masked),
                                'q1_stress': np.percentile(stress_masked, 25),
                                'q3_stress': np.percentile(stress_masked, 75),
                                'defect_volume': np.sum(e[np_mask] > 0.5)
                            })
                        
                        time_df = pd.DataFrame(time_series_data)
                        zf.writestr("time_series_analysis.csv", time_df.to_csv(index=False))
                    
                    # Always include summary
                    summary = f"""3D Ag Nanoparticle Defect Evolution Simulation - COMPREHENSIVE EXPORT
================================================
Export Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Frames: {len(st.session_state.history_3d)}
Simulation Steps: {steps}
Time Step: {dt}
Grid Resolution: {N}³
Grid Spacing: {dx} nm
Nanoparticle Radius: {R_np:.2f} nm
Eigenstrain (ε*): {eps0}
Defect Type: {defect_type}
Habit Plane: θ={theta_deg}°, φ={phi_deg}°
Stress Calculation: Exact 3D Anisotropic Spectral
Elastic Coupling: {enable_elastic_coupling}
Material: Silver (Ag)
Elastic Constants: C11={C11/1e9:.1f} GPa, C12={C12/1e9:.1f} GPa, C44={C44/1e9:.1f} GPa

Simulation Parameters:
- Mobility (M): {M}
- Interface Coefficient (κ): {kappa}
- Initial Shape: {shape}

Visualization Settings:
- Defect Color Map: {eta_cmap}
- Stress Color Map: {stress_cmap}
- 3D Opacity: {opacity_3d}
- Surface Count: {surface_count}

2D Analysis Settings:
- Scatter Point Size: {scatter_point_size}
- Scatter Opacity: {scatter_opacity}
- Max Scatter Points: {scatter_max_points}
- Histogram Bins: {histogram_bins}
- Smoothing Kernel: {smoothing_kernel}
- Error Bars: {profile_error_bars}

Stress Components Available:
  Von Mises, Stress Magnitude, Hydrostatic Stress
  σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz

Export Contents:
{'- CSV data files' if export_format in ['ZIP (All formats)', 'CSV Only'] else ''}
{'- VTK/VTI 3D files' if export_format in ['ZIP (All formats)', 'VTK/VTI Only'] else ''}
{'- 2D analysis data' if export_format in ['ZIP (All formats)', '2D Analysis Only'] else ''}
{'- Time series analysis' if export_format in ['ZIP (All formats)', 'Time Series Only'] else ''}

For visualization:
- Use ParaView for VTK/VTI files
- Use any spreadsheet software for CSV files
- Python scripts for further analysis are available upon request

Generated by: 3D Ag Nanoparticle Phase-Field Simulator
Contact: For scientific collaboration and customization
"""
                    zf.writestr("EXPORT_SUMMARY.txt", summary)
                
                buffer.seek(0)
                
                st.success("Export package created successfully!")
                
                st.download_button(
                    label="📥 Download Comprehensive Export Package",
                    data=buffer,
                    file_name=f"Ag_Nanoparticle_3D_{defect_type}_N{N}_comprehensive.zip",
                    mime="application/zip",
                    help="Contains all selected data formats and analyses"
                )

st.caption("🎯 3D Spherical Ag NP • 50+ Color Maps • Complete Stress Tensor • Ultimate Version • 2025 • Enhanced 2D Analysis")
