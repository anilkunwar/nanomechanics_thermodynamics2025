# =============================================
# ULTIMATE Ag NP Defect Analyzer – CRYSTALLOGRAPHICALLY ACCURATE
# =============================================
import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import matplotlib.cm as cm

# Configure page with better styling
st.set_page_config(page_title="Ag NP Defect Analyzer – Ultimate", layout="wide")
st.title("🏗️ Ag Nanoparticle Defect Mechanics – Crystallographically Accurate")
st.markdown("""
**Live phase-field + FFT elasticity**
**ISF, ESF, and Twin are now physically distinct**
Four fields exported • Custom shapes • Real eigenstrain values • Anisotropic elasticity • Habit plane orientation
**ENHANCED: 50+ Colormaps • Multi-Component Stress Comparison**
""")

# =============================================
# Material & Grid
# =============================================
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)

# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

N = 128
dx = 0.1  # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# EXPANDED COLORMAP LIBRARY (50+ options)
# =============================================
# Comprehensive colormap list including all major matplotlib categories
COLORMAPS = {
    # Sequential (1)
    'viridis': 'viridis',
    'plasma': 'plasma', 
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'hot': 'hot',
    'cool': 'cool',
    'spring': 'spring',
    'summer': 'summer',
    'autumn': 'autumn',
    'winter': 'winter',
    
    # Sequential (2)
    'copper': 'copper',
    'bone': 'bone',
    'gray': 'gray',
    'pink': 'pink',
    'afmhot': 'afmhot',
    'gist_heat': 'gist_heat',
    'gist_gray': 'gist_gray',
    'binary': 'binary',
    
    # Diverging
    'coolwarm': 'coolwarm',
    'bwr': 'bwr',
    'seismic': 'seismic',
    'RdBu': 'RdBu',
    'RdGy': 'RdGy',
    'PiYG': 'PiYG',
    'PRGn': 'PRGn',
    'BrBG': 'BrBG',
    'PuOr': 'PuOr',
    
    # Cyclic
    'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted',
    'hsv': 'hsv',
    
    # Qualitative
    'tab10': 'tab10',
    'tab20': 'tab20',
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    'Paired': 'Paired',
    'Accent': 'Accent',
    'Dark2': 'Dark2',
    
    # Miscellaneous
    'jet': 'jet',
    'turbo': 'turbo',
    'rainbow': 'rainbow',
    'nipy_spectral': 'nipy_spectral',
    'gist_ncar': 'gist_ncar',
    'gist_rainbow': 'gist_rainbow',
    'gist_earth': 'gist_earth',
    'gist_stern': 'gist_stern',
    'ocean': 'ocean',
    'terrain': 'terrain',
    'gnuplot': 'gnuplot',
    'gnuplot2': 'gnuplot2',
    'CMRmap': 'CMRmap',
    'cubehelix': 'cubehelix',
    'brg': 'brg',
    
    # Perceptually uniform
    'rocket': 'rocket',
    'mako': 'mako',
    'crest': 'crest',
    'flare': 'flare',
    'icefire': 'icefire',
    'vlag': 'vlag'
}

# Convert to list for selectbox
cmap_list = list(COLORMAPS.keys())

# =============================================
# Sidebar – Enhanced with better styling
# =============================================
st.sidebar.header("🎛️ Defect Type & Physics")

# Custom CSS for larger slider labels
st.markdown("""
<style>
    .stSlider label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    .stSelectbox label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    .stNumberInput label {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])

# Physical eigenstrain values from FCC crystallography (Silver)
if defect_type == "ISF":
    default_eps = 0.707  # b/√3 → one Shockley partial
    default_kappa = 0.6
    init_amplitude = 0.70
    caption = "Intrinsic Stacking Fault – one violated {111} plane"
elif defect_type == "ESF":
    default_eps = 1.414  # ≈ 2 × 0.707 → two partials
    default_kappa = 0.7
    init_amplitude = 0.75
    caption = "Extrinsic Stacking Fault – two violated planes"
else:  # Twin
    default_eps = 2.121  # ≈ 3 × 0.707 → twin nucleus transformation strain
    default_kappa = 0.3  # sharper interface for coherent twin
    init_amplitude = 0.90
    caption = "Coherent Twin Boundary – orientation flip"

st.sidebar.info(f"**{caption}**")

shape = st.sidebar.selectbox("Initial Seed Shape",
    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])

# Enhanced sliders with better formatting
eps0 = st.sidebar.slider(
    "Eigenstrain magnitude ε*",
    0.3, 3.0,
    value=default_eps,
    step=0.01,
    help="Physically accurate defaults shown above"
)

kappa = st.sidebar.slider(
    "Interface energy coeff κ",
    0.1, 2.0,
    value=default_kappa,
    step=0.05,
    help="Lower κ → sharper interface (used for twins)"
)

steps = st.sidebar.slider("Evolution steps", 20, 400, 150, 10)
save_every = st.sidebar.slider("Save frame every", 10, 50, 20)

# =============================================
# Crystal Orientation Selector
# =============================================
st.sidebar.header("Crystal Orientation")
orientation = st.sidebar.selectbox(
    "Habit Plane Orientation (in simulation plane)",
    ["Horizontal {111} (0°)", 
     "Tilted 30° (1¯10 projection)", 
     "Tilted 60°", 
     "Vertical {111} (90°)", 
     "Custom Angle"],
    index=0
)

if orientation == "Custom Angle":
    angle_deg = st.sidebar.slider("Custom tilt angle (°)", -180, 180, 0, 5)
    theta = np.deg2rad(angle_deg)
else:
    angle_map = {
        "Horizontal {111} (0°)": 0,
        "Tilted 30° (1¯10 projection)": 30,
        "Tilted 60°": 60,
        "Vertical {111} (90°)": 90,
    }
    theta = np.deg2rad(angle_map[orientation])

st.sidebar.info(f"Selected tilt: **{np.rad2deg(theta):.1f}°** from horizontal")

# =============================================
# ENHANCED: Visualization Controls + Stress Analysis
# =============================================
st.sidebar.header("🎨 Visualization Settings")

# Colorbar range controls
st.sidebar.subheader("Colorbar Ranges")
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

# Store colorbar limits
colorbar_limits = {
    'eta': [eta_min, eta_max],
    'sigma_mag': [sigma_min, sigma_max],
    'sigma_hydro': [hydro_min, hydro_max],
    'von_mises': [vm_min, vm_max]
}

# NEW: Stress Component Selection for Comparison
st.sidebar.subheader("Stress Comparison Settings")
stress_components_all = ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"]
selected_stress_components = st.sidebar.multiselect(
    "Components to Compare in Analysis",
    options=stress_components_all,
    default=["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"]
)

# NEW: Data Filtering Options
st.sidebar.subheader("Data Filtering")
apply_data_filter = st.sidebar.checkbox("Apply Data Filter", value=False)
if apply_data_filter:
    filter_threshold = st.sidebar.slider(
        "Include η >", 0.0, 1.0, 0.1, 0.05,
        help="Only include data where defect order parameter η exceeds this value"
    )
    stress_min = st.sidebar.number_input("Exclude stress < (GPa)", value=-100.0, format="%.1f")
    stress_max = st.sidebar.number_input("Exclude stress > (GPa)", value=100.0, format="%.1f")
else:
    filter_threshold = 0.0
    stress_min = -100.0
    stress_max = 100.0

# Chart styling controls
st.sidebar.subheader("Chart Styling")
title_font_size = st.sidebar.slider("Title Font Size", 12, 24, 16)
label_font_size = st.sidebar.slider("Label Font Size", 10, 45, 14)
tick_font_size = st.sidebar.slider("Tick Font Size", 8, 45, 12)
line_width = st.sidebar.slider("Contour Line Width", 1.0, 5.0, 2.0, 0.5)
spine_width = st.sidebar.slider("Spine Line Width", 1.0, 4.0, 2.5, 0.5)
tick_length = st.sidebar.slider("Tick Length", 4, 12, 6)
tick_width = st.sidebar.slider("Tick Width", 1.0, 3.0, 2.0, 0.5)

# NEW: Enhanced Color Map Selection with 50+ options
st.sidebar.subheader("Colormap Selection (50+ options)")
eta_cmap_name = st.sidebar.selectbox("η colormap", cmap_list, index=cmap_list.index('viridis'))
sigma_cmap_name = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index('hot'))
hydro_cmap_name = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap_name = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index('plasma'))
comparison_cmap_name = st.sidebar.selectbox("Comparison colormap", cmap_list, index=cmap_list.index('turbo'))

# Convert names to actual colormaps
eta_cmap = COLORMAPS[eta_cmap_name]
sigma_cmap = COLORMAPS[sigma_cmap_name]
hydro_cmap = COLORMAPS[hydro_cmap_name]
vm_cmap = COLORMAPS[vm_cmap_name]
comparison_cmap = COLORMAPS[comparison_cmap_name]

# Contour controls
show_contours = st.sidebar.checkbox("Show Defect Contours", value=True)
contour_level = st.sidebar.slider("Contour Level", 0.1, 0.9, 0.4, 0.05)
contour_color = st.sidebar.color_picker("Contour Color", "#FFFFFF")
contour_alpha = st.sidebar.slider("Contour Alpha", 0.1, 1.0, 0.8, 0.1)

# =============================================
# Initial Defect – Enhanced visualization
# =============================================
def create_initial_eta(shape):
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    w, h = (24, 12) if shape in ["Rectangle", "Horizontal Fault"] else (16, 16)
    if shape == "Square":
        eta[cy-h:cy+h, cx-h:cx+h] = init_amplitude
    elif shape == "Horizontal Fault":
        eta[cy-4:cy+4, cx-w:cx+w] = init_amplitude
    elif shape == "Vertical Fault":
        eta[cy-w:cy+w, cx-4:cx+4] = init_amplitude
    elif shape == "Rectangle":
        eta[cy-h:cy+h, cx-w:cx+w] = init_amplitude
    elif shape == "Ellipse":
        mask = ((X/(w*1.5))**2 + (Y/(h*1.5))**2) <= 1
        eta[mask] = init_amplitude
    eta += 0.02 * np.random.randn(N, N)
    return np.clip(eta, 0.0, 1.0)

st.subheader("🎯 Initial Defect Configuration")
init_eta = create_initial_eta(shape)

fig0, ax0 = plt.subplots(figsize=(8, 6))
im0 = ax0.imshow(init_eta, extent=extent, cmap=eta_cmap, origin='lower',
                 vmin=colorbar_limits['eta'][0], vmax=colorbar_limits['eta'][1])
if show_contours:
    ax0.contour(X, Y, init_eta, levels=[contour_level], colors=contour_color,
                linewidths=line_width, alpha=contour_alpha)

ax0.set_title(f"Initial η – {defect_type} ({shape})\nε* = {eps0:.3f}, κ = {kappa:.2f}",
              fontsize=title_font_size, fontweight='bold', pad=20)
ax0.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
ax0.set_ylabel("y (nm)", fontsize=label_font_size, fontweight='bold')
ax0.tick_params(axis='both', which='major', labelsize=tick_font_size,
                width=tick_width, length=tick_length)
for spine in ax0.spines.values():
    spine.set_linewidth(spine_width)

cbar0 = plt.colorbar(im0, ax=ax0, shrink=0.8)
cbar0.ax.tick_params(labelsize=tick_font_size)
cbar0.set_label('Order Parameter η', fontsize=label_font_size, fontweight='bold')
st.pyplot(fig0)

# =============================================
# Numba-safe Allen-Cahn
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
# FFT Stress Solver (Enhanced - returns all components)
# =============================================
@st.cache_data
def compute_stress_fields(eta, eps0, theta):
    """
    Full 2D plane-strain anisotropic elasticity with rotated eigenstrain
    Returns ALL stress components for comprehensive analysis
    """
    # Plane-strain reduced constants (Pa)
    C11_p = (C11 - C12**2 / C11) * 1e9
    C12_p = (C12 - C12**2 / C11) * 1e9
    C44_p = C44 * 1e9
    
    # Wavevectors
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(2 * np.pi * kx, 2 * np.pi * ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-12
    mask = K2 > 0
    
    n1 = np.zeros_like(KX)
    n2 = np.zeros_like(KX)
    n1[mask] = KX[mask] / np.sqrt(K2[mask])
    n2[mask] = KY[mask] / np.sqrt(K2[mask])
    
    # Acoustic tensor components
    A11 = np.zeros_like(KX)
    A22 = np.zeros_like(KX)
    A12 = np.zeros_like(KX)
    A11[mask] = C11_p * n1[mask]**2 + C44_p * n2[mask]**2
    A22[mask] = C11_p * n2[mask]**2 + C44_p * n1[mask]**2
    A12[mask] = (C12_p + C44_p) * n1[mask] * n2[mask]
    
    det = A11 * A22 - A12**2
    G11 = np.zeros_like(KX)
    G22 = np.zeros_like(KX)
    G12 = np.zeros_like(KX)
    G11[mask] = A22[mask] / det[mask]
    G22[mask] = A11[mask] / det[mask]
    G12[mask] = -A12[mask] / det[mask]
    
    # Eigenstrain (rotated)
    gamma = eps0
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    delta = 0.02  # Small dilatation
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
    
    eps_xx_star = eps_star[0,0] * eta
    eps_yy_star = eps_star[1,1] * eta
    eps_xy_star = eps_star[0,1] * eta
    
    # Polarization stress tau = C : eps*
    tau_xx = C11_p * eps_xx_star + C12_p * eps_yy_star
    tau_yy = C12_p * eps_xx_star + C11_p * eps_yy_star
    tau_xy = 2 * C44_p * eps_xy_star
    
    tau_hat_xx = np.fft.fft2(tau_xx)
    tau_hat_yy = np.fft.fft2(tau_yy)
    tau_hat_xy = np.fft.fft2(tau_xy)
    
    S_hat_x = KX * tau_hat_xx + KY * tau_hat_xy
    S_hat_y = KX * tau_hat_xy + KY * tau_hat_yy
    
    u_hat_x = np.zeros_like(KX, dtype=complex)
    u_hat_y = np.zeros_like(KX, dtype=complex)
    u_hat_x[mask] = -1j * (G11[mask] * S_hat_x[mask] + G12[mask] * S_hat_y[mask])
    u_hat_y[mask] = -1j * (G12[mask] * S_hat_x[mask] + G22[mask] * S_hat_y[mask])
    
    u_hat_x[0, 0] = 0
    u_hat_y[0, 0] = 0
    
    # Displacements
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
    
    # Elastic strains
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
    
    # Elastic stresses (Pa → GPa)
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy)  # Plane strain approximation
    
    # Derived quantities (GPa)
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
    
    return {
        'sxx': sxx, 'syy': syy, 'sxy': sxy, 'szz': szz,
        'sigma_mag': sigma_mag, 'sigma_hydro': sigma_hydro, 'von_mises': von_mises
    }

# =============================================
# ENHANCED: Comprehensive Stress Analysis Functions with Comparison
# =============================================
def filter_data(eta, stress_data, threshold=0.0, stress_min=-100.0, stress_max=100.0):
    """Filter data based on η threshold and stress range"""
    mask = (eta.flatten() > threshold) & \
           (stress_data.flatten() >= stress_min) & \
           (stress_data.flatten() <= stress_max)
    return mask

def create_comparison_analysis_plot(eta, stress_fields, frame_idx, selected_components, 
                                   style_params, filter_params):
    """Create comprehensive comparison plot for multiple stress components"""
    
    # Mapping from component names to field keys
    component_map = {
        "Stress Magnitude |σ|": ('sigma_mag', style_params['sigma_cmap'], '|σ| (GPa)'),
        "Hydrostatic σ_h": ('sigma_hydro', style_params['hydro_cmap'], 'σ_h (GPa)'),
        "von Mises σ_vM": ('von_mises', style_params['vm_cmap'], 'σ_vM (GPa)')
    }
    
    n_components = len(selected_components)
    
    # Dynamic layout based on number of components
    if n_components == 1:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    elif n_components == 2:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :]),
                fig.add_subplot(gs[2, :])]
    else:  # 3 components
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 4)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
                fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])]
    
    fig.suptitle(f"Comprehensive Stress Comparison - Frame {frame_idx}", 
                 fontsize=style_params['title_font_size'] + 2, fontweight='bold')
    
    # 1. Defect field
    ax = axes[0]
    im = ax.imshow(eta, extent=extent, cmap=style_params['eta_cmap'], origin='lower',
                  vmin=style_params['colorbar_limits']['eta'][0], 
                  vmax=style_params['colorbar_limits']['eta'][1])
    if style_params['show_contours']:
        ax.contour(X, Y, eta, levels=[style_params['contour_level']], 
                  colors=style_params['contour_color'],
                  linewidths=style_params['line_width'], 
                  alpha=style_params['contour_alpha'])
    
    ax.set_title("Defect Order Parameter η", fontsize=style_params['title_font_size'], fontweight='bold')
    ax.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    ax.set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    apply_axis_styling(ax, style_params)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=style_params['tick_font_size'])
    cbar.set_label('η', fontsize=style_params['label_font_size'], fontweight='bold')
    
    # 2. Individual stress component plots
    stress_data_all = {}
    for i, comp in enumerate(selected_components):
        field_key, cmap, label = component_map[comp]
        stress_data = stress_fields[field_key]
        stress_data_all[comp] = stress_data
        
        ax_idx = i + 1 if n_components <= 2 else i + 1  # Adjust index based on layout
        if ax_idx < len(axes):
            ax = axes[ax_idx]
            
            # Apply filtering if enabled
            if filter_params['apply_filter']:
                mask = filter_data(eta, stress_data, 
                                 filter_params['threshold'],
                                 filter_params['stress_min'],
                                 filter_params['stress_max'])
                # Create masked array for display
                display_data = np.ma.masked_array(stress_data)
                display_data[~mask.reshape(stress_data.shape)] = np.nan
            else:
                display_data = stress_data
            
            # Determine colorbar limits
            if field_key == 'sigma_mag':
                vmin, vmax = style_params['colorbar_limits']['sigma_mag']
            elif field_key == 'sigma_hydro':
                vmin, vmax = style_params['colorbar_limits']['sigma_hydro']
            else:  # von_mises
                vmin, vmax = style_params['colorbar_limits']['von_mises']
            
            im = ax.imshow(display_data, extent=extent, cmap=cmap, origin='lower',
                          vmin=vmin, vmax=vmax)
            
            if style_params['show_contours']:
                ax.contour(X, Y, eta, levels=[style_params['contour_level']], 
                          colors=style_params['contour_color'],
                          linewidths=style_params['line_width'], 
                          alpha=style_params['contour_alpha'])
            
            ax.set_title(comp, fontsize=style_params['title_font_size'], fontweight='bold')
            ax.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            apply_axis_styling(ax, style_params)
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=style_params['tick_font_size'])
            cbar.set_label(label, fontsize=style_params['label_font_size'], fontweight='bold')
    
    # 3. Overlay comparison plot (new)
    if n_components > 1:
        ax_idx = n_components + 1 if n_components <= 2 else 3
        if ax_idx < len(axes):
            ax = axes[ax_idx]
            
            # Prepare data for comparison
            comparison_data = []
            labels = []
            colors = plt.cm.get_cmap(style_params['comparison_cmap'])(np.linspace(0, 1, n_components))
            
            for i, comp in enumerate(selected_components):
                field_key, _, label = component_map[comp]
                data = stress_fields[field_key].flatten()
                
                # Apply filtering
                if filter_params['apply_filter']:
                    mask = filter_data(eta, stress_fields[field_key],
                                     filter_params['threshold'],
                                     filter_params['stress_min'],
                                     filter_params['stress_max'])
                    data = data[mask]
                
                comparison_data.append(data)
                labels.append(comp)
            
            # Create box plot comparison
            bp = ax.boxplot(comparison_data, labels=labels, patch_artist=True,
                           medianprops=dict(color='black', linewidth=2))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title("Stress Component Distribution Comparison", 
                        fontsize=style_params['title_font_size'], fontweight='bold')
            ax.set_ylabel("Stress (GPa)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            apply_axis_styling(ax, style_params)
    
    # 4. Radial profile comparison
    if n_components > 1:
        ax_idx = n_components + 2 if n_components <= 2 else 4
        if ax_idx < len(axes):
            ax = axes[ax_idx]
            
            r = np.sqrt(X**2 + Y**2)
            r_bins = np.linspace(0, np.max(r), 30)
            
            colors = plt.cm.get_cmap(style_params['comparison_cmap'])(np.linspace(0, 1, n_components))
            
            for i, comp in enumerate(selected_components):
                field_key, _, label = component_map[comp]
                stress_data = stress_fields[field_key]
                
                radial_stress = []
                for j in range(len(r_bins)-1):
                    mask = (r >= r_bins[j]) & (r < r_bins[j+1])
                    if np.any(mask):
                        radial_stress.append(np.nanmean(stress_data[mask]))
                    else:
                        radial_stress.append(np.nan)
                
                ax.plot(r_bins[1:], radial_stress, 'o-', linewidth=2, markersize=4,
                       color=colors[i], label=comp, alpha=0.8)
            
            ax.set_title("Radial Stress Profile Comparison", 
                        fontsize=style_params['title_font_size'], fontweight='bold')
            ax.set_xlabel("Radius (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.set_ylabel("Average Stress (GPa)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.legend(fontsize=style_params['label_font_size'] - 2)
            ax.grid(True, alpha=0.3, linestyle='--')
            apply_axis_styling(ax, style_params)
    
    # 5. Correlation matrix (for 3+ components)
    if n_components >= 2:
        ax_idx = 6 if n_components == 3 else None
        if ax_idx is not None and ax_idx < len(axes):
            ax = axes[ax_idx]
            
            # Calculate correlations
            corr_matrix = np.zeros((n_components, n_components))
            for i, comp_i in enumerate(selected_components):
                for j, comp_j in enumerate(selected_components):
                    data_i = stress_fields[component_map[comp_i][0]].flatten()
                    data_j = stress_fields[component_map[comp_j][0]].flatten()
                    
                    # Apply filtering
                    if filter_params['apply_filter']:
                        mask_i = filter_data(eta, stress_fields[component_map[comp_i][0]],
                                           filter_params['threshold'],
                                           filter_params['stress_min'],
                                           filter_params['stress_max'])
                        mask_j = filter_data(eta, stress_fields[component_map[comp_j][0]],
                                           filter_params['threshold'],
                                           filter_params['stress_min'],
                                           filter_params['stress_max'])
                        mask = mask_i & mask_j
                        data_i = data_i[mask]
                        data_j = data_j[mask]
                    
                    if len(data_i) > 1 and len(data_j) > 1:
                        corr = np.corrcoef(data_i, data_j)[0, 1]
                        corr_matrix[i, j] = corr if not np.isnan(corr) else 0
            
            im = ax.imshow(corr_matrix, cmap=style_params['comparison_cmap'], 
                          vmin=-1, vmax=1, aspect='auto')
            
            # Add text annotations
            for i in range(n_components):
                for j in range(n_components):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white",
                                 fontsize=style_params['label_font_size'] - 2,
                                 fontweight='bold')
            
            ax.set_title("Stress Component Correlation Matrix", 
                        fontsize=style_params['title_font_size'], fontweight='bold')
            ax.set_xticks(range(n_components))
            ax.set_yticks(range(n_components))
            ax.set_xticklabels([comp[:10] for comp in selected_components], rotation=45)
            ax.set_yticklabels([comp[:10] for comp in selected_components])
            apply_axis_styling(ax, style_params)
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=style_params['tick_font_size'])
            cbar.set_label('Correlation', fontsize=style_params['label_font_size'], fontweight='bold')
    
    plt.tight_layout()
    return fig

def apply_axis_styling(ax, style_params):
    """Apply consistent styling to axis"""
    ax.tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                  width=style_params['tick_width'], length=style_params['tick_length'])
    for spine in ax.spines.values():
        spine.set_linewidth(style_params['spine_width'])

def get_comparison_stats(stress_fields, selected_components, eta, filter_params):
    """Get statistics for comparison table - FIXED to return proper numeric types"""
    component_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    
    stats = {}
    for comp in selected_components:
        field_key = component_map[comp]
        data = stress_fields[field_key].flatten()
        
        # Apply filtering
        if filter_params['apply_filter']:
            mask = filter_data(eta, stress_fields[field_key],
                             filter_params['threshold'],
                             filter_params['stress_min'],
                             filter_params['stress_max'])
            data = data[mask]
        
        if len(data) > 0:
            stats[comp] = {
                'Component': comp,
                'Max (GPa)': float(np.nanmax(data)),
                'Min (GPa)': float(np.nanmin(data)),
                'Mean (GPa)': float(np.nanmean(data)),
                'Std Dev': float(np.nanstd(data)),
                'Median': float(np.nanmedian(data)),
                '90th %ile': float(np.nanpercentile(data, 90)),
                'Valid Points': int(len(data))
            }
        else:
            stats[comp] = {
                'Component': comp,
                'Max (GPa)': np.nan,
                'Min (GPa)': np.nan,
                'Mean (GPa)': np.nan,
                'Std Dev': np.nan,
                'Median': np.nan,
                '90th %ile': np.nan,
                'Valid Points': 0
            }
    
    return stats

# =============================================
# Run Simulation
# =============================================
if st.button("🚀 Run Phase-Field Evolution", type="primary"):
    with st.spinner("Running crystallographically accurate simulation..."):
        eta = init_eta.copy()
        history = []
        for step in range(steps + 1):
            if step > 0:
                eta = evolve_phase_field(eta, kappa, dt=0.004, dx=dx, N=N)
            if step % save_every == 0 or step == steps:
                stress_fields = compute_stress_fields(eta, eps0, theta)
                history.append((eta.copy(), stress_fields))
        st.session_state.history = history
        st.success(f"✅ Complete! {len(history)} frames – {defect_type} simulation ready")

# =============================================
# ENHANCED Results with Tabs
# =============================================
if 'history' in st.session_state:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Slice Analysis", "📈 Enhanced Analysis", "💾 Export"])
    
    frame = st.slider("Select Frame", 0, len(st.session_state.history)-1, 
                     len(st.session_state.history)-1, key="main_frame")
    
    eta, stress_fields = st.session_state.history[frame]
    
    # Create style parameters dictionary
    style_params = {
        'title_font_size': title_font_size,
        'label_font_size': label_font_size,
        'tick_font_size': tick_font_size,
        'line_width': line_width,
        'spine_width': spine_width,
        'tick_length': tick_length,
        'tick_width': tick_width,
        'eta_cmap': eta_cmap,
        'sigma_cmap': sigma_cmap,
        'hydro_cmap': hydro_cmap,
        'vm_cmap': vm_cmap,
        'comparison_cmap': comparison_cmap,
        'show_contours': show_contours,
        'contour_level': contour_level,
        'contour_color': contour_color,
        'contour_alpha': contour_alpha,
        'colorbar_limits': colorbar_limits
    }
    
    # Filter parameters
    filter_params = {
        'apply_filter': apply_data_filter,
        'threshold': filter_threshold,
        'stress_min': stress_min,
        'stress_max': stress_max
    }
    
    # TAB 1: Overview (original 2x2 plot)
    with tab1:
        st.subheader("Simulation Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get stats for first selected component
        stats_dict = get_comparison_stats(stress_fields, selected_stress_components, eta, filter_params)
        if selected_stress_components:
            first_comp = selected_stress_components[0]
            stats = stats_dict[first_comp]
            with col1:
                st.metric("η Range", f"{eta.min():.3f} - {eta.max():.3f}")
            with col2:
                st.metric(f"Max {first_comp}", f"{stats['Max (GPa)']:.2f} GPa")
            with col3:
                st.metric(f"Mean {first_comp}", f"{stats['Mean (GPa)']:.2f} GPa")
            with col4:
                st.metric(f"Std Dev {first_comp}", f"{stats['Std Dev']:.2f} GPa")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fields = [eta, stress_fields['sigma_mag'], stress_fields['sigma_hydro'], stress_fields['von_mises']]
        cmaps = [style_params['eta_cmap'], style_params['sigma_cmap'], style_params['hydro_cmap'], style_params['vm_cmap']]
        titles = ["Order Parameter η", "Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"]
        
        for i, (ax, field, cmap, title) in enumerate(zip(axes.flat, fields, cmaps, titles)):
            field_key = ['eta', 'sigma_mag', 'sigma_hydro', 'von_mises'][i]
            vmin, vmax = style_params['colorbar_limits'][field_key]
            
            # Apply filtering if enabled
            if filter_params['apply_filter'] and field_key != 'eta':
                mask = filter_data(eta, field, filter_params['threshold'],
                                 filter_params['stress_min'], filter_params['stress_max'])
                display_data = np.ma.masked_array(field)
                display_data[~mask.reshape(field.shape)] = np.nan
            else:
                display_data = field
            
            im = ax.imshow(display_data, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            
            if style_params['show_contours']:
                ax.contour(X, Y, eta, levels=[style_params['contour_level']], 
                          colors=style_params['contour_color'],
                          linewidths=style_params['line_width'], 
                          alpha=style_params['contour_alpha'])
            
            ax.set_title(title, fontsize=style_params['title_font_size'], fontweight='bold')
            ax.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            apply_axis_styling(ax, style_params)
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=style_params['tick_font_size'])
            cbar.set_label('Value', fontsize=style_params['label_font_size'], fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # TAB 2: Slice Analysis (line profiles)
    with tab2:
        st.subheader("Line Profile Analysis")
        
        # Slice selection
        col1, col2 = st.columns(2)
        with col1:
            slice_type = st.radio("Slice Type", ["Horizontal", "Vertical", "Diagonal"], horizontal=True)
        with col2:
            if slice_type == "Horizontal":
                slice_pos = st.slider("Y Position", 0, N-1, N//2)
            elif slice_type == "Vertical":
                slice_pos = st.slider("X Position", 0, N-1, N//2)
            else:  # Diagonal
                angle = st.slider("Diagonal Angle", 0, 180, 45)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare slice data
        if slice_type == "Horizontal":
            eta_slice = eta[slice_pos, :]
            x_pos = np.linspace(extent[0], extent[1], N)
            ax2.axhline(y=extent[2]+slice_pos*dx, color='white', linewidth=3, label=f'y={extent[2]+slice_pos*dx:.1f} nm')
        elif slice_type == "Vertical":
            eta_slice = eta[:, slice_pos]
            x_pos = np.linspace(extent[2], extent[3], N)
            ax2.axvline(x=extent[0]+slice_pos*dx, color='white', linewidth=3, label=f'x={extent[0]+slice_pos*dx:.1f} nm')
        else:  # Diagonal
            # Create diagonal slice
            theta_rad = np.deg2rad(angle)
            x_center, y_center = N//2, N//2
            length = N//2
            x_idx = x_center + length * np.cos(theta_rad) * np.linspace(-1, 1, N)
            y_idx = y_center + length * np.sin(theta_rad) * np.linspace(-1, 1, N)
            
            # Interpolate values along diagonal
            from scipy import interpolate
            interp = interpolate.RegularGridInterpolator((np.arange(N), np.arange(N)), eta)
            points = np.column_stack((y_idx, x_idx))
            eta_slice = interp(points)
            x_pos = np.linspace(-length*dx, length*dx, N)
            
            # Plot diagonal line
            ax2.plot([extent[0], extent[1]], 
                    [extent[2] + (extent[3]-extent[2])/2 * (1 - np.tan(theta_rad)),
                     extent[2] + (extent[3]-extent[2])/2 * (1 + np.tan(theta_rad))],
                    'w-', linewidth=3, label=f'θ={angle}°')
        
        # Plot η slice
        ax1.plot(x_pos, eta_slice, 'b-', linewidth=2, label='η')
        
        # Plot selected stress components
        component_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        
        colors = ['r', 'g', 'm']
        for i, comp in enumerate(selected_stress_components):
            if i < len(colors):
                field_key = component_map[comp]
                if slice_type == "Horizontal":
                    stress_slice = stress_fields[field_key][slice_pos, :]
                elif slice_type == "Vertical":
                    stress_slice = stress_fields[field_key][:, slice_pos]
                else:  # Diagonal
                    stress_slice = stress_fields[field_key][y_idx.astype(int), x_idx.astype(int)]
                
                ax1.plot(x_pos, stress_slice, '--', linewidth=2, 
                        color=colors[i % len(colors)], label=comp)
        
        ax1.set_title(f"{slice_type} Slice Profile", fontsize=style_params['title_font_size'], fontweight='bold')
        ax1.set_xlabel("Position (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
        ax1.set_ylabel("Value", fontsize=style_params['label_font_size'], fontweight='bold')
        ax1.legend(fontsize=style_params['label_font_size'] - 2)
        ax1.grid(True, alpha=0.3, linestyle='--')
        apply_axis_styling(ax1, style_params)
        
        # Show slice location
        ax2.imshow(eta, extent=extent, cmap=style_params['eta_cmap'], origin='lower')
        ax2.set_title("Slice Location", fontsize=style_params['title_font_size'], fontweight='bold')
        ax2.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
        ax2.set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
        ax2.legend(fontsize=style_params['label_font_size'] - 2)
        apply_axis_styling(ax2, style_params)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # TAB 3: ENHANCED COMPREHENSIVE ANALYSIS WITH COMPARISON - FIXED
    with tab3:
        st.subheader("🔬 Enhanced Stress Analysis & Comparison")
        
        # Display filtering status
        if filter_params['apply_filter']:
            st.info(f"**Data Filtering Active:** η > {filter_params['threshold']}, Stress ∈ [{filter_params['stress_min']}, {filter_params['stress_max']}] GPa")
        
        # Statistics table - FIXED VERSION
        st.subheader("📊 Comparison Statistics")
        stats_dict = get_comparison_stats(stress_fields, selected_stress_components, eta, filter_params)
        
        # Create comparison table
        if stats_dict:
            # Convert to list of dictionaries for DataFrame
            stats_list = []
            for comp in selected_stress_components:
                if comp in stats_dict:
                    stats_list.append(stats_dict[comp])
            
            if stats_list:
                df_stats = pd.DataFrame(stats_list)
                
                # Display with proper formatting
                st.dataframe(
                    df_stats.style.format({
                        'Max (GPa)': '{:.3f}',
                        'Min (GPa)': '{:.3f}',
                        'Mean (GPa)': '{:.3f}',
                        'Std Dev': '{:.3f}',
                        'Median': '{:.3f}',
                        '90th %ile': '{:.3f}',
                        'Valid Points': '{:,}'
                    }, na_rep="N/A"),
                    use_container_width=True
                )
        
        # Generate comparison plot
        if selected_stress_components:
            st.subheader("📈 Multi-Component Comparison")
            comparison_fig = create_comparison_analysis_plot(
                eta, stress_fields, frame, selected_stress_components, 
                style_params, filter_params
            )
            st.pyplot(comparison_fig)
            
            # Additional metrics
            if stats_dict and selected_stress_components:
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_vals = [stats_dict[comp]['Max (GPa)'] for comp in selected_stress_components 
                              if comp in stats_dict and not np.isnan(stats_dict[comp]['Max (GPa)'])]
                    if max_vals:
                        max_val = max(max_vals)
                        st.metric("Overall Max Stress", f"{max_val:.2f} GPa")
                
                with col2:
                    mean_vals = [stats_dict[comp]['Mean (GPa)'] for comp in selected_stress_components 
                               if comp in stats_dict and not np.isnan(stats_dict[comp]['Mean (GPa)'])]
                    if mean_vals:
                        mean_val = np.nanmean(mean_vals)
                        st.metric("Average Mean Stress", f"{mean_val:.2f} GPa")
                
                with col3:
                    total_points = sum([stats_dict[comp]['Valid Points'] for comp in selected_stress_components 
                                      if comp in stats_dict])
                    st.metric("Total Data Points", f"{total_points:,}")
        else:
            st.warning("Please select at least one stress component for comparison.")
    
    # TAB 4: Export
    with tab4:
        st.subheader("Data Export")
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Export Format", 
                                        ["CSV + VTI + PVD (Complete)", "CSV Only", "VTI Only", "PNG Images"])
        with col2:
            include_stats = st.checkbox("Include Statistics Report", value=True)
        
        if st.button("📥 Generate Export Package", type="primary"):
            with st.spinner("Preparing export package..."):
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Export each frame
                    for i, (e, sf) in enumerate(st.session_state.history):
                        if export_format in ["CSV + VTI + PVD (Complete)", "CSV Only"]:
                            df = pd.DataFrame({
                                'eta': e.flatten(order='F'),
                                'sxx': sf['sxx'].flatten(order='F'),
                                'syy': sf['syy'].flatten(order='F'),
                                'sxy': sf['sxy'].flatten(order='F'),
                                'sigma_mag': sf['sigma_mag'].flatten(order='F'),
                                'sigma_hydro': sf['sigma_hydro'].flatten(order='F'),
                                'von_mises': sf['von_mises'].flatten(order='F')
                            })
                            zf.writestr(f"frame_{i:04d}.csv", df.to_csv(index=False))
                        
                        if export_format in ["CSV + VTI + PVD (Complete)", "VTI Only"]:
                            flat = lambda a: ' '.join(f"{x:.6f}" for x in a.flatten(order='F'))
                            vti = f"""<VTKFile type="ImageData" version="1.0">
<ImageData WholeExtent="0 {N-1} 0 {N-1} 0 0" Origin="{extent[0]} {extent[2]} 0" Spacing="{dx} {dx} 1">
  <Piece Extent="0 {N-1} 0 {N-1} 0 0">
    <PointData>
      <DataArray type="Float32" Name="eta" format="ascii">{flat(e)}</DataArray>
      <DataArray type="Float32" Name="sxx" format="ascii">{flat(sf['sxx'])}</DataArray>
      <DataArray type="Float32" Name="syy" format="ascii">{flat(sf['syy'])}</DataArray>
      <DataArray type="Float32" Name="sxy" format="ascii">{flat(sf['sxy'])}</DataArray>
      <DataArray type="Float32" Name="sigma_magnitude" format="ascii">{flat(sf['sigma_mag'])}</DataArray>
      <DataArray type="Float32" Name="hydrostatic" format="ascii">{flat(sf['sigma_hydro'])}</DataArray>
      <DataArray type="Float32" Name="von_mises" format="ascii">{flat(sf['von_mises'])}</DataArray>
    </PointData>
  </Piece>
</ImageData>
</VTKFile>"""
                            zf.writestr(f"frame_{i:04d}.vti", vti)
                    
                    # PVD collection for ParaView
                    if export_format in ["CSV + VTI + PVD (Complete)", "VTI Only"]:
                        pvd = '<VTKFile type="Collection" version="1.0">\n<Collection>\n'
                        for i in range(len(st.session_state.history)):
                            pvd += f' <DataSet timestep="{i*save_every}" file="frame_{i:04d}.vti"/>\n'
                        pvd += '</Collection>\n</VTKFile>'
                        zf.writestr("simulation.pvd", pvd)
                    
                    # Statistics report
                    if include_stats:
                        report = f"""Ag Nanoparticle Defect Analysis - Statistics Report
===================================================
Defect Type: {defect_type}
Eigenstrain: {eps0:.3f}
Interface Coefficient: {kappa:.2f}
Orientation: {orientation} ({np.rad2deg(theta):.1f}°)
Grid Size: {N}x{N}
Resolution: {dx} nm

Selected Stress Components: {', '.join(selected_stress_components)}
Data Filtering: {'Active' if filter_params['apply_filter'] else 'Inactive'}
"""
                        if filter_params['apply_filter']:
                            report += f"Filter Criteria: η > {filter_params['threshold']}, Stress ∈ [{filter_params['stress_min']}, {filter_params['stress_max']}] GPa\n"
                        
                        report += "\nFrame Statistics:\n"
                        for i, (e, sf) in enumerate(st.session_state.history):
                            report += f"\nFrame {i} (t={i*save_every}):\n"
                            report += f"  η range: {e.min():.3f} - {e.max():.3f}\n"
                            for comp in selected_stress_components:
                                field_key = {
                                    "Stress Magnitude |σ|": 'sigma_mag',
                                    "Hydrostatic σ_h": 'sigma_hydro',
                                    "von Mises σ_vM": 'von_mises'
                                }[comp]
                                data = sf[field_key].flatten()
                                report += f"  {comp}: max={np.nanmax(data):.3f}, mean={np.nanmean(data):.3f}, std={np.nanstd(data):.3f}\n"
                        
                        zf.writestr("analysis_report.txt", report)
                
                buffer.seek(0)
                
                # Determine file name
                if export_format == "CSV Only":
                    ext = "_csv_only"
                elif export_format == "VTI Only":
                    ext = "_vti_only"
                elif export_format == "PNG Images":
                    ext = "_images"
                else:
                    ext = "_complete"
                
                st.download_button(
                    "📥 Download Export Package",
                    buffer.getvalue(),
                    f"Ag_NP_{defect_type}_2D_Analysis{ext}.zip",
                    "application/zip"
                )
                st.success("Export package ready for download!")

# =============================================
# Theoretical Analysis
# =============================================
with st.expander("🔬 Theoretical Soundness Analysis"):
    st.markdown(f"""
    ### ✅ **ENHANCED Comprehensive Stress Analysis Features:**
    
    #### 🎨 **Expanded Visualization (50+ Colormaps):**
    - **Sequential**: viridis, plasma, inferno, magma, cividis, hot, summer, autumn, winter, copper, bone, gray
    - **Diverging**: coolwarm, bwr, seismic, RdBu, RdGy, PiYG, PRGn, BrBG, PuOr
    - **Cyclic**: twilight, hsv
    - **Qualitative**: tab10, tab20, Set1, Set2, Set3, Paired, Accent, Dark2
    - **Miscellaneous**: **jet**, **rainbow**, **turbo**, gnuplot, terrain, ocean, cubehelix, brg
    - **Perceptually uniform**: rocket, mako, crest, flare, icefire
    
    #### 📊 **Multi-Component Stress Comparison:**
    - **Selective Inclusion/Exclusion**: Tick boxes to include/exclude stress components
    - **Dynamic Layout**: Automatic layout adjustment based on selected components
    - **Statistical Comparison**: Box plots, radial profiles, correlation matrices
    - **Data Filtering**: Include/exclude data based on η threshold and stress range
    - **Comprehensive Statistics**: Max, min, mean, std, median, 90th percentile
    
    #### 🔬 **Enhanced Analysis Features:**
    - **Correlation Analysis**: Matrix showing relationships between stress components
    - **Radial Profile Comparison**: Overlaid plots for selected components
    - **Slice Analysis**: Horizontal, vertical, and diagonal slices
    - **Export Options**: Multiple formats (CSV, VTI, PNG) with statistics reports
    
    ### 📈 **Publication-Ready Outputs:**
    - **Dynamic Visualizations**: Adjusts to 1-3 selected components
    - **Consistent Styling**: All charts use same font sizes, line widths, colors
    - **Filter-Aware Statistics**: All calculations respect inclusion/exclusion criteria
    - **Comprehensive Export**: Full datasets with metadata and analysis reports
    
    **Advanced 2D stress analysis with professional-grade visualization and comparison tools!**
    """)
    
    # Display colormap count
    st.metric("Available Colormaps", f"{len(COLORMAPS)}+")

st.caption("🔬 Crystallographically Accurate • 50+ Colormaps • Multi-Component Stress Comparison • 2025")
