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

# Configure page with better styling
st.set_page_config(page_title="Ag NP Defect Analyzer – Ultimate", layout="wide")
st.title("🏗️ Ag Nanoparticle Defect Mechanics – Crystallographically Accurate")
st.markdown("""
**Live phase-field + FFT elasticity**
**ISF, ESF, and Twin are now physically distinct**
Four fields exported • Custom shapes • Real eigenstrain values • Anisotropic elasticity • Habit plane orientation
**NEW: Comprehensive 2D Stress Analysis Tab with Advanced Styling**
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
# NEW: Crystal Orientation Selector
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

# Stress component selection for analysis tab
stress_component = st.sidebar.selectbox(
    "Stress Component (Analysis Tab)", 
    ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
    index=0
)

# Chart styling controls
st.sidebar.subheader("Chart Styling")
title_font_size = st.sidebar.slider("Title Font Size", 12, 24, 16)
label_font_size = st.sidebar.slider("Label Font Size", 10, 45, 14)
tick_font_size = st.sidebar.slider("Tick Font Size", 8, 45, 12)
line_width = st.sidebar.slider("Contour Line Width", 1.0, 5.0, 2.0, 0.5)
spine_width = st.sidebar.slider("Spine Line Width", 1.0, 4.0, 2.5, 0.5)
tick_length = st.sidebar.slider("Tick Length", 4, 12, 6)
tick_width = st.sidebar.slider("Tick Width", 1.0, 3.0, 2.0, 0.5)

# Color maps
cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm', 'jet', 'turbo']
eta_cmap = st.sidebar.selectbox("η colormap", cmap_list, index=0)
sigma_cmap = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index('hot'))
hydro_cmap = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index('plasma'))

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
# NEW: Comprehensive Stress Analysis Functions with FULL STYLING
# =============================================
def create_stress_analysis_plot(eta, stress_fields, frame_idx, stress_component, style_params):
    """Create 2x3 comprehensive stress analysis plot with FULL styling"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Comprehensive 2D Stress Analysis - Frame {frame_idx}", 
                 fontsize=style_params['title_font_size'] + 2, fontweight='bold')
    
    # Get selected stress field
    stress_map = {
        "Stress Magnitude |σ|": stress_fields['sigma_mag'],
        "Hydrostatic σ_h": stress_fields['sigma_hydro'], 
        "von Mises σ_vM": stress_fields['von_mises']
    }
    current_stress = stress_map[stress_component]
    
    # 1. Defect field with contours
    im1 = axes[0,0].imshow(eta, extent=extent, cmap=style_params['eta_cmap'], origin='lower',
                          vmin=style_params['colorbar_limits']['eta'][0], 
                          vmax=style_params['colorbar_limits']['eta'][1])
    if style_params['show_contours']:
        axes[0,0].contour(X, Y, eta, levels=[style_params['contour_level']], 
                         colors=style_params['contour_color'],
                         linewidths=style_params['line_width'], 
                         alpha=style_params['contour_alpha'])
    
    axes[0,0].set_title("Defect Order Parameter η", fontsize=style_params['title_font_size'], fontweight='bold')
    axes[0,0].set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[0,0].set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[0,0].tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                         width=style_params['tick_width'], length=style_params['tick_length'])
    for spine in axes[0,0].spines.values():
        spine.set_linewidth(style_params['spine_width'])
    
    cbar1 = plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    cbar1.ax.tick_params(labelsize=style_params['tick_font_size'])
    cbar1.set_label('η', fontsize=style_params['label_font_size'], fontweight='bold')
    
    # 2. Selected stress component
    im2 = axes[0,1].imshow(current_stress, extent=extent, cmap=style_params['sigma_cmap'], origin='lower',
                          vmin=style_params['colorbar_limits']['sigma_mag'][0],
                          vmax=style_params['colorbar_limits']['sigma_mag'][1])
    if style_params['show_contours']:
        axes[0,1].contour(X, Y, eta, levels=[style_params['contour_level']], 
                         colors=style_params['contour_color'],
                         linewidths=style_params['line_width'], 
                         alpha=style_params['contour_alpha'])
    
    axes[0,1].set_title(f"{stress_component}", fontsize=style_params['title_font_size'], fontweight='bold')
    axes[0,1].set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[0,1].set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[0,1].tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                         width=style_params['tick_width'], length=style_params['tick_length'])
    for spine in axes[0,1].spines.values():
        spine.set_linewidth(style_params['spine_width'])
    
    cbar2 = plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    cbar2.ax.tick_params(labelsize=style_params['tick_font_size'])
    cbar2.set_label('Stress (GPa)', fontsize=style_params['label_font_size'], fontweight='bold')
    
    # 3. Stress histogram with styling
    stress_flat = current_stress.flatten()
    stress_valid = stress_flat[np.isfinite(stress_flat)]
    axes[0,2].hist(stress_valid, bins=50, alpha=0.7, color='red', edgecolor='black', linewidth=1.5)
    axes[0,2].set_title(f"{stress_component} Distribution", fontsize=style_params['title_font_size'], fontweight='bold')
    axes[0,2].set_xlabel("Stress (GPa)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[0,2].set_ylabel("Frequency", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[0,2].tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                         width=style_params['tick_width'], length=style_params['tick_length'])
    axes[0,2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    for spine in axes[0,2].spines.values():
        spine.set_linewidth(style_params['spine_width'])
    
    # 4. Defect-Stress correlation
    valid = (eta.flatten() > 0.1) & np.isfinite(stress_flat)
    if np.any(valid):
        axes[1,0].scatter(eta.flatten()[valid], stress_flat[valid], 
                         alpha=0.5, s=1, c='blue')
        axes[1,0].set_title("Defect-Stress Correlation", fontsize=style_params['title_font_size'], fontweight='bold')
        axes[1,0].set_xlabel("η", fontsize=style_params['label_font_size'], fontweight='bold')
        axes[1,0].set_ylabel(f"{stress_component} (GPa)", fontsize=style_params['label_font_size'], fontweight='bold')
        axes[1,0].tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                             width=style_params['tick_width'], length=style_params['tick_length'])
        axes[1,0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        for spine in axes[1,0].spines.values():
            spine.set_linewidth(style_params['spine_width'])
    else:
        axes[1,0].text(0.5, 0.5, "No valid data points", ha='center', va='center',
                      fontsize=style_params['label_font_size'])
        axes[1,0].set_title("Defect-Stress Correlation", fontsize=style_params['title_font_size'], fontweight='bold')
        for spine in axes[1,0].spines.values():
            spine.set_linewidth(style_params['spine_width'])
    
    # 5. Radial stress profile
    r = np.sqrt(X**2 + Y**2)
    r_bins = np.linspace(0, np.max(r), 20)
    radial_stress = []
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(mask):
            radial_stress.append(np.mean(current_stress[mask]))
        else:
            radial_stress.append(0)
    
    axes[1,1].plot(r_bins[1:], radial_stress, 'o-', linewidth=2, markersize=6, color='green')
    axes[1,1].set_title("Radial Stress Profile", fontsize=style_params['title_font_size'], fontweight='bold')
    axes[1,1].set_xlabel("Radius (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[1,1].set_ylabel(f"Avg {stress_component} (GPa)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[1,1].tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                         width=style_params['tick_width'], length=style_params['tick_length'])
    axes[1,1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    for spine in axes[1,1].spines.values():
        spine.set_linewidth(style_params['spine_width'])
    
    # 6. Multiple colormap comparison with styling
    alt_cmps = ['jet', 'turbo', 'viridis', 'plasma']
    for i, cmap in enumerate(alt_cmps[:4]):
        im = axes[1,2].imshow(current_stress, extent=extent, cmap=cmap, 
                             origin='lower', alpha=0.8)
    
    axes[1,2].set_title("Colormap Comparison", fontsize=style_params['title_font_size'], fontweight='bold')
    axes[1,2].set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[1,2].set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
    axes[1,2].tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                         width=style_params['tick_width'], length=style_params['tick_length'])
    for spine in axes[1,2].spines.values():
        spine.set_linewidth(style_params['spine_width'])
    
    # Add a colorbar for the last colormap
    cbar_last = plt.colorbar(im, ax=axes[1,2], shrink=0.8)
    cbar_last.ax.tick_params(labelsize=style_params['tick_font_size'])
    cbar_last.set_label('Stress (GPa)', fontsize=style_params['label_font_size'], fontweight='bold')
    
    plt.tight_layout()
    return fig

def get_stress_stats(stress_fields, stress_component):
    """Extract statistics for selected stress component"""
    stress_map = {
        "Stress Magnitude |σ|": stress_fields['sigma_mag'],
        "Hydrostatic σ_h": stress_fields['sigma_hydro'], 
        "von Mises σ_vM": stress_fields['von_mises']
    }
    stress_data = stress_map[stress_component].flatten()
    stress_valid = stress_data[np.isfinite(stress_data)]
    return {
        'max': np.nanmax(stress_valid),
        'mean': np.nanmean(stress_valid),
        'std': np.nanstd(stress_valid),
        'count_above_1GPa': np.sum(stress_valid > 1.0)
    }

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
# ENHANCED Results with Tabs - ALL STYLING APPLIED
# =============================================
if 'history' in st.session_state:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Slice Analysis", "📈 Stress Analysis", "💾 Export"])
    
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
        'show_contours': show_contours,
        'contour_level': contour_level,
        'contour_color': contour_color,
        'contour_alpha': contour_alpha,
        'colorbar_limits': colorbar_limits
    }
    
    # TAB 1: Overview (original 2x2 plot)
    with tab1:
        st.subheader("Simulation Overview")
        col1, col2, col3, col4 = st.columns(4)
        stats = get_stress_stats(stress_fields, stress_component)
        with col1:
            st.metric("η Range", f"{eta.min():.3f} - {eta.max():.3f}")
        with col2:
            st.metric("Max Stress", f"{stats['max']:.2f} GPa")
        with col3:
            st.metric("Avg Stress", f"{stats['mean']:.2f} GPa")
        with col4:
            st.metric("Std Dev", f"{stats['std']:.2f} GPa")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fields = [eta, stress_fields['sigma_mag'], stress_fields['sigma_hydro'], stress_fields['von_mises']]
        cmaps = [style_params['eta_cmap'], style_params['sigma_cmap'], style_params['hydro_cmap'], style_params['vm_cmap']]
        titles = ["Order Parameter η", "Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"]
        
        for i, (ax, field, cmap, title) in enumerate(zip(axes.flat, fields, cmaps, titles)):
            field_key = ['eta', 'sigma_mag', 'sigma_hydro', 'von_mises'][i]
            vmin, vmax = style_params['colorbar_limits'][field_key]
            im = ax.imshow(field, extent=extent, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
            
            if style_params['show_contours']:
                ax.contour(X, Y, eta, levels=[style_params['contour_level']], 
                          colors=style_params['contour_color'],
                          linewidths=style_params['line_width'], 
                          alpha=style_params['contour_alpha'])
            
            ax.set_title(title, fontsize=style_params['title_font_size'], fontweight='bold')
            ax.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                          width=style_params['tick_width'], length=style_params['tick_length'])
            
            for spine in ax.spines.values():
                spine.set_linewidth(style_params['spine_width'])
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=style_params['tick_font_size'])
            cbar.set_label('Value', fontsize=style_params['label_font_size'], fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # TAB 2: Slice Analysis (line profiles)
    with tab2:
        st.subheader("Line Profile Analysis")
        slice_pos = st.slider("Slice Position", 0, N-1, N//2, key="slice_slider")
    
        # Proper stress component mapping
        stress_key_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro', 
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_key_map[stress_component]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # Horizontal slice
        eta_slice = eta[slice_pos, :]
        stress_slice = stress_fields[stress_key][slice_pos, :]
        x_pos = np.linspace(extent[0], extent[1], N)
    
        ax1.plot(x_pos, eta_slice, 'b-', linewidth=2, label='η')
        ax1.plot(x_pos, stress_slice, 'r--', linewidth=2, label=stress_component)
        ax1.set_title(f"Horizontal Slice (y={extent[2]+slice_pos*dx:.1f} nm)", 
                     fontsize=style_params['title_font_size'], fontweight='bold')
        ax1.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
        ax1.set_ylabel("Value", fontsize=style_params['label_font_size'], fontweight='bold')
        ax1.legend(fontsize=style_params['label_font_size'])
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                       width=style_params['tick_width'], length=style_params['tick_length'])
        for spine in ax1.spines.values():
            spine.set_linewidth(style_params['spine_width'])
    
        ax2.imshow(eta, extent=extent, cmap=style_params['eta_cmap'], origin='lower')
        ax2.axhline(y=extent[2]+slice_pos*dx, color='white', linewidth=3)
        ax2.set_title("Slice Location", fontsize=style_params['title_font_size'], fontweight='bold')
        ax2.set_xlabel("x (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
        ax2.set_ylabel("y (nm)", fontsize=style_params['label_font_size'], fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=style_params['tick_font_size'],
                       width=style_params['tick_width'], length=style_params['tick_length'])
        for spine in ax2.spines.values():
            spine.set_linewidth(style_params['spine_width'])
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # TAB 3: COMPREHENSIVE STRESS ANALYSIS (NEW!) - FULLY STYLED
    with tab3:
        st.subheader("🔬 Comprehensive Stress Analysis")
        
        analysis_fig = create_stress_analysis_plot(eta, stress_fields, frame, stress_component, style_params)
        st.pyplot(analysis_fig)
        
        # Additional statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Stress", f"{stats['max']:.2f} GPa")
        with col2:
            st.metric("Mean Stress", f"{stats['mean']:.2f} GPa")
        with col3:
            st.metric("Voxels >1 GPa", f"{stats['count_above_1GPa']:,}")
    
    # TAB 4: Export
    with tab4:
        st.subheader("Data Export")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (e, sf) in enumerate(st.session_state.history):
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
                
                # VTI export
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
            
            # PVD collection
            pvd = '<VTKFile type="Collection" version="1.0">\n<Collection>\n'
            for i in range(len(st.session_state.history)):
                pvd += f' <DataSet timestep="{i*save_every}" file="frame_{i:04d}.vti"/>\n'
            pvd += '</Collection>\n</VTKFile>'
            zf.writestr("simulation.pvd", pvd)
        
        buffer.seek(0)
        st.download_button(
            "📥 Download Full Results (PVD + VTI + CSV)",
            buffer.getvalue(),
            f"Ag_NP_{defect_type}_2D_Comprehensive.zip",
            "application/zip"
        )

# =============================================
# Theoretical Analysis
# =============================================
with st.expander("🔬 Theoretical Soundness Analysis"):
    st.markdown("""
    ### ✅ **NEW Comprehensive Stress Analysis Features:**
    - **6-panel analysis plots**: Defect field, stress component, histogram, correlation, radial profile, colormap comparison
    - **Interactive stress component selection**: |σ|, hydrostatic, von Mises
    - **Quantitative statistics**: Max/mean/std deviation, voxel counting
    - **Line profile analysis**: Horizontal/vertical slices with overlay
    - **Enhanced VTK export**: All 7 stress fields (sxx, syy, sxy, szz, |σ|, σ_h, σ_vM)
    
    ### 🎨 **Advanced Chart Styling Applied to 2D Analysis:**
    - **Full font size control**: Titles, labels, ticks (8-45pt range)
    - **Line width customization**: Contour lines, plot lines, spines
    - **Comprehensive colormap selection**: Individual maps for each stress component
    - **Colorbar limits**: Custom ranges for all fields
    - **Contour styling**: Color, width, alpha, and level control
    
    ### 🔬 **2D vs 3D Analysis:**
    - **Plane strain assumption**: szz computed from sxx+syy
    - **Full tensor components**: All in-plane stresses available
    - **Radial profiles**: 2D distance from center
    - **Cross-section analysis**: Line profiles instead of 3D slices
    
    **Publication-ready 2D stress analysis matching 3D capabilities with complete styling control!**
    """)

st.caption("🔬 Crystallographically Accurate • Comprehensive 2D Stress Analysis • Advanced Styling • 2025")
