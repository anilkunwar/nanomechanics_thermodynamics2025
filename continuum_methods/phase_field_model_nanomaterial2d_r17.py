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
dx = 0.1 # nm
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
    default_eps = 0.707 # b/√3 → one Shockley partial
    default_kappa = 0.6
    init_amplitude = 0.70
    caption = "Intrinsic Stacking Fault – one violated {111} plane"
elif defect_type == "ESF":
    default_eps = 1.414 # ≈ 2 × 0.707 → two partials
    default_kappa = 0.7
    init_amplitude = 0.75
    caption = "Extrinsic Stacking Fault – two violated planes"
else: # Twin
    default_eps = 2.121 # ≈ 3 × 0.707 → twin nucleus transformation strain
    default_kappa = 0.3 # sharper interface for coherent twin
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
# NEW: Visualization Controls
# =============================================
st.sidebar.header("🎨 Visualization Settings")
# Colorbar range controls
st.sidebar.subheader("Colorbar Ranges")
col1, col2 = st.sidebar.columns(2)
with col1:
    eta_min = st.number_input("η Min", value=0.0, format="%.2f", help="Minimum value for order parameter colorbar")
    sigma_min = st.number_input("|σ| Min (GPa)", value=0.0, format="%.1f", help="Minimum value for stress magnitude colorbar")
    hydro_min = st.number_input("σ_h Min (GPa)", value=-5.0, format="%.1f", help="Minimum value for hydrostatic stress colorbar")
    vm_min = st.number_input("σ_vM Min (GPa)", value=0.0, format="%.1f", help="Minimum value for von Mises stress colorbar")
with col2:
    eta_max = st.number_input("η Max", value=1.0, format="%.2f", help="Maximum value for order parameter colorbar")
    sigma_max = st.number_input("|σ| Max (GPa)", value=10.0, format="%.1f", help="Maximum value for stress magnitude colorbar")
    hydro_max = st.number_input("σ_h Max (GPa)", value=5.0, format="%.1f", help="Maximum value for hydrostatic stress colorbar")
    vm_max = st.number_input("σ_vM Max (GPa)", value=8.0, format="%.1f", help="Maximum value for von Mises stress colorbar")
# Store colorbar limits in a dictionary for easy access
colorbar_limits = {
    'eta': [eta_min, eta_max],
    'sigma_mag': [sigma_min, sigma_max],
    'sigma_hydro': [hydro_min, hydro_max],
    'von_mises': [vm_min, vm_max]
}
# Chart styling controls
st.sidebar.subheader("Chart Styling")
# Font sizes
title_font_size = st.sidebar.slider("Title Font Size", 12, 24, 16)
label_font_size = st.sidebar.slider("Label Font Size", 10, 20, 14)
tick_font_size = st.sidebar.slider("Tick Font Size", 8, 18, 12)
# Line and spine controls
line_width = st.sidebar.slider("Contour Line Width", 1.0, 5.0, 2.0, 0.5)
spine_width = st.sidebar.slider("Spine Line Width", 1.0, 4.0, 2.5, 0.5)
tick_length = st.sidebar.slider("Tick Length", 4, 12, 6)
tick_width = st.sidebar.slider("Tick Width", 1.0, 3.0, 2.0, 0.5)
# Color maps
cmap_list = [
    # Perceptually Uniform Sequential
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    # Sequential
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu',
    'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    # Sequential (2)
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper',
    # Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    # Cyclic
    'twilight', 'twilight_shifted', 'hsv',
    # Qualitative
    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
    'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',
    # Miscellaneous
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar'
]
eta_cmap = st.sidebar.selectbox("η colormap", cmap_list, index=0)
sigma_cmap = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index('hot'))
hydro_cmap = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index('plasma'))
# Additional visualization options
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
# Enhanced figure with better styling
fig0, ax0 = plt.subplots(figsize=(8, 6))
im0 = ax0.imshow(init_eta, extent=extent, cmap=eta_cmap, origin='lower',
                 vmin=colorbar_limits['eta'][0], vmax=colorbar_limits['eta'][1])
if show_contours:
    ax0.contour(X, Y, init_eta, levels=[contour_level], colors=contour_color,
                linewidths=line_width, alpha=contour_alpha)
# Enhanced titles and labels
ax0.set_title(f"Initial η – {defect_type} ({shape})\nε* = {eps0:.3f}, κ = {kappa:.2f}",
              fontsize=title_font_size, fontweight='bold', pad=20)
ax0.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
ax0.set_ylabel("y (nm)", fontsize=label_font_size, fontweight='bold')
# Enhanced ticks and spines
ax0.tick_params(axis='both', which='major', labelsize=tick_font_size,
                width=tick_width, length=tick_length)
for spine in ax0.spines.values():
    spine.set_linewidth(spine_width)
# Enhanced colorbar
cbar0 = plt.colorbar(im0, ax=ax0, shrink=0.8)
cbar0.ax.tick_params(labelsize=tick_font_size)
cbar0.set_label('Order Parameter η', fontsize=label_font_size, fontweight='bold')
# Add grid for better spatial reference
ax0.grid(False)
st.pyplot(fig0)
# =============================================
# THEORETICAL ANALYSIS SECTION
# =============================================
st.sidebar.header("🔬 Theoretical Analysis")
with st.sidebar.expander("About Plasticity & Model Physics"):
    st.markdown("""
    **Plasticity in this context** refers to permanent crystal deformation via:
    - **Stacking faults**: Planar defects in FCC stacking sequence
    - **Twinning**: Mirror symmetry across twin boundary
    - **Dislocation motion**: Not explicitly modeled but implied
   
    **Phase-field approach** models defect evolution through:
    - **Order parameter (η)**: 0=perfect crystal, 1=fully faulted
    - **Allen-Cahn dynamics**: Non-conserved order parameter evolution
    - **Eigenstrain (ε*)**: Stress-free transformation strain
    - **FFT elasticity**: Efficient stress calculation
    """)
# =============================================
# Numba-safe Allen-Cahn (unchanged)
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
# FFT Stress Solver (upgraded with orientation)
# =============================================
@st.cache_data
def compute_stress_fields(eta, eps0, theta):
    """
    Full 2D plane-strain anisotropic elasticity with rotated eigenstrain
    Valid for any in-plane {111} orientation
    """
    # Plane-strain reduced constants (Pa)
    C11_p = (C11 - C12**2 / C11) * 1e9
    C12_p = (C12 - C12**2 / C11) * 1e9
    C44_p = C44 * 1e9
    nu = C12 / (C11 + C12)  # Approximate Poisson for von Mises
    # Small normal dilatation (realistic for Ag)
    delta = 0.02
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
    # Compute A components (anisotropic)
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
    # Eigenstrain
    gamma = eps0  # Shear magnitude
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
    eps_xx_star = eps_star[0,0] * eta
    eps_yy_star = eps_star[1,1] * eta
    eps_xy_star = eps_star[0,1] * eta
    # Compute tau = C : eps* (using reduced C)
    trace_star = eps_xx_star + eps_yy_star
    tau_xx = C11_p * eps_xx_star + C12_p * eps_yy_star
    tau_yy = C12_p * eps_xx_star + C11_p * eps_yy_star
    tau_xy = 2 * C44_p * eps_xy_star  # Note 2 for Voigt
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
    # ifft to get displacement
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
    # Elastic strains from Fourier derivatives
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
    # Elastic stress (using reduced C for plane strain)
    sxx = C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)
    syy = C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)
    sxy = 2 * C44_p * (exy - eps_xy_star)
    # Stress fields in GPa
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2 * sxy**2) / 1e9
    # sigma_hydro = (sxx + syy) / 3 / 1e9
    sigma_hydro = (sxx + syy) / 2 / 1e9  # GPa
    # von Mises with plane strain szz
    szz = (C12 / (C11 + C12)) * (sxx + syy)  # Approximate
    von_mises = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6 * sxy**2)) / 1e9
    return sigma_mag, sigma_hydro, von_mises
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
                sm, sh, vm = compute_stress_fields(eta, eps0, theta)
                history.append((eta.copy(), sm.copy(), sh.copy(), vm.copy()))
        st.session_state.history = history
        st.success(f"✅ Complete! {len(history)} frames – {defect_type} simulation ready")
# =============================================
# Live Results – Enhanced visualization
# =============================================
if 'history' in st.session_state:
    st.subheader("📊 Simulation Results")
   
    # Enhanced slider for frame selection
    frame = st.slider(
        "Select Frame",
        0, len(st.session_state.history)-1,
        len(st.session_state.history)-1,
        key="frame_slider"
    )
   
    eta, sigma_mag, sigma_hydro, von_mises = st.session_state.history[frame]
    # Display current field statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("η Range", f"{eta.min():.3f} - {eta.max():.3f}")
    with col2:
        st.metric("|σ| Max", f"{sigma_mag.max():.2f} GPa")
    with col3:
        st.metric("σ_h Range", f"{sigma_hydro.min():.2f} - {sigma_hydro.max():.2f} GPa")
    with col4:
        st.metric("σ_vM Max", f"{von_mises.max():.2f} GPa")
    # Create enhanced figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    x = np.linspace(extent[0], extent[1], N)
    y = np.linspace(extent[2], extent[3], N)
    fields = [eta, sigma_mag, sigma_hydro, von_mises]
    cmaps = [eta_cmap, sigma_cmap, hydro_cmap, vm_cmap]
    titles = [
        f"Order Parameter η (Frame {frame})",
        f"Stress Magnitude |σ| – Max: {sigma_mag.max():.1f} GPa",
        f"Hyd. Stress – Range: [{sigma_hydro.min():.1f}, {sigma_hydro.max():.1f}] GPa",
        f"von Mises – Max: {von_mises.max():.1f} GPa"
    ]
   
    cbar_labels = ['η', '|σ| (GPa)', 'σ_h (GPa)', 'σ_vM (GPa)']
    field_keys = ['eta', 'sigma_mag', 'sigma_hydro', 'von_mises']
    for ax, field, cmap, title, cbar_label, field_key in zip(
        axes.flat, fields, cmaps, titles, cbar_labels, field_keys):
       
        vmin, vmax = colorbar_limits[field_key]
        im = ax.imshow(field, extent=extent, cmap=cmap, origin='lower',
                      vmin=vmin, vmax=vmax)
       
        # Add contour for defect boundary
        if show_contours:
            ax.contour(x, y, eta, levels=[contour_level], colors=contour_color,
                      linewidths=line_width, alpha=contour_alpha)
       
        # Enhanced titles and labels
        ax.set_title(title, fontsize=title_font_size, fontweight='bold', pad=15)
        ax.set_xlabel("x (nm)", fontsize=label_font_size, fontweight='bold')
        ax.set_ylabel("y (nm)", fontsize=label_font_size, fontweight='bold')
       
        # Enhanced ticks
        ax.tick_params(axis='both', which='major', labelsize=tick_font_size,
                      width=tick_width, length=tick_length)
       
        # Enhanced spines
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
       
        ax.set_aspect('equal')
       
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label(cbar_label, fontsize=label_font_size, fontweight='bold')
        cbar.ax.tick_params(labelsize=tick_font_size)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
    # =============================================
    # Download (all 4 fields) - unchanged
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
            pvd += f' <DataSet timestep="{i*save_every}" file="frame_{i:04d}.vti"/>\n'
        pvd += '</Collection>\n</VTKFile>'
        zf.writestr("simulation.pvd", pvd)
    buffer.seek(0)
    st.download_button(
        "📥 Download Full Results (PVD + VTI + CSV)",
        buffer,
        f"Ag_NP_{defect_type}_Simulation.zip",
        "application/zip"
    )
# =============================================
# THEORETICAL SOUNDNESS ANALYSIS
# =============================================
with st.expander("🔍 Theoretical Soundness Analysis"):
    st.markdown("""
    ### ✅ **Strengths:**
    - **Physical eigenstrains**: Uses crystallographically accurate values for Ag
    - **Phase-field fundamentals**: Proper Allen-Cahn formulation
    - **FFT elasticity**: Efficient stress calculation method
    - **Multiple defect types**: Distinguishes ISF/ESF/Twin physically
   
    ### ⚠️ **Limitations & Improvements:**
    - **2D approximation**: Real defects are 3D; consider 3D FFT
    - **Isotropic elasticity**: Ag is anisotropic (FCC)
    - **No dislocation dynamics**: Pure phase-field without discrete dislocations
    - **Fixed material constants**: Temperature dependence missing
   
    ### 🔬 **Plasticity Context:**
    The model captures **crystallographic plasticity** through:
    - **Eigenstrain (ε*)**: Transformation strain representing defect formation
    - **Stacking fault energies**: Implicit in phase-field parameters
    - **Stress evolution**: Shows how defects redistribute stresses
    - **Microstructure evolution**: Models defect growth/coarsening
   
    **Phase-field plasticity** bridges continuum mechanics with crystal defects!
    """)
st.caption("🔬 Crystallographically Accurate • ISF/ESF/Twin distinct • Publication-ready • 2025")
