# =============================================
# 3D Ag Nanoparticle Phase-Field + FFT – OPTIMIZED STRESS + ENHANCED VISUALIZATION
# =============================================
import streamlit as st
import numpy as np
from numba import jit, prange
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO

st.set_page_config(page_title="3D Ag NP Defect Evolution", layout="wide")
st.title("3D Phase-Field Simulation of Defects in Spherical Ag Nanoparticles")
st.markdown("""
**Realistic spherical nanoparticle • Internal planar defect (ISF/Twin)**  
**Optimized FFT Stress • Enhanced Color Maps • Consistent Results**
""")

# =============================================
# Enhanced Color Map Parameters
# =============================================
COLOR_MAPS = {
    'Matplotlib Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                           'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
                           'gray', 'bone', 'pink', 'copper', 'wistia'],
    
    'Diverging': ['RdBu', 'RdYlBu', 'RdYlGn', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                 'Spectral', 'coolwarm', 'bwr', 'seismic'],
    
    'Sequential': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
}

# =============================================
# Simulation Parameters with Enhanced Controls
# =============================================
st.sidebar.header("Simulation Parameters")
N = 64
dx = 0.25
dt = 0.005
kappa = 0.6
M = 1.0
C44 = 46.1

eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0, 1.414, 0.01)
steps = st.sidebar.slider("Evolution steps", 20, 200, 80, 10)
save_every = st.sidebar.slider("Save every", 5, 20, 10)

# =============================================
# Stress Calculation Method Selection
# =============================================
st.sidebar.header("Stress Calculation Method")
stress_method = st.sidebar.radio(
    "Select stress calculation approach:",
    ["Consistent FFT (Stress-focused)", "Classical FFT (Displacement-based)"],
    help="Consistent: Better stress localization at defects. Classical: More standard elasticity formulation."
)

# =============================================
# Enhanced Visualization Controls
# =============================================
st.sidebar.header("Visualization Controls")

# Color map selection - using only Matplotlib compatible names
viz_category = st.sidebar.selectbox("Color Map Category", list(COLOR_MAPS.keys()))
eta_cmap = st.sidebar.selectbox("Defect (η) Color Map", COLOR_MAPS[viz_category], 
                               index=0)
stress_cmap = st.sidebar.selectbox("Stress (σ) Color Map", COLOR_MAPS[viz_category], 
                                  index=min(1, len(COLOR_MAPS[viz_category])-1))

# Custom color scale limits
st.sidebar.subheader("Color Scale Limits")
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

# 3D visualization parameters
st.sidebar.subheader("3D Rendering")
opacity_3d = st.sidebar.slider("3D Opacity", 0.1, 1.0, 0.7, 0.1)
surface_count = st.sidebar.slider("Surface Count", 1, 10, 2)

# =============================================
# Physical Domain Setup
# =============================================
origin = -N * dx / 2
extent = [origin, origin + N*dx] * 3
X, Y, Z = np.meshgrid(
    np.linspace(origin, origin + (N-1)*dx, N),
    np.linspace(origin, origin + (N-1)*dx, N),
    np.linspace(origin, origin + (N-1)*dx, N),
    indexing='ij'
)

# Spherical nanoparticle mask
R_np = N * dx / 4
r = np.sqrt(X**2 + Y**2 + Z**2)
np_mask = r <= R_np

# Initial planar defect
eta = np.zeros((N, N, N))
thickness = 3
center_z = N // 2
eta[:, :, center_z-thickness:center_z+thickness+1] = 0.7
eta[~np_mask] = 0.0

# Add small noise inside NP only
np.random.seed(42)
eta += 0.02 * np.random.randn(N, N, N) * np_mask
eta = np.clip(eta, 0.0, 1.0)

# =============================================
# 3D Phase-Field Evolution (Numba)
# =============================================
@jit(nopython=True, parallel=True)
def evolve_3d(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    idx2 = 1.0 / (dx * dx)
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                if not np_mask[i,j,k]:
                    eta_new[i,j,k] = 0.0
                    continue
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6*eta[i,j,k]) * idx2
                dF = 2*eta[i,j,k]*(1-eta[i,j,k])*(eta[i,j,k]-0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt * M * (-dF + kappa * lap)
                eta_new[i,j,k] = max(0.0, min(1.0, eta_new[i,j,k]))
    return eta_new

# =============================================
# OPTIMIZED STRESS CALCULATION METHODS
# =============================================
@st.cache_data
def compute_stress_consistent(eta, eps0):
    """
    CONSISTENT FFT APPROACH (First Code)
    - Stress-focused formulation
    - Better stress localization at defect interfaces
    - Simplified but physically motivated projection
    """
    # Define eigenstrain for planar defect (shear component)
    eps_star = eps0 * eta * 0.5
    eps_fft = np.fft.fftn(eps_star)
    
    # Wave vectors with proper 2π scaling for Fourier space
    kx, ky, kz = np.meshgrid(
        np.fft.fftfreq(N, d=dx) * 2 * np.pi,
        np.fft.fftfreq(N, d=dx) * 2 * np.pi,
        np.fft.fftfreq(N, d=dx) * 2 * np.pi,
        indexing='ij'
    )
    
    # Avoid division by zero
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-12
    
    # Stress in Fourier space - simplified projection for defect interfaces
    sigma_hat = np.zeros_like(eps_fft)
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if k2[i,j,k] == 0:
                    continue
                
                # Unit wave vector
                n = np.array([kx[i,j,k], ky[i,j,k], kz[i,j,k]]) / np.sqrt(k2[i,j,k])
                
                # Stress projection emphasizing defect boundaries
                # This creates better stress concentration at interfaces
                sigma_hat[i,j,k] = (2 * C44 * eps_fft[i,j,k] * 
                                  (n[0]*n[1] + n[1]*n[0]))  # Shear stress component
    
    # Transform to real space and compute magnitude
    sigma_real = np.real(np.fft.ifftn(sigma_hat))
    stress_magnitude = np.abs(sigma_real) * C44
    
    # Apply mask and ensure physical values
    stress_magnitude = stress_magnitude * np_mask
    stress_magnitude = np.clip(stress_magnitude, 0, None)
    
    # Add baseline stress proportional to defect presence
    stress_magnitude += eps0 * eta * C44 * 0.1
    
    return stress_magnitude

@st.cache_data
def compute_stress_classical(eta, eps0):
    """
    CLASSICAL FFT APPROACH (Second Code)
    - Displacement/strain-based formulation
    - More standard elasticity calculation
    - Reconstructs full displacement fields
    """
    # Eigenstrain definition
    eps_star = eps0 * eta * 0.5
    eps_fft = np.fft.fftn(eps_star)
    
    # Standard FFT frequencies (no 2π scaling)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(N, d=dx),
                             np.fft.fftfreq(N, d=dx),
                             np.fft.fftfreq(N, d=dx), indexing='ij')
    
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-12
    
    # Classical approach: compute displacements first
    denom = 8 * C44**2 * k2**2
    ux_hat = -(kx*ky*eps_fft*2*C44) / denom
    uy_hat = -(ky*kx*eps_fft*2*C44) / denom
    uz_hat = np.zeros_like(eps_fft)
    
    # Inverse FFT to get displacement fields
    ux = np.real(np.fft.ifftn(ux_hat))
    uy = np.real(np.fft.ifftn(uy_hat))
    uz = np.real(np.fft.ifftn(uz_hat))
    
    # Compute strain components from displacements
    exx = np.gradient(ux, dx, axis=0)
    eyy = np.gradient(uy, dx, axis=1)
    ezz = np.gradient(uz, dx, axis=2)
    exy = 0.5 * (np.gradient(ux, dx, axis=1) + np.gradient(uy, dx, axis=0))
    
    # Stress magnitude from strains
    sigma = 2 * C44 * np.sqrt(exx**2 + eyy**2 + ezz**2 + 2*exy**2)
    
    # Apply mask
    sigma = sigma * np_mask
    
    return np.nan_to_num(sigma)

# =============================================
# Enhanced VTI Writer
# =============================================
def create_vti(eta, sigma, step, time):
    flat = lambda arr: ' '.join(map(str, arr.flatten(order='F')))
    vti = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}" 
             Origin="{origin:.3f} {origin:.3f} {origin:.3f}" 
             Spacing="{dx:.3f} {dx:.3f} {dx:.3f}">
    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
      <PointData Scalars="eta">
        <DataArray type="Float32" Name="eta" format="ascii">
          {flat(eta)}
        </DataArray>
        <DataArray type="Float32" Name="stress_magnitude" format="ascii">
          {flat(sigma)}
        </DataArray>
      </PointData>
      <CellData></CellData>
    </Piece>
  </ImageData>
</VTKFile>"""
    return vti

# =============================================
# Enhanced Visualization Functions
# =============================================
def create_plotly_isosurface(X, Y, Z, values, title, colorscale, 
                           isomin=None, isomax=None, opacity=0.7, 
                           surface_count=2, custom_min=None, custom_max=None):
    """Create enhanced Plotly isosurface with customizable color scales"""
    
    # Calculate automatic limits if not provided
    if isomin is None:
        isomin = np.percentile(values[np_mask], 10) if np.any(np_mask) else values.min()
    if isomax is None:
        isomax = np.percentile(values[np_mask], 90) if np.any(np_mask) else values.max()
    
    # Apply custom limits if specified
    if custom_min is not None and custom_max is not None:
        # Clip values for visualization
        values_clipped = np.clip(values, custom_min, custom_max)
        cmin, cmax = custom_min, custom_max
    else:
        values_clipped = values
        cmin, cmax = isomin, isomax
    
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(), 
        y=Y.flatten(), 
        z=Z.flatten(),
        value=values_clipped.flatten(),
        isomin=isomin,
        isomax=isomax,
        surface_count=surface_count,
        colorscale=colorscale,
        opacity=opacity,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title=title)
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)', 
            zaxis_title='Z (nm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        )
    )
    
    return fig

def safe_matplotlib_cmap(cmap_name, default='viridis'):
    """Safely get matplotlib colormap with fallback"""
    try:
        plt.get_cmap(cmap_name)
        return cmap_name
    except (ValueError, AttributeError):
        st.warning(f"Colormap '{cmap_name}' not found in Matplotlib. Using '{default}' instead.")
        return default

def create_matplotlib_comparison(eta_3d, sigma_3d, frame_idx, 
                               eta_cmap, stress_cmap, eta_lims, stress_lims):
    """Create comprehensive Matplotlib visualization with multiple color maps"""
    
    slice_pos = N // 2
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'3D Stress Visualization Comparison - Frame {frame_idx}\n'
                f'Matplotlib Color Maps (Slice z={slice_pos})', fontsize=16, y=0.95)
    
    # Define color limits with safety checks
    eta_data = eta_3d[np_mask]
    stress_data = sigma_3d[np_mask]
    
    if len(eta_data) > 0:
        eta_vmin, eta_vmax = (eta_lims if eta_lims else (eta_data.min(), eta_data.max()))
    else:
        eta_vmin, eta_vmax = 0.0, 1.0
        
    if len(stress_data) > 0:
        stress_vmin, stress_vmax = (stress_lims if stress_lims else (stress_data.min(), stress_data.max()))
    else:
        stress_vmin, stress_vmax = 0.0, 10.0
    
    # Use safe colormap names
    safe_eta_cmap = safe_matplotlib_cmap(eta_cmap, 'Blues')
    safe_stress_cmap = safe_matplotlib_cmap(stress_cmap, 'Reds')
    
    # Original visualization
    try:
        im1 = axes[0,0].imshow(eta_3d[:, :, slice_pos], 
                              cmap=safe_eta_cmap, vmin=eta_vmin, vmax=eta_vmax,
                              extent=[origin, origin+N*dx, origin, origin+N*dx])
        axes[0,0].set_title(f'Defect η ({safe_eta_cmap})')
        axes[0,0].set_xlabel('x (nm)'); axes[0,0].set_ylabel('y (nm)')
        plt.colorbar(im1, ax=axes[0,0], shrink=0.8)
    except Exception as e:
        axes[0,0].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('Defect η (Error)')
    
    try:
        im2 = axes[0,1].imshow(sigma_3d[:, :, slice_pos], 
                              cmap=safe_stress_cmap, vmin=stress_vmin, vmax=stress_vmax,
                              extent=[origin, origin+N*dx, origin, origin+N*dx])
        axes[0,1].set_title(f'Stress |σ| ({safe_stress_cmap})')
        axes[0,1].set_xlabel('x (nm)')
        plt.colorbar(im2, ax=axes[0,1], shrink=0.8)
    except Exception as e:
        axes[0,1].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Stress |σ| (Error)')
    
    # Method information
    axes[0,2].axis('off')
    info_text = f"""Stress Method: {stress_method}
Selected η: {eta_cmap}
Selected σ: {stress_cmap}
Custom Limits: {use_custom_limits}
Frame: {frame_idx}"""
    axes[0,2].text(0.1, 0.5, info_text, va='center', ha='left', fontsize=10)
    
    # Alternative color maps for comparison
    alt_cmaps = ['jet', 'viridis', 'plasma']
    alt_titles = ['Jet (Traditional)', 'Viridis (Perceptual)', 'Plasma (High Contrast)']
    
    for i, (cmap, title) in enumerate(zip(alt_cmaps, alt_titles)):
        try:
            im = axes[1,i].imshow(sigma_3d[:, :, slice_pos], 
                                 cmap=cmap, vmin=stress_vmin, vmax=stress_vmax,
                                 extent=[origin, origin+N*dx, origin, origin+N*dx])
            axes[1,i].set_title(f'Stress |σ| - {title}')
            axes[1,i].set_xlabel('x (nm)')
            if i == 0:
                axes[1,i].set_ylabel('y (nm)')
            plt.colorbar(im, ax=axes[1,i], shrink=0.8)
        except Exception as e:
            axes[1,i].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=axes[1,i].transAxes)
            axes[1,i].set_title(f'Stress |σ| - {title} (Error)')
    
    plt.tight_layout()
    return fig

# =============================================
# Run Enhanced Simulation
# =============================================
if st.button("Run 3D Evolution", type="primary"):
    with st.spinner(f"Running 3D phase-field + {stress_method} elasticity..."):
        eta_current = eta.copy()
        history = []
        vti_list = []
        times = []

        for step in range(steps + 1):
            current_time = step * dt
            if step > 0:
                eta_current = evolve_3d(eta_current, kappa, dt, dx, N)
            if step % save_every == 0 or step == steps:
                # Select stress calculation method
                if stress_method == "Consistent FFT (Stress-focused)":
                    sigma = compute_stress_consistent(eta_current, eps0)
                else:
                    sigma = compute_stress_classical(eta_current, eps0)
                    
                history.append((eta_current.copy(), sigma.copy()))
                vti_content = create_vti(eta_current, sigma, step, current_time)
                vti_list.append(vti_content)
                times.append(current_time)
                
                # Show progress with method-specific feedback
                sigma_np = sigma[np_mask]
                if len(sigma_np) > 0:
                    st.write(f"Step {step}/{steps} – {stress_method} – Max Stress: {sigma_np.max():.2f} GPa")
                else:
                    st.write(f"Step {step}/{steps} – {stress_method}")

        # Build PVD file
        pvd = '<?xml version="1.0"?>\n'
        pvd += '<VTKFile type="Collection" version="1.0">\n'
        pvd += '  <Collection>\n'
        for i, t in enumerate(times):
            pvd += f'    <DataSet timestep="{t:.6f}" group="" part="0" file="frame_{i:04d}.vti"/>\n'
        pvd += '  </Collection>\n</VTKFile>'

        st.session_state.history_3d = history
        st.session_state.vti_3d = vti_list
        st.session_state.pvd_3d = pvd
        st.session_state.stress_method = stress_method
        st.success(f"3D Simulation Complete! {len(history)} frames saved using {stress_method}")

# =============================================
# Enhanced Interactive Visualization
# =============================================
if 'history_3d' in st.session_state:
    frame_idx = st.slider("Select Frame", 0, len(st.session_state.history_3d)-1, 
                         len(st.session_state.history_3d)-1)
    eta_3d, sigma_3d = st.session_state.history_3d[frame_idx]
    
    # Prepare color limits
    eta_lims = (eta_min, eta_max) if use_custom_limits else None
    stress_lims = (stress_min, stress_max) if use_custom_limits else None
    
    st.header(f"3D Visualization - {st.session_state.stress_method}")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Defect Order Parameter η ({eta_cmap})")
        fig_eta = create_plotly_isosurface(
            X, Y, Z, eta_3d, "Defect Parameter η", 
            eta_cmap, isomin=0.3, isomax=0.9,
            opacity=opacity_3d, surface_count=surface_count,
            custom_min=eta_lims[0] if eta_lims else None,
            custom_max=eta_lims[1] if eta_lims else None
        )
        st.plotly_chart(fig_eta, use_container_width=True)

    with col2:
        st.subheader(f"Stress Magnitude |σ| ({stress_cmap})")
        # Calculate reasonable stress limits
        stress_data = sigma_3d[np_mask]
        if len(stress_data) > 0:
            stress_isomax = np.percentile(stress_data, 95)
        else:
            stress_isomax = sigma_3d.max()
            
        fig_sig = create_plotly_isosurface(
            X, Y, Z, sigma_3d, "Stress |σ| (GPa)", 
            stress_cmap, isomin=0.0, isomax=stress_isomax,
            opacity=opacity_3d, surface_count=surface_count,
            custom_min=stress_lims[0] if stress_lims else None,
            custom_max=stress_lims[1] if stress_lims else None
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    # Mid-slice comparison
    st.subheader("Mid-Plane Slice Analysis (z = center)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Defect parameter
    im1 = ax1.imshow(eta_3d[:, :, N//2], cmap='viridis', 
                    extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax1.set_title("Defect Parameter η")
    ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")
    plt.colorbar(im1, ax=ax1)
    
    # Stress magnitude
    im2 = ax2.imshow(sigma_3d[:, :, N//2], cmap='hot',
                    extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax2.set_title(f"Stress |σ| ({st.session_state.stress_method})")
    ax2.set_xlabel("x (nm)")
    plt.colorbar(im2, ax=ax2)
    
    st.pyplot(fig)
    
    # Enhanced Matplotlib comparison
    st.header("Matplotlib Color Map Comparison")
    st.markdown("""
    **Comparison of different color maps for stress visualization:**  
    - **Top row:** Selected color maps for defect and stress  
    - **Bottom row:** Popular alternatives (Jet, Viridis, Plasma) for comparison
    """)
    
    try:
        fig_mpl = create_matplotlib_comparison(
            eta_3d, sigma_3d, frame_idx, 
            eta_cmap, stress_cmap, eta_lims, stress_lims
        )
        st.pyplot(fig_mpl)
    except Exception as e:
        st.error(f"Error creating Matplotlib comparison: {str(e)}")
    
    # Stress statistics and method information
    with st.expander("Simulation Details & Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stress Statistics")
            stress_data = sigma_3d[np_mask]
            if len(stress_data) > 0:
                st.metric("Maximum Stress", f"{stress_data.max():.2f} GPa")
                st.metric("Average Stress", f"{stress_data.mean():.2f} GPa")
                st.metric("Stress Standard Deviation", f"{stress_data.std():.2f} GPa")
            else:
                st.write("No stress data available")
                
        with col2:
            st.subheader("Method Information")
            st.info(f"""
            **Current Method:** {st.session_state.stress_method}
            
            **Consistent FFT:**
            - Better stress localization at defect interfaces
            - Stress-focused formulation
            - Ideal for defect boundary analysis
            
            **Classical FFT:**
            - More standard elasticity approach
            - Displacement/strain-based calculation
            - Traditional formulation
            """)

    # =============================================
    # Enhanced Download Section
    # =============================================
    st.header("Data Export")
    
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, s) in enumerate(st.session_state.history_3d):
            # Enhanced CSV with metadata
            df = pd.DataFrame({
                'x': X.flatten(order='F'),
                'y': Y.flatten(order='F'), 
                'z': Z.flatten(order='F'),
                'eta': e.flatten(order='F'),
                'stress': s.flatten(order='F'),
                'in_nanoparticle': np_mask.flatten(order='F')
            })
            
            # Add simulation metadata
            metadata = f"""# 3D Ag Nanoparticle Simulation
# Frame: {i}
# Time: {times[i]:.3f}
# Parameters: eps0={eps0}, steps={steps}, dx={dx}
# Stress Method: {st.session_state.stress_method}
# Color Maps: eta={eta_cmap}, stress={stress_cmap}
"""
            csv_content = metadata + df.to_csv(index=False)
            zf.writestr(f"frame_{i:04d}.csv", csv_content)
            
            # VTI files
            zf.writestr(f"frame_{i:04d}.vti", st.session_state.vti_3d[i])
        
        # PVD file
        zf.writestr("simulation_3d.pvd", st.session_state.pvd_3d)
        
        # Simulation summary
        summary = f"""3D Ag Nanoparticle Defect Evolution Simulation
================================================
Total Frames: {len(st.session_state.history_3d)}
Simulation Steps: {steps}
Time Step: {dt}
Grid Resolution: {N}³
Eigenstrain (ε*): {eps0}
Stress Calculation: {st.session_state.stress_method}
Color Maps Used:
  - Defect (η): {eta_cmap}
  - Stress (σ): {stress_cmap}
Custom Color Limits: {use_custom_limits}
"""
        if use_custom_limits:
            summary += f"  η Limits: [{eta_min}, {eta_max}]\n"
            summary += f"  σ Limits: [{stress_min}, {stress_max}] GPa\n"
        
        zf.writestr("SIMULATION_SUMMARY.txt", summary)

    buffer.seek(0)
    st.download_button(
        label="Download Enhanced 3D Results (PVD + VTI + CSV + Metadata)",
        data=buffer,
        file_name="Ag_Nanoparticle_3D_Enhanced_Simulation.zip",
        mime="application/zip"
    )

st.caption("3D Spherical Ag NP • Optimized Stress Methods • Enhanced Color Maps • Multi-Platform Visualization • 2025")
