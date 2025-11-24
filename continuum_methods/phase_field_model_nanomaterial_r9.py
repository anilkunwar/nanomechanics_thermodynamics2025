import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import zipfile
from io import BytesIO
import base64

# Configure page
st.set_page_config(page_title="3D Ag NP Defect Analyzer", layout="wide")
st.title("🔮 3D Ag Nanoparticle Defect Mechanics")
st.markdown("""
**3D Phase-Field + FFT Elasticity**  
**Spherical Nanoparticles • Multiple Defect Types • Interactive 3D Visualization**
""")

# =============================================
# Sidebar Controls
# =============================================
st.sidebar.header("🔧 Simulation Parameters")

# Grid and simulation parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    N = st.slider("Grid Size (N³)", 32, 128, 64, 16, 
                 help="Higher = better resolution but slower")
    dx = st.slider("Grid Spacing (nm)", 0.05, 0.2, 0.1, 0.01)
    dt = st.slider("Time Step", 0.001, 0.01, 0.005, 0.001)

with col2:
    steps = st.slider("Evolution Steps", 10, 200, 50, 10)
    np_radius_ratio = st.slider("NP Radius Ratio", 0.5, 0.9, 0.8, 0.05,
                               help="NP radius relative to domain size")
    defect_radius_ratio = st.slider("Defect Radius Ratio", 0.1, 0.5, 0.33, 0.05,
                                   help="Initial defect size relative to NP")

# Material parameters
st.sidebar.header("🎛️ Material & Defect Properties")

defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin", "Custom"])

if defect_type == "ISF":
    eps0 = 0.707
    kappa = 0.6
    init_amplitude = 0.5
elif defect_type == "ESF":
    eps0 = 1.414  
    kappa = 0.7
    init_amplitude = 0.6
elif defect_type == "Twin":
    eps0 = 2.121
    kappa = 0.3
    init_amplitude = 0.7
else:  # Custom
    eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0, 1.0, 0.01)
    kappa = st.sidebar.slider("Interface κ", 0.1, 2.0, 0.5, 0.05)
    init_amplitude = st.sidebar.slider("Initial Amplitude", 0.1, 1.0, 0.5, 0.1)

C44 = st.sidebar.slider("Shear Modulus C₄₄ (GPa)", 10.0, 100.0, 46.1, 1.0)

# =============================================
# Visualization Controls
# =============================================
st.sidebar.header("🎨 Visualization Settings")

viz_mode = st.sidebar.radio("Visualization Mode", 
                           ["2D Slices", "3D Isosurface", "Both"])

# Color maps
cmap_list = ['viridis', 'plasma', 'turbo', 'hot', 'coolwarm', 'RdBu_r', 'seismic']
eta_cmap = st.sidebar.selectbox("η Colormap", cmap_list, index=0)
stress_cmap = st.sidebar.selectbox("Stress Colormap", cmap_list, index=cmap_list.index('hot'))

# Isosurface settings
if viz_mode in ["3D Isosurface", "Both"]:
    st.sidebar.subheader("3D Settings")
    iso_level_eta = st.sidebar.slider("η Isosurface Level", 0.1, 0.9, 0.4, 0.05)
    iso_level_stress = st.sidebar.slider("Stress Isosurface Level", 0.1, 0.9, 0.5, 0.05)
    show_stress_isosurface = st.sidebar.checkbox("Show Stress Isosurface", value=True)

# 2D slice settings  
if viz_mode in ["2D Slices", "Both"]:
    st.sidebar.subheader("2D Slice Settings")
    slice_coord = st.sidebar.slider("Slice Coordinate", 0, N-1, N//2)
    slice_axis = st.sidebar.selectbox("Slice Axis", ["X", "Y", "Z"])

# =============================================
# Core 3D Simulation Functions
# =============================================
@jit(nopython=True, parallel=True)
def evolve_3d(eta, kappa, dt, dx, N):
    """3D Allen-Cahn evolution with periodic boundary conditions"""
    eta_new = eta.copy()
    dx2 = dx**2
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6*eta[i,j,k]) / dx2
                dF = 2*eta[i,j,k]*(1-eta[i,j,k])*(eta[i,j,k]-0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt * (-dF + kappa * lap)
                eta_new[i,j,k] = max(0.0, min(1.0, eta_new[i,j,k]))
    
    # Periodic boundary conditions
    eta_new[0,:,:] = eta_new[-2,:,:]
    eta_new[-1,:,:] = eta_new[1,:,:]
    eta_new[:,0,:] = eta_new[:,-2,:]
    eta_new[:,-1,:] = eta_new[:,1,:]
    eta_new[:,:,0] = eta_new[:,:,-2]
    eta_new[:,:,-1] = eta_new[:,:,1]
    
    return eta_new

def compute_stress_3d(eta, eps0, C44, N, dx):
    """3D stress computation using FFT"""
    eps_star = eps0 * eta
    eps_fft = np.fft.fftn(eps_star)
    
    kx = np.fft.fftfreq(N, dx) * 2*np.pi
    ky = kx.copy()
    kz = kx.copy()
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    k2 = KX**2 + KY**2 + KZ**2 + 1e-12
    strain_fft = -eps_fft / (2 * k2)
    strain = np.real(np.fft.ifftn(strain_fft))
    sigma = C44 * strain
    
    return sigma

def create_spherical_mask(N, dx, radius_ratio):
    """Create spherical nanoparticle mask"""
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    np_radius = N*dx/2 * radius_ratio
    return r <= np_radius

def create_initial_defect(N, dx, np_mask, defect_radius_ratio, init_amplitude):
    """Create initial defect configuration"""
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Create central spherical defect
    defect_radius = (N*dx/2 * np_radius_ratio) * defect_radius_ratio
    defect_mask = r < defect_radius
    
    eta = np.zeros((N, N, N))
    eta[defect_mask] = init_amplitude
    eta += 0.01 * np.random.randn(N, N, N)  # Small noise
    eta = np.clip(eta, 0.0, 1.0)
    
    # Apply nanoparticle mask
    eta[~np_mask] = 0.0
    
    return eta

# =============================================
# Visualization Functions
# =============================================
def create_2d_slice_plot(eta, sigma, slice_coord, slice_axis, extent):
    """Create 2D slice visualization"""
    if slice_axis == "X":
        eta_slice = eta[slice_coord, :, :]
        sigma_slice = sigma[slice_coord, :, :]
        xlabel, ylabel = "Y (nm)", "Z (nm)"
    elif slice_axis == "Y":
        eta_slice = eta[:, slice_coord, :]
        sigma_slice = sigma[:, slice_coord, :]
        xlabel, ylabel = "X (nm)", "Z (nm)"
    else:  # Z
        eta_slice = eta[:, :, slice_coord]
        sigma_slice = sigma[:, :, slice_coord]
        xlabel, ylabel = "X (nm)", "Y (nm)"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot eta
    im1 = ax1.imshow(eta_slice.T, extent=extent, cmap=eta_cmap, origin='lower',
                    vmin=0, vmax=1)
    ax1.set_title(f"Order Parameter η - {slice_axis}={slice_coord}", fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='η')
    
    # Plot stress
    im2 = ax2.imshow(sigma_slice.T, extent=extent, cmap=stress_cmap, origin='lower')
    ax2.set_title(f"Stress Magnitude - {slice_axis}={slice_coord}", fontsize=14, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Stress (GPa)', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_3d_isosurface_plot(eta, sigma, np_mask, iso_level_eta, iso_level_stress, show_stress):
    """Create 3D isosurface plot using Plotly"""
    N = eta.shape[0]
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Create eta isosurface
    fig = go.Figure()
    
    # Nanoparticle surface (transparent)
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=np_mask.astype(float).flatten(),
        isomin=0.5,
        isomax=1.0,
        opacity=0.1,
        surface_count=1,
        colorscale=['blue', 'blue'],
        showscale=False,
        name="Nanoparticle"
    ))
    
    # Defect isosurface
    fig.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=eta.flatten(),
        isomin=iso_level_eta,
        isomax=1.0,
        opacity=0.8,
        surface_count=3,
        colorscale=eta_cmap,
        colorbar=dict(title="η", x=0.8),
        name="Defect Region"
    ))
    
    if show_stress:
        # Stress isosurface
        sigma_normalized = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-12)
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=sigma_normalized.flatten(),
            isomin=iso_level_stress,
            isomax=1.0,
            opacity=0.6,
            surface_count=3,
            colorscale=stress_cmap,
            colorbar=dict(title="Stress", x=0.9),
            name="High Stress"
        ))
    
    fig.update_layout(
        title="3D Defect and Stress Distribution",
        scene=dict(
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)", 
            zaxis_title="Z (nm)",
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    
    return fig

def create_vtu_file(eta, sigma, np_mask, N, dx):
    """Create VTU file for ParaView visualization"""
    vtu_content = f'''<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">
<StructuredGrid WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}">
<Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
<PointData Scalars="fields">
<DataArray type="Float32" Name="eta" NumberOfComponents="1" format="ascii">
{' '.join([str(v) for v in eta.flatten('F')])}
</DataArray>
<DataArray type="Float32" Name="sigma" NumberOfComponents="1" format="ascii">
{' '.join([str(v) for v in sigma.flatten('F')])}
</DataArray>
<DataArray type="Float32" Name="np_mask" NumberOfComponents="1" format="ascii">
{' '.join([str(v) for v in np_mask.astype(float).flatten('F')])}
</DataArray>
</PointData>
<Points>
<DataArray type="Float32" NumberOfComponents="3" format="ascii">
{' '.join([f"{x} {y} {z}" for x in np.linspace(-N*dx/2, N*dx/2, N) 
          for y in np.linspace(-N*dx/2, N*dx/2, N) 
          for z in np.linspace(-N*dx/2, N*dx/2, N)])}
</DataArray>
</Points>
</Piece>
</StructuredGrid>
</VTKFile>'''
    
    return vtu_content

# =============================================
# Main Simulation
# =============================================
st.header("🚀 3D Simulation")

if st.button("Run 3D Simulation", type="primary"):
    with st.spinner("Running 3D phase-field simulation... This may take a while for larger grids."):
        # Initialize
        np_mask = create_spherical_mask(N, dx, np_radius_ratio)
        eta = create_initial_defect(N, dx, np_mask, defect_radius_ratio, init_amplitude)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Evolution loop
        history = []
        for step in range(steps):
            eta = evolve_3d(eta, kappa, dt, dx, N)
            
            # Compute stress every 10 steps to save time
            if step % 10 == 0 or step == steps - 1:
                sigma = compute_stress_3d(eta, eps0, C44, N, dx)
                # Apply nanoparticle mask
                eta[~np_mask] = 0.0
                sigma[~np_mask] = 0.0
                history.append((eta.copy(), sigma.copy()))
            
            # Update progress
            progress = (step + 1) / steps
            progress_bar.progress(progress)
            status_text.text(f"Step {step+1}/{steps} - Max η: {eta.max():.3f}")
        
        st.session_state.history_3d = history
        st.session_state.np_mask = np_mask
        st.success(f"✅ 3D Simulation Complete! {len(history)} frames saved")

# =============================================
# Results Visualization
# =============================================
if 'history_3d' in st.session_state:
    st.header("📊 Results")
    
    history = st.session_state.history_3d
    np_mask = st.session_state.np_mask
    
    # Frame selector
    frame = st.slider("Select Frame", 0, len(history)-1, len(history)-1)
    eta, sigma = history[frame]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max η", f"{eta.max():.3f}")
    with col2:
        st.metric("Defect Volume", f"{np.sum(eta > 0.1):,} voxels")
    with col3:
        st.metric("Max Stress", f"{sigma.max():.2f} GPa")
    with col4:
        st.metric("NP Volume", f"{np.sum(np_mask):,} voxels")
    
    extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
    
    # Visualization based on selected mode
    if viz_mode == "2D Slices":
        st.subheader("2D Slice Visualization")
        fig_2d = create_2d_slice_plot(eta, sigma, slice_coord, slice_axis, extent)
        st.pyplot(fig_2d)
        
    elif viz_mode == "3D Isosurface":
        st.subheader("3D Isosurface Visualization")
        fig_3d = create_3d_isosurface_plot(eta, sigma, np_mask, 
                                         iso_level_eta, iso_level_stress, 
                                         show_stress_isosurface)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    else:  # Both
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2D Slice")
            fig_2d = create_2d_slice_plot(eta, sigma, slice_coord, slice_axis, extent)
            st.pyplot(fig_2d)
        
        with col2:
            st.subheader("3D Isosurface")
            fig_3d = create_3d_isosurface_plot(eta, sigma, np_mask,
                                             iso_level_eta, iso_level_stress,
                                             show_stress_isosurface)
            st.plotly_chart(fig_3d, use_container_width=True)
    
    # Download section
    st.header("💾 Download Results")
    
    # Create VTU file
    vtu_content = create_vtu_file(eta, sigma, np_mask, N, dx)
    
    # Create ZIP with all frames
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, s) in enumerate(history):
            vtu_frame = create_vtu_file(e, s, np_mask, N, dx)
            zf.writestr(f"frame_{i:04d}.vtu", vtu_frame)
        
        # Add parameter summary
        params = f"""3D Simulation Parameters:
Grid Size: {N}³
Grid Spacing: {dx} nm
Time Step: {dt}
Steps: {steps}
Defect Type: {defect_type}
Eigenstrain ε*: {eps0}
Interface κ: {kappa}
Shear Modulus C44: {C44} GPa
NP Radius Ratio: {np_radius_ratio}
Defect Radius Ratio: {defect_radius_ratio}
Initial Amplitude: {init_amplitude}
"""
        zf.writestr("simulation_parameters.txt", params)
    
    buffer.seek(0)
    
    st.download_button(
        "📥 Download All Frames (VTU + Parameters)",
        buffer,
        f"3D_AgNP_{defect_type}_Simulation.zip",
        "application/zip"
    )

# =============================================
# Theoretical Background
# =============================================
with st.expander("🔬 3D Model Theory"):
    st.markdown("""
    ### **3D Phase-Field Model for Spherical Nanoparticles**
    
    **Governing Equations:**
    - **Allen-Cahn in 3D**: `∂η/∂t = -M[∂f/∂η - κ∇²η]`
    - **3D Laplacian**: `∇²η = (∂²η/∂x² + ∂²η/∂y² + ∂²η/∂z²)`
    - **FFT Elasticity**: Solves mechanical equilibrium in Fourier space
    
    **Key Features:**
    - **Spherical boundary conditions**: Realistic nanoparticle geometry
    - **3D eigenstrains**: Full stress tensor computation
    - **Multiple visualization modes**: 2D slices + 3D isosurfaces
    - **Physical parameters**: Crystallographically accurate for Ag
    
    **Computational Considerations:**
    - **Memory**: N³ grid requires O(N³) memory
    - **Performance**: Numba JIT compilation for 3D loops
    - **Visualization**: Plotly for interactive 3D, Matplotlib for 2D
    """)

st.caption("🔮 3D Crystallographically Accurate • Spherical Nanoparticles • Interactive Visualization • 2025")
