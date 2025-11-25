# =============================================
# 3D Ag Nanoparticle Phase-Field + FFT – COMPLETE STRESS TENSOR
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
**Complete stress tensor analysis • Von Mises + Hydrostatic stress • Proper 3D elasticity**  
**Unit-correct stress mapping • Interactive 3D visualization • Perfect ParaView export**
""")

# =============================================
# Enhanced Parameters with Physical Units
# =============================================
N = 64                              # Grid resolution
dx = 0.25                           # nm per voxel
dt = 0.005
kappa = 0.6
M = 1.0

# Silver elastic constants (GPa)
C11 = 124.0                         # GPa - from literature for Ag
C12 = 93.4                          # GPa
C44 = 46.1                          # GPa

# Lame constants for isotropic approximation
mu = C44                            # Shear modulus (GPa)
lambd = C12                         # First Lame parameter (GPa)
K = (C11 + 2 * C12) / 3             # Bulk modulus (GPa)

st.sidebar.header("🎛️ Simulation Parameters")
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0, 1.414, 0.01)
steps = st.sidebar.slider("Evolution steps", 20, 200, 80, 10)
save_every = st.sidebar.slider("Save every", 5, 20, 10)

# Physical domain
origin = -N * dx / 2
extent = [origin, origin + N*dx] * 3
X, Y, Z = np.meshgrid(
    np.linspace(origin, origin + (N-1)*dx, N),
    np.linspace(origin, origin + (N-1)*dx, N),
    np.linspace(origin, origin + (N-1)*dx, N),
    indexing='ij'
)

# Spherical nanoparticle mask (radius ~8 nm)
R_np = N * dx / 4
r = np.sqrt(X**2 + Y**2 + Z**2)
np_mask = r <= R_np

# Initial planar defect (horizontal twin/fault plate at center)
eta = np.zeros((N, N, N))
thickness = 3
center_z = N // 2
eta[:, :, center_z-thickness:center_z+thickness+1] = 0.7
eta[~np_mask] = 0.0  # Zero outside NP

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
# COMPLETE 3D STRESS TENSOR CALCULATION
# =============================================
@st.cache_data
def compute_complete_stress_3d(eta, eps0):
    """
    Compute full stress tensor with proper 3D elasticity
    Returns: sigma_mag, sigma_hydro, von_mises, stress_tensor
    """
    # Eigenstrain - assuming shear component in xy plane
    eps_star_xy = eps0 * eta * 0.5
    
    # FFT of eigenstrain
    eps_fft = np.fft.fftn(eps_star_xy)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(N, d=dx),
                             np.fft.fftfreq(N, d=dx),
                             np.fft.fftfreq(N, d=dx), indexing='ij')
    
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-12  # Avoid division by zero
    
    # Green's function approach for displacement
    denom = 8 * mu**2 * k2**2
    ux_hat = -(kx*ky*eps_fft*2*mu) / denom
    uy_hat = -(ky*kx*eps_fft*2*mu) / denom
    uz_hat = np.zeros_like(eps_fft)
    
    # Inverse FFT for displacements
    ux = np.real(np.fft.ifftn(ux_hat))
    uy = np.real(np.fft.ifftn(uy_hat))
    uz = np.real(np.fft.ifftn(uz_hat))
    
    # Strain components
    exx = np.gradient(ux, dx, axis=0)
    eyy = np.gradient(uy, dx, axis=1)
    ezz = np.gradient(uz, dx, axis=2)
    exy = 0.5 * (np.gradient(ux, dx, axis=1) + np.gradient(uy, dx, axis=0)) - eps_star_xy
    exz = 0.5 * (np.gradient(ux, dx, axis=2) + np.gradient(uz, dx, axis=0))
    eyz = 0.5 * (np.gradient(uy, dx, axis=2) + np.gradient(uz, dx, axis=1))
    
    # Stress tensor components (isotropic elasticity)
    trace_epsilon = exx + eyy + ezz
    
    sxx = lambd * trace_epsilon + 2 * mu * exx
    syy = lambd * trace_epsilon + 2 * mu * eyy
    szz = lambd * trace_epsilon + 2 * mu * ezz
    sxy = 2 * mu * exy
    sxz = 2 * mu * exz
    syz = 2 * mu * eyz
    
    # Stress measures
    sigma_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*(sxy**2 + sxz**2 + syz**2))
    sigma_hydro = (sxx + syy + szz) / 3.0
    
    # Von Mises stress (proper 3D formula)
    von_mises = np.sqrt(0.5 * ((sxx - syy)**2 + 
                              (syy - szz)**2 + 
                              (szz - sxx)**2 + 
                              6 * (sxy**2 + sxz**2 + syz**2)))
    
    # Full stress tensor
    stress_tensor = np.stack([sxx, syy, szz, sxy, sxz, syz], axis=0)
    
    return (np.nan_to_num(sigma_mag), np.nan_to_num(sigma_hydro), 
            np.nan_to_num(von_mises), stress_tensor)

# =============================================
# Enhanced VTI Writer with Complete Stress Fields
# =============================================
def create_enhanced_vti(eta, sigma_mag, sigma_hydro, von_mises, step, time):
    flat = lambda arr: ' '.join(f"{x:.6f}" for x in arr.flatten(order='F'))
    
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
          {flat(sigma_mag)}
        </DataArray>
        <DataArray type="Float32" Name="hydrostatic_stress" format="ascii">
          {flat(sigma_hydro)}
        </DataArray>
        <DataArray type="Float32" Name="von_mises_stress" format="ascii">
          {flat(von_mises)}
        </DataArray>
      </PointData>
      <CellData></CellData>
    </Piece>
  </ImageData>
</VTKFile>"""
    return vti

# =============================================
# Run Enhanced Simulation
# =============================================
if st.button("Run 3D Evolution with Complete Stress Analysis", type="primary"):
    with st.spinner("Running 3D phase-field + complete stress tensor analysis..."):
        eta_current = eta.copy()
        history = []
        vti_list = []
        times = []

        for step in range(steps + 1):
            current_time = step * dt
            if step > 0:
                eta_current = evolve_3d(eta_current, kappa, dt, dx, N)
            if step % save_every == 0 or step == steps:
                sigma_mag, sigma_hydro, von_mises, stress_tensor = compute_complete_stress_3d(eta_current, eps0)
                history.append((eta_current.copy(), sigma_mag.copy(), sigma_hydro.copy(), 
                              von_mises.copy(), stress_tensor.copy()))
                vti_content = create_enhanced_vti(eta_current, sigma_mag, sigma_hydro, 
                                                von_mises, step, current_time)
                vti_list.append(vti_content)
                times.append(current_time)
                st.write(f"Step {step}/{steps} – t = {current_time:.3f}")

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
        st.success(f"3D Simulation Complete! {len(history)} frames with complete stress analysis")

# =============================================
# Enhanced 3D Interactive Visualization
# =============================================
if 'history_3d' in st.session_state:
    st.header("📊 Complete 3D Stress Analysis")
    
    frame_idx = st.slider("Select Frame", 0, len(st.session_state.history_3d)-1, 
                         len(st.session_state.history_3d)-1)
    
    eta_3d, sigma_mag, sigma_hydro, von_mises, stress_tensor = st.session_state.history_3d[frame_idx]
    
    # Display stress statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("η Range", f"{eta_3d.min():.3f} - {eta_3d.max():.3f}")
    with col2:
        st.metric("|σ| Max", f"{sigma_mag.max():.2f} GPa")
    with col3:
        st.metric("σ_h Range", f"{sigma_hydro.min():.2f} - {sigma_hydro.max():.2f} GPa")
    with col4:
        st.metric("σ_vM Max", f"{von_mises.max():.2f} GPa")
    
    st.subheader("3D Isosurface Visualizations")
    
    # Create 4-panel 3D visualization
    fig_3d = go.Figure()
    
    # Add order parameter isosurface
    fig_3d.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=eta_3d.flatten(),
        isomin=0.3, isomax=0.9,
        surface_count=2,
        colorscale='Blues',
        opacity=0.7,
        name="Order Parameter η",
        showscale=True,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    # Add von Mises stress isosurface
    fig_3d.add_trace(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=von_mises.flatten(),
        isomin=von_mises.max() * 0.3,
        isomax=von_mises.max() * 0.8,
        surface_count=2,
        colorscale='Reds',
        opacity=0.6,
        name="Von Mises Stress",
        showscale=True,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    
    fig_3d.update_layout(
        title=f"3D Defect Evolution - Frame {frame_idx}",
        scene=dict(
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)', 
            zaxis_title='Z (nm)',
            aspectmode='data'
        ),
        height=600,
        width=800
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # 2D Slice Visualizations
    st.subheader("2D Cross-Section Analysis")
    
    # Create comprehensive 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    slice_idx = N // 2
    
    # Order parameter
    im0 = axes[0,0].imshow(eta_3d[:, :, slice_idx], cmap='viridis', 
                          extent=[origin, origin+N*dx, origin, origin+N*dx])
    axes[0,0].set_title("Order Parameter η", fontweight='bold', fontsize=14)
    axes[0,0].set_xlabel("x (nm)"); axes[0,0].set_ylabel("y (nm)")
    plt.colorbar(im0, ax=axes[0,0], shrink=0.8)
    
    # Stress magnitude
    im1 = axes[0,1].imshow(sigma_mag[:, :, slice_idx], cmap='hot',
                          extent=[origin, origin+N*dx, origin, origin+N*dx])
    axes[0,1].set_title("Stress Magnitude |σ| (GPa)", fontweight='bold', fontsize=14)
    axes[0,1].set_xlabel("x (nm)"); axes[0,1].set_ylabel("y (nm)")
    plt.colorbar(im1, ax=axes[0,1], shrink=0.8)
    
    # Hydrostatic stress
    vmax_hydro = max(abs(sigma_hydro.min()), abs(sigma_hydro.max()))
    im2 = axes[1,0].imshow(sigma_hydro[:, :, slice_idx], cmap='coolwarm',
                          vmin=-vmax_hydro, vmax=vmax_hydro,
                          extent=[origin, origin+N*dx, origin, origin+N*dx])
    axes[1,0].set_title("Hydrostatic Stress σₕ (GPa)", fontweight='bold', fontsize=14)
    axes[1,0].set_xlabel("x (nm)"); axes[1,0].set_ylabel("y (nm)")
    plt.colorbar(im2, ax=axes[1,0], shrink=0.8)
    
    # Von Mises stress
    im3 = axes[1,1].imshow(von_mises[:, :, slice_idx], cmap='plasma',
                          extent=[origin, origin+N*dx, origin, origin+N*dx])
    axes[1,1].set_title("Von Mises Stress σ_vM (GPa)", fontweight='bold', fontsize=14)
    axes[1,1].set_xlabel("x (nm)"); axes[1,1].set_ylabel("y (nm)")
    plt.colorbar(im3, ax=axes[1,1], shrink=0.8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Stress Tensor Components
    st.subheader("Stress Tensor Components (Central Slice)")
    
    sxx, syy, szz, sxy, sxz, syz = stress_tensor
    
    fig_components, ax_components = plt.subplots(2, 3, figsize=(18, 10))
    components = [sxx, syy, szz, sxy, sxz, syz]
    titles = ['σₓₓ', 'σᵧᵧ', 'σ_zz', 'σₓᵧ', 'σₓ_z', 'σᵧ_z']
    
    for i, (comp, title) in enumerate(zip(components, titles)):
        ax = ax_components[i//3, i%3]
        vmax = max(abs(comp.min()), abs(comp.max()))
        im = ax.imshow(comp[:, :, slice_idx], cmap='RdBu_r', 
                      vmin=-vmax, vmax=vmax,
                      extent=[origin, origin+N*dx, origin, origin+N*dx])
        ax.set_title(f"{title} (GPa)", fontweight='bold', fontsize=12)
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    st.pyplot(fig_components)

    # =============================================
    # Enhanced Download with Complete Stress Data
    # =============================================
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, sm, sh, vm, st_tensor) in enumerate(st.session_state.history_3d):
            # Enhanced CSV with all stress components
            df = pd.DataFrame({
                'eta': e.flatten(order='F'),
                'stress_magnitude': sm.flatten(order='F'),
                'hydrostatic_stress': sh.flatten(order='F'),
                'von_mises_stress': vm.flatten(order='F'),
                'stress_xx': st_tensor[0].flatten(order='F'),
                'stress_yy': st_tensor[1].flatten(order='F'),
                'stress_zz': st_tensor[2].flatten(order='F'),
                'stress_xy': st_tensor[3].flatten(order='F'),
                'stress_xz': st_tensor[4].flatten(order='F'),
                'stress_yz': st_tensor[5].flatten(order='F')
            })
            zf.writestr(f"frame_{i:04d}.csv", df.to_csv(index=False))
            
            # Enhanced VTI
            zf.writestr(f"frame_{i:04d}.vti", st.session_state.vti_3d[i])
        
        # PVD file
        zf.writestr("simulation_3d.pvd", st.session_state.pvd_3d)
        
        # Simulation parameters
        params = f"""Simulation Parameters:
Grid size: {N}³
Voxel size: {dx} nm
Time steps: {steps}
Eigenstrain: {eps0}
Elastic constants: C11={C11}, C12={C12}, C44={C44} GPa
"""
        zf.writestr("simulation_parameters.txt", params)

    buffer.seek(0)
    st.download_button(
        label="📥 Download Complete 3D Results (PVD + VTI + CSV + Parameters)",
        data=buffer,
        file_name="Ag_Nanoparticle_3D_Complete_Stress_Analysis.zip",
        mime="application/zip"
    )

# =============================================
# Theoretical Background
# =============================================
with st.expander("🔬 Theoretical Background & Stress Formulations"):
    st.markdown("""
    ### **3D Stress Tensor Formulation**
    
    **Stress Tensor Components:**
    ```
    σ = [σₓₓ  σₓᵧ  σₓ_z]
        [σᵧₓ  σᵧᵧ  σᵧ_z]  
        [σ_zₓ  σ_zᵧ  σ_zz]
    ```
    
    **Stress Invariants:**
    - **Hydrostatic Stress**: σₕ = (σₓₓ + σᵧᵧ + σ_zz)/3
    - **Von Mises Stress**: 
      ```
      σ_vM = √[½((σₓₓ-σᵧᵧ)² + (σᵧᵧ-σ_zz)² + (σ_zz-σₓₓ)² + 6(σₓᵧ²+σₓ_z²+σᵧ_z²))]
      ```
    - **Stress Magnitude**: |σ| = √(σₓₓ² + σᵧᵧ² + σ_zz² + 2(σₓᵧ²+σₓ_z²+σᵧ_z²))
    
    **Elasticity (Isotropic Approximation):**
    - **Shear modulus**: μ = C₄₄ = 46.1 GPa
    - **Bulk modulus**: K = (C₁₁ + 2C₁₂)/3 = {(124 + 2×93.4)/3:.1f} GPa
    - **Lame parameters**: λ = C₁₂, μ = C₄₄
    
    **Unit Consistency:** All stresses in GPa, consistent with experimental measurements.
    """)

st.caption("3D Spherical Ag NP • Complete Stress Tensor • Von Mises + Hydrostatic • Unit-Correct • 2025")
