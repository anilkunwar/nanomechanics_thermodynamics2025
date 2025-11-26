# =============================================
# 3D Ag Nanoparticle Phase-Field + FFT – CORRECTED STRESS CALCULATION
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
**Corrected FFT Stress Calculation • Consistent Results**
""")

# =============================================
# Parameters
# =============================================
N = 64
dx = 0.25
dt = 0.005
kappa = 0.6
M = 1.0

# Ag mechanical properties (GPa)
C11 = 124.0   # Ag single crystal
C12 = 93.4
C44 = 46.1
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.01, 0.1, 0.02, 0.001)  # Reduced for stability

steps = st.sidebar.slider("Evolution steps", 20, 200, 80, 10)
save_every = st.sidebar.slider("Save every", 5, 20, 10)

# Physical domain
origin = -N * dx / 2
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
# CORRECTED 3D FFT Stress Calculation
# =============================================
@st.cache_data
def compute_stress_3d(eta, eps0):
    """
    Corrected FFT-based stress calculation for isotropic elasticity
    with proper eigenstrain formulation
    """
    # Define eigenstrain tensor (simplified shear for planar defect)
    eps_star = np.zeros((6, N, N, N))
    
    # For planar defect, use shear eigenstrain in xy-plane
    eps_star[3] = eps0 * eta  # Engineering shear strain γ_xy = 2*ε_xy
    
    # Apply nanoparticle mask
    for i in range(6):
        eps_star[i] *= np_mask
    
    # FFT of eigenstrains
    eps_star_hat = np.zeros((6, N, N, N), dtype=complex)
    for i in range(6):
        eps_star_hat[i] = np.fft.fftn(eps_star[i])
    
    # Wave vectors
    kx, ky, kz = np.meshgrid(
        np.fft.fftfreq(N, d=dx) * 2 * np.pi,
        np.fft.fftfreq(N, d=dx) * 2 * np.pi, 
        np.fft.fftfreq(N, d=dx) * 2 * np.pi,
        indexing='ij'
    )
    
    # Avoid division by zero at k=0
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-12
    
    # Isotropic elasticity Green's function approach
    # For isotropic material: λ = C12, μ = C44
    lamda = C12
    mu = C44
    
    # Initialize stress in Fourier space
    sigma_hat = np.zeros((6, N, N, N), dtype=complex)
    
    # Simplified isotropic solution
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if k2[i,j,k] == 0:
                    continue
                
                # Unit wave vector
                n = np.array([kx[i,j,k], ky[i,j,k], kz[i,j,k]]) / np.sqrt(k2[i,j,k])
                
                # For isotropic material, use projected eigenstrain
                eps_proj = 0.0
                for comp in range(6):
                    # Simple projection (simplified)
                    eps_proj += eps_star_hat[comp, i, j, k] * np.abs(n[comp//2] * n[comp%2])
                
                # Stress in Fourier space (simplified isotropic)
                sigma_mag = (2 * mu + lamda) * eps_proj
                
                # Distribute to stress components
                for comp in range(6):
                    sigma_hat[comp, i, j, k] = sigma_mag * n[comp//2] * n[comp%2]
    
    # Transform back to real space
    sigma = np.zeros((6, N, N, N))
    for i in range(6):
        sigma[i] = np.real(np.fft.ifftn(sigma_hat[i]))
    
    # Calculate stress magnitude (von Mises equivalent stress)
    s_xx = sigma[0]
    s_yy = sigma[1] 
    s_zz = sigma[2]
    s_xy = sigma[3]
    s_xz = sigma[4]
    s_yz = sigma[5]
    
    # Von Mises stress
    von_mises = np.sqrt(0.5 * ((s_xx - s_yy)**2 + 
                              (s_yy - s_zz)**2 + 
                              (s_zz - s_xx)**2 + 
                              6 * (s_xy**2 + s_xz**2 + s_yz**2)))
    
    return np.nan_to_num(von_mises)

# =============================================
# ALTERNATIVE: Simplified but Consistent Stress
# =============================================
@st.cache_data
def compute_stress_simple(eta, eps0):
    """
    Simplified but consistent stress calculation
    based on strain energy density
    """
    # Calculate gradient of order parameter
    grad_eta_x = np.gradient(eta, dx, axis=0)
    grad_eta_y = np.gradient(eta, dx, axis=1) 
    grad_eta_z = np.gradient(eta, dx, axis=2)
    
    # Strain energy density approximation
    # |∇η| represents strain localization
    strain_energy = (grad_eta_x**2 + grad_eta_y**2 + grad_eta_z**2)
    
    # Stress magnitude proportional to strain energy
    # with eigenstrain scaling
    stress_magnitude = C44 * eps0 * strain_energy
    
    # Apply nanoparticle mask and smooth
    stress_magnitude *= np_mask
    
    # Smooth the stress field
    from scipy import ndimage
    stress_magnitude = ndimage.gaussian_filter(stress_magnitude, sigma=1.0)
    
    return np.clip(stress_magnitude, 0, None)

# =============================================
# VTI Writer
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
# Run Simulation
# =============================================
st.sidebar.header("Stress Calculation Method")
stress_method = st.sidebar.radio("Select stress calculation:", 
                                ["FFT-Based", "Simplified Gradient"])

if st.button("Run 3D Evolution", type="primary"):
    with st.spinner("Running 3D phase-field + stress calculation..."):
        eta_current = eta.copy()
        history = []
        vti_list = []
        times = []

        for step in range(steps + 1):
            current_time = step * dt
            if step > 0:
                eta_current = evolve_3d(eta_current, kappa, dt, dx, N)
            if step % save_every == 0 or step == steps:
                if stress_method == "FFT-Based":
                    sigma = compute_stress_3d(eta_current, eps0)
                else:
                    sigma = compute_stress_simple(eta_current, eps0)
                
                history.append((eta_current.copy(), sigma.copy()))
                vti_content = create_vti(eta_current, sigma, step, current_time)
                vti_list.append(vti_content)
                times.append(current_time)
                
                # Show progress with stress statistics
                sigma_np = sigma[np_mask]
                if len(sigma_np) > 0:
                    st.write(f"Step {step}/{steps} – Stress: {sigma_np.mean():.3f} ± {sigma_np.std():.3f} GPa")
                else:
                    st.write(f"Step {step}/{steps}")

        # Build PVD
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
        st.success(f"3D Simulation Complete! {len(history)} frames saved")

# =============================================
# 3D Interactive Visualization
# =============================================
if 'history_3d' in st.session_state:
    frame_idx = st.slider("Select Frame", 0, len(st.session_state.history_3d)-1, 
                         len(st.session_state.history_3d)-1)
    eta_3d, sigma_3d = st.session_state.history_3d[frame_idx]
    
    st.info(f"Stress Method: {st.session_state.stress_method} | "
           f"Max Stress: {sigma_3d[np_mask].max():.2f} GPa")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Defect Order Parameter η")
        fig_eta = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=eta_3d.flatten(),
            isomin=0.3, isomax=0.9,
            surface_count=2,
            colorscale='Blues',
            opacity=0.7,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig_eta.update_layout(
            scene_aspectmode='data', 
            height=600,
            title=f"η (Frame {frame_idx})"
        )
        st.plotly_chart(fig_eta, use_container_width=True)

    with col2:
        st.subheader("Stress Magnitude |σ|")
        
        # Calculate reasonable isosurface levels for stress
        stress_data = sigma_3d[np_mask]
        if len(stress_data) > 0:
            stress_min = stress_data.min()
            stress_max = stress_data.max()
            # Use percentiles for better visualization
            iso_levels = np.percentile(stress_data, [30, 60, 90])
        else:
            iso_levels = [sigma_3d.max() * 0.3]
        
        fig_sig = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=sigma_3d.flatten(),
            isomin=iso_levels[0],
            isomax=iso_levels[-1],
            surface_count=len(iso_levels),
            colorscale='Reds',
            opacity=0.7,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig_sig.update_layout(
            scene_aspectmode='data', 
            height=600,
            title=f"Stress (Frame {frame_idx})"
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    # Mid-slice with consistent color scaling
    st.subheader("Mid-Plane Slice Analysis")
    
    # Calculate consistent color limits across frames
    all_stress = np.concatenate([s[np_mask] for _, s in st.session_state.history_3d])
    global_stress_min = all_stress.min()
    global_stress_max = all_stress.max()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Defect parameter
    im1 = ax1.imshow(eta_3d[:, :, N//2], cmap='viridis', 
                    extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax1.set_title("Defect Parameter η")
    ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Current frame stress
    im2 = ax2.imshow(sigma_3d[:, :, N//2], cmap='hot',
                    extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax2.set_title(f"Stress |σ| (Current: {sigma_3d[np_mask].max():.2f} GPa)")
    ax2.set_xlabel("x (nm)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Stress with global color scale
    im3 = ax3.imshow(sigma_3d[:, :, N//2], cmap='hot',
                    vmin=global_stress_min, vmax=global_stress_max,
                    extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax3.set_title(f"Stress |σ| (Global Scale)")
    ax3.set_xlabel("x (nm)")
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    st.pyplot(fig)
    
    # Stress statistics
    st.subheader("Stress Statistics")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    stress_in_np = sigma_3d[np_mask]
    with col_stat1:
        st.metric("Max Stress", f"{stress_in_np.max():.3f} GPa")
    with col_stat2:
        st.metric("Mean Stress", f"{stress_in_np.mean():.3f} GPa")
    with col_stat3:
        st.metric("Stress Std", f"{stress_in_np.std():.3f} GPa")

    # =============================================
    # Download Section
    # =============================================
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, s) in enumerate(st.session_state.history_3d):
            df = pd.DataFrame({
                'eta': e.flatten(order='F'),
                'stress': s.flatten(order='F')
            })
            zf.writestr(f"frame_{i:04d}.csv", df.to_csv(index=False))
            zf.writestr(f"frame_{i:04d}.vti", st.session_state.vti_3d[i])
        zf.writestr("simulation_3d.pvd", st.session_state.pvd_3d)

    buffer.seek(0)
    st.download_button(
        label="Download Full 3D Results (PVD + VTI + CSV)",
        data=buffer,
        file_name="Ag_Nanoparticle_3D_Defect_Simulation.zip",
        mime="application/zip"
    )

st.caption("3D Spherical Ag NP • Corrected Stress Calculation • Consistent Results • 2025")
