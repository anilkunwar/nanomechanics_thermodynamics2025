# =============================================
# 3D Ag Nanoparticle Phase-Field + FFT – PERFECT PVD/VTI EXPORT
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
Interactive 3D Plotly • Perfect ParaView export (PVD + VTI + CSV)
""")

# =============================================
# Parameters
# =============================================
N = 64                              # Grid resolution (64³ = fast, looks good)
dx = 0.25                           # nm per voxel
dt = 0.005
kappa = 0.6
M = 1.0
C44 = 46.1                          # GPa
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0, 1.414, 0.01)  # ESF default
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
# 3D FFT Stress Magnitude (simplified but accurate)
# =============================================
@st.cache_data
def compute_stress_3d(eta, eps0):
    eps_star = eps0 * eta * 0.5
    eps_fft = np.fft.fftn(eps_star)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(N, d=dx),
                             np.fft.fftfreq(N, d=dx),
                             np.fft.fftfreq(N, d=dx), indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1e-12
    denom = 8 * C44**2 * k2**2
    ux_hat = -(kx*ky*eps_fft*2*C44) / denom
    uy_hat = -(ky*kx*eps_fft*2*C44) / denom
    uz_hat = np.zeros_like(eps_fft)
    ux = np.real(np.fft.ifftn(ux_hat))
    uy = np.real(np.fft.ifftn(uy_hat))
    uz = np.real(np.fft.ifftn(uz_hat))
    exx = np.gradient(ux, dx, axis=0)
    eyy = np.gradient(uy, dx, axis=1)
    ezz = np.gradient(uz, dx, axis=2)
    exy = 0.5 * (np.gradient(ux, dx, axis=1) + np.gradient(uy, dx, axis=0))
    sigma = 2 * C44 * np.sqrt(exx**2 + eyy**2 + ezz**2 + 2*exy**2)
    return np.nan_to_num(sigma)

# =============================================
# VTI Writer (Perfect ParaView Compatibility)
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
if st.button("Run 3D Evolution", type="primary"):
    with st.spinner("Running 3D phase-field + FFT elasticity..."):
        eta_current = eta.copy()
        history = []
        vti_list = []
        times = []

        for step in range(steps + 1):
            current_time = step * dt
            if step > 0:
                eta_current = evolve_3d(eta_current, kappa, dt, dx, N)
            if step % save_every == 0 or step == steps:
                sigma = compute_stress_3d(eta_current, eps0)
                history.append((eta_current.copy(), sigma.copy()))
                vti_content = create_vti(eta_current, sigma, step, current_time)
                vti_list.append(vti_content)
                times.append(current_time)
                st.write(f"Step {step}/{steps} – t = {current_time:.3f}")

        # Build correct PVD
        pvd = '<?xml version="1.0"?>\n'
        pvd += '<VTKFile type="Collection" version="1.0">\n'
        pvd += '  <Collection>\n'
        for i, t in enumerate(times):
            pvd += f'    <DataSet timestep="{t:.6f}" group="" part="0" file="frame_{i:04d}.vti"/>\n'
        pvd += '  </Collection>\n</VTKFile>'

        st.session_state.history_3d = history
        st.session_state.vti_3d = vti_list
        st.session_state.pvd_3d = pvd
        st.success(f"3D Simulation Complete! {len(history)} frames saved")

# =============================================
# 3D Interactive Visualization
# =============================================
if 'history_3d' in st.session_state:
    frame_idx = st.slider("Select Frame", 0, len(st.session_state.history_3d)-1, len(st.session_state.history_3d)-1)
    eta_3d, sigma_3d = st.session_state.history_3d[frame_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Defect Order Parameter η (Isosurface)")
        fig_eta = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=eta_3d.flatten(),
            isomin=0.3, isomax=0.9,
            surface_count=2,
            colorscale='Blues',
            opacity=0.7,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        fig_eta.update_layout(scene_aspectmode='data', height=600)
        st.plotly_chart(fig_eta, use_container_width=True)

    with col2:
        st.subheader("Stress Magnitude |σ|")
        fig_sig = go.Figure(data=go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=sigma_3d.flatten(),
            isomin=sigma_3d.max()*0.3,
            colorscale='Reds',
            opacity=0.7
        ))
        fig_sig.update_layout(scene_aspectmode='data', height=600)
        st.plotly_chart(fig_sig, use_container_width=True)

    # Mid-slice
    st.subheader("Mid-Plane Slice (z = center)")
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].imshow(eta_3d[:, :, N//2], cmap='viridis', extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax[0].set_title("η"); ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("y (nm)")
    im = ax[1].imshow(sigma_3d[:, :, N//2], cmap='hot', extent=[origin, origin+N*dx, origin, origin+N*dx])
    ax[1].set_title("|σ| (GPa)"); ax[1].set_xlabel("x (nm)")
    plt.colorbar(im, ax=ax[1])
    st.pyplot(fig)

    # =============================================
    # PERFECT DOWNLOAD: PVD + VTI + CSV
    # =============================================
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, s) in enumerate(st.session_state.history_3d):
            # CSV
            df = pd.DataFrame({
                'eta': e.flatten(order='F'),
                'stress': s.flatten(order='F')
            })
            zf.writestr(f"frame_{i:04d}.csv", df.to_csv(index=False))
            # VTI
            zf.writestr(f"frame_{i:04d}.vti", st.session_state.vti_3d[i])
        # PVD
        zf.writestr("simulation_3d.pvd", st.session_state.pvd_3d)

    buffer.seek(0)
    st.download_button(
        label="Download Full 3D Results (PVD + VTI + CSV)",
        data=buffer,
        file_name="Ag_Nanoparticle_3D_Defect_Simulation.zip",
        mime="application/zip"
    )

st.caption("3D Spherical Ag NP • Planar Defect • Perfect ParaView Export • Interactive Plotly • 2025")
