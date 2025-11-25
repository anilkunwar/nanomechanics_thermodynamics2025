import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zipfile
from io import BytesIO

# =============================================
# Page Setup
# =============================================
st.set_page_config(page_title="3D Ag Nanoparticle Defect Analyzer", layout="wide")
st.title("3D Ag Nanoparticle Defect Mechanics")
st.markdown("""
**3D Phase-Field + FFT Elasticity • Spherical Nanoparticles • Full ParaView Export**
""")

# =============================================
# Sidebar Controls
# =============================================
st.sidebar.header("Simulation Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    N = st.slider("Grid Size N³", 32, 96, 64, 16)
    dx = st.slider("dx (nm)", 0.05, 0.25, 0.10, 0.01)
    dt = st.slider("dt", 0.001, 0.01, 0.005, 0.001)
with col2:
    total_steps = st.slider("Total Steps", 20, 200, 80, 10)
    save_every = st.slider("Save Every", 5, 30, 10, 5)
    np_radius_ratio = st.slider("NP Radius Ratio", 0.6, 0.95, 0.88, 0.02)
    defect_ratio = st.slider("Defect / NP Radius", 0.1, 0.6, 0.33, 0.05)

st.sidebar.header("Defect & Material")
defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin", "Custom"])
if defect_type == "ISF":
    eps0, kappa, init_eta = 0.707, 0.6, 0.6
elif defect_type == "ESF":
    eps0, kappa, init_eta = 1.414, 0.7, 0.65
elif defect_type == "Twin":
    eps0, kappa, init_eta = 2.121, 0.3, 0.85
else:
    eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.5, 1.0, 0.01)
    kappa = st.sidebar.slider("Interface κ", 0.1, 2.0, 0.5, 0.05)
    init_eta = st.sidebar.slider("Initial η", 0.3, 1.0, 0.7, 0.05)

C44 = st.sidebar.slider("C₄₄ (GPa)", 20.0, 80.0, 46.1, 1.0)

# Visualization
st.sidebar.header("Visualization")
viz_mode = st.sidebar.radio("View", ["2D Slices", "3D Interactive", "Both"])
cmap_eta = st.sidebar.selectbox("η cmap", ["viridis", "plasma", "turbo", "cividis"], index=0)
cmap_stress = st.sidebar.selectbox("Stress cmap", ["hot", "inferno", "magma"], index=0)

if "3D" in viz_mode:
    iso_eta = st.sidebar.slider("η Isosurface", 0.1, 0.9, 0.4, 0.05)
    iso_stress = st.sidebar.slider("Stress Isosurface (GPa)", 0.1, 6.0, 1.5, 0.1)

if "2D" in viz_mode:
    slice_axis = st.sidebar.selectbox("Slice Axis", ["X", "Y", "Z"])
    slice_pos = st.sidebar.slider("Slice Position", 0, N-1, N//2)

# =============================================
# Core Functions
# =============================================
@jit(nopython=True, parallel=True)
def evolve_3d(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    dx2 = dx**2
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6*eta[i,j,k]) / dx2
                chem = 2*eta[i,j,k]*(1-eta[i,j,k])*(eta[i,j,k]-0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt*(kappa*lap - chem)
                eta_new[i,j,k] = max(0.0, min(1.0, eta_new[i,j,k]))
    # Periodic BC
    eta_new[[0,-1],:,:] = eta_new[[-2,1],:,:]
    eta_new[:,[0,-1],:] = eta_new[:,[-2,1],:]
    eta_new[:,:,[0,-1]] = eta_new[:,:,[ -2,1]]
    return eta_new

def compute_stress_3d(eta, eps0, C44, N, dx):
    eps_star = eps0 * eta
    eps_hat = np.fft.fftn(eps_star)
    k = np.fft.fftfreq(N, d=dx) * 2j*np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2 + 1e-12
    strain_hat = -eps_hat / (2*k2)
    strain = np.real(np.fft.ifftn(strain_hat))
    return C44 * np.abs(strain)

# =============================================
# Geometry Setup (Cached)
# =============================================
@st.cache_data
def setup_simulation(N, dx, np_radius_ratio, defect_ratio, init_eta):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)

    np_mask = r <= (N*dx/2 * np_radius_ratio)
    defect_mask = r <= (N*dx/2 * np_radius_ratio * defect_ratio)

    eta = np.zeros((N,N,N))
    eta[defect_mask] = init_eta
    eta += 0.015 * np.random.randn(N,N,N)
    eta = np.clip(eta, 0.0, 1.0)
    eta[~np_mask] = 0.0

    return X, Y, Z, np_mask, eta

# =============================================
# Perfect VTU + PVD Writer
# =============================================
def write_vtu_frame(eta, sigma, X, Y, Z, step_idx):
    N = eta.shape[0]
    
    # Flatten data in Fortran order (required by VTK)
    def flatten(arr):
        return ' '.join(f"{float(x):.6f}" for x in arr.flatten(order='F'))

    # Coordinates
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    coord_str = ' '.join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in coords)

    vtu = f"""<?xml version="1.0" encoding="UTF-8"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">
  <StructuredGrid WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}">
    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
      <PointData>
        <DataArray type="Float32" Name="Order_Parameter_eta" NumberOfComponents="1" format="ascii">
          {flatten(eta)}
        </DataArray>
        <DataArray type="Float32" Name="Stress_Magnitude_GPa" NumberOfComponents="1" format="ascii">
          {flatten(sigma)}
        </DataArray>
        <DataArray type="Float32" Name="Strain_Energy_Density" NumberOfComponents="1" format="ascii">
          {flatten(sigma * eta)}  <!-- Approximate -->
        </DataArray>
        <DataArray type="UInt8" Name="Nanoparticle_Mask" NumberOfComponents="1" format="ascii">
          {flatten((X**2 + Y**2 + Z**2 <= (N*dx/2 * 0.88)**2).astype(np.uint8))}
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {coord_str}
        </DataArray>
      </Points>
    </Piece>
  </StructuredGrid>
</VTKFile>"""
    return vtu

# =============================================
# Run Simulation
# =============================================
if st.button("Run 3D Simulation", type="primary"):
    with st.spinner("Running 3D simulation..."):
        X, Y, Z, np_mask, eta = setup_simulation(N, dx, np_radius_ratio, defect_ratio, init_eta)

        progress = st.progress(0)
        status = st.empty()
        frames = []

        for step in range(total_steps + 1):
            if step > 0:
                eta = evolve_3d(eta, kappa, dt, dx, N)
                eta[~np_mask] = 0.0

            if step % save_every == 0 or step == total_steps:
                sigma = compute_stress_3d(eta, eps0, C44, N, dx)
                frames.append((eta.copy(), sigma.copy()))
                status.text(f"Frame {len(frames)} | η_max={eta.max():.3f} | σ_max={sigma.max():.2f} GPa")

            progress.progress(step / total_steps)

        st.session_state.frames = frames
        st.session_state.grid = (X, Y, Z)
        st.session_state.mask = np_mask
        st.success(f"Complete! {len(frames)} frames ready for download")

# =============================================
# Visualization & Download
# =============================================
if "frames" in st.session_state:
    frames = st.session_state.frames
    X, Y, Z = st.session_state.grid
    np_mask = st.session_state.mask

    frame_idx = st.slider("Frame", 0, len(frames)-1, len(frames)-1)
    eta, sigma = frames[frame_idx]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Max η", f"{eta.max():.3f}")
    col2.metric("Max Stress", f"{sigma.max():.2f} GPa")
    col3.metric("Defect Volume", f"{(eta>0.3).sum():,} voxels")

    # 2D Slice
    if "2D" in viz_mode:
        st.subheader("2D Slice")
        pos = slice_pos
        if slice_axis == "X": sl_e, sl_s = eta[pos], sigma[pos]
        elif slice_axis == "Y": sl_e, sl_s = eta[:,pos,:], sigma[:,pos,:]
        else: sl_e, sl_s = eta[:,:,pos], sigma[:,:,pos]

        fig, (a1, a2) = plt.subplots(1,2, figsize=(15,6))
        a1.imshow(sl_e.T, extent=[-N*dx/2,N*dx/2]*2, cmap=cmap_eta, origin='lower', vmin=0, vmax=1)
        a1.set_title("Order Parameter η")
        plt.colorbar(a1.images[0], ax=a1)
        a2.imshow(sl_s.T, extent=[-N*dx/2,N*dx/2]*2, cmap=cmap_stress, origin='lower')
        a2.set_title("Stress Magnitude (GPa)")
        plt.colorbar(a2.images[0], ax=a2)
        st.pyplot(fig)

    # 3D Plotly
    if "3D" in viz_mode:
        st.subheader("3D Interactive View")
        fig = go.Figure()
        fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                   value=np_mask.astype(float).flatten(),
                                   isomin=0.9, opacity=0.1, colorscale="Blues", showscale=False))
        fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                   value=eta.flatten(), isomin=iso_eta, opacity=0.9,
                                   colorscale=cmap_eta, colorbar=dict(title="η")))
        fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                   value=sigma.flatten(), isomin=iso_stress, opacity=0.7,
                                   colorscale=cmap_stress, colorbar=dict(title="Stress (GPa)")))
        fig.update_layout(scene_aspectmode='data', height=700)
        st.plotly_chart(fig, use_container_width=True)

    # FULL PARAVIEW EXPORT
    st.markdown("### Download Full ParaView Dataset")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, s) in enumerate(frames):
            vtu = write_vtu_frame(e, s, X, Y, Z, i)
            zf.writestr(f"frame_{i:04d}.vtu", vtu)

        # PVD Collection File
        pvd = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n<Collection>\n'
        for i in range(len(frames)):
            pvd += f'  <DataSet timestep="{i*save_every}" part="0" file="frame_{i:04d}.vtu"/>\n'
        pvd += '</Collection>\n</VTKFile>'
        zf.writestr("simulation.pvd", pvd)

        zf.writestr("README.txt",
f"""3D Ag Nanoparticle Defect Simulation
Defect Type: {defect_type}
Eigenstrain: {eps0:.3f}
Grid: {N}^3, dx = {dx:.3f} nm
Open simulation.pvd in ParaView and press Play!
""")

    buffer.seek(0)
    st.download_button(
        label="Download Complete ParaView Dataset (PVD + VTU)",
        data=buffer,
        file_name=f"3D_AgNP_{defect_type}_ParaView.zip",
        mime="application/zip"
    )

st.caption("3D simulation 2025")
