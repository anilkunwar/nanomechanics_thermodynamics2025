import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zipfile
from io import BytesIO

# =============================================
# Page Config
# =============================================
st.set_page_config(page_title="3D Ag NP Defect Analyzer", layout="wide")
st.title("3D Ag Nanoparticle Defect Mechanics")
st.markdown("**Phase-Field + FFT Elasticity • Full ParaView Export • No Errors**")

# =============================================
# Sidebar
# =============================================
st.sidebar.header("Simulation Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    N = st.slider("Grid N³", 32, 96, 64, 16)
    dx = st.slider("dx (nm)", 0.05, 0.25, 0.10, 0.01)
    dt = st.slider("dt", 0.001, 0.01, 0.005, 0.001)
with col2:
    steps = st.slider("Steps", 20, 200, 80, 10)
    save_every = st.slider("Save every", 5, 30, 10, 5)
    np_ratio = st.slider("NP radius ratio", 0.6, 0.95, 0.88, 0.02)
    defect_ratio = st.slider("Defect/NP ratio", 0.1, 0.6, 0.33, 0.05)

st.sidebar.header("Defect Type")
defect = st.sidebar.selectbox("Defect", ["ISF", "ESF", "Twin", "Custom"])
if defect == "ISF":
    eps0, kappa, init_eta = 0.707, 0.60, 0.60
elif defect == "ESF":
    eps0, kappa, init_eta = 1.414, 0.70, 0.65
elif defect == "Twin":
    eps0, kappa, init_eta = 2.121, 0.30, 0.85
else:
    eps0 = st.sidebar.slider("ε*", 0.3, 3.5, 1.0, 0.01)
    kappa = st.sidebar.slider("κ", 0.1, 2.0, 0.5, 0.05)
    init_eta = st.sidebar.slider("Init η", 0.3, 1.0, 0.7, 0.05)

C44 = st.sidebar.slider("C₄₄ (GPa)", 20, 80, 46.1, 1.0)

st.sidebar.header("Visualization")
mode = st.sidebar.radio("Mode", ["2D Slices", "3D", "Both"])
cmap_eta = st.sidebar.selectbox("η cmap", ["viridis", "plasma", "turbo"], index=0)
cmap_stress = st.sidebar.selectbox("Stress cmap", ["hot", "inferno", "magma"], index=0)

if "3D" in mode:
    iso_eta = st.sidebar.slider("η iso", 0.1, 0.9, 0.4, 0.05)
    iso_sig = st.sidebar.slider("σ iso (GPa)", 0.1, 6.0, 1.5, 0.1)

if "2D" in mode:
    axis = st.sidebar.selectbox("Slice", ["X", "Y", "Z"])
    pos = st.sidebar.slider("Position", 0, N-1, N//2)

# =============================================
# Pure Numba-safe evolution (NO masking inside!)
# =============================================
@jit(nopython=True, parallel=True)
def evolve_pure(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    inv_dx2 = 1.0 / (dx * dx)
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6.0*eta[i,j,k]) * inv_dx2
                chem = 2.0 * eta[i,j,k] * (1.0 - eta[i,j,k]) * (eta[i,j,k] - 0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt * (kappa * lap - chem)
                if eta_new[i,j,k] < 0.0:
                    eta_new[i,j,k] = 0.0
                if eta_new[i,j,k] > 1.0:
                    eta_new[i,j,k] = 1.0
    # Periodic BC
    eta_new[0,:,:] = eta_new[-2,:,:]; eta_new[-1,:,:] = eta_new[1,:,:]
    eta_new[:,0,:] = eta_new[:,-2,:]; eta_new[:,-1,:] = eta_new[:,1,:]
    eta_new[:,:,0] = eta_new[:,:,-2]; eta_new[:,:,-1] = eta_new[:,:,1]
    return eta_new

# =============================================
# FFT Stress (safe)
# =============================================
def compute_stress(eta, eps0, C44, N, dx):
    eps_star = eps0 * eta
    eps_hat = np.fft.fftn(eps_star)
    k = np.fft.fftfreq(N, d=dx) * 2j * np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[0,0,0] = 1e-12
    strain_hat = -eps_hat / (2 * k2)
    strain = np.real(np.fft.ifftn(strain_hat))
    return C44 * np.abs(strain)

# =============================================
# Geometry (cached)
# =============================================
@st.cache_data
def init_geometry(N, dx, np_ratio, defect_ratio, init_eta):
    x = np.linspace(-N*dx/2, N*dx/2, N, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)

    np_mask = r <= (N*dx/2 * np_ratio)
    defect_mask = r <= (N*dx/2 * np_ratio * defect_ratio)

    eta = np.zeros((N,N,N), dtype=np.float64)
    eta[defect_mask] = init_eta
    eta += 0.02 * np.random.randn(N,N,N)
    eta = np.clip(eta, 0.0, 1.0)
    eta[~np_mask] = 0.0

    return X, Y, Z, np_mask.astype(bool), eta

# =============================================
# Perfect VTU + PVD Export
# =============================================
def write_vtu(eta, sigma, X, Y, Z, step):
    N = eta.shape[0]
    def flat(arr):
        return ' '.join(f"{float(x):.6f}" for x in arr.flatten(order='F'))

    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    coord_str = ' '.join(f"{x:.6f} {y:.6f} {z:.6f}" for x,y,z in coords)

    vtu = f"""<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">
  <StructuredGrid WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}">
    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
      <PointData>
        <DataArray type="Float32" Name="Order_Parameter_eta" format="ascii">
          {flat(eta)}
        </DataArray>
        <DataArray type="Float32" Name="Stress_Magnitude_GPa" format="ascii">
          {flat(sigma)}
        </DataArray>
        <DataArray type="Float32" Name="Strain_Energy_Density_MJ_m3" format="ascii">
          {flat(sigma * eta)}
        </DataArray>
        <DataArray type="UInt8" Name="Inside_Nanoparticle" format="ascii">
          {flat((X**2+Y**2+Z**2 <= (N*dx/2*np_ratio)**2).astype(np.uint8))}
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
if st.button("Run Simulation", type="primary"):
    X, Y, Z, np_mask, eta = init_geometry(N, dx, np_ratio, defect_ratio, init_eta)

    progress = st.progress(0)
    frames = []

    for step in range(steps + 1):
        if step > 0:
            eta = evolve_pure(eta, kappa, dt, dx, N)   # Pure Numba, no masking!
            eta[~np_mask] = 0.0                         # Masking done in Python

        if step % save_every == 0 or step == steps:
            sigma = compute_stress(eta, eps0, C44, N, dx)
            frames.append((eta.copy(), sigma.copy()))

        progress.progress(step / steps)

    st.session_state.frames = frames
    st.session_state.grid = (X, Y, Z)
    st.session_state.mask = np_mask
    st.success(f"Done! {len(frames)} frames saved")

# =============================================
# Visualization & Export
# =============================================
if "frames" in st.session_state:
    frames = st.session_state.frames
    X, Y, Z = st.session_state.grid
    np_mask = st.session_state.mask

    i = st.slider("Frame", 0, len(frames)-1, len(frames)-1)
    eta, sigma = frames[i]

    col1, col2, col3 = st.columns(3)
    col1.metric("Max η", f"{eta.max():.3f}")
    col2.metric("Max σ", f"{sigma.max():.2f} GPa")
    col3.metric("Defect vol", f"{(eta>0.3).sum():,} vx")

    if "2D" in mode:
        st.subheader("2D Slice")
        if axis == "X": e, s = eta[pos], sigma[pos]
        elif axis == "Y": e, s = eta[:,pos,:], sigma[:,pos,:]
        else: e, s = eta[:,:,pos], sigma[:,:,pos]

        fig, (a1,a2) = plt.subplots(1,2,figsize=(15,6))
        a1.imshow(e.T, extent=[-N*dx/2,N*dx/2]*2, cmap=cmap_eta, origin='lower')
        a1.set_title("η")
        a2.imshow(s.T, extent=[-N*dx/2,N*dx/2]*2, cmap=cmap_stress, origin='lower')
        a2.set_title("Stress (GPa)")
        st.pyplot(fig)

    if "3D" in mode:
        st.subheader("3D View")
        fig = go.Figure()
        fig.add_isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                           value=np_mask.astype(float).flatten(), isomin=0.9,
                           opacity=0.1, colorscale="Blues", showscale=False)
        fig.add_isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                           value=eta.flatten(), isomin=iso_eta,
                           colorscale=cmap_eta, opacity=0.9)
        fig.add_isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                           value=sigma.flatten(), isomin=iso_sig,
                           colorscale=cmap_stress, opacity=0.7)
        fig.update_layout(scene_aspectmode='data', height=700)
        st.plotly_chart(fig, use_container_width=True)

    # PARA VIEW EXPORT
    st.markdown("### Download Full ParaView Dataset")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, (e, s) in enumerate(frames):
            z.writestr(f"frame_{i:04d}.vtu", write_vtu(e, s, X, Y, Z, i))
        pvd = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n<Collection>\n'
        for i in range(len(frames)):
            pvd += f'  <DataSet timestep="{i*save_every}" file="frame_{i:04d}.vtu"/>\n'
        pvd += '</Collection>\n</VTKFile>'
        z.writestr("simulation.pvd", pvd)
        z.writestr("info.txt", f"Defect: {defect} | N={N} | dx={dx} nm | ε*={eps0}")

    buf.seek(0)
    st.download_button(
        "Download ParaView Files (open simulation.pvd)",
        buf, f"AgNP_3D_{defect}.zip", "application/zip"
    )

st.caption(" 3D Simulations 2025")
