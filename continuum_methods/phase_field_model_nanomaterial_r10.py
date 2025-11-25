import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import zipfile

st.set_page_config(page_title="3D Ag NP Defect Analyzer", layout="wide")
st.title("3D Ag Nanoparticle Defect Mechanics")
st.markdown("**Phase-Field + FFT Elasticity • Full .pvd ParaView Export**")

# =============================================
# Sidebar
# =============================================
st.sidebar.header("Simulation Parameters")
c1, c2 = st.sidebar.columns(2)
with c1:
    N  = st.slider("Grid size N³", 32, 128, 64, 16)
    dx = st.slider("dx (nm)", 0.05, 0.2, 0.1, 0.01)
    dt = st.slider("Time step", 0.001, 0.01, 0.005, 0.001)
with c2:
    steps = st.slider("Steps", 10, 200, 80, 10)
    np_radius_ratio = st.slider("NP radius / box", 0.5, 0.95, 0.85, 0.05)
    defect_radius_ratio = st.slider("Initial defect / NP", 0.1, 0.5, 0.33, 0.05)

st.sidebar.header("Material")
defect_type = st.sidebar.selectbox("Defect", ["ISF", "ESF", "Twin", "Custom"])
if defect_type == "ISF":
    eps0, kappa, amp = 0.707, 0.60, 0.50
elif defect_type == "ESF":
    eps0, kappa, amp = 1.414, 0.70, 0.60
elif defect_type == "Twin":
    eps0, kappa, amp = 2.121, 0.30, 0.70
else:
    eps0 = st.sidebar.slider("ε*", 0.3, 3.0, 1.0, 0.01)
    kappa = st.sidebar.slider("κ", 0.1, 2.0, 0.5, 0.05)
    amp = st.sidebar.slider("Initial amplitude", 0.1, 1.0, 0.5, 0.05)

C44 = st.sidebar.slider("C₄₄ (GPa)", 10.0, 100.0, 46.1, 1.0)

st.sidebar.header("Visualization Settings")
viz_mode = st.sidebar.radio("Mode", ["2D Slices", "3D Isosurface", "Both"])
cmap_list = ['viridis', 'plasma', 'hot', 'jet', 'coolwarm', 'seismic', 'turbo']
eta_cmap = st.sidebar.selectbox("η Colormap", cmap_list, index=0)
stress_cmap = st.sidebar.selectbox("Stress Colormap", cmap_list, index=2)

if viz_mode in ["3D Isosurface", "Both"]:
    iso_level_eta = st.sidebar.slider("η Isosurface", 0.1, 0.9, 0.4, 0.05)
    iso_level_stress = st.sidebar.slider("Stress Iso Level", 0.1, 1.5, 0.6, 0.05)
    show_stress_iso = st.sidebar.checkbox("Show Stress Isosurface", True)

if viz_mode in ["2D Slices", "Both"]:
    slice_coord = st.sidebar.slider("Slice Position", 0, N-1, N//2)
    slice_axis = st.sidebar.selectbox("Slice Axis", ["X", "Y", "Z"])

# =============================================
# Core Functions
# =============================================
@jit(nopython=True, parallel=True)
def evolve_3d(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    inv_dx2 = 1.0 / (dx * dx)

    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6.0*eta[i,j,k]) * inv_dx2

                f = 2.0 * eta[i,j,k] * (1.0 - eta[i,j,k]) * (eta[i,j,k] - 0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt * (-f + kappa * lap)
                if eta_new[i,j,k] < 0.0:
                    eta_new[i,j,k] = 0.0
                elif eta_new[i,j,k] > 1.0:
                    eta_new[i,j,k] = 1.0

    # Explicit periodic boundaries
    eta_new[0, :, :]   = eta_new[N-2, :, :]
    eta_new[N-1, :, :] = eta_new[1, :, :]
    eta_new[:, 0, :]   = eta_new[:, N-2, :]
    eta_new[:, N-1, :] = eta_new[:, 1, :]
    eta_new[:, :, 0]   = eta_new[:, :, N-2]
    eta_new[:, :, N-1] = eta_new[:, :, 1]

    return eta_new

def spherical_mask(N, dx, ratio):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    return r <= N*dx/2 * ratio

def initial_defect(N, dx, mask, defect_ratio, amplitude):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    eta = np.zeros((N, N, N))
    eta[r < N*dx/2 * np_radius_ratio * defect_ratio] = amplitude
    eta += 0.01 * np.random.randn(N, N, N)
    eta = np.clip(eta, 0.0, 1.0)
    eta[~mask] = 0.0
    return eta

def stress_fft(eta, eps0, C44, N, dx):
    eps = eps0 * eta
    eps_k = np.fft.fftn(eps)
    k = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    k2[k2 == 0] = 1e-12
    strain_k = -eps_k / (2 * k2)
    strain = np.real(np.fft.ifftn(strain_k))
    return C44 * np.abs(strain)

# =============================================
# VTU & PVD
# =============================================
def write_vtu_ascii(eta, sigma, mask, N, dx, step):
    coords = np.linspace(-N*dx/2, N*dx/2, N, dtype=np.float32)
    def fmt(arr):
        return ' '.join(f"{float(x):.6f}" for x in arr.flatten(order='F'))

    points = '\n'.join(f"{coords[i]:.6f} {coords[j]:.6f} {coords[k]:.6f}"
                       for i in range(N) for j in range(N) for k in range(N))

    vtu = f"""<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0">
  <StructuredGrid WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}">
    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
      <PointData>
        <DataArray Name="eta"    type="Float32" format="ascii">{fmt(eta)}</DataArray>
        <DataArray Name="stress" type="Float32" format="ascii">{fmt(sigma)}</DataArray>
        <DataArray Name="NP"     type="Float32" format="ascii">{fmt(mask.astype(np.float32))}</DataArray>
      </PointData>
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {points}
        </DataArray>
      </Points>
    </Piece>
  </StructuredGrid>
</VTKFile>"""
    return vtu

def write_pvd(entries):
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="1.0">',
             '  <Collection>']
    for t, name in entries:
        lines.append(f'    <DataSet timestep="{t:.6f}" file="{name}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    return '\n'.join(lines)

# =============================================
# Simulation
# =============================================
if st.button("Run Simulation", type="primary"):
    with st.spinner("Running..."):
        mask = spherical_mask(N, dx, np_radius_ratio)
        eta  = initial_defect(N, dx, mask, defect_radius_ratio, amp)

        prog = st.progress(0)
        txt  = st.empty()
        history = []
        interval = max(1, steps // 40)  # ~40 frames

        for step in range(steps):
            eta = evolve_3d(eta, kappa, dt, dx, N)

            if step % interval == 0 or step == steps-1:
                sigma = stress_fft(eta, eps0, C44, N, dx)
                eta_masked = eta.copy()
                sigma_masked = sigma.copy()
                eta_masked[~mask] = 0.0
                sigma_masked[~mask] = 0.0
                time_val = step * dt
                history.append((eta_masked, sigma_masked, time_val))

            prog.progress((step+1)/steps)
            txt.text(f"Step {step+1}/{steps} – max η = {eta.max():.3f}")

        st.session_state.history = history
        st.session_state.mask    = mask
        st.success(f"Done! {len(history)} frames saved")

# =============================================
# Visualization & Download
# =============================================
if "history" in st.session_state:
    history = st.session_state.history
    mask    = st.session_state.mask

    frame = st.slider("Frame", 0, len(history)-1, len(history)-1)
    eta, sigma, t = history[frame]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max η", f"{eta.max():.3f}")
    col2.metric("Defect Vol", f"{np.sum(eta > 0.1):,}")
    col3.metric("Max stress", f"{sigma.max():.2f} GPa")
    col4.metric("Time", f"{t:.5f}")

    if viz_mode in ["2D Slices", "Both"]:
        st.subheader("2D Slice")
        if slice_axis == "X":
            eta_sl = eta[slice_coord, :, :]
            sigma_sl = sigma[slice_coord, :, :]
            xlab, ylab = "Y", "Z"
        elif slice_axis == "Y":
            eta_sl = eta[:, slice_coord, :]
            sigma_sl = sigma[:, slice_coord, :]
            xlab, ylab = "X", "Z"
        else:
            eta_sl = eta[:, :, slice_coord]
            sigma_sl = sigma[:, :, slice_coord]
            xlab, ylab = "X", "Y"

        extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(eta_sl.T, extent=extent, cmap=eta_cmap, origin='lower', vmin=0, vmax=1)
        ax1.set_title(f"η ({slice_axis}={slice_coord})")
        ax1.set_xlabel(f"{xlab} (nm)"); ax1.set_ylabel(f"{ylab} (nm)")
        plt.colorbar(ax1.images[0], ax=ax1, shrink=0.8)

        ax2.imshow(sigma_sl.T, extent=extent, cmap=stress_cmap, origin='lower')
        ax2.set_title("Stress Magnitude")
        ax2.set_xlabel(f"{xlab} (nm)"); ax2.set_ylabel(f"{ylab} (nm)")
        plt.colorbar(ax2.images[0], ax=ax2, shrink=0.8)
        st.pyplot(fig)

    if viz_mode in ["3D Isosurface", "Both"]:
        st.subheader("3D View")
        x = np.linspace(-N*dx/2, N*dx/2, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

        fig = go.Figure()
        fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                                    value=mask.ravel(), isomin=0.5, opacity=0.1,
                                    colorscale="Blues", showscale=False, name="NP"))
        fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                                    value=eta.ravel(), isomin=iso_level_eta, opacity=0.8,
                                    colorscale=eta_cmap, name="Defect"))
        if show_stress_iso:
            fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                                        value=sigma.ravel(), isomin=iso_level_stress * sigma.max(),
                                        opacity=0.6, colorscale=stress_cmap, name="Stress"))
        fig.update_layout(scene_aspectmode='data', height=700)
        st.plotly_chart(fig, use_container_width=True)

    # Download
    st.header("Download ParaView Dataset")
    if st.button("Create ZIP"):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            entries = []
            for i, (e, s, t) in enumerate(history):
                name = f"frame_{i:04d}.vtu"
                z.writestr(name, write_vtu_ascii(e, s, mask, N, dx, i))
                entries.append((t, name))
            z.writestr("simulation.pvd", write_pvd(entries))
            param_text = f"""Simulation Parameters:
Defect: {defect_type}
Grid: {N}^3, dx={dx} nm, dt={dt}
Steps: {steps}, Frames: {len(history)}
ε*={eps0}, κ={kappa}, C44={C44} GPa
"""
            z.writestr("params.txt", param_text)
        buf.seek(0)
        st.download_button(
            "Download .pvd + .vtu",
            buf,
            f"AgNP_{defect_type}_N{N}.zip",
            "application/zip"
        )
        st.info("Extract & open simulation.pvd in ParaView for animation!")

st.caption("Upgraded: Numba-fixed, Viz-enhanced, PVD-managed • 2025")
