import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import os
import tempfile

# =============================================
# Page Config
# =============================================
st.set_page_config(page_title="3D Ag NP Defect Analyzer", layout="wide")
st.title("3D Ag Nanoparticle Defect Mechanics")
st.markdown("""
**Phase-Field + FFT Elasticity • Spherical NPs • Full ParaView .pvd Export**
""")

# =============================================
# Sidebar Controls
# =============================================
st.sidebar.header("Simulation Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    N = st.slider("Grid Size (N³)", 32, 128, 64, 16, help="64–96 recommended")
    dx = st.slider("Grid Spacing (nm)", 0.05, 0.2, 0.1, 0.01)
    dt = st.slider("Time Step", 0.001, 0.01, 0.005, 0.001)
with col2:
    steps = st.slider("Evolution Steps", 10, 200, 80, 10)
    np_radius_ratio = st.slider("NP Radius Ratio", 0.5, 0.9, 0.8, 0.05)
    defect_radius_ratio = st.slider("Defect Radius Ratio", 0.1, 0.5, 0.33, 0.05)

st.sidebar.header("Material & Defect Properties")
defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin", "Custom"])
if defect_type == "ISF":
    eps0, kappa, init_amplitude = 0.707, 0.6, 0.5
elif defect_type == "ESF":
    eps0, kappa, init_amplitude = 1.414, 0.7, 0.6
elif defect_type == "Twin":
    eps0, kappa, init_amplitude = 2.121, 0.3, 0.7
else:
    eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0, 1.0, 0.01)
    kappa = st.sidebar.slider("Interface κ", 0.1, 2.0, 0.5, 0.05)
    init_amplitude = st.sidebar.slider("Initial Amplitude", 0.1, 1.0, 0.5, 0.1)

C44 = st.sidebar.slider("Shear Modulus C₄₄ (GPa)", 10.0, 100.0, 46.1, 1.0)

st.sidebar.header("Visualization")
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
# Core Simulation Functions
# =============================================
@jit(nopython=True, parallel=True)
def evolve_3d(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            for k in prange(1, N-1):
                lap = (eta[i+1,j,k] + eta[i-1,j,k] +
                       eta[i,j+1,k] + eta[i,j-1,k] +
                       eta[i,j,k+1] + eta[i,j,k-1] - 6*eta[i,j,k]) / dx2
                dF = 2*eta[i,j,k]*(1-eta[i,j,k])*(eta[i,j,k]-0.5)
                eta_new[i,j,k] = eta[i,j,k] + dt * (-dF + kappa * lap)
                eta_new[i,j,k] = max(0.0, min(1.0, eta_new[i,j,k]))
    # Periodic BC
    eta_new[[0, -1], :, :] = eta_new[[-2, 1], :, :]
    eta_new[:, [0, -1], :] = eta_new[:, [-2, 1], :]
    eta_new[:, :, [0, -1]] = eta_new[:, :, [-2, 1]]
    return eta_new

def compute_stress_3d(eta, eps0, C44, N, dx):
    eps_star = eps0 * eta
    eps_fft = np.fft.fftn(eps_star)
    k = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2 + 1e-12
    strain_fft = -eps_fft / (2 * k2)
    strain = np.real(np.fft.ifftn(strain_fft))
    return C44 * np.abs(strain)  # magnitude for visualization

def create_spherical_mask(N, dx, ratio):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    return np.sqrt(X**2 + Y**2 + Z**2) <= (N*dx/2 * ratio)

def create_initial_defect(N, dx, np_mask, defect_ratio, amp):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)
    defect_r = N*dx/2 * np_radius_ratio * defect_ratio
    eta = np.zeros_like(r)
    eta[r < defect_r] = amp
    eta += 0.01 * np.random.randn(N, N, N)
    eta = np.clip(eta, 0.0, 1.0)
    eta[~np_mask] = 0.0
    return eta

# =============================================
# VTU & PVD Export Functions (ASCII fallback – works everywhere)
# =============================================
def create_vtu_ascii(eta, sigma, np_mask, N, dx, step):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    points = [f"{x[i]:.6f} {x[j]:.6f} {x[k]:.6f}" 
              for i in range(N) for j in range(N) for k in range(N)]
    points_str = "\n".join(points)

    def flatten_fortran(arr):
        return ' '.join(f"{x:.6f}" for x in arr.flatten(order='F'))

    content = f'''<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="1.0" byte_order="LittleEndian">
  <StructuredGrid WholeExtent="0 {N-1} 0 {N-1} 0 {N-1}">
    <Piece Extent="0 {N-1} 0 {N-1} 0 {N-1}">
      <PointData>
        <DataArray type="Float32" Name="eta" format="ascii">
          {flatten_fortran(eta)}
        </DataArray>
        <DataArray type="Float32" Name="stress_magnitude" format="ascii">
          {flatten_fortran(sigma)}
        </DataArray>
        <DataArray type="Float32" Name="nanoparticle" format="ascii">
          {flatten_fortran(np_mask.astype(float))}
        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
    </Piece>
  </StructuredGrid>
</VTKFile>'''
    return content

def create_pvd_file(vtu_files_with_time):
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="1.0">',
             '  <Collection>']
    for t, filename in vtu_files_with_time:
        lines.append(f'    <DataSet timestep="{t:.6f}" file="{filename}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    return '\n'.join(lines)

# =============================================
# Main Simulation
# =============================================
if st.button("Run 3D Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        np_mask = create_spherical_mask(N, dx, np_radius_ratio)
        eta = create_initial_defect(N, dx, np_mask, defect_radius_ratio, init_amplitude)

        progress = st.progress(0)
        status = st.empty()
        history = []
        stress_compute_interval = max(1, steps // 50)  # at least 50 frames

        for step in range(steps):
            eta = evolve_3d(eta, kappa, dt, dx, N)

            if step % stress_compute_interval == 0 or step == steps - 1:
                sigma = compute_stress_3d(eta, eps0, C44, N, dx)
                eta[~np_mask] = 0.0
                sigma[~np_mask] = 0.0
                history.append((eta.copy(), sigma.copy(), step * dt * stress_compute_interval))

            progress.progress((step + 1) / steps)
            status.text(f"Step {step+1}/{steps} | Max η = {eta.max():.3f}")

        st.session_state.history = history
        st.session_state.np_mask = np_mask
        st.session_state.params = {
            "N": N, "dx": dx, "dt": dt, "steps": steps,
            "defect_type": defect_type, "eps0": eps0, "kappa": kappa,
            "C44": C44, "np_radius_ratio": np_radius_ratio,
            "defect_radius_ratio": defect_radius_ratio
        }
        st.success(f"Simulation complete! {len(history)} frames ready.")

# =============================================
# Visualization & Download
# =============================================
if 'history' in st.session_state:
    history = st.session_state.history
    np_mask = st.session_state.np_mask
    params = st.session_state.params

    frame_idx = st.slider("Frame", 0, len(history)-1, len(history)-1)
    eta, sigma, current_time = history[frame_idx]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max η", f"{eta.max():.3f}")
    col2.metric("Defect Volume", f"{(eta>0.3).sum():,} voxels")
    col3.metric("Max Stress", f"{sigma.max():.2f} GPa")
    col4.metric("Time", f"{current_time:.4f}")

    extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

    if viz_mode in ["2D Slices", "Both"]:
        st.subheader("2D Slice")
        axis_map = {"X": (eta[slice_coord,:,:], sigma[slice_coord,:,:], "Y", "Z"),
                    "Y": (eta[:,slice_coord,:], sigma[:,slice_coord,:], "X", "Z"),
                    "Z": (eta[:,:,slice_coord], sigma[:,:,slice_coord], "X", "Y")}
        eta_sl, sigma_sl, xlab, ylab = axis_map[slice_axis]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.imshow(eta_sl.T, extent=extent, cmap=eta_cmap, origin='lower', vmin=0, vmax=1)
        ax1.set_title(f"η — {slice_axis} slice at {slice_coord}")
        ax1.set_xlabel(xlab); ax1.set_ylabel(ylab)
        plt.colorbar(ax1.images[0], ax=ax1, label="η")

        im = ax2.imshow(sigma_sl.T, extent=extent, cmap=stress_cmap, origin='lower')
        ax2.set_title(f"Stress Magnitude — {slice_axis} slice")
        ax2.set_xlabel(xlab); ax2.set_ylabel(ylab)
        plt.colorbar(im, ax=ax2, label="Stress (GPa)")
        plt.tight_layout()
        st.pyplot(fig)

    if viz_mode in ["3D Isosurface", "Both"]:
        st.subheader("3D Interactive View")
        x = np.linspace(-N*dx/2, N*dx/2, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

        fig = go.Figure()
        # NP surface
        fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                    value=np_mask.flatten(),
                                    isomin=0.9, isomax=1.1,
                                    opacity=0.15, colorscale='Blues', showscale=False,
                                    name="Nanoparticle"))

        # Defect
        fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                    value=eta.flatten(),
                                    isomin=iso_level_eta, opacity=0.8,
                                    colorscale=eta_cmap, caps=dict(x_show=False, y_show=False, z_show=False),
                                    name="Defect"))

        if show_stress_iso:
            fig.add_trace(go.Isosurface(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                                        value=sigma.flatten(),
                                        isomin=iso_level_stress,
                                        opacity=0.6, colorscale=stress_cmap,
                                        name="High Stress"))

        fig.update_layout(scene_aspectmode='data', height=700)
        st.plotly_chart(fig, use_container_width=True)

    # =============================================
    # Professional ParaView Export (.pvd + .vtu)
    # =============================================
    st.header("Download Full ParaView Dataset")
    if st.button("Generate .pvd + .vtu Package"):
        with st.spinner("Creating ParaView files..."):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                vtu_list = []

                for idx, (e, s, t) in enumerate(history):
                    vtu_name = f"frame_{idx:04d}.vtu"
                    vtu_content = create_vtu_ascii(e, s, np_mask, N, dx, idx)
                    zf.writestr(vtu_name, vtu_content)
                    vtu_list.append((t, vtu_name))

                # Create master .pvd
                pvd_content = create_pvd_file(vtu_list)
                zf.writestr("simulation.pvd", pvd_content)

                # Parameters
                param_text = f"""3D Ag Nanoparticle Defect Simulation
Grid: {N}^3
dx = {dx} nm
Total steps: {steps}
Defect type: {defect_type}
Eigenstrain ε* = {eps0}
Interface energy κ = {kappa}
C44 = {C44} GPa
NP radius ratio = {np_radius_ratio}
Initial defect radius ratio = {defect_radius_ratio}
Generated with Streamlit + Numba + NumPy
"""
                zf.writestr("README.txt", param_text)

            buffer.seek(0)
            st.download_button(
                label="Download Complete ParaView Dataset (.pvd + .vtu)",
                data=buffer,
                file_name=f"AgNP_3D_{defect_type}_N{N}_ParaView.zip",
                mime="application/zip"
            )

        st.success("Package ready!")
        st.info("""
        **How to use in ParaView:**
        1. Extract the ZIP
        2. Open **simulation.pvd** → full timeline loads instantly
        3. Play animation, apply Threshold/Clip/Slice, color by any field
        4. Export movies, screenshots, or data
        """)

# =============================================
# Theory
# =============================================
with st.expander("Model Details & References"):
    st.markdown("""
    **Phase-field model** based on Allen-Cahn with eigenstrain-driven elasticity solved via FFT.
    - Spherical nanoparticle with free surfaces approximated via mask
    - Elastic solution valid for isotropic media (valid for FCC Ag in <100> orientation)
    - Stress = C₄₄ × |ε_el| (magnitude shown for clarity)
    """)

st.caption("Professional 3D Phase-Field Simulator • Full .pvd ParaView Export • 2025")
