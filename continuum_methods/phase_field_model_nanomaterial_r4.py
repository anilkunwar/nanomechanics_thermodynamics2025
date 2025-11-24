# =============================================
# ULTIMATE Ag NP Defect Analyzer – FINAL VERSION
# =============================================
import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO

st.set_page_config(page_title="Ag NP Defect Analyzer – Ultimate", layout="wide")
st.title("Ag Nanoparticle Defect Mechanics – Ultimate Edition")
st.markdown("""
**Live phase-field + FFT elasticity**  
Four fields exported: **η | |σ| | von Mises | hydrostatic**  
Custom defect shape • Adjustable interface width • Publication-ready plots
""")

# =============================================
# Material & Grid
# =============================================
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
C44 = 46.1          # GPa
N = 128
dx = 0.1            # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# Sidebar Controls
# =============================================
st.sidebar.header("Defect & Physics")
defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])
shape = st.sidebar.selectbox("Initial Seed Shape", 
    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])

eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0,
                         value=0.706 if defect_type != "Twin" else 2.121, step=0.01)

kappa = st.sidebar.slider("Interface energy coeff κ (controls width)", 0.1, 2.0, 0.5, 0.05)
steps = st.sidebar.slider("Evolution steps", 20, 400, 150, 10)
save_every = st.sidebar.slider("Save frame every", 10, 50, 20)

# Colormaps
cmap_list = ['viridis', 'plasma', 'turbo', 'jet', 'rainbow', 'hot', 'coolwarm', 'RdBu_r', 'seismic', 'magma']
eta_cmap   = st.sidebar.selectbox("η colormap", cmap_list, index=0)
sigma_cmap = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index('hot'))
hydro_cmap = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap    = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index('plasma'))

# =============================================
# Create Initial Defect Seed (custom shapes)
# =============================================
def create_initial_eta(shape, eps0):
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    w, h = 24, 12 if shape in ["Rectangle", "Horizontal Fault"] else 16, 16

    if shape == "Square":
        eta[cy-h:h+cy+h, cx-h:cx+h] = 0.7
    elif shape == "Horizontal Fault":
        eta[cy-4:cy+4, cx-w:cx+w] = 0.8
    elif shape == "Vertical Fault":
        eta[cy-w:cy+w, cx-4:cx+4] = 0.8
    elif shape == "Rectangle":
        eta[cy-h:cy+h, cx-w:cx+w] = 0.7
    elif shape == "Ellipse":
        mask = ((X/(w*1.5))**2 + (Y/(h*1.5))**2) <= 1
        eta[mask] = 0.75

    # Add small random noise to help nucleation
    eta += 0.02 * np.random.randn(N, N)
    eta = np.clip(eta, 0, 1)
    return eta

# Show initial condition
st.subheader("Initial Defect Configuration")
init_eta = create_initial_eta(shape, eps0)
fig0, ax0 = plt.subplots(figsize=(6,5))
im0 = ax0.imshow(init_eta, extent=extent, cmap=eta_cmap, origin='lower')
ax0.contour(X, Y, init_eta, levels=[0.4], colors='white', linewidths=2)
ax0.set_title(f"Initial η – {defect_type} ({shape})", fontsize=16, fontweight='bold')
ax0.set_xlabel("x (nm)", fontsize=14); ax0.set_ylabel("y (nm)", fontsize=14)
plt.colorbar(im0, ax=ax0, shrink=0.8)
ax0.tick_params(labelsize=12)
for spine in ax0.spines.values():
    spine.set_linewidth(2)
st.pyplot(fig0)

# =============================================
# Numba Phase-Field Evolution
# =============================================
@jit(nopython=True, parallel=True)
def evolve_phase_field(eta_in, M, kappa, dt, dx, N):
    eta = eta_in.copy()
    eta_new = np.empty_like(eta)
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx2
            dF = 2*eta[i,j]*(1-eta[i,j])*(eta[i,j]-0.5)
            eta_new[i,j] = eta[i,j] + dt*M*(-dF + kappa*lap)
            eta_new[i,j] = max(0.0, min(1.0, eta_new[i,j]))
    # Periodic BC
    eta_new[[0,-1], :] = eta_new[[-2,1], :]
    eta_new[:, [0,-1]] = eta_new[:, [-2,1]]
    return eta_new

# =============================================
# FFT Elasticity (robust)
# =============================================
@st.cache_data
def compute_all_stress(eta, eps0):
    eps_xy = eps0 * eta * 0.5
    exy_hat = np.fft.fft2(eps_xy)
    kx, ky = np.meshgrid(np.fft.fftfreq(N, dx), np.fft.fftfreq(N, dx))
    k2 = kx**2 + ky**2 + 1e-12
    kx, ky = 2j*np.pi*kx, 2j*np.pi*ky

    denom = 4 * C44 * (2*C44) * k2**2
    ux_hat = -(kx*ky*exy_hat*2*C44) / denom
    uy_hat = -(ky*kx*exy_hat*2*C44) / denom

    ux = np.real(np.fft.ifft2(ux_hat))
    uy = np.real(np.fft.ifft2(uy_hat))

    exx = np.gradient(ux, dx, axis=1)
    eyy = np.gradient(uy, dx, axis=0)
    exy = 0.5*(np.gradient(ux, dx, axis=0) + np.gradient(uy, dx, axis=1))
    exy -= eps_xy

    lam = C44
    sxx = lam*(exx + eyy) + 2*C44*exx
    syy = lam*(exx + eyy) + 2*C44*eyy
    sxy = 2*C44*exy

    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy)/3
    dev_xx = sxx - sigma_hydro
    dev_yy = syy - sigma_hydro
    von_mises = np.sqrt(0.5*(dev_xx**2 + dev_yy**2 + (dev_xx + dev_yy)**2 + 6*sxy**2))

    return sigma_mag, sigma_hydro, von_mises

# =============================================
# Run Simulation
# =============================================
if st.button("Run Phase-Field Evolution", type="primary"):
    with st.spinner("Running high-performance simulation..."):
        eta = init_eta.copy()
        history = []

        for step in range(steps + 1):
            if step > 0:
                eta = evolve_phase_field(eta, M=1.0, kappa=kappa, dt=0.004, dx=dx, N=N)
            if step % save_every == 0 or step == steps:
                smag, shydro, vm = compute_all_stress(eta, eps0)
                history.append((eta.copy(), smag.copy(), shydro.copy(), vm.copy()))

        st.session_state.history = history
        st.success(f"Complete! {len(history)} frames saved")

# =============================================
# Live Results (4 plots, large fonts)
# =============================================
if 'history' in st.session_state:
    frame = st.slider("Frame", 0, len(st.session_state.history)-1, len(st.session_state.history)-1)
    eta, sigma_mag, sigma_hydro, von_mises = st.session_state.history[frame]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.linspace(extent[0], extent[1], N)
    y = np.linspace(extent[2], extent[3], N)

    ims = [
        axes[0,0].imshow(eta, cmap=eta_cmap, extent=extent, origin='lower', interpolation='bilinear'),
        axes[0,1].imshow(sigma_mag, cmap=sigma_cmap, extent=extent, origin='lower'),
        axes[1,0].imshow(sigma_hydro, cmap=hydro_cmap, extent=extent, origin='lower'),
        axes[1,1].imshow(von_mises, cmap=vm_cmap, extent=extent, origin='lower')
    ]
    titles = [
        "Order Parameter η",
        f"|σ| Magnitude – {sigma_mag.max():.2f} GPa",
        f"Hydrostatic Stress – {sigma_hydro.mean():.2f} GPa",
        f"von Mises – {von_mises.max():.2f} GPa"
    ]

    for ax, im, title in zip(axes.flat, ims, titles):
        ax.contour(x, y, eta, levels=[0.4], colors='white', linewidths=2)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel("x (nm)", fontsize=14)
        ax.set_ylabel("y (nm)", fontsize=14)
        ax.tick_params(labelsize=12, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    plt.tight_layout()
    st.pyplot(fig)

    # =============================================
    # Export PVD + VTI (all 4 fields) + CSV
    # =============================================
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (e, sm, sh, vm) in enumerate(st.session_state.history):
            # CSV
            df = pd.DataFrame({
                'eta': e.flatten(order='F'),
                'stress_magnitude': sm.flatten(order='F'),
                'hydrostatic': sh.flatten(order='F'),
                'von_mises': vm.flatten(order='F')
            })
            zf.writestr(f"frame_{i:04d}.csv", df.to_csv(index=False))

            # VTI with all fields
            def flat(arr): return ' '.join(f"{x:.6f}" for x in arr.flatten(order='F'))
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

        # PVD
        pvd = '<VTKFile type="Collection" version="1.0">\n<Collection>\n'
        for i in range(len(st.session_state.history)):
            pvd += f'  <DataSet timestep="{i*save_every}" file="frame_{i:04d}.vti"/>\n'
        pvd += '</Collection>\n</VTKFile>'
        zf.writestr("simulation.pvd", pvd)

    buffer.seek(0)
    st.download_button(
        label="Download Full Results (PVD + VTI + CSV)",
        data=buffer,
        file_name="Ag_NP_Defect_Simulation_Ultimate.zip",
        mime="application/zip"
    )

st.caption("Ultimate Edition • All fields exported • Custom shapes • Adjustable κ • Publication-ready • 2025")
