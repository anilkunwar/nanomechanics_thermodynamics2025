import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
st.set_page_config(page_title="2D vs 3D Stress: 3 Options", layout="wide")
st.title("2D vs 3D FFT Elasticity — Comparison Modes")
st.markdown("""
Choose how the 2D solution is formed from (or compared to) the full 3D solution:
- **Option 1**: Use the exact 3D central slice (identical).
- **Option 2**: Use the kz=0 (z-average) of the 3D solution — includes long-range 3D interactions but only kz=0.
- **Option 3**: Use the 2D plane-strain solver, then tune a scalar stiffness factor to best match the 3D central slice.
""")
# ------------------- Parameters -------------------
st.sidebar.header("Simulation Parameters")
N = st.sidebar.slider("Grid size N", 32, 96, 64, step=16)
dx_nm = st.sidebar.number_input("dx (nm)", 0.05, 1.0, 0.25, 0.05)
eps0 = st.sidebar.slider("Eigenstrain ε*", 0.01, 0.5, 0.1, 0.01)
st.sidebar.subheader("Habit Plane Orientation")
theta_deg = st.sidebar.slider("Polar angle θ (°)", 0, 180, 55, 1)
phi_deg = st.sidebar.slider("Azimuthal angle φ (°)", 0, 360, 0, 5)
theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)
# Material
mu = 46.1e9 # Pa
lam = 93.4e9 - 2*mu/3.0
nu = lam / (2*(lam + mu))
st.sidebar.write(f"Material: μ={mu/1e9:.2f} GPa, λ={lam/1e9:.2f} GPa, ν={nu:.3f}")
# Mode selection
mode = st.sidebar.selectbox("2D formation mode", [
    "Option 1 — 3D central slice (exact)",
    "Option 2 — 3D kz=0 average (z-mean)",
    "Option 3 — Tuned 2D plane-strain (scalar fit)"
])
# ------------------- Defect creation -------------------
@st.cache_data
def create_defect(N, dx, theta, phi):
    x = np.linspace(-N*dx/2, N*dx/2, N)
    y = np.linspace(-N*dx/2, N*dx/2, N)
    z = np.linspace(-N*dx/2, N*dx/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    Rnp = N * dx / 4
    sphere = X**2 + Y**2 + Z**2 <= Rnp**2
    nx = np.cos(phi) * np.sin(theta)
    ny = np.sin(phi) * np.sin(theta)
    nz = np.cos(theta)
    dist = nx*X + ny*Y + nz*Z
    thickness = 3 * dx
    plane = np.abs(dist) <= thickness/2
    eta3d = np.zeros_like(X)
    eta3d[plane & sphere] = 1.0
    eta3d += 0.02 * np.random.randn(*eta3d.shape)
    eta3d = np.clip(eta3d, 0.0, 1.0)
    eta2d = eta3d[:, :, N//2]
    return eta3d, eta2d
eta_3d, eta_2d = create_defect(N, dx_nm, theta, phi)
# ------------------- 3D solver (returns full sigma components) -------------------
@st.cache_data
def compute_stress_3d_components(eta, eps0, theta, phi, dx):
    # simplified copy of earlier full 3D solver returning sxx, syy, szz, sxy, sxz, syz and eps_total
    dx_m = dx * 1e-9
    n = np.array([np.cos(phi)*np.sin(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(theta)])
    s = np.cross(n, [0,0,1])
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(n, [1,0,0])
    s = s/np.linalg.norm(s)
    gamma = eps0
    eps_star = np.zeros((3,3,N,N,N))
    for i in range(3):
        for j in range(3):
            eps_star[i,j] = 0.5*gamma*(n[i]*s[j] + s[i]*n[j]) * eta
    eps_hat = np.fft.fftn(eps_star, axes=(2,3,4))
    k = 2*np.pi * np.fft.fftfreq(N, d=dx_m)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1.0
    khat = np.zeros((3,N,N,N))
    khat[0] = KX / np.sqrt(K2)
    khat[1] = KY / np.sqrt(K2)
    khat[2] = KZ / np.sqrt(K2)
    khat[:,0,0,0] = 0.0
    nu_local = nu
    Gamma = np.zeros((3,3,3,3,N,N,N), dtype=complex)
    for i in range(3):
        for j in range(3):
            for k_ in range(3):
                for l in range(3):
                    term1 = 0.25*(khat[i]*khat[k_]*khat[j]*khat[l] + khat[i]*khat[l]*khat[j]*khat[k_])
                    term2 = khat[i]*khat[j]*khat[k_]*khat[l] / (1 - nu_local)
                    Gamma[i,j,k_,l] = (term1 / mu) - (term2 / (4*mu*(1 - nu_local))) / K2
    trace_hat = eps_hat[0,0] + eps_hat[1,1] + eps_hat[2,2]
    Ceps_hat = lam * np.eye(3)[:, :, None, None, None] * trace_hat[None, None, :, :, :] + 2*mu * eps_hat
    eps_ind_hat = np.zeros_like(Ceps_hat)
    for i in range(3):
        for j in range(3):
            tmp = np.zeros_like(Ceps_hat[0,0])
            for k_ in range(3):
                for l in range(3):
                    tmp += Gamma[i,j,k_,l] * Ceps_hat[k_,l]
            eps_ind_hat[i,j] = -tmp
    eps_ind = np.real(np.fft.ifftn(eps_ind_hat, axes=(2,3,4)))
    # total strain = induced + eigenstrain
    eps_total = eps_ind + eps_star
    trace = eps_total[0,0] + eps_total[1,1] + eps_total[2,2]
    I = np.eye(3)
    sigma = lam * trace[None,None,:,:,:] * I[:,:,None,None,None] + 2*mu * eps_total
    sxx = sigma[0,0]; syy = sigma[1,1]; szz = sigma[2,2]
    sxy = sigma[0,1]; sxz = sigma[0,2]; syz = sigma[1,2]
    return {
        'sxx': sxx, 'syy': syy, 'szz': szz,
        'sxy': sxy, 'sxz': sxz, 'syz': syz,
        'eps_total': eps_total
    }
# ------------------- 2D plane-strain solver (fixed) -------------------
@st.cache_data
def compute_stress_2d_plane(eta2d, eps0, theta, phi, dx):
    dx_m = dx * 1e-9
    gamma = eps0
    n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
    s = np.cross(n, np.array([0, 0, 1]))
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(n, np.array([1, 0, 0]))
    s /= np.linalg.norm(s)
    exx_star = gamma * n[0] * s[0] * eta2d
    eyy_star = gamma * n[1] * s[1] * eta2d
    exy_star = 0.5 * gamma * (n[0] * s[1] + s[0] * n[1]) * eta2d
    ezz_star = gamma * n[2] * s[2] * eta2d
    exz_star = 0.5 * gamma * (n[0] * s[2] + s[0] * n[2]) * eta2d
    eyz_star = 0.5 * gamma * (n[1] * s[2] + s[1] * n[2]) * eta2d
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    # Plane strain moduli
    Cp11_strain = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    Cp12_strain = E * nu / ((1 + nu) * (1 - 2 * nu))
    Cp66_strain = E / (2 * (1 + nu))
    # Plane stress moduli
    Cp11_stress = E / (1 - nu ** 2)
    Cp12_stress = E * nu / (1 - nu ** 2)
    Cp66_stress = E / (2 * (1 + nu))
    # Interpolate based on orientation (fix 4: effective modulus)
    f = np.abs(np.cos(theta))
    Cp11 = (1 - f) * Cp11_strain + f * Cp11_stress
    Cp12 = (1 - f) * Cp12_strain + f * Cp12_stress
    Cp66 = Cp66_strain  # same
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx_m)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX ** 2 + KY ** 2
    zero = (KX == 0) & (KY == 0)
    K2_safe = np.where(zero, 1.0, K2)
    n1 = KX / np.sqrt(K2_safe)
    n2 = KY / np.sqrt(K2_safe)
    n1[zero] = 0.0
    n2[zero] = 0.0
    A11 = Cp11 * n1 ** 2 + Cp66 * n2 ** 2
    A22 = Cp11 * n2 ** 2 + Cp66 * n1 ** 2
    A12 = (Cp12 + Cp66) * n1 * n2
    det = A11 * A22 - A12 ** 2
    det_safe = np.where(np.abs(det) < 1e-30, np.inf, det)
    G11 = A22 / det_safe
    G22 = A11 / det_safe
    G12 = -A12 / det_safe
    G11[zero] = 0.0
    G22[zero] = 0.0
    G12[zero] = 0.0
    # Include ezz_star contribution (fixes 1 and 2: generalized plane strain and Eshelby-like projection)
    txx = Cp11 * exx_star + Cp12 * eyy_star + lam * ezz_star
    tyy = Cp12 * exx_star + Cp11 * eyy_star + lam * ezz_star
    txy = 2 * Cp66 * exy_star
    Sx = np.fft.fft2(txx) * KX + np.fft.fft2(txy) * KY
    Sy = np.fft.fft2(txy) * KX + np.fft.fft2(tyy) * KY
    Sx[zero] = 0.0
    Sy[zero] = 0.0
    ux_hat = -1j * (G11 * Sx + G12 * Sy)
    uy_hat = -1j * (G12 * Sx + G22 * Sy)
    ux = np.real(np.fft.ifft2(ux_hat))
    uy = np.real(np.fft.ifft2(uy_hat))
    exx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(ux)))
    eyy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(uy)))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * np.fft.fft2(uy) + KY * np.fft.fft2(ux))))
    # Total stresses, including corrections
    sxx = Cp11 * (exx - exx_star) + Cp12 * (eyy - eyy_star) - lam * ezz_star
    syy = Cp12 * (exx - exx_star) + Cp11 * (eyy - eyy_star) - lam * ezz_star
    sxy = 2 * Cp66 * (exy - exy_star)
    szz = lam * (exx + eyy - exx_star - eyy_star - ezz_star) - 2 * mu * ezz_star
    sxz = -2 * mu * exz_star
    syz = -2 * mu * eyz_star
    return {
        'sxx': sxx, 'syy': syy, 'szz': szz, 'sxy': sxy, 'sxz': sxz, 'syz': syz,
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'exx_star': exx_star, 'eyy_star': eyy_star, 'exy_star': exy_star
    }
# ------------------- Run and produce selected 2D field -------------------
if st.button("Run comparison", type="primary"):
    with st.spinner("Running 3D solver (full)..."):
        sol3 = compute_stress_3d_components(eta_3d, eps0, theta, phi, dx_nm)
    # 3D central-slice stress magnitude (Pa)
    sxx3 = sol3['sxx']; syy3 = sol3['syy']; szz3 = sol3['szz']; sxy3 = sol3['sxy']; sxz3 = sol3['sxz']; syz3 = sol3['syz']
    slice_idx = N//2
    # central-slice von Mises (Pa)
    vm3_slice = np.sqrt(0.5*((sxx3-syy3)**2 + (syy3-szz3)**2 + (szz3-sxx3)**2 + 6*(sxy3**2 + sxz3**2 + syz3**2)))
    vm3_slice = vm3_slice[:, :, slice_idx]
    if mode.startswith("Option 1"):
        # Exact equality: 2D field is the 3D central slice
        arr3_disp = vm3_slice / 1e9
        arr2_disp = arr3_disp.copy()
        mode_note = "2D field = exact 3D central slice (identical)."
    elif mode.startswith("Option 2"):
        # kz=0 (z-average): average full 3D vm along z (this picks kz=0 Fourier component)
        vm3_full = np.sqrt(0.5*((sxx3-syy3)**2 + (syy3-szz3)**2 + (szz3-sxx3)**2 + 6*(sxy3**2 + sxz3**2 + syz3**2)))
        vm3_kz0 = np.mean(vm3_full, axis=2) # average over z
        arr3_disp = vm3_slice / 1e9
        arr2_disp = vm3_kz0 / 1e9
        mode_note = "2D field = z-average (kz=0) of full 3D von Mises."
    else:
        # Option 3: compute 2D plane-strain and tune scalar alpha to best match vm3_slice
        with st.spinner("Running 2D plane-strain solver..."):
            sol2 = compute_stress_2d_plane(eta_2d, eps0, theta, phi, dx_nm)
        vm2 = np.sqrt(0.5*((sol2['sxx']-sol2['syy'])**2 + (sol2['syy']-sol2['szz'])**2 + (sol2['szz']-sol2['sxx'])**2 + 6*(sol2['sxy']**2 + sol2['sxz']**2 + sol2['syz']**2)))
        # find scalar alpha to minimize L2 error between alpha*vm2 and vm3_slice (Pa)
        vm3_target = vm3_slice
        vm2_flat = vm2.flatten()
        vm3_flat = vm3_target.flatten()
        # simple 1D least squares: alpha = (vm2·vm3)/(vm2·vm2)
        denom = np.dot(vm2_flat, vm2_flat)
        if denom == 0:
            alpha = 1.0
        else:
            alpha = np.dot(vm2_flat, vm3_flat) / denom
        # clamp alpha to reasonable range to avoid nonsense
        alpha = float(np.clip(alpha, 0.1, 10.0))
        arr3_disp = vm3_slice / 1e9
        arr2_disp = (alpha * vm2) / 1e9
        mode_note = f"2D plane-strain computed then scaled by alpha={alpha:.4g} to best match 3D slice (least-squares)."
    # Display heatmaps and metrics (units GPa)
    diff = arr3_disp - arr2_disp
    max3 = np.nanmax(arr3_disp); max2 = np.nanmax(arr2_disp); maxdiff = np.nanmax(np.abs(diff))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("3D (slice) max σ_vM", f"{max3:.3f} GPa")
        fig1 = go.Figure(data=go.Heatmap(z=arr3_disp, colorscale="Hot"))
        fig1.update_layout(title="3D central slice (von Mises, GPa)", height=420)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.metric("2D max σ_vM", f"{max2:.3f} GPa")
        fig2 = go.Figure(data=go.Heatmap(z=arr2_disp, colorscale="Hot"))
        fig2.update_layout(title=f"2D field ({mode.split('—')[-1].strip()}) (von Mises, GPa)", height=420)
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        st.metric("Max |3D − 2D|", f"{maxdiff:.3f} GPa")
        fig3 = go.Figure(data=go.Heatmap(z=diff, colorscale="RdBu", zmid=0))
        fig3.update_layout(title="Difference (3D slice − 2D) (GPa)", height=420)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        fig4 = go.Figure(data=go.Heatmap(z=eta_2d, colorscale="Gray"))
        fig4.update_layout(title="Defect η (2D slice)", height=420)
        st.plotly_chart(fig4, use_container_width=True)
    st.markdown(f"**Mode note:** {mode_note}")
    st.subheader("Statistics")
    st.write(pd.DataFrame({
        'quantity': ['3D slice max (GPa)', '2D field max (GPa)', 'max abs diff (GPa)'],
        'value': [f"{max3:.6g}", f"{max2:.6g}", f"{maxdiff:.6g}"]
    }))
    # CSV download
    flat_eta = eta_2d.flatten()
    flat3 = arr3_disp.flatten()
    flat2 = arr2_disp.flatten()
    df = pd.DataFrame({'eta': flat_eta, '3D (GPa)': flat3, '2D (GPa)': flat2, 'diff (GPa)': (flat3-flat2)})
    st.download_button("Download comparison CSV", df.to_csv(index=False), file_name="mode_compare.csv")
