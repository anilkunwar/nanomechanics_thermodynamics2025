# =============================================
# 2D vs 3D Stress Solver Comparison – UPGRADED & STABLE
# - Fixed 2D numerical instabilities (k=0 handling, safe divisions)
# - Returns many diagnostics: hydrostatic, von Mises, Frobenius norm,
#   principal stresses/strains, CSV export of central slice.
# - Keep N moderate (>=32, <=96 advised) to avoid long runtimes.
# =============================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="2D vs 3D Stress Debug (Upgraded)", layout="wide")
st.title("2D vs 3D FFT Elasticity – Upgraded & Stable")
st.markdown("**Planar defect in Ag nanoparticle • Compare many stress/strain measures**")

# ------------------- Parameters -------------------
st.sidebar.header("Simulation Parameters")
N = st.sidebar.slider("Grid size N", 32, 128, 64, step=16)
dx_nm = st.sidebar.number_input("dx (nm)", 0.05, 1.0, 0.25, 0.05)

eps0 = st.sidebar.slider("Eigenstrain ε*", 0.3, 3.0, 1.414, 0.01)
st.sidebar.subheader("Habit Plane Orientation")
theta_deg = st.sidebar.slider("Polar angle θ (°)", 0, 180, 55, 1)
phi_deg   = st.sidebar.slider("Azimuthal angle φ (°)", 0, 360, 0, 5)

theta = np.deg2rad(theta_deg)
phi   = np.deg2rad(phi_deg)

# ------------------- Material: Silver -------------------
mu  = 46.1e9   # Pa
lam = 93.4e9 - 2*mu/3.0   # Correct λ from C12 ≈ λ + 2μ/3

# ------------------- Create Defect -------------------
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

# ------------------- Utilities -------------------
def _zero_mode_mask_2d(N):
    # returns boolean mask where kx==0 and ky==0
    k = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    return (KX == 0) & (KY == 0)

# ------------------- 3D Solver (unchanged handling of zero mode) -------------------
@st.cache_data
def compute_stress_3d_full(eta, eps0, theta, phi, dx, mu, lam):
    dx_m = dx * 1e-9
    N = eta.shape[0]

    n = np.array([np.cos(phi)*np.sin(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(theta)])
    s = np.cross(n, [0,0,1])
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(n, [1,0,0])
    s = s / np.linalg.norm(s)

    gamma = eps0 * 0.1
    eps_star = np.zeros((3,3,N,N,N))
    for i in range(3):
        for j in range(3):
            eps_star[i,j] = 0.5 * gamma * (n[i]*s[j] + s[i]*n[j]) * eta

    eps_hat = np.fft.fftn(eps_star, axes=(2,3,4))

    k = 2*np.pi * np.fft.fftfreq(N, d=dx_m)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    # safe handling: set K2[0,0,0] to 1 to avoid division by zero when forming khat
    K2_safe = K2.copy()
    K2_safe[0,0,0] = 1.0

    khat = np.zeros((3,N,N,N))
    khat[0] = KX / np.sqrt(K2_safe)
    khat[1] = KY / np.sqrt(K2_safe)
    khat[2] = KZ / np.sqrt(K2_safe)
    khat[:,0,0,0] = 0.0

    # Green tensor (isotropic approximation)
    Gamma = np.zeros((3,3,3,3,N,N,N), dtype=complex)
    nu = 0.37
    for i in range(3):
        for j in range(3):
            for k_ in range(3):
                for l in range(3):
                    term1 = 0.25*(khat[i]*khat[k_]*khat[j]*khat[l] + khat[i]*khat[l]*khat[j]*khat[k_])
                    term2 = khat[i]*khat[j]*khat[k_]*khat[l] / (1 - nu)
                    Gamma[i,j,k_,l] = (term1 / mu) - (term2 / (4*mu*(1 - nu))) / K2_safe

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
    eps_total = eps_ind - eps_star

    trace = eps_total[0,0] + eps_total[1,1] + eps_total[2,2]
    I = np.eye(3)
    sigma = lam * trace[None, None, :, :, :] * I[:, :, None, None, None] + 2*mu * eps_total

    sxx, syy, szz = sigma[0,0], sigma[1,1], sigma[2,2]
    sxy, sxz, syz = sigma[0,1], sigma[0,2], sigma[1,2]

    hydro = (sxx + syy + szz) / 3.0
    vm = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + sxz**2 + syz**2)))
    stress_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*(sxy**2 + sxz**2 + syz**2))

    # principal stresses per voxel
    tot_vox = N*N*N
    sigma_mat = np.zeros((tot_vox, 3, 3))
    idx = 0
    for i in range(N):
        for j in range(N):
            for k_ in range(N):
                sigma_mat[idx,0,0] = sxx[i,j,k_]
                sigma_mat[idx,1,1] = syy[i,j,k_]
                sigma_mat[idx,2,2] = szz[i,j,k_]
                sigma_mat[idx,0,1] = sxy[i,j,k_]
                sigma_mat[idx,1,0] = sxy[i,j,k_]
                sigma_mat[idx,0,2] = sxz[i,j,k_]
                sigma_mat[idx,2,0] = sxz[i,j,k_]
                sigma_mat[idx,1,2] = syz[i,j,k_]
                sigma_mat[idx,2,1] = syz[i,j,k_]
                idx += 1

    princ_vals = np.linalg.eigvalsh(sigma_mat)
    p1 = princ_vals[:,2].reshape((N,N,N))
    p2 = princ_vals[:,1].reshape((N,N,N))
    p3 = princ_vals[:,0].reshape((N,N,N))

    # principal strains
    eps_tot_flat = np.zeros((tot_vox,3,3))
    idx = 0
    for i in range(N):
        for j in range(N):
            for k_ in range(N):
                eps_tot_flat[idx,0,0] = eps_total[0,0,i,j,k_]
                eps_tot_flat[idx,1,1] = eps_total[1,1,i,j,k_]
                eps_tot_flat[idx,2,2] = eps_total[2,2,i,j,k_]
                eps_tot_flat[idx,0,1] = eps_total[0,1,i,j,k_]
                eps_tot_flat[idx,1,0] = eps_total[0,1,i,j,k_]
                eps_tot_flat[idx,0,2] = eps_total[0,2,i,j,k_]
                eps_tot_flat[idx,2,0] = eps_total[0,2,i,j,k_]
                eps_tot_flat[idx,1,2] = eps_total[1,2,i,j,k_]
                eps_tot_flat[idx,2,1] = eps_total[1,2,i,j,k_]
                idx += 1

    eps_princ_vals = np.linalg.eigvalsh(eps_tot_flat)
    ep1 = eps_princ_vals[:,2].reshape((N,N,N))
    ep2 = eps_princ_vals[:,1].reshape((N,N,N))
    ep3 = eps_princ_vals[:,0].reshape((N,N,N))

    out = {
        'sigma_comp': sigma,
        'sxx': sxx, 'syy': syy, 'szz': szz, 'sxy': sxy, 'sxz': sxz, 'syz': syz,
        'hydro': hydro, 'vm': vm, 'stress_mag': stress_mag,
        'p1': p1, 'p2': p2, 'p3': p3,
        'ep1': ep1, 'ep2': ep2, 'ep3': ep3,
        'eps_total': eps_total
    }
    return out

# ------------------- 2D Plane-Strain Solver (STABLE) -------------------
@st.cache_data
def compute_stress_2d_full(eta2d, eps0, theta, dx, mu, lam):
    # Robust plane-strain FFT solver with safe handling of k=0 and near-zero denom
    N = eta2d.shape[0]
    n = np.array([np.cos(theta), np.sin(theta)])
    s = np.array([-np.sin(theta), np.cos(theta)])
    gamma = eps0 * 0.1

    exx_star = gamma * n[0]*s[0] * eta2d
    eyy_star = gamma * n[1]*s[1] * eta2d
    exy_star = 0.5 * gamma * (n[0]*s[1] + s[0]*n[1]) * eta2d

    # isotropic plane-strain constants
    Cp11 = lam + 2*mu
    Cp12 = lam
    Cp66 = mu

    # wavevectors in SI units (1/m)
    k = 2*np.pi * np.fft.fftfreq(N, d=dx*1e-9)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2

    # zero-frequency (rigid) mode mask
    zero = (KX == 0) & (KY == 0)

    # safe normalization for direction cosines
    K2_safe = np.where(zero, 1.0, K2)
    n1 = KX / np.sqrt(K2_safe)
    n2 = KY / np.sqrt(K2_safe)
    n1[zero] = 0.0
    n2[zero] = 0.0

    # Acoustic tensor entries
    A11 = Cp11*n1**2 + Cp66*n2**2
    A22 = Cp11*n2**2 + Cp66*n1**2
    A12 = (Cp12 + Cp66)*n1*n2

    det = A11*A22 - A12**2
    # avoid division by zero: use np.where to place safe large denom
    det_safe = np.where(np.abs(det) < 1e-30, np.inf, det)

    G11 = A22 / det_safe
    G22 = A11 / det_safe
    G12 = -A12 / det_safe

    # enforce zero for rigid (k=0) mode
    G11[zero] = 0.0
    G22[zero] = 0.0
    G12[zero] = 0.0

    # compute traction-like fields in Fourier space
    txx = Cp11*exx_star + Cp12*eyy_star
    tyy = Cp12*exx_star + Cp11*eyy_star
    txy = 2*Cp66*exy_star

    Sx = np.fft.fft2(txx)*KX + np.fft.fft2(txy)*KY
    Sy = np.fft.fft2(txy)*KX + np.fft.fft2(tyy)*KY

    # zero rigid-mode tractions to avoid large displacements
    Sx[zero] = 0.0
    Sy[zero] = 0.0

    ux = np.fft.ifft2(-1j * (G11*Sx + G12*Sy))
    uy = np.fft.ifft2(-1j * (G12*Sx + G22*Sy))

    exx = np.real(np.fft.ifft2(1j*KX*np.fft.fft2(ux)))
    eyy = np.real(np.fft.ifft2(1j*KY*np.fft.fft2(uy)))
    exy = 0.5*np.real(np.fft.ifft2(1j*(KX*np.fft.fft2(uy) + KY*np.fft.fft2(ux))))

    # plane-strain out-of-plane component
    szz = lam*(exx + eyy)

    sxx = Cp11*(exx - exx_star) + Cp12*(eyy - eyy_star)
    syy = Cp12*(exx - exx_star) + Cp11*(eyy - eyy_star)
    sxy = 2*Cp66*(exy - exy_star)

    hydro = (sxx + syy + szz) / 3.0
    vm = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2)))
    stress_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*(sxy**2))

    # principal stresses (2D) via eigenvalues of 2x2 tensor
    tot = N*N
    sigma2_mat = np.zeros((tot,2,2))
    idx = 0
    for i in range(N):
        for j in range(N):
            sigma2_mat[idx,0,0] = sxx[i,j]
            sigma2_mat[idx,1,1] = syy[i,j]
            sigma2_mat[idx,0,1] = sxy[i,j]
            sigma2_mat[idx,1,0] = sxy[i,j]
            idx += 1
    pv = np.linalg.eigvalsh(sigma2_mat)
    princ1 = pv[:,1].reshape((N,N))
    princ2 = pv[:,0].reshape((N,N))

    # principal strains (2D)
    eps2_mat = np.zeros((tot,2,2))
    idx = 0
    for i in range(N):
        for j in range(N):
            eps2_mat[idx,0,0] = exx[i,j]
            eps2_mat[idx,1,1] = eyy[i,j]
            eps2_mat[idx,0,1] = exy[i,j]
            eps2_mat[idx,1,0] = exy[i,j]
            idx += 1
    epv = np.linalg.eigvalsh(eps2_mat)
    ep1 = epv[:,1].reshape((N,N))
    ep2 = epv[:,0].reshape((N,N))

    out = {
        'sxx': sxx, 'syy': syy, 'szz': szz, 'sxy': sxy,
        'hydro': hydro, 'vm': vm, 'stress_mag': stress_mag,
        'p1': princ1, 'p2': princ2,
        'ep1': ep1, 'ep2': ep2,
        'exx': exx, 'eyy': eyy, 'exy': exy
    }
    return out

# ------------------- UI: selection of fields -------------------
FIELDS_3D = {
    'von Mises (GPa)': ('vm', True),
    'Hydrostatic (GPa)': ('hydro', True),
    'Stress magnitude (GPa)': ('stress_mag', True),
    'Principal σ1 (GPa)': ('p1', True),
    'Principal σ2 (GPa)': ('p2', True),
    'Principal σ3 (GPa)': ('p3', True),
    'Principal ε1 (×1e-3)': ('ep1', False),
}

FIELDS_2D = {
    'von Mises (GPa)': ('vm', True),
    'Hydrostatic (GPa)': ('hydro', True),
    'Stress magnitude (GPa)': ('stress_mag', True),
    'Principal σ1 (GPa)': ('p1', True),
    'Principal σ2 (GPa)': ('p2', True),
    'Principal ε1 (×1e-3)': ('ep1', False),
}

field3d_choice = st.sidebar.selectbox("3D Field to display", list(FIELDS_3D.keys()))
field2d_choice = st.sidebar.selectbox("2D Field to display", list(FIELDS_2D.keys()), index=0)

# ------------------- Run Comparison -------------------
if st.button("Run 2D vs 3D Stress Comparison", type="primary"):
    with st.spinner("Running 3D solver (this may take a moment)..."):
        out3 = compute_stress_3d_full(eta_3d, eps0, theta, phi, dx_nm, mu, lam)
    with st.spinner("Running 2D solver..."):
        out2 = compute_stress_2d_full(eta_2d, eps0, theta, dx_nm, mu, lam)

    key3, to_gpa3 = FIELDS_3D[field3d_choice]
    key2, to_gpa2 = FIELDS_2D[field2d_choice]

    # pick central slice for 3D
    slice_idx = N//2
    arr3 = out3[key3][:, :, slice_idx]
    arr2 = out2[key2]

    # unit conversions and scaling for display
    if to_gpa3:
        arr3_disp = arr3 / 1e9
        arr2_disp = arr2 / 1e9
        units = "GPa"
    else:
        arr3_disp = arr3 * 1e3
        arr2_disp = arr2 * 1e3
        units = "×10^-3"

    diff = arr3_disp - arr2_disp

    max3 = np.nanmax(arr3_disp)
    max2 = np.nanmax(arr2_disp)
    maxdiff = np.nanmax(np.abs(diff))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"3D Max ({field3d_choice})", f"{max3:.3f} {units}")
        fig1 = go.Figure(data=go.Heatmap(z=arr3_disp, colorscale="Hot"))
        fig1.update_layout(title=f"3D (slice {slice_idx}) — {field3d_choice}", height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.metric(f"2D Max ({field2d_choice})", f"{max2:.3f} {units}")
        fig2 = go.Figure(data=go.Heatmap(z=arr2_disp, colorscale="Hot"))
        fig2.update_layout(title=f"2D Plane-Strain — {field2d_choice}", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.metric("Max |3D−2D|", f"{maxdiff:.3f} {units}")
        fig3 = go.Figure(data=go.Heatmap(z=diff, colorscale="RdBu", zmid=0))
        fig3.update_layout(title="Difference (3D slice − 2D)", height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = go.Figure(data=go.Heatmap(z=eta_2d, colorscale="Gray"))
        fig4.update_layout(title="Defect η (2D slice)", height=420)
        st.plotly_chart(fig4, use_container_width=True)

    st.success(f"Done — compared {field3d_choice} vs {field2d_choice}. Max diff = {maxdiff:.3f} {units}.")

    # Offer CSV download of the central slice comparison
    flat3 = arr3_disp.flatten()
    flat2 = arr2_disp.flatten()
    flat_eta = eta_2d.flatten()
    df_compare = pd.DataFrame({'eta': flat_eta, '3D': flat3, '2D': flat2, 'diff': (flat3-flat2)})
    csv = df_compare.to_csv(index=False)
    st.download_button("Download central-slice comparison CSV", csv, file_name='stress_compare_slice.csv', mime='text/csv')

    # show small stats
    st.write("### Quick stats")
    st.write(pd.DataFrame({
        'quantity': ['3D max', '2D max', 'max abs diff'],
        'value': [f"{max3:.5g} {units}", f"{max2:.5g} {units}", f"{maxdiff:.5g} {units}"]
    }))

    st.write("If you want additional fields exported (full 3D arrays) or a different slice/orientation, tell me and I can add that.")
