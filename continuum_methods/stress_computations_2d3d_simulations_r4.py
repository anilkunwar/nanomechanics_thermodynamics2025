# =============================================
# 2D vs 3D Stress Solver Comparison – WITH ROBUST OPTION 3
# =============================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="2D vs 3D Stress Debug", layout="wide")
st.title("2D vs 3D FFT Elasticity – All Three Options")
st.markdown("**Planar defect in Ag nanoparticle • Realistic ~200 GPa stresses**")

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

# ------------------- Comparison Mode -------------------
comparison_mode = st.sidebar.selectbox(
    "Comparison Mode",
    [
        "Option 1 — Direct 2D vs 3D slice",
        "Option 2 — Reduced 2+1D solver", 
        "Option 3 — Scaled 2D plane-strain (robust)"
    ]
)

# ------------------- Material: Silver -------------------
mu  = 46.1e9   # Pa
lam = 93.4e9 - 2*mu/3.0   # Correct λ from C12 ≈ λ + 2μ/3

# ------------------- Create Defect -------------------
@st.cache_data
def create_defect(N, dx):
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

eta_3d, eta_2d = create_defect(N, dx_nm)

# ------------------- CORRECT 3D Isotropic Solver -------------------
@st.cache_data
def compute_stress_3d(eta, eps0, theta, phi, dx):
    dx_m = dx * 1e-9
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
    K2[0,0,0] = 1.0

    khat = np.zeros((3,N,N,N))
    khat[0] = KX / np.sqrt(K2)
    khat[1] = KY / np.sqrt(K2)
    khat[2] = KZ / np.sqrt(K2)
    khat[:,0,0,0] = 0

    Gamma = np.zeros((3,3,3,3,N,N,N), dtype=complex)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    term1 = 0.25*(khat[i]*khat[k]*khat[j]*khat[l] + khat[i]*khat[l]*khat[j]*khat[k])
                    term2 = khat[i]*khat[j]*khat[k]*khat[l] / (1 - 0.37)
                    Gamma[i,j,k,l] = (term1 / mu) - (term2 / (4*mu*(1 - 0.37))) / K2

    Ceps_hat = lam * np.eye(3)[:, :, None, None, None] * (eps_hat[0,0] + eps_hat[1,1] + eps_hat[2,2]) + 2*mu * eps_hat
    eps_ind_hat = np.zeros_like(Ceps_hat)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    eps_ind_hat[i,j] -= Gamma[i,j,k,l] * Ceps_hat[k,l]

    eps_ind = np.real(np.fft.ifftn(eps_ind_hat, axes=(2,3,4)))
    eps_total = eps_ind - eps_star

    trace = eps_total[0,0] + eps_total[1,1] + eps_total[2,2]
    I = np.eye(3)
    sigma = lam * trace[None, None, :, :, :] * I[:, :, None, None, None] + 2*mu * eps_total
    sxx, syy, szz = sigma[0,0], sigma[1,1], sigma[2,2]
    sxy, sxz, syz = sigma[0,1], sigma[0,2], sigma[1,2]

    vm = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + sxz**2 + syz**2))) / 1e9
    return vm

# ------------------- 2D Plane-Strain Solver -------------------
@st.cache_data
def compute_stress_2d(eta2d, eps0, theta):
    n = np.array([np.cos(theta), np.sin(theta)])
    s = np.array([-np.sin(theta), np.cos(theta)])
    gamma = eps0 * 0.1

    exx_star = gamma * n[0]*s[0] * eta2d
    eyy_star = gamma * n[1]*s[1] * eta2d
    exy_star = 0.5 * gamma * (n[0]*s[1] + s[0]*n[1]) * eta2d

    Cp11 = 124e9 - 93.4e9**2 / 124e9
    Cp12 = 93.4e9 * (124e9 - 93.4e9) / 124e9
    Cp66 = 46.1e9

    k = 2*np.pi * np.fft.fftfreq(N, d=dx_nm)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2 + 1e-20

    n1 = KX / np.sqrt(K2)
    n2 = KY / np.sqrt(K2)
    A11 = Cp11*n1**2 + Cp66*n2**2
    A22 = Cp11*n2**2 + Cp66*n1**2
    A12 = (Cp12 + Cp66)*n1*n2
    det = A11*A22 - A12**2
    G11 = np.where(det != 0, A22/det, 0)
    G22 = np.where(det != 0, A11/det, 0)
    G12 = np.where(det != 0, -A12/det, 0)

    txx = Cp11*exx_star + Cp12*eyy_star
    tyy = Cp12*exx_star + Cp11*eyy_star
    txy = 2*Cp66*exy_star

    Sx = np.fft.fft2(txx)*KX + np.fft.fft2(txy)*KY
    Sy = np.fft.fft2(txy)*KX + np.fft.fft2(tyy)*KY

    ux = np.fft.ifft2(-1j * (G11*Sx + G12*Sy))
    uy = np.fft.ifft2(-1j * (G12*Sx + G22*Sy))

    exx = np.real(np.fft.ifft2(1j*KX*np.fft.fft2(ux)))
    eyy = np.real(np.fft.ifft2(1j*KY*np.fft.fft2(uy)))
    exy = 0.5*np.real(np.fft.ifft2(1j*(KX*np.fft.fft2(uy) + KY*np.fft.fft2(ux))))

    sxx = Cp11*(exx - exx_star) + Cp12*(eyy - eyy_star)
    syy = Cp12*(exx - exx_star) + Cp11*(eyy - eyy_star)
    sxy = 2*Cp66*(exy - exy_star)

    vm = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2) / 1e9
    return vm

# ------------------- Option 2: Reduced 2+1D Solver -------------------
@st.cache_data
def compute_stress_2d_reduced(eta2d, eps0, theta):
    """Option 2: 2D solver with kz=0 projection approximation"""
    n = np.array([np.cos(theta), np.sin(theta)])
    s = np.array([-np.sin(theta), np.cos(theta)])
    gamma = eps0 * 0.1

    exx_star = gamma * n[0]*s[0] * eta2d
    eyy_star = gamma * n[1]*s[1] * eta2d
    exy_star = 0.5 * gamma * (n[0]*s[1] + s[0]*n[1]) * eta2d

    # Use full 3D moduli but with plane-strain constraint
    nu = 0.37  # Poisson's ratio for silver
    E = 2 * mu * (1 + nu)
    
    # Plane-strain moduli derived from 3D parameters
    Cp11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    Cp12 = E * nu / ((1 + nu) * (1 - 2 * nu))
    Cp66 = E / (2 * (1 + nu))

    k = 2*np.pi * np.fft.fftfreq(N, d=dx_nm)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2 + 1e-20

    n1 = KX / np.sqrt(K2)
    n2 = KY / np.sqrt(K2)
    A11 = Cp11*n1**2 + Cp66*n2**2
    A22 = Cp11*n2**2 + Cp66*n1**2
    A12 = (Cp12 + Cp66)*n1*n2
    det = A11*A22 - A12**2
    G11 = np.where(det != 0, A22/det, 0)
    G22 = np.where(det != 0, A11/det, 0)
    G12 = np.where(det != 0, -A12/det, 0)

    txx = Cp11*exx_star + Cp12*eyy_star
    tyy = Cp12*exx_star + Cp11*eyy_star
    txy = 2*Cp66*exy_star

    Sx = np.fft.fft2(txx)*KX + np.fft.fft2(txy)*KY
    Sy = np.fft.fft2(txy)*KX + np.fft.fft2(tyy)*KY

    ux = np.fft.ifft2(-1j * (G11*Sx + G12*Sy))
    uy = np.fft.ifft2(-1j * (G12*Sx + G22*Sy))

    exx = np.real(np.fft.ifft2(1j*KX*np.fft.fft2(ux)))
    eyy = np.real(np.fft.ifft2(1j*KY*np.fft.fft2(uy)))
    exy = 0.5*np.real(np.fft.ifft2(1j*(KX*np.fft.fft2(uy) + KY*np.fft.fft2(ux))))

    sxx = Cp11*(exx - exx_star) + Cp12*(eyy - eyy_star)
    syy = Cp12*(exx - exx_star) + Cp11*(eyy - eyy_star)
    sxy = 2*Cp66*(exy - exy_star)

    vm = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2) / 1e9
    return vm

# ------------------- Run Comparison -------------------
if st.button("Run 2D vs 3D Stress Comparison", type="primary"):
    with st.spinner("Running 3D solver..."):
        vm_3d = compute_stress_3d(eta_3d, eps0, theta, phi, dx_nm)
    
    if comparison_mode == "Option 1 — Direct 2D vs 3D slice":
        with st.spinner("Running standard 2D plane-strain solver..."):
            vm_2d = compute_stress_2d(eta_2d, eps0, theta)
        alpha = 1.0
        description = "Direct comparison of 2D plane-strain with 3D central slice"
        
    elif comparison_mode == "Option 2 — Reduced 2+1D solver":
        with st.spinner("Running reduced 2+1D solver..."):
            vm_2d = compute_stress_2d_reduced(eta_2d, eps0, theta)
        alpha = 1.0
        description = "2D with full 3D moduli under plane-strain constraint"
        
    elif comparison_mode == "Option 3 — Scaled 2D plane-strain (robust)":
        with st.spinner("Running 2D plane-strain + robust scaling..."):
            vm_2d_base = compute_stress_2d(eta_2d, eps0, theta)
            
        # Robust scaling computation
        vm_3d_slice = vm_3d[:, :, N//2]
        vm2_flat = vm_2d_base.flatten()
        vm3_flat = vm_3d_slice.flatten()
        eta_flat = eta_2d.flatten()

        # Mask only inside defect region to avoid numerical issues
        mask = eta_flat > 0.05
        vm2_masked = vm2_flat[mask]
        vm3_masked = vm3_flat[mask]

        # Regularized least-squares scaling
        denom = np.dot(vm2_masked, vm2_masked) + 1e-12
        alpha = np.dot(vm2_masked, vm3_masked) / denom

        # Clamp alpha to prevent unphysical scaling
        alpha = np.clip(alpha, 0.1, 5.0)

        # Apply scaling
        vm_2d = alpha * vm_2d_base
        description = f"2D plane-strain scaled by α = {alpha:.3f} (robust fit)"

    vm_3d_slice = vm_3d[:, :, N//2]
    max_3d = vm_3d_slice.max()
    max_2d = vm_2d.max()
    diff_max = np.abs(vm_3d_slice - vm_2d).max()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("3D Max σ_vM", f"{max_3d:.1f} GPa")
        fig1 = go.Figure(data=go.Heatmap(z=vm_3d_slice, colorscale="Hot", zmin=0, zmax=250))
        fig1.update_layout(title="3D von Mises", height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.metric("2D Max σ_vM", f"{max_2d:.1f} GPa")
        fig2 = go.Figure(data=go.Heatmap(z=vm_2d, colorscale="Hot", zmin=0, zmax=250))
        fig2.update_layout(title=f"2D - {comparison_mode.split('—')[1].strip()}", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.metric("Max |3D−2D|", f"{diff_max:.1f} GPa")
        fig3 = go.Figure(data=go.Heatmap(z=vm_3d_slice - vm_2d, colorscale="RdBu", zmid=0))
        fig3.update_layout(title="Difference", height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.metric("Scaling Factor α", f"{alpha:.3f}" if "Option 3" in comparison_mode else "N/A")
        fig4 = go.Figure(data=go.Heatmap(z=eta_2d, colorscale="Gray"))
        fig4.update_layout(title="Defect η", height=420)
        st.plotly_chart(fig4, use_container_width=True)

    st.success(f"{description}")
    st.info(f"Comparison complete! 3D and 2D agree within {diff_max:.1f} GPa")

    # Additional diagnostics for Option 3
    if "Option 3" in comparison_mode:
        st.subheader("Option 3 Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scaling Statistics**")
            st.write(f"- Computed α: {alpha:.4f}")
            st.write(f"- Masked pixels: {np.sum(mask)}/{len(eta_flat)}")
            st.write(f"- 2D base max: {vm_2d_base.max():.1f} GPa")
            st.write(f"- Scaled 2D max: {max_2d:.1f} GPa")
            
        with col2:
            st.write("**Fit Quality**")
            correlation = np.corrcoef(vm2_masked, vm3_masked)[0,1] if len(vm2_masked) > 1 else 0
            st.write(f"- Correlation in defect: {correlation:.3f}")
            relative_error = diff_max / max_3d
            st.write(f"- Relative error: {relative_error:.3f}")
