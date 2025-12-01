# 2D vs 3D Stress — multi-field visualization WITH DEBUGGED SCALING
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="2D vs 3D Stress Debug (Multi-field)", layout="wide")
st.title("2D vs 3D FFT Elasticity — Multi-field Visualization with All Three Options")
st.markdown("**Based on your working 2D/3D solver — now visualizing many stress/strain types**")

# ------------------- Parameters -------------------
st.sidebar.header("Simulation Parameters")
N = st.sidebar.slider("Grid size N", 32, 128, 64, step=16)
dx_nm = st.sidebar.number_input("dx (nm)", 0.05, 1.0, 0.25, 0.05)

# IMPORTANT: Use the same eps0 range as your working baseline
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

# ------------------- Material: Silver (SAME AS YOUR WORKING BASELINE) -------------------
mu = 46.1e9   # Pa
lam = 93.4e9 - 2*mu/3.0   # Correct λ from C12 ≈ λ + 2μ/3
# Note: Your working baseline uses hardcoded nu = 0.37, not computed from lam/mu
nu = 0.37  # Using the same as your working baseline

st.sidebar.write(f"Material properties: μ = {mu/1e9:.1f} GPa, λ = {lam/1e9:.1f} GPa, ν = {nu:.3f}")

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

# ------------------- 3D Solver (USING SAME SCALING AS WORKING BASELINE) -------------------
@st.cache_data
def compute_stress_3d_full(eta, eps0, theta, phi, dx):
    dx_m = dx * 1e-9
    n = np.array([np.cos(phi)*np.sin(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(theta)])
    
    # Find shear direction s (perpendicular to n)
    s = np.cross(n, [0,0,1])
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(n, [1,0,0])
    s = s / np.linalg.norm(s)

    # CRITICAL FIX: Use the SAME scaling as working baseline
    gamma = eps0 * 0.1  # This is what your working code uses
    
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
            for k_ in range(3):
                for l in range(3):
                    term1 = 0.25*(khat[i]*khat[k_]*khat[j]*khat[l] + khat[i]*khat[l]*khat[j]*khat[k_])
                    term2 = khat[i]*khat[j]*khat[k_]*khat[l] / (1 - nu)
                    Gamma[i,j,k_,l] = (term1 / mu) - (term2 / (4*mu*(1 - nu))) / K2

    Ceps_hat = lam * np.eye(3)[:, :, None, None, None] * (eps_hat[0,0] + eps_hat[1,1] + eps_hat[2,2]) + 2*mu * eps_hat
    eps_ind_hat = np.zeros_like(Ceps_hat)
    for i in range(3):
        for j in range(3):
            for k_ in range(3):
                for l in range(3):
                    eps_ind_hat[i,j] -= Gamma[i,j,k_,l] * Ceps_hat[k_,l]

    eps_ind = np.real(np.fft.ifftn(eps_ind_hat, axes=(2,3,4)))
    # Note: Your working code uses eps_total = eps_ind - eps_star
    eps_total = eps_ind - eps_star

    trace = eps_total[0,0] + eps_total[1,1] + eps_total[2,2]
    I = np.eye(3)
    sigma = lam * trace[None, None, :, :, :] * I[:, :, None, None, None] + 2*mu * eps_total

    # components (Pa)
    sxx = sigma[0,0]; syy = sigma[1,1]; szz = sigma[2,2]
    sxy = sigma[0,1]; sxz = sigma[0,2]; syz = sigma[1,2]

    # derived measures (Pa)
    hydro = (sxx + syy + szz) / 3.0
    vm = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + sxz**2 + syz**2)))
    stress_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*(sxy**2 + sxz**2 + syz**2))

    # principal stresses on the central slice
    slice_idx = N//2
    sxx_s = sxx[:, :, slice_idx]; syy_s = syy[:, :, slice_idx]; szz_s = szz[:, :, slice_idx]
    sxy_s = sxy[:, :, slice_idx]

    princ1 = np.zeros((N,N)); princ2 = np.zeros((N,N)); princ3 = np.zeros((N,N))
    for (i,j) in np.ndindex(N,N):
        S = np.array([[sxx_s[i,j], sxy_s[i,j], 0],
                      [sxy_s[i,j], syy_s[i,j], 0],
                      [0, 0, szz_s[i,j]]])
        vals = np.linalg.eigvalsh(S)
        princ1[i,j], princ2[i,j], princ3[i,j] = vals[2], vals[1], vals[0]

    # principal strains on central slice
    ep1 = np.zeros((N,N)); ep2 = np.zeros((N,N)); ep3 = np.zeros((N,N))
    for (i,j) in np.ndindex(N,N):
        E = np.array([[eps_total[0,0,i,j,slice_idx], eps_total[0,1,i,j,slice_idx], 0],
                      [eps_total[1,0,i,j,slice_idx], eps_total[1,1,i,j,slice_idx], 0],
                      [0, 0, eps_total[2,2,i,j,slice_idx]]])
        evals = np.linalg.eigvalsh(E)
        ep1[i,j], ep2[i,j], ep3[i,j] = evals[2], evals[1], evals[0]

    out = {
        'sxx': sxx, 'syy': syy, 'szz': szz,
        'sxy': sxy, 'sxz': sxz, 'syz': syz,
        'hydro': hydro, 'vm': vm, 'stress_mag': stress_mag,
        'eps_total': eps_total, 'eps_star': eps_star,
        # central-slice principal stresses/strains (Pa)
        'p1_slice': princ1, 'p2_slice': princ2, 'p3_slice': princ3,
        'ep1_slice': ep1, 'ep2_slice': ep2, 'ep3_slice': ep3,
        'sigma': sigma
    }
    return out

# ------------------- 2D Plane-Strain Solver (SAME AS WORKING BASELINE) -------------------
@st.cache_data
def compute_stress_2d_baseline(eta2d, eps0, theta, dx):
    """EXACT COPY of your working baseline 2D solver"""
    n = np.array([np.cos(theta), np.sin(theta)])
    s = np.array([-np.sin(theta), np.cos(theta)])
    
    # CRITICAL: Use SAME scaling as working baseline
    gamma = eps0 * 0.1  # This is what your working code uses

    exx_star = gamma * n[0]*s[0] * eta2d
    eyy_star = gamma * n[1]*s[1] * eta2d
    exy_star = 0.5 * gamma * (n[0]*s[1] + s[0]*n[1]) * eta2d

    # EXACT SAME moduli as your working baseline
    Cp11 = 124e9 - 93.4e9**2 / 124e9
    Cp12 = 93.4e9 * (124e9 - 93.4e9) / 124e9
    Cp66 = 46.1e9

    # Note: Your baseline uses dx_nm (nm) for wavevectors, not dx_m
    k = 2*np.pi * np.fft.fftfreq(N, d=dx)  # dx in nm, same as baseline
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
    
    # Calculate out-of-plane stress for plane strain
    # For isotropic material: szz = ν(σxx + σyy)
    szz = nu * (sxx + syy)
    
    hydro = (sxx + syy + szz) / 3.0
    vm = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2)  # Pa
    stress_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*sxy**2)

    # 2D principal stresses
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
    p1 = pv[:,1].reshape((N,N)); p2 = pv[:,0].reshape((N,N))

    # 2D principal strains
    eps2_mat = np.zeros((tot,2,2))
    idx = 0
    for i in range(N):
        for j in range(N):
            eps2_mat[idx,0,0] = exx[i,j] - exx_star[i,j]
            eps2_mat[idx,1,1] = eyy[i,j] - eyy_star[i,j]
            eps2_mat[idx,0,1] = exy[i,j] - exy_star[i,j]
            eps2_mat[idx,1,0] = exy[i,j] - exy_star[i,j]
            idx += 1
    epv = np.linalg.eigvalsh(eps2_mat)
    ep1 = epv[:,1].reshape((N,N)); ep2 = epv[:,0].reshape((N,N))

    out = {
        'sxx': sxx, 'syy': syy, 'szz': szz, 'sxy': sxy,
        'hydro': hydro, 'vm': vm, 'stress_mag': stress_mag,
        'p1': p1, 'p2': p2,
        'ep1': ep1, 'ep2': ep2,
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'exx_star': exx_star, 'eyy_star': eyy_star, 'exy_star': exy_star,
        'base_vm': vm  # Store for Option 3
    }
    return out

# ------------------- 2D Reduced Solver (Option 2) -------------------
@st.cache_data
def compute_stress_2d_reduced(eta2d, eps0, theta, dx):
    """Alternative 2D solver with different moduli for Option 2"""
    n = np.array([np.cos(theta), np.sin(theta)])
    s = np.array([-np.sin(theta), np.cos(theta)])
    
    # Same scaling as baseline
    gamma = eps0 * 0.1

    exx_star = gamma * n[0]*s[0] * eta2d
    eyy_star = gamma * n[1]*s[1] * eta2d
    exy_star = 0.5 * gamma * (n[0]*s[1] + s[0]*n[1]) * eta2d

    # Alternative moduli derived from 3D parameters
    E = mu * (3*lam + 2*mu) / (lam + mu)  # Young's modulus
    
    Cp11 = E * (1 - nu) / ((1 + nu) * (1 - 2*nu))
    Cp12 = E * nu / ((1 + nu) * (1 - 2*nu))
    Cp66 = E / (2 * (1 + nu))

    k = 2*np.pi * np.fft.fftfreq(N, d=dx)  # dx in nm
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
    
    szz = nu * (sxx + syy)
    
    hydro = (sxx + syy + szz) / 3.0
    vm = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2)
    stress_mag = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*sxy**2)

    # 2D principal stresses
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
    p1 = pv[:,1].reshape((N,N)); p2 = pv[:,0].reshape((N,N))

    # 2D principal strains
    eps2_mat = np.zeros((tot,2,2))
    idx = 0
    for i in range(N):
        for j in range(N):
            eps2_mat[idx,0,0] = exx[i,j] - exx_star[i,j]
            eps2_mat[idx,1,1] = eyy[i,j] - eyy_star[i,j]
            eps2_mat[idx,0,1] = exy[i,j] - exy_star[i,j]
            eps2_mat[idx,1,0] = exy[i,j] - exy_star[i,j]
            idx += 1
    epv = np.linalg.eigvalsh(eps2_mat)
    ep1 = epv[:,1].reshape((N,N)); ep2 = epv[:,0].reshape((N,N))

    out = {
        'sxx': sxx, 'syy': syy, 'szz': szz, 'sxy': sxy,
        'hydro': hydro, 'vm': vm, 'stress_mag': stress_mag,
        'p1': p1, 'p2': p2,
        'ep1': ep1, 'ep2': ep2,
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'exx_star': exx_star, 'eyy_star': eyy_star, 'exy_star': exy_star,
        'base_vm': vm
    }
    return out

# ------------------- UI: field selection -------------------
FIELDS_3D = {
    'von Mises': ('vm', True),
    'Hydrostatic (mean) stress': ('hydro', True),
    'Stress magnitude (Frobenius)': ('stress_mag', True),
    'σ_xx': ('sxx', True),
    'σ_yy': ('syy', True), 
    'σ_xy': ('sxy', True),
    'Principal σ1 (central slice)': ('p1_slice', True),
    'Principal σ2 (central slice)': ('p2_slice', True),
    'Principal σ3 (central slice)': ('p3_slice', True),
    'Principal ε1 (central slice)': ('ep1_slice', False),
    'Principal ε2 (central slice)': ('ep2_slice', False),
}

FIELDS_2D = {
    'von Mises': ('vm', True),
    'Hydrostatic (mean) stress': ('hydro', True),
    'Stress magnitude (Frobenius)': ('stress_mag', True),
    'σ_xx': ('sxx', True),
    'σ_yy': ('syy', True),
    'σ_xy': ('sxy', True),
    'Principal σ1 (2D)': ('p1', True),
    'Principal σ2 (2D)': ('p2', True),
    'Principal ε1 (2D)': ('ep1', False),
    'Principal ε2 (2D)': ('ep2', False),
    'ε_xx (elastic)': ('exx', False),
    'ε_yy (elastic)': ('eyy', False),
}

field3d = st.sidebar.selectbox("3D field (central slice)", list(FIELDS_3D.keys()))
field2d = st.sidebar.selectbox("2D field", list(FIELDS_2D.keys()))

# ------------------- Run & visualize -------------------
if st.button("Run 2D vs 3D Comparison", type="primary"):
    # Compute 3D stress
    with st.spinner("Running 3D solver..."):
        out3 = compute_stress_3d_full(eta_3d, eps0, theta, phi, dx_nm)
    
    # Initialize alpha
    alpha = 1.0
    
    # Compute 2D stress based on selected option
    if comparison_mode == "Option 1 — Direct 2D vs 3D slice":
        with st.spinner("Running baseline 2D plane-strain solver..."):
            out2 = compute_stress_2d_baseline(eta_2d, eps0, theta, dx_nm)
        description = "Direct comparison: baseline 2D plane-strain vs 3D central slice"
        
    elif comparison_mode == "Option 2 — Reduced 2+1D solver":
        with st.spinner("Running reduced 2+1D solver..."):
            out2 = compute_stress_2d_reduced(eta_2d, eps0, theta, dx_nm)
        description = "Reduced 2+1D solver with moduli derived from 3D parameters"
        
    elif comparison_mode == "Option 3 — Scaled 2D plane-strain (robust)":
        with st.spinner("Running baseline 2D + robust scaling..."):
            out2_base = compute_stress_2d_baseline(eta_2d, eps0, theta, dx_nm)
            
        # Robust scaling computation - use vm stress for scaling
        vm_3d_slice = out3['vm'][:, :, N//2] / 1e9  # Convert to GPa
        vm_2d_base = out2_base['vm'] / 1e9  # Convert to GPa
        
        vm2_flat = vm_2d_base.flatten()
        vm3_flat = vm_3d_slice.flatten()
        eta_flat = eta_2d.flatten()

        # Mask only inside defect region
        mask = eta_flat > 0.05
        vm2_masked = vm2_flat[mask]
        vm3_masked = vm3_flat[mask]

        if len(vm2_masked) > 0 and len(vm3_masked) > 0:
            # Regularized least-squares scaling
            denom = np.dot(vm2_masked, vm2_masked) + 1e-12
            alpha = np.dot(vm2_masked, vm3_masked) / denom
            # Clamp alpha to reasonable range
            alpha = np.clip(alpha, 0.1, 10.0)
        else:
            alpha = 1.0

        # Apply scaling to all stress fields
        out2 = out2_base.copy()
        # Scale only stress fields (not strains)
        stress_fields = ['sxx', 'syy', 'szz', 'sxy', 'hydro', 'vm', 'stress_mag', 'p1', 'p2']
        for key in stress_fields:
            if key in out2:
                out2[key] = out2[key] * alpha
        
        description = f"Baseline 2D plane-strain scaled by α = {alpha:.3f} (robust fit)"

    # Extract arrays for display
    key3, is_stress3 = FIELDS_3D[field3d]
    key2, is_stress2 = FIELDS_2D[field2d]
    
    # Extract 3D array (central slice)
    if key3.endswith('_slice'):
        arr3 = out3[key3]
    else:
        arr3 = out3[key3][:, :, N//2]
    
    arr2 = out2[key2]

    # Convert units: stresses -> GPa; strains -> dimensionless
    if is_stress3 and is_stress2:
        arr3_disp = arr3 / 1e9
        arr2_disp = arr2 / 1e9
        units = "GPa"
    else:
        # Strains - keep as is
        arr3_disp = arr3
        arr2_disp = arr2  
        units = "strain"

    diff = arr3_disp - arr2_disp

    # Calculate statistics
    max3 = np.nanmax(arr3_disp)
    max2 = np.nanmax(arr2_disp)
    maxdiff = np.nanmax(np.abs(diff))
    mean3 = np.nanmean(arr3_disp)
    mean2 = np.nanmean(arr2_disp)

    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("3D Max", f"{max3:.2f} {units}")
        fig1 = go.Figure(data=go.Heatmap(z=arr3_disp, colorscale="Hot"))
        fig1.update_layout(title=f"3D central slice — {field3d}", height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.metric("2D Max", f"{max2:.2f} {units}")
        fig2 = go.Figure(data=go.Heatmap(z=arr2_disp, colorscale="Hot"))
        fig2.update_layout(title=f"2D — {field2d}", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.metric("Max |3D−2D|", f"{maxdiff:.2f} {units}")
        fig3 = go.Figure(data=go.Heatmap(z=diff, colorscale="RdBu", zmid=0))
        fig3.update_layout(title="Difference (3D slice − 2D)", height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "Option 3" in comparison_mode:
            st.metric("Scaling Factor α", f"{alpha:.3f}")
        else:
            st.metric("Defect η", f"Max: {eta_2d.max():.2f}")
        fig4 = go.Figure(data=go.Heatmap(z=eta_2d, colorscale="Gray"))
        fig4.update_layout(title="Defect η (2D slice)", height=420)
        st.plotly_chart(fig4, use_container_width=True)

    # Display description
    st.info(f"**{description}**")

    # Additional diagnostics
    st.subheader("Field Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**3D (central slice)**")
        st.write(f"- Max: {max3:.2f} {units}")
        st.write(f"- Mean: {mean3:.2f} {units}")
        st.write(f"- Std: {np.nanstd(arr3_disp):.2f} {units}")
        st.write(f"- Min: {np.nanmin(arr3_disp):.2f} {units}")
        
    with col2:
        st.write("**2D**")
        st.write(f"- Max: {max2:.2f} {units}")
        st.write(f"- Mean: {mean2:.2f} {units}")
        st.write(f"- Std: {np.nanstd(arr2_disp):.2f} {units}")
        st.write(f"- Min: {np.nanmin(arr2_disp):.2f} {units}")

    # Option 3 specific diagnostics
    if "Option 3" in comparison_mode:
        st.subheader("Option 3 Scaling Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scaling Statistics**")
            st.write(f"- Computed α: {alpha:.4f}")
            st.write(f"- Masked pixels: {np.sum(mask)}/{len(eta_flat)}")
            st.write(f"- Base 2D VM max: {vm_2d_base.max():.2f} GPa")
            st.write(f"- Scaled 2D VM max: {out2['vm'].max()/1e9:.2f} GPa")
            
        with col2:
            st.write("**Fit Quality**")
            if len(vm2_masked) > 1:
                correlation = np.corrcoef(vm2_masked, vm3_masked)[0,1]
                st.write(f"- Correlation in defect: {correlation:.3f}")
            relative_error = maxdiff / max3 if max3 != 0 else 0
            st.write(f"- Relative error: {relative_error:.3f}")

    if max3 > 0:
        relative_diff = maxdiff / max3
        if relative_diff < 0.2:
            st.success(f"Good agreement! Max difference = {maxdiff:.2f} {units} ({relative_diff:.1%} of 3D max)")
        elif relative_diff < 0.5:
            st.warning(f"Moderate agreement. Max difference = {maxdiff:.2f} {units} ({relative_diff:.1%} of 3D max)")
        else:
            st.error(f"Poor agreement. Max difference = {maxdiff:.2f} {units} ({relative_diff:.1%} of 3D max)")
    else:
        st.info(f"Comparison complete. Max difference = {maxdiff:.2f} {units}")

    # CSV download
    flat_eta = eta_2d.flatten()
    flat3 = arr3_disp.flatten()
    flat2 = arr2_disp.flatten()
    df = pd.DataFrame({
        'eta': flat_eta, 
        '3D': flat3, 
        '2D': flat2, 
        'diff': (flat3-flat2)
    })
    st.download_button(
        "Download slice comparison CSV", 
        df.to_csv(index=False), 
        file_name="slice_compare.csv"
    )
