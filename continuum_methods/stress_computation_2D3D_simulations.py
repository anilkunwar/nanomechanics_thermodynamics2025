# =============================================
# 2D vs 3D Stress Solver Comparison – DEBUG TOOL
# =============================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="2D vs 3D Stress Debug", layout="wide")
st.title("2D vs 3D FFT Elasticity Stress Comparison")
st.markdown("**Same defect • Same eigenstrain • Same orientation → Direct quantitative comparison**")

# ------------------- Parameters -------------------
N = st.sidebar.slider("Grid size N³ (3D) / N×N (2D)", 32, 128, 64, 16)
dx_nm = st.sidebar.number_input("dx (nm)", 0.1, 1.0, 0.25, 0.05)

eps0 = st.sidebar.slider("Eigenstrain ε*", 0.5, 3.0, 1.414, 0.01)  # ESF default
theta_deg = st.sidebar.slider("Habit plane tilt θ (°)", 0, 90, 54.7, 1.0)
phi_deg = st.sidebar.slider("Azimuth φ (°)", 0, 360, 0, 5)
theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)

# ------------------- Material (Ag) -------------------
C11 = 124e9; C12 = 93.4e9; C44 = 46.1e9
mu = C44
lam = C12 - 2*C44/3

# ------------------- Create identical defect in 2D & 3D -------------------
def create_defect_slice(N, dx):
    origin = -N * dx / 2
    X, Y, Z = np.meshgrid(
        np.linspace(origin, origin + (N-1)*dx, N),
        np.linspace(origin, origin + (N-1)*dx, N),
        np.linspace(origin, origin + (N-1)*dx, N),
        indexing='ij'
    )
    r = np.sqrt(X**2 + Y**2 + Z**2)
    R_np = N * dx / 4
    mask_np = r <= R_np

    # Planar defect on tilted {111}
    n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    dist = n[0]*X + n[1]*Y + n[2]*Z
    thickness = 3 * dx
    eta = np.zeros_like(X)
    eta[np.abs(dist) <= thickness/2] = 1.0
    eta[~mask_np] = 0.0
    eta += 0.02 * np.random.randn(*eta.shape)
    eta = np.clip(eta, 0, 1)

    # Extract central XY slice for 2D comparison
    slice_idx = N//2
    eta_2d = eta[:, :, slice_idx]

    return eta, eta_2d, X, Y, Z, mask_np

eta_3d, eta_2d, X, Y, Z, mask_np = create_defect_slice(N, dx_nm)

# ------------------- 3D Stress Solver (exact isotropic) -------------------
@st.cache_data
def compute_stress_3d(eta_3d, eps0, theta, phi, dx_nm):
    dx_m = dx_nm * 1e-9
    n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    s = np.cross(n, [0,0,1])
    if np.linalg.norm(s) < 1e-12:
        s = np.cross(n, [0,1,0])
    s = s / np.linalg.norm(s)

    eps_star = np.zeros((3,3,N,N,N))
    for i in range(3):
        for j in range(3):
            eps_star[i,j] = 0.5 * eps0 * 0.1 * (n[i]*s[j] + s[i]*n[j]) * eta_3d

    kx = 2*np.pi*np.fft.fftfreq(N, d=dx_m)
    ky = 2*np.pi*np.fft.fftfreq(N, d=dx_m)
    kz = 2*np.pi*np.fft.fftfreq(N, d=dx_m)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0,0,0] = 1.0

    trace = eps_star[0,0] + eps_star[1,1] + eps_star[2,2]
    Chat = np.zeros((3,3,N,N,N), dtype=complex)
    for i in range(3):
        for j in range(3):
            field = lam * trace * (i==j) + 2*mu * eps_star[i,j]
            Chat[i,j] = np.fft.fftn(field)

    sigma_hat = np.zeros_like(Chat)
    for i in range(3):
        for j in range(3):
            temp = 0
            for p in range(3):
                for q in range(3):
                    kp = [KX, KY, KZ][p]
                    kq = [KX, KY, KZ][q]
                    ki = [KX, KY, KZ][i]
                    kj = [KX, KY, KZ][j]
                    G = (ki*kj*kp*kq / K2) * (lam / (2*mu*(lam + 2*mu)))
                    G -= (ki*kp*kj*kq) / (2*mu*K2)
                    temp += G * Chat[p,q]
            sigma_hat[i,j] = -temp  # IMPORTANT: negative sign for stress!

    for i in range(3):
        for j in range(3):
            sigma_hat[i,j][0,0,0] = 0

    sigma_real = np.real(np.fft.ifftn(sigma_hat, axes=(2,3,4)))
    sxx = sigma_real[0,0]; syy = sigma_real[1,1]; szz = sigma_real[2,2]
    sxy = sigma_real[0,1]; sxz = sigma_real[0,2]; syz = sigma_real[1,2]

    von_mises_3d = np.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + sxz**2 + syz**2))) / 1e9
    sigma_mag_3d = np.sqrt(sxx**2 + syy**2 + szz**2 + 2*(sxy**2 + sxz**2 + syz**2)) / 1e9

    return von_mises_3d, sigma_mag_3d

# ------------------- 2D Plane-Strain Stress Solver -------------------
@st.cache_data
def compute_stress_2d(eta_2d, eps0, theta):
    # Same eigenstrain as 3D but projected to 2D
    n = np.array([np.cos(theta), np.sin(theta)])
    s = np.array([-np.sin(theta), np.cos(theta)])
    eps_xx = 0.5 * eps0 * 0.1 * 2 * n[0]*s[0] * eta_2d
    eps_yy = 0.5 * eps0 * 0.1 * 2 * n[1]*s[1] * eta_2d
    eps_xy = 0.5 * eps0 * 0.1 * (n[0]*s[1] + s[0]*n[1]) * eta_2d

    C11p = (C11 - C12**2/C11)
    C12p = C12*(C11-C12)/C11
    C44p = C44

    kx = np.fft.fftfreq(N, d=dx_nm)
    ky = np.fft.fftfreq(N, d=dx_nm)
    KX, KY = np.meshgrid(2*np.pi*kx, 2*np.pi*ky)
    K2 = KX**2 + KY**2 + 1e-20
    n1 = KX/np.sqrt(K2); n2 = KY/np.sqrt(K2)

    A11 = C11p*n1**2 + C44p*n2**2
    A22 = C11p*n2**2 + C44p*n1**2
    A12 = (C12p + C44p)*n1*n2
    det = A11*A22 - A12**2
    G11 = A22/det; G22 = A11/det; G12 = -A12/det

    tau_xx = C11p*eps_xx + C12p*eps_yy
    tau_yy = C12p*eps_xx + C11p*eps_yy
    tau_xy = 2*C44p*eps_xy

    txx = np.fft.fft2(tau_xx)
    tyy = np.fft.fft2(tau_yy)
    txy = np.fft.fft2(tau_xy)

    Sx = KX*txx + KY*txy
    Sy = KX*txy + KY*tyy

    ux = np.fft.ifft2(-1j*(G11*Sx + G12*Sy))
    uy = np.fft.ifft2(-1j*(G12*Sx + G22*Sy))

    exx = np.real(np.fft.ifft2(1j*KX* np.fft.fft2(ux)))
    eyy = np.real(np.fft.ifft2(1j*KY* np.fft.fft2(uy)))
    exy = 0.5*np.real(np.fft.ifft2(1j*(KX*np.fft.fft2(uy) + KY*np.fft.fft2(ux))))

    sxx = C11p*(exx - eps_xx) + C12p*(eyy - eps_yy)
    syy = C12p*(exx - eps_xx) + C11p*(eyy - eps_yy)
    sxy = 2*C44p*(exy - eps_xy)

    von_mises_2d = np.sqrt(sxx**2 + syy**2 - sxx*syy + 3*sxy**2) / 1e9
    sigma_mag_2d = np.sqrt(sxx**2 + syy**2 + 2*sxy**2) / 1e9

    return von_mises_2d, sigma_mag_2d

# ------------------- Run both solvers -------------------
if st.button("Run 2D vs 3D Comparison"):
    with st.spinner("Computing 3D stress..."):
        vm3, mag3 = compute_stress_3d(eta_3d, eps0, theta, phi, dx_nm)
    with st.spinner("Computing 2D stress..."):
        vm2, mag2 = compute_stress_2d(eta_2d, eps0, theta)

    slice_idx = N//2
    vm3_slice = vm3[:, :, slice_idx]
    mag3_slice = mag3[:, :, slice_idx]

    # ------------------- Results -------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Defect η (XY slice)")
        fig = go.Figure(data=go.Heatmap(z=eta_2d, colorscale="viridis"))
        fig.update_layout(height=500, title=f"η – θ={theta_deg:.1f}°")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("3D Solver (von Mises, GPa)")
        max3 = vm3_slice.max()
        st.metric("Max von Mises (3D)", f"{max3:.2f} GPa")
        fig3 = go.Figure(data=go.Heatmap(z=vm3_slice, colorscale="hot", zmin=0, zmax=max(max3, vm2.max())))
        fig3.update_layout(height=500, title=f"3D von Mises – Max {max3:.2f} GPa")
        st.plotly_chart(fig3, use_container_width=True)

    with col3:
        st.subheader("2D Plane-Strain Solver")
        max2 = vm2.max()
        st.metric("Max von Mises (2D)", f"{max2:.2f} GPa")
        fig2 = go.Figure(data=go.Heatmap(z=vm2, colorscale="hot", zmin=0, zmax=max(max3, max2)))
        fig2.update_layout(height=500, title=f"2D von Mises – Max {max2:.2f} GPa")
        st.plotly_chart(fig2, use_container_width=True)

    # Difference
    diff = vm3_slice - vm2
    st.subheader("Difference (3D − 2D)")
    st.write(f"Max absolute difference: {np.abs(diff).max():.2f} GPa")
    fig_diff = go.Figure(data=go.Heatmap(z=diff, colorscale="RdBu", zmid=0))
    fig_diff.update_layout(height=500, title="3D − 2D von Mises (GPa)")
    st.plotly_chart(fig_diff, use_container_width=True)

    st.success("Comparison complete! For planar defects on {111}, 2D and 3D should agree within ~10–20%.")
