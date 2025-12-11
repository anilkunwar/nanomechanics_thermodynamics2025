import streamlit as st
import numpy as np
from numba import jit, prange
import time
import hashlib
import json
from datetime import datetime
import pickle
import torch
import sqlite3
import pandas as pd
from io import BytesIO, StringIO
import zipfile

# Configure page
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Analyzer")

# Material & Grid
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1
N = 128
dx = 0.1 # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# SIMULATION ENGINE
# =============================================
def create_initial_eta(shape, defect_type):
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes[defect_type]
   
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    w, h = (24, 12) if shape in ["Rectangle", "Horizontal Fault"] else (16, 16)
   
    if shape == "Square":
        eta[cy-h:cy+h, cx-h:cx+h] = init_amplitude
    elif shape == "Horizontal Fault":
        eta[cy-4:cy+4, cx-w:cx+w] = init_amplitude
    elif shape == "Vertical Fault":
        eta[cy-w:cy+w, cx-4:cx+4] = init_amplitude
    elif shape == "Rectangle":
        eta[cy-h:cy+h, cx-w:cx+w] = init_amplitude
    elif shape == "Ellipse":
        mask = ((X/(w*1.5))**2 + (Y/(h*1.5))**2) <= 1
        eta[mask] = init_amplitude
   
    eta += 0.02 * np.random.randn(N, N)
    return np.clip(eta, 0.0, 1.0)

@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx2
            dF = 2*eta[i,j]*(1-eta[i,j])*(eta[i,j]-0.5)
            eta_new[i,j] = eta[i,j] + dt * (-dF + kappa * lap)
            eta_new[i,j] = np.maximum(0.0, np.minimum(1.0, eta_new[i,j]))
    eta_new[0,:] = eta_new[-2,:]; eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]; eta_new[:,-1] = eta_new[:,1]
    return eta_new

def compute_stress_fields(eta, eps0, theta):
    C11_p = (C11 - C12**2 / C11) * 1e9
    C12_p = (C12 - C12**2 / C11) * 1e9
    C44_p = C44 * 1e9
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(2 * np.pi * kx, 2 * np.pi * ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-12
    mask = K2 > 0
    n1 = np.zeros_like(KX)
    n2 = np.zeros_like(KX)
    n1[mask] = KX[mask] / np.sqrt(K2[mask])
    n2[mask] = KY[mask] / np.sqrt(K2[mask])
    A11 = np.zeros_like(KX)
    A22 = np.zeros_like(KX)
    A12 = np.zeros_like(KX)
    A11[mask] = C11_p * n1[mask]**2 + C44_p * n2[mask]**2
    A22[mask] = C11_p * n2[mask]**2 + C44_p * n1[mask]**2
    A12[mask] = (C12_p + C44_p) * n1[mask] * n2[mask]
    det = A11 * A22 - A12**2
    G11 = np.zeros_like(KX)
    G22 = np.zeros_like(KX)
    G12 = np.zeros_like(KX)
    G11[mask] = A22[mask] / det[mask]
    G22[mask] = A11[mask] / det[mask]
    G12[mask] = -A12[mask] / det[mask]
    gamma = eps0
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    delta = 0.02
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
    eps_xx_star = eps_star[0,0] * eta
    eps_yy_star = eps_star[1,1] * eta
    eps_xy_star = eps_star[0,1] * eta
    tau_xx = C11_p * eps_xx_star + C12_p * eps_yy_star
    tau_yy = C12_p * eps_xx_star + C11_p * eps_yy_star
    tau_xy = 2 * C44_p * eps_xy_star
    tau_hat_xx = np.fft.fft2(tau_xx)
    tau_hat_yy = np.fft.fft2(tau_yy)
    tau_hat_xy = np.fft.fft2(tau_xy)
    S_hat_x = KX * tau_hat_xx + KY * tau_hat_xy
    S_hat_y = KX * tau_hat_xy + KY * tau_hat_yy
    u_hat_x = np.zeros_like(KX, dtype=complex)
    u_hat_y = np.zeros_like(KX, dtype=complex)
    u_hat_x[mask] = -1j * (G11[mask] * S_hat_x[mask] + G12[mask] * S_hat_y[mask])
    u_hat_y[mask] = -1j * (G12[mask] * S_hat_x[mask] + G22[mask] * S_hat_y[mask])
    u_hat_x[0, 0] = 0
    u_hat_y[0, 0] = 0
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy)
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
    return {
        'sxx': sxx, 'syy': syy, 'sxy': sxy, 'szz': szz,
        'sigma_mag': sigma_mag, 'sigma_hydro': sigma_hydro, 'von_mises': von_mises
    }

def run_simulation(sim_params):
    eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
    history = []
    for step in range(sim_params['steps'] + 1):
        if step > 0:
            eta = evolve_phase_field(eta, sim_params['kappa'], dt=0.004, dx=dx, N=N)
        if step % sim_params['save_every'] == 0 or step == sim_params['steps']:
            stress_fields = compute_stress_fields(eta, sim_params['eps0'], sim_params['theta'])
            history.append((eta.copy(), stress_fields))
    return history

# =============================================
# SIDEBAR - Simulation Setup
# =============================================
st.sidebar.header("🎛️ Simulation Setup")
defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])
if defect_type == "ISF":
    default_eps = 0.707
    default_kappa = 0.6
elif defect_type == "ESF":
    default_eps = 1.414
    default_kappa = 0.7
else:
    default_eps = 2.121
    default_kappa = 0.3
shape = st.sidebar.selectbox("Initial Seed Shape",
    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])
eps0 = st.sidebar.slider("Eigenstrain magnitude ε*", 0.3, 3.0, default_eps, 0.01)
kappa = st.sidebar.slider("Interface energy coeff κ", 0.1, 2.0, default_kappa, 0.05)
steps = st.sidebar.slider("Evolution steps", 20, 200, 100, 10)
save_every = st.sidebar.slider("Save frame every", 10, 50, 20)
st.sidebar.subheader("Crystal Orientation")
orientation = st.sidebar.selectbox(
    "Habit Plane Orientation",
    ["Horizontal {111} (0°)",
     "Tilted 30° (1¯10 projection)",
     "Tilted 60°",
     "Vertical {111} (90°)",
     "Custom Angle"],
    index=0
)
if orientation == "Custom Angle":
    angle_deg = st.sidebar.slider("Custom tilt angle (°)", -180, 180, 0, 5)
    theta = np.deg2rad(angle_deg)
else:
    angle_map = {
        "Horizontal {111} (0°)": 0,
        "Tilted 30° (1¯10 projection)": 30,
        "Tilted 60°": 60,
        "Vertical {111} (90°)": 90,
    }
    theta = np.deg2rad(angle_map[orientation])

# Run button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    sim_params = {
        'defect_type': defect_type,
        'shape': shape,
        'eps0': eps0,
        'kappa': kappa,
        'orientation': orientation,
        'theta': theta,
        'steps': steps,
        'save_every': save_every
    }
    with st.spinner(f"Running {defect_type} simulation..."):
        start_time = time.time()
        history = run_simulation(sim_params)
        metadata = {
            'run_time': time.time() - start_time,
            'frames': len(history),
            'grid_size': N,
            'dx': dx,
            'created_at': datetime.now().isoformat()
        }
        # Generate sim_id
        param_str = json.dumps(sim_params, sort_keys=True)
        sim_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Save to session state
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        st.session_state.simulations[sim_id] = {
            'params': sim_params,
            'history': history,
            'metadata': metadata
        }
        
        st.success(f"Simulation Complete! ID: {sim_id}")
        
# Download section
st.header("📥 Download Files")
simulations = st.session_state.get('simulations', {})
if simulations:
    selected_sim_id = st.selectbox("Select Simulation ID", list(simulations.keys()))
    if selected_sim_id:
        sim_data = simulations[selected_sim_id]
        history = sim_data['history']
        sim_params = sim_data['params']
        metadata = sim_data['metadata']
        
        data = {'params': sim_params, 'history': [], 'metadata': metadata}
        for eta, stress_fields in history:
            data['history'].append({
                'eta': eta,
                'stresses': stress_fields
            })
        
        # PKL
        pkl_buffer = BytesIO()
        pickle.dump(data, pkl_buffer)
        pkl_buffer.seek(0)
        st.download_button("Download PKL", pkl_buffer, f"sim_{selected_sim_id}.pkl")
        
        # PT
        pt_buffer = BytesIO()
        tensor_data = data.copy()
        for frame in tensor_data['history']:
            frame['eta'] = torch.from_numpy(frame['eta'])
            for k, v in frame['stresses'].items():
                frame['stresses'][k] = torch.from_numpy(v)
        torch.save(tensor_data, pt_buffer)
        pt_buffer.seek(0)
        st.download_button("Download PT", pt_buffer, f"sim_{selected_sim_id}.pt")
        
        # DB (in-memory)
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE simulations (
                     id TEXT PRIMARY KEY,
                     defect_type TEXT,
                     shape TEXT,
                     orientation TEXT,
                     theta REAL,
                     eps0 REAL,
                     kappa REAL,
                     steps INTEGER,
                     created_at TEXT,
                     grid_size INTEGER,
                     dx REAL
                     )''')
        c.execute('''CREATE TABLE frames (
                     sim_id TEXT,
                     frame_idx INTEGER,
                     eta BLOB,
                     sxx BLOB,
                     syy BLOB,
                     sxy BLOB,
                     szz BLOB,
                     sigma_mag BLOB,
                     sigma_hydro BLOB,
                     von_mises BLOB
                     )''')
        c.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (selected_sim_id, sim_params['defect_type'], sim_params['shape'],
                   sim_params['orientation'], sim_params['theta'],
                   sim_params['eps0'], sim_params['kappa'], sim_params['steps'],
                   metadata['created_at'], metadata['grid_size'], metadata['dx']))
        for idx, frame in enumerate(data['history']):
            c.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (selected_sim_id, idx,
                       pickle.dumps(frame['eta']),
                       pickle.dumps(frame['stresses']['sxx']),
                       pickle.dumps(frame['stresses']['syy']),
                       pickle.dumps(frame['stresses']['sxy']),
                       pickle.dumps(frame['stresses']['szz']),
                       pickle.dumps(frame['stresses']['sigma_mag']),
                       pickle.dumps(frame['stresses']['sigma_hydro']),
                       pickle.dumps(frame['stresses']['von_mises'])))
        conn.commit()
        
        db_buffer = BytesIO()
        for line in conn.iterdump():
            db_buffer.write(('%s\n' % line).encode('utf-8'))
        db_buffer.seek(0)
        st.download_button("Download DB (SQL Dump)", db_buffer, f"sim_{selected_sim_id}.db.sql")
        
        # CSV (zip multiple)
        csv_zip_buffer = BytesIO()
        with zipfile.ZipFile(csv_zip_buffer, "w") as zf:
            for idx, (eta, stress_fields) in enumerate(history):
                df = pd.DataFrame({
                    'x': X.flatten(),
                    'y': Y.flatten(),
                    'eta': eta.flatten(),
                    'sxx': stress_fields['sxx'].flatten(),
                    'syy': stress_fields['syy'].flatten(),
                    'sxy': stress_fields['sxy'].flatten(),
                    'szz': stress_fields['szz'].flatten(),
                    'sigma_mag': stress_fields['sigma_mag'].flatten(),
                    'sigma_hydro': stress_fields['sigma_hydro'].flatten(),
                    'von_mises': stress_fields['von_mises'].flatten()
                })
                csv_str = df.to_csv(index=False)
                zf.writestr(f"frame_{idx}.csv", csv_str)
        csv_zip_buffer.seek(0)
        st.download_button("Download CSV (Zip)", csv_zip_buffer, f"sim_{selected_sim_id}_csv.zip")
else:
    st.info("No simulations available.")
