import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import time
import hashlib
import json
from datetime import datetime
import pickle
import torch
import sqlite3

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

@st.cache_data
def compute_stress_fields(eta, eps0, theta):
    # Plane-strain reduced constants (Pa)
    C11_p = (C11 - C12**2 / C11) * 1e9
    C12_p = (C12 - C12**2 / C11) * 1e9
    C44_p = C44 * 1e9
   
    # Wavevectors
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
   
    # Acoustic tensor components
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
   
    # Eigenstrain (rotated)
    gamma = eps0
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    delta = 0.02 # Small dilatation
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
   
    eps_xx_star = eps_star[0,0] * eta
    eps_yy_star = eps_star[1,1] * eta
    eps_xy_star = eps_star[0,1] * eta
   
    # Polarization stress tau = C : eps*
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
   
    # Displacements
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
   
    # Elastic strains
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
   
    # Elastic stresses (Pa → GPa)
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy) # Plane strain approximation
   
    # Derived quantities (GPa)
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
   
    return {
        'sxx': sxx, 'syy': syy, 'sxy': sxy, 'szz': szz,
        'sigma_mag': sigma_mag, 'sigma_hydro': sigma_hydro, 'von_mises': von_mises
    }

def run_simulation(sim_params):
    """Run a complete simulation with given parameters"""
    # Create initial defect
    eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
   
    # Run evolution
    history = []
    for step in range(sim_params['steps'] + 1):
        if step > 0:
            eta = evolve_phase_field(eta, sim_params['kappa'], dt=0.004, dx=dx, N=N)
        if step % sim_params['save_every'] == 0 or step == sim_params['steps']:
            stress_fields = compute_stress_fields(eta, sim_params['eps0'], sim_params['theta'])
            history.append((eta.copy(), stress_fields))
   
    return history

# =============================================
# SAVE TO FORMATS
# =============================================
def save_to_formats(sim_id, sim_params, history, metadata):
    data = {'params': sim_params, 'history': [], 'metadata': metadata}
    for eta, stress_fields in history:
        data['history'].append({
            'eta': eta,
            'stresses': stress_fields
        })

    # Save .pkl
    with open(f'sim_{sim_id}.pkl', 'wb') as f:
        pickle.dump(data, f)

    # Save .pt (convert to tensors)
    tensor_data = data.copy()
    for frame in tensor_data['history']:
        frame['eta'] = torch.from_numpy(frame['eta'])
        for k, v in frame['stresses'].items():
            frame['stresses'][k] = torch.from_numpy(v)
    torch.save(tensor_data, f'sim_{sim_id}.pt')

    # Save .db
    conn = sqlite3.connect('simulations.db')
    c = conn.cursor()
    # Create tables if not exist
    c.execute('''CREATE TABLE IF NOT EXISTS simulations (
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
    c.execute('''CREATE TABLE IF NOT EXISTS frames (
                 sim_id TEXT,
                 frame_idx INTEGER,
                 eta BLOB,
                 sxx BLOB,
                 syy BLOB,
                 sxy BLOB,
                 szz BLOB,
                 sigma_mag BLOB,
                 sigma_hydro BLOB,
                 von_mises BLOB,
                 FOREIGN KEY (sim_id) REFERENCES simulations(id)
                 )''')
    # Insert params/metadata
    c.execute("INSERT OR REPLACE INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (sim_id, sim_params['defect_type'], sim_params['shape'], sim_params['orientation'],
               sim_params['theta'], sim_params['eps0'], sim_params['kappa'], sim_params['steps'],
               metadata['created_at'], metadata['grid_size'], metadata['dx']))
    # Insert frames
    for idx, frame in enumerate(data['history']):
        c.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (sim_id, idx,
                   pickle.dumps(frame['eta']),
                   pickle.dumps(frame['stresses']['sxx']),
                   pickle.dumps(frame['stresses']['syy']),
                   pickle.dumps(frame['stresses']['sxy']),
                   pickle.dumps(frame['stresses']['szz']),
                   pickle.dumps(frame['stresses']['sigma_mag']),
                   pickle.dumps(frame['stresses']['sigma_hydro']),
                   pickle.dumps(frame['stresses']['von_mises'])))
    conn.commit()
    conn.close()

# =============================================
# SIMULATION DATABASE SYSTEM (Session State)
# =============================================
class SimulationDB:
    """In-memory simulation database for storing and retrieving simulations"""
   
    @staticmethod
    def generate_id(sim_params):
        """Generate unique ID for simulation"""
        param_str = json.dumps(sim_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
   
    @staticmethod
    def save_simulation(sim_params, history, metadata):
        """Save simulation to database"""
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
       
        sim_id = SimulationDB.generate_id(sim_params)
       
        # Store simulation data
        st.session_state.simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'history': history,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
       
        return sim_id
   
    @staticmethod
    def get_simulation(sim_id):
        """Retrieve simulation by ID"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            return st.session_state.simulations[sim_id]
        return None
   
    @staticmethod
    def get_all_simulations():
        """Get all stored simulations"""
        if 'simulations' in st.session_state:
            return st.session_state.simulations
        return {}
   
    @staticmethod
    def delete_simulation(sim_id):
        """Delete simulation from database"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            del st.session_state.simulations[sim_id]
            return True
        return False
   
    @staticmethod
    def get_simulation_list():
        """Get list of simulations for dropdown"""
        if 'simulations' not in st.session_state:
            return []
       
        simulations = []
        for sim_id, sim_data in st.session_state.simulations.items():
            params = sim_data['params']
            name = f"{params['defect_type']} - {params['orientation']} (ε*={params['eps0']:.2f}, κ={params['kappa']:.2f})"
            simulations.append({
                'id': sim_id,
                'name': name,
                'params': params
            })
       
        return simulations

# =============================================
# SIDEBAR - Global Settings
# =============================================
st.sidebar.header("🎛️ Simulation Setup")

defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])

# Physical eigenstrain values
if defect_type == "ISF":
    default_eps = 0.707
    default_kappa = 0.6
    init_amplitude = 0.70
elif defect_type == "ESF":
    default_eps = 1.414
    default_kappa = 0.7
    init_amplitude = 0.75
else: # Twin
    default_eps = 2.121
    default_kappa = 0.3
    init_amplitude = 0.90

shape = st.sidebar.selectbox("Initial Seed Shape",
    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])

eps0 = st.sidebar.slider(
    "Eigenstrain magnitude ε*",
    0.3, 3.0,
    value=default_eps,
    step=0.01
)

kappa = st.sidebar.slider(
    "Interface energy coeff κ",
    0.1, 2.0,
    value=default_kappa,
    step=0.05
)

steps = st.sidebar.slider("Evolution steps", 20, 200, 100, 10)
save_every = st.sidebar.slider("Save frame every", 10, 50, 20)

# Crystal Orientation
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
if st.sidebar.button("🚀 Run & Save Simulation", type="primary"):
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
            'dx': dx
        }
        sim_id = SimulationDB.save_simulation(sim_params, history, metadata)
        
        # Save to formats
        save_to_formats(sim_id, sim_params, history, metadata)
        
        st.success(f"Simulation saved with ID: {sim_id}")

# =============================================
# MAIN CONTENT - VISUALIZATION (STRESS HEATMAPS ONLY)
# =============================================
st.header("📊 Simulation Results")

simulations = SimulationDB.get_simulation_list()

if simulations:
    selected_sim = st.selectbox("Select Simulation", [sim['name'] for sim in simulations])
    sim_id = [sim['id'] for sim in simulations if sim['name'] == selected_sim][0]
    sim_data = SimulationDB.get_simulation(sim_id)
    
    if sim_data:
        # Display final stress heatmaps
        final_eta, final_stress = sim_data['history'][-1]
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(final_stress['sigma_mag'], extent=extent, cmap='viridis', origin='lower', aspect='equal')
        axs[0].set_title('Stress Magnitude')
        
        axs[1].imshow(final_stress['sigma_hydro'], extent=extent, cmap='coolwarm', origin='lower', aspect='equal')
        axs[1].set_title('Hydrostatic Stress')
        
        axs[2].imshow(final_stress['von_mises'], extent=extent, cmap='plasma', origin='lower', aspect='equal')
        axs[2].set_title('Von Mises Stress')
        
        st.pyplot(fig)
else:
    st.info("No simulations run yet.")
