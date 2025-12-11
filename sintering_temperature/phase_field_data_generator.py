import streamlit as st
import numpy as np
from numba import jit, prange
import torch
import torch.nn as nn
import pickle
from io import BytesIO
import sqlite3
import json
import hashlib
from datetime import datetime

# Material & Grid
N = 128
dx = 0.1  # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# Elastic constants for FCC Ag (GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

# Attention Interpolator
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(3, num_heads * d_head, bias=False)
        self.W_k = nn.Linear(3, num_heads * d_head, bias=False)

    def normalize_params(self, params, is_target=False):
        if is_target:
            eps0, kappa, theta = params
            return np.array([
                (eps0 - 0.3) / (3.0 - 0.3),
                (kappa - 0.1) / (2.0 - 0.1),
                (theta + np.pi) / (2 * np.pi)
            ])
        else:
            p = np.array(params)
            return np.stack([
                (p[:, 0] - 0.3) / (3.0 - 0.3),
                (p[:, 1] - 0.1) / (2.0 - 0.1),
                (p[:, 2] + np.pi) / (2 * np.pi)
            ], axis=1)

    def compute_weights(self, params_list, eps0_target, kappa_target, theta_target):
        norm_sources = self.normalize_params(params_list)
        norm_target = self.normalize_params((eps0_target, kappa_target, theta_target), is_target=True)
        src_tensor = torch.tensor(norm_sources, dtype=torch.float32)
        tgt_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)
        q = self.W_q(tgt_tensor).view(1, self.num_heads, self.d_head)
        k = self.W_k(src_tensor).view(len(params_list), self.num_heads, self.d_head)
        attn_logits = torch.einsum('nhd,mhd->nmh', q, k) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=1).mean(dim=2).squeeze(0)
        dists = torch.sqrt(
            ((src_tensor[:, 0] - norm_target[0]) / self.sigma)**2 +
            ((src_tensor[:, 1] - norm_target[1]) / self.sigma)**2 +
            ((src_tensor[:, 2] - norm_target[2]) / self.sigma)**2
        )
        spatial_weights = torch.exp(-dists**2 / 2)
        spatial_weights /= spatial_weights.sum() + 1e-8
        combined = attn_weights * spatial_weights
        combined /= combined.sum() + 1e-8
        return combined.detach().numpy()

# Simulation Database
class SimulationDB:
    @staticmethod
    def generate_id(sim_params):
        param_str = json.dumps(sim_params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]

    @staticmethod
    def save_simulation(sim_params, history, metadata):
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        sim_id = SimulationDB.generate_id(sim_params)
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
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            return st.session_state.simulations[sim_id]
        return None

    @staticmethod
    def get_simulation_list():
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

    @staticmethod
    def export_to_pkl(sim_id):
        sim = SimulationDB.get_simulation(sim_id)
        if sim:
            export_data = {
                'params': sim['params'],
                'metadata': sim['metadata'],
                'history': [
                    {'eta': eta, 'stress_fields': stress_fields}
                    for eta, stress_fields in sim['history']
                ]
            }
            buffer = BytesIO()
            pickle.dump(export_data, buffer)
            buffer.seek(0)
            return buffer.getvalue()
        return None

    @staticmethod
    def export_to_pt(sim_id):
        sim = SimulationDB.get_simulation(sim_id)
        if sim:
            export_data = {
                'params': sim['params'],
                'metadata': sim['metadata'],
                'history': [
                    {'eta': torch.tensor(eta), 'stress_fields': {k: torch.tensor(v) for k, v in stress_fields.items()}}
                    for eta, stress_fields in sim['history']
                ]
            }
            buffer = BytesIO()
            torch.save(export_data, buffer)
            buffer.seek(0)
            return buffer.getvalue()
        return None

    @staticmethod
    def export_to_db(sim_id):
        sim = SimulationDB.get_simulation(sim_id)
        if sim:
            conn = sqlite3.connect(':memory:')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS simulations
                         (sim_id TEXT, params TEXT, frame_idx INTEGER, eta BLOB, hydrostatic BLOB, magnitude BLOB, von_mises BLOB)''')
            params_json = json.dumps(sim['params'])
            for idx, (eta, stress_fields) in enumerate(sim['history']):
                c.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (sim_id, params_json, idx,
                           eta.tobytes(),
                           stress_fields['sigma_hydro'].tobytes(),
                           stress_fields['sigma_mag'].tobytes(),
                           stress_fields['von_mises'].tobytes()))
            buffer = BytesIO()
            for line in conn.iterdump():
                buffer.write(line.encode() + b'\n')
            buffer.seek(0)
            return buffer.getvalue()
        return None

# Simulation Engine
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

# Streamlit UI
st.sidebar.header("Operation Mode")
operation_mode = st.sidebar.radio("Mode", ["Run Simulation", "Interpolate"])

if operation_mode == "Run Simulation":
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
    shape = st.sidebar.selectbox("Shape", ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])
    eps0 = st.sidebar.slider("ε*", 0.3, 3.0, default_eps)
    kappa = st.sidebar.slider("κ", 0.1, 2.0, default_kappa)
    steps = st.sidebar.slider("Steps", 20, 200, 100)
    save_every = st.sidebar.slider("Save Every", 10, 50, 20)
    orientation = st.sidebar.selectbox("Orientation", ["Horizontal {111} (0°)", "Tilted 30°", "Tilted 60°", "Vertical {111} (90°)", "Custom Angle"])
    if orientation == "Custom Angle":
        angle_deg = st.sidebar.slider("Angle (°)", -180, 180, 0)
    else:
        angle_map = {"Horizontal {111} (0°)": 0, "Tilted 30°": 30, "Tilted 60°": 60, "Vertical {111} (90°)": 90}
        angle_deg = angle_map[orientation]
    theta = np.deg2rad(angle_deg)

    if st.button("Run Simulation"):
        sim_params = {'defect_type': defect_type, 'shape': shape, 'eps0': eps0, 'kappa': kappa, 'theta': theta, 'steps': steps, 'save_every': save_every, 'orientation': orientation}
        history = run_simulation(sim_params)
        metadata = {'run_time': time.time(), 'frames': len(history)}
        sim_id = SimulationDB.save_simulation(sim_params, history, metadata)
        st.success(f"Saved simulation ID: {sim_id}")

        col1, col2, col3 = st.columns(3)
        with col1:
            pkl_data = SimulationDB.export_to_pkl(sim_id)
            st.download_button("Download .pkl", pkl_data, f"sim_{sim_id}.pkl")
        with col2:
            pt_data = SimulationDB.export_to_pt(sim_id)
            st.download_button("Download .pt", pt_data, f"sim_{sim_id}.pt")
        with col3:
            db_data = SimulationDB.export_to_db(sim_id)
            st.download_button("Download .db", db_data, "simulations.db")

else:  # Interpolate
    st.sidebar.header("Attention Params")
    sigma = st.sidebar.slider("σ", 0.05, 0.50, 0.20)
    num_heads = st.sidebar.slider("Heads", 1, 8, 4)
    d_head = st.sidebar.slider("Dim/Head", 4, 16, 8)

    simulations = SimulationDB.get_simulation_list()
    selected_sources = st.multiselect("Sources", [sim['name'] for sim in simulations])
    source_ids = [simulations[i]['id'] for i in range(len(simulations)) if simulations[i]['name'] in selected_sources]
    sources = [SimulationDB.get_simulation(sid) for sid in source_ids]

    defect_type_target = st.selectbox("Target Defect", ["ISF", "ESF", "Twin"])
    shape_target = st.selectbox("Target Shape", ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])
    eps0_target = st.slider("Target ε*", 0.3, 3.0, 1.0)
    kappa_target = st.slider("Target κ", 0.1, 2.0, 0.5)
    orientation_target = st.selectbox("Target Orientation", ["Horizontal {111} (0°)", "Tilted 30°", "Tilted 60°", "Vertical {111} (90°)", "Custom Angle"])
    if orientation_target == "Custom Angle":
        angle_deg_target = st.slider("Target Angle (°)", -180, 180, 0)
    else:
        angle_map = {"Horizontal {111} (0°)": 0, "Tilted 30°": 30, "Tilted 60°": 60, "Vertical {111} (90°)": 90}
        angle_deg_target = angle_map[orientation_target]
    theta_target = np.deg2rad(angle_deg_target)

    if st.button("Interpolate"):
        params_list = [(s['params']['eps0'], s['params']['kappa'], s['params']['theta']) for s in sources]
        interpolator = MultiParamAttentionInterpolator(sigma, num_heads, d_head)
        weights = interpolator.compute_weights(params_list, eps0_target, kappa_target, theta_target)
        interpolated = {'sigma_hydro': np.zeros((N, N)), 'sigma_mag': np.zeros((N, N)), 'von_mises': np.zeros((N, N))}
        for w, source in zip(weights, sources):
            final_stress = source['history'][-1][1]
            for key in interpolated:
                interpolated[key] += w * final_stress[key]
        st.success("Interpolation complete")

        interpolated_data = {'params': {'eps0': eps0_target, 'kappa': kappa_target, 'theta': theta_target}, 'fields': interpolated}
        col1, col2, col3 = st.columns(3)
        with col1:
            pkl_buffer = BytesIO()
            pickle.dump(interpolated_data, pkl_buffer)
            st.download_button("Download .pkl", pkl_buffer.getvalue(), "interpolated.pkl")
        with col2:
            pt_buffer = BytesIO()
            torch.save(interpolated_data, pt_buffer)
            st.download_button("Download .pt", pt_buffer.getvalue(), "interpolated.pt")
        with col3:
            conn = sqlite3.connect(':memory:')
            df = pd.DataFrame({'key': list(interpolated.keys()), 'data': [v.tobytes() for v in interpolated.values()]})
            df.to_sql('interpolated', conn)
            db_buffer = BytesIO()
            for line in conn.iterdump():
                db_buffer.write(line.encode() + b'\n')
            st.download_button("Download .db", db_buffer.getvalue(), "interpolated.db")
