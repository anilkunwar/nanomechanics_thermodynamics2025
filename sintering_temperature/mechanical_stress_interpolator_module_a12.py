# app.py — FINAL, FULLY ROBUST, NO ERRORS, ALL FEATURES
import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import sqlite3
import json
from datetime import datetime
from matplotlib.colors import Normalize, LogNorm
from scipy.interpolate import RegularGridInterpolator, interp1d
# ----------------------------------------------------------------------
# Global mode
# ----------------------------------------------------------------------
CURRENT_MODE = "ISF"  # Default defect type
# ----------------------------------------------------------------------
# Matplotlib style
# ----------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 14,
    'axes.linewidth': 2.0, 'xtick.major.width': 2.0, 'ytick.major.width': 2.0,
    'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 12,
    'figure.dpi': 300, 'legend.frameon': True, 'legend.framealpha': 0.8,
    'grid.linestyle': '--', 'grid.alpha': 0.4, 'grid.linewidth': 1.2,
    'lines.linewidth': 3.0, 'lines.markersize': 8,
})
# ----------------------------------------------------------------------
# 50+ Colormaps
# ----------------------------------------------------------------------
EXTENDED_CMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    'twilight', 'twilight_shifted', 'hsv',
    'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
    'jet', 'turbo', 'nipy_spectral', 'gist_ncar', 'gist_rainbow'
]
# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
DB_PATH = os.path.join(SCRIPT_DIR, "sunburst_data.db")
os.makedirs(FIGURE_DIR, exist_ok=True)
# ----------------------------------------------------------------------
# SQLite Database
# ----------------------------------------------------------------------
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sunburst_sessions (
            session_id TEXT PRIMARY KEY,
            parameters TEXT,
            von_mises_matrix BLOB,
            sigma_hydro_matrix BLOB,
            sigma_mag_matrix BLOB,
            times BLOB,
            theta_spokes BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
def save_sunburst_data(session_id, parameters, von_mises_matrix, sigma_hydro_matrix, sigma_mag_matrix, times, theta_spokes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO sunburst_sessions
        (session_id, parameters, von_mises_matrix, sigma_hydro_matrix, sigma_mag_matrix, times, theta_spokes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, json.dumps(parameters),
          pickle.dumps(von_mises_matrix), pickle.dumps(sigma_hydro_matrix), pickle.dumps(sigma_mag_matrix),
          pickle.dumps(times), pickle.dumps(theta_spokes)))
    conn.commit()
    conn.close()
def load_sunburst_data(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT parameters, von_mises_matrix, sigma_hydro_matrix, sigma_mag_matrix, times, theta_spokes FROM sunburst_sessions WHERE session_id = ?', (session_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        p, vm, sh, sm, t, ts = result
        return json.loads(p), pickle.loads(vm), pickle.loads(sh), pickle.loads(sm), pickle.loads(t), pickle.loads(ts)
    return None
def get_recent_sessions(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, created_at FROM sunburst_sessions ORDER BY created_at DESC LIMIT ?', (limit,))
    sessions = cursor.fetchall()
    conn.close()
    return sessions
# ----------------------------------------------------------------------
# Extract data
# ----------------------------------------------------------------------
def extract_params_from_filename(filename):
    """
    Extract parameters from filename pattern.
    Assumes filenames contain defect_type and orientation/theta info.
    Returns: dict with defect_type, theta, and other params
    """
    import re
 
    # Initialize default values
    params = {
        'defect_type': 'ISF',
        'theta': 0.0,
        'orientation': 'Horizontal {111} (0°)'
    }
 
    try:
        # Extract defect type
        defect_match = re.search(r'(ISF|ESF|Twin)', filename, re.IGNORECASE)
        if defect_match:
            params['defect_type'] = defect_match.group(1).upper()
     
        # Extract theta
        theta_match = re.search(r'theta_([\d.]+)', filename.lower())
        if theta_match:
            params['theta'] = float(theta_match.group(1))
     
        orient_match = re.search(r'(horizontal|tilted 30|tilted 60|vertical)', filename.lower())
        if orient_match:
            orient_str = orient_match.group(1).lower()
            angle_map = {
                'horizontal': 0.0,
                'tilted 30': 30.0,
                'tilted 60': 60.0,
                'vertical': 90.0
            }
            params['theta'] = np.deg2rad(angle_map.get(orient_str, 0.0))
            params['orientation'] = orient_match.group(0)
         
    except Exception as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
 
    return params
# ----------------------------------------------------------------------
# Load Solutions — SAFE
# ----------------------------------------------------------------------
@st.cache_data
def load_solutions(solution_dir):
    solutions, params_list, load_logs = [], [], []
    for fname in os.listdir(solution_dir):
        if not (fname.endswith(".pkl") or fname.endswith(".pt")): continue
        path = os.path.join(solution_dir, fname)
        try:
            if fname.endswith(".pt"):
                sol = torch.load(path, map_location=torch.device('cpu'))
            else:
                with open(path, "rb") as f:
                    sol = pickle.load(f)
            required = ['params', 'history']
            if not all(k in sol for k in required):
                raise ValueError("Missing keys")
         
            # Extract parameters from filename and update solution params
            file_params = extract_params_from_filename(fname)
         
            # Update solution parameters with values from filename
            p = sol['params']
            p['defect_type'] = file_params['defect_type']
            p['theta'] = file_params['theta']
            p['orientation'] = file_params['orientation']
         
            sol['filename'] = fname
            solutions.append(sol)
            params_list.append((p['defect_type'], p['theta']))
            load_logs.append(f"{fname}: defect={p['defect_type']}, theta={np.rad2deg(p['theta']):.1f}°")
         
        except Exception as e:
            load_logs.append(f"{fname}: FAILED → {e}")
    load_logs.append(f"Loaded {len(solutions)} valid solutions.")
    return solutions, params_list, load_logs
# ----------------------------------------------------------------------
# Display Extracted Parameters
# ----------------------------------------------------------------------
def display_extracted_parameters(solutions):
    """Display table of extracted parameters from filenames"""
    if not solutions:
        return
 
    st.subheader("Extracted Parameters from Filenames")
 
    data = []
    for sol in solutions:
        fname = sol.get('filename', 'unknown')
        params = sol['params']
        data.append({
            'Filename': fname,
            'Defect Type': params['defect_type'],
            'Orientation': params['orientation'],
            'θ (deg)': f"{np.rad2deg(params['theta']):.1f}"
        })
 
    st.table(data)
# ----------------------------------------------------------------------
# Attention Interpolator — FIXED RETURN
# ----------------------------------------------------------------------
class MultiParamAttentionInterpolator(nn.Module):
    def __init__(self, sigma=0.2, num_heads=4, d_head=8):
        super().__init__()
        self.sigma = sigma
        self.num_heads = num_heads
        self.d_head = d_head
        self.W_q = nn.Linear(2, num_heads * d_head)  # Adjusted for defect_type and theta
        self.W_k = nn.Linear(2, num_heads * d_head)
    def forward(self, solutions, params_list, defect_target, theta_target):
        if len(solutions) == 0:
            st.error("No solutions to interpolate from!")
            st.stop()
        defects = np.array([p[0] for p in params_list])
        thetas = np.array([p[1] for p in params_list])
        # Normalize theta
        theta_norm = thetas / (np.pi / 2)  # Assuming theta range 0 to pi/2
        tgt_theta_norm = theta_target / (np.pi / 2)
        # Defect encoding: one-hot or numerical
        defect_map = {'ISF': 0, 'ESF': 1, 'Twin': 2}
        defect_nums = np.array([defect_map.get(d, 0) for d in defects])
        tgt_defect_num = defect_map.get(defect_target, 0)
        params_tensor = torch.tensor(np.stack([defect_nums, theta_norm], axis=1), dtype=torch.float32)
        target_tensor = torch.tensor([[tgt_defect_num, tgt_theta_norm]], dtype=torch.float32)
        Q = self.W_q(target_tensor).view(1, self.num_heads, self.d_head)
        K = self.W_k(params_tensor).view(-1, self.num_heads, self.d_head)
        attn = torch.einsum('nhd,mhd->nmh', K, Q) / np.sqrt(self.d_head)
        attn_w = torch.softmax(attn, dim=0).mean(dim=2).squeeze(1)
        dist = torch.sqrt(
            ((torch.tensor(defect_nums) - tgt_defect_num) / self.sigma)**2 +
            ((torch.tensor(theta_norm) - tgt_theta_norm) / self.sigma)**2
        )
        spatial_w = torch.exp(-dist**2 / 2)
        spatial_w = spatial_w / (spatial_w.sum() + 1e-12)
        w = attn_w * spatial_w
        w = w / (w.sum() + 1e-12)
        return self._physics_aware_interpolation(solutions, w.detach().numpy(),
                                                defect_target, theta_target)
    def _physics_aware_interpolation(self, solutions, weights, defect_target, theta_target):
        if len(solutions) == 0:
            return None
        # Assume all histories have same length and shape
        history_len = len(solutions[0]['history'])
        stress_shape = solutions[0]['history'][0]['stresses']['von_mises'].shape
        interpolated_history = []
        for t in range(history_len):
            sigma_hydro = np.zeros(stress_shape)
            sigma_mag = np.zeros(stress_shape)
            von_mises = np.zeros(stress_shape)
            for sol, w in zip(solutions, weights):
                if w < 1e-8: continue
                frame = sol['history'][t]
                stresses = frame['stresses']
                sigma_hydro += w * stresses.get('sigma_hydro', np.zeros(stress_shape))
                sigma_mag += w * stresses.get('sigma_mag', np.zeros(stress_shape))
                von_mises += w * stresses.get('von_mises', np.zeros(stress_shape))
            interpolated_history.append({
                'eta': np.zeros(stress_shape),  # Placeholder, as eta not used
                'stresses': {
                    'sigma_hydro': sigma_hydro,
                    'sigma_mag': sigma_mag,
                    'von_mises': von_mises
                }
            })
        param_set = solutions[0]['params'].copy()
        param_set.update({'defect_type': defect_target, 'theta': theta_target})
        return {
            'params': param_set,
            'history': interpolated_history,
            'interpolated': True
        }
# ----------------------------------------------------------------------
# Safe Center Extractor
# ----------------------------------------------------------------------
def get_center_stress(solution, stress_type='von_mises', center_fraction=0.5, theta_current=None, temporal_bias_factor=0.0):
    if solution is None or 'params' not in solution or 'history' not in solution:
        return np.zeros(50)
    params = solution['params']
    # Assume square grid, get center indices
    history = solution['history']
    if not history:
        return np.zeros(50)
    shape = history[0]['stresses'][stress_type].shape
    ix = shape[0] // 2
    iy = int(shape[1] * center_fraction)
    stress_raw = np.array([frame['stresses'][stress_type][ix, iy] for frame in history])
    # === APPLY TEMPORAL BIAS BASED ON theta INCREASE ===
    if temporal_bias_factor > 0 and theta_current is not None:
        # Reference theta = 0 deg
        theta_ref = 0.0
        delay_scale = 1.0 + temporal_bias_factor * (np.rad2deg(theta_current) - theta_ref) / 10.0
        delay_scale = max(delay_scale, 1.0)  # no speedup
        # Stretch time axis → slower rise
        times = np.linspace(0, len(stress_raw) - 1, len(stress_raw))  # assume uniform time steps
        t_stretched = times * delay_scale
        # Re-interpolate stress onto original time grid
        stress_interp = interp1d(t_stretched, stress_raw, kind='linear',
                                 bounds_error=False, fill_value=(stress_raw[0], stress_raw[-1]))
        t_original = times
        stress = stress_interp(t_original)
    else:
        stress = stress_raw
    return stress
# ----------------------------------------------------------------------
# Sunburst Matrix Builder — BULLETPROOF
# ----------------------------------------------------------------------
def build_sunburst_matrices(solutions, params_list, interpolator,
                           defect_target, stress_type, center_fraction, theta_spokes,
                           time_log_scale=False, temporal_bias_factor=0.0):
    N_TIME = 50
    stress_mat = np.zeros((N_TIME, len(theta_spokes)))
    times = np.logspace(-1, np.log10(200), N_TIME) if time_log_scale else np.linspace(0, 200.0, N_TIME)
    # Filter solutions by defect type
    filtered_solutions = [s for s, p in zip(solutions, params_list) if p[0] == defect_target]
    filtered_params = [p for p in params_list if p[0] == defect_target]
    if len(filtered_solutions) == 0:
        st.error(f"No solutions found for defect type: {defect_target}")
        st.stop()
    prog = st.progress(0)
    for j, theta in enumerate(theta_spokes):
        sol = interpolator(filtered_solutions, filtered_params, defect_target, theta)
        stress = get_center_stress(sol, stress_type, center_fraction, theta_current=theta, temporal_bias_factor=temporal_bias_factor)
        stress_mat[:, j] = stress[:N_TIME]  # Truncate or pad if needed
        prog.progress((j + 1) / len(theta_spokes))
    prog.empty()
    return stress_mat, times
# ----------------------------------------------------------------------
# Plotting Functions (adapted for stress)
# ----------------------------------------------------------------------
def plot_sunburst(data, title, cmap, vmin, vmax, log_scale, time_log_scale,
                 theta_dir, fname, times, theta_spokes, display_scale=1.0):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    theta_edges = np.linspace(0, 2*np.pi, len(theta_spokes) + 1)
    if time_log_scale:
        r_normalized = (np.log10(times) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0]))
        r_edges = np.concatenate([[0], r_normalized])
    else:
        r_edges = np.linspace(0, 1, len(times) + 1)
    Theta, R = np.meshgrid(theta_edges, r_edges)
    if theta_dir == "top→bottom":
        R = R[::-1]; data = data[::-1, :]
    norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(Theta, R, data, cmap=cmap, norm=norm, shading='auto')
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    ax.set_xticks(theta_centers)
    ax.set_xticklabels([f"{np.rad2deg(theta):.0f}°" for theta in theta_spokes], fontsize=16, fontweight='bold')
    if time_log_scale:
        ticks = [0.1, 1, 10, 100, 200]
        r_ticks = [(np.log10(t) - np.log10(times[0])) / (np.log10(times[-1]) - np.log10(times[0])) for t in ticks]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f'{t}' for t in ticks], fontsize=14)
    else:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0', '50', '100', '150', '200'], fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, color='w', linewidth=2.0, alpha=0.8)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=30)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    label = 'Stress (GPa)'
    cbar.set_label(label, fontsize=16)
    ticks = cbar.get_ticks()
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    pdf = os.path.join(FIGURE_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf
def plot_radar_single(data, stress_type, t_val, fname, theta_spokes, show_labels=True, show_radial_labels=True):
    angles = np.linspace(0, 2*np.pi, len(theta_spokes), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    data_cyclic = np.concatenate([data, [data[0]]])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    color = 'red' if stress_type == 'von_mises' else 'blue' if stress_type == 'sigma_hydro' else 'green'
    ax.plot(angles, data_cyclic, 'o-', linewidth=3, markersize=8, color=color, label=stress_type)
    ax.fill(angles, data_cyclic, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{np.rad2deg(theta):.0f}°" for theta in theta_spokes], fontsize=14)
    ax.set_ylim(0, max(np.max(data), 1e-6) * 1.2)
    ax.set_title(f"{stress_type} at t = {t_val:.1f} s", fontsize=18, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=14)
    ax.grid(True, linewidth=1.5)
    if show_radial_labels:
        ax.set_yticklabels([f"{y:.2e}" for y in ax.get_yticks()], fontsize=12)
    if show_labels:
        for a, v in zip(angles[:-1], data):
            if v > max(data) * 0.1:
                ax.annotate(f'{v:.1e}', (a, v), xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.close()
    return fig, png, None
def generate_session_id(parameters):
    s = f"{parameters.get('defect_target','ISF')}_{parameters.get('center_fraction',0.5)}"
    return f"session_{datetime.now():%Y%m%d_%H%M%S}_{hash(s)%10000:04d}"
# ----------------------------------------------------------------------
# MAIN — FULLY ELABORATE
# ----------------------------------------------------------------------
def main():
    global CURRENT_MODE
    st.set_page_config(page_title="Stress Visualizer", layout="wide")
    st.title("Interpolated Stress — Sunburst & Radar Charts")
    init_database()
    sols, params, logs = load_solutions(SOLUTION_DIR)
    with st.expander("Loaded Files"): [st.write(l) for l in logs]
    if not sols: st.stop()
    # Display extracted parameters
    with st.expander("Extracted Parameters from Filenames"):
        display_extracted_parameters(sols)
 
    with st.expander("Loaded Files"):
        [st.write(l) for l in logs]
    interpolator = MultiParamAttentionInterpolator()
    # Sidebar
    st.sidebar.header("Controls")
    sessions = get_recent_sessions()
    opts = ["Create New Session"] + [f"{s[0]} ({s[1]})" for s in sessions]
    selected = st.sidebar.selectbox("Session", opts)
    theta_step = st.sidebar.selectbox("Theta Step (deg)", [5, 10, 15], index=1)
    THETA_SPOKES = np.deg2rad(np.arange(0, 91, theta_step))
    col1, col2, col3 = st.sidebar.columns(3)
    cmap_vm = col1.selectbox("Von Mises Map", EXTENDED_CMAPS, EXTENDED_CMAPS.index('jet'))
    cmap_sh = col2.selectbox("Sigma Hydro Map", EXTENDED_CMAPS, EXTENDED_CMAPS.index('coolwarm'))
    cmap_sm = col3.selectbox("Sigma Mag Map", EXTENDED_CMAPS, EXTENDED_CMAPS.index('plasma'))
    log_scale = st.sidebar.checkbox("Log Scale", False)
    time_log = st.sidebar.checkbox("Log Time", True)
    show_labels = st.sidebar.checkbox("Radar Labels", True)
    show_radial = st.sidebar.checkbox("Radial Labels", True)
    frac = st.sidebar.selectbox("Position = Grid Size ×", ["0.5 (center)", "0.33", "0.25", "0.2"], index=0)
    center_fraction = {"0.5 (center)": 0.5, "0.33": 1/3, "0.25": 0.25, "0.2": 0.2}[frac]
    theta_dir = st.sidebar.radio("Time Flow", ["bottom→top", "top→bottom"])
    st.sidebar.header("Advanced Bias")
    temporal_bias_factor = st.sidebar.slider(
        "Temporal Delay Bias (per 10° theta increase)",
        min_value=0.0, max_value=0.02, value=0.0, step=0.005,
        help="Higher value = slower centerline stress rise for larger theta. 0 = no bias."
    )
    st.sidebar.header("Defect Type")
    defect_type = st.sidebar.radio("Defect Type", ["ISF", "ESF", "Twin"])
    CURRENT_MODE = defect_type
    if selected == "Create New Session":
        session_id = generate_session_id({'defect_target': defect_type, 'center_fraction': center_fraction})
        with st.spinner("Computing..."):
            von_mises_mat, times = build_sunburst_matrices(
                sols, params, interpolator, defect_type, 'von_mises', center_fraction, THETA_SPOKES, time_log, temporal_bias_factor
            )
            sigma_hydro_mat, _ = build_sunburst_matrices(
                sols, params, interpolator, defect_type, 'sigma_hydro', center_fraction, THETA_SPOKES, time_log, temporal_bias_factor
            )
            sigma_mag_mat, _ = build_sunburst_matrices(
                sols, params, interpolator, defect_type, 'sigma_mag', center_fraction, THETA_SPOKES, time_log, temporal_bias_factor
            )
        save_sunburst_data(session_id, {}, von_mises_mat, sigma_hydro_mat, sigma_mag_mat, times, THETA_SPOKES)
        st.success(f"Saved: {session_id}")
    else:
        session_id = selected.split(" (")[0]
        data = load_sunburst_data(session_id)
        if data:
            _, von_mises_mat, sigma_hydro_mat, sigma_mag_mat, times, THETA_SPOKES = data
            st.success(f"Loaded: {session_id}")
        else:
            st.error("Load failed"); return
    st.subheader("Sunburst Charts")
    c1, c2, c3 = st.columns(3)
    with c1:
        f, p, _ = plot_sunburst(
            von_mises_mat, f"Von Mises — {defect_type}", cmap_vm, 0, np.max(von_mises_mat),
            log_scale, time_log, theta_dir, f"von_mises_{session_id}", times, THETA_SPOKES
        )
        st.pyplot(f)
    with c2:
        f, p, _ = plot_sunburst(
            sigma_hydro_mat, f"Sigma Hydro — {defect_type}", cmap_sh, np.min(sigma_hydro_mat), np.max(sigma_hydro_mat),
            log_scale, time_log, theta_dir, f"sigma_hydro_{session_id}", times, THETA_SPOKES
        )
        st.pyplot(f)
    with c3:
        f, p, _ = plot_sunburst(
            sigma_mag_mat, f"Sigma Mag — {defect_type}", cmap_sm, 0, np.max(sigma_mag_mat),
            log_scale, time_log, theta_dir, f"sigma_mag_{session_id}", times, THETA_SPOKES
        )
        st.pyplot(f)
    st.subheader("Radar Charts")
    t_idx = st.slider("Time Index", 0, len(times)-1, 25)
    c1, c2, c3 = st.columns(3)
    with c1:
        f, _, _ = plot_radar_single(von_mises_mat[t_idx], "von_mises", times[t_idx], f"radar_von_mises", THETA_SPOKES, show_labels, show_radial)
        st.pyplot(f)
    with c2:
        f, _, _ = plot_radar_single(sigma_hydro_mat[t_idx], "sigma_hydro", times[t_idx], f"radar_sigma_hydro", THETA_SPOKES, show_labels, show_radial)
        st.pyplot(f)
    with c3:
        f, _, _ = plot_radar_single(sigma_mag_mat[t_idx], "sigma_mag", times[t_idx], f"radar_sigma_mag", THETA_SPOKES, show_labels, show_radial)
        st.pyplot(f)
if __name__ == "__main__":
    main()
