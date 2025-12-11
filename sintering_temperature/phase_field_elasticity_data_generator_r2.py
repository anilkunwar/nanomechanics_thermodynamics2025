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
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------
# Helper formatting / naming
# ----------------------------
def sanitize_token(text: str) -> str:
    """Sanitize small text tokens for filenames: remove spaces and special braces."""
    s = str(text)
    for ch in [" ", "{", "}", "/", "\\", ",", ";", "(", ")", "[", "]", "°"]:
        s = s.replace(ch, "")
    return s

def fmt_num_trim(x, ndigits=3):
    """Format a float with up to ndigits decimals and strip trailing zeros."""
    s = f"{x:.{ndigits}f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s

def build_sim_name(params: dict, sim_id: str = None) -> str:
    """
    Build a filename-friendly simulation name in the requested format:
    Example: ISF_orient0deg_Square_eps0-0.707_kappa-0.6[_<simid>]
    If sim_id provided, append short id to guarantee uniqueness.
    """
    defect = sanitize_token(params.get("defect_type", "def"))
    shape = sanitize_token(params.get("shape", "shape"))
    # Angle in degrees from theta
    theta = params.get("theta", 0.0)
    angle_deg = int(round(np.rad2deg(theta)))
    orient_token = f"orient{angle_deg}deg"
    eps0 = fmt_num_trim(params.get("eps0", 0.0), ndigits=3)
    kappa = fmt_num_trim(params.get("kappa", 0.0), ndigits=3)
    name = f"{defect}_{orient_token}_{shape}_eps0-{eps0}_kappa-{kappa}"
    if sim_id:
        name = f"{name}_{sim_id}"
    return name

# ----------------------------
# Colormap Configuration
# ----------------------------
def get_colormaps():
    """Return a list of 50 colormaps with categories."""
    # Sequential colormaps
    sequential = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
        'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
        'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
    ]
    
    # Diverging colormaps
    diverging = [
        'coolwarm', 'bwr', 'seismic', 'RdYlBu', 'RdYlGn',
        'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdBu', 'Spectral'
    ]
    
    # Qualitative colormaps
    qualitative = [
        'tab20c', 'tab20', 'Set3', 'Set2', 'Set1',
        'tab10', 'Accent', 'Dark2', 'Paired', 'Pastel1',
        'Pastel2', 'hsv', 'twilight', 'twilight_shifted'
    ]
    
    # Perceptually uniform cyclic
    cyclic = ['hsv', 'twilight', 'twilight_shifted', 'flag']
    
    # Combine all, ensuring we have exactly 50
    all_maps = []
    all_maps.extend(sequential[:30])  # Take first 30 sequential
    all_maps.extend(diverging)        # All 11 diverging
    all_maps.extend(qualitative[:9])  # Take first 9 qualitative
    
    # If we still don't have 50, add more sequential
    if len(all_maps) < 50:
        all_maps.extend(sequential[30:30 + (50 - len(all_maps))])
    
    return all_maps[:50]  # Ensure exactly 50

COLORMAPS = get_colormaps()

def get_colormap_display_name(cm_name):
    """Convert colormap name to display name."""
    if cm_name in ['coolwarm', 'bwr', 'seismic']:
        return f"{cm_name} (Diverging)"
    elif cm_name in ['viridis', 'plasma', 'inferno', 'magma']:
        return f"{cm_name} (Perceptually Uniform)"
    elif cm_name in ['hsv', 'twilight']:
        return f"{cm_name} (Cyclic)"
    else:
        return cm_name

# ----------------------------
# Configure page
# ----------------------------
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Analyzer")

# ----------------------------
# Material & Grid
# ----------------------------
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)
# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1
N = 128
dx = 0.1  # nm
extent = [-N * dx / 2, N * dx / 2, -N * dx / 2, N * dx / 2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# SIMULATION ENGINE
# =============================================
def create_initial_eta(shape, defect_type):
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes.get(defect_type, 0.7)

    eta = np.zeros((N, N))
    cx, cy = N // 2, N // 2
    w, h = (24, 12) if shape in ["Rectangle", "Horizontal Fault"] else (16, 16)

    if shape == "Square":
        eta[cy - h:cy + h, cx - h:cx + h] = init_amplitude
    elif shape == "Horizontal Fault":
        eta[cy - 4:cy + 4, cx - w:cx + w] = init_amplitude
    elif shape == "Vertical Fault":
        eta[cy - w:cy + w, cx - 4:cx + 4] = init_amplitude
    elif shape == "Rectangle":
        eta[cy - h:cy + h, cx - w:cx + w] = init_amplitude
    elif shape == "Ellipse":
        mask = ((X / (w * 1.5)) ** 2 + (Y / (h * 1.5)) ** 2) <= 1
        eta[mask] = init_amplitude

    eta += 0.02 * np.random.randn(N, N)
    return np.clip(eta, 0.0, 1.0)

@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, kappa, dt, dx, N):
    eta_new = eta.copy()
    dx2 = dx * dx
    for i in prange(1, N - 1):
        for j in prange(1, N - 1):
            lap = (eta[i + 1, j] + eta[i - 1, j] + eta[i, j + 1] + eta[i, j - 1] - 4 * eta[i, j]) / dx2
            dF = 2 * eta[i, j] * (1 - eta[i, j]) * (eta[i, j] - 0.5)
            eta_new[i, j] = eta[i, j] + dt * (-dF + kappa * lap)
            eta_new[i, j] = np.maximum(0.0, np.minimum(1.0, eta_new[i, j]))
    eta_new[0, :] = eta_new[-2, :]; eta_new[-1, :] = eta_new[1, :]
    eta_new[:, 0] = eta_new[:, -2]; eta_new[:, -1] = eta_new[:, 1]
    return eta_new

def compute_stress_fields(eta, eps0, theta):
    # convert GPa to Pa for elasticity calcs, then revert to GPa for returned stresses
    C11_p = (C11 - C12 ** 2 / C11) * 1e9
    C12_p = (C12 - C12 ** 2 / C11) * 1e9
    C44_p = C44 * 1e9
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(2 * np.pi * kx, 2 * np.pi * ky)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1e-12
    mask = K2 > 0
    n1 = np.zeros_like(KX)
    n2 = np.zeros_like(KX)
    n1[mask] = KX[mask] / np.sqrt(K2[mask])
    n2[mask] = KY[mask] / np.sqrt(K2[mask])
    A11 = np.zeros_like(KX)
    A22 = np.zeros_like(KX)
    A12 = np.zeros_like(KX)
    A11[mask] = C11_p * n1[mask] ** 2 + C44_p * n2[mask] ** 2
    A22[mask] = C11_p * n2[mask] ** 2 + C44_p * n1[mask] ** 2
    A12[mask] = (C12_p + C44_p) * n1[mask] * n2[mask]
    det = A11 * A22 - A12 ** 2
    # avoid division by zero
    G11 = np.zeros_like(KX)
    G22 = np.zeros_like(KX)
    G12 = np.zeros_like(KX)
    G11[mask] = A22[mask] / det[mask]
    G22[mask] = A11[mask] / det[mask]
    G12[mask] = -A12[mask] / det[mask]

    # eigenstrain construction
    gamma = eps0
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    delta = 0.02
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
    eps_xx_star = eps_star[0, 0] * eta
    eps_yy_star = eps_star[1, 1] * eta
    eps_xy_star = eps_star[0, 1] * eta

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

    # convert back to GPa for returned stresses
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy)
    sigma_mag = np.sqrt(sxx ** 2 + syy ** 2 + 2 * sxy ** 2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2 + 6 * sxy ** 2))

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

# Cache management section
st.sidebar.subheader("🔄 Cache Management")
if st.sidebar.button("🗑️ Clear All Simulations", type="secondary"):
    if 'simulations' in st.session_state:
        del st.session_state.simulations
        st.success("All simulations cleared!")
        st.rerun()

if st.sidebar.button("🔄 Run New Simulation (Clear Cache)", type="primary"):
    # Clear cache and run new simulation
    if 'simulations' in st.session_state:
        del st.session_state.simulations
    # Parameters will be collected below

st.sidebar.markdown("---")

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

# Run button (always visible)
run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

if run_button:
    sim_params = {
        'defect_type': defect_type,
        'shape': shape,
        'eps0': float(eps0),
        'kappa': float(kappa),
        'orientation': orientation,
        'theta': float(theta),
        'steps': int(steps),
        'save_every': int(save_every)
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
        # Generate sim_id (short hash)
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
        st.rerun()

# =============================================
# VISUALIZATION PANEL
# =============================================
st.header("📊 Simulation Viewer")

simulations = st.session_state.get("simulations", {})

if simulations:
    view_sim_id = st.selectbox("Select Simulation to Visualize", list(simulations.keys()), key="viewer_select")
    
    if view_sim_id:
        sim_data = simulations[view_sim_id]
        history = sim_data["history"]
        sim_params = sim_data['params']
        
        # Display simulation info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Defect Type", sim_params['defect_type'])
        with col2:
            st.metric("Shape", sim_params['shape'])
        with col3:
            st.metric("ε*", f"{sim_params['eps0']:.3f}")
        with col4:
            st.metric("κ", f"{sim_params['kappa']:.3f}")
        
        # Colormap selection
        st.subheader("🎨 Colormap Configuration")
        
        # Initialize colormap session state
        if 'colormaps' not in st.session_state:
            # Default colormaps
            st.session_state.colormaps = {
                'eta': 'magma',
                'sxx': 'coolwarm',
                'syy': 'coolwarm',
                'sxy': 'coolwarm',
                'sigma_hydro': 'viridis',
                'von_mises': 'plasma',
                'sigma_mag': 'inferno'
            }
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        with col1:
            new_eta_cm = st.selectbox(
                "η Field",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['eta']),
                format_func=get_colormap_display_name,
                key="eta_cm"
            )
            st.session_state.colormaps['eta'] = new_eta_cm
            
        with col2:
            new_sxx_cm = st.selectbox(
                "σxx",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['sxx']),
                format_func=get_colormap_display_name,
                key="sxx_cm"
            )
            st.session_state.colormaps['sxx'] = new_sxx_cm
            
        with col3:
            new_syy_cm = st.selectbox(
                "σyy",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['syy']),
                format_func=get_colormap_display_name,
                key="syy_cm"
            )
            st.session_state.colormaps['syy'] = new_syy_cm
            
        with col4:
            new_sxy_cm = st.selectbox(
                "σxy",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['sxy']),
                format_func=get_colormap_display_name,
                key="sxy_cm"
            )
            st.session_state.colormaps['sxy'] = new_sxy_cm
            
        with col5:
            new_hydro_cm = st.selectbox(
                "Hydrostatic",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['sigma_hydro']),
                format_func=get_colormap_display_name,
                key="hydro_cm"
            )
            st.session_state.colormaps['sigma_hydro'] = new_hydro_cm
            
        with col6:
            new_vm_cm = st.selectbox(
                "Von Mises",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['von_mises']),
                format_func=get_colormap_display_name,
                key="vm_cm"
            )
            st.session_state.colormaps['von_mises'] = new_vm_cm
            
        with col7:
            new_mag_cm = st.selectbox(
                "Magnitude",
                COLORMAPS,
                index=COLORMAPS.index(st.session_state.colormaps['sigma_mag']),
                format_func=get_colormap_display_name,
                key="mag_cm"
            )
            st.session_state.colormaps['sigma_mag'] = new_mag_cm
        
        # Frame selection with dynamic update
        num_frames = len(history)
        frame_idx = st.slider("Frame", 0, num_frames - 1, num_frames - 1, 
                             key=f"frame_slider_{view_sim_id}")
        
        eta, stress = history[frame_idx]
        
        # -----------------------------
        # η FIELD HEATMAP
        # -----------------------------
        st.subheader("🟣 Phase Field η")
        
        fig_eta, ax_eta = plt.subplots(figsize=(5, 5))
        im = ax_eta.imshow(eta, extent=extent, 
                          cmap=st.session_state.colormaps['eta'], 
                          origin="lower")
        ax_eta.set_title(f"η Field (Frame {frame_idx})")
        plt.colorbar(im, ax=ax_eta, shrink=0.7)
        st.pyplot(fig_eta)
        plt.close(fig_eta)
        
        # -----------------------------
        # STRESS FIELD HEATMAPS
        # -----------------------------
        st.subheader("💥 Stress Fields")
        
        fig_s, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        fields = [
            ("σxx", stress["sxx"], st.session_state.colormaps['sxx']),
            ("σyy", stress["syy"], st.session_state.colormaps['syy']),
            ("σxy", stress["sxy"], st.session_state.colormaps['sxy']),
            ("Hydrostatic (σ_h)", stress["sigma_hydro"], st.session_state.colormaps['sigma_hydro']),
            ("Von Mises", stress["von_mises"], st.session_state.colormaps['von_mises']),
            ("Magnitude", stress["sigma_mag"], st.session_state.colormaps['sigma_mag']),
        ]
        
        for ax, (title, field, cmap) in zip(axs.flatten(), fields):
            im = ax.imshow(field, extent=extent, cmap=cmap, origin="lower")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.7)
        
        st.pyplot(fig_s)
        plt.close(fig_s)
        
        # -----------------------------
        # ANIMATION CONTROLS
        # -----------------------------
        st.subheader("🎬 Animation Controls")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⏮️ First Frame"):
                st.session_state[f"frame_slider_{view_sim_id}"] = 0
                st.rerun()
        
        with col2:
            play = st.checkbox("▶️ Play Animation", key=f"play_{view_sim_id}")
            if play:
                # Auto-advance frames
                import time as sleep_time
                current_frame = st.session_state.get(f"current_frame_{view_sim_id}", 0)
                if current_frame < num_frames - 1:
                    st.session_state[f"frame_slider_{view_sim_id}"] = current_frame + 1
                    st.session_state[f"current_frame_{view_sim_id}"] = current_frame + 1
                else:
                    st.session_state[f"frame_slider_{view_sim_id}"] = 0
                    st.session_state[f"current_frame_{view_sim_id}"] = 0
                st.rerun()
            else:
                st.session_state[f"current_frame_{view_sim_id}"] = 0
        
        with col3:
            if st.button("⏭️ Last Frame"):
                st.session_state[f"frame_slider_{view_sim_id}"] = num_frames - 1
                st.rerun()
        
        # Display frame information
        st.info(f"**Frame {frame_idx + 1} of {num_frames}** | Simulation Time: {frame_idx * sim_params['save_every']} steps")
        
        # Quick navigation buttons
        st.subheader("Quick Frame Navigation")
        cols = st.columns(10)
        for i in range(min(10, num_frames)):
            with cols[i]:
                if st.button(f"F{i}", key=f"nav_{view_sim_id}_{i}"):
                    st.session_state[f"frame_slider_{view_sim_id}"] = i
                    st.rerun()
        
        # Delete current simulation option
        st.markdown("---")
        if st.button("🗑️ Delete This Simulation", key=f"delete_{view_sim_id}"):
            del st.session_state.simulations[view_sim_id]
            st.success(f"Simulation {view_sim_id} deleted!")
            st.rerun()
            
else:
    st.info("No simulations available. Run a simulation from the sidebar to visualize results.")
    
    # Show example of colormaps
    st.subheader("🎨 Available Colormaps (50 Total)")
    
    # Display colormap examples
    num_examples = 12
    example_colormaps = COLORMAPS[:num_examples]
    
    fig_ex, axs_ex = plt.subplots(3, 4, figsize=(15, 8))
    axs_ex = axs_ex.flatten()
    
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    for ax, cmap_name in zip(axs_ex, example_colormaps):
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.set_title(get_colormap_display_name(cmap_name), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    st.pyplot(fig_ex)
    plt.close(fig_ex)
    
    st.info(f"Total of {len(COLORMAPS)} colormaps available. Run a simulation to customize visualization.")

# ----------------------------
# Download section
# ----------------------------
st.header("🔥 Download Files")
simulations = st.session_state.get('simulations', {})
if simulations:
    selected_sim_id = st.selectbox("Select Simulation ID for Download", list(simulations.keys()), key="download_select")
    if selected_sim_id:
        sim_data = simulations[selected_sim_id]
        history = sim_data['history']
        sim_params = sim_data['params']
        metadata = sim_data['metadata']

        # attach sim_name into metadata & params
        sim_name = build_sim_name(sim_params, sim_id=selected_sim_id)
        metadata['sim_name'] = sim_name
        sim_params['sim_name'] = sim_name
        sim_params['sim_id'] = selected_sim_id

        # assemble data object
        data = {'params': sim_params, 'history': [], 'metadata': metadata}
        for eta, stress_fields in history:
            data['history'].append({
                'eta': eta,
                'stresses': stress_fields
            })

        # ----------------------------
        # PKL (pickle)
        # ----------------------------
        pkl_buffer = BytesIO()
        pickle.dump(data, pkl_buffer)
        pkl_buffer.seek(0)
        st.download_button("Download PKL", pkl_buffer, file_name=f"{sim_name}.pkl")

        # ----------------------------
        # PT (torch) - safe conversion
        # ----------------------------
        pt_buffer = BytesIO()

        def to_tensor(x):
            # numpy ndarray -> torch.from_numpy
            if isinstance(x, np.ndarray):
                try:
                    return torch.from_numpy(x)
                except Exception:
                    # fallback to tensor constructor
                    return torch.tensor(x)
            else:
                # covers numpy scalars, python floats, ints, lists, etc.
                return torch.tensor(x)

        tensor_data = {
            'params': sim_params,
            'metadata': metadata,
            'history': []
        }

        for frame in data['history']:
            frame_tensor = {
                'eta': to_tensor(frame['eta']),
                'stresses': {k: to_tensor(v) for k, v in frame['stresses'].items()}
            }
            tensor_data['history'].append(frame_tensor)

        torch.save(tensor_data, pt_buffer)
        pt_buffer.seek(0)
        st.download_button("Download PT", pt_buffer, file_name=f"{sim_name}.pt")

        # ----------------------------
        # DB (in-memory SQL dump)
        # ----------------------------
        db_dump_buffer = StringIO()
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        c.execute('''CREATE TABLE simulations (
                     id TEXT PRIMARY KEY,
                     sim_name TEXT,
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
        c.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (selected_sim_id, sim_name, sim_params['defect_type'], sim_params['shape'],
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
        for line in conn.iterdump():
            db_dump_buffer.write('%s\n' % line)
        db_dump_str = db_dump_buffer.getvalue()
        st.download_button("Download DB (SQL Dump)", db_dump_str, file_name=f"{sim_name}.sql")

        # ----------------------------
        # CSV (zip multiple frames)
        # ----------------------------
        csv_zip_buffer = BytesIO()
        with zipfile.ZipFile(csv_zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
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
                zf.writestr(f"{sim_name}_frame_{idx}.csv", csv_str)
        csv_zip_buffer.seek(0)
        st.download_button("Download CSV (Zip)", csv_zip_buffer, file_name=f"{sim_name}_csv.zip")

else:
    st.info("No simulations available for download.")
