# =============================================
# ULTIMATE Ag NP Defect Analyzer – MULTI-SIMULATION COMPARISON
# =============================================
import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime

# Configure page with better styling
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Comparison Platform")
st.markdown("""
**Run multiple simulations • Compare ISF/ESF/Twin with different orientations • Cloud-style storage**
**Run → Save → Compare • 50+ Colormaps • Publication-ready comparison plots**
""")

# =============================================
# Material & Grid
# =============================================
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)

# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

N = 128
dx = 0.1  # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# EXPANDED COLORMAP LIBRARY (50+ options)
# =============================================
COLORMAPS = {
    # Sequential (1)
    'viridis': 'viridis',
    'plasma': 'plasma', 
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'hot': 'hot',
    'cool': 'cool',
    'spring': 'spring',
    'summer': 'summer',
    'autumn': 'autumn',
    'winter': 'winter',
    
    # Sequential (2)
    'copper': 'copper',
    'bone': 'bone',
    'gray': 'gray',
    'pink': 'pink',
    'afmhot': 'afmhot',
    'gist_heat': 'gist_heat',
    'gist_gray': 'gist_gray',
    'binary': 'binary',
    
    # Diverging
    'coolwarm': 'coolwarm',
    'bwr': 'bwr',
    'seismic': 'seismic',
    'RdBu': 'RdBu',
    'RdGy': 'RdGy',
    'PiYG': 'PiYG',
    'PRGn': 'PRGn',
    'BrBG': 'BrBG',
    'PuOr': 'PuOr',
    
    # Cyclic
    'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted',
    'hsv': 'hsv',
    
    # Qualitative
    'tab10': 'tab10',
    'tab20': 'tab20',
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    'Paired': 'Paired',
    'Accent': 'Accent',
    'Dark2': 'Dark2',
    
    # Miscellaneous
    'jet': 'jet',
    'turbo': 'turbo',
    'rainbow': 'rainbow',
    'nipy_spectral': 'nipy_spectral',
    'gist_ncar': 'gist_ncar',
    'gist_rainbow': 'gist_rainbow',
    'gist_earth': 'gist_earth',
    'gist_stern': 'gist_stern',
    'ocean': 'ocean',
    'terrain': 'terrain',
    'gnuplot': 'gnuplot',
    'gnuplot2': 'gnuplot2',
    'CMRmap': 'CMRmap',
    'cubehelix': 'cubehelix',
    'brg': 'brg',
    
    # Perceptually uniform
    'rocket': 'rocket',
    'mako': 'mako',
    'crest': 'crest',
    'flare': 'flare',
    'icefire': 'icefire',
    'vlag': 'vlag'
}

cmap_list = list(COLORMAPS.keys())

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
# SIDEBAR - Global Settings (Available in Both Modes)
# =============================================
st.sidebar.header("🎨 Global Chart Styling")

# Chart styling controls (available in both modes)
title_font_size = st.sidebar.slider("Title Font Size", 12, 24, 16)
label_font_size = st.sidebar.slider("Label Font Size", 10, 45, 14)
line_width = st.sidebar.slider("Line Width", 1.0, 5.0, 2.0, 0.5)
spine_width = st.sidebar.slider("Spine Line Width", 1.0, 4.0, 2.5, 0.5)
tick_length = st.sidebar.slider("Tick Length", 4, 12, 6)
tick_width = st.sidebar.slider("Tick Width", 1.0, 3.0, 2.0, 0.5)

# Color maps selection (available in both modes for consistency)
st.sidebar.subheader("Default Colormap Selection")
eta_cmap_name = st.sidebar.selectbox("Default η colormap", cmap_list, index=cmap_list.index('viridis'))
sigma_cmap_name = st.sidebar.selectbox("Default |σ| colormap", cmap_list, index=cmap_list.index('hot'))
hydro_cmap_name = st.sidebar.selectbox("Default Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
vm_cmap_name = st.sidebar.selectbox("Default von Mises colormap", cmap_list, index=cmap_list.index('plasma'))

# =============================================
# SIDEBAR - Multi-Simulation Control Panel
# =============================================
st.sidebar.header("🚀 Multi-Simulation Manager")

# Operation mode
operation_mode = st.sidebar.radio(
    "Operation Mode",
    ["Run New Simulation", "Compare Saved Simulations"],
    index=0
)

if operation_mode == "Run New Simulation":
    st.sidebar.header("🎛️ New Simulation Setup")
    
    # Custom CSS for larger slider labels
    st.markdown("""
    <style>
        .stSlider label {
            font-size: 16px !important;
            font-weight: 600 !important;
        }
        .stSelectbox label {
            font-size: 16px !important;
            font-weight: 600 !important;
        }
        .stNumberInput label {
            font-size: 14px !important;
            font-weight: 600 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])
    
    # Physical eigenstrain values
    if defect_type == "ISF":
        default_eps = 0.707
        default_kappa = 0.6
        init_amplitude = 0.70
        caption = "Intrinsic Stacking Fault"
    elif defect_type == "ESF":
        default_eps = 1.414
        default_kappa = 0.7
        init_amplitude = 0.75
        caption = "Extrinsic Stacking Fault"
    else:  # Twin
        default_eps = 2.121
        default_kappa = 0.3
        init_amplitude = 0.90
        caption = "Coherent Twin Boundary"
    
    st.sidebar.info(f"**{caption}**")
    
    shape = st.sidebar.selectbox("Initial Seed Shape",
        ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"])
    
    # Enhanced sliders
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
    
    st.sidebar.info(f"Selected tilt: **{np.rad2deg(theta):.1f}°** from horizontal")
    
    # Visualization settings - Individual for this simulation
    st.sidebar.subheader("Simulation-Specific Colormaps")
    sim_eta_cmap_name = st.sidebar.selectbox("η colormap for this sim", cmap_list, 
                                           index=cmap_list.index(eta_cmap_name))
    sim_sigma_cmap_name = st.sidebar.selectbox("|σ| colormap for this sim", cmap_list, 
                                             index=cmap_list.index(sigma_cmap_name))
    sim_hydro_cmap_name = st.sidebar.selectbox("Hydrostatic colormap for this sim", cmap_list, 
                                             index=cmap_list.index(hydro_cmap_name))
    sim_vm_cmap_name = st.sidebar.selectbox("von Mises colormap for this sim", cmap_list, 
                                          index=cmap_list.index(vm_cmap_name))
    
    # Run button
    if st.sidebar.button("🚀 Run & Save Simulation", type="primary"):
        st.session_state.run_new_simulation = True
        st.session_state.sim_params = {
            'defect_type': defect_type,
            'shape': shape,
            'eps0': eps0,
            'kappa': kappa,
            'orientation': orientation,
            'theta': theta,
            'steps': steps,
            'save_every': save_every,
            'eta_cmap': sim_eta_cmap_name,
            'sigma_cmap': sim_sigma_cmap_name,
            'hydro_cmap': sim_hydro_cmap_name,
            'vm_cmap': sim_vm_cmap_name
        }

else:  # Compare Saved Simulations
    st.sidebar.header("🔍 Simulation Comparison Setup")
    
    # Get available simulations
    simulations = SimulationDB.get_simulation_list()
    
    if not simulations:
        st.sidebar.warning("No simulations saved yet. Run some simulations first!")
    else:
        # Multi-select for comparison
        sim_options = {f"{sim['name']} (ID: {sim['id']})": sim['id'] for sim in simulations}
        selected_sim_ids = st.sidebar.multiselect(
            "Select Simulations to Compare",
            options=list(sim_options.keys()),
            default=list(sim_options.keys())[:min(3, len(sim_options))]
        )
        
        # Convert back to IDs
        selected_ids = [sim_options[name] for name in selected_sim_ids]
        
        # Comparison settings
        st.sidebar.subheader("Comparison Settings")
        
        comparison_type = st.sidebar.selectbox(
            "Comparison Type",
            ["Side-by-Side Heatmaps", "Overlay Line Profiles", "Radial Profile Comparison", "Statistical Summary"]
        )
        
        stress_component = st.sidebar.selectbox(
            "Stress Component", 
            ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
            index=0
        )
        
        frame_selection = st.sidebar.radio(
            "Frame Selection",
            ["Final Frame", "Same Evolution Time", "Specific Frame Index"],
            horizontal=True
        )
        
        if frame_selection == "Specific Frame Index":
            frame_idx = st.sidebar.slider("Frame Index", 0, 100, 0)
        else:
            frame_idx = None
        
        # Comparison-specific styling
        st.sidebar.subheader("Comparison Styling")
        comparison_line_style = st.sidebar.selectbox(
            "Line Style",
            ["solid", "dashed", "dotted", "dashdot"],
            index=0
        )
        
        # Run comparison
        if st.sidebar.button("🔬 Run Comparison", type="primary"):
            st.session_state.run_comparison = True
            st.session_state.comparison_config = {
                'sim_ids': selected_ids,
                'type': comparison_type,
                'stress_component': stress_component,
                'frame_selection': frame_selection,
                'frame_idx': frame_idx,
                'line_style': comparison_line_style
            }

# =============================================
# SIMULATION ENGINE (Reusable Functions)
# =============================================
def create_initial_eta(shape, defect_type):
    """Create initial defect configuration"""
    # Set initial amplitude based on defect type
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
    """Phase field evolution with Allen-Cahn equation"""
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
    """FFT-based stress solver with rotated eigenstrain"""
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
    delta = 0.02  # Small dilatation
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
    szz = (C12 / (C11 + C12)) * (sxx + syy)  # Plane strain approximation
    
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
# MAIN CONTENT AREA
# =============================================
if operation_mode == "Run New Simulation":
    # Show simulation preview
    st.header("🎯 New Simulation Preview")
    
    if 'sim_params' in st.session_state:
        sim_params = st.session_state.sim_params
        
        # Display simulation parameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Defect Type", sim_params['defect_type'])
        with col2:
            st.metric("ε*", f"{sim_params['eps0']:.3f}")
        with col3:
            st.metric("κ", f"{sim_params['kappa']:.2f}")
        with col4:
            st.metric("Orientation", sim_params['orientation'])
        
        # Show initial configuration
        init_eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initial defect
        im1 = ax1.imshow(init_eta, extent=extent, cmap=COLORMAPS[sim_params['eta_cmap']], origin='lower')
        ax1.set_title(f"Initial {sim_params['defect_type']} - {sim_params['shape']}", 
                     fontsize=title_font_size, fontweight='bold')
        ax1.set_xlabel("x (nm)", fontsize=label_font_size)
        ax1.set_ylabel("y (nm)", fontsize=label_font_size)
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Stress preview (calculated from initial state)
        stress_preview = compute_stress_fields(init_eta, sim_params['eps0'], sim_params['theta'])
        im2 = ax2.imshow(stress_preview['sigma_mag'], extent=extent, 
                        cmap=COLORMAPS[sim_params['sigma_cmap']], origin='lower')
        ax2.set_title(f"Initial Stress Magnitude", fontsize=title_font_size, fontweight='bold')
        ax2.set_xlabel("x (nm)", fontsize=label_font_size)
        ax2.set_ylabel("y (nm)", fontsize=label_font_size)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        st.pyplot(fig)
        
        # Run simulation button
        if st.button("▶️ Start Full Simulation", type="primary"):
            with st.spinner(f"Running {sim_params['defect_type']} simulation..."):
                start_time = time.time()
                
                # Run simulation
                history = run_simulation(sim_params)
                
                # Create metadata
                metadata = {
                    'run_time': time.time() - start_time,
                    'frames': len(history),
                    'grid_size': N,
                    'dx': dx,
                    'colormaps': {
                        'eta': sim_params['eta_cmap'],
                        'sigma': sim_params['sigma_cmap'],
                        'hydro': sim_params['hydro_cmap'],
                        'vm': sim_params['vm_cmap']
                    }
                }
                
                # Save to database
                sim_id = SimulationDB.save_simulation(sim_params, history, metadata)
                
                st.success(f"""
                ✅ Simulation Complete!
                - **ID**: `{sim_id}`
                - **Frames**: {len(history)}
                - **Time**: {metadata['run_time']:.1f} seconds
                - **Saved to database**
                """)
                
                # Show final frame
                final_eta, final_stress = history[-1]
                
                fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
                
                im3 = ax3.imshow(final_eta, extent=extent, 
                                cmap=COLORMAPS[sim_params['eta_cmap']], origin='lower')
                ax3.set_title(f"Final {sim_params['defect_type']}", fontsize=title_font_size, fontweight='bold')
                ax3.set_xlabel("x (nm)", fontsize=label_font_size)
                ax3.set_ylabel("y (nm)", fontsize=label_font_size)
                plt.colorbar(im3, ax=ax3, shrink=0.8)
                
                im4 = ax4.imshow(final_stress['sigma_mag'], extent=extent,
                                cmap=COLORMAPS[sim_params['sigma_cmap']], origin='lower')
                ax4.set_title(f"Final Stress Magnitude", fontsize=title_font_size, fontweight='bold')
                ax4.set_xlabel("x (nm)", fontsize=label_font_size)
                ax4.set_ylabel("y (nm)", fontsize=label_font_size)
                plt.colorbar(im4, ax=ax4, shrink=0.8)
                
                st.pyplot(fig2)
                
                # Clear the run flag
                if 'run_new_simulation' in st.session_state:
                    del st.session_state.run_new_simulation
    
    else:
        st.info("Configure simulation parameters in the sidebar and click 'Run & Save Simulation'")
    
    # Show saved simulations
    st.header("📋 Saved Simulations")
    simulations = SimulationDB.get_simulation_list()
    
    if simulations:
        # Create a dataframe of saved simulations
        sim_data = []
        for sim in simulations:
            params = sim['params']
            sim_data.append({
                'ID': sim['id'],
                'Defect Type': params['defect_type'],
                'Orientation': params['orientation'],
                'ε*': params['eps0'],
                'κ': params['kappa'],
                'Shape': params['shape'],
                'Steps': params['steps']
            })
        
        df = pd.DataFrame(sim_data)
        st.dataframe(df, use_container_width=True)
        
        # Delete option
        with st.expander("🗑️ Delete Simulations"):
            delete_options = [f"{sim['name']} (ID: {sim['id']})" for sim in simulations]
            to_delete = st.multiselect("Select simulations to delete", delete_options)
            
            if st.button("Delete Selected", type="secondary"):
                for sim_name in to_delete:
                    # Extract ID from string
                    sim_id = sim_name.split("ID: ")[1].replace(")", "")
                    if SimulationDB.delete_simulation(sim_id):
                        st.success(f"Deleted simulation {sim_id}")
                st.rerun()
    else:
        st.info("No simulations saved yet. Run a simulation to see it here!")

else:  # COMPARE SAVED SIMULATIONS
    st.header("🔬 Multi-Simulation Comparison")
    
    if 'run_comparison' in st.session_state and st.session_state.run_comparison:
        config = st.session_state.comparison_config
        
        # Load selected simulations
        simulations = []
        valid_sim_ids = []
        
        for sim_id in config['sim_ids']:
            sim_data = SimulationDB.get_simulation(sim_id)
            if sim_data:
                simulations.append(sim_data)
                valid_sim_ids.append(sim_id)
            else:
                st.warning(f"Simulation {sim_id} not found!")
        
        if not simulations:
            st.error("No valid simulations selected for comparison!")
        else:
            st.success(f"Loaded {len(simulations)} simulations for comparison")
            
            # Determine frame index
            frame_idx = config['frame_idx']
            if config['frame_selection'] == "Final Frame":
                # Use final frame for each simulation
                frames = [len(sim['history']) - 1 for sim in simulations]
            elif config['frame_selection'] == "Same Evolution Time":
                # Use same evolution time (percentage of total steps)
                target_percentage = 0.8  # 80% of evolution
                frames = [int(len(sim['history']) * target_percentage) for sim in simulations]
            else:
                # Specific frame index
                frames = [min(frame_idx, len(sim['history']) - 1) for sim in simulations]
            
            # Get stress component mapping
            stress_map = {
                "Stress Magnitude |σ|": 'sigma_mag',
                "Hydrostatic σ_h": 'sigma_hydro',
                "von Mises σ_vM": 'von_mises'
            }
            stress_key = stress_map[config['stress_component']]
            
            # Create comparison based on type
            if config['type'] == "Side-by-Side Heatmaps":
                st.subheader("🌡️ Side-by-Side Heatmap Comparison")
                
                n_sims = len(simulations)
                cols = min(3, n_sims)
                rows = (n_sims + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)
                
                colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
                
                for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row, col]
                    
                    # Get data
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Use simulation's own colormap
                    sim_cmap = COLORMAPS[sim['params']['sigma_cmap']]
                    
                    im = ax.imshow(stress_data, extent=extent, cmap=sim_cmap, origin='lower')
                    ax.set_title(f"{sim['params']['defect_type']}\n{sim['params']['orientation']}", 
                               fontsize=title_font_size, fontweight='bold')
                    ax.set_xlabel("x (nm)", fontsize=label_font_size)
                    ax.set_ylabel("y (nm)", fontsize=label_font_size)
                    
                    plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Hide empty subplots
                for idx in range(n_sims, rows*cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics table
                st.subheader("📊 Comparison Statistics")
                stats_data = []
                for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key].flatten()
                    
                    stats_data.append({
                        'Simulation': f"{sim['params']['defect_type']} - {sim['params']['orientation']}",
                        'Max (GPa)': float(np.nanmax(stress_data)),
                        'Mean (GPa)': float(np.nanmean(stress_data)),
                        'Std Dev': float(np.nanstd(stress_data)),
                        'Frame': frame,
                        'ID': sim['id']
                    })
                
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats.style.format({
                    'Max (GPa)': '{:.3f}',
                    'Mean (GPa)': '{:.3f}',
                    'Std Dev': '{:.3f}'
                }), use_container_width=True)
            
            elif config['type'] == "Overlay Line Profiles":
                st.subheader("📈 Overlay Line Profile Comparison")
                
                # Slice position
                slice_pos = st.slider("Slice Position", 0, N-1, N//2)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot line profiles
                x_pos = np.linspace(extent[0], extent[1], N)
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    # Get data
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Extract slice
                    stress_slice = stress_data[slice_pos, :]
                    
                    # Plot with global line_width
                    label = f"{sim['params']['defect_type']} - {sim['params']['orientation']}"
                    line_style = config.get('line_style', 'solid')
                    ax1.plot(x_pos, stress_slice, color=color, linewidth=line_width, 
                           linestyle=line_style, label=label)
                
                ax1.set_title(f"{config['stress_component']} - Horizontal Slice", 
                            fontsize=title_font_size, fontweight='bold')
                ax1.set_xlabel("x (nm)", fontsize=label_font_size)
                ax1.set_ylabel("Stress (GPa)", fontsize=label_font_size)
                ax1.legend(fontsize=label_font_size-2)
                ax1.grid(True, alpha=0.3)
                
                # Show slice location on one of the simulations
                sim = simulations[0]
                eta, _ = sim['history'][frames[0]]
                ax2.imshow(eta, extent=extent, cmap=COLORMAPS[sim['params']['eta_cmap']], origin='lower')
                ax2.axhline(y=extent[2]+slice_pos*dx, color='white', linewidth=2)
                ax2.set_title("Slice Location", fontsize=title_font_size, fontweight='bold')
                ax2.set_xlabel("x (nm)", fontsize=label_font_size)
                ax2.set_ylabel("y (nm)", fontsize=label_font_size)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif config['type'] == "Radial Profile Comparison":
                st.subheader("🌀 Radial Stress Profile Comparison")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    # Get data
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Calculate radial profile
                    r = np.sqrt(X**2 + Y**2)
                    r_bins = np.linspace(0, np.max(r), 30)
                    radial_stress = []
                    
                    for i in range(len(r_bins)-1):
                        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
                        if np.any(mask):
                            radial_stress.append(np.nanmean(stress_data[mask]))
                        else:
                            radial_stress.append(np.nan)
                    
                    # Plot with global line_width
                    label = f"{sim['params']['defect_type']} - {sim['params']['orientation']}"
                    line_style = config.get('line_style', 'solid')
                    ax.plot(r_bins[1:], radial_stress, 'o-', color=color, 
                           linewidth=line_width, markersize=4, linestyle=line_style, label=label)
                
                ax.set_title(f"Radial {config['stress_component']} Profile", 
                           fontsize=title_font_size, fontweight='bold')
                ax.set_xlabel("Radius (nm)", fontsize=label_font_size)
                ax.set_ylabel("Average Stress (GPa)", fontsize=label_font_size)
                ax.legend(fontsize=label_font_size-2)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif config['type'] == "Statistical Summary":
                st.subheader("📊 Comprehensive Statistical Comparison")
                
                # Create box plot comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Box plot
                box_data = []
                labels = []
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key].flatten()
                    
                    box_data.append(stress_data[np.isfinite(stress_data)])
                    labels.append(f"{sim['params']['defect_type']}\n{sim['params']['orientation']}")
                
                bp = ax1.boxplot(box_data, labels=labels, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax1.set_title(f"{config['stress_component']} Distribution", 
                            fontsize=title_font_size, fontweight='bold')
                ax1.set_ylabel("Stress (GPa)", fontsize=label_font_size)
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Bar chart of maximum stresses
                max_stresses = [np.nanmax(data) for data in box_data]
                
                bars = ax2.bar(range(len(simulations)), max_stresses, color=colors, alpha=0.7)
                ax2.set_title("Maximum Stress Comparison", fontsize=title_font_size, fontweight='bold')
                ax2.set_ylabel("Max Stress (GPa)", fontsize=label_font_size)
                ax2.set_xticks(range(len(simulations)))
                ax2.set_xticklabels([f"{sim['params']['defect_type']}" for sim in simulations])
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar, val in zip(bars, max_stresses):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{val:.2f}', ha='center', va='bottom', fontsize=label_font_size-2)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed statistics table
                st.subheader("📋 Detailed Statistics")
                detailed_stats = []
                
                for idx, (sim, frame, data) in enumerate(zip(simulations, frames, box_data)):
                    detailed_stats.append({
                        'Simulation': f"{sim['params']['defect_type']} - {sim['params']['orientation']}",
                        'Count': len(data),
                        'Max': float(np.nanmax(data)),
                        'Min': float(np.nanmin(data)),
                        'Mean': float(np.nanmean(data)),
                        'Median': float(np.nanmedian(data)),
                        'Std': float(np.nanstd(data)),
                        'Q1': float(np.nanpercentile(data, 25)),
                        'Q3': float(np.nanpercentile(data, 75))
                    })
                
                df_detailed = pd.DataFrame(detailed_stats)
                st.dataframe(df_detailed.style.format({
                    'Max': '{:.3f}',
                    'Min': '{:.3f}',
                    'Mean': '{:.3f}',
                    'Median': '{:.3f}',
                    'Std': '{:.3f}',
                    'Q1': '{:.3f}',
                    'Q3': '{:.3f}'
                }), use_container_width=True)
            
            # Clear comparison flag
            if 'run_comparison' in st.session_state:
                del st.session_state.run_comparison
    
    else:
        st.info("Select simulations in the sidebar and click 'Run Comparison' to start!")
        
        # Show available simulations
        simulations = SimulationDB.get_simulation_list()
        
        if simulations:
            st.subheader("📚 Available Simulations")
            
            # Group by defect type
            defect_groups = {}
            for sim in simulations:
                defect = sim['params']['defect_type']
                if defect not in defect_groups:
                    defect_groups[defect] = []
                defect_groups[defect].append(sim)
            
            for defect_type, sims in defect_groups.items():
                with st.expander(f"{defect_type} ({len(sims)} simulations)"):
                    for sim in sims:
                        params = sim['params']
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.text(f"ID: {sim['id']}")
                        with col2:
                            st.text(f"Orientation: {params['orientation']}")
                        with col3:
                            st.text(f"ε*={params['eps0']:.2f}, κ={params['kappa']:.2f}")
        else:
            st.warning("No simulations available. Run some simulations first!")

# =============================================
# EXPORT FUNCTIONALITY
# =============================================
st.sidebar.header("💾 Export Options")

if st.sidebar.button("📥 Export All Simulations"):
    simulations = SimulationDB.get_all_simulations()
    
    if not simulations:
        st.sidebar.warning("No simulations to export!")
    else:
        with st.spinner("Creating export package..."):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # Export each simulation
                for sim_id, sim_data in simulations.items():
                    # Create directory for this simulation
                    sim_dir = f"simulation_{sim_id}"
                    
                    # Save parameters
                    params_json = json.dumps(sim_data['params'], indent=2)
                    zf.writestr(f"{sim_dir}/parameters.json", params_json)
                    
                    # Save metadata
                    metadata_json = json.dumps(sim_data['metadata'], indent=2)
                    zf.writestr(f"{sim_dir}/metadata.json", metadata_json)
                    
                    # Save each frame
                    for i, (eta, stress_fields) in enumerate(sim_data['history']):
                        # Create CSV for this frame
                        df = pd.DataFrame({
                            'eta': eta.flatten(order='F'),
                            'sxx': stress_fields['sxx'].flatten(order='F'),
                            'syy': stress_fields['syy'].flatten(order='F'),
                            'sxy': stress_fields['sxy'].flatten(order='F'),
                            'sigma_mag': stress_fields['sigma_mag'].flatten(order='F'),
                            'sigma_hydro': stress_fields['sigma_hydro'].flatten(order='F'),
                            'von_mises': stress_fields['von_mises'].flatten(order='F')
                        })
                        zf.writestr(f"{sim_dir}/frame_{i:04d}.csv", df.to_csv(index=False))
                
                # Create summary file
                summary = f"Multi-Simulation Export\n"
                summary += f"Generated: {datetime.now().isoformat()}\n"
                summary += f"Total simulations: {len(simulations)}\n\n"
                
                for sim_id, sim_data in simulations.items():
                    params = sim_data['params']
                    summary += f"Simulation {sim_id}:\n"
                    summary += f"  Defect: {params['defect_type']}\n"
                    summary += f"  Orientation: {params['orientation']}\n"
                    summary += f"  ε*: {params['eps0']}\n"
                    summary += f"  κ: {params['kappa']}\n"
                    summary += f"  Frames: {len(sim_data['history'])}\n\n"
                
                zf.writestr("README.txt", summary)
            
            buffer.seek(0)
            
            st.sidebar.download_button(
                "📥 Download Export Package",
                buffer.getvalue(),
                f"ag_np_multi_simulation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                "application/zip"
            )

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Soundness & Comparison Methodology"):
    st.markdown("""
    ### 🎯 **Multi-Simulation Comparison Platform**
    
    #### **📊 Database System:**
    - **Unique Simulation IDs**: Each simulation gets a unique hash-based ID
    - **Parameter Storage**: All simulation parameters stored with timestamps
    - **History Tracking**: Complete evolution history for each simulation
    - **Metadata**: Run time, frame count, grid parameters
    
    #### **🔬 Comparison Features:**
    1. **Side-by-Side Heatmaps**: Compare stress distributions visually
    2. **Overlay Line Profiles**: Plot multiple simulations on same axes
    3. **Radial Profile Comparison**: Compare stress decay from defect center
    4. **Statistical Summary**: Box plots, bar charts, detailed statistics
    
    #### **🎨 Visualization Options:**
    - **50+ Colormaps**: Including jet, turbo, rainbow, viridis, plasma, etc.
    - **Individual Settings**: Each simulation remembers its visualization preferences
    - **Publication-Ready**: High-quality plots with customizable styling
    
    #### **📈 Scientific Comparison Methodology:**
    
    **1. Defect Type Comparison (ISF vs ESF vs Twin):**
    - **Eigenstrain Differences**: ISF (ε*=0.707) vs ESF (ε*=1.414) vs Twin (ε*=2.121)
    - **Interface Sharpness**: Different κ values for each defect type
    - **Stress Concentration**: How different eigenstrains affect stress fields
    
    **2. Orientation Comparison:**
    - **Habit Plane Effects**: How {111} plane orientation affects stress distribution
    - **Anisotropy Analysis**: FCC silver's anisotropic elasticity response
    - **Projection Effects**: 2D projection of 3D crystallographic orientations
    
    **3. Evolution Comparison:**
    - **Frame Synchronization**: Compare at same evolution time or final state
    - **Temporal Analysis**: How different defects evolve over time
    
    ### **🔬 Key Physical Insights:**
    
    **ISF (Intrinsic Stacking Fault):**
    - Single Shockley partial dislocation
    - Moderate stress concentration
    - Well-defined habit plane orientation
    
    **ESF (Extrinsic Stacking Fault):**
    - Two partial dislocations
    - Higher stress concentration than ISF
    - Broader stress field
    
    **Twin Boundary:**
    - Coherent interface with orientation flip
    - Sharp interface (low κ)
    - Transformation strain effects
    
    ### **📊 Export Capabilities:**
    - **Complete Dataset**: All frames, all simulations
    - **Parameter Files**: JSON files with simulation settings
    - **Metadata**: Run information and statistics
    - **Ready for Post-Processing**: Compatible with ParaView, MATLAB, Python
    
    **Publication-ready multi-defect comparison platform for crystallographically accurate stress analysis!**
    """)
    
    # Display statistics
    simulations = SimulationDB.get_all_simulations()
    st.metric("Total Simulations Stored", len(simulations))

st.caption("🔬 Multi-Defect Comparison Platform • 50+ Colormaps • Cloud-style Storage • 2025")
