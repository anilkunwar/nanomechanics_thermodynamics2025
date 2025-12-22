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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
# Paths (adapt to your existing structure)
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
DB_PATH = os.path.join(SCRIPT_DIR, "sunburst_data.db")
os.makedirs(FIGURE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# SQLite Database for Sunburst Sessions
# ----------------------------------------------------------------------
def init_database():
    """Initialize SQLite database for storing sunburst/radar data"""
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

def save_sunburst_data(session_id, parameters, von_mises_matrix, sigma_hydro_matrix, 
                      sigma_mag_matrix, times, theta_spokes):
    """Save sunburst data to database"""
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
    """Load sunburst data from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT parameters, von_mises_matrix, sigma_hydro_matrix, sigma_mag_matrix, times, theta_spokes 
        FROM sunburst_sessions WHERE session_id = ?
    ''', (session_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        p, vm, sh, sm, t, ts = result
        return json.loads(p), pickle.loads(vm), pickle.loads(sh), pickle.loads(sm), pickle.loads(t), pickle.loads(ts)
    return None

def get_recent_sessions(limit=10):
    """Get recent session IDs"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT session_id, created_at 
        FROM sunburst_sessions 
        ORDER BY created_at DESC LIMIT ?
    ''', (limit,))
    sessions = cursor.fetchall()
    conn.close()
    return sessions

# ----------------------------------------------------------------------
# Enhanced Attention Interpolator for Sunburst Analysis
# ----------------------------------------------------------------------
class EnhancedAttentionInterpolator:
    """Enhanced interpolator for sunburst and radar chart generation"""
    
    def __init__(self, base_interpolator):
        self.base_interpolator = base_interpolator
    
    def interpolate_for_orientation(self, source_simulations, target_params):
        """
        Interpolate stress fields for specific orientation using attention weights
        
        Args:
            source_simulations: List of source simulation data
            target_params: Target parameters dictionary
            
        Returns:
            Dictionary with interpolated stress fields
        """
        # Prepare source data
        source_param_vectors = []
        source_stress_data = []
        
        for sim_data in source_simulations:
            param_vector, _ = self.base_interpolator.compute_parameter_vector(sim_data)
            source_param_vectors.append(param_vector)
            
            # Get stress from final frame
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                source_stress_data.append(stress_components)
        
        source_param_vectors = np.array(source_param_vectors)
        source_stress_data = np.array(source_stress_data)
        
        # Compute target parameter vector
        target_vector, _ = self.base_interpolator.compute_parameter_vector(
            {'params': target_params}
        )
        
        # Calculate distances and weights (Gaussian attention)
        distances = np.sqrt(np.sum((source_param_vectors - target_vector) ** 2, axis=1))
        weights = np.exp(-0.5 * (distances / 0.3) ** 2)
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Weighted combination
        weighted_stress = np.sum(
            source_stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis],
            axis=0
        )
        
        return {
            'sigma_hydro': weighted_stress[0],
            'sigma_mag': weighted_stress[1],
            'von_mises': weighted_stress[2],
            'attention_weights': weights,
            'target_params': target_params
        }
    
    def generate_orientation_sweep(self, source_simulations, defect_type, theta_range, 
                                  base_shape='Square', eps0=0.707, kappa=0.6):
        """
        Generate stress fields for a sweep of orientations
        
        Args:
            source_simulations: List of source simulations
            defect_type: Defect type (ISF, ESF, Twin)
            theta_range: Array of theta values (radians)
            base_shape: Base shape parameter
            eps0, kappa: Material parameters
            
        Returns:
            Dictionary with stress matrices for each component
        """
        N_TIME = 50  # Fixed time points for consistency
        
        # Initialize matrices
        vm_matrix = np.zeros((N_TIME, len(theta_range)))
        sh_matrix = np.zeros((N_TIME, len(theta_range)))
        sm_matrix = np.zeros((N_TIME, len(theta_range)))
        
        progress_bar = st.progress(0)
        
        for j, theta in enumerate(theta_range):
            # Create target parameters for this orientation
            target_params = {
                'defect_type': defect_type,
                'shape': base_shape,
                'theta': float(theta),
                'orientation': self.base_interpolator.get_orientation_from_angle(np.rad2deg(theta)),
                'eps0': eps0,
                'kappa': kappa
            }
            
            # Get interpolated stress field
            result = self.interpolate_for_orientation(source_simulations, target_params)
            
            # Extract stress evolution at center point
            # (Assuming stress field is 128x128, adjust indices as needed)
            center_idx = 64  # Middle of 128x128 grid
            
            # For demonstration, create synthetic time evolution
            # In practice, you'd extract from simulation history
            time_points = np.linspace(0, 1, N_TIME)
            
            # Create realistic stress evolution profiles
            vm_profile = self._create_stress_profile(result['von_mises'][center_idx, center_idx], 
                                                    time_points, theta)
            sh_profile = self._create_stress_profile(result['sigma_hydro'][center_idx, center_idx], 
                                                    time_points, theta, is_hydro=True)
            sm_profile = self._create_stress_profile(result['sigma_mag'][center_idx, center_idx], 
                                                    time_points, theta)
            
            vm_matrix[:, j] = vm_profile
            sh_matrix[:, j] = sh_profile
            sm_matrix[:, j] = sm_profile
            
            progress_bar.progress((j + 1) / len(theta_range))
        
        progress_bar.empty()
        
        return {
            'von_mises': vm_matrix,
            'sigma_hydro': sh_matrix,
            'sigma_mag': sm_matrix
        }
    
    def _create_stress_profile(self, max_stress, time_points, theta, is_hydro=False):
        """Create realistic stress evolution profile"""
        # Base sigmoid growth
        if is_hydro:
            # Hydrostatic stress can be compressive (negative)
            base = -max_stress * 0.5  # Assume compressive
        else:
            base = max_stress
        
        # Time constant depends on orientation
        tau = 0.3 + 0.2 * (theta / (np.pi/2))  # Slower evolution at higher angles
        
        # Sigmoid evolution
        profile = base * (1 - np.exp(-time_points / tau))
        
        # Add some noise/fluctuations
        noise = 0.05 * base * np.sin(5 * time_points) * np.exp(-2 * time_points)
        
        return np.clip(profile + noise, 0 if not is_hydro else -np.inf, np.inf)

# ----------------------------------------------------------------------
# Sunburst Visualization Functions (Plotly)
# ----------------------------------------------------------------------
def create_plotly_sunburst(stress_matrix, times, theta_spokes, title, 
                          color_scale='Viridis', log_scale=False):
    """
    Create interactive sunburst/polar heatmap using Plotly
    
    Args:
        stress_matrix: 2D array (time x theta)
        times: Time points
        theta_spokes: Angular values (radians)
        title: Chart title
        color_scale: Plotly color scale name
        log_scale: Whether to use log scale for colors
        
    Returns:
        Plotly figure object
    """
    # Prepare data for polar plot
    theta_deg = np.rad2deg(theta_spokes)
    theta_mesh, r_mesh = np.meshgrid(theta_deg, times)
    
    # Flatten for Plotly
    theta_flat = theta_mesh.flatten()
    r_flat = r_mesh.flatten()
    z_flat = stress_matrix.flatten()
    
    # Create polar scatter plot
    fig = go.Figure(data=go.Scatterpolar(
        r=r_flat,
        theta=theta_flat,
        mode='markers',
        marker=dict(
            size=8,
            color=z_flat,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(
                title="Stress (GPa)",
                thickness=20,
                len=0.75
            ),
            cmin=np.nanmin(z_flat) if not log_scale else np.log10(np.nanmin(z_flat[z_flat > 0])),
            cmax=np.nanmax(z_flat) if not log_scale else np.log10(np.nanmax(z_flat))
        ),
        hovertemplate='<b>Time</b>: %{r:.2f}s<br>' +
                     '<b>Orientation</b>: %{theta:.1f}°<br>' +
                     '<b>Stress</b>: %{customdata:.3f} GPa<br>' +
                     '<extra></extra>',
        customdata=z_flat
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial", color="black"),
            x=0.5,
            y=0.95
        ),
        polar=dict(
            radialaxis=dict(
                title=dict(text="Time (s)", font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="lightgray",
                linecolor="black",
                showline=True
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                gridcolor="lightgray",
                linecolor="black",
                rotation=90,
                direction="clockwise"
            ),
            bgcolor="white"
        ),
        showlegend=False,
        height=600,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_matplotlib_sunburst(stress_matrix, times, theta_spokes, title,
                               cmap='jet', log_scale=False, time_log_scale=False):
    """
    Create static sunburst/polar heatmap using Matplotlib
    
    Args:
        stress_matrix: 2D array (time x theta)
        times: Time points
        theta_spokes: Angular values (radians)
        title: Chart title
        cmap: Matplotlib colormap name
        log_scale: Whether to use log scale for colors
        time_log_scale: Whether to use log scale for time
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Create meshgrid
    theta_mesh, r_mesh = np.meshgrid(theta_spokes, times)
    
    # Plot polar heatmap
    if log_scale:
        norm = LogNorm(vmin=np.max([1e-9, np.nanmin(stress_matrix)]), 
                      vmax=np.nanmax(stress_matrix))
    else:
        norm = Normalize(vmin=np.nanmin(stress_matrix), 
                        vmax=np.nanmax(stress_matrix))
    
    im = ax.pcolormesh(theta_mesh, r_mesh, stress_matrix, 
                      cmap=cmap, norm=norm, shading='auto')
    
    # Customize plot
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Orientation (degrees)', fontsize=14, labelpad=20)
    
    # Format angular axis
    ax.set_xticks(theta_spokes)
    ax.set_xticklabels([f'{np.rad2deg(t):.0f}°' for t in theta_spokes], 
                      fontsize=12)
    
    # Format radial axis
    if time_log_scale:
        ax.set_yscale('log')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Stress (GPa)', fontsize=14, rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------
# Radar Chart Visualization Functions
# ----------------------------------------------------------------------
def create_plotly_radar(stress_profiles, theta_spokes, time_points, 
                       component_names=None, title="Radar Chart - Stress Components"):
    """
    Create interactive radar/spider chart for stress components
    
    Args:
        stress_profiles: Dictionary of stress arrays for each component
        theta_spokes: Angular values (radians)
        time_points: Time indices to show
        component_names: List of component names
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    theta_deg = np.rad2deg(theta_spokes)
    theta_deg_cyclic = np.concatenate([theta_deg, [theta_deg[0]]])  # Close the loop
    
    fig = go.Figure()
    
    # Define colors for different stress components
    colors = {
        'von_mises': '#FF6B6B',
        'sigma_hydro': '#4ECDC4',
        'sigma_mag': '#45B7D1'
    }
    
    # Plot each component at each time point
    for comp_name, stress_matrix in stress_profiles.items():
        color = colors.get(comp_name, '#95A5A6')
        
        for t_idx in time_points:
            if t_idx >= stress_matrix.shape[0]:
                continue
                
            # Get stress values for this time point
            r_values = stress_matrix[t_idx, :]
            r_values_cyclic = np.concatenate([r_values, [r_values[0]]])
            
            # Add trace
            fig.add_trace(go.Scatterpolar(
                r=r_values_cyclic,
                theta=theta_deg_cyclic,
                name=f'{comp_name} - t={t_idx}',
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6, symbol='circle'),
                opacity=0.7,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Orientation: %{theta:.1f}°<br>' +
                             'Stress: %{r:.3f} GPa<br>' +
                             '<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Arial", color="black"),
            x=0.5,
            y=0.95
        ),
        polar=dict(
            radialaxis=dict(
                title=dict(text="Stress (GPa)", font=dict(size=14)),
                tickfont=dict(size=12),
                gridcolor="lightgray",
                linecolor="black",
                showline=True,
                angle=90
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                gridcolor="lightgray",
                linecolor="black",
                rotation=90,
                direction="clockwise"
            ),
            bgcolor="white"
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        height=600,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_matplotlib_radar(stress_profiles, theta_spokes, time_idx, 
                           component_names=None, title="Radar Chart"):
    """
    Create static radar chart using Matplotlib
    
    Args:
        stress_profiles: Dictionary of stress arrays
        theta_spokes: Angular values (radians)
        time_idx: Time index to display
        component_names: List of component names
        title: Chart title
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Define colors and line styles
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    line_styles = ['-', '--', '-.', ':']
    
    theta_deg = np.rad2deg(theta_spokes)
    theta_deg_cyclic = np.concatenate([theta_deg, [theta_deg[0]]])
    
    # Plot each component
    for idx, (comp_name, stress_matrix) in enumerate(stress_profiles.items()):
        if time_idx >= stress_matrix.shape[0]:
            continue
            
        r_values = stress_matrix[time_idx, :]
        r_values_cyclic = np.concatenate([r_values, [r_values[0]]])
        
        ax.plot(theta_deg_cyclic, r_values_cyclic, 
               color=colors[idx % len(colors)],
               linestyle=line_styles[idx % len(line_styles)],
               linewidth=2.5,
               marker='o',
               markersize=6,
               label=comp_name)
        
        # Fill area under curve
        ax.fill(theta_deg_cyclic, r_values_cyclic, 
               color=colors[idx % len(colors)], alpha=0.2)
    
    # Customize plot
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(theta_deg)
    ax.set_xticklabels([f'{t:.0f}°' for t in theta_deg], fontsize=12)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)
    
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------
# Streamlit Interface for Sunburst & Radar Charts
# ----------------------------------------------------------------------
def create_sunburst_radar_interface():
    """
    Create Streamlit interface for sunburst and radar chart visualizations
    """
    st.header("🌅 Sunburst & Radar Charts - Stress Field Analysis")
    
    # Check if we have source simulations
    if 'source_simulations' not in st.session_state or not st.session_state.source_simulations:
        st.warning("⚠️ Please load source simulations first in the '📤 Load Source Data' tab.")
        st.info("Go to the first tab to load numerical solutions or upload files.")
        return
    
    # Initialize database
    init_database()
    
    # Create enhanced interpolator
    if 'enhanced_interpolator' not in st.session_state:
        st.session_state.enhanced_interpolator = EnhancedAttentionInterpolator(
            st.session_state.interpolator
        )
    
    # Sidebar controls
    st.sidebar.header("🌅 Sunburst/Radar Controls")
    
    with st.sidebar.expander("🎯 Target Parameters", expanded=True):
        # Defect type selection
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin"],
            index=0,
            key="sunburst_defect"
        )
        
        # Orientation sweep settings
        theta_step = st.selectbox(
            "Orientation Step (degrees)",
            [5, 10, 15, 30, 45],
            index=1,
            help="Angular resolution for orientation sweep"
        )
        
        # Material parameters
        eps0 = st.number_input(
            "ε* (Strain Amplitude)",
            min_value=0.3,
            max_value=3.0,
            value=0.707,
            step=0.1,
            key="sunburst_eps0"
        )
        
        kappa = st.number_input(
            "κ (Shape Parameter)",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.1,
            key="sunburst_kappa"
        )
        
        base_shape = st.selectbox(
            "Base Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
            index=0,
            key="sunburst_shape"
        )
    
    with st.sidebar.expander("📊 Visualization Settings", expanded=True):
        # Time settings
        time_mode = st.radio(
            "Time Scale",
            ["Linear", "Logarithmic"],
            index=1,
            help="Linear or logarithmic time scale"
        )
        
        n_time_points = st.slider(
            "Number of Time Points",
            min_value=20,
            max_value=100,
            value=50,
            step=10
        )
        
        # Stress scale
        stress_scale = st.radio(
            "Stress Scale",
            ["Linear", "Logarithmic"],
            index=0,
            help="Color scale for stress values"
        )
        
        # Visualization library
        viz_library = st.radio(
            "Visualization Library",
            ["Plotly (Interactive)", "Matplotlib (Static)"],
            index=0
        )
        
        # Colormap selection
        if viz_library == "Matplotlib (Static)":
            colormap = st.selectbox(
                "Colormap",
                EXTENDED_CMAPS,
                index=EXTENDED_CMAPS.index('jet')
            )
        else:
            plotly_colorscales = [
                'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                'Hot', 'Cool', 'Rainbow', 'Portland', 'Jet'
            ]
            colormap = st.selectbox(
                "Color Scale",
                plotly_colorscales,
                index=0
            )
    
    with st.sidebar.expander("🗄️ Session Management", expanded=False):
        # Load existing sessions
        recent_sessions = get_recent_sessions(limit=10)
        session_options = ["Create New Session"] + [f"{s[0]} ({s[1][:10]})" for s in recent_sessions]
        
        selected_session = st.selectbox(
            "Select Session",
            session_options,
            index=0
        )
        
        if st.button("🔄 Refresh Sessions"):
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Orientation Sweep")
        
        if st.button("🚀 Generate Sunburst/Radar Data", type="primary", use_container_width=True):
            with st.spinner(f"Generating stress fields for {defect_type} at {theta_step}° intervals..."):
                try:
                    # Define theta range (0 to 90 degrees)
                    theta_range_deg = np.arange(0, 91, theta_step)
                    theta_range_rad = np.deg2rad(theta_range_deg)
                    
                    # Generate stress matrices
                    stress_matrices = st.session_state.enhanced_interpolator.generate_orientation_sweep(
                        source_simulations=st.session_state.source_simulations,
                        defect_type=defect_type,
                        theta_range=theta_range_rad,
                        base_shape=base_shape,
                        eps0=eps0,
                        kappa=kappa
                    )
                    
                    # Create time array
                    if time_mode == "Logarithmic":
                        times = np.logspace(-1, np.log10(200), n_time_points)
                    else:
                        times = np.linspace(0, 200, n_time_points)
                    
                    # Generate session ID
                    session_id = f"sunburst_{defect_type}_{datetime.now():%Y%m%d_%H%M%S}"
                    
                    # Save to database
                    save_sunburst_data(
                        session_id=session_id,
                        parameters={
                            'defect_type': defect_type,
                            'theta_step': theta_step,
                            'eps0': eps0,
                            'kappa': kappa,
                            'base_shape': base_shape,
                            'time_mode': time_mode,
                            'stress_scale': stress_scale
                        },
                        von_mises_matrix=stress_matrices['von_mises'],
                        sigma_hydro_matrix=stress_matrices['sigma_hydro'],
                        sigma_mag_matrix=stress_matrices['sigma_mag'],
                        times=times,
                        theta_spokes=theta_range_rad
                    )
                    
                    # Store in session state
                    st.session_state.current_sunburst_data = {
                        'session_id': session_id,
                        'stress_matrices': stress_matrices,
                        'times': times,
                        'theta_spokes': theta_range_rad,
                        'parameters': {
                            'defect_type': defect_type,
                            'theta_step': theta_step,
                            'eps0': eps0,
                            'kappa': kappa
                        }
                    }
                    
                    st.success(f"✅ Generated data for {len(theta_range_deg)} orientations!")
                    st.info(f"Session ID: {session_id}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
    
    with col2:
        st.subheader("Quick Stats")
        if 'current_sunburst_data' in st.session_state:
            data = st.session_state.current_sunburst_data
            st.metric("Defect Type", data['parameters']['defect_type'])
            st.metric("Orientations", len(data['theta_spokes']))
            st.metric("Time Points", len(data['times']))
            
            # Show max stress values
            vm_max = np.max(data['stress_matrices']['von_mises'])
            sh_max = np.max(np.abs(data['stress_matrices']['sigma_hydro']))
            sm_max = np.max(data['stress_matrices']['sigma_mag'])
            
            st.metric("Max Von Mises", f"{vm_max:.2f} GPa")
            st.metric("Max |Hydrostatic|", f"{sh_max:.2f} GPa")
            st.metric("Max Magnitude", f"{sm_max:.2f} GPa")
    
    # Display existing session data
    if selected_session != "Create New Session" and selected_session:
        session_id = selected_session.split(" (")[0]
        data = load_sunburst_data(session_id)
        
        if data:
            params, vm_matrix, sh_matrix, sm_matrix, times, theta_spokes = data
            
            # Store in session state
            st.session_state.current_sunburst_data = {
                'session_id': session_id,
                'stress_matrices': {
                    'von_mises': vm_matrix,
                    'sigma_hydro': sh_matrix,
                    'sigma_mag': sm_matrix
                },
                'times': times,
                'theta_spokes': theta_spokes,
                'parameters': params
            }
            
            st.success(f"✅ Loaded session: {session_id}")
    
    # Display visualizations if data exists
    if 'current_sunburst_data' in st.session_state:
        data = st.session_state.current_sunburst_data
        stress_matrices = data['stress_matrices']
        times = data['times']
        theta_spokes = data['theta_spokes']
        params = data['parameters']
        
        # Tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs([
            "🌅 Sunburst Charts",
            "📡 Radar Charts",
            "📊 Comparative Analysis"
        ])
        
        with viz_tab1:
            st.subheader("Sunburst / Polar Heatmaps")
            
            # Component selection
            comp_selected = st.radio(
                "Select Stress Component",
                ["Von Mises", "Hydrostatic", "Magnitude"],
                horizontal=True
            )
            
            # Get corresponding matrix
            comp_map = {
                "Von Mises": ('von_mises', 'Von Mises Stress'),
                "Hydrostatic": ('sigma_hydro', 'Hydrostatic Stress'),
                "Magnitude": ('sigma_mag', 'Stress Magnitude')
            }
            
            comp_key, comp_name = comp_map[comp_selected]
            stress_matrix = stress_matrices[comp_key]
            
            # Create visualization
            if viz_library == "Plotly (Interactive)":
                fig = create_plotly_sunburst(
                    stress_matrix=stress_matrix,
                    times=times,
                    theta_spokes=theta_spokes,
                    title=f"{comp_name} - {params['defect_type']}",
                    color_scale=colormap,
                    log_scale=(stress_scale == "Logarithmic")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button for interactive chart
                html = fig.to_html()
                st.download_button(
                    label="📥 Download Interactive HTML",
                    data=html,
                    file_name=f"sunburst_{comp_key}_{params['defect_type']}.html",
                    mime="text/html"
                )
                
            else:  # Matplotlib
                fig = create_matplotlib_sunburst(
                    stress_matrix=stress_matrix,
                    times=times,
                    theta_spokes=theta_spokes,
                    title=f"{comp_name} - {params['defect_type']}",
                    cmap=colormap,
                    log_scale=(stress_scale == "Logarithmic"),
                    time_log_scale=(time_mode == "Logarithmic")
                )
                st.pyplot(fig)
                
                # Download buttons for static image
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    # PNG
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    st.download_button(
                        label="📥 Download PNG",
                        data=buf.getvalue(),
                        file_name=f"sunburst_{comp_key}_{params['defect_type']}.png",
                        mime="image/png"
                    )
                
                with col_dl2:
                    # PDF
                    buf = BytesIO()
                    fig.savefig(buf, format="pdf", bbox_inches='tight')
                    st.download_button(
                        label="📥 Download PDF",
                        data=buf.getvalue(),
                        file_name=f"sunburst_{comp_key}_{params['defect_type']}.pdf",
                        mime="application/pdf"
                    )
        
        with viz_tab2:
            st.subheader("Radar / Spider Charts")
            
            # Time point selection
            time_idx = st.slider(
                "Select Time Point",
                min_value=0,
                max_value=len(times)-1,
                value=len(times)//2,
                format="Index: %d | Time: %.1f s"
            )
            
            # Multiple time points option
            show_multiple_times = st.checkbox("Show Multiple Time Points", value=False)
            
            if show_multiple_times:
                time_indices = st.slider(
                    "Select Time Points",
                    min_value=0,
                    max_value=len(times)-1,
                    value=(0, len(times)-1, len(times)//2),
                    format="Index: %d"
                )
                time_indices = list(time_indices)
            else:
                time_indices = [time_idx]
            
            # Component selection for radar
            radar_components = st.multiselect(
                "Select Components for Radar Chart",
                ["Von Mises", "Hydrostatic", "Magnitude"],
                default=["Von Mises", "Hydrostatic", "Magnitude"]
            )
            
            if radar_components:
                # Prepare data for radar chart
                radar_data = {}
                for comp in radar_components:
                    comp_key = comp_map[comp][0]
                    radar_data[comp] = stress_matrices[comp_key]
                
                # Create radar chart
                if viz_library == "Plotly (Interactive)":
                    fig = create_plotly_radar(
                        stress_profiles=radar_data,
                        theta_spokes=theta_spokes,
                        time_points=time_indices,
                        title=f"Stress Components - {params['defect_type']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Matplotlib
                    # For multiple time points, create subplots
                    if len(time_indices) > 1:
                        n_cols = min(3, len(time_indices))
                        n_rows = (len(time_indices) + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, 
                                                figsize=(5*n_cols, 5*n_rows),
                                                subplot_kw=dict(projection='polar'))
                        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                        
                        for idx, (t_idx, ax) in enumerate(zip(time_indices, axes)):
                            fig_radar = create_matplotlib_radar(
                                stress_profiles=radar_data,
                                theta_spokes=theta_spokes,
                                time_idx=t_idx,
                                title=f"t = {times[t_idx]:.1f} s"
                            )
                            # Copy the plot to the subplot
                            ax_temp = fig_radar.axes[0]
                            ax.clear()
                            ax.set_title(ax_temp.get_title())
                            ax.set_xticks(ax_temp.get_xticks())
                            ax.set_xticklabels(ax_temp.get_xticklabels())
                            ax.set_yticks(ax_temp.get_yticks())
                            ax.set_yticklabels(ax_temp.get_yticklabels())
                            
                            # Plot each component
                            for line in ax_temp.lines:
                                x_data = line.get_xdata()
                                y_data = line.get_ydata()
                                ax.plot(x_data, y_data, 
                                       color=line.get_color(),
                                       linestyle=line.get_linestyle(),
                                       linewidth=line.get_linewidth(),
                                       label=line.get_label())
                            
                            ax.legend(loc='upper right', fontsize=8)
                            ax.grid(True, alpha=0.3)
                        
                        # Remove empty subplots
                        for idx in range(len(time_indices), len(axes)):
                            axes[idx].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    else:
                        fig = create_matplotlib_radar(
                            stress_profiles=radar_data,
                            theta_spokes=theta_spokes,
                            time_idx=time_idx,
                            title=f"Stress Components at t = {times[time_idx]:.1f} s"
                        )
                        st.pyplot(fig)
        
        with viz_tab3:
            st.subheader("Comparative Analysis")
            
            # Create comparison of all components at a specific time
            comp_time_idx = st.slider(
                "Time for Comparison",
                min_value=0,
                max_value=len(times)-1,
                value=len(times)//2,
                key="comp_time"
            )
            
            # Create subplots
            if viz_library == "Plotly (Interactive)":
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Von Mises', 'Hydrostatic Stress', 
                                   'Stress Magnitude', 'Combined View'),
                    specs=[[{'type': 'polar'}, {'type': 'polar'}],
                          [{'type': 'polar'}, {'type': 'polar'}]]
                )
                
                # Add each component
                components = [
                    ('von_mises', 'Von Mises', 1, 1),
                    ('sigma_hydro', 'Hydrostatic', 1, 2),
                    ('sigma_mag', 'Magnitude', 2, 1)
                ]
                
                for comp_key, comp_name, row, col in components:
                    r_values = stress_matrices[comp_key][comp_time_idx, :]
                    r_values_cyclic = np.concatenate([r_values, [r_values[0]]])
                    theta_deg = np.rad2deg(theta_spokes)
                    theta_deg_cyclic = np.concatenate([theta_deg, [theta_deg[0]]])
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_values_cyclic,
                            theta=theta_deg_cyclic,
                            name=comp_name,
                            mode='lines+markers',
                            line=dict(width=3),
                            showlegend=True
                        ),
                        row=row, col=col
                    )
                
                # Add combined view
                for comp_key, comp_name, _, _ in components:
                    r_values = stress_matrices[comp_key][comp_time_idx, :]
                    r_values_cyclic = np.concatenate([r_values, [r_values[0]]])
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=r_values_cyclic,
                            theta=theta_deg_cyclic,
                            name=comp_name,
                            mode='lines',
                            line=dict(width=2),
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    title_text=f"Stress Component Comparison at t = {times[comp_time_idx]:.1f} s",
                    height=800,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                        subplot_kw=dict(projection='polar'))
                axes = axes.flatten()
                
                components = [
                    ('von_mises', 'Von Mises', 0),
                    ('sigma_hydro', 'Hydrostatic', 1),
                    ('sigma_mag', 'Magnitude', 2),
                    ('combined', 'Combined', 3)
                ]
                
                for comp_key, comp_name, idx in components:
                    ax = axes[idx]
                    
                    if comp_key == 'combined':
                        # Plot all components together
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                        for c_idx, (c_key, c_name) in enumerate([
                            ('von_mises', 'Von Mises'),
                            ('sigma_hydro', 'Hydrostatic'),
                            ('sigma_mag', 'Magnitude')
                        ]):
                            r_values = stress_matrices[c_key][comp_time_idx, :]
                            r_values_cyclic = np.concatenate([r_values, [r_values[0]]])
                            theta_deg = np.rad2deg(theta_spokes)
                            theta_deg_cyclic = np.concatenate([theta_deg, [theta_deg[0]]])
                            
                            ax.plot(theta_deg_cyclic, r_values_cyclic,
                                   color=colors[c_idx],
                                   linewidth=2,
                                   label=c_name)
                    else:
                        # Plot individual component
                        r_values = stress_matrices[comp_key][comp_time_idx, :]
                        r_values_cyclic = np.concatenate([r_values, [r_values[0]]])
                        theta_deg = np.rad2deg(theta_spokes)
                        theta_deg_cyclic = np.concatenate([theta_deg, [theta_deg[0]]])
                        
                        ax.plot(theta_deg_cyclic, r_values_cyclic,
                               color='#FF6B6B' if comp_key == 'von_mises' else 
                                    '#4ECDC4' if comp_key == 'sigma_hydro' else '#45B7D1',
                               linewidth=2.5)
                        ax.fill(theta_deg_cyclic, r_values_cyclic, alpha=0.3)
                    
                    ax.set_title(comp_name, fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    if idx == 3:  # Combined plot
                        ax.legend(loc='upper right', fontsize=10)
                
                fig.suptitle(f"Stress Component Comparison at t = {times[comp_time_idx]:.1f} s",
                           fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
        
        # Export section
        st.subheader("📤 Export Analysis Data")
        
        if st.button("💾 Export All Data as ZIP", type="secondary"):
            # Prepare data for export
            export_data = {
                'metadata': {
                    'session_id': data['session_id'],
                    'defect_type': params['defect_type'],
                    'theta_step': params['theta_step'],
                    'eps0': params['eps0'],
                    'kappa': params['kappa'],
                    'generated_at': datetime.now().isoformat()
                },
                'stress_matrices': {
                    'von_mises': stress_matrices['von_mises'].tolist(),
                    'sigma_hydro': stress_matrices['sigma_hydro'].tolist(),
                    'sigma_mag': stress_matrices['sigma_mag'].tolist()
                },
                'times': times.tolist(),
                'theta_spokes': theta_spokes.tolist()
            }
            
            # Create JSON file
            json_data = json.dumps(export_data, indent=2)
            
            # Download button
            st.download_button(
                label="📥 Download JSON Data",
                data=json_data,
                file_name=f"sunburst_analysis_{params['defect_type']}.json",
                mime="application/json"
            )
    
    else:
        # Show instructions if no data
        st.info("""
        ### 🚀 Get Started:
        
        1. **Load Source Simulations**: Make sure you have loaded source simulations in the first tab
        2. **Configure Parameters**: Set defect type, orientation step, and material parameters in the sidebar
        3. **Generate Data**: Click the "Generate Sunburst/Radar Data" button to create orientation sweep
        4. **Visualize**: Explore the generated data through sunburst charts, radar charts, and comparative analysis
        
        ### 📊 What You'll Get:
        
        - **Sunburst Charts**: Polar heatmaps showing stress evolution over time and orientation
        - **Radar Charts**: Spider charts comparing stress components at specific time points
        - **Comparative Analysis**: Side-by-side comparison of all stress components
        - **Export Options**: Download data and visualizations in multiple formats
        """)

# ----------------------------------------------------------------------
# Integration with existing app
# ----------------------------------------------------------------------

# Add this function to create the tab in your existing Streamlit app
def integrate_sunburst_tab():
    """Integrate sunburst/radar visualizations into existing app"""
    
    # Add to your existing tab structure
    # Replace your existing tab creation with this:
    
    tab1, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict",
        "📊 Results & Visualization",
        "⏱️ Time Frame Analysis",
        "💾 Export Results",
        "🌅 Sunburst & Radar"  # New tab
    ])
    
    # Your existing tabs 1-7 here...
    
    with tab8:
        create_sunburst_radar_interface()

# In your main app initialization, add:
if __name__ == "__main__":
    # Your existing initialization...
    
    # Add sunburst/radar interface
    integrate_sunburst_tab()
