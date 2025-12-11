import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import warnings
import pickle
import torch
import sqlite3
from io import StringIO
import traceback

warnings.filterwarnings('ignore')

# =============================================
# ERROR HANDLING DECORATOR
# =============================================
def handle_errors(func):
    """Decorator to handle errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            st.error("Please check the console for detailed error information.")
            print(f"Error in {func.__name__}: {str(e)}")
            print(traceback.format_exc())
            return None
    return wrapper

# =============================================
# METADATA MANAGEMENT CLASS
# =============================================
class MetadataManager:
    """Centralized metadata management to ensure consistency"""
    
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        """Create standardized metadata dictionary"""
        if run_time is None:
            run_time = 0.0
            
        metadata = {
            'run_time': run_time,
            'frames': len(history) if history else 0,
            'grid_size': kwargs.get('grid_size', 128),
            'dx': kwargs.get('dx', 0.1),
            'created_at': datetime.now().isoformat(),
            'colormaps': kwargs.get('colormaps', {
                'eta': sim_params.get('eta_cmap', 'viridis'),
                'sigma': sim_params.get('sigma_cmap', 'hot'),
                'hydro': sim_params.get('hydro_cmap', 'coolwarm'),
                'vm': sim_params.get('vm_cmap', 'plasma')
            }),
            'material_properties': {
                'C11': 124.0,
                'C12': 93.4,
                'C44': 46.1,
                'lattice_constant': 0.4086
            },
            'simulation_parameters': {
                'dt': 0.004,
                'N': kwargs.get('grid_size', 128),
                'dx': kwargs.get('dx', 0.1)
            }
        }
        return metadata
    
    @staticmethod
    def validate_metadata(metadata):
        """Validate metadata structure and add missing fields"""
        if not isinstance(metadata, dict):
            metadata = {}
        
        required_fields = [
            'run_time', 'frames', 'grid_size', 'dx', 'created_at'
        ]
        
        for field in required_fields:
            if field not in metadata:
                if field == 'created_at':
                    metadata[field] = datetime.now().isoformat()
                elif field == 'run_time':
                    metadata[field] = 0.0
                elif field == 'frames':
                    metadata[field] = 0
                elif field == 'grid_size':
                    metadata[field] = 128
                elif field == 'dx':
                    metadata[field] = 0.1
        
        # Ensure colormaps exist
        if 'colormaps' not in metadata:
            metadata['colormaps'] = {
                'eta': 'viridis',
                'sigma': 'hot',
                'hydro': 'coolwarm',
                'vm': 'plasma'
            }
        
        return metadata
    
    @staticmethod
    def get_metadata_field(metadata, field, default=None):
        """Safely get metadata field with default"""
        try:
            return metadata.get(field, default)
        except:
            return default

# Configure page with better styling
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Comparison Platform")
st.markdown("""
**Run multiple simulations • Compare ISF/ESF/Twin with different orientations • Cloud-style storage**
**Run → Save → Compare • 50+ Colormaps • Publication-ready comparison plots • Advanced Post-Processing**
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
# ENHANCED LINE PROFILE SYSTEM
# =============================================
class EnhancedLineProfiler:
    """Enhanced line profile system with multiple orientations and proper scaling"""
    
    @staticmethod
    def extract_profile(data, profile_type, position_ratio=0.5, angle_deg=45):
        """
        Extract line profiles from 2D data with proper scaling
        
        Parameters:
        -----------
        data : 2D numpy array
            Input data (stress or defect field)
        profile_type : str
            Type of profile: 'horizontal', 'vertical', 'diagonal', 'anti_diagonal', 'custom'
        position_ratio : float
            Position ratio from center (0.0 to 1.0)
        angle_deg : float
            Angle for custom line profiles (degrees)
            
        Returns:
        --------
        distance : 1D array
            Distance along profile (nm)
        profile : 1D array
            Extracted profile values
        endpoints : tuple
            (x_start, y_start, x_end, y_end) in data coordinates
        """
        ny, nx = data.shape
        center_x, center_y = nx // 2, ny // 2
        
        # Calculate position offset based on ratio
        if profile_type in ['horizontal', 'vertical']:
            offset = int(min(nx, ny) * 0.4 * position_ratio)
        else:
            offset = int(min(nx, ny) * 0.3 * position_ratio)
        
        if profile_type == 'horizontal':
            # Horizontal profile
            row_idx = center_y + offset
            profile = data[row_idx, :]
            distance = np.linspace(extent[0], extent[1], nx)
            endpoints = (extent[0], row_idx * dx + extent[2], 
                        extent[1], row_idx * dx + extent[2])
            
        elif profile_type == 'vertical':
            # Vertical profile
            col_idx = center_x + offset
            profile = data[:, col_idx]
            distance = np.linspace(extent[2], extent[3], ny)
            endpoints = (col_idx * dx + extent[0], extent[2],
                        col_idx * dx + extent[0], extent[3])
            
        elif profile_type == 'diagonal':
            # Main diagonal (top-left to bottom-right)
            # Extract diagonal through center
            diag_length = int(min(nx, ny) * 0.8)  # Use 80% of min dimension
            start_idx = (center_x - diag_length//2, center_y - diag_length//2)
            
            profile = []
            distances = []
            for i in range(diag_length):
                x = start_idx[0] + i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    # Calculate actual distance along diagonal in nm
                    dist = i * dx * np.sqrt(2)
                    distances.append(dist - (diag_length//2) * dx * np.sqrt(2))
            
            distance = np.array(distances)
            profile = np.array(profile)
            
            # Calculate endpoints in physical coordinates
            x_start = start_idx[0] * dx + extent[0]
            y_start = start_idx[1] * dx + extent[2]
            x_end = (start_idx[0] + diag_length - 1) * dx + extent[0]
            y_end = (start_idx[1] + diag_length - 1) * dx + extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
            
        elif profile_type == 'anti_diagonal':
            # Anti-diagonal (top-right to bottom-left)
            diag_length = int(min(nx, ny) * 0.8)
            start_idx = (center_x + diag_length//2, center_y - diag_length//2)
            
            profile = []
            distances = []
            for i in range(diag_length):
                x = start_idx[0] - i
                y = start_idx[1] + i
                if 0 <= x < nx and 0 <= y < ny:
                    profile.append(data[y, x])
                    dist = i * dx * np.sqrt(2)
                    distances.append(dist - (diag_length//2) * dx * np.sqrt(2))
            
            distance = np.array(distances)
            profile = np.array(profile)
            
            x_start = start_idx[0] * dx + extent[0]
            y_start = start_idx[1] * dx + extent[2]
            x_end = (start_idx[0] - diag_length + 1) * dx + extent[0]
            y_end = (start_idx[1] + diag_length - 1) * dx + extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
            
        elif profile_type == 'custom':
            # Custom angle line profile
            angle_rad = np.deg2rad(angle_deg)
            length = int(min(nx, ny) * 0.8)
            
            # Calculate line endpoints
            dx_line = np.cos(angle_rad) * length//2
            dy_line = np.sin(angle_rad) * length//2
            
            profile = []
            distances = []
            
            # Interpolate along line
            for t in np.linspace(-length//2, length//2, length):
                x = center_x + t * np.cos(angle_rad) + offset * np.cos(angle_rad + np.pi/2)
                y = center_y + t * np.sin(angle_rad) + offset * np.sin(angle_rad + np.pi/2)
                
                if 0 <= x < nx-1 and 0 <= y < ny-1:
                    # Bilinear interpolation
                    x0, y0 = int(x), int(y)
                    x1, y1 = x0 + 1, y0 + 1
                    
                    # Check bounds
                    if x1 >= nx: x1 = nx - 1
                    if y1 >= ny: y1 = ny - 1
                    
                    # Interpolation weights
                    wx = x - x0
                    wy = y - y0
                    
                    # Bilinear interpolation
                    val = (data[y0, x0] * (1-wx) * (1-wy) +
                          data[y0, x1] * wx * (1-wy) +
                          data[y1, x0] * (1-wx) * wy +
                          data[y1, x1] * wx * wy)
                    
                    profile.append(val)
                    distances.append(t * dx)
            
            distance = np.array(distances)
            profile = np.array(profile)
            
            # Calculate endpoints
            x_start = (center_x - dx_line + offset * np.cos(angle_rad + np.pi/2)) * dx + extent[0]
            y_start = (center_y - dy_line + offset * np.sin(angle_rad + np.pi/2)) * dx + extent[2]
            x_end = (center_x + dx_line + offset * np.cos(angle_rad + np.pi/2)) * dx + extent[0]
            y_end = (center_y + dy_line + offset * np.sin(angle_rad + np.pi/2)) * dx + extent[2]
            endpoints = (x_start, y_start, x_end, y_end)
        
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        return distance, profile, endpoints
    
    @staticmethod
    def extract_multiple_profiles(data, profile_types, position_ratio=0.5, angle_deg=45):
        """
        Extract multiple line profiles from the same data
        """
        profiles = {}
        for profile_type in profile_types:
            distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                data, profile_type, position_ratio, angle_deg
            )
            profiles[profile_type] = {
                'distance': distance,
                'profile': profile,
                'endpoints': endpoints
            }
        return profiles
    
    @staticmethod
    def plot_profile_locations(ax, data, profile_configs, cmap='viridis', alpha=0.7):
        """
        Plot data with overlay of profile lines
        """
        # Plot the base data
        im = ax.imshow(data, extent=extent, cmap=cmap, origin='lower', aspect='equal')
        
        # Define colors for different profile types
        profile_colors = {
            'horizontal': 'red',
            'vertical': 'blue',
            'diagonal': 'green',
            'anti_diagonal': 'purple',
            'custom': 'orange'
        }
        
        # Plot each profile line
        for profile_type, config in profile_configs.items():
            if profile_type in config['profiles']:
                endpoints = config['profiles'][profile_type]['endpoints']
                color = profile_colors.get(profile_type, 'white')
                
                # Draw line
                ax.plot([endpoints[0], endpoints[2]], [endpoints[1], endpoints[3]],
                       color=color, linewidth=2, alpha=alpha, 
                       linestyle='--' if profile_type == 'custom' else '-',
                       label=f"{profile_type.title()} Profile")
        
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.legend(loc='upper right', fontsize=8)
        
        return im, ax

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
# JOURNAL-SPECIFIC STYLING TEMPLATES
# =============================================
class JournalTemplates:
    """Publication-quality journal templates"""
    
    @staticmethod
    def get_journal_styles():
        """Return journal-specific style parameters"""
        return {
            'nature': {
                'figure_width_single': 8.9,  # cm to inches
                'figure_width_double': 18.3,
                'font_family': 'Arial',
                'font_size_small': 7,
                'font_size_medium': 8,
                'font_size_large': 9,
                'line_width': 0.5,
                'axes_linewidth': 0.5,
                'tick_width': 0.5,
                'tick_length': 2,
                'grid_alpha': 0.1,
                'dpi': 600,
                'color_cycle': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            },
            'science': {
                'figure_width_single': 5.5,
                'figure_width_double': 11.4,
                'font_family': 'Helvetica',
                'font_size_small': 8,
                'font_size_medium': 9,
                'font_size_large': 10,
                'line_width': 0.75,
                'axes_linewidth': 0.75,
                'tick_width': 0.75,
                'tick_length': 3,
                'grid_alpha': 0.15,
                'dpi': 600,
                'color_cycle': ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
                              '#4DBEEE', '#A2142F', '#FF00FF', '#00FFFF', '#FFA500']
            },
            'advanced_materials': {
                'figure_width_single': 8.6,
                'figure_width_double': 17.8,
                'font_family': 'Arial',
                'font_size_small': 8,
                'font_size_medium': 9,
                'font_size_large': 10,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0.2,
                'dpi': 600,
                'color_cycle': ['#004488', '#DDAA33', '#BB5566', '#000000', '#44AA99',
                              '#882255', '#117733', '#999933', '#AA4499', '#88CCEE']
            },
            'prl': {
                'figure_width_single': 3.4,
                'figure_width_double': 7.0,
                'font_family': 'Times New Roman',
                'font_size_small': 8,
                'font_size_medium': 10,
                'font_size_large': 12,
                'line_width': 1.0,
                'axes_linewidth': 1.0,
                'tick_width': 1.0,
                'tick_length': 4,
                'grid_alpha': 0,
                'dpi': 600,
                'color_cycle': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                              '#0072B2', '#D55E00', '#CC79A7', '#999999', '#FFFFFF']
            },
            'custom': {
                'figure_width_single': 6.0,
                'figure_width_double': 12.0,
                'font_family': 'DejaVu Sans',
                'font_size_small': 10,
                'font_size_medium': 12,
                'font_size_large': 14,
                'line_width': 1.5,
                'axes_linewidth': 1.5,
                'tick_width': 1.0,
                'tick_length': 5,
                'grid_alpha': 0.3,
                'dpi': 300,
                'color_cycle': plt.cm.Set2(np.linspace(0, 1, 10))
            }
        }
    
    @staticmethod
    def apply_journal_style(fig, axes, journal_name='nature'):
        """Apply journal-specific styling to figure"""
        styles = JournalTemplates.get_journal_styles()
        style = styles.get(journal_name, styles['nature'])
        
        # Set rcParams for consistent styling
        rcParams.update({
            'font.family': style['font_family'],
            'font.size': style['font_size_medium'],
            'axes.linewidth': style['axes_linewidth'],
            'axes.labelsize': style['font_size_medium'],
            'axes.titlesize': style['font_size_large'],
            'xtick.labelsize': style['font_size_small'],
            'ytick.labelsize': style['font_size_small'],
            'legend.fontsize': style['font_size_small'],
            'figure.titlesize': style['font_size_large'],
            'lines.linewidth': style['line_width'],
            'lines.markersize': 4,
            'xtick.major.width': style['tick_width'],
            'ytick.major.width': style['tick_width'],
            'xtick.minor.width': style['tick_width'] * 0.5,
            'ytick.minor.width': style['tick_width'] * 0.5,
            'xtick.major.size': style['tick_length'],
            'ytick.major.size': style['tick_length'],
            'xtick.minor.size': style['tick_length'] * 0.6,
            'ytick.minor.size': style['tick_length'] * 0.6,
            'axes.grid': False,
            'savefig.dpi': style['dpi'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.prop_cycle': plt.cycler(color=style['color_cycle'])
        })
        
        # Apply to all axes
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        for ax in axes_flat:
            if ax is not None:
                # Add minor ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                # Set spine visibility
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['top'].set_linewidth(style['axes_linewidth'] * 0.5)
                ax.spines['right'].set_linewidth(style['axes_linewidth'] * 0.5)
                
                # Improve tick formatting
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=style['tick_length'])
                ax.tick_params(which='minor', length=style['tick_length'] * 0.6)
        
        return fig, style

# =============================================
# SIMULATION DATABASE SYSTEM (Session State)
# =============================================
class SimulationDB:
    """In-memory simulation database for storing and retrieving simulations"""
    
    @staticmethod
    def generate_id(sim_params):
        """Generate unique ID for simulation"""
        try:
            param_str = json.dumps(sim_params, sort_keys=True)
            return hashlib.md5(param_str.encode()).hexdigest()[:8]
        except:
            # Fallback to timestamp-based ID
            return datetime.now().strftime("%H%M%S%f")[:8]
    
    @staticmethod
    @handle_errors
    def save_simulation(sim_params, history, metadata=None):
        """Save simulation to database"""
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        
        sim_id = SimulationDB.generate_id(sim_params)
        
        # Ensure metadata is properly structured
        if metadata is None:
            metadata = MetadataManager.create_metadata(sim_params, history)
        else:
            metadata = MetadataManager.validate_metadata(metadata)
        
        # Store simulation data
        st.session_state.simulations[sim_id] = {
            'id': sim_id,
            'params': sim_params,
            'history': history,
            'metadata': metadata,
            'created_at': metadata.get('created_at', datetime.now().isoformat()),
            'last_modified': datetime.now().isoformat()
        }
        
        return sim_id
    
    @staticmethod
    @handle_errors
    def get_simulation(sim_id):
        """Retrieve simulation by ID"""
        if 'simulations' in st.session_state and sim_id in st.session_state.simulations:
            sim_data = st.session_state.simulations[sim_id]
            # Ensure metadata is validated
            if 'metadata' in sim_data:
                sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return sim_data
        return None
    
    @staticmethod
    def get_all_simulations():
        """Get all stored simulations"""
        if 'simulations' in st.session_state:
            # Validate metadata for all simulations
            for sim_id, sim_data in st.session_state.simulations.items():
                if 'metadata' in sim_data:
                    sim_data['metadata'] = MetadataManager.validate_metadata(sim_data['metadata'])
            return st.session_state.simulations
        return {}
    
    @staticmethod
    @handle_errors
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
            try:
                params = sim_data.get('params', {})
                name = f"{params.get('defect_type', 'Unknown')} - {params.get('orientation', 'Unknown')} (ε*={params.get('eps0', 0.0):.2f}, κ={params.get('kappa', 0.0):.2f})"
                simulations.append({
                    'id': sim_id,
                    'name': name,
                    'params': params
                })
            except:
                # Skip malformed simulations
                continue
        
        return simulations

# =============================================
# HELPER FUNCTIONS
# =============================================
@handle_errors
def sanitize_token(text: str) -> str:
    """Sanitize small text tokens for filenames: remove spaces and special braces."""
    try:
        s = str(text)
        for ch in [" ", "{", "}", "/", "\\", ",", ";", "(", ")", "[", "]", "°"]:
            s = s.replace(ch, "")
        return s
    except:
        return "unknown"

@handle_errors
def fmt_num_trim(x, ndigits=3):
    """Format a float with up to ndigits decimals and strip trailing zeros."""
    try:
        s = f"{x:.{ndigits}f}"
        s = s.rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
        return s
    except:
        return "0.0"

@handle_errors
def build_sim_name(params: dict, sim_id: str = None) -> str:
    """
    Build a filename-friendly simulation name in the requested format:
    Example: ISF_orient0deg_Square_eps0-0.707_kappa-0.6[_<simid>]
    If sim_id provided, append short id to guarantee uniqueness.
    """
    try:
        defect = sanitize_token(params.get("defect_type", "def"))
        shape = sanitize_token(params.get("shape", "shape"))
        # Angle in degrees from theta
        theta = params.get("theta", 0.0)
        try:
            angle_deg = int(round(np.rad2deg(theta)))
        except:
            angle_deg = 0
        orient_token = f"orient{angle_deg}deg"
        eps0 = fmt_num_trim(params.get("eps0", 0.0), ndigits=3)
        kappa = fmt_num_trim(params.get("kappa", 0.0), ndigits=3)
        name = f"{defect}_{orient_token}_{shape}_eps0-{eps0}_kappa-{kappa}"
        if sim_id:
            name = f"{name}_{sim_id}"
        return name
    except:
        return f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# =============================================
# SIMULATION ENGINE (Reusable Functions)
# =============================================
@handle_errors
def create_initial_eta(shape, defect_type):
    """Create initial defect configuration"""
    # Set initial amplitude based on defect type
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes.get(defect_type, 0.7)
    
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
    else:
        # Default to square
        eta[cy-h:cy+h, cx-h:cx+h] = init_amplitude
    
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
@handle_errors
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

@handle_errors
def run_simulation(sim_params):
    """Run a complete simulation with given parameters"""
    # Create initial defect
    eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
    
    # Run evolution
    history = []
    steps = sim_params.get('steps', 100)
    save_every = sim_params.get('save_every', 20)
    
    for step in range(steps + 1):
        if step > 0:
            eta = evolve_phase_field(eta, sim_params['kappa'], dt=0.004, dx=dx, N=N)
        if step % save_every == 0 or step == steps:
            stress_fields = compute_stress_fields(eta, sim_params['eps0'], sim_params['theta'])
            history.append((eta.copy(), stress_fields))
    
    return history

# =============================================
# SIDEBAR - Global Settings
# =============================================
st.sidebar.header("🔄 Cache Management")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🗑️ Clear All", type="secondary", help="Clear all simulations from memory"):
        if 'simulations' in st.session_state:
            del st.session_state.simulations
        st.success("All simulations cleared!")
        st.rerun()

with col2:
    if st.button("🔄 Refresh", type="secondary", help="Refresh the page"):
        st.rerun()

st.sidebar.markdown("---")

st.sidebar.header("🎨 Global Chart Styling")
try:
    eta_cmap_name = st.sidebar.selectbox("Default η colormap", cmap_list, index=cmap_list.index('viridis'))
    sigma_cmap_name = st.sidebar.selectbox("Default |σ| colormap", cmap_list, index=cmap_list.index('hot'))
    hydro_cmap_name = st.sidebar.selectbox("Default Hydrostatic colormap", cmap_list, index=cmap_list.index('coolwarm'))
    vm_cmap_name = st.sidebar.selectbox("Default von Mises colormap", cmap_list, index=cmap_list.index('plasma'))
except:
    # Fallback defaults
    eta_cmap_name = 'viridis'
    sigma_cmap_name = 'hot'
    hydro_cmap_name = 'coolwarm'
    vm_cmap_name = 'plasma'

# =============================================
# SIDEBAR - Multi-Simulation Control Panel
# =============================================
st.sidebar.header("🚀 Multi-Simulation Manager")

# Operation mode
operation_mode = st.sidebar.radio(
    "Operation Mode",
    ["Run New Simulation", "Compare Saved Simulations", "Single Simulation View"],
    index=0
)

if operation_mode == "Run New Simulation":
    st.sidebar.header("🎛️ New Simulation Setup")
    
    defect_type = st.sidebar.selectbox("Defect Type", ["ISF", "ESF", "Twin"])
    
    # Physical eigenstrain values
    if defect_type == "ISF":
        default_eps = 0.707
        default_kappa = 0.6
        caption = "Intrinsic Stacking Fault"
    elif defect_type == "ESF":
        default_eps = 1.414
        default_kappa = 0.7
        caption = "Extrinsic Stacking Fault"
    else:  # Twin
        default_eps = 2.121
        default_kappa = 0.3
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
        theta = np.deg2rad(angle_map.get(orientation, 0))
    
    st.sidebar.info(f"Selected tilt: **{np.rad2deg(theta):.1f}°** from horizontal")
    
    # Visualization settings - Individual for this simulation
    st.sidebar.subheader("Simulation-Specific Colormaps")
    try:
        sim_eta_cmap_name = st.sidebar.selectbox("η colormap for this sim", cmap_list, 
                                               index=cmap_list.index(eta_cmap_name))
        sim_sigma_cmap_name = st.sidebar.selectbox("|σ| colormap for this sim", cmap_list, 
                                                 index=cmap_list.index(sigma_cmap_name))
        sim_hydro_cmap_name = st.sidebar.selectbox("Hydrostatic colormap for this sim", cmap_list, 
                                                 index=cmap_list.index(hydro_cmap_name))
        sim_vm_cmap_name = st.sidebar.selectbox("von Mises colormap for this sim", cmap_list, 
                                              index=cmap_list.index(vm_cmap_name))
    except:
        sim_eta_cmap_name = eta_cmap_name
        sim_sigma_cmap_name = sigma_cmap_name
        sim_hydro_cmap_name = hydro_cmap_name
        sim_vm_cmap_name = vm_cmap_name
    
    # Run button
    if st.sidebar.button("🚀 Run & Save Simulation", type="primary"):
        try:
            sim_params = {
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
            
            with st.spinner(f"Running {defect_type} simulation..."):
                start_time = time.time()
                
                # Run simulation
                history = run_simulation(sim_params)
                
                # Create metadata
                metadata = MetadataManager.create_metadata(
                    sim_params, 
                    history, 
                    run_time=time.time() - start_time,
                    grid_size=N,
                    dx=dx
                )
                
                # Save to database
                sim_id = SimulationDB.save_simulation(sim_params, history, metadata)
                
                st.success(f"""
                ✅ Simulation Complete!
                - **ID**: `{sim_id}`
                - **Frames**: {len(history)}
                - **Time**: {metadata['run_time']:.1f} seconds
                - **Saved to database**
                """)
                
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error running simulation: {str(e)}")
            print(f"Error running simulation: {str(e)}")
            print(traceback.format_exc())

elif operation_mode == "Compare Saved Simulations":
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
            ["Side-by-Side Heatmaps", "Overlay Line Profiles", "Statistical Summary"],
            index=0
        )
        
        # ENHANCED LINE PROFILE SETTINGS
        if comparison_type == "Overlay Line Profiles":
            st.sidebar.subheader("📈 Enhanced Line Profile Settings")
            
            # Profile direction selection
            profile_direction = st.sidebar.selectbox(
                "Profile Direction",
                ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal", "Custom Angle", "Multiple Profiles"],
                index=0,
                help="Select line profile direction. 'Multiple Profiles' shows all available profiles."
            )
            
            # Position control
            position_ratio = st.sidebar.slider(
                "Profile Position Ratio",
                0.0, 1.0, 0.5, 0.1,
                help="Position of profile line relative to center (0 = center, 1 = edge)"
            )
            
            # Custom angle settings
            if profile_direction == "Custom Angle":
                custom_angle = st.sidebar.slider(
                    "Custom Angle (degrees)",
                    -180.0, 180.0, 45.0, 5.0
                )
            else:
                custom_angle = 45.0
            
            # Multiple profile selection
            if profile_direction == "Multiple Profiles":
                available_profiles = ["Horizontal", "Vertical", "Diagonal", "Anti-Diagonal"]
                selected_profiles = st.sidebar.multiselect(
                    "Select Profiles to Display",
                    available_profiles,
                    default=available_profiles[:2]
                )
            else:
                selected_profiles = [profile_direction]
        
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
        
        # Run comparison
        if st.sidebar.button("🔬 Run Comparison", type="primary"):
            st.session_state.run_comparison = True
            st.session_state.comparison_config = {
                'sim_ids': selected_ids,
                'type': comparison_type,
                'stress_component': stress_component,
                'frame_selection': frame_selection,
                'frame_idx': frame_idx
            }
            
            # Add enhanced line profile config
            if comparison_type == "Overlay Line Profiles":
                st.session_state.comparison_config.update({
                    'profile_direction': profile_direction,
                    'selected_profiles': selected_profiles,
                    'position_ratio': position_ratio,
                    'custom_angle': custom_angle if profile_direction == "Custom Angle" else None
                })

else:  # Single Simulation View
    st.sidebar.header("🔍 Single Simulation View")
    
    simulations = SimulationDB.get_simulation_list()
    
    if not simulations:
        st.sidebar.warning("No simulations saved yet. Run some simulations first!")
    else:
        sim_options = {f"{sim['name']} (ID: {sim['id']})": sim['id'] for sim in simulations}
        selected_sim = st.sidebar.selectbox(
            "Select Simulation to View",
            options=list(sim_options.keys())
        )
        
        if selected_sim:
            sim_id = sim_options[selected_sim]
            st.session_state.selected_sim_id = sim_id

# =============================================
# MAIN CONTENT AREA
# =============================================
if operation_mode == "Run New Simulation":
    # Show saved simulations
    st.header("📋 Saved Simulations")
    simulations = SimulationDB.get_simulation_list()
    
    if simulations:
        # Create a dataframe of saved simulations
        sim_data = []
        for sim in simulations:
            sim_full = SimulationDB.get_simulation(sim['id'])
            if sim_full:
                params = sim_full.get('params', {})
                metadata = sim_full.get('metadata', {})
                sim_data.append({
                    'ID': sim['id'],
                    'Defect Type': params.get('defect_type', 'Unknown'),
                    'Orientation': params.get('orientation', 'Unknown'),
                    'ε*': params.get('eps0', 0.0),
                    'κ': params.get('kappa', 0.0),
                    'Shape': params.get('shape', 'Unknown'),
                    'Steps': params.get('steps', 0),
                    'Frames': len(sim_full.get('history', [])),
                    'Created': sim_full.get('created_at', 'Unknown')[:19] if 'created_at' in sim_full else 'Unknown'
                })
        
        if sim_data:
            df = pd.DataFrame(sim_data)
            st.dataframe(df, use_container_width=True)
            
            # Delete option
            with st.expander("🗑️ Delete Simulations"):
                delete_options = [f"{sim['name']} (ID: {sim['id']})" for sim in simulations]
                to_delete = st.multiselect("Select simulations to delete", delete_options)
                
                if st.button("Delete Selected", type="secondary"):
                    for sim_name in to_delete:
                        # Extract ID from string
                        try:
                            sim_id = sim_name.split("ID: ")[1].replace(")", "")
                            if SimulationDB.delete_simulation(sim_id):
                                st.success(f"Deleted simulation {sim_id}")
                        except:
                            st.error(f"Could not delete simulation: {sim_name}")
                    st.rerun()
    else:
        st.info("No simulations saved yet. Run a simulation to see it here!")

elif operation_mode == "Compare Saved Simulations":
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
                frames = [len(sim.get('history', [])) - 1 for sim in simulations]
            elif config['frame_selection'] == "Same Evolution Time":
                # Use same evolution time (percentage of total steps)
                target_percentage = 0.8  # 80% of evolution
                frames = [int(len(sim.get('history', [])) * target_percentage) for sim in simulations]
            else:
                # Specific frame index
                frames = [min(frame_idx, len(sim.get('history', [])) - 1) for sim in simulations]
            
            # Get stress component mapping
            stress_map = {
                "Stress Magnitude |σ|": 'sigma_mag',
                "Hydrostatic σ_h": 'sigma_hydro',
                "von Mises σ_vM": 'von_mises'
            }
            stress_key = stress_map.get(config['stress_component'], 'sigma_mag')
            
            # Create comparison based on type
            if config['type'] == "Side-by-Side Heatmaps":
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
                
                for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row, col]
                    
                    # Get data
                    history = sim.get('history', [])
                    if history and frame < len(history):
                        eta, stress_fields = history[frame]
                        stress_data = stress_fields.get(stress_key)
                        
                        if stress_data is not None:
                            # Choose colormap
                            params = sim.get('params', {})
                            cmap_name = params.get('sigma_cmap', 'viridis')
                            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
                            
                            # Create heatmap
                            im = ax.imshow(stress_data, extent=extent, cmap=cmap, 
                                          origin='lower', aspect='equal')
                            
                            # Add contour lines for defect boundary
                            ax.contour(X, Y, eta, levels=[0.5], colors='white', 
                                      linewidths=1, linestyles='--', alpha=0.8)
                            
                            # Title
                            title = f"{params.get('defect_type', 'Unknown')}"
                            if params.get('orientation') != "Horizontal {111} (0°)":
                                title += f"\n{params.get('orientation', '').split(' ')[0]}"
                            
                            ax.set_title(title, fontsize=10, fontweight='semibold')
                            ax.set_xlabel("x (nm)", fontsize=9)
                            ax.set_ylabel("y (nm)", fontsize=9)
                            
                            # Colorbar
                            plt.colorbar(im, ax=ax, shrink=0.8).set_label(f"{config['stress_component']} (GPa)")
                    
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                        ax.set_title("No data")
                
                # Hide empty subplots
                for idx in range(n_sims, rows*cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            elif config['type'] == "Overlay Line Profiles":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Colors for different simulations
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                # Panel 1: Line profiles
                for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    history = sim.get('history', [])
                    if history and frame < len(history):
                        eta, stress_fields = history[frame]
                        stress_data = stress_fields.get(stress_key)
                        
                        if stress_data is not None:
                            # Extract profile based on configuration
                            if config.get('profile_direction') == "Multiple Profiles":
                                # Show first selected profile
                                profile_type = config.get('selected_profiles', ['Horizontal'])[0].lower().replace(" ", "_").replace("-", "")
                            else:
                                profile_type = config.get('profile_direction', 'Horizontal').lower().replace(" ", "_").replace("-", "")
                            
                            try:
                                if profile_type == "custom_angle":
                                    distance, profile, _ = EnhancedLineProfiler.extract_profile(
                                        stress_data, 'custom', config.get('position_ratio', 0.5), config.get('custom_angle', 45)
                                    )
                                else:
                                    distance, profile, _ = EnhancedLineProfiler.extract_profile(
                                        stress_data, profile_type, config.get('position_ratio', 0.5)
                                    )
                                
                                ax1.plot(distance, profile, color=color, linewidth=2,
                                        label=f"{sim.get('params', {}).get('defect_type', 'Unknown')} - {sim.get('params', {}).get('orientation', 'Unknown')}")
                            except:
                                pass
                
                ax1.set_xlabel("Position (nm)", fontsize=12)
                ax1.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=12)
                ax1.set_title(f"Line Profile Comparison", fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                # Panel 2: Statistics
                stats_data = []
                for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    history = sim.get('history', [])
                    if history and frame < len(history):
                        eta, stress_fields = history[frame]
                        stress_data = stress_fields.get(stress_key)
                        
                        if stress_data is not None:
                            flat_data = stress_data.flatten()
                            stats_data.append({
                                'Defect': sim.get('params', {}).get('defect_type', 'Unknown'),
                                'Max': float(np.nanmax(flat_data)),
                                'Mean': float(np.nanmean(flat_data)),
                                'Std': float(np.nanstd(flat_data))
                            })
                
                if stats_data:
                    defect_names = [stats['Defect'] for stats in stats_data]
                    max_stresses = [stats['Max'] for stats in stats_data]
                    
                    x_pos = np.arange(len(defect_names))
                    bars = ax2.bar(x_pos, max_stresses, color=colors[:len(stats_data)], alpha=0.7)
                    
                    ax2.set_xticks(x_pos)
                    ax2.set_xticklabels(defect_names, rotation=45, ha='right')
                    ax2.set_ylabel("Maximum Stress (GPa)", fontsize=12)
                    ax2.set_title("Peak Stress Comparison", fontsize=14, fontweight='bold')
                    
                    # Add value labels
                    for bar, val in zip(bars, max_stresses):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            elif config['type'] == "Statistical Summary":
                # Create statistical summary
                all_stats = []
                
                for sim, frame in zip(simulations, frames):
                    history = sim.get('history', [])
                    if history and frame < len(history):
                        eta, stress_fields = history[frame]
                        stress_data = stress_fields.get(stress_key)
                        
                        if stress_data is not None:
                            flat_data = stress_data.flatten()
                            flat_data = flat_data[np.isfinite(flat_data)]
                            
                            all_stats.append({
                                'Simulation': f"{sim.get('params', {}).get('defect_type', 'Unknown')} - {sim.get('params', {}).get('orientation', 'Unknown')}",
                                'N': len(flat_data),
                                'Max (GPa)': float(np.nanmax(flat_data)),
                                'Min (GPa)': float(np.nanmin(flat_data)),
                                'Mean (GPa)': float(np.nanmean(flat_data)),
                                'Median (GPa)': float(np.nanmedian(flat_data)),
                                'Std Dev (GPa)': float(np.nanstd(flat_data)),
                                'Skewness': float(stats.skew(flat_data)) if len(flat_data) > 0 else 0.0,
                                'Kurtosis': float(stats.kurtosis(flat_data)) if len(flat_data) > 0 else 0.0
                            })
                
                if all_stats:
                    df_stats = pd.DataFrame(all_stats)
                    st.dataframe(df_stats.style.format({
                        'Max (GPa)': '{:.3f}',
                        'Min (GPa)': '{:.3f}',
                        'Mean (GPa)': '{:.3f}',
                        'Median (GPa)': '{:.3f}',
                        'Std Dev (GPa)': '{:.3f}',
                        'Skewness': '{:.3f}',
                        'Kurtosis': '{:.3f}'
                    }), use_container_width=True)
                    
                    # Create box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    data_to_plot = []
                    labels = []
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                    
                    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                        history = sim.get('history', [])
                        if history and frame < len(history):
                            eta, stress_fields = history[frame]
                            stress_data = stress_fields.get(stress_key)
                            
                            if stress_data is not None:
                                flat_data = stress_data.flatten()
                                flat_data = flat_data[np.isfinite(flat_data)]
                                
                                data_to_plot.append(flat_data)
                                labels.append(f"{sim.get('params', {}).get('defect_type', 'Unknown')}")
                    
                    if data_to_plot:
                        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                                       showmeans=True, meanline=True, showfliers=False)
                        
                        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                        
                        ax.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=12)
                        ax.set_title("Statistical Distribution Comparison", fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
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
                defect = sim.get('params', {}).get('defect_type', 'Unknown')
                if defect not in defect_groups:
                    defect_groups[defect] = []
                defect_groups[defect].append(sim)
            
            for defect_type, sims in defect_groups.items():
                with st.expander(f"{defect_type} ({len(sims)} simulations)"):
                    for sim in sims:
                        params = sim.get('params', {})
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.text(f"ID: {sim.get('id', 'Unknown')}")
                        with col2:
                            st.text(f"Orientation: {params.get('orientation', 'Unknown')}")
                        with col3:
                            st.text(f"ε*={params.get('eps0', 0.0):.2f}, κ={params.get('kappa', 0.0):.2f}")
        else:
            st.warning("No simulations available. Run some simulations first!")

else:  # Single Simulation View
    st.header("📊 Single Simulation Viewer")
    
    if 'selected_sim_id' in st.session_state:
        sim_data = SimulationDB.get_simulation(st.session_state.selected_sim_id)
        
        if sim_data:
            history = sim_data.get("history", [])
            sim_params = sim_data.get('params', {})
            
            # Display simulation info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Defect Type", sim_params.get('defect_type', 'Unknown'))
            with col2:
                st.metric("Shape", sim_params.get('shape', 'Unknown'))
            with col3:
                st.metric("ε*", f"{sim_params.get('eps0', 0.0):.3f}")
            with col4:
                st.metric("κ", f"{sim_params.get('kappa', 0.0):.3f}")
            
            # Colormap selection
            st.subheader("🎨 Colormap Configuration")
            
            # Initialize colormap session state
            if 'colormaps' not in st.session_state:
                st.session_state.colormaps = {
                    'eta': sim_params.get('eta_cmap', 'magma'),
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
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('eta', 'magma')) if st.session_state.colormaps.get('eta') in cmap_list else 0,
                    key="eta_cm"
                )
                st.session_state.colormaps['eta'] = new_eta_cm
                
            with col2:
                new_sxx_cm = st.selectbox(
                    "σxx",
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('sxx', 'coolwarm')) if st.session_state.colormaps.get('sxx') in cmap_list else 0,
                    key="sxx_cm"
                )
                st.session_state.colormaps['sxx'] = new_sxx_cm
                
            with col3:
                new_syy_cm = st.selectbox(
                    "σyy",
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('syy', 'coolwarm')) if st.session_state.colormaps.get('syy') in cmap_list else 0,
                    key="syy_cm"
                )
                st.session_state.colormaps['syy'] = new_syy_cm
                
            with col4:
                new_sxy_cm = st.selectbox(
                    "σxy",
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('sxy', 'coolwarm')) if st.session_state.colormaps.get('sxy') in cmap_list else 0,
                    key="sxy_cm"
                )
                st.session_state.colormaps['sxy'] = new_sxy_cm
                
            with col5:
                new_hydro_cm = st.selectbox(
                    "Hydrostatic",
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('sigma_hydro', 'viridis')) if st.session_state.colormaps.get('sigma_hydro') in cmap_list else 0,
                    key="hydro_cm"
                )
                st.session_state.colormaps['sigma_hydro'] = new_hydro_cm
                
            with col6:
                new_vm_cm = st.selectbox(
                    "Von Mises",
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('von_mises', 'plasma')) if st.session_state.colormaps.get('von_mises') in cmap_list else 0,
                    key="vm_cm"
                )
                st.session_state.colormaps['von_mises'] = new_vm_cm
                
            with col7:
                new_mag_cm = st.selectbox(
                    "Magnitude",
                    cmap_list,
                    index=cmap_list.index(st.session_state.colormaps.get('sigma_mag', 'inferno')) if st.session_state.colormaps.get('sigma_mag') in cmap_list else 0,
                    key="mag_cm"
                )
                st.session_state.colormaps['sigma_mag'] = new_mag_cm
            
            if history:
                # Frame selection with dynamic update
                num_frames = len(history)
                frame_idx = st.slider("Frame", 0, num_frames - 1, num_frames - 1, 
                                     key=f"frame_slider_{st.session_state.selected_sim_id}")
                
                eta, stress = history[frame_idx]
                
                # -----------------------------
                # η FIELD HEATMAP
                # -----------------------------
                st.subheader("🟣 Phase Field η")
                
                fig_eta, ax_eta = plt.subplots(figsize=(5, 5))
                im = ax_eta.imshow(eta, extent=extent, 
                                  cmap=st.session_state.colormaps.get('eta', 'viridis'), 
                                  origin="lower", aspect='equal')
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
                    ("σxx", stress.get("sxx"), st.session_state.colormaps.get('sxx', 'coolwarm')),
                    ("σyy", stress.get("syy"), st.session_state.colormaps.get('syy', 'coolwarm')),
                    ("σxy", stress.get("sxy"), st.session_state.colormaps.get('sxy', 'coolwarm')),
                    ("Hydrostatic (σ_h)", stress.get("sigma_hydro"), st.session_state.colormaps.get('sigma_hydro', 'viridis')),
                    ("Von Mises", stress.get("von_mises"), st.session_state.colormaps.get('von_mises', 'plasma')),
                    ("Magnitude", stress.get("sigma_mag"), st.session_state.colormaps.get('sigma_mag', 'inferno')),
                ]
                
                for ax, (title, field, cmap) in zip(axs.flatten(), fields):
                    if field is not None:
                        im = ax.imshow(field, extent=extent, cmap=cmap, origin="lower", aspect='equal')
                        ax.set_title(title)
                        plt.colorbar(im, ax=ax, shrink=0.7)
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                        ax.set_title(title)
                
                st.pyplot(fig_s)
                plt.close(fig_s)
                
                # -----------------------------
                # ANIMATION CONTROLS
                # -----------------------------
                st.subheader("🎬 Animation Controls")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⏮️ First Frame"):
                        st.session_state[f"frame_slider_{st.session_state.selected_sim_id}"] = 0
                        st.rerun()
                
                with col2:
                    play = st.checkbox("▶️ Play Animation", key=f"play_{st.session_state.selected_sim_id}")
                    if play:
                        # Auto-advance frames
                        current_frame = st.session_state.get(f"current_frame_{st.session_state.selected_sim_id}", 0)
                        if current_frame < num_frames - 1:
                            st.session_state[f"frame_slider_{st.session_state.selected_sim_id}"] = current_frame + 1
                            st.session_state[f"current_frame_{st.session_state.selected_sim_id}"] = current_frame + 1
                        else:
                            st.session_state[f"frame_slider_{st.session_state.selected_sim_id}"] = 0
                            st.session_state[f"current_frame_{st.session_state.selected_sim_id}"] = 0
                        st.rerun()
                    else:
                        st.session_state[f"current_frame_{st.session_state.selected_sim_id}"] = 0
                
                with col3:
                    if st.button("⏭️ Last Frame"):
                        st.session_state[f"frame_slider_{st.session_state.selected_sim_id}"] = num_frames - 1
                        st.rerun()
                
                # Display frame information
                st.info(f"**Frame {frame_idx + 1} of {num_frames}** | Simulation Time: {frame_idx * sim_params.get('save_every', 20)} steps")
            
            # Delete current simulation option
            st.markdown("---")
            if st.button("🗑️ Delete This Simulation", key=f"delete_{st.session_state.selected_sim_id}"):
                SimulationDB.delete_simulation(st.session_state.selected_sim_id)
                if 'selected_sim_id' in st.session_state:
                    del st.session_state.selected_sim_id
                st.success(f"Simulation deleted!")
                st.rerun()
                
        else:
            st.error("Simulation not found!")
    else:
        st.info("Select a simulation from the sidebar to view")

# =============================================
# DOWNLOAD SECTION WITH ERROR HANDLING
# =============================================
st.header("🔥 Download Files")
simulations_dict = SimulationDB.get_all_simulations()

if simulations_dict:
    # Create a selection dropdown
    sim_options = []
    for sim_id, sim_data in simulations_dict.items():
        try:
            params = sim_data.get('params', {})
            name = f"{params.get('defect_type', 'Unknown')} - {params.get('orientation', 'Unknown')} (ID: {sim_id})"
            sim_options.append((name, sim_id))
        except:
            sim_options.append((f"Simulation {sim_id}", sim_id))
    
    if sim_options:
        selected_sim_name = st.selectbox("Select Simulation to Download", 
                                         [opt[0] for opt in sim_options])
        
        # Find the selected simulation
        selected_sim_id = None
        for name, sim_id in sim_options:
            if name == selected_sim_name:
                selected_sim_id = sim_id
                break
        
        if selected_sim_id:
            sim_data = simulations_dict.get(selected_sim_id)
            if sim_data:
                try:
                    history = sim_data.get('history', [])
                    sim_params = sim_data.get('params', {})
                    metadata = sim_data.get('metadata', {})
                    
                    # Validate metadata
                    metadata = MetadataManager.validate_metadata(metadata)
                    
                    # Get created_at from sim_data or metadata
                    created_at = sim_data.get('created_at', metadata.get('created_at', datetime.now().isoformat()))
                    
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
                    
                    st.subheader(f"Download options for: {sim_name}")
                    
                    # ----------------------------
                    # PKL (pickle)
                    # ----------------------------
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        try:
                            pkl_buffer = BytesIO()
                            pickle.dump(data, pkl_buffer)
                            pkl_buffer.seek(0)
                            st.download_button("📦 Download PKL", pkl_buffer, 
                                             file_name=f"{sim_name}.pkl",
                                             mime="application/octet-stream")
                        except Exception as e:
                            st.error(f"PKL export failed: {str(e)}")
                    
                    # ----------------------------
                    # PT (torch) - safe conversion
                    # ----------------------------
                    with col2:
                        try:
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
                            st.download_button("⚡ Download PT", pt_buffer, 
                                             file_name=f"{sim_name}.pt",
                                             mime="application/octet-stream")
                        except Exception as e:
                            st.error(f"PT export failed: {str(e)}")
                    
                    # ----------------------------
                    # DB (in-memory SQL dump) - FIXED VERSION
                    # ----------------------------
                    with col3:
                        try:
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
                            
                            # Use the created_at we extracted earlier
                            c.execute("INSERT INTO simulations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                      (selected_sim_id, sim_name, 
                                       sim_params.get('defect_type', 'Unknown'), 
                                       sim_params.get('shape', 'Unknown'),
                                       sim_params.get('orientation', 'Unknown'), 
                                       sim_params.get('theta', 0.0),
                                       sim_params.get('eps0', 0.0), 
                                       sim_params.get('kappa', 0.0), 
                                       sim_params.get('steps', 0),
                                       created_at,  # Use the extracted created_at
                                       metadata.get('grid_size', 128),
                                       metadata.get('dx', 0.1)))
                            
                            for idx, frame in enumerate(data['history']):
                                c.execute("INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                          (selected_sim_id, idx,
                                           pickle.dumps(frame['eta']),
                                           pickle.dumps(frame['stresses'].get('sxx', np.zeros((N, N)))),
                                           pickle.dumps(frame['stresses'].get('syy', np.zeros((N, N)))),
                                           pickle.dumps(frame['stresses'].get('sxy', np.zeros((N, N)))),
                                           pickle.dumps(frame['stresses'].get('szz', np.zeros((N, N)))),
                                           pickle.dumps(frame['stresses'].get('sigma_mag', np.zeros((N, N)))),
                                           pickle.dumps(frame['stresses'].get('sigma_hydro', np.zeros((N, N)))),
                                           pickle.dumps(frame['stresses'].get('von_mises', np.zeros((N, N))))))
                            conn.commit()
                            for line in conn.iterdump():
                                db_dump_buffer.write('%s\n' % line)
                            db_dump_str = db_dump_buffer.getvalue()
                            st.download_button("🗃️ Download DB (SQL Dump)", db_dump_str, 
                                             file_name=f"{sim_name}.sql",
                                             mime="application/sql")
                        except Exception as e:
                            st.error(f"DB export failed: {str(e)}")
                            print(f"DB export error: {str(e)}")
                            print(traceback.format_exc())
                    
                    # ----------------------------
                    # CSV (zip multiple frames)
                    # ----------------------------
                    with col4:
                        try:
                            csv_zip_buffer = BytesIO()
                            with zipfile.ZipFile(csv_zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                for idx, (eta, stress_fields) in enumerate(history):
                                    df = pd.DataFrame({
                                        'x': X.flatten(),
                                        'y': Y.flatten(),
                                        'eta': eta.flatten(),
                                        'sxx': stress_fields.get('sxx', np.zeros((N, N))).flatten(),
                                        'syy': stress_fields.get('syy', np.zeros((N, N))).flatten(),
                                        'sxy': stress_fields.get('sxy', np.zeros((N, N))).flatten(),
                                        'szz': stress_fields.get('szz', np.zeros((N, N))).flatten(),
                                        'sigma_mag': stress_fields.get('sigma_mag', np.zeros((N, N))).flatten(),
                                        'sigma_hydro': stress_fields.get('sigma_hydro', np.zeros((N, N))).flatten(),
                                        'von_mises': stress_fields.get('von_mises', np.zeros((N, N))).flatten()
                                    })
                                    csv_str = df.to_csv(index=False)
                                    zf.writestr(f"{sim_name}_frame_{idx}.csv", csv_str)
                            csv_zip_buffer.seek(0)
                            st.download_button("📊 Download CSV (Zip)", csv_zip_buffer, 
                                             file_name=f"{sim_name}_csv.zip",
                                             mime="application/zip")
                        except Exception as e:
                            st.error(f"CSV export failed: {str(e)}")
                    
                    # ----------------------------
                    # JSON Export
                    # ----------------------------
                    st.subheader("📋 Additional Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        try:
                            # JSON Parameters
                            params_json = json.dumps(sim_params, indent=2)
                            st.download_button("📄 Download Parameters (JSON)", params_json,
                                             file_name=f"{sim_name}_params.json",
                                             mime="application/json")
                        except Exception as e:
                            st.error(f"JSON params export failed: {str(e)}")
                    
                    with col2:
                        try:
                            # Complete JSON Export
                            def convert_to_serializable(obj):
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, np.generic):
                                    return obj.item()
                                elif isinstance(obj, dict):
                                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_to_serializable(item) for item in obj]
                                else:
                                    return obj
                            
                            serializable_data = convert_to_serializable(data)
                            complete_json = json.dumps(serializable_data, indent=2)
                            st.download_button("📑 Complete JSON Export", complete_json,
                                             file_name=f"{sim_name}_complete.json",
                                             mime="application/json")
                        except Exception as e:
                            st.error(f"Complete JSON export failed: {str(e)}")
                    
                    # ----------------------------
                    # Bulk Export All Simulations
                    # ----------------------------
                    st.subheader("📦 Bulk Export All Simulations")
                    
                    if st.button("🚀 Export All Simulations as ZIP"):
                        with st.spinner("Creating bulk export package..."):
                            try:
                                bulk_buffer = BytesIO()
                                with zipfile.ZipFile(bulk_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                                    # Export each simulation
                                    for sim_id_local, sim_data_local in simulations_dict.items():
                                        try:
                                            sim_params_local = sim_data_local.get('params', {})
                                            history_local = sim_data_local.get('history', [])
                                            metadata_local = sim_data_local.get('metadata', {})
                                            
                                            sim_name_local = build_sim_name(sim_params_local, sim_id=sim_id_local)
                                            sim_dir = f"simulation_{sim_id_local}"
                                            
                                            # Export parameters
                                            params_json = json.dumps(sim_params_local, indent=2)
                                            zf.writestr(f"{sim_dir}/parameters.json", params_json)
                                            
                                            # Export metadata
                                            metadata_local = MetadataManager.validate_metadata(metadata_local)
                                            metadata_json = json.dumps(metadata_local, indent=2)
                                            zf.writestr(f"{sim_dir}/metadata.json", metadata_json)
                                            
                                            # Export data frames
                                            for idx, (eta_local, stress_fields_local) in enumerate(history_local):
                                                df = pd.DataFrame({
                                                    'x': X.flatten(),
                                                    'y': Y.flatten(),
                                                    'eta': eta_local.flatten(),
                                                    'sxx': stress_fields_local.get('sxx', np.zeros((N, N))).flatten(),
                                                    'syy': stress_fields_local.get('syy', np.zeros((N, N))).flatten(),
                                                    'sxy': stress_fields_local.get('sxy', np.zeros((N, N))).flatten(),
                                                    'szz': stress_fields_local.get('szz', np.zeros((N, N))).flatten(),
                                                    'sigma_mag': stress_fields_local.get('sigma_mag', np.zeros((N, N))).flatten(),
                                                    'sigma_hydro': stress_fields_local.get('sigma_hydro', np.zeros((N, N))).flatten(),
                                                    'von_mises': stress_fields_local.get('von_mises', np.zeros((N, N))).flatten()
                                                })
                                                csv_str = df.to_csv(index=False)
                                                zf.writestr(f"{sim_dir}/frame_{idx:04d}.csv", csv_str)
                                        except Exception as e:
                                            st.warning(f"Could not export simulation {sim_id_local}: {str(e)}")
                                    
                                    # Create summary file
                                    summary = f"""MULTI-SIMULATION EXPORT SUMMARY
========================================
Generated: {datetime.now().isoformat()}
Total Simulations: {len(simulations_dict)}
Export Format: Complete Package (JSON + CSV)

SIMULATIONS:
------------
"""
                                    for sim_id_local, sim_data_local in simulations_dict.items():
                                        params_local = sim_data_local.get('params', {})
                                        summary += f"\nSimulation {sim_id_local}:"
                                        summary += f"\n  Defect: {params_local.get('defect_type', 'Unknown')}"
                                        summary += f"\n  Orientation: {params_local.get('orientation', 'Unknown')}"
                                        summary += f"\n  ε*: {params_local.get('eps0', 0.0)}"
                                        summary += f"\n  κ: {params_local.get('kappa', 0.0)}"
                                        summary += f"\n  Frames: {len(sim_data_local.get('history', []))}"
                                        summary += f"\n  Created: {sim_data_local.get('created_at', 'Unknown')}\n"
                                    
                                    zf.writestr("EXPORT_SUMMARY.txt", summary)
                                
                                bulk_buffer.seek(0)
                                
                                # Determine file name
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                filename = f"ag_np_all_simulations_{timestamp}.zip"
                                
                                st.download_button(
                                    "📥 Download All Simulations",
                                    bulk_buffer.getvalue(),
                                    filename,
                                    "application/zip"
                                )
                                st.success("Bulk export package ready!")
                            except Exception as e:
                                st.error(f"Bulk export failed: {str(e)}")
                                print(f"Bulk export error: {str(e)}")
                                print(traceback.format_exc())
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
                    print(f"Download preparation error: {str(e)}")
                    print(traceback.format_exc())
    else:
        st.warning("No valid simulations to download")
else:
    st.info("No simulations available for download.")

# =============================================
# DEBUG PANEL (Hidden by default)
# =============================================
with st.expander("🐛 Debug Information", expanded=False):
    st.subheader("Session State Contents")
    if 'simulations' in st.session_state:
        st.json({k: {
            'id': v.get('id'),
            'params_keys': list(v.get('params', {}).keys()),
            'history_len': len(v.get('history', [])),
            'metadata_keys': list(v.get('metadata', {}).keys()),
            'created_at': v.get('created_at')
        } for k, v in st.session_state.simulations.items()})
    else:
        st.write("No simulations in session state")
    
    st.subheader("Memory Usage")
    import sys
    total_size = 0
    if 'simulations' in st.session_state:
        for sim_id, sim_data in st.session_state.simulations.items():
            # Estimate size
            history_size = 0
            for eta, stress in sim_data.get('history', []):
                history_size += eta.nbytes
                for key, value in stress.items():
                    if hasattr(value, 'nbytes'):
                        history_size += value.nbytes
            total_size += history_size
    
    st.write(f"Estimated total data size: {total_size / (1024*1024):.2f} MB")
    st.write(f"Number of simulations: {len(st.session_state.get('simulations', {}))}")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Soundness & Advanced Analysis", expanded=False):
    st.markdown("""
    ### 🎯 **Enhanced Multi-Simulation Comparison Platform**
    
    #### **🔧 Key Fixes Implemented:**
    
    **1. Metadata Consistency:**
    - **Centralized Metadata Management**: All metadata now handled through `MetadataManager`
    - **Automatic Validation**: Missing fields are automatically added with defaults
    - **Error Handling**: Graceful degradation when metadata is incomplete
    
    **2. Error Handling:**
    - **Comprehensive Error Decorators**: All critical functions wrapped in error handlers
    - **User-Friendly Messages**: Clear error messages without exposing sensitive data
    - **Logging**: Detailed error logs for debugging
    
    **3. Data Integrity:**
    - **Default Values**: All dictionary accesses use `.get()` with defaults
    - **Type Checking**: Safe type conversions and validations
    - **Fallback Mechanisms**: Alternative calculations when primary methods fail
    
    #### **📊 Platform Features:**
    
    **1. Multi-Simulation Management:**
    - **Run New Simulations**: Configure and run individual simulations
    - **Compare Saved Simulations**: Compare multiple simulations side-by-side
    - **Single Simulation View**: Detailed view of individual simulations
    - **Persistent Storage**: Simulations saved in session state
    
    **2. Enhanced Visualization:**
    - **50+ Colormaps**: Wide variety of scientific colormaps
    - **Dynamic Visualization**: Real-time updates with simulation changes
    - **Multiple Plot Types**: Heatmaps, line profiles, statistical summaries
    
    **3. Advanced Analysis:**
    - **Line Profile Analysis**: Horizontal, vertical, diagonal, and custom angle profiles
    - **Statistical Comparison**: Comprehensive statistics across simulations
    - **Stress Component Analysis**: All stress tensor components
    
    **4. Robust Export Capabilities:**
    - **Multiple Formats**: PKL, PT, DB, CSV, JSON with error handling
    - **Individual Downloads**: Export single simulations
    - **Bulk Export**: Download all simulations as ZIP
    - **Complete Metadata**: All parameters and results included
    
    #### **🔬 Scientific Workflow:**
    
    1. **Run Simulations**: Configure defect type, orientation, and parameters
    2. **Save Results**: Automatically stored in the database
    3. **Compare Results**: Side-by-side comparison of multiple simulations
    4. **Analyze Data**: Line profiles, statistics, and correlation analysis
    5. **Export Results**: Multiple formats for further analysis or publication
    
    #### **🚀 Platform Benefits:**
    
    **For Researchers:**
    - **Rapid Prototyping**: Quick simulation setup and execution
    - **Comparative Analysis**: Easy comparison of different conditions
    - **Data Management**: Organized storage of simulation results
    - **Publication Support**: Ready-to-use figures and data
    
    **For Educators:**
    - **Interactive Learning**: Visual demonstration of defect physics
    - **Parameter Exploration**: Easy variation of physical parameters
    - **Visual Feedback**: Immediate visualization of results
    
    **Advanced crystallographic defect analysis platform with comprehensive error handling and robust data management!**
    """)
    
    # Display platform statistics
    simulations = SimulationDB.get_all_simulations()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulations", len(simulations))
    with col2:
        total_frames = sum([len(sim.get('history', [])) for sim in simulations.values()]) if simulations else 0
        st.metric("Total Frames", f"{total_frames:,}")
    with col3:
        st.metric("Colormaps", f"{len(COLORMAPS)}+")
    with col4:
        st.metric("Export Formats", "5+")

st.caption("🔬 Advanced Multi-Defect Analysis Platform • Robust Error Handling • 50+ Colormaps • 2025")
