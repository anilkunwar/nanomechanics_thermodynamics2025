import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse, Polygon, Arc
from matplotlib.collections import LineCollection, PatchCollection
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, map_coordinates, rotate
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# =============================================
# ENHANCED PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Ag NP Multi-Defect Analyzer Pro",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with gradient backgrounds and animations
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, 
            #667eea 0%, 
            #764ba2 25%, 
            #f093fb 50%, 
            #f5576c 75%, 
            #ffd166 100%
        ) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        padding: 2rem !important;
        margin-bottom: 1.5rem !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        letter-spacing: -0.5px !important;
    }
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #2D3748 !important;
        border-left: 5px solid #667eea !important;
        padding-left: 1.5rem !important;
        margin: 2rem 0 1.2rem 0 !important;
        background: linear-gradient(90deg, #F7FAFC 0%, #FFFFFF 100%) !important;
        padding: 1.2rem 1.5rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.08) !important;
    }
    .feature-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        margin: 1.5rem 0 !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .feature-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12) !important;
        border-color: #667eea !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2.5rem !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.4) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.5) !important;
    }
    .stButton>button:after {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 5px !important;
        height: 5px !important;
        background: rgba(255, 255, 255, 0.5) !important;
        opacity: 0 !important;
        border-radius: 100% !important;
        transform: scale(1, 1) translate(-50%) !important;
        transform-origin: 50% 50% !important;
    }
    .stButton>button:focus:after {
        animation: ripple 1s ease-out !important;
    }
    @keyframes ripple {
        0% { transform: scale(0, 0); opacity: 0.5; }
        100% { transform: scale(30, 30); opacity: 0; }
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important;
        padding: 1.8rem !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06) !important;
        border-left: 6px solid #667eea !important;
        margin: 1rem 0 !important;
        transition: transform 0.3s ease !important;
    }
    .metric-card:hover {
        transform: translateY(-3px) !important;
    }
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #1E3A8A 0%, #3730A3 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.5rem !important;
    }
    .metric-label {
        font-size: 1rem !important;
        color: #4A5568 !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px !important;
    }
    /* Enhanced tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem !important;
        background-color: #F7FAFC !important;
        padding: 0.8rem !important;
        border-radius: 15px !important;
        border: 1px solid #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px !important;
        white-space: pre-wrap !important;
        background-color: #EDF2F7 !important;
        border-radius: 12px !important;
        gap: 1rem !important;
        padding: 0 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #4A5568 !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E2E8F0 !important;
        color: #2D3748 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
        border-color: #667eea !important;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# ENHANCED HEADER
# =============================================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">🔬 Ag Nanoparticle Multi-Orientation Stress Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; padding: 2rem; background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); border-radius: 20px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);">
        <h3 style="color: #2D3748; font-weight: 700; margin-bottom: 1rem; font-size: 1.4rem;">🎯 Publication-Ready Stress Visualization • Multi-Orientation Profiling • Enhanced Post-Processing</h3>
        <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 1.5rem;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; color: #667eea; margin-bottom: 0.5rem;">📐</div>
                <div style="font-weight: 600; color: #4A5568;">Hydrostatic Stress<br><span style="color: #2D3748; font-size: 0.9rem;">(Diagonal Probes)</span></div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; color: #764ba2; margin-bottom: 0.5rem;">📈</div>
                <div style="font-weight: 600; color: #4A5568;">von Mises Stress<br><span style="color: #2D3748; font-size: 0.9rem;">(Vertical/Horizontal Probes)</span></div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; color: #f5576c; margin-bottom: 0.5rem;">🎨</div>
                <div style="font-weight: 600; color: #4A5568;">Stress Magnitude<br><span style="color: #2D3748; font-size: 0.9rem;">(Multi-Orientation Analysis)</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# MATERIAL & GRID PARAMETERS (Enhanced)
# =============================================
a = 0.4086  # FCC Ag lattice constant (nm)
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)

# Enhanced elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0  # ± 2 GPa
C12 = 93.4   # ± 2 GPa
C44 = 46.1   # ± 1 GPa

# Grid parameters
N = 192  # Increased for better visualization
dx = 0.08  # Finer grid spacing (nm)
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# Create radial and angular coordinate systems
R = np.sqrt(X**2 + Y**2)
THETA = np.arctan2(Y, X)

# =============================================
# ENHANCED COLORMAP LIBRARY WITH SCIENCE-OPTIMIZED PALETTES
# =============================================
COLORMAPS = {
    # Publication-optimized sequential
    'nature_sequential': mpl.colors.LinearSegmentedColormap.from_list(
        'nature_sequential', 
        ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    ),
    'science_diverging': mpl.colors.LinearSegmentedColormap.from_list(
        'science_diverging',
        ['#1a237e', '#3949ab', '#5c6bc0', '#9fa8da', '#e8eaf6', '#fce4ec', '#f48fb1', '#ec407a', '#d81b60', '#880e4f']
    ),
    'adv_mat_stress': mpl.colors.LinearSegmentedColormap.from_list(
        'adv_mat_stress',
        ['#00008b', '#1e90ff', '#00bfff', '#00ffff', '#ffff00', '#ffa500', '#ff4500', '#8b0000']
    ),
    
    # Perceptually uniform sequential
    'viridis': 'viridis',
    'plasma': 'plasma', 
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    
    # Diverging - Enhanced for stress visualization
    'coolwarm': 'coolwarm',
    'bwr': 'bwr',
    'seismic': 'seismic',
    'RdBu': 'RdBu',
    'RdGy': 'RdGy',
    
    # Cyclic
    'twilight': 'twilight',
    'twilight_shifted': 'twilight_shifted',
    
    # Qualitative
    'tab10': 'tab10',
    'tab20': 'tab20',
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    
    # Modern scientific
    'rocket': 'rocket',
    'mako': 'mako',
    'crest': 'crest',
    'icefire': 'icefire',
    
    # Colorblind friendly
    'colorblind': 'colorblind',
    'vik': 'vik',
    'roma': 'roma'
}

# Create specialized colormaps for different stress components
hydrostatic_cmaps = ['coolwarm', 'RdBu', 'seismic', 'bwr', 'science_diverging']
vonmises_cmaps = ['plasma', 'inferno', 'magma', 'rocket', 'adv_mat_stress']
magnitude_cmaps = ['viridis', 'mako', 'crest', 'nature_sequential']

cmap_list = list(COLORMAPS.keys())

# =============================================
# ADVANCED PROFILE EXTRACTION SYSTEM
# =============================================
class PublicationProfileExtractor:
    """Advanced system for extracting line profiles optimized for publication-quality visualization"""
    
    @staticmethod
    def extract_profile_2d(data, angle_deg, position='center', offset=0, 
                          length_ratio=0.7, sampling_factor=4, interpolation_order=3):
        """
        Extract high-resolution line profile with publication-quality accuracy
        
        Parameters:
        -----------
        data : 2D numpy array
            Input data
        angle_deg : float
            Angle in degrees (0° = horizontal, 90° = vertical, 45° = diagonal)
        position : str or tuple
            'center', 'offset', or (x, y) coordinates
        offset : float
            Offset from center in nm (perpendicular to profile)
        length_ratio : float
            Length as fraction of domain size (0-1)
        sampling_factor : int
            Oversampling factor for smooth curves
        interpolation_order : int
            Interpolation order (1=linear, 3=cubic spline)
        
        Returns:
        --------
        distances : 1D array
            Distance along profile (nm)
        profile : 1D array
            Extracted values
        endpoints : tuple
            (x_start, y_start, x_end, y_end) in physical coordinates
        metadata : dict
            Comprehensive extraction parameters and statistics
        """
        N = data.shape[0]
        angle_rad = np.deg2rad(angle_deg)
        
        # Determine center point with enhanced precision
        if position == 'center':
            x_center, y_center = 0.0, 0.0
        elif position == 'offset':
            perp_angle = angle_rad + np.pi/2
            x_center = offset * np.cos(perp_angle)
            y_center = offset * np.sin(perp_angle)
        elif isinstance(position, (tuple, list)) and len(position) == 2:
            x_center, y_center = position
        else:
            x_center, y_center = 0.0, 0.0
        
        # Calculate profile length with domain consideration
        domain_size = extent[1] - extent[0]
        half_length = domain_size * length_ratio / 2
        
        # Calculate endpoints with precision
        x_start = x_center - half_length * np.cos(angle_rad)
        y_start = y_center - half_length * np.sin(angle_rad)
        x_end = x_center + half_length * np.cos(angle_rad)
        y_end = y_center + half_length * np.sin(angle_rad)
        
        # Generate high-resolution sampling points
        num_points = int(N * length_ratio * sampling_factor)
        distances = np.linspace(-half_length, half_length, num_points)
        xs = x_center + distances * np.cos(angle_rad)
        ys = y_center + distances * np.sin(angle_rad)
        
        # Convert to array indices with boundary checking
        xi = (xs - extent[0]) / (extent[1] - extent[0]) * (N - 1)
        yi = (ys - extent[2]) / (extent[3] - extent[2]) * (N - 1)
        
        # Clip indices to valid range
        xi = np.clip(xi, 0, N - 1)
        yi = np.clip(yi, 0, N - 1)
        
        # Extract profile with enhanced interpolation
        profile = map_coordinates(data, [yi, xi], order=interpolation_order, 
                                 mode='nearest', cval=np.nan)
        
        # Remove NaN values
        valid_mask = np.isfinite(profile)
        if np.any(valid_mask):
            profile = profile[valid_mask]
            distances = distances[valid_mask]
        else:
            profile = np.zeros_like(distances)
        
        # Calculate comprehensive statistics
        if len(profile) > 0:
            profile_norm = profile - np.nanmin(profile)
            max_val = np.nanmax(profile_norm)
            
            # Calculate FWHM with enhanced accuracy
            fwhm = 0.0
            if max_val > 0:
                half_max = max_val / 2
                above_half = profile_norm >= half_max
                if np.sum(above_half) >= 2:
                    indices = np.where(above_half)[0]
                    left_idx = indices[0]
                    right_idx = indices[-1]
                    fwhm = float(distances[right_idx] - distances[left_idx])
            
            # Calculate peak characteristics
            peak_idx = np.nanargmax(profile)
            peak_pos = distances[peak_idx] if len(distances) > 0 else 0
            peak_value = profile[peak_idx] if len(profile) > 0 else 0
            
            # Calculate gradient characteristics
            if len(profile) > 1:
                gradient = np.gradient(profile, distances)
                max_gradient = np.nanmax(np.abs(gradient))
            else:
                max_gradient = 0.0
            
            # Calculate symmetry metrics
            symmetry = PublicationProfileExtractor._calculate_profile_symmetry(distances, profile)
            
            metadata = {
                'angle_deg': angle_deg,
                'position': position,
                'offset_nm': offset,
                'length_nm': 2 * half_length,
                'num_points': len(profile),
                'sampling_factor': sampling_factor,
                'interpolation_order': interpolation_order,
                'max_value': float(np.nanmax(profile)),
                'min_value': float(np.nanmin(profile)),
                'mean_value': float(np.nanmean(profile)),
                'std_value': float(np.nanstd(profile)),
                'peak_position_nm': float(peak_pos),
                'peak_value': float(peak_value),
                'fwhm_nm': fwhm,
                'max_gradient': float(max_gradient),
                'skewness': float(stats.skew(profile)),
                'kurtosis': float(stats.kurtosis(profile)),
                'symmetry_index': symmetry,
                'integrated_value': float(np.trapz(np.abs(profile), distances))
            }
        else:
            metadata = {
                'angle_deg': angle_deg,
                'position': position,
                'offset_nm': offset,
                'length_nm': 2 * half_length,
                'num_points': 0,
                'sampling_factor': sampling_factor,
                'interpolation_order': interpolation_order,
                'max_value': 0.0,
                'min_value': 0.0,
                'mean_value': 0.0,
                'std_value': 0.0,
                'peak_position_nm': 0.0,
                'peak_value': 0.0,
                'fwhm_nm': 0.0,
                'max_gradient': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'symmetry_index': 0.0,
                'integrated_value': 0.0
            }
        
        return distances, profile, (x_start, y_start, x_end, y_end), metadata
    
    @staticmethod
    def _calculate_profile_symmetry(distances, profile):
        """Calculate symmetry index of profile about center"""
        if len(profile) < 3:
            return 0.0
        
        center_idx = len(profile) // 2
        left_half = profile[:center_idx]
        right_half = profile[center_idx + 1:][::-1]  # Reverse for comparison
        
        min_len = min(len(left_half), len(right_half))
        if min_len == 0:
            return 0.0
        
        left_half = left_half[:min_len]
        right_half = right_half[:min_len]
        
        # Normalize for comparison
        left_norm = (left_half - np.mean(left_half)) / (np.std(left_half) + 1e-10)
        right_norm = (right_half - np.mean(right_half)) / (np.std(right_half) + 1e-10)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(left_norm, right_norm)[0, 1] if len(left_norm) > 1 else 0
        
        # Calculate RMS difference
        rms_diff = np.sqrt(np.mean((left_norm - right_norm) ** 2))
        
        # Combined symmetry index (0=perfect symmetry, 1=perfect asymmetry)
        symmetry_index = max(0, 1 - (abs(correlation) + (1 - rms_diff/np.sqrt(2))) / 2)
        
        return symmetry_index
    
    @staticmethod
    def extract_stress_component_profiles(stress_fields, component_name, angles, offsets=None, **kwargs):
        """
        Extract profiles for specific stress component with optimal angle selection
        
        Parameters:
        -----------
        stress_fields : dict
            Dictionary of stress components
        component_name : str
            'sigma_mag', 'sigma_hydro', or 'von_mises'
        angles : list
            List of angles to extract (optimized per component)
        offsets : list or None
            Optional offsets for each angle
        
        Returns:
        --------
        profiles : dict
            Dictionary of profiles keyed by angle
        """
        stress_map = {
            'sigma_mag': 'Stress Magnitude |σ|',
            'sigma_hydro': 'Hydrostatic Stress σ_h',
            'von_mises': 'von Mises Stress σ_vM'
        }
        
        if component_name not in stress_fields:
            raise ValueError(f"Stress component '{component_name}' not found")
        
        data = stress_fields[component_name]
        
        if offsets is None:
            offsets = [0] * len(angles)
        
        profiles = {}
        for angle, offset in zip(angles, offsets):
            key = f"{angle}°"
            distances, profile, endpoints, metadata = PublicationProfileExtractor.extract_profile_2d(
                data, angle, 'offset' if offset != 0 else 'center', offset, **kwargs
            )
            
            profiles[key] = {
                'distances': distances,
                'profile': profile,
                'endpoints': endpoints,
                'metadata': metadata,
                'component': component_name,
                'display_name': stress_map.get(component_name, component_name)
            }
        
        return profiles
    
    @staticmethod
    def create_multi_component_comparison(simulations, frames, component_angles_map):
        """
        Create comprehensive multi-component, multi-angle profile comparison
        
        Parameters:
        -----------
        simulations : list
            List of simulation data dictionaries
        frames : list
            Frame indices for each simulation
        component_angles_map : dict
            Mapping of component names to optimal angles
            Example: {'sigma_hydro': [45, 135], 'von_mises': [0, 90]}
        
        Returns:
        --------
        comparison_data : dict
            Nested dictionary of profiles for each simulation and component
        """
        comparison_data = {}
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            sim_id = sim.get('id', f'sim_{sim_idx}')
            eta, stress_fields = sim['history'][frame]
            
            comparison_data[sim_id] = {
                'params': sim['params'],
                'profiles': {}
            }
            
            for component, angles in component_angles_map.items():
                if component in stress_fields:
                    profiles = PublicationProfileExtractor.extract_stress_component_profiles(
                        stress_fields, component, angles,
                        length_ratio=0.7, sampling_factor=4
                    )
                    comparison_data[sim_id]['profiles'][component] = profiles
        
        return comparison_data

# =============================================
# PUBLICATION-QUALITY VISUALIZATION SYSTEM
# =============================================
class PublicationVisualizer:
    """Advanced visualization system for publication-quality stress analysis"""
    
    @staticmethod
    def create_enhanced_styling_params():
        """Get enhanced styling parameters for publication-quality figures"""
        return {
            'figure': {
                'dpi': 600,
                'facecolor': 'white',
                'edgecolor': 'black',
                'linewidth': 1.0
            },
            'axes': {
                'linewidth': 1.5,
                'labelsize': 14,
                'titlesize': 16,
                'titleweight': 'bold',
                'labelweight': 'semibold',
                'grid': True,
                'grid_alpha': 0.2,
                'grid_style': '--',
                'grid_linewidth': 0.5
            },
            'ticks': {
                'direction': 'in',
                'length': 6,
                'width': 1.5,
                'labelsize': 12,
                'major_size': 8,
                'minor_size': 4,
                'major_width': 1.5,
                'minor_width': 1.0
            },
            'font': {
                'family': 'DejaVu Sans',
                'size': 12,
                'weight': 'normal'
            },
            'legend': {
                'fontsize': 11,
                'frameon': True,
                'fancybox': True,
                'framealpha': 0.9,
                'edgecolor': 'black',
                'facecolor': 'white'
            },
            'lines': {
                'linewidth': 2.5,
                'markersize': 8,
                'markeredgewidth': 1.5,
                'alpha': 0.8
            },
            'colorbar': {
                'fontsize': 12,
                'extend': 'both',
                'shrink': 0.8,
                'aspect': 30,
                'pad': 0.05
            }
        }
    
    @staticmethod
    def apply_publication_style(fig, axes, style_params=None):
        """Apply publication-quality styling to figure and axes"""
        if style_params is None:
            style_params = PublicationVisualizer.create_enhanced_styling_params()
        
        # Set rcParams for consistent styling
        rcParams.update({
            'font.family': style_params['font']['family'],
            'font.size': style_params['font']['size'],
            'font.weight': style_params['font']['weight'],
            'axes.linewidth': style_params['axes']['linewidth'],
            'axes.labelsize': style_params['axes']['labelsize'],
            'axes.titlesize': style_params['axes']['titlesize'],
            'axes.titleweight': style_params['axes']['titleweight'],
            'axes.labelweight': style_params['axes']['labelweight'],
            'axes.grid': style_params['axes']['grid'],
            'grid.alpha': style_params['axes']['grid_alpha'],
            'grid.linestyle': style_params['axes']['grid_style'],
            'grid.linewidth': style_params['axes']['grid_linewidth'],
            'xtick.direction': style_params['ticks']['direction'],
            'ytick.direction': style_params['ticks']['direction'],
            'xtick.labelsize': style_params['ticks']['labelsize'],
            'ytick.labelsize': style_params['ticks']['labelsize'],
            'xtick.major.size': style_params['ticks']['major_size'],
            'ytick.major.size': style_params['ticks']['major_size'],
            'xtick.major.width': style_params['ticks']['major_width'],
            'ytick.major.width': style_params['ticks']['major_width'],
            'xtick.minor.size': style_params['ticks']['minor_size'],
            'ytick.minor.size': style_params['ticks']['minor_size'],
            'xtick.minor.width': style_params['ticks']['minor_width'],
            'ytick.minor.width': style_params['ticks']['minor_width'],
            'lines.linewidth': style_params['lines']['linewidth'],
            'lines.markersize': style_params['lines']['markersize'],
            'lines.markeredgewidth': style_params['lines']['markeredgewidth'],
            'legend.fontsize': style_params['legend']['fontsize'],
            'legend.frameon': style_params['legend']['frameon'],
            'legend.fancybox': style_params['legend']['fancybox'],
            'legend.framealpha': style_params['legend']['framealpha'],
            'legend.edgecolor': style_params['legend']['edgecolor'],
            'legend.facecolor': style_params['legend']['facecolor'],
            'savefig.dpi': style_params['figure']['dpi'],
            'savefig.facecolor': style_params['figure']['facecolor'],
            'savefig.edgecolor': style_params['figure']['edgecolor'],
            'savefig.bbox': 'tight'
        })
        
        # Flatten axes array
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        # Apply enhanced styling to each axis
        for ax in axes_flat:
            if ax is not None:
                # Add minor ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                # Set tick parameters
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=style_params['ticks']['length'], 
                              width=style_params['ticks']['width'])
                ax.tick_params(which='minor', length=style_params['ticks']['length']/2, 
                              width=style_params['ticks']['width']/2)
                
                # Enhance spine visibility
                for spine in ax.spines.values():
                    spine.set_linewidth(style_params['axes']['linewidth'])
                    spine.set_color('black')
        
        # Set figure facecolor
        fig.set_facecolor(style_params['figure']['facecolor'])
        
        # Apply tight layout with padding
        fig.set_constrained_layout(True)
        fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05)
        
        return fig
    
    @staticmethod
    def create_comprehensive_overlay_plot(simulations, frames, config, style_params=None):
        """
        Create comprehensive overlay plot with multiple stress components and orientations
        
        Parameters:
        -----------
        simulations : list
            List of simulation data dictionaries
        frames : list
            Frame indices for each simulation
        config : dict
            Configuration dictionary with plot settings
        style_params : dict or None
            Styling parameters (uses defaults if None)
        
        Returns:
        --------
        fig : matplotlib Figure
            Publication-quality figure
        """
        if style_params is None:
            style_params = PublicationVisualizer.create_enhanced_styling_params()
        
        # Get stress component
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map[config.get('stress_component', 'Stress Magnitude |σ|')]
        
        # Get orientations
        orientations = config.get('orientations', [0, 45, 90, 135])
        
        # Create figure with enhanced layout
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid specification for complex layout
        gs = fig.add_gridspec(4, 4, height_ratios=[1.2, 1, 1, 1.5], 
                             hspace=0.3, wspace=0.3)
        
        # Define subplots
        ax_main = fig.add_subplot(gs[0, :])  # Main overlay plot
        ax_horizontal = fig.add_subplot(gs[1, 0])  # Horizontal profiles
        ax_vertical = fig.add_subplot(gs[1, 1])   # Vertical profiles
        ax_diagonal1 = fig.add_subplot(gs[1, 2])  # 45° diagonal
        ax_diagonal2 = fig.add_subplot(gs[1, 3])  # 135° diagonal
        ax_stats = fig.add_subplot(gs[2, 0:2])    # Statistical summary
        ax_comparison = fig.add_subplot(gs[2, 2:]) # Component comparison
        ax_domain = fig.add_subplot(gs[3, :])     # Domain with probe lines
        
        # Get colors for simulations
        colors = plt.cm.Set2(np.linspace(0, 1, len(simulations)))
        
        # Extract and plot profiles
        all_profiles = {}
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            sim_id = sim.get('id', f'sim_{sim_idx}')
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            all_profiles[sim_id] = {}
            
            for angle in orientations:
                # Extract profile
                distances, profile, endpoints, metadata = PublicationProfileExtractor.extract_profile_2d(
                    stress_data, angle, 'center', 0, 0.7, 4, 3
                )
                
                all_profiles[sim_id][angle] = {
                    'distances': distances,
                    'profile': profile,
                    'metadata': metadata
                }
                
                # Plot in main overlay
                line_style = config.get('line_style', 'solid')
                line_width = style_params['lines']['linewidth']
                
                label = f"{sim['params']['defect_type']} - {angle}°"
                ax_main.plot(distances, profile, 
                           color=colors[sim_idx],
                           linestyle=line_style,
                           linewidth=line_width,
                           alpha=0.85,
                           label=label if sim_idx == 0 else None)
        
        # Configure main overlay panel
        ax_main.set_xlabel("Distance from Center (nm)", 
                          fontsize=style_params['axes']['labelsize'] + 2,
                          fontweight='bold')
        ax_main.set_ylabel(f"{config['stress_component']} (GPa)", 
                          fontsize=style_params['axes']['labelsize'] + 2,
                          fontweight='bold')
        ax_main.set_title("Multi-Orientation Stress Profile Overlay", 
                         fontsize=style_params['axes']['titlesize'] + 2,
                         fontweight='bold',
                         pad=20)
        ax_main.legend(fontsize=style_params['legend']['fontsize'], 
                      ncol=2, loc='upper right',
                      framealpha=0.95)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Create orientation-specific panels
        orientation_panels = [
            (ax_horizontal, 0, "Horizontal (0°)"),
            (ax_vertical, 90, "Vertical (90°)"),
            (ax_diagonal1, 45, "Diagonal (45°)"),
            (ax_diagonal2, 135, "Anti-Diagonal (135°)")
        ]
        
        for ax, target_angle, title in orientation_panels:
            if target_angle in orientations:
                for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    distances, profile, endpoints, metadata = PublicationProfileExtractor.extract_profile_2d(
                        stress_data, target_angle, 'center', 0, 0.7
                    )
                    
                    line_style = config.get('line_style', 'solid')
                    label = f"{sim['params']['defect_type']}" if sim_idx == 0 else None
                    
                    ax.plot(distances, profile, 
                           color=colors[sim_idx],
                           linestyle=line_style,
                           linewidth=line_width,
                           alpha=0.8,
                           label=label)
            
            ax.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
            ax.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
            ax.set_title(title, fontsize=style_params['axes']['titlesize'], 
                        fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            if target_angle == 0:
                ax.legend(fontsize=style_params['legend']['fontsize'] - 1)
        
        # Statistical summary panel
        if all_profiles:
            stats_data = []
            
            for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
                for angle, data in profiles.items():
                    if angle in orientations:
                        metadata = data['metadata']
                        stats_data.append({
                            'Simulation': simulations[sim_idx]['params']['defect_type'],
                            'Angle': f'{angle}°',
                            'Max (GPa)': metadata['max_value'],
                            'Mean (GPa)': metadata['mean_value'],
                            'FWHM (nm)': metadata['fwhm_nm'],
                            'Peak Pos (nm)': metadata['peak_position_nm'],
                            'Symmetry': f"{metadata['symmetry_index']:.3f}"
                        })
            
            # Create bar plot for maximum stresses
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                unique_sims = df_stats['Simulation'].unique()
                unique_angles = df_stats['Angle'].unique()
                
                x = np.arange(len(unique_sims))
                width = 0.8 / len(unique_angles)
                
                for i, angle in enumerate(unique_angles):
                    angle_data = df_stats[df_stats['Angle'] == angle]
                    max_values = []
                    for sim in unique_sims:
                        sim_data = angle_data[angle_data['Simulation'] == sim]
                        max_values.append(sim_data['Max (GPa)'].values[0] if not sim_data.empty else 0)
                    
                    bars = ax_stats.bar(x + i*width - width*(len(unique_angles)-1)/2, 
                                       max_values, width,
                                       label=angle, alpha=0.8)
                
                ax_stats.set_xlabel("Simulation", fontsize=style_params['axes']['labelsize'])
                ax_stats.set_ylabel("Maximum Stress (GPa)", fontsize=style_params['axes']['labelsize'])
                ax_stats.set_title("Peak Stress by Orientation", 
                                 fontsize=style_params['axes']['titlesize'],
                                 fontweight='bold', pad=10)
                ax_stats.set_xticks(x)
                ax_stats.set_xticklabels(unique_sims, rotation=45, ha='right')
                ax_stats.legend(fontsize=style_params['legend']['fontsize'] - 1)
                ax_stats.grid(True, alpha=0.3, axis='y')
        
        # Stress component comparison panel
        if simulations and len(simulations) > 0:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            
            # Extract profiles for different stress components
            components = ['sigma_mag', 'sigma_hydro', 'von_mises']
            component_names = ['|σ|', 'σ_h', 'σ_vM']
            
            # Use optimal angles for each component
            optimal_angles = {'sigma_mag': [0, 90], 'sigma_hydro': [45, 135], 'von_mises': [0, 90]}
            
            for comp_idx, component in enumerate(components):
                if component in stress_fields:
                    # Use first optimal angle for comparison
                    angle = optimal_angles[component][0] if component in optimal_angles else 0
                    distances, profile, endpoints, metadata = PublicationProfileExtractor.extract_profile_2d(
                        stress_fields[component], angle, 'center', 0, 0.7
                    )
                    
                    ax_comparison.plot(distances, profile, 
                                     linewidth=line_width,
                                     label=component_names[comp_idx])
            
            ax_comparison.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
            ax_comparison.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
            ax_comparison.set_title("Stress Component Comparison", 
                                  fontsize=style_params['axes']['titlesize'],
                                  fontweight='bold', pad=10)
            ax_comparison.legend(fontsize=style_params['legend']['fontsize'])
            ax_comparison.grid(True, alpha=0.3, linestyle='--')
            ax_comparison.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Domain with probe lines panel
        if simulations:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            
            # Set fixed aspect ratio
            ax_domain.set_aspect('equal')
            
            # Plot domain with enhanced colormap
            cmap_name = sim['params'].get('eta_cmap', 'viridis')
            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
            
            # Enhanced visualization with contour overlay
            im = ax_domain.imshow(eta, extent=extent, cmap=cmap, 
                                origin='lower', alpha=0.85)
            
            # Add contour lines for defect boundary
            contour = ax_domain.contour(X, Y, eta, levels=[0.5], 
                                      colors='white', linewidths=2, 
                                      linestyles='--', alpha=0.8)
            
            # Add probe lines with enhanced styling
            line_colors = ['red', 'blue', 'green', 'purple', 'orange']
            line_styles = ['-', '--', '-.', ':']
            
            for idx, angle in enumerate(orientations):
                # Extract line for visualization
                distances, profile, endpoints, metadata = PublicationProfileExtractor.extract_profile_2d(
                    eta, angle, 'center', 0, 0.6
                )
                x_start, y_start, x_end, y_end = endpoints
                
                # Draw probe line
                line = ax_domain.plot([x_start, x_end], [y_start, y_end], 
                                    color=line_colors[idx % len(line_colors)],
                                    linewidth=3,
                                    linestyle=line_styles[idx % len(line_styles)],
                                    alpha=0.9,
                                    label=f'{angle}°',
                                    solid_capstyle='round')[0]
                
                # Add angle annotation with enhanced styling
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                ax_domain.annotate(f'{angle}°', xy=(mid_x, mid_y),
                                 xytext=(15, 15), textcoords='offset points',
                                 color=line_colors[idx % len(line_colors)],
                                 fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.4", 
                                          facecolor='white', alpha=0.9,
                                          edgecolor=line_colors[idx % len(line_colors)],
                                          linewidth=2),
                                 arrowprops=dict(arrowstyle='->', 
                                               color=line_colors[idx % len(line_colors)],
                                               linewidth=2, alpha=0.7))
            
            ax_domain.set_xlabel("x (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_ylabel("y (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_title("Simulation Domain with Probe Lines", 
                              fontsize=style_params['axes']['titlesize'] + 1,
                              fontweight='bold', pad=15)
            ax_domain.legend(fontsize=style_params['legend']['fontsize'], 
                           loc='upper right', framealpha=0.95)
            
            # Add scale bar
            PublicationVisualizer._add_publication_scale_bar(ax_domain, 5.0, 
                                                           location='lower right',
                                                           color='white')
            
            # Add colorbar with enhanced styling
            cbar = plt.colorbar(im, ax=ax_domain, 
                              shrink=style_params['colorbar']['shrink'],
                              pad=style_params['colorbar']['pad'])
            cbar.set_label('Defect Parameter η', 
                          fontsize=style_params['colorbar']['fontsize'],
                          fontweight='bold')
            cbar.ax.tick_params(labelsize=style_params['ticks']['labelsize'] - 1)
        
        # Add panel labels (A, B, C, ...)
        panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        panel_axes = [ax_main, ax_horizontal, ax_vertical, ax_diagonal1, 
                     ax_diagonal2, ax_stats, ax_comparison, ax_domain]
        
        for ax, label in zip(panel_axes, panel_labels):
            if ax is not None:
                ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                       fontsize=20, fontweight='bold', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', alpha=0.9,
                                edgecolor='black', linewidth=1.5))
        
        # Apply publication styling
        fig = PublicationVisualizer.apply_publication_style(fig, panel_axes, style_params)
        
        return fig
    
    @staticmethod
    def _add_publication_scale_bar(ax, length_nm, location='lower right', color='white'):
        """Add publication-quality scale bar"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        if location == 'lower right':
            bar_x_start = xlim[1] - x_range * 0.15
            bar_x_end = bar_x_start - length_nm
            bar_y = ylim[0] + y_range * 0.05
            text_y = bar_y + y_range * 0.025
            text_ha = 'center'
            text_va = 'bottom'
        elif location == 'lower left':
            bar_x_start = xlim[0] + x_range * 0.05
            bar_x_end = bar_x_start + length_nm
            bar_y = ylim[0] + y_range * 0.05
            text_y = bar_y + y_range * 0.025
            text_ha = 'center'
            text_va = 'bottom'
        else:
            return
        
        # Draw enhanced scale bar
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
               color=color, linewidth=4, solid_capstyle='butt',
               zorder=1000)
        
        # Add enhanced end caps
        cap_length = y_range * 0.02
        ax.plot([bar_x_start, bar_x_start], 
               [bar_y - cap_length, bar_y + cap_length],
               color=color, linewidth=4, solid_capstyle='butt',
               zorder=1000)
        ax.plot([bar_x_end, bar_x_end], 
               [bar_y - cap_length, bar_y + cap_length],
               color=color, linewidth=4, solid_capstyle='butt',
               zorder=1000)
        
        # Add text with enhanced background
        text = ax.text((bar_x_start + bar_x_end) / 2, text_y,
                      f'{length_nm} nm', ha=text_ha, va=text_va,
                      color=color, fontsize=12, fontweight='bold',
                      zorder=1001)
        
        # Enhanced text background
        text.set_bbox(dict(boxstyle="round,pad=0.5", 
                          facecolor='black', alpha=0.7,
                          edgecolor='white', linewidth=2))
    
    @staticmethod
    def create_stress_component_specific_plot(simulations, frames, config, style_params=None):
        """
        Create stress component-specific visualization with optimal probe orientations
        
        Parameters:
        -----------
        simulations : list
            List of simulation data dictionaries
        frames : list
            Frame indices for each simulation
        config : dict
            Configuration with stress component and orientation settings
        style_params : dict or None
            Styling parameters
        
        Returns:
        --------
        fig : matplotlib Figure
            Publication-quality figure optimized for specific stress component
        """
        if style_params is None:
            style_params = PublicationVisualizer.create_enhanced_styling_params()
        
        # Get stress component
        stress_component = config.get('stress_component', 'Stress Magnitude |σ|')
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map[stress_component]
        
        # Define optimal probe orientations for each stress component
        optimal_orientations = {
            'sigma_mag': [0, 45, 90, 135],  # All orientations for magnitude
            'sigma_hydro': [45, 135],       # Diagonals for hydrostatic stress
            'von_mises': [0, 90]            # Horizontal/vertical for von Mises
        }
        
        # Get optimal orientations for this component
        orientations = config.get('orientations', 
                                optimal_orientations.get(stress_key, [0, 45, 90, 135]))
        
        # Create figure optimized for specific component
        if stress_key == 'sigma_hydro':
            # Specialized layout for hydrostatic stress (diagonals emphasized)
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1.5], 
                                 hspace=0.25, wspace=0.25)
            
            ax_main = fig.add_subplot(gs[0, :])  # Main diagonal comparison
            ax_diag1 = fig.add_subplot(gs[1, 0])  # 45° diagonal
            ax_diag2 = fig.add_subplot(gs[1, 1])  # 135° diagonal
            ax_symmetry = fig.add_subplot(gs[1, 2])  # Symmetry analysis
            ax_domain = fig.add_subplot(gs[2, :])  # Domain with diagonals
            
            # Emphasize diagonal orientations
            diagonal_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
        elif stress_key == 'von_mises':
            # Specialized layout for von Mises stress (horizontal/vertical emphasized)
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1.5], 
                                 hspace=0.25, wspace=0.25)
            
            ax_main = fig.add_subplot(gs[0, :])  # Main comparison
            ax_horizontal = fig.add_subplot(gs[1, 0])  # Horizontal
            ax_vertical = fig.add_subplot(gs[1, 1])    # Vertical
            ax_anisotropy = fig.add_subplot(gs[1, 2])  # Anisotropy analysis
            ax_domain = fig.add_subplot(gs[2, :])      # Domain
            
        else:  # Stress magnitude
            # General layout for magnitude
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, height_ratios=[1.2, 1, 1, 1.5], 
                                 hspace=0.3, wspace=0.3)
            
            ax_main = fig.add_subplot(gs[0, :])  # Main overlay
            ax_hv = fig.add_subplot(gs[1, 0:2])  # Horizontal/vertical comparison
            ax_diag = fig.add_subplot(gs[1, 2:])  # Diagonal comparison
            ax_stats = fig.add_subplot(gs[2, 0:2])  # Statistics
            ax_polar = fig.add_subplot(gs[2, 2:], projection='polar')  # Polar distribution
            ax_domain = fig.add_subplot(gs[3, :])  # Domain with all probes
        
        # Get colors for simulations
        colors = plt.cm.tab10(np.linspace(0, 1, len(simulations)))
        
        # Extract and analyze profiles
        analyzer = PublicationProfileExtractor()
        all_profiles = {}
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            sim_id = sim.get('id', f'sim_{sim_idx}')
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            all_profiles[sim_id] = {}
            
            for angle in orientations:
                distances, profile, endpoints, metadata = analyzer.extract_profile_2d(
                    stress_data, angle, 'center', 0, 0.7, 4, 3
                )
                
                all_profiles[sim_id][angle] = {
                    'distances': distances,
                    'profile': profile,
                    'metadata': metadata,
                    'endpoints': endpoints
                }
        
        # Create component-specific visualizations
        if stress_key == 'sigma_hydro':
            # Hydrostatic stress visualization
            PublicationVisualizer._create_hydrostatic_visualization(
                fig, ax_main, ax_diag1, ax_diag2, ax_symmetry, ax_domain,
                simulations, frames, stress_key, all_profiles, colors, config, style_params
            )
        elif stress_key == 'von_mises':
            # von Mises stress visualization
            PublicationVisualizer._create_vonmises_visualization(
                fig, ax_main, ax_horizontal, ax_vertical, ax_anisotropy, ax_domain,
                simulations, frames, stress_key, all_profiles, colors, config, style_params
            )
        else:
            # Stress magnitude visualization
            PublicationVisualizer._create_magnitude_visualization(
                fig, ax_main, ax_hv, ax_diag, ax_stats, ax_polar, ax_domain,
                simulations, frames, stress_key, all_profiles, colors, config, style_params
            )
        
        # Apply publication styling
        all_axes = [ax for ax in [ax_main, ax_domain] + 
                   ([ax_diag1, ax_diag2, ax_symmetry] if stress_key == 'sigma_hydro' else
                    [ax_horizontal, ax_vertical, ax_anisotropy] if stress_key == 'von_mises' else
                    [ax_hv, ax_diag, ax_stats, ax_polar]) if ax is not None]
        
        fig = PublicationVisualizer.apply_publication_style(fig, all_axes, style_params)
        
        return fig
    
    @staticmethod
    def _create_hydrostatic_visualization(fig, ax_main, ax_diag1, ax_diag2, ax_symmetry, ax_domain,
                                         simulations, frames, stress_key, all_profiles, colors, config, style_params):
        """Create specialized visualization for hydrostatic stress"""
        
        line_width = style_params['lines']['linewidth']
        
        # Main plot: Compare 45° and 135° diagonals
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            sim_id = sim.get('id', f'sim_{sim_idx}')
            
            # 45° diagonal
            if 45 in all_profiles[sim_id]:
                data45 = all_profiles[sim_id][45]
                ax_main.plot(data45['distances'], data45['profile'],
                           color=colors[sim_idx], linewidth=line_width,
                           linestyle='-', alpha=0.8,
                           label=f"{sim['params']['defect_type']} - 45°")
            
            # 135° diagonal
            if 135 in all_profiles[sim_id]:
                data135 = all_profiles[sim_id][135]
                ax_main.plot(data135['distances'], data135['profile'],
                           color=colors[sim_idx], linewidth=line_width,
                           linestyle='--', alpha=0.8,
                           label=f"{sim['params']['defect_type']} - 135°")
        
        ax_main.set_xlabel("Distance from Center (nm)", fontsize=style_params['axes']['labelsize'] + 1)
        ax_main.set_ylabel("Hydrostatic Stress σ_h (GPa)", fontsize=style_params['axes']['labelsize'] + 1)
        ax_main.set_title("Hydrostatic Stress: Diagonal Profile Comparison", 
                         fontsize=style_params['axes']['titlesize'] + 2,
                         fontweight='bold', pad=15)
        ax_main.legend(fontsize=style_params['legend']['fontsize'], ncol=2)
        ax_main.grid(True, alpha=0.3)
        ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Individual diagonal plots
        PublicationVisualizer._plot_individual_diagonal(ax_diag1, simulations, all_profiles, 45, colors, style_params)
        PublicationVisualizer._plot_individual_diagonal(ax_diag2, simulations, all_profiles, 135, colors, style_params)
        
        # Symmetry analysis
        if len(simulations) > 0:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            stress_data = stress_fields[stress_key]
            
            # Extract profiles for symmetry analysis
            analyzer = PublicationProfileExtractor()
            dist45, prof45, _, meta45 = analyzer.extract_profile_2d(stress_data, 45, 'center', 0, 0.7)
            dist135, prof135, _, meta135 = analyzer.extract_profile_2d(stress_data, 135, 'center', 0, 0.7)
            
            # Plot symmetry comparison
            ax_symmetry.plot(dist45, prof45, 'b-', linewidth=line_width, alpha=0.7, label='45°')
            ax_symmetry.plot(dist135, prof135, 'r--', linewidth=line_width, alpha=0.7, label='135°')
            ax_symmetry.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
            ax_symmetry.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
            ax_symmetry.set_title("Diagonal Symmetry Analysis", 
                                 fontsize=style_params['axes']['titlesize'],
                                 fontweight='bold')
            ax_symmetry.legend(fontsize=style_params['legend']['fontsize'])
            ax_symmetry.grid(True, alpha=0.3)
        
        # Domain plot with diagonals
        if simulations:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            
            ax_domain.set_aspect('equal')
            cmap = plt.cm.get_cmap(COLORMAPS.get('coolwarm', 'coolwarm'))
            im = ax_domain.imshow(stress_fields[stress_key], extent=extent, 
                                cmap=cmap, origin='lower', alpha=0.85)
            
            # Add diagonal lines
            for angle in [45, 135]:
                if angle in all_profiles[next(iter(all_profiles.keys()))]:
                    endpoints = all_profiles[next(iter(all_profiles.keys()))][angle]['endpoints']
                    x_start, y_start, x_end, y_end = endpoints
                    color = 'red' if angle == 45 else 'blue'
                    ax_domain.plot([x_start, x_end], [y_start, y_end],
                                 color=color, linewidth=3, alpha=0.9,
                                 label=f'{angle}°')
            
            ax_domain.set_xlabel("x (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_ylabel("y (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_title("Hydrostatic Stress Field with Diagonal Probes", 
                              fontsize=style_params['axes']['titlesize'] + 1,
                              fontweight='bold', pad=15)
            ax_domain.legend(fontsize=style_params['legend']['fontsize'])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_domain, shrink=0.8)
            cbar.set_label('Hydrostatic Stress σ_h (GPa)', 
                          fontsize=style_params['colorbar']['fontsize'])
    
    @staticmethod
    def _create_vonmises_visualization(fig, ax_main, ax_horizontal, ax_vertical, ax_anisotropy, ax_domain,
                                      simulations, frames, stress_key, all_profiles, colors, config, style_params):
        """Create specialized visualization for von Mises stress"""
        
        line_width = style_params['lines']['linewidth']
        
        # Main plot: Overlay of horizontal and vertical profiles
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            sim_id = sim.get('id', f'sim_{sim_idx}')
            
            # Horizontal profile
            if 0 in all_profiles[sim_id]:
                data0 = all_profiles[sim_id][0]
                ax_main.plot(data0['distances'], data0['profile'],
                           color=colors[sim_idx], linewidth=line_width,
                           linestyle='-', alpha=0.8,
                           label=f"{sim['params']['defect_type']} - Horizontal")
            
            # Vertical profile
            if 90 in all_profiles[sim_id]:
                data90 = all_profiles[sim_id][90]
                ax_main.plot(data90['distances'], data90['profile'],
                           color=colors[sim_idx], linewidth=line_width,
                           linestyle='--', alpha=0.8,
                           label=f"{sim['params']['defect_type']} - Vertical")
        
        ax_main.set_xlabel("Distance from Center (nm)", fontsize=style_params['axes']['labelsize'] + 1)
        ax_main.set_ylabel("von Mises Stress σ_vM (GPa)", fontsize=style_params['axes']['labelsize'] + 1)
        ax_main.set_title("von Mises Stress: Horizontal vs Vertical Profiles", 
                         fontsize=style_params['axes']['titlesize'] + 2,
                         fontweight='bold', pad=15)
        ax_main.legend(fontsize=style_params['legend']['fontsize'], ncol=2)
        ax_main.grid(True, alpha=0.3)
        ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Individual horizontal and vertical plots
        PublicationVisualizer._plot_individual_orientation(ax_horizontal, simulations, all_profiles, 0, colors, style_params, "Horizontal")
        PublicationVisualizer._plot_individual_orientation(ax_vertical, simulations, all_profiles, 90, colors, style_params, "Vertical")
        
        # Anisotropy analysis
        if len(simulations) > 0:
            # Calculate anisotropy ratio (vertical/horizontal)
            anisotropy_ratios = []
            sim_labels = []
            
            for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                sim_id = sim.get('id', f'sim_{sim_idx}')
                
                if 0 in all_profiles[sim_id] and 90 in all_profiles[sim_id]:
                    data0 = all_profiles[sim_id][0]
                    data90 = all_profiles[sim_id][90]
                    
                    # Calculate peak ratio
                    peak0 = data0['metadata']['max_value']
                    peak90 = data90['metadata']['max_value']
                    
                    if peak0 > 0:
                        anisotropy_ratio = peak90 / peak0
                        anisotropy_ratios.append(anisotropy_ratio)
                        sim_labels.append(sim['params']['defect_type'])
            
            if anisotropy_ratios:
                x_pos = np.arange(len(anisotropy_ratios))
                bars = ax_anisotropy.bar(x_pos, anisotropy_ratios, 
                                        color=colors[:len(anisotropy_ratios)], alpha=0.7)
                
                ax_anisotropy.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax_anisotropy.set_xlabel("Simulation", fontsize=style_params['axes']['labelsize'])
                ax_anisotropy.set_ylabel("Anisotropy Ratio (V/H)", fontsize=style_params['axes']['labelsize'])
                ax_anisotropy.set_title("Stress Anisotropy", 
                                       fontsize=style_params['axes']['titlesize'],
                                       fontweight='bold')
                ax_anisotropy.set_xticks(x_pos)
                ax_anisotropy.set_xticklabels(sim_labels, rotation=45, ha='right')
                ax_anisotropy.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, ratio in zip(bars, anisotropy_ratios):
                    ax_anisotropy.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                     f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Domain plot with horizontal/vertical lines
        if simulations:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            
            ax_domain.set_aspect('equal')
            cmap = plt.cm.get_cmap(COLORMAPS.get('plasma', 'plasma'))
            im = ax_domain.imshow(stress_fields[stress_key], extent=extent, 
                                cmap=cmap, origin='lower', alpha=0.85)
            
            # Add horizontal and vertical lines
            for angle in [0, 90]:
                if angle in all_profiles[next(iter(all_profiles.keys()))]:
                    endpoints = all_profiles[next(iter(all_profiles.keys()))][angle]['endpoints']
                    x_start, y_start, x_end, y_end = endpoints
                    color = 'red' if angle == 0 else 'blue'
                    ax_domain.plot([x_start, x_end], [y_start, y_end],
                                 color=color, linewidth=3, alpha=0.9,
                                 label=f'{angle}°')
            
            ax_domain.set_xlabel("x (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_ylabel("y (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_title("von Mises Stress Field with HV Probes", 
                              fontsize=style_params['axes']['titlesize'] + 1,
                              fontweight='bold', pad=15)
            ax_domain.legend(fontsize=style_params['legend']['fontsize'])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_domain, shrink=0.8)
            cbar.set_label('von Mises Stress σ_vM (GPa)', 
                          fontsize=style_params['colorbar']['fontsize'])
    
    @staticmethod
    def _create_magnitude_visualization(fig, ax_main, ax_hv, ax_diag, ax_stats, ax_polar, ax_domain,
                                       simulations, frames, stress_key, all_profiles, colors, config, style_params):
        """Create specialized visualization for stress magnitude"""
        
        line_width = style_params['lines']['linewidth']
        
        # Main plot: Overlay of all profiles
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            sim_id = sim.get('id', f'sim_{sim_idx}')
            
            for angle in [0, 45, 90, 135]:
                if angle in all_profiles[sim_id]:
                    data = all_profiles[sim_id][angle]
                    linestyle = ['-', '--', '-.', ':'][[0, 45, 90, 135].index(angle)]
                    ax_main.plot(data['distances'], data['profile'],
                               color=colors[sim_idx], linewidth=line_width,
                               linestyle=linestyle, alpha=0.7,
                               label=f"{sim['params']['defect_type']} - {angle}°")
        
        ax_main.set_xlabel("Distance from Center (nm)", fontsize=style_params['axes']['labelsize'] + 1)
        ax_main.set_ylabel("Stress Magnitude |σ| (GPa)", fontsize=style_params['axes']['labelsize'] + 1)
        ax_main.set_title("Stress Magnitude: Multi-Orientation Analysis", 
                         fontsize=style_params['axes']['titlesize'] + 2,
                         fontweight='bold', pad=15)
        ax_main.legend(fontsize=style_params['legend']['fontsize'] - 1, ncol=2)
        ax_main.grid(True, alpha=0.3)
        ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # HV comparison plot
        PublicationVisualizer._plot_hv_comparison(ax_hv, simulations, all_profiles, colors, style_params)
        
        # Diagonal comparison plot
        PublicationVisualizer._plot_diagonal_comparison(ax_diag, simulations, all_profiles, colors, style_params)
        
        # Statistics plot
        PublicationVisualizer._plot_magnitude_statistics(ax_stats, simulations, all_profiles, colors, style_params)
        
        # Polar distribution plot
        if len(simulations) > 0:
            sim = simulations[0]
            sim_id = next(iter(all_profiles.keys()))
            
            # Collect peak values at different angles
            angles = []
            peaks = []
            
            for angle in [0, 45, 90, 135]:
                if angle in all_profiles[sim_id]:
                    metadata = all_profiles[sim_id][angle]['metadata']
                    angles.append(np.deg2rad(angle))
                    peaks.append(metadata['max_value'])
            
            if angles and peaks:
                # Close the polar plot
                angles = angles + [angles[0]]
                peaks = peaks + [peaks[0]]
                
                ax_polar.plot(angles, peaks, 'b-', linewidth=line_width, alpha=0.7)
                ax_polar.fill(angles, peaks, 'b', alpha=0.3)
                ax_polar.set_title("Angular Distribution of Peak Stress", 
                                 fontsize=style_params['axes']['titlesize'],
                                 fontweight='bold', pad=20)
                ax_polar.grid(True, alpha=0.5)
        
        # Domain plot with all probe lines
        if simulations:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            
            ax_domain.set_aspect('equal')
            cmap = plt.cm.get_cmap(COLORMAPS.get('viridis', 'viridis'))
            im = ax_domain.imshow(stress_fields[stress_key], extent=extent, 
                                cmap=cmap, origin='lower', alpha=0.85)
            
            # Add all probe lines
            line_colors = ['red', 'blue', 'green', 'purple']
            for idx, angle in enumerate([0, 45, 90, 135]):
                if angle in all_profiles[next(iter(all_profiles.keys()))]:
                    endpoints = all_profiles[next(iter(all_profiles.keys()))][angle]['endpoints']
                    x_start, y_start, x_end, y_end = endpoints
                    ax_domain.plot([x_start, x_end], [y_start, y_end],
                                 color=line_colors[idx], linewidth=2.5, alpha=0.8,
                                 label=f'{angle}°')
            
            ax_domain.set_xlabel("x (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_ylabel("y (nm)", fontsize=style_params['axes']['labelsize'] + 1)
            ax_domain.set_title("Stress Magnitude Field with Multi-Orientation Probes", 
                              fontsize=style_params['axes']['titlesize'] + 1,
                              fontweight='bold', pad=15)
            ax_domain.legend(fontsize=style_params['legend']['fontsize'], ncol=2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_domain, shrink=0.8)
            cbar.set_label('Stress Magnitude |σ| (GPa)', 
                          fontsize=style_params['colorbar']['fontsize'])
    
    @staticmethod
    def _plot_individual_diagonal(ax, simulations, all_profiles, angle, colors, style_params):
        """Plot individual diagonal profile"""
        line_width = style_params['lines']['linewidth']
        
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            if angle in profiles:
                data = profiles[angle]
                ax.plot(data['distances'], data['profile'],
                       color=colors[sim_idx], linewidth=line_width,
                       alpha=0.8, label=simulations[sim_idx]['params']['defect_type'])
        
        ax.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
        ax.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
        ax.set_title(f"{angle}° Diagonal Profile", 
                    fontsize=style_params['axes']['titlesize'],
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        if angle == 45:
            ax.legend(fontsize=style_params['legend']['fontsize'] - 1)
    
    @staticmethod
    def _plot_individual_orientation(ax, simulations, all_profiles, angle, colors, style_params, orientation_name):
        """Plot individual orientation profile"""
        line_width = style_params['lines']['linewidth']
        
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            if angle in profiles:
                data = profiles[angle]
                ax.plot(data['distances'], data['profile'],
                       color=colors[sim_idx], linewidth=line_width,
                       alpha=0.8, label=simulations[sim_idx]['params']['defect_type'])
        
        ax.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
        ax.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
        ax.set_title(f"{orientation_name} Profile", 
                    fontsize=style_params['axes']['titlesize'],
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    @staticmethod
    def _plot_hv_comparison(ax, simulations, all_profiles, colors, style_params):
        """Plot horizontal vs vertical comparison"""
        line_width = style_params['lines']['linewidth']
        
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            if 0 in profiles and 90 in profiles:
                data0 = profiles[0]
                data90 = profiles[90]
                
                # Plot horizontal
                ax.plot(data0['distances'], data0['profile'],
                       color=colors[sim_idx], linewidth=line_width,
                       linestyle='-', alpha=0.6,
                       label=f"{simulations[sim_idx]['params']['defect_type']} - H")
                
                # Plot vertical
                ax.plot(data90['distances'], data90['profile'],
                       color=colors[sim_idx], linewidth=line_width,
                       linestyle='--', alpha=0.6,
                       label=f"{simulations[sim_idx]['params']['defect_type']} - V")
        
        ax.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
        ax.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
        ax.set_title("Horizontal vs Vertical Comparison", 
                    fontsize=style_params['axes']['titlesize'],
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    @staticmethod
    def _plot_diagonal_comparison(ax, simulations, all_profiles, colors, style_params):
        """Plot diagonal comparison"""
        line_width = style_params['lines']['linewidth']
        
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            if 45 in profiles and 135 in profiles:
                data45 = profiles[45]
                data135 = profiles[135]
                
                # Plot 45° diagonal
                ax.plot(data45['distances'], data45['profile'],
                       color=colors[sim_idx], linewidth=line_width,
                       linestyle='-', alpha=0.6,
                       label=f"{simulations[sim_idx]['params']['defect_type']} - 45°")
                
                # Plot 135° diagonal
                ax.plot(data135['distances'], data135['profile'],
                       color=colors[sim_idx], linewidth=line_width,
                       linestyle='--', alpha=0.6,
                       label=f"{simulations[sim_idx]['params']['defect_type']} - 135°")
        
        ax.set_xlabel("Distance (nm)", fontsize=style_params['axes']['labelsize'])
        ax.set_ylabel("Stress (GPa)", fontsize=style_params['axes']['labelsize'])
        ax.set_title("Diagonal Comparison", 
                    fontsize=style_params['axes']['titlesize'],
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    @staticmethod
    def _plot_magnitude_statistics(ax, simulations, all_profiles, colors, style_params):
        """Plot stress magnitude statistics"""
        
        # Collect statistics
        stats_data = []
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            for angle, data in profiles.items():
                metadata = data['metadata']
                stats_data.append({
                    'Simulation': simulations[sim_idx]['params']['defect_type'],
                    'Angle': f'{angle}°',
                    'Peak (GPa)': metadata['max_value'],
                    'FWHM (nm)': metadata['fwhm_nm'],
                    'Symmetry': metadata['symmetry_index']
                })
        
        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            
            # Create grouped bar plot for peak stresses
            unique_sims = df_stats['Simulation'].unique()
            unique_angles = sorted(df_stats['Angle'].unique())
            
            x = np.arange(len(unique_sims))
            width = 0.8 / len(unique_angles)
            
            for i, angle in enumerate(unique_angles):
                angle_data = df_stats[df_stats['Angle'] == angle]
                peak_values = []
                for sim in unique_sims:
                    sim_data = angle_data[angle_data['Simulation'] == sim]
                    peak_values.append(sim_data['Peak (GPa)'].values[0] if not sim_data.empty else 0)
                
                bars = ax.bar(x + i*width - width*(len(unique_angles)-1)/2, 
                            peak_values, width,
                            label=angle, alpha=0.8)
            
            ax.set_xlabel("Simulation", fontsize=style_params['axes']['labelsize'])
            ax.set_ylabel("Peak Stress (GPa)", fontsize=style_params['axes']['labelsize'])
            ax.set_title("Peak Stress Statistics by Orientation", 
                        fontsize=style_params['axes']['titlesize'],
                        fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(unique_sims, rotation=45, ha='right')
            ax.legend(fontsize=style_params['legend']['fontsize'] - 1)
            ax.grid(True, alpha=0.3, axis='y')

# =============================================
# ENHANCED VISUALIZATION CONTROLS
# =============================================
class EnhancedVisualizationControls:
    """Enhanced controls for publication-quality visualization"""
    
    @staticmethod
    def get_visualization_controls():
        """Get comprehensive visualization controls"""
        
        st.sidebar.markdown("---")
        st.sidebar.markdown('<h3 style="color: #2D3748; margin-bottom: 1rem;">🎨 Publication Visualization</h3>', 
                          unsafe_allow_html=True)
        
        controls = {}
        
        with st.sidebar.expander("📊 Plot Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                controls['figure_size'] = st.select_slider(
                    "Figure Size",
                    options=['Small', 'Medium', 'Large', 'Extra Large'],
                    value='Large'
                )
                controls['dpi'] = st.select_slider(
                    "Resolution (DPI)",
                    options=[150, 300, 600, 1200],
                    value=600
                )
            with col2:
                controls['color_scheme'] = st.selectbox(
                    "Color Scheme",
                    ['Default', 'Nature', 'Science', 'Advanced Materials', 'Custom'],
                    index=0
                )
                controls['export_format'] = st.multiselect(
                    "Export Formats",
                    ['PNG', 'PDF', 'SVG', 'EPS'],
                    default=['PNG', 'PDF']
                )
        
        with st.sidebar.expander("📐 Probe Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                controls['hydrostatic_angles'] = st.multiselect(
                    "Hydrostatic Stress Probes",
                    [0, 45, 90, 135],
                    default=[45, 135],
                    help="Optimal: 45°, 135° for hydrostatic stress"
                )
                controls['vonmises_angles'] = st.multiselect(
                    "von Mises Stress Probes",
                    [0, 45, 90, 135],
                    default=[0, 90],
                    help="Optimal: 0°, 90° for von Mises stress"
                )
            with col2:
                controls['magnitude_angles'] = st.multiselect(
                    "Stress Magnitude Probes",
                    [0, 45, 90, 135],
                    default=[0, 45, 90, 135],
                    help="All orientations for magnitude analysis"
                )
                controls['line_width'] = st.slider(
                    "Line Width",
                    1.0, 5.0, 2.5, 0.5
                )
        
        with st.sidebar.expander("🎯 Advanced Styling", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                controls['font_size'] = st.slider(
                    "Base Font Size",
                    8, 20, 12, 1
                )
                controls['grid_alpha'] = st.slider(
                    "Grid Opacity",
                    0.0, 1.0, 0.3, 0.05
                )
            with col2:
                controls['legend_position'] = st.selectbox(
                    "Legend Position",
                    ['best', 'upper right', 'upper left', 'lower right', 'lower left', 'center'],
                    index=0
                )
                controls['show_minor_ticks'] = st.checkbox(
                    "Show Minor Ticks",
                    True
                )
        
        with st.sidebar.expander("📈 Statistical Analysis", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                controls['show_statistics'] = st.checkbox(
                    "Show Statistics",
                    True
                )
                controls['calculate_symmetry'] = st.checkbox(
                    "Calculate Symmetry",
                    True
                )
            with col2:
                controls['show_confidence'] = st.checkbox(
                    "Show Confidence Intervals",
                    False
                )
                controls['error_bar_type'] = st.selectbox(
                    "Error Bar Type",
                    ['Standard Deviation', 'Standard Error', '95% Confidence'],
                    index=0
                )
        
        return controls

# =============================================
# MAIN VISUALIZATION INTERFACE
# =============================================
def create_enhanced_visualization_interface():
    """Create enhanced visualization interface"""
    
    st.markdown('<div class="section-header">📊 Publication-Quality Stress Visualization</div>', 
                unsafe_allow_html=True)
    
    # Get visualization controls
    controls = EnhancedVisualizationControls.get_visualization_controls()
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Component-Specific Analysis",
        "📈 Multi-Orientation Overlay", 
        "🌀 Comparative Analysis",
        "📊 Statistical Summary"
    ])
    
    with tab1:
        st.markdown('<h3 style="color: #2D3748; margin-bottom: 1.5rem;">Stress Component-Specific Visualization</h3>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            component = st.selectbox(
                "Stress Component",
                ["Hydrostatic σ_h", "von Mises σ_vM", "Stress Magnitude |σ|"],
                index=0,
                key="component_select"
            )
            
            # Show optimal probe recommendations
            if component == "Hydrostatic σ_h":
                st.info("**Optimal Probes:** 45°, 135° diagonals\n\nHydrostatic stress shows maximum sensitivity along crystal diagonals.")
            elif component == "von Mises σ_vM":
                st.info("**Optimal Probes:** 0°, 90° (horizontal/vertical)\n\nvon Mises stress is best analyzed along principal axes.")
            else:
                st.info("**Optimal Probes:** All orientations (0°, 45°, 90°, 135°)\n\nStress magnitude requires multi-orientation analysis.")
        
        with col2:
            # Get sample simulations from session state
            if 'simulations' in st.session_state and st.session_state.simulations:
                sim_options = list(st.session_state.simulations.keys())
                selected_sims = st.multiselect(
                    "Select Simulations",
                    sim_options,
                    default=sim_options[:min(3, len(sim_options))],
                    key="comp_sim_select"
                )
                
                if selected_sims:
                    frame_idx = st.slider(
                        "Frame Index",
                        0, 100, 50,
                        key="comp_frame_slider"
                    )
                else:
                    st.warning("No simulations selected")
            else:
                st.info("Run simulations first to enable visualization")
        
        with col3:
            # Visualization options
            colormap = st.selectbox(
                "Colormap",
                cmap_list,
                index=cmap_list.index('coolwarm' if component == "Hydrostatic σ_h" else 
                                     'plasma' if component == "von Mises σ_vM" else 
                                     'viridis'),
                key="comp_cmap"
            )
            
            show_domain = st.checkbox("Show Domain with Probes", True, key="comp_show_domain")
            show_stats = st.checkbox("Show Statistics", True, key="comp_show_stats")
        
        # Generate visualization button
        if st.button("🎨 Generate Publication Figure", type="primary", key="gen_comp_fig"):
            if 'simulations' in st.session_state and selected_sims:
                # Prepare simulation data
                simulations = []
                frames = []
                
                for sim_id in selected_sims:
                    sim_data = st.session_state.simulations[sim_id]
                    simulations.append(sim_data)
                    frames.append(min(frame_idx, len(sim_data['history']) - 1))
                
                # Create configuration
                config = {
                    'stress_component': component,
                    'orientations': controls['hydrostatic_angles'] if component == "Hydrostatic σ_h" else
                                   controls['vonmises_angles'] if component == "von Mises σ_vM" else
                                   controls['magnitude_angles'],
                    'line_style': 'solid',
                    'show_domain': show_domain,
                    'show_statistics': show_stats
                }
                
                # Create styling parameters
                style_params = PublicationVisualizer.create_enhanced_styling_params()
                style_params['lines']['linewidth'] = controls['line_width']
                style_params['figure']['dpi'] = controls['dpi']
                style_params['axes']['labelsize'] = controls['font_size'] + 2
                style_params['axes']['titlesize'] = controls['font_size'] + 4
                style_params['legend']['fontsize'] = controls['font_size']
                
                # Generate figure
                with st.spinner("Creating publication-quality figure..."):
                    fig = PublicationVisualizer.create_stress_component_specific_plot(
                        simulations, frames, config, style_params
                    )
                    
                    # Display figure
                    st.pyplot(fig)
                    
                    # Export options
                    with st.expander("💾 Export Options", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'PNG' in controls['export_format']:
                                buf = BytesIO()
                                fig.savefig(buf, format='png', dpi=controls['dpi'], 
                                           bbox_inches='tight', facecolor='white')
                                buf.seek(0)
                                
                                st.download_button(
                                    label="📸 Download PNG",
                                    data=buf,
                                    file_name=f"{component.replace(' ', '_')}_analysis.png",
                                    mime="image/png"
                                )
                        
                        with col2:
                            if 'PDF' in controls['export_format']:
                                buf = BytesIO()
                                fig.savefig(buf, format='pdf', dpi=controls['dpi'],
                                           bbox_inches='tight', facecolor='white')
                                buf.seek(0)
                                
                                st.download_button(
                                    label="📄 Download PDF",
                                    data=buf,
                                    file_name=f"{component.replace(' ', '_')}_analysis.pdf",
                                    mime="application/pdf"
                                )
                        
                        with col3:
                            # Copy figure settings
                            fig_settings = {
                                'component': component,
                                'simulations': selected_sims,
                                'frame': frame_idx,
                                'colormap': colormap,
                                'style_params': style_params
                            }
                            
                            st.download_button(
                                label="⚙️ Export Settings",
                                data=json.dumps(fig_settings, indent=2),
                                file_name="figure_settings.json",
                                mime="application/json"
                            )
            else:
                st.error("No simulation data available. Please run simulations first.")
    
    with tab2:
        st.markdown('<h3 style="color: #2D3748; margin-bottom: 1.5rem;">Multi-Orientation Overlay Analysis</h3>', 
                   unsafe_allow_html=True)
        
        # Multi-orientation overlay configuration
        col1, col2 = st.columns(2)
        
        with col1:
            overlay_component = st.selectbox(
                "Stress Component for Overlay",
                ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
                index=0,
                key="overlay_component"
            )
            
            overlay_orientations = st.multiselect(
                "Select Orientations",
                [0, 45, 90, 135],
                default=[0, 45, 90, 135],
                key="overlay_orientations"
            )
            
            line_style = st.selectbox(
                "Line Style",
                ['solid', 'dashed', 'dotted', 'dashdot'],
                index=0,
                key="overlay_line_style"
            )
        
        with col2:
            if 'simulations' in st.session_state and st.session_state.simulations:
                sim_options = list(st.session_state.simulations.keys())
                overlay_sims = st.multiselect(
                    "Select Simulations for Overlay",
                    sim_options,
                    default=sim_options[:min(4, len(sim_options))],
                    key="overlay_sims"
                )
                
                if overlay_sims:
                    overlay_frame = st.slider(
                        "Frame Index",
                        0, 100, 50,
                        key="overlay_frame"
                    )
            else:
                st.info("Run simulations first to enable overlay visualization")
        
        # Generate overlay plot
        if st.button("📈 Generate Overlay Plot", type="primary", key="gen_overlay"):
            if 'simulations' in st.session_state and overlay_sims:
                # Prepare data
                simulations = []
                frames = []
                
                for sim_id in overlay_sims:
                    sim_data = st.session_state.simulations[sim_id]
                    simulations.append(sim_data)
                    frames.append(min(overlay_frame, len(sim_data['history']) - 1))
                
                # Create configuration
                config = {
                    'stress_component': overlay_component,
                    'orientations': overlay_orientations,
                    'line_style': line_style
                }
                
                # Generate figure
                with st.spinner("Creating multi-orientation overlay..."):
                    fig = PublicationVisualizer.create_comprehensive_overlay_plot(
                        simulations, frames, config
                    )
                    
                    # Display figure
                    st.pyplot(fig)
                    
                    # Show statistics
                    with st.expander("📊 Overlay Statistics", expanded=True):
                        # Extract and display profile statistics
                        analyzer = PublicationProfileExtractor()
                        stats_data = []
                        
                        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                            eta, stress_fields = sim['history'][frame]
                            stress_key = {
                                "Stress Magnitude |σ|": 'sigma_mag',
                                "Hydrostatic σ_h": 'sigma_hydro',
                                "von Mises σ_vM": 'von_mises'
                            }[overlay_component]
                            
                            stress_data = stress_fields[stress_key]
                            
                            for angle in overlay_orientations:
                                distances, profile, endpoints, metadata = analyzer.extract_profile_2d(
                                    stress_data, angle, 'center', 0, 0.7
                                )
                                
                                stats_data.append({
                                    'Simulation': sim['params']['defect_type'],
                                    'Orientation': f'{angle}°',
                                    'Max (GPa)': metadata['max_value'],
                                    'Mean (GPa)': metadata['mean_value'],
                                    'FWHM (nm)': metadata['fwhm_nm'],
                                    'Peak Position (nm)': metadata['peak_position_nm']
                                })
                        
                        if stats_data:
                            df_stats = pd.DataFrame(stats_data)
                            st.dataframe(
                                df_stats.style.format({
                                    'Max (GPa)': '{:.3f}',
                                    'Mean (GPa)': '{:.3f}',
                                    'FWHM (nm)': '{:.2f}',
                                    'Peak Position (nm)': '{:.2f}'
                                }),
                                use_container_width=True
                            )
            else:
                st.error("Please select simulations for overlay analysis.")
    
    with tab3:
        st.markdown('<h3 style="color: #2D3748; margin-bottom: 1.5rem;">Comparative Stress Analysis</h3>', 
                   unsafe_allow_html=True)
        
        # Comparative analysis configuration
        col1, col2 = st.columns(2)
        
        with col1:
            compare_type = st.selectbox(
                "Comparison Type",
                ["Hydrostatic vs von Mises", "Diagonal vs Horizontal/Vertical", 
                 "Multi-Component Analysis", "Orientation Sensitivity"],
                index=0,
                key="compare_type"
            )
            
            if 'simulations' in st.session_state and st.session_state.simulations:
                compare_sims = st.multiselect(
                    "Select Simulations to Compare",
                    list(st.session_state.simulations.keys()),
                    default=list(st.session_state.simulations.keys())[:2],
                    key="compare_sims"
                )
            else:
                st.info("Run at least 2 simulations for comparison")
        
        with col2:
            compare_metric = st.selectbox(
                "Comparison Metric",
                ["Peak Stress", "Stress Distribution", "Symmetry Index", 
                 "FWHM", "Gradient Analysis"],
                index=0,
                key="compare_metric"
            )
            
            compare_frame = st.slider(
                "Comparison Frame",
                0, 100, 50,
                key="compare_frame"
            )
        
        # Generate comparative analysis
        if st.button("🔬 Generate Comparative Analysis", type="primary", key="gen_compare"):
            if 'simulations' in st.session_state and len(compare_sims) >= 2:
                # Prepare comparative visualization
                st.success(f"Comparative analysis of {len(compare_sims)} simulations")
                
                # Create comparative plots based on selection
                if compare_type == "Hydrostatic vs von Mises":
                    PublicationVisualizer._create_component_comparison(compare_sims, compare_frame)
                elif compare_type == "Diagonal vs Horizontal/Vertical":
                    PublicationVisualizer._create_orientation_comparison(compare_sims, compare_frame)
                # Add other comparison types as needed
            else:
                st.error("Select at least 2 simulations for comparative analysis.")
    
    with tab4:
        st.markdown('<h3 style="color: #2D3748; margin-bottom: 1.5rem;">Statistical Summary & Insights</h3>', 
                   unsafe_allow_html=True)
        
        # Statistical summary configuration
        if 'simulations' in st.session_state and st.session_state.simulations:
            # Calculate overall statistics
            total_sims = len(st.session_state.simulations)
            total_frames = sum(len(sim['history']) for sim in st.session_state.simulations.values())
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Simulations", total_sims)
            with col2:
                st.metric("Total Frames", total_frames)
            with col3:
                # Count by defect type
                defect_counts = {}
                for sim in st.session_state.simulations.values():
                    defect = sim['params']['defect_type']
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
                st.metric("Defect Types", len(defect_counts))
            with col4:
                # Average stress values
                avg_stresses = []
                for sim in st.session_state.simulations.values():
                    for eta, stress_fields in sim['history']:
                        avg_stresses.append(np.mean(stress_fields['sigma_mag']))
                st.metric("Avg Stress", f"{np.mean(avg_stresses):.2f} GPa")
            
            # Detailed statistics
            with st.expander("📈 Detailed Statistics", expanded=True):
                # Create comprehensive statistics table
                stats_table = []
                
                for sim_id, sim_data in st.session_state.simulations.items():
                    params = sim_data['params']
                    history = sim_data['history']
                    
                    if history:
                        # Get final frame statistics
                        eta, stress_fields = history[-1]
                        
                        stats_table.append({
                            'ID': sim_id[:8],
                            'Defect': params['defect_type'],
                            'ε*': params['eps0'],
                            'κ': params['kappa'],
                            'Frames': len(history),
                            'Max |σ| (GPa)': np.max(stress_fields['sigma_mag']),
                            'Mean σ_h (GPa)': np.mean(stress_fields['sigma_hydro']),
                            'Max σ_vM (GPa)': np.max(stress_fields['von_mises']),
                            'Defect Area (nm²)': np.sum(eta > 0.5) * dx**2
                        })
                
                if stats_table:
                    df_stats = pd.DataFrame(stats_table)
                    st.dataframe(
                        df_stats.style.format({
                            'ε*': '{:.3f}',
                            'κ': '{:.2f}',
                            'Max |σ| (GPa)': '{:.3f}',
                            'Mean σ_h (GPa)': '{:.3f}',
                            'Max σ_vM (GPa)': '{:.3f}',
                            'Defect Area (nm²)': '{:.1f}'
                        }),
                        use_container_width=True
                    )
            
            # Generate insights
            with st.expander("🔍 Scientific Insights", expanded=True):
                st.markdown("""
                ### Key Observations from Stress Analysis:
                
                **Hydrostatic Stress (σ_h):**
                - Best visualized along **45° and 135° diagonals**
                - Shows maximum sensitivity to crystal symmetry
                - Diagonal probes capture dilatational components
                
                **von Mises Stress (σ_vM):**
                - Optimal analysis with **0° and 90° probes**
                - Horizontal/vertical probes capture shear components
                - Shows material yield criteria sensitivity
                
                **Stress Magnitude (|σ|):**
                - Requires **multi-orientation analysis** (0°, 45°, 90°, 135°)
                - Provides comprehensive stress field characterization
                - Essential for failure analysis
                
                ### Publication Recommendations:
                1. **Hydrostatic stress**: Use diagonal probes with coolwarm colormap
                2. **von Mises stress**: Use horizontal/vertical probes with plasma colormap
                3. **Stress magnitude**: Multi-orientation overlay with viridis colormap
                4. **Comparative studies**: Include symmetry analysis and statistical validation
                
                ### Best Practices:
                - Always include scale bars and physical units
                - Use consistent color schemes across figures
                - Report statistical significance of observations
                - Include error bars for experimental comparisons
                """)
        else:
            st.info("Run simulations to generate statistical summary")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application function"""
    
    # Initialize session state for simulations
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    
    # Create main interface
    create_enhanced_visualization_interface()
    
    # Add enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%); border-radius: 20px; margin-top: 3rem;">
        <p style="font-size: 1.3rem; font-weight: 700; color: #2D3748; margin-bottom: 1rem;">
            🔬 Ag Nanoparticle Stress Analysis Platform Pro
        </p>
        <p style="color: #4A5568; margin-bottom: 1.5rem;">
            Publication-quality stress visualization • Multi-orientation profiling • Enhanced post-processing
        </p>
        <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #667eea; margin-bottom: 0.5rem;">📐</div>
                <div style="font-weight: 600; color: #4A5568;">Hydrostatic Stress</div>
                <div style="color: #718096; font-size: 0.9rem;">Diagonal probes (45°, 135°)</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #764ba2; margin-bottom: 0.5rem;">📈</div>
                <div style="font-weight: 600; color: #4A5568;">von Mises Stress</div>
                <div style="color: #718096; font-size: 0.9rem;">HV probes (0°, 90°)</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #f5576c; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-weight: 600; color: #4A5568;">Stress Magnitude</div>
                <div style="color: #718096; font-size: 0.9rem;">Multi-orientation</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #ffd166; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600; color: #4A5568;">Publication Ready</div>
                <div style="color: #718096; font-size: 0.9rem;">600 DPI export</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
