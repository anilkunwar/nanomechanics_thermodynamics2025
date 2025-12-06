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

# Grid parameters - Using original robust values
N = 128  # Original robust value
dx = 0.1  # Original robust value (nm)
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

# =============================================
# POST-PROCESSING STYLING SYSTEM
# =============================================
class FigureStyler:
    """Advanced figure styling and post-processing system"""
    
    @staticmethod
    def apply_advanced_styling(fig, axes, style_params):
        """Apply advanced styling to figure and axes"""
        
        # Apply to all axes in figure
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        for ax in axes_flat:
            if ax is not None:
                # Apply axis styling
                ax.tick_params(axis='both', which='major', 
                              labelsize=style_params.get('tick_font_size', 12),
                              width=style_params.get('tick_width', 2.0),
                              length=style_params.get('tick_length', 6))
                
                # Apply spine styling
                for spine in ax.spines.values():
                    spine.set_linewidth(style_params.get('spine_width', 2.5))
                    spine.set_color(style_params.get('spine_color', 'black'))
                
                # Apply grid if requested
                if style_params.get('show_grid', True):
                    ax.grid(True, 
                           alpha=style_params.get('grid_alpha', 0.3),
                           linestyle=style_params.get('grid_style', '--'),
                           linewidth=style_params.get('grid_width', 0.5))
                
                # Apply title styling
                if hasattr(ax, 'title'):
                    title = ax.get_title()
                    if title:
                        ax.set_title(title, 
                                    fontsize=style_params.get('title_font_size', 16),
                                    fontweight=style_params.get('title_weight', 'bold'),
                                    color=style_params.get('title_color', 'black'))
                
                # Apply label styling
                if ax.get_xlabel():
                    ax.set_xlabel(ax.get_xlabel(),
                                 fontsize=style_params.get('label_font_size', 14),
                                 fontweight=style_params.get('label_weight', 'bold'))
                if ax.get_ylabel():
                    ax.set_ylabel(ax.get_ylabel(),
                                 fontsize=style_params.get('label_font_size', 14),
                                 fontweight=style_params.get('label_weight', 'bold'))
        
        # Apply figure background
        if style_params.get('figure_facecolor'):
            fig.set_facecolor(style_params['figure_facecolor'])
        
        return fig
    
    @staticmethod
    def get_styling_controls():
        """Get comprehensive styling controls"""
        style_params = {}
        
        st.sidebar.header("🎨 Advanced Post-Processing")
        
        with st.sidebar.expander("📐 Font & Text Styling", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['title_font_size'] = st.slider("Title Size", 8, 32, 16)
                style_params['label_font_size'] = st.slider("Label Size", 8, 28, 14)
                style_params['tick_font_size'] = st.slider("Tick Size", 6, 20, 12)
            with col2:
                style_params['title_weight'] = st.selectbox("Title Weight", 
                                                           ['normal', 'bold', 'light', 'semibold'], 
                                                           index=1)
                style_params['label_weight'] = st.selectbox("Label Weight", 
                                                           ['normal', 'bold', 'light'], 
                                                           index=1)
                style_params['title_color'] = st.color_picker("Title Color", "#000000")
        
        with st.sidebar.expander("📏 Line & Border Styling", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['line_width'] = st.slider("Line Width", 0.5, 5.0, 2.0, 0.5)
                style_params['spine_width'] = st.slider("Spine Width", 1.0, 4.0, 2.5, 0.5)
                style_params['tick_width'] = st.slider("Tick Width", 0.5, 3.0, 2.0, 0.5)
            with col2:
                style_params['tick_length'] = st.slider("Tick Length", 2, 15, 6)
                style_params['spine_color'] = st.color_picker("Spine Color", "#000000")
                style_params['grid_width'] = st.slider("Grid Width", 0.1, 2.0, 0.5, 0.1)
        
        with st.sidebar.expander("🌐 Grid & Background", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['show_grid'] = st.checkbox("Show Grid", True)
                style_params['grid_style'] = st.selectbox("Grid Style", 
                                                         ['-', '--', '-.', ':'],
                                                         index=1)
                style_params['grid_alpha'] = st.slider("Grid Alpha", 0.0, 1.0, 0.3, 0.05)
            with col2:
                style_params['figure_facecolor'] = st.color_picker("Figure Background", "#FFFFFF")
                style_params['axes_facecolor'] = st.color_picker("Axes Background", "#FFFFFF")
        
        with st.sidebar.expander("📊 Legend & Annotation", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['legend_fontsize'] = st.slider("Legend Size", 8, 20, 12)
                style_params['legend_location'] = st.selectbox("Legend Location",
                                                              ['best', 'upper right', 'upper left', 
                                                               'lower right', 'lower left', 'center'],
                                                              index=0)
            with col2:
                style_params['show_legend'] = st.checkbox("Show Legend", True)
                style_params['legend_frame'] = st.checkbox("Legend Frame", True)
        
        with st.sidebar.expander("🎨 Colorbar Styling", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['colorbar_fontsize'] = st.slider("Colorbar Font", 8, 20, 12)
                style_params['colorbar_width'] = st.slider("Colorbar Width", 0.2, 1.0, 0.6, 0.05)
            with col2:
                style_params['colorbar_shrink'] = st.slider("Colorbar Shrink", 0.5, 1.0, 0.8, 0.05)
                style_params['colorbar_pad'] = st.slider("Colorbar Pad", 0.0, 0.2, 0.05, 0.01)
        
        return style_params

# =============================================
# ENHANCED FIGURE STYLER WITH PUBLICATION FEATURES
# =============================================
class EnhancedFigureStyler(FigureStyler):
    """Extended figure styler with publication-quality enhancements"""
    
    @staticmethod
    def apply_publication_styling(fig, axes, style_params):
        """Apply enhanced publication styling"""
        # Apply base styling
        fig = FigureStyler.apply_advanced_styling(fig, axes, style_params)
        
        # Get axes list
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        elif isinstance(axes, list):
            axes_flat = axes
        else:
            axes_flat = [axes]
        
        # Enhanced styling for each axis
        for ax in axes_flat:
            if ax is not None:
                # Set scientific notation for large/small numbers
                try:
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
                except AttributeError:
                    pass
                try:
                    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useMathText=True)
                except AttributeError:
                    pass
                
                # Add minor ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                
                # Set tick parameters
                ax.tick_params(which='both', direction='in', top=True, right=True)
                ax.tick_params(which='major', length=6, width=style_params.get('tick_width', 1.0))
                ax.tick_params(which='minor', length=3, width=style_params.get('tick_width', 1.0) * 0.5)
                
                # Format axis labels with LaTeX
                if style_params.get('use_latex', False):
                    xlabel = ax.get_xlabel()
                    ylabel = ax.get_ylabel()
                    if xlabel:
                        ax.set_xlabel(f'${xlabel}$')
                    if ylabel:
                        ax.set_ylabel(f'${ylabel}$')
        
        # Adjust layout
        fig.set_constrained_layout(True)
        
        return fig
    
    @staticmethod
    def get_publication_controls():
        """Get enhanced publication styling controls"""
        style_params = FigureStyler.get_styling_controls()
        
        st.sidebar.header("📰 Publication-Quality Settings")
        
        with st.sidebar.expander("🎯 Journal Templates", expanded=False):
            journal = st.selectbox(
                "Journal Style",
                ["Nature", "Science", "Advanced Materials", "Physical Review Letters", "Custom"],
                index=0,
                key="pub_journal_style"
            )
            
            style_params['journal_style'] = journal.lower()
            style_params['use_latex'] = st.checkbox("Use LaTeX Formatting", False, key="pub_use_latex")
            style_params['vector_output'] = st.checkbox("Enable Vector Export (PDF/SVG)", True, key="pub_vector_export")
        
        with st.sidebar.expander("📐 Advanced Layout", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['layout_pad'] = st.slider("Layout Padding", 0.5, 3.0, 1.0, 0.1,
                                                       key="pub_layout_pad")
                style_params['wspace'] = st.slider("Horizontal Spacing", 0.1, 1.0, 0.3, 0.05,
                                                   key="pub_wspace")
            with col2:
                style_params['hspace'] = st.slider("Vertical Spacing", 0.1, 1.0, 0.4, 0.05,
                                                   key="pub_hspace")
                style_params['figure_dpi'] = st.select_slider(
                    "Figure DPI", 
                    options=[150, 300, 600, 1200], 
                    value=600,
                    key="pub_figure_dpi"
                )
        
        with st.sidebar.expander("📈 Enhanced Plot Features", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['show_minor_ticks'] = st.checkbox("Show Minor Ticks", True,
                                                               key="pub_minor_ticks")
                style_params['show_error_bars'] = st.checkbox("Show Error Bars", True,
                                                              key="pub_error_bars")
                style_params['show_confidence'] = st.checkbox("Show Confidence Intervals", False,
                                                              key="pub_confidence")
            with col2:
                style_params['grid_style'] = st.selectbox(
                    "Grid Style", 
                    ['-', '--', '-.', ':'],
                    index=1,
                    key="pub_grid_style"
                )
                style_params['grid_zorder'] = st.slider("Grid Z-Order", 0, 10, 0,
                                                        key="pub_grid_zorder")
        
        with st.sidebar.expander("🎨 Enhanced Color Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                style_params['colorbar_extend'] = st.selectbox(
                    "Colorbar Extend", 
                    ['neither', 'both', 'min', 'max'],
                    index=0,
                    key="pub_colorbar_extend"
                )
                style_params['colorbar_format'] = st.selectbox(
                    "Colorbar Format", 
                    ['auto', 'sci', 'plain'],
                    index=0,
                    key="pub_colorbar_format"
                )
            with col2:
                style_params['cmap_normalization'] = st.selectbox(
                    "Colormap Normalization",
                    ['linear', 'log', 'power'],
                    index=0,
                    key="pub_cmap_normalization"
                )
                if style_params['cmap_normalization'] == 'power':
                    style_params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1,
                                                      key="pub_gamma")
        
        return style_params

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
# ENHANCED PUBLICATION-QUALITY PLOTTING FUNCTIONS
# =============================================
def create_publication_heatmaps(simulations, frames, config, style_params):
    """Publication-quality heatmap comparison"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    n_sims = len(simulations)
    cols = min(3, n_sims)
    rows = (n_sims + cols - 1) // cols
    
    # Create figure with journal sizing
    journal_styles = JournalTemplates.get_journal_styles()
    journal = style_params.get('journal_style', 'nature')
    fig_width = journal_styles[journal]['figure_width_double'] / 2.54  # Convert cm to inches
    
    fig, axes = plt.subplots(rows, cols, 
                            figsize=(fig_width, fig_width * 0.8 * rows/cols),
                            constrained_layout=True)
    
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
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Apply smoothing for better visualization
        if style_params.get('apply_smoothing', True):
            stress_data = gaussian_filter(stress_data, sigma=1)
        
        # Choose colormap
        cmap_name = sim['params']['sigma_cmap']
        cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
        
        # Create heatmap with enhanced settings
        im = ax.imshow(stress_data, extent=extent, cmap=cmap, 
                      origin='lower', aspect='auto')
        
        # Add contour lines for defect boundary
        contour = ax.contour(X, Y, eta, levels=[0.5], colors='white', 
                           linewidths=1, linestyles='--', alpha=0.8)
        
        # Enhanced title
        title = f"{sim['params']['defect_type']}"
        if sim['params']['orientation'] != "Horizontal {111} (0°)":
            title += f"\n{sim['params']['orientation'].split(' ')[0]}"
        
        ax.set_title(title, fontsize=style_params.get('title_font_size', 10),
                    fontweight='semibold', pad=10)
        
        # Axis labels only on edge plots
        if row == rows - 1:
            ax.set_xlabel("x (nm)", fontsize=style_params.get('label_font_size', 9))
        if col == 0:
            ax.set_ylabel("y (nm)", fontsize=style_params.get('label_font_size', 9))
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label(f"{config['stress_component']} (GPa)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    # Hide empty subplots
    for idx in range(n_sims, rows*cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
    
    return fig

def create_multi_orientation_analysis(simulations, frames, config, controls):
    """Create multi-orientation stress analysis with publication-quality visualization"""
    
    if not simulations:
        st.error("No simulations available for analysis")
        return
    
    # Get stress component
    stress_component = config.get('stress_component', 'Stress Magnitude |σ|')
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[stress_component]
    
    # Get optimal angles for this component
    if stress_key == 'sigma_hydro':
        orientations = controls.get('hydrostatic_angles', [45, 135])
    elif stress_key == 'von_mises':
        orientations = controls.get('vonmises_angles', [0, 90])
    else:
        orientations = controls.get('magnitude_angles', [0, 45, 90, 135])
    
    st.markdown(f'<div class="section-header">📊 {stress_component} - Multi-Orientation Analysis</div>', 
                unsafe_allow_html=True)
    
    # Create multi-panel visualization
    fig = plt.figure(figsize=(20, 16))
    fig.set_constrained_layout(True)
    
    # Define grid layout
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1.5], hspace=0.25, wspace=0.25)
    
    # Panel A: Main overlay plot
    ax_main = fig.add_subplot(gs[0, :])
    
    # Panel B: Individual orientation profiles
    orientation_axes = []
    for i, angle in enumerate(orientations[:4]):  # Show up to 4 orientations
        if i == 0:
            ax = fig.add_subplot(gs[1, i])
        else:
            ax = fig.add_subplot(gs[1, i])
        orientation_axes.append(ax)
    
    # Panel C: Statistical summary
    ax_stats = fig.add_subplot(gs[2, 0:2])
    
    # Panel D: Domain visualization
    ax_domain = fig.add_subplot(gs[2, 2:])
    
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
            
            # Plot in main overlay
            line_style = config.get('line_style', 'solid')
            label = f"{sim['params']['defect_type']} - {angle}°" if sim_idx == 0 else None
            
            ax_main.plot(distances, profile, 
                       color=colors[sim_idx],
                       linestyle=line_style,
                       linewidth=controls.get('line_width', 2.5),
                       alpha=0.85,
                       label=label)
    
    # Configure main overlay panel
    ax_main.set_xlabel("Distance from Center (nm)", fontsize=14, fontweight='bold')
    ax_main.set_ylabel(f"{stress_component} (GPa)", fontsize=14, fontweight='bold')
    ax_main.set_title(f"{stress_component}: Multi-Orientation Profile Overlay", 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.legend(fontsize=11, ncol=2, loc='upper right', framealpha=0.95)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot individual orientation profiles
    for i, (ax, angle) in enumerate(zip(orientation_axes, orientations[:len(orientation_axes)])):
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            if angle in profiles:
                data = profiles[angle]
                ax.plot(data['distances'], data['profile'],
                       color=colors[sim_idx],
                       linewidth=controls.get('line_width', 2.5),
                       alpha=0.8,
                       label=simulations[sim_idx]['params']['defect_type'] if i == 0 else None)
        
        ax.set_xlabel("Distance (nm)", fontsize=12)
        ax.set_ylabel("Stress (GPa)", fontsize=12)
        ax.set_title(f"{angle}° Profile", fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(fontsize=10)
    
    # Statistical summary panel
    if all_profiles:
        stats_data = []
        
        for sim_idx, (sim_id, profiles) in enumerate(all_profiles.items()):
            for angle, data in profiles.items():
                metadata = data['metadata']
                stats_data.append({
                    'Simulation': simulations[sim_idx]['params']['defect_type'],
                    'Angle': f'{angle}°',
                    'Max (GPa)': metadata['max_value'],
                    'Mean (GPa)': metadata['mean_value'],
                    'FWHM (nm)': metadata['fwhm_nm'],
                    'Peak Pos (nm)': metadata['peak_position_nm']
                })
        
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
            
            ax_stats.set_xlabel("Simulation", fontsize=12)
            ax_stats.set_ylabel("Maximum Stress (GPa)", fontsize=12)
            ax_stats.set_title("Peak Stress by Orientation", 
                             fontsize=14, fontweight='bold', pad=10)
            ax_stats.set_xticks(x)
            ax_stats.set_xticklabels(unique_sims, rotation=45, ha='right')
            ax_stats.legend(fontsize=10)
            ax_stats.grid(True, alpha=0.3, axis='y')
    
    # Domain visualization panel
    if simulations:
        sim = simulations[0]
        eta, stress_fields = sim['history'][frames[0]]
        
        ax_domain.set_aspect('equal')
        
        # Choose appropriate colormap
        if stress_key == 'sigma_hydro':
            cmap = plt.cm.get_cmap('coolwarm')
        elif stress_key == 'von_mises':
            cmap = plt.cm.get_cmap('plasma')
        else:
            cmap = plt.cm.get_cmap('viridis')
        
        im = ax_domain.imshow(stress_fields[stress_key], extent=extent, 
                            cmap=cmap, origin='lower', alpha=0.85)
        
        # Add probe lines
        line_colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i, angle in enumerate(orientations[:len(line_colors)]):
            if angle in all_profiles[next(iter(all_profiles.keys()))]:
                endpoints = all_profiles[next(iter(all_profiles.keys()))][angle]['endpoints']
                x_start, y_start, x_end, y_end = endpoints
                ax_domain.plot([x_start, x_end], [y_start, y_end],
                             color=line_colors[i], linewidth=3, alpha=0.9,
                             label=f'{angle}°')
        
        ax_domain.set_xlabel("x (nm)", fontsize=12)
        ax_domain.set_ylabel("y (nm)", fontsize=12)
        ax_domain.set_title("Stress Field with Probe Lines", 
                          fontsize=14, fontweight='bold', pad=15)
        ax_domain.legend(fontsize=10)
        
        # Add scale bar
        PublicationVisualizer._add_scale_bar(ax_domain, 5.0, location='lower right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_domain, shrink=0.8)
        cbar.set_label(f'{stress_component} (GPa)', fontsize=12)
    
    # Apply publication styling
    style_params = PublicationVisualizer.create_enhanced_styling_params()
    style_params['lines']['linewidth'] = controls.get('line_width', 2.5)
    style_params['figure']['dpi'] = controls.get('dpi', 600)
    
    all_axes = [ax_main, ax_stats, ax_domain] + orientation_axes
    fig = PublicationVisualizer.apply_publication_style(fig, all_axes, style_params)
    
    # Add panel labels
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for ax, label in zip(all_axes[:len(panel_labels)], panel_labels):
        if ax is not None:
            ax.text(-0.08, 1.08, label, transform=ax.transAxes,
                   fontsize=20, fontweight='bold', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='white', alpha=0.9,
                            edgecolor='black', linewidth=1.5))
    
    return fig

# Add missing scale bar method to PublicationVisualizer
PublicationVisualizer._add_scale_bar = lambda ax, length_nm, location='lower right', color='white': \
    ax.plot([0, length_nm], [0, 0], color=color, linewidth=4)

# =============================================
# SIDEBAR CONFIGURATION
# =============================================
st.sidebar.header("🎨 Global Chart Styling")

# Get enhanced publication controls
advanced_styling = EnhancedFigureStyler.get_publication_controls()

# Color maps selection
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
    ["Run New Simulation", "Compare Saved Simulations", "Publication-Quality Analysis"],
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
    
    # Visualization settings
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
            ["Side-by-Side Heatmaps", "Overlay Line Profiles", "Radial Profile Comparison", 
             "Statistical Summary", "Defect-Stress Correlation", "Stress Component Cross-Correlation",
             "Evolution Timeline", "Contour Comparison", "3D Surface Comparison"],
            index=0
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

else:  # Publication-Quality Analysis
    st.sidebar.header("📰 Publication-Quality Analysis")
    
    # Get enhanced visualization controls
    viz_controls = EnhancedVisualizationControls.get_visualization_controls()
    
    # Get available simulations
    simulations = SimulationDB.get_simulation_list()
    
    if not simulations:
        st.sidebar.warning("No simulations saved yet. Run some simulations first!")
    else:
        # Simulation selection
        sim_options = {f"{sim['name']} (ID: {sim['id']})": sim['id'] for sim in simulations}
        selected_sim_ids = st.sidebar.multiselect(
            "Select Simulations for Analysis",
            options=list(sim_options.keys()),
            default=list(sim_options.keys())[:min(3, len(sim_options))]
        )
        
        # Convert back to IDs
        selected_ids = [sim_options[name] for name in selected_sim_ids]
        
        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        
        analysis_component = st.sidebar.selectbox(
            "Stress Component for Analysis",
            ["Hydrostatic σ_h", "von Mises σ_vM", "Stress Magnitude |σ|"],
            index=0
        )
        
        frame_selection = st.sidebar.radio(
            "Frame Selection",
            ["Final Frame", "Specific Frame Index"],
            horizontal=True
        )
        
        if frame_selection == "Specific Frame Index":
            frame_idx = st.sidebar.slider("Analysis Frame Index", 0, 100, 50)
        else:
            frame_idx = None
        
        # Run analysis
        if st.sidebar.button("📊 Generate Publication Analysis", type="primary"):
            st.session_state.run_publication_analysis = True
            st.session_state.publication_config = {
                'sim_ids': selected_ids,
                'stress_component': analysis_component,
                'frame_selection': frame_selection,
                'frame_idx': frame_idx,
                'viz_controls': viz_controls
            }

# =============================================
# MAIN CONTENT AREA
# =============================================
if operation_mode == "Run New Simulation":
    # Show simulation preview
    st.markdown('<div class="section-header">🎯 New Simulation Preview</div>', 
                unsafe_allow_html=True)
    
    if 'sim_params' in st.session_state:
        sim_params = st.session_state.sim_params
        
        # Display simulation parameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">' + sim_params['defect_type'] + '</div><div class="metric-label">Defect Type</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sim_params["eps0"]:.3f}</div><div class="metric-label">ε*</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sim_params["kappa"]:.2f}</div><div class="metric-label">κ</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sim_params["orientation"]}</div><div class="metric-label">Orientation</div></div>', unsafe_allow_html=True)
        
        # Show initial configuration
        init_eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Apply styling
        fig = EnhancedFigureStyler.apply_advanced_styling(fig, [ax1, ax2], advanced_styling)
        
        # Initial defect
        im1 = ax1.imshow(init_eta, extent=extent, 
                        cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['eta_cmap'], 'viridis')), 
                        origin='lower')
        ax1.set_title(f"Initial {sim_params['defect_type']} - {sim_params['shape']}")
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        plt.colorbar(im1, ax=ax1, shrink=advanced_styling.get('colorbar_shrink', 0.8))
        
        # Stress preview (calculated from initial state)
        stress_preview = compute_stress_fields(init_eta, sim_params['eps0'], sim_params['theta'])
        im2 = ax2.imshow(stress_preview['sigma_mag'], extent=extent, 
                        cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['sigma_cmap'], 'hot')), 
                        origin='lower')
        ax2.set_title(f"Initial Stress Magnitude")
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        plt.colorbar(im2, ax=ax2, shrink=advanced_styling.get('colorbar_shrink', 0.8))
        
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
                
                # Show final frame with post-processing options
                with st.expander("📊 Post-Process Final Results", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        show_defect = st.checkbox("Show Defect Field", True)
                        show_stress = st.checkbox("Show Stress Field", True)
                    with col2:
                        custom_cmap = st.selectbox("Custom Colormap", cmap_list, 
                                                  index=cmap_list.index('viridis'))
                    
                    if show_defect or show_stress:
                        final_eta, final_stress = history[-1]
                        
                        n_plots = (1 if show_defect else 0) + (1 if show_stress else 0)
                        fig2, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
                        
                        if n_plots == 1:
                            axes = [axes]
                        
                        plot_idx = 0
                        if show_defect:
                            im = axes[plot_idx].imshow(final_eta, extent=extent, 
                                                      cmap=plt.cm.get_cmap(COLORMAPS.get(custom_cmap, 'viridis')), 
                                                      origin='lower')
                            axes[plot_idx].set_title(f"Final {sim_params['defect_type']}")
                            axes[plot_idx].set_xlabel("x (nm)")
                            axes[plot_idx].set_ylabel("y (nm)")
                            plt.colorbar(im, ax=axes[plot_idx], shrink=0.8)
                            plot_idx += 1
                        
                        if show_stress:
                            im = axes[plot_idx].imshow(final_stress['sigma_mag'], extent=extent,
                                                      cmap=plt.cm.get_cmap(COLORMAPS.get(custom_cmap, 'viridis')), 
                                                      origin='lower')
                            axes[plot_idx].set_title(f"Final Stress Magnitude")
                            axes[plot_idx].set_xlabel("x (nm)")
                            axes[plot_idx].set_ylabel("y (nm)")
                            plt.colorbar(im, ax=axes[plot_idx], shrink=0.8)
                        
                        # Apply advanced styling
                        fig2 = EnhancedFigureStyler.apply_advanced_styling(fig2, axes, advanced_styling)
                        st.pyplot(fig2)
                
                # Clear the run flag
                if 'run_new_simulation' in st.session_state:
                    del st.session_state.run_new_simulation
    
    else:
        st.info("Configure simulation parameters in the sidebar and click 'Run & Save Simulation'")
    
    # Show saved simulations
    st.markdown('<div class="section-header">📋 Saved Simulations</div>', 
                unsafe_allow_html=True)
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
                'Steps': params['steps'],
                'Frames': len(SimulationDB.get_simulation(sim['id'])['history'])
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

elif operation_mode == "Compare Saved Simulations":
    st.markdown('<div class="section-header">🔬 Multi-Simulation Comparison</div>', 
                unsafe_allow_html=True)
    
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
            
            # Create comparison based on type
            if config['type'] == "Side-by-Side Heatmaps":
                # Use enhanced publication-quality plotting
                st.subheader(f"📰 Publication-Quality {config['type']}")
                
                # Create enhanced plot
                fig = create_publication_heatmaps(simulations, frames, config, advanced_styling)
                
                # Display with enhanced options
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.pyplot(fig)
                
                with col2:
                    # Quick export info
                    st.info(f"""
                    **Publication Ready:**
                    - Journal: {advanced_styling.get('journal_style', 'custom').title()}
                    - DPI: {advanced_styling.get('figure_dpi', 600)}
                    - Vector: {'Yes' if advanced_styling.get('vector_output', True) else 'No'}
                    """)
                
                with col3:
                    # Show figure info
                    fig_size = fig.get_size_inches()
                    st.metric("Figure Size", f"{fig_size[0]:.1f} × {fig_size[1]:.1f} in")
                    st.metric("Resolution", f"{advanced_styling.get('figure_dpi', 600)} DPI")
            
            elif config['type'] == "Overlay Line Profiles":
                # Enhanced overlay plot using PublicationProfileExtractor
                st.subheader("📈 Enhanced Line Profile Comparison")
                
                # Use PublicationProfileExtractor for better profiles
                analyzer = PublicationProfileExtractor()
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    eta, stress_fields = sim['history'][frame]
                    stress_key = {
                        "Stress Magnitude |σ|": 'sigma_mag',
                        "Hydrostatic σ_h": 'sigma_hydro',
                        "von Mises σ_vM": 'von_mises'
                    }[config['stress_component']]
                    
                    stress_data = stress_fields[stress_key]
                    
                    # Extract horizontal profile
                    distances, profile, endpoints, metadata = analyzer.extract_profile_2d(
                        stress_data, 0, 'center', 0, 0.7, 4, 3
                    )
                    
                    label = f"{sim['params']['defect_type']} - {sim['params']['orientation']}"
                    ax1.plot(distances, profile, color=color, 
                           linewidth=advanced_styling.get('line_width', 2.0),
                           label=label)
                    
                    # Extract vertical profile
                    distances, profile, endpoints, metadata = analyzer.extract_profile_2d(
                        stress_data, 90, 'center', 0, 0.7, 4, 3
                    )
                    
                    ax2.plot(distances, profile, color=color,
                           linewidth=advanced_styling.get('line_width', 2.0),
                           label=label)
                
                ax1.set_xlabel("Horizontal Distance (nm)")
                ax1.set_ylabel("Stress (GPa)")
                ax1.set_title("Horizontal Profile")
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                
                ax2.set_xlabel("Vertical Distance (nm)")
                ax2.set_ylabel("Stress (GPa)")
                ax2.set_title("Vertical Profile")
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Apply advanced styling
                fig = EnhancedFigureStyler.apply_advanced_styling(fig, [ax1, ax2], advanced_styling)
                st.pyplot(fig)
            
            else:
                # For other comparison types, use simpler visualization
                st.info(f"Comparison type '{config['type']}' selected. Using basic visualization.")
                
                # Basic visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    eta, stress_fields = sim['history'][frame]
                    stress_key = {
                        "Stress Magnitude |σ|": 'sigma_mag',
                        "Hydrostatic σ_h": 'sigma_hydro',
                        "von Mises σ_vM": 'von_mises'
                    }[config['stress_component']]
                    
                    stress_data = stress_fields[stress_key]
                    
                    # Simple line plot of mean stress
                    mean_stress = np.mean(stress_data)
                    ax.bar(idx, mean_stress, color=color, alpha=0.7, 
                           label=f"{sim['params']['defect_type']}")
                
                ax.set_xlabel("Simulation")
                ax.set_ylabel(f"Mean {config['stress_component']} (GPa)")
                ax.set_title(f"{config['type']} Comparison")
                ax.legend()
                
                # Apply advanced styling
                fig = EnhancedFigureStyler.apply_advanced_styling(fig, ax, advanced_styling)
                st.pyplot(fig)
            
            # Clear comparison flag
            if 'run_comparison' in st.session_state:
                del st.session_state.run_comparison
    
    else:
        st.info("Select simulations in the sidebar and click 'Run Comparison' to start!")

else:  # Publication-Quality Analysis
    st.markdown('<div class="section-header">📰 Publication-Quality Stress Analysis</div>', 
                unsafe_allow_html=True)
    
    if 'run_publication_analysis' in st.session_state and st.session_state.run_publication_analysis:
        config = st.session_state.publication_config
        
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
            st.error("No valid simulations selected for analysis!")
        else:
            st.success(f"Loaded {len(simulations)} simulations for publication-quality analysis")
            
            # Determine frame index
            if config['frame_selection'] == "Final Frame":
                # Use final frame for each simulation
                frames = [len(sim['history']) - 1 for sim in simulations]
            else:
                # Specific frame index
                frames = [min(config['frame_idx'], len(sim['history']) - 1) for sim in simulations]
            
            # Create enhanced configuration
            enhanced_config = {
                'stress_component': config['stress_component'],
                'line_style': 'solid'
            }
            
            # Generate publication-quality visualization
            with st.spinner("Generating publication-quality analysis..."):
                fig = create_multi_orientation_analysis(
                    simulations, frames, enhanced_config, config['viz_controls']
                )
                
                # Display the figure
                st.pyplot(fig)
                
                # Export options
                with st.expander("💾 Export Publication Figure", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'PNG' in config['viz_controls']['export_format']:
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=config['viz_controls']['dpi'], 
                                       bbox_inches='tight', facecolor='white')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📸 Download PNG",
                                data=buf,
                                file_name=f"publication_analysis_{config['stress_component'].replace(' ', '_')}.png",
                                mime="image/png"
                            )
                    
                    with col2:
                        if 'PDF' in config['viz_controls']['export_format']:
                            buf = BytesIO()
                            fig.savefig(buf, format='pdf', dpi=config['viz_controls']['dpi'],
                                       bbox_inches='tight', facecolor='white')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📄 Download PDF",
                                data=buf,
                                file_name=f"publication_analysis_{config['stress_component'].replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )
                    
                    with col3:
                        # Export analysis data
                        analysis_data = []
                        analyzer = PublicationProfileExtractor()
                        
                        for sim, frame in zip(simulations, frames):
                            eta, stress_fields = sim['history'][frame]
                            stress_key = {
                                "Hydrostatic σ_h": 'sigma_hydro',
                                "von Mises σ_vM": 'von_mises',
                                "Stress Magnitude |σ|": 'sigma_mag'
                            }[config['stress_component']]
                            
                            stress_data = stress_fields[stress_key]
                            
                            orientation_results = {}
                            for angle in config['viz_controls'].get('hydrostatic_angles', [45, 135]):
                                distances, profile, endpoints, metadata = analyzer.extract_profile_2d(
                                    stress_data, angle, 'center', 0, 0.7
                                )
                                
                                orientation_results[f"{angle}°"] = {
                                    'distances': distances.tolist(),
                                    'profile': profile.tolist(),
                                    'metadata': metadata
                                }
                            
                            analysis_data.append({
                                'simulation_id': sim['id'],
                                'defect_type': sim['params']['defect_type'],
                                'orientation': sim['params']['orientation'],
                                'stress_component': config['stress_component'],
                                'profiles': orientation_results
                            })
                        
                        json_data = json.dumps(analysis_data, indent=2)
                        st.download_button(
                            label="📊 Export Analysis Data",
                            data=json_data,
                            file_name="analysis_data.json",
                            mime="application/json"
                        )
            
            # Clear analysis flag
            if 'run_publication_analysis' in st.session_state:
                del st.session_state.run_publication_analysis
    
    else:
        st.info("Configure publication-quality analysis settings in the sidebar and click 'Generate Publication Analysis'")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Soundness & Advanced Analysis", expanded=False):
    st.markdown("""
    ### 🎯 **Enhanced Multi-Simulation Comparison Platform**
    
    #### **📊 Advanced Publication-Quality Features:**
    
    **1. Multi-Orientation Stress Analysis:**
    - **Hydrostatic Stress (σ_h)**: Optimal analysis along 45° and 135° diagonals
    - **von Mises Stress (σ_vM)**: Best visualized with 0° and 90° probes
    - **Stress Magnitude (|σ|)**: Comprehensive multi-orientation analysis
    
    **2. Publication-Ready Visualization:**
    - **Multi-panel Figures**: Professional (A, B, C, D) labeling
    - **Enhanced Colormaps**: Science-optimized palettes for different stress components
    - **Scale Bars & Annotations**: Essential for microscopy-style images
    - **Statistical Overlays**: Comprehensive statistical analysis integrated
    
    **3. Advanced Profile Extraction:**
    - **High-Resolution Sampling**: 4x oversampling for smooth curves
    - **Cubic Spline Interpolation**: Publication-quality accuracy
    - **Comprehensive Metadata**: FWHM, peak positions, symmetry indices
    - **Boundary-Aware Extraction**: Intelligent handling of domain edges
    
    #### **🔬 Scientific Insights from Enhanced Analysis:**
    
    **Hydrostatic Stress Analysis:**
    - **Diagonal Sensitivity**: Maximum stress gradients along 45°/135° directions
    - **Dilatational Components**: Captures volumetric strain effects
    - **Crystal Symmetry**: Reveals FCC symmetry in stress distribution
    - **Publication Standard**: Coolwarm colormap recommended
    
    **von Mises Stress Analysis:**
    - **Shear Components**: Best visualized along principal axes
    - **Yield Criteria**: Direct relevance to material failure
    - **Anisotropy Detection**: Horizontal vs vertical stress differences
    - **Publication Standard**: Plasma colormap recommended
    
    **Stress Magnitude Analysis:**
    - **Comprehensive Coverage**: All orientations (0°, 45°, 90°, 135°)
    - **Failure Analysis**: Essential for fracture mechanics
    - **Multi-scale Analysis**: From atomic defects to continuum
    - **Publication Standard**: Viridis colormap recommended
    """)

# =============================================
# EXPORT FUNCTIONALITY
# =============================================
st.sidebar.header("💾 Export Options")

with st.sidebar.expander("📥 Advanced Export"):
    export_format = st.selectbox(
        "Export Format",
        ["Complete Package (JSON + CSV + PNG)", "JSON Parameters Only", 
         "Publication-Ready Figures", "Raw Data CSV"]
    )
    
    include_styling = st.checkbox("Include Styling Parameters", True)
    high_resolution = st.checkbox("High Resolution Figures", True)
    
    if st.button("📥 Generate Custom Export", type="primary"):
        simulations = SimulationDB.get_all_simulations()
        
        if not simulations:
            st.sidebar.warning("No simulations to export!")
        else:
            with st.spinner("Creating custom export package..."):
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Export each simulation
                    for sim_id, sim_data in simulations.items():
                        sim_dir = f"simulation_{sim_id}"
                        
                        # Export parameters
                        params_json = json.dumps(sim_data['params'], indent=2)
                        zf.writestr(f"{sim_dir}/parameters.json", params_json)
                        
                        # Export metadata
                        metadata_json = json.dumps(sim_data['metadata'], indent=2)
                        zf.writestr(f"{sim_dir}/metadata.json", metadata_json)
                        
                        # Export styling if requested
                        if include_styling:
                            styling_json = json.dumps(advanced_styling, indent=2)
                            zf.writestr(f"{sim_dir}/styling_parameters.json", styling_json)
                        
                        # Export data frames
                        if export_format in ["Complete Package (JSON + CSV + PNG)", "Raw Data CSV"]:
                            for i, (eta, stress_fields) in enumerate(sim_data['history']):
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
                    summary = f"""MULTI-SIMULATION EXPORT SUMMARY
========================================
Generated: {datetime.now().isoformat()}
Total Simulations: {len(simulations)}
Export Format: {export_format}
Includes Styling: {include_styling}
High Resolution: {high_resolution}

SIMULATIONS:
------------
"""
                    for sim_id, sim_data in simulations.items():
                        params = sim_data['params']
                        summary += f"\nSimulation {sim_id}:"
                        summary += f"\n  Defect: {params['defect_type']}"
                        summary += f"\n  Orientation: {params['orientation']}"
                        summary += f"\n  ε*: {params['eps0']}"
                        summary += f"\n  κ: {params['kappa']}"
                        summary += f"\n  Frames: {len(sim_data['history'])}"
                        summary += f"\n  Created: {sim_data['created_at']}\n"
                    
                    zf.writestr("EXPORT_SUMMARY.txt", summary)
                
                buffer.seek(0)
                
                # Determine file name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ag_np_analysis_export_{timestamp}.zip"
                
                st.sidebar.download_button(
                    "📥 Download Export Package",
                    buffer.getvalue(),
                    filename,
                    "application/zip"
                )
                st.sidebar.success("Export package ready!")

# =============================================
# FOOTER
# =============================================
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

st.caption("🔬 Advanced Multi-Defect Comparison • Publication-Quality Output • Journal Templates • 2025")
