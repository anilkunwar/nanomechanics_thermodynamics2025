import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse, Polygon
from matplotlib.collections import LineCollection
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
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Ag NP Multi-Defect Analyzer Pro",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        text-align: center !important;
        padding: 1rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🔬 Ag Nanoparticle Multi-Defect Analysis Platform Pro</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h3>🎯 Advanced Stress Analysis • Multi-Orientation Profiling • Fixed Aspect Ratio • Publication-Ready Output</h3>
    <p><strong>Run Multiple Simulations • Compare ISF/ESF/Twin with Different Orientations • Cloud-style Storage</strong></p>
    <p><strong>Run → Save → Compare • 50+ Colormaps • Enhanced Post-Processing • Scientific Insights</strong></p>
</div>
""", unsafe_allow_html=True)

# =============================================
# MATERIAL & GRID PARAMETERS
# =============================================
a = 0.4086  # FCC Ag lattice constant (nm)
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)

# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

# Grid parameters
N = 128  # Grid size
dx = 0.1  # Grid spacing (nm)
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# ENHANCED COLORMAP LIBRARY (60+ options)
# =============================================
COLORMAPS = {
    # Perceptually uniform sequential
    'viridis': 'viridis',
    'plasma': 'plasma', 
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    
    # Sequential (1)
    'summer': 'summer',
    'autumn': 'autumn',
    'winter': 'winter',
    'spring': 'spring',
    'cool': 'cool',
    'hot': 'hot',
    
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
    
    # Modern scientific
    'rocket': 'rocket',
    'mako': 'mako',
    'crest': 'crest',
    'flare': 'flare',
    'icefire': 'icefire',
    'vlag': 'vlag',
    'dense': 'dense',
    'deep': 'deep',
    'speed': 'speed',
    'phase': 'phase',
    
    # Colorblind friendly
    'colorblind': 'colorblind',
    'tol': 'tol',
    'vik': 'vik',
    'broc': 'broc',
    'cork': 'cork',
    'lisbon': 'lisbon',
    'roma': 'roma'
}

cmap_list = list(COLORMAPS.keys())

# =============================================
# ENHANCED PROFILE EXTRACTION SYSTEM
# =============================================
class AdvancedProfileExtractor:
    """Advanced system for extracting line profiles at arbitrary orientations with high precision"""
    
    @staticmethod
    def extract_profile_2d(data, angle_deg, position='center', offset=0, length_ratio=0.8, 
                          sampling_factor=2, interpolation_order=3):
        """
        Extract high-resolution line profile from 2D data
        
        Parameters:
        -----------
        data : 2D numpy array
            Input data
        angle_deg : float
            Angle in degrees (0° = horizontal, positive = counterclockwise)
        position : str or tuple
            'center', 'offset', or (x, y) coordinates
        offset : float
            Offset from center in nm (perpendicular to profile)
        length_ratio : float
            Length as fraction of domain size (0-1)
        sampling_factor : int
            Oversampling factor (default 2x)
        interpolation_order : int
            Interpolation order (1=linear, 3=cubic)
        
        Returns:
        --------
        distances : 1D array
            Distance along profile (nm)
        profile : 1D array
            Extracted values
        endpoints : tuple
            (x_start, y_start, x_end, y_end) in physical coordinates
        metadata : dict
            Extraction parameters and statistics
        """
        N = data.shape[0]
        angle_rad = np.deg2rad(angle_deg)
        
        # Determine center point
        if position == 'center':
            x_center, y_center = 0, 0
        elif position == 'offset':
            perp_angle = angle_rad + np.pi/2
            x_center = offset * np.cos(perp_angle)
            y_center = offset * np.sin(perp_angle)
        elif isinstance(position, (tuple, list)) and len(position) == 2:
            x_center, y_center = position
        else:
            x_center, y_center = 0, 0
        
        # Calculate profile length
        domain_size = extent[1] - extent[0]
        half_length = domain_size * length_ratio / 2
        
        # Calculate endpoints
        x_start = x_center - half_length * np.cos(angle_rad)
        y_start = y_center - half_length * np.sin(angle_rad)
        x_end = x_center + half_length * np.cos(angle_rad)
        y_end = y_center + half_length * np.sin(angle_rad)
        
        # Generate high-resolution sampling points
        num_points = int(N * length_ratio * sampling_factor)
        distances = np.linspace(-half_length, half_length, num_points)
        xs = x_center + distances * np.cos(angle_rad)
        ys = y_center + distances * np.sin(angle_rad)
        
        # Convert to array indices
        xi = (xs - extent[0]) / (extent[1] - extent[0]) * (N - 1)
        yi = (ys - extent[2]) / (extent[3] - extent[2]) * (N - 1)
        
        # Extract profile with interpolation
        profile = map_coordinates(data, [yi, xi], order=interpolation_order, 
                                 mode='constant', cval=0.0)
        
        # Calculate statistics
        metadata = {
            'angle_deg': angle_deg,
            'position': position,
            'offset_nm': offset,
            'length_nm': 2 * half_length,
            'num_points': num_points,
            'sampling_factor': sampling_factor,
            'interpolation_order': interpolation_order,
            'max_value': float(np.nanmax(profile)),
            'min_value': float(np.nanmin(profile)),
            'mean_value': float(np.nanmean(profile)),
            'std_value': float(np.nanstd(profile)),
            'fwhm_nm': AdvancedProfileExtractor.calculate_fwhm(distances, profile)
        }
        
        return distances, profile, (x_start, y_start, x_end, y_end), metadata
    
    @staticmethod
    def calculate_fwhm(distances, profile):
        """Calculate Full Width at Half Maximum"""
        profile_norm = profile - np.nanmin(profile)
        max_val = np.nanmax(profile_norm)
        half_max = max_val / 2
        
        # Find crossings
        above_half = profile_norm > half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            if len(indices) > 1:
                left_idx = indices[0]
                right_idx = indices[-1]
                return float(distances[right_idx] - distances[left_idx])
        return 0.0
    
    @staticmethod
    def extract_multiple_profiles(data, angles, offsets=None, **kwargs):
        """Extract multiple profiles at once"""
        if offsets is None:
            offsets = [0] * len(angles)
        
        profiles = {}
        for angle, offset in zip(angles, offsets):
            key = f"{angle}°_offset{offset}"
            distances, profile, endpoints, metadata = AdvancedProfileExtractor.extract_profile_2d(
                data, angle, 'offset' if offset != 0 else 'center', offset, **kwargs
            )
            profiles[key] = {
                'distances': distances,
                'profile': profile,
                'endpoints': endpoints,
                'metadata': metadata
            }
        return profiles
    
    @staticmethod
    def create_profile_grid(data, n_profiles=5, angle_range=(-90, 90), **kwargs):
        """Create a grid of profiles at evenly spaced angles"""
        angles = np.linspace(angle_range[0], angle_range[1], n_profiles)
        profiles = {}
        
        for angle in angles:
            key = f"{angle:.1f}°"
            distances, profile, endpoints, metadata = AdvancedProfileExtractor.extract_profile_2d(
                data, angle, **kwargs
            )
            profiles[key] = {
                'distances': distances,
                'profile': profile,
                'endpoints': endpoints,
                'metadata': metadata
            }
        
        return profiles

# =============================================
# FIXED ASPECT RATIO VISUALIZATION SYSTEM
# =============================================
class FixedAspectManager:
    """Comprehensive system for maintaining realistic aspect ratios in visualizations"""
    
    @staticmethod
    def get_domain_properties():
        """Get physical properties of simulation domain"""
        width_nm = extent[1] - extent[0]
        height_nm = extent[3] - extent[2]
        aspect_ratio = height_nm / width_nm
        diagonal_nm = np.sqrt(width_nm**2 + height_nm**2)
        
        return {
            'width_nm': width_nm,
            'height_nm': height_nm,
            'aspect_ratio': aspect_ratio,
            'diagonal_nm': diagonal_nm,
            'center': (0, 0),
            'extent': extent
        }
    
    @staticmethod
    def calculate_optimal_figure_size(n_rows=1, n_cols=1, domain_aspect=1.0, 
                                     profile_height_ratio=0.6):
        """
        Calculate optimal figure size for mixed layouts
        
        Parameters:
        -----------
        n_rows, n_cols : int
            Grid dimensions
        domain_aspect : float
            Aspect ratio of domain plots (height/width)
        profile_height_ratio : float
            Height ratio for profile plots relative to domain
        
        Returns:
        --------
        figsize : tuple
            Optimal (width, height) in inches
        """
        base_width = 6.0  # inches per column
        
        if n_rows == 1 and n_cols == 1:
            # Single plot - use domain aspect ratio
            width = base_width
            height = width * domain_aspect
        elif domain_aspect == 1.0:
            # Square domains - simple calculation
            width = base_width * n_cols
            height = width * (n_rows / n_cols)
        else:
            # Mixed aspect ratios
            width = base_width * n_cols
            # Account for different plot types
            domain_height = width / n_cols * domain_aspect
            profile_height = domain_height * profile_height_ratio
            height = max(domain_height, profile_height) * n_rows
        
        return (width, height)
    
    @staticmethod
    def apply_fixed_aspect(ax, data=None, aspect_type='equal', **kwargs):
        """
        Apply fixed aspect ratio to axis
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to modify
        data : 2D array or None
            Data for determining aspect
        aspect_type : str
            'equal' for 1:1, 'auto' for variable, 'custom' for specified ratio
        """
        if aspect_type == 'equal':
            ax.set_aspect('equal')
            # Ensure square domain
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # Adjust limits to make square
            if x_range > y_range:
                center_y = (ylim[0] + ylim[1]) / 2
                ax.set_ylim(center_y - x_range/2, center_y + x_range/2)
            elif y_range > x_range:
                center_x = (xlim[0] + xlim[1]) / 2
                ax.set_xlim(center_x - y_range/2, center_x + y_range/2)
        
        elif aspect_type == 'custom' and 'aspect_ratio' in kwargs:
            ax.set_aspect(kwargs['aspect_ratio'])
        
        elif data is not None:
            # Use data aspect ratio
            height, width = data.shape
            ax.set_aspect('auto')
        
        return ax
    
    @staticmethod
    def add_physical_scale(ax, length_nm=5.0, location='lower right', 
                          color='white', fontsize=10, linewidth=2):
        """
        Add physical scale bar with enhanced styling
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to add scale bar to
        length_nm : float
            Length of scale bar in nm
        location : str
            'lower right', 'lower left', 'upper right', 'upper left'
        """
        # Get axis limits in data coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Determine position based on location
        if location == 'lower right':
            bar_x_start = xlim[1] - x_range * 0.15
            bar_x_end = bar_x_start - length_nm
            bar_y = ylim[0] + y_range * 0.05
            text_y = bar_y + y_range * 0.02
            text_ha = 'center'
            text_va = 'bottom'
        elif location == 'lower left':
            bar_x_start = xlim[0] + x_range * 0.05
            bar_x_end = bar_x_start + length_nm
            bar_y = ylim[0] + y_range * 0.05
            text_y = bar_y + y_range * 0.02
            text_ha = 'center'
            text_va = 'bottom'
        elif location == 'upper right':
            bar_x_start = xlim[1] - x_range * 0.15
            bar_x_end = bar_x_start - length_nm
            bar_y = ylim[1] - y_range * 0.05
            text_y = bar_y - y_range * 0.02
            text_ha = 'center'
            text_va = 'top'
        else:  # upper left
            bar_x_start = xlim[0] + x_range * 0.05
            bar_x_end = bar_x_start + length_nm
            bar_y = ylim[1] - y_range * 0.05
            text_y = bar_y - y_range * 0.02
            text_ha = 'center'
            text_va = 'top'
        
        # Draw scale bar with enhanced styling
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
               color=color, linewidth=linewidth, solid_capstyle='butt',
               zorder=1000)
        
        # Add end caps
        cap_length = y_range * 0.01
        ax.plot([bar_x_start, bar_x_start], [bar_y - cap_length, bar_y + cap_length],
               color=color, linewidth=linewidth, solid_capstyle='butt',
               zorder=1000)
        ax.plot([bar_x_end, bar_x_end], [bar_y - cap_length, bar_y + cap_length],
               color=color, linewidth=linewidth, solid_capstyle='butt',
               zorder=1000)
        
        # Add text with background for better visibility
        text = ax.text((bar_x_start + bar_x_end) / 2, text_y,
                      f'{length_nm} nm', ha=text_ha, va=text_va,
                      color=color, fontsize=fontsize, fontweight='bold',
                      zorder=1001)
        
        # Add text background
        text.set_bbox(dict(boxstyle="round,pad=0.3", 
                          facecolor='black', alpha=0.5,
                          edgecolor='none'))
        
        return ax

# =============================================
# ENHANCED VISUALIZATION SYSTEM
# =============================================
class EnhancedVisualizer:
    """Advanced visualization system with publication-quality output"""
    
    @staticmethod
    def create_multi_panel_figure(n_panels=1, panel_layout='grid', 
                                  domain_panels=None, **kwargs):
        """
        Create multi-panel figure with optimal layout
        
        Parameters:
        -----------
        n_panels : int
            Number of panels
        panel_layout : str
            'grid', 'horizontal', 'vertical', or 'custom'
        domain_panels : list or None
            Indices of panels that show domain (for aspect ratio control)
        
        Returns:
        --------
        fig : matplotlib Figure
        axes : list or array of Axes
        """
        domain_props = FixedAspectManager.get_domain_properties()
        
        if panel_layout == 'grid':
            # Calculate optimal grid
            n_cols = min(3, n_panels)
            n_rows = (n_panels + n_cols - 1) // n_cols
            
            # Calculate figure size
            figsize = FixedAspectManager.calculate_optimal_figure_size(
                n_rows, n_cols, domain_props['aspect_ratio']
            )
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                    constrained_layout=True)
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
        
        elif panel_layout == 'horizontal':
            figsize = (6 * n_panels, 5)
            fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                                    constrained_layout=True)
            if n_panels == 1:
                axes = [axes]
        
        elif panel_layout == 'vertical':
            figsize = (6, 5 * n_panels)
            fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                                    constrained_layout=True)
            if n_panels == 1:
                axes = [axes]
        
        else:  # custom layout
            figsize = kwargs.get('figsize', (12, 8))
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            axes = []
            
            # Define custom grid
            gs = fig.add_gridspec(3, 3, **kwargs.get('gridspec_kw', {}))
            
            # Add subplots
            axes.append(fig.add_subplot(gs[0, :2]))
            axes.append(fig.add_subplot(gs[0, 2]))
            axes.append(fig.add_subplot(gs[1, :2]))
            axes.append(fig.add_subplot(gs[1, 2]))
            axes.append(fig.add_subplot(gs[2, :]))
        
        # Apply fixed aspect to domain panels
        if domain_panels is not None:
            if isinstance(axes, np.ndarray):
                axes_flat = axes.flatten()
            elif isinstance(axes, list):
                axes_flat = axes
            else:
                axes_flat = [axes]
            
            for idx, ax in enumerate(axes_flat):
                if idx in domain_panels and ax is not None:
                    FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
        
        return fig, axes
    
    @staticmethod
    def create_overlay_profiles_plot(simulations, frames, config, style_params):
        """Create enhanced overlay line profiles with fixed aspect domain"""
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map[config.get('stress_component', 'Stress Magnitude |σ|')]
        
        # Parse orientations
        angle_map = {
            "0° (Horizontal)": 0,
            "45° (Diagonal)": 45,
            "90° (Vertical)": 90,
            "135° (Diagonal)": 135,
        }
        
        selected_orientations = config.get('profile_orientations', ["0° (Horizontal)", "90° (Vertical)"])
        angles = []
        for orientation in selected_orientations:
            if orientation == "Custom":
                angles.append(config.get('custom_angle', 30))
            else:
                angles.append(angle_map.get(orientation, 0))
        
        # Create figure with optimal layout
        fig = plt.figure(figsize=(16, 12))
        fig.set_constrained_layout(True)
        
        # Create grid with specific ratios
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1.5], 
                             hspace=0.25, wspace=0.3)
        
        # Define panels
        ax_overlay = fig.add_subplot(gs[0, :])  # All profiles overlay
        ax_horizontal = fig.add_subplot(gs[1, 0])  # Horizontal profiles
        ax_vertical = fig.add_subplot(gs[1, 1])   # Vertical profiles
        ax_diagonal = fig.add_subplot(gs[1, 2])   # Diagonal profiles
        ax_domain = fig.add_subplot(gs[2, :])     # Domain with profiles - FIXED ASPECT
        
        # Get colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(simulations)))
        
        # Extract and plot profiles
        all_profiles_data = {}
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            for angle in angles:
                profile_length = config.get('profile_length', 80) / 100
                result = AdvancedProfileExtractor.extract_profile_2d(
                    stress_data, angle, 'center', 0, profile_length
                )
                distances, profile, endpoints, metadata = result
                
                # Store for statistics
                if angle not in all_profiles_data:
                    all_profiles_data[angle] = []
                all_profiles_data[angle].append({
                    'profile': profile,
                    'metadata': metadata,
                    'sim_idx': sim_idx
                })
                
                # Plot in overlay
                line_style = config.get('line_style', 'solid')
                label = f"{sim['params']['defect_type']} - {angle}°"
                ax_overlay.plot(distances, profile, 
                              color=colors[sim_idx],
                              linestyle=line_style,
                              linewidth=style_params.get('line_width', 2.0),
                              alpha=0.8,
                              label=label)
        
        # Configure overlay panel
        ax_overlay.set_xlabel("Distance from Center (nm)", 
                            fontsize=style_params.get('label_font_size', 12))
        ax_overlay.set_ylabel(f"{config['stress_component']} (GPa)", 
                            fontsize=style_params.get('label_font_size', 12))
        ax_overlay.set_title("Multi-Orientation Stress Profiles", 
                           fontsize=style_params.get('title_font_size', 14),
                           fontweight='bold')
        ax_overlay.legend(fontsize=style_params.get('legend_fontsize', 10), 
                         ncol=2, loc='upper right')
        ax_overlay.grid(True, alpha=0.3, linestyle='--')
        ax_overlay.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot individual orientation panels
        orientation_panels = [(ax_horizontal, 0, "Horizontal (0°)"), 
                             (ax_vertical, 90, "Vertical (90°)"),
                             (ax_diagonal, 45, "Diagonal (45°)")]
        
        for ax, target_angle, title in orientation_panels:
            if target_angle in [a for a in angles if abs(a - target_angle) < 1]:
                for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    result = AdvancedProfileExtractor.extract_profile_2d(
                        stress_data, target_angle, 'center', 0, 0.8
                    )
                    distances, profile, endpoints, metadata = result
                    
                    line_style = config.get('line_style', 'solid')
                    label = f"{sim['params']['defect_type']}" if sim_idx == 0 else None
                    
                    ax.plot(distances, profile, 
                           color=colors[sim_idx],
                           linestyle=line_style,
                           linewidth=style_params.get('line_width', 2.0),
                           alpha=0.8,
                           label=label)
            
            ax.set_xlabel("Distance (nm)", fontsize=10)
            ax.set_ylabel("Stress (GPa)", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            if target_angle == 0:
                ax.legend(fontsize=9)
        
        # Plot domain with profile lines - WITH FIXED ASPECT RATIO
        if simulations:
            sim = simulations[0]
            eta, _ = sim['history'][frames[0]]
            
            # Apply fixed aspect ratio
            FixedAspectManager.apply_fixed_aspect(ax_domain, aspect_type='equal')
            
            # Plot domain
            cmap_name = sim['params'].get('eta_cmap', 'viridis')
            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
            im = ax_domain.imshow(eta, extent=extent, cmap=cmap, 
                                origin='lower', alpha=0.8)
            
            # Add profile lines
            line_colors = ['red', 'blue', 'green', 'purple', 'orange']
            for idx, angle in enumerate(angles):
                profile_length = config.get('profile_length', 80) / 100
                result = AdvancedProfileExtractor.extract_profile_2d(
                    eta, angle, 'center', 0, profile_length
                )
                distances, profile, endpoints, metadata = result
                x_start, y_start, x_end, y_end = endpoints
                
                # Draw profile line
                ax_domain.plot([x_start, x_end], [y_start, y_end], 
                             color=line_colors[idx % len(line_colors)], 
                             linewidth=3, linestyle='-',
                             label=f'{angle}°',
                             alpha=0.9)
                
                # Add angle annotation
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                ax_domain.annotate(f'{angle}°', xy=(mid_x, mid_y),
                                 xytext=(10, 10), textcoords='offset points',
                                 color=line_colors[idx % len(line_colors)],
                                 fontsize=10, fontweight='bold',
                                 arrowprops=dict(arrowstyle='->', 
                                               color=line_colors[idx % len(line_colors)],
                                               alpha=0.7))
            
            ax_domain.set_xlabel("x (nm)", fontsize=11)
            ax_domain.set_ylabel("y (nm)", fontsize=11)
            ax_domain.set_title("Simulation Domain with Profile Lines", 
                              fontsize=12, fontweight='bold')
            ax_domain.legend(fontsize=9, loc='upper right')
            
            # Add scale bar
            FixedAspectManager.add_physical_scale(ax_domain, 5.0, 
                                                location='lower right')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_domain, shrink=0.8, pad=0.02)
            cbar.set_label('Defect Parameter η', fontsize=10)
        
        # Add panel labels
        for ax, label in zip([ax_overlay, ax_horizontal, ax_vertical, 
                            ax_diagonal, ax_domain], 
                           ['A', 'B', 'C', 'D', 'E']):
            ax.text(-0.05, 1.05, label, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='top')
        
        return fig
    
    @staticmethod
    def create_comprehensive_comparison(simulations, frames, config, style_params):
        """Create comprehensive comparison visualization"""
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map[config.get('stress_component', 'Stress Magnitude |σ|')]
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(20, 15))
        fig.set_constrained_layout(True)
        
        # Define complex grid
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.2], 
                             hspace=0.3, wspace=0.3)
        
        # Panel definitions
        ax1 = fig.add_subplot(gs[0, :2])  # Stress heatmap
        ax2 = fig.add_subplot(gs[0, 2:])  # Defect field
        ax3 = fig.add_subplot(gs[1, :])   # Multiple profiles
        ax4 = fig.add_subplot(gs[2, :2])  # Statistical summary
        ax5 = fig.add_subplot(gs[2, 2:])  # Correlation
        ax6 = fig.add_subplot(gs[3, :])   # Domain with all profiles
        
        # Get colors
        colors = plt.cm.Set2(np.linspace(0, 1, len(simulations)))
        
        # Panel 1: Stress heatmap (first simulation)
        if simulations:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            stress_data = stress_fields[stress_key]
            
            FixedAspectManager.apply_fixed_aspect(ax1, aspect_type='equal')
            cmap = plt.cm.get_cmap(COLORMAPS.get(sim['params'].get('sigma_cmap', 'hot'), 'hot'))
            im1 = ax1.imshow(stress_data, extent=extent, cmap=cmap, origin='lower')
            ax1.set_title(f"Stress Distribution: {sim['params']['defect_type']}", 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel("x (nm)", fontsize=10)
            ax1.set_ylabel("y (nm)", fontsize=10)
            plt.colorbar(im1, ax=ax1, shrink=0.8, label='Stress (GPa)')
            FixedAspectManager.add_physical_scale(ax1, 5.0, location='lower right')
        
        # Panel 2: Defect field
        if simulations:
            sim = simulations[0]
            eta, stress_fields = sim['history'][frames[0]]
            
            FixedAspectManager.apply_fixed_aspect(ax2, aspect_type='equal')
            cmap = plt.cm.get_cmap(COLORMAPS.get(sim['params'].get('eta_cmap', 'viridis'), 'viridis'))
            im2 = ax2.imshow(eta, extent=extent, cmap=cmap, origin='lower')
            ax2.set_title(f"Defect Field: {sim['params']['defect_type']}", 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel("x (nm)", fontsize=10)
            ax2.set_ylabel("y (nm)", fontsize=10)
            plt.colorbar(im2, ax=ax2, shrink=0.8, label='Defect Parameter η')
            FixedAspectManager.add_physical_scale(ax2, 5.0, location='lower right')
        
        # Panel 3: Multiple line profiles
        angles = [0, 45, 90, 135]  # Standard orientations
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            for angle in angles:
                result = AdvancedProfileExtractor.extract_profile_2d(
                    stress_data, angle, 'center', 0, 0.8
                )
                distances, profile, endpoints, metadata = result
                
                ax3.plot(distances, profile, 
                        color=colors[sim_idx],
                        linewidth=style_params.get('line_width', 1.5),
                        alpha=0.7,
                        label=f"{sim['params']['defect_type']} - {angle}°" if sim_idx == 0 else None)
        
        ax3.set_xlabel("Distance from Center (nm)", fontsize=11)
        ax3.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=11)
        ax3.set_title("Multi-Orientation Line Profiles", 
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 4: Statistical summary
        if simulations:
            stats_data = []
            for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                eta, stress_fields = sim['history'][frame]
                stress_data = stress_fields[stress_key].flatten()
                stress_data = stress_data[np.isfinite(stress_data)]
                
                stats_data.append({
                    'defect': sim['params']['defect_type'],
                    'max': np.max(stress_data),
                    'mean': np.mean(stress_data),
                    'std': np.std(stress_data),
                    'color': colors[sim_idx]
                })
            
            x_pos = np.arange(len(stats_data))
            max_vals = [d['max'] for d in stats_data]
            mean_vals = [d['mean'] for d in stats_data]
            
            bars_max = ax4.bar(x_pos - 0.2, max_vals, 0.4, 
                              color=[d['color'] for d in stats_data], 
                              alpha=0.7, label='Maximum')
            bars_mean = ax4.bar(x_pos + 0.2, mean_vals, 0.4, 
                               color=[d['color'] for d in stats_data], 
                               alpha=0.5, label='Mean')
            
            ax4.set_xlabel("Simulation", fontsize=10)
            ax4.set_ylabel("Stress (GPa)", fontsize=10)
            ax4.set_title("Stress Statistics", fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([d['defect'] for d in stats_data])
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Panel 5: Correlation plot
        if len(simulations) > 1:
            x_data = []
            y_data = []
            labels = []
            
            for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                eta, stress_fields = sim['history'][frame]
                stress_data = stress_fields[stress_key].flatten()
                eta_data = eta.flatten()
                
                # Sample for clarity
                sample_size = min(1000, len(stress_data))
                indices = np.random.choice(len(stress_data), sample_size, replace=False)
                
                x_data.extend(stress_data[indices])
                y_data.extend(eta_data[indices])
                labels.extend([sim['params']['defect_type']] * sample_size)
            
            scatter = ax5.scatter(x_data, y_data, c=[colors[i] for i, label in enumerate(labels) 
                                                   for _ in range(sample_size // len(simulations))],
                                 alpha=0.5, s=20, edgecolors='none')
            
            ax5.set_xlabel(f"{config['stress_component']} (GPa)", fontsize=10)
            ax5.set_ylabel("Defect Parameter η", fontsize=10)
            ax5.set_title("Stress-Defect Correlation", fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # Panel 6: Domain with all profile lines
        if simulations:
            sim = simulations[0]
            eta, _ = sim['history'][frames[0]]
            
            FixedAspectManager.apply_fixed_aspect(ax6, aspect_type='equal')
            
            # Plot domain
            cmap = plt.cm.get_cmap(COLORMAPS.get(sim['params'].get('eta_cmap', 'plasma'), 'plasma'))
            im6 = ax6.imshow(eta, extent=extent, cmap=cmap, origin='lower', alpha=0.7)
            
            # Add multiple profile lines at different angles
            angles_grid = np.linspace(0, 180, 13)[:-1]  # 12 lines at 15° intervals
            line_colors = plt.cm.rainbow(np.linspace(0, 1, len(angles_grid)))
            
            for idx, angle in enumerate(angles_grid):
                result = AdvancedProfileExtractor.extract_profile_2d(
                    eta, angle, 'center', 0, 0.7
                )
                distances, profile, endpoints, metadata = result
                x_start, y_start, x_end, y_end = endpoints
                
                ax6.plot([x_start, x_end], [y_start, y_end], 
                        color=line_colors[idx], 
                        linewidth=2, linestyle='-',
                        alpha=0.7)
                
                # Add angle label
                if idx % 2 == 0:  # Label every other line
                    mid_x = (x_start + x_end) / 2
                    mid_y = (y_start + y_end) / 2
                    ax6.text(mid_x, mid_y, f'{angle:.0f}°', 
                            color=line_colors[idx], fontsize=8,
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.2", 
                                     facecolor='white', alpha=0.7))
            
            ax6.set_xlabel("x (nm)", fontsize=11)
            ax6.set_ylabel("y (nm)", fontsize=11)
            ax6.set_title("Multi-Orientation Profile Grid", 
                         fontsize=13, fontweight='bold')
            FixedAspectManager.add_physical_scale(ax6, 5.0, location='lower right')
            
            # Add colorbar
            cbar = plt.colorbar(im6, ax=ax6, shrink=0.8, pad=0.02)
            cbar.set_label('Defect Parameter η', fontsize=10)
        
        # Add overall title
        fig.suptitle("Comprehensive Multi-Defect Analysis", 
                    fontsize=16, fontweight='bold', y=1.02)
        
        return fig

# =============================================
# SIMULATION DATABASE SYSTEM
# =============================================
class SimulationDatabase:
    """Enhanced database system for simulation management"""
    
    @staticmethod
    def initialize():
        """Initialize session state for simulations"""
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        if 'simulation_counter' not in st.session_state:
            st.session_state.simulation_counter = 0
        if 'comparison_history' not in st.session_state:
            st.session_state.comparison_history = []
    
    @staticmethod
    def generate_id(params):
        """Generate unique simulation ID"""
        param_str = json.dumps(params, sort_keys=True, default=str)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_obj = hashlib.md5((param_str + timestamp).encode())
        return f"SIM_{hash_obj.hexdigest()[:8]}"
    
    @staticmethod
    def save_simulation(params, history, metadata=None):
        """Save simulation to database"""
        SimulationDatabase.initialize()
        
        sim_id = SimulationDatabase.generate_id(params)
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {
                'created': datetime.now().isoformat(),
                'frames': len(history),
                'grid_size': N,
                'dx': dx,
                'run_time': 0.0
            }
        
        # Store simulation
        st.session_state.simulations[sim_id] = {
            'id': sim_id,
            'params': params,
            'history': history,
            'metadata': metadata,
            'tags': [params['defect_type'], params['orientation']]
        }
        
        st.session_state.simulation_counter += 1
        return sim_id
    
    @staticmethod
    def get_simulation(sim_id):
        """Retrieve simulation by ID"""
        SimulationDatabase.initialize()
        return st.session_state.simulations.get(sim_id)
    
    @staticmethod
    def get_all_simulations():
        """Get all simulations"""
        SimulationDatabase.initialize()
        return st.session_state.simulations
    
    @staticmethod
    def get_simulation_list():
        """Get list of simulations for selection"""
        SimulationDatabase.initialize()
        
        simulations = []
        for sim_id, sim_data in st.session_state.simulations.items():
            params = sim_data['params']
            metadata = sim_data['metadata']
            
            sim_info = {
                'id': sim_id,
                'name': f"{params['defect_type']} - {params['orientation']}",
                'params': params,
                'metadata': metadata,
                'display_name': f"{params['defect_type']} ({params['orientation']}) - ε*={params['eps0']:.2f}, κ={params['kappa']:.2f}"
            }
            simulations.append(sim_info)
        
        return simulations
    
    @staticmethod
    def delete_simulation(sim_id):
        """Delete simulation from database"""
        SimulationDatabase.initialize()
        if sim_id in st.session_state.simulations:
            del st.session_state.simulations[sim_id]
            return True
        return False
    
    @staticmethod
    def add_comparison(comparison_data):
        """Save comparison to history"""
        SimulationDatabase.initialize()
        comparison_data['timestamp'] = datetime.now().isoformat()
        st.session_state.comparison_history.append(comparison_data)
        
        # Keep only last 10 comparisons
        if len(st.session_state.comparison_history) > 10:
            st.session_state.comparison_history = st.session_state.comparison_history[-10:]

# =============================================
# SIMULATION ENGINE
# =============================================
@st.cache_data
def create_initial_eta(shape, defect_type, random_seed=42):
    """Create initial defect configuration with enhanced options"""
    np.random.seed(random_seed)
    
    # Set initial amplitude based on defect type
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes.get(defect_type, 0.75)
    
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    
    # Enhanced shape definitions
    if shape == "Square":
        size = 20
        eta[cy-size:cy+size, cx-size:cx+size] = init_amplitude
    elif shape == "Horizontal Fault":
        width, height = 40, 8
        eta[cy-height:cy+height, cx-width:cx+width] = init_amplitude
    elif shape == "Vertical Fault":
        width, height = 8, 40
        eta[cy-height:cy+height, cx-width:cx+width] = init_amplitude
    elif shape == "Rectangle":
        width, height = 30, 15
        eta[cy-height:cy+height, cx-width:cx+width] = init_amplitude
    elif shape == "Ellipse":
        a, b = 25, 15  # semi-major and semi-minor axes
        for i in range(N):
            for j in range(N):
                x = (i - cx) * dx
                y = (j - cy) * dx
                if (x/a)**2 + (y/b)**2 <= 1:
                    eta[i, j] = init_amplitude
    elif shape == "Circle":
        radius = 20
        for i in range(N):
            for j in range(N):
                x = (i - cx) * dx
                y = (j - cy) * dx
                if np.sqrt(x**2 + y**2) <= radius:
                    eta[i, j] = init_amplitude
    
    # Add controlled noise
    noise_level = 0.02
    eta += noise_level * np.random.randn(N, N)
    
    return np.clip(eta, 0.0, 1.0)

@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, kappa, dt=0.004, dx=dx, N=N):
    """Phase field evolution with Allen-Cahn equation (optimized)"""
    eta_new = eta.copy()
    dx2 = dx * dx
    prefactor = dt / dx2
    
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            # Laplacian using finite differences
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j])
            
            # Double well potential derivative
            eta_val = eta[i,j]
            dF = 2 * eta_val * (1 - eta_val) * (eta_val - 0.5)
            
            # Update
            eta_new[i,j] = eta_val + dt * (-dF + kappa * lap / dx2)
            
            # Clamp to [0, 1]
            if eta_new[i,j] < 0.0:
                eta_new[i,j] = 0.0
            elif eta_new[i,j] > 1.0:
                eta_new[i,j] = 1.0
    
    # Apply periodic boundary conditions
    eta_new[0,:] = eta_new[-2,:]
    eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]
    eta_new[:,-1] = eta_new[:,1]
    
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
    
    # Normal vectors
    n1 = np.zeros_like(KX)
    n2 = np.zeros_like(KX)
    n1[mask] = KX[mask] / np.sqrt(K2[mask])
    n2[mask] = KY[mask] / np.sqrt(K2[mask])
    
    # Acoustic tensor
    A11 = np.zeros_like(KX)
    A22 = np.zeros_like(KX)
    A12 = np.zeros_like(KX)
    A11[mask] = C11_p * n1[mask]**2 + C44_p * n2[mask]**2
    A22[mask] = C44_p * n1[mask]**2 + C11_p * n2[mask]**2
    A12[mask] = (C12_p + C44_p) * n1[mask] * n2[mask]
    
    det = A11 * A22 - A12**2
    G11 = np.zeros_like(KX)
    G22 = np.zeros_like(KX)
    G12 = np.zeros_like(KX)
    G11[mask] = A22[mask] / det[mask]
    G22[mask] = A11[mask] / det[mask]
    G12[mask] = -A12[mask] / det[mask]
    
    # Rotated eigenstrain
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
    
    # Polarization stress
    tau_xx = C11_p * eps_xx_star + C12_p * eps_yy_star
    tau_yy = C12_p * eps_xx_star + C11_p * eps_yy_star
    tau_xy = 2 * C44_p * eps_xy_star
    
    # Fourier transforms
    tau_hat_xx = np.fft.fft2(tau_xx)
    tau_hat_yy = np.fft.fft2(tau_yy)
    tau_hat_xy = np.fft.fft2(tau_xy)
    
    S_hat_x = KX * tau_hat_xx + KY * tau_hat_xy
    S_hat_y = KX * tau_hat_xy + KY * tau_hat_yy
    
    # Displacement in Fourier space
    u_hat_x = np.zeros_like(KX, dtype=complex)
    u_hat_y = np.zeros_like(KX, dtype=complex)
    u_hat_x[mask] = -1j * (G11[mask] * S_hat_x[mask] + G12[mask] * S_hat_y[mask])
    u_hat_y[mask] = -1j * (G12[mask] * S_hat_x[mask] + G22[mask] * S_hat_y[mask])
    u_hat_x[0, 0] = 0
    u_hat_y[0, 0] = 0
    
    # Inverse FFT for displacements
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
    
    # Strains
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
    
    # Stresses (Pa → GPa)
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy)  # Plane strain approximation
    
    # Derived quantities
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
    
    return {
        'sxx': sxx, 'syy': syy, 'sxy': sxy, 'szz': szz,
        'sigma_mag': sigma_mag, 'sigma_hydro': sigma_hydro, 'von_mises': von_mises,
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'ux': ux, 'uy': uy
    }

def run_simulation(params, progress_callback=None):
    """Run complete simulation with progress tracking"""
    # Extract parameters
    defect_type = params['defect_type']
    shape = params['shape']
    eps0 = params['eps0']
    kappa = params['kappa']
    theta = params['theta']
    steps = params.get('steps', 100)
    save_every = params.get('save_every', 20)
    
    # Create initial condition
    eta = create_initial_eta(shape, defect_type)
    
    # Run evolution
    history = []
    start_time = time.time()
    
    for step in range(steps + 1):
        if step > 0:
            eta = evolve_phase_field(eta, kappa)
        
        if step % save_every == 0 or step == steps:
            # Compute stress fields
            stress_fields = compute_stress_fields(eta, eps0, theta)
            history.append((eta.copy(), stress_fields))
            
            # Update progress
            if progress_callback:
                progress = (step + 1) / (steps + 1)
                progress_callback(progress)
    
    run_time = time.time() - start_time
    
    # Create metadata
    metadata = {
        'run_time': run_time,
        'frames': len(history),
        'grid_size': N,
        'dx': dx,
        'steps': steps,
        'save_every': save_every,
        'created': datetime.now().isoformat()
    }
    
    return history, metadata

# =============================================
# SIDEBAR CONFIGURATION
# =============================================
st.sidebar.header("⚙️ Platform Configuration")

# Operation mode
operation_mode = st.sidebar.radio(
    "Select Operation Mode",
    ["🏃 Run New Simulation", "🔍 Compare Simulations", "📊 Analysis Dashboard", "💾 Export Data"],
    index=0
)

# Initialize database
SimulationDatabase.initialize()

if "Run New Simulation" in operation_mode:
    st.sidebar.header("🎯 New Simulation Parameters")
    
    # Simulation name
    sim_name = st.sidebar.text_input("Simulation Name", "My Simulation")
    
    # Defect type
    defect_type = st.sidebar.selectbox(
        "Defect Type",
        ["ISF", "ESF", "Twin"],
        help="Intrinsic Stacking Fault, Extrinsic Stacking Fault, or Twin Boundary"
    )
    
    # Shape selection
    shape = st.sidebar.selectbox(
        "Initial Seed Shape",
        ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse", "Circle"],
        help="Initial geometry of the defect"
    )
    
    # Physical parameters with enhanced tooltips
    col1, col2 = st.sidebar.columns(2)
    with col1:
        eps0 = st.slider(
            "Eigenstrain ε*",
            0.3, 3.0,
            value=0.707 if defect_type == "ISF" else 1.414 if defect_type == "ESF" else 2.121,
            step=0.01,
            help="Magnitude of transformation eigenstrain"
        )
    with col2:
        kappa = st.slider(
            "Interface Energy κ",
            0.1, 2.0,
            value=0.6 if defect_type == "ISF" else 0.7 if defect_type == "ESF" else 0.3,
            step=0.05,
            help="Gradient energy coefficient"
        )
    
    # Evolution parameters
    st.sidebar.subheader("Evolution Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        steps = st.slider("Total Steps", 50, 500, 100, 10)
    with col2:
        save_every = st.slider("Save Every", 5, 50, 20, 5)
    
    # Crystal orientation
    st.sidebar.subheader("Crystal Orientation")
    orientation = st.sidebar.selectbox(
        "Habit Plane Orientation",
        [
            "Horizontal {111} (0°)",
            "Tilted 30° (1¯10 projection)",
            "Tilted 45°",
            "Tilted 60°",
            "Vertical {111} (90°)",
            "Custom Angle"
        ],
        index=0
    )
    
    if orientation == "Custom Angle":
        custom_angle = st.sidebar.slider("Custom Angle (°)", -180, 180, 0, 5)
        theta = np.deg2rad(custom_angle)
    else:
        angle_map = {
            "Horizontal {111} (0°)": 0,
            "Tilted 30° (1¯10 projection)": 30,
            "Tilted 45°": 45,
            "Tilted 60°": 60,
            "Vertical {111} (90°)": 90,
        }
        theta = np.deg2rad(angle_map[orientation])
    
    # Visualization settings
    st.sidebar.subheader("Visualization Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        eta_cmap = st.selectbox("η Colormap", cmap_list, index=cmap_list.index('viridis'))
    with col2:
        stress_cmap = st.selectbox("Stress Colormap", cmap_list, index=cmap_list.index('hot'))
    
    # Run button
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        # Prepare parameters
        params = {
            'name': sim_name,
            'defect_type': defect_type,
            'shape': shape,
            'eps0': eps0,
            'kappa': kappa,
            'orientation': orientation,
            'theta': theta,
            'steps': steps,
            'save_every': save_every,
            'eta_cmap': eta_cmap,
            'stress_cmap': stress_cmap
        }
        
        # Create progress container
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Running simulation... {progress*100:.1f}%")
        
        # Run simulation
        with st.spinner("Starting simulation..."):
            history, metadata = run_simulation(params, update_progress)
        
        # Save to database
        sim_id = SimulationDatabase.save_simulation(params, history, metadata)
        
        # Update status
        progress_container.empty()
        status_text.success(f"""
        ✅ Simulation Complete!
        - **ID**: `{sim_id}`
        - **Frames**: {len(history)}
        - **Time**: {metadata['run_time']:.1f} seconds
        """)
        
        # Store in session state for immediate display
        st.session_state.last_simulation = {
            'id': sim_id,
            'params': params,
            'history': history,
            'metadata': metadata
        }
        st.session_state.show_results = True
    
    # Display last simulation if available
    if 'last_simulation' in st.session_state and st.session_state.show_results:
        st.sidebar.subheader("Last Simulation Results")
        sim_data = st.session_state.last_simulation
        
        if st.sidebar.button("📊 Show Results", use_container_width=True):
            st.session_state.display_results = True
        
        if st.sidebar.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.show_results = False
            st.session_state.display_results = False
            st.rerun()

elif "Compare Simulations" in operation_mode:
    st.sidebar.header("🔍 Comparison Setup")
    
    # Get available simulations
    sim_list = SimulationDatabase.get_simulation_list()
    
    if not sim_list:
        st.sidebar.warning("No simulations found. Run some simulations first!")
    else:
        # Multi-select simulations
        sim_options = {sim['display_name']: sim['id'] for sim in sim_list}
        selected_names = st.sidebar.multiselect(
            "Select Simulations to Compare",
            options=list(sim_options.keys()),
            default=list(sim_options.keys())[:min(3, len(sim_options))],
            help="Select up to 6 simulations for comparison"
        )
        
        selected_ids = [sim_options[name] for name in selected_names]
        
        if selected_ids:
            # Comparison type
            comparison_type = st.sidebar.selectbox(
                "Comparison Type",
                [
                    "Side-by-Side Heatmaps",
                    "Overlay Line Profiles",
                    "Multi-Orientation Analysis",
                    "Statistical Comparison",
                    "Evolution Timeline",
                    "3D Surface Comparison"
                ],
                index=0
            )
            
            # Stress component
            stress_component = st.sidebar.selectbox(
                "Stress Component",
                ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
                index=0
            )
            
            # Frame selection
            frame_selection = st.sidebar.radio(
                "Frame Selection",
                ["Final Frame", "Mid Evolution", "Specific Frame"],
                horizontal=True
            )
            
            if frame_selection == "Specific Frame":
                frame_idx = st.sidebar.slider("Frame Index", 0, 100, 50)
            else:
                frame_idx = None
            
            # Enhanced options for line profiles
            if comparison_type in ["Overlay Line Profiles", "Multi-Orientation Analysis"]:
                st.sidebar.subheader("📐 Profile Settings")
                
                # Orientation selection
                profile_orientations = st.sidebar.multiselect(
                    "Select Profile Orientations",
                    ["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)", "135° (Diagonal)", "Custom"],
                    default=["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)"]
                )
                
                if "Custom" in profile_orientations:
                    custom_angle = st.sidebar.slider("Custom Angle (°)", -180, 180, 30, 5)
                
                # Profile parameters
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    profile_length = st.slider("Profile Length (%)", 10, 100, 80, 5)
                with col2:
                    line_style = st.selectbox("Line Style", ["solid", "dashed", "dotted", "dashdot"])
                
                # Advanced options
                with st.sidebar.expander("Advanced Options"):
                    sampling_factor = st.slider("Sampling Factor", 1, 5, 2)
                    show_statistics = st.checkbox("Show Statistics", True)
                    fixed_aspect = st.checkbox("Fixed Aspect Ratio", True, 
                                              help="Maintain 1:1 aspect ratio for domain plots")
            
            # Run comparison button
            if st.sidebar.button("🔬 Run Comparison", type="primary", use_container_width=True):
                # Prepare comparison config
                config = {
                    'type': comparison_type,
                    'sim_ids': selected_ids,
                    'stress_component': stress_component,
                    'frame_selection': frame_selection,
                    'frame_idx': frame_idx,
                }
                
                if comparison_type in ["Overlay Line Profiles", "Multi-Orientation Analysis"]:
                    config.update({
                        'profile_orientations': profile_orientations,
                        'profile_length': profile_length,
                        'line_style': line_style,
                        'sampling_factor': sampling_factor,
                        'show_statistics': show_statistics,
                        'fixed_aspect': fixed_aspect
                    })
                    if "Custom" in profile_orientations:
                        config['custom_angle'] = custom_angle
                
                # Store for display
                st.session_state.comparison_config = config
                st.session_state.run_comparison = True

elif "Analysis Dashboard" in operation_mode:
    st.sidebar.header("📊 Dashboard Settings")
    
    # Get statistics
    sim_list = SimulationDatabase.get_simulation_list()
    total_sims = len(sim_list)
    total_frames = sum(sim['metadata'].get('frames', 0) for sim in sim_list)
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <h3>📈 Platform Statistics</h3>
        <p><strong>Total Simulations:</strong> {total_sims}</p>
        <p><strong>Total Frames:</strong> {total_frames}</p>
        <p><strong>Database Size:</strong> {total_sims * 0.5:.1f} MB (est.)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis options
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Simulation Overview", "Defect Statistics", "Stress Analysis", "Trend Analysis"]
    )
    
    # Filter options
    with st.sidebar.expander("🔍 Filter Simulations"):
        defect_filter = st.multiselect(
            "Defect Type",
            ["ISF", "ESF", "Twin"],
            default=["ISF", "ESF", "Twin"]
        )
        
        orientation_filter = st.multiselect(
            "Orientation",
            ["Horizontal {111} (0°)", "Tilted 30°", "Tilted 45°", "Tilted 60°", "Vertical {111} (90°)"],
            default=["Horizontal {111} (0°)", "Vertical {111} (90°)"]
        )

elif "Export Data" in operation_mode:
    st.sidebar.header("💾 Export Configuration")
    
    # Export format
    export_format = st.sidebar.selectbox(
        "Export Format",
        [
            "Complete Package (ZIP)",
            "Publication Figures (PDF/PNG)",
            "Raw Data (CSV/HDF5)",
            "Analysis Report (PDF)",
            "Interactive Dashboard (HTML)"
        ],
        index=0
    )
    
    # Content selection
    st.sidebar.subheader("Content Selection")
    
    include_simulations = st.sidebar.multiselect(
        "Select Simulations",
        [sim['display_name'] for sim in SimulationDatabase.get_simulation_list()],
        help="Select simulations to include in export"
    )
    
    # Export options
    with st.sidebar.expander("📁 Export Options"):
        col1, col2 = st.columns(2)
        with col1:
            include_profiles = st.checkbox("Include Profiles", True)
            include_statistics = st.checkbox("Include Statistics", True)
        with col2:
            include_metadata = st.checkbox("Include Metadata", True)
            high_resolution = st.checkbox("High Resolution", True)
    
    # Generate export button
    if st.sidebar.button("📦 Generate Export", type="primary", use_container_width=True):
        st.session_state.generate_export = True
        st.session_state.export_config = {
            'format': export_format,
            'simulations': include_simulations,
            'options': {
                'profiles': include_profiles,
                'statistics': include_statistics,
                'metadata': include_metadata,
                'high_res': high_resolution
            }
        }

# =============================================
# MAIN CONTENT AREA
# =============================================

if "Run New Simulation" in operation_mode:
    st.header("🏃 Run New Simulation")
    
    # Show simulation preview if parameters are set
    if 'last_simulation' in st.session_state and st.session_state.get('display_results', False):
        sim_data = st.session_state.last_simulation
        params = sim_data['params']
        history = sim_data['history']
        
        # Display results
        st.success(f"**Simulation {sim_data['id']} Results**")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎬 Evolution", "📈 Analysis", "💾 Export"])
        
        with tab1:
            # Show final state
            if history:
                final_eta, final_stress = history[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
                    im = ax.imshow(final_eta, extent=extent, 
                                  cmap=plt.cm.get_cmap(COLORMAPS.get(params.get('eta_cmap', 'viridis'), 'viridis')),
                                  origin='lower', alpha=0.9)
                    ax.set_title(f"Final Defect Field: {params['defect_type']}")
                    ax.set_xlabel("x (nm)")
                    ax.set_ylabel("y (nm)")
                    FixedAspectManager.add_physical_scale(ax, 5.0)
                    plt.colorbar(im, ax=ax, shrink=0.8, label='η')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
                    im = ax.imshow(final_stress['sigma_mag'], extent=extent,
                                  cmap=plt.cm.get_cmap(COLORMAPS.get(params.get('stress_cmap', 'hot'), 'hot')),
                                  origin='lower', alpha=0.9)
                    ax.set_title("Final Stress Magnitude")
                    ax.set_xlabel("x (nm)")
                    ax.set_ylabel("y (nm)")
                    FixedAspectManager.add_physical_scale(ax, 5.0)
                    plt.colorbar(im, ax=ax, shrink=0.8, label='|σ| (GPa)')
                    st.pyplot(fig)
        
        with tab2:
            # Show evolution animation or sequence
            st.subheader("Evolution Timeline")
            
            # Select frame to display
            frame_idx = st.slider("Select Frame", 0, len(history)-1, len(history)-1)
            
            if frame_idx < len(history):
                eta_frame, stress_frame = history[frame_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
                    im = ax.imshow(eta_frame, extent=extent, 
                                  cmap=plt.cm.get_cmap(COLORMAPS.get(params.get('eta_cmap', 'viridis'), 'viridis')),
                                  origin='lower')
                    ax.set_title(f"Frame {frame_idx}")
                    ax.set_xlabel("x (nm)")
                    ax.set_ylabel("y (nm)")
                    st.pyplot(fig)
                
                with col2:
                    # Evolution metrics
                    eta_values = [h[0].mean() for h in history]
                    stress_values = [h[1]['sigma_mag'].mean() for h in history]
                    
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot(eta_values, label='Mean η', linewidth=2)
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Mean η")
                    ax.set_title("Defect Evolution")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        
        with tab3:
            # Analysis
            st.subheader("Detailed Analysis")
            
            if history:
                final_eta, final_stress = history[-1]
                
                # Extract profiles at different orientations
                angles = [0, 45, 90, 135]
                profiles = {}
                
                for angle in angles:
                    result = AdvancedProfileExtractor.extract_profile_2d(
                        final_stress['sigma_mag'], angle, 'center', 0, 0.8
                    )
                    profiles[angle] = result
                
                # Plot profiles
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                axes = axes.flatten()
                
                for idx, (angle, ax) in enumerate(zip(angles, axes)):
                    distances, profile, endpoints, metadata = profiles[angle]
                    ax.plot(distances, profile, linewidth=2)
                    ax.set_title(f"{angle}° Profile")
                    ax.set_xlabel("Distance (nm)")
                    ax.set_ylabel("Stress (GPa)")
                    ax.grid(True, alpha=0.3)
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab4:
            # Export options
            st.subheader("Export Simulation Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 Download Data", use_container_width=True):
                    # Create export
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, 'w') as zf:
                        # Add parameters
                        zf.writestr('parameters.json', json.dumps(params, indent=2))
                        
                        # Add data
                        for i, (eta, stress) in enumerate(history):
                            data = {
                                'eta': eta.tolist(),
                                'stress_magnitude': stress['sigma_mag'].tolist(),
                                'hydrostatic': stress['sigma_hydro'].tolist(),
                                'von_mises': stress['von_mises'].tolist()
                            }
                            zf.writestr(f'frame_{i:04d}.json', json.dumps(data, indent=2))
                    
                    buffer.seek(0)
                    st.download_button(
                        label="⬇️ Download ZIP",
                        data=buffer,
                        file_name=f"simulation_{sim_data['id']}.zip",
                        mime="application/zip"
                    )
            
            with col2:
                if st.button("🖼️ Save Figures", use_container_width=True):
                    st.info("Figure export functionality would be implemented here")
            
            with col3:
                if st.button("📋 Copy Summary", use_container_width=True):
                    summary = f"""
                    Simulation Summary: {sim_data['id']}
                    Defect Type: {params['defect_type']}
                    Orientation: {params['orientation']}
                    ε*: {params['eps0']:.3f}
                    κ: {params['kappa']:.2f}
                    Frames: {len(history)}
                    Run Time: {sim_data['metadata']['run_time']:.1f} seconds
                    """
                    st.code(summary)
    
    else:
        # Show welcome/instructions
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Welcome to the Simulation Platform</h3>
            <p>Configure your simulation parameters in the sidebar and click "Run Simulation" to start.</p>
            <p><strong>Available Features:</strong></p>
            <ul>
                <li>Multiple defect types (ISF, ESF, Twin)</li>
                <li>Various initial shapes and orientations</li>
                <li>Realistic stress calculation using FFT</li>
                <li>High-resolution line profile extraction</li>
                <li>Fixed aspect ratio visualization</li>
                <li>Publication-quality output</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example visualization
        st.subheader("Example Visualization")
        
        # Create example defect
        example_eta = create_initial_eta("Ellipse", "ISF")
        example_stress = compute_stress_fields(example_eta, 0.707, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
            im = ax.imshow(example_eta, extent=extent, cmap='viridis', origin='lower')
            ax.set_title("Example: ISF Defect")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            FixedAspectManager.add_physical_scale(ax, 5.0)
            plt.colorbar(im, ax=ax, shrink=0.8, label='η')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
            im = ax.imshow(example_stress['sigma_mag'], extent=extent, cmap='hot', origin='lower')
            ax.set_title("Example: Stress Field")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            FixedAspectManager.add_physical_scale(ax, 5.0)
            plt.colorbar(im, ax=ax, shrink=0.8, label='|σ| (GPa)')
            st.pyplot(fig)

elif "Compare Simulations" in operation_mode:
    st.header("🔍 Simulation Comparison")
    
    if 'comparison_config' in st.session_state and st.session_state.get('run_comparison', False):
        config = st.session_state.comparison_config
        
        # Load selected simulations
        simulations = []
        for sim_id in config['sim_ids']:
            sim_data = SimulationDatabase.get_simulation(sim_id)
            if sim_data:
                simulations.append(sim_data)
        
        if not simulations:
            st.error("No valid simulations selected!")
        else:
            st.success(f"Loaded {len(simulations)} simulations for comparison")
            
            # Determine frames
            frames = []
            for sim in simulations:
                if config['frame_selection'] == "Final Frame":
                    frames.append(len(sim['history']) - 1)
                elif config['frame_selection'] == "Mid Evolution":
                    frames.append(len(sim['history']) // 2)
                else:
                    frames.append(min(config.get('frame_idx', 0), len(sim['history']) - 1))
            
            # Create comparison visualization based on type
            if config['type'] == "Overlay Line Profiles":
                st.subheader("📈 Overlay Line Profiles Comparison")
                
                # Create visualization
                style_params = {
                    'line_width': 2.0,
                    'label_font_size': 12,
                    'title_font_size': 14,
                    'legend_fontsize': 10
                }
                
                fig = EnhancedVisualizer.create_overlay_profiles_plot(
                    simulations, frames, config, style_params
                )
                
                st.pyplot(fig)
                
                # Add statistics
                if config.get('show_statistics', True):
                    with st.expander("📊 Profile Statistics", expanded=True):
                        # Calculate statistics for each simulation
                        stats_data = []
                        stress_key = {
                            "Stress Magnitude |σ|": 'sigma_mag',
                            "Hydrostatic σ_h": 'sigma_hydro',
                            "von Mises σ_vM": 'von_mises'
                        }[config['stress_component']]
                        
                        for sim, frame in zip(simulations, frames):
                            eta, stress_fields = sim['history'][frame]
                            stress_data = stress_fields[stress_key]
                            
                            # Extract profiles at main orientations
                            angles = [0, 45, 90, 135]
                            for angle in angles:
                                result = AdvancedProfileExtractor.extract_profile_2d(
                                    stress_data, angle, 'center', 0, 0.8
                                )
                                distances, profile, endpoints, metadata = result
                                
                                stats_data.append({
                                    'Simulation': sim['params']['defect_type'],
                                    'Orientation': f"{angle}°",
                                    'Max (GPa)': metadata['max_value'],
                                    'Mean (GPa)': metadata['mean_value'],
                                    'FWHM (nm)': metadata['fwhm_nm']
                                })
                        
                        if stats_data:
                            df_stats = pd.DataFrame(stats_data)
                            st.dataframe(
                                df_stats.style.format({
                                    'Max (GPa)': '{:.3f}',
                                    'Mean (GPa)': '{:.3f}',
                                    'FWHM (nm)': '{:.2f}'
                                }),
                                use_container_width=True
                            )
            
            elif config['type'] == "Side-by-Side Heatmaps":
                st.subheader("🔥 Side-by-Side Heatmap Comparison")
                
                # Create grid of heatmaps
                n_sims = len(simulations)
                n_cols = min(3, n_sims)
                n_rows = (n_sims + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, 
                                        figsize=(6*n_cols, 5*n_rows),
                                        constrained_layout=True)
                
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                stress_key = {
                    "Stress Magnitude |σ|": 'sigma_mag',
                    "Hydrostatic σ_h": 'sigma_hydro',
                    "von Mises σ_vM": 'von_mises'
                }[config['stress_component']]
                
                for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    row = idx // n_cols
                    col = idx % n_cols
                    ax = axes[row, col]
                    
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Apply fixed aspect ratio
                    FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
                    
                    # Plot heatmap
                    cmap = plt.cm.get_cmap(COLORMAPS.get(
                        sim['params'].get('stress_cmap', 'hot'), 'hot'
                    ))
                    im = ax.imshow(stress_data, extent=extent, cmap=cmap, 
                                  origin='lower', vmin=0, vmax=np.max(stress_data)*0.8)
                    
                    # Add title and labels
                    title = f"{sim['params']['defect_type']}"
                    if sim['params']['orientation'] != "Horizontal {111} (0°)":
                        title += f"\n{sim['params']['orientation'].split(' ')[0]}"
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    
                    if row == n_rows - 1:
                        ax.set_xlabel("x (nm)", fontsize=10)
                    if col == 0:
                        ax.set_ylabel("y (nm)", fontsize=10)
                    
                    # Add scale bar
                    FixedAspectManager.add_physical_scale(ax, 5.0, location='lower right')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, shrink=0.8, label='Stress (GPa)')
                
                # Hide empty subplots
                for idx in range(n_sims, n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    axes[row, col].axis('off')
                
                st.pyplot(fig)
            
            elif config['type'] == "Multi-Orientation Analysis":
                st.subheader("🎯 Multi-Orientation Comprehensive Analysis")
                
                # Create comprehensive visualization
                style_params = {
                    'line_width': 2.0,
                    'label_font_size': 12,
                    'title_font_size': 14,
                    'legend_fontsize': 10
                }
                
                fig = EnhancedVisualizer.create_comprehensive_comparison(
                    simulations, frames, config, style_params
                )
                
                st.pyplot(fig)
            
            # Clear comparison flag
            st.session_state.run_comparison = False
    
    else:
        # Show comparison interface
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Advanced Comparison Tools</h3>
            <p>Select simulations in the sidebar and configure your comparison to get started.</p>
            <p><strong>Available Comparison Types:</strong></p>
            <ul>
                <li><strong>Side-by-Side Heatmaps:</strong> Visual comparison of stress distributions</li>
                <li><strong>Overlay Line Profiles:</strong> Compare stress profiles at multiple orientations</li>
                <li><strong>Multi-Orientation Analysis:</strong> Comprehensive analysis with fixed aspect ratios</li>
                <li><strong>Statistical Comparison:</strong> Quantitative comparison of simulation metrics</li>
                <li><strong>Evolution Timeline:</strong> Compare temporal evolution of defects</li>
                <li><strong>3D Surface Comparison:</strong> Three-dimensional visualization of stress fields</li>
            </ul>
            <p><strong>Enhanced Features:</strong></p>
            <ul>
                <li>Fixed 1:1 aspect ratio for realistic visualization</li>
                <li>High-resolution profile extraction at arbitrary angles</li>
                <li>Publication-quality figures with proper scaling</li>
                <li>Statistical analysis of profile characteristics</li>
                <li>Export-ready visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show available simulations
        sim_list = SimulationDatabase.get_simulation_list()
        if sim_list:
            st.subheader("📋 Available Simulations")
            
            # Create dataframe of simulations
            sim_data = []
            for sim in sim_list:
                params = sim['params']
                metadata = sim['metadata']
                
                sim_data.append({
                    'ID': sim['id'][:8],
                    'Name': sim['name'],
                    'Defect Type': params['defect_type'],
                    'Orientation': params['orientation'],
                    'ε*': f"{params['eps0']:.3f}",
                    'κ': f"{params['kappa']:.2f}",
                    'Frames': metadata.get('frames', 0),
                    'Created': metadata.get('created', 'Unknown')[:10]
                })
            
            df = pd.DataFrame(sim_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

elif "Analysis Dashboard" in operation_mode:
    st.header("📊 Analysis Dashboard")
    
    # Get simulation data
    sim_list = SimulationDatabase.get_simulation_list()
    
    if not sim_list:
        st.info("No simulations available for analysis. Run some simulations first!")
    else:
        # Create dashboard with metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Simulations", len(sim_list))
        
        with col2:
            total_frames = sum(sim['metadata'].get('frames', 0) for sim in sim_list)
            st.metric("Total Frames", total_frames)
        
        with col3:
            # Count by defect type
            defect_counts = {}
            for sim in sim_list:
                defect = sim['params']['defect_type']
                defect_counts[defect] = defect_counts.get(defect, 0) + 1
            
            st.metric("Defect Types", len(defect_counts))
        
        with col4:
            # Average run time
            avg_time = np.mean([sim['metadata'].get('run_time', 0) for sim in sim_list])
            st.metric("Avg Run Time", f"{avg_time:.1f}s")
        
        # Create visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Statistics", "🎯 Defect Analysis", "📊 Stress Analysis"])
        
        with tab1:
            # Statistical overview
            st.subheader("Simulation Statistics")
            
            # Create bar chart of defect types
            defect_types = [sim['params']['defect_type'] for sim in sim_list]
            unique_types, counts = np.unique(defect_types, return_counts=True)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(unique_types, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique_types))))
            ax.set_xlabel("Defect Type")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Defect Types")
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(count), ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Parameter distributions
            st.subheader("Parameter Distributions")
            
            eps_values = [sim['params']['eps0'] for sim in sim_list]
            kappa_values = [sim['params']['kappa'] for sim in sim_list]
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(eps_values, bins=10, edgecolor='black', alpha=0.7)
                ax.set_xlabel("Eigenstrain ε*")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of ε*")
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(kappa_values, bins=10, edgecolor='black', alpha=0.7)
                ax.set_xlabel("Interface Energy κ")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of κ")
                st.pyplot(fig)
        
        with tab2:
            # Defect analysis
            st.subheader("Defect Field Analysis")
            
            # Select a simulation for detailed analysis
            selected_sim = st.selectbox(
                "Select Simulation for Analysis",
                [sim['display_name'] for sim in sim_list],
                index=0
            )
            
            # Find selected simulation
            selected_data = None
            for sim in sim_list:
                if sim['display_name'] == selected_sim:
                    selected_data = SimulationDatabase.get_simulation(sim['id'])
                    break
            
            if selected_data:
                # Display defect analysis
                params = selected_data['params']
                history = selected_data['history']
                
                if history:
                    final_eta, final_stress = history[-1]
                    
                    # Calculate defect properties
                    eta_threshold = 0.5
                    defect_mask = final_eta > eta_threshold
                    defect_area = np.sum(defect_mask) * dx**2  # nm²
                    defect_perimeter = np.sum(np.gradient(defect_mask.astype(float))) * dx
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Defect Area", f"{defect_area:.2f} nm²")
                    with col2:
                        st.metric("Defect Perimeter", f"{defect_perimeter:.2f} nm")
                    with col3:
                        compactness = (4 * np.pi * defect_area) / (defect_perimeter**2) if defect_perimeter > 0 else 0
                        st.metric("Compactness", f"{compactness:.3f}")
                    
                    # Show defect shape analysis
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Defect field
                    FixedAspectManager.apply_fixed_aspect(ax1, aspect_type='equal')
                    im1 = ax1.imshow(final_eta, extent=extent, cmap='viridis', origin='lower')
                    ax1.set_title("Defect Field")
                    ax1.set_xlabel("x (nm)")
                    ax1.set_ylabel("y (nm)")
                    plt.colorbar(im1, ax=ax1, shrink=0.8, label='η')
                    
                    # Thresholded defect
                    FixedAspectManager.apply_fixed_aspect(ax2, aspect_type='equal')
                    im2 = ax2.imshow(defect_mask, extent=extent, cmap='binary', origin='lower')
                    ax2.set_title(f"Defect Area (η > {eta_threshold})")
                    ax2.set_xlabel("x (nm)")
                    ax2.set_ylabel("y (nm)")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab3:
            # Stress analysis
            st.subheader("Stress Field Analysis")
            
            # Select simulations for comparison
            selected_sims = st.multiselect(
                "Select Simulations for Stress Analysis",
                [sim['display_name'] for sim in sim_list],
                default=[sim['display_name'] for sim in sim_list[:3]]
            )
            
            if selected_sims:
                # Load selected simulations
                stress_data = []
                for sim_name in selected_sims:
                    for sim in sim_list:
                        if sim['display_name'] == sim_name:
                            sim_data = SimulationDatabase.get_simulation(sim['id'])
                            if sim_data and sim_data['history']:
                                final_eta, final_stress = sim_data['history'][-1]
                                stress_data.append({
                                    'name': sim_data['params']['defect_type'],
                                    'stress': final_stress['sigma_mag'],
                                    'params': sim_data['params']
                                })
                            break
                
                if stress_data:
                    # Calculate stress statistics
                    stats = []
                    for data in stress_data:
                        stress_values = data['stress'].flatten()
                        stats.append({
                            'Simulation': data['name'],
                            'Max (GPa)': np.max(stress_values),
                            'Mean (GPa)': np.mean(stress_values),
                            'Std (GPa)': np.std(stress_values),
                            '95th Percentile (GPa)': np.percentile(stress_values, 95)
                        })
                    
                    df_stats = pd.DataFrame(stats)
                    st.dataframe(
                        df_stats.style.format({
                            'Max (GPa)': '{:.3f}',
                            'Mean (GPa)': '{:.3f}',
                            'Std (GPa)': '{:.3f}',
                            '95th Percentile (GPa)': '{:.3f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Create stress comparison plot
                    fig, axes = plt.subplots(1, len(stress_data), figsize=(5*len(stress_data), 4))
                    if len(stress_data) == 1:
                        axes = [axes]
                    
                    for idx, (data, ax) in enumerate(zip(stress_data, axes)):
                        FixedAspectManager.apply_fixed_aspect(ax, aspect_type='equal')
                        im = ax.imshow(data['stress'], extent=extent, cmap='hot', 
                                      origin='lower', vmin=0)
                        ax.set_title(data['name'])
                        ax.set_xlabel("x (nm)")
                        if idx == 0:
                            ax.set_ylabel("y (nm)")
                        plt.colorbar(im, ax=ax, shrink=0.8, label='|σ| (GPa)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

elif "Export Data" in operation_mode:
    st.header("💾 Data Export")
    
    if 'generate_export' in st.session_state and st.session_state.generate_export:
        config = st.session_state.export_config
        
        # Create export
        with st.spinner("Creating export package..."):
            buffer = BytesIO()
            
            with zipfile.ZipFile(buffer, 'w') as zf:
                # Add README
                readme = f"""
                # Simulation Data Export
                Generated: {datetime.now().isoformat()}
                Export Format: {config['format']}
                
                ## Contents
                This package contains simulation data from the Ag NP Multi-Defect Analysis Platform.
                
                ## Files
                - simulations.json: Simulation parameters and metadata
                - data/: Simulation data files
                - figures/: Visualization figures
                - analysis/: Statistical analysis
                
                ## Platform Information
                - Version: 2.0.0
                - Grid Size: {N} x {N}
                - Grid Spacing: {dx} nm
                - Domain: {extent[0]:.1f} to {extent[1]:.1f} nm (x), {extent[2]:.1f} to {extent[3]:.1f} nm (y)
                
                ## Citation
                Please cite this data if used in publications.
                """
                zf.writestr("README.md", readme)
                
                # Add simulation data
                sim_list = SimulationDatabase.get_simulation_list()
                sim_data = []
                
                for sim in sim_list:
                    if sim['display_name'] in config['simulations']:
                        sim_data.append(sim)
                
                # Create comprehensive export
                export_data = {
                    'metadata': {
                        'export_date': datetime.now().isoformat(),
                        'export_format': config['format'],
                        'options': config['options'],
                        'platform_info': {
                            'grid_size': N,
                            'dx': dx,
                            'extent': extent,
                            'material_constants': {
                                'C11': C11,
                                'C12': C12,
                                'C44': C44
                            }
                        }
                    },
                    'simulations': []
                }
                
                for sim in sim_data:
                    sim_info = {
                        'id': sim['id'],
                        'name': sim['name'],
                        'params': sim['params'],
                        'metadata': sim['metadata']
                    }
                    export_data['simulations'].append(sim_info)
                
                zf.writestr('export_manifest.json', json.dumps(export_data, indent=2))
            
            buffer.seek(0)
            
            # Provide download
            st.success("Export package created successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Export Package",
                    data=buffer,
                    file_name=f"simulation_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄 Create New Export", use_container_width=True):
                    st.session_state.generate_export = False
                    st.rerun()
    
    else:
        # Show export interface
        st.markdown("""
        <div class="feature-card">
            <h3>💾 Data Export Center</h3>
            <p>Export your simulation data in various formats for analysis, publication, or sharing.</p>
            <p><strong>Available Export Formats:</strong></p>
            <ul>
                <li><strong>Complete Package (ZIP):</strong> All data, figures, and metadata</li>
                <li><strong>Publication Figures (PDF/PNG):</strong> High-resolution figures ready for publication</li>
                <li><strong>Raw Data (CSV/HDF5):</strong> Numerical data for further analysis</li>
                <li><strong>Analysis Report (PDF):</strong> Comprehensive report with statistics and visualizations</li>
                <li><strong>Interactive Dashboard (HTML):</strong> Self-contained interactive visualization</li>
            </ul>
            <p><strong>Export Options:</strong></p>
            <ul>
                <li>Include high-resolution line profiles</li>
                <li>Add statistical analysis</li>
                <li>Include complete metadata</li>
                <li>Generate publication-quality figures</li>
                <li>Create interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show export statistics
        sim_list = SimulationDatabase.get_simulation_list()
        if sim_list:
            st.subheader("📊 Export Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Available Simulations", len(sim_list))
            
            with col2:
                total_size = len(sim_list) * 2  # Rough estimate in MB
                st.metric("Estimated Size", f"{total_size} MB")
            
            with col3:
                st.metric("Export Ready", "✅")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Ag Nanoparticle Multi-Defect Analysis Platform Pro v2.0</strong></p>
    <p>Advanced stress analysis with multi-orientation profiling and fixed aspect ratio visualization</p>
    <p>© 2024 • Scientific Computing Group • All rights reserved</p>
</div>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE MANAGEMENT
# =============================================
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'display_results' not in st.session_state:
    st.session_state.display_results = False
if 'run_comparison' not in st.session_state:
    st.session_state.run_comparison = False
if 'generate_export' not in st.session_state:
    st.session_state.generate_export = False
