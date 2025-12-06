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
from scipy.ndimage import gaussian_filter, map_coordinates
import warnings
warnings.filterwarnings('ignore')

# Configure page with better styling
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Comparison Platform")
st.markdown("""
**Enhanced with Multi-Orientation Line Profiles • Fixed Aspect Ratio Domain • Advanced Post-Processing**
**Run → Save → Compare • 50+ Colormaps • Publication-ready comparison plots • Enhanced Visualization**
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
# ENHANCED LINE PROFILE EXTRACTION FUNCTIONS
# =============================================
def extract_line_profile(data, angle_deg, position='center', offset=0, length_ratio=0.8):
    """
    Extract line profile from 2D data at specified angle and position
    
    Parameters:
    -----------
    data : 2D numpy array
        Input data (stress field or defect parameter)
    angle_deg : float
        Angle in degrees from horizontal (0° = horizontal, 90° = vertical)
    position : str or tuple
        'center' for center of domain, 'offset' for offset position, or (x, y) coordinates
    offset : float
        Offset distance in nm (perpendicular to profile direction)
    length_ratio : float
        Length of profile as ratio of domain size (0-1)
    
    Returns:
    --------
    distance : 1D array
        Distance along profile (nm)
    profile : 1D array
        Extracted profile values
    endpoints : tuple
        (x_start, y_start, x_end, y_end) in physical coordinates
    """
    N = data.shape[0]
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Determine center point
    if position == 'center':
        x_center, y_center = 0, 0  # Physical coordinates
    elif position == 'offset':
        # Calculate perpendicular direction
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
    
    # Calculate start and end points
    x_start = x_center - half_length * np.cos(angle_rad)
    y_start = y_center - half_length * np.sin(angle_rad)
    x_end = x_center + half_length * np.cos(angle_rad)
    y_end = y_center + half_length * np.sin(angle_rad)
    
    # Generate points along the line
    num_points = 500  # High resolution for smooth profiles
    distances = np.linspace(-half_length, half_length, num_points)
    xs = x_center + distances * np.cos(angle_rad)
    ys = y_center + distances * np.sin(angle_rad)
    
    # Convert physical coordinates to array indices
    xi = (xs - extent[0]) / (extent[1] - extent[0]) * (N - 1)
    yi = (ys - extent[2]) / (extent[3] - extent[2]) * (N - 1)
    
    # Interpolate data at these points
    profile = map_coordinates(data, [yi, xi], order=3, mode='constant', cval=0.0)
    
    return distances, profile, (x_start, y_start, x_end, y_end)

def extract_multiple_profiles(data, angle_list, position_list, **kwargs):
    """
    Extract multiple line profiles at once
    
    Parameters:
    -----------
    data : 2D numpy array
        Input data
    angle_list : list of float
        List of angles in degrees
    position_list : list
        List of positions (same format as extract_line_profile)
    
    Returns:
    --------
    profiles : dict
        Dictionary with keys as (angle, position) and values as (distances, profile, endpoints)
    """
    profiles = {}
    for angle, pos in zip(angle_list, position_list):
        key = f"{angle}°_{pos}"
        if isinstance(pos, str) and pos == 'offset':
            # Generate multiple offsets
            for offset in [-2.0, 0, 2.0]:  # nm offsets
                subkey = f"{angle}°_offset{offset:+1f}"
                profiles[subkey] = extract_line_profile(data, angle, 'offset', offset, **kwargs)
        else:
            profiles[key] = extract_line_profile(data, angle, pos, **kwargs)
    return profiles

# =============================================
# FIXED ASPECT RATIO VISUALIZATION FUNCTIONS
# =============================================
class FixedAspectVisualizer:
    """Maintain fixed 1:1 aspect ratio for domain visualizations"""
    
    @staticmethod
    def calculate_domain_aspect_ratio():
        """Calculate the physical aspect ratio of the simulation domain"""
        width = extent[1] - extent[0]  # nm
        height = extent[3] - extent[2]  # nm
        return height / width  # Should be 1.0 for square domain
    
    @staticmethod
    def create_fixed_aspect_figure(n_rows=1, n_cols=1, base_width=8.0, domain_row=None, 
                                   domain_col=None, domain_height_ratio=1.0):
        """
        Create a figure with fixed aspect ratio for domain subplots
        
        Parameters:
        -----------
        n_rows, n_cols : int
            Grid dimensions
        base_width : float
            Base width in inches
        domain_row, domain_col : int or None
            Position of domain subplot (for mixed aspect ratios)
        domain_height_ratio : float
            Height ratio for domain subplot (typically 1.0 for square)
        
        Returns:
        --------
        fig : matplotlib Figure
        axes : numpy array of Axes
        """
        # Calculate figure height based on domain aspect ratio
        domain_aspect = FixedAspectVisualizer.calculate_domain_aspect_ratio()
        
        if domain_row is not None and domain_col is not None:
            # Mixed layout: some subplots with fixed aspect, others flexible
            fig_height = base_width * (n_rows / n_cols) * 1.2
            fig, axes = plt.subplots(n_rows, n_cols, 
                                    figsize=(base_width * n_cols, fig_height),
                                    constrained_layout=True)
        else:
            # All subplots have same aspect ratio
            if abs(domain_aspect - 1.0) < 0.01:  # Square domain
                fig_height = base_width * (n_rows / n_cols)
            else:
                fig_height = base_width * domain_aspect * (n_rows / n_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                    figsize=(base_width * n_cols, fig_height),
                                    constrained_layout=True)
        
        return fig, axes
    
    @staticmethod
    def apply_fixed_aspect_to_domain(ax, data=None):
        """
        Apply fixed 1:1 aspect ratio to domain visualization
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to modify
        data : 2D array or None
            If provided, uses data shape to determine aspect
        """
        ax.set_aspect('equal')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return ax
    
    @staticmethod
    def add_scale_bar_to_domain(ax, length_nm=5.0, location='lower right', color='white'):
        """Add scale bar to domain visualization"""
        # Use relative coordinates for positioning
        if location == 'lower right':
            x_pos = 0.95
            y_pos = 0.05
            ha = 'right'
            va = 'bottom'
        elif location == 'lower left':
            x_pos = 0.05
            y_pos = 0.05
            ha = 'left'
            va = 'bottom'
        else:
            x_pos = 0.95
            y_pos = 0.95
            ha = 'right'
            va = 'top'
        
        # Convert to data coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Bar position in data coordinates
        bar_x_start = xlim[1] - x_range * 0.15
        bar_x_end = bar_x_start - length_nm
        bar_y = ylim[0] + y_range * 0.05
        
        # Draw scale bar
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
               color=color, linewidth=2, solid_capstyle='butt')
        
        # Add text
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
               f'{length_nm} nm', ha='center', va='bottom',
               color=color, fontsize=8, fontweight='bold')
        
        return ax

# =============================================
# ENHANCED OVERLAY LINE PROFILE VISUALIZATION
# =============================================
def create_fixed_aspect_overlay_profiles(simulations, frames, config, style_params):
    """Create overlay line profiles with FIXED aspect ratio domain visualization"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    # Create figure with FIXED layout
    fig = plt.figure(figsize=(16, 12))
    fig.set_constrained_layout(True)
    
    # Create grid with specific height ratios to accommodate domain subplot
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])  # All profiles overlay
    ax2 = fig.add_subplot(gs[1, 0])  # Horizontal profile
    ax3 = fig.add_subplot(gs[1, 1])  # Vertical profile
    ax4 = fig.add_subplot(gs[1, 2])  # Diagonal profile
    ax5 = fig.add_subplot(gs[2, :])  # Domain with profile lines - FIXED ASPECT
    
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
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Plot all orientations overlay
    all_profiles_data = {}
    
    for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        for angle in angles:
            profile_length = config.get('profile_length', 80) / 100
            distances, profile, endpoints = extract_line_profile(
                stress_data, angle, 'center', 0, profile_length
            )
            
            # Store for statistics
            if angle not in all_profiles_data:
                all_profiles_data[angle] = []
            all_profiles_data[angle].append(profile)
            
            # Plot in overlay
            line_style = config.get('line_style', 'solid')
            label = f"{sim['params']['defect_type']} - {angle}°"
            ax1.plot(distances, profile, 
                    color=colors[sim_idx],
                    linestyle=line_style,
                    linewidth=style_params.get('line_width', 1.5) * 0.8,
                    alpha=0.7,
                    label=label)
    
    ax1.set_xlabel("Distance from Center (nm)", fontsize=style_params.get('label_font_size', 10))
    ax1.set_ylabel(f"{config['stress_component']} (GPa)", 
                  fontsize=style_params.get('label_font_size', 10))
    ax1.set_title(f"Overlay Line Profiles: {config['stress_component']}", 
                 fontsize=style_params.get('title_font_size', 12),
                 fontweight='bold')
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot individual orientations in separate subplots
    orientation_axes = [ax2, ax3, ax4]
    orientation_titles = ["Horizontal (0°)", "Vertical (90°)", "Diagonal (45°)"]
    
    for idx, (angle, ax, title) in enumerate(zip(angles[:3], orientation_axes, orientation_titles)):
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            profile_length = config.get('profile_length', 80) / 100
            distances, profile, endpoints = extract_line_profile(
                stress_data, angle, 'center', 0, profile_length
            )
            
            line_style = config.get('line_style', 'solid')
            label = f"{sim['params']['defect_type']}" if idx == 0 else None
            
            ax.plot(distances, profile, 
                   color=colors[sim_idx],
                   linestyle=line_style,
                   linewidth=style_params.get('line_width', 1.5),
                   alpha=0.8,
                   label=label)
        
        ax.set_xlabel("Distance (nm)", fontsize=9)
        ax.set_ylabel("Stress (GPa)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Plot domain with profile lines - WITH FIXED ASPECT RATIO
    if simulations:
        sim = simulations[0]
        eta, _ = sim['history'][frames[0]]
        
        # Apply FIXED aspect ratio
        FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax5)
        
        # Plot domain with realistic colors
        im = ax5.imshow(eta, extent=extent, 
                       cmap=enhanced_cmaps['plasma_enhanced'], 
                       origin='lower', 
                       aspect='auto')  # Auto aspect, but we fixed it above
        
        # Add profile lines with different colors
        line_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        for idx, angle in enumerate(angles):
            profile_length = config.get('profile_length', 80) / 100
            distances, profile, endpoints = extract_line_profile(
                eta, angle, 'center', 0, profile_length
            )
            x_start, y_start, x_end, y_end = endpoints
            
            # Draw profile line
            ax5.plot([x_start, x_end], [y_start, y_end], 
                    color=line_colors[idx % len(line_colors)], 
                    linewidth=2, linestyle='-',
                    label=f'{angle}°')
            
            # Add angle label at midpoint
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2
            ax5.text(mid_x, mid_y, f'{angle}°', 
                    color=line_colors[idx % len(line_colors)],
                    fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='white', alpha=0.7))
        
        ax5.set_xlabel("x (nm)", fontsize=10)
        ax5.set_ylabel("y (nm)", fontsize=10)
        ax5.set_title("Simulation Domain with Profile Lines", 
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8, loc='upper right')
        
        # Add scale bar
        FixedAspectVisualizer.add_scale_bar_to_domain(ax5, 5.0, location='lower right')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.8, pad=0.02)
        cbar.set_label('Defect Parameter η', fontsize=9)
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5], style_params)
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.05, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
    
    return fig

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
# ENHANCED POST-PROCESSING WITH FIXED ASPECT RATIO
# =============================================
class EnhancedFigureStyler:
    """Extended figure styler with publication-quality enhancements and fixed aspect ratio"""
    
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
        
        # Adjust layout
        fig.set_constrained_layout(True)
        
        return fig
    
    @staticmethod
    def get_enhanced_controls():
        """Get comprehensive styling controls including fixed aspect ratio option"""
        style_params = {}
        
        st.sidebar.header("🎨 Enhanced Post-Processing")
        
        with st.sidebar.expander("📐 Fixed Aspect Ratio Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                style_params['fix_domain_aspect'] = st.checkbox(
                    "Fix Domain Aspect Ratio (1:1)", 
                    True,
                    help="Maintain square aspect ratio for domain visualizations"
                )
            with col2:
                style_params['show_scale_bar'] = st.checkbox(
                    "Show Scale Bar on Domain", 
                    True,
                    help="Add scale bar to domain plots"
                )
        
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
    
    @staticmethod
    def apply_publication_styling(fig, axes, style_params):
        """Apply enhanced publication styling"""
        # Apply base styling
        fig = EnhancedFigureStyler.apply_advanced_styling(fig, axes, style_params)
        
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
        
        return fig

# =============================================
# ADVANCED PLOTTING ENHANCEMENTS
# =============================================
class PublicationEnhancer:
    """Advanced plotting enhancements for publication-quality figures"""
    
    @staticmethod
    def create_custom_colormaps():
        """Create enhanced scientific colormaps"""
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        
        # Perceptually uniform sequential
        plasma_enhanced = LinearSegmentedColormap.from_list('plasma_enhanced', [
            (0.0, '#0c0887'),
            (0.1, '#4b03a1'),
            (0.3, '#8b0aa5'),
            (0.5, '#b83289'),
            (0.7, '#db5c68'),
            (0.9, '#f48849'),
            (1.0, '#fec325')
        ])
        
        # Diverging with better contrast
        coolwarm_enhanced = LinearSegmentedColormap.from_list('coolwarm_enhanced', [
            (0.0, '#3a4cc0'),
            (0.25, '#8abcdd'),
            (0.5, '#f7f7f7'),
            (0.75, '#f0b7a4'),
            (1.0, '#b40426')
        ])
        
        # Categorical for defect types
        defect_categorical = ListedColormap([
            '#1f77b4',  # ISF - Blue
            '#ff7f0e',  # ESF - Orange
            '#2ca02c',  # Twin - Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b'   # Brown
        ])
        
        # Stress-specific colormap
        stress_map = LinearSegmentedColormap.from_list('stress_map', [
            (0.0, '#2c7bb6'),
            (0.2, '#abd9e9'),
            (0.4, '#ffffbf'),
            (0.6, '#fdae61'),
            (0.8, '#d7191c'),
            (1.0, '#800026')
        ])
        
        return {
            'plasma_enhanced': plasma_enhanced,
            'coolwarm_enhanced': coolwarm_enhanced,
            'defect_categorical': defect_categorical,
            'stress_map': stress_map
        }
    
    @staticmethod
    def add_profile_lines_to_domain(ax, profiles_dict, line_colors=None, line_styles=None, alpha=0.7):
        """Add profile lines to domain plot"""
        if line_colors is None:
            line_colors = plt.cm.rainbow(np.linspace(0, 1, len(profiles_dict)))
        
        if line_styles is None:
            line_styles = ['-', '--', '-.', ':'] * (len(profiles_dict) // 4 + 1)
        
        for idx, (key, (distances, profile, endpoints)) in enumerate(profiles_dict.items()):
            x_start, y_start, x_end, y_end = endpoints
            color = line_colors[idx % len(line_colors)]
            style = line_styles[idx % len(line_styles)]
            
            ax.plot([x_start, x_end], [y_start, y_end], 
                   color=color, linewidth=2, linestyle=style, alpha=alpha,
                   label=key.replace('_', ' '))
        
        return ax

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
# ENHANCED COMPARISON FUNCTIONS WITH FIXED ASPECT
# =============================================
def create_publication_heatmaps(simulations, frames, config, style_params):
    """Publication-quality heatmap comparison with fixed aspect ratio"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    n_sims = len(simulations)
    cols = min(3, n_sims)
    rows = (n_sims + cols - 1) // cols
    
    # Create figure with fixed layout
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 8*rows), constrained_layout=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    
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
        if cmap_name in enhanced_cmaps:
            cmap = enhanced_cmaps[cmap_name]
        else:
            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
        
        # Create heatmap with FIXED ASPECT RATIO
        if style_params.get('fix_domain_aspect', True):
            FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax)
        
        im = ax.imshow(stress_data, extent=extent, cmap=cmap, origin='lower')
        
        # Add contour lines for defect boundary
        contour = ax.contour(X, Y, eta, levels=[0.5], colors='white', 
                           linewidths=1, linestyles='--', alpha=0.8)
        
        # Add scale bar
        if style_params.get('show_scale_bar', True):
            FixedAspectVisualizer.add_scale_bar_to_domain(ax, 5.0, location='lower right')
        
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

def create_multi_orientation_profiles(simulations, frames, config, style_params):
    """Create multi-orientation line profiles with fixed aspect domain"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    # Parse orientation angles
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
    
    n_angles = len(angles)
    cols = min(2, n_angles)
    rows = (n_angles + cols - 1) // cols
    
    # Create figure with fixed layout
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8*rows/cols), constrained_layout=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Plot profiles for each orientation
    for angle_idx, angle in enumerate(angles):
        row = angle_idx // cols
        col = angle_idx % cols
        ax = axes[row, col]
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Extract profile at this angle
            profile_length = config.get('profile_length', 80) / 100
            distances, profile, endpoints = extract_line_profile(
                stress_data, angle, 'center', 0, profile_length
            )
            
            # Plot profile
            line_style = config.get('line_style', 'solid')
            label = f"{sim['params']['defect_type']}"
            
            ax.plot(distances, profile, 
                   color=colors[sim_idx],
                   linestyle=line_style,
                   linewidth=style_params.get('line_width', 1.5),
                   alpha=0.8,
                   label=label)
        
        ax.set_xlabel("Distance from Center (nm)", fontsize=style_params.get('label_font_size', 10))
        ax.set_ylabel(f"{config['stress_component']} (GPa)", 
                     fontsize=style_params.get('label_font_size', 10))
        ax.set_title(f"Line Profile at {angle}°", 
                    fontsize=style_params.get('title_font_size', 12),
                    fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if angle_idx == 0:
            ax.legend(fontsize=style_params.get('legend_fontsize', 10))
    
    # Hide empty subplots
    for idx in range(n_angles, rows*cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
    
    # Add super title
    fig.suptitle(f"Multi-Orientation Line Profiles: {config['stress_component']}", 
                fontsize=style_params.get('title_font_size', 14) + 2,
                fontweight='bold', y=1.02)
    
    return fig

def create_enhanced_comparison_plot(simulations, frames, config, style_params):
    """Create enhanced comparison plots with fixed aspect ratio"""
    
    if config['type'] == "Side-by-Side Heatmaps":
        return create_publication_heatmaps(simulations, frames, config, style_params)
    elif config['type'] == "Overlay Line Profiles":
        return create_fixed_aspect_overlay_profiles(simulations, frames, config, style_params)
    elif config['type'] == "Multi-Orientation Profiles":
        return create_multi_orientation_profiles(simulations, frames, config, style_params)
    else:
        # Fall back to simpler visualization
        return create_simple_comparison_plot(simulations, frames, config, style_params)

def create_simple_comparison_plot(simulations, frames, config, style_params):
    """Simple comparison plot for unsupported types"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_constrained_layout(True)
    
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
    
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Simple line plot of mean stress
        mean_stress = np.mean(stress_data)
        ax.bar(idx, mean_stress, color=color, alpha=0.7, 
               label=f"{sim['params']['defect_type']}")
    
    ax.set_xlabel("Simulation", fontsize=style_params.get('label_font_size', 12))
    ax.set_ylabel(f"Mean {config['stress_component']} (GPa)", 
                  fontsize=style_params.get('label_font_size', 12))
    ax.set_title(f"{config['type']} Comparison", 
                 fontsize=style_params.get('title_font_size', 14),
                 fontweight='bold')
    ax.legend(fontsize=style_params.get('legend_fontsize', 10))
    
    # Apply styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, ax, style_params)
    
    return fig

# =============================================
# SIDEBAR - Global Settings
# =============================================
st.sidebar.header("🎨 Global Chart Styling")

# Get enhanced controls including fixed aspect ratio
advanced_styling = EnhancedFigureStyler.get_enhanced_controls()

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
            ["Side-by-Side Heatmaps", "Overlay Line Profiles", "Multi-Orientation Profiles",
             "Radial Profile Comparison", "Statistical Summary", "Defect-Stress Correlation", 
             "Stress Component Cross-Correlation", "Evolution Timeline", "Contour Comparison", 
             "3D Surface Comparison"],
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
        
        # Enhanced line profile settings
        if comparison_type in ["Overlay Line Profiles", "Multi-Orientation Profiles"]:
            st.sidebar.subheader("📐 Line Profile Settings")
            
            # Profile orientations
            profile_orientations = st.sidebar.multiselect(
                "Profile Orientations",
                ["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)", "135° (Diagonal)", "Custom"],
                default=["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)"]
            )
            
            # Custom angle if selected
            if "Custom" in profile_orientations:
                custom_angle = st.sidebar.slider("Custom Angle (°)", -180, 180, 30, 5)
            
            # Profile length
            profile_length = st.sidebar.slider("Profile Length (% of domain)", 10, 100, 80, 5)
            
            # Line style options
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
            }
            
            # Add type-specific config
            if comparison_type in ["Overlay Line Profiles", "Multi-Orientation Profiles"]:
                st.session_state.comparison_config.update({
                    'profile_orientations': profile_orientations,
                    'profile_length': profile_length,
                    'line_style': comparison_line_style,
                })
                if "Custom" in profile_orientations:
                    st.session_state.comparison_config['custom_angle'] = custom_angle

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
        
        # Create figure with fixed aspect ratio
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.set_constrained_layout(True)
        
        # Apply fixed aspect ratio to domain plots
        FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax1)
        FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax2)
        
        # Initial defect
        im1 = ax1.imshow(init_eta, extent=extent, 
                        cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['eta_cmap'], 'viridis')), 
                        origin='lower')
        ax1.set_title(f"Initial {sim_params['defect_type']} - {sim_params['shape']}")
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Stress preview
        stress_preview = compute_stress_fields(init_eta, sim_params['eps0'], sim_params['theta'])
        im2 = ax2.imshow(stress_preview['sigma_mag'], extent=extent, 
                        cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['sigma_cmap'], 'hot')), 
                        origin='lower')
        ax2.set_title(f"Initial Stress Magnitude")
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Add scale bars
        if advanced_styling.get('show_scale_bar', True):
            FixedAspectVisualizer.add_scale_bar_to_domain(ax1, 5.0, location='lower right')
            FixedAspectVisualizer.add_scale_bar_to_domain(ax2, 5.0, location='lower right')
        
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
                with st.expander("📊 Final Results", expanded=True):
                    final_eta, final_stress = history[-1]
                    
                    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
                    fig2.set_constrained_layout(True)
                    
                    # Apply fixed aspect ratio
                    FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax3)
                    FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax4)
                    
                    # Defect field
                    im3 = ax3.imshow(final_eta, extent=extent, 
                                    cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['eta_cmap'], 'viridis')), 
                                    origin='lower')
                    ax3.set_title(f"Final {sim_params['defect_type']}")
                    ax3.set_xlabel("x (nm)")
                    ax3.set_ylabel("y (nm)")
                    plt.colorbar(im3, ax=ax3, shrink=0.8)
                    
                    # Stress field
                    im4 = ax4.imshow(final_stress['sigma_mag'], extent=extent,
                                    cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['sigma_cmap'], 'viridis')), 
                                    origin='lower')
                    ax4.set_title(f"Final Stress Magnitude")
                    ax4.set_xlabel("x (nm)")
                    ax4.set_ylabel("y (nm)")
                    plt.colorbar(im4, ax=ax4, shrink=0.8)
                    
                    # Add scale bars
                    if advanced_styling.get('show_scale_bar', True):
                        FixedAspectVisualizer.add_scale_bar_to_domain(ax3, 5.0, location='lower right')
                        FixedAspectVisualizer.add_scale_bar_to_domain(ax4, 5.0, location='lower right')
                    
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
                'Steps': params['steps'],
                'Frames': len(SimulationDB.get_simulation(sim['id'])['history'])
            })
        
        df = pd.DataFrame(sim_data)
        st.dataframe(df, use_container_width=True)

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
            
            # Create enhanced comparison
            st.subheader(f"📰 Enhanced {config['type']} Comparison")
            
            # Create enhanced plot
            fig = create_enhanced_comparison_plot(simulations, frames, config, advanced_styling)
            
            # Display with options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pyplot(fig)
            
            with col2:
                # Display aspect ratio info
                domain_aspect = FixedAspectVisualizer.calculate_domain_aspect_ratio()
                st.info(f"""
                **Fixed Aspect Ratio Settings:**
                - **Status**: {'Enabled' if advanced_styling.get('fix_domain_aspect', True) else 'Disabled'}
                - **Domain Aspect**: 1:{domain_aspect:.2f}
                - **Physical Size**: {extent[1]-extent[0]:.1f} × {extent[3]-extent[2]:.1f} nm
                """)
                
                # Profile info for line plots
                if config['type'] in ["Overlay Line Profiles", "Multi-Orientation Profiles"]:
                    orientations = config.get('profile_orientations', [])
                    st.metric("Profile Orientations", len(orientations))
                    st.metric("Profile Length", f"{config.get('profile_length', 80)}%")
            
            # Clear comparison flag
            if 'run_comparison' in st.session_state:
                del st.session_state.run_comparison
    
    else:
        st.info("Select simulations in the sidebar and click 'Run Comparison' to start!")

# =============================================
# ENHANCED EXPORT FUNCTIONALITY
# =============================================
st.sidebar.header("💾 Enhanced Export Options")

with st.sidebar.expander("📥 Advanced Export", expanded=False):
    export_format = st.selectbox(
        "Export Format",
        ["Complete Package (JSON + CSV + PNG)", "JSON Parameters Only", 
         "Publication-Ready Figures", "Profile Data CSV"]
    )
    
    include_aspect_info = st.checkbox("Include Aspect Ratio Information", True)
    include_profiles = st.checkbox("Include Line Profile Data", True)
    
    if st.button("📥 Generate Custom Export", type="primary"):
        simulations = SimulationDB.get_all_simulations()
        
        if not simulations:
            st.sidebar.warning("No simulations to export!")
        else:
            with st.spinner("Creating enhanced export package..."):
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
                        
                        # Export aspect ratio info
                        if include_aspect_info:
                            aspect_info = {
                                'domain_extent': extent,
                                'grid_size': N,
                                'dx': dx,
                                'physical_aspect_ratio': FixedAspectVisualizer.calculate_domain_aspect_ratio(),
                                'fix_aspect_in_plots': advanced_styling.get('fix_domain_aspect', True)
                            }
                            zf.writestr(f"{sim_dir}/aspect_ratio_info.json", json.dumps(aspect_info, indent=2))
                        
                        # Export profile data
                        if include_profiles and 'history' in sim_data:
                            profile_dir = f"{sim_dir}/profiles"
                            for frame_idx, (eta, stress_fields) in enumerate(sim_data['history']):
                                # Extract profiles at standard orientations
                                orientations = [0, 45, 90, 135]
                                for angle in orientations:
                                    distances, profile, endpoints = extract_line_profile(
                                        stress_fields['sigma_mag'], angle, 'center', 0, 0.8
                                    )
                                    
                                    profile_df = pd.DataFrame({
                                        'distance_nm': distances,
                                        'stress_GPa': profile,
                                        'angle_deg': [angle] * len(distances),
                                        'frame': [frame_idx] * len(distances)
                                    })
                                    
                                    zf.writestr(f"{profile_dir}/frame_{frame_idx:04d}_angle_{angle:03d}.csv", 
                                               profile_df.to_csv(index=False))
                    
                    # Create comprehensive summary
                    summary = f"""ENHANCED MULTI-SIMULATION EXPORT SUMMARY
================================================
Generated: {datetime.now().isoformat()}
Total Simulations: {len(simulations)}
Export Format: {export_format}
Include Aspect Info: {include_aspect_info}
Include Profiles: {include_profiles}

FIXED ASPECT RATIO SETTINGS:
----------------------------
Domain Extent: {extent[0]:.1f} to {extent[1]:.1f} nm (x), {extent[2]:.1f} to {extent[3]:.1f} nm (y)
Grid Resolution: {N} × {N} pixels
Pixel Size: {dx} nm
Physical Aspect Ratio: {FixedAspectVisualizer.calculate_domain_aspect_ratio():.3f}
Fixed Aspect in Visualizations: {advanced_styling.get('fix_domain_aspect', True)}

LINE PROFILE EXTRACTION:
------------------------
Profile Extraction Method: High-resolution interpolation
Default Profile Length: 80% of domain
Standard Orientations: 0°, 45°, 90°, 135°
Interpolation Order: Cubic spline
Number of Points per Profile: 500

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
                        summary += f"\n  Fixed Aspect: {advanced_styling.get('fix_domain_aspect', True)}"
                        summary += f"\n  Created: {sim_data['created_at']}\n"
                    
                    zf.writestr("EXPORT_SUMMARY.txt", summary)
                
                buffer.seek(0)
                
                # Determine file name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ag_np_fixed_aspect_export_{timestamp}.zip"
                
                st.sidebar.download_button(
                    "📥 Download Enhanced Export Package",
                    buffer.getvalue(),
                    filename,
                    "application/zip"
                )
                st.sidebar.success("Enhanced export package ready!")

# =============================================
# THEORETICAL ANALYSIS - UPDATED
# =============================================
with st.expander("🔬 Enhanced Theoretical Analysis", expanded=False):
    st.markdown("""
    ### 🎯 **Fixed Aspect Ratio Visualization System**
    
    #### **📐 Key Enhancements:**
    
    **1. Fixed 1:1 Aspect Ratio for Domain Visualization:**
    - **Consistent Spatial Representation**: Domain always appears as a square
    - **Physical Accuracy**: Maintains correct spatial relationships
    - **Scale Preservation**: 1 nm in x-direction equals 1 nm in y-direction
    - **No Distortion**: Eliminates stretching/squashing artifacts
    
    **2. Enhanced Overlay Line Profiles:**
    - **Fixed Layout Grid**: Consistent subplot arrangement
    - **Proper Scaling**: Domain subplot maintains 1:1 aspect ratio
    - **Clear Visualization**: Profile lines show correct spatial relationships
    - **Scale Bar Integration**: Physical scale reference on all domain plots
    
    **3. Improved Multi-Orientation Analysis:**
    - **Accurate Angle Representation**: Diagonal lines show true 45° angles
    - **Consistent Sampling**: Profile extraction respects spatial proportions
    - **Physical Coordinates**: All measurements in nanometers
    
    #### **🔬 Scientific Benefits:**
    
    **Physical Realism:**
    - **True Spatial Relationships**: Stress gradients appear correctly
    - **Accurate Defect Shapes**: Circular/elliptical defects appear undistorted
    - **Proper Scale**: All dimensions maintain physical meaning
    - **Consistent Visualization**: Comparisons between simulations are fair
    
    **Enhanced Analysis:**
    - **Accurate Profile Extraction**: Line profiles follow true spatial paths
    - **Correct Angle Measurement**: Diagonal profiles truly represent 45° lines
    - **Proper Scale Bars**: Provide accurate physical reference
    - **Publication Quality**: Meets journal standards for figure proportions
    
    **Technical Implementation:**
    - **`FixedAspectVisualizer` Class**: Dedicated aspect ratio management
    - **`set_aspect('equal')`**: Enforces 1:1 aspect ratio
    - **`set_xlim()`/`set_ylim()`**: Maintains consistent axis limits
    - **Grid Layout Optimization**: Proper spacing for fixed-aspect subplots
    
    #### **📊 Visualization Improvements:**
    
    **Before (Variable Aspect):**
    - Domain appears stretched or squashed
    - Spatial relationships distorted
    - Scale bars inaccurate
    - Diagonal lines not at true 45°
    
    **After (Fixed 1:1 Aspect):**
    - Domain always appears as square
    - True spatial relationships preserved
    - Accurate scale bars
    - Diagonal lines at correct angles
    - Consistent figure layout
    
    #### **🎯 Key Features:**
    
    **1. Automatic Aspect Ratio Enforcement:**
    ```python
    FixedAspectVisualizer.apply_fixed_aspect_to_domain(ax)
    ```
    
    **2. Consistent Scale Bars:**
    ```python
    FixedAspectVisualizer.add_scale_bar_to_domain(ax, 5.0)
    ```
    
    **3. Optimized Figure Layout:**
    - Proper subplot spacing
    - Consistent figure sizes
    - Balanced composition
    - Publication-ready formatting
    
    **4. Enhanced Profile Visualization:**
    - True spatial positioning of profile lines
    - Accurate angle labeling
    - Physical coordinate display
    - Consistent scaling across all plots
    
    #### **📈 Scientific Impact:**
    
    **For ISF Analysis:**
    - True circular symmetry visualization
    - Accurate stress gradient representation
    - Proper defect shape preservation
    
    **For ESF Analysis:**
    - Correct anisotropic stress patterns
    - Accurate interface visualization
    - True spatial stress distribution
    
    **For Twin Boundary Analysis:**
    - Proper interface angle representation
    - Accurate stress concentration locations
    - True spatial defect morphology
    
    #### **🔧 Technical Implementation Details:**
    
    **Aspect Ratio Enforcement:**
    1. Set axis aspect to 'equal'
    2. Fix x and y limits to domain extent
    3. Use constrained layout for proper spacing
    4. Optimize figure size for square subplots
    
    **Scale Bar Implementation:**
    1. Calculate physical position in data coordinates
    2. Draw line of specified length
    3. Add text label with physical units
    4. Ensure visibility against background
    
    **Profile Line Visualization:**
    1. Calculate true endpoints in physical coordinates
    2. Draw lines at correct angles
    3. Add angle labels at midpoints
    4. Use consistent color coding
    
    #### **📊 Publication Benefits:**
    
    **Journal Compliance:**
    - Proper figure proportions
    - Accurate scale bars
    - Consistent formatting
    - Professional appearance
    
    **Reproducibility:**
    - Fixed aspect ensures consistent output
    - Physical scales always accurate
    - Spatial relationships preserved
    - Comparable across different displays
    
    **Scientific Communication:**
    - Clear spatial representation
    - Accurate physical scales
    - True defect morphology
    - Correct stress distribution
    
    ### **🎯 Summary of Fixed Aspect Ratio System:**
    
    The enhanced system now maintains a consistent 1:1 aspect ratio for all domain visualizations, ensuring:
    
    1. **Physical Accuracy**: True spatial relationships
    2. **Visual Consistency**: No stretching or distortion
    3. **Scientific Validity**: Accurate representation of results
    4. **Publication Quality**: Professional figure formatting
    5. **Enhanced Analysis**: Better interpretation of spatial patterns
    
    **Advanced crystallographic stress analysis platform with fixed aspect ratio visualization for accurate spatial representation!**
    """)
    
    # Display platform statistics
    simulations = SimulationDB.get_all_simulations()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulations", len(simulations))
    with col2:
        domain_aspect = FixedAspectVisualizer.calculate_domain_aspect_ratio()
        st.metric("Domain Aspect Ratio", f"1:{domain_aspect:.2f}")
    with col3:
        st.metric("Profile Orientations", "0°, 45°, 90°, 135° + Custom")
    with col4:
        st.metric("Fixed Aspect", "✅ Enabled")

# =============================================
# FOOTER
# =============================================
st.caption("""
🔬 Enhanced Multi-Defect Comparison Platform • Fixed Aspect Ratio Visualization • 
Multi-Orientation Line Profiles • Publication-Quality Output • 2025
""")
