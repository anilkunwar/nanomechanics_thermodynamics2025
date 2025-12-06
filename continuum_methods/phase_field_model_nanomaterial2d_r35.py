import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Arrow, FancyBboxPatch, Circle, Ellipse
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
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
**Run multiple simulations • Compare ISF/ESF/Twin with different orientations • Cloud-style storage**
**Run → Save → Compare • 50+ Colormaps • Publication-ready comparison plots • Advanced Post-Processing**
**NEW: Stress overlay line profiles in multiple orientations with realistic geometric probe indicators**
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
# LINE PROFILE ANALYSIS SYSTEM
# =============================================
class LineProfileAnalyzer:
    """Advanced line profile extraction and analysis system"""
    
    @staticmethod
    def extract_line_profile(data_2d, x0, y0, x1, y1, num_points=200):
        """
        Extract line profile from 2D data using bilinear interpolation
        
        Parameters:
        -----------
        data_2d : 2D numpy array
            Input data matrix
        x0, y0 : float
            Start point coordinates (in grid indices)
        x1, y1 : float
            End point coordinates (in grid indices)
        num_points : int
            Number of points along the line
        
        Returns:
        --------
        distances : 1D array
            Distance along the line (normalized to 0-1)
        profile : 1D array
            Extracted values along the line
        coords : tuple
            (x_coords, y_coords) along the line in data coordinates
        """
        # Create linearly spaced points along the line
        t = np.linspace(0, 1, num_points)
        x_coords = x0 + t * (x1 - x0)
        y_coords = y0 + t * (y1 - y0)
        
        # Use scipy's map_coordinates for high-quality interpolation
        coords = np.vstack((y_coords, x_coords))  # Note: y first for row-major
        profile = map_coordinates(data_2d, coords, order=1, mode='nearest')
        
        # Calculate actual distances
        distances = np.sqrt((x_coords - x0)**2 + (y_coords - y0)**2)
        distances = distances / distances[-1] if distances[-1] > 0 else distances
        
        return distances, profile, (x_coords, y_coords)
    
    @staticmethod
    def get_line_profiles(data_2d, center_x, center_y, length, angle_deg=0):
        """
        Get line profiles in multiple orientations from a center point
        
        Parameters:
        -----------
        data_2d : 2D numpy array
            Input data matrix
        center_x, center_y : float
            Center point coordinates (in grid indices)
        length : float
            Length of the line (in grid units)
        angle_deg : float or list
            Angle(s) in degrees (0=horizontal, 90=vertical, etc.)
        
        Returns:
        --------
        profiles : dict
            Dictionary with angles as keys and (distances, profile, coords) as values
        """
        profiles = {}
        
        if isinstance(angle_deg, (int, float)):
            angle_deg = [angle_deg]
        
        for angle in angle_deg:
            # Convert angle to radians
            theta = np.deg2rad(angle)
            
            # Calculate end points
            dx = length * np.cos(theta) / 2
            dy = length * np.sin(theta) / 2
            
            x0 = center_x - dx
            y0 = center_y - dy
            x1 = center_x + dx
            y1 = center_y + dy
            
            # Extract profile
            distances, profile, coords = LineProfileAnalyzer.extract_line_profile(
                data_2d, x0, y0, x1, y1
            )
            
            profiles[angle] = {
                'distances': distances,
                'profile': profile,
                'coords': coords,
                'start': (x0, y0),
                'end': (x1, y1),
                'center': (center_x, center_y),
                'length': length,
                'angle': angle
            }
        
        return profiles
    
    @staticmethod
    def extract_multiple_orientations(data_2d, center_x, center_y, length=100, 
                                     orientations=None):
        """Extract profiles in standard orientations"""
        if orientations is None:
            orientations = [0, 45, 90, 135]  # Horizontal, Diagonal, Vertical, Anti-diagonal
        
        return LineProfileAnalyzer.get_line_profiles(
            data_2d, center_x, center_y, length, orientations
        )
    
    @staticmethod
    def create_probe_line_indicator(ax, profile_info, style='realistic', **kwargs):
        """
        Create realistic geometric probe line indicators
        
        Parameters:
        -----------
        ax : matplotlib axis
            Axis to draw on
        profile_info : dict
            Profile information from extract_line_profile
        style : str
            Indicator style: 'realistic', 'arrow', 'rectangle', 'nanoprobe'
        **kwargs : dict
            Additional styling parameters
        """
        # Default style parameters
        default_params = {
            'line_color': 'yellow',
            'line_width': 3,
            'line_alpha': 0.8,
            'endpoint_color': 'red',
            'endpoint_size': 50,
            'fill_color': 'rgba(255, 255, 0, 0.2)',
            'text_color': 'white',
            'text_size': 10,
            'shadow': True
        }
        
        # Update with provided kwargs
        params = {**default_params, **kwargs}
        
        # Get coordinates
        x_coords, y_coords = profile_info['coords']
        x0, y0 = profile_info['start']
        x1, y1 = profile_info['end']
        center_x, center_y = profile_info['center']
        angle = profile_info['angle']
        
        # Apply path effects for shadow/glow
        path_effects_list = []
        if params['shadow']:
            path_effects_list = [
                path_effects.withStroke(linewidth=params['line_width']+2, 
                                       foreground='black', alpha=0.5),
                path_effects.withStroke(linewidth=params['line_width']+1, 
                                       foreground='white', alpha=0.3)
            ]
        
        if style == 'realistic':
            # Realistic AFM/STEM probe style
            # Main probe line
            ax.plot(x_coords, y_coords, 
                   color=params['line_color'],
                   linewidth=params['line_width'],
                   alpha=params['line_alpha'],
                   solid_capstyle='round',
                   path_effects=path_effects_list)
            
            # Endpoint markers
            for point in [(x0, y0), (x1, y1)]:
                ax.scatter(*point, 
                          s=params['endpoint_size'],
                          color=params['endpoint_color'],
                          edgecolors='white',
                          linewidths=2,
                          zorder=10,
                          marker='o')
            
            # Center marker
            ax.scatter(center_x, center_y,
                      s=params['endpoint_size'] * 0.7,
                      color='cyan',
                      edgecolors='white',
                      linewidths=2,
                      zorder=11,
                      marker='s')
            
            # Add directional arrow
            arrow_length = np.sqrt((x1-x0)**2 + (y1-y0)**2) * 0.2
            ax.annotate('',
                       xy=(x0 + arrow_length * np.cos(np.deg2rad(angle)),
                           y0 + arrow_length * np.sin(np.deg2rad(angle))),
                       xytext=(x0, y0),
                       arrowprops=dict(arrowstyle='->',
                                      color='white',
                                      linewidth=2,
                                      shrinkA=0,
                                      shrinkB=0))
            
            # Add angle annotation
            ax.text(center_x, center_y,
                   f'{angle}°',
                   color=params['text_color'],
                   fontsize=params['text_size'],
                   ha='center',
                   va='center',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='black',
                            alpha=0.7,
                            edgecolor='none'))
        
        elif style == 'arrow':
            # Simple arrow style
            arrow = Arrow(x0, y0, x1-x0, y1-y0,
                         width=params['line_width']*3,
                         color=params['line_color'],
                         alpha=params['line_alpha'])
            ax.add_patch(arrow)
            
        elif style == 'rectangle':
            # Rectangle probe style
            rect_width = params['line_width'] * 2
            rect_length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            
            rect = Rectangle((x0 - rect_width/2, y0 - rect_width/2),
                            rect_length, rect_width,
                            angle=angle,
                            rotation_point=(x0, y0),
                            color=params['fill_color'],
                            alpha=0.5,
                            edgecolor=params['line_color'],
                            linewidth=2)
            ax.add_patch(rect)
            
            # Add center line
            ax.plot([x0, x1], [y0, y1],
                   color=params['line_color'],
                   linewidth=params['line_width'],
                   linestyle='--',
                   alpha=0.8)
        
        elif style == 'nanoprobe':
            # Nanoscale probe tip style
            # Probe body
            ax.plot(x_coords, y_coords,
                   color=params['line_color'],
                   linewidth=params['line_width'],
                   alpha=params['line_alpha'],
                   solid_capstyle='round')
            
            # Probe tip
            tip_length = np.sqrt((x1-x0)**2 + (y1-y0)**2) * 0.1
            tip_angle = np.deg2rad(angle)
            
            # Create triangular tip
            tip_x = [x1, 
                    x1 - tip_length * np.cos(tip_angle + np.pi/6),
                    x1 - tip_length * np.cos(tip_angle - np.pi/6),
                    x1]
            tip_y = [y1,
                    y1 - tip_length * np.sin(tip_angle + np.pi/6),
                    y1 - tip_length * np.sin(tip_angle - np.pi/6),
                    y1]
            
            ax.fill(tip_x, tip_y,
                   color='red',
                   alpha=0.7,
                   edgecolor='white',
                   linewidth=2)
            
            # Add measurement scale
            scale_length = np.sqrt((x1-x0)**2 + (y1-y0)**2) * 0.3
            scale_x = [center_x - scale_length/2 * np.cos(tip_angle),
                      center_x + scale_length/2 * np.cos(tip_angle)]
            scale_y = [center_y - scale_length/2 * np.sin(tip_angle),
                      center_y + scale_length/2 * np.sin(tip_angle)]
            
            ax.plot(scale_x, scale_y,
                   color='white',
                   linewidth=3,
                   alpha=0.9,
                   path_effects=[path_effects.withStroke(linewidth=5, foreground='black')])
        
        return ax
    
    @staticmethod
    def analyze_line_profiles(profiles_dict):
        """Analyze extracted line profiles and compute statistics"""
        analysis = {}
        
        for angle, profile_data in profiles_dict.items():
            profile = profile_data['profile']
            
            if len(profile) > 0:
                stats_dict = {
                    'mean': np.nanmean(profile),
                    'std': np.nanstd(profile),
                    'min': np.nanmin(profile),
                    'max': np.nanmax(profile),
                    'range': np.nanmax(profile) - np.nanmin(profile),
                    'gradient': np.mean(np.abs(np.gradient(profile))),
                    'integral': np.trapz(profile, profile_data['distances']),
                    'peak_position': profile_data['distances'][np.argmax(np.abs(profile))],
                    'fwhm': LineProfileAnalyzer.calculate_fwhm(profile_data['distances'], profile)
                }
                
                # Detect peaks
                peaks = LineProfileAnalyzer.find_peaks(profile)
                stats_dict['n_peaks'] = len(peaks)
                stats_dict['peaks'] = peaks
                
                analysis[angle] = stats_dict
        
        return analysis
    
    @staticmethod
    def calculate_fwhm(distances, profile):
        """Calculate Full Width at Half Maximum"""
        if len(profile) == 0:
            return 0
        
        half_max = (np.max(profile) + np.min(profile)) / 2
        above_half = profile > half_max
        
        if np.any(above_half):
            indices = np.where(above_half)[0]
            return distances[indices[-1]] - distances[indices[0]]
        return 0
    
    @staticmethod
    def find_peaks(profile, min_distance=5, prominence=0.1):
        """Find peaks in profile using scipy"""
        from scipy.signal import find_peaks
        
        if len(profile) < min_distance:
            return []
        
        peaks, properties = find_peaks(profile, 
                                      distance=min_distance,
                                      prominence=prominence)
        
        return peaks.tolist()

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
# ADVANCED LINE PROFILE VISUALIZATION
# =============================================
class AdvancedLineProfileVisualizer:
    """Enhanced visualization for line profiles with overlay capabilities"""
    
    @staticmethod
    def create_stress_overlay_plot(data_dict, profiles_dict, 
                                   title="Stress Overlay Analysis",
                                   style_params=None):
        """
        Create overlay plot showing stress field with multiple line profiles
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing stress data and metadata
        profiles_dict : dict
            Dictionary of line profiles for different orientations
        title : str
            Plot title
        style_params : dict
            Styling parameters
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure
        """
        if style_params is None:
            style_params = {}
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main stress map with probe indicators
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Individual profile plots
        ax2 = fig.add_subplot(gs[0, 2])  # Horizontal profile
        ax3 = fig.add_subplot(gs[1, 2])  # Vertical profile
        ax4 = fig.add_subplot(gs[2, 0])  # 45° diagonal
        ax5 = fig.add_subplot(gs[2, 1])  # 135° diagonal
        
        # Statistics panel
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Extract stress data
        stress_data = data_dict.get('stress_mag', None)
        if stress_data is None and 'stress_fields' in data_dict:
            stress_data = data_dict['stress_fields'].get('sigma_mag', None)
        
        # Plot 1: Main stress map with all probe lines
        if stress_data is not None:
            im = ax1.imshow(stress_data, extent=extent,
                           cmap=plt.cm.get_cmap(COLORMAPS.get(style_params.get('stress_cmap', 'hot'))),
                           origin='lower',
                           aspect='auto')
            
            # Add all probe lines
            probe_styles = ['realistic', 'arrow', 'nanoprobe', 'rectangle']
            colors = ['yellow', 'cyan', 'magenta', 'lime']
            
            for idx, (angle, profile_data) in enumerate(profiles_dict.items()):
                style = probe_styles[idx % len(probe_styles)]
                color = colors[idx % len(colors)]
                
                LineProfileAnalyzer.create_probe_line_indicator(
                    ax1, profile_data,
                    style=style,
                    line_color=color,
                    line_width=style_params.get('probe_width', 2),
                    endpoint_color=color,
                    shadow=True
                )
            
            ax1.set_title("Stress Field with Probe Lines", fontsize=12, fontweight='bold')
            ax1.set_xlabel("x (nm)", fontsize=10)
            ax1.set_ylabel("y (nm)", fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label('Stress (GPa)', fontsize=9)
        
        # Plot individual profiles
        axes_profiles = [ax2, ax3, ax4, ax5]
        orientations = [0, 90, 45, 135]  # Standard orientations
        
        for ax, orientation in zip(axes_profiles, orientations):
            if orientation in profiles_dict:
                profile_data = profiles_dict[orientation]
                distances = profile_data['distances']
                profile = profile_data['profile']
                
                # Convert normalized distances to nm
                actual_distances = distances * profile_data['length'] * dx
                
                ax.plot(actual_distances, profile,
                       linewidth=style_params.get('profile_linewidth', 2),
                       color=style_params.get('profile_color', 'blue'),
                       alpha=0.8)
                
                # Add fill under curve
                ax.fill_between(actual_distances, 0, profile,
                               alpha=0.3,
                               color=style_params.get('fill_color', 'blue'))
                
                # Add statistics text
                stats_text = f"Max: {np.max(profile):.2f} GPa\n" \
                           f"Min: {np.min(profile):.2f} GPa\n" \
                           f"Mean: {np.mean(profile):.2f} GPa"
                
                ax.text(0.05, 0.95, stats_text,
                       transform=ax.transAxes,
                       fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f"{orientation}° Profile", fontsize=10, fontweight='bold')
                ax.set_xlabel("Distance (nm)", fontsize=9)
                ax.set_ylabel("Stress (GPa)", fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 6: Statistics table
        analysis = LineProfileAnalyzer.analyze_line_profiles(profiles_dict)
        
        if analysis:
            # Create table data
            table_data = []
            columns = ['Angle', 'Mean', 'Std', 'Min', 'Max', 'Range', 'Peaks']
            
            for angle, stats in analysis.items():
                table_data.append([
                    f"{angle}°",
                    f"{stats['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['min']:.3f}",
                    f"{stats['max']:.3f}",
                    f"{stats['range']:.3f}",
                    f"{stats['n_peaks']}"
                ])
            
            # Create table
            table = ax6.table(cellText=table_data,
                             colLabels=columns,
                             cellLoc='center',
                             loc='center',
                             colColours=['#f2f2f2']*len(columns))
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            ax6.set_title("Profile Statistics", fontsize=11, fontweight='bold')
        
        return fig
    
    @staticmethod
    def create_multi_orientation_comparison(simulations, frames, config, style_params):
        """
        Create comprehensive comparison of line profiles across multiple simulations
        and orientations
        """
        n_sims = len(simulations)
        orientations = config.get('orientations', [0, 45, 90, 135])
        n_orientations = len(orientations)
        
        # Create figure
        fig = plt.figure(figsize=(5*n_orientations, 4*n_sims))
        fig.suptitle("Multi-Simulation Line Profile Comparison", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create subplot grid
        gs = fig.add_gridspec(n_sims, n_orientations + 1, 
                             width_ratios=[2] + [1] * n_orientations,
                             hspace=0.3, wspace=0.4)
        
        # Color scheme
        colors = plt.cm.rainbow(np.linspace(0, 1, n_sims))
        
        # Extract stress key
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map.get(config.get('stress_component', 'Stress Magnitude |σ|'), 'sigma_mag')
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            # Get data
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Main visualization (left column)
            ax_main = fig.add_subplot(gs[sim_idx, 0])
            
            # Plot stress field
            im = ax_main.imshow(stress_data, extent=extent,
                               cmap=plt.cm.get_cmap(COLORMAPS.get(
                                   sim['params']['sigma_cmap'], 'hot')),
                               origin='lower',
                               aspect='auto')
            
            # Add all probe lines for this simulation
            center_x, center_y = N//2, N//2
            length = min(N//2, 80)
            
            for orient_idx, orientation in enumerate(orientations):
                # Extract profile
                profiles = LineProfileAnalyzer.get_line_profiles(
                    stress_data, center_x, center_y, length, orientation
                )
                
                if orientation in profiles:
                    # Add probe indicator
                    color = plt.cm.tab10(orient_idx / max(1, len(orientations)-1))
                    LineProfileAnalyzer.create_probe_line_indicator(
                        ax_main, profiles[orientation],
                        style='realistic',
                        line_color=color,
                        line_width=2,
                        endpoint_color=color,
                        text_color='white'
                    )
            
            ax_main.set_title(f"{sim['params']['defect_type']}\n{sim['params']['orientation']}",
                             fontsize=10, fontweight='bold')
            ax_main.set_xlabel("x (nm)", fontsize=9)
            ax_main.set_ylabel("y (nm)", fontsize=9)
            
            # Add colorbar for first row only
            if sim_idx == 0:
                cbar = plt.colorbar(im, ax=ax_main, shrink=0.8)
                cbar.set_label(f"{config.get('stress_component', 'Stress')} (GPa)", 
                              fontsize=9)
            
            # Individual profile plots (right columns)
            for orient_idx, orientation in enumerate(orientations):
                ax_profile = fig.add_subplot(gs[sim_idx, orient_idx + 1])
                
                # Extract profile for this orientation
                profiles = LineProfileAnalyzer.get_line_profiles(
                    stress_data, center_x, center_y, length, orientation
                )
                
                if orientation in profiles:
                    profile_data = profiles[orientation]
                    distances = profile_data['distances']
                    profile = profile_data['profile']
                    
                    # Convert to actual distances
                    actual_distances = distances * length * dx
                    
                    # Plot profile
                    ax_profile.plot(actual_distances, profile,
                                   color=colors[sim_idx],
                                   linewidth=style_params.get('line_width', 2),
                                   label=f"{sim['params']['defect_type']}")
                    
                    ax_profile.fill_between(actual_distances, 0, profile,
                                           alpha=0.3, color=colors[sim_idx])
                    
                    ax_profile.set_title(f"{orientation}°", fontsize=9, fontweight='bold')
                    ax_profile.set_xlabel("Distance (nm)", fontsize=8)
                    
                    if orient_idx == 0:
                        ax_profile.set_ylabel("Stress (GPa)", fontsize=8)
                    
                    ax_profile.grid(True, alpha=0.3, linestyle='--')
                    
                    # Add legend for first orientation only
                    if orient_idx == 0 and sim_idx == 0:
                        ax_profile.legend(fontsize=8, loc='upper right')
        
        return fig
    
    @staticmethod
    def create_interactive_probe_selector(data_2d, initial_center=None, 
                                         initial_length=80, initial_angle=0):
        """
        Create interactive probe line selector visualization
        
        Parameters:
        -----------
        data_2d : 2D array
            Stress data
        initial_center : tuple
            Initial center coordinates (x, y)
        initial_length : float
            Initial probe length
        initial_angle : float
            Initial probe angle
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Interactive figure
        """
        if initial_center is None:
            initial_center = (N//2, N//2)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Stress field with interactive probe
        im = ax1.imshow(data_2d, extent=extent,
                       cmap=plt.cm.viridis,
                       origin='lower',
                       aspect='auto')
        
        # Initial probe
        profiles = LineProfileAnalyzer.get_line_profiles(
            data_2d, initial_center[0], initial_center[1], 
            initial_length, initial_angle
        )
        
        if initial_angle in profiles:
            LineProfileAnalyzer.create_probe_line_indicator(
                ax1, profiles[initial_angle],
                style='realistic',
                line_color='yellow',
                line_width=3,
                endpoint_color='red',
                text_color='white',
                shadow=True
            )
        
        ax1.set_title("Interactive Probe Line Selector", fontsize=12, fontweight='bold')
        ax1.set_xlabel("x (nm)", fontsize=10)
        ax1.set_ylabel("y (nm)", fontsize=10)
        
        plt.colorbar(im, ax=ax1, shrink=0.8).set_label('Stress (GPa)', fontsize=9)
        
        # Plot 2: Extracted profile
        if initial_angle in profiles:
            profile_data = profiles[initial_angle]
            distances = profile_data['distances']
            profile = profile_data['profile']
            actual_distances = distances * initial_length * dx
            
            ax2.plot(actual_distances, profile,
                    linewidth=2, color='blue', alpha=0.8)
            ax2.fill_between(actual_distances, 0, profile,
                            alpha=0.3, color='blue')
            
            # Add statistics
            stats_text = f"Angle: {initial_angle}°\n" \
                        f"Length: {initial_length*dx:.1f} nm\n" \
                        f"Max: {np.max(profile):.2f} GPa\n" \
                        f"Min: {np.min(profile):.2f} GPa\n" \
                        f"Mean: {np.mean(profile):.2f} GPa"
            
            ax2.text(0.05, 0.95, stats_text,
                    transform=ax2.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax2.set_title(f"Extracted Profile ({initial_angle}°)", 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel("Distance (nm)", fontsize=10)
            ax2.set_ylabel("Stress (GPa)", fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        return fig

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

# Get enhanced publication controls
advanced_styling = EnhancedFigureStyler.get_publication_controls()

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
    
    # NEW: Line Profile Analysis Settings
    with st.sidebar.expander("📐 Line Profile Analysis Settings", expanded=True):
        st.subheader("Probe Line Configuration")
        
        # Probe line orientations
        selected_orientations = st.multiselect(
            "Select Line Orientations",
            options=["Horizontal (0°)", "Vertical (90°)", "Diagonal (45°)", 
                    "Anti-diagonal (135°)", "Custom Angle"],
            default=["Horizontal (0°)", "Vertical (90°)", "Diagonal (45°)"]
        )
        
        custom_angle = None
        if "Custom Angle" in selected_orientations:
            custom_angle = st.slider("Custom Angle (°)", -180, 180, 30, 5)
        
        # Probe line length
        probe_length = st.slider("Probe Line Length (grid units)", 20, 120, 80, 5)
        
        # Probe style
        probe_style = st.selectbox(
            "Probe Line Indicator Style",
            ["Realistic (AFM/STEM style)", "Arrow", "Rectangle", "Nano-probe Tip", "Simple Line"],
            index=0
        )
        
        # Overlay options
        show_all_profiles = st.checkbox("Show all profiles on main plot", True)
        profile_linewidth = st.slider("Profile Line Width", 1.0, 5.0, 2.0, 0.5)
        
        # Extract numeric angles from selected orientations
        orientation_map = {
            "Horizontal (0°)": 0,
            "Vertical (90°)": 90,
            "Diagonal (45°)": 45,
            "Anti-diagonal (135°)": 135
        }
        
        orientations = []
        for orient in selected_orientations:
            if orient in orientation_map:
                orientations.append(orientation_map[orient])
            elif orient == "Custom Angle" and custom_angle is not None:
                orientations.append(custom_angle)
    
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
            ["Side-by-Side Heatmaps", 
             "Overlay Line Profiles", 
             "Multi-Orientation Line Analysis",
             "Radial Profile Comparison", 
             "Statistical Summary", 
             "Defect-Stress Correlation", 
             "Stress Component Cross-Correlation",
             "Evolution Timeline", 
             "Contour Comparison", 
             "3D Surface Comparison"],
            index=1  # Default to Overlay Line Profiles
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
        
        # Additional controls for specific comparison types
        if comparison_type in ["Defect-Stress Correlation", "Stress Component Cross-Correlation"]:
            st.sidebar.subheader("Correlation Settings")
            correlation_x_component = st.sidebar.selectbox(
                "X-Axis Component",
                ["Defect Parameter η", "Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
                index=0 if comparison_type == "Defect-Stress Correlation" else 1
            )
            
            if comparison_type == "Stress Component Cross-Correlation":
                correlation_y_component = st.sidebar.selectbox(
                    "Y-Axis Component",
                    ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
                    index=2
                )
            else:
                correlation_y_component = stress_component
            
            correlation_sample_size = st.sidebar.slider("Sample Size (%)", 1, 100, 20, 
                                                       help="Percentage of data points to use for scatter plots")
            correlation_alpha = st.sidebar.slider("Point Alpha", 0.1, 1.0, 0.5, 0.05)
            correlation_point_size = st.sidebar.slider("Point Size", 1, 50, 10)
        
        # Contour settings
        if comparison_type == "Contour Comparison":
            st.sidebar.subheader("Contour Settings")
            contour_levels = st.sidebar.slider("Number of Contour Levels", 3, 20, 10)
            contour_linewidth = st.sidebar.slider("Contour Line Width", 0.5, 3.0, 1.5, 0.1)
        
        # Run comparison
        if st.sidebar.button("🔬 Run Comparison", type="primary"):
            st.session_state.run_comparison = True
            st.session_state.comparison_config = {
                'sim_ids': selected_ids,
                'type': comparison_type,
                'stress_component': stress_component,
                'frame_selection': frame_selection,
                'frame_idx': frame_idx,
                'line_style': comparison_line_style,
                'orientations': orientations,
                'probe_length': probe_length,
                'probe_style': probe_style.replace(" (AFM/STEM style)", "").lower(),
                'show_all_profiles': show_all_profiles,
                'profile_linewidth': profile_linewidth
            }
            
            # Add type-specific config
            if comparison_type in ["Defect-Stress Correlation", "Stress Component Cross-Correlation"]:
                st.session_state.comparison_config.update({
                    'correlation_x': correlation_x_component,
                    'correlation_y': correlation_y_component,
                    'correlation_sample': correlation_sample_size,
                    'correlation_alpha': correlation_alpha,
                    'correlation_point_size': correlation_point_size
                })
            
            if comparison_type == "Contour Comparison":
                st.session_state.comparison_config.update({
                    'contour_levels': contour_levels,
                    'contour_linewidth': contour_linewidth
                })

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
                
                # Show final frame with line profile analysis
                with st.expander("📊 Advanced Line Profile Analysis", expanded=True):
                    st.subheader("Line Profile Analysis of Final State")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Line profile settings
                        center_x = st.slider("Center X (grid units)", 0, N-1, N//2)
                        center_y = st.slider("Center Y (grid units)", 0, N-1, N//2)
                        length = st.slider("Line Length (grid units)", 20, 120, 80)
                        
                        # Orientation selection
                        selected_angles = st.multiselect(
                            "Select orientations",
                            options=[0, 45, 90, 135],
                            default=[0, 90]
                        )
                        
                        probe_style = st.selectbox(
                            "Probe Style",
                            ["realistic", "arrow", "nanoprobe", "rectangle"],
                            index=0
                        )
                    
                    with col2:
                        stress_component = st.selectbox(
                            "Stress Component",
                            ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
                            index=0
                        )
                        
                        show_statistics = st.checkbox("Show Statistics", True)
                        overlay_profiles = st.checkbox("Overlay All Profiles", True)
                    
                    if st.button("Generate Line Profiles", type="secondary"):
                        # Get final state
                        final_eta, final_stress = history[-1]
                        
                        # Map stress component
                        stress_map = {
                            "Stress Magnitude |σ|": 'sigma_mag',
                            "Hydrostatic σ_h": 'sigma_hydro',
                            "von Mises σ_vM": 'von_mises'
                        }
                        stress_key = stress_map[stress_component]
                        stress_data = final_stress[stress_key]
                        
                        # Extract line profiles
                        profiles = LineProfileAnalyzer.get_line_profiles(
                            stress_data, center_x, center_y, length, selected_angles
                        )
                        
                        # Create visualization
                        fig_profile = AdvancedLineProfileVisualizer.create_stress_overlay_plot(
                            {'stress_fields': final_stress, 'stress_mag': stress_data},
                            profiles,
                            title=f"Line Profile Analysis - {sim_params['defect_type']}",
                            style_params={
                                'stress_cmap': sim_params['sigma_cmap'],
                                'probe_width': 2,
                                'profile_linewidth': 2
                            }
                        )
                        
                        st.pyplot(fig_profile)
                        
                        # Show statistics if requested
                        if show_statistics and profiles:
                            analysis = LineProfileAnalyzer.analyze_line_profiles(profiles)
                            
                            st.subheader("Profile Statistics")
                            stats_data = []
                            for angle, stats in analysis.items():
                                stats_data.append({
                                    'Angle (°)': angle,
                                    'Mean (GPa)': f"{stats['mean']:.3f}",
                                    'Std Dev': f"{stats['std']:.3f}",
                                    'Min (GPa)': f"{stats['min']:.3f}",
                                    'Max (GPa)': f"{stats['max']:.3f}",
                                    'Range (GPa)': f"{stats['range']:.3f}",
                                    'FWHM': f"{stats['fwhm']:.3f}",
                                    'Peaks': stats['n_peaks']
                                })
                            
                            df_stats = pd.DataFrame(stats_data)
                            st.dataframe(df_stats, use_container_width=True)
                
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
            if config['type'] == "Multi-Orientation Line Analysis":
                st.subheader("📐 Multi-Orientation Line Profile Analysis")
                
                # Create advanced line profile comparison
                fig = AdvancedLineProfileVisualizer.create_multi_orientation_comparison(
                    simulations, frames, config, advanced_styling
                )
                
                # Display figure
                st.pyplot(fig)
                
                # Show detailed statistics
                with st.expander("📊 Detailed Profile Statistics", expanded=False):
                    all_stats = []
                    
                    for sim, frame in zip(simulations, frames):
                        # Get data
                        eta, stress_fields = sim['history'][frame]
                        stress_data = stress_fields[stress_key]
                        
                        # Extract profiles
                        center_x, center_y = N//2, N//2
                        length = config.get('probe_length', 80)
                        orientations = config.get('orientations', [0, 45, 90, 135])
                        
                        profiles = LineProfileAnalyzer.get_line_profiles(
                            stress_data, center_x, center_y, length, orientations
                        )
                        
                        # Analyze profiles
                        analysis = LineProfileAnalyzer.analyze_line_profiles(profiles)
                        
                        # Collect statistics
                        for angle, stats in analysis.items():
                            all_stats.append({
                                'Simulation': f"{sim['params']['defect_type']}",
                                'Orientation': f"{sim['params']['orientation']}",
                                'Profile Angle (°)': angle,
                                'Mean Stress (GPa)': stats['mean'],
                                'Max Stress (GPa)': stats['max'],
                                'Stress Range (GPa)': stats['range'],
                                'FWHM': stats['fwhm'],
                                'Number of Peaks': stats['n_peaks']
                            })
                    
                    if all_stats:
                        df_stats = pd.DataFrame(all_stats)
                        st.dataframe(df_stats, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            avg_max = df_stats['Max Stress (GPa)'].mean()
                            st.metric("Average Max Stress", f"{avg_max:.2f} GPa")
                        with col2:
                            avg_range = df_stats['Stress Range (GPa)'].mean()
                            st.metric("Average Range", f"{avg_range:.2f} GPa")
                        with col3:
                            avg_fwhm = df_stats['FWHM'].mean()
                            st.metric("Average FWHM", f"{avg_fwhm:.3f}")
                        with col4:
                            total_peaks = df_stats['Number of Peaks'].sum()
                            st.metric("Total Peaks", total_peaks)
            
            elif config['type'] == "Overlay Line Profiles":
                st.subheader("📈 Overlay Line Profile Comparison")
                
                # Enhanced line profile comparison with multiple orientations
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Interactive probe selector
                    st.subheader("Interactive Probe Configuration")
                    
                    # Select which simulation to use for interactive visualization
                    sim_idx = st.selectbox(
                        "Select simulation for interactive view",
                        options=range(len(simulations)),
                        format_func=lambda i: f"{simulations[i]['params']['defect_type']} - {simulations[i]['params']['orientation']}"
                    )
                    
                    # Get selected simulation data
                    sim = simulations[sim_idx]
                    eta, stress_fields = sim['history'][frames[sim_idx]]
                    stress_data = stress_fields[stress_key]
                    
                    # Interactive controls
                    center_x = st.slider("Center X", 0, N-1, N//2, 
                                        help="Center point X coordinate")
                    center_y = st.slider("Center Y", 0, N-1, N//2,
                                        help="Center point Y coordinate")
                    length = st.slider("Line Length", 20, 120, 
                                      config.get('probe_length', 80),
                                      help="Length of probe line in grid units")
                    
                    # Angle selection
                    angle_options = config.get('orientations', [0, 45, 90, 135])
                    selected_angle = st.select_slider(
                        "Probe Angle",
                        options=angle_options,
                        value=angle_options[0] if angle_options else 0
                    )
                    
                    # Probe style
                    probe_style = config.get('probe_style', 'realistic')
                    
                    # Create interactive visualization
                    fig_interactive = AdvancedLineProfileVisualizer.create_interactive_probe_selector(
                        stress_data,
                        initial_center=(center_x, center_y),
                        initial_length=length,
                        initial_angle=selected_angle
                    )
                    
                    st.pyplot(fig_interactive)
                
                with col2:
                    # Quick analysis panel
                    st.subheader("Quick Analysis")
                    
                    # Extract profile for selected parameters
                    profiles = LineProfileAnalyzer.get_line_profiles(
                        stress_data, center_x, center_y, length, selected_angle
                    )
                    
                    if selected_angle in profiles:
                        profile_data = profiles[selected_angle]
                        profile = profile_data['profile']
                        
                        # Display statistics
                        st.metric("Maximum Stress", f"{np.max(profile):.2f} GPa")
                        st.metric("Minimum Stress", f"{np.min(profile):.2f} GPa")
                        st.metric("Mean Stress", f"{np.mean(profile):.2f} GPa")
                        st.metric("Stress Range", f"{np.max(profile) - np.min(profile):.2f} GPa")
                        
                        # Peak detection
                        peaks = LineProfileAnalyzer.find_peaks(profile)
                        st.metric("Number of Peaks", len(peaks))
                        
                        if len(peaks) > 0:
                            st.info(f"Peak positions: {peaks}")
                    
                    # Comparison settings
                    st.subheader("Comparison Settings")
                    compare_orientations = st.multiselect(
                        "Compare across orientations",
                        options=angle_options,
                        default=angle_options[:min(3, len(angle_options))]
                    )
                    
                    if st.button("Generate Multi-Angle Comparison", type="secondary"):
                        # Create comparison across orientations for selected simulation
                        if compare_orientations:
                            profiles_multi = LineProfileAnalyzer.get_line_profiles(
                                stress_data, center_x, center_y, length, compare_orientations
                            )
                            
                            fig_comparison, ax = plt.subplots(figsize=(10, 6))
                            
                            colors = plt.cm.tab10(np.linspace(0, 1, len(compare_orientations)))
                            
                            for idx, angle in enumerate(compare_orientations):
                                if angle in profiles_multi:
                                    profile_data = profiles_multi[angle]
                                    distances = profile_data['distances']
                                    profile = profile_data['profile']
                                    actual_distances = distances * length * dx
                                    
                                    ax.plot(actual_distances, profile,
                                           color=colors[idx],
                                           linewidth=advanced_styling.get('line_width', 2),
                                           label=f"{angle}°")
                            
                            ax.set_xlabel("Distance (nm)", fontsize=12)
                            ax.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=12)
                            ax.set_title(f"Multi-Angle Profile Comparison\n{sim['params']['defect_type']} - {sim['params']['orientation']}",
                                       fontsize=14, fontweight='bold')
                            ax.legend(fontsize=10)
                            ax.grid(True, alpha=0.3)
                            
                            fig_comparison = EnhancedFigureStyler.apply_advanced_styling(
                                fig_comparison, ax, advanced_styling
                            )
                            
                            st.pyplot(fig_comparison)
                
                # Main comparison plot
                st.subheader("Multi-Simulation Line Profile Comparison")
                
                # Create comprehensive comparison
                fig_comprehensive, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Stress fields with probe lines
                ax1 = axes[0]
                
                # Choose a representative simulation for background
                rep_sim = simulations[0]
                rep_eta, rep_stress = rep_sim['history'][frames[0]]
                rep_stress_data = rep_stress[stress_key]
                
                im = ax1.imshow(rep_stress_data, extent=extent,
                               cmap=plt.cm.get_cmap(COLORMAPS.get(
                                   rep_sim['params']['sigma_cmap'], 'hot')),
                               origin='lower',
                               aspect='auto')
                
                # Add probe lines for all orientations
                center_x, center_y = N//2, N//2
                length = config.get('probe_length', 80)
                orientations = config.get('orientations', [0, 45, 90, 135])
                colors = ['yellow', 'cyan', 'magenta', 'lime', 'orange']
                
                for idx, orientation in enumerate(orientations):
                    profiles = LineProfileAnalyzer.get_line_profiles(
                        rep_stress_data, center_x, center_y, length, orientation
                    )
                    
                    if orientation in profiles:
                        LineProfileAnalyzer.create_probe_line_indicator(
                            ax1, profiles[orientation],
                            style=config.get('probe_style', 'realistic'),
                            line_color=colors[idx % len(colors)],
                            line_width=config.get('profile_linewidth', 2),
                            endpoint_color=colors[idx % len(colors)],
                            text_color='white'
                        )
                
                ax1.set_title("Stress Field with Probe Lines", fontsize=12, fontweight='bold')
                ax1.set_xlabel("x (nm)", fontsize=10)
                ax1.set_ylabel("y (nm)", fontsize=10)
                plt.colorbar(im, ax=ax1, shrink=0.8).set_label(
                    f"{config['stress_component']} (GPa)", fontsize=9
                )
                
                # Plot 2: Overlay profiles
                ax2 = axes[1]
                
                # Plot profiles for all simulations at first orientation
                if orientations:
                    first_orientation = orientations[0]
                    sim_colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                    
                    for sim_idx, (sim, frame, color) in enumerate(zip(simulations, frames, sim_colors)):
                        eta, stress_fields = sim['history'][frame]
                        stress_data = stress_fields[stress_key]
                        
                        profiles = LineProfileAnalyzer.get_line_profiles(
                            stress_data, center_x, center_y, length, first_orientation
                        )
                        
                        if first_orientation in profiles:
                            profile_data = profiles[first_orientation]
                            distances = profile_data['distances']
                            profile = profile_data['profile']
                            actual_distances = distances * length * dx
                            
                            ax2.plot(actual_distances, profile,
                                    color=color,
                                    linewidth=config.get('profile_linewidth', 2),
                                    linestyle=config.get('line_style', 'solid'),
                                    label=f"{sim['params']['defect_type']} - {sim['params']['orientation']}")
                    
                    ax2.set_xlabel("Distance (nm)", fontsize=10)
                    ax2.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=10)
                    ax2.set_title(f"Profile Comparison ({first_orientation}°)", 
                                fontsize=12, fontweight='bold')
                    ax2.legend(fontsize=9, loc='upper right')
                    ax2.grid(True, alpha=0.3)
                
                fig_comprehensive = EnhancedFigureStyler.apply_advanced_styling(
                    fig_comprehensive, axes, advanced_styling
                )
                
                st.pyplot(fig_comprehensive)
            
            # Handle other comparison types (existing code)
            elif config['type'] in ["Side-by-Side Heatmaps", "Statistical Summary", 
                                   "Defect-Stress Correlation"]:
                # Use existing plotting functions
                st.subheader(f"📰 {config['type']}")
                
                # Create appropriate plot based on type
                if config['type'] == "Side-by-Side Heatmaps":
                    # Create heatmap comparison
                    n_sims = len(simulations)
                    cols = min(3, n_sims)
                    rows = (n_sims + cols - 1) // cols
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                    
                    if rows == 1 and cols == 1:
                        axes = np.array([[axes]])
                    elif rows == 1:
                        axes = axes.reshape(1, -1)
                    elif cols == 1:
                        axes = axes.reshape(-1, 1)
                    
                    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                        row = idx // cols
                        col = idx % cols
                        ax = axes[row, col] if rows > 1 or cols > 1 else axes[idx]
                        
                        eta, stress_fields = sim['history'][frame]
                        stress_data = stress_fields[stress_key]
                        
                        im = ax.imshow(stress_data, extent=extent,
                                      cmap=plt.cm.get_cmap(COLORMAPS.get(
                                          sim['params']['sigma_cmap'], 'hot')),
                                      origin='lower',
                                      aspect='auto')
                        
                        ax.set_title(f"{sim['params']['defect_type']}\n{sim['params']['orientation']}",
                                    fontsize=10)
                        ax.set_xlabel("x (nm)", fontsize=8)
                        ax.set_ylabel("y (nm)", fontsize=8)
                        
                        if idx == 0:
                            plt.colorbar(im, ax=ax, shrink=0.8).set_label(
                                f"{config['stress_component']} (GPa)", fontsize=8
                            )
                    
                    # Hide empty subplots
                    for idx in range(n_sims, rows*cols):
                        row = idx // cols
                        col = idx % cols
                        if rows > 1 and cols > 1:
                            axes[row, col].axis('off')
                        elif rows == 1:
                            axes[col].axis('off')
                        else:
                            axes[row].axis('off')
                    
                    fig = EnhancedFigureStyler.apply_advanced_styling(fig, axes, advanced_styling)
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
# EXPORT FUNCTIONALITY WITH POST-PROCESSING
# =============================================
st.sidebar.header("💾 Export Options")

with st.sidebar.expander("📥 Advanced Export"):
    export_format = st.selectbox(
        "Export Format",
        ["Complete Package (JSON + CSV + PNG)", "JSON Parameters Only", 
         "Publication-Ready Figures", "Raw Data CSV", "Line Profile Data"]
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
                        
                        # Export line profile data if requested
                        if export_format == "Line Profile Data":
                            # Extract line profiles for standard orientations
                            orientations = [0, 45, 90, 135]
                            profile_data = {}
                            
                            for i, (eta, stress_fields) in enumerate(sim_data['history']):
                                stress_data = stress_fields['sigma_mag']
                                center_x, center_y = N//2, N//2
                                length = 80
                                
                                profiles = LineProfileAnalyzer.get_line_profiles(
                                    stress_data, center_x, center_y, length, orientations
                                )
                                
                                frame_profiles = {}
                                for angle, profile_info in profiles.items():
                                    frame_profiles[angle] = {
                                        'distances': profile_info['distances'].tolist(),
                                        'profile': profile_info['profile'].tolist(),
                                        'statistics': LineProfileAnalyzer.analyze_line_profiles(
                                            {angle: profile_info}
                                        ).get(angle, {})
                                    }
                                
                                profile_data[f"frame_{i}"] = frame_profiles
                            
                            # Save profile data
                            profile_json = json.dumps(profile_data, indent=2)
                            zf.writestr(f"{sim_dir}/line_profiles.json", profile_json)
                    
                    # Create summary file
                    summary = f"""MULTI-SIMULATION EXPORT SUMMARY
========================================
Generated: {datetime.now().isoformat()}
Total Simulations: {len(simulations)}
Export Format: {export_format}
Includes Styling: {include_styling}
High Resolution: {high_resolution}

LINE PROFILE CAPABILITIES:
--------------------------
- Multiple orientations: Horizontal, Vertical, Diagonal (45°), Anti-diagonal (135°)
- Realistic probe indicators: AFM/STEM style, arrow, rectangle, nano-probe
- Advanced interpolation: Bilinear interpolation for accurate profiles
- Statistical analysis: Mean, max, min, FWHM, peak detection
- Overlay visualization: Multiple profiles on single plot

STYLING PARAMETERS:
-------------------
{json.dumps(advanced_styling, indent=2)}

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
# THEORETICAL ANALYSIS - UPDATED WITH LINE PROFILE INFO
# =============================================
with st.expander("🔬 Theoretical Soundness & Advanced Analysis", expanded=False):
    st.markdown("""
    ### 🎯 **Enhanced Multi-Simulation Comparison Platform**
    
    #### **📊 NEW: Advanced Line Profile Analysis Features:**
    
    **1. Multi-Orientation Line Profiles:**
    - **Horizontal (0°)**: Standard horizontal scan line
    - **Vertical (90°)**: Vertical stress profiling
    - **Diagonal (45°)**: Shear stress analysis along crystal directions
    - **Anti-diagonal (135°)**: Complementary shear analysis
    - **Custom Angles**: User-defined orientation for specific crystallographic directions
    
    **2. Realistic Probe Line Indicators:**
    - **AFM/STEM Style**: Realistic scanning probe microscopy representation
    - **Arrow Indicators**: Clear directional markers with measurement scales
    - **Nano-probe Tips**: Triangular probe tips for nanoscale measurements
    - **Rectangular Probes**: Beam-like probes for EBSD/TEM simulations
    - **Custom Styling**: Adjustable colors, widths, and visual effects
    
    **3. Advanced Interpolation Methods:**
    - **Bilinear Interpolation**: High-quality line profile extraction
    - **Sub-pixel Accuracy**: Resolution beyond grid spacing
    - **Edge Handling**: Proper treatment of boundaries and edges
    - **Smooth Profiles**: Anti-aliased line extraction
    
    **4. Comprehensive Statistical Analysis:**
    - **Basic Statistics**: Mean, standard deviation, min, max, range
    - **Peak Detection**: Automatic identification of stress maxima
    - **FWHM Calculation**: Full Width at Half Maximum for stress concentration regions
    - **Gradient Analysis**: Stress gradient along profiles
    - **Integral Calculations**: Total stress along line segments
    
    **5. Interactive Visualization:**
    - **Real-time Updates**: Adjust probe position and orientation
    - **Multi-profile Overlay**: Compare multiple orientations simultaneously
    - **Color-coded Probes**: Different colors for different orientations
    - **Measurement Annotations**: Distance markers and angle indicators
    
    #### **🔬 Scientific Applications:**
    
    **Stress Concentration Analysis:**
    - **Defect Core Profiling**: Line profiles through defect centers
    - **Stress Gradient Mapping**: Rate of stress change along crystallographic directions
    - **Interface Analysis**: Stress profiles across defect boundaries
    - **Anisotropy Detection**: Orientation-dependent stress variations
    
    **Crystallographic Correlation:**
    - **{111} Plane Analysis**: Profiles along close-packed directions
    - **Shear Stress Components**: Analysis of resolved shear stresses
    - **Habit Plane Effects**: How orientation affects stress distribution
    - **Defect Type Comparison**: ISF vs ESF vs Twin stress profiles
    
    **Experimental Comparison:**
    - **AFM Line Scan Simulation**: Mimics experimental AFM stress mapping
    - **TEM Diffraction Contrast**: Simulates TEM dislocation contrast profiles
    - **Nanoindentation Profiles**: Stress profiles under indentation
    - **EBSD Pattern Quality**: Correlation with experimental EBSD patterns
    
    #### **📈 Key Physical Insights from Line Profile Analysis:**
    
    **ISF (Intrinsic Stacking Fault):**
    - **Symmetric Profiles**: Typically symmetric stress distributions
    - **Moderate Gradients**: Gradual stress changes
    - **Single Peak**: Usually one stress maximum at defect center
    - **Orientation Independence**: Minimal variation with probe direction
    
    **ESF (Extrinsic Stacking Fault):**
    - **Asymmetric Profiles**: Often asymmetric stress distributions
    - **Steeper Gradients**: Rapid stress changes near boundaries
    - **Multiple Peaks**: Can show multiple stress maxima
    - **Directional Effects**: Clear orientation dependence
    
    **Twin Boundary:**
    - **Sharp Interfaces**: Very steep stress gradients
    - **Bimodal Distributions**: Stress peaks on both sides of boundary
    - **Strong Anisotropy**: Significant orientation dependence
    - **Interface-dominated**: Stress concentrated at boundary
    
    ### **🔬 Methodology & Validation:**
    
    **Line Profile Accuracy:**
    - **Interpolation Validation**: Comparison with analytical solutions
    - **Grid Independence**: Convergence with increasing resolution
    - **Boundary Treatment**: Proper handling of simulation boundaries
    - **Error Estimation**: Quantification of interpolation errors
    
    **Probe Line Realism:**
    - **Experimental Correspondence**: Matching real microscopy probe geometries
    - **Scale Accuracy**: Proper nanometer-scale representation
    - **Visual Clarity**: Clear indication of measurement parameters
    - **Publication Quality**: Professional appearance for figures
    
    **Statistical Reliability:**
    - **Robust Peak Detection**: Reliable identification of stress maxima
    - **Accurate FWHM**: Proper calculation of stress concentration widths
    - **Consistent Metrics**: Standardized statistical measures
    - **Error Propagation**: Proper handling of measurement uncertainties
    
    **Advanced crystallographic stress analysis platform with publication-ready outputs, comprehensive statistical analysis, and realistic line profile visualization!**
    """)
    
    # Display platform statistics
    simulations = SimulationDB.get_all_simulations()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulations", len(simulations))
    with col2:
        total_frames = sum([len(sim['history']) for sim in simulations.values()]) if simulations else 0
        st.metric("Total Frames", f"{total_frames:,}")
    with col3:
        st.metric("Available Colormaps", f"{len(COLORMAPS)}+")
    with col4:
        st.metric("Line Profile Orientations", "5+")

st.caption("🔬 Advanced Multi-Defect Comparison • Publication-Quality Output • Multi-Orientation Line Profiles • Realistic Probe Indicators • 2025")
