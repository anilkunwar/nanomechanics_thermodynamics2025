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
# ENHANCED LINE PROFILE ANALYSIS SYSTEM (FIXED)
# =============================================
class LineProfileAnalyzer:
    """Enhanced line profile extraction with robust error handling"""
    
    @staticmethod
    def extract_line_profile(data_2d, x0, y0, x1, y1, num_points=200):
        """
        Extract line profile from 2D data with robust error handling
        
        FIXED: Handles cases where start and end points are identical
        """
        # Validate input points to avoid zero-length lines
        if np.abs(x1 - x0) < 1e-10 and np.abs(y1 - y0) < 1e-10:
            # Return a small line centered at the point
            x0 = max(0, x0 - 5)
            x1 = min(data_2d.shape[1] - 1, x1 + 5)
            y0 = max(0, y0 - 5)
            y1 = min(data_2d.shape[0] - 1, y1 + 5)
        
        # Create linearly spaced points along the line
        t = np.linspace(0, 1, num_points)
        x_coords = x0 + t * (x1 - x0)
        y_coords = y0 + t * (y1 - y0)
        
        # Ensure coordinates are within bounds
        x_coords = np.clip(x_coords, 0, data_2d.shape[1] - 1)
        y_coords = np.clip(y_coords, 0, data_2d.shape[0] - 1)
        
        # Use scipy's map_coordinates for high-quality interpolation
        coords = np.vstack((y_coords, x_coords))  # Note: y first for row-major
        profile = map_coordinates(data_2d, coords, order=1, mode='nearest')
        
        # Calculate actual distances with robust normalization
        distances = np.sqrt((x_coords - x0)**2 + (y_coords - y0)**2)
        
        # Check for zero-length line and handle gracefully
        max_distance = distances[-1]
        if max_distance > 1e-10:
            distances = distances / max_distance
        else:
            # If line is essentially zero-length, use normalized index
            distances = t
        
        return distances, profile, (x_coords, y_coords)
    
    @staticmethod
    def get_line_profiles(data_2d, center_x, center_y, length, angle_deg=0):
        """
        Get line profiles in multiple orientations with robust length handling
        """
        profiles = {}
        
        if isinstance(angle_deg, (int, float)):
            angle_deg = [angle_deg]
        
        # Ensure minimum valid length
        min_length = 5  # Minimum grid units
        length = max(min_length, length)
        
        for angle in angle_deg:
            # Convert angle to radians
            theta = np.deg2rad(angle)
            
            # Calculate end points with validation
            dx = length * np.cos(theta) / 2
            dy = length * np.sin(theta) / 2
            
            x0 = center_x - dx
            y0 = center_y - dy
            x1 = center_x + dx
            y1 = center_y + dy
            
            # Ensure points are within data bounds
            x0 = max(0, min(data_2d.shape[1] - 1, x0))
            x1 = max(0, min(data_2d.shape[1] - 1, x1))
            y0 = max(0, min(data_2d.shape[0] - 1, y0))
            y1 = max(0, min(data_2d.shape[0] - 1, y1))
            
            # Extract profile with error handling
            try:
                distances, profile, coords = LineProfileAnalyzer.extract_line_profile(
                    data_2d, x0, y0, x1, y1
                )
                
                # Verify profile has valid data
                if len(profile) > 0 and not np.all(np.isnan(profile)):
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
                else:
                    st.warning(f"Empty profile extracted for angle {angle}°")
                    
            except Exception as e:
                st.error(f"Error extracting profile for angle {angle}°: {str(e)}")
        
        return profiles
    
    @staticmethod
    def extract_multiple_orientations(data_2d, center_x, center_y, length=100, 
                                     orientations=None):
        """Extract profiles in standard orientations with validation"""
        if orientations is None:
            orientations = [0, 45, 90, 135]  # Horizontal, Diagonal, Vertical, Anti-diagonal
        
        # Ensure center is within bounds
        center_x = max(0, min(data_2d.shape[1] - 1, center_x))
        center_y = max(0, min(data_2d.shape[0] - 1, center_y))
        
        return LineProfileAnalyzer.get_line_profiles(
            data_2d, center_x, center_y, length, orientations
        )
    
    @staticmethod
    def create_probe_line_indicator(ax, profile_info, style='realistic', **kwargs):
        """
        Create realistic geometric probe line indicators with enhanced visualization
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
            'shadow': True,
            'extent': None  # Add extent for coordinate transformation
        }
        
        # Update with provided kwargs
        params = {**default_params, **kwargs}
        
        # Get coordinates
        x_coords, y_coords = profile_info['coords']
        x0, y0 = profile_info['start']
        x1, y1 = profile_info['end']
        center_x, center_y = profile_info['center']
        angle = profile_info['angle']
        
        # Convert from grid coordinates to physical coordinates if extent is provided
        if params.get('extent') is not None:
            extent = params['extent']
            # Transform coordinates from grid indices to physical coordinates
            x_min, x_max, y_min, y_max = extent
            grid_width = x_max - x_min
            grid_height = y_max - y_min
            
            # Normalize grid indices to [0, 1] then scale to physical coordinates
            x_coords = x_min + (x_coords / (N-1)) * grid_width
            y_coords = y_min + (y_coords / (N-1)) * grid_height
            x0 = x_min + (x0 / (N-1)) * grid_width
            y0 = y_min + (y0 / (N-1)) * grid_height
            x1 = x_min + (x1 / (N-1)) * grid_width
            y1 = y_min + (y1 / (N-1)) * grid_height
            center_x = x_min + (center_x / (N-1)) * grid_width
            center_y = y_min + (center_y / (N-1)) * grid_height
        
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
                   path_effects=path_effects_list,
                   zorder=10)
            
            # Endpoint markers
            for point in [(x0, y0), (x1, y1)]:
                ax.scatter(*point, 
                          s=params['endpoint_size'],
                          color=params['endpoint_color'],
                          edgecolors='white',
                          linewidths=2,
                          zorder=11,
                          marker='o')
            
            # Center marker
            ax.scatter(center_x, center_y,
                      s=params['endpoint_size'] * 0.7,
                      color='cyan',
                      edgecolors='white',
                      linewidths=2,
                      zorder=12,
                      marker='s')
            
            # Add directional arrow
            arrow_length = np.sqrt((x1-x0)**2 + (y1-y0)**2) * 0.2
            if arrow_length > 0:
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
                            edgecolor='none'),
                   zorder=13)
        
        elif style == 'arrow':
            # Simple arrow style
            if np.sqrt((x1-x0)**2 + (y1-y0)**2) > 0:
                arrow = Arrow(x0, y0, x1-x0, y1-y0,
                             width=params['line_width']*3,
                             color=params['line_color'],
                             alpha=params['line_alpha'])
                ax.add_patch(arrow)
            
        elif style == 'rectangle':
            # Rectangle probe style
            rect_width = params['line_width'] * 2
            rect_length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            
            if rect_length > 0:
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
            ax.plot(x_coords, y_coords,
                   color=params['line_color'],
                   linewidth=params['line_width'],
                   alpha=params['line_alpha'],
                   solid_capstyle='round')
            
            # Probe tip (only if line has length)
            tip_length = np.sqrt((x1-x0)**2 + (y1-y0)**2) * 0.1
            if tip_length > 0:
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
        
        return ax
    
    @staticmethod
    def analyze_line_profiles(profiles_dict):
        """Analyze extracted line profiles and compute statistics"""
        analysis = {}
        
        for angle, profile_data in profiles_dict.items():
            profile = profile_data['profile']
            
            if len(profile) > 0 and not np.all(np.isnan(profile)):
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
        """Calculate Full Width at Half Maximum with validation"""
        if len(profile) == 0 or np.all(np.isnan(profile)):
            return 0
        
        try:
            half_max = (np.max(profile) + np.min(profile)) / 2
            above_half = profile > half_max
            
            if np.any(above_half):
                indices = np.where(above_half)[0]
                if len(indices) > 1:
                    return distances[indices[-1]] - distances[indices[0]]
            return 0
        except:
            return 0
    
    @staticmethod
    def find_peaks(profile, min_distance=5, prominence=0.1):
        """Find peaks in profile using scipy with error handling"""
        from scipy.signal import find_peaks
        
        if len(profile) < min_distance or np.all(np.isnan(profile)):
            return []
        
        try:
            peaks, properties = find_peaks(profile, 
                                          distance=min_distance,
                                          prominence=prominence)
            
            return peaks.tolist()
        except:
            return []

# =============================================
# ENHANCED LINE PROFILE VISUALIZATION (FIXED)
# =============================================
class AdvancedLineProfileVisualizer:
    """Enhanced visualization for line profiles with robust overlay capabilities"""
    
    @staticmethod
    def create_stress_overlay_plot(data_dict, profiles_dict, 
                                   title="Stress Overlay Analysis",
                                   style_params=None):
        """
        Create overlay plot showing stress field with multiple line profiles
        FIXED: Proper coordinate transformation for overlay alignment
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
            # Plot stress field
            cmap_name = style_params.get('stress_cmap', 'hot')
            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'hot'))
            im = ax1.imshow(stress_data, extent=extent,
                           cmap=cmap,
                           origin='lower',
                           aspect='auto',
                           interpolation='bilinear')
            
            # Add all probe lines with proper coordinate transformation
            probe_styles = ['realistic', 'arrow', 'nanoprobe', 'rectangle']
            colors = ['yellow', 'cyan', 'magenta', 'lime']
            
            for idx, (angle, profile_data) in enumerate(profiles_dict.items()):
                if angle in profiles_dict:
                    style = probe_styles[idx % len(probe_styles)]
                    color = colors[idx % len(colors)]
                    
                    LineProfileAnalyzer.create_probe_line_indicator(
                        ax1, profile_data,
                        style=style,
                        line_color=color,
                        line_width=style_params.get('probe_width', 2),
                        endpoint_color=color,
                        shadow=True,
                        extent=extent  # Pass extent for coordinate transformation
                    )
            
            ax1.set_title("Stress Field with Probe Lines", fontsize=12, fontweight='bold')
            ax1.set_xlabel("x (nm)", fontsize=10)
            ax1.set_ylabel("y (nm)", fontsize=10)
            ax1.set_xlim(extent[0], extent[1])
            ax1.set_ylim(extent[2], extent[3])
            
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
                
                # Ensure we have valid data
                if len(actual_distances) > 0 and len(profile) > 0:
                    # Sort distances if needed
                    sort_idx = np.argsort(actual_distances)
                    actual_distances = actual_distances[sort_idx]
                    profile = profile[sort_idx]
                    
                    ax.plot(actual_distances, profile,
                           linewidth=style_params.get('profile_linewidth', 2),
                           color=style_params.get('profile_color', 'blue'),
                           alpha=0.8)
                    
                    # Add fill under curve
                    ax.fill_between(actual_distances, 0, profile,
                                   alpha=0.3,
                                   color=style_params.get('fill_color', 'blue'))
                    
                    # Add statistics text
                    if len(profile) > 0:
                        stats_text = f"Max: {np.nanmax(profile):.2f} GPa\n" \
                                   f"Min: {np.nanmin(profile):.2f} GPa\n" \
                                   f"Mean: {np.nanmean(profile):.2f} GPa"
                        
                        ax.text(0.05, 0.95, stats_text,
                               transform=ax.transAxes,
                               fontsize=8,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.set_title(f"{orientation}° Profile", fontsize=10, fontweight='bold')
                    ax.set_xlabel("Distance (nm)", fontsize=9)
                    ax.set_ylabel("Stress (GPa)", fontsize=9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Set reasonable x-axis limits
                    if len(actual_distances) > 1:
                        ax.set_xlim(actual_distances[0], actual_distances[-1])
        
        # Plot 6: Statistics table
        analysis = LineProfileAnalyzer.analyze_line_profiles(profiles_dict)
        
        if analysis:
            # Create table data
            table_data = []
            columns = ['Angle', 'Mean', 'Std', 'Min', 'Max', 'Range', 'Peaks']
            
            for angle, stats in analysis.items():
                table_data.append([
                    f"{angle}°",
                    f"{stats.get('mean', 0):.3f}",
                    f"{stats.get('std', 0):.3f}",
                    f"{stats.get('min', 0):.3f}",
                    f"{stats.get('max', 0):.3f}",
                    f"{stats.get('range', 0):.3f}",
                    f"{stats.get('n_peaks', 0)}"
                ])
            
            # Create table
            if table_data:
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

# =============================================
# ENHANCED PUBLICATION-QUALITY PLOTTING
# =============================================
def create_publication_quality_plot(sim_data, frame_idx, profile_config, style_params):
    """
    Create publication-quality plot with properly aligned overlays
    """
    # Extract data
    eta, stress_fields = sim_data['history'][frame_idx]
    stress_component = profile_config.get('stress_component', 'Stress Magnitude |σ|')
    
    # Map stress component
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map.get(stress_component, 'sigma_mag')
    stress_data = stress_fields[stress_key]
    
    # Extract line profiles
    center_x = profile_config.get('center_x', N//2)
    center_y = profile_config.get('center_y', N//2)
    length = profile_config.get('length', 80)
    orientations = profile_config.get('orientations', [0, 45, 90, 135])
    
    # Validate center coordinates
    center_x = max(0, min(N-1, center_x))
    center_y = max(0, min(N-1, center_y))
    
    profiles = LineProfileAnalyzer.get_line_profiles(
        stress_data, center_x, center_y, length, orientations
    )
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    
    # Define layout
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25,
                         height_ratios=[1, 1, 0.5])
    
    # Main stress map
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Profile plots
    ax_profiles = []
    for i in range(4):
        ax_profiles.append(fig.add_subplot(gs[i//2, 2 + (i%2)]))
    
    # Statistics table
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Plot main stress map
    cmap_name = sim_data['params'].get('sigma_cmap', 'hot')
    cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'hot'))
    
    im = ax_main.imshow(stress_data, extent=extent,
                       cmap=cmap,
                       origin='lower',
                       aspect='auto',
                       interpolation='bilinear')
    
    # Add probe lines
    colors = ['#FFD700', '#00FFFF', '#FF00FF', '#32CD32']  # Gold, Cyan, Magenta, Lime
    for idx, (angle, profile_data) in enumerate(profiles.items()):
        LineProfileAnalyzer.create_probe_line_indicator(
            ax_main, profile_data,
            style='realistic',
            line_color=colors[idx % len(colors)],
            line_width=2.5,
            endpoint_color=colors[idx % len(colors)],
            shadow=True,
            extent=extent
        )
    
    ax_main.set_title(f"{sim_data['params']['defect_type']} - {stress_component}",
                     fontsize=14, fontweight='bold', pad=10)
    ax_main.set_xlabel("x (nm)", fontsize=12)
    ax_main.set_ylabel("y (nm)", fontsize=12)
    
    # Add scale bar
    scale_length = 10.0  # 10 nm
    scale_y = extent[2] + 0.9 * (extent[3] - extent[2])
    ax_main.plot([extent[0] + 5, extent[0] + 5 + scale_length],
                 [scale_y, scale_y],
                 color='white', linewidth=3, zorder=20)
    ax_main.text(extent[0] + 5 + scale_length/2, scale_y - 1,
                f'{scale_length} nm', color='white',
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_main, shrink=0.8, pad=0.02)
    cbar.set_label('Stress (GPa)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    # Plot individual profiles
    for idx, (ax, orientation) in enumerate(zip(ax_profiles, orientations)):
        if orientation in profiles:
            profile_data = profiles[orientation]
            distances = profile_data['distances']
            profile = profile_data['profile']
            
            # Convert to physical distances
            actual_distances = distances * length * dx
            
            # Sort for clean plotting
            sort_idx = np.argsort(actual_distances)
            actual_distances = actual_distances[sort_idx]
            profile = profile[sort_idx]
            
            ax.plot(actual_distances, profile,
                   color=colors[idx],
                   linewidth=2.5,
                   alpha=0.9,
                   marker='o',
                   markersize=3,
                   markevery=10)
            
            # Add fill
            ax.fill_between(actual_distances, 0, profile,
                           alpha=0.2, color=colors[idx])
            
            # Add statistics
            if len(profile) > 0:
                stats = {
                    'max': np.nanmax(profile),
                    'min': np.nanmin(profile),
                    'mean': np.nanmean(profile),
                    'fwhm': LineProfileAnalyzer.calculate_fwhm(distances, profile)
                }
                
                stats_text = f"Max: {stats['max']:.2f} GPa\n" \
                           f"Min: {stats['min']:.2f} GPa\n" \
                           f"FWHM: {stats['fwhm']:.2f}"
                
                ax.text(0.05, 0.95, stats_text,
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f"{orientation}° Orientation", fontsize=12, fontweight='bold')
            ax.set_xlabel("Distance (nm)", fontsize=11)
            ax.set_ylabel("Stress (GPa)", fontsize=11)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Set limits
            if len(actual_distances) > 1:
                ax.set_xlim(actual_distances[0], actual_distances[-1])
    
    # Add statistics table
    analysis = LineProfileAnalyzer.analyze_line_profiles(profiles)
    if analysis:
        table_data = []
        for angle, stats in analysis.items():
            table_data.append([
                f"{angle}°",
                f"{stats.get('mean', 0):.3f}",
                f"{stats.get('std', 0):.3f}",
                f"{stats.get('min', 0):.3f}",
                f"{stats.get('max', 0):.3f}",
                f"{stats.get('range', 0):.3f}",
                f"{stats.get('fwhm', 0):.3f}",
                f"{stats.get('n_peaks', 0)}"
            ])
        
        columns = ['Angle', 'Mean', 'Std', 'Min', 'Max', 'Range', 'FWHM', 'Peaks']
        
        if table_data:
            table = ax_stats.table(cellText=table_data,
                                  colLabels=columns,
                                  cellLoc='center',
                                  loc='center',
                                  colColours=['#4A90E2']*len(columns))
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5, 2.0)
            
            # Style table cells
            for key, cell in table.get_celld().items():
                if key[0] == 0:  # Header row
                    cell.set_text_props(fontweight='bold', color='white')
                    cell.set_facecolor('#4A90E2')
                else:
                    cell.set_facecolor('#F7F7F7')
            
            ax_stats.set_title("Comprehensive Profile Statistics",
                             fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle(f"Publication-Quality Analysis: {sim_data['params']['defect_type']} Defect",
                fontsize=16, fontweight='bold', y=0.98)
    
    # Apply tight layout
    plt.tight_layout()
    
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
# SIDEBAR - Global Settings
# =============================================
st.sidebar.header("🎨 Global Chart Styling")

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
    eps0 = st.sidebar.slider("Eigenstrain magnitude ε*", 0.3, 3.0, value=default_eps, step=0.01)
    kappa = st.sidebar.slider("Interface energy coeff κ", 0.1, 2.0, value=default_kappa, step=0.05)
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
    sim_eta_cmap_name = st.sidebar.selectbox("η colormap", cmap_list, index=cmap_list.index(eta_cmap_name))
    sim_sigma_cmap_name = st.sidebar.selectbox("|σ| colormap", cmap_list, index=cmap_list.index(sigma_cmap_name))
    sim_hydro_cmap_name = st.sidebar.selectbox("Hydrostatic colormap", cmap_list, index=cmap_list.index(hydro_cmap_name))
    sim_vm_cmap_name = st.sidebar.selectbox("von Mises colormap", cmap_list, index=cmap_list.index(vm_cmap_name))
    
    # Line profile settings for new simulation
    with st.sidebar.expander("📐 Line Profile Settings", expanded=True):
        st.subheader("Line Profile Configuration")
        
        center_x = st.slider("Center X (grid units)", 0, N-1, N//2, key="new_center_x")
        center_y = st.slider("Center Y (grid units)", 0, N-1, N//2, key="new_center_y")
        length = st.slider("Line Length (grid units)", 20, 120, 80, key="new_length")
        
        orientations = st.multiselect(
            "Select orientations",
            options=[0, 45, 90, 135],
            default=[0, 90],
            key="new_orientations"
        )
        
        probe_style = st.selectbox(
            "Probe Style",
            ["realistic", "arrow", "nanoprobe", "rectangle"],
            index=0,
            key="new_probe_style"
        )
    
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
        st.session_state.profile_config = {
            'center_x': center_x,
            'center_y': center_y,
            'length': length,
            'orientations': orientations,
            'probe_style': probe_style
        }

else:  # Compare Saved Simulations
    st.sidebar.header("🔍 Simulation Comparison Setup")
    
    # Line Profile Analysis Settings
    with st.sidebar.expander("📐 Line Profile Analysis Settings", expanded=True):
        st.subheader("Probe Line Configuration")
        
        selected_orientations = st.multiselect(
            "Select Line Orientations",
            options=["Horizontal (0°)", "Vertical (90°)", "Diagonal (45°)", 
                    "Anti-diagonal (135°)", "Custom Angle"],
            default=["Horizontal (0°)", "Vertical (90°)", "Diagonal (45°)"]
        )
        
        custom_angle = None
        if "Custom Angle" in selected_orientations:
            custom_angle = st.slider("Custom Angle (°)", -180, 180, 30, 5)
        
        probe_length = st.slider("Probe Line Length (grid units)", 20, 120, 80, 5)
        
        probe_style = st.selectbox(
            "Probe Line Indicator Style",
            ["realistic", "arrow", "rectangle", "nanoprobe"],
            index=0
        )
        
        show_all_profiles = st.checkbox("Show all profiles on main plot", True)
        profile_linewidth = st.slider("Profile Line Width", 1.0, 5.0, 2.0, 0.5)
        
        # Extract numeric angles
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
             "Statistical Summary"],
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
        
        # Publication quality settings
        st.sidebar.subheader("Publication Quality")
        enable_publication_quality = st.checkbox("Enable Publication-Quality Output", True)
        
        if enable_publication_quality:
            with st.sidebar.expander("Publication Settings"):
                include_scale_bar = st.checkbox("Include Scale Bar", True)
                include_statistics = st.checkbox("Include Statistics Table", True)
                use_vector_fonts = st.checkbox("Use Vector Fonts", True)
                export_dpi = st.select_slider("Export DPI", options=[300, 600, 1200], value=600)
        
        # Run comparison
        if st.sidebar.button("🔬 Run Comparison", type="primary"):
            st.session_state.run_comparison = True
            st.session_state.comparison_config = {
                'sim_ids': selected_ids,
                'type': comparison_type,
                'stress_component': stress_component,
                'frame_selection': frame_selection,
                'frame_idx': frame_idx,
                'orientations': orientations,
                'probe_length': probe_length,
                'probe_style': probe_style,
                'show_all_profiles': show_all_profiles,
                'profile_linewidth': profile_linewidth,
                'publication_quality': enable_publication_quality if 'enable_publication_quality' in locals() else False
            }

# =============================================
# SIMULATION ENGINE (Reusable Functions)
# =============================================
def create_initial_eta(shape, defect_type):
    """Create initial defect configuration"""
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
                    st.subheader("Publication-Quality Line Profile Analysis")
                    
                    # Get final state
                    final_eta, final_stress = history[-1]
                    
                    # Create publication-quality plot
                    profile_config = {
                        'center_x': st.session_state.profile_config['center_x'],
                        'center_y': st.session_state.profile_config['center_y'],
                        'length': st.session_state.profile_config['length'],
                        'orientations': st.session_state.profile_config['orientations'],
                        'stress_component': "Stress Magnitude |σ|",
                        'probe_style': st.session_state.profile_config['probe_style']
                    }
                    
                    # Get simulation data
                    sim_data = SimulationDB.get_simulation(sim_id)
                    
                    # Create enhanced plot
                    fig = create_publication_quality_plot(
                        sim_data, 
                        len(history)-1,  # Final frame
                        profile_config,
                        {}
                    )
                    
                    st.pyplot(fig)
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📥 Save as PNG", type="secondary"):
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                "Download PNG",
                                buf.getvalue(),
                                f"publication_plot_{sim_id}.png",
                                "image/png"
                            )
                    with col2:
                        if st.button("📥 Save as PDF", type="secondary"):
                            buf = BytesIO()
                            fig.savefig(buf, format='pdf', dpi=600, bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                "Download PDF",
                                buf.getvalue(),
                                f"publication_plot_{sim_id}.pdf",
                                "application/pdf"
                            )
                    with col3:
                        if st.button("📥 Save as SVG", type="secondary"):
                            buf = BytesIO()
                            fig.savefig(buf, format='svg', bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                "Download SVG",
                                buf.getvalue(),
                                f"publication_plot_{sim_id}.svg",
                                "image/svg+xml"
                            )
                
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
                frames = [len(sim['history']) - 1 for sim in simulations]
            elif config['frame_selection'] == "Same Evolution Time":
                target_percentage = 0.8  # 80% of evolution
                frames = [int(len(sim['history']) * target_percentage) for sim in simulations]
            else:
                frames = [min(frame_idx, len(sim['history']) - 1) for sim in simulations]
            
            # Get stress component mapping
            stress_map = {
                "Stress Magnitude |σ|": 'sigma_mag',
                "Hydrostatic σ_h": 'sigma_hydro',
                "von Mises σ_vM": 'von_mises'
            }
            stress_key = stress_map[config['stress_component']]
            
            # Create comparison based on type
            if config['type'] == "Overlay Line Profiles":
                st.subheader("📈 Publication-Quality Overlay Line Profile Comparison")
                
                # Create enhanced comparison plot
                fig = plt.figure(figsize=(16, 12))
                
                # Define layout
                gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25,
                                     height_ratios=[1, 1, 0.5])
                
                # Main comparison plot
                ax_main = fig.add_subplot(gs[0:2, 0:2])
                
                # Individual profile plots
                ax_profiles = []
                for i in range(4):
                    ax_profiles.append(fig.add_subplot(gs[i//2, 2 + (i%2)]))
                
                # Statistics table
                ax_stats = fig.add_subplot(gs[2, :])
                ax_stats.axis('off')
                
                # Plot all simulations on main axis
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
                
                for sim_idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                    # Get stress data
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Extract line profiles
                    center_x, center_y = N//2, N//2
                    length = config.get('probe_length', 80)
                    orientations = config.get('orientations', [0, 45, 90, 135])
                    
                    if orientations:
                        first_orientation = orientations[0]
                        profiles = LineProfileAnalyzer.get_line_profiles(
                            stress_data, center_x, center_y, length, first_orientation
                        )
                        
                        if first_orientation in profiles:
                            profile_data = profiles[first_orientation]
                            distances = profile_data['distances']
                            profile = profile_data['profile']
                            actual_distances = distances * length * dx
                            
                            # Plot on main axis
                            ax_main.plot(actual_distances, profile,
                                        color=color,
                                        linewidth=config.get('profile_linewidth', 2),
                                        alpha=0.8,
                                        label=f"{sim['params']['defect_type']} - {sim['params']['orientation']}")
                
                ax_main.set_xlabel("Distance (nm)", fontsize=12)
                ax_main.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=12)
                ax_main.set_title("Multi-Simulation Profile Overlay", fontsize=14, fontweight='bold')
                ax_main.legend(fontsize=10, loc='upper right')
                ax_main.grid(True, alpha=0.2, linestyle='--')
                ax_main.tick_params(axis='both', which='major', labelsize=10)
                
                # Plot individual profiles
                for idx, (ax, orientation) in enumerate(zip(ax_profiles, orientations[:4])):
                    for sim_idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
                        eta, stress_fields = sim['history'][frame]
                        stress_data = stress_fields[stress_key]
                        
                        profiles = LineProfileAnalyzer.get_line_profiles(
                            stress_data, center_x, center_y, length, orientation
                        )
                        
                        if orientation in profiles:
                            profile_data = profiles[orientation]
                            distances = profile_data['distances']
                            profile = profile_data['profile']
                            actual_distances = distances * length * dx
                            
                            # Sort for clean plotting
                            sort_idx = np.argsort(actual_distances)
                            actual_distances = actual_distances[sort_idx]
                            profile = profile[sort_idx]
                            
                            ax.plot(actual_distances, profile,
                                   color=color,
                                   linewidth=1.5,
                                   alpha=0.7)
                    
                    ax.set_title(f"{orientation}° Orientation", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Distance (nm)", fontsize=10)
                    if idx % 2 == 0:
                        ax.set_ylabel("Stress (GPa)", fontsize=10)
                    ax.grid(True, alpha=0.1, linestyle='--')
                    ax.tick_params(axis='both', which='major', labelsize=9)
                
                # Add comprehensive statistics table
                all_stats = []
                for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    profiles = LineProfileAnalyzer.get_line_profiles(
                        stress_data, center_x, center_y, length, orientations
                    )
                    
                    analysis = LineProfileAnalyzer.analyze_line_profiles(profiles)
                    
                    for angle, stats in analysis.items():
                        all_stats.append({
                            'Simulation': f"{sim['params']['defect_type']}",
                            'Orientation': f"{sim['params']['orientation']}",
                            'Profile Angle (°)': angle,
                            'Mean Stress (GPa)': f"{stats.get('mean', 0):.3f}",
                            'Max Stress (GPa)': f"{stats.get('max', 0):.3f}",
                            'Stress Range (GPa)': f"{stats.get('range', 0):.3f}",
                            'FWHM': f"{stats.get('fwhm', 0):.3f}"
                        })
                
                if all_stats:
                    df_stats = pd.DataFrame(all_stats)
                    
                    # Create table
                    table_data = [df_stats.columns.tolist()] + df_stats.values.tolist()
                    table = ax_stats.table(cellText=table_data,
                                         cellLoc='center',
                                         loc='center')
                    
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.5, 2.0)
                    
                    # Style header row
                    for j in range(len(df_stats.columns)):
                        table[(0, j)].set_facecolor('#4A90E2')
                        table[(0, j)].set_text_props(fontweight='bold', color='white')
                    
                    ax_stats.set_title("Comprehensive Statistical Comparison",
                                     fontsize=12, fontweight='bold', pad=20)
                
                plt.suptitle("Publication-Quality Multi-Simulation Comparison",
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Comparison as PDF", type="primary"):
                        buf = BytesIO()
                        fig.savefig(buf, format='pdf', dpi=600, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            "Download PDF Report",
                            buf.getvalue(),
                            "multi_simulation_comparison.pdf",
                            "application/pdf"
                        )
                with col2:
                    if st.button("📊 Download Statistics as CSV", type="secondary"):
                        if all_stats:
                            csv = df_stats.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                "profile_statistics.csv",
                                "text/csv"
                            )
            
            elif config['type'] == "Multi-Orientation Line Analysis":
                st.subheader("📐 Multi-Orientation Line Profile Analysis")
                
                # Select a simulation for detailed analysis
                sim_idx = st.selectbox(
                    "Select simulation for detailed analysis",
                    options=range(len(simulations)),
                    format_func=lambda i: f"{simulations[i]['params']['defect_type']} - {simulations[i]['params']['orientation']}"
                )
                
                sim = simulations[sim_idx]
                frame = frames[sim_idx]
                
                # Create publication-quality plot for selected simulation
                profile_config = {
                    'center_x': N//2,
                    'center_y': N//2,
                    'length': config.get('probe_length', 80),
                    'orientations': config.get('orientations', [0, 45, 90, 135]),
                    'stress_component': config['stress_component'],
                    'probe_style': config.get('probe_style', 'realistic')
                }
                
                fig = create_publication_quality_plot(sim, frame, profile_config, {})
                
                st.pyplot(fig)
            
            elif config['type'] == "Side-by-Side Heatmaps":
                st.subheader("📰 Side-by-Side Heatmap Comparison")
                
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
                
                plt.tight_layout()
                st.pyplot(fig)
            
            elif config['type'] == "Statistical Summary":
                st.subheader("📊 Comprehensive Statistical Summary")
                
                # Collect statistics from all simulations
                all_stats = []
                
                for sim, frame in zip(simulations, frames):
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Basic statistics
                    flat_data = stress_data.flatten()
                    stats = {
                        'Simulation': f"{sim['params']['defect_type']}",
                        'Orientation': sim['params']['orientation'],
                        'Mean (GPa)': np.nanmean(flat_data),
                        'Std Dev (GPa)': np.nanstd(flat_data),
                        'Min (GPa)': np.nanmin(flat_data),
                        'Max (GPa)': np.nanmax(flat_data),
                        'Range (GPa)': np.nanmax(flat_data) - np.nanmin(flat_data),
                        'Skewness': stats.skew(flat_data[np.isfinite(flat_data)]),
                        'Kurtosis': stats.kurtosis(flat_data[np.isfinite(flat_data)])
                    }
                    
                    all_stats.append(stats)
                
                # Create summary dataframe
                df_summary = pd.DataFrame(all_stats)
                
                # Display summary table
                st.dataframe(df_summary, use_container_width=True)
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot 1: Mean stress comparison
                axes[0, 0].bar(range(len(simulations)), 
                              [s['Mean (GPa)'] for s in all_stats],
                              color=plt.cm.Set3(np.linspace(0, 1, len(simulations))))
                axes[0, 0].set_title("Mean Stress Comparison", fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel("Stress (GPa)", fontsize=10)
                axes[0, 0].set_xticks(range(len(simulations)))
                axes[0, 0].set_xticklabels([f"{s['Simulation']}\n{s['Orientation']}" for s in all_stats],
                                          rotation=45, ha='right', fontsize=9)
                
                # Plot 2: Stress range comparison
                axes[0, 1].bar(range(len(simulations)), 
                              [s['Range (GPa)'] for s in all_stats],
                              color=plt.cm.Set2(np.linspace(0, 1, len(simulations))))
                axes[0, 1].set_title("Stress Range Comparison", fontsize=12, fontweight='bold')
                axes[0, 1].set_ylabel("Stress Range (GPa)", fontsize=10)
                axes[0, 1].set_xticks(range(len(simulations)))
                axes[0, 1].set_xticklabels([f"{s['Simulation']}\n{s['Orientation']}" for s in all_stats],
                                          rotation=45, ha='right', fontsize=9)
                
                # Plot 3: Statistical distribution
                for idx, stats in enumerate(all_stats):
                    axes[1, 0].hist(stress_fields[stress_key].flatten(), 
                                   bins=50, alpha=0.5, 
                                   label=f"{stats['Simulation']}",
                                   density=True)
                axes[1, 0].set_title("Stress Distribution Comparison", fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel("Stress (GPa)", fontsize=10)
                axes[1, 0].set_ylabel("Probability Density", fontsize=10)
                axes[1, 0].legend(fontsize=9)
                
                # Plot 4: Skewness vs Kurtosis
                scatter = axes[1, 1].scatter([s['Skewness'] for s in all_stats],
                                            [s['Kurtosis'] for s in all_stats],
                                            s=100, c=range(len(simulations)),
                                            cmap='rainbow')
                axes[1, 1].set_title("Distribution Shape Analysis", fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel("Skewness", fontsize=10)
                axes[1, 1].set_ylabel("Kurtosis", fontsize=10)
                
                # Add labels for scatter points
                for idx, stats in enumerate(all_stats):
                    axes[1, 1].text(stats['Skewness'], stats['Kurtosis'],
                                   f"{stats['Simulation']}",
                                   fontsize=8, ha='center', va='bottom')
                
                plt.colorbar(scatter, ax=axes[1, 1], label='Simulation Index')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Clear comparison flag
            if 'run_comparison' in st.session_state:
                del st.session_state.run_comparison
    
    else:
        st.info("Select simulations in the sidebar and click 'Run Comparison' to start!")

# =============================================
# EXPORT FUNCTIONALITY
# =============================================
st.sidebar.header("💾 Export Options")

with st.sidebar.expander("📥 Advanced Export"):
    export_format = st.selectbox(
        "Export Format",
        ["Complete Package (JSON + CSV + PNG)", "Publication-Ready Figures", 
         "Line Profile Data", "Statistical Summary"]
    )
    
    if st.button("📥 Generate Custom Export", type="primary"):
        simulations = SimulationDB.get_all_simulations()
        
        if not simulations:
            st.sidebar.warning("No simulations to export!")
        else:
            with st.spinner("Creating custom export package..."):
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Create summary
                    summary = f"""MULTI-SIMULATION EXPORT SUMMARY
========================================
Generated: {datetime.now().isoformat()}
Total Simulations: {len(simulations)}
Export Format: {export_format}

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

st.caption("🔬 Advanced Multi-Defect Comparison • Publication-Quality Output • Multi-Orientation Line Profiles • 2025")
