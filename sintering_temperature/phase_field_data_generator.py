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
warnings.filterwarnings('ignore')
import torch
import pickle
import sqlite3

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
dx = 0.1 # nm
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
            diag_length = int(min(nx, ny) * 0.8) # Use 80% of min dimension
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
                'figure_width_single': 8.9, # cm to inches
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
       
        # Tight layout
        # fig.tight_layout(rect=[0, 0, 1, 0.95]) # Removed to avoid conflict with constrained_layout
       
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
            '#1f77b4', # ISF - Blue
            '#ff7f0e', # ESF - Orange
            '#2ca02c', # Twin - Green
            '#d62728', # Red
            '#9467bd', # Purple
            '#8c564b' # Brown
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
    def add_error_shading(ax, x, y_mean, y_std, color='blue', alpha=0.3, label=''):
        """Add error shading to line plots"""
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                       color=color, alpha=alpha, label=label + ' ± std')
        return ax
   
    @staticmethod
    def add_confidence_band(ax, x, y_data, confidence=0.95, color='blue', alpha=0.2):
        """Add confidence band to line plots"""
        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        n = len(y_data)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        y_err = t_val * y_std / np.sqrt(n)
       
        ax.fill_between(x, y_mean - y_err, y_mean + y_err,
                       color=color, alpha=alpha, label=f'{int(confidence*100)}% CI')
        return ax, y_mean, y_err
   
    @staticmethod
    def add_scale_bar(ax, length_nm, location='lower right', color='black', linewidth=2):
        """Add scale bar to microscopy-style images"""
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
        elif location == 'upper right':
            x_pos = 0.95
            y_pos = 0.95
            ha = 'right'
            va = 'top'
        else:
            x_pos = 0.05
            y_pos = 0.95
            ha = 'left'
            va = 'top'
       
        # Convert to axis coordinates
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
               color=color, linewidth=linewidth, solid_capstyle='butt')
       
        # Add text
        ax.text((bar_x_start + bar_x_end) / 2, bar_y + y_range * 0.02,
               f'{length_nm} nm', ha='center', va='bottom',
               color=color, fontsize=8, fontweight='bold')
       
        return ax
   
    @staticmethod
    def create_fancy_legend(ax, lines, labels, **kwargs):
        """Create enhanced legend with better formatting"""
        legend = ax.legend(lines, labels, **kwargs)
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.9)
        return legend
   
    @staticmethod
    def add_annotations(ax, annotations, arrowstyle='->', **kwargs):
        """Add professional annotations with arrows"""
        for ann in annotations:
            ax.annotate(ann['text'], xy=ann['xy'], xytext=ann['xytext'],
                       arrowprops=dict(arrowstyle=arrowstyle, **kwargs),
                       **{k: v for k, v in ann.items() if k not in ['text', 'xy', 'xytext']})
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
    else: # Twin
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
            'angle_deg': np.rad2deg(theta),
            'steps': steps,
            'save_every': save_every,
            'eta_cmap': sim_eta_cmap_name,
            'sigma_cmap': sim_sigma_cmap_name,
            'hydro_cmap': sim_hydro_cmap_name,
            'vm_cmap': sim_vm_cmap_name
        }
else: # Compare Saved Simulations
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
                'line_style': comparison_line_style
            }
           
            # Add enhanced line profile config
            if comparison_type == "Overlay Line Profiles":
                st.session_state.comparison_config.update({
                    'profile_direction': profile_direction,
                    'selected_profiles': selected_profiles,
                    'position_ratio': position_ratio,
                    'custom_angle': custom_angle if profile_direction == "Custom Angle" else None
                })
           
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
    delta = 0.02 # Small dilatation
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
    szz = (C12 / (C11 + C12)) * (sxx + syy) # Plane strain approximation
   
    # Derived quantities (GPa)
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
   
    return {
        'sxx': sxx, 'syy': syy, 'sxy': sxy, 'szz': szz,
        'sigma_mag': sigma_mag, 'sigma_hydro': sigma_hydro, 'von_mises': von_mises,
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'eps_xx_star': eps_xx_star, 'eps_yy_star': eps_yy_star, 'eps_xy_star': eps_xy_star
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
    fig_width = journal_styles[journal]['figure_width_double'] / 2.54 # Convert cm to inches
   
    fig, axes = plt.subplots(rows, cols,
                            figsize=(fig_width, fig_width * 0.8 * rows/cols),
                            constrained_layout=True)
   
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
       
        # Create heatmap with enhanced settings
        im = ax.imshow(stress_data, extent=extent, cmap=cmap,
                      origin='lower', aspect='equal') # Fixed: aspect='equal' for proper scaling
       
        # Add contour lines for defect boundary
        contour = ax.contour(X, Y, eta, levels=[0.5], colors='white',
                           linewidths=1, linestyles='--', alpha=0.8)
       
        # Add scale bar
        PublicationEnhancer.add_scale_bar(ax, 5.0, location='lower right')
       
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
def create_enhanced_line_profiles(simulations, frames, config, style_params):
    """Enhanced line profile comparison with multiple directions"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
   
    # Get profile configuration
    profile_direction = config.get('profile_direction', 'Horizontal')
    selected_profiles = config.get('selected_profiles', ['Horizontal'])
    position_ratio = config.get('position_ratio', 0.5)
    custom_angle = config.get('custom_angle', 45.0)
   
    # Create figure layout based on number of profiles
    if profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        n_profiles = len(selected_profiles)
        fig = plt.figure(figsize=(16, 12))
        fig.set_constrained_layout(True)
       
        # Create subplot grid: 3 rows for multi-mode
        gs = fig.add_gridspec(3, 3)
       
        # Main profile plot (spanning 2 rows, 2 columns)
        ax_profiles = fig.add_subplot(gs[0:2, 0:2])
       
        # Statistical plot
        ax_stats = fig.add_subplot(gs[0, 2])
       
        # Location map
        ax_location = fig.add_subplot(gs[1, 2])
       
        # Individual profile plots row
        ax_individual = fig.add_subplot(gs[2, :])
       
        axes = [ax_profiles, ax_stats, ax_location, ax_individual]
       
    else:
        # Single profile mode
        fig = plt.figure(figsize=(14, 10))
        fig.set_constrained_layout(True)
       
        gs = fig.add_gridspec(2, 3)
        ax_profiles = fig.add_subplot(gs[0, 0:2])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_location = fig.add_subplot(gs[1, 0:2])
        ax_individual = fig.add_subplot(gs[1, 2])
       
        axes = [ax_profiles, ax_stats, ax_location, ax_individual]
   
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
   
    # Prepare data storage
    all_profiles = {profile_type: [] for profile_type in selected_profiles}
   
    # Define colors for different profile types
    profile_colors = {
        'Horizontal': 'red',
        'Vertical': 'blue',
        'Diagonal': 'green',
        'Anti-Diagonal': 'purple',
        'Custom': 'orange'
    }
   
    # Extract and plot profiles
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
       
        # Extract profiles for all selected types
        for profile_type in selected_profiles:
            # Handle custom angle
            if profile_type == "Custom Angle":
                profile_type_key = "custom"
                angle = custom_angle
            else:
                profile_type_key = profile_type.lower().replace(" ", "_").replace("-", "")
                angle = custom_angle
           
            # Extract profile
            if profile_type_key == "custom":
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    stress_data, 'custom', position_ratio, angle
                )
            else:
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    stress_data, profile_type_key, position_ratio, angle
                )
           
            # Store for statistics
            all_profiles[profile_type].append({
                'distance': distance,
                'profile': profile,
                'endpoints': endpoints,
                'color': color,
                'label': f"{sim['params']['defect_type']}"
            })
           
            # Plot on main axes if single profile mode
            if profile_direction != "Multiple Profiles" or len(selected_profiles) == 1:
                line_style = config.get('line_style', 'solid')
                ax_profiles.plot(distance, profile, color=color,
                               linewidth=style_params.get('line_width', 1.5),
                               linestyle=line_style,
                               label=f"{sim['params']['defect_type']}",
                               alpha=0.8)
   
    # Enhanced axis labeling for main profile plot
    if profile_direction != "Multiple Profiles" or len(selected_profiles) == 1:
        ax_profiles.set_xlabel("Position (nm)", fontsize=style_params.get('label_font_size', 10))
        ax_profiles.set_ylabel(f"{config['stress_component']} (GPa)",
                              fontsize=style_params.get('label_font_size', 10))
       
        profile_name = selected_profiles[0]
        if profile_name == "Custom Angle":
            profile_name = f"Custom ({custom_angle:.0f}°)"
        ax_profiles.set_title(f"{profile_name} Stress Profile",
                             fontsize=style_params.get('title_font_size', 12),
                             fontweight='bold')
       
        # Add legend
        PublicationEnhancer.create_fancy_legend(ax_profiles, *ax_profiles.get_legend_handles_labels(),
                                              loc='upper right', frameon=True,
                                              fancybox=True, shadow=False)
   
    # Multiple profiles mode
    elif profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        # Plot all profiles for each simulation
        line_styles = ['-', '--', '-.', ':']
       
        for idx, profile_type in enumerate(selected_profiles):
            profile_data = all_profiles[profile_type]
           
            for sim_idx, data in enumerate(profile_data):
                # Use different line styles for different profile types
                linestyle = line_styles[idx % len(line_styles)]
               
                ax_profiles.plot(data['distance'], data['profile'],
                               color=data['color'],
                               linewidth=style_params.get('line_width', 1.5),
                               linestyle=linestyle,
                               alpha=0.7,
                               label=f"{data['label']} - {profile_type}" if sim_idx == 0 else "")
       
        ax_profiles.set_xlabel("Position (nm)", fontsize=style_params.get('label_font_size', 10))
        ax_profiles.set_ylabel(f"{config['stress_component']} (GPa)",
                              fontsize=style_params.get('label_font_size', 10))
        ax_profiles.set_title("Multiple Stress Profiles",
                             fontsize=style_params.get('title_font_size', 12),
                             fontweight='bold')
       
        # Simplify legend for multiple profiles
        handles, labels = ax_profiles.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
       
        ax_profiles.legend(unique_handles, unique_labels, fontsize=8,
                          loc='upper right', frameon=True)
   
    # Panel B: Statistical summary
    if all_profiles:
        # Calculate statistics for each simulation
        stats_data = []
        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
           
            # Calculate basic statistics
            stats_data.append({
                'Defect': sim['params']['defect_type'],
                'Max': float(np.nanmax(stress_data)),
                'Mean': float(np.nanmean(stress_data)),
                'Std': float(np.nanstd(stress_data)),
                'color': colors[idx]
            })
       
        # Create bar plot
        defect_names = [stats['Defect'] for stats in stats_data]
        max_stresses = [stats['Max'] for stats in stats_data]
        colors_list = [stats['color'] for stats in stats_data]
       
        x_pos = np.arange(len(defect_names))
        bars = ax_stats.bar(x_pos, max_stresses, color=colors_list, alpha=0.7)
       
        ax_stats.set_xticks(x_pos)
        ax_stats.set_xticklabels(defect_names, rotation=45, ha='right')
        ax_stats.set_ylabel("Maximum Stress (GPa)", fontsize=9)
        ax_stats.set_title("Peak Stress Comparison",
                          fontsize=10, fontweight='bold')
       
        # Add value labels
        for bar, val in zip(bars, max_stresses):
            ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8)
   
    # Panel C: Show profile locations
    if simulations and selected_profiles:
        sim = simulations[0]
        eta, _ = sim['history'][frames[0]]
       
        # Prepare profile config for plotting
        profile_configs = {}
        for profile_type in selected_profiles:
            if profile_type == "Custom Angle":
                profile_type_key = "custom"
                angle = custom_angle
            else:
                profile_type_key = profile_type.lower().replace(" ", "_").replace("-", "")
                angle = custom_angle
           
            # Extract profile for visualization
            if profile_type_key == "custom":
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    eta, 'custom', position_ratio, angle
                )
            else:
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    eta, profile_type_key, position_ratio, angle
                )
           
            profile_configs[profile_type] = {
                'profiles': {profile_type: {'endpoints': endpoints}}
            }
       
        # Plot with profile locations
        im, ax_location = EnhancedLineProfiler.plot_profile_locations(
            ax_location, eta, profile_configs,
            cmap=enhanced_cmaps['plasma_enhanced'], alpha=0.7
        )
        ax_location.set_title("Profile Locations",
                             fontsize=10, fontweight='bold')
       
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_location, shrink=0.8)
        cbar.set_label('Defect Parameter η', fontsize=9)
   
    # Panel D: Individual profile comparison (for multiple profiles mode)
    if profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        # Plot each profile type in separate subplot
        n_cols = min(4, len(selected_profiles))
        n_rows = (len(selected_profiles) + n_cols - 1) // n_cols
       
        # Clear the individual axis and create subplots
        ax_individual.clear()
        fig.delaxes(ax_individual)
       
        # Create subplots for individual profiles
        gs_individual = gs[2, :].subgridspec(n_rows, n_cols)
        individual_axes = []
        for idx, profile_type in enumerate(selected_profiles):
            ax = fig.add_subplot(gs_individual[idx // n_cols, idx % n_cols])
            individual_axes.append(ax)
           
            profile_data = all_profiles[profile_type]
           
            for sim_idx, data in enumerate(profile_data):
                ax.plot(data['distance'], data['profile'],
                       color=data['color'],
                       linewidth=style_params.get('line_width', 1.0),
                       alpha=0.7,
                       label=data['label'] if sim_idx == 0 else "")
           
            ax.set_title(f"{profile_type} Profile", fontsize=9)
            ax.set_xlabel("Position (nm)", fontsize=8)
            ax.set_ylabel("Stress (GPa)", fontsize=8)
           
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
   
    # Apply publication styling
    if profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        # Get all axes for styling
        all_axes = [ax_profiles, ax_stats, ax_location] + individual_axes
        fig = EnhancedFigureStyler.apply_publication_styling(fig, all_axes, style_params)
       
        # Add panel labels
        for ax, label in zip([ax_profiles, ax_stats, ax_location] + individual_axes[:1], ['A', 'B', 'C', 'D']):
            ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='top')
    else:
        fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
       
        # Add panel labels
        for ax, label in zip(axes, ['A', 'B', 'C', 'D']):
            if ax is not None:
                ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='top')
   
    return fig
def create_publication_statistics(simulations, frames, config, style_params):
    """Publication-quality statistical analysis"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
   
    # Create multi-panel figure
    fig = plt.figure(figsize=(14, 10))
    fig.set_constrained_layout(True)
   
    # Define subplots
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2) # Box plot
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2) # Violin plot
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=2) # Histogram
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2) # Cumulative distribution
    ax5 = plt.subplot2grid((3, 4), (2, 0), colspan=4) # Statistical table
   
    # Get colors
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
   
    # Collect data
    all_data = []
    labels = []
   
    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key].flatten()
        stress_data = stress_data[np.isfinite(stress_data)]
       
        all_data.append(stress_data)
        labels.append(f"{sim['params']['defect_type']}\n({sim['params']['orientation'][:10]}...)")
   
    # Panel 1: Enhanced box plot
    bp = ax1.boxplot(all_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    meanprops=dict(color='white', linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='gray', linewidth=1),
                    capprops=dict(color='gray', linewidth=1),
                    boxprops=dict(linewidth=1))
   
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
   
    ax1.set_title(f"Distribution of {config['stress_component']}",
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel("Stress (GPa)", fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
   
    # Add mean values as text
    for i, data in enumerate(all_data):
        mean_val = np.mean(data)
        ax1.text(i + 1, mean_val, f'{mean_val:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
   
    # Panel 2: Violin plot
    parts = ax2.violinplot(all_data, showmeans=True, showmedians=True)
   
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
   
    ax2.set_title("Probability Density", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Stress (GPa)", fontsize=10)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels([sim['params']['defect_type'] for sim in simulations])
   
    # Panel 3: Histogram with KDE
    ax3.hist(all_data, bins=30, density=True, stacked=True,
            label=[sim['params']['defect_type'] for sim in simulations],
            color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
   
    # Add KDE
    for data, color, label in zip(all_data, colors, labels):
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data.min() for data in all_data),
                             max(data.max() for data in all_data), 100)
        ax3.plot(x_range, kde(x_range), color=color, linewidth=2, label=label.split('\n')[0])
   
    ax3.set_title("Histogram with KDE", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Stress (GPa)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=8)
   
    # Panel 4: Cumulative distribution
    for data, color, label in zip(all_data, colors, labels):
        sorted_data = np.sort(data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, y_vals, color=color, linewidth=2, label=label.split('\n')[0])
   
    ax4.set_title("Cumulative Distribution", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Stress (GPa)", fontsize=10)
    ax4.set_ylabel("Cumulative Probability", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--')
   
    # Panel 5: Statistical table
    ax5.axis('off')
   
    # Create comprehensive statistics table
    table_data = []
    columns = ['Defect', 'N', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max', 'Skew', 'Kurtosis']
   
    for idx, (data, sim) in enumerate(zip(all_data, simulations)):
        table_data.append([
            sim['params']['defect_type'],
            len(data),
            f"{np.mean(data):.3f}",
            f"{np.std(data):.3f}",
            f"{np.min(data):.3f}",
            f"{np.percentile(data, 25):.3f}",
            f"{np.median(data):.3f}",
            f"{np.percentile(data, 75):.3f}",
            f"{np.max(data):.3f}",
            f"{stats.skew(data):.3f}",
            f"{stats.kurtosis(data):.3f}"
        ])
   
    # Create table
    table = ax5.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#f2f2f2']*len(columns))
   
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
   
    # Color code cells
    for i in range(len(table_data)):
        for j in range(1, len(columns)): # Skip first column (Defect)
            table[(i+1, j)].set_facecolor(mpl.colors.to_rgba(colors[i], 0.3)) # Add alpha
   
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5], style_params)
   
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4], ['A', 'B', 'C', 'D']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
   
    return fig
def create_publication_correlation(simulations, frames, config, style_params):
    """Publication-quality correlation analysis"""
    # Component mapping
    component_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises',
        "Defect Parameter η": 'eta'
    }
   
    x_key = component_map[config.get('correlation_x', 'Defect Parameter η')]
    y_key = component_map[config.get('correlation_y', 'Stress Magnitude |σ|')]
   
    # Create multi-panel figure
    fig = plt.figure(figsize=(15, 12))
    fig.set_constrained_layout(True)
   
    # Define subplot grid
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2) # Scatter with regression
    ax2 = plt.subplot2grid((3, 3), (0, 2)) # Correlation coefficients
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=1) # Residuals
    ax4 = plt.subplot2grid((3, 3), (1, 1), colspan=1) # QQ plot
    ax5 = plt.subplot2grid((3, 3), (1, 2), colspan=1) # Histogram of residuals
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3) # Regression parameters
   
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
   
    # Store regression results
    regression_results = []
   
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
       
        # Prepare x data
        if x_key == 'eta':
            x_data = eta.flatten()
        else:
            x_data = stress_fields[x_key].flatten()
       
        # Prepare y data
        if y_key == 'eta':
            y_data = eta.flatten()
        else:
            y_data = stress_fields[y_key].flatten()
       
        # Sample data for clarity
        sample_size = min(5000, len(x_data))
        indices = np.random.choice(len(x_data), sample_size, replace=False)
        x_sampled = x_data[indices]
        y_sampled = y_data[indices]
       
        # Remove outliers
        q_low, q_high = np.percentile(x_sampled, [1, 99])
        mask = (x_sampled > q_low) & (x_sampled < q_high)
        x_sampled = x_sampled[mask]
        y_sampled = y_sampled[mask]
       
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_sampled, y_sampled)
       
        # Store results
        regression_results.append({
            'defect': sim['params']['defect_type'],
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'n': len(x_sampled)
        })
       
        # Panel 1: Scatter with regression line
        scatter = ax1.scatter(x_sampled, y_sampled, color=color, alpha=0.3,
                            s=10, edgecolors='none', label=sim['params']['defect_type'])
       
        # Add regression line
        x_range = np.linspace(np.min(x_sampled), np.max(x_sampled), 100)
        y_pred = slope * x_range + intercept
        ax1.plot(x_range, y_pred, color=color, linewidth=2, alpha=0.8,
                label=f"R = {r_value:.3f}")
       
        # Panel 3: Residuals
        y_pred_points = slope * x_sampled + intercept
        residuals = y_sampled - y_pred_points
       
        ax3.scatter(y_pred_points, residuals, color=color, alpha=0.3, s=10)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
       
        # Panel 4: QQ plot
        if idx == 0: # Plot QQ for first simulation
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.get_lines()[0].set_marker('.')
            ax4.get_lines()[0].set_markersize(5)
            ax4.get_lines()[0].set_alpha(0.5)
            ax4.get_lines()[1].set_color('red')
            ax4.get_lines()[1].set_linewidth(2)
       
        # Panel 5: Histogram of residuals
        ax5.hist(residuals, bins=30, density=True, alpha=0.5, color=color)
   
    # Enhance Panel 1
    ax1.set_xlabel(config.get('correlation_x', 'X Component'), fontsize=11)
    ax1.set_ylabel(config.get('correlation_y', 'Y Component'), fontsize=11)
    ax1.set_title(f"Scatter Plot with Linear Regression", fontsize=12, fontweight='bold')
   
    # Create enhanced legend
    PublicationEnhancer.create_fancy_legend(ax1, *ax1.get_legend_handles_labels(),
                                          loc='upper left', frameon=True,
                                          fancybox=True, shadow=True, ncol=2)
   
    # Panel 2: Correlation coefficients
    defect_names = [sim['params']['defect_type'] for sim in simulations]
    r_values = [result['r_value'] for result in regression_results]
   
    bars = ax2.bar(range(len(defect_names)), r_values, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(defect_names)))
    ax2.set_xticklabels(defect_names, rotation=45, ha='right')
    ax2.set_ylabel("Correlation Coefficient (R)", fontsize=10)
    ax2.set_title("Correlation Strength", fontsize=11, fontweight='bold')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=1)
   
    # Add value labels
    for bar, val in zip(bars, r_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
   
    # Enhance Panel 3: Residuals
    ax3.set_xlabel("Predicted Values", fontsize=10)
    ax3.set_ylabel("Residuals", fontsize=10)
    ax3.set_title("Residual Plot", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
   
    # Enhance Panel 4: QQ Plot
    ax4.set_title("Q-Q Plot of Residuals", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax4.set_ylabel("Sample Quantiles", fontsize=10)
   
    # Enhance Panel 5: Histogram of residuals
    ax5.set_title("Distribution of Residuals", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Residuals", fontsize=10)
    ax5.set_ylabel("Density", fontsize=10)
    ax5.legend([sim['params']['defect_type'] for sim in simulations], fontsize=8)
   
    # Panel 6: Regression parameters table
    ax6.axis('off')
   
    # Create detailed table
    table_data = []
    columns = ['Defect', 'Slope', 'Intercept', 'R', 'R²', 'p-value', 'Std Error', 'N']
   
    for result in regression_results:
        table_data.append([
            result['defect'],
            f"{result['slope']:.4f}",
            f"{result['intercept']:.4f}",
            f"{result['r_value']:.4f}",
            f"{result['r_squared']:.4f}",
            f"{result['p_value']:.3e}",
            f"{result['std_err']:.4f}",
            f"{result['n']:,}"
        ])
   
    table = ax6.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#f2f2f2']*len(columns))
   
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
   
    # Color code p-values
    for i in range(len(table_data)):
        p_val = float(table_data[i][5].replace('e-', 'E-'))
        if p_val < 0.001:
            table[(i+1, 5)].set_text_props(fontweight='bold', color='green')
        elif p_val < 0.01:
            table[(i+1, 5)].set_text_props(fontweight='bold', color='orange')
   
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5, ax6], style_params)
   
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
   
    return fig
def create_enhanced_comparison_plot(simulations, frames, config, style_params):
    """Create publication-quality comparison plots"""
   
    # Create figure based on comparison type
    if config['type'] == "Side-by-Side Heatmaps":
        return create_publication_heatmaps(simulations, frames, config, style_params)
    elif config['type'] == "Overlay Line Profiles":
        return create_enhanced_line_profiles(simulations, frames, config, style_params)
    elif config['type'] == "Statistical Summary":
        return create_publication_statistics(simulations, frames, config, style_params)
    elif config['type'] == "Defect-Stress Correlation":
        return create_publication_correlation(simulations, frames, config, style_params)
    else:
        # Fall back to simpler visualization for other types
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
    ax.set_ylabel(f"Mean {config['stress_component']} (GPa)", fontsize=style_params.get('label_font_size', 12))
