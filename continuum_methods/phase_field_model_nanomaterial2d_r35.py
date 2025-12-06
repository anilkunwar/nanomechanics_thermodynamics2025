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
from scipy.spatial import KDTree, Voronoi
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIGURATION WITH ENHANCED STYLING
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
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #ffd166 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2D3748 !important;
        border-left: 4px solid #667eea !important;
        padding-left: 1rem !important;
        margin: 1.5rem 0 1rem 0 !important;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e7f0 100%);
        padding: 1.8rem;
        border-radius: 20px;
        margin: 1.2rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.15);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.85rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5) !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 6px solid #667eea;
        margin: 0.8rem 0;
    }
    .metric-value {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
    }
    .metric-label {
        font-size: 0.9rem !important;
        color: #4A5568 !important;
        font-weight: 600 !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #F7FAFC;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #EDF2F7;
        border-radius: 8px;
        gap: 1rem;
        padding: 0 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# ENHANCED HEADER WITH ANIMATED GRADIENT
# =============================================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">🔬 Ag Nanoparticle Multi-Orientation Defect Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem;">
        <h3 style="color: #4A5568; font-weight: 600; margin-bottom: 0.5rem;">🎯 Advanced Multi-Orientation Stress Analysis • Fixed Aspect Ratio • Publication-Ready Output</h3>
        <p style="font-size: 1.1rem; color: #718096; margin-bottom: 0.5rem;">
            <strong>Diagonal & Arbitrary Orientations • Multi-Defect Comparison • 60+ Colormaps • Scientific Insights</strong>
        </p>
        <p style="font-size: 1rem; color: #A0AEC0;">
            <strong>Run → Analyze → Compare → Export • Enhanced Post-Processing • Statistical Validation</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# MATERIAL & GRID PARAMETERS WITH ENHANCED PHYSICS
# =============================================
a = 0.4086  # FCC Ag lattice constant (nm) - Experimental value
b = a / np.sqrt(6)  # Burgers vector magnitude for Shockley partial
d111 = a / np.sqrt(3)  # {111} interplanar spacing

# Enhanced elastic constants for FCC Ag with uncertainty bounds (GPa)
# Reference: Hirth & Lothe, Theory of Dislocations
C11 = 124.0  # ± 2 GPa
C12 = 93.4   # ± 2 GPa
C44 = 46.1   # ± 1 GPa

# Derived elastic moduli
E = C11 - 2 * C12**2 / (C11 + C12)  # Young's modulus approximation
nu = C12 / (C11 + C12)  # Poisson's ratio
G = C44  # Shear modulus

# Crystal directions (Miller indices)
directions = {
    '001': np.array([0, 0, 1]),
    '110': np.array([1, 1, 0]),
    '111': np.array([1, 1, 1]),
    '112': np.array([1, 1, 2])
}

# Grid parameters - Enhanced resolution
N = 256  # Increased for better diagonal sampling
dx = 0.08  # Finer grid spacing (nm)
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# Create radial coordinate system for circular analyses
R = np.sqrt(X**2 + Y**2)
THETA = np.arctan2(Y, X)

# =============================================
# ENHANCED COLORMAP LIBRARY WITH DIAGONAL-OPTIMIZED PALETTES
# =============================================
COLORMAPS = {
    # Enhanced diagonal-aware sequential
    'diagonal_sequential': mpl.colors.LinearSegmentedColormap.from_list(
        'diagonal_sequential', 
        ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    ),
    'stress_diagonal': mpl.colors.LinearSegmentedColormap.from_list(
        'stress_diagonal',
        ['#00008b', '#1e90ff', '#00bfff', '#00ffff', '#ffff00', '#ffa500', '#ff4500', '#8b0000']
    ),
    
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
    
    # Diverging - Enhanced for stress visualization
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

# Create specialized colormaps for diagonal analysis
cmap_list = list(COLORMAPS.keys())

# =============================================
# DIAGONAL & MULTI-ORIENTATION PROFILE EXTRACTION SYSTEM
# =============================================
class DiagonalProfileAnalyzer:
    """Advanced system for analyzing stress and defect profiles along diagonal and arbitrary orientations"""
    
    def __init__(self, X, Y, N, dx):
        self.X = X
        self.Y = Y
        self.N = N
        self.dx = dx
        self.center = (0, 0)
        
        # Pre-compute diagonal lines at 45° and 135°
        self.diagonal_45 = self._create_diagonal_line(45, length=1.0)
        self.diagonal_135 = self._create_diagonal_line(135, length=1.0)
        
        # Store frequently used orientations
        self.standard_orientations = {
            'horizontal': 0,
            'vertical': 90,
            'diagonal_45': 45,
            'diagonal_135': 135,
            'anti-diagonal_225': 225,
            'anti-diagonal_315': 315
        }
    
    def _create_diagonal_line(self, angle_deg, length=0.8):
        """Create a diagonal line through the center"""
        angle_rad = np.deg2rad(angle_deg)
        half_length = (self.N * self.dx * length) / 2
        
        x_start = -half_length * np.cos(angle_rad)
        y_start = -half_length * np.sin(angle_rad)
        x_end = half_length * np.cos(angle_rad)
        y_end = half_length * np.sin(angle_rad)
        
        return {
            'angle': angle_deg,
            'start': (x_start, y_start),
            'end': (x_end, y_end),
            'length': 2 * half_length,
            'points': self._generate_line_points(x_start, y_start, x_end, y_end)
        }
    
    def _generate_line_points(self, x_start, y_start, x_end, y_end, num_points=500):
        """Generate points along a line with high sampling"""
        return np.linspace(x_start, x_end, num_points), np.linspace(y_start, y_end, num_points)
    
    def extract_diagonal_profile(self, data, angle_deg=45, offset=0, length_ratio=0.8):
        """
        Extract profile along diagonal direction with enhanced accuracy
        
        Parameters:
        -----------
        data : 2D array
            Input data (stress or defect field)
        angle_deg : float
            Diagonal angle (45°, 135°, etc.)
        offset : float
            Offset from center (perpendicular distance in nm)
        length_ratio : float
            Length relative to domain size (0-1)
        
        Returns:
        --------
        distances : 1D array
            Distance along diagonal (nm)
        profile : 1D array
            Extracted values
        metadata : dict
            Analysis metadata
        """
        angle_rad = np.deg2rad(angle_deg)
        perp_angle = angle_rad + np.pi/2
        
        # Calculate diagonal length
        domain_size = self.N * self.dx
        half_length = domain_size * length_ratio / 2
        
        # Apply offset perpendicular to diagonal
        offset_x = offset * np.cos(perp_angle)
        offset_y = offset * np.sin(perp_angle)
        
        # Generate sampling points
        distances = np.linspace(-half_length, half_length, 500)
        xs = offset_x + distances * np.cos(angle_rad)
        ys = offset_y + distances * np.sin(angle_rad)
        
        # Convert to array indices with boundary checking
        xi = (xs - extent[0]) / (extent[1] - extent[0]) * (self.N - 1)
        yi = (ys - extent[2]) / (extent[3] - extent[2]) * (self.N - 1)
        
        # Clip indices to valid range
        xi = np.clip(xi, 0, self.N - 1)
        yi = np.clip(yi, 0, self.N - 1)
        
        # Extract profile with cubic interpolation
        profile = map_coordinates(data, [yi, xi], order=3, mode='nearest')
        
        # Calculate diagonal statistics
        metadata = {
            'angle_deg': angle_deg,
            'offset_nm': offset,
            'length_nm': 2 * half_length,
            'max_value': float(np.nanmax(profile)),
            'min_value': float(np.nanmin(profile)),
            'mean_value': float(np.nanmean(profile)),
            'std_value': float(np.nanstd(profile)),
            'fwhm_nm': self._calculate_diagonal_fwhm(distances, profile),
            'skewness': float(stats.skew(profile)),
            'kurtosis': float(stats.kurtosis(profile)),
            'gradient_max': float(np.nanmax(np.abs(np.gradient(profile)))),
            'symmetry_index': self._calculate_symmetry_index(profile)
        }
        
        return distances, profile, metadata
    
    def _calculate_diagonal_fwhm(self, distances, profile):
        """Calculate FWHM specifically for diagonal profiles"""
        if len(profile) == 0:
            return 0.0
        
        profile_norm = profile - np.nanmin(profile)
        max_val = np.nanmax(profile_norm)
        
        if max_val == 0:
            return 0.0
        
        half_max = max_val / 2
        
        # Find all crossings
        above_half = profile_norm >= half_max
        
        if np.sum(above_half) < 2:
            return 0.0
        
        # Get leftmost and rightmost crossings
        indices = np.where(above_half)[0]
        left_idx = indices[0]
        right_idx = indices[-1]
        
        return float(distances[right_idx] - distances[left_idx])
    
    def _calculate_symmetry_index(self, profile):
        """Calculate symmetry index (0=perfect symmetry, 1=perfect asymmetry)"""
        if len(profile) % 2 == 0:
            left_half = profile[:len(profile)//2]
            right_half = profile[len(profile)//2:][::-1]  # Reverse for comparison
        else:
            left_half = profile[:len(profile)//2]
            right_half = profile[len(profile)//2 + 1:][::-1]
        
        if len(left_half) != len(right_half) or len(left_half) == 0:
            return 1.0
        
        # Normalize profiles for comparison
        left_norm = (left_half - np.mean(left_half)) / np.std(left_half) if np.std(left_half) > 0 else left_half
        right_norm = (right_half - np.mean(right_half)) / np.std(right_half) if np.std(right_half) > 0 else right_half
        
        # Calculate correlation coefficient
        corr = np.corrcoef(left_norm, right_norm)[0, 1]
        
        # Convert to symmetry index (0 to 1)
        return max(0, 1 - abs(corr))
    
    def extract_multi_diagonal_profiles(self, data, angles=[45, 135], offsets=None):
        """Extract profiles along multiple diagonal orientations"""
        if offsets is None:
            offsets = [0] * len(angles)
        
        profiles = {}
        for angle, offset in zip(angles, offsets):
            distances, profile, metadata = self.extract_diagonal_profile(
                data, angle, offset
            )
            
            key = f"diagonal_{angle}°_offset{offset}"
            profiles[key] = {
                'distances': distances,
                'profile': profile,
                'metadata': metadata,
                'angle': angle,
                'offset': offset
            }
        
        return profiles
    
    def analyze_diagonal_symmetry(self, data, reference_angle=45):
        """
        Analyze symmetry properties between diagonal directions
        
        Returns:
        --------
        symmetry_metrics : dict
            Metrics comparing 45° and 135° diagonals
        """
        # Extract both diagonal profiles
        dist_45, prof_45, meta_45 = self.extract_diagonal_profile(data, 45)
        dist_135, prof_135, meta_135 = self.extract_diagonal_profile(data, 135)
        
        # Ensure same length for comparison
        min_len = min(len(prof_45), len(prof_135))
        prof_45 = prof_45[:min_len]
        prof_135 = prof_135[:min_len]
        
        # Calculate symmetry metrics
        correlation = np.corrcoef(prof_45, prof_135)[0, 1]
        rms_difference = np.sqrt(np.mean((prof_45 - prof_135)**2))
        max_difference = np.max(np.abs(prof_45 - prof_135))
        
        symmetry_metrics = {
            'correlation_coefficient': float(correlation),
            'rms_difference': float(rms_difference),
            'max_difference': float(max_difference),
            'relative_difference': float(rms_difference / (0.5 * (np.std(prof_45) + np.std(prof_135)))),
            'is_symmetric': abs(correlation) > 0.8 and rms_difference < 0.1 * np.max([np.std(prof_45), np.std(prof_135)])
        }
        
        return symmetry_metrics
    
    def create_diagonal_grid_analysis(self, data, n_diagonals=12, max_angle=180):
        """
        Create comprehensive analysis of stress along multiple diagonals
        
        Returns:
        --------
        grid_results : dict
            Analysis results for each diagonal
        """
        angles = np.linspace(0, max_angle, n_diagonals, endpoint=False)
        
        grid_results = {}
        for angle in angles:
            key = f"{angle:.1f}°"
            distances, profile, metadata = self.extract_diagonal_profile(data, angle)
            
            # Calculate additional metrics
            gradient = np.gradient(profile, distances)
            curvature = np.gradient(gradient, distances)
            
            # Find peaks and valleys
            from scipy.signal import find_peaks
            peaks, peak_props = find_peaks(profile, height=np.mean(profile))
            valleys, valley_props = find_peaks(-profile, height=-np.mean(profile))
            
            grid_results[key] = {
                'distances': distances,
                'profile': profile,
                'metadata': metadata,
                'gradient': gradient,
                'curvature': curvature,
                'peaks': {
                    'indices': peaks,
                    'positions': distances[peaks] if len(peaks) > 0 else [],
                    'heights': profile[peaks] if len(peaks) > 0 else []
                },
                'valleys': {
                    'indices': valleys,
                    'positions': distances[valleys] if len(valleys) > 0 else [],
                    'depths': profile[valleys] if len(valleys) > 0 else []
                }
            }
        
        return grid_results

# =============================================
# ENHANCED VISUALIZATION WITH DIAGONAL ANALYSIS
# =============================================
class DiagonalVisualizer:
    """Advanced visualization system for diagonal and multi-orientation analysis"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def create_diagonal_comparison_plot(self, simulations, frames, config, style_params):
        """Create comprehensive diagonal comparison visualization"""
        
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map[config.get('stress_component', 'Stress Magnitude |σ|')]
        
        # Get selected diagonal orientations
        diagonal_angles = config.get('diagonal_angles', [45, 135])
        
        # Create figure with optimal layout for diagonals
        fig = plt.figure(figsize=(18, 14))
        fig.set_constrained_layout(True)
        
        # Define grid for multiple panels
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2], 
                             hspace=0.25, wspace=0.3)
        
        # Panel definitions
        ax1 = fig.add_subplot(gs[0, :2])  # Domain with diagonal lines
        ax2 = fig.add_subplot(gs[0, 2:])  # 45° diagonal profiles
        ax3 = fig.add_subplot(gs[1, :2])  # 135° diagonal profiles
        ax4 = fig.add_subplot(gs[1, 2:])  # Diagonal symmetry analysis
        ax5 = fig.add_subplot(gs[2, 0:2]) # Radial distribution
        ax6 = fig.add_subplot(gs[2, 2:])  # Polar plot of stress maxima
        
        # Get colors for simulations
        colors = plt.cm.viridis(np.linspace(0, 1, len(simulations)))
        
        # Panel 1: Domain with diagonal lines
        if simulations:
            sim = simulations[0]
            eta, _ = sim['history'][frames[0]]
            
            # Plot domain with fixed aspect ratio
            ax1.set_aspect('equal')
            cmap = plt.cm.get_cmap(COLORMAPS.get(sim['params'].get('eta_cmap', 'viridis'), 'viridis'))
            im = ax1.imshow(eta, extent=extent, cmap=cmap, origin='lower', alpha=0.8)
            
            # Add diagonal lines with enhanced styling
            line_styles = ['-', '--', '-.', ':']
            line_colors = ['red', 'blue', 'green', 'orange']
            
            for idx, angle in enumerate(diagonal_angles):
                # Create diagonal line
                angle_rad = np.deg2rad(angle)
                length = 0.7 * max(extent[1] - extent[0], extent[3] - extent[2])
                x_start = -length/2 * np.cos(angle_rad)
                y_start = -length/2 * np.sin(angle_rad)
                x_end = length/2 * np.cos(angle_rad)
                y_end = length/2 * np.sin(angle_rad)
                
                # Plot line with arrow
                ax1.plot([x_start, x_end], [y_start, y_end], 
                        color=line_colors[idx % len(line_colors)],
                        linewidth=3,
                        linestyle=line_styles[idx % len(line_styles)],
                        alpha=0.9,
                        label=f'{angle}°')
                
                # Add angle annotation
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                ax1.annotate(f'{angle}°', xy=(mid_x, mid_y),
                           xytext=(10, 10), textcoords='offset points',
                           color=line_colors[idx % len(line_colors)],
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                    facecolor='white', alpha=0.8))
            
            ax1.set_xlabel("x (nm)", fontsize=12)
            ax1.set_ylabel("y (nm)", fontsize=12)
            ax1.set_title("Defect Domain with Diagonal Lines", 
                         fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10, loc='upper right')
            
            # Add scale bar
            self._add_enhanced_scale_bar(ax1, 5.0, location='lower right')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
            cbar.set_label('Defect Parameter η', fontsize=11)
        
        # Panel 2: 45° diagonal profiles
        ax2.set_title("45° Diagonal Profiles", fontsize=13, fontweight='bold')
        self._plot_diagonal_profiles(ax2, simulations, frames, 45, stress_key, colors, config)
        
        # Panel 3: 135° diagonal profiles
        ax3.set_title("135° Diagonal Profiles", fontsize=13, fontweight='bold')
        self._plot_diagonal_profiles(ax3, simulations, frames, 135, stress_key, colors, config)
        
        # Panel 4: Diagonal symmetry analysis
        if len(simulations) >= 2:
            self._plot_diagonal_symmetry(ax4, simulations, frames, stress_key, colors, config)
        else:
            ax4.text(0.5, 0.5, "Need at least 2 simulations\nfor symmetry analysis",
                    ha='center', va='center', fontsize=12)
            ax4.set_axis_off()
        
        # Panel 5: Radial distribution along diagonals
        self._plot_radial_distribution(ax5, simulations, frames, stress_key, colors, config)
        
        # Panel 6: Polar plot of stress maxima
        self._plot_polar_maxima(ax6, simulations, frames, stress_key, colors, config)
        
        # Add panel labels
        for ax, label in zip([ax1, ax2, ax3, ax4, ax5, ax6], ['A', 'B', 'C', 'D', 'E', 'F']):
            ax.text(-0.05, 1.05, label, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', va='top')
        
        return fig
    
    def _plot_diagonal_profiles(self, ax, simulations, frames, angle, stress_key, colors, config):
        """Plot diagonal profiles for multiple simulations"""
        
        line_style = config.get('line_style', 'solid')
        
        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Extract diagonal profile
            distances, profile, metadata = self.analyzer.extract_diagonal_profile(
                stress_data, angle, length_ratio=0.8
            )
            
            # Plot with enhanced styling
            ax.plot(distances, profile, 
                   color=colors[idx],
                   linewidth=2.5,
                   linestyle=line_style,
                   alpha=0.8,
                   label=f"{sim['params']['defect_type']}")
            
            # Add statistical annotations
            max_idx = np.argmax(profile)
            ax.annotate(f"max: {profile[max_idx]:.2f} GPa",
                       xy=(distances[max_idx], profile[max_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, color=colors[idx],
                       arrowprops=dict(arrowstyle='->', color=colors[idx], alpha=0.7))
        
        ax.set_xlabel("Distance along Diagonal (nm)", fontsize=11)
        ax.set_ylabel(f"Stress (GPa)", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        if angle == 45:
            ax.legend(fontsize=10, loc='upper right')
    
    def _plot_diagonal_symmetry(self, ax, simulations, frames, stress_key, colors, config):
        """Plot symmetry analysis between diagonals"""
        
        # Calculate symmetry metrics for each simulation
        symmetry_data = []
        
        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            symmetry_metrics = self.analyzer.analyze_diagonal_symmetry(stress_data)
            symmetry_data.append({
                'defect': sim['params']['defect_type'],
                'correlation': symmetry_metrics['correlation_coefficient'],
                'rms_diff': symmetry_metrics['rms_difference'],
                'is_symmetric': symmetry_metrics['is_symmetric'],
                'color': colors[idx]
            })
        
        # Create bar plot of symmetry metrics
        x_pos = np.arange(len(symmetry_data))
        width = 0.35
        
        # Plot correlation coefficients
        correlations = [d['correlation'] for d in symmetry_data]
        bars1 = ax.bar(x_pos - width/2, correlations, width, 
                      color=[d['color'] for d in symmetry_data],
                      alpha=0.7, label='Correlation')
        
        # Plot RMS differences (normalized)
        rms_diffs = [d['rms_diff'] for d in symmetry_data]
        max_rms = max(rms_diffs) if rms_diffs else 1.0
        normalized_rms = [d/max_rms for d in rms_diffs]
        bars2 = ax.bar(x_pos + width/2, normalized_rms, width,
                      color=[d['color'] for d in symmetry_data],
                      alpha=0.5, label='RMS Diff (norm)')
        
        ax.set_xlabel("Simulation", fontsize=11)
        ax.set_ylabel("Symmetry Metric", fontsize=11)
        ax.set_title("Diagonal Symmetry Analysis", fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d['defect'] for d in symmetry_data], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars1, correlations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add symmetry indicators
        for i, d in enumerate(symmetry_data):
            if d['is_symmetric']:
                ax.text(i, 1.05, "✓", ha='center', va='bottom', 
                       fontsize=12, color='green', fontweight='bold')
            else:
                ax.text(i, 1.05, "✗", ha='center', va='bottom',
                       fontsize=12, color='red', fontweight='bold')
    
    def _plot_radial_distribution(self, ax, simulations, frames, stress_key, colors, config):
        """Plot radial distribution of stress along diagonals"""
        
        # Define radial bins
        r_bins = np.linspace(0, N*dx/2, 30)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Calculate radial profiles along different diagonals
            radial_profiles = {}
            
            for angle in [0, 45, 90, 135]:
                distances, profile, _ = self.analyzer.extract_diagonal_profile(
                    stress_data, angle, length_ratio=0.8
                )
                
                # Bin by radial distance
                radial_means = []
                for i in range(len(r_bins)-1):
                    mask = (np.abs(distances) >= r_bins[i]) & (np.abs(distances) < r_bins[i+1])
                    if np.any(mask):
                        radial_means.append(np.mean(profile[mask]))
                    else:
                        radial_means.append(np.nan)
                
                radial_profiles[angle] = radial_means
            
            # Plot radial distribution for 45° diagonal
            valid_mask = ~np.isnan(radial_profiles[45])
            if np.any(valid_mask):
                ax.plot(r_centers[valid_mask], np.array(radial_profiles[45])[valid_mask],
                       color=colors[idx], linewidth=2, alpha=0.7,
                       label=f"{sim['params']['defect_type']}")
        
        ax.set_xlabel("Radial Distance (nm)", fontsize=11)
        ax.set_ylabel("Average Stress (GPa)", fontsize=11)
        ax.set_title("Radial Stress Distribution (45° Diagonal)", 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='upper right')
    
    def _plot_polar_maxima(self, ax, simulations, frames, stress_key, colors, config):
        """Create polar plot of stress maxima at different angles"""
        
        # Define angles for analysis
        angles = np.linspace(0, 360, 24, endpoint=False)
        
        # Convert to polar plot
        ax = plt.subplot(projection='polar')
        
        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Find maximum stress at each angle
            maxima = []
            for angle in angles:
                distances, profile, _ = self.analyzer.extract_diagonal_profile(
                    stress_data, angle, length_ratio=0.6
                )
                if len(profile) > 0:
                    maxima.append(np.nanmax(profile))
                else:
                    maxima.append(0.0)
            
            # Convert to radians and plot
            theta = np.deg2rad(angles)
            ax.plot(theta, maxima, color=colors[idx], linewidth=2, alpha=0.7,
                   label=f"{sim['params']['defect_type']}")
            
            # Fill area under curve
            ax.fill(theta, maxima, color=colors[idx], alpha=0.2)
        
        ax.set_title("Polar Distribution of Stress Maxima", 
                    fontsize=13, fontweight='bold', pad=20)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True, alpha=0.5)
    
    def _add_enhanced_scale_bar(self, ax, length_nm, location='lower right'):
        """Add enhanced scale bar with better styling"""
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
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
        else:
            return
        
        # Draw scale bar with enhanced styling
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
               color='white', linewidth=3, solid_capstyle='butt',
               zorder=1000)
        
        # Add end caps
        cap_length = y_range * 0.015
        ax.plot([bar_x_start, bar_x_start], [bar_y - cap_length, bar_y + cap_length],
               color='white', linewidth=3, solid_capstyle='butt',
               zorder=1000)
        ax.plot([bar_x_end, bar_x_end], [bar_y - cap_length, bar_y + cap_length],
               color='white', linewidth=3, solid_capstyle='butt',
               zorder=1000)
        
        # Add text with enhanced background
        text = ax.text((bar_x_start + bar_x_end) / 2, text_y,
                      f'{length_nm} nm', ha=text_ha, va=text_va,
                      color='white', fontsize=11, fontweight='bold',
                      zorder=1001)
        
        # Enhanced text background
        text.set_bbox(dict(boxstyle="round,pad=0.5", 
                          facecolor='black', alpha=0.7,
                          edgecolor='white', linewidth=1))

# =============================================
# ENHANCED SIMULATION ENGINE WITH DIAGONAL SUPPORT
# =============================================
@st.cache_data
def create_initial_eta_diagonal(shape, defect_type, orientation_angle=0, random_seed=42):
    """Create initial defect configuration with diagonal orientation support"""
    np.random.seed(random_seed)
    
    # Set initial amplitude based on defect type
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes.get(defect_type, 0.75)
    
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    
    # Convert orientation angle to radians
    theta = np.deg2rad(orientation_angle)
    
    # Rotation matrix for diagonal orientation
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    
    if shape == "Diagonal Fault":
        # Create a diagonal fault line
        width, height = 40, 8
        for i in range(N):
            for j in range(N):
                # Rotate coordinates
                x_rot = (i - cx) * np.cos(theta) - (j - cy) * np.sin(theta)
                y_rot = (i - cx) * np.sin(theta) + (j - cy) * np.cos(theta)
                
                if abs(x_rot) < width/2 and abs(y_rot) < height/2:
                    eta[i, j] = init_amplitude
    
    elif shape == "Rotated Rectangle":
        width, height = 30, 15
        for i in range(N):
            for j in range(N):
                x_rot = (i - cx) * np.cos(theta) - (j - cy) * np.sin(theta)
                y_rot = (i - cx) * np.sin(theta) + (j - cy) * np.cos(theta)
                
                if abs(x_rot) < width/2 and abs(y_rot) < height/2:
                    eta[i, j] = init_amplitude
    
    elif shape == "Rotated Ellipse":
        a, b = 25, 15
        for i in range(N):
            for j in range(N):
                x_rot = (i - cx) * np.cos(theta) - (j - cy) * np.sin(theta)
                y_rot = (i - cx) * np.sin(theta) + (j - cy) * np.cos(theta)
                
                if (x_rot/a)**2 + (y_rot/b)**2 <= 1:
                    eta[i, j] = init_amplitude
    
    else:
        # Use standard shapes for backward compatibility
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
            a, b = 25, 15
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
    
    # Add controlled noise with diagonal correlation
    noise_level = 0.02
    noise = noise_level * np.random.randn(N, N)
    
    # Apply slight diagonal correlation to noise
    if orientation_angle != 0:
        noise = gaussian_filter(noise, sigma=1, order=0, mode='reflect')
    
    eta += noise
    
    return np.clip(eta, 0.0, 1.0)

def compute_enhanced_stress_fields(eta, eps0, theta):
    """Enhanced stress solver with diagonal symmetry analysis"""
    # Standard stress computation
    stress_fields = compute_stress_fields(eta, eps0, theta)
    
    # Add diagonal analysis metrics
    analyzer = DiagonalProfileAnalyzer(X, Y, N, dx)
    
    # Analyze diagonal symmetry
    symmetry_45_135 = analyzer.analyze_diagonal_symmetry(stress_fields['sigma_mag'])
    
    # Extract profiles along main diagonals
    dist_45, prof_45, meta_45 = analyzer.extract_diagonal_profile(stress_fields['sigma_mag'], 45)
    dist_135, prof_135, meta_135 = analyzer.extract_diagonal_profile(stress_fields['sigma_mag'], 135)
    
    # Calculate diagonal stress concentration factors
    max_45 = np.max(prof_45) if len(prof_45) > 0 else 0
    max_135 = np.max(prof_135) if len(prof_135) > 0 else 0
    diagonal_concentration = max(max_45, max_135) / np.mean(stress_fields['sigma_mag']) if np.mean(stress_fields['sigma_mag']) > 0 else 0
    
    # Add enhanced metrics to stress fields
    stress_fields['diagonal_metrics'] = {
        'symmetry_45_135': symmetry_45_135,
        'profile_45': {
            'distances': dist_45,
            'profile': prof_45,
            'metadata': meta_45
        },
        'profile_135': {
            'distances': dist_135,
            'profile': prof_135,
            'metadata': meta_135
        },
        'diagonal_concentration_factor': diagonal_concentration,
        'max_diagonal_stress': max(max_45, max_135),
        'diagonal_asymmetry_ratio': abs(max_45 - max_135) / max(max_45, max_135) if max(max_45, max_135) > 0 else 0
    }
    
    return stress_fields

# =============================================
# ENHANCED SIDEBAR WITH DIAGONAL CONTROLS
# =============================================
st.sidebar.header("⚙️ Enhanced Platform Configuration")

# Operation mode with enhanced options
operation_mode = st.sidebar.radio(
    "Select Operation Mode",
    ["🏃 Run Diagonal Analysis", "🔍 Compare Multi-Orientations", "📊 Diagonal Dashboard", "💾 Enhanced Export"],
    index=0
)

# Initialize diagonal analyzer
diagonal_analyzer = DiagonalProfileAnalyzer(X, Y, N, dx)
diagonal_visualizer = DiagonalVisualizer(diagonal_analyzer)

if "🏃 Run Diagonal Analysis" in operation_mode:
    st.sidebar.header("🎯 Diagonal Analysis Parameters")
    
    # Simulation name
    sim_name = st.sidebar.text_input("Simulation Name", "Diagonal Analysis")
    
    # Defect type with diagonal-specific defaults
    defect_type = st.sidebar.selectbox(
        "Defect Type",
        ["ISF", "ESF", "Twin", "Mixed"],
        help="Select defect type with diagonal orientation effects"
    )
    
    # Enhanced shape selection with diagonal options
    shape = st.sidebar.selectbox(
        "Initial Seed Shape",
        ["Diagonal Fault", "Rotated Rectangle", "Rotated Ellipse", 
         "Square", "Horizontal Fault", "Vertical Fault", "Ellipse", "Circle"],
        help="Shapes optimized for diagonal analysis"
    )
    
    # Diagonal orientation control
    st.sidebar.subheader("📐 Diagonal Orientation")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        orientation_type = st.selectbox(
            "Orientation Type",
            ["Standard Diagonals (45°/135°)", "Arbitrary Angle", "Crystal Directions"],
            index=0
        )
    
    with col2:
        if orientation_type == "Arbitrary Angle":
            custom_diagonal = st.slider("Custom Diagonal Angle (°)", 0, 180, 45, 5)
        elif orientation_type == "Crystal Directions":
            crystal_dir = st.selectbox(
                "Crystal Direction",
                ["[110]", "[1-10]", "[112]", "[1-12]", "Custom"]
            )
            if crystal_dir == "Custom":
                custom_diagonal = st.slider("Custom Direction Angle (°)", 0, 180, 45, 5)
            else:
                # Map crystal directions to angles
                crystal_angles = {
                    "[110]": 45,
                    "[1-10]": 135,
                    "[112]": 30,
                    "[1-12]": 150
                }
                custom_diagonal = crystal_angles.get(crystal_dir, 45)
        else:
            custom_diagonal = 45  # Default for standard diagonals
    
    # Physical parameters enhanced for diagonal analysis
    st.sidebar.subheader("⚛️ Physical Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        eps0 = st.slider(
            "Eigenstrain ε*",
            0.3, 3.0,
            value=0.707 if defect_type == "ISF" else 1.414 if defect_type == "ESF" else 2.121,
            step=0.01,
            help="Magnitude of transformation eigenstrain affecting diagonal stress"
        )
    
    with col2:
        kappa = st.slider(
            "Interface Energy κ",
            0.1, 2.0,
            value=0.6 if defect_type == "ISF" else 0.7 if defect_type == "ESF" else 0.3,
            step=0.05,
            help="Gradient energy coefficient influencing diagonal interface width"
        )
    
    # Evolution parameters
    st.sidebar.subheader("🔄 Evolution Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        steps = st.slider("Total Steps", 50, 500, 100, 10)
    with col2:
        save_every = st.slider("Save Every", 5, 50, 20, 5)
    
    # Enhanced visualization settings for diagonals
    st.sidebar.subheader("🎨 Diagonal Visualization")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        eta_cmap = st.selectbox("η Colormap", cmap_list, 
                               index=cmap_list.index('diagonal_sequential'))
    with col2:
        stress_cmap = st.selectbox("Stress Colormap", cmap_list, 
                                  index=cmap_list.index('stress_diagonal'))
    
    # Diagonal analysis options
    with st.sidebar.expander("🔬 Advanced Diagonal Analysis", expanded=True):
        analyze_symmetry = st.checkbox("Analyze Diagonal Symmetry", True)
        extract_multiple_diagonals = st.checkbox("Extract Multiple Diagonals", True)
        if extract_multiple_diagonals:
            diagonal_angles = st.multiselect(
                "Select Diagonal Angles",
                [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
                default=[0, 45, 90, 135]
            )
        
        calculate_gradients = st.checkbox("Calculate Stress Gradients", True)
        analyze_anisotropy = st.checkbox("Analyze Stress Anisotropy", True)
    
    # Run button
    if st.sidebar.button("🚀 Run Enhanced Analysis", type="primary", use_container_width=True):
        # Prepare parameters
        params = {
            'name': sim_name,
            'defect_type': defect_type,
            'shape': shape,
            'eps0': eps0,
            'kappa': kappa,
            'orientation_type': orientation_type,
            'diagonal_angle': custom_diagonal,
            'steps': steps,
            'save_every': save_every,
            'eta_cmap': eta_cmap,
            'stress_cmap': stress_cmap,
            'analysis_options': {
                'analyze_symmetry': analyze_symmetry,
                'diagonal_angles': diagonal_angles if extract_multiple_diagonals else [45, 135],
                'calculate_gradients': calculate_gradients,
                'analyze_anisotropy': analyze_anisotropy
            }
        }
        
        # Create progress container
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_text = st.empty()
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Running diagonal analysis... {progress*100:.1f}%")
        
        # Run simulation
        with st.spinner("Starting enhanced diagonal analysis..."):
            # Create initial condition with diagonal orientation
            eta = create_initial_eta_diagonal(shape, defect_type, custom_diagonal)
            
            # Run evolution
            history = []
            start_time = time.time()
            
            for step in range(steps + 1):
                if step > 0:
                    eta = evolve_phase_field(eta, kappa)
                
                if step % save_every == 0 or step == steps:
                    # Compute enhanced stress fields
                    stress_fields = compute_enhanced_stress_fields(eta, eps0, np.deg2rad(custom_diagonal))
                    history.append((eta.copy(), stress_fields))
                    
                    # Update progress
                    progress = (step + 1) / (steps + 1)
                    update_progress(progress)
        
        run_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            'run_time': run_time,
            'frames': len(history),
            'grid_size': N,
            'dx': dx,
            'diagonal_angle': custom_diagonal,
            'analysis_options': params['analysis_options'],
            'created': datetime.now().isoformat()
        }
        
        # Store in session state
        st.session_state.diagonal_analysis = {
            'params': params,
            'history': history,
            'metadata': metadata,
            'analyzer': diagonal_analyzer,
            'visualizer': diagonal_visualizer
        }
        
        st.session_state.show_diagonal_results = True
        
        # Update status
        progress_container.empty()
        status_text.success(f"""
        ✅ Enhanced Analysis Complete!
        - **Analysis Type**: Diagonal Orientation ({custom_diagonal}°)
        - **Frames**: {len(history)}
        - **Time**: {run_time:.1f} seconds
        - **Grid Resolution**: {N} × {N}
        """)

# =============================================
# ENHANCED MAIN CONTENT AREA
# =============================================
if "🏃 Run Diagonal Analysis" in operation_mode:
    st.header("📐 Diagonal & Multi-Orientation Stress Analysis")
    
    if 'diagonal_analysis' in st.session_state and st.session_state.get('show_diagonal_results', False):
        analysis_data = st.session_state.diagonal_analysis
        params = analysis_data['params']
        history = analysis_data['history']
        
        # Display enhanced results with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Overview", "📊 Diagonal Profiles", "🌀 Symmetry Analysis", "📈 Statistical Insights", "💾 Export"])
        
        with tab1:
            st.subheader("Enhanced Diagonal Analysis Overview")
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Defect Type", params['defect_type'])
            with col2:
                st.metric("Diagonal Angle", f"{params['diagonal_angle']}°")
            with col3:
                st.metric("ε*", f"{params['eps0']:.3f}")
            with col4:
                st.metric("κ", f"{params['kappa']:.2f}")
            
            # Show final state with diagonal lines
            if history:
                final_eta, final_stress = history[-1]
                
                # Create enhanced visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
                
                # Panel 1: Defect field with diagonals
                ax1.set_aspect('equal')
                im1 = ax1.imshow(final_eta, extent=extent,
                                cmap=plt.cm.get_cmap(COLORMAPS.get(params['eta_cmap'], 'viridis')),
                                origin='lower')
                ax1.set_title(f"Defect Field: {params['defect_type']}", fontsize=14, fontweight='bold')
                ax1.set_xlabel("x (nm)")
                ax1.set_ylabel("y (nm)")
                
                # Add diagonal lines
                angles = params['analysis_options']['diagonal_angles']
                colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
                
                for angle, color in zip(angles, colors):
                    angle_rad = np.deg2rad(angle)
                    length = 0.6 * max(extent[1] - extent[0], extent[3] - extent[2])
                    x_start = -length/2 * np.cos(angle_rad)
                    y_start = -length/2 * np.sin(angle_rad)
                    x_end = length/2 * np.cos(angle_rad)
                    y_end = length/2 * np.sin(angle_rad)
                    
                    ax1.plot([x_start, x_end], [y_start, y_end],
                            color=color, linewidth=2, alpha=0.8,
                            label=f'{angle}°')
                
                ax1.legend(fontsize=10, loc='upper right')
                plt.colorbar(im1, ax=ax1, shrink=0.8, label='η')
                
                # Panel 2: Stress magnitude
                ax2.set_aspect('equal')
                im2 = ax2.imshow(final_stress['sigma_mag'], extent=extent,
                                cmap=plt.cm.get_cmap(COLORMAPS.get(params['stress_cmap'], 'hot')),
                                origin='lower')
                ax2.set_title("Stress Magnitude |σ|", fontsize=14, fontweight='bold')
                ax2.set_xlabel("x (nm)")
                ax2.set_ylabel("y (nm)")
                plt.colorbar(im2, ax=ax2, shrink=0.8, label='|σ| (GPa)')
                
                # Panel 3: Hydrostatic stress
                ax3.set_aspect('equal')
                im3 = ax3.imshow(final_stress['sigma_hydro'], extent=extent,
                                cmap=plt.cm.get_cmap(COLORMAPS.get('coolwarm', 'coolwarm')),
                                origin='lower')
                ax3.set_title("Hydrostatic Stress σ_h", fontsize=14, fontweight='bold')
                ax3.set_xlabel("x (nm)")
                ax3.set_ylabel("y (nm)")
                plt.colorbar(im3, ax=ax3, shrink=0.8, label='σ_h (GPa)')
                
                # Panel 4: von Mises stress
                ax4.set_aspect('equal')
                im4 = ax4.imshow(final_stress['von_mises'], extent=extent,
                                cmap=plt.cm.get_cmap(COLORMAPS.get('plasma', 'plasma')),
                                origin='lower')
                ax4.set_title("von Mises Stress σ_vM", fontsize=14, fontweight='bold')
                ax4.set_xlabel("x (nm)")
                ax4.set_ylabel("y (nm)")
                plt.colorbar(im4, ax=ax4, shrink=0.8, label='σ_vM (GPa)')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            st.subheader("📊 Comprehensive Diagonal Profile Analysis")
            
            if history:
                final_eta, final_stress = history[-1]
                
                # Extract diagonal profiles
                angles = params['analysis_options']['diagonal_angles']
                
                # Create profile comparison
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                
                stress_components = {
                    'Stress Magnitude': final_stress['sigma_mag'],
                    'Hydrostatic': final_stress['sigma_hydro'],
                    'von Mises': final_stress['von_mises'],
                    'Shear XY': final_stress['sxy']
                }
                
                for idx, (title, data) in enumerate(stress_components.items()):
                    if idx < len(axes):
                        ax = axes[idx]
                        
                        # Plot profiles for each diagonal angle
                        for angle in angles:
                            distances, profile, metadata = diagonal_analyzer.extract_diagonal_profile(
                                data, angle, length_ratio=0.8
                            )
                            
                            ax.plot(distances, profile, linewidth=2, alpha=0.8,
                                   label=f'{angle}° (max: {metadata["max_value"]:.2f} GPa)')
                        
                        ax.set_xlabel("Distance along Diagonal (nm)", fontsize=11)
                        ax.set_ylabel(f"{title} (GPa)", fontsize=11)
                        ax.set_title(f"{title} along Diagonals", fontsize=13, fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                        
                        if idx == 0:
                            ax.legend(fontsize=10, loc='upper right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show profile statistics
                with st.expander("📈 Profile Statistics", expanded=True):
                    stats_data = []
                    for angle in angles:
                        distances, profile, metadata = diagonal_analyzer.extract_diagonal_profile(
                            final_stress['sigma_mag'], angle
                        )
                        
                        stats_data.append({
                            'Angle': f'{angle}°',
                            'Max (GPa)': metadata['max_value'],
                            'Mean (GPa)': metadata['mean_value'],
                            'FWHM (nm)': metadata['fwhm_nm'],
                            'Skewness': metadata['skewness'],
                            'Gradient Max': metadata['gradient_max'],
                            'Symmetry Index': metadata['symmetry_index']
                        })
                    
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(
                        df_stats.style.format({
                            'Max (GPa)': '{:.3f}',
                            'Mean (GPa)': '{:.3f}',
                            'FWHM (nm)': '{:.2f}',
                            'Skewness': '{:.3f}',
                            'Gradient Max': '{:.3f}',
                            'Symmetry Index': '{:.3f}'
                        }),
                        use_container_width=True
                    )
        
        with tab3:
            st.subheader("🌀 Diagonal Symmetry & Anisotropy Analysis")
            
            if history:
                final_eta, final_stress = history[-1]
                
                # Create symmetry analysis visualization
                fig = plt.figure(figsize=(16, 12))
                
                # Panel 1: 45° vs 135° comparison
                ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
                ax2 = plt.subplot2grid((3, 3), (0, 2))
                ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
                ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
                
                # Extract diagonal profiles
                dist_45, prof_45, meta_45 = diagonal_analyzer.extract_diagonal_profile(
                    final_stress['sigma_mag'], 45
                )
                dist_135, prof_135, meta_135 = diagonal_analyzer.extract_diagonal_profile(
                    final_stress['sigma_mag'], 135
                )
                
                # Plot comparison
                ax1.plot(dist_45, prof_45, 'b-', linewidth=2, alpha=0.8, label='45°')
                ax1.plot(dist_135, prof_135, 'r--', linewidth=2, alpha=0.8, label='135°')
                ax1.set_xlabel("Distance (nm)")
                ax1.set_ylabel("Stress (GPa)")
                ax1.set_title("45° vs 135° Diagonal Profiles", fontsize=13, fontweight='bold')
                ax1.legend(fontsize=11)
                ax1.grid(True, alpha=0.3)
                
                # Plot difference
                ax2.plot(dist_45, prof_45 - prof_135, 'g-', linewidth=2, alpha=0.8)
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_xlabel("Distance (nm)")
                ax2.set_ylabel("Δ Stress (GPa)")
                ax2.set_title("Profile Difference", fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Polar plot of stress maxima
                angles = np.linspace(0, 360, 24, endpoint=False)
                maxima = []
                
                for angle in angles:
                    distances, profile, _ = diagonal_analyzer.extract_diagonal_profile(
                        final_stress['sigma_mag'], angle, length_ratio=0.6
                    )
                    maxima.append(np.nanmax(profile) if len(profile) > 0 else 0)
                
                ax3 = plt.subplot(233, projection='polar')
                theta = np.deg2rad(angles)
                ax3.plot(theta, maxima, 'b-', linewidth=2, alpha=0.8)
                ax3.fill(theta, maxima, 'b', alpha=0.2)
                ax3.set_title("Polar Distribution of Stress Maxima", fontsize=13, fontweight='bold', pad=20)
                ax3.grid(True, alpha=0.5)
                
                # Anisotropy analysis
                ax4.bar(range(len(angles)), maxima, alpha=0.7, color='steelblue')
                ax4.set_xlabel("Angle (°)")
                ax4.set_ylabel("Maximum Stress (GPa)")
                ax4.set_title("Stress Anisotropy Analysis", fontsize=13, fontweight='bold')
                ax4.set_xticks(range(0, len(angles), 4))
                ax4.set_xticklabels([f'{int(angle)}°' for angle in angles[::4]])
                ax4.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display symmetry metrics
                symmetry_metrics = diagonal_analyzer.analyze_diagonal_symmetry(final_stress['sigma_mag'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Correlation (45° vs 135°)", f"{symmetry_metrics['correlation_coefficient']:.3f}")
                with col2:
                    st.metric("RMS Difference", f"{symmetry_metrics['rms_difference']:.3f} GPa")
                with col3:
                    st.metric("Max Difference", f"{symmetry_metrics['max_difference']:.3f} GPa")
                with col4:
                    symmetry_status = "Symmetric" if symmetry_metrics['is_symmetric'] else "Asymmetric"
                    st.metric("Symmetry Status", symmetry_status)
        
        with tab4:
            st.subheader("📈 Advanced Statistical Insights")
            
            if history:
                final_eta, final_stress = history[-1]
                
                # Create comprehensive statistical analysis
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Panel 1: Stress distribution
                stress_data = final_stress['sigma_mag'].flatten()
                stress_data = stress_data[np.isfinite(stress_data)]
                
                axes[0, 0].hist(stress_data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
                axes[0, 0].set_xlabel("Stress (GPa)")
                axes[0, 0].set_ylabel("Probability Density")
                axes[0, 0].set_title("Stress Distribution", fontsize=13, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add Gaussian fit
                mu, sigma = np.mean(stress_data), np.std(stress_data)
                x = np.linspace(min(stress_data), max(stress_data), 100)
                y = stats.norm.pdf(x, mu, sigma)
                axes[0, 0].plot(x, y, 'r-', linewidth=2, alpha=0.8, label=f'Gaussian fit\nμ={mu:.2f}, σ={sigma:.2f}')
                axes[0, 0].legend(fontsize=10)
                
                # Panel 2: QQ plot
                stats.probplot(stress_data, dist="norm", plot=axes[0, 1])
                axes[0, 1].set_title("Q-Q Plot", fontsize=13, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Panel 3: Cumulative distribution
                sorted_stress = np.sort(stress_data)
                y_vals = np.arange(1, len(sorted_stress) + 1) / len(sorted_stress)
                axes[0, 2].plot(sorted_stress, y_vals, 'b-', linewidth=2, alpha=0.8)
                axes[0, 2].set_xlabel("Stress (GPa)")
                axes[0, 2].set_ylabel("Cumulative Probability")
                axes[0, 2].set_title("Cumulative Distribution", fontsize=13, fontweight='bold')
                axes[0, 2].grid(True, alpha=0.3)
                
                # Panel 4: Spatial autocorrelation
                # Simplified autocorrelation along diagonals
                angles = [0, 45, 90, 135]
                autocorr_data = []
                
                for angle in angles:
                    distances, profile, _ = diagonal_analyzer.extract_diagonal_profile(
                        final_stress['sigma_mag'], angle
                    )
                    if len(profile) > 10:
                        autocorr = np.correlate(profile, profile, mode='full')[-len(profile):]
                        autocorr = autocorr / np.max(autocorr)
                        axes[1, 0].plot(distances[:len(autocorr)], autocorr, linewidth=2, alpha=0.8, label=f'{angle}°')
                
                axes[1, 0].set_xlabel("Lag Distance (nm)")
                axes[1, 0].set_ylabel("Autocorrelation")
                axes[1, 0].set_title("Spatial Autocorrelation", fontsize=13, fontweight='bold')
                axes[1, 0].legend(fontsize=10)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Panel 5: Stress gradient distribution
                grad_x, grad_y = np.gradient(final_stress['sigma_mag'])
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                grad_flat = grad_magnitude.flatten()
                grad_flat = grad_flat[np.isfinite(grad_flat)]
                
                axes[1, 1].hist(grad_flat, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
                axes[1, 1].set_xlabel("Stress Gradient (GPa/nm)")
                axes[1, 1].set_ylabel("Probability Density")
                axes[1, 1].set_title("Stress Gradient Distribution", fontsize=13, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Panel 6: Defect-stress correlation
                eta_flat = final_eta.flatten()
                stress_flat = final_stress['sigma_mag'].flatten()
                
                # Sample for clarity
                sample_size = min(5000, len(eta_flat))
                indices = np.random.choice(len(eta_flat), sample_size, replace=False)
                
                scatter = axes[1, 2].scatter(eta_flat[indices], stress_flat[indices], 
                                           c=stress_flat[indices], cmap='hot', alpha=0.6, s=10)
                axes[1, 2].set_xlabel("Defect Parameter η")
                axes[1, 2].set_ylabel("Stress (GPa)")
                axes[1, 2].set_title("Defect-Stress Correlation", fontsize=13, fontweight='bold')
                axes[1, 2].grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=axes[1, 2], shrink=0.8, label='Stress (GPa)')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display comprehensive statistics
                with st.expander("📊 Comprehensive Statistics", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Stress Statistics:**")
                        st.json({
                            'Mean Stress (GPa)': float(np.mean(stress_data)),
                            'Median Stress (GPa)': float(np.median(stress_data)),
                            'Std Deviation (GPa)': float(np.std(stress_data)),
                            'Skewness': float(stats.skew(stress_data)),
                            'Kurtosis': float(stats.kurtosis(stress_data)),
                            'Minimum (GPa)': float(np.min(stress_data)),
                            'Maximum (GPa)': float(np.max(stress_data)),
                            '95th Percentile (GPa)': float(np.percentile(stress_data, 95))
                        })
                    
                    with col2:
                        st.markdown("**Spatial Statistics:**")
                        st.json({
                            'Mean Gradient (GPa/nm)': float(np.mean(grad_flat)),
                            'Max Gradient (GPa/nm)': float(np.max(grad_flat)),
                            'Correlation (η vs σ)': float(np.corrcoef(eta_flat, stress_flat)[0, 1]),
                            'Defect Area (nm²)': float(np.sum(final_eta > 0.5) * dx**2),
                            'Stress Concentration Factor': float(np.max(stress_data) / np.mean(stress_data)),
                            'Anisotropy Ratio': float(final_stress['diagonal_metrics']['diagonal_asymmetry_ratio'])
                        })
        
        with tab5:
            st.subheader("💾 Enhanced Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Download Complete Analysis", use_container_width=True):
                    # Create comprehensive export
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, 'w') as zf:
                        # Add parameters
                        zf.writestr('analysis_parameters.json', json.dumps(params, indent=2))
                        
                        # Add diagonal profiles
                        if history:
                            final_eta, final_stress = history[-1]
                            profiles_data = {}
                            
                            for angle in params['analysis_options']['diagonal_angles']:
                                distances, profile, metadata = diagonal_analyzer.extract_diagonal_profile(
                                    final_stress['sigma_mag'], angle
                                )
                                profiles_data[f'diagonal_{angle}'] = {
                                    'distances': distances.tolist(),
                                    'profile': profile.tolist(),
                                    'metadata': metadata
                                }
                            
                            zf.writestr('diagonal_profiles.json', json.dumps(profiles_data, indent=2))
                        
                        # Add symmetry analysis
                        if history:
                            final_eta, final_stress = history[-1]
                            symmetry_metrics = diagonal_analyzer.analyze_diagonal_symmetry(final_stress['sigma_mag'])
                            zf.writestr('symmetry_analysis.json', json.dumps(symmetry_metrics, indent=2))
                        
                        # Add summary report
                        summary = f"""
                        DIAGONAL ANALYSIS EXPORT SUMMARY
                        =================================
                        Generated: {datetime.now().isoformat()}
                        Simulation: {params['name']}
                        Defect Type: {params['defect_type']}
                        Diagonal Angle: {params['diagonal_angle']}°
                        Grid Size: {N} × {N}
                        Grid Spacing: {dx} nm
                        
                        ANALYSIS PARAMETERS:
                        --------------------
                        ε* = {params['eps0']}
                        κ = {params['kappa']}
                        Steps = {params['steps']}
                        Frames = {len(history)}
                        
                        DIAGONAL ANGLES ANALYZED:
                        -------------------------
                        {', '.join([str(a) for a in params['analysis_options']['diagonal_angles']])}
                        
                        EXPORT CONTENTS:
                        ----------------
                        1. analysis_parameters.json - Complete simulation parameters
                        2. diagonal_profiles.json - Extracted diagonal profiles
                        3. symmetry_analysis.json - Diagonal symmetry metrics
                        4. This summary file
                        
                        PLATFORM INFORMATION:
                        ---------------------
                        Ag NP Multi-Orientation Defect Analyzer Pro
                        Version: 3.0.0
                        Enhanced with diagonal analysis capabilities
                        """
                        zf.writestr('EXPORT_SUMMARY.txt', summary)
                    
                    buffer.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download ZIP Package",
                        data=buffer,
                        file_name=f"diagonal_analysis_{params['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
            
            with col2:
                if st.button("🖼️ Export Publication Figures", use_container_width=True):
                    st.info("Exporting high-resolution publication-ready figures...")
                    # In a full implementation, this would generate and save PDF/PNG figures
                    
                    # Create a sample figure for demonstration
                    if history:
                        final_eta, final_stress = history[-1]
                        
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Defect field
                        im1 = axes[0].imshow(final_eta, extent=extent, cmap='viridis', origin='lower', aspect='equal')
                        axes[0].set_title(f"Defect Field: {params['defect_type']}")
                        axes[0].set_xlabel("x (nm)")
                        axes[0].set_ylabel("y (nm)")
                        plt.colorbar(im1, ax=axes[0], shrink=0.8)
                        
                        # Stress field
                        im2 = axes[1].imshow(final_stress['sigma_mag'], extent=extent, cmap='hot', origin='lower', aspect='equal')
                        axes[1].set_title("Stress Magnitude")
                        axes[1].set_xlabel("x (nm)")
                        axes[1].set_ylabel("y (nm)")
                        plt.colorbar(im2, ax=axes[1], shrink=0.8)
                        
                        plt.tight_layout()
                        
                        # Save to buffer
                        buf = BytesIO()
                        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📸 Download Figure",
                            data=buf,
                            file_name=f"diagonal_analysis_figure_{params['name'].replace(' ', '_')}.png",
                            mime="image/png"
                        )
            
            with col3:
                if st.button("📋 Copy Analysis Summary", use_container_width=True):
                    summary = f"""
                    Diagonal Analysis Summary
                    =========================
                    Simulation: {params['name']}
                    Defect Type: {params['defect_type']}
                    Orientation: {params['diagonal_angle']}°
                    ε*: {params['eps0']:.3f}
                    κ: {params['kappa']:.2f}
                    
                    Results:
                    - Frames: {len(history)}
                    - Grid: {N}×{N} ({dx} nm spacing)
                    - Domain: {extent[0]:.1f} to {extent[1]:.1f} nm
                    
                    Key Insights:
                    - Diagonal stress concentration analyzed
                    - Multiple orientation profiles extracted
                    - Symmetry metrics calculated
                    - Publication-ready visualizations generated
                    """
                    
                    st.code(summary)
    
    else:
        # Show welcome/instructions with enhanced layout
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Welcome to Enhanced Diagonal Analysis Platform</h3>
            <p>This platform extends standard defect analysis with comprehensive diagonal and multi-orientation capabilities.</p>
            
            <h4>🔬 New Features:</h4>
            <ul>
                <li><strong>Diagonal Fault Analysis</strong>: Analyze defects along 45°, 135°, and arbitrary diagonals</li>
                <li><strong>Multi-Orientation Profiling</strong>: Extract stress profiles along any crystal direction</li>
                <li><strong>Symmetry Analysis</strong>: Quantify stress symmetry between different orientations</li>
                <li><strong>Enhanced Visualization</strong>: Publication-ready figures with diagonal overlays</li>
                <li><strong>Statistical Validation</strong>: Comprehensive statistical analysis of diagonal effects</li>
            </ul>
            
            <h4>📐 Key Capabilities:</h4>
            <ul>
                <li>Crystal-direction-aware defect initialization</li>
                <li>Real-time diagonal profile extraction</li>
                <li>Stress anisotropy quantification</li>
                <li>Polar distribution analysis</li>
                <li>Enhanced export with diagonal metadata</li>
            </ul>
            
            <p>Configure your diagonal analysis in the sidebar and click "Run Enhanced Analysis" to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example diagonal analysis
        st.subheader("📊 Example: Diagonal Stress Analysis")
        
        # Create example defect with diagonal orientation
        example_eta = create_initial_eta_diagonal("Diagonal Fault", "ISF", 45)
        example_stress = compute_enhanced_stress_fields(example_eta, 0.707, np.deg2rad(45))
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_aspect('equal')
            im = ax.imshow(example_eta, extent=extent, cmap='viridis', origin='lower')
            ax.set_title("Example: 45° Diagonal Fault")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            
            # Add diagonal lines
            angle_rad = np.deg2rad(45)
            length = 0.6 * max(extent[1] - extent[0], extent[3] - extent[2])
            x_start = -length/2 * np.cos(angle_rad)
            y_start = -length/2 * np.sin(angle_rad)
            x_end = length/2 * np.cos(angle_rad)
            y_end = length/2 * np.sin(angle_rad)
            
            ax.plot([x_start, x_end], [y_start, y_end], 'r-', linewidth=3, alpha=0.7, label='45° diagonal')
            ax.legend(loc='upper right')
            
            plt.colorbar(im, ax=ax, shrink=0.8, label='η')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_aspect('equal')
            im = ax.imshow(example_stress['sigma_mag'], extent=extent, cmap='hot', origin='lower')
            ax.set_title("Stress Magnitude along Diagonal")
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            
            # Extract and plot diagonal profile
            analyzer = DiagonalProfileAnalyzer(X, Y, N, dx)
            distances, profile, metadata = analyzer.extract_diagonal_profile(
                example_stress['sigma_mag'], 45
            )
            
            # Create inset for profile
            inset_ax = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
            inset_ax.plot(distances, profile, 'b-', linewidth=2)
            inset_ax.set_title("45° Profile", fontsize=10)
            inset_ax.set_xlabel("Distance (nm)", fontsize=8)
            inset_ax.set_ylabel("Stress (GPa)", fontsize=8)
            inset_ax.grid(True, alpha=0.3)
            inset_ax.tick_params(labelsize=8)
            
            plt.colorbar(im, ax=ax, shrink=0.8, label='|σ| (GPa)')
            st.pyplot(fig)

# =============================================
# FOOTER WITH ENHANCED STYLING
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; margin-top: 2rem;">
    <p style="font-size: 1.2rem; font-weight: 700; color: #2D3748; margin-bottom: 0.5rem;">
        🔬 Ag Nanoparticle Multi-Orientation Defect Analyzer Pro v3.0
    </p>
    <p style="color: #4A5568; margin-bottom: 0.5rem;">
        Advanced diagonal analysis • Multi-orientation profiling • Crystal-aware simulations
    </p>
    <p style="color: #718096; font-size: 0.9rem;">
        © 2024 • Scientific Computing Group • Enhanced with diagonal analysis capabilities
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================
# SESSION STATE MANAGEMENT
# =============================================
if 'show_diagonal_results' not in st.session_state:
    st.session_state.show_diagonal_results = False
if 'diagonal_analysis' not in st.session_state:
    st.session_state.diagonal_analysis = None
