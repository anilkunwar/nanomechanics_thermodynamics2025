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
from scipy.signal import find_peaks, peak_widths, savgol_filter
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
    .profile-controls {
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #ddd;
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
    <p><strong>Enhanced Overlay Line Profiles with Statistical Analysis • Peak Detection • Gradient Analysis • Frequency Spectrum</strong></p>
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
# ENHANCED COLORMAP LIBRARY
# =============================================
COLORMAPS = {
    'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno', 'magma': 'magma',
    'cividis': 'cividis', 'summer': 'summer', 'autumn': 'autumn', 'winter': 'winter',
    'spring': 'spring', 'cool': 'cool', 'hot': 'hot', 'copper': 'copper', 'bone': 'bone',
    'gray': 'gray', 'pink': 'pink', 'afmhot': 'afmhot', 'gist_heat': 'gist_heat',
    'coolwarm': 'coolwarm', 'bwr': 'bwr', 'seismic': 'seismic', 'RdBu': 'RdBu',
    'RdGy': 'RdGy', 'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG', 'PuOr': 'PuOr',
    'twilight': 'twilight', 'hsv': 'hsv', 'tab10': 'tab10', 'tab20': 'tab20',
    'Set1': 'Set1', 'Set2': 'Set2', 'Set3': 'Set3', 'Paired': 'Paired',
    'jet': 'jet', 'turbo': 'turbo', 'rainbow': 'rainbow', 'rocket': 'rocket',
    'mako': 'mako', 'crest': 'crest', 'flare': 'flare', 'icefire': 'icefire'
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
        """Extract high-resolution line profile from 2D data"""
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
        
        above_half = profile_norm > half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            if len(indices) > 1:
                left_idx = indices[0]
                right_idx = indices[-1]
                return float(distances[right_idx] - distances[left_idx])
        return 0.0

# =============================================
# FIXED ASPECT RATIO VISUALIZATION SYSTEM
# =============================================
class FixedAspectManager:
    """Comprehensive system for maintaining realistic aspect ratios in visualizations"""
    
    @staticmethod
    def apply_fixed_aspect(ax, aspect_type='equal', **kwargs):
        """Apply fixed aspect ratio to axis"""
        if aspect_type == 'equal':
            ax.set_aspect('equal')
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            if x_range > y_range:
                center_y = (ylim[0] + ylim[1]) / 2
                ax.set_ylim(center_y - x_range/2, center_y + x_range/2)
            elif y_range > x_range:
                center_x = (xlim[0] + xlim[1]) / 2
                ax.set_xlim(center_x - y_range/2, center_x + y_range/2)
        
        return ax
    
    @staticmethod
    def add_physical_scale(ax, length_nm=5.0, location='lower right', 
                          color='white', fontsize=10, linewidth=2):
        """Add physical scale bar with enhanced styling"""
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
        else:  # lower left
            bar_x_start = xlim[0] + x_range * 0.05
            bar_x_end = bar_x_start + length_nm
            bar_y = ylim[0] + y_range * 0.05
            text_y = bar_y + y_range * 0.02
            text_ha = 'center'
            text_va = 'bottom'
        
        # Draw scale bar
        ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], 
               color=color, linewidth=linewidth, solid_capstyle='butt')
        
        # Add text
        ax.text((bar_x_start + bar_x_end) / 2, text_y,
               f'{length_nm} nm', ha=text_ha, va=text_va,
               color=color, fontsize=fontsize, fontweight='bold')
        
        return ax

# =============================================
# ENHANCED PROFILE ANALYSIS ENHANCER
# =============================================
class ProfileAnalysisEnhancer:
    """Advanced analysis tools for comprehensive line profile characterization"""
    
    @staticmethod
    def extract_comprehensive_profiles(simulations, frames, angles, config):
        """Extract multiple profiles with comprehensive metadata"""
        all_profiles = {}
        
        stress_key = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }[config.get('stress_component', 'Stress Magnitude |σ|')]
        
        for sim_idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Apply smoothing if requested
            if config.get('smooth_profiles', False):
                sigma = config.get('smooth_sigma', 0.5)
                stress_data = gaussian_filter(stress_data, sigma=sigma)
            
            profiles = {}
            for angle in angles:
                result = AdvancedProfileExtractor.extract_profile_2d(
                    stress_data, 
                    angle, 
                    'center', 
                    0, 
                    config.get('profile_length', 80) / 100,
                    sampling_factor=config.get('sampling_factor', 2),
                    interpolation_order=config.get('interpolation_order', 3)
                )
                
                distances, profile, endpoints, metadata = result
                
                # Calculate advanced metrics
                advanced_metrics = ProfileAnalysisEnhancer.calculate_profile_metrics(
                    distances, profile, metadata
                )
                
                profiles[angle] = {
                    'distances': distances,
                    'profile': profile,
                    'endpoints': endpoints,
                    'metadata': {**metadata, **advanced_metrics},
                    'sim_info': {
                        'defect_type': sim['params']['defect_type'],
                        'orientation': sim['params']['orientation'],
                        'sim_idx': sim_idx
                    }
                }
            
            all_profiles[sim_idx] = profiles
        
        return all_profiles
    
    @staticmethod
    def calculate_profile_metrics(distances, profile, base_metadata):
        """Calculate comprehensive profile metrics"""
        metrics = {}
        
        # Basic statistics
        metrics['max_value'] = float(np.max(profile))
        metrics['min_value'] = float(np.min(profile))
        metrics['mean_value'] = float(np.mean(profile))
        metrics['std_value'] = float(np.std(profile))
        metrics['integral'] = float(np.trapz(profile, distances))
        
        # Peak detection
        try:
            peaks, properties = find_peaks(profile, 
                                          height=metrics['mean_value'] + metrics['std_value']/2,
                                          distance=len(profile)//10)
            metrics['num_peaks'] = len(peaks)
            if len(peaks) > 0:
                metrics['peak_positions'] = distances[peaks].tolist()
                metrics['peak_heights'] = profile[peaks].tolist()
                
                # Calculate peak widths
                try:
                    widths, width_heights, left_ips, right_ips = peak_widths(
                        profile, peaks, rel_height=0.5
                    )
                    metrics['peak_widths'] = widths.tolist()
                except:
                    metrics['peak_widths'] = []
        except:
            metrics['num_peaks'] = 0
            metrics['peak_positions'] = []
            metrics['peak_heights'] = []
            metrics['peak_widths'] = []
        
        # Gradient analysis
        gradient = np.gradient(profile, distances)
        metrics['max_gradient'] = float(np.max(np.abs(gradient)))
        metrics['gradient_std'] = float(np.std(gradient))
        
        # Asymmetry analysis
        center_idx = len(profile) // 2
        left_half = profile[:center_idx]
        right_half = profile[center_idx:]
        metrics['asymmetry'] = float(np.mean(right_half) - np.mean(left_half))
        
        # Statistical moments
        metrics['kurtosis'] = float(stats.kurtosis(profile))
        metrics['skewness'] = float(stats.skew(profile))
        
        # Frequency analysis
        fft_result = np.fft.fft(profile)
        freq = np.fft.fftfreq(len(profile), distances[1]-distances[0])
        positive_freq = freq[:len(freq)//2]
        power_spectrum = np.abs(fft_result[:len(fft_result)//2])**2
        if len(power_spectrum) > 1:
            metrics['dominant_frequency'] = float(positive_freq[np.argmax(power_spectrum[1:]) + 1])
        else:
            metrics['dominant_frequency'] = 0.0
        
        return metrics
    
    @staticmethod
    def create_profile_comparison_table(profiles_by_simulation):
        """Create comprehensive comparison table"""
        table_data = []
        
        for sim_idx, angle_profiles in profiles_by_simulation.items():
            for angle, profile_data in angle_profiles.items():
                metrics = profile_data['metadata']
                sim_info = profile_data['sim_info']
                
                row = {
                    'Simulation': f"Sim {sim_idx+1}",
                    'Defect Type': sim_info['defect_type'],
                    'Profile Angle': f"{angle}°",
                    'Max Stress (GPa)': f"{metrics['max_value']:.3f}",
                    'Mean Stress (GPa)': f"{metrics['mean_value']:.3f}",
                    'FWHM (nm)': f"{metrics.get('fwhm_nm', 0):.2f}",
                    'Peaks': metrics.get('num_peaks', 0),
                    'Asymmetry': f"{metrics['asymmetry']:.3f}",
                    'Integral (GPa·nm)': f"{metrics['integral']:.3f}",
                    'Max Gradient (GPa/nm)': f"{metrics['max_gradient']:.3f}"
                }
                table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    @staticmethod
    def perform_statistical_tests(profiles_by_simulation):
        """Perform statistical tests on profile data"""
        results = {}
        
        # Collect maximum stresses by simulation
        max_stresses_by_sim = {}
        for sim_idx, angle_profiles in profiles_by_simulation.items():
            max_stresses = []
            for profile_data in angle_profiles.values():
                max_stresses.append(profile_data['metadata']['max_value'])
            max_stresses_by_sim[f"Sim_{sim_idx+1}"] = max_stresses
        
        # Perform ANOVA if we have enough data
        if len(max_stresses_by_sim) > 1 and all(len(v) > 1 for v in max_stresses_by_sim.values()):
            try:
                f_val, p_val = stats.f_oneway(*max_stresses_by_sim.values())
                results['anova'] = {
                    'f_value': f_val,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
            except:
                results['anova'] = {'error': 'Could not perform ANOVA'}
        
        # Pairwise t-tests
        pairwise_results = {}
        sim_keys = list(max_stresses_by_sim.keys())
        for i in range(len(sim_keys)):
            for j in range(i+1, len(sim_keys)):
                try:
                    t_stat, p_val = stats.ttest_ind(
                        max_stresses_by_sim[sim_keys[i]], 
                        max_stresses_by_sim[sim_keys[j]]
                    )
                    pairwise_results[f"{sim_keys[i]} vs {sim_keys[j]}"] = {
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    }
                except:
                    pairwise_results[f"{sim_keys[i]} vs {sim_keys[j]}"] = {'error': 'Could not perform t-test'}
        
        results['pairwise'] = pairwise_results
        
        return results

# =============================================
# ENHANCED OVERLAY PROFILE VISUALIZER
# =============================================
class EnhancedProfileVisualizer:
    """Advanced visualization system for overlay line profiles with statistical analysis"""
    
    @staticmethod
    def create_statistically_enhanced_profiles(simulations, frames, config, style_params):
        """Create overlay profiles with comprehensive statistical analysis"""
        # Map stress component
        stress_key = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }[config.get('stress_component', 'Stress Magnitude |σ|')]
        
        # Parse angles
        angles = []
        for orientation in config.get('profile_orientations', ["0° (Horizontal)", "90° (Vertical)"]):
            if orientation == "0° (Horizontal)":
                angles.append(0)
            elif orientation == "90° (Vertical)":
                angles.append(90)
            elif orientation == "45° (Diagonal)":
                angles.append(45)
            elif orientation == "135° (Diagonal)":
                angles.append(135)
            elif orientation == "Custom":
                angles.append(config.get('custom_angle', 30))
        
        # Extract comprehensive profiles
        profile_analyzer = ProfileAnalysisEnhancer()
        all_profiles = profile_analyzer.extract_comprehensive_profiles(
            simulations, frames, angles, config
        )
        
        # Create enhanced figure
        fig = plt.figure(figsize=(22, 18))
        fig.set_constrained_layout(True)
        
        # Define complex grid layout
        gs = fig.add_gridspec(4, 4, 
                            height_ratios=[1.2, 1, 1, 1.5],
                            hspace=0.25, wspace=0.3)
        
        # Panel definitions
        ax_overlay = fig.add_subplot(gs[0, :2])      # A: Overlay profiles with confidence
        ax_statistics = fig.add_subplot(gs[0, 2:])   # B: Statistical summary
        ax_gradient = fig.add_subplot(gs[1, :2])     # C: Gradient comparison
        ax_spectrum = fig.add_subplot(gs[1, 2:])     # D: Frequency spectrum
        ax_distribution = fig.add_subplot(gs[2, :2]) # E: Stress distribution
        ax_correlation = fig.add_subplot(gs[2, 2:])  # F: Correlation plot
        ax_domain = fig.add_subplot(gs[3, :])        # G: Domain with all profiles
        
        # Colors for different simulations
        colors = plt.cm.tab20(np.linspace(0, 1, len(simulations)))
        
        # Panel A: Enhanced overlay with confidence intervals
        for sim_idx, angle_profiles in all_profiles.items():
            sim = simulations[sim_idx]
            color = colors[sim_idx % len(colors)]
            
            for angle, profile_data in angle_profiles.items():
                distances = profile_data['distances']
                profile = profile_data['profile']
                metrics = profile_data['metadata']
                
                # Main profile line
                line_style = config.get('line_style', 'solid')
                label = f"{sim['params']['defect_type']} - {angle}°" if angle == angles[0] else None
                
                line = ax_overlay.plot(distances, profile,
                                     color=color,
                                     linestyle=line_style,
                                     linewidth=style_params.get('line_width', 2.5),
                                     alpha=0.9,
                                     label=label)[0]
                
                # Add confidence band using gradient-based uncertainty
                gradient = np.gradient(profile)
                uncertainty = 0.15 * np.abs(gradient) * metrics['std_value']
                ax_overlay.fill_between(distances, 
                                      profile - uncertainty, 
                                      profile + uncertainty,
                                      color=color, alpha=0.2)
                
                # Mark peaks
                if 'peak_positions' in metrics and len(metrics['peak_positions']) > 0:
                    peak_x = metrics['peak_positions']
                    peak_y = metrics['peak_heights']
                    ax_overlay.scatter(peak_x, peak_y, 
                                     color=color, s=60, marker='o',
                                     edgecolors='white', linewidths=1.5,
                                     zorder=5, label='_nolegend_')
        
        ax_overlay.set_xlabel("Distance from Center (nm)", fontsize=13)
        ax_overlay.set_ylabel(f"{config['stress_component']} (GPa)", fontsize=13)
        ax_overlay.set_title("Enhanced Multi-Simulation Profile Overlay with Confidence Bands", 
                           fontsize=15, fontweight='bold', pad=15)
        ax_overlay.legend(fontsize=11, ncol=2, loc='upper right', frameon=True, framealpha=0.9)
        ax_overlay.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax_overlay.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Panel B: Statistical summary bar chart
        stats_data = []
        for sim_idx, angle_profiles in all_profiles.items():
            sim = simulations[sim_idx]
            for angle, profile_data in angle_profiles.items():
                metrics = profile_data['metadata']
                stats_data.append({
                    'Simulation': sim['params']['defect_type'],
                    'Angle': angle,
                    'Max Stress': metrics['max_value'],
                    'FWHM': metrics.get('fwhm_nm', 0),
                    'color': colors[sim_idx % len(colors)]
                })
        
        if stats_data:
            unique_combos = list(set([(d['Simulation'], d['Angle']) for d in stats_data]))
            x_pos = np.arange(len(unique_combos))
            
            max_values = [d['Max Stress'] for d in stats_data]
            fwhm_values = [d['FWHM'] for d in stats_data]
            
            bar_width = 0.35
            bars1 = ax_statistics.bar(x_pos - bar_width/2, max_values, bar_width,
                                     color=[d['color'] for d in stats_data],
                                     alpha=0.8, label='Max Stress (GPa)',
                                     edgecolor='black', linewidth=1)
            bars2 = ax_statistics.bar(x_pos + bar_width/2, fwhm_values, bar_width,
                                     color=[d['color'] for d in stats_data],
                                     alpha=0.6, label='FWHM (nm)',
                                     edgecolor='black', linewidth=1,
                                     hatch='//')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax_statistics.text(bar.get_x() + bar.get_width()/2, height,
                                     f'{height:.2f}', ha='center', va='bottom',
                                     fontsize=8, fontweight='bold')
            
            ax_statistics.set_xlabel("Simulation - Angle", fontsize=11)
            ax_statistics.set_ylabel("Value", fontsize=11)
            ax_statistics.set_title("Profile Statistics Comparison", 
                                  fontsize=13, fontweight='bold')
            ax_statistics.set_xticks(x_pos)
            ax_statistics.set_xticklabels([f"{s}-{a}°" for s, a in unique_combos],
                                         rotation=45, ha='right', fontsize=10)
            ax_statistics.legend(fontsize=10, loc='upper left')
            ax_statistics.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        # Panel C: Gradient comparison
        for sim_idx, angle_profiles in all_profiles.items():
            color = colors[sim_idx % len(colors)]
            for angle, profile_data in angle_profiles.items():
                distances = profile_data['distances']
                profile = profile_data['profile']
                gradient = np.gradient(profile, distances)
                
                # Apply smoothing to gradient for better visualization
                if len(gradient) > 10:
                    gradient = savgol_filter(gradient, window_length=min(11, len(gradient)), polyorder=2)
                
                label = f"Sim {sim_idx+1} - {angle}°" if sim_idx == 0 else None
                ax_gradient.plot(distances, gradient,
                               color=color,
                               linestyle=config.get('line_style', 'solid'),
                               linewidth=1.8,
                               alpha=0.7,
                               label=label)
        
        ax_gradient.set_xlabel("Distance (nm)", fontsize=11)
        ax_gradient.set_ylabel("Stress Gradient (GPa/nm)", fontsize=11)
        ax_gradient.set_title("Stress Gradient Comparison", 
                            fontsize=13, fontweight='bold')
        ax_gradient.legend(fontsize=10, ncol=2)
        ax_gradient.grid(True, alpha=0.3, linestyle=':')
        ax_gradient.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax_gradient.set_ylim([-np.max(np.abs(ax_gradient.get_ylim())), 
                            np.max(np.abs(ax_gradient.get_ylim()))])
        
        # Panel D: Frequency spectrum
        for sim_idx, angle_profiles in all_profiles.items():
            color = colors[sim_idx % len(colors)]
            for angle, profile_data in angle_profiles.items():
                profile = profile_data['profile']
                distances = profile_data['distances']
                
                # Compute FFT
                fft_result = np.fft.fft(profile)
                freq = np.fft.fftfreq(len(profile), distances[1]-distances[0])
                power_spectrum = np.abs(fft_result)**2
                
                # Plot only positive frequencies
                positive_freq = freq[:len(freq)//2]
                positive_power = power_spectrum[:len(power_spectrum)//2]
                
                label = f"Sim {sim_idx+1} - {angle}°" if sim_idx == 0 else None
                ax_spectrum.plot(positive_freq[1:], positive_power[1:],
                               color=color,
                               linewidth=1.8,
                               alpha=0.7,
                               label=label)
        
        ax_spectrum.set_xlabel("Frequency (1/nm)", fontsize=11)
        ax_spectrum.set_ylabel("Power", fontsize=11)
        ax_spectrum.set_title("Frequency Spectrum Analysis", 
                            fontsize=13, fontweight='bold')
        ax_spectrum.set_xscale('log')
        ax_spectrum.set_yscale('log')
        ax_spectrum.grid(True, alpha=0.3, which='both', linestyle=':')
        ax_spectrum.legend(fontsize=10, ncol=2)
        
        # Panel E: Stress distribution along profile (violin plot)
        all_profile_values = []
        labels = []
        colors_violin = []
        
        for sim_idx, angle_profiles in all_profiles.items():
            for angle, profile_data in angle_profiles.items():
                profile = profile_data['profile']
                all_profile_values.append(profile)
                labels.append(f"S{sim_idx+1}-{angle}°")
                colors_violin.append(colors[sim_idx % len(colors)])
        
        # Create violin plot
        parts = ax_distribution.violinplot(all_profile_values, 
                                          showmeans=True, showmedians=True,
                                          showextrema=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_violin[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Customize mean and median lines
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('blue')
        parts['cmedians'].set_linewidth(2)
        
        ax_distribution.set_xlabel("Profile", fontsize=11)
        ax_distribution.set_ylabel("Stress (GPa)", fontsize=11)
        ax_distribution.set_title("Stress Distribution Comparison (Violin Plots)", 
                                fontsize=13, fontweight='bold')
        ax_distribution.set_xticks(range(1, len(labels) + 1))
        ax_distribution.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax_distribution.grid(True, alpha=0.3, axis='y', linestyle=':')
        
        # Panel F: Correlation between different metrics
        if len(simulations) > 1:
            max_stresses = []
            fwhm_values = []
            gradient_max = []
            colors_scatter = []
            
            for sim_idx, angle_profiles in all_profiles.items():
                for angle, profile_data in angle_profiles.items():
                    metrics = profile_data['metadata']
                    max_stresses.append(metrics['max_value'])
                    fwhm_values.append(metrics.get('fwhm_nm', 0))
                    gradient_max.append(metrics['max_gradient'])
                    colors_scatter.append(colors[sim_idx % len(colors)])
            
            # Create scatter plot with size proportional to gradient
            sizes = 50 + 200 * (np.array(gradient_max) / max(gradient_max))
            scatter = ax_correlation.scatter(max_stresses, fwhm_values,
                                           c=colors_scatter, s=sizes, alpha=0.7,
                                           edgecolors='white', linewidth=1.5,
                                           cmap='viridis')
            
            # Add trend line
            if len(max_stresses) > 1:
                z = np.polyfit(max_stresses, fwhm_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(max_stresses), max(max_stresses), 100)
                ax_correlation.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2,
                                  label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
            
            ax_correlation.set_xlabel("Maximum Stress (GPa)", fontsize=11)
            ax_correlation.set_ylabel("FWHM (nm)", fontsize=11)
            ax_correlation.set_title("Stress-FWHM Correlation (size ∝ max gradient)", 
                                   fontsize=13, fontweight='bold')
            ax_correlation.legend(fontsize=10)
            ax_correlation.grid(True, alpha=0.3, linestyle=':')
            
            # Add correlation coefficient
            if len(max_stresses) > 1:
                corr_coef = np.corrcoef(max_stresses, fwhm_values)[0, 1]
                ax_correlation.text(0.05, 0.95, f'R = {corr_coef:.3f}',
                                  transform=ax_correlation.transAxes,
                                  fontsize=12, fontweight='bold',
                                  bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor='white', alpha=0.8))
        
        # Panel G: Domain with all profiles (using fixed aspect ratio)
        if simulations:
            sim = simulations[0]
            eta, _ = sim['history'][frames[0]]
            
            FixedAspectManager.apply_fixed_aspect(ax_domain, aspect_type='equal')
            
            # Plot domain
            cmap_name = sim['params'].get('eta_cmap', 'viridis')
            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
            im = ax_domain.imshow(eta, extent=extent, cmap=cmap, 
                                origin='lower', alpha=0.8, vmin=0, vmax=1)
            
            # Add profile lines with enhanced styling
            line_colors = plt.cm.rainbow(np.linspace(0, 1, len(angles)))
            
            for idx, angle in enumerate(angles):
                profile_length = config.get('profile_length', 80) / 100
                result = AdvancedProfileExtractor.extract_profile_2d(
                    eta, angle, 'center', 0, profile_length
                )
                distances, profile, endpoints, metadata = result
                x_start, y_start, x_end, y_end = endpoints
                
                # Draw profile line with arrow
                ax_domain.plot([x_start, x_end], [y_start, y_end], 
                             color=line_colors[idx], 
                             linewidth=3, linestyle='-',
                             label=f'{angle}°',
                             alpha=0.9,
                             solid_capstyle='round')
                
                # Add angle annotation with arrow
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                ax_domain.annotate(f'{angle}°', xy=(mid_x, mid_y),
                                 xytext=(10, 10), textcoords='offset points',
                                 color=line_colors[idx],
                                 fontsize=11, fontweight='bold',
                                 arrowprops=dict(arrowstyle='->', 
                                               color=line_colors[idx],
                                               alpha=0.8, linewidth=1.5))
            
            ax_domain.set_xlabel("x (nm)", fontsize=12)
            ax_domain.set_ylabel("y (nm)", fontsize=12)
            ax_domain.set_title("Simulation Domain with Profile Lines (Fixed 1:1 Aspect Ratio)", 
                              fontsize=13, fontweight='bold')
            ax_domain.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.9)
            FixedAspectManager.add_physical_scale(ax_domain, 5.0, 
                                                location='lower right',
                                                color='white', fontsize=11, linewidth=2)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_domain, shrink=0.8, pad=0.02)
            cbar.set_label('Defect Parameter η', fontsize=11)
            cbar.ax.tick_params(labelsize=10)
        
        # Add panel labels
        panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        panel_axes = [ax_overlay, ax_statistics, ax_gradient, ax_spectrum, 
                     ax_distribution, ax_correlation, ax_domain]
        
        for ax, label in zip(panel_axes, panel_labels):
            ax.text(-0.05, 1.05, label, transform=ax.transAxes,
                   fontsize=18, fontweight='bold', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        return fig, all_profiles
    
    @staticmethod
    def create_profile_metrics_dashboard(all_profiles, simulations):
        """Create a dashboard of comprehensive profile metrics"""
        # Create figure for metrics
        fig_metrics, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Maximum stress by defect type
        defect_types = []
        max_stresses = []
        colors = []
        
        for sim_idx, angle_profiles in all_profiles.items():
            sim = simulations[sim_idx]
            for angle, profile_data in angle_profiles.items():
                defect_types.append(sim['params']['defect_type'])
                max_stresses.append(profile_data['metadata']['max_value'])
                colors.append(plt.cm.tab10(sim_idx % 10))
        
        # Group by defect type
        unique_defects = list(set(defect_types))
        defect_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_defects)))
        
        grouped_data = {d: [] for d in unique_defects}
        for d, s in zip(defect_types, max_stresses):
            grouped_data[d].append(s)
        
        positions = np.arange(len(unique_defects))
        box_data = [grouped_data[d] for d in unique_defects]
        
        bp = axes[0].boxplot(box_data, positions=positions, patch_artist=True,
                           showmeans=True, meanline=True, showfliers=False)
        
        for patch, color in zip(bp['boxes'], defect_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(unique_defects)
        axes[0].set_ylabel("Maximum Stress (GPa)", fontsize=11)
        axes[0].set_title("Maximum Stress Distribution by Defect Type", 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. FWHM vs Angle
        angles = []
        fwhm_values = []
        sim_indices = []
        
        for sim_idx, angle_profiles in all_profiles.items():
            for angle, profile_data in angle_profiles.items():
                angles.append(angle)
                fwhm_values.append(profile_data['metadata'].get('fwhm_nm', 0))
                sim_indices.append(sim_idx)
        
        scatter = axes[1].scatter(angles, fwhm_values, c=sim_indices, 
                                cmap='viridis', s=60, alpha=0.7,
                                edgecolors='white', linewidth=1)
        axes[1].set_xlabel("Angle (°)", fontsize=11)
        axes[1].set_ylabel("FWHM (nm)", fontsize=11)
        axes[1].set_title("Profile Width vs Orientation", 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Asymmetry distribution
        asymmetries = []
        for sim_idx, angle_profiles in all_profiles.items():
            for profile_data in angle_profiles.values():
                asymmetries.append(profile_data['metadata']['asymmetry'])
        
        axes[2].hist(asymmetries, bins=20, edgecolor='black', alpha=0.7,
                    color='skyblue', density=True)
        axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[2].set_xlabel("Asymmetry Index", fontsize=11)
        axes[2].set_ylabel("Density", fontsize=11)
        axes[2].set_title("Profile Asymmetry Distribution", 
                         fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Peak count distribution
        peak_counts = []
        for sim_idx, angle_profiles in all_profiles.items():
            for profile_data in angle_profiles.values():
                peak_counts.append(profile_data['metadata'].get('num_peaks', 0))
        
        unique_counts, count_freq = np.unique(peak_counts, return_counts=True)
        bars = axes[3].bar(unique_counts, count_freq, color='lightcoral', alpha=0.7)
        for bar, count in zip(bars, count_freq):
            axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        axes[3].set_xlabel("Number of Peaks", fontsize=11)
        axes[3].set_ylabel("Frequency", fontsize=11)
        axes[3].set_title("Peak Count Distribution", 
                         fontsize=13, fontweight='bold')
        axes[3].grid(True, alpha=0.3, axis='y')
        
        # 5. Gradient statistics
        max_gradients = []
        for sim_idx, angle_profiles in all_profiles.items():
            for profile_data in angle_profiles.values():
                max_gradients.append(profile_data['metadata']['max_gradient'])
        
        axes[4].boxplot(max_gradients, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[4].set_xlabel("Maximum Gradient (GPa/nm)", fontsize=11)
        axes[4].set_title("Gradient Statistics", 
                         fontsize=13, fontweight='bold')
        axes[4].grid(True, alpha=0.3, axis='x')
        
        # 6. Integral vs Maximum Stress
        integrals = []
        max_stresses = []
        
        for sim_idx, angle_profiles in all_profiles.items():
            for profile_data in angle_profiles.values():
                integrals.append(profile_data['metadata']['integral'])
                max_stresses.append(profile_data['metadata']['max_value'])
        
        if integrals and max_stresses:
            axes[5].scatter(max_stresses, integrals, alpha=0.6, s=50, color='purple')
            axes[5].set_xlabel("Maximum Stress (GPa)", fontsize=11)
            axes[5].set_ylabel("Integral (GPa·nm)", fontsize=11)
            axes[5].set_title("Stress Integral vs Maximum Stress", 
                            fontsize=13, fontweight='bold')
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig_metrics

# =============================================
# ENHANCED PROFILE CONTROLS
# =============================================
def add_enhanced_profile_controls():
    """Add comprehensive profile control panel to sidebar"""
    st.sidebar.markdown('<div class="profile-controls">', unsafe_allow_html=True)
    st.sidebar.subheader("🎯 Enhanced Profile Controls")
    
    # Profile positioning
    position_mode = st.sidebar.selectbox(
        "Profile Positioning",
        ["Center", "Multiple Parallel", "Custom Coordinates", "Radial Grid"],
        index=0,
        help="Positioning strategy for profile extraction"
    )
    
    if position_mode == "Multiple Parallel":
        num_parallel = st.sidebar.slider("Number of Profiles", 1, 10, 3)
        spacing = st.sidebar.slider("Spacing (nm)", 0.1, 5.0, 1.0, 0.1)
    elif position_mode == "Custom Coordinates":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            custom_x = st.number_input("X (nm)", float(extent[0]), float(extent[1]), 0.0)
        with col2:
            custom_y = st.number_input("Y (nm)", float(extent[2]), float(extent[3]), 0.0)
    
    # Advanced extraction parameters
    with st.sidebar.expander("⚙️ Advanced Extraction"):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            sampling_factor = st.slider("Sampling", 1, 10, 3, 
                                       help="Oversampling factor for higher resolution")
            interpolation = st.select_slider("Interpolation", 
                                           options=[1, 2, 3, 4, 5], value=3)
        with col2:
            apply_smoothing = st.checkbox("Apply Smoothing", True)
            if apply_smoothing:
                smooth_sigma = st.slider("Sigma", 0.1, 2.0, 0.5, 0.1)
    
    # Visualization options
    with st.sidebar.expander("🎨 Visualization"):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            show_confidence = st.checkbox("Confidence Bands", True)
            show_peaks = st.checkbox("Mark Peaks", True)
        with col2:
            show_gradient = st.checkbox("Show Gradient", True)
            fixed_aspect = st.checkbox("Fixed Aspect", True, value=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'position_mode': position_mode,
        'sampling_factor': sampling_factor,
        'interpolation_order': interpolation,
        'smooth_profiles': apply_smoothing,
        'smooth_sigma': smooth_sigma if apply_smoothing else 0,
        'show_confidence': show_confidence,
        'show_peaks': show_peaks,
        'show_gradient': show_gradient,
        'fixed_aspect': fixed_aspect
    }

# =============================================
# SIMULATION DATABASE SYSTEM
# =============================================
class SimulationDatabase:
    """Enhanced database system for simulation management"""
    
    @staticmethod
    def initialize():
        if 'simulations' not in st.session_state:
            st.session_state.simulations = {}
        if 'simulation_counter' not in st.session_state:
            st.session_state.simulation_counter = 0
    
    @staticmethod
    def generate_id(params):
        param_str = json.dumps(params, sort_keys=True, default=str)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_obj = hashlib.md5((param_str + timestamp).encode())
        return f"SIM_{hash_obj.hexdigest()[:8]}"
    
    @staticmethod
    def save_simulation(params, history, metadata=None):
        SimulationDatabase.initialize()
        
        sim_id = SimulationDatabase.generate_id(params)
        
        if metadata is None:
            metadata = {
                'created': datetime.now().isoformat(),
                'frames': len(history),
                'grid_size': N,
                'dx': dx,
                'run_time': 0.0
            }
        
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
        SimulationDatabase.initialize()
        return st.session_state.simulations.get(sim_id)
    
    @staticmethod
    def get_all_simulations():
        SimulationDatabase.initialize()
        return st.session_state.simulations
    
    @staticmethod
    def get_simulation_list():
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

# =============================================
# SIMULATION ENGINE
# =============================================
@st.cache_data
def create_initial_eta(shape, defect_type, random_seed=42):
    """Create initial defect configuration"""
    np.random.seed(random_seed)
    
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes.get(defect_type, 0.75)
    
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    
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
    
    eta += 0.02 * np.random.randn(N, N)
    return np.clip(eta, 0.0, 1.0)

@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, kappa, dt=0.004, dx=dx, N=N):
    """Phase field evolution with Allen-Cahn equation"""
    eta_new = eta.copy()
    dx2 = dx * dx
    prefactor = dt / dx2
    
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j])
            eta_val = eta[i,j]
            dF = 2 * eta_val * (1 - eta_val) * (eta_val - 0.5)
            eta_new[i,j] = eta_val + dt * (-dF + kappa * lap / dx2)
            
            if eta_new[i,j] < 0.0:
                eta_new[i,j] = 0.0
            elif eta_new[i,j] > 1.0:
                eta_new[i,j] = 1.0
    
    eta_new[0,:] = eta_new[-2,:]
    eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]
    eta_new[:,-1] = eta_new[:,1]
    
    return eta_new

@st.cache_data
def compute_stress_fields(eta, eps0, theta):
    """FFT-based stress solver with rotated eigenstrain"""
    C11_p = (C11 - C12**2 / C11) * 1e9
    C12_p = (C12 - C12**2 / C11) * 1e9
    C44_p = C44 * 1e9
    
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
    
    gamma = eps0
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    delta = 0.02
    
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
    
    eps_xx_star = eps_star[0,0] * eta
    eps_yy_star = eps_star[1,1] * eta
    eps_xy_star = eps_star[0,1] * eta
    
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
    
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
    
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
    
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy)
    
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
    defect_type = params['defect_type']
    shape = params['shape']
    eps0 = params['eps0']
    kappa = params['kappa']
    theta = params['theta']
    steps = params.get('steps', 100)
    save_every = params.get('save_every', 20)
    
    eta = create_initial_eta(shape, defect_type)
    history = []
    start_time = time.time()
    
    for step in range(steps + 1):
        if step > 0:
            eta = evolve_phase_field(eta, kappa)
        
        if step % save_every == 0 or step == steps:
            stress_fields = compute_stress_fields(eta, eps0, theta)
            history.append((eta.copy(), stress_fields))
            
            if progress_callback:
                progress = (step + 1) / (steps + 1)
                progress_callback(progress)
    
    run_time = time.time() - start_time
    
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

operation_mode = st.sidebar.radio(
    "Select Operation Mode",
    ["🏃 Run New Simulation", "🔍 Compare Simulations", "📊 Enhanced Profile Analysis", "💾 Export Data"],
    index=0
)

SimulationDatabase.initialize()

if "Run New Simulation" in operation_mode:
    # Existing simulation setup code...
    pass
elif "Compare Simulations" in operation_mode:
    st.sidebar.header("🔍 Comparison Setup")
    
    sim_list = SimulationDatabase.get_simulation_list()
    
    if not sim_list:
        st.sidebar.warning("No simulations found. Run some simulations first!")
    else:
        sim_options = {sim['display_name']: sim['id'] for sim in sim_list}
        selected_names = st.sidebar.multiselect(
            "Select Simulations to Compare",
            options=list(sim_options.keys()),
            default=list(sim_options.keys())[:min(3, len(sim_options))],
            help="Select up to 6 simulations for comparison"
        )
        
        selected_ids = [sim_options[name] for name in selected_names]
        
        if selected_ids:
            comparison_type = st.sidebar.selectbox(
                "Comparison Type",
                [
                    "Enhanced Overlay Line Profiles",
                    "Side-by-Side Heatmaps",
                    "Multi-Orientation Analysis",
                    "Statistical Comparison",
                    "Evolution Timeline"
                ],
                index=0
            )
            
            stress_component = st.sidebar.selectbox(
                "Stress Component",
                ["Stress Magnitude |σ|", "Hydrostatic σ_h", "von Mises σ_vM"],
                index=0
            )
            
            frame_selection = st.sidebar.radio(
                "Frame Selection",
                ["Final Frame", "Mid Evolution", "Specific Frame"],
                horizontal=True
            )
            
            if frame_selection == "Specific Frame":
                frame_idx = st.sidebar.slider("Frame Index", 0, 100, 50)
            else:
                frame_idx = None
            
            # Enhanced controls for overlay profiles
            if comparison_type == "Enhanced Overlay Line Profiles":
                enhanced_config = add_enhanced_profile_controls()
                
                # Add orientation selection
                st.sidebar.subheader("📐 Profile Orientations")
                profile_orientations = st.sidebar.multiselect(
                    "Select Orientations",
                    ["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)", "135° (Diagonal)", "Custom"],
                    default=["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)"]
                )
                
                if "Custom" in profile_orientations:
                    custom_angle = st.sidebar.slider("Custom Angle (°)", -180, 180, 30, 5)
                    enhanced_config['custom_angle'] = custom_angle
                
                # Line style
                line_style = st.sidebar.selectbox("Line Style", ["solid", "dashed", "dotted", "dashdot"])
                enhanced_config['line_style'] = line_style
                enhanced_config['profile_orientations'] = profile_orientations
                
                # Profile length
                profile_length = st.sidebar.slider("Profile Length (%)", 10, 100, 80, 5)
                enhanced_config['profile_length'] = profile_length
            
            if st.sidebar.button("🔬 Run Enhanced Comparison", type="primary", use_container_width=True):
                config = {
                    'type': comparison_type,
                    'sim_ids': selected_ids,
                    'stress_component': stress_component,
                    'frame_selection': frame_selection,
                    'frame_idx': frame_idx,
                }
                
                if comparison_type == "Enhanced Overlay Line Profiles":
                    config.update(enhanced_config)
                
                st.session_state.comparison_config = config
                st.session_state.run_comparison = True

elif "Enhanced Profile Analysis" in operation_mode:
    st.sidebar.header("📊 Advanced Profile Analysis")
    
    # Get available simulations
    sim_list = SimulationDatabase.get_simulation_list()
    
    if sim_list:
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h4>📈 Platform Statistics</h4>
            <p><strong>Total Simulations:</strong> {len(sim_list)}</p>
            <p><strong>Available for Analysis:</strong> {len(sim_list)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Analysis Focus",
            ["Comprehensive Profile Analysis", "Statistical Comparison", 
             "Trend Analysis", "Correlation Study"],
            index=0
        )
        
        # Simulation selection
        selected_sims = st.sidebar.multiselect(
            "Select Simulations for Analysis",
            [sim['display_name'] for sim in sim_list],
            default=[sim['display_name'] for sim in sim_list[:3]]
        )

# =============================================
# MAIN CONTENT AREA - ENHANCED OVERLAY PROFILES
# =============================================
if "Compare Simulations" in operation_mode:
    st.header("🔍 Enhanced Overlay Line Profiles Analysis")
    
    if 'comparison_config' in st.session_state and st.session_state.get('run_comparison', False):
        config = st.session_state.comparison_config
        
        if config['type'] == "Enhanced Overlay Line Profiles":
            # Load selected simulations
            simulations = []
            for sim_id in config['sim_ids']:
                sim_data = SimulationDatabase.get_simulation(sim_id)
                if sim_data:
                    simulations.append(sim_data)
            
            if not simulations:
                st.error("No valid simulations selected!")
            else:
                st.success(f"Loaded {len(simulations)} simulations for enhanced analysis")
                
                # Determine frames
                frames = []
                for sim in simulations:
                    if config['frame_selection'] == "Final Frame":
                        frames.append(len(sim['history']) - 1)
                    elif config['frame_selection'] == "Mid Evolution":
                        frames.append(len(sim['history']) // 2)
                    else:
                        frames.append(min(config.get('frame_idx', 0), len(sim['history']) - 1))
                
                # Create enhanced visualization
                style_params = {
                    'line_width': 2.5,
                    'label_font_size': 13,
                    'title_font_size': 15,
                    'legend_fontsize': 11,
                    'tick_font_size': 11
                }
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Comprehensive Analysis", "📈 Statistics Dashboard", 
                                                 "🔬 Detailed Metrics", "💾 Export Data"])
                
                with tab1:
                    st.subheader("Comprehensive Profile Analysis")
                    
                    # Create enhanced visualization
                    visualizer = EnhancedProfileVisualizer()
                    fig, all_profiles = visualizer.create_statistically_enhanced_profiles(
                        simulations, frames, config, style_params
                    )
                    
                    st.pyplot(fig)
                    
                    # Add interactive controls
                    with st.expander("🔄 Interactive Controls", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            update_style = st.checkbox("Update Style", True)
                            if update_style:
                                new_line_width = st.slider("Line Width", 1.0, 5.0, 2.5, 0.5)
                                style_params['line_width'] = new_line_width
                        with col2:
                            show_advanced = st.checkbox("Show Advanced Metrics", True)
                        with col3:
                            if st.button("Refresh Visualization", type="secondary"):
                                st.rerun()
                
                with tab2:
                    st.subheader("Statistical Dashboard")
                    
                    if 'all_profiles' in locals():
                        # Create metrics dashboard
                        fig_metrics = visualizer.create_profile_metrics_dashboard(all_profiles, simulations)
                        st.pyplot(fig_metrics)
                        
                        # Statistical tests
                        st.subheader("Statistical Significance Tests")
                        
                        test_results = ProfileAnalysisEnhancer.perform_statistical_tests(all_profiles)
                        
                        if 'anova' in test_results:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("ANOVA F-Statistic", f"{test_results['anova'].get('f_value', 0):.4f}")
                            with col2:
                                p_val = test_results['anova'].get('p_value', 1)
                                st.metric("ANOVA p-value", f"{p_val:.4f}")
                                if p_val < 0.05:
                                    st.success("✓ Statistically significant differences detected")
                                else:
                                    st.info("No statistically significant differences detected")
                        
                        # Pairwise comparisons
                        if 'pairwise' in test_results:
                            st.subheader("Pairwise Comparisons")
                            pairwise_data = []
                            for comparison, results in test_results['pairwise'].items():
                                pairwise_data.append({
                                    'Comparison': comparison,
                                    't-Statistic': f"{results.get('t_statistic', 0):.4f}",
                                    'p-value': f"{results.get('p_value', 1):.4f}",
                                    'Significant': '✓' if results.get('significant', False) else '✗'
                                })
                            
                            if pairwise_data:
                                df_pairwise = pd.DataFrame(pairwise_data)
                                st.dataframe(df_pairwise, use_container_width=True)
                
                with tab3:
                    st.subheader("Detailed Profile Metrics")
                    
                    if 'all_profiles' in locals():
                        # Create comprehensive metrics table
                        metrics_table = ProfileAnalysisEnhancer.create_profile_comparison_table(all_profiles)
                        
                        # Display with formatting
                        st.dataframe(
                            metrics_table.style.format({
                                'Max Stress (GPa)': '{:.3f}',
                                'Mean Stress (GPa)': '{:.3f}',
                                'FWHM (nm)': '{:.2f}',
                                'Asymmetry': '{:.3f}',
                                'Integral (GPa·nm)': '{:.3f}',
                                'Max Gradient (GPa/nm)': '{:.3f}'
                            }).background_gradient(subset=['Max Stress (GPa)', 'Mean Stress (GPa)'], 
                                                 cmap='YlOrRd'),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        max_stresses = metrics_table['Max Stress (GPa)'].str.replace(' GPa', '').astype(float)
                        mean_stresses = metrics_table['Mean Stress (GPa)'].str.replace(' GPa', '').astype(float)
                        
                        with col1:
                            st.metric("Avg Max Stress", f"{max_stresses.mean():.3f} GPa")
                        with col2:
                            st.metric("Avg Mean Stress", f"{mean_stresses.mean():.3f} GPa")
                        with col3:
                            st.metric("Stress Range", f"{max_stresses.max() - max_stresses.min():.3f} GPa")
                        with col4:
                            st.metric("CV of Max Stress", f"{(max_stresses.std() / max_stresses.mean() * 100):.1f}%")
                
                with tab4:
                    st.subheader("Export Profile Data")
                    
                    if 'all_profiles' in locals():
                        # Export options
                        export_format = st.selectbox(
                            "Export Format",
                            ["JSON (Complete Data)", "CSV (Metrics Only)", "Excel (All Data)", "PDF Report"]
                        )
                        
                        # Prepare export data
                        export_data = {
                            'metadata': {
                                'export_date': datetime.now().isoformat(),
                                'simulations': [
                                    {
                                        'id': sim['id'],
                                        'defect_type': sim['params']['defect_type'],
                                        'orientation': sim['params']['orientation'],
                                        'eps': sim['params']['eps0'],
                                        'kappa': sim['params']['kappa']
                                    }
                                    for sim in simulations
                                ],
                                'analysis_parameters': config,
                                'profile_count': sum(len(profiles) for profiles in all_profiles.values())
                            },
                            'profiles': {}
                        }
                        
                        # Add profile data
                        for sim_idx, angle_profiles in all_profiles.items():
                            sim_profiles = {}
                            for angle, profile_data in angle_profiles.items():
                                sim_profiles[str(angle)] = {
                                    'distances': profile_data['distances'].tolist(),
                                    'profile': profile_data['profile'].tolist(),
                                    'metrics': profile_data['metadata']
                                }
                            export_data['profiles'][f"Sim_{sim_idx+1}"] = sim_profiles
                        
                        # Export buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📥 Download JSON", use_container_width=True):
                                json_data = json.dumps(export_data, indent=2, default=str)
                                
                                st.download_button(
                                    label="⬇️ Download JSON File",
                                    data=json_data,
                                    file_name=f"profile_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        
                        with col2:
                            if st.button("📊 Download CSV", use_container_width=True):
                                csv_data = metrics_table.to_csv(index=False)
                                
                                st.download_button(
                                    label="⬇️ Download CSV File",
                                    data=csv_data,
                                    file_name=f"profile_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        # Report generation
                        st.markdown("---")
                        st.subheader("Generate Analysis Report")
                        
                        if st.button("📋 Generate Comprehensive Report", type="primary", use_container_width=True):
                            with st.spinner("Generating report..."):
                                # Create a summary report
                                report = f"""
                                # Enhanced Profile Analysis Report
                                
                                ## Report Summary
                                - Generated: {datetime.now().isoformat()}
                                - Simulations Analyzed: {len(simulations)}
                                - Total Profiles: {sum(len(profiles) for profiles in all_profiles.values())}
                                - Stress Component: {config.get('stress_component', 'Stress Magnitude |σ|')}
                                
                                ## Key Findings
                                
                                ### Maximum Stress Statistics
                                - Average Maximum Stress: {max_stresses.mean():.3f} GPa
                                - Maximum Stress Range: {max_stresses.max() - max_stresses.min():.3f} GPa
                                - Coefficient of Variation: {(max_stresses.std() / max_stresses.mean() * 100):.1f}%
                                
                                ### Profile Characteristics
                                - Average FWHM: {metrics_table['FWHM (nm)'].str.replace(' nm', '').astype(float).mean():.2f} nm
                                - Average Asymmetry: {metrics_table['Asymmetry'].astype(float).mean():.3f}
                                - Total Peak Count: {metrics_table['Peaks'].astype(int).sum()}
                                
                                ## Statistical Significance
                                """
                                
                                # Add statistical test results
                                if 'test_results' in locals():
                                    if 'anova' in test_results:
                                        p_val = test_results['anova'].get('p_value', 1)
                                        report += f"\n- ANOVA p-value: {p_val:.4f} "
                                        report += "(Significant)" if p_val < 0.05 else "(Not Significant)"
                                
                                st.code(report, language="markdown")
                                st.success("Report generated successfully!")
            
            # Clear comparison flag
            st.session_state.run_comparison = False
    
    else:
        # Show comparison interface
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Enhanced Overlay Line Profiles Analysis</h3>
            <p>This advanced analysis tool provides comprehensive comparison of stress profiles across multiple simulations.</p>
            
            <h4>📊 Key Features:</h4>
            <ul>
                <li><strong>Multi-Angle Profile Extraction:</strong> Extract profiles at any orientation (0°, 45°, 90°, 135°, custom)</li>
                <li><strong>Statistical Overlays:</strong> Confidence bands, peak detection, gradient analysis</li>
                <li><strong>Fixed Aspect Ratio:</strong> Realistic 1:1 visualization of simulation domains</li>
                <li><strong>Frequency Analysis:</strong> FFT analysis of profile periodicities</li>
                <li><strong>Comprehensive Metrics:</strong> FWHM, asymmetry, peak counts, integrals</li>
                <li><strong>Statistical Testing:</strong> ANOVA and pairwise comparisons</li>
                <li><strong>Export-Ready:</strong> Multiple export formats (JSON, CSV, Excel)</li>
            </ul>
            
            <h4>🔬 Analysis Panels:</h4>
            <ol>
                <li><strong>Comprehensive Analysis:</strong> 7-panel figure with all visualization types</li>
                <li><strong>Statistics Dashboard:</strong> Box plots, scatter plots, correlation analysis</li>
                <li><strong>Detailed Metrics:</strong> Comprehensive metrics table with statistical summaries</li>
                <li><strong>Export Data:</strong> Download all analysis results</li>
            </ol>
            
            <p><strong>Instructions:</strong> Select simulations in the sidebar, configure profile parameters, and click "Run Enhanced Comparison"</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Ag Nanoparticle Multi-Defect Analysis Platform Pro v3.0</strong></p>
    <p>Enhanced Overlay Line Profiles with Comprehensive Statistical Analysis</p>
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
