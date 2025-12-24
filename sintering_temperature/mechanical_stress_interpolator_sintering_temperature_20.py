import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from io import BytesIO
import json
import zipfile
from numba import jit, prange

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# =============================================
# SINTERING TEMPERATURE CALCULATOR (ENHANCED)
# =============================================

class SinteringTemperatureCalculator:
    """Calculate sintering temperature based on hydrostatic stress at habit plane"""
    
    def __init__(self, T0=623.0, beta=0.95, G=30.0, sigma_peak=28.5):
        self.T0 = T0  # Reference temperature at zero stress (K)
        self.beta = beta  # Calibration factor
        self.G = G  # Shear modulus of Ag (GPa)
        self.sigma_peak = sigma_peak  # Peak hydrostatic stress (GPa)
        self.T_min = 367.0  # Minimum sintering temperature at peak stress (K)
        
        # Material properties for Ag
        self.kB = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.Q_a = 1.1  # Activation energy for Ag diffusion (eV)
        self.omega = 0.85 * (0.408e-9)**3  # Activation volume (m³)
        self.omega_eV_per_GPa = self.omega * 6.242e18  # Convert to eV/GPa
        
        # Defect-specific diffusion parameters
        self.defect_diffusion_params = {
            'ISF': {'Q_a': 1.05, 'D0': 1e-5, 'omega_factor': 0.8},
            'ESF': {'Q_a': 0.95, 'D0': 1e-4, 'omega_factor': 0.9},
            'Twin': {'Q_a': 0.85, 'D0': 1e-3, 'omega_factor': 1.0},
            'No Defect': {'Q_a': 1.1, 'D0': 1e-6, 'omega_factor': 0.5},
            'Unknown': {'Q_a': 1.1, 'D0': 1e-6, 'omega_factor': 1.0}
        }
    
    def compute_sintering_temperature_exponential(self, sigma_h):
        """Compute sintering temperature using exponential empirical model"""
        sigma_abs = np.abs(sigma_h)
        T_sinter = self.T0 * np.exp(-self.beta * sigma_abs / self.G)
        return T_sinter
    
    def compute_sintering_temperature_arrhenius(self, sigma_h, D0=1e-6, D_crit=1e-10):
        """Compute sintering temperature using stress-modified Arrhenius equation"""
        sigma_abs = np.abs(sigma_h)
        Q_eff = self.Q_a - self.omega_eV_per_GPa * sigma_abs
        # Ensure Q_eff is positive
        Q_eff = max(Q_eff, 0.1)
        if D0 / D_crit > 0:
            T_sinter = Q_eff / (self.kB * np.log(D0 / D_crit))
        else:
            T_sinter = self.T0
        return max(T_sinter, self.T_min)
    
    def compute_sintering_temperature_arrhenius_defect(self, sigma_h, defect_type='Twin', D_crit=1e-10):
        """Compute sintering temperature using defect-specific Arrhenius parameters"""
        sigma_abs = np.abs(sigma_h)
        
        if defect_type in self.defect_diffusion_params:
            params = self.defect_diffusion_params[defect_type]
            Q_a = params['Q_a']
            D0 = params['D0']
            omega_factor = params['omega_factor']
        else:
            Q_a = self.Q_a
            D0 = 1e-6
            omega_factor = 1.0
        
        Q_eff = Q_a - (self.omega_eV_per_GPa * omega_factor * sigma_abs)
        Q_eff = max(Q_eff, 0.1)
        
        if D0 / D_crit > 0:
            T_sinter = Q_eff / (self.kB * np.log(D0 / D_crit))
        else:
            T_sinter = self.T0
        
        return max(T_sinter, self.T_min)
    
    def compute_stress_for_temperature(self, T_sinter):
        """Compute required hydrostatic stress to achieve given sintering temperature"""
        if T_sinter <= 0:
            return 0.0
        sigma_h = -(self.G / self.beta) * np.log(T_sinter / self.T0)
        return sigma_h
    
    def compute_peak_stress_from_temperature(self, T_min=None):
        """Compute peak hydrostatic stress from minimum sintering temperature"""
        if T_min is None:
            T_min = self.T_min
        sigma_peak = -(self.G / self.beta) * np.log(T_min / self.T0)
        return sigma_peak
    
    def map_system_to_temperature(self, sigma_h):
        """Map hydrostatic stress to system classification"""
        sigma_abs = np.abs(sigma_h)
        
        if sigma_abs < 5.0:
            system = "System 1 (Perfect Crystal)"
            T_range = (620, 630)  # K
        elif sigma_abs < 20.0:
            system = "System 2 (Stacking Faults/Twins)"
            T_range = (450, 550)  # K
        else:
            system = "System 3 (Plastic Deformation)"
            T_range = (350, 400)  # K
            
        T_sinter = self.compute_sintering_temperature_exponential(sigma_abs)
        return system, T_range, T_sinter
    
    def get_theoretical_curve(self, max_stress=35.0, n_points=100):
        """Generate theoretical curve of T_sinter vs |σ_h|"""
        stresses = np.linspace(0, max_stress, n_points)
        T_exp = self.compute_sintering_temperature_exponential(stresses)
        T_arr = self.compute_sintering_temperature_arrhenius(stresses)
        
        return {
            'stresses': stresses,
            'T_exponential': T_exp,
            'T_arrhenius': T_arr,
            'T0': self.T0,
            'T_min': self.T_min,
            'sigma_peak': self.sigma_peak
        }
    
    def create_sintering_plot(self, stresses, temperatures, title="Sintering Temperature vs Hydrostatic Stress"):
        """Create detailed sintering temperature plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Main curve
        ax.plot(stresses, temperatures, 'b-', linewidth=3, label='Empirical Model')
        
        # System boundaries
        ax.axvspan(0, 5, alpha=0.1, color='green', label='System 1 (Perfect)')
        ax.axvspan(5, 20, alpha=0.1, color='orange', label='System 2 (SF/Twin)')
        ax.axvspan(20, 35, alpha=0.1, color='red', label='System 3 (Plastic)')
        
        # Reference points
        ax.plot(0, self.T0, 'go', markersize=12, label=f'System 1: {self.T0}K at 0 GPa')
        ax.plot(12.5, self.compute_sintering_temperature_exponential(12.5), 'yo', markersize=12, 
                label=f'System 2: ~475K at 12.5 GPa')
        ax.plot(self.sigma_peak, self.T_min, 'ro', markersize=12, 
                label=f'System 3: {self.T_min}K at {self.sigma_peak:.1f} GPa')
        
        # Lines for habit plane reference
        ax.axhline(self.T0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(self.T_min, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(self.sigma_peak, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Absolute Hydrostatic Stress |σ_h| (GPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sintering Temperature (K)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add second y-axis for Celsius
        ax2 = ax.twinx()
        celsius_ticks = ax.get_yticks()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
        ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        
        # Add annotations
        ax.text(0.02, 0.98, f'T₀ = {self.T0} K ({self.T0-273.15:.0f}°C) at σ_h = 0',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
        
        ax.text(0.02, 0.90, f'T_min = {self.T_min} K ({self.T_min-273.15:.0f}°C) at σ_h = {self.sigma_peak:.1f} GPa',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        
        return fig

# =============================================
# ENHANCED VISUALIZATION WITH SINTERING SUPPORT (ENHANCED)
# =============================================

class EnhancedSinteringVisualizer:
    """Enhanced visualizer for sintering temperature analysis"""
    
    def __init__(self, sintering_calculator=None):
        self.sintering_calculator = sintering_calculator or SinteringTemperatureCalculator()
    
    def create_comprehensive_sintering_dashboard(self, solutions, region_type='bulk'):
        """Create comprehensive dashboard for sintering temperature analysis"""
        
        # Analyze all solutions
        from original_analyzer import OriginalFileAnalyzer
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(solutions, region_type, 
                                                 stress_component='sigma_hydro', stress_type='max_abs')
        
        if not analyses:
            return None
        
        # Extract stresses and compute sintering temperatures
        stresses = []
        sintering_temps_exp = []
        sintering_temps_arr = []
        orientations = []
        systems = []
        defect_types = []
        
        for analysis in analyses:
            stress = analysis['region_stress']
            T_sinter_exp = self.sintering_calculator.compute_sintering_temperature_exponential(abs(stress))
            defect_type = analysis['params'].get('defect_type', 'Unknown')
            T_sinter_arr = self.sintering_calculator.compute_sintering_temperature_arrhenius_defect(abs(stress), defect_type)
            system_info = self.sintering_calculator.map_system_to_temperature(stress)
            
            stresses.append(abs(stress))
            sintering_temps_exp.append(T_sinter_exp)
            sintering_temps_arr.append(T_sinter_arr)
            orientations.append(analysis['theta_deg'])
            systems.append(system_info[0])
            defect_types.append(defect_type)
        
        # Create dashboard figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Sintering temperature vs stress
        ax1 = fig.add_subplot(2, 3, 1)
        scatter = ax1.scatter(stresses, sintering_temps_exp, c=orientations, 
                             cmap='hsv', s=50, alpha=0.7, edgecolors='black')
        
        # Add theoretical curve
        theory_data = self.sintering_calculator.get_theoretical_curve()
        ax1.plot(theory_data['stresses'], theory_data['T_exponential'], 
                'k--', alpha=0.5, label='Theoretical')
        
        ax1.set_xlabel('|σ_h| (GPa)', fontsize=10)
        ax1.set_ylabel('T_sinter (K)', fontsize=10)
        ax1.set_title('Sintering Temperature vs Hydrostatic Stress', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for orientation
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Orientation (°)', fontsize=9)
        
        # 2. Histogram of sintering temperatures
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(sintering_temps_exp, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(self.sintering_calculator.T0, color='green', linestyle='--', 
                   label=f'T₀ = {self.sintering_calculator.T0}K')
        ax2.axvline(self.sintering_calculator.T_min, color='red', linestyle='--',
                   label=f'T_min = {self.sintering_calculator.T_min}K')
        ax2.set_xlabel('Sintering Temperature (K)', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Distribution of Sintering Temperatures', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. System classification
        ax3 = fig.add_subplot(2, 3, 3)
        system_counts = {}
        for system in systems:
            system_counts[system] = system_counts.get(system, 0) + 1
        
        colors = {'System 1': 'green', 'System 2': 'orange', 'System 3': 'red'}
        bar_colors = [colors.get(sys.split()[0], 'gray') for sys in system_counts.keys()]
        
        ax3.bar(range(len(system_counts)), list(system_counts.values()), 
               color=bar_colors, edgecolor='black', alpha=0.7)
        ax3.set_xticks(range(len(system_counts)))
        ax3.set_xticklabels(list(system_counts.keys()), rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Number of Solutions', fontsize=10)
        ax3.set_title('System Classification Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Orientation vs sintering temperature
        ax4 = fig.add_subplot(2, 3, 4)
        scatter2 = ax4.scatter(orientations, sintering_temps_exp, c=stresses, 
                              cmap='plasma', s=50, alpha=0.7, edgecolors='black')
        ax4.axvline(54.7, color='green', linestyle='--', alpha=0.5, label='Habit Plane (54.7°)')
        ax4.set_xlabel('Orientation (°)', fontsize=10)
        ax4.set_ylabel('T_sinter (K)', fontsize=10)
        ax4.set_title('Sintering Temperature vs Orientation', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        cbar2 = plt.colorbar(scatter2, ax=ax4)
        cbar2.set_label('|σ_h| (GPa)', fontsize=9)
        
        # 5. Temperature reduction factor
        ax5 = fig.add_subplot(2, 3, 5)
        temp_reduction = [(self.sintering_calculator.T0 - T) / self.sintering_calculator.T0 * 100 
                         for T in sintering_temps_exp]
        ax5.scatter(stresses, temp_reduction, c='purple', s=50, alpha=0.7, edgecolors='black')
        ax5.set_xlabel('|σ_h| (GPa)', fontsize=10)
        ax5.set_ylabel('Temperature Reduction (%)', fontsize=10)
        ax5.set_title('Stress-Induced Temperature Reduction', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Defect type comparison
        ax6 = fig.add_subplot(2, 3, 6)
        defect_data = {}
        for defect in set(defect_types):
            defect_stresses = [s for s, d in zip(stresses, defect_types) if d == defect]
            if defect_stresses:
                defect_data[defect] = np.mean(defect_stresses)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        defect_names = list(defect_data.keys())
        defect_values = list(defect_data.values())
        
        ax6.bar(range(len(defect_data)), defect_values, color=colors[:len(defect_data)], edgecolor='black', alpha=0.7)
        ax6.set_xticks(range(len(defect_data)))
        ax6.set_xticklabels(defect_names, rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel('Average |σ_h| (GPa)', fontsize=10)
        ax6.set_title('Average Stress by Defect Type', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_defect_comparison_radar(self, solutions, region_type='bulk', angle_range=(0, 360), n_points=100):
        """Create radar chart comparing sintering temperatures for different defects"""
        
        defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Prepare data for each defect
        defect_data = {}
        
        for defect in defect_types:
            # Filter solutions by defect type
            defect_solutions = [s for s in solutions if s.get('params', {}).get('defect_type') == defect]
            
            if not defect_solutions:
                continue
            
            # Create orientation sweep for this defect
            from attention_interpolator import AttentionSpatialInterpolator
            interpolator = AttentionSpatialInterpolator()
            
            base_params = {
                'defect_type': defect,
                'shape': 'Square',
                'eps0': {'ISF': 0.71, 'ESF': 1.41, 'Twin': 2.12, 'No Defect': 0.0}.get(defect, 0.707),
                'kappa': 0.6,
                'theta': 0.0
            }
            
            sweep_result = interpolator.create_orientation_sweep(
                solutions, base_params, angle_range, n_points,
                region_type, 'sigma_hydro', 'max_abs'
            )
            
            if sweep_result:
                stresses = np.abs(np.array(sweep_result['stresses']))
                sintering_temps = self.sintering_calculator.compute_sintering_temperature_arrhenius_defect(stresses, defect)
                
                defect_data[defect] = {
                    'angles': sweep_result['angles'],
                    'stresses': stresses,
                    'sintering_temps': sintering_temps
                }
        
        # Create radar chart
        fig = go.Figure()
        
        for i, (defect, data) in enumerate(defect_data.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=data['sintering_temps'],
                theta=data['angles'],
                mode='lines+markers',
                name=defect,
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color),
                hovertemplate=f'Defect: {defect}<br>Orientation: %{{theta:.1f}}°<br>T_sinter: %{{r:.1f}} K<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Sintering Temperature Comparison by Defect Type',
                font=dict(size=18, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    title=dict(text='Sintering Temperature (K)', font=dict(size=14)),
                    gridcolor='rgba(100, 100, 100, 0.2)',
                    gridwidth=1
                ),
                angularaxis=dict(
                    gridcolor='rgba(100, 100, 100, 0.2)',
                    gridwidth=1,
                    rotation=90,
                    direction='clockwise'
                )
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            width=900,
            height=700
        )
        
        return fig

# =============================================
# ENHANCED COLOR MAPS (STREAMLINED)
# =============================================

class EnhancedColorMaps:
    """Streamlined colormap collection"""
    
    @staticmethod
    def get_all_colormaps():
        """Return all available colormaps categorized by type"""
        
        # Essential colormaps
        essential_maps = [
            # For stress visualization
            'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', 'seismic',
            # For temperature visualization
            'hot', 'afmhot', 'gist_heat', 'summer', 'autumn', 'winter', 'spring',
            # For defect classification
            'tab10', 'Set1', 'Set2', 'Set3',
            # Specialized for sintering
            'turbo', 'rainbow', 'hsv', 'twilight'
        ]
        
        # Custom enhanced maps
        custom_maps = {
            'stress_cmap': LinearSegmentedColormap.from_list(
                'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
            ),
            'sintering_temp': LinearSegmentedColormap.from_list(
                'sintering_temp', ['#8B0000', '#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#006400']
            ),
            'system_classification': ListedColormap(['#2E8B57', '#FF8C00', '#DC143C']),
            'defect_types': ListedColormap(['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        }
        
        # Combine all maps
        all_maps = essential_maps + list(custom_maps.keys())
        
        # Remove duplicates
        return list(dict.fromkeys(all_maps))
    
    @staticmethod
    def get_colormap(cmap_name):
        """Get a colormap by name with fallback"""
        custom_maps = {
            'stress_cmap': LinearSegmentedColormap.from_list(
                'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
            ),
            'sintering_temp': LinearSegmentedColormap.from_list(
                'sintering_temp', ['#8B0000', '#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#006400']
            ),
            'system_classification': ListedColormap(['#2E8B57', '#FF8C00', '#DC143C']),
            'defect_types': ListedColormap(['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        }
        
        if cmap_name in custom_maps:
            return custom_maps[cmap_name]
        
        try:
            return plt.colormaps.get_cmap(cmap_name)
        except AttributeError:
            return plt.cm.get_cmap(cmap_name)

# Initialize colormaps
COLORMAP_MANAGER = EnhancedColorMaps()
ALL_COLORMAPS = COLORMAP_MANAGER.get_all_colormaps()

# =============================================
# REGION ANALYSIS FUNCTIONS (MODIFIED)
# =============================================

def extract_region_stress(eta, stress_fields, region_type, stress_component='von_mises', stress_type='max_abs'):
    """Extract stress from specific regions (defect, interface, bulk)"""
    if eta is None or not isinstance(eta, np.ndarray):
        return 0.0
    
    # Create mask for the region
    if region_type == 'defect':
        mask = eta > 0.6
    elif region_type == 'interface':
        mask = (eta >= 0.4) & (eta <= 0.6)
    elif region_type == 'bulk':
        mask = eta < 0.4
    else:
        mask = np.ones_like(eta, dtype=bool)
    
    if not np.any(mask):
        return 0.0
    
    # Get stress data
    stress_data = np.zeros_like(eta)
    if stress_component == 'von_mises' and 'von_mises' in stress_fields:
        stress_data = stress_fields['von_mises']
    elif stress_component == 'sigma_hydro' and 'sigma_hydro' in stress_fields:
        stress_data = stress_fields['sigma_hydro']
    elif stress_component == 'sigma_mag' and 'sigma_mag' in stress_fields:
        stress_data = stress_fields['sigma_mag']
    elif stress_component in stress_fields:  # Generic case for any stress component
        stress_data = stress_fields[stress_component]
    
    # Extract region stress
    region_stress = stress_data[mask]
    
    if len(region_stress) == 0:
        return 0.0
    
    if stress_type == 'max_abs':
        return np.max(np.abs(region_stress))
    elif stress_type == 'mean_abs':
        return np.mean(np.abs(region_stress))
    elif stress_type == 'max':
        return np.max(region_stress)
    elif stress_type == 'min':
        return np.min(region_stress)
    elif stress_type == 'mean':
        return np.mean(region_stress)
    else:
        return np.mean(np.abs(region_stress))

def extract_all_stress_components(eta, stress_fields, region_type):
    """Extract all stress components (hydrostatic, von Mises, magnitude)"""
    if eta is None or not isinstance(eta, np.ndarray):
        return {}
    
    # Create mask for the region
    if region_type == 'defect':
        mask = eta > 0.6
    elif region_type == 'interface':
        mask = (eta >= 0.4) & (eta <= 0.6)
    elif region_type == 'bulk':
        mask = eta < 0.4
    else:
        mask = np.ones_like(eta, dtype=bool)
    
    if not np.any(mask):
        return {}
    
    results = {}
    
    # Extract each stress component
    for comp_name in ['von_mises', 'sigma_hydro', 'sigma_mag']:
        if comp_name in stress_fields:
            stress_data = stress_fields[comp_name][mask]
            if len(stress_data) > 0:
                results[comp_name] = {
                    'max_abs': float(np.max(np.abs(stress_data))),
                    'max': float(np.max(stress_data)),
                    'min': float(np.min(stress_data)),
                    'mean': float(np.mean(stress_data)),
                    'std': float(np.std(stress_data))
                }
    
    return results

# =============================================
# NUMBA-ACCELERATED FUNCTIONS
# =============================================

@jit(nopython=True, parallel=True)
def compute_gaussian_weights_numba(source_vectors, target_vector, sigma):
    """Numba-accelerated Gaussian weight computation"""
    n_sources = source_vectors.shape[0]
    weights = np.zeros(n_sources)
    
    for i in prange(n_sources):
        dist_sq = 0.0
        for j in range(source_vectors.shape[1]):
            diff = source_vectors[i, j] - target_vector[j]
            dist_sq += diff * diff
        weights[i] = np.exp(-0.5 * dist_sq / (sigma * sigma))
    
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(n_sources) / n_sources
    
    return weights

# =============================================
# ENHANCED NUMERICAL SOLUTIONS LOADER (STREAMLINED)
# =============================================

class EnhancedSolutionLoader:
    """Enhanced solution loader with support for multiple formats and caching"""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.pt_loading_method = "safe"
        
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
            if 'st' in globals():
                st.info(f"Created directory: {self.solutions_dir}")
    
    def scan_solutions(self) -> Dict[str, List[str]]:
        """Scan directory for solution files"""
        file_formats = {
            'pkl': [],
            'pt': [],
            'npz': []
        }
        
        for format_type, extensions in [
            ('pkl', ['*.pkl', '*.pickle']),
            ('pt', ['*.pt', '*.pth']),
            ('npz', ['*.npz'])
        ]:
            for ext in extensions:
                pattern = os.path.join(self.solutions_dir, ext)
                files = glob.glob(pattern)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_formats[format_type].extend(files)
        
        return file_formats
    
    def load_all_solutions(self, use_cache=True, pt_loading_method="safe"):
        """Load all solutions with caching, progress tracking, and error handling"""
        self.pt_loading_method = pt_loading_method
        solutions = []
        failed_files = []
        
        if not os.path.exists(self.solutions_dir):
            if 'st' in globals():
                st.warning(f"Directory {self.solutions_dir} not found. Creating it.")
            os.makedirs(self.solutions_dir, exist_ok=True)
            return solutions
        
        file_formats = self.scan_solutions()
        all_files = []
        for files in file_formats.values():
            all_files.extend(files)
        
        if not all_files:
            if 'st' in globals():
                st.info(f"No solution files found in {self.solutions_dir}")
            return solutions
        
        if 'st' in globals():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for idx, file_path in enumerate(all_files):
            try:
                filename = os.path.basename(file_path)
                
                cache_key = f"{filename}_{os.path.getmtime(file_path)}_{pt_loading_method}"
                if use_cache and cache_key in self.cache:
                    sim = self.cache[cache_key]
                    if sim.get('status') == 'success':
                        solutions.append(sim)
                    continue
                
                if 'st' in globals():
                    progress = (idx + 1) / len(all_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Loading {filename}...")
                
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                # Determine format
                if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                    format_type = 'pkl'
                elif file_path.endswith('.pt') or file_path.endswith('.pth'):
                    format_type = 'pt'
                elif file_path.endswith('.npz'):
                    format_type = 'npz'
                else:
                    continue
                
                # Read file
                if format_type == 'pkl':
                    data = pickle.loads(file_content)
                elif format_type == 'pt':
                    buffer = BytesIO(file_content)
                    data = torch.load(buffer, map_location='cpu', weights_only=True)
                elif format_type == 'npz':
                    buffer = BytesIO(file_content)
                    data = np.load(buffer, allow_pickle=True)
                    data = {key: data[key] for key in data.files}
                
                # Standardize data
                sim = {
                    'params': data.get('params', {}),
                    'history': data.get('history', []),
                    'metadata': data.get('metadata', {}),
                    'format': format_type,
                    'filename': filename,
                    'status': 'success'
                }
                
                if isinstance(sim['params'], dict) and 'history' in sim:
                    self.cache[cache_key] = sim
                    solutions.append(sim)
                else:
                    failed_files.append({
                        'filename': filename,
                        'error': 'Invalid data structure'
                    })
                    
            except Exception as e:
                failed_files.append({
                    'filename': os.path.basename(file_path),
                    'error': f"Loading error: {str(e)}"
                })
        
        if 'st' in globals():
            progress_bar.empty()
            status_text.empty()
        
        if failed_files and 'st' in globals():
            with st.expander(f"⚠️ Failed to load {len(failed_files)} files", expanded=False):
                for failed in failed_files[:10]:
                    st.error(f"**{failed['filename']}**: {failed['error']}")
                if len(failed_files) > 10:
                    st.info(f"... and {len(failed_files) - 10} more files failed to load.")
        
        return solutions

# =============================================
# ATTENTION-BASED INTERPOLATOR WITH PHYSICS CONSTRAINTS
# =============================================

class AttentionSpatialInterpolator:
    """Transformer-inspired attention interpolator with physics constraints"""
    
    def __init__(self, sigma=0.3, use_numba=True, attention_dim=32, num_heads=4):
        self.sigma = sigma
        self.use_numba = use_numba
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Parameter mappings with eigen strains
        self.defect_map = {
            'ISF': [1, 0, 0, 0],
            'ESF': [0, 1, 0, 0],
            'Twin': [0, 0, 1, 0],
            'No Defect': [0, 0, 0, 1],
            'Unknown': [0, 0, 0, 0]
        }
        
        self.eigen_strains = {
            'ISF': 0.71,
            'ESF': 1.41,
            'Twin': 2.12,
            'No Defect': 0.0,
            'Unknown': 0.0
        }
        
        # Initialize attention layers
        self.query_projection = nn.Linear(12, attention_dim)
        self.key_projection = nn.Linear(12, attention_dim)
        self.value_projection = nn.Linear(12, attention_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.output_projection = nn.Linear(attention_dim, 12)
    
    def compute_parameter_vector(self, params):
        """Convert parameters to numerical vector with 12 dimensions"""
        vector = []
        
        if not isinstance(params, dict):
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0.0], dtype=np.float32)
        
        # Defect type (4 dimensions)
        defect = params.get('defect_type', 'ISF')
        vector.extend(self.defect_map.get(defect, [0, 0, 0, 0]))
        
        # Shape (5 dimensions) - simplified for now
        shape = params.get('shape', 'Square')
        if shape == 'Square':
            vector.extend([1, 0, 0, 0, 0])
        elif shape == 'Horizontal Fault':
            vector.extend([0, 1, 0, 0, 0])
        elif shape == 'Vertical Fault':
            vector.extend([0, 0, 1, 0, 0])
        elif shape == 'Rectangle':
            vector.extend([0, 0, 0, 1, 0])
        elif shape == 'Ellipse':
            vector.extend([0, 0, 0, 0, 1])
        else:
            vector.extend([0, 0, 0, 0, 0])
        
        # Numeric parameters (3 dimensions)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        # Normalize parameters
        vector.append((eps0 - 0.3) / (3.0 - 0.3))  # eps0 normalized 0-1
        vector.append((kappa - 0.1) / (2.0 - 0.1))  # kappa normalized 0-1
        vector.append(theta / np.pi)  # theta normalized 0-1 (0 to π)
        
        return np.array(vector, dtype=np.float32)
    
    def compute_attention_weights(self, source_vectors, target_vector, use_spatial=True):
        """Compute attention weights using transformer-like attention with spatial regularization"""
        
        if len(source_vectors) == 0:
            return np.array([])
        
        # Convert to PyTorch tensors
        source_tensor = torch.FloatTensor(source_vectors).unsqueeze(0)  # (1, N, 12)
        target_tensor = torch.FloatTensor(target_vector).unsqueeze(0).unsqueeze(1)  # (1, 1, 12)
        
        # Project to attention space
        query = self.query_projection(target_tensor)  # (1, 1, attention_dim)
        keys = self.key_projection(source_tensor)     # (1, N, attention_dim)
        values = self.value_projection(source_tensor) # (1, N, attention_dim)
        
        # Multi-head attention
        attention_output, attention_weights = self.multihead_attention(
            query, keys, values
        )
        
        # Get attention weights (averaged over heads)
        attention_weights = attention_weights.squeeze().detach().numpy()
        
        # Apply spatial locality regularization
        if use_spatial and len(source_vectors) > 0:
            spatial_weights = self.compute_spatial_weights(source_vectors, target_vector)
            # Combine attention and spatial weights (70% attention, 30% spatial)
            combined_weights = 0.7 * attention_weights + 0.3 * spatial_weights
            # Normalize
            if np.sum(combined_weights) > 0:
                combined_weights = combined_weights / np.sum(combined_weights)
            else:
                combined_weights = np.ones_like(combined_weights) / len(combined_weights)
            return combined_weights
        
        return attention_weights
    
    def compute_spatial_weights(self, source_vectors, target_vector):
        """Compute spatial locality weights using Euclidean distance"""
        if len(source_vectors) == 0:
            return np.array([])
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
        
        # Apply Gaussian kernel
        weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
    
    def interpolate_all_stress_components(self, sources, target_angle_deg, target_params,
                                         region_type='bulk', stress_type='max_abs'):
        """Interpolate all stress components at a precise orientation angle"""
        
        # Convert angle to radians
        target_angle_rad = np.deg2rad(target_angle_deg)
        
        # Update target params with precise angle
        precise_target_params = target_params.copy()
        precise_target_params['theta'] = target_angle_rad
        
        # Compute target vector
        target_vector = self.compute_parameter_vector(precise_target_params)
        
        # Filter and validate sources
        valid_sources = []
        source_vectors = []
        source_stresses = {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []}
        
        for src in sources:
            if not isinstance(src, dict):
                continue
            if 'params' not in src or 'history' not in src:
                continue
            if not isinstance(src.get('params'), dict):
                continue
            
            valid_sources.append(src)
            
            # Get source parameter vector
            src_vector = self.compute_parameter_vector(src['params'])
            source_vectors.append(src_vector)
            
            # Extract all stress components from last frame
            history = src.get('history', [])
            if history:
                last_frame = history[-1]
                if isinstance(last_frame, tuple) and len(last_frame) >= 2:
                    eta, stress_fields = last_frame[0], last_frame[1]
                elif isinstance(last_frame, dict):
                    eta = last_frame.get('eta', np.zeros((128, 128)))
                    stress_fields = last_frame.get('stresses', {})
                else:
                    eta = np.zeros((128, 128))
                    stress_fields = {}
                
                # Extract all stress components
                stress_components = extract_all_stress_components(eta, stress_fields, region_type)
                for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                    if comp in stress_components:
                        source_stresses[comp].append(stress_components[comp][stress_type])
                    else:
                        source_stresses[comp].append(0.0)
            else:
                for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                    source_stresses[comp].append(0.0)
        
        if not valid_sources:
            return None
        
        source_vectors = np.array(source_vectors)
        
        # Compute attention weights
        if len(source_vectors) > 0:
            attention_weights = self.compute_attention_weights(
                source_vectors, target_vector, use_spatial=True
            )
            
            if len(attention_weights) == 0:
                attention_weights = np.ones(len(source_vectors)) / len(source_vectors)
            
            # Weighted combination for each stress component
            interpolated_stresses = {}
            for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                stresses = np.array(source_stresses[comp])
                weighted_stress = np.sum(attention_weights * stresses)
                interpolated_stresses[comp] = float(weighted_stress)
            
            return {
                'interpolated_stresses': interpolated_stresses,
                'attention_weights': attention_weights.tolist(),
                'target_params': precise_target_params,
                'target_angle_deg': float(target_angle_deg),
                'target_angle_rad': float(target_angle_rad),
                'region_type': region_type,
                'stress_type': stress_type,
                'num_sources': len(valid_sources)
            }
        
        return None

# =============================================
# ORIGINAL FILE ANALYZER WITH ORIENTATION SUPPORT (MODIFIED)
# =============================================

class OriginalFileAnalyzer:
    """Analyze original loaded files for different regions with orientation support"""
    
    def __init__(self):
        self.region_definitions = {
            'defect': {'min': 0.6, 'max': 1.0, 'name': 'Defect Region (η > 0.6)'},
            'interface': {'min': 0.4, 'max': 0.6, 'name': 'Interface Region (0.4 ≤ η ≤ 0.6)'},
            'bulk': {'min': 0.0, 'max': 0.4, 'name': 'Bulk Ag Material (η < 0.4)'}
        }
    
    def analyze_solution(self, solution, region_type='bulk', 
                        stress_component='von_mises', stress_type='max_abs'):
        """Analyze a single solution for a specific region"""
        if not solution or 'history' not in solution:
            return None
        
        history = solution.get('history', [])
        if not history:
            return None
        
        # Get the last frame
        last_frame = history[-1]
        
        # Extract eta and stress fields
        if isinstance(last_frame, tuple) and len(last_frame) >= 2:
            eta, stress_fields = last_frame[0], last_frame[1]
        elif isinstance(last_frame, dict):
            eta = last_frame.get('eta', np.zeros((128, 128)))
            stress_fields = last_frame.get('stresses', {})
        else:
            return None
        
        # Extract region stress for the specified component
        region_stress = extract_region_stress(eta, stress_fields, region_type, 
                                             stress_component, stress_type)
        
        # Extract all stress components
        all_stresses = extract_all_stress_components(eta, stress_fields, region_type)
        
        # Get solution parameters
        params = solution.get('params', {})
        theta = params.get('theta', 0.0)
        theta_deg = np.rad2deg(theta) if theta is not None else 0.0
        
        return {
            'region_stress': float(region_stress),
            'all_stresses': all_stresses,
            'params': params,
            'theta_rad': float(theta) if theta is not None else 0.0,
            'theta_deg': float(theta_deg),
            'filename': solution.get('filename', 'Unknown'),
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type
        }

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER FOR VICINITY
# =============================================

class VicinityVisualizer:
    """Specialized visualizer for habit plane vicinity analysis"""
    
    def __init__(self, habit_angle=54.7, vicinity_range=10.0):
        self.habit_angle = habit_angle
        self.vicinity_range = vicinity_range
    
    def create_vicinity_sunburst(self, angles, stresses, sintering_temps=None, 
                                 title="Habit Plane Vicinity Analysis"):
        """Create sunburst chart focused on habit plane vicinity"""
        
        # Filter data in vicinity
        mask = (angles >= self.habit_angle - self.vicinity_range) & \
               (angles <= self.habit_angle + self.vicinity_range)
        vic_angles = angles[mask]
        vic_stresses = stresses[mask]
        
        if len(vic_angles) == 0:
            return None
        
        fig = go.Figure()
        
        # Create polar plot
        if sintering_temps is not None:
            vic_temps = sintering_temps[mask]
            color_data = vic_temps
            colorbar_title = "Sintering Temp (K)"
            colorscale = 'thermal'
        else:
            color_data = vic_stresses
            colorbar_title = "Stress (GPa)"
            colorscale = 'RdBu'
        
        fig.add_trace(go.Scatterpolar(
            r=vic_stresses,
            theta=vic_angles,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=color_data,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            line=dict(color='rgba(100, 100, 100, 0.3)'),
            name='Stress Distribution',
            hovertemplate='Orientation: %{theta:.1f}°<br>Stress: %{r:.3f} GPa<br>Temperature: %{marker.color:.1f} K<extra></extra>' if sintering_temps is not None else 'Orientation: %{theta:.1f}°<br>Stress: %{r:.3f} GPa<extra></extra>'
        ))
        
        # Highlight habit plane
        habit_idx = np.argmin(np.abs(vic_angles - self.habit_angle))
        if habit_idx < len(vic_stresses):
            fig.add_trace(go.Scatterpolar(
                r=[vic_stresses[habit_idx]],
                theta=[vic_angles[habit_idx]],
                mode='markers',
                marker=dict(
                    size=20,
                    color='green',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate=f'Habit Plane ({self.habit_angle}°)<br>Stress: {vic_stresses[habit_idx]:.3f} GPa<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, title=dict(text='Stress (GPa)')),
                angularaxis=dict(
                    rotation=90,
                    direction='clockwise',
                    tickvals=np.arange(self.habit_angle - self.vicinity_range, 
                                      self.habit_angle + self.vicinity_range + 1, 5),
                    ticktext=[f'{x:.1f}°' for x in np.arange(self.habit_angle - self.vicinity_range, 
                                                            self.habit_angle + self.vicinity_range + 1, 5)]
                ),
                sector=[self.habit_angle - self.vicinity_range, self.habit_angle + self.vicinity_range]
            ),
            title=dict(
                text=title,
                font=dict(size=16, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            showlegend=True,
            width=700,
            height=600
        )
        
        return fig

# =============================================
# MAIN APPLICATION WITH VICINITY FOCUS
# =============================================

def main():
    st.set_page_config(
        page_title="Ag FCC Twin: Vicinity Analysis & Sintering Temperature Prediction",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }
    .habit-plane-highlight {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Ag FCC Twin: Vicinity Analysis & Sintering Temperature Prediction</h1>', unsafe_allow_html=True)
    
    # Habit plane information
    st.markdown("""
    <div class="habit-plane-highlight">
    <h3>🎯 Focus: Ag FCC Twin Habit Plane (54.7°)</h3>
    <p>This analysis focuses on the vicinity of the Ag FCC twin habit plane at 54.7°, where maximum stress concentrations occur due to eigen strain mismatches.</p>
    <p><strong>Key Parameters:</strong> ISF (ε*=0.71), ESF (ε*=1.41), Twin (ε*=2.12), No Defect (ε*=0.0)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = AttentionSpatialInterpolator()
    if 'sintering_calculator' not in st.session_state:
        st.session_state.sintering_calculator = SinteringTemperatureCalculator()
    if 'sintering_visualizer' not in st.session_state:
        st.session_state.sintering_visualizer = EnhancedSinteringVisualizer(
            st.session_state.sintering_calculator
        )
    if 'vicinity_visualizer' not in st.session_state:
        st.session_state.vicinity_visualizer = VicinityVisualizer(habit_angle=54.7, vicinity_range=10.0)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")
        
        # Analysis mode
        analysis_mode = st.radio(
            "Select analysis mode:",
            ["Vicinity Analysis (54.7° ± 10°)", "Defect Comparison", "Single Point Analysis"],
            index=0
        )
        
        # Region selection
        region_type = st.selectbox(
            "Analysis Region:",
            ["Bulk Ag Material (η < 0.4)", "Interface Region (0.4 ≤ η ≤ 0.6)", "Defect Region (η > 0.6)"],
            index=0
        )
        
        # Map region name to key
        region_map = {
            "Bulk Ag Material (η < 0.4)": "bulk",
            "Interface Region (0.4 ≤ η ≤ 0.6)": "interface",
            "Defect Region (η > 0.6)": "defect"
        }
        region_key = region_map[region_type]
        
        # Defect type with eigen strain auto-set
        defect_type = st.selectbox(
            "Defect Type:",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select defect type. Eigen strain will be set automatically."
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        
        # Other parameters
        shape = st.selectbox("Shape:", ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"], index=0)
        eps0 = st.slider("ε* (eigen strain):", 0.0, 3.0, default_eps0, 0.01)
        kappa = st.slider("κ (interface energy):", 0.1, 2.0, 0.6, 0.01)
        
        # Load solutions button
        if st.button("📂 Load Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions found. Check the numerical_solutions directory.")
    
    # Main content
    if not st.session_state.solutions:
        st.warning("Please load solutions first using the button in the sidebar.")
        return
    
    # Target parameters
    target_params = {
        'defect_type': defect_type,
        'shape': shape,
        'eps0': eps0,
        'kappa': kappa,
        'theta': 0.0
    }
    
    # Analysis execution
    if analysis_mode == "Vicinity Analysis (54.7° ± 10°)":
        st.markdown("## 🌐 Vicinity Analysis (54.7° ± 10°)")
        
        # Create vicinity sweep
        min_angle = 44.7
        max_angle = 64.7
        n_points = 100
        
        with st.spinner("Performing vicinity analysis..."):
            # Create orientation sweep
            interpolator = st.session_state.interpolator
            sweep_result = interpolator.create_orientation_sweep(
                st.session_state.solutions,
                target_params,
                (min_angle, max_angle),
                n_points,
                region_key,
                'sigma_hydro',
                'max_abs'
            )
            
            if sweep_result:
                # Convert to numpy arrays
                angles = np.array(sweep_result['angles'])
                stresses = np.array(sweep_result['stresses'])
                
                # Compute sintering temperatures
                sintering_temps = st.session_state.sintering_calculator.compute_sintering_temperature_arrhenius_defect(
                    np.abs(stresses), defect_type
                )
                
                # Create vicinity sunburst
                fig = st.session_state.vicinity_visualizer.create_vicinity_sunburst(
                    angles, stresses, sintering_temps,
                    title=f"Vicinity Analysis: {defect_type} in {region_key}"
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display stress at habit plane
                habit_idx = np.argmin(np.abs(angles - 54.7))
                habit_stress = stresses[habit_idx]
                habit_temp = sintering_temps[habit_idx]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Habit Plane Stress (54.7°)", f"{habit_stress:.3f} GPa")
                with col2:
                    st.metric("Sintering Temperature", f"{habit_temp:.1f} K")
                with col3:
                    st.metric("Temperature Reduction", f"{623 - habit_temp:.1f} K")
                
                # Create line plot
                fig_line, ax = plt.subplots(figsize=(10, 6))
                ax.plot(angles, stresses, 'b-', linewidth=2, label='Hydrostatic Stress')
                ax.set_xlabel('Orientation (°)', fontsize=12)
                ax.set_ylabel('Stress (GPa)', fontsize=12, color='blue')
                ax.tick_params(axis='y', labelcolor='blue')
                ax.grid(True, alpha=0.3)
                
                ax2 = ax.twinx()
                ax2.plot(angles, sintering_temps, 'r-', linewidth=2, label='Sintering Temperature')
                ax2.set_ylabel('Temperature (K)', fontsize=12, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                ax.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane')
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                
                st.pyplot(fig_line)
                plt.close(fig_line)
    
    elif analysis_mode == "Defect Comparison":
        st.markdown("## 📊 Defect Comparison")
        
        # Create defect comparison radar
        with st.spinner("Creating defect comparison..."):
            fig = st.session_state.sintering_visualizer.create_defect_comparison_radar(
                st.session_state.solutions,
                region_key,
                angle_range=(0, 360),
                n_points=100
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show eigen strain values
            st.markdown("### Eigen Strain Values")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ISF", "0.71")
            with col2:
                st.metric("ESF", "1.41")
            with col3:
                st.metric("Twin", "2.12")
            with col4:
                st.metric("No Defect", "0.00")
    
    elif analysis_mode == "Single Point Analysis":
        st.markdown("## 🎯 Single Point Analysis")
        
        target_angle = st.number_input("Target Orientation (°):", value=54.7, min_value=0.0, max_value=360.0)
        
        if st.button("Analyze", type="primary"):
            with st.spinner("Performing interpolation..."):
                # Interpolate all stress components
                result = st.session_state.interpolator.interpolate_all_stress_components(
                    st.session_state.solutions,
                    target_angle,
                    target_params,
                    region_key,
                    'max_abs'
                )
                
                if result:
                    st.success("Analysis complete!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Hydrostatic Stress", 
                                 f"{result['interpolated_stresses']['sigma_hydro']:.3f} GPa")
                    with col2:
                        st.metric("Von Mises Stress", 
                                 f"{result['interpolated_stresses']['von_mises']:.3f} GPa")
                    with col3:
                        st.metric("Stress Magnitude", 
                                 f"{result['interpolated_stresses']['sigma_mag']:.3f} GPa")
                    
                    # Compute sintering temperatures
                    sigma_h = result['interpolated_stresses']['sigma_hydro']
                    T_exp = st.session_state.sintering_calculator.compute_sintering_temperature_exponential(np.abs(sigma_h))
                    T_arr = st.session_state.sintering_calculator.compute_sintering_temperature_arrhenius_defect(np.abs(sigma_h), defect_type)
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("Exponential Model", f"{T_exp:.1f} K")
                    with col5:
                        st.metric("Arrhenius Model", f"{T_arr:.1f} K")
                    
                    # Show Arrhenius equation
                    st.markdown("### Arrhenius Equation")
                    st.latex(r"T_{\text{sinter}} = \frac{Q_a - \Omega|\sigma_h|}{k_B \ln(D_0/D_{\text{crit}})}")
                    
                    st.write(f"**Q_eff = {st.session_state.sintering_calculator.Q_a:.2f} eV - "
                            f"{st.session_state.sintering_calculator.omega_eV_per_GPa:.4f} eV/GPa × {np.abs(sigma_h):.2f} GPa**")

if __name__ == "__main__":
    main()
