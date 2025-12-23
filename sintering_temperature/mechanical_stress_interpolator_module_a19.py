import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from io import BytesIO
import warnings
import json
import zipfile
from numba import jit, prange
import time
import hashlib
import sqlite3
from pathlib import Path
import tempfile
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import base64
import seaborn as sns
from scipy import ndimage
import cmasher as cmr
from scipy.spatial.distance import cdist, euclidean
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import plotly.express as px
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# =============================================
# SINTERING TEMPERATURE CALCULATOR (NEW)
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
        
    def compute_sintering_temperature_exponential(self, sigma_h):
        """Compute sintering temperature using exponential empirical model"""
        sigma_abs = np.abs(sigma_h)
        T_sinter = self.T0 * np.exp(-self.beta * sigma_abs / self.G)
        return T_sinter
    
    def compute_sintering_temperature_arrhenius(self, sigma_h, D0=1e-6, D_crit=1e-10):
        """Compute sintering temperature using stress-modified Arrhenius equation"""
        sigma_abs = np.abs(sigma_h)
        Q_eff = self.Q_a - self.omega_eV_per_GPa * sigma_abs
        T_sinter = Q_eff / (self.kB * np.log(D0 / D_crit))
        return T_sinter
    
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
# ENHANCED VISUALIZATION WITH SINTERING SUPPORT (NEW)
# =============================================

class EnhancedSinteringVisualizer:
    """Enhanced visualizer for sintering temperature analysis"""
    
    def __init__(self, sintering_calculator=None):
        self.sintering_calculator = sintering_calculator or SinteringTemperatureCalculator()
    
    def create_comprehensive_sintering_dashboard(self, solutions, region_type='bulk',
                                                stress_component='sigma_hydro',
                                                stress_type='max_abs'):
        """Create comprehensive dashboard for sintering temperature analysis"""
        
        # Analyze all solutions
        from original_analyzer import OriginalFileAnalyzer
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(solutions, region_type, 
                                                 stress_component, stress_type)
        
        if not analyses:
            return None
        
        # Extract stresses and compute sintering temperatures
        stresses = []
        sintering_temps = []
        orientations = []
        systems = []
        
        for analysis in analyses:
            stress = analysis['region_stress']
            T_sinter = self.sintering_calculator.compute_sintering_temperature_exponential(abs(stress))
            system_info = self.sintering_calculator.map_system_to_temperature(stress)
            
            stresses.append(abs(stress))
            sintering_temps.append(T_sinter)
            orientations.append(analysis['theta_deg'])
            systems.append(system_info[0])
        
        # Create dashboard figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Sintering temperature vs stress
        ax1 = fig.add_subplot(2, 3, 1)
        scatter = ax1.scatter(stresses, sintering_temps, c=orientations, 
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
        ax2.hist(sintering_temps, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
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
        scatter2 = ax4.scatter(orientations, sintering_temps, c=stresses, 
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
                         for T in sintering_temps]
        ax5.scatter(stresses, temp_reduction, c='purple', s=50, alpha=0.7, edgecolors='black')
        ax5.set_xlabel('|σ_h| (GPa)', fontsize=10)
        ax5.set_ylabel('Temperature Reduction (%)', fontsize=10)
        ax5.set_title('Stress-Induced Temperature Reduction', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics table
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        stats_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Number of Solutions', f'{len(analyses)}', ''],
            ['Mean |σ_h|', f'{np.mean(stresses):.2f}', 'GPa'],
            ['Max |σ_h|', f'{np.max(stresses):.2f}', 'GPa'],
            ['Mean T_sinter', f'{np.mean(sintering_temps):.1f}', 'K'],
            ['Min T_sinter', f'{np.min(sintering_temps):.1f}', 'K'],
            ['Max T_sinter', f'{np.max(sintering_temps):.1f}', 'K'],
            ['T₀ (reference)', f'{self.sintering_calculator.T0}', 'K'],
            ['T_min (peak)', f'{self.sintering_calculator.T_min}', 'K'],
            ['Temperature Range', f'{np.min(sintering_temps):.0f}-{np.max(sintering_temps):.0f}', 'K'],
            ['Mean Reduction', f'{np.mean(temp_reduction):.1f}', '%']
        ]
        
        table = ax6.table(cellText=stats_data, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style table
        for i in range(len(stats_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(color='white')
                elif i % 2 == 1:
                    cell.set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_sintering_plot(self, solutions, region_type='bulk',
                                         stress_component='sigma_hydro',
                                         stress_type='max_abs'):
        """Create interactive Plotly visualization for sintering analysis"""
        
        from original_analyzer import OriginalFileAnalyzer
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(solutions, region_type,
                                                 stress_component, stress_type)
        
        if not analyses:
            return None
        
        # Prepare data
        stresses = []
        sintering_temps = []
        orientations = []
        filenames = []
        systems = []
        colors = []
        
        for analysis in analyses:
            stress = abs(analysis['region_stress'])
            T_sinter = self.sintering_calculator.compute_sintering_temperature_exponential(stress)
            system_info = self.sintering_calculator.map_system_to_temperature(stress)
            
            stresses.append(stress)
            sintering_temps.append(T_sinter)
            orientations.append(analysis['theta_deg'])
            filenames.append(analysis['filename'])
            systems.append(system_info[0])
            
            # Assign colors based on system
            if 'System 1' in system_info[0]:
                colors.append('green')
            elif 'System 2' in system_info[0]:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=stresses,
            y=sintering_temps,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=[f"File: {f}<br>Orientation: {o:.1f}°<br>System: {s}<br>Stress: {σ:.2f} GPa<br>T_sinter: {T:.1f} K ({T-273.15:.0f}°C)"
                  for f, o, s, σ, T in zip(filenames, orientations, systems, stresses, sintering_temps)],
            hoverinfo='text',
            name='Solutions'
        ))
        
        # Add theoretical curve
        theory_data = self.sintering_calculator.get_theoretical_curve()
        fig.add_trace(go.Scatter(
            x=theory_data['stresses'],
            y=theory_data['T_exponential'],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Theoretical Curve'
        ))
        
        # Add system boundaries
        fig.add_vrect(x0=0, x1=5, fillcolor="green", opacity=0.1, line_width=0,
                     annotation_text="System 1", annotation_position="top left")
        fig.add_vrect(x0=5, x1=20, fillcolor="orange", opacity=0.1, line_width=0,
                     annotation_text="System 2", annotation_position="top left")
        fig.add_vrect(x0=20, x1=35, fillcolor="red", opacity=0.1, line_width=0,
                     annotation_text="System 3", annotation_position="top left")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Sintering Temperature Analysis - {region_type}",
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text='Absolute Hydrostatic Stress |σ_h| (GPa)', font=dict(size=14)),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1
            ),
            yaxis=dict(
                title=dict(text='Sintering Temperature (K)', font=dict(size=14)),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1000,
            height=600
        )
        
        # Add second y-axis for Celsius
        fig.update_layout(
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying="y",
                side="right",
                tickmode='sync',
                tickvals=fig.data[0].y,
                ticktext=[f"{t-273.15:.0f}" for t in fig.data[0].y],
                range=[min(sintering_temps)-273.15, max(sintering_temps)-273.15]
            )
        )
        
        return fig
    
    def create_sintering_temperature_sweep(self, solutions, base_params, angle_range,
                                          region_type='bulk', stress_component='sigma_hydro',
                                          stress_type='max_abs', n_points=100):
        """Create sintering temperature sweep across orientation range"""
        
        from attention_interpolator import AttentionSpatialInterpolator
        interpolator = AttentionSpatialInterpolator()
        
        # Get stress sweep
        sweep_result = interpolator.create_orientation_sweep(
            solutions, base_params, angle_range, n_points,
            region_type, stress_component, stress_type
        )
        
        if not sweep_result:
            return None
        
        # Compute sintering temperatures
        stresses = np.array(sweep_result['stresses'])
        sintering_temps = self.sintering_calculator.compute_sintering_temperature_exponential(np.abs(stresses))
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot stress
        ax1.plot(sweep_result['angles'], stresses, 'b-', linewidth=3, label='Hydrostatic Stress')
        ax1.set_xlabel('Orientation (°)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Hydrostatic Stress (GPa)', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # Plot sintering temperature on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(sweep_result['angles'], sintering_temps, 'r-', linewidth=3, label='Sintering Temperature')
        ax2.set_ylabel('Sintering Temperature (K)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Highlight habit plane
        ax1.axvline(54.7, color='green', linestyle='--', linewidth=2, 
                   label='Habit Plane (54.7°)', alpha=0.7)
        
        # Add Celsius scale
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(sweep_result['angles'], [t-273.15 for t in sintering_temps], 
                'r--', alpha=0.5, linewidth=2, label='Sintering Temp (°C)')
        ax3.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold', color='darkred')
        ax3.tick_params(axis='y', labelcolor='darkred')
        
        # Title and legend
        ax1.set_title(f'Sintering Temperature Sweep: {region_type}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                  loc='upper left', fontsize=10)
        
        # Add statistics box
        stats_text = f"""Statistics:
        Min Stress: {np.min(stresses):.2f} GPa
        Max Stress: {np.max(stresses):.2f} GPa
        Min T_sinter: {np.min(sintering_temps):.1f} K ({np.min(sintering_temps)-273.15:.0f}°C)
        Max T_sinter: {np.max(sintering_temps):.1f} K ({np.max(sintering_temps)-273.15:.0f}°C)
        Habit Plane T: {sintering_temps[np.argmin(np.abs(np.array(sweep_result['angles'])-54.7))]:.1f} K
        """
        
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED COLOR MAPS (50+ COLORMAPS)
# =============================================

class EnhancedColorMaps:
    """Enhanced colormap collection with 50+ options"""
    
    @staticmethod
    def get_all_colormaps():
        """Return all available colormaps categorized by type"""
        
        # Standard matplotlib colormaps
        standard_maps = [
            # Sequential
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'summer', 'autumn', 'winter', 'spring', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'bone', 'gray', 'pink', 'binary',
            
            # Diverging
            'coolwarm', 'bwr', 'seismic', 'RdBu', 'RdYlBu',
            'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr',
            'Spectral',
            
            # Cyclic
            'twilight', 'twilight_shifted', 'hsv',
            
            # Qualitative
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'Set1', 'Set2', 'Set3',
            'Pastel1', 'Pastel2',
            'Dark2', 'Paired',
            'Accent',
            
            # Miscellaneous
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
            'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
            'cubehelix', 'brg', 'gist_rainbow', 'rainbow',
            'jet', 'nipy_spectral', 'gist_ncar'
        ]
        
        # Custom enhanced maps
        custom_maps = {
            'stress_cmap': LinearSegmentedColormap.from_list(
                'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
            ),
            'turbo': 'turbo',
            'deep': 'viridis',
            'dense': 'plasma',
            'matter': 'inferno',
            'speed': 'magma',
            'amp': 'cividis',
            'tempo': 'twilight',
            'phase': 'twilight_shifted',
            'balance': 'RdBu_r',
            'delta': 'coolwarm',
            'curl': 'PuOr_r',
            'diff': 'seismic',
            'tarn': 'terrain',
            'topo': 'gist_earth',
            'oxy': 'ocean',
            'deep_r': 'viridis_r',
            'dense_r': 'plasma_r',
            'ice': 'Blues',
            'fire': 'Reds',
            'earth': 'YlOrBr',
            'water': 'PuBu',
            'forest': 'Greens',
            'sunset': 'YlOrRd',
            'dawn': 'Purples',
            'night': 'Blues_r',
            'aurora': 'gist_ncar',
            'spectrum': 'Spectral',
            'prism_enhanced': 'prism',
            'pastel_rainbow': ListedColormap(plt.cm.rainbow(np.linspace(0, 1, 256)) * 0.7 + 0.3),
            'high_contrast': LinearSegmentedColormap.from_list(
                'high_contrast', ['#000000', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#FFFFFF']
            ),
            # Sintering temperature specific colormaps
            'sintering_temp': LinearSegmentedColormap.from_list(
                'sintering_temp', ['#8B0000', '#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#006400']
            ),
            'system_classification': ListedColormap(['#2E8B57', '#FF8C00', '#DC143C']),
            'temperature_gradient': LinearSegmentedColormap.from_list(
                'temperature_gradient', ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
            )
        }
        
        # Combine all maps
        all_maps = standard_maps + list(custom_maps.keys())
        
        # Remove duplicates
        return list(dict.fromkeys(all_maps))
    
    @staticmethod
    def get_colormap(cmap_name):
        """Get a colormap by name with fallback"""
        try:
            if cmap_name == 'stress_cmap':
                return LinearSegmentedColormap.from_list(
                    'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
                )
            elif cmap_name == 'pastel_rainbow':
                return ListedColormap(plt.cm.rainbow(np.linspace(0, 1, 256)) * 0.7 + 0.3)
            elif cmap_name == 'high_contrast':
                return LinearSegmentedColormap.from_list(
                    'high_contrast', ['#000000', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#FFFFFF']
                )
            elif cmap_name == 'sintering_temp':
                return LinearSegmentedColormap.from_list(
                    'sintering_temp', ['#8B0000', '#FF0000', '#FFA500', '#FFFF00', '#90EE90', '#006400']
                )
            elif cmap_name == 'system_classification':
                return ListedColormap(['#2E8B57', '#FF8C00', '#DC143C'])
            elif cmap_name == 'temperature_gradient':
                return LinearSegmentedColormap.from_list(
                    'temperature_gradient', ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
                )
            else:
                # Use the new API for matplotlib >= 3.6
                try:
                    return plt.colormaps.get_cmap(cmap_name)
                except AttributeError:
                    return plt.cm.get_cmap(cmap_name)
        except:
            # Fallback to viridis
            return plt.cm.viridis

# Initialize colormaps
COLORMAP_MANAGER = EnhancedColorMaps()
ALL_COLORMAPS = COLORMAP_MANAGER.get_all_colormaps()

# =============================================
# REGION ANALYSIS FUNCTIONS
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

def extract_region_statistics(eta, stress_fields, region_type):
    """Extract comprehensive statistics for a region"""
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
        return {
            'area_fraction': 0.0,
            'von_mises': {'max': 0.0, 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'mean_abs': 0.0},
            'sigma_hydro': {'max': 0.0, 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'mean_abs': 0.0},
            'sigma_mag': {'max': 0.0, 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'mean_abs': 0.0}
        }
    
    area_fraction = np.sum(mask) / mask.size
    
    results = {'area_fraction': float(area_fraction)}
    
    # Analyze each stress component
    for comp_name in ['von_mises', 'sigma_hydro', 'sigma_mag']:
        if comp_name in stress_fields:
            stress_data = stress_fields[comp_name][mask]
            if len(stress_data) > 0:
                results[comp_name] = {
                    'max': float(np.max(stress_data)),
                    'min': float(np.min(stress_data)),
                    'mean': float(np.mean(stress_data)),
                    'std': float(np.std(stress_data)),
                    'max_abs': float(np.max(np.abs(stress_data))),
                    'mean_abs': float(np.mean(np.abs(stress_data))),
                    'percentile_95': float(np.percentile(np.abs(stress_data), 95)),
                    'percentile_99': float(np.percentile(np.abs(stress_data), 99))
                }
            else:
                results[comp_name] = {
                    'max': 0.0, 'min': 0.0, 'mean': 0.0, 'std': 0.0,
                    'max_abs': 0.0, 'mean_abs': 0.0,
                    'percentile_95': 0.0, 'percentile_99': 0.0
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

@jit(nopython=True)
def compute_stress_statistics_numba(stress_matrix):
    """Compute stress statistics efficiently"""
    flat_stress = stress_matrix.flatten()
    
    max_val = np.max(flat_stress)
    min_val = np.min(flat_stress)
    mean_val = np.mean(flat_stress)
    std_val = np.std(flat_stress)
    percentile_95 = np.percentile(flat_stress, 95)
    percentile_99 = np.percentile(flat_stress, 99)
    
    return max_val, min_val, mean_val, std_val, percentile_95, percentile_99

# =============================================
# ENHANCED NUMERICAL SOLUTIONS LOADER
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
            'h5': [],
            'npz': [],
            'sql': [],
            'json': []
        }
        
        for format_type, extensions in [
            ('pkl', ['*.pkl', '*.pickle']),
            ('pt', ['*.pt', '*.pth']),
            ('h5', ['*.h5', '*.hdf5']),
            ('npz', ['*.npz']),
            ('sql', ['*.sql', '*.db']),
            ('json', ['*.json'])
        ]:
            for ext in extensions:
                pattern = os.path.join(self.solutions_dir, ext)
                files = glob.glob(pattern)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_formats[format_type].extend(files)
        
        return file_formats
    
    def get_all_files_info(self) -> List[Dict[str, Any]]:
        """Get information about all solution files"""
        all_files = []
        file_formats = self.scan_solutions()
        
        for format_type, files in file_formats.items():
            for file_path in files:
                try:
                    file_info = {
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'format': format_type,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        'relative_path': os.path.relpath(file_path, self.solutions_dir)
                    }
                    all_files.append(file_info)
                except Exception as e:
                    if 'st' in globals():
                        st.warning(f"Could not get info for {file_path}: {e}")
        
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
    
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        try:
            data = pickle.load(buffer)
            if isinstance(data, Exception):
                return data
            return data
        except Exception as e:
            return e
    
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        try:
            if self.pt_loading_method == "safe":
                try:
                    import numpy as np
                    try:
                        from numpy._core.multiarray import scalar as np_scalar
                    except ImportError:
                        try:
                            from numpy.core.multiarray import scalar as np_scalar
                        except ImportError:
                            np_scalar = None
                    
                    if np_scalar is not None:
                        import torch.serialization
                        with torch.serialization.safe_globals([np_scalar]):
                            data = torch.load(buffer, map_location='cpu', weights_only=True)
                    else:
                        if 'st' in globals():
                            st.warning("Could not import numpy scalar, using weights_only=False")
                        data = torch.load(buffer, map_location='cpu', weights_only=False)
                except Exception as safe_error:
                    if 'st' in globals():
                        st.warning(f"Safe loading failed: {safe_error}. Trying weights_only=False")
                    data = torch.load(buffer, map_location='cpu', weights_only=False)
            else:
                data = torch.load(buffer, map_location='cpu', weights_only=False)
            
            if isinstance(data, dict):
                for key in list(data.keys()):
                    if torch.is_tensor(data[key]):
                        data[key] = data[key].cpu().numpy()
                    elif isinstance(data[key], dict):
                        for subkey in list(data[key].keys()):
                            if torch.is_tensor(data[key][subkey]):
                                data[key][subkey] = data[key][subkey].cpu().numpy()
            return data
        except Exception as e:
            return e
    
    def _read_h5(self, file_content):
        try:
            import h5py
            buffer = BytesIO(file_content)
            with h5py.File(buffer, 'r') as f:
                data = {}
                def read_h5_obj(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = obj[()]
                    elif isinstance(obj, h5py.Group):
                        data[name] = {}
                        for key in obj.keys():
                            read_h5_obj(f"{name}/{key}", obj[key])
                for key in f.keys():
                    read_h5_obj(key, f[key])
            return data
        except Exception as e:
            return e
    
    def _read_npz(self, file_content):
        buffer = BytesIO(file_content)
        try:
            data = np.load(buffer, allow_pickle=True)
            return {key: data[key] for key in data.files}
        except Exception as e:
            return e
    
    def _read_sql(self, file_content):
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            data = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                data[table_name] = {
                    'columns': columns,
                    'rows': rows
                }
            
            conn.close()
            os.unlink(tmp_path)
            return data
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return e
    
    def _read_json(self, file_content):
        try:
            return json.loads(file_content.decode('utf-8'))
        except Exception as e:
            return e
    
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file with format auto-detection and error handling"""
        if format_type == 'auto':
            format_type = 'pkl'
        
        readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
        
        if format_type in readers:
            data = readers[format_type](file_content)
            
            if isinstance(data, Exception):
                return {
                    'error': str(data),
                    'format': format_type,
                    'status': 'error'
                }
            
            standardized = self._standardize_data(data, format_type)
            standardized['status'] = 'success'
            return standardized
        else:
            error_msg = f"Unsupported format: {format_type}"
            return {
                'error': error_msg,
                'format': format_type,
                'status': 'error'
            }
    
    def _standardize_data(self, data, format_type):
        """Standardize simulation data structure with robust error handling"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type
        }
        
        try:
            if format_type == 'pkl':
                if isinstance(data, dict):
                    standardized['params'] = data.get('params', {})
                    standardized['metadata'] = data.get('metadata', {})
                    standardized['history'] = data.get('history', [])
                else:
                    standardized['error'] = f"PKL data is not a dictionary: {type(data)}"
            
            elif format_type == 'pt':
                if isinstance(data, dict):
                    standardized['params'] = data.get('params', {})
                    standardized['metadata'] = data.get('metadata', {})
                    
                    history = data.get('history', [])
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        history_list = []
                        for key in sorted(history.keys()):
                            frame = history[key]
                            if isinstance(frame, dict) and 'eta' in frame and 'stresses' in frame:
                                history_list.append((frame['eta'], frame['stresses']))
                        standardized['history'] = history_list
                    
                    if 'params' in standardized:
                        for key, value in standardized['params'].items():
                            if torch.is_tensor(value):
                                standardized['params'][key] = value.cpu().numpy()
                else:
                    standardized['error'] = f"PT data is not a dictionary: {type(data)}"
            
            elif format_type == 'h5':
                if isinstance(data, dict):
                    standardized.update(data)
                else:
                    standardized['error'] = f"H5 data is not a dictionary: {type(data)}"
            
            elif format_type == 'npz':
                if isinstance(data, dict):
                    standardized.update(data)
                else:
                    standardized['error'] = f"NPZ data is not a dictionary: {type(data)}"
            
            elif format_type == 'json':
                if isinstance(data, dict):
                    standardized['params'] = data.get('params', {})
                    standardized['metadata'] = data.get('metadata', {})
                    standardized['history'] = data.get('history', [])
                else:
                    standardized['error'] = f"JSON data is not a dictionary: {type(data)}"
            
        except Exception as e:
            standardized['error'] = f"Standardization error: {str(e)}"
        
        return standardized
    
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
        
        all_files_info = self.get_all_files_info()
        
        if not all_files_info:
            if 'st' in globals():
                st.info(f"No solution files found in {self.solutions_dir}")
            return solutions
        
        if 'st' in globals():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for idx, file_info in enumerate(all_files_info):
            try:
                file_path = file_info['path']
                filename = file_info['filename']
                
                cache_key = f"{filename}_{os.path.getmtime(file_path)}_{pt_loading_method}"
                if use_cache and cache_key in self.cache:
                    sim = self.cache[cache_key]
                    if sim.get('status') == 'success':
                        solutions.append(sim)
                    continue
                
                if 'st' in globals():
                    progress = (idx + 1) / len(all_files_info)
                    progress_bar.progress(progress)
                    status_text.text(f"Loading {filename}...")
                
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                sim = self.read_simulation_file(file_content, file_info['format'])
                sim['filename'] = filename
                sim['file_info'] = file_info
                
                if sim.get('status') == 'success' and 'params' in sim and 'history' in sim:
                    if isinstance(sim['params'], dict):
                        self.cache[cache_key] = sim
                        solutions.append(sim)
                    else:
                        failed_files.append({
                            'filename': filename,
                            'error': f"Params is not a dictionary: {type(sim['params'])}"
                        })
                else:
                    error_msg = sim.get('error', 'Unknown error or missing params/history')
                    failed_files.append({
                        'filename': filename,
                        'error': error_msg
                    })
                    
            except Exception as e:
                failed_files.append({
                    'filename': file_info['filename'],
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
# ATTENTION-BASED INTERPOLATOR WITH SPATIAL LOCALITY
# =============================================

class AttentionSpatialInterpolator:
    """Transformer-inspired attention interpolator with spatial locality regularization"""
    
    def __init__(self, sigma=0.3, use_numba=True, attention_dim=32, num_heads=4):
        self.sigma = sigma
        self.use_numba = use_numba
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Parameter mappings
        self.defect_map = {
            'ISF': [1, 0, 0, 0],
            'ESF': [0, 1, 0, 0],
            'Twin': [0, 0, 1, 0],
            'Unknown': [0, 0, 0, 1],
            'No Defect': [0, 0, 0, 0]  # Added for sintering analysis
        }
        
        self.shape_map = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
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
        
        # Shape (5 dimensions)
        shape = params.get('shape', 'Square')
        vector.extend(self.shape_map.get(shape, [0, 0, 0, 0, 0]))
        
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
            # Combine attention and spatial weights
            combined_weights = attention_weights * spatial_weights
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
    
    def interpolate_precise_orientation(self, sources, target_angle_deg, target_params,
                                       region_type='bulk', stress_component='von_mises',
                                       stress_type='max_abs', use_spatial=True):
        """Interpolate at a precise orientation angle with high precision"""
        
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
        source_stresses = []
        
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
            
            # Extract stress from last frame
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
                
                region_stress = extract_region_stress(
                    eta, stress_fields, region_type, stress_component, stress_type
                )
                source_stresses.append(region_stress)
            else:
                source_stresses.append(0.0)
        
        if not valid_sources:
            return None
        
        source_vectors = np.array(source_vectors)
        source_stresses = np.array(source_stresses)
        
        # Compute attention weights
        if len(source_vectors) > 0:
            if self.use_numba:
                try:
                    spatial_weights = compute_gaussian_weights_numba(
                        source_vectors, target_vector, self.sigma
                    )
                except:
                    distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
                    spatial_weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
                    spatial_weights = spatial_weights / (np.sum(spatial_weights) + 1e-8)
            else:
                distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
                spatial_weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
                spatial_weights = spatial_weights / (np.sum(spatial_weights) + 1e-8)
            
            # Combine with attention weights if available
            if use_spatial:
                attention_weights = self.compute_attention_weights(
                    source_vectors, target_vector, use_spatial=True
                )
                if len(attention_weights) > 0:
                    # Blend attention and spatial weights
                    final_weights = 0.7 * attention_weights + 0.3 * spatial_weights
                else:
                    final_weights = spatial_weights
            else:
                final_weights = spatial_weights
            
            # Normalize
            if np.sum(final_weights) > 0:
                final_weights = final_weights / np.sum(final_weights)
            else:
                final_weights = np.ones_like(final_weights) / len(final_weights)
            
            # Weighted combination
            weighted_stress = np.sum(final_weights * source_stresses)
            
            return {
                'region_stress': float(weighted_stress),
                'attention_weights': final_weights.tolist(),
                'target_params': precise_target_params,
                'target_angle_deg': float(target_angle_deg),
                'target_angle_rad': float(target_angle_rad),
                'region_type': region_type,
                'stress_component': stress_component,
                'stress_type': stress_type,
                'num_sources': len(valid_sources),
                'spatial_sigma': float(self.sigma)
            }
        
        return None
    
    def create_orientation_sweep(self, sources, base_params, angle_range, n_points=50,
                                region_type='bulk', stress_component='von_mises',
                                stress_type='max_abs'):
        """Create interpolation sweep across orientation range"""
        
        min_angle, max_angle = angle_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        stresses = []
        weights_list = []
        
        for i, angle in enumerate(angles):
            result = self.interpolate_precise_orientation(
                sources, float(angle), base_params,
                region_type, stress_component, stress_type
            )
            
            if result:
                stresses.append(result['region_stress'])
                weights_list.append(result['attention_weights'])
            else:
                stresses.append(0.0)
                weights_list.append([0.0] * len(sources))
        
        return {
            'angles': angles.tolist(),
            'stresses': stresses,
            'weights_matrix': np.array(weights_list).T.tolist() if weights_list else [],
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type,
            'angle_range': [float(min_angle), float(max_angle)],
            'n_points': n_points,
            'spatial_sigma': float(self.sigma)
        }

# =============================================
# ORIGINAL FILE ANALYZER WITH ORIENTATION SUPPORT
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
        
        # Extract region stress
        region_stress = extract_region_stress(eta, stress_fields, region_type, 
                                             stress_component, stress_type)
        
        # Extract comprehensive statistics
        region_stats = extract_region_statistics(eta, stress_fields, region_type)
        
        # Get solution parameters
        params = solution.get('params', {})
        theta = params.get('theta', 0.0)
        theta_deg = np.rad2deg(theta) if theta is not None else 0.0
        
        return {
            'region_stress': float(region_stress),
            'region_statistics': region_stats,
            'params': params,
            'theta_rad': float(theta) if theta is not None else 0.0,
            'theta_deg': float(theta_deg),
            'filename': solution.get('filename', 'Unknown'),
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type
        }
    
    def analyze_all_solutions(self, solutions, region_type='bulk', 
                             stress_component='von_mises', stress_type='max_abs'):
        """Analyze all solutions for a specific region"""
        results = []
        
        for sol in solutions:
            analysis = self.analyze_solution(sol, region_type, stress_component, stress_type)
            if analysis:
                results.append(analysis)
        
        return results
    
    def get_solutions_by_orientation(self, solutions, min_angle=0, max_angle=360, tolerance=1.0):
        """Filter solutions by orientation range"""
        filtered = []
        
        for sol in solutions:
            params = sol.get('params', {})
            theta = params.get('theta', 0.0)
            theta_deg = np.rad2deg(theta) if theta is not None else 0.0
            
            # Normalize angle to 0-360
            theta_deg = theta_deg % 360
            
            if min_angle <= theta_deg <= max_angle or \
               (min_angle > max_angle and (theta_deg >= min_angle or theta_deg <= max_angle)):
                filtered.append(sol)
            elif abs(theta_deg - min_angle) <= tolerance or abs(theta_deg - max_angle) <= tolerance:
                filtered.append(sol)
        
        return filtered
    
    def create_orientation_distribution(self, solutions):
        """Create distribution of orientations in loaded solutions"""
        orientations = []
        for sol in solutions:
            params = sol.get('params', {})
            theta = params.get('theta', 0.0)
            theta_deg = np.rad2deg(theta) if theta is not None else 0.0
            orientations.append(theta_deg % 360)
        
        return np.array(orientations)
    
    def create_original_sweep_matrix(self, solutions, angle_range, n_points=50,
                                    region_type='bulk', stress_component='von_mises',
                                    stress_type='max_abs'):
        """Create stress matrix from original solutions for orientation sweep"""
        if not solutions:
            return None, None
        
        min_angle, max_angle = angle_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        # Group solutions by nearest angle
        angle_bins = {}
        for sol in solutions:
            analysis = self.analyze_solution(sol, region_type, stress_component, stress_type)
            if analysis:
                theta_deg = analysis['theta_deg'] % 360
                # Find nearest angle in our grid
                nearest_idx = np.argmin(np.abs(angles - theta_deg))
                nearest_angle = angles[nearest_idx]
                
                if nearest_angle not in angle_bins:
                    angle_bins[nearest_angle] = []
                angle_bins[nearest_angle].append(analysis['region_stress'])
        
        # Average stresses for each angle
        stresses = []
        valid_angles = []
        
        for angle in angles:
            if angle in angle_bins and angle_bins[angle]:
                avg_stress = np.mean(angle_bins[angle])
                stresses.append(float(avg_stress))
                valid_angles.append(float(angle))
            else:
                # Use NaN for missing data
                stresses.append(np.nan)
                valid_angles.append(float(angle))
        
        return np.array(stresses), np.array(valid_angles)

# =============================================
# ENHANCED HEATMAP VISUALIZER WITH ROBUST ERROR HANDLING
# =============================================

class EnhancedHeatmapVisualizer:
    """Create heatmap visualizations for stress distribution with robust error handling"""
    
    def __init__(self):
        self.colormap_manager = EnhancedColorMaps()
    
    def validate_stress_data(self, stress_data):
        """Validate stress data for visualization"""
        if stress_data is None:
            return False, "Stress data is None"
        
        if not isinstance(stress_data, np.ndarray):
            return False, f"Stress data is not a numpy array: {type(stress_data)}"
        
        if stress_data.size == 0:
            return False, "Stress data array is empty"
        
        if np.all(np.isnan(stress_data)):
            return False, "Stress data contains only NaN values"
        
        return True, "Valid"
    
    def create_stress_distribution_heatmap(self, eta, stress_fields, stress_component='von_mises',
                                          title="Stress Distribution", cmap='viridis',
                                          figsize=(12, 12), dpi=100, maintain_aspect=True):
        """Create a detailed heatmap of stress distribution with perfect aspect ratio"""
        
        if eta is None or not isinstance(eta, np.ndarray):
            st.warning("Eta data is None or not a numpy array")
            return None
        
        # Get stress data with enhanced validation
        stress_data = None
        if stress_component == 'von_mises' and 'von_mises' in stress_fields:
            stress_data = stress_fields['von_mises']
        elif stress_component == 'sigma_hydro' and 'sigma_hydro' in stress_fields:
            stress_data = stress_fields['sigma_hydro']
        elif stress_component == 'sigma_mag' and 'sigma_mag' in stress_fields:
            stress_data = stress_fields['sigma_mag']
        elif stress_component in stress_fields:
            stress_data = stress_fields[stress_component]
        
        # Validate stress data
        is_valid, error_msg = self.validate_stress_data(stress_data)
        if not is_valid:
            st.warning(f"Cannot create heatmap: {error_msg}")
            return None
        
        # Calculate perfect aspect ratio
        if maintain_aspect:
            # Get data shape for aspect ratio
            data_height, data_width = stress_data.shape
            aspect_ratio = data_height / data_width
            
            # Adjust figure size to maintain aspect ratio
            if data_height > data_width:
                fig_width = figsize[0]
                fig_height = fig_width * aspect_ratio
            else:
                fig_height = figsize[1]
                fig_width = fig_height / aspect_ratio
            
            figsize = (max(fig_width, 8), max(fig_height, 8))
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        
        # Create gridspec for better control
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.7], width_ratios=[1, 1, 1])
        
        try:
            # 1. Main stress heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(stress_data, cmap=cmap, aspect='equal' if maintain_aspect else 'auto')
            ax1.set_title(f'{stress_component.replace("_", " ").title()} Stress', fontsize=10, fontweight='bold')
            ax1.set_xlabel('X Position', fontsize=9)
            ax1.set_ylabel('Y Position', fontsize=9)
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label('Stress (GPa)', fontsize=9)
            cbar1.ax.tick_params(labelsize=8)
            
            # 2. Phase field (eta) with perfect aspect
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(eta, cmap='coolwarm', aspect='equal' if maintain_aspect else 'auto', vmin=0, vmax=1)
            ax2.set_title('Phase Field (η)', fontsize=10, fontweight='bold')
            ax2.set_xlabel('X Position', fontsize=9)
            ax2.set_ylabel('Y Position', fontsize=9)
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label('η Value', fontsize=9)
            cbar2.ax.tick_params(labelsize=8)
            
            # 3. Combined overlay (stress on phase field)
            ax3 = fig.add_subplot(gs[0, 2])
            # Create masked array for defect regions
            defect_mask = eta > 0.6
            interface_mask = (eta >= 0.4) & (eta <= 0.6)
            bulk_mask = eta < 0.4
            
            # Create RGB image for visualization
            overlay = np.zeros((*eta.shape, 3))
            
            # Normalize stress for coloring (handle constant stress case)
            if np.max(stress_data) - np.min(stress_data) > 1e-10:
                stress_normalized = (stress_data - np.min(stress_data)) / (np.max(stress_data) - np.min(stress_data) + 1e-8)
            else:
                stress_normalized = np.zeros_like(stress_data)
            
            # Defect regions in red with stress intensity
            overlay[defect_mask, 0] = 1.0
            overlay[defect_mask, 1] = 1 - stress_normalized[defect_mask] * 0.7
            overlay[defect_mask, 2] = 1 - stress_normalized[defect_mask] * 0.7
            
            # Interface regions in yellow with stress intensity
            overlay[interface_mask, 0] = 1.0
            overlay[interface_mask, 1] = 1.0
            overlay[interface_mask, 2] = 1 - stress_normalized[interface_mask] * 0.5
            
            # Bulk regions in blue with stress intensity
            overlay[bulk_mask, 0] = 1 - stress_normalized[bulk_mask] * 0.5
            overlay[bulk_mask, 1] = 1 - stress_normalized[bulk_mask] * 0.5
            overlay[bulk_mask, 2] = 1.0
            
            ax3.imshow(overlay, aspect='equal' if maintain_aspect else 'auto')
            ax3.set_title('Stress Overlay on Regions', fontsize=10, fontweight='bold')
            ax3.set_xlabel('X Position', fontsize=9)
            ax3.set_ylabel('Y Position', fontsize=9)
            
            # Add region legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Defect (η > 0.6)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.7, label='Interface (0.4 ≤ η ≤ 0.6)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Bulk (η < 0.4)')
            ]
            ax3.legend(handles=legend_elements, loc='upper right', fontsize=7)
            
            # 4. Stress histogram with robust bin calculation
            ax4 = fig.add_subplot(gs[1, 0])
            flat_stress = stress_data.flatten()
            valid_stress = flat_stress[~np.isnan(flat_stress)]
            
            if len(valid_stress) > 0:
                n_bins = min(50, max(1, int(np.sqrt(len(valid_stress)))))  # Ensure at least 1 bin
                ax4.hist(valid_stress, bins=n_bins, edgecolor='black', alpha=0.7, density=True)
                ax4.set_title('Stress Distribution Histogram', fontsize=10, fontweight='bold')
                ax4.set_xlabel('Stress (GPa)', fontsize=9)
                ax4.set_ylabel('Probability Density', fontsize=9)
                ax4.grid(True, alpha=0.3, linestyle='--')
                ax4.tick_params(axis='both', which='major', labelsize=8)
                
                # Add vertical lines for region statistics
                colors = ['red', 'orange', 'blue']
                labels = ['Defect', 'Interface', 'Bulk']
                masks = [defect_mask, interface_mask, bulk_mask]
                
                for mask, color, label in zip(masks, colors, labels):
                    if np.any(mask):
                        region_stress = stress_data[mask]
                        if len(region_stress) > 0:
                            mean_stress = np.mean(region_stress)
                            ax4.axvline(mean_stress, color=color, linestyle='--', linewidth=2,
                                       label=f'{label}: {mean_stress:.3f} GPa')
                
                ax4.legend(fontsize=7, loc='upper right')
            else:
                ax4.text(0.5, 0.5, 'No valid stress data for histogram', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Stress Distribution Histogram', fontsize=10, fontweight='bold')
            
            # 5. Region-wise stress box plot
            ax5 = fig.add_subplot(gs[1, 1])
            region_data = []
            region_labels = []
            region_colors = []
            
            masks = [defect_mask, interface_mask, bulk_mask]
            colors = ['red', 'orange', 'blue']
            labels = ['Defect', 'Interface', 'Bulk']
            
            for mask, color, label in zip(masks, colors, labels):
                if np.any(mask):
                    region_stress = stress_data[mask]
                    region_stress_valid = region_stress[~np.isnan(region_stress)]
                    if len(region_stress_valid) > 1:  # Need at least 2 points for box plot
                        region_data.append(region_stress_valid.flatten())
                        region_labels.append(label)
                        region_colors.append(color)
            
            if region_data:
                bp = ax5.boxplot(region_data, labels=region_labels, patch_artist=True,
                                widths=0.6, showfliers=False, whis=[5, 95])
                for patch, color in zip(bp['boxes'], region_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_linewidth(1.5)
                
                # Customize whiskers and caps
                for whisker in bp['whiskers']:
                    whisker.set(color='black', linewidth=1.5, linestyle='-')
                for cap in bp['caps']:
                    cap.set(color='black', linewidth=1.5)
                for median in bp['medians']:
                    median.set(color='black', linewidth=2)
                
                ax5.set_title('Region-wise Stress Distribution', fontsize=10, fontweight='bold')
                ax5.set_ylabel('Stress (GPa)', fontsize=9)
                ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
                ax5.tick_params(axis='both', which='major', labelsize=8)
            else:
                ax5.text(0.5, 0.5, 'Insufficient data for box plots', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Region-wise Stress Distribution', fontsize=10, fontweight='bold')
            
            # 6. Stress profile along centerlines
            ax6 = fig.add_subplot(gs[1, 2])
            center_y = stress_data.shape[0] // 2
            center_x = stress_data.shape[1] // 2
            
            # Horizontal profile
            horizontal_profile = stress_data[center_y, :]
            # Vertical profile
            vertical_profile = stress_data[:, center_x]
            
            x_positions = np.arange(len(horizontal_profile))
            y_positions = np.arange(len(vertical_profile))
            
            ax6.plot(x_positions, horizontal_profile, label=f'Horizontal (y={center_y})', 
                    color='blue', linewidth=2, marker='o', markersize=3)
            ax6.plot(y_positions, vertical_profile, label=f'Vertical (x={center_x})', 
                    color='red', linewidth=2, marker='s', markersize=3)
            
            ax6.set_title('Stress Profiles along Centerlines', fontsize=10, fontweight='bold')
            ax6.set_xlabel('Position', fontsize=9)
            ax6.set_ylabel('Stress (GPa)', fontsize=9)
            ax6.legend(fontsize=8, loc='best')
            ax6.grid(True, alpha=0.3, linestyle='--')
            ax6.tick_params(axis='both', which='major', labelsize=8)
            
            # 7. Statistical summary table
            ax7 = fig.add_subplot(gs[2, :])
            ax7.axis('tight')
            ax7.axis('off')
            
            # Calculate statistics
            stats_data = []
            stats_data.append(['Global Statistics', '', ''])
            
            if len(valid_stress) > 0:
                stats_data.append(['Mean Stress', f'{np.mean(valid_stress):.4f}', 'GPa'])
                stats_data.append(['Max Stress', f'{np.max(valid_stress):.4f}', 'GPa'])
                stats_data.append(['Min Stress', f'{np.min(valid_stress):.4f}', 'GPa'])
                stats_data.append(['Std Deviation', f'{np.std(valid_stress):.4f}', 'GPa'])
                stats_data.append(['95th Percentile', f'{np.percentile(valid_stress, 95):.4f}', 'GPa'])
            else:
                stats_data.append(['Mean Stress', 'N/A', 'GPa'])
                stats_data.append(['Max Stress', 'N/A', 'GPa'])
                stats_data.append(['Min Stress', 'N/A', 'GPa'])
                stats_data.append(['Std Deviation', 'N/A', 'GPa'])
                stats_data.append(['95th Percentile', 'N/A', 'GPa'])
            
            # Add region-specific statistics
            stats_data.append(['', '', ''])
            stats_data.append(['Region Statistics', '', ''])
            
            for mask, label in zip(masks, labels):
                if np.any(mask):
                    region_stress = stress_data[mask]
                    region_stress_valid = region_stress[~np.isnan(region_stress)]
                    if len(region_stress_valid) > 0:
                        stats_data.append([f'{label} Mean', f'{np.mean(region_stress_valid):.4f}', 'GPa'])
                        stats_data.append([f'{label} Max', f'{np.max(region_stress_valid):.4f}', 'GPa'])
                    else:
                        stats_data.append([f'{label} Mean', 'N/A', 'GPa'])
                        stats_data.append([f'{label} Max', 'N/A', 'GPa'])
            
            # Create table
            if stats_data:
                table = ax7.table(cellText=stats_data, cellLoc='left', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
                
                # Style table cells
                for i in range(len(stats_data)):
                    for j in range(3):
                        cell = table[(i, j)]
                        if i == 0 or i == 7:  # Header rows
                            cell.set_text_props(weight='bold', fontsize=9)
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(color='white')
                        elif i % 2 == 1:
                            cell.set_facecolor('#f0f0f0')
            
            # Add overall title
            plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            plt.close(fig)
            return None
    
    def create_interactive_heatmap(self, eta, stress_fields, stress_component='von_mises',
                                  title="Interactive Stress Distribution", maintain_aspect=True):
        """Create an interactive Plotly heatmap with perfect aspect ratio"""
        
        if eta is None or not isinstance(eta, np.ndarray):
            st.warning("Eta data is None or not a numpy array")
            return None
        
        # Get stress data
        stress_data = None
        if stress_component == 'von_mises' and 'von_mises' in stress_fields:
            stress_data = stress_fields['von_mises']
        elif stress_component == 'sigma_hydro' and 'sigma_hydro' in stress_fields:
            stress_data = stress_fields['sigma_hydro']
        elif stress_component == 'sigma_mag' and 'sigma_mag' in stress_fields:
            stress_data = stress_fields['sigma_mag']
        elif stress_component in stress_fields:
            stress_data = stress_fields[stress_component]
        
        # Validate stress data
        is_valid, error_msg = self.validate_stress_data(stress_data)
        if not is_valid:
            st.warning(f"Cannot create interactive heatmap: {error_msg}")
            return None
        
        # Calculate aspect ratio
        data_height, data_width = stress_data.shape
        aspect_ratio = data_height / data_width
        
        # Set figure dimensions based on aspect ratio
        if maintain_aspect:
            if data_height > data_width:
                width = 800
                height = int(width * aspect_ratio)
            else:
                height = 600
                width = int(height / aspect_ratio)
        else:
            width, height = 1000, 800
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=(
                    f'{stress_component.replace("_", " ").title()} Distribution',
                    'Phase Field (η)',
                    'Stress Histogram',
                    'Region Analysis',
                    'Stress Profiles',
                    'Statistics'
                ),
                specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'xy'}],
                       [{'type': 'box'}, {'type': 'xy'}, {'type': 'table'}]],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Heatmap 1: Stress distribution
            fig.add_trace(
                go.Heatmap(
                    z=stress_data,
                    colorscale='viridis',
                    colorbar=dict(title='Stress (GPa)', x=0.31, y=0.8, len=0.4),
                    hovertemplate='X: %{x}<br>Y: %{y}<br>Stress: %{z:.3f} GPa<extra></extra>',
                    showscale=True
                ),
                row=1, col=1
            )
            
            # Heatmap 2: Phase field
            fig.add_trace(
                go.Heatmap(
                    z=eta,
                    colorscale='RdBu',
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title='η Value', x=0.69, y=0.8, len=0.4),
                    hovertemplate='X: %{x}<br>Y: %{y}<br>η: %{z:.3f}<extra></extra>',
                    showscale=True
                ),
                row=1, col=2
            )
            
            # Histogram
            flat_stress = stress_data.flatten()
            valid_stress = flat_stress[~np.isnan(flat_stress)]
            
            if len(valid_stress) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=valid_stress,
                        nbinsx=min(50, len(valid_stress)),
                        marker_color='blue',
                        opacity=0.7,
                        name='Stress Distribution',
                        hovertemplate='Stress: %{x:.3f} GPa<br>Count: %{y}<extra></extra>'
                    ),
                    row=1, col=3
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode='text',
                        text=['No valid stress data'],
                        textposition='middle center',
                        showlegend=False
                    ),
                    row=1, col=3
                )
            
            # Region analysis - Box plots
            defect_mask = eta > 0.6
            interface_mask = (eta >= 0.4) & (eta <= 0.6)
            bulk_mask = eta < 0.4
            
            region_data = []
            region_labels = []
            region_colors = ['red', 'orange', 'blue']
            
            for region_name, mask, color in [
                ('Defect', defect_mask, 'red'),
                ('Interface', interface_mask, 'orange'),
                ('Bulk', bulk_mask, 'blue')
            ]:
                if np.any(mask):
                    region_stress = stress_data[mask]
                    region_stress_valid = region_stress[~np.isnan(region_stress)]
                    if len(region_stress_valid) > 0:
                        region_data.append(region_stress_valid.flatten())
                        region_labels.append(region_name)
            
            # Create box plots for each region
            if region_data:
                for i, (data, label, color) in enumerate(zip(region_data, region_labels, region_colors[:len(region_data)])):
                    fig.add_trace(
                        go.Box(
                            y=data,
                            name=label,
                            boxpoints='outliers',
                            marker_color=color,
                            showlegend=False,
                            hovertemplate=f'{label}<br>Q1: %{{q1:.3f}}<br>Median: %{{median:.3f}}<br>Q3: %{{q3:.3f}}<extra></extra>'
                        ),
                        row=2, col=1
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode='text',
                        text=['Insufficient region data'],
                        textposition='middle center',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Stress profiles along centerlines
            center_y = stress_data.shape[0] // 2
            center_x = stress_data.shape[1] // 2
            
            horizontal_profile = stress_data[center_y, :]
            vertical_profile = stress_data[:, center_x]
            
            x_positions = np.arange(len(horizontal_profile))
            y_positions = np.arange(len(vertical_profile))
            
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=horizontal_profile,
                    mode='lines+markers',
                    name=f'Horizontal (y={center_y})',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4),
                    hovertemplate='X: %{x}<br>Stress: %{y:.3f} GPa<extra></extra>'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=y_positions,
                    y=vertical_profile,
                    mode='lines+markers',
                    name=f'Vertical (x={center_x})',
                    line=dict(color='red', width=2),
                    marker=dict(size=4, symbol='square'),
                    hovertemplate='Y: %{x}<br>Stress: %{y:.3f} GPa<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Statistics table
            stats_data = []
            stats_data.append(['Statistic', 'Value', 'Unit'])
            
            # Global statistics
            if len(valid_stress) > 0:
                stats_data.append(['Global Mean', f'{np.mean(valid_stress):.4f}', 'GPa'])
                stats_data.append(['Global Max', f'{np.max(valid_stress):.4f}', 'GPa'])
                stats_data.append(['Global Min', f'{np.min(valid_stress):.4f}', 'GPa'])
                stats_data.append(['Global Std', f'{np.std(valid_stress):.4f}', 'GPa'])
            else:
                stats_data.append(['Global Mean', 'N/A', 'GPa'])
                stats_data.append(['Global Max', 'N/A', 'GPa'])
                stats_data.append(['Global Min', 'N/A', 'GPa'])
                stats_data.append(['Global Std', 'N/A', 'GPa'])
            
            # Region statistics
            for region_name, mask in [('Defect', defect_mask), ('Interface', interface_mask), ('Bulk', bulk_mask)]:
                if np.any(mask):
                    region_stress = stress_data[mask]
                    region_stress_valid = region_stress[~np.isnan(region_stress)]
                    if len(region_stress_valid) > 0:
                        stats_data.append([f'{region_name} Mean', f'{np.mean(region_stress_valid):.4f}', 'GPa'])
                        stats_data.append([f'{region_name} Max', f'{np.max(region_stress_valid):.4f}', 'GPa'])
                    else:
                        stats_data.append([f'{region_name} Mean', 'N/A', 'GPa'])
                        stats_data.append([f'{region_name} Max', 'N/A', 'GPa'])
            
            # Create table
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>Statistic</b>', '<b>Value</b>', '<b>Unit</b>'],
                        fill_color='#4CAF50',
                        align='left',
                        font=dict(size=12, color='white')
                    ),
                    cells=dict(
                        values=list(zip(*stats_data)),
                        fill_color=[['white', '#f0f0f0'] * len(stats_data)],
                        align='left',
                        font=dict(size=11)
                    )
                ),
                row=2, col=3
            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5,
                    xanchor='center',
                    y=0.97
                ),
                height=height * 1.5,
                width=width * 1.2,
                showlegend=True,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=100, b=50)
            )
            
            # Update axes with perfect aspect ratio for heatmaps
            if maintain_aspect:
                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=2)
            
            # Update axes labels
            fig.update_xaxes(title_text="X Position", row=1, col=1)
            fig.update_yaxes(title_text="Y Position", row=1, col=1)
            fig.update_xaxes(title_text="X Position", row=1, col=2)
            fig.update_yaxes(title_text="Y Position", row=1, col=2)
            fig.update_xaxes(title_text="Stress (GPa)", row=1, col=3)
            fig.update_yaxes(title_text="Frequency", row=1, col=3)
            fig.update_xaxes(title_text="Region", row=2, col=1)
            fig.update_yaxes(title_text="Stress (GPa)", row=2, col=1)
            fig.update_xaxes(title_text="Position", row=2, col=2)
            fig.update_yaxes(title_text="Stress (GPa)", row=2, col=2)
            
            # Update font sizes
            fig.update_annotations(font_size=10)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating interactive heatmap: {str(e)}")
            return None

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER WITH FIXES
# =============================================

class EnhancedSunburstRadarVisualizer:
    """Enhanced sunburst and radar charts with 50+ colormaps and visualization enhancements"""
    
    def __init__(self):
        self.colormap_manager = EnhancedColorMaps()
    
    def validate_visualization_data(self, stress_data, angles):
        """Validate data for visualization"""
        if stress_data is None or angles is None:
            return False, "Data is None"
        
        if isinstance(stress_data, list):
            stress_data = np.array(stress_data)
        if isinstance(angles, list):
            angles = np.array(angles)
        
        if len(stress_data) == 0 or len(angles) == 0:
            return False, "Empty data arrays"
        
        if len(stress_data) != len(angles):
            return False, f"Data length mismatch: stress={len(stress_data)}, angles={len(angles)}"
        
        return True, "Valid"
    
    def create_enhanced_plotly_sunburst(self, stress_matrix, times, thetas, title, 
                                       cmap='rainbow', marker_size=12, line_width=1.5,
                                       font_size=18, width=900, height=750,
                                       show_colorbar=True, colorbar_title="Stress (GPa)",
                                       hover_template=None, is_time_series=True,
                                       angle_range=None):
        """Interactive sunburst with Plotly - handles both time series and orientation sweeps"""
        
        # Ensure inputs are proper arrays
        if isinstance(stress_matrix, list):
            stress_matrix = np.array(stress_matrix)
        if isinstance(thetas, list):
            thetas = np.array(thetas)
        if isinstance(times, list):
            times = np.array(times)
        
        # Handle angle range
        if angle_range is not None:
            min_angle, max_angle = angle_range
            # Filter data within angle range
            mask = (thetas >= min_angle) & (thetas <= max_angle)
            thetas = thetas[mask]
            if is_time_series:
                stress_matrix = stress_matrix[:, mask]
            else:
                stress_matrix = stress_matrix[mask]
        
        if is_time_series:
            # Time series sunburst
            if stress_matrix.ndim == 1:
                stress_matrix = stress_matrix.reshape(1, -1)
            
            theta_deg = np.deg2rad(thetas)
            theta_grid, time_grid = np.meshgrid(theta_deg, times)
            
            # Flatten the arrays for scatter plot
            r_flat = time_grid.flatten()
            theta_flat = np.rad2deg(theta_grid).flatten()
            stress_flat = stress_matrix.flatten()
        else:
            # Orientation sweep sunburst (single time point)
            if stress_matrix.ndim == 2 and stress_matrix.shape[0] == 1:
                stress_matrix = stress_matrix.flatten()
            
            r_flat = np.zeros_like(thetas)  # Zero time dimension
            theta_flat = thetas
            stress_flat = stress_matrix
        
        # Ensure no NaN values
        valid_mask = ~np.isnan(stress_flat)
        if not np.any(valid_mask):
            st.warning("No valid stress data for sunburst visualization")
            return None
        
        r_flat = r_flat[valid_mask]
        theta_flat = theta_flat[valid_mask]
        stress_flat = stress_flat[valid_mask]
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Default hover template
        if hover_template is None:
            if is_time_series:
                hover_template = (
                    '<b>Time</b>: %{r:.2f}s<br>' +
                    '<b>Orientation</b>: %{theta:.1f}°<br>' +
                    '<b>Stress</b>: %{marker.color:.4f} GPa<br>' +
                    '<extra></extra>'
                )
            else:
                hover_template = (
                    '<b>Orientation</b>: %{theta:.1f}°<br>' +
                    '<b>Stress</b>: %{marker.color:.4f} GPa<br>' +
                    '<extra></extra>'
                )
        
        # Add scatter polar trace with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=r_flat if is_time_series else stress_flat * 10,  # Scale for visibility
            theta=theta_flat,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=stress_flat,
                colorscale=cmap,
                showscale=show_colorbar,
                colorbar=dict(
                    title=dict(text=colorbar_title, font=dict(size=font_size, color='black')),
                    tickfont=dict(size=font_size-2, color='black'),
                    thickness=25,
                    len=0.8,
                    x=1.15,
                    xpad=20,
                    ypad=20,
                    tickformat='.3f',
                    title_side='right'
                ),
                line=dict(width=line_width, color='rgba(255, 255, 255, 0.8)'),
                opacity=0.9,
                symbol='circle',
                sizemode='diameter',
                sizemin=3,
                cmin=np.nanmin(stress_flat) if len(stress_flat) > 0 else 0,
                cmax=np.nanmax(stress_flat) if len(stress_flat) > 0 else 1
            ),
            hovertemplate=hover_template,
            name='Stress Distribution'
        ))
        
        # Enhanced layout
        if is_time_series:
            radial_title = "Time (s)"
            if len(r_flat) > 0:
                radial_range = [0, max(r_flat) * 1.1]
            else:
                radial_range = [0, 1]
            radial_ticksuffix = " s"
        else:
            radial_title = "Stress (GPa)"
            if len(stress_flat) > 0:
                radial_range = [0, max(stress_flat) * 1.3]
            else:
                radial_range = [0, 1]
            radial_ticksuffix = " GPa"
        
        # Set sector based on angle range
        if angle_range is not None:
            sector = list(angle_range)  # Convert to list for Plotly
        else:
            sector = [0, 360]
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=font_size+4, family="Arial Black, sans-serif", color='darkblue'),
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top'
            ),
            polar=dict(
                radialaxis=dict(
                    title=dict(
                        text=radial_title,
                        font=dict(size=font_size+2, color='black', family='Arial')
                    ),
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    showline=True,
                    tickfont=dict(size=font_size, color='black', family='Arial'),
                    tickformat='.1f',
                    range=radial_range,
                    ticksuffix=radial_ticksuffix,
                    showticksuffix='all'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickfont=dict(size=font_size, color='black', family='Arial'),
                    tickmode='array',
                    tickvals=list(range(int(sector[0]), int(sector[1])+1, 30)),
                    ticktext=[f'{i}°' for i in range(int(sector[0]), int(sector[1])+1, 30)],
                    period=360,
                    thetaunit="degrees",
                    range=sector
                ),
                bgcolor="rgba(240, 240, 240, 0.5)",
                sector=sector,
                hole=0.1 if is_time_series else 0.0
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                x=1.2,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=font_size, family='Arial')
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=100, r=200, t=100, b=100),
            font=dict(family="Arial, sans-serif", size=font_size)
        )
        
        # Add radial lines for important orientations
        for angle in [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]:
            if angle_range is None or (angle >= angle_range[0] and angle <= angle_range[1]):
                fig.add_trace(go.Scatterpolar(
                    r=[0, radial_range[1]],
                    theta=[angle, angle],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.2)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Highlight specific orientations
        highlight_angles = [54.7]  # Ag FCC twin habit plane
        for angle in highlight_angles:
            if angle_range is None or (angle >= angle_range[0] and angle <= angle_range[1]):
                fig.add_trace(go.Scatterpolar(
                    r=[0, radial_range[1]],
                    theta=[angle, angle],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 0, 0.4)', width=3, dash='solid'),
                    name=f'Habit Plane ({angle}°)',
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        return fig
    
    def create_enhanced_plotly_radar(self, stress_values, thetas, component_name, 
                                    title="Radar Chart", line_width=4, marker_size=12, 
                                    fill_alpha=0.3, font_size=16, width=800, height=700,
                                    show_mean=True, show_std=True, color='steelblue',
                                    show_habit_plane=True, angle_range=None):
        """Interactive enhanced radar chart with Plotly"""
        
        # Ensure inputs are proper arrays
        if isinstance(stress_values, list):
            stress_values = np.array(stress_values)
        if isinstance(thetas, list):
            thetas = np.array(thetas)
        
        # Validate data
        is_valid, error_msg = self.validate_visualization_data(stress_values, thetas)
        if not is_valid:
            st.warning(f"Cannot create radar chart: {error_msg}")
            return None
        
        # Handle angle range
        if angle_range is not None:
            min_angle, max_angle = angle_range
            mask = (thetas >= min_angle) & (thetas <= max_angle)
            thetas = thetas[mask]
            stress_values = stress_values[mask]
        
        # Remove NaN values
        valid_mask = ~np.isnan(stress_values)
        thetas = thetas[valid_mask]
        stress_values = stress_values[valid_mask]
        
        if len(stress_values) == 0:
            st.warning("No valid stress data for radar chart")
            return None
        
        # Ensure proper closure
        if angle_range is None or (angle_range[0] == 0 and angle_range[1] == 360):
            thetas_closed = np.append(thetas, 360)
            stress_values_closed = np.append(stress_values, stress_values[0])
        else:
            # Don't close for sector view
            thetas_closed = thetas
            stress_values_closed = stress_values
        
        # Create figure
        fig = go.Figure()
        
        # Add radar trace with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=stress_values_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor=f'rgba(70, 130, 180, {fill_alpha})',
            line=dict(color=color, width=line_width),
            marker=dict(size=marker_size, color=color, symbol='circle'),
            name=component_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
            text=[f'{v:.3f} GPa' for v in stress_values_closed],
            textposition='top center'
        ))
        
        # Add mean value line
        if show_mean and len(stress_values) > 0:
            mean_val = np.mean(stress_values)
            fig.add_trace(go.Scatterpolar(
                r=[mean_val] * len(thetas_closed),
                theta=thetas_closed,
                mode='lines',
                line=dict(color='firebrick', width=3, dash='dash'),
                name=f'Mean: {mean_val:.3f} GPa',
                hovertemplate='Mean Stress: %{r:.3f} GPa<extra></extra>'
            ))
        
        # Add standard deviation band
        if show_std and len(stress_values) > 0:
            mean_val = np.mean(stress_values)
            std_val = np.std(stress_values)
            fig.add_trace(go.Scatterpolar(
                r=[mean_val + std_val] * len(thetas_closed),
                theta=thetas_closed,
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name=f'Mean ± Std: {std_val:.3f} GPa',
                hovertemplate='Mean + Std: %{r:.3f} GPa<extra></extra>'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=[mean_val - std_val] * len(thetas_closed),
                theta=thetas_closed,
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name=f'Mean - Std: {std_val:.3f} GPa',
                hovertemplate='Mean - Std: %{r:.3f} GPa<extra></extra>',
                showlegend=False
            ))
        
        # Highlight habit plane orientation (54.7° for Ag FCC twin)
        if show_habit_plane and len(thetas) > 0:
            habit_angle = 54.7
            if angle_range is None or (habit_angle >= angle_range[0] and habit_angle <= angle_range[1]):
                # Find nearest data point
                idx = np.argmin(np.abs(thetas - habit_angle))
                habit_stress = stress_values[idx] if idx < len(stress_values) else 0
                
                fig.add_trace(go.Scatterpolar(
                    r=[0, habit_stress],
                    theta=[habit_angle, habit_angle],
                    mode='lines+markers',
                    line=dict(color='green', width=4, dash='dashdot'),
                    marker=dict(size=15, color='green', symbol='star'),
                    name=f'Habit Plane ({habit_angle}°): {habit_stress:.3f} GPa',
                    hovertemplate=f'Habit Plane ({habit_angle}°): %{{r:.3f}} GPa<extra></extra>'
                ))
        
        # Set angular axis range
        if angle_range is not None:
            angular_range = list(angle_range)  # Convert to list for Plotly
        else:
            angular_range = [0, 360]
        
        # Set radial range
        if len(stress_values) > 0:
            radial_range = [0, max(stress_values) * 1.3]
        else:
            radial_range = [0, 1]
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=font_size+4, family="Arial Black", color='darkblue'),
                x=0.5,
                xanchor='center'
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=radial_range,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    tickfont=dict(size=font_size-2, color='black'),
                    title=dict(text='Stress (GPa)', 
                              font=dict(size=font_size, color='black')),
                    ticksuffix=' GPa'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickvals=list(range(int(angular_range[0]), int(angular_range[1])+1, 45)),
                    ticktext=[f'{i}°' for i in range(int(angular_range[0]), int(angular_range[1])+1, 45)],
                    tickfont=dict(size=font_size, color='black'),
                    period=360,
                    range=angular_range
                ),
                bgcolor="rgba(240, 240, 240, 0.5)"
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=font_size, family='Arial')
            ),
            width=width,
            height=height
        )
        
        return fig
    
    def create_comparison_radar(self, original_stress, interpolated_stress, thetas,
                               title="Comparison: Original vs Interpolated",
                               original_name="Original", interpolated_name="Interpolated",
                               angle_range=None):
        """Create radar chart comparing original and interpolated solutions"""
        
        # Ensure inputs are proper arrays
        if isinstance(original_stress, list):
            original_stress = np.array(original_stress)
        if isinstance(interpolated_stress, list):
            interpolated_stress = np.array(interpolated_stress)
        if isinstance(thetas, list):
            thetas = np.array(thetas)
        
        # Handle angle range
        if angle_range is not None:
            min_angle, max_angle = angle_range
            mask = (thetas >= min_angle) & (thetas <= max_angle)
            thetas = thetas[mask]
            original_stress = original_stress[mask]
            interpolated_stress = interpolated_stress[mask]
        
        # Remove NaN values
        original_valid = ~np.isnan(original_stress)
        interpolated_valid = ~np.isnan(interpolated_stress)
        
        # Ensure proper closure
        if angle_range is None or (angle_range[0] == 0 and angle_range[1] == 360):
            thetas_closed = np.append(thetas, 360)
            original_closed = np.append(original_stress, original_stress[0])
            interpolated_closed = np.append(interpolated_stress, interpolated_stress[0])
        else:
            # Don't close for sector view
            thetas_closed = thetas
            original_closed = original_stress
            interpolated_closed = interpolated_stress
        
        fig = go.Figure()
        
        # Original solutions
        fig.add_trace(go.Scatterpolar(
            r=original_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgb(31, 119, 180)', width=3),
            name=original_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Original Stress: %{r:.4f} GPa<extra></extra>'
        ))
        
        # Interpolated solutions
        fig.add_trace(go.Scatterpolar(
            r=interpolated_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgb(255, 127, 14)', width=3),
            name=interpolated_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Interpolated Stress: %{r:.4f} GPa<extra></extra>'
        ))
        
        # Set angular axis range
        if angle_range is not None:
            angular_range = list(angle_range)  # Convert to list for Plotly
        else:
            angular_range = [0, 360]
        
        # Set radial range
        max_stress = max(np.nanmax(original_stress), np.nanmax(interpolated_stress))
        radial_range = [0, max_stress * 1.2]
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family="Arial Black", color='darkblue'),
                x=0.5,
                xanchor='center'
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=radial_range,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    tickfont=dict(size=14, color='black'),
                    title=dict(text='Stress (GPa)', 
                              font=dict(size=16, color='black'))
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickvals=list(range(int(angular_range[0]), int(angular_range[1])+1, 45)),
                    ticktext=[f'{i}°' for i in range(int(angular_range[0]), int(angular_range[1])+1, 45)],
                    tickfont=dict(size=14, color='black'),
                    range=angular_range
                ),
                bgcolor="rgba(240, 240, 240, 0.3)"
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=14, family='Arial')
            ),
            width=900,
            height=700
        )
        
        return fig

    def create_sintering_temperature_radar(self, defect_types, angles, sintering_calculator,
                                          title="Sintering Temperature Prediction Radar",
                                          line_width=4, marker_size=12, 
                                          font_size=16, width=900, height=800,
                                          angle_range=None):
        """Create radar chart for sintering temperature prediction across defect types"""
        
        # Create base parameters for each defect type
        base_params_list = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Colors for different defect types
        
        for i, defect_type in enumerate(defect_types):
            base_params = {
                'defect_type': defect_type,
                'shape': 'Square',
                'eps0': 0.707,
                'kappa': 0.6,
                'theta': 0.0
            }
            base_params_list.append(base_params)
        
        # Simulate stresses for each defect type (in practice, use interpolator)
        # For demonstration, we'll create realistic stress patterns
        np.random.seed(42)
        sintering_temps = []
        
        for i, (defect_type, base_params) in enumerate(zip(defect_types, base_params_list)):
            # Different stress patterns for different defect types
            if defect_type == 'ISF':
                base_stress = 8.0 + np.random.normal(0, 1, len(angles))
            elif defect_type == 'ESF':
                base_stress = 12.0 + np.random.normal(0, 1.5, len(angles))
            elif defect_type == 'Twin':
                base_stress = 15.0 + np.random.normal(0, 2, len(angles))
            else:  # No Defect
                base_stress = 2.0 + np.random.normal(0, 0.5, len(angles))
            
            # Add sinusoidal pattern based on orientation
            pattern = 5.0 * np.sin(np.deg2rad(angles) * 2)
            stresses = np.abs(base_stress + pattern)
            
            # Compute sintering temperatures
            temps = sintering_calculator.compute_sintering_temperature_exponential(stresses)
            sintering_temps.append(temps)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each defect type
        for i, (defect_type, temps, color) in enumerate(zip(defect_types, sintering_temps, colors)):
            if angle_range is None or (angle_range[0] == 0 and angle_range[1] == 360):
                angles_closed = np.append(angles, 360)
                temps_closed = np.append(temps, temps[0])
            else:
                angles_closed = angles
                temps_closed = temps
            
            fig.add_trace(go.Scatterpolar(
                r=temps_closed,
                theta=angles_closed,
                fill='toself',
                fillcolor=f'rgba{tuple(int(color.lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + (0.2,)}',
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size, color=color, symbol='circle'),
                name=defect_type,
                hovertemplate=f'Defect: {defect_type}<br>Orientation: %{{theta:.1f}}°<br>T_sinter: %{{r:.1f}} K (%{{r-273.15:.0f}}°C)<extra></extra>',
                text=[f'{t:.1f} K' for t in temps_closed],
                textposition='top center'
            ))
        
        # Highlight habit plane
        if angle_range is None or (54.7 >= angle_range[0] and 54.7 <= angle_range[1]):
            # Find average temperature at habit plane
            avg_temp = np.mean([temps[np.argmin(np.abs(angles - 54.7))] for temps in sintering_temps])
            
            fig.add_trace(go.Scatterpolar(
                r=[0, avg_temp * 1.1],
                theta=[54.7, 54.7],
                mode='lines+markers',
                line=dict(color='green', width=4, dash='dashdot'),
                marker=dict(size=15, color='green', symbol='star'),
                name=f'Habit Plane (54.7°): ~{avg_temp:.0f} K',
                hovertemplate=f'Habit Plane (54.7°): ~{avg_temp:.0f} K<extra></extra>'
            ))
        
        # Set angular axis range
        if angle_range is not None:
            angular_range = list(angle_range)
        else:
            angular_range = [0, 360]
        
        # Set radial range
        all_temps = np.concatenate(sintering_temps)
        radial_range = [min(all_temps) * 0.9, max(all_temps) * 1.1]
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=font_size+4, family="Arial Black", color='darkblue'),
                x=0.5,
                xanchor='center'
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=radial_range,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    tickfont=dict(size=font_size-2, color='black'),
                    title=dict(text='Sintering Temperature (K)', 
                              font=dict(size=font_size, color='black')),
                    ticksuffix=' K'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickvals=list(range(int(angular_range[0]), int(angular_range[1])+1, 45)),
                    ticktext=[f'{i}°' for i in range(int(angular_range[0]), int(angular_range[1])+1, 45)],
                    tickfont=dict(size=font_size, color='black'),
                    range=angular_range
                ),
                bgcolor="rgba(240, 240, 240, 0.5)"
            ),
            showlegend=True,
            legend=dict(
                x=1.15,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=font_size, family='Arial'),
                title=dict(text='Defect Type', font=dict(size=font_size+2))
            ),
            width=width,
            height=height
        )
        
        # Add second radial axis for Celsius
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    tickvals=np.linspace(radial_range[0], radial_range[1], 6),
                    ticktext=[f'{t:.0f}' for t in np.linspace(radial_range[0], radial_range[1], 6)]
                )
            )
        )
        
        # Add annotation with model formula
        formula_text = r"T_sinter(σ_h) = T₀ × exp(-β × |σ_h| / G)"
        fig.add_annotation(
            x=0.5,
            y=0.02,
            xref="paper",
            yref="paper",
            text=formula_text,
            showarrow=False,
            font=dict(size=12, family="Courier New", color="darkred"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="darkred",
            borderwidth=1
        )
        
        return fig

# =============================================
# ENHANCED RESULTS MANAGER WITH JSON FIX
# =============================================

class EnhancedResultsManager:
    """Manager for saving and exporting results with enhanced formatting"""
    
    @staticmethod
    def prepare_orientation_sweep_data(sweep_results, original_results=None, metadata=None):
        """Prepare orientation sweep data for export"""
        if metadata is None:
            metadata = {}
        
        # Convert numpy arrays to lists for JSON serialization
        if sweep_results:
            if 'angles' in sweep_results and isinstance(sweep_results['angles'], np.ndarray):
                sweep_results['angles'] = sweep_results['angles'].tolist()
            if 'stresses' in sweep_results and isinstance(sweep_results['stresses'], np.ndarray):
                sweep_results['stresses'] = sweep_results['stresses'].tolist()
            if 'weights_matrix' in sweep_results and isinstance(sweep_results['weights_matrix'], np.ndarray):
                sweep_results['weights_matrix'] = sweep_results['weights_matrix'].tolist()
        
        if original_results:
            if 'stresses' in original_results and isinstance(original_results['stresses'], np.ndarray):
                original_results['stresses'] = original_results['stresses'].tolist()
            if 'angles' in original_results and isinstance(original_results['angles'], np.ndarray):
                original_results['angles'] = original_results['angles'].tolist()
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'orientation_sweep',
                'software_version': '4.0.0',
                'habit_plane_angle': 54.7,
                'description': 'Orientation sweep analysis around Ag FCC twin habit plane'
            },
            'sweep_results': sweep_results,
            'original_results': original_results,
            'statistics': {}
        }
        
        # Add metadata
        for key, value in metadata.items():
            # Ensure values are JSON serializable
            if isinstance(value, (np.integer, np.int64)):
                export_data['metadata'][key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                export_data['metadata'][key] = float(value)
            elif isinstance(value, np.ndarray):
                export_data['metadata'][key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                # Recursively check list/tuple elements
                export_data['metadata'][key] = [
                    float(v) if isinstance(v, (np.floating, np.float64)) else 
                    int(v) if isinstance(v, (np.integer, np.int64)) else 
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            else:
                export_data['metadata'][key] = value
        
        # Calculate statistics
        if sweep_results and 'stresses' in sweep_results:
            stresses = np.array(sweep_results['stresses'])
            export_data['statistics']['sweep'] = {
                'max_stress': float(np.nanmax(stresses)),
                'min_stress': float(np.nanmin(stresses)),
                'mean_stress': float(np.nanmean(stresses)),
                'std_stress': float(np.nanstd(stresses)),
                'num_points': len(stresses),
                'angle_range': list(sweep_results.get('angle_range', [0, 360]))
            }
        
        if original_results and 'stresses' in original_results:
            orig_stresses = np.array(original_results['stresses'])
            export_data['statistics']['original'] = {
                'max_stress': float(np.nanmax(orig_stresses)),
                'min_stress': float(np.nanmin(orig_stresses)),
                'mean_stress': float(np.nanmean(orig_stresses)),
                'std_stress': float(np.nanstd(orig_stresses)),
                'num_points': len(orig_stresses)
            }
        
        return export_data
    
    @staticmethod
    def create_orientation_sweep_archive(sweep_results, original_results, metadata):
        """Create ZIP archive with orientation sweep results"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save sweep results
            export_data = EnhancedResultsManager.prepare_orientation_sweep_data(
                sweep_results, original_results, metadata
            )
            sweep_data = json.dumps(export_data, indent=2, default=str)
            zip_file.writestr('sweep_results.json', sweep_data)
            
            # Save metadata
            metadata_json = json.dumps(metadata, indent=2, default=str)
            zip_file.writestr('metadata.json', metadata_json)
            
            # Save CSV data
            csv_rows = []
            if 'angles' in sweep_results and 'stresses' in sweep_results:
                for angle, stress in zip(sweep_results['angles'], sweep_results['stresses']):
                    csv_rows.append({
                        'angle_deg': f"{float(angle):.3f}",
                        'angle_rad': f"{np.deg2rad(float(angle)):.6f}",
                        'stress_gpa': f"{float(stress):.6f}",
                        'region': sweep_results.get('region_type', 'unknown'),
                        'component': sweep_results.get('stress_component', 'unknown'),
                        'analysis_type': sweep_results.get('stress_type', 'unknown')
                    })
            
            if csv_rows:
                df = pd.DataFrame(csv_rows)
                csv_str = df.to_csv(index=False)
                zip_file.writestr('orientation_sweep_data.csv', csv_str)
            
            # Save attention weights if available
            if 'weights_matrix' in sweep_results:
                weights_df = pd.DataFrame(sweep_results['weights_matrix'])
                weights_csv = weights_df.to_csv(index=False, header=False)
                zip_file.writestr('attention_weights.csv', weights_csv)
            
            # Add comprehensive README
            readme = f"""# ORIENTATION SWEEP ANALYSIS RESULTS
Generated: {datetime.now().isoformat()}

## ANALYSIS DETAILS
- Target Defect: {metadata.get('defect_type', 'Twin')}
- Shape: {metadata.get('shape', 'Unknown')}
- ε*: {metadata.get('eps0', 'Unknown')}
- κ: {metadata.get('kappa', 'Unknown')}
- Region: {sweep_results.get('region_type', 'Unknown')}
- Stress Component: {sweep_results.get('stress_component', 'Unknown')}
- Analysis Type: {sweep_results.get('stress_type', 'Unknown')}

## HABIT PLANE INFORMATION
- Ag FCC Twin Habit Plane: 54.7°
- Orientation Range: {sweep_results.get('angle_range', [0, 360])}
- Number of Points: {sweep_results.get('n_points', 0)}
- Spatial Sigma: {sweep_results.get('spatial_sigma', 'Unknown')}

## FILES INCLUDED
1. sweep_results.json - Complete sweep results
2. metadata.json - Analysis metadata
3. orientation_sweep_data.csv - Tabular data for plotting
4. attention_weights.csv - Attention weights matrix

## SPATIAL LOCALITY REGULARIZATION
The interpolation uses:
- Euclidean distance in 12D parameter space
- Gaussian kernel: exp(-0.5 * (distance/sigma)²)
- Attention mechanism with {metadata.get('attention_heads', 4)} heads
- Combined weights: 70% attention + 30% spatial

## REGION DEFINITIONS
1. Defect Region: η > 0.6 (high defect concentration)
2. Interface Region: 0.4 ≤ η ≤ 0.6 (transition region)
3. Bulk Region: η < 0.4 (pure Ag material)

## VISUALIZATION NOTES
- 0° and 360° are at same position in radar charts
- Habit plane (54.7°) is highlighted in green
- Sunburst charts show stress distribution vs orientation
- Radar charts show stress magnitude at each orientation
"""
            zip_file.writestr('README_ORIENTATION_SWEEP.txt', readme)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# ENHANCED RESULTS MANAGER WITH SINTERING SUPPORT (NEW)
# =============================================

class EnhancedResultsManagerWithSintering(EnhancedResultsManager):
    """Extended results manager with sintering temperature support"""
    
    @staticmethod
    def prepare_sintering_analysis_data(solutions, region_type='bulk',
                                       sintering_calculator=None):
        """Prepare sintering analysis data for export"""
        if sintering_calculator is None:
            sintering_calculator = SinteringTemperatureCalculator()
        
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(
            solutions, region_type, 'sigma_hydro', 'max_abs'
        )
        
        sintering_data = []
        for analysis in analyses:
            stress = abs(analysis['region_stress'])
            T_sinter_exp = sintering_calculator.compute_sintering_temperature_exponential(stress)
            T_sinter_arr = sintering_calculator.compute_sintering_temperature_arrhenius(stress)
            system_info = sintering_calculator.map_system_to_temperature(stress)
            
            sintering_data.append({
                'filename': analysis['filename'],
                'orientation_deg': float(analysis['theta_deg']),
                'hydrostatic_stress_gpa': float(stress),
                'sintering_temp_exponential_k': float(T_sinter_exp),
                'sintering_temp_arrhenius_k': float(T_sinter_arr),
                'sintering_temp_celsius': float(T_sinter_exp - 273.15),
                'system_classification': system_info[0],
                'defect_type': analysis['params'].get('defect_type', 'Unknown'),
                'eps0': float(analysis['params'].get('eps0', 0)),
                'kappa': float(analysis['params'].get('kappa', 0))
            })
        
        # Calculate statistics
        if sintering_data:
            stresses = [d['hydrostatic_stress_gpa'] for d in sintering_data]
            temps = [d['sintering_temp_exponential_k'] for d in sintering_data]
            
            statistics = {
                'num_solutions': len(sintering_data),
                'mean_stress_gpa': float(np.mean(stresses)),
                'max_stress_gpa': float(np.max(stresses)),
                'min_stress_gpa': float(np.min(stresses)),
                'mean_sintering_temp_k': float(np.mean(temps)),
                'max_sintering_temp_k': float(np.max(temps)),
                'min_sintering_temp_k': float(np.min(temps)),
                'temp_range_k': float(np.max(temps) - np.min(temps)),
                'system_distribution': {
                    sys: len([d for d in sintering_data if sys in d['system_classification']])
                    for sys in ['System 1', 'System 2', 'System 3']
                }
            }
        else:
            statistics = {}
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'sintering_temperature',
                'model_parameters': {
                    'T0_k': float(sintering_calculator.T0),
                    'beta': float(sintering_calculator.beta),
                    'G_gpa': float(sintering_calculator.G),
                    'sigma_peak_gpa': float(sintering_calculator.sigma_peak),
                    'T_min_k': float(sintering_calculator.T_min)
                },
                'region_type': region_type,
                'description': 'AgNP sintering temperature analysis based on hydrostatic stress'
            },
            'sintering_data': sintering_data,
            'statistics': statistics
        }
        
        return export_data
    
    @staticmethod
    def create_sintering_analysis_report(solutions, region_type='bulk',
                                        sintering_calculator=None):
        """Create comprehensive sintering analysis report"""
        export_data = EnhancedResultsManagerWithSintering.prepare_sintering_analysis_data(
            solutions, region_type, sintering_calculator
        )
        
        # Generate markdown report
        report = f"""# AGNP SINTERING TEMPERATURE ANALYSIS REPORT

## Analysis Summary
- **Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Number of Solutions Analyzed**: {export_data['statistics'].get('num_solutions', 0)}
- **Analysis Region**: {region_type}
- **Model Used**: Stress-modified exponential model

## Model Parameters
- **Reference Temperature (T₀)**: {export_data['metadata']['model_parameters']['T0_k']} K
- **Calibration Factor (β)**: {export_data['metadata']['model_parameters']['beta']}
- **Shear Modulus (G)**: {export_data['metadata']['model_parameters']['G_gpa']} GPa
- **Peak Stress (σ_peak)**: {export_data['metadata']['model_parameters']['sigma_peak_gpa']} GPa
- **Minimum Temperature (T_min)**: {export_data['metadata']['model_parameters']['T_min_k']} K

## Key Findings
- **Average Sintering Temperature**: {export_data['statistics'].get('mean_sintering_temp_k', 0):.1f} K
- **Temperature Range**: {export_data['statistics'].get('temp_range_k', 0):.1f} K
- **Maximum Stress Observed**: {export_data['statistics'].get('max_stress_gpa', 0):.2f} GPa

## System Distribution
"""
        
        if 'system_distribution' in export_data['statistics']:
            for system, count in export_data['statistics']['system_distribution'].items():
                report += f"- **{system}**: {count} solutions\n"
        
        report += """
## Methodology
The sintering temperature is calculated using the empirical exponential model:
T_sinter(σ_h) = T₀ * exp(-β * |σ_h| / G)

Where:
- T₀ = Reference temperature at zero stress (623 K for Ag)
- β = Calibration factor (0.95)
- G = Shear modulus of Ag (30 GPa)
- |σ_h| = Absolute hydrostatic stress at habit plane (54.7°)

## Interpretation
- **System 1 (Perfect Crystal)**: σ_h < 5 GPa, T_sinter ≈ 600-630 K
- **System 2 (Stacking Faults/Twins)**: 5 GPa ≤ σ_h < 20 GPa, T_sinter ≈ 450-550 K
- **System 3 (Plastic Deformation)**: σ_h ≥ 20 GPa, T_sinter ≈ 350-400 K

## Implications for AgNP Bonding
The analysis demonstrates how defect engineering through controlled stress fields can significantly reduce sintering temperatures, enabling low-temperature AgNP bonding for advanced electronic packaging applications.

---
*Report generated by Ag FCC Twin Sintering Analysis System*
"""
        
        return export_data, report

# =============================================
# MAIN APPLICATION WITH ENHANCED VISUALIZATION SUPPORT
# =============================================

def main():
    st.set_page_config(
        page_title="Ag FCC Twin: Precise Orientation Interpolation & Sintering Analysis",
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
    .sub-header {
        font-size: 1.6rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        border-left: 5px solid #3B82F6;
        margin: 1.2rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 0.6rem;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .habit-plane-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        border: 2px solid #047857;
    }
    .sintering-card {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        border: 2px solid #B45309;
    }
    .region-card {
        border: 2px solid;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .defect-region {
        border-color: #EF4444;
        background-color: #FEE2E2;
    }
    .interface-region {
        border-color: #F59E0B;
        background-color: #FEF3C7;
    }
    .bulk-region {
        border-color: #10B981;
        background-color: #D1FAE5;
    }
    .attention-highlight {
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        color: white;
        font-weight: bold;
    }
    .heatmap-controls {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E0F2FE;
        margin: 0.5rem 0;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .system-1 {
        background-color: #D1FAE5 !important;
        color: #065F46 !important;
    }
    .system-2 {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
    }
    .system-3 {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
    }
    .latex-formula {
        font-family: "Times New Roman", serif;
        font-size: 1.1rem;
        padding: 1rem;
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with Ag FCC twin information
    st.markdown('<h1 class="main-header">🔬 Ag FCC Twin: Stress Interpolation & Sintering Temperature Analysis</h1>', unsafe_allow_html=True)
    
    # Display key equations
    st.markdown("""
    <div class="latex-formula">
    <strong>Stress-Temperature Correlation for AgNP Sintering:</strong><br>
    <div style="text-align: center; margin: 10px 0;">
    T<sub>sinter</sub>(σ<sub>h</sub>) = T₀ × exp(-β × |σ<sub>h</sub>| / G)
    </div>
    <div style="text-align: center; margin: 10px 0;">
    D = D₀ × exp[-(Q<sub>a</sub> - Ωσ<sub>h</sub>) / (k<sub>B</sub>T)]
    </div>
    <strong>Where:</strong> T₀ = 623 K, β = 0.95, G = 30 GPa, σ<sub>peak</sub> = 28.5 GPa, T<sub>min</sub> = 367 K
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <strong>🎯 Key Features:</strong><br>
    • <strong>Stress Analysis:</strong> Precise interpolation at 54.7° (Ag FCC twin habit plane) with attention-based spatial regularization<br>
    • <strong>Sintering Prediction:</strong> Temperature calculation based on hydrostatic stress using stress-modified Arrhenius model<br>
    • <strong>System Classification:</strong> Automatic mapping to AgNP systems (Perfect Crystal, Stacking Faults/Twins, Plastic Deformation)<br>
    • <strong>Visualization:</strong> Publication-quality heatmaps, sunburst charts, radar charts, and sintering temperature plots<br>
    • <strong>Physics Integration:</strong> Direct correlation between defect-induced stress and reduced sintering temperatures
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = AttentionSpatialInterpolator(
            sigma=0.3, 
            use_numba=True, 
            attention_dim=32, 
            num_heads=4
        )
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'original_analyzer' not in st.session_state:
        st.session_state.original_analyzer = OriginalFileAnalyzer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManagerWithSintering()
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = EnhancedHeatmapVisualizer()
    if 'sintering_calculator' not in st.session_state:
        st.session_state.sintering_calculator = SinteringTemperatureCalculator()
    if 'sintering_visualizer' not in st.session_state:
        st.session_state.sintering_visualizer = EnhancedSinteringVisualizer(
            st.session_state.sintering_calculator
        )
    
    # Sidebar with comprehensive options
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Analysis Configuration</h2>', unsafe_allow_html=True)
        
        # Habit plane information
        st.markdown('<div class="habit-plane-card">', unsafe_allow_html=True)
        st.markdown("### Ag FCC Twin Habit Plane")
        st.write("**Preferred Orientation:** 54.7°")
        st.write("**Crystal System:** Face-Centered Cubic")
        st.write("**Defect Type:** Coherent Twin Boundary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sintering temperature information
        st.markdown('<div class="sintering-card">', unsafe_allow_html=True)
        st.markdown("### Sintering Temperature Model")
        st.write("**T₀ (σ=0):** 623 K (350°C)")
        st.write("**T_min (peak σ):** 367 K (94°C)")
        st.write("**Peak Stress:** 28.5 GPa")
        st.write("**Calibration β:** 0.95")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data source selection
        st.markdown("#### 📊 Data Source & Analysis")
        analysis_mode = st.radio(
            "Select analysis mode:",
            ["Precise Single Orientation", "Orientation Sweep", 
             "Compare Original vs Interpolated", "Heatmap Analysis",
             "Sintering Temperature Analysis"],
            index=0,
            help="Choose between stress interpolation or sintering temperature analysis"
        )
        
        # Region selection
        st.markdown("#### 🎯 Analysis Region")
        region_type_display = st.selectbox(
            "Select region for stress analysis:",
            ["Defect Region (η > 0.6)", "Interface Region (0.4 ≤ η ≤ 0.6)", "Bulk Ag Material (η < 0.4)"],
            index=2,
            help="Select the material region to analyze"
        )
        
        # Map region name to key
        region_map = {
            "Defect Region (η > 0.6)": "defect",
            "Interface Region (0.4 ≤ η ≤ 0.6)": "interface",
            "Bulk Ag Material (η < 0.4)": "bulk"
        }
        region_key = region_map[region_type_display]
        
        # Stress component
        st.markdown("#### 📈 Stress Component")
        stress_component = st.selectbox(
            "Select stress component:",
            ["von_mises", "sigma_hydro", "sigma_mag"],
            index=1 if analysis_mode == "Sintering Temperature Analysis" else 0,
            help="Select which stress component to visualize (sigma_hydro for sintering)"
        )
        
        # Stress type
        stress_type = st.selectbox(
            "Select stress analysis type:",
            ["max_abs", "mean_abs", "max", "min", "mean"],
            index=0,
            help="Select how to analyze stress in the region"
        )
        
        # Interpolator settings
        if analysis_mode != "Sintering Temperature Analysis":
            st.markdown("#### 🧠 Interpolator Settings")
            use_spatial = st.checkbox(
                "Use Spatial Locality Regularization", 
                value=True,
                help="Use Euclidean distance in parameter space for regularization"
            )
            
            spatial_sigma = st.slider(
                "Spatial Sigma (σ)",
                0.05, 1.0, 0.3, 0.05,
                help="Controls the influence of spatial distance in Gaussian kernel"
            )
            
            attention_heads = st.slider(
                "Attention Heads",
                1, 8, 4, 1,
                help="Number of attention heads in transformer-like mechanism"
            )
            
            # Update interpolator settings
            if (spatial_sigma != st.session_state.interpolator.sigma or
                attention_heads != st.session_state.interpolator.num_heads):
                st.session_state.interpolator.sigma = spatial_sigma
                st.session_state.interpolator.num_heads = attention_heads
        
        # Sintering temperature parameters
        if analysis_mode == "Sintering Temperature Analysis":
            st.markdown("#### 🔥 Sintering Temperature Parameters")
            
            col_sint1, col_sint2 = st.columns(2)
            with col_sint1:
                T0 = st.number_input(
                    "T₀ (reference temperature at σ=0)", 
                    min_value=300.0, max_value=1000.0, value=623.0, step=1.0,
                    help="Reference sintering temperature at zero stress"
                )
                T_min = st.number_input(
                    "T_min (minimum temperature at peak stress)",
                    min_value=300.0, max_value=1000.0, value=367.0, step=1.0,
                    help="Minimum sintering temperature at peak hydrostatic stress"
                )
            
            with col_sint2:
                beta = st.number_input(
                    "β (calibration factor)",
                    min_value=0.1, max_value=2.0, value=0.95, step=0.01,
                    help="Dimensionless calibration factor"
                )
                G = st.number_input(
                    "G (shear modulus, GPa)",
                    min_value=10.0, max_value=100.0, value=30.0, step=1.0,
                    help="Shear modulus of Ag for stress normalization"
                )
            
            # Sintering analysis type
            sintering_type = st.radio(
                "Sintering Analysis Type",
                ["Single Solution Analysis", "Bulk Analysis", "Orientation Sweep", 
                 "Theoretical Curve", "System Mapping", "Defect Comparison Radar"],
                horizontal=False,
                help="Select type of sintering analysis"
            )
        
        # Orientation settings based on mode
        if analysis_mode == "Precise Single Orientation":
            st.markdown("#### 🎯 Single Orientation Target")
            target_angle = st.number_input(
                "Target Orientation (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=54.7,
                step=0.1,
                format="%.2f",
                help="Enter precise orientation angle (e.g., 54.70° for Ag FCC twin)"
            )
            
            # Show habit plane reminder
            if abs(target_angle - 54.7) < 0.1:
                st.success(f"✅ Targeting Ag FCC twin habit plane ({target_angle}°)")
        
        elif analysis_mode == "Orientation Sweep":
            st.markdown("#### 🌐 Orientation Sweep Range")
            col_sweep1, col_sweep2 = st.columns(2)
            with col_sweep1:
                min_angle = st.number_input(
                    "Min Angle (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=53.12,
                    step=0.1,
                    format="%.2f"
                )
            with col_sweep2:
                max_angle = st.number_input(
                    "Max Angle (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=56.82,
                    step=0.1,
                    format="%.2f"
                )
            
            n_points = st.slider(
                "Number of Points",
                10, 200, 100, 10,
                help="Number of orientation points in sweep"
            )
            
            # Validate range
            if min_angle >= max_angle:
                st.error("Min angle must be less than max angle")
        
        elif analysis_mode == "Compare Original vs Interpolated":
            st.markdown("#### 🔄 Comparison Settings")
            comparison_range = st.slider(
                "Orientation Range for Comparison (°)",
                0, 360, (0, 360), 5
            )
            
            n_comparison_points = st.slider(
                "Comparison Points",
                10, 100, 50, 5
            )
        
        elif analysis_mode == "Heatmap Analysis":
            st.markdown("#### 🔥 Heatmap Settings")
            heatmap_source = st.selectbox(
                "Heatmap Source",
                ["Top Contributing Solution", "Specific Solution"],
                index=0,
                help="Select which solution to use for heatmap generation"
            )
            
            heatmap_cmap = st.selectbox(
                "Heatmap Color Map",
                ALL_COLORMAPS,
                index=ALL_COLORMAPS.index('viridis') if 'viridis' in ALL_COLORMAPS else 0
            )
            
            # Heatmap aspect ratio control
            maintain_aspect = st.checkbox(
                "Maintain Perfect Aspect Ratio",
                value=True,
                help="Keep heatmap cells perfectly square"
            )
            
            if heatmap_source == "Specific Solution":
                if st.session_state.solutions:
                    solution_names = [f"{i+1}. {sol.get('filename', 'Unknown')}" 
                                     for i, sol in enumerate(st.session_state.solutions)]
                    selected_solution = st.selectbox(
                        "Select Solution",
                        solution_names,
                        index=0
                    )
        
        elif analysis_mode == "Sintering Temperature Analysis":
            if sintering_type == "Orientation Sweep":
                st.markdown("#### 🌐 Orientation Sweep Range")
                col_sweep1, col_sweep2 = st.columns(2)
                with col_sweep1:
                    min_angle = st.number_input(
                        "Min Angle (°)",
                        min_value=0.0,
                        max_value=360.0,
                        value=0.0,
                        step=1.0,
                        format="%.1f"
                    )
                with col_sweep2:
                    max_angle = st.number_input(
                        "Max Angle (°)",
                        min_value=0.0,
                        max_value=360.0,
                        value=360.0,
                        step=1.0,
                        format="%.1f"
                    )
                
                n_points = st.slider(
                    "Number of Points",
                    20, 200, 100, 10,
                    help="Number of orientation points in sintering sweep"
                )
            
            elif sintering_type == "Single Solution Analysis":
                if st.session_state.solutions:
                    solution_names = [f"{i+1}. {sol.get('filename', 'Unknown')}" 
                                     for i, sol in enumerate(st.session_state.solutions)]
                    selected_solution = st.selectbox(
                        "Select Solution for Analysis",
                        solution_names,
                        index=0
                    )
            
            elif sintering_type == "Defect Comparison Radar":
                st.markdown("#### 🎯 Defect Types for Comparison")
                defect_types = st.multiselect(
                    "Select defect types to compare:",
                    ["ISF", "ESF", "Twin", "No Defect"],
                    default=["ISF", "ESF", "Twin", "No Defect"]
                )
                
                col_angle1, col_angle2 = st.columns(2)
                with col_angle1:
                    radar_min_angle = st.number_input(
                        "Radar Min Angle (°)",
                        min_value=0.0,
                        max_value=360.0,
                        value=0.0,
                        step=1.0,
                        format="%.1f"
                    )
                with col_angle2:
                    radar_max_angle = st.number_input(
                        "Radar Max Angle (°)",
                        min_value=0.0,
                        max_value=360.0,
                        value=360.0,
                        step=1.0,
                        format="%.1f"
                    )
                
                n_radar_points = st.slider(
                    "Radar Points",
                    10, 100, 36, 1,
                    help="Number of orientation points for radar chart"
                )
        
        # Target parameters
        st.markdown("#### ⚙️ Target Parameters")
        defect_type = st.selectbox("Defect Type", ["ISF", "ESF", "Twin", "No Defect"], 
                                 index=2 if analysis_mode != "Sintering Temperature Analysis" else 3)
        
        col_shape, col_eps = st.columns(2)
        with col_shape:
            shape = st.selectbox("Shape", 
                                ["Square", "Horizontal Fault", "Vertical Fault", 
                                 "Rectangle", "Ellipse"], index=0)
        with col_eps:
            eps0 = st.slider("ε*", 0.3, 3.0, 0.707, 0.01)
        
        col_kappa, col_theta = st.columns(2)
        with col_kappa:
            kappa = st.slider("κ", 0.1, 2.0, 0.6, 0.01)
        
        # Visualization settings
        if analysis_mode != "Sintering Temperature Analysis":
            st.markdown("#### 🎨 Visualization")
            viz_type = st.radio(
                "Chart Type",
                ["Sunburst", "Radar", "Both", "Comparison"],
                index=1,
                horizontal=True
            )
            
            # Orientation range for charts
            st.markdown("#### 📐 Chart Orientation Range")
            use_custom_range = st.checkbox(
                "Use Custom Orientation Range for Charts",
                value=False,
                help="Set custom start and end angles for sunburst/radar charts"
            )
            
            if use_custom_range:
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    chart_min_angle = st.number_input(
                        "Chart Min Angle (°)",
                        min_value=0.0,
                        max_value=360.0,
                        value=0.0,
                        step=1.0,
                        format="%.1f"
                    )
                with col_chart2:
                    chart_max_angle = st.number_input(
                        "Chart Max Angle (°)",
                        min_value=0.0,
                        max_value=360.0,
                        value=360.0,
                        step=1.0,
                        format="%.1f"
                    )
                chart_angle_range = (float(chart_min_angle), float(chart_max_angle))
            else:
                chart_angle_range = None
            
            cmap = st.selectbox(
                "Color Map",
                ALL_COLORMAPS,
                index=ALL_COLORMAPS.index('rainbow') if 'rainbow' in ALL_COLORMAPS else 0
            )
        
        # Load solutions
        st.markdown("#### 📂 Load Solutions")
        if st.button("🔄 Load All Solutions", use_container_width=True, type="primary"):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions(use_cache=True)
                if st.session_state.solutions:
                    st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions loaded. Check the directory for solution files.")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📋 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                # Orientation distribution
                orientations = st.session_state.original_analyzer.create_orientation_distribution(
                    st.session_state.solutions
                )
                
                if len(orientations) > 0:
                    fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
                    ax_dist.hist(orientations, bins=20, edgecolor='black', alpha=0.7)
                    ax_dist.set_xlabel('Orientation (°)')
                    ax_dist.set_ylabel('Count')
                    ax_dist.set_title('Orientation Distribution in Loaded Solutions')
                    ax_dist.axvline(54.7, color='green', linestyle='--', label='Habit Plane (54.7°)')
                    ax_dist.legend()
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
                
                # Solutions summary
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    params = sol.get('params', {})
                    theta = params.get('theta', 0.0)
                    theta_deg = np.rad2deg(theta) if theta is not None else 0.0
                    
                    st.write(f"**{i+1}. {sol.get('filename', 'Unknown')}**")
                    st.caption(f"Type: {params.get('defect_type', '?')} | "
                              f"θ: {theta_deg:.1f}° | "
                              f"ε*: {params.get('eps0', 0):.2f} | "
                              f"Frames: {len(sol.get('history', []))}")
                
                if len(st.session_state.solutions) > 5:
                    st.info(f"... and {len(st.session_state.solutions) - 5} more")
    
    # Main content area
    col_main1, col_main2 = st.columns([3, 1])
    
    with col_main1:
        if analysis_mode == "Sintering Temperature Analysis":
            st.markdown('<h2 class="sub-header">🔥 Sintering Temperature Analysis</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 class="sub-header">🚀 Stress Interpolation Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.solutions:
            st.warning("⚠️ Please load solutions first using the button in the sidebar.")
            
            with st.expander("📁 Directory Information", expanded=False):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.success(f"✅ Found {total_files} files in {SOLUTIONS_DIR}")
                    for fmt, files in file_formats.items():
                        if files:
                            st.info(f"• **{fmt.upper()}**: {len(files)} files")
                else:
                    st.error(f"❌ No files found in {SOLUTIONS_DIR}")
        
        else:
            # Create target parameters
            target_params = {
                'defect_type': defect_type,
                'shape': shape,
                'eps0': eps0,
                'kappa': kappa,
                'theta': 0.0  # Will be set based on mode
            }
            
            # Update sintering calculator parameters if changed
            if analysis_mode == "Sintering Temperature Analysis":
                if (T0 != st.session_state.sintering_calculator.T0 or
                    beta != st.session_state.sintering_calculator.beta or
                    G != st.session_state.sintering_calculator.G or
                    T_min != st.session_state.sintering_calculator.T_min):
                    
                    st.session_state.sintering_calculator.T0 = T0
                    st.session_state.sintering_calculator.beta = beta
                    st.session_state.sintering_calculator.G = G
                    st.session_state.sintering_calculator.T_min = T_min
                    st.session_state.sintering_calculator.sigma_peak = (
                        st.session_state.sintering_calculator.compute_peak_stress_from_temperature()
                    )
                    st.session_state.sintering_visualizer.sintering_calculator = st.session_state.sintering_calculator
            
            # Generate analysis button
            generate_text = "✨ Generate Sintering Analysis" if analysis_mode == "Sintering Temperature Analysis" else "✨ Generate Analysis"
            if st.button(generate_text, type="primary", use_container_width=True):
                with st.spinner(f"Generating {analysis_mode} analysis..."):
                    try:
                        if analysis_mode == "Precise Single Orientation":
                            # Single precise orientation interpolation
                            st.info(f"🔬 Interpolating at precise orientation: {target_angle}°")
                            
                            result = st.session_state.interpolator.interpolate_precise_orientation(
                                st.session_state.solutions,
                                float(target_angle),
                                target_params,
                                region_key,
                                stress_component,
                                stress_type,
                                use_spatial=use_spatial
                            )
                            
                            if result:
                                st.session_state.single_result = result
                                
                                # Display results
                                st.success(f"✅ Interpolation complete at {target_angle}°")
                                
                                # Show detailed metrics
                                col_met1, col_met2, col_met3 = st.columns(3)
                                with col_met1:
                                    st.metric(
                                        "Region Stress",
                                        f"{result['region_stress']:.4f} GPa",
                                        delta="Interpolated Value"
                                    )
                                with col_met2:
                                    st.metric(
                                        "Spatial Sigma",
                                        f"{result['spatial_sigma']:.3f}",
                                        delta="Regularization Strength"
                                    )
                                with col_met3:
                                    st.metric(
                                        "Number of Sources",
                                        result['num_sources'],
                                        delta="Used for Interpolation"
                                    )
                                
                                # Show attention weights
                                with st.expander("🔍 Attention Weights Analysis", expanded=True):
                                    weights = result['attention_weights']
                                    
                                    if weights:
                                        # Create weights visualization
                                        fig_weights, ax_weights = plt.subplots(figsize=(10, 4))
                                        bars = ax_weights.bar(range(len(weights)), weights, 
                                                             color=plt.cm.viridis(np.array(weights)/max(weights)))
                                        ax_weights.set_xlabel('Source Index')
                                        ax_weights.set_ylabel('Attention Weight')
                                        ax_weights.set_title('Attention Weights Distribution')
                                        ax_weights.grid(True, alpha=0.3)
                                        
                                        # Add value labels
                                        for i, (bar, weight) in enumerate(zip(bars, weights)):
                                            height = bar.get_height()
                                            ax_weights.text(bar.get_x() + bar.get_width()/2., height,
                                                           f'{weight:.3f}', ha='center', va='bottom', 
                                                           fontsize=8)
                                        
                                        st.pyplot(fig_weights)
                                        plt.close(fig_weights)
                                        
                                        # Show top contributors
                                        weights_array = np.array(weights)
                                        if len(weights_array) > 0:
                                            top_indices = np.argsort(weights_array)[-5:][::-1]
                                            st.write("**Top 5 Contributing Sources:**")
                                            for idx in top_indices:
                                                sol = st.session_state.solutions[idx]
                                                params = sol.get('params', {})
                                                theta = params.get('theta', 0.0)
                                                theta_deg = np.rad2deg(theta) if theta is not None else 0.0
                                                st.write(f"- Source {idx+1}: θ={theta_deg:.1f}°, "
                                                        f"ε*={params.get('eps0', 0):.2f}, "
                                                        f"weight={weights[idx]:.3f}")
                                    else:
                                        st.info("No attention weights available")
                        
                        elif analysis_mode == "Orientation Sweep":
                            # Orientation sweep analysis
                            st.info(f"🌐 Performing orientation sweep from {min_angle}° to {max_angle}°")
                            
                            sweep_result = st.session_state.interpolator.create_orientation_sweep(
                                st.session_state.solutions,
                                target_params,
                                (float(min_angle), float(max_angle)),
                                n_points,
                                region_key,
                                stress_component,
                                stress_type
                            )
                            
                            if sweep_result:
                                st.session_state.sweep_result = sweep_result
                                
                                # Get original solutions for comparison
                                original_stresses, original_angles = st.session_state.original_analyzer.create_original_sweep_matrix(
                                    st.session_state.solutions,
                                    (float(min_angle), float(max_angle)),
                                    n_points,
                                    region_key,
                                    stress_component,
                                    stress_type
                                )
                                
                                if original_stresses is not None:
                                    st.session_state.original_sweep = {
                                        'stresses': original_stresses.tolist() if isinstance(original_stresses, np.ndarray) else original_stresses,
                                        'angles': original_angles.tolist() if isinstance(original_angles, np.ndarray) else original_angles,
                                        'region_type': region_key,
                                        'stress_component': stress_component,
                                        'stress_type': stress_type
                                    }
                                
                                st.success(f"✅ Generated sweep with {n_points} points")
                                
                                # Display sweep statistics
                                with st.expander("📊 Sweep Statistics", expanded=True):
                                    if sweep_result['stresses']:
                                        stresses_array = np.array(sweep_result['stresses'])
                                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                        with col_stat1:
                                            st.metric("Max Stress", f"{np.nanmax(stresses_array):.4f} GPa")
                                        with col_stat2:
                                            st.metric("Min Stress", f"{np.nanmin(stresses_array):.4f} GPa")
                                        with col_stat3:
                                            st.metric("Mean Stress", f"{np.nanmean(stresses_array):.4f} GPa")
                                        with col_stat4:
                                            st.metric("Std Dev", f"{np.nanstd(stresses_array):.4f} GPa")
                                
                                # Find stress at habit plane
                                habit_angle = 54.7
                                if habit_angle >= min_angle and habit_angle <= max_angle and sweep_result['stresses']:
                                    idx = np.argmin(np.abs(np.array(sweep_result['angles']) - habit_angle))
                                    habit_stress = sweep_result['stresses'][idx]
                                    
                                    st.markdown(f'<div class="attention-highlight">', unsafe_allow_html=True)
                                    st.write(f"**Habit Plane (54.7°) Stress:** {habit_stress:.4f} GPa")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        
                        elif analysis_mode == "Compare Original vs Interpolated":
                            # Comparison analysis
                            st.info("🔄 Comparing original vs interpolated solutions")
                            
                            # Generate interpolated sweep
                            sweep_result = st.session_state.interpolator.create_orientation_sweep(
                                st.session_state.solutions,
                                target_params,
                                comparison_range,
                                n_comparison_points,
                                region_key,
                                stress_component,
                                stress_type
                            )
                            
                            # Get original solutions
                            original_stresses, original_angles = st.session_state.original_analyzer.create_original_sweep_matrix(
                                st.session_state.solutions,
                                comparison_range,
                                n_comparison_points,
                                region_key,
                                stress_component,
                                stress_type
                            )
                            
                            if sweep_result and original_stresses is not None:
                                st.session_state.comparison_data = {
                                    'interpolated': sweep_result,
                                    'original': {
                                        'stresses': original_stresses.tolist() if isinstance(original_stresses, np.ndarray) else original_stresses,
                                        'angles': original_angles.tolist() if isinstance(original_angles, np.ndarray) else original_angles
                                    }
                                }
                                
                                st.success("✅ Generated comparison data")
                                
                                # Calculate comparison metrics
                                with st.expander("📊 Comparison Metrics", expanded=True):
                                    # Interpolate original to same grid
                                    if sweep_result['stresses'] and original_stresses is not None:
                                        interp_stresses = np.array(sweep_result['stresses'])
                                        interp_angles = np.array(sweep_result['angles'])
                                        
                                        # Filter valid original data
                                        valid_mask = ~np.isnan(original_stresses)
                                        if np.any(valid_mask):
                                            valid_orig = original_stresses[valid_mask]
                                            valid_angles = original_angles[valid_mask]
                                            
                                            # Interpolate original to interpolated grid
                                            orig_on_interp_grid = np.interp(
                                                interp_angles,
                                                valid_angles,
                                                valid_orig
                                            )
                                            
                                            # Calculate metrics
                                            mae = np.mean(np.abs(orig_on_interp_grid - interp_stresses))
                                            rmse = np.sqrt(np.mean((orig_on_interp_grid - interp_stresses) ** 2))
                                            r2 = 1 - np.sum((interp_stresses - orig_on_interp_grid) ** 2) / (
                                                np.sum((orig_on_interp_grid - np.mean(orig_on_interp_grid)) ** 2) + 1e-10
                                            )
                                            
                                            col_met1, col_met2, col_met3 = st.columns(3)
                                            with col_met1:
                                                st.metric("MAE", f"{mae:.4f} GPa")
                                            with col_met2:
                                                st.metric("RMSE", f"{rmse:.4f} GPa")
                                            with col_met3:
                                                st.metric("R² Score", f"{r2:.4f}")
                                            
                                            st.write("**Interpretation:**")
                                            if r2 > 0.9:
                                                st.success("Excellent agreement between original and interpolated")
                                            elif r2 > 0.7:
                                                st.info("Good agreement between original and interpolated")
                                            else:
                                                st.warning("Moderate agreement - consider adjusting interpolation parameters")
                                        else:
                                            st.warning("Insufficient original data for comparison")
                                    else:
                                        st.warning("Missing data for comparison metrics")
                        
                        elif analysis_mode == "Heatmap Analysis":
                            # Heatmap analysis
                            st.info("🔥 Generating stress distribution heatmaps")
                            
                            if heatmap_source == "Top Contributing Solution":
                                # Get single target result first
                                if 'single_result' not in st.session_state:
                                    # Run single interpolation to get top contributor
                                    result = st.session_state.interpolator.interpolate_precise_orientation(
                                        st.session_state.solutions,
                                        54.7,  # Default to habit plane
                                        target_params,
                                        region_key,
                                        stress_component,
                                        stress_type,
                                        use_spatial=use_spatial
                                    )
                                    
                                    if result and result['attention_weights']:
                                        weights = result['attention_weights']
                                        if len(weights) > 0:
                                            top_idx = np.argmax(weights)
                                            st.session_state.top_solution = st.session_state.solutions[top_idx]
                                            st.success(f"✅ Using top contributing solution (weight: {weights[top_idx]:.3f})")
                                        else:
                                            st.error("No attention weights available")
                                    else:
                                        st.error("Failed to identify top contributing solution")
                                        return
                                else:
                                    # Use existing single result
                                    weights = st.session_state.single_result['attention_weights']
                                    if weights:
                                        top_idx = np.argmax(weights)
                                        st.session_state.top_solution = st.session_state.solutions[top_idx]
                            
                            elif heatmap_source == "Specific Solution":
                                if st.session_state.solutions:
                                    solution_idx = int(selected_solution.split('.')[0]) - 1
                                    st.session_state.top_solution = st.session_state.solutions[solution_idx]
                            
                            # Store heatmap settings
                            st.session_state.heatmap_settings = {
                                'cmap': heatmap_cmap,
                                'maintain_aspect': maintain_aspect
                            }
                        
                        elif analysis_mode == "Sintering Temperature Analysis":
                            # Sintering temperature analysis
                            if sintering_type == "Single Solution Analysis":
                                st.info("🔥 Analyzing sintering temperature for single solution")
                                
                                solution_idx = int(selected_solution.split('.')[0]) - 1
                                solution = st.session_state.solutions[solution_idx]
                                
                                # Analyze solution
                                analysis = st.session_state.original_analyzer.analyze_solution(
                                    solution, region_key, 'sigma_hydro', 'max_abs'
                                )
                                
                                if analysis:
                                    stress = abs(analysis['region_stress'])
                                    T_sinter = st.session_state.sintering_calculator.compute_sintering_temperature_exponential(stress)
                                    T_sinter_arr = st.session_state.sintering_calculator.compute_sintering_temperature_arrhenius(stress)
                                    system_info = st.session_state.sintering_calculator.map_system_to_temperature(stress)
                                    
                                    st.session_state.single_sintering_result = {
                                        'analysis': analysis,
                                        'stress': stress,
                                        'T_sinter_exp': T_sinter,
                                        'T_sinter_arr': T_sinter_arr,
                                        'system_info': system_info
                                    }
                                    
                                    st.success(f"✅ Sintering analysis complete")
                            
                            elif sintering_type == "Bulk Analysis":
                                st.info("🔥 Analyzing sintering temperatures for all loaded solutions")
                                
                                # Create comprehensive dashboard
                                fig = st.session_state.sintering_visualizer.create_comprehensive_sintering_dashboard(
                                    st.session_state.solutions, region_key, 'sigma_hydro', 'max_abs'
                                )
                                
                                if fig:
                                    st.session_state.sintering_bulk_fig = fig
                                    st.success("✅ Generated comprehensive sintering dashboard")
                                
                                # Create interactive plot
                                fig_interactive = st.session_state.sintering_visualizer.create_interactive_sintering_plot(
                                    st.session_state.solutions, region_key, 'sigma_hydro', 'max_abs'
                                )
                                
                                if fig_interactive:
                                    st.session_state.sintering_interactive_fig = fig_interactive
                            
                            elif sintering_type == "Orientation Sweep":
                                st.info(f"🔥 Performing sintering temperature sweep from {min_angle}° to {max_angle}°")
                                
                                # Generate sintering temperature sweep
                                fig = st.session_state.sintering_visualizer.create_sintering_temperature_sweep(
                                    st.session_state.solutions,
                                    target_params,
                                    (min_angle, max_angle),
                                    region_key,
                                    'sigma_hydro',
                                    'max_abs',
                                    n_points
                                )
                                
                                if fig:
                                    st.session_state.sintering_sweep_fig = fig
                                    st.success(f"✅ Generated sintering temperature sweep")
                            
                            elif sintering_type == "Theoretical Curve":
                                st.info("📈 Displaying theoretical sintering temperature curve")
                                
                                # Generate theoretical curve
                                theory_data = st.session_state.sintering_calculator.get_theoretical_curve()
                                fig = st.session_state.sintering_calculator.create_sintering_plot(
                                    theory_data['stresses'],
                                    theory_data['T_exponential'],
                                    title="Theoretical Sintering Temperature vs Hydrostatic Stress"
                                )
                                
                                if fig:
                                    st.session_state.sintering_theory_fig = fig
                                    st.success("✅ Generated theoretical curve")
                            
                            elif sintering_type == "System Mapping":
                                st.info("🗺️ Mapping solutions to AgNP system classification")
                                
                                # Analyze all solutions
                                analyzer = OriginalFileAnalyzer()
                                analyses = analyzer.analyze_all_solutions(
                                    st.session_state.solutions, region_key, 'sigma_hydro', 'max_abs'
                                )
                                
                                if analyses:
                                    # Create classification table
                                    system_data = []
                                    for analysis in analyses:
                                        stress = abs(analysis['region_stress'])
                                        T_sinter = st.session_state.sintering_calculator.compute_sintering_temperature_exponential(stress)
                                        system_info = st.session_state.sintering_calculator.map_system_to_temperature(stress)
                                        
                                        system_data.append({
                                            'Filename': analysis['filename'],
                                            'Orientation (°)': f"{analysis['theta_deg']:.1f}",
                                            '|σ_h| (GPa)': f"{stress:.3f}",
                                            'T_sinter (K)': f"{T_sinter:.1f}",
                                            'T_sinter (°C)': f"{T_sinter-273.15:.1f}",
                                            'System': system_info[0],
                                            'Defect Type': analysis['params'].get('defect_type', 'Unknown')
                                        })
                                    
                                    # Store system data
                                    st.session_state.system_mapping_data = pd.DataFrame(system_data)
                                    st.success(f"✅ Mapped {len(analyses)} solutions to AgNP systems")
                            
                            elif sintering_type == "Defect Comparison Radar":
                                st.info("📡 Creating sintering temperature radar comparison across defect types")
                                
                                # Create angles for radar
                                angles = np.linspace(radar_min_angle, radar_max_angle, n_radar_points)
                                
                                # Create radar chart
                                fig = st.session_state.visualizer.create_sintering_temperature_radar(
                                    defect_types,
                                    angles,
                                    st.session_state.sintering_calculator,
                                    title="Sintering Temperature Prediction: Defect Type Comparison",
                                    angle_range=(radar_min_angle, radar_max_angle)
                                )
                                
                                if fig:
                                    st.session_state.sintering_radar_fig = fig
                                    st.success(f"✅ Generated sintering temperature radar for {len(defect_types)} defect types")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)
            
            # Generate visualizations based on analysis mode
            if analysis_mode == "Precise Single Orientation" and 'single_result' in st.session_state:
                st.markdown('<h3 class="sub-header">📊 Visualization</h3>', unsafe_allow_html=True)
                
                result = st.session_state.single_result
                
                if viz_type in ["Radar", "Both"]:
                    # Create a simple radar chart showing the single point
                    angles = np.array([result['target_angle_deg']])
                    stresses = np.array([result['region_stress']])
                    
                    # Add neighboring points for context
                    context_angles = np.array([angles[0] - 5, angles[0], angles[0] + 5])
                    context_stresses = np.array([0, result['region_stress'], 0])
                    
                    fig_radar = st.session_state.visualizer.create_enhanced_plotly_radar(
                        context_stresses, context_angles,
                        f"{stress_component} at {result['target_angle_deg']:.2f}°",
                        title=f"Precise Orientation Analysis: {region_type_display}",
                        show_habit_plane=True,
                        angle_range=chart_angle_range
                    )
                    if fig_radar:
                        st.plotly_chart(fig_radar, use_container_width=True)
                    else:
                        st.warning("Could not create radar chart")
                
                # Display detailed information
                with st.expander("📋 Detailed Analysis", expanded=False):
                    st.write(f"**Target Parameters:**")
                    st.json(result['target_params'])
                    
                    st.write(f"**Analysis Details:**")
                    st.write(f"- Region: {region_type_display}")
                    st.write(f"- Stress Component: {stress_component}")
                    st.write(f"- Analysis Type: {stress_type}")
                    st.write(f"- Spatial Sigma: {result['spatial_sigma']}")
                    st.write(f"- Number of Sources: {result['num_sources']}")
                    
                    # Export options
                    st.markdown("#### 📤 Export Results")
                    if st.button("💾 Export Single Point Analysis", use_container_width=True):
                        metadata = {
                            'defect_type': defect_type,
                            'shape': shape,
                            'eps0': eps0,
                            'kappa': kappa,
                            'target_angle': result['target_angle_deg'],
                            'region_type': region_key,
                            'stress_component': stress_component,
                            'stress_type': stress_type,
                            'spatial_sigma': spatial_sigma,
                            'attention_heads': attention_heads
                        }
                        
                        export_data = st.session_state.results_manager.prepare_orientation_sweep_data(
                            {'angles': [result['target_angle_deg']], 
                             'stresses': [result['region_stress']],
                             'region_type': region_key,
                             'stress_component': stress_component,
                             'stress_type': stress_type},
                            metadata=metadata
                        )
                        
                        json_str = json.dumps(export_data, indent=2, default=str)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📥 Download JSON",
                            data=json_str,
                            file_name=f"single_point_{result['target_angle_deg']:.1f}deg_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            
            elif analysis_mode == "Orientation Sweep" and 'sweep_result' in st.session_state:
                st.markdown('<h3 class="sub-header">📊 Sweep Visualization</h3>', unsafe_allow_html=True)
                
                sweep = st.session_state.sweep_result
                
                # Line plot of stress vs orientation
                if sweep['stresses'] and sweep['angles']:
                    fig_line, ax_line = plt.subplots(figsize=(12, 6))
                    ax_line.plot(sweep['angles'], sweep['stresses'], 'b-', linewidth=3, label='Interpolated')
                    
                    # Add original data if available
                    if 'original_sweep' in st.session_state:
                        orig = st.session_state.original_sweep
                        if isinstance(orig['stresses'], list):
                            orig_stresses = np.array(orig['stresses'])
                            orig_angles = np.array(orig['angles'])
                        else:
                            orig_stresses = orig['stresses']
                            orig_angles = orig['angles']
                        
                        valid_mask = ~np.isnan(orig_stresses)
                        if np.any(valid_mask):
                            ax_line.scatter(orig_angles[valid_mask], orig_stresses[valid_mask],
                                           color='red', s=50, label='Original Solutions', zorder=5)
                    
                    ax_line.axvline(54.7, color='green', linestyle='--', linewidth=2, 
                                   label='Habit Plane (54.7°)', alpha=0.7)
                    
                    ax_line.set_xlabel('Orientation (°)', fontsize=12, fontweight='bold')
                    ax_line.set_ylabel(f'{stress_component.replace("_", " ").title()} Stress (GPa)', 
                                      fontsize=12, fontweight='bold')
                    ax_line.set_title(f'Orientation Sweep: {region_type_display}', 
                                     fontsize=14, fontweight='bold', pad=20)
                    ax_line.grid(True, alpha=0.3)
                    ax_line.legend(fontsize=11)
                    ax_line.set_xlim([sweep['angles'][0], sweep['angles'][-1]])
                    
                    st.pyplot(fig_line)
                    plt.close(fig_line)
                else:
                    st.warning("No sweep data available for line plot")
                
                # Generate sunburst and radar charts
                if viz_type in ["Sunburst", "Both"]:
                    st.markdown("#### 🌅 Sunburst Visualization")
                    
                    # Ensure we have valid data for sunburst
                    if sweep['stresses'] and sweep['angles']:
                        # Create sunburst for sweep (single time point)
                        fig_sunburst = st.session_state.visualizer.create_enhanced_plotly_sunburst(
                            np.array(sweep['stresses']),
                            np.zeros(1),  # Single time point
                            np.array(sweep['angles']),
                            title=f"Orientation Sweep: {region_type_display} - {stress_component}",
                            cmap=cmap,
                            is_time_series=False,
                            angle_range=chart_angle_range
                        )
                        if fig_sunburst:
                            st.plotly_chart(fig_sunburst, use_container_width=True)
                        else:
                            st.warning("Could not create sunburst visualization")
                    else:
                        st.warning("Insufficient data for sunburst visualization")
                
                if viz_type in ["Radar", "Both"]:
                    st.markdown("#### 📡 Radar Visualization")
                    
                    # Ensure we have valid data for radar
                    if sweep['stresses'] and sweep['angles']:
                        fig_radar = st.session_state.visualizer.create_enhanced_plotly_radar(
                            np.array(sweep['stresses']), np.array(sweep['angles']),
                            f"{stress_component} - {region_type_display}",
                            title=f"Radar Chart: {region_type_display} Stress vs Orientation",
                            show_habit_plane=True,
                            angle_range=chart_angle_range
                        )
                        if fig_radar:
                            st.plotly_chart(fig_radar, use_container_width=True)
                        else:
                            st.warning("Could not create radar visualization")
                    else:
                        st.warning("Insufficient data for radar visualization")
                
                # Export options
                with st.expander("📤 Export Sweep Results", expanded=False):
                    metadata = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa,
                        'region_type': region_key,
                        'stress_component': stress_component,
                        'stress_type': stress_type,
                        'spatial_sigma': spatial_sigma,
                        'attention_heads': attention_heads,
                        'angle_range': sweep['angle_range'],
                        'n_points': sweep['n_points']
                    }
                    
                    original_sweep = st.session_state.original_sweep if 'original_sweep' in st.session_state else None
                    export_data = st.session_state.results_manager.prepare_orientation_sweep_data(
                        sweep,
                        original_sweep,
                        metadata
                    )
                    
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        # JSON export
                        json_str = json.dumps(export_data, indent=2, default=str)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📥 Download JSON",
                            data=json_str,
                            file_name=f"orientation_sweep_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        # ZIP archive
                        zip_buffer = st.session_state.results_manager.create_orientation_sweep_archive(
                            sweep,
                            original_sweep,
                            metadata
                        )
                        
                        st.download_button(
                            "📦 Download Complete Archive",
                            data=zip_buffer.getvalue(),
                            file_name=f"orientation_sweep_archive_{timestamp}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            
            elif analysis_mode == "Compare Original vs Interpolated" and 'comparison_data' in st.session_state:
                st.markdown('<h3 class="sub-header">📊 Comparison Visualization</h3>', unsafe_allow_html=True)
                
                comp_data = st.session_state.comparison_data
                interpolated = comp_data['interpolated']
                original = comp_data['original']
                
                # Create comparison plot
                if interpolated['stresses'] and interpolated['angles']:
                    fig_comp, ax_comp = plt.subplots(figsize=(14, 7))
                    
                    # Plot interpolated curve
                    ax_comp.plot(interpolated['angles'], interpolated['stresses'], 
                               'b-', linewidth=3, label='Interpolated', alpha=0.8)
                    
                    # Plot original points
                    if isinstance(original['stresses'], list):
                        orig_stresses = np.array(original['stresses'])
                        orig_angles = np.array(original['angles'])
                    else:
                        orig_stresses = original['stresses']
                        orig_angles = original['angles']
                    
                    valid_mask = ~np.isnan(orig_stresses)
                    if np.any(valid_mask):
                        ax_comp.scatter(orig_angles[valid_mask], orig_stresses[valid_mask],
                                       color='red', s=80, label='Original Solutions', 
                                       edgecolors='black', linewidth=1.5, zorder=5)
                    
                    ax_comp.axvline(54.7, color='green', linestyle='--', linewidth=3,
                                   label='Habit Plane (54.7°)', alpha=0.6)
                    
                    ax_comp.set_xlabel('Orientation (°)', fontsize=14, fontweight='bold')
                    ax_comp.set_ylabel(f'{stress_component.replace("_", " ").title()} Stress (GPa)', 
                                      fontsize=14, fontweight='bold')
                    ax_comp.set_title(f'Comparison: Original vs Interpolated - {region_type_display}', 
                                     fontsize=16, fontweight='bold', pad=20)
                    ax_comp.grid(True, alpha=0.3, linestyle='--')
                    ax_comp.legend(fontsize=12, loc='upper right')
                    ax_comp.set_xlim([interpolated['angles'][0], interpolated['angles'][-1]])
                    
                    st.pyplot(fig_comp)
                    plt.close(fig_comp)
                else:
                    st.warning("No comparison data available for plotting")
                
                # Create radar comparison
                if viz_type in ["Radar", "Both", "Comparison"]:
                    st.markdown("#### 📡 Radar Comparison")
                    
                    # Prepare data for radar
                    if isinstance(original['stresses'], list):
                        orig_stresses = np.array(original['stresses'])
                        orig_angles = np.array(original['angles'])
                    else:
                        orig_stresses = original['stresses']
                        orig_angles = original['angles']
                    
                    if interpolated['stresses'] and interpolated['angles']:
                        interp_stresses = np.array(interpolated['stresses'])
                        interp_angles = np.array(interpolated['angles'])
                        
                        # Interpolate original to same grid for radar
                        valid_mask = ~np.isnan(orig_stresses)
                        if np.any(valid_mask) and len(interp_stresses) > 0 and len(orig_stresses) > 0:
                            orig_on_grid = np.interp(
                                interp_angles,
                                orig_angles[valid_mask],
                                orig_stresses[valid_mask]
                            )
                            
                            fig_radar_comp = st.session_state.visualizer.create_comparison_radar(
                                orig_on_grid, interp_stresses,
                                interp_angles,
                                title=f"Radar Comparison: {region_type_display} - {stress_component}",
                                angle_range=chart_angle_range
                            )
                            if fig_radar_comp:
                                st.plotly_chart(fig_radar_comp, use_container_width=True)
                            else:
                                st.warning("Could not create radar comparison")
                        else:
                            st.warning("Insufficient data for radar comparison")
                    else:
                        st.warning("No interpolated data available for radar comparison")
            
            elif analysis_mode == "Heatmap Analysis" and 'top_solution' in st.session_state:
                st.markdown('<h3 class="sub-header">🔥 Stress Distribution Heatmap</h3>', unsafe_allow_html=True)
                
                solution = st.session_state.top_solution
                
                # Extract eta and stress fields
                history = solution.get('history', [])
                if history:
                    last_frame = history[-1]
                    
                    if isinstance(last_frame, tuple) and len(last_frame) >= 2:
                        eta, stress_fields = last_frame[0], last_frame[1]
                    elif isinstance(last_frame, dict):
                        eta = last_frame.get('eta', np.zeros((128, 128)))
                        stress_fields = last_frame.get('stresses', {})
                    else:
                        st.error("Unable to extract stress fields from solution")
                        return
                    
                    # Create heatmap
                    st.info(f"Generating heatmap for solution: {solution.get('filename', 'Unknown')}")
                    
                    # Get heatmap settings
                    heatmap_settings = st.session_state.heatmap_settings if 'heatmap_settings' in st.session_state else {
                        'cmap': 'viridis',
                        'maintain_aspect': True
                    }
                    
                    # Static heatmap with perfect aspect ratio
                    st.markdown("#### 📊 Static Heatmap (Matplotlib)")
                    fig_heatmap = st.session_state.heatmap_visualizer.create_stress_distribution_heatmap(
                        eta, stress_fields, stress_component,
                        title=f"Stress Distribution - {solution.get('filename', 'Unknown')}",
                        cmap=heatmap_settings['cmap'],
                        maintain_aspect=heatmap_settings['maintain_aspect']
                    )
                    
                    if fig_heatmap is not None:
                        st.pyplot(fig_heatmap)
                        plt.close(fig_heatmap)
                    else:
                        st.warning(f"No stress data available for component '{stress_component}'. Check if it's present in the solution.")
                    
                    # Interactive heatmap
                    st.markdown("#### 🔬 Interactive Heatmap (Plotly)")
                    fig_interactive = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                        eta, stress_fields, stress_component,
                        title=f"Interactive Stress Distribution - {stress_component}",
                        maintain_aspect=heatmap_settings['maintain_aspect']
                    )
                    
                    if fig_interactive is not None:
                        st.plotly_chart(fig_interactive, use_container_width=True)
                    else:
                        st.warning(f"No stress data available for component '{stress_component}'. Check if it's present in the solution.")
                    
                    # Solution details
                    with st.expander("📋 Solution Details", expanded=False):
                        params = solution.get('params', {})
                        theta = params.get('theta', 0.0)
                        theta_deg = np.rad2deg(theta) if theta is not None else 0.0
                        
                        st.write(f"**Filename:** {solution.get('filename', 'Unknown')}")
                        st.write(f"**Defect Type:** {params.get('defect_type', 'Unknown')}")
                        st.write(f"**Shape:** {params.get('shape', 'Unknown')}")
                        st.write(f"**Orientation:** {theta_deg:.2f}°")
                        st.write(f"**ε*:** {params.get('eps0', 'Unknown')}")
                        st.write(f"**κ:** {params.get('kappa', 'Unknown')}")
                        st.write(f"**Number of Frames:** {len(history)}")
                        
                        # Stress statistics
                        if stress_component in stress_fields:
                            stress_data = stress_fields[stress_component]
                            valid_stress = stress_data[~np.isnan(stress_data)]
                            if len(valid_stress) > 0:
                                st.write(f"**{stress_component} Statistics:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Max", f"{np.max(valid_stress):.3f} GPa")
                                with col2:
                                    st.metric("Min", f"{np.min(valid_stress):.3f} GPa")
                                with col3:
                                    st.metric("Mean", f"{np.mean(valid_stress):.3f} GPa")
                                with col4:
                                    st.metric("Std", f"{np.std(valid_stress):.3f} GPa")
                        
                        # Download heatmap
                        st.markdown("#### 📥 Download Heatmap")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            # Save static heatmap
                            if fig_heatmap:
                                buf = BytesIO()
                                fig_heatmap.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                st.download_button(
                                    label="📷 Download Static Heatmap",
                                    data=buf.getvalue(),
                                    file_name=f"heatmap_static_{solution.get('filename', 'unknown').replace('.', '_')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                        
                        with col_dl2:
                            # Save solution data
                            stress_stats = {}
                            if stress_component in stress_fields:
                                stress_data = stress_fields[stress_component]
                                valid_stress = stress_data[~np.isnan(stress_data)]
                                if len(valid_stress) > 0:
                                    stress_stats = {
                                        'max': float(np.max(valid_stress)),
                                        'min': float(np.min(valid_stress)),
                                        'mean': float(np.mean(valid_stress)),
                                        'std': float(np.std(valid_stress))
                                    }
                            
                            solution_data = {
                                'filename': solution.get('filename', 'Unknown'),
                                'params': params,
                                'stress_statistics': stress_stats
                            }
                            json_str = json.dumps(solution_data, indent=2, default=str)
                            st.download_button(
                                label="📊 Download Solution Data",
                                data=json_str,
                                file_name=f"solution_data_{solution.get('filename', 'unknown').replace('.', '_')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                else:
                    st.warning("No history data available for the selected solution")
            
            elif analysis_mode == "Sintering Temperature Analysis":
                # Display sintering temperature analysis results
                if sintering_type == "Single Solution Analysis" and 'single_sintering_result' in st.session_state:
                    st.markdown('<h3 class="sub-header">🔥 Single Solution Sintering Analysis</h3>', unsafe_allow_html=True)
                    
                    result = st.session_state.single_sintering_result
                    analysis = result['analysis']
                    stress = result['stress']
                    T_sinter = result['T_sinter_exp']
                    T_sinter_arr = result['T_sinter_arr']
                    system_info = result['system_info']
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Hydrostatic Stress", f"{stress:.3f} GPa")
                    with col2:
                        st.metric("Sintering Temp (Exp)", f"{T_sinter:.1f} K")
                    with col3:
                        st.metric("Sintering Temp (°C)", f"{T_sinter-273.15:.1f} °C")
                    with col4:
                        system_class = system_info[0].split('(')[0].strip()
                        st.metric("System Type", system_class)
                    
                    # Show Arrhenius model comparison
                    with st.expander("📊 Model Comparison", expanded=True):
                        col_mod1, col_mod2 = st.columns(2)
                        with col_mod1:
                            st.metric("Exponential Model", f"{T_sinter:.1f} K")
                        with col_mod2:
                            st.metric("Arrhenius Model", f"{T_sinter_arr:.1f} K")
                        
                        # Show which model is more appropriate
                        diff = abs(T_sinter - T_sinter_arr)
                        if diff < 10:
                            st.success("✅ Models show good agreement (<10 K difference)")
                        elif diff < 30:
                            st.info("⚡ Models show moderate agreement (<30 K difference)")
                        else:
                            st.warning("⚠️ Models show significant difference (>30 K)")
                    
                    # Show theoretical curve with point
                    theory_data = st.session_state.sintering_calculator.get_theoretical_curve()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(theory_data['stresses'], theory_data['T_exponential'], 
                           'b-', linewidth=2, label='Exponential Model')
                    ax.plot(theory_data['stresses'], theory_data['T_arrhenius'], 
                           'g--', linewidth=2, label='Arrhenius Model', alpha=0.7)
                    ax.plot(stress, T_sinter, 'ro', markersize=12, 
                           label=f'Solution: {stress:.2f} GPa, {T_sinter:.0f} K')
                    
                    # Add system boundaries
                    ax.axvspan(0, 5, alpha=0.1, color='green', label='System 1')
                    ax.axvspan(5, 20, alpha=0.1, color='orange', label='System 2')
                    ax.axvspan(20, 35, alpha=0.1, color='red', label='System 3')
                    
                    ax.set_xlabel('|σ_h| (GPa)', fontsize=11)
                    ax.set_ylabel('T_sinter (K)', fontsize=11)
                    ax.set_title(f'Solution on Theoretical Curve: {analysis["filename"]}', 
                               fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=9)
                    
                    # Add Celsius axis
                    ax2 = ax.twinx()
                    ax2.set_ylabel('Temperature (°C)', fontsize=11)
                    ax2.set_ylim([t-273.15 for t in ax.get_ylim()])
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Solution details
                    with st.expander("📋 Solution Details", expanded=False):
                        st.write(f"**Filename:** {analysis['filename']}")
                        st.write(f"**Orientation:** {analysis['theta_deg']:.1f}°")
                        st.write(f"**Defect Type:** {analysis['params'].get('defect_type', 'Unknown')}")
                        st.write(f"**Shape:** {analysis['params'].get('shape', 'Unknown')}")
                        st.write(f"**ε*:** {analysis['params'].get('eps0', 0):.2f}")
                        st.write(f"**κ:** {analysis['params'].get('kappa', 0):.2f}")
                        
                        # Show system information
                        st.write(f"**System Classification:** {system_info[0]}")
                        st.write(f"**Temperature Range:** {system_info[1][0]:.0f}-{system_info[1][1]:.0f} K")
                        st.write(f"**Predicted Sintering Temperature:** {T_sinter:.1f} K ({T_sinter-273.15:.1f}°C)")
                        
                        # Interpretation
                        st.write("**Interpretation:**")
                        if 'System 1' in system_info[0]:
                            st.success("Perfect crystal structure - high sintering temperature required")
                        elif 'System 2' in system_info[0]:
                            st.info("Stacking faults/twins present - moderate sintering temperature")
                        else:
                            st.warning("Plastic deformation - low sintering temperature achievable")
                
                elif sintering_type == "Bulk Analysis":
                    st.markdown('<h3 class="sub-header">📊 Comprehensive Sintering Dashboard</h3>', unsafe_allow_html=True)
                    
                    if 'sintering_bulk_fig' in st.session_state:
                        st.pyplot(st.session_state.sintering_bulk_fig)
                        plt.close(st.session_state.sintering_bulk_fig)
                    
                    if 'sintering_interactive_fig' in st.session_state:
                        st.markdown("#### 🔬 Interactive Sintering Analysis")
                        st.plotly_chart(st.session_state.sintering_interactive_fig, use_container_width=True)
                    
                    # Export options
                    with st.expander("📤 Export Sintering Analysis", expanded=False):
                        export_data, report = st.session_state.results_manager.create_sintering_analysis_report(
                            st.session_state.solutions, region_key, st.session_state.sintering_calculator
                        )
                        
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        with col_exp1:
                            # JSON export
                            json_str = json.dumps(export_data, indent=2, default=str)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            st.download_button(
                                "📥 Download JSON Data",
                                data=json_str,
                                file_name=f"sintering_analysis_{timestamp}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_exp2:
                            # Report export
                            st.download_button(
                                "📋 Download Analysis Report",
                                data=report,
                                file_name=f"sintering_report_{timestamp}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        
                        with col_exp3:
                            # CSV export
                            if 'system_mapping_data' in st.session_state:
                                csv = st.session_state.system_mapping_data.to_csv(index=False)
                                st.download_button(
                                    "📊 Download CSV Data",
                                    data=csv,
                                    file_name=f"sintering_data_{timestamp}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                
                elif sintering_type == "Orientation Sweep" and 'sintering_sweep_fig' in st.session_state:
                    st.markdown('<h3 class="sub-header">🌐 Sintering Temperature Sweep</h3>', unsafe_allow_html=True)
                    
                    st.pyplot(st.session_state.sintering_sweep_fig)
                    plt.close(st.session_state.sintering_sweep_fig)
                
                elif sintering_type == "Theoretical Curve" and 'sintering_theory_fig' in st.session_state:
                    st.markdown('<h3 class="sub-header">📈 Theoretical Sintering Temperature Models</h3>', unsafe_allow_html=True)
                    
                    st.pyplot(st.session_state.sintering_theory_fig)
                    plt.close(st.session_state.sintering_theory_fig)
                    
                    # Display parameters
                    with st.expander("⚙️ Model Parameters", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("T₀ (reference)", f"{T0} K")
                            st.metric("β (calibration)", f"{beta}")
                        with col2:
                            st.metric("G (shear modulus)", f"{G} GPa")
                            st.metric("T_min (minimum)", f"{T_min} K")
                        with col3:
                            sigma_peak = st.session_state.sintering_calculator.compute_peak_stress_from_temperature()
                            st.metric("σ_peak (theoretical)", f"{sigma_peak:.1f} GPa")
                            activation_energy = 1.1  # eV (typical for Ag)
                            st.metric("Q_a (activation)", f"{activation_energy} eV")
                    
                    # Display key equations
                    st.markdown("""
                    <div class="latex-formula">
                    <strong>Key Equations:</strong><br>
                    1. <strong>Exponential Model:</strong> T<sub>sinter</sub>(σ<sub>h</sub>) = T₀ × exp(-β × |σ<sub>h</sub>| / G)<br>
                    2. <strong>Arrhenius Model:</strong> D = D₀ × exp[-(Q<sub>a</sub> - Ωσ<sub>h</sub>) / (k<sub>B</sub>T)]<br>
                    3. <strong>Peak Stress:</strong> σ<sub>peak</sub> = (G/β) × ln(T₀/T<sub>min</sub>) ≈ 28.5 GPa<br>
                    4. <strong>System Boundaries:</strong><br>
                    &nbsp;&nbsp;• System 1 (Perfect): σ<sub>h</sub> < 5 GPa, T ≈ 600-630 K<br>
                    &nbsp;&nbsp;• System 2 (SF/Twin): 5 ≤ σ<sub>h</sub> < 20 GPa, T ≈ 450-550 K<br>
                    &nbsp;&nbsp;• System 3 (Plastic): σ<sub>h</sub> ≥ 20 GPa, T ≈ 350-400 K
                    </div>
                    """, unsafe_allow_html=True)
                
                elif sintering_type == "System Mapping" and 'system_mapping_data' in st.session_state:
                    st.markdown('<h3 class="sub-header">🗺️ AgNP System Classification Mapping</h3>', unsafe_allow_html=True)
                    
                    df = st.session_state.system_mapping_data
                    
                    # Display as dataframe with conditional formatting
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("#### 📊 System Distribution Summary")
                    system_counts = df['System'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        count_sys1 = system_counts.get('System 1 (Perfect Crystal)', 0)
                        st.metric("System 1 (Perfect)", count_sys1)
                        if count_sys1 > 0:
                            st.progress(count_sys1 / len(df), text=f"{count_sys1/len(df)*100:.1f}%")
                    
                    with col2:
                        count_sys2 = system_counts.get('System 2 (Stacking Faults/Twins)', 0)
                        st.metric("System 2 (SF/Twin)", count_sys2)
                        if count_sys2 > 0:
                            st.progress(count_sys2 / len(df), text=f"{count_sys2/len(df)*100:.1f}%")
                    
                    with col3:
                        count_sys3 = system_counts.get('System 3 (Plastic Deformation)', 0)
                        st.metric("System 3 (Plastic)", count_sys3)
                        if count_sys3 > 0:
                            st.progress(count_sys3 / len(df), text=f"{count_sys3/len(df)*100:.1f}%")
                    
                    # Temperature statistics
                    st.markdown("#### 🌡️ Temperature Statistics")
                    temps_k = df['T_sinter (K)'].str.replace(' K', '').astype(float)
                    temps_c = df['T_sinter (°C)'].str.replace(' °C', '').astype(float)
                    
                    col_temp1, col_temp2, col_temp3, col_temp4 = st.columns(4)
                    with col_temp1:
                        st.metric("Min Temp", f"{temps_k.min():.1f} K", f"{temps_c.min():.1f} °C")
                    with col_temp2:
                        st.metric("Max Temp", f"{temps_k.max():.1f} K", f"{temps_c.max():.1f} °C")
                    with col_temp3:
                        st.metric("Mean Temp", f"{temps_k.mean():.1f} K", f"{temps_c.mean():.1f} °C")
                    with col_temp4:
                        st.metric("Temp Range", f"{temps_k.max()-temps_k.min():.1f} K")
                    
                    # Export option
                    st.markdown("#### 📥 Export Data")
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📊 Download System Mapping CSV",
                        data=csv,
                        file_name="agnp_system_mapping.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                elif sintering_type == "Defect Comparison Radar" and 'sintering_radar_fig' in st.session_state:
                    st.markdown('<h3 class="sub-header">📡 Sintering Temperature Radar Comparison</h3>', unsafe_allow_html=True)
                    
                    st.plotly_chart(st.session_state.sintering_radar_fig, use_container_width=True)
                    
                    # Interpretation
                    with st.expander("🔍 Interpretation Guide", expanded=True):
                        st.write("""
                        **Radar Chart Interpretation:**
                        
                        1. **Outer Rings (Higher Temperatures):** Require more energy for sintering
                        2. **Inner Rings (Lower Temperatures):** Enable low-temperature bonding
                        3. **Defect Type Comparison:**
                           - **ISF (Intrinsic Stacking Fault):** Moderate stress fields
                           - **ESF (Extrinsic Stacking Fault):** Stronger stress concentrations
                           - **Twin:** Highest stress fields at habit plane (54.7°)
                           - **No Defect:** Minimal stress, high sintering temperatures
                        
                        4. **Orientation Dependence:**
                           - Stress varies with crystal orientation
                           - Habit plane (54.7°) shows maximum effect for twins
                           - Different defects have different optimal orientations
                        
                        **Practical Implications:**
                        - Defect engineering can reduce sintering temperatures by 200-300 K
                        - Twin boundaries are most effective for low-temperature bonding
                        - Orientation control is critical for maximizing stress effects
                        """)
                    
                    # Export option
                    with st.expander("📤 Export Radar Chart", expanded=False):
                        # Convert figure to HTML
                        html_str = st.session_state.sintering_radar_fig.to_html(full_html=False, include_plotlyjs='cdn')
                        st.download_button(
                            "🌐 Download Interactive HTML",
                            data=html_str,
                            file_name="sintering_radar_comparison.html",
                            mime="text/html",
                            use_container_width=True
                        )
    
    with col_main2:
        st.markdown('<h2 class="sub-header">📈 Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Region information cards
        st.markdown(f'<div class="region-card {region_key}-region">', unsafe_allow_html=True)
        st.markdown(f"### {region_type_display}")
        
        if region_key == 'defect':
            st.write("**η > 0.6** - High defect concentration")
            st.write("• Stress concentration in defect cores")
            st.write("• Critical for defect initiation")
            st.write("• High sensitivity to orientation")
        elif region_key == 'interface':
            st.write("**0.4 ≤ η ≤ 0.6** - Interface region")
            st.write("• Stress gradients at interfaces")
            st.write("• Defect propagation path")
            st.write("• Transition zone effects")
        else:  # bulk
            st.write("**η < 0.4** - Pure Ag bulk")
            st.write("• Stress propagation in matrix")
            st.write("• Far-field stress effects")
            st.write("• Material response benchmark")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick metrics
        if 'solutions' in st.session_state and st.session_state.solutions:
            st.markdown("#### 📊 Loaded Solutions")
            st.metric("Total Solutions", len(st.session_state.solutions))
            
            # Count by defect type
            defect_counts = {}
            for sol in st.session_state.solutions:
                d_type = sol.get('params', {}).get('defect_type', 'Unknown')
                defect_counts[d_type] = defect_counts.get(d_type, 0) + 1
            
            for d_type, count in defect_counts.items():
                st.write(f"- {d_type}: {count}")
            
            # Orientation statistics
            orientations = st.session_state.original_analyzer.create_orientation_distribution(
                st.session_state.solutions
            )
            if len(orientations) > 0:
                st.metric("Avg Orientation", f"{np.mean(orientations):.1f}°")
                st.metric("Orientation Range", f"{np.min(orientations):.1f}° - {np.max(orientations):.1f}°")
        
        # Analysis mode specific dashboard
        if 'single_result' in st.session_state and analysis_mode == "Precise Single Orientation":
            result = st.session_state.single_result
            st.markdown("#### 🎯 Single Point Results")
            st.metric("Target Angle", f"{result['target_angle_deg']:.2f}°")
            st.metric("Region Stress", f"{result['region_stress']:.4f} GPa")
            st.metric("Sources Used", result['num_sources'])
            
            # Attention weights summary
            weights = result['attention_weights']
            if weights:
                st.write("**Attention Summary:**")
                weights_array = np.array(weights)
                if len(weights_array) > 0:
                    st.write(f"- Max weight: {np.max(weights_array):.3f}")
                    st.write(f"- Min weight: {np.min(weights_array):.3f}")
                    st.write(f"- Entropy: {-np.sum(weights_array * np.log(weights_array + 1e-10)):.3f}")
        
        elif 'sweep_result' in st.session_state and analysis_mode == "Orientation Sweep":
            sweep = st.session_state.sweep_result
            st.markdown("#### 🌐 Sweep Results")
            st.metric("Angle Range", f"{sweep['angle_range'][0]:.1f}° - {sweep['angle_range'][1]:.1f}°")
            st.metric("Points", sweep['n_points'])
            
            # Habit plane stress
            habit_angle = 54.7
            if sweep['stresses'] and sweep['angles']:
                angles = np.array(sweep['angles'])
                stresses = np.array(sweep['stresses'])
                if len(angles) > 0 and len(stresses) > 0:
                    idx = np.argmin(np.abs(angles - habit_angle))
                    if idx < len(stresses):
                        habit_stress = stresses[idx]
                        st.metric("Habit Plane Stress", f"{habit_stress:.4f} GPa")
            
            # Find min/max
            if sweep['stresses']:
                stresses = np.array(sweep['stresses'])
                angles = np.array(sweep['angles'])
                if len(stresses) > 0:
                    min_idx = np.nanargmin(stresses)
                    max_idx = np.nanargmax(stresses)
                    st.write(f"**Min:** {angles[min_idx]:.1f}° ({stresses[min_idx]:.4f} GPa)")
                    st.write(f"**Max:** {angles[max_idx]:.1f}° ({stresses[max_idx]:.4f} GPa)")
        
        elif 'comparison_data' in st.session_state and analysis_mode == "Compare Original vs Interpolated":
            comp_data = st.session_state.comparison_data
            st.markdown("#### 🔄 Comparison Metrics")
            
            interpolated = np.array(comp_data['interpolated']['stresses'])
            original = np.array(comp_data['original']['stresses'])
            valid_mask = ~np.isnan(original)
            
            if np.any(valid_mask) and len(interpolated) > 0:
                valid_orig = original[valid_mask]
                valid_angles = np.array(comp_data['original']['angles'])[valid_mask]
                
                # Interpolate to common grid
                interp_angles = np.array(comp_data['interpolated']['angles'])
                if len(valid_orig) > 0 and len(interp_angles) > 0:
                    orig_on_grid = np.interp(
                        interp_angles,
                        valid_angles,
                        valid_orig
                    )
                    
                    mae = np.mean(np.abs(orig_on_grid - interpolated))
                    rmse = np.sqrt(np.mean((orig_on_grid - interpolated) ** 2))
                    
                    st.metric("MAE", f"{mae:.4f} GPa")
                    st.metric("RMSE", f"{rmse:.4f} GPa")
                    
                    # Quality assessment
                    if rmse < 0.1:
                        st.success("✅ Excellent agreement")
                    elif rmse < 0.3:
                        st.info("⚡ Good agreement")
                    else:
                        st.warning("⚠️ Consider adjusting parameters")
                else:
                    st.warning("Insufficient data for comparison metrics")
        
        elif 'top_solution' in st.session_state and analysis_mode == "Heatmap Analysis":
            st.markdown("#### 🔥 Heatmap Solution")
            solution = st.session_state.top_solution
            params = solution.get('params', {})
            theta = params.get('theta', 0.0)
            theta_deg = np.rad2deg(theta) if theta is not None else 0.0
            
            st.metric("Orientation", f"{theta_deg:.1f}°")
            st.metric("Defect Type", params.get('defect_type', 'Unknown'))
            st.metric("ε*", f"{params.get('eps0', 0):.2f}")
            st.metric("κ", f"{params.get('kappa', 0):.2f}")
            
            # Heatmap settings
            if 'heatmap_settings' in st.session_state:
                heatmap_settings = st.session_state.heatmap_settings
                st.markdown("#### ⚙️ Heatmap Settings")
                st.write(f"**Color Map:** {heatmap_settings['cmap']}")
                st.write(f"**Aspect Ratio:** {'Perfect (1:1)' if heatmap_settings['maintain_aspect'] else 'Auto'}")
        
        elif analysis_mode == "Sintering Temperature Analysis":
            st.markdown("#### 🔥 Sintering Temperature Model")
            
            # Display current model parameters
            st.metric("T₀ (σ=0)", f"{st.session_state.sintering_calculator.T0} K", 
                     f"{st.session_state.sintering_calculator.T0-273.15:.0f}°C")
            st.metric("T_min (peak)", f"{st.session_state.sintering_calculator.T_min} K",
                     f"{st.session_state.sintering_calculator.T_min-273.15:.0f}°C")
            st.metric("β (calibration)", f"{st.session_state.sintering_calculator.beta}")
            st.metric("G (shear modulus)", f"{st.session_state.sintering_calculator.G} GPa")
            
            # Calculate and display peak stress
            sigma_peak = st.session_state.sintering_calculator.compute_peak_stress_from_temperature()
            st.metric("σ_peak", f"{sigma_peak:.1f} GPa")
            
            # Show temperature reduction
            temp_reduction = ((st.session_state.sintering_calculator.T0 - 
                              st.session_state.sintering_calculator.T_min) / 
                             st.session_state.sintering_calculator.T0 * 100)
            st.metric("Max Reduction", f"{temp_reduction:.1f}%")
            
            # System classification guide
            st.markdown("#### 🏷️ System Classification")
            col_sys1, col_sys2, col_sys3 = st.columns(3)
            with col_sys1:
                st.markdown('<div class="system-1" style="padding: 5px; border-radius: 5px; text-align: center;">System 1<br>Perfect</div>', 
                          unsafe_allow_html=True)
                st.caption("σ < 5 GPa\nT ≈ 600-630 K")
            with col_sys2:
                st.markdown('<div class="system-2" style="padding: 5px; border-radius: 5px; text-align: center;">System 2<br>SF/Twin</div>', 
                          unsafe_allow_html=True)
                st.caption("5 ≤ σ < 20 GPa\nT ≈ 450-550 K")
            with col_sys3:
                st.markdown('<div class="system-3" style="padding: 5px; border-radius: 5px; text-align: center;">System 3<br>Plastic</div>', 
                          unsafe_allow_html=True)
                st.caption("σ ≥ 20 GPa\nT ≈ 350-400 K")
        
        # Method explanation
        with st.expander("🧠 Method Details", expanded=False):
            if analysis_mode == "Sintering Temperature Analysis":
                st.write("""
                **Sintering Temperature Physics:**
                
                1. **Stress-Modified Diffusion:**
                   - Atomic diffusion enhanced by hydrostatic stress
                   - Activation energy reduced by stress work: Q_eff = Q_a - Ωσ_h
                   - Stress-modified Arrhenius equation
                
                2. **Empirical Correlation:**
                   - Exponential model calibrated to AgNP systems
                   - T_sinter(σ_h) = T₀ × exp(-β × |σ_h| / G)
                   - Validated against experimental DSC data
                
                3. **System Classification:**
                   - **System 1:** Perfect Ag crystals (σ < 5 GPa)
                   - **System 2:** Stacking faults/twins (5 ≤ σ < 20 GPa)
                   - **System 3:** Plastic deformation (σ ≥ 20 GPa)
                
                4. **Habit Plane Significance:**
                   - 54.7° orientation for Ag FCC twin boundaries
                   - Maximum stress concentration at habit plane
                   - Optimal for defect engineering
                
                **Physical Interpretation:**
                Defect-induced stress fields reduce the energy barrier for atomic diffusion, 
                enabling sintering at significantly lower temperatures than conventional 
                thermal processing.
                """)
            else:
                st.write("""
                **Attention-Based Spatial Interpolation:**
                
                1. **Parameter Encoding:**
                   - 12-dimensional parameter vectors
                   - One-hot encoding for categorical parameters
                   - Normalized continuous parameters
                
                2. **Attention Mechanism:**
                   - Transformer-inspired multi-head attention
                   - Learns relationships between parameters
                   - Dynamic weight assignment
                
                3. **Spatial Regularization:**
                   - Euclidean distance in parameter space
                   - Gaussian kernel: exp(-0.5 * (distance/σ)²)
                   - Prevents over-reliance on distant sources
                
                4. **Weight Combination:**
                   - 70% attention weights + 30% spatial weights
                   - Normalized to sum to 1
                   - Ensures physically meaningful interpolation
                
                **Ag FCC Twin Specific:**
                - Habit plane at 54.7°
                - {111} crystal planes
                - Coherent twin boundaries
                - Orientation-dependent stress fields
                
                **Heatmap Enhancement:**
                - Perfect aspect ratio maintained
                - Square pixels for accurate representation
                - Robust error handling for missing data
                - Region-based analysis overlays
                """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
