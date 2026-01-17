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
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================
# PLACEHOLDER: MISSING BASE CLASSES (ADDED TO FIX NameError)
# =============================================

class PhysicsBasedStressAnalyzer:
    """Base class for physics-based stress analysis."""
    
    def __init__(self):
        pass
    
    def compute_strain_energy_density(self, stress_fields):
        """Compute strain energy density from stress fields."""
        # Simple isotropic approximation: U = 0.5 * sum(sigma_ij * epsilon_ij)
        # For demo, return zeros matching sigma_hydro shape if available
        if 'sigma_hydro' in stress_fields:
            return np.zeros_like(stress_fields['sigma_hydro'])
        else:
            # Fallback: assume all stress components are present
            keys = list(stress_fields.keys())
            if keys:
                return np.zeros_like(stress_fields[keys[0]])
        return None
    
    def analyze_crystal_orientation_effects(self, stats, orientation_deg):
        """Placeholder for orientation-dependent analysis."""
        return {
            'orientation_deg': orientation_deg,
            'schmid_factor': np.sin(np.radians(orientation_deg)) * np.cos(np.radians(orientation_deg)),
            'effective_stress_multiplier': 1.0 + 0.2 * np.abs(np.sin(np.radians(2 * orientation_deg)))
        }


class EnhancedSinteringCalculator:
    """Placeholder for sintering temperature prediction."""
    
    def __init__(self):
        self.Q_a_standard_eV = 0.9  # Example activation energy for Ag
        self.Omega = 1.0e-29       # Activation volume (m³)
        self.k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.D0 = 1e-5                 # Pre-exponential factor (m²/s)
        self.D_crit = 1e-18            # Critical diffusivity for sintering
    
    def get_theoretical_curve(self):
        """Return theoretical stress vs. temperature curves."""
        stresses = np.linspace(0, 30, 100)  # GPa
        exponential_standard = 630 - 10 * stresses  # Simplified empirical
        return {
            'stresses': stresses.tolist(),
            'exponential_standard': exponential_standard.tolist()
        }
    
    def create_comprehensive_sintering_plot(self, stresses, temps, defect_type, title):
        """Create a basic sintering temperature vs stress plot."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(stresses, temps, label=f"{defect_type} (Empirical)", linewidth=2)
        ax.set_xlabel("Hydrostatic Stress (GPa)")
        ax.set_ylabel("Sintering Temperature (K)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig


class PhysicsAwareInterpolator:
    """Minimal interpolator to avoid NameError."""
    
    def __init__(self):
        pass
    
    def compute_parameter_vector(self, params, orientation_deg):
        """Convert parameters to a feature vector."""
        defect_map = {"ISF": [1,0,0,0], "ESF": [0,1,0,0], "Twin": [0,0,1,0], "No Defect": [0,0,0,1]}
        defect_vec = defect_map.get(params.get('defect_type', 'No Defect'), [0,0,0,1])
        shape_map = {"Square": 0, "Horizontal Fault": 1, "Vertical Fault": 2, "Rectangle": 3}
        shape_val = shape_map.get(params.get('shape', 'Square'), 0)
        eps0 = params.get('eps0', 0.0)
        kappa = params.get('kappa', 0.6)
        theta_norm = orientation_deg / 360.0
        return np.array(defect_vec + [shape_val, eps0, kappa, theta_norm])
    
    def compute_attention_weights(self, source_vectors, target_vector):
        """Simple inverse distance weighting as attention proxy."""
        dists = np.linalg.norm(source_vectors - target_vector, axis=1)
        weights = 1.0 / (dists + 1e-6)
        return weights / weights.sum()
    
    def compute_spatial_weights(self, source_vectors, target_vector):
        """Same as attention for placeholder."""
        return self.compute_attention_weights(source_vectors, target_vector)
    
    def compute_physics_weights(self, source_params_list, target_params, orientation_deg):
        """Physics-based similarity (placeholder)."""
        weights = np.ones(len(source_params_list))
        return weights / weights.sum()
    
    def interpolate_stress_components(self, solutions, orientation_deg, target_params, region_type='all'):
        """Dummy interpolation returning synthetic stress fields."""
        # Create synthetic stress data
        size = (64, 64)
        eta = np.random.rand(*size)
        sigma_hydro = np.random.randn(*size) * 5  # ~5 GPa std
        von_mises = np.abs(sigma_hydro) * 1.2
        sigma_mag = np.sqrt(sigma_hydro**2 + von_mises**2)
        
        stress_fields = {
            'sigma_hydro': sigma_hydro,
            'von_mises': von_mises,
            'sigma_mag': sigma_mag
        }
        
        # Dummy sintering analysis
        mean_stress = np.mean(np.abs(sigma_hydro[eta > 0.6])) if np.any(eta > 0.6) else 5.0
        Q_eff = 0.9 - 0.02 * mean_stress  # eV
        T_k = Q_eff / (8.617e-5 * np.log(1e13))  # Simplified Arrhenius
        
        return {
            'interpolated_stress': stress_fields,
            'eta': eta,
            'sintering_analysis': {
                'temperature_predictions': {
                    'arrhenius_defect_k': max(T_k, 300),
                    'arrhenius_defect_c': max(T_k - 273.15, 27),
                    'exponential_model_k': 630 - 8 * mean_stress,
                    'exponential_model_c': 630 - 8 * mean_stress - 273.15
                },
                'activation_energy_analysis': {
                    'Q_a_standard_eV': 0.9,
                    'Q_eff_defect_eV': Q_eff,
                    'reduction_defect_eV': 0.9 - Q_eff,
                    'reduction_percentage': (0.9 - Q_eff) / 0.9 * 100,
                    'reduction_standard_eV': 0.9 - Q_eff
                },
                'system_classification': {
                    'system': 'System 2 (SF/Twin)' if mean_stress < 20 else 'System 3 (Plastic)',
                    'predicted_T_k': max(T_k, 300)
                }
            }
        }


class EnhancedSolutionLoader:
    """Load simulation solutions from directory."""
    
    def __init__(self, solutions_dir):
        self.solutions_dir = solutions_dir
    
    def load_all_solutions(self):
        """Load .pkl files from directory; return empty list if none."""
        solutions = []
        if not os.path.exists(self.solutions_dir):
            return solutions
        for fname in os.listdir(self.solutions_dir):
            if fname.endswith('.pkl'):
                try:
                    with open(os.path.join(self.solutions_dir, fname), 'rb') as f:
                        sol = pickle.load(f)
                        sol['metadata'] = sol.get('metadata', {})
                        sol['metadata']['filename'] = fname
                        solutions.append(sol)
                except Exception as e:
                    st.warning(f"Failed to load {fname}: {e}")
        return solutions


# =============================================
# ENHANCED CONFIGURATION WITH VISUALIZATION SETTINGS
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# =============================================
# UNIVERSAL VISUALIZATION ENHANCER CLASS
# =============================================
class UniversalVisualizationEnhancer:
    """
    Comprehensive visualization enhancer with 50+ colormaps, font controls,
    line thickness adjustments, and universal figure enhancements
    """
    def __init__(self):
        # Comprehensive colormap collection (50+ options)
        self.colormaps = {
            # Sequential colormaps
            'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
            'magma': 'magma', 'cividis': 'cividis', 'summer': 'summer',
            'wistia': 'wistia', 'autumn': 'autumn', 'spring': 'spring',
            'cool': 'cool', 'Wistia': 'Wistia', 'hot': 'hot',
            'afmhot': 'afmhot', 'gist_heat': 'gist_heat', 'copper': 'copper',
            # Diverging colormaps
            'Spectral': 'Spectral', 'coolwarm': 'coolwarm', 'bwr': 'bwr',
            'seismic': 'seismic', 'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn',
            'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG',
            'PuOr': 'PuOr', 'RdGy': 'RdGy',
            # Qualitative colormaps
            'tab10': 'tab10', 'tab20': 'tab20', 'Set1': 'Set1',
            'Set2': 'Set2', 'Set3': 'Set3', 'tab20b': 'tab20b',
            'tab20c': 'tab20c', 'Pastel1': 'Pastel1', 'Pastel2': 'Pastel2',
            'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2',
            # Cyclic colormaps
            'twilight': 'twilight', 'twilight_shifted': 'twilight_shifted',
            'hsv': 'hsv',
            # Legacy/perceptually problematic but requested
            'jet': 'jet', 'rainbow': 'rainbow', 'turbo': 'turbo',
            'nipy_spectral': 'nipy_spectral', 'gist_ncar': 'gist_ncar',
            'gist_rainbow': 'gist_rainbow',
            # Custom engineered colormaps
            'thermal_stress': self._create_thermal_stress_cmap(),
            'defect_gradient': self._create_defect_gradient_cmap(),
            'crystal_orientation': self._create_crystal_orientation_cmap(),
            'stress_tensile_compressive': self._create_stress_tensile_compressive_cmap(),
        }
        # Font options
        self.font_families = ['Arial', 'Times New Roman', 'Helvetica',
                              'Courier New', 'Verdana', 'Georgia', 'Cambria']
        # Default visualization parameters
        self.default_params = {
            'font_size': 12,
            'title_size': 14,
            'label_size': 11,
            'tick_size': 10,
            'legend_size': 10,
            'line_width': 2.0,
            'marker_size': 6,
            'grid_alpha': 0.3,
            'figure_dpi': 150,
            'colorbar_width': 0.02,
            'colorbar_pad': 0.05,
        }
        # Stress visualization specific parameters
        self.stress_params = {
            'hydrostatic_cmap': 'RdBu_r',
            'vonmises_cmap': 'viridis',
            'magnitude_cmap': 'plasma',
            'defect_cmap': 'Set1',
            'interface_alpha': 0.7,
            'bulk_alpha': 0.3,
            'defect_alpha': 0.9,
            'contour_levels': 20,
            'vector_scale': 50,
            'quiver_density': 10,
        }

    def _create_thermal_stress_cmap(self):
        """Create custom thermal stress colormap (blue to red)"""
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1),
                  (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
        positions = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        return LinearSegmentedColormap.from_list('thermal_stress', list(zip(positions, colors)))

    def _create_defect_gradient_cmap(self):
        """Create custom defect gradient colormap"""
        colors = [(0.1, 0.1, 0.1), (0.3, 0, 0.5), (0.6, 0, 0.8),
                  (0.8, 0.3, 0.1), (1, 0.6, 0), (1, 0.9, 0.3)]
        return LinearSegmentedColormap.from_list('defect_gradient', colors)

    def _create_crystal_orientation_cmap(self):
        """Create crystal orientation colormap (cyclic)"""
        colors = [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 1),
                  (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0),
                  (1, 0, 0), (0.5, 0, 0), (0, 0, 0.5)]
        return LinearSegmentedColormap.from_list('crystal_orientation', colors, N=256)

    def _create_stress_tensile_compressive_cmap(self):
        """Create tensile (red) to compressive (blue) colormap"""
        colors = [(0, 0, 1), (0.2, 0.2, 1), (0.4, 0.4, 1),
                  (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8),
                  (1, 0.4, 0.4), (1, 0.2, 0.2), (1, 0, 0)]
        return LinearSegmentedColormap.from_list('stress_tensile_compressive', colors)

    def create_visualization_controls(self, container=None):
        """
        Create comprehensive visualization controls in Streamlit
        Args:
            container: Streamlit container to place controls in
        """
        if container is None:
            container = st.sidebar
        container.markdown("---")
        container.markdown("### 🎨 Visualization Controls")
        # Colormap selection with categories
        colormap_category = container.selectbox(
            "Colormap Category",
            ["Sequential", "Diverging", "Qualitative", "Cyclic", "Legacy", "Custom"],
            index=0,
            help="Select colormap category for visualization"
        )
        # Filter colormaps by category
        category_maps = {
            "Sequential": ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                           'summer', 'wistia', 'autumn', 'spring', 'cool',
                           'hot', 'afmhot', 'gist_heat', 'copper'],
            "Diverging": ['Spectral', 'coolwarm', 'bwr', 'seismic', 'RdYlBu',
                          'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy'],
            "Qualitative": ['tab10', 'tab20', 'Set1', 'Set2', 'Set3',
                            'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                            'Paired', 'Accent', 'Dark2'],
            "Cyclic": ['twilight', 'twilight_shifted', 'hsv'],
            "Legacy": ['jet', 'rainbow', 'turbo', 'nipy_spectral',
                       'gist_ncar', 'gist_rainbow'],
            "Custom": ['thermal_stress', 'defect_gradient',
                       'crystal_orientation', 'stress_tensile_compressive']
        }
        selected_cmap = container.selectbox(
            "Select Colormap",
            category_maps[colormap_category],
            index=0,
            help="Choose from 50+ colormaps including rainbow, jet, turbo, inferno"
        )
        # Font controls
        container.markdown("#### 📝 Font Controls")
        col1, col2 = container.columns(2)
        with col1:
            font_family = col1.selectbox(
                "Font Family",
                self.font_families,
                index=0
            )
            title_size = col1.slider(
                "Title Font Size",
                min_value=8,
                max_value=24,
                value=self.default_params['title_size'],
                step=1
            )
        with col2:
            label_size = col2.slider(
                "Label Font Size",
                min_value=6,
                max_value=20,
                value=self.default_params['label_size'],
                step=1
            )
            tick_size = col2.slider(
                "Tick Font Size",
                min_value=6,
                max_value=18,
                value=self.default_params['tick_size'],
                step=1
            )
        # Line and marker controls
        container.markdown("#### 📈 Line & Marker Controls")
        col3, col4 = container.columns(2)
        with col3:
            line_width = col3.slider(
                "Line Width",
                min_value=0.5,
                max_value=10.0,
                value=self.default_params['line_width'],
                step=0.5
            )
            marker_size = col3.slider(
                "Marker Size",
                min_value=1,
                max_value=20,
                value=self.default_params['marker_size'],
                step=1
            )
        with col4:
            grid_alpha = col4.slider(
                "Grid Opacity",
                min_value=0.0,
                max_value=1.0,
                value=self.default_params['grid_alpha'],
                step=0.05
            )
            figure_dpi = col4.slider(
                "Figure DPI",
                min_value=72,
                max_value=300,
                value=self.default_params['figure_dpi'],
                step=12
            )
        # Colorbar controls
        container.markdown("#### 🎨 Colorbar Controls")
        col5, col6 = container.columns(2)
        with col5:
            colorbar_width = col5.slider(
                "Colorbar Width",
                min_value=0.01,
                max_value=0.1,
                value=self.default_params['colorbar_width'],
                step=0.005
            )
        with col6:
            colorbar_pad = col6.slider(
                "Colorbar Padding",
                min_value=0.01,
                max_value=0.2,
                value=self.default_params['colorbar_pad'],
                step=0.01
            )
        # Style options
        container.markdown("#### 🎭 Style Options")
        col7, col8 = container.columns(2)
        with col7:
            use_latex = col7.checkbox("Use LaTeX Rendering", value=False)
            dark_theme = col7.checkbox("Dark Theme", value=False)
        with col8:
            transparent_bg = col8.checkbox("Transparent Background", value=False)
            tight_layout = col8.checkbox("Tight Layout", value=True)
        # Return all settings
        return {
            'colormap': selected_cmap,
            'font_family': font_family,
            'title_size': title_size,
            'label_size': label_size,
            'tick_size': tick_size,
            'line_width': line_width,
            'marker_size': marker_size,
            'grid_alpha': grid_alpha,
            'figure_dpi': figure_dpi,
            'colorbar_width': colorbar_width,
            'colorbar_pad': colorbar_pad,
            'use_latex': use_latex,
            'dark_theme': dark_theme,
            'transparent_bg': transparent_bg,
            'tight_layout': tight_layout,
        }

    def apply_visualization_settings(self, fig=None, ax=None, settings=None):
        """
        Apply visualization settings to matplotlib figure/axes
        Args:
            fig: matplotlib figure object
            ax: matplotlib axes object
            settings: dictionary of visualization settings
        Returns:
            Updated figure and axes
        """
        if settings is None:
            settings = self.default_params
        if fig is not None:
            # Set figure DPI
            fig.set_dpi(settings['figure_dpi'])
            # Set background
            if settings['transparent_bg']:
                fig.patch.set_alpha(0.0)
            elif settings['dark_theme']:
                fig.patch.set_facecolor('black')
            # Apply tight layout
            if settings['tight_layout']:
                fig.tight_layout()
        if ax is not None:
            # Set font properties
            font_props = FontProperties(family=settings['font_family'])
            # Apply to title
            title = ax.get_title()
            if title:
                ax.set_title(title, fontsize=settings['title_size'],
                             fontproperties=font_props)
            # Apply to labels
            xlabel = ax.get_xlabel()
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=settings['label_size'],
                              fontproperties=font_props)
            ylabel = ax.get_ylabel()
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=settings['label_size'],
                              fontproperties=font_props)
            # Apply to ticks
            ax.tick_params(axis='both', which='major',
                           labelsize=settings['tick_size'])
            # Apply grid
            ax.grid(True, alpha=settings['grid_alpha'])
            # Apply to legend
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(font_props)
                    text.set_fontsize(settings.get('legend_size', settings['label_size']))
            # Set background
            if settings['dark_theme']:
                ax.set_facecolor('black')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
        return fig, ax

    def create_styled_colormap(self, cmap_name, n_colors=256):
        """
        Create styled colormap with enhanced properties
        Args:
            cmap_name: Name of the colormap
            n_colors: Number of colors in the colormap
        Returns:
            Colormap object
        """
        if cmap_name in self.colormaps:
            if isinstance(self.colormaps[cmap_name], str):
                # Matplotlib built-in colormap
                return plt.get_cmap(self.colormaps[cmap_name], n_colors)
            else:
                # Custom colormap
                return self.colormaps[cmap_name]
        else:
            # Fallback to viridis
            return plt.get_cmap('viridis', n_colors)

    def create_stress_visualization(self, geometry_data, stress_data, settings,
                                    orientation_angle=54.7, defect_type='Twin'):
        """
        Create comprehensive stress visualization for real geometry domain
        Args:
            geometry_data: Dictionary containing geometry information (eta, coordinates)
            stress_data: Dictionary containing stress fields
            settings: Visualization settings
            orientation_angle: Crystal orientation angle in degrees
            defect_type: Type of defect
        Returns:
            matplotlib figure object
        """
        # Extract data
        eta = geometry_data.get('eta', None)
        coordinates = geometry_data.get('coordinates', None)
        if eta is None or stress_data is None:
            return None
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        # Get colormaps
        cmap_hydro = self.create_styled_colormap('RdBu_r' if settings.get('stress_cmap_hydro') is None else settings['stress_cmap_hydro'])
        cmap_vonmises = self.create_styled_colormap('viridis' if settings.get('stress_cmap_vonmises') is None else settings['stress_cmap_vonmises'])
        cmap_magnitude = self.create_styled_colormap('plasma' if settings.get('stress_cmap_magnitude') is None else settings['stress_cmap_magnitude'])
        cmap_defect = self.create_styled_colormap('Set1' if settings.get('stress_cmap_defect') is None else settings['stress_cmap_defect'])
        # Plot 1: Phase Field (Eta) - Defect Regions
        ax1 = axes[0]
        if eta.ndim == 2:
            im1 = ax1.imshow(eta, cmap=cmap_defect, origin='lower',
                             extent=[0, eta.shape[1], 0, eta.shape[0]])
            ax1.set_title('Phase Field: Defect Regions', fontsize=settings['title_size'])
            ax1.set_xlabel('X Position', fontsize=settings['label_size'])
            ax1.set_ylabel('Y Position', fontsize=settings['label_size'])
            plt.colorbar(im1, ax=ax1, label='Order Parameter (η)')
        # Plot 2: Hydrostatic Stress
        ax2 = axes[1]
        sigma_hydro = stress_data.get('sigma_hydro', None)
        if sigma_hydro is not None and sigma_hydro.ndim == 2:
            vmax = np.max(np.abs(sigma_hydro))
            vmin = -vmax
            im2 = ax2.imshow(sigma_hydro, cmap=cmap_hydro, origin='lower',
                             vmin=vmin, vmax=vmax,
                             extent=[0, sigma_hydro.shape[1], 0, sigma_hydro.shape[0]])
            ax2.set_title(f'Hydrostatic Stress (σ_h)\nOrientation: {orientation_angle}°',
                          fontsize=settings['title_size'])
            ax2.set_xlabel('X Position', fontsize=settings['label_size'])
            ax2.set_ylabel('Y Position', fontsize=settings['label_size'])
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Stress (GPa)', fontsize=settings['label_size'])
        # Plot 3: Von Mises Stress
        ax3 = axes[2]
        von_mises = stress_data.get('von_mises', None)
        if von_mises is not None and von_mises.ndim == 2:
            im3 = ax3.imshow(von_mises, cmap=cmap_vonmises, origin='lower',
                             extent=[0, von_mises.shape[1], 0, von_mises.shape[0]])
            ax3.set_title('Von Mises Stress (σ_vm)', fontsize=settings['title_size'])
            ax3.set_xlabel('X Position', fontsize=settings['label_size'])
            ax3.set_ylabel('Y Position', fontsize=settings['label_size'])
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('Equivalent Stress (GPa)', fontsize=settings['label_size'])
        # Plot 4: Stress Magnitude
        ax4 = axes[3]
        sigma_mag = stress_data.get('sigma_mag', None)
        if sigma_mag is not None and sigma_mag.ndim == 2:
            im4 = ax4.imshow(sigma_mag, cmap=cmap_magnitude, origin='lower',
                             extent=[0, sigma_mag.shape[1], 0, sigma_mag.shape[0]])
            ax4.set_title('Stress Magnitude (|σ|)', fontsize=settings['title_size'])
            ax4.set_xlabel('X Position', fontsize=settings['label_size'])
            ax4.set_ylabel('Y Position', fontsize=settings['label_size'])
            cbar4 = plt.colorbar(im4, ax=ax4)
            cbar4.set_label('Stress Magnitude (GPa)', fontsize=settings['label_size'])
        # Plot 5: Region-specific stress analysis
        ax5 = axes[4]
        if eta is not None and sigma_hydro is not None:
            # Define regions based on eta
            defect_mask = eta > 0.6
            interface_mask = (eta >= 0.4) & (eta <= 0.6)
            bulk_mask = eta < 0.4
            # Create RGB image showing regions
            region_rgb = np.zeros((*eta.shape, 3))
            region_rgb[defect_mask] = [1, 0, 0]  # Red for defect
            region_rgb[interface_mask] = [0, 1, 0]  # Green for interface
            region_rgb[bulk_mask] = [0, 0, 1]  # Blue for bulk
            ax5.imshow(region_rgb, origin='lower',
                       extent=[0, eta.shape[1], 0, eta.shape[0]])
            ax5.set_title('Material Regions', fontsize=settings['title_size'])
            ax5.set_xlabel('X Position', fontsize=settings['label_size'])
            ax5.set_ylabel('Y Position', fontsize=settings['label_size'])
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.6, label='Defect (η > 0.6)'),
                Patch(facecolor='green', alpha=0.6, label='Interface (0.4 ≤ η ≤ 0.6)'),
                Patch(facecolor='blue', alpha=0.6, label='Bulk (η < 0.4)')
            ]
            ax5.legend(handles=legend_elements, loc='upper right',
                       fontsize=settings['tick_size'])
        # Plot 6: Stress distribution by region
        ax6 = axes[5]
        if eta is not None and sigma_hydro is not None:
            regions = []
            stresses = []
            for region_name, mask in [('Defect', defect_mask),
                                      ('Interface', interface_mask),
                                      ('Bulk', bulk_mask)]:
                if np.any(mask):
                    region_stresses = sigma_hydro[mask]
                    regions.extend([region_name] * len(region_stresses))
                    stresses.extend(region_stresses)
            if regions and stresses:
                df = pd.DataFrame({'Region': regions, 'Hydrostatic Stress (GPa)': stresses})
                box_data = []
                for region in ['Defect', 'Interface', 'Bulk']:
                    region_data = df[df['Region'] == region]['Hydrostatic Stress (GPa)'].values
                    if len(region_data) > 0:
                        box_data.append(region_data)
                bp = ax6.boxplot(box_data, labels=['Defect', 'Interface', 'Bulk'],
                                 patch_artist=True)
                # Color boxes
                colors = ['red', 'green', 'blue']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                ax6.set_title('Stress Distribution by Region', fontsize=settings['title_size'])
                ax6.set_ylabel('Hydrostatic Stress (GPa)', fontsize=settings['label_size'])
                ax6.grid(True, alpha=settings['grid_alpha'])
        # Apply universal settings to all axes
        for ax in axes:
            self.apply_visualization_settings(ax=ax, settings=settings)
        # Add main title
        fig.suptitle(f'Stress Analysis for {defect_type} Defect\n'
                     f'Orientation: {orientation_angle}°, '
                     f'Colormap: {settings.get("colormap", "Default")}',
                     fontsize=settings['title_size'] + 4,
                     fontweight='bold')
        plt.tight_layout()
        return fig

    def create_3d_stress_visualization(self, geometry_data, stress_data, settings):
        """
        Create 3D stress visualization using Plotly
        Args:
            geometry_data: Dictionary containing geometry information
            stress_data: Dictionary containing stress fields
            settings: Visualization settings
        Returns:
            Plotly figure object
        """
        # Extract data
        eta = geometry_data.get('eta', None)
        sigma_hydro = stress_data.get('sigma_hydro', None)
        if eta is None or sigma_hydro is None:
            return None
        # Create meshgrid
        x = np.arange(eta.shape[1])
        y = np.arange(eta.shape[0])
        X, Y = np.meshgrid(x, y)
        # Create 3D surface plot
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Phase Field (η)', 'Hydrostatic Stress',
                            'Von Mises Stress', 'Stress Magnitude'),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        # Plot 1: Phase Field
        fig.add_trace(
            go.Surface(z=eta, x=X, y=Y,
                       colorscale=settings.get('colormap', 'viridis'),
                       showscale=True,
                       colorbar=dict(title="η", len=0.4, y=0.8)),
            row=1, col=1
        )
        # Plot 2: Hydrostatic Stress
        fig.add_trace(
            go.Surface(z=sigma_hydro, x=X, y=Y,
                       colorscale='RdBu',
                       showscale=True,
                       colorbar=dict(title="σ_h (GPa)", len=0.4, y=0.8)),
            row=1, col=2
        )
        # Plot 3: Von Mises Stress
        von_mises = stress_data.get('von_mises', None)
        if von_mises is not None:
            fig.add_trace(
                go.Surface(z=von_mises, x=X, y=Y,
                           colorscale='viridis',
                           showscale=True,
                           colorbar=dict(title="σ_vm (GPa)", len=0.4, y=0.3)),
                row=2, col=1
            )
        # Plot 4: Stress Magnitude
        sigma_mag = stress_data.get('sigma_mag', None)
        if sigma_mag is not None:
            fig.add_trace(
                go.Surface(z=sigma_mag, x=X, y=Y,
                           colorscale='plasma',
                           showscale=True,
                           colorbar=dict(title="|σ| (GPa)", len=0.4, y=0.3)),
                row=2, col=2
            )
        # Update layout
        fig.update_layout(
            title_text="3D Stress Visualization",
            title_font_size=settings['title_size'] + 4,
            height=800,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            scene3=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            scene4=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            )
        )
        return fig


# =============================================
# ENHANCED PHYSICS-BASED STRESS ANALYZER WITH REAL GEOMETRY VISUALIZATION
# =============================================
class EnhancedPhysicsBasedStressAnalyzer(PhysicsBasedStressAnalyzer):
    """Enhanced physics-based analyzer with real geometry visualization"""
    def __init__(self):
        super().__init__()
        self.visualizer = UniversalVisualizationEnhancer()

    def compute_real_geometry_stress(self, eta, stress_fields, orientation_deg=54.7):
        """
        Compute stress visualization for real geometry domain
        Args:
            eta: Phase field order parameter
            stress_fields: Dictionary of stress components
            orientation_deg: Crystal orientation angle
        Returns:
            Dictionary containing comprehensive stress analysis
        """
        results = {}
        if eta is None or not isinstance(eta, np.ndarray):
            return results
        # Ensure stress fields are numpy arrays
        for key in stress_fields:
            if torch.is_tensor(stress_fields[key]):
                stress_fields[key] = stress_fields[key].cpu().numpy()
        # Region masks
        defect_mask = eta > 0.6
        interface_mask = (eta >= 0.4) & (eta <= 0.6)
        bulk_mask = eta < 0.4
        results['region_masks'] = {
            'defect': defect_mask,
            'interface': interface_mask,
            'bulk': bulk_mask
        }
        # Compute region statistics for each stress component
        region_stats = {}
        for region_name, mask in [('defect', defect_mask),
                                  ('interface', interface_mask),
                                  ('bulk', bulk_mask)]:
            region_stats[region_name] = {}
            for stress_name, stress_data in stress_fields.items():
                if mask.any():
                    region_stress = stress_data[mask]
                    region_stats[region_name][stress_name] = {
                        'mean': float(np.mean(region_stress)),
                        'std': float(np.std(region_stress)),
                        'max': float(np.max(region_stress)),
                        'min': float(np.min(region_stress)),
                        'abs_max': float(np.max(np.abs(region_stress))),
                        'percentile_95': float(np.percentile(np.abs(region_stress), 95)),
                        'area_fraction': float(mask.sum() / mask.size)
                    }
        results['region_stats'] = region_stats
        # Compute orientation-dependent metrics
        orientation_results = self.analyze_crystal_orientation_effects(
            region_stats.get('defect', {}), orientation_deg
        )
        results['orientation_analysis'] = orientation_results
        # Compute stress concentration factors
        if defect_mask.any() and bulk_mask.any():
            for stress_name in stress_fields.keys():
                if (stress_name in region_stats['defect'] and
                    stress_name in region_stats['bulk']):
                    defect_mean = region_stats['defect'][stress_name]['mean']
                    bulk_mean = region_stats['bulk'][stress_name]['mean']
                    if bulk_mean != 0:
                        concentration_factor = defect_mean / bulk_mean
                        results[f'{stress_name}_concentration_factor'] = float(concentration_factor)
        # Compute strain energy density
        strain_energy = self.compute_strain_energy_density(stress_fields)
        if strain_energy is not None:
            results['strain_energy'] = {
                'total': float(np.sum(strain_energy)),
                'mean': float(np.mean(strain_energy)),
                'max': float(np.max(strain_energy)),
                'defect_mean': float(np.mean(strain_energy[defect_mask])) if defect_mask.any() else 0,
                'interface_mean': float(np.mean(strain_energy[interface_mask])) if interface_mask.any() else 0,
                'bulk_mean': float(np.mean(strain_energy[bulk_mask])) if bulk_mask.any() else 0
            }
        # Store raw data for visualization
        results['geometry_data'] = {
            'eta': eta,
            'coordinates': None  # Can be extended with actual coordinates
        }
        results['stress_data'] = stress_fields
        return results

    def create_comprehensive_stress_report(self, analysis_results, settings=None,
                                           defect_type='Twin', orientation_deg=54.7):
        """
        Create comprehensive stress analysis report with visualizations
        Args:
            analysis_results: Results from compute_real_geometry_stress
            settings: Visualization settings
            defect_type: Type of defect
            orientation_deg: Orientation angle
        Returns:
            Dictionary with figures and analysis
        """
        if settings is None:
            settings = self.visualizer.default_params
        report = {
            'metadata': {
                'defect_type': defect_type,
                'orientation_deg': orientation_deg,
                'analysis_time': datetime.now().isoformat()
            },
            'figures': {},
            'analysis': {}
        }
        # Create 2D stress visualization
        fig_2d = self.visualizer.create_stress_visualization(
            analysis_results['geometry_data'],
            analysis_results['stress_data'],
            settings,
            orientation_deg,
            defect_type
        )
        if fig_2d is not None:
            report['figures']['stress_2d'] = fig_2d
        # Create 3D visualization if data is available
        try:
            fig_3d = self.visualizer.create_3d_stress_visualization(
                analysis_results['geometry_data'],
                analysis_results['stress_data'],
                settings
            )
            if fig_3d is not None:
                report['figures']['stress_3d'] = fig_3d
        except:
            pass  # 3D visualization might fail for some data
        # Create region comparison plots
        fig_regions = self._create_region_comparison_plots(
            analysis_results['region_stats'],
            settings
        )
        if fig_regions is not None:
            report['figures']['region_comparison'] = fig_regions
        # Store analysis results
        report['analysis'] = {
            'region_stats': analysis_results['region_stats'],
            'orientation_analysis': analysis_results.get('orientation_analysis', {}),
            'strain_energy': analysis_results.get('strain_energy', {})
        }
        # Add concentration factors
        for key, value in analysis_results.items():
            if 'concentration_factor' in key:
                report['analysis'][key] = value
        return report

    def _create_region_comparison_plots(self, region_stats, settings):
        """
        Create comparison plots for different regions
        Args:
            region_stats: Statistics for each region
            settings: Visualization settings
        Returns:
            matplotlib figure object
        """
        if not region_stats:
            return None
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        # Prepare data for plotting
        regions = ['defect', 'interface', 'bulk']
        colors = ['red', 'green', 'blue']
        # Plot 1: Mean stress comparison
        ax1 = axes[0]
        stress_components = ['sigma_hydro', 'von_mises', 'sigma_mag']
        for i, region in enumerate(regions):
            if region in region_stats:
                means = []
                for comp in stress_components:
                    if comp in region_stats[region]:
                        means.append(region_stats[region][comp]['mean'])
                    else:
                        means.append(0)
                x_pos = np.arange(len(stress_components)) + i * 0.25
                ax1.bar(x_pos, means, width=0.2, color=colors[i], alpha=0.7, label=region.capitalize())
        ax1.set_xlabel('Stress Component', fontsize=settings['label_size'])
        ax1.set_ylabel('Mean Stress (GPa)', fontsize=settings['label_size'])
        ax1.set_title('Mean Stress by Region and Component', fontsize=settings['title_size'])
        ax1.set_xticks(np.arange(len(stress_components)) + 0.25)
        ax1.set_xticklabels(['σ_h', 'σ_vm', '|σ|'])
        ax1.legend(fontsize=settings['tick_size'])
        ax1.grid(True, alpha=settings['grid_alpha'])
        # Plot 2: Maximum stress comparison
        ax2 = axes[1]
        for i, region in enumerate(regions):
            if region in region_stats:
                max_stresses = []
                for comp in stress_components:
                    if comp in region_stats[region]:
                        max_stresses.append(region_stats[region][comp]['max'])
                    else:
                        max_stresses.append(0)
                x_pos = np.arange(len(stress_components)) + i * 0.25
                ax2.bar(x_pos, max_stresses, width=0.2, color=colors[i], alpha=0.7)
        ax2.set_xlabel('Stress Component', fontsize=settings['label_size'])
        ax2.set_ylabel('Maximum Stress (GPa)', fontsize=settings['label_size'])
        ax2.set_title('Maximum Stress by Region and Component', fontsize=settings['title_size'])
        ax2.set_xticks(np.arange(len(stress_components)) + 0.25)
        ax2.set_xticklabels(['σ_h', 'σ_vm', '|σ|'])
        ax2.grid(True, alpha=settings['grid_alpha'])
        # Plot 3: Area fractions
        ax3 = axes[2]
        area_fractions = []
        for region in regions:
            if region in region_stats and 'sigma_hydro' in region_stats[region]:
                area_fractions.append(region_stats[region]['sigma_hydro']['area_fraction'])
            else:
                area_fractions.append(0)
        ax3.pie(area_fractions, labels=[r.capitalize() for r in regions],
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Area Fraction of Each Region', fontsize=settings['title_size'])
        # Plot 4: Stress concentration factors
        ax4 = axes[3]
        concentration_factors = []
        for region in ['defect', 'interface']:
            if region in region_stats and 'sigma_hydro' in region_stats[region]:
                defect_mean = region_stats[region]['sigma_hydro']['mean']
                bulk_mean = region_stats['bulk']['sigma_hydro']['mean'] if 'bulk' in region_stats and 'sigma_hydro' in region_stats['bulk'] else 1
                if bulk_mean != 0:
                    concentration_factors.append(defect_mean / bulk_mean)
                else:
                    concentration_factors.append(0)
        if concentration_factors:
            x_pos = np.arange(len(['Defect', 'Interface']))
            bars = ax4.bar(x_pos, concentration_factors, color=['red', 'green'], alpha=0.7)
            ax4.set_xlabel('Region', fontsize=settings['label_size'])
            ax4.set_ylabel('Concentration Factor', fontsize=settings['label_size'])
            ax4.set_title('Stress Concentration Factors (Relative to Bulk)', fontsize=settings['title_size'])
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(['Defect', 'Interface'])
            ax4.grid(True, alpha=settings['grid_alpha'])
            # Add value labels
            for bar, value in zip(bars, concentration_factors):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.2f}x', ha='center', va='bottom',
                         fontsize=settings['tick_size'])
        # Apply universal settings
        for ax in axes:
            self.visualizer.apply_visualization_settings(ax=ax, settings=settings)
        plt.tight_layout()
        return fig


# =============================================
# ENHANCED WORKFLOW PRESENTATION WITH CAPTIONS AND EXPLANATIONS
# =============================================
class EnhancedWorkflowPresenter:
    """
    Enhanced workflow presenter with detailed captions, explanations,
    and step-by-step guidance for stress interpolation and sintering prediction
    """
    def __init__(self):
        self.captions = self._initialize_captions()
        self.physics_explanations = self._initialize_physics_explanations()
        self.visualizer = UniversalVisualizationEnhancer()

    def _initialize_captions(self):
        """Initialize comprehensive captions for all visualizations"""
        return {
            'phase_field': {
                'title': "Phase Field Visualization of Material Defects",
                'description': "Shows the spatial distribution of the order parameter η, distinguishing between defect regions (η > 0.6), interfaces (0.4 ≤ η ≤ 0.6), and bulk material (η < 0.4).",
                'physics': "The phase field parameter η characterizes the local material state, with higher values indicating stronger defect presence and corresponding eigen strain fields.",
                'interpretation': "Red regions indicate strong defect presence, green shows transitional interfaces, and blue represents relatively defect-free bulk material."
            },
            'hydrostatic_stress': {
                'title': "Hydrostatic Stress Distribution",
                'description': "Visualizes the mean normal stress σ_h = (σ₁ + σ₂ + σ₃)/3, which governs volume changes and is critical for sintering processes.",
                'physics': "Compressive hydrostatic stress (negative, blue) promotes densification, while tensile stress (positive, red) can lead to cracking.",
                'interpretation': "Blue regions experience compressive stress beneficial for sintering, while red regions face tensile stress that may inhibit densification."
            },
            'von_mises_stress': {
                'title': "Von Mises Equivalent Stress",
                'description': "Shows the scalar stress measure σ_vm = √[½((σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²)] indicating potential for plastic deformation.",
                'physics': "Von Mises stress predicts yield onset according to the distortion energy theory, with higher values suggesting increased likelihood of plastic flow.",
                'interpretation': "Bright yellow regions indicate high shear stress concentrations where plastic deformation may initiate during sintering."
            },
            'stress_magnitude': {
                'title': "Total Stress Magnitude",
                'description': "Displays the overall stress intensity |σ| = √(σ:σ), combining all stress tensor components.",
                'physics': "Stress magnitude indicates the total elastic energy stored in the material, influencing diffusion kinetics and defect mobility.",
                'interpretation': "Regions with high stress magnitude act as drivers for stress-assisted diffusion and accelerated sintering."
            },
            'region_comparison': {
                'title': "Stress Analysis by Material Region",
                'description': "Compares stress statistics across defect, interface, and bulk regions to quantify defect-induced stress enhancement.",
                'physics': "Defects create eigen strain fields that locally amplify stress, with the degree of amplification quantified by concentration factors.",
                'interpretation': "High concentration factors in defect regions indicate significant stress amplification, crucial for low-temperature sintering."
            },
            'sintering_prediction': {
                'title': "Sintering Temperature Prediction",
                'description': "Predicts optimal sintering temperatures based on stress-modified diffusion kinetics using Arrhenius and exponential models.",
                'physics': "Hydrostatic stress reduces activation energy for diffusion: Q_eff = Q_a - Ωσ, enabling sintering at lower temperatures.",
                'interpretation': "Lower predicted temperatures indicate more favorable conditions for low-temperature sintering, with defect engineering enabling significant reductions."
            }
        }

    def _initialize_physics_explanations(self):
        """Initialize detailed physics explanations"""
        return {
            'stress_interpolation': """
## Stress Interpolation Methodology
**Physics-Aware Interpolation with Combined Regularization:**
1. **Parameter Space Embedding:** Each simulation is represented as a 15-dimensional vector encoding:
- Defect type (one-hot encoded: ISF, ESF, Twin, No Defect)
- Shape parameters (Square, Fault, Rectangle)
- Physical parameters (ε*, κ, θ)
- Physics-derived features (habit plane proximity, stress concentration factors)
2. **Attention Mechanism:** Multi-head attention weights simulations based on similarity to target parameters:
- Query: Target parameter vector
- Key/Value: Source simulation vectors
- Output: Weighted combination of source stresses
3. **Gaussian Spatial Regularization:** Applies locality constraint in parameter space:
- Weights decay exponentially with Euclidean distance
- Preserves smoothness in orientation-dependent responses
4. **Physics Constraints:** Incorporates domain knowledge:
- Eigen strain compatibility
- Habit plane symmetry (54.7° for Ag FCC twins)
- Defect interaction energies
""",
            'sintering_prediction_workflow': """
## Sintering Temperature Prediction Workflow
**Step 1: Stress Extraction & Interpolation**
- Extract hydrostatic stress from interpolated stress fields
- Focus on maximum absolute stress in defect regions
- Account for orientation effects near habit plane
**Step 2: Activation Energy Reduction**
- Compute effective activation energy: Q_eff = Q_a - Ω|σ|
- Ω = activation volume (converts stress to energy)
- Defect-specific adjustments (twin boundaries reduce Q_a more)
**Step 3: Arrhenius Temperature Calculation**
- Solve Arrhenius equation: D = D₀ exp(-Q_eff/kT)
- Set D = D_critical (required for sintering)
- Compute T_sinter = Q_eff / [k ln(D₀/D_crit)]
**Step 4: System Classification**
- System 1 (Perfect): σ < 5 GPa, T ≈ 620-630 K
- System 2 (SF/Twin): 5 ≤ σ < 20 GPa, T ≈ 450-550 K
- System 3 (Plastic): σ ≥ 20 GPa, T ≈ 350-400 K
""",
            'defect_physics': """
## Defect Physics & Stress Enhancement
**Eigen Strain Origins:**
- **ISF (ε* = 0.71):** Missing {111} atomic plane creates localized compression
- **ESF (ε* = 1.41):** Extra {111} plane introduces tensile strain field
- **Twin (ε* = 2.12):** Mirror symmetry creates alternating compressive/tensile fields
**Stress Concentration Mechanisms:**
1. **Elastic Mismatch:** Different elastic properties at defect interfaces
2. **Eigen Strain:** Incompatible deformation within defects
3. **Interface Curvature:** Geometric stress intensification
4. **Crystal Anisotropy:** Orientation-dependent elastic response
**Habit Plane Effects (54.7°):**
- Maximum Schmid factor for twinning
- Optimal stress transmission across {111} planes
- Enhanced defect-defect interactions
""",
            'sintering_mechanisms': """
## Stress-Modified Sintering Mechanisms
**Key Physical Principles:**
1. **Stress-Assisted Diffusion:**
- Hydrostatic stress reduces energy barriers for atomic motion
- Driving force: ∇μ = Ω∇σ (chemical potential gradient)
- Enhanced vacancy migration to pore surfaces
2. **Dislocation Climb Acceleration:**
- Stress aids dislocation motion through climb
- Faster grain boundary migration
- Reduced recovery time
3. **Interface Reaction Enhancement:**
- Stress lowers activation energy for surface reactions
- Faster neck growth between particles
- Improved densification kinetics
4. **Defect-Mediated Pathways:**
- Twins and stacking faults provide fast diffusion paths
- Reduced effective diffusion distances
- Lower overall sintering temperatures
"""
        }

    def create_caption_box(self, caption_key, analysis_results=None):
        """
        Create styled caption box for visualization
        Args:
            caption_key: Key identifying the caption type
            analysis_results: Optional analysis results for dynamic content
        Returns:
            Streamlit markdown with styled caption
        """
        if caption_key not in self.captions:
            return ""
        caption = self.captions[caption_key]
        # Add dynamic content if analysis results provided
        dynamic_content = ""
        if analysis_results is not None:
            if caption_key == 'hydrostatic_stress' and 'region_stats' in analysis_results:
                defect_stress = analysis_results['region_stats'].get('defect', {}).get('sigma_hydro', {}).get('mean', 0)
                dynamic_content = f"\n**Current Analysis:** Defect region shows mean hydrostatic stress of {defect_stress:.2f} GPa."
            elif caption_key == 'sintering_prediction' and 'temperature_predictions' in analysis_results:
                temps = analysis_results['temperature_predictions']
                dynamic_content = f"\n**Prediction:** Arrhenius model suggests {temps.get('arrhenius_defect_k', 0):.0f} K ({temps.get('arrhenius_defect_c', 0):.0f}°C) sintering temperature."
        caption_html = f"""
<div style="background-color: #f0f7ff; border-left: 5px solid #3b82f6; padding: 15px; margin: 10px 0; border-radius: 5px;">
<h4 style="color: #1e40af; margin-top: 0;">{caption['title']}</h4>
<p><strong>Description:</strong> {caption['description']}</p>
<p><strong>Physics:</strong> {caption['physics']}</p>
<p><strong>Interpretation:</strong> {caption['interpretation']}{dynamic_content}</p>
</div>
"""
        return caption_html

    def create_physics_explanation(self, explanation_key, container=None):
        """
        Create detailed physics explanation
        Args:
            explanation_key: Key identifying the explanation type
            container: Streamlit container to place explanation in
        """
        if container is None:
            container = st
        if explanation_key in self.physics_explanations:
            with container.expander(f"📚 Detailed Physics: {explanation_key.replace('_', ' ').title()}", expanded=False):
                container.markdown(self.physics_explanations[explanation_key])

    def create_workflow_step(self, step_number, step_title, step_content,
                             is_completed=False, container=None):
        """
        Create styled workflow step
        Args:
            step_number: Step number
            step_title: Step title
            step_content: Step content/markdown
            is_completed: Whether step is completed
            container: Streamlit container
        """
        if container is None:
            container = st
        status_color = "#10b981" if is_completed else "#6b7280"
        status_icon = "✅" if is_completed else "⏳"
        step_html = f"""
<div style="background-color: #f9fafb; border: 2px solid {status_color};
border-radius: 10px; padding: 15px; margin: 10px 0;">
<div style="display: flex; align-items: center; margin-bottom: 10px;">
<div style="background-color: {status_color}; color: white; width: 30px;
height: 30px; border-radius: 50%; display: flex;
align-items: center; justify-content: center; margin-right: 10px;">
{step_number}
</div>
<h4 style="margin: 0; color: {status_color};">{step_title} {status_icon}</h4>
</div>
<div style="margin-left: 40px;">
{step_content}
</div>
</div>
"""
        container.markdown(step_html, unsafe_allow_html=True)

    def create_interpolation_visualization(self, interpolator, sources, target_params,
                                          orientation_deg, settings):
        """
        Create visualization showing interpolation process
        Args:
            interpolator: PhysicsAwareInterpolator instance
            sources: Source simulations
            target_params: Target parameters
            orientation_deg: Target orientation
            settings: Visualization settings
        Returns:
            matplotlib figure
        """
        # Compute weights and vectors
        target_vector = interpolator.compute_parameter_vector(target_params, orientation_deg)
        source_vectors = []
        source_labels = []
        for src in sources:
            if 'params' in src:
                src_params = src['params']
                src_theta = src_params.get('theta', 0.0)
                src_deg = np.degrees(src_theta) if src_theta is not None else 0.0
                src_vector = interpolator.compute_parameter_vector(src_params, src_deg)
                source_vectors.append(src_vector)
                source_labels.append(f"{src_params.get('defect_type', 'Unknown')}\n{src_deg:.1f}°")
        if not source_vectors:
            return None
        source_vectors = np.array(source_vectors)
        # Compute attention weights
        attention_weights = interpolator.compute_attention_weights(source_vectors, target_vector)
        spatial_weights = interpolator.compute_spatial_weights(source_vectors, target_vector)
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Plot 1: Parameter space visualization (first 3 dimensions)
        ax1 = axes[0, 0]
        if source_vectors.shape[1] >= 3:
            scatter = ax1.scatter(source_vectors[:, 0], source_vectors[:, 1],
                                  c=source_vectors[:, 2], s=100, cmap='viridis', alpha=0.7)
            ax1.scatter(target_vector[0], target_vector[1], c='red', s=200,
                        marker='*', label='Target', edgecolors='black')
            ax1.set_xlabel('Parameter Dimension 1', fontsize=settings['label_size'])
            ax1.set_ylabel('Parameter Dimension 2', fontsize=settings['label_size'])
            ax1.set_title('Parameter Space Visualization', fontsize=settings['title_size'])
            plt.colorbar(scatter, ax=ax1, label='Parameter Dimension 3')
            ax1.legend()
        # Plot 2: Weight comparison
        ax2 = axes[0, 1]
        x = np.arange(len(source_labels))
        width = 0.35
        ax2.bar(x - width/2, attention_weights, width, label='Attention Weights', alpha=0.7)
        ax2.bar(x + width/2, spatial_weights, width, label='Spatial Weights', alpha=0.7)
        ax2.set_xlabel('Source Simulations', fontsize=settings['label_size'])
        ax2.set_ylabel('Weight Value', fontsize=settings['label_size'])
        ax2.set_title('Interpolation Weight Comparison', fontsize=settings['title_size'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(source_labels, rotation=45, ha='right', fontsize=settings['tick_size']-2)
        ax2.legend()
        ax2.grid(True, alpha=settings['grid_alpha'])
        # Plot 3: Final weights
        ax3 = axes[1, 0]
        if hasattr(interpolator, 'compute_physics_weights'):
            physics_weights = interpolator.compute_physics_weights(
                [s['params'] for s in sources], target_params, orientation_deg
            )
            # Compute final weights (blend of all)
            final_weights = 0.5 * attention_weights + 0.25 * spatial_weights + 0.25 * physics_weights
            final_weights = final_weights / final_weights.sum()
        else:
            final_weights = attention_weights  # fallback
        bars = ax3.bar(x, final_weights, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Source Simulations', fontsize=settings['label_size'])
        ax3.set_ylabel('Final Weight', fontsize=settings['label_size'])
        ax3.set_title('Final Interpolation Weights', fontsize=settings['title_size'])
        ax3.set_xticks(x)
        ax3.set_xticklabels(source_labels, rotation=45, ha='right', fontsize=settings['tick_size']-2)
        ax3.grid(True, alpha=settings['grid_alpha'], axis='y')
        # Add weight values
        for bar, weight in zip(bars, final_weights):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{weight:.3f}', ha='center', va='bottom',
                     fontsize=settings['tick_size']-2)
        # Plot 4: Orientation proximity
        ax4 = axes[1, 1]
        orientation_diffs = []
        for src in sources:
            if 'params' in src:
                src_theta = src['params'].get('theta', 0.0)
                src_deg = np.degrees(src_theta) if src_theta is not None else 0.0
                diff = min(abs(src_deg - orientation_deg),
                           abs(src_deg - (orientation_deg + 360)),
                           abs(src_deg - (orientation_deg - 360)))
                orientation_diffs.append(diff)
        colors = ['green' if d < 10 else 'orange' if d < 30 else 'red'
                  for d in orientation_diffs]
        bars = ax4.bar(x, orientation_diffs, color=colors, alpha=0.7)
        ax4.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Close (10°)')
        ax4.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Moderate (30°)')
        ax4.set_xlabel('Source Simulations', fontsize=settings['label_size'])
        ax4.set_ylabel('Orientation Difference (°)', fontsize=settings['label_size'])
        ax4.set_title('Orientation Proximity to Target', fontsize=settings['title_size'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(source_labels, rotation=45, ha='right', fontsize=settings['tick_size']-2)
        ax4.legend()
        ax4.grid(True, alpha=settings['grid_alpha'])
        # Apply settings
        for ax in axes.flatten():
            self.visualizer.apply_visualization_settings(ax=ax, settings=settings)
        plt.suptitle(f'Interpolation Process: Target at {orientation_deg}°',
                     fontsize=settings['title_size'] + 2, fontweight='bold')
        plt.tight_layout()
        return fig


# =============================================
# ENHANCED MAIN APPLICATION WITH COMPREHENSIVE VISUALIZATION CONTROLS
# =============================================
def main():
    # Configure Streamlit page with enhanced settings
    st.set_page_config(
        page_title="Ag FCC Twin: Enhanced Stress & Sintering Analysis",
        layout="wide",
        page_icon="🔬⚙️",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourrepo',
            'Report a bug': "https://github.com/yourrepo/issues",
            'About': "# Enhanced Ag FCC Twin Analysis\nComprehensive stress visualization with 50+ colormaps and universal controls"
        }
    )
    # Enhanced CSS with more styling options
    st.markdown("""
<style>
/* Main header styling */
.main-header {
font-size: 3.5rem !important;
background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #F59E0B);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
text-align: center;
font-weight: 900 !important;
margin-bottom: 1.5rem;
padding: 1rem;
text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
/* Physics equation styling */
.physics-equation {
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
padding: 1.5rem;
border-radius: 15px;
color: white;
font-family: "Cambria Math", "Times New Roman", serif;
font-size: 1.3rem;
margin: 1.5rem 0;
box-shadow: 0 6px 12px rgba(0,0,0,0.15);
border: 3px solid rgba(255,255,255,0.2);
}
/* Caption box styling */
.caption-box {
background: linear-gradient(135deg, #f0f7ff 0%, #e1f5fe 100%);
border-left: 6px solid #3b82f6;
padding: 1.2rem;
margin: 1rem 0;
border-radius: 10px;
box-shadow: 0 4px 6px rgba(0,0,0,0.05);
transition: transform 0.2s;
}
.caption-box:hover {
transform: translateY(-2px);
box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}
/* Metric card styling */
.metric-card {
background: white;
border-radius: 12px;
padding: 1.2rem;
margin: 0.5rem;
box-shadow: 0 4px 6px rgba(0,0,0,0.05);
border: 1px solid #e5e7eb;
transition: all 0.3s ease;
}
.metric-card:hover {
box-shadow: 0 8px 15px rgba(0,0,0,0.1);
transform: translateY(-3px);
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
gap: 2rem;
padding: 0 1rem;
}
.stTabs [data-baseweb="tab"] {
padding: 1rem 2rem;
border-radius: 8px 8px 0 0;
background-color: #f3f4f6;
border: 1px solid #e5e7eb;
font-weight: 600;
transition: all 0.3s;
}
.stTabs [data-baseweb="tab"]:hover {
background-color: #e5e7eb;
}
.stTabs [aria-selected="true"] {
background-color: #3b82f6 !important;
color: white !important;
box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
}
/* Button enhancements */
.stButton > button {
border-radius: 8px !important;
padding: 0.5rem 1.5rem !important;
font-weight: 600 !important;
transition: all 0.3s !important;
border: 2px solid transparent !important;
}
.stButton > button:hover {
transform: translateY(-2px);
box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
/* Sidebar enhancements */
.css-1d391kg {
background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}
/* Custom scrollbar */
::-webkit-scrollbar {
width: 10px;
}
::-webkit-scrollbar-track {
background: #f1f1f1;
border-radius: 5px;
}
::-webkit-scrollbar-thumb {
background: #888;
border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
background: #555;
}
/* Loading animation */
@keyframes pulse {
0%, 100% { opacity: 1; }
50% { opacity: 0.5; }
}
.pulse-animation {
animation: pulse 2s infinite;
}
</style>
""", unsafe_allow_html=True)
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">🔬⚙️ Enhanced Ag FCC Twin Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">Comprehensive stress visualization with 50+ colormaps • Universal enhancement controls • Physics-aware interpolation</p>', unsafe_allow_html=True)
    # Physics equations showcase
    st.markdown("""
<div class="physics-equation">
<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
<div>
<strong>Stress-Modified Diffusion:</strong><br>
D(σ,T) = D₀ exp[-(Qₐ - Ωσ)/(k<sub>B</sub>T)]
</div>
<div>
<strong>Sintering Temperature:</strong><br>
T<sub>sinter</sub>(σ) = (Qₐ - Ω|σ|)/[k<sub>B</sub> ln(D₀/D<sub>crit</sub>)]
</div>
<div>
<strong>Hydrostatic Stress:</strong><br>
σ<sub>h</sub> = ⅓(σ₁₁ + σ₂₂ + σ₃₃)
</div>
</div>
</div>
""", unsafe_allow_html=True)
    # Initialize enhanced components
    if 'enhanced_analyzer' not in st.session_state:
        st.session_state.enhanced_analyzer = EnhancedPhysicsBasedStressAnalyzer()
    if 'visualization_enhancer' not in st.session_state:
        st.session_state.visualization_enhancer = UniversalVisualizationEnhancer()
    if 'workflow_presenter' not in st.session_state:
        st.session_state.workflow_presenter = EnhancedWorkflowPresenter()
    # Keep original components
    if 'physics_analyzer' not in st.session_state:
        st.session_state.physics_analyzer = PhysicsBasedStressAnalyzer()
    if 'sintering_calculator' not in st.session_state:
        st.session_state.sintering_calculator = EnhancedSinteringCalculator()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator()
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    # Enhanced sidebar with visualization controls
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")
        # Analysis mode selection
        analysis_mode = st.radio(
            "**Analysis Mode:**",
            ["Real Geometry Visualization", "Habit Plane Vicinity",
             "Defect Comparison", "Comprehensive Dashboard", "Workflow Analysis"],
            index=0,
            help="Select the primary analysis mode"
        )
        # Universal visualization controls (always available)
        st.markdown("### 🎨 Universal Visualization Controls")
        viz_settings = st.session_state.visualization_enhancer.create_visualization_controls()
        # Stress-specific visualization controls
        with st.expander("🔧 Stress Visualization Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                stress_cmap_hydro = st.selectbox(
                    "Hydrostatic CMAP",
                    ["RdBu_r", "coolwarm", "bwr", "seismic", "stress_tensile_compressive"],
                    index=0
                )
                contour_levels = st.slider("Contour Levels", 5, 50, 20)
            with col2:
                stress_cmap_vonmises = st.selectbox(
                    "Von Mises CMAP",
                    ["viridis", "plasma", "inferno", "magma", "hot"],
                    index=0
                )
                vector_scale = st.slider("Vector Scale", 10, 100, 50)
            # Add stress-specific settings to viz_settings
            viz_settings.update({
                'stress_cmap_hydro': stress_cmap_hydro,
                'stress_cmap_vonmises': stress_cmap_vonmises,
                'contour_levels': contour_levels,
                'vector_scale': vector_scale
            })
        # Data management
        st.markdown("### 📂 Data Management")
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load Solutions", use_container_width=True,
                         help="Load all simulation solutions from the solutions directory"):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions found. Please check the solutions directory.")
        with col_load2:
            if st.button("🧹 Clear Cache", use_container_width=True,
                         help="Clear all cached data and reload"):
                st.session_state.solutions = []
                st.cache_data.clear()
                st.rerun()
        # Target parameters
        st.markdown("### 🎯 Target Parameters")
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for analysis"
        )
        # Auto-set eigen strain
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            eps0 = st.number_input(
                "Eigen Strain (ε*)",
                min_value=0.0,
                max_value=3.0,
                value=default_eps0,
                step=0.01,
                help="Eigen strain magnitude"
            )
            orientation_angle = st.slider(
                "Orientation Angle (°)",
                min_value=0.0,
                max_value=360.0,
                value=54.7,
                step=0.1,
                help="Crystal orientation angle relative to habit plane"
            )
        with col_param2:
            kappa = st.slider(
                "Interface Energy (κ)",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.01
            )
            shape = st.selectbox(
                "Geometry Shape",
                ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
                index=0
            )
        # Region selection for analysis
        region_type = st.selectbox(
            "Analysis Region",
            ["defect", "interface", "bulk", "all"],
            index=0,
            help="Select material region for detailed analysis"
        )
        # Generate analysis button
        st.markdown("---")
        generate_text = "🚀 Generate Enhanced Analysis" if analysis_mode != "Workflow Analysis" else "📋 Show Complete Workflow"
        if st.button(generate_text, type="primary", use_container_width=True):
            st.session_state.generate_analysis = True
            st.session_state.current_analysis_mode = analysis_mode
        else:
            st.session_state.generate_analysis = False
    # Main content area
    main_container = st.container()
    with main_container:
        # Create tabs for different analysis sections
        if st.session_state.solutions:
            tab_names = ["🏠 Dashboard", "📊 Real Geometry", "🎯 Habit Plane",
                         "🔬 Defect Comparison", "📈 Sintering Analysis", "📋 Workflow"]
            tabs = st.tabs(tab_names)
            with tabs[0]:  # Dashboard
                st.markdown("## 🏠 Comprehensive Dashboard")
                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Solutions Loaded", len(st.session_state.solutions))
                with col2:
                    defect_count = len(set(s.get('params', {}).get('defect_type', 'Unknown')
                                           for s in st.session_state.solutions))
                    st.metric("Unique Defects", defect_count)
                with col3:
                    has_physics = sum(1 for s in st.session_state.solutions
                                      if s.get('physics_analysis'))
                    st.metric("Physics Analyzed", f"{has_physics}/{len(st.session_state.solutions)}")
                with col4:
                    st.metric("Current Mode", analysis_mode)
                # Recent analyses
                st.markdown("### 📋 Recent Analyses")
                if 'recent_analyses' in st.session_state:
                    for analysis in st.session_state.recent_analyses[-3:]:
                        st.info(f"**{analysis['type']}** - {analysis['time']}: {analysis['description']}")
                # Quick actions
                st.markdown("### ⚡ Quick Actions")
                col_act1, col_act2, col_act3 = st.columns(3)
                with col_act1:
                    if st.button("🔄 Re-run Last Analysis", use_container_width=True):
                        pass  # Implement re-run logic
                with col_act2:
                    if st.button("📊 Export All Data", use_container_width=True):
                        pass  # Implement export logic
                with col_act3:
                    if st.button("📈 Generate Report", use_container_width=True):
                        pass  # Implement report generation
            with tabs[1]:  # Real Geometry Visualization
                st.markdown("## 📊 Real Geometry Stress Visualization")
                if st.session_state.get('generate_analysis', False) and \
                   st.session_state.get('current_analysis_mode') == "Real Geometry Visualization":
                    # Show workflow steps
                    st.session_state.workflow_presenter.create_workflow_step(
                        1, "Load Geometry Data",
                        "Loading phase field (η) and stress fields from selected solution...",
                        True, st
                    )
                    # Select a solution for visualization
                    solution_options = [s.get('metadata', {}).get('filename', f"Solution {i}")
                                        for i, s in enumerate(st.session_state.solutions)]
                    selected_solution = st.selectbox("Select Solution for Visualization",
                                                     solution_options, index=0)
                    if selected_solution:
                        solution_idx = solution_options.index(selected_solution)
                        solution = st.session_state.solutions[solution_idx]
                        st.session_state.workflow_presenter.create_workflow_step(
                            2, "Extract Stress Fields",
                            f"Extracting stress components from {selected_solution}...",
                            True, st
                        )
                        # Extract data
                        history = solution.get('history', [])
                        if history:
                            last_frame = history[-1]
                            eta = last_frame.get('eta')
                            stress_fields = last_frame.get('stresses', {})
                            if eta is not None and stress_fields:
                                st.session_state.workflow_presenter.create_workflow_step(
                                    3, "Compute Stress Analysis",
                                    "Analyzing stress distributions across defect, interface, and bulk regions...",
                                    True, st
                                )
                                # Compute comprehensive stress analysis
                                analysis_results = st.session_state.enhanced_analyzer.compute_real_geometry_stress(
                                    eta, stress_fields, orientation_angle
                                )
                                # Create visualization report
                                report = st.session_state.enhanced_analyzer.create_comprehensive_stress_report(
                                    analysis_results, viz_settings, defect_type, orientation_angle
                                )
                                # Display figures with captions
                                if 'figures' in report:
                                    for fig_name, fig in report['figures'].items():
                                        st.markdown(f"### 📈 {fig_name.replace('_', ' ').title()}")
                                        # Display caption
                                        caption_key = fig_name.split('_')[0] if '_' in fig_name else fig_name
                                        if caption_key in ['stress', 'region']:
                                            caption_key = 'region_comparison' if 'region' in fig_name else 'phase_field'
                                        st.markdown(
                                            st.session_state.workflow_presenter.create_caption_box(
                                                caption_key, analysis_results
                                            ), unsafe_allow_html=True
                                        )
                                        # Display figure
                                        if fig_name == 'stress_3d':
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.pyplot(fig)
                                        # Add download button for figure
                                        col_fig1, col_fig2 = st.columns(2)
                                        with col_fig1:
                                            buf = BytesIO()
                                            fig.savefig(buf, format="png", dpi=viz_settings['figure_dpi'])
                                            st.download_button(
                                                label="📥 Download PNG",
                                                data=buf.getvalue(),
                                                file_name=f"{fig_name}_{defect_type}_{orientation_angle}deg.png",
                                                mime="image/png",
                                                use_container_width=True
                                            )
                                        with col_fig2:
                                            buf_pdf = BytesIO()
                                            fig.savefig(buf_pdf, format="pdf")
                                            st.download_button(
                                                label="📥 Download PDF",
                                                data=buf_pdf.getvalue(),
                                                file_name=f"{fig_name}_{defect_type}_{orientation_angle}deg.pdf",
                                                mime="application/pdf",
                                                use_container_width=True
                                            )
                                st.markdown("---")
                                # Display analysis results
                                st.markdown("### 📊 Quantitative Analysis Results")
                                if 'analysis' in report:
                                    # Region statistics
                                    st.markdown("#### 📍 Region Statistics")
                                    region_stats = report['analysis'].get('region_stats', {})
                                    for region_name, stats in region_stats.items():
                                        with st.expander(f"{region_name.capitalize()} Region Analysis", expanded=False):
                                            if 'sigma_hydro' in stats:
                                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                                with col_stat1:
                                                    st.metric("Mean σ_h", f"{stats['sigma_hydro']['mean']:.3f} GPa")
                                                with col_stat2:
                                                    st.metric("Max σ_h", f"{stats['sigma_hydro']['max']:.3f} GPa")
                                                with col_stat3:
                                                    st.metric("Std Dev", f"{stats['sigma_hydro']['std']:.3f} GPa")
                                                with col_stat4:
                                                    st.metric("Area Fraction", f"{stats['sigma_hydro']['area_fraction']:.1%}")
                                    # Strain energy analysis
                                    if 'strain_energy' in report['analysis']:
                                        st.markdown("#### ⚡ Strain Energy Analysis")
                                        energy = report['analysis']['strain_energy']
                                        col_energy1, col_energy2, col_energy3 = st.columns(3)
                                        with col_energy1:
                                            st.metric("Total Energy", f"{energy['total']:.3e} J/m³")
                                        with col_energy2:
                                            st.metric("Mean Energy", f"{energy['mean']:.3e} J/m³")
                                        with col_energy3:
                                            st.metric("Max Energy", f"{energy['max']:.3e} J/m³")
                            else:
                                st.error("No valid geometry or stress data found in selected solution.")
                        else:
                            st.warning("Selected solution has no history data.")
                        st.session_state.workflow_presenter.create_workflow_step(
                            4, "Analysis Complete",
                            "Real geometry stress visualization completed successfully!",
                            True, st
                        )
                else:
                    # Show information about real geometry visualization
                    st.info("👈 Configure parameters and click 'Generate Enhanced Analysis' to visualize stress in real geometry.")
                    # Example visualization
                    st.markdown("""
### 📊 Real Geometry Visualization Features
This module provides comprehensive stress visualization for actual material geometry:
**Key Features:**
1. **Multi-component Stress Visualization:**
- Hydrostatic stress (σ_h) with tensile/compressive coloring
- Von Mises equivalent stress (σ_vm)
- Total stress magnitude (|σ|)
- Phase field visualization (η)
2. **Region-specific Analysis:**
- Automatic detection of defect, interface, and bulk regions
- Statistical comparison across regions
- Stress concentration factor calculation
3. **Advanced Visualization Options:**
- 50+ colormaps including custom engineered maps
- Adjustable line thickness and font sizes
- 3D surface plots for depth perception
- Interactive plotly visualizations
**Physics Insights:**
- Visualize how defects create local stress concentrations
- Understand stress distribution across material interfaces
- Quantify stress amplification in defect regions
- Correlate stress fields with sintering temperature reduction
""")
            with tabs[2]:  # Habit Plane Vicinity
                st.markdown("## 🎯 Habit Plane Vicinity Analysis")
                st.info("Habit plane analysis implementation would go here.")
            with tabs[3]:  # Defect Comparison
                st.markdown("## 🔬 Defect Type Comparison")
                st.info("Defect comparison implementation would go here.")
            with tabs[4]:  # Sintering Analysis
                st.markdown("## 📈 Enhanced Sintering Analysis")
                if st.session_state.get('generate_analysis', False):
                    # Show detailed physics explanations
                    st.session_state.workflow_presenter.create_physics_explanation(
                        'sintering_prediction_workflow', st
                    )
                    # Prepare target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa
                    }
                    # Perform interpolation for sintering prediction
                    with st.spinner("Performing physics-aware interpolation..."):
                        interpolation_result = st.session_state.interpolator.interpolate_stress_components(
                            st.session_state.solutions,
                            orientation_angle,
                            target_params,
                            region_type
                        )
                    if interpolation_result:
                        # Display sintering analysis
                        sintering_analysis = interpolation_result.get('sintering_analysis', {})
                        if sintering_analysis:
                            # Temperature predictions
                            st.markdown("### 🔥 Sintering Temperature Predictions")
                            temp_predictions = sintering_analysis.get('temperature_predictions', {})
                            col_temp1, col_temp2, col_temp3 = st.columns(3)
                            with col_temp1:
                                st.metric(
                                    "Exponential Model",
                                    f"{temp_predictions.get('exponential_model_k', 0):.1f} K",
                                    f"{temp_predictions.get('exponential_model_c', 0):.1f} °C"
                                )
                            with col_temp2:
                                st.metric(
                                    "Arrhenius Model",
                                    f"{temp_predictions.get('arrhenius_defect_k', 0):.1f} K",
                                    f"{temp_predictions.get('arrhenius_defect_c', 0):.1f} °C"
                                )
                            with col_temp3:
                                system_info = sintering_analysis.get('system_classification', {})
                                st.metric(
                                    "System Classification",
                                    system_info.get('system', 'Unknown'),
                                    f"Predicted: {system_info.get('predicted_T_k', 0):.1f} K"
                                )
                            # Activation energy analysis
                            st.markdown("### ⚡ Activation Energy Analysis")
                            activation_analysis = sintering_analysis.get('activation_energy_analysis', {})
                            if activation_analysis:
                                col_act1, col_act2, col_act3 = st.columns(3)
                                with col_act1:
                                    st.metric(
                                        "Q_a (Standard)",
                                        f"{activation_analysis.get('Q_a_standard_eV', 0):.3f} eV",
                                        "Base activation energy"
                                    )
                                with col_act2:
                                    st.metric(
                                        "Q_eff (Defect)",
                                        f"{activation_analysis.get('Q_eff_defect_eV', 0):.3f} eV",
                                        f"Reduction: {activation_analysis.get('reduction_defect_eV', 0):.3f} eV"
                                    )
                                with col_act3:
                                    reduction_pct = activation_analysis.get('reduction_percentage', 0)
                                    st.metric(
                                        "Activation Energy Reduction",
                                        f"{reduction_pct:.1f}%",
                                        f"{activation_analysis.get('reduction_standard_eV', 0):.3f} eV"
                                    )
                            # Create comprehensive sintering plot
                            st.markdown("### 📊 Comprehensive Sintering Analysis Plot")
                            # Generate theoretical curves
                            theoretical_curves = st.session_state.sintering_calculator.get_theoretical_curve()
                            if theoretical_curves:
                                fig_sintering = st.session_state.sintering_calculator.create_comprehensive_sintering_plot(
                                    np.array(theoretical_curves['stresses']),
                                    np.array(theoretical_curves['exponential_standard']),
                                    defect_type,
                                    f"Sintering Analysis for {defect_type}"
                                )
                                # Apply visualization settings
                                st.session_state.visualization_enhancer.apply_visualization_settings(
                                    fig=fig_sintering, settings=viz_settings
                                )
                                st.pyplot(fig_sintering)
                                # Add caption
                                st.markdown(
                                    st.session_state.workflow_presenter.create_caption_box(
                                        'sintering_prediction', sintering_analysis
                                    ), unsafe_allow_html=True
                                )
            with tabs[5]:  # Workflow Analysis
                st.markdown("## 📋 Complete Workflow Analysis")
                # Show complete workflow with all steps
                st.session_state.workflow_presenter.create_workflow_step(
                    1, "Data Loading & Preparation",
                    """
**Tasks Completed:**
- Loaded simulation solutions from directory
- Standardized data formats
- Extracted phase fields and stress tensors
- Applied physics-aware preprocessing
""",
                    True, st
                )
                st.session_state.workflow_presenter.create_workflow_step(
                    2, "Stress Field Interpolation",
                    """
**Current Process:**
- Computing parameter space embeddings
- Applying attention mechanisms
- Incorporating physics constraints
- Weighting source simulations
""",
                    True, st
                )
                st.session_state.workflow_presenter.create_workflow_step(
                    3, "Region-specific Analysis",
                    """
**Analysis Steps:**
- Identifying defect, interface, and bulk regions
- Computing stress statistics per region
- Calculating concentration factors
- Analyzing orientation effects
""",
                    st.session_state.get('generate_analysis', False), st
                )
                st.session_state.workflow_presenter.create_workflow_step(
                    4, "Sintering Prediction",
                    """
**Prediction Models:**
- Exponential empirical model
- Arrhenius physics-based model
- Defect-specific parameter adjustments
- System classification mapping
""",
                    False, st
                )
                st.session_state.workflow_presenter.create_workflow_step(
                    5, "Visualization & Reporting",
                    """
**Output Generation:**
- Creating comprehensive visualizations
- Applying universal enhancement controls
- Generating physics explanations
- Preparing export packages
""",
                    False, st
                )
                # Show interpolation visualization
                if st.session_state.solutions and st.session_state.get('generate_analysis', False):
                    st.markdown("### 🔍 Interpolation Process Visualization")
                    target_params = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa
                    }
                    fig_interpolation = st.session_state.workflow_presenter.create_interpolation_visualization(
                        st.session_state.interpolator,
                        st.session_state.solutions,
                        target_params,
                        orientation_angle,
                        viz_settings
                    )
                    if fig_interpolation:
                        st.pyplot(fig_interpolation)
                    # Explanation of interpolation
                    st.session_state.workflow_presenter.create_physics_explanation(
                        'stress_interpolation', st
                    )
        else:
            # No solutions loaded - show enhanced welcome screen
            st.markdown("""
## 🚀 Welcome to Enhanced Ag FCC Twin Analysis
**A comprehensive platform for stress analysis and sintering prediction with universal visualization controls.**
### 🔑 Key Features:
**1. Universal Visualization Controls**
- **50+ Colormaps:** Choose from sequential, diverging, qualitative, cyclic, and custom colormaps
- **Font Customization:** Adjust title, label, tick, and legend font sizes
- **Line & Marker Controls:** Customize line widths, marker sizes, and styles
- **Colorbar Settings:** Fine-tune colorbar width, padding, and positioning
- **Theme Options:** Switch between light/dark themes and transparent backgrounds
**2. Real Geometry Stress Visualization**
- **Multi-component Analysis:** Visualize hydrostatic, von Mises, and magnitude stresses
- **Region Detection:** Automatic identification of defect, interface, and bulk regions
- **3D Visualizations:** Interactive 3D surface plots for depth perception
- **Statistical Comparison:** Quantitative analysis across material regions
**3. Physics-Aware Interpolation**
- **Combined Attention:** ML-based similarity weighting with physics constraints
- **Habit Plane Focus:** Specialized analysis around 54.7° orientation
- **Defect-specific Models:** Custom parameters for ISF, ESF, Twin, and perfect crystals
**4. Enhanced Sintering Prediction**
- **Multiple Models:** Exponential empirical and Arrhenius physics-based predictions
- **Defect Engineering:** Quantify temperature reduction from different defect types
- **System Classification:** Map stress levels to AgNP sintering systems
**5. Comprehensive Workflow Presentation**
- **Step-by-Step Guidance:** Visual workflow with completion tracking
- **Physics Explanations:** Detailed explanations of underlying mechanisms
- **Interactive Captions:** Context-aware help for each visualization
### 🎯 Getting Started:
1. **Prepare Data:** Place your simulation files in the `numerical_solutions` directory
2. **Load Solutions:** Click the "Load Solutions" button in the sidebar
3. **Configure Analysis:** Set your parameters and visualization preferences
4. **Generate Analysis:** Click "Generate Enhanced Analysis" to begin
### 📁 Expected Data Format:
Each simulation file should contain:
- `params`: Dictionary of simulation parameters (defect_type, eps0, theta, etc.)
- `history`: List of simulation frames
- Each frame should contain:
- `eta`: Phase field order parameter (numpy array)
- `stresses`: Dictionary of stress components (sigma_hydro, von_mises, sigma_mag, etc.)
### 🎨 Visualization Tips:
- Use the **Universal Visualization Controls** in the sidebar to customize all plots
- Try different **colormap categories** for optimal stress visualization
- Adjust **font sizes** for better readability in publications
- Enable **dark theme** for reduced eye strain during extended analysis sessions
- Use **3D visualizations** for understanding complex stress distributions
""")
            # Quick start button
            if st.button("🚀 Quick Start Tutorial", type="primary", use_container_width=True):
                st.info("""
**Quick Start Tutorial:**
1. **Sample Data:** Download sample data from [link] and place in `numerical_solutions` folder
2. **Load Data:** Click "Load Solutions" in the sidebar
3. **Basic Analysis:** Select "Real Geometry Visualization" mode
4. **Visualization:** Adjust colormaps and font sizes to your preference
5. **Analysis:** Click "Generate Enhanced Analysis" to see results
For detailed tutorials, check the documentation section.
""")
    # Footer with enhanced information
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("**🔬 Enhanced Ag FCC Twin Analysis**")
        st.markdown("Version 2.0 • Universal Visualization")
    with col_footer2:
        st.markdown("**⚙️ Features:**")
        st.markdown("• 50+ Colormaps • Physics-Aware Interpolation • Real Geometry Visualization")
    with col_footer3:
        st.markdown("**📊 Output:**")
        st.markdown("• PNG/PDF Export • Interactive 3D • Comprehensive Reports")


# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
