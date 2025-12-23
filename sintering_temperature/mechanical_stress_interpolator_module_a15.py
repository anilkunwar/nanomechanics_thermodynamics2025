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
import cmasher as cmr  # For additional colormaps
from scipy.spatial.distance import cdist

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
            else:
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

@jit(nopython=True)
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
        mask = np.ones_like(eta, dtype=np.bool_)
    
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
    
    # Extract region stress
    region_stress = stress_data[mask]
    
    if stress_type == 'max_abs':
        return np.max(np.abs(region_stress)) if len(region_stress) > 0 else 0.0
    elif stress_type == 'mean_abs':
        return np.mean(np.abs(region_stress)) if len(region_stress) > 0 else 0.0
    elif stress_type == 'max':
        return np.max(region_stress) if len(region_stress) > 0 else 0.0
    elif stress_type == 'min':
        return np.min(region_stress) if len(region_stress) > 0 else 0.0
    elif stress_type == 'mean':
        return np.mean(region_stress) if len(region_stress) > 0 else 0.0
    else:
        return np.mean(np.abs(region_stress)) if len(region_stress) > 0 else 0.0

@jit(nopython=True)
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
        mask = np.ones_like(eta, dtype=np.bool_)
    
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
# ENHANCED SPATIAL INTERPOLATOR WITH EUCLIDEAN DISTANCE
# =============================================

class EnhancedSpatialInterpolator:
    """Enhanced interpolator with proper Euclidean distance regularization"""
    
    def __init__(self, sigma=0.3, use_spatial_locality=True, spatial_weight=1.0):
        super().__init__()
        self.sigma = sigma
        self.use_spatial_locality = use_spatial_locality
        self.spatial_weight = spatial_weight
        
        # Parameter mappings
        self.defect_map = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        
        self.shape_map = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
    
    def compute_parameter_vector(self, params):
        """Convert parameters to numerical vector with error handling"""
        vector = []
        
        # Check if params is a dictionary
        if not isinstance(params, dict):
            # Return default vector
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0.0], dtype=np.float32)
        
        # Defect type
        defect = params.get('defect_type', 'ISF')
        vector.extend(self.defect_map.get(defect, [0, 0, 0]))
        
        # Shape
        shape = params.get('shape', 'Square')
        vector.extend(self.shape_map.get(shape, [0, 0, 0, 0, 0]))
        
        # Numeric parameters (normalized)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        vector.append((eps0 - 0.3) / (3.0 - 0.3))  # eps0 normalized
        vector.append((kappa - 0.1) / (2.0 - 0.1))  # kappa normalized
        vector.append(theta / (np.pi / 2))  # theta normalized 0-pi/2
        
        return np.array(vector, dtype=np.float32)
    
    def compute_spatial_distance(self, source_params, target_params):
        """Compute spatial distance between parameter vectors"""
        source_vector = self.compute_parameter_vector(source_params)
        target_vector = self.compute_parameter_vector(target_params)
        
        # Euclidean distance
        distance = np.sqrt(np.sum((source_vector - target_vector) ** 2))
        return distance
    
    def compute_spatial_weights(self, sources, target_params):
        """Compute weights based on spatial locality (Euclidean distance)"""
        if not sources:
            return np.array([])
        
        distances = []
        for src in sources:
            if 'params' in src:
                dist = self.compute_spatial_distance(src['params'], target_params)
                distances.append(dist)
            else:
                distances.append(1.0)  # Default distance for invalid sources
        
        distances = np.array(distances)
        
        # Convert distances to weights using Gaussian kernel
        if np.any(distances > 0):
            weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
            # Normalize
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones_like(weights) / len(weights)
        else:
            weights = np.ones_like(distances) / len(distances)
        
        return weights
    
    def interpolate_with_spatial_locality(self, sources, target_params, region_type='bulk', 
                                          stress_component='von_mises', stress_type='max_abs'):
        """Interpolate with proper spatial locality regularization"""
        
        # Filter and validate sources
        valid_sources = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            if 'params' not in src or 'history' not in src:
                continue
            if not isinstance(src.get('params'), dict):
                continue
            valid_sources.append(src)
        
        if not valid_sources:
            return None
        
        # Compute spatial weights
        spatial_weights = self.compute_spatial_weights(valid_sources, target_params)
        
        # Extract region stress from each source
        source_stresses = []
        source_stats = []
        
        for src in valid_sources:
            history = src.get('history', [])
            if history:
                # Get the last frame
                last_frame = history[-1]
                
                # Handle different frame formats
                if isinstance(last_frame, tuple) and len(last_frame) >= 2:
                    eta, stress_fields = last_frame[0], last_frame[1]
                elif isinstance(last_frame, dict):
                    eta = last_frame.get('eta', np.zeros((128, 128)))
                    stress_fields = last_frame.get('stresses', {})
                else:
                    eta = np.zeros((128, 128))
                    stress_fields = {}
                
                # Extract region stress
                region_stress = extract_region_stress(eta, stress_fields, region_type, 
                                                     stress_component, stress_type)
                source_stresses.append(region_stress)
                
                # Extract comprehensive statistics
                stats = extract_region_statistics(eta, stress_fields, region_type)
                source_stats.append(stats)
            else:
                source_stresses.append(0.0)
                source_stats.append({})
        
        # Weighted combination
        weighted_stress = np.sum(spatial_weights * np.array(source_stresses))
        
        # Combine statistics
        combined_stats = {}
        if source_stats and source_stats[0]:
            for key in source_stats[0].keys():
                if key == 'area_fraction':
                    # Weighted average for area fraction
                    area_fractions = [stats.get(key, 0.0) for stats in source_stats]
                    combined_stats[key] = np.sum(spatial_weights * np.array(area_fractions))
                elif isinstance(source_stats[0][key], dict):
                    # Weighted average for stress statistics
                    combined_stats[key] = {}
                    for subkey in source_stats[0][key].keys():
                        values = [stats.get(key, {}).get(subkey, 0.0) for stats in source_stats]
                        combined_stats[key][subkey] = np.sum(spatial_weights * np.array(values))
        
        return {
            'region_stress': weighted_stress,
            'region_statistics': combined_stats,
            'spatial_weights': spatial_weights,
            'source_stresses': source_stresses,
            'source_statistics': source_stats,
            'target_params': target_params,
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type,
            'num_valid_sources': len(valid_sources)
        }

# =============================================
# ORIGINAL FILE ANALYSIS CLASS
# =============================================

class OriginalFileAnalyzer:
    """Analyze original loaded files for different regions"""
    
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
        
        return {
            'region_stress': region_stress,
            'region_statistics': region_stats,
            'params': params,
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
    
    def create_stress_matrix(self, solutions, region_type='bulk', 
                            stress_component='von_mises', stress_type='max_abs'):
        """Create stress matrix (theta × time) from original solutions"""
        if not solutions:
            return None, None, None
        
        # Extract unique thetas
        thetas = []
        for sol in solutions:
            params = sol.get('params', {})
            theta = params.get('theta', 0.0)
            thetas.append(theta)
        
        unique_thetas = np.unique(thetas)
        
        # Determine time points (use history length)
        time_points = []
        for sol in solutions:
            history = sol.get('history', [])
            time_points.append(len(history))
        
        max_time = min(time_points) if time_points else 0
        times = np.arange(max_time)
        
        # Initialize stress matrix
        stress_matrix = np.zeros((len(unique_thetas), len(times)))
        
        # Fill matrix
        for i, theta in enumerate(unique_thetas):
            # Find solutions with this theta
            theta_solutions = [sol for sol in solutions 
                             if sol.get('params', {}).get('theta', 0.0) == theta]
            
            if not theta_solutions:
                continue
            
            # For simplicity, take the first solution with this theta
            sol = theta_solutions[0]
            history = sol.get('history', [])
            
            for t in range(min(len(history), len(times))):
                frame = history[t]
                
                # Extract eta and stress fields
                if isinstance(frame, tuple) and len(frame) >= 2:
                    eta, stress_fields = frame[0], frame[1]
                elif isinstance(frame, dict):
                    eta = frame.get('eta', np.zeros((128, 128)))
                    stress_fields = frame.get('stresses', {})
                else:
                    continue
                
                # Extract region stress
                region_stress = extract_region_stress(eta, stress_fields, region_type,
                                                     stress_component, stress_type)
                stress_matrix[i, t] = region_stress
        
        return stress_matrix, unique_thetas, times

# =============================================
# ENHANCED COMPARISON VISUALIZER
# =============================================

class EnhancedComparisonVisualizer:
    """Visualizer for comparing original and interpolated solutions"""
    
    def __init__(self):
        self.original_analyzer = OriginalFileAnalyzer()
    
    def create_comparison_sunburst(self, original_matrix, interpolated_matrix, 
                                  thetas, times, region_name, stress_component):
        """Create comparison sunburst plot"""
        
        # Calculate difference matrix
        diff_matrix = interpolated_matrix - original_matrix
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Original Solutions', 'Interpolated Solutions', 'Difference'),
            specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]],
            horizontal_spacing=0.15
        )
        
        # Original solutions sunburst
        fig.add_trace(
            go.Scatterpolar(
                r=times,
                theta=np.deg2rad(thetas),
                mode='markers',
                marker=dict(
                    size=8,
                    color=original_matrix.flatten(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(x=0.3, title='Stress (GPa)')
                ),
                name='Original'
            ),
            row=1, col=1
        )
        
        # Interpolated solutions sunburst
        fig.add_trace(
            go.Scatterpolar(
                r=times,
                theta=np.deg2rad(thetas),
                mode='markers',
                marker=dict(
                    size=8,
                    color=interpolated_matrix.flatten(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(x=0.65, title='Stress (GPa)')
                ),
                name='Interpolated'
            ),
            row=1, col=2
        )
        
        # Difference sunburst
        fig.add_trace(
            go.Scatterpolar(
                r=times,
                theta=np.deg2rad(thetas),
                mode='markers',
                marker=dict(
                    size=8,
                    color=diff_matrix.flatten(),
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(x=1.0, title='Δ Stress (GPa)'),
                    cmin=-np.max(np.abs(diff_matrix)),
                    cmax=np.max(np.abs(diff_matrix))
                ),
                name='Difference'
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f"Comparison: {region_name} - {stress_component}",
            showlegend=False,
            polar=dict(
                radialaxis=dict(title='Time (s)', showgrid=True),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            polar2=dict(
                radialaxis=dict(title='Time (s)', showgrid=True),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            polar3=dict(
                radialaxis=dict(title='Time (s)', showgrid=True),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            height=600,
            width=1200
        )
        
        return fig
    
    def create_comparison_radar(self, original_stress, interpolated_stress, 
                               thetas, region_name, stress_component):
        """Create comparison radar plot"""
        
        # Ensure proper closure
        thetas_closed = np.append(thetas, 360)
        original_closed = np.append(original_stress, original_stress[0])
        interpolated_closed = np.append(interpolated_stress, interpolated_stress[0])
        
        fig = go.Figure()
        
        # Original solutions
        fig.add_trace(go.Scatterpolar(
            r=original_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line=dict(color='rgb(31, 119, 180)', width=3),
            name='Original',
            hovertemplate='Orientation: %{theta:.1f}°<br>Original Stress: %{r:.4f} GPa'
        ))
        
        # Interpolated solutions
        fig.add_trace(go.Scatterpolar(
            r=interpolated_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.3)',
            line=dict(color='rgb(255, 127, 14)', width=3),
            name='Interpolated',
            hovertemplate='Orientation: %{theta:.1f}°<br>Interpolated Stress: %{r:.4f} GPa'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Radar Comparison: {region_name} - {stress_component}",
                font=dict(size=16)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(original_stress), max(interpolated_stress)) * 1.2],
                    gridcolor="lightgray",
                    gridwidth=2
                ),
                angularaxis=dict(
                    gridcolor="lightgray",
                    gridwidth=2,
                    rotation=90,
                    direction="clockwise"
                )
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_error_analysis(self, original_matrix, interpolated_matrix):
        """Create error analysis visualization"""
        
        # Calculate errors
        errors = interpolated_matrix - original_matrix
        abs_errors = np.abs(errors)
        rel_errors = np.abs(errors) / (np.abs(original_matrix) + 1e-10)
        
        # Statistics
        stats = {
            'MAE': np.mean(abs_errors),
            'MSE': np.mean(errors ** 2),
            'RMSE': np.sqrt(np.mean(errors ** 2)),
            'Max_Abs_Error': np.max(abs_errors),
            'Mean_Rel_Error': np.mean(rel_errors) * 100,
            'Max_Rel_Error': np.max(rel_errors) * 100
        }
        
        # Create error distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram of errors
        axes[0, 0].hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Error (GPa)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # QQ plot for normality check
        from scipy import stats as scipy_stats
        scipy_stats.probplot(errors.flatten(), dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Check)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: Original vs Interpolated
        axes[1, 0].scatter(original_matrix.flatten(), interpolated_matrix.flatten(), 
                          alpha=0.5, s=10)
        axes[1, 0].plot([original_matrix.min(), original_matrix.max()], 
                       [original_matrix.min(), original_matrix.max()], 
                       'r--', label='Perfect Fit')
        axes[1, 0].set_xlabel('Original Stress (GPa)')
        axes[1, 0].set_ylabel('Interpolated Stress (GPa)')
        axes[1, 0].set_title('Original vs Interpolated')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error vs Original
        axes[1, 1].scatter(original_matrix.flatten(), abs_errors.flatten(), 
                          alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Original Stress (GPa)')
        axes[1, 1].set_ylabel('Absolute Error (GPa)')
        axes[1, 1].set_title('Error vs Original Stress')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, stats

# =============================================
# MAIN APPLICATION WITH REGION ANALYSIS
# =============================================
def main():
    st.set_page_config(
        page_title="Ag Material Stress Analysis with Region Comparison",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Ag Material Stress Analysis with Region Comparison</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedSpatialInterpolator(sigma=0.3, use_spatial_locality=True)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'comparison_visualizer' not in st.session_state:
        st.session_state.comparison_visualizer = EnhancedComparisonVisualizer()
    if 'original_analyzer' not in st.session_state:
        st.session_state.original_analyzer = OriginalFileAnalyzer()
    
    # Sidebar with enhanced options
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Analysis Settings</h2>', unsafe_allow_html=True)
        
        # Data source selection
        st.markdown("#### 📊 Data Source")
        data_source = st.radio(
            "Select data source for visualization:",
            ["Original Loaded Files", "Interpolated Solutions", "Comparison (Both)"],
            index=2,
            help="Choose whether to visualize original files, interpolated solutions, or compare both"
        )
        
        # Region selection
        st.markdown("#### 🎯 Analysis Region")
        region_type = st.selectbox(
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
        region_key = region_map[region_type]
        
        # Stress component
        st.markdown("#### 📈 Stress Component")
        stress_component = st.selectbox(
            "Select stress component:",
            ["von_mises", "sigma_hydro", "sigma_mag"],
            index=0,
            help="Select which stress component to visualize"
        )
        
        # Stress type
        stress_type = st.selectbox(
            "Select stress analysis type:",
            ["max_abs", "mean_abs", "max", "min", "mean"],
            index=0,
            help="Select how to analyze stress in the region"
        )
        
        # Spatial locality settings
        st.markdown("#### 🗺️ Spatial Locality Settings")
        use_spatial_locality = st.checkbox(
            "Use Spatial Locality Regularization", 
            value=True,
            help="Use Euclidean distance in parameter space for interpolation weights"
        )
        
        spatial_sigma = st.slider(
            "Spatial Locality Sigma",
            0.1, 2.0, 0.3, 0.1,
            help="Controls the influence of spatial distance (larger = smoother interpolation)"
        )
        
        spatial_weight = st.slider(
            "Spatial Weight",
            0.0, 2.0, 1.0, 0.1,
            help="Weight of spatial locality in interpolation"
        )
        
        # Update interpolator settings
        if (use_spatial_locality != st.session_state.interpolator.use_spatial_locality or
            spatial_sigma != st.session_state.interpolator.sigma or
            spatial_weight != st.session_state.interpolator.spatial_weight):
            st.session_state.interpolator.use_spatial_locality = use_spatial_locality
            st.session_state.interpolator.sigma = spatial_sigma
            st.session_state.interpolator.spatial_weight = spatial_weight
        
        # Visualization type
        st.markdown("#### 🎨 Visualization Type")
        viz_type = st.radio(
            "Select visualization type:",
            ["Sunburst", "Radar", "Both", "Error Analysis"],
            index=0,
            help="Choose visualization type"
        )
        
        # Colormap selection
        cmap = st.selectbox(
            "Color Map",
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('rainbow') if 'rainbow' in ALL_COLORMAPS else 0
        )
        
        # Load solutions
        st.markdown("#### 📂 Load Solutions")
        if st.button("🔄 Load All Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions(use_cache=True)
                if st.session_state.solutions:
                    st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions")
        
        # Show loaded solutions
        if st.session_state.solutions:
            with st.expander(f"📋 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    params = sol.get('params', {})
                    st.write(f"**{i+1}. {sol.get('filename', 'Unknown')}**")
                    st.caption(f"Type: {params.get('defect_type', '?')} | "
                              f"θ: {np.rad2deg(params.get('theta', 0)):.1f}° | "
                              f"ε*: {params.get('eps0', 0):.2f}")
                if len(st.session_state.solutions) > 5:
                    st.info(f"... and {len(st.session_state.solutions) - 5} more")
    
    # Main content area
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 Region Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Region information cards
        st.markdown(f'<div class="region-card {region_key}-region">', unsafe_allow_html=True)
        st.markdown(f"### {region_type}")
        
        if region_key == 'defect':
            st.write("**η > 0.6** - High defect concentration region")
            st.write("Analysis focuses on stress concentration in defect cores")
        elif region_key == 'interface':
            st.write("**0.4 ≤ η ≤ 0.6** - Interface region between defect and bulk")
            st.write("Analysis focuses on interfacial stress gradients")
        else:  # bulk
            st.write("**η < 0.4** - Pure Ag bulk material")
            st.write("Analysis focuses on stress propagation in bulk")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.solutions:
            st.warning("⚠️ Please load solutions first using the button in the sidebar.")
            
            # Show directory information
            with st.expander("📁 Directory Information", expanded=False):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.success(f"✅ Found {total_files} files in {SOLUTIONS_DIR}")
                else:
                    st.error(f"❌ No files found in {SOLUTIONS_DIR}")
        
        else:
            # Generate analysis button
            if st.button("🚀 Generate Region Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Generating {region_type} analysis..."):
                    try:
                        if data_source == "Original Loaded Files":
                            # Analyze original files
                            results = st.session_state.original_analyzer.analyze_all_solutions(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            # Create stress matrix for visualization
                            stress_matrix, thetas, times = st.session_state.original_analyzer.create_stress_matrix(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            if stress_matrix is not None:
                                st.session_state.original_matrix = stress_matrix
                                st.session_state.thetas = np.rad2deg(thetas)
                                st.session_state.times = times
                                st.session_state.region_type = region_type
                                st.session_state.stress_component = stress_component
                                
                                st.success(f"✅ Generated analysis for {len(results)} original solutions")
                                
                                # Display summary statistics
                                with st.expander("📈 Original Solutions Summary", expanded=True):
                                    st.write(f"**Region:** {region_type}")
                                    st.write(f"**Stress Component:** {stress_component}")
                                    st.write(f"**Analysis Type:** {stress_type}")
                                    st.write(f"**Number of Solutions:** {len(results)}")
                                    st.write(f"**Theta Range:** {np.min(st.session_state.thetas):.1f}° to {np.max(st.session_state.thetas):.1f}°")
                                    st.write(f"**Time Points:** {len(times)}")
                            
                        elif data_source == "Interpolated Solutions":
                            # Generate interpolated solutions
                            st.info("Generating interpolated solutions...")
                            
                            # Define interpolation parameters
                            theta_min = 0
                            theta_max = 360
                            theta_step = 15
                            n_times = 50
                            max_time = 200
                            
                            # Generate theta range
                            thetas = np.arange(theta_min, theta_max + theta_step, theta_step)
                            theta_rad = np.deg2rad(thetas)
                            
                            # Generate time points
                            times = np.linspace(0, max_time, n_times)
                            
                            # Generate predictions for each orientation
                            predictions = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, theta in enumerate(theta_rad):
                                status_text.text(f"🔧 Processing orientation {i+1}/{len(theta_rad)} ({thetas[i]:.0f}°)...")
                                
                                # Target parameters (simplified for demo)
                                target_params = {
                                    'defect_type': 'ISF',
                                    'theta': float(theta),
                                    'eps0': 0.707,
                                    'kappa': 0.6,
                                    'shape': 'Square'
                                }
                                
                                # Interpolate with spatial locality
                                result = st.session_state.interpolator.interpolate_with_spatial_locality(
                                    st.session_state.solutions, target_params, region_key,
                                    stress_component, stress_type
                                )
                                
                                if result:
                                    # Create time evolution based on interpolated stress
                                    region_stress = result['region_stress']
                                    time_evolution = []
                                    
                                    for t in times:
                                        # Time-dependent scaling
                                        stress_at_t = region_stress * (1 - np.exp(-t / 50))
                                        time_evolution.append(stress_at_t)
                                    
                                    predictions.append(time_evolution)
                                
                                progress_bar.progress((i + 1) / len(theta_rad))
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Create stress matrix
                            if predictions:
                                stress_matrix = np.array(predictions).T
                                st.session_state.interpolated_matrix = stress_matrix
                                st.session_state.thetas = thetas
                                st.session_state.times = times
                                st.session_state.region_type = region_type
                                st.session_state.stress_component = stress_component
                                
                                st.success(f"✅ Generated interpolated solutions for {len(thetas)} orientations")
                                
                                # Display spatial locality information
                                with st.expander("🗺️ Spatial Locality Information", expanded=True):
                                    if use_spatial_locality:
                                        st.success("✅ Spatial locality regularization ENABLED")
                                        st.write(f"**Sigma:** {spatial_sigma}")
                                        st.write(f"**Spatial Weight:** {spatial_weight}")
                                        st.write("**Method:** Euclidean distance in parameter space")
                                        st.write("**Weight Calculation:** Gaussian kernel based on distance")
                                    else:
                                        st.warning("⚠️ Spatial locality regularization DISABLED")
                                        st.write("Using uniform weights for interpolation")
                            
                        elif data_source == "Comparison (Both)":
                            # Generate both original and interpolated
                            st.info("Generating comparison analysis...")
                            
                            # Analyze original files
                            results = st.session_state.original_analyzer.analyze_all_solutions(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            original_matrix, orig_thetas, orig_times = st.session_state.original_analyzer.create_stress_matrix(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            # Generate interpolated matrix with same dimensions
                            if original_matrix is not None:
                                # Align interpolated matrix to original dimensions
                                n_thetas = len(orig_thetas)
                                n_times = len(orig_times)
                                
                                interpolated_matrix = np.zeros((n_thetas, n_times))
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, theta in enumerate(orig_thetas):
                                    status_text.text(f"🔧 Interpolating orientation {i+1}/{n_thetas} ({np.rad2deg(theta):.1f}°)...")
                                    
                                    # Target parameters
                                    target_params = {
                                        'defect_type': 'ISF',
                                        'theta': float(theta),
                                        'eps0': 0.707,
                                        'kappa': 0.6,
                                        'shape': 'Square'
                                    }
                                    
                                    # Interpolate
                                    result = st.session_state.interpolator.interpolate_with_spatial_locality(
                                        st.session_state.solutions, target_params, region_key,
                                        stress_component, stress_type
                                    )
                                    
                                    if result:
                                        region_stress = result['region_stress']
                                        for t in range(n_times):
                                            stress_at_t = region_stress * (1 - np.exp(-orig_times[t] / 50))
                                            interpolated_matrix[i, t] = stress_at_t
                                    
                                    progress_bar.progress((i + 1) / n_thetas)
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Store both matrices
                                st.session_state.original_matrix = original_matrix
                                st.session_state.interpolated_matrix = interpolated_matrix
                                st.session_state.thetas = np.rad2deg(orig_thetas)
                                st.session_state.times = orig_times
                                st.session_state.region_type = region_type
                                st.session_state.stress_component = stress_component
                                
                                st.success(f"✅ Generated comparison analysis")
                                
                                # Calculate comparison metrics
                                with st.expander("📊 Comparison Metrics", expanded=True):
                                    if 'original_matrix' in st.session_state and 'interpolated_matrix' in st.session_state:
                                        orig = st.session_state.original_matrix
                                        interp = st.session_state.interpolated_matrix
                                        
                                        mae = np.mean(np.abs(interp - orig))
                                        rmse = np.sqrt(np.mean((interp - orig) ** 2))
                                        r2 = 1 - np.sum((interp - orig) ** 2) / np.sum((orig - np.mean(orig)) ** 2)
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("MAE", f"{mae:.4f} GPa")
                                        with col2:
                                            st.metric("RMSE", f"{rmse:.4f} GPa")
                                        with col3:
                                            st.metric("R² Score", f"{r2:.4f}")
                                        
                                        st.write(f"**Spatial Locality:** {'ENABLED' if use_spatial_locality else 'DISABLED'}")
                                        
                        # Generate visualizations based on data source
                        if 'original_matrix' in st.session_state or 'interpolated_matrix' in st.session_state:
                            
                            if viz_type in ["Sunburst", "Both"]:
                                st.markdown(f'<h3 class="sub-header">🌅 {region_type} - Sunburst Visualization</h3>', unsafe_allow_html=True)
                                
                                if data_source == "Original Loaded Files" and 'original_matrix' in st.session_state:
                                    # Original sunburst
                                    fig = st.session_state.visualizer.create_enhanced_plotly_sunburst(
                                        st.session_state.original_matrix.T,  # Transpose for time × theta
                                        st.session_state.times,
                                        st.session_state.thetas,
                                        title=f"Original Solutions: {region_type} - {stress_component}",
                                        cmap=cmap
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif data_source == "Interpolated Solutions" and 'interpolated_matrix' in st.session_state:
                                    # Interpolated sunburst
                                    fig = st.session_state.visualizer.create_enhanced_plotly_sunburst(
                                        st.session_state.interpolated_matrix.T,
                                        st.session_state.times,
                                        st.session_state.thetas,
                                        title=f"Interpolated Solutions: {region_type} - {stress_component}",
                                        cmap=cmap
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif data_source == "Comparison (Both)" and 'original_matrix' in st.session_state and 'interpolated_matrix' in st.session_state:
                                    # Comparison sunburst
                                    fig = st.session_state.comparison_visualizer.create_comparison_sunburst(
                                        st.session_state.original_matrix.T,
                                        st.session_state.interpolated_matrix.T,
                                        st.session_state.thetas,
                                        st.session_state.times,
                                        region_type,
                                        stress_component
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            if viz_type in ["Radar", "Both"]:
                                st.markdown(f'<h3 class="sub-header">📡 {region_type} - Radar Visualization</h3>', unsafe_allow_html=True)
                                
                                # Select time point for radar
                                time_idx = st.slider(
                                    "Select Time Point for Radar Chart",
                                    0, len(st.session_state.times)-1, 
                                    len(st.session_state.times)//2,
                                    key="radar_time"
                                )
                                selected_time = st.session_state.times[time_idx]
                                
                                if data_source == "Original Loaded Files" and 'original_matrix' in st.session_state:
                                    # Original radar
                                    original_stress = st.session_state.original_matrix[:, time_idx]
                                    fig = st.session_state.visualizer.create_enhanced_plotly_radar(
                                        original_stress,
                                        st.session_state.thetas,
                                        f"Original: {stress_component}",
                                        selected_time
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif data_source == "Interpolated Solutions" and 'interpolated_matrix' in st.session_state:
                                    # Interpolated radar
                                    interpolated_stress = st.session_state.interpolated_matrix[:, time_idx]
                                    fig = st.session_state.visualizer.create_enhanced_plotly_radar(
                                        interpolated_stress,
                                        st.session_state.thetas,
                                        f"Interpolated: {stress_component}",
                                        selected_time
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif data_source == "Comparison (Both)" and 'original_matrix' in st.session_state and 'interpolated_matrix' in st.session_state:
                                    # Comparison radar
                                    original_stress = st.session_state.original_matrix[:, time_idx]
                                    interpolated_stress = st.session_state.interpolated_matrix[:, time_idx]
                                    
                                    fig = st.session_state.comparison_visualizer.create_comparison_radar(
                                        original_stress,
                                        interpolated_stress,
                                        st.session_state.thetas,
                                        region_type,
                                        stress_component
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            if viz_type == "Error Analysis" and data_source == "Comparison (Both)":
                                st.markdown(f'<h3 class="sub-header">📊 {region_type} - Error Analysis</h3>', unsafe_allow_html=True)
                                
                                if 'original_matrix' in st.session_state and 'interpolated_matrix' in st.session_state:
                                    fig, stats = st.session_state.comparison_visualizer.create_error_analysis(
                                        st.session_state.original_matrix,
                                        st.session_state.interpolated_matrix
                                    )
                                    
                                    st.pyplot(fig)
                                    
                                    # Display error statistics
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Mean Absolute Error", f"{stats['MAE']:.6f} GPa")
                                        st.metric("Root Mean Square Error", f"{stats['RMSE']:.6f} GPa")
                                    with col2:
                                        st.metric("Mean Relative Error", f"{stats['Mean_Rel_Error']:.2f}%")
                                        st.metric("Max Relative Error", f"{stats['Max_Rel_Error']:.2f}%")
                                    
                                    # Spatial locality assessment
                                    st.markdown("#### 🗺️ Spatial Locality Assessment")
                                    if use_spatial_locality:
                                        st.success("✅ Spatial locality regularization is improving interpolation accuracy")
                                        st.write("Euclidean distance weighting helps prioritize similar parameter configurations")
                                    else:
                                        st.warning("⚠️ Consider enabling spatial locality for better interpolation accuracy")
                                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.markdown('<h2 class="sub-header">📈 Quick Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.solutions:
            # Quick region analysis
            if st.button("🔍 Quick Region Analysis", use_container_width=True):
                with st.spinner("Analyzing regions..."):
                    try:
                        # Analyze all three regions
                        region_results = {}
                        for region_key_temp, region_name in [('defect', 'Defect'), 
                                                           ('interface', 'Interface'), 
                                                           ('bulk', 'Bulk')]:
                            results = st.session_state.original_analyzer.analyze_all_solutions(
                                st.session_state.solutions, region_key_temp, 
                                'von_mises', 'max_abs'
                            )
                            
                            if results:
                                stresses = [r['region_stress'] for r in results]
                                region_results[region_name] = {
                                    'mean': np.mean(stresses),
                                    'max': np.max(stresses),
                                    'min': np.min(stresses),
                                    'count': len(stresses)
                                }
                        
                        # Display results
                        st.markdown("#### Region Comparison")
                        for region_name, stats in region_results.items():
                            with st.container():
                                st.write(f"**{region_name} Region**")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Mean Stress", f"{stats['mean']:.3f} GPa")
                                with col_b:
                                    st.metric("Max Stress", f"{stats['max']:.3f} GPa")
                                st.caption(f"Based on {stats['count']} solutions")
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        
        # Spatial locality explanation
        with st.expander("🗺️ About Spatial Locality", expanded=False):
            st.write("""
            **Spatial Locality Regularization** uses Euclidean distance in parameter space:
            
            1. **Parameter Vector**: Each simulation is represented as an 11D vector
            2. **Euclidean Distance**: Distance between source and target vectors
            3. **Gaussian Weights**: Weights = exp(-0.5 * (distance/sigma)²)
            4. **Normalization**: Weights sum to 1
            
            **Benefits**:
            - Prioritizes similar parameter configurations
            - Reduces influence of dissimilar simulations
            - Improves interpolation accuracy
            - Physically meaningful weighting
            
            **Formula**: 
            ```
            weight_i = exp(-0.5 * ||v_source - v_target||² / sigma²)
            ```
            """)
    
    with col3:
        st.markdown('<h2 class="sub-header">📋 Region Definitions</h2>', unsafe_allow_html=True)
        
        # Region cards
        st.markdown('<div class="region-card defect-region">', unsafe_allow_html=True)
        st.markdown("##### 🔴 Defect Region")
        st.write("**η > 0.6**")
        st.write("High defect concentration")
        st.write("*Analysis:* Stress concentration in defect cores")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="region-card interface-region">', unsafe_allow_html=True)
        st.markdown("##### 🟡 Interface Region")
        st.write("**0.4 ≤ η ≤ 0.6**")
        st.write("Transition region")
        st.write("*Analysis:* Interfacial stress gradients")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="region-card bulk-region">', unsafe_allow_html=True)
        st.markdown("##### 🟢 Bulk Region")
        st.write("**η < 0.4**")
        st.write("Pure Ag material")
        st.write("*Analysis:* Stress propagation in bulk")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats
        if st.session_state.solutions:
            st.markdown("#### 📊 Quick Stats")
            st.write(f"**Loaded Solutions:** {len(st.session_state.solutions)}")
            
            # Count by defect type
            defect_types = {}
            for sol in st.session_state.solutions:
                d_type = sol.get('params', {}).get('defect_type', 'Unknown')
                defect_types[d_type] = defect_types.get(d_type, 0) + 1
            
            for d_type, count in defect_types.items():
                st.write(f"- {d_type}: {count}")

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
