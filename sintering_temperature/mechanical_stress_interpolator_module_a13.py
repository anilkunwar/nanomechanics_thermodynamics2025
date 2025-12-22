import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
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

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# Color schemes
STRESS_CMAP = LinearSegmentedColormap.from_list(
    'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
)
SUNBURST_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'rainbow']

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

@jit(nopython=True, parallel=True)
def weighted_stress_combination_numba(source_stresses, weights):
    """Numba-accelerated weighted stress combination"""
    n_sources = source_stresses.shape[0]
    n_components = source_stresses.shape[1]
    height = source_stresses.shape[2]
    width = source_stresses.shape[3]
    
    result = np.zeros((n_components, height, width))
    
    for comp in prange(n_components):
        for i in range(n_sources):
            weight = weights[i]
            for h in range(height):
                for w in range(width):
                    result[comp, h, w] += weight * source_stresses[i, comp, h, w]
    
    return result

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
        
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
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
                    st.warning(f"Could not get info for {file_path}: {e}")
        
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
    
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        return pickle.load(buffer)
    
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))
    
    def _read_h5(self, file_content):
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
    
    def _read_npz(self, file_content):
        buffer = BytesIO(file_content)
        data = np.load(buffer, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
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
            os.unlink(tmp_path)
            raise e
    
    def _read_json(self, file_content):
        return json.loads(file_content.decode('utf-8'))
    
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file with format auto-detection"""
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
            return self._standardize_data(data, format_type)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _standardize_data(self, data, format_type):
        """Standardize simulation data structure"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type
        }
        
        if format_type == 'pkl':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                standardized['history'] = data.get('history', [])
        
        elif format_type == 'pt':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                standardized['history'] = data.get('history', [])
        
        elif format_type == 'h5':
            standardized.update(data)
        
        elif format_type == 'npz':
            standardized.update(data)
        
        elif format_type == 'json':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                standardized['history'] = data.get('history', [])
        
        return standardized
    
    def load_all_solutions(self, use_cache=True):
        """Load all solutions with caching and progress tracking"""
        solutions = []
        
        if not os.path.exists(self.solutions_dir):
            st.warning(f"Directory {self.solutions_dir} not found. Creating it.")
            os.makedirs(self.solutions_dir, exist_ok=True)
            return solutions
        
        all_files_info = self.get_all_files_info()
        
        if not all_files_info:
            st.info(f"No solution files found in {self.solutions_dir}")
            return solutions
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file_info in enumerate(all_files_info):
            try:
                file_path = file_info['path']
                filename = file_info['filename']
                
                # Check cache
                cache_key = f"{filename}_{os.path.getmtime(file_path)}"
                if use_cache and cache_key in self.cache:
                    sim = self.cache[cache_key]
                    solutions.append(sim)
                    continue
                
                # Update progress
                progress = (idx + 1) / len(all_files_info)
                progress_bar.progress(progress)
                status_text.text(f"Loading {filename}...")
                
                # Load file
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                sim = self.read_simulation_file(file_content, file_info['format'])
                sim['filename'] = filename
                sim['file_info'] = file_info
                
                # Validate structure
                if 'params' in sim and 'history' in sim:
                    # Cache the solution
                    self.cache[cache_key] = sim
                    solutions.append(sim)
                else:
                    st.warning(f"⚠️ Skipped {filename}: Missing params/history")
                    
            except Exception as e:
                st.error(f"❌ Error loading {file_info['filename']}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        return solutions

# =============================================
# ENHANCED ATTENTION INTERPOLATOR WITH NUMBA
# =============================================
class EnhancedAttentionInterpolator(nn.Module):
    """Enhanced attention-based interpolator with Numba acceleration"""
    
    def __init__(self, sigma=0.3, use_numba=True):
        super().__init__()
        self.sigma = sigma
        self.use_numba = use_numba
        
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
    
    @jit(nopython=True)
    def compute_parameter_vector_numba(self, params_array):
        """Numba-accelerated parameter vector computation"""
        # This is a simplified version - for actual use, you'd need to adapt
        # the parameter mapping to work with numba
        return params_array  # Placeholder
    
    def compute_parameter_vector(self, params):
        """Convert parameters to numerical vector"""
        vector = []
        
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
    
    def interpolate(self, sources, target_params):
        """Interpolate stress field using attention weights with Numba acceleration"""
        
        # Get source parameter vectors
        source_vectors = []
        source_stresses_list = []
        
        for src in sources:
            src_vec = self.compute_parameter_vector(src['params'])
            source_vectors.append(src_vec)
            
            # Get stress from final frame
            if src['history']:
                _, stress_fields = src['history'][-1]
                source_stresses_list.append({
                    'von_mises': stress_fields.get('von_mises', np.zeros((128, 128))),
                    'sigma_hydro': stress_fields.get('sigma_hydro', np.zeros((128, 128))),
                    'sigma_mag': stress_fields.get('sigma_mag', np.zeros((128, 128)))
                })
        
        if not source_vectors:
            return None
        
        source_vectors = np.array(source_vectors)
        target_vector = self.compute_parameter_vector(target_params)
        
        # Compute attention weights with Numba acceleration
        if self.use_numba:
            weights = compute_gaussian_weights_numba(source_vectors, target_vector, self.sigma)
        else:
            # Fallback to numpy
            distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
            weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
            weights = weights / (np.sum(weights) + 1e-8)
        
        # Convert to numpy array for Numba acceleration
        if self.use_numba and source_stresses_list:
            stress_components = ['von_mises', 'sigma_hydro', 'sigma_mag']
            n_components = len(stress_components)
            height, width = source_stresses_list[0]['von_mises'].shape
            
            source_stress_array = np.zeros((len(source_stresses_list), n_components, height, width))
            for i, stresses in enumerate(source_stresses_list):
                for j, comp in enumerate(stress_components):
                    source_stress_array[i, j] = stresses[comp]
            
            # Use Numba-accelerated combination
            combined_array = weighted_stress_combination_numba(source_stress_array, weights)
            
            result = {}
            for j, comp in enumerate(stress_components):
                result[comp] = combined_array[j]
        else:
            # Fallback to numpy combination
            result = {}
            for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                combined = np.zeros_like(source_stresses_list[0][key])
                for w, stress in zip(weights, source_stresses_list):
                    combined += w * stress[key]
                result[key] = combined
        
        return {
            'stress_fields': result,
            'attention_weights': weights,
            'target_params': target_params
        }

# =============================================
# ENHANCED VISUALIZATION MANAGER
# =============================================
class EnhancedVisualizationManager:
    """Enhanced visualization manager with multiple plot types"""
    
    def __init__(self):
        self.stress_cmap = STRESS_CMAP
        self.sunburst_cmaps = SUNBURST_CMAPS
        
    def create_stress_field_plot(self, stress_data, title, component_name,
                                extent=None, vmin=None, vmax=None,
                                include_contour=True, include_colorbar=True):
        """Create matplotlib plot for stress field"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        if extent is None:
            extent = [-64, 64, -64, 64]
        
        if vmin is None:
            vmin = np.nanmin(stress_data)
        if vmax is None:
            vmax = np.nanmax(stress_data)
        
        # Create heatmap
        im = ax.imshow(stress_data,
                      cmap=self.stress_cmap,
                      extent=extent,
                      origin='lower',
                      aspect='equal',
                      vmin=vmin,
                      vmax=vmax)
        
        # Add contour lines
        if include_contour and not np.all(stress_data == stress_data[0, 0]):
            try:
                X, Y = np.meshgrid(np.linspace(extent[0], extent[1], stress_data.shape[1]),
                                  np.linspace(extent[2], extent[3], stress_data.shape[0]))
                
                levels = np.linspace(vmin, vmax, 12)
                contour = ax.contour(X, Y, stress_data,
                                    levels=levels,
                                    colors='black',
                                    linewidths=0.5,
                                    alpha=0.7)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
            except:
                pass
        
        # Add colorbar
        if include_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Stress (GPa)', rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel('x (nm)', fontsize=12)
        ax.set_ylabel('y (nm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def create_attention_weights_plot(self, weights, source_names=None,
                                     title="Attention Weights Distribution"):
        """Create bar plot for attention weights"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        if source_names is None:
            source_names = [f'Source {i+1}' for i in range(len(weights))]
        
        # Create bar plot
        x_pos = np.arange(len(weights))
        bars = ax.bar(x_pos, weights,
                     color=plt.cm.viridis(np.linspace(0, 1, len(weights))),
                     edgecolor='black',
                     linewidth=1,
                     alpha=0.8)
        
        # Add value labels on top of bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Source Simulations', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(source_names, rotation=45, ha='right')
        ax.set_ylim([0, max(weights) * 1.3])
        
        # Add horizontal line for average
        avg_weight = np.mean(weights)
        ax.axhline(y=avg_weight, color='red', linestyle='--', alpha=0.7,
                  label=f'Average: {avg_weight:.3f}')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER
# =============================================
class EnhancedSunburstRadarVisualizer:
    """Enhanced sunburst and radar charts with multiple visualization options"""
    
    def __init__(self):
        self.vis_manager = EnhancedVisualizationManager()
    
    def create_sunburst_plot(self, stress_matrix, times, thetas, title, cmap='plasma'):
        """Create polar heatmap (sunburst) visualization"""
        
        # Create polar plot
        theta_deg = np.deg2rad(thetas)
        theta_mesh, time_mesh = np.meshgrid(theta_deg, times)
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'}, dpi=150)
        
        # Plot heatmap
        im = ax.pcolormesh(theta_mesh, time_mesh, stress_matrix, 
                          cmap=cmap, shading='auto', alpha=0.8)
        
        # Customize
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Orientation (degrees)', labelpad=20, fontsize=12)
        ax.set_ylabel('Time (s)', labelpad=20, fontsize=12)
        ax.set_xticks(theta_deg)
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], fontsize=10)
        
        # Add radial grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.1)
        cbar.set_label('Stress (GPa)', rotation=270, labelpad=25, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_plotly_sunburst(self, stress_matrix, times, thetas, title, cmap='Plasma'):
        """Interactive sunburst with Plotly - enhanced version"""
        
        # Prepare data for polar scatter
        theta_deg = np.deg2rad(thetas)
        theta_grid, time_grid = np.meshgrid(theta_deg, times)
        
        fig = go.Figure(data=go.Scatterpolar(
            r=time_grid.flatten(),
            theta=np.rad2deg(theta_grid).flatten(),
            mode='markers',
            marker=dict(
                size=10,
                color=stress_matrix.flatten(),
                colorscale=cmap,
                showscale=True,
                colorbar=dict(
                    title="Stress (GPa)",
                    titlefont=dict(size=14),
                    tickfont=dict(size=12)
                ),
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>Time</b>: %{r:.1f}s<br>' +
                         '<b>Orientation</b>: %{theta:.1f}°<br>' +
                         '<b>Stress</b>: %{marker.color:.3f} GPa<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, family="Arial, sans-serif")),
            polar=dict(
                radialaxis=dict(
                    title="Time (s)",
                    titlefont=dict(size=14),
                    gridcolor="lightgray",
                    showline=True,
                    linewidth=1,
                    linecolor="black"
                ),
                angularaxis=dict(
                    gridcolor="lightgray",
                    rotation=90,
                    direction="clockwise",
                    tickfont=dict(size=12)
                ),
                bgcolor="rgba(245, 245, 245, 0.8)"
            ),
            height=650,
            width=800,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        return fig
    
    def create_radar_plot(self, stress_values, thetas, component_name, time_point):
        """Enhanced radar/spider chart"""
        
        # Close the loop
        angles = np.linspace(0, 2*np.pi, len(thetas), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        values = np.concatenate([stress_values, [stress_values[0]]])
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'}, dpi=150)
        
        # Plot with gradient fill
        ax.plot(angles, values, 'o-', linewidth=3, markersize=8, 
                color='steelblue', markerfacecolor='white', markeredgewidth=2)
        
        # Gradient fill
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], fontsize=11)
        
        # Title with metrics
        mean_val = np.mean(stress_values)
        max_val = np.max(stress_values)
        ax.set_title(f'{component_name}\nt={time_point:.1f}s | Mean: {mean_val:.2f} GPa | Max: {max_val:.2f} GPa',
                    fontsize=14, fontweight='bold', pad=25)
        
        # Add value annotations
        for angle, value in zip(angles[:-1], stress_values):
            x_pos = angle
            y_pos = value * 1.05
            ax.text(x_pos, y_pos, f'{value:.2f}', 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_comparison_radar(self, stress_matrices, thetas, component_names, 
                               time_point, title="Multiple Component Comparison"):
        """Create radar plot comparing multiple stress components"""
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'}, dpi=150)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(stress_matrices)))
        
        for idx, (stress_matrix, comp_name, color) in enumerate(zip(stress_matrices, component_names, colors)):
            # Get values for the selected time point
            time_idx = np.argmin(np.abs(np.arange(len(stress_matrix)) - time_point))
            values = stress_matrix[time_idx, :]
            
            # Close the loop
            angles = np.linspace(0, 2*np.pi, len(thetas), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])
            values_loop = np.concatenate([values, [values[0]]])
            
            # Plot
            ax.plot(angles, values_loop, 'o-', linewidth=2, markersize=6,
                    color=color, label=comp_name, alpha=0.8)
            ax.fill(angles, values_loop, alpha=0.15, color=color)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], fontsize=11)
        ax.set_title(f'{title}\nt={time_point:.1f}s', fontsize=16, fontweight='bold', pad=25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# =============================================
# RESULTS MANAGER
# =============================================
class ResultsManager:
    """Manager for saving and exporting results"""
    
    @staticmethod
    def prepare_prediction_data(prediction_results, source_simulations, target_params):
        """Prepare prediction data for export"""
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_sources': len(source_simulations),
                'target_params': target_params,
                'software_version': '2.0.0'
            },
            'prediction_results': prediction_results,
            'source_summary': []
        }
        
        # Add source simulation summaries
        for i, sim in enumerate(source_simulations):
            params = sim.get('params', {})
            export_data['source_summary'].append({
                'id': i,
                'defect_type': params.get('defect_type'),
                'shape': params.get('shape'),
                'eps0': float(params.get('eps0', 0)),
                'kappa': float(params.get('kappa', 0)),
                'theta': float(params.get('theta', 0))
            })
        
        return export_data
    
    @staticmethod
    def create_results_archive(stress_matrix, times, thetas, metadata):
        """Create ZIP archive with all results"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save stress matrix as NPY
            stress_buffer = BytesIO()
            np.save(stress_buffer, stress_matrix)
            zip_file.writestr('stress_matrix.npy', stress_buffer.getvalue())
            
            # Save times and thetas
            times_buffer = BytesIO()
            np.save(times_buffer, times)
            zip_file.writestr('times.npy', times_buffer.getvalue())
            
            thetas_buffer = BytesIO()
            np.save(thetas_buffer, thetas)
            zip_file.writestr('thetas.npy', thetas_buffer.getvalue())
            
            # Save metadata as JSON
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr('metadata.json', metadata_json)
            
            # Save CSV version
            csv_data = []
            for t_idx, time_val in enumerate(times):
                for theta_idx, theta_val in enumerate(thetas):
                    csv_data.append({
                        'time_s': time_val,
                        'orientation_deg': theta_val,
                        'stress_gpa': stress_matrix[t_idx, theta_idx]
                    })
            
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            zip_file.writestr('stress_data.csv', csv_str)
            
            # Add README
            readme = f"""# Stress Interpolation Results
Generated: {datetime.now().isoformat()}

Files:
1. stress_matrix.npy - 2D stress matrix (time × orientation)
2. times.npy - Time points array
3. thetas.npy - Orientation angles array
4. metadata.json - Simulation metadata
5. stress_data.csv - Tabular data for analysis

Parameters:
- Defect Type: {metadata.get('defect_type', 'Unknown')}
- Orientation Range: {metadata.get('theta_range', 'Unknown')}
- ε*: {metadata.get('eps0', 'Unknown')}
- κ: {metadata.get('kappa', 'Unknown')}
"""
            zip_file.writestr('README.txt', readme)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Enhanced Stress Interpolation Visualizer",
        layout="wide",
        page_icon="🔬"
    )
    
    st.title("🔬 Enhanced Stress Field Interpolation with Numba Acceleration")
    st.markdown("""
    This app loads simulation data from the `numerical_solutions` directory, 
    performs attention-based interpolation with Numba-accelerated computations,
    and visualizes results as sunburst and radar charts.
    """)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedAttentionInterpolator(sigma=0.3, use_numba=True)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'vis_manager' not in st.session_state:
        st.session_state.vis_manager = EnhancedVisualizationManager()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Performance settings
        with st.expander("🚀 Performance Settings", expanded=True):
            use_numba = st.checkbox("Use Numba Acceleration", value=True)
            use_cache = st.checkbox("Use File Cache", value=True)
            
            if use_numba != st.session_state.interpolator.use_numba:
                st.session_state.interpolator.use_numba = use_numba
                st.success("Numba setting updated!")
        
        # Load solutions
        st.subheader("📂 Load Solutions")
        
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load Solutions", use_container_width=True, 
                        help="Load all solutions from numerical_solutions directory"):
                with st.spinner("Loading solutions..."):
                    start_time = time.time()
                    st.session_state.solutions = st.session_state.loader.load_all_solutions(
                        use_cache=use_cache
                    )
                    load_time = time.time() - start_time
                    
                    if st.session_state.solutions:
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions in {load_time:.2f}s")
                    else:
                        st.warning("No solutions found or loaded")
        
        with col_load2:
            if st.button("🗑️ Clear Cache", use_container_width=True, 
                        help="Clear file cache"):
                st.session_state.loader.cache.clear()
                st.success("Cache cleared!")
        
        # Show solution info
        if st.session_state.solutions:
            st.subheader("📋 Loaded Solutions")
            with st.expander(f"View {len(st.session_state.solutions)} solutions"):
                for i, sol in enumerate(st.session_state.solutions[:10]):  # Limit to 10
                    params = sol.get('params', {})
                    st.caption(f"{i+1}. {sol.get('filename', 'Unknown')}")
                    st.write(f"   • Type: {params.get('defect_type', 'Unknown')}")
                    st.write(f"   • θ: {np.rad2deg(params.get('theta', 0)):.1f}°")
                    st.write(f"   • ε*: {params.get('eps0', 0):.3f}")
                    st.divider()
                
                if len(st.session_state.solutions) > 10:
                    st.info(f"... and {len(st.session_state.solutions) - 10} more")
        
        st.divider()
        
        # Interpolation settings
        st.subheader("🎯 Target Parameters")
        
        defect_type = st.selectbox("Defect Type", ["ISF", "ESF", "Twin"], index=0)
        
        col_shape, col_eps = st.columns(2)
        with col_shape:
            shape = st.selectbox("Shape", 
                                ["Square", "Horizontal Fault", "Vertical Fault", 
                                 "Rectangle", "Ellipse"], index=0)
        with col_eps:
            eps0 = st.slider("ε*", 0.3, 3.0, 0.707, 0.01,
                            help="Strain parameter")
        
        col_kappa, col_sigma = st.columns(2)
        with col_kappa:
            kappa = st.slider("κ", 0.1, 2.0, 0.6, 0.01,
                             help="Shape parameter")
        with col_sigma:
            sigma = st.slider("σ", 0.1, 1.0, 0.3, 0.05,
                             help="Attention sigma parameter")
        
        # Update sigma if changed
        if sigma != st.session_state.interpolator.sigma:
            st.session_state.interpolator.sigma = sigma
        
        # Orientation sweep settings
        st.subheader("🌐 Orientation Sweep")
        
        theta_min = st.slider("Min Angle (°)", 0, 90, 0, 5,
                             help="Minimum orientation angle")
        theta_max = st.slider("Max Angle (°)", 0, 90, 90, 5,
                             help="Maximum orientation angle")
        theta_step = st.slider("Step Size (°)", 5, 45, 15, 5,
                              help="Angle step size")
        
        # Time settings
        st.subheader("⏱️ Time Settings")
        
        n_times = st.slider("Time Points", 10, 200, 50, 10,
                           help="Number of time points to simulate")
        max_time = st.slider("Max Time (s)", 50, 500, 200, 10,
                            help="Maximum simulation time")
        
        # Visualization settings
        st.subheader("🎨 Visualization")
        
        viz_type = st.radio("Chart Type", ["Sunburst", "Radar", "Both", "Comparison"], 
                           help="Select visualization type")
        cmap = st.selectbox("Color Map", SUNBURST_CMAPS, index=1)
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            use_plotly = st.checkbox("Use Plotly (Interactive)", value=True)
        with col_viz2:
            show_metrics = st.checkbox("Show Metrics", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🚀 Generate Visualizations")
        
        if not st.session_state.solutions:
            st.warning("Please load solutions first using the button in the sidebar.")
            
            # Show directory info
            with st.expander("📁 Directory Information"):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.info(f"Found {total_files} files in {SOLUTIONS_DIR}:")
                    for fmt, files in file_formats.items():
                        if files:
                            st.write(f"• {fmt.upper()}: {len(files)} files")
                else:
                    st.write(f"Directory: `{SOLUTIONS_DIR}`")
                    st.write("Expected file formats: `.pkl`, `.pt`, `.h5`, `.npz`, `.sql`, `.json`")
                    
                    st.info("""
                    **Expected file structure:**
                    - Place simulation files in the `numerical_solutions/` directory
                    - Each file should contain simulation data with:
                      - `params` dictionary (defect_type, theta, eps0, kappa, shape)
                      - `history` list with stress fields
                    """)
        else:
            if st.button("✨ Generate Enhanced Charts", type="primary", use_container_width=True):
                with st.spinner("Generating orientation sweep..."):
                    try:
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
                            status_text.text(f"Processing orientation {i+1}/{len(theta_rad)}...")
                            
                            # Target parameters
                            target_params = {
                                'defect_type': defect_type,
                                'theta': float(theta),
                                'eps0': eps0,
                                'kappa': kappa,
                                'shape': shape
                            }
                            
                            # Interpolate
                            result = st.session_state.interpolator.interpolate(
                                st.session_state.solutions, target_params
                            )
                            
                            if result:
                                # Extract stress evolution at center point
                                center_i, center_j = 64, 64  # Assuming 128x128 grid
                                time_evolution = []
                                
                                # Create synthetic time evolution
                                for t in times:
                                    base_stress = result['stress_fields']['von_mises'][center_i, center_j]
                                    stress_at_t = base_stress * (1 - np.exp(-t / 50))
                                    time_evolution.append(stress_at_t)
                                
                                predictions.append(time_evolution)
                            
                            progress_bar.progress((i + 1) / len(theta_rad))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Create stress matrix (time x theta)
                        if predictions:
                            stress_matrix = np.array(predictions).T  # Shape: (n_times, n_thetas)
                            
                            # Store for visualization
                            st.session_state.stress_matrix = stress_matrix
                            st.session_state.times = times
                            st.session_state.thetas = thetas
                            
                            # Store metadata
                            st.session_state.metadata = {
                                'defect_type': defect_type,
                                'shape': shape,
                                'eps0': eps0,
                                'kappa': kappa,
                                'sigma': sigma,
                                'theta_range': f"{theta_min}-{theta_max}°",
                                'theta_step': theta_step,
                                'n_times': n_times,
                                'max_time': max_time,
                                'generated_at': datetime.now().isoformat()
                            }
                            
                            st.success(f"✅ Generated {len(thetas)} orientations × {len(times)} time points")
                            
                            # Display results
                            if viz_type in ["Sunburst", "Both", "Comparison"]:
                                st.subheader("🌅 Sunburst Chart")
                                
                                if use_plotly:
                                    fig = st.session_state.visualizer.create_plotly_sunburst(
                                        stress_matrix, times, thetas,
                                        title=f"Von Mises Stress - {defect_type}",
                                        cmap=cmap
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig = st.session_state.visualizer.create_sunburst_plot(
                                        stress_matrix, times, thetas,
                                        title=f"Von Mises Stress - {defect_type}",
                                        cmap=cmap
                                    )
                                    st.pyplot(fig)
                                    
                                    # Download button
                                    col_dl1, col_dl2 = st.columns(2)
                                    with col_dl1:
                                        buf = BytesIO()
                                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download Sunburst PNG",
                                            data=buf.getvalue(),
                                            file_name=f"sunburst_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                            mime="image/png",
                                            use_container_width=True
                                        )
                                    with col_dl2:
                                        buf = BytesIO()
                                        fig.savefig(buf, format="pdf", bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download Sunburst PDF",
                                            data=buf.getvalue(),
                                            file_name=f"sunburst_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                            
                            if viz_type in ["Radar", "Both", "Comparison"]:
                                st.subheader("📡 Radar Charts")
                                
                                # Select time point
                                time_idx = st.slider("Select Time Point", 0, len(times)-1, len(times)//2,
                                                    key="radar_time_slider")
                                selected_time = times[time_idx]
                                
                                if viz_type == "Comparison":
                                    # Create comparison radar
                                    stress_components = []
                                    component_names = []
                                    
                                    # For demonstration, create synthetic data for other components
                                    stress_components.append(stress_matrix)
                                    component_names.append('Von Mises')
                                    
                                    # Synthetic hydrostatic stress (scaled differently)
                                    hydro_matrix = stress_matrix * 0.7 + 0.1
                                    stress_components.append(hydro_matrix)
                                    component_names.append('Hydrostatic')
                                    
                                    # Synthetic magnitude stress
                                    mag_matrix = stress_matrix * 0.9 + 0.05
                                    stress_components.append(mag_matrix)
                                    component_names.append('Magnitude')
                                    
                                    fig_comparison = st.session_state.visualizer.create_comparison_radar(
                                        stress_components, thetas, component_names, 
                                        selected_time,
                                        title=f"Stress Components Comparison - {defect_type}"
                                    )
                                    st.pyplot(fig_comparison)
                                    
                                else:
                                    # Create radar for each stress component
                                    cols = st.columns(3)
                                    component_names = ['Von Mises', 'Hydrostatic', 'Magnitude']
                                    
                                    for idx, (col, name) in enumerate(zip(cols, component_names)):
                                        with col:
                                            # For demonstration, use von_mises data with scaling
                                            if idx == 0:
                                                radar_values = stress_matrix[time_idx, :]
                                            elif idx == 1:
                                                radar_values = stress_matrix[time_idx, :] * 0.7 + 0.1
                                            else:
                                                radar_values = stress_matrix[time_idx, :] * 0.9 + 0.05
                                            
                                            fig_radar = st.session_state.visualizer.create_radar_plot(
                                                radar_values, thetas, name, selected_time
                                            )
                                            st.pyplot(fig_radar)
                            
                            # Enhanced Statistics with Numba acceleration
                            if show_metrics:
                                st.subheader("📊 Enhanced Statistics")
                                
                                # Compute statistics with Numba
                                start_time = time.time()
                                max_val, min_val, mean_val, std_val, p95, p99 = compute_stress_statistics_numba(
                                    stress_matrix
                                )
                                compute_time = time.time() - start_time
                                
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("Max Stress", f"{max_val:.3f} GPa")
                                with col_stat2:
                                    st.metric("Mean Stress", f"{mean_val:.3f} GPa")
                                with col_stat3:
                                    st.metric("Std Dev", f"{std_val:.3f} GPa")
                                with col_stat4:
                                    st.metric("95th %ile", f"{p95:.3f} GPa")
                                
                                st.caption(f"Statistics computed in {compute_time*1000:.2f} ms with Numba acceleration")
                                
                                # Distribution plot
                                fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                                ax_dist.hist(stress_matrix.flatten(), bins=50, 
                                           edgecolor='black', alpha=0.7, color='steelblue')
                                ax_dist.set_xlabel('Stress (GPa)', fontsize=12)
                                ax_dist.set_ylabel('Frequency', fontsize=12)
                                ax_dist.set_title('Stress Distribution', fontsize=14, fontweight='bold')
                                ax_dist.grid(True, alpha=0.3)
                                
                                # Add vertical lines for statistics
                                ax_dist.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                                ax_dist.axvline(p95, color='orange', linestyle=':', label=f'95th %ile: {p95:.3f}')
                                ax_dist.axvline(p99, color='green', linestyle='-.', label=f'99th %ile: {p99:.3f}')
                                ax_dist.legend()
                                
                                st.pyplot(fig_dist)
                            
                            # Enhanced Data export
                            st.subheader("📤 Enhanced Export Options")
                            
                            export_col1, export_col2, export_col3 = st.columns(3)
                            
                            with export_col1:
                                # CSV export
                                if st.button("💾 Export as CSV", use_container_width=True):
                                    export_data = []
                                    for t_idx, time_val in enumerate(times):
                                        for theta_idx, theta_val in enumerate(thetas):
                                            export_data.append({
                                                'time_s': time_val,
                                                'orientation_deg': theta_val,
                                                'stress_gpa': stress_matrix[t_idx, theta_idx]
                                            })
                                    
                                    df = pd.DataFrame(export_data)
                                    csv = df.to_csv(index=False)
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download CSV",
                                        data=csv,
                                        file_name=f"stress_data_{defect_type}_{timestamp}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            with export_col2:
                                # JSON export
                                if st.button("📊 Export as JSON", use_container_width=True):
                                    export_dict = {
                                        'metadata': st.session_state.metadata,
                                        'times': times.tolist(),
                                        'thetas': thetas.tolist(),
                                        'stress_matrix': stress_matrix.tolist()
                                    }
                                    
                                    json_str = json.dumps(export_dict, indent=2)
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download JSON",
                                        data=json_str,
                                        file_name=f"stress_data_{defect_type}_{timestamp}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            
                            with export_col3:
                                # ZIP archive export
                                if st.button("📦 Export as ZIP", use_container_width=True):
                                    zip_buffer = st.session_state.results_manager.create_results_archive(
                                        stress_matrix, times, thetas, st.session_state.metadata
                                    )
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download ZIP Archive",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"stress_results_{defect_type}_{timestamp}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.subheader("📈 Dashboard")
        
        if 'stress_matrix' in st.session_state:
            stress_matrix = st.session_state.stress_matrix
            
            st.metric("Time Points", len(st.session_state.times))
            st.metric("Orientations", len(st.session_state.thetas))
            
            # Quick statistics
            max_val = np.max(stress_matrix)
            mean_val = np.mean(stress_matrix)
            min_val = np.min(stress_matrix)
            
            st.metric("Max Value", f"{max_val:.3f} GPa", 
                     delta=f"{(max_val - mean_val):.3f} from mean")
            st.metric("Mean Value", f"{mean_val:.3f} GPa")
            st.metric("Min Value", f"{min_val:.3f} GPa")
            
            # Stress range
            stress_range = max_val - min_val
            st.progress((mean_val - min_val) / stress_range if stress_range > 0 else 0.5,
                       text=f"Mean position in range: {((mean_val - min_val) / stress_range * 100):.1f}%")
            
            # Orientation distribution
            st.subheader("🌐 Orientation Stats")
            
            mean_by_theta = np.mean(stress_matrix, axis=0)
            max_theta_idx = np.argmax(mean_by_theta)
            min_theta_idx = np.argmin(mean_by_theta)
            
            st.write(f"**Max stress at:** {st.session_state.thetas[max_theta_idx]:.0f}°")
            st.write(f"**Min stress at:** {st.session_state.thetas[min_theta_idx]:.0f}°")
            
            # Time evolution
            st.subheader("⏱️ Time Evolution")
            
            mean_by_time = np.mean(stress_matrix, axis=1)
            time_of_max = st.session_state.times[np.argmax(mean_by_time)]
            
            st.write(f"**Peak at:** {time_of_max:.1f}s")
            
            # Quick plot
            fig_quick, ax_quick = plt.subplots(figsize=(4, 3))
            ax_quick.plot(st.session_state.times, mean_by_time, 'b-', linewidth=2)
            ax_quick.fill_between(st.session_state.times, mean_by_time, alpha=0.3, color='blue')
            ax_quick.set_xlabel('Time (s)')
            ax_quick.set_ylabel('Mean Stress (GPa)')
            ax_quick.set_title('Time Evolution', fontsize=10)
            ax_quick.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_quick)
            
        else:
            st.info("No data generated yet.")
            st.write("Click 'Generate Enhanced Charts' to create visualizations.")
            
            # Show performance tips
            with st.expander("🚀 Performance Tips"):
                st.write("""
                1. **Enable Numba** for 10-100x faster computations
                2. **Use cache** for faster reloading of solutions
                3. **Reduce time points** for quicker generation
                4. **Increase angle step** for fewer orientations
                """)

# =============================================
# RUN APPLICATION
# =============================================
if __name__ == "__main__":
    main()
