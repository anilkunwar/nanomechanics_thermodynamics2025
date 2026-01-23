```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
from matplotlib.cm import get_cmap
import plotly.graph_objects as go
import plotly.express as px
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
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
import seaborn as sns
from scipy.ndimage import zoom, rotate
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# =============================================
# GLOBAL STYLING CONFIGURATION
# =============================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'image.cmap': 'viridis'
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

COLORMAP_OPTIONS = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'afmhot', 'gist_heat',
                   'copper', 'summer', 'Wistia', 'spring', 'autumn', 'winter', 'bone', 'gray', 'pink',
                   'gist_gray', 'gist_yarg', 'binary', 'gist_earth', 'terrain', 'ocean', 'gist_stern', 'gnuplot',
                   'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
                   'gist_ncar', 'hsv'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                  'RdGy', 'RdYlGn', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                    'Paired', 'Accent', 'Dark2'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
                             'turbo'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'RdBu_r', 'Spectral',
                             'coolwarm', 'bwr', 'seismic', 'BrBG']
}

# =============================================
# TARGETED SOLUTION LOADER WITH ANGLE BRACKETING
# =============================================
class TargetedSolutionLoader:
    """Solution loader with targeted angle bracketing capability"""
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.angle_tolerance = 5.0  # Degrees tolerance for finding bracketing sources
    
    def _ensure_directory(self):
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth']:
            import glob
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        file_info = []
        for file_path in all_files:
            try:
                info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'format': 'pkl' if file_path.endswith(('.pkl', '.pickle')) else 'pt'
                }
                file_info.append(info)
            except:
                continue
        return file_info
    
    def read_simulation_file(self, file_path, format_type='auto'):
        try:
            with open(file_path, 'rb') as f:
                if format_type == 'pt' or file_path.endswith(('.pt', '.pth')):
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                else:
                    data = pickle.load(f)
                
                standardized = self._standardize_data(data, file_path)
                return standardized
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data, file_path):
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False,
                'angle_degrees': None
            }
        }
        try:
            if isinstance(data, dict):
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                
                if 'history' in data:
                    history = data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        history_list = []
                        for key in sorted(history.keys()):
                            if isinstance(history[key], dict):
                                history_list.append(history[key])
                        standardized['history'] = history_list
                
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                
                self._convert_tensors(standardized)
                
                # Extract angle in degrees for easier sorting
                if 'theta' in standardized['params']:
                    theta_rad = standardized['params']['theta']
                    standardized['metadata']['angle_degrees'] = np.degrees(theta_rad)
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
        return standardized
    
    def _convert_tensors(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.cpu().numpy()
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    data[i] = item.cpu().numpy()
                elif isinstance(item, (dict, list)):
                    self._convert_tensors(item)
    
    def find_bracketing_sources(self, target_angle_deg, target_defect_type, solutions):
        """
        Find the two nearest sources that bracket the target angle for the given defect type.
        Returns: (lower_source, upper_source, lower_idx, upper_idx, lower_angle, upper_angle)
        """
        # Filter solutions by defect type
        same_defect_solutions = []
        for i, sol in enumerate(solutions):
            if 'params' in sol and sol['params'].get('defect_type') == target_defect_type:
                angle_deg = sol['metadata'].get('angle_degrees')
                if angle_deg is not None:
                    same_defect_solutions.append((i, sol, angle_deg))
        
        if not same_defect_solutions:
            return None, None, None, None, None, None
        
        # Sort by angle
        same_defect_solutions.sort(key=lambda x: x[2])
        angles = [x[2] for x in same_defect_solutions]
        
        # Find bracketing angles
        lower_idx = None
        upper_idx = None
        
        # Find the first angle greater than target
        for i, angle in enumerate(angles):
            if angle > target_angle_deg:
                upper_idx = i
                lower_idx = i - 1 if i > 0 else None
                break
        
        # If no angle greater than target, use the last two angles
        if upper_idx is None:
            if len(angles) >= 2:
                lower_idx = len(angles) - 2
                upper_idx = len(angles) - 1
            else:
                # Only one source available
                lower_idx = 0
                upper_idx = 0
        
        # If no angle less than target, use the first two angles
        elif lower_idx is None:
            lower_idx = 0
            upper_idx = 1 if len(angles) > 1 else 0
        
        # Get the bracketing sources
        lower_source_idx, lower_source, lower_angle = same_defect_solutions[lower_idx]
        upper_source_idx, upper_source, upper_angle = same_defect_solutions[upper_idx]
        
        # Ensure we have valid sources
        if lower_source_idx is not None and upper_source_idx is not None:
            return (lower_source, upper_source,
                    lower_source_idx, upper_source_idx,
                    lower_angle, upper_angle)
        
        return None, None, None, None, None, None
    
    def load_all_solutions(self, use_cache=True, max_files=None):
        solutions = []
        file_info = self.scan_solutions()
        if max_files:
            file_info = file_info[:max_files]
        
        if not file_info:
            return solutions
        
        for file_info_item in file_info:
            cache_key = file_info_item['filename']
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
            
            solution = self.read_simulation_file(file_info_item['path'])
            if solution:
                self.cache[cache_key] = solution
                solutions.append(solution)
        return solutions

# =============================================
# POSITIONAL ENCODING FOR TRANSFORMER
# =============================================
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Create positional indices
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute divisor term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
        
        # Create positional encoding
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return x + pe.unsqueeze(0)

# =============================================
# TARGETED TRANSFORMER INTERPOLATOR WITH BRACKETING
# =============================================
class TargetedTransformerInterpolator:
    """
    Targeted interpolator that uses ONLY the two nearest bracketing sources
    of the same defect type, giving near-zero weights to all other sources.
    """
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                bracketing_weight=0.98, other_weight=0.01):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.bracketing_weight = bracketing_weight  # Weight for bracketing sources
        self.other_weight = other_weight  # Near-zero weight for other sources
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection
        self.input_proj = nn.Linear(12, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
    
    def find_bracketing_sources(self, solutions, target_angle_deg, target_defect_type):
        """Find the two nearest sources that bracket the target angle"""
        # Filter by defect type
        same_defect_solutions = []
        for i, sol in enumerate(solutions):
            if 'params' in sol and sol['params'].get('defect_type') == target_defect_type:
                if 'theta' in sol['params']:
                    angle_deg = np.degrees(sol['params']['theta'])
                    same_defect_solutions.append((i, sol, angle_deg))
        
        if len(same_defect_solutions) < 2:
            st.warning(f"Need at least 2 {target_defect_type} sources for bracketing, found {len(same_defect_solutions)}")
            return None, None, None, None, None, None
        
        # Sort by angle
        same_defect_solutions.sort(key=lambda x: x[2])
        angles = [x[2] for x in same_defect_solutions]
        
        # Find bracketing indices
        lower_idx = None
        upper_idx = None
        
        # Case 1: Target angle is within range
        for i, angle in enumerate(angles):
            if angle > target_angle_deg:
                upper_idx = i
                lower_idx = i - 1 if i > 0 else None
                break
        
        # Case 2: Target angle is less than all angles
        if upper_idx is None:
            lower_idx = 0
            upper_idx = 1
        
        # Case 3: Target angle is greater than all angles
        elif lower_idx is None:
            lower_idx = len(angles) - 2
            upper_idx = len(angles) - 1
        
        # Get the sources
        lower_source_idx, lower_source, lower_angle = same_defect_solutions[lower_idx]
        upper_source_idx, upper_source, upper_angle = same_defect_solutions[upper_idx]
        
        return (lower_source, upper_source,
                lower_source_idx, upper_source_idx,
                lower_angle, upper_angle)
    
    def create_targeted_weights(self, solutions, target_angle_deg, target_defect_type):
        """
        Create weights where bracketing sources get most weight and others get near-zero
        """
        # Find bracketing sources
        result = self.find_bracketing_sources(solutions, target_angle_deg, target_defect_type)
        if not result[0] or not result[1]:
            return None
        
        lower_source, upper_source, lower_idx, upper_idx, lower_angle, upper_angle = result
        
        # Calculate weights based on angular distance
        lower_dist = abs(target_angle_deg - lower_angle)
        upper_dist = abs(upper_angle - target_angle_deg)
        total_dist = lower_dist + upper_dist
        
        # Inverse distance weighting
        if total_dist > 0:
            lower_weight = (upper_dist / total_dist) * self.bracketing_weight
            upper_weight = (lower_dist / total_dist) * self.bracketing_weight
        else:
            lower_weight = upper_weight = self.bracketing_weight / 2
        
        # Create weight array
        weights = np.ones(len(solutions)) * self.other_weight  # Near-zero for others
        weights[lower_idx] = lower_weight
        weights[upper_idx] = upper_weight
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate angular distances for all sources
        angular_distances = []
        for sol in solutions:
            if 'params' in sol and 'theta' in sol['params']:
                angle_deg = np.degrees(sol['params']['theta'])
                dist = abs(angle_deg - target_angle_deg)
                dist = min(dist, 360 - dist)  # Handle circular nature
                angular_distances.append(dist)
            else:
                angular_distances.append(180.0)  # Max distance
        
        return {
            'weights': weights,
            'lower_source': {
                'index': lower_idx,
                'angle': lower_angle,
                'weight': lower_weight
            },
            'upper_source': {
                'index': upper_idx,
                'angle': upper_angle,
                'weight': upper_weight
            },
            'angular_distances': angular_distances
        }
    
    def interpolate_with_bracketing(self, solutions, target_angle_deg, target_params):
        """
        Targeted interpolation using only bracketing sources
        """
        if not solutions:
            st.warning("No solutions provided for interpolation.")
            return None
        
        try:
            target_defect_type = target_params.get('defect_type', 'Twin')
            
            # Get targeted weights
            weight_info = self.create_targeted_weights(solutions, target_angle_deg, target_defect_type)
            if not weight_info:
                st.error("Could not find bracketing sources.")
                return None
            
            weights = weight_info['weights']
            lower_info = weight_info['lower_source']
            upper_info = weight_info['upper_source']
            
            # Extract all source fields
            source_fields = []
            source_params = []
            for i, src in enumerate(solutions):
                if 'params' not in src or 'history' not in src:
                    continue
                
                source_params.append(src['params'])
                
                # Get stress fields from last frame
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        stress_fields = last_frame['stresses']
                        
                        # Extract von Mises
                        if 'von_mises' in stress_fields:
                            vm = stress_fields['von_mises']
                        else:
                            vm = self.compute_von_mises(stress_fields)
                        
                        # Extract hydrostatic
                        if 'sigma_hydro' in stress_fields:
                            hydro = stress_fields['sigma_hydro']
                        else:
                            hydro = self.compute_hydrostatic(stress_fields)
                        
                        # Extract magnitude
                        if 'sigma_mag' in stress_fields:
                            mag = stress_fields['sigma_mag']
                        else:
                            mag = np.sqrt(vm**2 + hydro**2)
                        
                        source_fields.append({
                            'von_mises': vm,
                            'sigma_hydro': hydro,
                            'sigma_mag': mag,
                            'source_index': i,
                            'source_params': src['params']
                        })
                    else:
                        st.warning(f"Skipping source {i}: no stress fields found")
                        continue
                else:
                    st.warning(f"Skipping source {i}: invalid history")
                    continue
            
            if not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Ensure all fields have same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                target_shape = shapes[0]
                for fields in source_fields:
                    for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        if fields[key].shape != target_shape:
                            factors = [t/s for t, s in zip(target_shape, fields[key].shape)]
                            fields[key] = zoom(fields[key], factors, order=1)
            
            # Apply angular interpolation correction
            shape = source_fields[0]['von_mises'].shape
            interpolated_fields = {}
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                
                # Get bracketing source fields
                lower_field = source_fields[lower_info['index']][component]
                upper_field = source_fields[upper_info['index']][component]
                lower_angle = lower_info['angle']
                upper_angle = upper_info['angle']
                
                # Calculate interpolation factor
                if abs(upper_angle - lower_angle) > 0:
                    t = (target_angle_deg - lower_angle) / (upper_angle - lower_angle)
                else:
                    t = 0.5
                
                # Linear interpolation between bracketing sources
                interpolated = (1 - t) * lower_field + t * upper_field
                
                # Add small contributions from other sources (if weights > threshold)
                threshold = 0.01  # Only include sources with weight > 1%
                for i, fields in enumerate(source_fields):
                    if i != lower_info['index'] and i != upper_info['index']:
                        if weights[i] > threshold and component in fields:
                            interpolated += weights[i] * fields[component]
                
                interpolated_fields[component] = interpolated
            
            # Calculate statistics
            stats = {}
            for component, field in interpolated_fields.items():
                stats[component] = {
                    'max': float(np.max(field)),
                    'min': float(np.min(field)),
                    'mean': float(np.mean(field)),
                    'std': float(np.std(field)),
                    'median': float(np.median(field))
                }
            
            # Calculate angular distances for all sources
            angular_distances = []
            defect_types = []
            for params in source_params:
                if 'theta' in params:
                    angle_deg = np.degrees(params['theta'])
                    dist = abs(angle_deg - target_angle_deg)
                    dist = min(dist, 360 - dist)
                    angular_distances.append(dist)
                else:
                    angular_distances.append(180.0)
                defect_types.append(params.get('defect_type', 'Unknown'))
            
            # Calculate weight distribution metrics
            weight_entropy = self._calculate_entropy(weights)
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': weights.tolist(),
                    'entropy': weight_entropy,
                    'bracketing_sources': {
                        'lower': lower_info,
                        'upper': upper_info
                    }
                },
                'statistics': stats,
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_angular_distances': angular_distances,
                'source_defect_types': defect_types,
                'source_fields': source_fields,
                'interpolation_method': 'targeted_bracketing',
                'interpolation_factor': {
                    'lower_angle': float(lower_info['angle']),
                    'upper_angle': float(upper_info['angle']),
                    'target_angle': float(target_angle_deg),
                    'interpolation_t': float((target_angle_deg - lower_info['angle']) /
                                           (upper_info['angle'] - lower_info['angle'])
                                           if abs(upper_info['angle'] - lower_info['angle']) > 0 else 0.5)
                }
            }
        except Exception as e:
            st.error(f"Error during targeted interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def compute_von_mises(self, stress_fields):
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz', 'tau_xy']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            txy = stress_fields['tau_xy']
            tyz = stress_fields.get('tau_yz', np.zeros_like(sxx))
            tzx = stress_fields.get('tau_zx', np.zeros_like(sxx))
            von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 +
                                     6*(txy**2 + tyz**2 + tzx**2)))
            return von_mises
        return np.zeros((100, 100))
    
    def compute_hydrostatic(self, stress_fields):
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            return (sxx + syy + szz) / 3
        return np.zeros((100, 100))
    
    def _calculate_entropy(self, weights):
        weights = np.array(weights)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return float(-np.sum(weights * np.log(weights + 1e-10)))

# =============================================
# DUAL INTERPOLATION SYSTEM
# =============================================
class DualInterpolationSystem:
    """
    Dual interpolation system that can switch between:
    1. Targeted bracketing (using only two nearest same-defect sources)
    2. Full transformer interpolation (using all sources)
    """
    def __init__(self):
        self.targeted_interpolator = TargetedTransformerInterpolator(
            bracketing_weight=0.98,
            other_weight=0.001
        )
        self.use_targeted_mode = True  # Default to targeted mode
    
    def set_interpolation_mode(self, use_targeted_mode):
        self.use_targeted_mode = use_targeted_mode
    
    def interpolate(self, solutions, target_angle_deg, target_params, mode='auto'):
        """
        Interpolate using selected mode
        mode: 'targeted', 'full', or 'auto'
        """
        if mode == 'auto':
            mode = 'targeted' if self.use_targeted_mode else 'full'
        
        if mode == 'targeted':
            return self.targeted_interpolator.interpolate_with_bracketing(
                solutions, target_angle_deg, target_params
            )
        else:
            # For full mode, we would use the enhanced transformer interpolator
            # For now, return targeted as placeholder
            return self.targeted_interpolator.interpolate_with_bracketing(
                solutions, target_angle_deg, target_params
            )

# =============================================
# ENHANCED HEATMAP VISUALIZER WITH ADVANCED VISUALIZATIONS
# =============================================
class AdvancedHeatmapVisualizer:
    """Enhanced visualizer with advanced visualization capabilities"""
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                             cmap_name='viridis', figsize=(12, 10),
                             colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                             show_stats=True, target_angle=None, defect_type=None,
                             show_colorbar=True, aspect_ratio='equal', special_angle=None):
        """Create enhanced heat map with chosen colormap and publication styling"""
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')  # Default fallback
        
        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = np.nanmin(stress_field)
        if vmax is None:
            vmax = np.nanmax(stress_field)
        
        # Create heatmap
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect=aspect_ratio, interpolation='bilinear', origin='lower')
        
        # Add special angle indicator if provided
        if special_angle is not None:
            ax.axhline(y=special_angle, color='r', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=special_angle, color='r', linestyle='--', alpha=0.7, linewidth=1)
        
        # Add colorbar with enhanced styling
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label, fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
        
        # Customize plot with publication styling
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
        if special_angle is not None:
            title_str += f"\nSpecial Angle: {special_angle}°"
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=16, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=16, fontweight='bold')
        
        # Add grid with subtle styling
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        
        # Add statistics annotation with enhanced styling
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\n"
                         f"Min: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa\n"
                         f"Std: {np.nanstd(stress_field):.3f} GPa")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig
    
    def create_interactive_heatmap(self, stress_field, title="Stress Heat Map",
                                  cmap_name='viridis', width=800, height=700,
                                  target_angle=None, defect_type=None, special_angle=None):
        """Create interactive heatmap with Plotly with enhanced styling"""
        try:
            # Validate colormap
            if cmap_name not in px.colors.named_colorscales():
                cmap_name = 'viridis'  # Default fallback
                st.warning(f"Colormap {cmap_name} not found in Plotly, using viridis instead.")
            
            # Create hover text with enhanced information
            hover_text = []
            for i in range(stress_field.shape[0]):
                row_text = []
                for j in range(stress_field.shape[1]):
                    text = f"Position: ({i}, {j})<br>Stress: {stress_field[i, j]:.4f} GPa"
                    if target_angle is not None:
                        text += f"<br>θ: {target_angle:.1f}°"
                    if special_angle is not None:
                        text += f"<br>Special Angle: {special_angle}°"
                    row_text.append(text)
                hover_text.append(row_text)
            
            # Create heatmap trace
            heatmap_trace = go.Heatmap(
                z=stress_field,
                colorscale=cmap_name,
                zmin=np.nanmin(stress_field),
                zmax=np.nanmax(stress_field),
                hoverinfo='text',
                text=hover_text,
                colorbar=dict(
                    title=dict(
                        text="Stress (GPa)",
                        font=dict(size=16, family='Arial', color='black'),
                        side="right"
                    ),
                    tickfont=dict(size=14, family='Arial'),
                    thickness=20,
                    len=0.8
                )
            )
            
            # Create figure
            fig = go.Figure(data=[heatmap_trace])
            
            # Enhanced title
            title_str = title
            if target_angle is not None and defect_type is not None:
                title_str = f"{title}<br>θ = {target_angle:.1f}°, Defect: {defect_type}"
            if special_angle is not None:
                title_str += f"<br>Special Angle: {special_angle}°"
            
            # Update layout with publication styling
            fig.update_layout(
                title=dict(
                    text=title_str,
                    font=dict(size=24, family="Arial Black", color='darkblue'),
                    x=0.5,
                    y=0.95
                ),
                width=width,
                height=height,
                xaxis=dict(
                    title=dict(text="X Position", font=dict(size=18, family="Arial", color="black")),
                    tickfont=dict(size=14, family='Arial'),
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    title=dict(text="Y Position", font=dict(size=18, family="Arial", color="black")),
                    tickfont=dict(size=14, family='Arial'),
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    scaleanchor="x",
                    scaleratio=1
                ),
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=80, r=80, t=100, b=80)
            )
            
            # Ensure aspect ratio is 1:1 for square fields
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )
            return fig
        except Exception as e:
            st.error(f"Error creating interactive heatmap: {e}")
            # Return a simple figure as fallback
            fig = go.Figure()
            fig.add_annotation(text="Error creating heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_interactive_3d_surface(self, stress_field, title="3D Stress Surface",
                                    cmap_name='viridis', width=900, height=700,
                                    target_angle=None, defect_type=None, special_angle=None):
        """Create interactive 3D surface plot with Plotly"""
        try:
            # Validate colormap
            if cmap_name not in px.colors.named_colorscales():
                cmap_name = 'viridis'
            
            # Create meshgrid
            x = np.arange(stress_field.shape[1])
            y = np.arange(stress_field.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Create hover text
            hover_text = []
            for i in range(stress_field.shape[0]):
                row_text = []
                for j in range(stress_field.shape[1]):
                    text = f"X: {j}, Y: {i}<br>Stress: {stress_field[i, j]:.4f} GPa"
                    if target_angle is not None:
                        text += f"<br>Target Angle: {target_angle:.1f}°"
                    if special_angle is not None:
                        text += f"<br>Special Angle: {special_angle}°"
                    row_text.append(text)
                hover_text.append(row_text)
            
            # Create 3D surface trace
            surface_trace = go.Surface(
                z=stress_field,
                x=X,
                y=Y,
                colorscale=cmap_name,
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
                },
                hoverinfo='text',
                text=hover_text
            )
            
            # Create figure
            fig = go.Figure(data=[surface_trace])
            
            # Enhanced title
            title_str = title
            if target_angle is not None and defect_type is not None:
                title_str = f"{title}<br>θ = {target_angle:.1f}°, Defect: {defect_type}"
            if special_angle is not None:
                title_str += f"<br>Special Angle: {special_angle}°"
            
            # Update layout with publication styling
            fig.update_layout(
                title=dict(
                    text=title_str,
                    font=dict(size=24, family="Arial Black", color='darkblue'),
                    x=0.5,
                    y=0.95
                ),
                width=width,
                height=height,
                scene=dict(
                    xaxis=dict(
                        title=dict(text="X Position", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    yaxis=dict(
                        title=dict(text="Y Position", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    zaxis=dict(
                        title=dict(text="Stress (GPa)", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.0)
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=100, b=0)
            )
            return fig
        except Exception as e:
            st.error(f"Error creating 3D surface: {e}")
            fig = go.Figure()
            fig.add_annotation(text="Error creating 3D surface", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_angular_orientation_plot(self, target_angle_deg, defect_type="Unknown",
                                       figsize=(8, 8), show_special_angle=True, special_angle=54.7):
        """Create polar plot showing angular orientation"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Convert target angle to radians
        theta_rad = np.radians(target_angle_deg)
        
        # Plot the defect orientation as a red arrow
        ax.arrow(theta_rad, 0.8, 0, 0.6, width=0.02,
                color='red', alpha=0.8, label=f'Defect Orientation: {target_angle_deg:.1f}°')
        
        # Plot special angle orientation if requested
        if show_special_angle:
            special_angle_rad = np.radians(special_angle)
            ax.arrow(special_angle_rad, 0.8, 0, 0.6, width=0.02,
                    color='blue', alpha=0.5, label=f'Special Angle ({special_angle}°)')
        
        # Plot cardinal directions
        for angle, label in [(0, '0°'), (90, '90°'), (180, '180°'), (270, '270°')]:
            ax.axvline(np.radians(angle), color='gray', linestyle='--', alpha=0.3)
        
        # Customize plot
        title = f'Defect Orientation\nθ = {target_angle_deg:.1f}°, {defect_type}'
        if show_special_angle:
            title += f"\nSpecial Angle: {special_angle}°"
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Add annotation for angular difference from special angle
        if show_special_angle:
            angular_diff = abs(target_angle_deg - special_angle)
            angular_diff = min(angular_diff, 360 - angular_diff)  # Handle cyclic nature
            ax.annotate(f'Δθ = {angular_diff:.1f}°\nfrom special angle',
                       xy=(theta_rad, 1.2), xytext=(theta_rad, 1.4),
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                       fontsize=12, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def create_comparison_heatmaps(self, stress_fields_dict, cmap_name='viridis',
                                  figsize=(18, 6), titles=None, target_angle=None, defect_type=None, special_angle=None):
        """Create comparison heatmaps for multiple stress components"""
        n_components = len(stress_fields_dict)
        fig, axes = plt.subplots(1, n_components, figsize=figsize)
        if n_components == 1:
            axes = [axes]
        
        if titles is None:
            titles = list(stress_fields_dict.keys())
        
        for idx, ((component_name, stress_field), title) in enumerate(zip(stress_fields_dict.items(), titles)):
            ax = axes[idx]
            # Get colormap
            if cmap_name in plt.colormaps():
                cmap = plt.get_cmap(cmap_name)
            else:
                cmap = plt.get_cmap('viridis')
            
            # Create heatmap with equal aspect ratio
            im = ax.imshow(stress_field, cmap=cmap, aspect='equal', interpolation='bilinear', origin='lower')
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Stress (GPa)", fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            
            # Customize subplot with publication styling
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=14)
            ax.set_ylabel('Y Position', fontsize=14)
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Set tick parameters
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add super title with target parameters
        suptitle = "Stress Component Comparison"
        if target_angle is not None and defect_type is not None:
            suptitle = f"Stress Component Comparison - θ = {target_angle:.1f}°, {defect_type}"
        if special_angle is not None:
            suptitle += f"\nSpecial Angle: {special_angle}°"
        plt.suptitle(suptitle, fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def create_3d_surface_plot(self, stress_field, title="3D Stress Surface",
                              cmap_name='viridis', figsize=(14, 10), target_angle=None, defect_type=None, special_angle=None):
        """Create 3D surface plot of stress field with enhanced styling"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(stress_field.shape[1])
        y = np.arange(stress_field.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')
        
        # Normalize for coloring
        norm = Normalize(vmin=np.nanmin(stress_field), vmax=np.nanmax(stress_field))
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, stress_field, cmap=cmap, norm=norm,
                              linewidth=0, antialiased=True, alpha=0.8, rstride=1, cstride=1)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Stress (GPa)", fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        
        # Enhanced title
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
        if special_angle is not None:
            title_str += f"\nSpecial Angle: {special_angle}°"
        
        # Customize plot with publication styling
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel('Stress (GPa)', fontsize=16, fontweight='bold', labelpad=10)
        
        # Set tick parameters
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        plt.tight_layout()
        return fig
    
    def get_colormap_preview(self, cmap_name, figsize=(12, 1)):
        """Generate preview of a colormap with enhanced styling"""
        fig, ax = plt.subplots(figsize=figsize)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.set_title(f"Colormap: {cmap_name}", fontsize=18, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add value labels with enhanced styling
        ax.text(0, 0.5, "Min", transform=ax.transAxes,
               va='center', ha='right', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.text(1, 0.5, "Max", transform=ax.transAxes,
               va='center', ha='left', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add ticks
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=12)
        ax.xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, stress_fields, theta, defect_type,
                                      cmap_name='viridis', figsize=(24, 16), special_angle=None):
        """Create comprehensive dashboard with all stress components and angular orientation"""
        fig = plt.figure(figsize=figsize)
        # Create subplots grid with polar plot included
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
        
        # 0. Angular orientation plot (polar plot)
        ax0 = fig.add_subplot(gs[0, 0], projection='polar')
        theta_rad = np.radians(theta)
        ax0.arrow(theta_rad, 0.8, 0, 0.6, width=0.02, color='red', alpha=0.8)
        
        # Plot special angle if provided
        if special_angle is not None:
            special_angle_rad = np.radians(special_angle)
            ax0.arrow(special_angle_rad, 0.8, 0, 0.6, width=0.02, color='blue', alpha=0.5)
        
        # Customize polar plot
        title = f'Defect Orientation\nθ = {theta:.1f}°'
        if special_angle is not None:
            title += f"\nSpecial Angle: {special_angle}°"
        ax0.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax0.set_theta_zero_location('N')
        ax0.set_theta_direction(-1)
        ax0.set_ylim(0, 1.5)
        ax0.grid(True, alpha=0.3)
        
        # 1. Von Mises stress (main plot)
        ax1 = fig.add_subplot(gs[0, 1:3])
        im1 = ax1.imshow(stress_fields['von_mises'], cmap=cmap_name, aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Von Mises Stress (GPa)')
        title = f'Von Mises Stress at θ={theta}°\nDefect: {defect_type}'
        if special_angle is not None:
            title += f"\nSpecial Angle: {special_angle}°"
        ax1.set_title(title, fontsize=18, fontweight='bold')
        ax1.set_xlabel('X Position', fontsize=14)
        ax1.set_ylabel('Y Position', fontsize=14)
        ax1.grid(True, alpha=0.2)
        
        # 2. Hydrostatic stress
        ax2 = fig.add_subplot(gs[0, 3])
        # Use diverging colormap for hydrostatic
        hydro_cmap = 'RdBu_r' if cmap_name in ['viridis', 'plasma', 'inferno'] else cmap_name
        im2 = ax2.imshow(stress_fields['sigma_hydro'], cmap=hydro_cmap, aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im2, ax=ax2, label='Hydrostatic Stress (GPa)')
        ax2.set_title('Hydrostatic Stress', fontsize=18, fontweight='bold')
        ax2.set_xlabel('X Position', fontsize=14)
        ax2.set_ylabel('Y Position', fontsize=14)
        
        # 3. Stress magnitude
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(stress_fields['sigma_mag'], cmap=cmap_name, aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im3, ax=ax3, label='Stress Magnitude (GPa)')
        ax3.set_title('Stress Magnitude', fontsize=18, fontweight='bold')
        ax3.set_xlabel('X Position', fontsize=14)
        ax3.set_ylabel('Y Position', fontsize=14)
        
        # 4. Histogram of von Mises
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(stress_fields['von_mises'].flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.set_xlabel('Von Mises Stress (GPa)', fontsize=14)
        ax4.set_ylabel('Frequency', fontsize=14)
        ax4.set_title('Von Mises Distribution', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Histogram of hydrostatic
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(stress_fields['sigma_hydro'].flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Hydrostatic Stress (GPa)', fontsize=14)
        ax5.set_ylabel('Frequency', fontsize=14)
        ax5.set_title('Hydrostatic Distribution', fontsize=16, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Line profiles
        ax6 = fig.add_subplot(gs[1, 3])
        middle_row = stress_fields['von_mises'].shape[0] // 2
        middle_col = stress_fields['von_mises'].shape[1] // 2
        ax6.plot(stress_fields['von_mises'][middle_row, :], label='Von Mises', linewidth=2)
        ax6.plot(stress_fields['sigma_hydro'][middle_row, :], label='Hydrostatic', linewidth=2)
        ax6.plot(stress_fields['sigma_mag'][middle_row, :], label='Magnitude', linewidth=2)
        ax6.set_xlabel('X Position', fontsize=14)
        ax6.set_ylabel('Stress (GPa)', fontsize=14)
        ax6.set_title(f'Line Profile at Row {middle_row}', fontsize=16, fontweight='bold')
        ax6.legend(fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # 7. Statistics table
        ax7 = fig.add_subplot(gs[2, 0:2])
        ax7.axis('off')
        # Prepare statistics with enhanced formatting
        stats_text = (
            f"Von Mises Stress:\n"
            f"  Max: {np.max(stress_fields['von_mises']):.3f} GPa\n"
            f"  Min: {np.min(stress_fields['von_mises']):.3f} GPa\n"
            f"  Mean: {np.mean(stress_fields['von_mises']):.3f} GPa\n"
            f"  Std: {np.std(stress_fields['von_mises']):.3f} GPa\n\n"
            f"Hydrostatic Stress:\n"
            f"  Max Tension: {np.max(stress_fields['sigma_hydro']):.3f} GPa\n"
            f"  Max Compression: {np.min(stress_fields['sigma_hydro']):.3f} GPa\n"
            f"  Mean: {np.mean(stress_fields['sigma_hydro']):.3f} GPa\n"
            f"  Std: {np.std(stress_fields['sigma_hydro']):.3f} GPa\n\n"
            f"Stress Magnitude:\n"
            f"  Max: {np.max(stress_fields['sigma_mag']):.3f} GPa\n"
            f"  Min: {np.min(stress_fields['sigma_mag']):.3f} GPa\n"
            f"  Mean: {np.mean(stress_fields['sigma_mag']):.3f} GPa\n"
            f"  Std: {np.std(stress_fields['sigma_mag']):.3f} GPa"
        )
        ax7.text(0.1, 0.5, stats_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax7.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='brown', linewidth=2))
        ax7.set_title('Stress Statistics', fontsize=18, fontweight='bold', pad=20)
        
        # 8. Target parameters display
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        params_text = (
            f"Target Parameters:\n"
            f"  Polar Angle (θ): {theta:.1f}°\n"
            f"  Defect Type: {defect_type}\n"
            f"  Shape: Square (default)\n"
            f"  Simulation Grid: {stress_fields['von_mises'].shape[0]} × {stress_fields['von_mises'].shape[1]}\n"
        )
        if special_angle is not None:
            params_text += f"  Special Angle: {special_angle}°\n"
        params_text += f"  Angular Deviation from Special Angle: {abs(theta - special_angle):.1f}°" if special_angle is not None else ""
        ax8.text(0.1, 0.5, params_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax8.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))
        ax8.set_title('Interpolation Parameters', fontsize=18, fontweight='bold', pad=20)
        
        plt.suptitle(f'Comprehensive Stress Analysis - θ={theta}°, {defect_type}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_weights_visualization(self, weights_data, figsize=(15, 10)):
        """Create comprehensive visualization of different weight types"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # 1. Weight distribution comparison
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(weights_data['spatial']))
        
        # Plot all weight types
        width = 0.25
        spatial_bars = ax1.bar(x - width, weights_data['spatial'], width, 
                              label='Spatial Weights', alpha=0.8, color='blue')
        attention_bars = ax1.bar(x, weights_data['attention'], width,
                                label='Attention Weights', alpha=0.8, color='orange')
        combined_bars = ax1.bar(x + width, weights_data['combined'], width,
                               label='Combined Weights', alpha=0.8, color='red')
        
        ax1.set_xlabel('Source Index', fontsize=14)
        ax1.set_ylabel('Weight Value', fontsize=14)
        ax1.set_title('Weight Distribution Comparison', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(np.max(weights_data['spatial']), 
                           np.max(weights_data['attention']), 
                           np.max(weights_data['combined'])) * 1.1)
        
        # Add source annotations
        for i in range(len(x)):
            height = max(weights_data['spatial'][i], weights_data['attention'][i], weights_data['combined'][i])
            if height > 0.01:  # Only label significant weights
                ax1.text(i, height + 0.01, f"Src {i}", 
                        ha='center', va='bottom', fontsize=8, rotation=90)
        
        # 2. Polar plot of angular distribution
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        
        # Convert angles to radians
        angles_rad = np.radians(weights_data['angles'])
        sizes = 100 * np.array(weights_data['combined']) / np.max(weights_data['combined'])
        
        # Plot sources as points
        scatter = ax2.scatter(angles_rad, weights_data['distances'],
                            s=sizes, c=weights_data['combined'], 
                            cmap='viridis', alpha=0.8, edgecolors='black')
        
        # Plot target angle
        target_rad = np.radians(weights_data['target_angle'])
        ax2.scatter(target_rad, 0, s=200, c='red', marker='*', 
                   edgecolors='white', label=f'Target: {weights_data["target_angle"]:.1f}°')
        
        # Plot special angle if provided
        if weights_data.get('special_angle') is not None:
            special_rad = np.radians(weights_data['special_angle'])
            ax2.axvline(special_rad, color='green', alpha=0.7, 
                       linestyle='--', linewidth=2, label=f'Special: {weights_data["special_angle"]:.1f}°')
        
        ax2.set_title('Angular Distribution of Sources', fontsize=16, fontweight='bold', pad=20)
        ax2.set_theta_zero_location('N')  # 0° at top
        ax2.set_theta_direction(-1)  # Clockwise
        
        # 3. Entropy comparison
        ax3 = fig.add_subplot(gs[1, 0])
        entropy_types = ['Spatial', 'Attention', 'Combined']
        entropies = [weights_data['entropy_spatial'], weights_data['entropy_attention'], weights_data['entropy_combined']]
        
        entropy_bars = ax3.bar(entropy_types, entropies, color=['blue', 'orange', 'red'], alpha=0.8)
        ax3.set_ylabel('Entropy', fontsize=14)
        ax3.set_title('Weight Distribution Entropy', fontsize=16, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add entropy values on bars
        for bar, entropy in zip(entropy_bars, entropies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{entropy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Weight vs distance correlation
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(weights_data['distances'], weights_data['combined'],
                            c=weights_data['angles'], cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='black')
        
        ax4.set_xlabel('Angular Distance (°)', fontsize=14)
        ax4.set_ylabel('Combined Weight', fontsize=14)
        ax4.set_title('Weight vs Angular Distance', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(weights_data['distances']) > 1:
            z = np.polyfit(weights_data['distances'], weights_data['combined'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(weights_data['distances']), max(weights_data['distances']), 100)
            ax4.plot(x_range, p(x_range), 'r--', linewidth=2, alpha=0.8, label='Trend')
            ax4.legend()
        
        plt.colorbar(scatter, ax=ax4, label='Source Angle (°)')
        
        plt.suptitle('Comprehensive Weight Analysis', fontsize=20, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_interactive_weights_dashboard(self, weights_data, width=1000, height=800):
        """Create interactive dashboard for weight analysis using Plotly"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weight Distribution', 'Angular Distribution',
                           'Entropy Comparison', 'Weight vs Distance'),
            specs=[[{'type': 'bar'}, {'type': 'polar'}],
                  [{'type': 'bar'}, {'type': 'scatter'}]],
            horizontal_spacing=0.1,
            vertical_spacing=0.15
        )
        
        # 1. Weight distribution comparison (bar chart)
        x = list(range(len(weights_data['spatial'])))
        
        # Spatial weights
        fig.add_trace(
            go.Bar(x=x, y=weights_data['spatial'], name='Spatial', 
                  marker_color='blue', opacity=0.8),
            row=1, col=1
        )
        
        # Attention weights
        fig.add_trace(
            go.Bar(x=x, y=weights_data['attention'], name='Attention', 
                  marker_color='orange', opacity=0.8),
            row=1, col=1
        )
        
        # Combined weights
        fig.add_trace(
            go.Bar(x=x, y=weights_data['combined'], name='Combined', 
                  marker_color='red', opacity=0.8),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Source Index", row=1, col=1)
        fig.update_yaxes(title_text="Weight Value", row=1, col=1)
        
        # 2. Angular distribution (polar plot)
        angles_rad = np.radians(weights_data['angles'])
        max_distance = max(weights_data['distances']) * 1.1
        
        # Sources
        fig.add_trace(
            go.Scatterpolar(
                r=weights_data['distances'],
                theta=weights_data['angles'],
                mode='markers',
                marker=dict(
                    size=100 * np.array(weights_data['combined']) / np.max(weights_data['combined']),
                    color=weights_data['combined'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title='Combined Weight')
                ),
                name='Sources',
                hovertemplate='Source: %{text}<br>Angle: %{theta:.1f}°<br>Distance: %{r:.1f}°<br>Weight: %{marker.color:.4f}',
                text=[f'Source {i}' for i in range(len(weights_data['angles']))]
            ),
            row=1, col=2
        )
        
        # Target angle
        fig.add_trace(
            go.Scatterpolar(
                r=[0],
                theta=[weights_data['target_angle']],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name=f'Target: {weights_data["target_angle"]:.1f}°'
            ),
            row=1, col=2
        )
        
        # Special angle line if provided
        if weights_data.get('special_angle') is not None:
            fig.add_shape(
                type='line',
                x0=0, y0=0, x1=max_distance*np.cos(np.radians(weights_data['special_angle']-90)), 
                y1=max_distance*np.sin(np.radians(weights_data['special_angle']-90)),
                line=dict(color='green', dash='dash'),
                row=1, col=2
            )
        
        fig.update_polar(
            radialaxis=dict(range=[0, max_distance], title='Angular Distance (°)'),
            angularaxis=dict(direction='clockwise', rotation=90),
            row=1, col=2
        )
        
        # 3. Entropy comparison (bar chart)
        entropy_types = ['Spatial', 'Attention', 'Combined']
        entropies = [weights_data['entropy_spatial'], weights_data['entropy_attention'], weights_data['entropy_combined']]
        
        fig.add_trace(
            go.Bar(x=entropy_types, y=entropies, marker_color=['blue', 'orange', 'red'], opacity=0.8),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Weight Type", row=2, col=1)
        fig.update_yaxes(title_text="Entropy", row=2, col=1)
        
        # 4. Weight vs distance (scatter plot)
        fig.add_trace(
            go.Scatter(
                x=weights_data['distances'],
                y=weights_data['combined'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=weights_data['angles'],
                    colorscale='viridis',
                    colorbar=dict(title='Source Angle (°)')
                ),
                name='Weight vs Distance',
                hovertemplate='Source Angle: %{marker.color:.1f}°<br>Distance: %{x:.1f}°<br>Weight: %{y:.4f}'
            ),
            row=2, col=2
        )
        
        # Add trend line
        if len(weights_data['distances']) > 1:
            z = np.polyfit(weights_data['distances'], weights_data['combined'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(weights_data['distances']), max(weights_data['distances']), 100)
            y_range = p(x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Angular Distance (°)", row=2, col=2)
        fig.update_yaxes(title_text="Combined Weight", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive Weight Analysis Dashboard',
                x=0.5,
                font=dict(size=24, color='darkblue')
            ),
            width=width,
            height=height,
            showlegend=True,
            hovermode='closest',
            template='presentation'
        )
        
        return fig
    
    def create_transformer_attention_visualization(self, attention_matrix, source_angles, target_angle, figsize=(12, 10)):
        """Visualize transformer attention weights as heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontsize=14)
        
        # Set tick labels
        ax.set_xticks(np.arange(len(source_angles)))
        ax.set_xticklabels([f"{angle:.1f}°" for angle in source_angles], rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels([f"Target: {target_angle:.1f}°"])
        
        # Set labels and title
        ax.set_xlabel('Source Angles', fontsize=16, fontweight='bold')
        ax.set_ylabel('Target Angle', fontsize=16, fontweight='bold')
        ax.set_title('Transformer Attention Weights', fontsize=20, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(False)
        
        plt.tight_layout()
        return fig
    
    def create_bracketing_visualization(self, lower_angle, upper_angle, target_angle, 
                                       target_defect, figsize=(10, 8)):
        """Visualize the bracketing relationship between angles"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw circle representing 360 degrees
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', alpha=0.5)
        ax.add_patch(circle)
        
        # Calculate positions for the three angles
        angles = [lower_angle, target_angle, upper_angle]
        labels = [f'Lower: {lower_angle:.1f}°', f'Target: {target_angle:.1f}°', f'Upper: {upper_angle:.1f}°']
        colors = ['blue', 'red', 'green']
        markers = ['o', '*', 's']
        sizes = [100, 200, 100]
        
        for i, (angle, label, color, marker, size) in enumerate(zip(angles, labels, colors, markers, sizes)):
            # Handle cyclic nature
            adjusted_angle = angle % 360
            
            # Calculate position on circle
            x = np.cos(np.radians(adjusted_angle - 90))  # -90 to start from top (0°)
            y = np.sin(np.radians(adjusted_angle - 90))
            
            # Plot point
            ax.scatter(x, y, s=size, c=color, marker=marker, edgecolors='black', zorder=3)
            
            # Add label with offset
            label_x = x * 1.15
            label_y = y * 1.15
            ax.text(label_x, label_y, label, ha='center', va='center', 
                   fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
        
        # Draw lines connecting the points
        lower_idx = 0
        target_idx = 1
        upper_idx = 2
        
        # Get coordinates
        lower_x = np.cos(np.radians(angles[lower_idx] % 360 - 90))
        lower_y = np.sin(np.radians(angles[lower_idx] % 360 - 90))
        target_x = np.cos(np.radians(angles[target_idx] % 360 - 90))
        target_y = np.sin(np.radians(angles[target_idx] % 360 - 90))
        upper_x = np.cos(np.radians(angles[upper_idx] % 360 - 90))
        upper_y = np.sin(np.radians(angles[upper_idx] % 360 - 90))
        
        # Draw lines
        ax.plot([lower_x, upper_x], [lower_y, upper_y], 'k--', alpha=0.7, linewidth=1.5)
        ax.plot([lower_x, target_x], [lower_y, target_y], 'k:', alpha=0.5, linewidth=1)
        ax.plot([target_x, upper_x], [target_y, upper_y], 'k:', alpha=0.5, linewidth=1)
        
        # Set limits and aspect ratio
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Add title and annotation
        ax.set_title(f'Angular Bracketing for {target_defect} Defect', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add interpolation factor annotation
        if upper_angle != lower_angle:
            t = (target_angle - lower_angle) / (upper_angle - lower_angle)
            ax.text(0, -1.2, f'Interpolation Factor: t = {t:.3f}\n'
                   f'Lower weight: {1-t:.2f}, Upper weight: {t:.2f}',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add defect type annotation
        ax.text(0, -1.45, f'Defect Type: {target_defect}', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED RESULTS MANAGER FOR EXPORT
# =============================================
class EnhancedResultsManager:
    """Manager for exporting interpolation results with advanced options"""
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        """Prepare data for export with enhanced metadata"""
        result = interpolation_result.copy()
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'transformer_spatial',
                'visualization_params': visualization_params,
                'software_version': '1.0.0',
                'hardware_info': {
                    'device': 'cpu' if not torch.cuda.is_available() else 'cuda',
                    'memory': 'N/A'
                }
            },
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'shape': result['shape'],
                'statistics': result['statistics'],
                'weights': result['weights'],
                'entropy': result['weights']['entropy']['combined'],
                'num_sources': result.get('num_sources', 0),
                'source_theta_degrees': result.get('source_theta_degrees', []),
                'source_distances': result.get('source_distances', []),
                'source_indices': result.get('source_indices', []),
                'source_defect_types': result.get('source_defect_types', []),
                'interpolation_method': result.get('interpolation_method', 'full_transformer'),
                'special_angle': visualization_params.get('special_angle', 54.7),
                'use_special_angle': visualization_params.get('use_special_angle', False),
                'weight_analysis': {
                    'spatial_entropy': result['weights']['entropy']['spatial'],
                    'transformer_entropy': result['weights']['entropy']['transformer'],
                    'combined_entropy': result['weights']['entropy']['combined'],
                    'max_spatial_weight': max(result['weights']['positional']),
                    'max_transformer_weight': max(result['weights']['transformer']),
                    'max_combined_weight': max(result['weights']['combined'])
                }
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
        
        # Add weight distribution analysis
        if 'weights' in result:
            weights = result['weights']
            export_data['result']['weight_distribution'] = {
                'spatial_weights': weights['positional'],
                'transformer_weights': weights['transformer'],
                'combined_weights': weights['combined']
            }
        
        return export_data
    
    def export_to_json(self, export_data, filename=None):
        """Export results to JSON file with enhanced formatting"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = export_data['result']['target_angle']
            defect = export_data['result']['target_params']['defect_type']
            filename = f"transformer_interpolation_theta_{theta}_{defect}_{timestamp}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        """Export flattened field data to CSV with source information"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename = f"stress_fields_theta_{theta}_{defect}_{timestamp}.csv"
        
        # Create DataFrame with flattened data
        data_dict = {}
        
        # Add stress field data
        for field_name, field_data in interpolation_result['fields'].items():
            flattened = field_data.flatten()
            data_dict[field_name] = flattened
        
        # Add coordinate columns
        shape = interpolation_result['shape']
        x_coords, y_coords = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        data_dict['x_position'] = x_coords.flatten()
        data_dict['y_position'] = y_coords.flatten()
        
        # Add weight information if available
        if 'weights' in interpolation_result:
            combined_weights = interpolation_result['weights']['combined']
            # Create weight columns for each source
            for i, weight in enumerate(combined_weights):
                data_dict[f'source_{i}_weight'] = np.full(len(data_dict['x_position']), weight)
        
        df = pd.DataFrame(data_dict)
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
    def export_visualization_as_png(self, fig, filename=None, dpi=300):
        """Export matplotlib figure as PNG"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"visualization_{timestamp}.png"
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        return buf.getvalue(), filename
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)

# =============================================
# MAIN APPLICATION WITH ENHANCED VISUALIZATION
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Enhanced Stress Field Interpolation with Advanced Visualization",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 2.0rem !important;
        color: #374151 !important;
        font-weight: 800 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 6px 6px 0 0;
        gap: 1.2rem;
        padding-top: 12px;
        padding-bottom: 12px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: 700;
    }
    .param-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #e9ecef;
    }
    .param-key {
        font-weight: 700;
        color: #1E3A8A;
        font-size: 1.1rem;
    }
    .param-value {
        font-weight: 600;
        color: #059669;
        font-size: 1.1rem;
    }
    .angular-weighting-plot {
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 15px;
        background-color: #F8FAFC;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Enhanced Stress Field Interpolation with Advanced Visualization</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Advanced physics-aware stress interpolation with comprehensive visualization.</strong><br>
    • <strong>CORRECTED:</strong> Eigenstrain values match source files (Twin: 2.12, ESF: 1.414, ISF: 0.707)<br>
    • <strong>ENHANCED:</strong> Full weight analysis (spatial, transformer attention, combined)<br>
    • <strong>INTERACTIVE:</strong> 3D surface plots and interactive heatmaps with hover information<br>
    • <strong>COMPREHENSIVE:</strong> Bracketing visualization for targeted interpolation<br>
    • <strong>PROFESSIONAL:</strong> Publication-quality visualizations with detailed styling<br>
    • <strong>COMPLETE:</strong> Export results in multiple formats with full metadata
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = TargetedSolutionLoader(SOLUTIONS_DIR)
    if 'dual_interpolator' not in st.session_state:
        st.session_state.dual_interpolator = DualInterpolationSystem()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = AdvancedHeatmapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'use_targeted_mode' not in st.session_state:
        st.session_state.use_targeted_mode = False
    if 'special_angle' not in st.session_state:
        st.session_state.special_angle = 54.7
    if 'use_special_angle' not in st.session_state:
        st.session_state.use_special_angle = True
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                    
                    # Show defect distribution
                    defect_counts = {}
                    for sol in st.session_state.solutions:
                        if 'params' in sol:
                            defect = sol['params'].get('defect_type', 'Unknown')
                            defect_counts[defect] = defect_counts.get(defect, 0) + 1
                    
                    st.markdown("**Loaded solutions by defect type:**")
                    for defect, count in defect_counts.items():
                        st.markdown(f"- **{defect}**: {count} sources")
                else:
                    st.warning("No solutions found in directory")
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.success("Cache cleared")
        
        st.divider()
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Interpolation mode
        interpolation_mode = st.radio(
            "Interpolation Mode",
            options=["🎯 Targeted Bracketing", "🧠 Full Transformer"],
            index=1,
            help="Targeted: Uses only two nearest same-defect sources\nFull: Uses all sources with transformer attention"
        )
        st.session_state.use_targeted_mode = (interpolation_mode == "🎯 Targeted Bracketing")
        
        # Special angle configuration
        st.markdown("#### 🎯 Special Angle Configuration")
        use_special_angle = st.checkbox("Enable Special Angle Bias", value=True,
                                        help="Enable bias toward special angle for better accuracy near important orientations")
        
        special_angle = st.slider(
            "Special Angle (degrees)",
            min_value=0.0,
            max_value=180.0,
            value=54.7,
            step=0.1,
            help="The 'special' angle to bias toward (default 54.7° habit plane). Can be disabled above.",
            disabled=not use_special_angle
        )
        st.session_state.special_angle = special_angle
        st.session_state.use_special_angle = use_special_angle
        
        # Custom polar angle
        custom_theta = st.slider(
            "Target Angle θ (degrees)",
            min_value=0.0,
            max_value=180.0,
            value=54.7,
            step=0.1,
            help="Angle for which to interpolate stress fields"
        )
        
        # Defect type
        defect_type = st.selectbox(
            "Target Defect Type",
            options=['ISF', 'ESF', 'Twin', 'No Defect'],
            index=2,
            help="Defect type for interpolation"
        )
        
        # Show available angles for selected defect
        if st.session_state.solutions:
            same_defect_angles = []
            for sol in st.session_state.solutions:
                if 'params' in sol and sol['params'].get('defect_type') == defect_type:
                    if 'theta' in sol['params']:
                        angle = np.degrees(sol['params']['theta'])
                        same_defect_angles.append(angle)
            
            if same_defect_angles:
                same_defect_angles.sort()
                st.info(f"Available {defect_type} angles: {', '.join([f'{a:.1f}°' for a in same_defect_angles])}")
        
        st.divider()
        
        # Material parameters with CORRECTED eigenstrains
        st.markdown('<h2 class="section-header">🧬 Material Properties</h2>', unsafe_allow_html=True)
        
        # Kappa parameter
        kappa = st.slider(
            "Kappa (κ)",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.01,
            help="Material stiffness parameter"
        )
        
        # CORRECTED Eigenstrain calculation
        st.markdown("#### 🧮 Eigenstrain Calculation")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            auto_eigen = st.checkbox("Auto-calculate eigenstrain", value=True)
        with col_e2:
            # CORRECTED eigenstrain values to match source files
            corrected_eigenstrain_map = {
                'ISF': 0.707,    # Corrected to match source files
                'ESF': 1.414,    # Corrected to match source files  
                'Twin': 2.12,    # Corrected to match source files
                'No Defect': 0.0
            }
            
            if auto_eigen:
                # Auto-calculate based on defect type using corrected values
                eigen_strain = corrected_eigenstrain_map[defect_type]
                st.metric("Eigenstrain ε₀", f"{eigen_strain:.3f}", 
                         delta=f"Corrected values")
            else:
                eigen_strain = st.slider(
                    "Eigenstrain ε₀",
                    min_value=0.0,
                    max_value=3.0,
                    value=corrected_eigenstrain_map[defect_type],  # Start with corrected default
                    step=0.001,
                    help="Corrected values: Twin=2.12, ESF=1.414, ISF=0.707"
                )
        
        # Shape
        shape = st.selectbox(
            "Shape",
            options=['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle'],
            index=0,
            help="Geometry of the defect region"
        )
        
        st.divider()
        
        # Transformer parameters
        st.markdown('<h2 class="section-header">🧠 Transformer Configuration</h2>', unsafe_allow_html=True)
        
        # Temperature parameter
        temperature = st.slider(
            "Attention Temperature",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Softmax temperature for attention weights (lower = sharper distribution)"
        )
        
        # Advanced parameters for targeted mode
        if st.session_state.use_targeted_mode:
            st.markdown("#### 🎯 Targeted Mode Parameters")
            bracketing_weight = st.slider(
                "Bracketing Source Weight",
                min_value=0.9,
                max_value=0.999,
                value=0.98,
                step=0.001,
                help="Total weight allocated to the two bracketing sources"
            )
            other_weight = st.slider(
                "Other Source Weight",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Weight allocated to non-bracketing sources (should be very small)"
            )
            
            # Update interpolator parameters
            st.session_state.dual_interpolator.targeted_interpolator.bracketing_weight = bracketing_weight
            st.session_state.dual_interpolator.targeted_interpolator.other_weight = other_weight
        
        st.divider()
        
        # Run interpolation
        st.markdown("#### 🚀 Interpolation Control")
        if st.button("🚀 Perform Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing interpolation with enhanced visualization..."):
                    # Setup target parameters with CORRECTED eigenstrains
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': eigen_strain,
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
                    }
                    
                    # Set interpolation mode
                    mode = 'targeted' if st.session_state.use_targeted_mode else 'full'
                    
                    # Perform interpolation
                    result = st.session_state.dual_interpolator.interpolate(
                        st.session_state.solutions,
                        custom_theta,
                        target_params,
                        mode=mode
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        st.session_state.target_angle = custom_theta
                        st.session_state.target_defect = defect_type
                        
                        # Show success message with bracketing info for targeted mode
                        if st.session_state.use_targeted_mode and 'weights' in result:
                            if 'bracketing_sources' in result['weights']:
                                bracketing = result['weights']['bracketing_sources']
                                lower = bracketing['lower']
                                upper = bracketing['upper']
                                st.markdown(f"""
                                <div class="info-box">
                                <strong>✅ Targeted Interpolation Successful!</strong><br>
                                • Used <strong>{lower['index']} ({lower['angle']:.1f}°)</strong> as lower bracket<br>
                                • Used <strong>{upper['index']} ({upper['angle']:.1f}°)</strong> as upper bracket<br>
                                • Bracketing weight: <strong>{(lower['weight'] + upper['weight'])*100:.1f}%</strong><br>
                                • Other sources weight: <strong>{(1 - lower['weight'] - upper['weight'])*100:.3f}%</strong>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("Interpolation completed with full transformer attention!")
                    else:
                        st.error("Interpolation failed. Check the console for errors.")
    
    # Main content area
    if st.session_state.solutions:
        st.markdown(f"### 📊 Loaded {len(st.session_state.solutions)} Solutions")
        
        # Display loaded solutions
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Loaded Files", len(st.session_state.solutions))
        with col_info2:
            if st.session_state.interpolation_result:
                st.metric("Interpolated Angle", f"{st.session_state.interpolation_result['target_angle']:.1f}°")
        with col_info3:
            if st.session_state.interpolation_result:
                st.metric("Grid Size", f"{st.session_state.interpolation_result['shape'][0]}×{st.session_state.interpolation_result['shape'][1]}")
        
        # Display source information
        if st.session_state.solutions:
            source_thetas = []
            source_defects = []
            for sol in st.session_state.solutions:
                if 'params' in sol and 'theta' in sol['params']:
                    theta_deg = np.degrees(sol['params']['theta']) % 360
                    source_thetas.append(theta_deg)
                if 'defect_type' in sol.get('params', {}):
                    source_defects.append(sol['params']['defect_type'])
            
            if source_thetas:
                st.markdown(f"**Source Angles Range:** {min(source_thetas):.1f}° to {max(source_thetas):.1f}°")
                st.markdown(f"**Mean Source Angle:** {np.mean(source_thetas):.1f}°")
                
                # Show defect type distribution
                if source_defects:
                    defect_counts = pd.Series(source_defects).value_counts()
                    st.markdown("**Source Defect Types:**")
                    for defect, count in defect_counts.items():
                        st.markdown(f"- **{defect}**: {count} sources")
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Results Overview",
            "🎨 Advanced Visualization", 
            "🧠 Weights Analysis",
            "🔄 Comparison Dashboard",
            "💾 Export Results"
        ])
        
        with tab1:
            # Results overview
            st.markdown('<h2 class="section-header">📊 Interpolation Results</h2>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Max Von Mises",
                    f"{result['statistics']['von_mises']['max']:.3f} GPa",
                    delta=f"±{result['statistics']['von_mises']['std']:.3f}"
                )
            with col2:
                st.metric(
                    "Hydrostatic Range",
                    f"{result['statistics']['sigma_hydro']['max_tension']:.3f}/{result['statistics']['sigma_hydro']['max_compression']:.3f} GPa"
                )
            with col3:
                st.metric(
                    "Mean Stress Magnitude",
                    f"{result['statistics']['sigma_mag']['mean']:.3f} GPa"
                )
            with col4:
                st.metric(
                    "Source Count",
                    result['num_sources'],
                    delta=f"Entropy: {result['weights']['entropy']['combined']:.3f}"
                )
            
            # Target parameters display
            st.markdown("#### 🎯 Target Parameters")
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.markdown(f"""
                <div class="param-table">
                <div class="param-key">Angle (θ)</div>
                <div class="param-value">{result['target_angle']:.2f}°</div>
                <div class="param-key">Defect Type</div>
                <div class="param-value">{result['target_params']['defect_type']}</div>
                </div>
                """, unsafe_allow_html=True)
            with param_col2:
                st.markdown(f"""
                <div class="param-table">
                <div class="param-key">Eigenstrain (ε₀)</div>
                <div class="param-value">{result['target_params']['eps0']:.3f}</div>
                <div class="param-key">Kappa (κ)</div>
                <div class="param-value">{result['target_params']['kappa']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with param_col3:
                st.markdown(f"""
                <div class="param-table">
                <div class="param-key">Shape</div>
                <div class="param-value">{result['target_params'].get('shape', 'Square')}</div>
                <div class="param-key">Grid Size</div>
                <div class="param-value">{result['shape'][0]}×{result['shape'][1]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Special angle configuration
            if st.session_state.use_special_angle:
                st.markdown(f"#### 🎯 Special Angle: {st.session_state.special_angle:.1f}°")
                angular_diff = abs(result['target_angle'] - st.session_state.special_angle)
                st.markdown(f"**Angular distance from special angle:** {angular_diff:.1f}°")
        
        with tab2:
            # Advanced visualization tab
            st.markdown('<h2 class="section-header">🎨 Advanced Visualization</h2>', unsafe_allow_html=True)
            
            # Visualization controls
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                component = st.selectbox(
                    "Stress Component",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0,
                    key="viz_component"
                )
            with col_viz2:
                cmap_name = st.selectbox(
                    "Colormap",
                    options=COLORMAP_OPTIONS['Publication Standard'],
                    index=0,
                    key="cmap_name"
                )
            
            # Show colormap preview
            if cmap_name:
                col_preview1, col_preview2, col_preview3 = st.columns([1, 2, 1])
                with col_preview2:
                    fig_cmap = st.session_state.visualizer.get_colormap_preview(cmap_name)
                    st.pyplot(fig_cmap)
            
            # Visualization type selection
            viz_type = st.radio(
                "Visualization Type",
                options=["2D Heatmap", "3D Surface", "Interactive Heatmap", "Interactive 3D"],
                horizontal=True
            )
            
            if component in result['fields']:
                stress_field = result['fields'][component]
                
                if viz_type == "2D Heatmap":
                    # 2D heatmap
                    fig_2d = st.session_state.visualizer.create_stress_heatmap(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        special_angle=st.session_state.special_angle if st.session_state.use_special_angle else None,
                        figsize=(12, 10)
                    )
                    st.pyplot(fig_2d)
                elif viz_type == "3D Surface":
                    # 3D surface plot
                    fig_3d = st.session_state.visualizer.create_3d_surface_plot(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        special_angle=st.session_state.special_angle if st.session_state.use_special_angle else None,
                        figsize=(14, 10)
                    )
                    st.pyplot(fig_3d)
                elif viz_type == "Interactive Heatmap":
                    # Interactive heatmap
                    fig_interactive = st.session_state.visualizer.create_interactive_heatmap(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        special_angle=st.session_state.special_angle if st.session_state.use_special_angle else None,
                        width=800,
                        height=700
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                elif viz_type == "Interactive 3D":
                    # Interactive 3D surface
                    fig_3d_interactive = st.session_state.visualizer.create_interactive_3d_surface(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        special_angle=st.session_state.special_angle if st.session_state.use_special_angle else None,
                        width=900,
                        height=700
                    )
                    st.plotly_chart(fig_3d_interactive, use_container_width=True)
        
        with tab3:
            # Weights analysis tab
            st.markdown('<h2 class="section-header">🧠 Weights Analysis</h2>', unsafe_allow_html=True)
            
            if 'weights' in result:
                weights = result['weights']
                
                # Weight statistics
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                with col_w1:
                    st.metric("Transformer Entropy", f"{weights['entropy']['transformer']:.3f}")
                with col_w2:
                    st.metric("Spatial Entropy", f"{weights['entropy']['spatial']:.3f}")
                with col_w3:
                    st.metric("Combined Entropy", f"{weights['entropy']['combined']:.3f}")
                with col_w4:
                    max_weight_idx = np.argmax(weights['combined'])
                    max_theta = result['source_theta_degrees'][max_weight_idx] if max_weight_idx < len(result['source_theta_degrees']) else 0.0
                    max_defect = result.get('source_defect_types', [result['target_params']['defect_type']])[max_weight_idx]
                    st.metric("Top Contributor", f"{max_defect} @ {max_theta:.1f}°")
                
                # Prepare weight data for visualization
                weights_data = {
                    'spatial': weights['positional'],
                    'attention': weights['transformer'],
                    'combined': weights['combined'],
                    'entropy_spatial': weights['entropy']['spatial'],
                    'entropy_attention': weights['entropy']['transformer'],
                    'entropy_combined': weights['entropy']['combined'],
                    'angles': result['source_theta_degrees'],
                    'distances': result.get('source_distances', [0] * len(weights['combined'])),
                    'target_angle': result['target_angle'],
                    'special_angle': st.session_state.special_angle if st.session_state.use_special_angle else None
                }
                
                # Static weights visualization
                st.markdown("#### 📊 Static Weights Visualization")
                fig_weights = st.session_state.visualizer.create_weights_visualization(weights_data, figsize=(15, 10))
                st.pyplot(fig_weights)
                
                # Interactive weights dashboard
                st.markdown("#### 🔄 Interactive Weights Dashboard")
                fig_interactive_weights = st.session_state.visualizer.create_interactive_weights_dashboard(weights_data)
                st.plotly_chart(fig_interactive_weights, use_container_width=True)
                
                # Bracketing visualization for targeted mode
                if st.session_state.use_targeted_mode and 'bracketing_sources' in weights:
                    st.markdown("#### 🎯 Bracketing Visualization")
                    bracketing = weights['bracketing_sources']
                    lower_angle = bracketing['lower']['angle']
                    upper_angle = bracketing['upper']['angle']
                    target_angle = result['target_angle']
                    target_defect = result['target_params']['defect_type']
                    
                    fig_bracketing = st.session_state.visualizer.create_bracketing_visualization(
                        lower_angle, upper_angle, target_angle, target_defect, figsize=(10, 8)
                    )
                    st.pyplot(fig_bracketing)
        
        with tab4:
            # COMPARISON DASHBOARD
            st.markdown('<h2 class="section-header">🔄 Comparison Dashboard</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Compare interpolated results with source solutions</strong><br>
            • Select any source solution as ground truth<br>
            • Visualize differences between interpolation and ground truth<br>
            • Analyze spatial correlation patterns<br>
            • Compare stress field profiles
            </div>
            """, unsafe_allow_html=True)
            
            # Ground truth selection
            st.markdown("#### 🎯 Select Ground Truth Source")
            if 'source_theta_degrees' in result and result['source_theta_degrees']:
                ground_truth_options = []
                for i, theta in enumerate(result['source_theta_degrees']):
                    defect_type = result.get('source_defect_types', [result['target_params']['defect_type']])[i]
                    weight = result['weights']['combined'][i]
                    ground_truth_options.append(
                        f"Source {i}: {defect_type} @ {theta:.1f}° (weight={weight:.3f})"
                    )
                
                selected_option = st.selectbox(
                    "Choose ground truth source:",
                    options=ground_truth_options,
                    index=0,
                    key="ground_truth_select"
                )
                
                # Parse selected index
                selected_index = int(selected_option.split(":")[0].split(" ")[1])
                
                # Display selected source info
                selected_theta = result['source_theta_degrees'][selected_index]
                selected_weight = result['weights']['combined'][selected_index]
                selected_defect = result.get('source_defect_types', [result['target_params']['defect_type']])[selected_index]
                
                col_gt1, col_gt2, col_gt3, col_gt4 = st.columns(4)
                with col_gt1:
                    st.metric("Selected Source", selected_index)
                with col_gt2:
                    st.metric("Source Angle", f"{selected_theta:.1f}°")
                with col_gt3:
                    st.metric("Source Defect", selected_defect)
                with col_gt4:
                    st.metric("Contribution Weight", f"{selected_weight:.3f}")
                
                # Create comparison dashboard
                if 'source_fields' in result and selected_index < len(result['source_fields']):
                    comparison_component = st.selectbox(
                        "Component for Comparison",
                        options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                        index=0,
                        key="comp_component"
                    )
                    
                    if comparison_component in result['fields']:
                        # Prepare source fields for comparison
                        source_fields = []
                        for fields in result['source_fields']:
                            if comparison_component in fields:
                                source_fields.append(fields[comparison_component])
                        
                        if selected_index < len(source_fields):
                            # Create comparison figure
                            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                            
                            # Interpolated result
                            im1 = axes[0].imshow(result['fields'][comparison_component], cmap='viridis')
                            axes[0].set_title('Interpolated Result', fontsize=14, fontweight='bold')
                            plt.colorbar(im1, ax=axes[0])
                            
                            # Ground truth
                            im2 = axes[1].imshow(source_fields[selected_index], cmap='viridis')
                            axes[1].set_title(f'Ground Truth\nSource {selected_index}: {selected_defect} @ {selected_theta:.1f}°', 
                                            fontsize=14, fontweight='bold')
                            plt.colorbar(im2, ax=axes[1])
                            
                            # Difference
                            diff = result['fields'][comparison_component] - source_fields[selected_index]
                            im3 = axes[2].imshow(diff, cmap='RdBu_r')
                            axes[2].set_title('Difference', fontsize=14, fontweight='bold')
                            plt.colorbar(im3, ax=axes[2])
                            
                            plt.suptitle(f'Comparison: {comparison_component.replace("_", " ").title()} Stress', 
                                        fontsize=16, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
        
        with tab5:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Export interpolation results for further analysis</strong><br>
            • Export full results as JSON with metadata<br>
            • Export stress field data as CSV for external analysis<br>
            • Download visualizations as PNG images<br>
            • Save comparison dashboard for publication<br>
            • Includes corrected eigenstrain values and weight analysis
            </div>
            """, unsafe_allow_html=True)
            
            # Export options
            export_format = st.radio(
                "Export Format",
                options=["JSON (Full Results)", "CSV (Field Data)", "PNG (Visualizations)"],
                horizontal=True
            )
            
            if export_format == "JSON (Full Results)":
                # JSON export
                visualization_params = {
                    'component': component if 'component' in locals() else 'von_mises',
                    'colormap': cmap_name if 'cmap_name' in locals() else 'viridis',
                    'visualization_type': viz_type if 'viz_type' in locals() else '2D Heatmap',
                    'special_angle': st.session_state.special_angle,
                    'use_special_angle': st.session_state.use_special_angle
                }
                export_data = st.session_state.results_manager.prepare_export_data(
                    result, visualization_params
                )
                json_str, json_filename = st.session_state.results_manager.export_to_json(export_data)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
                
                # Show preview
                with st.expander("🔍 JSON Preview", expanded=False):
                    st.json(export_data)
            elif export_format == "CSV (Field Data)":
                # CSV export
                csv_str, csv_filename = st.session_state.results_manager.export_to_csv(result)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Show preview
                with st.expander("🔍 CSV Preview", expanded=False):
                    # Create a sample of the data
                    sample_data = {}
                    for field_name, field_data in result['fields'].items():
                        sample_data[field_name] = field_data.flatten()[:100]  # First 100 values
                    df_sample = pd.DataFrame(sample_data)
                    st.dataframe(df_sample.head(10))
            elif export_format == "PNG (Visualizations)":
                # PNG export options
                st.markdown("#### 📸 Select Visualizations to Export")
                export_plots = st.multiselect(
                    "Choose plots to export:",
                    options=[
                        "Von Mises Heatmap", 
                        "Hydrostatic Heatmap", 
                        "Stress Magnitude Heatmap",
                        "3D Surface Plot", 
                        "Interactive Dashboard", 
                        "Weights Analysis", 
                        "Bracketing Visualization"
                    ],
                    default=["Von Mises Heatmap", "Weights Analysis"]
                )
                
                if st.button("🖼️ Generate and Download Visualizations", use_container_width=True):
                    # Create a zip file with all selected plots
                    import zipfile
                    from io import BytesIO
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        plot_count = 0
                        
                        if "Von Mises Heatmap" in export_plots:
                            fig = st.session_state.visualizer.create_stress_heatmap(
                                result['fields']['von_mises'],
                                title="Von Mises Stress",
                                cmap_name='viridis',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                special_angle=st.session_state.special_angle if st.session_state.use_special_angle else None
                            )
                            png_data, filename = st.session_state.results_manager.export_visualization_as_png(fig)
                            zip_file.writestr(f"von_mises_{result['target_angle']:.1f}.png", png_data)
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Weights Analysis" in export_plots and 'weights' in result:
                            weights = result['weights']
                            weights_data = {
                                'spatial': weights['positional'],
                                'attention': weights['transformer'],
                                'combined': weights['combined'],
                                'entropy_spatial': weights['entropy']['spatial'],
                                'entropy_attention': weights['entropy']['transformer'],
                                'entropy_combined': weights['entropy']['combined'],
                                'angles': result['source_theta_degrees'],
                                'distances': result.get('source_distances', [0] * len(weights['combined'])),
                                'target_angle': result['target_angle'],
                                'special_angle': st.session_state.special_angle if st.session_state.use_special_angle else None
                            }
                            fig = st.session_state.visualizer.create_weights_visualization(weights_data)
                            png_data, filename = st.session_state.results_manager.export_visualization_as_png(fig)
                            zip_file.writestr(f"weights_analysis_{result['target_angle']:.1f}.png", png_data)
                            plot_count += 1
                            plt.close(fig)
                    
                    zip_buffer.seek(0)
                    
                    # Download button for zip file
                    st.download_button(
                        label=f"📦 Download {plot_count} Visualization(s) as ZIP",
                        data=zip_buffer,
                        file_name=f"visualizations_theta_{result['target_angle']:.1f}_{result['target_params']['defect_type']}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    st.success(f"Generated {plot_count} visualization(s) for download.")
    
    # No results yet - show instructions
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); border-radius: 20px; color: white;">
        <h2>🚀 Ready to Begin!</h2>
        <p style="font-size: 1.2rem; margin-bottom: 30px;">
        Follow these steps to start interpolating stress fields with advanced visualization:
        </p>
        <ol style="text-align: left; display: inline-block; font-size: 1.1rem;">
        <li>Load simulation files from the sidebar</li>
        <li>Configure target parameters (angle, defect type, etc.)</li>
        <li>Adjust transformer and visualization parameters</li>
        <li>Click "Perform Interpolation"</li>
        <li>Explore advanced visualizations in the tabs above</li>
        </ol>
        <p style="margin-top: 30px; font-size: 1.1rem;">
        <strong>Key Features:</strong>
        <ul style="text-align: left; display: inline-block;">
        <li>✅ <strong>Corrected eigenstrain values</strong> (Twin: 2.12, ESF: 1.414, ISF: 0.707)</li>
        <li>✅ <strong>Advanced weight visualization</strong> (spatial, attention, combined)</li>
        <li>✅ <strong>Interactive 3D visualizations</strong> with hover information</li>
        <li>✅ <strong>Bracketing analysis</strong> for targeted interpolation</li>
        <li>✅ <strong>Comprehensive export options</strong> with full metadata</li>
        </ul>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 📚 Advanced Visualization Guide")
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    with col_guide1:
        st.markdown("""
        #### 🎨 Visualization Types
        1. **2D Heatmaps** with publication-quality styling
        2. **3D Surface Plots** for spatial stress distribution
        3. **Interactive Heatmaps** with hover information
        4. **Interactive 3D Surfaces** with rotation and zoom
        5. **Weight Distribution Analysis** for model transparency
        """)
    with col_guide2:
        st.markdown("""
        #### 🧠 Weight Analysis
        - **Spatial Weights**: Based on angular proximity and parameter similarity
        - **Attention Weights**: From transformer neural network
        - **Combined Weights**: Hybrid approach (configurable balance)
        - **Entropy Metrics**: Measure of weight distribution concentration
        - **Bracketing Visualization**: For targeted interpolation mode
        """)
    with col_guide3:
        st.markdown("""
        #### 📤 Export Options
        - **JSON**: Full results with metadata and stress field data
        - **CSV**: Flattened stress field data for external analysis
        - **PNG**: High-resolution visualizations (300+ DPI)
        - **ZIP**: Package of multiple visualizations
        - **Interactive Plots**: Save Plotly figures as HTML
        """)
    
    # Physics explanation
    with st.expander("🧬 Physics Background: Corrected Eigenstrains and Advanced Visualization", expanded=True):
        st.markdown("""
        **Key Corrections and Enhancements:**

        1. **Corrected Eigenstrain Values:**
           - Fixed to match original source files:
             - Twin boundary: **2.12** (previously 0.707)
             - ESF (Extrinsic Stacking Fault): **1.414** (previously 0.333) 
             - ISF (Intrinsic Stacking Fault): **0.707** (previously 0.289)
           - These values now correctly reflect the physical magnitudes for FCC crystal defects

        2. **Advanced Weight Visualization:**
           - **Spatial Weights**: Show influence of angular proximity and parameter similarity
           - **Attention Weights**: Visualize transformer's learned importance
           - **Combined Weights**: See the final blending of spatial and attention mechanisms
           - **Polar Plots**: Display angular distribution of source influence
           - **Entropy Analysis**: Quantify weight distribution concentration

        3. **Interactive 3D Visualization:**
           - **Interactive Heatmaps** with hover information for detailed inspection
           - **3D Surface Plots** that can be rotated, zoomed, and panned
           - **Dynamic Color Maps** that adapt to data range
           - **Real-time Statistics** display on hover

        4. **Bracketing Analysis:**
           - For targeted interpolation mode, visualize the bracketing relationship
           - See exactly which sources are used and their relative weights
           - Understand why certain sources dominate the interpolation

        5. **Publication Quality:**
           - All visualizations follow publication standards
           - Consistent color schemes and labeling
           - Proper axis labels and units
           - High-resolution (300+ DPI) export options
           - Comprehensive metadata for reproducibility

        This implementation ensures that both the physics and visualization are state-of-the-art, providing researchers with the tools needed for rigorous analysis and clear communication of results.
        """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
```
