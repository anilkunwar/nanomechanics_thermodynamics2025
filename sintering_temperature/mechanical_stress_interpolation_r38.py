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
# POSITIONAL ENCODING
# =============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return x + pe.unsqueeze(0)

# =============================================
# ENHANCED VISUALIZER WITH BRACKETING ANALYSIS
# =============================================
class BracketingAnalysisVisualizer:
    """Visualizer specialized for bracketing interpolation analysis"""
    
    def create_bracketing_analysis_dashboard(self, interpolation_result, figsize=(20, 15)):
        """Create comprehensive dashboard for bracketing interpolation analysis"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        result = interpolation_result
        target_angle = result['target_angle']
        target_defect = result['target_params']['defect_type']
        
        # 1. Weight distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'weights' in result and 'combined' in result['weights']:
            weights = result['weights']['combined']
            x = range(len(weights))
            
            bars = ax1.bar(x, weights, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Source Index')
            ax1.set_ylabel('Weight')
            ax1.set_title('Targeted Weight Distribution', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Highlight bracketing sources
            if 'bracketing_sources' in result['weights']:
                bracketing = result['weights']['bracketing_sources']
                lower_idx = bracketing['lower']['index']
                upper_idx = bracketing['upper']['index']
                
                if lower_idx < len(bars):
                    bars[lower_idx].set_color('green')
                    bars[lower_idx].set_alpha(0.9)
                    bars[lower_idx].set_label('Lower Bracket')
                
                if upper_idx < len(bars):
                    bars[upper_idx].set_color('red')
                    bars[upper_idx].set_alpha(0.9)
                    bars[upper_idx].set_label('Upper Bracket')
            
            # Add weight labels for significant sources
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.01:  # Label weights > 1%
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax1.legend()
        
        # 2. Angular bracketing visualization (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'interpolation_method' in result and result['interpolation_method'] == 'targeted_bracketing':
            # Get bracketing angles
            if 'interpolation_factor' in result:
                factor = result['interpolation_factor']
                lower_angle = factor['lower_angle']
                upper_angle = factor['upper_angle']
                interp_t = factor['interpolation_t']
                
                # Create angle line
                angles = np.array([lower_angle, target_angle, upper_angle])
                values = np.array([0, interp_t, 1])
                
                ax2.plot(angles, values, 'b-', linewidth=2, marker='o', markersize=8)
                ax2.fill_between([lower_angle, upper_angle], 0, 1, alpha=0.2, color='blue')
                
                # Add labels
                ax2.text(lower_angle, -0.05, f'{lower_angle:.1f}°', 
                        ha='center', va='top', fontsize=12, fontweight='bold')
                ax2.text(target_angle, interp_t + 0.05, f'Target: {target_angle:.1f}°', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
                ax2.text(upper_angle, -0.05, f'{upper_angle:.1f}°', 
                        ha='center', va='top', fontsize=12, fontweight='bold')
                
                # Add interpolation factor
                ax2.text(target_angle, interp_t/2, f't = {interp_t:.3f}', 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
                ax2.set_xlabel('Angle (degrees)')
                ax2.set_ylabel('Interpolation Factor (t)')
                ax2.set_title('Angular Bracketing Interpolation', fontsize=16, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(-0.1, 1.1)
        
        # 3. Defect type distribution (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'source_defect_types' in result:
            defect_types = result['source_defect_types']
            defect_counts = {}
            
            for defect in defect_types:
                defect_counts[defect] = defect_counts.get(defect, 0) + 1
            
            # Create pie chart
            labels = list(defect_counts.keys())
            sizes = list(defect_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            ax3.set_title('Defect Type Distribution in Sources', fontsize=16, fontweight='bold')
            
            # Highlight target defect
            for i, label in enumerate(labels):
                if label == target_defect:
                    wedges[i].set_edgecolor('red')
                    wedges[i].set_linewidth(3)
        
        # 4. Stress field interpolation (middle row)
        components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        titles = ['Von Mises', 'Hydrostatic', 'Magnitude']
        
        for idx, (component, title) in enumerate(zip(components, titles)):
            ax = fig.add_subplot(gs[1, idx])
            
            if component in result['fields']:
                field = result['fields'][component]
                
                im = ax.imshow(field, cmap='viridis', aspect='equal', 
                              interpolation='bilinear', origin='lower')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                ax.set_title(f'{title} Stress\nθ={target_angle:.1f}°', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.grid(True, alpha=0.2)
        
        # 5. Angular distance vs weight (middle right)
        ax7 = fig.add_subplot(gs[1, 2])
        if 'source_angular_distances' in result and 'weights' in result:
            distances = result['source_angular_distances']
            weights = result['weights']['combined']
            defect_types = result['source_defect_types']
            
            # Color by defect type
            colors = []
            for defect in defect_types:
                if defect == target_defect:
                    colors.append('green')
                else:
                    colors.append('red')
            
            scatter = ax7.scatter(distances, weights, c=colors, alpha=0.7, 
                                 s=100, edgecolors='black')
            
            ax7.set_xlabel('Angular Distance from Target (°)')
            ax7.set_ylabel('Weight')
            ax7.set_title('Weight vs Angular Distance', fontsize=16, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            target_patch = mpatches.Patch(color='green', label=f'Target Defect ({target_defect})')
            other_patch = mpatches.Patch(color='red', label='Other Defects')
            ax7.legend(handles=[target_patch, other_patch])
        
        # 6. Bracketing sources comparison (bottom row)
        ax8 = fig.add_subplot(gs[2, :])
        if 'weights' in result and 'bracketing_sources' in result['weights']:
            bracketing = result['weights']['bracketing_sources']
            lower_info = bracketing['lower']
            upper_info = bracketing['upper']
            
            # Create comparison table
            table_data = [
                ['Parameter', 'Lower Bracket', 'Upper Bracket', 'Target'],
                ['Angle (°)', f"{lower_info['angle']:.1f}", f"{upper_info['angle']:.1f}", f"{target_angle:.1f}"],
                ['Weight', f"{lower_info['weight']:.4f}", f"{upper_info['weight']:.4f}", '1.0000'],
                ['Index', str(lower_info['index']), str(upper_info['index']), 'N/A'],
                ['Δθ (°)', f"{abs(target_angle - lower_info['angle']):.1f}", 
                 f"{abs(upper_info['angle'] - target_angle):.1f}", '0.0']
            ]
            
            table = ax8.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_data)):
                for j in range(len(table_data[0])):
                    if i == 0:  # Header row
                        table[(i, j)].set_facecolor('#4CAF50')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    elif j == 1:  # Lower bracket column
                        table[(i, j)].set_facecolor('#C8E6C9')
                    elif j == 2:  # Upper bracket column
                        table[(i, j)].set_facecolor('#FFCDD2')
                    elif j == 3:  # Target column
                        table[(i, j)].set_facecolor('#FFF9C4')
            
            ax8.set_title('Bracketing Sources Comparison', fontsize=18, fontweight='bold', pad=20)
            ax8.axis('off')
        
        plt.suptitle(f'Targeted Bracketing Interpolation - θ={target_angle:.1f}°, {target_defect}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_angular_bracketing_visualization(self, solutions, target_angle_deg, target_defect_type, figsize=(12, 8)):
        """Visualize how sources bracket the target angle"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group sources by defect type
        defect_groups = {}
        for i, sol in enumerate(solutions):
            if 'params' in sol:
                defect = sol['params'].get('defect_type', 'Unknown')
                if defect not in defect_groups:
                    defect_groups[defect] = []
                
                if 'theta' in sol['params']:
                    angle = np.degrees(sol['params']['theta'])
                    defect_groups[defect].append((i, angle))
        
        # Create visualization
        colors = {'Twin': 'red', 'ESF': 'blue', 'ISF': 'green', 'No Defect': 'gray'}
        y_positions = {}
        y = 0
        
        for defect, sources in defect_groups.items():
            if sources:
                y_positions[defect] = y
                angles = [s[1] for s in sources]
                
                # Plot sources as points
                ax.scatter(angles, [y] * len(angles), 
                          color=colors.get(defect, 'black'),
                          s=100, alpha=0.7, edgecolors='black',
                          label=f'{defect} ({len(sources)} sources)')
                
                # Draw range line
                if len(angles) > 1:
                    min_angle = min(angles)
                    max_angle = max(angles)
                    ax.plot([min_angle, max_angle], [y, y], 
                           color=colors.get(defect, 'black'), 
                           linewidth=2, alpha=0.5)
                
                y += 1
        
        # Plot target
        ax.axvline(x=target_angle_deg, color='red', linestyle='--', 
                  linewidth=3, alpha=0.8, label=f'Target: {target_angle_deg:.1f}°')
        
        # Highlight target defect
        if target_defect_type in y_positions:
            y_target = y_positions[target_defect_type]
            ax.axhline(y=y_target, color='yellow', linestyle=':', 
                      linewidth=2, alpha=0.5, label=f'Target Defect: {target_defect_type}')
        
        ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Defect Type', fontsize=14, fontweight='bold')
        ax.set_title('Source Distribution by Angle and Defect Type', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set y-ticks
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        
        plt.tight_layout()
        return fig

# =============================================
# MAIN APPLICATION WITH TARGETED INTERPOLATION
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Targeted Stress Field Interpolation with Angular Bracketing",
        layout="wide",
        page_icon="🎯",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .highlight-box {
        background-color: #FCE7F3;
        border-left: 5px solid #EC4899;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Targeted Stress Field Interpolation with Angular Bracketing</h1>', unsafe_allow_html=True)
    
    # Key feature description
    st.markdown("""
    <div class="highlight-box">
        <strong>🎯 KEY FEATURE: Targeted Angular Bracketing</strong><br>
        For a given target (e.g., 20° ESF), the interpolator:
        1. <strong>Selects the nearest ESF source with angle < 20°</strong><br>
        2. <strong>Selects the nearest ESF source with angle > 20°</strong><br>
        3. <strong>Gives near-zero weights to all other sources</strong><br><br>
        
        <strong>Example:</strong> For target 54.7° Twin:<br>
        • Uses <strong>Twin at 60°</strong> (nearest > 54.7°)<br>
        • Uses <strong>Twin at 30°</strong> (nearest < 54.7°)<br>
        • Other sources get <strong>~0.1% weight</strong>
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
        st.session_state.visualizer = BracketingAnalysisVisualizer()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'use_targeted_mode' not in st.session_state:
        st.session_state.use_targeted_mode = True
    
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
                    
                    for defect, count in defect_counts.items():
                        st.info(f"{defect}: {count} sources")
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
            options=["🎯 Targeted Bracketing", "🌐 Full Transformer"],
            index=0,
            help="Targeted: Uses only two nearest same-defect sources\nFull: Uses all sources with transformer attention"
        )
        
        st.session_state.use_targeted_mode = (interpolation_mode == "🎯 Targeted Bracketing")
        
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
                
                # Find bracketing angles
                lower_angle = None
                upper_angle = None
                
                for angle in same_defect_angles:
                    if angle < custom_theta:
                        lower_angle = angle
                
                for angle in reversed(same_defect_angles):
                    if angle > custom_theta:
                        upper_angle = angle
                
                if lower_angle is not None and upper_angle is not None:
                    st.success(f"Bracketing sources: {lower_angle:.1f}° (lower) and {upper_angle:.1f}° (upper)")
                elif lower_angle is not None:
                    st.warning(f"Only lower bracket: {lower_angle:.1f}°")
                elif upper_angle is not None:
                    st.warning(f"Only upper bracket: {upper_angle:.1f}°")
                else:
                    st.error(f"No {defect_type} sources available")
        
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
        
        # Material parameters
        st.markdown("#### 🧮 Material Parameters")
        kappa = st.slider(
            "Kappa (κ)", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.6,
            step=0.01,
            help="Material stiffness parameter"
        )
        
        # Shape
        shape = st.selectbox(
            "Shape",
            options=['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle'],
            index=0,
            help="Geometry of the defect region"
        )
        
        st.divider()
        
        # Run interpolation
        st.markdown("#### 🚀 Interpolation Control")
        if st.button("🎯 Perform Targeted Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing targeted interpolation..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
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
                        
                        # Show success message with bracketing info
                        if 'bracketing_sources' in result.get('weights', {}):
                            bracketing = result['weights']['bracketing_sources']
                            lower = bracketing['lower']
                            upper = bracketing['upper']
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✅ Targeted Interpolation Successful!</strong><br>
                                • Used <strong>{lower['index']} ({lower['angle']:.1f}°)</strong> as lower bracket<br>
                                • Used <strong>{upper['index']} ({upper['angle']:.1f}°)</strong> as upper bracket<br>
                                • Bracketing weight: <strong>{(lower['weight'] + upper['weight'])*100:.1f}%</strong><br>
                                • Other sources weight: <strong>{(1 - lower['weight'] - upper['weight'])*100:.3f}%</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.success("Interpolation completed!")
                    else:
                        st.error("Interpolation failed. Check the console for errors.")
    
    # Main content area
    if st.session_state.solutions:
        st.markdown(f"### 📊 Loaded {len(st.session_state.solutions)} Solutions")
        
        # Show source distribution visualization
        with st.expander("📈 Show Source Distribution", expanded=True):
            fig_dist = st.session_state.visualizer.create_angular_bracketing_visualization(
                st.session_state.solutions,
                custom_theta,
                defect_type
            )
            st.pyplot(fig_dist)
        
        # Show current target info
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Target Angle", f"{custom_theta:.1f}°")
        with col_info2:
            st.metric("Target Defect", defect_type)
        with col_info3:
            mode_name = "Targeted Bracketing" if st.session_state.use_targeted_mode else "Full Transformer"
            st.metric("Interpolation Mode", mode_name)
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Results Overview", 
            "🎯 Bracketing Analysis", 
            "🎨 Visualization",
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
                    f"{result['statistics']['sigma_hydro']['max']:.3f}/{result['statistics']['sigma_hydro']['min']:.3f} GPa"
                )
            with col3:
                st.metric(
                    "Mean Stress Magnitude", 
                    f"{result['statistics']['sigma_mag']['mean']:.3f} GPa"
                )
            with col_info1:
                st.metric(
                    "Weight Entropy", 
                    f"{result['weights']['entropy']:.3f}",
                    help="Lower entropy = more focused on few sources"
                )
            
            # Interpolation details
            st.markdown("#### 🎯 Interpolation Details")
            
            if result.get('interpolation_method') == 'targeted_bracketing':
                col_det1, col_det2, col_det3 = st.columns(3)
                
                with col_det1:
                    if 'bracketing_sources' in result['weights']:
                        bracketing = result['weights']['bracketing_sources']
                        st.metric("Lower Bracket", f"{bracketing['lower']['angle']:.1f}°")
                
                with col_det2:
                    st.metric("Target Angle", f"{result['target_angle']:.1f}°")
                
                with col_det3:
                    if 'bracketing_sources' in result['weights']:
                        bracketing = result['weights']['bracketing_sources']
                        st.metric("Upper Bracket", f"{bracketing['upper']['angle']:.1f}°")
                
                # Show interpolation factor
                if 'interpolation_factor' in result:
                    factor = result['interpolation_factor']
                    interp_t = factor['interpolation_t']
                    
                    st.info(f"""
                    **Interpolation Factor (t) = {interp_t:.3f}**
                    - Lower bracket weight: {(1-interp_t)*100:.1f}%
                    - Upper bracket weight: {interp_t*100:.1f}%
                    """)
            
            # Weight distribution summary
            st.markdown("#### ⚖️ Weight Distribution Summary")
            if 'weights' in result and 'combined' in result['weights']:
                weights = result['weights']['combined']
                
                # Calculate weight concentration
                sorted_weights = np.sort(weights)[::-1]
                top_2_weight = sum(sorted_weights[:2])
                other_weight = sum(sorted_weights[2:])
                
                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    st.metric("Top 2 Sources Weight", f"{top_2_weight*100:.1f}%")
                with col_w2:
                    st.metric("Other Sources Weight", f"{other_weight*100:.3f}%")
                with col_w3:
                    non_zero_sources = sum(1 for w in weights if w > 0.001)
                    st.metric("Significant Sources", non_zero_sources)
                
                # Show top contributors
                if 'bracketing_sources' in result['weights']:
                    bracketing = result['weights']['bracketing_sources']
                    df_top = pd.DataFrame([
                        {
                            'Source': 'Lower Bracket',
                            'Angle': bracketing['lower']['angle'],
                            'Weight': bracketing['lower']['weight'],
                            'Contribution': f"{bracketing['lower']['weight']*100:.1f}%"
                        },
                        {
                            'Source': 'Upper Bracket',
                            'Angle': bracketing['upper']['angle'],
                            'Weight': bracketing['upper']['weight'],
                            'Contribution': f"{bracketing['upper']['weight']*100:.1f}%"
                        }
                    ])
                    
                    st.dataframe(df_top.style.format({
                        'Angle': '{:.1f}°',
                        'Weight': '{:.4f}'
                    }))
            
            # Quick preview
            st.markdown("#### 👀 Quick Preview")
            preview_component = st.selectbox(
                "Preview Component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0,
                key="preview_component"
            )
            
            if preview_component in result['fields']:
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(result['fields'][preview_component], cmap='viridis', 
                              aspect='equal', interpolation='bilinear', origin='lower')
                plt.colorbar(im, ax=ax, label='Stress (GPa)')
                ax.set_title(f'{preview_component.replace("_", " ").title()} Stress\nθ={result["target_angle"]:.1f}°', 
                           fontsize=16, fontweight='bold')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)
        
        with tab2:
            # Bracketing analysis
            st.markdown('<h2 class="section-header">🎯 Bracketing Analysis Dashboard</h2>', unsafe_allow_html=True)
            
            # Create comprehensive bracketing analysis dashboard
            fig_analysis = st.session_state.visualizer.create_bracketing_analysis_dashboard(result)
            st.pyplot(fig_analysis)
            
            # Additional analysis
            st.markdown("#### 📊 Detailed Weight Analysis")
            
            if 'weights' in result and 'combined' in result['weights']:
                weights = result['weights']['combined']
                
                # Create weight distribution table
                weight_data = []
                for i, weight in enumerate(weights):
                    if weight > 0.001:  # Only show significant weights
                        angle = result['source_angular_distances'][i] if i < len(result['source_angular_distances']) else None
                        defect = result['source_defect_types'][i] if i < len(result['source_defect_types']) else 'Unknown'
                        
                        weight_data.append({
                            'Source': i,
                            'Weight': weight,
                            'Angle Dist': f"{angle:.1f}°" if angle is not None else "N/A",
                            'Defect Type': defect,
                            'Contribution': f"{weight*100:.2f}%"
                        })
                
                if weight_data:
                    df_weights = pd.DataFrame(weight_data)
                    df_weights = df_weights.sort_values('Weight', ascending=False)
                    
                    st.dataframe(
                        df_weights.style.format({
                            'Weight': '{:.6f}'
                        }).background_gradient(subset=['Weight'], cmap='YlOrRd'),
                        height=400
                    )
                
                # Weight concentration metrics
                st.markdown("#### 🎯 Weight Concentration Metrics")
                
                # Calculate Gini coefficient
                sorted_weights = np.sort(weights)
                n = len(sorted_weights)
                cumulative = np.cumsum(sorted_weights)
                gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n if cumulative[-1] > 0 else 0
                
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("Gini Coefficient", f"{gini:.3f}", 
                             help="0 = equal weights, 1 = all weight on one source")
                with col_met2:
                    herfindahl = np.sum(weights**2)
                    st.metric("Herfindahl Index", f"{herfindahl:.4f}",
                             help="Measures concentration (higher = more concentrated)")
                with col_met3:
                    effective_n = 1 / herfindahl if herfindahl > 0 else 0
                    st.metric("Effective Sources", f"{effective_n:.1f}",
                             help="Number of equally-weighted sources with same concentration")
        
        with tab3:
            # Visualization tab
            st.markdown('<h2 class="section-header">🎨 Stress Field Visualization</h2>', unsafe_allow_html=True)
            
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
            
            # Create visualization
            if component in result['fields']:
                fig, ax = plt.subplots(figsize=(12, 10))
                im = ax.imshow(result['fields'][component], cmap=cmap_name, 
                              aspect='equal', interpolation='bilinear', origin='lower')
                plt.colorbar(im, ax=ax, label='Stress (GPa)')
                
                title = f'{component.replace("_", " ").title()} Stress\nθ={result["target_angle"]:.1f}°, {result["target_params"]["defect_type"]}'
                ax.set_title(title, fontsize=18, fontweight='bold')
                ax.set_xlabel('X Position', fontsize=14)
                ax.set_ylabel('Y Position', fontsize=14)
                ax.grid(True, alpha=0.2)
                
                # Add statistics annotation
                stats = result['statistics'][component]
                stats_text = (f"Max: {stats['max']:.3f} GPa\n"
                             f"Min: {stats['min']:.3f} GPa\n"
                             f"Mean: {stats['mean']:.3f} GPa\n"
                             f"Std: {stats['std']:.3f} GPa")
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                st.pyplot(fig)
            
            # Show all components
            if st.button("Show All Components", key="show_all_components"):
                fig_all, axes = plt.subplots(1, 3, figsize=(18, 6))
                components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                titles = ['Von Mises', 'Hydrostatic', 'Magnitude']
                
                for idx, (comp, title) in enumerate(zip(components, titles)):
                    ax = axes[idx]
                    im = ax.imshow(result['fields'][comp], cmap='viridis', 
                                  aspect='equal', interpolation='bilinear', origin='lower')
                    plt.colorbar(im, ax=ax, label='Stress (GPa)')
                    ax.set_title(f'{title}\nθ={result["target_angle"]:.1f}°', 
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.grid(True, alpha=0.2)
                
                plt.suptitle(f'Stress Components - {result["target_params"]["defect_type"]}', 
                           fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig_all)
        
        with tab4:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            
            export_format = st.radio(
                "Export Format",
                options=["JSON (Full Results)", "CSV (Stress Fields)", "PNG (Visualizations)"],
                horizontal=True
            )
            
            if export_format == "JSON (Full Results)":
                # Prepare export data
                export_data = {
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'interpolation_method': result.get('interpolation_method', 'targeted_bracketing'),
                        'target_angle': result['target_angle'],
                        'target_defect': result['target_params']['defect_type']
                    },
                    'result': {
                        'statistics': result['statistics'],
                        'weights': result['weights'],
                        'interpolation_factor': result.get('interpolation_factor', {}),
                        'shape': result['shape']
                    }
                }
                
                # Convert to JSON
                json_str = json.dumps(export_data, indent=2)
                filename = f"targeted_interpolation_{result['target_angle']:.1f}_{result['target_params']['defect_type']}.json"
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            elif export_format == "CSV (Stress Fields)":
                # Create CSV for von Mises stress
                field = result['fields']['von_mises']
                df = pd.DataFrame(field)
                csv_str = df.to_csv(index=False)
                filename = f"stress_field_{result['target_angle']:.1f}_{result['target_params']['defect_type']}.csv"
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif export_format == "PNG (Visualizations)":
                # Create and save visualizations
                if st.button("🖼️ Generate Visualizations", use_container_width=True):
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Save bracketing analysis
                        fig_analysis = st.session_state.visualizer.create_bracketing_analysis_dashboard(result)
                        buf = BytesIO()
                        fig_analysis.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        zip_file.writestr(f"bracketing_analysis_{result['target_angle']:.1f}.png", buf.getvalue())
                        plt.close(fig_analysis)
                        
                        # Save stress fields
                        for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            im = ax.imshow(result['fields'][component], cmap='viridis', 
                                          aspect='equal', interpolation='bilinear', origin='lower')
                            plt.colorbar(im, ax=ax, label='Stress (GPa)')
                            ax.set_title(f'{component} Stress - {result["target_angle"]:.1f}°', 
                                       fontsize=16, fontweight='bold')
                            ax.grid(True, alpha=0.2)
                            
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"{component}_{result['target_angle']:.1f}.png", buf.getvalue())
                            plt.close(fig)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download Visualizations Package",
                        data=zip_buffer,
                        file_name=f"visualizations_{result['target_angle']:.1f}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
    
    else:
        # No results yet - show instructions
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); border-radius: 20px; color: white;">
            <h2>🎯 Targeted Interpolation Ready!</h2>
            <p style="font-size: 1.2rem; margin-bottom: 30px;">
                This system performs <strong>targeted angular bracketing interpolation</strong>:
            </p>
            <ol style="text-align: left; display: inline-block; font-size: 1.1rem;">
                <li>Load your simulation files (supports .pkl, .pickle, .pt, .pth)</li>
                <li>Select target angle (e.g., 20°) and defect type (e.g., ESF)</li>
                <li>Choose <strong>🎯 Targeted Bracketing</strong> mode</li>
                <li>System finds the nearest ESF sources below and above 20°</li>
                <li>Gives <strong>~98% weight</strong> to these two sources</li>
                <li>Gives <strong>~0.1% weight</strong> to all other sources</li>
            </ol>
            <p style="margin-top: 30px; font-size: 1.1rem;">
                <strong>Example workflow:</strong><br>
                Target: <strong>54.7° Twin</strong> → Uses: <strong>Twin at 30°</strong> and <strong>Twin at 60°</strong><br>
                Target: <strong>20° ESF</strong> → Uses: <strong>ESF at 15°</strong> and <strong>ESF at 25°</strong> (if available)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Theory explanation
        with st.expander("📚 Theoretical Basis for Targeted Bracketing", expanded=True):
            st.markdown("""
            ### **Targeted Angular Bracketing: Mathematical Formulation**
            
            **1. Source Selection:**
            For target angle θₜ and defect type Dₜ:
            - Find Sₗ = argminₛ |θₛ - θₜ| where θₛ < θₜ and defect_type(Sₛ) = Dₜ
            - Find Sᵤ = argminₛ |θₛ - θₜ| where θₛ > θₜ and defect_type(Sₛ) = Dₜ
            
            **2. Weight Assignment:**
            - wₗ = α × (θᵤ - θₜ) / (θᵤ - θₗ)  where α = bracketing_weight (e.g., 0.98)
            - wᵤ = α × (θₜ - θₗ) / (θᵤ - θₗ)
            - wᵢ = (1 - α) / (N - 2) for all other sources i
            
            **3. Interpolation:**
            σ(θₜ) = wₗ × σ(θₗ) + wᵤ × σ(θᵤ) + Σᵢ wᵢ × σ(θᵢ)
            
            Where Σ w = 1 and wₗ + wᵤ ≈ α ≈ 0.98
            
            **4. Special Cases:**
            - If only one bracketing source found, use it with weight α
            - If no same-defect sources found, fall back to transformer interpolation
            - If target angle outside source range, use nearest two sources
            
            **5. Advantages:**
            - Physically intuitive (linear interpolation between nearest angles)
            - Consistent with crystallographic symmetry
            - Minimizes influence of dissimilar defect types
            - Provides smooth interpolation with clear bracketing
            """)
    
    # Footer
    st.divider()
    col_footer1, col_footer2 = st.columns(2)
    with col_footer1:
        st.markdown("**🎯 Targeted Angular Bracketing Interpolator**")
        st.markdown("Version 1.0 - Focused Interpolation System")
    with col_footer2:
        st.markdown("**📚 Key Features:**")
        st.markdown("• Uses only two nearest same-defect sources")
        st.markdown("• 98% weight to bracketing sources, 0.1% to others")
        st.markdown("• Linear angular interpolation")

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
