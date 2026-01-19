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
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

# =============================================
# GLOBAL STYLING CONFIGURATION
# =============================================
# Publication quality styling
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

# Enhanced colormap options with publication standards
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
# ENHANCED SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    """Enhanced solution loader with physics-aware processing"""
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        """Scan directory for solution files"""
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth']:
            import glob
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Sort by modification time (newest first)
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
        """Read simulation file with physics-aware processing"""
        try:
            with open(file_path, 'rb') as f:
                if format_type == 'pt' or file_path.endswith(('.pt', '.pth')):
                    # PyTorch file
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                else:
                    # Pickle file
                    data = pickle.load(f)
            
            # Standardize data structure
            standardized = self._standardize_data(data, file_path)
            return standardized
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data, file_path):
        """Standardize simulation data with physics metadata"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False
            }
        }
        
        try:
            if isinstance(data, dict):
                # Extract parameters
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                
                # Extract history
                if 'history' in data:
                    history = data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        # Convert dict to list
                        history_list = []
                        for key in sorted(history.keys()):
                            if isinstance(history[key], dict):
                                history_list.append(history[key])
                        standardized['history'] = history_list
                
                # Extract additional metadata
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                
                # Convert tensors to numpy arrays
                self._convert_tensors(standardized)
                
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
        
        return standardized
    
    def _convert_tensors(self, data):
        """Convert PyTorch tensors to numpy arrays recursively"""
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
    
    def load_all_solutions(self, use_cache=True, max_files=None):
        """Load all solutions with physics processing"""
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
# TRANSFORMER SPATIAL INTERPOLATOR WITH ENHANCED SPATIAL LOCALITY
# =============================================
class TransformerSpatialInterpolator:
    """Transformer-inspired stress interpolator with spatial locality regularization and adjustable weight factor"""
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0, locality_weight_factor=0.7):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor  # NEW: Control factor for spatial vs attention weights
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection - FIXED: Now expects exactly 15 input features
        self.input_proj = nn.Linear(15, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
    
    def set_spatial_parameters(self, spatial_sigma=None, locality_weight_factor=None):
        """Update spatial parameters dynamically"""
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
        if locality_weight_factor is not None:
            self.locality_weight_factor = locality_weight_factor
    
    def debug_feature_dimensions(self, params_list, target_angle_deg):
        """Debug method to check feature dimensions"""
        encoded = self.encode_parameters(params_list, target_angle_deg)
        print(f"Debug: Encoded shape: {encoded.shape}")
        print(f"Debug: Number of features: {encoded.shape[1]}")
        
        # Print first encoded vector
        if len(params_list) > 0:
            print(f"Debug: First encoded vector: {encoded[0]}")
            print(f"Debug: Number of non-zero elements: {torch.sum(encoded[0] != 0).item()}")
        
        return encoded.shape
    
    def compute_positional_weights(self, source_params, target_params):
        """Compute spatial locality weights based on parameter similarity with distance decay"""
        weights = []
        for src in source_params:
            # Compute parameter distance
            param_dist = 0.0
            
            # Compare key parameters
            key_params = ['eps0', 'kappa', 'theta', 'defect_type']
            for param in key_params:
                if param in src and param in target_params:
                    if param == 'defect_type':
                        # Categorical similarity
                        param_dist += 0.0 if src[param] == target_params[param] else 1.0
                    elif param == 'theta':
                        # Angular distance (cyclic)
                        src_theta = src.get(param, 0.0)
                        tgt_theta = target_params.get(param, 0.0)
                        diff = abs(src_theta - tgt_theta)
                        diff = min(diff, 2*np.pi - diff)  # Handle periodicity
                        param_dist += diff / np.pi
                    else:
                        # Normalized Euclidean distance
                        max_val = {'eps0': 3.0, 'kappa': 2.0}.get(param, 1.0)
                        param_dist += abs(src.get(param, 0) - target_params.get(param, 0)) / max_val
            
            # Apply Gaussian kernel with spatial_sigma controlling the decay rate
            weight = np.exp(-0.5 * (param_dist / self.spatial_sigma) ** 2)
            weights.append(weight)
        
        return np.array(weights)
    
    def encode_parameters(self, params_list, target_angle_deg):
        """Encode parameters into transformer input - FIXED to return exactly 15 features"""
        encoded = []
        for params in params_list:
            # Create feature vector
            features = []
            
            # Numeric features (3 features)
            features.append(params.get('eps0', 0.707) / 3.0)
            features.append(params.get('kappa', 0.6) / 2.0)
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)
            
            # One-hot encoding for defect type (4 features)
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            
            # Shape encoding (4 features)
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
            
            # Orientation features (3 features)
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            features.append(np.exp(-angle_diff / 45.0))
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))  # FIX: Added this feature
            
            # Habit plane proximity (1 feature)
            habit_distance = abs(theta_deg - 54.7)
            features.append(np.exp(-habit_distance / 15.0))
            
            # Verify we have exactly 15 features
            if len(features) != 15:
                st.warning(f"Warning: Expected 15 features, got {len(features)}. Padding or truncating.")
            
            # Pad with zeros if fewer than 15
            while len(features) < 15:
                features.append(0.0)
            
            # Truncate if more than 15
            features = features[:15]
            
            encoded.append(features)
        
        return torch.FloatTensor(encoded)
    
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """Interpolate full spatial stress fields using transformer attention with enhanced spatial locality"""
        if not sources:
            st.warning("No sources provided for interpolation.")
            return None
        
        try:
            # Extract source parameters and fields
            source_params = []
            source_fields = []
            source_indices = []  # Track original indices
            
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    st.warning(f"Skipping source {i}: missing params or history")
                    continue
                
                source_params.append(src['params'])
                source_indices.append(i)
                
                # Get last frame stress fields
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        # Extract all stress components
                        stress_fields = last_frame['stresses']
                        
                        # Get von Mises if available, otherwise compute
                        if 'von_mises' in stress_fields:
                            vm = stress_fields['von_mises']
                        else:
                            # Compute von Mises from components
                            vm = self.compute_von_mises(stress_fields)
                        
                        # Get hydrostatic stress
                        if 'sigma_hydro' in stress_fields:
                            hydro = stress_fields['sigma_hydro']
                        else:
                            hydro = self.compute_hydrostatic(stress_fields)
                        
                        # Get stress magnitude
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
            
            if not source_params or not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Check if all fields have same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                # Resize to common shape
                target_shape = shapes[0]  # Use first shape
                resized_fields = []
                for fields in source_fields:
                    resized = {}
                    for key, field in fields.items():
                        if key in ['von_mises', 'sigma_hydro', 'sigma_mag'] and field.shape != target_shape:
                            # Resize using interpolation
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            resized[key] = zoom(field, factors, order=1)
                        elif key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            resized[key] = field
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                source_fields = resized_fields
            
            # Debug: Check feature dimensions
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Ensure we have exactly 15 features
            if source_features.shape[1] != 15 or target_features.shape[1] != 15:
                st.warning(f"Feature dimension mismatch: source_features shape={source_features.shape}, target_features shape={target_features.shape}")
            
            # Force reshape to 15 features
            if source_features.shape[1] < 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            # Compute positional weights with distance decay
            pos_weights = self.compute_positional_weights(source_params, target_params)
            
            # Normalize positional weights
            if np.sum(pos_weights) > 0:
                pos_weights = pos_weights / np.sum(pos_weights)
            else:
                pos_weights = np.ones_like(pos_weights) / len(pos_weights)
            
            # Prepare transformer input
            batch_size = 1
            seq_len = len(source_features) + 1  # Sources + target
            
            # Create sequence: [target, source1, source2, ...]
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)  # [1, seq_len, features]
            
            # Apply input projection
            proj_features = self.input_proj(all_features)
            
            # Add positional encoding
            proj_features = self.pos_encoder(proj_features)
            
            # Transformer encoding
            transformer_output = self.transformer(proj_features)
            
            # Extract target representation (first in sequence)
            target_rep = transformer_output[:, 0, :]
            
            # Compute attention to sources
            source_reps = transformer_output[:, 1:, :]
            
            # Compute scaled dot-product attention
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1) / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            
            # Apply softmax to get attention weights
            transformer_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().numpy()
            
            # NEW: Apply locality weight factor to balance spatial and transformer weights
            # locality_weight_factor = 0.7 means 70% transformer weights, 30% spatial weights
            # This can be adjusted to give more emphasis to spatial locality
            combined_weights = (
                self.locality_weight_factor * transformer_weights + 
                (1 - self.locality_weight_factor) * pos_weights
            )
            
            # Normalize combined weights
            combined_weights = combined_weights / np.sum(combined_weights)
            
            # Calculate entropy for weight distribution analysis
            entropy_transformer = self._calculate_entropy(transformer_weights)
            entropy_spatial = self._calculate_entropy(pos_weights)
            entropy_combined = self._calculate_entropy(combined_weights)
            
            # Interpolate spatial fields
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += combined_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            # Compute additional metrics
            max_vm = np.max(interpolated_fields['von_mises'])
            max_hydro = np.max(np.abs(interpolated_fields['sigma_hydro']))
            
            # Extract source theta values for visualization
            source_theta_degrees = []
            source_distances = []  # Store distances to target
            
            target_theta_rad = target_params.get('theta', 0.0)
            target_theta_deg = np.degrees(target_theta_rad)
            
            for src in source_params:
                theta_rad = src.get('theta', 0.0)
                theta_deg = np.degrees(theta_rad)
                source_theta_degrees.append(theta_deg)
                
                # Calculate angular distance
                angular_dist = abs(theta_deg - target_theta_deg)
                angular_dist = min(angular_dist, 360 - angular_dist)  # Handle circular nature
                source_distances.append(angular_dist)
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'transformer': transformer_weights.tolist(),
                    'positional': pos_weights.tolist(),
                    'combined': combined_weights.tolist(),
                    'entropy': {
                        'transformer': entropy_transformer,
                        'spatial': entropy_spatial,
                        'combined': entropy_combined
                    }
                },
                'statistics': {
                    'von_mises': {
                        'max': float(max_vm),
                        'mean': float(np.mean(interpolated_fields['von_mises'])),
                        'std': float(np.std(interpolated_fields['von_mises'])),
                        'min': float(np.min(interpolated_fields['von_mises']))
                    },
                    'sigma_hydro': {
                        'max_tension': float(np.max(interpolated_fields['sigma_hydro'])),
                        'max_compression': float(np.min(interpolated_fields['sigma_hydro'])),
                        'mean': float(np.mean(interpolated_fields['sigma_hydro'])),
                        'std': float(np.std(interpolated_fields['sigma_hydro']))
                    },
                    'sigma_mag': {
                        'max': float(np.max(interpolated_fields['sigma_mag'])),
                        'mean': float(np.mean(interpolated_fields['sigma_mag'])),
                        'std': float(np.std(interpolated_fields['sigma_mag'])),
                        'min': float(np.min(interpolated_fields['sigma_mag']))
                    }
                },
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_theta_degrees': source_theta_degrees,
                'source_distances': source_distances,
                'source_indices': source_indices,
                'source_fields': source_fields  # Store source fields for comparison
            }
            
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def compute_von_mises(self, stress_fields):
        """Compute von Mises stress from stress components"""
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
        
        return np.zeros((100, 100))  # Default shape
    
    def compute_hydrostatic(self, stress_fields):
        """Compute hydrostatic stress from stress components"""
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            return (sxx + syy + szz) / 3
        
        return np.zeros((100, 100))  # Default shape
    
    def _calculate_entropy(self, weights):
        """Calculate entropy of weight distribution"""
        weights = np.array(weights)
        weights = weights[weights > 0]  # Remove zeros
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return -np.sum(weights * np.log(weights + 1e-10))  # Add small epsilon to avoid log(0)

# =============================================
# ENHANCED HEATMAP VISUALIZER WITH COMPARISON DASHBOARD
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with comparison dashboard and publication styling"""
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                            cmap_name='viridis', figsize=(12, 10),
                            colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                            show_stats=True, target_angle=None, defect_type=None,
                            show_colorbar=True, aspect_ratio='equal'):
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
        
        # Add colorbar with enhanced styling
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label, fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
        
        # Customize plot with publication styling
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
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
                                 target_angle=None, defect_type=None):
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
                    if target_angle is not None:
                        row_text.append(f"Position: ({i}, {j})<br>Stress: {stress_field[i, j]:.4f} GPa<br>θ: {target_angle:.1f}°")
                    else:
                        row_text.append(f"Position: ({i}, {j})<br>Stress: {stress_field[i, j]:.4f} GPa")
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
    
    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info, 
                                   target_angle, defect_type, component='von_mises',
                                   cmap_name='viridis', figsize=(20, 15),
                                   ground_truth_index=None):
        """
        Create comprehensive comparison dashboard showing:
        1. Interpolated result
        2. Ground truth (selected source or closest match)
        3. Difference between interpolated and ground truth
        4. Weight distribution analysis
        5. Angular distribution of sources
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Determine vmin and vmax for consistent scaling
        all_values = [interpolated_fields[component]]
        if ground_truth_index is not None and ground_truth_index < len(source_fields):
            all_values.append(source_fields[ground_truth_index][component])
        
        vmin = min(np.min(field) for field in all_values)
        vmax = max(np.max(field) for field in all_values)
        
        # 1. Interpolated result (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(interpolated_fields[component], cmap=cmap_name, 
                        vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f"{component.replace('_', ' ').title()} (GPa)")
        ax1.set_title(f'Interpolated Result\nθ = {target_angle:.1f}°, {defect_type}', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.2)
        
        # 2. Ground truth comparison (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        if ground_truth_index is not None and ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index][component]
            gt_theta = source_info['theta_degrees'][ground_truth_index]
            gt_distance = source_info['distances'][ground_truth_index]
            
            im2 = ax2.imshow(gt_field, cmap=cmap_name, 
                            vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f"{component.replace('_', ' ').title()} (GPa)")
            ax2.set_title(f'Ground Truth\nθ = {gt_theta:.1f}° (Δ={gt_distance:.1f}°)', 
                         fontsize=16, fontweight='bold')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.grid(True, alpha=0.2)
        else:
            ax2.text(0.5, 0.5, 'Select Ground Truth Source', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax2.set_title('Ground Truth Selection', fontsize=16, fontweight='bold')
            ax2.set_axis_off()
        
        # 3. Difference plot (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if ground_truth_index is not None and ground_truth_index < len(source_fields):
            diff_field = interpolated_fields[component] - source_fields[ground_truth_index][component]
            max_diff = np.max(np.abs(diff_field))
            
            im3 = ax3.imshow(diff_field, cmap='RdBu_r', 
                            vmin=-max_diff, vmax=max_diff, aspect='equal', 
                            interpolation='bilinear', origin='lower')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Difference (GPa)')
            ax3.set_title(f'Difference\nMax Abs Error: {max_diff:.3f} GPa', 
                         fontsize=16, fontweight='bold')
            ax3.set_xlabel('X Position')
            ax3.set_ylabel('Y Position')
            ax3.grid(True, alpha=0.2)
            
            # Calculate and display error metrics
            mse = np.mean(diff_field**2)
            mae = np.mean(np.abs(diff_field))
            rmse = np.sqrt(mse)
            
            error_text = (f"MSE: {mse:.4f}\n"
                         f"MAE: {mae:.4f}\n"
                         f"RMSE: {rmse:.4f}")
            ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax3.text(0.5, 0.5, 'Difference will appear\nwhen ground truth is selected', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax3.set_title('Difference Analysis', fontsize=16, fontweight='bold')
            ax3.set_axis_off()
        
        # 4. Weight distribution analysis (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'weights' in source_info:
            weights = source_info['weights']['combined']
            x = range(len(weights))
            
            bars = ax4.bar(x, weights, alpha=0.7, color='steelblue', edgecolor='black')
            ax4.set_xlabel('Source Index')
            ax4.set_ylabel('Weight')
            ax4.set_title('Source Weight Distribution', fontsize=16, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add theta labels to bars
            for i, bar in enumerate(bars):
                if i < len(source_info['theta_degrees']):
                    theta = source_info['theta_degrees'][i]
                    height = bar.get_height()
                    if height > 0.01:  # Only label significant bars
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                f'θ={theta:.0f}°', ha='center', va='bottom',
                                fontsize=8, rotation=90)
            
            # Highlight the selected ground truth if applicable
            if ground_truth_index is not None and ground_truth_index < len(weights):
                bars[ground_truth_index].set_color('red')
                bars[ground_truth_index].set_alpha(0.9)
        
        # 5. Angular distribution of sources (middle center)
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')
        if 'theta_degrees' in source_info and 'distances' in source_info:
            # Convert angles to radians for polar plot
            angles_rad = np.radians(source_info['theta_degrees'])
            distances = source_info['distances']
            
            # Plot sources as points with size proportional to weight
            if 'weights' in source_info:
                weights = source_info['weights']['combined']
                sizes = 100 * np.array(weights) / np.max(weights)  # Normalize sizes
            else:
                sizes = 50 * np.ones(len(angles_rad))
            
            scatter = ax5.scatter(angles_rad, distances, 
                                 s=sizes, alpha=0.7, c='blue', edgecolors='black')
            
            # Plot target angle
            target_rad = np.radians(target_angle)
            ax5.scatter(target_rad, 0, s=200, c='red', marker='*', edgecolors='black', label='Target')
            
            # Plot habit plane (54.7°)
            habit_rad = np.radians(54.7)
            ax5.axvline(habit_rad, color='green', alpha=0.5, linestyle='--', label='Habit Plane (54.7°)')
            
            ax5.set_title('Angular Distribution of Sources', fontsize=16, fontweight='bold', pad=20)
            ax5.set_theta_zero_location('N')  # 0° at top
            ax5.set_theta_direction(-1)  # Clockwise
            ax5.legend(loc='upper right', fontsize=10)
        
        # 6. Component comparison (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        component_names = ['Von Mises', 'Hydrostatic', 'Stress Magnitude']
        
        # Get statistics for each component
        stats_data = []
        if 'statistics' in source_info:
            for comp in components:
                if comp in source_info['statistics']:
                    stats = source_info['statistics'][comp]
                    stats_data.append({
                        'component': comp,
                        'max': stats['max'],
                        'mean': stats['mean'],
                        'std': stats['std']
                    })
        
        if stats_data:
            x = np.arange(len(components))
            width = 0.25
            
            # Plot max values
            max_values = [stats['max'] for stats in stats_data]
            ax6.bar(x - width, max_values, width, label='Max', color='red', alpha=0.7)
            
            # Plot mean values
            mean_values = [stats['mean'] for stats in stats_data]
            ax6.bar(x, mean_values, width, label='Mean', color='blue', alpha=0.7)
            
            # Plot std values
            std_values = [stats['std'] for stats in stats_data]
            ax6.bar(x + width, std_values, width, label='Std', color='green', alpha=0.7)
            
            ax6.set_xlabel('Stress Component')
            ax6.set_ylabel('Value (GPa)')
            ax6.set_title('Component Statistics Comparison', fontsize=16, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(component_names)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Local spatial correlation analysis (bottom row)
        ax7 = fig.add_subplot(gs[2, :])
        
        if ground_truth_index is not None and ground_truth_index < len(source_fields):
            # Calculate spatial correlation between interpolated and ground truth
            interp_flat = interpolated_fields[component].flatten()
            gt_flat = source_fields[ground_truth_index][component].flatten()
            
            # Create scatter plot
            ax7.scatter(gt_flat, interp_flat, alpha=0.5, s=10)
            
            # Add correlation line
            min_val = min(np.min(gt_flat), np.min(interp_flat))
            max_val = max(np.max(gt_flat), np.max(interp_flat))
            ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
            
            # Calculate correlation coefficient
            from scipy.stats import pearsonr
            corr_coef, _ = pearsonr(gt_flat, interp_flat)
            
            ax7.set_xlabel(f'Ground Truth {component.replace("_", " ").title()} (GPa)')
            ax7.set_ylabel(f'Interpolated {component.replace("_", " ").title()} (GPa)')
            ax7.set_title(f'Spatial Correlation Analysis\nPearson Correlation: {corr_coef:.3f}', 
                         fontsize=16, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # Add statistics text box
            mse = np.mean((interp_flat - gt_flat)**2)
            mae = np.mean(np.abs(interp_flat - gt_flat))
            stats_text = (f'Correlation: {corr_coef:.3f}\n'
                         f'MSE: {mse:.4f}\n'
                         f'MAE: {mae:.4f}')
            ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                    fontsize=12, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'Comprehensive Stress Field Analysis - Target θ={target_angle:.1f}°, {defect_type}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig
    
    def create_interactive_3d_surface(self, stress_field, title="3D Stress Surface",
                                     cmap_name='viridis', width=900, height=700,
                                     target_angle=None, defect_type=None):
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
                    row_text.append(f"X: {j}, Y: {i}<br>Stress: {stress_field[i, j]:.4f} GPa")
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
                                       figsize=(8, 8), show_habit_plane=True):
        """Create polar plot showing angular orientation"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Convert target angle to radians
        theta_rad = np.radians(target_angle_deg)
        
        # Plot the defect orientation as a red arrow
        ax.arrow(theta_rad, 0.8, 0, 0.6, width=0.02, 
                color='red', alpha=0.8, label=f'Defect Orientation: {target_angle_deg:.1f}°')
        
        # Plot habit plane orientation (54.7°) if requested
        if show_habit_plane:
            habit_plane_rad = np.radians(54.7)
            ax.arrow(habit_plane_rad, 0.8, 0, 0.6, width=0.02, 
                    color='blue', alpha=0.5, label='Habit Plane (54.7°)')
        
        # Plot cardinal directions
        for angle, label in [(0, '0°'), (90, '90°'), (180, '180°'), (270, '270°')]:
            ax.axvline(np.radians(angle), color='gray', linestyle='--', alpha=0.3)
        
        # Customize plot
        ax.set_title(f'Defect Orientation\nθ = {target_angle_deg:.1f}°, {defect_type}', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Add annotation for angular difference from habit plane
        if show_habit_plane:
            angular_diff = abs(target_angle_deg - 54.7)
            ax.annotate(f'Δθ = {angular_diff:.1f}°\nfrom habit plane',
                       xy=(theta_rad, 1.2), xytext=(theta_rad, 1.4),
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                       fontsize=12, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def create_comparison_heatmaps(self, stress_fields_dict, cmap_name='viridis',
                                  figsize=(18, 6), titles=None, target_angle=None, defect_type=None):
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
        
        plt.suptitle(suptitle, fontsize=22, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def create_3d_surface_plot(self, stress_field, title="3D Stress Surface",
                             cmap_name='viridis', figsize=(14, 10), target_angle=None, defect_type=None):
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
                                      cmap_name='viridis', figsize=(24, 16)):
        """Create comprehensive dashboard with all stress components and angular orientation"""
        fig = plt.figure(figsize=figsize)
        
        # Create subplots grid with polar plot included
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
        
        # 0. Angular orientation plot (polar plot)
        ax0 = fig.add_subplot(gs[0, 0], projection='polar')
        theta_rad = np.radians(theta)
        ax0.arrow(theta_rad, 0.8, 0, 0.6, width=0.02, color='red', alpha=0.8)
        
        # Plot habit plane
        habit_plane_rad = np.radians(54.7)
        ax0.arrow(habit_plane_rad, 0.8, 0, 0.6, width=0.02, color='blue', alpha=0.5)
        
        # Customize polar plot
        ax0.set_title(f'Defect Orientation\nθ = {theta:.1f}°', fontsize=16, fontweight='bold', pad=15)
        ax0.set_theta_zero_location('N')
        ax0.set_theta_direction(-1)
        ax0.set_ylim(0, 1.5)
        ax0.grid(True, alpha=0.3)
        
        # 1. Von Mises stress (main plot)
        ax1 = fig.add_subplot(gs[0, 1:3])
        im1 = ax1.imshow(stress_fields['von_mises'], cmap=cmap_name, aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Von Mises Stress (GPa)')
        ax1.set_title(f'Von Mises Stress at θ={theta}°\nDefect: {defect_type}',
                    fontsize=18, fontweight='bold')
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
            f"  Habit Plane: 54.7°\n"
            f"  Angular Deviation: {abs(theta - 54.7):.1f}°"
        )
        
        ax8.text(0.1, 0.5, params_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax8.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))
        ax8.set_title('Interpolation Parameters', fontsize=18, fontweight='bold', pad=20)
        
        plt.suptitle(f'Comprehensive Stress Analysis - θ={theta}°, {defect_type}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig

# =============================================
# RESULTS MANAGER FOR EXPORT
# =============================================
class ResultsManager:
    """Manager for exporting interpolation results"""
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        """Prepare data for export"""
        result = interpolation_result.copy()
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'transformer_spatial',
                'visualization_params': visualization_params
            },
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'shape': result['shape'],
                'statistics': result['statistics'],
                'weights': result['weights'],
                'num_sources': result.get('num_sources', 0),
                'source_theta_degrees': result.get('source_theta_degrees', []),
                'source_distances': result.get('source_distances', []),
                'source_indices': result.get('source_indices', [])
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
        
        return export_data
    
    def export_to_json(self, export_data, filename=None):
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = export_data['result']['target_angle']
            defect = export_data['result']['target_params']['defect_type']
            filename = f"transformer_interpolation_theta_{theta}_{defect}_{timestamp}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        """Export flattened field data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename = f"stress_fields_theta_{theta}_{defect}_{timestamp}.csv"
        
        # Create DataFrame with flattened data
        data_dict = {}
        for field_name, field_data in interpolation_result['fields'].items():
            data_dict[field_name] = field_data.flatten()
        
        df = pd.DataFrame(data_dict)
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
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
        else:
            return str(obj)

# =============================================
# HELPER FUNCTIONS
# =============================================
def _calculate_entropy(weights):
    """Calculate entropy of weight distribution"""
    weights = np.array(weights)
    weights = weights[weights > 0]  # Remove zeros
    if len(weights) == 0:
        return 0.0
    weights = weights / weights.sum()
    return -np.sum(weights * np.log(weights + 1e-10))  # Add small epsilon to avoid log(0)

# =============================================
# MAIN APPLICATION WITH ENHANCED COMPARISON DASHBOARD
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Transformer Stress Interpolation with Comparison Dashboard",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Transformer Stress Field Interpolation with Comparison Dashboard</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
        <strong>🔬 Physics-aware stress interpolation with comprehensive comparison dashboard.</strong><br>
        • Load simulation files from numerical_solutions directory<br>
        • Interpolate stress fields at custom polar angles (default: 54.7°)<br>
        • Visualize von Mises, hydrostatic, and stress magnitude fields<br>
        • <strong>New:</strong> Adjustable spatial locality weight factor for better visual similarity to nearby sources<br>
        • <strong>New:</strong> Comprehensive comparison dashboard with ground truth selection and difference analysis<br>
        • Choose from 50+ colormaps including jet, turbo, rainbow, inferno<br>
        • Publication-ready visualizations with angular orientation plots<br>
        • Export results in multiple formats
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        # Initialize with adjustable spatial locality weight factor
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator(
            spatial_sigma=0.2,
            locality_weight_factor=0.7  # Default: 70% transformer weights, 30% spatial locality
        )
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'selected_ground_truth' not in st.session_state:
        st.session_state.selected_ground_truth = None
    
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
                else:
                    st.warning("No solutions found in directory")
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.session_state.selected_ground_truth = None
                st.success("Cache cleared")
        
        # Debug button
        if st.button("🔍 Debug Feature Dimensions", use_container_width=True):
            if st.session_state.solutions:
                source_params = [sol['params'] for sol in st.session_state.solutions[:1]]
                shape = st.session_state.transformer_interpolator.debug_feature_dimensions(
                    source_params, 54.7
                )
                st.write(f"Feature dimensions: {shape}")
                st.write(f"Number of solutions: {len(st.session_state.solutions)}")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📁 Loaded Solutions ({len(st.session_state.solutions)})"):
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    params = sol.get('params', {})
                    theta_deg = np.degrees(params.get('theta', 0))
                    st.write(f"**Solution {i+1}:** {params.get('defect_type', 'Unknown')} at θ={theta_deg:.1f}°")
                if len(st.session_state.solutions) > 5:
                    st.write(f"... and {len(st.session_state.solutions) - 5} more")
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        custom_theta = st.number_input(
            "Custom Polar Angle θ (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=54.7,
            step=0.1,
            help="Set custom polar angle for interpolation (default: 54.7°)"
        )
        
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for interpolation"
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        eps0 = eigen_strains[defect_type]
        
        shape = st.selectbox(
            "Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
            index=0
        )
        
        kappa = st.slider(
            "Kappa Parameter",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.1,
            help="Material parameter"
        )
        
        # Transformer parameters with adjustable spatial locality weight factor
        st.markdown('<h2 class="section-header">🤖 Transformer Parameters</h2>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            spatial_sigma = st.slider(
                "Spatial Sigma",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Spatial locality regularization parameter"
            )
        with col_t2:
            locality_weight_factor = st.slider(
                "Spatial Locality Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="0 = pure transformer attention, 1 = pure spatial locality"
            )
        
        col_t3, col_t4 = st.columns(2)
        with col_t3:
            attention_temp = st.slider(
                "Attention Temperature",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Temperature for attention scaling"
            )
        
        # Visualization parameters
        st.markdown('<h2 class="section-header">🎨 Visualization</h2>', unsafe_allow_html=True)
        colormap_category = st.selectbox(
            "Colormap Category",
            list(COLORMAP_OPTIONS.keys()),
            index=4,  # Default to Publication Standard
            help="Select colormap category for publication-quality figures"
        )
        colormap_name = st.selectbox(
            "Select Colormap",
            COLORMAP_OPTIONS[colormap_category],
            index=0
        )
        
        visualization_type = st.selectbox(
            "Visualization Type",
            ["2D Heatmap", "Interactive Heatmap", "3D Surface", "Interactive 3D Surface",
             "Comparison View", "Comprehensive Dashboard", "Angular Orientation", "Comparison Dashboard"],
            index=7,  # Default to Comparison Dashboard
            help="Select visualization type"
        )
        
        # Ground truth selection for comparison
        st.markdown('<h2 class="section-header">🔍 Ground Truth Comparison</h2>', unsafe_allow_html=True)
        if st.session_state.solutions:
            source_options = []
            for i, sol in enumerate(st.session_state.solutions):
                params = sol.get('params', {})
                theta_deg = np.degrees(params.get('theta', 0))
                source_options.append(f"{i}: {params.get('defect_type', 'Unknown')} at θ={theta_deg:.1f}°")
            
            ground_truth_index = st.selectbox(
                "Select Ground Truth Source",
                options=range(len(source_options)),
                format_func=lambda x: source_options[x],
                help="Select a source simulation to compare with interpolated result"
            )
            st.session_state.selected_ground_truth = ground_truth_index
        
        # Interpolation button
        st.markdown("---")
        if st.button("✨ Perform Transformer Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                # Update transformer parameters with adjustable spatial locality weight factor
                st.session_state.transformer_interpolator.set_spatial_parameters(
                    spatial_sigma=spatial_sigma,
                    locality_weight_factor=locality_weight_factor
                )
                st.session_state.transformer_interpolator.temperature = attention_temp
                
                # Prepare target parameters
                target_params = {
                    'defect_type': defect_type,
                    'eps0': eps0,
                    'kappa': kappa,
                    'theta': np.radians(custom_theta),
                    'shape': shape
                }
                
                # Perform interpolation
                with st.spinner("Performing transformer-based spatial interpolation..."):
                    try:
                        result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                            st.session_state.solutions,
                            custom_theta,
                            target_params
                        )
                        if result:
                            st.session_state.interpolation_result = result
                            st.success(f"✅ Successfully interpolated stress fields at θ={custom_theta:.1f}° using {result['num_sources']} sources")
                            
                            # Show weight distribution summary
                            weights = result['weights']['combined']
                            if len(weights) > 0:
                                max_weight_idx = np.argmax(weights)
                                max_weight = weights[max_weight_idx]
                                max_theta = result['source_theta_degrees'][max_weight_idx] if max_weight_idx < len(result['source_theta_degrees']) else 0
                                st.info(f"Highest weight source: Index {max_weight_idx} (θ={max_theta:.1f}°) with weight {max_weight:.3f}")
                        else:
                            st.error("❌ Failed to interpolate stress fields. Check data compatibility.")
                    except Exception as e:
                        st.error(f"❌ Error during interpolation: {str(e)}")
    
    # Main content
    if not st.session_state.solutions:
        st.warning("📁 Please load solutions first using the button in the sidebar.")
        
        # Directory information
        with st.expander("📁 Directory Information", expanded=True):
            st.info(f"**Solutions Directory:** {SOLUTIONS_DIR}")
            st.write("""
            **Expected file formats:** .pkl, .pickle, .pt, .pth
            **Expected data structure:**
            - Each file should contain a dictionary with:
              - 'params': Dictionary of simulation parameters
              - 'history': List of simulation frames
              - Each frame should contain 'stresses' dictionary with stress fields
            """)
        
        # Quick guide
        st.markdown("""
        ## 📋 Quick Start Guide
        1. **Prepare Data**: Place your simulation files in the `numerical_solutions` directory
        2. **Load Solutions**: Click the "Load Solutions" button in the sidebar
        3. **Set Parameters**: Configure target angle and defect type
        4. **Perform Interpolation**: Click "Perform Transformer Interpolation"
        5. **Visualize Results**: Choose visualization type and colormap
        
        ## 🔬 Key Features
        ### Transformer Architecture
        - Multi-head attention across source simulations
        - Spatial locality regularization for smooth interpolation
        - Physics-aware parameter encoding
        
        ### Enhanced Spatial Locality
        - **Adjustable weight factor**: Control balance between transformer attention and spatial locality
        - **Better visual similarity**: Results look more like spatially nearer objects
        - **Distance decay**: Gaussian kernel with controllable sigma parameter
        
        ### Comprehensive Comparison Dashboard
        - **Interpolated result**: Full stress field interpolation
        - **Ground truth comparison**: Select any source for direct comparison
        - **Difference analysis**: Quantitative error metrics (MSE, MAE, RMSE)
        - **Weight distribution**: Visualize how sources are weighted
        - **Angular distribution**: Polar plot of source orientations
        - **Spatial correlation**: Scatter plot of interpolated vs ground truth
        
        ### Stress Components
        - **Von Mises Stress (σ_vm)**: Equivalent tensile stress
        - **Hydrostatic Stress (σ_h)**: Mean normal stress (trace/3)
        - **Stress Magnitude (σ_mag)**: Overall stress intensity
        
        ### Visualization Options
        - 50+ colormaps including jet, turbo, rainbow, inferno
        - Publication-ready figures with consistent aspect ratios
        - Angular orientation polar plots
        - Interactive 3D surfaces with Plotly
        - Comprehensive dashboards with statistics
        """)
    else:
        # Improved tabs layout with comparison dashboard
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Results", 
            "🎨 Visualization", 
            "🔍 Comparison Dashboard",
            "🎯 Target Parameters",
            "⚖️ Weights Analysis",
            "📤 Export"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 Interpolation Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    vm_stats = result['statistics']['von_mises']
                    st.metric("Von Mises Max", f"{vm_stats['max']:.3f} GPa", 
                             f"Mean: {vm_stats['mean']:.3f} GPa")
                
                with col2:
                    hydro_stats = result['statistics']['sigma_hydro']
                    st.metric("Hydrostatic Range", 
                             f"{hydro_stats['max_tension']:.3f}/{hydro_stats['max_compression']:.3f} GPa",
                             f"Mean: {hydro_stats['mean']:.3f} GPa")
                
                with col3:
                    mag_stats = result['statistics']['sigma_mag']
                    st.metric("Stress Magnitude Max", f"{mag_stats['max']:.3f} GPa",
                             f"Mean: {mag_stats['mean']:.3f} GPa")
                
                with col4:
                    st.metric("Field Shape", f"{result['shape'][0]}×{result['shape'][1]}",
                             f"θ={result['target_angle']:.1f}° | Sources: {result.get('num_sources', 0)}")
                
                # Display parameters
                with st.expander("🔍 Interpolation Parameters", expanded=True):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.write("**Target Parameters:**")
                        for key, value in result['target_params'].items():
                            if key == 'theta':
                                st.write(f"- {key}: {np.degrees(value):.2f}°")
                            else:
                                st.write(f"- {key}: {value}")
                    
                    with col_p2:
                        st.write("**Interpolation Settings:**")
                        st.write(f"- Spatial Sigma: {spatial_sigma}")
                        st.write(f"- Locality Weight Factor: {locality_weight_factor}")
                        st.write(f"- Attention Temperature: {attention_temp}")
                        st.write(f"- Number of Sources: {result.get('num_sources', len(result['weights']['combined']))}")
                
                # Quick preview
                st.markdown("#### 👀 Quick Preview")
                
                # Create a quick preview figure
                fig_preview, axes = plt.subplots(1, 3, figsize=(15, 4))
                components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude']
                
                for idx, (comp, title) in enumerate(zip(components, titles)):
                    ax = axes[idx]
                    im = ax.imshow(result['fields'][comp], cmap='viridis', aspect='equal', origin='lower')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title(title, fontsize=12)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.grid(True, alpha=0.2)
                
                plt.suptitle(f"Stress Fields at θ={result['target_angle']:.1f}°", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig_preview)
                plt.close(fig_preview)
            else:
                st.info("🔍 Configure parameters and click 'Perform Transformer Interpolation' to generate results")
        
        with tab2:
            st.markdown('<h2 class="section-header">🎨 Enhanced Stress Field Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                if visualization_type == "Angular Orientation":
                    st.markdown("#### 🧭 Angular Orientation Visualization")
                    
                    # Create angular orientation plot
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        result['target_params']['defect_type']
                    )
                    st.pyplot(fig_angular)
                    plt.close(fig_angular)
                    
                    # Show angular statistics
                    col_a1, col_a2, col_a3 = st.columns(3)
                    with col_a1:
                        st.metric("Target Angle θ", f"{result['target_angle']:.1f}°")
                    with col_a2:
                        habit_deviation = abs(result['target_angle'] - 54.7)
                        st.metric("Deviation from Habit Plane", f"{habit_deviation:.1f}°")
                    with col_a3:
                        st.metric("Defect Type", result['target_params']['defect_type'])
                elif visualization_type == "Comparison Dashboard":
                    # This is handled in tab3
                    st.info("The Comparison Dashboard is available in the '🔍 Comparison Dashboard' tab.")
                else:
                    # Component selection
                    stress_component = st.selectbox(
                        "Select Stress Component",
                        ["von_mises", "sigma_hydro", "sigma_mag"],
                        index=0,
                        key="viz_component"
                    )
                    
                    # Component names for display
                    component_names = {
                        'von_mises': 'Von Mises Stress',
                        'sigma_hydro': 'Hydrostatic Stress',
                        'sigma_mag': 'Stress Magnitude'
                    }
                    
                    # Create visualization based on selected type
                    if visualization_type == "2D Heatmap":
                        fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                            result['fields'][stress_component],
                            title=f"{component_names[stress_component]}",
                            cmap_name=colormap_name,
                            colorbar_label=f"{component_names[stress_component]} (GPa)",
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type']
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Colormap preview
                        with st.expander("🎨 Colormap Preview", expanded=False):
                            fig_preview = st.session_state.heatmap_visualizer.get_colormap_preview(colormap_name)
                            st.pyplot(fig_preview)
                            plt.close(fig_preview)
                    
                    elif visualization_type == "Interactive Heatmap":
                        fig = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                            result['fields'][stress_component],
                            title=f"{component_names[stress_component]}",
                            cmap_name=colormap_name,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif visualization_type == "3D Surface":
                        # Simple 3D surface (matplotlib)
                        fig = plt.figure(figsize=(12, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        x = np.arange(result['fields'][stress_component].shape[1])
                        y = np.arange(result['fields'][stress_component].shape[0])
                        X, Y = np.meshgrid(x, y)
                        
                        # Get z values from the stress field
                        Z = result['fields'][stress_component]
                        
                        # Plot surface
                        surf = ax.plot_surface(X, Y, Z, cmap=colormap_name, 
                                             linewidth=0, antialiased=True)
                        
                        # Add color bar
                        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                        
                        # Set labels and title
                        title_str = f"3D {component_names[stress_component]} at θ={result['target_angle']:.1f}°"
                        ax.set_title(title_str)
                        ax.set_xlabel('X Position')
                        ax.set_ylabel('Y Position')
                        ax.set_zlabel('Stress (GPa)')
                        
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    elif visualization_type == "Interactive 3D Surface":
                        fig = st.session_state.heatmap_visualizer.create_interactive_3d_surface(
                            result['fields'][stress_component],
                            title=f"{component_names[stress_component]}",
                            cmap_name=colormap_name,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif visualization_type == "Comparison View":
                        comparison_fields = {
                            'Von Mises': result['fields']['von_mises'],
                            'Hydrostatic': result['fields']['sigma_hydro'],
                            'Magnitude': result['fields']['sigma_mag']
                        }
                        fig = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                            comparison_fields,
                            cmap_name=colormap_name,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type']
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    elif visualization_type == "Comprehensive Dashboard":
                        fig = st.session_state.heatmap_visualizer.create_comprehensive_dashboard(
                            result['fields'],
                            result['target_angle'],
                            result['target_params']['defect_type'],
                            cmap_name=colormap_name
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Additional statistics
                    if visualization_type not in ["Angular Orientation", "Comparison Dashboard"]:
                        with st.expander("📊 Detailed Statistics", expanded=False):
                            # Map component names to statistics keys
                            stats_mapping = {
                                'von_mises': 'von_mises',
                                'sigma_hydro': 'sigma_hydro',
                                'sigma_mag': 'sigma_mag'
                            }
                            
                            if stress_component in stats_mapping and stats_mapping[stress_component] in result['statistics']:
                                stats = result['statistics'][stats_mapping[stress_component]]
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                with col_s1:
                                    st.metric("Maximum", f"{stats['max']:.3f} GPa")
                                with col_s2:
                                    st.metric("Minimum", f"{stats.get('min', 0):.3f} GPa")
                                with col_s3:
                                    st.metric("Mean", f"{stats['mean']:.3f} GPa")
                                with col_s4:
                                    st.metric("Std Dev", f"{stats['std']:.3f} GPa")
                                
                                # Histogram
                                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                                ax_hist.hist(result['fields'][stress_component].flatten(), bins=50, alpha=0.7, edgecolor='black')
                                ax_hist.set_xlabel(f'{component_names[stress_component]} (GPa)')
                                ax_hist.set_ylabel('Frequency')
                                ax_hist.set_title(f'Distribution of {component_names[stress_component]}')
                                ax_hist.grid(True, alpha=0.3)
                                st.pyplot(fig_hist)
                                plt.close(fig_hist)
                            else:
                                st.warning(f"Statistics not available for component: {stress_component}")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">🔍 Comprehensive Comparison Dashboard</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Component selection for comparison dashboard
                component = st.selectbox(
                    "Select Component for Comparison",
                    ["von_mises", "sigma_hydro", "sigma_mag"],
                    index=0,
                    key="comparison_component"
                )
                
                # Create source info for dashboard
                source_info = {
                    'theta_degrees': result['source_theta_degrees'],
                    'distances': result['source_distances'],
                    'weights': result['weights'],
                    'statistics': result['statistics']
                }
                
                # Get source fields for comparison
                source_fields = result.get('source_fields', [])
                
                # Create the comparison dashboard
                fig_dashboard = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                    result['fields'],
                    source_fields,
                    source_info,
                    result['target_angle'],
                    result['target_params']['defect_type'],
                    component=component,
                    cmap_name=colormap_name,
                    ground_truth_index=st.session_state.selected_ground_truth
                )
                
                st.pyplot(fig_dashboard)
                plt.close(fig_dashboard)
                
                # Show quantitative comparison metrics
                if st.session_state.selected_ground_truth is not None:
                    gt_index = st.session_state.selected_ground_truth
                    if gt_index < len(source_fields):
                        st.markdown("#### 📊 Quantitative Comparison Metrics")
                        
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        # Calculate metrics
                        interp_flat = result['fields'][component].flatten()
                        gt_flat = source_fields[gt_index][component].flatten()
                        
                        from scipy.stats import pearsonr
                        corr_coef, _ = pearsonr(gt_flat, interp_flat)
                        mse = np.mean((interp_flat - gt_flat)**2)
                        mae = np.mean(np.abs(interp_flat - gt_flat))
                        rmse = np.sqrt(mse)
                        
                        with col_m1:
                            st.metric("Pearson Correlation", f"{corr_coef:.3f}")
                        with col_m2:
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                        with col_m3:
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                        with col_m4:
                            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                        
                        # Show angular information
                        gt_theta = source_info['theta_degrees'][gt_index]
                        angular_diff = abs(gt_theta - result['target_angle'])
                        st.info(f"**Ground Truth Source:** θ={gt_theta:.1f}° | **Angular Difference:** {angular_diff:.1f}°")
                
                # Weight analysis
                st.markdown("#### ⚖️ Weight Distribution Insights")
                if 'weights' in result:
                    weights = result['weights']['combined']
                    if len(weights) > 0:
                        # Find top contributors
                        sorted_indices = np.argsort(weights)[::-1]
                        
                        st.write("**Top 5 Contributing Sources:**")
                        top_data = []
                        for rank, idx in enumerate(sorted_indices[:5]):
                            weight = weights[idx]
                            theta = source_info['theta_degrees'][idx] if idx < len(source_info['theta_degrees']) else 0
                            angular_diff = abs(theta - result['target_angle'])
                            distance = source_info['distances'][idx] if idx < len(source_info['distances']) else 0
                            top_data.append({
                                'Rank': rank + 1,
                                'Source Index': idx,
                                'Weight': f"{weight:.4f}",
                                'θ (degrees)': f"{theta:.1f}°",
                                'Δθ': f"{angular_diff:.1f}°",
                                'Distance': f"{distance:.1f}"
                            })
                        
                        st.table(top_data)
                        
                        # Weight concentration metrics
                        total_weight = np.sum(weights)
                        top3_weight = np.sum(weights[sorted_indices[:3]])
                        top5_weight = np.sum(weights[sorted_indices[:5]])
                        
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.metric("Top 3 Concentration", f"{(top3_weight/total_weight*100):.1f}%")
                        with col_c2:
                            st.metric("Top 5 Concentration", f"{(top5_weight/total_weight*100):.1f}%")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">🎯 Target Query Parameters</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                target_params = result['target_params']
                
                # Create a comprehensive target parameters display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Target Parameters")
                    
                    # Display parameters in a nice format
                    for key, value in target_params.items():
                        if key == 'theta':
                            st.write(f"- **{key}**: {np.degrees(value):.2f}°")
                        else:
                            st.write(f"- **{key}**: {value}")
                    
                    # Additional metrics
                    st.markdown("#### 📈 Interpolation Metrics")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Number of Sources", f"{result.get('num_sources', 0)}")
                        st.metric("Field Size", f"{result['shape'][0]}×{result['shape'][1]}")
                    with col_m2:
                        habit_deviation = abs(result['target_angle'] - 54.7)
                        st.metric("Deviation from Habit Plane", f"{habit_deviation:.1f}°")
                        st.metric("Spatial Locality Weight", f"{locality_weight_factor:.2f}")
                
                with col2:
                    # Angular orientation visualization
                    st.markdown("#### 🧭 Angular Orientation")
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        result['target_params']['defect_type'],
                        figsize=(8, 8)
                    )
                    st.pyplot(fig_angular)
                    plt.close(fig_angular)
                    
                    # Habit plane information
                    st.markdown("#### 🔬 Habit Plane Reference")
                    st.info("""
                    **Habit Plane Orientation:** 54.7°
                    
                    This is the preferential crystallographic orientation where defects 
                    typically form in face-centered cubic (FCC) materials. The angular 
                    deviation from this plane influences defect formation energy and 
                    stress field characteristics.
                    """)
                    
                    # Source simulation information
                    st.markdown("#### 📂 Source Simulations Information")
                    if 'source_theta_degrees' in result and result['source_theta_degrees']:
                        source_thetas = result['source_theta_degrees']
                        st.write(f"Angular range of source simulations: {min(source_thetas):.1f}° to {max(source_thetas):.1f}°")
                        st.write(f"Average source angle: {np.mean(source_thetas):.1f}°")
                        st.write(f"Source count: {len(source_thetas)}")
                    else:
                        st.info("No source theta information available.")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab5:
            st.markdown('<h2 class="section-header">⚖️ Weights Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                weights = result['weights']
                
                # Weights visualization
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    # Transformer weights
                    fig_trans, ax_trans = plt.subplots(figsize=(10, 5))
                    x_pos = np.arange(len(weights['transformer']))
                    ax_trans.bar(x_pos, weights['transformer'], alpha=0.7, color='orange', edgecolor='black')
                    ax_trans.set_xlabel('Source Index')
                    ax_trans.set_ylabel('Weight')
                    ax_trans.set_title('Transformer Attention Weights')
                    ax_trans.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_trans)
                    plt.close(fig_trans)
                
                with col_w2:
                    # Positional weights
                    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
                    x_pos = np.arange(len(weights['positional']))
                    ax_pos.bar(x_pos, weights['positional'], alpha=0.7, color='green', edgecolor='black')
                    ax_pos.set_xlabel('Source Index')
                    ax_pos.set_ylabel('Weight')
                    ax_pos.set_title('Spatial Locality Weights')
                    ax_pos.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_pos)
                    plt.close(fig_pos)
                
                # Combined weights comparison
                fig_comb, ax_comb = plt.subplots(figsize=(12, 6))
                x = range(len(weights['combined']))
                width = 0.25
                
                ax_comb.bar([i - width for i in x], weights['transformer'], width, label='Transformer', alpha=0.7, color='orange')
                ax_comb.bar(x, weights['positional'], width, label='Positional', alpha=0.7, color='green')
                ax_comb.bar([i + width for i in x], weights['combined'], width, label='Combined', alpha=0.7, color='steelblue')
                
                ax_comb.set_xlabel('Source Index')
                ax_comb.set_ylabel('Weight')
                ax_comb.set_title('Weight Comparison', fontsize=16)
                ax_comb.legend()
                ax_comb.grid(True, alpha=0.3)
                st.pyplot(fig_comb)
                plt.close(fig_comb)
                
                # Weight statistics
                with st.expander("📊 Advanced Weight Statistics", expanded=True):
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        entropy_trans = _calculate_entropy(weights['transformer'])
                        st.metric("Transformer Weight Entropy", f"{entropy_trans:.3f}", 
                                 "Higher = more uniform distribution")
                    
                    with col_stat2:
                        entropy_pos = _calculate_entropy(weights['positional'])
                        st.metric("Positional Weight Entropy", f"{entropy_pos:.3f}")
                    
                    with col_stat3:
                        entropy_comb = _calculate_entropy(weights['combined'])
                        st.metric("Combined Weight Entropy", f"{entropy_comb:.3f}")
                    
                    # Top contributors
                    st.markdown("#### 🏆 Top 5 Contributing Sources")
                    combined_weights = np.array(weights['combined'])
                    top_indices = np.argsort(combined_weights)[-5:][::-1]
                    
                    top_data = []
                    for i, idx in enumerate(top_indices):
                        top_data.append({
                            'Rank': i+1,
                            'Source Index': idx,
                            'Combined Weight': f"{combined_weights[idx]:.4f}",
                            'Transformer Weight': f"{weights['transformer'][idx]:.4f}",
                            'Positional Weight': f"{weights['positional'][idx]:.4f}"
                        })
                    
                    if 'source_theta_degrees' in result:
                        for i, idx in enumerate(top_indices):
                            if idx < len(result['source_theta_degrees']):
                                top_data[i]['θ'] = f"{result['source_theta_degrees'][idx]:.1f}°"
                    
                    st.table(top_data)
                    
                    # Weight concentration metrics
                    st.markdown("#### 📊 Weight Concentration")
                    sorted_weights = np.sort(combined_weights)[::-1]
                    top3_concentration = np.sum(sorted_weights[:3]) / np.sum(combined_weights)
                    top5_concentration = np.sum(sorted_weights[:5]) / np.sum(combined_weights)
                    
                    col_conc1, col_conc2 = st.columns(2)
                    with col_conc1:
                        st.metric("Top 3 Concentration", f"{top3_concentration*100:.1f}%")
                    with col_conc2:
                        st.metric("Top 5 Concentration", f"{top5_concentration*100:.1f}%")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab6:
            st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Export options
                st.markdown("#### 📤 Export Formats")
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    # Export as JSON
                    if st.button("📄 Export as JSON", use_container_width=True, key="export_json"):
                        visualization_params = {
                            'colormap': colormap_name,
                            'visualization_type': visualization_type,
                            'spatial_locality_weight': locality_weight_factor,
                            'spatial_sigma': spatial_sigma
                        }
                        export_data = st.session_state.results_manager.prepare_export_data(
                            result, visualization_params
                        )
                        json_str, filename = st.session_state.results_manager.export_to_json(export_data)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True,
                            key="download_json"
                        )
                
                with col_e2:
                    # Export as CSV
                    if st.button("📊 Export as CSV", use_container_width=True, key="export_csv"):
                        csv_str, filename = st.session_state.results_manager.export_to_csv(result)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_str,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv"
                        )
                
                with col_e3:
                    # Export plot as PNG
                    if st.button("🖼️ Export Plot", use_container_width=True, key="export_plot"):
                        # Create a figure to export
                        fig_export = st.session_state.heatmap_visualizer.create_stress_heatmap(
                            result['fields']['von_mises'],
                            title=f"Von Mises Stress at θ={result['target_angle']:.1f}°",
                            cmap_name=colormap_name,
                            show_stats=False,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type']
                        )
                        buf = BytesIO()
                        fig_export.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        buf.seek(0)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"stress_heatmap_theta_{result['target_angle']:.1f}_{timestamp}.png"
                        st.download_button(
                            label="📥 Download PNG (300 DPI)",
                            data=buf,
                            file_name=filename,
                            mime="image/png",
                            use_container_width=True,
                            key="download_png"
                        )
                        plt.close(fig_export)
                
                # Bulk export
                st.markdown("---")
                st.markdown("#### 📦 Bulk Export All Components")
                if st.button("📦 Export All Components", use_container_width=True, type="secondary", key="export_all"):
                    # Create zip file with all components
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Export each component as CSV
                        for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            component_data = result['fields'][component]
                            df = pd.DataFrame(component_data)
                            csv_str = df.to_csv(index=False)
                            zip_file.writestr(f"{component}_theta_{result['target_angle']:.1f}.csv", csv_str)
                        
                        # Export metadata
                        metadata = {
                            'target_angle': result['target_angle'],
                            'target_params': result['target_params'],
                            'statistics': result['statistics'],
                            'weights': result['weights'],
                            'num_sources': result.get('num_sources', 0),
                            'source_theta_degrees': result.get('source_theta_degrees', []),
                            'source_distances': result.get('source_distances', []),
                            'shape': result['shape'],
                            'exported_at': datetime.now().isoformat(),
                            'interpolation_method': 'transformer_spatial',
                            'visualization_params': {
                                'colormap': colormap_name,
                                'visualization_type': visualization_type,
                                'spatial_locality_weight': locality_weight_factor,
                                'spatial_sigma': spatial_sigma
                            }
                        }
                        json_str = json.dumps(metadata, indent=2)
                        zip_file.writestr("metadata.json", json_str)
                        
                        # Export angular orientation plot
                        fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                            result['target_angle'],
                            result['target_params']['defect_type'],
                            figsize=(8, 8)
                        )
                        angular_buf = BytesIO()
                        fig_angular.savefig(angular_buf, format="png", dpi=300, bbox_inches="tight")
                        angular_buf.seek(0)
                        zip_file.writestr(f"angular_orientation_theta_{result['target_angle']:.1f}.png", angular_buf.getvalue())
                        plt.close(fig_angular)
                    
                    zip_buffer.seek(0)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"stress_components_theta_{result['target_angle']:.1f}_{timestamp}.zip"
                    st.download_button(
                        label="📥 Download ZIP (Complete Dataset)",
                        data=zip_buffer.getvalue(),
                        file_name=filename,
                        mime="application/zip",
                        use_container_width=True,
                        key="download_zip"
                    )
                
                # Export statistics
                with st.expander("📊 Export Statistics Table", expanded=False):
                    # Create statistics table
                    stats_data = []
                    for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        if component in result['statistics']:
                            stats = result['statistics'][component]
                            stats_data.append({
                                'Component': component.replace('_', ' ').title(),
                                'Max (GPa)': f"{stats['max']:.3f}",
                                'Min (GPa)': f"{stats.get('min', stats.get('max_compression', 0)):.3f}",
                                'Mean (GPa)': f"{stats['mean']:.3f}",
                                'Std (GPa)': f"{stats['std']:.3f}"
                            })
                    
                    if stats_data:
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True)
                        
                        # Export statistics as CSV
                        csv_stats = df_stats.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Statistics CSV",
                            data=csv_stats,
                            file_name=f"statistics_theta_{result['target_angle']:.1f}.csv",
                            mime="text/csv",
                            key="download_stats"
                        )
                    else:
                        st.info("No statistics data available for export.")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")

# Run the application
if __name__ == "__main__":
    main()
