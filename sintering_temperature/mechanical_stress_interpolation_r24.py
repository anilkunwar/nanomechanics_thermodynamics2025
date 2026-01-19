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
# TRANSFORMER SPATIAL INTERPOLATOR
# =============================================

class TransformerSpatialInterpolator:
    """Transformer-inspired stress interpolator with spatial locality regularization"""
    
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        
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
        """Compute spatial locality weights based on parameter similarity"""
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
            
            # Apply Gaussian kernel
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
        """Interpolate full spatial stress fields using transformer attention"""
        
        if not sources:
            st.warning("No sources provided for interpolation.")
            return None
        
        try:
            # Extract source parameters and fields
            source_params = []
            source_fields = []
            
            for src in sources:
                if 'params' not in src or 'history' not in src:
                    st.warning(f"Skipping source: missing params or history")
                    continue
                
                source_params.append(src['params'])
                
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
                            'sigma_mag': mag
                        })
                    else:
                        st.warning(f"Skipping source: no stress fields found")
                        continue
                else:
                    st.warning(f"Skipping source: invalid history")
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
                        if field.shape != target_shape:
                            # Resize using interpolation
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            resized[key] = zoom(field, factors, order=1)
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
            
            # Compute positional weights
            pos_weights = self.compute_positional_weights(source_params, target_params)
            pos_weights = pos_weights / pos_weights.sum() if pos_weights.sum() > 0 else np.ones_like(pos_weights) / len(pos_weights)
            
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
            transformer_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().numpy()
            
            # Combine positional and transformer weights
            combined_weights = 0.7 * transformer_weights + 0.3 * pos_weights
            combined_weights = combined_weights / combined_weights.sum()
            
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
            for src in source_params:
                theta_rad = src.get('theta', 0.0)
                source_theta_degrees.append(np.degrees(theta_rad))
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'transformer': transformer_weights.tolist(),
                    'positional': pos_weights.tolist(),
                    'combined': combined_weights.tolist()
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
                'source_theta_degrees': source_theta_degrees
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

# =============================================
# ENHANCED HEATMAP VISUALIZER WITH ALL IMPROVEMENTS
# =============================================

class HeatMapVisualizer:
    """Enhanced heat map visualizer with multiple colormap options and publication styling"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map", 
                             cmap_name='viridis', figsize=(12, 10), 
                             colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                             show_stats=True, target_angle=None, defect_type=None):
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
        
        # Create heatmap with equal aspect ratio for publication quality
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='equal', interpolation='bilinear', origin='lower')
        
        # Add colorbar with enhanced styling
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
                    #title="X Position",
                    #titlefont=dict(size=18, family='Arial', color='black'),
                    title=dict(text="X Position", font=dict(size=18, family="Arial", color="black")),
                    tickfont=dict(size=14, family='Arial'),
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    #title="Y Position",
                    #titlefont=dict(size=18, family='Arial', color='black'),
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
                        #title="X Position",
                        #titlefont=dict(size=18, color='black'),
                        title=dict(text="X Position", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    yaxis=dict(
                        #title="Y Position",
                        #titlefont=dict(size=18, color='black'),
                        title=dict(text="Y Position", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    zaxis=dict(
                        #title="Stress (GPa)",
                        #titlefont=dict(size=18, color='black'),
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
                'source_theta_degrees': result.get('source_theta_degrees', [])
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
    return -np.sum(weights * np.log(weights))

# =============================================
# MAIN APPLICATION
# =============================================

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Transformer Stress Interpolation",
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
    st.markdown('<h1 class="main-header">🤖 Transformer Stress Field Interpolation</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Physics-aware stress interpolation using transformer architecture with spatial locality regularization.</strong><br>
    • Load simulation files from numerical_solutions directory<br>
    • Interpolate stress fields at custom polar angles (default: 54.7°)<br>
    • Visualize von Mises, hydrostatic, and stress magnitude fields<br>
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
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator()
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Solutions", use_container_width=True):
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
            with st.expander(f"📊 Loaded Solutions ({len(st.session_state.solutions)})"):
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    params = sol.get('params', {})
                    theta_deg = np.degrees(params.get('theta', 0))
                    st.write(f"**Solution {i+1}:** {params.get('defect_type', 'Unknown')} at θ={theta_deg:.1f}°")
                if len(st.session_state.solutions) > 5:
                    st.write(f"... and {len(st.session_state.solutions) - 5} more")
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle
        custom_theta = st.number_input(
            "Custom Polar Angle θ (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=54.7,
            step=0.1,
            help="Set custom polar angle for interpolation (default: 54.7°)"
        )
        
        # Defect type
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for interpolation"
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        eps0 = eigen_strains[defect_type]
        
        # Shape
        shape = st.selectbox(
            "Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
            index=0
        )
        
        # Kappa parameter
        kappa = st.slider(
            "Kappa Parameter",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.1,
            help="Material parameter"
        )
        
        # Transformer parameters
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
        
        # Enhanced visualization options
        visualization_type = st.selectbox(
            "Visualization Type",
            ["2D Heatmap", "Interactive Heatmap", "3D Surface", "Interactive 3D Surface", 
             "Comparison View", "Comprehensive Dashboard", "Angular Orientation", "Target Parameters"],
            index=0,
            help="Select visualization type with enhanced options"
        )
        
        # Interpolation button
        st.markdown("---")
        if st.button("🚀 Perform Transformer Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                # Update transformer parameters
                st.session_state.transformer_interpolator.spatial_sigma = spatial_sigma
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
                        else:
                            st.error("❌ Failed to interpolate stress fields. Check data compatibility.")
                    except Exception as e:
                        st.error(f"❌ Error during interpolation: {str(e)}")
    
    # Main content
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        
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
        ## 🚀 Quick Start Guide
        
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
        
        ### Stress Components
        - **Von Mises Stress (σ_vm)**: Equivalent tensile stress
        - **Hydrostatic Stress (σ_h)**: Mean normal stress (trace/3)
        - **Stress Magnitude (σ_mag)**: Overall stress intensity
        
        ### Enhanced Visualization Options
        - 50+ colormaps including jet, turbo, rainbow, inferno
        - Publication-ready figures with consistent aspect ratios
        - Angular orientation polar plots
        - Interactive 3D surfaces with Plotly
        - Comprehensive dashboards with statistics
        """)
    else:
        # Enhanced tabs with Target Parameters
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Results",
            "📈 Visualization",
            "🎯 Target Parameters",
            "🔍 Weights Analysis",
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
                with st.expander("🔧 Interpolation Parameters", expanded=True):
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
                    im = ax.imshow(result['fields'][comp], cmap='viridis', aspect='equal', interpolation='bilinear', origin='lower')
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
                st.info("👈 Configure parameters and click 'Perform Transformer Interpolation' to generate results")
        
        with tab2:
            st.markdown('<h2 class="section-header">📈 Enhanced Stress Field Visualization</h2>', unsafe_allow_html=True)
            
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
                    
                elif visualization_type == "Target Parameters":
                    st.markdown("#### 🎯 Target Query Parameters Display")
                    
                    # Create a comprehensive target parameters display
                    col_tp1, col_tp2 = st.columns(2)
                    
                    with col_tp1:
                        st.markdown('<div class="param-table">', unsafe_allow_html=True)
                        st.markdown("##### 🔧 Target Parameters")
                        
                        target_params = result['target_params']
                        
                        # Display parameters in a nice format
                        for key, value in target_params.items():
                            if key == 'theta':
                                value_display = f"{np.degrees(value):.1f}°"
                            elif key == 'eps0':
                                value_display = f"{value:.3f}"
                            elif key == 'kappa':
                                value_display = f"{value:.2f}"
                            else:
                                value_display = str(value)
                            
                            st.markdown(f"""
                            <div style="padding: 8px 0; border-bottom: 1px solid #e0e0e0;">
                                <span class="param-key">{key.replace('_', ' ').title()}:</span> 
                                <span class="param-value">{value_display}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_tp2:
                        # Show angular visualization
                        fig_small_polar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
                        theta_rad = np.radians(result['target_angle'])
                        ax.arrow(theta_rad, 0.7, 0, 0.5, width=0.02, color='red', alpha=0.8)
                        ax.arrow(np.radians(54.7), 0.7, 0, 0.5, width=0.02, color='blue', alpha=0.5)
                        ax.set_title(f'θ = {result["target_angle"]:.1f}°', fontsize=16)
                        ax.set_theta_zero_location('N')
                        ax.set_theta_direction(-1)
                        ax.set_ylim(0, 1.3)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig_small_polar)
                        plt.close(fig_small_polar)
                    
                else:
                    # Component selection for other visualization types
                    if visualization_type not in ["Comparison View", "Comprehensive Dashboard"]:
                        stress_component = st.selectbox(
                            "Select Stress Component",
                            ["von_mises", "sigma_hydro", "sigma_mag"],
                            index=0,
                            key="viz_component"
                        )
                    else:
                        stress_component = None
                    
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
                        fig = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                            result['fields'][stress_component],
                            title=f"3D {component_names[stress_component]}",
                            cmap_name=colormap_name,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type']
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    elif visualization_type == "Interactive 3D Surface":
                        fig = st.session_state.heatmap_visualizer.create_interactive_3d_surface(
                            result['fields'][stress_component],
                            title=f"3D {component_names[stress_component]}",
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
                
                # Additional statistics for all visualization types except Angular Orientation
                if visualization_type not in ["Angular Orientation", "Target Parameters"]:
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
                            if stress_component:
                                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                                ax_hist.hist(result['fields'][stress_component].flatten(), bins=50, alpha=0.7, edgecolor='black')
                                ax_hist.set_xlabel(f'{component_names[stress_component]} (GPa)', fontsize=14)
                                ax_hist.set_ylabel('Frequency', fontsize=14)
                                ax_hist.set_title(f'Distribution of {component_names[stress_component]}', fontsize=16, fontweight='bold')
                                ax_hist.grid(True, alpha=0.3)
                                st.pyplot(fig_hist)
                                plt.close(fig_hist)
                        else:
                            st.warning(f"Statistics not available for component: {stress_component}")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">🎯 Target Query Parameters</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                target_params = result['target_params']
                
                # Create a comprehensive target parameters display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="param-table">', unsafe_allow_html=True)
                    st.markdown("##### 🔧 Target Interpolation Parameters")
                    
                    # Display parameters in a nice table format
                    param_data = []
                    
                    # Extract and format parameters
                    for key, value in target_params.items():
                        if key == 'theta':
                            display_key = 'Polar Angle θ'
                            display_value = f"{np.degrees(value):.1f}°"
                            icon = "📐"
                        elif key == 'eps0':
                            display_key = 'Eigen Strain ε₀'
                            display_value = f"{value:.3f}"
                            icon = "⚡"
                        elif key == 'kappa':
                            display_key = 'Material Parameter κ'
                            display_value = f"{value:.2f}"
                            icon = "🧲"
                        elif key == 'defect_type':
                            display_key = 'Defect Type'
                            display_value = value
                            icon = "🔬"
                        elif key == 'shape':
                            display_key = 'Shape'
                            display_value = value
                            icon = "🟦"
                        else:
                            display_key = key.replace('_', ' ').title()
                            display_value = str(value)
                            icon = "⚙️"
                        
                        param_data.append({
                            'icon': icon,
                            'parameter': display_key,
                            'value': display_value
                        })
                    
                    # Display as metrics
                    for param in param_data:
                        st.metric(f"{param['icon']} {param['parameter']}", param['value'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional metrics
                    st.markdown('<div class="param-table">', unsafe_allow_html=True)
                    st.markdown("##### 📊 Interpolation Metrics")
                    
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Target Angle θ", f"{result['target_angle']:.1f}°")
                        st.metric("Number of Sources", f"{result.get('num_sources', 0)}")
                    
                    with col_m2:
                        habit_deviation = abs(result['target_angle'] - 54.7)
                        st.metric("Deviation from Habit Plane", f"{habit_deviation:.1f}°")
                        st.metric("Field Size", f"{result['shape'][0]}×{result['shape'][1]}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Angular orientation visualization
                    st.markdown("##### 🧭 Angular Orientation")
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        result['target_params']['defect_type'],
                        figsize=(8, 8)
                    )
                    st.pyplot(fig_angular)
                    plt.close(fig_angular)
                    
                    # Habit plane information
                    st.markdown("##### 🔬 Habit Plane Reference")
                    st.info("""
                    **Habit Plane Orientation:** 54.7°
                    
                    This is the preferential crystallographic orientation where defects 
                    typically form in face-centered cubic (FCC) materials. The angular 
                    deviation from this plane influences defect formation energy and 
                    stress field characteristics.
                    """)
                
                # Source simulation information
                st.markdown("##### 🔗 Source Simulations Information")
                
                if 'source_theta_degrees' in result and result['source_theta_degrees']:
                    source_thetas = result['source_theta_degrees']
                    weights = result['weights']['combined']
                    
                    # Create a bar plot showing source contributions with theta labels
                    fig_sources, ax_sources = plt.subplots(figsize=(12, 6))
                    x_pos = np.arange(len(source_thetas))
                    bars = ax_sources.bar(x_pos, weights, alpha=0.7, color='steelblue', edgecolor='black')
                    
                    # Add theta labels to bars
                    for i, (bar, theta) in enumerate(zip(bars, source_thetas)):
                        height = bar.get_height()
                        ax_sources.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                      f'θ={theta:.0f}°', ha='center', va='bottom', 
                                      fontsize=10, fontweight='bold', rotation=90)
                    
                    ax_sources.set_xlabel('Source Index', fontsize=14)
                    ax_sources.set_ylabel('Weight', fontsize=14)
                    #ax_sources.set_title('Source Contributions with Angular Orientation', fontsize=16, fontweight='bold')
                    ax_sources.grid(True, alpha=0.3, axis='y')
                    
                    # Highlight top 3 contributors
                    if len(weights) >= 3:
                        top_indices = np.argsort(weights)[-3:][::-1]
                        for idx in top_indices:
                            bars[idx].set_color('red')
                            bars[idx].set_alpha(0.9)
                    
                    st.pyplot(fig_sources)
                    plt.close(fig_sources)
                    
                    # Display source statistics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        avg_source_theta = np.mean(source_thetas)
                        st.metric("Average Source θ", f"{avg_source_theta:.1f}°")
                    
                    with col_s2:
                        min_source_theta = np.min(source_thetas)
                        st.metric("Minimum Source θ", f"{min_source_theta:.1f}°")
                    
                    with col_s3:
                        max_source_theta = np.max(source_thetas)
                        st.metric("Maximum Source θ", f"{max_source_theta:.1f}°")
                    
                    # Angular distribution statistics
                    st.markdown("##### 📐 Angular Distribution Statistics")
                    
                    # Calculate angular distances from target
                    angular_distances = []
                    for source_theta in source_thetas:
                        dist = abs(source_theta - result['target_angle'])
                        # Handle periodicity (360° circle)
                        dist = min(dist, 360 - dist)
                        angular_distances.append(dist)
                    
                    col_d1, col_d2, col_d3 = st.columns(3)
                    with col_d1:
                        avg_distance = np.mean(angular_distances)
                        st.metric("Avg Angular Distance", f"{avg_distance:.1f}°")
                    
                    with col_d2:
                        min_distance = np.min(angular_distances)
                        st.metric("Min Angular Distance", f"{min_distance:.1f}°")
                    
                    with col_d3:
                        max_distance = np.max(angular_distances)
                        st.metric("Max Angular Distance", f"{max_distance:.1f}°")
                    
                    # Top 5 closest sources by angle
                    if len(source_thetas) >= 5:
                        st.markdown("##### 🏆 Top 5 Closest Sources by Angle")
                        
                        # Calculate angular differences
                        angular_diffs = []
                        for i, source_theta in enumerate(source_thetas):
                            diff = abs(source_theta - result['target_angle'])
                            diff = min(diff, 360 - diff)  # Handle periodicity
                            angular_diffs.append((i, source_theta, diff, weights[i]))
                        
                        # Sort by angular difference
                        angular_diffs.sort(key=lambda x: x[2])
                        
                        # Display table
                        diff_data = []
                        for i, (idx, theta, diff, weight) in enumerate(angular_diffs[:5]):
                            diff_data.append({
                                'Rank': i+1,
                                'Source Index': idx,
                                'θ': f"{theta:.1f}°",
                                'Δθ': f"{diff:.1f}°",
                                'Weight': f"{weight:.3f}"
                            })
                        
                        df_diffs = pd.DataFrame(diff_data)
                        st.dataframe(df_diffs, use_container_width=True)
                else:
                    st.info("No source theta information available.")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">🔍 Weights Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                weights = result['weights']
                
                # Enhanced weights visualization with theta labels
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    # Transformer weights with theta labels
                    fig_trans, ax_trans = plt.subplots(figsize=(10, 5))
                    x_pos = np.arange(len(weights['transformer']))
                    bars = ax_trans.bar(x_pos, weights['transformer'], alpha=0.7, color='orange', edgecolor='black')
                    
                    # Add theta labels if available
                    if 'source_theta_degrees' in result:
                        source_thetas = result['source_theta_degrees']
                        for i, (bar, theta) in enumerate(zip(bars, source_thetas)):
                            height = bar.get_height()
                            if height > 0.05:  # Only label significant bars
                                ax_trans.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                            f'θ={theta:.0f}°', ha='center', va='bottom', 
                                            fontsize=9, fontweight='bold', rotation=90)
                    
                    ax_trans.set_xlabel('Source Index', fontsize=14)
                    ax_trans.set_ylabel('Weight', fontsize=14)
                    ax_trans.set_title('Transformer Attention Weights', fontsize=16, fontweight='bold')
                    ax_trans.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_trans)
                    plt.close(fig_trans)
                
                with col_w2:
                    # Positional weights with theta labels
                    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
                    bars = ax_pos.bar(x_pos, weights['positional'], alpha=0.7, color='green', edgecolor='black')
                    
                    # Add theta labels if available
                    if 'source_theta_degrees' in result:
                        for i, (bar, theta) in enumerate(zip(bars, source_thetas)):
                            height = bar.get_height()
                            if height > 0.05:  # Only label significant bars
                                ax_pos.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                          f'θ={theta:.0f}°', ha='center', va='bottom', 
                                          fontsize=9, fontweight='bold', rotation=90)
                    
                    ax_pos.set_xlabel('Source Index', fontsize=14)
                    ax_pos.set_ylabel('Weight', fontsize=14)
                    ax_pos.set_title('Spatial Locality Weights', fontsize=16, fontweight='bold')
                    ax_pos.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_pos)
                    plt.close(fig_pos)
                
                # Combined weights with enhanced visualization
                fig_comb, ax_comb = plt.subplots(figsize=(12, 6))
                x = range(len(weights['combined']))
                width = 0.25
                
                ax_comb.bar([i - width for i in x], weights['transformer'], width, label='Transformer', alpha=0.7, color='orange')
                ax_comb.bar(x, weights['positional'], width, label='Positional', alpha=0.7, color='green')
                ax_comb.bar([i + width for i in x], weights['combined'], width, label='Combined', alpha=0.7, color='steelblue')
                
                # Add theta labels on x-axis if available
                if 'source_theta_degrees' in result:
                    ax_comb.set_xticks(x)
                    ax_comb.set_xticklabels([f'{t:.0f}°' for t in source_thetas], rotation=45, fontsize=10)
                
                ax_comb.set_xlabel('Source Index (θ values shown below)', fontsize=14)
                ax_comb.set_ylabel('Weight', fontsize=14)
                ax_comb.set_title('Weight Comparison', fontsize=18, fontweight='bold')
                ax_comb.legend(fontsize=12)
                ax_comb.grid(True, alpha=0.3)
                st.pyplot(fig_comb)
                plt.close(fig_comb)
                
                # Weight statistics
                with st.expander("📊 Advanced Weight Statistics", expanded=True):
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        entropy_trans = _calculate_entropy(weights['transformer'])
                        st.metric("Transformer Weight Entropy", 
                                 f"{entropy_trans:.3f}",
                                 "Higher = more uniform distribution")
                    
                    with col_stat2:
                        entropy_pos = _calculate_entropy(weights['positional'])
                        st.metric("Positional Weight Entropy",
                                 f"{entropy_pos:.3f}")
                    
                    with col_stat3:
                        entropy_comb = _calculate_entropy(weights['combined'])
                        st.metric("Combined Weight Entropy",
                                 f"{entropy_comb:.3f}")
                    
                    # Top contributors with enhanced information
                    st.markdown("##### 🏆 Top 5 Contributing Sources")
                    
                    combined_weights = np.array(weights['combined'])
                    top_indices = np.argsort(combined_weights)[-5:][::-1]
                    
                    top_data = []
                    for i, idx in enumerate(top_indices):
                        source_info = {
                            'Rank': i+1,
                            'Source Index': idx,
                            'Combined Weight': f"{combined_weights[idx]:.4f}",
                            'Transformer Weight': f"{weights['transformer'][idx]:.4f}",
                            'Positional Weight': f"{weights['positional'][idx]:.4f}"
                        }
                        
                        # Add theta information if available
                        if 'source_theta_degrees' in result and idx < len(result['source_theta_degrees']):
                            source_info['θ'] = f"{result['source_theta_degrees'][idx]:.1f}°"
                            angular_diff = abs(result['source_theta_degrees'][idx] - result['target_angle'])
                            angular_diff = min(angular_diff, 360 - angular_diff)
                            source_info['Δθ'] = f"{angular_diff:.1f}°"
                        
                        top_data.append(source_info)
                    
                    df_top = pd.DataFrame(top_data)
                    st.dataframe(df_top, use_container_width=True)
                    
                    # Weight distribution statistics
                    st.markdown("##### 📈 Weight Distribution Analysis")
                    
                    col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
                    with col_dist1:
                        max_weight = np.max(combined_weights)
                        st.metric("Maximum Weight", f"{max_weight:.4f}")
                    
                    with col_dist2:
                        min_weight = np.min(combined_weights)
                        st.metric("Minimum Weight", f"{min_weight:.4f}")
                    
                    with col_dist3:
                        mean_weight = np.mean(combined_weights)
                        st.metric("Mean Weight", f"{mean_weight:.4f}")
                    
                    with col_dist4:
                        std_weight = np.std(combined_weights)
                        st.metric("Std Dev", f"{std_weight:.4f}")
                    
                    # Weight concentration metrics
                    st.markdown("##### 🎯 Weight Concentration")
                    
                    # Calculate top-3 weight concentration
                    sorted_weights = np.sort(combined_weights)[::-1]
                    top3_concentration = np.sum(sorted_weights[:3]) / np.sum(combined_weights)
                    top5_concentration = np.sum(sorted_weights[:5]) / np.sum(combined_weights)
                    
                    col_conc1, col_conc2 = st.columns(2)
                    with col_conc1:
                        st.metric("Top 3 Concentration", f"{top3_concentration*100:.1f}%")
                    
                    with col_conc2:
                        st.metric("Top 5 Concentration", f"{top5_concentration*100:.1f}%")
                    
                    # Weight distribution histogram
                    fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
                    ax_dist.hist(combined_weights, bins=20, alpha=0.7, color='purple', edgecolor='black')
                    ax_dist.set_xlabel('Weight Value', fontsize=14)
                    ax_dist.set_ylabel('Frequency', fontsize=14)
                    ax_dist.set_title('Combined Weight Distribution', fontsize=16, fontweight='bold')
                    ax_dist.grid(True, alpha=0.3)
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab5:
            st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Export options
                st.markdown("##### 💾 Export Formats")
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    # Export as JSON
                    if st.button("💾 Export as JSON", use_container_width=True, key="export_json"):
                        visualization_params = {
                            'colormap': colormap_name,
                            'visualization_type': visualization_type
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
                        # Create a publication-ready figure to export
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
                st.markdown("##### 📦 Bulk Export All Components")
                
                if st.button("🚀 Export All Components", use_container_width=True, type="secondary", key="export_all"):
                    # Create zip file with all components
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Export each component as CSV
                        for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            component_data = result['fields'][component]
                            df = pd.DataFrame(component_data)
                            csv_str = df.to_csv(index=False)
                            zip_file.writestr(f"{component}_theta_{result['target_angle']:.1f}.csv", csv_str)
                        
                        # Export metadata with enhanced information
                        metadata = {
                            'target_angle': result['target_angle'],
                            'target_params': result['target_params'],
                            'statistics': result['statistics'],
                            'weights': result['weights'],
                            'num_sources': result.get('num_sources', 0),
                            'source_theta_degrees': result.get('source_theta_degrees', []),
                            'shape': result['shape'],
                            'exported_at': datetime.now().isoformat(),
                            'interpolation_method': 'transformer_spatial',
                            'visualization_params': {
                                'colormap': colormap_name,
                                'visualization_type': visualization_type
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
                with st.expander("📈 Export Statistics Table", expanded=False):
                    try:
                        # Create statistics table with proper key mapping
                        stats_data = []
                        
                        # Define component mapping for statistics
                        component_mapping = [
                            {'field_name': 'von_mises', 'display_name': 'Von Mises Stress', 'stat_key': 'von_mises'},
                            {'field_name': 'sigma_hydro', 'display_name': 'Hydrostatic Stress', 'stat_key': 'sigma_hydro'},
                            {'field_name': 'sigma_mag', 'display_name': 'Stress Magnitude', 'stat_key': 'sigma_mag'}
                        ]
                        
                        for comp_info in component_mapping:
                            stat_key = comp_info['stat_key']
                            display_name = comp_info['display_name']
                            
                            if stat_key in result['statistics']:
                                component_stats = result['statistics'][stat_key]
                                
                                # Extract stats with safe access
                                max_val = component_stats.get('max', 0.0)
                                min_val = component_stats.get('min', component_stats.get('max_compression', 0.0))
                                if 'max_compression' in component_stats:
                                    min_val = component_stats['max_compression']
                                elif 'min' in component_stats:
                                    min_val = component_stats['min']
                                
                                mean_val = component_stats.get('mean', 0.0)
                                std_val = component_stats.get('std', 0.0)
                                
                                stats_data.append({
                                    'Component': display_name,
                                    'Max (GPa)': f"{max_val:.3f}",
                                    'Min (GPa)': f"{min_val:.3f}",
                                    'Mean (GPa)': f"{mean_val:.3f}",
                                    'Std (GPa)': f"{std_val:.3f}"
                                })
                            else:
                                st.warning(f"Statistics not found for {stat_key}")
                        
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
                            
                    except KeyError as e:
                        st.error(f"KeyError accessing statistics: {e}")
                        st.info("Statistics structure may be different than expected.")
                        st.json(result['statistics'] if 'statistics' in result else {})
                    except Exception as e:
                        st.error(f"Error creating statistics table: {e}")
                
                # Export target parameters
                with st.expander("🎯 Export Target Parameters", expanded=False):
                    # Create target parameters table
                    target_params = result['target_params']
                    params_data = []
                    
                    for key, value in target_params.items():
                        if key == 'theta':
                            display_value = f"{np.degrees(value):.1f}°"
                        elif key == 'eps0':
                            display_value = f"{value:.3f}"
                        elif key == 'kappa':
                            display_value = f"{value:.2f}"
                        else:
                            display_value = str(value)
                        
                        params_data.append({
                            'Parameter': key.replace('_', ' ').title(),
                            'Value': display_value
                        })
                    
                    # Add interpolation parameters
                    params_data.append({
                        'Parameter': 'Target Angle θ',
                        'Value': f"{result['target_angle']:.1f}°"
                    })
                    params_data.append({
                        'Parameter': 'Number of Sources',
                        'Value': str(result.get('num_sources', 0))
                    })
                    params_data.append({
                        'Parameter': 'Field Shape',
                        'Value': f"{result['shape'][0]} × {result['shape'][1]}"
                    })
                    
                    df_params = pd.DataFrame(params_data)
                    st.dataframe(df_params, use_container_width=True)
                    
                    # Export parameters as CSV
                    csv_params = df_params.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Parameters CSV",
                        data=csv_params,
                        file_name=f"parameters_theta_{result['target_angle']:.1f}.csv",
                        mime="text/csv",
                        key="download_params"
                    )
            else:
                st.info("No interpolation results available. Please perform interpolation first.")

# Run the application
if __name__ == "__main__":
    main()
