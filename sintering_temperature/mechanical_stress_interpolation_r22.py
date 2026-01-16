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
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# Colormap options with 50+ choices
COLORMAP_OPTIONS = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'afmhot', 'gist_heat', 
                   'copper', 'summer', 'Wistia', 'spring', 'autumn', 'winter', 'bone', 'gray', 'pink', 'copper',
                   'gist_gray', 'gist_yarg', 'binary', 'gist_earth', 'terrain', 'ocean', 'gist_stern', 'gnuplot',
                   'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
                   'gist_ncar', 'hsv'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr', 
                  'RdGy', 'RdYlGn', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2', 
                    'Paired', 'Accent', 'Dark2'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
                             'hsv', 'turbo']
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
                    'hydrostatic': {
                        'max_tension': float(np.max(interpolated_fields['sigma_hydro'])),
                        'max_compression': float(np.min(interpolated_fields['sigma_hydro'])),
                        'mean': float(np.mean(interpolated_fields['sigma_hydro'])),
                        'std': float(np.std(interpolated_fields['sigma_hydro']))
                    },
                    'magnitude': {
                        'max': float(np.max(interpolated_fields['sigma_mag'])),
                        'mean': float(np.mean(interpolated_fields['sigma_mag'])),
                        'std': float(np.std(interpolated_fields['sigma_mag'])),
                        'min': float(np.min(interpolated_fields['sigma_mag']))
                    }
                },
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields)
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
# HEATMAP VISUALIZER WITH 50+ COLORMAPS
# =============================================

class HeatMapVisualizer:
    """Enhanced heat map visualizer with multiple colormap options"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map", 
                             cmap_name='viridis', figsize=(12, 10), 
                             colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                             show_stats=True):
        """Create enhanced heat map with chosen colormap"""
        
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
                      aspect='auto', interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=12, fontweight='bold')
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics annotation
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\n"
                         f"Min: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_interactive_heatmap(self, stress_field, title="Stress Heat Map",
                                  cmap_name='viridis', width=800, height=700):
        """Create interactive heatmap with Plotly"""
        
        # Create hover text
        hover_text = []
        for i in range(stress_field.shape[0]):
            row_text = []
            for j in range(stress_field.shape[1]):
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
                title="Stress (GPa)",
                titleside="right",
                titlefont=dict(size=14, family='Arial')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[heatmap_trace])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            width=width,
            height=height,
            xaxis=dict(
                title="X Position",
                gridcolor='rgba(100, 100, 100, 0.3)'
            ),
            yaxis=dict(
                title="Y Position",
                gridcolor='rgba(100, 100, 100, 0.3)'
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_comparison_heatmaps(self, stress_fields_dict, cmap_name='viridis',
                                  figsize=(18, 12), titles=None):
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
            cmap = plt.get_cmap(cmap_name)
            
            # Create heatmap
            im = ax.imshow(stress_field, cmap=cmap, aspect='auto', interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Stress (GPa)", fontsize=10)
            
            # Customize subplot
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        plt.suptitle("Stress Component Comparison", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def create_3d_surface_plot(self, stress_field, title="3D Stress Surface",
                              cmap_name='viridis', figsize=(14, 10)):
        """Create 3D surface plot of stress field"""
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(stress_field.shape[1])
        y = np.arange(stress_field.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Get colormap
        cmap = plt.get_cmap(cmap_name)
        
        # Normalize for coloring
        norm = Normalize(vmin=np.nanmin(stress_field), vmax=np.nanmax(stress_field))
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, stress_field, cmap=cmap, norm=norm,
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Stress (GPa)", fontsize=12)
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=12, labelpad=10)
        ax.set_ylabel('Y Position', fontsize=12, labelpad=10)
        ax.set_zlabel('Stress (GPa)', fontsize=12, labelpad=10)
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        return fig
    
    def get_colormap_preview(self, cmap_name, figsize=(12, 1)):
        """Generate preview of a colormap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.set_title(f"Colormap: {cmap_name}", fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add value labels
        ax.text(0, 0.5, f"{0:.1f}", transform=ax.transAxes,
                va='center', ha='right', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(1, 0.5, f"{1:.1f}", transform=ax.transAxes,
                va='center', ha='left', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, stress_fields, theta, defect_type,
                                      cmap_name='viridis', figsize=(20, 15)):
        """Create comprehensive dashboard with all stress components"""
        
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Von Mises stress (main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(stress_fields['von_mises'], cmap=cmap_name, aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Von Mises Stress (GPa)')
        ax1.set_title(f'Von Mises Stress at θ={theta}°\nDefect: {defect_type}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.2)
        
        # 2. Hydrostatic stress
        ax2 = fig.add_subplot(gs[0, 2])
        # Use diverging colormap for hydrostatic
        hydro_cmap = 'RdBu_r' if cmap_name in ['viridis', 'plasma', 'inferno'] else cmap_name
        im2 = ax2.imshow(stress_fields['sigma_hydro'], cmap=hydro_cmap, aspect='auto')
        plt.colorbar(im2, ax=ax2, label='Hydrostatic Stress (GPa)')
        ax2.set_title('Hydrostatic Stress', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        
        # 3. Stress magnitude
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(stress_fields['sigma_mag'], cmap=cmap_name, aspect='auto')
        plt.colorbar(im3, ax=ax3, label='Stress Magnitude (GPa)')
        ax3.set_title('Stress Magnitude', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        
        # 4. Histogram of von Mises
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.hist(stress_fields['von_mises'].flatten(), bins=50, alpha=0.7, color='blue')
        ax4.set_xlabel('Von Mises Stress (GPa)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Von Mises Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Histogram of hydrostatic
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(stress_fields['sigma_hydro'].flatten(), bins=50, alpha=0.7, color='green')
        ax5.set_xlabel('Hydrostatic Stress (GPa)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Hydrostatic Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Line profiles
        ax6 = fig.add_subplot(gs[2, 0])
        middle_row = stress_fields['von_mises'].shape[0] // 2
        middle_col = stress_fields['von_mises'].shape[1] // 2
        ax6.plot(stress_fields['von_mises'][middle_row, :], label='Von Mises', linewidth=2)
        ax6.plot(stress_fields['sigma_hydro'][middle_row, :], label='Hydrostatic', linewidth=2)
        ax6.plot(stress_fields['sigma_mag'][middle_row, :], label='Magnitude', linewidth=2)
        ax6.set_xlabel('X Position')
        ax6.set_ylabel('Stress (GPa)')
        ax6.set_title(f'Line Profile at Row {middle_row}', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Statistics table
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('off')
        
        # Prepare statistics
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
        
        ax7.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax7.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax7.set_title('Stress Statistics', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Comprehensive Stress Analysis - θ={theta}°, {defect_type}', 
                    fontsize=18, fontweight='bold', y=0.98)
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
                'num_sources': result.get('num_sources', 0)
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
        font-size: 3.0rem !important;
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
        font-size: 1.8rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1rem;
        border-radius: 0.6rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px 5px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
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
                    st.write(f"**Solution {i+1}:** {params.get('defect_type', 'Unknown')}")
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
            index=0
        )
        
        colormap_name = st.selectbox(
            "Select Colormap",
            COLORMAP_OPTIONS[colormap_category],
            index=0
        )
        
        visualization_type = st.selectbox(
            "Visualization Type",
            ["2D Heatmap", "Interactive Heatmap", "3D Surface", "Comparison View", "Comprehensive Dashboard"],
            index=0
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
        
        ### Visualization Options
        - 50+ colormaps including jet, turbo, rainbow, inferno
        - 2D heatmaps, 3D surfaces, interactive plots
        - Comprehensive dashboards with statistics
        """)
    else:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Results",
            "📈 Visualization",
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
                    hydro_stats = result['statistics']['hydrostatic']
                    st.metric("Hydrostatic Range", 
                             f"{hydro_stats['max_tension']:.3f}/{hydro_stats['max_compression']:.3f} GPa",
                             f"Mean: {hydro_stats['mean']:.3f} GPa")
                
                with col3:
                    mag_stats = result['statistics']['magnitude']
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
                    im = ax.imshow(result['fields'][comp], cmap='viridis', aspect='auto')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title(title, fontsize=10)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.grid(True, alpha=0.2)
                
                plt.suptitle(f"Stress Fields at θ={result['target_angle']:.1f}°", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_preview)
                plt.close(fig_preview)
                
            else:
                st.info("👈 Configure parameters and click 'Perform Transformer Interpolation' to generate results")
        
        with tab2:
            st.markdown('<h2 class="section-header">📈 Stress Field Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
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
                        title=f"{component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                        cmap_name=colormap_name,
                        colorbar_label=f"{component_names[stress_component]} (GPa)"
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Colormap preview
                    with st.expander("🎨 Colormap Preview"):
                        fig_preview = st.session_state.heatmap_visualizer.get_colormap_preview(colormap_name)
                        st.pyplot(fig_preview)
                        plt.close(fig_preview)
                
                elif visualization_type == "Interactive Heatmap":
                    fig = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                        result['fields'][stress_component],
                        title=f"{component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                        cmap_name=colormap_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif visualization_type == "3D Surface":
                    fig = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                        result['fields'][stress_component],
                        title=f"3D {component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                        cmap_name=colormap_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                
                elif visualization_type == "Comparison View":
                    comparison_fields = {
                        'Von Mises': result['fields']['von_mises'],
                        'Hydrostatic': result['fields']['sigma_hydro'],
                        'Magnitude': result['fields']['sigma_mag']
                    }
                    
                    fig = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                        comparison_fields,
                        cmap_name=colormap_name
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
                with st.expander("📊 Detailed Statistics", expanded=False):
                    stats = result['statistics'][stress_component]
                    
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
                    ax_hist.hist(result['fields'][stress_component].flatten(), bins=50, alpha=0.7)
                    ax_hist.set_xlabel(f'{component_names[stress_component]} (GPa)')
                    ax_hist.set_ylabel('Frequency')
                    ax_hist.set_title(f'Distribution of {component_names[stress_component]}')
                    ax_hist.grid(True, alpha=0.3)
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">🔍 Weights Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                weights = result['weights']
                
                # Weights visualization
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    # Transformer weights
                    fig_trans, ax_trans = plt.subplots(figsize=(8, 4))
                    ax_trans.bar(range(len(weights['transformer'])), weights['transformer'])
                    ax_trans.set_xlabel('Source Index')
                    ax_trans.set_ylabel('Weight')
                    ax_trans.set_title('Transformer Attention Weights')
                    ax_trans.grid(True, alpha=0.3)
                    st.pyplot(fig_trans)
                    plt.close(fig_trans)
                
                with col_w2:
                    # Positional weights
                    fig_pos, ax_pos = plt.subplots(figsize=(8, 4))
                    ax_pos.bar(range(len(weights['positional'])), weights['positional'])
                    ax_pos.set_xlabel('Source Index')
                    ax_pos.set_ylabel('Weight')
                    ax_pos.set_title('Spatial Locality Weights')
                    ax_pos.grid(True, alpha=0.3)
                    st.pyplot(fig_pos)
                    plt.close(fig_pos)
                
                # Combined weights
                fig_comb, ax_comb = plt.subplots(figsize=(10, 4))
                x = range(len(weights['combined']))
                width = 0.25
                
                ax_comb.bar([i - width for i in x], weights['transformer'], width, label='Transformer', alpha=0.7)
                ax_comb.bar(x, weights['positional'], width, label='Positional', alpha=0.7)
                ax_comb.bar([i + width for i in x], weights['combined'], width, label='Combined', alpha=0.7)
                
                ax_comb.set_xlabel('Source Index')
                ax_comb.set_ylabel('Weight')
                ax_comb.set_title('Weight Comparison')
                ax_comb.legend()
                ax_comb.grid(True, alpha=0.3)
                st.pyplot(fig_comb)
                plt.close(fig_comb)
                
                # Weight statistics
                with st.expander("📊 Weight Statistics", expanded=False):
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Transformer Weight Entropy", 
                                 f"{_calculate_entropy(weights['transformer']):.3f}")
                    with col_stat2:
                        st.metric("Positional Weight Entropy",
                                 f"{_calculate_entropy(weights['positional']):.3f}")
                    with col_stat3:
                        st.metric("Combined Weight Entropy",
                                 f"{_calculate_entropy(weights['combined']):.3f}")
                    
                    # Top contributors
                    st.write("**Top 5 Contributing Sources:**")
                    combined_weights = np.array(weights['combined'])
                    top_indices = np.argsort(combined_weights)[-5:][::-1]
                    
                    for i, idx in enumerate(top_indices):
                        st.write(f"{i+1}. Source {idx}: {combined_weights[idx]:.3f} "
                                f"(Transformer: {weights['transformer'][idx]:.3f}, "
                                f"Positional: {weights['positional'][idx]:.3f})")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Export options
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
                        # Create a figure to export
                        fig_export = st.session_state.heatmap_visualizer.create_stress_heatmap(
                            result['fields']['von_mises'],
                            title=f"Von Mises Stress at θ={result['target_angle']:.1f}°",
                            cmap_name=colormap_name,
                            show_stats=False
                        )
                        
                        buf = BytesIO()
                        fig_export.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        buf.seek(0)
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"stress_heatmap_theta_{result['target_angle']:.1f}_{timestamp}.png"
                        
                        st.download_button(
                            label="📥 Download PNG",
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
                        
                        # Export metadata
                        metadata = {
                            'target_angle': result['target_angle'],
                            'target_params': result['target_params'],
                            'statistics': result['statistics'],
                            'weights': result['weights'],
                            'exported_at': datetime.now().isoformat()
                        }
                        json_str = json.dumps(metadata, indent=2)
                        zip_file.writestr("metadata.json", json_str)
                    
                    zip_buffer.seek(0)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"stress_components_theta_{result['target_angle']:.1f}_{timestamp}.zip"
                    
                    st.download_button(
                        label="📥 Download ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=filename,
                        mime="application/zip",
                        use_container_width=True,
                        key="download_zip"
                    )
                
                # Export statistics
                with st.expander("📈 Export Statistics Table", expanded=False):
                    # Create statistics table
                    stats_data = []
                    for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        component_stats = result['statistics'][component]
                        stats_data.append({
                            'Component': component.replace('_', ' ').title(),
                            'Max (GPa)': f"{component_stats['max']:.3f}",
                            'Min (GPa)': f"{component_stats.get('min', 0):.3f}",
                            'Mean (GPa)': f"{component_stats['mean']:.3f}",
                            'Std (GPa)': f"{component_stats['std']:.3f}"
                        })
                    
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
                st.info("No interpolation results available. Please perform interpolation first.")

# Run the application
if __name__ == "__main__":
    main()
