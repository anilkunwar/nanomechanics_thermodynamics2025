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
from scipy.ndimage import zoom, rotate, map_coordinates
from scipy.interpolate import interp1d, RegularGridInterpolator
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PHYSICS CONSTANTS FOR DIFFUSION CALCULATION
# =============================================
# Material parameters for Silver (Ag)
PHYSICS_CONSTANTS = {
    'Omega': 1.56e-29,  # Atomic volume for Ag (m³)
    'k_B': 1.38e-23,    # Boltzmann constant (J/K)
    'Q_bulk': 1.1e-19,  # Activation energy for bulk diffusion (J) ~0.7 eV
    'Q_gb': 8.0e-20,    # Activation energy for grain boundary diffusion (J) ~0.5 eV
    'Q_surface': 6.4e-20, # Activation energy for surface diffusion (J) ~0.4 eV
    'D0_bulk': 1.0e-6,  # Pre-exponential factor for bulk diffusion (m²/s)
    'D0_gb': 1.0e-5,    # Pre-exponential factor for GB diffusion (m²/s)
    'D0_surface': 1.0e-4, # Pre-exponential factor for surface diffusion (m²/s)
}

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
# ORIENTATION-AWARE TRANSFORMER WITH SPATIAL LOCALITY REGULARIZATION
# =============================================
class OrientationAwareTransformerInterpolator:
    """Transformer-inspired stress interpolator with enhanced orientation awareness and spatial locality regularization"""
    
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0,
                 theta_weight_factor=3.0, orientation_attention_heads=2):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.theta_weight_factor = theta_weight_factor  # Boost for theta in positional weights
        self.orientation_attention_heads = orientation_attention_heads  # Special heads for orientation
        
        # Main transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Orientation-specific attention heads
        self.orientation_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=orientation_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Input projection - expects 15 features
        self.input_proj = nn.Linear(15, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Learnable orientation bias
        self.orientation_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Spatial locality regularization weights
        self.spatial_weights = nn.Parameter(torch.ones(1, 1))
        
    def compute_orientation_aware_weights(self, source_params, target_params):
        """
        Compute spatial locality weights with boosted orientation contribution.
        Key enhancement: θ (orientation) gets higher weight than eps0/kappa.
        """
        weights = []
        
        for src in source_params:
            # Compute weighted parameter distance with orientation boost
            param_dist = 0.0
            param_weights = {'eps0': 1.0, 'kappa': 1.0, 'theta': self.theta_weight_factor, 'defect_type': 1.5}
            
            for param, weight_factor in param_weights.items():
                if param in src and param in target_params:
                    if param == 'defect_type':
                        # Categorical similarity with moderate weight
                        param_dist += weight_factor * (0.0 if src[param] == target_params[param] else 1.0)
                    
                    elif param == 'theta':
                        # ANGULAR DISTANCE WITH BOOSTED WEIGHT
                        src_theta = src.get(param, 0.0)
                        tgt_theta = target_params.get(param, 0.0)
                        
                        # Circular distance with periodicity
                        diff = abs(src_theta - tgt_theta)
                        diff = min(diff, 2*np.pi - diff)
                        
                        # Apply boosted weight and normalize
                        param_dist += weight_factor * (diff / np.pi)
                    
                    else:
                        # Normalized Euclidean distance with regular weight
                        max_val = {'eps0': 3.0, 'kappa': 2.0}.get(param, 1.0)
                        param_dist += weight_factor * (abs(src.get(param, 0) - target_params.get(param, 0)) / max_val)
            
            # Apply Gaussian kernel with spatial regularization
            spatial_weight = np.exp(-0.5 * (param_dist / self.spatial_sigma) ** 2)
            
            # Add orientation-specific regularization
            if 'theta' in src and 'theta' in target_params:
                src_theta = src['theta']
                tgt_theta = target_params['theta']
                angular_similarity = 1.0 - (min(abs(src_theta - tgt_theta), 
                                              2*np.pi - abs(src_theta - tgt_theta)) / np.pi)
                orientation_boost = np.exp(2.0 * angular_similarity)  # Exponential boost for similar orientations
                spatial_weight *= orientation_boost
            
            weights.append(spatial_weight)
        
        return np.array(weights)
    
    def encode_parameters_with_orientation_emphasis(self, params_list, target_angle_deg):
        """
        Encode parameters with enhanced orientation features.
        Returns exactly 15 features with strong orientation representation.
        """
        encoded = []
        
        for params in params_list:
            features = []
            
            # Numeric features (3 features)
            features.append(params.get('eps0', 0.707) / 3.0)
            features.append(params.get('kappa', 0.6) / 2.0)
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)  # Normalized theta
            
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
            
            # ENHANCED ORIENTATION FEATURES (4 features with stronger representation)
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            
            # 1. Angular similarity with exponential decay
            features.append(np.exp(-angle_diff / 30.0))  # Stronger decay for orientation
            
            # 2. Sin/Cos of theta (captures periodicity)
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))
            
            # 3. Angular distance normalized
            features.append(angle_diff / 180.0)  # Normalized distance
            
            # Habit plane proximity (1 feature)
            habit_distance = abs(theta_deg - 54.7)
            features.append(np.exp(-habit_distance / 15.0))
            
            # Verify we have exactly 15 features
            if len(features) != 15:
                # Pad with zeros if fewer than 15
                while len(features) < 15:
                    features.append(0.0)
                # Truncate if more than 15
                features = features[:15]
            
            encoded.append(features)
        
        return torch.FloatTensor(encoded)
    
    def apply_orientation_attention(self, features, source_thetas, target_theta):
        """
        Apply orientation-specific attention mechanism.
        Enhances attention to sources with similar orientation.
        """
        batch_size, seq_len, d_model = features.shape
        
        # Create orientation-based attention mask
        if seq_len > 1:  # Have sources
            orientation_similarities = []
            
            # Compute angular similarities for each source
            for src_theta in source_thetas:
                # Convert to degrees if in radians
                if src_theta > 2*np.pi:  # Likely already in degrees
                    src_deg = src_theta
                else:
                    src_deg = np.degrees(src_theta)
                
                # Circular distance
                diff = abs(src_deg - target_theta)
                diff = min(diff, 360 - diff)
                
                # Similarity measure (1 = identical, 0 = opposite)
                similarity = 1.0 - (diff / 180.0)
                orientation_similarities.append(similarity)
            
            # Create attention bias matrix
            attn_bias = torch.zeros(seq_len, seq_len)
            
            # Boost attention between similar orientations
            for i in range(len(orientation_similarities)):
                for j in range(len(orientation_similarities)):
                    if i != j:
                        # Similar orientations get higher attention weight
                        sim_i = orientation_similarities[i]
                        sim_j = orientation_similarities[j]
                        attn_bias[i, j] = (sim_i + sim_j) / 2.0
            
            # Apply orientation-aware attention
            oriented_features, _ = self.orientation_attention(
                features, features, features,
                attn_mask=attn_bias.unsqueeze(0)
            )
            
            return oriented_features
        
        return features
    
    def interpolate_spatial_fields_with_orientation_regularization(self, sources, target_angle_deg, target_params):
        """
        Interpolate spatial fields with orientation-aware regularization.
        Key improvements:
        1. NO field rotation (preserves boundary conditions)
        2. Boosted theta contribution in weights
        3. Orientation-specific attention mechanism
        4. Spatial locality regularization
        """
        if not sources:
            st.warning("No sources provided for interpolation.")
            return None
        
        try:
            # Extract source parameters and fields
            source_params = []
            source_fields = []
            source_thetas = []  # Store theta for orientation attention
            
            for src in sources:
                if 'params' not in src or 'history' not in src:
                    continue
                
                params = src['params']
                source_params.append(params)
                
                # Store theta for orientation processing
                theta_rad = params.get('theta', 0.0)
                source_thetas.append(theta_rad)
                
                # Get stress fields from last frame
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        stress_fields = last_frame['stresses']
                        
                        # Extract or compute stress components
                        if 'von_mises' in stress_fields:
                            vm = stress_fields['von_mises']
                        else:
                            vm = self.compute_von_mises(stress_fields)
                        
                        if 'sigma_hydro' in stress_fields:
                            hydro = stress_fields['sigma_hydro']
                        else:
                            hydro = self.compute_hydrostatic(stress_fields)
                        
                        if 'sigma_mag' in stress_fields:
                            mag = stress_fields['sigma_mag']
                        else:
                            mag = np.sqrt(vm**2 + hydro**2)
                        
                        source_fields.append({
                            'von_mises': vm,
                            'sigma_hydro': hydro,
                            'sigma_mag': mag,
                            'original_theta': theta_rad  # Keep original orientation
                        })
            
            if not source_params or not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Ensure all fields have same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                # Resize to common shape (use median shape)
                target_shape = np.median(shapes, axis=0).astype(int)
                resized_fields = []
                for fields in source_fields:
                    resized = {}
                    for key, field in fields.items():
                        if key != 'original_theta' and field.shape != tuple(target_shape):
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            resized[key] = zoom(field, factors, order=1)
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                source_fields = resized_fields
            
            # Encode parameters with orientation emphasis
            source_features = self.encode_parameters_with_orientation_emphasis(source_params, target_angle_deg)
            target_features = self.encode_parameters_with_orientation_emphasis([target_params], target_angle_deg)
            
            # Ensure feature dimension consistency
            if source_features.shape[1] != 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] != 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            # Compute orientation-aware positional weights with boosted theta
            pos_weights = self.compute_orientation_aware_weights(source_params, target_params)
            pos_weights = pos_weights / pos_weights.sum() if pos_weights.sum() > 0 else np.ones_like(pos_weights) / len(pos_weights)
            
            # Prepare transformer input sequence
            seq_len = len(source_features) + 1
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            
            # Apply input projection
            proj_features = self.input_proj(all_features)
            
            # Add positional encoding and orientation bias
            proj_features = self.pos_encoder(proj_features)
            proj_features = proj_features + self.orientation_bias
            
            # Apply orientation-specific attention
            proj_features = self.apply_orientation_attention(
                proj_features, source_thetas, target_angle_deg
            )
            
            # Main transformer encoding
            transformer_output = self.transformer(proj_features)
            
            # Extract target representation
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
            
            # Compute scaled dot-product attention
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / (np.sqrt(self.d_model) * self.temperature)
            transformer_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().numpy()
            
            # Combine positional and transformer weights with orientation regularization
            # Orientation-similar sources get higher combined weight
            orientation_similarities = []
            for src_theta in source_thetas:
                if src_theta > 2*np.pi:
                    src_deg = src_theta
                else:
                    src_deg = np.degrees(src_theta)
                diff = abs(src_deg - target_angle_deg)
                diff = min(diff, 360 - diff)
                similarity = np.exp(-diff / 45.0)  # Similarity measure
                orientation_similarities.append(similarity)
            
            # Weight adjustment based on orientation similarity
            orientation_adjustment = np.array(orientation_similarities) ** 2  # Square to emphasize similar orientations
            
            # Final weight combination
            combined_weights = (0.6 * transformer_weights + 
                              0.3 * pos_weights + 
                              0.1 * orientation_adjustment)
            combined_weights = combined_weights / combined_weights.sum()
            
            # Interpolate fields WITHOUT rotation (preserve boundary conditions)
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    interpolated += combined_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            # Compute statistics
            max_vm = np.max(interpolated_fields['von_mises'])
            max_hydro_tension = np.max(interpolated_fields['sigma_hydro'])
            max_hydro_compression = np.min(interpolated_fields['sigma_hydro'])
            
            # Store source theta values in degrees for visualization
            source_theta_degrees = [np.degrees(theta) if theta < 2*np.pi else theta for theta in source_thetas]
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'transformer': transformer_weights.tolist(),
                    'positional': pos_weights.tolist(),
                    'orientation_adjustment': orientation_adjustment.tolist(),
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
                        'max_tension': float(max_hydro_tension),
                        'max_compression': float(max_hydro_compression),
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
                'orientation_similarities': orientation_similarities,
                'theta_weight_factor': self.theta_weight_factor
            }
            
        except Exception as e:
            st.error(f"Error during orientation-aware interpolation: {str(e)}")
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
        return np.zeros((100, 100))
    
    def compute_hydrostatic(self, stress_fields):
        """Compute hydrostatic stress from stress components"""
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            return (sxx + syy + szz) / 3
        return np.zeros((100, 100))

# =============================================
# ENHANCED VISUALIZER WITH ORIENTATION ANALYSIS
# =============================================
class EnhancedOrientationVisualizer:
    """Enhanced visualizer with proper orientation analysis and physics"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.defect_colors = {
            'ISF': '#FF6B6B',    # Red
            'ESF': '#4ECDC4',    # Teal
            'Twin': '#45B7D1',   # Blue
            'No Defect': '#96CEB4' # Green
        }
    
    def create_orientation_analysis_dashboard(self, interpolation_result, diffusion_T=650):
        """
        Create comprehensive dashboard showing orientation effects on stress fields.
        """
        if not interpolation_result:
            return None
        
        fields = interpolation_result['fields']
        target_angle = interpolation_result['target_angle']
        defect_type = interpolation_result['target_params']['defect_type']
        source_thetas = interpolation_result.get('source_theta_degrees', [])
        weights = interpolation_result.get('weights', {})
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                f'Von Mises Stress (θ={target_angle:.1f}°)',
                f'Hydrostatic Stress (θ={target_angle:.1f}°)',
                'Source Orientation Distribution',
                'Stress Magnitude',
                'Diffusion Enhancement',
                'Weight vs Orientation Similarity',
                'Angular Stress Profile',
                'Orientation Similarity Heatmap',
                '3D Orientation Space'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}, {'type': 'scatter3d'}]
            ],
            column_widths=[0.3, 0.3, 0.4],
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # Row 1, Col 1: Von Mises Stress
        fig.add_trace(go.Heatmap(
            z=fields['von_mises'],
            colorscale='Viridis',
            colorbar=dict(title="Von Mises (GPa)", x=0.28),
            hovertemplate="Von Mises: %{z:.3f} GPa<extra></extra>"
        ), row=1, col=1)
        
        # Row 1, Col 2: Hydrostatic Stress
        fig.add_trace(go.Heatmap(
            z=fields['sigma_hydro'],
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Hydrostatic (GPa)", x=0.61),
            hovertemplate="Hydrostatic: %{z:.3f} GPa<extra></extra>"
        ), row=1, col=2)
        
        # Row 1, Col 3: Source Orientation Distribution
        if source_thetas:
            # Create polar histogram
            theta_rad = np.radians(source_thetas)
            combined_weights = weights.get('combined', [1/len(source_thetas)]*len(source_thetas))
            
            fig.add_trace(go.Scatterpolar(
                r=combined_weights,
                theta=source_thetas,
                mode='markers',
                marker=dict(
                    size=12,
                    color=combined_weights,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Weight", x=0.95)
                ),
                hovertemplate="θ: %{theta:.1f}°<br>Weight: %{r:.3f}<extra></extra>",
                name='Source Orientations'
            ), row=1, col=3)
            
            # Add target orientation
            fig.add_trace(go.Scatterpolar(
                r=[max(combined_weights)*1.1],
                theta=[target_angle],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                hovertemplate=f"Target: θ={target_angle:.1f}°<extra></extra>",
                name='Target Orientation'
            ), row=1, col=3)
            
            fig.update_polars(
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90
                ),
                radialaxis=dict(
                    visible=True,
                    range=[0, max(combined_weights)*1.2]
                ),
                row=1, col=3
            )
        
        # Row 2, Col 1: Stress Magnitude
        fig.add_trace(go.Heatmap(
            z=fields['sigma_mag'],
            colorscale='Plasma',
            colorbar=dict(title="Magnitude (GPa)", x=0.28),
            hovertemplate="Stress Magnitude: %{z:.3f} GPa<extra></extra>"
        ), row=2, col=1)
        
        # Row 2, Col 2: Diffusion Enhancement
        diffusion_field = self.compute_diffusion_enhancement_factor(fields['sigma_hydro'], diffusion_T)
        fig.add_trace(go.Heatmap(
            z=diffusion_field,
            colorscale='RdBu',
            zmin=0.1, zmax=10,
            colorbar=dict(
                title="D/D_bulk",
                tickvals=[0.1, 0.5, 1, 2, 5, 10],
                ticktext=['0.1x', '0.5x', '1x', '2x', '5x', '10x'],
                x=0.61
            ),
            hovertemplate="D/D_bulk: %{z:.3f}<extra></extra>"
        ), row=2, col=2)
        
        # Row 2, Col 3: Weight vs Orientation Similarity
        if source_thetas and 'orientation_similarities' in interpolation_result:
            similarities = interpolation_result['orientation_similarities']
            combined_weights = weights.get('combined', [])
            
            fig.add_trace(go.Scatter(
                x=similarities,
                y=combined_weights,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=source_thetas,
                    colorscale='Rainbow',
                    showscale=True,
                    colorbar=dict(title="θ (°)", x=0.95)
                ),
                text=[f"{theta:.0f}°" for theta in source_thetas],
                textposition="top center",
                hovertemplate="Similarity: %{x:.3f}<br>Weight: %{y:.3f}<br>θ: %{text}<extra></extra>",
                name='Weight vs Similarity'
            ), row=2, col=3)
            
            # Add trend line
            if len(similarities) > 1:
                z = np.polyfit(similarities, combined_weights, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(min(similarities), max(similarities), 100)
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=p(x_fit),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend',
                    hovertemplate="Similarity: %{x:.3f}<br>Predicted Weight: %{y:.3f}<extra></extra>"
                ), row=2, col=3)
            
            fig.update_xaxes(title_text="Orientation Similarity", row=2, col=3)
            fig.update_yaxes(title_text="Interpolation Weight", row=2, col=3)
        
        # Row 3, Col 1: Angular Stress Profile
        center_x = fields['sigma_hydro'].shape[1] // 2
        center_y = fields['sigma_hydro'].shape[0] // 2
        radius = min(center_x, center_y) // 2
        
        angles = np.linspace(0, 360, 72, endpoint=False)
        stresses = []
        
        for angle in angles:
            angle_rad = np.radians(angle)
            x_sample = center_x + radius * np.cos(angle_rad)
            y_sample = center_y + radius * np.sin(angle_rad)
            xi = int(np.clip(x_sample, 0, fields['sigma_hydro'].shape[1]-1))
            yi = int(np.clip(y_sample, 0, fields['sigma_hydro'].shape[0]-1))
            stresses.append(fields['sigma_hydro'][yi, xi])
        
        fig.add_trace(go.Scatterpolar(
            r=stresses,
            theta=angles,
            mode='lines',
            line=dict(color='blue', width=3),
            hovertemplate="θ: %{theta:.1f}°<br>Stress: %{r:.3f} GPa<extra></extra>",
            name='Hydrostatic Stress'
        ), row=3, col=1)
        
        # Add reference lines
        fig.add_trace(go.Scatterpolar(
            r=[0, max(stresses)*1.1],
            theta=[target_angle, target_angle],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate=f"Target θ: {target_angle:.1f}°<extra></extra>",
            name='Target Orientation'
        ), row=3, col=1)
        
        # Row 3, Col 2: Orientation Similarity Heatmap
        if source_thetas:
            # Create similarity matrix
            n_sources = len(source_thetas)
            similarity_matrix = np.zeros((n_sources, n_sources))
            
            for i in range(n_sources):
                for j in range(n_sources):
                    if i != j:
                        diff = abs(source_thetas[i] - source_thetas[j])
                        diff = min(diff, 360 - diff)
                        similarity_matrix[i, j] = 1.0 - (diff / 180.0)
            
            fig.add_trace(go.Heatmap(
                z=similarity_matrix,
                colorscale='RdBu',
                zmid=0.5,
                colorbar=dict(title="Similarity", x=0.95),
                hovertemplate="Source %{y} vs %{x}<br>Similarity: %{z:.3f}<extra></extra>",
                name='Orientation Similarity'
            ), row=3, col=2)
            
            fig.update_xaxes(title_text="Source Index", row=3, col=2)
            fig.update_yaxes(title_text="Source Index", row=3, col=2)
        
        # Row 3, Col 3: 3D Orientation Space
        if source_thetas and 'orientation_similarities' in interpolation_result:
            similarities = interpolation_result['orientation_similarities']
            combined_weights = weights.get('combined', [])
            
            # Convert to 3D coordinates
            theta_rad = np.radians(source_thetas)
            x = similarities * np.cos(theta_rad)
            y = similarities * np.sin(theta_rad)
            z = combined_weights
            
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=8,
                    color=source_thetas,
                    colorscale='Rainbow',
                    showscale=True,
                    colorbar=dict(title="θ (°)", x=1.0)
                ),
                hovertemplate="θ: %{marker.color:.1f}°<br>Similarity: %{x:.3f}<br>Weight: %{z:.3f}<extra></extra>",
                name='3D Orientation Space'
            ), row=3, col=3)
            
            # Add target point
            target_similarity = 1.0  # Perfect similarity to itself
            target_x = target_similarity * np.cos(np.radians(target_angle))
            target_y = target_similarity * np.sin(np.radians(target_angle))
            target_z = np.max(combined_weights) * 1.2 if combined_weights else 1.0
            
            fig.add_trace(go.Scatter3d(
                x=[target_x],
                y=[target_y],
                z=[target_z],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                hovertemplate=f"Target: θ={target_angle:.1f}°<extra></extra>",
                name='Target'
            ), row=3, col=3)
            
            fig.update_scenes(
                xaxis=dict(title="X (Similarity × cosθ)"),
                yaxis=dict(title="Y (Similarity × sinθ)"),
                zaxis=dict(title="Interpolation Weight"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Orientation Analysis Dashboard: {defect_type} at θ={target_angle:.1f}°",
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            width=1400,
            height=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def compute_diffusion_enhancement_factor(self, sigma_hydro, T=650, mode='physics_corrected'):
        """Compute diffusion enhancement factor"""
        sigma_pa = sigma_hydro * 1e9
        Omega = PHYSICS_CONSTANTS['Omega']
        k_B = PHYSICS_CONSTANTS['k_B']
        
        if mode == 'physics_corrected':
            return np.exp(Omega * sigma_pa / (k_B * T))
        return np.ones_like(sigma_hydro)
    
    def create_orientation_weight_analysis(self, interpolation_result):
        """Create detailed analysis of orientation effects on weights"""
        if not interpolation_result:
            return None
        
        weights = interpolation_result['weights']
        source_thetas = interpolation_result.get('source_theta_degrees', [])
        target_angle = interpolation_result['target_angle']
        
        if not source_thetas:
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Weight Components vs Orientation',
                'Angular Distance vs Weight',
                'Weight Composition',
                'Orientation Similarity Distribution'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'histogram'}]
            ]
        )
        
        # Plot 1: Weight components vs orientation
        angular_distances = []
        for theta in source_thetas:
            diff = abs(theta - target_angle)
            diff = min(diff, 360 - diff)
            angular_distances.append(diff)
        
        fig.add_trace(go.Scatter(
            x=source_thetas,
            y=weights.get('transformer', []),
            mode='markers+lines',
            name='Transformer Weights',
            line=dict(color='orange', width=2),
            marker=dict(size=10, symbol='circle'),
            hovertemplate="θ: %{x:.1f}°<br>Weight: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=source_thetas,
            y=weights.get('positional', []),
            mode='markers+lines',
            name='Positional Weights',
            line=dict(color='green', width=2),
            marker=dict(size=10, symbol='square'),
            hovertemplate="θ: %{x:.1f}°<br>Weight: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=source_thetas,
            y=weights.get('combined', []),
            mode='markers+lines',
            name='Combined Weights',
            line=dict(color='blue', width=3),
            marker=dict(size=12, symbol='star'),
            hovertemplate="θ: %{x:.1f}°<br>Weight: %{y:.3f}<extra></extra>"
        ), row=1, col=1)
        
        # Add target line
        fig.add_vline(x=target_angle, line_dash="dash", line_color="red", 
                     annotation_text=f"Target: {target_angle:.1f}°", 
                     annotation_position="top right", row=1, col=1)
        
        fig.update_xaxes(title_text="Source θ (°)", row=1, col=1)
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        
        # Plot 2: Angular distance vs weight
        fig.add_trace(go.Scatter(
            x=angular_distances,
            y=weights.get('combined', []),
            mode='markers',
            marker=dict(
                size=12,
                color=source_thetas,
                colorscale='Rainbow',
                showscale=True,
                colorbar=dict(title="θ (°)", x=1.0)
            ),
            hovertemplate="Δθ: %{x:.1f}°<br>Weight: %{y:.3f}<br>θ: %{marker.color:.1f}°<extra></extra>",
            name='Weight vs Δθ'
        ), row=1, col=2)
        
        # Add trend line
        if len(angular_distances) > 1:
            z = np.polyfit(angular_distances, weights.get('combined', []), 2)
            p = np.poly1d(z)
            x_fit = np.linspace(min(angular_distances), max(angular_distances), 100)
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=p(x_fit),
                mode='lines',
                line=dict(color='red', dash='dash', width=3),
                name='Trend',
                hovertemplate="Δθ: %{x:.1f}°<br>Predicted Weight: %{y:.3f}<extra></extra>"
            ), row=1, col=2)
        
        fig.update_xaxes(title_text="Angular Distance Δθ (°)", row=1, col=2)
        fig.update_yaxes(title_text="Combined Weight", row=1, col=2)
        
        # Plot 3: Weight composition
        weight_types = ['Transformer', 'Positional', 'Orientation Adj.']
        avg_weights = [
            np.mean(weights.get('transformer', [])),
            np.mean(weights.get('positional', [])),
            np.mean(weights.get('orientation_adjustment', [0]*len(source_thetas)))
        ]
        
        fig.add_trace(go.Bar(
            x=weight_types,
            y=avg_weights,
            marker_color=['orange', 'green', 'purple'],
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            name='Average Weights'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="Weight Type", row=2, col=1)
        fig.update_yaxes(title_text="Average Weight", row=2, col=1)
        
        # Plot 4: Orientation similarity distribution
        if 'orientation_similarities' in interpolation_result:
            similarities = interpolation_result['orientation_similarities']
            
            fig.add_trace(go.Histogram(
                x=similarities,
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7,
                name='Similarity Distribution',
                hovertemplate="Similarity: %{x:.3f}<br>Count: %{y}<extra></extra>"
            ), row=2, col=2)
            
            fig.update_xaxes(title_text="Orientation Similarity", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text=f"Orientation Weight Analysis - θ={target_angle:.1f}°",
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            width=1000,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig

# =============================================
# RESULTS MANAGER WITH ORIENTATION METRICS
# =============================================
class OrientationAwareResultsManager:
    """Results manager with enhanced orientation metrics"""
    
    def __init__(self):
        pass
    
    def prepare_orientation_metrics(self, interpolation_result):
        """Calculate comprehensive orientation metrics"""
        if not interpolation_result:
            return {}
        
        metrics = {
            'basic': {},
            'orientation': {},
            'weight_analysis': {}
        }
        
        # Basic statistics
        stats = interpolation_result['statistics']
        metrics['basic'] = {
            'von_mises_max': stats['von_mises']['max'],
            'hydrostatic_tension_max': stats['sigma_hydro']['max_tension'],
            'hydrostatic_compression_max': stats['sigma_hydro']['max_compression'],
            'stress_magnitude_max': stats['sigma_mag']['max']
        }
        
        # Orientation metrics
        source_thetas = interpolation_result.get('source_theta_degrees', [])
        target_angle = interpolation_result['target_angle']
        
        if source_thetas:
            # Calculate angular statistics
            angular_distances = []
            for theta in source_thetas:
                diff = abs(theta - target_angle)
                diff = min(diff, 360 - diff)
                angular_distances.append(diff)
            
            metrics['orientation'] = {
                'target_angle': target_angle,
                'num_sources': len(source_thetas),
                'mean_source_angle': np.mean(source_thetas),
                'std_source_angle': np.std(source_thetas),
                'min_angular_distance': np.min(angular_distances),
                'max_angular_distance': np.max(angular_distances),
                'mean_angular_distance': np.mean(angular_distances),
                'angular_coverage': 360 if len(source_thetas) == 0 else 
                    (max(source_thetas) - min(source_thetas)) % 360
            }
        
        # Weight analysis metrics
        weights = interpolation_result.get('weights', {})
        if 'combined' in weights and source_thetas:
            combined_weights = weights['combined']
            
            # Calculate weight concentration
            sorted_weights = np.sort(combined_weights)[::-1]
            top3_concentration = np.sum(sorted_weights[:3]) / np.sum(combined_weights)
            top5_concentration = np.sum(sorted_weights[:5]) / np.sum(combined_weights)
            
            # Calculate orientation-weight correlation
            if len(source_thetas) > 1:
                angular_distances = [min(abs(t-target_angle), 360-abs(t-target_angle)) for t in source_thetas]
                weight_orientation_corr = np.corrcoef(angular_distances, combined_weights)[0, 1]
            else:
                weight_orientation_corr = 0.0
            
            metrics['weight_analysis'] = {
                'weight_entropy': -np.sum(combined_weights * np.log(combined_weights + 1e-10)),
                'weight_concentration_top3': top3_concentration,
                'weight_concentration_top5': top5_concentration,
                'weight_std': np.std(combined_weights),
                'weight_orientation_correlation': weight_orientation_corr,
                'theta_weight_factor': interpolation_result.get('theta_weight_factor', 1.0)
            }
        
        return metrics
    
    def export_orientation_report(self, interpolation_result, filename=None):
        """Export comprehensive orientation report"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename = f"orientation_report_theta_{theta}_{defect}_{timestamp}.json"
        
        # Prepare report data
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'method': 'orientation_aware_transformer',
                'theta_weight_factor': interpolation_result.get('theta_weight_factor', 1.0)
            },
            'interpolation_result': interpolation_result,
            'orientation_metrics': self.prepare_orientation_metrics(interpolation_result)
        }
        
        json_str = json.dumps(report, indent=2, default=self._json_serializer)
        return json_str, filename
    
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
# MAIN APPLICATION WITH ORIENTATION CONTROLS
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Orientation-Aware Stress Interpolation",
        layout="wide",
        page_icon="🎯",
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
    .orientation-box {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.1rem;
    }
    .physics-note {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
        color: #856404;
    }
    .success-box {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
        color: #155724;
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
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Orientation-Aware Stress Field Interpolation</h1>', unsafe_allow_html=True)
    
    # Physics explanation
    st.markdown("""
    <div class="physics-note">
    <strong>🔬 Enhanced Orientation-Aware Regularization:</strong><br>
    • <strong>Boosted θ contribution</strong> in spatial locality weights (θ weight = 3.0× eps0/kappa)<br>
    • <strong>Orientation-specific attention heads</strong> in transformer architecture<br>
    • <strong>NO field rotation</strong> - preserves boundary conditions and material constraints<br>
    • <strong>Spatial locality regularization</strong> with orientation similarity adjustment<br>
    • <strong>Physics-corrected diffusion</strong>: D/D_bulk = exp(Ωσ/kT) for both tension/compression
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'orientation_interpolator' not in st.session_state:
        st.session_state.orientation_interpolator = OrientationAwareTransformerInterpolator()
    if 'orientation_visualizer' not in st.session_state:
        st.session_state.orientation_visualizer = EnhancedOrientationVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = OrientationAwareResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    
    # Sidebar with enhanced orientation controls
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
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.success("All data cleared")
        
        # ORIENTATION-AWARE PARAMETERS
        st.markdown('<h2 class="section-header">🎯 Orientation Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle
        custom_theta = st.number_input(
            "Target Polar Angle θ (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=54.7,
            step=0.1,
            help="Target orientation angle for interpolation"
        )
        
        # Defect type
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select defect type"
        )
        
        # Auto-set eigen strain
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
            step=0.1
        )
        
        # ENHANCED ORIENTATION REGULARIZATION PARAMETERS
        st.markdown('<h2 class="section-header">🔧 Orientation Regularization</h2>', unsafe_allow_html=True)
        
        # Theta weight factor
        theta_weight_factor = st.slider(
            "θ Weight Boost Factor",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="How much more important is θ compared to eps0/kappa (higher = more orientation-sensitive)"
        )
        
        # Spatial sigma
        spatial_sigma = st.slider(
            "Spatial Locality Sigma",
            min_value=0.05,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Controls spatial locality regularization"
        )
        
        # Attention temperature
        attention_temp = st.slider(
            "Attention Temperature",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Temperature for attention scaling"
        )
        
        # Orientation attention heads
        orientation_heads = st.slider(
            "Orientation Attention Heads",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
            help="Number of attention heads dedicated to orientation processing"
        )
        
        # Visualization
        st.markdown('<h2 class="section-header">🎨 Visualization</h2>', unsafe_allow_html=True)
        colormap_category = st.selectbox(
            "Colormap Category",
            list(COLORMAP_OPTIONS.keys()),
            index=4
        )
        colormap_name = st.selectbox(
            "Select Colormap",
            COLORMAP_OPTIONS[colormap_category],
            index=0
        )
        
        # Diffusion temperature
        diffusion_T = st.slider(
            "Temperature (K)",
            min_value=300.0,
            max_value=1200.0,
            value=650.0,
            step=50.0,
            help="Sintering temperature for diffusion calculation"
        )
        
        # Interpolation button
        st.markdown("---")
        if st.button("🚀 Perform Orientation-Aware Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                # Update interpolator with orientation parameters
                st.session_state.orientation_interpolator.theta_weight_factor = theta_weight_factor
                st.session_state.orientation_interpolator.spatial_sigma = spatial_sigma
                st.session_state.orientation_interpolator.temperature = attention_temp
                st.session_state.orientation_interpolator.orientation_attention_heads = orientation_heads
                
                # Prepare target parameters
                target_params = {
                    'defect_type': defect_type,
                    'eps0': eps0,
                    'kappa': kappa,
                    'theta': np.radians(custom_theta),
                    'shape': shape
                }
                
                # Perform interpolation
                with st.spinner("Performing orientation-aware interpolation..."):
                    try:
                        result = st.session_state.orientation_interpolator.interpolate_spatial_fields_with_orientation_regularization(
                            st.session_state.solutions,
                            custom_theta,
                            target_params
                        )
                        if result:
                            st.session_state.interpolation_result = result
                            st.success(f"✅ Successfully interpolated at θ={custom_theta:.1f}°")
                            
                            # Show orientation metrics
                            if result.get('source_theta_degrees'):
                                source_thetas = result['source_theta_degrees']
                                angular_diffs = [min(abs(t-custom_theta), 360-abs(t-custom_theta)) for t in source_thetas]
                                avg_diff = np.mean(angular_diffs)
                                st.info(f"• Used {len(source_thetas)} sources\n• Average angular difference: {avg_diff:.1f}°\n• θ weight factor: {theta_weight_factor:.1f}")
                        else:
                            st.error("❌ Failed to interpolate stress fields.")
                    except Exception as e:
                        st.error(f"❌ Error during interpolation: {str(e)}")
    
    # Main content
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        
        # Quick guide
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        1. **Prepare Data**: Place simulation files in `numerical_solutions` directory
        2. **Load Solutions**: Click "Load Solutions" in sidebar
        3. **Set Parameters**: Configure target angle and defect type
        4. **Adjust Orientation Settings**: Set θ weight boost and regularization parameters
        5. **Perform Interpolation**: Click "Perform Orientation-Aware Interpolation"
        6. **Analyze Results**: Explore orientation effects in different tabs
        
        ## 🔬 Key Orientation Features
        
        ### Enhanced Orientation Regularization
        - **θ Weight Boost**: Control how much orientation matters vs other parameters
        - **Orientation Attention Heads**: Dedicated transformer heads for angular relationships
        - **Spatial Locality**: Regularization that respects boundary conditions
        
        ### Physics-Preserving Interpolation
        - **NO field rotation**: Preserves boundary conditions and material constraints
        - **Orientation similarity adjustment**: Weight sources based on angular proximity
        - **Crystallographic awareness**: Respects habit plane orientations (54.7°)
        
        ### Comprehensive Analysis
        - **Orientation dashboards**: Visualize angular relationships and weights
        - **Diffusion physics**: Calculate vacancy-mediated diffusion enhancement
        - **Weight analysis**: Understand how orientation affects interpolation
        """)
    
    else:
        # Enhanced tabs with orientation focus
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Results",
            "🎯 Orientation Dashboard",
            "📈 Weight Analysis",
            "🌊 Diffusion Physics",
            "🔍 Source Analysis",
            "📐 Orientation Metrics",
            "📤 Export"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 Interpolation Results</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Key metrics with orientation focus
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    vm_stats = result['statistics']['von_mises']
                    st.metric("Von Mises Max", f"{vm_stats['max']:.3f} GPa",
                              f"Mean: {vm_stats['mean']:.3f} GPa")
                with col2:
                    hydro_stats = result['statistics']['sigma_hydro']
                    tensile_fraction = np.sum(result['fields']['sigma_hydro'] > 0) / result['fields']['sigma_hydro'].size
                    st.metric("Hydrostatic Range",
                              f"{hydro_stats['max_tension']:.3f}/{hydro_stats['max_compression']:.3f} GPa",
                              f"Tensile: {tensile_fraction*100:.1f}%")
                with col3:
                    # Orientation metrics
                    source_thetas = result.get('source_theta_degrees', [])
                    if source_thetas:
                        avg_angular_diff = np.mean([min(abs(t-result['target_angle']), 
                                                       360-abs(t-result['target_angle'])) 
                                                    for t in source_thetas])
                        st.metric("Orientation Metrics",
                                  f"{len(source_thetas)} sources",
                                  f"Avg Δθ: {avg_angular_diff:.1f}°")
                    else:
                        st.metric("Sources", "0", "No orientation data")
                with col4:
                    mag_stats = result['statistics']['sigma_mag']
                    st.metric("Stress Magnitude Max", f"{mag_stats['max']:.3f} GPa",
                              f"Mean: {mag_stats['mean']:.3f} GPa")
                
                # Quick stress field preview
                st.markdown("#### 👀 Stress Field Preview")
                fig_preview, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude']
                
                for idx, (comp, title) in enumerate(zip(components, titles)):
                    ax = axes[idx]
                    field = result['fields'][comp]
                    
                    if comp == 'sigma_hydro':
                        cmap = 'RdBu_r'
                        vmax = max(abs(np.min(field)), abs(np.max(field)))
                        vmin = -vmax
                    else:
                        cmap = 'viridis'
                        vmin, vmax = None, None
                    
                    im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                                   aspect='equal', interpolation='bilinear', origin='lower')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title(title, fontsize=12)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.grid(True, alpha=0.2)
                
                plt.suptitle(f"Stress Fields at θ={result['target_angle']:.1f}°, {result['target_params']['defect_type']}",
                             fontsize=14)
                plt.tight_layout()
                st.pyplot(fig_preview)
                plt.close(fig_preview)
                
                # Orientation summary
                if result.get('source_theta_degrees'):
                    st.markdown("#### 🎯 Orientation Summary")
                    source_thetas = result['source_theta_degrees']
                    col_o1, col_o2, col_o3 = st.columns(3)
                    with col_o1:
                        st.metric("Target θ", f"{result['target_angle']:.1f}°")
                    with col_o2:
                        min_theta = min(source_thetas)
                        max_theta = max(source_thetas)
                        st.metric("Source θ Range", f"{min_theta:.0f}° to {max_theta:.0f}°")
                    with col_o3:
                        habit_dev = abs(result['target_angle'] - 54.7)
                        st.metric("Habit Plane Deviation", f"{habit_dev:.1f}°")
            else:
                st.info("👈 Configure parameters and click 'Perform Orientation-Aware Interpolation' to generate results")
        
        with tab2:
            st.markdown('<h2 class="section-header">🎯 Orientation Analysis Dashboard</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Comprehensive orientation dashboard
                fig_dashboard = st.session_state.orientation_visualizer.create_orientation_analysis_dashboard(
                    result, diffusion_T
                )
                if fig_dashboard:
                    st.plotly_chart(fig_dashboard, use_container_width=True)
                else:
                    st.error("Failed to create orientation dashboard")
                
                # Orientation insights
                st.markdown("""
                <div class="success-box">
                🔬 <strong>Orientation Insights:</strong><br>
                1. **Weight Distribution**: Sources with similar orientation to target get higher weights<br>
                2. **Angular Similarity**: Cosine similarity measures orientation alignment<br>
                3. **Source Coverage**: Good interpolation requires sources covering angular space<br>
                4. **Boundary Preservation**: NO field rotation means boundary conditions are respected<br>
                5. **Habit Plane**: Defects near 54.7° show maximum tensile stress and diffusion enhancement
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">📈 Weight Analysis</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Detailed weight analysis
                fig_weights = st.session_state.orientation_visualizer.create_orientation_weight_analysis(result)
                if fig_weights:
                    st.plotly_chart(fig_weights, use_container_width=True)
                
                # Weight statistics
                weights = result.get('weights', {})
                if 'combined' in weights:
                    combined_weights = weights['combined']
                    
                    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                    with col_w1:
                        entropy = -np.sum(combined_weights * np.log(combined_weights + 1e-10))
                        st.metric("Weight Entropy", f"{entropy:.3f}",
                                  "Higher = more uniform")
                    with col_w2:
                        top3 = np.sum(np.sort(combined_weights)[::-1][:3]) / np.sum(combined_weights)
                        st.metric("Top 3 Concentration", f"{top3*100:.1f}%")
                    with col_w3:
                        std_weight = np.std(combined_weights)
                        st.metric("Weight Std Dev", f"{std_weight:.3f}")
                    with col_w4:
                        max_weight = np.max(combined_weights)
                        st.metric("Max Weight", f"{max_weight:.3f}")
                    
                    # Top contributors table
                    if result.get('source_theta_degrees'):
                        source_thetas = result['source_theta_degrees']
                        top_indices = np.argsort(combined_weights)[-5:][::-1]
                        
                        st.markdown("##### 🏆 Top 5 Contributing Sources")
                        top_data = []
                        for i, idx in enumerate(top_indices):
                            angular_diff = min(abs(source_thetas[idx] - result['target_angle']),
                                             360 - abs(source_thetas[idx] - result['target_angle']))
                            top_data.append({
                                'Rank': i+1,
                                'Source Index': idx,
                                'θ': f"{source_thetas[idx]:.1f}°",
                                'Δθ': f"{angular_diff:.1f}°",
                                'Combined Weight': f"{combined_weights[idx]:.4f}",
                                'Transformer Weight': f"{weights.get('transformer', [0]*len(combined_weights))[idx]:.4f}",
                                'Positional Weight': f"{weights.get('positional', [0]*len(combined_weights))[idx]:.4f}"
                            })
                        
                        df_top = pd.DataFrame(top_data)
                        st.dataframe(df_top, use_container_width=True)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">🌊 Diffusion Physics Analysis</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Diffusion enhancement calculation
                hydro_field = result['fields']['sigma_hydro']
                diffusion_field = st.session_state.orientation_visualizer.compute_diffusion_enhancement_factor(
                    hydro_field, diffusion_T
                )
                
                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                with col_d1:
                    max_enhancement = np.max(diffusion_field)
                    st.metric("Max Enhancement", f"{max_enhancement:.2f}x")
                with col_d2:
                    min_enhancement = np.min(diffusion_field)
                    st.metric("Min Enhancement", f"{min_enhancement:.2f}x")
                with col_d3:
                    mean_enhancement = np.mean(diffusion_field)
                    st.metric("Mean Enhancement", f"{mean_enhancement:.2f}x")
                with col_d4:
                    enhancement_gt_2 = np.mean(diffusion_field > 2)
                    st.metric("Enhancement > 2x", f"{enhancement_gt_2*100:.1f}%")
                
                # Diffusion heatmap
                st.markdown("#### 🔥 Diffusion Enhancement Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Use diverging colormap centered at 1.0
                cmap = plt.get_cmap('RdBu_r')
                norm = LogNorm(vmin=0.1, vmax=10)
                
                im = ax.imshow(diffusion_field, cmap=cmap, norm=norm,
                               aspect='equal', interpolation='bilinear', origin='lower')
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("D/D_bulk Ratio (log scale)", fontsize=14, fontweight='bold')
                cbar.ax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
                cbar.ax.set_yticklabels(['0.1x', '0.2x', '0.5x', '1x', '2x', '5x', '10x'])
                
                # Add orientation arrow
                center_x = diffusion_field.shape[1] // 2
                center_y = diffusion_field.shape[0] // 2
                length = min(diffusion_field.shape) * 0.3
                
                # Convert target angle to vector
                theta_rad = np.radians(result['target_angle'])
                dx = length * np.cos(theta_rad)
                dy = length * np.sin(theta_rad)
                
                ax.arrow(center_x, center_y, dx, dy,
                         head_width=length*0.1, head_length=length*0.15,
                         fc='green', ec='green', alpha=0.8, linewidth=2)
                
                ax.text(center_x + dx*1.2, center_y + dy*1.2,
                       f'Defect at θ={result["target_angle"]:.1f}°',
                       color='green', fontsize=12, fontweight='bold',
                       verticalalignment='center', horizontalalignment='center')
                
                ax.set_title(f"Diffusion Enhancement at T={diffusion_T}K\n{result['target_params']['defect_type']} at θ={result['target_angle']:.1f}°",
                            fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('X Position', fontsize=14)
                ax.set_ylabel('Y Position', fontsize=14)
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Physics explanation
                st.markdown("""
                <div class="physics-note">
                <strong>Physics of Vacancy-Mediated Diffusion:</strong><br>
                The diffusion coefficient under hydrostatic stress σ is:<br>
                <code>D(σ) = D₀ exp(-(Q - Ωσ)/kT)</code><br>
                Relative to stress-free bulk:<br>
                <code>D(σ)/D_bulk = exp(Ωσ/kT)</code><br>
                Where Ω = 1.56×10⁻²⁹ m³ (Ag), k = 1.38×10⁻²³ J/K, T = temperature<br>
                <br>
                <strong>Key Effects:</strong><br>
                • <span style='color:red'>Tensile stress (σ > 0)</span>: D/D_bulk > 1 → ENHANCEMENT<br>
                • <span style='color:blue'>Compressive stress (σ < 0)</span>: D/D_bulk < 1 → SUPPRESSION<br>
                • Orientation affects stress distribution → affects diffusion pattern<br>
                • Habit plane (54.7°) typically shows maximum tensile stress
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab5:
            st.markdown('<h2 class="section-header">🔍 Source Analysis</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                source_thetas = result.get('source_theta_degrees', [])
                
                if source_thetas:
                    # Create source analysis dashboard
                    st.markdown("#### 📊 Source Orientation Distribution")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Histogram of source orientations
                    ax1.hist(source_thetas, bins=36, range=(0, 360), alpha=0.7, edgecolor='black')
                    ax1.axvline(x=result['target_angle'], color='red', linestyle='--', linewidth=2, label=f'Target: {result["target_angle"]:.1f}°')
                    ax1.axvline(x=54.7, color='blue', linestyle=':', linewidth=2, label='Habit Plane: 54.7°')
                    ax1.set_xlabel('Orientation θ (°)', fontsize=14)
                    ax1.set_ylabel('Number of Sources', fontsize=14)
                    ax1.set_title('Source Orientation Distribution', fontsize=16, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(fontsize=12)
                    
                    # Polar plot of sources with weights
                    ax2 = plt.subplot(122, projection='polar')
                    weights = result['weights']['combined']
                    colors = plt.cm.viridis(weights / np.max(weights))
                    
                    for theta, weight, color in zip(np.radians(source_thetas), weights, colors):
                        ax2.plot([0, theta], [0, weight], color=color, linewidth=2, alpha=0.7)
                        ax2.scatter(theta, weight, color=color, s=100, alpha=0.8)
                    
                    # Add target
                    ax2.scatter(np.radians(result['target_angle']), np.max(weights)*1.1, 
                               color='red', s=200, marker='*', label='Target', zorder=10)
                    
                    ax2.set_theta_zero_location('N')
                    ax2.set_theta_direction(-1)
                    ax2.set_title('Source Weights in Polar Coordinates', fontsize=16, fontweight='bold', pad=20)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Source statistics
                    st.markdown("#### 📈 Source Statistics")
                    
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        angular_coverage = (max(source_thetas) - min(source_thetas)) % 360
                        st.metric("Angular Coverage", f"{angular_coverage:.1f}°")
                    with col_s2:
                        angular_spacing = angular_coverage / len(source_thetas) if len(source_thetas) > 1 else 0
                        st.metric("Average Angular Spacing", f"{angular_spacing:.1f}°")
                    with col_s3:
                        target_proximity = np.min([min(abs(t-result['target_angle']), 
                                                      360-abs(t-result['target_angle'])) 
                                                  for t in source_thetas])
                        st.metric("Closest Source Δθ", f"{target_proximity:.1f}°")
                    with col_s4:
                        habit_proximities = [min(abs(t-54.7), 360-abs(t-54.7)) for t in source_thetas]
                        closest_to_habit = np.min(habit_proximities)
                        st.metric("Closest to Habit Plane", f"{closest_to_habit:.1f}°")
                    
                    # Source table
                    st.markdown("#### 📋 Source Details")
                    source_data = []
                    for i, theta in enumerate(source_thetas):
                        angular_diff = min(abs(theta - result['target_angle']), 360-abs(theta - result['target_angle']))
                        habit_diff = min(abs(theta - 54.7), 360-abs(theta - 54.7))
                        source_data.append({
                            'Source': i+1,
                            'θ (°)': f"{theta:.1f}",
                            'Δθ from Target (°)': f"{angular_diff:.1f}",
                            'Δθ from Habit (°)': f"{habit_diff:.1f}",
                            'Weight': f"{result['weights']['combined'][i]:.4f}",
                            'Orientation Similarity': f"{result['orientation_similarities'][i]:.3f}" if 'orientation_similarities' in result else "N/A"
                        })
                    
                    df_sources = pd.DataFrame(source_data)
                    st.dataframe(df_sources, use_container_width=True)
                    
                else:
                    st.warning("No source orientation data available.")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab6:
            st.markdown('<h2 class="section-header">📐 Orientation Metrics</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Calculate comprehensive orientation metrics
                metrics = st.session_state.results_manager.prepare_orientation_metrics(result)
                
                if metrics:
                    # Display metrics in organized sections
                    st.markdown("#### 📊 Basic Statistics")
                    basic_metrics = metrics.get('basic', {})
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    metrics_list = list(basic_metrics.items())
                    
                    for i, (name, value) in enumerate(metrics_list[:4]):
                        col = [col_m1, col_m2, col_m3, col_m4][i]
                        with col:
                            st.metric(
                                name.replace('_', ' ').title(),
                                f"{value:.3f}" if isinstance(value, float) else str(value)
                            )
                    
                    # Orientation metrics
                    st.markdown("#### 🎯 Orientation Metrics")
                    orientation_metrics = metrics.get('orientation', {})
                    
                    if orientation_metrics:
                        col_o1, col_o2, col_o3, col_o4 = st.columns(4)
                        with col_o1:
                            st.metric("Target θ", f"{orientation_metrics.get('target_angle', 0):.1f}°")
                        with col_o2:
                            st.metric("Number of Sources", orientation_metrics.get('num_sources', 0))
                        with col_o3:
                            st.metric("Mean Source θ", f"{orientation_metrics.get('mean_source_angle', 0):.1f}°")
                        with col_o4:
                            st.metric("Mean Δθ", f"{orientation_metrics.get('mean_angular_distance', 0):.1f}°")
                        
                        col_o5, col_o6, col_o7, col_o8 = st.columns(4)
                        with col_o5:
                            st.metric("Min Δθ", f"{orientation_metrics.get('min_angular_distance', 0):.1f}°")
                        with col_o6:
                            st.metric("Max Δθ", f"{orientation_metrics.get('max_angular_distance', 0):.1f}°")
                        with col_o7:
                            st.metric("θ Std Dev", f"{orientation_metrics.get('std_source_angle', 0):.1f}°")
                        with col_o8:
                            st.metric("Angular Coverage", f"{orientation_metrics.get('angular_coverage', 0):.1f}°")
                    
                    # Weight analysis metrics
                    st.markdown("#### ⚖️ Weight Analysis Metrics")
                    weight_metrics = metrics.get('weight_analysis', {})
                    
                    if weight_metrics:
                        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                        with col_w1:
                            st.metric("Weight Entropy", f"{weight_metrics.get('weight_entropy', 0):.3f}")
                        with col_w2:
                            st.metric("Top 3 Concentration", f"{weight_metrics.get('weight_concentration_top3', 0)*100:.1f}%")
                        with col_w3:
                            st.metric("Weight Std Dev", f"{weight_metrics.get('weight_std', 0):.3f}")
                        with col_w4:
                            corr = weight_metrics.get('weight_orientation_correlation', 0)
                            st.metric("Weight-Orientation Correlation", f"{corr:.3f}")
                        
                        # Interpretation
                        st.markdown("""
                        <div class="success-box">
                        🔬 <strong>Interpretation Guide:</strong><br>
                        • <strong>Weight Entropy</strong>: Higher = more uniform weights, Lower = concentrated on few sources<br>
                        • <strong>Top 3 Concentration</strong>: Percentage of total weight carried by top 3 sources<br>
                        • <strong>Weight-Orientation Correlation</strong>: Positive = similar orientations get higher weights<br>
                        • <strong>θ Weight Factor</strong>: How much orientation matters vs other parameters (current: {:.1f}×)
                        </div>
                        """.format(weight_metrics.get('theta_weight_factor', 1.0)), unsafe_allow_html=True)
                    
                    # Quality assessment
                    st.markdown("#### 🏆 Interpolation Quality Assessment")
                    
                    quality_score = 0.0
                    quality_factors = []
                    
                    # Factor 1: Angular coverage
                    if orientation_metrics.get('angular_coverage', 0) > 180:
                        quality_score += 0.3
                        quality_factors.append("✅ Good angular coverage (>180°)")
                    elif orientation_metrics.get('angular_coverage', 0) > 90:
                        quality_score += 0.2
                        quality_factors.append("⚠️ Moderate angular coverage (90-180°)")
                    else:
                        quality_factors.append("❌ Limited angular coverage (<90°)")
                    
                    # Factor 2: Weight distribution
                    if weight_metrics.get('weight_entropy', 0) > 1.0:
                        quality_score += 0.3
                        quality_factors.append("✅ Good weight distribution (entropy > 1.0)")
                    elif weight_metrics.get('weight_entropy', 0) > 0.5:
                        quality_score += 0.2
                        quality_factors.append("⚠️ Moderate weight distribution")
                    else:
                        quality_factors.append("❌ Concentrated weights (entropy < 0.5)")
                    
                    # Factor 3: Orientation correlation
                    if weight_metrics.get('weight_orientation_correlation', 0) > 0.3:
                        quality_score += 0.2
                        quality_factors.append("✅ Strong orientation correlation (>0.3)")
                    elif weight_metrics.get('weight_orientation_correlation', 0) > 0:
                        quality_score += 0.1
                        quality_factors.append("⚠️ Weak orientation correlation")
                    else:
                        quality_factors.append("❌ Negative orientation correlation")
                    
                    # Factor 4: Source count
                    if orientation_metrics.get('num_sources', 0) >= 5:
                        quality_score += 0.2
                        quality_factors.append("✅ Sufficient sources (≥5)")
                    elif orientation_metrics.get('num_sources', 0) >= 3:
                        quality_score += 0.1
                        quality_factors.append("⚠️ Limited sources (3-4)")
                    else:
                        quality_factors.append("❌ Insufficient sources (<3)")
                    
                    # Display quality score
                    col_q1, col_q2, col_q3 = st.columns([1, 2, 1])
                    with col_q2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, 
                            {'#FF6B6B' if quality_score < 0.4 else '#FFA726' if quality_score < 0.7 else '#4CAF50'}, 
                            {'#C62828' if quality_score < 0.4 else '#F57C00' if quality_score < 0.7 else '#2E7D32'});
                            border-radius: 10px; color: white;">
                        <h3>Quality Score: {quality_score:.1f}/1.0</h3>
                        <p>{'Poor' if quality_score < 0.4 else 'Fair' if quality_score < 0.7 else 'Good'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display quality factors
                    st.markdown("##### Quality Factors:")
                    for factor in quality_factors:
                        st.write(factor)
                    
                    # Recommendations
                    st.markdown("##### 💡 Recommendations:")
                    if quality_score < 0.4:
                        st.error("""
                        **Improvement needed:**
                        1. Add more source simulations with different orientations
                        2. Increase angular coverage (aim for >180°)
                        3. Consider adjusting θ weight factor in sidebar
                        """)
                    elif quality_score < 0.7:
                        st.warning("""
                        **Moderate quality:**
                        1. Could benefit from more sources
                        2. Check if angular distribution is uniform
                        3. Verify θ weight factor is appropriate for your application
                        """)
                    else:
                        st.success("""
                        **Good quality:**
                        1. Sufficient sources with good angular coverage
                        2. Appropriate weight distribution
                        3. Orientation-aware regularization working effectively
                        """)
                else:
                    st.warning("Could not calculate orientation metrics.")
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab7:
            st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Export options
                st.markdown("##### 💾 Export Formats")
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    if st.button("📊 Export Orientation Report", use_container_width=True):
                        report_str, filename = st.session_state.results_manager.export_orientation_report(result)
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=report_str,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True,
                            key="download_report"
                        )
                
                with col_e2:
                    # Export stress fields as CSV
                    if st.button("📈 Export Stress Fields", use_container_width=True):
                        # Create DataFrame with all stress components
                        data_dict = {}
                        for field_name, field_data in result['fields'].items():
                            data_dict[field_name] = field_data.flatten()
                        
                        df = pd.DataFrame(data_dict)
                        csv_str = df.to_csv(index=False)
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"stress_fields_theta_{result['target_angle']:.1f}_{timestamp}.csv"
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_str,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv"
                        )
                
                with col_e3:
                    # Export plot
                    if st.button("🖼️ Export Plot", use_container_width=True):
                        # Create comprehensive figure
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        
                        components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                        titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude']
                        
                        for idx, (comp, title) in enumerate(zip(components, titles)):
                            ax = axes[idx//2, idx%2]
                            field = result['fields'][comp]
                            
                            if comp == 'sigma_hydro':
                                cmap = 'RdBu_r'
                                vmax = max(abs(np.min(field)), abs(np.max(field)))
                                vmin = -vmax
                            else:
                                cmap = 'viridis'
                                vmin, vmax = None, None
                            
                            im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                                          aspect='equal', interpolation='bilinear', origin='lower')
                            plt.colorbar(im, ax=ax, shrink=0.8)
                            ax.set_title(f"{title}\nθ={result['target_angle']:.1f}°", fontsize=12)
                            ax.set_xlabel('X')
                            ax.set_ylabel('Y')
                            ax.grid(True, alpha=0.2)
                        
                        # Add orientation plot in last subplot
                        ax = axes[1, 1]
                        if result.get('source_theta_degrees'):
                            source_thetas = result['source_theta_degrees']
                            weights = result['weights']['combined']
                            
                            ax.scatter(source_thetas, weights, s=100, alpha=0.7, 
                                      c=weights, cmap='viridis')
                            ax.axvline(x=result['target_angle'], color='red', 
                                      linestyle='--', label=f'Target: {result["target_angle"]:.1f}°')
                            ax.set_xlabel('Source θ (°)', fontsize=12)
                            ax.set_ylabel('Weight', fontsize=12)
                            ax.set_title('Weight vs Orientation', fontsize=12)
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize=10)
                        
                        plt.suptitle(f"Stress Field Analysis: {result['target_params']['defect_type']} at θ={result['target_angle']:.1f}°",
                                    fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        
                        # Save to buffer
                        buf = BytesIO()
                        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        buf.seek(0)
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"orientation_analysis_theta_{result['target_angle']:.1f}_{timestamp}.png"
                        
                        st.download_button(
                            label="📥 Download PNG (300 DPI)",
                            data=buf,
                            file_name=filename,
                            mime="image/png",
                            use_container_width=True,
                            key="download_png"
                        )
                        plt.close(fig)
                
                # Bulk export
                st.markdown("---")
                st.markdown("##### 📦 Bulk Export All Data")
                
                if st.button("🚀 Export Complete Dataset", use_container_width=True, type="secondary"):
                    # Create zip file with all data
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Export stress fields
                        for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            field_data = result['fields'][component]
                            df = pd.DataFrame(field_data)
                            csv_str = df.to_csv(index=False)
                            zip_file.writestr(f"{component}_theta_{result['target_angle']:.1f}.csv", csv_str)
                        
                        # Export diffusion enhancement
                        hydro_field = result['fields']['sigma_hydro']
                        diffusion_field = st.session_state.orientation_visualizer.compute_diffusion_enhancement_factor(
                            hydro_field, diffusion_T
                        )
                        df_diff = pd.DataFrame(diffusion_field)
                        csv_diff = df_diff.to_csv(index=False)
                        zip_file.writestr(f"diffusion_enhancement_T{diffusion_T:.0f}K.csv", csv_diff)
                        
                        # Export weights
                        weights_df = pd.DataFrame(result['weights'])
                        csv_weights = weights_df.to_csv(index=False)
                        zip_file.writestr("interpolation_weights.csv", csv_weights)
                        
                        # Export orientation metrics
                        metrics = st.session_state.results_manager.prepare_orientation_metrics(result)
                        metrics_json = json.dumps(metrics, indent=2, default=st.session_state.results_manager._json_serializer)
                        zip_file.writestr("orientation_metrics.json", metrics_json)
                        
                        # Export complete report
                        report_str, _ = st.session_state.results_manager.export_orientation_report(result)
                        zip_file.writestr("complete_report.json", report_str)
                    
                    zip_buffer.seek(0)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"complete_dataset_theta_{result['target_angle']:.1f}_{timestamp}.zip"
                    
                    st.download_button(
                        label="📥 Download ZIP (Complete Dataset)",
                        data=zip_buffer.getvalue(),
                        file_name=filename,
                        mime="application/zip",
                        use_container_width=True,
                        key="download_zip"
                    )
                
                # Configuration summary
                st.markdown("---")
                st.markdown("##### ⚙️ Current Configuration")
                
                config_summary = {
                    'Target Angle': f"{result['target_angle']:.1f}°",
                    'Defect Type': result['target_params']['defect_type'],
                    'θ Weight Factor': result.get('theta_weight_factor', 'N/A'),
                    'Number of Sources': result.get('num_sources', 0),
                    'Spatial Sigma': st.session_state.orientation_interpolator.spatial_sigma,
                    'Attention Temperature': st.session_state.orientation_interpolator.temperature,
                    'Diffusion Temperature': f"{diffusion_T:.0f}K"
                }
                
                df_config = pd.DataFrame(list(config_summary.items()), columns=['Parameter', 'Value'])
                st.dataframe(df_config, use_container_width=True, hide_index=True)
                
            else:
                st.info("No interpolation results available. Please perform interpolation first.")

# Run the application
if __name__ == "__main__":
    main()
