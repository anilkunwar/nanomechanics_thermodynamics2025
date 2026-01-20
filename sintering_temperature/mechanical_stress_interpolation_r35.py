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
# ENHANCED TRANSFORMER SPATIAL INTERPOLATOR WITH HYDROSTATIC OPTIMIZATION
# =============================================
class EnhancedTransformerSpatialInterpolator:
    """Transformer-inspired stress interpolator with enhanced hydrostatic optimization options"""
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0, 
                 locality_weight_factor=0.7, hydro_method="standard", hydro_params=None):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor
        self.hydro_method = hydro_method
        self.hydro_params = hydro_params if hydro_params is not None else {}
        
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
    
    def set_hydro_method(self, method, params=None):
        """Set hydrostatic interpolation method"""
        self.hydro_method = method
        if params:
            self.hydro_params = params
    
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
    
    def encode_parameters(self, params_list, target_angle_deg, source_hydro_data=None):
        """Encode parameters into transformer input - FIXED to return exactly 15 features"""
        encoded = []
        for idx, params in enumerate(params_list):
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
            
            # Add hydrostatic-specific features if method requires and data is available
            if self.hydro_method == "enhanced_features" and source_hydro_data is not None and idx < len(source_hydro_data):
                hydro_field = source_hydro_data[idx]
                # Add hydrostatic statistics as features
                features.append(np.mean(np.sign(hydro_field)))  # Mean sign
                features.append(np.mean(hydro_field > 0))  # Tension ratio
                features.append(np.std(hydro_field))  # Variability
            
            # Verify we have exactly 15 features (or more if enhanced)
            target_features = 15
            if self.hydro_method == "enhanced_features" and source_hydro_data is not None:
                target_features = 18  # Added 3 hydro features
            
            if len(features) != target_features:
                # Pad with zeros if fewer
                while len(features) < target_features:
                    features.append(0.0)
                # Truncate if more
                features = features[:target_features]
            
            encoded.append(features)
        
        return torch.FloatTensor(encoded)
    
    def interpolate_hydrostatic_channel_split(self, source_fields, combined_weights, shape):
        """Interpolate hydrostatic stress using channel splitting method"""
        # Split each source's hydrostatic field into positive and negative channels
        hydro_pos_fields = []
        hydro_neg_fields = []
        
        for fields in source_fields:
            hydro = fields['sigma_hydro']
            # Split into positive (tension) and negative (compression) channels
            hydro_pos = np.maximum(hydro, 0)
            hydro_neg = -np.minimum(hydro, 0)  # Positive values for compression magnitude
            hydro_pos_fields.append(hydro_pos)
            hydro_neg_fields.append(hydro_neg)
        
        # Interpolate each channel separately
        interpolated_pos = np.zeros(shape)
        interpolated_neg = np.zeros(shape)
        
        for i in range(len(source_fields)):
            interpolated_pos += combined_weights[i] * hydro_pos_fields[i]
            interpolated_neg += combined_weights[i] * hydro_neg_fields[i]
        
        # Recombine: tension - compression
        interpolated_hydro = interpolated_pos - interpolated_neg
        
        # Calculate channel statistics
        pos_mean = np.mean(interpolated_pos)
        neg_mean = np.mean(interpolated_neg)
        pos_max = np.max(interpolated_pos)
        neg_max = np.max(interpolated_neg)
        
        return interpolated_hydro, {
            'tension_mean': float(pos_mean),
            'compression_mean': float(neg_mean),
            'tension_max': float(pos_max),
            'compression_max': float(neg_max),
            'tension_ratio': float(pos_mean / (pos_mean + neg_mean + 1e-10))
        }
    
    def interpolate_hydrostatic_weighted_by_sign(self, source_fields, combined_weights, shape, transformer_weights, pos_weights):
        """Interpolate hydrostatic with sign-aware weighting"""
        hydro_fields = [fields['sigma_hydro'] for fields in source_fields]
        
        # Calculate sign agreement between sources
        sign_agreement = np.zeros(shape)
        for i in range(len(hydro_fields)):
            for j in range(i + 1, len(hydro_fields)):
                # Calculate sign correlation between source i and j
                sign_corr = np.mean(np.sign(hydro_fields[i]) == np.sign(hydro_fields[j]))
                # Add to agreement matrix (simplified)
                sign_agreement += sign_corr
        
        # Normalize sign agreement
        if np.max(sign_agreement) > 0:
            sign_agreement = sign_agreement / np.max(sign_agreement)
        
        # Adjust weights based on sign agreement in each region
        adjusted_weights = combined_weights.copy()
        
        # For regions with low sign agreement, increase locality weight factor
        low_agreement_mask = sign_agreement < 0.5
        if np.any(low_agreement_mask):
            # In low agreement regions, favor spatial weights more
            adjusted_locality = 0.3  # Lower transformer influence
            for i in range(len(source_fields)):
                if low_agreement_mask.any():  # Apply to whole field for simplicity
                    adjusted_weights[i] = (
                        adjusted_locality * transformer_weights[i] + 
                        (1 - adjusted_locality) * pos_weights[i]
                    )
        
        # Normalize adjusted weights
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        # Interpolate with adjusted weights
        interpolated_hydro = np.zeros(shape)
        for i, fields in enumerate(source_fields):
            interpolated_hydro += adjusted_weights[i] * fields['sigma_hydro']
        
        return interpolated_hydro, {
            'sign_agreement_mean': float(np.mean(sign_agreement)),
            'sign_agreement_min': float(np.min(sign_agreement)),
            'low_agreement_fraction': float(np.mean(low_agreement_mask)),
            'weight_adjustment_applied': True
        }
    
    def interpolate_hydrostatic_angular_filter(self, source_fields, combined_weights, shape, source_distances, angular_threshold=10.0):
        """Interpolate hydrostatic with angular distance filtering"""
        # Filter sources by angular distance
        filtered_indices = [i for i, dist in enumerate(source_distances) if dist <= angular_threshold]
        
        if not filtered_indices:
            st.warning(f"No sources within {angular_threshold}° angular distance. Using all sources.")
            filtered_indices = list(range(len(source_fields)))
        
        # Re-normalize weights for filtered sources
        filtered_weights = combined_weights[filtered_indices]
        if np.sum(filtered_weights) > 0:
            filtered_weights = filtered_weights / np.sum(filtered_weights)
        
        # Interpolate using only filtered sources
        interpolated_hydro = np.zeros(shape)
        for idx, source_idx in enumerate(filtered_indices):
            interpolated_hydro += filtered_weights[idx] * source_fields[source_idx]['sigma_hydro']
        
        return interpolated_hydro, {
            'angular_threshold': angular_threshold,
            'filtered_sources': len(filtered_indices),
            'total_sources': len(source_fields),
            'filter_ratio': len(filtered_indices) / len(source_fields)
        }
    
    def interpolate_hydrostatic_magnitude_preserving(self, source_fields, combined_weights, shape):
        """Interpolate hydrostatic with magnitude preservation"""
        hydro_fields = [fields['sigma_hydro'] for fields in source_fields]
        
        # First interpolate normally
        interpolated_hydro = np.zeros(shape)
        for i, fields in enumerate(source_fields):
            interpolated_hydro += combined_weights[i] * fields['sigma_hydro']
        
        # Calculate magnitude statistics from sources
        source_magnitudes = [np.abs(hydro) for hydro in hydro_fields]
        
        # Interpolate magnitude separately
        interpolated_magnitude = np.zeros(shape)
        for i, magnitude in enumerate(source_magnitudes):
            interpolated_magnitude += combined_weights[i] * magnitude
        
        # Preserve sign from interpolated hydro but use interpolated magnitude
        interpolated_hydro_preserved = np.sign(interpolated_hydro) * interpolated_magnitude
        
        # Blend: use preserved version in high-gradient regions
        gradient = np.abs(np.gradient(interpolated_hydro)[0]) + np.abs(np.gradient(interpolated_hydro)[1])
        gradient_normalized = gradient / (np.max(gradient) + 1e-10)
        
        # In high gradient regions, use magnitude-preserved version
        blend_factor = gradient_normalized
        final_hydro = (1 - blend_factor) * interpolated_hydro + blend_factor * interpolated_hydro_preserved
        
        return final_hydro, {
            'gradient_mean': float(np.mean(gradient)),
            'gradient_max': float(np.max(gradient)),
            'blend_factor_mean': float(np.mean(blend_factor)),
            'magnitude_preservation_applied': True
        }
    
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """Interpolate full spatial stress fields using transformer attention with hydrostatic optimization"""
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
            
            # Prepare hydrostatic data for enhanced feature encoding if needed
            hydro_data = None
            if self.hydro_method == "enhanced_features":
                hydro_data = [fields['sigma_hydro'] for fields in source_fields]
            
            # Encode parameters with optional hydrostatic features
            source_features = self.encode_parameters(source_params, target_angle_deg, hydro_data)
            target_features = self.encode_parameters([target_params], target_angle_deg, None)
            
            # Ensure we have exactly the right number of features
            expected_features = source_features.shape[1]
            if target_features.shape[1] < expected_features:
                padding = torch.zeros(target_features.shape[0], expected_features - target_features.shape[1])
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
            
            # Apply input projection (adjust for feature dimension)
            if self.hydro_method == "enhanced_features":
                # Update input projection for enhanced features
                self.input_proj = nn.Linear(expected_features, self.d_model)
            
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
            
            # Apply locality weight factor to balance spatial and transformer weights
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
            
            # Extract source distances for angular filtering if needed
            source_theta_degrees = []
            source_distances = []
            
            target_theta_rad = target_params.get('theta', 0.0)
            target_theta_deg = np.degrees(target_theta_rad)
            
            for src in source_params:
                theta_rad = src.get('theta', 0.0)
                theta_deg = np.degrees(theta_rad)
                source_theta_degrees.append(theta_deg)
                
                # Calculate angular distance
                angular_dist = abs(theta_deg - target_theta_deg)
                angular_dist = min(angular_dist, 360 - angular_dist)
                source_distances.append(angular_dist)
            
            # Interpolate von Mises and stress magnitude (standard method)
            for component in ['von_mises', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += combined_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            # Interpolate hydrostatic stress based on selected method
            hydro_stats = {}
            if self.hydro_method == "standard":
                # Standard linear interpolation
                interpolated_hydro = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    interpolated_hydro += combined_weights[i] * fields['sigma_hydro']
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats['method'] = 'standard'
                
            elif self.hydro_method == "channel_split":
                # Channel splitting method
                interpolated_hydro, channel_stats = self.interpolate_hydrostatic_channel_split(
                    source_fields, combined_weights, shape
                )
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats.update(channel_stats)
                hydro_stats['method'] = 'channel_split'
                
            elif self.hydro_method == "sign_aware":
                # Sign-aware weighting method
                interpolated_hydro, sign_stats = self.interpolate_hydrostatic_weighted_by_sign(
                    source_fields, combined_weights, shape, transformer_weights, pos_weights
                )
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats.update(sign_stats)
                hydro_stats['method'] = 'sign_aware'
                
            elif self.hydro_method == "angular_filter":
                # Angular distance filtering
                angular_threshold = self.hydro_params.get('angular_threshold', 10.0)
                interpolated_hydro, filter_stats = self.interpolate_hydrostatic_angular_filter(
                    source_fields, combined_weights, shape, source_distances, angular_threshold
                )
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats.update(filter_stats)
                hydro_stats['method'] = 'angular_filter'
                
            elif self.hydro_method == "magnitude_preserving":
                # Magnitude preserving method
                interpolated_hydro, magnitude_stats = self.interpolate_hydrostatic_magnitude_preserving(
                    source_fields, combined_weights, shape
                )
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats.update(magnitude_stats)
                hydro_stats['method'] = 'magnitude_preserving'
                
            elif self.hydro_method == "enhanced_features":
                # Enhanced feature encoding (already applied in encoding)
                # Use standard interpolation but with enhanced features
                interpolated_hydro = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    interpolated_hydro += combined_weights[i] * fields['sigma_hydro']
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats['method'] = 'enhanced_features'
                hydro_stats['feature_count'] = source_features.shape[1]
                
            else:
                # Fallback to standard
                interpolated_hydro = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    interpolated_hydro += combined_weights[i] * fields['sigma_hydro']
                interpolated_fields['sigma_hydro'] = interpolated_hydro
                hydro_stats['method'] = 'standard'
            
            # Compute additional metrics
            max_vm = np.max(interpolated_fields['von_mises'])
            max_hydro = np.max(np.abs(interpolated_fields['sigma_hydro']))
            
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
                'hydro_optimization': hydro_stats,
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_theta_degrees': source_theta_degrees,
                'source_distances': source_distances,
                'source_indices': source_indices,
                'source_fields': source_fields
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
# ENHANCED HEATMAP VISUALIZER WITH HYDROSTATIC ANALYSIS
# =============================================
class EnhancedHeatMapVisualizer:
    """Enhanced heat map visualizer with hydrostatic analysis dashboard"""
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
    
    def create_hydrostatic_analysis_dashboard(self, interpolated_fields, source_fields, 
                                             hydro_stats, target_angle, defect_type,
                                             figsize=(20, 16)):
        """Create comprehensive hydrostatic stress analysis dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25)
        
        # 1. Hydrostatic stress map (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        hydro_field = interpolated_fields['sigma_hydro']
        max_abs = np.max(np.abs(hydro_field))
        
        im1 = ax1.imshow(hydro_field, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs,
                        aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Hydrostatic Stress (GPa)')
        ax1.set_title(f'Hydrostatic Stress\nMethod: {hydro_stats.get("method", "standard")}', 
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.2)
        
        # 2. Tension/Compression separation (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        tension_mask = hydro_field > 0
        compression_mask = hydro_field < 0
        
        # Create RGB visualization
        rgb_image = np.zeros((*hydro_field.shape, 3))
        rgb_image[tension_mask, 0] = hydro_field[tension_mask] / np.max(hydro_field[tension_mask]) if np.any(tension_mask) else 0  # Red for tension
        rgb_image[compression_mask, 2] = -hydro_field[compression_mask] / np.max(-hydro_field[compression_mask]) if np.any(compression_mask) else 0  # Blue for compression
        
        ax2.imshow(rgb_image, aspect='equal', origin='lower')
        ax2.set_title('Tension (Red) / Compression (Blue)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        
        # Add tension/compression statistics
        tension_area = np.sum(tension_mask) / tension_mask.size * 100
        compression_area = np.sum(compression_mask) / compression_mask.size * 100
        neutral_area = 100 - tension_area - compression_area
        
        stats_text = (f"Tension: {tension_area:.1f}%\n"
                     f"Compression: {compression_area:.1f}%\n"
                     f"Neutral: {neutral_area:.1f}%")
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 3. Hydrostatic statistics (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        stats = hydro_stats.copy()
        # Remove method from display
        method = stats.pop('method', 'standard')
        
        stats_text = f"Hydrostatic Optimization Method: {method}\n\n"
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
            else:
                stats_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes,
                fontsize=12, family='monospace', fontweight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))
        ax3.set_title('Hydrostatic Optimization Statistics', fontsize=16, fontweight='bold', pad=20)
        
        # 4. Source hydrostatic comparison (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        if len(source_fields) > 0:
            # Plot distribution of source hydrostatic values
            source_hydro_means = []
            source_hydro_stds = []
            
            for fields in source_fields:
                hydro = fields['sigma_hydro']
                source_hydro_means.append(np.mean(hydro))
                source_hydro_stds.append(np.std(hydro))
            
            x_pos = np.arange(len(source_fields))
            ax4.bar(x_pos, source_hydro_means, yerr=source_hydro_stds, 
                   alpha=0.7, color='steelblue', edgecolor='black', capsize=5)
            ax4.axhline(y=np.mean(hydro_field), color='red', linestyle='--', linewidth=2, 
                       label=f'Interpolated Mean: {np.mean(hydro_field):.3f}')
            
            ax4.set_xlabel('Source Index')
            ax4.set_ylabel('Mean Hydrostatic Stress (GPa)')
            ax4.set_title('Source Hydrostatic Statistics', fontsize=16, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Histogram of hydrostatic values (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(hydro_field.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Hydrostatic Stress (GPa)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Hydrostatic Distribution', fontsize=16, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add Gaussian fit
        from scipy.stats import norm
        mu, std = norm.fit(hydro_field.flatten())
        x = np.linspace(np.min(hydro_field), np.max(hydro_field), 100)
        p = norm.pdf(x, mu, std)
        ax5.plot(x, p * len(hydro_field.flatten()) * (x[1]-x[0]), 'r-', linewidth=2,
                label=f'Gaussian Fit\nμ={mu:.3f}, σ={std:.3f}')
        ax5.legend()
        
        # 6. Cumulative distribution (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        sorted_values = np.sort(hydro_field.flatten())
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax6.plot(sorted_values, cdf, 'b-', linewidth=2)
        ax6.set_xlabel('Hydrostatic Stress (GPa)')
        ax6.set_ylabel('Cumulative Probability')
        ax6.set_title('Cumulative Distribution Function', fontsize=16, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add quartile markers
        for q in [0.25, 0.5, 0.75]:
            idx = int(q * len(sorted_values))
            ax6.axvline(x=sorted_values[idx], color='r', linestyle='--', alpha=0.5)
            ax6.text(sorted_values[idx], q, f'Q{int(q*100)}={sorted_values[idx]:.3f}', 
                    fontsize=10, ha='right', va='bottom')
        
        # 7. Gradient analysis (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        grad_x, grad_y = np.gradient(hydro_field)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        im7 = ax7.imshow(gradient_magnitude, cmap='hot', aspect='equal', 
                        interpolation='bilinear', origin='lower')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04, label='Gradient Magnitude (GPa/px)')
        ax7.set_title('Hydrostatic Stress Gradient', fontsize=16, fontweight='bold')
        ax7.set_xlabel('X Position')
        ax7.set_ylabel('Y Position')
        ax7.grid(True, alpha=0.2)
        
        # 8. Sign consistency analysis (bottom center)
        ax8 = fig.add_subplot(gs[2, 1])
        if len(source_fields) >= 2:
            # Calculate sign agreement between all source pairs
            sign_agreement = np.zeros(hydro_field.shape)
            source_hydro_fields = [fields['sigma_hydro'] for fields in source_fields]
            
            pair_count = 0
            for i in range(len(source_hydro_fields)):
                for j in range(i + 1, len(source_hydro_fields)):
                    agreement = (np.sign(source_hydro_fields[i]) == np.sign(source_hydro_fields[j])).astype(float)
                    sign_agreement += agreement
                    pair_count += 1
            
            if pair_count > 0:
                sign_agreement = sign_agreement / pair_count
            
            im8 = ax8.imshow(sign_agreement, cmap='viridis', vmin=0, vmax=1,
                           aspect='equal', interpolation='bilinear', origin='lower')
            plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, label='Sign Agreement Ratio')
            ax8.set_title('Source Sign Agreement', fontsize=16, fontweight='bold')
            ax8.set_xlabel('X Position')
            ax8.set_ylabel('Y Position')
            ax8.grid(True, alpha=0.2)
            
            # Add average agreement
            avg_agreement = np.mean(sign_agreement)
            ax8.text(0.02, 0.98, f'Avg Agreement: {avg_agreement:.3f}', 
                    transform=ax8.transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 9. Method comparison explanation (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        method_descriptions = {
            'standard': 'Standard linear interpolation. Simple but prone to sign cancellation.',
            'channel_split': 'Separates tension/compression channels. Reduces sign cancellation.',
            'sign_aware': 'Adjusts weights based on sign agreement. Better for mixed-sign regions.',
            'angular_filter': 'Uses only nearby angular sources. Physically more consistent.',
            'magnitude_preserving': 'Preserves stress magnitude while interpolating sign.',
            'enhanced_features': 'Uses hydrostatic statistics as additional features.'
        }
        
        method = hydro_stats.get('method', 'standard')
        description = method_descriptions.get(method, 'Standard linear interpolation.')
        
        explanation_text = (
            f"Current Method: {method}\n\n"
            f"{description}\n\n"
            f"Key Insights:\n"
            f"• Hydrostatic stress has both tension (+) and compression (-)\n"
            f"• Linear averaging can cancel opposite signs\n"
            f"• Method aims to preserve physical realism\n"
            f"• Check error metrics in comparison dashboard"
        )
        
        ax9.text(0.1, 0.5, explanation_text, transform=ax9.transAxes,
                fontsize=11, family='monospace', fontweight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2))
        ax9.set_title('Method Explanation', fontsize=16, fontweight='bold', pad=20)
        
        plt.suptitle(f'Hydrostatic Stress Analysis - θ={target_angle:.1f}°, {defect_type} - Method: {method}',
                    fontsize=24, fontweight='bold', y=0.98)
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
    #
    
    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info, 
                                   target_angle, defect_type, component='von_mises',
                                   cmap_name='viridis', figsize=(20, 15),
                                   ground_truth_index=None):
        """Create comprehensive comparison dashboard"""
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

# =============================================
# HYDROSTATIC ENHANCEMENT EXPLANATIONS
# =============================================
HYDROSTATIC_ENHANCEMENTS = {
    "standard": {
        "name": "Standard Linear Interpolation",
        "description": "Basic weighted average of source hydrostatic fields. Simple but prone to sign cancellation.",
        "explanation": """
        • Uses linear weighted average: Σ(w_i * σ_hydro_i)
        • Prone to sign cancellation when sources have opposite signs
        • Simple and computationally efficient
        • Best for sources with consistent sign patterns
        """,
        "pros": ["Simple", "Fast", "Deterministic"],
        "cons": ["Sign cancellation", "Poor for mixed-sign regions", "Blurs transitions"],
        "implementation": "Direct weighted average"
    },
    "channel_split": {
        "name": "Channel Splitting Method",
        "description": "Separates tension and compression channels, interpolates separately, then recombines.",
        "explanation": """
        • Splits each source: σ⁺ = max(σ, 0), σ⁻ = -min(σ, 0)
        • Interpolates tension and compression channels separately
        • Recombines: σ_interp = σ⁺_interp - σ⁻_interp
        • Reduces sign cancellation significantly
        • Preserves magnitude better than standard method
        """,
        "pros": ["Reduces sign cancellation", "Preserves magnitude", "Better physical realism"],
        "cons": ["Doubles computation", "May over-smooth", "Requires renormalization"],
        "implementation": "Separate interpolation of tension/compression channels"
    },
    "sign_aware": {
        "name": "Sign-Aware Weight Adjustment",
        "description": "Adjusts interpolation weights based on sign agreement between sources.",
        "explanation": """
        • Analyzes sign agreement between source pairs
        • In low-agreement regions, increases spatial locality weight
        • Dynamically adjusts weights to favor physically consistent sources
        • Better handles regions with conflicting sign patterns
        • Adaptive to local sign coherence
        """,
        "pros": ["Adaptive to sign patterns", "Better for mixed regions", "Preserves local structure"],
        "cons": ["Computationally intensive", "Requires pair-wise analysis", "May overfit to noise"],
        "implementation": "Weight adjustment based on sign agreement metrics"
    },
    "angular_filter": {
        "name": "Angular Distance Filtering",
        "description": "Uses only sources within specified angular distance from target.",
        "explanation": """
        • Filters sources by angular distance: Δθ < threshold
        • Uses only physically similar sources for interpolation
        • Reduces influence of dissimilar sources with different sign patterns
        • Particularly effective near habit plane (54.7°) where hydrostatic is sensitive
        • Threshold adjustable via parameter
        """,
        "pros": ["Physically motivated", "Reduces dissimilar source influence", "Simple to implement"],
        "cons": ["May discard useful information", "Sensitive to threshold choice", "Risk of too few sources"],
        "implementation": "Source filtering based on angular distance"
    },
    "magnitude_preserving": {
        "name": "Magnitude-Preserving Interpolation",
        "description": "Interpolates magnitude and sign separately to preserve stress intensity.",
        "explanation": """
        • Interpolates magnitude: |σ|_interp = Σ(w_i * |σ_i|)
        • Interpolates sign separately or uses weighted sign
        • Recombines: σ_interp = sign(σ_weighted) * |σ|_interp
        • Preserves stress intensity while allowing sign interpolation
        • Blends methods based on gradient magnitude
        """,
        "pros": ["Preserves magnitude", "Better for gradient regions", "Hybrid approach"],
        "cons": ["Complex implementation", "May create artificial patterns", "Blending parameter sensitive"],
        "implementation": "Separate magnitude/sign interpolation with gradient-based blending"
    },
    "enhanced_features": {
        "name": "Enhanced Feature Encoding",
        "description": "Adds hydrostatic-specific features to transformer encoding.",
        "explanation": """
        • Augments parameter encoding with hydrostatic statistics
        • Features: mean sign, tension ratio, variability
        • Transformer learns better similarities for signed fields
        • Requires updating input projection layer
        • More informed attention mechanism
        """,
        "pros": ["Better similarity learning", "Informed attention", "Holistic approach"],
        "cons": ["Breaks existing encoding", "Requires model adjustment", "Increased complexity"],
        "implementation": "Extended feature vector with hydrostatic statistics"
    }
}

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
                'interpolation_method': 'enhanced_transformer_spatial',
                'hydro_method': interpolation_result.get('hydro_optimization', {}).get('method', 'standard'),
                'visualization_params': visualization_params
            },
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'shape': result['shape'],
                'statistics': result['statistics'],
                'weights': result['weights'],
                'hydro_optimization': result.get('hydro_optimization', {}),
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
            hydro_method = export_data['metadata']['hydro_method']
            filename = f"enhanced_interpolation_theta_{theta}_{defect}_{hydro_method}_{timestamp}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        """Export flattened field data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            hydro_method = interpolation_result.get('hydro_optimization', {}).get('method', 'standard')
            filename = f"enhanced_stress_fields_theta_{theta}_{defect}_{hydro_method}_{timestamp}.csv"
        
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
# MAIN APPLICATION WITH HYDROSTATIC ENHANCEMENT
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Enhanced Transformer Stress Interpolation with Hydrostatic Optimization",
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
    .method-card {
        background-color: #F7FEF5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .method-title {
        color: #065F46;
        font-weight: 800;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    .pros-cons {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    .pros {
        background-color: #D1FAE5;
        padding: 0.8rem;
        border-radius: 6px;
    }
    .cons {
        background-color: #FEE2E2;
        padding: 0.8rem;
        border-radius: 6px;
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
    st.markdown('<h1 class="main-header">🔬 Enhanced Transformer Stress Interpolation with Hydrostatic Optimization</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
        <strong>🔬 Advanced stress interpolation with specialized hydrostatic optimization methods.</strong><br>
        • Load simulation files from numerical_solutions directory<br>
        • Interpolate stress fields at custom polar angles<br>
        • <strong>NEW:</strong> Choose from 6 hydrostatic interpolation enhancement methods<br>
        • Comprehensive analysis of hydrostatic stress behavior<br>
        • Compare different optimization strategies<br>
        • Publication-ready visualizations with detailed explanations
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        # Initialize with standard method
        st.session_state.transformer_interpolator = EnhancedTransformerSpatialInterpolator(
            spatial_sigma=0.2,
            locality_weight_factor=0.7,
            hydro_method="standard"
        )
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = EnhancedHeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'selected_ground_truth' not in st.session_state:
        st.session_state.selected_ground_truth = None
    if 'selected_hydro_method' not in st.session_state:
        st.session_state.selected_hydro_method = "standard"
    if 'hydro_params' not in st.session_state:
        st.session_state.hydro_params = {}
    
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
        
        st.divider()
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle
        custom_theta = st.slider(
            "Polar Angle θ (degrees)", 
            min_value=0.0, 
            max_value=180.0, 
            value=54.7,
            step=0.1,
            help="Angle in degrees (0° to 180°). Default habit plane is 54.7°"
        )
        
        # Defect type
        defect_type = st.selectbox(
            "Defect Type",
            options=['ISF', 'ESF', 'Twin', 'No Defect'],
            index=2,
            help="Type of crystal defect to simulate"
        )
        
        # Shape selection
        shape = st.selectbox(
            "Shape",
            options=['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle'],
            index=0,
            help="Geometry of the defect region"
        )
        
        # Kappa parameter
        kappa = st.slider(
            "Kappa (material property)", 
            min_value=0.1, 
            max_value=2.0, 
            value=0.6,
            step=0.01,
            help="Material stiffness parameter"
        )
        
        # Eigenstrain auto-calculation
        st.markdown("#### 🧮 Eigenstrain Calculation")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            auto_eigen = st.checkbox("Auto-calculate eigenstrain", value=True)
        with col_e2:
            if auto_eigen:
                # Auto-calculate based on defect type
                eigen_strain = {
                    'ISF': 0.289,
                    'ESF': 0.333,
                    'Twin': 0.707,
                    'No Defect': 0.0
                }[defect_type]
                st.metric("Eigenstrain ε₀", f"{eigen_strain:.3f}")
            else:
                eigen_strain = st.slider(
                    "Eigenstrain ε₀", 
                    min_value=0.0, 
                    max_value=3.0, 
                    value=0.707,
                    step=0.001
                )
        
        st.divider()
        
        # HYDROSTATIC ENHANCEMENT SELECTION
        st.markdown('<h2 class="section-header">💧 Hydrostatic Enhancement</h2>', unsafe_allow_html=True)
        
        # Method selection with detailed descriptions
        hydro_method_options = list(HYDROSTATIC_ENHANCEMENTS.keys())
        hydro_method_display = [HYDROSTATIC_ENHANCEMENTS[m]["name"] for m in hydro_method_options]
        
        selected_method_display = st.selectbox(
            "Hydrostatic Interpolation Method",
            options=hydro_method_display,
            index=0,
            help="Select enhancement method for hydrostatic stress interpolation"
        )
        
        # Map back to method key
        selected_method = hydro_method_options[hydro_method_display.index(selected_method_display)]
        st.session_state.selected_hydro_method = selected_method
        
        # Method-specific parameters
        st.session_state.hydro_params = {}
        
        if selected_method == "angular_filter":
            angular_threshold = st.slider(
                "Angular Distance Threshold (°)",
                min_value=1.0,
                max_value=45.0,
                value=10.0,
                step=1.0,
                help="Only use sources within this angular distance"
            )
            st.session_state.hydro_params['angular_threshold'] = angular_threshold
        
        # Show method details
        method_info = HYDROSTATIC_ENHANCEMENTS[selected_method]
        
        with st.expander(f"📚 About {method_info['name']}", expanded=False):
            st.markdown(f"**Description:** {method_info['description']}")
            st.markdown("**Explanation:**")
            st.markdown(method_info['explanation'])
            st.markdown("**Implementation:**")
            st.markdown(method_info['implementation'])
            
            # Pros and cons
            st.markdown("**Pros & Cons:**")
            col_pros, col_cons = st.columns(2)
            with col_pros:
                st.markdown("**✅ Pros:**")
                for pro in method_info['pros']:
                    st.markdown(f"• {pro}")
            with col_cons:
                st.markdown("**❌ Cons:**")
                for con in method_info['cons']:
                    st.markdown(f"• {con}")
        
        st.divider()
        
        # Transformer parameters
        st.markdown('<h2 class="section-header">🧠 Transformer Parameters</h2>', unsafe_allow_html=True)
        
        # Spatial locality parameters
        st.markdown("#### 📍 Spatial Locality Controls")
        spatial_sigma = st.slider(
            "Spatial Sigma", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.2,
            step=0.01,
            help="Controls the decay rate of spatial weights (higher = slower decay)"
        )
        
        spatial_weight_factor = st.slider(
            "Spatial Locality Weight Factor", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            step=0.01,
            help="Balance between transformer attention and spatial locality (0 = pure spatial, 1 = pure transformer)"
        )
        
        # Attention temperature
        temperature = st.slider(
            "Attention Temperature", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0,
            step=0.1,
            help="Softmax temperature for attention weights (lower = sharper distribution)"
        )
        
        # Update transformer parameters
        if st.button("🔄 Update All Parameters", type="primary", use_container_width=True):
            # Update transformer interpolator
            st.session_state.transformer_interpolator.set_spatial_parameters(
                spatial_sigma=spatial_sigma,
                locality_weight_factor=spatial_weight_factor
            )
            st.session_state.transformer_interpolator.set_hydro_method(
                selected_method,
                st.session_state.hydro_params
            )
            st.session_state.transformer_interpolator.temperature = temperature
            
            st.success(f"Parameters updated! Hydrostatic method: {method_info['name']}")
    
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
                st.metric("Hydro Method", 
                         HYDROSTATIC_ENHANCEMENTS[st.session_state.selected_hydro_method]["name"])
        
        # Display source information
        if st.session_state.solutions:
            source_thetas = []
            for sol in st.session_state.solutions:
                if 'params' in sol and 'theta' in sol['params']:
                    theta_deg = np.degrees(sol['params']['theta'])
                    source_thetas.append(theta_deg)
            
            if source_thetas:
                st.markdown(f"**Source Angles Range:** {min(source_thetas):.1f}° to {max(source_thetas):.1f}°")
                st.markdown(f"**Mean Source Angle:** {np.mean(source_thetas):.1f}°")
    
    # Run interpolation button
    st.markdown("---")
    col_run1, col_run2, col_run3 = st.columns([1, 2, 1])
    with col_run2:
        if st.button("🚀 Perform Enhanced Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner(f"Performing interpolation with {HYDROSTATIC_ENHANCEMENTS[st.session_state.selected_hydro_method]['name']}..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': eigen_strain,
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
                    }
                    
                    # Perform interpolation with selected method
                    result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        custom_theta,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        method_name = HYDROSTATIC_ENHANCEMENTS[st.session_state.selected_hydro_method]["name"]
                        st.success(f"Interpolation successful using {method_name}! Used {result['num_sources']} source solutions.")
                        st.session_state.selected_ground_truth = None
                    else:
                        st.error("Interpolation failed. Check the console for errors.")
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Results Overview", 
            "💧 Hydrostatic Analysis",
            "🎨 Visualization", 
            "⚖️ Weights Analysis",
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
                hydro_stats = result['statistics']['sigma_hydro']
                st.metric(
                    "Hydrostatic Range", 
                    f"{hydro_stats['max_tension']:.3f}/{hydro_stats['max_compression']:.3f} GPa"
                )
            with col3:
                st.metric(
                    "Mean Stress Magnitude", 
                    f"{result['statistics']['sigma_mag']['mean']:.3f} GPa"
                )
            with col4:
                hydro_method = result.get('hydro_optimization', {}).get('method', 'standard')
                method_name = HYDROSTATIC_ENHANCEMENTS[hydro_method]["name"]
                st.metric(
                    "Hydrostatic Method", 
                    method_name
                )
            
            # Target parameters and hydrostatic optimization details
            st.markdown("#### 🎯 Target Parameters & Hydrostatic Optimization")
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                st.markdown(f"""
                <div class="param-table">
                    <div class="param-key">Angle (θ)</div>
                    <div class="param-value">{result['target_angle']:.2f}°</div>
                    <div class="param-key">Defect Type</div>
                    <div class="param-value">{result['target_params']['defect_type']}</div>
                    <div class="param-key">Eigenstrain (ε₀)</div>
                    <div class="param-value">{result['target_params']['eps0']:.3f}</div>
                    <div class="param-key">Kappa (κ)</div>
                    <div class="param-value">{result['target_params']['kappa']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with param_col2:
                hydro_opt = result.get('hydro_optimization', {})
                method_name = HYDROSTATIC_ENHANCEMENTS[hydro_opt.get('method', 'standard')]["name"]
                
                opt_details = f"""
                <div class="param-table">
                    <div class="param-key">Hydrostatic Method</div>
                    <div class="param-value">{method_name}</div>
                """
                
                for key, value in hydro_opt.items():
                    if key != 'method':
                        if isinstance(value, float):
                            display_value = f"{value:.4f}"
                        else:
                            display_value = str(value)
                        opt_details += f"""
                        <div class="param-key">{key.replace('_', ' ').title()}</div>
                        <div class="param-value">{display_value}</div>
                        """
                
                opt_details += "</div>"
                st.markdown(opt_details, unsafe_allow_html=True)
            
            # Quick preview of stress fields
            st.markdown("#### 👀 Quick Preview")
            preview_component = st.selectbox(
                "Preview Component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0,
                key="preview_component"
            )
            
            if preview_component in result['fields']:
                # Use diverging colormap for hydrostatic
                cmap = 'RdBu_r' if preview_component == 'sigma_hydro' else 'viridis'
                
                fig_preview = st.session_state.heatmap_visualizer.create_stress_heatmap(
                    result['fields'][preview_component],
                    title=f"{preview_component.replace('_', ' ').title()} Stress",
                    cmap_name=cmap,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(10, 8)
                )
                st.pyplot(fig_preview)
        
        with tab2:
            # Hydrostatic Analysis Tab
            st.markdown('<h2 class="section-header">💧 Hydrostatic Stress Analysis</h2>', unsafe_allow_html=True)
            
            # Method comparison explanation
            hydro_method = result.get('hydro_optimization', {}).get('method', 'standard')
            method_info = HYDROSTATIC_ENHANCEMENTS[hydro_method]
            
            st.markdown(f"""
            <div class="method-card">
                <div class="method-title">{method_info['name']}</div>
                <p><strong>Description:</strong> {method_info['description']}</p>
                <p><strong>Why it helps:</strong> {method_info['explanation'].split('•')[1].strip() if '•' in method_info['explanation'] else method_info['explanation'][:100]}...</p>
                
                <div class="pros-cons">
                    <div class="pros">
                        <strong>✅ Pros:</strong><br>
                        {"<br>".join([f"• {pro}" for pro in method_info['pros']])}
                    </div>
                    <div class="cons">
                        <strong>❌ Cons:</strong><br>
                        {"<br>".join([f"• {con}" for con in method_info['cons']])}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create hydrostatic analysis dashboard
            st.markdown("#### 📊 Hydrostatic Analysis Dashboard")
            
            fig_hydro_analysis = st.session_state.heatmap_visualizer.create_hydrostatic_analysis_dashboard(
                result['fields'],
                result['source_fields'],
                result.get('hydro_optimization', {}),
                result['target_angle'],
                result['target_params']['defect_type']
            )
            st.pyplot(fig_hydro_analysis)
            
            # Why hydrostatic is challenging
            with st.expander("🔍 Why Hydrostatic Stress is Challenging for Interpolation", expanded=False):
                st.markdown("""
                **Physics-Based Reasons:**
                
                1. **Signed Quantity:** Hydrostatic stress can be positive (tension) or negative (compression), unlike von Mises which is always positive
                2. **Sign Cancellation:** Linear averaging of opposite signs can cancel out, creating artificial zero regions
                3. **Sharp Transitions:** Hydrostatic fields often have sharp sign changes near defects
                4. **Sensitivity to Orientation:** Small angular changes can flip hydrostatic sign patterns
                
                **Interpolation Challenges:**
                
                1. **Linear vs. Non-linear:** Standard interpolation assumes linearity, but sign flips are non-linear
                2. **Weight Distribution:** Weights optimized for magnitude may not work well for sign patterns
                3. **Source Alignment:** Sources with different sign patterns average poorly
                
                **Enhanced Methods Address These By:**
                
                - **Channel Splitting:** Separates tension/compression to avoid cancellation
                - **Sign-Aware Weights:** Adjusts weights based on sign agreement
                - **Angular Filtering:** Uses only physically similar sources
                - **Magnitude Preservation:** Maintains stress intensity while interpolating sign
                """)
            
            # Method comparison suggestions
            with st.expander("🔄 Try Different Methods for Comparison", expanded=False):
                st.markdown("""
                **For your current parameters, consider trying:**
                
                1. **If hydrostatic has mixed signs:** Try **Channel Splitting** or **Sign-Aware** methods
                2. **If sources are closely spaced in angle:** Try **Angular Filtering** with 5-10° threshold
                3. **If you want simplest improvement:** **Channel Splitting** usually gives good results
                4. **For research/comparison:** Try all methods and compare error metrics
                
                **Quick comparison workflow:**
                1. Run interpolation with current method
                2. Note hydrostatic error metrics from Comparison Dashboard
                3. Change method in sidebar and re-run
                4. Compare error metrics (MSE, MAE, correlation)
                5. Export results for systematic comparison
                """)
        
        with tab3:
            # Visualization tab
            st.markdown('<h2 class="section-header">🎨 Advanced Visualization</h2>', unsafe_allow_html=True)
            
            # Visualization controls
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            with col_viz1:
                component = st.selectbox(
                    "Stress Component",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0,
                    key="viz_component"
                )
            with col_viz2:
                cmap_category = st.selectbox(
                    "Colormap Category",
                    options=list(COLORMAP_OPTIONS.keys()),
                    index=0,
                    key="cmap_category"
                )
                cmap_options = COLORMAP_OPTIONS[cmap_category]
            with col_viz3:
                cmap_name = st.selectbox(
                    "Colormap",
                    options=cmap_options,
                    index=0,
                    key="cmap_name"
                )
            
            # Auto-select diverging colormap for hydrostatic
            if component == 'sigma_hydro' and cmap_name not in COLORMAP_OPTIONS['Diverging']:
                st.info("💡 For hydrostatic stress, consider using a diverging colormap (RdBu, coolwarm, etc.) to better show tension/compression.")
            
            # Visualization type selection
            viz_type = st.radio(
                "Visualization Type",
                options=["2D Heatmap", "3D Surface", "Interactive Heatmap", "Interactive 3D", "Angular Orientation"],
                horizontal=True
            )
            
            if component in result['fields']:
                stress_field = result['fields'][component]
                
                if viz_type == "2D Heatmap":
                    # 2D heatmap
                    fig_2d = st.session_state.heatmap_visualizer.create_stress_heatmap(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(12, 10)
                    )
                    st.pyplot(fig_2d)
                    
                elif viz_type == "3D Surface":
                    # 3D surface plot
                    from matplotlib import cm
                    fig_3d, ax_3d = plt.subplots(figsize=(14, 10), subplot_kw={'projection': '3d'})
                    
                    # Create meshgrid
                    x = np.arange(stress_field.shape[1])
                    y = np.arange(stress_field.shape[0])
                    X, Y = np.meshgrid(x, y)
                    
                    # Plot surface
                    surf = ax_3d.plot_surface(X, Y, stress_field, cmap=cmap_name,
                                             linewidth=0, antialiased=True, alpha=0.8)
                    
                    # Add colorbar
                    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Stress (GPa)')
                    
                    # Labels and title
                    ax_3d.set_xlabel('X Position')
                    ax_3d.set_ylabel('Y Position')
                    ax_3d.set_zlabel('Stress (GPa)')
                    ax_3d.set_title(f'{component.replace("_", " ").title()} Stress - θ={result["target_angle"]:.1f}°', 
                                   fontsize=16, fontweight='bold')
                    
                    st.pyplot(fig_3d)
                    
                elif viz_type == "Interactive Heatmap":
                    # Interactive heatmap
                    fig_interactive = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        width=800,
                        height=700
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                
                elif viz_type == "Angular Orientation":
                    # Angular orientation plot
                    fig_angular, ax_angular = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
                    
                    # Convert target angle to radians
                    theta_rad = np.radians(result['target_angle'])
                    
                    # Plot the defect orientation
                    ax_angular.arrow(theta_rad, 0.8, 0, 0.6, width=0.02, 
                                   color='red', alpha=0.8, label=f'Target: {result["target_angle"]:.1f}°')
                    
                    # Plot habit plane
                    habit_plane_rad = np.radians(54.7)
                    ax_angular.arrow(habit_plane_rad, 0.8, 0, 0.6, width=0.02, 
                                   color='blue', alpha=0.5, label='Habit Plane (54.7°)')
                    
                    # Plot source angles
                    source_angles = result.get('source_theta_degrees', [])
                    if source_angles:
                        source_rad = np.radians(source_angles)
                        weights = result['weights']['combined']
                        sizes = 50 + 150 * np.array(weights) / np.max(weights)
                        ax_angular.scatter(source_rad, [0.5] * len(source_rad), 
                                         s=sizes, alpha=0.6, c='green', edgecolors='black', label='Sources')
                    
                    # Customize plot
                    ax_angular.set_title(f'Angular Orientation\nθ={result["target_angle"]:.1f}°, {result["target_params"]["defect_type"]}', 
                                       fontsize=16, fontweight='bold', pad=20)
                    ax_angular.set_theta_zero_location('N')
                    ax_angular.set_theta_direction(-1)
                    ax_angular.set_ylim(0, 1.5)
                    ax_angular.grid(True, alpha=0.3)
                    ax_angular.legend(loc='upper right', fontsize=10)
                    
                    st.pyplot(fig_angular)
            
            # Comparison of all components
            st.markdown("#### 🔄 Component Comparison")
            if st.button("Show All Components Comparison", key="show_all_components"):
                fig_all, axes_all = plt.subplots(1, 3, figsize=(18, 6))
                
                components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude']
                cmaps = ['viridis', 'RdBu_r', 'plasma']
                
                for idx, (comp, title, cmap) in enumerate(zip(components, titles, cmaps)):
                    ax = axes_all[idx]
                    field = result['fields'][comp]
                    
                    # Determine scaling
                    if comp == 'sigma_hydro':
                        vmax = np.max(np.abs(field))
                        vmin = -vmax
                    else:
                        vmax = np.max(field)
                        vmin = np.min(field)
                    
                    im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                                  aspect='equal', interpolation='bilinear', origin='lower')
                    plt.colorbar(im, ax=ax, label='Stress (GPa)')
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.grid(True, alpha=0.2)
                
                plt.suptitle(f'Stress Component Comparison - θ={result["target_angle"]:.1f}°, {result["target_params"]["defect_type"]}',
                           fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig_all)
        
        with tab4:
            # Weights analysis tab
            st.markdown('<h2 class="section-header">⚖️ Weight Distribution Analysis</h2>', unsafe_allow_html=True)
            
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
                    st.metric("Top Contributor", f"Source {max_weight_idx}")
                
                # Weight distribution plot
                st.markdown("#### 📊 Weight Distribution")
                fig_weights, ax_weights = plt.subplots(figsize=(12, 6))
                
                x = range(len(weights['combined']))
                width = 0.25
                
                # Plot different weight types
                ax_weights.bar([i - width for i in x], weights['transformer'], width, 
                              label='Transformer Weights', alpha=0.7, color='blue')
                ax_weights.bar(x, weights['positional'], width, 
                              label='Spatial Weights', alpha=0.7, color='green')
                ax_weights.bar([i + width for i in x], weights['combined'], width, 
                              label='Combined Weights', alpha=0.7, color='red')
                
                ax_weights.set_xlabel('Source Index')
                ax_weights.set_ylabel('Weight')
                ax_weights.set_title('Weight Distribution Across Sources', fontsize=16, fontweight='bold')
                ax_weights.legend()
                ax_weights.grid(True, alpha=0.3)
                
                # Add theta labels
                if 'source_theta_degrees' in result:
                    for i, theta in enumerate(result['source_theta_degrees']):
                        ax_weights.text(i, max(weights['combined'][i], weights['transformer'][i], weights['positional'][i]) + 0.01,
                                      f'θ={theta:.0f}°', ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig_weights)
                
                # Top contributors table
                st.markdown("#### 🏆 Top 5 Contributors")
                weight_data = []
                for i in range(len(weights['combined'])):
                    weight_data.append({
                        'Source': i,
                        'Combined Weight': weights['combined'][i],
                        'Transformer Weight': weights['transformer'][i],
                        'Spatial Weight': weights['positional'][i],
                        'Theta (°)': result['source_theta_degrees'][i] if i < len(result['source_theta_degrees']) else 0,
                        'Distance (°)': result['source_distances'][i] if i < len(result['source_distances']) else 0
                    })
                
                df_weights = pd.DataFrame(weight_data)
                df_weights = df_weights.sort_values('Combined Weight', ascending=False).head(5)
                st.dataframe(df_weights.style.format({
                    'Combined Weight': '{:.4f}',
                    'Transformer Weight': '{:.4f}',
                    'Spatial Weight': '{:.4f}',
                    'Theta (°)': '{:.1f}',
                    'Distance (°)': '{:.1f}'
                }))
        
        with tab5:
            # Comparison dashboard
            st.markdown('<h2 class="section-header">🔄 Comparison Dashboard</h2>', unsafe_allow_html=True)
            
            # Ground truth selection
            st.markdown("#### 🎯 Select Ground Truth Source")
            
            if 'source_theta_degrees' in result and result['source_theta_degrees']:
                # Create dropdown options
                ground_truth_options = []
                for i, theta in enumerate(result['source_theta_degrees']):
                    distance = result['source_distances'][i]
                    weight = result['weights']['combined'][i]
                    ground_truth_options.append(
                        f"Source {i}: θ={theta:.1f}° (Δ={distance:.1f}°, weight={weight:.3f})"
                    )
                
                selected_option = st.selectbox(
                    "Choose ground truth source:",
                    options=ground_truth_options,
                    index=0 if not st.session_state.selected_ground_truth else st.session_state.selected_ground_truth,
                    key="ground_truth_select"
                )
                
                # Parse selected index
                selected_index = int(selected_option.split(":")[0].split(" ")[1])
                st.session_state.selected_ground_truth = selected_index
                
                # Display selected source info
                selected_theta = result['source_theta_degrees'][selected_index]
                selected_distance = result['source_distances'][selected_index]
                selected_weight = result['weights']['combined'][selected_index]
                
                col_gt1, col_gt2, col_gt3, col_gt4 = st.columns(4)
                with col_gt1:
                    st.metric("Selected Source", selected_index)
                with col_gt2:
                    st.metric("Source Angle", f"{selected_theta:.1f}°")
                with col_gt3:
                    st.metric("Angular Distance", f"{selected_distance:.1f}°")
                with col_gt4:
                    st.metric("Contribution Weight", f"{selected_weight:.3f}")
                
                # Visualization options for comparison
                st.markdown("#### 🎨 Comparison Visualization")
                comp_component = st.selectbox(
                    "Component for Comparison",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=1,  # Default to hydrostatic for comparison
                    key="comp_component"
                )
                
                # Use appropriate colormap for component
                comp_cmap = 'RdBu_r' if comp_component == 'sigma_hydro' else 'viridis'
                
                # Create comparison dashboard
                if comp_component in result['fields']:
                    # Prepare source info for dashboard
                    source_info = {
                        'theta_degrees': result['source_theta_degrees'],
                        'distances': result['source_distances'],
                        'weights': result['weights'],
                        'statistics': result['statistics']
                    }
                    
                    # Create the comparison dashboard
                    fig_comparison = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                        interpolated_fields=result['fields'],
                        source_fields=result['source_fields'],
                        source_info=source_info,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        component=comp_component,
                        cmap_name=comp_cmap,
                        figsize=(20, 15),
                        ground_truth_index=selected_index
                    )
                    
                    st.pyplot(fig_comparison)
                    
                    # Calculate and display detailed error metrics
                    if selected_index < len(result['source_fields']):
                        ground_truth_field = result['source_fields'][selected_index][comp_component]
                        interpolated_field = result['fields'][comp_component]
                        
                        # Calculate errors
                        error_field = interpolated_field - ground_truth_field
                        mse = np.mean(error_field**2)
                        mae = np.mean(np.abs(error_field))
                        rmse = np.sqrt(mse)
                        
                        # Calculate correlation
                        from scipy.stats import pearsonr
                        try:
                            corr_coef, _ = pearsonr(ground_truth_field.flatten(), interpolated_field.flatten())
                        except:
                            corr_coef = 0.0
                        
                        # Display metrics
                        st.markdown("#### 📊 Error Metrics")
                        err_col1, err_col2, err_col3, err_col4 = st.columns(4)
                        with err_col1:
                            st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
                        with err_col2:
                            st.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
                        with err_col3:
                            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.6f}")
                        with err_col4:
                            st.metric("Pearson Correlation", f"{corr_coef:.4f}")
                        
                        # Hydrostatic-specific analysis
                        if comp_component == 'sigma_hydro':
                            with st.expander("💧 Hydrostatic-Specific Analysis", expanded=True):
                                # Sign agreement analysis
                                gt_sign = np.sign(ground_truth_field)
                                interp_sign = np.sign(interpolated_field)
                                sign_agreement = np.mean(gt_sign == interp_sign)
                                
                                # Tension/compression analysis
                                gt_tension = np.sum(ground_truth_field > 0)
                                gt_compression = np.sum(ground_truth_field < 0)
                                interp_tension = np.sum(interpolated_field > 0)
                                interp_compression = np.sum(interpolated_field < 0)
                                
                                col_h1, col_h2, col_h3, col_h4 = st.columns(4)
                                with col_h1:
                                    st.metric("Sign Agreement", f"{sign_agreement:.3f}")
                                with col_h2:
                                    st.metric("GT Tension Pixels", f"{gt_tension}")
                                with col_h3:
                                    st.metric("GT Compression Pixels", f"{gt_compression}")
                                with col_h4:
                                    st.metric("Interp Tension Pixels", f"{interp_tension}")
        
        with tab6:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            
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
                    'visualization_type': viz_type if 'viz_type' in locals() else '2D Heatmap'
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
                
            elif export_format == "PNG (Visualizations)":
                # PNG export options
                st.markdown("#### 📸 Select Visualizations to Export")
                
                export_plots = st.multiselect(
                    "Choose plots to export:",
                    options=[
                        "Von Mises Heatmap",
                        "Hydrostatic Heatmap", 
                        "Stress Magnitude Heatmap",
                        "Hydrostatic Analysis Dashboard",
                        "Comparison Dashboard",
                        "Weight Distribution",
                        "All Components Comparison"
                    ],
                    default=["Hydrostatic Analysis Dashboard", "Comparison Dashboard"]
                )
                
                if st.button("🖼️ Generate and Download Visualizations", use_container_width=True):
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        plot_count = 0
                        
                        # Generate selected plots
                        if "Von Mises Heatmap" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                result['fields']['von_mises'],
                                title="Von Mises Stress",
                                cmap_name='viridis',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"von_mises_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Hydrostatic Heatmap" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                result['fields']['sigma_hydro'],
                                title="Hydrostatic Stress",
                                cmap_name='RdBu_r',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"hydrostatic_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Stress Magnitude Heatmap" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                result['fields']['sigma_mag'],
                                title="Stress Magnitude",
                                cmap_name='plasma',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"stress_magnitude_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Hydrostatic Analysis Dashboard" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_hydrostatic_analysis_dashboard(
                                result['fields'],
                                result['source_fields'],
                                result.get('hydro_optimization', {}),
                                result['target_angle'],
                                result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"hydrostatic_analysis_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Comparison Dashboard" in export_plots and st.session_state.selected_ground_truth is not None:
                            source_info = {
                                'theta_degrees': result['source_theta_degrees'],
                                'distances': result['source_distances'],
                                'weights': result['weights'],
                                'statistics': result['statistics']
                            }
                            
                            fig = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                                interpolated_fields=result['fields'],
                                source_fields=result['source_fields'],
                                source_info=source_info,
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                component='sigma_hydro',
                                cmap_name='RdBu_r',
                                ground_truth_index=st.session_state.selected_ground_truth
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"comparison_dashboard_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Weight Distribution" in export_plots:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            x = range(len(result['weights']['combined']))
                            width = 0.25
                            
                            ax.bar([i - width for i in x], result['weights']['transformer'], width, 
                                  label='Transformer Weights', alpha=0.7, color='blue')
                            ax.bar(x, result['weights']['positional'], width, 
                                  label='Spatial Weights', alpha=0.7, color='green')
                            ax.bar([i + width for i in x], result['weights']['combined'], width, 
                                  label='Combined Weights', alpha=0.7, color='red')
                            
                            ax.set_xlabel('Source Index')
                            ax.set_ylabel('Weight')
                            ax.set_title('Weight Distribution', fontsize=16, fontweight='bold')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"weight_distribution_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "All Components Comparison" in export_plots:
                            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                            components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                            titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude']
                            cmaps = ['viridis', 'RdBu_r', 'plasma']
                            
                            for idx, (comp, title, cmap) in enumerate(zip(components, titles, cmaps)):
                                ax = axes[idx]
                                field = result['fields'][comp]
                                
                                if comp == 'sigma_hydro':
                                    vmax = np.max(np.abs(field))
                                    vmin = -vmax
                                else:
                                    vmax = np.max(field)
                                    vmin = np.min(field)
                                
                                im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                                              aspect='equal', interpolation='bilinear', origin='lower')
                                plt.colorbar(im, ax=ax, label='Stress (GPa)')
                                ax.set_title(title, fontsize=14, fontweight='bold')
                                ax.set_xlabel('X Position')
                                ax.set_ylabel('Y Position')
                                ax.grid(True, alpha=0.2)
                            
                            plt.suptitle(f'Stress Component Comparison - θ={result["target_angle"]:.1f}°', fontsize=16, fontweight='bold', y=1.02)
                            plt.tight_layout()
                            
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"all_components_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                    
                    zip_buffer.seek(0)
                    
                    # Download button for zip file
                    st.download_button(
                        label=f"📦 Download {plot_count} Visualization(s) as ZIP",
                        data=zip_buffer,
                        file_name=f"enhanced_visualizations_theta_{result['target_angle']:.1f}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    st.success(f"Generated {plot_count} visualization(s) for download.")
    
    else:
        # No results yet - show instructions
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); border-radius: 20px; color: white;">
            <h2>🚀 Ready to Begin Enhanced Interpolation!</h2>
            <p style="font-size: 1.2rem; margin-bottom: 30px;">
                Follow these steps to start interpolating stress fields with hydrostatic optimization:
            </p>
            <ol style="text-align: left; display: inline-block; font-size: 1.1rem;">
                <li>Load simulation files from the sidebar</li>
                <li>Configure target parameters (angle, defect type, etc.)</li>
                <li><strong>NEW:</strong> Select hydrostatic enhancement method from dropdown</li>
                <li>Adjust transformer and spatial locality parameters</li>
                <li>Click "Perform Enhanced Interpolation"</li>
                <li>Explore results in the tabs above</li>
            </ol>
            <p style="margin-top: 30px; font-size: 1.1rem;">
                <strong>Key Feature:</strong> Choose from 6 specialized hydrostatic interpolation methods to improve accuracy for signed stress fields.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Method comparison table
        st.markdown("### 📊 Hydrostatic Enhancement Methods Comparison")
        
        methods_data = []
        for method_key, method_info in HYDROSTATIC_ENHANCEMENTS.items():
            methods_data.append({
                'Method': method_info['name'],
                'Description': method_info['description'][:100] + "...",
                'Key Feature': method_info['explanation'].split('•')[1].strip()[:80] + "..." if '•' in method_info['explanation'] else method_info['explanation'][:80] + "...",
                'Pros': len(method_info['pros']),
                'Cons': len(method_info['cons']),
                'Complexity': ['Low', 'Medium', 'High'][min(2, len(method_info['pros']) + len(method_info['cons']) - 2)]
            })
        
        df_methods = pd.DataFrame(methods_data)
        st.dataframe(df_methods, use_container_width=True)
        
        # Quick start guide
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            #### 🎯 Recommended Starting Points
            
            **For general use:**
            - **Channel Splitting**: Best balance of improvement vs complexity
            - **Angular Filtering**: Good for closely spaced sources
            
            **For research/comparison:**
            - Try all methods with same parameters
            - Compare error metrics in Comparison Dashboard
            - Export results for systematic analysis
            
            **For specific cases:**
            - Mixed signs: Channel Splitting or Sign-Aware
            - Clustered sources: Angular Filtering
            - Maximum accuracy: Enhanced Features
            """)
        
        with col_guide2:
            st.markdown("""
            #### 🔍 Understanding Hydrostatic Challenges
            
            **Why hydrostatic is hard:**
            1. **Signed quantity**: Can be + (tension) or - (compression)
            2. **Sign cancellation**: Averaging +5 and -5 gives 0, not realistic
            3. **Sharp transitions**: Signs change abruptly near defects
            4. **Orientation sensitivity**: Small angle changes affect sign patterns
            
            **Enhanced methods address:**
            - **Channel Splitting**: Avoids sign cancellation
            - **Sign-Aware**: Adapts to local sign patterns
            - **Angular Filtering**: Uses physically similar sources
            - **Magnitude Preserving**: Maintains stress intensity
            """)
    
    # Footer
    st.divider()
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("**🔬 Enhanced Transformer Interpolator**")
        st.markdown("Version 3.0.0")
    with col_footer2:
        st.markdown("**💧 Hydrostatic Optimization:**")
        st.markdown("• 6 specialized methods")
        st.markdown("• Detailed analysis dashboard")
        st.markdown("• Comparative error metrics")
    with col_footer3:
        st.markdown("**📊 Features:**")
        st.markdown("• Adjustable spatial locality")
        st.markdown("• Comprehensive comparison")
        st.markdown("• Publication-ready plots")

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
