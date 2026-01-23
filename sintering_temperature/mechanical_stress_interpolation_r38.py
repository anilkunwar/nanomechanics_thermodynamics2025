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
    """Transformer-inspired stress interpolator with enhanced spatial locality regularization"""
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0, locality_weight_factor=0.7):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor  # Control factor for spatial vs attention weights
        
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
        """Compute spatial locality weights with enhanced angular distance weighting"""
        weights = []
        
        # Get target angle in degrees and normalize to [0, 360)
        target_theta = target_params.get('theta', 0.0)
        target_theta_deg = np.degrees(target_theta) % 360
        
        # Habit plane angle (54.7° for martensitic transformations)
        habit_plane_angle = 54.7
        
        for src in source_params:
            # Get source angle in degrees and normalize to [0, 360)
            src_theta = src.get('theta', 0.0)
            src_theta_deg = np.degrees(src_theta) % 360
            
            # Calculate angular distance considering cyclic nature
            raw_diff = abs(src_theta_deg - target_theta_deg)
            angular_diff = min(raw_diff, 360 - raw_diff)
            
            # Enhanced angular distance weighting with aggressive prioritization of proximal angles
            if angular_diff <= 1.0:  # Very very close (within 1 degree)
                angular_weight = 0.95
            elif angular_diff <= 2.5:  # Very close (within 2.5 degrees)
                angular_weight = 0.9
            elif angular_diff <= 5.0:  # Close (within 5 degrees)
                angular_weight = 0.8
            elif angular_diff <= 7.5:  # Moderately close (within 7.5 degrees)
                angular_weight = 0.7
            elif angular_diff <= 10.0:  # Somewhat close (within 10 degrees)
                angular_weight = 0.5
            elif angular_diff <= 15.0:  # Medium distance
                angular_weight = 0.3
            elif angular_diff <= 20.0:  # Somewhat far
                angular_weight = 0.15
            elif angular_diff <= 30.0:  # Far
                angular_weight = 0.08
            elif angular_diff <= 45.0:  # Very far
                angular_weight = 0.04
            else:  # Extremely far
                angular_weight = 0.01
            
            # Special handling for habit plane (54.7°) proximity
            # Check if either source or target is near habit plane
            habit_proximity_bonus = 0.0
            
            # Distance from source to habit plane
            src_to_habit = abs(src_theta_deg - habit_plane_angle)
            src_to_habit = min(src_to_habit, 360 - src_to_habit)
            
            # Distance from target to habit plane
            tgt_to_habit = abs(target_theta_deg - habit_plane_angle)
            tgt_to_habit = min(tgt_to_habit, 360 - tgt_to_habit)
            
            # If both are near habit plane, give extra bonus
            if src_to_habit <= 10.0 and tgt_to_habit <= 10.0:
                habit_proximity_bonus = 0.15 * (1.0 - max(src_to_habit, tgt_to_habit) / 10.0)
            
            # Apply habit plane proximity bonus
            angular_weight = min(1.0, angular_weight + habit_proximity_bonus)
            
            # Consider symmetry relationships (some angles may be equivalent)
            # Check if source and target are symmetric about habit plane
            if abs((src_theta_deg + target_theta_deg) / 2.0 - habit_plane_angle) <= 5.0:
                symmetry_bonus = 0.08
                angular_weight = min(1.0, angular_weight + symmetry_bonus)
            
            # Other parameter distances (with reduced importance compared to angular distance)
            param_dist = 0.0
            key_params = ['eps0', 'kappa', 'defect_type', 'shape']
            for param in key_params:
                src_val = src.get(param)
                tgt_val = target_params.get(param)
                if src_val is not None and tgt_val is not None:
                    if param == 'defect_type':
                        if src_val != tgt_val:
                            param_dist += 0.1  # Small penalty for different defect types
                    elif param == 'shape':
                        if src_val != tgt_val:
                            param_dist += 0.1  # Small penalty for different shapes
                    elif param == 'eps0':
                        eps0_diff = abs(src_val - tgt_val)
                        param_dist += eps0_diff / 3.0 * 0.1  # Reduced weight
                    elif param == 'kappa':
                        kappa_diff = abs(src_val - tgt_val)
                        param_dist += kappa_diff / 2.0 * 0.1  # Reduced weight
            
            # Combine with Gaussian kernel for smooth transitions
            param_weight = np.exp(-0.5 * (param_dist / self.spatial_sigma) ** 2)
            
            # Final combination: Angular weight dominates (85-90%), parameters contribute (10-15%)
            # Give even more weight to angular proximity for very close angles
            if angular_diff <= 5.0:
                # For very close angles, give more weight to angular proximity
                combined_weight = 0.9 * angular_weight + 0.1 * param_weight
            elif angular_diff <= 10.0:
                combined_weight = 0.85 * angular_weight + 0.15 * param_weight
            else:
                combined_weight = 0.8 * angular_weight + 0.2 * param_weight
            
            weights.append(combined_weight)
        
        return np.array(weights)
    
    def visualize_angular_weighting(self, target_angle_deg=54.7, figsize=(12, 8)):
        """Visualize the enhanced angular weighting function"""
        angles = np.linspace(0, 180, 361)
        weights = []
        for angle in angles:
            # Calculate angular distance
            angular_diff = abs(angle - target_angle_deg)
            angular_diff = min(angular_diff, 360 - angular_diff)
            
            # Apply enhanced weighting function
            if angular_diff <= 1.0:
                weight = 0.95
            elif angular_diff <= 2.5:
                weight = 0.9
            elif angular_diff <= 5.0:
                weight = 0.8
            elif angular_diff <= 7.5:
                weight = 0.7
            elif angular_diff <= 10.0:
                weight = 0.5
            elif angular_diff <= 15.0:
                weight = 0.3
            elif angular_diff <= 20.0:
                weight = 0.15
            elif angular_diff <= 30.0:
                weight = 0.08
            elif angular_diff <= 45.0:
                weight = 0.04
            else:
                weight = 0.01
            
            weights.append(weight)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(angles, weights, 'b-', linewidth=3, label='Spatial Weight')
        ax.axvline(x=target_angle_deg, color='r', linestyle='--', linewidth=2,
                  label=f'Target: {target_angle_deg}°')
        ax.axvline(x=54.7, color='g', linestyle='-.', linewidth=2,
                  label='Habit Plane: 54.7°')
        ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Spatial Weight', fontsize=14, fontweight='bold')
        ax.set_title(f'Enhanced Angular Weighting Function\nTarget Angle: {target_angle_deg}°',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 1.0])
        
        # Add annotations for weight regions
        regions = [
            (1.0, 0.95, 'Very Very Close (≤1°)'),
            (2.5, 0.9, 'Very Close (≤2.5°)'),
            (5.0, 0.8, 'Close (≤5.0°)'),
            (10.0, 0.5, 'Moderate (≤10.0°)'),
            (20.0, 0.15, 'Far (≤20.0°)'),
            (45.0, 0.04, 'Very Far (≤45.0°)')
        ]
        for x, y, label in regions:
            ax.annotate(label, xy=(target_angle_deg + x, y),
                       xytext=(target_angle_deg + x + 5, y + 0.05),
                       arrowprops=dict(arrowstyle='->', color='gray'),
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
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
            angle_diff = min(angle_diff, 360 - angle_diff)  # Handle cyclic nature
            features.append(np.exp(-angle_diff / 45.0))
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))  # FIX: Added this feature
            
            # Habit plane proximity (1 feature)
            habit_distance = abs(theta_deg - 54.7)
            habit_distance = min(habit_distance, 360 - habit_distance)  # Handle cyclic nature
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
            
            # Compute enhanced positional weights with aggressive angular distance weighting
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
            
            # Apply locality weight factor to balance spatial and transformer weights
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
            target_theta_deg = np.degrees(target_theta_rad) % 360  # Normalize to [0, 360)
            
            for src in source_params:
                theta_rad = src.get('theta', 0.0)
                theta_deg = np.degrees(theta_rad) % 360  # Normalize to [0, 360)
                source_theta_degrees.append(theta_deg)
                
                # Calculate angular distance (considering cyclic nature)
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
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                all_values.append(gt_field)
        vmin = min(np.nanmin(field) for field in all_values)
        vmax = max(np.nanmax(field) for field in all_values)
        
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
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
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
                ax2.text(0.5, 0.5, f'Component "{component}"\nmissing in ground truth',
                        ha='center', va='center', fontsize=14, fontweight='bold')
                ax2.set_axis_off()
        else:
            ax2.text(0.5, 0.5, 'Select Ground Truth Source',
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax2.set_title('Ground Truth Selection', fontsize=16, fontweight='bold')
            ax2.set_axis_off()
        
        # 3. Difference plot (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                diff_field = interpolated_fields[component] - gt_field
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
                ax3.text(0.5, 0.5, 'Ground truth missing\nfor difference plot',
                        ha='center', va='center', fontsize=14, fontweight='bold')
                ax3.set_axis_off()
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
            if ground_truth_index is not None and 0 <= ground_truth_index < len(weights):
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
                sizes = 100 * np.array(weights) / (np.max(weights) + 1e-8)  # Normalize sizes
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
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                interp_flat = interpolated_fields[component].flatten()
                gt_flat = gt_field.flatten()
                
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
            angular_diff = min(angular_diff, 360 - angular_diff)  # Handle cyclic nature
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
# MAIN APPLICATION WITH COMPLETE IMPLEMENTATION
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Transformer Stress Interpolation with Enhanced Spatial Locality",
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
    st.markdown('<h1 class="main-header">🔬 Transformer Stress Field Interpolation with Enhanced Spatial Locality</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Physics-aware stress interpolation with aggressive angular proximity weighting.</strong><br>
    • Load simulation files from numerical_solutions directory<br>
    • Interpolate stress fields at custom polar angles (default: 54.7°)<br>
    • <strong>NEW:</strong> Enhanced spatial locality with aggressive angular distance weighting<br>
    • <strong>NEW:</strong> Angular weights: 0.95 for ≤1°, 0.9 for ≤2.5°, 0.8 for ≤5°, 0.01 for >45°<br>
    • <strong>NEW:</strong> Habit plane (54.7°) awareness with proximity bonuses<br>
    • <strong>NEW:</strong> Cyclic angle handling (0° = 360°, 2° distance with 358°)<br>
    • Comprehensive comparison dashboard with ground truth selection<br>
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
                source_params = [sol['params'] for sol in st.session_state.solutions]
                shape = st.session_state.transformer_interpolator.debug_feature_dimensions(
                    source_params, 54.7
                )
                st.write(f"Feature dimensions: {shape}")
        
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
        
        # Transformer parameters
        st.markdown('<h2 class="section-header">🧠 Enhanced Spatial Locality</h2>', unsafe_allow_html=True)
        
        # Spatial locality parameters
        st.markdown("#### 📍 Angular Distance Weighting")
        st.info("""
        **Enhanced Weighting Scheme:**
        - ≤1°: 0.95 weight
        - ≤2.5°: 0.9 weight  
        - ≤5°: 0.8 weight
        - ≤7.5°: 0.7 weight
        - ≤10°: 0.5 weight
        - ≤15°: 0.3 weight
        - ≤20°: 0.15 weight
        - ≤30°: 0.08 weight
        - ≤45°: 0.04 weight
        - >45°: 0.01 weight
        """)
        
        spatial_sigma = st.slider(
            "Spatial Sigma",
            min_value=0.01,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Controls the decay rate of non-angular parameter weights"
        )
        
        # KEY FEATURE: Adjustable spatial locality weight factor
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
        
        # Visualize angular weighting function
        if st.button("📊 Visualize Angular Weighting", use_container_width=True):
            fig_angular_weights = st.session_state.transformer_interpolator.visualize_angular_weighting(
                target_angle_deg=custom_theta
            )
            st.pyplot(fig_angular_weights)
        
        # Update transformer parameters
        if st.button("🔄 Update Transformer Parameters", use_container_width=True):
            st.session_state.transformer_interpolator.set_spatial_parameters(
                spatial_sigma=spatial_sigma,
                locality_weight_factor=spatial_weight_factor
            )
            st.session_state.transformer_interpolator.temperature = temperature
            st.success("Transformer parameters updated!")
        
        st.divider()
        
        # Run interpolation
        st.markdown("#### 🚀 Interpolation Control")
        if st.button("🚀 Perform Transformer Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing interpolation with enhanced spatial locality..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': eigen_strain,
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
                    }
                    
                    # Perform interpolation
                    result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        custom_theta,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        st.success(f"Interpolation successful! Used {result['num_sources']} source solutions.")
                        st.session_state.selected_ground_truth = None
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
            for sol in st.session_state.solutions:
                if 'params' in sol and 'theta' in sol['params']:
                    theta_deg = np.degrees(sol['params']['theta']) % 360  # Normalize to [0, 360)
                    source_thetas.append(theta_deg)
            
            if source_thetas:
                st.markdown(f"**Source Angles Range:** {min(source_thetas):.1f}° to {max(source_thetas):.1f}°")
                st.markdown(f"**Mean Source Angle:** {np.mean(source_thetas):.1f}°")
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Results Overview",
            "🎨 Visualization", 
            "⚖️ Enhanced Weights Analysis",
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
                    "Number of Sources",
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
            
            # Enhanced spatial locality configuration
            st.markdown("#### 📍 Enhanced Spatial Locality Configuration")
            locality_col1, locality_col2, locality_col3 = st.columns(3)
            with locality_col1:
                st.metric(
                    "Spatial Weight Factor",
                    f"{st.session_state.transformer_interpolator.locality_weight_factor:.2f}",
                    help="0 = pure spatial weights, 1 = pure transformer weights"
                )
            with locality_col2:
                st.metric(
                    "Spatial Sigma",
                    f"{st.session_state.transformer_interpolator.spatial_sigma:.2f}",
                    help="Controls non-angular parameter weight decay rate"
                )
            with locality_col3:
                # Calculate average angular distance of top contributors
                if 'weights' in result and 'source_distances' in result:
                    top_indices = np.argsort(result['weights']['combined'])[-3:]  # Top 3 contributors
                    avg_angular_dist = np.mean([result['source_distances'][i] for i in top_indices])
                    st.metric(
                        "Avg Angular Distance (Top 3)",
                        f"{avg_angular_dist:.1f}°"
                    )
            
            # Quick preview of stress fields
            st.markdown("#### 👀 Quick Preview")
            preview_component = st.selectbox(
                "Preview Component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0,
                key="preview_component"
            )
            if preview_component in result['fields']:
                fig_preview = st.session_state.heatmap_visualizer.create_stress_heatmap(
                    result['fields'][preview_component],
                    title=f"{preview_component.replace('_', ' ').title()} Stress",
                    cmap_name='viridis',
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(10, 8)
                )
                st.pyplot(fig_preview)
        
        with tab2:
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
            
            # Show colormap preview
            if cmap_name:
                col_preview1, col_preview2, col_preview3 = st.columns([1, 2, 1])
                with col_preview2:
                    fig_cmap = st.session_state.heatmap_visualizer.get_colormap_preview(cmap_name)
                    st.pyplot(fig_cmap)
            
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
                    
                    # Show statistics
                    with st.expander("📊 Detailed Statistics", expanded=False):
                        stats = result['statistics'][component]
                        for key, value in stats.items():
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                elif viz_type == "3D Surface":
                    # 3D surface plot
                    fig_3d = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(14, 10)
                    )
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
                elif viz_type == "Interactive 3D":
                    # Interactive 3D surface
                    fig_3d_interactive = st.session_state.heatmap_visualizer.create_interactive_3d_surface(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        width=900,
                        height=700
                    )
                    st.plotly_chart(fig_3d_interactive, use_container_width=True)
                elif viz_type == "Angular Orientation":
                    # Angular orientation plot
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(10, 10),
                        show_habit_plane=True
                    )
                    st.pyplot(fig_angular)
                
                # Comparison of all components
                st.markdown("#### 🔄 Component Comparison")
                if st.button("Show All Components Comparison", key="show_all_components"):
                    fig_all = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                        result['fields'],
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(18, 6)
                    )
                    st.pyplot(fig_all)
        
        with tab3:
            # Enhanced Weights analysis tab
            st.markdown('<h2 class="section-header">⚖️ Enhanced Weight Distribution Analysis</h2>', unsafe_allow_html=True)
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
                    st.metric("Top Contributor", f"θ={max_theta:.1f}°")
                
                # Visualize angular weighting function for current target
                st.markdown("#### 📊 Angular Weighting Function")
                fig_angular_weights = st.session_state.transformer_interpolator.visualize_angular_weighting(
                    target_angle_deg=result['target_angle']
                )
                st.pyplot(fig_angular_weights)
                
                # Weight distribution plot
                st.markdown("#### 📊 Source Weight Distribution")
                fig_weights, ax_weights = plt.subplots(figsize=(14, 6))
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
                
                # Add theta labels and angular distances
                for i, theta in enumerate(result['source_theta_degrees']):
                    if i < len(result['source_distances']):
                        dist = result['source_distances'][i]
                        max_height = max(weights['combined'][i], weights['transformer'][i], weights['positional'][i])
                        ax_weights.text(i, max_height + 0.01,
                                       f'θ={theta:.0f}°\nΔ={dist:.1f}°',
                                       ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig_weights)
                
                # Top contributors table with enhanced information
                st.markdown("#### 🏆 Top Contributors Analysis")
                weight_data = []
                for i in range(len(weights['combined'])):
                    angular_diff = result['source_distances'][i] if i < len(result['source_distances']) else 0.0
                    angular_weight_category = "Unknown"
                    if angular_diff <= 1.0:
                        angular_weight_category = "Very Very Close (≤1°)"
                    elif angular_diff <= 2.5:
                        angular_weight_category = "Very Close (≤2.5°)"
                    elif angular_diff <= 5.0:
                        angular_weight_category = "Close (≤5.0°)"
                    elif angular_diff <= 10.0:
                        angular_weight_category = "Moderate (≤10.0°)"
                    elif angular_diff <= 20.0:
                        angular_weight_category = "Far (≤20.0°)"
                    elif angular_diff <= 45.0:
                        angular_weight_category = "Very Far (≤45.0°)"
                    else:
                        angular_weight_category = "Extremely Far (>45°)"
                    
                    weight_data.append({
                        'Source': i,
                        'Theta (°)': result['source_theta_degrees'][i] if i < len(result['source_theta_degrees']) else 0.0,
                        'Angular Distance (°)': angular_diff,
                        'Angular Weight Category': angular_weight_category,
                        'Combined Weight': weights['combined'][i],
                        'Spatial Weight': weights['positional'][i],
                        'Transformer Weight': weights['transformer'][i],
                        'Defect Type': st.session_state.solutions[i]['params']['defect_type'] if i < len(st.session_state.solutions) else 'Unknown'
                    })
            
                df_weights = pd.DataFrame(weight_data)
                df_weights = df_weights.sort_values('Combined Weight', ascending=False)
                
                # Display top 10 contributors
                st.dataframe(df_weights.head(10).style.format({
                    'Theta (°)': '{:.1f}',
                    'Angular Distance (°)': '{:.1f}',
                    'Combined Weight': '{:.4f}',
                    'Spatial Weight': '{:.4f}',
                    'Transformer Weight': '{:.4f}'
                }).background_gradient(subset=['Combined Weight'], cmap='YlOrRd'))
                
                # Angular distribution plot
                st.markdown("#### 🧭 Angular Distribution Analysis")
                fig_polar = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                    result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(10, 10),
                    show_habit_plane=True
                )
                st.pyplot(fig_polar)
                
                # Correlation between angular distance and weights
                st.markdown("#### 📈 Angular Distance vs. Weight Correlation")
                if 'source_distances' in result:
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                    scatter = ax_corr.scatter(result['source_distances'], weights['combined'],
                                             alpha=0.6, s=100, c=weights['positional'], cmap='viridis')
                    ax_corr.set_xlabel('Angular Distance (°)')
                    ax_corr.set_ylabel('Combined Weight')
                    ax_corr.set_title('Angular Distance vs. Weight Correlation', fontsize=16, fontweight='bold')
                    ax_corr.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax_corr, label='Spatial Weight')
                    
                    # Add trend line
                    if len(result['source_distances']) > 1:
                        z = np.polyfit(result['source_distances'], weights['combined'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(min(result['source_distances']), max(result['source_distances']), 100)
                        ax_corr.plot(x_range, p(x_range), "r--", alpha=0.8, label='Trend')
                        ax_corr.legend()
                    st.pyplot(fig_corr)
        
        with tab4:
            # COMPARISON DASHBOARD
            st.markdown('<h2 class="section-header">🔄 Comparison Dashboard</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Compare interpolated results with ground truth sources</strong><br>
            • Select a source solution as ground truth<br>
            • Visualize differences between interpolation and ground truth<br>
            • Calculate error metrics (MSE, MAE, RMSE, correlation)<br>
            • Analyze spatial correlation patterns
            </div>
            """, unsafe_allow_html=True)
            
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
                index=0,
                key="comp_component"
            )
            comp_cmap = st.selectbox(
                "Colormap for Comparison",
                options=COLORMAP_OPTIONS['Publication Standard'],
                index=0,
                key="comp_cmap"
            )
            
            # Create comparison dashboard
            if comp_component in result['fields']:
                # Prepare source info for dashboard
                source_info = {
                    'theta_degrees': result['source_theta_degrees'],
                    'distances': result['source_distances'],
                    'weights': result['weights']
                }
                
                # Get source fields from result
                source_fields_list = result['source_fields']
                fig_comparison = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                    interpolated_fields=result['fields'],
                    source_fields=source_fields_list,  # Pass full source field dictionaries
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
                if selected_index < len(source_fields_list):
                    ground_truth_field = source_fields_list[selected_index][comp_component]
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
                    
                    # Additional analysis
                    with st.expander("🔍 Detailed Error Analysis", expanded=False):
                        st.markdown("##### Error Distribution")
                        fig_err_hist, ax_err_hist = plt.subplots(figsize=(10, 6))
                        ax_err_hist.hist(error_field.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
                        ax_err_hist.set_xlabel('Error (GPa)')
                        ax_err_hist.set_ylabel('Frequency')
                        ax_err_hist.set_title('Error Distribution', fontsize=16, fontweight='bold')
                        ax_err_hist.grid(True, alpha=0.3)
                        st.pyplot(fig_err_hist)
                        
                        # Spatial error pattern
                        st.markdown("##### Spatial Error Pattern")
                        fig_err_spatial, ax_err_spatial = plt.subplots(figsize=(10, 8))
                        im_err = ax_err_spatial.imshow(error_field, cmap='RdBu_r',
                                                      vmin=-np.max(np.abs(error_field)),
                                                      vmax=np.max(np.abs(error_field)),
                                                      aspect='equal', interpolation='bilinear', origin='lower')
                        plt.colorbar(im_err, ax=ax_err_spatial, label='Error (GPa)')
                        ax_err_spatial.set_title('Spatial Error Distribution', fontsize=16, fontweight='bold')
                        ax_err_spatial.set_xlabel('X Position')
                        ax_err_spatial.set_ylabel('Y Position')
                        st.pyplot(fig_err_spatial)
                else:
                    st.warning("No source information available for comparison.")
        
        with tab5:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Export interpolation results for further analysis</strong><br>
            • Export full results as JSON with metadata<br>
            • Export stress field data as CSV for external analysis<br>
            • Download visualizations as PNG images<br>
            • Save comparison dashboard for publication
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
                        "Angular Orientation",
                        "Angular Weighting Function", 
                        "Weight Distribution",
                        "Comparison Dashboard"
                    ],
                    default=["Von Mises Heatmap", "Angular Weighting Function", "Comparison Dashboard"]
                )
                
                if st.button("🖼️ Generate and Download Visualizations", use_container_width=True):
                    # Create a zip file with all selected plots
                    import zipfile
                    from io import BytesIO
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Generate and save each selected plot
                        plot_count = 0
                        
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
                        
                        if "3D Surface Plot" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                                result['fields']['von_mises'],
                                title="3D Von Mises Stress",
                                cmap_name='viridis',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"3d_surface_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Angular Orientation" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                                result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                show_habit_plane=True
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"angular_orientation_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Angular Weighting Function" in export_plots:
                            fig = st.session_state.transformer_interpolator.visualize_angular_weighting(
                                target_angle_deg=result['target_angle']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"angular_weighting_theta_{result['target_angle']:.1f}.png", buf.getvalue())
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
                        
                        if "Comparison Dashboard" in export_plots and st.session_state.selected_ground_truth is not None:
                            source_info = {
                                'theta_degrees': result['source_theta_degrees'],
                                'distances': result['source_distances'],
                                'weights': result['weights']
                            }
                            # Get source fields from result
                            source_fields_list = result['source_fields']
                            fig = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                                interpolated_fields=result['fields'],
                                source_fields=source_fields_list,
                                source_info=source_info,
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                component='von_mises',
                                cmap_name='viridis',
                                ground_truth_index=st.session_state.selected_ground_truth
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"comparison_dashboard_theta_{result['target_angle']:.1f}.png", buf.getvalue())
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
    else:
        # No results yet - show instructions
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); border-radius: 20px; color: white;">
        <h2>🚀 Ready to Begin!</h2>
        <p style="font-size: 1.2rem; margin-bottom: 30px;">
        Follow these steps to start interpolating stress fields with enhanced spatial locality:
        </p>
        <ol style="text-align: left; display: inline-block; font-size: 1.1rem;">
        <li>Load simulation files from the sidebar</li>
        <li>Configure target parameters (angle, defect type, etc.)</li>
        <li>Adjust enhanced spatial locality parameters</li>
        <li>Visualize the angular weighting function</li>
        <li>Click "Perform Transformer Interpolation"</li>
        <li>Explore results in the tabs above</li>
        </ol>
        <p style="margin-top: 30px; font-size: 1.1rem;">
        <strong>New Enhanced Feature:</strong> Aggressive angular proximity weighting with:
        <ul style="text-align: left; display: inline-block;">
        <li>0.95 weight for angles within 1°</li>
        <li>0.9 weight for angles within 2.5°</li>
        <li>0.8 weight for angles within 5°</li>
        <li>0.01 weight for angles beyond 45°</li>
        <li>Cyclic angle handling (0° = 360°)</li>
        <li>Habit plane (54.7°) awareness</li>
        </ul>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 📚 Quick Start Guide")
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    with col_guide1:
        st.markdown("""
        #### 📂 Data Preparation
        1. Place simulation files in `numerical_solutions/` directory
        2. Supported formats: `.pkl`, `.pickle`, `.pt`, `.pth`
        3. Files should contain 'params' and 'history' keys
        4. Stress fields should be in 'stresses' dictionary
        """)
    with col_guide2:
        st.markdown("""
        #### 🎯 Enhanced Spatial Locality
        - **Angular Weighting:** Aggressive proximity-based weights
        - **Habit Plane Awareness:** Special handling for 54.7°
        - **Cyclic Angles:** 0° = 360°, proper distance calculations
        - **Spatial Weight Factor:** Balance spatial vs transformer weights
        - **Parameter Decay:** Controls non-angular parameter influence
        """)
    with col_guide3:
        st.markdown("""
        #### 🔍 Advanced Features
        - Visualize angular weighting function
        - Detailed weight distribution analysis
        - Select any source as ground truth
        - Comprehensive error metrics
        - Export results for publication
        """)
    
    # Physics explanation
    with st.expander("🧬 Physics Background: Enhanced Angular Weighting", expanded=True):
        st.markdown("""
        **Enhanced Spatial Locality for Material Science:**
        1. **Angular Proximity Priority:** The enhanced weighting aggressively prioritizes angles within 5° of the target, giving weights up to 0.95 for angles within 1°
        2. **Habit Plane (54.7°) Special Handling:**
           - Extra weight bonus when both source and target are near the habit plane
           - Symmetry bonuses for angles symmetric about 54.7°
           - Recognition of equivalent crystallographic orientations
        3. **Cyclic Angle Mathematics:**
           - Proper handling of 0° = 360° equivalence
           - 2° distance between 358° and 0°
           - Minimal angular distance calculation across the circle
        4. **Physics-Based Weight Decay:**
           - Rapid decay beyond 10° angular distance
           - Minimal weight (0.01) for angles beyond 45°
           - Non-angular parameters contribute only 10-20% of total weight
        5. **Material Science Relevance:**
           - Stress fields change rapidly with orientation near habit planes
           - Angular proximity is more important than exact parameter matching
           - Physical symmetry reduces effective angular distances
        
        The enhanced transformer spatial interpolator leverages this physics knowledge through:
        - **Aggressive angular weighting** that prioritizes physically similar orientations
        - **Habit plane awareness** with proximity bonuses
        - **Cyclic distance calculations** that respect crystallographic symmetry
        - **Parameter-appropriate weighting** that gives angular distance primary importance
        """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
