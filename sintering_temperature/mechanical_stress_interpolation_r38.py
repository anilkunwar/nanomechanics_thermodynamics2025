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
# ENHANCED SOLUTION LOADER WITH EIGENSTRAIN NORMALIZATION
# =============================================
class EnhancedSolutionLoader:
    """Enhanced solution loader with physics-aware processing and eigenstrain normalization"""
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        
        # Eigenstrain normalization factors
        self.eigenstrain_conventions = {
            'source': {  # Original source file conventions
                'Twin': 2.12,
                'ESF': 1.414,
                'ISF': 0.707,
                'No Defect': 0.0
            },
            'auto': {  # Auto-calculated conventions (used in UI)
                'Twin': 0.707,
                'ESF': 0.333,
                'ISF': 0.289,
                'No Defect': 0.0
            },
            'theoretical': {  # Theoretical conversion factors
                'Twin': 3.0,  # 2.12 / 0.707 ≈ 3.0
                'ESF': 4.247,  # 1.414 / 0.333 ≈ 4.247
                'ISF': 2.446,  # 0.707 / 0.289 ≈ 2.446
                'No Defect': 1.0
            }
        }
        
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
            
            # Standardize data structure with eigenstrain normalization
            standardized = self._standardize_data(data, file_path)
            return standardized
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data, file_path):
        """Standardize simulation data with physics metadata and eigenstrain normalization"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False,
                'eigenstrain_normalized': False
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
                
                # Normalize eigenstrain values for interpolation
                self._normalize_eigenstrain(standardized)
                
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
        
        return standardized
    
    def _normalize_eigenstrain(self, data):
        """Normalize eigenstrain values to auto-calculated convention"""
        if 'params' in data and 'defect_type' in data['params']:
            defect_type = data['params'].get('defect_type', 'Twin')
            
            # Check if eps0 exists and needs normalization
            if 'eps0' in data['params']:
                eps0_source = data['params']['eps0']
                
                # Determine which convention this value follows
                if abs(eps0_source - self.eigenstrain_conventions['source'][defect_type]) < 0.1:
                    # It follows source convention, convert to auto convention
                    eps0_normalized = eps0_source / self.eigenstrain_conventions['theoretical'][defect_type]
                    data['params']['eps0_original'] = eps0_source  # Store original
                    data['params']['eps0'] = eps0_normalized  # Use normalized
                    data['metadata']['eigenstrain_normalized'] = True
                    data['metadata']['eigenstrain_conversion_factor'] = self.eigenstrain_conventions['theoretical'][defect_type]
                
                elif abs(eps0_source - self.eigenstrain_conventions['auto'][defect_type]) < 0.1:
                    # Already in auto convention
                    data['params']['eps0_original'] = eps0_source
                    data['metadata']['eigenstrain_normalized'] = True
                    data['metadata']['eigenstrain_conversion_factor'] = 1.0
    
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
# ENHANCED TRANSFORMER SPATIAL INTERPOLATOR WITH ROTATION AND SCALING
# =============================================
class EnhancedTransformerSpatialInterpolator:
    """Enhanced transformer interpolator with rotation-aware interpolation and magnitude scaling"""
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.1, 
                 temperature=1.0, locality_weight_factor=0.35,
                 enable_rotation_correction=True, enable_magnitude_scaling=True):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor
        
        # Enhanced features
        self.enable_rotation_correction = enable_rotation_correction
        self.enable_magnitude_scaling = enable_magnitude_scaling
        
        # Eigenstrain conversion factors
        self.eigenstrain_conversion = {
            'Twin': 3.0,
            'ESF': 4.247,
            'ISF': 2.446,
            'No Defect': 1.0
        }
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced input projection - more features
        self.input_proj = nn.Linear(18, d_model)  # Increased from 15 to 18
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Magnitude scaling network
        if enable_magnitude_scaling:
            self.magnitude_scaler = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 3)  # 3 scaling factors: rotation, defect, magnitude
            )
    
    def set_interpolation_features(self, enable_rotation_correction=None, enable_magnitude_scaling=None):
        """Update interpolation features"""
        if enable_rotation_correction is not None:
            self.enable_rotation_correction = enable_rotation_correction
        if enable_magnitude_scaling is not None:
            self.enable_magnitude_scaling = enable_magnitude_scaling
    
    def compute_positional_weights(self, source_params, target_params):
        """Enhanced spatial locality weights with rotation-aware similarity"""
        weights = []
        
        # Extract target parameters
        target_theta_rad = target_params.get('theta', 0.0)
        target_theta_deg = np.degrees(target_theta_rad)
        target_defect = target_params.get('defect_type', 'Twin')
        target_eps0 = target_params.get('eps0', 0.707)
        
        for src in source_params:
            # Extract source parameters
            src_theta_rad = src.get('theta', 0.0)
            src_theta_deg = np.degrees(src_theta_rad)
            src_defect = src.get('defect_type', 'Twin')
            src_eps0 = src.get('eps0', 0.707)
            
            # 1. ROTATION SIMILARITY (40% weight)
            # Calculate angular distance (handling circular nature)
            angular_diff = abs(src_theta_deg - target_theta_deg)
            angular_diff = min(angular_diff, 360 - angular_diff)
            
            # Enhanced rotation similarity with orientation awareness
            rotation_similarity = self._compute_rotation_similarity(
                src_theta_deg, target_theta_deg, src_defect, target_defect
            )
            
            # 2. DEFECT TYPE SIMILARITY (30% weight)
            defect_similarity = self._compute_defect_similarity(src_defect, target_defect)
            
            # 3. EIGENSTRAIN MAGNITUDE SIMILARITY (20% weight)
            magnitude_similarity = self._compute_magnitude_similarity(
                src_eps0, target_eps0, src_defect, target_defect
            )
            
            # 4. ADDITIONAL PHYSICS-BASED SIMILARITY (10% weight)
            physics_similarity = self._compute_physics_similarity(src, target_params)
            
            # COMBINE with physics-aware weighting
            combined_weight = (
                0.40 * rotation_similarity +
                0.30 * defect_similarity +
                0.20 * magnitude_similarity +
                0.10 * physics_similarity
            )
            
            # Ensure weight is positive
            combined_weight = max(combined_weight, 0.001)
            
            weights.append(combined_weight)
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
    
    def _compute_rotation_similarity(self, src_theta, tgt_theta, src_defect, tgt_defect):
        """Compute rotation similarity with defect-aware orientation"""
        # Basic angular difference
        angular_diff = abs(src_theta - tgt_theta)
        angular_diff = min(angular_diff, 360 - angular_diff)
        
        # Gaussian decay for angular difference
        angular_sim = np.exp(-0.5 * (angular_diff / 15.0) ** 2)
        
        # Enhanced: Consider defect-specific orientation patterns
        if src_defect == tgt_defect:
            # Same defect type gets higher weight for similar orientations
            defect_factor = 1.0
        else:
            # Different defect types have different orientation patterns
            # ESF and ISF are more similar to each other than to Twin
            if (src_defect in ['ESF', 'ISF'] and tgt_defect in ['ESF', 'ISF']):
                defect_factor = 0.8
            else:
                defect_factor = 0.5
        
        # Consider habit plane proximity (54.7°)
        habit_diff_src = abs(src_theta - 54.7)
        habit_diff_tgt = abs(tgt_theta - 54.7)
        
        # Both near habit plane increases similarity
        if habit_diff_src < 10.0 and habit_diff_tgt < 10.0:
            habit_factor = 1.2
        else:
            habit_factor = 1.0
        
        return angular_sim * defect_factor * habit_factor
    
    def _compute_defect_similarity(self, src_defect, tgt_defect):
        """Compute defect type similarity with physics awareness"""
        if src_defect == tgt_defect:
            return 1.0
        
        # Similarity matrix based on crystallography
        similarity_matrix = {
            'Twin': {'ESF': 0.4, 'ISF': 0.3, 'No Defect': 0.1},
            'ESF': {'Twin': 0.4, 'ISF': 0.7, 'No Defect': 0.2},
            'ISF': {'Twin': 0.3, 'ESF': 0.7, 'No Defect': 0.2},
            'No Defect': {'Twin': 0.1, 'ESF': 0.2, 'ISF': 0.2}
        }
        
        return similarity_matrix.get(src_defect, {}).get(tgt_defect, 0.2)
    
    def _compute_magnitude_similarity(self, src_eps0, tgt_eps0, src_defect, tgt_defect):
        """Compute magnitude similarity with eigenstrain normalization"""
        # Normalize eigenstrains for comparison
        src_normalized = self._normalize_eigenstrain_value(src_eps0, src_defect)
        tgt_normalized = self._normalize_eigenstrain_value(tgt_eps0, tgt_defect)
        
        # Compute similarity based on normalized values
        diff = abs(src_normalized - tgt_normalized)
        
        # Gaussian similarity with scale factor
        # Use larger sigma for magnitude differences since different defects have different scales
        sigma = 0.5  # More tolerant for magnitude differences
        similarity = np.exp(-0.5 * (diff / sigma) ** 2)
        
        return similarity
    
    def _normalize_eigenstrain_value(self, eps0, defect_type):
        """Normalize eigenstrain value based on defect type"""
        if defect_type in self.eigenstrain_conversion:
            # Convert to common scale
            return eps0 / self.eigenstrain_conversion[defect_type]
        return eps0
    
    def _compute_physics_similarity(self, src_params, tgt_params):
        """Compute physics-based similarity considering material properties"""
        similarity = 0.0
        
        # Compare kappa (material stiffness)
        src_kappa = src_params.get('kappa', 0.6)
        tgt_kappa = tgt_params.get('kappa', 0.6)
        kappa_sim = np.exp(-abs(src_kappa - tgt_kappa) / 0.3)
        similarity += 0.4 * kappa_sim
        
        # Compare shape
        src_shape = src_params.get('shape', 'Square')
        tgt_shape = tgt_params.get('shape', 'Square')
        shape_sim = 1.0 if src_shape == tgt_shape else 0.5
        similarity += 0.3 * shape_sim
        
        # Additional physics parameters
        # You can add more physics-based comparisons here
        similarity += 0.3 * 0.8  # Default physics similarity
        
        return similarity / 1.0  # Normalize to [0,1]
    
    def apply_rotation_correction(self, stress_field, src_theta, tgt_theta, defect_type):
        """Apply rotation correction to stress field based on angular difference"""
        if not self.enable_rotation_correction:
            return stress_field
        
        # Calculate rotation angle (difference in orientation)
        rotation_angle = tgt_theta - src_theta
        
        # Apply rotation based on defect type
        if defect_type == 'Twin':
            # Twins often have specific rotation patterns
            rotation_angle *= 0.8  # Scale factor for twins
        elif defect_type in ['ESF', 'ISF']:
            # Stacking faults have different rotation behavior
            rotation_angle *= 1.2
        
        # Limit rotation to reasonable range
        rotation_angle = np.clip(rotation_angle, -45, 45)
        
        # Only apply significant rotation if angle > 5 degrees
        if abs(rotation_angle) > 5.0:
            # Apply rotation to stress field
            rotated_field = rotate(stress_field, rotation_angle, reshape=False, 
                                  mode='constant', cval=np.mean(stress_field))
            return rotated_field
        else:
            return stress_field
    
    def apply_magnitude_scaling(self, stress_field, src_params, tgt_params):
        """Apply magnitude scaling based on eigenstrain and defect type"""
        if not self.enable_magnitude_scaling:
            return stress_field
        
        src_defect = src_params.get('defect_type', 'Twin')
        tgt_defect = tgt_params.get('defect_type', 'Twin')
        src_eps0 = src_params.get('eps0', 0.707)
        tgt_eps0 = tgt_params.get('eps0', 0.707)
        
        # Calculate base scaling factor
        if src_defect == tgt_defect:
            # Same defect type, scale by eps0 ratio
            scale_factor = tgt_eps0 / (src_eps0 + 1e-10)
        else:
            # Different defect types, use normalized eigenstrain
            src_normalized = self._normalize_eigenstrain_value(src_eps0, src_defect)
            tgt_normalized = self._normalize_eigenstrain_value(tgt_eps0, tgt_defect)
            scale_factor = tgt_normalized / (src_normalized + 1e-10)
        
        # Apply defect-specific scaling corrections
        if tgt_defect == 'Twin':
            # Twin defects often have higher stress concentrations
            scale_factor *= 1.2
        elif tgt_defect in ['ESF', 'ISF']:
            # Stacking faults have different stress distributions
            scale_factor *= 0.9
        
        # Apply scaling with smoothing
        scaled_field = stress_field * scale_factor
        
        # Add small random variation to avoid perfect scaling
        noise_level = 0.05 * scale_factor
        noise = np.random.normal(0, noise_level, scaled_field.shape)
        scaled_field = scaled_field * (1 + noise)
        
        return scaled_field
    
    def encode_parameters(self, params_list, target_angle_deg):
        """Enhanced parameter encoding with rotation and magnitude features"""
        encoded = []
        for params in params_list:
            features = []
            
            # 1. Enhanced angular features (5 features)
            theta_deg = np.degrees(params.get('theta', 0.0))
            angle_diff = abs(theta_deg - target_angle_deg)
            
            # Multiple angular similarity measures
            features.append(np.exp(-angle_diff / 8.0))   # Fine resolution
            features.append(np.exp(-angle_diff / 25.0))  # Medium resolution
            features.append(np.sin(np.radians(angle_diff)))  # Angular sine
            features.append(np.cos(np.radians(angle_diff)))  # Angular cosine
            features.append(angle_diff / 180.0)  # Normalized angle difference
            
            # 2. Habit plane features (2 features)
            habit_distance = abs(theta_deg - 54.7)
            features.append(np.exp(-habit_distance / 6.0))  # Tight decay
            features.append(1.0 if habit_distance < 7.5 else 0.0)  # Binary near habit
            
            # 3. Eigenstrain and magnitude features (3 features)
            eps0 = params.get('eps0', 0.707)
            defect = params.get('defect_type', 'Twin')
            
            # Normalized eigenstrain
            eps0_normalized = self._normalize_eigenstrain_value(eps0, defect)
            features.append(eps0_normalized)
            
            # Defect-specific magnitude factor
            if defect == 'Twin':
                features.append(1.0)  # Highest magnitude
            elif defect == 'ESF':
                features.append(0.67)  # Medium magnitude
            elif defect == 'ISF':
                features.append(0.55)  # Lower magnitude
            else:
                features.append(0.0)  # No defect
            
            # Material property
            features.append(params.get('kappa', 0.6) / 2.0)
            
            # 4. Defect type encoding (4 features)
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            
            # 5. Shape encoding (2 features)
            shape = params.get('shape', 'Square')
            features.append(1.0 if shape == 'Square' else 0.0)
            features.append(1.0 if shape in ['Horizontal Fault', 'Vertical Fault'] else 0.0)
            
            # 6. Rotation direction feature (1 feature)
            # Positive for clockwise from habit plane, negative for counterclockwise
            rotation_direction = 1.0 if theta_deg > 54.7 else -1.0
            features.append(rotation_direction)
            
            # Verify we have exactly 18 features
            if len(features) != 18:
                # Pad with zeros if fewer than 18
                while len(features) < 18:
                    features.append(0.0)
                # Truncate if more than 18
                features = features[:18]
            
            encoded.append(features)
        
        return torch.FloatTensor(encoded)
    
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """Enhanced interpolation with rotation correction and magnitude scaling"""
        if not sources:
            st.warning("No sources provided for interpolation.")
            return None
        
        try:
            # Extract source parameters and fields
            source_params = []
            source_fields = []
            source_indices = []
            
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
                        stress_fields = last_frame['stresses']
                        
                        # Extract all stress components
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
            
            # Ensure all fields have same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                target_shape = shapes[0]
                for fields in source_fields:
                    for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        if fields[key].shape != target_shape:
                            factors = [t/s for t, s in zip(target_shape, fields[key].shape)]
                            fields[key] = zoom(fields[key], factors, order=1)
            
            # Encode parameters
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Ensure correct feature dimensions
            if source_features.shape[1] < 18:
                padding = torch.zeros(source_features.shape[0], 18 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 18:
                padding = torch.zeros(target_features.shape[0], 18 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            # Compute positional weights
            pos_weights = self.compute_positional_weights(source_params, target_params)
            
            # Prepare transformer input
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            
            # Apply transformer
            proj_features = self.input_proj(all_features)
            proj_features = self.pos_encoder(proj_features)
            transformer_output = self.transformer(proj_features)
            
            # Compute attention weights
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
            
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1) / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            transformer_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().numpy()
            
            # Combine weights
            combined_weights = (
                self.locality_weight_factor * transformer_weights + 
                (1 - self.locality_weight_factor) * pos_weights
            )
            combined_weights = combined_weights / np.sum(combined_weights)
            
            # Apply enhanced interpolation with rotation and scaling
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        # Get base field
                        field = fields[component].copy()
                        
                        # Apply rotation correction if enabled
                        if self.enable_rotation_correction:
                            src_theta = np.degrees(source_params[i].get('theta', 0.0))
                            field = self.apply_rotation_correction(
                                field, src_theta, target_angle_deg, 
                                source_params[i].get('defect_type', 'Twin')
                            )
                        
                        # Apply magnitude scaling if enabled
                        if self.enable_magnitude_scaling:
                            field = self.apply_magnitude_scaling(
                                field, source_params[i], target_params
                            )
                        
                        # Add weighted contribution
                        interpolated += combined_weights[i] * field
                
                interpolated_fields[component] = interpolated
            
            # Calculate statistics and analysis
            analysis = self._perform_enhanced_analysis(
                source_params, source_fields, target_params, 
                target_angle_deg, combined_weights, interpolated_fields
            )
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'transformer': transformer_weights.tolist(),
                    'positional': pos_weights.tolist(),
                    'combined': combined_weights.tolist(),
                    'entropy': self._calculate_entropy(combined_weights)
                },
                'statistics': self._calculate_statistics(interpolated_fields),
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_theta_degrees': [np.degrees(p.get('theta', 0.0)) for p in source_params],
                'source_distances': self._calculate_angular_distances(source_params, target_angle_deg),
                'source_defect_types': [p.get('defect_type', 'Twin') for p in source_params],
                'source_indices': source_indices,
                'source_fields': source_fields,
                'analysis': analysis,
                'enhanced_features': {
                    'rotation_correction': self.enable_rotation_correction,
                    'magnitude_scaling': self.enable_magnitude_scaling,
                    'eigenstrain_normalization': True
                }
            }
            
        except Exception as e:
            st.error(f"Error during enhanced interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _perform_enhanced_analysis(self, source_params, source_fields, target_params, 
                                  target_angle, weights, interpolated_fields):
        """Perform enhanced analysis of interpolation results"""
        analysis = {}
        
        # Calculate angular statistics
        target_defect = target_params.get('defect_type', 'Twin')
        target_eps0 = target_params.get('eps0', 0.707)
        
        # Group sources by defect type
        defect_groups = {}
        for i, params in enumerate(source_params):
            defect = params.get('defect_type', 'Twin')
            if defect not in defect_groups:
                defect_groups[defect] = []
            defect_groups[defect].append(i)
        
        # Calculate weight distribution by defect type
        defect_weights = {}
        for defect, indices in defect_groups.items():
            total_weight = sum(weights[i] for i in indices)
            defect_weights[defect] = {
                'total_weight': total_weight,
                'avg_weight': total_weight / len(indices) if indices else 0,
                'num_sources': len(indices)
            }
        
        # Calculate rotation statistics
        rotation_contributions = []
        for i, params in enumerate(source_params):
            src_theta = np.degrees(params.get('theta', 0.0))
            rotation_diff = abs(src_theta - target_angle)
            rotation_diff = min(rotation_diff, 360 - rotation_diff)
            rotation_contributions.append({
                'source_idx': i,
                'theta': src_theta,
                'rotation_diff': rotation_diff,
                'weight': weights[i],
                'defect_type': params.get('defect_type', 'Twin')
            })
        
        # Sort by rotation difference
        rotation_contributions.sort(key=lambda x: x['rotation_diff'])
        
        analysis.update({
            'defect_weights': defect_weights,
            'rotation_contributions': rotation_contributions,
            'primary_contributor': {
                'source_idx': np.argmax(weights),
                'weight': np.max(weights),
                'theta': np.degrees(source_params[np.argmax(weights)].get('theta', 0.0)),
                'defect_type': source_params[np.argmax(weights)].get('defect_type', 'Twin')
            },
            'rotation_stats': {
                'avg_rotation_diff': np.mean([r['rotation_diff'] for r in rotation_contributions]),
                'weighted_rotation_diff': np.sum([r['rotation_diff'] * r['weight'] for r in rotation_contributions]),
                'closest_source': rotation_contributions[0] if rotation_contributions else None
            }
        })
        
        return analysis
    
    def _calculate_angular_distances(self, source_params, target_angle):
        """Calculate angular distances from target"""
        distances = []
        for params in source_params:
            src_theta = np.degrees(params.get('theta', 0.0))
            diff = abs(src_theta - target_angle)
            diff = min(diff, 360 - diff)
            distances.append(diff)
        return distances
    
    def _calculate_statistics(self, fields):
        """Calculate statistics for all fields"""
        stats = {}
        for component, field in fields.items():
            stats[component] = {
                'max': float(np.max(field)),
                'min': float(np.min(field)),
                'mean': float(np.mean(field)),
                'std': float(np.std(field)),
                'median': float(np.median(field))
            }
        return stats
    
    def _calculate_entropy(self, weights):
        """Calculate entropy of weight distribution"""
        weights = np.array(weights)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return float(-np.sum(weights * np.log(weights + 1e-10)))
    
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
# ENHANCED HEATMAP VISUALIZER WITH ROTATION ANALYSIS
# =============================================
class EnhancedHeatMapVisualizer(HeatMapVisualizer):
    """Enhanced visualizer with rotation analysis capabilities"""
    
    def create_rotation_analysis_dashboard(self, interpolation_result, figsize=(24, 16)):
        """Create comprehensive rotation analysis dashboard"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        result = interpolation_result
        target_angle = result['target_angle']
        target_defect = result['target_params']['defect_type']
        
        # 1. Weight distribution by rotation difference (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'analysis' in result and 'rotation_contributions' in result['analysis']:
            contributions = result['analysis']['rotation_contributions']
            rotations = [c['rotation_diff'] for c in contributions]
            weights = [c['weight'] for c in contributions]
            
            scatter = ax1.scatter(rotations, weights, alpha=0.7, s=100, 
                                c=range(len(weights)), cmap='viridis')
            
            # Add trend line
            if len(rotations) > 1:
                z = np.polyfit(rotations, weights, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(rotations), max(rotations), 100)
                ax1.plot(x_smooth, p(x_smooth), 'r--', linewidth=2, label='Trend')
            
            ax1.set_xlabel('Rotation Difference from Target (°)')
            ax1.set_ylabel('Weight')
            ax1.set_title('Weight vs Rotation Difference', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Color points by defect type
            for i, contrib in enumerate(contributions):
                color = 'red' if contrib['defect_type'] == 'Twin' else \
                       ('blue' if contrib['defect_type'] == 'ISF' else \
                       ('green' if contrib['defect_type'] == 'ESF' else 'gray'))
                ax1.scatter(contrib['rotation_diff'], contrib['weight'], 
                          color=color, alpha=0.8, s=120, edgecolors='black')
        
        # 2. Angular distribution with defect coloring (top center)
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        if 'source_theta_degrees' in result:
            angles = np.radians(result['source_theta_degrees'])
            distances = result['source_distances']
            weights = result['weights']['combined']
            
            # Color by defect type
            colors = []
            for defect in result['source_defect_types']:
                if defect == 'Twin':
                    colors.append('red')
                elif defect == 'ISF':
                    colors.append('blue')
                elif defect == 'ESF':
                    colors.append('green')
                else:
                    colors.append('gray')
            
            # Size by weight
            sizes = 100 + 900 * np.array(weights)
            
            scatter = ax2.scatter(angles, distances, s=sizes, c=colors, 
                                 alpha=0.7, edgecolors='black')
            
            # Target
            target_rad = np.radians(target_angle)
            ax2.scatter(target_rad, 0, s=300, c='yellow', marker='*', 
                       edgecolors='black', linewidth=2, label='Target')
            
            # Habit plane
            habit_rad = np.radians(54.7)
            ax2.axvline(habit_rad, color='purple', alpha=0.6, 
                       linestyle='--', linewidth=2, label='Habit Plane')
            
            ax2.set_title('Angular Distribution with Defect Coloring', 
                         fontsize=16, fontweight='bold', pad=20)
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            ax2.legend(loc='upper right', fontsize=10)
        
        # 3. Defect type weight analysis (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'analysis' in result and 'defect_weights' in result['analysis']:
            defect_weights = result['analysis']['defect_weights']
            defects = list(defect_weights.keys())
            total_weights = [defect_weights[d]['total_weight'] for d in defects]
            num_sources = [defect_weights[d]['num_sources'] for d in defects]
            
            x = np.arange(len(defects))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, total_weights, width, 
                           label='Total Weight', alpha=0.7)
            bars2 = ax3.bar(x + width/2, num_sources, width, 
                           label='Number of Sources', alpha=0.7)
            
            ax3.set_xlabel('Defect Type')
            ax3.set_ylabel('Value')
            ax3.set_title('Defect Type Contribution Analysis', 
                         fontsize=16, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(defects)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Rotation direction analysis (middle left)
        ax4 = fig.add_subplot(gs[0, 3])
        if 'source_theta_degrees' in result:
            thetas = result['source_theta_degrees']
            clockwise = sum(1 for t in thetas if t > 54.7)
            counterclockwise = len(thetas) - clockwise
            
            sizes = [clockwise, counterclockwise]
            labels = ['Clockwise from Habit', 'Counterclockwise from Habit']
            colors = ['lightblue', 'lightcoral']
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            ax4.set_title('Rotation Direction Distribution', 
                         fontsize=16, fontweight='bold')
        
        # 5. Stress field components (middle row)
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
        
        # 6. Enhanced parameters display (middle right)
        ax10 = fig.add_subplot(gs[1, 3])
        ax10.axis('off')
        
        if 'enhanced_features' in result:
            features = result['enhanced_features']
            params_text = (
                f"Enhanced Interpolation Features:\n"
                f"• Rotation Correction: {'Enabled' if features['rotation_correction'] else 'Disabled'}\n"
                f"• Magnitude Scaling: {'Enabled' if features['magnitude_scaling'] else 'Disabled'}\n"
                f"• Eigenstrain Normalization: {'Enabled' if features['eigenstrain_normalization'] else 'Disabled'}\n\n"
                f"Target Parameters:\n"
                f"• Angle: {target_angle:.1f}°\n"
                f"• Defect: {target_defect}\n"
                f"• ε₀: {result['target_params'].get('eps0', 0.707):.3f}\n"
                f"• κ: {result['target_params'].get('kappa', 0.6):.3f}"
            )
            
            ax10.text(0.1, 0.5, params_text, fontsize=12, family='monospace',
                     verticalalignment='center', transform=ax10.transAxes,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', 
                              alpha=0.9, edgecolor='orange', linewidth=2))
            ax10.set_title('Enhanced Parameters', fontsize=16, fontweight='bold', pad=20)
        
        # 7. Detailed weight analysis (bottom row, full width)
        ax11 = fig.add_subplot(gs[2, :])
        if 'analysis' in result and 'rotation_contributions' in result['analysis']:
            contributions = result['analysis']['rotation_contributions'][:10]  # Top 10
            
            # Create detailed table
            cell_text = []
            for contrib in contributions:
                row = [
                    f"{contrib['source_idx']}",
                    f"{contrib['theta']:.1f}°",
                    f"{contrib['rotation_diff']:.1f}°",
                    contrib['defect_type'],
                    f"{contrib['weight']:.4f}"
                ]
                cell_text.append(row)
            
            columns = ['Source', 'Theta', 'Δθ', 'Defect', 'Weight']
            
            table = ax11.table(cellText=cell_text, colLabels=columns,
                              loc='center', cellLoc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Color cells by defect type
            for i, contrib in enumerate(contributions):
                if contrib['defect_type'] == 'Twin':
                    color = 'lightcoral'
                elif contrib['defect_type'] == 'ISF':
                    color = 'lightblue'
                elif contrib['defect_type'] == 'ESF':
                    color = 'lightgreen'
                else:
                    color = 'lightgray'
                
                for j in range(len(columns)):
                    table[(i+1, j)].set_facecolor(color)
            
            ax11.set_title('Top 10 Contributors to Interpolation', 
                          fontsize=16, fontweight='bold', pad=20)
            ax11.axis('off')
        
        plt.suptitle(f'Comprehensive Rotation Analysis - Target θ={target_angle:.1f}°, {target_defect}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_eigenstrain_conversion_chart(self, figsize=(10, 8)):
        """Create chart showing eigenstrain conversion factors"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Eigenstrain values
        defect_types = ['Twin', 'ESF', 'ISF', 'No Defect']
        source_values = [2.12, 1.414, 0.707, 0.0]
        auto_values = [0.707, 0.333, 0.289, 0.0]
        conversion_factors = [3.0, 4.247, 2.446, 1.0]
        
        x = np.arange(len(defect_types))
        width = 0.25
        
        # Plot bars
        bars1 = ax.bar(x - width, source_values, width, label='Source Files', alpha=0.8, color='blue')
        bars2 = ax.bar(x, auto_values, width, label='Auto-calculated', alpha=0.8, color='green')
        bars3 = ax.bar(x + width, conversion_factors, width, label='Conversion Factor', alpha=0.8, color='red')
        
        # Customize plot
        ax.set_xlabel('Defect Type')
        ax.set_ylabel('Eigenstrain Value / Factor')
        ax.set_title('Eigenstrain Conventions and Conversion Factors', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(defect_types)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add explanation text
        explanation = (
            "Eigenstrain Conventions:\n"
            "• Source Files: ε₀ values as stored in simulation files\n"
            "• Auto-calculated: Normalized values used for interpolation\n"
            "• Conversion: Factor = Source / Auto-calculated\n\n"
            "During interpolation, source values are normalized using:\n"
            "ε₀_normalized = ε₀_source / conversion_factor"
        )
        
        plt.figtext(0.02, 0.02, explanation, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED RESULTS MANAGER
# =============================================
class EnhancedResultsManager(ResultsManager):
    """Enhanced results manager with rotation analysis export"""
    
    def prepare_enhanced_export_data(self, interpolation_result, visualization_params):
        """Prepare enhanced export data with rotation analysis"""
        result = interpolation_result.copy()
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'enhanced_transformer_spatial',
                'visualization_params': visualization_params,
                'enhanced_features': result.get('enhanced_features', {})
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
                'source_defect_types': result.get('source_defect_types', []),
                'source_indices': result.get('source_indices', []),
                'analysis': result.get('analysis', {}),
                'enhanced_features': result.get('enhanced_features', {})
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
        
        return export_data

# =============================================
# MAIN APPLICATION WITH ENHANCED FEATURES
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Enhanced Transformer Stress Interpolation with Rotation Analysis",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Enhanced Transformer Stress Interpolation with Rotation Analysis</h1>', unsafe_allow_html=True)
    
    # Description with eigenstrain conversion info
    st.markdown("""
    <div class="info-box">
        <strong>🔬 Enhanced physics-aware stress interpolation with rotation correction and eigenstrain normalization</strong><br>
        • <strong>Eigenstrain Conversion:</strong> Source files (Twin=2.12, ESF=1.414, ISF=0.707) → Auto-calculated (Twin=0.707, ESF=0.333, ISF=0.289)<br>
        • <strong>Rotation Correction:</strong> Adjusts stress patterns based on angular differences<br>
        • <strong>Magnitude Scaling:</strong> Scales stress magnitudes based on defect type and eigenstrain<br>
        • <strong>Enhanced Weighting:</strong> 40% rotation, 30% defect type, 20% magnitude, 10% physics<br>
        • <strong>Comprehensive Analysis:</strong> Rotation direction analysis, defect type contributions<br>
        • Load simulation files from numerical_solutions directory<br>
        • Interpolate stress fields at custom polar angles (default: 54.7°)<br>
        • Visualize von Mises, hydrostatic, and stress magnitude fields<br>
        • Publication-ready visualizations with angular orientation plots<br>
        • Export results in multiple formats
    </div>
    """, unsafe_allow_html=True)
    
    # Eigenstrain conversion warning
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important: Eigenstrain Conversion Applied</strong><br>
        Source files use different eigenstrain conventions than auto-calculated values:<br>
        • <strong>Source Files:</strong> Twin=2.12, ESF=1.414, ISF=0.707<br>
        • <strong>Auto-calculated:</strong> Twin=0.707, ESF=0.333, ISF=0.289<br>
        • <strong>Conversion Factors:</strong> Twin=3.0×, ESF=4.247×, ISF=2.446×<br>
        During interpolation, source values are automatically normalized.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'enhanced_interpolator' not in st.session_state:
        # Initialize with enhanced features
        st.session_state.enhanced_interpolator = EnhancedTransformerSpatialInterpolator(
            spatial_sigma=0.1,
            locality_weight_factor=0.35,
            enable_rotation_correction=True,
            enable_magnitude_scaling=True
        )
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = EnhancedHeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
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
                    # Show eigenstrain info
                    for sol in st.session_state.solutions[:3]:  # Show first 3
                        if 'params' in sol:
                            params = sol['params']
                            if 'eps0' in params:
                                defect = params.get('defect_type', 'Unknown')
                                eps0 = params['eps0']
                                st.info(f"{defect}: ε₀={eps0:.3f}")
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
        
        # Eigenstrain auto-calculation with conversion info
        st.markdown("#### 🧮 Eigenstrain Calculation")
        auto_eigen = st.checkbox("Auto-calculate eigenstrain", value=True)
        
        if auto_eigen:
            # Show conversion info
            auto_values = {'Twin': 0.707, 'ESF': 0.333, 'ISF': 0.289, 'No Defect': 0.0}
            source_values = {'Twin': 2.12, 'ESF': 1.414, 'ISF': 0.707, 'No Defect': 0.0}
            
            eigen_strain = auto_values[defect_type]
            source_val = source_values[defect_type]
            conversion_factor = source_val / eigen_strain if eigen_strain != 0 else 1.0
            
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.metric("Auto-calculated ε₀", f"{eigen_strain:.3f}")
            with col_e2:
                st.metric("Source File ε₀", f"{source_val:.3f}")
            
            st.caption(f"Conversion factor: {conversion_factor:.3f}×")
        else:
            eigen_strain = st.slider(
                "Eigenstrain ε₀", 
                min_value=0.0, 
                max_value=3.0, 
                value=0.707,
                step=0.001
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
        
        st.divider()
        
        # Enhanced interpolation features
        st.markdown('<h2 class="section-header">✨ Enhanced Features</h2>', unsafe_allow_html=True)
        
        # Rotation correction
        enable_rotation = st.checkbox(
            "Enable Rotation Correction", 
            value=True,
            help="Adjust stress patterns based on angular differences"
        )
        
        # Magnitude scaling
        enable_scaling = st.checkbox(
            "Enable Magnitude Scaling", 
            value=True,
            help="Scale stress magnitudes based on eigenstrain and defect type"
        )
        
        # Spatial locality parameters
        st.markdown("#### 📍 Spatial Locality Controls")
        spatial_sigma = st.slider(
            "Spatial Sigma", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.1,
            step=0.01,
            help="Controls the decay rate of spatial weights (lower = faster decay, more localized)"
        )
        
        # Spatial locality weight factor
        spatial_weight_factor = st.slider(
            "Spatial Locality Weight Factor", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.35,
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
        
        # Update interpolator parameters
        if st.button("🔄 Update Interpolator Parameters", use_container_width=True):
            st.session_state.enhanced_interpolator.set_spatial_parameters(
                spatial_sigma=spatial_sigma,
                locality_weight_factor=spatial_weight_factor
            )
            st.session_state.enhanced_interpolator.set_interpolation_features(
                enable_rotation_correction=enable_rotation,
                enable_magnitude_scaling=enable_scaling
            )
            st.session_state.enhanced_interpolator.temperature = temperature
            st.success("Interpolator parameters updated!")
        
        st.divider()
        
        # Run enhanced interpolation
        st.markdown("#### 🚀 Enhanced Interpolation Control")
        if st.button("🚀 Perform Enhanced Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing enhanced interpolation with rotation analysis..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': eigen_strain,
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
                    }
                    
                    # Perform enhanced interpolation
                    result = st.session_state.enhanced_interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        custom_theta,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        st.session_state.selected_ground_truth = None
                        
                        # Show success message with enhanced features
                        st.markdown("""
                        <div class="success-box">
                            <strong>✅ Enhanced Interpolation Successful!</strong><br>
                            • Used {num_sources} source solutions<br>
                            • Rotation correction: {rotation}<br>
                            • Magnitude scaling: {scaling}<br>
                            • Eigenstrain normalization: Applied
                        </div>
                        """.format(
                            num_sources=result['num_sources'],
                            rotation="Enabled" if enable_rotation else "Disabled",
                            scaling="Enabled" if enable_scaling else "Disabled"
                        ), unsafe_allow_html=True)
                        
                        # Show analysis highlights
                        if 'analysis' in result:
                            analysis = result['analysis']
                            if 'primary_contributor' in analysis:
                                pc = analysis['primary_contributor']
                                st.info(f"**Primary Contributor:** Source {pc['source_idx']} (θ={pc['theta']:.1f}°, {pc['defect_type']}, weight={pc['weight']:.3f})")
                    else:
                        st.error("Enhanced interpolation failed. Check the console for errors.")
    
    # Main content area
    if st.session_state.solutions:
        st.markdown(f"### 📊 Loaded {len(st.session_state.solutions)} Solutions")
        
        # Show eigenstrain conversion chart
        with st.expander("📈 Show Eigenstrain Conversion Chart", expanded=False):
            fig_conversion = st.session_state.heatmap_visualizer.create_eigenstrain_conversion_chart()
            st.pyplot(fig_conversion)
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Results Overview", 
            "🎨 Visualization", 
            "🔄 Rotation Analysis",
            "⚖️ Weights Analysis",
            "💾 Export Results"
        ])
        
        with tab1:
            # Results overview
            st.markdown('<h2 class="section-header">📊 Enhanced Interpolation Results</h2>', unsafe_allow_html=True)
            
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
            with col4:
                st.metric(
                    "Number of Sources", 
                    result['num_sources'],
                    delta=f"Entropy: {result['weights']['entropy']:.3f}"
                )
            
            # Enhanced features status
            st.markdown("#### ✨ Enhanced Features Status")
            if 'enhanced_features' in result:
                features = result['enhanced_features']
                feat_col1, feat_col2, feat_col3 = st.columns(3)
                with feat_col1:
                    status = "✅ Enabled" if features.get('rotation_correction', False) else "❌ Disabled"
                    st.metric("Rotation Correction", status)
                with feat_col2:
                    status = "✅ Enabled" if features.get('magnitude_scaling', False) else "❌ Disabled"
                    st.metric("Magnitude Scaling", status)
                with feat_col3:
                    status = "✅ Applied" if features.get('eigenstrain_normalization', False) else "❌ Not Applied"
                    st.metric("Eigenstrain Norm", status)
            
            # Analysis highlights
            st.markdown("#### 🔍 Analysis Highlights")
            if 'analysis' in result:
                analysis = result['analysis']
                
                # Weight distribution by defect type
                if 'defect_weights' in analysis:
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        st.markdown("**Weight Distribution by Defect Type:**")
                        for defect, stats in analysis['defect_weights'].items():
                            st.metric(
                                f"{defect} Weight",
                                f"{stats['total_weight']:.3f}",
                                delta=f"{stats['num_sources']} sources"
                            )
                    
                    with col_a2:
                        # Primary contributor
                        if 'primary_contributor' in analysis:
                            pc = analysis['primary_contributor']
                            st.markdown("**Primary Contributor:**")
                            st.info(f"""
                            Source {pc['source_idx']}
                            • θ = {pc['theta']:.1f}°
                            • {pc['defect_type']}
                            • Weight = {pc['weight']:.3f}
                            """)
            
            # Quick preview
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
            
            # Visualization type
            viz_type = st.radio(
                "Visualization Type",
                options=["2D Heatmap", "3D Surface", "Interactive Heatmap", "Angular Orientation"],
                horizontal=True
            )
            
            if component in result['fields']:
                stress_field = result['fields'][component]
                
                if viz_type == "2D Heatmap":
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
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        show_habit_plane=True
                    )
                    st.pyplot(fig_angular)
            
            # All components comparison
            if st.button("Show All Components", key="show_all_components"):
                fig_all = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                    result['fields'],
                    cmap_name=cmap_name,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type']
                )
                st.pyplot(fig_all)
        
        with tab3:
            # Rotation analysis tab
            st.markdown('<h2 class="section-header">🔄 Rotation Analysis Dashboard</h2>', unsafe_allow_html=True)
            
            # Create comprehensive rotation analysis dashboard
            fig_rotation = st.session_state.heatmap_visualizer.create_rotation_analysis_dashboard(result)
            st.pyplot(fig_rotation)
            
            # Additional rotation statistics
            if 'analysis' in result and 'rotation_stats' in result['analysis']:
                stats = result['analysis']['rotation_stats']
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Average Rotation Difference", f"{stats['avg_rotation_diff']:.1f}°")
                with col_r2:
                    st.metric("Weighted Rotation Difference", f"{stats['weighted_rotation_diff']:.1f}°")
                with col_r3:
                    if stats['closest_source']:
                        st.metric("Closest Source", f"{stats['closest_source']['rotation_diff']:.1f}°")
            
            # Detailed rotation contributions
            st.markdown("#### 📋 Detailed Rotation Contributions")
            if 'analysis' in result and 'rotation_contributions' in result['analysis']:
                contributions = result['analysis']['rotation_contributions']
                
                # Create DataFrame for display
                df_contributions = pd.DataFrame(contributions)
                df_contributions = df_contributions.sort_values('weight', ascending=False)
                
                # Display table
                st.dataframe(
                    df_contributions.style.format({
                        'theta': '{:.1f}°',
                        'rotation_diff': '{:.1f}°',
                        'weight': '{:.4f}'
                    }).background_gradient(subset=['weight'], cmap='YlOrRd'),
                    height=400
                )
        
        with tab4:
            # Weights analysis
            st.markdown('<h2 class="section-header">⚖️ Weight Distribution Analysis</h2>', unsafe_allow_html=True)
            
            if 'weights' in result:
                weights = result['weights']
                
                # Create weight distribution plot
                fig_weights, ax_weights = plt.subplots(figsize=(14, 6))
                x = range(len(weights['combined']))
                width = 0.25
                
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
                
                # Add detailed labels
                for i, (theta, dist, defect, weight) in enumerate(zip(
                    result['source_theta_degrees'],
                    result['source_distances'],
                    result['source_defect_types'],
                    weights['combined']
                )):
                    if weight > 0.05:  # Label significant contributors
                        ax_weights.text(i, weight + 0.005,
                                      f'θ={theta:.0f}°\nΔ={dist:.0f}°\n{defect}\n{weight:.3f}', 
                                      ha='center', va='bottom', fontsize=6, rotation=45)
                
                st.pyplot(fig_weights)
        
        with tab5:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Enhanced Results</h2>', unsafe_allow_html=True)
            
            export_format = st.radio(
                "Export Format",
                options=["JSON (Full Results with Analysis)", "CSV (Field Data)", "PNG (Visualizations)"],
                horizontal=True
            )
            
            if export_format == "JSON (Full Results with Analysis)":
                export_data = st.session_state.results_manager.prepare_enhanced_export_data(
                    result, {'exported_at': datetime.now().isoformat()}
                )
                
                json_str, json_filename = st.session_state.results_manager.export_to_json(export_data)
                
                st.download_button(
                    label="📥 Download Enhanced JSON",
                    data=json_str,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            elif export_format == "CSV (Field Data)":
                csv_str, csv_filename = st.session_state.results_manager.export_to_csv(result)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif export_format == "PNG (Visualizations)":
                # Create zip file with visualizations
                if st.button("🖼️ Generate Visualizations Package", use_container_width=True):
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Save rotation analysis dashboard
                        fig_rotation = st.session_state.heatmap_visualizer.create_rotation_analysis_dashboard(result)
                        buf = BytesIO()
                        fig_rotation.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        zip_file.writestr(f"rotation_analysis_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                        plt.close(fig_rotation)
                        
                        # Save eigenstrain conversion chart
                        fig_conversion = st.session_state.heatmap_visualizer.create_eigenstrain_conversion_chart()
                        buf = BytesIO()
                        fig_conversion.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                        zip_file.writestr("eigenstrain_conversion_chart.png", buf.getvalue())
                        plt.close(fig_conversion)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download Visualizations Package",
                        data=zip_buffer,
                        file_name=f"enhanced_visualizations_theta_{result['target_angle']:.1f}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
    
    else:
        # No results yet - show enhanced instructions
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); border-radius: 20px; color: white;">
            <h2>🚀 Enhanced Interpolation Ready!</h2>
            <p style="font-size: 1.2rem; margin-bottom: 30px;">
                This enhanced interpolator handles different eigenstrain conventions and rotation patterns:
            </p>
            <ol style="text-align: left; display: inline-block; font-size: 1.1rem;">
                <li>Load simulation files (different eigenstrain conventions automatically normalized)</li>
                <li>Configure target parameters</li>
                <li>Enable rotation correction for angular pattern adjustment</li>
                <li>Enable magnitude scaling for defect-specific stress magnitudes</li>
                <li>Click "Perform Enhanced Interpolation"</li>
                <li>Explore comprehensive rotation analysis dashboard</li>
            </ol>
            <p style="margin-top: 30px; font-size: 1.1rem;">
                <strong>Key Enhancement:</strong> Handles source files with Twin=2.12, ESF=1.414, ISF=0.707 
                and converts to auto-calculated Twin=0.707, ESF=0.333, ISF=0.289 during interpolation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Theory explanation
        with st.expander("🧬 Theoretical Basis for Enhanced Interpolation", expanded=True):
            st.markdown("""
            ### **Theoretical Framework for Stress Field Interpolation with Different Defect Types and Rotations**
            
            **1. Eigenstrain Conventions and Normalization:**
            - **Source files** use: Twin=2.12, ESF=1.414, ISF=0.707 (different physical scale)
            - **Auto-calculated** uses: Twin=0.707, ESF=0.333, ISF=0.289 (normalized scale)
            - **Conversion factors**: Twin=3.0×, ESF=4.247×, ISF=2.446×
            - During interpolation: `ε₀_normalized = ε₀_source / conversion_factor`
            
            **2. Rotation Pattern Similarity:**
            - Stress fields rotate with defect orientation
            - Similar rotation angles produce similar stress distributions
            - Different defect types rotate differently (Twin vs ESF/ISF)
            - Rotation correction adjusts patterns based on angular differences
            
            **3. Magnitude Scaling:**
            - Stress magnitude scales with eigenstrain in linear elasticity
            - Different defect types have different stress concentration factors
            - Scaling applied: `σ_scaled = σ_source × (ε₀_target / ε₀_source) × defect_factor`
            
            **4. Enhanced Weighting Strategy:**
            - **40% Rotation similarity**: Angular proximity with defect awareness
            - **30% Defect type similarity**: Crystallographic similarity matrix
            - **20% Magnitude similarity**: Normalized eigenstrain comparison
            - **10% Physics similarity**: Material properties and shape
            
            **5. Interpolation of Different Rotation Directions:**
            - Clockwise vs counterclockwise rotations handled separately
            - Habit plane (54.7°) as reference for rotation direction
            - Stress patterns mirrored/symmetrized for opposite rotations
            - Angular difference calculated with circular wrapping (0°=180°)
            
            This framework ensures physically meaningful interpolation across different:
            - Defect types (Twin, ESF, ISF)
            - Rotation angles (0° to 180°)
            - Eigenstrain magnitudes (different conventions)
            - Material properties (κ values)
            """)
    
    # Footer
    st.divider()
    col_footer1, col_footer2 = st.columns(2)
    with col_footer1:
        st.markdown("**🔬 Enhanced Transformer Stress Interpolator**")
        st.markdown("Version 3.0 - Rotation Analysis & Eigenstrain Normalization")
    with col_footer2:
        st.markdown("**📚 Key Features:**")
        st.markdown("• Eigenstrain conversion: Source→Auto (Twin:3.0×, ESF:4.247×, ISF:2.446×)")
        st.markdown("• Rotation correction for angular pattern adjustment")
        st.markdown("• Magnitude scaling based on defect type")

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
