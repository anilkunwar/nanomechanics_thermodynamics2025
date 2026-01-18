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
from scipy.interpolate import interp1d, griddata
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PHYSICS CONSTANTS FOR DIFFUSION CALCULATION
# =============================================
# Material parameters for Silver (Ag)
PHYSICS_CONSTANTS = {
    'Omega': 1.56e-29,      # Atomic volume for Ag (m³)
    'k_B': 1.38e-23,        # Boltzmann constant (J/K)
    'Q_bulk': 1.1e-19,      # Activation energy for bulk diffusion (J) ~0.7 eV
    'Q_gb': 8.0e-20,        # Activation energy for grain boundary diffusion (J) ~0.5 eV
    'Q_surface': 6.4e-20,   # Activation energy for surface diffusion (J) ~0.4 eV
    'D0_bulk': 1.0e-6,      # Pre-exponential factor for bulk diffusion (m²/s)
    'D0_gb': 1.0e-5,        # Pre-exponential factor for GB diffusion (m²/s)
    'D0_surface': 1.0e-4,   # Pre-exponential factor for surface diffusion (m²/s)
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
    """Enhanced heat map visualizer with diffusion physics and proper orientation"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.defect_colors = {
            'ISF': '#FF6B6B',    # Red
            'ESF': '#4ECDC4',    # Teal
            'Twin': '#45B7D1',   # Blue
            'No Defect': '#96CEB4' # Green
        }
    
    # =============================================
    # DIFFUSION PHYSICS METHODS
    # =============================================
    
    def compute_diffusion_enhancement_factor(self, sigma_hydro, T=650, mode='physics_corrected'):
        """
        Compute diffusion enhancement factor D/D_bulk for vacancy-mediated diffusion.
        
        CORRECTED PHYSICS: Both tensile AND compressive stress affect diffusion.
        
        Parameters:
        -----------
        sigma_hydro : array-like
            Hydrostatic stress in GPa (positive = tensile, negative = compressive)
        T : float
            Temperature in Kelvin
        mode : str
            'physics_corrected': Full physics - both tensile and compressive affect diffusion
            'temperature_reduction': Shows equivalent temperature reduction
            'activation_energy': Shows effective activation energy change
        
        Returns:
        --------
        enhancement_ratio : array-like
            D/D_bulk ratio (>1 = enhancement, <1 = suppression)
        """
        # Convert stress from GPa to Pa
        sigma_pa = sigma_hydro * 1e9
        
        # Extract constants
        Omega = PHYSICS_CONSTANTS['Omega']
        k_B = PHYSICS_CONSTANTS['k_B']
        Q_bulk = PHYSICS_CONSTANTS['Q_bulk']
        
        if mode == 'physics_corrected':
            # CORRECTED: Full exponential for both tensile and compressive
            # D/D_bulk = exp(Ωσ/kT)
            # σ > 0 (tensile): D/D_bulk > 1 (enhancement)
            # σ < 0 (compressive): D/D_bulk < 1 (suppression)
            # σ = 0: D/D_bulk = 1 (no change)
            enhancement = np.exp(Omega * sigma_pa / (k_B * T))
            return enhancement
            
        elif mode == 'temperature_reduction':
            # Calculate equivalent temperature that would give same D as stressed material
            # D_bulk(T) = D_stressed(T_eff)
            # exp(-Q/kT) = exp(-(Q - Ωσ)/kT_eff)
            # => T_eff = T * (1 - Ωσ/Q)^-1
            T_eff = T / (1 - Omega * sigma_pa / Q_bulk)
            # Avoid extreme values
            T_eff = np.clip(T_eff, 0.1 * T, 10 * T)
            return T_eff / T  # Normalized temperature factor
            
        elif mode == 'activation_energy':
            # Calculate effective activation energy
            # Q_eff = Q - Ωσ
            Q_eff = Q_bulk - Omega * sigma_pa
            return Q_eff / Q_bulk  # Normalized activation energy
            
        elif mode == 'vacancy_concentration':
            # Calculate vacancy concentration ratio
            # C_v/C_v0 = exp(Ωσ/kT)
            return np.exp(Omega * sigma_pa / (k_B * T))
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # =============================================
    # ORIENTATION CORRECTION METHODS
    # =============================================
    
    def rotate_stress_field(self, stress_field, theta_degrees, order=3):
        """
        Rotate stress field by theta_degrees counterclockwise.
        
        Parameters:
        -----------
        stress_field : 2D numpy array
            Stress field data
        theta_degrees : float
            Rotation angle in degrees (counterclockwise)
        order : int
            Interpolation order (1=bilinear, 3=cubic)
        
        Returns:
        --------
        rotated_field : 2D numpy array
            Rotated stress field
        """
        # Rotate the field (negative because rotate uses clockwise convention)
        rotated = rotate(stress_field, angle=-theta_degrees, 
                        reshape=False, order=order, mode='constant', cval=0.0)
        return rotated
    
    def create_oriented_stress_heatmap(self, stress_field, theta_degrees, 
                                      title="Stress Heat Map", cmap_name='viridis',
                                      figsize=(12, 10), rotation_marker=True):
        """
        Create stress heatmap with proper defect orientation.
        
        Parameters:
        -----------
        stress_field : 2D numpy array
            Stress field data
        theta_degrees : float
            Defect orientation angle in degrees
        rotation_marker : bool
            Whether to add orientation marker
        """
        # Rotate the field to align defect with horizontal
        # (So defect at theta appears at angle theta in the plot)
        rotated_field = self.rotate_stress_field(stress_field, theta_degrees)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')
        
        # Create heatmap
        im = ax.imshow(rotated_field, cmap=cmap, aspect='equal', 
                      interpolation='bilinear', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Stress (GPa)", fontsize=16, fontweight='bold')
        
        # Add orientation marker
        if rotation_marker:
            # Draw line at defect orientation
            center_x = rotated_field.shape[1] // 2
            center_y = rotated_field.shape[0] // 2
            length = min(rotated_field.shape) * 0.4
            
            # Since we rotated the field, defect is now at 0° (horizontal)
            end_x = center_x + length
            end_y = center_y
            
            ax.plot([center_x, end_x], [center_y, end_y], 
                   color='red', linewidth=3, linestyle='--', alpha=0.8)
            
            # Add arrow head
            ax.arrow(center_x, center_y, length*0.8, 0, 
                    head_width=length*0.1, head_length=length*0.15,
                    fc='red', ec='red', alpha=0.8)
            
            # Add text
            ax.text(end_x + length*0.1, center_y, 
                   f'Defect Orientation\nθ = {theta_degrees:.1f}°',
                   color='red', fontsize=12, fontweight='bold',
                   verticalalignment='center')
        
        # Set title
        ax.set_title(f"{title}\n(Rotated to show defect at θ={theta_degrees:.1f}°)", 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (rotated)', fontsize=16)
        ax.set_ylabel('Y Position (rotated)', fontsize=16)
        
        plt.tight_layout()
        return fig
    
    # =============================================
    # 3D DIFFUSION VISUALIZATION METHODS
    # =============================================
    
    def create_3d_diffusion_doughnut_plot(self, defect_data_dict, T=650, 
                                         width=1000, height=800):
        """
        Create 3D interactive 'doughnut' plot showing diffusion enhancement/suppression.
        
        Visualizes D/D_bulk as radial distance from center:
        - r > 1: Outside unit circle = enhancement
        - r < 1: Inside unit circle = suppression
        - r = 1: On unit circle = no change
        """
        
        fig = go.Figure()
        
        # Create unit circle for reference (D/D_bulk = 1)
        theta_unit = np.linspace(0, 2*np.pi, 100)
        x_unit = np.cos(theta_unit)
        y_unit = np.sin(theta_unit)
        
        fig.add_trace(go.Scatter3d(
            x=x_unit,
            y=y_unit,
            z=np.zeros_like(x_unit),
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            name='D/D_bulk = 1 (No Change)',
            hovertemplate='Reference: D/D_bulk = 1<extra></extra>'
        ))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (defect_type, data) in enumerate(defect_data_dict.items()):
            angles = np.array(data['angles'])
            stresses_gpa = np.array(data['stresses']['sigma_hydro'])
            
            # Compute diffusion ratio
            diffusion_ratio = self.compute_diffusion_enhancement_factor(stresses_gpa, T, 'physics_corrected')
            
            # Convert to cylindrical coordinates for 3D plot
            # Radial coordinate = diffusion ratio (log scale for better visualization)
            # Use log10 scale to show both enhancement and suppression symmetrically
            r_log = np.sign(diffusion_ratio - 1) * np.log10(np.abs(diffusion_ratio - 1) + 1)
            r_vis = np.where(diffusion_ratio >= 1, 
                            diffusion_ratio,  # Enhancement: use actual value
                            2 - diffusion_ratio)  # Suppression: mirror inside
            
            theta_rad = np.radians(angles)
            x = r_vis * np.cos(theta_rad)
            y = r_vis * np.sin(theta_rad)
            z = stresses_gpa  # Stress as height
            
            # Color coding: red for enhancement, blue for suppression
            colorscale = [[0, 'blue'], [0.5, 'gray'], [1, 'red']]
            color_values = (diffusion_ratio - np.min(diffusion_ratio)) / (np.max(diffusion_ratio) - np.min(diffusion_ratio))
            
            # Main 3D line
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines+markers',
                line=dict(
                    width=6,
                    color=color_values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(
                        title="D/D_bulk",
                        tickmode='array',
                        tickvals=[0, 0.5, 1],
                        ticktext=['Suppression', 'No Change', 'Enhancement'],
                        titleside="right"
                    )
                ),
                marker=dict(
                    size=5,
                    color=color_values,
                    colorscale=colorscale,
                    line=dict(width=1, color='white')
                ),
                name=f"{defect_type}",
                hovertemplate=(
                    f"Defect: {defect_type}<br>" +
                    "θ: %{customdata[0]:.1f}°<br>" +
                    "Stress: %{customdata[1]:.3f} GPa<br>" +
                    "D/D_bulk: %{customdata[2]:.3f}<br>" +
                    "Effect: <span style='color:%{customdata[3]}'>%{customdata[4]}</span><br>" +
                    "<extra></extra>"
                ),
                customdata=np.column_stack([
                    angles,
                    stresses_gpa,
                    diffusion_ratio,
                    ['red' if r >= 1 else 'blue' for r in diffusion_ratio],
                    ['Enhancement' if r >= 1 else 'Suppression' for r in diffusion_ratio]
                ])
            ))
            
            # Add enhancement/suppression regions as surfaces
            # Create mesh for filled regions
            theta_mesh = np.linspace(0, 2*np.pi, 50)
            r_mesh = np.linspace(0.1, 3, 20)
            Theta, R = np.meshgrid(theta_mesh, r_mesh)
            
            # Convert to Cartesian
            X_mesh = R * np.cos(Theta)
            Y_mesh = R * np.sin(Theta)
            
            # Create enhancement region surface (outside unit circle)
            enhancement_mask = R >= 1
            X_enh = np.where(enhancement_mask, X_mesh, np.nan)
            Y_enh = np.where(enhancement_mask, Y_mesh, np.nan)
            
            fig.add_trace(go.Surface(
                x=X_enh,
                y=Y_enh,
                z=np.zeros_like(X_enh) - 1,  # Place at bottom
                opacity=0.1,
                colorscale=[[0, 'rgba(255, 0, 0, 0.1)'], [1, 'rgba(255, 0, 0, 0.1)']],
                showscale=False,
                name='Enhancement Region',
                hovertemplate='Enhancement Region (D/D_bulk > 1)<extra></extra>'
            ))
            
            # Create suppression region surface (inside unit circle)
            suppression_mask = R <= 1
            X_sup = np.where(suppression_mask, X_mesh, np.nan)
            Y_sup = np.where(suppression_mask, Y_mesh, np.nan)
            
            fig.add_trace(go.Surface(
                x=X_sup,
                y=Y_sup,
                z=np.zeros_like(X_sup) - 1,  # Place at bottom
                opacity=0.1,
                colorscale=[[0, 'rgba(0, 0, 255, 0.1)'], [1, 'rgba(0, 0, 255, 0.1)']],
                showscale=False,
                name='Suppression Region',
                hovertemplate='Suppression Region (D/D_bulk < 1)<extra></extra>'
            ))
        
        # Add habit plane reference
        habit_angle = 54.7
        theta_habit = np.radians(habit_angle)
        
        # Create habit plane line
        r_max = max([np.max(self.compute_diffusion_enhancement_factor(
            np.array(data['stresses']['sigma_hydro']), T, 'physics_corrected'
        )) for data in defect_data_dict.values()])
        
        r_habit = np.linspace(0.1, r_max * 1.2, 2)
        x_habit = r_habit * np.cos(theta_habit)
        y_habit = r_habit * np.sin(theta_habit)
        
        fig.add_trace(go.Scatter3d(
            x=x_habit,
            y=y_habit,
            z=np.zeros_like(x_habit),
            mode='lines',
            line=dict(color='green', width=4, dash='dot'),
            name='Habit Plane (54.7°)',
            hovertemplate='Habit Plane: 54.7°<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Diffusion Enhancement/Suppression Doughnut (T={T}K)<br>"
                     f"<span style='font-size:14px;color:gray'>Radial distance = D/D_bulk | Height = Stress (GPa)</span>",
                font=dict(size=22, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title="X (D/D_bulk × cosθ)",
                    titlefont=dict(size=14),
                    gridcolor='lightgray',
                    backgroundcolor='rgba(240, 240, 240, 0.1)',
                    range=[-3, 3]
                ),
                yaxis=dict(
                    title="Y (D/D_bulk × sinθ)",
                    titlefont=dict(size=14),
                    gridcolor='lightgray',
                    backgroundcolor='rgba(240, 240, 240, 0.1)',
                    range=[-3, 3]
                ),
                zaxis=dict(
                    title="Hydrostatic Stress (GPa)",
                    titlefont=dict(size=14),
                    gridcolor='lightgray',
                    backgroundcolor='rgba(240, 240, 240, 0.1)',
                    range=[-3, 3]
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=0.8)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.6)
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        return fig
    
    def create_3d_diffusion_surface_plot(self, defect_data_dict, T=650, 
                                        width=1000, height=700):
        """
        Create 3D surface plot of diffusion enhancement factor.
        """
        
        fig = go.Figure()
        
        # Create common angle grid
        angles_common = np.linspace(0, 360, 181)
        
        for defect_type, data in defect_data_dict.items():
            angles = np.array(data['angles'])
            stresses_gpa = np.array(data['stresses']['sigma_hydro'])
            
            # Interpolate to common grid
            angles_ext = np.concatenate([angles - 360, angles, angles + 360])
            stresses_ext = np.concatenate([stresses_gpa, stresses_gpa, stresses_gpa])
            
            interp_func = interp1d(angles_ext, stresses_ext, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
            stresses_interp = interp_func(angles_common)
            
            # Compute diffusion enhancement
            diffusion_ratio = self.compute_diffusion_enhancement_factor(stresses_interp, T, 'physics_corrected')
            
            # Create meshgrid for surface
            theta_mesh = np.radians(angles_common)
            r_mesh = np.linspace(0.1, np.max(diffusion_ratio) * 1.1, 50)
            Theta, R = np.meshgrid(theta_mesh, r_mesh)
            
            # Create surface coordinates
            X = R * np.cos(Theta)
            Y = R * np.sin(Theta)
            
            # Create Z values (stress) repeated for each radius
            Z = np.tile(stresses_interp, (len(r_mesh), 1))
            
            # Color by diffusion ratio
            colors = np.tile(diffusion_ratio, (len(r_mesh), 1))
            
            # Add surface
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=colors,
                colorscale='RdBu',
                cmin=0.1,  # For suppression
                cmax=10,   # For enhancement
                colorbar=dict(
                    title="D/D_bulk",
                    tickmode='array',
                    tickvals=[0.1, 1, 10],
                    ticktext=['0.1x', '1x', '10x']
                ),
                opacity=0.8,
                name=defect_type,
                hovertemplate=(
                    f"Defect: {defect_type}<br>" +
                    "Angle: %{theta:.1f}°<br>" +
                    "D/D_bulk: %{surfacecolor:.2f}<br>" +
                    "Stress: %{z:.3f} GPa<br>" +
                    "<extra></extra>"
                ),
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Diffusion Enhancement Surface (T={T}K)<br>"
                     f"<span style='font-size:14px;color:gray'>Color = D/D_bulk | Height = Stress (GPa)</span>",
                font=dict(size=22, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title="X (cosθ)",
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title="Y (sinθ)",
                    gridcolor='lightgray'
                ),
                zaxis=dict(
                    title="Hydrostatic Stress (GPa)",
                    gridcolor='lightgray'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=width,
            height=height,
            showlegend=True
        )
        
        return fig
    
    # =============================================
    # DIFFUSION HEATMAP METHODS
    # =============================================
    
    def create_diffusion_enhancement_heatmap(self, stress_field, T=650, 
                                            title="Diffusion Enhancement Heatmap",
                                            cmap_name='RdBu', width=800, height=700):
        """
        Create heatmap of diffusion enhancement factor from stress field.
        """
        # Compute diffusion enhancement
        diffusion_field = self.compute_diffusion_enhancement_factor(stress_field, T, 'physics_corrected')
        
        # Create log-scaled version for better visualization
        log_diffusion = np.log10(diffusion_field)
        
        # Create hover text
        hover_text = []
        for i in range(diffusion_field.shape[0]):
            row_text = []
            for j in range(diffusion_field.shape[1]):
                stress_val = stress_field[i, j]
                diff_val = diffusion_field[i, j]
                effect = "Enhancement" if diff_val > 1 else "Suppression" if diff_val < 1 else "No Change"
                color = "green" if diff_val > 1 else "red" if diff_val < 1 else "gray"
                
                row_text.append(
                    f"Position: ({i}, {j})<br>"
                    f"Stress: {stress_val:.4f} GPa<br>"
                    f"D/D_bulk: {diff_val:.4f}<br>"
                    f"Effect: <span style='color:{color}'>{effect}</span>"
                )
            hover_text.append(row_text)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=log_diffusion,
            colorscale=cmap_name,
            hoverinfo='text',
            text=hover_text,
            colorbar=dict(
                title=dict(
                    text="log(D/D_bulk)",
                    font=dict(size=14)
                ),
                tickmode='array',
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=['0.01x', '0.1x', '1x', '10x', '100x']
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} (T={T}K)",
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            width=width,
            height=height,
            xaxis=dict(
                title="X Position",
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title="Y Position",
                scaleanchor="x",
                scaleratio=1
            )
        )
        
        return fig
    
    def create_oriented_diffusion_heatmap(self, stress_field, theta_degrees, T=650,
                                         title="Oriented Diffusion Enhancement",
                                         cmap_name='RdBu', width=800, height=700):
        """
        Create rotated diffusion enhancement heatmap with proper orientation.
        """
        # Compute diffusion enhancement
        diffusion_field = self.compute_diffusion_enhancement_factor(stress_field, T, 'physics_corrected')
        
        # Rotate both stress and diffusion fields
        rotated_stress = self.rotate_stress_field(stress_field, theta_degrees)
        rotated_diffusion = self.rotate_stress_field(diffusion_field, theta_degrees)
        
        # Create log-scaled version
        log_diffusion = np.log10(rotated_diffusion)
        
        # Create hover text
        hover_text = []
        for i in range(rotated_diffusion.shape[0]):
            row_text = []
            for j in range(rotated_diffusion.shape[1]):
                stress_val = rotated_stress[i, j]
                diff_val = rotated_diffusion[i, j]
                effect = "Enhancement" if diff_val > 1 else "Suppression" if diff_val < 1 else "No Change"
                color = "green" if diff_val > 1 else "red" if diff_val < 1 else "gray"
                
                row_text.append(
                    f"Position: ({i}, {j})<br>"
                    f"Stress: {stress_val:.4f} GPa<br>"
                    f"D/D_bulk: {diff_val:.4f}<br>"
                    f"Effect: <span style='color:{color}'>{effect}</span><br>"
                    f"Defect θ: {theta_degrees:.1f}°"
                )
            hover_text.append(row_text)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=log_diffusion,
            colorscale=cmap_name,
            hoverinfo='text',
            text=hover_text,
            colorbar=dict(
                title=dict(
                    text="log(D/D_bulk)",
                    font=dict(size=14)
                ),
                tickmode='array',
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=['0.01x', '0.1x', '1x', '10x', '100x']
            )
        ))
        
        # Add orientation line
        center_x = rotated_diffusion.shape[1] // 2
        center_y = rotated_diffusion.shape[0] // 2
        length = min(rotated_diffusion.shape) * 0.4
        
        fig.add_shape(
            type="line",
            x0=center_x, y0=center_y,
            x1=center_x + length, y1=center_y,
            line=dict(color="red", width=3, dash="dash")
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} - θ={theta_degrees:.1f}° (T={T}K)",
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            width=width,
            height=height,
            xaxis=dict(
                title="X Position (rotated)",
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title="Y Position (rotated)",
                scaleanchor="x",
                scaleratio=1
            ),
            annotations=[
                dict(
                    x=center_x + length + 20,
                    y=center_y,
                    text=f"Defect Orientation",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    font=dict(size=12, color="red")
                )
            ]
        )
        
        return fig
    
    # =============================================
    # COMPREHENSIVE VISUALIZATION METHODS
    # =============================================
    
    def create_comprehensive_diffusion_analysis(self, stress_field, theta_degrees, 
                                               T=650, defect_type="Unknown",
                                               width=1200, height=900):
        """
        Create comprehensive analysis with stress and diffusion visualizations.
        """
        # Compute diffusion enhancement
        diffusion_field = self.compute_diffusion_enhancement_factor(stress_field, T, 'physics_corrected')
        
        # Rotate fields for oriented views
        rotated_stress = self.rotate_stress_field(stress_field, theta_degrees)
        rotated_diffusion = self.rotate_stress_field(diffusion_field, theta_degrees)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                f'Original Stress (θ={theta_degrees:.1f}°)',
                f'Rotated Stress View',
                f'Stress Histogram',
                f'Original Diffusion (T={T}K)',
                f'Rotated Diffusion View',
                f'Diffusion Histogram',
                f'Stress-Diffusion Correlation',
                f'Angular Distribution',
                f'3D Visualization'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'xy'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'xy'}],
                [{'type': 'scatter'}, {'type': 'polar'}, {'type': 'scatter3d'}]
            ],
            column_widths=[0.3, 0.3, 0.4],
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # 1. Original stress heatmap
        fig.add_trace(go.Heatmap(
            z=stress_field,
            colorscale='Viridis',
            colorbar=dict(title="Stress (GPa)", x=0.3),
            showscale=True,
            hovertemplate="Stress: %{z:.3f} GPa<extra></extra>"
        ), row=1, col=1)
        
        # 2. Rotated stress heatmap
        fig.add_trace(go.Heatmap(
            z=rotated_stress,
            colorscale='Viridis',
            colorbar=dict(title="Stress (GPa)", x=0.63),
            showscale=True,
            hovertemplate="Rotated Stress: %{z:.3f} GPa<extra></extra>"
        ), row=1, col=2)
        
        # 3. Stress histogram
        fig.add_trace(go.Histogram(
            x=stress_field.flatten(),
            nbinsx=50,
            marker_color='blue',
            opacity=0.7,
            name='Stress Distribution',
            hovertemplate="Stress: %{x:.3f} GPa<br>Count: %{y}<extra></extra>"
        ), row=1, col=3)
        
        # 4. Original diffusion heatmap
        log_diffusion = np.log10(diffusion_field)
        fig.add_trace(go.Heatmap(
            z=log_diffusion,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title="log(D/D_bulk)",
                x=0.3,
                tickmode='array',
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=['0.01x', '0.1x', '1x', '10x', '100x']
            ),
            showscale=True,
            hovertemplate="D/D_bulk: %{z:.3f}<extra></extra>"
        ), row=2, col=1)
        
        # 5. Rotated diffusion heatmap
        log_rotated_diffusion = np.log10(rotated_diffusion)
        fig.add_trace(go.Heatmap(
            z=log_rotated_diffusion,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title="log(D/D_bulk)",
                x=0.63,
                tickmode='array',
                tickvals=[-2, -1, 0, 1, 2],
                ticktext=['0.01x', '0.1x', '10x', '100x']
            ),
            showscale=True,
            hovertemplate="Rotated D/D_bulk: %{z:.3f}<extra></extra>"
        ), row=2, col=2)
        
        # 6. Diffusion histogram
        fig.add_trace(go.Histogram(
            x=diffusion_field.flatten(),
            nbinsx=50,
            marker_color='red',
            opacity=0.7,
            name='Diffusion Distribution',
            hovertemplate="D/D_bulk: %{x:.3f}<br>Count: %{y}<extra></extra>"
        ), row=2, col=3)
        
        # 7. Stress-Diffusion correlation
        fig.add_trace(go.Scatter(
            x=stress_field.flatten()[::10],  # Sample every 10th point
            y=diffusion_field.flatten()[::10],
            mode='markers',
            marker=dict(
                size=4,
                color=diffusion_field.flatten()[::10],
                colorscale='RdBu',
                showscale=False,
                cmin=0.1,
                cmax=10
            ),
            name='Stress vs Diffusion',
            hovertemplate="Stress: %{x:.3f} GPa<br>D/D_bulk: %{y:.3f}<extra></extra>"
        ), row=3, col=1)
        
        # Add theoretical curve
        stress_range = np.linspace(np.min(stress_field), np.max(stress_field), 100)
        diffusion_theory = self.compute_diffusion_enhancement_factor(stress_range, T, 'physics_corrected')
        fig.add_trace(go.Scatter(
            x=stress_range,
            y=diffusion_theory,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Theory',
            hovertemplate="Theory<br>Stress: %{x:.3f} GPa<br>D/D_bulk: %{y:.3f}<extra></extra>"
        ), row=3, col=1)
        
        # 8. Angular distribution (polar plot)
        # Sample along a circle
        angles = np.linspace(0, 360, 36, endpoint=False)
        center_x = stress_field.shape[1] // 2
        center_y = stress_field.shape[0] // 2
        radius = min(center_x, center_y) // 2
        
        stresses_angular = []
        for angle in angles:
            # Sample in rotated coordinates
            angle_rad = np.radians(angle + theta_degrees)
            x_sample = center_x + radius * np.cos(angle_rad)
            y_sample = center_y + radius * np.sin(angle_rad)
            
            xi = int(np.clip(x_sample, 0, stress_field.shape[1]-1))
            yi = int(np.clip(y_sample, 0, stress_field.shape[0]-1))
            stresses_angular.append(stress_field[yi, xi])
        
        stresses_angular = np.array(stresses_angular)
        diffusion_angular = self.compute_diffusion_enhancement_factor(stresses_angular, T)
        
        fig.add_trace(go.Scatterpolar(
            r=stresses_angular,
            theta=angles,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'),
            name='Stress',
            hovertemplate="Angle: %{theta:.1f}°<br>Stress: %{r:.3f} GPa<extra></extra>"
        ), row=3, col=2)
        
        fig.add_trace(go.Scatterpolar(
            r=diffusion_angular,
            theta=angles,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            name='D/D_bulk',
            hovertemplate="Angle: %{theta:.1f}°<br>D/D_bulk: %{r:.3f}<extra></extra>",
            showlegend=False
        ), row=3, col=2)
        
        # 9. 3D visualization
        # Sample points for 3D plot
        sample_size = 20
        x_indices = np.linspace(0, stress_field.shape[1]-1, sample_size, dtype=int)
        y_indices = np.linspace(0, stress_field.shape[0]-1, sample_size, dtype=int)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # Convert to rotated coordinates
        theta_rad = np.radians(theta_degrees)
        X_rot = X * np.cos(theta_rad) - Y * np.sin(theta_rad)
        Y_rot = X * np.sin(theta_rad) + Y * np.cos(theta_rad)
        Z_stress = stress_field[Y, X]
        Z_diffusion = diffusion_field[Y, X]
        
        fig.add_trace(go.Scatter3d(
            x=X_rot.flatten(),
            y=Y_rot.flatten(),
            z=Z_stress.flatten(),
            mode='markers',
            marker=dict(
                size=5,
                color=Z_diffusion.flatten(),
                colorscale='RdBu',
                cmin=0.1,
                cmax=10,
                colorbar=dict(title="D/D_bulk", x=0.95),
                showscale=True
            ),
            name='3D Points',
            hovertemplate=(
                "X: %{x:.1f}<br>"
                "Y: %{y:.1f}<br>"
                "Stress: %{z:.3f} GPa<br>"
                "D/D_bulk: %{marker.color:.3f}<extra></extra>"
            )
        ), row=3, col=3)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comprehensive Diffusion Analysis: {defect_type} at θ={theta_degrees:.1f}° (T={T}K)",
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Stress (GPa)", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)
        
        fig.update_xaxes(title_text="D/D_bulk", row=2, col=3)
        fig.update_yaxes(title_text="Count", row=2, col=3)
        
        fig.update_xaxes(title_text="Stress (GPa)", row=3, col=1)
        fig.update_yaxes(title_text="D/D_bulk", type="log", row=3, col=1)
        
        # Update polar
        fig.update_polars(
            radialaxis=dict(title="Value"),
            angularaxis=dict(rotation=90, direction="clockwise"),
            row=3, col=2
        )
        
        # Update 3D scene
        fig.update_scenes(
            xaxis=dict(title="X (rotated)"),
            yaxis=dict(title="Y (rotated)"),
            zaxis=dict(title="Stress (GPa)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            row=3, col=3
        )
        
        return fig
    
    # =============================================
    # ORIGINAL VISUALIZATION METHODS (for compatibility)
    # =============================================
    
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

# =============================================
# ENHANCED DEFECT COMPARISON DATABASE
# =============================================

class DefectComparisonDatabase:
    """Database for comparing multiple defect types and their stress fields"""
    
    def __init__(self):
        self.defect_data = {}
        self.visualizer = HeatMapVisualizer()
    
    def add_defect_data(self, defect_type, angles, stresses_dict):
        """Add defect data to comparison database"""
        self.defect_data[defect_type] = {
            'angles': np.array(angles),
            'stresses': stresses_dict,
            'defect_type': defect_type
        }
    
    def get_defect_comparison(self):
        """Get defect comparison data"""
        return self.defect_data
    
    def clear_data(self):
        """Clear all defect data"""
        self.defect_data = {}
    
    def compute_diffusion_statistics(self, T=650):
        """Compute diffusion statistics for all defects"""
        stats = {}
        
        for defect_type, data in self.defect_data.items():
            stresses_gpa = np.array(data['stresses']['sigma_hydro'])
            enhancement = self.visualizer.compute_diffusion_enhancement_factor(
                stresses_gpa, T, 'physics_corrected'
            )
            
            # Basic statistics
            stats[defect_type] = {
                'max_enhancement': float(np.max(enhancement)),
                'min_enhancement': float(np.min(enhancement)),
                'mean_enhancement': float(np.mean(enhancement)),
                'std_enhancement': float(np.std(enhancement)),
                'tensile_fraction': float(np.mean(stresses_gpa > 0)),
                'max_tensile_stress': float(np.max(stresses_gpa[stresses_gpa > 0]) if np.any(stresses_gpa > 0) else 0),
                'max_compressive_stress': float(np.min(stresses_gpa[stresses_gpa < 0]) if np.any(stresses_gpa < 0) else 0),
                'peak_enhancement_angle': float(data['angles'][np.argmax(enhancement)]),
                'temperature_650K': {
                    'enhancement': enhancement.tolist(),
                    'angles': data['angles'].tolist()
                }
            }
        
        return stats

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
# MAIN APPLICATION WITH ALL ENHANCEMENTS
# =============================================

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Transformer Stress & Diffusion Analysis",
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
    .diffusion-box {
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
    st.markdown('<h1 class="main-header">🤖 Transformer Stress Field & Diffusion Analysis</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="physics-note">
    <strong>🔬 Physics-aware stress interpolation with vacancy-mediated diffusion enhancement analysis.</strong><br>
    • Interpolate stress fields at custom polar angles<br>
    • Compute diffusion enhancement factor D/D_bulk for vacancy-mediated diffusion<br>
    • 3D interactive visualization of enhancement vs. angle and stress<br>
    • Corrected defect orientation visualization with proper rotation<br>
    • Physics-corrected: Both tensile (enhancement) and compressive (suppression) stress affect diffusion
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
    if 'defect_comparison' not in st.session_state:
        st.session_state.defect_comparison = {}
    if 'defect_db' not in st.session_state:
        st.session_state.defect_db = DefectComparisonDatabase()
    
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
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.session_state.defect_comparison = {}
                st.session_state.defect_db.clear_data()
                st.success("All data cleared")
        
        # Diffusion physics parameters
        st.markdown('<h2 class="section-header">🌡️ Diffusion Parameters</h2>', unsafe_allow_html=True)
        
        # Temperature
        diffusion_T = st.slider(
            "Temperature (K)",
            min_value=300.0,
            max_value=1200.0,
            value=650.0,
            step=50.0,
            help="Sintering temperature for diffusion calculation"
        )
        
        # Atomic volume
        atomic_volume = st.number_input(
            "Atomic Volume (m³)",
            value=1.56e-29,
            format="%.2e",
            help="Atomic volume Ω for vacancy formation"
        )
        
        # Update physics constants
        PHYSICS_CONSTANTS['Omega'] = atomic_volume
        
        # Diffusion mode
        diffusion_mode = st.selectbox(
            "Diffusion Model",
            ["physics_corrected", "temperature_reduction", "activation_energy"],
            index=0,
            help="physics_corrected: Both tensile and compressive affect diffusion"
        )
        
        # Visualization type
        visualization_type = st.selectbox(
            "Visualization Type",
            ["Stress Heatmaps", "Diffusion Heatmaps", "3D Diffusion Doughnut", 
             "3D Diffusion Surface", "Comprehensive Analysis", "Oriented Views"],
            index=0,
            help="Select visualization type"
        )
        
        # Colormap selection
        colormap_category = st.selectbox(
            "Colormap Category",
            list(COLORMAP_OPTIONS.keys()),
            index=4,
            help="Select colormap category"
        )
        
        colormap_name = st.selectbox(
            "Select Colormap",
            COLORMAP_OPTIONS[colormap_category],
            index=0
        )
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle
        custom_theta = st.number_input(
            "Custom Polar Angle θ (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=54.7,
            step=0.1,
            help="Set custom polar angle for interpolation"
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
        
        # Add to comparison database button
        st.markdown("---")
        if st.button("📊 Add to Defect Comparison", use_container_width=True, type="secondary"):
            if st.session_state.interpolation_result:
                # Extract angular stress data
                result = st.session_state.interpolation_result
                
                # For demonstration, create synthetic angular distribution
                # In real application, extract from actual simulation
                angles = np.linspace(0, 360, 36, endpoint=False)
                
                # Create stress distribution based on defect type
                if defect_type == "Twin":
                    # Twin has strong tensile peaks near habit plane
                    base_stress = 0.5
                    peak_multiplier = 3.0
                    habit_angle = 54.7
                    stresses = base_stress + peak_multiplier * np.exp(-((angles - habit_angle) / 30)**2)
                    stresses += 0.5 * np.exp(-((angles - habit_angle + 180) / 30)**2)
                elif defect_type == "ISF":
                    # ISF has moderate stress
                    stresses = 1.0 + 0.8 * np.sin(np.radians(2 * angles))
                elif defect_type == "ESF":
                    # ESF has complex stress pattern
                    stresses = 0.8 + 0.6 * np.sin(np.radians(angles)) + 0.4 * np.sin(np.radians(3 * angles))
                else:  # No Defect
                    stresses = np.ones_like(angles) * 0.1
                
                # Add some noise
                stresses += np.random.normal(0, 0.1, len(angles))
                
                # Store in database
                st.session_state.defect_db.add_defect_data(
                    defect_type,
                    angles,
                    {'sigma_hydro': stresses}
                )
                
                st.success(f"Added {defect_type} to comparison database")
            else:
                st.warning("Please perform interpolation first")
        
        # Clear comparison button
        if st.button("🗑️ Clear Comparison", use_container_width=True):
            st.session_state.defect_db.clear_data()
            st.success("Comparison database cleared")
        
        # Interpolation button
        st.markdown("---")
        if st.button("🚀 Perform Transformer Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
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
                            st.success(f"✅ Successfully interpolated stress fields at θ={custom_theta:.1f}°")
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
        
        1. **Load Solutions**: Click the "Load Solutions" button in the sidebar
        2. **Set Parameters**: Configure target angle and defect type
        3. **Perform Interpolation**: Click "Perform Transformer Interpolation"
        4. **Add to Comparison**: Add multiple defects to comparison database
        5. **Analyze Diffusion**: Use the Diffusion Analysis tab for 3D visualization
        
        ## 🔬 Diffusion Enhancement Physics
        
        ### Vacancy-Mediated Diffusion
        - **Tensile Stress (σ_hydro > 0)**: Opens atomic volume, enhances vacancy formation
          - Enhancement factor: D/D_bulk = exp(Ωσ/kT)
          - Can increase diffusion by 10-100x at sintering temperatures
          
        - **Compressive Stress (σ_hydro ≤ 0)**: Closes atomic volume, inhibits vacancy formation
          - Suppression factor: D/D_bulk = exp(Ωσ/kT) < 1
          - Can decrease diffusion by 10-100x
          
        - **Habit Plane (54.7°)**: Maximum tensile stress concentration
          - Peak diffusion enhancement occurs here
          
        ### Key Parameters
        - Ω = Atomic volume (1.56×10⁻²⁹ m³ for Ag)
        - k_B = Boltzmann constant (1.38×10⁻²³ J/K)
        - T = Temperature (650K for Ag sintering)
        """)
    else:
        # Enhanced tabs with all visualizations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Results",
            "🎯 Stress Visualization",
            "🌊 Diffusion Analysis",
            "🔄 Oriented Views",
            "📈 3D Diffusion",
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
                    tensile_fraction = np.sum(result['fields']['sigma_hydro'] > 0) / result['fields']['sigma_hydro'].size
                    st.metric("Hydrostatic Range", 
                             f"{hydro_stats['max_tension']:.3f}/{hydro_stats['max_compression']:.3f} GPa",
                             f"Tensile: {tensile_fraction*100:.1f}%")
                
                with col3:
                    # Calculate diffusion enhancement
                    max_hydro = hydro_stats['max_tension']
                    enhancement = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                        max_hydro, diffusion_T, 'physics_corrected'
                    )
                    st.metric("Max Diffusion Enhancement", 
                             f"{enhancement:.1f}x",
                             f"at {max_hydro:.2f} GPa tensile")
                
                with col4:
                    mag_stats = result['statistics']['sigma_mag']
                    st.metric("Stress Magnitude Max", f"{mag_stats['max']:.3f} GPa",
                             f"Mean: {mag_stats['mean']:.3f} GPa")
                
                # Quick preview
                st.markdown("#### 👀 Stress Field Preview")
                
                # Create a quick preview figure
                fig_preview, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                titles = ['Von Mises', 'Hydrostatic', 'Magnitude']
                
                for idx, (comp, title) in enumerate(zip(components, titles)):
                    ax = axes[idx]
                    field = result['fields'][comp]
                    
                    # Use diverging colormap for hydrostatic
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
                
                # Add diffusion enhancement preview
                ax = axes[3]
                hydro_field = result['fields']['sigma_hydro']
                diffusion_field = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                    hydro_field, diffusion_T, 'physics_corrected'
                )
                
                # Log scale for enhancement
                im = ax.imshow(diffusion_field, cmap='hot', norm=LogNorm(),
                              aspect='equal', interpolation='bilinear', origin='lower')
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(f'Diffusion Enhancement\nT={diffusion_T}K', fontsize=12)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.grid(True, alpha=0.2)
                
                plt.suptitle(f"Stress Fields at θ={result['target_angle']:.1f}°, {result['target_params']['defect_type']}", 
                           fontsize=14)
                plt.tight_layout()
                st.pyplot(fig_preview)
                plt.close(fig_preview)
                
            else:
                st.info("👈 Configure parameters and click 'Perform Transformer Interpolation' to generate results")
        
        with tab2:
            st.markdown('<h2 class="section-header">🎯 Stress Field Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Component selection
                stress_component = st.selectbox(
                    "Select Stress Component",
                    ["von_mises", "sigma_hydro", "sigma_mag"],
                    index=0,
                    key="stress_component"
                )
                
                component_names = {
                    'von_mises': 'Von Mises Stress',
                    'sigma_hydro': 'Hydrostatic Stress',
                    'sigma_mag': 'Stress Magnitude'
                }
                
                # Create visualization based on selected type
                if visualization_type == "Stress Heatmaps":
                    # Original heatmap
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
                    
                    # Interactive heatmap
                    fig_interactive = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                        result['fields'][stress_component],
                        title=f"{component_names[stress_component]}",
                        cmap_name=colormap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type']
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                
                elif visualization_type == "Oriented Views":
                    # Oriented heatmap
                    fig_oriented = st.session_state.heatmap_visualizer.create_oriented_stress_heatmap(
                        result['fields'][stress_component],
                        result['target_angle'],
                        title=f"{component_names[stress_component]}",
                        cmap_name=colormap_name
                    )
                    st.pyplot(fig_oriented)
                    plt.close(fig_oriented)
                
                # Angular orientation plot
                with st.expander("🧭 Angular Orientation", expanded=False):
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        result['target_params']['defect_type']
                    )
                    st.pyplot(fig_angular)
                    plt.close(fig_angular)
                
                # Statistics
                with st.expander("📊 Statistics", expanded=False):
                    stats_mapping = {
                        'von_mises': 'von_mises',
                        'sigma_hydro': 'sigma_hydro',
                        'sigma_mag': 'sigma_mag'
                    }
                    
                    if stress_component in stats_mapping:
                        stats = result['statistics'][stats_mapping[stress_component]]
                        
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("Maximum", f"{stats['max']:.3f} GPa")
                        with col_s2:
                            if 'min' in stats:
                                st.metric("Minimum", f"{stats['min']:.3f} GPa")
                            elif 'max_compression' in stats:
                                st.metric("Max Compression", f"{stats['max_compression']:.3f} GPa")
                        with col_s3:
                            st.metric("Mean", f"{stats['mean']:.3f} GPa")
                        with col_s4:
                            st.metric("Std Dev", f"{stats['std']:.3f} GPa")
                        
                        # Histogram
                        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                        ax_hist.hist(result['fields'][stress_component].flatten(), bins=50, alpha=0.7, edgecolor='black')
                        ax_hist.set_xlabel(f'{component_names[stress_component]} (GPa)', fontsize=14)
                        ax_hist.set_ylabel('Frequency', fontsize=14)
                        ax_hist.set_title(f'Distribution of {component_names[stress_component]}', fontsize=16, fontweight='bold')
                        ax_hist.grid(True, alpha=0.3)
                        st.pyplot(fig_hist)
                        plt.close(fig_hist)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">🌊 Diffusion Enhancement Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                st.markdown("""
                <div class="physics-note">
                <strong>Physics of Vacancy-Mediated Diffusion:</strong><br>
                - <strong>Tensile Stress (σ > 0)</strong>: D/D_bulk = exp(Ωσ/kT) > 1 → <span style="color:green">ENHANCEMENT</span><br>
                - <strong>Compressive Stress (σ < 0)</strong>: D/D_bulk = exp(Ωσ/kT) < 1 → <span style="color:red">SUPPRESSION</span><br>
                - <strong>Zero Stress (σ = 0)</strong>: D/D_bulk = 1 → No change<br>
                - <strong>Habit Plane (54.7°)</strong>: Maximum tensile stress → Maximum enhancement
                </div>
                """, unsafe_allow_html=True)
                
                # Select stress component for diffusion analysis
                stress_component = st.selectbox(
                    "Select Stress Component for Diffusion",
                    ["sigma_hydro", "von_mises", "sigma_mag"],
                    index=0,
                    key="diffusion_component"
                )
                
                if visualization_type == "Diffusion Heatmaps":
                    # Diffusion enhancement heatmap
                    fig_diffusion = st.session_state.heatmap_visualizer.create_diffusion_enhancement_heatmap(
                        result['fields'][stress_component],
                        T=diffusion_T,
                        title=f"Diffusion Enhancement from {component_names.get(stress_component, stress_component)}",
                        cmap_name='RdBu',
                        width=800,
                        height=700
                    )
                    st.plotly_chart(fig_diffusion, use_container_width=True)
                
                elif visualization_type == "Oriented Views":
                    # Oriented diffusion heatmap
                    fig_oriented_diff = st.session_state.heatmap_visualizer.create_oriented_diffusion_heatmap(
                        result['fields'][stress_component],
                        result['target_angle'],
                        T=diffusion_T,
                        title=f"Oriented Diffusion Enhancement",
                        cmap_name='RdBu',
                        width=800,
                        height=700
                    )
                    st.plotly_chart(fig_oriented_diff, use_container_width=True)
                
                elif visualization_type == "Comprehensive Analysis":
                    # Comprehensive analysis
                    fig_comprehensive = st.session_state.heatmap_visualizer.create_comprehensive_diffusion_analysis(
                        result['fields'][stress_component],
                        result['target_angle'],
                        T=diffusion_T,
                        defect_type=result['target_params']['defect_type'],
                        width=1200,
                        height=900
                    )
                    st.plotly_chart(fig_comprehensive, use_container_width=True)
                
                # Diffusion statistics
                with st.expander("📊 Diffusion Statistics", expanded=True):
                    # Calculate overall statistics
                    stress_field = result['fields'][stress_component]
                    diffusion_field = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                        stress_field, diffusion_T, 'physics_corrected'
                    )
                    
                    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                    with col_d1:
                        enhancement_mask = diffusion_field > 1
                        enhancement_area = np.sum(enhancement_mask) / diffusion_field.size * 100
                        st.metric("Enhancement Area", f"{enhancement_area:.1f}%")
                    
                    with col_d2:
                        suppression_mask = diffusion_field < 1
                        suppression_area = np.sum(suppression_mask) / diffusion_field.size * 100
                        st.metric("Suppression Area", f"{suppression_area:.1f}%")
                    
                    with col_d3:
                        max_enhancement = np.max(diffusion_field)
                        st.metric("Max Enhancement", f"{max_enhancement:.1f}x")
                    
                    with col_d4:
                        min_enhancement = np.min(diffusion_field)
                        st.metric("Min Enhancement", f"{min_enhancement:.3f}x")
                    
                    # Physics insights
                    st.markdown("""
                    <div class="success-box">
                    🔬 <strong>Key Insights:</strong><br>
                    1. <strong>Habit plane orientation (54.7°)</strong> shows maximum tensile stress → maximum diffusion enhancement<br>
                    2. <strong>Enhancement regions</strong> accelerate sintering and creep processes<br>
                    3. <strong>Suppression regions</strong> improve creep resistance and thermal stability<br>
                    4. <strong>Temperature dependence</strong>: Higher temperatures reduce enhancement magnitude
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">🔄 Oriented Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Display defect orientation information
                col_o1, col_o2, col_o3 = st.columns(3)
                with col_o1:
                    st.metric("Defect Orientation θ", f"{result['target_angle']:.1f}°")
                with col_o2:
                    habit_deviation = abs(result['target_angle'] - 54.7)
                    st.metric("Deviation from Habit Plane", f"{habit_deviation:.1f}°")
                with col_o3:
                    st.metric("Defect Type", result['target_params']['defect_type'])
                
                # Select component for oriented view
                component = st.selectbox(
                    "Select Component",
                    ["von_mises", "sigma_hydro", "sigma_mag"],
                    index=0,
                    key="oriented_component"
                )
                
                # Create oriented heatmap
                fig_oriented = st.session_state.heatmap_visualizer.create_oriented_stress_heatmap(
                    result['fields'][component],
                    result['target_angle'],
                    title=f"{component_names.get(component, component)} - Oriented View",
                    cmap_name=colormap_name,
                    rotation_marker=True
                )
                
                st.pyplot(fig_oriented)
                plt.close(fig_oriented)
                
                # Create oriented diffusion heatmap
                st.markdown("#### 🌀 Oriented Diffusion Enhancement")
                
                diffusion_field = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                    result['fields']['sigma_hydro'], diffusion_T, 'physics_corrected'
                )
                
                fig_oriented_diff = st.session_state.heatmap_visualizer.create_oriented_diffusion_heatmap(
                    result['fields']['sigma_hydro'],
                    result['target_angle'],
                    T=diffusion_T,
                    title="Oriented Diffusion Enhancement",
                    cmap_name='RdBu',
                    width=800,
                    height=700
                )
                
                st.plotly_chart(fig_oriented_diff, use_container_width=True)
                
                # Orientation physics explanation
                st.markdown("""
                <div class="physics-note">
                <strong>Orientation Correction Explained:</strong><br>
                
                **Problem**: Raw stress fields are in simulation coordinates, not defect-oriented coordinates.<br>
                **Solution**: We rotate the visualization by the defect angle θ so that:
                - Defect appears horizontal in rotated view
                - Stress concentration aligns with defect plane
                - Habit plane appears at correct relative angle
                
                **Key Transformation**:
                - Original coordinates: (x, y) in simulation grid
                - Rotated coordinates: (x', y') = R(θ) × (x, y)
                - This shows the actual stress pattern relative to defect orientation
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab5:
            st.markdown('<h2 class="section-header">📈 3D Diffusion Visualization</h2>', unsafe_allow_html=True)
            
            # Check defect comparison data
            defect_data = st.session_state.defect_db.get_defect_comparison()
            
            if not defect_data:
                st.warning("""
                **No defect comparison data available.**
                
                Please:
                1. Perform interpolation for at least one defect type
                2. Click "Add to Defect Comparison" in the sidebar
                3. Repeat for multiple defect types
                4. Return to this tab for 3D visualization
                """)
                
                # Show example with synthetic data
                if st.button("Load Example Data for 3D Visualization"):
                    # Create example data
                    angles = np.linspace(0, 360, 36, endpoint=False)
                    
                    # Twin defect
                    twin_stresses = 0.5 + 3.0 * np.exp(-((angles - 54.7) / 30)**2)
                    twin_stresses += 0.5 * np.exp(-((angles - 54.7 + 180) / 30)**2)
                    twin_stresses += np.random.normal(0, 0.1, len(angles))
                    
                    # ISF defect
                    isf_stresses = 1.0 + 0.8 * np.sin(np.radians(2 * angles))
                    isf_stresses += np.random.normal(0, 0.1, len(angles))
                    
                    # ESF defect
                    esf_stresses = 0.8 + 0.6 * np.sin(np.radians(angles)) + 0.4 * np.sin(np.radians(3 * angles))
                    esf_stresses += np.random.normal(0, 0.1, len(angles))
                    
                    # Add to database
                    st.session_state.defect_db.add_defect_data("Twin", angles, {'sigma_hydro': twin_stresses})
                    st.session_state.defect_db.add_defect_data("ISF", angles, {'sigma_hydro': isf_stresses})
                    st.session_state.defect_db.add_defect_data("ESF", angles, {'sigma_hydro': esf_stresses})
                    
                    st.success("Loaded example defect data. Refresh to see visualizations.")
            else:
                # Show defect comparison stats
                stats = st.session_state.defect_db.compute_diffusion_statistics(diffusion_T)
                
                st.markdown("#### 📊 Defect Comparison Summary")
                cols = st.columns(len(defect_data))
                for idx, (defect_type, stat) in enumerate(stats.items()):
                    with cols[idx]:
                        st.metric(
                            defect_type,
                            f"{stat['max_enhancement']:.1f}x",
                            f"Peak at {stat['peak_enhancement_angle']:.0f}°"
                        )
                
                # 3D Doughnut Plot
                st.markdown("#### 🍩 3D Diffusion Doughnut Plot")
                st.markdown("""
                **Visualization Guide**:
                - **Radial distance** = D/D_bulk ratio
                - **Outside unit circle** = Enhancement (D/D_bulk > 1)
                - **Inside unit circle** = Suppression (D/D_bulk < 1)
                - **On unit circle** = No change (D/D_bulk = 1)
                - **Height** = Hydrostatic stress magnitude
                """)
                
                fig_doughnut = st.session_state.heatmap_visualizer.create_3d_diffusion_doughnut_plot(
                    defect_data,
                    T=diffusion_T,
                    width=1000,
                    height=700
                )
                
                st.plotly_chart(fig_doughnut, use_container_width=True)
                
                # 3D Surface Plot
                st.markdown("#### 🌊 3D Diffusion Surface Plot")
                
                fig_surface = st.session_state.heatmap_visualizer.create_3d_diffusion_surface_plot(
                    defect_data,
                    T=diffusion_T,
                    width=1000,
                    height=700
                )
                
                st.plotly_chart(fig_surface, use_container_width=True)
                
                # Detailed statistics
                with st.expander("📈 Detailed Statistics", expanded=False):
                    stats_df = pd.DataFrame([
                        {
                            'Defect': defect_type,
                            'Max D/D_bulk': f"{stat['max_enhancement']:.2f}",
                            'Mean D/D_bulk': f"{stat['mean_enhancement']:.2f}",
                            'Tensile Area %': f"{stat['tensile_fraction']*100:.1f}%",
                            'Max Tensile (GPa)': f"{stat['max_tensile_stress']:.3f}",
                            'Max Compressive (GPa)': f"{stat['max_compressive_stress']:.3f}",
                            'Peak Angle (°)': f"{stat['peak_enhancement_angle']:.1f}"
                        }
                        for defect_type, stat in stats.items()
                    ])
                    
                    st.dataframe(stats_df, use_container_width=True)
        
        with tab6:
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
                    # Export as JSON
                    if st.button("💾 Export as JSON", use_container_width=True, key="export_json"):
                        visualization_params = {
                            'colormap': colormap_name,
                            'visualization_type': visualization_type,
                            'diffusion_T': diffusion_T
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
                
                # Export diffusion data
                st.markdown("---")
                st.markdown("##### 🌊 Export Diffusion Analysis")
                
                if st.button("📈 Export Diffusion Data", use_container_width=True):
                    # Compute diffusion enhancement
                    diffusion_field = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                        result['fields']['sigma_hydro'], diffusion_T, 'physics_corrected'
                    )
                    
                    # Create DataFrame
                    df_diffusion = pd.DataFrame({
                        'x_position': np.repeat(range(result['fields']['sigma_hydro'].shape[1]), 
                                              result['fields']['sigma_hydro'].shape[0]),
                        'y_position': np.tile(range(result['fields']['sigma_hydro'].shape[0]),
                                            result['fields']['sigma_hydro'].shape[1]),
                        'hydrostatic_stress_gpa': result['fields']['sigma_hydro'].flatten(),
                        'diffusion_enhancement': diffusion_field.flatten(),
                        'log_diffusion': np.log10(diffusion_field).flatten()
                    })
                    
                    csv_diffusion = df_diffusion.to_csv(index=False)
                    filename = f"diffusion_analysis_theta_{result['target_angle']:.1f}_T{diffusion_T:.0f}K.csv"
                    
                    st.download_button(
                        label="📥 Download Diffusion CSV",
                        data=csv_diffusion,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No interpolation results available. Please perform interpolation first.")

# Run the application
if __name__ == "__main__":
    main()
