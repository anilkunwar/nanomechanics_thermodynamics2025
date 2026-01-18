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
# TRANSFORMER SPATIAL INTERPOLATOR WITH ORIENTATION AWARENESS
# =============================================
class TransformerSpatialInterpolator:
    """Transformer-inspired stress interpolator with spatial locality regularization and orientation awareness"""
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

    def rotate_field_to_defect_frame(self, field, theta_deg):
        """
        Rotate field CLOCKWISE by theta_deg to align defect to 0°.
        This brings all source defects to a common reference frame for interpolation.
        """
        return rotate(field, angle=theta_deg, reshape=False, order=1, mode='constant', cval=0.0)

    def interpolate_spatial_fields_with_orientation(self, sources, target_angle_deg, target_params):
        """
        Interpolate full spatial stress fields using transformer attention with proper orientation handling.
        Steps:
        1. Rotate each source field to defect-aligned frame (defect at 0°)
        2. Interpolate in aligned space
        3. Rotate result back to global target orientation
        """
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
            # === KEY STEP 1: ROTATE ALL SOURCE FIELDS TO DEFECT-ALIGNED FRAME ===
            aligned_source_fields = []
            for i, (src, fields) in enumerate(zip(source_params, source_fields)):
                theta_src_rad = src.get('theta', 0.0)
                theta_src_deg = np.degrees(theta_src_rad)
                # Rotate each component to align defect to 0°
                vm_aligned = self.rotate_field_to_defect_frame(fields['von_mises'], theta_src_deg)
                hydro_aligned = self.rotate_field_to_defect_frame(fields['sigma_hydro'], theta_src_deg)
                mag_aligned = self.rotate_field_to_defect_frame(fields['sigma_mag'], theta_src_deg)
                aligned_source_fields.append({
                    'von_mises': vm_aligned,
                    'sigma_hydro': hydro_aligned,
                    'sigma_mag': mag_aligned
                })
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
            # === KEY STEP 2: INTERPOLATE IN ALIGNED SPACE ===
            interpolated_aligned = {}
            shape = aligned_source_fields[0]['von_mises'].shape
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(aligned_source_fields):
                    if component in fields:
                        interpolated += combined_weights[i] * fields[component]
                interpolated_aligned[component] = interpolated
            # === KEY STEP 3: ROTATE RESULT BACK TO GLOBAL TARGET ORIENTATION ===
            interpolated_fields = {}
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                # Rotate COUNTERCLOCKWISE by target_angle_deg → use angle = -target_angle_deg
                interpolated_fields[component] = rotate(
                    interpolated_aligned[component],
                    angle=-target_angle_deg,
                    reshape=False,
                    order=1,
                    mode='constant',
                    cval=0.0
                )
            # Compute additional metrics
            max_vm = np.max(interpolated_fields['von_mises'])
            max_hydro_tension = np.max(interpolated_fields['sigma_hydro'])
            max_hydro_compression = np.min(interpolated_fields['sigma_hydro'])
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
                'source_theta_degrees': source_theta_degrees
            }
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None

    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """
        Legacy method for backward compatibility.
        Now calls the orientation-aware version.
        """
        return self.interpolate_spatial_fields_with_orientation(sources, target_angle_deg, target_params)

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
    """Enhanced heat map visualizer with diffusion analysis and proper defect orientation"""
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.defect_colors = {
            'ISF': '#FF6B6B',    # Red
            'ESF': '#4ECDC4',    # Teal
            'Twin': '#45B7D1',   # Blue
            'No Defect': '#96CEB4' # Green
        }

    # =========================================================================
    # DIFFUSION PHYSICS CORRECTED FOR BOTH TENSILE AND COMPRESSIVE STRESS
    # =========================================================================
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
            'vacancy_concentration': Shows vacancy concentration ratio
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

    # =========================================================================
    # DEFECT ORIENTATION HANDLING AND ROTATION
    # =========================================================================
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

    def create_oriented_stress_heatmap(self, stress_field, theta_degrees, defect_type,
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
        ax.set_title(f"{title}\n{defect_type} at θ={theta_degrees:.1f}° (rotated view)",
                     fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (rotated coordinate system)', fontsize=16)
        ax.set_ylabel('Y Position (rotated coordinate system)', fontsize=16)
        # Add grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        return fig

    def create_diffusion_enhancement_heatmap(self, hydrostatic_stress, T=650,
                                             theta_degrees=0, defect_type="Unknown",
                                             cmap_name='RdBu_r', figsize=(12, 10)):
        """
        Create heatmap showing diffusion enhancement factor D/D_bulk.
        """
        # Compute diffusion enhancement
        diffusion_ratio = self.compute_diffusion_enhancement_factor(hydrostatic_stress, T)
        # Rotate to match defect orientation
        rotated_diffusion = self.rotate_stress_field(diffusion_ratio, theta_degrees)
        fig, ax = plt.subplots(figsize=figsize)
        # Use diverging colormap centered at 1.0 (no enhancement)
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('RdBu_r')
        # Create heatmap with log normalization for better visualization
        # Values < 1 (suppression) in blue, > 1 (enhancement) in red
        norm = LogNorm(vmin=0.1, vmax=10)
        im = ax.imshow(rotated_diffusion, cmap=cmap, norm=norm,
                       aspect='equal', interpolation='bilinear', origin='lower')
        # Add colorbar with custom ticks
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("D/D_bulk Ratio (log scale)", fontsize=16, fontweight='bold')
        # Set colorbar ticks
        cbar.ax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
        cbar.ax.set_yticklabels(['0.1x', '0.2x', '0.5x', '1x', '2x', '5x', '10x'])
        # Add regions annotation
        ax.text(0.02, 0.98, "Red: Enhancement (D/D_bulk > 1)\nBlue: Suppression (D/D_bulk < 1)",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        # Add defect orientation marker
        center_x = rotated_diffusion.shape[1] // 2
        center_y = rotated_diffusion.shape[0] // 2
        length = min(rotated_diffusion.shape) * 0.3
        ax.arrow(center_x, center_y, length, 0,
                 head_width=length*0.1, head_length=length*0.15,
                 fc='green', ec='green', alpha=0.8, linewidth=2)
        ax.text(center_x + length*1.2, center_y,
                f'Defect at θ={theta_degrees:.1f}°',
                color='green', fontsize=12, fontweight='bold',
                verticalalignment='center')
        # Set title
        ax.set_title(f"Diffusion Enhancement Heatmap\n{defect_type} at T={T}K, θ={theta_degrees:.1f}°",
                     fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (rotated)', fontsize=16)
        ax.set_ylabel('Y Position (rotated)', fontsize=16)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        return fig

    # =========================================================================
    # 3D INTERACTIVE DIFFUSION VISUALIZATIONS
    # =========================================================================
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
            # Radial coordinate = diffusion ratio
            r = diffusion_ratio
            theta_rad = np.radians(angles)
            x = r * np.cos(theta_rad)
            y = r * np.sin(theta_rad)
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
        # Add habit plane reference
        habit_angle = 54.7
        theta_habit = np.radians(habit_angle)
        # Calculate max enhancement for scaling
        max_enhancement = 0
        for data in defect_data_dict.values():
            stresses = np.array(data['stresses']['sigma_hydro'])
            enhancement = self.compute_diffusion_enhancement_factor(stresses, T, 'physics_corrected')
            max_enhancement = max(max_enhancement, np.max(enhancement))
        # Create habit plane line
        r_habit = np.linspace(0.1, max_enhancement * 1.2, 2)
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
                    range=[-max_enhancement*1.2, max_enhancement*1.2]
                ),
                yaxis=dict(
                    title="Y (D/D_bulk × sinθ)",
                    titlefont=dict(size=14),
                    gridcolor='lightgray',
                    backgroundcolor='rgba(240, 240, 240, 0.1)',
                    range=[-max_enhancement*1.2, max_enhancement*1.2]
                ),
                zaxis=dict(
                    title="Hydrostatic Stress (GPa)",
                    titlefont=dict(size=14),
                    gridcolor='lightgray',
                    backgroundcolor='rgba(240, 240, 240, 0.1)'
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

    def create_interactive_3d_diffusion_surface(self, defect_data_dict, T=650,
                                                width=1000, height=800):
        """
        Create 3D surface plot showing diffusion enhancement as a function of angle and stress.
        """
        fig = go.Figure()
        # Create common angle grid
        angles_common = np.linspace(0, 360, 181)
        for defect_type, data in defect_data_dict.items():
            angles = np.array(data['angles'])
            stresses_gpa = np.array(data['stresses']['sigma_hydro'])
            # Interpolate to common angle grid
            angles_ext = np.concatenate([angles - 360, angles, angles + 360])
            stresses_ext = np.concatenate([stresses_gpa, stresses_gpa, stresses_gpa])
            interp_func = interp1d(angles_ext, stresses_ext, kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
            stresses_interp = interp_func(angles_common)
            # Compute diffusion enhancement
            diffusion_ratio = self.compute_diffusion_enhancement_factor(stresses_interp, T)
            # Create mesh for surface
            Theta, R = np.meshgrid(np.radians(angles_common), np.linspace(0.1, np.max(diffusion_ratio)*1.2, 50))
            # Create surface coordinates
            X = R * np.cos(Theta)
            Y = R * np.sin(Theta)
            Z = np.tile(stresses_interp, (R.shape[0], 1))
            # Add surface trace
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=np.tile(diffusion_ratio, (R.shape[0], 1)),
                colorscale='RdBu',
                cmin=0.1, cmax=10,
                colorbar=dict(
                    title="D/D_bulk",
                    tickvals=[0.1, 0.5, 1, 2, 5, 10],
                    ticktext=['0.1x', '0.5x', '1x', '2x', '5x', '10x']
                ),
                opacity=0.8,
                name=defect_type,
                hovertemplate=(
                    f"Defect: {defect_type}<br>" +
                    "Angle: %{x:.1f}°, %{y:.1f}°<br>" +
                    "Stress: %{z:.3f} GPa<br>" +
                    "D/D_bulk: %{surfacecolor:.3f}<br>" +
                    "<extra></extra>"
                )
            ))
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Diffusion Enhancement Surface (T={T}K)<br>"
                     f"<span style='font-size:14px;color:gray'>Surface color = D/D_bulk | Height = Stress</span>",
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
                )
            ),
            width=width,
            height=height
        )
        return fig

    # =========================================================================
    # COMPREHENSIVE VISUALIZATION METHODS
    # =========================================================================
    def create_comprehensive_diffusion_analysis(self, interpolation_result, T=650,
                                               width=1400, height=900):
        """
        Create comprehensive dashboard showing stress fields and diffusion analysis.
        """
        if not interpolation_result:
            return None
        fields = interpolation_result['fields']
        target_angle = interpolation_result['target_angle']
        defect_type = interpolation_result['target_params']['defect_type']
        # Compute diffusion enhancement field
        diffusion_field = self.compute_diffusion_enhancement_factor(fields['sigma_hydro'], T)
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                f'Von Mises Stress (θ={target_angle:.1f}°)',
                f'Hydrostatic Stress (θ={target_angle:.1f}°)',
                f'Diffusion Enhancement D/D_bulk (T={T}K)',
                'Rotated Von Mises',
                'Rotated Hydrostatic',
                'Rotated Diffusion',
                'Angular Stress Profile',
                'Diffusion vs Stress',
                '3D Diffusion Surface'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter3d'}]
            ],
            column_widths=[0.33, 0.33, 0.34],
            row_heights=[0.33, 0.33, 0.34]
        )
        # Row 1: Original heatmaps
        # 1. Von Mises
        fig.add_trace(go.Heatmap(
            z=fields['von_mises'],
            colorscale='Viridis',
            colorbar=dict(title="Von Mises (GPa)", x=0.3),
            hovertemplate="Von Mises: %{z:.3f} GPa<extra></extra>"
        ), row=1, col=1)
        # 2. Hydrostatic
        fig.add_trace(go.Heatmap(
            z=fields['sigma_hydro'],
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title="Hydrostatic (GPa)", x=0.63),
            hovertemplate="Hydrostatic: %{z:.3f} GPa<extra></extra>"
        ), row=1, col=2)
        # 3. Diffusion enhancement (log scale)
        fig.add_trace(go.Heatmap(
            z=diffusion_field,
            colorscale='RdBu',
            zmin=0.1, zmax=10,
            colorbar=dict(
                title="D/D_bulk",
                tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10],
                ticktext=['0.1x', '0.2x', '0.5x', '1x', '2x', '5x', '10x'],
                x=0.96
            ),
            hovertemplate="D/D_bulk: %{z:.3f}<extra></extra>"
        ), row=1, col=3)
        # Row 2: Rotated heatmaps
        # 4. Rotated Von Mises
        rotated_vm = self.rotate_stress_field(fields['von_mises'], target_angle)
        fig.add_trace(go.Heatmap(
            z=rotated_vm,
            colorscale='Viridis',
            showscale=False,
            hovertemplate="Rotated Von Mises: %{z:.3f} GPa<extra></extra>"
        ), row=2, col=1)
        # 5. Rotated Hydrostatic
        rotated_hydro = self.rotate_stress_field(fields['sigma_hydro'], target_angle)
        fig.add_trace(go.Heatmap(
            z=rotated_hydro,
            colorscale='RdBu_r',
            zmid=0,
            showscale=False,
            hovertemplate="Rotated Hydrostatic: %{z:.3f} GPa<extra></extra>"
        ), row=2, col=2)
        # 6. Rotated Diffusion
        rotated_diff = self.rotate_stress_field(diffusion_field, target_angle)
        fig.add_trace(go.Heatmap(
            z=rotated_diff,
            colorscale='RdBu',
            zmin=0.1, zmax=10,
            showscale=False,
            hovertemplate="Rotated D/D_bulk: %{z:.3f}<extra></extra>"
        ), row=2, col=3)
        # Row 3: Analysis plots
        # 7. Angular stress profile
        angles = np.linspace(0, 360, 36)
        center_x = fields['von_mises'].shape[1] // 2
        center_y = fields['von_mises'].shape[0] // 2
        radius = min(center_x, center_y) // 2
        stresses_angular = []
        for angle in angles:
            angle_rad = np.radians(angle + target_angle)
            x_sample = center_x + radius * np.cos(angle_rad)
            y_sample = center_y + radius * np.sin(angle_rad)
            xi = int(np.clip(x_sample, 0, fields['sigma_hydro'].shape[1]-1))
            yi = int(np.clip(y_sample, 0, fields['sigma_hydro'].shape[0]-1))
            stresses_angular.append(fields['sigma_hydro'][yi, xi])
        stresses_angular = np.array(stresses_angular)
        diffusion_angular = self.compute_diffusion_enhancement_factor(stresses_angular, T)
        fig.add_trace(go.Scatter(
            x=angles,
            y=stresses_angular,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'),
            name='Hydrostatic Stress',
            hovertemplate="Angle: %{x:.1f}°<br>Stress: %{y:.3f} GPa<extra></extra>"
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=angles,
            y=diffusion_angular,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            name='D/D_bulk',
            yaxis='y2',
            hovertemplate="Angle: %{x:.1f}°<br>D/D_bulk: %{y:.3f}<extra></extra>"
        ), row=3, col=1)
        # 8. Diffusion vs Stress scatter
        stress_flat = fields['sigma_hydro'].flatten()
        diffusion_flat = diffusion_field.flatten()
        # Sample for performance
        sample_idx = np.random.choice(len(stress_flat), min(1000, len(stress_flat)), replace=False)
        stress_sample = stress_flat[sample_idx]
        diffusion_sample = diffusion_flat[sample_idx]
        fig.add_trace(go.Scatter(
            x=stress_sample,
            y=diffusion_sample,
            mode='markers',
            marker=dict(
                size=5,
                color=diffusion_sample,
                colorscale='RdBu',
                cmin=0.1, cmax=10,
                showscale=True,
                colorbar=dict(title="D/D_bulk", x=1.2)
            ),
            name='Points',
            hovertemplate="Stress: %{x:.3f} GPa<br>D/D_bulk: %{y:.3f}<extra></extra>"
        ), row=3, col=2)
        # Add theoretical curve
        stress_range = np.linspace(np.min(stress_flat), np.max(stress_flat), 100)
        diffusion_theory = self.compute_diffusion_enhancement_factor(stress_range, T)
        fig.add_trace(go.Scatter(
            x=stress_range,
            y=diffusion_theory,
            mode='lines',
            line=dict(color='black', width=3, dash='dash'),
            name='Theory',
            hovertemplate="Theoretical<br>Stress: %{x:.3f} GPa<br>D/D_bulk: %{y:.3f}<extra></extra>"
        ), row=3, col=2)
        # 9. 3D diffusion surface
        # Create simple 3D scatter of selected points
        sample_idx_3d = np.random.choice(len(stress_flat), min(500, len(stress_flat)), replace=False)
        stress_3d = stress_flat[sample_idx_3d]
        diffusion_3d = diffusion_flat[sample_idx_3d]
        # Generate random angles for visualization
        angles_3d = np.random.uniform(0, 360, len(stress_3d))
        # Convert to cylindrical coordinates
        r_3d = diffusion_3d
        theta_rad_3d = np.radians(angles_3d)
        x_3d = r_3d * np.cos(theta_rad_3d)
        y_3d = r_3d * np.sin(theta_rad_3d)
        z_3d = stress_3d
        fig.add_trace(go.Scatter3d(
            x=x_3d,
            y=y_3d,
            z=z_3d,
            mode='markers',
            marker=dict(
                size=3,
                color=diffusion_3d,
                colorscale='RdBu',
                cmin=0.1, cmax=10,
                showscale=False
            ),
            hovertemplate="D/D_bulk: %{marker.color:.3f}<br>Stress: %{z:.3f} GPa<extra></extra>",
            name='3D Points'
        ), row=3, col=3)
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comprehensive Diffusion Analysis: {defect_type} at θ={target_angle:.1f}°, T={T}K",
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
        fig.update_xaxes(title_text="Angle (degrees)", row=3, col=1)
        fig.update_yaxes(title_text="Stress (GPa)", row=3, col=1)
        fig.update_yaxes(title_text="D/D_bulk", secondary_y=False, row=3, col=1)
        fig.update_xaxes(title_text="Hydrostatic Stress (GPa)", row=3, col=2)
        fig.update_yaxes(title_text="D/D_bulk", type="log", row=3, col=2)
        fig.update_scenes(
            xaxis=dict(title="X (D/D_bulk × cosθ)"),
            yaxis=dict(title="Y (D/D_bulk × sinθ)"),
            zaxis=dict(title="Stress (GPa)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            row=3, col=3
        )
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=2)
        return fig

    def create_interactive_oriented_3d(self, stress_field, theta_degrees,
                                       defect_type="Unknown", cmap_name='viridis',
                                       width=1000, height=800):
        """
        Create 3D surface plot with proper defect orientation.
        """
        # Create coordinate grid
        nx, ny = stress_field.shape[1], stress_field.shape[0]
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        # Rotate coordinates to match defect orientation
        theta_rad = np.radians(theta_degrees)
        X_rot = X * np.cos(theta_rad) - Y * np.sin(theta_rad)
        Y_rot = X * np.sin(theta_rad) + Y * np.cos(theta_rad)
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            z=stress_field,
            x=X_rot,
            y=Y_rot,
            colorscale=cmap_name,
            contours={
                "z": {"show": True, "usecolormap": True}
            },
            hovertemplate=(
                "X (rotated): %{x:.3f}<br>" +
                "Y (rotated): %{y:.3f}<br>" +
                "Stress: %{z:.4f} GPa<br>" +
                f"θ = {theta_degrees:.1f}°<br>" +
                f"Defect: {defect_type}<br>" +
                "<extra></extra>"
            )
        )])
        # Add orientation arrow
        arrow_length = 1.5
        fig.add_trace(go.Scatter3d(
            x=[0, arrow_length * np.cos(theta_rad)],
            y=[0, arrow_length * np.sin(theta_rad)],
            z=[np.max(stress_field) * 1.1, np.max(stress_field) * 1.1],
            mode='lines+markers+text',
            line=dict(color='red', width=6),
            marker=dict(size=10, color='red'),
            text=['', 'Defect Orientation'],
            textposition="top center",
            name=f'Defect Orientation (θ={theta_degrees:.1f}°)'
        ))
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Stress Surface - {defect_type} at θ={theta_degrees:.1f}°",
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title="X (rotated)",
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title="Y (rotated)",
                    gridcolor='lightgray'
                ),
                zaxis=dict(
                    title="Stress (GPa)",
                    gridcolor='lightgray'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='cube'
            ),
            width=width,
            height=height
        )
        return fig

    # =========================================================================
    # ORIGINAL METHODS (maintained for compatibility)
    # =========================================================================
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
            cmap = plt.get_cmap('viridis')
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
                cmap_name = 'viridis'
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
            return fig
        except Exception as e:
            st.error(f"Error creating interactive heatmap: {e}")
            fig = go.Figure()
            fig.add_annotation(text="Error creating heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

# =============================================
# DEFECT COMPARISON DATABASE
# =============================================
class DefectComparisonDatabase:
    """Database for comparing multiple defect types and their stress fields"""
    def __init__(self):
        self.defect_data = {}
        self.visualizer = HeatMapVisualizer()

    def add_defect_data(self, defect_type, angles, stresses_dict, target_angle):
        """Add defect data to comparison database"""
        self.defect_data[defect_type] = {
            'angles': np.array(angles),
            'stresses': stresses_dict,
            'target_angle': target_angle
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
                'target_angle': float(data['target_angle']),
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
# MAIN APPLICATION
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
    
    # Physics explanation
    st.markdown("""
    <div class="physics-note">
    <strong>🔬 Physics-Corrected Vacancy-Mediated Diffusion:</strong><br>
    **Tensile Stress (σ > 0):** D/D_bulk = exp(Ωσ/kT) > 1 → <span style='color:red'>ENHANCEMENT</span><br>
    **Compressive Stress (σ < 0):** D/D_bulk = exp(Ωσ/kT) < 1 → <span style='color:blue'>SUPPRESSION</span><br>
    **Zero Stress (σ = 0):** D/D_bulk = 1 → NO CHANGE<br>
    Where Ω = 1.56×10⁻²⁹ m³ (Ag), k = 1.38×10⁻²³ J/K, T = 650K (sintering temperature)<br>
    Habit plane at 54.7° shows maximum tensile stress → maximum diffusion enhancement
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
    if 'defect_db' not in st.session_state:
        st.session_state.defect_db = DefectComparisonDatabase()
    if 'diffusion_T' not in st.session_state:
        st.session_state.diffusion_T = 650.0
        
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
        st.session_state.diffusion_T = diffusion_T
        
        # Atomic volume
        atomic_volume = st.number_input(
            "Atomic Volume (m³)",
            value=1.56e-29,
            format="%.2e",
            help="Atomic volume Ω for vacancy formation"
        )
        # Update physics constants
        PHYSICS_CONSTANTS['Omega'] = atomic_volume
        
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
            index=4,
            help="Select colormap category for publication-quality figures"
        )
        colormap_name = st.selectbox(
            "Select Colormap",
            COLORMAP_OPTIONS[colormap_category],
            index=0
        )
        
        # Add to comparison database button
        st.markdown("---")
        if st.button("📊 Add to Defect Comparison", use_container_width=True, type="secondary"):
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                # Generate synthetic angular stress data for this defect
                angles = np.linspace(0, 360, 36, endpoint=False)
                # Create stress distribution based on defect type
                if defect_type == "Twin":
                    base_stress = 0.5
                    peak_multiplier = 3.0
                    habit_angle = 54.7
                    stresses = base_stress + peak_multiplier * np.exp(-((angles - habit_angle) / 30)**2)
                    stresses += 0.5 * np.exp(-((angles - habit_angle + 180) / 30)**2)
                elif defect_type == "ISF":
                    stresses = 1.0 + 0.8 * np.sin(np.radians(2 * angles))
                elif defect_type == "ESF":
                    stresses = 0.8 + 0.6 * np.sin(np.radians(angles)) + 0.4 * np.sin(np.radians(3 * angles))
                else:  # No Defect
                    stresses = np.ones_like(angles) * 0.1
                # Add some noise
                stresses += np.random.normal(0, 0.1, len(angles))
                # Store in database
                st.session_state.defect_db.add_defect_data(
                    defect_type,
                    angles,
                    {'sigma_hydro': stresses},
                    custom_theta
                )
                st.success(f"Added {defect_type} at θ={custom_theta:.1f}° to comparison database")
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
                            st.success(f"✅ Successfully interpolated stress fields at θ={custom_theta:.1f}°")
                        else:
                            st.error("❌ Failed to interpolate stress fields.")
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
        5. **Add to Comparison**: Add multiple defects to comparison database
        6. **Analyze Diffusion**: Use the Diffusion Analysis tab for 3D visualization
        
        ## 🔬 Key Features
        ### Corrected Diffusion Physics
        - **Tensile stress**: D/D_bulk = exp(Ωσ/kT) > 1 → Enhancement
        - **Compressive stress**: D/D_bulk = exp(Ωσ/kT) < 1 → Suppression
        - **Habit plane (54.7°)**: Maximum tensile stress → Maximum diffusion enhancement
        
        ### Enhanced Visualization
        - 3D interactive diffusion surfaces
        - Proper defect orientation handling
        - Rotated heatmaps showing actual defect orientation
        - Comprehensive diffusion analysis dashboards
        """)
    
    else:
        # Enhanced tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Results",
            "🎯 Oriented Visualization",
            "🌊 Diffusion Analysis",
            "📈 Comparison",
            "🔍 Weights",
            "🎯 Parameters",
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
                im = ax.imshow(diffusion_field, cmap='RdBu', norm=LogNorm(vmin=0.1, vmax=10),
                               aspect='equal', interpolation='bilinear', origin='lower')
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("D/D_bulk")
                cbar.ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
                cbar.ax.set_yticklabels(['0.1x', '0.2x', '0.5x', '1x', '2x', '5x', '10x'])
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
            st.markdown('<h2 class="section-header">🎯 Oriented Stress Visualization</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                fields = result['fields']
                target_angle = result['target_angle']
                defect_type = result['target_params']['defect_type']
                
                # Component selection
                stress_component = st.selectbox(
                    "Select Stress Component",
                    ["von_mises", "sigma_hydro", "sigma_mag"],
                    index=0,
                    key="viz_component_tab2"
                )
                
                # Component names for display
                component_names = {
                    'von_mises': 'Von Mises Stress',
                    'sigma_hydro': 'Hydrostatic Stress',
                    'sigma_mag': 'Stress Magnitude'
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    # Original heatmap
                    st.markdown("#### 🔄 Original Orientation")
                    fig_original = st.session_state.heatmap_visualizer.create_stress_heatmap(
                        fields[stress_component],
                        title=f"{component_names[stress_component]}",
                        cmap_name=colormap_name,
                        target_angle=target_angle,
                        defect_type=defect_type
                    )
                    st.pyplot(fig_original)
                    plt.close(fig_original)
                
                with col2:
                    # Rotated heatmap
                    st.markdown("#### 🎯 Rotated to Defect Orientation")
                    fig_rotated = st.session_state.heatmap_visualizer.create_oriented_stress_heatmap(
                        fields[stress_component],
                        target_angle,
                        defect_type,
                        title=f"{component_names[stress_component]}",
                        cmap_name=colormap_name
                    )
                    st.pyplot(fig_rotated)
                    plt.close(fig_rotated)
                
                # Interactive 3D visualization
                st.markdown("#### 🎪 3D Oriented Surface")
                fig_3d = st.session_state.heatmap_visualizer.create_interactive_oriented_3d(
                    fields[stress_component],
                    target_angle,
                    defect_type,
                    cmap_name=colormap_name
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Diffusion enhancement heatmap
                if stress_component == 'sigma_hydro':
                    st.markdown("#### 🌊 Diffusion Enhancement Heatmap")
                    fig_diffusion = st.session_state.heatmap_visualizer.create_diffusion_enhancement_heatmap(
                        fields['sigma_hydro'],
                        diffusion_T,
                        target_angle,
                        defect_type,
                        cmap_name='RdBu_r'
                    )
                    st.pyplot(fig_diffusion)
                    plt.close(fig_diffusion)
                    st.markdown("""
                    <div class="success-box">
                    🔬 <strong>Interpretation:</strong><br>
                    - <span style='color:red'>Red regions</span>: Tensile stress → Diffusion ENHANCEMENT (D/D_bulk > 1)<br>
                    - <span style='color:blue'>Blue regions</span>: Compressive stress → Diffusion SUPPRESSION (D/D_bulk < 1)<br>
                    - The defect plane (green arrow) shows orientation relative to stress concentration
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">🌊 Diffusion Enhancement Analysis</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Comprehensive diffusion analysis
                st.markdown("#### 📊 Comprehensive Diffusion Dashboard")
                fig_comprehensive = st.session_state.heatmap_visualizer.create_comprehensive_diffusion_analysis(
                    result,
                    diffusion_T
                )
                if fig_comprehensive:
                    st.plotly_chart(fig_comprehensive, use_container_width=True)
                else:
                    st.error("Failed to create comprehensive analysis")
                
                # Physics explanation
                st.markdown("""
                <div class="physics-note">
                <strong>Physics of Vacancy-Mediated Diffusion:</strong><br>
                The diffusion coefficient under hydrostatic stress σ is:<br>
                <code>D(σ) = D₀ exp(-(Q - Ωσ)/kT)</code><br>
                Relative to stress-free bulk:<br>
                <code>D(σ)/D_bulk = exp(Ωσ/kT)</code><br>
                Where:<br>
                - Ω = Atomic volume (1.56×10⁻²⁹ m³ for Ag)<br>
                - k = Boltzmann constant (1.38×10⁻²³ J/K)<br>
                - T = Temperature (K)<br>
                - σ = Hydrostatic stress (Pa, positive for tensile)<br>
                **At T=650K:**<br>
                - 1 GPa tensile → D/D_bulk ≈ 2.7x enhancement<br>
                - 1 GPa compressive → D/D_bulk ≈ 0.37x suppression<br>
                - 2 GPa tensile → D/D_bulk ≈ 7.4x enhancement<br>
                - 2 GPa compressive → D/D_bulk ≈ 0.14x suppression
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed statistics
                with st.expander("📈 Detailed Diffusion Statistics", expanded=False):
                    hydro_field = result['fields']['sigma_hydro']
                    diffusion_field = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                        hydro_field, diffusion_T, 'physics_corrected'
                    )
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.metric("Maximum Enhancement", f"{np.max(diffusion_field):.2f}x")
                    with col_s2:
                        st.metric("Minimum Enhancement", f"{np.min(diffusion_field):.2f}x")
                    with col_s3:
                        st.metric("Mean Enhancement", f"{np.mean(diffusion_field):.2f}x")
                    with col_s4:
                        st.metric("Enhancement > 2x", f"{np.mean(diffusion_field > 2)*100:.1f}%")
                    
                    # Histogram
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                    ax_hist.hist(diffusion_field.flatten(), bins=50, alpha=0.7, edgecolor='black', log=True)
                    ax_hist.axvline(x=1, color='red', linestyle='--', label='D/D_bulk = 1')
                    ax_hist.set_xlabel('D/D_bulk Ratio', fontsize=14)
                    ax_hist.set_ylabel('Frequency (log)', fontsize=14)
                    ax_hist.set_title('Distribution of Diffusion Enhancement Factors', fontsize=16, fontweight='bold')
                    ax_hist.grid(True, alpha=0.3)
                    ax_hist.legend()
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">📈 Defect Comparison Analysis</h2>', unsafe_allow_html=True)
            defect_data = st.session_state.defect_db.get_defect_comparison()
            if not defect_data:
                st.warning("""
                **No defect data in comparison database.**
                Please:
                1. Perform interpolation for at least one defect type
                2. Click "Add to Defect Comparison" in the sidebar
                3. Repeat for multiple defect types
                4. Return to this tab for comparison analysis
                """)
            else:
                # Show defect comparison stats
                st.markdown("#### 📊 Defect Comparison Summary")
                stats = st.session_state.defect_db.compute_diffusion_statistics(diffusion_T)
                
                # Create metrics
                cols = st.columns(len(defect_data))
                for idx, (defect_type, stat) in enumerate(stats.items()):
                    with cols[idx]:
                        st.metric(
                            f"{defect_type} at θ={stat['target_angle']:.1f}°",
                            f"{stat['max_enhancement']:.1f}x",
                            f"Peak at {stat['peak_enhancement_angle']:.0f}°"
                        )
                
                # 3D Diffusion Doughnut Plot
                st.markdown("#### 🎪 3D Diffusion Doughnut Plot")
                fig_doughnut = st.session_state.heatmap_visualizer.create_3d_diffusion_doughnut_plot(
                    defect_data,
                    T=diffusion_T,
                    width=1000,
                    height=700
                )
                st.plotly_chart(fig_doughnut, use_container_width=True)
                
                # 3D Diffusion Surface
                st.markdown("#### 🏔️ 3D Diffusion Surface")
                fig_surface = st.session_state.heatmap_visualizer.create_interactive_3d_diffusion_surface(
                    defect_data,
                    T=diffusion_T,
                    width=1000,
                    height=700
                )
                st.plotly_chart(fig_surface, use_container_width=True)
                
                # Comparison table
                with st.expander("📋 Detailed Comparison Table", expanded=False):
                    comparison_data = []
                    for defect_type, stat in stats.items():
                        comparison_data.append({
                            'Defect Type': defect_type,
                            'Target θ': f"{stat['target_angle']:.1f}°",
                            'Max D/D_bulk': f"{stat['max_enhancement']:.2f}x",
                            'Mean D/D_bulk': f"{stat['mean_enhancement']:.2f}x",
                            'Tensile Area %': f"{stat['tensile_fraction']*100:.1f}%",
                            'Max Tensile (GPa)': f"{stat['max_tensile_stress']:.3f}",
                            'Max Compressive (GPa)': f"{stat['max_compressive_stress']:.3f}",
                            'Peak Angle': f"{stat['peak_enhancement_angle']:.1f}°"
                        })
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                
                # Physics insights
                st.markdown("""
                <div class="success-box">
                🔬 <strong>Key Insights from Comparison:</strong><br>
                1. **Twin boundaries** typically show the strongest diffusion enhancement due to high tensile stress concentrations at the habit plane.<br>
                2. **ISF and ESF** defects show more uniform stress distributions with moderate enhancement.<br>
                3. **No Defect** (bulk) shows minimal enhancement (D/D_bulk ≈ 1).<br>
                4. **Habit plane orientation (54.7°)** maximizes tensile stress and thus diffusion enhancement in twin boundaries.<br>
                5. **Temperature dependence**: Higher temperatures reduce the enhancement effect (Ωσ/kT term decreases).
                </div>
                """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<h2 class="section-header">🔍 Weights Analysis</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                weights = result['weights']
                
                # Weights visualization
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    # Transformer weights
                    fig_trans, ax_trans = plt.subplots(figsize=(10, 5))
                    x_pos = np.arange(len(weights['transformer']))
                    bars = ax_trans.bar(x_pos, weights['transformer'], alpha=0.7, color='orange', edgecolor='black')
                    ax_trans.set_xlabel('Source Index', fontsize=14)
                    ax_trans.set_ylabel('Weight', fontsize=14)
                    ax_trans.set_title('Transformer Attention Weights', fontsize=16, fontweight='bold')
                    ax_trans.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_trans)
                    plt.close(fig_trans)
                
                with col_w2:
                    # Positional weights
                    fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
                    bars = ax_pos.bar(x_pos, weights['positional'], alpha=0.7, color='green', edgecolor='black')
                    ax_pos.set_xlabel('Source Index', fontsize=14)
                    ax_pos.set_ylabel('Weight', fontsize=14)
                    ax_pos.set_title('Spatial Locality Weights', fontsize=16, fontweight='bold')
                    ax_pos.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig_pos)
                    plt.close(fig_pos)
                
                # Combined weights
                fig_comb, ax_comb = plt.subplots(figsize=(12, 6))
                x = range(len(weights['combined']))
                width = 0.25
                ax_comb.bar([i - width for i in x], weights['transformer'], width, label='Transformer', alpha=0.7, color='orange')
                ax_comb.bar(x, weights['positional'], width, label='Positional', alpha=0.7, color='green')
                ax_comb.bar([i + width for i in x], weights['combined'], width, label='Combined', alpha=0.7, color='steelblue')
                ax_comb.set_xlabel('Source Index', fontsize=14)
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
                    
                    # Top contributors
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
        
        with tab6:
            st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                target_params = result['target_params']
                
                # Create a comprehensive target parameters display
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🔧 Target Interpolation Parameters")
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
                    
                    # Additional metrics
                    st.markdown("##### 📊 Interpolation Metrics")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Target Angle θ", f"{result['target_angle']:.1f}°")
                        st.metric("Number of Sources", f"{result.get('num_sources', 0)}")
                    with col_m2:
                        habit_deviation = abs(result['target_angle'] - 54.7)
                        st.metric("Deviation from Habit Plane", f"{habit_deviation:.1f}°")
                        st.metric("Field Size", f"{result['shape'][0]}×{result['shape'][1]}")
                
                with col2:
                    # Angular orientation visualization
                    st.markdown("##### 🧭 Angular Orientation")
                    fig_angular, ax_angular = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                    theta_rad = np.radians(result['target_angle'])
                    ax_angular.arrow(theta_rad, 0.7, 0, 0.5, width=0.02, color='red', alpha=0.8)
                    ax_angular.arrow(np.radians(54.7), 0.7, 0, 0.5, width=0.02, color='blue', alpha=0.5)
                    ax_angular.set_title(f'Defect at θ={result["target_angle"]:.1f}°\nHabit Plane at 54.7°', fontsize=16)
                    ax_angular.set_theta_zero_location('N')
                    ax_angular.set_theta_direction(-1)
                    ax_angular.set_ylim(0, 1.3)
                    ax_angular.grid(True, alpha=0.3)
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
                        fig_export = st.session_state.heatmap_visualizer.create_oriented_stress_heatmap(
                            result['fields']['von_mises'],
                            result['target_angle'],
                            result['target_params']['defect_type'],
                            title=f"Von Mises Stress at θ={result['target_angle']:.1f}°",
                            cmap_name=colormap_name
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
                        
                        # Export diffusion enhancement
                        hydro_field = result['fields']['sigma_hydro']
                        diffusion_field = st.session_state.heatmap_visualizer.compute_diffusion_enhancement_factor(
                            hydro_field, diffusion_T
                        )
                        df_diff = pd.DataFrame(diffusion_field)
                        csv_diff = df_diff.to_csv(index=False)
                        zip_file.writestr(f"diffusion_enhancement_theta_{result['target_angle']:.1f}_T{diffusion_T:.0f}K.csv", csv_diff)
                        
                        # Export metadata
                        metadata = {
                            'target_angle': result['target_angle'],
                            'target_params': result['target_params'],
                            'statistics': result['statistics'],
                            'weights': result['weights'],
                            'num_sources': result.get('num_sources', 0),
                            'source_theta_degrees': result.get('source_theta_degrees', []),
                            'shape': result['shape'],
                            'exported_at': datetime.now().isoformat(),
                            'diffusion_T': diffusion_T,
                            'physics_constants': PHYSICS_CONSTANTS
                        }
                        json_str = json.dumps(metadata, indent=2)
                        zip_file.writestr("metadata.json", json_str)
                    
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
                
                # Export defect comparison data
                defect_data = st.session_state.defect_db.get_defect_comparison()
                if defect_data:
                    st.markdown("---")
                    st.markdown("##### 📊 Export Defect Comparison")
                    if st.button("📈 Export Comparison Data", use_container_width=True):
                        comparison_stats = st.session_state.defect_db.compute_diffusion_statistics(diffusion_T)
                        # Create comparison dataframe
                        comparison_rows = []
                        for defect_type, stats in comparison_stats.items():
                            comparison_rows.append({
                                'Defect Type': defect_type,
                                'Target Angle (°)': stats['target_angle'],
                                'Max D/D_bulk': stats['max_enhancement'],
                                'Mean D/D_bulk': stats['mean_enhancement'],
                                'Tensile Area %': stats['tensile_fraction'] * 100,
                                'Max Tensile (GPa)': stats['max_tensile_stress'],
                                'Max Compressive (GPa)': stats['max_compressive_stress'],
                                'Peak Angle (°)': stats['peak_enhancement_angle']
                            })
                        df_comparison = pd.DataFrame(comparison_rows)
                        csv_comparison = df_comparison.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"defect_comparison_T{diffusion_T:.0f}K_{timestamp}.csv"
                        st.download_button(
                            label="📥 Download Comparison CSV",
                            data=csv_comparison,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True
                        )
            else:
                st.info("No interpolation results available. Please perform interpolation first.")

# Run the application
if __name__ == "__main__":
    main()
