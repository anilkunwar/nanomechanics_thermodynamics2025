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
# ENHANCED SOLUTION LOADER WITH ERROR HANDLING
# =============================================

class EnhancedSolutionLoader:
    """Enhanced solution loader with robust error handling"""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
            st.info(f"Created solutions directory at: {self.solutions_dir}")
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        """Scan directory for solution files with error handling"""
        all_files = []
        
        try:
            import glob
            for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth', '*.npz', '*.h5', '*.hdf5']:
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
                        'format': self._get_file_format(file_path)
                    }
                    file_info.append(info)
                except Exception as e:
                    st.warning(f"Could not get info for {file_path}: {e}")
                    continue
            return file_info
        except Exception as e:
            st.error(f"Error scanning directory: {e}")
            return []
    
    def _get_file_format(self, file_path: str) -> str:
        """Determine file format from extension"""
        if file_path.endswith(('.pkl', '.pickle')):
            return 'pkl'
        elif file_path.endswith(('.pt', '.pth')):
            return 'pt'
        elif file_path.endswith('.npz'):
            return 'npz'
        elif file_path.endswith(('.h5', '.hdf5')):
            return 'h5'
        else:
            return 'unknown'
    
    def read_simulation_file(self, file_path, format_type='auto'):
        """Read simulation file with robust error handling"""
        try:
            if format_type == 'auto':
                format_type = self._get_file_format(file_path)
            
            with open(file_path, 'rb') as f:
                if format_type in ['pt', 'pth']:
                    # PyTorch file
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                elif format_type in ['pkl', 'pickle']:
                    # Pickle file
                    data = pickle.load(f)
                elif format_type == 'npz':
                    # NumPy file
                    import numpy as np
                    data = dict(np.load(f, allow_pickle=True))
                else:
                    st.error(f"Unsupported format: {format_type}")
                    return None
            
            # Standardize data structure
            standardized = self._standardize_data(data, file_path)
            return standardized
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data, file_path):
        """Standardize simulation data with robust handling"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'format_checked': False
            }
        }
        
        try:
            if isinstance(data, dict):
                # Extract parameters with fallbacks
                param_sources = ['params', 'parameters', 'sim_params', 'model_params']
                for source in param_sources:
                    if source in data:
                        standardized['params'] = data[source]
                        break
                
                # Extract history with fallbacks
                history_sources = ['history', 'results', 'frames', 'time_series']
                for source in history_sources:
                    if source in data:
                        history_data = data[source]
                        if isinstance(history_data, list):
                            standardized['history'] = history_data
                        elif isinstance(history_data, dict):
                            # Convert dict to list
                            history_list = []
                            for key in sorted(history_data.keys()):
                                if isinstance(history_data[key], dict):
                                    history_list.append(history_data[key])
                            standardized['history'] = history_list
                        break
                
                # Extract additional metadata
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                
                # Ensure required parameters have defaults
                if 'defect_type' not in standardized['params']:
                    standardized['params']['defect_type'] = 'Unknown'
                if 'theta' not in standardized['params']:
                    standardized['params']['theta'] = 0.0
                if 'eps0' not in standardized['params']:
                    standardized['params']['eps0'] = 0.707
                if 'kappa' not in standardized['params']:
                    standardized['params']['kappa'] = 0.6
            
            # Convert tensors to numpy arrays
            self._convert_tensors(standardized)
            
            standardized['metadata']['format_checked'] = True
            standardized['metadata']['has_history'] = len(standardized['history']) > 0
            
            # Check if stress data is available
            if standardized['history']:
                last_frame = standardized['history'][-1]
                if isinstance(last_frame, dict) and 'stresses' in last_frame:
                    standardized['metadata']['has_stresses'] = True
                else:
                    standardized['metadata']['has_stresses'] = False
            
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
            standardized['metadata']['format_checked'] = False
        
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
    
    def load_all_solutions(self, use_cache=True, max_files=None, show_progress=True):
        """Load all solutions with progress tracking"""
        solutions = []
        file_info = self.scan_solutions()
        
        if not file_info:
            st.warning(f"No simulation files found in {self.solutions_dir}")
            return solutions
        
        if max_files:
            file_info = file_info[:max_files]
        
        if show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for idx, file_info_item in enumerate(file_info):
            if show_progress:
                progress = (idx + 1) / len(file_info)
                progress_bar.progress(progress)
                status_text.text(f"Loading {idx + 1}/{len(file_info)}: {file_info_item['filename']}")
            
            cache_key = file_info_item['filename']
            
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
            
            solution = self.read_simulation_file(file_info_item['path'])
            if solution and solution['metadata']['format_checked']:
                self.cache[cache_key] = solution
                solutions.append(solution)
        
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
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
        self.pe = None
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        if self.pe is None or seq_len > self.pe.shape[0]:
            # Create positional indices
            position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
            
            # Compute divisor term
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-np.log(10000.0) / d_model))
            
            # Create positional encoding
            pe = torch.zeros(seq_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe
            
        return x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)

# =============================================
# ROBUST TRANSFORMER SPATIAL INTERPOLATOR
# =============================================

class RobustTransformerInterpolator:
    """Robust transformer-inspired stress interpolator with spatial locality regularization"""
    
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0,
                 feature_dim=20, dropout=0.1):
        """
        Initialize robust transformer interpolator
        
        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            spatial_sigma: Spatial regularization parameter
            temperature: Attention temperature scaling
            feature_dim: Dimension of input features
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.feature_dim = feature_dim
        
        # Transformer encoder with robust configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        # Add layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection with correct dimensions
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def compute_spatial_weights(self, source_params, target_params):
        """Compute spatial locality weights with robust distance metrics"""
        weights = []
        
        for src in source_params:
            try:
                # Compute parameter distance with multiple metrics
                param_dist = 0.0
                weight_factors = []
                
                # Compare key parameters with robust extraction
                key_params = ['eps0', 'kappa', 'theta', 'defect_type', 'shape']
                
                for param in key_params:
                    src_val = src.get(param)
                    tgt_val = target_params.get(param)
                    
                    if src_val is not None and tgt_val is not None:
                        if param == 'defect_type':
                            # Categorical similarity
                            similarity = 1.0 if str(src_val) == str(tgt_val) else 0.0
                            weight_factors.append(similarity)
                        elif param == 'theta':
                            # Angular distance (cyclic) with robust handling
                            try:
                                src_theta = float(src_val)
                                tgt_theta = float(tgt_val)
                                diff = abs(src_theta - tgt_theta)
                                diff = min(diff, 2*np.pi - diff)  # Handle periodicity
                                normalized_diff = diff / np.pi
                                weight_factors.append(np.exp(-normalized_diff))
                            except:
                                weight_factors.append(0.5)  # Default similarity
                        elif param == 'shape':
                            # Shape similarity
                            similarity = 1.0 if str(src_val) == str(tgt_val) else 0.3
                            weight_factors.append(similarity)
                        else:
                            # Numeric parameter similarity
                            try:
                                src_val_num = float(src_val)
                                tgt_val_num = float(tgt_val)
                                max_val = {'eps0': 3.0, 'kappa': 2.0}.get(param, 1.0)
                                diff = abs(src_val_num - tgt_val_num) / max_val
                                weight_factors.append(np.exp(-diff))
                            except:
                                weight_factors.append(0.5)  # Default similarity
                    else:
                        # Missing parameter - use default similarity
                        weight_factors.append(0.5)
                
                # Combine weight factors (geometric mean for robustness)
                if weight_factors:
                    # Avoid zeros in geometric mean
                    weight_factors = [max(f, 0.01) for f in weight_factors]
                    combined_weight = np.exp(np.mean(np.log(weight_factors)))
                    
                    # Apply Gaussian kernel with spatial sigma
                    distance = 1.0 - combined_weight
                    weight = np.exp(-0.5 * (distance / self.spatial_sigma) ** 2)
                else:
                    weight = 0.5  # Default weight
                
                weights.append(weight)
                
            except Exception as e:
                # Fallback weight on error
                st.warning(f"Error computing spatial weight: {e}")
                weights.append(0.5)
        
        # Normalize weights
        weights_array = np.array(weights)
        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones_like(weights_array) / len(weights_array)
        
        return weights_array
    
    def extract_features(self, params, target_angle_deg=None):
        """Extract features from parameters with robust error handling"""
        features = []
        
        try:
            # Basic numeric features (3)
            eps0 = float(params.get('eps0', 0.707))
            kappa = float(params.get('kappa', 0.6))
            theta = float(params.get('theta', 0.0))
            
            features.append(eps0 / 3.0)  # Normalized eps0
            features.append((kappa - 0.1) / (2.0 - 0.1))  # Normalized kappa
            features.append(theta / np.pi)  # Normalized theta
            
            # Defect type encoding (5)
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect', 'Unknown']
            defect = str(params.get('defect_type', 'Unknown'))
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            
            # Shape encoding (4)
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle', 'Other']
            shape = str(params.get('shape', 'Square'))
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
            
            # Orientation features (2)
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            if target_angle_deg is not None:
                angle_diff = min(abs(theta_deg - target_angle_deg),
                                abs(theta_deg - (target_angle_deg + 360)),
                                abs(theta_deg - (target_angle_deg - 360)))
                features.append(np.exp(-angle_diff / 45.0))
            else:
                features.append(0.5)  # Default
            
            features.append(np.sin(np.radians(2 * theta_deg)))
            
            # Habit plane proximity (2)
            habit_distance = abs(theta_deg - 54.7)
            features.append(np.exp(-habit_distance / 15.0))
            features.append(1.0 if habit_distance < 10.0 else 0.0)
            
            # Stress factor features (4)
            stress_factor = 1.0 + eps0 * 0.5
            features.append(stress_factor / 2.0)  # Normalized
            
            # Eigen strain magnitude
            features.append(abs(eps0) / 3.0)
            
            # Additional physics features
            features.append(np.cos(np.radians(2 * theta_deg)))
            features.append(1.0 if theta_deg > 180 else 0.0)  # Hemisphere indicator
            
        except Exception as e:
            st.warning(f"Error extracting features: {e}")
            # Return zero features with correct dimension
            features = [0.0] * self.feature_dim
        
        # Ensure correct feature dimension
        if len(features) < self.feature_dim:
            # Pad with zeros
            features.extend([0.0] * (self.feature_dim - len(features)))
        elif len(features) > self.feature_dim:
            # Truncate
            features = features[:self.feature_dim]
        
        return features
    
    def prepare_stress_fields(self, sources):
        """Prepare stress fields from source data with robust handling"""
        source_params = []
        source_fields = []
        valid_indices = []
        
        for idx, src in enumerate(sources):
            try:
                if 'params' not in src:
                    st.warning(f"Source {idx}: Missing parameters")
                    continue
                
                source_params.append(src['params'])
                
                # Extract stress fields from history
                stress_fields = None
                if 'history' in src and src['history']:
                    last_frame = src['history'][-1]
                    if isinstance(last_frame, dict) and 'stresses' in last_frame:
                        stress_fields = last_frame['stresses']
                
                if stress_fields is None:
                    # Try alternative stress field locations
                    if 'stresses' in src:
                        stress_fields = src['stresses']
                    elif 'stress_fields' in src:
                        stress_fields = src['stress_fields']
                    elif 'stress' in src:
                        stress_fields = src['stress']
                
                if stress_fields is None:
                    st.warning(f"Source {idx}: No stress fields found")
                    continue
                
                # Extract or compute required stress components
                field_data = {}
                
                # Von Mises stress
                if 'von_mises' in stress_fields:
                    vm = stress_fields['von_mises']
                elif 'vonMises' in stress_fields:
                    vm = stress_fields['vonMises']
                elif 'vm' in stress_fields:
                    vm = stress_fields['vm']
                else:
                    vm = self.compute_von_mises(stress_fields)
                
                # Hydrostatic stress
                if 'sigma_hydro' in stress_fields:
                    hydro = stress_fields['sigma_hydro']
                elif 'hydrostatic' in stress_fields:
                    hydro = stress_fields['hydrostatic']
                elif 'hydro' in stress_fields:
                    hydro = stress_fields['hydro']
                else:
                    hydro = self.compute_hydrostatic(stress_fields)
                
                # Stress magnitude
                if 'sigma_mag' in stress_fields:
                    mag = stress_fields['sigma_mag']
                elif 'magnitude' in stress_fields:
                    mag = stress_fields['magnitude']
                elif 'mag' in stress_fields:
                    mag = stress_fields['mag']
                else:
                    mag = np.sqrt(vm**2 + hydro**2)
                
                # Ensure fields are numpy arrays
                if torch.is_tensor(vm):
                    vm = vm.cpu().numpy()
                if torch.is_tensor(hydro):
                    hydro = hydro.cpu().numpy()
                if torch.is_tensor(mag):
                    mag = mag.cpu().numpy()
                
                field_data['von_mises'] = vm
                field_data['sigma_hydro'] = hydro
                field_data['sigma_mag'] = mag
                
                source_fields.append(field_data)
                valid_indices.append(idx)
                
            except Exception as e:
                st.warning(f"Error processing source {idx}: {e}")
                continue
        
        return source_params, source_fields, valid_indices
    
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """Robust interpolation of spatial stress fields"""
        
        if not sources:
            st.error("No sources provided for interpolation")
            return None
        
        try:
            # Prepare source data
            source_params, source_fields, valid_indices = self.prepare_stress_fields(sources)
            
            if not source_params or not source_fields:
                st.error("No valid source data found")
                return None
            
            # Ensure all fields have same shape
            try:
                base_shape = source_fields[0]['von_mises'].shape
                resized_fields = []
                
                for fields in source_fields:
                    resized = {}
                    for key, field in fields.items():
                        if field.shape != base_shape:
                            # Calculate zoom factors
                            factors = [t/s for t, s in zip(base_shape, field.shape)]
                            resized_field = zoom(field, factors, order=1, mode='nearest')
                            resized[key] = resized_field
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                
                source_fields = resized_fields
            except Exception as e:
                st.warning(f"Could not resize fields to common shape: {e}")
                # Continue with original shapes
            
            # Compute spatial weights
            try:
                spatial_weights = self.compute_spatial_weights(source_params, target_params)
            except Exception as e:
                st.warning(f"Error computing spatial weights: {e}")
                spatial_weights = np.ones(len(source_params)) / len(source_params)
            
            # Extract features
            try:
                # Extract source features
                source_features = []
                for params in source_params:
                    features = self.extract_features(params, target_angle_deg)
                    source_features.append(features)
                
                # Extract target features
                target_features = [self.extract_features(target_params, target_angle_deg)]
                
                # Convert to tensors
                source_tensor = torch.FloatTensor(source_features)
                target_tensor = torch.FloatTensor(target_features)
                
                # Check dimensions
                if source_tensor.shape[1] != self.feature_dim:
                    st.error(f"Feature dimension mismatch: expected {self.feature_dim}, got {source_tensor.shape[1]}")
                    return None
                
            except Exception as e:
                st.error(f"Error extracting features: {e}")
                # Fallback to spatial interpolation only
                return self.spatial_only_interpolation(source_fields, spatial_weights, target_angle_deg, target_params)
            
            # Prepare transformer input
            try:
                # Combine target and sources
                all_features = torch.cat([target_tensor, source_tensor], dim=0).unsqueeze(0)  # [1, N+1, feature_dim]
                
                # Apply input projection
                proj_features = self.input_proj(all_features)
                
                # Add positional encoding
                proj_features = self.pos_encoder(proj_features)
                
                # Apply layer norm
                proj_features = self.norm(proj_features)
                
                # Transformer encoding
                transformer_output = self.transformer(proj_features)
                
                # Extract representations
                target_rep = transformer_output[:, 0, :]  # Target representation
                source_reps = transformer_output[:, 1:, :]  # Source representations
                
                # Compute attention weights
                attn_scores = torch.matmul(target_rep, source_reps.transpose(1, 2)) / np.sqrt(self.d_model)
                attn_scores = attn_scores / self.temperature
                transformer_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().numpy()
                
            except Exception as e:
                st.error(f"Error in transformer processing: {e}")
                # Fallback to equal weights
                transformer_weights = np.ones(len(source_params)) / len(source_params)
            
            # Combine weights
            try:
                combined_weights = 0.7 * transformer_weights + 0.3 * spatial_weights
                if combined_weights.sum() > 0:
                    combined_weights = combined_weights / combined_weights.sum()
                else:
                    combined_weights = np.ones_like(combined_weights) / len(combined_weights)
            except:
                combined_weights = np.ones(len(source_params)) / len(source_params)
            
            # Interpolate fields
            interpolated_fields = {}
            try:
                # Determine output shape
                output_shape = source_fields[0]['von_mises'].shape
                
                for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                    interpolated = np.zeros(output_shape)
                    
                    for i, fields in enumerate(source_fields):
                        if component in fields:
                            interpolated += combined_weights[i] * fields[component]
                    
                    interpolated_fields[component] = interpolated
                    
            except Exception as e:
                st.error(f"Error interpolating fields: {e}")
                return None
            
            # Prepare result
            result = {
                'fields': interpolated_fields,
                'weights': {
                    'transformer': transformer_weights.tolist() if 'transformer_weights' in locals() else [],
                    'spatial': spatial_weights.tolist(),
                    'combined': combined_weights.tolist() if 'combined_weights' in locals() else []
                },
                'statistics': self.compute_field_statistics(interpolated_fields),
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': interpolated_fields['von_mises'].shape if 'von_mises' in interpolated_fields else (0, 0),
                'metadata': {
                    'num_sources': len(source_params),
                    'num_valid_sources': len(valid_indices),
                    'interpolation_method': 'transformer_spatial'
                }
            }
            
            return result
            
        except Exception as e:
            st.error(f"Fatal error in interpolation: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def spatial_only_interpolation(self, source_fields, weights, target_angle_deg, target_params):
        """Fallback interpolation using only spatial weights"""
        try:
            # Determine output shape
            if not source_fields:
                return None
            
            output_shape = source_fields[0]['von_mises'].shape
            interpolated_fields = {}
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(output_shape)
                
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += weights[i] * fields[component]
                
                interpolated_fields[component] = interpolated
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'spatial': weights.tolist(),
                    'combined': weights.tolist()
                },
                'statistics': self.compute_field_statistics(interpolated_fields),
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': output_shape,
                'metadata': {
                    'num_sources': len(source_fields),
                    'interpolation_method': 'spatial_only_fallback'
                }
            }
            
        except Exception as e:
            st.error(f"Error in spatial-only interpolation: {e}")
            return None
    
    def compute_field_statistics(self, fields):
        """Compute comprehensive statistics for stress fields"""
        stats = {}
        
        for component, field in fields.items():
            try:
                flattened = field.flatten()
                stats[component] = {
                    'max': float(np.nanmax(flattened)),
                    'min': float(np.nanmin(flattened)),
                    'mean': float(np.nanmean(flattened)),
                    'std': float(np.nanstd(flattened)),
                    'median': float(np.nanmedian(flattened)),
                    'q25': float(np.nanpercentile(flattened, 25)),
                    'q75': float(np.nanpercentile(flattened, 75)),
                    'range': float(np.nanmax(flattened) - np.nanmin(flattened))
                }
            except:
                stats[component] = {
                    'max': 0.0, 'min': 0.0, 'mean': 0.0, 'std': 0.0,
                    'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'range': 0.0
                }
        
        return stats
    
    def compute_von_mises(self, stress_fields):
        """Compute von Mises stress from stress components with robust handling"""
        try:
            # Try different component naming conventions
            component_maps = {
                'sxx': ['sigma_xx', 'sxx', 'stress_xx', 'xx'],
                'syy': ['sigma_yy', 'syy', 'stress_yy', 'yy'],
                'szz': ['sigma_zz', 'szz', 'stress_zz', 'zz'],
                'txy': ['tau_xy', 'txy', 'stress_xy', 'xy'],
                'tyz': ['tau_yz', 'tyz', 'stress_yz', 'yz'],
                'tzx': ['tau_zx', 'tzx', 'stress_zx', 'zx']
            }
            
            # Extract components with fallbacks
            components = {}
            for key, aliases in component_maps.items():
                for alias in aliases:
                    if alias in stress_fields:
                        val = stress_fields[alias]
                        if torch.is_tensor(val):
                            val = val.cpu().numpy()
                        components[key] = val
                        break
                if key not in components:
                    components[key] = np.zeros((100, 100))  # Default shape
            
            sxx = components['sxx']
            syy = components['syy']
            szz = components['szz']
            txy = components['txy']
            tyz = components['tyz']
            tzx = components['tzx']
            
            # Compute von Mises stress
            von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 
                                     6*(txy**2 + tyz**2 + tzx**2)))
            
            return von_mises
            
        except Exception as e:
            st.warning(f"Error computing von Mises: {e}")
            # Return a default field
            return np.zeros((100, 100))
    
    def compute_hydrostatic(self, stress_fields):
        """Compute hydrostatic stress with robust handling"""
        try:
            # Try to find normal stress components
            sxx = None
            syy = None
            szz = None
            
            for key in ['sigma_xx', 'sxx', 'stress_xx', 'xx']:
                if key in stress_fields:
                    sxx = stress_fields[key]
                    break
            
            for key in ['sigma_yy', 'syy', 'stress_yy', 'yy']:
                if key in stress_fields:
                    syy = stress_fields[key]
                    break
            
            for key in ['sigma_zz', 'szz', 'stress_zz', 'zz']:
                if key in stress_fields:
                    szz = stress_fields[key]
                    break
            
            # Convert tensors to numpy
            if torch.is_tensor(sxx):
                sxx = sxx.cpu().numpy()
            if torch.is_tensor(syy):
                syy = syy.cpu().numpy()
            if torch.is_tensor(szz):
                szz = szz.cpu().numpy()
            
            # Handle missing components
            if sxx is None:
                sxx = np.zeros((100, 100))
            if syy is None:
                syy = np.zeros((100, 100))
            if szz is None:
                szz = np.zeros((100, 100))
            
            return (sxx + syy + szz) / 3
            
        except Exception as e:
            st.warning(f"Error computing hydrostatic stress: {e}")
            return np.zeros((100, 100))

# =============================================
# ENHANCED HEATMAP VISUALIZER
# =============================================

class EnhancedHeatMapVisualizer:
    """Enhanced heat map visualizer with 50+ colormaps"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map", 
                             cmap_name='viridis', figsize=(12, 10), 
                             colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                             show_stats=True, show_grid=True):
        """Create enhanced heat map with chosen colormap"""
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')  # Default fallback
            st.warning(f"Colormap '{cmap_name}' not found. Using 'viridis'.")
        
        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = np.nanmin(stress_field)
        if vmax is None:
            vmax = np.nanmax(stress_field)
        
        # Handle case where all values are the same
        if vmin == vmax:
            vmin = vmin - 0.1
            vmax = vmax + 0.1
        
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
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics annotation
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\n"
                         f"Min: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa\n"
                         f"Std: {np.nanstd(stress_field):.3f} GPa")
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
                val = stress_field[i, j]
                row_text.append(f"Position: ({i}, {j})<br>Stress: {val:.4f} GPa")
            hover_text.append(row_text)
        
        # Get color scale
        if cmap_name not in px.colors.named_colorscales():
            cmap_name = 'viridis'  # Default
        
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
                titlefont=dict(size=14, family='Arial'),
                thickness=20,
                len=0.8
            )
        )
        
        # Create figure
        fig = go.Figure(data=[heatmap_trace])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95
            ),
            width=width,
            height=height,
            xaxis=dict(
                title="X Position",
                gridcolor='rgba(100, 100, 100, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                title="Y Position",
                gridcolor='rgba(100, 100, 100, 0.3)',
                showgrid=True
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_comparison_heatmaps(self, stress_fields_dict, cmap_name='viridis',
                                  figsize=(18, 12), titles=None):
        """Create comparison heatmaps for multiple stress components"""
        
        n_components = len(stress_fields_dict)
        
        if n_components == 0:
            # Create empty figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
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
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')
        
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

# =============================================
# RESULTS MANAGER WITH EXPORT
# =============================================

class ResultsManager:
    """Manager for exporting interpolation results"""
    
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        """Prepare data for export"""
        try:
            result = interpolation_result.copy()
            export_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'interpolation_method': result.get('metadata', {}).get('interpolation_method', 'transformer_spatial'),
                    'visualization_params': visualization_params,
                    'target_angle': result.get('target_angle', 0),
                    'target_params': result.get('target_params', {}),
                    'field_shape': result.get('shape', (0, 0))
                },
                'statistics': result.get('statistics', {}),
                'weights': result.get('weights', {}),
                'source_info': result.get('metadata', {})
            }
            
            # Convert numpy arrays to lists for JSON serialization
            for field_name, field_data in result.get('fields', {}).items():
                if hasattr(field_data, 'tolist'):
                    export_data[f'{field_name}_data'] = field_data.tolist()
                else:
                    export_data[f'{field_name}_data'] = field_data
            
            return export_data
        except Exception as e:
            st.error(f"Error preparing export data: {e}")
            return None
    
    def export_to_json(self, export_data, filename=None):
        """Export results to JSON file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                theta = export_data['metadata']['target_angle']
                defect = export_data['metadata']['target_params'].get('defect_type', 'Unknown')
                filename = f"transformer_interpolation_theta_{theta}_{defect}_{timestamp}.json"
            
            json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
            return json_str, filename
        except Exception as e:
            st.error(f"Error exporting to JSON: {e}")
            return None, None
    
    def export_to_csv(self, interpolation_result, filename=None):
        """Export flattened field data to CSV"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                theta = interpolation_result.get('target_angle', 0)
                defect = interpolation_result.get('target_params', {}).get('defect_type', 'Unknown')
                filename = f"stress_fields_theta_{theta}_{defect}_{timestamp}.csv"
            
            # Create DataFrame with flattened data
            data_dict = {}
            for field_name, field_data in interpolation_result.get('fields', {}).items():
                data_dict[field_name] = field_data.flatten()
            
            df = pd.DataFrame(data_dict)
            csv_str = df.to_csv(index=False)
            return csv_str, filename
        except Exception as e:
            st.error(f"Error exporting to CSV: {e}")
            return None, None
    
    def create_zip_export(self, interpolation_result, visualization_params, include_plots=True):
        """Create comprehensive ZIP export"""
        try:
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # 1. Export metadata and results as JSON
                export_data = self.prepare_export_data(interpolation_result, visualization_params)
                if export_data:
                    json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
                    zip_file.writestr("results.json", json_str)
                
                # 2. Export field data as CSV
                csv_str, _ = self.export_to_csv(interpolation_result)
                if csv_str:
                    zip_file.writestr("stress_fields.csv", csv_str)
                
                # 3. Export README
                readme = self._create_readme(interpolation_result)
                zip_file.writestr("README.txt", readme)
                
                # 4. Export plots if requested
                if include_plots and 'fields' in interpolation_result:
                    import tempfile
                    import os
                    
                    # Create temporary directory for plots
                    with tempfile.TemporaryDirectory() as tmpdir:
                        visualizer = EnhancedHeatMapVisualizer()
                        
                        # Export each component
                        for component, field in interpolation_result['fields'].items():
                            fig = visualizer.create_stress_heatmap(
                                field,
                                title=f"{component.replace('_', ' ').title()}",
                                cmap_name='viridis'
                            )
                            
                            plot_path = os.path.join(tmpdir, f"{component}.png")
                            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                            plt.close(fig)
                            
                            # Add to zip
                            with open(plot_path, 'rb') as f:
                                zip_file.writestr(f"plots/{component}.png", f.read())
            
            zip_buffer.seek(0)
            return zip_buffer
            
        except Exception as e:
            st.error(f"Error creating ZIP export: {e}")
            return None
    
    def _create_readme(self, interpolation_result):
        """Create README file"""
        readme = f"""# Stress Field Interpolation Results

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Target Parameters
- Angle: {interpolation_result.get('target_angle', 'N/A')}°
- Defect Type: {interpolation_result.get('target_params', {}).get('defect_type', 'N/A')}
- Eigen Strain: {interpolation_result.get('target_params', {}).get('eps0', 'N/A')}
- Shape: {interpolation_result.get('target_params', {}).get('shape', 'N/A')}

## Field Information
- Shape: {interpolation_result.get('shape', (0, 0))}
- Number of Sources: {interpolation_result.get('metadata', {}).get('num_sources', 0)}
- Interpolation Method: {interpolation_result.get('metadata', {}).get('interpolation_method', 'N/A')}

## Files Included
1. results.json - Complete results with metadata
2. stress_fields.csv - Flattened stress field data
3. README.txt - This file
4. plots/ - Directory with visualization plots

## Data Structure
- Von Mises Stress: Equivalent tensile stress
- Hydrostatic Stress: Mean normal stress (σ₁+σ₂+σ₃)/3
- Stress Magnitude: Overall stress intensity

## Usage
The CSV file contains flattened arrays of each stress component.
Use numpy.reshape(data, {interpolation_result.get('shape', (0, 0))}) to reconstruct 2D fields.
"""
        return readme
    
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
# MAIN APPLICATION
# =============================================

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Robust Transformer Stress Interpolation",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem !important;
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
        font-size: 1.6rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px 5px 0 0;
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
    st.markdown('<h1 class="main-header">🤖 Robust Transformer Stress Field Interpolation</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Advanced stress interpolation using transformer architecture with spatial locality regularization.</strong><br>
    • Load simulation files from multiple formats<br>
    • Robust error handling and fallback mechanisms<br>
    • Interpolate stress fields at custom polar angles (default: 54.7°)<br>
    • Visualize von Mises, hydrostatic, and stress magnitude fields<br>
    • Choose from 50+ colormaps including jet, turbo, rainbow, inferno<br>
    • Comprehensive export options
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        # Initialize with correct feature dimension (20 based on extract_features)
        st.session_state.transformer_interpolator = RobustTransformerInterpolator(
            d_model=64,
            nhead=8,
            num_layers=3,
            spatial_sigma=0.2,
            temperature=1.0,
            feature_dim=20,
            dropout=0.1
        )
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = EnhancedHeatMapVisualizer()
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
            if st.button("🔄 Load Solutions", use_container_width=True, type="primary"):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions(show_progress=True)
                    if st.session_state.solutions:
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                    else:
                        st.warning("No valid solutions found in directory")
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.success("Cache cleared")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📊 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                valid_count = sum(1 for s in st.session_state.solutions if s['metadata']['format_checked'])
                st.write(f"**Valid solutions:** {valid_count}/{len(st.session_state.solutions)}")
                
                # Show defect type distribution
                defect_counts = {}
                for sol in st.session_state.solutions:
                    if sol['metadata']['format_checked']:
                        defect = sol['params'].get('defect_type', 'Unknown')
                        defect_counts[defect] = defect_counts.get(defect, 0) + 1
                
                if defect_counts:
                    st.write("**Defect types:**")
                    for defect, count in defect_counts.items():
                        st.write(f"- {defect}: {count}")
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle
        custom_theta = st.number_input(
            "Custom Polar Angle θ (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=54.7,
            step=0.1,
            help="Set custom polar angle for interpolation (default: 54.7°)",
            key="target_angle"
        )
        
        # Defect type
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect", "Unknown"],
            index=2,
            help="Select the defect type for interpolation",
            key="defect_type"
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0, "Unknown": 1.0}
        eps0 = eigen_strains.get(defect_type, 1.0)
        
        # Allow manual override
        eps0 = st.number_input(
            "Eigen Strain (ε*)",
            min_value=0.0,
            max_value=5.0,
            value=eps0,
            step=0.01,
            help="Eigen strain value",
            key="eps0"
        )
        
        # Shape
        shape = st.selectbox(
            "Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Other"],
            index=0,
            key="shape"
        )
        
        # Kappa parameter
        kappa = st.number_input(
            "Interface Energy (κ)",
            min_value=0.1,
            max_value=5.0,
            value=0.6,
            step=0.01,
            help="Interface energy parameter",
            key="kappa"
        )
        
        # Transformer parameters
        st.markdown('<h2 class="section-header">🤖 Transformer Parameters</h2>', unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            spatial_sigma = st.slider(
                "Spatial Sigma",
                min_value=0.01,
                max_value=1.0,
                value=0.2,
                step=0.01,
                help="Spatial locality regularization parameter",
                key="spatial_sigma"
            )
        
        with col_t2:
            attention_temp = st.slider(
                "Attention Temperature",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Temperature for attention scaling",
                key="attention_temp"
            )
        
        # Advanced parameters
        with st.expander("⚙️ Advanced Parameters", expanded=False):
            d_model = st.slider(
                "Model Dimension",
                min_value=16,
                max_value=256,
                value=64,
                step=16,
                help="Transformer model dimension"
            )
            
            nhead = st.slider(
                "Attention Heads",
                min_value=1,
                max_value=16,
                value=8,
                step=1,
                help="Number of attention heads"
            )
            
            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Dropout rate for regularization"
            )
        
        # Visualization parameters
        st.markdown('<h2 class="section-header">🎨 Visualization</h2>', unsafe_allow_html=True)
        
        colormap_category = st.selectbox(
            "Colormap Category",
            list(COLORMAP_OPTIONS.keys()),
            index=0,
            key="colormap_category"
        )
        
        colormap_name = st.selectbox(
            "Select Colormap",
            COLORMAP_OPTIONS[colormap_category],
            index=0,
            key="colormap_name"
        )
        
        visualization_type = st.selectbox(
            "Visualization Type",
            ["2D Heatmap", "Interactive Heatmap", "3D Surface", "Comparison View", "Dashboard"],
            index=0,
            key="visualization_type"
        )
        
        # Interpolation button
        st.markdown("---")
        if st.button("🚀 Perform Transformer Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("❌ Please load solutions first!")
            else:
                # Count valid solutions
                valid_solutions = [s for s in st.session_state.solutions if s['metadata']['format_checked']]
                
                if len(valid_solutions) == 0:
                    st.error("❌ No valid solutions found. Check your data format.")
                else:
                    # Update transformer parameters
                    st.session_state.transformer_interpolator.spatial_sigma = spatial_sigma
                    st.session_state.transformer_interpolator.temperature = attention_temp
                    st.session_state.transformer_interpolator.d_model = d_model
                    st.session_state.transformer_interpolator.nhead = nhead
                    
                    # Prepare target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': eps0,
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
                    }
                    
                    # Perform interpolation
                    with st.spinner(f"Performing transformer interpolation with {len(valid_solutions)} sources..."):
                        try:
                            result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                                valid_solutions,
                                custom_theta,
                                target_params
                            )
                            
                            if result:
                                st.session_state.interpolation_result = result
                                st.success(f"✅ Successfully interpolated stress fields at θ={custom_theta:.1f}°")
                                st.info(f"Method: {result.get('metadata', {}).get('interpolation_method', 'N/A')}")
                            else:
                                st.error("❌ Interpolation failed. Check the error messages above.")
                        except Exception as e:
                            st.error(f"❌ Fatal error during interpolation: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
    
    # Main content
    if not st.session_state.solutions:
        st.markdown('<h2 class="section-header">📁 Data Loading</h2>', unsafe_allow_html=True)
        
        # Directory information
        with st.expander("📁 Directory Information", expanded=True):
            st.info(f"**Solutions Directory:** `{SOLUTIONS_DIR}`")
            
            # Check if directory exists
            if os.path.exists(SOLUTIONS_DIR):
                # List files in directory
                import glob
                files = glob.glob(os.path.join(SOLUTIONS_DIR, "*"))
                if files:
                    st.write("**Files found:**")
                    for f in files[:10]:  # Show first 10
                        st.write(f"- `{os.path.basename(f)}`")
                    if len(files) > 10:
                        st.write(f"... and {len(files) - 10} more files")
                else:
                    st.warning("No files found in directory")
            else:
                st.warning(f"Directory does not exist. It will be created when you load solutions.")
            
            st.markdown("""
            **Supported file formats:**
            - `.pkl`, `.pickle` - Python pickle files
            - `.pt`, `.pth` - PyTorch model files
            - `.npz` - NumPy compressed arrays
            - `.h5`, `.hdf5` - HDF5 files
            
            **Expected data structure:**
            Each file should contain a dictionary with:
            - `params`: Dictionary of simulation parameters (defect_type, eps0, theta, etc.)
            - `history`: List of simulation frames (each with stress fields)
            - `stresses`: Direct stress field data (alternative)
            """)
        
        # Quick guide
        st.markdown('<h2 class="section-header">🚀 Quick Start Guide</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        1. **Prepare Data**: Place your simulation files in the `numerical_solutions` directory
        2. **Load Solutions**: Click the "Load Solutions" button in the sidebar
        3. **Set Parameters**: Configure target angle and defect type
        4. **Perform Interpolation**: Click "Perform Transformer Interpolation"
        5. **Visualize Results**: Choose visualization type and colormap
        
        ### 🔬 Key Features
        
        #### Robust Transformer Architecture
        - Multi-head attention with spatial locality regularization
        - Automatic fallback mechanisms for error recovery
        - Feature dimension matching to prevent matrix errors
        
        #### Stress Components
        - **Von Mises Stress (σ_vm)**: Equivalent tensile stress
        - **Hydrostatic Stress (σ_h)**: Mean normal stress (trace/3)
        - **Stress Magnitude (σ_mag)**: Overall stress intensity
        
        #### Visualization Options
        - 50+ colormaps including jet, turbo, rainbow, inferno
        - 2D heatmaps, 3D surfaces, interactive plots
        - Comprehensive dashboards with statistics
        - Multiple export formats
        
        #### Error Handling
        - Graceful degradation on missing data
        - Detailed error messages for debugging
        - Fallback to spatial-only interpolation when needed
        """)
    else:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Results",
            "📈 Visualization",
            "🔍 Analysis",
            "⚖️ Weights",
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
                             f"{hydro_stats['max']:.3f}/{hydro_stats['min']:.3f} GPa",
                             f"Mean: {hydro_stats['mean']:.3f} GPa")
                
                with col3:
                    mag_stats = result['statistics']['sigma_mag']
                    st.metric("Stress Magnitude Max", f"{mag_stats['max']:.3f} GPa",
                             f"Mean: {mag_stats['mean']:.3f} GPa")
                
                with col4:
                    st.metric("Field Shape", f"{result['shape'][0]}×{result['shape'][1]}",
                             f"θ={result['target_angle']:.1f}°")
                
                # Display metadata
                with st.expander("🔧 Interpolation Metadata", expanded=True):
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.write("**Target Parameters:**")
                        for key, value in result['target_params'].items():
                            if key == 'theta':
                                st.write(f"- {key}: {np.degrees(value):.2f}°")
                            else:
                                st.write(f"- {key}: {value}")
                        
                        st.write("**Interpolation Settings:**")
                        st.write(f"- Spatial Sigma: {spatial_sigma}")
                        st.write(f"- Attention Temperature: {attention_temp}")
                    
                    with col_m2:
                        st.write("**Interpolation Info:**")
                        metadata = result.get('metadata', {})
                        st.write(f"- Method: {metadata.get('interpolation_method', 'N/A')}")
                        st.write(f"- Sources: {metadata.get('num_sources', 'N/A')}")
                        st.write(f"- Valid Sources: {metadata.get('num_valid_sources', 'N/A')}")
                        
                        # Weight information
                        if 'weights' in result:
                            weights = result['weights']
                            if 'combined' in weights:
                                st.write(f"- Weight Entropy: {self._calculate_entropy(weights['combined']):.3f}")
                
                # Quick preview
                st.markdown("#### 👀 Quick Field Preview")
                
                # Create a quick preview figure
                if 'fields' in result:
                    fig_preview, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                    titles = ['Von Mises', 'Hydrostatic', 'Magnitude']
                    
                    for idx, (comp, title) in enumerate(zip(components, titles)):
                        if comp in result['fields']:
                            ax = axes[idx]
                            field = result['fields'][comp]
                            im = ax.imshow(field, cmap='viridis', aspect='auto')
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
                
                # Show available data statistics
                valid_solutions = [s for s in st.session_state.solutions if s['metadata']['format_checked']]
                
                if valid_solutions:
                    st.markdown("#### 📋 Available Data")
                    
                    col_d1, col_d2, col_d3 = st.columns(3)
                    
                    with col_d1:
                        st.metric("Valid Solutions", len(valid_solutions))
                    
                    with col_d2:
                        # Count solutions with stress data
                        has_stress = sum(1 for s in valid_solutions if s['metadata'].get('has_stresses', False))
                        st.metric("With Stress Data", has_stress)
                    
                    with col_d3:
                        # Average field size
                        sizes = []
                        for sol in valid_solutions[:10]:  # Sample first 10
                            if sol['history']:
                                last_frame = sol['history'][-1]
                                if isinstance(last_frame, dict) and 'stresses' in last_frame:
                                    for field in last_frame['stresses'].values():
                                        if hasattr(field, 'shape'):
                                            sizes.append(field.shape)
                        if sizes:
                            avg_size = np.mean(sizes, axis=0).astype(int)
                            st.metric("Avg Field Size", f"{avg_size[0]}×{avg_size[1]}")
                        else:
                            st.metric("Field Size", "Unknown")
        
        with tab2:
            st.markdown('<h2 class="section-header">📈 Stress Field Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                if 'fields' not in result or not result['fields']:
                    st.error("No field data available for visualization")
                else:
                    # Component selection
                    stress_component = st.selectbox(
                        "Select Stress Component",
                        ["von_mises", "sigma_hydro", "sigma_mag"],
                        index=0,
                        key="viz_component"
                    )
                    
                    if stress_component not in result['fields']:
                        st.error(f"Component '{stress_component}' not found in results")
                    else:
                        field = result['fields'][stress_component]
                        
                        # Component names for display
                        component_names = {
                            'von_mises': 'Von Mises Stress',
                            'sigma_hydro': 'Hydrostatic Stress',
                            'sigma_mag': 'Stress Magnitude'
                        }
                        
                        # Create visualization based on selected type
                        if visualization_type == "2D Heatmap":
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                field,
                                title=f"{component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                                cmap_name=colormap_name,
                                colorbar_label=f"{component_names[stress_component]} (GPa)"
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
                                field,
                                title=f"{component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                                cmap_name=colormap_name
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif visualization_type == "3D Surface":
                            fig = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                                field,
                                title=f"3D {component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                                cmap_name=colormap_name
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        elif visualization_type == "Comparison View":
                            comparison_fields = {}
                            titles = []
                            
                            for comp in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                                if comp in result['fields']:
                                    comparison_fields[component_names[comp]] = result['fields'][comp]
                                    titles.append(component_names[comp])
                            
                            if comparison_fields:
                                fig = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                                    comparison_fields,
                                    cmap_name=colormap_name,
                                    titles=titles
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        elif visualization_type == "Dashboard":
                            # Create a comprehensive dashboard
                            col_dash1, col_dash2 = st.columns([2, 1])
                            
                            with col_dash1:
                                # Main heatmap
                                fig_main = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                    field,
                                    title=f"{component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                                    cmap_name=colormap_name,
                                    figsize=(10, 8)
                                )
                                st.pyplot(fig_main)
                                plt.close(fig_main)
                            
                            with col_dash2:
                                # Statistics panel
                                stats = result['statistics'][stress_component]
                                
                                st.markdown("#### 📊 Statistics")
                                st.metric("Maximum", f"{stats['max']:.3f} GPa")
                                st.metric("Minimum", f"{stats['min']:.3f} GPa")
                                st.metric("Mean", f"{stats['mean']:.3f} GPa")
                                st.metric("Std Dev", f"{stats['std']:.3f} GPa")
                                st.metric("Range", f"{stats['range']:.3f} GPa")
                                st.metric("Q25-Q75", f"{stats['q25']:.3f}-{stats['q75']:.3f} GPa")
                                
                                # Histogram
                                fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
                                ax_hist.hist(field.flatten(), bins=50, alpha=0.7, color='steelblue')
                                ax_hist.set_xlabel(f'{component_names[stress_component]} (GPa)')
                                ax_hist.set_ylabel('Frequency')
                                ax_hist.set_title('Field Distribution')
                                ax_hist.grid(True, alpha=0.3)
                                st.pyplot(fig_hist)
                                plt.close(fig_hist)
                        
                        # Additional controls
                        with st.expander("⚙️ Visualization Controls", expanded=False):
                            col_ctrl1, col_ctrl2 = st.columns(2)
                            
                            with col_ctrl1:
                                auto_range = st.checkbox("Auto Range", value=True)
                                
                                if not auto_range:
                                    vmin = st.number_input("Min Value", value=float(np.min(field)))
                                    vmax = st.number_input("Max Value", value=float(np.max(field)))
                                else:
                                    vmin = None
                                    vmax = None
                            
                            with col_ctrl2:
                                interpolation = st.selectbox(
                                    "Interpolation",
                                    ['bilinear', 'nearest', 'bicubic', 'spline16'],
                                    index=0
                                )
                                
                                show_grid = st.checkbox("Show Grid", value=True)
                            
                            # Update visualization with custom settings
                            if st.button("Update Visualization"):
                                fig_custom = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                    field,
                                    title=f"{component_names[stress_component]} at θ={result['target_angle']:.1f}°",
                                    cmap_name=colormap_name,
                                    vmin=vmin,
                                    vmax=vmax,
                                    show_grid=show_grid
                                )
                                st.pyplot(fig_custom)
                                plt.close(fig_custom)
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">🔍 Field Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                if 'fields' not in result:
                    st.error("No field data available for analysis")
                else:
                    # Detailed statistics
                    st.markdown("#### 📊 Detailed Statistics")
                    
                    # Create statistics table
                    stats_data = []
                    for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        if component in result['statistics']:
                            comp_stats = result['statistics'][component]
                            stats_data.append({
                                'Component': component.replace('_', ' ').title(),
                                'Max (GPa)': f"{comp_stats['max']:.4f}",
                                'Min (GPa)': f"{comp_stats['min']:.4f}",
                                'Mean (GPa)': f"{comp_stats['mean']:.4f}",
                                'Std (GPa)': f"{comp_stats['std']:.4f}",
                                'Range (GPa)': f"{comp_stats['range']:.4f}",
                                'Median (GPa)': f"{comp_stats['median']:.4f}"
                            })
                    
                    if stats_data:
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True)
                    
                    # Field properties analysis
                    st.markdown("#### 🔬 Field Properties")
                    
                    col_prop1, col_prop2 = st.columns(2)
                    
                    with col_prop1:
                        # Calculate field gradients
                        if 'von_mises' in result['fields']:
                            field = result['fields']['von_mises']
                            
                            # Compute gradients
                            grad_x, grad_y = np.gradient(field)
                            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                            
                            st.metric("Max Gradient", f"{np.max(grad_magnitude):.4f} GPa/px")
                            st.metric("Mean Gradient", f"{np.mean(grad_magnitude):.4f} GPa/px")
                            
                            # Compute Laplacian (curvature)
                            laplacian = np.gradient(grad_x, axis=0) + np.gradient(grad_y, axis=1)
                            st.metric("Max Curvature", f"{np.max(np.abs(laplacian)):.6f}")
                    
                    with col_prop2:
                        # Calculate spatial statistics
                        if 'von_mises' in result['fields']:
                            field = result['fields']['von_mises']
                            
                            # Spatial autocorrelation
                            from scipy import signal
                            autocorr = signal.correlate2d(field, field, mode='same')
                            autocorr_norm = autocorr / np.max(autocorr)
                            
                            st.metric("Autocorrelation Peak", f"{np.max(autocorr_norm):.3f}")
                            
                            # Field uniformity
                            uniformity = 1 - (np.std(field) / np.mean(field)) if np.mean(field) != 0 else 0
                            st.metric("Uniformity Index", f"{uniformity:.3f}")
                            
                            # Hotspot analysis
                            hotspot_threshold = np.mean(field) + 2 * np.std(field)
                            hotspot_area = np.sum(field > hotspot_threshold) / field.size
                            st.metric("Hotspot Area", f"{hotspot_area:.2%}")
                    
                    # Advanced analysis
                    with st.expander("📈 Advanced Analysis", expanded=False):
                        # Fourier analysis
                        if 'von_mises' in result['fields']:
                            field = result['fields']['von_mises']
                            
                            # Compute FFT
                            fft_result = np.fft.fft2(field)
                            fft_magnitude = np.abs(np.fft.fftshift(fft_result))
                            
                            # Plot FFT
                            fig_fft, ax_fft = plt.subplots(1, 2, figsize=(12, 5))
                            
                            ax_fft[0].imshow(field, cmap='viridis')
                            ax_fft[0].set_title('Original Field')
                            ax_fft[0].set_xlabel('X')
                            ax_fft[0].set_ylabel('Y')
                            
                            ax_fft[1].imshow(np.log1p(fft_magnitude), cmap='hot')
                            ax_fft[1].set_title('Fourier Transform (log scale)')
                            ax_fft[1].set_xlabel('Frequency X')
                            ax_fft[1].set_ylabel('Frequency Y')
                            
                            plt.tight_layout()
                            st.pyplot(fig_fft)
                            plt.close(fig_fft)
                            
                            # Dominant frequencies
                            dominant_freq = np.unravel_index(np.argmax(fft_magnitude), fft_magnitude.shape)
                            center = np.array(fft_magnitude.shape) // 2
                            freq_distance = np.sqrt(np.sum((np.array(dominant_freq) - center)**2))
                            
                            st.write(f"**Dominant frequency distance from center:** {freq_distance:.2f}")
                            
                            # Energy in low frequencies
                            low_freq_mask = np.zeros_like(fft_magnitude)
                            radius = min(field.shape) // 4
                            y, x = np.ogrid[:fft_magnitude.shape[0], :fft_magnitude.shape[1]]
                            mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
                            low_freq_energy = np.sum(fft_magnitude[mask]) / np.sum(fft_magnitude)
                            
                            st.write(f"**Energy in low frequencies:** {low_freq_energy:.2%}")
            else:
                st.info("No interpolation results available for analysis.")
        
        with tab4:
            st.markdown('<h2 class="section-header">⚖️ Weight Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                if 'weights' not in result:
                    st.warning("No weight information available")
                else:
                    weights = result['weights']
                    
                    # Display weight statistics
                    col_w1, col_w2, col_w3 = st.columns(3)
                    
                    with col_w1:
                        if 'transformer' in weights and weights['transformer']:
                            entropy = self._calculate_entropy(weights['transformer'])
                            st.metric("Transformer Entropy", f"{entropy:.3f}")
                        else:
                            st.metric("Transformer Entropy", "N/A")
                    
                    with col_w2:
                        if 'spatial' in weights and weights['spatial']:
                            entropy = self._calculate_entropy(weights['spatial'])
                            st.metric("Spatial Entropy", f"{entropy:.3f}")
                        else:
                            st.metric("Spatial Entropy", "N/A")
                    
                    with col_w3:
                        if 'combined' in weights and weights['combined']:
                            entropy = self._calculate_entropy(weights['combined'])
                            st.metric("Combined Entropy", f"{entropy:.3f}")
                        else:
                            st.metric("Combined Entropy", "N/A")
                    
                    # Weight visualization
                    st.markdown("#### 📊 Weight Distributions")
                    
                    fig_weights, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    weight_types = ['transformer', 'spatial', 'combined']
                    titles = ['Transformer Weights', 'Spatial Weights', 'Combined Weights']
                    
                    for idx, (wtype, title) in enumerate(zip(weight_types, titles)):
                        ax = axes[idx]
                        if wtype in weights and weights[wtype]:
                            wdata = weights[wtype]
                            ax.bar(range(len(wdata)), wdata)
                            ax.set_xlabel('Source Index')
                            ax.set_ylabel('Weight')
                            ax.set_title(title)
                            ax.grid(True, alpha=0.3)
                            
                            # Highlight top 3 weights
                            if len(wdata) >= 3:
                                top_indices = np.argsort(wdata)[-3:][::-1]
                                for i, top_idx in enumerate(top_indices):
                                    ax.bar(top_idx, wdata[top_idx], color='red' if i == 0 else 'orange')
                        else:
                            ax.text(0.5, 0.5, f"No {wtype} weights", 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(title)
                    
                    plt.tight_layout()
                    st.pyplot(fig_weights)
                    plt.close(fig_weights)
                    
                    # Top contributors
                    st.markdown("#### 🏆 Top Contributing Sources")
                    
                    if 'combined' in weights and weights['combined']:
                        combined_weights = np.array(weights['combined'])
                        top_indices = np.argsort(combined_weights)[-5:][::-1]
                        
                        for i, idx in enumerate(top_indices):
                            col_tc1, col_tc2, col_tc3 = st.columns([1, 2, 1])
                            
                            with col_tc1:
                                st.write(f"**#{i+1}**")
                            
                            with col_tc2:
                                weight_info = []
                                if 'transformer' in weights:
                                    weight_info.append(f"Trans: {weights['transformer'][idx]:.3f}")
                                if 'spatial' in weights:
                                    weight_info.append(f"Spatial: {weights['spatial'][idx]:.3f}")
                                if 'combined' in weights:
                                    weight_info.append(f"**Combined: {weights['combined'][idx]:.3f}**")
                                
                                st.write(f"Source {idx}: {' | '.join(weight_info)}")
                            
                            with col_tc3:
                                # Show contribution percentage
                                if combined_weights.sum() > 0:
                                    percentage = (combined_weights[idx] / combined_weights.sum()) * 100
                                    st.write(f"**{percentage:.1f}%**")
                    
                    # Weight correlation analysis
                    with st.expander("🔗 Weight Correlations", expanded=False):
                        if 'transformer' in weights and 'spatial' in weights:
                            trans_weights = np.array(weights['transformer'])
                            spatial_weights = np.array(weights['spatial'])
                            
                            if len(trans_weights) == len(spatial_weights) and len(trans_weights) > 1:
                                # Calculate correlation
                                correlation = np.corrcoef(trans_weights, spatial_weights)[0, 1]
                                st.write(f"**Correlation between transformer and spatial weights:** {correlation:.3f}")
                                
                                # Scatter plot
                                fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                                ax_scatter.scatter(trans_weights, spatial_weights, alpha=0.6)
                                ax_scatter.set_xlabel('Transformer Weights')
                                ax_scatter.set_ylabel('Spatial Weights')
                                ax_scatter.set_title('Weight Correlation')
                                ax_scatter.grid(True, alpha=0.3)
                                
                                # Add regression line
                                if len(trans_weights) > 1:
                                    z = np.polyfit(trans_weights, spatial_weights, 1)
                                    p = np.poly1d(z)
                                    ax_scatter.plot(trans_weights, p(trans_weights), "r--", alpha=0.8)
                                
                                st.pyplot(fig_scatter)
                                plt.close(fig_scatter)
            else:
                st.info("No interpolation results available for weight analysis.")
        
        with tab5:
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
                            'visualization_type': visualization_type,
                            'target_angle': custom_theta
                        }
                        export_data = st.session_state.results_manager.prepare_export_data(result, visualization_params)
                        
                        if export_data:
                            json_str, filename = st.session_state.results_manager.export_to_json(export_data)
                            
                            if json_str:
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_str,
                                    file_name=filename,
                                    mime="application/json",
                                    use_container_width=True
                                )
                
                with col_e2:
                    # Export as CSV
                    if st.button("📊 Export as CSV", use_container_width=True, key="export_csv"):
                        csv_str, filename = st.session_state.results_manager.export_to_csv(result)
                        
                        if csv_str:
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_str,
                                file_name=filename,
                                mime="text/csv",
                                use_container_width=True
                            )
                
                with col_e3:
                    # Export plot as PNG
                    if st.button("🖼️ Export Plot", use_container_width=True, key="export_plot"):
                        # Create a figure to export
                        if 'von_mises' in result['fields']:
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
                                use_container_width=True
                            )
                            plt.close(fig_export)
                
                # Bulk export
                st.markdown("---")
                st.markdown("#### 📦 Comprehensive Export")
                
                if st.button("🚀 Export Complete Package", use_container_width=True, type="secondary"):
                    with st.spinner("Creating export package..."):
                        visualization_params = {
                            'colormap': colormap_name,
                            'visualization_type': visualization_type
                        }
                        
                        zip_buffer = st.session_state.results_manager.create_zip_export(
                            result, 
                            visualization_params,
                            include_plots=True
                        )
                        
                        if zip_buffer:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"stress_interpolation_package_{timestamp}.zip"
                            
                            st.download_button(
                                label="📥 Download ZIP Package",
                                data=zip_buffer.getvalue(),
                                file_name=filename,
                                mime="application/zip",
                                use_container_width=True
                            )
                        else:
                            st.error("Failed to create export package")
                
                # Export statistics
                with st.expander("📈 Export Statistics", expanded=False):
                    # Create statistics table
                    if 'statistics' in result:
                        stats_list = []
                        for component, comp_stats in result['statistics'].items():
                            stats_list.append({
                                'Component': component.replace('_', ' ').title(),
                                'Max_GPa': comp_stats['max'],
                                'Min_GPa': comp_stats['min'],
                                'Mean_GPa': comp_stats['mean'],
                                'Std_GPa': comp_stats['std'],
                                'Range_GPa': comp_stats['range'],
                                'Median_GPa': comp_stats['median']
                            })
                        
                        if stats_list:
                            df_stats = pd.DataFrame(stats_list)
                            st.dataframe(df_stats, use_container_width=True)
                            
                            # Export statistics as CSV
                            csv_stats = df_stats.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Statistics CSV",
                                data=csv_stats,
                                file_name=f"statistics_theta_{result['target_angle']:.1f}.csv",
                                mime="text/csv"
                            )
            else:
                st.info("No interpolation results available for export.")
    
    # Helper function for entropy calculation
    def _calculate_entropy(weights):
        """Calculate entropy of weight distribution"""
        try:
            weights = np.array(weights)
            weights = weights[weights > 0]  # Remove zeros
            if len(weights) == 0:
                return 0.0
            weights = weights / weights.sum()
            return -np.sum(weights * np.log(weights))
        except:
            return 0.0

# Run the application
if __name__ == "__main__":
    main()
