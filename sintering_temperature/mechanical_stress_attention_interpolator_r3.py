import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import warnings
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sqlite3
from io import StringIO
import traceback
import h5py
import msgpack
import dill
import joblib
from pathlib import Path
import tempfile
import base64
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
from collections import OrderedDict

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")

if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ENHANCED DATA LOADING AND STANDARDIZATION
# =============================================
class EnhancedDataLoader:
    """Enhanced data loader with better error handling and format support"""
    
    @staticmethod
    def detect_file_format(file_path):
        """Detect file format from extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.pkl', '.pickle']:
            return 'pkl'
        elif ext in ['.pt', '.pth']:
            return 'pt'
        elif ext in ['.h5', '.hdf5']:
            return 'h5'
        elif ext == '.npz':
            return 'npz'
        elif ext == '.json':
            return 'json'
        elif ext in ['.npy']:
            return 'npy'
        else:
            return 'unknown'
    
    @staticmethod
    def load_file(file_path):
        """Load file with multiple format support"""
        format_type = EnhancedDataLoader.detect_file_format(file_path)
        
        try:
            if format_type == 'pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif format_type == 'pt':
                return torch.load(file_path, map_location='cpu')
            elif format_type == 'h5':
                data = {}
                with h5py.File(file_path, 'r') as f:
                    def load_item(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            data[name] = obj[()]
                    f.visititems(load_item)
                return data
            elif format_type == 'npz':
                return dict(np.load(file_path, allow_pickle=True))
            elif format_type == 'npy':
                return np.load(file_path, allow_pickle=True).item()
            elif format_type == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                # Try to load as pickle as fallback
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except:
                    raise ValueError(f"Unsupported file format: {format_type}")
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def standardize_data(raw_data, file_path):
        """Standardize data to common format with robust error handling"""
        
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'loaded_successfully': False,
            'error': None
        }
        
        try:
            if raw_data is None:
                standardized['error'] = "Raw data is None"
                return standardized
            
            # Handle different data structures
            data = raw_data
            
            # Case 1: Direct dictionary with expected structure
            if isinstance(data, dict):
                # Extract parameters from various possible keys
                param_sources = ['params', 'parameters', 'simulation_params', 'config', 
                                'settings', 'sim_params', 'param']
                
                for source in param_sources:
                    if source in data and isinstance(data[source], dict):
                        standardized['params'] = data[source]
                        break
                
                # If no params found, try to extract directly from root
                if not standardized['params']:
                    # Look for common parameter keys
                    param_keys = ['defect_type', 'shape', 'eps0', 'kappa', 'orientation', 
                                 'theta', 'epsilon', 'curvature', 'angle']
                    standardized['params'] = {k: data.get(k) for k in param_keys if k in data}
                
                # Extract history from various possible keys
                history_sources = ['history', 'frames', 'results', 'data', 'simulation_data',
                                  'stress_history', 'field_history']
                
                for source in history_sources:
                    if source in data:
                        if isinstance(data[source], list):
                            standardized['history'] = data[source]
                            break
                        elif isinstance(data[source], dict):
                            # Convert dict to list of frames
                            frames = []
                            for key, value in sorted(data[source].items()):
                                if isinstance(value, dict):
                                    frames.append(value)
                            standardized['history'] = frames
                            break
                
                # Extract metadata
                metadata_sources = ['metadata', 'info', 'simulation_info', 'details']
                for source in metadata_sources:
                    if source in data and isinstance(data[source], dict):
                        standardized['metadata'] = data[source]
                        break
            
            # Case 2: List or tuple (assume [params, history] or similar)
            elif isinstance(data, (list, tuple)):
                if len(data) >= 2:
                    if isinstance(data[0], dict):
                        standardized['params'] = data[0]
                    if isinstance(data[1], (list, tuple)):
                        standardized['history'] = list(data[1])
            
            # Case 3: NumPy array or similar
            elif hasattr(data, 'shape'):
                # Assume it's a stress field
                standardized['history'] = [(np.zeros_like(data), {'stress_field': data})]
                standardized['params'] = {'data_type': 'array'}
            
            # Ensure history is in correct format
            if standardized['history']:
                converted_history = []
                for frame in standardized['history']:
                    if isinstance(frame, tuple) and len(frame) == 2:
                        # Already in (eta, stress_fields) format
                        converted_history.append(frame)
                    elif isinstance(frame, dict):
                        # Convert dict to (eta, stress_fields) format
                        eta = frame.get('eta', frame.get('displacement', frame.get('field', 
                                 frame.get('u', frame.get('v', np.zeros((128, 128)))))))
                        stresses = {}
                        
                        # Collect all stress-like fields
                        for key, value in frame.items():
                            if key not in ['eta', 'displacement', 'field', 'u', 'v']:
                                if hasattr(value, 'shape') and len(value.shape) >= 2:
                                    stresses[key] = value
                        
                        converted_history.append((eta, stresses))
                    else:
                        # Unknown format, create placeholder
                        converted_history.append((np.zeros((128, 128)), {}))
                
                standardized['history'] = converted_history
            
            # Add default parameters if missing
            if not standardized['params']:
                standardized['params'] = {
                    'defect_type': 'Unknown',
                    'shape': 'Unknown',
                    'eps0': 1.0,
                    'kappa': 0.5,
                    'orientation': 'Unknown',
                    'theta': 0.0
                }
            
            standardized['loaded_successfully'] = True
            
        except Exception as e:
            standardized['error'] = str(e)
            standardized['loaded_successfully'] = False
        
        return standardized

# =============================================
# PARAMETER PROCESSING UTILITIES
# =============================================
class ParameterProcessor:
    """Handles parameter encoding and normalization"""
    
    @staticmethod
    def compute_parameter_vector(sim_data, param_dim=15):
        """Compute standardized parameter vector from simulation data"""
        params = sim_data.get('params', {})
        
        param_vector = []
        
        # 1. Defect type encoding (3 dimensions)
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1],
            'Void': [0.5, 0.5, 0],
            'Dislocation': [0, 0.5, 0.5],
            'Crack': [0.5, 0, 0.5]
        }
        
        defect_type = str(params.get('defect_type', 'ISF')).strip()
        param_vector.extend(defect_encoding.get(defect_type, [0.333, 0.333, 0.333]))
        
        # 2. Shape encoding (5 dimensions)
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1],
            'Circle': [0.2, 0.2, 0.2, 0.2, 0.2]
        }
        
        shape = str(params.get('shape', 'Square')).strip()
        param_vector.extend(shape_encoding.get(shape, [0.2, 0.2, 0.2, 0.2, 0.2]))
        
        # 3. Numerical parameters (3 dimensions)
        eps0 = float(params.get('eps0', params.get('epsilon', params.get('strain', 0.707))))
        kappa = float(params.get('kappa', params.get('kappa_factor', params.get('curvature', 0.6))))
        theta = float(params.get('theta', params.get('angle', params.get('orientation_angle', 0.0))))
        
        # Normalize to [0, 1]
        eps0_norm = max(0.0, min(1.0, (eps0 - 0.3) / (3.0 - 0.3)))
        kappa_norm = max(0.0, min(1.0, (kappa - 0.1) / (2.0 - 0.1)))
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        
        param_vector.extend([eps0_norm, kappa_norm, theta_norm])
        
        # 4. Orientation encoding (4 dimensions)
        orientation = str(params.get('orientation', 'Horizontal {111} (0°)')).strip()
        orientation_encoding = {
            'Horizontal {111} (0°)': [1, 0, 0, 0],
            'Tilted 30° (1¯10 projection)': [0, 1, 0, 0],
            'Tilted 60°': [0, 0, 1, 0],
            'Vertical {111} (90°)': [0, 0, 0, 1]
        }
        
        if orientation.startswith('Custom ('):
            # For custom angles, use soft encoding based on angle
            try:
                angle_str = orientation.replace('Custom (', '').replace('°)', '')
                angle = float(angle_str)
                # Encode based on angle ranges
                if angle <= 15:
                    encoding = [0.8, 0.1, 0.05, 0.05]
                elif angle <= 45:
                    encoding = [0.1, 0.8, 0.05, 0.05]
                elif angle <= 75:
                    encoding = [0.05, 0.05, 0.8, 0.1]
                else:
                    encoding = [0.05, 0.05, 0.1, 0.8]
                param_vector.extend(encoding)
            except:
                param_vector.extend([0.25, 0.25, 0.25, 0.25])
        else:
            param_vector.extend(orientation_encoding.get(orientation, [0.25, 0.25, 0.25, 0.25]))
        
        # Ensure exactly param_dim dimensions
        if len(param_vector) < param_dim:
            param_vector.extend([0.0] * (param_dim - len(param_vector)))
        elif len(param_vector) > param_dim:
            param_vector = param_vector[:param_dim]
        
        return np.array(param_vector, dtype=np.float32)
    
    @staticmethod
    def extract_stress_fields(sim_data, default_shape=(128, 128)):
        """Extract stress fields from simulation data"""
        history = sim_data.get('history', [])
        
        if not history:
            # Return zero fields
            return {
                'sigma_hydro': np.zeros(default_shape),
                'sigma_mag': np.zeros(default_shape),
                'von_mises': np.zeros(default_shape)
            }
        
        # Get the last frame
        last_frame = history[-1]
        
        if isinstance(last_frame, tuple) and len(last_frame) == 2:
            _, stress_fields = last_frame
        elif isinstance(last_frame, dict):
            stress_fields = last_frame
        else:
            stress_fields = {}
        
        # Determine actual shape from available data
        actual_shape = default_shape
        for field_name in ['sigma_hydro', 'sigma_mag', 'von_mises', 'eta', 'displacement']:
            if field_name in stress_fields:
                field_data = stress_fields[field_name]
                if hasattr(field_data, 'shape') and len(field_data.shape) >= 2:
                    actual_shape = field_data.shape[-2:]  # Get last two dimensions
                    break
        
        # Extract or create stress components
        result = {}
        
        # Hydrostatic stress
        if 'sigma_hydro' in stress_fields:
            result['sigma_hydro'] = stress_fields['sigma_hydro']
        elif 'stress_hydro' in stress_fields:
            result['sigma_hydro'] = stress_fields['stress_hydro']
        elif 'hydrostatic' in stress_fields:
            result['sigma_hydro'] = stress_fields['hydrostatic']
        else:
            result['sigma_hydro'] = np.zeros(actual_shape)
        
        # Stress magnitude
        if 'sigma_mag' in stress_fields:
            result['sigma_mag'] = stress_fields['sigma_mag']
        elif 'stress_mag' in stress_fields:
            result['sigma_mag'] = stress_fields['stress_mag']
        elif 'magnitude' in stress_fields:
            result['sigma_mag'] = stress_fields['magnitude']
        else:
            result['sigma_mag'] = np.zeros(actual_shape)
        
        # Von Mises stress
        if 'von_mises' in stress_fields:
            result['von_mises'] = stress_fields['von_mises']
        elif 'von_mises_stress' in stress_fields:
            result['von_mises'] = stress_fields['von_mises_stress']
        elif 'vm_stress' in stress_fields:
            result['von_mises'] = stress_fields['vm_stress']
        else:
            result['von_mises'] = np.zeros(actual_shape)
        
        # Ensure all are 2D arrays
        for key in result:
            if result[key].ndim > 2:
                result[key] = result[key].squeeze()
        
        return result

# =============================================
# KERNEL REGRESSION INTERPOLATOR (ENHANCED)
# =============================================
class EnhancedKernelInterpolator:
    """Enhanced kernel regression interpolator with better error handling"""
    
    def __init__(self, kernel_type='rbf', length_scale=0.3, nu=1.5, param_dim=15):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.nu = nu
        self.param_dim = param_dim
        self.source_params = None
        self.source_stress = None
        self.kernel_matrix = None
        self.is_fitted = False
        
        self.param_processor = ParameterProcessor()
    
    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix between parameter vectors"""
        if self.kernel_type == 'rbf':
            # Gaussian RBF kernel
            dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-dists / (2 * self.length_scale ** 2))
        
        elif self.kernel_type == 'matern':
            # Matern kernel
            dists = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))
            if self.nu == 0.5:
                return np.exp(-dists / self.length_scale)
            elif self.nu == 1.5:
                d_scaled = np.sqrt(3) * dists / self.length_scale
                return (1 + d_scaled) * np.exp(-d_scaled)
            else:  # nu = 2.5
                d_scaled = np.sqrt(5) * dists / self.length_scale
                return (1 + d_scaled + d_scaled**2/3) * np.exp(-d_scaled)
        
        elif self.kernel_type == 'rational_quadratic':
            # Rational quadratic kernel
            dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return (1 + dists / (2 * self.nu * self.length_scale ** 2)) ** (-self.nu)
        
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def fit(self, source_simulations):
        """Fit kernel regression model to source simulations"""
        if not source_simulations:
            raise ValueError("No source simulations provided")
        
        # Filter valid simulations
        valid_sims = []
        for sim_data in source_simulations:
            if sim_data.get('loaded_successfully', False):
                valid_sims.append(sim_data)
        
        if len(valid_sims) < 1:
            raise ValueError(f"No valid simulations found. Loaded {len(source_simulations)} but none were valid.")
        
        st.info(f"Fitting kernel regression with {len(valid_sims)} valid simulations")
        
        # Extract parameters and stress fields
        source_params = []
        source_stress = []
        
        for sim_data in valid_sims:
            # Compute parameter vector
            param_vector = self.param_processor.compute_parameter_vector(sim_data, self.param_dim)
            source_params.append(param_vector)
            
            # Extract stress fields
            stress_fields = self.param_processor.extract_stress_fields(sim_data)
            
            # Stack stress components
            stress_components = np.stack([
                stress_fields['sigma_hydro'],
                stress_fields['sigma_mag'],
                stress_fields['von_mises']
            ], axis=0)
            source_stress.append(stress_components)
        
        self.source_params = np.array(source_params)
        self.source_stress = np.array(source_stress)
        
        # Precompute kernel matrix
        if len(self.source_params) > 0:
            self.kernel_matrix = self._compute_kernel(self.source_params, self.source_params)
            # Add regularization for numerical stability
            self.kernel_matrix += np.eye(len(self.source_params)) * 1e-8
            self.is_fitted = True
        else:
            raise ValueError("Failed to extract parameters from valid simulations")
        
        return self
    
    def predict(self, target_params):
        """Predict stress field for target parameters"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Compute target parameter vector
        target_vector = self.param_processor.compute_parameter_vector(
            {'params': target_params}, self.param_dim
        )
        
        # Compute kernel weights
        k_star = self._compute_kernel(self.source_params, target_vector.reshape(1, -1))
        
        # Solve for weights
        try:
            weights = np.linalg.solve(self.kernel_matrix, k_star)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            weights = np.linalg.pinv(self.kernel_matrix) @ k_star
        
        # Normalize weights
        weights = weights.flatten()
        weights = weights / (np.sum(np.abs(weights)) + 1e-8)
        
        # Weighted combination of source stress fields
        weighted_stress = np.sum(
            self.source_stress * weights[:, np.newaxis, np.newaxis, np.newaxis], 
            axis=0
        )
        
        predicted_stress = {
            'sigma_hydro': weighted_stress[0],
            'sigma_mag': weighted_stress[1],
            'von_mises': weighted_stress[2],
            'method': 'kernel',
            'kernel_weights': weights,
            'source_count': len(self.source_params),
            'target_params': target_params
        }
        
        return predicted_stress, weights

# =============================================
# COMPLETE TRANSFORMER INTERPOLATOR
# =============================================
class TransformerInterpolator(nn.Module):
    """Complete transformer-based interpolator with full prediction capabilities"""
    
    def __init__(self, param_dim=15, stress_dim=3, d_model=64, nhead=4, 
                 num_layers=3, dim_feedforward=128, dropout=0.1, device='cpu'):
        super().__init__()
        
        self.param_dim = param_dim
        self.stress_dim = stress_dim
        self.d_model = d_model
        self.device = device
        
        # Parameter embedding
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stress encoder (convolutional)
        self.stress_encoder = nn.Sequential(
            nn.Conv2d(stress_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((8, 8))  # Reduce spatial dimensions
        )
        
        # Transformer encoder for parameter relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention between parameters and stress
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Stress decoder
        self.stress_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, stress_dim, kernel_size=4, stride=2, padding=1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 64, d_model * 128),  # 8x8=64 spatial positions
            nn.ReLU(),
            nn.Linear(d_model * 128, stress_dim * 128 * 128)
        )
        
        # Attention weight storage
        self.attention_weights = None
        
        self.to(device)
    
    def forward(self, source_params, source_stress, target_params):
        """
        Forward pass for transformer interpolation
        
        Args:
            source_params: (batch_size, num_sources, param_dim)
            source_stress: (batch_size, num_sources, stress_dim, H, W)
            target_params: (batch_size, param_dim)
        
        Returns:
            predicted_stress: (batch_size, stress_dim, H, W)
            attention_weights: Attention weights for visualization
        """
        batch_size, num_sources = source_params.shape[:2]
        
        # 1. Encode source parameters
        source_params_flat = source_params.view(-1, self.param_dim)
        source_param_emb = self.param_embed(source_params_flat)
        source_param_emb = source_param_emb.view(batch_size, num_sources, self.d_model)
        
        # 2. Encode source stress fields
        source_stress_flat = source_stress.view(-1, self.stress_dim, 128, 128)
        source_stress_emb = self.stress_encoder(source_stress_flat)
        source_stress_emb = source_stress_emb.view(batch_size, num_sources, self.d_model, 8, 8)
        source_stress_emb = source_stress_emb.flatten(3).transpose(2, 3)  # (batch, num_sources, 64, d_model)
        
        # 3. Combine parameter and stress embeddings
        source_emb = source_param_emb.unsqueeze(2) + source_stress_emb.mean(dim=2, keepdim=True)
        source_emb = source_emb.view(batch_size, num_sources * 64, self.d_model)
        
        # 4. Encode with transformer
        memory = self.transformer_encoder(source_emb)
        
        # 5. Encode target parameters
        target_param_emb = self.param_embed(target_params)  # (batch_size, d_model)
        target_queries = target_param_emb.unsqueeze(1).repeat(1, 64, 1)  # (batch_size, 64, d_model)
        
        # 6. Cross-attention: target queries attend to source memory
        attn_output, attn_weights = self.cross_attention(
            target_queries, memory, memory
        )
        self.attention_weights = attn_weights
        
        # 7. Decode to stress field
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, self.d_model, 8, 8)
        decoded_stress = self.stress_decoder(attn_output)
        
        # 8. Rescale to original size if needed
        if decoded_stress.shape[-2:] != (128, 128):
            decoded_stress = F.interpolate(decoded_stress, size=(128, 128), mode='bilinear')
        
        return decoded_stress, attn_weights
    
    def compute_loss(self, predicted_stress, target_stress):
        """Compute loss between predicted and target stress"""
        mse_loss = F.mse_loss(predicted_stress, target_stress)
        
        # Add gradient penalty for smoothness
        if predicted_stress.requires_grad:
            grad_x = torch.abs(predicted_stress[:, :, :, 1:] - predicted_stress[:, :, :, :-1])
            grad_y = torch.abs(predicted_stress[:, :, 1:, :] - predicted_stress[:, :, :-1, :])
            smoothness_loss = grad_x.mean() + grad_y.mean()
        else:
            smoothness_loss = 0.0
        
        return mse_loss + 0.01 * smoothness_loss

class TransformerModelManager:
    """Manager for transformer model training and prediction"""
    
    def __init__(self, param_dim=15, stress_dim=3, d_model=64, nhead=4, 
                 num_layers=3, device='cpu'):
        self.device = device
        self.param_dim = param_dim
        self.stress_dim = stress_dim
        
        self.model = TransformerInterpolator(
            param_dim=param_dim,
            stress_dim=stress_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            device=device
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        self.param_processor = ParameterProcessor()
        self.is_trained = False
        
    def prepare_training_data(self, source_simulations, num_sources=3):
        """Prepare training data from source simulations"""
        if len(source_simulations) < num_sources + 1:
            raise ValueError(f"Need at least {num_sources + 1} simulations for training")
        
        # Convert to parameter vectors and stress tensors
        param_vectors = []
        stress_tensors = []
        
        for sim_data in source_simulations:
            if not sim_data.get('loaded_successfully', False):
                continue
            
            param_vector = self.param_processor.compute_parameter_vector(sim_data, self.param_dim)
            stress_fields = self.param_processor.extract_stress_fields(sim_data)
            
            stress_tensor = torch.stack([
                torch.from_numpy(stress_fields['sigma_hydro']).float(),
                torch.from_numpy(stress_fields['sigma_mag']).float(),
                torch.from_numpy(stress_fields['von_mises']).float()
            ], dim=0)
            
            param_vectors.append(param_vector)
            stress_tensors.append(stress_tensor)
        
        if len(param_vectors) < num_sources + 1:
            raise ValueError("Not enough valid simulations for training")
        
        # Create training pairs using leave-one-out cross-validation
        train_pairs = []
        
        for i in range(len(param_vectors)):
            # Target is simulation i
            target_params = param_vectors[i]
            target_stress = stress_tensors[i]
            
            # Sources are other simulations
            source_indices = [j for j in range(len(param_vectors)) if j != i]
            
            # Randomly select num_sources
            if len(source_indices) > num_sources:
                import random
                source_indices = random.sample(source_indices, num_sources)
            
            source_params = [param_vectors[idx] for idx in source_indices]
            source_stress = [stress_tensors[idx] for idx in source_indices]
            
            train_pairs.append({
                'source_params': np.array(source_params),
                'source_stress': np.stack(source_stress, axis=0),
                'target_params': target_params,
                'target_stress': target_stress
            })
        
        return train_pairs
    
    def train(self, training_data, epochs=50, batch_size=1, validation_split=0.2):
        """Train the transformer model"""
        if not training_data:
            raise ValueError("No training data provided")
        
        # Split into training and validation
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        train_losses = []
        val_losses = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_idx in range(0, len(train_data), batch_size):
                batch = train_data[batch_idx:batch_idx + batch_size]
                
                if not batch:
                    continue
                
                # Prepare batch
                batch_source_params = []
                batch_source_stress = []
                batch_target_params = []
                batch_target_stress = []
                
                for item in batch:
                    batch_source_params.append(torch.from_numpy(item['source_params']).float())
                    batch_source_stress.append(item['source_stress'])
                    batch_target_params.append(torch.from_numpy(item['target_params']).float())
                    batch_target_stress.append(item['target_stress'].unsqueeze(0))
                
                # Stack batch
                source_params = torch.stack(batch_source_params, dim=0).to(self.device)
                source_stress = torch.stack(batch_source_stress, dim=0).to(self.device)
                target_params = torch.stack(batch_target_params, dim=0).to(self.device)
                target_stress = torch.cat(batch_target_stress, dim=0).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predicted_stress, _ = self.model(source_params, source_stress, target_params)
                
                # Compute loss
                loss = self.model.compute_loss(predicted_stress, target_stress)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / max(1, len(train_data) / batch_size)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx in range(0, len(val_data), batch_size):
                    batch = val_data[batch_idx:batch_idx + batch_size]
                    
                    if not batch:
                        continue
                    
                    batch_source_params = []
                    batch_source_stress = []
                    batch_target_params = []
                    batch_target_stress = []
                    
                    for item in batch:
                        batch_source_params.append(torch.from_numpy(item['source_params']).float())
                        batch_source_stress.append(item['source_stress'])
                        batch_target_params.append(torch.from_numpy(item['target_params']).float())
                        batch_target_stress.append(item['target_stress'].unsqueeze(0))
                    
                    source_params = torch.stack(batch_source_params, dim=0).to(self.device)
                    source_stress = torch.stack(batch_source_stress, dim=0).to(self.device)
                    target_params = torch.stack(batch_target_params, dim=0).to(self.device)
                    target_stress = torch.cat(batch_target_stress, dim=0).to(self.device)
                    
                    predicted_stress, _ = self.model(source_params, source_stress, target_params)
                    loss = self.model.compute_loss(predicted_stress, target_stress)
                    
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / max(1, len(val_data) / batch_size)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            self.scheduler.step(avg_val_loss)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping check
            if epoch > 10 and avg_val_loss > np.mean(val_losses[-10:]):
                st.info("Early stopping triggered")
                break
        
        progress_bar.empty()
        status_text.empty()
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return train_losses, val_losses
    
    def save_model(self, path=None):
        """Save trained model"""
        if path is None:
            path = os.path.join(NUMERICAL_SOLUTIONS_DIR, 'transformer_model.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'param_dim': self.param_dim,
            'stress_dim': self.stress_dim
        }, path)
        
        st.success(f"✅ Transformer model saved to {path}")
    
    def load_model(self, path=None):
        """Load trained model"""
        if path is None:
            path = os.path.join(NUMERICAL_SOLUTIONS_DIR, 'transformer_model.pth')
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.is_trained = True
            st.success("✅ Transformer model loaded successfully")
        else:
            st.warning("No saved transformer model found")
    
    def predict(self, source_simulations, target_params):
        """Make prediction using trained transformer"""
        if not self.is_trained:
            raise ValueError("Transformer model must be trained before prediction")
        
        # Prepare source data
        source_param_vectors = []
        source_stress_tensors = []
        
        for sim_data in source_simulations:
            if not sim_data.get('loaded_successfully', False):
                continue
            
            param_vector = self.param_processor.compute_parameter_vector(sim_data, self.param_dim)
            stress_fields = self.param_processor.extract_stress_fields(sim_data)
            
            stress_tensor = torch.stack([
                torch.from_numpy(stress_fields['sigma_hydro']).float(),
                torch.from_numpy(stress_fields['sigma_mag']).float(),
                torch.from_numpy(stress_fields['von_mises']).float()
            ], dim=0)
            
            source_param_vectors.append(param_vector)
            source_stress_tensors.append(stress_tensor)
        
        if not source_param_vectors:
            raise ValueError("No valid source simulations for prediction")
        
        # Prepare target parameters
        target_vector = self.param_processor.compute_parameter_vector(
            {'params': target_params}, self.param_dim
        )
        
        # Convert to tensors
        source_params_tensor = torch.from_numpy(np.array(source_param_vectors)).float().unsqueeze(0).to(self.device)
        source_stress_tensor = torch.stack(source_stress_tensors, dim=0).unsqueeze(0).to(self.device)
        target_params_tensor = torch.from_numpy(target_vector).float().unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            predicted_tensor, attention_weights = self.model(
                source_params_tensor, source_stress_tensor, target_params_tensor
            )
        
        # Convert to numpy
        predicted_stress = predicted_tensor.squeeze(0).cpu().numpy()
        
        result = {
            'sigma_hydro': predicted_stress[0],
            'sigma_mag': predicted_stress[1],
            'von_mises': predicted_stress[2],
            'method': 'transformer',
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None,
            'source_count': len(source_param_vectors),
            'target_params': target_params
        }
        
        return result, source_param_vectors

# =============================================
# HYBRID INTERPOLATOR (ENHANCED)
# =============================================
class EnhancedHybridInterpolator:
    """Enhanced hybrid interpolator combining kernel and transformer methods"""
    
    def __init__(self, kernel_weight=0.5, transformer_weight=0.5, 
                 kernel_config=None, transformer_config=None, device='cpu'):
        
        # Normalize weights
        total = kernel_weight + transformer_weight
        if total > 0:
            self.kernel_weight = kernel_weight / total
            self.transformer_weight = transformer_weight / total
        else:
            self.kernel_weight = 0.5
            self.transformer_weight = 0.5
        
        self.device = device
        
        # Initialize kernel interpolator
        kernel_config = kernel_config or {'kernel_type': 'rbf', 'length_scale': 0.3}
        self.kernel_interpolator = EnhancedKernelInterpolator(**kernel_config)
        
        # Initialize transformer manager
        transformer_config = transformer_config or {
            'param_dim': 15,
            'stress_dim': 3,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 3
        }
        self.transformer_manager = TransformerModelManager(**transformer_config, device=device)
        
        # State
        self.source_simulations = []
        self.is_kernel_fitted = False
        self.is_transformer_trained = False
    
    def fit(self, source_simulations):
        """Fit both kernel and transformer components"""
        self.source_simulations = [s for s in source_simulations if s.get('loaded_successfully', False)]
        
        if not self.source_simulations:
            raise ValueError("No valid source simulations found")
        
        st.info(f"Fitting hybrid model with {len(self.source_simulations)} simulations")
        
        # Fit kernel component
        try:
            self.kernel_interpolator.fit(self.source_simulations)
            self.is_kernel_fitted = True
            st.success("✅ Kernel component fitted successfully")
        except Exception as e:
            st.warning(f"⚠️ Kernel component fitting failed: {str(e)}")
            self.is_kernel_fitted = False
            self.kernel_weight = 0.0
        
        # Prepare training data for transformer
        try:
            if len(self.source_simulations) >= 4:
                training_data = self.transformer_manager.prepare_training_data(
                    self.source_simulations, num_sources=min(3, len(self.source_simulations)-1)
                )
                self.transformer_manager.load_model()  # Try to load pre-trained model
                self.is_transformer_trained = True
                st.success("✅ Transformer component ready (loaded pre-trained model)")
            else:
                st.warning("⚠️ Need at least 4 simulations for transformer training")
                self.transformer_weight = 0.0
                if self.is_kernel_fitted:
                    self.kernel_weight = 1.0
        except Exception as e:
            st.warning(f"⚠️ Transformer component setup failed: {str(e)}")
            self.transformer_weight = 0.0
            if self.is_kernel_fitted:
                self.kernel_weight = 1.0
        
        # Re-normalize weights
        total = self.kernel_weight + self.transformer_weight
        if total > 0:
            self.kernel_weight /= total
            self.transformer_weight /= total
        
        return self
    
    def train_transformer(self, epochs=50, batch_size=1):
        """Train the transformer component"""
        if not self.source_simulations:
            raise ValueError("No source simulations available")
        
        if len(self.source_simulations) < 4:
            raise ValueError("Need at least 4 simulations for transformer training")
        
        try:
            training_data = self.transformer_manager.prepare_training_data(
                self.source_simulations, num_sources=min(3, len(self.source_simulations)-1)
            )
            
            st.info(f"Training transformer with {len(training_data)} training pairs")
            
            train_losses, val_losses = self.transformer_manager.train(
                training_data, epochs=epochs, batch_size=batch_size
            )
            
            self.is_transformer_trained = True
            
            return train_losses, val_losses
            
        except Exception as e:
            st.error(f"❌ Transformer training failed: {str(e)}")
            self.transformer_weight = 0.0
            if self.is_kernel_fitted:
                self.kernel_weight = 1.0
            raise
    
    def predict(self, target_params):
        """Make hybrid prediction"""
        if not self.source_simulations:
            raise ValueError("No source simulations available")
        
        # Get predictions from each component
        kernel_pred = None
        transformer_pred = None
        kernel_weights = None
        source_param_vectors = None
        
        # Kernel prediction
        if self.is_kernel_fitted and self.kernel_weight > 0:
            try:
                kernel_pred, kernel_weights = self.kernel_interpolator.predict(target_params)
            except Exception as e:
                st.warning(f"Kernel prediction failed: {str(e)}")
                self.kernel_weight = 0.0
        
        # Transformer prediction
        if self.transformer_manager.is_trained and self.transformer_weight > 0:
            try:
                transformer_pred, source_param_vectors = self.transformer_manager.predict(
                    self.source_simulations, target_params
                )
            except Exception as e:
                st.warning(f"Transformer prediction failed: {str(e)}")
                self.transformer_weight = 0.0
        
        # Re-normalize weights if some components failed
        total = self.kernel_weight + self.transformer_weight
        if total == 0:
            raise ValueError("Both interpolation components failed")
        
        kernel_weight = self.kernel_weight / total
        transformer_weight = self.transformer_weight / total
        
        # Combine predictions
        if kernel_pred is not None and transformer_pred is not None:
            # Both components available - weighted combination
            hybrid_pred = {
                'sigma_hydro': (kernel_weight * kernel_pred['sigma_hydro'] + 
                               transformer_weight * transformer_pred['sigma_hydro']),
                'sigma_mag': (kernel_weight * kernel_pred['sigma_mag'] + 
                             transformer_weight * transformer_pred['sigma_mag']),
                'von_mises': (kernel_weight * kernel_pred['von_mises'] + 
                             transformer_weight * transformer_pred['von_mises']),
                'method': 'hybrid',
                'kernel_component': kernel_pred,
                'transformer_component': transformer_pred,
                'kernel_weight': kernel_weight,
                'transformer_weight': transformer_weight,
                'source_count': len(self.source_simulations),
                'target_params': target_params
            }
            
            return hybrid_pred, {
                'kernel_weights': kernel_weights,
                'source_param_vectors': source_param_vectors,
                'kernel_weight': kernel_weight,
                'transformer_weight': transformer_weight
            }
        
        elif kernel_pred is not None:
            # Only kernel available
            kernel_pred['method'] = 'kernel_only'
            return kernel_pred, {'kernel_weights': kernel_weights}
        
        elif transformer_pred is not None:
            # Only transformer available
            transformer_pred['method'] = 'transformer_only'
            return transformer_pred, {'source_param_vectors': source_param_vectors}
        
        else:
            raise ValueError("No interpolation components available")

# =============================================
# COMPLETE UNIFIED MANAGER
# =============================================
class CompleteUnifiedManager:
    """Complete unified manager with all three interpolation methods"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Initialize data loader
        self.data_loader = EnhancedDataLoader()
        
        # Initialize parameter processor
        self.param_processor = ParameterProcessor()
        
        # Initialize methods
        self.methods = {
            'kernel': EnhancedKernelInterpolator(kernel_type='rbf', length_scale=0.3),
            'transformer': TransformerModelManager(device=device),
            'hybrid': EnhancedHybridInterpolator(device=device)
        }
        
        self.current_method = 'kernel'
        self.source_simulations = []
        
        # Load pre-trained transformer if available
        self._load_pretrained_transformer()
    
    def _load_pretrained_transformer(self):
        """Load pre-trained transformer model if available"""
        try:
            model_path = os.path.join(NUMERICAL_SOLUTIONS_DIR, 'transformer_model.pth')
            if os.path.exists(model_path):
                self.methods['transformer'].load_model(model_path)
        except Exception as e:
            st.warning(f"Could not load pre-trained transformer: {str(e)}")
    
    def set_method(self, method):
        """Set current interpolation method"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        self.current_method = method
        return self
    
    def load_source_simulations(self, source_simulations):
        """Load and validate source simulations"""
        # Filter valid simulations
        valid_sims = []
        for sim_data in source_simulations:
            if isinstance(sim_data, dict) and sim_data.get('loaded_successfully', False):
                valid_sims.append(sim_data)
        
        if not valid_sims:
            st.warning("⚠️ No valid simulations found. Please check your data files.")
            return self
        
        self.source_simulations = valid_sims
        
        # Fit/train the current method
        try:
            if self.current_method == 'kernel':
                self.methods['kernel'].fit(valid_sims)
                st.success(f"✅ Kernel regression ready with {len(valid_sims)} simulations")
            
            elif self.current_method == 'transformer':
                if len(valid_sims) >= 2:
                    # For transformer, we just need to ensure we have data
                    # Training happens separately
                    st.info(f"Transformer loaded {len(valid_sims)} simulations. Click 'Train Transformer' to train.")
                else:
                    st.warning("⚠️ Need at least 2 simulations for transformer")
            
            elif self.current_method == 'hybrid':
                self.methods['hybrid'].fit(valid_sims)
                st.success(f"✅ Hybrid model ready with {len(valid_sims)} simulations")
        
        except Exception as e:
            st.error(f"❌ Error initializing {self.current_method}: {str(e)}")
        
        return self
    
    def train_transformer(self, epochs=50, batch_size=1):
        """Train the transformer model"""
        if self.current_method in ['transformer', 'hybrid']:
            try:
                if self.current_method == 'transformer':
                    if len(self.source_simulations) < 4:
                        st.warning("Need at least 4 simulations for transformer training")
                        return [], []
                    
                    # Prepare training data
                    training_data = self.methods['transformer'].prepare_training_data(
                        self.source_simulations, num_sources=min(3, len(self.source_simulations)-1)
                    )
                    
                    # Train
                    train_losses, val_losses = self.methods['transformer'].train(
                        training_data, epochs=epochs, batch_size=batch_size
                    )
                    
                    return train_losses, val_losses
                
                else:  # hybrid
                    train_losses, val_losses = self.methods['hybrid'].train_transformer(
                        epochs=epochs, batch_size=batch_size
                    )
                    
                    return train_losses, val_losses
                
            except Exception as e:
                st.error(f"❌ Transformer training failed: {str(e)}")
                return [], []
        
        return [], []
    
    def predict(self, target_params):
        """Make prediction using current method"""
        if not self.source_simulations:
            raise ValueError("No source simulations available")
        
        if self.current_method == 'kernel':
            return self.methods['kernel'].predict(target_params)
        
        elif self.current_method == 'transformer':
            if not self.methods['transformer'].is_trained:
                # Try to use pre-trained model or fallback
                try:
                    self.methods['transformer'].load_model()
                except:
                    st.warning("Transformer not trained. Please train first or use another method.")
                    raise ValueError("Transformer model not trained")
            
            return self.methods['transformer'].predict(self.source_simulations, target_params)
        
        elif self.current_method == 'hybrid':
            return self.methods['hybrid'].predict(target_params)
        
        else:
            raise ValueError(f"Unknown method: {self.current_method}")

# =============================================
# COMPLETE INTERFACE WITH ALL FUNCTIONALITY
# =============================================
def create_complete_unified_interface():
    """Create complete interface with all three interpolation methods"""
    
    st.header("🧬 Complete Stress Field Interpolation System")
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        st.sidebar.success("⚡ GPU available for transformer")
    else:
        st.sidebar.info("💻 Using CPU")
    
    # Initialize complete manager
    if 'complete_manager' not in st.session_state:
        st.session_state.complete_manager = CompleteUnifiedManager(device=device)
    
    # Initialize data loader
    if 'enhanced_data_loader' not in st.session_state:
        st.session_state.enhanced_data_loader = EnhancedDataLoader()
    
    # Initialize source simulations
    if 'complete_source_simulations' not in st.session_state:
        st.session_state.complete_source_simulations = []
    
    # Sidebar configuration
    st.sidebar.header("🔧 Method Selection")
    
    method = st.sidebar.selectbox(
        "Choose Method",
        [
            "⚡ Kernel Regression",
            "🧠 Transformer Cross-Attention", 
            "🧬 Hybrid Kernel-Transformer"
        ],
        index=0
    )
    
    # Map to method keys
    method_map = {
        "⚡ Kernel Regression": 'kernel',
        "🧠 Transformer Cross-Attention": 'transformer', 
        "🧬 Hybrid Kernel-Transformer": 'hybrid'
    }
    
    selected_method = method_map[method]
    st.session_state.complete_manager.set_method(selected_method)
    
    # Method-specific configuration
    with st.sidebar.expander("⚙️ Configuration", expanded=True):
        
        if selected_method == 'kernel':
            col1, col2 = st.columns(2)
            with col1:
                kernel_type = st.selectbox("Kernel", ["rbf", "matern", "rational_quadratic"], index=0)
            with col2:
                length_scale = st.slider("Length Scale", 0.05, 1.0, 0.3, 0.05)
            
            if st.button("🔄 Update Kernel", key="update_kernel"):
                st.session_state.complete_manager.methods['kernel'] = EnhancedKernelInterpolator(
                    kernel_type=kernel_type, length_scale=length_scale
                )
                if st.session_state.complete_source_simulations:
                    st.session_state.complete_manager.load_source_simulations(
                        st.session_state.complete_source_simulations
                    )
        
        elif selected_method == 'transformer':
            col1, col2 = st.columns(2)
            with col1:
                d_model = st.selectbox("Model Dim", [32, 64, 128], index=1)
                nhead = st.selectbox("Heads", [2, 4, 8], index=1)
            with col2:
                num_layers = st.selectbox("Layers", [2, 3, 4], index=1)
                dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
            
            if st.button("🔄 Update Transformer", key="update_transformer"):
                st.session_state.complete_manager.methods['transformer'] = TransformerModelManager(
                    d_model=d_model, nhead=nhead, num_layers=num_layers, device=device
                )
        
        else:  # hybrid
            col1, col2 = st.columns(2)
            with col1:
                kernel_weight = st.slider("Kernel Weight", 0.0, 1.0, 0.5, 0.1)
            with col2:
                transformer_weight = st.slider("Transformer Weight", 0.0, 1.0, 0.5, 0.1)
            
            if st.button("🔄 Update Hybrid", key="update_hybrid"):
                st.session_state.complete_manager.methods['hybrid'] = EnhancedHybridInterpolator(
                    kernel_weight=kernel_weight, transformer_weight=transformer_weight, device=device
                )
                if st.session_state.complete_source_simulations:
                    st.session_state.complete_manager.load_source_simulations(
                        st.session_state.complete_source_simulations
                    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Load Data", 
        "🏋️ Train Model",
        "🎯 Configure Target", 
        "🚀 Predict",
        "📊 Results",
        "📁 Manage"
    ])
    
    with tab1:
        st.subheader("Load Simulation Data")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📂 From Solutions Directory")
            
            if os.path.exists(NUMERICAL_SOLUTIONS_DIR):
                # Find all simulation files
                all_files = []
                for ext in ['*.pkl', '*.pt', '*.h5', '*.npz', '*.json', '*.npy']:
                    all_files.extend(glob.glob(os.path.join(NUMERICAL_SOLUTIONS_DIR, ext)))
                
                if all_files:
                    file_options = {os.path.basename(f): f for f in all_files}
                    selected_files = st.multiselect(
                        "Select files",
                        options=list(file_options.keys()),
                        key="select_complete"
                    )
                    
                    if selected_files and st.button("📥 Load Selected", key="load_selected"):
                        with st.spinner("Loading files..."):
                            loaded = 0
                            for filename in selected_files:
                                file_path = file_options[filename]
                                try:
                                    raw_data = st.session_state.enhanced_data_loader.load_file(file_path)
                                    if raw_data is not None:
                                        sim_data = EnhancedDataLoader.standardize_data(raw_data, file_path)
                                        if sim_data['loaded_successfully']:
                                            st.session_state.complete_source_simulations.append(sim_data)
                                            loaded += 1
                                        else:
                                            st.warning(f"{filename}: {sim_data.get('error', 'Unknown error')}")
                                except Exception as e:
                                    st.error(f"Error loading {filename}: {str(e)}")
                            
                            if loaded > 0:
                                st.success(f"Loaded {loaded} new files")
                                st.rerun()
                else:
                    st.info("No simulation files found in directory")
            else:
                st.warning(f"Directory not found: {NUMERICAL_SOLUTIONS_DIR}")
        
        with col2:
            st.markdown("### 📤 Upload Files")
            
            uploaded_files = st.file_uploader(
                "Upload simulation files",
                type=['pkl', 'pt', 'h5', 'npz', 'json', 'npy'],
                accept_multiple_files=True,
                key="upload_complete"
            )
            
            if uploaded_files and st.button("📥 Process Uploads", key="process_uploads"):
                with st.spinner("Processing..."):
                    for uploaded_file in uploaded_files:
                        try:
                            if uploaded_file.name.endswith('.pkl'):
                                raw_data = pickle.loads(uploaded_file.getvalue())
                            elif uploaded_file.name.endswith('.pt'):
                                raw_data = torch.load(BytesIO(uploaded_file.getvalue()), map_location='cpu')
                            elif uploaded_file.name.endswith(('.h5', '.hdf5')):
                                import h5py
                                with h5py.File(BytesIO(uploaded_file.getvalue()), 'r') as f:
                                    raw_data = {}
                                    def load_item(name, obj):
                                        if isinstance(obj, h5py.Dataset):
                                            raw_data[name] = obj[()]
                                    f.visititems(load_item)
                            elif uploaded_file.name.endswith('.npz'):
                                raw_data = dict(np.load(BytesIO(uploaded_file.getvalue()), allow_pickle=True))
                            elif uploaded_file.name.endswith('.json'):
                                raw_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
                            elif uploaded_file.name.endswith('.npy'):
                                raw_data = np.load(BytesIO(uploaded_file.getvalue()), allow_pickle=True).item()
                            else:
                                continue
                            
                            sim_data = EnhancedDataLoader.standardize_data(raw_data, uploaded_file.name)
                            st.session_state.complete_source_simulations.append(sim_data)
                            st.success(f"✅ {uploaded_file.name}")
                            
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}: {str(e)}")
        
        # Display loaded simulations
        if st.session_state.complete_source_simulations:
            st.subheader("📋 Loaded Simulations")
            
            valid_sims = [s for s in st.session_state.complete_source_simulations 
                         if s.get('loaded_successfully', False)]
            invalid_sims = [s for s in st.session_state.complete_source_simulations 
                           if not s.get('loaded_successfully', False)]
            
            if valid_sims:
                # Summary table
                summary_data = []
                for i, sim_data in enumerate(valid_sims[:20]):  # Show first 20
                    params = sim_data.get('params', {})
                    summary_data.append({
                        'ID': i+1,
                        'Defect': str(params.get('defect_type', 'Unknown'))[:15],
                        'Shape': str(params.get('shape', 'Unknown'))[:15],
                        'ε*': f"{params.get('eps0', 0):.3f}",
                        'κ': f"{params.get('kappa', 0):.3f}",
                        'Frames': len(sim_data.get('history', [])),
                        'Status': '✅'
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🔗 Load into Manager"):
                        st.session_state.complete_manager.load_source_simulations(
                            st.session_state.complete_source_simulations
                        )
                with col2:
                    st.info(f"**Valid:** {len(valid_sims)} | **Invalid:** {len(invalid_sims)}")
            
            if invalid_sims:
                with st.expander("⚠️ Invalid Simulations"):
                    for sim_data in invalid_sims:
                        st.error(f"{sim_data.get('filename', 'Unknown')}: {sim_data.get('error', 'Unknown error')}")
            
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.complete_source_simulations = []
                st.success("Cleared all simulations")
                st.rerun()
    
    with tab2:
        st.subheader("Train Models")
        
        if not st.session_state.complete_source_simulations:
            st.warning("Please load simulations first")
        else:
            valid_sims = [s for s in st.session_state.complete_source_simulations 
                         if s.get('loaded_successfully', False)]
            
            if len(valid_sims) < 2:
                st.warning(f"Need at least 2 valid simulations (have {len(valid_sims)})")
            else:
                st.info(f"Available: {len(valid_sims)} valid simulations")
                
                if selected_method == 'kernel':
                    st.success("✅ Kernel regression doesn't require training - ready to use!")
                    
                    if st.button("🔄 Re-fit Kernel"):
                        with st.spinner("Fitting kernel..."):
                            st.session_state.complete_manager.methods['kernel'].fit(valid_sims)
                            st.success("Kernel re-fitted successfully")
                
                elif selected_method == 'transformer':
                    if len(valid_sims) < 4:
                        st.warning(f"Need at least 4 simulations for transformer training (have {len(valid_sims)})")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            epochs = st.slider("Epochs", 10, 200, 50, 10)
                            batch_size = st.slider("Batch Size", 1, 8, 1, 1)
                        
                        with col2:
                            lr = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")
                            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, 0.05)
                        
                        if st.button("🚀 Train Transformer", type="primary"):
                            with st.spinner("Training transformer..."):
                                try:
                                    # Update learning rate
                                    st.session_state.complete_manager.methods['transformer'].optimizer.param_groups[0]['lr'] = lr
                                    
                                    # Train
                                    train_losses, val_losses = st.session_state.complete_manager.train_transformer(
                                        epochs=epochs, batch_size=batch_size
                                    )
                                    
                                    if train_losses:
                                        # Plot training curve
                                        fig, ax = plt.subplots(figsize=(10, 4))
                                        ax.plot(train_losses, label='Train Loss', linewidth=2)
                                        if val_losses:
                                            ax.plot(val_losses, label='Val Loss', linewidth=2)
                                        ax.set_xlabel('Epoch')
                                        ax.set_ylabel('Loss')
                                        ax.set_title('Training Progress')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        st.pyplot(fig)
                                        
                                        st.success(f"✅ Training complete! Final loss: {train_losses[-1]:.6f}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Training failed: {str(e)}")
                
                else:  # hybrid
                    st.info("Hybrid model combines kernel and transformer components")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        train_transformer = st.checkbox("Train Transformer Component", value=True)
                        epochs = st.slider("Training Epochs", 10, 100, 30, 5)
                    
                    with col2:
                        if train_transformer and len(valid_sims) >= 4:
                            if st.button("🚀 Train Hybrid Model", type="primary"):
                                with st.spinner("Training hybrid model..."):
                                    try:
                                        train_losses, val_losses = st.session_state.complete_manager.train_transformer(
                                            epochs=epochs
                                        )
                                        
                                        if train_losses:
                                            fig, ax = plt.subplots(figsize=(10, 4))
                                            ax.plot(train_losses, label='Train Loss', linewidth=2)
                                            if val_losses:
                                                ax.plot(val_losses, label='Val Loss', linewidth=2)
                                            ax.set_xlabel('Epoch')
                                            ax.set_ylabel('Loss')
                                            ax.set_title('Transformer Component Training')
                                            ax.legend()
                                            ax.grid(True, alpha=0.3)
                                            st.pyplot(fig)
                                            
                                            st.success("✅ Hybrid model training complete!")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Training failed: {str(e)}")
                        else:
                            st.info("Kernel component is ready. Transformer needs training.")
    
    with tab3:
        st.subheader("Configure Target Parameters")
        
        if not st.session_state.complete_source_simulations:
            st.warning("Please load simulations first")
        else:
            valid_sims = [s for s in st.session_state.complete_source_simulations 
                         if s.get('loaded_successfully', False)]
            
            if not valid_sims:
                st.error("No valid simulations loaded")
            else:
                # Extract parameter ranges
                defects = set()
                shapes = set()
                eps0_vals = []
                kappa_vals = []
                
                for sim_data in valid_sims:
                    params = sim_data.get('params', {})
                    if 'defect_type' in params:
                        defects.add(str(params['defect_type']))
                    if 'shape' in params:
                        shapes.add(str(params['shape']))
                    if 'eps0' in params:
                        try:
                            eps0_vals.append(float(params['eps0']))
                        except:
                            pass
                    if 'kappa' in params:
                        try:
                            kappa_vals.append(float(params['kappa']))
                        except:
                            pass
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Target defect type
                    if defects:
                        target_defect = st.selectbox(
                            "Defect Type",
                            list(defects),
                            index=0,
                            key="target_defect_complete"
                        )
                    else:
                        target_defect = st.selectbox(
                            "Defect Type",
                            ["ISF", "ESF", "Twin", "Void", "Dislocation"],
                            index=0,
                            key="target_defect_default"
                        )
                    
                    # Target shape
                    if shapes:
                        target_shape = st.selectbox(
                            "Shape",
                            list(shapes),
                            index=0,
                            key="target_shape_complete"
                        )
                    else:
                        target_shape = st.selectbox(
                            "Shape",
                            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                            index=0,
                            key="target_shape_default"
                        )
                    
                    # Target ε*
                    if eps0_vals:
                        eps0_min, eps0_max = min(eps0_vals), max(eps0_vals)
                        target_eps0 = st.slider(
                            "ε*",
                            float(eps0_min), float(eps0_max),
                            float((eps0_min + eps0_max) / 2),
                            0.01,
                            key="target_eps0_complete"
                        )
                    else:
                        target_eps0 = st.slider(
                            "ε*", 0.3, 3.0, 1.414, 0.01,
                            key="target_eps0_default"
                        )
                
                with col2:
                    # Target κ
                    if kappa_vals:
                        kappa_min, kappa_max = min(kappa_vals), max(kappa_vals)
                        target_kappa = st.slider(
                            "κ",
                            float(kappa_min), float(kappa_max),
                            float((kappa_min + kappa_max) / 2),
                            0.05,
                            key="target_kappa_complete"
                        )
                    else:
                        target_kappa = st.slider(
                            "κ", 0.1, 2.0, 0.7, 0.05,
                            key="target_kappa_default"
                        )
                    
                    # Target orientation
                    orientation_mode = st.radio(
                        "Orientation Mode",
                        ["Predefined", "Custom"],
                        horizontal=True,
                        key="orientation_mode_complete"
                    )
                    
                    if orientation_mode == "Predefined":
                        target_orientation = st.selectbox(
                            "Orientation",
                            ["Horizontal {111} (0°)", 
                             "Tilted 30° (1¯10 projection)", 
                             "Tilted 60°", 
                             "Vertical {111} (90°)"],
                            index=0,
                            key="target_orientation_predefined"
                        )
                        angle_map = {
                            "Horizontal {111} (0°)": 0,
                            "Tilted 30° (1¯10 projection)": 30,
                            "Tilted 60°": 60,
                            "Vertical {111} (90°)": 90,
                        }
                        target_theta = np.deg2rad(angle_map[target_orientation])
                    else:
                        target_angle = st.slider(
                            "Angle (degrees)", 0.0, 90.0, 45.0, 0.5,
                            key="target_angle_custom"
                        )
                        target_theta = np.deg2rad(target_angle)
                        target_orientation = f"Custom ({target_angle:.1f}°)"
                
                # Store target parameters
                st.session_state.target_params_complete = {
                    'defect_type': target_defect,
                    'shape': target_shape,
                    'eps0': target_eps0,
                    'kappa': target_kappa,
                    'orientation': target_orientation,
                    'theta': target_theta
                }
                
                # Show summary
                st.subheader("🎯 Target Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Defect Type", target_defect)
                    st.metric("Shape", target_shape)
                    st.metric("ε*", f"{target_eps0:.3f}")
                with col2:
                    st.metric("κ", f"{target_kappa:.3f}")
                    st.metric("Orientation", target_orientation)
                    st.metric("Angle", f"{np.rad2deg(target_theta):.1f}°")
    
    with tab4:
        st.subheader("Make Prediction")
        
        if 'target_params_complete' not in st.session_state:
            st.warning("Please configure target parameters first")
        elif not st.session_state.complete_source_simulations:
            st.warning("Please load simulations first")
        else:
            st.info(f"**Method:** {method}")
            
            # Additional options
            if selected_method == 'hybrid':
                col1, col2 = st.columns(2)
                with col1:
                    use_kernel = st.checkbox("Use Kernel", value=True)
                with col2:
                    use_transformer = st.checkbox("Use Transformer", value=True)
            
            if st.button("🚀 Run Prediction", type="primary", key="run_prediction"):
                with st.spinner("Running prediction..."):
                    try:
                        predicted_stress, extra_info = st.session_state.complete_manager.predict(
                            st.session_state.target_params_complete
                        )
                        
                        st.session_state.prediction_results_complete = {
                            'stress_fields': predicted_stress,
                            'extra_info': extra_info,
                            'method': selected_method,
                            'target_params': st.session_state.target_params_complete,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.success("✅ Prediction complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
    
    with tab5:
        st.subheader("Results & Analysis")
        
        if 'prediction_results_complete' not in st.session_state:
            st.info("👈 Run a prediction first")
        else:
            results = st.session_state.prediction_results_complete
            
            # Show method info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", results['method'].upper())
            with col2:
                valid_sims = len([s for s in st.session_state.complete_source_simulations 
                                 if s.get('loaded_successfully', False)])
                st.metric("Source Simulations", valid_sims)
            with col3:
                if results['method'] == 'hybrid' and 'extra_info' in results:
                    weights = results['extra_info']
                    if 'kernel_weight' in weights:
                        st.metric("Kernel Weight", f"{weights['kernel_weight']:.2f}")
            
            # Display stress fields
            st.subheader("🎯 Predicted Stress Fields")
            
            stress_fields = results['stress_fields']
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            titles = ['Hydrostatic Stress', 'Stress Magnitude', 'Von Mises Stress']
            components = ['sigma_hydro', 'sigma_mag', 'von_mises']
            cmaps = ['coolwarm', 'viridis', 'plasma']
            
            for ax, title, comp, cmap in zip(axes, titles, components, cmaps):
                if comp in stress_fields:
                    data = stress_fields[comp]
                    if hasattr(data, 'shape'):
                        # Ensure 2D
                        if data.ndim > 2:
                            data = data.squeeze()
                        
                        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='equal')
                        ax.set_title(f"{title} (GPa)")
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(title)
            
            st.pyplot(fig)
            
            # Show weights/attention if available
            if 'extra_info' in results:
                extra_info = results['extra_info']
                
                if 'kernel_weights' in extra_info:
                    st.subheader("🔍 Kernel Weights")
                    
                    kernel_weights = extra_info['kernel_weights']
                    
                    fig_weights, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(range(len(kernel_weights)), kernel_weights, alpha=0.7)
                    ax.set_xlabel('Source Simulation')
                    ax.set_ylabel('Weight')
                    ax.set_title('Kernel Regression Weights')
                    ax.set_xticks(range(len(kernel_weights)))
                    ax.set_xticklabels([f'S{i+1}' for i in range(len(kernel_weights))])
                    st.pyplot(fig_weights)
            
            # Stress statistics
            st.subheader("📊 Stress Statistics")
            
            stats_data = []
            for comp in components:
                if comp in stress_fields:
                    data = stress_fields[comp]
                    if hasattr(data, 'flatten'):
                        flat_data = data.flatten()
                        stats_data.append({
                            'Component': comp,
                            'Max (GPa)': float(np.nanmax(flat_data)),
                            'Min (GPa)': float(np.nanmin(flat_data)),
                            'Mean (GPa)': float(np.nanmean(flat_data)),
                            'Std Dev': float(np.nanstd(flat_data))
                        })
            
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats.style.format({
                    'Max (GPa)': '{:.3f}',
                    'Min (GPa)': '{:.3f}',
                    'Mean (GPa)': '{:.3f}',
                    'Std Dev': '{:.3f}'
                }), use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Save Results"):
                    export_data = {
                        'prediction': results,
                        'source_count': len(st.session_state.complete_source_simulations),
                        'target_params': st.session_state.target_params_complete,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    filename = f"prediction_{results['method']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    filepath = os.path.join(NUMERICAL_SOLUTIONS_DIR, filename)
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(export_data, f)
                    
                    st.success(f"✅ Saved to {filename}")
            
            with col2:
                # Download button
                export_buffer = BytesIO()
                pickle.dump(results, export_buffer)
                export_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download",
                    data=export_buffer,
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
            
            with col3:
                # Generate report
                report = f"""
                PREDICTION REPORT
                =================
                
                Method: {results['method']}
                Timestamp: {datetime.now().isoformat()}
                
                Target Parameters:
                - Defect Type: {st.session_state.target_params_complete['defect_type']}
                - Shape: {st.session_state.target_params_complete['shape']}
                - ε*: {st.session_state.target_params_complete['eps0']:.3f}
                - κ: {st.session_state.target_params_complete['kappa']:.3f}
                - Orientation: {st.session_state.target_params_complete['orientation']}
                
                Stress Statistics:
                """
                
                for stat in stats_data:
                    report += f"\n{stat['Component']}:"
                    report += f"\n  Max: {stat['Max (GPa)']:.3f} GPa"
                    report += f"\n  Min: {stat['Min (GPa)']:.3f} GPa"
                    report += f"\n  Mean: {stat['Mean (GPa)']:.3f} GPa"
                    report += f"\n  Std: {stat['Std Dev']:.3f}"
                
                st.download_button(
                    label="📄 Report",
                    data=report,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with tab6:
        st.subheader("File & Model Management")
        
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
        
        # File browser
        if os.path.exists(NUMERICAL_SOLUTIONS_DIR):
            files = glob.glob(os.path.join(NUMERICAL_SOLUTIONS_DIR, "*"))
            
            if files:
                file_data = []
                for file in files:
                    try:
                        stat = os.stat(file)
                        file_data.append({
                            'Name': os.path.basename(file),
                            'Size (KB)': stat.st_size // 1024,
                            'Modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                            'Type': os.path.splitext(file)[1]
                        })
                    except:
                        continue
                
                df_files = pd.DataFrame(file_data)
                st.dataframe(df_files, use_container_width=True)
                
                # File actions
                if file_data:
                    selected_file = st.selectbox(
                        "Select file",
                        [f['Name'] for f in file_data]
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🗑️ Delete"):
                            file_path = os.path.join(NUMERICAL_SOLUTIONS_DIR, selected_file)
                            try:
                                os.remove(file_path)
                                st.success(f"Deleted {selected_file}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    with col2:
                        if st.button("🔄 Refresh"):
                            st.rerun()
                    
                    with col3:
                        if st.button("📦 Backup All"):
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                for file in files:
                                    zip_file.write(file, os.path.basename(file))
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                label="Download ZIP",
                                data=zip_buffer,
                                file_name="backup.zip",
                                mime="application/zip"
                            )
            else:
                st.info("No files in directory")
        
        # Model management
        st.subheader("🧠 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Transformer Model"):
                try:
                    st.session_state.complete_manager.methods['transformer'].save_model()
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        
        with col2:
            if st.button("📥 Load Transformer Model"):
                try:
                    st.session_state.complete_manager.methods['transformer'].load_model()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application with complete functionality"""
    
    # Set page config
    st.set_page_config(
        page_title="Complete Stress Interpolator",
        page_icon="🧠",
        layout="wide"
    )
    
    # Title
    st.title("🧠 Complete Stress Field Interpolation System")
    st.markdown("### Three Methods: ⚡ Kernel Regression • 🧠 Transformer Cross-Attention • 🧬 Hybrid")
    
    # Sidebar info
    st.sidebar.title("📊 System Information")
    
    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"**Device:** {device}")
    st.sidebar.info(f"**Directory:**\n`{NUMERICAL_SOLUTIONS_DIR}`")
    
    # Quick start guide
    with st.sidebar.expander("🚀 Quick Start", expanded=False):
        st.markdown("""
        1. **Load Data**: Upload or select simulation files
        2. **Choose Method**: Select interpolation method
        3. **Configure**: Set target parameters
        4. **Predict**: Run interpolation
        5. **Analyze**: View and export results
        """)
    
    # Method comparison
    with st.sidebar.expander("📊 Method Comparison", expanded=False):
        st.markdown("""
        | Method | Speed | Accuracy | Training |
        |--------|-------|----------|----------|
        | ⚡ Kernel | Fast | Good | None |
        | 🧠 Transformer | Medium | Best | Required |
        | 🧬 Hybrid | Medium | Very Good | Optional |
        """)
    
    # Create the interface
    create_complete_unified_interface()
    
    # Theory section
    with st.expander("🔬 Detailed Theory", expanded=False):
        st.markdown("""
        ## **Complete Interpolation System Theory**
        
        ### **⚡ Kernel Regression**
        - **Gaussian RBF kernel**: $K(x, x') = \\exp(-\\|x - x'\\|^2 / (2\\sigma^2))$
        - **Nadaraya-Watson estimator**: Weighted average of source simulations
        - **Advantages**: Fast, interpretable, no training required
        - **Limitations**: Fixed similarity metric, no spatial awareness
        
        ### **🧠 Transformer Cross-Attention**
        - **Architecture**: 
          - Parameter embedding network
          - Stress field encoder (convolutional)
          - Transformer encoder for parameter relationships
          - Cross-attention between parameters and stress
          - Stress field decoder
        - **Training**: Leave-one-out cross-validation
        - **Advantages**: Learns complex relationships, spatial awareness
        - **Limitations**: Requires training data, computationally intensive
        
        ### **🧬 Hybrid Approach**
        - **Weighted combination**: $f_{hybrid} = \\alpha f_{kernel} + \\beta f_{transformer}$
        - **Adaptive weights**: Can be tuned based on data availability
        - **Advantages**: Combines strengths of both methods, robust
        - **Applications**: Best for mixed-quality data, uncertain scenarios
        
        ### **🔄 Workflow**
        1. **Data Loading**: Support for multiple file formats (PKL, PT, H5, NPZ, JSON)
        2. **Parameter Encoding**: 15-dimensional feature vector including:
           - Defect type (one-hot)
           - Shape encoding
           - Normalized numerical parameters (ε*, κ, θ)
           - Orientation encoding
        3. **Model Training/Fitting**:
           - Kernel: Instant fitting
           - Transformer: Requires 4+ simulations, GPU recommended
        4. **Prediction**: Interpolate stress fields for new parameters
        5. **Analysis**: Visualize results, compare methods, export data
        
        ### **🔬 Scientific Validation**
        - **Stress continuity**: Ensured through regularization
        - **Parameter sensitivity**: Captured through attention mechanisms
        - **Physical constraints**: Implicitly learned from training data
        - **Extrapolation warning**: Provided when target parameters are outside training range
        
        ### **🚀 Performance Optimization**
        - **GPU acceleration**: For transformer training and prediction
        - **Batch processing**: For multiple target predictions
        - **Model caching**: Save and load trained models
        - **Memory efficiency**: Patch-based processing for large stress fields
        """)

if __name__ == "__main__":
    main()
