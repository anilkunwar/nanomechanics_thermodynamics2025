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
# DATA LOADING AND STANDARDIZATION
# =============================================
class DataLoader:
    """Handles loading and standardization of simulation data"""
    
    @staticmethod
    def load_simulation_file(file_path):
        """Load simulation data from various file formats"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.pkl', '.pickle']:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif ext in ['.pt', '.pth']:
                return torch.load(file_path, map_location='cpu')
            elif ext in ['.h5', '.hdf5']:
                data = {}
                with h5py.File(file_path, 'r') as f:
                    def load_item(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            data[name] = obj[()]
                    f.visititems(load_item)
                return data
            elif ext == '.npz':
                return dict(np.load(file_path, allow_pickle=True))
            elif ext == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def standardize_simulation_data(raw_data, file_path):
        """Standardize simulation data to a common format"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'loaded_successfully': False
        }
        
        try:
            # Handle different data formats
            if isinstance(raw_data, dict):
                # Extract parameters
                if 'params' in raw_data:
                    standardized['params'] = raw_data['params']
                elif 'parameters' in raw_data:
                    standardized['params'] = raw_data['parameters']
                elif 'simulation_params' in raw_data:
                    standardized['params'] = raw_data['simulation_params']
                else:
                    # Try to extract parameters from the data structure
                    param_keys = ['defect_type', 'shape', 'eps0', 'kappa', 'orientation', 'theta']
                    standardized['params'] = {k: raw_data.get(k) for k in param_keys if k in raw_data}
                
                # Extract history
                if 'history' in raw_data:
                    history = raw_data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                elif 'frames' in raw_data:
                    standardized['history'] = raw_data['frames']
                elif 'results' in raw_data:
                    standardized['history'] = raw_data['results']
                
                # Extract metadata
                if 'metadata' in raw_data:
                    standardized['metadata'] = raw_data['metadata']
                elif 'info' in raw_data:
                    standardized['metadata'] = raw_data['info']
                
                # Extract stress fields from history if not already in the right format
                if standardized['history']:
                    # Convert history to standard format
                    converted_history = []
                    for frame in standardized['history']:
                        if isinstance(frame, tuple) and len(frame) == 2:
                            # Already in (eta, stress_fields) format
                            converted_history.append(frame)
                        elif isinstance(frame, dict):
                            # Try to extract eta and stress fields
                            eta = frame.get('eta', frame.get('displacement', frame.get('field', np.zeros((128, 128)))))
                            stresses = {}
                            
                            # Try to find stress components
                            for stress_type in ['sigma_hydro', 'sigma_mag', 'von_mises', 'stress_xx', 'stress_yy', 'stress_xy']:
                                if stress_type in frame:
                                    stresses[stress_type] = frame[stress_type]
                                elif f'stress_{stress_type}' in frame:
                                    stresses[stress_type] = frame[f'stress_{stress_type}']
                            
                            # If no specific stresses found, check for general stress field
                            if not stresses and 'stress' in frame:
                                stress_field = frame['stress']
                                if isinstance(stress_field, np.ndarray) and stress_field.ndim == 2:
                                    stresses['von_mises'] = stress_field
                            
                            converted_history.append((eta, stresses))
                        else:
                            # Unsupported format, create placeholder
                            converted_history.append((np.zeros((128, 128)), {}))
                    
                    standardized['history'] = converted_history
                
                standardized['loaded_successfully'] = True
                
            else:
                # Try to extract data from non-dict format
                standardized['params'] = {'raw_data_type': type(raw_data).__name__}
                standardized['history'] = []
                
            return standardized
            
        except Exception as e:
            st.warning(f"Error standardizing data from {file_path}: {str(e)}")
            standardized['error'] = str(e)
            return standardized

# =============================================
# KERNEL REGRESSION INTERPOLATOR
# =============================================
class KernelRegressionInterpolator:
    """Kernel-based regression interpolator using Gaussian RBF kernels"""
    
    def __init__(self, kernel_type='rbf', length_scale=0.3, nu=1.5):
        """
        Initialize kernel regression interpolator
        
        Args:
            kernel_type: 'rbf' (Radial Basis Function), 'matern', 'rational_quadratic'
            length_scale: Length scale parameter for kernel
            nu: Smoothness parameter for Matern kernel
        """
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.nu = nu
        self.source_params = None
        self.source_stress = None
        self.kernel_matrix = None
        
    def _compute_kernel(self, X1, X2):
        """Compute kernel matrix between two sets of parameters"""
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
        
        # Extract parameters and stress fields
        source_params = []
        source_stress = []
        
        for sim_data in source_simulations:
            # Skip simulations that failed to load
            if not sim_data.get('loaded_successfully', False):
                continue
                
            param_vector = self.compute_parameter_vector(sim_data)
            source_params.append(param_vector)
            
            # Get stress from final frame
            history = sim_data.get('history', [])
            if history:
                # Handle different history formats
                last_frame = history[-1]
                if isinstance(last_frame, tuple) and len(last_frame) == 2:
                    _, stress_fields = last_frame
                elif isinstance(last_frame, dict):
                    stress_fields = last_frame
                else:
                    stress_fields = {}
                
                # Create default zero arrays with correct shape
                default_shape = (128, 128)
                
                # Try to get actual shape from any available field
                for field_name in ['sigma_hydro', 'sigma_mag', 'von_mises', 'eta', 'displacement']:
                    if field_name in stress_fields:
                        field_data = stress_fields[field_name]
                        if hasattr(field_data, 'shape'):
                            default_shape = field_data.shape
                            break
                
                # Extract or create stress components
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros(default_shape)),
                    stress_fields.get('sigma_mag', np.zeros(default_shape)),
                    stress_fields.get('von_mises', np.zeros(default_shape))
                ], axis=0)
                source_stress.append(stress_components)
            else:
                # No history, use zeros
                source_stress.append(np.zeros((3, 128, 128)))
        
        if not source_params:
            raise ValueError("No valid source simulations found")
        
        self.source_params = np.array(source_params)
        self.source_stress = np.array(source_stress)
        
        # Precompute kernel matrix for fast predictions
        self.kernel_matrix = self._compute_kernel(self.source_params, self.source_params)
        
        # Add small regularization for numerical stability
        self.kernel_matrix += np.eye(len(source_params)) * 1e-8
        
        return self
    
    def predict(self, target_params):
        """Predict stress field for target parameters"""
        if self.source_params is None or self.source_stress is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Compute target parameter vector
        target_vector = self.compute_parameter_vector({'params': target_params})
        
        # Compute kernel weights
        k_star = self._compute_kernel(self.source_params, target_vector.reshape(1, -1))
        
        # Solve for weights (kernel regression)
        try:
            weights = np.linalg.solve(self.kernel_matrix, k_star)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            weights = np.linalg.pinv(self.kernel_matrix) @ k_star
        
        # Weighted combination of source stress fields
        weights = weights.flatten()
        weights = weights / (np.sum(np.abs(weights)) + 1e-8)
        
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
            'source_count': len(self.source_params)
        }
        
        return predicted_stress, weights
    
    def compute_parameter_vector(self, sim_data):
        """Compute parameter vector from simulation data"""
        params = sim_data.get('params', {})
        
        param_vector = []
        
        # 1. Defect type encoding
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1],
            'Void': [0, 0, 0],
            'Dislocation': [0, 0, 0]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        
        # 2. Shape encoding
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1],
            'Circle': [0, 0, 0, 0, 0]
        }
        shape = params.get('shape', 'Square')
        param_vector.extend(shape_encoding.get(shape, [0, 0, 0, 0, 0]))
        
        # 3. Numerical parameters (normalized)
        eps0 = params.get('eps0', params.get('epsilon', params.get('strain', 0.707)))
        kappa = params.get('kappa', params.get('kappa_factor', params.get('curvature', 0.6)))
        theta = params.get('theta', params.get('angle', params.get('orientation_angle', 0.0)))
        
        # Normalize within reasonable ranges
        eps0_norm = max(0.0, min(1.0, (eps0 - 0.3) / (3.0 - 0.3))) if isinstance(eps0, (int, float)) else 0.5
        param_vector.append(eps0_norm)
        
        kappa_norm = max(0.0, min(1.0, (kappa - 0.1) / (2.0 - 0.1))) if isinstance(kappa, (int, float)) else 0.5
        param_vector.append(kappa_norm)
        
        # Handle theta (angle in radians)
        if isinstance(theta, (int, float)):
            theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        else:
            theta_norm = 0.0
        param_vector.append(theta_norm)
        
        # 4. Orientation encoding
        orientation = params.get('orientation', 'Horizontal {111} (0°)')
        orientation_encoding = {
            'Horizontal {111} (0°)': [1, 0, 0, 0],
            'Tilted 30° (1¯10 projection)': [0, 1, 0, 0],
            'Tilted 60°': [0, 0, 1, 0],
            'Vertical {111} (90°)': [0, 0, 0, 1]
        }
        
        if isinstance(orientation, str) and orientation.startswith('Custom ('):
            param_vector.extend([0, 0, 0, 0])
        else:
            param_vector.extend(orientation_encoding.get(str(orientation), [0, 0, 0, 0]))
        
        # Ensure we have exactly 15 dimensions
        if len(param_vector) < 15:
            param_vector.extend([0] * (15 - len(param_vector)))
        elif len(param_vector) > 15:
            param_vector = param_vector[:15]
        
        return np.array(param_vector, dtype=np.float32)

# =============================================
# SIMPLE TRANSFORMER INTERPOLATOR (SIMPLIFIED)
# =============================================
class SimpleTransformerInterpolator:
    """Simplified transformer-based interpolator for demonstration"""
    
    def __init__(self, param_dim=15, d_model=64, num_heads=4, device='cpu'):
        self.param_dim = param_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        
        # Simple attention mechanism
        self.W_q = nn.Linear(param_dim, d_model).to(device)
        self.W_k = nn.Linear(param_dim, d_model).to(device)
        self.W_v = nn.Linear(param_dim, d_model).to(device)
        
        self.source_params = None
        self.source_stress = None
        
    def fit(self, source_simulations):
        """Fit the transformer model to source simulations"""
        if not source_simulations:
            raise ValueError("No source simulations provided")
        
        # Extract parameters and stress fields
        source_params = []
        source_stress = []
        
        for sim_data in source_simulations:
            if not sim_data.get('loaded_successfully', False):
                continue
                
            param_vector = self.compute_parameter_vector(sim_data)
            source_params.append(param_vector)
            
            # Get stress from final frame
            history = sim_data.get('history', [])
            if history:
                last_frame = history[-1]
                if isinstance(last_frame, tuple) and len(last_frame) == 2:
                    _, stress_fields = last_frame
                elif isinstance(last_frame, dict):
                    stress_fields = last_frame
                else:
                    stress_fields = {}
                
                default_shape = (128, 128)
                for field_name in ['sigma_hydro', 'sigma_mag', 'von_mises', 'eta']:
                    if field_name in stress_fields:
                        field_data = stress_fields[field_name]
                        if hasattr(field_data, 'shape'):
                            default_shape = field_data.shape
                            break
                
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros(default_shape)),
                    stress_fields.get('sigma_mag', np.zeros(default_shape)),
                    stress_fields.get('von_mises', np.zeros(default_shape))
                ], axis=0)
                source_stress.append(stress_components)
            else:
                source_stress.append(np.zeros((3, 128, 128)))
        
        if not source_params:
            raise ValueError("No valid source simulations found")
        
        self.source_params = torch.tensor(np.array(source_params), dtype=torch.float32).to(self.device)
        self.source_stress = torch.tensor(np.array(source_stress), dtype=torch.float32).to(self.device)
        
        return self
    
    def predict(self, target_params):
        """Predict stress field using attention mechanism"""
        if self.source_params is None or self.source_stress is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Compute target parameter vector
        target_vector = self.compute_parameter_vector({'params': target_params})
        target_tensor = torch.tensor(target_vector, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        # Compute attention weights
        Q = self.W_q(target_tensor)  # Query
        K = self.W_k(self.source_params)  # Keys
        V = self.W_v(self.source_params)  # Values (for parameter similarity)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to stress fields
        attention_weights_np = attention_weights.detach().cpu().numpy().flatten()
        
        # Weighted combination of source stress fields
        source_stress_np = self.source_stress.detach().cpu().numpy()
        weighted_stress = np.sum(
            source_stress_np * attention_weights_np[:, np.newaxis, np.newaxis, np.newaxis], 
            axis=0
        )
        
        predicted_stress = {
            'sigma_hydro': weighted_stress[0],
            'sigma_mag': weighted_stress[1],
            'von_mises': weighted_stress[2],
            'method': 'transformer',
            'attention_weights': attention_weights_np,
            'source_count': len(self.source_params)
        }
        
        return predicted_stress, attention_weights_np
    
    def compute_parameter_vector(self, sim_data):
        """Compute parameter vector from simulation data"""
        # Use the same implementation as KernelRegressionInterpolator
        kernel_interp = KernelRegressionInterpolator()
        return kernel_interp.compute_parameter_vector(sim_data)

# =============================================
# HYBRID INTERPOLATOR
# =============================================
class HybridInterpolator:
    """Hybrid interpolator combining kernel regression and transformer attention"""
    
    def __init__(self, kernel_weight=0.5, transformer_weight=0.5, device='cpu'):
        """
        Initialize hybrid interpolator
        
        Args:
            kernel_weight: Weight for kernel regression component (0-1)
            transformer_weight: Weight for transformer component (0-1)
        """
        # Normalize weights
        total = kernel_weight + transformer_weight
        if total > 0:
            self.kernel_weight = kernel_weight / total
            self.transformer_weight = transformer_weight / total
        else:
            self.kernel_weight = 0.5
            self.transformer_weight = 0.5
            
        self.device = device
        
        # Initialize component interpolators
        self.kernel_interpolator = KernelRegressionInterpolator(kernel_type='rbf', length_scale=0.3)
        self.transformer_interpolator = SimpleTransformerInterpolator(device=device)
        
        # State variables
        self.source_simulations = []
        self.is_kernel_fitted = False
        self.is_transformer_fitted = False
        
    def fit(self, source_simulations):
        """Fit both kernel and transformer components"""
        self.source_simulations = [s for s in source_simulations if s.get('loaded_successfully', False)]
        
        if not self.source_simulations:
            raise ValueError("No valid source simulations found")
        
        # Fit kernel component
        try:
            self.kernel_interpolator.fit(self.source_simulations)
            self.is_kernel_fitted = True
        except Exception as e:
            st.warning(f"Kernel component fitting failed: {str(e)}")
            self.is_kernel_fitted = False
        
        # Fit transformer component
        try:
            self.transformer_interpolator.fit(self.source_simulations)
            self.is_transformer_fitted = True
        except Exception as e:
            st.warning(f"Transformer component fitting failed: {str(e)}")
            self.transformer_weight = 0.0
            if self.is_kernel_fitted:
                self.kernel_weight = 1.0
            self.is_transformer_fitted = False
        
        return self
    
    def predict(self, target_params):
        """Make prediction using hybrid approach"""
        if not self.source_simulations:
            raise ValueError("No source simulations available")
        
        # Get predictions from each component
        kernel_pred = None
        transformer_pred = None
        kernel_weights = None
        transformer_weights = None
        
        if self.is_kernel_fitted and self.kernel_weight > 0:
            try:
                kernel_pred, kernel_weights = self.kernel_interpolator.predict(target_params)
            except Exception as e:
                st.warning(f"Kernel prediction failed: {str(e)}")
                self.kernel_weight = 0.0
        
        if self.is_transformer_fitted and self.transformer_weight > 0:
            try:
                transformer_pred, transformer_weights = self.transformer_interpolator.predict(target_params)
            except Exception as e:
                st.warning(f"Transformer prediction failed: {str(e)}")
                self.transformer_weight = 0.0
        
        # Normalize weights again in case some components failed
        total = self.kernel_weight + self.transformer_weight
        if total > 0:
            kernel_weight = self.kernel_weight / total
            transformer_weight = self.transformer_weight / total
        else:
            raise ValueError("Both interpolation components failed")
        
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
                'source_count': len(self.source_simulations)
            }
            
            return hybrid_pred, {
                'kernel_weights': kernel_weights,
                'transformer_weights': transformer_weights,
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
            return transformer_pred, {'transformer_weights': transformer_weights}
        
        else:
            raise ValueError("No interpolation components available for prediction")

# =============================================
# UNIFIED INTERPOLATOR MANAGER
# =============================================
class UnifiedInterpolatorManager:
    """Manager for all interpolation methods"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.methods = {
            'kernel': KernelRegressionInterpolator(),
            'transformer': SimpleTransformerInterpolator(device=device),
            'hybrid': HybridInterpolator(device=device)
        }
        self.current_method = 'kernel'
        self.source_simulations = []
        
    def set_method(self, method):
        """Set current interpolation method"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        self.current_method = method
        return self
    
    def load_source_simulations(self, source_simulations):
        """Load source simulations for interpolation"""
        # Filter out simulations that failed to load
        valid_simulations = [s for s in source_simulations if s.get('loaded_successfully', False)]
        
        if not valid_simulations:
            st.warning("No valid simulations found. Please check your data files.")
            return self
        
        self.source_simulations = valid_simulations
        
        # Fit/train each method
        try:
            if self.current_method == 'kernel':
                self.methods['kernel'].fit(valid_simulations)
            
            elif self.current_method == 'transformer':
                self.methods['transformer'].fit(valid_simulations)
            
            elif self.current_method == 'hybrid':
                self.methods['hybrid'].fit(valid_simulations)
            
            st.success(f"✅ Successfully loaded {len(valid_simulations)} simulations for {self.current_method} interpolation")
            
        except Exception as e:
            st.error(f"❌ Error fitting {self.current_method} model: {str(e)}")
            print(traceback.format_exc())
        
        return self
    
    def predict(self, target_params):
        """Make prediction using current method"""
        if not self.source_simulations:
            raise ValueError("No source simulations available")
        
        if self.current_method == 'kernel':
            return self.methods['kernel'].predict(target_params)
        
        elif self.current_method == 'transformer':
            return self.methods['transformer'].predict(target_params)
        
        elif self.current_method == 'hybrid':
            return self.methods['hybrid'].predict(target_params)
        
        else:
            raise ValueError(f"Unknown method: {self.current_method}")

# =============================================
# ENHANCED INTERFACE WITH ALL METHODS
# =============================================
def create_unified_interpolation_interface():
    """Create unified interface with all three interpolation methods"""
    
    st.header("🧬 Unified Stress Field Interpolation System")
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        st.sidebar.success("⚡ GPU available for transformer")
    else:
        st.sidebar.info("💻 Using CPU")
    
    # Initialize unified manager in session state
    if 'unified_manager' not in st.session_state:
        st.session_state.unified_manager = UnifiedInterpolatorManager(device=device)
    
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    # Initialize source simulations
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
    
    # Sidebar configuration
    st.sidebar.header("🔧 Interpolation Method Selection")
    
    method = st.sidebar.selectbox(
        "Choose Interpolation Method",
        [
            "⚡ Kernel Regression (Fast & Simple)",
            "🧠 Transformer Attention (Advanced)",
            "🧬 Hybrid Kernel-Transformer (Balanced)"
        ],
        index=0,
        help="Select the interpolation method based on your needs"
    )
    
    # Map display name to method key
    method_map = {
        "⚡ Kernel Regression (Fast & Simple)": 'kernel',
        "🧠 Transformer Attention (Advanced)": 'transformer',
        "🧬 Hybrid Kernel-Transformer (Balanced)": 'hybrid'
    }
    
    selected_method = method_map[method]
    st.session_state.unified_manager.set_method(selected_method)
    
    # Method-specific settings
    with st.sidebar.expander("⚙️ Method Configuration", expanded=True):
        if selected_method == 'kernel':
            kernel_type = st.selectbox(
                "Kernel Type",
                ["rbf", "matern", "rational_quadratic"],
                index=0
            )
            
            length_scale = st.slider(
                "Kernel Length Scale",
                0.05, 1.0, 0.3, 0.05
            )
            
            if st.button("🔄 Configure Kernel"):
                st.session_state.unified_manager.methods['kernel'] = KernelRegressionInterpolator(
                    kernel_type=kernel_type,
                    length_scale=length_scale
                )
                if st.session_state.source_simulations:
                    st.session_state.unified_manager.methods['kernel'].fit(
                        st.session_state.source_simulations
                    )
                st.success("Kernel configuration updated!")
        
        elif selected_method == 'transformer':
            d_model = st.slider("Model Dimension", 32, 256, 64, 16)
            num_heads = st.selectbox("Attention Heads", [2, 4, 8], index=1)
            
            if st.button("🔄 Configure Transformer"):
                st.session_state.unified_manager.methods['transformer'] = SimpleTransformerInterpolator(
                    d_model=d_model,
                    num_heads=num_heads,
                    device=device
                )
                if st.session_state.source_simulations:
                    st.session_state.unified_manager.methods['transformer'].fit(
                        st.session_state.source_simulations
                    )
                st.success("Transformer configuration updated!")
        
        else:  # hybrid
            col1, col2 = st.columns(2)
            
            with col1:
                kernel_weight = st.slider(
                    "Kernel Weight",
                    0.0, 1.0, 0.5, 0.1
                )
            
            with col2:
                transformer_weight = st.slider(
                    "Transformer Weight",
                    0.0, 1.0, 0.5, 0.1
                )
            
            if st.button("🔄 Configure Hybrid"):
                st.session_state.unified_manager.methods['hybrid'] = HybridInterpolator(
                    kernel_weight=kernel_weight,
                    transformer_weight=transformer_weight,
                    device=device
                )
                if st.session_state.source_simulations:
                    st.session_state.unified_manager.methods['hybrid'].fit(
                        st.session_state.source_simulations
                    )
                st.success("Hybrid configuration updated!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Load Data", 
        "🎯 Configure Target", 
        "🚀 Predict", 
        "📊 Results & Analysis",
        "📁 File Management"
    ])
    
    with tab1:
        st.subheader("Load Source Simulation Files")
        
        # File loading interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📂 From Numerical Solutions")
            
            # Scan directory
            if os.path.exists(NUMERICAL_SOLUTIONS_DIR):
                pkl_files = glob.glob(os.path.join(NUMERICAL_SOLUTIONS_DIR, "*.pkl"))
                
                if not pkl_files:
                    st.warning("No .pkl files found in numerical solutions directory")
                else:
                    file_options = {os.path.basename(f): f for f in pkl_files}
                    selected_files = st.multiselect(
                        "Select simulation files",
                        options=list(file_options.keys()),
                        key="select_numerical_unified"
                    )
                    
                    if selected_files and st.button("📥 Load Selected Files"):
                        with st.spinner("Loading and standardizing files..."):
                            loaded_count = 0
                            for filename in selected_files:
                                file_path = file_options[filename]
                                try:
                                    raw_data = st.session_state.data_loader.load_simulation_file(file_path)
                                    if raw_data is not None:
                                        sim_data = st.session_state.data_loader.standardize_simulation_data(
                                            raw_data, file_path
                                        )
                                        st.session_state.source_simulations.append(sim_data)
                                        loaded_count += 1
                                except Exception as e:
                                    st.error(f"Error loading {filename}: {str(e)}")
                            
                            if loaded_count > 0:
                                st.success(f"Loaded {loaded_count} new files!")
                                st.rerun()
            else:
                st.warning(f"Directory not found: {NUMERICAL_SOLUTIONS_DIR}")
        
        with col2:
            st.markdown("### 📤 Upload Files")
            
            uploaded_files = st.file_uploader(
                "Upload simulation files",
                type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'json'],
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("📥 Process Uploaded Files"):
                with st.spinner("Processing uploads..."):
                    for uploaded_file in uploaded_files:
                        try:
                            if uploaded_file.name.endswith('.pkl'):
                                raw_data = pickle.loads(uploaded_file.getvalue())
                            elif uploaded_file.name.endswith('.pt'):
                                raw_data = torch.load(BytesIO(uploaded_file.getvalue()), map_location='cpu')
                            elif uploaded_file.name.endswith(('.h5', '.hdf5')):
                                raw_data = {}
                                with h5py.File(BytesIO(uploaded_file.getvalue()), 'r') as f:
                                    def load_item(name, obj):
                                        if isinstance(obj, h5py.Dataset):
                                            raw_data[name] = obj[()]
                                    f.visititems(load_item)
                            elif uploaded_file.name.endswith('.npz'):
                                raw_data = dict(np.load(BytesIO(uploaded_file.getvalue()), allow_pickle=True))
                            elif uploaded_file.name.endswith('.json'):
                                raw_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
                            else:
                                st.warning(f"Unsupported file format: {uploaded_file.name}")
                                continue
                            
                            sim_data = st.session_state.data_loader.standardize_simulation_data(
                                raw_data, uploaded_file.name
                            )
                            st.session_state.source_simulations.append(sim_data)
                            st.success(f"✅ Loaded: {uploaded_file.name}")
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Display loaded simulations
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Simulations")
            
            # Filter successful loads
            successful_sims = [s for s in st.session_state.source_simulations if s.get('loaded_successfully', False)]
            failed_sims = [s for s in st.session_state.source_simulations if not s.get('loaded_successfully', False)]
            
            if successful_sims:
                # Create summary table
                summary_data = []
                for i, sim_data in enumerate(successful_sims):
                    params = sim_data.get('params', {})
                    summary_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': f"{params.get('eps0', 'N/A'):.3f}",
                        'κ': f"{params.get('kappa', 'N/A'):.3f}",
                        'Orientation': str(params.get('orientation', 'Unknown'))[:30],
                        'Frames': len(sim_data.get('history', [])),
                        'Status': '✅'
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                # Load into unified manager
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🔗 Load into Interpolator"):
                        st.session_state.unified_manager.load_source_simulations(
                            st.session_state.source_simulations
                        )
                with col2:
                    st.info(f"**Valid simulations:** {len(successful_sims)} | **Failed:** {len(failed_sims)}")
            
            if failed_sims:
                with st.expander("❌ Failed Loads"):
                    for sim_data in failed_sims:
                        st.error(f"{sim_data.get('filename', 'Unknown')}: {sim_data.get('error', 'Unknown error')}")
            
            if st.button("🗑️ Clear All Simulations", type="secondary"):
                st.session_state.source_simulations = []
                st.success("All simulations cleared!")
                st.rerun()
    
    with tab2:
        st.subheader("Configure Target Parameters")
        
        if not st.session_state.source_simulations:
            st.warning("⚠️ Please load source simulations first")
        else:
            # Get parameter ranges from loaded simulations
            param_ranges = {
                'defect_types': set(),
                'shapes': set(),
                'eps0_min': float('inf'),
                'eps0_max': float('-inf'),
                'kappa_min': float('inf'),
                'kappa_max': float('-inf')
            }
            
            for sim_data in st.session_state.source_simulations:
                if not sim_data.get('loaded_successfully', False):
                    continue
                    
                params = sim_data.get('params', {})
                
                if 'defect_type' in params:
                    param_ranges['defect_types'].add(params['defect_type'])
                if 'shape' in params:
                    param_ranges['shapes'].add(params['shape'])
                if 'eps0' in params:
                    eps0 = params['eps0']
                    if isinstance(eps0, (int, float)):
                        param_ranges['eps0_min'] = min(param_ranges['eps0_min'], eps0)
                        param_ranges['eps0_max'] = max(param_ranges['eps0_max'], eps0)
                if 'kappa' in params:
                    kappa = params['kappa']
                    if isinstance(kappa, (int, float)):
                        param_ranges['kappa_min'] = min(param_ranges['kappa_min'], kappa)
                        param_ranges['kappa_max'] = max(param_ranges['kappa_max'], kappa)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Use available defect types or default
                if param_ranges['defect_types']:
                    target_defect = st.selectbox(
                        "Target Defect Type",
                        list(param_ranges['defect_types']),
                        index=0
                    )
                else:
                    target_defect = st.selectbox(
                        "Target Defect Type",
                        ["ISF", "ESF", "Twin"],
                        index=0
                    )
                
                # Use available shapes or default
                if param_ranges['shapes']:
                    target_shape = st.selectbox(
                        "Target Shape",
                        list(param_ranges['shapes']),
                        index=0
                    )
                else:
                    target_shape = st.selectbox(
                        "Target Shape",
                        ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                        index=0
                    )
                
                # Use computed eps0 range or default
                if param_ranges['eps0_max'] > param_ranges['eps0_min']:
                    target_eps0 = st.slider(
                        "Target ε*",
                        float(param_ranges['eps0_min']),
                        float(param_ranges['eps0_max']),
                        float((param_ranges['eps0_min'] + param_ranges['eps0_max']) / 2),
                        0.01
                    )
                else:
                    target_eps0 = st.slider(
                        "Target ε*",
                        0.3, 3.0, 1.414, 0.01
                    )
            
            with col2:
                # Use computed kappa range or default
                if param_ranges['kappa_max'] > param_ranges['kappa_min']:
                    target_kappa = st.slider(
                        "Target κ",
                        float(param_ranges['kappa_min']),
                        float(param_ranges['kappa_max']),
                        float((param_ranges['kappa_min'] + param_ranges['kappa_max']) / 2),
                        0.05
                    )
                else:
                    target_kappa = st.slider(
                        "Target κ",
                        0.1, 2.0, 0.7, 0.05
                    )
                
                orientation_mode = st.radio(
                    "Orientation",
                    ["Predefined", "Custom Angle"],
                    horizontal=True
                )
                
                if orientation_mode == "Predefined":
                    target_orientation = st.selectbox(
                        "Select Orientation",
                        ["Horizontal {111} (0°)", 
                         "Tilted 30° (1¯10 projection)", 
                         "Tilted 60°", 
                         "Vertical {111} (90°)"],
                        index=0
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
                        "Custom Angle (degrees)",
                        0.0, 90.0, 45.0, 0.5
                    )
                    target_theta = np.deg2rad(target_angle)
                    
                    # Map to orientation string
                    if 0 <= target_angle <= 15:
                        target_orientation = 'Horizontal {111} (0°)'
                    elif 15 < target_angle <= 45:
                        target_orientation = 'Tilted 30° (1¯10 projection)'
                    elif 45 < target_angle <= 75:
                        target_orientation = 'Tilted 60°'
                    else:
                        target_orientation = 'Vertical {111} (90°)'
                
                st.info(f"**θ:** {np.rad2deg(target_theta):.1f}°")
            
            # Store target parameters
            st.session_state.target_params = {
                'defect_type': target_defect,
                'shape': target_shape,
                'eps0': target_eps0,
                'kappa': target_kappa,
                'orientation': target_orientation,
                'theta': target_theta
            }
            
            # Show parameter summary
            st.subheader("🎯 Target Parameters Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Defect Type", target_defect)
                st.metric("Shape", target_shape)
                st.metric("ε*", f"{target_eps0:.3f}")
            with col2:
                st.metric("κ", f"{target_kappa:.3f}")
                st.metric("Orientation", target_orientation)
                st.metric("Angle", f"{np.rad2deg(target_theta):.1f}°")
    
    with tab3:
        st.subheader("Make Predictions")
        
        if not st.session_state.source_simulations:
            st.warning("⚠️ Please load source simulations first")
        elif 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure target parameters first")
        else:
            st.info(f"**Current Method:** {method}")
            
            # Prediction button
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Running prediction..."):
                    try:
                        predicted_stress, weights = st.session_state.unified_manager.predict(
                            st.session_state.target_params
                        )
                        
                        st.session_state.prediction_results = {
                            'stress_fields': predicted_stress,
                            'weights': weights,
                            'method': selected_method,
                            'target_params': st.session_state.target_params,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.success("✅ Prediction complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        print(traceback.format_exc())
    
    with tab4:
        st.subheader("Results & Analysis")
        
        if 'prediction_results' not in st.session_state:
            st.info("👈 Run a prediction first")
        else:
            results = st.session_state.prediction_results
            
            # Show method info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", results.get('method', 'Unknown').upper())
            with col2:
                valid_sims = len([s for s in st.session_state.source_simulations 
                                 if s.get('loaded_successfully', False)])
                st.metric("Source Simulations", valid_sims)
            with col3:
                if results.get('method') == 'hybrid' and 'weights' in results:
                    weights = results['weights']
                    if 'kernel_weight' in weights:
                        st.metric("Kernel Weight", f"{weights['kernel_weight']:.2f}")
            
            # Display stress fields
            st.subheader("🎯 Predicted Stress Fields")
            
            stress_fields = results['stress_fields']
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            titles = ['Hydrostatic Stress (GPa)', 'Stress Magnitude (GPa)', 'Von Mises Stress (GPa)']
            components = ['sigma_hydro', 'sigma_mag', 'von_mises']
            cmaps = ['coolwarm', 'viridis', 'plasma']
            
            # Get grid extent
            def get_grid_extent(N=128, dx=0.1):
                return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
            
            extent = get_grid_extent()
            
            for ax, title, comp, cmap in zip(axes, titles, components, cmaps):
                if comp in stress_fields:
                    data = stress_fields[comp]
                    # Ensure data is 2D
                    if data.ndim > 2:
                        data = data.squeeze()
                    
                    im = ax.imshow(data, extent=extent, cmap=cmap,
                                  origin='lower', aspect='equal')
                    ax.set_title(title)
                    ax.set_xlabel('x (nm)')
                    ax.set_ylabel('y (nm)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(title)
            
            st.pyplot(fig)
            
            # Show weights or attention
            if 'weights' in results:
                st.subheader("🔍 Weight Analysis")
                
                weights_data = results['weights']
                
                if 'kernel_weights' in weights_data:
                    # Kernel weights
                    kernel_weights = weights_data['kernel_weights']
                    
                    fig_weights, ax = plt.subplots(figsize=(10, 4))
                    x_pos = np.arange(len(kernel_weights))
                    bars = ax.bar(x_pos, kernel_weights, alpha=0.7, color='steelblue')
                    ax.set_xlabel('Source Simulation')
                    ax.set_ylabel('Kernel Weight')
                    ax.set_title('Kernel Regression Weights')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'S{i+1}' for i in range(len(kernel_weights))])
                    
                    for bar, weight in zip(bars, kernel_weights):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
                    
                    st.pyplot(fig_weights)
                
                if 'transformer_weights' in weights_data:
                    # Transformer attention weights
                    transformer_weights = weights_data['transformer_weights']
                    
                    fig_attn, ax = plt.subplots(figsize=(10, 4))
                    x_pos = np.arange(len(transformer_weights))
                    bars = ax.bar(x_pos, transformer_weights, alpha=0.7, color='orange')
                    ax.set_xlabel('Source Simulation')
                    ax.set_ylabel('Attention Weight')
                    ax.set_title('Transformer Attention Weights')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'S{i+1}' for i in range(len(transformer_weights))])
                    
                    for bar, weight in zip(bars, transformer_weights):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
                    
                    st.pyplot(fig_attn)
            
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
                if st.button("💾 Save Prediction"):
                    export_data = {
                        'prediction': results,
                        'source_count': len(st.session_state.source_simulations),
                        'target_params': st.session_state.target_params,
                        'method': results.get('method', 'unknown'),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    filename = f"prediction_{results.get('method', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
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
                    label="📥 Download Results",
                    data=export_buffer,
                    file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
            
            with col3:
                # Report generation
                report = f"""
                STRESS FIELD PREDICTION REPORT
                ================================
                
                Method: {results.get('method', 'Unknown')}
                Timestamp: {datetime.now().isoformat()}
                Source Simulations: {len(st.session_state.source_simulations)}
                
                TARGET PARAMETERS:
                ------------------
                Defect Type: {st.session_state.target_params.get('defect_type', 'Unknown')}
                Shape: {st.session_state.target_params.get('shape', 'Unknown')}
                ε*: {st.session_state.target_params.get('eps0', 'Unknown')}
                κ: {st.session_state.target_params.get('kappa', 'Unknown')}
                Orientation: {st.session_state.target_params.get('orientation', 'Unknown')}
                """
                
                for stat in stats_data:
                    report += f"\n{stat['Component']}:\n"
                    report += f"  Max: {stat['Max (GPa)']:.3f} GPa\n"
                    report += f"  Min: {stat['Min (GPa)']:.3f} GPa\n"
                    report += f"  Mean: {stat['Mean (GPa)']:.3f} GPa\n"
                    report += f"  Std Dev: {stat['Std Dev']:.3f}\n"
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with tab5:
        st.subheader("📁 File Management")
        
        # File management interface
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
        
        if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
            st.warning(f"Directory does not exist: {NUMERICAL_SOLUTIONS_DIR}")
            if st.button("📁 Create Directory"):
                os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
                st.success("Directory created!")
                st.rerun()
        else:
            # List files
            files = glob.glob(os.path.join(NUMERICAL_SOLUTIONS_DIR, "*"))
            
            if not files:
                st.warning("No files found")
            else:
                file_data = []
                for file in files:
                    try:
                        stat = os.stat(file)
                        file_data.append({
                            'Filename': os.path.basename(file),
                            'Size (KB)': stat.st_size // 1024,
                            'Modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                            'Type': os.path.splitext(file)[1][1:] or 'Unknown'
                        })
                    except:
                        continue
                
                df_files = pd.DataFrame(file_data)
                st.dataframe(df_files, use_container_width=True)
                
                # File actions
                if file_data:
                    selected_file = st.selectbox(
                        "Select file for action",
                        options=[f['Filename'] for f in file_data]
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🗑️ Delete File"):
                            file_path = os.path.join(NUMERICAL_SOLUTIONS_DIR, selected_file)
                            try:
                                os.remove(file_path)
                                st.success(f"Deleted {selected_file}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {str(e)}")
                    
                    with col2:
                        if st.button("🔄 Refresh"):
                            st.rerun()
                    
                    with col3:
                        if st.button("📥 Download All as ZIP"):
                            # Create zip file
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for file in files:
                                    zip_file.write(file, os.path.basename(file))
                            
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                label="Click to Download",
                                data=zip_buffer,
                                file_name="numerical_solutions_backup.zip",
                                mime="application/zip"
                            )

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application with unified interpolation methods"""
    
    # Display title
    st.title("🧬 Unified Stress Field Interpolation System")
    st.markdown("### Three Methods: Kernel Regression • Transformer Attention • Hybrid")
    
    # Sidebar info
    st.sidebar.title("📊 System Info")
    
    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"**Device:** {device}")
    st.sidebar.info(f"**Solutions Directory:**\n`{NUMERICAL_SOLUTIONS_DIR}`")
    
    # Method descriptions
    with st.sidebar.expander("📚 Method Descriptions"):
        st.markdown("""
        **⚡ Kernel Regression:**
        - Fast & simple Gaussian RBF interpolation
        - No training required
        - Best for quick estimates
        
        **🧠 Transformer Attention:**
        - Advanced attention mechanism
        - Learns parameter relationships
        - Good for complex patterns
        
        **🧬 Hybrid Kernel-Transformer:**
        - Combines both methods
        - Weighted ensemble
        - Balances speed and accuracy
        """)
    
    # Create the interface
    create_unified_interpolation_interface()
    
    # Theory section
    with st.expander("🔬 **Theory: Interpolation Methods Comparison**", expanded=False):
        st.markdown("""
        ## **Three Interpolation Methods for Stress Fields**
        
        ### **1. ⚡ Kernel Regression**
        
        **Mathematical Formulation:**
        ```python
        # Gaussian RBF Kernel
        K(x, x') = exp(-||x - x'||² / (2σ²))
        
        # Prediction (Nadaraya-Watson estimator)
        f̂(x) = Σ_i w_i * f(x_i)
        w_i = K(x, x_i) / Σ_j K(x, x_j)
        ```
        
        **Advantages:**
        - **Fast computation:** O(N²) for N source simulations
        - **No training required:** Ready to use immediately
        - **Interpretable:** Weights show source contributions
        - **Theoretically sound:** Well-established statistical method
        
        **Limitations:**
        - **Fixed similarity metric:** Gaussian kernel may not capture complex relationships
        
        ### **2. 🧠 Transformer Attention**
        
        **Architecture:**
        ```python
        # Attention mechanism
        Attention(Q, K, V) = softmax(Q·Kᵀ/√d_k) · V
        
        # Where:
        # Q: Target parameter queries
        # K: Source parameter keys
        # V: Source stress values
        ```
        
        **Innovations:**
        - **Parameter-aware:** Learns complex parameter-stress relationships
        - **Flexible similarity:** Learns optimal similarity metric from data
        - **Non-linear mapping:** Can capture complex patterns
        
        **Training Requirements:**
        - **Data:** Multiple source simulations for meaningful learning
        - **Hardware:** Works on CPU or GPU
        
        ### **3. 🧬 Hybrid Kernel-Transformer**
        
        **Ensemble Strategy:**
        ```python
        # Weighted combination
        f_hybrid(x) = α * f_kernel(x) + β * f_transformer(x)
        
        # Where α + β = 1
        # Adaptive weights can be learned or set manually
        ```
        
        **Benefits:**
        - **Robustness:** Combines strengths of both methods
        - **Fallback mechanism:** If one method fails, other provides backup
        - **Flexible weighting:** Adjust based on problem characteristics
        
        ### **🧪 Performance Comparison**
        
        | Metric | Kernel | Transformer | Hybrid |
        |--------|--------|-------------|--------|
        | **Speed** | ⚡⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡ |
        | **Accuracy** | ⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ |
        | **Training Required** | No | Yes | Partial |
        | **Interpretability** | High | Medium | Medium |
        
        ### **🎯 Application Guidelines**
        
        **Use Kernel Regression when:**
        - You need quick predictions
        - You have limited computational resources
        - Interpretability is important
        
        **Use Transformer when:**
        - You need good accuracy
        - You have sufficient data
        - Complex parameter-stress relationships exist
        
        **Use Hybrid when:**
        - You want balanced performance
        - You're unsure which method is best
        - You need robustness and fallback options
        """)

if __name__ == "__main__":
    main()
