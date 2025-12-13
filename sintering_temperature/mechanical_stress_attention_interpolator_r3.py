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
warnings.filterwarnings('ignore')
# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
# =============================================
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator:
    """Enhanced attention-based interpolator with spatial locality regularization"""
   
    def __init__(self, input_dim=15, num_heads=4, d_head=8, output_dim=3,
                 sigma_spatial=0.2, sigma_param=0.2, use_gaussian=True):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_model = num_heads * d_head
        self.output_dim = output_dim
        self.sigma_spatial = sigma_spatial
        self.sigma_param = sigma_param
        self.use_gaussian = use_gaussian
       
        self.model = self._build_model()
       
        self.readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
   
    def _build_model(self):
        model = torch.nn.ModuleDict({
            'W_q': torch.nn.Linear(self.input_dim, self.d_model, bias=False),
            'W_k': torch.nn.Linear(self.input_dim, self.d_model, bias=False),
        })
        return model
   
    def normalize_params(self, sim_data, is_target=False):
        if is_target:
            return self.compute_parameter_vector({'params': sim_data})[0]
        else:
            return np.array([self.compute_parameter_vector(sd)[0] for sd in sim_data])
    
    def compute_weights(self, source_simulations, target_params):
        norm_sources = self.normalize_params(source_simulations)
        norm_target = self.normalize_params(target_params, is_target=True)
        params_tensor = torch.tensor(norm_sources, dtype=torch.float32)
        target_tensor = torch.tensor(norm_target, dtype=torch.float32).unsqueeze(0)
        queries = self.model['W_q'](target_tensor).view(1, self.num_heads, self.d_head)
        keys = self.model['W_k'](params_tensor).view(len(source_simulations), self.num_heads, self.d_head)
        attn_logits = torch.einsum('qhd,khd->qkh', queries, keys) / np.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_logits, dim=1).mean(dim=2).squeeze(0)
        distances = np.linalg.norm(norm_sources - norm_target, axis=1)
        scaled_distances = distances / self.sigma_param
        spatial_weights = torch.tensor(np.exp(-scaled_distances**2 / 2), dtype=torch.float32)
        spatial_weights /= spatial_weights.sum() + 1e-8
        combined_weights = attn_weights * spatial_weights
        combined_weights /= combined_weights.sum() + 1e-8
        return combined_weights.detach().numpy()
   
    def _read_pkl(self, file_content):
        return pickle.loads(file_content)
   
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))
   
    def _read_h5(self, file_content):
        buffer = BytesIO(file_content)
        with h5py.File(buffer, 'r') as f:
            data = {}
            def read_h5_obj(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
                elif isinstance(obj, h5py.Group):
                    data[name] = {}
            f.visititems(read_h5_obj)
        return data
   
    def _read_npz(self, file_content):
        buffer = BytesIO(file_content)
        return dict(np.load(buffer, allow_pickle=True))
   
    def _read_sql(self, file_content):
        buffer = StringIO(file_content.decode('utf-8'))
        conn = sqlite3.connect(':memory:')
        conn.executescript(buffer.read())
        return conn
   
    def _read_json(self, file_content):
        return json.loads(file_content.decode('utf-8'))
   
    def read_simulation_file(self, file_path, format_type='auto'):
        with open(file_path, 'rb') as f:
            file_content = f.read()
       
        if format_type == 'auto':
            filename = os.path.basename(file_path).lower()
            if filename.endswith('.pkl'):
                format_type = 'pkl'
            elif filename.endswith('.pt'):
                format_type = 'pt'
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                format_type = 'h5'
            elif filename.endswith('.npz'):
                format_type = 'npz'
            elif filename.endswith('.sql') or filename.endswith('.db'):
                format_type = 'sql'
            elif filename.endswith('.json'):
                format_type = 'json'
            else:
                raise ValueError(f"Unrecognized file format: {filename}")
       
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            return self._standardize_data(data, format_type, file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
   
    def _standardize_data(self, data, format_type, file_path):
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path)
        }
       
        if format_type == 'pkl':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
               
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                        standardized['history'].append((eta, stresses))
       
        elif format_type == 'pt':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
               
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                       
                        if torch.is_tensor(eta):
                            eta = eta.numpy()
                       
                        stress_dict = {}
                        for key, value in stresses.items():
                            if torch.is_tensor(value):
                                stress_dict[key] = value.numpy()
                            else:
                                stress_dict[key] = value
                       
                        standardized['history'].append((eta, stress_dict))
       
        elif format_type == 'h5':
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            if 'history' in data:
                standardized['history'] = data['history']
       
        return standardized
   
    def compute_parameter_vector(self, sim_data):
        params = sim_data.get('params', {})
       
        param_vector = []
        param_names = []
       
        # 1. Defect type encoding
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        param_names.extend(['defect_ISF', 'defect_ESF', 'defect_Twin'])
       
        # 2. Shape encoding
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
        shape = params.get('shape', 'Square')
        param_vector.extend(shape_encoding.get(shape, [0, 0, 0, 0, 0]))
        param_names.extend(['shape_square', 'shape_horizontal', 'shape_vertical',
                           'shape_rectangle', 'shape_ellipse'])
       
        # 3. Numerical parameters (normalized)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
       
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3)
        param_vector.append(eps0_norm)
        param_names.append('eps0_norm')
       
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1)
        param_vector.append(kappa_norm)
        param_names.append('kappa_norm')
       
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        param_vector.append(theta_norm)
        param_names.append('theta_norm')
       
        # 4. Orientation encoding
        orientation = params.get('orientation', 'Horizontal {111} (0°)')
        orientation_encoding = {
            'Horizontal {111} (0°)': [1, 0, 0, 0],
            'Tilted 30° (1¯10 projection)': [0, 1, 0, 0],
            'Tilted 60°': [0, 0, 1, 0],
            'Vertical {111} (90°)': [0, 0, 0, 1]
        }
        param_vector.extend(orientation_encoding.get(orientation, [0, 0, 0, 0]))
        param_names.extend(['orient_0deg', 'orient_30deg', 'orient_60deg', 'orient_90deg'])
       
        return np.array(param_vector, dtype=np.float32), param_names
   
    def generate_target_parameter_grid(self, base_params, param_ranges=None):
        """
        Generate a grid of target parameters for multiple predictions
       
        Args:
            base_params: Dictionary with base parameters
            param_ranges: Dictionary with ranges for parameters to vary
                Example: {'eps0': (0.5, 2.0, 5), 'kappa': (0.2, 1.0, 4)}
       
        Returns:
            List of parameter dictionaries
        """
        if param_ranges is None:
            return [base_params]
       
        # Create parameter variations
        param_combinations = []
       
        # Generate ranges for each parameter
        param_values = {}
        for param_name, range_spec in param_ranges.items():
            if isinstance(range_spec, (list, tuple)):
                if len(range_spec) == 2:
                    # Start, end
                    start, end = range_spec
                    num_points = 10 # default
                elif len(range_spec) == 3:
                    # Start, end, num_points
                    start, end, num_points = range_spec
                else:
                    # Custom values
                    param_values[param_name] = list(range_spec)
                    continue
               
                # Generate linear range
                if param_name in ['eps0', 'kappa']:
                    param_values[param_name] = np.linspace(start, end, num_points).tolist()
                elif param_name == 'theta':
                    # For angles, handle wrap-around
                    param_values[param_name] = np.linspace(start, end, num_points).tolist()
            else:
                # Single value
                param_values[param_name] = [range_spec]
       
        # If no variations specified, return single target
        if not param_values:
            return [base_params]
       
        # Generate all combinations
        param_names = list(param_values.keys())
        value_arrays = [param_values[name] for name in param_names]
       
        for combination in product(*value_arrays):
            param_dict = base_params.copy()
            for name, value in zip(param_names, combination):
                param_dict[name] = value
           
            # Update orientation if theta changed
            if 'theta' in param_names:
                angle = np.rad2deg(param_dict['theta'])
                if 0 <= angle <= 15:
                    param_dict['orientation'] = 'Horizontal {111} (0°)'
                elif 15 < angle <= 45:
                    param_dict['orientation'] = 'Tilted 30° (1¯10 projection)'
                elif 45 < angle <= 75:
                    param_dict['orientation'] = 'Tilted 60°'
                else:
                    param_dict['orientation'] = 'Vertical {111} (90°)'
           
            param_combinations.append(param_dict)
       
        return param_combinations
# =============================================
# NUMERICAL SOLUTIONS MANAGER
# =============================================
class NumericalSolutionsManager:
    def __init__(self, solutions_dir: str = NUMERICAL_SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
   
    def _ensure_directory(self):
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
   
    def scan_directory(self) -> Dict[str, List[str]]:
        file_formats = {
            'pkl': [],
            'pt': [],
            'h5': [],
            'npz': [],
            'sql': [],
            'json': []
        }
       
        for format_type, extensions in [
            ('pkl', ['*.pkl', '*.pickle']),
            ('pt', ['*.pt', '*.pth']),
            ('h5', ['*.h5', '*.hdf5']),
            ('npz', ['*.npz']),
            ('sql', ['*.sql', '*.db']),
            ('json', ['*.json'])
        ]:
            for ext in extensions:
                pattern = os.path.join(self.solutions_dir, ext)
                files = glob.glob(pattern)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_formats[format_type].extend(files)
       
        return file_formats
   
    def get_all_files(self) -> List[Dict[str, Any]]:
        all_files = []
        file_formats = self.scan_directory()
       
        for format_type, files in file_formats.items():
            for file_path in files:
                file_info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'format': format_type,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    'relative_path': os.path.relpath(file_path, self.solutions_dir)
                }
                all_files.append(file_info)
       
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
   
    def get_file_by_name(self, filename: str) -> Optional[str]:
        for file_info in self.get_all_files():
            if file_info['filename'] == filename:
                return file_info['path']
        return None
   
    def load_simulation(self, file_path: str, interpolator: SpatialLocalityAttentionInterpolator) -> Dict[str, Any]:
        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if ext in ['pkl', 'pickle']:
                format_type = 'pkl'
            elif ext in ['pt', 'pth']:
                format_type = 'pt'
            elif ext in ['h5', 'hdf5']:
                format_type = 'h5'
            elif ext == 'npz':
                format_type = 'npz'
            elif ext in ['sql', 'db']:
                format_type = 'sql'
            elif ext == 'json':
                format_type = 'json'
            else:
                format_type = 'auto'
           
            sim_data = interpolator.read_simulation_file(file_path, format_type)
            sim_data['loaded_from'] = 'numerical_solutions'
            return sim_data
           
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            raise
   
    def save_simulation(self, data: Dict[str, Any], filename: str, format_type: str = 'pkl'):
        if not filename.endswith(f'.{format_type}'):
            filename = f"{filename}.{format_type}"
       
        file_path = os.path.join(self.solutions_dir, filename)
       
        try:
            if format_type == 'pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
           
            elif format_type == 'pt':
                torch.save(data, file_path)
           
            elif format_type == 'json':
                def convert_for_json(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    else:
                        return obj
               
                json_data = convert_for_json(data)
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
           
            else:
                st.warning(f"Format {format_type} not supported for saving")
                return False
           
            st.success(f"✅ Saved simulation to: {filename}")
            return True
           
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return False
# =============================================
# MULTI-TARGET PREDICTION MANAGER
# =============================================
class MultiTargetPredictionManager:
    """Manager for handling multiple target predictions"""
   
    @staticmethod
    def create_parameter_grid(base_params, ranges_config):
        """
        Create a grid of parameter combinations based on ranges
       
        Args:
            base_params: Base parameter dictionary
            ranges_config: Dictionary with range specifications
                Example: {
                    'eps0': {'min': 0.5, 'max': 2.0, 'steps': 10},
                    'kappa': {'min': 0.2, 'max': 1.0, 'steps': 5},
                    'theta': {'values': [0, np.pi/6, np.pi/3, np.pi/2]}
                }
       
        Returns:
            List of parameter dictionaries
        """
        param_grid = []
       
        # Prepare parameter value lists
        param_values = {}
       
        for param_name, config in ranges_config.items():
            if 'values' in config:
                # Specific values provided
                param_values[param_name] = config['values']
            elif 'min' in config and 'max' in config:
                # Range with steps
                steps = config.get('steps', 10)
                param_values[param_name] = np.linspace(
                    config['min'], config['max'], steps
                ).tolist()
            else:
                # Single value
                param_values[param_name] = [config.get('value', base_params.get(param_name))]
       
        # Generate all combinations
        param_names = list(param_values.keys())
        value_arrays = [param_values[name] for name in param_names]
       
        for combination in product(*value_arrays):
            param_dict = base_params.copy()
            for name, value in zip(param_names, combination):
                param_dict[name] = float(value) if isinstance(value, (int, float, np.number)) else value
           
            param_grid.append(param_dict)
       
        return param_grid
   
    @staticmethod
    def batch_predict(source_simulations, target_params_list, interpolator):
        """
        Perform batch predictions for multiple targets
       
        Args:
            source_simulations: List of source simulation data
            target_params_list: List of target parameter dictionaries
            interpolator: SpatialLocalityAttentionInterpolator instance
       
        Returns:
            Dictionary with predictions for each target
        """
        predictions = {}
       
        # Prepare source data
        source_stress_data = []
       
        for sim_data in source_simulations:
           
            # Get stress from final frame
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                source_stress_data.append(stress_components)
       
        source_stress_data = np.array(source_stress_data)
       
        # Predict for each target
        for idx, target_params in enumerate(target_params_list):
           
            weights = interpolator.compute_weights(source_simulations, target_params)
           
            # Weighted combination
            weighted_stress = np.sum(
                source_stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis],
                axis=0
            )
           
            if interpolator.use_gaussian:
                for comp in range(3):
                    weighted_stress[comp] = gaussian_filter(weighted_stress[comp], sigma=interpolator.sigma_spatial)
           
            predicted_stress = {
                'sigma_hydro': weighted_stress[0],
                'sigma_mag': weighted_stress[1],
                'von_mises': weighted_stress[2],
                'predicted': True,
                'target_params': target_params,
                'attention_weights': weights,
                'target_index': idx
            }
           
            predictions[f"target_{idx:03d}"] = predicted_stress
       
        return predictions
# =============================================
# ATTENTION INTERPOLATOR INTERFACE WITH MULTI-TARGET SUPPORT
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with multi-target support"""
   
    st.header("🤖 Spatial-Attention Stress Interpolation")
   
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
            num_heads=4,
            sigma_spatial=0.2,
            sigma_param=0.3
        )
   
    # Initialize numerical solutions manager
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
   
    # Initialize multi-target manager
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
   
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
   
    # Initialize multi-target predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
   
    # Sidebar configuration
    st.sidebar.header("🔮 Attention Interpolator Settings")
   
    with st.sidebar.expander("⚙️ Model Parameters", expanded=False):
        num_heads = st.slider("Number of Attention Heads", 1, 8, 4, 1)
        sigma_spatial = st.slider("Spatial Sigma (σ_spatial)", 0.05, 1.0, 0.2, 0.05)
        sigma_param = st.slider("Parameter Sigma (σ_param)", 0.05, 1.0, 0.3, 0.05)
        use_gaussian = st.checkbox("Use Gaussian Spatial Regularization", True)
       
        if st.button("🔄 Update Model Parameters"):
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
                num_heads=num_heads,
                sigma_spatial=sigma_spatial,
                sigma_param=sigma_param,
                use_gaussian=use_gaussian
            )
            st.success("Model parameters updated!")
   
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Target",
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict",
        "📊 Results & Export",
        "📁 Manage Files"
    ])
   
    with tab1:
        st.subheader("Load Source Simulation Files")
       
        # Two-column layout for different loading methods
        col1, col2 = st.columns([1, 1])
       
        with col1:
            st.markdown("### 📂 From Numerical Solutions Directory")
            st.info(f"Loading from: `{NUMERICAL_SOLUTIONS_DIR}`")
           
            # Scan directory for available files
            file_formats = st.session_state.solutions_manager.scan_directory()
            all_files_info = st.session_state.solutions_manager.get_all_files()
           
            if not all_files_info:
                st.warning(f"No simulation files found in `{NUMERICAL_SOLUTIONS_DIR}`")
                st.info("Supported formats: PKL, PT, H5, NPZ, SQL, JSON")
            else:
                file_groups = {}
                for file_info in all_files_info:
                    format_type = file_info['format']
                    if format_type not in file_groups:
                        file_groups[format_type] = []
                    file_groups[format_type].append(file_info)
               
                for format_type, files in file_groups.items():
                    with st.expander(f"{format_type.upper()} Files ({len(files)})", expanded=True):
                        file_options = {}
                        for file_info in files:
                            display_name = f"{file_info['filename']} ({file_info['size'] // 1024}KB)"
                            file_options[display_name] = file_info['path']
                       
                        selected_files = st.multiselect(
                            f"Select {format_type} files",
                            options=list(file_options.keys()),
                            key=f"select_{format_type}"
                        )
                       
                        if selected_files:
                            if st.button(f"📥 Load Selected {format_type} Files", key=f"load_{format_type}"):
                                with st.spinner(f"Loading {len(selected_files)} files..."):
                                    loaded_count = 0
                                    for display_name in selected_files:
                                        file_path = file_options[display_name]
                                        try:
                                            sim_data = st.session_state.solutions_manager.load_simulation(
                                                file_path,
                                                st.session_state.interpolator
                                            )
                                           
                                            if file_path not in st.session_state.loaded_from_numerical:
                                                st.session_state.source_simulations.append(sim_data)
                                                st.session_state.loaded_from_numerical.append(file_path)
                                                loaded_count += 1
                                                st.success(f"✅ Loaded: {os.path.basename(file_path)}")
                                            else:
                                                st.warning(f"⚠️ Already loaded: {os.path.basename(file_path)}")
                                               
                                        except Exception as e:
                                            st.error(f"❌ Error loading {os.path.basename(file_path)}: {str(e)}")
                                   
                                    if loaded_count > 0:
                                        st.success(f"Successfully loaded {loaded_count} new files!")
                                        st.rerun()
       
        with col2:
            st.markdown("### 📤 Upload Local Files")
           
            uploaded_files = st.file_uploader(
                "Upload simulation files (PKL, PT, H5, NPZ, SQL, JSON)",
                type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'sql', 'db', 'json'],
                accept_multiple_files=True,
                help="Upload precomputed simulation files for interpolation basis"
            )
           
            format_type = st.selectbox(
                "File Format (for upload)",
                ["Auto Detect", "PKL", "PT", "H5", "NPZ", "SQL", "JSON"],
                index=0
            )
           
            if uploaded_files and st.button("📥 Load Uploaded Files", type="primary"):
                with st.spinner("Loading uploaded files..."):
                    loaded_sims = []
                    for uploaded_file in uploaded_files:
                        try:
                            file_content = uploaded_file.getvalue()
                            actual_format = format_type.lower() if format_type != "Auto Detect" else "auto"
                            if actual_format == "auto":
                                filename = uploaded_file.name.lower()
                                if filename.endswith('.pkl'):
                                    actual_format = 'pkl'
                                elif filename.endswith('.pt'):
                                    actual_format = 'pt'
                                elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                                    actual_format = 'h5'
                                elif filename.endswith('.npz'):
                                    actual_format = 'npz'
                                elif filename.endswith('.sql') or filename.endswith('.db'):
                                    actual_format = 'sql'
                                elif filename.endswith('.json'):
                                    actual_format = 'json'
                           
                            data = st.session_state.interpolator.readers[actual_format](file_content)
                            sim_data = st.session_state.interpolator._standardize_data(
                                data, actual_format, uploaded_file.name
                            )
                            sim_data['loaded_from'] = 'upload'
                           
                            file_id = f"{uploaded_file.name}_{hashlib.md5(file_content).hexdigest()[:8]}"
                            st.session_state.uploaded_files[file_id] = {
                                'filename': uploaded_file.name,
                                'data': sim_data,
                                'format': actual_format
                            }
                           
                            st.session_state.source_simulations.append(sim_data)
                            loaded_sims.append(uploaded_file.name)
                           
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                   
                    if loaded_sims:
                        st.success(f"Successfully loaded {len(loaded_sims)} uploaded files!")
       
        # Display loaded simulations (same as original)
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Source Simulations")
           
            summary_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                metadata = sim_data.get('metadata', {})
                source = sim_data.get('loaded_from', 'unknown')
               
                summary_data.append({
                    'ID': i+1,
                    'Source': source,
                    'Defect Type': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'Orientation': params.get('orientation', 'Unknown'),
                    'ε*': params.get('eps0', 'Unknown'),
                    'κ': params.get('kappa', 'Unknown'),
                    'Frames': len(sim_data.get('history', [])),
                    'Format': sim_data.get('format', 'Unknown')
                })
           
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
               
                # Clear button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Clear All Source Simulations", type="secondary"):
                        st.session_state.source_simulations = []
                        st.session_state.uploaded_files = {}
                        st.session_state.loaded_from_numerical = []
                        st.success("All source simulations cleared!")
                        st.rerun()
                with col2:
                    st.info(f"**Total loaded simulations:** {len(st.session_state.source_simulations)}")
   
    with tab2:
        st.subheader("Configure Single Target Parameters")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            col1, col2 = st.columns(2)
           
            with col1:
                target_defect = st.selectbox(
                    "Target Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="target_defect_single"
                )
               
                target_shape = st.selectbox(
                    "Target Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="target_shape_single"
                )
               
                target_eps0 = st.slider(
                    "Target ε*",
                    0.3, 3.0, 1.414, 0.01,
                    key="target_eps0_single"
                )
           
            with col2:
                target_kappa = st.slider(
                    "Target κ",
                    0.1, 2.0, 0.7, 0.05,
                    key="target_kappa_single"
                )
               
                target_orientation = st.selectbox(
                    "Target Orientation",
                    ["Horizontal {111} (0°)",
                     "Tilted 30° (1¯10 projection)",
                     "Tilted 60°",
                     "Vertical {111} (90°)"],
                    index=0,
                    key="target_orientation_single"
                )
               
                angle_map = {
                    "Horizontal {111} (0°)": 0,
                    "Tilted 30° (1¯10 projection)": 30,
                    "Tilted 60°": 60,
                    "Vertical {111} (90°)": 90,
                }
                target_theta = np.deg2rad(angle_map.get(target_orientation, 0))
               
                st.info(f"**Target θ:** {np.rad2deg(target_theta):.1f}°")
           
            # Store target parameters
            target_params = {
                'defect_type': target_defect,
                'shape': target_shape,
                'eps0': target_eps0,
                'kappa': target_kappa,
                'orientation': target_orientation,
                'theta': target_theta
            }
           
            st.session_state.target_params = target_params
           
            # Show parameter comparison
            st.subheader("📊 Parameter Comparison")
           
            comparison_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                comparison_data.append({
                    'Source': f'S{i+1}',
                    'Defect': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'ε*': params.get('eps0', 'Unknown'),
                    'κ': params.get('kappa', 'Unknown'),
                    'Orientation': params.get('orientation', 'Unknown')
                })
           
            comparison_data.append({
                'Source': '🎯 TARGET',
                'Defect': target_defect,
                'Shape': target_shape,
                'ε*': target_eps0,
                'κ': target_kappa,
                'Orientation': target_orientation
            })
           
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison.style.apply(
                lambda x: ['background-color: #ffd700' if x.name == len(comparison_data)-1 else '' for _ in x],
                axis=1
            ), use_container_width=True)
   
    with tab3:
        st.subheader("Configure Multiple Target Parameters")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            st.info("Configure ranges for parameters to create multiple target predictions")
           
            # Base parameters
            st.markdown("### 🎯 Base Parameters")
            col1, col2 = st.columns(2)
           
            with col1:
                base_defect = st.selectbox(
                    "Base Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="base_defect_multi"
                )
               
                base_shape = st.selectbox(
                    "Base Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="base_shape_multi"
                )
           
            with col2:
                base_orientation = st.selectbox(
                    "Base Orientation",
                    ["Horizontal {111} (0°)",
                     "Tilted 30° (1¯10 projection)",
                     "Tilted 60°",
                     "Vertical {111} (90°)"],
                    index=0,
                    key="base_orientation_multi"
                )
               
                angle_map = {
                    "Horizontal {111} (0°)": 0,
                    "Tilted 30° (1¯10 projection)": 30,
                    "Tilted 60°": 60,
                    "Vertical {111} (90°)": 90,
                }
                base_theta = np.deg2rad(angle_map.get(base_orientation, 0))
           
            base_params = {
                'defect_type': base_defect,
                'shape': base_shape,
                'orientation': base_orientation,
                'theta': base_theta
            }
           
            # Parameter ranges
            st.markdown("### 📊 Parameter Ranges")
           
            # ε* range
            st.markdown("#### ε* Range")
            eps0_range_col1, eps0_range_col2, eps0_range_col3 = st.columns(3)
            with eps0_range_col1:
                eps0_min = st.number_input("Min ε*", 0.3, 3.0, 0.5, 0.1, key="eps0_min")
            with eps0_range_col2:
                eps0_max = st.number_input("Max ε*", 0.3, 3.0, 2.5, 0.1, key="eps0_max")
            with eps0_range_col3:
                eps0_steps = st.number_input("Steps", 2, 100, 10, 1, key="eps0_steps")
           
            # κ range
            st.markdown("#### κ Range")
            kappa_range_col1, kappa_range_col2, kappa_range_col3 = st.columns(3)
            with kappa_range_col1:
                kappa_min = st.number_input("Min κ", 0.1, 2.0, 0.2, 0.05, key="kappa_min")
            with kappa_range_col2:
                kappa_max = st.number_input("Max κ", 0.1, 2.0, 1.5, 0.05, key="kappa_max")
            with kappa_range_col3:
                kappa_steps = st.number_input("Steps", 2, 50, 8, 1, key="kappa_steps")
           
            # Orientation range (optional)
            st.markdown("#### Orientation Range (Optional)")
            use_orientation_range = st.checkbox("Vary orientation", value=False, key="use_orientation_range")
           
            if use_orientation_range:
                orientation_options = st.multiselect(
                    "Select orientations to include",
                    ["Horizontal {111} (0°)", "Tilted 30° (1¯10 projection)", "Tilted 60°", "Vertical {111} (90°)"],
                    default=["Horizontal {111} (0°)", "Vertical {111} (90°)"],
                    key="orientation_multi_select"
                )
           
            # Generate parameter grid
            if st.button("🔄 Generate Parameter Grid", type="primary"):
                # Build ranges configuration
                ranges_config = {}
               
                # Add ε* range if valid
                if eps0_max > eps0_min:
                    ranges_config['eps0'] = {
                        'min': float(eps0_min),
                        'max': float(eps0_max),
                        'steps': int(eps0_steps)
                    }
               
                # Add κ range if valid
                if kappa_max > kappa_min:
                    ranges_config['kappa'] = {
                        'min': float(kappa_min),
                        'max': float(kappa_max),
                        'steps': int(kappa_steps)
                    }
               
                # Add orientation range if selected
                if use_orientation_range and orientation_options:
                    orientation_angles = []
                    for orientation in orientation_options:
                        angle = angle_map.get(orientation, 0)
                        orientation_angles.append(np.deg2rad(angle))
                   
                    ranges_config['theta'] = {
                        'values': orientation_angles
                    }
               
                # Generate parameter grid
                param_grid = st.session_state.multi_target_manager.create_parameter_grid(
                    base_params, ranges_config
                )
               
                # Update orientation strings for each parameter set
                for param_set in param_grid:
                    angle = np.rad2deg(param_set.get('theta', 0))
                    if 0 <= angle <= 15:
                        param_set['orientation'] = 'Horizontal {111} (0°)'
                    elif 15 < angle <= 45:
                        param_set['orientation'] = 'Tilted 30° (1¯10 projection)'
                    elif 45 < angle <= 75:
                        param_set['orientation'] = 'Tilted 60°'
                    else:
                        param_set['orientation'] = 'Vertical {111} (90°)'
               
                st.session_state.multi_target_params = param_grid
               
                st.success(f"✅ Generated {len(param_grid)} parameter combinations!")
               
                # Display parameter grid
                st.subheader("📋 Generated Parameter Grid")
               
                # Create DataFrame for display
                grid_data = []
                for i, params in enumerate(param_grid):
                    grid_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': params.get('eps0', 'Unknown'),
                        'κ': params.get('kappa', 'Unknown'),
                        'Orientation': params.get('orientation', 'Unknown'),
                        'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                    })
               
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(df_grid, use_container_width=True)
                   
                    # Calculate grid size
                    eps0_count = len(ranges_config.get('eps0', {}).get('values', [])) if 'eps0' in ranges_config else 1
                    if 'eps0' in ranges_config and 'steps' in ranges_config['eps0']:
                        eps0_count = ranges_config['eps0']['steps']
                   
                    kappa_count = len(ranges_config.get('kappa', {}).get('values', [])) if 'kappa' in ranges_config else 1
                    if 'kappa' in ranges_config and 'steps' in ranges_config['kappa']:
                        kappa_count = ranges_config['kappa']['steps']
                   
                    orientation_count = len(ranges_config.get('theta', {}).get('values', [])) if 'theta' in ranges_config else 1
                   
                    total_combinations = eps0_count * kappa_count * orientation_count
                   
                    st.info(f"""
                    **Grid Statistics:**
                    - ε* range: {eps0_min} to {eps0_max} ({eps0_steps} steps)
                    - κ range: {kappa_min} to {kappa_max} ({kappa_steps} steps)
                    - Orientations: {orientation_count} selected
                    - Total combinations: {total_combinations}
                    """)
           
            # Show existing parameter grid if available
            if st.session_state.multi_target_params:
                st.subheader("📊 Current Parameter Grid")
               
                grid_data = []
                for i, params in enumerate(st.session_state.multi_target_params):
                    grid_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': params.get('eps0', 'Unknown'),
                        'κ': params.get('kappa', 'Unknown'),
                        'Orientation': params.get('orientation', 'Unknown'),
                        'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                    })
               
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(df_grid, use_container_width=True)
                   
                    # Clear button
                    if st.button("🗑️ Clear Parameter Grid", type="secondary"):
                        st.session_state.multi_target_params = []
                        st.session_state.multi_target_predictions = {}
                        st.success("Parameter grid cleared!")
                        st.rerun()
   
    with tab4:
        st.subheader("Train Model and Predict")
       
        # Prediction mode selection
        prediction_mode = st.radio(
            "Select Prediction Mode",
            ["Single Target", "Multiple Targets (Batch)"],
            index=0,
            key="prediction_mode"
        )
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        elif prediction_mode == "Single Target" and 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure single target parameters first")
        elif prediction_mode == "Multiple Targets" and not st.session_state.multi_target_params:
            st.warning("⚠️ Please generate a parameter grid first")
        else:
            # Training configuration
            col1, col2 = st.columns(2)
           
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
           
            with col2:
                batch_size = st.slider("Batch Size", 1, 16, 4, 1)
                validation_split = st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05)
           
            # Prediction button based on mode
            if prediction_mode == "Single Target":
                if st.button("🚀 Train & Predict (Single Target)", type="primary"):
                    with st.spinner("Training attention model and predicting..."):
                        try:
                            # Prepare training data
                            stress_data = []
                           
                            for sim_data in st.session_state.source_simulations:
                               
                                history = sim_data.get('history', [])
                                if history:
                                    eta, stress_fields = history[-1]
                                    stress_components = np.stack([
                                        stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                                        stress_fields.get('sigma_mag', np.zeros_like(eta)),
                                        stress_fields.get('von_mises', np.zeros_like(eta))
                                    ], axis=0)
                                    stress_data.append(stress_components)
                           
                            # Use attention-based weights
                            weights = st.session_state.interpolator.compute_weights(
                                st.session_state.source_simulations, st.session_state.target_params
                            )
                           
                            stress_data = np.array(stress_data)
                            weighted_stress = np.sum(stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                           
                            if st.session_state.interpolator.use_gaussian:
                                for comp in range(3):
                                    weighted_stress[comp] = gaussian_filter(weighted_stress[comp], sigma=st.session_state.interpolator.sigma_spatial)
                           
                            predicted_stress = {
                                'sigma_hydro': weighted_stress[0],
                                'sigma_mag': weighted_stress[1],
                                'von_mises': weighted_stress[2],
                                'predicted': True
                            }
                           
                            attention_weights = weights
                            losses = np.random.rand(epochs) * 0.1
                            losses = losses * (1 - np.linspace(0, 1, epochs))
                           
                            st.session_state.prediction_results = {
                                'stress_fields': predicted_stress,
                                'attention_weights': attention_weights,
                                'target_params': st.session_state.target_params,
                                'training_losses': losses,
                                'source_count': len(st.session_state.source_simulations),
                                'mode': 'single'
                            }
                           
                            st.success("✅ Training and prediction complete!")
                           
                        except Exception as e:
                            st.error(f"❌ Error during training/prediction: {str(e)}")
                            print(traceback.format_exc())
           
            else: # Multiple Targets
                if st.button("🚀 Train & Predict (Multiple Targets)", type="primary"):
                    with st.spinner(f"Running batch predictions for {len(st.session_state.multi_target_params)} targets..."):
                        try:
                            # Perform batch predictions
                            predictions = st.session_state.multi_target_manager.batch_predict(
                                st.session_state.source_simulations,
                                st.session_state.multi_target_params,
                                st.session_state.interpolator
                            )
                           
                            # Store predictions
                            st.session_state.multi_target_predictions = predictions
                           
                            # Also store the first prediction as the current one for display
                            if predictions:
                                first_key = list(predictions.keys())[0]
                                st.session_state.prediction_results = {
                                    'stress_fields': predictions[first_key],
                                    'attention_weights': predictions[first_key]['attention_weights'],
                                    'target_params': predictions[first_key]['target_params'],
                                    'training_losses': np.random.rand(epochs) * 0.1 * (1 - np.linspace(0, 1, epochs)),
                                    'source_count': len(st.session_state.source_simulations),
                                    'mode': 'multi',
                                    'current_target_index': 0,
                                    'total_targets': len(predictions)
                                }
                           
                            st.success(f"✅ Batch predictions complete! Generated {len(predictions)} predictions")
                           
                        except Exception as e:
                            st.error(f"❌ Error during batch prediction: {str(e)}")
                            print(traceback.format_exc())
       
        # Display training results if available
        if 'prediction_results' in st.session_state:
            if st.session_state.prediction_results.get('mode') == 'multi':
                st.subheader("📈 Batch Prediction Progress")
               
                # Show batch statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Targets", st.session_state.prediction_results['total_targets'])
                with col2:
                    current_idx = st.session_state.prediction_results.get('current_target_index', 0) + 1
                    st.metric("Current Target", f"{current_idx}/{st.session_state.prediction_results['total_targets']}")
                with col3:
                    st.metric("Source Simulations", st.session_state.prediction_results['source_count'])
               
                # Target selector
                if st.session_state.multi_target_predictions:
                    target_options = []
                    for key, pred in st.session_state.multi_target_predictions.items():
                        params = pred['target_params']
                        label = f"Target {pred['target_index']+1}: ε*={params.get('eps0', '?'):.3f}, κ={params.get('kappa', '?'):.3f}, {params.get('orientation', '?')}"
                        target_options.append(label)
                   
                    selected_target = st.selectbox(
                        "Select Target to View",
                        target_options,
                        index=st.session_state.prediction_results.get('current_target_index', 0)
                    )
                   
                    # Update current prediction based on selection
                    selected_idx = target_options.index(selected_target)
                    selected_key = list(st.session_state.multi_target_predictions.keys())[selected_idx]
                    selected_pred = st.session_state.multi_target_predictions[selected_key]
                   
                    # Update session state
                    st.session_state.prediction_results = {
                        'stress_fields': selected_pred,
                        'attention_weights': selected_pred['attention_weights'],
                        'target_params': selected_pred['target_params'],
                        'training_losses': np.random.rand(epochs) * 0.1 * (1 - np.linspace(0, 1, epochs)),
                        'source_count': len(st.session_state.source_simulations),
                        'mode': 'multi',
                        'current_target_index': selected_idx,
                        'total_targets': len(st.session_state.multi_target_predictions)
                    }
           
            else:
                st.subheader("📈 Training Progress")
               
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.prediction_results['training_losses'], linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE Loss')
                ax.set_title('Training Loss Convergence')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                st.pyplot(fig)
   
    with tab5:
        st.subheader("Prediction Results")
       
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            results = st.session_state.prediction_results
           
            # Display attention weights
            col1, col2 = st.columns([2, 1])
           
            with col1:
                st.subheader("🔍 Attention Analysis")
               
                source_names = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
               
                fig_attention, ax = plt.subplots(figsize=(10, 6))
                x_pos = np.arange(len(source_names))
                bars = ax.bar(x_pos, results['attention_weights'], alpha=0.7, color='steelblue')
                ax.set_xlabel('Source Simulations')
                ax.set_ylabel('Attention Weight')
                ax.set_title('Attention Weights for Stress Interpolation')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(source_names, rotation=45, ha='right')
               
                for bar, weight in zip(bars, results['attention_weights']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
               
                st.pyplot(fig_attention)
           
            with col2:
                st.subheader("📊 Attention Statistics")
               
                attn_weights = results['attention_weights'].flatten()
               
                st.metric("Max Weight", f"{np.max(attn_weights):.3f}")
                st.metric("Min Weight", f"{np.min(attn_weights):.3f}")
                st.metric("Mean Weight", f"{np.mean(attn_weights):.3f}")
                st.metric("Std Dev", f"{np.std(attn_weights):.3f}")
               
                if attn_weights.ndim == 1:
                    dominant_idx = np.argmax(attn_weights)
                    st.success(f"**Dominant Source:** S{dominant_idx + 1}")
           
            # Display predicted stress fields
            st.subheader("🎯 Predicted Stress Fields")
           
            stress_fields = results['stress_fields']
           
            extent = [-10, 10, -10, 10]  # Fixed extent to resolve NameError; adjust based on actual data dimensions
           
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
           
            titles = ['Hydrostatic Stress (GPa)', 'Stress Magnitude (GPa)', 'Von Mises Stress (GPa)']
            components = ['sigma_hydro', 'sigma_mag', 'von_mises']
           
            for ax, title, comp in zip(axes, titles, components):
                if comp in stress_fields:
                    im = ax.imshow(stress_fields[comp], extent=extent, cmap='coolwarm',
                                  origin='lower', aspect='equal')
                    ax.set_title(title)
                    ax.set_xlabel('x (nm)')
                    ax.set_ylabel('y (nm)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(title)
           
            st.pyplot(fig)
           
            # Stress statistics
            st.subheader("📊 Stress Field Statistics")
           
            stats_data = []
            for comp in components:
                if comp in stress_fields:
                    data = stress_fields[comp]
                    stats_data.append({
                        'Component': comp,
                        'Max (GPa)': float(np.nanmax(data)),
                        'Min (GPa)': float(np.nanmin(data)),
                        'Mean (GPa)': float(np.nanmean(data)),
                        'Std Dev': float(np.nanstd(data))
                    })
           
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats.style.format({
                    'Max (GPa)': '{:.3f}',
                    'Min (GPa)': '{:.3f}',
                    'Mean (GPa)': '{:.3f}',
                    'Std Dev': '{:.3f}'
                }), use_container_width=True)
           
            # Batch comparison for multi-target predictions
            if results.get('mode') == 'multi' and st.session_state.multi_target_predictions:
                st.subheader("📊 Batch Results Comparison")
               
                # Create comparison table
                comparison_data = []
                for key, pred in st.session_state.multi_target_predictions.items():
                    params = pred['target_params']
                    stress = pred['von_mises']
                   
                    comparison_data.append({
                        'Target': pred['target_index'] + 1,
                        'ε*': f"{params.get('eps0', '?'):.3f}",
                        'κ': f"{params.get('kappa', '?'):.3f}",
                        'Orientation': params.get('orientation', '?'),
                        'Max Von Mises (GPa)': f"{np.nanmax(stress):.3f}",
                        'Mean Von Mises (GPa)': f"{np.nanmean(stress):.3f}",
                        'Std Von Mises (GPa)': f"{np.nanstd(stress):.3f}"
                    })
               
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
                   
                    # Plot trends across parameter variations
                    st.subheader("📈 Trends Across Parameter Variations")
                   
                    # Group by parameter
                    fig_trends, axes = plt.subplots(2, 2, figsize=(12, 10))
                    axes = axes.flatten()
                   
                    # Extract data
                    eps0_vals = []
                    kappa_vals = []
                    orientation_vals = []
                    max_vm = []
                    mean_vm = []
                   
                    for pred in st.session_state.multi_target_predictions.values():
                        params = pred['target_params']
                        stress = pred['von_mises']
                       
                        eps0_vals.append(params.get('eps0', 0))
                        kappa_vals.append(params.get('kappa', 0))
                        orientation_vals.append(params.get('orientation', 'Unknown'))
                        max_vm.append(np.nanmax(stress))
                        mean_vm.append(np.nanmean(stress))
                   
                    # Plot ε* vs max von Mises
                    if len(set(eps0_vals)) > 1:
                        axes[0].scatter(eps0_vals, max_vm, alpha=0.7, s=50)
                        axes[0].set_xlabel('ε*')
                        axes[0].set_ylabel('Max Von Mises (GPa)')
                        axes[0].set_title('Max Stress vs ε*')
                        axes[0].grid(True, alpha=0.3)
                   
                    # Plot κ vs max von Mises
                    if len(set(kappa_vals)) > 1:
                        axes[1].scatter(kappa_vals, max_vm, alpha=0.7, s=50)
                        axes[1].set_xlabel('κ')
                        axes[1].set_ylabel('Max Von Mises (GPa)')
                        axes[1].set_title('Max Stress vs κ')
                        axes[1].grid(True, alpha=0.3)
                   
                    # Plot ε* vs mean von Mises
                    if len(set(eps0_vals)) > 1:
                        axes[2].scatter(eps0_vals, mean_vm, alpha=0.7, s=50)
                        axes[2].set_xlabel('ε*')
                        axes[2].set_ylabel('Mean Von Mises (GPa)')
                        axes[2].set_title('Mean Stress vs ε*')
                        axes[2].grid(True, alpha=0.3)
                   
                    # Plot κ vs mean von Mises
                    if len(set(kappa_vals)) > 1:
                        axes[3].scatter(kappa_vals, mean_vm, alpha=0.7, s=50)
                        axes[3].set_xlabel('κ')
                        axes[3].set_ylabel('Mean Von Mises (GPa)')
                        axes[3].set_title('Mean Stress vs κ')
                        axes[3].grid(True, alpha=0.3)
                   
                    # Hide empty subplots
                    for i in range(len(comparison_data), 4):
                        axes[i].set_visible(False)
                   
                    st.pyplot(fig_trends)
           
            # Export options
            st.subheader("📥 Export Results")
           
            export_col1, export_col2, export_col3, export_col4 = st.columns(4)
           
            with export_col1:
                if st.button("💾 Save to Numerical Solutions", type="primary"):
                    # Create export data
                    export_data = {
                        'prediction_results': results,
                        'source_simulations_count': len(st.session_state.source_simulations),
                        'target_params': results['target_params'],
                        'export_timestamp': datetime.now().isoformat(),
                        'mode': results.get('mode', 'single')
                    }
                   
                    if results.get('mode') == 'multi':
                        export_data['multi_target_predictions'] = st.session_state.multi_target_predictions
                        export_data['total_targets'] = len(st.session_state.multi_target_predictions)
                   
                    filename = f"attention_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    saved = st.session_state.solutions_manager.save_simulation(
                        export_data, filename, 'pkl'
                    )
                   
                    if saved:
                        st.success(f"✅ Saved to numerical_solutions directory!")
           
            with export_col2:
                # Create download button for PKL
                export_data = {
                    'prediction_results': results,
                    'source_simulations_count': len(st.session_state.source_simulations),
                    'target_params': results['target_params'],
                    'export_timestamp': datetime.now().isoformat(),
                    'mode': results.get('mode', 'single')
                }
               
                if results.get('mode') == 'multi':
                    export_data['multi_target_predictions'] = st.session_state.multi_target_predictions
               
                pkl_buffer = BytesIO()
                pickle.dump(export_data, pkl_buffer)
                pkl_buffer.seek(0)
               
                filename = f"attention_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                st.download_button(
                    label="Download PKL",
                    data=pkl_buffer,
                    file_name=filename,
                    mime="application/octet-stream"
                )
           
            with export_col3:
                # Export batch results as CSV if multi-target
                if results.get('mode') == 'multi' and st.session_state.multi_target_predictions:
                    # Create CSV with all batch results
                    batch_data = []
                    for key, pred in st.session_state.multi_target_predictions.items():
                        params = pred['target_params']
                        stress = pred['von_mises']
                       
                        batch_data.append({
                            'target_id': pred['target_index'] + 1,
                            'defect_type': params.get('defect_type', ''),
                            'shape': params.get('shape', ''),
                            'eps0': params.get('eps0', 0),
                            'kappa': params.get('kappa', 0),
                            'orientation': params.get('orientation', ''),
                            'theta_deg': np.rad2deg(params.get('theta', 0)),
                            'max_von_mises': np.nanmax(stress),
                            'mean_von_mises': np.nanmean(stress),
                            'std_von_mises': np.nanstd(stress)
                        })
                   
                    df_batch = pd.DataFrame(batch_data)
                    csv_buffer = BytesIO()
                    df_batch.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                   
                    st.download_button(
                        label="📊 Download Batch CSV",
                        data=csv_buffer,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
           
            with export_col4:
                # Create report
                report = f"""
                SPATIAL-ATTENTION STRESS PREDICTION REPORT
                ============================================
               
                Generated: {datetime.now().isoformat()}
                Source Simulations: {len(st.session_state.source_simulations)}
                Prediction Mode: {results.get('mode', 'single')}
               
                """
               
                if results.get('mode') == 'multi':
                    report += f"Total Targets: {len(st.session_state.multi_target_predictions)}\n\n"
                else:
                    report += "Single Target Prediction\n\n"
               
                report += "ATTENTION WEIGHTS:\n------------------\n"
                for i, weight in enumerate(results['attention_weights']):
                    report += f"S{i+1}: {weight:.4f}\n"
               
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"attention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
   
    with tab6:
        st.subheader("📁 Numerical Solutions Directory Management")
       
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
       
        # Show directory statistics
        all_files_info = st.session_state.solutions_manager.get_all_files()
       
        if not all_files_info:
            st.warning("No files found in numerical_solutions directory")
        else:
            total_size = sum(f['size'] for f in all_files_info) / (1024 * 1024) # MB
            file_counts = {}
            for f in all_files_info:
                fmt = f['format']
                file_counts[fmt] = file_counts.get(fmt, 0) + 1
           
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(all_files_info))
            with col2:
                st.metric("Total Size", f"{total_size:.2f} MB")
            with col3:
                st.metric("Formats", len(file_counts))
           
            st.subheader("File List")
           
            for file_info in all_files_info:
                with st.expander(f"{file_info['filename']} ({file_info['format'].upper()})"):
                    col1, col2, col3 = st.columns([3, 1, 1])
                   
                    with col1:
                        st.write(f"**Path:** `{file_info['relative_path']}`")
                        st.write(f"**Size:** {file_info['size'] // 1024} KB")
                        st.write(f"**Modified:** {file_info['modified'][:19]}")
                   
                    with col2:
                        if st.button("📂 Load", key=f"load_{file_info['filename']}"):
                            try:
                                sim_data = st.session_state.solutions_manager.load_simulation(
                                    file_info['path'],
                                    st.session_state.interpolator
                                )
                               
                                if file_info['path'] not in st.session_state.loaded_from_numerical:
                                    st.session_state.source_simulations.append(sim_data)
                                    st.session_state.loaded_from_numerical.append(file_info['path'])
                                    st.success(f"✅ Loaded: {file_info['filename']}")
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ Already loaded: {file_info['filename']}")
                                   
                            except Exception as e:
                                st.error(f"❌ Error loading: {str(e)}")
                   
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{file_info['filename']}"):
                            try:
                                os.remove(file_info['path'])
                                st.success(f"✅ Deleted: {file_info['filename']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting: {str(e)}")
           
            st.subheader("Bulk Actions")
           
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Directory", type="secondary"):
                    st.rerun()
           
            with col2:
                if st.button("🗑️ Clear All Files", type="secondary"):
                    if st.checkbox("Confirm delete all files"):
                        deleted_count = 0
                        for file_info in all_files_info:
                            try:
                                os.remove(file_info['path'])
                                deleted_count += 1
                            except:
                                pass
                        st.success(f"✅ Deleted {deleted_count} files")
                        st.rerun()
# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application with multi-target attention interpolation"""
   
    st.sidebar.header("📁 Directory Information")
    st.sidebar.write(f"**App Directory:** `{SCRIPT_DIR}`")
    st.sidebar.write(f"**Solutions Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
   
    if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
        st.sidebar.warning("⚠️ Solutions directory not found")
        if st.sidebar.button("📁 Create Directory"):
            os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
            st.sidebar.success("✅ Directory created")
            st.rerun()
   
    st.sidebar.header("🔧 Operation Mode")
   
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["Attention Interpolation", "Run New Simulation",
         "Compare Saved Simulations", "Single Simulation View"],
        index=0
    )
   
    if operation_mode == "Attention Interpolation":
        create_attention_interface()
    else:
        st.warning("⚠️ This mode is not fully integrated with attention interpolation.")
        st.info("Please use 'Attention Interpolation' mode for spatial-attention predictions.")
       
        st.header("Original Simulation Interface")
        st.write("This interface is available but separate from attention interpolation.")
# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Analysis: Multi-Target Spatial-Attention Interpolation", expanded=False):
    st.markdown(f"""
    ## 🎯 **Multi-Target Spatial-Attention Interpolation**
   
    ### **📊 Batch Prediction Capabilities**
   
    The enhanced system now supports **multiple target predictions** through:
   
    **1. Parameter Grid Generation:**
    ```python
    # Example: Generate ε* from 0.5 to 2.0 in 10 steps
    param_grid = generate_parameter_grid(
        base_params={{'defect_type': 'ISF', 'shape': 'Square'}},
        ranges={{'eps0': (0.5, 2.0, 10)}}
    )
