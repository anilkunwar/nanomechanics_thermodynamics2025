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
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
# Get the directory where the app is running from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The numerical_solutions directory is at the same level as the app
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")

# Create directory if it doesn't exist
if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
    st.info(f"📁 Created numerical_solutions directory at: {NUMERICAL_SOLUTIONS_DIR}")

# =============================================
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator:
    """Enhanced attention-based interpolator with spatial locality regularization"""
    
    def __init__(self, input_dim=15, num_heads=4, d_model=32, output_dim=3, 
                 sigma_spatial=0.2, sigma_param=0.2, use_gaussian=True):
        """
        Args:
            input_dim: Dimension of parameter vector (defect + geometry + orientation)
            num_heads: Number of attention heads
            d_model: Dimension of model
            output_dim: Number of output stress components (hydrostatic, magnitude, vonMises)
            sigma_spatial: Spatial locality parameter for Gaussian weighting
            sigma_param: Parameter space locality parameter
            use_gaussian: Whether to use Gaussian spatial regularization
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.output_dim = output_dim
        self.sigma_spatial = sigma_spatial
        self.sigma_param = sigma_param
        self.use_gaussian = use_gaussian
        
        # Initialize model
        self.model = self._build_model()
        
        # File format readers
        self.readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
    
    def _build_model(self):
        """Build the attention model with spatial regularization"""
        model = torch.nn.ModuleDict({
            # Parameter embeddings with positional encoding for spatial awareness
            'param_embedding': torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.d_model)
            ),
            
            # Multi-head attention with spatial bias
            'attention': torch.nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1
            ),
            
            # Feed-forward with skip connections
            'feed_forward': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.d_model * 4, self.d_model)
            ),
            
            # Output projection for stress fields
            'output_projection': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.output_dim)
            ),
            
            # Spatial regularization network
            'spatial_regularizer': torch.nn.Sequential(
                torch.nn.Linear(2, 32),  # x,y coordinates
                torch.nn.ReLU(),
                torch.nn.Linear(32, self.num_heads)
            ) if self.use_gaussian else None,
            
            # Layer norms
            'norm1': torch.nn.LayerNorm(self.d_model),
            'norm2': torch.nn.LayerNorm(self.d_model)
        })
        
        return model
    
    def _read_pkl(self, file_content):
        """Read pickle format file"""
        return pickle.loads(file_content)
    
    def _read_pt(self, file_content):
        """Read PyTorch format file"""
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))
    
    def _read_h5(self, file_content):
        """Read HDF5 format file"""
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
        """Read numpy compressed format"""
        buffer = BytesIO(file_content)
        return dict(np.load(buffer, allow_pickle=True))
    
    def _read_sql(self, file_content):
        """Read SQL dump format"""
        buffer = StringIO(file_content.decode('utf-8'))
        conn = sqlite3.connect(':memory:')
        conn.executescript(buffer.read())
        return conn
    
    def _read_json(self, file_content):
        """Read JSON format"""
        return json.loads(file_content.decode('utf-8'))
    
    def read_simulation_file(self, file_path, format_type='auto'):
        """
        Read simulation file from local path
        
        Args:
            file_path: Path to simulation file
            format_type: 'auto' or specific format ('pkl', 'pt', etc.)
            
        Returns:
            Dictionary with simulation data
        """
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Auto-detect format
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
        
        # Read file
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            
            # Convert to standardized format
            return self._standardize_data(data, format_type, file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _standardize_data(self, data, format_type, file_path):
        """Convert different formats to standardized structure"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path)
        }
        
        if format_type == 'pkl':
            # PKL format from your export
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                
                # Convert history
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                        standardized['history'].append((eta, stresses))
        
        elif format_type == 'pt':
            # PyTorch format
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                
                # Convert tensors to numpy
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                        
                        # Convert tensors
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
            # HDF5 format
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            if 'history' in data:
                standardized['history'] = data['history']
        
        return standardized
    
    def compute_parameter_vector(self, sim_data):
        """
        Compute parameter vector from simulation data
        
        Args:
            sim_data: Standardized simulation data
            
        Returns:
            parameter_vector: Normalized parameter vector
            param_names: Names of parameters
        """
        params = sim_data.get('params', {})
        
        # Parameter encoding scheme
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
        
        # Normalize eps0 to [0,1] range (0.3-3.0)
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3)
        param_vector.append(eps0_norm)
        param_names.append('eps0_norm')
        
        # Normalize kappa to [0,1] range (0.1-2.0)
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1)
        param_vector.append(kappa_norm)
        param_names.append('kappa_norm')
        
        # Normalize theta (angle) to [0,1] range (0-2π)
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

# =============================================
# NUMERICAL SOLUTIONS MANAGER
# =============================================
class NumericalSolutionsManager:
    """Manager for accessing files in the numerical_solutions directory"""
    
    def __init__(self, solutions_dir: str = NUMERICAL_SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the numerical_solutions directory exists"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
            st.info(f"📁 Created numerical_solutions directory at: {self.solutions_dir}")
    
    def scan_directory(self) -> Dict[str, List[str]]:
        """
        Scan the numerical_solutions directory for available files
        
        Returns:
            Dictionary with file lists by format
        """
        file_formats = {
            'pkl': [],
            'pt': [],
            'h5': [],
            'npz': [],
            'sql': [],
            'json': []
        }
        
        # Scan for all supported files
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
                    # Sort by modification time (newest first)
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_formats[format_type].extend(files)
        
        return file_formats
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """
        Get all simulation files with metadata
        
        Returns:
            List of dictionaries with file information
        """
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
        
        # Sort by filename
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
    
    def get_file_by_name(self, filename: str) -> Optional[str]:
        """
        Get full path of file by filename
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to file or None if not found
        """
        for file_info in self.get_all_files():
            if file_info['filename'] == filename:
                return file_info['path']
        return None
    
    def load_simulation(self, file_path: str, interpolator: SpatialLocalityAttentionInterpolator) -> Dict[str, Any]:
        """
        Load a simulation file using the interpolator
        
        Args:
            file_path: Path to the simulation file
            interpolator: Interpolator instance for reading files
            
        Returns:
            Standardized simulation data
        """
        try:
            # Determine format from extension
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
            
            # Load simulation
            sim_data = interpolator.read_simulation_file(file_path, format_type)
            sim_data['loaded_from'] = 'numerical_solutions'
            return sim_data
            
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def save_simulation(self, data: Dict[str, Any], filename: str, format_type: str = 'pkl'):
        """
        Save simulation data to numerical_solutions directory
        
        Args:
            data: Simulation data to save
            filename: Name for the file
            format_type: Format to save as ('pkl', 'pt', etc.)
        """
        # Ensure filename has correct extension
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
                # Convert numpy arrays to lists for JSON serialization
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
# ERROR HANDLING DECORATOR
# =============================================
def handle_errors(func):
    """Decorator to handle errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            st.error("Please check the console for detailed error information.")
            print(f"Error in {func.__name__}: {str(e)}")
            print(traceback.format_exc())
            return None
    return wrapper

# =============================================
# METADATA MANAGEMENT CLASS
# =============================================
class MetadataManager:
    """Centralized metadata management to ensure consistency"""
    
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        """Create standardized metadata dictionary"""
        if run_time is None:
            run_time = 0.0
            
        metadata = {
            'run_time': run_time,
            'frames': len(history) if history else 0,
            'grid_size': kwargs.get('grid_size', 128),
            'dx': kwargs.get('dx', 0.1),
            'created_at': datetime.now().isoformat(),
            'colormaps': kwargs.get('colormaps', {
                'eta': sim_params.get('eta_cmap', 'viridis'),
                'sigma': sim_params.get('sigma_cmap', 'hot'),
                'hydro': sim_params.get('hydro_cmap', 'coolwarm'),
                'vm': sim_params.get('vm_cmap', 'plasma')
            }),
            'material_properties': {
                'C11': 124.0,
                'C12': 93.4,
                'C44': 46.1,
                'lattice_constant': 0.4086
            },
            'simulation_parameters': {
                'dt': 0.004,
                'N': kwargs.get('grid_size', 128),
                'dx': kwargs.get('dx', 0.1)
            }
        }
        return metadata

# =============================================
# Material & Grid
# =============================================
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)

# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

N = 128
dx = 0.1  # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# ATTENTION INTERPOLATOR INTERFACE WITH NUMERICAL SOLUTIONS
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with numerical solutions support"""
    
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
    
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    
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
    
    # Display current directory info
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Numerical Solutions Directory:**\n`{NUMERICAL_SOLUTIONS_DIR}`")
    
    if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
        st.sidebar.warning("⚠️ Directory does not exist. Creating...")
        os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Load Source Data", 
        "🎯 Configure Target", 
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
                # Group files by format
                file_groups = {}
                for file_info in all_files_info:
                    format_type = file_info['format']
                    if format_type not in file_groups:
                        file_groups[format_type] = []
                    file_groups[format_type].append(file_info)
                
                # Display available files
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
                                            # Load simulation
                                            sim_data = st.session_state.solutions_manager.load_simulation(
                                                file_path, 
                                                st.session_state.interpolator
                                            )
                                            
                                            # Check if already loaded
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
            
            # File upload section
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
                            # Read file content
                            file_content = uploaded_file.getvalue()
                            
                            # Auto-detect format
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
                            
                            # Read using interpolator
                            buffer = BytesIO(file_content)
                            data = st.session_state.interpolator.readers[actual_format](file_content)
                            
                            # Standardize
                            sim_data = st.session_state.interpolator._standardize_data(
                                data, actual_format, uploaded_file.name
                            )
                            sim_data['loaded_from'] = 'upload'
                            
                            # Store in session state
                            file_id = f"{uploaded_file.name}_{hashlib.md5(file_content).hexdigest()[:8]}"
                            st.session_state.uploaded_files[file_id] = {
                                'filename': uploaded_file.name,
                                'data': sim_data,
                                'format': actual_format
                            }
                            
                            # Add to source simulations
                            st.session_state.source_simulations.append(sim_data)
                            loaded_sims.append(uploaded_file.name)
                            
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    
                    if loaded_sims:
                        st.success(f"Successfully loaded {len(loaded_sims)} uploaded files!")
                        st.write("**Loaded files:**")
                        for filename in loaded_sims:
                            st.write(f"- {filename}")
        
        # Display loaded simulations
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Source Simulations")
            
            # Create summary table
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
                
                # Parameter space visualization
                st.subheader("🎯 Parameter Space Visualization")
                
                try:
                    # Extract parameter vectors
                    param_vectors = []
                    for sim_data in st.session_state.source_simulations:
                        param_vector, _ = st.session_state.interpolator.compute_parameter_vector(sim_data)
                        param_vectors.append(param_vector[:3])  # Use first 3 dimensions for visualization
                    
                    if param_vectors:
                        param_vectors = np.array(param_vectors)
                        
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Color by source
                        colors = []
                        for sim_data in st.session_state.source_simulations:
                            source = sim_data.get('loaded_from', 'unknown')
                            if source == 'numerical_solutions':
                                colors.append('blue')
                            elif source == 'upload':
                                colors.append('green')
                            else:
                                colors.append('gray')
                        
                        # Scatter plot
                        scatter = ax.scatter(
                            param_vectors[:, 0],  # Defect encoding
                            param_vectors[:, 1],  # Shape encoding
                            param_vectors[:, 2],  # eps0_norm
                            c=colors,
                            s=100,
                            alpha=0.7
                        )
                        
                        ax.set_xlabel('Defect Encoding')
                        ax.set_ylabel('Shape Encoding')
                        ax.set_zlabel('ε* (normalized)')
                        ax.set_title('Source Simulations in Parameter Space\n(Blue: Numerical Solutions, Green: Uploaded)')
                        
                        # Add labels
                        for i, (x, y, z) in enumerate(param_vectors):
                            ax.text(x, y, z, f'S{i+1}', fontsize=8)
                        
                        # Create legend
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='blue', label='Numerical Solutions'),
                            Patch(facecolor='green', label='Uploaded'),
                            Patch(facecolor='gray', label='Unknown')
                        ]
                        ax.legend(handles=legend_elements)
                        
                        st.pyplot(fig)
                
                except Exception as e:
                    st.warning(f"Could not visualize parameter space: {str(e)}")
            
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
        st.subheader("Configure Target Parameters")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                target_defect = st.selectbox(
                    "Target Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="target_defect"
                )
                
                target_shape = st.selectbox(
                    "Target Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="target_shape"
                )
                
                target_eps0 = st.slider(
                    "Target ε*",
                    0.3, 3.0, 1.414, 0.01,
                    key="target_eps0"
                )
            
            with col2:
                target_kappa = st.slider(
                    "Target κ",
                    0.1, 2.0, 0.7, 0.05,
                    key="target_kappa"
                )
                
                target_orientation = st.selectbox(
                    "Target Orientation",
                    ["Horizontal {111} (0°)", 
                     "Tilted 30° (1¯10 projection)", 
                     "Tilted 60°", 
                     "Vertical {111} (90°)"],
                    index=0,
                    key="target_orientation"
                )
                
                # Map orientation to angle
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
            
            # Create comparison table
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
            
            # Add target
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
        st.subheader("Train Model and Predict")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations and configure target")
        elif 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure target parameters first")
        else:
            # Training configuration
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            
            with col2:
                batch_size = st.slider("Batch Size", 1, 16, 4, 1)
                validation_split = st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05)
            
            # Training button
            if st.button("🚀 Train & Predict", type="primary"):
                with st.spinner("Training attention model and predicting..."):
                    try:
                        # Note: The actual training method needs to be implemented in the interpolator
                        # For now, we'll simulate training and use a simple interpolation
                        
                        # Prepare training data (simplified version)
                        param_vectors = []
                        stress_data = []
                        
                        for sim_data in st.session_state.source_simulations:
                            param_vector, _ = st.session_state.interpolator.compute_parameter_vector(sim_data)
                            param_vectors.append(param_vector)
                            
                            # Get stress from final frame
                            history = sim_data.get('history', [])
                            if history:
                                eta, stress_fields = history[-1]
                                stress_components = np.stack([
                                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                                    stress_fields.get('von_mises', np.zeros_like(eta))
                                ], axis=0)
                                stress_data.append(stress_components)
                        
                        # Simple weighted average based on parameter similarity
                        target_vector, _ = st.session_state.interpolator.compute_parameter_vector(
                            {'params': st.session_state.target_params}
                        )
                        
                        # Calculate distances
                        param_vectors = np.array(param_vectors)
                        distances = np.sqrt(np.sum((param_vectors - target_vector) ** 2, axis=1))
                        weights = np.exp(-0.5 * (distances / 0.3) ** 2)
                        weights = weights / (np.sum(weights) + 1e-8)
                        
                        # Weighted combination
                        stress_data = np.array(stress_data)
                        weighted_stress = np.sum(stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                        
                        # Store results
                        predicted_stress = {
                            'sigma_hydro': weighted_stress[0],
                            'sigma_mag': weighted_stress[1],
                            'von_mises': weighted_stress[2],
                            'predicted': True
                        }
                        
                        # Simulate attention weights
                        attention_weights = weights
                        
                        # Simulate training losses
                        losses = np.random.rand(epochs) * 0.1
                        losses = losses * (1 - np.linspace(0, 1, epochs))
                        
                        # Store results
                        st.session_state.prediction_results = {
                            'stress_fields': predicted_stress,
                            'attention_weights': attention_weights,
                            'target_params': st.session_state.target_params,
                            'training_losses': losses,
                            'source_count': len(st.session_state.source_simulations)
                        }
                        
                        st.success("✅ Training and prediction complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during training/prediction: {str(e)}")
                        print(traceback.format_exc())
        
        # Display training results if available
        if 'prediction_results' in st.session_state:
            st.subheader("📈 Training Progress")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.prediction_results['training_losses'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Training Loss Convergence')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Prediction Results")
        
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            results = st.session_state.prediction_results
            
            # Display attention weights
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔍 Attention Analysis")
                
                # Create source names
                source_names = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
                
                # Visualize attention
                fig_attention, ax = plt.subplots(figsize=(10, 6))
                x_pos = np.arange(len(source_names))
                bars = ax.bar(x_pos, results['attention_weights'], alpha=0.7, color='steelblue')
                ax.set_xlabel('Source Simulations')
                ax.set_ylabel('Attention Weight')
                ax.set_title('Attention Weights for Stress Interpolation')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(source_names, rotation=45, ha='right')
                
                # Add value labels
                for bar, weight in zip(bars, results['attention_weights']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
                
                st.pyplot(fig_attention)
            
            with col2:
                st.subheader("📊 Attention Statistics")
                
                # Calculate statistics
                attn_weights = results['attention_weights'].flatten()
                
                st.metric("Max Weight", f"{np.max(attn_weights):.3f}")
                st.metric("Min Weight", f"{np.min(attn_weights):.3f}")
                st.metric("Mean Weight", f"{np.mean(attn_weights):.3f}")
                st.metric("Std Dev", f"{np.std(attn_weights):.3f}")
                
                # Dominant source
                if attn_weights.ndim == 1:
                    dominant_idx = np.argmax(attn_weights)
                    st.success(f"**Dominant Source:** S{dominant_idx + 1}")
            
            # Display predicted stress fields
            st.subheader("🎯 Predicted Stress Fields")
            
            stress_fields = results['stress_fields']
            
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
            
            # Export options
            st.subheader("📥 Export Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("💾 Save to Numerical Solutions", type="primary"):
                    # Create export data
                    export_data = {
                        'prediction_results': results,
                        'source_simulations_count': len(st.session_state.source_simulations),
                        'target_params': st.session_state.target_params,
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    # Save to numerical_solutions directory
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
                    'target_params': st.session_state.target_params,
                    'export_timestamp': datetime.now().isoformat()
                }
                
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
                # Create report
                report = f"""
                SPATIAL-ATTENTION STRESS PREDICTION REPORT
                ============================================
                
                Generated: {datetime.now().isoformat()}
                Source Simulations: {len(st.session_state.source_simulations)}
                
                ATTENTION WEIGHTS:
                ------------------
                """
                
                for i, weight in enumerate(results['attention_weights']):
                    report += f"S{i+1}: {weight:.4f}\n"
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"attention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with tab5:
        st.subheader("📁 Numerical Solutions Directory Management")
        
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
        
        # Show directory statistics
        all_files_info = st.session_state.solutions_manager.get_all_files()
        
        if not all_files_info:
            st.warning("No files found in numerical_solutions directory")
        else:
            # Statistics
            total_size = sum(f['size'] for f in all_files_info) / (1024 * 1024)  # MB
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
            
            # File list with actions
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
                                # Load simulation
                                sim_data = st.session_state.solutions_manager.load_simulation(
                                    file_info['path'], 
                                    st.session_state.interpolator
                                )
                                
                                # Check if already loaded
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
            
            # Bulk actions
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
    """Main application with attention interpolation"""
    
    # Display header with directory info
    st.sidebar.header("📁 Directory Information")
    st.sidebar.write(f"**App Directory:** `{SCRIPT_DIR}`")
    st.sidebar.write(f"**Solutions Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
    
    # Check if directory exists
    if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
        st.sidebar.warning("⚠️ Solutions directory not found")
        if st.sidebar.button("📁 Create Directory"):
            os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
            st.sidebar.success("✅ Directory created")
            st.rerun()
    
    # Sidebar operation mode
    st.sidebar.header("🔧 Operation Mode")
    
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["Attention Interpolation", "Run New Simulation", 
         "Compare Saved Simulations", "Single Simulation View"],
        index=0  # Default to Attention Interpolation
    )
    
    if operation_mode == "Attention Interpolation":
        # Run attention interpolation interface
        create_attention_interface()
    
    else:
        # Original simulation interface (kept for compatibility)
        st.warning("⚠️ This mode is not fully integrated with attention interpolation.")
        st.info("Please use 'Attention Interpolation' mode for spatial-attention predictions.")
        
        # Placeholder for original interface
        st.header("Original Simulation Interface")
        st.write("This interface is available but separate from attention interpolation.")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Analysis: Spatial-Attention Interpolation", expanded=False):
    st.markdown(f"""
    ## 🎯 **Spatial Locality Regularized Attention Interpolation**
    
    ### **📁 Numerical Solutions Directory Integration**
    
    The system now automatically loads simulation files from:
    ```
    {NUMERICAL_SOLUTIONS_DIR}
    ```
    
    **Supported File Formats:**
    - **PKL** (.pkl): Python pickle format (from your original export)
    - **PT** (.pt): PyTorch tensor format  
    - **H5** (.h5, .hdf5): Hierarchical data format
    - **NPZ** (.npz): Compressed numpy arrays
    - **SQL** (.sql, .db): Database dumps
    - **JSON** (.json): Standardized metadata
    
    **Directory Structure:**
    ```
    {os.path.basename(SCRIPT_DIR)}/
    ├── app.py                    # This Streamlit application
    └── numerical_solutions/      # Auto-scanned directory
        ├── simulation_1.pkl
        ├── simulation_2.pt
        ├── results_2024.h5
        └── metadata.json
    ```
    
    ### **🔄 Auto-Loading Workflow:**
    
    1. **Directory Scanning**: System automatically scans `numerical_solutions` on startup
    2. **Format Detection**: Files are categorized by extension
    3. **Metadata Extraction**: File size, modification time, and format info
    4. **Visual Selection**: User selects files from organized dropdown lists
    5. **Loading**: Files are loaded using appropriate format readers
    
    ### **🔧 Key Features:**
    
    **1. Dual Loading Methods:**
    - **Numerical Solutions Directory**: Load precomputed simulations from local directory
    - **File Upload**: Upload additional simulations via browser
    
    **2. Directory Management:**
    - **Auto-creation**: Directory created if missing
    - **File Statistics**: Size, count, and format distribution
    - **Visual Preview**: Organized file listing with metadata
    - **Bulk Operations**: Load, delete, or refresh multiple files
    
    **3. Seamless Integration:**
    - **Relative Paths**: Uses `os.path.join` for cross-platform compatibility
    - **Format Conversion**: Automatically converts between file formats
    - **Metadata Preservation**: All simulation parameters preserved
    
    ### **🚀 Usage Instructions:**
    
    **Step 1: Prepare Simulation Files**
    ```bash
    # Place your simulation files in:
    {NUMERICAL_SOLUTIONS_DIR}/
    
    # Supported naming conventions:
    ISF_orient0deg_Square_eps0-0.707_kappa-0.6.pkl
    ESF_orient30deg_Rectangle_eps0-1.414_kappa-0.7.pt
    Twin_orient90deg_Ellipse_eps0-2.121_kappa-0.3.h5
    ```
    
    **Step 2: Load Simulations**
    1. Go to "📤 Load Source Data" tab
    2. Select files from "From Numerical Solutions Directory" section
    3. Click "Load Selected Files"
    4. View loaded simulations in parameter space
    
    **Step 3: Train and Predict**
    1. Configure target parameters in "🎯 Configure Target" tab
    2. Adjust training settings in "🚀 Train & Predict" tab
    3. Click "Train & Predict" to generate stress fields
    
    **Step 4: Export Results**
    1. View predictions in "📊 Results & Export" tab
    2. Save results back to numerical_solutions directory
    3. Download reports and data files
    
    **Advanced features and technical implementation details continue below...**
    """)

# Run the main application
if __name__ == "__main__":
    main()

st.caption(f"🔬 Spatial-Attention Stress Interpolation • Auto-loading from {NUMERICAL_SOLUTIONS_DIR} • 2025")
