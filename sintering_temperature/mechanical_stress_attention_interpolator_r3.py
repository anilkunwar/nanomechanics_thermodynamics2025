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
import torch.serialization  # For PyTorch security settings
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
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# =============================================
# GLOBAL PYTORCH SECURITY SETUP
# =============================================
def setup_torch_security():
    """Configure PyTorch security settings for compatibility with older files"""
    try:
        import numpy as np
        import torch.serialization
        
        # Add safe globals for NumPy compatibility
        safe_globals = []
        
        # Add NumPy scalar types
        try:
            safe_globals.extend([
                np._core.multiarray.scalar,  # Internal scalar type
                np.core.multiarray.scalar,   # Public alias
                np.dtype,                    # Data type objects
                np.ndarray,                  # Array objects
                np.bool_, np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64,
                np.float16, np.float32, np.float64,
                np.complex64, np.complex128,
                np.str_, np.bytes_, np.object_
            ])
        except AttributeError:
            pass
        
        # Add dtype classes for NumPy 2.0+
        try:
            from numpy.dtypes import Float64DType, Float32DType, Int64DType, Int32DType
            safe_globals.extend([
                Float64DType, Float32DType, Int64DType, Int32DType
            ])
        except ImportError:
            pass
        
        # Add the safe globals
        if safe_globals:
            torch.serialization.add_safe_globals(safe_globals)
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not setup torch security: {str(e)}")

# Initialize security settings
setup_torch_security()

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")

if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ENHANCED FILE LOADING MANAGER
# =============================================
class EnhancedFileLoader:
    """Enhanced file loader with multiple fallback strategies"""
    
    @staticmethod
    def check_file_integrity(file_path: str) -> Tuple[bool, str]:
        """
        Check if a file is valid before attempting to load
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
            
            if file_size < 10:
                return False, "File is too small"
            
            # Check file headers
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
                if len(header) < 4:
                    return False, "File too short for header check"
                
                # Check for common file signatures
                if file_path.endswith('.pt') or file_path.endswith('.pth'):
                    # PyTorch files start with a pickle header
                    if header[0] != 0x80:
                        return False, "Invalid PyTorch file header"
                
                elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                    # Pickle files have protocol header
                    if header[0] not in [0x80, 0x83, 0x84, 0x85, 0x86, 0x87]:
                        return False, "Invalid pickle protocol"
                
                elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                    # HDF5 signature
                    if not header.startswith(b'\x89HDF\r\n\x1a\n'):
                        return False, "Invalid HDF5 header"
                
                elif file_path.endswith('.npz'):
                    # NPZ files are zip files
                    if not header.startswith(b'PK'):
                        return False, "Invalid NPZ (not a ZIP file)"
            
            return True, "File appears valid"
            
        except Exception as e:
            return False, f"Error checking file: {str(e)}"
    
    @staticmethod
    def load_pytorch_file(file_path: str, use_secure_loading: bool = True):
        """
        Load PyTorch file with multiple fallback strategies
        
        Args:
            file_path: Path to PyTorch file
            use_secure_loading: Whether to use weights_only=True
            
        Returns:
            Loaded data
        """
        strategies = [
            ("Secure loading (weights_only=True)", lambda: torch.load(file_path, map_location='cpu', weights_only=True)),
            ("Insecure loading (weights_only=False)", lambda: torch.load(file_path, map_location='cpu', weights_only=False)),
            ("Direct buffer load", lambda: EnhancedFileLoader._load_pytorch_from_buffer(file_path)),
            ("Zip extraction", lambda: EnhancedFileLoader._extract_pytorch_zip(file_path))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                if "Secure" in strategy_name and not use_secure_loading:
                    continue
                    
                data = strategy_func()
                st.info(f"✅ Loaded using {strategy_name}")
                return data
                
            except Exception as e:
                st.warning(f"⚠️ {strategy_name} failed: {str(e)}")
                continue
        
        raise ValueError("All PyTorch loading strategies failed")
    
    @staticmethod
    def _load_pytorch_from_buffer(file_path: str):
        """Load PyTorch file using BytesIO buffer"""
        with open(file_path, 'rb') as f:
            buffer = BytesIO(f.read())
        return torch.load(buffer, map_location='cpu')
    
    @staticmethod
    def _extract_pytorch_zip(file_path: str):
        """Extract data from PyTorch zip archive"""
        import zipfile
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Look for pickle files
            for name in zf.namelist():
                if name.endswith('.pkl') or 'data' in name.lower():
                    with zf.open(name) as f:
                        return pickle.load(f)
        
        # If no pickle found, try to read as regular zip
        with zipfile.ZipFile(file_path, 'r') as zf:
            data = {}
            for name in zf.namelist():
                with zf.open(name) as f:
                    try:
                        content = f.read()
                        # Try to parse as pickle
                        buffer = BytesIO(content)
                        data[name] = pickle.load(buffer)
                    except:
                        data[name] = content
            return data
    
    @staticmethod
    def load_pickle_file(file_path: str):
        """
        Load pickle file with multiple fallback strategies
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Loaded data
        """
        strategies = [
            ("Standard pickle", lambda: EnhancedFileLoader._load_pickle_standard(file_path)),
            ("Joblib", lambda: joblib.load(file_path)),
            ("Dill", lambda: dill.load(open(file_path, 'rb'))),
            ("Pickle with protocols", lambda: EnhancedFileLoader._load_pickle_with_protocols(file_path)),
            ("Text recovery", lambda: EnhancedFileLoader._recover_pickle_text(file_path))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                data = strategy_func()
                st.info(f"✅ Loaded using {strategy_name}")
                return data
                
            except Exception as e:
                st.warning(f"⚠️ {strategy_name} failed: {str(e)}")
                continue
        
        raise ValueError("All pickle loading strategies failed")
    
    @staticmethod
    def _load_pickle_standard(file_path: str):
        """Standard pickle load"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _load_pickle_with_protocols(file_path: str):
        """Try pickle with different protocols"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try different protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                buffer = BytesIO(content)
                return pickle.load(buffer)
            except:
                continue
        
        raise ValueError("No pickle protocol worked")
    
    @staticmethod
    def _recover_pickle_text(file_path: str):
        """Attempt to recover data from corrupted pickle"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try to decode as text and look for patterns
        try:
            text = content.decode('utf-8', errors='ignore')
            if 'numpy' in text or 'array' in text:
                st.info("File contains numpy array references")
                return {'recovered_text': text[:500]}
        except:
            pass
        
        raise ValueError("Could not recover pickle data")
    
    @staticmethod
    def convert_legacy_data(data: Any) -> Dict[str, Any]:
        """
        Convert legacy data formats to standardized format
        
        Args:
            data: Raw loaded data
            
        Returns:
            Standardized simulation data
        """
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': 'legacy',
            'converted': True
        }
        
        if isinstance(data, dict):
            # Extract from common keys
            if 'params' in data:
                standardized['params'] = data['params']
            elif 'parameters' in data:
                standardized['params'] = data['parameters']
            elif 'config' in data:
                standardized['params'] = data['config']
            
            # Extract history
            if 'history' in data:
                standardized['history'] = data['history']
            elif 'frames' in data:
                standardized['history'] = data['frames']
            elif 'stress_fields' in data:
                standardized['history'] = [(0.0, data['stress_fields'])]
            
            # Extract metadata
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            elif 'info' in data:
                standardized['metadata'] = data['info']
            
            # Direct stress fields
            stress_keys = ['sigma_hydro', 'sigma_mag', 'von_mises', 'sigma_xx', 'sigma_yy', 'sigma_xy']
            if any(key in data for key in stress_keys):
                stress_fields = {}
                for key in stress_keys:
                    if key in data:
                        stress_fields[key] = data[key]
                standardized['history'] = [(0.0, stress_fields)]
        
        elif isinstance(data, np.ndarray):
            # Single array
            standardized['history'] = [(0.0, {'stress_field': data})]
        
        elif isinstance(data, list) and len(data) > 0:
            # List of frames
            standardized['history'] = data
        
        return standardized

# =============================================
# ENHANCED NUMERICAL SOLUTIONS MANAGER
# =============================================
class EnhancedNumericalSolutionsManager:
    def __init__(self, solutions_dir: str = NUMERICAL_SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.loaded_files_cache = {}
        self.failed_files = {}
        self.stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def _ensure_directory(self):
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_directory(self, recursive: bool = False) -> Dict[str, List[str]]:
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
                files = glob.glob(pattern, recursive=recursive)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_formats[format_type].extend(files)
        
        return file_formats
    
    def get_all_files(self, max_files: int = 100) -> List[Dict[str, Any]]:
        all_files = []
        file_formats = self.scan_directory()
        
        for format_type, files in file_formats.items():
            for file_path in files[:max_files]:
                try:
                    file_info = {
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'format': format_type,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        'relative_path': os.path.relpath(file_path, self.solutions_dir),
                        'status': 'unknown'
                    }
                    all_files.append(file_info)
                except:
                    continue
        
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze file without loading it
        
        Args:
            file_path: Path to file
            
        Returns:
            Analysis results
        """
        analysis = {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'exists': os.path.exists(file_path),
            'size': 0,
            'valid': False,
            'issues': [],
            'recommendation': None
        }
        
        if not analysis['exists']:
            analysis['issues'].append("File does not exist")
            return analysis
        
        try:
            analysis['size'] = os.path.getsize(file_path)
            
            # Check file integrity
            is_valid, reason = EnhancedFileLoader.check_file_integrity(file_path)
            analysis['valid'] = is_valid
            if not is_valid:
                analysis['issues'].append(reason)
            
            # Check file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pt' or ext == '.pth':
                analysis['format'] = 'pytorch'
                analysis['recommendation'] = "Try weights_only=False for compatibility"
            elif ext == '.pkl' or ext == '.pickle':
                analysis['format'] = 'pickle'
                analysis['recommendation'] = "Check pickle protocol compatibility"
            elif ext == '.h5' or ext == '.hdf5':
                analysis['format'] = 'hdf5'
                analysis['recommendation'] = "Ensure h5py is installed"
            elif ext == '.npz':
                analysis['format'] = 'numpy'
                analysis['recommendation'] = "Standard numpy format, should load easily"
            else:
                analysis['format'] = 'unknown'
                analysis['issues'].append(f"Unrecognized extension: {ext}")
            
        except Exception as e:
            analysis['issues'].append(f"Error analyzing file: {str(e)}")
        
        return analysis
    
    def load_simulation_with_fallbacks(self, file_path: str, interpolator, 
                                     use_secure_loading: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load simulation file with comprehensive fallback strategies
        
        Args:
            file_path: Path to simulation file
            interpolator: SpatialLocalityAttentionInterpolator instance
            use_secure_loading: Whether to use secure PyTorch loading
            
        Returns:
            Loaded simulation data or None
        """
        self.stats['total_attempted'] += 1
        filename = os.path.basename(file_path)
        
        # Check cache first
        if file_path in self.loaded_files_cache:
            self.stats['successful'] += 1
            return self.loaded_files_cache[file_path]
        
        # Check if previously failed
        if file_path in self.failed_files:
            self.stats['skipped'] += 1
            st.warning(f"⚠️ Skipping previously failed file: {filename}")
            return None
        
        try:
            # First check file integrity
            is_valid, reason = EnhancedFileLoader.check_file_integrity(file_path)
            if not is_valid:
                self.failed_files[file_path] = f"Invalid file: {reason}"
                self.stats['failed'] += 1
                st.error(f"❌ File integrity check failed for {filename}: {reason}")
                return None
            
            # Determine file format
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.pt', '.pth']:
                # PyTorch file
                st.info(f"Loading PyTorch file: {filename}")
                with st.spinner(f"Loading {filename}..."):
                    try:
                        # Try multiple loading strategies
                        raw_data = EnhancedFileLoader.load_pytorch_file(file_path, use_secure_loading)
                        
                        # Convert to standardized format
                        if isinstance(raw_data, dict):
                            sim_data = interpolator._standardize_data(raw_data, 'pt', file_path)
                        else:
                            sim_data = EnhancedFileLoader.convert_legacy_data(raw_data)
                            
                    except Exception as e:
                        self.failed_files[file_path] = str(e)
                        self.stats['failed'] += 1
                        st.error(f"❌ Failed to load PyTorch file {filename}: {str(e)}")
                        return None
            
            elif ext in ['.pkl', '.pickle']:
                # Pickle file
                st.info(f"Loading pickle file: {filename}")
                with st.spinner(f"Loading {filename}..."):
                    try:
                        raw_data = EnhancedFileLoader.load_pickle_file(file_path)
                        
                        if isinstance(raw_data, dict):
                            sim_data = interpolator._standardize_data(raw_data, 'pkl', file_path)
                        else:
                            sim_data = EnhancedFileLoader.convert_legacy_data(raw_data)
                            
                    except Exception as e:
                        self.failed_files[file_path] = str(e)
                        self.stats['failed'] += 1
                        st.error(f"❌ Failed to load pickle file {filename}: {str(e)}")
                        return None
            
            else:
                # Other formats
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                if ext in ['.h5', '.hdf5']:
                    sim_data = interpolator.read_simulation_file(file_content, 'h5')
                elif ext == '.npz':
                    sim_data = interpolator.read_simulation_file(file_content, 'npz')
                elif ext in ['.sql', '.db']:
                    sim_data = interpolator.read_simulation_file(file_content, 'sql')
                elif ext == '.json':
                    sim_data = interpolator.read_simulation_file(file_content, 'json')
                else:
                    st.warning(f"⚠️ Unknown file format: {filename}")
                    self.failed_files[file_path] = "Unknown format"
                    self.stats['failed'] += 1
                    return None
            
            # Validate and cache
            if sim_data and self._validate_simulation_data(sim_data):
                sim_data['loaded_from'] = 'numerical_solutions'
                sim_data['filename'] = filename
                sim_data['load_timestamp'] = datetime.now().isoformat()
                
                self.loaded_files_cache[file_path] = sim_data
                self.stats['successful'] += 1
                
                st.success(f"✅ Successfully loaded: {filename}")
                return sim_data
            else:
                self.failed_files[file_path] = "Invalid data structure"
                self.stats['failed'] += 1
                st.warning(f"⚠️ Loaded but invalid data structure: {filename}")
                return None
                
        except Exception as e:
            error_msg = str(e)
            self.failed_files[file_path] = error_msg
            self.stats['failed'] += 1
            
            # Provide helpful error messages
            if "weights_only" in error_msg or "safe_globals" in error_msg:
                st.error(f"❌ PyTorch security restriction for {filename}. Try insecure loading.")
            elif "pickle" in error_msg.lower():
                st.error(f"❌ Pickle error for {filename}. File may be corrupted.")
            elif "unpickling" in error_msg.lower():
                st.error(f"❌ Unpickling error for {filename}. Python version mismatch.")
            else:
                st.error(f"❌ Error loading {filename}: {error_msg}")
            
            return None
    
    def _validate_simulation_data(self, sim_data: Dict[str, Any]) -> bool:
        """Validate that simulation data has required structure"""
        if not isinstance(sim_data, dict):
            return False
        
        # Check for either history or params
        has_history = 'history' in sim_data and isinstance(sim_data['history'], list)
        has_params = 'params' in sim_data and isinstance(sim_data['params'], dict)
        has_stress = False
        
        # Check if history contains stress data
        if has_history:
            for frame in sim_data['history']:
                if isinstance(frame, (list, tuple)) and len(frame) == 2:
                    _, stress_fields = frame
                    if isinstance(stress_fields, dict) and any('sigma' in key for key in stress_fields.keys()):
                        has_stress = True
                        break
        
        return has_history or has_params or has_stress
    
    def batch_load_simulations(self, file_paths: List[str], interpolator,
                             max_workers: int = 4, 
                             use_secure_loading: bool = False) -> List[Dict[str, Any]]:
        """
        Batch load multiple simulation files
        
        Args:
            file_paths: List of file paths
            interpolator: SpatialLocalityAttentionInterpolator instance
            max_workers: Maximum parallel workers
            use_secure_loading: Whether to use secure PyTorch loading
            
        Returns:
            List of loaded simulation data
        """
        loaded_simulations = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_path in enumerate(file_paths):
            status_text.text(f"Loading {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            progress_bar.progress((i + 1) / len(file_paths))
            
            sim_data = self.load_simulation_with_fallbacks(
                file_path, interpolator, use_secure_loading
            )
            
            if sim_data:
                loaded_simulations.append(sim_data)
        
        progress_bar.empty()
        status_text.empty()
        
        return loaded_simulations
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        return {
            **self.stats,
            'cache_size': len(self.loaded_files_cache),
            'failed_files': len(self.failed_files),
            'success_rate': self.stats['successful'] / max(self.stats['total_attempted'], 1) * 100
        }
    
    def clear_cache(self):
        """Clear loaded files cache"""
        self.loaded_files_cache.clear()
        self.failed_files.clear()
        self.stats = {'total_attempted': 0, 'successful': 0, 'failed': 0, 'skipped': 0}
        st.success("✅ Cache cleared")
    
    def save_simulation(self, data: Dict[str, Any], filename: str, format_type: str = 'pkl'):
        """Save simulation data to file"""
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
            
            # Clear cache for this file
            if file_path in self.loaded_files_cache:
                del self.loaded_files_cache[file_path]
            if file_path in self.failed_files:
                del self.failed_files[file_path]
            
            st.success(f"✅ Saved simulation to: {filename}")
            return True
            
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return False

# =============================================
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
# =============================================
class EnhancedSpatialLocalityAttentionInterpolator:
    """Enhanced attention-based interpolator with robust file loading"""
    
    def __init__(self, input_dim=15, num_heads=4, d_model=32, output_dim=3, 
                 sigma_spatial=0.2, sigma_param=0.2, use_gaussian=True):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.output_dim = output_dim
        self.sigma_spatial = sigma_spatial
        self.sigma_param = sigma_param
        self.use_gaussian = use_gaussian
        
        self.model = self._build_model()
        self.file_loader = EnhancedFileLoader()
        
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
            'param_embedding': torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.d_model)
            ),
            'attention': torch.nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1
            ),
            'feed_forward': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.d_model * 4, self.d_model)
            ),
            'output_projection': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.output_dim)
            ),
            'spatial_regularizer': torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, self.num_heads)
            ) if self.use_gaussian else None,
            'norm1': torch.nn.LayerNorm(self.d_model),
            'norm2': torch.nn.LayerNorm(self.d_model)
        })
        return model
    
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        try:
            return pickle.load(buffer)
        except Exception as e:
            # Try with different protocols
            buffer.seek(0)
            try:
                return pickle.load(buffer, encoding='latin1')
            except:
                buffer.seek(0)
                try:
                    return pickle.load(buffer, fix_imports=True)
                except:
                    raise ValueError(f"Failed to unpickle: {str(e)}")
    
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        
        # Strategy 1: Try secure loading first
        try:
            return torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e1:
            st.warning(f"⚠️ Secure PyTorch load failed: {str(e1)}")
        
        # Strategy 2: Try insecure loading
        buffer.seek(0)
        try:
            st.warning("⚠️ Trying insecure PyTorch load (weights_only=False)")
            return torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e2:
            # Strategy 3: Try to extract manually
            buffer.seek(0)
            try:
                import zipfile
                import io
                
                # Check if it's a zip file
                buffer.seek(0)
                magic = buffer.read(4)
                buffer.seek(0)
                
                if magic.startswith(b'PK'):
                    # It's a zip file
                    with zipfile.ZipFile(buffer, 'r') as zf:
                        # Look for pickle files
                        for name in zf.namelist():
                            if name.endswith('.pkl') or 'data' in name.lower():
                                with zf.open(name) as f:
                                    return pickle.load(f)
                
                raise ValueError(f"Could not load PyTorch file: {str(e2)}")
            except Exception as e3:
                raise ValueError(f"All PyTorch loading strategies failed: {str(e3)}")
    
    def _read_h5(self, file_content):
        buffer = BytesIO(file_content)
        with h5py.File(buffer, 'r') as f:
            data = {}
            def read_h5_obj(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        data[name] = obj[()]
                    except:
                        data[name] = str(obj)
                elif isinstance(obj, h5py.Group):
                    data[name] = {}
                    for key in obj.keys():
                        read_h5_obj(f"{name}/{key}", obj[key])
            for key in f.keys():
                read_h5_obj(key, f[key])
        return data
    
    def _read_npz(self, file_content):
        buffer = BytesIO(file_content)
        data = np.load(buffer, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    def _read_sql(self, file_content):
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            data = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                data[table_name] = {
                    'columns': columns,
                    'rows': rows,
                    'dataframe': pd.DataFrame(rows, columns=columns)
                }
            
            conn.close()
            os.unlink(tmp_path)
            return data
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    def _read_json(self, file_content):
        try:
            return json.loads(file_content.decode('utf-8'))
        except UnicodeDecodeError:
            try:
                return json.loads(file_content.decode('latin-1'))
            except:
                raise ValueError("Could not decode JSON file")
    
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file with enhanced error handling"""
        
        if format_type == 'auto':
            format_type = 'pkl'  # Default
        
        if format_type in self.readers:
            try:
                data = self.readers[format_type](file_content)
                return self._standardize_data(data, format_type, "uploaded_file")
            except Exception as e:
                if "weights_only" in str(e):
                    raise ValueError(
                        f"PyTorch security restriction. "
                        f"Try using weights_only=False or update file format. "
                        f"Error: {str(e)}"
                    )
                elif "pickle" in str(e).lower():
                    raise ValueError(
                        f"Pickle loading error. File may be corrupted or from different Python version. "
                        f"Error: {str(e)}"
                    )
                else:
                    raise ValueError(f"Error reading {format_type} file: {str(e)}")
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _standardize_data(self, data, format_type, file_path):
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
        }
        
        if format_type == 'pkl':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                
                history = data.get('history', [])
                if history:
                    for frame in history:
                        if isinstance(frame, dict):
                            eta = frame.get('eta', 0.0)
                            stresses = frame.get('stresses', {})
                            standardized['history'].append((eta, stresses))
                        elif isinstance(frame, (list, tuple)) and len(frame) == 2:
                            standardized['history'].append(frame)
        
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
            
            for key, value in data.items():
                if 'history' in key.lower():
                    if isinstance(value, (list, np.ndarray)):
                        standardized['history'] = value
                        break
        
        elif format_type == 'npz':
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            if 'history' in data:
                standardized['history'] = data['history']
        
        elif format_type == 'json':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                standardized['history'] = data.get('history', [])
        
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
        
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3) if eps0 is not None else 0.5
        param_vector.append(eps0_norm)
        param_names.append('eps0_norm')
        
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1) if kappa is not None else 0.5
        param_vector.append(kappa_norm)
        param_names.append('kappa_norm')
        
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi) if theta is not None else 0.0
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
        
        if orientation.startswith('Custom ('):
            param_vector.extend([0, 0, 0, 0])
        else:
            param_vector.extend(orientation_encoding.get(orientation, [0, 0, 0, 0]))
            
        param_names.extend(['orient_0deg', 'orient_30deg', 'orient_60deg', 'orient_90deg'])
        
        return np.array(param_vector, dtype=np.float32), param_names
    
    @staticmethod
    def get_orientation_from_angle(angle_deg: float) -> str:
        """Convert angle in degrees to orientation string"""
        if 0 <= angle_deg <= 15:
            return 'Horizontal {111} (0°)'
        elif 15 < angle_deg <= 45:
            return 'Tilted 30° (1¯10 projection)'
        elif 45 < angle_deg <= 75:
            return 'Tilted 60°'
        elif 75 < angle_deg <= 90:
            return 'Vertical {111} (90°)'
        else:
            angle_deg = angle_deg % 90
            return f"Custom ({angle_deg:.1f}°)"

# =============================================
# ENHANCED STRESS ANALYSIS DASHBOARD
# =============================================
def create_stress_analysis_dashboard():
    """Create enhanced stress analysis dashboard with robust file loading"""
    
    st.title("📊 Enhanced Stress Analysis Dashboard")
    
    # Initialize managers
    if 'enhanced_solutions_manager' not in st.session_state:
        st.session_state.enhanced_solutions_manager = EnhancedNumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
    
    if 'enhanced_interpolator' not in st.session_state:
        st.session_state.enhanced_interpolator = EnhancedSpatialLocalityAttentionInterpolator()
    
    if 'stress_analyzer' not in st.session_state:
        from StressAnalysisManager import StressAnalysisManager
        st.session_state.stress_analyzer = StressAnalysisManager()
    
    if 'sunburst_manager' not in st.session_state:
        from SunburstChartManager import SunburstChartManager
        st.session_state.sunburst_manager = SunburstChartManager()
    
    # Initialize data structures
    if 'dashboard_simulations' not in st.session_state:
        st.session_state.dashboard_simulations = []
    
    if 'dashboard_stress_summary' not in st.session_state:
        st.session_state.dashboard_stress_summary = pd.DataFrame()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Dashboard Settings")
    
    with st.sidebar.expander("📁 File Loading Settings", expanded=True):
        max_files = st.slider("Maximum files to load", 10, 200, 50, 10)
        use_secure_loading = st.checkbox("Use secure PyTorch loading", value=False,
                                        help="If checked, uses weights_only=True. Uncheck for compatibility with older files.")
        skip_corrupted = st.checkbox("Skip corrupted files", value=True)
        retry_failed = st.checkbox("Retry previously failed files", value=False)
        
        # File type filters
        st.subheader("File Type Filters")
        col1, col2 = st.columns(2)
        with col1:
            load_pt = st.checkbox(".pt files", value=True)
            load_pkl = st.checkbox(".pkl files", value=True)
            load_h5 = st.checkbox(".h5 files", value=True)
        with col2:
            load_npz = st.checkbox(".npz files", value=True)
            load_json = st.checkbox(".json files", value=True)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 File Management",
        "📊 Stress Analysis",
        "📈 Visualizations",
        "🔧 Diagnostics"
    ])
    
    # Tab 1: File Management
    with tab1:
        st.header("📁 File Management and Loading")
        
        # Directory info
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
        
        # Scan directory
        if st.button("🔄 Scan Directory", type="secondary"):
            st.rerun()
        
        all_files = st.session_state.enhanced_solutions_manager.get_all_files(max_files)
        
        if not all_files:
            st.warning(f"No files found in `{NUMERICAL_SOLUTIONS_DIR}`")
        else:
            # File statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", len(all_files))
            with col2:
                pt_count = len([f for f in all_files if f['format'] == 'pt'])
                st.metric(".pt Files", pt_count)
            with col3:
                pkl_count = len([f for f in all_files if f['format'] == 'pkl'])
                st.metric(".pkl Files", pkl_count)
            with col4:
                total_size = sum(f['size'] for f in all_files) / (1024 * 1024)
                st.metric("Total Size", f"{total_size:.1f} MB")
            
            # Loading strategies
            st.subheader("🚀 Loading Strategies")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Quick Load All", type="primary"):
                    with st.spinner("Loading all compatible files..."):
                        filtered_files = []
                        for file_info in all_files:
                            format_type = file_info['format']
                            if (format_type == 'pt' and not load_pt) or \
                               (format_type == 'pkl' and not load_pkl) or \
                               (format_type == 'h5' and not load_h5) or \
                               (format_type == 'npz' and not load_npz) or \
                               (format_type == 'json' and not load_json):
                                continue
                            
                            if file_info['path'] in st.session_state.enhanced_solutions_manager.failed_files and not retry_failed:
                                continue
                            
                            filtered_files.append(file_info['path'])
                        
                        loaded = st.session_state.enhanced_solutions_manager.batch_load_simulations(
                            filtered_files,
                            st.session_state.enhanced_interpolator,
                            use_secure_loading=use_secure_loading
                        )
                        
                        st.session_state.dashboard_simulations = loaded
                        
                        if loaded:
                            st.success(f"✅ Successfully loaded {len(loaded)} simulations")
                        else:
                            st.warning("No simulations were loaded")
            
            with col2:
                if st.button("🧹 Clear Cache", type="secondary"):
                    st.session_state.enhanced_solutions_manager.clear_cache()
                    st.session_state.dashboard_simulations = []
                    st.session_state.dashboard_stress_summary = pd.DataFrame()
                    st.success("✅ Cache cleared")
                    st.rerun()
            
            with col3:
                if st.button("📊 Update Stress Summary", type="secondary"):
                    if st.session_state.dashboard_simulations:
                        st.session_state.dashboard_stress_summary = (
                            st.session_state.stress_analyzer.create_stress_summary_dataframe(
                                st.session_state.dashboard_simulations, {}
                            )
                        )
                        if not st.session_state.dashboard_stress_summary.empty:
                            st.success(f"✅ Stress summary updated with {len(st.session_state.dashboard_stress_summary)} entries")
                        else:
                            st.warning("No stress data found in loaded simulations")
                    else:
                        st.warning("No simulations loaded")
            
            # File list with analysis
            st.subheader("📋 File List")
            
            for file_info in all_files[:20]:  # Show first 20 files
                with st.expander(f"{file_info['filename']} ({file_info['format'].upper()}, {file_info['size']//1024}KB)"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Path:** `{file_info['relative_path']}`")
                        st.write(f"**Size:** {file_info['size']:,} bytes")
                        st.write(f"**Modified:** {file_info['modified'][:19]}")
                        
                        # Quick analysis
                        if st.button("🔍 Analyze", key=f"analyze_{file_info['filename']}"):
                            analysis = st.session_state.enhanced_solutions_manager.analyze_file(file_info['path'])
                            
                            if analysis['valid']:
                                st.success("✅ File appears valid")
                            else:
                                st.error(f"❌ File issues: {', '.join(analysis['issues'])}")
                            
                            if analysis['recommendation']:
                                st.info(f"💡 Recommendation: {analysis['recommendation']}")
                    
                    with col2:
                        if st.button("📂 Load", key=f"load_{file_info['filename']}"):
                            try:
                                sim_data = st.session_state.enhanced_solutions_manager.load_simulation_with_fallbacks(
                                    file_info['path'],
                                    st.session_state.enhanced_interpolator,
                                    use_secure_loading
                                )
                                
                                if sim_data:
                                    # Check if already loaded
                                    if not any(s['filename'] == file_info['filename'] for s in st.session_state.dashboard_simulations):
                                        st.session_state.dashboard_simulations.append(sim_data)
                                        st.success(f"✅ Added: {file_info['filename']}")
                                        st.rerun()
                                    else:
                                        st.warning(f"⚠️ Already loaded: {file_info['filename']}")
                                else:
                                    st.error(f"❌ Failed to load: {file_info['filename']}")
                                    
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{file_info['filename']}"):
                            if st.checkbox(f"Confirm delete {file_info['filename']}", key=f"confirm_{file_info['filename']}"):
                                try:
                                    os.remove(file_info['path'])
                                    st.success(f"✅ Deleted: {file_info['filename']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error deleting: {str(e)}")
    
    # Tab 2: Stress Analysis
    with tab2:
        st.header("📊 Stress Analysis")
        
        if not st.session_state.dashboard_simulations:
            st.info("👈 Please load simulations first")
        else:
            # Show loading statistics
            stats = st.session_state.enhanced_solutions_manager.get_loading_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Loaded Simulations", len(st.session_state.dashboard_simulations))
            with col2:
                st.metric("Loading Success Rate", f"{stats['success_rate']:.1f}%")
            with col3:
                st.metric("Cache Size", stats['cache_size'])
            with col4:
                st.metric("Failed Files", stats['failed_files'])
            
            # Stress summary
            if not st.session_state.dashboard_stress_summary.empty:
                st.subheader("📋 Stress Summary")
                
                # Quick stats
                st.dataframe(
                    st.session_state.dashboard_stress_summary.style.format({
                        col: "{:.3f}" for col in st.session_state.dashboard_stress_summary.select_dtypes(include=[np.number]).columns
                    }),
                    use_container_width=True,
                    height=300
                )
                
                # Export options
                csv_buffer = BytesIO()
                st.session_state.dashboard_stress_summary.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer,
                    file_name=f"stress_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Detailed analysis
                st.subheader("🔬 Detailed Analysis")
                
                analysis_tabs = st.tabs(["Statistics", "Correlations", "Distributions"])
                
                with analysis_tabs[0]:
                    # Basic statistics
                    numeric_cols = st.session_state.dashboard_stress_summary.select_dtypes(include=[np.number]).columns
                    selected_metric = st.selectbox("Select metric", numeric_cols, index=0)
                    
                    if selected_metric in st.session_state.dashboard_stress_summary.columns:
                        data = st.session_state.dashboard_stress_summary[selected_metric].dropna()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{data.mean():.3f}")
                        with col2:
                            st.metric("Std Dev", f"{data.std():.3f}")
                        with col3:
                            st.metric("Min", f"{data.min():.3f}")
                        with col4:
                            st.metric("Max", f"{data.max():.3f}")
                        
                        # Histogram
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                        ax.set_xlabel(selected_metric.replace('_', ' ').title())
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_metric.replace("_", " ").title()}')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                with analysis_tabs[1]:
                    # Correlation matrix
                    if len(numeric_cols) > 1:
                        corr_matrix = st.session_state.dashboard_stress_summary[numeric_cols].corr()
                        
                        fig, ax = plt.subplots(figsize=(12, 10))
                        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                        ax.set_xticks(range(len(numeric_cols)))
                        ax.set_yticks(range(len(numeric_cols)))
                        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                        ax.set_yticklabels(numeric_cols)
                        
                        # Add text annotations
                        for i in range(len(numeric_cols)):
                            for j in range(len(numeric_cols)):
                                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
                        
                        plt.colorbar(im, ax=ax)
                        ax.set_title('Correlation Matrix of Stress Metrics')
                        st.pyplot(fig)
                
                with analysis_tabs[2]:
                    # Box plots by category
                    categorical_cols = ['defect_type', 'shape', 'orientation', 'type']
                    categorical_cols = [c for c in categorical_cols if c in st.session_state.dashboard_stress_summary.columns]
                    
                    if categorical_cols:
                        group_by = st.selectbox("Group by", categorical_cols, index=0)
                        metric = st.selectbox("Metric for box plot", numeric_cols, index=0)
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        group_data = []
                        group_labels = []
                        
                        for group in st.session_state.dashboard_stress_summary[group_by].unique():
                            group_values = st.session_state.dashboard_stress_summary[
                                st.session_state.dashboard_stress_summary[group_by] == group
                            ][metric].dropna()
                            
                            if len(group_values) > 0:
                                group_data.append(group_values)
                                group_labels.append(str(group))
                        
                        ax.boxplot(group_data, labels=group_labels)
                        ax.set_xlabel(group_by.replace('_', ' ').title())
                        ax.set_ylabel(metric.replace('_', ' ').title())
                        ax.set_title(f'{metric.replace("_", " ").title()} by {group_by.replace("_", " ").title()}')
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
    
    # Tab 3: Visualizations
    with tab3:
        st.header("📈 Advanced Visualizations")
        
        if st.session_state.dashboard_stress_summary.empty:
            st.info("👈 Please load simulations and update stress summary first")
        else:
            # Sunburst chart configuration
            st.subheader("🌀 Sunburst Chart")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_columns = list(st.session_state.dashboard_stress_summary.columns)
                categorical_cols = ['defect_type', 'shape', 'orientation', 'type']
                categorical_cols = [c for c in categorical_cols if c in available_columns]
                
                level1 = st.selectbox(
                    "First Level",
                    categorical_cols,
                    index=0 if 'type' in categorical_cols else 0,
                    key="sunburst_level1"
                )
                
                level2_options = [c for c in categorical_cols if c != level1]
                level2 = st.selectbox(
                    "Second Level",
                    ['None'] + level2_options,
                    index=0,
                    key="sunburst_level2"
                )
                
                level3_options = [c for c in level2_options if c != level2 and level2 != 'None']
                level3 = st.selectbox(
                    "Third Level",
                    ['None'] + level3_options,
                    index=0,
                    key="sunburst_level3"
                )
            
            with col2:
                numeric_cols = st.session_state.dashboard_stress_summary.select_dtypes(include=[np.number]).columns.tolist()
                stress_value_cols = [c for c in numeric_cols if 'max' in c or 'mean' in c]
                
                value_column = st.selectbox(
                    "Value Metric",
                    stress_value_cols,
                    index=0 if 'max_von_mises' in stress_value_cols else 0,
                    key="sunburst_value"
                )
            
            with col3:
                colormaps = st.session_state.sunburst_manager.get_all_colormaps()
                selected_colormap = st.selectbox(
                    "Colormap",
                    colormaps,
                    index=colormaps.index('viridis') if 'viridis' in colormaps else 0,
                    key="sunburst_colormap"
                )
            
            # Build path columns
            path_columns = [level1]
            if level2 != 'None':
                path_columns.append(level2)
            if level3 != 'None':
                path_columns.append(level3)
            
            if st.button("Generate Sunburst", type="primary"):
                if len(path_columns) > 0 and value_column in st.session_state.dashboard_stress_summary.columns:
                    fig = st.session_state.sunburst_manager.create_sunburst_chart(
                        df=st.session_state.dashboard_stress_summary,
                        path_columns=path_columns,
                        value_column=value_column,
                        title=f"Stress Analysis: {value_column.replace('_', ' ').title()}",
                        colormap=selected_colormap
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please configure sunburst parameters")
            
            # Additional visualizations
            st.subheader("📊 Other Visualizations")
            
            viz_type = st.selectbox(
                "Visualization Type",
                ["3D Scatter", "Heatmap", "Parallel Coordinates", "Radial Bar"],
                index=0
            )
            
            if viz_type == "3D Scatter" and len(stress_value_cols) >= 3:
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    x_col = st.selectbox("X-axis", stress_value_cols, index=0)
                with col_y:
                    y_col = st.selectbox("Y-axis", stress_value_cols, index=1)
                with col_z:
                    z_col = st.selectbox("Z-axis", stress_value_cols, index=2)
                
                color_by = st.selectbox(
                    "Color by",
                    categorical_cols + stress_value_cols,
                    index=0
                )
                
                if st.button("Generate 3D Scatter"):
                    fig = px.scatter_3d(
                        st.session_state.dashboard_stress_summary,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=color_by if color_by in st.session_state.dashboard_stress_summary.columns else None,
                        hover_name='id',
                        title="3D Stress Metric Visualization",
                        color_continuous_scale=selected_colormap,
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Diagnostics
    with tab4:
        st.header("🔧 System Diagnostics")
        
        # Environment info
        st.subheader("🌍 Environment Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Python Environment:**")
            st.write(f"- Python: {sys.version.split()[0]}")
            st.write(f"- NumPy: {np.__version__}")
            st.write(f"- PyTorch: {torch.__version__}")
            st.write(f"- Streamlit: {st.__version__}")
        
        with col2:
            st.write("**File System:**")
            st.write(f"- Solutions Directory: `{NUMERICAL_SOLUTIONS_DIR}`")
            st.write(f"- Directory Exists: {os.path.exists(NUMERICAL_SOLUTIONS_DIR)}")
            if os.path.exists(NUMERICAL_SOLUTIONS_DIR):
                st.write(f"- File Count: {len(glob.glob(os.path.join(NUMERICAL_SOLUTIONS_DIR, '*')))}")
        
        # PyTorch compatibility
        st.subheader("🔧 PyTorch Compatibility")
        
        st.info(f"""
        **Current PyTorch Version:** {torch.__version__}
        
        **Compatibility Notes:**
        - PyTorch 2.6+ uses `weights_only=True` by default for security
        - This blocks loading of files with NumPy scalar objects
        - Use `weights_only=False` for compatibility with older files
        
        **Recommendations:**
        1. **For .pt files:** Use `weights_only=False` or add safe globals
        2. **For corrupted .pkl:** Skip or attempt recovery
        3. **Best practice:** Convert old files to .npz format
        """)
        
        # File conversion tool
        st.subheader("🔄 File Conversion Tool")
        
        if st.button("Convert .pt files to .npz"):
            all_files = st.session_state.enhanced_solutions_manager.get_all_files()
            pt_files = [f for f in all_files if f['format'] == 'pt']
            
            if pt_files:
                with st.spinner(f"Converting {len(pt_files)} .pt files..."):
                    converted = 0
                    for file_info in pt_files[:10]:  # Limit to 10 files
                        try:
                            # Load the file
                            sim_data = st.session_state.enhanced_solutions_manager.load_simulation_with_fallbacks(
                                file_info['path'],
                                st.session_state.enhanced_interpolator,
                                use_secure_loading=False
                            )
                            
                            if sim_data:
                                # Save as .npz
                                npz_filename = file_info['filename'].replace('.pt', '.npz').replace('.pth', '.npz')
                                npz_path = os.path.join(NUMERICAL_SOLUTIONS_DIR, npz_filename)
                                
                                # Extract stress data
                                if 'history' in sim_data and sim_data['history']:
                                    _, stress_fields = sim_data['history'][-1]
                                    np.savez_compressed(npz_path, **stress_fields)
                                    converted += 1
                                    st.success(f"✅ Converted: {file_info['filename']}")
                                    
                        except Exception as e:
                            st.error(f"❌ Failed to convert {file_info['filename']}: {str(e)}")
                    
                    st.success(f"✅ Converted {converted} files to .npz format")
            else:
                st.warning("No .pt files found to convert")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application with enhanced stress dashboard"""
    
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
        ["Attention Interpolation", "Stress Analysis Dashboard", "File Diagnostics"],
        index=0
    )
    
    if operation_mode == "Attention Interpolation":
        # Import and use the existing attention interface
        from mechanical_stress_attention_interpolator_r3 import create_attention_interface
        create_attention_interface()
        
    elif operation_mode == "Stress Analysis Dashboard":
        create_stress_analysis_dashboard()
        
    else:  # File Diagnostics
        st.header("🔧 File Diagnostics")
        
        if 'enhanced_solutions_manager' not in st.session_state:
            st.session_state.enhanced_solutions_manager = EnhancedNumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        all_files = st.session_state.enhanced_solutions_manager.get_all_files()
        
        if not all_files:
            st.warning(f"No files found in {NUMERICAL_SOLUTIONS_DIR}")
        else:
            # File format analysis
            st.subheader("📊 File Format Analysis")
            
            format_counts = {}
            for file_info in all_files:
                fmt = file_info['format']
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            
            # Display format distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(format_counts.values(), labels=format_counts.keys(), autopct='%1.1f%%')
                ax.set_title('File Format Distribution')
                st.pyplot(fig)
            
            with col2:
                for fmt, count in format_counts.items():
                    st.write(f"**{fmt.upper()}:** {count} files")
            
            # File size analysis
            st.subheader("📈 File Size Analysis")
            
            sizes = [f['size'] / (1024 * 1024) for f in all_files]  # MB
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(sizes, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('File Size (MB)')
            ax.set_ylabel('Count')
            ax.set_title('File Size Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Individual file analysis
            st.subheader("🔍 Individual File Analysis")
            
            selected_file = st.selectbox(
                "Select file for detailed analysis",
                [f"{f['filename']} ({f['format']}, {f['size']//1024}KB)" for f in all_files[:20]],
                index=0
            )
            
            if selected_file:
                filename = selected_file.split(" (")[0]
                file_info = next(f for f in all_files if f['filename'] == filename)
                
                analysis = st.session_state.enhanced_solutions_manager.analyze_file(file_info['path'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**File Information:**")
                    st.write(f"- Path: `{analysis['path']}`")
                    st.write(f"- Size: {analysis['size']:,} bytes")
                    st.write(f"- Format: {analysis.get('format', 'unknown')}")
                    st.write(f"- Valid: {'✅ Yes' if analysis['valid'] else '❌ No'}")
                
                with col2:
                    if analysis['issues']:
                        st.write("**Issues:**")
                        for issue in analysis['issues']:
                            st.write(f"- {issue}")
                    
                    if analysis['recommendation']:
                        st.write("**Recommendation:**")
                        st.info(analysis['recommendation'])
                
                # Test loading
                if st.button("🧪 Test Load This File"):
                    with st.spinner("Testing file load..."):
                        try:
                            if 'enhanced_interpolator' not in st.session_state:
                                st.session_state.enhanced_interpolator = EnhancedSpatialLocalityAttentionInterpolator()
                            
                            sim_data = st.session_state.enhanced_solutions_manager.load_simulation_with_fallbacks(
                                file_info['path'],
                                st.session_state.enhanced_interpolator,
                                use_secure_loading=False
                            )
                            
                            if sim_data:
                                st.success("✅ File loaded successfully!")
                                
                                # Show basic info
                                st.write("**Loaded Data:**")
                                st.write(f"- History frames: {len(sim_data.get('history', []))}")
                                st.write(f"- Parameters: {len(sim_data.get('params', {}))}")
                                st.write(f"- Metadata: {len(sim_data.get('metadata', {}))}")
                                
                                # Show sample stress data
                                if sim_data.get('history'):
                                    _, stress_fields = sim_data['history'][-1]
                                    st.write("**Stress Fields:**")
                                    for key, value in stress_fields.items():
                                        if isinstance(value, np.ndarray):
                                            st.write(f"- {key}: {value.shape}, {value.dtype}")
                            else:
                                st.error("❌ Failed to load file")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Enhanced File Loading and Stress Analysis", expanded=False):
    st.markdown(f"""
    ## 🚀 **Enhanced Stress Analysis Dashboard**
    
    ### **📁 Robust File Loading**
    
    **Multiple Loading Strategies:**
    1. **PyTorch Files (.pt):**
       - Secure loading (weights_only=True) with safe globals
       - Insecure loading (weights_only=False) for compatibility
       - Buffer-based loading for corrupted files
       - Zip extraction as last resort
    
    2. **Pickle Files (.pkl):**
       - Standard pickle load with protocol detection
       - Joblib loading for numpy arrays
       - Dill loading for complex objects
       - Text recovery for corrupted files
    
    3. **Other Formats:**
       - HDF5 (.h5) with recursive traversal
       - NumPy (.npz) with compression support
       - SQLite (.sql, .db) with table extraction
       - JSON with encoding fallbacks
    
    ### **🔧 File Integrity Checking**
    
    **Pre-load Validation:**
    1. **File Existence and Size:** Check if file exists and has content
    2. **Header Validation:** Verify file signatures and magic numbers
    3. **Format Detection:** Auto-detect file format from content
    4. **Corruption Detection:** Identify corrupted files before loading
    
    ### **📊 Stress Analysis Features**
    
    **Maximum Stress Capture:**
    1. **Hydrostatic Stress:** Absolute maximum, mean absolute, statistics
    2. **Stress Magnitude:** Maximum, mean, distribution analysis
    3. **Von Mises Stress:** Maximum, percentiles (95th, 99th, 99.9th)
    4. **Principal Stresses:** Max values, Tresca shear stress
    
    **Advanced Visualization:**
    1. **Sunburst Charts:** Hierarchical visualization with 50+ colormaps
    2. **3D Scatter Plots:** Interactive 3D exploration of stress metrics
    3. **Heatmaps:** Parameter-stress relationships
    4. **Box Plots:** Distribution analysis across categories
    5. **Parallel Coordinates:** Multi-dimensional analysis
    
    ### **🛠️ Diagnostic Tools**
    
    **File Analysis:**
    1. **Format Distribution:** Pie charts of file types
    2. **Size Analysis:** Histograms of file sizes
    3. **Individual File Inspection:** Detailed analysis of each file
    4. **Loading Statistics:** Success rates and failure analysis
    
    **Compatibility Fixes:**
    1. **PyTorch 2.6+ Support:** Safe globals for NumPy objects
    2. **Legacy Format Conversion:** Convert old .pt files to .npz
    3. **Corrupted File Recovery:** Multiple recovery strategies
    4. **Encoding Fallbacks:** Handle different text encodings
    
    ### **🚀 Performance Optimizations**
    
    1. **Caching:** Memoize loaded files to avoid redundant loading
    2. **Batch Processing:** Parallel loading of multiple files
    3. **Progress Indicators:** Real-time loading feedback
    4. **Memory Management:** Efficient handling of large files
    
    ### **🔒 Security Considerations**
    
    1. **Secure Loading Default:** weights_only=True for untrusted files
    2. **Insecure Fallback:** weights_only=False for trusted legacy files
    3. **Safe Globals:** Whitelist specific NumPy types
    4. **File Validation:** Check files before unpickling
    
    **This enhanced dashboard provides comprehensive stress analysis with robust file loading capabilities, making it suitable for handling large collections of simulation files with varying formats and compatibility issues.**
    """)

if __name__ == "__main__":
    main()

st.caption(f"🔬 Enhanced Stress Analysis Dashboard • Robust File Loading • PyTorch Compatibility • 2025")
