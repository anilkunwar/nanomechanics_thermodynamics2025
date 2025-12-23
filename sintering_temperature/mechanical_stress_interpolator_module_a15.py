import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
import plotly.graph_objects as go
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
from numba import jit, prange
import time
import hashlib
import sqlite3
from pathlib import Path
import tempfile
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import base64
import seaborn as sns
from scipy import ndimage
import cmasher as cmr  # For additional colormaps
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# =============================================
# ENHANCED COLOR MAPS (50+ COLORMAPS)
# =============================================
class EnhancedColorMaps:
    """Enhanced colormap collection with 50+ options"""
    
    @staticmethod
    def get_all_colormaps():
        """Return all available colormaps categorized by type"""
        
        # Standard matplotlib colormaps
        standard_maps = [
            # Sequential
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'summer', 'autumn', 'winter', 'spring', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'bone', 'gray', 'pink', 'binary',
            
            # Diverging
            'coolwarm', 'bwr', 'seismic', 'RdBu', 'RdYlBu',
            'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr',
            'Spectral',
            
            # Cyclic
            'twilight', 'twilight_shifted', 'hsv',
            
            # Qualitative
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'Set1', 'Set2', 'Set3',
            'Pastel1', 'Pastel2',
            'Dark2', 'Paired',
            'Accent',
            
            # Miscellaneous
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
            'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
            'cubehelix', 'brg', 'gist_rainbow', 'rainbow',
            'jet', 'nipy_spectral', 'gist_ncar'
        ]
        
        # Custom enhanced maps
        custom_maps = {
            'stress_cmap': LinearSegmentedColormap.from_list(
                'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
            ),
            'turbo': 'turbo',
            'deep': 'viridis',
            'dense': 'plasma',
            'matter': 'inferno',
            'speed': 'magma',
            'amp': 'cividis',
            'tempo': 'twilight',
            'phase': 'twilight_shifted',
            'balance': 'RdBu_r',
            'delta': 'coolwarm',
            'curl': 'PuOr_r',
            'diff': 'seismic',
            'tarn': 'terrain',
            'topo': 'gist_earth',
            'oxy': 'ocean',
            'deep_r': 'viridis_r',
            'dense_r': 'plasma_r',
            'ice': 'Blues',
            'fire': 'Reds',
            'earth': 'YlOrBr',
            'water': 'PuBu',
            'forest': 'Greens',
            'sunset': 'YlOrRd',
            'dawn': 'Purples',
            'night': 'Blues_r',
            'aurora': 'gist_ncar',
            'spectrum': 'Spectral',
            'prism_enhanced': 'prism',
            'pastel_rainbow': ListedColormap(plt.cm.rainbow(np.linspace(0, 1, 256)) * 0.7 + 0.3),
            'high_contrast': LinearSegmentedColormap.from_list(
                'high_contrast', ['#000000', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#FFFFFF']
            )
        }
        
        # Combine all maps
        all_maps = standard_maps + list(custom_maps.keys())
        
        # Remove duplicates
        return list(dict.fromkeys(all_maps))
    
    @staticmethod
    def get_colormap(cmap_name):
        """Get a colormap by name with fallback"""
        try:
            if cmap_name == 'stress_cmap':
                return LinearSegmentedColormap.from_list(
                    'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
                )
            elif cmap_name == 'pastel_rainbow':
                return ListedColormap(plt.cm.rainbow(np.linspace(0, 1, 256)) * 0.7 + 0.3)
            elif cmap_name == 'high_contrast':
                return LinearSegmentedColormap.from_list(
                    'high_contrast', ['#000000', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000', '#FFFFFF']
                )
            else:
                return plt.cm.get_cmap(cmap_name)
        except:
            # Fallback to viridis
            return plt.cm.viridis

# Initialize colormaps
COLORMAP_MANAGER = EnhancedColorMaps()
ALL_COLORMAPS = COLORMAP_MANAGER.get_all_colormaps()

# =============================================
# REGION ANALYSIS FUNCTIONS - FIXED FOR NUMBA
# =============================================

# Use simple Python functions instead of @jit for dictionary handling
def extract_region_stress(eta, stress_fields, region_type, stress_component='von_mises', stress_type='max_abs'):
    """Extract stress from specific regions (defect, interface, bulk)"""
    if eta is None or not isinstance(eta, np.ndarray):
        return 0.0
    
    # Create mask for the region
    if region_type == 'defect':
        mask = eta > 0.6
    elif region_type == 'interface':
        mask = (eta >= 0.4) & (eta <= 0.6)
    elif region_type == 'bulk':
        mask = eta < 0.4
    else:
        mask = np.ones_like(eta, dtype=bool)
    
    if not np.any(mask):
        return 0.0
    
    # Get stress data
    stress_data = np.zeros_like(eta)
    if stress_component == 'von_mises' and 'von_mises' in stress_fields:
        stress_data = stress_fields['von_mises']
    elif stress_component == 'sigma_hydro' and 'sigma_hydro' in stress_fields:
        stress_data = stress_fields['sigma_hydro']
    elif stress_component == 'sigma_mag' and 'sigma_mag' in stress_fields:
        stress_data = stress_fields['sigma_mag']
    
    # Extract region stress
    region_stress = stress_data[mask]
    
    if stress_type == 'max_abs':
        return np.max(np.abs(region_stress)) if len(region_stress) > 0 else 0.0
    elif stress_type == 'mean_abs':
        return np.mean(np.abs(region_stress)) if len(region_stress) > 0 else 0.0
    elif stress_type == 'max':
        return np.max(region_stress) if len(region_stress) > 0 else 0.0
    elif stress_type == 'min':
        return np.min(region_stress) if len(region_stress) > 0 else 0.0
    elif stress_type == 'mean':
        return np.mean(region_stress) if len(region_stress) > 0 else 0.0
    else:
        return np.mean(np.abs(region_stress)) if len(region_stress) > 0 else 0.0

def extract_region_statistics(eta, stress_fields, region_type):
    """Extract comprehensive statistics for a region"""
    if eta is None or not isinstance(eta, np.ndarray):
        return {}
    
    # Create mask for the region
    if region_type == 'defect':
        mask = eta > 0.6
    elif region_type == 'interface':
        mask = (eta >= 0.4) & (eta <= 0.6)
    elif region_type == 'bulk':
        mask = eta < 0.4
    else:
        mask = np.ones_like(eta, dtype=bool)
    
    if not np.any(mask):
        return {
            'area_fraction': 0.0,
            'von_mises': {'max': 0.0, 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'mean_abs': 0.0},
            'sigma_hydro': {'max': 0.0, 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'mean_abs': 0.0},
            'sigma_mag': {'max': 0.0, 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'mean_abs': 0.0}
        }
    
    area_fraction = np.sum(mask) / mask.size
    
    results = {'area_fraction': float(area_fraction)}
    
    # Analyze each stress component
    for comp_name in ['von_mises', 'sigma_hydro', 'sigma_mag']:
        if comp_name in stress_fields:
            stress_data = stress_fields[comp_name][mask]
            if len(stress_data) > 0:
                results[comp_name] = {
                    'max': float(np.max(stress_data)),
                    'min': float(np.min(stress_data)),
                    'mean': float(np.mean(stress_data)),
                    'std': float(np.std(stress_data)),
                    'max_abs': float(np.max(np.abs(stress_data))),
                    'mean_abs': float(np.mean(np.abs(stress_data))),
                    'percentile_95': float(np.percentile(np.abs(stress_data), 95)),
                    'percentile_99': float(np.percentile(np.abs(stress_data), 99))
                }
            else:
                results[comp_name] = {
                    'max': 0.0, 'min': 0.0, 'mean': 0.0, 'std': 0.0,
                    'max_abs': 0.0, 'mean_abs': 0.0,
                    'percentile_95': 0.0, 'percentile_99': 0.0
                }
    
    return results

# =============================================
# NUMBA-ACCELERATED FUNCTIONS (Existing)
# =============================================

@jit(nopython=True, parallel=True)
def compute_gaussian_weights_numba(source_vectors, target_vector, sigma):
    """Numba-accelerated Gaussian weight computation"""
    n_sources = source_vectors.shape[0]
    weights = np.zeros(n_sources)
    
    for i in prange(n_sources):
        dist_sq = 0.0
        for j in range(source_vectors.shape[1]):
            diff = source_vectors[i, j] - target_vector[j]
            dist_sq += diff * diff
        weights[i] = np.exp(-0.5 * dist_sq / (sigma * sigma))
    
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    else:
        weights = np.ones(n_sources) / n_sources
    
    return weights

@jit(nopython=True)
def compute_stress_statistics_numba(stress_matrix):
    """Compute stress statistics efficiently"""
    flat_stress = stress_matrix.flatten()
    
    max_val = np.max(flat_stress)
    min_val = np.min(flat_stress)
    mean_val = np.mean(flat_stress)
    std_val = np.std(flat_stress)
    percentile_95 = np.percentile(flat_stress, 95)
    percentile_99 = np.percentile(flat_stress, 99)
    
    return max_val, min_val, mean_val, std_val, percentile_95, percentile_99

# =============================================
# ENHANCED NUMERICAL SOLUTIONS LOADER (Existing - Fixed)
# =============================================
class EnhancedSolutionLoader:
    """Enhanced solution loader with support for multiple formats and caching"""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.pt_loading_method = "safe"  # "safe" or "unsafe"
        
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
            if 'st' in globals():
                st.info(f"Created directory: {self.solutions_dir}")
    
    def scan_solutions(self) -> Dict[str, List[str]]:
        """Scan directory for solution files"""
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
    
    def get_all_files_info(self) -> List[Dict[str, Any]]:
        """Get information about all solution files"""
        all_files = []
        file_formats = self.scan_solutions()
        
        for format_type, files in file_formats.items():
            for file_path in files:
                try:
                    file_info = {
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'format': format_type,
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        'relative_path': os.path.relpath(file_path, self.solutions_dir)
                    }
                    all_files.append(file_info)
                except Exception as e:
                    if 'st' in globals():
                        st.warning(f"Could not get info for {file_path}: {e}")
        
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
    
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        try:
            data = pickle.load(buffer)
            if isinstance(data, Exception):
                return data
            return data
        except Exception as e:
            return e
    
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        try:
            if self.pt_loading_method == "safe":
                # FIXED: Handle PyTorch 2.6 weights_only=True issue
                try:
                    # Try to import the numpy scalar class
                    import numpy as np
                    # Different numpy versions have different paths
                    try:
                        from numpy._core.multiarray import scalar as np_scalar
                    except ImportError:
                        try:
                            from numpy.core.multiarray import scalar as np_scalar
                        except ImportError:
                            np_scalar = None
                    
                    if np_scalar is not None:
                        # Use safe_globals context manager
                        import torch.serialization
                        with torch.serialization.safe_globals([np_scalar]):
                            data = torch.load(buffer, map_location='cpu', weights_only=True)
                    else:
                        # Fallback to weights_only=False with warning
                        if 'st' in globals():
                            st.warning("Could not import numpy scalar, using weights_only=False")
                        data = torch.load(buffer, map_location='cpu', weights_only=False)
                except Exception as safe_error:
                    # If safe loading fails, try unsafe as last resort
                    if 'st' in globals():
                        st.warning(f"Safe loading failed: {safe_error}. Trying weights_only=False")
                    data = torch.load(buffer, map_location='cpu', weights_only=False)
            else:
                # Unsafe loading (only use if you trust the source)
                data = torch.load(buffer, map_location='cpu', weights_only=False)
            
            # Convert tensors to numpy arrays for compatibility
            if isinstance(data, dict):
                for key in list(data.keys()):
                    if torch.is_tensor(data[key]):
                        data[key] = data[key].cpu().numpy()
                    elif isinstance(data[key], dict):
                        for subkey in list(data[key].keys()):
                            if torch.is_tensor(data[key][subkey]):
                                data[key][subkey] = data[key][subkey].cpu().numpy()
            return data
        except Exception as e:
            return e
    
    def _read_h5(self, file_content):
        try:
            import h5py
            buffer = BytesIO(file_content)
            with h5py.File(buffer, 'r') as f:
                data = {}
                def read_h5_obj(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = obj[()]
                    elif isinstance(obj, h5py.Group):
                        data[name] = {}
                        for key in obj.keys():
                            read_h5_obj(f"{name}/{key}", obj[key])
                for key in f.keys():
                    read_h5_obj(key, f[key])
            return data
        except Exception as e:
            return e
    
    def _read_npz(self, file_content):
        buffer = BytesIO(file_content)
        try:
            data = np.load(buffer, allow_pickle=True)
            return {key: data[key] for key in data.files}
        except Exception as e:
            return e
    
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
                    'rows': rows
                }
            
            conn.close()
            os.unlink(tmp_path)
            return data
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return e
    
    def _read_json(self, file_content):
        try:
            return json.loads(file_content.decode('utf-8'))
        except Exception as e:
            return e
    
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file with format auto-detection and error handling"""
        if format_type == 'auto':
            # Try to auto-detect based on content or extension
            format_type = 'pkl'  # Default fallback
        
        readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
        
        if format_type in readers:
            data = readers[format_type](file_content)
            
            # Check if the reader returned an error/exception
            if isinstance(data, Exception):
                # Return a structured error object
                return {
                    'error': str(data),
                    'format': format_type,
                    'status': 'error'
                }
            
            # Standardize the data
            standardized = self._standardize_data(data, format_type)
            standardized['status'] = 'success'
            return standardized
        else:
            error_msg = f"Unsupported format: {format_type}"
            return {
                'error': error_msg,
                'format': format_type,
                'status': 'error'
            }
    
    def _standardize_data(self, data, format_type):
        """Standardize simulation data structure with robust error handling"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type
        }
        
        try:
            if format_type == 'pkl':
                if isinstance(data, dict):
                    standardized['params'] = data.get('params', {})
                    standardized['metadata'] = data.get('metadata', {})
                    standardized['history'] = data.get('history', [])
                else:
                    standardized['error'] = f"PKL data is not a dictionary: {type(data)}"
            
            elif format_type == 'pt':
                if isinstance(data, dict):
                    standardized['params'] = data.get('params', {})
                    standardized['metadata'] = data.get('metadata', {})
                    
                    # Handle history - may be stored differently
                    history = data.get('history', [])
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        # Convert dict history to list format
                        history_list = []
                        for key in sorted(history.keys()):
                            frame = history[key]
                            if isinstance(frame, dict) and 'eta' in frame and 'stresses' in frame:
                                history_list.append((frame['eta'], frame['stresses']))
                        standardized['history'] = history_list
                    
                    # Additional cleanup for tensor data
                    if 'params' in standardized:
                        for key, value in standardized['params'].items():
                            if torch.is_tensor(value):
                                standardized['params'][key] = value.cpu().numpy()
                else:
                    standardized['error'] = f"PT data is not a dictionary: {type(data)}"
            
            elif format_type == 'h5':
                # Handle H5 format
                if isinstance(data, dict):
                    standardized.update(data)
                else:
                    standardized['error'] = f"H5 data is not a dictionary: {type(data)}"
            
            elif format_type == 'npz':
                # Handle NPZ format
                if isinstance(data, dict):
                    standardized.update(data)
                else:
                    standardized['error'] = f"NPZ data is not a dictionary: {type(data)}"
            
            elif format_type == 'json':
                if isinstance(data, dict):
                    standardized['params'] = data.get('params', {})
                    standardized['metadata'] = data.get('metadata', {})
                    standardized['history'] = data.get('history', [])
                else:
                    standardized['error'] = f"JSON data is not a dictionary: {type(data)}"
            
        except Exception as e:
            standardized['error'] = f"Standardization error: {str(e)}"
        
        return standardized
    
    def load_all_solutions(self, use_cache=True, pt_loading_method="safe"):
        """Load all solutions with caching, progress tracking, and error handling"""
        self.pt_loading_method = pt_loading_method
        solutions = []
        failed_files = []
        
        if not os.path.exists(self.solutions_dir):
            if 'st' in globals():
                st.warning(f"Directory {self.solutions_dir} not found. Creating it.")
            os.makedirs(self.solutions_dir, exist_ok=True)
            return solutions
        
        all_files_info = self.get_all_files_info()
        
        if not all_files_info:
            if 'st' in globals():
                st.info(f"No solution files found in {self.solutions_dir}")
            return solutions
        
        if 'st' in globals():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for idx, file_info in enumerate(all_files_info):
            try:
                file_path = file_info['path']
                filename = file_info['filename']
                
                # Check cache
                cache_key = f"{filename}_{os.path.getmtime(file_path)}_{pt_loading_method}"
                if use_cache and cache_key in self.cache:
                    sim = self.cache[cache_key]
                    if sim.get('status') == 'success':
                        solutions.append(sim)
                    continue
                
                # Update progress
                if 'st' in globals():
                    progress = (idx + 1) / len(all_files_info)
                    progress_bar.progress(progress)
                    status_text.text(f"Loading {filename}...")
                
                # Load file
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                sim = self.read_simulation_file(file_content, file_info['format'])
                sim['filename'] = filename
                sim['file_info'] = file_info
                
                # Validate structure
                if sim.get('status') == 'success' and 'params' in sim and 'history' in sim:
                    # Validate params structure
                    if isinstance(sim['params'], dict):
                        # Cache the solution
                        self.cache[cache_key] = sim
                        solutions.append(sim)
                    else:
                        failed_files.append({
                            'filename': filename,
                            'error': f"Params is not a dictionary: {type(sim['params'])}"
                        })
                else:
                    error_msg = sim.get('error', 'Unknown error or missing params/history')
                    failed_files.append({
                        'filename': filename,
                        'error': error_msg
                    })
                    
            except Exception as e:
                failed_files.append({
                    'filename': file_info['filename'],
                    'error': f"Loading error: {str(e)}"
                })
        
        if 'st' in globals():
            progress_bar.empty()
            status_text.empty()
        
        # Display failed files if any
        if failed_files and 'st' in globals():
            with st.expander(f"⚠️ Failed to load {len(failed_files)} files", expanded=False):
                for failed in failed_files[:10]:  # Show first 10
                    st.error(f"**{failed['filename']}**: {failed['error']}")
                if len(failed_files) > 10:
                    st.info(f"... and {len(failed_files) - 10} more files failed to load.")
        
        return solutions

# =============================================
# ENHANCED SPATIAL INTERPOLATOR WITH EUCLIDEAN DISTANCE
# =============================================

class EnhancedSpatialInterpolator:
    """Enhanced interpolator with proper Euclidean distance regularization"""
    
    def __init__(self, sigma=0.3, use_spatial_locality=True, spatial_weight=1.0):
        super().__init__()
        self.sigma = sigma
        self.use_spatial_locality = use_spatial_locality
        self.spatial_weight = spatial_weight
        
        # Parameter mappings
        self.defect_map = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        
        self.shape_map = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
    
    def compute_parameter_vector(self, params):
        """Convert parameters to numerical vector with error handling"""
        vector = []
        
        # Check if params is a dictionary
        if not isinstance(params, dict):
            # Return default vector
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0.0], dtype=np.float32)
        
        # Defect type
        defect = params.get('defect_type', 'ISF')
        vector.extend(self.defect_map.get(defect, [0, 0, 0]))
        
        # Shape
        shape = params.get('shape', 'Square')
        vector.extend(self.shape_map.get(shape, [0, 0, 0, 0, 0]))
        
        # Numeric parameters (normalized)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        vector.append((eps0 - 0.3) / (3.0 - 0.3))  # eps0 normalized
        vector.append((kappa - 0.1) / (2.0 - 0.1))  # kappa normalized
        vector.append(theta / (np.pi / 2))  # theta normalized 0-pi/2
        
        return np.array(vector, dtype=np.float32)
    
    def compute_spatial_distance(self, source_params, target_params):
        """Compute spatial distance between parameter vectors"""
        source_vector = self.compute_parameter_vector(source_params)
        target_vector = self.compute_parameter_vector(target_params)
        
        # Euclidean distance
        distance = np.sqrt(np.sum((source_vector - target_vector) ** 2))
        return distance
    
    def compute_spatial_weights(self, sources, target_params):
        """Compute weights based on spatial locality (Euclidean distance)"""
        if not sources:
            return np.array([])
        
        distances = []
        for src in sources:
            if 'params' in src:
                dist = self.compute_spatial_distance(src['params'], target_params)
                distances.append(dist)
            else:
                distances.append(1.0)  # Default distance for invalid sources
        
        distances = np.array(distances)
        
        # Convert distances to weights using Gaussian kernel
        if np.any(distances > 0):
            weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
            # Normalize
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones_like(weights) / len(weights)
        else:
            weights = np.ones_like(distances) / len(distances)
        
        return weights
    
    def interpolate_with_spatial_locality(self, sources, target_params, region_type='bulk', 
                                          stress_component='von_mises', stress_type='max_abs'):
        """Interpolate with proper spatial locality regularization"""
        
        # Filter and validate sources
        valid_sources = []
        for src in sources:
            if not isinstance(src, dict):
                continue
            if 'params' not in src or 'history' not in src:
                continue
            if not isinstance(src.get('params'), dict):
                continue
            valid_sources.append(src)
        
        if not valid_sources:
            return None
        
        # Compute spatial weights
        spatial_weights = self.compute_spatial_weights(valid_sources, target_params)
        
        # Extract region stress from each source
        source_stresses = []
        source_stats = []
        
        for src in valid_sources:
            history = src.get('history', [])
            if history:
                # Get the last frame
                last_frame = history[-1]
                
                # Handle different frame formats
                if isinstance(last_frame, tuple) and len(last_frame) >= 2:
                    eta, stress_fields = last_frame[0], last_frame[1]
                elif isinstance(last_frame, dict):
                    eta = last_frame.get('eta', np.zeros((128, 128)))
                    stress_fields = last_frame.get('stresses', {})
                else:
                    eta = np.zeros((128, 128))
                    stress_fields = {}
                
                # Extract region stress
                region_stress = extract_region_stress(eta, stress_fields, region_type, 
                                                     stress_component, stress_type)
                source_stresses.append(region_stress)
                
                # Extract comprehensive statistics
                stats = extract_region_statistics(eta, stress_fields, region_type)
                source_stats.append(stats)
            else:
                source_stresses.append(0.0)
                source_stats.append({})
        
        # Weighted combination
        weighted_stress = np.sum(spatial_weights * np.array(source_stresses))
        
        # Combine statistics
        combined_stats = {}
        if source_stats and source_stats[0]:
            for key in source_stats[0].keys():
                if key == 'area_fraction':
                    # Weighted average for area fraction
                    area_fractions = [stats.get(key, 0.0) for stats in source_stats]
                    combined_stats[key] = np.sum(spatial_weights * np.array(area_fractions))
                elif isinstance(source_stats[0][key], dict):
                    # Weighted average for stress statistics
                    combined_stats[key] = {}
                    for subkey in source_stats[0][key].keys():
                        values = [stats.get(key, {}).get(subkey, 0.0) for stats in source_stats]
                        combined_stats[key][subkey] = np.sum(spatial_weights * np.array(values))
        
        return {
            'region_stress': weighted_stress,
            'region_statistics': combined_stats,
            'spatial_weights': spatial_weights,
            'source_stresses': source_stresses,
            'source_statistics': source_stats,
            'target_params': target_params,
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type,
            'num_valid_sources': len(valid_sources)
        }

# =============================================
# ORIGINAL FILE ANALYSIS CLASS
# =============================================

class OriginalFileAnalyzer:
    """Analyze original loaded files for different regions"""
    
    def __init__(self):
        self.region_definitions = {
            'defect': {'min': 0.6, 'max': 1.0, 'name': 'Defect Region (η > 0.6)'},
            'interface': {'min': 0.4, 'max': 0.6, 'name': 'Interface Region (0.4 ≤ η ≤ 0.6)'},
            'bulk': {'min': 0.0, 'max': 0.4, 'name': 'Bulk Ag Material (η < 0.4)'}
        }
    
    def analyze_solution(self, solution, region_type='bulk', 
                        stress_component='von_mises', stress_type='max_abs'):
        """Analyze a single solution for a specific region"""
        if not solution or 'history' not in solution:
            return None
        
        history = solution.get('history', [])
        if not history:
            return None
        
        # Get the last frame
        last_frame = history[-1]
        
        # Extract eta and stress fields
        if isinstance(last_frame, tuple) and len(last_frame) >= 2:
            eta, stress_fields = last_frame[0], last_frame[1]
        elif isinstance(last_frame, dict):
            eta = last_frame.get('eta', np.zeros((128, 128)))
            stress_fields = last_frame.get('stresses', {})
        else:
            return None
        
        # Extract region stress
        region_stress = extract_region_stress(eta, stress_fields, region_type, 
                                             stress_component, stress_type)
        
        # Extract comprehensive statistics
        region_stats = extract_region_statistics(eta, stress_fields, region_type)
        
        # Get solution parameters
        params = solution.get('params', {})
        
        return {
            'region_stress': region_stress,
            'region_statistics': region_stats,
            'params': params,
            'filename': solution.get('filename', 'Unknown'),
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type
        }
    
    def analyze_all_solutions(self, solutions, region_type='bulk', 
                             stress_component='von_mises', stress_type='max_abs'):
        """Analyze all solutions for a specific region"""
        results = []
        
        for sol in solutions:
            analysis = self.analyze_solution(sol, region_type, stress_component, stress_type)
            if analysis:
                results.append(analysis)
        
        return results
    
    def create_stress_matrix_from_original(self, solutions, region_type='bulk', 
                                          stress_component='von_mises', stress_type='max_abs'):
        """Create stress matrix from original solutions for visualization"""
        if not solutions:
            return None, None, None
        
        # Group solutions by theta
        solutions_by_theta = {}
        for sol in solutions:
            params = sol.get('params', {})
            theta = params.get('theta', 0.0)
            theta_deg = np.rad2deg(theta)
            
            if theta_deg not in solutions_by_theta:
                solutions_by_theta[theta_deg] = []
            solutions_by_theta[theta_deg].append(sol)
        
        # Get unique thetas
        thetas = np.array(sorted(solutions_by_theta.keys()))
        
        # Determine time points from first solution
        first_sol = solutions[0]
        history = first_sol.get('history', [])
        times = np.arange(len(history))
        
        # Initialize stress matrix
        stress_matrix = np.zeros((len(times), len(thetas)))
        
        # Fill matrix
        for theta_idx, theta in enumerate(thetas):
            theta_solutions = solutions_by_theta.get(theta, [])
            if not theta_solutions:
                continue
            
            # Use the first solution for this theta
            sol = theta_solutions[0]
            history = sol.get('history', [])
            
            for time_idx in range(min(len(history), len(times))):
                frame = history[time_idx]
                
                # Extract eta and stress fields
                if isinstance(frame, tuple) and len(frame) >= 2:
                    eta, stress_fields = frame[0], frame[1]
                elif isinstance(frame, dict):
                    eta = frame.get('eta', np.zeros((128, 128)))
                    stress_fields = frame.get('stresses', {})
                else:
                    continue
                
                # Extract region stress
                region_stress = extract_region_stress(eta, stress_fields, region_type,
                                                     stress_component, stress_type)
                stress_matrix[time_idx, theta_idx] = region_stress
        
        return stress_matrix, times, thetas

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER (Existing - with region support)
# =============================================
class EnhancedSunburstRadarVisualizer:
    """Enhanced sunburst and radar charts with 50+ colormaps and visualization enhancements"""
    
    def __init__(self):
        self.colormap_manager = EnhancedColorMaps()
    
    def create_enhanced_plotly_sunburst(self, stress_matrix, times, thetas, title, 
                                       cmap='rainbow', marker_size=12, line_width=1.5,
                                       font_size=18, width=900, height=750,
                                       show_colorbar=True, colorbar_title="Stress (GPa)",
                                       hover_template=None):
        """Interactive sunburst with Plotly - fully enhanced version"""
        
        # Prepare data for polar scatter
        theta_deg = np.deg2rad(thetas)
        theta_grid, time_grid = np.meshgrid(theta_deg, times)
        
        # Flatten the arrays for scatter plot
        r_flat = time_grid.flatten()
        theta_flat = np.rad2deg(theta_grid).flatten()
        stress_flat = stress_matrix.flatten()
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Default hover template
        if hover_template is None:
            hover_template = (
                '<b>Time</b>: %{r:.2f}s<br>' +
                '<b>Orientation</b>: %{theta:.1f}°<br>' +
                '<b>Stress</b>: %{marker.color:.4f} GPa<br>' +
                '<extra></extra>'
            )
        
        # Add scatter polar trace with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=r_flat,
            theta=theta_flat,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=stress_flat,
                colorscale=cmap,
                showscale=show_colorbar,
                colorbar=dict(
                    title=dict(text=colorbar_title, font=dict(size=font_size, color='black')),
                    tickfont=dict(size=font_size-2, color='black'),
                    thickness=25,
                    len=0.8,
                    x=1.15,
                    xpad=20,
                    ypad=20,
                    tickformat='.3f',
                    title_side='right'
                ),
                line=dict(width=line_width, color='rgba(255, 255, 255, 0.8)'),
                opacity=0.9,
                symbol='circle',
                sizemode='diameter',
                sizemin=3
            ),
            hovertemplate=hover_template,
            name='Stress Distribution'
        ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=font_size+4, family="Arial Black, sans-serif", color='darkblue'),
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top'
            ),
            polar=dict(
                radialaxis=dict(
                    title=dict(
                        text="Time (s)",
                        font=dict(size=font_size+2, color='black', family='Arial')
                    ),
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    showline=True,
                    tickfont=dict(size=font_size, color='black', family='Arial'),
                    tickformat='.1f',
                    range=[0, max(times) * 1.1],
                    ticksuffix=" s",
                    showticksuffix='all'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickfont=dict(size=font_size, color='black', family='Arial'),
                    tickmode='array',
                    tickvals=list(range(0, 360, 30)),
                    ticktext=[f'{i}°' for i in range(0, 360, 30)],
                    period=360,
                    thetaunit="degrees"
                ),
                bgcolor="rgba(240, 240, 240, 0.5)",
                sector=[0, 360],
                hole=0.1
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                x=1.2,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=font_size, family='Arial')
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=100, r=200, t=100, b=100),
            font=dict(family="Arial, sans-serif", size=font_size)
        )
        
        # Add radial lines for orientation reference
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            fig.add_trace(go.Scatterpolar(
                r=[0, max(times)],
                theta=[angle, angle],
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        return fig
    
    def create_enhanced_plotly_radar(self, stress_values, thetas, component_name, time_point,
                                    line_width=4, marker_size=12, fill_alpha=0.3,
                                    font_size=16, width=800, height=700,
                                    show_mean=True, show_std=True, color='steelblue'):
        """Interactive enhanced radar chart with Plotly"""
        
        # Ensure proper closure
        thetas_closed = np.append(thetas, 360)
        stress_values_closed = np.append(stress_values, stress_values[0])
        
        # Create figure
        fig = go.Figure()
        
        # Add radar trace with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=stress_values_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor=f'rgba(70, 130, 180, {fill_alpha})',
            line=dict(color=color, width=line_width),
            marker=dict(size=marker_size, color=color, symbol='circle'),
            name=component_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Stress: %{r:.4f} GPa',
            text=[f'{v:.3f} GPa' for v in stress_values_closed],
            textposition='top center'
        ))
        
        # Add mean value line
        if show_mean:
            mean_val = np.mean(stress_values)
            fig.add_trace(go.Scatterpolar(
                r=[mean_val] * len(thetas_closed),
                theta=thetas_closed,
                mode='lines',
                line=dict(color='firebrick', width=3, dash='dash'),
                name=f'Mean: {mean_val:.3f} GPa',
                hovertemplate='Mean Stress: %{r:.3f} GPa'
            ))
        
        # Add standard deviation band
        if show_std:
            mean_val = np.mean(stress_values)
            std_val = np.std(stress_values)
            fig.add_trace(go.Scatterpolar(
                r=[mean_val + std_val] * len(thetas_closed),
                theta=thetas_closed,
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name=f'Mean ± Std: {std_val:.3f} GPa',
                hovertemplate='Mean + Std: %{r:.3f} GPa'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=[mean_val - std_val] * len(thetas_closed),
                theta=thetas_closed,
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name=f'Mean - Std: {std_val:.3f} GPa',
                hovertemplate='Mean - Std: %{r:.3f} GPa',
                showlegend=False
            ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f'{component_name} Stress at t={time_point:.1f}s',
                font=dict(size=font_size+4, family="Arial Black", color='darkblue'),
                x=0.5,
                xanchor='center'
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(stress_values) * 1.3],
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    tickfont=dict(size=font_size-2, color='black'),
                    title=dict(text='Stress (GPa)', 
                              font=dict(size=font_size, color='black')),
                    ticksuffix=' GPa'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickvals=list(range(0, 360, 45)),
                    ticktext=[f'{i}°' for i in range(0, 360, 45)],
                    tickfont=dict(size=font_size, color='black'),
                    period=360
                ),
                bgcolor="rgba(240, 240, 240, 0.5)"
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=font_size, family='Arial')
            ),
            width=width,
            height=height
        )
        
        return fig

# =============================================
# ENHANCED COMPARISON VISUALIZER
# =============================================

class EnhancedComparisonVisualizer:
    """Visualizer for comparing original and interpolated solutions"""
    
    def __init__(self):
        self.original_analyzer = OriginalFileAnalyzer()
    
    def create_comparison_sunburst(self, original_matrix, interpolated_matrix, 
                                  thetas, times, region_name, stress_component):
        """Create comparison sunburst plot"""
        
        # Calculate difference matrix
        diff_matrix = interpolated_matrix - original_matrix
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Original Solutions', 'Interpolated Solutions', 'Difference'),
            specs=[[{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}]],
            horizontal_spacing=0.15
        )
        
        # Original solutions sunburst
        fig.add_trace(
            go.Scatterpolar(
                r=times,
                theta=np.deg2rad(thetas),
                mode='markers',
                marker=dict(
                    size=8,
                    color=original_matrix.flatten(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(x=0.3, title='Stress (GPa)')
                ),
                name='Original'
            ),
            row=1, col=1
        )
        
        # Interpolated solutions sunburst
        fig.add_trace(
            go.Scatterpolar(
                r=times,
                theta=np.deg2rad(thetas),
                mode='markers',
                marker=dict(
                    size=8,
                    color=interpolated_matrix.flatten(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(x=0.65, title='Stress (GPa)')
                ),
                name='Interpolated'
            ),
            row=1, col=2
        )
        
        # Difference sunburst
        fig.add_trace(
            go.Scatterpolar(
                r=times,
                theta=np.deg2rad(thetas),
                mode='markers',
                marker=dict(
                    size=8,
                    color=diff_matrix.flatten(),
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(x=1.0, title='Δ Stress (GPa)'),
                    cmin=-np.max(np.abs(diff_matrix)),
                    cmax=np.max(np.abs(diff_matrix))
                ),
                name='Difference'
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f"Comparison: {region_name} - {stress_component}",
            showlegend=False,
            polar=dict(
                radialaxis=dict(title='Time (s)', showgrid=True),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            polar2=dict(
                radialaxis=dict(title='Time (s)', showgrid=True),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            polar3=dict(
                radialaxis=dict(title='Time (s)', showgrid=True),
                angularaxis=dict(rotation=90, direction='clockwise')
            ),
            height=600,
            width=1200
        )
        
        return fig
    
    def create_comparison_radar(self, original_stress, interpolated_stress, 
                               thetas, region_name, stress_component):
        """Create comparison radar plot"""
        
        # Ensure proper closure
        thetas_closed = np.append(thetas, 360)
        original_closed = np.append(original_stress, original_stress[0])
        interpolated_closed = np.append(interpolated_stress, interpolated_stress[0])
        
        fig = go.Figure()
        
        # Original solutions
        fig.add_trace(go.Scatterpolar(
            r=original_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line=dict(color='rgb(31, 119, 180)', width=3),
            name='Original',
            hovertemplate='Orientation: %{theta:.1f}°<br>Original Stress: %{r:.4f} GPa'
        ))
        
        # Interpolated solutions
        fig.add_trace(go.Scatterpolar(
            r=interpolated_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.3)',
            line=dict(color='rgb(255, 127, 14)', width=3),
            name='Interpolated',
            hovertemplate='Orientation: %{theta:.1f}°<br>Interpolated Stress: %{r:.4f} GPa'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Radar Comparison: {region_name} - {stress_component}",
                font=dict(size=16)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(original_stress), max(interpolated_stress)) * 1.2],
                    gridcolor="lightgray",
                    gridwidth=2
                ),
                angularaxis=dict(
                    gridcolor="lightgray",
                    gridwidth=2,
                    rotation=90,
                    direction="clockwise"
                )
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5
            ),
            width=800,
            height=600
        )
        
        return fig

# =============================================
# ENHANCED RESULTS MANAGER WITH REGION SUPPORT
# =============================================
class EnhancedResultsManager:
    """Manager for saving and exporting results with enhanced formatting and region support"""
    
    @staticmethod
    def prepare_region_analysis_data(analysis_results, region_type, stress_component, stress_type):
        """Prepare region analysis data for export"""
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'region_type': region_type,
                'stress_component': stress_component,
                'stress_type': stress_type,
                'analysis_type': 'region_stress_analysis',
                'description': f'Stress analysis for {region_type} region'
            },
            'analysis_results': analysis_results,
            'statistics': {
                'num_solutions': len(analysis_results),
                'stress_values': [r['region_stress'] for r in analysis_results],
                'mean_stress': np.mean([r['region_stress'] for r in analysis_results]),
                'max_stress': np.max([r['region_stress'] for r in analysis_results]),
                'min_stress': np.min([r['region_stress'] for r in analysis_results]),
                'std_stress': np.std([r['region_stress'] for r in analysis_results])
            }
        }
        
        return export_data
    
    @staticmethod
    def create_region_analysis_archive(stress_matrix, times, thetas, region_type, 
                                      stress_component, stress_type, metadata=None):
        """Create ZIP archive with region analysis results"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save stress matrix as NPY
            stress_buffer = BytesIO()
            np.save(stress_buffer, stress_matrix)
            zip_file.writestr(f'{region_type}_{stress_component}_stress_matrix.npy', stress_buffer.getvalue())
            
            # Save times and thetas
            times_buffer = BytesIO()
            np.save(times_buffer, times)
            zip_file.writestr('times.npy', times_buffer.getvalue())
            
            thetas_buffer = BytesIO()
            np.save(thetas_buffer, thetas)
            zip_file.writestr('thetas.npy', thetas_buffer.getvalue())
            
            # Create and save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'region_type': region_type,
                'stress_component': stress_component,
                'stress_type': stress_type,
                'generated_at': datetime.now().isoformat(),
                'matrix_shape': stress_matrix.shape,
                'time_range': f"{min(times):.2f} to {max(times):.2f} s",
                'theta_range': f"{min(thetas):.1f} to {max(thetas):.1f} °"
            })
            
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr('metadata.json', metadata_json)
            
            # Save CSV version
            csv_data = []
            for t_idx, time_val in enumerate(times):
                for theta_idx, theta_val in enumerate(thetas):
                    csv_data.append({
                        'time_s': f"{time_val:.3f}",
                        'orientation_deg': f"{theta_val:.1f}",
                        'stress_gpa': f"{stress_matrix[t_idx, theta_idx]:.6f}",
                        'region': region_type,
                        'stress_component': stress_component,
                        'analysis_type': stress_type
                    })
            
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            zip_file.writestr(f'{region_type}_{stress_component}_data.csv', csv_str)
            
            # Save statistics summary
            stats = {
                'max_stress': float(np.max(stress_matrix)),
                'min_stress': float(np.min(stress_matrix)),
                'mean_stress': float(np.mean(stress_matrix)),
                'std_stress': float(np.std(stress_matrix)),
                'percentile_95': float(np.percentile(stress_matrix, 95)),
                'percentile_99': float(np.percentile(stress_matrix, 99)),
                'region_type': region_type,
                'stress_component': stress_component
            }
            
            stats_json = json.dumps(stats, indent=2)
            zip_file.writestr('statistics_summary.json', stats_json)
            
            # Add README
            readme = f"""# REGION STRESS ANALYSIS RESULTS
Generated: {datetime.now().isoformat()}

## ANALYSIS DETAILS
- Region: {region_type}
- Stress Component: {stress_component}
- Analysis Type: {stress_type}
- Time Points: {len(times)} ({min(times):.2f} to {max(times):.2f} s)
- Orientations: {len(thetas)} ({min(thetas):.1f} to {max(thetas):.1f} °)
- Matrix Dimensions: {stress_matrix.shape[0]} × {stress_matrix.shape[1]}

## REGION DEFINITIONS
- Defect Region: η > 0.6 (High defect concentration)
- Interface Region: 0.4 ≤ η ≤ 0.6 (Transition region)
- Bulk Region: η < 0.4 (Pure Ag material)

## FILES
1. {region_type}_{stress_component}_stress_matrix.npy - 2D stress matrix
2. times.npy - Time points array
3. thetas.npy - Orientation angles array
4. metadata.json - Complete metadata
5. {region_type}_{stress_component}_data.csv - Tabular data
6. statistics_summary.json - Statistical summary

## SPATIAL LOCALITY REGULARIZATION
The interpolation uses Euclidean distance in parameter space:
- Distance: sqrt(Σ(v_source - v_target)²)
- Weights: exp(-0.5 * (distance/sigma)²)
- Normalized to sum to 1
"""
            zip_file.writestr('README_REGION_ANALYSIS.txt', readme)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# MAIN APPLICATION WITH REGION ANALYSIS
# =============================================
def main():
    st.set_page_config(
        page_title="Ag Material Stress Analysis with Region Comparison",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
    }
    .region-card {
        border: 2px solid;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .defect-region {
        border-color: #EF4444;
        background-color: #FEE2E2;
    }
    .interface-region {
        border-color: #F59E0B;
        background-color: #FEF3C7;
    }
    .bulk-region {
        border-color: #10B981;
        background-color: #D1FAE5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Ag Material Stress Analysis with Region Comparison</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>📊 Analysis Features:</strong> Three region types (Defect, Interface, Bulk) with sunburst/radar visualizations<br>
    <strong>📈 Data Sources:</strong> Original loaded files AND/OR interpolated solutions with comparison<br>
    <strong>🗺️ Spatial Locality:</strong> Euclidean distance regularization for accurate interpolation<br>
    <strong>🎯 Purpose:</strong> Validate interpolated solutions against original files
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedSpatialInterpolator(sigma=0.3, use_spatial_locality=True)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'comparison_visualizer' not in st.session_state:
        st.session_state.comparison_visualizer = EnhancedComparisonVisualizer()
    if 'original_analyzer' not in st.session_state:
        st.session_state.original_analyzer = OriginalFileAnalyzer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    
    # Sidebar with enhanced options
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Analysis Settings</h2>', unsafe_allow_html=True)
        
        # Data source selection
        st.markdown("#### 📊 Data Source")
        data_source = st.radio(
            "Select data source for visualization:",
            ["Original Loaded Files", "Interpolated Solutions", "Comparison (Both)"],
            index=0,
            help="Choose whether to visualize original files, interpolated solutions, or compare both"
        )
        
        # Region selection
        st.markdown("#### 🎯 Analysis Region")
        region_type_display = st.selectbox(
            "Select region for stress analysis:",
            ["Defect Region (η > 0.6)", "Interface Region (0.4 ≤ η ≤ 0.6)", "Bulk Ag Material (η < 0.4)"],
            index=2,
            help="Select the material region to analyze"
        )
        
        # Map region name to key
        region_map = {
            "Defect Region (η > 0.6)": "defect",
            "Interface Region (0.4 ≤ η ≤ 0.6)": "interface",
            "Bulk Ag Material (η < 0.4)": "bulk"
        }
        region_key = region_map[region_type_display]
        
        # Stress component
        st.markdown("#### 📈 Stress Component")
        stress_component = st.selectbox(
            "Select stress component:",
            ["von_mises", "sigma_hydro", "sigma_mag"],
            index=0,
            help="Select which stress component to visualize"
        )
        
        # Stress type
        stress_type = st.selectbox(
            "Select stress analysis type:",
            ["max_abs", "mean_abs", "max", "min", "mean"],
            index=0,
            help="Select how to analyze stress in the region"
        )
        
        # Spatial locality settings
        st.markdown("#### 🗺️ Spatial Locality Settings")
        use_spatial_locality = st.checkbox(
            "Use Spatial Locality Regularization", 
            value=True,
            help="Use Euclidean distance in parameter space for interpolation weights"
        )
        
        spatial_sigma = st.slider(
            "Spatial Locality Sigma",
            0.1, 2.0, 0.3, 0.1,
            help="Controls the influence of spatial distance (larger = smoother interpolation)"
        )
        
        spatial_weight = st.slider(
            "Spatial Weight",
            0.0, 2.0, 1.0, 0.1,
            help="Weight of spatial locality in interpolation"
        )
        
        # Update interpolator settings
        if (use_spatial_locality != st.session_state.interpolator.use_spatial_locality or
            spatial_sigma != st.session_state.interpolator.sigma or
            spatial_weight != st.session_state.interpolator.spatial_weight):
            st.session_state.interpolator.use_spatial_locality = use_spatial_locality
            st.session_state.interpolator.sigma = spatial_sigma
            st.session_state.interpolator.spatial_weight = spatial_weight
        
        # Visualization type
        st.markdown("#### 🎨 Visualization Type")
        viz_type = st.radio(
            "Select visualization type:",
            ["Sunburst", "Radar", "Both"],
            index=0,
            help="Choose visualization type"
        )
        
        # Colormap selection
        cmap = st.selectbox(
            "Color Map",
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('rainbow') if 'rainbow' in ALL_COLORMAPS else 0
        )
        
        # Interpolation parameters (only show for interpolated/comparison)
        if data_source in ["Interpolated Solutions", "Comparison (Both)"]:
            st.markdown("#### 🎯 Interpolation Parameters")
            
            defect_type = st.selectbox("Defect Type", ["ISF", "ESF", "Twin"], index=0)
            
            col_shape, col_eps = st.columns(2)
            with col_shape:
                shape = st.selectbox("Shape", 
                                    ["Square", "Horizontal Fault", "Vertical Fault", 
                                     "Rectangle", "Ellipse"], index=0)
            with col_eps:
                eps0 = st.slider("ε*", 0.3, 3.0, 0.707, 0.01,
                                help="Strain parameter")
            
            col_kappa, col_theta = st.columns(2)
            with col_kappa:
                kappa = st.slider("κ", 0.1, 2.0, 0.6, 0.01,
                                 help="Shape parameter")
            with col_theta:
                theta_min = st.slider("Min Angle (°)", 0, 360, 0, 5)
                theta_max = st.slider("Max Angle (°)", 0, 360, 360, 5)
                theta_step = st.slider("Step Size (°)", 5, 45, 15, 5)
            
            # Time settings
            st.markdown("#### ⏱️ Time Settings")
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                n_times = st.slider("Time Points", 10, 200, 50, 10)
            with col_time2:
                max_time = st.slider("Max Time (s)", 50, 500, 200, 10)
        
        # Load solutions
        st.markdown("#### 📂 Load Solutions")
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load All Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions(use_cache=True)
                    if st.session_state.solutions:
                        st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions")
        
        with col_load2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.session_state.loader.cache.clear()
                st.session_state.solutions = []
                st.success("Cache cleared!")
        
        # Show loaded solutions
        if st.session_state.solutions:
            with st.expander(f"📋 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    params = sol.get('params', {})
                    st.write(f"**{i+1}. {sol.get('filename', 'Unknown')}**")
                    st.caption(f"Type: {params.get('defect_type', '?')} | "
                              f"θ: {np.rad2deg(params.get('theta', 0)):.1f}° | "
                              f"ε*: {params.get('eps0', 0):.2f}")
                if len(st.session_state.solutions) > 5:
                    st.info(f"... and {len(st.session_state.solutions) - 5} more")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📊 Region Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Region information cards
        st.markdown(f'<div class="region-card {region_key}-region">', unsafe_allow_html=True)
        st.markdown(f"### {region_type_display}")
        
        if region_key == 'defect':
            st.write("**η > 0.6** - High defect concentration region")
            st.write("Analysis focuses on stress concentration in defect cores")
        elif region_key == 'interface':
            st.write("**0.4 ≤ η ≤ 0.6** - Interface region between defect and bulk")
            st.write("Analysis focuses on interfacial stress gradients")
        else:  # bulk
            st.write("**η < 0.4** - Pure Ag bulk material")
            st.write("Analysis focuses on stress propagation in bulk")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.solutions:
            st.warning("⚠️ Please load solutions first using the button in the sidebar.")
            
            # Show directory information
            with st.expander("📁 Directory Information", expanded=False):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.success(f"✅ Found {total_files} files in {SOLUTIONS_DIR}")
                else:
                    st.error(f"❌ No files found in {SOLUTIONS_DIR}")
        
        else:
            # Generate analysis button
            if st.button("🚀 Generate Region Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Generating {region_type_display} analysis..."):
                    try:
                        if data_source == "Original Loaded Files":
                            # Analyze original files
                            results = st.session_state.original_analyzer.analyze_all_solutions(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            # Create stress matrix for visualization
                            stress_matrix, times, thetas = st.session_state.original_analyzer.create_stress_matrix_from_original(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            if stress_matrix is not None:
                                st.session_state.original_matrix = stress_matrix
                                st.session_state.thetas = thetas
                                st.session_state.times = times
                                st.session_state.region_type = region_type_display
                                st.session_state.stress_component = stress_component
                                st.session_state.stress_type = stress_type
                                
                                st.success(f"✅ Generated analysis for {len(results)} original solutions")
                                
                                # Display summary statistics
                                with st.expander("📈 Original Solutions Summary", expanded=True):
                                    st.write(f"**Region:** {region_type_display}")
                                    st.write(f"**Stress Component:** {stress_component}")
                                    st.write(f"**Analysis Type:** {stress_type}")
                                    st.write(f"**Number of Solutions:** {len(results)}")
                                    if len(thetas) > 0:
                                        st.write(f"**Theta Range:** {np.min(thetas):.1f}° to {np.max(thetas):.1f}°")
                                    st.write(f"**Time Points:** {len(times)}")
                                    
                                    # Calculate statistics
                                    if stress_matrix.size > 0:
                                        max_val = np.max(stress_matrix)
                                        mean_val = np.mean(stress_matrix)
                                        min_val = np.min(stress_matrix)
                                        std_val = np.std(stress_matrix)
                                        
                                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                        with col_stat1:
                                            st.metric("Max Stress", f"{max_val:.4f} GPa")
                                        with col_stat2:
                                            st.metric("Mean Stress", f"{mean_val:.4f} GPa")
                                        with col_stat3:
                                            st.metric("Min Stress", f"{min_val:.4f} GPa")
                                        with col_stat4:
                                            st.metric("Std Dev", f"{std_val:.4f} GPa")
                            
                        elif data_source == "Interpolated Solutions":
                            # Generate interpolated solutions
                            st.info("Generating interpolated solutions...")
                            
                            # Generate theta range
                            thetas = np.arange(theta_min, theta_max + theta_step, theta_step)
                            theta_rad = np.deg2rad(thetas)
                            
                            # Generate time points
                            times = np.linspace(0, max_time, n_times)
                            
                            # Generate predictions for each orientation
                            predictions = []
                            spatial_weights_all = []
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, theta in enumerate(theta_rad):
                                status_text.text(f"🔧 Processing orientation {i+1}/{len(theta_rad)} ({thetas[i]:.0f}°)...")
                                
                                # Target parameters
                                target_params = {
                                    'defect_type': defect_type,
                                    'theta': float(theta),
                                    'eps0': eps0,
                                    'kappa': kappa,
                                    'shape': shape
                                }
                                
                                # Interpolate with spatial locality
                                result = st.session_state.interpolator.interpolate_with_spatial_locality(
                                    st.session_state.solutions, target_params, region_key,
                                    stress_component, stress_type
                                )
                                
                                if result:
                                    # Create time evolution based on interpolated stress
                                    region_stress = result['region_stress']
                                    time_evolution = []
                                    
                                    for t in times:
                                        # Time-dependent scaling
                                        stress_at_t = region_stress * (1 - np.exp(-t / 50))
                                        time_evolution.append(stress_at_t)
                                    
                                    predictions.append(time_evolution)
                                    spatial_weights_all.append(result['spatial_weights'])
                                
                                progress_bar.progress((i + 1) / len(theta_rad))
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Create stress matrix
                            if predictions:
                                stress_matrix = np.array(predictions).T
                                st.session_state.interpolated_matrix = stress_matrix
                                st.session_state.thetas = thetas
                                st.session_state.times = times
                                st.session_state.region_type = region_type_display
                                st.session_state.stress_component = stress_component
                                st.session_state.stress_type = stress_type
                                st.session_state.spatial_weights = spatial_weights_all
                                
                                st.success(f"✅ Generated interpolated solutions for {len(thetas)} orientations")
                                
                                # Display spatial locality information
                                with st.expander("🗺️ Spatial Locality Information", expanded=True):
                                    if use_spatial_locality:
                                        st.success("✅ Spatial locality regularization ENABLED")
                                        st.write(f"**Sigma:** {spatial_sigma}")
                                        st.write(f"**Spatial Weight:** {spatial_weight}")
                                        st.write("**Method:** Euclidean distance in parameter space")
                                        st.write("**Weight Calculation:** Gaussian kernel based on distance")
                                        
                                        # Show spatial weights for first orientation
                                        if spatial_weights_all and len(spatial_weights_all) > 0:
                                            with st.expander("View Spatial Weights for First Orientation", expanded=False):
                                                weights = spatial_weights_all[0]
                                                st.write(f"Weights for θ = {thetas[0]:.1f}°:")
                                                for i, w in enumerate(weights[:5]):
                                                    sol = st.session_state.solutions[i]
                                                    params = sol.get('params', {})
                                                    st.write(f"  - Solution {i+1} ({params.get('defect_type', '?')}, "
                                                            f"θ={np.rad2deg(params.get('theta', 0)):.1f}°): {w:.4f}")
                                                if len(weights) > 5:
                                                    st.write(f"  - ... and {len(weights)-5} more")
                                    else:
                                        st.warning("⚠️ Spatial locality regularization DISABLED")
                                        st.write("Using uniform weights for interpolation")
                            
                        elif data_source == "Comparison (Both)":
                            # Generate both original and interpolated
                            st.info("Generating comparison analysis...")
                            
                            # First, create original matrix
                            original_matrix, orig_times, orig_thetas = st.session_state.original_analyzer.create_stress_matrix_from_original(
                                st.session_state.solutions, region_key, stress_component, stress_type
                            )
                            
                            if original_matrix is not None:
                                # Generate interpolated matrix with same dimensions
                                thetas = orig_thetas
                                times = orig_times
                                
                                # Convert thetas to radians for interpolation
                                theta_rad = np.deg2rad(thetas)
                                
                                interpolated_matrix = np.zeros_like(original_matrix)
                                spatial_weights_all = []
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, theta in enumerate(theta_rad):
                                    status_text.text(f"🔧 Interpolating orientation {i+1}/{len(theta_rad)} ({thetas[i]:.1f}°)...")
                                    
                                    # Target parameters
                                    target_params = {
                                        'defect_type': defect_type,
                                        'theta': float(theta),
                                        'eps0': eps0,
                                        'kappa': kappa,
                                        'shape': shape
                                    }
                                    
                                    # Interpolate
                                    result = st.session_state.interpolator.interpolate_with_spatial_locality(
                                        st.session_state.solutions, target_params, region_key,
                                        stress_component, stress_type
                                    )
                                    
                                    if result:
                                        region_stress = result['region_stress']
                                        spatial_weights_all.append(result['spatial_weights'])
                                        for t_idx, t in enumerate(times):
                                            # Time-dependent scaling
                                            stress_at_t = region_stress * (1 - np.exp(-t / 50))
                                            interpolated_matrix[t_idx, i] = stress_at_t
                                    
                                    progress_bar.progress((i + 1) / len(theta_rad))
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Store both matrices
                                st.session_state.original_matrix = original_matrix
                                st.session_state.interpolated_matrix = interpolated_matrix
                                st.session_state.thetas = thetas
                                st.session_state.times = times
                                st.session_state.region_type = region_type_display
                                st.session_state.stress_component = stress_component
                                st.session_state.stress_type = stress_type
                                st.session_state.spatial_weights = spatial_weights_all
                                
                                st.success(f"✅ Generated comparison analysis")
                                
                                # Calculate comparison metrics
                                with st.expander("📊 Comparison Metrics", expanded=True):
                                    orig = original_matrix
                                    interp = interpolated_matrix
                                    
                                    mae = np.mean(np.abs(interp - orig))
                                    rmse = np.sqrt(np.mean((interp - orig) ** 2))
                                    r2 = 1 - np.sum((interp - orig) ** 2) / (np.sum((orig - np.mean(orig)) ** 2) + 1e-10)
                                    max_diff = np.max(np.abs(interp - orig))
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("MAE", f"{mae:.4f} GPa")
                                    with col2:
                                        st.metric("RMSE", f"{rmse:.4f} GPa")
                                    with col3:
                                        st.metric("R² Score", f"{r2:.4f}")
                                    with col4:
                                        st.metric("Max Difference", f"{max_diff:.4f} GPa")
                                    
                                    st.write(f"**Spatial Locality:** {'ENABLED' if use_spatial_locality else 'DISABLED'}")
                                    if use_spatial_locality:
                                        st.write(f"**Sigma:** {spatial_sigma}")
                                        st.write(f"**Spatial Weight:** {spatial_weight}")
                        
                        # Generate visualizations based on data source
                        if 'original_matrix' in st.session_state or 'interpolated_matrix' in st.session_state:
                            
                            # Get the active matrix based on data source
                            if data_source == "Original Loaded Files" and 'original_matrix' in st.session_state:
                                active_matrix = st.session_state.original_matrix
                                title_suffix = "Original"
                            elif data_source == "Interpolated Solutions" and 'interpolated_matrix' in st.session_state:
                                active_matrix = st.session_state.interpolated_matrix
                                title_suffix = "Interpolated"
                            elif data_source == "Comparison (Both)" and 'original_matrix' in st.session_state and 'interpolated_matrix' in st.session_state:
                                # For comparison, we'll handle separately
                                pass
                            
                            # Generate sunburst visualization
                            if viz_type in ["Sunburst", "Both"] and data_source != "Comparison (Both)":
                                st.markdown(f'<h3 class="sub-header">🌅 {region_type_display} - Sunburst Visualization</h3>', unsafe_allow_html=True)
                                
                                fig_sunburst = st.session_state.visualizer.create_enhanced_plotly_sunburst(
                                    active_matrix,
                                    st.session_state.times,
                                    st.session_state.thetas,
                                    title=f"{title_suffix}: {region_type_display} - {stress_component} ({stress_type})",
                                    cmap=cmap,
                                    colorbar_title=f"{stress_component} Stress (GPa)"
                                )
                                st.plotly_chart(fig_sunburst, use_container_width=True)
                            
                            # Generate radar visualization
                            if viz_type in ["Radar", "Both"] and data_source != "Comparison (Both)":
                                st.markdown(f'<h3 class="sub-header">📡 {region_type_display} - Radar Visualization</h3>', unsafe_allow_html=True)
                                
                                # Select time point for radar
                                if len(st.session_state.times) > 0:
                                    time_idx = st.slider(
                                        "Select Time Point for Radar Chart",
                                        0, len(st.session_state.times)-1, 
                                        min(len(st.session_state.times)//2, len(st.session_state.times)-1),
                                        key="radar_time"
                                    )
                                    selected_time = st.session_state.times[time_idx]
                                    
                                    stress_values = active_matrix[time_idx, :]
                                    
                                    fig_radar = st.session_state.visualizer.create_enhanced_plotly_radar(
                                        stress_values,
                                        st.session_state.thetas,
                                        f"{title_suffix}: {stress_component}",
                                        selected_time
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)
                            
                            # Generate comparison visualizations
                            if data_source == "Comparison (Both)":
                                if viz_type in ["Sunburst", "Both"]:
                                    st.markdown(f'<h3 class="sub-header">🌅 Comparison Sunburst</h3>', unsafe_allow_html=True)
                                    
                                    fig_comparison = st.session_state.comparison_visualizer.create_comparison_sunburst(
                                        st.session_state.original_matrix,
                                        st.session_state.interpolated_matrix,
                                        st.session_state.thetas,
                                        st.session_state.times,
                                        region_type_display,
                                        stress_component
                                    )
                                    st.plotly_chart(fig_comparison, use_container_width=True)
                                
                                if viz_type in ["Radar", "Both"]:
                                    st.markdown(f'<h3 class="sub-header">📡 Comparison Radar</h3>', unsafe_allow_html=True)
                                    
                                    if len(st.session_state.times) > 0:
                                        time_idx = st.slider(
                                            "Select Time Point for Radar Comparison",
                                            0, len(st.session_state.times)-1, 
                                            min(len(st.session_state.times)//2, len(st.session_state.times)-1),
                                            key="comparison_radar_time"
                                        )
                                        selected_time = st.session_state.times[time_idx]
                                        
                                        original_stress = st.session_state.original_matrix[time_idx, :]
                                        interpolated_stress = st.session_state.interpolated_matrix[time_idx, :]
                                        
                                        fig_radar_comparison = st.session_state.comparison_visualizer.create_comparison_radar(
                                            original_stress,
                                            interpolated_stress,
                                            st.session_state.thetas,
                                            region_type_display,
                                            stress_component
                                        )
                                        st.plotly_chart(fig_radar_comparison, use_container_width=True)
                                
                                # Show spatial weights analysis
                                with st.expander("🗺️ Spatial Weights Analysis", expanded=False):
                                    if 'spatial_weights' in st.session_state and st.session_state.spatial_weights:
                                        weights_matrix = np.array(st.session_state.spatial_weights)
                                        st.write(f"**Spatial Weights Matrix Shape:** {weights_matrix.shape}")
                                        st.write(f"**Mean Weight:** {np.mean(weights_matrix):.4f}")
                                        st.write(f"**Std of Weights:** {np.std(weights_matrix):.4f}")
                                        
                                        # Show weights heatmap
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        im = ax.imshow(weights_matrix.T, aspect='auto', cmap='viridis')
                                        ax.set_xlabel('Orientation Index')
                                        ax.set_ylabel('Source Solution Index')
                                        ax.set_title('Spatial Locality Weights Heatmap')
                                        plt.colorbar(im, ax=ax, label='Weight')
                                        st.pyplot(fig)
                        
                        # Export options
                        if data_source != "Comparison (Both)" and ('original_matrix' in st.session_state or 'interpolated_matrix' in st.session_state):
                            st.markdown('<h3 class="sub-header">📤 Export Results</h3>', unsafe_allow_html=True)
                            
                            col_exp1, col_exp2, col_exp3 = st.columns(3)
                            
                            with col_exp1:
                                if st.button("💾 Export as CSV", use_container_width=True):
                                    export_data = []
                                    for t_idx, time_val in enumerate(st.session_state.times):
                                        for theta_idx, theta_val in enumerate(st.session_state.thetas):
                                            export_data.append({
                                                'time_s': time_val,
                                                'orientation_deg': theta_val,
                                                'orientation_rad': np.deg2rad(theta_val),
                                                'stress_gpa': active_matrix[t_idx, theta_idx],
                                                'region': region_key,
                                                'stress_component': stress_component,
                                                'analysis_type': stress_type
                                            })
                                    
                                    df = pd.DataFrame(export_data)
                                    csv = df.to_csv(index=False)
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download CSV",
                                        data=csv,
                                        file_name=f"{region_key}_{stress_component}_{data_source.replace(' ', '_').lower()}_{timestamp}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            with col_exp2:
                                if st.button("📊 Export as JSON", use_container_width=True):
                                    metadata = {
                                        'data_source': data_source,
                                        'region_type': region_key,
                                        'region_display': region_type_display,
                                        'stress_component': stress_component,
                                        'stress_type': stress_type,
                                        'times': st.session_state.times.tolist(),
                                        'thetas': st.session_state.thetas.tolist(),
                                        'spatial_locality': {
                                            'enabled': use_spatial_locality,
                                            'sigma': spatial_sigma,
                                            'weight': spatial_weight
                                        } if data_source in ["Interpolated Solutions", "Comparison (Both)"] else None
                                    }
                                    
                                    export_dict = {
                                        'metadata': metadata,
                                        'stress_matrix': active_matrix.tolist(),
                                        'statistics': {
                                            'max': float(np.max(active_matrix)),
                                            'min': float(np.min(active_matrix)),
                                            'mean': float(np.mean(active_matrix)),
                                            'std': float(np.std(active_matrix))
                                        }
                                    }
                                    
                                    json_str = json.dumps(export_dict, indent=2)
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download JSON",
                                        data=json_str,
                                        file_name=f"{region_key}_{stress_component}_{data_source.replace(' ', '_').lower()}_{timestamp}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            
                            with col_exp3:
                                if st.button("📦 Export Complete Archive", use_container_width=True):
                                    metadata = {
                                        'data_source': data_source,
                                        'region_type': region_key,
                                        'region_display': region_type_display,
                                        'stress_component': stress_component,
                                        'stress_type': stress_type,
                                        'spatial_locality': {
                                            'enabled': use_spatial_locality,
                                            'sigma': spatial_sigma,
                                            'weight': spatial_weight
                                        } if data_source in ["Interpolated Solutions", "Comparison (Both)"] else None
                                    }
                                    
                                    zip_buffer = st.session_state.results_manager.create_region_analysis_archive(
                                        active_matrix, st.session_state.times, st.session_state.thetas,
                                        region_key, stress_component, stress_type, metadata
                                    )
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download ZIP Archive",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"{region_key}_{stress_component}_{data_source.replace(' ', '_').lower()}_{timestamp}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.markdown('<h2 class="sub-header">📈 Quick Dashboard</h2>', unsafe_allow_html=True)
        
        if 'original_matrix' in st.session_state or 'interpolated_matrix' in st.session_state:
            # Determine which matrix to show
            if data_source == "Comparison (Both)" and 'original_matrix' in st.session_state:
                matrix_to_show = st.session_state.original_matrix
                matrix_name = "Original"
            elif 'original_matrix' in st.session_state:
                matrix_to_show = st.session_state.original_matrix
                matrix_name = "Original"
            elif 'interpolated_matrix' in st.session_state:
                matrix_to_show = st.session_state.interpolated_matrix
                matrix_name = "Interpolated"
            else:
                matrix_to_show = None
                matrix_name = ""
            
            if matrix_to_show is not None:
                # Quick metrics
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Time Points", len(st.session_state.times))
                st.metric("Orientations", len(st.session_state.thetas))
                st.metric("Matrix", matrix_name)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Stress statistics
                max_val = np.max(matrix_to_show)
                mean_val = np.mean(matrix_to_show)
                min_val = np.min(matrix_to_show)
                std_val = np.std(matrix_to_show)
                
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("Max Stress", f"{max_val:.4f} GPa")
                    st.metric("Mean Stress", f"{mean_val:.4f} GPa")
                with col_metrics2:
                    st.metric("Min Stress", f"{min_val:.4f} GPa")
                    st.metric("Std Dev", f"{std_val:.4f} GPa")
                
                # Region-specific insights
                st.markdown(f'<h4 class="sub-header">🎯 {region_type_display} Insights</h4>', unsafe_allow_html=True)
                
                if region_key == 'defect':
                    st.write("**Defect Core Stress:**")
                    st.write("- High stress concentration")
                    st.write("- Direct defect impact")
                    st.write("- Critical for failure")
                elif region_key == 'interface':
                    st.write("**Interface Stress:**")
                    st.write("- Stress gradients")
                    st.write("- Transition effects")
                    st.write("- Defect propagation")
                else:  # bulk
                    st.write("**Bulk Material Stress:**")
                    st.write("- Stress propagation")
                    st.write("- Far-field effects")
                    st.write("- Material response")
                
                # Orientation analysis
                if len(st.session_state.thetas) > 0:
                    st.markdown('<h4 class="sub-header">🌐 Orientation Analysis</h4>', unsafe_allow_html=True)
                    
                    mean_by_theta = np.mean(matrix_to_show, axis=0)
                    max_theta_idx = np.argmax(mean_by_theta)
                    min_theta_idx = np.argmin(mean_by_theta)
                    
                    st.write(f"**Peak stress at:** {st.session_state.thetas[max_theta_idx]:.0f}°")
                    st.write(f"**Min stress at:** {st.session_state.thetas[min_theta_idx]:.0f}°")
                    
                    # Quick orientation plot
                    fig_theta, ax_theta = plt.subplots(figsize=(5, 3), dpi=120)
                    ax_theta.plot(st.session_state.thetas, mean_by_theta, 'b-', linewidth=2)
                    ax_theta.fill_between(st.session_state.thetas, mean_by_theta, alpha=0.3, color='blue')
                    ax_theta.set_xlabel('Orientation (°)', fontsize=9)
                    ax_theta.set_ylabel('Mean Stress (GPa)', fontsize=9)
                    ax_theta.set_title('Stress vs Orientation', fontsize=10, fontweight='bold')
                    ax_theta.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_theta)
                
                # Spatial locality info
                if data_source in ["Interpolated Solutions", "Comparison (Both)"] and use_spatial_locality:
                    st.markdown('<h4 class="sub-header">🗺️ Spatial Locality</h4>', unsafe_allow_html=True)
                    st.success("✅ Active")
                    st.write(f"Sigma: {spatial_sigma}")
                    st.write(f"Weight: {spatial_weight}")
                    st.write("Euclidean distance regularization")
        
        else:
            st.info("📊 No data generated yet.")
            st.write("Click 'Generate Region Analysis' to begin.")
            
            # Quick tips
            with st.expander("💡 Quick Tips", expanded=False):
                st.write("""
                **Three Region Types:**
                1. **Defect Region (η > 0.6):** Stress in defect cores
                2. **Interface Region (0.4 ≤ η ≤ 0.6):** Stress at interfaces
                3. **Bulk Region (η < 0.4):** Stress in pure Ag material
                
                **Spatial Locality Regularization:**
                - Uses Euclidean distance in parameter space
                - Weights = exp(-0.5 * (distance/sigma)²)
                - Prioritizes similar configurations
                - Improves interpolation accuracy
                
                **Comparison Purpose:**
                - Validate interpolated solutions
                - Benchmark against originals
                - Assess interpolation quality
                """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
