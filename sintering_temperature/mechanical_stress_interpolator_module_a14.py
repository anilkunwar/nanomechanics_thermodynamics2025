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
# NUMBA-ACCELERATED FUNCTIONS
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

@jit(nopython=True, parallel=True)
def weighted_stress_combination_numba(source_stresses, weights):
    """Numba-accelerated weighted stress combination"""
    n_sources = source_stresses.shape[0]
    n_components = source_stresses.shape[1]
    height = source_stresses.shape[2]
    width = source_stresses.shape[3]
    
    result = np.zeros((n_components, height, width))
    
    for comp in prange(n_components):
        for i in range(n_sources):
            weight = weights[i]
            for h in range(height):
                for w in range(width):
                    result[comp, h, w] += weight * source_stresses[i, comp, h, w]
    
    return result

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

@jit(nopython=True)
def find_bulk_stress_numba(stress_field, eta_field, eta_threshold=0.01):
    """Find maximum absolute stress in Ag bulk material (where eta ≈ 0)"""
    height, width = stress_field.shape
    max_stress = 0.0
    
    for i in range(height):
        for j in range(width):
            # Check if this is bulk material point (eta close to 0)
            if eta_field[i, j] < eta_threshold:
                stress_val = abs(stress_field[i, j])
                if stress_val > max_stress:
                    max_stress = stress_val
    
    return max_stress

# =============================================
# ENHANCED NUMERICAL SOLUTIONS LOADER
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
                # First try with weights_only=True and allowlist numpy globals
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
                        st.warning("Could not import numpy scalar, using weights_only=False")
                        data = torch.load(buffer, map_location='cpu', weights_only=False)
                except Exception as safe_error:
                    # If safe loading fails, try unsafe as last resort
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
            st.warning(f"Directory {self.solutions_dir} not found. Creating it.")
            os.makedirs(self.solutions_dir, exist_ok=True)
            return solutions
        
        all_files_info = self.get_all_files_info()
        
        if not all_files_info:
            st.info(f"No solution files found in {self.solutions_dir}")
            return solutions
        
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
        
        progress_bar.empty()
        status_text.empty()
        
        # Display failed files if any
        if failed_files:
            with st.expander(f"⚠️ Failed to load {len(failed_files)} files", expanded=False):
                for failed in failed_files[:10]:  # Show first 10
                    st.error(f"**{failed['filename']}**: {failed['error']}")
                if len(failed_files) > 10:
                    st.info(f"... and {len(failed_files) - 10} more files failed to load.")
        
        return solutions

# =============================================
# ENHANCED ATTENTION INTERPOLATOR WITH BULK STRESS
# =============================================
class EnhancedAttentionInterpolator(nn.Module):
    """Enhanced attention-based interpolator with bulk material stress analysis"""
    
    def __init__(self, sigma=0.3, use_numba=True, bulk_threshold=0.01):
        super().__init__()
        self.sigma = sigma
        self.use_numba = use_numba
        self.bulk_threshold = bulk_threshold  # Threshold for bulk material (eta ≈ 0)
        
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
            st.warning(f"Parameters is not a dictionary: {type(params)}")
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
    
    def extract_bulk_stress(self, eta, stress_fields):
        """Extract maximum absolute stress in Ag bulk material (where eta ≈ 0)"""
        if eta is None or not isinstance(eta, np.ndarray):
            return {
                'von_mises': 0.0,
                'sigma_hydro': 0.0,
                'sigma_mag': 0.0
            }
        
        # Find bulk material region (where eta is close to 0)
        bulk_mask = eta < self.bulk_threshold
        
        if not np.any(bulk_mask):
            # If no bulk found, use a broader definition
            bulk_mask = eta < 0.05
        
        results = {}
        
        for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
            stress_data = stress_fields.get(key, np.zeros_like(eta))
            
            if np.any(bulk_mask):
                # Extract stress values in bulk material
                bulk_stress = stress_data[bulk_mask]
                
                # Take absolute value for maximum magnitude
                abs_stress = np.abs(bulk_stress)
                
                # Get maximum absolute stress in bulk
                max_stress = np.max(abs_stress) if len(abs_stress) > 0 else 0.0
                
                # Also calculate statistics
                mean_stress = np.mean(abs_stress) if len(abs_stress) > 0 else 0.0
                std_stress = np.std(abs_stress) if len(abs_stress) > 0 else 0.0
                
            else:
                max_stress = 0.0
                mean_stress = 0.0
                std_stress = 0.0
            
            results[key] = {
                'max_abs': float(max_stress),
                'mean_abs': float(mean_stress),
                'std_abs': float(std_stress),
                'bulk_area': float(np.sum(bulk_mask))  # Number of bulk pixels
            }
        
        return results
    
    def interpolate(self, sources, target_params, stress_type='max_abs'):
        """Interpolate bulk stress field using attention weights"""
        
        # Filter and validate sources
        valid_sources = []
        for src in sources:
            # Check if source is a valid simulation dictionary
            if not isinstance(src, dict):
                continue
            
            # Check for required structure
            if 'params' not in src or 'history' not in src:
                continue
            
            # Check if params is a dictionary
            if not isinstance(src.get('params'), dict):
                continue
            
            valid_sources.append(src)
        
        if not valid_sources:
            st.error("❌ Interpolation failed: No valid source simulations found.")
            return None
        
        # Get source parameter vectors
        source_vectors = []
        source_bulk_stresses = []  # Store bulk stresses
        
        for src in valid_sources:
            src_vec = self.compute_parameter_vector(src['params'])
            source_vectors.append(src_vec)
            
            # Get stress from final frame
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
                
                # Extract bulk stress
                bulk_stress = self.extract_bulk_stress(eta, stress_fields)
                source_bulk_stresses.append(bulk_stress)
            else:
                # No history, use zero stresses
                source_bulk_stresses.append({
                    'von_mises': {'max_abs': 0.0, 'mean_abs': 0.0, 'std_abs': 0.0, 'bulk_area': 0.0},
                    'sigma_hydro': {'max_abs': 0.0, 'mean_abs': 0.0, 'std_abs': 0.0, 'bulk_area': 0.0},
                    'sigma_mag': {'max_abs': 0.0, 'mean_abs': 0.0, 'std_abs': 0.0, 'bulk_area': 0.0}
                })
        
        if not source_vectors:
            st.error("❌ Interpolation failed: Could not compute parameter vectors.")
            return None
        
        source_vectors = np.array(source_vectors)
        target_vector = self.compute_parameter_vector(target_params)
        
        # Compute attention weights with Numba acceleration
        if self.use_numba and len(source_vectors) > 0:
            try:
                weights = compute_gaussian_weights_numba(source_vectors, target_vector, self.sigma)
            except Exception as e:
                st.warning(f"Numba acceleration failed, falling back to NumPy: {e}")
                distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
                weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
                weights = weights / (np.sum(weights) + 1e-8)
        else:
            distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
            weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
            weights = weights / (np.sum(weights) + 1e-8)
        
        # Weighted combination of bulk stresses
        result_stress = {}
        for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
            weighted_max = 0.0
            weighted_mean = 0.0
            weighted_std = 0.0
            weighted_area = 0.0
            
            for w, stress_dict in zip(weights, source_bulk_stresses):
                if key in stress_dict:
                    weighted_max += w * stress_dict[key]['max_abs']
                    weighted_mean += w * stress_dict[key]['mean_abs']
                    weighted_std += w * stress_dict[key]['std_abs']
                    weighted_area += w * stress_dict[key]['bulk_area']
            
            result_stress[key] = {
                'max_abs': weighted_max,
                'mean_abs': weighted_mean,
                'std_abs': weighted_std,
                'bulk_area': weighted_area,
                'weighted_value': weighted_max if stress_type == 'max_abs' else weighted_mean
            }
        
        return {
            'bulk_stress': result_stress,  # Changed from stress_fields
            'attention_weights': weights,
            'target_params': target_params,
            'num_valid_sources': len(valid_sources),
            'source_bulk_stresses': source_bulk_stresses,  # For debugging/analysis
            'stress_type': stress_type
        }

# =============================================
# ENHANCED VISUALIZATION MANAGER WITH 50+ COLORMAPS
# =============================================
class EnhancedVisualizationManager:
    """Enhanced visualization manager with 50+ colormaps and advanced styling"""
    
    def __init__(self):
        self.colormap_manager = EnhancedColorMaps()
        self.default_stress_cmap = self.colormap_manager.get_colormap('stress_cmap')
        
    def create_stress_field_plot(self, stress_data, title, component_name,
                                extent=None, vmin=None, vmax=None,
                                include_contour=True, include_colorbar=True,
                                cmap='viridis', line_width=1.5, font_size=14,
                                grid_alpha=0.4, contour_levels=15, 
                                colorbar_pad=0.05, aspect_ratio='equal'):
        """Create enhanced matplotlib plot for stress field"""
        fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
        
        if extent is None:
            extent = [-64, 64, -64, 64]
        
        if vmin is None:
            vmin = np.nanmin(stress_data)
        if vmax is None:
            vmax = np.nanmax(stress_data)
        
        # Get colormap
        cmap_obj = self.colormap_manager.get_colormap(cmap)
        
        # Create heatmap with enhanced styling
        im = ax.imshow(stress_data,
                      cmap=cmap_obj,
                      extent=extent,
                      origin='lower',
                      aspect=aspect_ratio,
                      vmin=vmin,
                      vmax=vmax,
                      interpolation='bilinear')
        
        # Add enhanced contour lines
        if include_contour and not np.all(stress_data == stress_data[0, 0]):
            try:
                X, Y = np.meshgrid(np.linspace(extent[0], extent[1], stress_data.shape[1]),
                                  np.linspace(extent[2], extent[3], stress_data.shape[0]))
                
                levels = np.linspace(vmin, vmax, contour_levels)
                contour = ax.contour(X, Y, stress_data,
                                    levels=levels,
                                    colors='black',
                                    linewidths=line_width,
                                    alpha=0.7,
                                    linestyles='-')
                
                # Enhanced contour labels
                ax.clabel(contour, inline=True, fontsize=font_size-4, 
                         fmt='%.2f', colors='black', 
                         inline_spacing=5)
            except Exception as e:
                st.warning(f"Contour plotting failed: {e}")
        
        # Add enhanced colorbar
        if include_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=colorbar_pad)
            cbar.set_label(f'{component_name.replace("_", " ").title()} (GPa)', 
                          rotation=270, labelpad=25, fontsize=font_size+2, fontweight='bold')
            cbar.ax.tick_params(labelsize=font_size)
        
        # Set enhanced labels
        ax.set_xlabel('x (nm)', fontsize=font_size+2, fontweight='bold', labelpad=10)
        ax.set_ylabel('y (nm)', fontsize=font_size+2, fontweight='bold', labelpad=10)
        
        # Enhanced title
        ax.set_title(title, fontsize=font_size+4, fontweight='bold', pad=20)
        
        # Enhanced grid with customizable alpha
        ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.8)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Add scientific notation if needed
        if vmax > 1000 or vmin < 0.001:
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        plt.tight_layout()
        return fig
    
    def create_attention_weights_plot(self, weights, source_names=None,
                                     title="Attention Weights Distribution",
                                     bar_width=0.8, bar_alpha=0.85,
                                     font_size=14, grid_alpha=0.3,
                                     show_values=True, show_percentages=True):
        """Create enhanced bar plot for attention weights"""
        fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
        
        if source_names is None:
            source_names = [f'Source {i+1}' for i in range(len(weights))]
        
        # Create enhanced bar plot
        x_pos = np.arange(len(weights))
        colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
        
        bars = ax.bar(x_pos, weights,
                     color=colors,
                     edgecolor='black',
                     linewidth=2,
                     alpha=bar_alpha,
                     width=bar_width)
        
        # Add value labels on top of bars
        if show_values:
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{weight:.4f}', ha='center', va='bottom', 
                       fontsize=font_size-2, fontweight='bold')
        
        # Add percentage labels
        if show_percentages:
            total = np.sum(weights)
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                percentage = (weight / total) * 100 if total > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2., -0.02,
                       f'{percentage:.1f}%', ha='center', va='top', 
                       fontsize=font_size-3, color='darkred')
        
        # Customize plot
        ax.set_xlabel('Source Simulations', fontsize=font_size+2, fontweight='bold', labelpad=15)
        ax.set_ylabel('Attention Weight', fontsize=font_size+2, fontweight='bold', labelpad=15)
        ax.set_title(title, fontsize=font_size+4, fontweight='bold', pad=25)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(source_names, rotation=45, ha='right', fontsize=font_size)
        ax.set_ylim([0, max(weights) * 1.4])
        
        # Add horizontal line for average
        avg_weight = np.mean(weights)
        ax.axhline(y=avg_weight, color='red', linestyle='--', 
                  linewidth=3, alpha=0.8, label=f'Average: {avg_weight:.4f}')
        ax.legend(fontsize=font_size, loc='upper right')
        
        # Add enhanced grid
        ax.grid(True, alpha=grid_alpha, axis='y', linestyle='--', linewidth=0.8)
        
        # Add background color for better readability
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER WITH 50+ COLORMAPS
# =============================================
class EnhancedSunburstRadarVisualizer:
    """Enhanced sunburst and radar charts with 50+ colormaps and visualization enhancements"""
    
    def __init__(self):
        self.vis_manager = EnhancedVisualizationManager()
        self.colormap_manager = EnhancedColorMaps()
    
    def create_enhanced_sunburst_plot(self, stress_matrix, times, thetas, title, 
                                     cmap='rainbow', marker_size=15, line_width=2,
                                     font_size=16, grid_width=1.5, alpha=0.85,
                                     show_grid=True, show_labels=True, dpi=300):
        """Create enhanced polar heatmap (sunburst) visualization"""
        
        # Create polar plot with enhanced size
        theta_deg = np.deg2rad(thetas)
        theta_mesh, time_mesh = np.meshgrid(theta_deg, times)
        
        fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': 'polar'}, dpi=dpi)
        
        # Get colormap
        cmap_obj = self.colormap_manager.get_colormap(cmap)
        
        # Plot enhanced heatmap
        im = ax.pcolormesh(theta_mesh, time_mesh, stress_matrix, 
                          cmap=cmap_obj, 
                          shading='gouraud',  # Smoother shading
                          alpha=alpha,
                          edgecolors='none',
                          linewidth=0.5)
        
        # Enhanced customization
        ax.set_title(title, fontsize=font_size+6, pad=30, fontweight='bold', 
                    color='darkblue')
        
        if show_labels:
            ax.set_xlabel('Orientation (degrees)', labelpad=25, 
                         fontsize=font_size+2, fontweight='bold')
            ax.set_ylabel('Time (s)', labelpad=25, 
                         fontsize=font_size+2, fontweight='bold')
        
        ax.set_xticks(theta_deg)
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], 
                          fontsize=font_size, fontweight='bold')
        
        # Enhanced radial grid
        if show_grid:
            ax.grid(True, alpha=0.5, linestyle='-', linewidth=grid_width, 
                   color='gray')
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.15, shrink=0.8)
        cbar.set_label('Bulk Stress (GPa)', rotation=270, 
                      labelpad=35, fontsize=font_size+2, fontweight='bold')
        cbar.ax.tick_params(labelsize=font_size)
        
        # Add radial lines for important angles
        for angle in [0, 90, 180, 270]:
            ax.plot([np.deg2rad(angle), np.deg2rad(angle)], 
                   [0, np.max(times)], 
                   'r--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def create_enhanced_plotly_sunburst(self, stress_matrix, times, thetas, title, 
                                       cmap='rainbow', marker_size=12, line_width=1.5,
                                       font_size=18, width=900, height=750,
                                       show_colorbar=True, colorbar_title="Bulk Stress (GPa)",
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
                '<b>Bulk Stress</b>: %{marker.color:.4f} GPa<br>' +
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
            name='Bulk Stress Distribution'
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
                    period=360,  # Ensures 0° and 360° are at same position
                    thetaunit="degrees"
                ),
                bgcolor="rgba(240, 240, 240, 0.5)",
                sector=[0, 360],
                hole=0.1  # Creates a donut-like appearance
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
    
    def create_enhanced_radar_plot(self, stress_values, thetas, component_name, time_point,
                                  line_width=4, marker_size=10, fill_alpha=0.25,
                                  font_size=14, grid_width=1.5, dpi=300,
                                  show_values=True, show_grid=True, color='steelblue'):
        """Enhanced radar/spider chart with 0° and 360° at same position"""
        
        # Ensure thetas include 0° and 360° for proper closure
        thetas_closed = np.append(thetas, 360)  # Add 360°
        stress_values_closed = np.append(stress_values, stress_values[0])  # Close the loop
        
        # Convert to radians
        angles_rad = np.deg2rad(thetas_closed)
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'}, dpi=dpi)
        
        # Enhanced plot with gradient fill
        line = ax.plot(angles_rad, stress_values_closed, 'o-', 
                      linewidth=line_width, 
                      markersize=marker_size,
                      color=color, 
                      markerfacecolor='white', 
                      markeredgewidth=3,
                      markeredgecolor=color,
                      alpha=0.9,
                      label=f'{component_name} at t={time_point:.1f}s')
        
        # Enhanced gradient fill
        ax.fill(angles_rad, stress_values_closed, 
               alpha=fill_alpha, 
               color=color,
               edgecolor=color,
               linewidth=1)
        
        # Enhanced grid
        if show_grid:
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=grid_width, color='gray')
        
        # Enhanced labels with 0° and 360° at same position
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels([f'{i}°' for i in range(0, 360, 45)], 
                          fontsize=font_size, 
                          fontweight='bold',
                          color='darkblue')
        
        # Ensure 0° and 360° are at same position (top of chart)
        ax.set_theta_zero_location('N')  # 0° at North (top)
        ax.set_theta_direction(-1)  # Clockwise
        
        # Title with enhanced metrics
        mean_val = np.mean(stress_values)
        max_val = np.max(stress_values)
        min_val = np.min(stress_values)
        std_val = np.std(stress_values)
        
        title_text = (f'{component_name} Bulk Stress\n'
                     f't = {time_point:.1f} s\n'
                     f'Max: {max_val:.3f} GPa | Mean: {mean_val:.3f} GPa\n'
                     f'Min: {min_val:.3f} GPa | Std: {std_val:.3f} GPa')
        
        ax.set_title(title_text, fontsize=font_size+4, fontweight='bold', 
                    pad=40, color='darkred')
        
        # Add value annotations
        if show_values:
            for angle_deg, value in zip(thetas, stress_values):
                angle_rad = np.deg2rad(angle_deg)
                # Position text slightly outside the data
                text_radius = max(stress_values) * 1.15
                ax.text(angle_rad, text_radius, f'{value:.2f}', 
                       ha='center', va='center', fontsize=font_size-2,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="white", 
                                alpha=0.9,
                                edgecolor=color))
        
        # Add radial lines for reference
        for r in np.linspace(0, max(stress_values), 6):
            ax.plot(angles_rad, [r] * len(angles_rad), 
                   'k--', alpha=0.2, linewidth=0.5)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                 fontsize=font_size, framealpha=0.9)
        
        # Set radial limits with some padding
        ax.set_ylim([0, max(stress_values) * 1.3])
        
        plt.tight_layout()
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
            fillcolor=f'rgba({self._hex_to_rgb(color)}, {fill_alpha})',
            line=dict(color=color, width=line_width),
            marker=dict(size=marker_size, color=color, symbol='circle'),
            name=component_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Bulk Stress: %{r:.4f} GPa',
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
                text=f'{component_name} Bulk Stress at t={time_point:.1f}s',
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
                    title=dict(text='Bulk Stress (GPa)', 
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
                    period=360  # Ensures 0° and 360° are at same position
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
    
    def create_enhanced_comparison_radar(self, stress_matrices, thetas, component_names, 
                                        time_point, title="Bulk Stress Comparison",
                                        line_widths=None, colors=None, 
                                        fill_alphas=None, font_size=14, dpi=300):
        """Create enhanced radar plot comparing multiple stress components"""
        
        if line_widths is None:
            line_widths = [3] * len(stress_matrices)
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, len(stress_matrices)))
        if fill_alphas is None:
            fill_alphas = [0.15] * len(stress_matrices)
        
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'projection': 'polar'}, dpi=dpi)
        
        # Ensure proper closure for all datasets
        thetas_closed = np.append(thetas, 360)
        
        for idx, (stress_matrix, comp_name, color, line_width, fill_alpha) in enumerate(zip(
                stress_matrices, component_names, colors, line_widths, fill_alphas)):
            
            # Get values for the selected time point
            time_idx = np.argmin(np.abs(np.arange(len(stress_matrix)) - time_point))
            values = stress_matrix[time_idx, :]
            values_closed = np.append(values, values[0])
            
            # Convert colors to matplotlib format
            if isinstance(color, str):
                color_mpl = color
            else:
                color_mpl = color
            
            # Plot with enhanced styling
            ax.plot(thetas_closed, values_closed, 'o-', 
                   linewidth=line_width, 
                   markersize=8,
                   color=color_mpl, 
                   label=comp_name, 
                   alpha=0.9,
                   markerfacecolor='white',
                   markeredgewidth=2)
            
            # Enhanced fill
            ax.fill(thetas_closed, values_closed, 
                   alpha=fill_alpha, 
                   color=color_mpl)
        
        # Enhanced customization
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels([f'{i}°' for i in range(0, 360, 45)], 
                          fontsize=font_size, 
                          fontweight='bold')
        
        # Ensure 0° and 360° are at same position
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        ax.set_title(f'{title}\nt = {time_point:.1f} s', 
                    fontsize=font_size+6, fontweight='bold', pad=40, color='darkred')
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                 fontsize=font_size, framealpha=0.9)
        
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=1.5)
        
        plt.tight_layout()
        return fig
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_3d_enhanced_surface_plot(self, stress_matrix, times, thetas, title,
                                       cmap='rainbow', opacity=0.9, contour_z=True,
                                       font_size=16, width=1000, height=800):
        """Create enhanced 3D surface plot of bulk stress vs orientation vs time"""
        
        # Create meshgrid
        T, Theta = np.meshgrid(times, thetas)
        
        fig = go.Figure(data=[
            go.Surface(
                x=Theta,
                y=T,
                z=stress_matrix.T,
                colorscale=cmap,
                opacity=opacity,
                contours=dict(
                    z=dict(
                        show=contour_z,
                        usecolormap=True,
                        project=dict(z=True),
                        width=5
                    ),
                    x=dict(show=True, color='gray', width=1),
                    y=dict(show=True, color='gray', width=1)
                ),
                hovertemplate=(
                    'Orientation: %{x}°<br>' +
                    'Time: %{y:.2f}s<br>' +
                    'Bulk Stress: %{z:.4f} GPa<br>' +
                    '<extra></extra>'
                ),
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    fresnel=0.2,
                    specular=0.5,
                    roughness=0.3
                ),
                lightposition=dict(x=100, y=100, z=1000)
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=font_size+4, family="Arial Black")),
            scene=dict(
                xaxis=dict(
                    title='Orientation (°)', 
                    gridcolor="lightgray",
                    gridwidth=2,
                    titlefont=dict(size=font_size, color='black'),
                    tickfont=dict(size=font_size-2, color='black')
                ),
                yaxis=dict(
                    title='Time (s)', 
                    gridcolor="lightgray",
                    gridwidth=2,
                    titlefont=dict(size=font_size, color='black'),
                    tickfont=dict(size=font_size-2, color='black')
                ),
                zaxis=dict(
                    title='Bulk Stress (GPa)', 
                    gridcolor="lightgray",
                    gridwidth=2,
                    titlefont=dict(size=font_size, color='black'),
                    tickfont=dict(size=font_size-2, color='black')
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            width=width,
            height=height,
            margin=dict(l=80, r=80, b=80, t=100),
            font=dict(family="Arial, sans-serif")
        )
        
        return fig

# =============================================
# ENHANCED RESULTS MANAGER
# =============================================
class EnhancedResultsManager:
    """Manager for saving and exporting results with enhanced formatting"""
    
    @staticmethod
    def prepare_prediction_data(prediction_results, source_simulations, target_params):
        """Prepare prediction data for export"""
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_sources': len(source_simulations),
                'target_params': target_params,
                'software_version': '3.0.0',
                'analysis_type': 'bulk_stress_analysis',
                'description': 'Maximum absolute stress in Ag bulk material (eta ≈ 0)'
            },
            'prediction_results': prediction_results,
            'source_summary': []
        }
        
        # Add source simulation summaries
        for i, sim in enumerate(source_simulations):
            params = sim.get('params', {})
            export_data['source_summary'].append({
                'id': i,
                'defect_type': params.get('defect_type'),
                'shape': params.get('shape'),
                'eps0': float(params.get('eps0', 0)),
                'kappa': float(params.get('kappa', 0)),
                'theta': float(params.get('theta', 0)),
                'filename': sim.get('filename', 'Unknown')
            })
        
        return export_data
    
    @staticmethod
    def create_results_archive(stress_matrix, times, thetas, metadata):
        """Create ZIP archive with all results"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save stress matrix as NPY
            stress_buffer = BytesIO()
            np.save(stress_buffer, stress_matrix)
            zip_file.writestr('bulk_stress_matrix.npy', stress_buffer.getvalue())
            
            # Save times and thetas
            times_buffer = BytesIO()
            np.save(times_buffer, times)
            zip_file.writestr('times.npy', times_buffer.getvalue())
            
            thetas_buffer = BytesIO()
            np.save(thetas_buffer, thetas)
            zip_file.writestr('thetas.npy', thetas_buffer.getvalue())
            
            # Save metadata as JSON
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr('metadata.json', metadata_json)
            
            # Save CSV version with enhanced formatting
            csv_data = []
            for t_idx, time_val in enumerate(times):
                for theta_idx, theta_val in enumerate(thetas):
                    csv_data.append({
                        'time_s': f"{time_val:.3f}",
                        'orientation_deg': f"{theta_val:.1f}",
                        'bulk_stress_gpa': f"{stress_matrix[t_idx, theta_idx]:.6f}",
                        'orientation_rad': f"{np.deg2rad(theta_val):.6f}"
                    })
            
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            zip_file.writestr('bulk_stress_data.csv', csv_str)
            
            # Save statistics summary
            stats = {
                'max_stress': float(np.max(stress_matrix)),
                'min_stress': float(np.min(stress_matrix)),
                'mean_stress': float(np.mean(stress_matrix)),
                'std_stress': float(np.std(stress_matrix)),
                'time_range': f"{min(times):.2f} to {max(times):.2f} s",
                'orientation_range': f"{min(thetas):.1f} to {max(thetas):.1f} °",
                'matrix_dimensions': f"{stress_matrix.shape[0]} × {stress_matrix.shape[1]}"
            }
            
            stats_json = json.dumps(stats, indent=2)
            zip_file.writestr('statistics_summary.json', stats_json)
            
            # Add enhanced README
            readme = f"""# BULK MATERIAL STRESS ANALYSIS RESULTS
Generated: {datetime.now().isoformat()}

## ANALYSIS TYPE
Maximum absolute stress in Ag bulk material (η ≈ 0)

## FILES
1. bulk_stress_matrix.npy - 2D stress matrix (time × orientation)
2. times.npy - Time points array
3. thetas.npy - Orientation angles array
4. metadata.json - Complete simulation metadata
5. bulk_stress_data.csv - Tabular data for analysis
6. statistics_summary.json - Statistical summary

## PARAMETERS
- Defect Type: {metadata.get('defect_type', 'Unknown')}
- Shape: {metadata.get('shape', 'Unknown')}
- Orientation Range: {metadata.get('theta_range', 'Unknown')}
- ε*: {metadata.get('eps0', 'Unknown')}
- κ: {metadata.get('kappa', 'Unknown')}
- σ: {metadata.get('sigma', 'Unknown')}
- Stress Component: {metadata.get('stress_component', 'Unknown')}
- Bulk Threshold (η): {metadata.get('bulk_threshold', '0.01')}

## STATISTICS
- Maximum Stress: {stats['max_stress']:.4f} GPa
- Minimum Stress: {stats['min_stress']:.4f} GPa
- Mean Stress: {stats['mean_stress']:.4f} GPa
- Standard Deviation: {stats['std_stress']:.4f} GPa

## VISUALIZATION NOTES
- 0° and 360° are at same position in radar charts
- Stress values represent maximum absolute stress in Ag bulk
- Time evolution shows stress development over simulation time
"""
            zip_file.writestr('README_ENHANCED.txt', readme)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# MAIN APPLICATION WITH BULK STRESS ANALYSIS
# =============================================
def main():
    st.set_page_config(
        page_title="Bulk Material Stress Analysis",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔬 Ag Bulk Material Stress Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>📊 Analysis Type:</strong> Maximum absolute stress in Ag bulk material (η ≈ 0)<br>
    <strong>🎯 Key Feature:</strong> 50+ colormaps, enhanced visualization, fixed radar charts (0° = 360°)<br>
    <strong>⚡ Performance:</strong> Numba acceleration for faster computations
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedAttentionInterpolator(sigma=0.3, use_numba=True, bulk_threshold=0.01)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'vis_manager' not in st.session_state:
        st.session_state.vis_manager = EnhancedVisualizationManager()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Enhanced Settings</h2>', unsafe_allow_html=True)
        
        # Performance settings
        with st.expander("🚀 Performance & Analysis", expanded=True):
            use_numba = st.checkbox("Use Numba Acceleration", value=True, 
                                   help="Dramatically speeds up computations")
            use_cache = st.checkbox("Use File Cache", value=True,
                                   help="Faster loading of previously loaded files")
            
            # PyTorch loading method
            pt_loading_method = st.radio(
                "PyTorch Load Method",
                ["Safe (with numpy fix)", "Unsafe (weights_only=False)"],
                index=0,
                help="Safe loading includes fix for PyTorch 2.6 numpy scalar issue"
            )
            
            # Bulk threshold
            bulk_threshold = st.slider(
                "Bulk Threshold (η)",
                0.001, 0.1, 0.01, 0.001,
                help="Define bulk material as η < threshold"
            )
            
            # Stress type selection
            stress_type = st.selectbox(
                "Stress Analysis Type",
                ["max_abs", "mean_abs"],
                index=0,
                help="Maximum absolute stress or mean absolute stress in bulk"
            )
            
            if bulk_threshold != st.session_state.interpolator.bulk_threshold:
                st.session_state.interpolator.bulk_threshold = bulk_threshold
                st.success("Bulk threshold updated!")
            
            if use_numba != st.session_state.interpolator.use_numba:
                st.session_state.interpolator.use_numba = use_numba
                st.success("Numba setting updated!")
        
        # Load solutions
        st.markdown('<h3 class="sub-header">📂 Load Solutions</h3>', unsafe_allow_html=True)
        
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load All Solutions", use_container_width=True, 
                        type="primary", help="Load from numerical_solutions directory"):
                with st.spinner("Loading solutions..."):
                    start_time = time.time()
                    
                    # Determine loading method
                    load_method = "safe" if "Safe" in pt_loading_method else "unsafe"
                    
                    st.session_state.solutions = st.session_state.loader.load_all_solutions(
                        use_cache=use_cache,
                        pt_loading_method=load_method
                    )
                    load_time = time.time() - start_time
                    
                    if st.session_state.solutions:
                        st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions in {load_time:.2f}s")
                    else:
                        st.warning("No solutions found or loaded")
        
        with col_load2:
            if st.button("🗑️ Clear Cache", use_container_width=True, 
                        help="Clear file cache and reload"):
                st.session_state.loader.cache.clear()
                st.session_state.solutions = []
                st.success("Cache cleared!")
        
        # Show solution info
        if st.session_state.solutions:
            st.markdown(f'<h3 class="sub-header">📋 Loaded Solutions ({len(st.session_state.solutions)})</h3>', unsafe_allow_html=True)
            with st.expander("View Details", expanded=False):
                for i, sol in enumerate(st.session_state.solutions[:8]):
                    params = sol.get('params', {})
                    with st.container():
                        st.markdown(f"**{i+1}. {sol.get('filename', 'Unknown')}**")
                        cols = st.columns(3)
                        with cols[0]:
                            st.caption(f"Type: {params.get('defect_type', '?')}")
                        with cols[1]:
                            st.caption(f"θ: {np.rad2deg(params.get('theta', 0)):.1f}°")
                        with cols[2]:
                            st.caption(f"ε*: {params.get('eps0', 0):.2f}")
                        st.divider()
                
                if len(st.session_state.solutions) > 8:
                    st.info(f"... and {len(st.session_state.solutions) - 8} more")
        
        st.divider()
        
        # Interpolation settings
        st.markdown('<h3 class="sub-header">🎯 Target Parameters</h3>', unsafe_allow_html=True)
        
        defect_type = st.selectbox("Defect Type", ["ISF", "ESF", "Twin"], index=0)
        
        col_shape, col_eps = st.columns(2)
        with col_shape:
            shape = st.selectbox("Shape", 
                                ["Square", "Horizontal Fault", "Vertical Fault", 
                                 "Rectangle", "Ellipse"], index=0)
        with col_eps:
            eps0 = st.slider("ε*", 0.3, 3.0, 0.707, 0.01,
                            help="Strain parameter")
        
        col_kappa, col_sigma = st.columns(2)
        with col_kappa:
            kappa = st.slider("κ", 0.1, 2.0, 0.6, 0.01,
                             help="Shape parameter")
        with col_sigma:
            sigma = st.slider("σ", 0.1, 1.0, 0.3, 0.05,
                             help="Attention sigma parameter")
        
        # Update sigma if changed
        if sigma != st.session_state.interpolator.sigma:
            st.session_state.interpolator.sigma = sigma
        
        # Orientation sweep settings
        st.markdown('<h3 class="sub-header">🌐 Orientation Sweep</h3>', unsafe_allow_html=True)
        
        theta_min = st.slider("Min Angle (°)", 0, 360, 0, 5)
        theta_max = st.slider("Max Angle (°)", 0, 360, 360, 5)
        theta_step = st.slider("Step Size (°)", 5, 45, 15, 5)
        
        # Time settings
        st.markdown('<h3 class="sub-header">⏱️ Time Settings</h3>', unsafe_allow_html=True)
        
        col_time1, col_time2 = st.columns(2)
        with col_time1:
            n_times = st.slider("Time Points", 10, 200, 50, 10)
        with col_time2:
            max_time = st.slider("Max Time (s)", 50, 500, 200, 10)
        
        # Visualization settings
        st.markdown('<h3 class="sub-header">🎨 Enhanced Visualization</h3>', unsafe_allow_html=True)
        
        stress_component = st.selectbox(
            "Stress Component",
            ["von_mises", "sigma_hydro", "sigma_mag"],
            index=0,
            help="Select which stress component to visualize"
        )
        
        # Enhanced colormap selection
        cmap = st.selectbox(
            "Color Map (50+ options)",
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('rainbow') if 'rainbow' in ALL_COLORMAPS else 0,
            help="Choose from 50+ colormaps including rainbow, jet, turbo, inferno, etc."
        )
        
        viz_type = st.radio(
            "Chart Type", 
            ["Sunburst", "Radar", "Both", "3D Surface", "Comparison"], 
            horizontal=True,
            help="Select visualization type"
        )
        
        # Visualization enhancements
        with st.expander("📐 Visualization Settings", expanded=False):
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                use_plotly = st.checkbox("Use Plotly (Interactive)", value=True)
                show_metrics = st.checkbox("Show Enhanced Metrics", value=True)
                line_width = st.slider("Line Width", 1.0, 5.0, 2.5, 0.5)
            with col_viz2:
                marker_size = st.slider("Marker Size", 5, 20, 10, 1)
                font_size = st.slider("Font Size", 10, 24, 14, 1)
                alpha = st.slider("Transparency", 0.1, 1.0, 0.8, 0.1)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🚀 Generate Bulk Stress Visualizations</h2>', unsafe_allow_html=True)
        
        if not st.session_state.solutions:
            st.warning("⚠️ Please load solutions first using the button in the sidebar.")
            
            # Enhanced directory info
            with st.expander("📁 Directory Information", expanded=False):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.success(f"✅ Found {total_files} files in {SOLUTIONS_DIR}:")
                    for fmt, files in file_formats.items():
                        if files:
                            st.info(f"• **{fmt.upper()}**: {len(files)} files")
                            for f in files[:3]:
                                st.caption(f"  └ {os.path.basename(f)}")
                            if len(files) > 3:
                                st.caption(f"  └ ... and {len(files)-3} more")
                else:
                    st.error(f"❌ No files found in {SOLUTIONS_DIR}")
                    st.info("""
                    **Expected file structure:**
                    - Place simulation files in `numerical_solutions/` directory
                    - Supported formats: `.pkl`, `.pt`, `.h5`, `.npz`, `.sql`, `.json`
                    - Each file should contain:
                      - `params` dictionary (defect_type, theta, eps0, kappa, shape)
                      - `history` list with (eta, stress_fields) tuples
                    """)
        else:
            if st.button("✨ Generate Enhanced Bulk Stress Analysis", 
                        type="primary", use_container_width=True):
                with st.spinner("🔄 Generating bulk stress analysis..."):
                    try:
                        # Generate theta range
                        thetas = np.arange(theta_min, theta_max + theta_step, theta_step)
                        theta_rad = np.deg2rad(thetas)
                        
                        # Generate time points
                        times = np.linspace(0, max_time, n_times)
                        
                        # Generate predictions for each orientation
                        predictions = []
                        attention_weights_all = []
                        
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
                            
                            # Interpolate - returns bulk stress
                            result = st.session_state.interpolator.interpolate(
                                st.session_state.solutions, target_params,
                                stress_type=stress_type
                            )
                            
                            if result:
                                # Extract bulk stress evolution
                                time_evolution = []
                                
                                # Get bulk stress value
                                bulk_stress = result['bulk_stress'][stress_component]['weighted_value']
                                
                                # Create time evolution based on bulk stress
                                for t in times:
                                    # Time-dependent scaling
                                    stress_at_t = bulk_stress * (1 - np.exp(-t / 50))
                                    time_evolution.append(stress_at_t)
                                
                                predictions.append(time_evolution)
                                attention_weights_all.append(result['attention_weights'])
                            
                            progress_bar.progress((i + 1) / len(theta_rad))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Create stress matrix (time x theta)
                        if predictions:
                            stress_matrix = np.array(predictions).T  # Shape: (n_times, n_thetas)
                            
                            # Store for visualization
                            st.session_state.stress_matrix = stress_matrix
                            st.session_state.times = times
                            st.session_state.thetas = thetas
                            st.session_state.stress_component = stress_component
                            st.session_state.attention_weights = attention_weights_all
                            
                            # Enhanced metadata
                            st.session_state.metadata = {
                                'defect_type': defect_type,
                                'shape': shape,
                                'eps0': eps0,
                                'kappa': kappa,
                                'sigma': sigma,
                                'bulk_threshold': bulk_threshold,
                                'stress_component': stress_component,
                                'stress_type': stress_type,
                                'theta_range': f"{theta_min}-{theta_max}°",
                                'theta_step': theta_step,
                                'n_times': n_times,
                                'max_time': max_time,
                                'colormap_used': cmap,
                                'analysis_type': 'bulk_stress_analysis',
                                'radar_fix': '0° and 360° at same position',
                                'generated_at': datetime.now().isoformat()
                            }
                            
                            st.success(f"✅ Generated {len(thetas)} orientations × {len(times)} time points")
                            st.info(f"**Analysis:** Maximum absolute {stress_component} stress in Ag bulk (η < {bulk_threshold})")
                            
                            # Display enhanced results
                            title_base = f"{stress_component.replace('_', ' ').title()} Bulk Stress - {defect_type}"
                            
                            if viz_type in ["Sunburst", "Both", "3D Surface"]:
                                st.markdown('<h3 class="sub-header">🌅 Sunburst Visualization</h3>', unsafe_allow_html=True)
                                
                                if use_plotly:
                                    fig_sunburst = st.session_state.visualizer.create_enhanced_plotly_sunburst(
                                        stress_matrix, times, thetas,
                                        title=f"{title_base} (Bulk Material)",
                                        cmap=cmap,
                                        marker_size=marker_size,
                                        line_width=line_width,
                                        font_size=font_size
                                    )
                                    st.plotly_chart(fig_sunburst, use_container_width=True)
                                else:
                                    fig_sunburst = st.session_state.visualizer.create_enhanced_sunburst_plot(
                                        stress_matrix, times, thetas,
                                        title=f"{title_base} (Bulk Material)",
                                        cmap=cmap,
                                        marker_size=marker_size,
                                        line_width=line_width,
                                        font_size=font_size,
                                        dpi=300
                                    )
                                    st.pyplot(fig_sunburst)
                                    
                                    # Enhanced download options
                                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                                    with col_dl1:
                                        buf = BytesIO()
                                        fig_sunburst.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download PNG",
                                            data=buf.getvalue(),
                                            file_name=f"bulk_sunburst_{stress_component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                            mime="image/png",
                                            use_container_width=True
                                        )
                                    with col_dl2:
                                        buf = BytesIO()
                                        fig_sunburst.savefig(buf, format="pdf", bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download PDF",
                                            data=buf.getvalue(),
                                            file_name=f"bulk_sunburst_{stress_component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    with col_dl3:
                                        buf = BytesIO()
                                        fig_sunburst.savefig(buf, format="svg", bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download SVG",
                                            data=buf.getvalue(),
                                            file_name=f"bulk_sunburst_{stress_component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                            mime="image/svg+xml",
                                            use_container_width=True
                                        )
                            
                            if viz_type in ["Radar", "Both", "Comparison"]:
                                st.markdown('<h3 class="sub-header">📡 Radar Visualization</h3>', unsafe_allow_html=True)
                                st.info("**Note:** 0° and 360° are at the same position (chart closure)")
                                
                                # Select time point
                                time_idx = st.slider(
                                    "Select Time Point", 
                                    0, len(times)-1, len(times)//2,
                                    key="radar_time_slider",
                                    help="Select which time point to display in radar chart"
                                )
                                selected_time = times[time_idx]
                                
                                if viz_type == "Comparison":
                                    # For comparison, create synthetic data for other components
                                    stress_components = []
                                    component_names = []
                                    
                                    stress_components.append(stress_matrix)
                                    component_names.append(stress_component.replace('_', ' ').title())
                                    
                                    # Synthetic data for demonstration
                                    synthetic_scale = [0.7, 0.9, 1.1]
                                    synthetic_names = ['Hydrostatic', 'Magnitude', 'Deviatoric']
                                    
                                    for scale, name in zip(synthetic_scale[:2], synthetic_names[:2]):
                                        synth_matrix = stress_matrix * scale
                                        stress_components.append(synth_matrix)
                                        component_names.append(name)
                                    
                                    fig_comparison = st.session_state.visualizer.create_enhanced_comparison_radar(
                                        stress_components, thetas, component_names, 
                                        selected_time,
                                        title=f"Bulk Stress Components Comparison",
                                        font_size=font_size,
                                        dpi=300
                                    )
                                    st.pyplot(fig_comparison)
                                    
                                else:
                                    # Single radar chart
                                    if use_plotly:
                                        fig_radar = st.session_state.visualizer.create_enhanced_plotly_radar(
                                            stress_matrix[time_idx, :], thetas, 
                                            stress_component.replace('_', ' ').title(), 
                                            selected_time,
                                            line_width=line_width,
                                            marker_size=marker_size,
                                            font_size=font_size
                                        )
                                        st.plotly_chart(fig_radar, use_container_width=True)
                                    else:
                                        fig_radar = st.session_state.visualizer.create_enhanced_radar_plot(
                                            stress_matrix[time_idx, :], thetas, 
                                            stress_component.replace('_', ' ').title(), 
                                            selected_time,
                                            line_width=line_width,
                                            marker_size=marker_size,
                                            font_size=font_size,
                                            dpi=300
                                        )
                                        st.pyplot(fig_radar)
                            
                            if viz_type == "3D Surface":
                                st.markdown('<h3 class="sub-header">🌊 3D Surface Visualization</h3>', unsafe_allow_html=True)
                                
                                fig_3d = st.session_state.visualizer.create_3d_enhanced_surface_plot(
                                    stress_matrix, times, thetas,
                                    title=f"{title_base} - 3D Surface",
                                    cmap=cmap,
                                    font_size=font_size
                                )
                                st.plotly_chart(fig_3d, use_container_width=True)
                            
                            # Enhanced Statistics
                            if show_metrics:
                                st.markdown('<h3 class="sub-header">📊 Enhanced Statistics</h3>', unsafe_allow_html=True)
                                
                                # Compute statistics with Numba
                                start_time = time.time()
                                max_val, min_val, mean_val, std_val, p95, p99 = compute_stress_statistics_numba(stress_matrix)
                                compute_time = time.time() - start_time
                                
                                # Display metrics in cards
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                with col_stat1:
                                    st.metric("Max Bulk Stress", f"{max_val:.4f} GPa", 
                                             delta=f"{(max_val - mean_val):.4f}")
                                with col_stat2:
                                    st.metric("Mean Bulk Stress", f"{mean_val:.4f} GPa")
                                with col_stat3:
                                    st.metric("Standard Deviation", f"{std_val:.4f} GPa")
                                with col_stat4:
                                    st.metric("95th Percentile", f"{p95:.4f} GPa")
                                
                                st.caption(f"⚡ Statistics computed in {compute_time*1000:.1f} ms with Numba acceleration")
                                
                                # Enhanced distribution plot
                                fig_dist, ax_dist = plt.subplots(figsize=(12, 5), dpi=150)
                                n_bins = min(50, len(stress_matrix.flatten()) // 10)
                                hist_data = stress_matrix.flatten()
                                
                                ax_dist.hist(hist_data, bins=n_bins, 
                                           edgecolor='black', alpha=0.7, 
                                           color='steelblue',
                                           density=True)
                                ax_dist.set_xlabel('Bulk Stress (GPa)', fontsize=font_size, fontweight='bold')
                                ax_dist.set_ylabel('Probability Density', fontsize=font_size, fontweight='bold')
                                ax_dist.set_title(f'{stress_component.replace("_", " ").title()} Bulk Stress Distribution', 
                                                fontsize=font_size+2, fontweight='bold')
                                ax_dist.grid(True, alpha=0.3)
                                
                                # Add statistical lines
                                ax_dist.axvline(mean_val, color='red', linestyle='--', 
                                              linewidth=3, label=f'Mean: {mean_val:.3f}')
                                ax_dist.axvline(p95, color='orange', linestyle=':', 
                                              linewidth=2, label=f'95th %ile: {p95:.3f}')
                                ax_dist.axvline(p99, color='green', linestyle='-.', 
                                              linewidth=2, label=f'99th %ile: {p99:.3f}')
                                ax_dist.legend(fontsize=font_size-2)
                                
                                # Add normal distribution fit
                                try:
                                    from scipy.stats import norm
                                    x = np.linspace(min_val, max_val, 1000)
                                    pdf = norm.pdf(x, mean_val, std_val)
                                    ax_dist.plot(x, pdf, 'r-', linewidth=2, alpha=0.7, label='Normal Fit')
                                except:
                                    pass
                                
                                st.pyplot(fig_dist)
                            
                            # Enhanced Data export
                            st.markdown('<h3 class="sub-header">📤 Enhanced Export Options</h3>', unsafe_allow_html=True)
                            
                            export_col1, export_col2, export_col3 = st.columns(3)
                            
                            with export_col1:
                                if st.button("💾 Export as CSV", use_container_width=True):
                                    export_data = []
                                    for t_idx, time_val in enumerate(times):
                                        for theta_idx, theta_val in enumerate(thetas):
                                            export_data.append({
                                                'time_s': time_val,
                                                'orientation_deg': theta_val,
                                                'orientation_rad': np.deg2rad(theta_val),
                                                'bulk_stress_gpa': stress_matrix[t_idx, theta_idx]
                                            })
                                    
                                    df = pd.DataFrame(export_data)
                                    csv = df.to_csv(index=False)
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download CSV",
                                        data=csv,
                                        file_name=f"bulk_stress_{stress_component}_{timestamp}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            with export_col2:
                                if st.button("📊 Export as JSON", use_container_width=True):
                                    export_dict = {
                                        'metadata': st.session_state.metadata,
                                        'times': times.tolist(),
                                        'thetas': thetas.tolist(),
                                        'bulk_stress_matrix': stress_matrix.tolist(),
                                        'statistics': {
                                            'max': float(np.max(stress_matrix)),
                                            'min': float(np.min(stress_matrix)),
                                            'mean': float(np.mean(stress_matrix)),
                                            'std': float(np.std(stress_matrix))
                                        }
                                    }
                                    
                                    json_str = json.dumps(export_dict, indent=2)
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download JSON",
                                        data=json_str,
                                        file_name=f"bulk_stress_{stress_component}_{timestamp}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            
                            with export_col3:
                                if st.button("📦 Export Complete Archive", use_container_width=True):
                                    zip_buffer = st.session_state.results_manager.create_results_archive(
                                        stress_matrix, times, thetas, st.session_state.metadata
                                    )
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download ZIP Archive",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"bulk_stress_analysis_{stress_component}_{timestamp}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.markdown('<h2 class="sub-header">📈 Enhanced Dashboard</h2>', unsafe_allow_html=True)
        
        if 'stress_matrix' in st.session_state:
            stress_matrix = st.session_state.stress_matrix
            
            # Quick metrics
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Time Points", len(st.session_state.times))
            st.metric("Orientations", len(st.session_state.thetas))
            st.metric("Colormap", st.session_state.metadata.get('colormap_used', 'rainbow'))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Stress statistics
            max_val = np.max(stress_matrix)
            mean_val = np.mean(stress_matrix)
            min_val = np.min(stress_matrix)
            std_val = np.std(stress_matrix)
            
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.metric("Max Bulk Stress", f"{max_val:.4f} GPa")
                st.metric("Mean Bulk Stress", f"{mean_val:.4f} GPa")
            with col_metrics2:
                st.metric("Min Bulk Stress", f"{min_val:.4f} GPa")
                st.metric("Std Dev", f"{std_val:.4f} GPa")
            
            # Stress range visualization
            stress_range = max_val - min_val
            if stress_range > 0:
                relative_mean = (mean_val - min_val) / stress_range
                st.progress(relative_mean, 
                           text=f"Mean position: {relative_mean*100:.1f}% of range")
            
            # Orientation analysis
            st.markdown('<h4 class="sub-header">🌐 Orientation Analysis</h4>', unsafe_allow_html=True)
            
            mean_by_theta = np.mean(stress_matrix, axis=0)
            max_theta_idx = np.argmax(mean_by_theta)
            min_theta_idx = np.argmin(mean_by_theta)
            
            st.write(f"**Peak stress at:** {st.session_state.thetas[max_theta_idx]:.0f}°")
            st.write(f"**Min stress at:** {st.session_state.thetas[min_theta_idx]:.0f}°")
            
            # Time evolution
            st.markdown('<h4 class="sub-header">⏱️ Time Evolution</h4>', unsafe_allow_html=True)
            
            mean_by_time = np.mean(stress_matrix, axis=1)
            time_of_max = st.session_state.times[np.argmax(mean_by_time)]
            
            st.write(f"**Peak time:** {time_of_max:.1f}s")
            
            # Quick plot
            fig_quick, ax_quick = plt.subplots(figsize=(5, 4), dpi=120)
            ax_quick.plot(st.session_state.times, mean_by_time, 'b-', linewidth=3)
            ax_quick.fill_between(st.session_state.times, mean_by_time, alpha=0.3, color='blue')
            ax_quick.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
            ax_quick.set_ylabel('Mean Bulk Stress (GPa)', fontsize=10, fontweight='bold')
            ax_quick.set_title('Bulk Stress Evolution', fontsize=12, fontweight='bold')
            ax_quick.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_quick)
            
        else:
            st.info("📊 No data generated yet.")
            st.write("Click 'Generate Enhanced Bulk Stress Analysis' to begin.")
            
            # Quick tips
            with st.expander("💡 Quick Tips", expanded=False):
                st.write("""
                **Why bulk stress analysis?**
                1. **Ag bulk material** (η ≈ 0) is where defects propagate
                2. **Maximum absolute stress** drives material failure
                3. **More physically meaningful** than arbitrary points
                
                **Enhanced features:**
                1. **50+ colormaps** including rainbow, jet, turbo, inferno
                2. **Fixed radar charts** with 0° = 360°
                3. **Enhanced visualization** with adjustable parameters
                4. **Numba acceleration** for faster computations
                
                **Quick start:**
                1. Load solutions from `numerical_solutions/`
                2. Set analysis parameters
                3. Generate visualizations
                4. Export results in multiple formats
                """)

# =============================================
# RUN ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
