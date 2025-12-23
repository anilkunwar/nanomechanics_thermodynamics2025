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
import cmasher as cmr
from scipy.spatial.distance import cdist, euclidean
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import plotly.express as px

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
# REGION ANALYSIS FUNCTIONS
# =============================================

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
# ENHANCED NUMERICAL SOLUTIONS LOADER
# =============================================
class EnhancedSolutionLoader:
    """Enhanced solution loader with support for multiple formats and caching"""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.pt_loading_method = "safe"
        
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
                try:
                    import numpy as np
                    try:
                        from numpy._core.multiarray import scalar as np_scalar
                    except ImportError:
                        try:
                            from numpy.core.multiarray import scalar as np_scalar
                        except ImportError:
                            np_scalar = None
                    
                    if np_scalar is not None:
                        import torch.serialization
                        with torch.serialization.safe_globals([np_scalar]):
                            data = torch.load(buffer, map_location='cpu', weights_only=True)
                    else:
                        if 'st' in globals():
                            st.warning("Could not import numpy scalar, using weights_only=False")
                        data = torch.load(buffer, map_location='cpu', weights_only=False)
                except Exception as safe_error:
                    if 'st' in globals():
                        st.warning(f"Safe loading failed: {safe_error}. Trying weights_only=False")
                    data = torch.load(buffer, map_location='cpu', weights_only=False)
            else:
                data = torch.load(buffer, map_location='cpu', weights_only=False)
            
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
            format_type = 'pkl'
        
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
            
            if isinstance(data, Exception):
                return {
                    'error': str(data),
                    'format': format_type,
                    'status': 'error'
                }
            
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
                    
                    history = data.get('history', [])
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        history_list = []
                        for key in sorted(history.keys()):
                            frame = history[key]
                            if isinstance(frame, dict) and 'eta' in frame and 'stresses' in frame:
                                history_list.append((frame['eta'], frame['stresses']))
                        standardized['history'] = history_list
                    
                    if 'params' in standardized:
                        for key, value in standardized['params'].items():
                            if torch.is_tensor(value):
                                standardized['params'][key] = value.cpu().numpy()
                else:
                    standardized['error'] = f"PT data is not a dictionary: {type(data)}"
            
            elif format_type == 'h5':
                if isinstance(data, dict):
                    standardized.update(data)
                else:
                    standardized['error'] = f"H5 data is not a dictionary: {type(data)}"
            
            elif format_type == 'npz':
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
                
                cache_key = f"{filename}_{os.path.getmtime(file_path)}_{pt_loading_method}"
                if use_cache and cache_key in self.cache:
                    sim = self.cache[cache_key]
                    if sim.get('status') == 'success':
                        solutions.append(sim)
                    continue
                
                if 'st' in globals():
                    progress = (idx + 1) / len(all_files_info)
                    progress_bar.progress(progress)
                    status_text.text(f"Loading {filename}...")
                
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                sim = self.read_simulation_file(file_content, file_info['format'])
                sim['filename'] = filename
                sim['file_info'] = file_info
                
                if sim.get('status') == 'success' and 'params' in sim and 'history' in sim:
                    if isinstance(sim['params'], dict):
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
        
        if failed_files and 'st' in globals():
            with st.expander(f"⚠️ Failed to load {len(failed_files)} files", expanded=False):
                for failed in failed_files[:10]:
                    st.error(f"**{failed['filename']}**: {failed['error']}")
                if len(failed_files) > 10:
                    st.info(f"... and {len(failed_files) - 10} more files failed to load.")
        
        return solutions

# =============================================
# ATTENTION-BASED INTERPOLATOR WITH SPATIAL LOCALITY
# =============================================

class AttentionSpatialInterpolator:
    """Transformer-inspired attention interpolator with spatial locality regularization"""
    
    def __init__(self, sigma=0.3, use_numba=True, attention_dim=32, num_heads=4):
        self.sigma = sigma
        self.use_numba = use_numba
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        
        # Parameter mappings
        self.defect_map = {
            'ISF': [1, 0, 0, 0],
            'ESF': [0, 1, 0, 0],
            'Twin': [0, 0, 1, 0],
            'Unknown': [0, 0, 0, 1]
        }
        
        self.shape_map = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
        
        # Initialize attention layers
        self.query_projection = nn.Linear(12, attention_dim)
        self.key_projection = nn.Linear(12, attention_dim)
        self.value_projection = nn.Linear(12, attention_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.output_projection = nn.Linear(attention_dim, 12)
    
    def compute_parameter_vector(self, params):
        """Convert parameters to numerical vector with 12 dimensions"""
        vector = []
        
        if not isinstance(params, dict):
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0.5, 0.0], dtype=np.float32)
        
        # Defect type (4 dimensions)
        defect = params.get('defect_type', 'ISF')
        vector.extend(self.defect_map.get(defect, [0, 0, 0, 0]))
        
        # Shape (5 dimensions)
        shape = params.get('shape', 'Square')
        vector.extend(self.shape_map.get(shape, [0, 0, 0, 0, 0]))
        
        # Numeric parameters (3 dimensions)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        # Normalize parameters
        vector.append((eps0 - 0.3) / (3.0 - 0.3))  # eps0 normalized 0-1
        vector.append((kappa - 0.1) / (2.0 - 0.1))  # kappa normalized 0-1
        vector.append(theta / np.pi)  # theta normalized 0-1 (0 to π)
        
        return np.array(vector, dtype=np.float32)
    
    def compute_attention_weights(self, source_vectors, target_vector, use_spatial=True):
        """Compute attention weights using transformer-like attention with spatial regularization"""
        
        # Convert to PyTorch tensors
        source_tensor = torch.FloatTensor(source_vectors).unsqueeze(0)  # (1, N, 12)
        target_tensor = torch.FloatTensor(target_vector).unsqueeze(0).unsqueeze(1)  # (1, 1, 12)
        
        # Project to attention space
        query = self.query_projection(target_tensor)  # (1, 1, attention_dim)
        keys = self.key_projection(source_tensor)     # (1, N, attention_dim)
        values = self.value_projection(source_tensor) # (1, N, attention_dim)
        
        # Multi-head attention
        attention_output, attention_weights = self.multihead_attention(
            query, keys, values
        )
        
        # Get attention weights (averaged over heads)
        attention_weights = attention_weights.squeeze().detach().numpy()
        
        # Apply spatial locality regularization
        if use_spatial and len(source_vectors) > 0:
            spatial_weights = self.compute_spatial_weights(source_vectors, target_vector)
            # Combine attention and spatial weights
            combined_weights = attention_weights * spatial_weights
            # Normalize
            if np.sum(combined_weights) > 0:
                combined_weights = combined_weights / np.sum(combined_weights)
            else:
                combined_weights = np.ones_like(combined_weights) / len(combined_weights)
            return combined_weights
        
        return attention_weights
    
    def compute_spatial_weights(self, source_vectors, target_vector):
        """Compute spatial locality weights using Euclidean distance"""
        if len(source_vectors) == 0:
            return np.array([])
        
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
        
        # Apply Gaussian kernel
        weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
    
    def interpolate_precise_orientation(self, sources, target_angle_deg, target_params,
                                       region_type='bulk', stress_component='von_mises',
                                       stress_type='max_abs', use_spatial=True):
        """Interpolate at a precise orientation angle with high precision"""
        
        # Convert angle to radians
        target_angle_rad = np.deg2rad(target_angle_deg)
        
        # Update target params with precise angle
        precise_target_params = target_params.copy()
        precise_target_params['theta'] = target_angle_rad
        
        # Compute target vector
        target_vector = self.compute_parameter_vector(precise_target_params)
        
        # Filter and validate sources
        valid_sources = []
        source_vectors = []
        source_stresses = []
        
        for src in sources:
            if not isinstance(src, dict):
                continue
            if 'params' not in src or 'history' not in src:
                continue
            if not isinstance(src.get('params'), dict):
                continue
            
            valid_sources.append(src)
            
            # Get source parameter vector
            src_vector = self.compute_parameter_vector(src['params'])
            source_vectors.append(src_vector)
            
            # Extract stress from last frame
            history = src.get('history', [])
            if history:
                last_frame = history[-1]
                if isinstance(last_frame, tuple) and len(last_frame) >= 2:
                    eta, stress_fields = last_frame[0], last_frame[1]
                elif isinstance(last_frame, dict):
                    eta = last_frame.get('eta', np.zeros((128, 128)))
                    stress_fields = last_frame.get('stresses', {})
                else:
                    eta = np.zeros((128, 128))
                    stress_fields = {}
                
                region_stress = extract_region_stress(
                    eta, stress_fields, region_type, stress_component, stress_type
                )
                source_stresses.append(region_stress)
            else:
                source_stresses.append(0.0)
        
        if not valid_sources:
            return None
        
        source_vectors = np.array(source_vectors)
        source_stresses = np.array(source_stresses)
        
        # Compute attention weights
        if len(source_vectors) > 0:
            if self.use_numba:
                try:
                    spatial_weights = compute_gaussian_weights_numba(
                        source_vectors, target_vector, self.sigma
                    )
                except:
                    distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
                    spatial_weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
                    spatial_weights = spatial_weights / (np.sum(spatial_weights) + 1e-8)
            else:
                distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
                spatial_weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
                spatial_weights = spatial_weights / (np.sum(spatial_weights) + 1e-8)
            
            # Combine with attention weights if available
            if use_spatial:
                attention_weights = self.compute_attention_weights(
                    source_vectors, target_vector, use_spatial=True
                )
                # Blend attention and spatial weights
                final_weights = 0.7 * attention_weights + 0.3 * spatial_weights
            else:
                final_weights = spatial_weights
            
            # Normalize
            if np.sum(final_weights) > 0:
                final_weights = final_weights / np.sum(final_weights)
            else:
                final_weights = np.ones_like(final_weights) / len(final_weights)
            
            # Weighted combination
            weighted_stress = np.sum(final_weights * source_stresses)
            
            return {
                'region_stress': weighted_stress,
                'attention_weights': final_weights,
                'target_params': precise_target_params,
                'target_angle_deg': target_angle_deg,
                'target_angle_rad': target_angle_rad,
                'region_type': region_type,
                'stress_component': stress_component,
                'stress_type': stress_type,
                'num_sources': len(valid_sources),
                'spatial_sigma': self.sigma
            }
        
        return None
    
    def create_orientation_sweep(self, sources, base_params, angle_range, n_points=50,
                                region_type='bulk', stress_component='von_mises',
                                stress_type='max_abs'):
        """Create interpolation sweep across orientation range"""
        
        min_angle, max_angle = angle_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        stresses = []
        weights_list = []
        
        progress_bar = st.progress(0) if 'st' in globals() else None
        status_text = st.empty() if 'st' in globals() else None
        
        for i, angle in enumerate(angles):
            if 'st' in globals():
                status_text.text(f"Interpolating at {angle:.2f}° ({i+1}/{n_points})...")
                progress_bar.progress((i + 1) / n_points)
            
            result = self.interpolate_precise_orientation(
                sources, angle, base_params,
                region_type, stress_component, stress_type
            )
            
            if result:
                stresses.append(result['region_stress'])
                weights_list.append(result['attention_weights'])
            else:
                stresses.append(0.0)
                weights_list.append(np.zeros(len(sources)))
        
        if 'st' in globals():
            progress_bar.empty()
            status_text.empty()
        
        return {
            'angles': angles,
            'stresses': np.array(stresses),
            'weights_matrix': np.array(weights_list).T,  # Sources × Angles
            'region_type': region_type,
            'stress_component': stress_component,
            'stress_type': stress_type,
            'angle_range': angle_range,
            'n_points': n_points
        }

# =============================================
# ORIGINAL FILE ANALYZER WITH ORIENTATION SUPPORT
# =============================================

class OriginalFileAnalyzer:
    """Analyze original loaded files for different regions with orientation support"""
    
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
        theta = params.get('theta', 0.0)
        theta_deg = np.rad2deg(theta) if theta is not None else 0.0
        
        return {
            'region_stress': region_stress,
            'region_statistics': region_stats,
            'params': params,
            'theta_rad': theta,
            'theta_deg': theta_deg,
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
    
    def get_solutions_by_orientation(self, solutions, min_angle=0, max_angle=360, tolerance=1.0):
        """Filter solutions by orientation range"""
        filtered = []
        
        for sol in solutions:
            params = sol.get('params', {})
            theta = params.get('theta', 0.0)
            theta_deg = np.rad2deg(theta) if theta is not None else 0.0
            
            # Normalize angle to 0-360
            theta_deg = theta_deg % 360
            
            if min_angle <= theta_deg <= max_angle or \
               (min_angle > max_angle and (theta_deg >= min_angle or theta_deg <= max_angle)):
                filtered.append(sol)
            elif abs(theta_deg - min_angle) <= tolerance or abs(theta_deg - max_angle) <= tolerance:
                filtered.append(sol)
        
        return filtered
    
    def create_orientation_distribution(self, solutions):
        """Create distribution of orientations in loaded solutions"""
        orientations = []
        for sol in solutions:
            params = sol.get('params', {})
            theta = params.get('theta', 0.0)
            theta_deg = np.rad2deg(theta) if theta is not None else 0.0
            orientations.append(theta_deg % 360)
        
        return np.array(orientations)
    
    def create_original_sweep_matrix(self, solutions, angle_range, n_points=50,
                                    region_type='bulk', stress_component='von_mises',
                                    stress_type='max_abs'):
        """Create stress matrix from original solutions for orientation sweep"""
        if not solutions:
            return None, None
        
        min_angle, max_angle = angle_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        # Group solutions by nearest angle
        angle_bins = {}
        for sol in solutions:
            analysis = self.analyze_solution(sol, region_type, stress_component, stress_type)
            if analysis:
                theta_deg = analysis['theta_deg'] % 360
                # Find nearest angle in our grid
                nearest_idx = np.argmin(np.abs(angles - theta_deg))
                nearest_angle = angles[nearest_idx]
                
                if nearest_angle not in angle_bins:
                    angle_bins[nearest_angle] = []
                angle_bins[nearest_angle].append(analysis['region_stress'])
        
        # Average stresses for each angle
        stresses = []
        valid_angles = []
        
        for angle in angles:
            if angle in angle_bins and angle_bins[angle]:
                avg_stress = np.mean(angle_bins[angle])
                stresses.append(avg_stress)
                valid_angles.append(angle)
            else:
                # Use NaN for missing data
                stresses.append(np.nan)
                valid_angles.append(angle)
        
        return np.array(stresses), np.array(valid_angles)

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER
# =============================================

class EnhancedSunburstRadarVisualizer:
    """Enhanced sunburst and radar charts with 50+ colormaps and visualization enhancements"""
    
    def __init__(self):
        self.colormap_manager = EnhancedColorMaps()
    
    def create_enhanced_plotly_sunburst(self, stress_matrix, times, thetas, title, 
                                       cmap='rainbow', marker_size=12, line_width=1.5,
                                       font_size=18, width=900, height=750,
                                       show_colorbar=True, colorbar_title="Stress (GPa)",
                                       hover_template=None, is_time_series=True):
        """Interactive sunburst with Plotly - handles both time series and orientation sweeps"""
        
        if is_time_series:
            # Time series sunburst
            theta_deg = np.deg2rad(thetas)
            theta_grid, time_grid = np.meshgrid(theta_deg, times)
            
            # Flatten the arrays for scatter plot
            r_flat = time_grid.flatten()
            theta_flat = np.rad2deg(theta_grid).flatten()
            stress_flat = stress_matrix.flatten()
        else:
            # Orientation sweep sunburst (single time point)
            r_flat = np.zeros_like(thetas)  # Zero time dimension
            theta_flat = thetas
            stress_flat = stress_matrix
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Default hover template
        if hover_template is None:
            if is_time_series:
                hover_template = (
                    '<b>Time</b>: %{r:.2f}s<br>' +
                    '<b>Orientation</b>: %{theta:.1f}°<br>' +
                    '<b>Stress</b>: %{marker.color:.4f} GPa<br>' +
                    '<extra></extra>'
                )
            else:
                hover_template = (
                    '<b>Orientation</b>: %{theta:.1f}°<br>' +
                    '<b>Stress</b>: %{marker.color:.4f} GPa<br>' +
                    '<extra></extra>'
                )
        
        # Add scatter polar trace with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=r_flat if is_time_series else stress_flat * 10,  # Scale for visibility
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
        if is_time_series:
            radial_title = "Time (s)"
            radial_range = [0, max(times) * 1.1]
            radial_ticksuffix = " s"
        else:
            radial_title = "Stress (GPa)"
            radial_range = [0, max(stress_flat) * 1.3]
            radial_ticksuffix = " GPa"
        
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
                        text=radial_title,
                        font=dict(size=font_size+2, color='black', family='Arial')
                    ),
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    showline=True,
                    tickfont=dict(size=font_size, color='black', family='Arial'),
                    tickformat='.1f',
                    range=radial_range,
                    ticksuffix=radial_ticksuffix,
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
                hole=0.1 if is_time_series else 0.0
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
        
        # Add radial lines for important orientations
        for angle in [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]:
            fig.add_trace(go.Scatterpolar(
                r=[0, radial_range[1]],
                theta=[angle, angle],
                mode='lines',
                line=dict(color='rgba(255, 0, 0, 0.2)', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Highlight specific orientations
        highlight_angles = [54.7]  # Ag FCC twin habit plane
        for angle in highlight_angles:
            fig.add_trace(go.Scatterpolar(
                r=[0, radial_range[1]],
                theta=[angle, angle],
                mode='lines',
                line=dict(color='rgba(0, 255, 0, 0.4)', width=3, dash='solid'),
                name=f'Habit Plane ({angle}°)',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        return fig
    
    def create_enhanced_plotly_radar(self, stress_values, thetas, component_name, 
                                    title="Radar Chart", line_width=4, marker_size=12, 
                                    fill_alpha=0.3, font_size=16, width=800, height=700,
                                    show_mean=True, show_std=True, color='steelblue',
                                    show_habit_plane=True):
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
        
        # Highlight habit plane orientation (54.7° for Ag FCC twin)
        if show_habit_plane:
            habit_angle = 54.7
            # Find nearest data point
            idx = np.argmin(np.abs(thetas - habit_angle))
            habit_stress = stress_values[idx]
            
            fig.add_trace(go.Scatterpolar(
                r=[0, habit_stress],
                theta=[habit_angle, habit_angle],
                mode='lines+markers',
                line=dict(color='green', width=4, dash='dashdot'),
                marker=dict(size=15, color='green', symbol='star'),
                name=f'Habit Plane ({habit_angle}°): {habit_stress:.3f} GPa',
                hovertemplate=f'Habit Plane ({habit_angle}°): %{r:.3f} GPa'
            ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
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
    
    def create_comparison_radar(self, original_stress, interpolated_stress, thetas,
                               title="Comparison: Original vs Interpolated",
                               original_name="Original", interpolated_name="Interpolated"):
        """Create radar chart comparing original and interpolated solutions"""
        
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
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgb(31, 119, 180)', width=3),
            name=original_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Original Stress: %{r:.4f} GPa'
        ))
        
        # Interpolated solutions
        fig.add_trace(go.Scatterpolar(
            r=interpolated_closed,
            theta=thetas_closed,
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgb(255, 127, 14)', width=3),
            name=interpolated_name,
            hovertemplate='Orientation: %{theta:.1f}°<br>Interpolated Stress: %{r:.4f} GPa'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family="Arial Black", color='darkblue'),
                x=0.5,
                xanchor='center'
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(original_stress), max(interpolated_stress)) * 1.2],
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    tickfont=dict(size=14, color='black'),
                    title=dict(text='Stress (GPa)', 
                              font=dict(size=16, color='black'))
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
                    tickfont=dict(size=14, color='black')
                ),
                bgcolor="rgba(240, 240, 240, 0.3)"
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=2,
                font=dict(size=14, family='Arial')
            ),
            width=900,
            height=700
        )
        
        return fig

# =============================================
# ENHANCED RESULTS MANAGER
# =============================================

class EnhancedResultsManager:
    """Manager for saving and exporting results with enhanced formatting"""
    
    @staticmethod
    def prepare_orientation_sweep_data(sweep_results, original_results=None, metadata=None):
        """Prepare orientation sweep data for export"""
        if metadata is None:
            metadata = {}
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'orientation_sweep',
                'software_version': '4.0.0',
                'habit_plane_angle': 54.7,
                'description': 'Orientation sweep analysis around Ag FCC twin habit plane'
            },
            'sweep_results': sweep_results,
            'original_results': original_results,
            'statistics': {}
        }
        
        # Add metadata
        export_data['metadata'].update(metadata)
        
        # Calculate statistics
        if 'angles' in sweep_results and 'stresses' in sweep_results:
            stresses = sweep_results['stresses']
            export_data['statistics']['sweep'] = {
                'max_stress': float(np.nanmax(stresses)),
                'min_stress': float(np.nanmin(stresses)),
                'mean_stress': float(np.nanmean(stresses)),
                'std_stress': float(np.nanstd(stresses)),
                'num_points': len(stresses),
                'angle_range': f"{sweep_results.get('angle_range', [0, 360])}"
            }
        
        if original_results and 'stresses' in original_results:
            orig_stresses = original_results['stresses']
            export_data['statistics']['original'] = {
                'max_stress': float(np.nanmax(orig_stresses)),
                'min_stress': float(np.nanmin(orig_stresses)),
                'mean_stress': float(np.nanmean(orig_stresses)),
                'std_stress': float(np.nanstd(orig_stresses)),
                'num_points': len(orig_stresses)
            }
        
        return export_data
    
    @staticmethod
    def create_orientation_sweep_archive(sweep_results, original_results, metadata):
        """Create ZIP archive with orientation sweep results"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save sweep results
            sweep_data = json.dumps(sweep_results, indent=2, default=float)
            zip_file.writestr('sweep_results.json', sweep_data)
            
            # Save original results if available
            if original_results:
                orig_data = json.dumps(original_results, indent=2, default=float)
                zip_file.writestr('original_results.json', orig_data)
            
            # Save metadata
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr('metadata.json', metadata_json)
            
            # Save CSV data
            csv_rows = []
            if 'angles' in sweep_results and 'stresses' in sweep_results:
                for angle, stress in zip(sweep_results['angles'], sweep_results['stresses']):
                    csv_rows.append({
                        'angle_deg': f"{angle:.3f}",
                        'angle_rad': f"{np.deg2rad(angle):.6f}",
                        'stress_gpa': f"{stress:.6f}",
                        'region': sweep_results.get('region_type', 'unknown'),
                        'component': sweep_results.get('stress_component', 'unknown'),
                        'analysis_type': sweep_results.get('stress_type', 'unknown')
                    })
            
            df = pd.DataFrame(csv_rows)
            csv_str = df.to_csv(index=False)
            zip_file.writestr('orientation_sweep_data.csv', csv_str)
            
            # Save attention weights if available
            if 'weights_matrix' in sweep_results:
                weights_df = pd.DataFrame(sweep_results['weights_matrix'])
                weights_csv = weights_df.to_csv(index=False, header=False)
                zip_file.writestr('attention_weights.csv', weights_csv)
            
            # Add comprehensive README
            readme = f"""# ORIENTATION SWEEP ANALYSIS RESULTS
Generated: {datetime.now().isoformat()}

## ANALYSIS DETAILS
- Target Defect: {metadata.get('defect_type', 'Twin')}
- Shape: {metadata.get('shape', 'Unknown')}
- ε*: {metadata.get('eps0', 'Unknown')}
- κ: {metadata.get('kappa', 'Unknown')}
- Region: {sweep_results.get('region_type', 'Unknown')}
- Stress Component: {sweep_results.get('stress_component', 'Unknown')}
- Analysis Type: {sweep_results.get('stress_type', 'Unknown')}

## HABIT PLANE INFORMATION
- Ag FCC Twin Habit Plane: 54.7°
- Orientation Range: {sweep_results.get('angle_range', [0, 360])}
- Number of Points: {sweep_results.get('n_points', 0)}
- Spatial Sigma: {sweep_results.get('spatial_sigma', 'Unknown')}

## FILES INCLUDED
1. sweep_results.json - Complete sweep results
2. original_results.json - Original file analysis
3. metadata.json - Analysis metadata
4. orientation_sweep_data.csv - Tabular data for plotting
5. attention_weights.csv - Attention weights matrix

## SPATIAL LOCALITY REGULARIZATION
The interpolation uses:
- Euclidean distance in 12D parameter space
- Gaussian kernel: exp(-0.5 * (distance/sigma)²)
- Attention mechanism with {sweep_results.get('num_heads', 4)} heads
- Combined weights: 70% attention + 30% spatial

## REGION DEFINITIONS
1. Defect Region: η > 0.6 (high defect concentration)
2. Interface Region: 0.4 ≤ η ≤ 0.6 (transition region)
3. Bulk Region: η < 0.4 (pure Ag material)

## VISUALIZATION NOTES
- 0° and 360° are at same position in radar charts
- Habit plane (54.7°) is highlighted in green
- Sunburst charts show stress distribution vs orientation
- Radar charts show stress magnitude at each orientation
"""
            zip_file.writestr('README_ORIENTATION_SWEEP.txt', readme)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# MAIN APPLICATION WITH PRECISE ORIENTATION INTERPOLATION
# =============================================

def main():
    st.set_page_config(
        page_title="Ag FCC Twin: Precise Orientation Interpolation & Analysis",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
    }
    .sub-header {
        font-size: 1.6rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        border-left: 5px solid #3B82F6;
        margin: 1.2rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 0.6rem;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .habit-plane-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        border: 2px solid #047857;
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
    .attention-highlight {
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with Ag FCC twin information
    st.markdown('<h1 class="main-header">🔬 Ag FCC Twin: Precise Orientation Interpolation</h1>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    <strong>🎯 Key Feature:</strong> Precise interpolation at 54.7° (Ag FCC twin habit plane) with attention-based spatial regularization<br>
    <strong>📊 Analysis:</strong> Three region types (Defect, Interface, Bulk) with sunburst/radar visualizations<br>
    <strong>🧠 Method:</strong> Transformer-inspired attention + Euclidean distance spatial regularization<br>
    <strong>🔍 Purpose:</strong> Validate interpolated solutions against original files for habit plane analysis
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = AttentionSpatialInterpolator(
            sigma=0.3, 
            use_numba=True, 
            attention_dim=32, 
            num_heads=4
        )
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'original_analyzer' not in st.session_state:
        st.session_state.original_analyzer = OriginalFileAnalyzer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    
    # Sidebar with comprehensive options
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Analysis Configuration</h2>', unsafe_allow_html=True)
        
        # Habit plane information
        st.markdown('<div class="habit-plane-card">', unsafe_allow_html=True)
        st.markdown("### Ag FCC Twin Habit Plane")
        st.write("**Preferred Orientation:** 54.7°")
        st.write("**Crystal System:** Face-Centered Cubic")
        st.write("**Defect Type:** Coherent Twin Boundary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data source selection
        st.markdown("#### 📊 Data Source & Analysis")
        analysis_mode = st.radio(
            "Select analysis mode:",
            ["Precise Single Orientation", "Orientation Sweep", "Compare Original vs Interpolated"],
            index=0,
            help="Choose between precise single point, sweep, or comparison analysis"
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
        
        # Interpolator settings
        st.markdown("#### 🧠 Interpolator Settings")
        use_spatial = st.checkbox(
            "Use Spatial Locality Regularization", 
            value=True,
            help="Use Euclidean distance in parameter space for regularization"
        )
        
        spatial_sigma = st.slider(
            "Spatial Sigma (σ)",
            0.05, 1.0, 0.3, 0.05,
            help="Controls the influence of spatial distance in Gaussian kernel"
        )
        
        attention_heads = st.slider(
            "Attention Heads",
            1, 8, 4, 1,
            help="Number of attention heads in transformer-like mechanism"
        )
        
        # Update interpolator settings
        if (spatial_sigma != st.session_state.interpolator.sigma or
            attention_heads != st.session_state.interpolator.num_heads):
            st.session_state.interpolator.sigma = spatial_sigma
            st.session_state.interpolator.num_heads = attention_heads
        
        # Orientation settings based on mode
        if analysis_mode == "Precise Single Orientation":
            st.markdown("#### 🎯 Single Orientation Target")
            target_angle = st.number_input(
                "Target Orientation (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=54.7,
                step=0.1,
                format="%.2f",
                help="Enter precise orientation angle (e.g., 54.70° for Ag FCC twin)"
            )
            
            # Show habit plane reminder
            if abs(target_angle - 54.7) < 0.1:
                st.success(f"✅ Targeting Ag FCC twin habit plane ({target_angle}°)")
        
        elif analysis_mode == "Orientation Sweep":
            st.markdown("#### 🌐 Orientation Sweep Range")
            col_sweep1, col_sweep2 = st.columns(2)
            with col_sweep1:
                min_angle = st.number_input(
                    "Min Angle (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=53.12,
                    step=0.1,
                    format="%.2f"
                )
            with col_sweep2:
                max_angle = st.number_input(
                    "Max Angle (°)",
                    min_value=0.0,
                    max_value=360.0,
                    value=56.82,
                    step=0.1,
                    format="%.2f"
                )
            
            n_points = st.slider(
                "Number of Points",
                10, 200, 100, 10,
                help="Number of orientation points in sweep"
            )
            
            # Validate range
            if min_angle >= max_angle:
                st.error("Min angle must be less than max angle")
        
        elif analysis_mode == "Compare Original vs Interpolated":
            st.markdown("#### 🔄 Comparison Settings")
            comparison_range = st.slider(
                "Orientation Range for Comparison (°)",
                0, 360, (0, 360), 5
            )
            
            n_comparison_points = st.slider(
                "Comparison Points",
                10, 100, 50, 5
            )
        
        # Target parameters
        st.markdown("#### ⚙️ Target Parameters")
        defect_type = st.selectbox("Defect Type", ["ISF", "ESF", "Twin"], index=2)
        
        col_shape, col_eps = st.columns(2)
        with col_shape:
            shape = st.selectbox("Shape", 
                                ["Square", "Horizontal Fault", "Vertical Fault", 
                                 "Rectangle", "Ellipse"], index=0)
        with col_eps:
            eps0 = st.slider("ε*", 0.3, 3.0, 0.707, 0.01)
        
        col_kappa, col_theta = st.columns(2)
        with col_kappa:
            kappa = st.slider("κ", 0.1, 2.0, 0.6, 0.01)
        
        # Visualization settings
        st.markdown("#### 🎨 Visualization")
        viz_type = st.radio(
            "Chart Type",
            ["Sunburst", "Radar", "Both", "Comparison"],
            index=1,
            horizontal=True
        )
        
        cmap = st.selectbox(
            "Color Map",
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('rainbow') if 'rainbow' in ALL_COLORMAPS else 0
        )
        
        # Load solutions
        st.markdown("#### 📂 Load Solutions")
        if st.button("🔄 Load All Solutions", use_container_width=True, type="primary"):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions(use_cache=True)
                if st.session_state.solutions:
                    st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📋 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                # Orientation distribution
                orientations = st.session_state.original_analyzer.create_orientation_distribution(
                    st.session_state.solutions
                )
                
                if len(orientations) > 0:
                    fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
                    ax_dist.hist(orientations, bins=20, edgecolor='black', alpha=0.7)
                    ax_dist.set_xlabel('Orientation (°)')
                    ax_dist.set_ylabel('Count')
                    ax_dist.set_title('Orientation Distribution in Loaded Solutions')
                    ax_dist.axvline(54.7, color='green', linestyle='--', label='Habit Plane (54.7°)')
                    ax_dist.legend()
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
                
                # Solutions summary
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    params = sol.get('params', {})
                    theta = params.get('theta', 0.0)
                    theta_deg = np.rad2deg(theta) if theta is not None else 0.0
                    
                    st.write(f"**{i+1}. {sol.get('filename', 'Unknown')}**")
                    st.caption(f"Type: {params.get('defect_type', '?')} | "
                              f"θ: {theta_deg:.1f}° | "
                              f"ε*: {params.get('eps0', 0):.2f} | "
                              f"Frames: {len(sol.get('history', []))}")
                
                if len(st.session_state.solutions) > 5:
                    st.info(f"... and {len(st.session_state.solutions) - 5} more")
    
    # Main content area
    col_main1, col_main2 = st.columns([3, 1])
    
    with col_main1:
        st.markdown('<h2 class="sub-header">🚀 Precise Orientation Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.solutions:
            st.warning("⚠️ Please load solutions first using the button in the sidebar.")
            
            with st.expander("📁 Directory Information", expanded=False):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.success(f"✅ Found {total_files} files in {SOLUTIONS_DIR}")
                    for fmt, files in file_formats.items():
                        if files:
                            st.info(f"• **{fmt.upper()}**: {len(files)} files")
                else:
                    st.error(f"❌ No files found in {SOLUTIONS_DIR}")
        
        else:
            # Create target parameters
            target_params = {
                'defect_type': defect_type,
                'shape': shape,
                'eps0': eps0,
                'kappa': kappa,
                'theta': 0.0  # Will be set based on mode
            }
            
            # Generate analysis button
            if st.button("✨ Generate Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Generating {analysis_mode} analysis..."):
                    try:
                        if analysis_mode == "Precise Single Orientation":
                            # Single precise orientation interpolation
                            st.info(f"🔬 Interpolating at precise orientation: {target_angle}°")
                            
                            result = st.session_state.interpolator.interpolate_precise_orientation(
                                st.session_state.solutions,
                                target_angle,
                                target_params,
                                region_key,
                                stress_component,
                                stress_type,
                                use_spatial=use_spatial
                            )
                            
                            if result:
                                st.session_state.single_result = result
                                
                                # Display results
                                st.success(f"✅ Interpolation complete at {target_angle}°")
                                
                                # Show detailed metrics
                                col_met1, col_met2, col_met3 = st.columns(3)
                                with col_met1:
                                    st.metric(
                                        "Region Stress",
                                        f"{result['region_stress']:.4f} GPa",
                                        delta="Interpolated Value"
                                    )
                                with col_met2:
                                    st.metric(
                                        "Spatial Sigma",
                                        f"{result['spatial_sigma']:.3f}",
                                        delta="Regularization Strength"
                                    )
                                with col_met3:
                                    st.metric(
                                        "Number of Sources",
                                        result['num_sources'],
                                        delta="Used for Interpolation"
                                    )
                                
                                # Show attention weights
                                with st.expander("🔍 Attention Weights Analysis", expanded=True):
                                    weights = result['attention_weights']
                                    
                                    # Create weights visualization
                                    fig_weights, ax_weights = plt.subplots(figsize=(10, 4))
                                    bars = ax_weights.bar(range(len(weights)), weights, 
                                                         color=plt.cm.viridis(weights))
                                    ax_weights.set_xlabel('Source Index')
                                    ax_weights.set_ylabel('Attention Weight')
                                    ax_weights.set_title('Attention Weights Distribution')
                                    ax_weights.grid(True, alpha=0.3)
                                    
                                    # Add value labels
                                    for i, (bar, weight) in enumerate(zip(bars, weights)):
                                        height = bar.get_height()
                                        ax_weights.text(bar.get_x() + bar.get_width()/2., height,
                                                       f'{weight:.3f}', ha='center', va='bottom', 
                                                       fontsize=8)
                                    
                                    st.pyplot(fig_weights)
                                    plt.close(fig_weights)
                                    
                                    # Show top contributors
                                    top_indices = np.argsort(weights)[-5:][::-1]
                                    st.write("**Top 5 Contributing Sources:**")
                                    for idx in top_indices:
                                        sol = st.session_state.solutions[idx]
                                        params = sol.get('params', {})
                                        theta = params.get('theta', 0.0)
                                        theta_deg = np.rad2deg(theta) if theta is not None else 0.0
                                        st.write(f"- Source {idx+1}: θ={theta_deg:.1f}°, "
                                                f"ε*={params.get('eps0', 0):.2f}, "
                                                f"weight={weights[idx]:.3f}")
                        
                        elif analysis_mode == "Orientation Sweep":
                            # Orientation sweep analysis
                            st.info(f"🌐 Performing orientation sweep from {min_angle}° to {max_angle}°")
                            
                            sweep_result = st.session_state.interpolator.create_orientation_sweep(
                                st.session_state.solutions,
                                target_params,
                                (min_angle, max_angle),
                                n_points,
                                region_key,
                                stress_component,
                                stress_type
                            )
                            
                            if sweep_result:
                                st.session_state.sweep_result = sweep_result
                                
                                # Get original solutions for comparison
                                original_stresses, original_angles = st.session_state.original_analyzer.create_original_sweep_matrix(
                                    st.session_state.solutions,
                                    (min_angle, max_angle),
                                    n_points,
                                    region_key,
                                    stress_component,
                                    stress_type
                                )
                                
                                st.session_state.original_sweep = {
                                    'stresses': original_stresses,
                                    'angles': original_angles,
                                    'region_type': region_key,
                                    'stress_component': stress_component,
                                    'stress_type': stress_type
                                }
                                
                                st.success(f"✅ Generated sweep with {n_points} points")
                                
                                # Display sweep statistics
                                with st.expander("📊 Sweep Statistics", expanded=True):
                                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                    with col_stat1:
                                        st.metric("Max Stress", f"{np.nanmax(sweep_result['stresses']):.4f} GPa")
                                    with col_stat2:
                                        st.metric("Min Stress", f"{np.nanmin(sweep_result['stresses']):.4f} GPa")
                                    with col_stat3:
                                        st.metric("Mean Stress", f"{np.nanmean(sweep_result['stresses']):.4f} GPa")
                                    with col_stat4:
                                        st.metric("Std Dev", f"{np.nanstd(sweep_result['stresses']):.4f} GPa")
                                
                                # Find stress at habit plane
                                habit_angle = 54.7
                                idx = np.argmin(np.abs(sweep_result['angles'] - habit_angle))
                                habit_stress = sweep_result['stresses'][idx]
                                
                                st.markdown(f'<div class="attention-highlight">', unsafe_allow_html=True)
                                st.write(f"**Habit Plane (54.7°) Stress:** {habit_stress:.4f} GPa")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        elif analysis_mode == "Compare Original vs Interpolated":
                            # Comparison analysis
                            st.info("🔄 Comparing original vs interpolated solutions")
                            
                            # Generate interpolated sweep
                            sweep_result = st.session_state.interpolator.create_orientation_sweep(
                                st.session_state.solutions,
                                target_params,
                                comparison_range,
                                n_comparison_points,
                                region_key,
                                stress_component,
                                stress_type
                            )
                            
                            # Get original solutions
                            original_stresses, original_angles = st.session_state.original_analyzer.create_original_sweep_matrix(
                                st.session_state.solutions,
                                comparison_range,
                                n_comparison_points,
                                region_key,
                                stress_component,
                                stress_type
                            )
                            
                            if sweep_result and len(original_stresses) > 0:
                                st.session_state.comparison_data = {
                                    'interpolated': sweep_result,
                                    'original': {
                                        'stresses': original_stresses,
                                        'angles': original_angles
                                    }
                                }
                                
                                st.success("✅ Generated comparison data")
                                
                                # Calculate comparison metrics
                                with st.expander("📊 Comparison Metrics", expanded=True):
                                    # Interpolate original to same grid
                                    interp_stresses = sweep_result['stresses']
                                    interp_angles = sweep_result['angles']
                                    
                                    # Filter valid original data
                                    valid_mask = ~np.isnan(original_stresses)
                                    if np.any(valid_mask):
                                        valid_orig = original_stresses[valid_mask]
                                        valid_angles = original_angles[valid_mask]
                                        
                                        # Interpolate original to interpolated grid
                                        orig_on_interp_grid = np.interp(
                                            interp_angles,
                                            valid_angles,
                                            valid_orig
                                        )
                                        
                                        # Calculate metrics
                                        mae = np.mean(np.abs(orig_on_interp_grid - interp_stresses))
                                        rmse = np.sqrt(np.mean((orig_on_interp_grid - interp_stresses) ** 2))
                                        r2 = 1 - np.sum((interp_stresses - orig_on_interp_grid) ** 2) / (
                                            np.sum((orig_on_interp_grid - np.mean(orig_on_interp_grid)) ** 2) + 1e-10
                                        )
                                        
                                        col_met1, col_met2, col_met3 = st.columns(3)
                                        with col_met1:
                                            st.metric("MAE", f"{mae:.4f} GPa")
                                        with col_met2:
                                            st.metric("RMSE", f"{rmse:.4f} GPa")
                                        with col_met3:
                                            st.metric("R² Score", f"{r2:.4f}")
                                        
                                        st.write("**Interpretation:**")
                                        if r2 > 0.9:
                                            st.success("Excellent agreement between original and interpolated")
                                        elif r2 > 0.7:
                                            st.info("Good agreement between original and interpolated")
                                        else:
                                            st.warning("Moderate agreement - consider adjusting interpolation parameters")
                                    else:
                                        st.warning("Insufficient original data for comparison")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.exception(e)
            
            # Generate visualizations based on analysis mode
            if analysis_mode == "Precise Single Orientation" and 'single_result' in st.session_state:
                st.markdown('<h3 class="sub-header">📊 Visualization</h3>', unsafe_allow_html=True)
                
                result = st.session_state.single_result
                
                if viz_type in ["Radar", "Both"]:
                    # Create a simple radar chart showing the single point
                    angles = np.array([result['target_angle_deg']])
                    stresses = np.array([result['region_stress']])
                    
                    # Add neighboring points for context
                    context_angles = np.array([angles[0] - 5, angles[0], angles[0] + 5])
                    context_stresses = np.array([0, result['region_stress'], 0])
                    
                    fig_radar = st.session_state.visualizer.create_enhanced_plotly_radar(
                        context_stresses, context_angles,
                        f"{stress_component} at {result['target_angle_deg']:.2f}°",
                        title=f"Precise Orientation Analysis: {region_type_display}",
                        show_habit_plane=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Display detailed information
                with st.expander("📋 Detailed Analysis", expanded=False):
                    st.write(f"**Target Parameters:**")
                    st.json(result['target_params'])
                    
                    st.write(f"**Analysis Details:**")
                    st.write(f"- Region: {region_type_display}")
                    st.write(f"- Stress Component: {stress_component}")
                    st.write(f"- Analysis Type: {stress_type}")
                    st.write(f"- Spatial Sigma: {result['spatial_sigma']}")
                    st.write(f"- Number of Sources: {result['num_sources']}")
                    
                    # Export options
                    st.markdown("#### 📤 Export Results")
                    if st.button("💾 Export Single Point Analysis", use_container_width=True):
                        metadata = {
                            'defect_type': defect_type,
                            'shape': shape,
                            'eps0': eps0,
                            'kappa': kappa,
                            'target_angle': result['target_angle_deg'],
                            'region_type': region_key,
                            'stress_component': stress_component,
                            'stress_type': stress_type,
                            'spatial_sigma': spatial_sigma,
                            'attention_heads': attention_heads
                        }
                        
                        export_data = st.session_state.results_manager.prepare_orientation_sweep_data(
                            {'angles': [result['target_angle_deg']], 
                             'stresses': [result['region_stress']],
                             'region_type': region_key,
                             'stress_component': stress_component,
                             'stress_type': stress_type},
                            metadata=metadata
                        )
                        
                        json_str = json.dumps(export_data, indent=2)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📥 Download JSON",
                            data=json_str,
                            file_name=f"single_point_{result['target_angle_deg']:.1f}deg_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
            
            elif analysis_mode == "Orientation Sweep" and 'sweep_result' in st.session_state:
                st.markdown('<h3 class="sub-header">📊 Sweep Visualization</h3>', unsafe_allow_html=True)
                
                sweep = st.session_state.sweep_result
                
                # Line plot of stress vs orientation
                fig_line, ax_line = plt.subplots(figsize=(12, 6))
                ax_line.plot(sweep['angles'], sweep['stresses'], 'b-', linewidth=3, label='Interpolated')
                
                # Add original data if available
                if 'original_sweep' in st.session_state:
                    orig = st.session_state.original_sweep
                    valid_mask = ~np.isnan(orig['stresses'])
                    if np.any(valid_mask):
                        ax_line.scatter(orig['angles'][valid_mask], orig['stresses'][valid_mask],
                                       color='red', s=50, label='Original Solutions', zorder=5)
                
                ax_line.axvline(54.7, color='green', linestyle='--', linewidth=2, 
                               label='Habit Plane (54.7°)', alpha=0.7)
                
                ax_line.set_xlabel('Orientation (°)', fontsize=12, fontweight='bold')
                ax_line.set_ylabel(f'{stress_component.replace("_", " ").title()} Stress (GPa)', 
                                  fontsize=12, fontweight='bold')
                ax_line.set_title(f'Orientation Sweep: {region_type_display}', 
                                 fontsize=14, fontweight='bold', pad=20)
                ax_line.grid(True, alpha=0.3)
                ax_line.legend(fontsize=11)
                ax_line.set_xlim([sweep['angles'][0], sweep['angles'][-1]])
                
                st.pyplot(fig_line)
                plt.close(fig_line)
                
                # Generate sunburst and radar charts
                if viz_type in ["Sunburst", "Both"]:
                    st.markdown("#### 🌅 Sunburst Visualization")
                    
                    # Create sunburst for sweep (single time point)
                    fig_sunburst = st.session_state.visualizer.create_enhanced_plotly_sunburst(
                        sweep['stresses'],
                        np.zeros(1),  # Single time point
                        sweep['angles'],
                        title=f"Orientation Sweep: {region_type_display} - {stress_component}",
                        cmap=cmap,
                        is_time_series=False
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                
                if viz_type in ["Radar", "Both"]:
                    st.markdown("#### 📡 Radar Visualization")
                    
                    fig_radar = st.session_state.visualizer.create_enhanced_plotly_radar(
                        sweep['stresses'], sweep['angles'],
                        f"{stress_component} - {region_type_display}",
                        title=f"Radar Chart: {region_type_display} Stress vs Orientation",
                        show_habit_plane=True
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Export options
                with st.expander("📤 Export Sweep Results", expanded=False):
                    metadata = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa,
                        'region_type': region_key,
                        'stress_component': stress_component,
                        'stress_type': stress_type,
                        'spatial_sigma': spatial_sigma,
                        'attention_heads': attention_heads,
                        'angle_range': list(sweep['angle_range']),
                        'n_points': sweep['n_points']
                    }
                    
                    export_data = st.session_state.results_manager.prepare_orientation_sweep_data(
                        sweep,
                        st.session_state.original_sweep if 'original_sweep' in st.session_state else None,
                        metadata
                    )
                    
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        # JSON export
                        json_str = json.dumps(export_data, indent=2)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            "📥 Download JSON",
                            data=json_str,
                            file_name=f"orientation_sweep_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_exp2:
                        # ZIP archive
                        zip_buffer = st.session_state.results_manager.create_orientation_sweep_archive(
                            sweep,
                            st.session_state.original_sweep if 'original_sweep' in st.session_state else None,
                            metadata
                        )
                        
                        st.download_button(
                            "📦 Download Complete Archive",
                            data=zip_buffer.getvalue(),
                            file_name=f"orientation_sweep_archive_{timestamp}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
            
            elif analysis_mode == "Compare Original vs Interpolated" and 'comparison_data' in st.session_state:
                st.markdown('<h3 class="sub-header">📊 Comparison Visualization</h3>', unsafe_allow_html=True)
                
                comp_data = st.session_state.comparison_data
                interpolated = comp_data['interpolated']
                original = comp_data['original']
                
                # Create comparison plot
                fig_comp, ax_comp = plt.subplots(figsize=(14, 7))
                
                # Plot interpolated curve
                ax_comp.plot(interpolated['angles'], interpolated['stresses'], 
                           'b-', linewidth=3, label='Interpolated', alpha=0.8)
                
                # Plot original points
                valid_mask = ~np.isnan(original['stresses'])
                if np.any(valid_mask):
                    ax_comp.scatter(original['angles'][valid_mask], original['stresses'][valid_mask],
                                   color='red', s=80, label='Original Solutions', 
                                   edgecolors='black', linewidth=1.5, zorder=5)
                
                ax_comp.axvline(54.7, color='green', linestyle='--', linewidth=3,
                               label='Habit Plane (54.7°)', alpha=0.6)
                
                ax_comp.set_xlabel('Orientation (°)', fontsize=14, fontweight='bold')
                ax_comp.set_ylabel(f'{stress_component.replace("_", " ").title()} Stress (GPa)', 
                                  fontsize=14, fontweight='bold')
                ax_comp.set_title(f'Comparison: Original vs Interpolated - {region_type_display}', 
                                 fontsize=16, fontweight='bold', pad=20)
                ax_comp.grid(True, alpha=0.3, linestyle='--')
                ax_comp.legend(fontsize=12, loc='upper right')
                ax_comp.set_xlim([interpolated['angles'][0], interpolated['angles'][-1]])
                
                st.pyplot(fig_comp)
                plt.close(fig_comp)
                
                # Create radar comparison
                if viz_type in ["Radar", "Both", "Comparison"]:
                    st.markdown("#### 📡 Radar Comparison")
                    
                    # Interpolate original to same grid for radar
                    valid_mask = ~np.isnan(original['stresses'])
                    if np.any(valid_mask):
                        orig_on_grid = np.interp(
                            interpolated['angles'],
                            original['angles'][valid_mask],
                            original['stresses'][valid_mask]
                        )
                        
                        fig_radar_comp = st.session_state.visualizer.create_comparison_radar(
                            orig_on_grid, interpolated['stresses'],
                            interpolated['angles'],
                            title=f"Radar Comparison: {region_type_display} - {stress_component}"
                        )
                        st.plotly_chart(fig_radar_comp, use_container_width=True)
    
    with col_main2:
        st.markdown('<h2 class="sub-header">📈 Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Region information cards
        st.markdown(f'<div class="region-card {region_key}-region">', unsafe_allow_html=True)
        st.markdown(f"### {region_type_display}")
        
        if region_key == 'defect':
            st.write("**η > 0.6** - High defect concentration")
            st.write("• Stress concentration in defect cores")
            st.write("• Critical for defect initiation")
            st.write("• High sensitivity to orientation")
        elif region_key == 'interface':
            st.write("**0.4 ≤ η ≤ 0.6** - Interface region")
            st.write("• Stress gradients at interfaces")
            st.write("• Defect propagation path")
            st.write("• Transition zone effects")
        else:  # bulk
            st.write("**η < 0.4** - Pure Ag bulk")
            st.write("• Stress propagation in matrix")
            st.write("• Far-field stress effects")
            st.write("• Material response benchmark")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick metrics
        if 'solutions' in st.session_state and st.session_state.solutions:
            st.markdown("#### 📊 Loaded Solutions")
            st.metric("Total Solutions", len(st.session_state.solutions))
            
            # Count by defect type
            defect_counts = {}
            for sol in st.session_state.solutions:
                d_type = sol.get('params', {}).get('defect_type', 'Unknown')
                defect_counts[d_type] = defect_counts.get(d_type, 0) + 1
            
            for d_type, count in defect_counts.items():
                st.write(f"- {d_type}: {count}")
            
            # Orientation statistics
            orientations = st.session_state.original_analyzer.create_orientation_distribution(
                st.session_state.solutions
            )
            if len(orientations) > 0:
                st.metric("Avg Orientation", f"{np.mean(orientations):.1f}°")
                st.metric("Orientation Range", f"{np.min(orientations):.1f}° - {np.max(orientations):.1f}°")
        
        # Analysis mode specific dashboard
        if 'single_result' in st.session_state:
            result = st.session_state.single_result
            st.markdown("#### 🎯 Single Point Results")
            st.metric("Target Angle", f"{result['target_angle_deg']:.2f}°")
            st.metric("Region Stress", f"{result['region_stress']:.4f} GPa")
            st.metric("Sources Used", result['num_sources'])
            
            # Attention weights summary
            weights = result['attention_weights']
            if len(weights) > 0:
                st.write("**Attention Summary:**")
                st.write(f"- Max weight: {np.max(weights):.3f}")
                st.write(f"- Min weight: {np.min(weights):.3f}")
                st.write(f"- Entropy: {-np.sum(weights * np.log(weights + 1e-10)):.3f}")
        
        elif 'sweep_result' in st.session_state:
            sweep = st.session_state.sweep_result
            st.markdown("#### 🌐 Sweep Results")
            st.metric("Angle Range", f"{sweep['angle_range'][0]:.1f}° - {sweep['angle_range'][1]:.1f}°")
            st.metric("Points", sweep['n_points'])
            
            # Habit plane stress
            habit_idx = np.argmin(np.abs(sweep['angles'] - 54.7))
            habit_stress = sweep['stresses'][habit_idx]
            st.metric("Habit Plane Stress", f"{habit_stress:.4f} GPa")
            
            # Find min/max
            min_idx = np.nanargmin(sweep['stresses'])
            max_idx = np.nanargmax(sweep['stresses'])
            st.write(f"**Min:** {sweep['angles'][min_idx]:.1f}° ({sweep['stresses'][min_idx]:.4f} GPa)")
            st.write(f"**Max:** {sweep['angles'][max_idx]:.1f}° ({sweep['stresses'][max_idx]:.4f} GPa)")
        
        elif 'comparison_data' in st.session_state:
            comp_data = st.session_state.comparison_data
            st.markdown("#### 🔄 Comparison Metrics")
            
            interpolated = comp_data['interpolated']['stresses']
            original = comp_data['original']['stresses']
            valid_mask = ~np.isnan(original)
            
            if np.any(valid_mask):
                valid_orig = original[valid_mask]
                valid_angles = comp_data['original']['angles'][valid_mask]
                
                # Interpolate to common grid
                orig_on_grid = np.interp(
                    comp_data['interpolated']['angles'],
                    valid_angles,
                    valid_orig
                )
                
                mae = np.mean(np.abs(orig_on_grid - interpolated))
                rmse = np.sqrt(np.mean((orig_on_grid - interpolated) ** 2))
                
                st.metric("MAE", f"{mae:.4f} GPa")
                st.metric("RMSE", f"{rmse:.4f} GPa")
                
                # Quality assessment
                if rmse < 0.1:
                    st.success("✅ Excellent agreement")
                elif rmse < 0.3:
                    st.info("⚡ Good agreement")
                else:
                    st.warning("⚠️ Consider adjusting parameters")
        
        # Method explanation
        with st.expander("🧠 Method Details", expanded=False):
            st.write("""
            **Attention-Based Spatial Interpolation:**
            
            1. **Parameter Encoding:**
               - 12-dimensional parameter vectors
               - One-hot encoding for categorical parameters
               - Normalized continuous parameters
            
            2. **Attention Mechanism:**
               - Transformer-inspired multi-head attention
               - Learns relationships between parameters
               - Dynamic weight assignment
            
            3. **Spatial Regularization:**
               - Euclidean distance in parameter space
               - Gaussian kernel: exp(-0.5 * (distance/σ)²)
               - Prevents over-reliance on distant sources
            
            4. **Weight Combination:**
               - 70% attention weights + 30% spatial weights
               - Normalized to sum to 1
               - Ensures physically meaningful interpolation
            
            **Ag FCC Twin Specific:**
            - Habit plane at 54.7°
            - {111} crystal planes
            - Coherent twin boundaries
            - Orientation-dependent stress fields
            """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
