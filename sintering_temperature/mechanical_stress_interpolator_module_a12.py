import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
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

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# Color schemes
STRESS_CMAP = LinearSegmentedColormap.from_list(
    'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
)
SUNBURST_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'rainbow']

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
def find_interface_stress_numba(stress_field, eta_field, threshold=0.1):
    """Find maximum stress at material interface (where 0 < eta < 1)"""
    height, width = stress_field.shape
    max_stress = 0.0
    
    for i in range(height):
        for j in range(width):
            # Check if this is an interface point (eta between threshold and 1-threshold)
            if eta_field[i, j] > threshold and eta_field[i, j] < (1.0 - threshold):
                stress_val = abs(stress_field[i, j])
                if stress_val > max_stress:
                    max_stress = stress_val
    
    return max_stress

# =============================================
# FIXED ENHANCED NUMERICAL SOLUTIONS LOADER
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
# UPDATED ATTENTION INTERPOLATOR WITH INTERFACE STRESS
# =============================================
class EnhancedAttentionInterpolator(nn.Module):
    """Enhanced attention-based interpolator with interface stress analysis"""
    
    def __init__(self, sigma=0.3, use_numba=True, interface_threshold=0.1):
        super().__init__()
        self.sigma = sigma
        self.use_numba = use_numba
        self.interface_threshold = interface_threshold  # Threshold for interface detection (0 < eta < 1-threshold)
        
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
    
    def extract_interface_stress(self, eta, stress_fields):
        """Extract maximum stress at material interface (where 0 < eta < 1)"""
        if eta is None or not isinstance(eta, np.ndarray):
            return {
                'von_mises': 0.0,
                'sigma_hydro': 0.0,
                'sigma_mag': 0.0
            }
        
        # Find interface region (where eta is between threshold values)
        interface_mask = (eta > self.interface_threshold) & (eta < (1.0 - self.interface_threshold))
        
        if not np.any(interface_mask):
            # If no interface found, use a broader definition
            interface_mask = (eta > 0.001) & (eta < 0.999)
        
        results = {}
        
        for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
            stress_data = stress_fields.get(key, np.zeros_like(eta))
            
            if np.any(interface_mask):
                # Extract stress values at interface
                interface_stress = stress_data[interface_mask]
                
                # Take absolute value for maximum magnitude
                abs_stress = np.abs(interface_stress)
                
                # Get maximum stress at interface
                max_stress = np.max(abs_stress) if len(abs_stress) > 0 else 0.0
                
                # Alternative: take 95th percentile to avoid outliers
                # max_stress = np.percentile(abs_stress, 95) if len(abs_stress) > 5 else np.max(abs_stress)
            else:
                max_stress = 0.0
            
            results[key] = float(max_stress)
        
        return results
    
    def interpolate(self, sources, target_params):
        """Interpolate stress field and extract interface stresses"""
        
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
        source_interface_stresses = []  # Store interface stresses instead of full fields
        
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
                
                # Extract interface stress
                interface_stress = self.extract_interface_stress(eta, stress_fields)
                source_interface_stresses.append(interface_stress)
            else:
                # No history, use zero stresses
                source_interface_stresses.append({
                    'von_mises': 0.0,
                    'sigma_hydro': 0.0,
                    'sigma_mag': 0.0
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
        
        # Weighted combination of interface stresses
        result_stress = {}
        for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
            weighted_sum = 0.0
            for w, stress_dict in zip(weights, source_interface_stresses):
                weighted_sum += w * stress_dict.get(key, 0.0)
            result_stress[key] = weighted_sum
        
        return {
            'interface_stress': result_stress,  # Changed from stress_fields
            'attention_weights': weights,
            'target_params': target_params,
            'num_valid_sources': len(valid_sources),
            'source_interface_stresses': source_interface_stresses  # For debugging/analysis
        }

# =============================================
# ENHANCED VISUALIZATION MANAGER
# =============================================
class EnhancedVisualizationManager:
    """Enhanced visualization manager with multiple plot types"""
    
    def __init__(self):
        self.stress_cmap = STRESS_CMAP
        self.sunburst_cmaps = SUNBURST_CMAPS
        
    def create_stress_field_plot(self, stress_data, title, component_name,
                                extent=None, vmin=None, vmax=None,
                                include_contour=True, include_colorbar=True):
        """Create matplotlib plot for stress field"""
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        if extent is None:
            extent = [-64, 64, -64, 64]
        
        if vmin is None:
            vmin = np.nanmin(stress_data)
        if vmax is None:
            vmax = np.nanmax(stress_data)
        
        # Create heatmap
        im = ax.imshow(stress_data,
                      cmap=self.stress_cmap,
                      extent=extent,
                      origin='lower',
                      aspect='equal',
                      vmin=vmin,
                      vmax=vmax)
        
        # Add contour lines
        if include_contour and not np.all(stress_data == stress_data[0, 0]):
            try:
                X, Y = np.meshgrid(np.linspace(extent[0], extent[1], stress_data.shape[1]),
                                  np.linspace(extent[2], extent[3], stress_data.shape[0]))
                
                levels = np.linspace(vmin, vmax, 12)
                contour = ax.contour(X, Y, stress_data,
                                    levels=levels,
                                    colors='black',
                                    linewidths=0.5,
                                    alpha=0.7)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
            except:
                pass
        
        # Add colorbar
        if include_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Stress (GPa)', rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel('x (nm)', fontsize=12)
        ax.set_ylabel('y (nm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig
    
    def create_attention_weights_plot(self, weights, source_names=None,
                                     title="Attention Weights Distribution"):
        """Create bar plot for attention weights"""
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        
        if source_names is None:
            source_names = [f'Source {i+1}' for i in range(len(weights))]
        
        # Create bar plot
        x_pos = np.arange(len(weights))
        bars = ax.bar(x_pos, weights,
                     color=plt.cm.viridis(np.linspace(0, 1, len(weights))),
                     edgecolor='black',
                     linewidth=1,
                     alpha=0.8)
        
        # Add value labels on top of bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Source Simulations', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(source_names, rotation=45, ha='right')
        ax.set_ylim([0, max(weights) * 1.3])
        
        # Add horizontal line for average
        avg_weight = np.mean(weights)
        ax.axhline(y=avg_weight, color='red', linestyle='--', alpha=0.7,
                  label=f'Average: {avg_weight:.3f}')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED SUNBURST & RADAR VISUALIZER
# =============================================
class EnhancedSunburstRadarVisualizer:
    """Enhanced sunburst and radar charts with multiple visualization options"""
    
    def __init__(self):
        self.vis_manager = EnhancedVisualizationManager()
    
    def create_sunburst_plot(self, stress_matrix, times, thetas, title, cmap='plasma'):
        """Create polar heatmap (sunburst) visualization"""
        
        # Create polar plot
        theta_deg = np.deg2rad(thetas)
        theta_mesh, time_mesh = np.meshgrid(theta_deg, times)
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'}, dpi=150)
        
        # Plot heatmap
        im = ax.pcolormesh(theta_mesh, time_mesh, stress_matrix, 
                          cmap=cmap, shading='auto', alpha=0.8)
        
        # Customize
        ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Orientation (degrees)', labelpad=20, fontsize=12)
        ax.set_ylabel('Time (s)', labelpad=20, fontsize=12)
        ax.set_xticks(theta_deg)
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], fontsize=10)
        
        # Add radial grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.1)
        cbar.set_label('Interface Stress (GPa)', rotation=270, labelpad=25, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_plotly_sunburst(self, stress_matrix, times, thetas, title, cmap='Plasma'):
        """Interactive sunburst with Plotly - enhanced version"""
        
        # Prepare data for polar scatter
        theta_deg = np.deg2rad(thetas)
        theta_grid, time_grid = np.meshgrid(theta_deg, times)
        
        # Flatten the arrays for scatter plot
        r_flat = time_grid.flatten()
        theta_flat = np.rad2deg(theta_grid).flatten()
        stress_flat = stress_matrix.flatten()
        
        # Create the plotly figure
        fig = go.Figure()
        
        # Add scatter polar trace
        fig.add_trace(go.Scatterpolar(
            r=r_flat,
            theta=theta_flat,
            mode='markers',
            marker=dict(
                size=8,
                color=stress_flat,
                colorscale=cmap,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Interface Stress (GPa)", font=dict(size=12)),
                    tickfont=dict(size=10),
                    thickness=20,
                    len=0.75,
                    x=1.1,
                    xpad=10
                ),
                line=dict(width=0.5, color='white'),
                opacity=0.8
            ),
            hovertemplate=(
                '<b>Time</b>: %{r:.1f}s<br>' +
                '<b>Orientation</b>: %{theta:.1f}°<br>' +
                '<b>Interface Stress</b>: %{marker.color:.3f} GPa<br>' +
                '<extra></extra>'
            ),
            name='Interface Stress Distribution'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial, sans-serif", color='black'),
                x=0.5,
                xanchor='center'
            ),
            polar=dict(
                radialaxis=dict(
                    title=dict(
                        text="Time (s)",
                        font=dict(size=14, color='black')
                    ),
                    gridcolor="lightgray",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    showline=True,
                    tickfont=dict(size=11, color='black'),
                    tickformat='.1f',
                    range=[0, max(times) * 1.05]
                ),
                angularaxis=dict(
                    gridcolor="lightgray",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    rotation=90,
                    direction="clockwise",
                    tickfont=dict(size=11, color='black'),
                    tickmode='array',
                    tickvals=list(range(0, 360, 45)),
                    ticktext=[f'{i}°' for i in range(0, 360, 45)]
                ),
                bgcolor="rgba(245, 245, 245, 0.8)",
                sector=[0, 360]
            ),
            width=800,
            height=650,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=80, r=150, t=80, b=80),  # Extra right margin for colorbar
            font=dict(family="Arial, sans-serif")
        )
        
        # Add grid lines
        fig.update_polars(
            radialaxis_gridcolor="rgba(150, 150, 150, 0.3)",
            angularaxis_gridcolor="rgba(150, 150, 150, 0.3)"
        )
        
        return fig
    
    def create_radar_plot(self, stress_values, thetas, component_name, time_point):
        """Enhanced radar/spider chart for interface stress"""
        
        # Close the loop
        angles = np.linspace(0, 2*np.pi, len(thetas), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        values = np.concatenate([stress_values, [stress_values[0]]])
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'}, dpi=150)
        
        # Plot with gradient fill
        ax.plot(angles, values, 'o-', linewidth=3, markersize=8, 
                color='steelblue', markerfacecolor='white', markeredgewidth=2)
        
        # Gradient fill
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], fontsize=11)
        
        # Title with metrics
        mean_val = np.mean(stress_values)
        max_val = np.max(stress_values)
        ax.set_title(f'{component_name} Interface Stress\nt={time_point:.1f}s | Mean: {mean_val:.2f} GPa | Max: {max_val:.2f} GPa',
                    fontsize=14, fontweight='bold', pad=25)
        
        # Add value annotations
        for angle, value in zip(angles[:-1], stress_values):
            x_pos = angle
            y_pos = value * 1.05
            ax.text(x_pos, y_pos, f'{value:.2f}', 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_comparison_radar(self, stress_matrices, thetas, component_names, 
                               time_point, title="Interface Stress Comparison"):
        """Create radar plot comparing multiple stress components at interface"""
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'}, dpi=150)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(stress_matrices)))
        
        for idx, (stress_matrix, comp_name, color) in enumerate(zip(stress_matrices, component_names, colors)):
            # Get values for the selected time point
            time_idx = np.argmin(np.abs(np.arange(len(stress_matrix)) - time_point))
            values = stress_matrix[time_idx, :]
            
            # Close the loop
            angles = np.linspace(0, 2*np.pi, len(thetas), endpoint=False)
            angles = np.concatenate([angles, [angles[0]]])
            values_loop = np.concatenate([values, [values[0]]])
            
            # Plot
            ax.plot(angles, values_loop, 'o-', linewidth=2, markersize=6,
                    color=color, label=comp_name, alpha=0.8)
            ax.fill(angles, values_loop, alpha=0.15, color=color)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas], fontsize=11)
        ax.set_title(f'{title}\nt={time_point:.1f}s', fontsize=16, fontweight='bold', pad=25)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# =============================================
# RESULTS MANAGER
# =============================================
class ResultsManager:
    """Manager for saving and exporting results"""
    
    @staticmethod
    def prepare_prediction_data(prediction_results, source_simulations, target_params):
        """Prepare prediction data for export"""
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_sources': len(source_simulations),
                'target_params': target_params,
                'software_version': '2.0.0'
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
                'theta': float(params.get('theta', 0))
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
            zip_file.writestr('interface_stress_matrix.npy', stress_buffer.getvalue())
            
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
            
            # Save CSV version
            csv_data = []
            for t_idx, time_val in enumerate(times):
                for theta_idx, theta_val in enumerate(thetas):
                    csv_data.append({
                        'time_s': time_val,
                        'orientation_deg': theta_val,
                        'interface_stress_gpa': stress_matrix[t_idx, theta_idx]
                    })
            
            df = pd.DataFrame(csv_data)
            csv_str = df.to_csv(index=False)
            zip_file.writestr('interface_stress_data.csv', csv_str)
            
            # Add README
            readme = f"""# Interface Stress Interpolation Results
Generated: {datetime.now().isoformat()}

Files:
1. interface_stress_matrix.npy - 2D stress matrix (time × orientation)
2. times.npy - Time points array
3. thetas.npy - Orientation angles array
4. metadata.json - Simulation metadata
5. interface_stress_data.csv - Tabular data for analysis

Parameters:
- Defect Type: {metadata.get('defect_type', 'Unknown')}
- Orientation Range: {metadata.get('theta_range', 'Unknown')}
- ε*: {metadata.get('eps0', 'Unknown')}
- κ: {metadata.get('kappa', 'Unknown')}
- Stress Type: Maximum at Material Interface (0 < eta < 1)
"""
            zip_file.writestr('README.txt', readme)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# MAIN APPLICATION WITH INTERFACE STRESS ANALYSIS
# =============================================
def main():
    st.set_page_config(
        page_title="Interface Stress Interpolation Visualizer",
        layout="wide",
        page_icon="🔬"
    )
    
    st.title("🔬 Interface Stress Field Interpolation")
    st.markdown("""
    This app analyzes **maximum stress at material interfaces** (where 0 < eta < 1) 
    instead of center points. More physically meaningful for defect analysis!
    """)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedAttentionInterpolator(sigma=0.3, use_numba=True, interface_threshold=0.1)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSunburstRadarVisualizer()
    if 'vis_manager' not in st.session_state:
        st.session_state.vis_manager = EnhancedVisualizationManager()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Performance settings
        with st.expander("🚀 Performance Settings", expanded=True):
            use_numba = st.checkbox("Use Numba Acceleration", value=True)
            use_cache = st.checkbox("Use File Cache", value=True)
            
            # PyTorch loading method selection
            pt_loading_method = st.radio(
                "PyTorch Load Method",
                ["Safe (with fix)", "Unsafe (weights_only=False)"],
                index=0,
                help="Safe loading with numpy scalar fix. Unsafe loading may be needed for older files."
            )
            
            # Interface threshold
            interface_threshold = st.slider(
                "Interface Threshold (eta)",
                0.001, 0.5, 0.1, 0.01,
                help="Define interface as threshold < eta < 1-threshold"
            )
            
            if interface_threshold != st.session_state.interpolator.interface_threshold:
                st.session_state.interpolator.interface_threshold = interface_threshold
                st.success("Interface threshold updated!")
            
            if use_numba != st.session_state.interpolator.use_numba:
                st.session_state.interpolator.use_numba = use_numba
                st.success("Numba setting updated!")
        
        # Load solutions
        st.subheader("📂 Load Solutions")
        
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load Solutions", use_container_width=True, 
                        help="Load all solutions from numerical_solutions directory"):
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
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions in {load_time:.2f}s")
                    else:
                        st.warning("No solutions found or loaded")
        
        with col_load2:
            if st.button("🗑️ Clear Cache", use_container_width=True, 
                        help="Clear file cache"):
                st.session_state.loader.cache.clear()
                st.success("Cache cleared!")
        
        # Show solution info
        if st.session_state.solutions:
            st.subheader("📋 Loaded Solutions")
            with st.expander(f"View {len(st.session_state.solutions)} solutions"):
                for i, sol in enumerate(st.session_state.solutions[:10]):  # Limit to 10
                    params = sol.get('params', {})
                    st.caption(f"{i+1}. {sol.get('filename', 'Unknown')}")
                    st.write(f"   • Type: {params.get('defect_type', 'Unknown')}")
                    st.write(f"   • θ: {np.rad2deg(params.get('theta', 0)):.1f}°")
                    st.write(f"   • ε*: {params.get('eps0', 0):.3f}")
                    st.divider()
                
                if len(st.session_state.solutions) > 10:
                    st.info(f"... and {len(st.session_state.solutions) - 10} more")
        
        st.divider()
        
        # Interpolation settings
        st.subheader("🎯 Target Parameters")
        
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
        st.subheader("🌐 Orientation Sweep")
        
        theta_min = st.slider("Min Angle (°)", 0, 360, 0, 5,
                             help="Minimum orientation angle")
        theta_max = st.slider("Max Angle (°)", 0, 360, 360, 5,
                             help="Maximum orientation angle")
        theta_step = st.slider("Step Size (°)", 5, 45, 15, 5,
                              help="Angle step size")
        
        # Time settings
        st.subheader("⏱️ Time Settings")
        
        n_times = st.slider("Time Points", 10, 200, 50, 10,
                           help="Number of time points to simulate")
        max_time = st.slider("Max Time (s)", 50, 500, 200, 10,
                            help="Maximum simulation time")
        
        # Visualization settings
        st.subheader("🎨 Visualization")
        
        stress_component = st.selectbox(
            "Stress Component",
            ["von_mises", "sigma_hydro", "sigma_mag"],
            index=0,
            help="Select which stress component to visualize"
        )
        
        viz_type = st.radio("Chart Type", ["Sunburst", "Radar", "Both", "Comparison"], 
                           help="Select visualization type")
        cmap = st.selectbox("Color Map", SUNBURST_CMAPS, index=1)
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            use_plotly = st.checkbox("Use Plotly (Interactive)", value=True)
        with col_viz2:
            show_metrics = st.checkbox("Show Metrics", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🚀 Generate Interface Stress Visualizations")
        
        if not st.session_state.solutions:
            st.warning("Please load solutions first using the button in the sidebar.")
            
            # Show directory info
            with st.expander("📁 Directory Information"):
                file_formats = st.session_state.loader.scan_solutions()
                total_files = sum(len(files) for files in file_formats.values())
                
                if total_files > 0:
                    st.info(f"Found {total_files} files in {SOLUTIONS_DIR}:")
                    for fmt, files in file_formats.items():
                        if files:
                            st.write(f"• {fmt.upper()}: {len(files)} files")
                else:
                    st.write(f"Directory: `{SOLUTIONS_DIR}`")
                    st.write("Expected file formats: `.pkl`, `.pt`, `.h5`, `.npz`, `.sql`, `.json`")
                    
                    st.info("""
                    **Expected file structure:**
                    - Place simulation files in the `numerical_solutions/` directory
                    - Each file should contain simulation data with:
                      - `params` dictionary (defect_type, theta, eps0, kappa, shape)
                      - `history` list with (eta, stress_fields) tuples
                    """)
        else:
            if st.button("✨ Generate Interface Stress Charts", type="primary", use_container_width=True):
                with st.spinner("Generating orientation sweep for interface stress..."):
                    try:
                        # Generate theta range
                        thetas = np.arange(theta_min, theta_max + theta_step, theta_step)
                        theta_rad = np.deg2rad(thetas)
                        
                        # Generate time points
                        times = np.linspace(0, max_time, n_times)
                        
                        # Generate predictions for each orientation
                        predictions = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, theta in enumerate(theta_rad):
                            status_text.text(f"Processing orientation {i+1}/{len(theta_rad)}...")
                            
                            # Target parameters
                            target_params = {
                                'defect_type': defect_type,
                                'theta': float(theta),
                                'eps0': eps0,
                                'kappa': kappa,
                                'shape': shape
                            }
                            
                            # Interpolate - now returns interface stress
                            result = st.session_state.interpolator.interpolate(
                                st.session_state.solutions, target_params
                            )
                            
                            if result:
                                # Extract interface stress evolution
                                time_evolution = []
                                
                                # Create time evolution using interface stress
                                interface_stress = result['interface_stress'][stress_component]
                                
                                # Create synthetic time evolution based on interface stress
                                for t in times:
                                    # Use interface stress as base, with time-dependent scaling
                                    stress_at_t = interface_stress * (1 - np.exp(-t / 50))
                                    time_evolution.append(stress_at_t)
                                
                                predictions.append(time_evolution)
                            
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
                            
                            # Store metadata
                            st.session_state.metadata = {
                                'defect_type': defect_type,
                                'shape': shape,
                                'eps0': eps0,
                                'kappa': kappa,
                                'sigma': sigma,
                                'interface_threshold': interface_threshold,
                                'stress_component': stress_component,
                                'theta_range': f"{theta_min}-{theta_max}°",
                                'theta_step': theta_step,
                                'n_times': n_times,
                                'max_time': max_time,
                                'generated_at': datetime.now().isoformat(),
                                'analysis_type': 'interface_stress'
                            }
                            
                            st.success(f"✅ Generated {len(thetas)} orientations × {len(times)} time points")
                            st.info(f"**Analysis:** Maximum {stress_component} stress at material interface (η ∈ [{interface_threshold}, {1-interface_threshold}])")
                            
                            # Display results
                            if viz_type in ["Sunburst", "Both", "Comparison"]:
                                st.subheader("🌅 Sunburst Chart - Interface Stress")
                                
                                title = f"{stress_component.replace('_', ' ').title()} Interface Stress - {defect_type}"
                                
                                if use_plotly:
                                    fig = st.session_state.visualizer.create_plotly_sunburst(
                                        stress_matrix, times, thetas,
                                        title=title,
                                        cmap=cmap
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig = st.session_state.visualizer.create_sunburst_plot(
                                        stress_matrix, times, thetas,
                                        title=title,
                                        cmap=cmap
                                    )
                                    st.pyplot(fig)
                                    
                                    # Download button
                                    col_dl1, col_dl2 = st.columns(2)
                                    with col_dl1:
                                        buf = BytesIO()
                                        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download Sunburst PNG",
                                            data=buf.getvalue(),
                                            file_name=f"interface_sunburst_{stress_component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                            mime="image/png",
                                            use_container_width=True
                                        )
                                    with col_dl2:
                                        buf = BytesIO()
                                        fig.savefig(buf, format="pdf", bbox_inches='tight')
                                        st.download_button(
                                            "📥 Download Sunburst PDF",
                                            data=buf.getvalue(),
                                            file_name=f"interface_sunburst_{stress_component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                            
                            if viz_type in ["Radar", "Both", "Comparison"]:
                                st.subheader("📡 Radar Charts - Interface Stress")
                                
                                # Select time point
                                time_idx = st.slider("Select Time Point", 0, len(times)-1, len(times)//2,
                                                    key="radar_time_slider")
                                selected_time = times[time_idx]
                                
                                if viz_type == "Comparison":
                                    # For comparison, we would need multiple stress components
                                    # For now, show single component
                                    fig_radar = st.session_state.visualizer.create_radar_plot(
                                        stress_matrix[time_idx, :], thetas, 
                                        stress_component.replace('_', ' ').title(), 
                                        selected_time
                                    )
                                    st.pyplot(fig_radar)
                                else:
                                    # Create radar for the selected stress component
                                    fig_radar = st.session_state.visualizer.create_radar_plot(
                                        stress_matrix[time_idx, :], thetas, 
                                        stress_component.replace('_', ' ').title(), 
                                        selected_time
                                    )
                                    st.pyplot(fig_radar)
                            
                            # Enhanced Statistics with Numba acceleration
                            if show_metrics:
                                st.subheader("📊 Interface Stress Statistics")
                                
                                # Compute statistics with Numba
                                start_time = time.time()
                                max_val, min_val, mean_val, std_val, p95, p99 = compute_stress_statistics_numba(
                                    stress_matrix
                                )
                                compute_time = time.time() - start_time
                                
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("Max Interface Stress", f"{max_val:.3f} GPa")
                                with col_stat2:
                                    st.metric("Mean Interface Stress", f"{mean_val:.3f} GPa")
                                with col_stat3:
                                    st.metric("Std Dev", f"{std_val:.3f} GPa")
                                with col_stat4:
                                    st.metric("95th %ile", f"{p95:.3f} GPa")
                                
                                st.caption(f"Statistics computed in {compute_time*1000:.2f} ms with Numba acceleration")
                                
                                # Distribution plot
                                fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                                ax_dist.hist(stress_matrix.flatten(), bins=50, 
                                           edgecolor='black', alpha=0.7, color='steelblue')
                                ax_dist.set_xlabel('Interface Stress (GPa)', fontsize=12)
                                ax_dist.set_ylabel('Frequency', fontsize=12)
                                ax_dist.set_title(f'{stress_component.replace("_", " ").title()} Interface Stress Distribution', 
                                                fontsize=14, fontweight='bold')
                                ax_dist.grid(True, alpha=0.3)
                                
                                # Add vertical lines for statistics
                                ax_dist.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                                ax_dist.axvline(p95, color='orange', linestyle=':', label=f'95th %ile: {p95:.3f}')
                                ax_dist.axvline(p99, color='green', linestyle='-.', label=f'99th %ile: {p99:.3f}')
                                ax_dist.legend()
                                
                                st.pyplot(fig_dist)
                            
                            # Enhanced Data export
                            st.subheader("📤 Enhanced Export Options")
                            
                            export_col1, export_col2, export_col3 = st.columns(3)
                            
                            with export_col1:
                                # CSV export
                                if st.button("💾 Export as CSV", use_container_width=True):
                                    export_data = []
                                    for t_idx, time_val in enumerate(times):
                                        for theta_idx, theta_val in enumerate(thetas):
                                            export_data.append({
                                                'time_s': time_val,
                                                'orientation_deg': theta_val,
                                                'interface_stress_gpa': stress_matrix[t_idx, theta_idx]
                                            })
                                    
                                    df = pd.DataFrame(export_data)
                                    csv = df.to_csv(index=False)
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download CSV",
                                        data=csv,
                                        file_name=f"interface_stress_{stress_component}_{timestamp}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            with export_col2:
                                # JSON export
                                if st.button("📊 Export as JSON", use_container_width=True):
                                    export_dict = {
                                        'metadata': st.session_state.metadata,
                                        'times': times.tolist(),
                                        'thetas': thetas.tolist(),
                                        'interface_stress_matrix': stress_matrix.tolist()
                                    }
                                    
                                    json_str = json.dumps(export_dict, indent=2)
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download JSON",
                                        data=json_str,
                                        file_name=f"interface_stress_{stress_component}_{timestamp}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            
                            with export_col3:
                                # ZIP archive export
                                if st.button("📦 Export as ZIP", use_container_width=True):
                                    zip_buffer = st.session_state.results_manager.create_results_archive(
                                        stress_matrix, times, thetas, st.session_state.metadata
                                    )
                                    
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    st.download_button(
                                        "📥 Download ZIP Archive",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"interface_stress_results_{stress_component}_{timestamp}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.subheader("📈 Dashboard")
        
        if 'stress_matrix' in st.session_state:
            stress_matrix = st.session_state.stress_matrix
            
            st.metric("Time Points", len(st.session_state.times))
            st.metric("Orientations", len(st.session_state.thetas))
            
            # Quick statistics
            max_val = np.max(stress_matrix)
            mean_val = np.mean(stress_matrix)
            min_val = np.min(stress_matrix)
            
            st.metric("Max Interface Stress", f"{max_val:.3f} GPa", 
                     delta=f"{(max_val - mean_val):.3f} from mean")
            st.metric("Mean Interface Stress", f"{mean_val:.3f} GPa")
            st.metric("Min Interface Stress", f"{min_val:.3f} GPa")
            
            # Stress range
            stress_range = max_val - min_val
            if stress_range > 0:
                progress_value = (mean_val - min_val) / stress_range
                progress_text = f"Mean position in range: {((mean_val - min_val) / stress_range * 100):.1f}%"
            else:
                progress_value = 0.5
                progress_text = "No stress variation"
            
            st.progress(progress_value, text=progress_text)
            
            # Orientation distribution
            st.subheader("🌐 Orientation Stats")
            
            mean_by_theta = np.mean(stress_matrix, axis=0)
            max_theta_idx = np.argmax(mean_by_theta)
            min_theta_idx = np.argmin(mean_by_theta)
            
            st.write(f"**Max stress at:** {st.session_state.thetas[max_theta_idx]:.0f}°")
            st.write(f"**Min stress at:** {st.session_state.thetas[min_theta_idx]:.0f}°")
            
            # Time evolution
            st.subheader("⏱️ Time Evolution")
            
            mean_by_time = np.mean(stress_matrix, axis=1)
            time_of_max = st.session_state.times[np.argmax(mean_by_time)]
            
            st.write(f"**Peak at:** {time_of_max:.1f}s")
            
            # Quick plot
            fig_quick, ax_quick = plt.subplots(figsize=(4, 3))
            ax_quick.plot(st.session_state.times, mean_by_time, 'b-', linewidth=2)
            ax_quick.fill_between(st.session_state.times, mean_by_time, alpha=0.3, color='blue')
            ax_quick.set_xlabel('Time (s)')
            ax_quick.set_ylabel('Mean Interface Stress (GPa)')
            ax_quick.set_title('Interface Stress Evolution', fontsize=10)
            ax_quick.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_quick)
            
        else:
            st.info("No data generated yet.")
            st.write("Click 'Generate Interface Stress Charts' to analyze material interface stresses.")
            
            # Show analysis explanation
            with st.expander("🔬 About Interface Stress Analysis"):
                st.write("""
                **Why interface stress is more meaningful:**
                1. **Material Interface** (0 < η < 1) is where defects actually occur
                2. **Maximum stress** at interface drives defect propagation
                3. **Center point** may not capture true mechanical behavior
                4. **Comparison across orientations** is more physically relevant
                
                **Analysis method:**
                - Extract maximum absolute stress where η ∈ [threshold, 1-threshold]
                - Compare across different defect orientations
                - Visualize time evolution of interface stress
                """)

# =============================================
# RUN APPLICATION
# =============================================
if __name__ == "__main__":
    main()
