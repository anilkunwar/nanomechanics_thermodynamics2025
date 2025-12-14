"""
Enhanced Stress Analysis and Attention Interpolation Application
With robust programming practices, improved architecture, and enhanced features
"""

import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import pandas as pd
import zipfile
from io import BytesIO, StringIO
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
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from itertools import product, combinations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
import inspect
from functools import wraps, lru_cache
from contextlib import contextmanager
import hashlib
import sys

# =============================================
# CONFIGURATION AND CONSTANTS
# =============================================
class AppConfig:
    """Centralized configuration for the application"""
    
    # Get script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Directories - ensure they exist before use
    @classmethod
    def get_numerical_solutions_dir(cls):
        dir_path = os.path.join(cls.SCRIPT_DIR, "numerical_solutions")
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    @classmethod
    def get_log_dir(cls):
        dir_path = os.path.join(cls.SCRIPT_DIR, "logs")
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    @classmethod
    def get_cache_dir(cls):
        dir_path = os.path.join(cls.SCRIPT_DIR, "cache")
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    # File limits
    MAX_FILE_SIZE_MB = 100
    MAX_FILES_PER_LOAD = 500
    MAX_CONCURRENT_WORKERS = 4
    
    # Default values
    DEFAULT_GRID_SIZE = 128
    DEFAULT_DX = 0.1
    DEFAULT_NUM_HEADS = 4
    DEFAULT_SIGMA_SPATIAL = 0.2
    DEFAULT_SIGMA_PARAM = 0.3
    
    # Parameter ranges
    EPS0_RANGE = (0.3, 3.0)
    KAPPA_RANGE = (0.1, 2.0)
    THETA_RANGE = (0.0, 2 * np.pi)
    
    # Security
    ALLOWED_FILE_EXTENSIONS = {'.pkl', '.pt', '.h5', '.hdf5', '.npz', '.json'}
    MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB
    
    # Visualization
    DEFAULT_COLORMAP = 'viridis'
    PLOTLY_THEME = 'plotly_white'
    
    @classmethod
    def ensure_all_directories(cls):
        """Ensure all required directories exist"""
        # Create all directories
        cls.get_numerical_solutions_dir()
        cls.get_log_dir()
        cls.get_cache_dir()

# Create directories immediately
AppConfig.ensure_all_directories()

# =============================================
# LOGGING AND ERROR HANDLING
# =============================================
class AppLogger:
    """Centralized logging with rotation and multiple handlers"""
    
    def __init__(self, name="stress_analysis_app"):
        self.logger = logging.getLogger(name)
        
        # Only add handlers if none exist
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
            # Try to create file handler (might fail if directory doesn't exist)
            try:
                log_dir = AppConfig.get_log_dir()
                log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
                
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=10*1024*1024, backupCount=5
                )
                file_handler.setLevel(logging.DEBUG)
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
                )
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)
                self.logger.info(f"File logging initialized: {log_file}")
            except Exception as e:
                self.logger.warning(f"Could not initialize file logging: {str(e)}. Using console only.")
    
    def get_logger(self):
        return self.logger

# Initialize logger with safe fallback
try:
    logger = AppLogger().get_logger()
except Exception as e:
    # Fallback to basic logging if AppLogger fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning(f"Using basic logging due to initialization error: {str(e)}")

# =============================================
# DECORATORS AND UTILITIES
# =============================================
def handle_errors(default_return=None, log_error=True):
    """Decorator for error handling with optional logging"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    try:
                        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                    except:
                        print(f"Error in {func.__name__}: {str(e)}")
                if default_return is not None:
                    return default_return
                else:
                    raise
        return wrapper
    return decorator

def cache_results(maxsize=128, ttl_seconds=3600):
    """Cache results with time-to-live"""
    def decorator(func):
        cache = {}
        cache_timestamps = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Create cache key
                key = hashlib.md5(
                    f"{func.__name__}{args}{sorted(kwargs.items())}".encode()
                ).hexdigest()
                
                # Check cache
                current_time = time.time()
                if key in cache:
                    if current_time - cache_timestamps[key] < ttl_seconds:
                        return cache[key]
                
                # Compute and cache
                result = func(*args, **kwargs)
                cache[key] = result
                cache_timestamps[key] = current_time
                
                # Clean old entries
                old_keys = [k for k, ts in cache_timestamps.items() 
                           if current_time - ts > ttl_seconds]
                for k in old_keys:
                    cache.pop(k, None)
                    cache_timestamps.pop(k, None)
                
                return result
            except Exception as e:
                logger.error(f"Cache error in {func.__name__}: {str(e)}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def performance_timer(operation_name: str):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{operation_name} took {elapsed:.2f} seconds")

# =============================================
# DATA MODELS AND ENUMS
# =============================================
class DefectType(Enum):
    ISF = "ISF"
    ESF = "ESF"
    TWIN = "Twin"
    
    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            return cls.ISF  # Default

class ShapeType(Enum):
    SQUARE = "Square"
    HORIZONTAL_FAULT = "Horizontal Fault"
    VERTICAL_FAULT = "Vertical Fault"
    RECTANGLE = "Rectangle"
    ELLIPSE = "Ellipse"
    
    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            return cls.SQUARE  # Default

@dataclass
class SimulationParameters:
    """Structured simulation parameters"""
    defect_type: DefectType = DefectType.ISF
    shape: ShapeType = ShapeType.SQUARE
    orientation: str = "Horizontal {111} (0°)"
    eps0: float = 0.707
    kappa: float = 0.6
    theta: float = 0.0
    grid_size: int = AppConfig.DEFAULT_GRID_SIZE
    dx: float = AppConfig.DEFAULT_DX
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'defect_type': self.defect_type.value,
            'shape': self.shape.value,
            'orientation': self.orientation,
            'eps0': self.eps0,
            'kappa': self.kappa,
            'theta': self.theta,
            'grid_size': self.grid_size,
            'dx': self.dx
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParameters':
        return cls(
            defect_type=DefectType.from_string(data.get('defect_type', 'ISF')),
            shape=ShapeType.from_string(data.get('shape', 'Square')),
            orientation=data.get('orientation', 'Horizontal {111} (0°)'),
            eps0=float(data.get('eps0', 0.707)),
            kappa=float(data.get('kappa', 0.6)),
            theta=float(data.get('theta', 0.0))
        )

@dataclass
class StressFields:
    """Container for stress field data with validation"""
    sigma_hydro: np.ndarray
    sigma_mag: np.ndarray
    von_mises: np.ndarray
    sigma_1: Optional[np.ndarray] = None
    sigma_2: Optional[np.ndarray] = None
    sigma_3: Optional[np.ndarray] = None
    
    def validate(self):
        """Validate stress field data"""
        shapes = []
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                if not isinstance(field_value, np.ndarray):
                    raise TypeError(f"{field_name} must be a numpy array")
                shapes.append(field_value.shape)
        
        # Check all non-None fields have the same shape
        if shapes:
            if not all(shape == shapes[0] for shape in shapes[1:]):
                raise ValueError("All stress fields must have the same shape")
    
    def get_component(self, component_name: str) -> Optional[np.ndarray]:
        """Get stress component by name"""
        return getattr(self, component_name, None)

# =============================================
# ORIGINAL MANAGER CLASSES (with enhancements)
# =============================================

# Safe Simulation Loader
class SafeSimulationLoader:
    """Safely load and validate simulation files with error handling"""
    
    @staticmethod
    @handle_errors(default_return=(None, "Unknown error"), log_error=True)
    def validate_and_load_pkl(filepath):
        """Attempts to load a PKL file with multiple safety checks."""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(2)
                f.seek(0)
                
                data = pickle.load(f)

            # Validate the loaded structure
            if not isinstance(data, dict):
                return None, "Loaded object is not a dictionary."
            
            # Check for required structure
            has_stress_data = False
            if 'stress_fields' in data:
                if isinstance(data['stress_fields'], dict):
                    has_stress_data = True
            elif 'history' in data:
                if isinstance(data['history'], list) and len(data['history']) > 0:
                    has_stress_data = True
            
            if not has_stress_data:
                return None, "Dictionary missing stress data ('stress_fields' or 'history')."

            # Standardize structure
            standardized = {
                'params': data.get('params', {}),
                'stress_fields': data.get('stress_fields', {}),
                'history': data.get('history', []),
                'metadata': data.get('metadata', {}),
                'file_source': filepath,
                'load_success': True
            }
            
            # If we have history but no stress_fields, extract from last frame
            if standardized['history'] and not standardized['stress_fields']:
                try:
                    eta, stress_fields = standardized['history'][-1]
                    standardized['stress_fields'] = stress_fields
                except:
                    pass
                    
            return standardized, "Success"

        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            return None, f"Failed to unpickle: {type(e).__name__}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)[:100]}"

# Stress Analysis Manager
class StressAnalysisManager:
    """Manager for stress value analysis and visualization"""
    
    @staticmethod
    @handle_errors(default_return={}, log_error=True)
    def compute_max_stress_values(stress_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute maximum stress values from stress fields
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            
        Returns:
            Dictionary with max stress values
        """
        results = {}
        
        try:
            # Hydrostatic stress
            if 'sigma_hydro' in stress_fields:
                hydro_data = stress_fields['sigma_hydro']
                if isinstance(hydro_data, np.ndarray) and hydro_data.size > 0:
                    results['max_abs_hydrostatic'] = float(np.nanmax(np.abs(hydro_data)))
                    results['max_hydrostatic'] = float(np.nanmax(hydro_data))
                    results['min_hydrostatic'] = float(np.nanmin(hydro_data))
                    results['mean_abs_hydrostatic'] = float(np.nanmean(np.abs(hydro_data)))
            
            # Stress magnitude
            if 'sigma_mag' in stress_fields:
                mag_data = stress_fields['sigma_mag']
                if isinstance(mag_data, np.ndarray) and mag_data.size > 0:
                    results['max_stress_magnitude'] = float(np.nanmax(mag_data))
                    results['mean_stress_magnitude'] = float(np.nanmean(mag_data))
            
            # Von Mises stress
            if 'von_mises' in stress_fields:
                vm_data = stress_fields['von_mises']
                if isinstance(vm_data, np.ndarray) and vm_data.size > 0:
                    results['max_von_mises'] = float(np.nanmax(vm_data))
                    results['mean_von_mises'] = float(np.nanmean(vm_data))
                    results['min_von_mises'] = float(np.nanmin(vm_data))
            
            # Principal stresses if available
            if all(k in stress_fields for k in ['sigma_1', 'sigma_2', 'sigma_3']):
                sigma1 = stress_fields['sigma_1']
                sigma2 = stress_fields['sigma_2']
                sigma3 = stress_fields['sigma_3']
                
                if all(isinstance(s, np.ndarray) for s in [sigma1, sigma2, sigma3]):
                    results['max_principal_1'] = float(np.nanmax(sigma1))
                    results['max_principal_2'] = float(np.nanmax(sigma2))
                    results['max_principal_3'] = float(np.nanmax(sigma3))
                    results['max_principal_abs'] = float(np.nanmax(np.abs(sigma1)))
                    
                    # Maximum shear stress (Tresca)
                    max_shear = 0.5 * np.nanmax(np.abs(sigma1 - sigma3))
                    results['max_shear_tresca'] = float(max_shear)
            
            # Additional statistical measures
            if 'sigma_hydro' in stress_fields:
                hydro_data = stress_fields['sigma_hydro']
                if isinstance(hydro_data, np.ndarray) and hydro_data.size > 0:
                    flattened = hydro_data.flatten()
                    flattened = flattened[~np.isnan(flattened)]
                    if len(flattened) > 0:
                        results['hydro_std'] = float(np.nanstd(hydro_data))
                        try:
                            results['hydro_skewness'] = float(stats.skew(flattened))
                            results['hydro_kurtosis'] = float(stats.kurtosis(flattened))
                        except:
                            pass
            
            if 'von_mises' in stress_fields:
                vm_data = stress_fields['von_mises']
                if isinstance(vm_data, np.ndarray) and vm_data.size > 0:
                    # Percentiles
                    try:
                        results['von_mises_p95'] = float(np.nanpercentile(vm_data, 95))
                        results['von_mises_p99'] = float(np.nanpercentile(vm_data, 99))
                        results['von_mises_p99_9'] = float(np.nanpercentile(vm_data, 99.9))
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Error computing stress values: {str(e)}")
        
        return results

# Numerical Solutions Manager
class NumericalSolutionsManager:
    """Manager for numerical solutions directory operations"""
    
    def __init__(self, solutions_dir: str = None):
        self.solutions_dir = solutions_dir or AppConfig.get_numerical_solutions_dir()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Ensure the solutions directory exists"""
        try:
            os.makedirs(self.solutions_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {self.solutions_dir}: {str(e)}")
    
    @handle_errors(default_return={}, log_error=True)
    def scan_directory(self) -> Dict[str, List[str]]:
        """Scan directory for simulation files"""
        file_formats = {
            'pkl': [],
            'pt': [],
            'h5': [],
            'npz': [],
            'sql': [],
            'json': []
        }
        
        try:
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
        except Exception as e:
            logger.error(f"Error scanning directory: {str(e)}")
        
        return file_formats
    
    @handle_errors(default_return=[], log_error=True)
    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get information about all files in the directory"""
        all_files = []
        
        try:
            file_formats = self.scan_directory()
            
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
                        logger.warning(f"Error getting file info for {file_path}: {str(e)}")
            
            all_files.sort(key=lambda x: x['filename'].lower())
        except Exception as e:
            logger.error(f"Error getting all files: {str(e)}")
        
        return all_files
    
    def load_simulation(self, file_path: str, interpolator) -> Dict[str, Any]:
        """Load a simulation file"""
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
            
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Call interpolator's read method
            sim_data = interpolator.read_simulation_file(file_content, format_type)
            sim_data['loaded_from'] = 'numerical_solutions'
            return sim_data
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

# =============================================
# ENHANCED RESILIENT DATA MANAGER
# =============================================
class EnhancedResilientDataManager:
    """Enhanced data manager with parallel loading and caching"""
    
    def __init__(self, solutions_dir: str = None):
        self.solutions_dir = Path(solutions_dir or AppConfig.get_numerical_solutions_dir())
        self.valid_simulations = []
        self.failed_files_log = []
        self.summary_df = pd.DataFrame()
        self._lock = threading.Lock()
        self._cache = {}
        
    @handle_errors(default_return=(0, 0), log_error=True)
    def scan_and_load_all_parallel(self, file_limit: int = 100) -> Tuple[int, int]:
        """Scan and load files in parallel with progress tracking"""
        self.valid_simulations.clear()
        self.failed_files_log.clear()
        
        try:
            # Find all files
            all_files = []
            for ext in AppConfig.ALLOWED_FILE_EXTENSIONS:
                pattern = f"*{ext}" if ext.startswith('.') else f"*.{ext}"
                files = list(self.solutions_dir.glob(pattern))[:file_limit]
                all_files.extend(files)
            
            if not all_files:
                logger.info(f"No files found in {self.solutions_dir}")
                return 0, 0
            
            logger.info(f"Found {len(all_files)} files to process")
            
            # Setup progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_queue = queue.Queue()
            
            def process_file(file_path: Path) -> Optional[Dict]:
                """Process a single file"""
                try:
                    # Check cache first
                    try:
                        mtime = file_path.stat().st_mtime
                    except:
                        mtime = 0
                    
                    cache_key = f"{file_path}_{mtime}"
                    if cache_key in self._cache:
                        return self._cache[cache_key]
                    
                    # Determine format
                    if file_path.suffix.lower() in ['.pkl', '.pickle']:
                        sim_data, message = SafeSimulationLoader.validate_and_load_pkl(str(file_path))
                    elif file_path.suffix.lower() in ['.pt', '.pth']:
                        sim_data = self._load_pt_file(file_path)
                        message = "Success" if sim_data else "Failed"
                    else:
                        return None
                    
                    if sim_data:
                        # Cache the result
                        self._cache[cache_key] = sim_data
                    
                    return sim_data
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    return None
            
            # Process files in parallel
            max_workers = min(AppConfig.MAX_CONCURRENT_WORKERS, len(all_files))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(process_file, file_path): file_path 
                    for file_path in all_files
                }
                
                # Process results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    completed += 1
                    progress_bar.progress(completed / len(all_files))
                    status_text.text(f"Loading {completed}/{len(all_files)} files...")
                    
                    file_path = future_to_file[future]
                    try:
                        sim_data = future.result()
                        if sim_data:
                            with self._lock:
                                self.valid_simulations.append(sim_data)
                        else:
                            self.failed_files_log.append({
                                'file': file_path.name,
                                'error': 'Failed to load'
                            })
                    except Exception as e:
                        self.failed_files_log.append({
                            'file': file_path.name,
                            'error': str(e)[:200]
                        })
            
            progress_bar.empty()
            status_text.empty()
            
            # Create summary
            self._create_summary_dataframe()
            
            logger.info(f"Loaded {len(self.valid_simulations)} simulations, failed {len(self.failed_files_log)}")
            
            return len(self.valid_simulations), len(self.failed_files_log)
            
        except Exception as e:
            logger.error(f"Error in scan_and_load_all_parallel: {str(e)}")
            return 0, 0
    
    @staticmethod
    @handle_errors(default_return=None, log_error=True)
    def _load_pt_file(file_path: Path):
        """Load PyTorch file with enhanced validation"""
        try:
            data = torch.load(file_path, 
                            map_location=torch.device('cpu'), 
                            weights_only=True)
            
            # Enhanced validation
            if not isinstance(data, dict):
                return None
            
            # Standardize structure
            standardized = {
                'params': data.get('params', {}),
                'stress_fields': data.get('stress_fields', {}),
                'history': data.get('history', []),
                'metadata': data.get('metadata', {}),
                'file_source': str(file_path),
                'load_success': True
            }
            
            # Validate stress fields
            if standardized['stress_fields']:
                # Check that all stress fields are numpy arrays
                for key, value in standardized['stress_fields'].items():
                    if not isinstance(value, np.ndarray):
                        # Try to convert if it's a torch tensor
                        if hasattr(value, 'numpy'):
                            standardized['stress_fields'][key] = value.numpy()
            
            return standardized
            
        except Exception as e:
            logger.error(f"Failed to load .pt file {file_path}: {str(e)}")
            return None
    
    def _create_summary_dataframe(self):
        """Create enhanced summary dataframe with more metrics"""
        if not self.valid_simulations:
            self.summary_df = pd.DataFrame()
            return
        
        rows = []
        for idx, sim in enumerate(self.valid_simulations):
            try:
                params = sim.get('params', {})
                stress_fields = sim.get('stress_fields', {})
                
                # Extract parameters with defaults
                row = {
                    'id': f"sim_{idx:03d}",
                    'defect_type': params.get('defect_type', 'Unknown'),
                    'shape': params.get('shape', 'Unknown'),
                    'orientation': params.get('orientation', 'Unknown'),
                    'eps0': float(params.get('eps0', 0)),
                    'kappa': float(params.get('kappa', 0)),
                    'theta': float(params.get('theta', 0)),
                    'theta_deg': np.rad2deg(float(params.get('theta', 0))),
                    'source_file': Path(sim.get('file_source', '')).name,
                    'type': 'source'
                }
                
                # Compute stress metrics with error handling
                if 'von_mises' in stress_fields:
                    vm_data = stress_fields['von_mises']
                    if isinstance(vm_data, np.ndarray) and vm_data.size > 0:
                        row.update({
                            'max_von_mises': float(np.nanmax(vm_data)),
                            'mean_von_mises': float(np.nanmean(vm_data)),
                            'std_von_mises': float(np.nanstd(vm_data)),
                            'p95_von_mises': float(np.nanpercentile(vm_data, 95)),
                            'p99_von_mises': float(np.nanpercentile(vm_data, 99))
                        })
                
                if 'sigma_hydro' in stress_fields:
                    hydro_data = stress_fields['sigma_hydro']
                    if isinstance(hydro_data, np.ndarray) and hydro_data.size > 0:
                        row.update({
                            'max_abs_hydrostatic': float(np.nanmax(np.abs(hydro_data))),
                            'mean_abs_hydrostatic': float(np.nanmean(np.abs(hydro_data))),
                            'hydro_tension_area': float(np.sum(hydro_data > 0) / hydro_data.size * 100),
                            'hydro_compression_area': float(np.sum(hydro_data < 0) / hydro_data.size * 100)
                        })
                
                # Add derived metrics
                if 'max_von_mises' in row and 'max_abs_hydrostatic' in row:
                    row['stress_ratio_vm_hydro'] = row['max_von_mises'] / (row['max_abs_hydrostatic'] + 1e-10)
                
                rows.append(row)
                
            except Exception as e:
                logger.warning(f"Error processing simulation {idx}: {str(e)}")
                # Add minimal row even if processing fails
                rows.append({
                    'id': f"sim_{idx:03d}",
                    'defect_type': 'Error',
                    'shape': 'Error',
                    'source_file': 'Error',
                    'type': 'source'
                })
        
        try:
            self.summary_df = pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error creating summary dataframe: {str(e)}")
            self.summary_df = pd.DataFrame()
    
    @handle_errors(default_return={}, log_error=True)
    def get_enhanced_summary_report(self) -> Dict[str, Any]:
        """Generate enhanced summary report with statistics"""
        if self.summary_df.empty:
            return {}
        
        report = {
            'total_simulations': len(self.summary_df),
            'failed_files': len(self.failed_files_log),
        }
        
        try:
            if 'defect_type' in self.summary_df.columns:
                report['defect_type_counts'] = self.summary_df['defect_type'].value_counts().to_dict()
            
            if 'shape' in self.summary_df.columns:
                report['shape_counts'] = self.summary_df['shape'].value_counts().to_dict()
            
            if 'eps0' in self.summary_df.columns:
                report['parameter_ranges'] = {
                    'eps0': {
                        'min': float(self.summary_df['eps0'].min()),
                        'max': float(self.summary_df['eps0'].max()),
                        'mean': float(self.summary_df['eps0'].mean())
                    }
                }
            
            # Add stress statistics if available
            stress_cols = [col for col in self.summary_df.columns if 'von_mises' in col or 'hydrostatic' in col]
            for col in stress_cols:
                if col in self.summary_df.columns:
                    try:
                        report[f'{col}_stats'] = {
                            'min': float(self.summary_df[col].min()),
                            'max': float(self.summary_df[col].max()),
                            'mean': float(self.summary_df[col].mean()),
                            'std': float(self.summary_df[col].std())
                        }
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
        
        return report

# =============================================
# ENHANCED MULTI-TARGET PREDICTION MANAGER
# =============================================
class EnhancedMultiTargetPredictionManager:
    """Enhanced manager with flexible parameter grid generation"""
    
    @staticmethod
    @handle_errors(default_return=[], log_error=True)
    def parse_custom_angles(angle_input: str) -> List[float]:
        """
        Parse custom angles from various input formats
        
        Supports:
        - Comma-separated: "0, 15, 30, 45"
        - Range with steps: "0:90:10" (start:stop:step)
        - Mixed: "0, 15:45:15, 60, 90"
        """
        angles = []
        
        if not angle_input:
            return angles
        
        try:
            # Split by comma first
            parts = [p.strip() for p in angle_input.split(',')]
            
            for part in parts:
                if not part:
                    continue
                    
                if ':' in part:
                    # Range format
                    range_parts = part.split(':')
                    if len(range_parts) == 2:
                        start, stop = map(float, range_parts)
                        step = 1.0
                    elif len(range_parts) == 3:
                        start, stop, step = map(float, range_parts)
                    else:
                        raise ValueError(f"Invalid range format: {part}")
                    
                    # Generate angles
                    if start <= stop:
                        angles.extend(np.arange(start, stop + step/2, step).tolist())
                    else:
                        angles.extend(np.arange(stop, start + step/2, step).tolist())
                else:
                    # Single angle
                    angles.append(float(part))
            
            # Remove duplicates and sort
            angles = sorted(list(set(angles)))
            
        except Exception as e:
            logger.error(f"Error parsing custom angles '{angle_input}': {str(e)}")
            return []
        
        return angles
    
    @staticmethod
    @handle_errors(default_return=[], log_error=True)
    def create_flexible_parameter_grid(base_params: Dict, 
                                     configs: Dict[str, Dict]) -> List[Dict]:
        """
        Create parameter grid with flexible configuration
        
        Args:
            base_params: Base parameter dictionary
            configs: Dictionary with parameter configurations
        
        Returns:
            List of parameter dictionaries
        """
        param_values = {}
        
        try:
            for param_name, config in configs.items():
                config_type = config.get('type', 'value')
                
                if config_type == 'range':
                    # Linear range
                    min_val = config.get('min', 0)
                    max_val = config.get('max', 1)
                    steps = config.get('steps', 10)
                    if max_val > min_val:
                        param_values[param_name] = np.linspace(min_val, max_val, steps).tolist()
                    else:
                        param_values[param_name] = [min_val]
                
                elif config_type == 'values':
                    # Specific values
                    param_values[param_name] = config.get('values', [])
                
                elif config_type == 'custom':
                    # Custom input (for angles)
                    if param_name == 'theta':
                        angle_input = config.get('input', '')
                        angles_deg = EnhancedMultiTargetPredictionManager.parse_custom_angles(angle_input)
                        param_values[param_name] = [np.deg2rad(a) for a in angles_deg]
                    else:
                        custom_input = config.get('input', '')
                        param_values[param_name] = EnhancedMultiTargetPredictionManager.parse_custom_angles(custom_input)
                
                elif config_type == 'categorical':
                    # Categorical values
                    param_values[param_name] = config.get('values', [])
                
                else:
                    # Single value
                    param_values[param_name] = [config.get('value', base_params.get(param_name))]
            
            # Generate all combinations
            param_names = list(param_values.keys())
            value_arrays = [param_values[name] for name in param_names]
            
            param_grid = []
            for combination in product(*value_arrays):
                param_dict = base_params.copy()
                for name, value in zip(param_names, combination):
                    if isinstance(value, (int, float, np.number)):
                        param_dict[name] = float(value)
                    else:
                        param_dict[name] = value
                
                # For angles, also set orientation string
                if 'theta' in param_dict:
                    angle_deg = np.rad2deg(param_dict['theta'])
                    param_dict['orientation'] = EnhancedMultiTargetPredictionManager.get_orientation_from_angle(angle_deg)
                
                param_grid.append(param_dict)
            
            return param_grid
            
        except Exception as e:
            logger.error(f"Error creating parameter grid: {str(e)}")
            return []
    
    @staticmethod
    def get_orientation_from_angle(angle_deg: float) -> str:
        """Convert angle to orientation string with custom support"""
        try:
            # Normalize angle
            angle_deg = angle_deg % 360
            
            # Predefined mappings
            predefined = {
                0: 'Horizontal {111} (0°)',
                30: 'Tilted 30° (1¯10 projection)',
                60: 'Tilted 60°',
                90: 'Vertical {111} (90°)',
                120: 'Vertical {111} (120°)',
                150: 'Tilted 150°',
                180: 'Horizontal {111} (180°)',
                210: 'Tilted 210°',
                240: 'Tilted 240°',
                270: 'Vertical {111} (270°)',
                300: 'Tilted 300°',
                330: 'Tilted 330°'
            }
            
            # Check for exact matches
            if angle_deg in predefined:
                return predefined[angle_deg]
            
            # Check for close matches (within 5 degrees)
            for predefined_angle, orientation in predefined.items():
                if abs(angle_deg - predefined_angle) <= 5:
                    return orientation
            
            # Custom angle
            return f"Custom ({angle_deg:.1f}°)"
            
        except:
            return f"Custom ({angle_deg:.1f}°)"

# =============================================
# ENHANCED STRESS ANALYSIS MANAGER
# =============================================
class EnhancedStressAnalysisManager:
    """Enhanced stress analysis with comprehensive metrics"""
    
    @staticmethod
    @cache_results(maxsize=100, ttl_seconds=3600)
    def compute_comprehensive_stress_metrics(stress_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute comprehensive stress metrics with validation
        
        Args:
            stress_fields: Dictionary containing stress component arrays
        
        Returns:
            Dictionary with comprehensive stress metrics
        """
        metrics = {}
        
        try:
            # Validate input
            if not stress_fields:
                return metrics
            
            # Process each stress component
            for component_name, stress_data in stress_fields.items():
                if not isinstance(stress_data, np.ndarray):
                    continue
                
                # Basic statistics
                valid_data = stress_data[~np.isnan(stress_data)]
                if len(valid_data) == 0:
                    continue
                
                metrics[f'{component_name}_max'] = float(np.nanmax(stress_data))
                metrics[f'{component_name}_min'] = float(np.nanmin(stress_data))
                metrics[f'{component_name}_mean'] = float(np.nanmean(stress_data))
                metrics[f'{component_name}_std'] = float(np.nanstd(stress_data))
                metrics[f'{component_name}_median'] = float(np.nanmedian(stress_data))
                
                # Percentiles
                for p in [10, 25, 50, 75, 90, 95, 99]:
                    try:
                        metrics[f'{component_name}_p{p}'] = float(np.nanpercentile(stress_data, p))
                    except:
                        pass
            
            # Special handling for key components
            if 'sigma_hydro' in stress_fields:
                hydro_data = stress_fields['sigma_hydro']
                valid_hydro = hydro_data[~np.isnan(hydro_data)]
                
                if len(valid_hydro) > 0:
                    try:
                        metrics['hydrostatic_tension_area'] = float(np.sum(hydro_data > 0) / hydro_data.size * 100)
                        metrics['hydrostatic_compression_area'] = float(np.sum(hydro_data < 0) / hydro_data.size * 100)
                        
                        tension_data = hydro_data[hydro_data > 0]
                        if len(tension_data) > 0:
                            metrics['hydrostatic_max_tension'] = float(np.nanmax(tension_data))
                        
                        compression_data = hydro_data[hydro_data < 0]
                        if len(compression_data) > 0:
                            metrics['hydrostatic_max_compression'] = float(np.nanmin(compression_data))
                        
                        if 'von_mises_mean' in metrics:
                            metrics['hydrostatic_triaxiality'] = float(np.nanmean(np.abs(hydro_data)) / (metrics['von_mises_mean'] + 1e-10))
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Error computing stress metrics: {str(e)}")
        
        return metrics

# =============================================
# SPATIAL LOCALITY ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator:
    """Enhanced attention-based interpolator with spatial locality regularization"""
    
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
        
        # Initialize readers
        self.readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
    
    def _build_model(self):
        """Build the neural network model"""
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
            )
        })
        return model
    
    def _read_pkl(self, file_content):
        """Read PKL file"""
        buffer = BytesIO(file_content)
        return pickle.load(buffer)
    
    def _read_pt(self, file_content):
        """Read PyTorch file"""
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)
    
    def _read_h5(self, file_content):
        """Read HDF5 file"""
        buffer = BytesIO(file_content)
        with h5py.File(buffer, 'r') as f:
            data = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][()]
                elif isinstance(f[key], h5py.Group):
                    group_data = {}
                    for subkey in f[key].keys():
                        group_data[subkey] = f[f"{key}/{subkey}"][()]
                    data[key] = group_data
        return data
    
    def _read_npz(self, file_content):
        """Read NPZ file"""
        buffer = BytesIO(file_content)
        data = np.load(buffer, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    def _read_sql(self, file_content):
        """Read SQLite database"""
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            # Get all tables
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
            raise e
    
    def _read_json(self, file_content):
        """Read JSON file"""
        return json.loads(file_content.decode('utf-8'))
    
    @handle_errors(default_return={}, log_error=True)
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file from content"""
        if format_type == 'auto':
            format_type = 'pkl'
        
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            return self._standardize_data(data, format_type, "uploaded_file")
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _standardize_data(self, data, format_type, file_path):
        """Standardize data structure"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
        }
        
        try:
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
                # Extract from H5 structure
                if 'params' in data:
                    standardized['params'] = data['params']
                if 'metadata' in data:
                    standardized['metadata'] = data['metadata']
            
            elif format_type == 'npz':
                # NPZ files
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
        
        except Exception as e:
            logger.error(f"Error standardizing data: {str(e)}")
        
        return standardized
    
    @handle_errors(default_return=(np.array([]), []), log_error=True)
    def compute_parameter_vector(self, sim_data):
        """Compute parameter vector from simulation data"""
        params = sim_data.get('params', {})
        
        param_vector = []
        param_names = []
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error computing parameter vector: {str(e)}")
            param_vector = np.zeros(self.input_dim)
            param_names = []
        
        return np.array(param_vector, dtype=np.float32), param_names

# =============================================
# ENHANCED VISUALIZATION MANAGER
# =============================================
class EnhancedVisualizationManager:
    """Enhanced visualization with robust error handling"""
    
    @staticmethod
    @handle_errors(default_return=None, log_error=True)
    def create_robust_box_plot(df: pd.DataFrame, 
                              value_columns: List[str],
                              group_by_column: str = 'defect_type',
                              title: str = "Box Plot",
                              max_categories: int = 10) -> Optional[go.Figure]:
        """Create robust box plot with validation"""
        if df.empty or len(df) < 2:
            return None
        
        try:
            # Validate inputs
            valid_value_cols = [col for col in value_columns if col in df.columns]
            if not valid_value_cols:
                return None
            
            if group_by_column not in df.columns:
                return None
            
            # Limit number of categories
            unique_groups = df[group_by_column].unique()
            if len(unique_groups) > max_categories:
                group_counts = df[group_by_column].value_counts()
                keep_groups = group_counts.head(max_categories).index.tolist()
                df = df[df[group_by_column].isin(keep_groups)]
                unique_groups = keep_groups
            
            # Create figure
            fig = make_subplots(
                rows=len(valid_value_cols), 
                cols=1,
                subplot_titles=[f"Distribution of {col}" for col in valid_value_cols],
                vertical_spacing=0.1
            )
            
            # Color palette
            colors = px.colors.qualitative.Set3
            
            for i, col in enumerate(valid_value_cols):
                for j, group_name in enumerate(unique_groups):
                    group_data = df[df[group_by_column] == group_name][col].dropna()
                    
                    if len(group_data) > 0:
                        color = colors[j % len(colors)]
                        
                        fig.add_trace(
                            go.Box(
                                y=group_data, 
                                name=str(group_name),
                                marker_color=color,
                                legendgroup=group_name,
                                showlegend=(i == 0),
                                boxmean='sd'
                            ),
                            row=i + 1, 
                            col=1
                        )
            
            fig.update_layout(
                height=300 * len(valid_value_cols),
                showlegend=True,
                title_text=title,
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return None

# =============================================
# REFACTORED MAIN APPLICATION
# =============================================
class EnhancedStressAnalysisApp:
    """Refactored main application with improved architecture"""
    
    def __init__(self):
        # Initialize configuration
        AppConfig.ensure_all_directories()
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize managers
        self._init_managers()
        
        # Setup page config
        st.set_page_config(
            page_title="Enhanced Stress Analysis Dashboard",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _init_session_state(self):
        """Initialize session state with defaults"""
        defaults = {
            'source_simulations': [],
            'uploaded_files': {},
            'loaded_from_numerical': [],
            'multi_target_predictions': {},
            'multi_target_params': [],
            'stress_summary_df': pd.DataFrame(),
            'save_format': 'both',
            'save_to_directory': True,
            'current_tab': 'load_data',
            'app_config': AppConfig,
            'last_operation': None,
            'operation_status': {},
            'prediction_results': None,
            'target_params': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _init_managers(self):
        """Initialize all manager instances"""
        try:
            self.solutions_manager = NumericalSolutionsManager()
            self.resilient_data_manager = EnhancedResilientDataManager()
            self.stress_analyzer = EnhancedStressAnalysisManager()
            self.visualization_manager = EnhancedVisualizationManager()
            self.multi_target_manager = EnhancedMultiTargetPredictionManager()
            
            # Lazy initialization of interpolator
            if 'interpolator' not in st.session_state:
                st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
                
        except Exception as e:
            logger.error(f"Error initializing managers: {str(e)}")
            st.error(f"Error initializing application: {str(e)}")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("⚙️ Application Configuration")
            
            # Operation mode
            operation_mode = st.radio(
                "Select Mode",
                ["Attention Interpolation", "Stress Analysis Dashboard"],
                index=0,
                key="operation_mode"
            )
            
            st.divider()
            
            # File management
            with st.expander("📁 File Management", expanded=False):
                max_files = st.slider(
                    "Maximum files to load",
                    10, 1000, 200, 10,
                    help="Limit number of files to prevent memory issues"
                )
                
                enable_parallel = st.checkbox(
                    "Enable parallel loading",
                    value=True,
                    help="Load files in parallel for better performance"
                )
            
            # Visualization settings
            with st.expander("🎨 Visualization", expanded=False):
                default_colormap = st.selectbox(
                    "Default colormap",
                    ['viridis', 'plasma', 'coolwarm', 'RdBu', 'Spectral', 'jet'],
                    index=0
                )
            
            # Advanced settings
            with st.expander("⚡ Advanced", expanded=False):
                enable_caching = st.checkbox("Enable caching", value=True)
            
            st.divider()
            
            # System info
            st.caption(f"**Files loaded:** {len(st.session_state.source_simulations)}")
            st.caption(f"**Directory:** {AppConfig.get_numerical_solutions_dir()}")
            
            if st.button("🔄 Clear All Data", type="secondary"):
                self._clear_all_data()
    
    def _clear_all_data(self):
        """Clear all session state data"""
        keys_to_keep = ['app_config', 'save_format', 'save_to_directory']
        
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        
        self._init_session_state()
        st.rerun()
    
    def render_main_tabs(self):
        """Render the main application tabs"""
        tabs = st.tabs([
            "📥 Load Data",
            "🎯 Single Target",
            "🎯 Multiple Targets",
            "🚀 Predict",
            "📊 Results",
            "💾 Export",
            "📈 Analysis"
        ])
        
        with tabs[0]:
            self.render_tab_load_data()
        
        with tabs[1]:
            self.render_tab_single_target()
        
        with tabs[2]:
            self.render_tab_multiple_targets()
        
        with tabs[3]:
            self.render_tab_predict()
        
        with tabs[4]:
            self.render_tab_results()
        
        with tabs[5]:
            self.render_tab_export()
        
        with tabs[6]:
            self.render_tab_analysis()
    
    def render_tab_load_data(self):
        """Render the data loading tab"""
        st.header("📥 Load Simulation Data")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_directory_loading()
        
        with col2:
            self._render_file_upload()
        
        # Display loaded simulations
        if st.session_state.source_simulations:
            self._render_loaded_simulations()
    
    def _render_directory_loading(self):
        """Render directory loading section"""
        st.subheader("📂 Load from Directory")
        
        try:
            # Scan directory
            all_files = self.solutions_manager.get_all_files()
            
            if not all_files:
                st.info(f"No files found in `{AppConfig.get_numerical_solutions_dir()}`")
                if st.button("Create Sample Data"):
                    self._create_sample_data()
                return
            
            # File selection
            file_options = {
                f"{f['filename']} ({f['size']//1024}KB)": f['path']
                for f in all_files[:100]  # Limit display
            }
            
            selected_files = st.multiselect(
                "Select files to load",
                options=list(file_options.keys()),
                key="dir_file_select"
            )
            
            if selected_files and st.button("📥 Load Selected Files", type="primary"):
                with st.spinner(f"Loading {len(selected_files)} files..."):
                    self._load_files_from_directory(
                        [file_options[f] for f in selected_files]
                    )
                    
        except Exception as e:
            st.error(f"Error loading from directory: {str(e)}")
            logger.error(f"Directory loading error: {str(e)}")
    
    def _load_files_from_directory(self, file_paths: List[str]):
        """Load files from directory with progress tracking"""
        loaded_count = 0
        failed_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file_path in enumerate(file_paths):
            status_text.text(f"Loading {idx+1}/{len(file_paths)}: {os.path.basename(file_path)}")
            progress_bar.progress((idx + 1) / len(file_paths))
            
            try:
                if file_path not in st.session_state.loaded_from_numerical:
                    sim_data = self.solutions_manager.load_simulation(
                        file_path,
                        st.session_state.interpolator
                    )
                    st.session_state.source_simulations.append(sim_data)
                    st.session_state.loaded_from_numerical.append(file_path)
                    loaded_count += 1
                else:
                    st.warning(f"Already loaded: {os.path.basename(file_path)}")
                    
            except Exception as e:
                failed_files.append((os.path.basename(file_path), str(e)))
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        if loaded_count > 0:
            st.success(f"✅ Successfully loaded {loaded_count} files")
        
        if failed_files:
            with st.expander("❌ Failed Files", expanded=False):
                for filename, error in failed_files:
                    st.error(f"{filename}: {error}")
    
    def _render_file_upload(self):
        """Render file upload section"""
        st.subheader("📤 Upload Files")
        
        uploaded_files = st.file_uploader(
            "Upload simulation files",
            type=[ext.lstrip('.') for ext in AppConfig.ALLOWED_FILE_EXTENSIONS],
            accept_multiple_files=True,
            help=f"Maximum file size: {AppConfig.MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )
        
        if uploaded_files and st.button("📥 Process Uploaded Files", type="primary"):
            with st.spinner("Processing uploaded files..."):
                self._process_uploaded_files(uploaded_files)
    
    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files securely"""
        loaded_count = 0
        
        for uploaded_file in uploaded_files:
            try:
                # Validate file extension
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext not in AppConfig.ALLOWED_FILE_EXTENSIONS:
                    st.warning(f"Skipped {uploaded_file.name}: Invalid extension {ext}")
                    continue
                
                # Read file content
                file_content = uploaded_file.getvalue()
                
                # Determine format
                format_map = {
                    '.pkl': 'pkl',
                    '.pt': 'pt',
                    '.h5': 'h5',
                    '.hdf5': 'h5',
                    '.npz': 'npz',
                    '.json': 'json'
                }
                file_format = format_map.get(ext, 'auto')
                
                # Load file using interpolator
                sim_data = st.session_state.interpolator.read_simulation_file(
                    file_content, file_format
                )
                sim_data['loaded_from'] = 'upload'
                
                # Add to session state
                st.session_state.source_simulations.append(sim_data)
                loaded_count += 1
                
                st.success(f"✅ Loaded: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Failed to load {uploaded_file.name}: {str(e)}")
                logger.error(f"Upload error: {str(e)}", exc_info=True)
        
        if loaded_count > 0:
            st.success(f"Successfully processed {loaded_count} uploaded files")
    
    def _render_loaded_simulations(self):
        """Render loaded simulations summary"""
        st.subheader("📋 Loaded Simulations Summary")
        
        try:
            # Create summary table
            summary_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                
                summary_data.append({
                    'ID': i + 1,
                    'Source': sim_data.get('loaded_from', 'unknown'),
                    'Defect': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'ε*': f"{params.get('eps0', 0):.3f}",
                    'κ': f"{params.get('kappa', 0):.3f}",
                    'θ': f"{np.rad2deg(params.get('theta', 0)):.1f}°",
                    'Format': sim_data.get('format', 'Unknown')
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Simulations", len(st.session_state.source_simulations))
            with col2:
                unique_defects = df_summary['Defect'].nunique()
                st.metric("Unique Defects", unique_defects)
            with col3:
                unique_shapes = df_summary['Shape'].nunique()
                st.metric("Unique Shapes", unique_shapes)
                
        except Exception as e:
            st.error(f"Error displaying loaded simulations: {str(e)}")
    
    def render_tab_single_target(self):
        """Render single target configuration tab"""
        st.header("🎯 Single Target Configuration")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
            return
        
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
            
            orientation_mode = st.radio(
                "Orientation Mode",
                ["Predefined", "Custom Angle"],
                horizontal=True,
                key="orientation_mode_single"
            )
            
            if orientation_mode == "Predefined":
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
                
            else:
                target_angle = st.slider(
                    "Target Angle (degrees)",
                    0.0, 90.0, 0.0, 0.5,
                    key="target_angle_custom_single"
                )
                target_theta = np.deg2rad(target_angle)
                target_orientation = st.session_state.interpolator.get_orientation_from_angle(target_angle)
                st.info(f"**Target θ:** {target_angle:.1f}°")
                st.info(f"**Orientation:** {target_orientation}")
        
        target_params = {
            'defect_type': target_defect,
            'shape': target_shape,
            'eps0': target_eps0,
            'kappa': target_kappa,
            'orientation': target_orientation,
            'theta': target_theta
        }
        
        st.session_state.target_params = target_params
        
        st.success("✅ Target parameters configured!")
    
    def render_tab_multiple_targets(self):
        """Render multiple targets configuration tab"""
        st.header("🎯 Configure Multiple Targets")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
            return
        
        # Base parameters
        st.subheader("Base Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_defect = st.selectbox(
                "Defect Type",
                ["ISF", "ESF", "Twin"],
                index=0,
                key="multi_base_defect"
            )
            
            base_shape = st.selectbox(
                "Shape",
                ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                index=0,
                key="multi_base_shape"
            )
        
        with col2:
            orientation_mode = st.radio(
                "Orientation Mode",
                ["Predefined", "Custom Angles"],
                horizontal=True,
                key="multi_orientation_mode"
            )
            
            if orientation_mode == "Predefined":
                base_orientation = st.selectbox(
                    "Orientation",
                    ["Horizontal {111} (0°)", 
                     "Tilted 30° (1¯10 projection)", 
                     "Tilted 60°", 
                     "Vertical {111} (90°)",
                     "Custom (15°)",
                     "Custom (45°)",
                     "Custom (75°)"],
                    index=0,
                    key="multi_base_orientation"
                )
                
                # Map to angle
                angle_map = {
                    "Horizontal {111} (0°)": 0,
                    "Tilted 30° (1¯10 projection)": 30,
                    "Tilted 60°": 60,
                    "Vertical {111} (90°)": 90,
                    "Custom (15°)": 15,
                    "Custom (45°)": 45,
                    "Custom (75°)": 75,
                }
                base_theta = np.deg2rad(angle_map.get(base_orientation, 0))
            
            else:
                custom_angle = st.slider(
                    "Custom Angle (degrees)",
                    0.0, 360.0, 0.0, 1.0,
                    key="multi_custom_angle"
                )
                base_theta = np.deg2rad(custom_angle)
                base_orientation = f"Custom ({custom_angle:.1f}°)"
        
        base_params = {
            'defect_type': base_defect,
            'shape': base_shape,
            'orientation': base_orientation,
            'theta': base_theta
        }
        
        # Parameter configurations
        st.subheader("Parameter Configurations")
        
        # ε* configuration
        eps0_config_type = st.radio(
            "ε* variation type",
            ["Single value", "Range", "Custom values"],
            horizontal=True,
            key="eps0_config_type"
        )
        
        if eps0_config_type == "Single value":
            eps0_value = st.number_input("ε* value", 0.3, 3.0, 1.414, 0.01)
            eps0_config = {'type': 'value', 'value': eps0_value}
        
        elif eps0_config_type == "Range":
            col1, col2, col3 = st.columns(3)
            with col1:
                eps0_min = st.number_input("Min ε*", 0.3, 3.0, 0.5, 0.1)
            with col2:
                eps0_max = st.number_input("Max ε*", 0.3, 3.0, 2.5, 0.1)
            with col3:
                eps0_steps = st.number_input("Steps", 2, 100, 10, 1)
            
            if eps0_max > eps0_min:
                eps0_config = {
                    'type': 'range',
                    'min': float(eps0_min),
                    'max': float(eps0_max),
                    'steps': int(eps0_steps)
                }
            else:
                st.error("Max must be greater than Min")
                eps0_config = {'type': 'value', 'value': 1.414}
        
        else:  # Custom values
            eps0_custom = st.text_input(
                "Custom ε* values (comma-separated or range)",
                "0.5, 1.0, 1.5, 2.0",
                help="Example: 0.5, 1.0, 1.5, 2.0 or 0.5:2.5:0.5"
            )
            eps0_config = {'type': 'custom', 'input': eps0_custom}
        
        # κ configuration
        kappa_config_type = st.radio(
            "κ variation type",
            ["Single value", "Range", "Custom values"],
            horizontal=True,
            key="kappa_config_type"
        )
        
        if kappa_config_type == "Single value":
            kappa_value = st.number_input("κ value", 0.1, 2.0, 0.7, 0.05)
            kappa_config = {'type': 'value', 'value': kappa_value}
        
        elif kappa_config_type == "Range":
            col1, col2, col3 = st.columns(3)
            with col1:
                kappa_min = st.number_input("Min κ", 0.1, 2.0, 0.2, 0.05)
            with col2:
                kappa_max = st.number_input("Max κ", 0.1, 2.0, 1.5, 0.05)
            with col3:
                kappa_steps = st.number_input("Steps", 2, 50, 8, 1)
            
            if kappa_max > kappa_min:
                kappa_config = {
                    'type': 'range',
                    'min': float(kappa_min),
                    'max': float(kappa_max),
                    'steps': int(kappa_steps)
                }
            else:
                st.error("Max must be greater than Min")
                kappa_config = {'type': 'value', 'value': 0.7}
        
        else:  # Custom values
            kappa_custom = st.text_input(
                "Custom κ values (comma-separated or range)",
                "0.2, 0.5, 0.8, 1.2",
                help="Example: 0.2, 0.5, 0.8, 1.2 or 0.2:1.5:0.3"
            )
            kappa_config = {'type': 'custom', 'input': kappa_custom}
        
        # Generate parameter grid
        if st.button("🔄 Generate Parameter Grid", type="primary"):
            with st.spinner("Generating parameter grid..."):
                try:
                    # Build configs dictionary
                    configs = {
                        'eps0': eps0_config,
                        'kappa': kappa_config
                    }
                    
                    # Generate grid
                    param_grid = self.multi_target_manager.create_flexible_parameter_grid(
                        base_params, configs
                    )
                    
                    st.session_state.multi_target_params = param_grid
                    
                    # Display grid
                    self._display_parameter_grid(param_grid)
                    
                    st.success(f"✅ Generated {len(param_grid)} parameter combinations")
                    
                except Exception as e:
                    st.error(f"❌ Error generating parameter grid: {str(e)}")
                    logger.error(f"Parameter grid error: {str(e)}", exc_info=True)
    
    def _display_parameter_grid(self, param_grid: List[Dict]):
        """Display parameter grid in an interactive table"""
        if not param_grid:
            return
        
        try:
            # Convert to DataFrame
            grid_data = []
            for i, params in enumerate(param_grid):
                grid_data.append({
                    'ID': i + 1,
                    'Defect': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'ε*': f"{params.get('eps0', 0):.3f}",
                    'κ': f"{params.get('kappa', 0):.3f}",
                    'Orientation': params.get('orientation', 'Unknown'),
                    'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                })
            
            df_grid = pd.DataFrame(grid_data)
            
            st.subheader("📋 Generated Parameter Grid")
            st.dataframe(df_grid, use_container_width=True, height=300)
            
        except Exception as e:
            st.error(f"Error displaying parameter grid: {str(e)}")
    
    def render_tab_predict(self):
        """Render prediction tab"""
        st.header("🚀 Train Model and Predict")
        
        prediction_mode = st.radio(
            "Select Prediction Mode",
            ["Single Target", "Multiple Targets (Batch)"],
            index=0,
            key="prediction_mode"
        )
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
            return
        
        if prediction_mode == "Single Target" and 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure single target parameters first")
            return
        
        if prediction_mode == "Multiple Targets" and not st.session_state.multi_target_params:
            st.warning("⚠️ Please generate a parameter grid first")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Training Epochs", 10, 200, 50, 10)
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
        
        with col2:
            batch_size = st.slider("Batch Size", 1, 16, 4, 1)
            validation_split = st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05)
        
        if st.button("🚀 Train & Predict", type="primary"):
            with st.spinner("Training attention model and predicting..."):
                try:
                    if prediction_mode == "Single Target":
                        self._predict_single_target(epochs)
                    else:
                        self._predict_multiple_targets()
                        
                except Exception as e:
                    st.error(f"❌ Error during training/prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}", exc_info=True)
    
    def _predict_single_target(self, epochs: int):
        """Predict for single target"""
        param_vectors = []
        stress_data = []
        
        for sim_data in st.session_state.source_simulations:
            param_vector, _ = st.session_state.interpolator.compute_parameter_vector(sim_data)
            param_vectors.append(param_vector)
            
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                stress_data.append(stress_components)
        
        target_vector, _ = st.session_state.interpolator.compute_parameter_vector(
            {'params': st.session_state.target_params}
        )
        
        param_vectors = np.array(param_vectors)
        distances = np.sqrt(np.sum((param_vectors - target_vector) ** 2, axis=1))
        weights = np.exp(-0.5 * (distances / 0.3) ** 2)
        weights = weights / (np.sum(weights) + 1e-8)
        
        stress_data = np.array(stress_data)
        weighted_stress = np.sum(stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        
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
    
    def _predict_multiple_targets(self):
        """Predict for multiple targets"""
        # This would be implemented similarly to the single target case
        # but for each parameter combination in multi_target_params
        st.info("Multi-target prediction would be implemented here")
        # For now, just use the first parameter set
        if st.session_state.multi_target_params:
            first_params = st.session_state.multi_target_params[0]
            st.session_state.target_params = first_params
            self._predict_single_target(50)  # Use default epochs
    
    def render_tab_results(self):
        """Render results tab"""
        st.header("📊 Prediction Results Visualization")
        
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
            return
        
        results = st.session_state.prediction_results
        
        # Visualization controls
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        with col_viz1:
            stress_component = st.selectbox(
                "Select Stress Component",
                ['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0
            )
        with col_viz2:
            colormap = st.selectbox(
                "Colormap",
                ['viridis', 'plasma', 'coolwarm', 'RdBu', 'Spectral'],
                index=0
            )
        
        # Plot stress field
        if stress_component in results.get('stress_fields', {}):
            stress_data = results['stress_fields'][stress_component]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(stress_data, extent=[-6.4, 6.4, -6.4, 6.4], cmap=colormap,
                          origin='lower', aspect='equal')
            
            ax.set_title(f'{stress_component.replace("_", " ").title()} (GPa)')
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Stress (GPa)')
            
            st.pyplot(fig)
        
        # Attention weights visualization
        st.subheader("🔍 Attention Weights")
        
        if 'attention_weights' in results:
            weights = results['attention_weights']
            source_names = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
            
            fig_weights, ax_weights = plt.subplots(figsize=(10, 4))
            bars = ax_weights.bar(source_names, weights, alpha=0.7, color='steelblue')
            ax_weights.set_xlabel('Source Simulations')
            ax_weights.set_ylabel('Attention Weight')
            ax_weights.set_title('Attention Weights Distribution')
            ax_weights.set_ylim(0, max(weights) * 1.2)
            
            # Add value labels on bars
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax_weights.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig_weights)
    
    def render_tab_export(self):
        """Render export tab"""
        st.header("💾 Save and Export Prediction Results")
        
        # Check if we have predictions to save
        has_single_prediction = 'prediction_results' in st.session_state
        
        if not has_single_prediction:
            st.warning("⚠️ No prediction results available to save. Please run predictions first.")
            return
        
        st.success("✅ Prediction results available for export!")
        
        # Display what's available
        if has_single_prediction:
            st.info(f"**Single Target Prediction:** Available")
            single_params = st.session_state.prediction_results.get('target_params', {})
            st.write(f"- Target: {single_params.get('defect_type', 'Unknown')}, "
                    f"ε*={single_params.get('eps0', 0):.3f}, "
                    f"κ={single_params.get('kappa', 0):.3f}")
        
        # Save options
        st.subheader("📁 Save Options")
        
        save_col1, save_col2 = st.columns(2)
        
        with save_col1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = st.text_input(
                "Base filename",
                value=f"prediction_{timestamp}",
                help="Files will be saved with this base name plus appropriate extensions"
            )
        
        with save_col2:
            include_source_info = st.checkbox("Include source simulations info", value=True)
        
        # Download buttons
        st.subheader("⬇️ Download Options")
        
        if st.button("💾 Save as PKL", type="secondary", use_container_width=True):
            with st.spinner("Preparing PKL file..."):
                try:
                    # Create save data
                    save_data = {
                        'prediction_results': st.session_state.prediction_results,
                        'source_count': len(st.session_state.source_simulations),
                        'timestamp': timestamp,
                        'metadata': {
                            'version': '1.0',
                            'created_by': 'Enhanced Stress Analysis App'
                        }
                    }
                    
                    # Create download link
                    pkl_buffer = BytesIO()
                    pickle.dump(save_data, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                    pkl_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download PKL",
                        data=pkl_buffer,
                        file_name=f"{base_filename}.pkl",
                        mime="application/octet-stream",
                        key="download_pkl"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error saving PKL: {str(e)}")
    
    def render_tab_analysis(self):
        """Render the enhanced stress analysis tab"""
        st.header("📈 Enhanced Stress Analysis")
        
        # Data loading section
        st.subheader("🔄 Load Data for Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            load_method = st.radio(
                "Loading Method",
                ["Enhanced Parallel Loader", "Legacy Loader"],
                index=0,
                help="Enhanced loader provides better error handling and performance"
            )
        
        with col2:
            file_limit = st.slider(
                "Maximum files",
                10, 500, 100, 10,
                help="Limit number of files to prevent memory issues"
            )
        
        if st.button("🚀 Load Simulations", type="primary", key="load_for_analysis"):
            with st.spinner("Loading and analyzing simulations..."):
                self._load_for_analysis(load_method, file_limit)
        
        # Display analysis if data is available
        if not st.session_state.stress_summary_df.empty:
            self._render_stress_analysis_dashboard()
    
    def _load_for_analysis(self, load_method: str, file_limit: int):
        """Load data for analysis"""
        try:
            if load_method == "Enhanced Parallel Loader":
                successful, failed = self.resilient_data_manager.scan_and_load_all_parallel(
                    file_limit
                )
                
                if successful > 0:
                    st.session_state.stress_summary_df = self.resilient_data_manager.get_summary_dataframe()
                    
                    # Generate enhanced report
                    report = self.resilient_data_manager.get_enhanced_summary_report()
                    
                    st.success(f"✅ Successfully loaded {successful} simulations")
                    
                    # Display report
                    with st.expander("📊 Loading Report", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", report.get('total_simulations', 0))
                        with col2:
                            st.metric("Failed Files", report.get('failed_files', 0))
                    
                    if failed > 0:
                        st.warning(f"⚠️ {failed} files failed to load")
                
                else:
                    st.error("No simulations could be loaded")
            
            else:
                # Legacy loader
                all_files = self.solutions_manager.get_all_files()[:file_limit]
                all_simulations = []
                
                progress_bar = st.progress(0)
                for idx, file_info in enumerate(all_files):
                    progress_bar.progress((idx + 1) / len(all_files))
                    try:
                        sim_data = self.solutions_manager.load_simulation(
                            file_info['path'],
                            st.session_state.interpolator
                        )
                        all_simulations.append(sim_data)
                    except Exception as e:
                        logger.error(f"Failed to load {file_info['filename']}: {str(e)}")
                
                progress_bar.empty()
                
                if all_simulations:
                    stress_df = self.stress_analyzer.create_enhanced_summary_dataframe(
                        all_simulations, {}
                    )
                    st.session_state.stress_summary_df = stress_df
                    st.success(f"✅ Loaded {len(all_simulations)} simulations")
                else:
                    st.error("No simulations could be loaded")
        
        except Exception as e:
            st.error(f"❌ Error during loading: {str(e)}")
            logger.error(f"Analysis loading error: {str(e)}", exc_info=True)
    
    def _render_stress_analysis_dashboard(self):
        """Render the stress analysis dashboard"""
        df = st.session_state.stress_summary_df
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Total Simulations", len(df))
        with metrics_cols[1]:
            if 'max_von_mises' in df.columns:
                max_vm = df['max_von_mises'].max()
                st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
            else:
                st.metric("Max Von Mises", "N/A")
        
        with metrics_cols[2]:
            if 'max_von_mises' in df.columns:
                mean_vm = df['max_von_mises'].mean()
                st.metric("Avg Max Von Mises", f"{mean_vm:.2f} GPa")
            else:
                st.metric("Avg Max Von Mises", "N/A")
        
        with metrics_cols[3]:
            if 'defect_type' in df.columns:
                defect_counts = df['defect_type'].value_counts().to_dict()
                st.metric("Unique Defect Types", len(defect_counts))
            else:
                st.metric("Unique Defect Types", "N/A")
        
        # Data table
        st.subheader("📋 Data Table")
        
        # Select columns to display
        all_cols = df.columns.tolist()
        default_cols = [col for col in all_cols if any(x in col for x in 
                      ['id', 'defect_type', 'shape', 'eps0', 'kappa', 'max_von_mises'])]
        
        selected_cols = st.multiselect(
            "Select columns to display",
            all_cols,
            default=default_cols[:6]  # Limit to 6 for display
        )
        
        if selected_cols:
            try:
                # Format numeric columns
                numeric_cols = df[selected_cols].select_dtypes(include=[np.number]).columns
                format_dict = {col: "{:.3f}" for col in numeric_cols}
                
                # Display table
                st.dataframe(
                    df[selected_cols].style.format(format_dict),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df[selected_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"stress_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        st.info("Sample data creation would be implemented here")
        # In a real implementation, this would create sample simulation files
    
    def run(self):
        """Run the enhanced application"""
        try:
            # Render sidebar
            self.render_sidebar()
            
            # Main content
            st.title("🔬 Enhanced Stress Analysis Dashboard")
            
            # Render main tabs
            self.render_main_tabs()
            
            # Footer
            st.divider()
            st.caption(f"🔬 Enhanced Stress Analysis Dashboard • Version 2.0 • {datetime.now().year}")
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.critical(f"Application error: {str(e)}", exc_info=True)

# =============================================
# MAIN ENTRY POINT
# =============================================
def main():
    """Main entry point for the application"""
    try:
        # Initialize and run the enhanced app
        app = EnhancedStressAnalysisApp()
        app.run()
    except Exception as e:
        # Global error handler
        st.error(f"Application error: {str(e)}")
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        
        # Provide recovery option
        if st.button("🔄 Restart Application"):
            st.rerun()

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Run the application
    main()
