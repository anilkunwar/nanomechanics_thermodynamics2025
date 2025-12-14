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

# =============================================
# CONFIGURATION AND CONSTANTS
# =============================================
class AppConfig:
    """Centralized configuration for the application"""
    
    # Directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
    
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
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [cls.NUMERICAL_SOLUTIONS_DIR, cls.LOG_DIR, cls.CACHE_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# =============================================
# LOGGING AND ERROR HANDLING
# =============================================
class AppLogger:
    """Centralized logging with rotation and multiple handlers"""
    
    def __init__(self, name="stress_analysis_app"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = os.path.join(AppConfig.LOG_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        return self.logger

# Initialize logger
logger = AppLogger().get_logger()

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
                    logger.error(f"Error in {func.__name__}: {str(e)}", 
                               exc_info=True)
                if default_return is not None:
                    return default_return
                else:
                    # Re-raise for critical functions
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
# ENHANCED SECURE FILE LOADER
# =============================================
class SecureFileLoader:
    """Secure file loading with validation and sanitization"""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate file extension against allowed list"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in AppConfig.ALLOWED_FILE_EXTENSIONS
    
    @staticmethod
    def validate_file_size(file_content: bytes) -> bool:
        """Validate file size"""
        return len(file_content) <= AppConfig.MAX_UPLOAD_SIZE
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove directory path
        basename = os.path.basename(filename)
        # Remove potentially dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        return ''.join(c for c in basename if c in safe_chars)
    
    @staticmethod
    @handle_errors(default_return=None, log_error=True)
    def safe_load_pickle(file_content: bytes):
        """Safely load pickle file with restricted classes"""
        buffer = BytesIO(file_content)
        
        # Custom unpickler with restricted classes
        class RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Only allow safe classes
                allowed_modules = {
                    'numpy': ['ndarray', 'dtype'],
                    'builtins': ['list', 'dict', 'tuple', 'int', 'float', 'str'],
                    'collections': ['OrderedDict'],
                    'pandas': ['DataFrame', 'Series'],
                }
                
                if module in allowed_modules and name in allowed_modules[module]:
                    return super().find_class(module, name)
                # Forbid everything else
                raise pickle.UnpicklingError(f"Forbidden class: {module}.{name}")
        
        return RestrictedUnpickler(buffer).load()
    
    @staticmethod
    @handle_errors(default_return=None, log_error=True)
    def safe_load_torch(file_content: bytes):
        """Safely load torch file with weights_only"""
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)
    
    @staticmethod
    def load_file(file_content: bytes, file_format: str) -> Dict[str, Any]:
        """Load file with format-specific secure loader"""
        loaders = {
            'pkl': SecureFileLoader.safe_load_pickle,
            'pt': SecureFileLoader.safe_load_torch,
            'h5': lambda x: h5py.File(BytesIO(x), 'r'),
            'npz': lambda x: np.load(BytesIO(x), allow_pickle=False),
            'json': lambda x: json.loads(x.decode('utf-8'))
        }
        
        if file_format not in loaders:
            raise ValueError(f"Unsupported format: {file_format}")
        
        return loaders[file_format](file_content)

# =============================================
# ENHANCED RESILIENT DATA MANAGER
# =============================================
class EnhancedResilientDataManager:
    """Enhanced data manager with parallel loading and caching"""
    
    def __init__(self, solutions_dir: str):
        self.solutions_dir = Path(solutions_dir)
        self.valid_simulations = []
        self.failed_files_log = []
        self.summary_df = pd.DataFrame()
        self._lock = threading.Lock()
        self._cache = {}
        
    @handle_errors(default_return=([], []), log_error=True)
    def scan_and_load_all_parallel(self, file_limit: int = 100) -> Tuple[int, int]:
        """Scan and load files in parallel with progress tracking"""
        self.valid_simulations.clear()
        self.failed_files_log.clear()
        
        # Find all files
        all_files = []
        for ext in AppConfig.ALLOWED_FILE_EXTENSIONS:
            pattern = f"*{ext}" if ext.startswith('.') else f"*.{ext}"
            files = list(self.solutions_dir.glob(pattern))[:file_limit]
            all_files.extend(files)
        
        if not all_files:
            return 0, 0
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_queue = queue.Queue()
        
        def process_file(file_path: Path) -> Optional[Dict]:
            """Process a single file"""
            try:
                # Check cache first
                cache_key = f"{file_path}_{file_path.stat().st_mtime}"
                if cache_key in self._cache:
                    return self._cache[cache_key]
                
                # Determine format
                if file_path.suffix.lower() in ['.pkl', '.pickle']:
                    loader = SafeSimulationLoader.validate_and_load_pkl
                elif file_path.suffix.lower() in ['.pt', '.pth']:
                    loader = EnhancedResilientDataManager._load_pt_file
                else:
                    return None
                
                # Load file
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                sim_data, message = loader(file_content, str(file_path))
                
                if sim_data:
                    # Cache the result
                    self._cache[cache_key] = sim_data
                
                return sim_data
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                return None
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=min(AppConfig.MAX_CONCURRENT_WORKERS, len(all_files))) as executor:
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
        
        return len(self.valid_simulations), len(self.failed_files_log)
    
    @staticmethod
    @handle_errors(default_return=(None, "Error loading file"), log_error=True)
    def _load_pt_file(file_content: bytes, file_path: str):
        """Load PyTorch file with enhanced validation"""
        try:
            data = torch.load(BytesIO(file_content), 
                            map_location=torch.device('cpu'), 
                            weights_only=True)
            
            # Enhanced validation
            if not isinstance(data, dict):
                return None, "Data must be a dictionary"
            
            # Standardize structure
            standardized = {
                'params': data.get('params', {}),
                'stress_fields': data.get('stress_fields', {}),
                'history': data.get('history', []),
                'metadata': data.get('metadata', {}),
                'file_source': file_path,
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
                        else:
                            return None, f"Stress field {key} is not a numpy array or tensor"
            
            return standardized, "Success"
            
        except Exception as e:
            return None, f"Failed to load .pt file: {str(e)[:100]}"
    
    def _create_summary_dataframe(self):
        """Create enhanced summary dataframe with more metrics"""
        if not self.valid_simulations:
            self.summary_df = pd.DataFrame()
            return
        
        rows = []
        for idx, sim in enumerate(self.valid_simulations):
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
            try:
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
                    
            except Exception as e:
                logger.warning(f"Error computing metrics for simulation {idx}: {str(e)}")
            
            rows.append(row)
        
        self.summary_df = pd.DataFrame(rows)
    
    def get_enhanced_summary_report(self) -> Dict[str, Any]:
        """Generate enhanced summary report with statistics"""
        if self.summary_df.empty:
            return {}
        
        report = {
            'total_simulations': len(self.summary_df),
            'defect_type_counts': self.summary_df['defect_type'].value_counts().to_dict(),
            'shape_counts': self.summary_df['shape'].value_counts().to_dict(),
            'failed_files': len(self.failed_files_log),
            'parameter_ranges': {
                'eps0': {
                    'min': float(self.summary_df['eps0'].min()),
                    'max': float(self.summary_df['eps0'].max()),
                    'mean': float(self.summary_df['eps0'].mean())
                },
                'kappa': {
                    'min': float(self.summary_df['kappa'].min()),
                    'max': float(self.summary_df['kappa'].max()),
                    'mean': float(self.summary_df['kappa'].mean())
                }
            }
        }
        
        # Add stress statistics if available
        stress_cols = [col for col in self.summary_df.columns if 'von_mises' in col or 'hydrostatic' in col]
        for col in stress_cols:
            if col in self.summary_df.columns:
                report[f'{col}_stats'] = {
                    'min': float(self.summary_df[col].min()),
                    'max': float(self.summary_df[col].max()),
                    'mean': float(self.summary_df[col].mean()),
                    'std': float(self.summary_df[col].std())
                }
        
        return report

# =============================================
# ENHANCED MULTI-TARGET PREDICTION MANAGER
# =============================================
class EnhancedMultiTargetPredictionManager:
    """Enhanced manager with flexible parameter grid generation"""
    
    @staticmethod
    def parse_custom_angles(angle_input: str) -> List[float]:
        """
        Parse custom angles from various input formats
        
        Supports:
        - Comma-separated: "0, 15, 30, 45"
        - Range with steps: "0:90:10" (start:stop:step)
        - Mixed: "0, 15:45:15, 60, 90"
        """
        angles = []
        
        # Split by comma first
        parts = [p.strip() for p in angle_input.split(',')]
        
        for part in parts:
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
                angles.extend(np.arange(start, stop + step/2, step).tolist())
            else:
                # Single angle
                try:
                    angles.append(float(part))
                except ValueError:
                    raise ValueError(f"Invalid angle: {part}")
        
        # Remove duplicates and sort
        return sorted(list(set(angles)))
    
    @staticmethod
    def create_flexible_parameter_grid(base_params: Dict, 
                                     configs: Dict[str, Dict]) -> List[Dict]:
        """
        Create parameter grid with flexible configuration
        
        Args:
            base_params: Base parameter dictionary
            configs: Dictionary with parameter configurations
                Example: {
                    'eps0': {'type': 'range', 'min': 0.5, 'max': 2.0, 'steps': 10},
                    'kappa': {'type': 'values', 'values': [0.2, 0.4, 0.6, 0.8]},
                    'theta': {'type': 'custom', 'input': '0, 15, 30, 45'},
                    'defect_type': {'type': 'categorical', 'values': ['ISF', 'ESF']}
                }
        
        Returns:
            List of parameter dictionaries
        """
        param_values = {}
        
        for param_name, config in configs.items():
            config_type = config.get('type', 'value')
            
            if config_type == 'range':
                # Linear range
                min_val = config.get('min', 0)
                max_val = config.get('max', 1)
                steps = config.get('steps', 10)
                param_values[param_name] = np.linspace(min_val, max_val, steps).tolist()
            
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
    
    @staticmethod
    def get_orientation_from_angle(angle_deg: float) -> str:
        """Convert angle to orientation string with custom support"""
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
    
    @staticmethod
    @handle_errors(default_return={}, log_error=True)
    def batch_predict_parallel(source_simulations: List[Dict],
                             target_params_list: List[Dict],
                             interpolator: 'SpatialLocalityAttentionInterpolator',
                             max_workers: int = 4) -> Dict[str, Any]:
        """
        Perform batch predictions in parallel
        
        Args:
            source_simulations: List of source simulation data
            target_params_list: List of target parameter dictionaries
            interpolator: Interpolator instance
            max_workers: Number of parallel workers
        
        Returns:
            Dictionary with predictions
        """
        predictions = {}
        
        # Prepare source data once
        source_param_vectors = []
        source_stress_data = []
        
        for sim_data in source_simulations:
            param_vector, _ = interpolator.compute_parameter_vector(sim_data)
            source_param_vectors.append(param_vector)
            
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                source_stress_data.append(stress_components)
        
        source_param_vectors = np.array(source_param_vectors)
        source_stress_data = np.array(source_stress_data)
        
        # Function to predict single target
        def predict_single_target(target_idx: int, target_params: Dict) -> Dict:
            try:
                target_vector, _ = interpolator.compute_parameter_vector(
                    {'params': target_params}
                )
                
                # Calculate distances and weights
                distances = np.sqrt(np.sum((source_param_vectors - target_vector) ** 2, axis=1))
                weights = np.exp(-0.5 * (distances / 0.3) ** 2)
                weights = weights / (np.sum(weights) + 1e-8)
                
                # Weighted combination
                weighted_stress = np.sum(
                    source_stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis], 
                    axis=0
                )
                
                return {
                    'sigma_hydro': weighted_stress[0],
                    'sigma_mag': weighted_stress[1],
                    'von_mises': weighted_stress[2],
                    'predicted': True,
                    'target_params': target_params,
                    'attention_weights': weights,
                    'target_index': target_idx
                }
            except Exception as e:
                logger.error(f"Error predicting target {target_idx}: {str(e)}")
                return None
        
        # Parallel prediction
        with ThreadPoolExecutor(max_workers=min(max_workers, len(target_params_list))) as executor:
            # Submit all prediction tasks
            future_to_idx = {
                executor.submit(predict_single_target, idx, params): idx
                for idx, params in enumerate(target_params_list)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    prediction = future.result()
                    if prediction:
                        predictions[f"target_{idx:03d}"] = prediction
                except Exception as e:
                    logger.error(f"Failed to get prediction for target {idx}: {str(e)}")
        
        return predictions

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
                raise ValueError("Empty stress fields")
            
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
                    metrics[f'{component_name}_p{p}'] = float(np.nanpercentile(stress_data, p))
                
                # Area fractions
                total_pixels = stress_data.size
                if total_pixels > 0:
                    metrics[f'{component_name}_area_above_mean'] = float(np.sum(stress_data > metrics[f'{component_name}_mean']) / total_pixels * 100)
                    metrics[f'{component_name}_area_above_p95'] = float(np.sum(stress_data > metrics[f'{component_name}_p95']) / total_pixels * 100)
            
            # Special handling for key components
            if 'sigma_hydro' in stress_fields:
                hydro_data = stress_fields['sigma_hydro']
                valid_hydro = hydro_data[~np.isnan(hydro_data)]
                
                if len(valid_hydro) > 0:
                    metrics.update({
                        'hydrostatic_tension_area': float(np.sum(hydro_data > 0) / hydro_data.size * 100),
                        'hydrostatic_compression_area': float(np.sum(hydro_data < 0) / hydro_data.size * 100),
                        'hydrostatic_max_tension': float(np.nanmax(np.where(hydro_data > 0, hydro_data, np.nan))),
                        'hydrostatic_max_compression': float(np.nanmin(np.where(hydro_data < 0, hydro_data, np.nan))),
                        'hydrostatic_triaxiality': float(np.nanmean(np.abs(hydro_data)) / (metrics.get('von_mises_mean', 1) + 1e-10))
                    })
            
            if 'von_mises' in stress_fields and 'sigma_hydro' in stress_fields:
                vm_data = stress_fields['von_mises']
                hydro_data = stress_fields['sigma_hydro']
                
                if vm_data.size > 0 and hydro_data.size > 0:
                    # Lode parameter
                    s1 = stress_fields.get('sigma_1')
                    s2 = stress_fields.get('sigma_2')
                    s3 = stress_fields.get('sigma_3')
                    
                    if all(s is not None for s in [s1, s2, s3]):
                        try:
                            # Avoid division by zero
                            denom = (s1 - s3) + 1e-10
                            lode = (2 * s2 - s1 - s3) / denom
                            metrics['lode_parameter_mean'] = float(np.nanmean(lode))
                            metrics['lode_parameter_std'] = float(np.nanstd(lode))
                        except:
                            pass
            
            # Stress gradient metrics
            if 'sigma_hydro' in stress_fields:
                hydro_grad = EnhancedStressAnalysisManager.compute_stress_gradients(
                    stress_fields['sigma_hydro']
                )
                metrics['hydro_grad_max'] = float(np.nanmax(hydro_grad))
                metrics['hydro_grad_mean'] = float(np.nanmean(hydro_grad))
        
        except Exception as e:
            logger.error(f"Error computing stress metrics: {str(e)}")
            # Return basic metrics if available
            pass
        
        return metrics
    
    @staticmethod
    def compute_stress_gradients(stress_field: np.ndarray) -> np.ndarray:
        """Compute stress gradients"""
        grad_y, grad_x = np.gradient(stress_field)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    @staticmethod
    def create_enhanced_summary_dataframe(source_simulations: List[Dict],
                                        predictions: Dict = None) -> pd.DataFrame:
        """
        Create enhanced summary dataframe with comprehensive metrics
        """
        rows = []
        
        # Process source simulations
        for i, sim_data in enumerate(source_simulations):
            params = sim_data.get('params', {})
            stress_fields = sim_data.get('stress_fields', {})
            
            # Compute comprehensive metrics
            metrics = EnhancedStressAnalysisManager.compute_comprehensive_stress_metrics(stress_fields)
            
            # Create row
            row = {
                'id': f'source_{i:03d}',
                'type': 'source',
                'defect_type': params.get('defect_type', 'Unknown'),
                'shape': params.get('shape', 'Unknown'),
                'orientation': params.get('orientation', 'Unknown'),
                'eps0': params.get('eps0', np.nan),
                'kappa': params.get('kappa', np.nan),
                'theta': params.get('theta', np.nan),
                'theta_deg': np.rad2deg(params.get('theta', 0)),
                'source_file': sim_data.get('file_source', ''),
                **metrics
            }
            rows.append(row)
        
        # Process predictions
        if predictions:
            for pred_key, pred_data in predictions.items():
                if isinstance(pred_data, dict) and 'target_params' in pred_data:
                    params = pred_data.get('target_params', {})
                    
                    # Extract stress fields from prediction
                    pred_stress_fields = {}
                    for key in ['sigma_hydro', 'sigma_mag', 'von_mises']:
                        if key in pred_data:
                            pred_stress_fields[key] = pred_data[key]
                    
                    # Compute metrics
                    metrics = EnhancedStressAnalysisManager.compute_comprehensive_stress_metrics(pred_stress_fields)
                    
                    # Create row
                    row = {
                        'id': pred_key,
                        'type': 'prediction',
                        'defect_type': params.get('defect_type', 'Unknown'),
                        'shape': params.get('shape', 'Unknown'),
                        'orientation': params.get('orientation', 'Unknown'),
                        'eps0': params.get('eps0', np.nan),
                        'kappa': params.get('kappa', np.nan),
                        'theta': params.get('theta', np.nan),
                        'theta_deg': np.rad2deg(params.get('theta', 0)),
                        'prediction_key': pred_key,
                        **metrics
                    }
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            
            # Add derived columns
            if 'von_mises_max' in df.columns and 'sigma_hydro_max' in df.columns:
                df['stress_ratio_vm_hydro_max'] = df['von_mises_max'] / (df['sigma_hydro_max'].abs() + 1e-10)
            
            if 'hydrostatic_triaxiality' in df.columns:
                df['stress_state'] = df['hydrostatic_triaxiality'].apply(
                    lambda x: 'Shear' if x < 0.2 else 'Tension' if x < 0.8 else 'Compression'
                )
            
            return df
        else:
            return pd.DataFrame()

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
        """
        Create robust box plot with validation
        """
        if df.empty or len(df) < 2:
            return None
        
        # Validate inputs
        valid_value_cols = [col for col in value_columns if col in df.columns]
        if not valid_value_cols:
            return None
        
        if group_by_column not in df.columns:
            return None
        
        # Limit number of categories for performance
        unique_groups = df[group_by_column].unique()
        if len(unique_groups) > max_categories:
            # Keep most frequent categories
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
                            boxmean='sd'  # Show mean and standard deviation
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
    
    @staticmethod
    @handle_errors(default_return=None, log_error=True)
    def create_correlation_matrix(df: pd.DataFrame,
                                numeric_columns: List[str] = None,
                                title: str = "Correlation Matrix") -> Optional[go.Figure]:
        """
        Create correlation matrix heatmap
        """
        if df.empty or len(df) < 2:
            return None
        
        # Select numeric columns
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        if len(numeric_columns) < 2:
            return None
        
        # Compute correlation matrix
        corr_matrix = df[numeric_columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverinfo="x+y+z"
        ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            height=500,
            width=600
        )
        
        return fig
    
    @staticmethod
    @handle_errors(default_return=None, log_error=True)
    def create_interactive_stress_plot(stress_field: np.ndarray,
                                     extent: Tuple[float, float, float, float],
                                     title: str = "Stress Field",
                                     colormap: str = 'viridis') -> go.Figure:
        """
        Create interactive stress field plot
        """
        if stress_field is None or stress_field.size == 0:
            return None
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=stress_field,
            colorscale=colormap,
            colorbar=dict(title="Stress (GPa)")
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis=dict(
                title="x (nm)",
                scaleanchor="y",
                constrain="domain"
            ),
            yaxis=dict(
                title="y (nm)",
                constrain="domain"
            ),
            height=600,
            width=600
        )
        
        # Add contour lines
        if stress_field.shape[0] <= 100:  # Performance consideration
            try:
                # Create contour
                x = np.linspace(extent[0], extent[1], stress_field.shape[1])
                y = np.linspace(extent[2], extent[3], stress_field.shape[0])
                
                fig.add_trace(
                    go.Contour(
                        z=stress_field,
                        x=x,
                        y=y,
                        contours=dict(
                            coloring='lines',
                            showlabels=True,
                            labelfont=dict(size=8, color='white')
                        ),
                        line=dict(width=1),
                        colorscale=[[0, 'rgba(0,0,0,0.5)'], [1, 'rgba(0,0,0,0.5)']],
                        showscale=False
                    )
                )
            except:
                pass
        
        return fig

# =============================================
# REFACTORED MAIN APPLICATION
# =============================================
class EnhancedStressAnalysisApp:
    """Refactored main application with improved architecture"""
    
    def __init__(self):
        # Initialize configuration
        AppConfig.ensure_directories()
        
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
            'operation_status': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _init_managers(self):
        """Initialize all manager instances"""
        self.solutions_manager = NumericalSolutionsManager(AppConfig.NUMERICAL_SOLUTIONS_DIR)
        self.resilient_data_manager = EnhancedResilientDataManager(AppConfig.NUMERICAL_SOLUTIONS_DIR)
        self.stress_analyzer = EnhancedStressAnalysisManager()
        self.visualization_manager = EnhancedVisualizationManager()
        self.multi_target_manager = EnhancedMultiTargetPredictionManager()
        
        # Lazy initialization of interpolator
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
    
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
                
                plot_quality = st.select_slider(
                    "Plot quality",
                    options=['Low', 'Medium', 'High'],
                    value='Medium'
                )
            
            # Advanced settings
            with st.expander("⚡ Advanced", expanded=False):
                enable_caching = st.checkbox("Enable caching", value=True)
                log_level = st.selectbox(
                    "Log level",
                    ['INFO', 'DEBUG', 'WARNING', 'ERROR'],
                    index=0
                )
            
            st.divider()
            
            # System info
            st.caption(f"**Files loaded:** {len(st.session_state.source_simulations)}")
            st.caption(f"**Directory:** {AppConfig.NUMERICAL_SOLUTIONS_DIR}")
            
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
        
        # Scan directory
        all_files = self.solutions_manager.get_all_files()
        
        if not all_files:
            st.info(f"No files found in `{AppConfig.NUMERICAL_SOLUTIONS_DIR}`")
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
                # Validate file
                if not SecureFileLoader.validate_file_extension(uploaded_file.name):
                    st.warning(f"Skipped {uploaded_file.name}: Invalid extension")
                    continue
                
                # Read file content
                file_content = uploaded_file.getvalue()
                
                if not SecureFileLoader.validate_file_size(file_content):
                    st.warning(f"Skipped {uploaded_file.name}: File too large")
                    continue
                
                # Determine format
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                format_map = {
                    '.pkl': 'pkl',
                    '.pt': 'pt',
                    '.h5': 'h5',
                    '.hdf5': 'h5',
                    '.npz': 'npz',
                    '.json': 'json'
                }
                file_format = format_map.get(ext, 'auto')
                
                # Load file securely
                raw_data = SecureFileLoader.load_file(file_content, file_format)
                
                # Standardize data
                sim_data = st.session_state.interpolator._standardize_data(
                    raw_data, file_format, uploaded_file.name
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
    
    def render_tab_configure_multiple(self):
        """Render the multiple target configuration tab"""
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
        
        with st.expander("ε* Configuration", expanded=True):
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
                try:
                    eps0_values = EnhancedMultiTargetPredictionManager.parse_custom_angles(eps0_custom)
                    eps0_config = {'type': 'custom', 'input': eps0_custom}
                except:
                    st.error("Invalid format")
                    eps0_config = {'type': 'value', 'value': 1.414}
        
        # Similar configuration for kappa and theta...
        # [Additional configuration sections for kappa and theta]
        
        # Generate parameter grid
        if st.button("🔄 Generate Parameter Grid", type="primary"):
            with st.spinner("Generating parameter grid..."):
                try:
                    # Build configs dictionary
                    configs = {
                        'eps0': eps0_config,
                        # Add kappa and theta configs similarly
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
        
        # Interactive table with selection
        st.subheader("📋 Generated Parameter Grid")
        
        # Add selection column
        df_grid['Select'] = False
        
        # Use st.data_editor for interactive selection
        edited_df = st.data_editor(
            df_grid,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select rows for prediction",
                    default=False,
                )
            },
            disabled=["ID", "Defect", "Shape", "ε*", "κ", "Orientation", "θ°"],
            hide_index=True,
            use_container_width=True
        )
        
        # Filter selected parameters
        selected_indices = edited_df[edited_df['Select']].index.tolist()
        if selected_indices:
            st.info(f"Selected {len(selected_indices)} parameter sets")
            
            if st.button("🎯 Predict Selected Only", type="secondary"):
                selected_params = [param_grid[i] for i in selected_indices]
                st.session_state.multi_target_params = selected_params
                st.success(f"Updated to {len(selected_params)} selected parameters")
    
    def render_tab_stress_analysis(self):
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
                            st.metric("Defect Types", len(report.get('defect_type_counts', {})))
                        with col3:
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
            max_vm = df['von_mises_max'].max() if 'von_mises_max' in df.columns else 0
            st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
        with metrics_cols[2]:
            mean_vm = df['von_mises_mean'].mean() if 'von_mises_mean' in df.columns else 0
            st.metric("Avg Von Mises", f"{mean_vm:.2f} GPa")
        with metrics_cols[3]:
            max_hydro = df['sigma_hydro_max'].max() if 'sigma_hydro_max' in df.columns else 0
            st.metric("Max Hydrostatic", f"{max_hydro:.2f} GPa")
        
        # Visualization tabs
        viz_tabs = st.tabs([
            "📈 Distributions",
            "🔗 Correlations",
            "🌡️ Stress Analysis",
            "📋 Data Table"
        ])
        
        with viz_tabs[0]:
            self._render_distribution_plots(df)
        
        with viz_tabs[1]:
            self._render_correlation_analysis(df)
        
        with viz_tabs[2]:
            self._render_detailed_stress_analysis(df)
        
        with viz_tabs[3]:
            self._render_data_table(df)
    
    def _render_distribution_plots(self, df: pd.DataFrame):
        """Render distribution plots"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot of von Mises by defect type
            fig = self.visualization_manager.create_robust_box_plot(
                df,
                value_columns=['von_mises_max', 'von_mises_mean', 'von_mises_p95'],
                group_by_column='defect_type',
                title="Von Mises Stress by Defect Type"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot
            if 'eps0' in df.columns and 'von_mises_max' in df.columns:
                fig = px.scatter(
                    df,
                    x='eps0',
                    y='von_mises_max',
                    color='defect_type',
                    size='kappa' if 'kappa' in df.columns else None,
                    hover_data=['shape', 'orientation', 'theta_deg'],
                    title="Von Mises vs ε*",
                    trendline="lowess"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_analysis(self, df: pd.DataFrame):
        """Render correlation analysis"""
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Let user select columns
        selected_cols = st.multiselect(
            "Select columns for correlation",
            numeric_cols,
            default=[col for col in numeric_cols if any(x in col for x in ['von_mises', 'sigma_hydro', 'eps0', 'kappa'])]
        )
        
        if len(selected_cols) >= 2:
            fig = self.visualization_manager.create_correlation_matrix(
                df, selected_cols, "Correlation Matrix"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Pair plot for selected columns
        if len(selected_cols) >= 2 and len(df) < 100:  # Performance consideration
            if st.checkbox("Show pair plot (for small datasets)"):
                fig = px.scatter_matrix(
                    df,
                    dimensions=selected_cols[:5],  # Limit to 5 for performance
                    color='defect_type' if 'defect_type' in df.columns else None,
                    title="Pair Plot"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_detailed_stress_analysis(self, df: pd.DataFrame):
        """Render detailed stress analysis"""
        st.subheader("Detailed Stress Metrics")
        
        # Select simulation for detailed view
        sim_options = df['id'].tolist()
        selected_sim = st.selectbox("Select simulation for detailed view", sim_options)
        
        if selected_sim:
            sim_data = df[df['id'] == selected_sim].iloc[0]
            
            # Display stress metrics
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Max Von Mises", f"{sim_data.get('von_mises_max', 0):.2f} GPa")
                st.metric("Mean Von Mises", f"{sim_data.get('von_mises_mean', 0):.2f} GPa")
                st.metric("95th %ile", f"{sim_data.get('von_mises_p95', 0):.2f} GPa")
            
            with cols[1]:
                st.metric("Max Hydrostatic", f"{sim_data.get('sigma_hydro_max', 0):.2f} GPa")
                st.metric("Mean |Hydrostatic|", f"{sim_data.get('sigma_hydro_mean', 0):.2f} GPa")
                if 'hydrostatic_triaxiality' in sim_data:
                    st.metric("Triaxiality", f"{sim_data.get('hydrostatic_triaxiality', 0):.3f}")
            
            with cols[2]:
                if 'hydrostatic_tension_area' in sim_data:
                    st.metric("Tension Area", f"{sim_data.get('hydrostatic_tension_area', 0):.1f}%")
                if 'hydrostatic_compression_area' in sim_data:
                    st.metric("Compression Area", f"{sim_data.get('hydrostatic_compression_area', 0):.1f}%")
                if 'stress_ratio_vm_hydro_max' in sim_data:
                    st.metric("VM/Hydro Ratio", f"{sim_data.get('stress_ratio_vm_hydro_max', 0):.3f}")
    
    def _render_data_table(self, df: pd.DataFrame):
        """Render interactive data table"""
        st.subheader("📋 Complete Data Table")
        
        # Column selection
        all_cols = df.columns.tolist()
        default_cols = [col for col in all_cols if any(x in col for x in ['id', 'defect', 'shape', 'eps0', 'kappa', 'von_mises', 'sigma_hydro'])]
        
        selected_cols = st.multiselect(
            "Select columns to display",
            all_cols,
            default=default_cols
        )
        
        if selected_cols:
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
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        st.info("Creating sample data...")
        # This would create sample simulation files
        # Implementation depends on your specific data format
        pass
    
    def run(self):
        """Run the enhanced application"""
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        st.title("🔬 Enhanced Stress Analysis Dashboard")
        
        # Navigation tabs
        tabs = st.tabs([
            "📥 Load Data",
            "🎯 Single Target",
            "🎯 Multiple Targets",
            "🚀 Predict",
            "📊 Results",
            "💾 Export",
            "📈 Analysis"
        ])
        
        # Render each tab
        with tabs[0]:
            self.render_tab_load_data()
        
        with tabs[1]:
            # Single target configuration (simplified)
            st.header("🎯 Single Target Configuration")
            if len(st.session_state.source_simulations) >= 2:
                self._render_single_target_config()
            else:
                st.warning("Please load at least 2 source simulations first")
        
        with tabs[2]:
            self.render_tab_configure_multiple()
        
        with tabs[3]:
            # Prediction tab
            self._render_prediction_tab()
        
        with tabs[4]:
            # Results tab
            self._render_results_tab()
        
        with tabs[5]:
            # Export tab
            self._render_export_tab()
        
        with tabs[6]:
            # Enhanced analysis tab
            self.render_tab_stress_analysis()
        
        # Footer
        st.divider()
        st.caption(f"🔬 Enhanced Stress Analysis Dashboard • Version 2.0 • {datetime.now().year}")

# =============================================
# MAIN ENTRY POINT
# =============================================
if __name__ == "__main__":
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
