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
import torch.serialization  # Import torch serialization for safe globals
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# =============================================
# GLOBAL PYTORCH SECURITY FIXES
# =============================================
def setup_torch_security_globals():
    """
    Configure safe globals for PyTorch serialization to handle NumPy objects
    This fixes the 'weights_only=True' security restrictions in PyTorch 2.6+
    """
    try:
        import numpy as np
        import torch.serialization
        
        # List of safe globals to add for NumPy compatibility
        safe_globals = []
        
        # Try to add various NumPy scalar types that might appear in files
        try:
            # NumPy 2.0+ style
            safe_globals.extend([
                np._core.multiarray.scalar,  # Internal scalar type
                np.core.multiarray.scalar,   # Public alias
                np.dtype,                    # Data type objects
                np.ndarray,                  # Array objects
                np.bool_,                    # Boolean type
                np.int8, np.int16, np.int32, np.int64,  # Integer types
                np.uint8, np.uint16, np.uint32, np.uint64,
                np.float16, np.float32, np.float64,     # Float types
                np.complex64, np.complex128,
                np.str_, np.bytes_,                     # String types
                np.object_,                            # Object type
            ])
        except AttributeError:
            pass
        
        # Try to add dtype classes (NumPy 2.0+)
        try:
            from numpy.dtypes import Float64DType, Float32DType, Int64DType, Int32DType
            safe_globals.extend([
                Float64DType, Float32DType, Int64DType, Int32DType
            ])
        except ImportError:
            pass
        
        # Add the safe globals to torch serialization
        if safe_globals:
            torch.serialization.add_safe_globals(safe_globals)
            st.sidebar.info(f"✅ Added {len(safe_globals)} safe globals for PyTorch loading")
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not setup torch security globals: {str(e)}")

# Initialize torch security settings
setup_torch_security_globals()

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")

if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# STRESS ANALYSIS MANAGER
# =============================================
class StressAnalysisManager:
    """Manager for stress value analysis and visualization"""
    
    @staticmethod
    def compute_max_stress_values(stress_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute maximum stress values from stress fields
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            
        Returns:
            Dictionary with max stress values
        """
        results = {}
        
        # Hydrostatic stress - take absolute maximum
        if 'sigma_hydro' in stress_fields:
            hydro_data = stress_fields['sigma_hydro']
            results['max_abs_hydrostatic'] = float(np.max(np.abs(hydro_data)))
            results['max_hydrostatic'] = float(np.max(hydro_data))
            results['min_hydrostatic'] = float(np.min(hydro_data))
            results['mean_abs_hydrostatic'] = float(np.mean(np.abs(hydro_data)))
        
        # Stress magnitude - take maximum
        if 'sigma_mag' in stress_fields:
            mag_data = stress_fields['sigma_mag']
            results['max_stress_magnitude'] = float(np.max(mag_data))
            results['mean_stress_magnitude'] = float(np.mean(mag_data))
        
        # Von Mises stress - take maximum
        if 'von_mises' in stress_fields:
            vm_data = stress_fields['von_mises']
            results['max_von_mises'] = float(np.max(vm_data))
            results['mean_von_mises'] = float(np.mean(vm_data))
            results['min_von_mises'] = float(np.min(vm_data))
        
        # Principal stresses if available
        if 'sigma_1' in stress_fields and 'sigma_2' in stress_fields and 'sigma_3' in stress_fields:
            sigma1 = stress_fields['sigma_1']
            sigma2 = stress_fields['sigma_2']
            sigma3 = stress_fields['sigma_3']
            
            results['max_principal_1'] = float(np.max(sigma1))
            results['max_principal_2'] = float(np.max(sigma2))
            results['max_principal_3'] = float(np.max(sigma3))
            results['max_principal_abs'] = float(np.max(np.abs(sigma1)))
            
            # Maximum shear stress (Tresca)
            max_shear = 0.5 * np.max(np.abs(sigma1 - sigma3))
            results['max_shear_tresca'] = float(max_shear)
        
        # Additional statistical measures
        if 'sigma_hydro' in stress_fields:
            hydro_data = stress_fields['sigma_hydro']
            results['hydro_std'] = float(np.std(hydro_data))
            results['hydro_skewness'] = float(stats.skew(hydro_data.flatten()))
            results['hydro_kurtosis'] = float(stats.kurtosis(hydro_data.flatten()))
        
        if 'von_mises' in stress_fields:
            vm_data = stress_fields['von_mises']
            # Percentiles
            results['von_mises_p95'] = float(np.percentile(vm_data, 95))
            results['von_mises_p99'] = float(np.percentile(vm_data, 99))
            results['von_mises_p99_9'] = float(np.percentile(vm_data, 99.9))
        
        return results
    
    @staticmethod
    def extract_stress_peaks(stress_fields: Dict[str, np.ndarray], 
                           threshold_percentile: float = 95) -> Dict[str, Dict]:
        """
        Extract stress peak locations and values
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            threshold_percentile: Percentile for peak detection
            
        Returns:
            Dictionary with peak information for each stress component
        """
        peaks = {}
        
        for component_name, stress_data in stress_fields.items():
            if not isinstance(stress_data, np.ndarray):
                continue
                
            # Calculate threshold based on percentile
            threshold = np.percentile(stress_data, threshold_percentile)
            
            # Find peaks (indices where value > threshold)
            peak_indices = np.where(stress_data > threshold)
            
            if len(peak_indices[0]) > 0:
                # Get peak values
                peak_values = stress_data[peak_indices]
                
                # Find global maximum
                max_idx = np.argmax(peak_values)
                max_pos = (peak_indices[0][max_idx], peak_indices[1][max_idx])
                
                peaks[component_name] = {
                    'num_peaks': len(peak_values),
                    'max_value': float(np.max(peak_values)),
                    'max_position': max_pos,
                    'mean_peak_value': float(np.mean(peak_values)),
                    'peak_indices': peak_indices,
                    'peak_values': peak_values,
                    'threshold': float(threshold)
                }
        
        return peaks
    
    @staticmethod
    def compute_stress_gradients(stress_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute stress gradients for each stress component
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            
        Returns:
            Dictionary with gradient magnitudes
        """
        gradients = {}
        
        for component_name, stress_data in stress_fields.items():
            if not isinstance(stress_data, np.ndarray):
                continue
            
            # Compute gradients using numpy's gradient
            grad_y, grad_x = np.gradient(stress_data)
            
            # Compute gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            gradients[f'{component_name}_grad_x'] = grad_x
            gradients[f'{component_name}_grad_y'] = grad_y
            gradients[f'{component_name}_grad_mag'] = grad_magnitude
            
            # Statistics
            gradients[f'{component_name}_max_grad'] = float(np.max(grad_magnitude))
            gradients[f'{component_name}_mean_grad'] = float(np.mean(grad_magnitude))
        
        return gradients
    
    @staticmethod
    def create_stress_summary_dataframe(source_simulations: List[Dict], 
                                      predictions: Dict) -> pd.DataFrame:
        """
        Create comprehensive stress summary DataFrame
        
        Args:
            source_simulations: List of source simulation data
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with stress summary for all simulations and predictions
        """
        summary_rows = []
        
        # Process source simulations
        for i, sim_data in enumerate(source_simulations):
            params = sim_data.get('params', {})
            history = sim_data.get('history', [])
            
            if history:
                # Use final frame
                eta, stress_fields = history[-1]
                
                # Compute max stress values
                max_stress = StressAnalysisManager.compute_max_stress_values(stress_fields)
                
                # Create row
                row = {
                    'id': f'source_{i}',
                    'type': 'source',
                    'defect_type': params.get('defect_type', 'Unknown'),
                    'shape': params.get('shape', 'Unknown'),
                    'orientation': params.get('orientation', 'Unknown'),
                    'eps0': params.get('eps0', np.nan),
                    'kappa': params.get('kappa', np.nan),
                    'theta_deg': np.deg2rad(params.get('theta', 0)) if params.get('theta') else np.nan,
                    **max_stress
                }
                summary_rows.append(row)
        
        # Process predictions
        if predictions:
            for pred_key, pred_data in predictions.items():
                if isinstance(pred_data, dict) and 'target_params' in pred_data:
                    params = pred_data['target_params']
                    
                    # Compute max stress values
                    max_stress = StressAnalysisManager.compute_max_stress_values(pred_data)
                    
                    # Create row
                    row = {
                        'id': pred_key,
                        'type': 'prediction',
                        'defect_type': params.get('defect_type', 'Unknown'),
                        'shape': params.get('shape', 'Unknown'),
                        'orientation': params.get('orientation', 'Unknown'),
                        'eps0': params.get('eps0', np.nan),
                        'kappa': params.get('kappa', np.nan),
                        'theta_deg': np.deg2rad(params.get('theta', 0)) if params.get('theta') else np.nan,
                        **max_stress
                    }
                    summary_rows.append(row)
        
        # Create DataFrame
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            # Calculate additional metrics
            if 'max_von_mises' in df.columns and 'max_abs_hydrostatic' in df.columns:
                df['stress_ratio_vm_hydro'] = df['max_von_mises'] / (df['max_abs_hydrostatic'] + 1e-10)
            return df
        else:
            return pd.DataFrame()

# =============================================
# SUNBURST CHART MANAGER
# =============================================
class SunburstChartManager:
    """Manager for creating sunburst charts and other hierarchical visualizations"""
    
    @staticmethod
    def get_all_colormaps() -> List[str]:
        """Get list of all available matplotlib colormaps"""
        colormaps = sorted([m for m in plt.colormaps() if not m.endswith('_r')])
        return colormaps
    
    @staticmethod
    def create_sunburst_chart(df: pd.DataFrame, 
                            path_columns: List[str],
                            value_column: str,
                            title: str = "Sunburst Chart",
                            colormap: str = "viridis") -> go.Figure:
        """
        Create a sunburst chart using plotly
        
        Args:
            df: DataFrame containing the data
            path_columns: List of column names for hierarchical path
            value_column: Column name for values
            title: Chart title
            colormap: Matplotlib colormap name
            
        Returns:
            Plotly Figure object
        """
        # Prepare data for sunburst
        df_plot = df.copy()
        
        # Create path string
        df_plot['path'] = df_plot[path_columns].astype(str).agg(' / '.join, axis=1)
        
        # Create sunburst chart
        fig = px.sunburst(
            df_plot,
            path=path_columns,
            values=value_column,
            color=value_column,
            color_continuous_scale=colormap,
            title=title,
            hover_data={col: True for col in df.columns if col not in path_columns + [value_column]}
        )
        
        fig.update_traces(
            textinfo="label+percent entry",
            hovertemplate="<b>%{label}</b><br>" +
                         f"{value_column}: %{{value:.3f}}<br>" +
                         "%{parent}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            height=600,
            title_x=0.5
        )
        
        return fig
    
    @staticmethod
    def create_treemap_chart(df: pd.DataFrame,
                           path_columns: List[str],
                           value_column: str,
                           title: str = "Treemap Chart",
                           colormap: str = "viridis") -> go.Figure:
        """
        Create a treemap chart as alternative to sunburst
        
        Args:
            df: DataFrame containing the data
            path_columns: List of column names for hierarchical path
            value_column: Column name for values
            title: Chart title
            colormap: Matplotlib colormap name
            
        Returns:
            Plotly Figure object
        """
        fig = px.treemap(
            df,
            path=path_columns,
            values=value_column,
            color=value_column,
            color_continuous_scale=colormap,
            title=title,
            hover_data={col: True for col in df.columns if col not in path_columns + [value_column]}
        )
        
        fig.update_traces(
            textinfo="label+value+percent entry",
            hovertemplate="<b>%{label}</b><br>" +
                         f"{value_column}: %{{value:.3f}}<br>" +
                         "%{parent}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            height=600,
            title_x=0.5
        )
        
        return fig
    
    @staticmethod
    def create_parallel_categories(df: pd.DataFrame,
                                 dimensions: List[str],
                                 color_column: str,
                                 title: str = "Parallel Categories") -> go.Figure:
        """
        Create parallel categories diagram
        
        Args:
            df: DataFrame containing the data
            dimensions: List of categorical columns to include
            color_column: Numerical column for coloring
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = px.parallel_categories(
            df,
            dimensions=dimensions,
            color=df[color_column],
            color_continuous_scale='viridis',
            title=title,
            labels={col: col.replace('_', ' ').title() for col in dimensions}
        )
        
        fig.update_layout(
            height=500,
            title_x=0.5
        )
        
        return fig
    
    @staticmethod
    def create_radial_bar_chart(df: pd.DataFrame,
                              categories: List[str],
                              values: List[str],
                              title: str = "Radial Bar Chart") -> go.Figure:
        """
        Create radial bar chart for comparison
        
        Args:
            df: DataFrame containing the data
            categories: Column name for categories
            values: List of value columns to plot
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        for i, value_col in enumerate(values):
            fig.add_trace(go.Barpolar(
                r=df[value_col].values,
                theta=df[categories].values,
                name=value_col,
                marker_color=px.colors.sequential.Viridis[i/len(values)],
                opacity=0.8
            ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df[values].max().max() * 1.1]
                ),
                angularaxis=dict(
                    direction="clockwise",
                    period=len(df)
                )
            ),
            showlegend=True,
            height=500
        )
        
        return fig

# =============================================
# ENHANCED NUMERICAL SOLUTIONS MANAGER WITH ROBUST LOADING
# =============================================
class NumericalSolutionsManager:
    def __init__(self, solutions_dir: str = NUMERICAL_SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.loaded_files = {}  # Cache for loaded files
        self.failed_files = {}  # Track failed files with reasons
    
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
                    'relative_path': os.path.relpath(file_path, self.solutions_dir),
                    'status': 'unknown'
                }
                all_files.append(file_info)
        
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
    
    def get_file_by_name(self, filename: str) -> Optional[str]:
        for file_info in self.get_all_files():
            if file_info['filename'] == filename:
                return file_info['path']
        return None
    
    def check_file_integrity(self, file_path: str) -> Tuple[bool, str]:
        """
        Check if a file is valid and can be loaded
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        try:
            with open(file_path, 'rb') as f:
                # Quick check for empty or very small files
                if os.path.getsize(file_path) < 10:
                    return False, "File is too small (possibly empty)"
                
                # Read first few bytes to check magic numbers
                header = f.read(8)
                
                # Check for common file signatures
                if file_path.endswith('.pt') or file_path.endswith('.pth'):
                    # PyTorch files start with a specific pickle header
                    if len(header) < 8:
                        return False, "File too short for PyTorch format"
                
                elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                    # Pickle files have protocol header
                    if header and header[0] not in [0x80, 0x83, 0x84, 0x85, 0x86, 0x87]:
                        return False, "Invalid pickle protocol header"
                
                elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                    # HDF5 files start with specific signature
                    if not header.startswith(b'\x89HDF\r\n\x1a\n'):
                        return False, "Invalid HDF5 header"
                
                elif file_path.endswith('.npz'):
                    # NPZ files are zip files
                    if not header.startswith(b'PK'):
                        return False, "Invalid NPZ (not a ZIP file)"
                
            return True, "File appears valid"
            
        except Exception as e:
            return False, f"Error checking file: {str(e)}"
    
    def load_simulation(self, file_path: str, interpolator) -> Optional[Dict[str, Any]]:
        """
        Load simulation with enhanced error handling and fallbacks
        
        Args:
            file_path: Path to simulation file
            interpolator: SpatialLocalityAttentionInterpolator instance
            
        Returns:
            Loaded simulation data or None if failed
        """
        # Check cache first
        if file_path in self.loaded_files:
            return self.loaded_files[file_path]
        
        # Check if file failed before
        if file_path in self.failed_files:
            st.warning(f"Skipping previously failed file: {os.path.basename(file_path)}")
            return None
        
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        try:
            # First check file integrity
            is_valid, reason = self.check_file_integrity(file_path)
            if not is_valid:
                self.failed_files[file_path] = f"Invalid file: {reason}"
                st.warning(f"❌ File integrity check failed for {filename}: {reason}")
                return None
            
            # Determine format
            if ext in ['pkl', 'pickle']:
                format_type = 'pkl'
                sim_data = self._load_pkl_with_fallbacks(file_path, interpolator)
            elif ext in ['pt', 'pth']:
                format_type = 'pt'
                sim_data = self._load_pt_with_fallbacks(file_path, interpolator)
            elif ext in ['h5', 'hdf5']:
                format_type = 'h5'
                sim_data = interpolator.read_simulation_file(file_path, format_type)
            elif ext == 'npz':
                format_type = 'npz'
                sim_data = interpolator.read_simulation_file(file_path, format_type)
            elif ext in ['sql', 'db']:
                format_type = 'sql'
                sim_data = interpolator.read_simulation_file(file_path, format_type)
            elif ext == 'json':
                format_type = 'json'
                sim_data = interpolator.read_simulation_file(file_path, format_type)
            else:
                st.warning(f"❌ Unknown file format: {filename}")
                self.failed_files[file_path] = "Unknown format"
                return None
            
            if sim_data:
                sim_data['loaded_from'] = 'numerical_solutions'
                sim_data['filename'] = filename
                sim_data['format'] = format_type
                
                # Cache successful load
                self.loaded_files[file_path] = sim_data
                
                # Validate data structure
                if self._validate_simulation_data(sim_data):
                    st.success(f"✅ Successfully loaded: {filename}")
                    return sim_data
                else:
                    st.warning(f"⚠️ Loaded but invalid data structure: {filename}")
                    return None
            else:
                self.failed_files[file_path] = "Loader returned None"
                return None
                
        except Exception as e:
            error_msg = str(e)
            self.failed_files[file_path] = error_msg
            
            # Provide specific guidance based on error type
            if "weights_only" in error_msg or "safe_globals" in error_msg:
                st.error(f"❌ PyTorch security restriction for {filename}. Try: torch.load(file, weights_only=False)")
            elif "pickle" in error_msg.lower():
                st.error(f"❌ Pickle error for {filename}. File may be corrupted.")
            else:
                st.error(f"❌ Error loading {filename}: {error_msg}")
            
            return None
    
    def _load_pkl_with_fallbacks(self, file_path: str, interpolator) -> Optional[Dict[str, Any]]:
        """
        Load pickle file with multiple fallback strategies
        """
        filename = os.path.basename(file_path)
        
        # Strategy 1: Try standard pickle.load
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return interpolator.read_simulation_file(file_content, 'pkl')
        except Exception as e1:
            st.warning(f"⚠️ Standard pickle load failed for {filename}: {str(e1)}")
        
        # Strategy 2: Try with different pickle protocols
        try:
            with open(file_path, 'rb') as f:
                # Try with pickle.load with different find_classes
                import pickle as pkl
                data = pkl.load(f)
                return self._convert_legacy_data(data)
        except Exception as e2:
            st.warning(f"⚠️ Alternative pickle load failed for {filename}: {str(e2)}")
        
        # Strategy 3: Try joblib (handles numpy arrays better)
        try:
            import joblib
            data = joblib.load(file_path)
            return self._convert_legacy_data(data)
        except Exception as e3:
            st.warning(f"⚠️ Joblib load failed for {filename}: {str(e3)}")
        
        # Strategy 4: Try to read as text and parse
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # Read first 1KB
                if 'pickle' in content.lower() or 'numpy' in content.lower():
                    st.info(f"ℹ️ {filename} might be a text file with pickle data")
        except:
            pass
        
        return None
    
    def _load_pt_with_fallbacks(self, file_path: str, interpolator) -> Optional[Dict[str, Any]]:
        """
        Load PyTorch file with multiple fallback strategies for compatibility
        """
        filename = os.path.basename(file_path)
        
        # Strategy 1: Try with weights_only=True (secure mode)
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return interpolator.read_simulation_file(file_content, 'pt')
        except Exception as e1:
            st.warning(f"⚠️ Secure PyTorch load failed for {filename}: {str(e1)}")
        
        # Strategy 2: Try with weights_only=False (insecure but compatible)
        try:
            buffer = BytesIO()
            with open(file_path, 'rb') as f:
                buffer.write(f.read())
            buffer.seek(0)
            
            # Warning about security
            st.warning(f"⚠️ Using insecure PyTorch load for {filename} (weights_only=False)")
            
            data = torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)
            return self._convert_legacy_data(data)
        except Exception as e2:
            st.error(f"❌ Insecure PyTorch load also failed for {filename}: {str(e2)}")
        
        # Strategy 3: Try to extract numpy arrays directly
        try:
            # Open as zip file (PyTorch files are zip archives)
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Look for pickle files inside
                for name in zf.namelist():
                    if name.endswith('.pkl') or name == 'data.pkl':
                        with zf.open(name) as f:
                            import pickle
                            data = pickle.load(f)
                            return self._convert_legacy_data(data)
        except Exception as e3:
            st.warning(f"⚠️ Zip extraction failed for {filename}: {str(e3)}")
        
        return None
    
    def _convert_legacy_data(self, data: Any) -> Dict[str, Any]:
        """
        Convert legacy data formats to standardized format
        """
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': 'legacy'
        }
        
        if isinstance(data, dict):
            # Try to extract parameters
            if 'params' in data:
                standardized['params'] = data['params']
            elif 'parameters' in data:
                standardized['params'] = data['parameters']
            
            # Try to extract history
            if 'history' in data:
                standardized['history'] = data['history']
            elif 'frames' in data:
                standardized['history'] = data['frames']
            
            # Try to extract metadata
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            elif 'info' in data:
                standardized['metadata'] = data['info']
            
            # If it's a stress field directly
            if 'sigma_hydro' in data or 'von_mises' in data:
                standardized['history'] = [(0.0, data)]  # Single frame
            
        # Handle numpy arrays
        elif isinstance(data, np.ndarray):
            standardized['history'] = [(0.0, {'stress_field': data})]
        
        return standardized
    
    def _validate_simulation_data(self, sim_data: Dict[str, Any]) -> bool:
        """Validate that simulation data has required structure"""
        if not isinstance(sim_data, dict):
            return False
        
        # Check for history or params
        has_history = 'history' in sim_data and isinstance(sim_data['history'], list)
        has_params = 'params' in sim_data and isinstance(sim_data['params'], dict)
        
        return has_history or has_params
    
    def save_simulation(self, data: Dict[str, Any], filename: str, format_type: str = 'pkl'):
        if not filename.endswith(f'.{format_type}'):
            filename = f"{filename}.{format_type}"
        
        file_path = os.path.join(self.solutions_dir, filename)
        
        try:
            if format_type == 'pkl':
                # Use highest protocol for compatibility
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            elif format_type == 'pt':
                # Use PyTorch with secure settings
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
            if file_path in self.loaded_files:
                del self.loaded_files[file_path]
            if file_path in self.failed_files:
                del self.failed_files[file_path]
            
            st.success(f"✅ Saved simulation to: {filename}")
            return True
            
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return False
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """Get statistics about loading success/failure"""
        return {
            'loaded': len(self.loaded_files),
            'failed': len(self.failed_files),
            'total_files': len(self.get_all_files()),
            'failed_details': self.failed_files.copy()
        }

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
        source_param_vectors = []
        source_stress_data = []
        
        for sim_data in source_simulations:
            param_vector, _ = interpolator.compute_parameter_vector(sim_data)
            source_param_vectors.append(param_vector)
            
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
        
        source_param_vectors = np.array(source_param_vectors)
        source_stress_data = np.array(source_stress_data)
        
        # Predict for each target
        for idx, target_params in enumerate(target_params_list):
            # Compute target parameter vector
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
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
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
        
        # Initialize stress analysis manager
        self.stress_analyzer = StressAnalysisManager()
        self.sunburst_manager = SunburstChartManager()
        
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
    
    # =============================================
    # ENHANCED READER METHODS WITH COMPATIBILITY FIXES
    # =============================================
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        try:
            # Try standard pickle load
            return pickle.load(buffer)
        except (pickle.UnpicklingError, KeyError, ValueError) as e:
            # Try with different protocols
            buffer.seek(0)
            try:
                # Try with protocol 4 (Python 3.4-3.7)
                return pickle.load(buffer)
            except:
                buffer.seek(0)
                try:
                    # Try with protocol 5 (Python 3.8+)
                    return pickle.load(buffer)
                except Exception as e2:
                    raise ValueError(f"Failed to unpickle file: {str(e2)}")
    
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        
        # First try with weights_only=True (secure mode)
        try:
            return torch.load(buffer, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e1:
            st.warning(f"⚠️ Secure PyTorch load failed: {str(e1)}")
            
            # Fallback: try with weights_only=False
            buffer.seek(0)
            try:
                st.warning("⚠️ Trying insecure PyTorch load (weights_only=False)")
                return torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)
            except Exception as e2:
                # Last resort: try to extract data manually
                buffer.seek(0)
                try:
                    # PyTorch files are zip archives
                    import zipfile
                    import io
                    import struct
                    
                    # Check if it's a zip file
                    buffer.seek(0)
                    magic = buffer.read(4)
                    buffer.seek(0)
                    
                    if magic.startswith(b'PK'):
                        # It's a zip file, try to extract pickle
                        with zipfile.ZipFile(buffer, 'r') as zf:
                            # Look for data.pkl or similar
                            for name in zf.namelist():
                                if 'pkl' in name.lower():
                                    with zf.open(name) as f:
                                        data = pickle.load(f)
                                        return data
                    
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
                        # Try to read as attribute
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
        result = {}
        for key in data.files:
            try:
                result[key] = data[key]
            except:
                result[key] = str(data[key])
        return result
    
    def _read_sql(self, file_content):
        # SQLite databases need to be saved to a temporary file
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
            # Try other encodings
            try:
                return json.loads(file_content.decode('latin-1'))
            except:
                raise ValueError("Could not decode JSON file")
    
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file from content with enhanced error handling"""
        
        if format_type == 'auto':
            # Default to pkl if we can't determine
            format_type = 'pkl'
        
        if format_type in self.readers:
            try:
                data = self.readers[format_type](file_content)
                return self._standardize_data(data, format_type, "uploaded_file")
            except Exception as e:
                # Try to provide more helpful error messages
                if "weights_only" in str(e):
                    raise ValueError(
                        f"PyTorch security restriction. "
                        f"Try: torch.load(file, weights_only=False) or update file format. "
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
            'filename': os.path.basename(file_path) if isinstance(file_path, str) else "uploaded",
            'load_timestamp': datetime.now().isoformat()
        }
        
        # Handle different formats
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
                            # Assume (eta, stresses) format
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
            # Extract from H5 structure
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            
            # Try to find history data
            for key, value in data.items():
                if 'history' in key.lower():
                    if isinstance(value, (list, np.ndarray)):
                        standardized['history'] = value
                        break
        
        elif format_type == 'npz':
            # NPZ files are similar to dict
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
        
        # Ensure history is a list
        if not isinstance(standardized['history'], list):
            standardized['history'] = []
        
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
        
        # 4. Orientation encoding - handle custom angles
        orientation = params.get('orientation', 'Horizontal {111} (0°)')
        orientation_encoding = {
            'Horizontal {111} (0°)': [1, 0, 0, 0],
            'Tilted 30° (1¯10 projection)': [0, 1, 0, 0],
            'Tilted 60°': [0, 0, 1, 0],
            'Vertical {111} (90°)': [0, 0, 0, 1]
        }
        
        # Check if orientation is a custom angle string like "Custom (15°)"
        if orientation.startswith('Custom ('):
            # For custom angles, we don't use one-hot encoding
            # Instead we rely on theta_norm for the angle information
            param_vector.extend([0, 0, 0, 0])  # All zeros for custom
        else:
            param_vector.extend(orientation_encoding.get(orientation, [0, 0, 0, 0]))
            
        param_names.extend(['orient_0deg', 'orient_30deg', 'orient_60deg', 'orient_90deg'])
        
        return np.array(param_vector, dtype=np.float32), param_names
    
    @staticmethod
    def get_orientation_from_angle(angle_deg: float) -> str:
        """Convert angle in degrees to orientation string with custom support"""
        if 0 <= angle_deg <= 15:
            return 'Horizontal {111} (0°)'
        elif 15 < angle_deg <= 45:
            return 'Tilted 30° (1¯10 projection)'
        elif 45 < angle_deg <= 75:
            return 'Tilted 60°'
        elif 75 < angle_deg <= 90:
            return 'Vertical {111} (90°)'
        else:
            # Handle angles outside 0-90 by wrapping
            angle_deg = angle_deg % 90
            return f"Custom ({angle_deg:.1f}°)"

# =============================================
# GRID AND EXTENT CONFIGURATION
# =============================================
def get_grid_extent(N=128, dx=0.1):
    """Get grid extent for visualization"""
    return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

# =============================================
# ENHANCED ATTENTION INTERFACE WITH STRESS ANALYSIS
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with enhanced stress analysis"""
    
    st.header("🤖 Spatial-Attention Stress Interpolation with Analysis")
    
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
    
    # Initialize stress analysis manager
    if 'stress_analyzer' not in st.session_state:
        st.session_state.stress_analyzer = StressAnalysisManager()
    
    # Initialize sunburst chart manager
    if 'sunburst_manager' not in st.session_state:
        st.session_state.sunburst_manager = SunburstChartManager()
    
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    
    # Initialize multi-target predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
    
    # Initialize stress summary DataFrame
    if 'stress_summary_df' not in st.session_state:
        st.session_state.stress_summary_df = pd.DataFrame()
    
    # Get grid extent for visualization
    extent = get_grid_extent()
    
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 Load Source Data", 
        "🎯 Configure Target", 
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict", 
        "📊 Results & Export",
        "📁 Manage Files",
        "📈 Stress Analysis & Sunburst"
    ])
    
    # Tab 1: Load Source Data (keeping existing code)
    with tab1:
        st.subheader("Load Source Simulation Files")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📂 From Numerical Solutions Directory")
            st.info(f"Loading from: `{NUMERICAL_SOLUTIONS_DIR}`")
            
            file_formats = st.session_state.solutions_manager.scan_directory()
            all_files_info = st.session_state.solutions_manager.get_all_files()
            
            if not all_files_info:
                st.warning(f"No simulation files found in `{NUMERICAL_SOLUTIONS_DIR}`")
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
                                            
                                            if sim_data and file_path not in st.session_state.loaded_from_numerical:
                                                st.session_state.source_simulations.append(sim_data)
                                                st.session_state.loaded_from_numerical.append(file_path)
                                                loaded_count += 1
                                                st.success(f"✅ Loaded: {os.path.basename(file_path)}")
                                            elif file_path in st.session_state.loaded_from_numerical:
                                                st.warning(f"⚠️ Already loaded: {os.path.basename(file_path)}")
                                            else:
                                                st.error(f"❌ Failed to load: {os.path.basename(file_path)}")
                                                
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
                accept_multiple_files=True
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
                            
                            sim_data = st.session_state.interpolator.read_simulation_file(
                                file_content, actual_format
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
        
        # Display loaded simulations
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
                        st.session_state.solutions_manager.loaded_files.clear()
                        st.session_state.solutions_manager.failed_files.clear()
                        st.success("All source simulations cleared!")
                        st.rerun()
                with col2:
                    st.info(f"**Total loaded simulations:** {len(st.session_state.source_simulations)}")
    
    # Tabs 2-6: Keep existing code as before...
    # (Keeping the same implementation for tabs 2-6 as in the previous code)
    
    # =============================================
    # ENHANCED TAB 7: STRESS ANALYSIS WITH BETTER LOADING
    # =============================================
    with tab7:
        st.header("📈 Stress Analysis and Sunburst Visualization")
        
        # Show loading statistics
        if hasattr(st.session_state.solutions_manager, 'get_loading_stats'):
            stats = st.session_state.solutions_manager.get_loading_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", stats['total_files'])
            with col2:
                st.metric("Successfully Loaded", stats['loaded'])
            with col3:
                st.metric("Failed to Load", stats['failed'])
            with col4:
                success_rate = (stats['loaded'] / max(stats['total_files'], 1)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Update stress summary DataFrame
        if st.button("🔄 Update Stress Summary", type="primary"):
            with st.spinner("Computing stress statistics..."):
                st.session_state.stress_summary_df = (
                    st.session_state.stress_analyzer.create_stress_summary_dataframe(
                        st.session_state.source_simulations,
                        st.session_state.multi_target_predictions
                    )
                )
                if not st.session_state.stress_summary_df.empty:
                    st.success(f"✅ Stress summary updated with {len(st.session_state.stress_summary_df)} entries")
                else:
                    st.warning("No data available for stress analysis")
        
        # Advanced loading options
        with st.expander("🔧 Advanced Loading Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                max_files = st.number_input("Maximum files to load", 1, 1000, 50, 10)
                load_pt = st.checkbox("Load .pt files", value=True)
                load_pkl = st.checkbox("Load .pkl files", value=True)
                load_h5 = st.checkbox("Load .h5 files", value=True)
            
            with col2:
                use_weights_only = st.checkbox("Use secure PyTorch loading", value=False, 
                                             help="If checked, uses weights_only=True. Uncheck for compatibility with older files.")
                skip_corrupted = st.checkbox("Skip corrupted files", value=True)
                retry_failed = st.checkbox("Retry previously failed files", value=False)
            
            if st.button("🔄 Bulk Load All Compatible Files", type="secondary"):
                all_files_info = st.session_state.solutions_manager.get_all_files()
                loaded_count = 0
                failed_count = 0
                
                # Filter files by format
                filtered_files = []
                for file_info in all_files_info[:max_files]:
                    format_type = file_info['format']
                    if (format_type == 'pt' and not load_pt) or \
                       (format_type == 'pkl' and not load_pkl) or \
                       (format_type == 'h5' and not load_h5):
                        continue
                    
                    # Skip if previously failed and not retrying
                    if file_info['path'] in st.session_state.solutions_manager.failed_files and not retry_failed:
                        continue
                    
                    filtered_files.append(file_info)
                
                # Load files
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file_info in enumerate(filtered_files):
                    status_text.text(f"Loading {i+1}/{len(filtered_files)}: {file_info['filename']}")
                    progress_bar.progress((i + 1) / len(filtered_files))
                    
                    try:
                        sim_data = st.session_state.solutions_manager.load_simulation(
                            file_info['path'],
                            st.session_state.interpolator
                        )
                        
                        if sim_data:
                            # Check if already loaded
                            if file_info['path'] not in st.session_state.loaded_from_numerical:
                                st.session_state.source_simulations.append(sim_data)
                                st.session_state.loaded_from_numerical.append(file_info['path'])
                                loaded_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        if not skip_corrupted:
                            st.error(f"❌ Error loading {file_info['filename']}: {str(e)}")
                        failed_count += 1
                
                progress_bar.empty()
                status_text.empty()
                
                if loaded_count > 0:
                    st.success(f"✅ Successfully loaded {loaded_count} new files!")
                    if failed_count > 0:
                        st.warning(f"⚠️ Failed to load {failed_count} files")
                else:
                    st.warning("No new files were loaded")
        
        # Display stress summary if available
        if not st.session_state.stress_summary_df.empty:
            st.subheader("📋 Stress Summary Statistics")
            
            # Show DataFrame
            st.dataframe(
                st.session_state.stress_summary_df.style.format({
                    col: "{:.3f}" for col in st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns
                }),
                use_container_width=True,
                height=400
            )
            
            # Download stress summary
            csv_buffer = BytesIO()
            st.session_state.stress_summary_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Stress Summary CSV",
                data=csv_buffer,
                file_name=f"stress_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # SUNBURST CHART CONFIGURATION
            st.subheader("🌀 Sunburst Chart Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Select hierarchical levels
                available_columns = list(st.session_state.stress_summary_df.columns)
                categorical_cols = ['defect_type', 'shape', 'orientation', 'type']
                categorical_cols = [c for c in categorical_cols if c in available_columns]
                
                level1 = st.selectbox(
                    "First Level (Center)",
                    categorical_cols,
                    index=0 if 'type' in categorical_cols else 0
                )
                
                level2_options = [c for c in categorical_cols if c != level1]
                level2 = st.selectbox(
                    "Second Level",
                    ['None'] + level2_options,
                    index=0
                )
                
                level3_options = [c for c in level2_options if c != level2 and level2 != 'None']
                level3 = st.selectbox(
                    "Third Level",
                    ['None'] + level3_options,
                    index=0
                )
            
            with col2:
                # Select value column for sizing
                numeric_cols = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                stress_value_cols = [c for c in numeric_cols if 'max' in c or 'mean' in c]
                
                value_column = st.selectbox(
                    "Value Metric",
                    stress_value_cols,
                    index=0 if 'max_von_mises' in stress_value_cols else 0
                )
            
            with col3:
                # Select colormap
                colormaps = st.session_state.sunburst_manager.get_all_colormaps()
                selected_colormap = st.selectbox(
                    "Colormap",
                    colormaps,
                    index=colormaps.index('viridis') if 'viridis' in colormaps else 0
                )
                
                # Chart type selection
                chart_type = st.radio(
                    "Chart Type",
                    ["Sunburst", "Treemap", "Parallel Categories", "Radial Bar"],
                    horizontal=True
                )
            
            # Build path columns list
            path_columns = [level1]
            if level2 != 'None':
                path_columns.append(level2)
            if level3 != 'None':
                path_columns.append(level3)
            
            # Additional filters
            st.subheader("🔍 Filter Options")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # Filter by defect type
                defect_types = st.session_state.stress_summary_df['defect_type'].unique() if 'defect_type' in st.session_state.stress_summary_df.columns else []
                selected_defects = st.multiselect(
                    "Filter by Defect Type",
                    defect_types,
                    default=defect_types.tolist() if len(defect_types) > 0 else []
                )
            
            with filter_col2:
                # Filter by shape
                shapes = st.session_state.stress_summary_df['shape'].unique() if 'shape' in st.session_state.stress_summary_df.columns else []
                selected_shapes = st.multiselect(
                    "Filter by Shape",
                    shapes,
                    default=shapes.tolist() if len(shapes) > 0 else []
                )
            
            with filter_col3:
                # Filter by simulation type
                sim_types = st.session_state.stress_summary_df['type'].unique() if 'type' in st.session_state.stress_summary_df.columns else []
                selected_types = st.multiselect(
                    "Filter by Simulation Type",
                    sim_types,
                    default=sim_types.tolist() if len(sim_types) > 0 else []
                )
            
            # Apply filters
            df_filtered = st.session_state.stress_summary_df.copy()
            
            if len(selected_defects) > 0:
                df_filtered = df_filtered[df_filtered['defect_type'].isin(selected_defects)]
            
            if len(selected_shapes) > 0:
                df_filtered = df_filtered[df_filtered['shape'].isin(selected_shapes)]
            
            if len(selected_types) > 0:
                df_filtered = df_filtered[df_filtered['type'].isin(selected_types)]
            
            # Generate chart button
            if st.button("🌀 Generate Visualization", type="primary"):
                if len(df_filtered) == 0:
                    st.warning("No data available after filtering")
                elif len(path_columns) == 0:
                    st.warning("Please select at least one hierarchical level")
                elif value_column not in df_filtered.columns:
                    st.warning(f"Value column '{value_column}' not found in data")
                else:
                    with st.spinner("Generating visualization..."):
                        try:
                            if chart_type == "Sunburst":
                                fig = st.session_state.sunburst_manager.create_sunburst_chart(
                                    df=df_filtered,
                                    path_columns=path_columns,
                                    value_column=value_column,
                                    title=f"Stress Analysis: {value_column.replace('_', ' ').title()}",
                                    colormap=selected_colormap
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif chart_type == "Treemap":
                                fig = st.session_state.sunburst_manager.create_treemap_chart(
                                    df=df_filtered,
                                    path_columns=path_columns,
                                    value_column=value_column,
                                    title=f"Stress Analysis: {value_column.replace('_', ' ').title()}",
                                    colormap=selected_colormap
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif chart_type == "Parallel Categories":
                                if len(path_columns) >= 2:
                                    dimensions = path_columns[:min(4, len(path_columns))]
                                    fig = st.session_state.sunburst_manager.create_parallel_categories(
                                        df=df_filtered,
                                        dimensions=dimensions,
                                        color_column=value_column,
                                        title=f"Stress Analysis: {value_column.replace('_', ' ').title()}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Parallel Categories requires at least 2 dimensions")
                            
                            elif chart_type == "Radial Bar":
                                if len(path_columns) > 0:
                                    category_col = path_columns[0]
                                    selected_values = st.multiselect(
                                        "Select Value Columns for Radial Bar",
                                        stress_value_cols,
                                        default=stress_value_cols[:min(3, len(stress_value_cols))]
                                    )
                                    
                                    if len(selected_values) > 0:
                                        fig = st.session_state.sunburst_manager.create_radial_bar_chart(
                                            df=df_filtered,
                                            categories=category_col,
                                            values=selected_values,
                                            title="Stress Component Comparison"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Please select at least one value column")
                                else:
                                    st.warning("Please select at least one category column")
                        
                        except Exception as e:
                            st.error(f"Error generating chart: {str(e)}")
        else:
            st.info("👈 Please load simulations first to enable stress analysis")

# =============================================
# ENHANCED MAIN APPLICATION WITH FILE DIAGNOSTICS
# =============================================
def main():
    """Main application with enhanced stress analysis and file diagnostics"""
    
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
        create_attention_interface()
    
    elif operation_mode == "Stress Analysis Dashboard":
        st.header("📊 Stress Analysis Dashboard")
        
        # Initialize managers
        if 'stress_analyzer' not in st.session_state:
            st.session_state.stress_analyzer = StressAnalysisManager()
        if 'sunburst_manager' not in st.session_state:
            st.session_state.sunburst_manager = SunburstChartManager()
        
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
        
        # File diagnostics section
        st.subheader("📋 File Diagnostics")
        
        all_files = st.session_state.solutions_manager.get_all_files()
        
        if not all_files:
            st.warning(f"No files found in {NUMERICAL_SOLUTIONS_DIR}")
        else:
            # File statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Files", len(all_files))
            with col2:
                pt_files = len([f for f in all_files if f['format'] == 'pt'])
                st.metric(".pt Files", pt_files)
            with col3:
                pkl_files = len([f for f in all_files if f['format'] == 'pkl'])
                st.metric(".pkl Files", pkl_files)
            with col4:
                total_size = sum(f['size'] for f in all_files) / (1024 * 1024)
                st.metric("Total Size", f"{total_size:.1f} MB")
            
            # Quick load button
            if st.button("📥 Quick Load All Compatible Files", type="primary"):
                with st.spinner("Loading compatible files..."):
                    all_simulations = []
                    loaded_files = []
                    failed_files = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, file_info in enumerate(all_files[:100]):  # Limit to 100 files
                        progress_bar.progress((i + 1) / min(len(all_files), 100))
                        
                        try:
                            # Skip files over 100MB
                            if file_info['size'] > 100 * 1024 * 1024:
                                st.warning(f"Skipping large file: {file_info['filename']} ({file_info['size'] // (1024*1024)}MB)")
                                continue
                            
                            sim_data = st.session_state.solutions_manager.load_simulation(
                                file_info['path'],
                                st.session_state.interpolator
                            )
                            
                            if sim_data:
                                all_simulations.append(sim_data)
                                loaded_files.append(file_info['filename'])
                            else:
                                failed_files.append(file_info['filename'])
                                
                        except Exception as e:
                            failed_files.append(file_info['filename'])
                    
                    progress_bar.empty()
                    
                    if all_simulations:
                        # Create comprehensive stress summary
                        stress_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                            all_simulations, {}
                        )
                        
                        if not stress_df.empty:
                            st.session_state.stress_summary_df = stress_df
                            st.session_state.source_simulations = all_simulations
                            st.success(f"✅ Loaded {len(loaded_files)} simulations for analysis")
                            
                            if failed_files:
                                with st.expander(f"⚠️ {len(failed_files)} files failed to load"):
                                    for f in failed_files[:10]:  # Show first 10
                                        st.write(f"- {f}")
                                    if len(failed_files) > 10:
                                        st.write(f"... and {len(failed_files) - 10} more")
                        else:
                            st.warning("No stress data found in loaded simulations")
                    else:
                        st.error("No simulations could be loaded. Check file formats and compatibility.")
            
            # Manual file inspection
            with st.expander("🔍 Inspect Individual Files", expanded=False):
                selected_file = st.selectbox(
                    "Select file to inspect",
                    [f"{f['filename']} ({f['format']}, {f['size']//1024}KB)" for f in all_files],
                    index=0
                )
                
                if selected_file and st.button("Inspect File"):
                    # Extract filename from selection
                    filename = selected_file.split(" (")[0]
                    file_info = next(f for f in all_files if f['filename'] == filename)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**File Information:**")
                        st.write(f"Path: `{file_info['path']}`")
                        st.write(f"Format: {file_info['format']}")
                        st.write(f"Size: {file_info['size']} bytes")
                        st.write(f"Modified: {file_info['modified']}")
                        
                        # Check integrity
                        is_valid, reason = st.session_state.solutions_manager.check_file_integrity(file_info['path'])
                        if is_valid:
                            st.success("✅ File appears valid")
                        else:
                            st.error(f"❌ File may be corrupted: {reason}")
                    
                    with col2:
                        if st.button("Try to Load This File"):
                            try:
                                sim_data = st.session_state.solutions_manager.load_simulation(
                                    file_info['path'],
                                    st.session_state.interpolator
                                )
                                
                                if sim_data:
                                    st.success("✅ File loaded successfully!")
                                    st.write("**Contents:**")
                                    st.json(sim_data, expanded=False)
                                else:
                                    st.error("❌ Failed to load file")
                                    
                            except Exception as e:
                                st.error(f"❌ Error loading file: {str(e)}")
        
        # Show the stress analysis tab interface if we have data
        if not st.session_state.stress_summary_df.empty:
            st.subheader("📊 Analysis Dashboard")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Simulations", len(st.session_state.stress_summary_df))
            with col2:
                max_vm = st.session_state.stress_summary_df['max_von_mises'].max() if 'max_von_mises' in st.session_state.stress_summary_df.columns else 0
                st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
            with col3:
                mean_vm = st.session_state.stress_summary_df['max_von_mises'].mean() if 'max_von_mises' in st.session_state.stress_summary_df.columns else 0
                st.metric("Avg Max Von Mises", f"{mean_vm:.2f} GPa")
            with col4:
                defect_counts = st.session_state.stress_summary_df['defect_type'].value_counts().to_dict() if 'defect_type' in st.session_state.stress_summary_df.columns else {}
                st.metric("Unique Defect Types", len(defect_counts))
            
            # Show the stress analysis tab interface
            create_attention_interface()
        else:
            st.info("Please load simulations first to enable the stress analysis dashboard")
    
    elif operation_mode == "File Diagnostics":
        st.header("🔧 File Diagnostics and Repair Tools")
        
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        all_files = st.session_state.solutions_manager.get_all_files()
        
        if not all_files:
            st.warning(f"No files found in {NUMERICAL_SOLUTIONS_DIR}")
            return
        
        st.subheader("📊 File Statistics")
        
        # File format breakdown
        format_counts = {}
        for file_info in all_files:
            fmt = file_info['format']
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        fig_format, ax = plt.subplots(figsize=(8, 4))
        ax.bar(format_counts.keys(), format_counts.values())
        ax.set_xlabel('File Format')
        ax.set_ylabel('Count')
        ax.set_title('File Format Distribution')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig_format)
        
        # File size distribution
        sizes = [f['size'] / (1024 * 1024) for f in all_files]  # Convert to MB
        
        fig_size, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sizes, bins=20, alpha=0.7, color='steelblue')
        ax.set_xlabel('File Size (MB)')
        ax.set_ylabel('Count')
        ax.set_title('File Size Distribution')
        st.pyplot(fig_size)
        
        # File repair tools
        st.subheader("🛠️ File Repair Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Check All File Integrity"):
                with st.spinner("Checking file integrity..."):
                    results = []
                    for file_info in all_files[:50]:  # Limit to 50 files
                        is_valid, reason = st.session_state.solutions_manager.check_file_integrity(file_info['path'])
                        results.append({
                            'File': file_info['filename'],
                            'Valid': '✅' if is_valid else '❌',
                            'Status': reason
                        })
                    
                    if results:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, use_container_width=True)
        
        with col2:
            if st.button("🧹 Clear Loading Cache"):
                if hasattr(st.session_state.solutions_manager, 'loaded_files'):
                    st.session_state.solutions_manager.loaded_files.clear()
                if hasattr(st.session_state.solutions_manager, 'failed_files'):
                    st.session_state.solutions_manager.failed_files.clear()
                st.success("✅ Loading cache cleared")
        
        # PyTorch compatibility information
        st.subheader("ℹ️ PyTorch Compatibility Information")
        
        st.info(f"""
        **Current Environment:**
        - PyTorch Version: {torch.__version__}
        - NumPy Version: {np.__version__}
        - Python Version: {sys.version.split()[0]}
        
        **Common Issues and Solutions:**
        
        1. **PyTorch 2.6+ Security Changes:**
           - Files created with older PyTorch versions may not load with `weights_only=True`
           - **Solution:** Use `weights_only=False` or add safe globals for NumPy objects
        
        2. **NumPy Version Incompatibility:**
           - Files created with NumPy 1.x may not load with NumPy 2.0+
           - **Solution:** Use `torch.serialization.add_safe_globals()` to add missing types
        
        3. **Corrupted Files:**
           - Incomplete downloads or disk errors can corrupt files
           - **Solution:** Recreate files from source or download again
        
        **Recommendations for Your Files:**
        - Consider converting old .pt files to .npz format for better compatibility
        - Use PyTorch's `torch.save(data, file, pickle_protocol=4)` for better compatibility
        - For critical data, store in multiple formats (.npz and .h5)
        """)

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Enhanced Theoretical Analysis: Stress Metrics and Visualization", expanded=False):
    st.markdown(f"""
    ## 📊 **Enhanced Stress Analysis and Visualization**
    
    ### **🏔️ Maximum Stress Value Capture**
    
    **New Stress Metrics:**
    1. **Hydrostatic Stress (σ_hydro):**
       - Max Absolute Value: `max_abs_hydrostatic`
       - Maximum Value: `max_hydrostatic`
       - Minimum Value: `min_hydrostatic`
       - Mean Absolute Value: `mean_abs_hydrostatic`
       - Standard Deviation: `hydro_std`
       - Skewness: `hydro_skewness`
       - Kurtosis: `hydro_kurtosis`
    
    2. **Stress Magnitude:**
       - Maximum: `max_stress_magnitude`
       - Mean: `mean_stress_magnitude`
    
    3. **Von Mises Stress (σ_vM):**
       - Maximum: `max_von_mises`
       - Mean: `mean_von_mises`
       - Minimum: `min_von_mises`
       - 95th Percentile: `von_mises_p95`
       - 99th Percentile: `von_mises_p99`
       - 99.9th Percentile: `von_mises_p99_9`
    
    ### **🌀 Sunburst Chart Features**
    
    **Hierarchical Visualization:**
    1. **Multi-level Hierarchy:**
       - First Level (Center): Defect type, Shape, or Simulation type
       - Second Level: Orientation, ε*, κ, etc.
       - Third Level: Additional parameters or categories
    
    2. **Value Metrics:** Any stress metric can be used for:
       - Area sizing in sunburst
       - Color mapping
       - Value display
    
    3. **50+ Colormaps:** Full matplotlib colormap support
    
    **Advanced Chart Types:**
    1. **Sunburst Chart:** Radial hierarchical visualization
    2. **Treemap Chart:** Rectangular hierarchical visualization
    3. **Parallel Categories:** Multi-dimensional categorical visualization
    4. **Radial Bar Chart:** Circular bar chart for comparisons
    
    ### **🔧 File Compatibility Fixes**
    
    **PyTorch 2.6+ Compatibility:**
    1. **Safe Globals Registration:** Automatically registers NumPy scalar types
    2. **Multiple Loading Strategies:** Tries secure mode first, falls back to insecure
    3. **File Integrity Checks:** Validates files before attempting to load
    
    **Corrupted File Handling:**
    1. **Graceful Degradation:** Skips corrupted files instead of crashing
    2. **Detailed Error Messages:** Provides specific guidance for each error type
    3. **Cache Management:** Tracks successful and failed loads to avoid retries
    
    **Enhanced Diagnostics:**
    1. **File Statistics:** Shows distribution of file formats and sizes
    2. **Integrity Checking:** Validates file headers and structure
    3. **Loading Statistics:** Tracks success rates and provides feedback
    """)

if __name__ == "__main__":
    main()

st.caption(f"🔬 Enhanced Multi-Target Spatial-Attention Stress Interpolation • Stress Analysis Dashboard • File Compatibility Fixes • 2025")
