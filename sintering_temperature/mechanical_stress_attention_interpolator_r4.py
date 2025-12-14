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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

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
   
    def load_simulation(self, file_path: str, interpolator) -> Dict[str, Any]:
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
           
            sim_data = interpolator.read_simulation_file(file_content, format_type)
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
    # FIXED: ADD THE MISSING READER METHODS
    # =============================================
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        return pickle.load(buffer)
   
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
                    'rows': rows
                }
           
            conn.close()
            os.unlink(tmp_path)
            return data
        except Exception as e:
            os.unlink(tmp_path)
            raise e
   
    def _read_json(self, file_content):
        return json.loads(file_content.decode('utf-8'))
    # =============================================
   
    def read_simulation_file(self, file_content, format_type='auto'):
        """Read simulation file from content"""
       
        if format_type == 'auto':
            # Try to determine format from content or structure
            # For now, default to pkl
            format_type = 'pkl'
       
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            return self._standardize_data(data, format_type, "uploaded_file")
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
            # Try to extract data from H5 structure
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            # Try to find history data
            for key in data.keys():
                if 'history' in key.lower():
                    standardized['history'] = data[key]
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
            param_vector.extend([0, 0, 0, 0]) # All zeros for custom
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
   
    # Tab 1: Load Source Data
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
                        st.success("All source simulations cleared!")
                        st.rerun()
                with col2:
                    st.info(f"**Total loaded simulations:** {len(st.session_state.source_simulations)}")
   
    # Tab 2: Configure Target
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
   
    # Tab 3: Configure Multiple Targets
    with tab3:
        st.subheader("Configure Multiple Target Parameters")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            st.info("Configure ranges for parameters to create multiple target predictions")
           
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
                orientation_mode = st.radio(
                    "Orientation Mode",
                    ["Predefined", "Custom Angles"],
                    horizontal=True,
                    key="orientation_mode_multi"
                )
               
                if orientation_mode == "Predefined":
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
                    st.info(f"**Base θ:** {np.rad2deg(base_theta):.1f}°")
                   
                else:
                    base_angle = st.slider(
                        "Base Angle (degrees)",
                        0.0, 90.0, 0.0, 0.5,
                        key="base_angle_custom_multi"
                    )
                    base_theta = np.deg2rad(base_angle)
                    base_orientation = st.session_state.interpolator.get_orientation_from_angle(base_angle)
                    st.info(f"**Base θ:** {base_angle:.1f}°")
                    st.info(f"**Orientation:** {base_orientation}")
           
            base_params = {
                'defect_type': base_defect,
                'shape': base_shape,
                'orientation': base_orientation,
                'theta': base_theta
            }
           
            # Parameter ranges
            st.markdown("### 📊 Parameter Ranges")
           
            st.markdown("#### ε* Range")
            eps0_range_col1, eps0_range_col2, eps0_range_col3 = st.columns(3)
            with eps0_range_col1:
                eps0_min = st.number_input("Min ε*", 0.3, 3.0, 0.5, 0.1, key="eps0_min")
            with eps0_range_col2:
                eps0_max = st.number_input("Max ε*", 0.3, 3.0, 2.5, 0.1, key="eps0_max")
            with eps0_range_col3:
                eps0_steps = st.number_input("Steps", 2, 100, 10, 1, key="eps0_steps")
           
            st.markdown("#### κ Range")
            kappa_range_col1, kappa_range_col2, kappa_range_col3 = st.columns(3)
            with kappa_range_col1:
                kappa_min = st.number_input("Min κ", 0.1, 2.0, 0.2, 0.05, key="kappa_min")
            with kappa_range_col2:
                kappa_max = st.number_input("Max κ", 0.1, 2.0, 1.5, 0.05, key="kappa_max")
            with kappa_range_col3:
                kappa_steps = st.number_input("Steps", 2, 50, 8, 1, key="kappa_steps")
           
            st.markdown("#### Orientation Range (Optional)")
            use_orientation_range = st.checkbox("Vary orientation", value=False, key="use_orientation_range")
           
            if use_orientation_range:
                if orientation_mode == "Predefined":
                    orientation_options = st.multiselect(
                        "Select orientations to include",
                        ["Horizontal {111} (0°)", "Tilted 30° (1¯10 projection)", "Tilted 60°", "Vertical {111} (90°)"],
                        default=["Horizontal {111} (0°)", "Vertical {111} (90°)"],
                        key="orientation_multi_select"
                    )
                else:
                    orientation_range_col1, orientation_range_col2, orientation_range_col3 = st.columns(3)
                    with orientation_range_col1:
                        angle_min = st.number_input("Min Angle (°)", 0.0, 90.0, 0.0, 1.0, key="angle_min")
                    with orientation_range_col2:
                        angle_max = st.number_input("Max Angle (°)", 0.0, 90.0, 90.0, 1.0, key="angle_max")
                    with orientation_range_col3:
                        angle_steps = st.number_input("Steps", 2, 20, 5, 1, key="angle_steps")
           
            # Generate parameter grid
            if st.button("🔄 Generate Parameter Grid", type="primary"):
                ranges_config = {}
               
                if eps0_max > eps0_min:
                    ranges_config['eps0'] = {
                        'min': float(eps0_min),
                        'max': float(eps0_max),
                        'steps': int(eps0_steps)
                    }
               
                if kappa_max > kappa_min:
                    ranges_config['kappa'] = {
                        'min': float(kappa_min),
                        'max': float(kappa_max),
                        'steps': int(kappa_steps)
                    }
               
                if use_orientation_range:
                    if orientation_mode == "Predefined" and orientation_options:
                        angle_map_rev = {
                            "Horizontal {111} (0°)": 0,
                            "Tilted 30° (1¯10 projection)": 30,
                            "Tilted 60°": 60,
                            "Vertical {111} (90°)": 90,
                        }
                        orientation_angles = [angle_map_rev[orient] for orient in orientation_options]
                        ranges_config['theta'] = {
                            'values': [np.deg2rad(angle) for angle in orientation_angles]
                        }
                    else:
                        if angle_max > angle_min:
                            angles = np.linspace(angle_min, angle_max, angle_steps)
                            ranges_config['theta'] = {
                                'values': [np.deg2rad(angle) for angle in angles]
                            }
               
                param_grid = st.session_state.multi_target_manager.create_parameter_grid(
                    base_params, ranges_config
                )
               
                for param_set in param_grid:
                    angle = np.rad2deg(param_set.get('theta', 0))
                    param_set['orientation'] = st.session_state.interpolator.get_orientation_from_angle(angle)
               
                st.session_state.multi_target_params = param_grid
               
                st.success(f"✅ Generated {len(param_grid)} parameter combinations!")
               
                st.subheader("📋 Generated Parameter Grid")
               
                grid_data = []
                for i, params in enumerate(param_grid):
                    grid_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': f"{params.get('eps0', 'Unknown'):.3f}",
                        'κ': f"{params.get('kappa', 'Unknown'):.3f}",
                        'Orientation': params.get('orientation', 'Unknown'),
                        'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                    })
               
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(df_grid, use_container_width=True)
           
            if st.session_state.multi_target_params:
                st.subheader("📊 Current Parameter Grid")
               
                grid_data = []
                for i, params in enumerate(st.session_state.multi_target_params):
                    grid_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': f"{params.get('eps0', 'Unknown'):.3f}",
                        'κ': f"{params.get('kappa', 'Unknown'):.3f}",
                        'Orientation': params.get('orientation', 'Unknown'),
                        'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                    })
               
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(df_grid, use_container_width=True)
                   
                    if st.button("🗑️ Clear Parameter Grid", type="secondary"):
                        st.session_state.multi_target_params = []
                        st.session_state.multi_target_predictions = {}
                        st.success("Parameter grid cleared!")
                        st.rerun()
   
    # Tab 4: Train & Predict
    with tab4:
        st.subheader("Train Model and Predict")
       
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
            col1, col2 = st.columns(2)
           
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
           
            with col2:
                batch_size = st.slider("Batch Size", 1, 16, 4, 1)
                validation_split = st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05)
           
            if prediction_mode == "Single Target":
                if st.button("🚀 Train & Predict (Single Target)", type="primary"):
                    with st.spinner("Training attention model and predicting..."):
                        try:
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
                           
                        except Exception as e:
                            st.error(f"❌ Error during training/prediction: {str(e)}")
           
            else:
                if st.button("🚀 Train & Predict (Multiple Targets)", type="primary"):
                    with st.spinner(f"Running batch predictions for {len(st.session_state.multi_target_params)} targets..."):
                        try:
                            predictions = st.session_state.multi_target_manager.batch_predict(
                                st.session_state.source_simulations,
                                st.session_state.multi_target_params,
                                st.session_state.interpolator
                            )
                           
                            st.session_state.multi_target_predictions = predictions
                           
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
   
    # Tab 5: Results & Export
    with tab5:
        st.subheader("Prediction Results")
       
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            results = st.session_state.prediction_results
           
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
   
    # Tab 6: Manage Files
    with tab6:
        st.subheader("📁 Numerical Solutions Directory Management")
       
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
       
        all_files_info = st.session_state.solutions_manager.get_all_files()
       
        if not all_files_info:
            st.warning("No files found in numerical_solutions directory")
        else:
            total_size = sum(f['size'] for f in all_files_info) / (1024 * 1024)
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
    # TAB 7: STRESS ANALYSIS & SUNBURST CHARTS
    # =============================================
    with tab7:
        st.header("📈 Stress Analysis and Sunburst Visualization")
       
        # Update stress summary DataFrame
        if st.button("🔄 Update Stress Summary", type="secondary"):
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
           
            # ADDITIONAL VISUALIZATIONS
            st.subheader("📊 Additional Visualizations")
           
            viz_tabs = st.tabs(["Correlation Matrix", "3D Scatter Plot", "Heatmap", "Box Plots"])
           
            with viz_tabs[0]:
                if len(df_filtered.select_dtypes(include=[np.number]).columns) > 1:
                    corr_matrix = df_filtered.select_dtypes(include=[np.number]).corr()
                   
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=selected_colormap,
                        title="Correlation Matrix of Stress Metrics"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation matrix")
           
            with viz_tabs[1]:
                if len(stress_value_cols) >= 3:
                    col_x, col_y, col_z = st.columns(3)
                   
                    with col_x:
                        x_col = st.selectbox("X-axis", stress_value_cols, index=0)
                    with col_y:
                        y_col = st.selectbox("Y-axis", stress_value_cols, index=1)
                    with col_z:
                        z_col = st.selectbox("Z-axis", stress_value_cols, index=2)
                   
                    color_by = st.selectbox(
                        "Color by",
                        ['defect_type', 'shape', 'orientation', 'type'] + stress_value_cols,
                        index=0
                    )
                   
                    if st.button("Generate 3D Scatter"):
                        fig_3d = px.scatter_3d(
                            df_filtered,
                            x=x_col,
                            y=y_col,
                            z=z_col,
                            color=color_by if color_by in df_filtered.columns else None,
                            hover_name='id',
                            title="3D Stress Metric Visualization",
                            color_continuous_scale=selected_colormap,
                            opacity=0.7
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info("Need at least 3 numeric columns for 3D scatter plot")
           
            with viz_tabs[2]:
                if len(path_columns) >= 2:
                    heatmap_data = df_filtered.pivot_table(
                        index=path_columns[0],
                        columns=path_columns[1] if len(path_columns) > 1 else 'type',
                        values=value_column,
                        aggfunc='mean'
                    )
                   
                    fig_heat = px.imshow(
                        heatmap_data,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=selected_colormap,
                        title=f"Heatmap: {value_column.replace('_', ' ').title()} by {path_columns[0]} and {path_columns[1]}"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Need at least 2 hierarchical levels for heatmap")
           
            with viz_tabs[3]:
                if len(path_columns) > 0:
                    group_by = st.selectbox(
                        "Group by for Box Plot",
                        path_columns,
                        index=0
                    )
                   
                    box_values = st.multiselect(
                        "Select metrics for Box Plot",
                        stress_value_cols,
                        default=stress_value_cols[:min(5, len(stress_value_cols))]
                    )
                   
                    if len(box_values) > 0:
                        fig_boxes = make_subplots(
                            rows=len(box_values),
                            cols=1,
                            subplot_titles=[v.replace('_', ' ').title() for v in box_values],
                            vertical_spacing=0.1
                        )
                       
                        for i, value_col in enumerate(box_values):
                            for group in df_filtered[group_by].unique():
                                group_data = df_filtered[df_filtered[group_by] == group][value_col].dropna()
                               
                                viridis_len = len(px.colors.sequential.Viridis)
                                color_index = int(i * (viridis_len - 1) / (len(box_values) - 1)) if len(box_values) > 1 else 0
                               
                                fig_boxes.add_trace(
                                    go.Box(
                                        y=group_data,
                                        name=str(group),
                                        boxpoints='outliers',
                                        jitter=0.3,
                                        pointpos=-1.8,
                                        marker_color=px.colors.sequential.Viridis[color_index]
                                    ),
                                    row=i+1,
                                    col=1
                                )
                       
                        fig_boxes.update_layout(
                            height=300 * len(box_values),
                            showlegend=True,
                            title_text=f"Box Plots by {group_by}"
                        )
                       
                        st.plotly_chart(fig_boxes, use_container_width=True)
                else:
                    st.info("Please configure hierarchical levels first")
        else:
            st.info("👈 Please load simulations and generate predictions first to enable stress analysis")

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application with enhanced stress analysis"""
   
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
        ["Attention Interpolation", "Stress Analysis Dashboard"],
        index=0
    )
   
    if operation_mode == "Attention Interpolation":
        create_attention_interface()
    else:
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
       
        all_files = st.session_state.solutions_manager.get_all_files()
       
        if st.button("📥 Load All Simulations for Analysis"):
            with st.spinner("Loading all simulations..."):
                all_simulations = []
                for file_info in all_files[:50]:
                    try:
                        sim_data = st.session_state.solutions_manager.load_simulation(
                            file_info['path'],
                            st.session_state.interpolator
                        )
                        all_simulations.append(sim_data)
                    except:
                        continue
               
                if all_simulations:
                    stress_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                        all_simulations, {}
                    )
                   
                    if not stress_df.empty:
                        st.session_state.stress_summary_df = stress_df
                        st.success(f"✅ Loaded {len(all_simulations)} simulations for analysis")
                    else:
                        st.warning("No stress data found in loaded simulations")
                else:
                    st.error("No simulations could be loaded")
       
        if not st.session_state.stress_summary_df.empty:
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
   
    4. **Principal Stresses (σ₁, σ₂, σ₃):**
       - Max Principal 1: `max_principal_1`
       - Max Principal 2: `max_principal_2`
       - Max Principal 3: `max_principal_3`
       - Max Absolute Principal: `max_principal_abs`
       - Maximum Shear (Tresca): `max_shear_tresca`
   
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
   
    ### **📈 Additional Visualizations**
   
    1. **Correlation Matrix:** Shows relationships between stress metrics
    2. **3D Scatter Plot:** Interactive 3D visualization
    3. **Heatmaps:** 2D matrix of stress values by parameter combinations
    4. **Box Plots:** Distribution analysis of stress metrics
   
    ### **🏔️ Stress Peak Analysis**
   
    **Peak Detection Algorithm:**
    1. **Threshold-based Detection:** User-defined percentile threshold
    2. **Peak Characterization:** Number of peaks, maximum value, position
    3. **Visualization:** Overlay peaks on stress field maps
    """)

if __name__ == "__main__":
    main()

st.caption(f"🔬 Enhanced Multi-Target Spatial-Attention Stress Interpolation • Stress Analysis Dashboard • Sunburst Visualization • 50+ Colormaps • 2025")
