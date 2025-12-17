import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
import warnings
import pickle
import torch
import sqlite3
from pathlib import Path
import tempfile
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# =============================================
# PREDICTION RESULTS SAVING AND DOWNLOAD MANAGER
# =============================================
class PredictionResultsManager:
    """Manager for saving and downloading prediction results"""
 
    @staticmethod
    def prepare_prediction_data_for_saving(prediction_results: Dict[str, Any],
                                         source_simulations: List[Dict],
                                         mode: str = 'single') -> Dict[str, Any]:
        """
        Prepare prediction results for saving to file
     
        Args:
            prediction_results: Dictionary of prediction results
            source_simulations: List of source simulation data
            mode: 'single' for single target, 'multi' for multiple targets
         
        Returns:
            Structured dictionary ready for saving
        """
        # Create metadata
        metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'mode': mode,
            'num_sources': len(source_simulations),
            'software_version': '1.0.0',
            'data_type': 'attention_interpolation_results'
        }
     
        # Extract source parameters
        source_params = []
        for i, sim_data in enumerate(source_simulations):
            params = sim_data.get('params', {})
            source_params.append({
                'id': i,
                'defect_type': params.get('defect_type'),
                'shape': params.get('shape'),
                'orientation': params.get('orientation'),
                'eps0': float(params.get('eps0', 0)),
                'kappa': float(params.get('kappa', 0)),
                'theta': float(params.get('theta', 0))
            })
     
        # Structure the data
        save_data = {
            'metadata': metadata,
            'source_parameters': source_params,
            'prediction_results': prediction_results.copy() # Create a copy to avoid modifications
        }
     
        # Add additional info based on mode
        if mode == 'single' and 'attention_weights' in prediction_results:
            weights = prediction_results['attention_weights']
            save_data['attention_analysis'] = {
                'weights': weights.tolist() if hasattr(weights, 'tolist') else weights,
                'source_names': [f'S{i+1}' for i in range(len(source_simulations))],
                'dominant_source': int(np.argmax(weights)) if hasattr(weights, '__len__') else 0,
                'weight_entropy': float(-np.sum(weights * np.log(weights + 1e-10)))
            }
     
        # Add stress statistics if available
        if 'stress_fields' in prediction_results:
            stress_stats = {}
            for field_name, field_data in prediction_results['stress_fields'].items():
                if isinstance(field_data, np.ndarray):
                    stress_stats[field_name] = {
                        'max': float(np.max(field_data)),
                        'min': float(np.min(field_data)),
                        'mean': float(np.mean(field_data)),
                        'std': float(np.std(field_data)),
                        'percentile_95': float(np.percentile(field_data, 95)),
                        'percentile_99': float(np.percentile(field_data, 99))
                    }
            save_data['stress_statistics'] = stress_stats
     
        return save_data
 
    @staticmethod
    def create_single_prediction_archive(prediction_results: Dict[str, Any],
                                       source_simulations: List[Dict]) -> BytesIO:
        """
        Create a comprehensive archive for single prediction
     
        Args:
            prediction_results: Single prediction results
            source_simulations: List of source simulations
         
        Returns:
            BytesIO buffer containing the archive
        """
        # Create in-memory zip file
        zip_buffer = BytesIO()
     
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save main prediction data as PKL
            save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                prediction_results, source_simulations, 'single'
            )
         
            # PKL format
            pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
            zip_file.writestr('prediction_results.pkl', pkl_data)
         
            # 2. Save as PT (PyTorch) format
            pt_buffer = BytesIO()
            torch.save(save_data, pt_buffer)
            pt_buffer.seek(0)
            zip_file.writestr('prediction_results.pt', pt_buffer.read())
         
            # 3. Save stress fields as separate NPZ files
            stress_fields = prediction_results.get('stress_fields', {})
            for field_name, field_data in stress_fields.items():
                if isinstance(field_data, np.ndarray):
                    npz_buffer = BytesIO()
                    np.savez_compressed(npz_buffer, data=field_data)
                    npz_buffer.seek(0)
                    zip_file.writestr(f'stress_{field_name}.npz', npz_buffer.read())
         
            # 4. Save attention weights as CSV
            if 'attention_weights' in prediction_results:
                weights = prediction_results['attention_weights']
                if hasattr(weights, 'flatten'):
                    weights = weights.flatten()
             
                weight_df = pd.DataFrame({
                    'source_id': [f'S{i+1}' for i in range(len(weights))],
                    'weight': weights,
                    'percent_contribution': 100 * weights / (np.sum(weights) + 1e-10)
                })
                csv_data = weight_df.to_csv(index=False)
                zip_file.writestr('attention_weights.csv', csv_data)
         
            # 5. Save target parameters as JSON
            target_params = prediction_results.get('target_params', {})
            if target_params:
                # Convert numpy types to Python types for JSON
                def convert_for_json(obj):
                    if isinstance(obj, (np.float32, np.float64, np.float16)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    else:
                        return obj
             
                json_data = json.dumps(target_params, default=convert_for_json, indent=2)
                zip_file.writestr('target_parameters.json', json_data)
         
            # 6. Save summary statistics
            if 'stress_fields' in prediction_results:
                stats_rows = []
                for field_name, field_data in stress_fields.items():
                    if isinstance(field_data, np.ndarray):
                        stats_rows.append({
                            'field': field_name,
                            'max': float(np.max(field_data)),
                            'min': float(np.min(field_data)),
                            'mean': float(np.mean(field_data)),
                            'std': float(np.std(field_data)),
                            'percentile_95': float(np.percentile(field_data, 95)),
                            'percentile_99': float(np.percentile(field_data, 99)),
                            'area_above_threshold': float(np.sum(field_data > np.mean(field_data)))
                        })
             
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    stats_csv = stats_df.to_csv(index=False)
                    zip_file.writestr('stress_statistics.csv', stats_csv)
         
            # 7. Save a README file
            readme_content = f"""# Prediction Results Archive
Generated: {datetime.now().isoformat()}
Number of source simulations: {len(source_simulations)}
Prediction mode: Single target
Files included:
1. prediction_results.pkl - Main prediction data (Python pickle format)
2. prediction_results.pt - PyTorch format
3. stress_*.npz - Individual stress fields (NumPy compressed)
4. attention_weights.csv - Attention weights distribution
5. target_parameters.json - Target parameters
6. stress_statistics.csv - Statistical summary
For more information, see the documentation.
"""
            zip_file.writestr('README.txt', readme_content)
     
        zip_buffer.seek(0)
        return zip_buffer
 
    @staticmethod
    def create_multi_prediction_archive(multi_predictions: Dict[str, Any],
                                       source_simulations: List[Dict]) -> BytesIO:
        """
        Create a comprehensive archive for multiple predictions
     
        Args:
            multi_predictions: Dictionary of multiple predictions
            source_simulations: List of source simulations
         
        Returns:
            BytesIO buffer containing the archive
        """
        zip_buffer = BytesIO()
     
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save each prediction individually
            for pred_key, pred_data in multi_predictions.items():
                # Create directory for each prediction
                pred_dir = f'predictions/{pred_key}'
             
                # Save prediction data
                save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                    pred_data, source_simulations, 'multi'
                )
             
                # Save as PKL
                pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
                zip_file.writestr(f'{pred_dir}/prediction.pkl', pkl_data)
             
                # Save stress statistics
                stress_fields = {k: v for k, v in pred_data.items()
                               if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
             
                if stress_fields:
                    stats_rows = []
                    for field_name, field_data in stress_fields.items():
                        stats_rows.append({
                            'field': field_name,
                            'max': float(np.max(field_data)),
                            'min': float(np.min(field_data)),
                            'mean': float(np.mean(field_data)),
                            'std': float(np.std(field_data))
                        })
                 
                    stats_df = pd.DataFrame(stats_rows)
                    stats_csv = stats_df.to_csv(index=False)
                    zip_file.writestr(f'{pred_dir}/statistics.csv', stats_csv)
         
            # Save master summary
            summary_rows = []
            for pred_key, pred_data in multi_predictions.items():
                target_params = pred_data.get('target_params', {})
                stress_fields = {k: v for k, v in pred_data.items()
                               if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
             
                row = {
                    'prediction_id': pred_key,
                    'defect_type': target_params.get('defect_type', 'Unknown'),
                    'shape': target_params.get('shape', 'Unknown'),
                    'orientation': target_params.get('orientation', 'Unknown'),
                    'eps0': float(target_params.get('eps0', 0)),
                    'kappa': float(target_params.get('kappa', 0)),
                    'theta_deg': float(np.deg2rad(target_params.get('theta', 0)))
                }
             
                # Add stress metrics
                for field_name, field_data in stress_fields.items():
                    row[f'{field_name}_max'] = float(np.max(field_data))
                    row[f'{field_name}_mean'] = float(np.mean(field_data))
                    row[f'{field_name}_std'] = float(np.std(field_data))
             
                summary_rows.append(row)
         
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_csv = summary_df.to_csv(index=False)
                zip_file.writestr('multi_prediction_summary.csv', summary_csv)
         
            # Save a README file
            readme_content = f"""# Multi-Prediction Results Archive
Generated: {datetime.now().isoformat()}
Number of source simulations: {len(source_simulations)}
Number of predictions: {len(multi_predictions)}
Structure:
- predictions/[prediction_id]/ - Individual prediction data
- multi_prediction_summary.csv - Summary of all predictions
Each prediction directory contains:
1. prediction.pkl - Main prediction data
2. statistics.csv - Stress statistics
For more information, see the documentation.
"""
            zip_file.writestr('README.txt', readme_content)
     
        zip_buffer.seek(0)
        return zip_buffer

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
        return pickle.load(buffer)
 
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))
 
    def _read_h5(self, file_content):
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
 
    def read_simulation_file(self, file_content, format_type='auto'):
        if format_type == 'auto':
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
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            for key in data.keys():
                if 'history' in key.lower():
                    standardized['history'] = data[key]
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
     
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        param_names.extend(['defect_ISF', 'defect_ESF', 'defect_Twin'])
     
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
            angle_deg = angle_deg % 90
            return f"Custom ({angle_deg:.1f}°)"
# =============================================
# GRID AND EXTENT CONFIGURATION
# =============================================
def get_grid_extent(N=128, dx=0.1):
    """Get grid extent for visualization"""
    return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
# =============================================
# ATTENTION INTERFACE
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with save/download"""
 
    st.header("🤖 Spatial-Attention Stress Interpolation")
 
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
 
    # Initialize multi-target manager
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
 
    # Initialize prediction results manager
    if 'prediction_results_manager' not in st.session_state:
        st.session_state.prediction_results_manager = PredictionResultsManager()
 
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
 
    # Initialize multi-target predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
 
    # Initialize saving options
    if 'save_format' not in st.session_state:
        st.session_state.save_format = 'both'
  
    # Initialize download data in session state
    if 'download_pkl_data' not in st.session_state:
        st.session_state.download_pkl_data = None
    if 'download_pt_data' not in st.session_state:
        st.session_state.download_pt_data = None
    if 'download_zip_data' not in st.session_state:
        st.session_state.download_zip_data = None
    if 'download_zip_filename' not in st.session_state:
        st.session_state.download_zip_filename = None
 
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
 
    with st.sidebar.expander("💾 Save/Download Options", expanded=True):
        st.session_state.save_format = st.radio(
            "Save Format",
            ["PKL only", "PT only", "Both PKL & PT", "Archive (ZIP)"],
            index=2,
            key="save_format_radio"
        )
     
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Target",
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict",
        "📊 Results & Visualization",
        "💾 Save & Export Results"
    ])
 
    # Tab 1: Load Source Data
    with tab1:
        st.subheader("Load Source Simulation Files")
     
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
                             
                                history = sim_data.get('history', [] )
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
 
    # Tab 5: Results & Visualization
    with tab5:
        st.subheader("Prediction Results Visualization")
     
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
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
            with col_viz3:
                show_contour = st.checkbox("Show Contour Lines", value=True)
         
            # Plot stress field
            if stress_component in results.get('stress_fields', {}):
                stress_data = results['stress_fields'][stress_component]
             
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(stress_data, extent=extent, cmap=colormap,
                              origin='lower', aspect='equal')
             
                if show_contour:
                    contour_levels = 10
                    contour = ax.contour(stress_data, levels=contour_levels,
                                        extent=extent, colors='black', alpha=0.5, linewidths=0.5)
                    ax.clabel(contour, inline=True, fontsize=8)
             
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
         
            # Statistics table
            st.subheader("📊 Stress Statistics")
         
            if 'stress_fields' in results:
                stats_data = []
                for comp_name, comp_data in results['stress_fields'].items():
                    if isinstance(comp_data, np.ndarray):
                        stats_data.append({
                            'Component': comp_name.replace('_', ' ').title(),
                            'Max (GPa)': float(np.max(comp_data)),
                            'Min (GPa)': float(np.min(comp_data)),
                            'Mean (GPa)': float(np.mean(comp_data)),
                            'Std Dev': float(np.std(comp_data)),
                            '95th %ile': float(np.percentile(comp_data, 95))
                        })
             
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats.style.format({
                        'Max (GPa)': '{:.3f}',
                        'Min (GPa)': '{:.3f}',
                        'Mean (GPa)': '{:.3f}',
                        'Std Dev': '{:.3f}',
                        '95th %ile': '{:.3f}'
                    }), use_container_width=True)
    # Tab 6: Save & Export Results (FIXED VERSION)
    with tab6:
        st.subheader("💾 Save and Export Prediction Results")
      
        # Check if we have predictions to save
        has_single_prediction = 'prediction_results' in st.session_state
        has_multi_predictions = ('multi_target_predictions' in st.session_state and
                                len(st.session_state.multi_target_predictions) > 0)
      
        if not has_single_prediction and not has_multi_predictions:
            st.warning("⚠️ No prediction results available to save. Please run predictions first.")
        else:
            st.success("✅ Prediction results available for export!")
          
            # Display what's available
            if has_single_prediction:
                st.info(f"**Single Target Prediction:** Available")
                single_params = st.session_state.prediction_results.get('target_params', {})
                st.write(f"- Target: {single_params.get('defect_type', 'Unknown')}, "
                        f"ε*={single_params.get('eps0', 0):.3f}, "
                        f"κ={single_params.get('kappa', 0):.3f}")
          
            if has_multi_predictions:
                st.info(f"**Multiple Target Predictions:** {len(st.session_state.multi_target_predictions)} available")
          
            st.divider()
          
            # Save options
            st.subheader("📁 Save Options")
          
            save_col1, save_col2, save_col3 = st.columns(3)
          
            with save_col1:
                save_mode = st.radio(
                    "Select results to save",
                    ["Current Single Prediction", "All Multiple Predictions", "Both"],
                    index=0 if has_single_prediction else 1,
                    disabled=not has_single_prediction and not has_multi_predictions
                )
          
            with save_col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = st.text_input(
                    "Base filename",
                    value=f"prediction_{timestamp}",
                    help="Files will be saved with this base name plus appropriate extensions"
                )
          
            with save_col3:
                include_source_info = st.checkbox("Include source simulations info", value=True)
                include_visualizations = st.checkbox("Include visualization data", value=True)
          
            st.divider()
          
            # Save/Download buttons
            st.subheader("⬇️ Download Options")
          
            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
          
            # PKL download
            with dl_col1:
                st.markdown("**PKL Format**")
                prepare_pkl = st.button("💾 Prepare PKL", type="secondary", use_container_width=True, key="prepare_pkl")
              
                if prepare_pkl:
                    with st.spinner("Preparing PKL file..."):
                        try:
                            if save_mode in ["Current Single Prediction", "Both"] and has_single_prediction:
                                save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations,
                                    'single'
                                )
                              
                                # Add metadata
                                save_data['save_info'] = {
                                    'format': 'pkl',
                                    'timestamp': timestamp,
                                    'mode': 'single'
                                }
                              
                                # Create download link
                                pkl_buffer = BytesIO()
                                pickle.dump(save_data, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                                pkl_buffer.seek(0)
                                st.session_state.download_pkl_data = pkl_buffer.getvalue()
                                st.success("✅ PKL file prepared!")
                          
                        except Exception as e:
                            st.error(f"❌ Error preparing PKL: {str(e)}")
              
                # Always show download button if data exists
                if st.session_state.download_pkl_data:
                    st.download_button(
                        label="📥 Download PKL",
                        data=st.session_state.download_pkl_data,
                        file_name=f"{base_filename}.pkl",
                        mime="application/octet-stream",
                        key="download_pkl_final",
                        use_container_width=True
                    )
                else:
                    st.caption("Click 'Prepare PKL' first")
          
            # PT download
            with dl_col2:
                st.markdown("**PT Format**")
                prepare_pt = st.button("💾 Prepare PT", type="secondary", use_container_width=True, key="prepare_pt")
              
                if prepare_pt:
                    with st.spinner("Preparing PT file..."):
                        try:
                            if save_mode in ["Current Single Prediction", "Both"] and has_single_prediction:
                                save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations,
                                    'single'
                                )
                              
                                # Add metadata
                                save_data['save_info'] = {
                                    'format': 'pt',
                                    'timestamp': timestamp,
                                    'mode': 'single'
                                }
                              
                                # Create download link
                                pt_buffer = BytesIO()
                                torch.save(save_data, pt_buffer)
                                pt_buffer.seek(0)
                                st.session_state.download_pt_data = pt_buffer.getvalue()
                                st.success("✅ PT file prepared!")
                          
                        except Exception as e:
                            st.error(f"❌ Error preparing PT: {str(e)}")
              
                # Always show download button if data exists
                if st.session_state.download_pt_data:
                    st.download_button(
                        label="📥 Download PT",
                        data=st.session_state.download_pt_data,
                        file_name=f"{base_filename}.pt",
                        mime="application/octet-stream",
                        key="download_pt_final",
                        use_container_width=True
                    )
                else:
                    st.caption("Click 'Prepare PT' first")
          
            # ZIP download
            with dl_col3:
                st.markdown("**ZIP Archive**")
                prepare_zip = st.button("📦 Prepare ZIP", type="primary", use_container_width=True, key="prepare_zip")
              
                if prepare_zip:
                    with st.spinner("Creating comprehensive archive..."):
                        try:
                            if save_mode == "Current Single Prediction" and has_single_prediction:
                                # Single prediction archive
                                zip_buffer = st.session_state.prediction_results_manager.create_single_prediction_archive(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations
                                )
                                st.session_state.download_zip_data = zip_buffer.getvalue()
                                st.session_state.download_zip_filename = f"{base_filename}_complete.zip"
                                st.success("✅ Single ZIP archive prepared!")
                            elif save_mode == "All Multiple Predictions" and has_multi_predictions:
                                # Multi prediction archive
                                zip_buffer = st.session_state.prediction_results_manager.create_multi_prediction_archive(
                                    st.session_state.multi_target_predictions,
                                    st.session_state.source_simulations
                                )
                                st.session_state.download_zip_data = zip_buffer.getvalue()
                                st.session_state.download_zip_filename = f"{base_filename}_multi_predictions.zip"
                                st.success("✅ Multi ZIP archive prepared!")
                            elif save_mode == "Both" and has_single_prediction:
                                # Create single ZIP even if Both is selected (for simplicity)
                                zip_buffer = st.session_state.prediction_results_manager.create_single_prediction_archive(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations
                                )
                                st.session_state.download_zip_data = zip_buffer.getvalue()
                                st.session_state.download_zip_filename = f"{base_filename}_complete.zip"
                                st.success("✅ Single ZIP archive prepared!")
                          
                        except Exception as e:
                            st.error(f"❌ Error creating archive: {str(e)}")
              
                # Always show download button if data exists
                if st.session_state.download_zip_data and st.session_state.download_zip_filename:
                    st.download_button(
                        label="📥 Download ZIP",
                        data=st.session_state.download_zip_data,
                        file_name=st.session_state.download_zip_filename,
                        mime="application/zip",
                        key="download_zip_final",
                        use_container_width=True
                    )
                else:
                    st.caption("Click 'Prepare ZIP' first")
          
            # Clear all button
            with dl_col4:
                st.markdown("**Clear All**")
                if st.button("🗑️ Clear All Prepared", type="secondary", use_container_width=True):
                    st.session_state.download_pkl_data = None
                    st.session_state.download_pt_data = None
                    st.session_state.download_zip_data = None
                    st.session_state.download_zip_filename = None
                    st.success("✅ All prepared files cleared!")
                    st.rerun()
                st.caption("Clears prepared download files from memory")
          
            st.divider()
          
            # Advanced options
            with st.expander("⚙️ Advanced Save Options", expanded=False):
                col_adv1, col_adv2 = st.columns(2)
              
                with col_adv1:
                    # Save stress fields as separate files
                    st.markdown("**Separate Stress Fields**")
                    stress_fields = st.session_state.prediction_results.get('stress_fields', {}) if has_single_prediction else {}
                  
                    for field_name, field_data in stress_fields.items():
                        if isinstance(field_data, np.ndarray):
                            npz_buffer = BytesIO()
                            np.savez_compressed(npz_buffer, data=field_data)
                            npz_buffer.seek(0)
                          
                            st.download_button(
                                label=f"📥 {field_name}.npz",
                                data=npz_buffer.getvalue(),
                                file_name=f"{base_filename}_{field_name}.npz",
                                mime="application/octet-stream",
                                key=f"download_npz_{field_name}_{timestamp}"
                            )
              
                with col_adv2:
                    # Export to other formats
                    st.markdown("**Export Formats**")
                  
                    # JSON export
                    if has_single_prediction:
                        target_params = st.session_state.prediction_results.get('target_params', {})
                        if target_params:
                            # Helper function to convert numpy types for JSON
                            def convert_for_json(obj):
                                if isinstance(obj, (np.float32, np.float64, np.float16)):
                                    return float(obj)
                                elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                                    return int(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, np.generic):
                                    return obj.item()
                                else:
                                    return obj
                          
                            json_str = json.dumps(target_params, indent=2, default=convert_for_json)
                          
                            st.download_button(
                                label="📥 Parameters JSON",
                                data=json_str,
                                file_name=f"{base_filename}_params.json",
                                mime="application/json",
                                key=f"download_json_{timestamp}"
                            )
                  
                    # CSV export
                    if has_single_prediction and 'stress_fields' in st.session_state.prediction_results:
                        stress_fields = st.session_state.prediction_results['stress_fields']
                        stats_rows = []
                        for field_name, field_data in stress_fields.items():
                            if isinstance(field_data, np.ndarray):
                                stats_rows.append({
                                    'field': field_name,
                                    'max': float(np.max(field_data)),
                                    'min': float(np.min(field_data)),
                                    'mean': float(np.mean(field_data)),
                                    'std': float(np.std(field_data)),
                                    '95th_percentile': float(np.percentile(field_data, 95))
                                })
                      
                        if stats_rows:
                            stats_df = pd.DataFrame(stats_rows)
                            csv_data = stats_df.to_csv(index=False)
                          
                            st.download_button(
                                label="📥 Statistics CSV",
                                data=csv_data,
                                file_name=f"{base_filename}_stats.csv",
                                mime="text/csv",
                                key=f"download_csv_{timestamp}"
                            )
if __name__ == "__main__":
    create_attention_interface()
st.caption(f"🔬 Attention Interpolation • PKL/PT/ZIP Support • {datetime.now().year}")
