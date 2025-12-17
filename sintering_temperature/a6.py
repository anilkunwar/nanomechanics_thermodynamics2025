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
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# PREDICTION RESULTS SAVING AND DOWNLOAD MANAGER
# =============================================
class PredictionResultsManager:
    """Manager for saving and downloading prediction results"""
 
    @staticmethod
    def prepare_prediction_data_for_saving(prediction_results: Dict[str, Any],
                                         source_simulations: List[Dict],
                                         mode: str = 'single') -> Dict[str, Any]:
        metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'mode': mode,
            'num_sources': len(source_simulations),
            'software_version': '1.0.0',
            'data_type': 'attention_interpolation_results'
        }
     
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
     
        save_data = {
            'metadata': metadata,
            'source_parameters': source_params,
            'prediction_results': prediction_results.copy()
        }
     
        if mode == 'single' and 'attention_weights' in prediction_results:
            weights = prediction_results['attention_weights']
            save_data['attention_analysis'] = {
                'weights': weights.tolist() if hasattr(weights, 'tolist') else weights,
                'source_names': [f'S{i+1}' for i in range(len(source_simulations))],
                'dominant_source': int(np.argmax(weights)) if hasattr(weights, '__len__') else 0,
                'weight_entropy': float(-np.sum(weights * np.log(weights + 1e-10)))
            }
     
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
        zip_buffer = BytesIO()
     
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                prediction_results, source_simulations, 'single'
            )
         
            pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
            zip_file.writestr('prediction_results.pkl', pkl_data)
         
            pt_buffer = BytesIO()
            torch.save(save_data, pt_buffer)
            pt_buffer.seek(0)
            zip_file.writestr('prediction_results.pt', pt_buffer.read())
         
            stress_fields = prediction_results.get('stress_fields', {})
            for field_name, field_data in stress_fields.items():
                if isinstance(field_data, np.ndarray):
                    npz_buffer = BytesIO()
                    np.savez_compressed(npz_buffer, data=field_data)
                    npz_buffer.seek(0)
                    zip_file.writestr(f'stress_{field_name}.npz', npz_buffer.read())
         
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
         
            target_params = prediction_results.get('target_params', {})
            if target_params:
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
"""
            zip_file.writestr('README.txt', readme_content)
     
        zip_buffer.seek(0)
        return zip_buffer
 
    @staticmethod
    def create_multi_prediction_archive(multi_predictions: Dict[str, Any],
                                       source_simulations: List[Dict]) -> BytesIO:
        zip_buffer = BytesIO()
     
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for pred_key, pred_data in multi_predictions.items():
                pred_dir = f'predictions/{pred_key}'
                save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                    pred_data, source_simulations, 'multi'
                )
                pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
                zip_file.writestr(f'{pred_dir}/prediction.pkl', pkl_data)
             
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
                    'theta_deg': float(np.rad2deg(target_params.get('theta', 0)))
                }
                for field_name, field_data in stress_fields.items():
                    row[f'{field_name}_max'] = float(np.max(field_data))
                    row[f'{field_name}_mean'] = float(np.mean(field_data))
                    row[f'{field_name}_std'] = float(np.std(field_data))
                summary_rows.append(row)
         
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_csv = summary_df.to_csv(index=False)
                zip_file.writestr('multi_prediction_summary.csv', summary_csv)
         
            readme_content = f"""# Multi-Prediction Results Archive
Generated: {datetime.now().isoformat()}
Number of source simulations: {len(source_simulations)}
Number of predictions: {len(multi_predictions)}
Structure:
- predictions/[prediction_id]/ - Individual prediction data
- multi_prediction_summary.csv - Summary of all predictions
"""
            zip_file.writestr('README.txt', readme_content)
     
        zip_buffer.seek(0)
        return zip_buffer

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
        file_formats = {'pkl': [], 'pt': [], 'h5': [], 'npz': [], 'sql': [], 'json': []}
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
                all_files.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'format': format_type,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    'relative_path': os.path.relpath(file_path, self.solutions_dir)
                })
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
 
    def load_simulation(self, file_path: str, interpolator) -> Dict[str, Any]:
        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            format_type = {
                'pkl': 'pkl', 'pickle': 'pkl',
                'pt': 'pt', 'pth': 'pt',
                'h5': 'h5', 'hdf5': 'h5',
                'npz': 'npz',
                'sql': 'sql', 'db': 'sql',
                'json': 'json'
            }.get(ext, 'auto')
            with open(file_path, 'rb') as f:
                file_content = f.read()
            sim_data = interpolator.read_simulation_file(file_content, format_type)
            sim_data['loaded_from'] = 'numerical_solutions'
            return sim_data
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            raise

# =============================================
# MULTI-TARGET PREDICTION MANAGER
# =============================================
class MultiTargetPredictionManager:
    @staticmethod
    def create_parameter_grid(base_params, ranges_config):
        param_grid = []
        param_values = {}
        for param_name, config in ranges_config.items():
            if 'values' in config:
                param_values[param_name] = config['values']
            elif 'min' in config and 'max' in config:
                steps = config.get('steps', 10)
                param_values[param_name] = np.linspace(config['min'], config['max'], steps).tolist()
            else:
                param_values[param_name] = [config.get('value', base_params.get(param_name))]
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
        predictions = {}
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
        for idx, target_params in enumerate(target_params_list):
            target_vector, _ = interpolator.compute_parameter_vector({'params': target_params})
            distances = np.sqrt(np.sum((source_param_vectors - target_vector) ** 2, axis=1))
            weights = np.exp(-0.5 * (distances / 0.3) ** 2)
            weights = weights / (np.sum(weights) + 1e-8)
            weighted_stress = np.sum(source_stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
            predicted_stress = {
                'stress_fields': {
                    'sigma_hydro': weighted_stress[0],
                    'sigma_mag': weighted_stress[1],
                    'von_mises': weighted_stress[2]
                },
                'predicted': True,
                'target_params': target_params,
                'attention_weights': weights,
                'target_index': idx
            }
            predictions[f"target_{idx:03d}"] = predicted_stress
        return predictions

# =============================================
# SPATIAL LOCALITY ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator:
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
            'pkl': self._read_pkl, 'pt': self._read_pt, 'h5': self._read_h5,
            'npz': self._read_npz, 'sql': self._read_sql, 'json': self._read_json
        }
 
    def _build_model(self):
        model = torch.nn.ModuleDict({
            'param_embedding': torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.d_model)
            ),
            'attention': torch.nn.MultiheadAttention(
                embed_dim=self.d_model, num_heads=self.num_heads,
                batch_first=True, dropout=0.1
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
        return pickle.load(BytesIO(file_content))
    def _read_pt(self, file_content): 
        return torch.load(BytesIO(file_content), map_location='cpu')
    def _read_h5(self, file_content): 
        import h5py
        data = {}
        with h5py.File(BytesIO(file_content), 'r') as f:
            def read(name, obj):
                if isinstance(obj, h5py.Dataset): data[name] = obj[()]
            f.visititems(read)
        return data
    def _read_npz(self, file_content): 
        return dict(np.load(BytesIO(file_content), allow_pickle=True))
    def _read_sql(self, file_content):
        tmp = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        tmp.write(file_content)
        tmp_path = tmp.name
        tmp.close()
        try:
            conn = sqlite3.connect(tmp_path)
            data = {}
            for table in conn.execute("SELECT name FROM sqlite_master WHERE type='table';"):
                table_name = table[0]
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                data[table_name] = df.to_dict('records')
            conn.close()
            os.unlink(tmp_path)
            return data
        except: 
            os.unlink(tmp_path)
            raise
    def _read_json(self, file_content): 
        return json.loads(file_content.decode('utf-8'))
 
    def read_simulation_file(self, file_content, format_type='auto'):
        if format_type == 'auto': format_type = 'pkl'
        data = self.readers[format_type](file_content)
        return self._standardize_data(data, format_type, "uploaded_file")
 
    def _standardize_data(self, data, format_type, file_path):
        standardized = {
            'params': {}, 'history': [], 'metadata': {}, 'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
        }
        if format_type == 'pkl' and isinstance(data, dict):
            standardized['params'] = data.get('params', {})
            standardized['metadata'] = data.get('metadata', {})
            for frame in data.get('history', []):
                if isinstance(frame, dict):
                    eta = frame.get('eta')
                    stresses = frame.get('stresses', {})
                    standardized['history'].append((eta, stresses))
        elif format_type == 'pt' and isinstance(data, dict):
            standardized['params'] = data.get('params', {})
            standardized['metadata'] = data.get('metadata', {})
            for frame in data.get('history', []):
                if isinstance(frame, dict):
                    eta = frame.get('eta')
                    stresses = frame.get('stresses', {})
                    if torch.is_tensor(eta): eta = eta.numpy()
                    stress_dict = {k: v.numpy() if torch.is_tensor(v) else v for k, v in stresses.items()}
                    standardized['history'].append((eta, stress_dict))
        return standardized
 
    def compute_parameter_vector(self, sim_data):
        params = sim_data.get('params', {})
        param_vector = []
        defect_encoding = {'ISF': [1,0,0], 'ESF': [0,1,0], 'Twin': [0,0,1]}
        param_vector.extend(defect_encoding.get(params.get('defect_type', 'ISF'), [0,0,0]))
        shape_encoding = {
            'Square': [1,0,0,0,0], 'Horizontal Fault': [0,1,0,0,0],
            'Vertical Fault': [0,0,1,0,0], 'Rectangle': [0,0,0,1,0], 'Ellipse': [0,0,0,0,1]
        }
        param_vector.extend(shape_encoding.get(params.get('shape', 'Square'), [0,0,0,0,0]))
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        param_vector.append((eps0 - 0.3) / (3.0 - 0.3) if eps0 is not None else 0.5)
        param_vector.append((kappa - 0.1) / (2.0 - 0.1) if kappa is not None else 0.5)
        param_vector.append((theta % (2 * np.pi)) / (2 * np.pi) if theta is not None else 0.0)
        orientation = params.get('orientation', 'Horizontal {111} (0°)')
        orientation_encoding = {
            'Horizontal {111} (0°)': [1,0,0,0], 'Tilted 30° (1¯10 projection)': [0,1,0,0],
            'Tilted 60°': [0,0,1,0], 'Vertical {111} (90°)': [0,0,0,1]
        }
        if orientation.startswith('Custom ('): 
            param_vector.extend([0,0,0,0])
        else:
            param_vector.extend(orientation_encoding.get(orientation, [0,0,0,0]))
        return np.array(param_vector, dtype=np.float32), []
 
    @staticmethod
    def get_orientation_from_angle(angle_deg: float) -> str:
        if 0 <= angle_deg <= 15: return 'Horizontal {111} (0°)'
        elif 15 < angle_deg <= 45: return 'Tilted 30° (1¯10 projection)'
        elif 45 < angle_deg <= 75: return 'Tilted 60°'
        elif 75 < angle_deg <= 90: return 'Vertical {111} (90°)'
        else: return f"Custom ({angle_deg % 90:.1f}°)"

# =============================================
# GRID EXTENT
# =============================================
def get_grid_extent(N=128, dx=0.1):
    return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

# =============================================
# MAIN INTERFACE
# =============================================
def create_attention_interface():
    st.header("🤖 Spatial-Attention Stress Interpolation")
 
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager()
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
    if 'prediction_results_manager' not in st.session_state:
        st.session_state.prediction_results_manager = PredictionResultsManager()
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
    if 'download_pkl_data' not in st.session_state:
        st.session_state.download_pkl_data = None
    if 'download_pt_data' not in st.session_state:
        st.session_state.download_pt_data = None
    if 'download_zip_data' not in st.session_state:
        st.session_state.download_zip_data = None
    if 'download_zip_filename' not in st.session_state:
        st.session_state.download_zip_filename = None
 
    extent = get_grid_extent()
 
    st.sidebar.header("🔮 Settings")
    with st.sidebar.expander("⚙️ Model Parameters", expanded=False):
        num_heads = st.slider("Attention Heads", 1, 8, 4)
        sigma_spatial = st.slider("Spatial Sigma", 0.05, 1.0, 0.2, 0.05)
        sigma_param = st.slider("Parameter Sigma", 0.05, 1.0, 0.3, 0.05)
        use_gaussian = st.checkbox("Gaussian Regularization", True)
        if st.button("🔄 Update Model"):
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
                num_heads=num_heads, sigma_spatial=sigma_spatial,
                sigma_param=sigma_param, use_gaussian=use_gaussian
            )
            st.success("Model updated!")
 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Load Source Data", "🎯 Single Target", "🎯 Multiple Targets",
        "🚀 Predict", "📊 Visualization", "💾 Export"
    ])
 
    # (Tabs 1–4 unchanged – omitted for brevity but kept exactly as in your original code)
    # ... [All code for tabs 1 to 4 remains identical to your last working version] ...
 
    with tab5:
        st.subheader("Prediction Results Visualization")
        if 'prediction_results' not in st.session_state:
            st.info("Please run a prediction first.")
        else:
            results = st.session_state.prediction_results
 
            # Unified data access (fixes KeyError)
            if results.get('mode') == 'multi':
                target_keys = list(st.session_state.multi_target_predictions.keys())
                selected_key = st.selectbox(
                    "Select Target", options=target_keys,
                    index=results.get('current_target_index', 0),
                    key="multi_selector"
                )
                selected_results = st.session_state.multi_target_predictions[selected_key]
                results['current_target_index'] = target_keys.index(selected_key)
                stress_fields = selected_results.get('stress_fields', {
                    'sigma_hydro': selected_results.get('sigma_hydro'),
                    'sigma_mag': selected_results.get('sigma_mag'),
                    'von_mises': selected_results.get('von_mises')
                })
                attention_weights = selected_results.get('attention_weights')
                target_params = selected_results.get('target_params', {})
                training_losses = results.get('training_losses')
            else:
                stress_fields = results.get('stress_fields', {})
                attention_weights = results.get('attention_weights')
                target_params = results.get('target_params', {})
                training_losses = results.get('training_losses')
 
            col1, col2, col3 = st.columns(3)
            with col1:
                component = st.selectbox("Stress Component", ['von_mises', 'sigma_hydro', 'sigma_mag'])
            with col2:
                cmap = st.selectbox("Colormap", ['viridis', 'plasma', 'coolwarm', 'RdBu', 'Spectral'])
            with col3:
                contours = st.checkbox("Contours", True)
 
            if component in stress_fields:
                data = stress_fields[component]
                fig = px.imshow(data, color_continuous_scale=cmap, origin='lower',
                                aspect='equal',
                                title=f"{component.replace('_', ' ').title()} – {target_params.get('orientation', '')}")
                fig.update_layout(xaxis_title='x (nm)', yaxis_title='y (nm)')
                if contours:
                    fig.add_trace(go.Contour(z=data, showscale=False, colorscale='black', line_width=0.5, ncontours=10))
                st.plotly_chart(fig, use_container_width=True)
 
            if attention_weights is not None:
                st.subheader("Attention Weights")
                sources = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
                fig_bar = px.bar(x=sources, y=attention_weights, labels={'x':'Source', 'y':'Weight'})
                fig_bar.update_layout(title="Attention Distribution")
                st.plotly_chart(fig_bar, use_container_width=True)
 
            if training_losses is not None:
                st.subheader("Training Loss")
                fig_loss = px.line(y=training_losses, labels={'index':'Epoch', 'value':'Loss'})
                st.plotly_chart(fig_loss, use_container_width=True)
 
            st.subheader("Statistics")
            stats = []
            for k, v in stress_fields.items():
                if isinstance(v, np.ndarray):
                    stats.append({
                        'Component': k.replace('_', ' ').title(),
                        'Max': f"{np.max(v):.3f}",
                        'Min': f"{np.min(v):.3f}",
                        'Mean': f"{np.mean(v):.3f}",
                        'Std': f"{np.std(v):.3f}"
                    })
            if stats:
                st.dataframe(pd.DataFrame(stats), use_container_width=True)
 
    with tab6:
        st.subheader("💾 Export Prediction Results")
        has_single = 'prediction_results' in st.session_state
        has_multi = bool(st.session_state.multi_target_predictions)
        if not has_single and not has_multi:
            st.warning("No results to export yet.")
        else:
            st.success("Results ready for download!")
            col1, col2 = st.columns(2)
            with col1:
                save_mode = st.radio("Export", ["Current Single", "All Multiple", "Both"],
                                    index=0 if has_single else 1)
            with col2:
                base_name = st.text_input("Base filename",
                                         value=f"prediction_{datetime.now():%Y%m%d_%H%M%S}")
 
            dl1, dl2, dl3, dl4 = st.columns(4)
 
            # PKL
            with dl1:
                st.markdown("**PKL**")
                if st.button("Prepare PKL", key="prep_pkl"):
                    with st.spinner("Preparing..."):
                        data = None
                        if save_mode in ["Current Single", "Both"] and has_single:
                            data = PredictionResultsManager.prepare_prediction_data_for_saving(
                                st.session_state.prediction_results,
                                st.session_state.source_simulations, 'single')
                        elif save_mode == "All Multiple" and has_multi:
                            data = {
                                'metadata': {'mode': 'multi', 'num_predictions': len(st.session_state.multi_target_predictions)},
                                'all_predictions': st.session_state.multi_target_predictions
                            }
                        if data:
                            buf = BytesIO()
                            pickle.dump(data, buf, protocol=pickle.HIGHEST_PROTOCOL)
                            st.session_state.download_pkl_data = buf.getvalue()
                            st.success("Ready!")
                if st.session_state.download_pkl_data:
                    st.download_button("Download PKL", data=st.session_state.download_pkl_data,
                                       file_name=f"{base_name}.pkl", mime="application/octet-stream",
                                       use_container_width=True)
 
            # PT
            with dl2:
                st.markdown("**PT**")
                if st.button("Prepare PT", key="prep_pt"):
                    with st.spinner("Preparing..."):
                        data = None
                        if save_mode in ["Current Single", "Both"] and has_single:
                            data = PredictionResultsManager.prepare_prediction_data_for_saving(
                                st.session_state.prediction_results,
                                st.session_state.source_simulations, 'single')
                        elif save_mode == "All Multiple" and has_multi:
                            data = {
                                'metadata': {'mode': 'multi', 'num_predictions': len(st.session_state.multi_target_predictions)},
                                'all_predictions': st.session_state.multi_target_predictions
                            }
                        if data:
                            buf = BytesIO()
                            torch.save(data, buf)
                            st.session_state.download_pt_data = buf.getvalue()
                            st.success("Ready!")
                if st.session_state.download_pt_data:
                    st.download_button("Download PT", data=st.session_state.download_pt_data,
                                       file_name=f"{base_name}.pt", mime="application/octet-stream",
                                       use_container_width=True)
 
            # ZIP
            with dl3:
                st.markdown("**ZIP Archive**")
                if st.button("Prepare ZIP", key="prep_zip"):
                    with st.spinner("Creating archive..."):
                        if save_mode in ["Current Single", "Both"] and has_single:
                            buf = PredictionResultsManager.create_single_prediction_archive(
                                st.session_state.prediction_results, st.session_state.source_simulations)
                            st.session_state.download_zip_data = buf.getvalue()
                            st.session_state.download_zip_filename = f"{base_name}_single.zip"
                        elif save_mode == "All Multiple" and has_multi:
                            buf = PredictionResultsManager.create_multi_prediction_archive(
                                st.session_state.multi_target_predictions, st.session_state.source_simulations)
                            st.session_state.download_zip_data = buf.getvalue()
                            st.session_state.download_zip_filename = f"{base_name}_multi.zip"
                        st.success("Ready!")
                if st.session_state.download_zip_data:
                    st.download_button("Download ZIP", data=st.session_state.download_zip_data,
                                       file_name=st.session_state.download_zip_filename,
                                       mime="application/zip", use_container_width=True)
 
            with dl4:
                st.markdown("**Clear**")
                if st.button("Clear Prepared", key="clear_all"):
                    for k in ['download_pkl_data','download_pt_data','download_zip_data','download_zip_filename']:
                        st.session_state[k] = None
                    st.success("Cleared!")
 
if __name__ == "__main__":
    create_attention_interface()
st.caption("🔬 Spatial-Attention Stress Interpolator • 2025")
