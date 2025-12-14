# part1_attention_interpolation.py
import numpy as np
import pandas as pd
import torch
import pickle
import json
import h5py
import msgpack
import dill
import joblib
import sqlite3
import tempfile
import os
import glob
from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_RESULTS_DIR = os.path.join(SCRIPT_DIR, "ml_results")
if not os.path.exists(ML_RESULTS_DIR):
    os.makedirs(ML_RESULTS_DIR, exist_ok=True)

# =============================================
# SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
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
        return torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)
    
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
        """Read simulation file from content"""
        
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
# STRESS ANALYSIS MANAGER
# =============================================
class StressAnalysisManager:
    """Manager for stress value analysis"""
    
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
        
        # Hydrostatic stress
        if 'sigma_hydro' in stress_fields:
            hydro_data = stress_fields['sigma_hydro']
            results['max_abs_hydrostatic'] = float(np.max(np.abs(hydro_data)))
            results['max_hydrostatic'] = float(np.max(hydro_data))
            results['min_hydrostatic'] = float(np.min(hydro_data))
            results['mean_abs_hydrostatic'] = float(np.mean(np.abs(hydro_data)))
        
        # Stress magnitude
        if 'sigma_mag' in stress_fields:
            mag_data = stress_fields['sigma_mag']
            results['max_stress_magnitude'] = float(np.max(mag_data))
            results['mean_stress_magnitude'] = float(np.mean(mag_data))
        
        # Von Mises stress
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
            results['von_mises_p95'] = float(np.percentile(vm_data, 95))
            results['von_mises_p99'] = float(np.percentile(vm_data, 99))
            results['von_mises_p99_9'] = float(np.percentile(vm_data, 99.9))
        
        return results
    
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
            if 'max_von_mises' in df.columns and 'max_abs_hydrostatic' in df.columns:
                df['stress_ratio_vm_hydro'] = df['max_von_mises'] / (df['max_abs_hydrostatic'] + 1e-10)
            return df
        else:
            return pd.DataFrame()

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
# RESULTS SAVER
# =============================================
class ResultsSaver:
    """Save attention interpolation results to ml_results directory"""
    
    def __init__(self, save_dir=ML_RESULTS_DIR):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
    def save_results(self, 
                    source_simulations: List[Dict],
                    predictions: Dict,
                    stress_summary_df: pd.DataFrame,
                    metadata: Dict[str, Any] = None,
                    filename_prefix: str = "attention_results"):
        """
        Save all interpolation results to ml_results directory
        
        Args:
            source_simulations: List of source simulation data
            predictions: Dictionary of predictions
            stress_summary_df: DataFrame with stress summary
            metadata: Additional metadata
            filename_prefix: Prefix for saved files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results dictionary
        results = {
            'source_simulations': source_simulations,
            'predictions': predictions,
            'stress_summary_df': stress_summary_df,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'version': '1.0'
        }
        
        # Save in multiple formats for compatibility
        self._save_pickle(results, filename_prefix, timestamp)
        self._save_hdf5(results, filename_prefix, timestamp)
        self._save_npz(results, filename_prefix, timestamp)
        self._save_parquet(stress_summary_df, filename_prefix, timestamp)
        
        # Also save CSV for easy viewing
        if not stress_summary_df.empty:
            csv_path = os.path.join(self.save_dir, f"{filename_prefix}_{timestamp}.csv")
            stress_summary_df.to_csv(csv_path, index=False)
            print(f"✅ Saved CSV to: {csv_path}")
    
    def _save_pickle(self, results, prefix, timestamp):
        """Save results as pickle file"""
        pickle_path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Saved pickle to: {pickle_path}")
    
    def _save_hdf5(self, results, prefix, timestamp):
        """Save results as HDF5 file"""
        hdf5_path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.h5")
        with h5py.File(hdf5_path, 'w') as f:
            # Save stress summary as dataset
            if 'stress_summary_df' in results and not results['stress_summary_df'].empty:
                df = results['stress_summary_df']
                for col in df.columns:
                    if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        f.create_dataset(f'stress_summary/{col}', data=df[col].values)
            
            # Save metadata
            if 'metadata' in results:
                metadata_group = f.create_group('metadata')
                for key, value in results['metadata'].items():
                    if isinstance(value, (str, int, float)):
                        metadata_group.attrs[key] = value
            
            # Save timestamp
            f.attrs['timestamp'] = timestamp
            f.attrs['version'] = results.get('version', '1.0')
        print(f"✅ Saved HDF5 to: {hdf5_path}")
    
    def _save_npz(self, results, prefix, timestamp):
        """Save results as NPZ file"""
        npz_path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.npz")
        np.savez_compressed(
            npz_path,
            source_simulations=results['source_simulations'],
            predictions=results['predictions'],
            stress_summary_columns=list(results['stress_summary_df'].columns) if not results['stress_summary_df'].empty else [],
            stress_summary_data=results['stress_summary_df'].values if not results['stress_summary_df'].empty else [],
            timestamp=timestamp
        )
        print(f"✅ Saved NPZ to: {npz_path}")
    
    def _save_parquet(self, df, prefix, timestamp):
        """Save DataFrame as Parquet file"""
        if not df.empty:
            parquet_path = os.path.join(self.save_dir, f"{prefix}_{timestamp}.parquet")
            df.to_parquet(parquet_path, index=False)
            print(f"✅ Saved Parquet to: {parquet_path}")

# =============================================
# MAIN ATTENTION INTERPOLATION FUNCTION
# =============================================
def run_attention_interpolation(source_files_dir: str = None,
                               source_files: List[str] = None,
                               target_params_config: Dict = None,
                               multi_target_config: Dict = None,
                               save_results: bool = True) -> Dict[str, Any]:
    """
    Run attention interpolation and save results
    
    Args:
        source_files_dir: Directory containing source simulation files
        source_files: List of specific source files to load
        target_params_config: Configuration for single target
        multi_target_config: Configuration for multiple targets
        save_results: Whether to save results to ml_results
    
    Returns:
        Dictionary containing all results
    """
    print("🚀 Starting Attention Interpolation...")
    
    # Initialize interpolator
    interpolator = SpatialLocalityAttentionInterpolator()
    stress_analyzer = StressAnalysisManager()
    multi_target_manager = MultiTargetPredictionManager()
    
    # Load source simulations
    source_simulations = []
    
    if source_files_dir:
        print(f"📂 Loading source files from: {source_files_dir}")
        # Load all files from directory
        for ext in ['*.pkl', '*.pt', '*.h5', '*.hdf5', '*.npz', '*.json']:
            pattern = os.path.join(source_files_dir, ext)
            files = glob.glob(pattern)
            for file_path in files:
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    sim_data = interpolator.read_simulation_file(content)
                    source_simulations.append(sim_data)
                    print(f"   ✅ Loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"   ❌ Error loading {file_path}: {str(e)}")
    
    if source_files:
        for file_path in source_files:
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                sim_data = interpolator.read_simulation_file(content)
                source_simulations.append(sim_data)
                print(f"   ✅ Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {str(e)}")
    
    if not source_simulations:
        raise ValueError("No source simulations loaded!")
    
    print(f"📊 Loaded {len(source_simulations)} source simulations")
    
    # Configure target parameters
    if multi_target_config:
        print("🎯 Configuring multiple targets...")
        base_params = multi_target_config.get('base_params', {})
        ranges_config = multi_target_config.get('ranges_config', {})
        target_params_list = multi_target_manager.create_parameter_grid(base_params, ranges_config)
        
        # Run batch predictions
        print(f"🚀 Running batch predictions for {len(target_params_list)} targets...")
        predictions = multi_target_manager.batch_predict(
            source_simulations, target_params_list, interpolator
        )
        print(f"✅ Generated {len(predictions)} predictions")
        
    elif target_params_config:
        print("🎯 Configuring single target...")
        target_params = target_params_config
        
        # Prepare source data for single target prediction
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
        
        predictions = {
            'single_target': {
                'sigma_hydro': weighted_stress[0],
                'sigma_mag': weighted_stress[1],
                'von_mises': weighted_stress[2],
                'predicted': True,
                'target_params': target_params,
                'attention_weights': weights
            }
        }
        print("✅ Generated single target prediction")
    else:
        predictions = {}
        print("⚠️ No target configuration provided, only source simulations loaded")
    
    # Create stress summary
    print("📊 Creating stress summary...")
    stress_summary_df = stress_analyzer.create_stress_summary_dataframe(
        source_simulations, predictions
    )
    print(f"✅ Stress summary created with {len(stress_summary_df)} entries")
    
    # Prepare metadata
    metadata = {
        'source_count': len(source_simulations),
        'prediction_count': len(predictions),
        'interpolator_config': {
            'num_heads': interpolator.num_heads,
            'sigma_spatial': interpolator.sigma_spatial,
            'sigma_param': interpolator.sigma_param,
            'use_gaussian': interpolator.use_gaussian
        },
        'run_timestamp': datetime.now().isoformat()
    }
    
    # Save results if requested
    if save_results:
        print("💾 Saving results to ml_results directory...")
        saver = ResultsSaver()
        saver.save_results(
            source_simulations=source_simulations,
            predictions=predictions,
            stress_summary_df=stress_summary_df,
            metadata=metadata,
            filename_prefix="attention_interpolation"
        )
    
    # Return all results
    results = {
        'source_simulations': source_simulations,
        'predictions': predictions,
        'stress_summary_df': stress_summary_df,
        'metadata': metadata,
        'interpolator': interpolator
    }
    
    print("🎉 Attention interpolation completed successfully!")
    return results

# =============================================
# EXAMPLE USAGE
# =============================================
if __name__ == "__main__":
    # Example configuration for single target
    target_config = {
        'defect_type': 'ISF',
        'shape': 'Square',
        'orientation': 'Horizontal {111} (0°)',
        'eps0': 1.414,
        'kappa': 0.7,
        'theta': 0.0
    }
    
    # Example configuration for multiple targets
    multi_target_config = {
        'base_params': {
            'defect_type': 'ISF',
            'shape': 'Square',
            'orientation': 'Horizontal {111} (0°)'
        },
        'ranges_config': {
            'eps0': {'min': 0.5, 'max': 2.0, 'steps': 5},
            'kappa': {'min': 0.2, 'max': 1.0, 'steps': 4},
            'theta': {'values': [0, np.pi/4, np.pi/2]}
        }
    }
    
    try:
        # Run with single target
        results = run_attention_interpolation(
            source_files_dir="numerical_solutions",  # Change to your directory
            target_params_config=target_config,
            save_results=True
        )
        
        # Or run with multiple targets
        # results = run_attention_interpolation(
        #     source_files_dir="numerical_solutions",
        #     multi_target_config=multi_target_config,
        #     save_results=True
        # )
        
        print(f"\n📋 Results Summary:")
        print(f"   Source simulations: {len(results['source_simulations'])}")
        print(f"   Predictions: {len(results['predictions'])}")
        print(f"   Stress summary entries: {len(results['stress_summary_df'])}")
        print(f"   Results saved to: {ML_RESULTS_DIR}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
