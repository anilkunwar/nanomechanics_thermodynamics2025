import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
from matplotlib.cm import get_cmap
import plotly.graph_objects as go
import plotly.express as px
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
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
import seaborn as sns
from scipy.ndimage import zoom
import re
import warnings
warnings.filterwarnings('ignore')
# =============================================
# GLOBAL STYLING CONFIGURATION
# =============================================
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'image.cmap': 'viridis'
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

COLORMAP_OPTIONS = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'afmhot', 'gist_heat',
                  'copper', 'summer', 'Wistia', 'spring', 'autumn', 'winter', 'bone', 'gray', 'pink',
                  'gist_gray', 'gist_yarg', 'binary', 'gist_earth', 'terrain', 'ocean', 'gist_stern', 'gnuplot',
                  'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
                  'gist_ncar', 'hsv'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                 'RdGy', 'RdYlGn', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                   'Paired', 'Accent', 'Dark2'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
                            'turbo'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'RdBu_r', 'Spectral',
                            'coolwarm', 'bwr', 'seismic', 'BrBG']
}

# =============================================
# PHYSICS PARAMETERS ENHANCEMENT
# =============================================
class PhysicsParameters:
    """Enhanced physics parameters with correct eigenstrain values"""
    EIGENSTRAIN_VALUES = {
        'Twin': 2.12,
        'ISF': 0.289,
        'ESF': 0.333,
        'No Defect': 0.0
    }
    THEORETICAL_BASIS = {
        'Twin': {
            'value': 2.12,
            'formula': r'$\epsilon_0 = \frac{\sqrt{6}}{2} \times 0.866$',
            'description': 'Twinning transformation strain from crystallographic theory',
            'reference': 'Hirth & Lothe (1982) Theory of Dislocations'
        },
        'ISF': {
            'value': 0.289,
            'formula': r'$\epsilon_0 = \frac{1}{2\sqrt{3}} \times 0.707$',
            'description': 'Intrinsic Stacking Fault shear strain',
            'reference': 'Christian & Mahajan (1995) Prog. Mater. Sci.'
        },
        'ESF': {
            'value': 0.333,
            'formula': r'$\epsilon_0 = \frac{1}{3} \times 1.0$',
            'description': 'Extrinsic Stacking Fault displacement',
            'reference': 'Sutton & Balluffi (1995) Interfaces in Crystalline Materials'
        }
    }
    
    @staticmethod
    def get_eigenstrain(defect_type: str) -> float:
        return PhysicsParameters.EIGENSTRAIN_VALUES.get(defect_type, 0.0)
        
    @staticmethod
    def get_theoretical_info(defect_type: str) -> Dict:
        return PhysicsParameters.THEORETICAL_BASIS.get(defect_type, {})

# =============================================
# DIFFUSION PHYSICS PARAMETERS
# =============================================
class DiffusionPhysics:
    """Enhanced diffusion physics with multiple theoretical models"""
    k_B_eV = 8.617333262145e-5
    k_B_J = 1.380649e-23
    
    MATERIAL_PROPERTIES = {
        'Silver': {
            'atomic_volume': 1.56e-29, 'atomic_mass': 107.8682, 'density': 10.49e6,
            'melting_point': 1234.93, 'bulk_modulus': 100e9, 'shear_modulus': 30e9,
            'activation_energy': 1.1, 'prefactor': 7.2e-7, 'atomic_radius': 1.44e-10,
            'vacancy_formation_energy': 1.1, 'vacancy_migration_energy': 0.66,
        },
        'Copper': {
            'atomic_volume': 1.18e-29, 'atomic_mass': 63.546, 'density': 8.96e6,
            'melting_point': 1357.77, 'bulk_modulus': 140e9, 'shear_modulus': 48e9,
            'activation_energy': 1.0, 'prefactor': 3.1e-7, 'atomic_radius': 1.28e-10,
            'vacancy_formation_energy': 1.0, 'vacancy_migration_energy': 0.70,
        },
        'Aluminum': {
            'atomic_volume': 1.66e-29, 'atomic_mass': 26.9815, 'density': 2.70e6,
            'melting_point': 933.47, 'bulk_modulus': 76e9, 'shear_modulus': 26e9,
            'activation_energy': 0.65, 'prefactor': 1.7e-6, 'atomic_radius': 1.43e-10,
            'vacancy_formation_energy': 0.65, 'vacancy_migration_energy': 0.55,
        },
        'Nickel': {
            'atomic_volume': 1.09e-29, 'atomic_mass': 58.6934, 'density': 8.908e6,
            'melting_point': 1728.0, 'bulk_modulus': 180e9, 'shear_modulus': 76e9,
            'activation_energy': 1.4, 'prefactor': 1.9e-7, 'atomic_radius': 1.24e-10,
            'vacancy_formation_energy': 1.4, 'vacancy_migration_energy': 0.9,
        },
        'Iron': {
            'atomic_volume': 1.18e-29, 'atomic_mass': 55.845, 'density': 7.874e6,
            'melting_point': 1811.0, 'bulk_modulus': 170e9, 'shear_modulus': 82e9,
            'activation_energy': 2.0, 'prefactor': 2.0e-8, 'atomic_radius': 1.24e-10,
            'vacancy_formation_energy': 2.0, 'vacancy_migration_energy': 1.2,
        }
    }
    
    @staticmethod
    def get_material_properties(material='Silver'):
        return DiffusionPhysics.MATERIAL_PROPERTIES.get(material, DiffusionPhysics.MATERIAL_PROPERTIES['Silver'])
        
    @staticmethod
    def compute_diffusion_enhancement(sigma_hydro_GPa, T_K=650, material='Silver',
                                      model='physics_corrected', stress_unit='GPa'):
        props = DiffusionPhysics.get_material_properties(material)
        Omega = props['atomic_volume']
        
        if stress_unit == 'GPa':
            sigma_hydro_Pa = sigma_hydro_GPa * 1e9
        else:
            sigma_hydro_Pa = sigma_hydro_GPa
            
        if model == 'physics_corrected':
            exponent = Omega * sigma_hydro_Pa / (DiffusionPhysics.k_B_J * T_K)
            D_ratio = np.exp(exponent)
        elif model == 'temperature_reduction':
            Q_J = props['activation_energy'] * 1.602e-19
            with np.errstate(divide='ignore', invalid='ignore'):
                T_eff = T_K / (1 - Omega * sigma_hydro_Pa / Q_J)
                T_eff = np.where(np.isfinite(T_eff), T_eff, T_K)
            D_ratio = np.exp(Q_J / DiffusionPhysics.k_B_J * (1/T_K - 1/T_eff))
        elif model == 'activation_energy':
            Q_J = props['activation_energy'] * 1.602e-19
            Q_eff = Q_J - Omega * sigma_hydro_Pa
            D_bulk = np.exp(-Q_J / (DiffusionPhysics.k_B_J * T_K))
            D_stressed = np.exp(-Q_eff / (DiffusionPhysics.k_B_J * T_K))
            D_ratio = D_stressed / D_bulk
        elif model == 'vacancy_concentration':
            exponent = Omega * sigma_hydro_Pa / (DiffusionPhysics.k_B_J * T_K)
            D_ratio = np.exp(exponent)
        else:
            raise ValueError(f"Unknown model: {model}")
            
        return D_ratio
    
    @staticmethod
    def compute_effective_diffusion_coefficient(sigma_hydro_GPa, T_K=650, material='Silver'):
        props = DiffusionPhysics.get_material_properties(material)
        D0 = props['prefactor'] * np.exp(-props['activation_energy'] / (DiffusionPhysics.k_B_eV * T_K))
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(sigma_hydro_GPa, T_K, material, 'physics_corrected')
        return D0 * D_ratio
    
    @staticmethod
    def compute_mass_flux_gradient(D_ratio_field, unstressed_D, concentration_gradient):
        D_eff = unstressed_D * D_ratio_field
        mass_flux = -D_eff * concentration_gradient
        return mass_flux
    
    @staticmethod
    def compute_dislocation_sink_strength(stress_field, material='Silver'):
        props = DiffusionPhysics.get_material_properties(material)
        shear_modulus = props['shear_modulus']
        sigma_vm = np.sqrt(np.sum(stress_field**2, axis=0))
        sink_enhancement = 1 + 0.1 * sigma_vm / shear_modulus
        return sink_enhancement
    
    @staticmethod
    def compute_activation_volume(sigma_hydro_GPa, D_ratio):
        props = DiffusionPhysics.get_material_properties('Silver')
        Omega = props['atomic_volume']
        if isinstance(D_ratio, np.ndarray) and D_ratio.size > 1:
            lnD = np.log(D_ratio)
            V_act = np.gradient(lnD) / np.gradient(sigma_hydro_GPa) * 1.38e-23 * 650 / 1e9
        else:
            V_act = Omega
        return V_act
    
    @staticmethod
    def compute_diffusion_length(D_ratio, time, D0):
        D_eff = D0 * D_ratio
        diffusion_length = np.sqrt(4 * D_eff * time)
        return diffusion_length

# =============================================
# ENHANCED SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    """Enhanced solution loader with physics-aware processing"""
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        
    def _ensure_directory(self):
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
            
    def scan_solutions(self) -> List[Dict[str, Any]]:
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth']:
            import glob
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        all_files.sort(key=os.path.getmtime, reverse=True)
        
        file_info = []
        for file_path in all_files:
            try:
                info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'format': 'pkl' if file_path.endswith(('.pkl', '.pickle')) else 'pt'
                }
                file_info.append(info)
            except:
                continue
                
        return file_info
    
    def read_simulation_file(self, file_path, format_type='auto'):
        try:
            with open(file_path, 'rb') as f:
                if format_type == 'pt' or file_path.endswith(('.pt', '.pth')):
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                else:
                    data = pickle.load(f)
            
            standardized = self._standardize_data(data, file_path)
            return standardized
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data, file_path):
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False
            }
        }
        
        try:
            if isinstance(data, dict):
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                    
                if 'history' in data:
                    history = data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        history_list = []
                        for key in sorted(history.keys()):
                            if isinstance(history[key], dict):
                                history_list.append(history[key])
                        standardized['history'] = history_list
                
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                    
            self._convert_tensors(standardized)
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
            
        return standardized
    
    def _convert_tensors(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.cpu().numpy()
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    data[i] = item.cpu().numpy()
                elif isinstance(item, (dict, list)):
                    self._convert_tensors(item)
    
    def load_all_solutions(self, use_cache=True, max_files=None):
        solutions = []
        file_info = self.scan_solutions()
        
        if max_files:
            file_info = file_info[:max_files]
            
        if not file_info:
            return solutions
            
        for file_info_item in file_info:
            cache_key = file_info_item['filename']
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
                
            solution = self.read_simulation_file(file_info_item['path'])
            if solution:
                self.cache[cache_key] = solution
                solutions.append(solution)
                
        return solutions

# =============================================
# POSITIONAL ENCODING FOR TRANSFORMER
# =============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)

# =============================================
# ENHANCED TRANSFORMER SPATIAL INTERPOLATOR WITH BRACKETING THEORY
# =============================================
class TransformerSpatialInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                spatial_sigma=10.0, temperature=1.0, locality_weight_factor=0.5):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(15, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def set_spatial_parameters(self, spatial_sigma=None, locality_weight_factor=None):
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
        if locality_weight_factor is not None:
            self.locality_weight_factor = locality_weight_factor
            
    def debug_feature_dimensions(self, params_list, target_angle_deg):
        encoded = self.encode_parameters(params_list, target_angle_deg)
        return encoded.shape
        
    def compute_angular_bracketing_kernel(self, source_params, target_params):
        spatial_weights = []
        defect_mask = []
        angular_distances = []
        
        target_theta = target_params.get('theta', 0.0)
        target_theta_deg = np.degrees(target_theta) % 360
        target_defect = target_params.get('defect_type', 'Twin')
        
        for src in source_params:
            src_theta = src.get('theta', 0.0)
            src_theta_deg = np.degrees(src_theta) % 360
            
            raw_diff = abs(src_theta_deg - target_theta_deg)
            angular_dist = min(raw_diff, 360 - raw_diff)
            angular_distances.append(angular_dist)
            
            if src.get('defect_type') == target_defect:
                defect_mask.append(1.0)
            else:
                defect_mask.append(1e-6)
                
            weight = np.exp(-0.5 * (angular_dist / self.spatial_sigma) ** 2)
            spatial_weights.append(weight)
            
        return np.array(spatial_weights), np.array(defect_mask), np.array(angular_distances)
    
    def visualize_angular_kernel(self, target_angle_deg=54.7, figsize=(12, 8)):
        angles = np.linspace(0, 180, 361)
        weights = []
        
        dummy_sources = [{'theta': np.radians(a), 'defect_type': 'Twin'} for a in angles]
        dummy_target = {'theta': np.radians(target_angle_deg), 'defect_type': 'Twin'}
        
        spatial_weights, _, _ = self.compute_angular_bracketing_kernel(dummy_sources, dummy_target)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(angles, spatial_weights, 'b-', linewidth=3, label='Bracketing Kernel Weight')
        ax.axvline(x=target_angle_deg, color='r', linestyle='--', linewidth=2, label=f'Target: {target_angle_deg}°')
        ax.axvline(x=54.7, color='g', linestyle='-.', linewidth=2, label='Habit Plane: 54.7°')
        
        ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Spatial Kernel Weight', fontsize=14, fontweight='bold')
        ax.set_title(f'Angular Bracketing Regularization Kernel\nSigma: {self.spatial_sigma}°',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 1.1])
        
        ax.fill_between([target_angle_deg - self.spatial_sigma, target_angle_deg + self.spatial_sigma],
                       0, 1, color='blue', alpha=0.1, label=f'±1$\sigma$ Region')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def encode_parameters(self, params_list, target_angle_deg):
        encoded = []
        for params in params_list:
            features = []
            
            # Physics parameters
            features.append(params.get('eps0', 0.707) / 3.0)  # normalize
            features.append(params.get('kappa', 0.6) / 2.0)   # normalize
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)  # normalize to [0,1]
            
            # One-hot encoding for defect type
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
                
            # One-hot encoding for shape
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
                
            # Angular features
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)
            features.append(np.exp(-angle_diff / 45.0))  # Angular proximity
            
            # Trigonometric features for periodicity
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))
            
            # Distance to habit plane (54.7°)
            habit_distance = abs(theta_deg - 54.7)
            habit_distance = min(habit_distance, 360 - habit_distance)
            features.append(np.exp(-habit_distance / 15.0))
            
            # Pad to 15 features if needed
            while len(features) < 15:
                features.append(0.0)
            features = features[:15]
            
            encoded.append(features)
            
        return torch.FloatTensor(encoded)
        
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        if not sources:
            return None
            
        try:
            source_params = []
            source_fields = []
            source_indices = []
            
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    continue
                    
                source_params.append(src['params'])
                source_indices.append(i)
                
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        stress_fields = last_frame['stresses']
                        vm = stress_fields.get('von_mises', self.compute_von_mises(stress_fields))
                        hydro = stress_fields.get('sigma_hydro', self.compute_hydrostatic(stress_fields))
                        mag = stress_fields.get('sigma_mag', np.sqrt(vm**2 + hydro**2))
                        
                        source_fields.append({
                            'von_mises': vm, 'sigma_hydro': hydro, 'sigma_mag': mag,
                            'source_index': i, 'source_params': src['params']
                        })
                    else:
                        continue
                else:
                    continue
                    
            if not source_params or not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
                
            # Ensure all fields have the same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                target_shape = shapes[0]
                resized_fields = []
                for fields in source_fields:
                    resized = {}
                    for key, field in fields.items():
                        if key in ['von_mises', 'sigma_hydro', 'sigma_mag'] and field.shape != target_shape:
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            resized[key] = zoom(field, factors, order=1)
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                source_fields = resized_fields
                
            # Encode source and target parameters
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Ensure feature dimensions match
            if source_features.shape[1] != 15 or target_features.shape[1] != 15:
                st.warning(f"Feature dimension mismatch")
                if source_features.shape[1] < 15:
                    padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                    source_features = torch.cat([source_features, padding], dim=1)
                if target_features.shape[1] < 15:
                    padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                    target_features = torch.cat([target_features, padding], dim=1)
            
            # Compute spatial kernel and defect mask
            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )
            
            # Transformer processing
            batch_size = 1
            seq_len = len(source_features) + 1
            
            # Combine target and source features
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            
            # Project to transformer dimension
            proj_features = self.input_proj(all_features)
            
            # Add positional encoding
            proj_features = self.pos_encoder(proj_features)
            
            # Pass through transformer
            transformer_output = self.transformer(proj_features)
            
            # Extract target and source representations
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
            
            # Compute attention scores
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            
            # Apply spatial kernel and defect mask as bias
            spatial_kernel_tensor = torch.FloatTensor(spatial_kernel).unsqueeze(0)
            defect_mask_tensor = torch.FloatTensor(defect_mask).unsqueeze(0)
            
            biased_scores = attn_scores * spatial_kernel_tensor * defect_mask_tensor
            final_attention_weights = torch.softmax(biased_scores, dim=-1).squeeze().detach().cpu().numpy()
            entropy_final = self._calculate_entropy(final_attention_weights)
            
            # Interpolate fields using attention weights
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += final_attention_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
                
            # Compute diffusion enhancement
            if 'sigma_hydro' in interpolated_fields:
                sigma_hydro = interpolated_fields['sigma_hydro']
                D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro, T_K=650, material='Silver', model='physics_corrected'
                )
                
                props = DiffusionPhysics.get_material_properties('Silver')
                D0 = props['prefactor'] * np.exp(-props['activation_energy'] / 
                                               (DiffusionPhysics.k_B_eV * 650))
                D_eff = D0 * D_ratio
                
                vacancy_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro, T_K=650, material='Silver', model='vacancy_concentration'
                )
                
                interpolated_fields['diffusion_ratio'] = D_ratio
                interpolated_fields['diffusion_effective'] = D_eff
                interpolated_fields['vacancy_ratio'] = vacancy_ratio
                
                # Compute gradient of diffusion field
                grad_x, grad_y = np.gradient(D_ratio)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                interpolated_fields['diffusion_gradient'] = grad_magnitude
                
            max_vm = np.max(interpolated_fields['von_mises'])
            max_hydro = np.max(np.abs(interpolated_fields['sigma_hydro']))
            
            source_theta_degrees = [np.degrees(src.get('theta', 0.0)) % 360 for src in source_params]
            
            diffusion_statistics = {}
            if 'diffusion_ratio' in interpolated_fields:
                D_ratio = interpolated_fields['diffusion_ratio']
                diffusion_statistics = {
                    'max_enhancement': float(np.max(D_ratio)),
                    'min_enhancement': float(np.min(D_ratio)),
                    'mean_enhancement': float(np.mean(D_ratio)),
                    'std_enhancement': float(np.std(D_ratio)),
                    'enhanced_area_fraction': float(np.sum(D_ratio > 1.0) / D_ratio.size),
                    'suppressed_area_fraction': float(np.sum(D_ratio < 1.0) / D_ratio.size),
                    'max_tensile_enhancement': float(np.max(D_ratio[sigma_hydro > 0]) if np.any(sigma_hydro > 0) else 0),
                    'max_compressive_suppression': float(np.min(D_ratio[sigma_hydro < 0]) if np.any(sigma_hydro < 0) else 0)
                }
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist(),
                    'entropy': entropy_final
                },
                'statistics': {
                    'von_mises': {
                        'max': float(max_vm), 'mean': float(np.mean(interpolated_fields['von_mises'])),
                        'std': float(np.std(interpolated_fields['von_mises'])),
                        'min': float(np.min(interpolated_fields['von_mises']))
                    },
                    'sigma_hydro': {
                        'max_tension': float(np.max(interpolated_fields['sigma_hydro'])),
                        'max_compression': float(np.min(interpolated_fields['sigma_hydro'])),
                        'mean': float(np.mean(interpolated_fields['sigma_hydro'])),
                        'std': float(np.std(interpolated_fields['sigma_hydro']))
                    },
                    'sigma_mag': {
                        'max': float(np.max(interpolated_fields['sigma_mag'])),
                        'mean': float(np.mean(interpolated_fields['sigma_mag'])),
                        'std': float(np.std(interpolated_fields['sigma_mag'])),
                        'min': float(np.min(interpolated_fields['sigma_mag']))
                    },
                    'diffusion': diffusion_statistics
                },
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_theta_degrees': source_theta_degrees,
                'source_distances': angular_distances,
                'source_indices': source_indices,
                'source_fields': source_fields
            }
            
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            return None
            
    def compute_von_mises(self, stress_fields):
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz', 'tau_xy']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            txy = stress_fields['tau_xy']
            tyz = stress_fields.get('tau_yz', np.zeros_like(sxx))
            tzx = stress_fields.get('tau_zx', np.zeros_like(sxx))
            
            von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 +
                                     6*(txy**2 + tyz**2 + tzx**2)))
            return von_mises
        return np.zeros((100, 100))
        
    def compute_hydrostatic(self, stress_fields):
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            return (stress_fields['sigma_xx'] + stress_fields['sigma_yy'] + stress_fields.get('sigma_zz', np.zeros_like(stress_fields['sigma_xx']))) / 3
        return np.zeros((100, 100))
        
    def _calculate_entropy(self, weights):
        weights = np.array(weights)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return -np.sum(weights * np.log(weights + 1e-10))

# =============================================
# ENHANCED HEAT MAP VISUALIZER WITH CUSTOMIZATION
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with diffusion visualization and custom labels"""
    def __init__(self):
        # --- EXTENDED COLORMAP LIST ---
        self.colormaps = {
            'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'Sequential': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                          'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                          'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                          'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG'],
            'Diverging': ['PuOr', 'RdYlBu', 'RdGy', 'RdYlGn', 'Spectral', 'coolwarm',
                         'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv'],
            'Cyclic': ['hsv', 'twilight', 'twilight_shifted'],
            'Qualitative': ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                           'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'],
            'Miscellaneous/Rainbow': ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                                     'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                                     'cubehelix', 'brg', 'gist_rainbow', 'rainbow',
                                     'jet', 'nipy_spectral', 'gist_ncar', 'turbo'],
            # CORRECTED KEY FOR ERROR FIX
            'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                                    'RdBu', 'RdBu_r', 'Spectral', 'coolwarm', 'bwr',
                                    'seismic', 'BrBG']
        }
        self.diffusion_physics = DiffusionPhysics()
        
    # --- HELPER FOR CUSTOM LABELS ---
    def _apply_label(self, ax, default_text, override_text, func_name, **kwargs):
        """Applies text to axis or removes it if override is empty string."""
        if override_text is not None:
            if override_text.strip() == "":
                # Remove label
                getattr(ax, func_name)(None)
            else:
                getattr(ax, func_name)(override_text, **kwargs)
        else:
            getattr(ax, func_name)(default_text, **kwargs)
    
    def create_diffusion_heatmap(self, sigma_hydro_field, title="Diffusion Enhancement Map",
                               T_K=650, material='Silver', cmap_name='RdBu_r',
                               figsize=(12, 10), dpi=300, log_scale=True,
                               show_stats=True, target_angle=None, defect_type=None,
                               show_colorbar=True, aspect_ratio='equal',
                               model='physics_corrected', label_config=None):
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('RdBu_r')
            
        if log_scale:
            log_data = np.log10(np.clip(D_ratio, 0.1, 10))
            vmin, vmax = -1, 1
            im = ax.imshow(log_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', origin='lower')
                          
            show_cb = show_colorbar
            if label_config: show_cb = not label_config.get('hide_colorbar', False)
            if show_cb:
                cbar_label_text = label_config['colorbar_label'] if label_config and 'colorbar_label' in label_config else r"$log_{10}(D/D_0)$"
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if cbar_label_text != "":
                    cbar.set_label(cbar_label_text, fontsize=16, fontweight='bold')
                ticks = np.array([0.1, 0.5, 1, 2, 5, 10])
                log_ticks = np.log10(ticks)
                cbar.set_ticks(log_ticks)
                cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
                cbar.ax.tick_params(labelsize=14)
        else:
            vmin, vmax = 0.1, 10
            im = ax.imshow(D_ratio, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', origin='lower',
                          norm=LogNorm(vmin=vmin, vmax=vmax) if log_scale else None)
                          
            show_cb = show_colorbar
            if label_config: show_cb = not label_config.get('hide_colorbar', False)
            if show_cb:
                cbar_label_text = label_config['colorbar_label'] if label_config and 'colorbar_label' in label_config else r"$D/D_0$"
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if cbar_label_text != "":
                    cbar.set_label(cbar_label_text, fontsize=16, fontweight='bold')
                cbar.ax.tick_params(labelsize=14)
        
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, {defect_type}, T = {T_K} K"
            
        if label_config and 'title_suffix' in label_config and label_config['title_suffix']:
            title_str += f" {label_config['title_suffix']}"
            
        if label_config and label_config.get('hide_title', False):
            ax.set_title(None)
        else:
            ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
            
        xlabel_text = label_config['xlabel'] if label_config and 'xlabel' in label_config else "X Position (nm)"
        ylabel_text = label_config['ylabel'] if label_config and 'ylabel' in label_config else "Y Position (nm)"
        self._apply_label(ax, "X Position (nm)", xlabel_text, 'set_xlabel', fontsize=16, fontweight='bold')
        self._apply_label(ax, "Y Position (nm)", ylabel_text, 'set_ylabel', fontsize=16, fontweight='bold')
        
        show_grid = not label_config.get('hide_grid', False) if label_config else True
        if show_grid: ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        else: ax.grid(False)
        
        if show_stats:
            enhancement_regions = D_ratio > 1.0
            suppression_regions = D_ratio < 1.0
            stats_text = (f"Max Enhancement: {np.max(D_ratio):.2f}x\nMin (Suppression): {np.min(D_ratio):.2f}x\n"
                         f"Mean: {np.mean(D_ratio):.2f}x\nEnhanced Area: {np.sum(enhancement_regions)/D_ratio.size*100:.1f}%\n"
                         f"Suppressed Area: {np.sum(suppression_regions)/D_ratio.size*100:.1f}%")
                         
            show_stats_box = not label_config.get('hide_stats_box', False) if label_config else True
            if show_stats_box:
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
            
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig, D_ratio
        
    def create_publication_quality_diffusion_plot(self, sigma_hydro_field, title="Diffusion Enhancement",
                                      T_K=650, material='Silver', cmap_name='RdBu_r', figsize=(10, 8),
                                      target_angle=None, defect_type=None, log_scale=True,
                                      model='physics_corrected', label_config=None):
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        if log_scale:
            plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
            vmin, vmax = -1, 1
        else:
            plot_data = D_ratio
            vmin, vmax = 0.1, 10
            
        cmap = plt.get_cmap(cmap_name)
        im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect='equal', interpolation='bicubic', origin='lower')
        
        if not (label_config and label_config.get('hide_colorbar', False)):
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if log_scale:
                lbl = label_config['colorbar_label'] if label_config and 'colorbar_label' in label_config else r"$log_{10}(D/D_0)$"
            else:
                lbl = label_config['colorbar_label'] if label_config and 'colorbar_label' in label_config else r"$D/D_0$"
            if lbl != "": cbar.set_label(lbl, fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
        
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str += f" (θ = {target_angle:.1f}°, {defect_type})"
            
        if label_config and 'title_suffix' in label_config and label_config['title_suffix']:
            title_str += f" {label_config['title_suffix']}"
            
        if not (label_config and label_config.get('hide_title', False)):
            ax.set_title(title_str, fontsize=18, fontweight='bold', pad=15)
            
        x_lbl = label_config['xlabel'] if label_config and 'xlabel' in label_config else "X Position (nm)"
        y_lbl = label_config['ylabel'] if label_config and 'ylabel' in label_config else "Y Position (nm)"
        self._apply_label(ax, "X Position (nm)", x_lbl, 'set_xlabel', fontsize=14, fontweight='bold')
        self._apply_label(ax, "Y Position (nm)", y_lbl, 'set_ylabel', fontsize=14, fontweight='bold')
        
        if not (label_config and label_config.get('hide_grid', False)):
            for spine in ax.spines.values():
                spine.set_linewidth(0.5); spine.set_color('gray')
                
        plt.tight_layout()
        return fig, D_ratio
    
    def create_interactive_diffusion_heatmap(self, sigma_hydro_field, title="Diffusion Enhancement Map",
                                  T_K=650, material='Silver', cmap_name='RdBu',
                                  width=800, height=700, log_scale=True, target_angle=None, defect_type=None,
                                  model='physics_corrected'):
        """Create interactive diffusion heatmap with Plotly"""
        try:
            # Compute diffusion enhancement
            D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                sigma_hydro_field, T_K, material, model
            )
            
            # Apply log transform if requested
            if log_scale:
                plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
                z_label = "log₁₀(D/D₀)"
                text_format = "log₁₀(D/D₀) = {value:.3f}"
            else:
                plot_data = D_ratio
                z_label = "D/D₀"
                text_format = "D/D₀ = {value:.3f}"
            
            # Create hover text with enhanced information
            hover_text = []
            for i in range(plot_data.shape[0]):
                row_text = []
                for j in range(plot_data.shape[1]):
                    base_text = f"Position: ({i}, {j})<br>"
                    base_text += f"σₕ: {sigma_hydro_field[i, j]:.4f} GPa<br>"
                    if log_scale:
                        value = D_ratio[i, j]
                        base_text += f"D/D₀: {value:.4f}<br>"
                        base_text += f"log₁₀(D/D₀): {plot_data[i, j]:.3f}"
                    else:
                        base_text += f"D/D₀: {plot_data[i, j]:.4f}"
                    row_text.append(base_text)
                hover_text.append(row_text)
            
            # Validate and map colormap name to Plotly format
            plotly_colormap_mapping = {
                'viridis': 'Viridis',
                'plasma': 'Plasma',
                'inferno': 'Inferno',
                'magma': 'Magma',
                'cividis': 'Cividis',
                'turbo': 'Turbo',
                'hot': 'Hot',
                'afmhot': 'Hot',
                'gist_heat': 'Hot',
                'copper': 'Cividis',
                'summer': 'Viridis',
                'Wistia': 'YlOrRd',
                'spring': 'Viridis',
                'autumn': 'OrRd',
                'winter': 'Blues',
                'bone': 'Greys',
                'gray': 'Greys',
                'pink': 'Viridis',
                'gist_gray': 'Greys',
                'gist_yarg': 'Greys',
                'binary': 'Greys',
                'gist_earth': 'Earth',
                'terrain': 'Earth',
                'ocean': 'Deep',
                'gist_stern': 'Viridis',
                'gnuplot': 'Viridis',
                'gnuplot2': 'Viridis',
                'CMRmap': 'Viridis',
                'cubehelix': 'Viridis',
                'brg': 'Viridis',
                'gist_rainbow': 'Rainbow',
                'rainbow': 'Rainbow',
                'jet': 'Jet',
                'nipy_spectral': 'Rainbow',
                'gist_ncar': 'Rainbow',
                'hsv': 'Rainbow',
                'RdBu': 'RdBu',
                'RdYlBu': 'RdYlBu',
                'Spectral': 'Spectral',
                'coolwarm': 'RdBu',
                'bwr': 'RdBu',
                'seismic': 'RdBu',
                'BrBG': 'BrBG',
                'PiYG': 'PiYG',
                'PRGn': 'PRGn',
                'PuOr': 'PuOr',
                'RdGy': 'RdGy',
                'RdYlGn': 'RdYlGn',
                'Spectral_r': 'Spectral',
                'coolwarm_r': 'RdBu_r',
                'bwr_r': 'RdBu_r',
                'seismic_r': 'RdBu_r',
                'tab10': 'Plotly3',
                'tab20': 'Plotly3',
                'Set1': 'Set1',
                'Set2': 'Set2',
                'Set3': 'Set3',
                'tab20b': 'Plotly3',
                'tab20c': 'Plotly3',
                'Pastel1': 'Pastel1',
                'Pastel2': 'Pastel2',
                'Paired': 'Paired',
                'Accent': 'Accent',
                'Dark2': 'Dark2',
                'twilight': 'Viridis',
                'twilight_shifted': 'Viridis',
            }
            
            # Handle reverse colormaps
            is_reverse = cmap_name.endswith('_r')
            base_cmap = cmap_name[:-2] if is_reverse else cmap_name
            
            # Get the mapped colormap
            if base_cmap in plotly_colormap_mapping:
                plotly_cmap = plotly_colormap_mapping[base_cmap]
                if is_reverse:
                    plotly_cmap = f"{plotly_cmap}_r"
            else:
                plotly_cmap = 'RdBu'  # Standard default for diffusion
            
            # Create heatmap trace
            heatmap_trace = go.Heatmap(
                z=plot_data,
                colorscale=plotly_cmap,
                zmin=np.nanmin(plot_data),
                zmax=np.nanmax(plot_data),
                hoverinfo='text',
                text=hover_text,
                colorbar=dict(
                    title=dict(
                        text=z_label,
                        font=dict(size=16, color='black'),
                    ),
                    tickfont=dict(size=14),
                    thickness=20,
                    len=0.8
                )
            )
            
            # Create figure
            fig = go.Figure(data=[heatmap_trace])
            
            # Enhanced title
            title_str = title
            if target_angle is not None and defect_type is not None:
                title_str = f"{title}<br>θ = {target_angle:.1f}°, Defect: {defect_type}, T = {T_K} K"
            
            # Update layout with publication styling
            fig.update_layout(
                title=dict(
                    text=title_str,
                    font=dict(size=24, color='darkblue'),
                    x=0.5,
                    y=0.95
                ),
                width=width,
                height=height,
                xaxis=dict(
                    title=dict(text="X Position", font=dict(size=18, color="black")),
                    tickfont=dict(size=14),
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    title=dict(text="Y Position", font=dict(size=18, color="black")),
                    tickfont=dict(size=14),
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    scaleanchor="x",
                    scaleratio=1
                ),
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=80, r=80, t=100, b=80)
            )
            
            # Ensure aspect ratio is 1:1 for square fields
            fig.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )
            
            return fig, D_ratio
        except Exception as e:
            st.error(f"Error creating interactive diffusion heatmap: {e}")
            # Return a simple figure as fallback
            fig = go.Figure()
            fig.add_annotation(text="Error creating diffusion heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, np.ones((10, 10))
    
    def create_diffusion_3d_surface_plot(self, sigma_hydro_field, title="3D Diffusion Surface",
                             T_K=650, material='Silver', cmap_name='RdBu_r',
                             figsize=(14, 10), dpi=300, log_scale=True,
                             target_angle=None, defect_type=None,
                             model='physics_corrected'):
        """Create 3D surface plot of diffusion enhancement"""
        # Compute diffusion enhancement
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        
        # Apply log transform if requested
        if log_scale:
            plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
            z_label = r"$log_{10}(D/D_0)$"
        else:
            plot_data = D_ratio
            z_label = r"$D/D_0$"
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(plot_data.shape[1])
        y = np.arange(plot_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('RdBu_r')  # Good for showing enhancement/suppression
            
        # Create surface plot
        if log_scale:
            vmin, vmax = -1, 1
        else:
            vmin, vmax = np.min(plot_data), np.max(plot_data)
            
        surf = ax.plot_surface(X, Y, plot_data, cmap=cmap, vmin=vmin, vmax=vmax,
                             linewidth=0, antialiased=True, alpha=0.8, rstride=1, cstride=1)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
        cbar.set_label(z_label, fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        
        # Customized title
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, {defect_type}, T = {T_K} K"
        
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel(z_label, fontsize=16, fontweight='bold', labelpad=10)
        
        # Set tick parameters
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)
        
        # Adjust view angle for better visualization
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        return fig, D_ratio
    
    def create_interactive_diffusion_3d(self, sigma_hydro_field, title="Interactive 3D Diffusion",
                                     T_K=650, material='Silver', cmap_name='RdBu',
                                     width=900, height=700, log_scale=True,
                                     target_angle=None, defect_type=None,
                                     model='physics_corrected'):
        """Create interactive 3D diffusion surface plot with Plotly"""
        try:
            # Compute diffusion enhancement
            D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                sigma_hydro_field, T_K, material, model
            )
            
            # Apply log transform if requested
            if log_scale:
                plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
                z_label = "log₁₀(D/D₀)"
            else:
                plot_data = D_ratio
                z_label = "D/D₀"
            
            # Create meshgrid
            x = np.arange(plot_data.shape[1])
            y = np.arange(plot_data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Create hover text
            hover_text = []
            for i in range(plot_data.shape[0]):
                row_text = []
                for j in range(plot_data.shape[1]):
                    if log_scale:
                        value = D_ratio[i, j]
                        base_text = f"Position: ({i}, {j})<br>"
                        base_text += f"σₕ: {sigma_hydro_field[i, j]:.4f} GPa<br>"
                        base_text += f"D/D₀: {value:.4f}<br>"
                        base_text += f"log₁₀(D/D₀): {plot_data[i, j]:.3f}"
                    else:
                        base_text = f"Position: ({i}, {j})<br>"
                        base_text += f"σₕ: {sigma_hydro_field[i, j]:.4f} GPa<br>"
                        base_text += f"D/D₀: {plot_data[i, j]:.4f}"
                    row_text.append(base_text)
                hover_text.append(row_text)
            
            # Validate and map colormap name to Plotly format
            plotly_colormap_mapping = {
                'viridis': 'Viridis',
                'plasma': 'Plasma',
                'inferno': 'Inferno',
                'magma': 'Magma',
                'cividis': 'Cividis',
                'turbo': 'Turbo',
                'hot': 'Hot',
                'afmhot': 'Hot',
                'gist_heat': 'Hot',
                'copper': 'Cividis',
                'summer': 'Viridis',
                'Wistia': 'YlOrRd',
                'spring': 'Viridis',
                'autumn': 'OrRd',
                'winter': 'Blues',
                'bone': 'Greys',
                'gray': 'Greys',
                'pink': 'Viridis',
                'gist_gray': 'Greys',
                'gist_yarg': 'Greys',
                'binary': 'Greys',
                'gist_earth': 'Earth',
                'terrain': 'Earth',
                'ocean': 'Deep',
                'gist_stern': 'Viridis',
                'gnuplot': 'Viridis',
                'gnuplot2': 'Viridis',
                'CMRmap': 'Viridis',
                'cubehelix': 'Viridis',
                'brg': 'Viridis',
                'gist_rainbow': 'Rainbow',
                'rainbow': 'Rainbow',
                'jet': 'Jet',
                'nipy_spectral': 'Rainbow',
                'gist_ncar': 'Rainbow',
                'hsv': 'Rainbow',
                'RdBu': 'RdBu',
                'RdYlBu': 'RdYlBu',
                'Spectral': 'Spectral',
                'coolwarm': 'RdBu',
                'bwr': 'RdBu',
                'seismic': 'RdBu',
                'BrBG': 'BrBG',
                'PiYG': 'PiYG',
                'PRGn': 'PRGn',
                'PuOr': 'PuOr',
                'RdGy': 'RdGy',
                'RdYlGn': 'RdYlGn',
                'Spectral_r': 'Spectral',
                'coolwarm_r': 'RdBu_r',
                'bwr_r': 'RdBu_r',
                'seismic_r': 'RdBu_r',
                'tab10': 'Plotly3',
                'tab20': 'Plotly3',
                'Set1': 'Set1',
                'Set2': 'Set2',
                'Set3': 'Set3',
                'tab20b': 'Plotly3',
                'tab20c': 'Plotly3',
                'Pastel1': 'Pastel1',
                'Pastel2': 'Pastel2',
                'Paired': 'Paired',
                'Accent': 'Accent',
                'Dark2': 'Dark2',
                'twilight': 'Viridis',
                'twilight_shifted': 'Viridis',
            }
            
            # Handle reverse colormaps
            is_reverse = cmap_name.endswith('_r')
            base_cmap = cmap_name[:-2] if is_reverse else cmap_name
            
            # Get the mapped colormap
            if base_cmap in plotly_colormap_mapping:
                plotly_cmap = plotly_colormap_mapping[base_cmap]
                if is_reverse:
                    plotly_cmap = f"{plotly_cmap}_r"
            else:
                plotly_cmap = 'RdBu'  # Standard default for diffusion
            
            # Create surface trace
            try:
                surface_trace = go.Surface(
                    z=plot_data,
                    x=X,
                    y=Y,
                    colorscale=plotly_cmap,
                    opacity=0.9,
                    contours={
                        "z": {
                            "show": True,
                            "usecolormap": True,
                            "project": {"z": True},
                            "highlightcolor": "limegreen",
                            "width": 2
                        }
                    },
                    hoverinfo='text',
                    text=hover_text,
                    colorbar=dict(
                        title=dict(
                            text=z_label,
                            font=dict(size=16, color='black'),
                        ),
                        tickfont=dict(size=14),
                        thickness=25,
                        len=0.8
                    )
                )
            except Exception as e:
                st.error(f"Warning: Could not use colormap {plotly_cmap}, falling back to 'RdBu'. Error: {e}")
                surface_trace = go.Surface(
                    z=plot_data,
                    x=X,
                    y=Y,
                    colorscale='RdBu',
                    opacity=0.9,
                    contours={
                        "z": {
                            "show": True,
                            "usecolormap": True,
                            "project": {"z": True},
                            "highlightcolor": "limegreen",
                            "width": 2
                        }
                    },
                    hoverinfo='text',
                    text=hover_text,
                    colorbar=dict(
                        title=dict(
                            text=z_label,
                            font=dict(size=16, color='black'),
                        ),
                        tickfont=dict(size=14),
                        thickness=25,
                        len=0.8
                    )
                )
            
            # Create figure
            fig = go.Figure(data=[surface_trace])
            
            # Enhanced title
            title_str = title
            if target_angle is not None and defect_type is not None:
                title_str = f"{title}<br>θ = {target_angle:.1f}°, {defect_type}, T = {T_K} K"
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title_str,
                    font=dict(size=24, color='darkblue'),
                    x=0.5,
                    y=0.95
                ),
                width=width,
                height=height,
                scene=dict(
                    xaxis=dict(
                        title=dict(text="X Position", font=dict(size=18, color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        showbackground=True
                    ),
                    yaxis=dict(
                        title=dict(text="Y Position", font=dict(size=18, color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        showbackground=True
                    ),
                    zaxis=dict(
                        title=dict(text=z_label, font=dict(size=18, color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        showbackground=True
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=100, b=0)
            )
            
            # Add annotation for key statistics
            max_enhance = np.max(D_ratio)
            min_enhance = np.min(D_ratio)
            mean_enhance = np.mean(D_ratio)
            fig.add_annotation(
                text=f"Max: {max_enhance:.2f}x<br>Min: {min_enhance:.2f}x<br>Mean: {mean_enhance:.2f}x",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color="black"),
                align="left",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            return fig, D_ratio
        except Exception as e:
            st.error(f"Error creating 3D diffusion surface: {e}")
            fig = go.Figure()
            fig.add_annotation(text="Error creating 3D diffusion surface", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, np.ones((10, 10))
    
    def create_diffusion_gradient_map(self, D_ratio_field, title="Diffusion Gradient Analysis",
                                     figsize=(12, 10), dpi=300):
        """Create comprehensive visualization of diffusion gradients"""
        # Compute gradient magnitude and direction
        grad_x, grad_y = np.gradient(D_ratio_field)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        
        # 1. Gradient magnitude
        ax1 = axes[0, 0]
        im1 = ax1.imshow(grad_magnitude, cmap='hot', aspect='equal',
                        interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Gradient Magnitude')
        ax1.set_title('Diffusion Gradient\nMagnitude', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # 2. Gradient direction (quiver plot)
        ax2 = axes[0, 1]
        stride = max(1, D_ratio_field.shape[0] // 20)
        Y, X = np.mgrid[0:D_ratio_field.shape[0]:stride, 0:D_ratio_field.shape[1]:stride]
        U = grad_x[::stride, ::stride]
        V = grad_y[::stride, ::stride]
        ax2.quiver(X, Y, U, V, grad_magnitude[::stride, ::stride], cmap='hot',
                  angles='xy', scale_units='xy', scale=0.5)
        ax2.set_title('Gradient Vector Field', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_aspect('equal')
        
        # 3. D/D₀ field
        ax3 = axes[1, 0]
        im3 = ax3.imshow(D_ratio_field, cmap='RdBu_r', aspect='equal',
                        interpolation='bilinear', origin='lower',
                        norm=LogNorm(vmin=0.1, vmax=10))
        plt.colorbar(im3, ax=ax3, label='D/D₀')
        ax3.set_title('Diffusion Enhancement\n(D/D₀)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        
        # 4. Laplacian (divergence of gradient)
        ax4 = axes[1, 1]
        laplacian = np.gradient(grad_x)[0] + np.gradient(grad_y)[1]
        im4 = ax4.imshow(laplacian, cmap='coolwarm', aspect='equal',
                        interpolation='bilinear', origin='lower')
        plt.colorbar(im4, ax=ax4, label='Laplacian (∇²D)')
        ax4.set_title('Diffusion Field\nCurvature', fontsize=14, fontweight='bold')
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Y Position')
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig, {'magnitude': grad_magnitude, 'direction_x': grad_x, 'direction_y': grad_y}
    
    def create_phase_diagram(self, sigma_range=(-5, 5), T_range=(300, 1200), 
                           material='Silver', figsize=(14, 10), dpi=300):
        """Create a phase diagram showing diffusion enhancement across stress and temperature"""
        # Create grid
        sigma_values = np.linspace(sigma_range[0], sigma_range[1], 100)
        T_values = np.linspace(T_range[0], T_range[1], 100)
        Sigma, T = np.meshgrid(sigma_values, T_values)
        
        # Compute diffusion enhancement
        D_ratio = np.zeros_like(Sigma)
        for i in range(Sigma.shape[0]):
            for j in range(Sigma.shape[1]):
                D_ratio[i, j] = DiffusionPhysics.compute_diffusion_enhancement(
                    Sigma[i, j], T[i, j], material, 'physics_corrected'
                )
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        
        # 1. Contour plot
        ax1 = axes[0, 0]
        levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
        contour = ax1.contourf(Sigma, T, D_ratio, levels=levels, cmap='RdBu_r', extend='both')
        contour_lines = ax1.contour(Sigma, T, D_ratio, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
        ax1.clabel(contour_lines, inline=True, fontsize=8)
        ax1.set_xlabel('Hydrostatic Stress (GPa)', fontsize=12)
        ax1.set_ylabel('Temperature (K)', fontsize=12)
        ax1.set_title(f'Diffusion Enhancement Contours\n{material}', fontsize=14, fontweight='bold')
        plt.colorbar(contour, ax=ax1, label='D/D₀')
        
        # 2. 3D surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf = ax2.plot_surface(Sigma, T, np.log10(D_ratio), cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.8)
        ax2.set_xlabel('Stress (GPa)', fontsize=10)
        ax2.set_ylabel('Temperature (K)', fontsize=10)
        ax2.set_zlabel('log₁₀(D/D₀)', fontsize=10)
        ax2.set_title('3D Diffusion Landscape', fontsize=14, fontweight='bold')
        
        # 3. Temperature slices
        ax3 = axes[1, 0]
        T_slices = [300, 500, 700, 900, 1100]
        colors = plt.cm.plasma(np.linspace(0, 1, len(T_slices)))
        for T_val, color in zip(T_slices, colors):
            idx = np.argmin(np.abs(T_values - T_val))
            profile = D_ratio[idx, :]
            ax3.plot(sigma_values, profile, color=color, linewidth=2, label=f'T = {T_val} K')
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Hydrostatic Stress (GPa)', fontsize=12)
        ax3.set_ylabel('D/D₀', fontsize=12)
        ax3.set_title('Temperature Dependence', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Stress slices
        ax4 = axes[1, 1]
        sigma_slices = [-4, -2, 0, 2, 4]
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(sigma_slices)))
        for sigma_val, color in zip(sigma_slices, colors):
            idx = np.argmin(np.abs(sigma_values - sigma_val))
            profile = D_ratio[:, idx]
            ax4.plot(T_values, profile, color=color, linewidth=2, label=f'σ = {sigma_val} GPa')
        ax4.set_xlabel('Temperature (K)', fontsize=12)
        ax4.set_ylabel('D/D₀', fontsize=12)
        ax4.set_title('Stress Dependence', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.suptitle(f'Diffusion Enhancement Phase Diagram\n{material}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig, D_ratio
    
    def create_stress_diffusion_correlation(self, sigma_hydro_field, D_ratio_field,
                                          title="Stress-Diffusion Correlation",
                                          figsize=(12, 10), dpi=300):
        """Create comprehensive correlation analysis between stress and diffusion"""
        # Flatten arrays for scatter plot
        sigma_flat = sigma_hydro_field.flatten()
        D_flat = D_ratio_field.flatten()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        
        # 1. Direct scatter plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(sigma_flat, D_flat, alpha=0.3, s=10, c='blue', edgecolors='none')
        # Add theoretical curve
        sigma_range = np.linspace(np.min(sigma_flat), np.max(sigma_flat), 100)
        props = DiffusionPhysics.get_material_properties('Silver')
        Omega = props['atomic_volume']
        T = 650
        k_B = DiffusionPhysics.k_B_J
        D_theoretical = np.exp(Omega * sigma_range * 1e9 / (k_B * T))
        ax1.plot(sigma_range, D_theoretical, 'r-', linewidth=2, label='Theoretical')
        ax1.set_xlabel('Hydrostatic Stress (GPa)')
        ax1.set_ylabel('D/D₀')
        ax1.set_title('Stress-Diffusion Correlation', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. 2D histogram
        ax2 = axes[0, 1]
        h = ax2.hist2d(sigma_flat, D_flat, bins=50, cmap='viridis', norm=LogNorm())
        plt.colorbar(h[3], ax=ax2, label='Count')
        ax2.set_xlabel('Hydrostatic Stress (GPa)')
        ax2.set_ylabel('D/D₀')
        ax2.set_title('2D Histogram', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        
        # 3. Correlation by stress magnitude
        ax3 = axes[1, 0]
        stress_bins = np.linspace(np.min(sigma_flat), np.max(sigma_flat), 20)
        mean_D = []
        std_D = []
        for i in range(len(stress_bins)-1):
            mask = (sigma_flat >= stress_bins[i]) & (sigma_flat < stress_bins[i+1])
            if np.sum(mask) > 0:
                mean_D.append(np.mean(D_flat[mask]))
                std_D.append(np.std(D_flat[mask]))
            else:
                mean_D.append(np.nan)
                std_D.append(np.nan)
        bin_centers = (stress_bins[:-1] + stress_bins[1:]) / 2
        ax3.errorbar(bin_centers, mean_D, yerr=std_D, fmt='o-', capsize=3,
                    linewidth=2, markersize=5, label='Mean ± Std')
        ax3.set_xlabel('Hydrostatic Stress (GPa)')
        ax3.set_ylabel('Mean D/D₀')
        ax3.set_title('Binned Correlation', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 4. Statistics box
        ax4 = axes[1, 1]
        ax4.axis('off')
        # Calculate correlation statistics
        from scipy.stats import pearsonr, spearmanr
        # Log transform for better correlation
        D_log = np.log10(D_flat)
        pearson_corr, pearson_p = pearsonr(sigma_flat, D_log)
        spearman_corr, spearman_p = spearmanr(sigma_flat, D_log)
        stats_text = (
            f"Pearson Correlation: {pearson_corr:.3f} (p={pearson_p:.2e})\n"
            f"Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.2e})\n\n"
            f"Max Enhancement: {np.max(D_flat):.2f}x\n"
            f"Min Enhancement: {np.min(D_flat):.2f}x\n"
            f"Mean Enhancement: {np.mean(D_flat):.2f}x\n\n"
            f"Tensile Regions (σ>0): {np.sum(sigma_flat>0)/len(sigma_flat)*100:.1f}%\n"
            f"Compressive Regions (σ<0): {np.sum(sigma_flat<0)/len(sigma_flat)*100:.1f}%"
        )
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax4.set_title('Correlation Statistics', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig

# =============================================
# RESULTS MANAGER FOR EXPORT
# =============================================
class ResultsManager:
    def __init__(self): pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        result = interpolation_result.copy()
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'transformer_bracketing_theory',
                'visualization_params': visualization_params
            },
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'shape': result['shape'],
                'statistics': result['statistics'],
                'weights': result['weights'],
                'num_sources': result.get('num_sources', 0),
                'source_theta_degrees': result.get('source_theta_degrees', []),
                'source_distances': result.get('source_distances', []),
                'source_indices': result.get('source_indices', [])
            }
        }
        
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
            
        return export_data
    
    def add_diffusion_to_export(self, interpolation_result, export_data):
        if 'diffusion_ratio' in interpolation_result['fields']:
            export_data['result']['diffusion_statistics'] = interpolation_result.get('diffusion_statistics', {})
            
            for field_name in ['diffusion_ratio', 'diffusion_effective', 'vacancy_ratio', 'diffusion_gradient']:
                if field_name in interpolation_result['fields']:
                    export_data['result'][f'{field_name}_data'] = interpolation_result['fields'][field_name].tolist()
                    
        return export_data
    
    def export_to_json(self, export_data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = export_data['result']['target_angle']
            defect = export_data['result']['target_params']['defect_type']
            filename = f"bracketing_interpolation_theta_{theta}_{defect}_{timestamp}.json"
            
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename = f"stress_fields_theta_{theta}_{defect}_{timestamp}.csv"
            
        data_dict = {}
        for field_name, field_data in interpolation_result['fields'].items():
            data_dict[field_name] = field_data.flatten()
            
        df = pd.DataFrame(data_dict)
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
    def _json_serializer(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, datetime): return obj.isoformat()
        elif isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        else: return str(obj)

# =============================================
# MAIN APPLICATION WITH CUSTOMIZATION UI
# =============================================
def main():
    st.set_page_config(page_title="Angular Bracketing Theory", layout="wide", page_icon="🧠", initial_sidebar_state="expanded")
    
    st.markdown("""
    <style>
    .main-header { font-size: 3.2rem !important; color: #1E3A8A !important; text-align: center; padding: 1rem;
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 900 !important; margin-bottom: 1rem; }
    .section-header { font-size: 2.0rem !important; color: #374151 !important; font-weight: 800 !important;
    border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    .info-box { background-color: #F0F9FF; border-left: 5px solid #3B82F6; padding: 1.2rem;
    border-radius: 0.6rem; margin: 1.2rem 0; font-size: 1.1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2.5rem; }
    .stTabs [data-baseweb="tab"] { height: 55px; white-space: pre-wrap; background-color: #F3F4F6;
    border-radius: 6px 6px 0 0; gap: 1.2rem; padding-top: 12px; padding-bottom: 12px; font-size: 1.1rem; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #3B82F6 !important; color: white !important; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 Angular Bracketing Theory</h1>', unsafe_allow_html=True)
    
    # Initialize Session State
    if 'solutions' not in st.session_state: st.session_state.solutions = []
    if 'loader' not in st.session_state: st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator(spatial_sigma=10.0, locality_weight_factor=0.5)
    if 'heatmap_visualizer' not in st.session_state: st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state: st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state: st.session_state.interpolation_result = None
    if 'selected_ground_truth' not in st.session_state: st.session_state.selected_ground_truth = None
    if 'diffusion_physics' not in st.session_state: st.session_state.diffusion_physics = DiffusionPhysics()
    
    # Sidebar: Configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Data Loading
        st.markdown("#### 📁 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                st.success(f"Loaded {len(st.session_state.solutions)} solutions" if st.session_state.solutions else "No solutions found")
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []; st.session_state.interpolation_result = None
                st.session_state.selected_ground_truth = None; st.success("Cache cleared")
        
        st.divider()
        
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        custom_theta = st.number_input("Target Angle θ (degrees)", min_value=0.0, max_value=180.0, value=54.7, step=0.1)
        defect_type = st.selectbox("Defect Type", options=['ISF', 'ESF', 'Twin', 'No Defect'], index=2)
        shape = st.selectbox("Shape", options=['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle'], index=0)
        kappa = st.slider("Kappa", min_value=0.1, max_value=2.0, value=0.6, step=0.01)
        auto_eigen = st.checkbox("Auto-calculate eigenstrain", value=True)
        if auto_eigen:
            eigen_strain = PhysicsParameters.get_eigenstrain(defect_type)
            st.metric("Eigenstrain ε₀", f"{eigen_strain:.3f}")
        else:
            eigen_strain = st.slider("Eigenstrain ε₀", min_value=0.0, max_value=3.0, value=2.12, step=0.001)
        
        st.divider()
        
        st.markdown('<h2 class="section-header">🌡️ Diffusion Physics</h2>', unsafe_allow_html=True)
        diffusion_material = st.selectbox("Material", options=['Silver', 'Copper', 'Aluminum', 'Nickel', 'Iron'], index=0)
        diffusion_T = st.slider("Temperature (K)", min_value=300, max_value=1500, value=650, step=10)
        diffusion_model = st.selectbox("Diffusion Model", options=['physics_corrected', 'temperature_reduction', 'activation_energy', 'vacancy_concentration'], index=0)
        
        st.divider()
        
        st.markdown('<h2 class="section-header">⚛️ Theory & Attention</h2>', unsafe_allow_html=True)
        spatial_sigma = st.slider("Angular Kernel Sigma (degrees)", min_value=1.0, max_value=45.0, value=10.0, step=0.5)
        locality_weight_factor = st.slider("Theory vs. Learned", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        temperature = st.slider("Attention Temperature", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        st.divider()
        
        if st.button("🧠 Perform Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Processing..."):
                    target_params = {'defect_type': defect_type, 'eps0': eigen_strain, 'kappa': kappa, 'theta': np.radians(custom_theta), 'shape': shape}
                    result = st.session_state.transformer_interpolator.interpolate_spatial_fields(st.session_state.solutions, custom_theta, target_params)
                    if result:
                        st.session_state.interpolation_result = result
                        st.success("Interpolation successful.")
                        st.session_state.selected_ground_truth = None
                    else:
                        st.error("Interpolation failed.")

    # Only show the Diffusion Visualization tab after interpolation
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        diffusion_viz_tab, = st.tabs(["🔬 Diffusion Visualization"])
        
        with diffusion_viz_tab:
            st.markdown('<h2 class="section-header">🔬 Diffusion Enhancement Visualization</h2>', unsafe_allow_html=True)
            
            # Check if required diffusion data is available
            if 'sigma_hydro' not in result['fields']:
                st.warning("Hydrostatic stress field not available for diffusion calculations.")
                return
            
            # Display theoretical background with enhanced styling
            st.markdown("""
            <div class="info-box">
            <strong>📚 Diffusion Enhancement Theory:</strong><br>
            The diffusion coefficient under stress follows: $D(σ_h)/D_0 = \\exp(Ωσ_h/(k_B T))$<br>
            Where Ω is atomic volume, σ_h is hydrostatic stress, k_B is Boltzmann constant, T is temperature.<br>
            • <strong>Tensile stress (σ_h > 0)</strong> → Enhanced diffusion (D/D₀ > 1) - PEAKS<br>
            • <strong>Compressive stress (σ_h < 0)</strong> → Suppressed diffusion (D/D₀ < 1) - VALLEYS<br>
            • <strong>Habit plane (θ ≈ 54.7°)</strong> → Maximum tensile stress → Peak diffusion enhancement
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization controls
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            
            with col_viz1:
                diffusion_viz_type = st.selectbox(
                    "Visualization Type",
                    options=[
                        'Diffusion Enhancement (D/D₀)', 
                        '3D Surface', 
                        'Interactive Heatmap', 
                        'Interactive 3D',
                        'Vacancy Concentration',
                        'Diffusion Gradient',
                        'Phase Diagram',
                        'Stress-Diffusion Correlation'
                    ],
                    index=0,
                    key="diffusion_viz_type"
                )
                
            with col_viz2:
                cmap_category = st.selectbox(
                    "Colormap Category",
                    options=list(COLORMAP_OPTIONS.keys()),
                    index=1,  # Diverging category has good options for diffusion
                    key="diffusion_cmap_category"
                )
                cmap_options = COLORMAP_OPTIONS[cmap_category]
                
            with col_viz3:
                cmap_name = st.selectbox(
                    "Colormap",
                    options=cmap_options,
                    index=cmap_options.index('RdBu_r') if 'RdBu_r' in cmap_options else 0,
                    key="diffusion_cmap_name"
                )
            
            # Log scale toggle
            col_log1, col_log2 = st.columns([1, 3])
            with col_log1:
                log_scale = st.checkbox("Use Log Scale", value=True, key="diffusion_log_scale")
                st.caption("Log scale better shows enhancement/suppression")
            
            # Show colormap preview
            if st.checkbox("Preview Colormap", key="preview_diffusion_colormap"):
                fig_cmap, ax_cmap = plt.subplots(figsize=(12, 1), dpi=150)
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                ax_cmap.imshow(gradient, aspect='auto', cmap=cmap_name)
                ax_cmap.set_title(f"Colormap: {cmap_name}", fontsize=12)
                ax_cmap.set_xticks([])
                ax_cmap.set_yticks([])
                st.pyplot(fig_cmap)
            
            # Display the selected visualization
            sigma_hydro_field = result['fields']['sigma_hydro']
            
            if diffusion_viz_type == 'Diffusion Enhancement (D/D₀)':
                # 2D heatmap of diffusion enhancement
                fig_diff, D_ratio = st.session_state.heatmap_visualizer.create_diffusion_heatmap(
                    sigma_hydro_field,
                    title="Diffusion Enhancement Map",
                    T_K=diffusion_T,
                    material=diffusion_material,
                    cmap_name=cmap_name,
                    figsize=(12, 10),
                    log_scale=log_scale,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    model=diffusion_model
                )
                st.pyplot(fig_diff)
                
                # Show key statistics
                st.markdown("#### 📊 Diffusion Statistics")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Max Enhancement", f"{np.max(D_ratio):.2f}x")
                with col_stat2:
                    st.metric("Min (Suppression)", f"{np.min(D_ratio):.2f}x")
                with col_stat3:
                    st.metric("Mean", f"{np.mean(D_ratio):.2f}x")
                with col_stat4:
                    enhanced_area = np.sum(D_ratio > 1.0) / D_ratio.size * 100
                    st.metric("Enhanced Area", f"{enhanced_area:.1f}%")
            
            elif diffusion_viz_type == '3D Surface':
                # 3D surface plot of diffusion enhancement
                fig_3d, D_ratio = st.session_state.heatmap_visualizer.create_diffusion_3d_surface_plot(
                    sigma_hydro_field,
                    title="3D Diffusion Landscape",
                    T_K=diffusion_T,
                    material=diffusion_material,
                    cmap_name=cmap_name,
                    figsize=(14, 10),
                    log_scale=log_scale,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    model=diffusion_model
                )
                st.pyplot(fig_3d)
            
            elif diffusion_viz_type == 'Interactive Heatmap':
                # Interactive heatmap
                fig_interactive, D_ratio = st.session_state.heatmap_visualizer.create_interactive_diffusion_heatmap(
                    sigma_hydro_field,
                    title="Diffusion Enhancement Map",
                    T_K=diffusion_T,
                    material=diffusion_material,
                    cmap_name=cmap_name,
                    width=800,
                    height=700,
                    log_scale=log_scale,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    model=diffusion_model
                )
                st.plotly_chart(fig_interactive, use_container_width=True)
            
            elif diffusion_viz_type == 'Interactive 3D':
                # Interactive 3D surface
                fig_3d_interactive, D_ratio = st.session_state.heatmap_visualizer.create_interactive_diffusion_3d(
                    sigma_hydro_field,
                    title="Interactive 3D Diffusion",
                    T_K=diffusion_T,
                    material=diffusion_material,
                    cmap_name=cmap_name,
                    width=900,
                    height=700,
                    log_scale=log_scale,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    model=diffusion_model
                )
                st.plotly_chart(fig_3d_interactive, use_container_width=True)
            
            elif diffusion_viz_type == 'Vacancy Concentration':
                # Vacancy concentration map
                st.markdown("#### 🧬 Vacancy Concentration Enhancement (C_v/C_v₀)")
                # Compute vacancy concentration ratio
                C_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro_field, diffusion_T, diffusion_material, 'vacancy_concentration'
                )
                
                fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
                im = ax.imshow(C_ratio, cmap='plasma', aspect='equal',
                              interpolation='bilinear', origin='lower',
                              norm=LogNorm(vmin=0.1, vmax=10))
                plt.colorbar(im, ax=ax, label='C_v/C_v₀')
                ax.set_title(f'Vacancy Concentration Enhancement\n{diffusion_material}, T = {diffusion_T} K',
                           fontsize=20, fontweight='bold', pad=20)
                ax.set_xlabel('X Position (nm)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Y Position (nm)', fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                
                # Add statistics
                stats_text = (
                    f"Max: {np.max(C_ratio):.2f}x\n"
                    f"Min: {np.min(C_ratio):.2f}x\n"
                    f"Mean: {np.mean(C_ratio):.2f}x"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                st.pyplot(fig)
            
            elif diffusion_viz_type == 'Diffusion Gradient':
                # Diffusion gradient analysis
                st.markdown("#### 📊 Diffusion Gradient Analysis")
                # Compute diffusion ratio field first
                D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro_field, diffusion_T, diffusion_material, diffusion_model
                )
                
                fig_grad, grad_info = st.session_state.heatmap_visualizer.create_diffusion_gradient_map(
                    D_ratio,
                    title=f"Diffusion Gradient Analysis\nθ = {result['target_angle']:.1f}°, {result['target_params']['defect_type']}",
                    figsize=(12, 10)
                )
                st.pyplot(fig_grad)
                
                # Show gradient statistics
                st.markdown("#### 📈 Gradient Statistics")
                grad_mag = grad_info['magnitude']
                col_grad1, col_grad2, col_grad3 = st.columns(3)
                with col_grad1:
                    st.metric("Max Gradient", f"{np.max(grad_mag):.3f}")
                with col_grad2:
                    st.metric("Mean Gradient", f"{np.mean(grad_mag):.3f}")
                with col_grad3:
                    steep_regions = np.sum(grad_mag > np.mean(grad_mag) + np.std(grad_mag))
                    st.metric("Steep Regions", f"{steep_regions}")
            
            elif diffusion_viz_type == 'Phase Diagram':
                # Diffusion phase diagram
                st.markdown("#### 🔬 Diffusion Phase Diagram")
                fig_phase, _ = st.session_state.heatmap_visualizer.create_phase_diagram(
                    sigma_range=(-5, 5),
                    T_range=(300, 1200),
                    material=diffusion_material,
                    figsize=(14, 10)
                )
                st.pyplot(fig_phase)
                
                # Add marker for current conditions
                mean_stress = np.mean(sigma_hydro_field)
                st.info(f"Current conditions: σ = {mean_stress:.2f} GPa, T = {diffusion_T} K")
            
            elif diffusion_viz_type == 'Stress-Diffusion Correlation':
                # Stress-diffusion correlation analysis
                st.markdown("#### 📊 Stress-Diffusion Correlation")
                D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro_field, diffusion_T, diffusion_material, diffusion_model
                )
                
                fig_corr = st.session_state.heatmap_visualizer.create_stress_diffusion_correlation(
                    sigma_hydro_field, D_ratio,
                    title=f"Stress-Diffusion Correlation\nθ = {result['target_angle']:.1f}°, {result['target_params']['defect_type']}",
                    figsize=(12, 10)
                )
                st.pyplot(fig_corr)
            
            # Material comparison button
            st.markdown("---")
            st.markdown("#### 🔄 Compare Different Materials")
            if st.button("Compare Materials", use_container_width=True):
                materials = ['Silver', 'Copper', 'Aluminum']
                fields = [sigma_hydro_field] * len(materials)
                titles = [f"{mat} at {diffusion_T}K" for mat in materials]
                
                fig_compare, axes = plt.subplots(1, len(materials), figsize=(18, 6), dpi=300)
                if len(materials) == 1:
                    axes = [axes]
                
                for i, (mat, title) in enumerate(zip(materials, titles)):
                    D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                        sigma_hydro_field, diffusion_T, mat, diffusion_model
                    )
                    
                    if log_scale:
                        plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
                        vmin, vmax = -1, 1
                    else:
                        plot_data = D_ratio
                        vmin, vmax = 0.1, 10
                    
                    im = axes[i].imshow(plot_data, cmap=cmap_name, vmin=vmin, vmax=vmax,
                                      aspect='equal', interpolation='bilinear', origin='lower')
                    
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04,
                                label=r"$log_{10}(D/D_0)$" if log_scale else r"$D/D_0$")
                    
                    axes[i].set_title(title, fontsize=16, fontweight='bold')
                    axes[i].set_xlabel('X Position')
                    axes[i].set_ylabel('Y Position')
                
                plt.suptitle(f'Diffusion Enhancement Comparison\nθ = {result["target_angle"]:.1f}°, {result["target_params"]["defect_type"]}',
                           fontsize=20, fontweight='bold', y=0.98)
                plt.tight_layout()
                st.pyplot(fig_compare)

if __name__ == "__main__":
    main()
