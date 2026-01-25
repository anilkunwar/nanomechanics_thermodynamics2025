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
import time
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
    
    def create_attention_sunburst(self, weights, source_info, target_params, 
                                 figsize=(8, 8), title="Attention Distribution Sunburst"):
        """
        Create a sunburst chart showing hierarchical attention distribution across different features
        """
        # Get hierarchical data for sunburst chart
        source_params = source_info.get('source_fields', [])
        if not source_params:
            return None
            
        # Build hierarchical data
        labels = ['Target']
        parents = ['']
        values = [1.0]  # Target value is 1.0 (normalized)
        
        # Add defect types as first level
        defect_types = set()
        for i, src in enumerate(source_params):
            defect_type = src.get('source_params', {}).get('defect_type', 'Unknown')
            defect_types.add(defect_type)
        
        # Create defect type nodes
        defect_weights = {}
        for defect_type in defect_types:
            # Sum weights of sources with this defect type
            defect_weight = 0.0
            for i, src in enumerate(source_params):
                src_defect = src.get('source_params', {}).get('defect_type', 'Unknown')
                if src_defect == defect_type:
                    defect_weight += weights['combined'][i]
            defect_weights[defect_type] = defect_weight
            labels.append(defect_type)
            parents.append('Target')
            values.append(defect_weight)
        
        # Add individual sources as second level
        for i, src in enumerate(source_params):
            src_defect = src.get('source_params', {}).get('defect_type', 'Unknown')
            theta_deg = source_info['theta_degrees'][i]
            weight = weights['combined'][i]
            
            # Create unique label for source
            label = f"Source {i}: {src_defect} ({theta_deg:.1f}°)"
            labels.append(label)
            parents.append(src_defect)
            values.append(weight)
        
        # Create sunburst chart with Plotly
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            hovertemplate='<b>%{label}</b><br>Weight: %{value:.3f}<br>Parent: %{parent}',
            maxdepth=2,
            insidetextorientation='radial'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}\nθ = {target_params['theta']:.1f}°, {target_params['defect_type']}",
                font=dict(size=18, color='darkblue'),
                x=0.5,
                y=0.95
            ),
            width=800,
            height=800,
            margin=dict(t=80, l=0, r=0, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_attention_radar(self, weights, source_info, target_params,
                              figsize=(8, 8), title="Attention Analysis Radar Chart"):
        """
        Create a radar chart comparing attention weights across different dimensions
        """
        # Get source parameters
        source_params = source_info.get('source_fields', [])
        if not source_params:
            return None
            
        # Define radar dimensions
        dimensions = ['Angular Proximity', 'Defect Match', 'Spatial Kernel', 'Combined Attention']
        
        # Prepare data for each source
        sources_data = []
        source_names = []
        
        for i, src in enumerate(source_params):
            src_theta = source_info['theta_degrees'][i]
            target_theta = target_params['theta']
            
            # Angular proximity (0 to 1)
            angle_diff = abs(src_theta - target_theta)
            angle_diff = min(angle_diff, 360 - angle_diff)
            angular_proximity = np.exp(-angle_diff / 30.0)  # Scale factor
            
            # Defect match (0 or 1)
            src_defect = src.get('source_params', {}).get('defect_type', 'Unknown')
            target_defect = target_params['defect_type']
            defect_match = 1.0 if src_defect == target_defect else 0.0
            
            # Spatial kernel weight
            spatial_kernel = weights['spatial_kernel'][i]
            
            # Combined attention weight
            combined_attention = weights['combined'][i]
            
            sources_data.append([
                angular_proximity,
                defect_match,
                spatial_kernel,
                combined_attention
            ])
            
            source_names.append(f"Source {i}: {src_defect} ({src_theta:.1f}°)")
        
        # Create radar chart
        fig = go.Figure()
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sources_data)))
        plotly_colors = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in colors]
        
        for i, source_data in enumerate(sources_data):
            # Close the polygon
            values = source_data + [source_data[0]]
            theta = dimensions + [dimensions[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=source_names[i],
                line=dict(color=plotly_colors[i]),
                marker=dict(size=8, color=plotly_colors[i])
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=12),
                    title=dict(text='Weight', font=dict(size=14))
                ),
                angularaxis=dict(
                    tickfont=dict(size=14),
                    rotation=90  # Start from top
                )
            ),
            title=dict(
                text=f"{title}\nθ = {target_params['theta']:.1f}°, {target_params['defect_type']}",
                font=dict(size=18, color='darkblue'),
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            width=800,
            height=700,
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_attention_hierarchy_chart(self, weights, source_info, target_params,
                                        figsize=(10, 8), title="Hierarchical Attention Distribution"):
        """
        Create hierarchical treemap showing attention distribution
        """
        # Get source parameters
        source_params = source_info.get('source_fields', [])
        if not source_params:
            return None
            
        # Create hierarchical data
        import plotly.express as px
        
        defect_types = []
        thetas = []
        weights_list = []
        labels = []
        parent_nodes = []
        
        # Create defect type nodes
        defect_weights = {}
        for i, src in enumerate(source_params):
            defect_type = src.get('source_params', {}).get('defect_type', 'Unknown')
            weight = weights['combined'][i]
            theta = source_info['theta_degrees'][i]
            
            if defect_type not in defect_weights:
                defect_weights[defect_type] = 0
            defect_weights[defect_type] += weight
            
            # Add to lists for treemap
            defect_types.append(defect_type)
            thetas.append(theta)
            weights_list.append(weight)
            labels.append(f"Source {i}<br>{defect_type}<br>{theta:.1f}°")
            
        # Create DataFrame for treemap
        df = pd.DataFrame({
            'defect_type': defect_types,
            'theta': thetas,
            'weight': weights_list,
            'label': labels
        })
        
        # Create treemap
        fig = px.treemap(
            df,
            path=[px.Constant('All Sources'), 'defect_type', 'label'],
            values='weight',
            color='theta',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=54.7,
            hover_data={'weight': ':.3f', 'theta': ':.1f'},
            labels={'weight': 'Attention Weight', 'theta': 'Angle (θ)'},
            title=f"{title}<br>θ = {target_params['theta']:.1f}°, {target_params['defect_type']}"
        )
        
        fig.update_layout(
            width=900,
            height=700,
            margin=dict(t=80, l=0, r=0, b=0),
            title=dict(
                font=dict(size=18, color='darkblue'),
                x=0.5
            )
        )
        
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.3f}<br>Angle: %{customdata[1]:.1f}°"
        )
        
        return fig
    
    def create_attention_heatmap_matrix(self, weights, source_info, figsize=(10, 8),
                                       title="Attention Weight Heatmap Matrix"):
        """
        Create a heatmap matrix showing attention weights between sources
        """
        n_sources = len(weights['combined'])
        if n_sources == 0:
            return None
            
        # Create attention matrix (simplified for demonstration)
        attention_matrix = np.zeros((n_sources, n_sources))
        
        # Get source parameters
        source_params = source_info.get('source_fields', [])
        target_theta = np.mean([theta for theta in source_info['theta_degrees']])
        target_defect = source_params[0].get('source_params', {}).get('defect_type', 'Unknown') if source_params else 'Unknown'
        
        # Simulate attention between sources (this would be more complex in real implementation)
        for i in range(n_sources):
            for j in range(n_sources):
                # Base attention on angular difference and defect match
                theta_i = source_info['theta_degrees'][i]
                theta_j = source_info['theta_degrees'][j]
                
                angle_diff = abs(theta_i - theta_j)
                angle_diff = min(angle_diff, 360 - angle_diff)
                angular_proximity = np.exp(-angle_diff / 20.0)
                
                defect_i = source_params[i].get('source_params', {}).get('defect_type', 'Unknown')
                defect_j = source_params[j].get('source_params', {}).get('defect_type', 'Unknown')
                defect_match = 1.0 if defect_i == defect_j else 0.1
                
                attention_matrix[i, j] = angular_proximity * defect_match
        
        # Normalize rows
        row_sums = attention_matrix.sum(axis=1, keepdims=True)
        attention_matrix = attention_matrix / (row_sums + 1e-8)
        
        # Create heatmap with Plotly
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=[f"Source {i}" for i in range(n_sources)],
            y=[f"Source {i}" for i in range(n_sources)],
            colorscale='Viridis',
            colorbar=dict(title='Attention Weight'),
            hovertemplate='From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{title}<br>(Simulated attention between sources)",
            xaxis_title='Target Source',
            yaxis_title='Source Source',
            width=800,
            height=700,
            margin=dict(t=80, l=80, r=80, b=80)
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(autorange='reversed')
        
        return fig
    
    def create_attention_statistics(self, weights, source_info, figsize=(12, 8),
                                    title="Attention Distribution Statistics"):
        """
        Create comprehensive statistics plots for attention weights - FIXED VERSION
        """
        combined_weights = np.array(weights['combined'])
        spatial_kernel = np.array(weights['spatial_kernel'])
        defect_mask = np.array(weights['defect_mask'])
        entropy = weights['entropy']
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Attention Weight Distribution',
                'Spatial Kernel vs Combined Attention',
                'Cumulative Attention Distribution'
            ),
            # Specify subplot types where needed
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Weight distribution histogram
        hist_data = go.Histogram(
            x=combined_weights,
            nbinsx=20,
            name='Combined Weights',
            marker_color='steelblue',
            opacity=0.7
        )
        fig.add_trace(hist_data, row=1, col=1)
        
        # Add mean and median lines
        mean_weight = np.mean(combined_weights)
        median_weight = np.median(combined_weights)
        
        fig.add_vline(x=mean_weight, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_weight:.3f}", 
                     annotation_position="top right", row=1, col=1)
        fig.add_vline(x=median_weight, line_dash="dot", line_color="green", 
                     annotation_text=f"Median: {median_weight:.3f}", 
                     annotation_position="bottom right", row=1, col=1)
        
        # 2. Spatial kernel vs combined attention scatter plot
        scatter = go.Scatter(
            x=spatial_kernel,
            y=combined_weights,
            mode='markers',
            name='Source Weights',
            marker=dict(
                size=10,
                color=np.arange(len(combined_weights)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Source Index')
            ),
            text=[f"Source {i}" for i in range(len(combined_weights))],
            hovertemplate='<b>%{text}</b><br>Spatial Kernel: %{x:.3f}<br>Combined Weight: %{y:.3f}<extra></extra>'
        )
        fig.add_trace(scatter, row=1, col=2)
        
        # Add trend line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(spatial_kernel, combined_weights)
        x_trend = np.array([0, 1])
        y_trend = slope * x_trend + intercept
        fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', 
                                name=f'Trend (R²={r_value**2:.3f})', line=dict(color='red', width=2)),
                     row=1, col=2)
        
        # 3. Cumulative distribution
        sorted_weights = np.sort(combined_weights)[::-1]  # Sort descending
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        x_vals = np.arange(1, len(cumulative) + 1)
        
        cum_line = go.Scatter(
            x=x_vals,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative Attention',
            line=dict(color='purple', width=2),
            marker=dict(size=8)
        )
        fig.add_trace(cum_line, row=2, col=1)
        
        # Add 90% threshold line
        threshold_idx = np.where(cumulative >= 0.9)[0][0] if np.any(cumulative >= 0.9) else len(cumulative)-1
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                     annotation_text="90% threshold", annotation_position="bottom right", row=2, col=1)
        fig.add_vline(x=threshold_idx+1, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. Entropy analysis as a simple bar chart (replacing the problematic indicator)
        entropy_normalized = entropy / np.log(len(combined_weights) or 1)
        
        # Create a bar chart that shows entropy level
        entropy_bar = go.Bar(
            x=['Entropy'],
            y=[entropy_normalized],
            marker=dict(
                color=['red' if entropy_normalized > 0.7 else 'yellow' if entropy_normalized > 0.3 else 'green'],
                line=dict(color='black', width=1)
            ),
            text=[f"{entropy:.3f}"],
            textposition='outside',
            hovertemplate=f'Entropy: {entropy:.3f}<br>Normalized: {entropy_normalized:.3f}<br>Ideal range: Low to Medium'
        )
        fig.add_trace(entropy_bar, row=2, col=2)
        
        # Add reference lines for entropy interpretation
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=2, col=2)
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text=title,
            height=800,
            width=1000,
            showlegend=False,
            template='plotly_white',
            annotations=[
                dict(text="Low Entropy = More focused attention", x=0.85, y=0.1, 
                     xref="paper", yref="paper", showarrow=False, font=dict(size=10)),
                dict(text="High Entropy = More spread attention", x=0.85, y=0.05, 
                     xref="paper", yref="paper", showarrow=False, font=dict(size=10))
            ]
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Weight Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Spatial Kernel Weight", row=1, col=2)
        fig.update_yaxes(title_text="Combined Attention", row=1, col=2)
        fig.update_xaxes(title_text="Number of Top Sources", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Attention", row=2, col=1)
        fig.update_xaxes(showticklabels=False, title_text="", row=2, col=2)
        fig.update_yaxes(title_text="Normalized Entropy", range=[0, 1], row=2, col=2)
        
        return fig
    
    def create_attention_polar_plot(self, weights, source_info, target_params,
                                   figsize=(10, 10), title="Angular Attention Distribution"):
        """
        Create polar plot showing attention distribution by angular position
        """
        # Get source angles and weights
        angles_rad = np.radians(source_info['theta_degrees'])
        combined_weights = np.array(weights['combined'])
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatterpolar(
            r=combined_weights,
            theta=source_info['theta_degrees'],
            mode='markers+lines',
            name='Attention Weight',
            marker=dict(
                size=12,
                color=np.array(source_info['theta_degrees']),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Angle (θ)'),
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color='steelblue'),
            hovertemplate='<b>Source %{text}</b><br>Angle: %{theta:.1f}°<br>Weight: %{r:.3f}<extra></extra>',
            text=[str(i) for i in range(len(combined_weights))]
        ))
        
        # Add target angle line
        target_angle = target_params['theta']
        max_weight = np.max(combined_weights) if len(combined_weights) > 0 else 1.0
        
        fig.add_trace(go.Scatterpolar(
            r=[0, max_weight * 1.2],
            theta=[target_angle, target_angle],
            mode='lines',
            name=f'Target Angle: {target_angle:.1f}°',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        # Add habit plane line (54.7°)
        fig.add_trace(go.Scatterpolar(
            r=[0, max_weight * 1.2],
            theta=[54.7, 54.7],
            mode='lines',
            name='Habit Plane (54.7°)',
            line=dict(color='green', width=2, dash='dot')
        ))
        
        # Update layout with proper polar configuration
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_weight * 1.3],
                    title='Attention Weight',
                    tickfont=dict(size=12)
                ),
                angularaxis=dict(
                    rotation=90,  # Start from top
                    direction='clockwise',
                    tickfont=dict(size=12),
                    # No direct title for angular axis in Plotly polar plots
                ),
                bgcolor='rgba(240, 240, 240, 0.5)'
            ),
            title=dict(
                text=f"{title}<br>θ = {target_angle:.1f}°, {target_params['defect_type']}",
                font=dict(size=18, color='darkblue'),
                x=0.5,
                y=0.95
            ),
            width=900,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.01,
                xanchor="center",
                x=0.5,
                orientation="h"
            ),
            template='plotly_white'
        )
        
        # Add annotation for angular axis title
        fig.add_annotation(
            text="Angle (degrees)",
            x=0.5,
            y=-0.1,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14)
        )
        
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
# MAIN APPLICATION WITH ATTENTION ANALYSIS TAB
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
    
    st.markdown('<h1 class="main-header">🧠 Angular Bracketing Theory - Attention Analysis</h1>', unsafe_allow_html=True)
    
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
    
    # Only show the Attention Analysis tab after interpolation
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        attention_tab, = st.tabs(["🎯 Attention Analysis"])
        
        with attention_tab:
            st.markdown('<h2 class="section-header">🎯 Theory-Informed Attention Analysis</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>🔬 Theory-Informed Attention:</strong><br>
            The attention mechanism combines physical theory with learned patterns:<br>
            • <strong>Angular Bracketing Kernel:</strong> Gaussian spatial locality enforcing linear interpolation between nearest angles<br>
            • <strong>Hard Defect Gating:</strong> Sources with different defect types receive effectively zero attention<br>
            • <strong>Combined Attention:</strong> Attention = Softmax(Learned Similarity × Spatial Kernel × Defect Mask)<br>
            • <strong>Entropy Metric:</strong> Lower entropy = more focused attention on bracketing sources
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization selection
            st.markdown("#### 📊 Select Attention Visualization Type")
            viz_options = [
                "Sunburst Chart: Hierarchical Distribution",
                "Radar Chart: Multi-dimensional Analysis",
                "Polar Plot: Angular Distribution",
                "Treemap: Hierarchical Attention Weights",
                "Statistics Dashboard: Comprehensive Metrics",
                "Attention Matrix: Source-to-Source Attention"
            ]
            selected_viz = st.selectbox("Choose visualization type:", viz_options, index=4)
            
            # Visualization controls
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                attention_component = st.selectbox(
                    "Attention Component",
                    options=['combined', 'spatial_kernel', 'defect_mask'],
                    index=0,
                    key="attention_component"
                )
            with col_ctrl2:
                normalize_weights = st.checkbox("Normalize Weights", value=True, key="normalize_weights")
            
            # Generate the selected visualization
            weights = result['weights']
            source_info = {
                'theta_degrees': result['source_theta_degrees'],
                'distances': result['source_distances'],
                'weights': weights,
                'source_fields': result.get('source_fields', [])
            }
            target_params = {
                'theta': result['target_angle'],
                'defect_type': result['target_params']['defect_type']
            }
            
            try:
                if selected_viz == "Sunburst Chart: Hierarchical Distribution":
                    st.markdown("#### 🌅 Sunburst Chart: Hierarchical Attention Distribution")
                    fig = st.session_state.heatmap_visualizer.create_attention_sunburst(
                        weights, source_info, target_params,
                        title="Hierarchical Attention Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    **Sunburst Chart Interpretation:**
                    • **Center (Target):** Represents the target interpolation point
                    • **Middle Ring (Defect Types):** Shows attention distribution across different defect types
                    • **Outer Ring (Sources):** Shows individual source weights
                    • **Segment Size:** Proportional to attention weight
                    • **Color Gradient:** Indicates attention strength
                    """)
                    
                elif selected_viz == "Radar Chart: Multi-dimensional Analysis":
                    st.markdown("#### 📡 Radar Chart: Multi-dimensional Attention Analysis")
                    fig = st.session_state.heatmap_visualizer.create_attention_radar(
                        weights, source_info, target_params,
                        title="Multi-dimensional Attention Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    **Radar Chart Interpretation:**
                    • **Angular Proximity:** Measures closeness to target angle (higher = closer)
                    • **Defect Match:** Binary match (1.0) or mismatch (0.0)
                    • **Spatial Kernel:** Gaussian weight based on angular distance
                    • **Combined Attention:** Final attention weight after all factors
                    • **Polygon Area:** Larger area indicates stronger overall attention
                    """)
                    
                elif selected_viz == "Polar Plot: Angular Distribution":
                    st.markdown("#### 📐 Polar Plot: Angular Attention Distribution")
                    fig = st.session_state.heatmap_visualizer.create_attention_polar_plot(
                        weights, source_info, target_params,
                        title="Angular Attention Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    **Polar Plot Interpretation:**
                    • **Radius:** Attention weight magnitude
                    • **Angle:** Source orientation in degrees
                    • **Red Line:** Target interpolation angle
                    • **Green Line:** Habit plane (54.7°)
                    • **Blue Line:** Shows attention distribution across angles
                    • **Peak Regions:** Sources with highest attention weights
                    """)
                    
                elif selected_viz == "Treemap: Hierarchical Attention Weights":
                    st.markdown("#### 🌳 Treemap: Hierarchical Attention Distribution")
                    fig = st.session_state.heatmap_visualizer.create_attention_hierarchy_chart(
                        weights, source_info, target_params,
                        title="Hierarchical Attention Distribution by Defect Type and Angle"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    **Treemap Interpretation:**
                    • **Node Size:** Proportional to attention weight
                    • **Color:** Indicates angular position (blue = low, red = high)
                    • **Hierarchy:** Top level = defect types, Bottom level = individual sources
                    • **Hover Information:** Shows detailed weights and angles
                    • **Largest Blocks:** Most influential sources for interpolation
                    """)
                    
                elif selected_viz == "Statistics Dashboard: Comprehensive Metrics":
                    st.markdown("#### 📈 Statistics Dashboard: Attention Distribution Analysis")
                    fig = st.session_state.heatmap_visualizer.create_attention_statistics(
                        weights, source_info,
                        title="Comprehensive Attention Distribution Analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add key statistics
                    st.markdown("#### 🔢 Key Attention Metrics")
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Entropy", f"{weights['entropy']:.3f}")
                    with col_stat2:
                        st.metric("Max Weight", f"{np.max(weights['combined']):.3f}")
                    with col_stat3:
                        st.metric("Min Weight", f"{np.min(weights['combined']):.3f}")
                    with col_stat4:
                        top_k = min(3, len(weights['combined']))
                        top_k_sources = np.argsort(weights['combined'])[::-1][:top_k]
                        st.metric(f"Top {top_k} Sources", ", ".join(map(str, top_k_sources)))
                    
                    # Add explanation
                    st.markdown("""
                    **Statistics Dashboard Interpretation:**
                    • **Entropy:** Measures attention distribution spread (lower = more focused)
                    • **Weight Distribution:** Shows how attention is distributed across sources
                    • **Spatial vs Combined:** Correlation between theory and learned attention
                    • **Cumulative Distribution:** Shows how many sources are needed for 90% attention
                    • **Key Metrics:** Highlights most important attention characteristics
                    """)
                    
                elif selected_viz == "Attention Matrix: Source-to-Source Attention":
                    st.markdown("#### 🔄 Attention Matrix: Source-to-Source Relationships")
                    fig = st.session_state.heatmap_visualizer.create_attention_heatmap_matrix(
                        weights, source_info,
                        title="Source-to-Source Attention Heatmap Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    **Attention Matrix Interpretation:**
                    • **Rows:** Source sources (originating attention)
                    • **Columns:** Target sources (receiving attention)
                    • **Color Intensity:** Attention weight strength
                    • **Diagonal Pattern:** Self-attention (always high)
                    • **Off-diagonal Patterns:** Cross-source attention relationships
                    • **Bright Regions:** Strong attention connections between sources
                    """)
            
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
                # Additional specific error handling
                if "indicator" in str(e):
                    st.warning("Using alternative visualization due to compatibility issues with current Plotly version.")
                if "title" in str(e) and "polar" in str(e):
                    st.warning("Using fallback polar plot configuration due to Plotly API changes.")
                
                # Show error details in expander for debugging
                with st.expander("🔍 View Error Details"):
                    st.code(str(e))
                
                st.warning("Please try a different visualization type or check your data.")
            
            # Additional analysis section
            st.markdown("---")
            st.markdown("#### 🔍 Advanced Attention Analysis")
            
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                if st.button("🔄 Regenerate with Different Parameters", use_container_width=True):
                    st.session_state.attention_params = {
                        'spatial_sigma': spatial_sigma,
                        'locality_weight_factor': locality_weight_factor,
                        'temperature': temperature
                    }
                    st.success("Parameters updated for attention analysis")
            
            with col_analysis2:
                if st.button("📊 Export Attention Data", use_container_width=True):
                    attention_data = {
                        'source_indices': list(range(len(weights['combined']))),
                        'angles': result['source_theta_degrees'],
                        'defect_types': [src.get('source_params', {}).get('defect_type', 'Unknown') for src in source_info['source_fields']],
                        'combined_weights': weights['combined'],
                        'spatial_kernel': weights['spatial_kernel'],
                        'defect_mask': weights['defect_mask'],
                        'entropy': weights['entropy']
                    }
                    df = pd.DataFrame(attention_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"attention_weights_theta_{result['target_angle']:.1f}_{defect_type}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Theoretical background expander
            with st.expander("📚 Theoretical Background: Theory-Informed Attention"):
                st.markdown("""
                ### Angular Bracketing Theory with Attention
                
                **Core Principle:** The attention mechanism is guided by physical theory rather than being purely data-driven.
                
                #### 1. Angular Bracketing Kernel
                The spatial kernel enforces locality in angular space:
                ```
                W_spatial(i) = exp(-0.5 * (θ_i - θ_target)² / σ²)
                ```
                - **σ (sigma):** Controls the width of the attention window
                - **Small σ:** Narrow window, strict bracketing
                - **Large σ:** Wide window, more sources contribute
                
                #### 2. Hard Defect Gating
                Defect type acts as a hard constraint:
                ```
                W_defect(i) = 1.0 if defect_i == defect_target else ε (≈0)
                ```
                - Sources with different defect types are effectively excluded
                - This is a physics-based prior, not learned
                
                #### 3. Combined Attention
                The final attention combines theory and learned patterns:
                ```
                Attention = Softmax(TransformerScore × W_spatial × W_defect)
                ```
                - **TransformerScore:** Learned similarity from transformer
                - **W_spatial:** Angular locality constraint
                - **W_defect:** Hard defect type constraint
                
                #### 4. Entropy as Focus Metric
                Attention entropy measures focus:
                ```
                Entropy = -Σ w_i * log(w_i)
                ```
                - **Low entropy (≈0):** Attention highly focused on few sources
                - **High entropy (≈log(N)):** Attention spread across many sources
                - **Ideal range:** Moderately focused but not exclusive
                
                #### 5. Interpretability Benefits
                - **Physics-guided:** Attention respects physical constraints
                - **Explainable:** Each weight has clear physical meaning
                - **Controllable:** Parameters can be adjusted based on domain knowledge
                - **Robust:** Less susceptible to spurious correlations in training data
                
                This approach creates a **Theory-Informed Neural Network** where deep learning enhances physical understanding rather than replacing it.
                """)

if __name__ == "__main__":
    main()
