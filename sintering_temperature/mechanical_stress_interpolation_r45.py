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
        """Get correct eigenstrain value for defect type"""
        return PhysicsParameters.EIGENSTRAIN_VALUES.get(defect_type, 0.0)
    
    @staticmethod
    def get_theoretical_info(defect_type: str) -> Dict:
        """Get theoretical basis information for defect type"""
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
            pass
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
# ENHANCED TRANSFORMER SPATIAL INTERPOLATOR
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
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(15, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def set_spatial_parameters(self, spatial_sigma=None, locality_weight_factor=None):
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
        if locality_weight_factor is not None:
            self.locality_weight_factor = locality_weight_factor

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
    
    def visualize_angular_kernel(self, target_angle_deg=54.7, figsize=(12, 8), 
                               xlabel="Source Index", ylabel="Spatial Weight"):
        angles = np.linspace(0, 180, 361)
        weights = []
        dummy_sources = [{'theta': np.radians(a), 'defect_type': 'Twin'} for a in angles]
        dummy_target = {'theta': np.radians(target_angle_deg), 'defect_type': 'Twin'}
        spatial_weights, _, _ = self.compute_angular_bracketing_kernel(dummy_sources, dummy_target)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(angles, spatial_weights, 'b-', linewidth=3, label='Bracketing Kernel Weight')
        ax.axvline(x=target_angle_deg, color='r', linestyle='--', linewidth=2, label=f'Target: {target_angle_deg}°')
        ax.axvline(x=54.7, color='g', linestyle='-.', linewidth=2, label='Habit Plane: 54.7°')
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(f'Angular Bracketing Regularization Kernel\nSigma: {self.spatial_sigma}°', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        return fig
    
    def encode_parameters(self, params_list, target_angle_deg):
        encoded = []
        for params in params_list:
            features = []
            features.append(params.get('eps0', 0.707) / 3.0)
            features.append(params.get('kappa', 0.6) / 2.0)
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)
            features.append(np.exp(-angle_diff / 45.0))
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))
            habit_distance = abs(theta_deg - 54.7)
            habit_distance = min(habit_distance, 360 - habit_distance)
            features.append(np.exp(-habit_distance / 15.0))
            if len(features) != 15:
                pass
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
                        vm = stress_fields['von_mises'] if 'von_mises' in stress_fields else np.zeros((100, 100))
                        hydro = stress_fields['sigma_hydro'] if 'sigma_hydro' in stress_fields else np.zeros((100, 100))
                        mag = stress_fields['sigma_mag'] if 'sigma_mag' in stress_fields else np.sqrt(vm**2 + hydro**2)
                        source_fields.append({'von_mises': vm, 'sigma_hydro': hydro, 'sigma_mag': mag, 
                                            'source_index': i, 'source_params': src['params']})
            
            if not source_params or not source_fields:
                return None
            
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
                        elif key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            resized[key] = field
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                source_fields = resized_fields
            
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            if source_features.shape[1] < 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )
            
            batch_size = 1
            seq_len = len(source_features) + 1
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            proj_features = self.input_proj(all_features)
            proj_features = self.pos_encoder(proj_features)
            transformer_output = self.transformer(proj_features)
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
            
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            spatial_kernel_tensor = torch.FloatTensor(spatial_kernel).unsqueeze(0)
            defect_mask_tensor = torch.FloatTensor(defect_mask).unsqueeze(0)
            biased_scores = attn_scores * spatial_kernel_tensor * defect_mask_tensor
            final_attention_weights = torch.softmax(biased_scores, dim=-1).squeeze().detach().cpu().numpy()
            entropy_final = -np.sum(final_attention_weights * np.log(final_attention_weights + 1e-10))
            
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += final_attention_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            if 'sigma_hydro' in interpolated_fields:
                sigma_hydro = interpolated_fields['sigma_hydro']
                D_ratio = DiffusionPhysics.compute_diffusion_enhancement(sigma_hydro, T_K=650, material='Silver', model='physics_corrected')
                interpolated_fields['diffusion_ratio'] = D_ratio
            
            source_theta_degrees = [np.degrees(src.get('theta', 0.0)) % 360 for src in source_params]
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist(),
                    'entropy': entropy_final
                },
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'source_theta_degrees': source_theta_degrees,
                'source_distances': angular_distances,
                'source_fields': source_fields
            }
        except Exception as e:
            return None

# =============================================
# ENHANCED HEAT MAP VISUALIZER WITH CUSTOMIZATION
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with diffusion visualization capabilities and full label customization"""
    
    def __init__(self):
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
            'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                                   'RdBu', 'RdBu_r', 'Spectral', 'coolwarm', 'bwr', 
                                   'seismic', 'BrBG']
        }
        self.diffusion_physics = DiffusionPhysics()
    
    def _get_label(self, custom, default):
        """Helper to return custom label if present, else default."""
        return custom if custom else default
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                            cmap_name='viridis', figsize=(12, 10),
                            colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                            show_stats=True, target_angle=None, defect_type=None,
                            show_colorbar=True, aspect_ratio='equal', dpi=300,
                            xlabel="X Position", ylabel="Y Position"):
        """Generates heatmap with customizable labels."""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if cmap_name not in plt.colormaps():
            cmap = plt.get_cmap('viridis')
        else:
            cmap = plt.get_cmap(cmap_name)
        if vmin is None: vmin = np.nanmin(stress_field)
        if vmax is None: vmax = np.nanmax(stress_field)
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect=aspect_ratio, interpolation='bilinear', origin='lower')
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label, fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\nMin: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa\nStd: {np.nanstd(stress_field):.3f} GPa")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig
    
    def create_diffusion_heatmap(self, sigma_hydro_field, title="Diffusion Enhancement Map",
                               T_K=650, material='Silver', cmap_name='RdBu_r',
                               figsize=(12, 10), dpi=300, log_scale=True,
                               show_stats=True, target_angle=None, defect_type=None,
                               show_colorbar=True, aspect_ratio='equal',
                               model='physics_corrected',
                               xlabel="X Position", ylabel="Y Position", zlabel="log(D/D0)"):
        """Generates diffusion heatmap with customizable labels."""
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(sigma_hydro_field, T_K, material, model)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if cmap_name not in plt.colormaps():
            cmap = plt.get_cmap('RdBu_r')
        else:
            cmap = plt.get_cmap(cmap_name)
        if log_scale:
            log_data = np.log10(np.clip(D_ratio, 0.1, 10))
            vmin, vmax = -1, 1
            im = ax.imshow(log_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', origin='lower')
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(zlabel, fontsize=16, fontweight='bold')
                ticks = np.array([0.1, 0.5, 1, 2, 5, 10])
                log_ticks = np.log10(ticks)
                cbar.set_ticks(log_ticks)
                cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
                cbar.ax.tick_params(labelsize=14)
        else:
            vmin, vmax = 0.1, 10
            im = ax.imshow(D_ratio, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', origin='lower')
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(zlabel, fontsize=16, fontweight='bold')
                cbar.ax.tick_params(labelsize=14)
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, {defect_type}, T = {T_K} K"
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig, D_ratio

    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info,
                                   target_angle, defect_type, component='von_mises',
                                   cmap_name='viridis', figsize=(24, 18),
                                   ground_truth_index=None, defect_type_filter=None,
                                   xlabel="X Position", ylabel="Y Position", 
                                   clabel="Stress (GPa)", show_annotations=True):
        """
        Create comprehensive comparison dashboard with customizable labels.
        """
        if defect_type_filter:
            filtered_indices = []
            for i, src in enumerate(source_fields):
                src_defect = src.get('source_params', {}).get('defect_type', 'Unknown')
                if src_defect == defect_type_filter:
                    filtered_indices.append(i)
            filtered_ground_truth_index = None
            if ground_truth_index is not None and ground_truth_index in filtered_indices:
                try:
                    filtered_ground_truth_index = filtered_indices.index(ground_truth_index)
                except ValueError:
                    pass
            filtered_source_fields = [source_fields[i] for i in filtered_indices]
            filtered_source_info = {
                'theta_degrees': [source_info['theta_degrees'][i] for i in filtered_indices],
                'distances': [source_info['distances'][i] for i in filtered_indices],
                'weights': {
                    'combined': [source_info['weights']['combined'][i] for i in filtered_indices],
                    'spatial_kernel': [source_info['weights']['spatial_kernel'][i] for i in filtered_indices],
                    'defect_mask': [source_info['weights']['defect_mask'][i] for i in filtered_indices],
                    'entropy': source_info['weights']['entropy']
                }
            }
            source_fields = filtered_source_fields
            source_info = filtered_source_info
            ground_truth_index = filtered_ground_truth_index
            if not source_fields:
                return None

        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        all_values = [interpolated_fields[component]]
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                all_values.append(gt_field)
        if len(all_values) > 0:
            vmin = min(np.nanmin(field) for field in all_values)
            vmax = max(np.nanmax(field) for field in all_values)
        else:
            vmin, vmax = 0, 1
        if cmap_name not in plt.colormaps():
            cmap_name = 'viridis'
        
        # Plot 1: Interpolated
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(interpolated_fields[component], cmap=cmap_name,
                        vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.06)
        cbar1.set_label(clabel, fontsize=12, fontweight='bold')
        ax1.set_title(f'Interpolated Result\nθ = {target_angle:.1f}°, {defect_type}',
                     fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel(xlabel, fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=11)
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Ground Truth
        ax2 = fig.add_subplot(gs[0, 1])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                gt_theta = source_info['theta_degrees'][ground_truth_index]
                gt_distance = source_info['distances'][ground_truth_index]
                im2 = ax2.imshow(gt_field, cmap=cmap_name,
                                vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
                cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.06)
                cbar2.set_label(clabel, fontsize=12, fontweight='bold')
                ax2.set_title(f'Ground Truth\nθ = {gt_theta:.1f}° (Δ={gt_distance:.1f}°)',
                             fontsize=14, fontweight='bold', pad=10)
                ax2.set_xlabel(xlabel, fontsize=11)
                ax2.set_ylabel(ylabel, fontsize=11)
                ax2.grid(True, alpha=0.2)
            else:
                ax2.text(0.5, 0.5, 'Missing data', ha='center', va='center', fontsize=12, fontweight='bold')
                ax2.set_axis_off()
        else:
            ax2.text(0.5, 0.5, 'Select Source', ha='center', va='center', fontsize=12, fontweight='bold')
            ax2.set_axis_off()
        
        # Plot 3: Difference
        ax3 = fig.add_subplot(gs[0, 2])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                diff_field = interpolated_fields[component] - gt_field
                max_diff = np.max(np.abs(diff_field))
                im3 = ax3.imshow(diff_field, cmap='RdBu_r',
                                vmin=-max_diff, vmax=max_diff, aspect='equal',
                                interpolation='bilinear', origin='lower')
                cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.06)
                cbar3.set_label('Difference', fontsize=12, fontweight='bold')
                ax3.set_title(f'Difference\nMax Abs Error: {max_diff:.3f}',
                             fontsize=14, fontweight='bold', pad=10)
                ax3.set_xlabel(xlabel, fontsize=11)
                ax3.set_ylabel(ylabel, fontsize=11)
                ax3.grid(True, alpha=0.2)
                
                # Remove statistics annotation if show_annotations is False
                if show_annotations:
                    mse = np.mean(diff_field**2)
                    mae = np.mean(np.abs(diff_field))
                    error_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}"
                    ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes,
                            fontsize=10, fontweight='bold', verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))
        
        # Plot 4: Weights
        ax4 = fig.add_subplot(gs[1, 0])
        if 'weights' in source_info:
            final_weights = source_info['weights']['combined']
            x = range(len(final_weights))
            bars = ax4.bar(x, final_weights, alpha=0.7, color='steelblue', edgecolor='black')
            ax4.set_xlabel('Source Index', fontsize=11)
            ax4.set_ylabel('Weight', fontsize=11)
            ax4.set_title('Attention Weights', fontsize=14, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3)
            if ground_truth_index is not None and 0 <= ground_truth_index < len(bars):
                bars[ground_truth_index].set_color('red')

        # Plot 5: Angular Distribution
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')
        if 'theta_degrees' in source_info:
            angles_rad = np.radians(source_info['theta_degrees'])
            weights = source_info['weights']['combined']
            sizes = 100 * np.array(weights) / (np.max(weights) + 1e-8)
            ax5.scatter(angles_rad, [0]*len(angles_rad), s=sizes, alpha=0.7, c='blue')
            ax5.set_title('Angular Distribution', fontsize=14, fontweight='bold', pad=15)
            ax5.set_theta_zero_location('N')
        
        # Plot 6: Statistics (Bar)
        ax6 = fig.add_subplot(gs[1, 2])
        # Remove default labels if custom ones are empty or specific
        ax6.set_xlabel('Component', fontsize=11)
        ax6.set_ylabel('Value (GPa)', fontsize=11)
        ax6.set_title('Statistics', fontsize=14, fontweight='bold', pad=10)
        ax6.grid(True, alpha=0.3)

        # Plot 7: Defect Mask
        ax7 = fig.add_subplot(gs[2, 0])
        if 'weights' in source_info:
            defect_masks = source_info['weights']['defect_mask']
            ax7.bar(range(len(defect_masks)), defect_masks, color='purple', alpha=0.6)
            ax7.set_xlabel('Source Index', fontsize=11)
            ax7.set_ylabel('Gating Weight', fontsize=11)
            ax7.set_title('Defect Filter', fontsize=14, fontweight='bold', pad=10)
        
        # Plot 8: Correlation
        ax8 = fig.add_subplot(gs[2, 1:])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                interp_flat = interpolated_fields[component].flatten()
                gt_flat = gt_field.flatten()
                ax8.scatter(gt_flat, interp_flat, alpha=0.5, s=10, c='navy')
                ax8.plot([min(gt_flat), max(gt_flat)], [min(gt_flat), max(gt_flat)], 'r--', linewidth=2)
                ax8.set_xlabel(f'Ground Truth {component}', fontsize=11)
                ax8.set_ylabel(f'Interpolated {component}', fontsize=11)
                ax8.set_title('Correlation', fontsize=14, fontweight='bold', pad=10)
                ax8.grid(True, alpha=0.3)
                if show_annotations:
                    from scipy.stats import pearsonr
                    try:
                        corr_coef, _ = pearsonr(gt_flat, interp_flat)
                        ax8.text(0.05, 0.95, f"Pearson: {corr_coef:.3f}", transform=ax8.transAxes,
                                fontsize=10, fontweight='bold', verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))
                    except:
                        pass

        plt.suptitle(f'Theory-Informed Interpolation: Target θ={target_angle:.1f}°, {defect_type}',
                    fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        return fig
    
    def create_interactive_heatmap(self, stress_field, title="Stress Heat Map",
                                 cmap_name='viridis', width=800, height=700,
                                 target_angle=None, defect_type=None,
                                 xlabel="X Position", ylabel="Y Position"):
        """Interactive Plotly heatmap with custom labels."""
        try:
            if cmap_name not in px.colors.named_colorscales():
                cmap_name = 'viridis'
            hover_text = []
            for i in range(stress_field.shape[0]):
                row_text = []
                for j in range(stress_field.shape[1]):
                    row_text.append(f"Position: ({i}, {j})<br>Stress: {stress_field[i, j]:.4f}")
                hover_text.append(row_text)
            heatmap_trace = go.Heatmap(
                z=stress_field, colorscale=cmap_name,
                zmin=np.nanmin(stress_field), zmax=np.nanmax(stress_field),
                hoverinfo='text', text=hover_text,
                colorbar=dict(title=dict(text="Stress (GPa)", font=dict(size=16)), 
                            tickfont=dict(size=14), thickness=20, len=0.8)
            )
            fig = go.Figure(data=[heatmap_trace])
            fig.update_layout(
                title=dict(text=title, font=dict(size=24), x=0.5, y=0.95),
                width=width, height=height,
                xaxis=dict(title=dict(text=xlabel, font=dict(size=18))),
                yaxis=dict(title=dict(text=ylabel, font=dict(size=18))),
                hovermode='closest', plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=80, r=80, t=100, b=80)
            )
            return fig
        except Exception as e:
            return None

# =============================================
# RESULTS MANAGER FOR EXPORT
# =============================================
class ResultsManager:
    def __init__(self):
        pass
    def prepare_export_data(self, interpolation_result, visualization_params):
        result = interpolation_result.copy()
        return {
            'metadata': {'generated_at': datetime.now().isoformat(), 'params': visualization_params},
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'fields': {k: v.tolist() for k, v in result['fields'].items()}
            }
        }
    def export_to_json(self, export_data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"bracketing_interpolation_{timestamp}.json"
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    def _json_serializer(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, datetime): return obj.isoformat()
        if isinstance(obj, torch.Tensor): return obj.cpu().numpy().tolist()
        return str(obj)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Angular Bracketing Theory with Transformer Attention",
        layout="wide", page_icon="🎯", initial_sidebar_state="expanded"
    )
    
    # CSS Styling
    st.markdown("""
    <style>
    .main-header { font-size: 3.2rem !important; color: #1E3A8A !important; text-align: center; padding: 1rem;
                  background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900 !important; margin-bottom: 1rem; }
    .section-header { font-size: 2.0rem !important; color: #374151 !important; font-weight: 800 !important;
                     border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    .info-box { background-color: #F0F9FF; border-left: 5px solid #3B82F6; padding: 1.2rem;
                 border-radius: 0.6rem; margin: 1.2rem 0; font-size: 1.1rem; }
    .physics-note { background-color: #FFF3CD; border-left: 5px solid #FFC107; padding: 1rem;
                   border-radius: 0.5rem; margin: 1rem 0; font-size: 1rem; }
    .diffusion-box { background-color: #E8F5E9; border-left: 5px solid #4CAF50; padding: 1rem;
                    border-radius: 0.5rem; margin: 1rem 0; font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎯 Angular Bracketing Theory with Transformer Attention</h1>', unsafe_allow_html=True)
    
    # Initialize Session State
    defaults = {
        'solutions': [], 'loader': None, 'interpolator': None, 'visualizer': None,
        'manager': None, 'interpolation_result': None, 'selected_ground_truth': None,
        # Customization Defaults
        'custom_xlabel': "X Position",
        'custom_ylabel': "Y Position",
        'custom_zlabel': "Stress (GPa)",
        'custom_diffusion_label': "log(D/D0)",
        'show_annotations': True,
        'show_grid': True
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Customization Section
        st.markdown("#### 🖌️ Label & Style Customization")
        with st.expander("Edit Labels & Legends"):
            st.session_state.custom_xlabel = st.text_input("X Axis Label", value=st.session_state.custom_xlabel)
            st.session_state.custom_ylabel = st.text_input("Y Axis Label", value=st.session_state.custom_ylabel)
            st.session_state.custom_zlabel = st.text_input("Z/Colorbar Label (Default Stress)", value=st.session_state.custom_zlabel)
            st.session_state.custom_diffusion_label = st.text_input("Diffusion Label", value=st.session_state.custom_diffusion_label)
            st.session_state.show_annotations = st.checkbox("Show Plot Annotations (Text/Markers)", value=st.session_state.show_annotations)
            st.session_state.show_grid = st.checkbox("Show Grid Lines", value=st.session_state.show_grid)

        # Physics Configuration
        st.markdown("#### 🎯 Target Parameters")
        custom_theta = st.number_input("Target Angle θ (degrees)", min_value=0.0, max_value=180.0, value=54.7, step=0.1)
        defect_type = st.selectbox("Defect Type", options=['ISF', 'ESF', 'Twin', 'No Defect'], index=2)
        shape = st.selectbox("Shape", options=['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle'], index=0)
        eigen_strain = st.slider("Eigenstrain ε₀", min_value=0.0, max_value=3.0, value=2.12 if defect_type == 'Twin' else 0.707, step=0.001)
        
        # Diffusion
        st.markdown("#### 🌡️ Diffusion Physics")
        diffusion_material = st.selectbox("Material", options=['Silver', 'Copper', 'Aluminum', 'Nickel', 'Iron'], index=0)
        diffusion_T = st.slider("Temperature (K)", min_value=300, max_value=1500, value=650, step=10)
        diffusion_model = st.selectbox("Diffusion Model", options=['physics_corrected', 'temperature_reduction', 'activation_energy', 'vacancy_concentration'], index=0)
        
        # Transformer Theory
        st.markdown("#### 🧠 Angular Bracketing Theory")
        spatial_sigma = st.slider("Angular Kernel Sigma", min_value=1.0, max_value=45.0, value=10.0, step=0.5)
        temperature = st.slider("Attention Temperature", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        # Controls
        st.markdown("#### 🚀 Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Solutions"):
                st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
        with col2:
            if st.button("Run Interpolation"):
                if not st.session_state.solutions:
                    st.error("Load solutions first")
                else:
                    st.session_state.interpolator = TransformerSpatialInterpolator(spatial_sigma=spatial_sigma)
                    target_params = {'defect_type': defect_type, 'eps0': eigen_strain, 'kappa': 0.6, 'theta': np.radians(custom_theta), 'shape': shape}
                    st.session_state.interpolation_result = st.session_state.interpolator.interpolate_spatial_fields(
                        st.session_state.solutions, custom_theta, target_params
                    )
                    if st.session_state.interpolation_result:
                        st.success("Interpolation complete")

    # Main Content
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Results", "🎨 Stress Visualization", "🌡️ Diffusion", "⚖️ Attention", "🔄 Comparison"
        ])
        
        # Prepare custom labels dictionary for ease of passing
        custom_labels = {
            'xlabel': st.session_state.custom_xlabel,
            'ylabel': st.session_state.custom_ylabel,
            'zlabel': st.session_state.custom_zlabel,
            'diff_label': st.session_state.custom_diffusion_label,
            'annotations': st.session_state.show_annotations,
            'grid': st.session_state.show_grid
        }
        
        # Helper to init visualizer if needed
        if st.session_state.visualizer is None:
            st.session_state.visualizer = HeatMapVisualizer()
        visualizer = st.session_state.visualizer

        with tab1:
            st.markdown('<h2 class="section-header">📊 Results Overview</h2>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Max Stress", f"{np.max(result['fields']['von_mises']):.3f}")
            with col2: st.metric("Target Angle", f"{result['target_angle']:.1f}°")
            with col3: st.metric("Defect Type", result['target_params']['defect_type'])
            
            st.markdown("### Quick Preview")
            comp = st.selectbox("Preview Component", ['von_mises', 'sigma_hydro', 'diffusion_ratio'], index=0)
            if comp == 'diffusion_ratio':
                fig, _ = visualizer.create_diffusion_heatmap(
                    result['fields']['sigma_hydro'], cmap_name='RdBu_r',
                    xlabel=custom_labels['xlabel'], ylabel=custom_labels['ylabel'],
                    zlabel=custom_labels['diff_label'], show_annotations=custom_labels['annotations']
                )
            else:
                fig = visualizer.create_stress_heatmap(
                    result['fields'][comp], xlabel=custom_labels['xlabel'], ylabel=custom_labels['ylabel'],
                    colorbar_label=custom_labels['zlabel'], show_annotations=custom_labels['annotations']
                )
            st.pyplot(fig)

        with tab2:
            st.markdown('<h2 class="section-header">🎨 Stress Visualization</h2>', unsafe_allow_html=True)
            cmap = st.selectbox("Colormap", list(visualizer.colormaps['Publication Standard']), index=0)
            fig = visualizer.create_stress_heatmap(
                result['fields']['von_mises'], cmap_name=cmap,
                xlabel=custom_labels['xlabel'], ylabel=custom_labels['ylabel'],
                colorbar_label=custom_labels['zlabel'], show_annotations=custom_labels['annotations']
            )
            st.pyplot(fig)
            
            st.markdown("### Interactive")
            fig_int = visualizer.create_interactive_heatmap(
                result['fields']['von_mises'], xlabel=custom_labels['xlabel'], ylabel=custom_labels['ylabel']
            )
            st.plotly_chart(fig_int, use_container_width=True)

        with tab3:
            st.markdown('<h2 class="section-header">🌡️ Diffusion Visualization</h2>', unsafe_allow_html=True)
            fig_diff, _ = visualizer.create_diffusion_heatmap(
                result['fields']['sigma_hydro'], cmap_name='RdBu_r',
                xlabel=custom_labels['xlabel'], ylabel=custom_labels['ylabel'],
                zlabel=custom_labels['diff_label'], show_annotations=custom_labels['annotations']
            )
            st.pyplot(fig_diff)

        with tab4:
            st.markdown('<h2 class="section-header">⚖️ Attention Analysis</h2>', unsafe_allow_html=True)
            st.info(f"Angular Sigma: {st.session_state.interpolator.spatial_sigma}°")
            fig_kern = st.session_state.interpolator.visualize_angular_kernel(
                result['target_angle'], xlabel="Angle (deg)", ylabel="Weight"
            )
            st.pyplot(fig_kern)

        with tab5:
            st.markdown('<h2 class="section-header">🔄 Comparison Dashboard</h2>', unsafe_allow_html=True)
            
            # Source Selection for Dashboard
            st.markdown("#### Select Ground Truth")
            if 'source_theta_degrees' in result:
                options = [f"Source {i} (θ={t:.1f}°)" for i, t in enumerate(result['source_theta_degrees'])]
                idx_sel = st.selectbox("Source", options=options, format_func=lambda x: x.split(" (")[0])
                match = re.search(r'\d+', idx_sel)
                idx_val = int(match.group()) if match else 0
            else:
                idx_val = None
            
            fig_dash = visualizer.create_comparison_dashboard(
                interpolated_fields=result['fields'],
                source_fields=result['source_fields'],
                source_info={
                    'theta_degrees': result['source_theta_degrees'],
                    'distances': result['source_distances'],
                    'weights': result['weights']
                },
                target_angle=result['target_angle'],
                defect_type=result['target_params']['defect_type'],
                ground_truth_index=idx_val,
                xlabel=custom_labels['xlabel'],
                ylabel=custom_labels['ylabel'],
                clabel=custom_labels['zlabel'],
                show_annotations=custom_labels['annotations']
            )
            st.pyplot(fig_dash)

if __name__ == "__main__":
    main()
