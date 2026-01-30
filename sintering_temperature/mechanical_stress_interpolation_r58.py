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
# DOMAIN CONFIGURATION - 12.8 nm × 12.8 nm
# =============================================
class DomainConfiguration:
    """Configuration for the 12.8 nm × 12.8 nm simulation domain"""
    N = 128
    dx = 0.1
    DOMAIN_LENGTH = N * dx
    DOMAIN_HALF = DOMAIN_LENGTH / 2.0

    @classmethod
    def get_extent(cls):
        return [-cls.DOMAIN_HALF, cls.DOMAIN_HALF, -cls.DOMAIN_HALF, cls.DOMAIN_HALF]

    @classmethod
    def get_coordinates(cls):
        x = np.linspace(-cls.DOMAIN_HALF, cls.DOMAIN_HALF, cls.N, endpoint=False)
        y = np.linspace(-cls.DOMAIN_HALF, cls.DOMAIN_HALF, cls.N, endpoint=False)
        return np.meshgrid(x, y)

    @classmethod
    def get_domain_info(cls):
        return {
            'grid_points': cls.N,
            'grid_spacing_nm': cls.dx,
            'domain_length_nm': cls.DOMAIN_LENGTH,
            'domain_half_nm': cls.DOMAIN_HALF,
            'area_nm2': cls.DOMAIN_LENGTH ** 2,
            'extent': cls.get_extent(),
            'description': f"Square domain of {cls.DOMAIN_LENGTH} nm × {cls.DOMAIN_LENGTH} nm centered at origin"
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
    """
    Transformer-inspired interpolator with Enhanced Spatial Locality Regularization
    based on Angular Bracketing Theory.
    """
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
            
    def compute_angular_bracketing_kernel(self, source_params, target_params):
        """
        Compute the Angular Bracketing Kernel and Defect Mask.
        """
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
                
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                target_shape = shapes[0]
                for fields in source_fields:
                    for key, field in fields.items():
                        if key in ['von_mises', 'sigma_hydro', 'sigma_mag'] and field.shape != target_shape:
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            fields[key] = zoom(field, factors, order=1)
            
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            if source_features.shape[1] != 15 or target_features.shape[1] != 15:
                if source_features.shape[1] < 15:
                    padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                    source_features = torch.cat([source_features, padding], dim=1)
                if target_features.shape[1] < 15:
                    padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                    target_features = torch.cat([target_features, padding], dim=1)
            
            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )
            
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
            
            pre_mask_scores = attn_scores * spatial_kernel_tensor
            pre_mask_weights = torch.softmax(pre_mask_scores, dim=-1).squeeze().detach().cpu().numpy()
            post_mask_scores = pre_mask_scores * defect_mask_tensor
            post_mask_weights = torch.softmax(post_mask_scores, dim=-1).squeeze().detach().cpu().numpy()
            entropy_final = self._calculate_entropy(post_mask_weights)
            
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += post_mask_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            source_theta_degrees = [np.degrees(src.get('theta', 0.0)) % 360 for src in source_params]
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': post_mask_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist(),
                    'pre_mask': pre_mask_weights.tolist(),
                    'entropy': entropy_final
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
            return (stress_fields['sigma_xx'] + stress_fields['sigma_yy'] + 
                   stress_fields.get('sigma_zz', np.zeros_like(stress_fields['sigma_xx']))) / 3
        return np.zeros((100, 100))
        
    def _calculate_entropy(self, weights):
        weights = np.array(weights)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return -np.sum(weights * np.log(weights + 1e-10))

# =============================================
# ENHANCED HEAT MAP VISUALIZER WITH RELATIONSHIP VISUALIZATIONS
# =============================================
class HeatMapVisualizer:
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.diffusion_physics = DiffusionPhysics()

    # ==================== HIERARCHICAL RADAR CHARTS: TIER 1 (ANGULAR) & TIER 2 (DEFECT TYPE) ====================
    
    def create_hierarchical_polar_chart(self, weights, source_info, target_params, 
                                       include_defect_mask=True, figsize=(10, 10)):
        """
        Hierarchical polar chart with Angular Variation as Tier 1 (angular axis) 
        and Defect Type as Tier 2 (ring/radius encoding).
        
        If include_defect_mask=False: Shows pre-mask weights (Learned + Spatial)
        If include_defect_mask=True: Shows post-mask weights (Final Combined)
        """
        angles_deg = np.array(source_info['theta_degrees'])
        target_angle = target_params['theta']
        target_defect = target_params['defect_type']
        
        if include_defect_mask:
            weights_to_plot = np.array(weights['combined'])
            title_suffix = "Post-Mask (Final Combined Weights)"
            color_scale = 'Reds'
        else:
            weights_to_plot = np.array(weights['pre_mask'])
            title_suffix = "Pre-Mask (Learned + Spatial Only)"
            color_scale = 'Blues'
        
        fig = go.Figure()
        
        # Create concentric rings for angular bins (Tier 1 hierarchy)
        angular_bins = np.arange(0, 361, 30)  # 30-degree sectors
        max_weight = np.max(weights_to_plot) * 1.2
        
        # Add angular sector background rings (Tier 1: Angular Variation)
        for i in range(len(angular_bins)-1):
            bin_center = (angular_bins[i] + angular_bins[i+1]) / 2
            bin_width = angular_bins[i+1] - angular_bins[i]
            
            # Determine if this sector contains the target
            is_target_sector = (target_angle >= angular_bins[i] and target_angle < angular_bins[i+1])
            sector_color = 'rgba(255,0,0,0.1)' if is_target_sector else 'rgba(200,200,200,0.05)'
            
            fig.add_trace(go.Barpolar(
                r=[max_weight],
                theta=[bin_center],
                width=[bin_width],
                marker_color=sector_color,
                opacity=0.3,
                showlegend=False,
                hoverinfo='text',
                text=f"Sector {angular_bins[i]}-{angular_bins[i+1]}°"
            ))
        
        # Group sources by defect type (Tier 2 hierarchy)
        source_params = source_info.get('source_fields', [])
        defect_types = []
        for src in source_params:
            dt = src.get('source_params', {}).get('defect_type', 'Unknown')
            defect_types.append(dt)
        
        unique_defects = list(set(defect_types))
        colors = px.colors.qualitative.Set1[:len(unique_defects)]
        defect_color_map = {dt: colors[i] for i, dt in enumerate(unique_defects)}
        
        # Plot sources grouped by defect type
        for defect_type in unique_defects:
            mask = np.array(defect_types) == defect_type
            subset_angles = angles_deg[mask]
            subset_weights = weights_to_plot[mask]
            subset_indices = np.where(mask)[0]
            
            # Adjust radius slightly for each defect type to create concentric grouping
            radius_offset = 0.02 * (unique_defects.index(defect_type) + 1)
            
            fig.add_trace(go.Scatterpolar(
                r=subset_weights + radius_offset,
                theta=subset_angles,
                mode='markers+text',
                name=f"{defect_type}",
                marker=dict(
                    size=12 + 8 * (subset_weights / np.max(weights_to_plot)),
                    color=defect_color_map[defect_type],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[f"S{i}<br>{w:.3f}" for i, w in zip(subset_indices, subset_weights)],
                textposition='top center',
                hovertemplate='<b>Source %{text}</b><br>Angle: %{theta:.1f}°<br>Weight: %{r:.4f}<extra></extra>'
            ))
        
        # Highlight Target with prominent line and marker
        fig.add_trace(go.Scatterpolar(
            r=[0, max_weight * 1.1],
            theta=[target_angle, target_angle],
            mode='lines+markers',
            name=f'TARGET: {target_defect} @ {target_angle:.1f}°',
            line=dict(color='red', width=4, dash='solid'),
            marker=dict(size=15, symbol='star', color='red', line=dict(width=2, color='darkred'))
        ))
        
        # Update layout with hierarchical title
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_weight * 1.15],
                    title='Weight Magnitude',
                    tickfont=dict(size=11)
                ),
                angularaxis=dict(
                    rotation=90,
                    direction='clockwise',
                    tickfont=dict(size=12),
                    title='Angular Position (Tier 1)'
                ),
                bgcolor='rgba(240,240,240,0.5)'
            ),
            title=dict(
                text=f"Hierarchical Polar Chart: {title_suffix}<br>"
                     f"<sub>Tier 1: Angular Variation | Tier 2: Defect Type (Color) | Red Star: Target</sub>",
                font=dict(size=18, color='#1E3A8A'),
                x=0.5
            ),
            width=900,
            height=800,
            showlegend=True,
            legend=dict(
                title=dict(text='Defect Type (Tier 2)'),
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add annotation explaining the hierarchy
        fig.add_annotation(
            text="Hierarchy: Angular Position → Defect Type → Weight Magnitude",
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            showarrow=False,
            font=dict(size=12, color="#666"),
            bgcolor="white",
            opacity=0.8
        )
        
        return fig

    def create_comparison_radar_charts(self, weights, source_info, target_params, figsize=(12, 6)):
        """
        Two radar charts side by side:
        1. Pre-Mask: Attention + Angular (without defect mask)
        2. Post-Mask: Attention + Angular + Defect Mask
        """
        source_params = source_info.get('source_fields', [])
        if not source_params:
            return None
        
        dimensions = ['Learned Attention', 'Spatial Kernel', 'Combined Weight', 'Angular Proximity']
        
        # Prepare data for pre-mask (without defect mask effect)
        pre_mask_data = []
        # Prepare data for post-mask (with defect mask)
        post_mask_data = []
        
        source_labels = []
        
        for i, src in enumerate(source_params):
            src_theta = source_info['theta_degrees'][i]
            target_theta = target_params['theta']
            
            angle_diff = abs(src_theta - target_theta)
            angle_diff = min(angle_diff, 360 - angle_diff)
            angular_proximity = np.exp(-angle_diff / 30.0)
            
            learned_att = weights['pre_mask'][i] if weights['pre_mask'][i] > 0 else 0.001
            spatial = weights['spatial_kernel'][i]
            pre_combined = weights['pre_mask'][i]
            post_combined = weights['combined'][i]
            
            pre_mask_data.append([learned_att, spatial, pre_combined, angular_proximity])
            post_mask_data.append([learned_att, spatial, post_combined, angular_proximity])
            
            defect_type = src.get('source_params', {}).get('defect_type', 'Unknown')
            source_labels.append(f"S{i}: {defect_type}")
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "polar"}, {"type": "polar"}]],
            subplot_titles=(
                'Pre-Mask: Attention + Angular<br>(Without Defect Mask)', 
                'Post-Mask: Attention + Angular + Defect<br>(With Defect Mask Applied)'
            ),
            horizontal_spacing=0.15
        )
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(pre_mask_data)))
        
        for i, (pre_data, post_data) in enumerate(zip(pre_mask_data, post_mask_data)):
            # Pre-mask radar (left)
            pre_values = pre_data + [pre_data[0]]
            fig.add_trace(go.Scatterpolar(
                r=pre_values,
                theta=dimensions + [dimensions[0]],
                fill='toself',
                name=source_labels[i],
                line=dict(color=f'rgba({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)},0.6)', width=2),
                fillcolor=f'rgba({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)},0.1)',
                showlegend=(i==0),
                legendgroup=source_labels[i]
            ), row=1, col=1)
            
            # Post-mask radar (right) - different line style for masked sources
            post_values = post_data + [post_data[0]]
            line_dash = 'solid' if weights['defect_mask'][i] > 0.5 else 'dot'
            line_width = 3 if weights['defect_mask'][i] > 0.5 else 1
            
            fig.add_trace(go.Scatterpolar(
                r=post_values,
                theta=dimensions + [dimensions[0]],
                fill='toself',
                name=source_labels[i] + (" (excluded)" if weights['defect_mask'][i] < 0.5 else ""),
                line=dict(
                    color=f'rgba({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)},0.8)', 
                    width=line_width,
                    dash=line_dash
                ),
                fillcolor=f'rgba({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)},0.15)',
                showlegend=False,
                legendgroup=source_labels[i]
            ), row=1, col=2)
        
        # Add target reference line to both
        for col in [1, 2]:
            fig.add_trace(go.Scatterpolar(
                r=[0, 1],
                theta=[target_params['theta'], target_params['theta']],
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=10, color='red'),
                name='Target Angle',
                showlegend=(col==1)
            ), row=1, col=col)
        
        fig.update_layout(
            height=600,
            width=1200,
            title=dict(
                text=f"Weight Component Analysis: Effect of Defect Mask<br>"
                     f"<sub>Target: {target_params['defect_type']} @ {target_params['theta']:.1f}° | "
                     f"Solid lines = Same defect type (included), Dotted = Different (masked out)</sub>",
                font=dict(size=16, color='#1E3A8A'),
                x=0.5
            ),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            polar2=dict(radialaxis=dict(visible=True, range=[0, 1]))
        )
        
        return fig

    # ==================== SANKEY DIAGRAM FOR WEIGHT FLOW ====================
    
    def create_attention_sankey_diagram(self, weights, source_info, target_params, 
                                       spatial_sigma, domain_size_nm=12.8):
        """
        Sankey diagram showing flow from Sources through Weight Components to Target.
        Left: Sources (arranged by angular proximity)
        Middle: Weight Components (Learned, Spatial, Defect Mask)
        Right: Target
        """
        source_params = source_info.get('source_fields', [])
        n_sources = len(source_params)
        
        if n_sources == 0:
            return None
        
        labels = []
        source_indices = []
        target_indices = []
        values = []
        link_colors = []
        
        # Left layer: Sources (sorted by angular proximity to target)
        angles = np.array(source_info['theta_degrees'])
        target_angle = target_params['theta']
        distances = np.abs(angles - target_angle)
        distances = np.minimum(distances, 360 - distances)
        sorted_indices = np.argsort(distances)
        
        source_node_ids = {}
        for i, idx in enumerate(sorted_indices):
            src = source_params[idx]
            defect_type = src.get('source_params', {}).get('defect_type', 'Unknown')
            angle = angles[idx]
            labels.append(f"Source {idx}<br>{defect_type}<br>{angle:.1f}°")
            source_node_ids[idx] = i
        
        # Middle layer: Weight Components
        learned_idx = n_sources
        spatial_idx = n_sources + 1
        defect_idx = n_sources + 2
        
        labels.extend([
            "Learned<br>Attention (ᾱ)",
            f"Spatial Kernel<br>(σ={spatial_sigma}°)",
            "Defect Mask<br>(𝟙(τᵢ=τ*))"
        ])
        
        # Target node
        target_idx = n_sources + 3
        labels.append(f"TARGET<br>{target_params['defect_type']}<br>{target_angle:.1f}°")
        
        # Create flows
        # Source -> Components (flows proportional to each component's contribution)
        for idx in sorted_indices:
            weight_pre = weights['pre_mask'][idx]
            weight_post = weights['combined'][idx]
            spatial_w = weights['spatial_kernel'][idx]
            defect_w = weights['defect_mask'][idx]
            
            # Approximate learned component (pre_mask / spatial, normalized)
            learned_w = weight_pre / (spatial_w + 1e-10)
            learned_w = min(learned_w, 1.0)  # Cap at 1 for visualization
            
            src_id = source_node_ids[idx]
            
            # Flow to Learned Attention
            source_indices.append(src_id)
            target_indices.append(learned_idx)
            values.append(learned_w * 100)  # Scale up for visibility
            link_colors.append('rgba(100, 149, 237, 0.4)')  # Cornflower blue
            
            # Flow to Spatial Kernel
            source_indices.append(src_id)
            target_indices.append(spatial_idx)
            values.append(spatial_w * 100)
            link_colors.append('rgba(60, 179, 113, 0.4)')  # Medium sea green
            
            # Flow to Defect Mask (only if matches)
            source_indices.append(src_id)
            target_indices.append(defect_idx)
            values.append(defect_w * 100)
            if defect_w > 0.5:
                link_colors.append('rgba(255, 99, 71, 0.6)')  # Tomato red if active
            else:
                link_colors.append('rgba(128, 128, 128, 0.2)')  # Gray if inactive
        
        # Components -> Target (flows represent aggregation)
        for comp_idx, comp_name, comp_color in [(learned_idx, 'learned', 'rgba(100, 149, 237, 0.6)'),
                                                 (spatial_idx, 'spatial', 'rgba(60, 179, 113, 0.6)'),
                                                 (defect_idx, 'defect', 'rgba(255, 99, 71, 0.6)')]:
            source_indices.append(comp_idx)
            target_indices.append(target_idx)
            # Sum of incoming flows as magnitude
            incoming = sum([v for s, t, v in zip(source_indices, target_indices, values) 
                          if t == comp_idx])
            values.append(incoming)
            link_colors.append(comp_color)
        
        # Node colors
        node_colors = []
        for i in range(n_sources):
            idx = sorted_indices[i]
            if weights['defect_mask'][idx] > 0.5:
                node_colors.append('rgba(255, 99, 71, 0.8)')  # Red for matching
            else:
                node_colors.append('rgba(128, 128, 128, 0.6)')  # Gray for non-matching
        node_colors.extend(['cornflowerblue', 'mediumseagreen', 'tomato', 'darkred'])
        
        fig = go.Figure(data=[go.Sankey(
            arrangement="freeform",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                x=[0.1]*n_sources + [0.5, 0.5, 0.5, 0.9],  # Position: sources left, components middle, target right
                y=list(np.linspace(0.1, 0.9, n_sources)) + [0.2, 0.5, 0.8, 0.5]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=link_colors,
                hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Value: %{value:.2f}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=f"Attention Weight Flow Sankey Diagram<br>"
                     f"<sub>Sources (Left) → Weight Components (Middle) → Target (Right)<br>"
                     f"Formula: wᵢ = [ᾱᵢ · exp(-(Δφᵢ)²/(2σ²)) · 𝟙(τᵢ=τ*)] / Σ[...]</sub>",
                font=dict(size=18, color='#1E3A8A'),
                x=0.5
            ),
            width=1200,
            height=800,
            font=dict(size=12),
            paper_bgcolor='white'
        )
        
        return fig

    # ==================== CHORD DIAGRAM REPLACEMENT (CIRCULAR NETWORK) ====================
    
    def create_chord_diagram(self, weights, source_info, target_params):
        """
        Circular network diagram (chord-style) showing connections between sources
        based on attention proximity, colored by defect type.
        """
        n_sources = len(source_info['theta_degrees'])
        if n_sources < 2:
            return None
        
        angles_deg = np.array(source_info['theta_degrees'])
        target_angle = target_params['theta']
        target_defect = target_params['defect_type']
        
        # Sort by angle for circular layout
        sorted_idx = np.argsort(angles_deg)
        sorted_angles = angles_deg[sorted_idx]
        
        # Convert to radians and arrange in circle
        theta = np.radians(sorted_angles)
        r = 1.0
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        fig = go.Figure()
        
        # Get defect types and colors
        source_params = source_info.get('source_fields', [])
        defect_types = [source_params[i].get('source_params', {}).get('defect_type', 'Unknown') 
                       for i in sorted_idx]
        unique_defects = list(set(defect_types))
        colors = px.colors.qualitative.Set1[:len(unique_defects)]
        color_map = {dt: colors[i] for i, dt in enumerate(unique_defects)}
        
        # Draw chords (connections between sources with similar attention weights)
        combined_weights = np.array(weights['combined'])[sorted_idx]
        spatial_weights = np.array(weights['spatial_kernel'])[sorted_idx]
        
        for i in range(n_sources):
            for j in range(i+1, n_sources):
                # Connection strength based on attention similarity and spatial proximity
                att_sim = combined_weights[i] * combined_weights[j]
                
                if att_sim > 0.01:  # Threshold
                    # Bezier curve control point
                    x0, y0 = x[i], y[i]
                    x1, y1 = x[j], y[j]
                    
                    # Control point inside circle, distance based on spatial weight
                    mid_x = (x0 + x1) / 2
                    mid_y = (y0 + y1) / 2
                    dist = np.sqrt(mid_x**2 + mid_y**2)
                    if dist > 0:
                        scale = 0.3 * (spatial_weights[i] + spatial_weights[j]) / 2
                        cx = mid_x * scale / dist
                        cy = mid_y * scale / dist
                    else:
                        cx, cy = 0, 0
                    
                    # Quadratic Bezier
                    t = np.linspace(0, 1, 50)
                    bx = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
                    by = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
                    
                    line_width = max(0.5, att_sim * 10)
                    line_color = color_map[defect_types[i]]
                    
                    fig.add_trace(go.Scatter(
                        x=bx, y=by,
                        mode='lines',
                        line=dict(width=line_width, color=line_color),
                        opacity=0.6,
                        hoverinfo='skip',
                        showlegend=False
                    ))
        
        # Add nodes
        node_colors = [color_map[dt] for dt in defect_types]
        node_sizes = [20 + 30 * w for w in combined_weights]
        
        # Highlight target angle with red star at appropriate position
        target_rad = np.radians(target_angle)
        target_x = 1.15 * np.cos(target_rad)
        target_y = 1.15 * np.sin(target_rad)
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=[f"{sorted_idx[i]}" for i in range(n_sources)],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertemplate='<b>Source %{text}</b><br>Angle: %{customdata[0]:.1f}°<br>Defect: %{customdata[1]}<br>Weight: %{customdata[2]:.4f}<extra></extra>',
            customdata=list(zip(sorted_angles, defect_types, combined_weights)),
            showlegend=False
        ))
        
        # Add target indicator
        fig.add_trace(go.Scatter(
            x=[0, target_x], y=[0, target_y],
            mode='lines+markers',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=[0, 20], color=['red', 'red'], symbol=['none', 'star']),
            name=f'Target: {target_defect} @ {target_angle:.1f}°'
        ))
        
        # Add legend for defect types
        for dt in unique_defects:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[dt]),
                name=dt
            ))
        
        fig.update_layout(
            title=dict(
                text=f"Chord Diagram: Source Interconnection by Attention Weight<br>"
                     f"<sub>Node size ∝ Weight | Line thickness ∝ Attention Product | Color = Defect Type</sub>",
                font=dict(size=18, color='#1E3A8A'),
                x=0.5
            ),
            xaxis=dict(visible=False, range=[-1.3, 1.3], scaleanchor='y'),
            yaxis=dict(visible=False, range=[-1.3, 1.3]),
            width=800,
            height=800,
            showlegend=True,
            legend=dict(
                title=dict(text='Defect Type'),
                orientation='h',
                yanchor='bottom',
                y=-0.1
            ),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Angular Bracketing Theory",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # FIXED SYNTAX: Using proper multi-line strings without stray backslashes
    st.markdown("""
    <style>
    .main-header { font-size: 3.2rem !important; color: #1E3A8A !important; text-align: center; padding: 1rem;
    background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-weight: 900 !important; margin-bottom: 1rem; }
    .section-header { font-size: 2.0rem !important; color: #374151 !important; font-weight: 800 !important;
    border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.8rem; margin-bottom: 1.2rem; }
    .info-box { background-color: #F0F9FF; border-left: 5px solid #3B82F6; padding: 1.2rem;
    border-radius: 0.6rem; margin: 1.2rem 0; font-size: 1.1rem; }
    .formula-box { background-color: #F0F7FF; border-left: 5px solid #3B82F6; padding: 1.5rem;
    border-radius: 0.8rem; margin: 1.5rem 0; font-family: 'Courier New', monospace; font-size: 1.1rem; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 Angular Bracketing Theory - Attention Analysis</h1>', unsafe_allow_html=True)
    
    # FIXED SYNTAX: Properly formatted formula without line continuation issues
    st.markdown(r"""
    <div class="formula-box">
    <strong>Attention Weight Formula:</strong><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
    
    <strong>Where:</strong><br>
    • $\bar{\alpha}_i$: Learned attention score from transformer<br>
    • $\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$: Spatial kernel (angular bracketing)<br>
    • $\mathbb{I}(\tau_i = \tau^*)$: Defect type indicator (1 if match, 0 otherwise)<br>
    • $\sigma$: Angular kernel width
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Session State
    if 'solutions' not in st.session_state: st.session_state.solutions = []
    if 'loader' not in st.session_state: st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator(spatial_sigma=10.0, locality_weight_factor=0.5)
    if 'heatmap_visualizer' not in st.session_state: st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'interpolation_result' not in st.session_state: st.session_state.interpolation_result = None
    
    # Sidebar: Configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
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
                st.success("Cache cleared")
        
        st.divider()
        
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        custom_theta = st.number_input("Target Angle θ (degrees)", min_value=0.0, max_value=180.0, value=54.7, step=0.1)
        defect_type = st.selectbox("Defect Type", options=['ISF', 'ESF', 'Twin', 'No Defect'], index=2)
        
        st.divider()
        
        st.markdown('<h2 class="section-header">⚛️ Theory & Attention</h2>', unsafe_allow_html=True)
        spatial_sigma = st.slider("Angular Kernel Sigma (degrees)", min_value=1.0, max_value=45.0, value=10.0, step=0.5)
        
        st.divider()
        
        if st.button("🧠 Perform Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Processing..."):
                    target_params = {
                        'defect_type': defect_type, 
                        'eps0': PhysicsParameters.get_eigenstrain(defect_type), 
                        'kappa': 0.6, 
                        'theta': np.radians(custom_theta), 
                        'shape': 'Square'
                    }
                    result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                        st.session_state.solutions, custom_theta, target_params
                    )
                    if result:
                        st.session_state.interpolation_result = result
                        st.success("Interpolation successful.")
                    else:
                        st.error("Interpolation failed.")
    
    # Main content
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        weights = result['weights']
        source_info = {
            'theta_degrees': result['source_theta_degrees'],
            'distances': result['source_distances'],
            'source_fields': result.get('source_fields', [])
        }
        target_params = {
            'theta': result['target_angle'],
            'defect_type': result['target_params']['defect_type']
        }
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Hierarchical Polar", 
            "⚖️ Pre vs Post Mask", 
            "🔀 Sankey Flow", 
            "🕸️ Chord Diagram",
            "📈 Statistics"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">Hierarchical Polar Chart</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Tier 1 (Angular):</strong> Sources arranged by angular position around the circle<br>
            <strong>Tier 2 (Defect Type):</strong> Color coding indicates defect type<br>
            <strong>Target Highlight:</strong> Red star indicates the target query angle and defect type
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Without Defect Mask (Pre-Mask)")
                fig1 = st.session_state.heatmap_visualizer.create_hierarchical_polar_chart(
                    weights, source_info, target_params, include_defect_mask=False
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.subheader("With Defect Mask (Post-Mask)")
                fig2 = st.session_state.heatmap_visualizer.create_hierarchical_polar_chart(
                    weights, source_info, target_params, include_defect_mask=True
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.markdown('<h2 class="section-header">Pre-Mask vs Post-Mask Comparison</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Left:</strong> Learned Attention + Spatial Kernel only (before defect masking)<br>
            <strong>Right:</strong> Final weights after Defect Mask application (1 if match, ~0 if different)<br>
            <strong>Note:</strong> Dotted lines indicate sources masked out due to defect type mismatch
            </div>
            """, unsafe_allow_html=True)
            
            fig_comp = st.session_state.heatmap_visualizer.create_comparison_radar_charts(
                weights, source_info, target_params
            )
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True)
        
        with tab3:
            st.markdown('<h2 class="section-header">Sankey Diagram: Weight Flow</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Visualizes how attention flows from Sources (left) through three components:
            <strong>Learned Attention</strong>, <strong>Spatial Kernel</strong>, and <strong>Defect Mask</strong> (middle),
            finally aggregating at the <strong>Target</strong> (right).
            </div>
            """, unsafe_allow_html=True)
            
            domain_info = DomainConfiguration.get_domain_info()
            fig_sankey = st.session_state.heatmap_visualizer.create_attention_sankey_diagram(
                weights, source_info, target_params, spatial_sigma, 
                domain_size_nm=domain_info['domain_length_nm']
            )
            if fig_sankey:
                st.plotly_chart(fig_sankey, use_container_width=True)
        
        with tab4:
            st.markdown('<h2 class="section-header">Chord Diagram: Source Interconnections</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Circular layout showing sources arranged by angle. 
            <strong>Chords</strong> connect sources with thickness proportional to attention weight product.
            <strong>Node color</strong> indicates defect type.
            <strong>Red dashed line</strong> points to target angle.
            </div>
            """, unsafe_allow_html=True)
            
            fig_chord = st.session_state.heatmap_visualizer.create_chord_diagram(
                weights, source_info, target_params
            )
            if fig_chord:
                st.plotly_chart(fig_chord, use_container_width=True)
        
        with tab5:
            st.markdown('<h2 class="section-header">Weight Statistics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entropy", f"{weights['entropy']:.4f}")
            with col2:
                st.metric("Max Combined Weight", f"{np.max(weights['combined']):.4f}")
            with col3:
                st.metric("Sources Matching Defect", f"{sum(1 for m in weights['defect_mask'] if m > 0.5)}")
            
            # Data table
            df_weights = pd.DataFrame({
                'Source': range(len(weights['combined'])),
                'Angle (°)': source_info['theta_degrees'],
                'Distance to Target': source_info['distances'],
                'Learned (Pre-Mask)': weights['pre_mask'],
                'Spatial Kernel': weights['spatial_kernel'],
                'Defect Mask': weights['defect_mask'],
                'Combined (Final)': weights['combined']
            })
            
            st.dataframe(
                df_weights.style.background_gradient(subset=['Combined (Final)'], cmap='Reds')
                .background_gradient(subset=['Spatial Kernel'], cmap='Blues'),
                use_container_width=True
            )

if __name__ == "__main__":
    main()
