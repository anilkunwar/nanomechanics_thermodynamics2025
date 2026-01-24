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
        'Twin': 2.12,  # Corrected from 0.707
        'ISF': 0.289,  # Corrected from 0.707
        'ESF': 0.333,  # Corrected from 1.414
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
    
    k_B_eV = 8.617333262145e-5  # eV/K
    k_B_J = 1.380649e-23  # J/K
    
    MATERIAL_PROPERTIES = {
        'Silver': {
            'atomic_volume': 1.56e-29,  # m³/atom
            'atomic_mass': 107.8682,    # g/mol
            'density': 10.49e6,         # g/m³
            'melting_point': 1234.93,   # K
            'bulk_modulus': 100e9,      # Pa
            'shear_modulus': 30e9,      # Pa
            'activation_energy': 1.1,   # eV
            'prefactor': 7.2e-7,        # m²/s
            'atomic_radius': 1.44e-10,  # m
            'vacancy_formation_energy': 1.1,  # eV
            'vacancy_migration_energy': 0.66,  # eV
        },
        'Copper': {
            'atomic_volume': 1.18e-29,  # m³/atom
            'atomic_mass': 63.546,      # g/mol
            'density': 8.96e6,         # g/m³
            'melting_point': 1357.77,   # K
            'bulk_modulus': 140e9,      # Pa
            'shear_modulus': 48e9,      # Pa
            'activation_energy': 1.0,   # eV
            'prefactor': 3.1e-7,        # m²/s
            'atomic_radius': 1.28e-10,  # m
            'vacancy_formation_energy': 1.0,  # eV
            'vacancy_migration_energy': 0.70,  # eV
        },
        'Aluminum': {
            'atomic_volume': 1.66e-29,  # m³/atom
            'atomic_mass': 26.9815,     # g/mol
            'density': 2.70e6,         # g/m³
            'melting_point': 933.47,    # K
            'bulk_modulus': 76e9,       # Pa
            'shear_modulus': 26e9,      # Pa
            'activation_energy': 0.65,  # eV
            'prefactor': 1.7e-6,        # m²/s
            'atomic_radius': 1.43e-10,  # m
            'vacancy_formation_energy': 0.65,  # eV
            'vacancy_migration_energy': 0.55,  # eV
        },
        'Nickel': {
            'atomic_volume': 1.09e-29,  # m³/atom
            'atomic_mass': 58.6934,     # g/mol
            'density': 8.908e6,        # g/m³
            'melting_point': 1728.0,    # K
            'bulk_modulus': 180e9,      # Pa
            'shear_modulus': 76e9,      # Pa
            'activation_energy': 1.4,   # eV
            'prefactor': 1.9e-7,        # m²/s
            'atomic_radius': 1.24e-10,  # m
            'vacancy_formation_energy': 1.4,  # eV
            'vacancy_migration_energy': 0.9,  # eV
        },
        'Iron': {
            'atomic_volume': 1.18e-29,  # m³/atom
            'atomic_mass': 55.845,      # g/mol
            'density': 7.874e6,        # g/m³
            'melting_point': 1811.0,    # K
            'bulk_modulus': 170e9,      # Pa
            'shear_modulus': 82e9,      # Pa
            'activation_energy': 2.0,   # eV
            'prefactor': 2.0e-8,        # m²/s
            'atomic_radius': 1.24e-10,  # m
            'vacancy_formation_energy': 2.0,  # eV
            'vacancy_migration_energy': 1.2,  # eV
        }
    }
    
    @staticmethod
    def get_material_properties(material='Silver'):
        """Get material properties for diffusion calculations"""
        return DiffusionPhysics.MATERIAL_PROPERTIES.get(
            material, 
            DiffusionPhysics.MATERIAL_PROPERTIES['Silver']
        )
    
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
                            'von_mises': vm,
                            'sigma_hydro': hydro,
                            'sigma_mag': mag,
                            'source_index': i,
                            'source_params': src['params']
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
            
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            if source_features.shape[1] != 15 or target_features.shape[1] != 15:
                st.warning(f"Feature dimension mismatch")
            
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
            
            entropy_final = self._calculate_entropy(final_attention_weights)
            
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
# ENHANCED HEAT MAP VISUALIZER WITH EXTENSIVE CUSTOMIZATION
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with diffusion visualization and granular customization"""
    
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
    
    def _apply_custom_styles(self, ax, label_config):
        """Applies granular styling to an axes object."""
        
        # Fonts
        fontsize_title = label_config.get('fontsize_title', 14)
        fontsize_label = label_config.get('fontsize_label', 11)
        fontsize_tick = label_config.get('fontsize_tick', 10)
        fontsize_text = label_config.get('fontsize_text', 9)
        fontsize_legend = label_config.get('fontsize_legend', 9)
        
        # Ticks
        tick_width = label_config.get('tick_width', 0.8)
        tick_length = label_config.get('tick_length', 4.0)
        tick_color = label_config.get('tick_color', 'black')
        
        # Grid
        show_grid = not label_config.get('hide_grid', False)
        grid_linewidth = label_config.get('grid_linewidth', 0.5)
        grid_alpha = label_config.get('grid_alpha', 0.2)
        grid_color = label_config.get('grid_color', 'gray')
        grid_linestyle = label_config.get('grid_linestyle', '--')
        
        # Spines (Axes Border)
        spine_width = label_config.get('spine_width', 0.8)
        spine_color = label_config.get('spine_color', 'gray')
        
        # Apply Ticks
        ax.tick_params(axis='both', which='major', labelsize=fontsize_tick, 
                       width=tick_width, length=tick_length, color=tick_color)
        
        # Apply Grid
        if show_grid:
            ax.grid(True, linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha, color=grid_color)
        else:
            ax.grid(False)
            
        # Apply Spines
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
            spine.set_color(spine_color)
            
        return {
            'title': fontsize_title,
            'label': fontsize_label,
            'tick': fontsize_tick,
            'text': fontsize_text,
            'legend': fontsize_legend
        }

    def _apply_label(self, ax, default_text, override_text, func_name, fontsize, **kwargs):
        if override_text is not None:
            if override_text.strip() == "":
                getattr(ax, func_name)(None)
            else:
                getattr(ax, func_name)(override_text, fontsize=fontsize, **kwargs)
        else:
            getattr(ax, func_name)(default_text, fontsize=fontsize, **kwargs)
            
    def _get_bbox_style(self, label_config):
        """Returns bbox dict based on config."""
        return dict(
            boxstyle='round',
            facecolor='white',
            alpha=label_config.get('box_alpha', 0.9),
            edgecolor=label_config.get('box_edgecolor', 'black'),
            linewidth=label_config.get('box_linewidth', 1.0)
        )

    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info,
                                   target_angle, defect_type, component='von_mises',
                                   cmap_name='viridis', figsize=(24, 18),
                                   ground_truth_index=None, defect_type_filter=None,
                                   label_config=None):
        """
        Create comprehensive comparison dashboard with EXTENSIVE CUSTOMIZATION.
        """
        # --- STEP 1: FILTER SOURCES ---
        if defect_type_filter:
            filtered_indices = [i for i, src in enumerate(source_fields) 
                             if src.get('source_params', {}).get('defect_type', 'Unknown') == defect_type_filter]
            filtered_ground_truth_index = None
            if ground_truth_index is not None and ground_truth_index in filtered_indices:
                try: filtered_ground_truth_index = filtered_indices.index(ground_truth_index)
                except: pass
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
            if not source_fields: return None

        # --- STEP 2: FIGURE SETUP ---
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        all_values = [interpolated_fields[component]]
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None: all_values.append(gt_field)
        vmin = min(np.nanmin(field) for field in all_values) if all_values else 0
        vmax = max(np.nanmax(field) for field in all_values) if all_values else 1
        if cmap_name not in plt.colormaps(): cmap_name = 'viridis'

        # --- STEP 3: PLOTTING WITH CUSTOMIZATION ---
        
        # --- PLOT 1: Interpolated result ---
        ax1 = fig.add_subplot(gs[0, 0])
        styles1 = self._apply_custom_styles(ax1, label_config)
        im1 = ax1.imshow(interpolated_fields[component], cmap=cmap_name,
                        vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
        
        show_cb = not label_config.get('hide_colorbar', False)
        if show_cb:
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.06)
            cbar1.set_label(f"{component.replace('_', ' ').title()} (GPa)", fontsize=styles1['label'], fontweight='bold')
            cbar1.ax.tick_params(labelsize=styles1['tick'])
        
        title_suffix = f" {label_config['title_suffix']}" if label_config.get('title_suffix') else ""
        if not label_config.get('hide_title', False):
            ax1.set_title(f'Interpolated Result\nθ = {target_angle:.1f}°, {defect_type}{title_suffix}',
                         fontsize=styles1['title'], fontweight='bold', pad=10)
        self._apply_label(ax1, "X Position", label_config.get('xlabel'), 'set_xlabel', fontsize=styles1['label'])
        self._apply_label(ax1, "Y Position", label_config.get('ylabel'), 'set_ylabel', fontsize=styles1['label'])

        # --- PLOT 2: Ground truth ---
        ax2 = fig.add_subplot(gs[0, 1])
        styles2 = self._apply_custom_styles(ax2, label_config)
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                gt_theta = source_info['theta_degrees'][ground_truth_index]
                gt_distance = source_info['distances'][ground_truth_index]
                im2 = ax2.imshow(gt_field, cmap=cmap_name,
                                vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
                if show_cb:
                    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.06)
                    cbar2.set_label(f"{component.replace('_', ' ').title()} (GPa)", fontsize=styles2['label'], fontweight='bold')
                    cbar2.ax.tick_params(labelsize=styles2['tick'])
                if not label_config.get('hide_title', False):
                    ax2.set_title(f'Ground Truth ({defect_type_filter or "All"})\nθ = {gt_theta:.1f}° (Δ={gt_distance:.1f}°){title_suffix}',
                                 fontsize=styles2['title'], fontweight='bold', pad=10)
                self._apply_label(ax2, "X Position", label_config.get('xlabel'), 'set_xlabel', fontsize=styles2['label'])
                self._apply_label(ax2, "Y Position", label_config.get('ylabel'), 'set_ylabel', fontsize=styles2['label'])
            else:
                ax2.text(0.5, 0.5, f'Component "{component}"\nmissing in ground truth',
                        ha='center', va='center', fontsize=styles2['text'], fontweight='bold')
                ax2.set_axis_off()
        else:
            ax2.text(0.5, 0.5, 'Select Ground Truth Source',
                    ha='center', va='center', fontsize=styles2['text'], fontweight='bold')
            ax2.set_title('Ground Truth Selection', fontsize=styles2['title'], fontweight='bold', pad=10)
            ax2.set_axis_off()
            
        # --- PLOT 3: Difference ---
        ax3 = fig.add_subplot(gs[0, 2])
        styles3 = self._apply_custom_styles(ax3, label_config)
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                diff_field = interpolated_fields[component] - gt_field
                max_diff = np.max(np.abs(diff_field))
                im3 = ax3.imshow(diff_field, cmap='RdBu_r',
                                vmin=-max_diff, vmax=max_diff, aspect='equal',
                                interpolation='bilinear', origin='lower')
                if show_cb:
                    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.06)
                    cbar3.set_label('Difference (GPa)', fontsize=styles3['label'], fontweight='bold')
                    cbar3.ax.tick_params(labelsize=styles3['tick'])
                if not label_config.get('hide_title', False):
                    ax3.set_title(f'Difference\nMax Abs Error: {max_diff:.3f} GPa{title_suffix}',
                                 fontsize=styles3['title'], fontweight='bold', pad=10)
                self._apply_label(ax3, "X Position", label_config.get('xlabel'), 'set_xlabel', fontsize=styles3['label'])
                self._apply_label(ax3, "Y Position", label_config.get('ylabel'), 'set_ylabel', fontsize=styles3['label'])
                
                mse = np.mean(diff_field**2)
                mae = np.mean(np.abs(diff_field))
                rmse = np.sqrt(mse)
                error_text = (f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}")
                if not label_config.get('hide_stats_box', False):
                    ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes,
                            fontsize=styles3['text'], fontweight='bold', verticalalignment='top',
                            bbox=self._get_bbox_style(label_config))
            else:
                ax3.text(0.5, 0.5, 'Ground truth missing\nfor difference plot',
                        ha='center', va='center', fontsize=styles3['text'], fontweight='bold')
                ax3.set_axis_off()
        else:
            ax3.text(0.5, 0.5, 'Difference will appear\nwhen ground truth is selected',
                    ha='center', va='center', fontsize=styles3['text'], fontweight='bold')
            ax3.set_title('Difference Analysis', fontsize=styles3['title'], fontweight='bold', pad=10)
            ax3.set_axis_off()
            
        # --- PLOT 4: Weight distribution ---
        ax4 = fig.add_subplot(gs[1, 0])
        styles4 = self._apply_custom_styles(ax4, label_config)
        if 'weights' in source_info:
            final_weights = source_info['weights']['combined']
            x = range(len(final_weights))
            bars = ax4.bar(x, final_weights, alpha=0.7, color='steelblue', edgecolor='black', label='Final Attention')
            if 'spatial_kernel' in source_info['weights']:
                spatial_k = source_info['weights']['spatial_kernel']
                ax4.plot(x, spatial_k, 'g--', linewidth=2, label='Spatial Kernel', alpha=0.8)
            if not label_config.get('hide_title', False):
                ax4.set_title('Attention vs Spatial Kernel', fontsize=styles4['title'], fontweight='bold', pad=10)
            self._apply_label(ax4, "Source Index", label_config.get('xlabel'), 'set_xlabel', fontsize=styles4['label'])
            self._apply_label(ax4, "Weight", label_config.get('ylabel'), 'set_ylabel', fontsize=styles4['label'])
            
            # Legend customization
            if not label_config.get('hide_legend', False):
                ax4.legend(loc='best', fontsize=styles4['legend'], framealpha=0.9)
            
            if ground_truth_index is not None and 0 <= ground_truth_index < len(bars):
                bars[ground_truth_index].set_color('red')
                bars[ground_truth_index].set_alpha(0.9)
        
        # --- PLOT 5: Angular distribution ---
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')
        styles5 = self._apply_custom_styles(ax5, label_config) # Note: Polar ticks are different but general params apply where valid
        if 'theta_degrees' in source_info and 'distances' in source_info:
            angles_rad = np.radians(source_info['theta_degrees'])
            distances = source_info['distances']
            weights = source_info['weights']['combined']
            sizes = 100 * np.array(weights) / (np.max(weights) + 1e-8)
            
            scatter = ax5.scatter(angles_rad, distances,
                                s=sizes, alpha=0.7, c='blue', edgecolors='black')
            target_rad = np.radians(target_angle)
            lbl_tgt = label_config.get('legend_target', 'Target')
            if lbl_tgt and lbl_tgt != "": ax5.scatter(target_rad, 0, s=200, c='red', marker='*', edgecolors='black', label=lbl_tgt)
            
            habit_rad = np.radians(54.7)
            lbl_hp = label_config.get('legend_habit', 'Habit Plane (54.7°)')
            if lbl_hp and lbl_hp != "": ax5.axvline(habit_rad, color='green', alpha=0.5, linestyle='--', label=lbl_hp)
            
            if not label_config.get('hide_title', False):
                ax5.set_title('Angular Distribution', fontsize=styles5['title'], fontweight='bold', pad=15)
            ax5.set_theta_zero_location('N')
            ax5.set_theta_direction(-1)
            if not label_config.get('hide_legend', False):
                ax5.legend(loc='upper right', fontsize=styles5['legend'], bbox_to_anchor=(1.3, 1.1))

        # --- PLOT 6: Component stats ---
        ax6 = fig.add_subplot(gs[1, 2])
        styles6 = self._apply_custom_styles(ax6, label_config)
        components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        component_names = ['Von Mises', 'Hydrostatic', 'Stress Magnitude']
        stats_data = []
        if 'statistics' in source_info:
            for comp in components:
                if comp in source_info['statistics']:
                    stats = source_info['statistics'][comp]
                    stats_data.append({'component': comp, 'max': stats['max'], 'mean': stats['mean'], 'std': stats['std']})
        if stats_data:
            x = np.arange(len(components))
            width = 0.25
            max_values = [s['max'] for s in stats_data]
            mean_values = [s['mean'] for s in stats_data]
            std_values = [s['std'] for s in stats_data]
            
            lbl_max = label_config.get('legend_max', 'Max')
            lbl_mean = label_config.get('legend_mean', 'Mean')
            lbl_std = label_config.get('legend_std', 'Std')
            
            if lbl_max: ax6.bar(x - width, max_values, width, label=lbl_max, color='red', alpha=0.7)
            if lbl_mean: ax6.bar(x, mean_values, width, label=lbl_mean, color='blue', alpha=0.7)
            if lbl_std: ax6.bar(x + width, std_values, width, label=lbl_std, color='green', alpha=0.7)
            
            if not label_config.get('hide_title', False):
                ax6.set_title('Component Statistics', fontsize=styles6['title'], fontweight='bold', pad=10)
            self._apply_label(ax6, "Stress Component", label_config.get('xlabel'), 'set_xlabel', fontsize=styles6['label'])
            self._apply_label(ax6, "Value (GPa)", label_config.get('ylabel'), 'set_ylabel', fontsize=styles6['label'])
            ax6.set_xticks(x)
            ax6.set_xticklabels(component_names, rotation=15, fontsize=styles6['tick'])
            if not label_config.get('hide_legend', False):
                ax6.legend(loc='best', fontsize=styles6['legend'])

        # --- PLOT 7: Defect Type Gating ---
        ax7 = fig.add_subplot(gs[2, 0])
        styles7 = self._apply_custom_styles(ax7, label_config)
        if 'weights' in source_info and 'defect_mask' in source_info['weights']:
            defect_masks = source_info['weights']['defect_mask']
            x_pos = np.arange(len(defect_masks))
            lbl_def = label_config.get('legend_defect', 'Defect Mask')
            if lbl_def:
                ax7.bar(x_pos, defect_masks, color='purple', alpha=0.6, label=lbl_def)
            
            for i, mask_val in enumerate(defect_masks):
                if mask_val > 0.1:
                    ax7.text(i, mask_val + 0.02, 'Active', ha='center', fontsize=styles7['text'], fontweight='bold')
            
            if not label_config.get('hide_title', False):
                ax7.set_title('Defect Type Filter Active', fontsize=styles7['title'], fontweight='bold', pad=10)
            self._apply_label(ax7, "Source Index", label_config.get('xlabel'), 'set_xlabel', fontsize=styles7['label'])
            self._apply_label(ax7, "Gating Weight", label_config.get('ylabel'), 'set_ylabel', fontsize=styles7['label'])
            ax7.set_ylim([0, 1.1])
            if not label_config.get('hide_legend', False):
                ax7.legend(loc='best', fontsize=styles7['legend'])

        # --- PLOT 8: Correlation ---
        ax8 = fig.add_subplot(gs[2, 1:])
        styles8 = self._apply_custom_styles(ax8, label_config)
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                interp_flat = interpolated_fields[component].flatten()
                gt_flat = gt_field.flatten()
                scatter = ax8.scatter(gt_flat, interp_flat, alpha=0.5, s=10, c='navy')
                
                min_val = min(np.min(gt_flat), np.min(interp_flat))
                max_val = max(np.max(gt_flat), np.max(interp_flat))
                lbl_corr = label_config.get('legend_corr', 'Perfect Correlation')
                if lbl_corr:
                    ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label=lbl_corr)
                
                try:
                    from scipy.stats import pearsonr
                    corr_coef, _ = pearsonr(gt_flat, interp_flat)
                except: corr_coef = 0.0
                
                if not label_config.get('hide_title', False):
                    ax8.set_title(f'Spatial Correlation Analysis\nPearson: {corr_coef:.3f}{title_suffix}', 
                                 fontsize=styles8['title'], fontweight='bold', pad=10)
                
                self._apply_label(ax8, f'Ground Truth {component.replace("_", " ").title()} (GPa)', label_config.get('xlabel'), 'set_xlabel', fontsize=styles8['label'])
                self._apply_label(ax8, f'Interpolated {component.replace("_", " ").title()} (GPa)', label_config.get('ylabel'), 'set_ylabel', fontsize=styles8['label'])
                
                mse = np.mean((interp_flat - gt_flat)**2)
                mae = np.mean(np.abs(interp_flat - gt_flat))
                stats_text = (f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nPearson: {corr_coef:.3f}')
                if not label_config.get('hide_stats_box', False):
                    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                            fontsize=styles8['text'], fontweight='bold', verticalalignment='top',
                            bbox=self._get_bbox_style(label_config))
                
                if lbl_corr:
                    if not label_config.get('hide_legend', False):
                        ax8.legend(loc='best', fontsize=styles8['legend'])

        # Global Suptitle
        if not label_config.get('hide_suptitle', False):
            suptitle_txt = f'Theory-Informed Interpolation: Target θ={target_angle:.1f}°, {defect_type}'
            if label_config.get('suptitle') and label_config['suptitle'] != "":
                suptitle_txt = label_config['suptitle']
            plt.suptitle(suptitle_txt, fontsize=label_config.get('suptitle_fontsize', 22), 
                        fontweight='bold', y=0.98)
        else:
            gs.update(top=0.95)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
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
    st.set_page_config(
        page_title="Angular Bracketing Theory",
        layout="wide",
        page_icon="🎯",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem !important; color: #1E3A8A !important; text-align: center; padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900 !important;
    }
    .section-header {
        font-size: 1.8rem !important; color: #374151 !important; font-weight: 800 !important;
        border-left: 6px solid #3B82F6; padding-left: 1.2rem; margin-top: 1.5rem; margin-bottom: 1.0rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🎯 Advanced Dashboard Designer</h1>', unsafe_allow_html=True)
    
    # Session State
    if 'solutions' not in st.session_state: st.session_state.solutions = []
    if 'loader' not in st.session_state: st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator(spatial_sigma=10.0, locality_weight_factor=0.5)
    if 'heatmap_visualizer' not in st.session_state: st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state: st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state: st.session_state.interpolation_result = None
    if 'selected_ground_truth' not in st.session_state: st.session_state.selected_ground_truth = None
    
    # Sidebar: Configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Core Config</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Load Solutions", use_container_width=True):
                with st.spinner("Loading..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                st.success(f"Loaded {len(st.session_state.solutions)} solutions")
        with col2:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.success("Cleared")
        
        # Target Params
        st.markdown("#### 🎯 Target")
        custom_theta = st.number_input("Angle (deg)", min_value=0.0, max_value=180.0, value=54.7, step=0.1)
        defect_type = st.selectbox("Defect", options=['ISF', 'ESF', 'Twin', 'No Defect'], index=2)
        eigen_strain = PhysicsParameters.get_eigenstrain(defect_type)
        st.metric("ε₀", f"{eigen_strain:.3f}")

        # Interpolation
        if st.button("🎯 Run Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("No solutions")
            else:
                with st.spinner("Computing..."):
                    target_params = {'defect_type': defect_type, 'eps0': eigen_strain, 'kappa': 0.6, 'theta': np.radians(custom_theta), 'shape': 'Square'}
                    result = st.session_state.transformer_interpolator.interpolate_spatial_fields(st.session_state.solutions, custom_theta, target_params)
                    if result:
                        st.session_state.interpolation_result = result
                        st.success("Done!")
                    else: st.error("Failed")

        # --- EXTENSIVE CUSTOMIZATION SECTION ---
        st.divider()
        st.markdown('<h2 class="section-header">🎨 Advanced Styling</h2>', unsafe_allow_html=True)
        st.info("Customize fonts, lines, and colors for the dashboard.")
        
        with st.expander("📝 Text & Fonts"):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                title_fs = st.slider("Title Size", 10, 40, 14)
                label_fs = st.slider("Label Size", 8, 30, 11)
                tick_fs = st.slider("Tick Size", 6, 24, 10)
            with col_f2:
                text_fs = st.slider("Text Box Size", 6, 24, 9)
                legend_fs = st.slider("Legend Size", 6, 24, 9)
                suptitle_fs = st.slider("Main Title Size", 16, 40, 22)
            
            # Text Overrides
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                title_suff = st.text_input("Title Suffix", "")
                xlabel_txt = st.text_input("X-Label Override", "")
            with col_o2:
                ylabel_txt = st.text_input("Y-Label Override", "")
                suptitle_txt = st.text_input("Suptitle Override", "")
            
            # Hide toggles
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                hide_title = st.checkbox("Hide Subplot Titles")
            with col_h2:
                hide_stats = st.checkbox("Hide Stats Boxes")
            with col_h3:
                hide_legend = st.checkbox("Hide Legends")
                
        with st.expander("📏 Lines & Grids"):
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                grid_lw = st.slider("Grid Width", 0.0, 2.0, 0.5)
                grid_alp = st.slider("Grid Alpha", 0.0, 1.0, 0.2)
                spine_lw = st.slider("Border Width", 0.5, 3.0, 0.8)
            with col_l2:
                tick_w = st.slider("Tick Width", 0.5, 3.0, 0.8)
                tick_l = st.slider("Tick Length", 1.0, 10.0, 4.0)
                box_lw = st.slider("Box Border Width", 0.1, 3.0, 1.0)
                
            col_l3 = st.columns(1)
            with col_l3:
                hide_grid = st.checkbox("Hide Grid", False)
                hide_cb = st.checkbox("Hide Colorbars", False)

        with st.expander("🏷️ Legend Labels"):
            # Provide overrides for legend items
            leg_attn = st.text_input("Attention Legend", "Final Attention")
            leg_spat = st.text_input("Spatial Legend", "Spatial Kernel")
            leg_max = st.text_input("Max Legend", "Max")
            leg_mean = st.text_input("Mean Legend", "Mean")
            leg_std = st.text_input("Std Legend", "Std")
            leg_def = st.text_input("Defect Legend", "Defect Mask")
            leg_tgt = st.text_input("Target Legend", "Target")
            leg_hp = st.text_input("Habit Legend", "Habit Plane")
            leg_corr = st.text_input("Corr Legend", "Perfect Correlation")

        # --- COMPILE CONFIG ---
        label_config = {
            'fontsize_title': title_fs,
            'fontsize_label': label_fs,
            'fontsize_tick': tick_fs,
            'fontsize_text': text_fs,
            'fontsize_legend': legend_fs,
            'suptitle_fontsize': suptitle_fs,
            'title_suffix': title_suff,
            'xlabel': xlabel_txt,
            'ylabel': ylabel_txt,
            'suptitle': suptitle_txt if suptitle_txt else None,
            'hide_title': hide_title,
            'hide_stats_box': hide_stats,
            'hide_legend': hide_legend,
            'grid_linewidth': grid_lw,
            'grid_alpha': grid_alp,
            'spine_width': spine_lw,
            'tick_width': tick_w,
            'tick_length': tick_l,
            'box_linewidth': box_lw,
            'hide_grid': hide_grid,
            'hide_colorbar': hide_cb,
            'legend_attention': leg_attn,
            'legend_spatial': leg_spat,
            'legend_max': leg_max,
            'legend_mean': leg_mean,
            'legend_std': leg_std,
            'legend_defect': leg_def,
            'legend_target': leg_tgt,
            'legend_habit': leg_hp,
            'legend_corr': leg_corr
        }
        # -------------------------------

    # Main Area
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # REMOVED TABS: Overview, Stress Viz, Diffusion Viz
        # KEPT TABS: Dashboard, Export
        tab_dash, tab_exp = st.tabs(["🔄 Comparison Dashboard", "💾 Export Results"])
        
        with tab_dash:
            st.markdown('<h2 class="section-header">🔄 Comparison Dashboard</h2>', unsafe_allow_html=True)
            
            # Defect Filter
            col_filter1, col_filter2 = st.columns([1, 3])
            with col_filter1:
                unique_defects = set()
                for sol in st.session_state.solutions:
                    if 'params' in sol and 'defect_type' in sol['params']:
                        unique_defects.add(sol['params']['defect_type'])
                defect_options = ['All'] + sorted(list(unique_defects))
                selected_defect_filter = st.selectbox("Filter Defect", options=defect_options, index=0)
            actual_defect_filter = None if selected_defect_filter == 'All' else selected_defect_filter
            
            # GT Selection
            if 'source_theta_degrees' in result and result['source_theta_degrees']:
                ground_truth_options = []
                for i, theta in enumerate(result['source_theta_degrees']):
                    d_type = result['source_fields'][i].get('source_params', {}).get('defect_type', 'Unknown')
                    weight = result['weights']['combined'][i]
                    tag = "✅" if actual_defect_filter is None or d_type == actual_defect_filter else "❌"
                    ground_truth_options.append(f"{tag} Source {i}: {d_type} (θ={theta:.1f}°, w={weight:.3f})")
                
                selected_option = st.selectbox("Ground Truth", options=ground_truth_options, index=0)
                match = re.search(r'Source (\d+):', selected_option)
                if match:
                    st.session_state.selected_ground_truth = int(match.group(1))
            
            # Viz Options
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                comp_component = st.selectbox("Component", options=['von_mises', 'sigma_hydro', 'sigma_mag', 'diffusion_ratio'], index=0)
            with col_viz2:
                all_cmaps = []
                for group in st.session_state.heatmap_visualizer.colormaps.values():
                    all_cmaps.extend(group)
                comp_cmap = st.selectbox("Colormap", options=all_cmaps, index=all_cmaps.index('viridis'))
            
            # Plot Dashboard
            if comp_component in result['fields']:
                source_info = {
                    'theta_degrees': result['source_theta_degrees'],
                    'distances': result['source_distances'],
                    'weights': result['weights'],
                    'source_fields': result.get('source_fields', [])
                }
                
                if comp_component == 'diffusion_ratio':
                    # Logic for diffusion ratio plotting similar to previous iteration
                    pass 
                else:
                    fig_comparison = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                        interpolated_fields=result['fields'],
                        source_fields=result.get('source_fields', []),
                        source_info=source_info,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        component=comp_component,
                        cmap_name=comp_cmap,
                        figsize=(24, 18),
                        ground_truth_index=st.session_state.selected_ground_truth,
                        defect_type_filter=actual_defect_filter,
                        label_config=label_config
                    )
                    st.pyplot(fig_comparison)
        
        with tab_exp:
            st.markdown('<h2 class="section-header">💾 Export</h2>', unsafe_allow_html=True)
            # Export logic (kept from previous code)
            export_format = st.radio("Format", ["JSON", "CSV"], horizontal=True)
            if export_format == "JSON":
                viz_params = {'component': 'von_mises', 'colormap': 'viridis'}
                data = st.session_state.results_manager.prepare_export_data(result, viz_params)
                js, fn = st.session_state.results_manager.export_to_json(data)
                st.download_button("Download JSON", data=js, file_name=fn, mime="application/json")
            elif export_format == "CSV":
                csv, fn = st.session_state.results_manager.export_to_csv(result)
                st.download_button("Download CSV", data=csv, file_name=fn, mime="text/csv")

if __name__ == "__main__":
    main()
