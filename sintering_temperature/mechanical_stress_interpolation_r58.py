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
import networkx as nx
import warnings
from math import cos, sin, pi
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
# EXTENSIVE COLORMAP LIBRARY (50+ COLORMAPS)
# =============================================
COLORMAP_OPTIONS = {
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'rocket', 'mako', 'flare', 'crest'],
    'Sequential': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 
                   'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 
                   'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 
                   'hot', 'afmhot', 'gist_heat', 'copper'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                  'RdGy', 'RdYlGn', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r'],
    'Rainbow Family': ['rainbow', 'jet', 'gist_rainbow', 'nipy_spectral', 'gist_ncar', 'hsv', 'prism', 'flag',
                       'rainbow_r', 'jet_r', 'gist_rainbow_r', 'nipy_spectral_r', 'gist_ncar_r', 'hsv_r'],
    'Cyclic': ['hsv', 'twilight', 'twilight_shifted', 'phase'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                    'Paired', 'Accent', 'Dark2'],
    'Geographical': ['terrain', 'ocean', 'gist_earth', 'gist_stern', 'brg', 'CMRmap', 'cubehelix',
                     'terrain_r', 'ocean_r', 'gist_earth_r', 'gist_stern_r'],
    'Temperature': ['hot', 'afmhot', 'gist_heat', 'coolwarm', 'bwr', 'seismic'],
    'Specialized': ['gist_earth', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'RdBu_r', 'Spectral',
                             'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn']
}

# Create comprehensive list of all colormaps
ALL_COLORMAPS = []
for category in COLORMAP_OPTIONS.values():
    ALL_COLORMAPS.extend(category)
ALL_COLORMAPS = sorted(list(set(ALL_COLORMAPS)))  # Remove duplicates

# =============================================
# DOMAIN SIZE CONFIGURATION - 12.8 nm × 12.8 nm
# =============================================
class DomainConfiguration:
    """Configuration for the 12.8 nm × 12.8 nm simulation domain"""
    # Domain parameters
    N = 128                    # Number of grid points in each direction
    dx = 0.1                   # Grid spacing in nm
    DOMAIN_LENGTH = N * dx     # 12.8 nm
    DOMAIN_HALF = DOMAIN_LENGTH / 2.0  # 6.4 nm

    @classmethod
    def get_extent(cls):
        """Get extent for plotting: [-6.4 nm, 6.4 nm, -6.4 nm, 6.4 nm]"""
        return [-cls.DOMAIN_HALF, cls.DOMAIN_HALF, -cls.DOMAIN_HALF, cls.DOMAIN_HALF]

    @classmethod
    def get_coordinates(cls):
        """Get coordinate arrays for the domain"""
        x = np.linspace(-cls.DOMAIN_HALF, cls.DOMAIN_HALF, cls.N, endpoint=False)
        y = np.linspace(-cls.DOMAIN_HALF, cls.DOMAIN_HALF, cls.N, endpoint=False)
        return np.meshgrid(x, y)

    @classmethod
    def get_domain_info(cls):
        """Get domain information dictionary"""
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
                
            source_theta_degrees = [np.degrees(src.get('theta', 0.0)) % 360 for src in source_params]
            
            # Prepare weight analysis data with all components
            sources_data = []
            for i, (field, theta_deg, angular_dist, spatial_w, defect_w, combined_w) in enumerate(zip(
                source_fields, source_theta_degrees, angular_distances, 
                spatial_kernel, defect_mask, final_attention_weights
            )):
                # Calculate attention weight (from transformer)
                attention_weight = attn_scores.squeeze().detach().cpu().numpy()[i] if i < len(attn_scores.squeeze()) else 0.0
                
                sources_data.append({
                    'source_index': i,
                    'theta_deg': theta_deg,
                    'angular_dist': angular_dist,
                    'defect_type': field['source_params'].get('defect_type', 'Unknown'),
                    'spatial_weight': spatial_w,
                    'defect_weight': defect_w,
                    'attention_weight': attention_weight,
                    'combined_weight': combined_w,
                    'target_defect_match': field['source_params'].get('defect_type') == target_params['defect_type'],
                    'is_query': False
                })
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist(),
                    'learned_attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
                    'entropy': entropy_final
                },
                'sources_data': sources_data,
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
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
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
# ENHANCED WEIGHT VISUALIZER WITH ADVANCED DIAGRAMS
# =============================================
class WeightVisualizer:
    def __init__(self):
        # Enhanced color scheme with high contrast
        self.color_scheme = {
            'Twin': '#FF6B6B',      # Bright Red
            'ISF': '#4ECDC4',       # Turquoise
            'ESF': '#95E1D3',       # Light Teal
            'No Defect': '#FFD93D', # Bright Yellow
            'Query': '#9D4EDD',     # Purple
            'Spatial': '#36A2EB',   # Bright Blue
            'Defect': '#FF6384',    # Pink
            'Attention': '#4BC0C0', # Cyan
            'Combined': '#9966FF'   # Purple
        }
        
        # Font configuration for high readability
        self.font_config = {
            'family': 'Arial, sans-serif',
            'size_title': 24,
            'size_labels': 18,
            'size_ticks': 14,
            'color': '#2C3E50',
            'weight': 'bold'
        }
        
        # High contrast color maps for different visualizations
        self.color_maps = {
            'sankey': 'viridis',
            'chord': 'plasma',
            'radar': 'inferno',
            'sunburst': 'magma',
            'treemap': 'turbo',
            'heatmap': 'jet'
        }
    
    def get_colormap(self, cmap_name, n_colors=10):
        """Get color palette from colormap"""
        try:
            cmap = plt.get_cmap(cmap_name)
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
                   for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
        except:
            # Fallback to viridis
            cmap = plt.get_cmap('viridis')
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
                   for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
    #
    def create_enhanced_sankey_diagram(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create enhanced Sankey diagram with improved fonts, contrast, and colors
        """
        # Create nodes for Sankey diagram
        labels = ['Target']  # Start with target node
        
        # Add source nodes with enhanced labels
        for source in sources_data:
            angle = source['theta_deg']
            defect = source['defect_type']
            labels.append(f"S{source['source_index']}\n{defect}\n{angle:.1f}°")
        
        # Add component nodes
        component_start = len(labels)
        labels.extend(['Spatial Kernel', 'Defect Match', 'Attention Score', 'Combined Weight'])
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        link_labels = []
        
        # Get vibrant color palette
        color_palette = self.get_colormap(self.color_maps['sankey'], len(sources_data) + 4)
        
        # Links from sources to components
        for i, source in enumerate(sources_data):
            source_idx = i + 1  # +1 because index 0 is target
            
            # Get vibrant color for this source
            source_color = color_palette[i % len(color_palette)]
            
            # Link to spatial kernel
            source_indices.append(source_idx)
            target_indices.append(component_start)
            spatial_value = source['spatial_weight'] * 100  # Scale for visibility
            values.append(spatial_value)
            colors.append(f'rgba(54, 162, 235, 0.8)')  # Blue for spatial
            link_labels.append(f"Spatial: {source['spatial_weight']:.3f}")
            
            # Link to defect match
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            defect_value = source['defect_weight'] * 100
            values.append(defect_value)
            colors.append(f'rgba(255, 99, 132, 0.8)')  # Pink for defect
            link_labels.append(f"Defect: {source['defect_weight']:.3f}")
            
            # Link to attention score
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_value = attention_w * 100
            values.append(attention_value)
            colors.append(f'rgba(75, 192, 192, 0.8)')  # Cyan for attention
            link_labels.append(f"Attention: {attention_w:.3f}")
            
            # Link to combined weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            combined_value = source['combined_weight'] * 100
            values.append(combined_value)
            colors.append(f'rgba(153, 102, 255, 0.8)')  # Purple for combined
            link_labels.append(f"Combined: {source['combined_weight']:.3f}")
        
        # Links from components to target
        for comp_idx, comp_name in enumerate(['Spatial', 'Defect', 'Attention', 'Combined']):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)  # Target node
            
            # Sum of all flows into this component
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values) 
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.5)  # Reduce flow to target for visual clarity
            
            # Use component-specific colors
            if comp_name == 'Spatial':
                colors.append(f'rgba(54, 162, 235, 0.6)')
            elif comp_name == 'Defect':
                colors.append(f'rgba(255, 99, 132, 0.6)')
            elif comp_name == 'Attention':
                colors.append(f'rgba(75, 192, 192, 0.6)')
            else:  # Combined
                colors.append(f'rgba(153, 102, 255, 0.6)')
            
            link_labels.append(f"{comp_name} → Target")
        
        # Create enhanced Sankey diagram - FIXED: Remove hoverinfo and hoverlabel from node dict
        fig = go.Figure(data=[go.Sankey(
            # CORRECTED: hoverinfo and hoverlabel are not valid node properties in Plotly Sankey
            node=dict(
                pad=25,  # Increased padding
                thickness=30,  # Thicker nodes
                line=dict(color="black", width=2),  # Black border for contrast
                label=labels,
                color=["#FF6B6B"] +  # Target color (bright red)
                      [color_palette[i % len(color_palette)] for i in range(len(sources_data))] +  # Source colors
                      ["#36A2EB", "#FF6384", "#4BC0C0", "#9966FF"],  # Component colors
                # REMOVED INVALID PROPERTIES:
                # hoverinfo='label+value',  # ❌ NOT valid for Sankey nodes
                # hoverlabel=dict(...)      # ❌ NOT valid for Sankey nodes
                # Use hovertemplate instead for custom hover text
                hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<extra></extra>',
                customdata=link_labels,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            # Set hoverinfo at trace level if needed
            hoverinfo='all'
        )])
        
        # Enhanced layout with better fonts and contrast
        fig.update_layout(
            title=dict(
                text=f'<b>SANKEY DIAGRAM: ATTENTION COMPONENT FLOW</b><br>Target: {target_angle}° {target_defect} | σ={spatial_sigma}°',
                font=dict(
                    family=self.font_config['family'],
                    size=self.font_config['size_title'],
                    color=self.font_config['color'],
                    weight='bold'
                ),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            font=dict(
                family=self.font_config['family'],
                size=self.font_config['size_labels'],
                color=self.font_config['color']
            ),
            width=1400,  # Wider for better visibility
            height=900,   # Taller for better visibility
            plot_bgcolor='rgba(240, 240, 245, 0.9)',  # Light background
            paper_bgcolor='white',
            margin=dict(t=100, l=50, r=50, b=50),
            # Move hover configuration to layout level for entire figure
            hoverlabel=dict(
                font=dict(
                    family=self.font_config['family'],
                    size=self.font_config['size_labels'],
                    color='white'
                ),
                bgcolor='rgba(44, 62, 80, 0.9)',  # Dark background for hover
                bordercolor='white'
            )
        )
        
        # Add annotations for color coding
        annotations = [
            dict(
                x=0.02, y=1.05,
                xref='paper', yref='paper',
                text='<b>COLOR CODING:</b>',
                showarrow=False,
                font=dict(size=14, color='darkblue', weight='bold')
            ),
            dict(
                x=0.02, y=1.02,
                xref='paper', yref='paper',
                text='• Spatial: <span style="color:#36A2EB">█</span>',
                showarrow=False,
                font=dict(size=12, color='#36A2EB')
            ),
            dict(
                x=0.15, y=1.02,
                xref='paper', yref='paper',
                text='• Defect: <span style="color:#FF6384">█</span>',
                showarrow=False,
                font=dict(size=12, color='#FF6384')
            ),
            dict(
                x=0.28, y=1.02,
                xref='paper', yref='paper',
                text='• Attention: <span style="color:#4BC0C0">█</span>',
                showarrow=False,
                font=dict(size=12, color='#4BC0C0')
            ),
            dict(
                x=0.41, y=1.02,
                xref='paper', yref='paper',
                text='• Combined: <span style="color:#9966FF">█</span>',
                showarrow=False,
                font=dict(size=12, color='#9966FF')
            )
        ]
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    
    def create_enhanced_chord_diagram(self, sources_data, target_angle, target_defect):
        """
        Create enhanced chord diagram with target at center and weight components in different colors
        """
        # Create circular layout
        n_sources = len(sources_data)
        center_x, center_y = 0, 0
        radius = 1.5
        
        # Calculate positions for sources in a circle
        source_positions = []
        for i in range(n_sources):
            angle = 2 * pi * i / n_sources
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            source_positions.append((x, y))
        
        # Create figure
        fig = go.Figure()
        
        # Add center target node (larger and highlighted)
        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[center_y],
            mode='markers+text',
            name='Target',
            marker=dict(
                size=50,
                color=self.color_scheme['Query'],
                symbol='star',
                line=dict(width=3, color='white')
            ),
            text=[f'Target\n{target_defect}\n{target_angle}°'],
            textposition="middle center",
            textfont=dict(
                size=16,
                color='white',
                weight='bold'
            ),
            hoverinfo='text',
            hovertext=f'Target: {target_defect} at {target_angle}°'
        ))
        
        # Add source nodes in circle
        source_x = []
        source_y = []
        source_text = []
        source_colors = []
        source_sizes = []
        
        for i, source in enumerate(sources_data):
            x, y = source_positions[i]
            source_x.append(x)
            source_y.append(y)
            
            # Node color based on defect type
            defect_color = self.color_scheme.get(source['defect_type'], '#CCCCCC')
            source_colors.append(defect_color)
            
            # Node size based on combined weight
            node_size = 20 + source['combined_weight'] * 60
            source_sizes.append(node_size)
            
            # Node text
            source_text.append(
                f"Source {i}<br>"
                f"Defect: {source['defect_type']}<br>"
                f"Angle: {source['theta_deg']:.1f}°<br>"
                f"Combined: {source['combined_weight']:.3f}"
            )
        
        # Add source nodes trace
        fig.add_trace(go.Scatter(
            x=source_x,
            y=source_y,
            mode='markers+text',
            name='Sources',
            marker=dict(
                size=source_sizes,
                color=source_colors,
                line=dict(width=2, color='white')
            ),
            text=[f"S{i}" for i in range(n_sources)],
            textposition="top center",
            textfont=dict(size=12, color='white', weight='bold'),
            hoverinfo='text',
            hovertext=source_text
        ))
        
        # Add connecting lines for each weight component
        for i, source in enumerate(sources_data):
            sx, sy = source_positions[i]
            
            # Calculate control points for curved lines
            cx = (sx + center_x) / 2
            cy = (sy + center_y) / 2
            
            # Normal for perpendicular offset
            dx = center_x - sx
            dy = center_y - sy
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                nx = -dy / length * 0.3
                ny = dx / length * 0.3
            else:
                nx, ny = 0, 0
            
            # Create curved paths for each component
            t = np.linspace(0, 1, 50)
            
            # Spatial weight line (Blue)
            spatial_curve_x = (1-t)**2 * sx + 2*(1-t)*t * (cx + nx*0.5) + t**2 * center_x
            spatial_curve_y = (1-t)**2 * sy + 2*(1-t)*t * (cy + ny*0.5) + t**2 * center_y
            spatial_width = max(1, source['spatial_weight'] * 15)
            
            fig.add_trace(go.Scatter(
                x=spatial_curve_x,
                y=spatial_curve_y,
                mode='lines',
                name=f'Source {i} - Spatial',
                line=dict(
                    width=spatial_width,
                    color='rgba(54, 162, 235, 0.7)',  # Blue
                    dash='solid'
                ),
                hoverinfo='text',
                hovertext=f"Spatial Weight: {source['spatial_weight']:.3f}",
                showlegend=False
            ))
            
            # Defect weight line (Pink)
            defect_curve_x = (1-t)**2 * sx + 2*(1-t)*t * (cx + nx*0) + t**2 * center_x
            defect_curve_y = (1-t)**2 * sy + 2*(1-t)*t * (cy + ny*0) + t**2 * center_y
            defect_width = max(1, source['defect_weight'] * 15)
            
            fig.add_trace(go.Scatter(
                x=defect_curve_x,
                y=defect_curve_y,
                mode='lines',
                name=f'Source {i} - Defect',
                line=dict(
                    width=defect_width,
                    color='rgba(255, 99, 132, 0.7)',  # Pink
                    dash='dot'
                ),
                hoverinfo='text',
                hovertext=f"Defect Weight: {source['defect_weight']:.3f}",
                showlegend=False
            ))
            
            # Attention weight line (Cyan)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_curve_x = (1-t)**2 * sx + 2*(1-t)*t * (cx - nx*0.5) + t**2 * center_x
            attention_curve_y = (1-t)**2 * sy + 2*(1-t)*t * (cy - ny*0.5) + t**2 * center_y
            attention_width = max(1, attention_w * 15)
            
            fig.add_trace(go.Scatter(
                x=attention_curve_x,
                y=attention_curve_y,
                mode='lines',
                name=f'Source {i} - Attention',
                line=dict(
                    width=attention_width,
                    color='rgba(75, 192, 192, 0.7)',  # Cyan
                    dash='dash'
                ),
                hoverinfo='text',
                hovertext=f"Attention Weight: {attention_w:.3f}",
                showlegend=False
            ))
            
            # Combined weight line (Purple) - Thicker and on top
            combined_curve_x = (1-t)**2 * sx + 2*(1-t)*t * (cx - nx*1.0) + t**2 * center_x
            combined_curve_y = (1-t)**2 * sy + 2*(1-t)*t * (cy - ny*1.0) + t**2 * center_y
            combined_width = max(2, source['combined_weight'] * 20)
            
            fig.add_trace(go.Scatter(
                x=combined_curve_x,
                y=combined_curve_y,
                mode='lines',
                name=f'Source {i} - Combined',
                line=dict(
                    width=combined_width,
                    color='rgba(153, 102, 255, 0.8)',  # Purple
                    dash='solid'
                ),
                hoverinfo='text',
                hovertext=f"Combined Weight: {source['combined_weight']:.3f}",
                showlegend=False
            ))
        
        # Update layout for enhanced chord diagram
        fig.update_layout(
            title=dict(
                text=f'<b>ENHANCED CHORD DIAGRAM: WEIGHT COMPONENT VISUALIZATION</b><br>Target: {target_angle}° {target_defect}',
                font=dict(
                    family=self.font_config['family'],
                    size=self.font_config['size_title'],
                    color=self.font_config['color'],
                    weight='bold'
                ),
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
                font=dict(
                    size=self.font_config['size_labels'],
                    family=self.font_config['family']
                ),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1200,
            height=1000,
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='white',
            hovermode='closest',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-2.5, 2.5]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-2.5, 2.5]
            )
        )
        
        # Add legend annotations for line types
        annotations = [
            dict(
                x=0.02, y=1.05,
                xref='paper', yref='paper',
                text='<b>LINE TYPES:</b>',
                showarrow=False,
                font=dict(size=14, color='darkblue', weight='bold')
            ),
            dict(
                x=0.02, y=1.02,
                xref='paper', yref='paper',
                text='<span style="color:#36A2EB">━━━━</span> Spatial Kernel',
                showarrow=False,
                font=dict(size=12, color='#36A2EB')
            ),
            dict(
                x=0.02, y=1.0,
                xref='paper', yref='paper',
                text='<span style="color:#FF6384">⸺⸺⸺</span> Defect Match',
                showarrow=False,
                font=dict(size=12, color='#FF6384')
            ),
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text='<span style="color:#4BC0C0">- - - -</span> Attention Score',
                showarrow=False,
                font=dict(size=12, color='#4BC0C0')
            ),
            dict(
                x=0.02, y=0.96,
                xref='paper', yref='paper',
                text='<span style="color:#9966FF">━━━━</span> Combined Weight',
                showarrow=False,
                font=dict(size=12, color='#9966FF', weight='bold')
            )
        ]
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def create_hierarchical_radar_chart(self, sources_data, target_angle, target_defect, spatial_sigma):
        """Create hierarchical radar chart with enhanced styling"""
        fig = go.Figure()
        
        # Group by defect type
        defect_groups = {}
        for source in sources_data:
            defect = source['defect_type']
            if defect not in defect_groups:
                defect_groups[defect] = []
            defect_groups[defect].append(source)
        
        # Use vibrant colormap
        color_palette = self.get_colormap('plasma', len(defect_groups))
        
        # Create traces for each defect type
        for idx, (defect, sources) in enumerate(defect_groups.items()):
            angles = [s['theta_deg'] for s in sources]
            spatial_weights = [s['spatial_weight'] for s in sources]
            attention_weights = [s.get('attention_weight', 0.0) for s in sources]
            combined_weights = [s['combined_weight'] for s in sources]
            
            # Normalize for radar display
            max_weight = max(max(spatial_weights), max(attention_weights), max(combined_weights), 1e-6)
            spatial_norm = [w/max_weight for w in spatial_weights]
            attention_norm = [w/max_weight for w in attention_weights]
            combined_norm = [w/max_weight for w in combined_weights]
            
            # Use colormap colors
            color = color_palette[idx % len(color_palette)]
            
            # Plot spatial weights (outer ring)
            fig.add_trace(go.Scatterpolar(
                r=spatial_norm,
                theta=angles,
                mode='lines+markers',
                name=f'{defect} - Spatial',
                line=dict(color=color, width=3, dash='dash'),
                marker=dict(size=10, color=color),
                hoverinfo='text',
                text=[f'{defect}<br>Angle: {a}°<br>Spatial: {sw:.3f}' 
                      for a, sw in zip(angles, spatial_weights)]
            ))
            
            # Plot attention weights (middle ring)
            fig.add_trace(go.Scatterpolar(
                r=attention_norm,
                theta=angles,
                mode='lines+markers',
                name=f'{defect} - Attention',
                line=dict(color=color, width=3, dash='dot'),
                marker=dict(size=10, color=color),
                hoverinfo='text',
                text=[f'{defect}<br>Angle: {a}°<br>Attention: {aw:.3f}' 
                      for a, aw in zip(angles, attention_weights)]
            ))
            
            # Plot combined weights (inner ring)
            fig.add_trace(go.Scatterpolar(
                r=combined_norm,
                theta=angles,
                mode='lines+markers',
                name=f'{defect} - Combined',
                line=dict(color=color, width=4),
                marker=dict(size=12, color=color),
                hoverinfo='text',
                text=[f'{defect}<br>Angle: {a}°<br>Combined: {cw:.3f}<br>Defect Match: {s["target_defect_match"]}' 
                      for a, cw, s in zip(angles, combined_weights, sources)]
            ))
        
        # Add target angle line with high visibility
        fig.add_trace(go.Scatterpolar(
            r=[0, 1],
            theta=[target_angle, target_angle],
            mode='lines',
            name='Target Angle',
            line=dict(color='#FF0000', width=5),
            hoverinfo='text',
            hovertext=f'Target Angle: {target_angle}°'
        ))
        
        # Update layout with enhanced styling
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=14, family=self.font_config['family']),
                    title=dict(
                        text='Normalized Weight',
                        font=dict(size=16, family=self.font_config['family'], weight='bold')
                    )
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90,
                    tickfont=dict(size=14, family=self.font_config['family'])
                ),
                bgcolor='rgba(240, 240, 245, 0.5)'
            ),
            title=dict(
                text=f'<b>HIERARCHICAL RADAR CHART</b><br>Target: {target_angle}° {target_defect} | σ={spatial_sigma}°',
                font=dict(
                    family=self.font_config['family'],
                    size=self.font_config['size_title'],
                    color=self.font_config['color'],
                    weight='bold'
                ),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(
                    size=self.font_config['size_labels'],
                    family=self.font_config['family']
                )
            ),
            width=1200,
            height=900,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        return fig
    
    def create_weight_formula_breakdown(self, sources_data, target_angle, target_defect, spatial_sigma):
        """Create comprehensive weight formula breakdown visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Weight Component Breakdown</b>',
                '<b>Cumulative Weight Distribution</b>',
                '<b>Defect Type Analysis</b>',
                '<b>Angular Weight Distribution</b>'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Sort sources by angle
        sources_data = sorted(sources_data, key=lambda x: x['theta_deg'])
        
        angles = [s['theta_deg'] for s in sources_data]
        source_indices = [s['source_index'] for s in sources_data]
        
        # Prepare data
        spatial_values = [s['spatial_weight'] for s in sources_data]
        attention_values = [s.get('attention_weight', 0.0) for s in sources_data]
        defect_values = [s['defect_weight'] for s in sources_data]
        combined_values = [s['combined_weight'] for s in sources_data]
        
        # Plot 1: Component breakdown (stacked bar)
        fig.add_trace(go.Bar(
            x=source_indices,
            y=spatial_values,
            name='Spatial Kernel',
            marker_color='#36A2EB',
            text=[f'{v:.3f}' for v in spatial_values],
            textposition='outside',
            textfont=dict(size=10)
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices,
            y=attention_values,
            name='Attention Score',
            marker_color='#4BC0C0',
            text=[f'{v:.3f}' for v in attention_values],
            textposition='outside',
            textfont=dict(size=10)
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices,
            y=defect_values,
            name='Defect Match',
            marker_color='#FF6384',
            text=[f'{v:.3f}' for v in defect_values],
            textposition='outside',
            textfont=dict(size=10)
        ), row=1, col=1)
        
        # Plot 2: Cumulative distribution
        sorted_weights = np.sort(combined_values)[::-1]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights) if np.sum(sorted_weights) > 0 else np.zeros_like(sorted_weights)
        x_vals = np.arange(1, len(cumulative) + 1)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative Weight',
            line=dict(color='#9966FF', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(153, 102, 255, 0.2)'
        ), row=1, col=2)
        
        # Add 90% threshold
        if len(cumulative) > 0 and np.sum(sorted_weights) > 0:
            threshold_idx = np.where(cumulative >= 0.9)[0]
            if len(threshold_idx) > 0:
                fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                             annotation_text="90% threshold", row=1, col=2)
                fig.add_vline(x=threshold_idx[0]+1, line_dash="dash", line_color="red", row=1, col=2)
        
        # Plot 3: Defect type analysis
        defect_counts = {}
        defect_weights = {}
        for source in sources_data:
            defect = source['defect_type']
            if defect not in defect_counts:
                defect_counts[defect] = 0
                defect_weights[defect] = 0.0
            defect_counts[defect] += 1
            defect_weights[defect] += source['combined_weight']
        
        fig.add_trace(go.Bar(
            x=list(defect_counts.keys()),
            y=list(defect_counts.values()),
            name='Count by Defect',
            marker_color=[self.color_scheme[d] for d in defect_counts.keys()],
            text=[f'{c} sources' for c in defect_counts.values()],
            textposition='auto'
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=list(defect_weights.keys()),
            y=list(defect_weights.values()),
            name='Weight by Defect',
            marker_color=[self.color_scheme[d] for d in defect_weights.keys()],
            text=[f'{w:.3f}' for w in defect_weights.values()],
            textposition='auto'
        ), row=2, col=1)
        
        # Plot 4: Angular distribution with different symbols for weight types
        # Combined weights
        fig.add_trace(go.Scatter(
            x=angles,
            y=combined_values,
            mode='markers+lines',
            name='Combined Weight',
            line=dict(color='#9966FF', width=3),
            marker=dict(
                size=15,
                color='#9966FF',
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            text=[f"S{i}: {d}°<br>Combined: {w:.3f}" for i, d, w in zip(source_indices, angles, combined_values)]
        ), row=2, col=2)
        
        # Spatial weights
        fig.add_trace(go.Scatter(
            x=angles,
            y=spatial_values,
            mode='markers',
            name='Spatial Weight',
            marker=dict(
                size=10,
                color='#36A2EB',
                symbol='square',
                line=dict(width=1, color='white')
            ),
            text=[f"S{i}: {d}°<br>Spatial: {w:.3f}" for i, d, w in zip(source_indices, angles, spatial_values)]
        ), row=2, col=2)
        
        # Attention weights
        fig.add_trace(go.Scatter(
            x=angles,
            y=attention_values,
            mode='markers',
            name='Attention Weight',
            marker=dict(
                size=10,
                color='#4BC0C0',
                symbol='diamond',
                line=dict(width=1, color='white')
            ),
            text=[f"S{i}: {d}°<br>Attention: {w:.3f}" for i, d, w in zip(source_indices, angles, attention_values)]
        ), row=2, col=2)
        
        # Defect weights
        fig.add_trace(go.Scatter(
            x=angles,
            y=defect_values,
            mode='markers',
            name='Defect Weight',
            marker=dict(
                size=10,
                color='#FF6384',
                symbol='triangle-up',
                line=dict(width=1, color='white')
            ),
            text=[f"S{i}: {d}°<br>Defect: {w:.3f}" for i, d, w in zip(source_indices, angles, defect_values)]
        ), row=2, col=2)
        
        # Add habit plane reference
        fig.add_vline(x=54.7, line_dash="dot", line_color="green", 
                     annotation_text="Habit Plane (54.7°)", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=f'<b>WEIGHT FORMULA ANALYSIS</b><br>wᵢ(θ*) = [ᾱᵢ(θ*)·exp(-(Δφᵢ)²/(2σ²))·𝟙(τᵢ=τ*)]/Σ[...] + 10⁻⁶<br>σ={spatial_sigma}°, Target: {target_defect}',
                font=dict(
                    family=self.font_config['family'],
                    size=20,
                    color=self.font_config['color'],
                    weight='bold'
                ),
                x=0.5,
                y=0.98
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
            width=1400,
            height=1000,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Source Index", row=1, col=1, title_font=dict(size=14))
        fig.update_yaxes(title_text="Weight Component Value", row=1, col=1, title_font=dict(size=14))
        fig.update_xaxes(title_text="Number of Top Sources", row=1, col=2, title_font=dict(size=14))
        fig.update_yaxes(title_text="Cumulative Weight", row=1, col=2, title_font=dict(size=14))
        fig.update_xaxes(title_text="Defect Type", row=2, col=1, title_font=dict(size=14))
        fig.update_yaxes(title_text="Count / Weight", row=2, col=1, title_font=dict(size=14))
        fig.update_xaxes(title_text="Angle (degrees)", row=2, col=2, title_font=dict(size=14))
        fig.update_yaxes(title_text="Weight Value", row=2, col=2, title_font=dict(size=14))
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Advanced Weight Analysis - Angular Bracketing Theory",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #EF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2.2rem !important;
        color: #2C3E50 !important;
        font-weight: 800 !important;
        border-left: 8px solid #3B82F6;
        padding-left: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(to right, #F0F9FF, white);
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
    }
    .formula-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 5px solid #3B82F6;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.3rem;
        line-height: 1.8;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left: 5px solid #EF4444;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        font-size: 1.2rem;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 3rem;
        background: linear-gradient(90deg, #F8FAFC, #EFF6FF);
        padding: 0.5rem 1rem;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0 0;
        padding: 15px 20px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #64748B;
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E0F2FE;
        border-color: #3B82F6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: 700;
        border-color: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }
    .colormap-preview {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 8px;
        border-radius: 4px;
        vertical-align: middle;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 ADVANCED WEIGHT ANALYSIS - ANGULAR BRACKETING THEORY</h1>', unsafe_allow_html=True)
    
    # Domain information - FIXED: Use proper string formatting
    domain_info = DomainConfiguration.get_domain_info()
    domain_length = domain_info['domain_length_nm']
    grid_points = domain_info['grid_points']
    grid_spacing = domain_info['grid_spacing_nm']
    domain_half = domain_info['domain_half_nm']
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; background: linear-gradient(135deg, #E0F7FA, #E3F2FD); padding: 1.5rem; border-radius: 15px; border: 2px solid #3B82F6;">
    <h3 style="color: #1E3A8A; margin-bottom: 0.5rem;">Domain Configuration</h3>
    <p style="color: #4B5563; font-size: 1.1rem; margin: 0.25rem;">
    <strong>Size:</strong> {domain_length} nm × {domain_length} nm | 
    <strong>Grid:</strong> {grid_points} × {grid_points} points | 
    <strong>Spacing:</strong> {grid_spacing} nm | 
    <strong>Extent:</strong> ±{domain_half} nm
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Weight formula display
    st.markdown(r"""
    <div class="formula-box">
    <strong style="font-size: 1.5rem;">🎯 ATTENTION WEIGHT FORMULA:</strong><br><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
    
    <br><strong style="font-size: 1.3rem;">📊 COMPONENTS:</strong><br>
    • <span style="color:#4BC0C0">$\bar{\alpha}_i$</span>: Learned attention score from transformer<br>
    • <span style="color:#36A2EB">$\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$</span>: Spatial kernel (angular bracketing)<br>
    • <span style="color:#FF6384">$\mathbb{I}(\tau_i = \tau^*)$</span>: Defect type indicator (1 if match, 0 otherwise)<br>
    • <span style="color:#9966FF">$\sigma$</span>: Angular kernel width (controllable parameter)
    </div>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown(f"""
    <div class="info-box">
    <strong style="font-size: 1.3rem;">🔬 PHYSICS-AWARE WEIGHT ANALYSIS: ENHANCED VISUALIZATION</strong><br><br>
    • <strong>Enhanced Sankey Diagrams:</strong> High-contrast, large-font visualization of attention component flow<br>
    • <strong>Advanced Chord Diagrams:</strong> Circular layout with target at center, weight components in different colors<br>
    • <strong>Hierarchical Radar Charts:</strong> Angular variation (Tier 1) × Defect type (Tier 2) with 50+ colormaps<br>
    • <strong>Weight Formula Breakdown:</strong> Detailed component analysis with enhanced readability<br>
    • <strong>Defect Type Gating:</strong> Hard constraint for physical validity, visualized in all diagrams<br>
    • <strong>Domain Size:</strong> {domain_length} nm × {domain_length} nm centered at 0 (±{domain_half} nm)
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator(
            spatial_sigma=10.0,
            locality_weight_factor=0.5
        )
    if 'weight_visualizer' not in st.session_state:
        st.session_state.weight_visualizer = WeightVisualizer()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ CONFIGURATION</h2>', unsafe_allow_html=True)
        
        # Domain information
        st.markdown("#### 📐 DOMAIN INFORMATION")
        st.info(f"""
        **Grid:** {grid_points} × {grid_points} points
        **Spacing:** {grid_spacing} nm
        **Size:** {domain_length} nm × {domain_length} nm
        **Extent:** ±{domain_half} nm
        **Area:** {domain_info['area_nm2']:.1f} nm²
        """)
        
        # Data loading
        st.markdown("#### 📂 DATA MANAGEMENT")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 LOAD SOLUTIONS", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    if st.session_state.solutions:
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                    else:
                        st.warning("No solutions found in directory")
        with col2:
            if st.button("🧹 CLEAR CACHE", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.success("Cache cleared")
        st.divider()
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 TARGET PARAMETERS</h2>', unsafe_allow_html=True)
        col_angle1, col_angle2 = st.columns([2, 1])
        with col_angle1:
            custom_theta = st.number_input(
                "Target Angle θ (degrees)",
                min_value=0.0,
                max_value=180.0,
                value=54.7,
                step=0.1,
                format="%.1f",
                help="Angle in degrees (0° to 180°). Default habit plane is 54.7°"
            )
        with col_angle2:
            st.markdown("###")
            if st.button("HABIT PLANE", use_container_width=True):
                custom_theta = 54.7
                st.rerun()
        
        # Defect type
        defect_type = st.selectbox(
            "Defect Type",
            options=['ISF', 'ESF', 'Twin', 'No Defect'],
            index=2,
            help="Type of crystal defect to simulate"
        )
        
        # Spatial sigma parameter
        st.markdown("#### 📐 ANGULAR BRACKETING KERNEL")
        spatial_sigma = st.slider(
            "Kernel Width σ (degrees)",
            min_value=1.0,
            max_value=45.0,
            value=10.0,
            step=0.5,
            help="Width of Gaussian angular bracketing window"
        )
        
        # Colormap selection
        st.markdown("#### 🎨 COLORMAP SELECTION")
        selected_colormap = st.selectbox(
            "Choose Visualization Colormap",
            options=ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('viridis'),
            help="Select from 50+ colormaps for enhanced visualization"
        )
        
        # Preview colormap
        if selected_colormap:
            try:
                cmap = plt.get_cmap(selected_colormap)
                colors_html = ""
                for i in range(10):
                    r, g, b, _ = cmap(i/9)
                    colors_html += f'<span class="colormap-preview" style="background-color: rgb({int(r*255)}, {int(g*255)}, {int(b*255)});"></span>'
                st.markdown(f"**Preview:** {colors_html}", unsafe_allow_html=True)
            except:
                pass
        
        # Run interpolation
        st.markdown("#### 🚀 INTERPOLATION CONTROL")
        if st.button("🎯 PERFORM THEORY-INFORMED INTERPOLATION", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing interpolation with Angular Bracketing Theory..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': PhysicsParameters.get_eigenstrain(defect_type),
                        'kappa': 0.6,
                        'theta': np.radians(custom_theta),
                        'shape': 'Square'
                    }
                    
                    # Perform interpolation
                    result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        custom_theta,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        st.session_state.transformer_interpolator.set_spatial_parameters(spatial_sigma=spatial_sigma)
                        st.success(f"Interpolation successful! Theory-Informed Attention applied with σ={spatial_sigma}°")
                    else:
                        st.error("Interpolation failed. Check console for errors.")
    
    # Main content area
    if st.session_state.solutions:
        st.markdown(f"### 📊 LOADED {len(st.session_state.solutions)} SOLUTIONS")
        
        # Display source information
        if st.session_state.solutions:
            source_thetas = []
            source_defects = []
            for sol in st.session_state.solutions:
                if 'params' in sol and 'theta' in sol['params']:
                    theta_deg = np.degrees(sol['params']['theta']) % 360
                    source_thetas.append(theta_deg)
                if 'params' in sol and 'defect_type' in sol['params']:
                    source_defects.append(sol['params']['defect_type'])
            
            if source_thetas:
                st.markdown(f"**Source Angles Range:** {min(source_thetas):.1f}° to {max(source_thetas):.1f}°")
                st.markdown(f"**Mean Source Angle:** {np.mean(source_thetas):.1f}°")
            if source_defects:
                defect_counts = {}
                for defect in source_defects:
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
                st.markdown("**Defect Types:** " + ", ".join([f"{k}: {v}" for k, v in defect_counts.items()]))
    
    # Results display with all enhanced visualizations
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 RESULTS OVERVIEW",
            "🌀 ENHANCED SANKEY", 
            "🔗 ADVANCED CHORD",
            "🕸️ HIERARCHICAL RADAR",
            "📊 FORMULA BREAKDOWN"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 INTERPOLATION RESULTS</h2>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">Max Von Mises<br>' + f"{np.max(result['fields']['von_mises']):.3f} GPa</div>", unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">Target Angle<br>' + f"{result['target_angle']:.1f}°</div>", unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">Defect Type<br>' + f"{result['target_params']['defect_type']}</div>", unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">Attention Entropy<br>' + f"{result['weights']['entropy']:.3f}</div>", unsafe_allow_html=True)
            
            # Weight table
            st.markdown("#### 📋 WEIGHT COMPONENTS TABLE")
            if 'sources_data' in result:
                df = pd.DataFrame(result['sources_data'])
                st.dataframe(
                    df[['source_index', 'theta_deg', 'defect_type', 'spatial_weight', 
                        'defect_weight', 'attention_weight', 'combined_weight', 'target_defect_match']].style
                    .background_gradient(subset=['combined_weight'], cmap=selected_colormap)
                    .format({
                        'theta_deg': '{:.1f}',
                        'spatial_weight': '{:.4f}',
                        'defect_weight': '{:.4f}',
                        'attention_weight': '{:.4f}',
                        'combined_weight': '{:.4f}'
                    })
                )
                
        with tab2:
            st.markdown('<h2 class="section-header">🌀 ENHANCED SANKEY DIAGRAM</h2>', unsafe_allow_html=True)
            st.markdown("""
            **ENHANCED VISUALIZATION FEATURES:**
            - **Large Font Sizes:** Improved readability with 24pt titles and 18pt labels
            - **High Contrast Colors:** Vibrant color scheme with distinct component colors
            - **Enhanced Flow Visualization:** Thicker nodes and links for better visibility
            - **Interactive Hover:** Detailed information on hover with dark backgrounds
            - **Color Coding Legend:** Clear explanation of color meanings
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_enhanced_sankey_diagram(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type'],
                    spatial_sigma
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.markdown('<h2 class="section-header">🔗 ADVANCED CHORD DIAGRAM</h2>', unsafe_allow_html=True)
            st.markdown("""
            **CIRCULAR LAYOUT FEATURES:**
            - **Target at Center:** Central positioning for focus
            - **Weight Components in Different Colors:** Spatial (Blue), Defect (Pink), Attention (Cyan), Combined (Purple)
            - **Line Thickness Proportional to Weight:** Visual representation of importance
            - **Curved Connection Lines:** Aesthetic visualization of relationships
            - **Different Line Styles:** Dashed, dotted, and solid lines for component distinction
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_enhanced_chord_diagram(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type']
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab4:
            st.markdown('<h2 class="section-header">🕸️ HIERARCHICAL RADAR CHART</h2>', unsafe_allow_html=True)
            st.markdown("""
            **HIERARCHICAL VISUALIZATION:**
            - **Tier 1 (Angular Variation):** Outer ring shows source angles
            - **Tier 2 (Weight Components):** Multiple rings for different weight types
            - **Colormap Integration:** Uses selected colormap for defect type coloring
            - **Target Highlight:** Red line indicates target angle
            - **Interactive Exploration:** Hover for detailed weight information
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_hierarchical_radar_chart(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type'],
                    spatial_sigma
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab5:
            st.markdown('<h2 class="section-header">📊 WEIGHT FORMULA BREAKDOWN</h2>', unsafe_allow_html=True)
            st.markdown("""
            **COMPREHENSIVE ANALYSIS:**
            - **Component Breakdown:** Stacked bars showing contribution of each weight component
            - **Cumulative Distribution:** Shows how weights accumulate across sources
            - **Defect Type Analysis:** Distribution of weights by defect type
            - **Angular Distribution:** Weight variation with angle, different symbols for weight types
            - **Habit Plane Reference:** Green line at 54.7° for reference
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_weight_formula_breakdown(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type'],
                    spatial_sigma
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Physics interpretation section
        st.markdown("---")
        st.markdown("### 🧪 PHYSICS INTERPRETATION OF WEIGHT FORMULA")
        
        col_phys1, col_phys2 = st.columns(2)
        
        with col_phys1:
            st.markdown("""
            **DEFECT TYPE AS HARD CONSTRAINT:**
            - Sources with different defect types receive near-zero attention
            - Critical for physical validity
            - Different defect types have fundamentally different stress fields
            
            **ANGULAR PROXIMITY DRIVES ATTENTION:**
            - Gaussian kernel creates "bracketing window"
            - Sources within ±σ degrees of target receive highest weights
            - Bracketing structure optimal for interpolation
            """)
        
        with col_phys2:
            st.markdown(f"""
            **LEARNED SIMILARITY REFINES SELECTION:**
            - Transformer captures subtle stress field patterns
            - Modulates the physics priors rather than replacing them
            - Enables interpolation beyond simple angular proximity
            
            **DOMAIN SIZE AWARENESS:**
            - All visualizations reference the **{domain_length} nm** domain
            - Angular positions mapped to physical domain coordinates
            - Stress fields computed on exact grid spacing
            """)
    
    else:
        # Welcome message when no results
        st.markdown(f"""
        ## 🎯 WELCOME TO ADVANCED WEIGHT ANALYSIS
        
        ### DOMAIN CONFIGURATION:
        - **Size:** {domain_length} nm × {domain_length} nm
        - **Grid:** {grid_points} × {grid_points} points
        - **Spacing:** {grid_spacing} nm
        - **Extent:** ±{domain_half} nm
        
        ### GETTING STARTED:
        1. **Load Solutions** from the sidebar
        2. **Configure Target Parameters** (angle, defect type)
        3. **Set Angular Bracketing Parameters** (kernel width σ)
        4. **Select Colormap** from 50+ options
        5. **Click "Perform Theory-Informed Interpolation"** to run
        6. **Explore Enhanced Visualizations** across all tabs
        
        ### KEY FEATURES:
        - **Enhanced Sankey Diagrams:** High-contrast, large-font visualization
        - **Advanced Chord Diagrams:** Circular layout with target at center
        - **Hierarchical Radar Charts:** Angular variation × Defect type
        - **Weight Formula Breakdown:** Detailed component analysis
        - **Defect Type Gating:** Hard constraint for physical validity
        
        ### WEIGHT FORMULA VISUALIZATION:
        The attention weight formula implements **Angular Bracketing Theory**:
        - Defect type as hard constraint
        - Angular proximity as spatial locality kernel
        - Learned transformer attention as refinement
        - Combined weights ensure physics-aware interpolation
        """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
