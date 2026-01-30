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
import warnings
import re
warnings.filterwarnings('ignore')

# =============================================
# GLOBAL STYLING CONFIGURATION
# =============================================
# Publication quality styling
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

# Enhanced colormap options with publication standards
COLORMAP_OPTIONS = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'hot', 'afmhot', 'gist_heat',
                   'copper', 'summer', 'Wistia', 'spring', 'autumn', 'winter', 'bone', 'gray', 'pink',
                   'gist_gray', 'gist_yarg', 'binary', 'gist_earth', 'terrain', 'ocean', 'gist_stern', 'gnuplot',
                   'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
                   'gist_ncar', 'hsv', 'tab20c', 'tab20b', 'Set3', 'Set2', 'Set1', 'tab10', 'Pastel2', 'Pastel1',
                   'Paired', 'Accent', 'Dark2', 'tab20', 'flag', 'prism'],
    'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr',
                  'RdGy', 'RdYlGn', 'Spectral_r', 'coolwarm_r', 'bwr_r', 'seismic_r', 'twilight', 'twilight_shifted',
                  'hsv_r', 'gist_rainbow_r', 'rainbow_r', 'jet_r', 'nipy_spectral_r', 'gist_ncar_r'],
    'Qualitative': ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                    'Paired', 'Accent', 'Dark2', 'Set3_r', 'Set2_r', 'Set1_r', 'tab20_r', 'tab10_r',
                    'tab20c_r', 'tab20b_r'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'twilight_shifted',
                              'turbo', 'viridis_r', 'plasma_r', 'inferno_r', 'magma_r', 'cividis_r', 'turbo_r'],
    'Publication Standard': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'RdBu', 'RdBu_r', 'Spectral',
                             'coolwarm', 'bwr', 'seismic', 'BrBG', 'PiYG', 'PRGn', 'PuOr'],
    'Rainbow Family': ['rainbow', 'gist_rainbow', 'nipy_spectral', 'gist_ncar', 'hsv', 'jet', 'turbo',
                       'rainbow_r', 'gist_rainbow_r', 'nipy_spectral_r', 'gist_ncar_r', 'hsv_r', 'jet_r', 'turbo_r'],
    'Geographical': ['terrain', 'ocean', 'gist_earth', 'gist_stern', 'brg', 'CMRmap', 'cubehelix',
                     'terrain_r', 'ocean_r', 'gist_earth_r', 'gist_stern_r', 'brg_r', 'CMRmap_r', 'cubehelix_r'],
    'Temperature': ['hot', 'afmhot', 'gist_heat', 'coolwarm', 'bwr', 'seismic',
                    'hot_r', 'afmhot_r', 'gist_heat_r', 'coolwarm_r', 'bwr_r', 'seismic_r'],
    'Grayscale': ['gray', 'bone', 'pink', 'binary', 'gist_gray', 'gist_yarg',
                  'gray_r', 'bone_r', 'pink_r', 'binary_r', 'gist_gray_r', 'gist_yarg_r']
}

# FIXED: Properly indented block
ALL_COLORMAPS = []
for category in COLORMAP_OPTIONS.values():
    ALL_COLORMAPS.extend(category)
ALL_COLORMAPS = sorted(list(set(ALL_COLORMAPS)))

# =============================================
# DOMAIN SIZE CONFIGURATION - 12.8 nm × 12.8 nm
# =============================================
class DomainConfiguration:
    """Configuration for the 12.8 nm × 12.8 nm simulation domain"""
    # Domain parameters
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
# PHYSICS PARAMETERS
# =============================================
class PhysicsParameters:
    EIGENSTRAIN_VALUES = {
        'Twin': 2.12,
        'ISF': 0.289,
        'ESF': 0.333,
        'No Defect': 0.0
    }

    @staticmethod
    def get_eigenstrain(defect_type: str) -> float:
        return PhysicsParameters.EIGENSTRAIN_VALUES.get(defect_type, 0.0)

# =============================================
# ENHANCED SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self.cache = {}

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
# POSITIONAL ENCODING
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
# ENHANCED TRANSFORMER INTERPOLATOR WITH WEIGHT VISUALIZATION
# =============================================
class EnhancedTransformerInterpolator:
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
                        vm = self.compute_von_mises(stress_fields)
                        hydro = self.compute_hydrostatic(stress_fields)
                        mag = np.sqrt(vm**2 + hydro**2)

                        source_fields.append({
                            'von_mises': vm,
                            'sigma_hydro': hydro,
                            'sigma_mag': mag,
                            'source_index': i,
                            'source_params': src['params']
                        })

            if not source_params or not source_fields:
                return None

            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )

            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)

            if source_features.shape[1] != 15 or target_features.shape[1] != 15:
                if source_features.shape[1] < 15:
                    padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                    source_features = torch.cat([source_features, padding], dim=1)
                if target_features.shape[1] < 15:
                    padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                    target_features = torch.cat([target_features, padding], dim=1)

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

            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape

            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += final_attention_weights[i] * fields[component]
                interpolated_fields[component] = interpolated

            source_theta_degrees = []
            for src in source_params:
                theta_rad = src.get('theta', 0.0)
                theta_deg = np.degrees(theta_rad) % 360
                source_theta_degrees.append(theta_deg)

            return {
                'fields': interpolated_fields,
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist(),
                    'learned_attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
                    'entropy': self._calculate_entropy(final_attention_weights)
                },
                'statistics': {
                    'von_mises': {
                        'max': float(np.max(interpolated_fields['von_mises'])),
                        'mean': float(np.mean(interpolated_fields['von_mises'])),
                        'std': float(np.std(interpolated_fields['von_mises'])),
                        'min': float(np.min(interpolated_fields['von_mises']))
                    }
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
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            return (sxx + syy + szz) / 3
        return np.zeros((100, 100))

    def _calculate_entropy(self, weights):
        weights = np.array(weights)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return -np.sum(weights * np.log(weights + 1e-10))

    def prepare_weight_analysis_data(self, result):
        """Prepare data for weight analysis visualizations"""
        sources_data = []
        
        for i, field in enumerate(result['source_fields']):
            sources_data.append({
                'source_index': i,
                'theta_deg': result['source_theta_degrees'][i],
                'angular_dist': result['source_distances'][i],
                'defect_type': field['source_params']['defect_type'],
                'spatial_weight': result['weights']['spatial_kernel'][i],
                'attention_weight': result['weights']['learned_attention'][i],
                'defect_weight': result['weights']['defect_mask'][i],
                'combined_weight': result['weights']['combined'][i],
                'target_defect_match': field['source_params']['defect_type'] == result['target_params']['defect_type']
            })
        
        return sources_data

# =============================================
# ADVANCED WEIGHT VISUALIZER
# =============================================
class WeightVisualizer:
    def __init__(self):
        self.color_scheme = {
            'Twin': '#FF6B6B',
            'ISF': '#4ECDC4',
            'ESF': '#95E1D3',
            'No Defect': '#FFD93D',
            'Query': '#9D4EDD'
        }
    
    def create_hierarchical_radar_chart(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create hierarchical radar chart with:
        Tier 1: Angular variation (outer ring)
        Tier 2: Defect type (inner rings)
        """
        fig = go.Figure()
        
        # Group by defect type
        defect_groups = {}
        for source in sources_data:
            defect = source['defect_type']
            if defect not in defect_groups:
                defect_groups[defect] = []
            defect_groups[defect].append(source)
        
        # Create traces for each defect type
        for defect, sources in defect_groups.items():
            angles = [s['theta_deg'] for s in sources]
            spatial_weights = [s['spatial_weight'] for s in sources]
            attention_weights = [s['attention_weight'] for s in sources]
            combined_weights = [s['combined_weight'] for s in sources]
            
            # Normalize for radar display
            max_weight = max(max(spatial_weights), max(attention_weights), max(combined_weights))
            spatial_norm = [w/max_weight for w in spatial_weights]
            attention_norm = [w/max_weight for w in attention_weights]
            combined_norm = [w/max_weight for w in combined_weights]
            
            # Create circular coordinates
            theta = np.radians(angles)
            
            # Plot spatial weights (outer ring)
            fig.add_trace(go.Scatterpolar(
                r=spatial_norm,
                theta=angles,
                mode='lines+markers',
                name=f'{defect} - Spatial',
                line=dict(color=self.color_scheme[defect], width=2, dash='dash'),
                marker=dict(size=8),
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
                line=dict(color=self.color_scheme[defect], width=2, dash='dot'),
                marker=dict(size=8),
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
                line=dict(color=self.color_scheme[defect], width=3),
                marker=dict(size=10),
                hoverinfo='text',
                text=[f'{defect}<br>Angle: {a}°<br>Combined: {cw:.3f}<br>Defect Match: {s["target_defect_match"]}' 
                      for a, cw, s in zip(angles, combined_weights, sources)]
            ))
        
        # Add target angle line
        fig.add_trace(go.Scatterpolar(
            r=[0, 1],
            theta=[target_angle, target_angle],
            mode='lines',
            name='Target Angle',
            line=dict(color=self.color_scheme['Query'], width=4),
            hoverinfo='none'
        ))
        
        # Add defect type filter visualization
        for defect in defect_groups.keys():
            if defect == target_defect:
                fig.add_trace(go.Scatterpolar(
                    r=[0.9],
                    theta=[target_angle],
                    mode='markers+text',
                    name=f'Target: {defect}',
                    marker=dict(
                        size=20,
                        color=self.color_scheme[defect],
                        symbol='star'
                    ),
                    text=[f'Target: {defect}'],
                    textposition='top center',
                    hoverinfo='text',
                    textfont=dict(size=14, color='white')
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=12),
                    title=dict(text='Normalized Weight', font=dict(size=14))
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90,
                    tickfont=dict(size=12),
                    title=dict(text='Angle (degrees)', font=dict(size=14))
                ),
                bgcolor='#f8f9fa'
            ),
            title=dict(
                text=f'Hierarchical Weight Analysis<br>Target: {target_angle}° {target_defect} | σ={spatial_sigma}°',
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            width=1000,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        return fig
    
    def create_weight_comparison_radar(self, sources_data, target_defect):
        """
        Create dual radar chart comparing weights with and without defect mask
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'polar'}, {'type': 'polar'}]],
            subplot_titles=('Without Defect Mask', 'With Defect Mask')
        )
        
        # Prepare data
        angles = [s['theta_deg'] for s in sources_data]
        spatial_weights = [s['spatial_weight'] for s in sources_data]
        attention_weights = [s['attention_weight'] for s in sources_data]
        combined_weights = [s['combined_weight'] for s in sources_data]
        
        # Calculate weights without defect mask
        weights_without_mask = []
        for i in range(len(sources_data)):
            # Combine spatial and attention weights, ignoring defect
            w = spatial_weights[i] * attention_weights[i]
            weights_without_mask.append(w)
        
        # Normalize
        max_without = max(weights_without_mask) if weights_without_mask else 1
        max_with = max(combined_weights) if combined_weights else 1
        weights_without_norm = [w/max_without for w in weights_without_mask]
        weights_with_norm = [w/max_with for w in combined_weights]
        
        # Colors based on defect match
        colors = []
        for s in sources_data:
            if s['defect_type'] == target_defect:
                colors.append('#2E8B57')  # Green for match
            else:
                colors.append('#DC143C')  # Red for mismatch
        
        # Plot without defect mask
        fig.add_trace(go.Scatterpolar(
            r=weights_without_norm,
            theta=angles,
            mode='markers+lines',
            name='Weight (no defect mask)',
            marker=dict(
                size=12,
                color=colors,
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color='#4682B4'),
            hoverinfo='text',
            text=[f'Angle: {a}°<br>Weight: {w:.3f}<br>Defect: {s["defect_type"]}<br>Match: {s["target_defect_match"]}'
                  for a, w, s in zip(angles, weights_without_mask, sources_data)]
        ), row=1, col=1)
        
        # Plot with defect mask
        fig.add_trace(go.Scatterpolar(
            r=weights_with_norm,
            theta=angles,
            mode='markers+lines',
            name='Weight (with defect mask)',
            marker=dict(
                size=12,
                color=colors,
                symbol='star' if target_defect == 'Twin' else 'diamond',
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color='#FF8C00'),
            hoverinfo='text',
            text=[f'Angle: {a}°<br>Weight: {w:.3f}<br>Defect: {s["defect_type"]}<br>Match: {s["target_defect_match"]}'
                  for a, w, s in zip(angles, combined_weights, sources_data)]
        ), row=1, col=2)
        
        # Update polar layouts
        for col in [1, 2]:
            fig.update_polars(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90,
                    tickfont=dict(size=10)
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            title=dict(
                text=f'Weight Comparison: Impact of Defect Type Gating<br>Target Defect: {target_defect}',
                font=dict(size=18, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            width=1200,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_chord_diagram(self, sources_data, target_angle, target_defect):
        """
        Create chord diagram showing weight relationships
        """
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes for sources and target
        for source in sources_data:
            node_id = f"S{source['source_index']}"
            G.add_node(node_id, 
                      angle=source['theta_deg'],
                      defect=source['defect_type'],
                      weight=source['combined_weight'])
        
        # Add target node
        G.add_node('Target', angle=target_angle, defect=target_defect, weight=1.0)
        
        # Add edges based on weights
        for source in sources_data:
            source_id = f"S{source['source_index']}"
            weight = source['combined_weight']
            
            # Edge to target
            if source['target_defect_match']:
                G.add_edge(source_id, 'Target', 
                          weight=weight,
                          color=self.color_scheme[source['defect_type']])
            
            # Edges to other sources (spatial proximity)
            for other in sources_data:
                if other['source_index'] != source['source_index']:
                    other_id = f"S{other['source_index']}"
                    angular_diff = min(
                        abs(source['theta_deg'] - other['theta_deg']),
                        360 - abs(source['theta_deg'] - other['theta_deg'])
                    )
                    if angular_diff < 30:  # Connect if within 30 degrees
                        spatial_weight = np.exp(-0.5 * (angular_diff / 10) ** 2)
                        G.add_edge(source_id, other_id,
                                  weight=spatial_weight * 0.5,
                                  color='#CCCCCC')
        
        # Convert to plotly
        pos = nx.circular_layout(G)
        
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_color = G.edges[edge].get('color', '#888888')
            edge_width = G.edges[edge].get('weight', 0.1) * 10
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == 'Target':
                node_text.append(f'Target<br>{target_defect}<br>{target_angle}°')
                node_color.append(self.color_scheme['Query'])
                node_size.append(30)
            else:
                source_idx = int(node[1:])
                source = sources_data[source_idx]
                node_text.append(f'S{source_idx}<br>{source["defect_type"]}<br>{source["theta_deg"]}°<br>W:{source["combined_weight"]:.3f}')
                node_color.append(self.color_scheme[source['defect_type']])
                node_size.append(20 + source['combined_weight'] * 40)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            showlegend=False
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=dict(
                text=f'Weight Relationship Chord Diagram<br>Target: {target_angle}° {target_defect}',
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        return fig
    
    def create_weight_formula_breakdown(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create visualization showing weight formula breakdown for each source
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Weight Component Breakdown', 'Cumulative Weight Distribution'),
            vertical_spacing=0.15
        )
        
        # Sort sources by angle
        sources_data = sorted(sources_data, key=lambda x: x['theta_deg'])
        
        angles = [s['theta_deg'] for s in sources_data]
        source_indices = [s['source_index'] for s in sources_data]
        
        # Prepare stacked bar data
        spatial_values = [s['spatial_weight'] for s in sources_data]
        attention_values = [s['attention_weight'] for s in sources_data]
        defect_values = [s['defect_weight'] for s in sources_data]
        combined_values = [s['combined_weight'] for s in sources_data]
        
        # Colors for components
        colors = {
            'spatial': '#4ECDC4',
            'attention': '#FF6B6B',
            'defect': '#95E1D3',
            'combined': '#9D4EDD'
        }
        
        # Plot 1: Component breakdown
        fig.add_trace(go.Bar(
            x=angles,
            y=spatial_values,
            name='Spatial Kernel',
            marker_color=colors['spatial'],
            text=[f'S:{v:.3f}' for v in spatial_values],
            textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=angles,
            y=attention_values,
            name='Attention',
            marker_color=colors['attention'],
            text=[f'A:{v:.3f}' for v in attention_values],
            textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=angles,
            y=defect_values,
            name='Defect Mask',
            marker_color=colors['defect'],
            text=[f'D:{v:.3f}' for v in defect_values],
            textposition='outside'
        ), row=1, col=1)
        
        # Add target angle line
        fig.add_trace(go.Scatter(
            x=[target_angle, target_angle],
            y=[0, max(combined_values) * 1.1],
            mode='lines',
            name='Target Angle',
            line=dict(color='#FF0000', width=2, dash='dash'),
            hoverinfo='text',
            text=f'Target: {target_angle}°'
        ), row=1, col=1)
        
        # Plot 2: Cumulative distribution
        cumulative_weights = np.cumsum(combined_values)
        fig.add_trace(go.Scatter(
            x=angles,
            y=cumulative_weights,
            mode='lines+markers',
            name='Cumulative Weight',
            line=dict(color=colors['combined'], width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(157, 78, 221, 0.2)'
        ), row=2, col=1)
        
        # Add defect type annotations
        for s in sources_data:
            fig.add_annotation(
                x=s['theta_deg'],
                y=cumulative_weights[sources_data.index(s)] + 0.02,
                text=s['defect_type'][0],  # First letter of defect type
                showarrow=False,
                font=dict(size=10, color='white'),
                bgcolor=self.color_scheme[s['defect_type']],
                bordercolor='white',
                borderwidth=1,
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=f'Weight Formula: wᵢ(θ*) = [ᾱᵢ(θ*) · exp(-(Δφᵢ)²/(2σ²)) · 𝟙(τᵢ=τ*)] / Σ[...] + 10⁻⁶<br>σ={spatial_sigma}°, Target: {target_defect}',
                font=dict(size=16, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            width=1200,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Angle (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Weight Component Value", row=1, col=1)
        fig.update_xaxes(title_text="Angle (degrees)", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Weight", row=2, col=1)
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Weight Component Visualization - Angular Bracketing Theory",
        layout="wide",
        page_icon="🎯",
        initial_sidebar_state="expanded"
    )
    
    # FIXED: Correct CSS without syntax errors
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #374151;
        font-weight: 800;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .weight-formula {
        background-color: #F0F7FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .highlight-query {
        background-color: #FFF3CD;
        border: 2px solid #FFC107;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with weight formula
    st.markdown('<h1 class="main-header">🎯 Weight Component Visualization - Angular Bracketing Theory</h1>', unsafe_allow_html=True)
    
    # Weight formula display - FIXED: Correct LaTeX formatting
    st.markdown(r"""
    <div class="weight-formula">
    <strong>Attention Weight Formula:</strong><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_k \bar{\alpha}_k \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
    
    <strong>Components:</strong>
    • $\bar{\alpha}_i$: Learned attention score<br>
    • $\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$: Spatial kernel (angular bracketing)<br>
    • $\mathbb{I}(\tau_i = \tau^*)$: Defect type indicator (1 if match, 0 otherwise)
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedTransformerInterpolator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = WeightVisualizer()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Load data
        if st.button("📂 Load Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions found")
        
        # Target parameters
        st.markdown("#### 🎯 Target Parameters")
        target_angle = st.slider("Target Angle (degrees)", 0.0, 180.0, 54.7, 0.1)
        target_defect = st.selectbox("Target Defect Type", ['Twin', 'ISF', 'ESF', 'No Defect'], index=2)
        spatial_sigma = st.slider("Spatial Kernel Width σ (degrees)", 1.0, 45.0, 10.0, 0.5)
        
        # Run interpolation
        if st.button("🚀 Run Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing interpolation..."):
                    target_params = {
                        'defect_type': target_defect,
                        'eps0': PhysicsParameters.get_eigenstrain(target_defect),
                        'theta': np.radians(target_angle),
                        'shape': 'Square'
                    }
                    
                    result = st.session_state.interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        target_angle,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        st.success("Interpolation completed!")
                    else:
                        st.error("Interpolation failed")
    
    # Main content
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Hierarchical Radar",
            "⚖️ Weight Comparison",
            "🕸️ Chord Diagram",
            "📈 Formula Breakdown"
        ])
        
        # Prepare weight analysis data
        sources_data = st.session_state.interpolator.prepare_weight_analysis_data(result)
        
        with tab1:
            st.markdown('<h3 class="section-header">Hierarchical Radar Chart</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Visualization Strategy:**
            - **Tier 1 (Outer Ring):** Angular variation around the circle
            - **Tier 2 (Inner Rings):** Different weight components for each defect type
            - **Target Highlight:** Red line at target angle, star marker for target defect
            """)
            
            fig1 = st.session_state.visualizer.create_hierarchical_radar_chart(
                sources_data,
                result['target_angle'],
                result['target_params']['defect_type'],
                spatial_sigma
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Analysis
            st.markdown("#### 🔍 Radar Chart Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Target Angle", f"{result['target_angle']:.1f}°")
                st.metric("Target Defect", result['target_params']['defect_type'])
            with col2:
                st.metric("Spatial Sigma", f"{spatial_sigma}°")
                st.metric("Number of Sources", len(sources_data))
        
        with tab2:
            st.markdown('<h3 class="section-header">Weight Comparison: With vs Without Defect Mask</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Comparison Purpose:**
            - **Left Radar:** Shows attention + spatial weights ignoring defect type
            - **Right Radar:** Shows final weights with defect type gating applied
            - **Color Coding:** Green = matching defect type, Red = mismatched defect type
            """)
            
            fig2 = st.session_state.visualizer.create_weight_comparison_radar(
                sources_data,
                result['target_params']['defect_type']
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Statistics
            matching_sources = [s for s in sources_data if s['target_defect_match']]
            mismatching_sources = [s for s in sources_data if not s['target_defect_match']]
            
            st.markdown("#### 📊 Defect Match Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Matching Sources", len(matching_sources))
            with col2:
                st.metric("Mismatching Sources", len(mismatching_sources))
            with col3:
                total_weight_matching = sum(s['combined_weight'] for s in matching_sources)
                st.metric("Total Weight (Matching)", f"{total_weight_matching:.3f}")
        
        with tab3:
            st.markdown('<h3 class="section-header">Chord Diagram: Weight Relationships</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Network Visualization:**
            - **Nodes:** Sources (circles) and Target (star)
            - **Edges:** Weight relationships (thickness = connection strength)
            - **Colors:** Defect type specific coloring
            - **Sizes:** Node size proportional to combined weight
            """)
            
            fig3 = st.session_state.visualizer.create_chord_diagram(
                sources_data,
                result['target_angle'],
                result['target_params']['defect_type']
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            st.markdown('<h3 class="section-header">Weight Formula Component Breakdown</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Bar Chart Analysis:**
            - **Stacked Bars:** Shows contribution of each weight component
            - **Cumulative Plot:** Shows how weights accumulate around target angle
            - **Defect Type Annotations:** Letters indicate defect type (T=Twin, I=ISF, etc.)
            """)
            
            fig4 = st.session_state.visualizer.create_weight_formula_breakdown(
                sources_data,
                result['target_angle'],
                result['target_params']['defect_type'],
                spatial_sigma
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Detailed weight table
            st.markdown("#### 📋 Detailed Weight Table")
            weight_df = pd.DataFrame(sources_data)
            st.dataframe(
                weight_df[['source_index', 'theta_deg', 'defect_type', 'spatial_weight', 
                          'attention_weight', 'defect_weight', 'combined_weight']].style
                .background_gradient(subset=['combined_weight'], cmap='Blues')
                .format({
                    'spatial_weight': '{:.4f}',
                    'attention_weight': '{:.4f}',
                    'defect_weight': '{:.4f}',
                    'combined_weight': '{:.4f}'
                })
            )
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to Weight Component Visualization
        
        This tool visualizes the attention weight components in Angular Bracketing Theory:
        
        ### Key Features:
        1. **Hierarchical Radar Charts**: Angular variation (Tier 1) × Defect type (Tier 2)
        2. **Weight Comparison**: With vs without defect mask
        3. **Chord Diagrams**: Relationship networks between sources
        4. **Formula Breakdown**: Detailed component analysis
        
        ### Getting Started:
        1. Click **"Load Solutions"** in the sidebar
        2. Configure target parameters (angle, defect type)
        3. Adjust spatial kernel width (σ)
        4. Click **"Run Interpolation"** to generate visualizations
        
        ### Weight Formula:
        The core formula combines three components:
        - **Learned Attention** (ᾱᵢ): Transformer similarity score
        - **Spatial Kernel**: Angular proximity weight
        - **Defect Mask**: Hard constraint for defect type matching
        
        All visualizations highlight the query/target to show dynamic weight computation.
        """)

# =============================================
# RUN APPLICATION
# =============================================
if __name__ == "__main__":
    main()
