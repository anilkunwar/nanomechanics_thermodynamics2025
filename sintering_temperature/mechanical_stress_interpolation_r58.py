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
            
            # Prepare weight analysis data
            sources_data = []
            for i, (field, theta_deg, angular_dist, spatial_w, defect_w, combined_w) in enumerate(zip(
                source_fields, source_theta_degrees, angular_distances, 
                spatial_kernel, defect_mask, final_attention_weights
            )):
                sources_data.append({
                    'source_index': i,
                    'theta_deg': theta_deg,
                    'angular_dist': angular_dist,
                    'defect_type': field['source_params'].get('defect_type', 'Unknown'),
                    'spatial_weight': spatial_w,
                    'defect_weight': defect_w,
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
# ENHANCED WEIGHT VISUALIZER WITH ALL DIAGRAMS
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
            attention_weights = [s.get('attention_weight', 0.0) for s in sources]
            combined_weights = [s['combined_weight'] for s in sources]
            
            # Normalize for radar display
            max_weight = max(max(spatial_weights), max(attention_weights), max(combined_weights), 1e-6)
            spatial_norm = [w/max_weight for w in spatial_weights]
            attention_norm = [w/max_weight for w in attention_weights]
            combined_norm = [w/max_weight for w in combined_weights]
            
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
        
        # FIXED: Correct polar layout without angularaxis title
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
                    tickfont=dict(size=12)
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
        
        # Add annotation for angular axis title
        fig.add_annotation(
            text="Angle (degrees)",
            x=0.5,
            y=-0.1,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="black")
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
        attention_weights = [s.get('attention_weight', 0.0) for s in sources_data]
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
    
    def create_sankey_diagram(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create Sankey diagram showing flow of attention components
        """
        # Create nodes for Sankey diagram
        labels = ['Target']  # Start with target node
        
        # Add source nodes
        for source in sources_data:
            labels.append(f"S{source['source_index']}: {source['defect_type']}")
        
        # Add component nodes
        component_start = len(labels)
        labels.extend(['Spatial Kernel', 'Defect Match', 'Attention Score', 'Combined Weight'])
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        
        # Links from sources to components
        for i, source in enumerate(sources_data):
            source_idx = i + 1  # +1 because index 0 is target
            
            # Link to spatial kernel
            source_indices.append(source_idx)
            target_indices.append(component_start)
            values.append(source['spatial_weight'] * 10)  # Scale for visibility
            colors.append('rgba(78, 205, 196, 0.6)')
            
            # Link to defect match
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            values.append(source['defect_weight'] * 10)
            colors.append('rgba(255, 107, 107, 0.6)')
            
            # Link to attention score
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            attention_w = source.get('attention_weight', source['combined_weight'])
            values.append(attention_w * 10)
            colors.append('rgba(149, 225, 211, 0.6)')
            
            # Link to combined weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            values.append(source['combined_weight'] * 10)
            colors.append('rgba(157, 78, 221, 0.6)')
        
        # Links from components to target
        for comp_idx in range(4):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)  # Target node
            # Sum of all flows into this component
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values) 
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.3)  # Reduce flow to target
            colors.append('rgba(255, 217, 61, 0.6)')
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#9D4EDD"] +  # Target color
                      [self.color_scheme[s['defect_type']] for s in sources_data] +  # Source colors
                      ["#4ECDC4", "#FF6B6B", "#95E1D3", "#FFD93D"]  # Component colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='Flow: %{value:.2f}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=f'Sankey Diagram: Attention Component Flow<br>Target: {target_angle}° {target_defect}, σ={spatial_sigma}°',
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            font=dict(size=12),
            width=1200,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_weight_formula_breakdown(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create visualization showing weight formula breakdown for each source
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Weight Component Breakdown',
                'Cumulative Weight Distribution',
                'Defect Type Distribution',
                'Angular Distribution of Weights'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Sort sources by angle
        sources_data = sorted(sources_data, key=lambda x: x['theta_deg'])
        
        angles = [s['theta_deg'] for s in sources_data]
        source_indices = [s['source_index'] for s in sources_data]
        
        # Prepare stacked bar data
        spatial_values = [s['spatial_weight'] for s in sources_data]
        attention_values = [s.get('attention_weight', 0.0) for s in sources_data]
        defect_values = [s['defect_weight'] for s in sources_data]
        combined_values = [s['combined_weight'] for s in sources_data]
        
        # Colors for components
        colors = {
            'spatial': '#4ECDC4',
            'attention': '#FF6B6B',
            'defect': '#95E1D3',
            'combined': '#9D4EDD'
        }
        
        # Plot 1: Component breakdown (stacked bar)
        fig.add_trace(go.Bar(
            x=source_indices,
            y=spatial_values,
            name='Spatial Kernel',
            marker_color=colors['spatial'],
            text=[f'{v:.3f}' for v in spatial_values],
            textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices,
            y=attention_values,
            name='Attention Score',
            marker_color=colors['attention'],
            text=[f'{v:.3f}' for v in attention_values],
            textposition='outside'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=source_indices,
            y=defect_values,
            name='Defect Match',
            marker_color=colors['defect'],
            text=[f'{v:.3f}' for v in defect_values],
            textposition='outside'
        ), row=1, col=1)
        
        # Add target angle line
        fig.add_vline(x=target_angle, line_dash="dash", line_color="red", 
                     annotation_text=f"Target: {target_angle}°", 
                     annotation_position="top right", row=1, col=1)
        
        # Plot 2: Cumulative distribution
        sorted_weights = np.sort(combined_values)[::-1]
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights) if np.sum(sorted_weights) > 0 else np.zeros_like(sorted_weights)
        x_vals = np.arange(1, len(cumulative) + 1)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative Weight',
            line=dict(color=colors['combined'], width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(157, 78, 221, 0.2)'
        ), row=1, col=2)
        
        # Add 90% threshold line
        if len(cumulative) > 0:
            threshold_idx = np.where(cumulative >= 0.9)[0]
            if len(threshold_idx) > 0:
                fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                             annotation_text="90% threshold", row=1, col=2)
                fig.add_vline(x=threshold_idx[0]+1, line_dash="dash", line_color="red", row=1, col=2)
        
        # Plot 3: Defect type distribution
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
        
        # Plot 4: Angular distribution
        fig.add_trace(go.Scatter(
            x=angles,
            y=combined_values,
            mode='markers+lines',
            name='Weight by Angle',
            line=dict(color=colors['combined'], width=2),
            marker=dict(
                size=12,
                color=[self.color_scheme[s['defect_type']] for s in sources_data],
                line=dict(width=2, color='white')
            ),
            text=[f"S{i}: {d}°<br>W:{w:.3f}" for i, d, w in zip(source_indices, angles, combined_values)]
        ), row=2, col=2)
        
        # Add habit plane line
        fig.add_vline(x=54.7, line_dash="dot", line_color="green", 
                     annotation_text="Habit Plane (54.7°)", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            title=dict(
                text=f'Weight Formula Analysis: wᵢ(θ*) = [ᾱᵢ(θ*)·exp(-(Δφᵢ)²/(2σ²))·𝟙(τᵢ=τ*)]/Σ[...] + 10⁻⁶<br>σ={spatial_sigma}°, Target: {target_defect}',
                font=dict(size=16, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            width=1400,
            height=900,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Source Index", row=1, col=1)
        fig.update_yaxes(title_text="Weight Component Value", row=1, col=1)
        fig.update_xaxes(title_text="Number of Top Sources", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Weight", row=1, col=2)
        fig.update_xaxes(title_text="Defect Type", row=2, col=1)
        fig.update_yaxes(title_text="Count / Weight", row=2, col=1)
        fig.update_xaxes(title_text="Angle (degrees)", row=2, col=2)
        fig.update_yaxes(title_text="Combined Weight", row=2, col=2)
        
        return fig
    
    def create_attention_sunburst(self, sources_data, target_angle, target_defect, title="Attention Distribution Sunburst"):
        """
        Create sunburst chart showing hierarchical attention distribution
        """
        # Build hierarchical data
        labels = ['All Sources']
        parents = ['']
        values = [1.0]
        
        # Group by defect type
        defect_weights = {}
        for source in sources_data:
            defect = source['defect_type']
            if defect not in defect_weights:
                defect_weights[defect] = 0.0
            defect_weights[defect] += source['combined_weight']
        
        # Add defect type nodes
        for defect, weight in defect_weights.items():
            labels.append(defect)
            parents.append('All Sources')
            values.append(weight)
        
        # Add source nodes
        for source in sources_data:
            label = f"S{source['source_index']}<br>{source['theta_deg']:.1f}°"
            labels.append(label)
            parents.append(source['defect_type'])
            values.append(source['combined_weight'])
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hoverinfo="label+value+percent parent",
            marker=dict(
                colors=[self.color_scheme.get(l, '#CCCCCC') for l in labels]
            ),
            maxdepth=3
        ))
        
        fig.update_layout(
            title=dict(
                text=f"{title}<br>Target: {target_angle}° {target_defect}",
                font=dict(size=18, color='darkblue'),
                x=0.5
            ),
            width=800,
            height=800,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
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
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 2.0rem !important;
        color: #374151 !important;
        font-weight: 800 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .formula-box {
        background-color: #F0F7FF;
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        line-height: 1.6;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 6px 6px 0 0;
        gap: 1.2rem;
        padding-top: 12px;
        padding-bottom: 12px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 Advanced Weight Analysis - Angular Bracketing Theory</h1>', unsafe_allow_html=True)
    
    # Domain information
    domain_info = DomainConfiguration.get_domain_info()
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
    <h3 style="color: #1E3A8A;">Domain: {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm</h3>
    <p style="color: #666; font-size: 1.1rem;">
    Grid: {domain_info['grid_points']} × {domain_info['grid_points']} points | 
    Spacing: {domain_info['grid_spacing_nm']} nm | 
    Extent: ±{domain_info['domain_half_nm']} nm
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Weight formula display
    st.markdown(r"""
    <div class="formula-box">
    <strong>🎯 Attention Weight Formula:</strong><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
    
    <strong>📊 Components:</strong>
    • $\bar{\alpha}_i$: Learned attention score from transformer<br>
    • $\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$: Spatial kernel (angular bracketing)<br>
    • $\mathbb{I}(\tau_i = \tau^*)$: Defect type indicator (1 if match, 0 otherwise)<br>
    • $\sigma$: Angular kernel width (controllable parameter)
    </div>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown(f"""
    <div class="info-box">
    <strong>🔬 Physics-Aware Weight Analysis: Hierarchical Visualization of Attention Components</strong><br>
    • <strong>Hierarchical Radar Charts:</strong> Angular variation (Tier 1) × Defect type (Tier 2)<br>
    • <strong>Sankey Diagrams:</strong> Flow visualization of attention component aggregation<br>
    • <strong>Chord Diagrams:</strong> Network visualization of weight relationships<br>
    • <strong>Sunburst Charts:</strong> Hierarchical breakdown of attention distribution<br>
    • <strong>Weight Formula Breakdown:</strong> Detailed component analysis for each source<br>
    • <strong>Defect Type Gating:</strong> Hard constraint for physical validity<br>
    • <strong>Domain Size:</strong> {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm centered at 0 (±{domain_info['domain_half_nm']} nm)
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
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Domain information
        st.markdown("#### 📐 Domain Information")
        st.info(f"""
        **Grid:** {domain_info['grid_points']} × {domain_info['grid_points']} points
        **Spacing:** {domain_info['grid_spacing_nm']} nm
        **Size:** {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm
        **Extent:** ±{domain_info['domain_half_nm']} nm
        **Area:** {domain_info['area_nm2']:.1f} nm²
        """)
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    if st.session_state.solutions:
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                    else:
                        st.warning("No solutions found in directory")
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.success("Cache cleared")
        st.divider()
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
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
            if st.button("Set to Habit Plane", use_container_width=True):
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
        st.markdown("#### 📐 Angular Bracketing Kernel")
        spatial_sigma = st.slider(
            "Kernel Width σ (degrees)",
            min_value=1.0,
            max_value=45.0,
            value=10.0,
            step=0.5,
            help="Width of Gaussian angular bracketing window"
        )
        
        # Run interpolation
        st.markdown("#### 🚀 Interpolation Control")
        if st.button("🎯 Perform Theory-Informed Interpolation", type="primary", use_container_width=True):
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
        st.markdown(f"### 📊 Loaded {len(st.session_state.solutions)} Solutions")
        
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
    
    # Results display with all visualizations
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Results Overview",
            "🕸️ Hierarchical Radar",
            "⚖️ Weight Comparison", 
            "🌀 Sankey Diagram",
            "🔗 Chord Diagram",
            "📊 Formula Breakdown"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 Interpolation Results</h2>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Von Mises", f"{np.max(result['fields']['von_mises']):.3f} GPa")
            with col2:
                st.metric("Target Angle", f"{result['target_angle']:.1f}°")
            with col3:
                st.metric("Defect Type", result['target_params']['defect_type'])
            with col4:
                st.metric("Attention Entropy", f"{result['weights']['entropy']:.3f}")
            
            # Weight table
            st.markdown("#### 📋 Weight Components Table")
            if 'sources_data' in result:
                df = pd.DataFrame(result['sources_data'])
                st.dataframe(
                    df[['source_index', 'theta_deg', 'defect_type', 'spatial_weight', 
                        'defect_weight', 'combined_weight', 'target_defect_match']].style
                    .background_gradient(subset=['combined_weight'], cmap='Blues')
                    .format({
                        'theta_deg': '{:.1f}',
                        'spatial_weight': '{:.4f}',
                        'defect_weight': '{:.4f}',
                        'combined_weight': '{:.4f}'
                    })
                )
                
        with tab2:
            st.markdown('<h2 class="section-header">🕸️ Hierarchical Radar Chart</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Visualization Strategy:**
            - **Tier 1 (Angular Variation):** Outer ring shows angles around the circle
            - **Tier 2 (Defect Types):** Inner rings show different weight components for each defect type
            - **Target Highlight:** Red line at target angle, star marker for target defect
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_hierarchical_radar_chart(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type'],
                    spatial_sigma
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.markdown('<h2 class="section-header">⚖️ Weight Comparison: With vs Without Defect Mask</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Comparison Purpose:**
            - **Left Radar:** Shows attention + spatial weights ignoring defect type
            - **Right Radar:** Shows final weights with defect type gating applied
            - **Color Coding:** Green = matching defect type, Red = mismatched defect type
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_weight_comparison_radar(
                    result['sources_data'],
                    result['target_params']['defect_type']
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab4:
            st.markdown('<h2 class="section-header">🌀 Sankey Diagram: Attention Component Flow</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Flow Visualization:**
            - **Left Nodes:** Individual sources
            - **Middle Nodes:** Weight components (Spatial, Defect, Attention, Combined)
            - **Right Node:** Target interpolation point
            - **Flow Width:** Proportional to weight contribution
            - **Color Coding:** Defect type specific
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_sankey_diagram(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type'],
                    spatial_sigma
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab5:
            st.markdown('<h2 class="section-header">🔗 Chord Diagram: Weight Relationships</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Network Visualization:**
            - **Nodes:** Sources (circles) and Target (star)
            - **Edges:** Weight relationships (thickness = connection strength)
            - **Colors:** Defect type specific coloring
            - **Sizes:** Node size proportional to combined weight
            """)
            
            if 'sources_data' in result:
                fig = st.session_state.weight_visualizer.create_chord_diagram(
                    result['sources_data'],
                    result['target_angle'],
                    result['target_params']['defect_type']
                )
                st.plotly_chart(fig, use_container_width=True)
            
        with tab6:
            st.markdown('<h2 class="section-header">📊 Weight Formula Component Breakdown</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Comprehensive Analysis:**
            - **Component Breakdown:** Stacked bars showing contribution of each weight component
            - **Cumulative Distribution:** Shows how weights accumulate
            - **Defect Type Analysis:** Distribution of weights by defect type
            - **Angular Distribution:** Weight variation with angle
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
        st.markdown("### 🧪 Physics Interpretation of Weight Formula")
        
        col_phys1, col_phys2 = st.columns(2)
        
        with col_phys1:
            st.markdown("""
            **Defect Type as Hard Constraint:**
            - Sources with different defect types receive near-zero attention
            - Critical for physical validity
            - Different defect types have fundamentally different stress fields
            
            **Angular Proximity Drives Attention:**
            - Gaussian kernel creates "bracketing window"
            - Sources within ±σ degrees of target receive highest weights
            - Bracketing structure optimal for interpolation
            """)
        
        with col_phys2:
            st.markdown("""
            **Learned Similarity Refines Selection:**
            - Transformer captures subtle stress field patterns
            - Modulates the physics priors rather than replacing them
            - Enables interpolation beyond simple angular proximity
            
            **Domain Size Awareness:**
            - All visualizations reference the **{domain_info['domain_length_nm']} nm** domain
            - Angular positions mapped to physical domain coordinates
            - Stress fields computed on exact grid spacing
            """.format(**domain_info))
    
    else:
        # Welcome message when no results
        st.markdown("""
        ## 🎯 Welcome to Advanced Weight Analysis
        
        ### Domain Configuration:
        - **Size:** {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm
        - **Grid:** {domain_info['grid_points']} × {domain_info['grid_points']} points
        - **Spacing:** {domain_info['grid_spacing_nm']} nm
        - **Extent:** ±{domain_info['domain_half_nm']} nm
        
        ### Getting Started:
        1. **Load Solutions** from the sidebar
        2. **Configure Target Parameters** (angle, defect type)
        3. **Set Angular Bracketing Parameters** (kernel width σ)
        4. **Click "Perform Theory-Informed Interpolation"** to run
        5. **Explore Visualizations** across all tabs
        
        ### Key Features:
        - **Hierarchical Radar Charts:** Angular variation × Defect type
        - **Sankey Diagrams:** Flow visualization of weight components
        - **Chord Diagrams:** Network relationships between sources
        - **Weight Formula Breakdown:** Detailed component analysis
        - **Defect Type Gating:** Hard constraint for physical validity
        
        ### Weight Formula Visualization:
        The attention weight formula implements **Angular Bracketing Theory**:
        - Defect type as hard constraint
        - Angular proximity as spatial locality kernel
        - Learned transformer attention as refinement
        - Combined weights ensure physics-aware interpolation
        """.format(**domain_info))

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
