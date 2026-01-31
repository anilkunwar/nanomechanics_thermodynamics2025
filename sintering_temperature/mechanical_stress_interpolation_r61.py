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
# SET PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# =============================================
st.set_page_config(
    page_title="Advanced Physics-Aware Interpolation with Sankey Visualization",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

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
# SANKEY VALIDATION UTILITIES
# =============================================
class SankeyValidator:
    """Utility class for validating Sankey diagram parameters"""
    
    # Valid hoverinfo values for Sankey nodes (from Plotly documentation)
    VALID_NODE_HOVERINFO = {
        'skip',     # Do not show hover info for this node
        'none',     # Show no hover info
        'all',      # Show all available info
        'text',     # Show only text (labels)
        'value',    # Show only value
        'percent'   # Show only percentage
    }
    
    # Valid hoverinfo values for Sankey links
    VALID_LINK_HOVERINFO = {
        'skip', 'none', 'all', 'value', 'percent', 'label'
    }
    
    # Valid hoverinfo combinations (must be strings joined with +)
    VALID_COMBINATIONS = {
        'text+value',
        'text+percent', 
        'value+percent',
        'text+value+percent'
    }
    
    @staticmethod
    def validate_hoverinfo(value, element_type='node'):
        """
        Validate and sanitize hoverinfo value for Sankey elements
        
        Args:
            value: The hoverinfo value to validate
            element_type: 'node' or 'link'
            
        Returns:
            Validated hoverinfo string
        """
        if value is None:
            return 'skip'  # Default safe value
            
        # Convert to string
        value_str = str(value).strip()
        
        # Check if it's a combination
        if '+' in value_str:
            parts = [part.strip() for part in value_str.split('+')]
            # Reconstruct and check if valid combination
            reconstructed = '+'.join(parts)
            if reconstructed in SankeyValidator.VALID_COMBINATIONS:
                return reconstructed
            else:
                # Fall back to safe default
                return 'skip'
        
        # Check single values
        if element_type == 'node':
            valid_set = SankeyValidator.VALID_NODE_HOVERINFO
        else:  # link
            valid_set = SankeyValidator.VALID_LINK_HOVERINFO
            
        if value_str in valid_set:
            return value_str
        
        # Fallback to safe default
        return 'skip'
    
    @staticmethod
    def validate_node_dict(node_dict):
        """Validate entire node dictionary for Sankey diagram"""
        validated = dict(node_dict)
        
        # Ensure all arrays have same length
        required_keys = ['label', 'color']
        for key in required_keys:
            if key not in validated:
                validated[key] = []
        
        # Get array lengths
        label_len = len(validated.get('label', []))
        color_len = len(validated.get('color', []))
        
        # Validate hoverinfo
        if 'hoverinfo' in validated:
            validated['hoverinfo'] = SankeyValidator.validate_hoverinfo(
                validated['hoverinfo'], 'node'
            )
        
        # Ensure arrays match in length
        if label_len != color_len:
            st.warning(f"Node array length mismatch: labels={label_len}, colors={color_len}")
            # Truncate to minimum length
            min_len = min(label_len, color_len)
            if 'label' in validated:
                validated['label'] = validated['label'][:min_len]
            if 'color' in validated:
                validated['color'] = validated['color'][:min_len]
        
        return validated
    
    @staticmethod
    def validate_link_dict(link_dict):
        """Validate entire link dictionary for Sankey diagram"""
        validated = dict(link_dict)
        
        # Ensure all arrays have same length
        required_keys = ['source', 'target', 'value']
        for key in required_keys:
            if key not in validated:
                validated[key] = []
        
        # Get array lengths
        source_len = len(validated.get('source', []))
        target_len = len(validated.get('target', []))
        value_len = len(validated.get('value', []))
        
        # Validate hoverinfo
        if 'hoverinfo' in validated:
            validated['hoverinfo'] = SankeyValidator.validate_hoverinfo(
                validated['hoverinfo'], 'link'
            )
        
        # Ensure arrays match in length
        lengths = [source_len, target_len, value_len]
        if len(set(lengths)) > 1:
            st.warning(f"Link array length mismatch: source={source_len}, target={target_len}, value={value_len}")
            # Truncate to minimum length
            min_len = min(lengths)
            validated['source'] = validated['source'][:min_len]
            validated['target'] = validated['target'][:min_len]
            validated['value'] = validated['value'][:min_len]
        
        return validated
    
    @staticmethod
    def create_safe_sankey_trace(node_dict, link_dict, **kwargs):
        """
        Create a Sankey trace with validated parameters
        
        Args:
            node_dict: Node dictionary
            link_dict: Link dictionary
            **kwargs: Additional arguments for go.Sankey
            
        Returns:
            Validated go.Sankey trace
        """
        # Validate node and link dictionaries
        safe_node_dict = SankeyValidator.validate_node_dict(node_dict)
        safe_link_dict = SankeyValidator.validate_link_dict(link_dict)
        
        # Create trace with validated parameters
        trace = go.Sankey(
            node=safe_node_dict,
            link=safe_link_dict,
            **kwargs
        )
        
        return trace

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
    
    SHAPE_OPTIONS = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle', 'Circle', 'Ellipse']
    KAPPA_RANGE = (0.1, 2.0)  # Typical range for kappa parameter
    
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
    
    @staticmethod
    def get_shape_description(shape: str) -> str:
        descriptions = {
            'Square': 'Standard square defect geometry',
            'Horizontal Fault': 'Horizontal stacking fault geometry',
            'Vertical Fault': 'Vertical stacking fault geometry',
            'Rectangle': 'Rectangular defect geometry',
            'Circle': 'Circular defect geometry',
            'Ellipse': 'Elliptical defect geometry'
        }
        return descriptions.get(shape, shape)

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
                # Extract parameters
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                else:
                    # Try to find parameters in the data structure
                    for key in ['param', 'physics_params', 'simulation_params']:
                        if key in data:
                            standardized['params'] = data[key]
                            break
                
                # Extract history
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
                
                # Extract metadata
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                    
                # Extract any other relevant data
                if 'stresses' in data:
                    if 'history' not in standardized or not standardized['history']:
                        standardized['history'] = [{'stresses': data['stresses']}]
                    elif isinstance(standardized['history'], list):
                        if len(standardized['history']) > 0:
                            standardized['history'][-1]['stresses'] = data['stresses']
                
                # Ensure required parameters exist
                if 'defect_type' not in standardized['params']:
                    standardized['params']['defect_type'] = 'Twin'
                if 'theta' not in standardized['params']:
                    standardized['params']['theta'] = 0.0
                if 'shape' not in standardized['params']:
                    standardized['params']['shape'] = 'Square'
                if 'kappa' not in standardized['params']:
                    standardized['params']['kappa'] = 0.6
                if 'eps0' not in standardized['params']:
                    standardized['params']['eps0'] = PhysicsParameters.get_eigenstrain(
                        standardized['params']['defect_type']
                    )
                    
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
    
    def load_selected_files(self, file_paths: List[str]):
        """Load specific files by path"""
        solutions = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                solution = self.read_simulation_file(file_path)
                if solution:
                    solutions.append(solution)
            else:
                st.warning(f"File not found: {file_path}")
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
        target_shape = target_params.get('shape', 'Square')
        target_kappa = target_params.get('kappa', 0.6)
        
        for src in source_params:
            src_theta = src.get('theta', 0.0)
            src_theta_deg = np.degrees(src_theta) % 360
            
            raw_diff = abs(src_theta_deg - target_theta_deg)
            angular_dist = min(raw_diff, 360 - raw_diff)
            angular_distances.append(angular_dist)
            
            # Defect type match
            if src.get('defect_type') == target_defect:
                defect_match = 1.0
            else:
                defect_match = 1e-6
                
            # Shape similarity (add shape matching factor)
            shape_similarity = 1.0 if src.get('shape') == target_shape else 0.5
            
            # Kappa similarity (Gaussian similarity in kappa space)
            src_kappa = src.get('kappa', 0.6)
            kappa_sigma = 0.2  # Width for kappa similarity
            kappa_similarity = np.exp(-0.5 * ((src_kappa - target_kappa) / kappa_sigma) ** 2)
            
            # Combined similarity factor
            similarity_factor = defect_match * shape_similarity * kappa_similarity
            
            # Angular kernel weight
            weight = np.exp(-0.5 * (angular_dist / self.spatial_sigma) ** 2) * similarity_factor
            
            spatial_weights.append(weight)
            defect_mask.append(similarity_factor)
            
        return np.array(spatial_weights), np.array(defect_mask), np.array(angular_distances)
    
    def encode_parameters(self, params_list, target_angle_deg):
        encoded = []
        for params in params_list:
            features = []
            
            # Physics parameters
            eps0 = params.get('eps0', 0.707)
            features.append(eps0 / 3.0)  # normalize
            
            kappa = params.get('kappa', 0.6)
            features.append(kappa / 2.0)  # normalize
            
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)  # normalize to [0,1]
            
            # One-hot encoding for defect type
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
                
            # One-hot encoding for shape
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle', 'Circle', 'Ellipse']
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
            
            # Kappa feature
            features.append(kappa)
            
            # Shape complexity feature
            shape_complexity = {
                'Square': 1.0,
                'Rectangle': 1.2,
                'Circle': 1.5,
                'Ellipse': 1.8,
                'Horizontal Fault': 2.0,
                'Vertical Fault': 2.0
            }.get(shape, 1.0)
            features.append(shape_complexity)
            
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
                    'shape': field['source_params'].get('shape', 'Square'),
                    'kappa': field['source_params'].get('kappa', 0.6),
                    'eps0': field['source_params'].get('eps0', 0.707),
                    'spatial_weight': spatial_w,
                    'defect_weight': defect_w,
                    'attention_weight': attention_weight,
                    'combined_weight': combined_w,
                    'target_defect_match': field['source_params'].get('defect_type') == target_params['defect_type'],
                    'target_shape_match': field['source_params'].get('shape') == target_params['shape'],
                    'kappa_diff': abs(field['source_params'].get('kappa', 0.6) - target_params.get('kappa', 0.6)),
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
# ENHANCED SANKEY VISUALIZER WITH VALIDATION
# =============================================
class EnhancedSankeyVisualizer:
    def __init__(self):
        # Vibrant color scheme
        self.color_scheme = {
            'Twin': '#FF6B6B',
            'ISF': '#4ECDC4',
            'ESF': '#95E1D3',
            'No Defect': '#FFD93D',
            'Query': '#9D4EDD',
            'Spatial': '#36A2EB',
            'Defect': '#FF6384',
            'Attention': '#4BC0C0',
            'Combined': '#9966FF',
            'Shape': '#FF9F40',
            'Kappa': '#FF6384'
        }
        
        # Shape-specific colors
        self.shape_colors = {
            'Square': '#4CAF50',
            'Rectangle': '#8BC34A',
            'Circle': '#2196F3',
            'Ellipse': '#03A9F4',
            'Horizontal Fault': '#FF9800',
            'Vertical Fault': '#FF5722'
        }
    
    def create_customizable_sankey_diagram(self, sources_data, target_angle, target_defect, 
                                           target_shape, target_kappa, spatial_sigma, 
                                           font_config, visual_params):
        """
        Create enhanced Sankey diagram with customizable parameters
        
        Args:
            sources_data: List of source data dictionaries
            target_angle: Target angle in degrees
            target_defect: Target defect type
            target_shape: Target shape
            target_kappa: Target kappa value
            spatial_sigma: Angular kernel width
            font_config: Dictionary with font size configurations
            visual_params: Dictionary with visual enhancement parameters
        """
        # Create nodes for Sankey diagram
        labels = [f'Target\n{target_defect}\n{target_shape}\nκ={target_kappa:.2f}']
        
        # Add source nodes with detailed information
        for source in sources_data:
            angle = source['theta_deg']
            defect = source['defect_type']
            shape = source['shape']
            kappa = source['kappa']
            labels.append(f"S{source['source_index']}\n{defect}\n{shape}\n{angle:.1f}°\nκ={kappa:.2f}")
        
        # Add component nodes
        component_start = len(labels)
        labels.extend(['Spatial Kernel', 'Defect Match', 'Shape Match', 'Kappa Similarity', 'Attention Score', 'Combined Weight'])
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        link_labels = []
        
        # Calculate link values
        for i, source in enumerate(sources_data):
            source_idx = i + 1
            
            # Spatial kernel link
            source_indices.append(source_idx)
            target_indices.append(component_start)
            spatial_value = source['spatial_weight'] * 100
            values.append(spatial_value)
            colors.append('rgba(54, 162, 235, 0.8)')  # Blue
            link_labels.append(f"Spatial: {source['spatial_weight']:.3f}")
            
            # Defect match link
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            defect_value = source['defect_weight'] * 100
            values.append(defect_value)
            colors.append('rgba(255, 99, 132, 0.8)')  # Pink
            link_labels.append(f"Defect: {source['defect_weight']:.3f}")
            
            # Shape match link
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            shape_match = 1.0 if source.get('target_shape_match', False) else 0.3
            shape_value = shape_match * 100
            values.append(shape_value)
            colors.append('rgba(255, 159, 64, 0.8)')  # Orange
            link_labels.append(f"Shape: {shape_match:.3f}")
            
            # Kappa similarity link
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            kappa_sim = np.exp(-source.get('kappa_diff', 1.0))
            kappa_value = kappa_sim * 100
            values.append(kappa_value)
            colors.append('rgba(255, 99, 132, 0.6)')  # Light Pink
            link_labels.append(f"Kappa: {kappa_sim:.3f}")
            
            # Attention score link
            source_indices.append(source_idx)
            target_indices.append(component_start + 4)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_value = attention_w * 100
            values.append(attention_value)
            colors.append('rgba(75, 192, 192, 0.8)')  # Cyan
            link_labels.append(f"Attention: {attention_w:.3f}")
            
            # Combined weight link
            source_indices.append(source_idx)
            target_indices.append(component_start + 5)
            combined_value = source['combined_weight'] * 100
            values.append(combined_value)
            colors.append('rgba(153, 102, 255, 0.8)')  # Purple
            link_labels.append(f"Combined: {source['combined_weight']:.3f}")
        
        # Links from components to target
        for comp_idx in range(6):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)
            
            # Sum of all flows into this component
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values) 
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.3)  # Reduce flow to target
            
            # Component-specific colors
            comp_colors = [
                'rgba(54, 162, 235, 0.6)',    # Spatial
                'rgba(255, 99, 132, 0.6)',    # Defect
                'rgba(255, 159, 64, 0.6)',    # Shape
                'rgba(255, 99, 132, 0.4)',    # Kappa
                'rgba(75, 192, 192, 0.6)',    # Attention
                'rgba(153, 102, 255, 0.6)'    # Combined
            ]
            colors.append(comp_colors[comp_idx])
            
            link_labels.append(f"Component {comp_idx} → Target")
        
        # Prepare node colors
        node_colors = self._get_node_colors(labels, sources_data, target_defect, target_shape)
        
        # Prepare node dictionary with VALIDATED parameters
        node_dict = {
            'pad': visual_params.get('node_padding', 30),
            'thickness': visual_params.get('node_thickness', 35),
            'line': dict(
                color="black", 
                width=visual_params.get('border_width', 2)
            ),
            'label': labels,
            'color': node_colors,
            # Use hovertemplate instead of hoverinfo for better control
            'hovertemplate': '<b>%{label}</b><br>Total Flow: %{value:.2f}<extra></extra>',
            # SAFE hoverinfo value - using 'skip' as recommended
            'hoverinfo': 'skip'
        }
        
        # Prepare link dictionary
        link_dict = {
            'source': source_indices,
            'target': target_indices,
            'value': values,
            'color': colors,
            # Use hovertemplate for links as well
            'hovertemplate': '<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<extra></extra>',
            'line': dict(width=0.5, color='rgba(255,255,255,0.3)')
        }
        
        # Create Sankey diagram using safe validator
        try:
            sankey_trace = SankeyValidator.create_safe_sankey_trace(
                node_dict=node_dict,
                link_dict=link_dict,
                arrangement='snap',  # Improved arrangement
                valueformat='.2f',   # Format for values
                valuesuffix=' units' # Suffix for values
            )
            
            fig = go.Figure(data=[sankey_trace])
            
        except Exception as e:
            st.error(f"Error creating Sankey diagram: {str(e)}")
            # Create a simple fallback figure
            fig = go.Figure()
            fig.update_layout(
                title="Sankey Diagram Error - Using fallback visualization",
                annotations=[dict(
                    text="Sankey diagram could not be created. Check data and parameters.",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=16)
                )]
            )
            return fig
        
        # Enhanced layout with configurable font sizes
        fig.update_layout(
            title=dict(
                text=f'<b>PHYSICS-AWARE SANKEY DIAGRAM</b><br>Target: {target_angle}° {target_defect} | Shape: {target_shape} | κ={target_kappa} | σ={spatial_sigma}°',
                font=dict(
                    family='Arial, sans-serif',
                    size=font_config.get('title_font_size', 26),
                    color='#2C3E50',
                    weight='bold'
                ),
                x=0.5,
                y=0.97,
                xanchor='center',
                yanchor='top'
            ),
            font=dict(
                family='Arial, sans-serif',
                size=font_config.get('label_font_size', 16),
                color='#2C3E50'
            ),
            width=visual_params.get('width', 1600),
            height=visual_params.get('height', 1000),
            plot_bgcolor=visual_params.get('plot_bgcolor', 'rgba(240, 240, 245, 0.9)'),
            paper_bgcolor=visual_params.get('paper_bgcolor', 'white'),
            margin=dict(
                t=font_config.get('title_font_size', 26) + 80,
                l=50, 
                r=50, 
                b=50
            ),
            # Configure hover behavior at layout level
            hovermode='x unified',
            hoverlabel=dict(
                font=dict(
                    family='Arial, sans-serif',
                    size=font_config.get('hover_font_size', 12),
                    color='white'
                ),
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white'
            )
        )
        
        # Add annotations
        annotations = self._create_sankey_annotations(font_config)
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def _get_node_colors(self, labels, sources_data, target_defect, target_shape):
        """Get color palette for nodes"""
        colors = []
        
        # Target node
        colors.append(self.color_scheme['Query'])
        
        # Source nodes - use shape-specific colors when available
        for i in range(len(sources_data)):
            shape = sources_data[i]['shape']
            if shape in self.shape_colors:
                colors.append(self.shape_colors[shape])
            else:
                defect = sources_data[i]['defect_type']
                colors.append(self.color_scheme.get(defect, '#CCCCCC'))
        
        # Component nodes
        colors.extend([
            self.color_scheme['Spatial'],    # Spatial Kernel
            self.color_scheme['Defect'],     # Defect Match
            self.color_scheme['Shape'],      # Shape Match
            self.color_scheme['Kappa'],      # Kappa Similarity
            self.color_scheme['Attention'],  # Attention Score
            self.color_scheme['Combined']    # Combined Weight
        ])
        
        return colors
    
    def _create_sankey_annotations(self, font_config):
        """Create Sankey diagram annotations"""
        return [
            dict(
                x=0.02, y=1.05,
                xref='paper', yref='paper',
                text='<b>COLOR CODING:</b>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 14),
                    color='darkblue',
                    weight='bold'
                )
            ),
            dict(
                x=0.02, y=1.02,
                xref='paper', yref='paper',
                text='• Spatial: <span style="color:#36A2EB">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 12),
                    color='#36A2EB'
                )
            ),
            dict(
                x=0.15, y=1.02,
                xref='paper', yref='paper',
                text='• Defect: <span style="color:#FF6384">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 12),
                    color='#FF6384'
                )
            ),
            dict(
                x=0.28, y=1.02,
                xref='paper', yref='paper',
                text='• Shape: <span style="color:#FF9F40">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 12),
                    color='#FF9F40'
                )
            ),
            dict(
                x=0.41, y=1.02,
                xref='paper', yref='paper',
                text='• Kappa: <span style="color:#FF6384">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 12),
                    color='#FF6384'
                )
            ),
            dict(
                x=0.54, y=1.02,
                xref='paper', yref='paper',
                text='• Attention: <span style="color:#4BC0C0">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 12),
                    color='#4BC0C0'
                )
            ),
            dict(
                x=0.67, y=1.02,
                xref='paper', yref='paper',
                text='• Combined: <span style="color:#9966FF">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 12),
                    color='#9966FF'
                )
            )
        ]
    
    def create_parameter_analysis_chart(self, sources_data, target_params):
        """Create parameter analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Parameter Distribution</b>',
                '<b>Weight vs Angle</b>',
                '<b>Shape Analysis</b>',
                '<b>Kappa Distribution</b>'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Prepare data
        df = pd.DataFrame(sources_data)
        
        # Plot 1: Parameter distribution
        fig.add_trace(go.Scatter(
            x=df['theta_deg'],
            y=df['eps0'],
            mode='markers',
            marker=dict(
                size=15,
                color=df['combined_weight'] * 100,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Weight %")
            ),
            text=[f"S{i}: θ={θ:.1f}°, ε₀={eps:.3f}, w={w:.3f}" 
                  for i, θ, eps, w in zip(df['source_index'], df['theta_deg'], df['eps0'], df['combined_weight'])],
            hoverinfo='text'
        ), row=1, col=1)
        
        # Plot 2: Weight vs Angle
        fig.add_trace(go.Scatter(
            x=df['theta_deg'],
            y=df['combined_weight'],
            mode='markers+lines',
            line=dict(color='#9966FF', width=3),
            marker=dict(size=10, color='#9966FF'),
            text=[f"S{i}: {θ:.1f}°" for i, θ in zip(df['source_index'], df['theta_deg'])],
            hoverinfo='text'
        ), row=1, col=2)
        
        # Add target angle line
        fig.add_vline(x=target_params.get('target_angle', 54.7), 
                     line_dash="dash", line_color="red", row=1, col=2)
        
        # Plot 3: Shape analysis
        shape_counts = df['shape'].value_counts()
        fig.add_trace(go.Bar(
            x=shape_counts.index,
            y=shape_counts.values,
            marker_color=[self.shape_colors.get(s, '#CCCCCC') for s in shape_counts.index],
            text=[f"{s}: {c}" for s, c in zip(shape_counts.index, shape_counts.values)],
            textposition='auto'
        ), row=2, col=1)
        
        # Plot 4: Kappa distribution
        fig.add_trace(go.Box(
            y=df['kappa'],
            name='Kappa',
            marker_color='#FF6384',
            boxmean=True
        ), row=2, col=2)
        
        fig.add_hline(y=target_params.get('kappa', 0.6), 
                     line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>PARAMETER ANALYSIS</b><br>Target: {target_params.get("defect_type")} | Shape: {target_params.get("shape")} | κ={target_params.get("kappa")}',
                font=dict(size=20, family='Arial, sans-serif')
            ),
            showlegend=False,
            width=1400,
            height=900,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Angle (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Eigenstrain (ε₀)", row=1, col=1)
        fig.update_xaxes(title_text="Angle (degrees)", row=1, col=2)
        fig.update_yaxes(title_text="Combined Weight", row=1, col=2)
        fig.update_xaxes(title_text="Shape Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="", row=2, col=2)
        fig.update_yaxes(title_text="Kappa Value", row=2, col=2)
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
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
    .sankey-controls {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .font-control-group {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .control-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: block;
        color: white;
    }
    .file-upload-box {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(59, 130, 246, 0.05);
        margin: 1rem 0;
    }
    .debug-info {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🧪 PHYSICS-AWARE INTERPOLATION WITH ENHANCED SANKEY VISUALIZATION</h1>', unsafe_allow_html=True)
    
    # Domain information
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
    <strong style="font-size: 1.5rem;">🎯 ATTENTION WEIGHT FORMULA WITH SHAPE & KAPPA:</strong><br><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*) \cdot S(s_i, s^*) \cdot K(\kappa_i, \kappa^*)}{\sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*) \cdot S(s_k, s^*) \cdot K(\kappa_k, \kappa^*)}$$
    
    <br><strong style="font-size: 1.3rem;">📊 COMPONENTS:</strong><br>
    • <span style="color:#4BC0C0">$\bar{\alpha}_i$</span>: Learned attention score<br>
    • <span style="color:#36A2EB">$\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$</span>: Spatial kernel<br>
    • <span style="color:#FF6384">$\mathbb{I}(\tau_i = \tau^*)$</span>: Defect type match<br>
    • <span style="color:#FF9F40">$S(s_i, s^*)$</span>: Shape similarity<br>
    • <span style="color:#FF6384">$K(\kappa_i, \kappa^*)$</span>: Kappa similarity<br>
    • <span style="color:#9966FF">$\sigma$</span>: Angular kernel width
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
    if 'sankey_visualizer' not in st.session_state:
        st.session_state.sankey_visualizer = EnhancedSankeyVisualizer()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ CONFIGURATION</h2>', unsafe_allow_html=True)
        
        # Debug mode toggle
        with st.expander("🔧 Debug Options", expanded=False):
            st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=False)
            if st.session_state.debug_mode:
                st.info("Debug mode enabled. Additional validation info will be shown.")
        
        # Domain information
        st.markdown("#### 📐 DOMAIN INFORMATION")
        st.info(f"""
        **Grid:** {grid_points} × {grid_points} points
        **Spacing:** {grid_spacing} nm
        **Size:** {domain_length} nm × {domain_length} nm
        **Extent:** ±{domain_half} nm
        **Area:** {domain_info['area_nm2']:.1f} nm²
        """)
        
        # File upload section
        st.markdown("#### 📂 FILE MANAGEMENT")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload simulation files",
            type=['pkl', 'pickle', 'pt', 'pth'],
            accept_multiple_files=True,
            help="Upload pickle or torch files containing simulation data"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 LOAD FROM FOLDER", use_container_width=True):
                with st.spinner("Loading solutions from folder..."):
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
        
        # Load uploaded files
        if uploaded_files:
            if st.button("📥 LOAD UPLOADED FILES", use_container_width=True, type="primary"):
                with st.spinner("Processing uploaded files..."):
                    solutions = []
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        temp_path = os.path.join(SOLUTIONS_DIR, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load the file
                        solution = st.session_state.loader.read_simulation_file(temp_path)
                        if solution:
                            solutions.append(solution)
                    
                    if solutions:
                        st.session_state.solutions = solutions
                        st.success(f"Loaded {len(solutions)} uploaded files")
                    else:
                        st.error("Failed to load uploaded files")
        
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
        
        # SHAPE parameter
        shape = st.selectbox(
            "Shape",
            options=PhysicsParameters.SHAPE_OPTIONS,
            index=0,
            help="Geometry shape of the defect"
        )
        
        # KAPPA parameter
        kappa = st.slider(
            "Kappa (κ)",
            min_value=float(PhysicsParameters.KAPPA_RANGE[0]),
            max_value=float(PhysicsParameters.KAPPA_RANGE[1]),
            value=0.6,
            step=0.05,
            help="Material parameter (typically 0.1 to 2.0)"
        )
        
        # Display physics information
        if defect_type in PhysicsParameters.EIGENSTRAIN_VALUES:
            eps0 = PhysicsParameters.get_eigenstrain(defect_type)
            info = PhysicsParameters.get_theoretical_info(defect_type)
            st.info(f"""
            **Physics Parameters:**
            - Eigenstrain (ε₀): {eps0}
            - Formula: {info.get('formula', 'N/A')}
            - Description: {info.get('description', 'N/A')}
            """)
        
        st.markdown(f"""
        **Shape Description:** {PhysicsParameters.get_shape_description(shape)}
        """)
        
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
        
        # Sankey customization
        st.markdown('<h2 class="section-header">🎨 SANKEY CUSTOMIZATION</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="sankey-controls">', unsafe_allow_html=True)
        
        # Font size controls
        st.markdown('<div class="font-control-group">', unsafe_allow_html=True)
        st.markdown('<span class="control-label">📝 Font Sizes</span>', unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            title_font_size = st.slider(
                "Title Size",
                min_value=16,
                max_value=40,
                value=26,
                step=1,
                help="Title font size in points"
            )
        
        with col_f2:
            label_font_size = st.slider(
                "Label Size",
                min_value=10,
                max_value=30,
                value=16,
                step=1,
                help="Node label font size in points"
            )
        
        annotation_font_size = st.slider(
            "Annotation Size",
            min_value=10,
            max_value=24,
            value=14,
            step=1,
            help="Annotation text font size"
        )
        
        hover_font_size = st.slider(
            "Hover Text Size",
            min_value=10,
            max_value=20,
            value=12,
            step=1,
            help="Hover text font size"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual controls
        st.markdown('<div class="font-control-group">', unsafe_allow_html=True)
        st.markdown('<span class="control-label">🎨 Visual Parameters</span>', unsafe_allow_html=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            node_padding = st.slider(
                "Node Padding",
                min_value=10,
                max_value=50,
                value=30,
                step=1,
                help="Padding between nodes"
            )
        
        with col_v2:
            node_thickness = st.slider(
                "Node Thickness",
                min_value=15,
                max_value=50,
                value=35,
                step=1,
                help="Thickness of nodes"
            )
        
        border_width = st.slider(
            "Border Width",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Border width of nodes"
        )
        
        # Layout controls
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            width = st.slider(
                "Width",
                min_value=800,
                max_value=2000,
                value=1600,
                step=50,
                help="Diagram width in pixels"
            )
        
        with col_w2:
            height = st.slider(
                "Height",
                min_value=600,
                max_value=1200,
                value=1000,
                step=50,
                help="Diagram height in pixels"
            )
        
        # Background color
        bg_color = st.color_picker(
            "Background Color",
            value="#F0F0F5",
            help="Select plot background color"
        )
        
        # Hoverinfo configuration (SAFE OPTIONS ONLY)
        st.markdown("#### 🖱️ HOVER BEHAVIOR")
        hoverinfo_option = st.selectbox(
            "Hover Info Display",
            options=['skip', 'none', 'all', 'text', 'value', 'percent', 
                    'text+value', 'text+percent', 'value+percent', 'text+value+percent'],
            index=2,  # Default to 'all'
            help="Control what information appears on hover (validated for Sankey)"
        )
        
        # Validate the hoverinfo option
        validated_hoverinfo = SankeyValidator.validate_hoverinfo(hoverinfo_option, 'node')
        if hoverinfo_option != validated_hoverinfo and st.session_state.debug_mode:
            st.warning(f"Hoverinfo '{hoverinfo_option}' was sanitized to '{validated_hoverinfo}'")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Preset buttons
        st.markdown("#### 🚀 QUICK PRESETS")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("📱 Mobile", use_container_width=True):
                st.session_state.mobile_preset = True
                st.rerun()
        
        with col_p2:
            if st.button("🖥️ Desktop", use_container_width=True):
                st.session_state.desktop_preset = True
                st.rerun()
        
        with col_p3:
            if st.button("📊 Presentation", use_container_width=True):
                st.session_state.presentation_preset = True
                st.rerun()
        
        # Run interpolation
        st.markdown("#### 🚀 INTERPOLATION CONTROL")
        if st.button("🎯 PERFORM INTERPOLATION", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing physics-aware interpolation..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': PhysicsParameters.get_eigenstrain(defect_type),
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
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
                        st.success(f"Interpolation successful! Applied with σ={spatial_sigma}°")
                    else:
                        st.error("Interpolation failed. Check console for errors.")
    
    # Handle presets
    if 'mobile_preset' in st.session_state and st.session_state.mobile_preset:
        title_font_size = 20
        label_font_size = 12
        annotation_font_size = 10
        width = 800
        height = 600
        hoverinfo_option = 'skip'  # Simpler for mobile
        st.session_state.mobile_preset = False
    
    if 'desktop_preset' in st.session_state and st.session_state.desktop_preset:
        title_font_size = 24
        label_font_size = 16
        annotation_font_size = 14
        width = 1200
        height = 800
        hoverinfo_option = 'text+value'
        st.session_state.desktop_preset = False
    
    if 'presentation_preset' in st.session_state and st.session_state.presentation_preset:
        title_font_size = 28
        label_font_size = 18
        annotation_font_size = 16
        width = 1600
        height = 1000
        hoverinfo_option = 'all'
        st.session_state.presentation_preset = False
    
    # Prepare configuration
    font_config = {
        'title_font_size': title_font_size,
        'label_font_size': label_font_size,
        'annotation_font_size': annotation_font_size,
        'hover_font_size': hover_font_size
    }
    
    visual_params = {
        'node_padding': node_padding,
        'node_thickness': node_thickness,
        'border_width': border_width,
        'width': width,
        'height': height,
        'plot_bgcolor': bg_color,
        'paper_bgcolor': 'white',
        'node_hoverinfo': hoverinfo_option  # Will be validated before use
    }
    
    # Main content area
    # Display loaded solutions
    if st.session_state.solutions:
        st.markdown(f'<h2 class="section-header">📊 LOADED {len(st.session_state.solutions)} SOLUTIONS</h2>', unsafe_allow_html=True)
        
        # Create summary table
        summary_data = []
        for i, sol in enumerate(st.session_state.solutions):
            params = sol.get('params', {})
            summary_data.append({
                'ID': i,
                'Defect': params.get('defect_type', 'Unknown'),
                'Angle (°)': np.degrees(params.get('theta', 0)) if 'theta' in params else 0,
                'Shape': params.get('shape', 'Unknown'),
                'Kappa': params.get('kappa', 'N/A'),
                'Eps0': params.get('eps0', 'N/A'),
                'History Frames': len(sol.get('history', []))
            })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Statistics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Solutions", len(st.session_state.solutions))
            with col_s2:
                defect_types = df_summary['Defect'].unique()
                st.metric("Defect Types", len(defect_types))
            with col_s3:
                angle_range = f"{df_summary['Angle (°)'].min():.1f}° - {df_summary['Angle (°)'].max():.1f}°"
                st.metric("Angle Range", angle_range)
            with col_s4:
                shapes = df_summary['Shape'].unique()
                st.metric("Shape Types", len(shapes))
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Debug information if enabled
        if st.session_state.debug_mode:
            with st.expander("🔍 Debug Information", expanded=True):
                st.markdown('<div class="debug-info">', unsafe_allow_html=True)
                st.write("### Sankey Validation Debug")
                st.write(f"Hoverinfo value from UI: {hoverinfo_option}")
                st.write(f"Validated hoverinfo: {validated_hoverinfo}")
                st.write(f"Number of sources: {len(result.get('sources_data', []))}")
                st.write(f"Visual params keys: {list(visual_params.keys())}")
                st.write(f"Font config keys: {list(font_config.keys())}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 RESULTS OVERVIEW",
            "🌀 ENHANCED SANKEY", 
            "📊 PARAMETER ANALYSIS",
            "💾 EXPORT DATA"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 INTERPOLATION RESULTS</h2>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                Max Von Mises<br>
                {np.max(result["fields"]["von_mises"]):.3f} GPa
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                Target Angle<br>
                {result["target_angle"]:.1f}°
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                Defect Type<br>
                {result["target_params"]["defect_type"]}
                </div>
                ''', unsafe_allow_html=True)
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                Attention Entropy<br>
                {result["weights"]["entropy"]:.3f}
                </div>
                ''', unsafe_allow_html=True)
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.markdown(f'''
                <div class="metric-card">
                Shape<br>
                {result["target_params"]["shape"]}
                </div>
                ''', unsafe_allow_html=True)
            with col6:
                st.markdown(f'''
                <div class="metric-card">
                Kappa (κ)<br>
                {result["target_params"]["kappa"]}
                </div>
                ''', unsafe_allow_html=True)
            with col7:
                st.markdown(f'''
                <div class="metric-card">
                Eigenstrain (ε₀)<br>
                {result["target_params"]["eps0"]:.3f}
                </div>
                ''', unsafe_allow_html=True)
            with col8:
                st.markdown(f'''
                <div class="metric-card">
                Sources Used<br>
                {result["num_sources"]}
                </div>
                ''', unsafe_allow_html=True)
            
            # Weight table
            st.markdown("#### 📋 WEIGHT COMPONENTS TABLE")
            if 'sources_data' in result:
                df = pd.DataFrame(result['sources_data'])
                st.dataframe(
                    df[['source_index', 'theta_deg', 'defect_type', 'shape', 'kappa', 'eps0',
                        'spatial_weight', 'defect_weight', 'attention_weight', 'combined_weight',
                        'target_defect_match', 'target_shape_match', 'kappa_diff']].style
                    .background_gradient(subset=['combined_weight'], cmap='viridis')
                    .format({
                        'theta_deg': '{:.1f}',
                        'kappa': '{:.3f}',
                        'eps0': '{:.3f}',
                        'spatial_weight': '{:.4f}',
                        'defect_weight': '{:.4f}',
                        'attention_weight': '{:.4f}',
                        'combined_weight': '{:.4f}',
                        'kappa_diff': '{:.3f}'
                    })
                )
                
        with tab2:
            st.markdown('<h2 class="section-header">🌀 ENHANCED SANKEY DIAGRAM</h2>', unsafe_allow_html=True)
            
            # Add validation info in debug mode
            if st.session_state.debug_mode:
                st.info(f"""
                **Sankey Configuration:**
                - Hoverinfo: '{validated_hoverinfo}' (validated from '{hoverinfo_option}')
                - Font sizes: Title={title_font_size}pt, Labels={label_font_size}pt
                - Layout: {width}×{height}px, Node padding={node_padding}px
                - Using SankeyValidator for safe parameter validation
                """)
            
            if 'sources_data' in result:
                try:
                    fig = st.session_state.sankey_visualizer.create_customizable_sankey_diagram(
                        sources_data=result['sources_data'],
                        target_angle=result['target_angle'],
                        target_defect=result['target_params']['defect_type'],
                        target_shape=result['target_params']['shape'],
                        target_kappa=result['target_params']['kappa'],
                        spatial_sigma=spatial_sigma,
                        font_config=font_config,
                        visual_params=visual_params
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export options
                    col_e1, col_e2, col_e3 = st.columns(3)
                    with col_e1:
                        if st.button("💾 Save Sankey as HTML", use_container_width=True):
                            html = fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="Download HTML",
                                data=html,
                                file_name="enhanced_sankey.html",
                                mime="text/html"
                            )
                    
                    with col_e2:
                        if st.button("🖼️ Save Sankey as PNG", use_container_width=True):
                            img_bytes = fig.to_image(format="png", width=width, height=height)
                            st.download_button(
                                label="Download PNG",
                                data=img_bytes,
                                file_name="enhanced_sankey.png",
                                mime="image/png"
                            )
                    
                    with col_e3:
                        if st.button("📊 Save Configuration", use_container_width=True):
                            config = {
                                'font_config': font_config,
                                'visual_params': visual_params,
                                'target_params': result['target_params'],
                                'validated_hoverinfo': validated_hoverinfo,
                                'saved_at': datetime.now().isoformat()
                            }
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(config, indent=2),
                                file_name="sankey_config.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"Error creating Sankey diagram: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    1. Check that all arrays have consistent lengths
                    2. Verify hoverinfo values are valid Sankey options
                    3. Ensure node colors match labels in length
                    4. Try using simpler hoverinfo options like 'skip' or 'none'
                    """)
                    if st.session_state.debug_mode:
                        st.exception(e)
                
        with tab3:
            st.markdown('<h2 class="section-header">📊 PARAMETER ANALYSIS</h2>', unsafe_allow_html=True)
            
            if 'sources_data' in result:
                try:
                    fig = st.session_state.sankey_visualizer.create_parameter_analysis_chart(
                        sources_data=result['sources_data'],
                        target_params=result['target_params']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating parameter analysis: {str(e)}")
                    if st.session_state.debug_mode:
                        st.exception(e)
                
        with tab4:
            st.markdown('<h2 class="section-header">💾 EXPORT DATA</h2>', unsafe_allow_html=True)
            
            # Export all data
            if st.button("📦 Export Complete Results", use_container_width=True, type="primary"):
                # Create a comprehensive data package
                export_data = {
                    'interpolation_result': result,
                    'target_parameters': result['target_params'],
                    'weights_analysis': result['weights'],
                    'sources_data': result['sources_data'],
                    'metadata': {
                        'exported_at': datetime.now().isoformat(),
                        'num_sources': result['num_sources'],
                        'spatial_sigma': spatial_sigma,
                        'domain_info': DomainConfiguration.get_domain_info(),
                        'sankey_config': {
                            'font_config': font_config,
                            'visual_params': visual_params,
                            'hoverinfo': validated_hoverinfo
                        }
                    }
                }
                
                # Convert to JSON
                json_data = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="Download JSON Results",
                    data=json_data,
                    file_name="interpolation_results.json",
                    mime="application/json"
                )
            
            # Export weights CSV
            if 'sources_data' in result:
                df_weights = pd.DataFrame(result['sources_data'])
                csv_data = df_weights.to_csv(index=False)
                
                st.download_button(
                    label="📊 Export Weights as CSV",
                    data=csv_data,
                    file_name="weights_analysis.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome message when no results
        st.markdown(f"""
        <div class="info-box">
        <strong style="font-size: 1.3rem;">🔬 WELCOME TO PHYSICS-AWARE INTERPOLATION</strong><br><br>
        • <strong>Load Simulation Files:</strong> Upload or load from folder<br>
        • <strong>Configure Target Parameters:</strong> Set defect type, shape, kappa, and angle<br>
        • <strong>Customize Visualization:</strong> Adjust Sankey diagram appearance<br>
        • <strong>Perform Interpolation:</strong> Run physics-aware interpolation<br>
        • <strong>Analyze Results:</strong> Explore enhanced Sankey and parameter analysis<br>
        • <strong>Export Data:</strong> Save results in multiple formats
        </div>
        
        ### GETTING STARTED:
        1. **Upload Files** or **Load from Folder** in the sidebar
        2. **Set Target Parameters** (defect type, shape, kappa, angle)
        3. **Adjust Sankey Customization** (font sizes, layout, hover behavior)
        4. **Click "Perform Interpolation"** to run
        5. **Explore Results** across all tabs
        
        ### 🔧 FIXED SANKEY VALIDATION:
        - **Hoverinfo validation:** All hoverinfo values are now validated
        - **Safe defaults:** Invalid hoverinfo values are sanitized to 'skip'
        - **Array length checking:** Ensures all arrays match in length
        - **Debug mode:** Enable in sidebar for validation details
        
        ### KEY FEATURES:
        - **Physics-Aware Interpolation:** Incorporates defect type, shape, and kappa
        - **Enhanced Sankey Diagrams:** Fully customizable with large fonts
        - **Parameter Analysis:** Visualize shape and kappa distributions
        - **Multiple Export Formats:** HTML, PNG, JSON, CSV
        - **Domain Awareness:** {domain_length} nm × {domain_length} nm simulation domain
        """, unsafe_allow_html=True)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
