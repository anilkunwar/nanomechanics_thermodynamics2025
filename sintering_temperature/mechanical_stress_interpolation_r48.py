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

# Create comprehensive list of all colormaps
ALL_COLORMAPS = []
for category in COLORMAP_OPTIONS.values():
    ALL_COLORMAPS.extend(category)
ALL_COLORMAPS = sorted(list(set(ALL_COLORMAPS)))  # Remove duplicates

# =============================================
# PHYSICS PARAMETERS ENHANCEMENT
# =============================================
class PhysicsParameters:
    """Enhanced physics parameters with correct eigenstrain values"""
    
    # Correct eigenstrain values for different defect types
    EIGENSTRAIN_VALUES = {
        'Twin': 2.12,  # Corrected from 0.707
        'ISF': 0.289,  # Corrected from 0.707
        'ESF': 0.333,  # Corrected from 1.414
        'No Defect': 0.0
    }
    
    # Theoretical basis for eigenstrain calculations
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
    
    # Boltzmann constant in eV/K
    k_B_eV = 8.617333262145e-5  # eV/K
    
    # Boltzmann constant in J/K
    k_B_J = 1.380649e-23  # J/K
    
    # Material properties for various metals at sintering temperatures
    MATERIAL_PROPERTIES = {
        'Silver': {
            'atomic_volume': 1.56e-29,  # m³/atom
            'atomic_mass': 107.8682,    # g/mol
            'density': 10.49e6,         # g/m³
            'melting_point': 1234.93,   # K
            'bulk_modulus': 100e9,      # Pa
            'shear_modulus': 30e9,      # Pa
            'activation_energy': 1.1,   # eV for vacancy diffusion
            'prefactor': 7.2e-7,        # m²/s diffusion prefactor
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
            'activation_energy': 1.0,   # eV for vacancy diffusion
            'prefactor': 3.1e-7,        # m²/s diffusion prefactor
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
            'activation_energy': 0.65,  # eV for vacancy diffusion
            'prefactor': 1.7e-6,        # m²/s diffusion prefactor
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
            'activation_energy': 1.4,   # eV for vacancy diffusion
            'prefactor': 1.9e-7,        # m²/s diffusion prefactor
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
            'activation_energy': 2.0,   # eV for vacancy diffusion
            'prefactor': 2.0e-8,        # m²/s diffusion prefactor
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
        """
        Compute diffusion enhancement factor D(σ)/D₀
        
        Parameters:
        -----------
        sigma_hydro_GPa : float or array
            Hydrostatic stress in GPa (positive = tensile, negative = compressive)
        T_K : float
            Temperature in Kelvin
        material : str
            Material name ('Silver', 'Copper', 'Aluminum', 'Nickel', 'Iron')
        model : str
            'physics_corrected' : Full exponential formula
            'temperature_reduction' : Effective temperature model
            'activation_energy' : Activation energy modification
            'vacancy_concentration' : Vacancy concentration change
        stress_unit : str
            Unit of stress ('GPa' or 'Pa')
        
        Returns:
        --------
        D_ratio : float or array
            Diffusion coefficient ratio D(σ)/D₀
        """
        # Get material properties
        props = DiffusionPhysics.get_material_properties(material)
        Omega = props['atomic_volume']  # Atomic volume in m³
        
        # Convert stress to Pa if needed
        if stress_unit == 'GPa':
            sigma_hydro_Pa = sigma_hydro_GPa * 1e9
        else:
            sigma_hydro_Pa = sigma_hydro_GPa
        
        if model == 'physics_corrected':
            # Full exponential formula: D/D₀ = exp(Ωσ_h / (k_B T))
            exponent = Omega * sigma_hydro_Pa / (DiffusionPhysics.k_B_J * T_K)
            D_ratio = np.exp(exponent)
            
        elif model == 'temperature_reduction':
            # Effective temperature model: T_eff = T / (1 - Ωσ_h / Q)
            # where Q is activation energy in Joules
            Q_J = props['activation_energy'] * 1.602e-19  # Convert eV to J
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                T_eff = T_K / (1 - Omega * sigma_hydro_Pa / Q_J)
                T_eff = np.where(np.isfinite(T_eff), T_eff, T_K)
            D_ratio = np.exp(Q_J / DiffusionPhysics.k_B_J * (1/T_K - 1/T_eff))
            
        elif model == 'activation_energy':
            # Activation energy modification: Q_eff = Q - Ωσ_h
            Q_J = props['activation_energy'] * 1.602e-19  # Convert eV to J
            Q_eff = Q_J - Omega * sigma_hydro_Pa
            D_bulk = np.exp(-Q_J / (DiffusionPhysics.k_B_J * T_K))
            D_stressed = np.exp(-Q_eff / (DiffusionPhysics.k_B_J * T_K))
            D_ratio = D_stressed / D_bulk
            
        elif model == 'vacancy_concentration':
            # Vacancy concentration ratio: C_v/C_v0 = exp(Ωσ_h / (k_B T))
            exponent = Omega * sigma_hydro_Pa / (DiffusionPhysics.k_B_J * T_K)
            D_ratio = np.exp(exponent)  # Assuming diffusion ∝ vacancy concentration
            
        else:
            raise ValueError(f"Unknown model: {model}")
        
        return D_ratio
    
    @staticmethod
    def compute_effective_diffusion_coefficient(sigma_hydro_GPa, T_K=650, material='Silver'):
        """
        Compute effective diffusion coefficient including stress effects
        
        Returns:
        --------
        D_eff : float or array
            Effective diffusion coefficient in m²/s
        """
        props = DiffusionPhysics.get_material_properties(material)
        # Calculate unstressed diffusion coefficient using Arrhenius equation
        D0 = props['prefactor'] * np.exp(-props['activation_energy'] / (DiffusionPhysics.k_B_eV * T_K))
        
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_GPa, T_K, material, 'physics_corrected'
        )
        
        return D0 * D_ratio
    
    @staticmethod
    def compute_mass_flux_gradient(D_ratio_field, unstressed_D, concentration_gradient):
        """
        Compute mass flux considering stress-enhanced diffusion
        
        Parameters:
        -----------
        D_ratio_field : array
            Local D/D₀ ratio field
        unstressed_D : float
            Unstressed diffusion coefficient (m²/s)
        concentration_gradient : array
            Concentration gradient (1/m)
        
        Returns:
        --------
        mass_flux : array
            Mass flux (atoms/m²·s)
        """
        D_eff = unstressed_D * D_ratio_field
        mass_flux = -D_eff * concentration_gradient  # Fick's first law
        
        return mass_flux
    
    @staticmethod
    def compute_dislocation_sink_strength(stress_field, material='Silver'):
        """
        Compute dislocation sink strength enhancement due to stress
        
        Theory: Stress affects dislocation bias for vacancies
        """
        props = DiffusionPhysics.get_material_properties(material)
        shear_modulus = props['shear_modulus']
        
        # Normalized stress effect on sink strength
        # Approximate: sink_strength ∝ 1 + |σ_shear|/μ
        sigma_vm = np.sqrt(np.sum(stress_field**2, axis=0))  # Approximate
        sink_enhancement = 1 + 0.1 * sigma_vm / shear_modulus
        
        return sink_enhancement
    
    @staticmethod
    def compute_activation_volume(sigma_hydro_GPa, D_ratio):
        """
        Compute apparent activation volume from diffusion enhancement
        
        V_act = k_B T * d(ln D)/dσ
        
        Returns:
        --------
        V_act : float or array
            Activation volume in atomic volumes
        """
        # For small stresses, activation volume ~ Ω
        props = DiffusionPhysics.get_material_properties('Silver')
        Omega = props['atomic_volume']
        
        # Approximate derivative
        if isinstance(D_ratio, np.ndarray) and D_ratio.size > 1:
            # For arrays, compute local activation volume
            lnD = np.log(D_ratio)
            # Use finite difference approximation
            V_act = np.gradient(lnD) / np.gradient(sigma_hydro_GPa) * 1.38e-23 * 650 / 1e9
        else:
            # For scalar
            V_act = Omega  # Default approximation
        
        return V_act
    
    @staticmethod
    def compute_diffusion_length(D_ratio, time, D0):
        """
        Compute diffusion length considering stress enhancement
        
        L = sqrt(4 D_eff t)
        
        Returns:
        --------
        L : float or array
            Diffusion length (m)
        """
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
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        """Scan directory for solution files"""
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth']:
            import glob
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        # Sort by modification time (newest first)
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
        """Read simulation file with physics-aware processing"""
        try:
            with open(file_path, 'rb') as f:
                if format_type == 'pt' or file_path.endswith(('.pt', '.pth')):
                    # PyTorch file
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                else:
                    # Pickle file
                    data = pickle.load(f)
            
            # Standardize data structure
            standardized = self._standardize_data(data, file_path)
            return standardized
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data, file_path):
        """Standardize simulation data with physics metadata"""
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
                
                # Extract history
                if 'history' in data:
                    history = data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        # Convert dict to list
                        history_list = []
                        for key in sorted(history.keys()):
                            if isinstance(history[key], dict):
                                history_list.append(history[key])
                        standardized['history'] = history_list
                
                # Extract additional metadata
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                
                # Convert tensors to numpy arrays
                self._convert_tensors(standardized)
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
        
        return standardized
    
    def _convert_tensors(self, data):
        """Convert PyTorch tensors to numpy arrays recursively"""
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
        """Load all solutions with physics processing"""
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
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Create positional indices
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute divisor term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))
        
        # Create positional encoding
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
    
    Key Principles:
    1. Defect Type is a HARD CONSTRAINT (Major Aspect).
    2. Angular Orientation drives the Spatial Locality Kernel (Major Aspect).
    3. Attention = Learned Similarity * Angular Kernel * Defect Mask.
    """
    def __init__(self, d_model=64, nhead=8, num_layers=3, 
                 spatial_sigma=10.0, temperature=1.0, locality_weight_factor=0.5):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma  # Width of the angular bracketing kernel
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor 
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection - FIXED: Now expects exactly 15 input features
        self.input_proj = nn.Linear(15, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
    
    def set_spatial_parameters(self, spatial_sigma=None, locality_weight_factor=None):
        """Update spatial parameters dynamically"""
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
        if locality_weight_factor is not None:
            self.locality_weight_factor = locality_weight_factor

    def debug_feature_dimensions(self, params_list, target_angle_deg):
        """Debug method to check feature dimensions"""
        encoded = self.encode_parameters(params_list, target_angle_deg)
        print(f"Debug: Encoded shape: {encoded.shape}")
        print(f"Debug: Number of features: {encoded.shape[1]}")
        
        # Print first encoded vector
        if len(params_list) > 0:
            print(f"Debug: First encoded vector: {encoded[0]}")
            print(f"Debug: Number of non-zero elements: {torch.sum(encoded[0] != 0).item()}")
        
        return encoded.shape
    
    def compute_angular_bracketing_kernel(self, source_params, target_params):
        """
        Compute the Angular Bracketing Kernel and Defect Mask.
        
        Returns:
            spatial_weights: Gaussian decay based on angular distance.
            defect_mask: 1.0 for same defect type, epsilon for different.
            angular_distances: List of angular distances.
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
            
            # Calculate cyclic angular distance
            raw_diff = abs(src_theta_deg - target_theta_deg)
            angular_dist = min(raw_diff, 360 - raw_diff)
            angular_distances.append(angular_dist)
            
            # --- MAJOR ASPECT 1: Defect Type Gating ---
            # If defect types differ, weight is effectively zero.
            if src.get('defect_type') == target_defect:
                defect_mask.append(1.0)
            else:
                defect_mask.append(1e-6) # Near zero to avoid NaN in log, but effectively ignored
            
            # --- MAJOR ASPECT 2: Angular Bracketing Kernel ---
            # Gaussian kernel centered at target angle.
            # High weight for sources that bracket the target angle.
            # Sigma controls the "width" of the bracketing window.
            weight = np.exp(-0.5 * (angular_dist / self.spatial_sigma) ** 2)
            spatial_weights.append(weight)
            
        return np.array(spatial_weights), np.array(defect_mask), np.array(angular_distances)
    
    def visualize_angular_kernel(self, target_angle_deg=54.7, figsize=(12, 8)):
        """Visualize the Angular Bracketing Kernel (Spatial Locality)"""
        angles = np.linspace(0, 180, 361)
        weights = []
        
        # Simulate source params list for visualization
        dummy_sources = [{'theta': np.radians(a), 'defect_type': 'Twin'} for a in angles]
        dummy_target = {'theta': np.radians(target_angle_deg), 'defect_type': 'Twin'}
        
        # Compute weights using the kernel logic
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
        
        # Highlight sigma region
        ax.fill_between([target_angle_deg - self.spatial_sigma, target_angle_deg + self.spatial_sigma], 
                       0, 1, color='blue', alpha=0.1, label=f'±1$\sigma$ Region')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def encode_parameters(self, params_list, target_angle_deg):
        """Encode parameters into transformer input - FIXED to return exactly 15 features"""
        encoded = []
        for params in params_list:
            # Create feature vector
            features = []
            
            # Numeric features (3 features)
            features.append(params.get('eps0', 0.707) / 3.0)
            features.append(params.get('kappa', 0.6) / 2.0)
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)
            
            # One-hot encoding for defect type (4 features)
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            
            # Shape encoding (4 features)
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
            
            # Orientation features (3 features)
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)  # Handle cyclic nature
            features.append(np.exp(-angle_diff / 45.0))
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))
            
            # Habit plane proximity (1 feature)
            habit_distance = abs(theta_deg - 54.7)
            habit_distance = min(habit_distance, 360 - habit_distance)  # Handle cyclic nature
            features.append(np.exp(-habit_distance / 15.0))
            
            # Verify we have exactly 15 features
            if len(features) != 15:
                st.warning(f"Warning: Expected 15 features, got {len(features)}. Padding or truncating.")
            
            # Pad with zeros if fewer than 15
            while len(features) < 15:
                features.append(0.0)
            
            # Truncate if more than 15
            features = features[:15]
            
            encoded.append(features)
        
        return torch.FloatTensor(encoded)
    
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """
        Interpolate full spatial stress fields using Theory-Informed Attention.
        
        Attention Logic:
        1. Compute Transformer Embeddings.
        2. Compute Angular Bracketing Kernel (Spatial Locality).
        3. Compute Defect Type Mask (Hard Constraint).
        4. Attention = Softmax(Transformer_Score * Spatial_Kernel * Defect_Mask).
        """
        if not sources:
            st.warning("No sources provided for interpolation.")
            return None
        
        try:
            # Extract source parameters and fields
            source_params = []
            source_fields = []
            source_indices = []  # Track original indices
            
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    st.warning(f"Skipping source {i}: missing params or history")
                    continue
                
                source_params.append(src['params'])
                source_indices.append(i)
                
                # Get last frame stress fields
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        # Extract all stress components
                        stress_fields = last_frame['stresses']
                        
                        # Get von Mises if available, otherwise compute
                        if 'von_mises' in stress_fields:
                            vm = stress_fields['von_mises']
                        else:
                            # Compute von Mises from components
                            vm = self.compute_von_mises(stress_fields)
                        
                        # Get hydrostatic stress
                        if 'sigma_hydro' in stress_fields:
                            hydro = stress_fields['sigma_hydro']
                        else:
                            hydro = self.compute_hydrostatic(stress_fields)
                        
                        # Get stress magnitude
                        if 'sigma_mag' in stress_fields:
                            mag = stress_fields['sigma_mag']
                        else:
                            mag = np.sqrt(vm**2 + hydro**2)
                        
                        source_fields.append({
                            'von_mises': vm,
                            'sigma_hydro': hydro,
                            'sigma_mag': mag,
                            'source_index': i,
                            'source_params': src['params']
                        })
                    else:
                        st.warning(f"Skipping source {i}: no stress fields found")
                        continue
                else:
                    st.warning(f"Skipping source {i}: invalid history")
                    continue
            
            if not source_params or not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Check if all fields have same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                # Resize to common shape
                target_shape = shapes[0]  # Use first shape
                resized_fields = []
                for fields in source_fields:
                    resized = {}
                    for key, field in fields.items():
                        if key in ['von_mises', 'sigma_hydro', 'sigma_mag'] and field.shape != target_shape:
                            # Resize using interpolation
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            resized[key] = zoom(field, factors, order=1)
                        elif key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                            resized[key] = field
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                source_fields = resized_fields
            
            # Debug: Check feature dimensions
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Ensure we have exactly 15 features
            if source_features.shape[1] != 15 or target_features.shape[1] != 15:
                st.warning(f"Feature dimension mismatch: source_features shape={source_features.shape}, target_features shape={target_features.shape}")
            
            # Force reshape to 15 features
            if source_features.shape[1] < 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            # --- STEP 1: COMPUTE ANGULAR BRACKETING KERNEL & DEFECT MASK ---
            # This encodes the PHYSICS priors before looking at the transformer
            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )
            
            # --- STEP 2: TRANSFORMER ENCODING ---
            # Prepare transformer input
            batch_size = 1
            seq_len = len(source_features) + 1  # Sources + target
            
            # Create sequence: [target, source1, source2, ...]
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)  # [1, seq_len, features]
            
            # Apply input projection
            proj_features = self.input_proj(all_features)
            
            # Add positional encoding
            proj_features = self.pos_encoder(proj_features)
            
            # Transformer encoding
            transformer_output = self.transformer(proj_features)
            
            # Extract target representation (first in sequence) and source representations
            target_rep = transformer_output[:, 0, :]  # [1, d_model]
            source_reps = transformer_output[:, 1:, :]  # [1, N, d_model]
            
            # --- STEP 3: THEORY-INFORMED ATTENTION ---
            # Instead of using the Transformer's internal softmax directly,
            # we calculate attention using the embeddings but biased by our theory.
            
            # 3a. Learned Similarity (Dot Product)
            # Score_t = Target_rep . Source_reps^T
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1) # [1, N]
            attn_scores = attn_scores / np.sqrt(self.d_model)
            
            # 3b. Apply Temperature
            attn_scores = attn_scores / self.temperature
            
            # 3c. Convert kernels to tensors
            spatial_kernel_tensor = torch.FloatTensor(spatial_kernel).unsqueeze(0) # [1, N]
            defect_mask_tensor = torch.FloatTensor(defect_mask).unsqueeze(0) # [1, N]
            
            # 3d. Combine: Attention = Learned * Spatial * Defect
            # This enforces that Attention is high ONLY if:
            # 1. Features match (Learned)
            # 2. Angle is close (Spatial Kernel)
            # 3. Defect type matches (Defect Mask)
            biased_scores = attn_scores * spatial_kernel_tensor * defect_mask_tensor
            
            # 3e. Final Softmax
            final_attention_weights = torch.softmax(biased_scores, dim=-1).squeeze().detach().cpu().numpy()
            
            # --- METRICS ---
            entropy_final = self._calculate_entropy(final_attention_weights)
            
            # --- STEP 4: INTERPOLATION ---
            # Interpolate spatial fields using the Theory-Informed Attention Weights
            interpolated_fields = {}
            shape = source_fields[0]['von_mises'].shape
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += final_attention_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            # --- STEP 5: COMPUTE DIFFUSION ENHANCEMENT ---
            # Add diffusion calculations to the interpolated fields
            if 'sigma_hydro' in interpolated_fields:
                sigma_hydro = interpolated_fields['sigma_hydro']
                
                # Compute diffusion enhancement using default parameters (Silver, 650K)
                D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro, T_K=650, material='Silver', model='physics_corrected'
                )
                
                # Compute effective diffusion coefficient
                props = DiffusionPhysics.get_material_properties('Silver')
                D0 = props['prefactor'] * np.exp(-props['activation_energy'] / 
                                                (DiffusionPhysics.k_B_eV * 650))
                D_eff = D0 * D_ratio
                
                # Compute vacancy concentration ratio
                vacancy_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                    sigma_hydro, T_K=650, material='Silver', model='vacancy_concentration'
                )
                
                # Add to interpolated fields
                interpolated_fields['diffusion_ratio'] = D_ratio
                interpolated_fields['diffusion_effective'] = D_eff
                interpolated_fields['vacancy_ratio'] = vacancy_ratio
                
                # Compute gradient of diffusion enhancement
                grad_x, grad_y = np.gradient(D_ratio)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                interpolated_fields['diffusion_gradient'] = grad_magnitude
            
            # Compute additional metrics
            max_vm = np.max(interpolated_fields['von_mises'])
            max_hydro = np.max(np.abs(interpolated_fields['sigma_hydro']))
            
            # Extract source theta values for visualization
            source_theta_degrees = []
            
            for src in source_params:
                theta_rad = src.get('theta', 0.0)
                theta_deg = np.degrees(theta_rad) % 360  # Normalize to [0, 360)
                source_theta_degrees.append(theta_deg)
            
            # Prepare diffusion statistics if available
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
                        'max': float(max_vm),
                        'mean': float(np.mean(interpolated_fields['von_mises'])),
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
                'source_fields': source_fields  # Store source fields for comparison
            }
        
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def compute_von_mises(self, stress_fields):
        """Compute von Mises stress from stress components"""
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
        return np.zeros((100, 100))  # Default shape
    
    def compute_hydrostatic(self, stress_fields):
        """Compute hydrostatic stress from stress components"""
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            return (sxx + syy + szz) / 3
        return np.zeros((100, 100))  # Default shape
    
    def _calculate_entropy(self, weights):
        """Calculate entropy of weight distribution"""
        weights = np.array(weights)
        weights = weights[weights > 0]  # Remove zeros
        
        if len(weights) == 0:
            return 0.0
        
        weights = weights / weights.sum()
        return -np.sum(weights * np.log(weights + 1e-10))  # Add small epsilon to avoid log(0)

# =============================================
# ENHANCED HEATMAP VISUALIZER WITH DIFFUSION
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with diffusion visualization capabilities"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.diffusion_physics = DiffusionPhysics()
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                            cmap_name='viridis', figsize=(12, 10),
                            colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                            show_stats=True, target_angle=None, defect_type=None,
                            show_colorbar=True, aspect_ratio='equal', dpi=300):
        """Create enhanced heat map with chosen colormap and publication styling"""
        
        # Create figure with specified DPI
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')  # Default fallback
        
        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = np.nanmin(stress_field)
        if vmax is None:
            vmax = np.nanmax(stress_field)
        
        # Create heatmap
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect=aspect_ratio, interpolation='bilinear', origin='lower')
        
        # Add colorbar with enhanced styling
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(colorbar_label, fontsize=16, fontweight='bold')
            cbar.ax.tick_params(labelsize=14)
        
        # Customize plot with publication styling
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
        
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=16, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=16, fontweight='bold')
        
        # Add grid with subtle styling
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        
        # Add statistics annotation with enhanced styling
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\n"
                         f"Min: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa\n"
                         f"Std: {np.nanstd(stress_field):.3f} GPa")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()
        return fig
    
    def create_diffusion_heatmap(self, sigma_hydro_field, title="Diffusion Enhancement Map",
                               T_K=650, material='Silver', cmap_name='RdBu_r',
                               figsize=(12, 10), dpi=300, log_scale=True,
                               show_stats=True, target_angle=None, defect_type=None,
                               show_colorbar=True, aspect_ratio='equal',
                               model='physics_corrected'):
        """
        Create heatmap showing diffusion enhancement due to hydrostatic stress
        """
        # Compute diffusion enhancement
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('RdBu_r')  # Good for showing enhancement/suppression
        
        # Apply log scale if requested
        if log_scale:
            # Transform to log10 for visualization
            # Clip to avoid log(0) and extreme values
            log_data = np.log10(np.clip(D_ratio, 0.1, 10))
            vmin, vmax = -1, 1  # Show from 0.1x to 10x enhancement
            im = ax.imshow(log_data, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', origin='lower')
            
            # Create custom colorbar labels for log scale
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(r"$log_{10}(D/D_0)$", fontsize=16, fontweight='bold')
                
                # Set ticks at meaningful values (0.1, 0.5, 1, 2, 10)
                ticks = np.array([0.1, 0.5, 1, 2, 5, 10])
                log_ticks = np.log10(ticks)
                cbar.set_ticks(log_ticks)
                cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
                cbar.ax.tick_params(labelsize=14)
        else:
            # Linear scale
            vmin, vmax = 0.1, 10
            im = ax.imshow(D_ratio, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect=aspect_ratio, interpolation='bilinear', origin='lower',
                          norm=LogNorm(vmin=vmin, vmax=vmax) if log_scale else None)
            
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(r"$D/D_0$", fontsize=16, fontweight='bold')
                cbar.ax.tick_params(labelsize=14)
        
        # Enhanced title
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, {defect_type}, T = {T_K} K"
        
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (nm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Y Position (nm)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        
        # Add habit plane reference if angle is close to 54.7°
        if target_angle is not None:
            angular_diff = abs(target_angle - 54.7)
            angular_diff = min(angular_diff, 360 - angular_diff)
            if angular_diff < 5:
                ax.text(0.02, 0.02, f"Near habit plane (Δθ = {angular_diff:.1f}°)",
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Add statistics annotation
        if show_stats:
            # Compute statistics
            enhancement_regions = D_ratio > 1.0
            suppression_regions = D_ratio < 1.0
            
            stats_text = (
                f"Max Enhancement: {np.max(D_ratio):.2f}x\n"
                f"Min (Suppression): {np.min(D_ratio):.2f}x\n"
                f"Mean: {np.mean(D_ratio):.2f}x\n"
                f"Enhanced Area: {np.sum(enhancement_regions)/D_ratio.size*100:.1f}%\n"
                f"Suppressed Area: {np.sum(suppression_regions)/D_ratio.size*100:.1f}%"
            )
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        plt.tight_layout()
        return fig, D_ratio
    
    def create_diffusion_3d_surface(self, sigma_hydro_field, title="3D Diffusion Surface",
                                  T_K=650, material='Silver', cmap_name='RdBu_r',
                                  figsize=(14, 10), dpi=300, log_scale=True,
                                  target_angle=None, defect_type=None,
                                  model='physics_corrected'):
        """
        Create 3D surface plot showing diffusion enhancement landscape
        """
        # Compute diffusion enhancement
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        
        # Apply log transform for visualization if requested
        if log_scale:
            plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
            z_label = r"$log_{10}(D/D_0)$"
        else:
            plot_data = D_ratio
            z_label = r"$D/D_0$"
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(D_ratio.shape[1])
        y = np.arange(D_ratio.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('RdBu_r')
        
        # Normalize for coloring
        if log_scale:
            norm = Normalize(vmin=-1, vmax=1)
        else:
            norm = Normalize(vmin=0.1, vmax=10)
        
        # Create surface plot with enhanced styling
        surf = ax.plot_surface(X, Y, plot_data, cmap=cmap, norm=norm,
                              linewidth=0.5, antialiased=True, alpha=0.85,
                              rstride=1, cstride=1, edgecolor=(0, 0, 0, 0.1))
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label(z_label, fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        
        # Enhanced title
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, {defect_type}, T = {T_K} K"
        
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel(z_label, fontsize=16, fontweight='bold', labelpad=10)
        
        # Set tick parameters
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        # Adjust view angle for better visibility
        ax.view_init(elev=30, azim=225)
        
        # Add text annotation showing key statistics
        max_enhance = np.max(D_ratio)
        min_enhance = np.min(D_ratio)
        mean_enhance = np.mean(D_ratio)
        
        stats_text = f"Max: {max_enhance:.2f}x\nMin: {min_enhance:.2f}x\nMean: {mean_enhance:.2f}x"
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        return fig, D_ratio

    def create_interactive_diffusion_3d(self, sigma_hydro_field, title="Interactive Diffusion 3D",
                                      T_K=650, material='Silver', cmap_name='RdBu',
                                      width=1000, height=800, log_scale=True,
                                      target_angle=None, defect_type=None,
                                      model='physics_corrected'):
        """
        Create interactive 3D surface plot using Plotly
        """
        # Compute diffusion enhancement
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, model
        )
        
        # Create meshgrid
        x = np.arange(D_ratio.shape[1])
        y = np.arange(D_ratio.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Apply log transform if requested
        if log_scale:
            plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
            z_label = "log₁₀(D/D₀)"
            hover_format = "%.3f"
        else:
            plot_data = D_ratio
            z_label = "D/D₀"
            hover_format = "%.2f"
        
        # Create hover text
        hover_text = []
        for i in range(D_ratio.shape[0]):
            row_text = []
            for j in range(D_ratio.shape[1]):
                hydro_stress = sigma_hydro_field[i, j]
                enhancement = D_ratio[i, j]
                row_text.append(
                    f"Position: ({i}, {j})<br>"
                    f"σₕ: {hydro_stress:.3f} GPa<br>"
                    f"D/D₀: {enhancement:.3f}<br>"
                    f"{'Enhanced' if enhancement > 1 else 'Suppressed'}"
                )
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
            plotly_cmap = 'Viridis'
        
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
                        font=dict(size=16, family='Arial', color='black')
                    ),
                    tickfont=dict(size=14),
                    thickness=25,
                    len=0.8
                )
            )
        except Exception as e:
            print(f"Warning: Could not use colormap {plotly_cmap}, falling back to 'Viridis'. Error: {e}")
            surface_trace = go.Surface(
                z=plot_data,
                x=X,
                y=Y,
                colorscale='Viridis',
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
                        font=dict(size=16, family='Arial', color='black')
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
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text=title_str,
                font=dict(size=24, family="Arial Black", color='darkblue'),
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
                    backgroundcolor='white',
                    showbackground=True
                ),
                yaxis=dict(
                    title=dict(text="Y Position", font=dict(size=18, color="black")),
                    tickfont=dict(size=14),
                    gridcolor='rgb(200, 200, 200)',
                    backgroundcolor='white',
                    showbackground=True
                ),
                zaxis=dict(
                    title=dict(text=z_label, font=dict(size=18, color="black")),
                    tickfont=dict(size=14),
                    gridcolor='rgb(200, 200, 200)',
                    backgroundcolor='white',
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
            font=dict(size=12, color="black", family="Arial"),
            align="left",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )
        
        return fig, D_ratio
    
    def create_publication_quality_plot(self, stress_field, title="Publication Quality Plot",
                                      cmap_name='viridis', figsize=(10, 8),
                                      target_angle=None, defect_type=None):
        """Create publication-quality figure with enhanced styling"""
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        # Use perceptually uniform colormap
        cmap = plt.get_cmap(cmap_name)
        
        # Create heatmap with high-quality interpolation
        im = ax.imshow(stress_field, cmap=cmap, aspect='equal', 
                      interpolation='bicubic', origin='lower')
        
        # Professional colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Stress (GPa)", fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Professional title and labels
        title_str = title
        if target_angle is not None:
            title_str = f"{title} (θ = {target_angle:.1f}°)"
        
        ax.set_title(title_str, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('X Position (nm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (nm)', fontsize=14, fontweight='bold')
        
        # Remove unnecessary spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('gray')
        
        # Add scale bar if dimensions are known
        if stress_field.shape[0] > 50:
            # Add scale bar (assuming 1 pixel = 1 nm)
            scale_length = stress_field.shape[1] // 10
            ax.plot([10, 10 + scale_length], 
                   [stress_field.shape[0] - 10, stress_field.shape[0] - 10], 
                   'k-', linewidth=3)
            ax.text(10 + scale_length/2, stress_field.shape[0] - 20, 
                   f'{scale_length} nm', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_heatmap(self, stress_field, title="Stress Heat Map",
                                 cmap_name='viridis', width=800, height=700,
                                 target_angle=None, defect_type=None):
        """Create interactive heatmap with Plotly with enhanced styling"""
        try:
            # Validate colormap
            if cmap_name not in px.colors.named_colorscales():
                cmap_name = 'viridis'  # Default fallback
                st.warning(f"Colormap {cmap_name} not found in Plotly, using viridis instead.")
            
            # Create hover text with enhanced information
            hover_text = []
            for i in range(stress_field.shape[0]):
                row_text = []
                for j in range(stress_field.shape[1]):
                    if target_angle is not None:
                        row_text.append(f"Position: ({i}, {j})<br>Stress: {stress_field[i, j]:.4f} GPa<br>θ: {target_angle:.1f}°")
                    else:
                        row_text.append(f"Position: ({i}, {j})<br>Stress: {stress_field[i, j]:.4f} GPa")
                hover_text.append(row_text)
            
            # Create heatmap trace
            heatmap_trace = go.Heatmap(
                z=stress_field,
                colorscale=cmap_name,
                zmin=np.nanmin(stress_field),
                zmax=np.nanmax(stress_field),
                hoverinfo='text',
                text=hover_text,
                colorbar=dict(
                    title=dict(
                        text="Stress (GPa)",
                        font=dict(size=16, family='Arial', color='black'),
                        side="right"
                    ),
                    tickfont=dict(size=14, family='Arial'),
                    thickness=20,
                    len=0.8
                )
            )
            
            # Create figure
            fig = go.Figure(data=[heatmap_trace])
            
            # Enhanced title
            title_str = title
            if target_angle is not None and defect_type is not None:
                title_str = f"{title}<br>θ = {target_angle:.1f}°, Defect: {defect_type}"
            
            # Update layout with publication styling
            fig.update_layout(
                title=dict(
                    text=title_str,
                    font=dict(size=24, family="Arial Black", color='darkblue'),
                    x=0.5,
                    y=0.95
                ),
                width=width,
                height=height,
                xaxis=dict(
                    title=dict(text="X Position", font=dict(size=18, family="Arial", color="black")),
                    tickfont=dict(size=14, family='Arial'),
                    gridcolor='rgba(150, 150, 150, 0.3)',
                    scaleanchor="y",
                    scaleratio=1
                ),
                yaxis=dict(
                    title=dict(text="Y Position", font=dict(size=18, family="Arial", color="black")),
                    tickfont=dict(size=14, family='Arial'),
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
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating interactive heatmap: {e}")
            # Return a simple figure as fallback
            fig = go.Figure()
            fig.add_annotation(text="Error creating heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    #
    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info,
                               target_angle, defect_type, component='von_mises',
                               cmap_name='viridis', figsize=(24, 18),
                               ground_truth_index=None, defect_type_filter=None,
                               config=None):
        """
        ENHANCED COMPARISON DASHBOARD with all requested features:
        1. Over 50 colormap options including rainbow, turbo, jet, inferno
        2. Proper vertical and horizontal padding to prevent text overlap
        3. Legends placed in best locations
        4. Defect type names clearly displayed in source selection
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary for customizing the dashboard. Can include:
            - 'show_colorbar': bool - Show colorbars on heatmaps
            - 'show_grid': bool - Show grid lines
            - 'show_stats': bool - Show statistics text boxes
            - 'show_legends': bool - Show plot legends
            - 'legend_position': str - Legend position ('best', 'upper right', etc.)
            - 'title_fontsize': int - Title font size
            - 'label_fontsize': int - Axis label font size
            - 'tick_fontsize': int - Tick label font size
            - 'colorbar_pad': float - Colorbar padding
            - 'custom_titles': dict - Custom titles for each subplot
            - 'custom_labels': dict - Custom axis labels
            - 'hide_plots': list - List of plot indices to hide (0-7)
            - 'custom_colormaps': dict - Custom colormap for each heatmap
        """
        
        # Default configuration with enhanced spacing
        default_config = {
            'show_colorbar': True,
            'show_grid': True,
            'show_stats': True,
            'show_legends': True,
            'legend_position': 'best',
            'title_fontsize': 16,
            'label_fontsize': 12,
            'tick_fontsize': 10,
            'colorbar_pad': 0.06,
            'grid_alpha': 0.2,
            'custom_titles': {},
            'custom_labels': {},
            'hide_plots': [],
            'custom_colormaps': {},
            'custom_legend_labels': {},
            'colorbar_label_fontsize': 12,
            'stats_box_alpha': 0.9,
            'tight_layout': True,
            'figure_dpi': 300,
            'plot_background_color': None,
            'grid_linewidth': 0.5,
            'grid_linestyle': '--',
            'vertical_padding': 0.3,
            'horizontal_padding': 0.3,
            'wspace': 0.4,
            'hspace': 0.4
        }
        
        # Merge user config with defaults
        if config is None:
            config = {}
        merged_config = default_config.copy()
        merged_config.update(config)
        
        # --- STEP 1: FILTER SOURCES BY DEFECT TYPE ---
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
                st.warning(f"No sources found matching defect type: {defect_type_filter}")
                return None
    
        # --- STEP 2: CREATE FIGURE WITH ENHANCED SPACING ---
        fig = plt.figure(figsize=figsize, dpi=merged_config['figure_dpi'])
        
        # Enhanced grid specification with proper spacing
        gs = fig.add_gridspec(3, 3, 
                             hspace=merged_config['hspace'], 
                             wspace=merged_config['wspace'],
                             left=0.07, right=0.95, top=0.92, bottom=0.05,
                             width_ratios=[1, 1, 1],
                             height_ratios=[1, 1, 1])
        
        # Get custom titles and labels
        custom_titles = merged_config.get('custom_titles', {})
        custom_labels = merged_config.get('custom_labels', {})
        custom_legend_labels = merged_config.get('custom_legend_labels', {})
        custom_colormaps = merged_config.get('custom_colormaps', {})
        
        # Define plot titles with customization
        plot_titles = {
            'plot1': custom_titles.get('plot1', f'Interpolated Result\nθ = {target_angle:.1f}°, {defect_type}'),
            'plot2': custom_titles.get('plot2', f'Ground Truth\nθ = {source_info["theta_degrees"][ground_truth_index]:.1f}°' if ground_truth_index is not None else 'Ground Truth Selection'),
            'plot3': custom_titles.get('plot3', 'Difference Analysis'),
            'plot4': custom_titles.get('plot4', 'Attention vs Spatial Kernel'),
            'plot5': custom_titles.get('plot5', 'Angular Distribution'),
            'plot6': custom_titles.get('plot6', 'Component Statistics'),
            'plot7': custom_titles.get('plot7', 'Defect Type Filter'),
            'plot8': custom_titles.get('plot8', 'Spatial Correlation')
        }
        
        # Define axis labels with customization
        axis_labels = {
            'xlabel': custom_labels.get('xlabel', 'X Position'),
            'ylabel': custom_labels.get('ylabel', 'Y Position'),
            'colorbar': custom_labels.get('colorbar', f'{component.replace("_", " ").title()} (GPa)')
        }
        
        # Define legend labels with customization
        legend_labels = {
            'final_attention': custom_legend_labels.get('final_attention', 'Final Attention'),
            'spatial_kernel': custom_legend_labels.get('spatial_kernel', 'Spatial Kernel'),
            'perfect_correlation': custom_legend_labels.get('perfect_correlation', 'Perfect Correlation'),
            'target': custom_legend_labels.get('target', f'Target (θ={target_angle:.1f}°)'),
            'habit_plane': custom_legend_labels.get('habit_plane', 'Habit Plane (54.7°)'),
            'defect_mask': custom_legend_labels.get('defect_mask', 'Defect Mask'),
            'statistics': custom_legend_labels.get('statistics', {'max': 'Max', 'mean': 'Mean', 'std': 'Std'})
        }
        
        # Define colormaps with customization
        colormaps = {
            'plot1': custom_colormaps.get('plot1', cmap_name),
            'plot2': custom_colormaps.get('plot2', cmap_name),
            'plot3': custom_colormaps.get('plot3', 'RdBu_r'),
            'diffusion': custom_colormaps.get('diffusion', 'RdBu_r')
        }
        
        # Determine vmin and vmax for consistent scaling
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
        
        # Function to create consistent colorbars
        def add_colorbar_with_padding(im, ax, label):
            cbar = plt.colorbar(im, ax=ax, 
                               fraction=0.046, 
                               pad=merged_config['colorbar_pad'])
            cbar.set_label(label, 
                          fontsize=merged_config['colorbar_label_fontsize'], 
                          fontweight='bold')
            cbar.ax.tick_params(labelsize=merged_config['tick_fontsize'])
            return cbar
        
        # --- PLOT 1: Interpolated result ---
        plot1_visible = 0 not in merged_config['hide_plots']
        if plot1_visible:
            ax1 = fig.add_subplot(gs[0, 0])
            if merged_config.get('plot_background_color'):
                ax1.set_facecolor(merged_config['plot_background_color'])
            
            if component in interpolated_fields:
                im1 = ax1.imshow(interpolated_fields[component], 
                               cmap=colormaps['plot1'],
                               vmin=vmin, vmax=vmax, aspect='equal', 
                               interpolation='bilinear', origin='lower')
                
                if merged_config['show_colorbar']:
                    add_colorbar_with_padding(im1, ax1, axis_labels['colorbar'])
                
                ax1.set_title(plot_titles['plot1'], 
                             fontsize=merged_config['title_fontsize'], 
                             fontweight='bold', pad=12)
                ax1.set_xlabel(axis_labels['xlabel'], 
                              fontsize=merged_config['label_fontsize'])
                ax1.set_ylabel(axis_labels['ylabel'], 
                              fontsize=merged_config['label_fontsize'])
                
                if merged_config['show_grid']:
                    ax1.grid(True, alpha=merged_config['grid_alpha'],
                            linestyle=merged_config['grid_linestyle'],
                            linewidth=merged_config['grid_linewidth'])
                
                if merged_config['show_stats']:
                    stats_text = (f"Max: {np.max(interpolated_fields[component]):.3f} GPa\n"
                                f"Min: {np.min(interpolated_fields[component]):.3f} GPa\n"
                                f"Mean: {np.mean(interpolated_fields[component]):.3f} GPa")
                    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                            fontsize=merged_config['tick_fontsize'], fontweight='bold',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', 
                                    facecolor='white', 
                                    alpha=merged_config['stats_box_alpha'],
                                    edgecolor='gray', pad=0.3))
        
        # --- PLOT 2: Ground truth comparison ---
        plot2_visible = 1 not in merged_config['hide_plots']
        if plot2_visible:
            ax2 = fig.add_subplot(gs[0, 1])
            if merged_config.get('plot_background_color'):
                ax2.set_facecolor(merged_config['plot_background_color'])
            
            if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
                gt_field = source_fields[ground_truth_index].get(component)
                if gt_field is not None:
                    gt_theta = source_info['theta_degrees'][ground_truth_index]
                    gt_distance = source_info['distances'][ground_truth_index]
                    
                    im2 = ax2.imshow(gt_field, cmap=colormaps['plot2'],
                                    vmin=vmin, vmax=vmax, aspect='equal', 
                                    interpolation='bilinear', origin='lower')
                    
                    if merged_config['show_colorbar']:
                        add_colorbar_with_padding(im2, ax2, axis_labels['colorbar'])
                    
                    ax2.set_title(plot_titles['plot2'], 
                                 fontsize=merged_config['title_fontsize'], 
                                 fontweight='bold', pad=12)
                    ax2.set_xlabel(axis_labels['xlabel'], 
                                  fontsize=merged_config['label_fontsize'])
                    ax2.set_ylabel(axis_labels['ylabel'], 
                                  fontsize=merged_config['label_fontsize'])
                    
                    if merged_config['show_grid']:
                        ax2.grid(True, alpha=merged_config['grid_alpha'],
                                linestyle=merged_config['grid_linestyle'],
                                linewidth=merged_config['grid_linewidth'])
                    
                    if merged_config['show_stats']:
                        stats_text = (f"Max: {np.max(gt_field):.3f} GPa\n"
                                     f"Min: {np.min(gt_field):.3f} GPa\n"
                                     f"Mean: {np.mean(gt_field):.3f} GPa")
                        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                                fontsize=merged_config['tick_fontsize'], fontweight='bold',
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', 
                                        facecolor='white', 
                                        alpha=merged_config['stats_box_alpha'],
                                        edgecolor='gray', pad=0.3))
        
        # --- PLOT 3: Difference plot ---
        plot3_visible = 2 not in merged_config['hide_plots']
        if plot3_visible:
            ax3 = fig.add_subplot(gs[0, 2])
            if merged_config.get('plot_background_color'):
                ax3.set_facecolor(merged_config['plot_background_color'])
            
            if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
                gt_field = source_fields[ground_truth_index].get(component)
                if gt_field is not None and component in interpolated_fields:
                    diff_field = interpolated_fields[component] - gt_field
                    max_diff = np.max(np.abs(diff_field))
                    
                    im3 = ax3.imshow(diff_field, cmap=colormaps['plot3'],
                                    vmin=-max_diff if max_diff > 0 else None,
                                    vmax=max_diff if max_diff > 0 else None,
                                    aspect='equal', interpolation='bilinear', origin='lower')
                    
                    if merged_config['show_colorbar']:
                        cbar3 = add_colorbar_with_padding(im3, ax3, 'Difference (GPa)')
                    
                    ax3.set_title(plot_titles['plot3'], 
                                 fontsize=merged_config['title_fontsize'], 
                                 fontweight='bold', pad=12)
                    ax3.set_xlabel(axis_labels['xlabel'], 
                                  fontsize=merged_config['label_fontsize'])
                    ax3.set_ylabel(axis_labels['ylabel'], 
                                  fontsize=merged_config['label_fontsize'])
                    
                    if merged_config['show_grid']:
                        ax3.grid(True, alpha=merged_config['grid_alpha'],
                                linestyle=merged_config['grid_linestyle'],
                                linewidth=merged_config['grid_linewidth'])
                    
                    if merged_config['show_stats']:
                        mse = np.mean(diff_field**2)
                        mae = np.mean(np.abs(diff_field))
                        rmse = np.sqrt(mse)
                        
                        error_text = (f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}")
                        ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes,
                                fontsize=merged_config['tick_fontsize'], fontweight='bold',
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', 
                                        facecolor='white', 
                                        alpha=merged_config['stats_box_alpha'],
                                        edgecolor='black', linewidth=1))
        
        # --- PLOT 4: Weight distribution analysis ---
        plot4_visible = 3 not in merged_config['hide_plots']
        if plot4_visible and 'weights' in source_info:
            ax4 = fig.add_subplot(gs[1, 0])
            if merged_config.get('plot_background_color'):
                ax4.set_facecolor(merged_config['plot_background_color'])
            
            final_weights = source_info['weights']['combined']
            x = range(len(final_weights))
            bars = ax4.bar(x, final_weights, alpha=0.7, 
                          color='steelblue', edgecolor='black', 
                          label=legend_labels['final_attention'])
            
            if 'spatial_kernel' in source_info['weights']:
                spatial_k = source_info['weights']['spatial_kernel']
                ax4.plot(x, spatial_k, 'g--', linewidth=2, 
                        label=legend_labels['spatial_kernel'], alpha=0.8)
            
            if ground_truth_index is not None and 0 <= ground_truth_index < len(bars):
                bars[ground_truth_index].set_color('red')
                bars[ground_truth_index].set_alpha(0.9)
                bars[ground_truth_index].set_edgecolor('black')
            
            ax4.set_xlabel('Source Index', fontsize=merged_config['label_fontsize'])
            ax4.set_ylabel('Weight', fontsize=merged_config['label_fontsize'])
            ax4.set_title(plot_titles['plot4'], 
                         fontsize=merged_config['title_fontsize'], 
                         fontweight='bold', pad=12)
            
            if merged_config['show_grid']:
                ax4.grid(True, alpha=merged_config['grid_alpha'], axis='y')
            
            if merged_config['show_legends']:
                ax4.legend(loc=merged_config['legend_position'], 
                          fontsize=merged_config['tick_fontsize'],
                          framealpha=0.9)
        
        # --- PLOT 5: Angular distribution of sources ---
        plot5_visible = 4 not in merged_config['hide_plots']
        if plot5_visible and 'theta_degrees' in source_info and 'distances' in source_info:
            ax5 = fig.add_subplot(gs[1, 1], projection='polar')
            
            angles_rad = np.radians(source_info['theta_degrees'])
            distances = source_info['distances']
            
            if 'weights' in source_info:
                weights = source_info['weights']['combined']
                sizes = 100 * np.array(weights) / (np.max(weights) + 1e-8)
            else:
                sizes = 50 * np.ones_like(angles_rad)
            
            scatter = ax5.scatter(angles_rad, distances, s=sizes, alpha=0.7, 
                                 c='blue', edgecolors='black', linewidths=0.5)
            
            target_rad = np.radians(target_angle)
            ax5.scatter(target_rad, 0, s=200, c='red', marker='*', 
                       edgecolors='black', linewidth=1.5, 
                       label=legend_labels['target'])
            
            habit_rad = np.radians(54.7)
            ax5.axvline(habit_rad, color='green', alpha=0.5, 
                       linestyle='--', linewidth=1.5, 
                       label=legend_labels['habit_plane'])
            
            ax5.set_title(plot_titles['plot5'], 
                         fontsize=merged_config['title_fontsize'], 
                         fontweight='bold', pad=15)
            ax5.set_theta_zero_location('N')
            ax5.set_theta_direction(-1)
            ax5.set_ylim(0, max(distances) * 1.2 if len(distances) > 0 else 1)
            
            if merged_config['show_grid']:
                ax5.grid(True, alpha=merged_config['grid_alpha'])
            
            if merged_config['show_legends']:
                ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                          fontsize=merged_config['tick_fontsize'],
                          framealpha=0.9, fancybox=True, shadow=True)
        
        # --- PLOT 6: Component statistics comparison ---
        plot6_visible = 5 not in merged_config['hide_plots']
        if plot6_visible and 'statistics' in source_info:
            ax6 = fig.add_subplot(gs[1, 2])
            if merged_config.get('plot_background_color'):
                ax6.set_facecolor(merged_config['plot_background_color'])
            
            components = ['von_mises', 'sigma_hydro', 'sigma_mag']
            component_names = ['Von Mises', 'Hydrostatic', 'Stress Magnitude']
            
            stats_data = []
            for comp in components:
                if comp in source_info['statistics']:
                    stats = source_info['statistics'][comp]
                    # FIXED: Handle different key names for sigma_hydro vs other components
                    if comp == 'sigma_hydro':
                        # sigma_hydro uses 'max_tension' instead of 'max'
                        max_val = stats.get('max_tension', 0)
                    else:
                        # von_mises and sigma_mag use 'max'
                        max_val = stats.get('max', 0)
                    
                    stats_data.append({
                        'component': comp,
                        'max': max_val,
                        'mean': stats.get('mean', 0),
                        'std': stats.get('std', 0)
                    })
            
            if stats_data:
                x = np.arange(len(components))
                width = 0.25
                
                max_values = [stats['max'] for stats in stats_data]
                mean_values = [stats['mean'] for stats in stats_data]
                std_values = [stats['std'] for stats in stats_data]
                
                ax6.bar(x - width, max_values, width, 
                       label=legend_labels['statistics']['max'], 
                       color='red', alpha=0.7)
                ax6.bar(x, mean_values, width, 
                       label=legend_labels['statistics']['mean'], 
                       color='blue', alpha=0.7)
                ax6.bar(x + width, std_values, width, 
                       label=legend_labels['statistics']['std'], 
                       color='green', alpha=0.7)
                
                ax6.set_xlabel('Stress Component', 
                              fontsize=merged_config['label_fontsize'])
                ax6.set_ylabel('Value (GPa)', 
                              fontsize=merged_config['label_fontsize'])
                ax6.set_title(plot_titles['plot6'], 
                             fontsize=merged_config['title_fontsize'], 
                             fontweight='bold', pad=12)
                ax6.set_xticks(x)
                ax6.set_xticklabels(component_names, rotation=15,
                                   fontsize=merged_config['tick_fontsize'])
                
                if merged_config['show_legends']:
                    ax6.legend(loc=merged_config['legend_position'], 
                              fontsize=merged_config['tick_fontsize'],
                              framealpha=0.9)
                
                if merged_config['show_grid']:
                    ax6.grid(True, alpha=merged_config['grid_alpha'])
        
        # --- PLOT 7: Defect Type Gating Visualization ---
        plot7_visible = 6 not in merged_config['hide_plots']
        if plot7_visible and 'weights' in source_info and 'defect_mask' in source_info['weights']:
            ax7 = fig.add_subplot(gs[2, 0])
            if merged_config.get('plot_background_color'):
                ax7.set_facecolor(merged_config['plot_background_color'])
            
            defect_masks = source_info['weights']['defect_mask']
            x_pos = np.arange(len(defect_masks))
            
            ax7.bar(x_pos, defect_masks, color='purple', alpha=0.6, 
                   label=legend_labels['defect_mask'])
            
            # Highlight active defect types
            for i, mask_val in enumerate(defect_masks):
                if mask_val > 0.1:
                    ax7.text(i, mask_val + 0.02, 'Active', ha='center', 
                            fontsize=8, fontweight='bold')
            
            ax7.set_xlabel('Source Index', 
                          fontsize=merged_config['label_fontsize'])
            ax7.set_ylabel('Gating Weight', 
                          fontsize=merged_config['label_fontsize'])
            ax7.set_title(plot_titles['plot7'], 
                         fontsize=merged_config['title_fontsize'], 
                         fontweight='bold', pad=12)
            ax7.set_ylim([0, 1.1])
            
            if merged_config['show_legends']:
                ax7.legend(loc=merged_config['legend_position'], 
                          fontsize=merged_config['tick_fontsize'],
                          framealpha=0.9)
        
        # --- PLOT 8: Spatial Correlation ---
        plot8_visible = 7 not in merged_config['hide_plots']
        if plot8_visible and ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            ax8 = fig.add_subplot(gs[2, 1:])
            if merged_config.get('plot_background_color'):
                ax8.set_facecolor(merged_config['plot_background_color'])
            
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None and component in interpolated_fields:
                interp_flat = interpolated_fields[component].flatten()
                gt_flat = gt_field.flatten()
                
                scatter = ax8.scatter(gt_flat, interp_flat, alpha=0.5, s=10, c='navy')
                
                min_val = min(np.min(gt_flat), np.min(interp_flat))
                max_val = max(np.max(gt_flat), np.max(interp_flat))
                ax8.plot([min_val, max_val], [min_val, max_val], 'r--', 
                        linewidth=2, label=legend_labels['perfect_correlation'], 
                        alpha=0.7)
                
                from scipy.stats import pearsonr
                try:
                    corr_coef, _ = pearsonr(gt_flat, interp_flat)
                except:
                    corr_coef = 0.0
                
                ax8.set_xlabel(f'Ground Truth {component.replace("_", " ").title()} (GPa)', 
                              fontsize=merged_config['label_fontsize'])
                ax8.set_ylabel(f'Interpolated {component.replace("_", " ").title()} (GPa)', 
                              fontsize=merged_config['label_fontsize'])
                ax8.set_title(f'{plot_titles["plot8"]}\nPearson: {corr_coef:.3f}', 
                             fontsize=merged_config['title_fontsize'], 
                             fontweight='bold', pad=12)
                
                if merged_config['show_grid']:
                    ax8.grid(True, alpha=merged_config['grid_alpha'])
                
                if merged_config['show_legends']:
                    ax8.legend(loc=merged_config['legend_position'], 
                              fontsize=merged_config['tick_fontsize'],
                              framealpha=0.9)
                
                if merged_config['show_stats'] and corr_coef > 0:
                    r_squared = corr_coef ** 2
                    stats_text = f'R² = {r_squared:.3f}'
                    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                            fontsize=merged_config['tick_fontsize'], fontweight='bold',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', 
                                    facecolor='white', 
                                    alpha=merged_config['stats_box_alpha']))
        
        # Add custom main title if specified
        custom_suptitle = merged_config.get('custom_suptitle')
        if custom_suptitle:
            suptitle_text = custom_suptitle
        else:
            suptitle_text = f'Theory-Informed Interpolation: Target θ={target_angle:.1f}°, {defect_type}'
        
        plt.suptitle(suptitle_text, fontsize=22, fontweight='bold', y=0.98)
        
        if merged_config['tight_layout']:
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        return fig
    
    #
    def create_interactive_3d_surface(self, stress_field, title="3D Stress Surface",
                                 cmap_name='viridis', width=900, height=700,
                                 target_angle=None, defect_type=None):
        """Create interactive 3D surface plot with Plotly with proper colormap handling"""
        try:
            # Create meshgrid
            x = np.arange(stress_field.shape[1])
            y = np.arange(stress_field.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Create hover text
            hover_text = []
            for i in range(stress_field.shape[0]):
                row_text = []
                for j in range(stress_field.shape[1]):
                    if target_angle is not None:
                        row_text.append(f"X: {j}, Y: {i}<br>Stress: {stress_field[i, j]:.4f} GPa<br>θ: {target_angle:.1f}°")
                    else:
                        row_text.append(f"X: {j}, Y: {i}<br>Stress: {stress_field[i, j]:.4f} GPa")
                hover_text.append(row_text)
            
            # Define colormap mapping for Plotly - FIXED with comprehensive mapping
            plotly_colormap_mapping = {
                # Sequential colormaps
                'viridis': 'Viridis',
                'plasma': 'Plasma',
                'inferno': 'Inferno',
                'magma': 'Magma',
                'cividis': 'Cividis',
                'turbo': 'Turbo',
                'hot': 'Hot',
                'afmhot': 'Hot',
                'gist_heat': 'Hot',
                'copper': 'copper',
                'summer': 'summer',
                'spring': 'spring',
                'autumn': 'autumn',
                'winter': 'winter',
                'bone': 'greys',
                'gray': 'greys',
                'pink': 'pink',
                'gist_gray': 'greys',
                'gist_yarg': 'greys',
                'binary': 'greys',
                'gist_earth': 'earth',
                'terrain': 'earth',
                'ocean': 'deep',
                'gist_stern': 'viridis',
                'gnuplot': 'viridis',
                'gnuplot2': 'viridis',
                'CMRmap': 'viridis',
                'cubehelix': 'viridis',
                'brg': 'rainbow',
                'gist_rainbow': 'rainbow',
                'rainbow': 'rainbow',
                'jet': 'jet',
                'nipy_spectral': 'rainbow',
                'gist_ncar': 'rainbow',
                'hsv': 'hsv',
                
                # Diverging colormaps
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
                
                # Qualitative colormaps
                'tab10': 'Set1',
                'tab20': 'Set2',
                'Set1': 'Set1',
                'Set2': 'Set2',
                'Set3': 'Set3',
                'tab20b': 'Set2',
                'tab20c': 'Set3',
                'Pastel1': 'Pastel1',
                'Pastel2': 'Pastel2',
                'Paired': 'Paired',
                'Accent': 'Accent',
                'Dark2': 'Dark2',
                
                # Special cases
                'twilight': 'viridis',
                'twilight_shifted': 'viridis',
                'flag': 'Set1',
                'prism': 'Set1',
                'Wistia': 'YlOrRd'
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
                # Fallback to Viridis
                plotly_cmap = 'Viridis' if not is_reverse else 'Viridis_r'
            
            # Create surface trace with PROPER colormap handling
            try:
                surface_trace = go.Surface(
                    z=stress_field,
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
                            text="Stress (GPa)",
                            font=dict(size=16, family='Arial', color='black')
                        ),
                        tickfont=dict(size=14),
                        thickness=25,
                        len=0.8
                    )
                )
            except Exception as e:
                print(f"Warning: Could not use colormap {plotly_cmap}, falling back to 'Viridis'. Error: {e}")
                surface_trace = go.Surface(
                    z=stress_field,
                    x=X,
                    y=Y,
                    colorscale='Viridis',
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
                            text="Stress (GPa)",
                            font=dict(size=16, family='Arial', color='black')
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
                title_str = f"{title}<br>θ = {target_angle:.1f}°, Defect: {defect_type}"
            
            # Update layout with publication styling
            fig.update_layout(
                title=dict(
                    text=title_str,
                    font=dict(size=24, family="Arial Black", color='darkblue'),
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
                        backgroundcolor='white',
                        showbackground=True
                    ),
                    yaxis=dict(
                        title=dict(text="Y Position", font=dict(size=18, color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white',
                        showbackground=True
                    ),
                    zaxis=dict(
                        title=dict(text="Stress (GPa)", font=dict(size=18, color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white',
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
            max_stress = np.max(stress_field)
            min_stress = np.min(stress_field)
            mean_stress = np.mean(stress_field)
            
            fig.add_annotation(
                text=f"Max: {max_stress:.2f} GPa<br>Min: {min_stress:.2f} GPa<br>Mean: {mean_stress:.2f} GPa",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color="black", family="Arial"),
                align="left",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating 3D surface: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            fig = go.Figure()
            fig.add_annotation(text="Error creating 3D surface", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_angular_orientation_plot(self, target_angle_deg, defect_type="Unknown",
                                       figsize=(8, 8), show_habit_plane=True):
        """Create polar plot showing angular orientation"""
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection='polar')
        
        # Convert target angle to radians
        theta_rad = np.radians(target_angle_deg)
        
        # Plot defect orientation as a red arrow
        ax.arrow(theta_rad, 0.8, 0, 0.6, width=0.02,
                color='red', alpha=0.8, label=f'Defect Orientation: {target_angle_deg:.1f}°')
        
        # Plot habit plane orientation (54.7°) if requested
        if show_habit_plane:
            habit_plane_rad = np.radians(54.7)
            ax.arrow(habit_plane_rad, 0.8, 0, 0.6, width=0.02,
                    color='blue', alpha=0.5, label='Habit Plane (54.7°)')
        
        # Plot cardinal directions
        for angle, label in [(0, '0°'), (90, '90°'), (180, '180°'), (270, '270°')]:
            ax.axvline(np.radians(angle), color='gray', linestyle='--', alpha=0.3)
        
        # Customize plot
        ax.set_title(f'Defect Orientation\nθ = {target_angle_deg:.1f}°, {defect_type}',
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Add annotation for angular difference from habit plane
        if show_habit_plane:
            angular_diff = abs(target_angle_deg - 54.7)
            angular_diff = min(angular_diff, 360 - angular_diff)  # Handle cyclic nature
            ax.annotate(f'Δθ = {angular_diff:.1f}°\nfrom habit plane',
                       xy=(theta_rad, 1.2), xytext=(theta_rad, 1.4),
                       arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                       fontsize=12, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def create_comparison_heatmaps(self, stress_fields_dict, cmap_name='viridis',
                                 figsize=(18, 6), titles=None, target_angle=None, defect_type=None):
        """Create comparison heatmaps for multiple stress components"""
        n_components = len(stress_fields_dict)
        fig, axes = plt.subplots(1, n_components, figsize=figsize, dpi=300)
        
        if n_components == 1:
            axes = [axes]
        
        if titles is None:
            titles = list(stress_fields_dict.keys())
        
        for idx, ((component_name, stress_field), title) in enumerate(zip(stress_fields_dict.items(), titles)):
            ax = axes[idx]
            
            # Get colormap
            if cmap_name in plt.colormaps():
                cmap = plt.get_cmap(cmap_name)
            else:
                cmap = plt.get_cmap('viridis')
            
            # Create heatmap with equal aspect ratio
            im = ax.imshow(stress_field, cmap=cmap, aspect='equal', interpolation='bilinear', origin='lower')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Stress (GPa)", fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            
            # Customize subplot with publication styling
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=14)
            ax.set_ylabel('Y Position', fontsize=14)
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Set tick parameters
            ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add super title with target parameters
        suptitle = "Stress Component Comparison"
        if target_angle is not None and defect_type is not None:
            suptitle = f"Stress Component Comparison - θ = {target_angle:.1f}°, {defect_type}"
        plt.suptitle(suptitle, fontsize=22, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        return fig
    
    def create_3d_surface_plot(self, stress_field, title="3D Stress Surface",
                             cmap_name='viridis', figsize=(14, 10), target_angle=None, defect_type=None):
        """Create 3D surface plot of stress field with enhanced styling"""
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(stress_field.shape[1])
        y = np.arange(stress_field.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Get colormap
        if cmap_name in plt.colormaps():
            cmap = plt.get_cmap(cmap_name)
        else:
            cmap = plt.get_cmap('viridis')
        
        # Normalize for coloring
        norm = Normalize(vmin=np.nanmin(stress_field), vmax=np.nanmax(stress_field))
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, stress_field, cmap=cmap, norm=norm,
                              linewidth=0, antialiased=True, alpha=0.8, rstride=1, cstride=1)
        
        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Stress (GPa)", fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        
        # Enhanced title
        title_str = title
        if target_angle is not None and defect_type is not None:
            title_str = f"{title}\nθ = {target_angle:.1f}°, Defect: {defect_type}"
        
        # Customize plot with publication styling
        ax.set_title(title_str, fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y Position', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel('Stress (GPa)', fontsize=16, fontweight='bold', labelpad=10)
        
        # Set tick parameters
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='z', labelsize=14)
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        return fig
    
    def get_colormap_preview(self, cmap_name, figsize=(12, 1)):
        """Generate preview of a colormap with enhanced styling"""
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.set_title(f"Colormap: {cmap_name}", fontsize=18, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add value labels with enhanced styling
        ax.text(0, 0.5, "Min", transform=ax.transAxes,
               va='center', ha='right', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.text(1, 0.5, "Max", transform=ax.transAxes,
               va='center', ha='left', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add ticks
        ax.set_xticks([0, 128, 255])
        ax.set_xticklabels(['0.0', '0.5', '1.0'], fontsize=12)
        ax.xaxis.set_ticks_position('bottom')
        
        plt.tight_layout()
        return fig
    
    def create_diffusion_comparison_dashboard(self, sigma_hydro_fields, titles,
                                            T_K=650, material='Silver',
                                            cmap_name='RdBu_r', figsize=(20, 15),
                                            log_scale=True, show_stats=True,
                                            model='physics_corrected'):
        """
        Create comparison dashboard showing multiple diffusion scenarios
        """
        n_scenarios = len(sigma_hydro_fields)
        
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(3, n_scenarios + 1, hspace=0.3, wspace=0.3,
                             width_ratios=[1]*n_scenarios + [0.3])
        
        D_ratios = []
        all_stats = []
        
        # Create subplots for each scenario
        for i, (field, title) in enumerate(zip(sigma_hydro_fields, titles)):
            # Compute diffusion enhancement
            D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                field, T_K, material, model
            )
            D_ratios.append(D_ratio)
            
            # Statistics
            stats = {
                'max': np.max(D_ratio),
                'min': np.min(D_ratio),
                'mean': np.mean(D_ratio),
                'std': np.std(D_ratio),
                'enhanced_area': np.sum(D_ratio > 1.0) / D_ratio.size * 100,
                'suppressed_area': np.sum(D_ratio < 1.0) / D_ratio.size * 100
            }
            all_stats.append(stats)
            
            # Create subplot
            ax = fig.add_subplot(gs[0, i])
            
            # Plot data (log scale if requested)
            if log_scale:
                plot_data = np.log10(np.clip(D_ratio, 0.1, 10))
                vmin, vmax = -1, 1
                cbar_label = r"$log_{10}(D/D_0)$"
            else:
                plot_data = D_ratio
                vmin, vmax = 0.1, 10
                cbar_label = r"$D/D_0$"
            
            im = ax.imshow(plot_data, cmap=cmap_name, vmin=vmin, vmax=vmax,
                          aspect='equal', interpolation='bilinear', origin='lower')
            
            # Add colorbar (individual for each subplot)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)
            ax.grid(True, alpha=0.2)
            
            # Add statistics annotation
            if show_stats:
                stats_text = (f"Max: {stats['max']:.2f}x\n"
                             f"Min: {stats['min']:.2f}x\n"
                             f"Enhanced: {stats['enhanced_area']:.1f}%")
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Create statistics comparison plot (middle row)
        ax_stats = fig.add_subplot(gs[1, :])
        
        x = np.arange(n_scenarios)
        width = 0.2
        
        max_vals = [s['max'] for s in all_stats]
        min_vals = [s['min'] for s in all_stats]
        mean_vals = [s['mean'] for s in all_stats]
        enhanced_areas = [s['enhanced_area'] for s in all_stats]
        
        ax_stats.bar(x - 1.5*width, max_vals, width, label='Max Enhancement', color='red', alpha=0.7)
        ax_stats.bar(x - 0.5*width, mean_vals, width, label='Mean', color='blue', alpha=0.7)
        ax_stats.bar(x + 0.5*width, min_vals, width, label='Min (Suppression)', color='green', alpha=0.7)
        ax_stats.bar(x + 1.5*width, enhanced_areas, width, label='Enhanced Area %', color='orange', alpha=0.7)
        
        ax_stats.set_xlabel('Scenario')
        ax_stats.set_ylabel('Value')
        ax_stats.set_title('Diffusion Statistics Comparison', fontsize=16, fontweight='bold')
        ax_stats.set_xticks(x)
        ax_stats.set_xticklabels(titles, rotation=45, ha='right')
        ax_stats.legend(fontsize=10)
        ax_stats.grid(True, alpha=0.3)
        
        # Create line profile comparison (bottom row)
        ax_profile = fig.add_subplot(gs[2, :])
        
        for i, D_ratio in enumerate(D_ratios):
            # Take horizontal line profile through center
            center_row = D_ratio.shape[0] // 2
            profile = D_ratio[center_row, :]
            
            x_profile = np.linspace(0, 1, len(profile))
            ax_profile.plot(x_profile, profile, label=titles[i], linewidth=2)
        
        ax_profile.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='D/D₀ = 1')
        ax_profile.set_xlabel('Normalized X Position')
        ax_profile.set_ylabel('D/D₀')
        ax_profile.set_title('Diffusion Profile Comparison (Center Line)', fontsize=16, fontweight='bold')
        ax_profile.legend(fontsize=10)
        ax_profile.grid(True, alpha=0.3)
        
        # Set y-axis to log scale for better visualization
        ax_profile.set_yscale('log')
        
        plt.suptitle(f'Diffusion Enhancement Comparison - {material}, T = {T_K} K',
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig, D_ratios, all_stats
    
    def create_phase_diagram_visualization(self, sigma_range=(-5, 5), T_range=(300, 1200),
                                         material='Silver', figsize=(14, 10), dpi=300):
        """
        Create phase diagram showing diffusion enhancement as function of stress and temperature
        """
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
        ax1.set_title(f'Diffusion Enhancement Contours - {material}', fontsize=14, fontweight='bold')
        plt.colorbar(contour, ax=ax1, label='D/D₀')
        
        # 2. 3D surface
        ax2 = axes[0, 1]
        ax2 = fig.add_subplot(222, projection='3d')
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
        
        plt.suptitle(f'Diffusion Enhancement Phase Diagram - {material}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig, D_ratio
    
    def create_diffusion_animation_frame(self, sigma_hydro_field, frame_idx, total_frames,
                                       T_K=650, material='Silver', figsize=(10, 8)):
        """
        Create a single frame for diffusion animation (e.g., time evolution)
        """
        # Compute diffusion enhancement for this frame
        D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
            sigma_hydro_field, T_K, material, 'physics_corrected'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        
        # Plot with frame-specific title
        im = ax.imshow(np.log10(np.clip(D_ratio, 0.1, 10)), cmap='RdBu_r',
                      vmin=-1, vmax=1, aspect='equal', interpolation='bilinear', origin='lower')
        
        ax.set_title(f'Diffusion Enhancement - Frame {frame_idx}/{total_frames}\nT = {T_K} K', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        plt.colorbar(im, ax=ax, label='log₁₀(D/D₀)')
        plt.tight_layout()
        
        return fig, D_ratio
    
    def create_vacancy_concentration_map(self, sigma_hydro_field, T_K=650, material='Silver',
                                       figsize=(12, 10), dpi=300):
        """
        Create heatmap showing vacancy concentration enhancement
        C_v/C_v0 = exp(Ωσ_h/(k_B T))
        """
        # Get material properties
        props = DiffusionPhysics.get_material_properties(material)
        Omega = props['atomic_volume']
        
        # Convert stress to Pa
        sigma_hydro_Pa = sigma_hydro_field * 1e9
        
        # Compute vacancy concentration ratio
        k_B = DiffusionPhysics.k_B_J
        exponent = Omega * sigma_hydro_Pa / (k_B * T_K)
        C_ratio = np.exp(exponent)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Use perceptually uniform sequential colormap for vacancy concentration
        im = ax.imshow(C_ratio, cmap='plasma', aspect='equal',
                      interpolation='bilinear', origin='lower',
                      norm=LogNorm(vmin=0.1, vmax=10))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'$C_v/C_{v0}$', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        
        # Title and labels
        ax.set_title(f'Vacancy Concentration Enhancement\n{material}, T = {T_K} K',
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (nm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (nm)', fontsize=14, fontweight='bold')
        
        # Add statistics
        stats_text = (f"Max: {np.max(C_ratio):.2f}x\n"
                     f"Min: {np.min(C_ratio):.2f}x\n"
                     f"Mean: {np.mean(C_ratio):.2f}x")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=12, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        return fig, C_ratio
    
    def create_diffusion_gradient_map(self, D_ratio_field, figsize=(12, 10), dpi=300):
        """
        Create visualization of diffusion coefficient gradients
        Shows where diffusion changes most rapidly
        """
        # Compute gradient magnitude
        grad_x, grad_y = np.gradient(D_ratio_field)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize for visualization
        grad_norm = grad_magnitude / np.max(grad_magnitude)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        
        # 1. Gradient magnitude
        ax1 = axes[0, 0]
        im1 = ax1.imshow(grad_magnitude, cmap='hot', aspect='equal',
                        interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, label='Gradient Magnitude')
        ax1.set_title('Diffusion Gradient Magnitude', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # 2. Gradient direction (quiver plot, downsampled)
        ax2 = axes[0, 1]
        stride = max(1, D_ratio_field.shape[0] // 20)  # Downsample for clarity
        Y, X = np.mgrid[0:D_ratio_field.shape[0]:stride, 0:D_ratio_field.shape[1]:stride]
        U = grad_x[::stride, ::stride]
        V = grad_y[::stride, ::stride]
        
        ax2.quiver(X, Y, U, V, grad_magnitude[::stride, ::stride],
                  cmap='hot', angles='xy', scale_units='xy', scale=0.5)
        ax2.set_title('Gradient Vector Field', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_aspect('equal')
        
        # 3. D/D₀ field for reference
        ax3 = axes[1, 0]
        im3 = ax3.imshow(D_ratio_field, cmap='RdBu_r', aspect='equal',
                        interpolation='bilinear', origin='lower',
                        norm=LogNorm(vmin=0.1, vmax=10))
        plt.colorbar(im3, ax=ax3, label='D/D₀')
        ax3.set_title('Diffusion Enhancement', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        
        # 4. Laplacian (divergence of gradient)
        ax4 = axes[1, 1]
        laplacian = np.gradient(grad_x)[0] + np.gradient(grad_y)[1]
        im4 = ax4.imshow(laplacian, cmap='coolwarm', aspect='equal',
                        interpolation='bilinear', origin='lower')
        plt.colorbar(im4, ax=ax4, label='Laplacian (∇²D)')
        ax4.set_title('Diffusion Field Curvature', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Y Position')
        
        plt.suptitle('Diffusion Gradient Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig, {'magnitude': grad_magnitude, 'direction_x': grad_x, 'direction_y': grad_y}
    
    def create_stress_diffusion_correlation(self, sigma_hydro_field, D_ratio_field,
                                          figsize=(12, 10), dpi=300):
        """
        Create correlation plot between stress and diffusion enhancement
        """
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
        ax1.set_title('Stress-Diffusion Correlation', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. 2D histogram
        ax2 = axes[0, 1]
        h = ax2.hist2d(sigma_flat, D_flat, bins=50, cmap='viridis', norm=LogNorm())
        plt.colorbar(h[3], ax=ax2, label='Count')
        ax2.set_xlabel('Hydrostatic Stress (GPa)')
        ax2.set_ylabel('D/D₀')
        ax2.set_title('2D Histogram', fontsize=12, fontweight='bold')
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
        ax3.set_title('Binned Correlation', fontsize=12, fontweight='bold')
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
            f"Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.2e})\n"
            f"Max Enhancement: {np.max(D_flat):.2f}x\n"
            f"Min Enhancement: {np.min(D_flat):.2f}x\n"
            f"Mean Enhancement: {np.mean(D_flat):.2f}x\n"
            f"Tensile Regions (σ>0): {np.sum(sigma_flat>0)/len(sigma_flat)*100:.1f}%\n"
            f"Compressive Regions (σ<0): {np.sum(sigma_flat<0)/len(sigma_flat)*100:.1f}%"
        )
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax4.set_title('Correlation Statistics', fontsize=12, fontweight='bold')
        
        plt.suptitle('Stress-Diffusion Enhancement Correlation Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig

# =============================================
# DASHBOARD CONFIGURATION INTERFACE
# =============================================
def create_dashboard_configuration_interface():
    """Create Streamlit interface for customizing the dashboard appearance."""
    
    st.markdown("#### 🎨 Dashboard Customization")
    
    config = {}
    
    with st.expander("📊 General Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            config['show_colorbar'] = st.checkbox("Show Colorbars", value=True)
            config['show_grid'] = st.checkbox("Show Grid Lines", value=True)
            config['show_stats'] = st.checkbox("Show Statistics", value=True)
            config['show_legends'] = st.checkbox("Show Legends", value=True)
        
        with col2:
            config['legend_position'] = st.selectbox(
                "Legend Position",
                ['best', 'upper right', 'upper left', 'lower right', 'lower left', 'right', 'center'],
                index=0
            )
            config['tight_layout'] = st.checkbox("Use Tight Layout", value=True)
    
    with st.expander("🔠 Font Sizes", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            config['title_fontsize'] = st.slider("Title Font Size", 8, 20, 14)
            config['label_fontsize'] = st.slider("Label Font Size", 8, 16, 11)
        with col2:
            config['tick_fontsize'] = st.slider("Tick Font Size", 6, 14, 10)
            config['colorbar_label_fontsize'] = st.slider("Colorbar Label Size", 8, 16, 12)
        with col3:
            config['colorbar_pad'] = st.slider("Colorbar Padding", 0.02, 0.15, 0.06, 0.01)
    
    with st.expander("🎨 Colors & Styles", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            config['grid_alpha'] = st.slider("Grid Opacity", 0.1, 1.0, 0.2, 0.1)
            config['grid_linewidth'] = st.slider("Grid Line Width", 0.1, 2.0, 0.5, 0.1)
            config['stats_box_alpha'] = st.slider("Stats Box Opacity", 0.1, 1.0, 0.9, 0.1)
        
        with col2:
            grid_styles = ['-', '--', '-.', ':']
            config['grid_linestyle'] = st.selectbox("Grid Line Style", grid_styles, index=1)
            
            bg_color = st.color_picker("Background Color", value=None)
            if bg_color:
                config['plot_background_color'] = bg_color
    
    with st.expander("📝 Custom Titles & Labels", expanded=False):
        st.info("Customize titles for individual plots (optional)")
        
        custom_titles = {}
        for i, plot_name in enumerate(['Interpolated Result', 'Ground Truth', 'Difference', 
                                       'Attention Analysis', 'Angular Distribution', 
                                       'Statistics', 'Defect Filter', 'Correlation']):
            custom_title = st.text_input(f"Plot {i+1}: {plot_name}", value="", key=f"title_{i}")
            if custom_title:
                custom_titles[f'plot{i+1}'] = custom_title
        
        if custom_titles:
            config['custom_titles'] = custom_titles
        
        st.markdown("---")
        st.info("Customize axis labels")
        
        col1, col2 = st.columns(2)
        with col1:
            xlabel = st.text_input("X-axis Label", value="X Position")
            ylabel = st.text_input("Y-axis Label", value="Y Position")
        
        with col2:
            colorbar_label = st.text_input("Colorbar Label", value="Value (GPa)")
            suptitle = st.text_input("Main Title", value="")
        
        if xlabel or ylabel or colorbar_label:
            config['custom_labels'] = {
                'xlabel': xlabel,
                'ylabel': ylabel,
                'colorbar': colorbar_label
            }
        
        if suptitle:
            config['custom_suptitle'] = suptitle
    
    with st.expander("🌈 Custom Colormaps", expanded=False):
        st.info("Select different colormaps for each heatmap")
        
        all_cmaps = ALL_COLORMAPS
        
        custom_colormaps = {}
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_colormaps['plot1'] = st.selectbox("Plot 1 Colormap", all_cmaps, 
                                                    index=all_cmaps.index('viridis') if 'viridis' in all_cmaps else 0)
            custom_colormaps['plot3'] = st.selectbox("Plot 3 Colormap", all_cmaps,
                                                    index=all_cmaps.index('RdBu_r') if 'RdBu_r' in all_cmaps else 0)
        with col2:
            custom_colormaps['plot2'] = st.selectbox("Plot 2 Colormap", all_cmaps,
                                                    index=all_cmaps.index('viridis') if 'viridis' in all_cmaps else 0)
            custom_colormaps['diffusion'] = st.selectbox("Diffusion Colormap", all_cmaps,
                                                        index=all_cmaps.index('RdBu_r') if 'RdBu_r' in all_cmaps else 0)
        
        if any(v != 'viridis' for v in custom_colormaps.values()):
            config['custom_colormaps'] = custom_colormaps
    
    with st.expander("🔧 Advanced Controls", expanded=False):
        st.warning("Use with caution - hiding plots may break layout")
        
        hide_options = st.multiselect(
            "Hide Plots",
            options=[
                "1 - Interpolated Result",
                "2 - Ground Truth", 
                "3 - Difference",
                "4 - Attention Analysis",
                "5 - Angular Distribution",
                "6 - Statistics",
                "7 - Defect Filter",
                "8 - Correlation"
            ]
        )
        
        hide_indices = []
        for option in hide_options:
            index = int(option.split(" - ")[0]) - 1
            hide_indices.append(index)
        
        if hide_indices:
            config['hide_plots'] = hide_indices
        
        config['figure_dpi'] = st.selectbox("Figure DPI", [150, 300, 600, 1200], index=1)
        
        # Spacing controls
        st.markdown("---")
        st.info("Layout Spacing")
        config['hspace'] = st.slider("Vertical Spacing (hspace)", 0.1, 1.0, 0.4, 0.1)
        config['wspace'] = st.slider("Horizontal Spacing (wspace)", 0.1, 1.0, 0.4, 0.1)
    
    # Legend customization
    with st.expander("📖 Legend Customization", expanded=False):
        st.info("Customize legend labels")
        
        custom_legend_labels = {}
        col1, col2 = st.columns(2)
        with col1:
            custom_legend_labels['final_attention'] = st.text_input("Final Attention Label", 
                                                                   value="Final Attention")
            custom_legend_labels['spatial_kernel'] = st.text_input("Spatial Kernel Label",
                                                                  value="Spatial Kernel")
            custom_legend_labels['target'] = st.text_input("Target Label", value="Target")
        
        with col2:
            custom_legend_labels['perfect_correlation'] = st.text_input("Correlation Line Label",
                                                                       value="Perfect Correlation")
            custom_legend_labels['habit_plane'] = st.text_input("Habit Plane Label",
                                                               value="Habit Plane (54.7°)")
            custom_legend_labels['defect_mask'] = st.text_input("Defect Mask Label",
                                                               value="Defect Mask")
        
        st.markdown("---")
        st.info("Statistics legend labels")
        
        stat_labels = {}
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            stat_labels['max'] = st.text_input("Max Label", value="Max")
        with col_stat2:
            stat_labels['mean'] = st.text_input("Mean Label", value="Mean")
        with col_stat3:
            stat_labels['std'] = st.text_input("Std Label", value="Std")
        
        custom_legend_labels['statistics'] = stat_labels
        config['custom_legend_labels'] = custom_legend_labels
    
    # Add reset button
    col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
    with col_reset2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            config = {}
            st.success("Configuration reset to defaults")
            st.rerun()
    
    return config

# =============================================
# RESULTS MANAGER FOR EXPORT
# =============================================
class ResultsManager:
    """Manager for exporting interpolation results"""
    
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        """Prepare data for export"""
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
        
        # Convert numpy arrays to lists for JSON serialization
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
        
        return export_data
    
    def add_diffusion_to_export(self, interpolation_result, export_data):
        """
        Add diffusion data to export results
        """
        # Add diffusion data to export
        if 'diffusion_ratio' in interpolation_result['fields']:
            export_data['result']['diffusion_statistics'] = interpolation_result.get(
                'diffusion_statistics', {}
            )
            
            # Add diffusion field data
            for field_name in ['diffusion_ratio', 'diffusion_effective', 'vacancy_ratio', 'diffusion_gradient']:
                if field_name in interpolation_result['fields']:
                    export_data['result'][f'{field_name}_data'] = interpolation_result['fields'][field_name].tolist()
        
        return export_data
    
    def export_to_json(self, export_data, filename=None):
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = export_data['result']['target_angle']
            defect = export_data['result']['target_params']['defect_type']
            filename = f"bracketing_interpolation_theta_{theta}_{defect}_{timestamp}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        """Export flattened field data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename = f"stress_fields_theta_{theta}_{defect}_{timestamp}.csv"
        
        # Create DataFrame with flattened data
        data_dict = {}
        for field_name, field_data in interpolation_result['fields'].items():
            data_dict[field_name] = field_data.flatten()
        
        df = pd.DataFrame(data_dict)
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return str(obj)

# =============================================
# MAIN APPLICATION WITH COMPLETE IMPLEMENTATION
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Angular Bracketing Theory with Transformer Attention",
        layout="wide",
        page_icon="🎯",
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
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
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
    .param-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #e9ecef;
    }
    .param-key {
        font-weight: 700;
        color: #1E3A8A;
        font-size: 1.1rem;
    }
    .param-value {
        font-weight: 600;
        color: #059669;
        font-size: 1.1rem;
    }
    .physics-note {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1rem;
    }
    .diffusion-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Angular Bracketing Theory with Transformer Attention</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Physics-Aware Interpolation: Angular Orientation & Defect Type as Primary Drivers.</strong><br>
    • <strong>Angular Bracketing Kernel:</strong> Gaussian spatial locality enforcing linear interpolation between nearest angles.<br>
    • <strong>Hard Defect Gating:</strong> Sources with different defect types receive effectively zero attention.<br>
    • <strong>Theory-Informed Attention:</strong> Attention = Softmax(Learned Similarity × Spatial Kernel × Defect Mask).<br>
    • <strong>Diffusion Enhancement:</strong> D/D₀ = exp(Ωσ_h/(k_B T)) - Peak for tensile stress, valley for compressive stress.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        # Initialize with default sigma (angular window size)
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator(
            spatial_sigma=10.0, # Degrees: +/- 10 deg window has high weight
            locality_weight_factor=0.5 # 50% Learned, 50% Theory
        )
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'selected_ground_truth' not in st.session_state:
        st.session_state.selected_ground_truth = None
    if 'diffusion_physics' not in st.session_state:
        st.session_state.diffusion_physics = DiffusionPhysics()
    if 'custom_dashboard_fig' not in st.session_state:
        st.session_state.custom_dashboard_fig = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
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
                st.session_state.selected_ground_truth = None
                st.session_state.custom_dashboard_fig = None
                st.success("Cache cleared")
        
        st.divider()
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Target angle input
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
        
        # Quick angle presets
        st.markdown("**Quick Presets:**")
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        with col_preset1:
            if st.button("0°", use_container_width=True):
                custom_theta = 0.0
                st.rerun()
        with col_preset2:
            if st.button("45°", use_container_width=True):
                custom_theta = 45.0
                st.rerun()
        with col_preset3:
            if st.button("90°", use_container_width=True):
                custom_theta = 90.0
                st.rerun()
        
        # Defect type
        defect_type = st.selectbox(
            "Defect Type",
            options=['ISF', 'ESF', 'Twin', 'No Defect'],
            index=2,
            help="Type of crystal defect to simulate"
        )
        
        # Shape selection
        shape = st.selectbox(
            "Shape",
            options=['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle'],
            index=0,
            help="Geometry of defect region"
        )
        
        # Kappa parameter
        kappa = st.slider(
            "Kappa (material property)",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.01,
            help="Material stiffness parameter"
        )
        
        # Eigenstrain auto-calculation - CORRECTED VALUES
        st.markdown("#### 🧮 Eigenstrain Calculation")
        
        # Display theoretical basis
        if defect_type in PhysicsParameters.THEORETICAL_BASIS:
            theory_info = PhysicsParameters.get_theoretical_info(defect_type)
            with st.expander("📚 Theoretical Basis"):
                st.markdown(f"""
                **Formula:** {theory_info['formula']}
                
                **Description:** {theory_info['description']}
                
                **Reference:** {theory_info['reference']}
                """)
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            auto_eigen = st.checkbox("Auto-calculate eigenstrain", value=True)
        with col_e2:
            if auto_eigen:
                # Auto-calculate based on defect type with CORRECT VALUES
                eigen_strain = PhysicsParameters.get_eigenstrain(defect_type)
                st.metric("Eigenstrain ε₀", f"{eigen_strain:.3f}")
                
                # Show correction note
                st.markdown(f"""
                <div class="physics-note">
                <strong>Physics Correction Applied:</strong><br>
                Using correct eigenstrain for {defect_type} = {eigen_strain:.3f}
                </div>
                """, unsafe_allow_html=True)
            else:
                eigen_strain = st.slider(
                    "Eigenstrain ε₀",
                    min_value=0.0,
                    max_value=3.0,
                    value=2.12 if defect_type == 'Twin' else 0.707,
                    step=0.001
                )
        
        st.divider()
        
        # Diffusion parameters
        st.markdown('<h2 class="section-header">🌡️ Diffusion Physics</h2>', unsafe_allow_html=True)
        
        # Material selection for diffusion
        diffusion_material = st.selectbox(
            "Material",
            options=['Silver', 'Copper', 'Aluminum', 'Nickel', 'Iron'],
            index=0,
            help="Material for diffusion calculations"
        )
        
        # Temperature
        diffusion_T = st.slider(
            "Temperature (K)",
            min_value=300,
            max_value=1500,
            value=650,
            step=10,
            help="Temperature for diffusion calculations (650K for sintering)"
        )
        
        # Diffusion model
        diffusion_model = st.selectbox(
            "Diffusion Model",
            options=['physics_corrected', 'temperature_reduction', 
                    'activation_energy', 'vacancy_concentration'],
            index=0,
            help="Model for calculating diffusion enhancement"
        )
        
        # Show material properties
        with st.expander("📊 Material Properties"):
            props = DiffusionPhysics.get_material_properties(diffusion_material)
            col_prop1, col_prop2 = st.columns(2)
            with col_prop1:
                st.metric("Atomic Volume", f"{props['atomic_volume']:.2e} m³")
                st.metric("Activation Energy", f"{props['activation_energy']} eV")
                st.metric("Bulk Modulus", f"{props['bulk_modulus']/1e9:.1f} GPa")
            with col_prop2:
                st.metric("Melting Point", f"{props['melting_point']} K")
                st.metric("Prefactor", f"{props['prefactor']:.2e} m²/s")
                st.metric("Shear Modulus", f"{props['shear_modulus']/1e9:.1f} GPa")
        
        st.divider()
        
        # Transformer parameters
        st.markdown('<h2 class="section-header">🧠 Angular Bracketing Theory & Attention</h2>', unsafe_allow_html=True)
        
        # Spatial Locality Kernel Sigma
        st.markdown("#### 📐 Spatial Locality (Angular Kernel)")
        st.info("""
        **Angular Bracketing Kernel:**
        - A Gaussian kernel centered at the target angle.
        - Controls the 'window' of sources considered valid for interpolation.
        - High Sigma = Wide window (includes far angles).
        - Low Sigma = Narrow window (strict bracketing).
        """)
        
        spatial_sigma = st.slider(
            "Angular Kernel Sigma (degrees)",
            min_value=1.0,
            max_value=45.0,
            value=10.0,
            step=0.5,
            help="Width of the angular bracketing window (standard deviation of Gaussian)"
        )
        
        # Theory vs Learned balance
        st.markdown("#### ⚖️ Attention Balance")
        locality_weight_factor = st.slider(
            "Theory (Bracketing) vs. Learned (Transformer)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="1.0 = Pure Theory (Hard Bracketing), 0.0 = Pure Transformer (Learned)"
        )
        
        # Attention temperature
        temperature = st.slider(
            "Attention Temperature",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Softmax temperature (lower = sharper attention on bracketing sources)"
        )
        
        # Visualize Kernel
        if st.button("📊 Visualize Angular Kernel", use_container_width=True):
            fig_kernel = st.session_state.transformer_interpolator.visualize_angular_kernel(
                target_angle_deg=custom_theta
            )
            st.pyplot(fig_kernel)
        
        # Update parameters
        if st.button("🔄 Update Interpolator Parameters", use_container_width=True):
            st.session_state.transformer_interpolator.set_spatial_parameters(
                spatial_sigma=spatial_sigma,
                locality_weight_factor=locality_weight_factor
            )
            st.session_state.transformer_interpolator.temperature = temperature
            st.success("Parameters updated!")
        
        st.divider()
        
        # Publication quality settings
        st.markdown("#### 📐 Publication Settings")
        
        dpi_setting = st.selectbox(
            "Figure DPI",
            options=[150, 300, 600, 1200],
            index=1,
            help="Higher DPI for publication quality (300-600 recommended)"
        )
        
        export_format = st.selectbox(
            "Export Format",
            options=['PNG', 'PDF', 'SVG'],
            index=0,
            help="Format for saving publication figures"
        )
        
        st.divider()
        
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
                        'eps0': eigen_strain,
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
                        st.success(f"Interpolation successful! Theory-Informed Attention applied.")
                        st.session_state.selected_ground_truth = None
                        st.session_state.custom_dashboard_fig = None
                    else:
                        st.error("Interpolation failed. Check console for errors.")
    
    # Main content area
    if st.session_state.solutions:
        st.markdown(f"### 📊 Loaded {len(st.session_state.solutions)} Solutions")
        
        # Display loaded solutions
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Loaded Files", len(st.session_state.solutions))
        with col_info2:
            if st.session_state.interpolation_result:
                st.metric("Interpolated Angle", f"{st.session_state.interpolation_result['target_angle']:.1f}°")
        with col_info3:
            if st.session_state.interpolation_result:
                st.metric("Grid Size", f"{st.session_state.interpolation_result['shape'][0]}×{st.session_state.interpolation_result['shape'][1]}")
        
        # Display source information
        if st.session_state.solutions:
            source_thetas = []
            source_defects = []
            for sol in st.session_state.solutions:
                if 'params' in sol and 'theta' in sol['params']:
                    theta_deg = np.degrees(sol['params']['theta']) % 360  # Normalize to [0, 360)
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
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Results Overview",
            "🎨 Stress Visualization",
            "🌡️ Diffusion Visualization",
            "⚖️ Attention Analysis",
            "🔄 Comparison Dashboard",
            "📊 Publication Figures",
            "💾 Export Results"
        ])
        
        with tab1:
            # Results overview
            st.markdown('<h2 class="section-header">📊 Interpolation Results</h2>', unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Max Von Mises",
                    f"{result['statistics']['von_mises']['max']:.3f} GPa",
                    delta=f"±{result['statistics']['von_mises']['std']:.3f}"
                )
            with col2:
                st.metric(
                    "Hydrostatic Range",
                    f"{result['statistics']['sigma_hydro']['max_tension']:.3f}/{result['statistics']['sigma_hydro']['max_compression']:.3f} GPa"
                )
            with col3:
                st.metric(
                    "Mean Stress Magnitude",
                    f"{result['statistics']['sigma_mag']['mean']:.3f} GPa"
                )
            with col4:
                st.metric(
                    "Entropy (Attention)",
                    f"{result['weights']['entropy']:.3f}",
                    delta="Lower is more focused"
                )
            
            # Diffusion metrics if available
            if 'diffusion' in result['statistics'] and result['statistics']['diffusion']:
                st.markdown('<h3 class="section-header">🌡️ Diffusion Statistics</h3>', unsafe_allow_html=True)
                diff_stats = result['statistics']['diffusion']
                
                col_diff1, col_diff2, col_diff3, col_diff4 = st.columns(4)
                with col_diff1:
                    st.metric(
                        "Max Enhancement",
                        f"{diff_stats['max_enhancement']:.2f}x",
                        delta="Peak diffusion"
                    )
                with col_diff2:
                    st.metric(
                        "Min (Suppression)",
                        f"{diff_stats['min_enhancement']:.2f}x",
                        delta="Valley diffusion"
                    )
                with col_diff3:
                    st.metric(
                        "Mean Enhancement",
                        f"{diff_stats['mean_enhancement']:.2f}x"
                    )
                with col_diff4:
                    st.metric(
                        "Enhanced Area",
                        f"{diff_stats['enhanced_area_fraction']*100:.1f}%"
                    )
            
            # Physics parameters display
            st.markdown("#### 🧮 Physics Parameters")
            col_phys1, col_phys2, col_phys3 = st.columns(3)
            with col_phys1:
                st.metric("Target Angle", f"{result['target_angle']:.1f}°")
            with col_phys2:
                st.metric("Defect Type", result['target_params']['defect_type'])
            with col_phys3:
                st.metric("Eigenstrain ε₀", f"{result['target_params']['eps0']:.3f}")
            
            # Quick preview
            st.markdown("#### 👀 Quick Preview")
            preview_component = st.selectbox(
                "Preview Component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag', 'diffusion_ratio'],
                index=0,
                key="preview_component"
            )
            
            if preview_component in result['fields']:
                if preview_component == 'diffusion_ratio':
                    # Special handling for diffusion visualization
                    fig_preview, D_ratio = st.session_state.heatmap_visualizer.create_diffusion_heatmap(
                        result['fields']['sigma_hydro'],
                        title="Diffusion Enhancement Preview",
                        T_K=diffusion_T,
                        material=diffusion_material,
                        cmap_name='RdBu_r',
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(10, 8),
                        dpi=150,
                        model=diffusion_model
                    )
                else:
                    fig_preview = st.session_state.heatmap_visualizer.create_stress_heatmap(
                        result['fields'][preview_component],
                        title=f"{preview_component.replace('_', ' ').title()} Stress",
                        cmap_name='viridis',
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(10, 8),
                        dpi=150
                    )
                st.pyplot(fig_preview)
        
        with tab2:
            # Stress Visualization tab
            st.markdown('<h2 class="section-header">🎨 Stress Field Visualization</h2>', unsafe_allow_html=True)
            
            # Visualization controls
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            with col_viz1:
                stress_component = st.selectbox(
                    "Stress Component",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0,
                    key="stress_component"
                )
            with col_viz2:
                cmap_category = st.selectbox(
                    "Colormap Category",
                    options=list(COLORMAP_OPTIONS.keys()),
                    index=0,
                    key="cmap_category"
                )
                cmap_options = COLORMAP_OPTIONS[cmap_category]
            with col_viz3:
                cmap_name = st.selectbox(
                    "Colormap",
                    options=cmap_options,
                    index=0,
                    key="cmap_name"
                )
            
            # Show colormap preview
            if st.checkbox("Preview Colormap"):
                fig_cmap = st.session_state.heatmap_visualizer.get_colormap_preview(cmap_name)
                st.pyplot(fig_cmap)
            
            # Visualization type selection
            stress_viz_type = st.radio(
                "Visualization Type",
                options=["2D Heatmap", "3D Surface", "Interactive Heatmap", "Interactive 3D", "Angular Orientation"],
                horizontal=True
            )
            
            if stress_component in result['fields']:
                stress_field = result['fields'][stress_component]
                
                if stress_viz_type == "2D Heatmap":
                    # 2D heatmap
                    fig_2d = st.session_state.heatmap_visualizer.create_stress_heatmap(
                        stress_field,
                        title=f"{stress_component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(12, 10),
                        dpi=300
                    )
                    st.pyplot(fig_2d)
                
                elif stress_viz_type == "3D Surface":
                    # 3D surface plot
                    fig_3d = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                        stress_field,
                        title=f"{stress_component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(14, 10)
                    )
                    st.pyplot(fig_3d)
                
                elif stress_viz_type == "Interactive Heatmap":
                    # Interactive heatmap
                    fig_interactive = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                        stress_field,
                        title=f"{stress_component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        width=800,
                        height=700
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                
                elif stress_viz_type == "Interactive 3D":
                    # Interactive 3D surface
                    fig_3d_interactive = st.session_state.heatmap_visualizer.create_interactive_3d_surface(
                        stress_field,
                        title=f"{stress_component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        width=900,
                        height=700
                    )
                    st.plotly_chart(fig_3d_interactive, use_container_width=True)
                
                elif stress_viz_type == "Angular Orientation":
                    # Angular orientation plot
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(10, 10),
                        show_habit_plane=True
                    )
                    st.pyplot(fig_angular)
            
            # Comparison of all components
            st.markdown("#### 🔄 Stress Component Comparison")
            if st.button("Show All Stress Components Comparison", key="show_all_stress_components"):
                fig_all = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                    {k: v for k, v in result['fields'].items() if k in ['von_mises', 'sigma_hydro', 'sigma_mag']},
                    cmap_name=cmap_name,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(18, 6)
                )
                st.pyplot(fig_all)
        
        with tab3:
            # DIFFUSION VISUALIZATION TAB
            st.markdown('<h2 class="section-header">🌡️ Diffusion Enhancement Visualization</h2>', unsafe_allow_html=True)
            
            # Theoretical explanation
            st.markdown("""
            <div class="diffusion-box">
            <strong>📚 Diffusion Enhancement Theory:</strong><br>
            The diffusion coefficient under stress follows: $D(σ_h)/D_0 = \\exp(Ωσ_h/(k_B T))$<br>
            Where Ω is atomic volume, σ_h is hydrostatic stress, k_B is Boltzmann constant, T is temperature.<br>
            • <strong>Tensile stress (σ_h > 0)</strong> → Enhanced diffusion (D/D₀ > 1) - PEAKS<br>
            • <strong>Compressive stress (σ_h < 0)</strong> → Suppressed diffusion (D/D₀ < 1) - VALLEYS<br>
            • <strong>Habit plane (θ ≈ 54.7°)</strong> → Maximum tensile stress → Peak diffusion enhancement
            </div>
            """, unsafe_allow_html=True)
            
            # Diffusion visualization controls
            col_diff1, col_diff2, col_diff3 = st.columns(3)
            with col_diff1:
                diffusion_visualization_type = st.selectbox(
                    "Visualization Type",
                    options=['Diffusion Enhancement (D/D₀)', 'Vacancy Concentration', 
                            'Effective Diffusion Coefficient', 'Gradient Analysis',
                            'Phase Diagram', 'Correlation Analysis'],
                    index=0,
                    key="diffusion_viz_type"
                )
            
            with col_diff2:
                diffusion_colormap = st.selectbox(
                    "Colormap",
                    options=ALL_COLORMAPS,
                    index=ALL_COLORMAPS.index('RdBu_r') if 'RdBu_r' in ALL_COLORMAPS else 0,
                    key="diffusion_colormap"
                )
            
            with col_diff3:
                diffusion_log_scale = st.checkbox(
                    "Log Scale",
                    value=True,
                    help="Use log scale for better visualization of enhancement/suppression"
                )
            
            # Check if sigma_hydro is available
            if 'sigma_hydro' in result['fields']:
                sigma_hydro_field = result['fields']['sigma_hydro']
                
                if diffusion_visualization_type == 'Diffusion Enhancement (D/D₀)':
                    # 2D heatmap of diffusion enhancement
                    st.markdown("#### 📈 Diffusion Enhancement Map (D/D₀)")
                    
                    fig_diff, D_ratio = st.session_state.heatmap_visualizer.create_diffusion_heatmap(
                        sigma_hydro_field,
                        title="Diffusion Enhancement Map",
                        T_K=diffusion_T,
                        material=diffusion_material,
                        cmap_name=diffusion_colormap,
                        figsize=(12, 10),
                        log_scale=diffusion_log_scale,
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
                
                elif diffusion_visualization_type == 'Vacancy Concentration':
                    # Vacancy concentration map
                    st.markdown("#### 🧬 Vacancy Concentration Enhancement (C_v/C_v₀)")
                    
                    fig_vacancy, C_ratio = st.session_state.heatmap_visualizer.create_vacancy_concentration_map(
                        sigma_hydro_field,
                        T_K=diffusion_T,
                        material=diffusion_material,
                        figsize=(12, 10)
                    )
                    st.pyplot(fig_vacancy)
                
                elif diffusion_visualization_type == 'Effective Diffusion Coefficient':
                    # Effective diffusion coefficient
                    st.markdown("#### 🚀 Effective Diffusion Coefficient (m²/s)")
                    
                    D_eff = DiffusionPhysics.compute_effective_diffusion_coefficient(
                        sigma_hydro_field, diffusion_T, diffusion_material
                    )
                    
                    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
                    im = ax.imshow(D_eff, cmap='viridis', aspect='equal',
                                  interpolation='bilinear', origin='lower',
                                  norm=LogNorm())
                    plt.colorbar(im, ax=ax, label='Effective D (m²/s)')
                    
                    ax.set_title(f'Effective Diffusion Coefficient\n{material}, T = {diffusion_T} K',
                                fontsize=20, fontweight='bold')
                    ax.set_xlabel('X Position (nm)')
                    ax.set_ylabel('Y Position (nm)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics
                    st.metric("Maximum D_eff", f"{np.max(D_eff):.2e} m²/s")
                    st.metric("Minimum D_eff", f"{np.min(D_eff):.2e} m²/s")
                    st.metric("Mean D_eff", f"{np.mean(D_eff):.2e} m²/s")
                
                elif diffusion_visualization_type == 'Gradient Analysis':
                    # Gradient analysis
                    st.markdown("#### 📊 Diffusion Gradient Analysis")
                    
                    D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                        sigma_hydro_field, diffusion_T, diffusion_material, diffusion_model
                    )
                    
                    fig_grad, grad_info = st.session_state.heatmap_visualizer.create_diffusion_gradient_map(
                        D_ratio, figsize=(12, 10)
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
                
                elif diffusion_visualization_type == 'Phase Diagram':
                    # Phase diagram
                    st.markdown("#### 🔬 Diffusion Phase Diagram")
                    
                    fig_phase, _ = st.session_state.heatmap_visualizer.create_phase_diagram_visualization(
                        sigma_range=(-5, 5),
                        T_range=(300, 1200),
                        material=diffusion_material,
                        figsize=(14, 10)
                    )
                    st.pyplot(fig_phase)
                    
                    # Add marker for current conditions
                    mean_stress = np.mean(sigma_hydro_field)
                    st.info(f"Current conditions marked on phase diagram: σ = {mean_stress:.2f} GPa, T = {diffusion_T} K")
                
                elif diffusion_visualization_type == 'Correlation Analysis':
                    # Correlation analysis
                    st.markdown("#### 📊 Stress-Diffusion Correlation")
                    
                    D_ratio = DiffusionPhysics.compute_diffusion_enhancement(
                        sigma_hydro_field, diffusion_T, diffusion_material, diffusion_model
                    )
                    
                    fig_corr = st.session_state.heatmap_visualizer.create_stress_diffusion_correlation(
                        sigma_hydro_field, D_ratio, figsize=(12, 10)
                    )
                    st.pyplot(fig_corr)
                
                # Advanced visualization options
                st.markdown("---")
                st.markdown("#### 🔬 Advanced 3D Visualizations")
                
                col_3d1, col_3d2 = st.columns(2)
                
                with col_3d1:
                    if st.button("3D Diffusion Surface", use_container_width=True):
                        fig_3d, _ = st.session_state.heatmap_visualizer.create_diffusion_3d_surface(
                            sigma_hydro_field,
                            title="3D Diffusion Landscape",
                            T_K=diffusion_T,
                            material=diffusion_material,
                            cmap_name=diffusion_colormap,
                            log_scale=diffusion_log_scale,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type'],
                            model=diffusion_model
                        )
                        st.pyplot(fig_3d)
                
                with col_3d2:
                    if st.button("Interactive 3D Diffusion", use_container_width=True):
                        fig_interactive, _ = st.session_state.heatmap_visualizer.create_interactive_diffusion_3d(
                            sigma_hydro_field,
                            title="Interactive Diffusion 3D",
                            T_K=diffusion_T,
                            material=diffusion_material,
                            cmap_name=diffusion_colormap,
                            log_scale=diffusion_log_scale,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type'],
                            model=diffusion_model
                        )
                        st.plotly_chart(fig_interactive, use_container_width=True)
                
                # Material comparison
                st.markdown("---")
                st.markdown("#### 🔄 Compare Different Materials")
                
                if st.button("Compare Materials", use_container_width=True):
                    materials = ['Silver', 'Copper', 'Aluminum']
                    fields = [sigma_hydro_field] * len(materials)
                    titles = [f"{mat} at {diffusion_T}K" for mat in materials]
                    
                    fig_compare, D_ratios, stats = st.session_state.heatmap_visualizer.create_diffusion_comparison_dashboard(
                        fields, titles, T_K=diffusion_T, material=diffusion_material,
                        cmap_name=diffusion_colormap, figsize=(20, 15), log_scale=diffusion_log_scale
                    )
                    st.pyplot(fig_compare)
                
                # Theoretical background expander
                with st.expander("📚 Detailed Theoretical Background"):
                    props = DiffusionPhysics.get_material_properties(diffusion_material)
                    Omega = props['atomic_volume']
                    Q = props['activation_energy']
                    
                    st.markdown(f"""
                    ### Complete Diffusion Enhancement Theory
                    
                    #### 1. Basic Equation:
                    $$
                    \\frac{{D(\\sigma_h)}}{{D_0}} = \\exp\\left(\\frac{{\\Omega \\sigma_h}}{{k_B T}}\\right)
                    $$
                    
                    Where:
                    - $\\Omega$ = Atomic volume = **{Omega:.2e} m³** for {diffusion_material}
                    - $\\sigma_h$ = Hydrostatic stress in Pa (positive = tensile, negative = compressive)
                    - $k_B$ = Boltzmann constant = $1.38 \\times 10^{{-23}}$ J/K
                    - $T$ = Temperature = **{diffusion_T} K**
                    
                    #### 2. Practical Example:
                    For $\\sigma_h = 1$ GPa (tension):
                    $$
                    \\frac{{D}}{{D_0}} = \\exp\\left(\\frac{{{Omega:.2e} \\times 1 \\times 10^9}}{{1.38 \\times 10^{{-23}} \\times {diffusion_T}}}\\right) = \\exp({Omega*1e9/(1.38e-23*diffusion_T):.2f}) \\approx {np.exp(Omega*1e9/(1.38e-23*diffusion_T)):.2f}
                    $$
                    
                    #### 3. Physical Interpretation:
                    - **Tensile stress** ($\\sigma_h > 0$): Increases vacancy concentration → Enhanced diffusion (PEAK)
                    - **Compressive stress** ($\\sigma_h < 0$): Decreases vacancy concentration → Suppressed diffusion (VALLEY)
                    - **Zero stress**: $D/D_0 = 1$ (reference)
                    
                    #### 4. Material Properties:
                    - Activation Energy (Q) = {Q} eV
                    - Diffusion Prefactor = {props['prefactor']:.2e} m²/s
                    - Bulk Modulus = {props['bulk_modulus']/1e9:.1f} GPa
                    - Shear Modulus = {props['shear_modulus']/1e9:.1f} GPa
                    
                    #### 5. Applications:
                    1. **Sintering**: Enhanced diffusion at particle contacts accelerates densification
                    2. **Creep**: Stress-enhanced diffusion controls deformation rates
                    3. **Phase transformations**: Diffusion-controlled transformations accelerate under stress
                    4. **Grain growth**: Stress gradients drive diffusion fluxes
                    """)
            else:
                st.warning("Hydrostatic stress field not available for diffusion calculations.")
        
        with tab4:
            # Attention Analysis tab
            st.markdown('<h2 class="section-header">⚖️ Theory-Informed Attention Analysis</h2>', unsafe_allow_html=True)
            
            if 'weights' in result:
                weights = result['weights']
                
                # 1. Visualization of Defect Mask
                st.markdown("#### 🚧 Defect Type Gating (Hard Constraint)")
                fig_mask, ax_mask = plt.subplots(figsize=(12, 4), dpi=150)
                x = range(len(weights['defect_mask']))
                ax_mask.bar(x, weights['defect_mask'], color='purple', alpha=0.7, label='Defect Mask Value')
                ax_mask.set_xlabel('Source Index')
                ax_mask.set_ylabel('Gating Weight (1.0 = Same Type, 0.0 = Different)')
                ax_mask.set_title('Defect Type Hard Filter', fontsize=16, fontweight='bold')
                ax_mask.set_ylim([0, 1.1])
                ax_mask.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig_mask)
                
                # 2. Visualization of Angular Bracketing Kernel
                st.markdown("#### 📐 Angular Bracketing Kernel (Spatial Locality)")
                fig_kernel, ax_kernel = plt.subplots(figsize=(12, 4), dpi=150)
                ax_kernel.plot(x, weights['spatial_kernel'], 'g--', linewidth=2, marker='o', label='Gaussian Kernel')
                ax_kernel.fill_between(x, weights['spatial_kernel'], alpha=0.1, color='green')
                ax_kernel.set_xlabel('Source Index (sorted by angle ideally)')
                ax_kernel.set_ylabel('Spatial Weight')
                ax_kernel.set_title(f'Angular Locality (Sigma={st.session_state.transformer_interpolator.spatial_sigma}°)', fontsize=16, fontweight='bold')
                ax_kernel.grid(True, alpha=0.3)
                ax_kernel.legend()
                st.pyplot(fig_kernel)
                
                # 3. Final Attention Weights
                st.markdown("#### 🎯 Final Attention Weights (Product)")
                fig_attn, ax_attn = plt.subplots(figsize=(12, 6), dpi=150)
                width = 0.25
                ax_attn.bar([i - width for i in x], weights['spatial_kernel'], width,
                             label='Spatial Kernel', alpha=0.5, color='green')
                ax_attn.bar(x, weights['combined'], width,
                             label='Final Attention', alpha=0.8, color='blue')
                ax_attn.set_xlabel('Source Index')
                ax_attn.set_ylabel('Weight')
                ax_attn.set_title('Combined Attention Distribution', fontsize=16, fontweight='bold')
                ax_attn.legend()
                ax_attn.grid(True, alpha=0.3)
                st.pyplot(fig_attn)
                
                # 4. Top Contributors Analysis
                st.markdown("#### 🏆 Top Contributors (Source vs Target)")
                contrib_data = []
                for i in range(len(weights['combined'])):
                    angle_dist = result['source_distances'][i] if i < len(result['source_distances']) else 0.0
                    is_same_defect = weights['defect_mask'][i] > 0.5
                    contrib_data.append({
                        'Source': i,
                        'Theta (°)': result['source_theta_degrees'][i] if i < len(result['source_theta_degrees']) else 0.0,
                        'Angular Dist (°)': angle_dist,
                        'Defect Match': 'Yes' if is_same_defect else 'No',
                        'Spatial Weight': f"{weights['spatial_kernel'][i]:.4f}",
                        'Final Attention': f"{weights['combined'][i]:.4f}"
                    })
                
                df_contrib = pd.DataFrame(contrib_data)
                df_contrib = df_contrib.sort_values('Final Attention', ascending=False)
                st.dataframe(df_contrib.head(10).style.background_gradient(subset=['Final Attention'], cmap='Blues'))
        
        with tab5:
            # ENHANCED COMPARISON DASHBOARD WITH ALL REQUESTED FEATURES
            st.markdown('<h2 class="section-header">🔄 Enhanced Comparison Dashboard</h2>', unsafe_allow_html=True)
            
            # Create configuration interface
            st.markdown("#### ⚙️ Dashboard Customization")
            dashboard_config = create_dashboard_configuration_interface()
            
            # Defect type filtering
            col_filter1, col_filter2 = st.columns([1, 3])
            with col_filter1:
                unique_defects = set()
                for sol in st.session_state.solutions:
                    if 'params' in sol and 'defect_type' in sol['params']:
                        unique_defects.add(sol['params']['defect_type'])
                
                defect_options = ['All'] + sorted(list(unique_defects))
                
                selected_defect_filter = st.selectbox(
                    "Filter by Defect Type",
                    options=defect_options,
                    index=0,
                    help="Select a specific defect type (e.g., Twin, ISF) to filter the comparison sources."
                )
            
            actual_defect_filter = None if selected_defect_filter == 'All' else selected_defect_filter
            
            # Enhanced Ground truth selection with defect type display
            st.markdown("#### 🎯 Select Ground Truth Source")
            if 'source_theta_degrees' in result and result['source_theta_degrees']:
                ground_truth_options = []
                for i, theta in enumerate(result['source_theta_degrees']):
                    d_type = result['source_fields'][i].get('source_params', {}).get('defect_type', 'Unknown')
                    distance = result['source_distances'][i]
                    weight = result['weights']['combined'][i]
                    
                    tag = "✅" if actual_defect_filter is None or d_type == actual_defect_filter else "❌"
                    
                    ground_truth_options.append(
                        f"{tag} Source {i}: {d_type} (θ={theta:.1f}°, Δ={distance:.1f}°, w={weight:.3f})"
                    )
                
                selected_option = st.selectbox(
                    "Choose ground truth source:",
                    options=ground_truth_options,
                    index=0,
                    key="ground_truth_select"
                )
                
                # Regex parsing to extract Source index
                match = re.search(r'Source (\d+):', selected_option)
                if match:
                    selected_index = int(match.group(1))
                    st.session_state.selected_ground_truth = selected_index
            
            # Visualization options with enhanced colormap selection
            st.markdown("#### 🎨 Comparison Visualization")
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                comp_component = st.selectbox(
                    "Component for Comparison",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag', 'diffusion_ratio'],
                    index=0,
                    key="comp_component_dash"
                )
            
            with col_viz2:
                comp_cmap = st.selectbox(
                    "Default Colormap",
                    options=ALL_COLORMAPS,
                    index=ALL_COLORMAPS.index('viridis') if 'viridis' in ALL_COLORMAPS else 0,
                    key="comp_cmap_dash"
                )
            
            # Generate dashboard button
            if st.button("🔄 Generate Enhanced Dashboard", type="primary", use_container_width=True):
                if comp_component in result['fields']:
                    source_info = {
                        'theta_degrees': result['source_theta_degrees'],
                        'distances': result['source_distances'],
                        'weights': result['weights'],
                        'source_fields': result.get('source_fields', []),
                        'statistics': result.get('statistics', {})
                    }
                    
                    source_fields_list = result.get('source_fields', [])
                    
                    # Store the generated figure in session state
                    if comp_component == 'diffusion_ratio':
                        st.warning("Note: Diffusion comparison uses interpolated hydrostatic stress to compute diffusion")
                        if 'sigma_hydro' in result['fields']:
                            interpolated_diffusion = DiffusionPhysics.compute_diffusion_enhancement(
                                result['fields']['sigma_hydro'], diffusion_T, diffusion_material, diffusion_model
                            )
                            ground_truth_diffusion = None
                            if st.session_state.selected_ground_truth is not None:
                                idx = st.session_state.selected_ground_truth
                                gt_hydro = source_fields_list[idx]['sigma_hydro']
                                ground_truth_diffusion = DiffusionPhysics.compute_diffusion_enhancement(
                                    gt_hydro, diffusion_T, diffusion_material, diffusion_model
                                )
                            
                            fig_comparison = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                                interpolated_fields={'diffusion_ratio': interpolated_diffusion},
                                source_fields=[{'diffusion_ratio': ground_truth_diffusion}] if ground_truth_diffusion is not None else [],
                                source_info=source_info,
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                component='diffusion_ratio',
                                cmap_name=comp_cmap,
                                figsize=(24, 18),
                                ground_truth_index=0 if ground_truth_diffusion is not None else None,
                                defect_type_filter=actual_defect_filter,
                                config=dashboard_config
                            )
                            if fig_comparison:
                                st.session_state.custom_dashboard_fig = fig_comparison
                    else:
                        fig_comparison = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                            interpolated_fields=result['fields'],
                            source_fields=source_fields_list,
                            source_info=source_info,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type'],
                            component=comp_component,
                            cmap_name=comp_cmap,
                            figsize=(24, 18),
                            ground_truth_index=st.session_state.selected_ground_truth,
                            defect_type_filter=actual_defect_filter,
                            config=dashboard_config
                        )
                        if fig_comparison:
                            st.session_state.custom_dashboard_fig = fig_comparison
            
            # Display the dashboard if it exists
            if 'custom_dashboard_fig' in st.session_state and st.session_state.custom_dashboard_fig is not None:
                st.pyplot(st.session_state.custom_dashboard_fig)
                
                # Save dashboard option
                st.markdown("#### 💾 Save Custom Dashboard")
                col_save1, col_save2 = st.columns(2)
                with col_save1:
                    save_format = st.selectbox("Format", ['PNG', 'PDF', 'SVG'], index=0, key="dashboard_save_format")
                    filename = st.text_input("Filename", value=f"custom_dashboard_{result['target_angle']:.1f}_{defect_type}")
                
                with col_save2:
                    dpi = st.selectbox("DPI", [150, 300, 600, 1200], index=1, key="dashboard_dpi")
                    
                    if st.button("💾 Save Dashboard Figure", use_container_width=True):
                        buf = BytesIO()
                        st.session_state.custom_dashboard_fig.savefig(buf, format=save_format.lower(), 
                                                                    dpi=dpi, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label=f"📥 Download {save_format}",
                            data=buf,
                            file_name=f"{filename}.{save_format.lower()}",
                            mime=f"image/{save_format.lower()}" if save_format != 'PDF' else "application/pdf",
                            use_container_width=True
                        )
                
                # Error metrics
                if st.session_state.selected_ground_truth is not None and comp_component != 'diffusion_ratio':
                    try:
                        ground_truth_field = source_fields_list[st.session_state.selected_ground_truth][comp_component]
                        interpolated_field = result['fields'][comp_component]
                        error_field = interpolated_field - ground_truth_field
                        mse = np.mean(error_field**2)
                        mae = np.mean(np.abs(error_field))
                        rmse = np.sqrt(mse)
                        from scipy.stats import pearsonr
                        try:
                            corr_coef, _ = pearsonr(ground_truth_field.flatten(), interpolated_field.flatten())
                        except:
                            corr_coef = 0.0
                        
                        st.markdown("#### 📊 Error Metrics")
                        err_col1, err_col2, err_col3, err_col4 = st.columns(4)
                        with err_col1:
                            st.metric("MSE", f"{mse:.6f}")
                        with err_col2:
                            st.metric("MAE", f"{mae:.6f}")
                        with err_col3:
                            st.metric("RMSE", f"{rmse:.6f}")
                        with err_col4:
                            st.metric("Pearson Corr", f"{corr_coef:.4f}")
                    except:
                        pass
            else:
                st.info("Configure your dashboard settings above and click 'Generate Enhanced Dashboard' to create the visualization.")
        
        with tab6:
            # Publication Figures tab
            st.markdown('<h2 class="section-header">📊 Publication Quality Figures</h2>', unsafe_allow_html=True)
            
            st.info("""
            **Publication Quality Features:**
            - High DPI (300-600) for print quality
            - Perceptually uniform colormaps
            - Proper axis labels with units
            - Scale bars for spatial reference
            - Clean, professional styling
            """)
            
            # Publication figure settings
            col_pub1, col_pub2, col_pub3 = st.columns(3)
            with col_pub1:
                pub_component = st.selectbox(
                    "Component",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag', 'diffusion_ratio'],
                    index=0,
                    key="pub_component"
                )
            with col_pub2:
                pub_cmap = st.selectbox(
                    "Colormap",
                    options=ALL_COLORMAPS,
                    index=ALL_COLORMAPS.index('viridis') if 'viridis' in ALL_COLORMAPS else 0,
                    key="pub_cmap"
                )
            with col_pub3:
                pub_dpi = st.selectbox(
                    "DPI",
                    options=[300, 600, 1200],
                    index=0,
                    key="pub_dpi"
                )
            
            # Create publication quality figure
            if pub_component in result['fields']:
                if pub_component == 'diffusion_ratio':
                    # Publication quality diffusion heatmap
                    st.markdown("#### 📈 Publication Quality Diffusion Heatmap")
                    fig_pub, _ = st.session_state.heatmap_visualizer.create_diffusion_heatmap(
                        result['fields']['sigma_hydro'],
                        title="Diffusion Enhancement",
                        T_K=diffusion_T,
                        material=diffusion_material,
                        cmap_name=pub_cmap,
                        figsize=(10, 8),
                        log_scale=True,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        model=diffusion_model
                    )
                else:
                    # Publication quality stress heatmap
                    st.markdown("#### 📈 Publication Quality Stress Heatmap")
                    fig_pub = st.session_state.heatmap_visualizer.create_publication_quality_plot(
                        result['fields'][pub_component],
                        title=f"{pub_component.replace('_', ' ').title()} Stress",
                        cmap_name=pub_cmap,
                        figsize=(10, 8),
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type']
                    )
                
                st.pyplot(fig_pub)
                
                # Save figure options
                st.markdown("#### 💾 Save Publication Figure")
                col_save1, col_save2 = st.columns(2)
                with col_save1:
                    save_format = st.selectbox(
                        "Format",
                        options=['PNG', 'PDF', 'SVG'],
                        index=0
                    )
                with col_save2:
                    filename_base = f"{pub_component}_theta_{result['target_angle']:.1f}_{result['target_params']['defect_type']}"
                    filename = st.text_input("Filename", value=filename_base)
                
                if st.button("💾 Save Figure", use_container_width=True):
                    # Create a buffer to save the figure
                    buf = BytesIO()
                    fig_pub.savefig(buf, format=save_format.lower(), dpi=pub_dpi, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download {save_format}",
                        data=buf,
                        file_name=f"{filename}.{save_format.lower()}",
                        mime=f"image/{save_format.lower()}" if save_format != 'PDF' else "application/pdf",
                        use_container_width=True
                    )
                
                # Create multi-panel figure for publication
                st.markdown("#### 🖼️ Multi-Panel Publication Figure")
                if st.button("Create Multi-Panel Figure", use_container_width=True):
                    fig_multi, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
                    axes = axes.flatten()
                    
                    if pub_component == 'diffusion_ratio':
                        # Diffusion-focused multi-panel
                        components = ['sigma_hydro', 'diffusion_ratio', 'vacancy_ratio', 'diffusion_gradient']
                        titles = ['Hydrostatic Stress', 'Diffusion Enhancement', 'Vacancy Concentration', 'Diffusion Gradient']
                    else:
                        # Stress-focused multi-panel
                        components = ['von_mises', 'sigma_hydro', 'sigma_mag', 'sigma_hydro']
                        titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude', 'Hydrostatic Stress']
                    
                    for idx, (comp, title) in enumerate(zip(components, titles)):
                        ax = axes[idx]
                        
                        if comp in result['fields']:
                            field = result['fields'][comp]
                            if comp == 'diffusion_ratio':
                                im = ax.imshow(np.log10(np.clip(field, 0.1, 10)), cmap='RdBu_r',
                                              vmin=-1, vmax=1, aspect='equal', interpolation='bilinear', origin='lower')
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('log₁₀(D/D₀)')
                            elif comp == 'vacancy_ratio':
                                im = ax.imshow(field, cmap='plasma', aspect='equal',
                                              interpolation='bilinear', origin='lower', norm=LogNorm(vmin=0.1, vmax=10))
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('C_v/C_v₀')
                            else:
                                im = ax.imshow(field, cmap=pub_cmap, aspect='equal',
                                              interpolation='bilinear', origin='lower')
                                cbar = plt.colorbar(im, ax=ax)
                                cbar.set_label('Stress (GPa)')
                            
                            ax.set_title(title, fontsize=14, fontweight='bold')
                        else:
                            ax.text(0.5, 0.5, f"Data not available\nfor {comp}",
                                   ha='center', va='center', fontsize=12)
                            ax.set_axis_off()
                        
                        ax.set_xlabel('X Position (nm)', fontsize=10)
                        ax.set_ylabel('Y Position (nm)', fontsize=10)
                        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                    
                    if pub_component == 'diffusion_ratio':
                        suptitle = f"Diffusion Analysis: θ = {result['target_angle']:.1f}°, {result['target_params']['defect_type']}, T = {diffusion_T} K"
                    else:
                        suptitle = f"Stress Analysis: θ = {result['target_angle']:.1f}°, {result['target_params']['defect_type']}"
                    
                    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
                    plt.tight_layout()
                    st.pyplot(fig_multi)
        
        with tab7:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            
            export_format = st.radio(
                "Export Format",
                options=["JSON (Full Results)", "CSV (Field Data)", "All Data (ZIP)"],
                horizontal=True
            )
            
            if export_format == "JSON (Full Results)":
                viz_params = {
                    'component': 'von_mises',
                    'colormap': 'viridis',
                    'visualization_type': '2D Heatmap',
                    'dpi': 300,
                    'diffusion_material': diffusion_material,
                    'diffusion_T': diffusion_T,
                    'diffusion_model': diffusion_model
                }
                export_data = st.session_state.results_manager.prepare_export_data(
                    result, viz_params
                )
                
                # Add diffusion data
                export_data = st.session_state.results_manager.add_diffusion_to_export(
                    result, export_data
                )
                
                json_str, json_filename = st.session_state.results_manager.export_to_json(export_data)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
            
            elif export_format == "CSV (Field Data)":
                csv_str, csv_filename = st.session_state.results_manager.export_to_csv(result)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif export_format == "All Data (ZIP)":
                # Create a ZIP file with all data
                import zipfile
                import tempfile
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Create JSON
                    viz_params = {
                        'component': 'von_mises',
                        'colormap': 'viridis',
                        'visualization_type': '2D Heatmap',
                        'dpi': 300,
                        'diffusion_material': diffusion_material,
                        'diffusion_T': diffusion_T,
                        'diffusion_model': diffusion_model
                    }
                    export_data = st.session_state.results_manager.prepare_export_data(result, viz_params)
                    export_data = st.session_state.results_manager.add_diffusion_to_export(result, export_data)
                    
                    json_str, json_filename = st.session_state.results_manager.export_to_json(export_data)
                    
                    json_path = os.path.join(tmpdir, json_filename)
                    with open(json_path, 'w') as f:
                        f.write(json_str)
                    
                    # Create CSV
                    csv_str, csv_filename = st.session_state.results_manager.export_to_csv(result)
                    csv_path = os.path.join(tmpdir, csv_filename)
                    with open(csv_path, 'w') as f:
                        f.write(csv_str)
                    
                    # Create publication figure - diffusion enhancement
                    if 'sigma_hydro' in result['fields']:
                        fig_pub, _ = st.session_state.heatmap_visualizer.create_diffusion_heatmap(
                            result['fields']['sigma_hydro'],
                            title=f"Diffusion Enhancement",
                            T_K=diffusion_T,
                            material=diffusion_material,
                            cmap_name='RdBu_r',
                            figsize=(10, 8),
                            log_scale=True,
                            target_angle=result['target_angle'],
                            defect_type=result['target_params']['defect_type'],
                            model=diffusion_model
                        )
                        fig_path = os.path.join(tmpdir, f"diffusion_enhancement.png")
                        fig_pub.savefig(fig_path, dpi=300, bbox_inches='tight')
                    
                    # Create ZIP
                    zip_filename = f"interpolation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    zip_path = os.path.join(tmpdir, zip_filename)
                    
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        zipf.write(json_path, os.path.basename(json_path))
                        zipf.write(csv_path, os.path.basename(csv_path))
                        if 'fig_path' in locals():
                            zipf.write(fig_path, os.path.basename(fig_path))
                    
                    # Read ZIP file
                    with open(zip_path, 'rb') as f:
                        zip_data = f.read()
                    
                    st.download_button(
                        label="📦 Download ZIP (All Data)",
                        data=zip_data,
                        file_name=zip_filename,
                        mime="application/zip",
                        use_container_width=True
                    )
    
    else:
        # Show welcome message when no results
        st.markdown("""
        ## 🎯 Welcome to Angular Bracketing Theory Interpolator
        
        ### Getting Started:
        1. **Load Solutions** from the sidebar
        2. **Configure Target Parameters** (angle, defect type, etc.)
        3. **Set Diffusion Physics** (material, temperature, model)
        4. **Set Angular Bracketing Parameters** (kernel sigma, attention balance)
        5. **Click "Perform Theory-Informed Interpolation"** to run
        
        ### Key Features:
        - **Physics-aware interpolation** using angular bracketing theory
        - **Diffusion enhancement visualization** with full theoretical implementation
        - **Correct eigenstrain values** for different defect types
        - **Publication-quality visualizations** with multiple export options
        - **Interactive comparison dashboard** for validation
        - **Theory-informed attention** combining learned and physical constraints
        
        ### 🧪 Diffusion Physics:
        - **D/D₀ = exp(Ωσ_h/(k_B T))**
        - **Peak enhancement** for positive hydrostatic stress (tensile)
        - **Valley suppression** for negative hydrostatic stress (compressive)
        - **Multiple materials**: Silver, Copper, Aluminum, Nickel, Iron
        - **Temperature dependence**: 300K to 1500K
        """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
