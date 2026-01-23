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

# Plotly colormap names (different from matplotlib)
PLOTLY_COLORMAPS = [
    'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
    'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
    'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis',
    'cool', 'coolwarm', 'copper', 'cubehelix', 'darkmint',
    'dark28', 'dense', 'earth', 'edge', 'electric', 'emrld',
    'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
    'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno',
    'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm',
    'mygbm', 'oranges', 'orrd', 'oryel', 'oxy', 'peach',
    'phase', 'picnic', 'pinkyl', 'piyg', 'plasma', 'plotly3',
    'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd',
    'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
    'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
    'spectral', 'speed', 'sunset', 'sunsetdark', 'teal',
    'tealgrn', 'tealrose', 'tempo', 'temps', 'thermal',
    'tropic', 'turbid', 'turbo', 'twilight', 'viridis',
    'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
]

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
        
        # Create surface plot with enhanced styling - FIXED: removed edgealpha
        surf = ax.plot_surface(X, Y, plot_data, cmap=cmap, norm=norm,
                              linewidth=0.5, antialiased=True, alpha=0.85,
                              rstride=1, cstride=1, edgecolor='k')  # Removed edgealpha
        
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
        
        # Validate colormap for Plotly
        # Convert matplotlib colormap names to plotly names
        cmap_mapping = {
            'RdBu': 'RdBu',
            'RdBu_r': 'RdBu_r',
            'viridis': 'Viridis',
            'plasma': 'Plasma',
            'inferno': 'Inferno',
            'magma': 'Magma',
            'cividis': 'Cividis',
            'coolwarm': 'Coolwarm',
            'Spectral': 'Spectral',
            'turbo': 'Turbo',
            'jet': 'Jet',
            'hot': 'Hot'
        }
        
        # Use mapped name or default to RdBu
        plotly_cmap = cmap_mapping.get(cmap_name, 'RdBu')
        
        # Create surface trace
        surface_trace = go.Surface(
            z=plot_data,
            x=X,
            y=Y,
            colorscale=plotly_cmap,  # Use validated colormap
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
            # Validate colormap for Plotly
            cmap_mapping = {
                'RdBu': 'RdBu',
                'RdBu_r': 'RdBu_r',
                'viridis': 'Viridis',
                'plasma': 'Plasma',
                'inferno': 'Inferno',
                'magma': 'Magma',
                'cividis': 'Cividis',
                'coolwarm': 'Coolwarm',
                'Spectral': 'Spectral',
                'turbo': 'Turbo',
                'jet': 'Jet',
                'hot': 'Hot'
            }
            
            plotly_cmap = cmap_mapping.get(cmap_name, 'Viridis')
            
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
                colorscale=plotly_cmap,
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
    
    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info,
                                   target_angle, defect_type, component='von_mises',
                                   cmap_name='viridis', figsize=(20, 15),
                                   ground_truth_index=None):
        """
        Create comprehensive comparison dashboard showing:
        1. Interpolated result
        2. Ground truth (selected source or closest match)
        3. Difference between interpolated and ground truth
        4. Weight distribution analysis (including Bracketing Kernel)
        5. Angular distribution of sources
        """
        
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Determine vmin and vmax for consistent scaling
        all_values = [interpolated_fields[component]]
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                all_values.append(gt_field)
        vmin = min(np.nanmin(field) for field in all_values)
        vmax = max(np.nanmax(field) for field in all_values)
        
        # 1. Interpolated result (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(interpolated_fields[component], cmap=cmap_name,
                        vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=f"{component.replace('_', ' ').title()} (GPa)")
        ax1.set_title(f'Interpolated Result\nθ = {target_angle:.1f}°, {defect_type}',
                     fontsize=16, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.2)
        
        # 2. Ground truth comparison (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                gt_theta = source_info['theta_degrees'][ground_truth_index]
                gt_distance = source_info['distances'][ground_truth_index]
                im2 = ax2.imshow(gt_field, cmap=cmap_name,
                                vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f"{component.replace('_', ' ').title()} (GPa)")
                ax2.set_title(f'Ground Truth\nθ = {gt_theta:.1f}° (Δ={gt_distance:.1f}°)',
                             fontsize=16, fontweight='bold')
                ax2.set_xlabel('X Position')
                ax2.set_ylabel('Y Position')
                ax2.grid(True, alpha=0.2)
            else:
                ax2.text(0.5, 0.5, f'Component "{component}"\nmissing in ground truth',
                        ha='center', va='center', fontsize=14, fontweight='bold')
                ax2.set_axis_off()
        else:
            ax2.text(0.5, 0.5, 'Select Ground Truth Source',
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax2.set_title('Ground Truth Selection', fontsize=16, fontweight='bold')
            ax2.set_axis_off()
        
        # 3. Difference plot (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                diff_field = interpolated_fields[component] - gt_field
                max_diff = np.max(np.abs(diff_field))
                im3 = ax3.imshow(diff_field, cmap='RdBu_r',
                                vmin=-max_diff, vmax=max_diff, aspect='equal',
                                interpolation='bilinear', origin='lower')
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Difference (GPa)')
                ax3.set_title(f'Difference\nMax Abs Error: {max_diff:.3f} GPa',
                             fontsize=16, fontweight='bold')
                ax3.set_xlabel('X Position')
                ax3.set_ylabel('Y Position')
                ax3.grid(True, alpha=0.2)
                
                # Calculate and display error metrics
                mse = np.mean(diff_field**2)
                mae = np.mean(np.abs(diff_field))
                rmse = np.sqrt(mse)
                error_text = (f"MSE: {mse:.4f}\n"
                             f"MAE: {mae:.4f}\n"
                             f"RMSE: {rmse:.4f}")
                ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes,
                        fontsize=12, fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            else:
                ax3.text(0.5, 0.5, 'Ground truth missing\nfor difference plot',
                        ha='center', va='center', fontsize=14, fontweight='bold')
                ax3.set_axis_off()
        else:
            ax3.text(0.5, 0.5, 'Difference will appear\nwhen ground truth is selected',
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax3.set_title('Difference Analysis', fontsize=16, fontweight='bold')
            ax3.set_axis_off()
        
        # 4. Weight distribution analysis (Middle Left) - ENHANCED
        ax4 = fig.add_subplot(gs[1, 0])
        if 'weights' in source_info:
            # Plot Final Attention Weights
            final_weights = source_info['weights']['combined']
            x = range(len(final_weights))
            bars = ax4.bar(x, final_weights, alpha=0.7, color='steelblue', edgecolor='black', label='Final Attention')
            
            # Overlay Spatial Kernel for comparison
            if 'spatial_kernel' in source_info['weights']:
                spatial_k = source_info['weights']['spatial_kernel']
                ax4.plot(x, spatial_k, 'g--', linewidth=2, label='Spatial Kernel', alpha=0.8)
            
            ax4.set_xlabel('Source Index')
            ax4.set_ylabel('Weight')
            ax4.set_title('Attention vs Spatial Kernel', fontsize=16, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.legend()
            
            # Highlight ground truth if applicable
            if ground_truth_index is not None and 0 <= ground_truth_index < len(bars):
                bars[ground_truth_index].set_color('red')
                bars[ground_truth_index].set_alpha(0.9)
        
        # 5. Angular distribution of sources (Middle Center)
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')
        if 'theta_degrees' in source_info and 'distances' in source_info:
            # Convert angles to radians for polar plot
            angles_rad = np.radians(source_info['theta_degrees'])
            distances = source_info['distances']
            
            # Plot sources as points with size proportional to final weight
            weights = source_info['weights']['combined']
            sizes = 100 * np.array(weights) / (np.max(weights) + 1e-8)  # Normalize sizes
            
            scatter = ax5.scatter(angles_rad, distances,
                                s=sizes, alpha=0.7, c='blue', edgecolors='black')
            
            # Plot target angle
            target_rad = np.radians(target_angle)
            ax5.scatter(target_rad, 0, s=200, c='red', marker='*', edgecolors='black', label='Target')
            
            # Plot habit plane (54.7°)
            habit_rad = np.radians(54.7)
            ax5.axvline(habit_rad, color='green', alpha=0.5, linestyle='--', label='Habit Plane (54.7°)')
            
            ax5.set_title('Angular Distribution of Sources', fontsize=16, fontweight='bold', pad=20)
            ax5.set_theta_zero_location('N')  # 0° at top
            ax5.set_theta_direction(-1)  # Clockwise
            ax5.legend(loc='upper right', fontsize=10)
        
        # 6. Component comparison (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        component_names = ['Von Mises', 'Hydrostatic', 'Stress Magnitude']
        
        # Get statistics for each component
        stats_data = []
        if 'statistics' in source_info:
            for comp in components:
                if comp in source_info['statistics']:
                    stats = source_info['statistics'][comp]
                    stats_data.append({
                        'component': comp,
                        'max': stats['max'],
                        'mean': stats['mean'],
                        'std': stats['std']
                    })
        
        if stats_data:
            x = np.arange(len(components))
            width = 0.25
            
            # Plot max values
            max_values = [stats['max'] for stats in stats_data]
            ax6.bar(x - width, max_values, width, label='Max', color='red', alpha=0.7)
            
            # Plot mean values
            mean_values = [stats['mean'] for stats in stats_data]
            ax6.bar(x, mean_values, width, label='Mean', color='blue', alpha=0.7)
            
            # Plot std values
            std_values = [stats['std'] for stats in stats_data]
            ax6.bar(x + width, std_values, width, label='Std', color='green', alpha=0.7)
            
            ax6.set_xlabel('Stress Component')
            ax6.set_ylabel('Value (GPa)')
            ax6.set_title('Component Statistics Comparison', fontsize=16, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(component_names)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Defect Type Gating Visualization (Bottom Left)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'weights' in source_info and 'defect_mask' in source_info['weights']:
            defect_masks = source_info['weights']['defect_mask']
            
            # Group by defect type
            defect_types_in_data = []
            for src in source_info.get('source_fields', []):
                dt = src.get('source_params', {}).get('defect_type', 'Unknown')
                if dt not in defect_types_in_data:
                    defect_types_in_data.append(dt)
            
            x_pos = np.arange(len(defect_masks))
            ax7.bar(x_pos, defect_masks, color='purple', alpha=0.6, label='Defect Mask')
            
            # Highlight active defect types
            for i, mask_val in enumerate(defect_masks):
                if mask_val > 0.1:
                    ax7.text(i, mask_val + 0.02, 'Active', ha='center', fontsize=8, fontweight='bold')
            
            ax7.set_xlabel('Source Index')
            ax7.set_ylabel('Gating Weight')
            ax7.set_title('Defect Type Hard Gating', fontsize=16, fontweight='bold')
            ax7.set_ylim([0, 1.1])
            ax7.legend()
        
        # 8. Spatial Correlation (Bottom Center/Right Span)
        ax8 = fig.add_subplot(gs[2, 1:])
        if ground_truth_index is not None and 0 <= ground_truth_index < len(source_fields):
            gt_field = source_fields[ground_truth_index].get(component)
            if gt_field is not None:
                interp_flat = interpolated_fields[component].flatten()
                gt_flat = gt_field.flatten()
                
                # Create scatter plot
                ax8.scatter(gt_flat, interp_flat, alpha=0.5, s=10)
                
                # Add correlation line
                min_val = min(np.min(gt_flat), np.min(interp_flat))
                max_val = max(np.max(gt_flat), np.max(interp_flat))
                ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
                
                # Calculate correlation coefficient
                from scipy.stats import pearsonr
                try:
                    corr_coef, _ = pearsonr(gt_flat, interp_flat)
                except:
                    corr_coef = 0.0
                
                ax8.set_xlabel(f'Ground Truth {component.replace("_", " ").title()} (GPa)')
                ax8.set_ylabel(f'Interpolated {component.replace("_", " ").title()} (GPa)')
                ax8.set_title(f'Spatial Correlation Analysis\nPearson: {corr_coef:.3f}', fontsize=16, fontweight='bold')
                ax8.grid(True, alpha=0.3)
                ax8.legend()
                
                # Add stats box
                mse = np.mean((interp_flat - gt_flat)**2)
                mae = np.mean(np.abs(interp_flat - gt_flat))
                stats_text = (f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nPearson: {corr_coef:.3f}')
                ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                        fontsize=12, fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle(f'Theory-Informed Interpolation - Target θ={target_angle:.1f}°, {defect_type}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_interactive_3d_surface(self, stress_field, title="3D Stress Surface",
                                     cmap_name='viridis', width=900, height=700,
                                     target_angle=None, defect_type=None):
        """Create interactive 3D surface plot with Plotly"""
        try:
            # Validate colormap for Plotly
            cmap_mapping = {
                'RdBu': 'RdBu',
                'RdBu_r': 'RdBu_r',
                'viridis': 'Viridis',
                'plasma': 'Plasma',
                'inferno': 'Inferno',
                'magma': 'Magma',
                'cividis': 'Cividis',
                'coolwarm': 'Coolwarm',
                'Spectral': 'Spectral',
                'turbo': 'Turbo',
                'jet': 'Jet',
                'hot': 'Hot'
            }
            
            plotly_cmap = cmap_mapping.get(cmap_name, 'Viridis')
            
            # Create meshgrid
            x = np.arange(stress_field.shape[1])
            y = np.arange(stress_field.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Create hover text
            hover_text = []
            for i in range(stress_field.shape[0]):
                row_text = []
                for j in range(stress_field.shape[1]):
                    row_text.append(f"X: {j}, Y: {i}<br>Stress: {stress_field[i, j]:.4f} GPa")
                hover_text.append(row_text)
            
            # Create 3D surface trace
            surface_trace = go.Surface(
                z=stress_field,
                x=X,
                y=Y,
                colorscale=plotly_cmap,
                opacity=0.9,
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
                },
                hoverinfo='text',
                text=hover_text
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
                        title=dict(text="X Position", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    yaxis=dict(
                        title=dict(text="Y Position", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    zaxis=dict(
                        title=dict(text="Stress (GPa)", font=dict(size=18, family="Arial", color="black")),
                        tickfont=dict(size=14),
                        gridcolor='rgb(200, 200, 200)',
                        backgroundcolor='white'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.0)
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7)
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=100, b=0)
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Error creating 3D surface: {e}")
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
        
        # Create surface plot - FIXED: removed edgealpha
        surf = ax.plot_surface(X, Y, stress_field, cmap=cmap, norm=norm,
                              linewidth=0, antialiased=True, alpha=0.8, rstride=1, cstride=1)  # Removed edgealpha
        
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
                               linewidth=0, antialiased=True, alpha=0.8)  # Removed edgealpha
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
        page
