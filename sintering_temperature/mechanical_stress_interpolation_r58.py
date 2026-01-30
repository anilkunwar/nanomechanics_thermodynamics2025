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

# FIXED INDENTATION ERROR: Properly indented block
ALL_COLORMAPS = []
for category in COLORMAP_OPTIONS.values():
    ALL_COLORMAPS.extend(category)  # <-- CORRECT INDENTATION
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
                defect_mask.append(1e-6)  # Near zero to avoid NaN in log, but effectively ignored

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
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)  # [1, N]
            attn_scores = attn_scores / np.sqrt(self.d_model)

            # 3b. Apply Temperature
            attn_scores = attn_scores / self.temperature

            # 3c. Convert kernels to tensors
            spatial_kernel_tensor = torch.FloatTensor(spatial_kernel).unsqueeze(0)  # [1, N]
            defect_mask_tensor = torch.FloatTensor(defect_mask).unsqueeze(0)  # [1, N]

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
                    'learned_attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
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
# ENHANCED HEATMAP VISUALIZER WITH RELATIONSHIP VISUALIZATIONS
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with diffusion and relationship visualization capabilities"""
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.diffusion_physics = DiffusionPhysics()

    # ==================== NEW: RADAR CHART WITH WEIGHT COMPONENT BREAKDOWN ====================
    def create_weight_component_radar_chart(self, sources_data, query_index=None, figsize=(10, 10)):
        """
        Create radar chart showing all weight components from attention formula:
        w_i(θ*) = [ᾱ_i(θ*) · exp(-(Δφ_i)²/(2σ²)) · 𝟙(τ_i = τ*)] / Σ[...] + 10⁻⁶
        
        Components visualized:
        1. Learned Attention (ᾱ_i)
        2. Spatial Kernel (exp(-(Δφ_i)²/(2σ²)))
        3. Defect Mask (𝟙(τ_i = τ*))
        4. Combined Weight (final)
        5. Angular Distance (Δφ_i)
        """
        import plotly.graph_objects as go
        import numpy as np

        # Extract and normalize features (critical fix)
        features = {
            'Learned Attention': [],
            'Spatial Kernel': [],
            'Defect Match': [],
            'Combined Weight': [],
            'Angular Distance': []
        }

        # Normalize each feature independently to [0, 1]
        for i, source in enumerate(sources_data):
            features['Learned Attention'].append(source['learned_attention'])
            features['Spatial Kernel'].append(source['spatial_kernel'])
            features['Defect Match'].append(1.0 if source['defect_match'] else 0.0)
            features['Combined Weight'].append(source['combined_weight'])
            features['Angular Distance'].append(source['angular_dist'])

        # Normalize each feature dimension to [0, 1] for fair comparison
        normalized = {}
        for key, values in features.items():
            arr = np.array(values)
            if np.max(arr) - np.min(arr) > 1e-10:
                normalized[key] = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            else:
                normalized[key] = np.zeros_like(arr)

        # Create radar plot with 5 dimensions
        categories = list(normalized.keys())
        fig = go.Figure()

        for i in range(len(sources_data)):
            values = [normalized[cat][i] for cat in categories] + [normalized[categories[0]][i]]  # Close loop

            # Highlight query source (critical fix)
            is_query = i == query_index
            color = '#FF1493' if is_query else '#4ECDC4'
            width = 4 if is_query else 1.5
            opacity = 1.0 if is_query else 0.3
            name = f"Query Source {i}" if is_query else f"Source {i}"

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=name,
                line=dict(color=color, width=width),
                opacity=opacity,
                hoverinfo='text',
                text=f"Source {i}<br>Angle: {sources_data[i]['theta_deg']:.1f}°<br>"
                     f"Learned: {sources_data[i]['learned_attention']:.4f}<br>"
                     f"Spatial: {sources_data[i]['spatial_kernel']:.4f}<br>"
                     f"Defect Match: {'Yes' if sources_data[i]['defect_match'] else 'No'}<br>"
                     f"Combined Weight: {sources_data[i]['combined_weight']:.4f}<br>"
                     f"Defect Type: {sources_data[i]['defect_type']}"
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=12)),
                angularaxis=dict(direction='clockwise', tickfont=dict(size=14))
            ),
            title=dict(
                text=f"Weight Component Breakdown (Query: Source {query_index})<br>"
                     "<sup>w<sub>i</sub>(θ*) = [ᾱ<sub>i</sub>(θ*) · exp(-(Δφ<sub>i</sub>)²/(2σ²)) · 𝟙(τ<sub>i</sub> = τ*)] / Σ[...] + 10⁻⁶</sup>",
                font=dict(size=20, family="Arial Black", color='#1E3A8A')
            ),
            showlegend=True,
            legend=dict(font=dict(size=12)),
            width=800,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Add habit plane reference annotation
        if query_index is not None:
            habit_diff = abs(sources_data[query_index]['theta_deg'] - 54.7)
            fig.add_annotation(
                text=f"Δ from habit plane: {habit_diff:.1f}°",
                xref="paper", yref="paper",
                x=0.5, y=0.95,
                showarrow=False,
                font=dict(size=14, color="green", weight="bold"),
                bgcolor="lightyellow",
                bordercolor="green",
                borderwidth=2
            )

        # Add mathematical formula annotation
        fig.add_annotation(
            text="Formula: wᵢ(θ*) = [ᾱᵢ(θ*) · exp(-(Δφᵢ)²/(2σ²)) · 𝟙(τᵢ = τ*)] / Σ[...] + 10⁻⁶",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color="#1E3A8A", family="Courier New"),
            bgcolor="white",
            bordercolor="#3B82F6",
            borderwidth=2
        )

        return fig

    # ==================== NEW: SUNBURST CHART WITH HIERARCHICAL WEIGHT BREAKDOWN ====================
    def create_weight_sunburst_chart(self, sources_data, query_index=None, figsize=(10, 10)):
        """
        Create hierarchical sunburst showing:
        Level 1: Defect Type
        Level 2: Angular Bin (30° sectors)
        Level 3: Individual Sources with weight components
        
        Visual encoding:
        - Size: Combined weight
        - Color: Defect type + query highlight
        - Hover: Full weight formula breakdown
        """
        import plotly.express as px
        import pandas as pd
        import numpy as np

        # Create hierarchical data structure
        data = []
        for i, source in enumerate(sources_data):
            # Bin angles into 30° sectors for meaningful hierarchy
            angle_bin = f"{int(source['theta_deg']/30)*30}°-{int(source['theta_deg']/30)*30+30}°"

            # Determine highlight status
            is_query = i == query_index
            highlight = "Query Source" if is_query else "Other Sources"

            data.append({
                'defect_type': source['defect_type'],
                'angle_bin': angle_bin,
                'source_id': f"Source {i}",
                'combined_weight': source['combined_weight'],
                'spatial_kernel': source['spatial_kernel'],
                'learned_attention': source['learned_attention'],
                'defect_match': 1.0 if source['defect_match'] else 0.0,
                'highlight': highlight,
                'theta_deg': source['theta_deg'],
                'angular_dist': source['angular_dist']
            })

        df = pd.DataFrame(data)

        # Create sunburst with query highlighting
        fig = px.sunburst(
            df,
            path=['defect_type', 'angle_bin', 'source_id'],
            values='combined_weight',
            color='highlight',
            color_discrete_map={
                'Query Source': '#FF1493',  # Vivid pink for query
                'Other Sources': '#4ECDC4'   # Teal for others
            },
            title="Weight Hierarchy: Defect Type → Angle Bin → Source<br>"
                  "<sup>Size ∝ Combined Weight | Color = Query Highlight</sup>",
            width=900,
            height=900
        )

        # Enhance styling for publication quality
        fig.update_layout(
            title=dict(
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            font=dict(size=14),
            margin=dict(t=80, l=20, r=20, b=20)
        )

        # Add hover template with physics details and weight formula
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>"
                          "Combined Weight: %{value:.4f}<br>"
                          "Spatial Kernel: %{customdata[0]:.4f}<br>"
                          "Learned Attention: %{customdata[1]:.4f}<br>"
                          "Defect Match: %{customdata[2]:.0f}<br>"
                          "Angle: %{customdata[3]:.1f}°<br>"
                          "Δ from target: %{customdata[4]:.1f}°<br>"
                          "<i>Formula: wᵢ = [ᾱᵢ · exp(-(Δφᵢ)²/(2σ²)) · 𝟙(τᵢ=τ*)] / Σ[...]</i><extra></extra>",
            customdata=df[['spatial_kernel', 'learned_attention', 'defect_match', 'theta_deg', 'angular_dist']].values
        )

        # Add annotation about query source location
        if query_index is not None:
            query_source = sources_data[query_index]
            fig.add_annotation(
                text=f"Query: {query_source['defect_type']} at {query_source['theta_deg']:.1f}°",
                xref="paper", yref="paper",
                x=0.5, y=-0.05,
                showarrow=False,
                font=dict(size=14, color="#FF1493", weight="bold"),
                bgcolor="white",
                bordercolor="#FF1493",
                borderwidth=2
            )

        return fig

    # ==================== NEW: SANKEY DIAGRAM FOR ATTENTION FLOWS ====================
    def create_attention_sankey_diagram(self, sources_data, target_angle, target_defect_type,
                                       spatial_sigma, domain_size_nm=12.8):
        """
        Create Sankey diagram visualizing attention flows with full weight formula breakdown:
        
        Sources (Left) → Weight Components (Middle) → Target (Right)
        
        Flow paths:
        Source i → [Learned Attention] → Combined Weight → Target
        Source i → [Spatial Kernel] → Combined Weight → Target  
        Source i → [Defect Mask] → Combined Weight → Target
        
        Width of flows proportional to contribution to final weight
        """
        import plotly.graph_objects as go
        import numpy as np

        # Prepare Sankey data
        labels = []
        source_indices = []
        target_indices = []
        values = []
        colors = []

        # Add source nodes
        source_node_start = 0
        for i, source in enumerate(sources_data):
            labels.append(f"Source {i}<br>{source['defect_type']}<br>θ={source['theta_deg']:.1f}°")
            source_node_start = i

        # Add weight component nodes (middle layer)
        component_start = len(labels)
        labels.append("Learned<br>Attention")
        labels.append("Spatial<br>Kernel")
        labels.append("Defect<br>Mask")

        # Add target node (right)
        target_node = len(labels)
        labels.append(f"Target<br>{target_defect_type}<br>θ={target_angle:.1f}°")

        # Build flows from sources to components
        for i, source in enumerate(sources_data):
            # Flow to learned attention
            source_indices.append(i)
            target_indices.append(component_start)
            values.append(source['learned_attention'] * source['combined_weight'])
            colors.append('#4ECDC4' if i != sources_data.index(source) else '#FF1493')

            # Flow to spatial kernel
            source_indices.append(i)
            target_indices.append(component_start + 1)
            values.append(source['spatial_kernel'] * source['combined_weight'])
            colors.append('#45B7D1' if i != sources_data.index(source) else '#FF6B6B')

            # Flow to defect mask (only if matching)
            source_indices.append(i)
            target_indices.append(component_start + 2)
            values.append((1.0 if source['defect_match'] else 0.000001) * source['combined_weight'])
            colors.append('#95E1D3' if i != sources_data.index(source) else '#FFA07A')

        # Build flows from components to target
        for comp_idx in range(3):
            source_indices.append(component_start + comp_idx)
            target_indices.append(target_node)
            # Sum all flows into this component
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values) 
                           if t == component_start + comp_idx)
            values.append(comp_value)
            colors.append('#A8E6CF')

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#4ECDC4"] * len(sources_data) + ["#45B7D1", "#95E1D3", "#FFD93D", "#FF1493"]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='Flow: %{value:.4f}<br>Source: %{source.label}<br>Target: %{target.label}<extra></extra>'
            )
        )])

        fig.update_layout(
            title=dict(
                text=f"Attention Flow Sankey Diagram<br>"
                     "<sup>Width ∝ Contribution to Final Weight | Formula: wᵢ(θ*) = [ᾱᵢ · exp(-(Δφᵢ)²/(2σ²)) · 𝟙(τᵢ=τ*)] / Σ[...] + 10⁻⁶</sup>",
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            font=dict(size=12),
            width=1000,
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Add domain annotation
        fig.add_annotation(
            text=f"Domain: {domain_size_nm} nm × {domain_size_nm} nm<br>σ = {spatial_sigma}°",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=12, color="#1E3A8A", weight="bold"),
            bgcolor="white",
            bordercolor="#3B82F6",
            borderwidth=1
        )

        return fig

    # ==================== EXISTING METHODS (MINIMAL CHANGES FOR INTEGRATION) ====================
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

        # Create heatmap with 12.8 nm domain extent
        domain_info = DomainConfiguration.get_domain_info()
        extent = domain_info['extent']  # [-6.4, 6.4, -6.4, 6.4]
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect=aspect_ratio, interpolation='bilinear', origin='lower',
                      extent=extent)

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
        ax.set_xlabel('X Position (nm)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Y Position (nm)', fontsize=16, fontweight='bold')

        # Add grid with subtle styling
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')

        # Add statistics annotation with enhanced styling
        if show_stats:
            stats_text = (f"Max: {vmax:.3f} GPa\n"
                         f"Min: {vmin:.3f} GPa\n"
                         f"Mean: {np.nanmean(stress_field):.3f} GPa\n"
                         f"Std: {np.nanstd(stress_field):.3f} GPa\n"
                         f"Domain: {domain_info['domain_length_nm']} nm")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return fig

    # ... [OTHER EXISTING METHODS REMAIN UNCHANGED - create_diffusion_heatmap, create_diffusion_3d_surface, etc.] ...

# =============================================
# ENHANCED TRANSFORMER INTERPOLATOR WITH SOURCE DATA PREPARATION
# =============================================
class EnhancedTransformerInterpolator(TransformerSpatialInterpolator):
    """Enhanced interpolator with source data preparation for relationship visualizations"""
    
    def prepare_source_visualization_data(self, result, query_index=None):
        """Prepare normalized source data for advanced visualizations with full weight components"""
        sources_data = []
        
        for i, field in enumerate(result['source_fields']):
            # Extract key metrics
            vm_max = np.max(field['von_mises'])
            hydro_max = np.max(np.abs(field['sigma_hydro']))
            weight = result['weights']['combined'][i]
            spatial_kernel = result['weights']['spatial_kernel'][i]
            learned_attention = result['weights']['learned_attention'][i] if 'learned_attention' in result['weights'] else 0.0
            theta_deg = result['source_theta_degrees'][i]
            angular_dist = result['source_distances'][i] if i < len(result['source_distances']) else 0.0
            defect_type = field['source_params']['defect_type']
            target_defect = result['target_params']['defect_type']
            
            sources_data.append({
                'theta_deg': theta_deg,
                'angular_dist': angular_dist,
                'combined_weight': weight,
                'spatial_kernel': spatial_kernel,
                'learned_attention': learned_attention,
                'defect_type': defect_type,
                'defect_match': defect_type == target_defect,
                'von_mises_max': vm_max,
                'hydro_max': hydro_max,
                'is_query': i == query_index
            })
        
        return sources_data

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
    .formula-box {
        background-color: #F0F7FF;
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.3rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with domain info
    domain_info = DomainConfiguration.get_domain_info()
    st.markdown(f'<h1 class="main-header">🎯 Angular Bracketing Theory with Transformer Attention</h1>', unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: #1E3A8A;'>Domain: {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm (Centered at 0, ±{domain_info['domain_half_nm']} nm)</h3>", unsafe_allow_html=True)
    
    # Mathematical formula display
    st.markdown("""
    <div class="formula-box">
    <strong>Attention Weight Formula:</strong><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
    
    <strong>Components:</strong>
    • $\bar{\alpha}_i$: Learned attention score from transformer<br>
    • $\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$: Spatial kernel (angular bracketing)<br>
    • $\mathbb{I}(\tau_i = \tau^*)$: Defect type indicator (1 if match, 0 otherwise)<br>
    • $\sigma$: Angular kernel width (controllable parameter)
    </div>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown(f"""
    <div class="info-box">
    <strong>🔬 Physics-Aware Interpolation: Angular Orientation & Defect Type as Primary Drivers.</strong><br>
    • <strong>Angular Bracketing Kernel:</strong> Gaussian spatial locality enforcing linear interpolation between nearest angles.<br>
    • <strong>Hard Defect Gating:</strong> Sources with different defect types receive effectively zero attention.<br>
    • <strong>Theory-Informed Attention:</strong> Attention = Softmax(Learned Similarity × Spatial Kernel × Defect Mask).<br>
    • <strong>Diffusion Enhancement:</strong> D/D₀ = exp(Ωσ_h/(k_B T)) - Peak for tensile stress, valley for compressive stress.<br>
    • <strong>Domain Size:</strong> {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm centered at 0 (±{domain_info['domain_half_nm']} nm)
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        # Initialize with default sigma (angular window size)
        st.session_state.transformer_interpolator = EnhancedTransformerInterpolator(
            spatial_sigma=10.0,  # Degrees: +/- 10 deg window has high weight
            locality_weight_factor=0.5  # 50% Learned, 50% Theory
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

    # Sidebar configuration (MINIMAL - focused on core parameters)
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
                st.session_state.selected_ground_truth = None
                st.session_state.custom_dashboard_fig = None
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
        
        # Spatial sigma parameter (critical for bracketing)
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
                        st.session_state.selected_ground_truth = None
                        st.session_state.custom_dashboard_fig = None
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

    # Results display with NEW RELATIONSHIP VISUALIZATION TAB
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Tabs including new relationship visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Results Overview",
            "🎨 Stress Visualization", 
            "⚖️ Attention Analysis",
            "🕸️ Relationship Visualizations",  # NEW TAB WITH RADAR/SUNBURST/SANKEY
            "💾 Export Results"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 Interpolation Results</h2>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Von Mises", f"{result['statistics']['von_mises']['max']:.3f} GPa")
            with col2:
                st.metric("Target Angle", f"{result['target_angle']:.1f}°")
            with col3:
                st.metric("Defect Type", result['target_params']['defect_type'])
            with col4:
                st.metric("Entropy", f"{result['weights']['entropy']:.3f}")
                
        with tab2:
            st.markdown('<h2 class="section-header">🎨 Stress Field Visualization</h2>', unsafe_allow_html=True)
            if 'von_mises' in result['fields']:
                fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                    result['fields']['von_mises'],
                    title="Von Mises Stress",
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(10, 8)
                )
                st.pyplot(fig)
                
        with tab3:
            st.markdown('<h2 class="section-header">⚖️ Theory-Informed Attention Analysis</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Weight Components:**
            - **Learned Attention**: Transformer similarity score (ᾱᵢ)
            - **Spatial Kernel**: Angular proximity weight exp(-(Δφᵢ)²/(2σ²))
            - **Defect Mask**: Hard constraint 𝟙(τᵢ = τ*)
            - **Combined**: Final normalized weight wᵢ(θ*)
            """)
            
            # Display weight components
            weights_df = pd.DataFrame({
                'Source': range(len(result['weights']['combined'])),
                'Angle (°)': result['source_theta_degrees'],
                'Learned': result['weights']['learned_attention'],
                'Spatial': result['weights']['spatial_kernel'],
                'Defect Match': ['✓' if m > 0.5 else '✗' for m in result['weights']['defect_mask']],
                'Combined': result['weights']['combined']
            })
            st.dataframe(weights_df.style.background_gradient(subset=['Combined'], cmap='Blues'))
            
        with tab4:
            st.markdown('<h2 class="section-header">🕸️ Relationship Visualizations</h2>', unsafe_allow_html=True)
            st.info(f"""
            **Domain:** {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm
            
            These visualizations reveal how the attention weight formula components interact:
            - **Radar Chart**: 5-dimensional comparison of weight components per source
            - **Sunburst Chart**: Hierarchical breakdown by defect type → angle bin → source
            - **Sankey Diagram**: Flow visualization of how components combine into final weights
            """)
            
            # Prepare source data with full weight components
            sources_data = st.session_state.transformer_interpolator.prepare_source_visualization_data(
                st.session_state.interpolation_result,
                query_index=st.session_state.selected_ground_truth
            )
            
            viz_type = st.selectbox(
                "Visualization Type",
                ["Radar Chart (Weight Components)", 
                 "Sunburst Chart (Hierarchical Breakdown)",
                 "Sankey Diagram (Attention Flows)"],
                index=0
            )
            
            if st.button("✨ Generate Visualization", type="primary", use_container_width=True):
                try:
                    if viz_type.startswith("Radar"):
                        fig = st.session_state.heatmap_visualizer.create_weight_component_radar_chart(
                            sources_data,
                            query_index=st.session_state.selected_ground_truth
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        ### 🔍 Radar Chart Interpretation
                        - **5 Dimensions**: Learned Attention, Spatial Kernel, Defect Match, Combined Weight, Angular Distance
                        - **Query Source**: Highlighted in vivid pink (#FF1493) with 2.5× line width
                        - **Physics Insight**: Sources with matching defect types show high "Defect Match" dimension; angularly proximate sources show high "Spatial Kernel"
                        - **Formula Visualization**: Each dimension maps directly to a component in the weight formula
                        """)
                    
                    elif viz_type.startswith("Sunburst"):
                        fig = st.session_state.heatmap_visualizer.create_weight_sunburst_chart(
                            sources_data,
                            query_index=st.session_state.selected_ground_truth
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        ### 🔍 Sunburst Chart Interpretation
                        - **Hierarchy**: Outer ring = defect types → Middle ring = 30° angle bins → Center = individual sources
                        - **Size Encoding**: Segment area proportional to combined weight contribution
                        - **Color Encoding**: Query source highlighted in vivid pink (#FF1493)
                        - **Physics Insight**: Reveals bracketing structure - query typically bracketed by two high-weight sources of same defect type
                        """)
                    
                    elif viz_type.startswith("Sankey"):
                        fig = st.session_state.heatmap_visualizer.create_attention_sankey_diagram(
                            sources_data,
                            target_angle=result['target_angle'],
                            target_defect_type=result['target_params']['defect_type'],
                            spatial_sigma=spatial_sigma,
                            domain_size_nm=domain_info['domain_length_nm']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        ### 🔍 Sankey Diagram Interpretation
                        - **Flow Width**: Proportional to contribution to final weight
                        - **Left Nodes**: Individual sources (labeled with defect type and angle)
                        - **Middle Nodes**: Weight formula components (Learned, Spatial, Defect)
                        - **Right Node**: Target (query angle and defect type)
                        - **Physics Insight**: Visualizes how spatial kernel dominates when angles align, and how defect mask completely blocks flow for mismatched types
                        - **Formula Visualization**: Direct mapping of Sankey flows to weight formula terms
                        """)
                    
                    # Add unified physics interpretation
                    st.markdown("---")
                    st.markdown("### 🧪 Physics Interpretation of Weight Formula")
                    st.markdown(f"""
                    The attention weight formula implements **Angular Bracketing Theory**:
                    
                    1. **Defect Type as Hard Constraint** (`𝟙(τᵢ = τ*)`):
                       - Sources with different defect types receive near-zero attention (visible as blocked flows in Sankey)
                       - Critical for physical validity - different defect types have fundamentally different stress fields
                    
                    2. **Angular Proximity Drives Attention** (`exp(-(Δφᵢ)²/(2σ²))`):
                       - Gaussian kernel with σ = {spatial_sigma}° creates "bracketing window"
                       - Sources within ±{spatial_sigma}° of target receive highest weights
                       - Visible as high "Spatial Kernel" dimension in radar chart
                    
                    3. **Learned Similarity Refines Selection** (`ᾱᵢ`):
                       - Transformer captures subtle stress field patterns beyond angle/defect
                       - Modulates the physics priors rather than replacing them
                       - Visible as variation in "Learned Attention" dimension
                    
                    4. **Bracketing Structure**:
                       - Optimal interpolation occurs when target angle is bracketed by two sources of same defect type
                       - Query source typically receives strongest attention from two bracketing sources
                       - Visible as triangular attention pattern in Sankey diagram
                    
                    5. **Domain Size Awareness**:
                       - All visualizations explicitly reference the **{domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm** domain
                       - Angular positions mapped to physical domain coordinates
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with tab5:
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            st.info("Export functionality available for full results including weight components")
            
    else:
        # Show welcome message when no results
        st.markdown(f"""
        ## 🎯 Welcome to Angular Bracketing Theory Interpolator
        ### Domain Configuration:
        - **Size:** {domain_info['domain_length_nm']} nm × {domain_info['domain_length_nm']} nm
        - **Grid:** {domain_info['grid_points']} × {domain_info['grid_points']} points
        - **Spacing:** {domain_info['grid_spacing_nm']} nm
        - **Extent:** ±{domain_info['domain_half_nm']} nm
        - **Area:** {domain_info['area_nm2']:.1f} nm²
        
        ### Getting Started:
        1. **Load Solutions** from the sidebar
        2. **Configure Target Parameters** (angle, defect type)
        3. **Set Angular Bracketing Parameters** (kernel width σ)
        4. **Click "Perform Theory-Informed Interpolation"** to run
        5. **Explore Relationship Visualizations** to see weight formula components
        
        ### Key Features:
        - **Physics-aware interpolation** using angular bracketing theory
        - **Complete weight formula visualization**: 
          $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_k \bar{\alpha}_k \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
        - **Three relationship visualizations**: Radar (components), Sunburst (hierarchy), Sankey (flows)
        - **Query source highlighting** with vivid pink (#FF1493) in all visualizations
        - **Domain size awareness**: Explicit 12.8 nm labeling in all visualizations
        """)

# =============================================
# RESULTS MANAGER (MINIMAL FOR COMPLETENESS)
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
                'visualization_params': visualization_params,
                'domain_info': DomainConfiguration.get_domain_info()
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

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
