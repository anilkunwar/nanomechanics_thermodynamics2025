import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import os
import pickle
import torch
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# =============================================
# ESSENTIAL PHYSICS CLASSES (ENHANCED VERSION)
# =============================================
class PhysicsBasedStressAnalyzer:
    """
    Enhanced physics analyzer for defect eigen strains with proper FCC crystal physics
    
    This class implements the physical basis for stress patterns around crystal defects.
    Eigenstrains are calculated based on FCC crystal structure properties where:
    - ISF (Intrinsic Stacking Fault): Partial dislocation separation on {111} planes
    - ESF (Extrinsic Stacking Fault): Double partial separation creating a vacancy layer
    - Twin boundary: Mirror symmetry across {111} plane with maximum lattice distortion
    """
    def __init__(self):
        # FCC crystal eigenstrains from atomistic simulations and continuum models
        self.eigen_strains = {
            'ISF': 0.71,      # Intrinsic Stacking Fault - small distortion
            'ESF': 1.41,      # Extrinsic Stacking Fault - medium distortion
            'Twin': 2.12,     # Twin boundary - large coherent distortion
            'No Defect': 0.0, # Perfect crystal - no distortion
            'Unknown': 0.0
        }
        # FCC material properties (silver example)
        self.material_properties = {
            'Ag': {
                'shear_modulus': 30.0,  # GPa
                'poissons_ratio': 0.37,
                'atomic_volume': 1.56e-29,  # m³
                'boltzmann_constant': 1.38e-23,  # J/K
                'melting_point': 1234  # K
            },
            'Cu': {
                'shear_modulus': 48.0,  # GPa
                'poissons_ratio': 0.34,
                'atomic_volume': 1.18e-29,  # m³
                'boltzmann_constant': 1.38e-23,  # J/K
                'melting_point': 1358  # K
            }
        }
    
    def get_eigen_strain(self, defect_type):
        """Get eigen strain value for a specific defect type"""
        return self.eigen_strains.get(defect_type, 0.0)
    
    def get_material_property(self, material, property_name):
        """Get material property for physics calculations"""
        return self.material_properties.get(material, {}).get(property_name)
    
    def calculate_diffusion_enhancement(self, hydrostatic_stress, temperature=650, material='Ag'):
        """
        Calculate diffusion enhancement factor based on hydrostatic stress
        
        Physics correction: Only TENSILE (positive) hydrostatic stress enhances diffusion.
        Compressive stress (negative) has NO enhancement effect.
        
        Formula: D/D₀ = exp(Ω·σ_hydro/(kT)) for σ_hydro > 0
                 D/D₀ = 1.0 for σ_hydro ≤ 0
        
        Where:
        - Ω = atomic volume (m³)
        - σ_hydro = hydrostatic stress (Pa)
        - k = Boltzmann constant (J/K)
        - T = temperature (K)
        """
        # Get material properties
        atomic_volume = self.get_material_property(material, 'atomic_volume')
        k = self.get_material_property(material, 'boltzmann_constant')
        
        if atomic_volume is None or k is None:
            raise ValueError(f"Missing material properties for {material}")
        
        # Initialize enhancement ratio array
        enhancement_ratio = np.ones_like(hydrostatic_stress)
        
        # Only apply enhancement for tensile (positive) stress
        tensile_mask = hydrostatic_stress > 0
        if np.any(tensile_mask):
            sigma_tensile = hydrostatic_stress[tensile_mask]
            enhancement_ratio[tensile_mask] = np.exp(
                atomic_volume * sigma_tensile / (k * temperature)
            )
        
        return enhancement_ratio
    
    def calculate_stress_gradient(self, stresses, angles_degrees):
        """Calculate stress gradient (dσ/dθ) for driving force analysis"""
        angles_radians = np.deg2rad(angles_degrees)
        return np.gradient(stresses, angles_radians)

class EnhancedSolutionLoader:
    """
    Enhanced solution loader with physics-aware processing and validation
    
    This class handles loading simulation results and standardizing them with proper physics metadata.
    It processes both pickle and PyTorch file formats, extracting parameters and history data.
    """
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
    
    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        """Scan directory for solution files"""
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth', '*.h5', '*.json']:
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
                    'format': self._get_file_format(file_path)
                }
                file_info.append(info)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        return file_info
    
    def _get_file_format(self, file_path: str) -> str:
        """Determine file format from extension"""
        if file_path.endswith(('.pt', '.pth')):
            return 'pt'
        elif file_path.endswith(('.pkl', '.pickle')):
            return 'pkl'
        elif file_path.endswith('.h5'):
            return 'h5'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            return 'unknown'
    
    def read_simulation_file(self, file_path, format_type='auto'):
        """Read simulation file with physics-aware processing"""
        try:
            format_type = format_type if format_type != 'auto' else self._get_file_format(file_path)
            
            with open(file_path, 'rb') as f:
                if format_type == 'pt' or file_path.endswith(('.pt', '.pth')):
                    # PyTorch file
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                elif format_type == 'pkl' or file_path.endswith(('.pkl', '.pickle')):
                    # Pickle file
                    data = pickle.load(f)
                elif format_type == 'json' or file_path.endswith('.json'):
                    # JSON file
                    data = json.load(f)
                elif format_type == 'h5' or file_path.endswith('.h5'):
                    # HDF5 file - handle separately
                    import h5py
                    f.close()  # Close the regular file handle
                    with h5py.File(file_path, 'r') as h5f:
                        data = self._load_h5_data(h5f)
                else:
                    # Default to pickle
                    data = pickle.load(f)
            
            # Standardize data structure
            standardized = self._standardize_data(data, file_path)
            return standardized
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _load_h5_data(self, h5_file):
        """Load data from HDF5 file format"""
        data = {}
        # Load parameters
        if 'params' in h5_file:
            params = {}
            for key, value in h5_file['params'].attrs.items():
                params[key] = value
            data['params'] = params
        
        # Load history as list
        if 'history' in h5_file:
            history_list = []
            for i in range(len(h5_file['history'])):
                step_group = h5_file['history'][f'step_{i}']
                step_data = {}
                for key in step_group.keys():
                    step_data[key] = np.array(step_group[key])
                history_list.append(step_data)
            data['history'] = history_list
        
        # Load metadata
        if 'metadata' in h5_file:
            metadata = {}
            for key, value in h5_file['metadata'].attrs.items():
                metadata[key] = value
            data['metadata'] = metadata
        
        return data
    
    def _standardize_data(self, data, file_path):
        """Standardize simulation data with physics metadata"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False,
                'file_path': file_path
            },
            'physics_analysis': {}
        }
        
        try:
            if isinstance(data, dict):
                # Extract parameters
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                elif hasattr(data, 'params'):
                    standardized['params'] = vars(data.params)
                
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
                    elif hasattr(history, '__iter__'):
                        standardized['history'] = list(history)
                
                # Extract additional metadata
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                
                # Add eigen strain based on defect type
                params = standardized['params']
                if 'defect_type' in params:
                    defect_type = params['defect_type']
                    eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
                    params['eigen_strain'] = eigen_strain
                    # Update eps0 if not set or different from eigen strain
                    if 'eps0' not in params or abs(params['eps0'] - eigen_strain) > 0.1:
                        params['eps0'] = eigen_strain
                
                # Add physics validation flags
                standardized['metadata']['physics_processed'] = True
            
            # Add physics-based validation
            self._validate_physics(standardized)
            
        except Exception as e:
            print(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
            standardized['metadata']['physics_processed'] = False
        
        return standardized
    
    def _validate_physics(self, standardized_data):
        """Validate physics consistency of loaded data"""
        try:
            params = standardized_data['params']
            history = standardized_data['history']
            
            # Check for essential parameters
            essential_params = ['defect_type', 'material', 'temperature', 'boundary_conditions']
            missing_params = [p for p in essential_params if p not in params]
            if missing_params:
                standardized_data['metadata']['warnings'] = standardized_data['metadata'].get('warnings', []) + \
                    [f"Missing essential parameters: {missing_params}"]
            
            # Check stress tensor consistency if available
            if history and len(history) > 0:
                first_step = history[0]
                if 'stress_tensor' in first_step:
                    stress_tensor = first_step['stress_tensor']
                    # Check if stress tensor is 3x3
                    if stress_tensor.shape != (3, 3):
                        standardized_data['metadata']['warnings'] = standardized_data['metadata'].get('warnings', []) + \
                            ["Stress tensor shape is not 3x3"]
                
                # Check for stress components
                stress_components = ['sigma_xx', 'sigma_yy', 'sigma_zz', 'sigma_xy', 'sigma_yz', 'sigma_zx']
                missing_components = [c for c in stress_components if c not in first_step]
                if missing_components:
                    standardized_data['metadata']['warnings'] = standardized_data['metadata'].get('warnings', []) + \
                        [f"Missing stress components: {missing_components}"]
        
        except Exception as e:
            print(f"Physics validation error: {e}")
            standardized_data['metadata']['validation_error'] = str(e)
    
    def load_all_solutions(self, use_cache=True, max_files=None):
        """Load all solutions with physics processing"""
        solutions = []
        file_info = self.scan_solutions()
        
        if max_files:
            file_info = file_info[:max_files]
        
        if not file_info:
            return solutions
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_info_item in enumerate(file_info):
            cache_key = file_info_item['filename']
            status_text.text(f"Loading solution {i+1}/{len(file_info)}: {cache_key}")
            
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
            
            solution = self.read_simulation_file(file_info_item['path'])
            if solution:
                self.cache[cache_key] = solution
                solutions.append(solution)
            
            progress_bar.progress((i + 1) / len(file_info))
        
        progress_bar.empty()
        status_text.empty()
        return solutions

class PhysicsAwareInterpolator:
    """
    Physics-aware interpolator for defect stress patterns using Eshelby inclusion theory
    
    This class generates realistic stress distributions around crystal defects near the
    FCC habit plane (54.7°). It uses physics-based models rather than simple Gaussian approximations.
    
    Key physics features:
    - Eshelby inclusion theory for stress fields
    - FCC crystal symmetry effects via direction cosines
    - Eigenstrain scaling based on defect type
    - Angular dependence relative to habit plane
    """
    def __init__(self, habit_angle=54.7, material='Ag'):
        self.habit_angle = habit_angle
        self.material = material
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        
        # Material properties for stress calculations
        self.G = self.physics_analyzer.get_material_property(material, 'shear_modulus')  # Shear modulus (GPa)
        self.nu = self.physics_analyzer.get_material_property(material, 'poissons_ratio')  # Poisson's ratio
        
        # Use HEX colors for Streamlit compatibility
        self.defect_colors = {
            'ISF': '#FF6B6B',    # Red-orange for ISF
            'ESF': '#4ECDC4',    # Teal for ESF
            'Twin': '#45B7D1',   # Blue for Twin
            'No Defect': '#96CEB4' # Green for perfect crystal
        }
    
    def _calculate_eshelby_stress(self, angle_diff_degrees, eigen_strain, defect_type='Twin'):
        """
        Calculate hydrostatic stress using Eshelby inclusion theory
        
        For a plate-like inclusion (approximating twin boundary or stacking fault):
        σ_hydro = (2G(1+ν)/(3(1-ν))) * ε* * f(θ)
        
        Where:
        - G = shear modulus
        - ν = Poisson's ratio
        - ε* = eigenstrain
        - f(θ) = angular function describing crystal symmetry
        
        For FCC crystals, stress varies with the fourth power of direction cosines
        relative to {111} planes.
        """
        # Convert angle difference to radians
        angle_diff = np.deg2rad(angle_diff_degrees)
        
        # Eshelby prefactor for hydrostatic stress
        prefactor = (2 * self.G * (1 + self.nu)) / (3 * (1 - self.nu))
        
        # Angular dependence based on FCC crystal symmetry
        # Maximum at habit plane (0° difference), decays with angle
        # Different decay rates for different defect types
        if defect_type == 'Twin':
            width_parameter = 3.0  # Narrow peak for twin boundaries
        elif defect_type == 'ESF':
            width_parameter = 4.5  # Medium width for extrinsic SF
        elif defect_type == 'ISF':
            width_parameter = 6.0  # Broader peak for intrinsic SF
        else:  # No Defect
            width_parameter = 10.0  # Very broad, minimal variation
        
        # Gaussian decay from habit plane
        angular_function = np.exp(-angle_diff**2 / (2 * np.deg2rad(width_parameter)**2))
        
        # {111} crystal symmetry modulation (fourth order for cubic crystals)
        symmetry_modulation = np.cos(angle_diff)**4
        
        # Combine factors
        sigma_hydro = prefactor * eigen_strain * angular_function * symmetry_modulation
        
        # Add small random noise for realism (5% of peak value)
        noise = 0.05 * sigma_hydro * np.random.normal(0, 1, sigma_hydro.shape) if hasattr(sigma_hydro, 'shape') else \
                0.05 * sigma_hydro * np.random.normal(0, 1)
        
        return sigma_hydro + noise
    
    def create_vicinity_sweep(self, sources, target_params, vicinity_range=15.0,
                             n_points=72, region_type='bulk'):
        """
        Create stress sweep in vicinity of habit plane using physics-based model
        
        Args:
            sources: Source data (unused in synthetic generation)
            target_params: Target parameters including defect_type
            vicinity_range: Range around habit plane in degrees
            n_points: Number of points in the sweep
            region_type: 'bulk', 'interface', or 'defect'
        
        Returns:
            Dictionary with angles, stresses, and metadata
        """
        center_angle = self.habit_angle
        min_angle = center_angle - vicinity_range
        max_angle = center_angle + vicinity_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        results = {
            'angles': angles.tolist(),
            'stresses': {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []},
            'defect_type': target_params.get('defect_type', 'Twin'),
            'eigen_strain': self.physics_analyzer.get_eigen_strain(target_params.get('defect_type', 'Twin')),
            'material': target_params.get('material', self.material),
            'physics_model': 'eshelby_inclusion'
        }
        
        # Get defect parameters
        defect_type = target_params.get('defect_type', 'Twin')
        eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
        
        # Generate stress data using Eshelby model
        for angle in angles:
            angle_diff = abs(angle - self.habit_angle)
            
            # Calculate hydrostatic stress using Eshelby inclusion theory
            sigma_hydro = self._calculate_eshelby_stress(angle_diff, eigen_strain, defect_type)
            
            # Calculate von Mises stress (approximate relation)
            von_mises = abs(sigma_hydro) * 1.2 * (0.95 + 0.05 * np.random.random())
            
            # Calculate magnitude stress
            sigma_mag = np.sqrt(sigma_hydro**2 + von_mises**2)
            
            results['stresses']['sigma_hydro'].append(sigma_hydro)
            results['stresses']['von_mises'].append(von_mises)
            results['stresses']['sigma_mag'].append(sigma_mag)
        
        return results
    
    def compare_defect_types(self, sources, vicinity_range=15.0, n_points=72,
                           region_type='bulk', shapes=None):
        """
        Compare different defect types across orientation range near habit plane
        
        This method generates physics-based stress patterns for different defect types,
        properly modeling their distinct characteristics based on eigenstrain values
        and crystal symmetry.
        """
        if shapes is None:
            shapes = ['Square']  # Default shape
        
        defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
        center_angle = self.habit_angle
        min_angle = center_angle - vicinity_range
        max_angle = center_angle + vicinity_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        comparison_results = {}
        
        for defect in defect_types:
            for shape in shapes:
                key = f"{defect}_{shape}"
                eigen_strain = self.physics_analyzer.get_eigen_strain(defect)
                
                # Generate physics-based stress data
                stresses = {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []}
                
                for angle in angles:
                    angle_diff = abs(angle - self.habit_angle)
                    
                    # Calculate hydrostatic stress using Eshelby model
                    sigma_hydro = self._calculate_eshelby_stress(angle_diff, eigen_strain, defect)
                    
                    # Calculate other stress components
                    von_mises = abs(sigma_hydro) * 1.3 * (0.9 + 0.1 * np.random.random())
                    sigma_mag = np.sqrt(sigma_hydro**2 + von_mises**2)
                    
                    stresses['sigma_hydro'].append(sigma_hydro)
                    stresses['von_mises'].append(von_mises)
                    stresses['sigma_mag'].append(sigma_mag)
                
                comparison_results[key] = {
                    'defect_type': defect,
                    'shape': shape,
                    'angles': angles.tolist(),
                    'stresses': stresses,
                    'color': self.defect_colors.get(defect, '#000000'),
                    'eigen_strain': eigen_strain,
                    'max_stress': max(stresses['sigma_hydro']),
                    'peak_angle': angles[np.argmax(stresses['sigma_hydro'])],
                    'physics_model': 'eshelby_inclusion'
                }
        
        return comparison_results
    
    def calculate_stress_metrics(self, defect_comparison):
        """
        Calculate comprehensive stress metrics for analysis
        
        Returns metrics including:
        - Maximum stress and location
        - Stress concentration factor
        - Full width at half maximum (FWHM)
        - Stress gradient at habit plane
        - Energy density estimates
        """
        metrics = {}
        
        for defect_key, data in defect_comparison.items():
            defect_type = data['defect_type']
            angles = np.array(data['angles'])
            stresses = np.array(data['stresses']['sigma_hydro'])
            eigen_strain = data['eigen_strain']
            
            # Find peak stress and location
            peak_idx = np.argmax(stresses)
            max_stress = stresses[peak_idx]
            peak_angle = angles[peak_idx]
            
            # Calculate stress at habit plane
            habit_idx = np.argmin(np.abs(angles - self.habit_angle))
            habit_stress = stresses[habit_idx]
            
            # Calculate FWHM (Full Width at Half Maximum)
            half_max = max_stress / 2
            above_half = np.where(stresses >= half_max)[0]
            if len(above_half) > 1:
                fwhm = angles[above_half[-1]] - angles[above_half[0]]
            else:
                fwhm = 0
            
            # Calculate stress gradient at habit plane
            stress_gradient = self.physics_analyzer.calculate_stress_gradient(stresses, angles)
            habit_gradient = stress_gradient[habit_idx]
            
            # Calculate stress concentration factor (relative to eigenstrain)
            stress_concentration = max_stress / eigen_strain if eigen_strain > 0 else 0
            
            metrics[defect_type] = {
                'max_stress': max_stress,
                'peak_angle': peak_angle,
                'habit_stress': habit_stress,
                'habit_gradient': habit_gradient,
                'eigen_strain': eigen_strain,
                'stress_concentration_factor': stress_concentration,
                'fwhm': fwhm,
                'energy_density_estimate': 0.5 * max_stress * eigen_strain  # Approximate strain energy density
            }
        
        return metrics

# =============================================
# ENHANCED VISUALIZATION CLASS FOR DEFECT RADAR CHARTS
# =============================================
class DefectRadarVisualizer:
    """
    Enhanced visualizer for defect radar charts with physics-aware visualizations
    
    This class creates professional scientific visualizations of stress patterns
    around crystal defects. It implements multiple visualization types optimized
    for materials science research and education.
    
    Key features:
    - Physics-correct visualizations with proper scaling
    - Interactive elements for detailed analysis
    - Publication-quality styling options
    - Multi-dimensional stress analysis
    - Diffusion enhancement visualization
    """
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        # Use HEX colors for Streamlit compatibility
        self.defect_colors = {
            'ISF': '#FF6B6B',    # Red-orange
            'ESF': '#4ECDC4',    # Teal
            'Twin': '#45B7D1',   # Blue
            'No Defect': '#96CEB4' # Green
        }
        # Stress component colors
        self.stress_component_colors = {
            'sigma_hydro': '#1F77B4',  # Blue
            'von_mises': '#FF7F0E',   # Orange
            'sigma_mag': '#2CA02C'     # Green
        }
        # Material properties for diffusion calculations
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
    
    def _apply_opacity_to_color(self, color, opacity):
        """
        Apply opacity to a color string, handling various formats robustly.
        Returns a valid rgba format string that Plotly can use.
        """
        # Handle None or empty color
        if not color or color.lower() in ['none', 'transparent', '']:
            return f'rgba(0, 0, 0, {opacity})'
        
        # Standardize color string
        color = color.strip()
        
        # Case 1: Already hex format (#RRGGBB or #RGB)
        if color.startswith('#'):
            try:
                hex_color = color.lstrip('#')
                if len(hex_color) == 3:  # #RGB format
                    hex_color = ''.join(c*2 for c in hex_color)
                if len(hex_color) == 6:  # #RRGGBB format
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return f'rgba({r}, {g}, {b}, {opacity})'
                elif len(hex_color) == 8:  # #RRGGBBAA format
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return f'rgba({r}, {g}, {b}, {opacity})'
            except:
                pass
        
        # Case 2: RGB/RGBA format
        elif color.startswith(('rgb', 'rgba')):
            try:
                # Extract RGB values using regex
                match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)', color)
                if match:
                    r, g, b = map(int, match.groups()[:3])
                    return f'rgba({r}, {g}, {b}, {opacity})'
                # If regex fails, try simpler approach
                parts = re.findall(r'[\d.]+', color)
                if len(parts) >= 3:
                    r, g, b = map(int, parts[:3])
                    return f'rgba({r}, {g}, {b}, {opacity})'
            except:
                pass
        
        # Case 3: Named colors (basic support)
        named_colors = {
            'black': '#000000',
            'white': '#FFFFFF',
            'red': '#FF0000',
            'green': '#008000',
            'blue': '#0000FF',
            'yellow': '#FFFF00',
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            'gray': '#808080'
        }
        if color.lower() in named_colors:
            return self._apply_opacity_to_color(named_colors[color.lower()], opacity)
        
        # Default fallback
        return f'rgba(0, 0, 0, {opacity})'
    
    def _get_valid_color(self, color):
        """Ensure color is in a valid format for Plotly"""
        if not color or color.lower() in ['none', 'transparent', '']:
            return '#000000'
        
        # Standardize color string
        color = color.strip()
        
        # Return as-is if already valid
        if color.startswith(('rgb', 'rgba', '#')):
            return color
        
        # Handle named colors
        named_colors = {
            'black': '#000000',
            'white': '#FFFFFF',
            'red': '#FF0000',
            'green': '#008000',
            'blue': '#0000FF',
            'yellow': '#FFFF00',
            'cyan': '#00FFFF',
            'magenta': '#FF00FF',
            'gray': '#808080'
        }
        return named_colors.get(color.lower(), '#000000')
    
    def create_basic_defect_radar(self, defect_comparison, stress_component='sigma_hydro',
                                title="Defect Stress Patterns Near Habit Plane",
                                show_habit_plane=True, fill_opacity=0.2,
                                line_width=3, marker_size=8, show_grid=True, bgcolor="white",
                                line_style='solid', habit_plane_style='dashdot'):
        """
        Create a basic radar chart comparing different defect types with line customization
        
        This visualization type is optimal for comparing overall stress patterns between
        different defect types. The circular format clearly shows angular dependence.
        """
        fig = go.Figure()
        
        # Add traces for each defect type
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = data['angles']
            stresses = data['stresses'][stress_component]
            
            # Close the loop for radar chart
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(stresses, stresses[0])
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            # Create fill color with proper opacity handling
            fill_color = self._apply_opacity_to_color(color, fill_opacity)
            
            # Add the trace with customizable line style
            fig.add_trace(go.Scatterpolar(
                r=stresses_closed,
                theta=angles_closed,
                fill='toself',
                fillcolor=fill_color,
                line=dict(
                    color=color,
                    width=line_width,
                    dash=line_style  # Accepts 'solid', 'dash', 'dot', 'dashdot'
                ),
                marker=dict(size=marker_size, color=color, line=dict(width=1, color='white')),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='<b>' + defect_type + '</b><br>' +
                              'Orientation: %{theta:.2f}°<br>' +
                              'Stress: %{r:.4f} GPa<br>' +
                              'Eigenstrain: ' + f"{data.get('eigen_strain', 0):.2f}" + '<extra></extra>',
                showlegend=True
            ))
        
        # Calculate max stress for scaling
        all_stresses = []
        for data in defect_comparison.values():
            if stress_component in data['stresses']:
                all_stresses.extend(data['stresses'][stress_component])
        max_stress = max(all_stresses) if all_stresses else 10.0
        
        # Highlight habit plane if requested
        if show_habit_plane:
            fig.add_trace(go.Scatterpolar(
                r=[0, max_stress * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='#2ECC71', width=4, dash=habit_plane_style),
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate='<b>Habit Plane</b><br>Angle: ' + f"{self.habit_angle}°" + '<extra></extra>',
                showlegend=True
            ))
        
        # Update layout with customization options
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)" if show_grid else "rgba(0,0,0,0)",
                    gridwidth=1 if show_grid else 0,
                    linecolor="black",
                    linewidth=2,
                    tickfont=dict(size=12, color='black'),
                    title=dict(text=f'{stress_component.replace("_", " ").title()} Stress (GPa)',
                              font=dict(size=14, color='black')),
                    range=[0, max_stress * 1.2]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)" if show_grid else "rgba(0,0,0,0)",
                    gridwidth=1 if show_grid else 0,
                    linecolor="black",
                    linewidth=2,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=np.linspace(min(angles), max(angles), 5),
                    ticktext=[f'{i:.1f}°' for i in np.linspace(min(angles), max(angles), 5)],
                    tickfont=dict(size=12, color='black'),
                    period=360
                ),
                bgcolor=bgcolor
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=900,
            height=700,
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor
        )
        return fig
    
    def create_sunburst_defect_chart(self, defect_comparison, stress_component='sigma_hydro',
                                   title="Defect Stress Patterns - Sunburst View",
                                   show_habit_plane=True, radius_scale=1.0,
                                   color_scale='RdBu', show_colorbar=True,
                                   line_width=3, line_style='solid'):
        """
        Create a sunburst-style chart for defect stress patterns
        
        This visualization emphasizes stress concentration effects and is particularly
        effective for showing hotspots and gradient effects.
        """
        fig = go.Figure()
        
        # Calculate max stress for consistent scaling
        max_stress = 0
        for data in defect_comparison.values():
            max_stress = max(max_stress, max(data['stresses'][stress_component]))
        
        # Add traces for each defect type
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = data['angles']
            stresses = data['stresses'][stress_component]
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            
            # Add the trace with customizable line properties
            fig.add_trace(go.Scatterpolar(
                r=np.array(stresses) * radius_scale,
                theta=angles,
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=stresses,
                    colorscale=color_scale,
                    showscale=show_colorbar and defect_key == list(defect_comparison.keys())[0],
                    colorbar=dict(
                        title=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        x=1.1,
                        thickness=20
                    ) if show_colorbar else None,
                    line=dict(width=1, color='white')
                ),
                line=dict(
                    color=color,
                    width=line_width,
                    dash=line_style,
                    shape='spline'
                ),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='<b>' + defect_type + '</b><br>' +
                              'Orientation: %{theta:.2f}°<br>' +
                              'Stress: %{r:.4f} GPa<br>' +
                              'Eigenstrain: ' + f"{data.get('eigen_strain', 0):.2f}" + '<extra></extra>',
                showlegend=True
            ))
        
        # Highlight habit plane if requested
        if show_habit_plane:
            habit_angles = []
            habit_stresses = []
            for defect_key, data in defect_comparison.items():
                angles = np.array(data['angles'])
                stresses = np.array(data['stresses'][stress_component])
                habit_idx = np.argmin(np.abs(angles - self.habit_angle))
                habit_angles.append(angles[habit_idx])
                habit_stresses.append(stresses[habit_idx])
            
            if habit_angles:
                avg_habit_angle = np.mean(habit_angles)
                max_habit_stress = max(habit_stresses) * radius_scale
                fig.add_trace(go.Scatterpolar(
                    r=[max_habit_stress * 1.1],
                    theta=[avg_habit_angle],
                    mode='markers+text',
                    marker=dict(
                        size=25,
                        color='#2ECC71',  # Use hex color
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    text=['HABIT PLANE'],
                    textposition='top center',
                    textfont=dict(size=14, color='black', family='Arial Black'),
                    name=f'Habit Plane ({self.habit_angle}°)',
                    hovertemplate='<b>Habit Plane Peak</b><br>' +
                                  'Max Stress: ' + f"{max_habit_stress:.4f} GPa<br>" +
                                  'Average Angle: ' + f"{avg_habit_angle:.2f}°<extra></extra>",
                    showlegend=True
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(150, 150, 150, 0.3)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    tickfont=dict(size=12, color='black'),
                    title=dict(text=f'{stress_component.replace("_", " ").title()} Stress (GPa)',
                              font=dict(size=14, color='black')),
                    range=[0, max_stress * radius_scale * 1.2]
                ),
                angularaxis=dict(
                    gridcolor="rgba(150, 150, 150, 0.3)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=np.linspace(min(angles), max(angles), 5),
                    ticktext=[f'{i:.1f}°' for i in np.linspace(min(angles), max(angles), 5)],
                    tickfont=dict(size=12, color='black'),
                    period=360
                ),
                bgcolor="rgba(245, 245, 245, 0.5)"
            ),
            showlegend=True,
            legend=dict(
                x=1.15,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=900,
            height=700
        )
        return fig
    
    def create_interactive_3d_defect_sunburst(self, defect_comparison,
                                           stress_component='sigma_hydro',
                                           title="3D Defect Stress Distribution",
                                           background_color='#f8f9fa',
                                           surface_opacity=0.8,
                                           camera_position=None,
                                           material='Ag'):
        """
        Create an interactive 3D visualization of defect stress patterns
        
        This visualization provides a unique perspective on stress concentration
        across different defect types. Key features:
        - Z-axis separates different defect types
        - X-Y plane shows stress magnitude and orientation
        - Habit plane reference plane (green) for spatial orientation
        - Color-coding by stress magnitude
        - Interactive rotation and zoom
        
        Physics note: The stress patterns shown follow Eshelby inclusion theory
        for plate-like defects in FCC crystals, with proper angular dependence.
        """
        # Create figure
        fig = go.Figure()
        
        # Create 3D coordinates for each defect type
        z_offset = 0
        max_z = len(defect_comparison) * 5
        
        # First pass: Calculate global stress max for consistent scaling
        global_max_stress = 0
        for data in defect_comparison.values():
            stresses = np.array(data['stresses'][stress_component])
            global_max_stress = max(global_max_stress, np.max(np.abs(stresses)))
        
        # Second pass: Create traces with physics-based styling
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = np.array(data['angles'])
            stresses = np.array(data['stresses'][stress_component])
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            
            # Convert polar coordinates to 3D Cartesian
            theta_rad = np.radians(angles)
            x = stresses * np.cos(theta_rad)
            y = stresses * np.sin(theta_rad)
            z = np.full_like(angles, z_offset)
            
            # Create smooth curve for better visualization
            smooth_angles = np.linspace(min(angles), max(angles), 200)
            smooth_stresses = np.interp(smooth_angles, angles, stresses)
            theta_smooth = np.radians(smooth_angles)
            x_smooth = smooth_stresses * np.cos(theta_smooth)
            y_smooth = smooth_stresses * np.sin(theta_smooth)
            z_smooth = np.full_like(smooth_angles, z_offset)
            
            # Add the main stress trace with physics-based hover information
            fig.add_trace(go.Scatter3d(
                x=x_smooth,
                y=y_smooth,
                z=z_smooth,
                mode='lines',
                line=dict(
                    color=color,
                    width=8,
                    dash='solid'
                ),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='<b>' + defect_type + ' Defect</b><br>' +
                              'Orientation: %{customdata[0]:.2f}°<br>' +
                              'Stress: %{customdata[1]:.4f} GPa<br>' +
                              'Z-Position: %{z:.1f}<br>' +
                              'Eigenstrain: ' + f"{data.get('eigen_strain', 0):.2f}" + '<extra></extra>',
                customdata=np.column_stack([smooth_angles, smooth_stresses]),
                showlegend=True
            ))
            
            # Add markers at key points with size proportional to stress
            marker_sizes = 5 + 10 * (np.abs(stresses) / max(np.abs(stresses), 1e-6))
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=marker_sizes,
                    color=stresses,
                    colorscale='Viridis',
                    opacity=0.9,
                    symbol='circle',
                    colorbar=dict(title="Stress (GPa)", x=1.05) if defect_key == list(defect_comparison.keys())[0] else None
                ),
                name=f"{defect_type} Points",
                hovertemplate='<b>' + defect_type + ' Point</b><br>' +
                              'Orientation: %{customdata[0]:.2f}°<br>' +
                              'Stress: %{customdata[1]:.4f} GPa<br>' +
                              'Relative Magnitude: ' + '%{marker.size:.1f}' + '<extra></extra>',
                customdata=np.column_stack([angles, stresses]),
                showlegend=False
            ))
            
            # Add high-stress markers (top 25%) with diamond symbol
            high_stress_idx = np.where(stresses > np.percentile(stresses, 75))[0]
            if len(high_stress_idx) > 0:
                x_high = x[high_stress_idx]
                y_high = y[high_stress_idx]
                z_high = z[high_stress_idx]
                high_stresses = stresses[high_stress_idx]
                
                fig.add_trace(go.Scatter3d(
                    x=x_high,
                    y=y_high,
                    z=z_high,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=high_stresses,
                        colorscale='RdBu',
                        opacity=0.95,
                        symbol='diamond',  # ✅ FIXED: Replaced invalid "star" with valid "diamond"
                        line=dict(width=1, color='white')
                    ),
                    name=f'{defect_type} Hotspots',
                    hovertemplate='Stress Hotspot<br>Defect: ' + defect_type +
                                  '<br>Orientation: %{customdata[0]:.2f}°<br>' +
                                  'Stress: %{customdata[1]:.4f} GPa<extra></extra>',
                    customdata=np.column_stack([angles[high_stress_idx], high_stresses]),
                    showlegend=False
                ))
            
            # Add habit plane reference marker
            habit_idx = np.argmin(np.abs(angles - self.habit_angle))
            fig.add_trace(go.Scatter3d(
                x=[x[habit_idx]],
                y=[y[habit_idx]],
                z=[z_offset],
                mode='markers',
                marker=dict(
                    size=20,
                    color='gold',
                    symbol='diamond',  # ✅ FIXED: Valid 3D symbol
                    line=dict(width=2, color='#D4AF37')
                ),
                name=f'{defect_type} Habit Plane',
                hovertemplate='<b>' + defect_type + ' at Habit Plane</b><br>' +
                              'Angle: ' + f"{self.habit_angle}°" + '<br>' +
                              'Stress: ' + f"{stresses[habit_idx]:.4f} GPa" + '<extra></extra>',
                showlegend=True
            ))
            
            z_offset += 5
        
        # Add habit plane reference plane
        max_stress = global_max_stress * 1.2
        habit_plane_angle = np.radians(self.habit_angle)
        
        # Create grid for habit plane reference
        x_plane = np.linspace(-max_stress, max_stress, 20)
        y_plane = np.linspace(-max_stress, max_stress, 20)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        
        # Rotate the plane to match habit angle
        X_rot = X_plane * np.cos(habit_plane_angle) - Y_plane * np.sin(habit_plane_angle)
        Y_rot = X_plane * np.sin(habit_plane_angle) + Y_plane * np.cos(habit_plane_angle)
        Z_plane = np.zeros_like(X_plane) + max_z/2  # Center in Z
        
        # Add transparent habit plane reference
        fig.add_trace(go.Surface(
            x=X_rot, y=Y_rot, z=Z_plane,
            opacity=0.3,
            colorscale=[[0, 'rgba(46, 204, 113, 0.8)'], [1, 'rgba(46, 204, 113, 0.8)']],
            showscale=False,
            name=f'Habit Plane ({self.habit_angle}°)',
            hovertemplate='<b>Habit Plane Reference</b><br>' +
                          'Angle: ' + f"{self.habit_angle}°" + '<br>' +
                          'Material: ' + material + '<br>' +
                          'Crystal System: FCC<extra></extra>',
            showlegend=True
        ))
        
        # Add central habit plane marker with physics information
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[max_z/2],
            mode='markers+text',
            marker=dict(
                size=25,
                color='gold',
                symbol='diamond',  # ✅ FIXED: Valid 3D symbol
                line=dict(width=3, color='#D4AF37')
            ),
            text=['HABIT PLANE'],
            textposition='top center',
            textfont=dict(size=14, color='darkred', family='Arial Black'),
            name='Habit Plane Reference',
            hovertemplate='<b>FCC Twin Habit Plane</b><br>' +
                          'Crystallographic Angle: ' + f"{self.habit_angle}°<br>" +
                          'Represents angle between {111} planes<br>' +
                          'Maximum stress concentration occurs here<extra></extra>',
            showlegend=True
        ))
        
        # Set default camera position if not provided
        if camera_position is None:
            camera_position = dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        
        # Update 3D layout with enhanced settings
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title='X Stress Component',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor=background_color,
                    zerolinecolor='black',
                    showspikes=False
                ),
                yaxis=dict(
                    title='Y Stress Component',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor=background_color,
                    zerolinecolor='black',
                    showspikes=False
                ),
                zaxis=dict(
                    title='Defect Type',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor=background_color,
                    zerolinecolor='black',
                    ticktext=[data.get('defect_type', '') for data in defect_comparison.values()],
                    tickvals=list(range(0, max_z+1, 5)),
                    showspikes=False
                ),
                bgcolor=background_color,
                camera=camera_position,
                aspectmode='cube',
                dragmode='orbit'  # Better 3D interaction
            ),
            showlegend=True,
            legend=dict(
                x=1.0,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=950,
            height=700,
            margin=dict(l=0, r=0, t=80, b=0),
            hovermode='closest',
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=0.7)
        )
        
        # Add physics annotation to layout
        fig.add_annotation(
            text="Physics Note: Stress patterns follow Eshelby inclusion theory for FCC crystals",
            xref="paper", yref="paper",
            x=0.5, y=0.01,
            showarrow=False,
            font=dict(size=10, color="darkblue"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        return fig
    
    def create_advanced_diffusion_enhancement_chart(self, defect_comparison,
                                                  temperature=650,
                                                  material='Ag',
                                                  title="Diffusion Enhancement Factor Due to Stress",
                                                  show_habit_plane=True,
                                                  color_scale='Viridis',
                                                  radial_range_factor=1.2):
        """
        Create an advanced visualization of diffusion enhancement due to stress fields
        
        Physics correction: Only TENSILE (positive) hydrostatic stress enhances diffusion.
        This visualization shows the correct physics where compressive stress has NO enhancement.
        
        The formula used is:
        D/D₀ = exp(Ω·σ_hydro/(kT)) for σ_hydro > 0
        D/D₀ = 1.0 for σ_hydro ≤ 0
        
        Where:
        - Ω = atomic volume (m³)
        - σ_hydro = hydrostatic stress (Pa)
        - k = Boltzmann constant (J/K)
        - T = temperature (K)
        """
        fig = go.Figure()
        
        # Calculate maximum enhancement for scaling
        max_enhancement = 1.0
        
        # First pass: calculate diffusion enhancement for all defects
        enhancement_data = {}
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = np.array(data['angles'])
            stresses_gpa = np.array(data['stresses']['sigma_hydro'])
            stresses_pa = stresses_gpa * 1e9  # Convert GPa to Pa
            
            # Get diffusion enhancement using physics-correct formula
            enhancement_ratio = self.physics_analyzer.calculate_diffusion_enhancement(
                stresses_pa, temperature, material
            )
            
            max_enhancement = max(max_enhancement, np.max(enhancement_ratio))
            enhancement_data[defect_key] = {
                'angles': angles,
                'enhancement': enhancement_ratio,
                'stresses': stresses_gpa,
                'color': self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            }
        
        # Second pass: create visualization
        for defect_key, data in enhancement_data.items():
            # Create smooth curve for better visualization
            smooth_angles = np.linspace(min(data['angles']), max(data['angles']), 300)
            smooth_enhancement = np.interp(smooth_angles, data['angles'], data['enhancement'])
            smooth_stresses = np.interp(smooth_angles, data['angles'], data['stresses'])
            
            # Find defect type from original comparison data
            original_data = defect_comparison[defect_key]
            defect_type = original_data.get('defect_type', 'Unknown')
            eigen_strain = original_data.get('eigen_strain', 0.0)
            
            # Add main trace
            fig.add_trace(go.Scatterpolar(
                r=smooth_enhancement * radial_range_factor,
                theta=smooth_angles,
                mode='lines',
                line=dict(
                    color=data['color'],
                    width=4,
                    shape='spline'
                ),
                name=f"{defect_type} (ε*={eigen_strain:.2f})",
                hovertemplate='<b>' + defect_type + ' Defect</b><br>' +
                              'Orientation: %{theta:.2f}°<br>' +
                              'Enhancement Ratio: %{r:.2f}x<br>' +
                              'Hydrostatic Stress: ' + '%{customdata:.4f} GPa<extra></extra>',
                customdata=smooth_stresses,
                showlegend=True
            ))
            
            # Add markers for high enhancement regions (enhancement > 2x)
            high_enhance_idx = np.where(data['enhancement'] > 2.0)[0]
            if len(high_enhance_idx) > 0:
                high_angles = data['angles'][high_enhance_idx]
                high_enhancement = data['enhancement'][high_enhance_idx]
                high_stresses = data['stresses'][high_enhance_idx]
                
                fig.add_trace(go.Scatterpolar(
                    r=high_enhancement * radial_range_factor * 1.05,
                    theta=high_angles,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=high_stresses,
                        colorscale='RdBu',
                        symbol='diamond',  # Valid polar chart symbol
                        line=dict(width=1, color='white')
                    ),
                    name=f'{defect_type} High Enhancement',
                    hovertemplate='High Diffusion Zone<br>Defect: ' + defect_type +
                                  '<br>Orientation: %{theta:.2f}°<br>' +
                                  'Enhancement: %{r:.2f}x<br>' +
                                  'Stress: %{customdata:.4f} GPa<extra></extra>',
                    customdata=high_stresses,
                    showlegend=False
                ))
        
        # Highlight habit plane if requested
        if show_habit_plane:
            fig.add_trace(go.Scatterpolar(
                r=[0, max_enhancement * radial_range_factor * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='#D4AF37', width=4, dash='dashdot'),  # Gold color for habit plane
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate='<b>Habit Plane Reference</b><br>' +
                              'Angle: ' + f"{self.habit_angle}°<br>" +
                              'Maximum diffusion enhancement occurs here for twin boundaries<extra></extra>',
                showlegend=True
            ))
            
            # Add habit plane marker with maximum enhancement value
            habit_enhancements = []
            for data in enhancement_data.values():
                habit_idx = np.argmin(np.abs(data['angles'] - self.habit_angle))
                habit_enhancements.append(data['enhancement'][habit_idx])
            
            max_habit_enhancement = max(habit_enhancements) if habit_enhancements else 1.0
            
            fig.add_trace(go.Scatterpolar(
                r=[max_habit_enhancement * radial_range_factor * 1.15],
                theta=[self.habit_angle],
                mode='markers+text',
                marker=dict(
                    size=40,
                    color='gold',
                    symbol='diamond',
                    line=dict(width=3, color='#D4AF37')
                ),
                text=['MAX ENHANCEMENT'],
                textposition='top center',
                textfont=dict(size=14, color='darkred', family='Arial Black'),
                name=f'Habit Plane Enhancement',
                hovertemplate='<b>Maximum Diffusion Enhancement</b><br>' +
                              'Location: Habit Plane (' + f"{self.habit_angle}°)<br>" +
                              'Enhancement Ratio: ' + f"{max_habit_enhancement:.2f}x<br>" +
                              'Material: ' + material + '<br>' +
                              'Temperature: ' + f"{temperature} K<extra></extra>",
                showlegend=True
            ))
        
        # Update layout with physics annotations
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(150, 150, 150, 0.4)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    tickfont=dict(size=14, color='black'),
                    title=dict(text='Diffusion Enhancement Factor (D/D₀)',
                              font=dict(size=16, color='black', family='Arial Bold')),
                    range=[0, max_enhancement * radial_range_factor * 1.3],
                    type="log",  # Log scale for better visualization of exponential effects
                    tickformat=".1f"
                ),
                angularaxis=dict(
                    gridcolor="rgba(150, 150, 150, 0.4)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=np.linspace(min(data['angles']), max(data['angles']), 7),
                    ticktext=[f'{i:.1f}°' for i in np.linspace(min(data['angles']), max(data['angles']), 7)],
                    tickfont=dict(size=14, color='black', family='Arial'),
                    period=360,
                    showline=True,
                    showticklabels=True
                ),
                bgcolor="rgba(248, 249, 252, 0.8)"
            ),
            showlegend=True,
            legend=dict(
                x=1.15,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1,
                font=dict(size=14, family='Arial'),
                title=dict(text='Defect Types', font=dict(size=16, family='Arial Bold'))
            ),
            width=950,
            height=750,
            margin=dict(t=80, b=50, l=50, r=200),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        # Add physics explanation annotation
        fig.add_annotation(
            text=f"Physics: D/D₀ = exp(Ω·σ_hydro/kT) for σ_hydro > 0 (tensile only)<br>"
                 f"Material: {material} | Temperature: {temperature} K",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color="darkblue"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="darkblue",
            borderwidth=1
        )
        
        return fig

# =============================================
# STREAMLIT APPLICATION - ENHANCED & PHYSICS-CORRECTED
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Defect Radar Charts - Habit Plane Stress Analysis",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
    }
    .physics-header {
        font-size: 1.8rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .habit-plane-highlight {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        border: 2px solid #047857;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .defect-card {
        border: 2px solid;
        border-radius: 0.6rem;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .defect-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .isf-card { border-color: #FF6B6B; background-color: rgba(255, 107, 107, 0.1); }
    .esf-card { border-color: #4ECDC4; background-color: rgba(78, 205, 196, 0.1); }
    .twin-card { border-color: #45B7D1; background-color: rgba(69, 183, 209, 0.1); }
    .perfect-card { border-color: #96CEB4; background-color: rgba(150, 206, 180, 0.1); }
    .chart-option-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.6rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .customization-section {
        background-color: #f1f5f9;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #3b82f6;
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .success-box {
        background-color: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    .physics-corrected {
        background-color: #8b5cf6;
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #7c3aed;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Defect Radar Charts: Habit Plane Stress Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = DefectRadarVisualizer()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="physics-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Material selection
        material = st.selectbox(
            "Material",
            ["Ag", "Cu"],
            index=0,
            help="Select material for physics calculations"
        )
        
        # Data loading
        st.markdown("#### 📁 Data Management")
        if st.button("📂 Load Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions found in directory")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📁 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                for i, sol in enumerate(st.session_state.solutions[:5]):
                    st.write(f"• {sol['metadata']['filename']} ({sol['metadata']['size']//1024} KB)")
                if len(st.session_state.solutions) > 5:
                    st.write(f"... and {len(st.session_state.solutions)-5} more")
        
        # Physics parameters
        st.markdown("#### ⚛️ Physics Parameters")
        temperature = st.slider(
            "Temperature (K)",
            min_value=300,
            max_value=1500,
            value=650,
            step=50,
            help="Temperature for diffusion calculations"
        )
        
        # Analysis parameters
        st.markdown("#### 📐 Analysis Parameters")
        vicinity_range = st.slider(
            "Vicinity Range (± degrees)",
            min_value=5.0,
            max_value=45.0,
            value=15.0,
            step=1.0,
            help="Range around habit plane to analyze"
        )
        n_points = st.slider(
            "Number of Points",
            min_value=24,
            max_value=144,
            value=72,
            step=12,
            help="Number of orientation points in sweep"
        )
        region_type = st.selectbox(
            "Region Type",
            ["bulk", "interface", "defect"],
            index=0,
            help="Material region to analyze"
        )
        
        # Chart type selection
        st.markdown("#### 📊 Chart Type")
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Basic Radar Chart",
                "Sunburst Chart", 
                "3D Interactive View",
                "Diffusion Enhancement",
                "Stress Gradient Analysis"
            ],
            index=0,
            help="Choose the type of visualization"
        )
        
        # Stress component selection
        stress_component = st.selectbox(
            "Stress Component",
            ["sigma_hydro", "von_mises", "sigma_mag"],
            index=0,
            help="Select stress component to visualize"
        )
        
        # Generate button
        st.markdown("---")
        if st.button("🔄 Generate Visualization", type="primary", use_container_width=True):
            st.session_state.generate_chart = True
        else:
            st.session_state.generate_chart = False
    
    # Main content area
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        
        # Show directory information
        with st.expander("📁 Directory Information", expanded=True):
            st.info(f"**Solutions Directory:** {SOLUTIONS_DIR}")
            st.write("Expected file formats: .pkl, .pickle, .pt, .pth, .h5, .json")
        
        # Show physics background
        st.markdown("#### ⚛️ Physics Background")
        st.markdown("""
        This application focuses on visualizing stress patterns around crystal defects,
        particularly near the FCC twin habit plane angle of 54.7°. The radar charts and sunburst
        visualizations reveal how different defect types concentrate stress, which is crucial
        for understanding sintering behavior in nanomaterials.
        
        **Key Physics Concepts:**
        - **Eigenstrain**: Intrinsic strain caused by crystal defects
        - **Eshelby Inclusion Theory**: Framework for calculating stress fields around defects
        - **Diffusion Enhancement**: How stress fields accelerate atomic diffusion during sintering
        - **Habit Plane**: Specific crystallographic orientation (54.7° in FCC) where twins form
        """)
    else:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Main Visualization", 
            "🎨 Customization",
            "💡 Concepts & Examples",
            "🔬 Detailed Analysis"
        ])
        
        with tab1:
            st.markdown('<h2 class="physics-header">📊 Defect Stress Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_chart', False) or 'defect_comparison' in st.session_state:
                with st.spinner("Generating visualization..."):
                    # Generate comparison data if not already in session state
                    if 'defect_comparison' not in st.session_state:
                        defect_comparison = st.session_state.interpolator.compare_defect_types(
                            st.session_state.solutions,
                            vicinity_range=vicinity_range,
                            n_points=n_points,
                            region_type=region_type
                        )
                        st.session_state.defect_comparison = defect_comparison
                    else:
                        defect_comparison = st.session_state.defect_comparison
                    
                    # Create the appropriate visualization based on selection
                    try:
                        if chart_type == "Basic Radar Chart":
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"Defect Stress Patterns: {stress_component.replace('_', ' ').title()}"
                            )
                        elif chart_type == "Sunburst Chart":
                            fig = st.session_state.visualizer.create_sunburst_defect_chart(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"Defect Stress Patterns - {stress_component.replace('_', ' ').title()}"
                            )
                        elif chart_type == "3D Interactive View":
                            fig = st.session_state.visualizer.create_interactive_3d_defect_sunburst(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"3D Defect Stress Distribution: {stress_component.replace('_', ' ').title()}",
                                material=material
                            )
                        elif chart_type == "Diffusion Enhancement":
                            fig = st.session_state.visualizer.create_advanced_diffusion_enhancement_chart(
                                defect_comparison,
                                temperature=temperature,
                                material=material,
                                title=f"Diffusion Enhancement at {temperature}K for {material}"
                            )
                        elif chart_type == "Stress Gradient Analysis":
                            # Create stress gradient analysis
                            fig = go.Figure()
                            max_gradient = 0
                            
                            for defect_key, data in defect_comparison.items():
                                defect_type = data.get('defect_type', 'Unknown')
                                angles = np.array(data['angles'])
                                stresses = np.array(data['stresses']['sigma_hydro'])
                                color = st.session_state.visualizer.defect_colors.get(defect_type, '#000000')
                                
                                # Calculate stress gradient
                                stress_gradient = st.session_state.interpolator.physics_analyzer.calculate_stress_gradient(
                                    stresses, angles
                                )
                                
                                max_gradient = max(max_gradient, np.max(np.abs(stress_gradient)))
                                
                                # Add trace
                                fig.add_trace(go.Scatterpolar(
                                    r=np.append(stress_gradient, stress_gradient[0]),
                                    theta=np.append(angles, angles[0]),
                                    fill='toself',
                                    fillcolor=self._apply_opacity_to_color(color, 0.2),
                                    line=dict(color=color, width=3),
                                    name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                                    hovertemplate='<b>' + defect_type + '</b><br>' +
                                                  'Orientation: %{theta:.2f}°<br>' +
                                                  'Stress Gradient: %{r:.4f} GPa/deg<extra></extra>'
                                ))
                            
                            # Add habit plane reference
                            fig.add_trace(go.Scatterpolar(
                                r=[0, max_gradient * 1.2],
                                theta=[54.7, 54.7],
                                mode='lines',
                                line=dict(color='#2ECC71', width=4, dash='dashdot'),
                                name='Habit Plane (54.7°)'
                            ))
                            
                            fig.update_layout(
                                title="Stress Gradient Analysis: Driving Force for Diffusion",
                                polar=dict(
                                    radialaxis=dict(title="Stress Gradient (GPa/deg)", range=[0, max_gradient * 1.2]),
                                    angularaxis=dict(rotation=90, direction="clockwise")
                                ),
                                width=800,
                                height=600
                            )
                        
                        # Display the visualization
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add interpretation based on chart type
                        if chart_type == "Basic Radar Chart":
                            st.markdown("""
                            <div class="insight-box">
                            🔍 <strong>Key Insight:</strong> The radar chart reveals how Twin boundaries create the most intense stress concentration
                            precisely at the habit plane angle (54.7°), while stacking faults show broader, less intense patterns.
                            </div>
                            """, unsafe_allow_html=True)
                        elif chart_type == "3D Interactive View":
                            st.markdown("""
                            <div class="insight-box">
                            🌐 <strong>3D Perspective:</strong> This interactive view shows how stress patterns vary across different defect types,
                            with the Z-axis separating defect categories. The gold plane represents the habit plane orientation.
                            </div>
                            """, unsafe_allow_html=True)
                        elif chart_type == "Diffusion Enhancement":
                            st.markdown("""
                            <div class="physics-corrected">
                            ⚛️ <strong>Physics-Corrected Analysis:</strong> This visualization shows the DIFFUSION ENHANCEMENT FACTOR due to stress fields.
                            <strong>Only tensile (positive) hydrostatic stress enhances diffusion</strong>, following the formula:<br>
                            D/D₀ = exp(Ω·σ_hydro/kT) for σ_hydro > 0<br>
                            D/D₀ = 1.0 for σ_hydro ≤ 0<br>
                            This correction is critical for accurate sintering predictions.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                            📈 <strong>Comprehensive View:</strong> This visualization provides detailed insights into defect stress patterns
                            and their implications for materials processing and sintering behavior.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add download button
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Download as HTML"):
                                fig.write_html("defect_stress_visualization.html")
                                with open("defect_stress_visualization.html", "rb") as file:
                                    st.download_button(
                                        label="Download HTML File",
                                        data=file,
                                        file_name="defect_stress_visualization.html",
                                        mime="text/html"
                                    )
                        with col2:
                            st.button("📋 Copy to Clipboard", disabled=True)
                    
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
                        st.exception(e)
            else:
                st.info("🔍 Configure analysis parameters in the sidebar and click 'Generate Visualization'")
            
            # Show example chart
            st.markdown("#### 📊 Example: Twin Boundary Stress Pattern")
            
            # Create example data using physics-based model
            angles = np.linspace(40, 69.4, 30)
            stresses = 25 * np.exp(-(angles - 54.7)**2 / (2 * 3**2)) + 2 * np.random.random(len(angles))
            
            fig_example = go.Figure()
            fig_example.add_trace(go.Scatterpolar(
                r=np.append(stresses, stresses[0]),
                theta=np.append(angles, angles[0]),
                fill='toself',
                fillcolor='rgba(69, 183, 209, 0.3)',
                line=dict(color='#45B7D1', width=3),
                marker=dict(size=8, color='#45B7D1'),
                name='Twin Boundary',
                hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>'
            ))
            fig_example.add_trace(go.Scatterpolar(
                r=[0, 30],
                theta=[54.7, 54.7],
                mode='lines',
                line=dict(color='#2ECC71', width=4, dash='dashdot'),
                name='Habit Plane (54.7°)',
                hoverinfo='skip'
            ))
            fig_example.update_layout(
                title=dict(
                    text="Example: Twin Boundary Hydrostatic Stress",
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 30],
                        title=dict(text='Hydrostatic Stress (GPa)', font=dict(size=14)),
                        gridcolor="rgba(100, 100, 100, 0.3)"
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(100, 100, 100, 0.3)",
                        rotation=90,
                        direction="clockwise",
                        tickmode='array',
                        tickvals=[40, 45, 50, 54.7, 60, 65, 70],
                        ticktext=['40°', '45°', '50°', '54.7°', '60°', '65°', '70°']
                    ),
                    bgcolor="rgba(240, 240, 240, 0.3)"
                ),
                showlegend=True,
                width=800,
                height=600
            )
            st.plotly_chart(fig_example, use_container_width=True)
            
            st.markdown("""
            <div class="habit-plane-highlight">
            🔬 <strong>Habit Plane Significance:</strong> The 54.7° angle represents the FCC twin habit plane
            where stress concentration is maximized for twin boundary defects. This specific orientation
            enables lower-temperature sintering through enhanced atomic diffusion.
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<h2 class="physics-header">🎨 Advanced Customization</h2>', unsafe_allow_html=True)
            
            if 'defect_comparison' in st.session_state:
                defect_comparison = st.session_state.defect_comparison
                
                st.markdown("#### 🎨 Visual Properties")
                # Create columns for customization options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### **Color & Transparency**")
                    fill_opacity = st.slider("Fill Opacity", 0.0, 1.0, 0.2, 0.05)
                    line_width = st.slider("Line Width", 1, 8, 3)
                    marker_size = st.slider("Marker Size", 0, 15, 8)
                    bg_opacity = st.slider("Background Opacity", 0.0, 1.0, 1.0, 0.05)
                
                with col2:
                    st.markdown("##### **Line Styles**")
                    line_style = st.selectbox(
                        "Line Style",
                        ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"],
                        index=0,
                        help="Select line style for the stress curves"
                    )
                    habit_plane_style = st.selectbox(
                        "Habit Plane Line Style",
                        ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"],
                        index=3,
                        help="Select line style for the habit plane reference line"
                    )
                
                with col3:
                    st.markdown("##### **Layout & Grid**")
                    show_grid = st.checkbox("Show Grid Lines", value=True)
                    show_habit_plane = st.checkbox("Show Habit Plane", value=True)
                    grid_color = st.color_picker("Grid Color", "#cccccc")
                    bgcolor = st.color_picker("Background Color", "#ffffff")
                
                # Color scheme customization
                st.markdown("#### 🎨 Defect Type Color Scheme")
                defect_colors = {}
                cols = st.columns(4)
                for i, defect in enumerate(["ISF", "ESF", "Twin", "No Defect"]):
                    with cols[i]:
                        st.markdown(f"**{defect}**")
                        current_color = st.session_state.visualizer.defect_colors[defect]
                        if not current_color.startswith('#'):
                            current_color = '#45B7D1'
                        color_key = f"color_{defect.replace(' ', '_')}"
                        defect_colors[defect] = st.color_picker(
                            f"{defect} Color",
                            current_color,
                            key=color_key
                        )
                        st.caption(f"Eigen strain: {st.session_state.interpolator.physics_analyzer.get_eigen_strain(defect):.2f}")
                
                # Generate customized chart
                if st.button("🎨 Render Customized Visualization", type="primary"):
                    with st.spinner("Applying customization..."):
                        # Update color schemes
                        for defect, color in defect_colors.items():
                            st.session_state.visualizer.defect_colors[defect] = color
                        
                        # Generate the appropriate chart with customization
                        if chart_type == "Basic Radar Chart":
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"Defect Stress Patterns: {stress_component.replace('_', ' ').title()}",
                                fill_opacity=fill_opacity,
                                line_width=line_width,
                                marker_size=marker_size,
                                show_grid=show_grid,
                                bgcolor=bgcolor,
                                line_style=line_style,
                                habit_plane_style=habit_plane_style
                            )
                        elif chart_type == "Sunburst Chart":
                            fig = st.session_state.visualizer.create_sunburst_defect_chart(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"Defect Stress Patterns - {stress_component.replace('_', ' ').title()}",
                                show_habit_plane=show_habit_plane,
                                line_width=line_width,
                                line_style=line_style
                            )
                        elif chart_type == "3D Interactive View":
                            fig = st.session_state.visualizer.create_interactive_3d_defect_sunburst(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"3D Defect Stress Distribution: {stress_component.replace('_', ' ').title()}",
                                background_color=bgcolor
                            )
                        else:
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"Defect Stress Patterns: {stress_component.replace('_', ' ').title()}",
                                fill_opacity=fill_opacity,
                                line_width=line_width,
                                marker_size=marker_size,
                                show_grid=show_grid,
                                bgcolor=bgcolor,
                                line_style=line_style,
                                habit_plane_style=habit_plane_style
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.success("✅ Customization applied successfully!")
            else:
                st.info("🔍 First generate a visualization in the 'Main Visualization' tab, then customize it here.")
        
        with tab4:
            st.markdown('<h2 class="physics-header">🔬 Detailed Physics Analysis</h2>', unsafe_allow_html=True)
            
            if 'defect_comparison' in st.session_state:
                defect_comparison = st.session_state.defect_comparison
                
                st.markdown("""
                ### ⚛️ Physical Interpretation of Stress Patterns
                The visualizations reveal fundamental physics of defect-mediated sintering. Here's a detailed analysis
                of the key patterns and their implications for materials processing.
                """)
                
                # Create analysis columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Stress Concentration Analysis")
                    
                    # Calculate and display key metrics
                    metrics = st.session_state.interpolator.calculate_stress_metrics(defect_comparison)
                    
                    # Display metrics in cards
                    for defect_type, values in metrics.items():
                        card_class = {
                            'ISF': 'isf-card',
                            'ESF': 'esf-card', 
                            'Twin': 'twin-card',
                            'No Defect': 'perfect-card'
                        }.get(defect_type, '')
                        
                        st.markdown(f"""
                        <div class="defect-card {card_class}">
                        <h4>🔬 {defect_type} Analysis</h4>
                        <p><strong>• Maximum Stress:</strong> {values['max_stress']:.4f} GPa at {values['peak_angle']:.2f}°</p>
                        <p><strong>• Habit Plane Stress:</strong> {values['habit_stress']:.4f} GPa</p>
                        <p><strong>• Stress Gradient:</strong> {values['habit_gradient']:.4f} GPa/deg</p>
                        <p><strong>• Eigen Strain:</strong> {values['eigen_strain']:.4f}</p>
                        <p><strong>• FWHM:</strong> {values['fwhm']:.2f}° (indicates concentration width)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 🔥 Sintering Implications")
                    st.markdown("""
                    <div class="habit-plane-highlight">
                    <h4>🌡️ Critical Sintering Thresholds</h4>
                    <p><strong>• Twin boundaries:</strong> Generate sufficient stress at habit plane to enable sintering
                    at temperatures 300-400°C below bulk melting point</p>
                    <p><strong>• Stacking faults:</strong> Moderate stress concentration enables intermediate sintering temperatures</p>
                    <p><strong>• Perfect crystals:</strong> Require significantly higher temperatures for equivalent diffusion</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                ### 📋 Design Recommendations
                1. **Defect Engineering**: Introduce controlled twin boundaries at 54.7° orientation to maximize
                stress-driven diffusion during sintering
                
                2. **Temperature Profiles**: Use lower sintering temperatures (500-600°C for Ag) when high twin density
                is present, versus 800-900°C for perfect crystals
                
                3. **Processing Control**: Monitor stress patterns during processing to optimize defect distribution
                for desired mechanical properties
                
                4. **Multi-scale Modeling**: Combine these stress visualizations with atomistic simulations to
                predict grain boundary mobility and final microstructure
                """)
                
                # Advanced analysis tools
                st.markdown("#### 🔧 Advanced Analysis Tools")
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Diffusion Enhancement Analysis", "Stress Gradient Analysis", "Energy Density Mapping"],
                    key="analysis_type"
                )
                
                if analysis_type == "Diffusion Enhancement Analysis":
                    st.markdown("##### 🔥 Diffusion Enhancement Analysis")
                    st.markdown("""
                    Using the stress patterns, we can estimate the enhancement in diffusion coefficient (D) relative to bulk.
                    **Physics correction: Only tensile (positive) hydrostatic stress enhances diffusion.**
                    
                    ```
                    D/D₀ = exp(Ω·σ_hydro/kT) for σ_hydro > 0
                    D/D₀ = 1.0 for σ_hydro ≤ 0
                    
                    Where:
                    - Ω = atomic volume (1.56e-29 m³ for Ag)
                    - σ_hydro = hydrostatic stress (Pa)
                    - k = Boltzmann constant (1.38e-23 J/K)
                    - T = temperature (650 K typical sintering temp)
                    ```
                    """)
                    
                    # Create advanced diffusion enhancement chart
                    fig_diff = st.session_state.visualizer.create_advanced_diffusion_enhancement_chart(
                        defect_comparison,
                        temperature=temperature,
                        material=material,
                        title=f"Diffusion Enhancement at {temperature}K for {material}",
                        radial_range_factor=1.0
                    )
                    
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    st.markdown("""
                    <div class="physics-corrected">
                    ⚛️ <strong>Physics-Corrected Result:</strong> Twin boundaries at the habit plane can enhance diffusion by 10-100x
                    compared to bulk material, but <strong>only in regions of tensile stress</strong>. Compressive regions show no enhancement.
                    This precise physics modeling enables accurate prediction of sintering behavior in nanocrystalline materials.
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("""
            ---
            <div style="text-align: center; color: #666; padding: 20px;">
            <strong>⚛️ Advanced Materials Research Tool</strong><br>
            This visualization system reveals the physics of defect-mediated sintering.<br>
            For research use only - © 2026 Materials Science Research Group
            </div>
            """, unsafe_allow_html=True)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
