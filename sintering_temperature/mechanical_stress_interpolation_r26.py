import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
import plotly.graph_objects as go
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
from numba import jit, prange
import time
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# =============================================
# PHYSICS-BASED STRESS ANALYZER WITH EIGEN STRAIN INTEGRATION
# =============================================

class PhysicsBasedStressAnalyzer:
    """Physics-based analyzer with eigen strain integration for defect types"""
    
    def __init__(self):
        # Eigen strains for different defect types (from physics literature)
        self.eigen_strains = {
            'ISF': 0.71,      # Intrinsic Stacking Fault
            'ESF': 1.41,      # Extrinsic Stacking Fault
            'Twin': 2.12,     # Twin boundary
            'No Defect': 0.0, # Perfect crystal
            'Unknown': 0.0
        }
        
        # Material properties for Ag (Silver) FCC structure
        self.ag_properties = {
            'shear_modulus': 30.0,     # GPa
            'youngs_modulus': 83.0,    # GPa
            'poissons_ratio': 0.37,
            'burgers_vector': 0.289,   # nm
            'stacking_fault_energy': 22.0,  # mJ/m²
            'lattice_constant': 0.408,  # nm
            'melting_point': 1234.93,  # K
            'thermal_expansion': 18.9e-6,  # 1/K
            'density': 10.49,          # g/cm³
            'atomic_weight': 107.8682  # g/mol
        }
        
        # Crystal orientation relationships for FCC
        self.crystal_orientations = {
            'habit_plane_angle': 54.7,  # Ag FCC twin habit plane
            'primary_slip_systems': 12,
            'schmid_factor_twinning': 0.5,
            'critical_resolved_shear_stress': 0.5  # MPa
        }
    
    def get_eigen_strain(self, defect_type):
        """Get eigen strain value for a specific defect type"""
        return self.eigen_strains.get(defect_type, 0.0)
    
    def compute_defect_energy(self, eigen_strain, volume=1.0):
        """Compute approximate defect formation energy"""
        # Simplified: energy ∝ eigen_strain² × volume × elastic modulus
        E = self.ag_properties['youngs_modulus'] * 1e9  # Convert to Pa
        energy = 0.5 * E * (eigen_strain ** 2) * volume * 1e-27  # Joules
        return energy * 6.242e18  # Convert to eV
    
    def extract_all_stress_components(self, eta, stress_fields, region_type='bulk'):
        """Extract all stress components with physics-based interpretation"""
        
        if eta is None or not isinstance(eta, np.ndarray):
            return {}
        
        # Region masks based on phase field order parameter
        if region_type == 'defect':
            mask = eta > 0.6
        elif region_type == 'interface':
            mask = (eta >= 0.4) & (eta <= 0.6)
        elif region_type == 'bulk':
            mask = eta < 0.4
        else:
            mask = np.ones_like(eta, dtype=bool)
        
        if not np.any(mask):
            return {}
        
        results = {}
        
        # Extract each stress component
        stress_components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        for comp_name in stress_components:
            if comp_name in stress_fields:
                stress_data = stress_fields[comp_name][mask]
                if len(stress_data) > 0:
                    results[comp_name] = {
                        'max_abs': float(np.max(np.abs(stress_data))),
                        'max': float(np.max(stress_data)),
                        'min': float(np.min(stress_data)),
                        'mean': float(np.mean(stress_data)),
                        'std': float(np.std(stress_data)),
                        'percentile_95': float(np.percentile(np.abs(stress_data), 95)),
                        'percentile_99': float(np.percentile(np.abs(stress_data), 99)),
                        'skewness': float(pd.Series(stress_data).skew()),
                        'kurtosis': float(pd.Series(stress_data).kurtosis())
                    }
        
        # Additional physics-based metrics
        if 'sigma_hydro' in results:
            hydro_data = stress_fields['sigma_hydro'][mask]
            results['physics_metrics'] = {
                'tensile_fraction': float(np.sum(hydro_data > 0) / len(hydro_data)),
                'compressive_fraction': float(np.sum(hydro_data < 0) / len(hydro_data)),
                'mean_hydrostatic_pressure': float(-np.mean(hydro_data)),  # Positive for compression
                'max_hydrostatic_tension': float(np.max(hydro_data[hydro_data > 0])) if np.any(hydro_data > 0) else 0.0,
                'max_hydrostatic_compression': float(np.min(hydro_data[hydro_data < 0])) if np.any(hydro_data < 0) else 0.0
            }
        
        return results
    
    def compute_stress_intensity_factor(self, stress_data, eigen_strain, defect_type='Twin'):
        """Compute stress intensity factor based on eigen strain and defect type"""
        
        if defect_type in self.eigen_strains:
            eigen_strain_val = self.eigen_strains[defect_type]
        else:
            eigen_strain_val = eigen_strain
        
        # For hydrostatic stress
        if 'sigma_hydro' in stress_data:
            sigma_h = np.abs(stress_data['sigma_hydro']['max_abs'])
            # Simplified K calculation based on linear elastic fracture mechanics
            K = sigma_h * np.sqrt(np.pi * eigen_strain_val * self.ag_properties['burgers_vector']) * 1e-3  # MPa√m
            return K
        return 0.0
    
    def compute_strain_energy_density(self, stress_fields):
        """Compute strain energy density from stress components"""
        if 'sigma_xx' in stress_fields and 'sigma_yy' in stress_fields and 'sigma_zz' in stress_fields:
            # For 3D stress state
            sigma_xx = stress_fields['sigma_xx']
            sigma_yy = stress_fields['sigma_yy']
            sigma_zz = stress_fields['sigma_zz']
            tau_xy = stress_fields.get('tau_xy', np.zeros_like(sigma_xx))
            tau_yz = stress_fields.get('tau_yz', np.zeros_like(sigma_xx))
            tau_zx = stress_fields.get('tau_zx', np.zeros_like(sigma_xx))
            
            # Strain energy density = 0.5 * σ:ε = 0.5 * C⁻¹:σ:σ
            E = self.ag_properties['youngs_modulus']
            nu = self.ag_properties['poissons_ratio']
            G = self.ag_properties['shear_modulus']
            
            # For isotropic material
            energy = (1/(2*E)) * (sigma_xx**2 + sigma_yy**2 + sigma_zz**2) \
                    - (nu/E) * (sigma_xx*sigma_yy + sigma_yy*sigma_zz + sigma_zz*sigma_xx) \
                    + (1/(2*G)) * (tau_xy**2 + tau_yz**2 + tau_zx**2)
            
            return energy
        return None
    
    def analyze_crystal_orientation_effects(self, stress_data, orientation_deg):
        """Analyze orientation-dependent stress effects"""
        habit_angle = self.crystal_orientations['habit_plane_angle']
        angle_diff = abs(orientation_deg - habit_angle)
        
        # Schmid factor calculation for twinning
        schmid_factor = self.crystal_orientations['schmid_factor_twinning'] * \
                       np.cos(np.radians(angle_diff))
        
        results = {
            'orientation_deg': orientation_deg,
            'habit_plane_distance_deg': angle_diff,
            'schmid_factor': float(schmid_factor),
            'habit_plane_proximity': 1.0 - (angle_diff / 45.0),  # Normalized proximity
            'is_near_habit_plane': angle_diff < 5.0
        }
        
        # Add orientation-dependent stress scaling
        if 'sigma_hydro' in stress_data:
            hydro_max = stress_data['sigma_hydro']['max_abs']
            # Stress enhancement near habit plane (simplified)
            orientation_factor = 1.0 + 0.5 * np.exp(-angle_diff/15.0)
            results['orientation_corrected_stress'] = float(hydro_max * orientation_factor)
        
        return results
    
    def compute_defect_interaction_energy(self, defect_type1, defect_type2, distance=1.0):
        """Compute interaction energy between two defects"""
        eps1 = self.get_eigen_strain(defect_type1)
        eps2 = self.get_eigen_strain(defect_type2)
        
        G = self.ag_properties['shear_modulus'] * 1e9  # Pa
        b = self.ag_properties['burgers_vector'] * 1e-9  # m
        
        # Simplified elastic interaction energy
        interaction_energy = (G * b**2 * eps1 * eps2) / (2 * np.pi * distance) * 1e9  # eV
        
        return interaction_energy

# =============================================
# ENHANCED SINTERING TEMPERATURE CALCULATOR WITH ARRHENIUS FOCUS
# =============================================

class EnhancedSinteringCalculator:
    """Enhanced sintering calculator with Arrhenius focus and defect-specific parameters"""
    
    def __init__(self, T0=623.0, beta=0.95, G=30.0, sigma_peak=28.5):
        self.T0 = T0  # Reference temperature at zero stress (K)
        self.beta = beta  # Calibration factor
        self.G = G  # Shear modulus of Ag (GPa)
        self.sigma_peak = sigma_peak  # Peak hydrostatic stress (GPa)
        self.T_min = 367.0  # Minimum sintering temperature at peak stress (K)
        
        # Material properties for Ag
        self.kB = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.Q_a = 1.1  # Activation energy for Ag diffusion (eV)
        self.omega = 0.85 * (0.408e-9)**3  # Activation volume (m³)
        self.omega_eV_per_GPa = self.omega * 6.242e18  # Convert to eV/GPa
        
        # Defect-specific diffusion parameters
        self.defect_diffusion_params = {
            'ISF': {
                'Q_a': 1.05,  # Slightly reduced activation energy
                'D0': 1e-5,    # Higher pre-exponential
                'omega_factor': 0.8,
                'description': 'Intrinsic Stacking Fault',
                'eigen_strain': 0.71
            },
            'ESF': {
                'Q_a': 0.95,
                'D0': 1e-4,
                'omega_factor': 0.9,
                'description': 'Extrinsic Stacking Fault',
                'eigen_strain': 1.41
            },
            'Twin': {
                'Q_a': 0.85,
                'D0': 1e-3,
                'omega_factor': 1.0,
                'description': 'Twin Boundary',
                'eigen_strain': 2.12
            },
            'No Defect': {
                'Q_a': 1.1,
                'D0': 1e-6,
                'omega_factor': 0.5,
                'description': 'Perfect Crystal',
                'eigen_strain': 0.0
            }
        }
        
        # System classification boundaries
        self.system_boundaries = {
            'System 1 (Perfect Crystal)': {'min_stress': 0.0, 'max_stress': 5.0, 'T_range': (620, 630)},
            'System 2 (Stacking Faults/Twins)': {'min_stress': 5.0, 'max_stress': 20.0, 'T_range': (450, 550)},
            'System 3 (Plastic Deformation)': {'min_stress': 20.0, 'max_stress': 35.0, 'T_range': (350, 400)}
        }
    
    def compute_sintering_temperature_exponential(self, sigma_h):
        """Compute sintering temperature using exponential empirical model"""
        sigma_abs = np.abs(sigma_h)
        T_sinter = self.T0 * np.exp(-self.beta * sigma_abs / self.G)
        return np.clip(T_sinter, self.T_min, self.T0)
    
    def compute_sintering_temperature_arrhenius(self, sigma_h, D0=1e-6, D_crit=1e-10):
        """Compute sintering temperature using stress-modified Arrhenius equation"""
        sigma_abs = np.abs(sigma_h)
        Q_eff = self.Q_a - self.omega_eV_per_GPa * sigma_abs
        
        # Ensure Q_eff is positive
        Q_eff = max(Q_eff, 0.1)
        
        if D0 / D_crit > 0:
            T_sinter = Q_eff / (self.kB * np.log(D0 / D_crit))
        else:
            T_sinter = self.T0
        
        return np.clip(T_sinter, self.T_min, self.T0)
    
    def compute_sintering_temperature_arrhenius_defect(self, sigma_h, defect_type='Twin', D_crit=1e-10):
        """Compute sintering temperature using defect-specific Arrhenius parameters"""
        
        sigma_abs = np.abs(sigma_h)
        
        if defect_type in self.defect_diffusion_params:
            params = self.defect_diffusion_params[defect_type]
            Q_a = params['Q_a']
            D0 = params['D0']
            omega_factor = params['omega_factor']
        else:
            Q_a = self.Q_a
            D0 = 1e-6
            omega_factor = 1.0
        
        Q_eff = Q_a - (self.omega_eV_per_GPa * omega_factor * sigma_abs)
        
        # Ensure Q_eff is positive
        Q_eff = max(Q_eff, 0.1)
        
        # Arrhenius equation: D = D0 * exp(-Q_eff/(kB*T))
        # Solve for T when D = D_crit
        if D0 / D_crit > 0:
            T_sinter = Q_eff / (self.kB * np.log(D0 / D_crit))
        else:
            T_sinter = self.T0
        
        return np.clip(T_sinter, self.T_min, self.T0)
    
    def compute_detailed_sintering_analysis(self, sigma_h, defect_type='Twin', orientation_deg=0.0):
        """Compute comprehensive sintering analysis"""
        
        sigma_abs = np.abs(sigma_h)
        
        # Compute temperatures using all models
        T_exp = self.compute_sintering_temperature_exponential(sigma_abs)
        T_arr_standard = self.compute_sintering_temperature_arrhenius(sigma_abs)
        T_arr_defect = self.compute_sintering_temperature_arrhenius_defect(sigma_abs, defect_type)
        
        # Get defect parameters
        defect_params = self.defect_diffusion_params.get(defect_type, self.defect_diffusion_params['No Defect'])
        
        # Calculate activation energy reduction
        Q_eff_standard = self.Q_a - self.omega_eV_per_GPa * sigma_abs
        Q_eff_defect = defect_params['Q_a'] - (self.omega_eV_per_GPa * defect_params['omega_factor'] * sigma_abs)
        
        # Calculate diffusion enhancement factors
        T_ref = 300.0  # Reference temperature in K
        D_enhancement_300K = np.exp((self.omega_eV_per_GPa * sigma_abs) / (self.kB * T_ref))
        
        # Get system classification
        system_info = self.map_system_to_temperature(sigma_abs)
        
        results = {
            'stress_analysis': {
                'hydrostatic_stress_gpa': float(sigma_abs),
                'stress_sign': 'compressive' if sigma_h < 0 else 'tensile',
                'absolute_stress': float(sigma_abs),
                'normalized_stress': float(sigma_abs / self.sigma_peak)
            },
            'temperature_predictions': {
                'exponential_model_k': float(T_exp),
                'exponential_model_c': float(T_exp - 273.15),
                'arrhenius_standard_k': float(T_arr_standard),
                'arrhenius_standard_c': float(T_arr_standard - 273.15),
                'arrhenius_defect_k': float(T_arr_defect),
                'arrhenius_defect_c': float(T_arr_defect - 273.15),
                'recommended_model': 'arrhenius_defect' if defect_type != 'No Defect' else 'exponential'
            },
            'activation_energy_analysis': {
                'Q_a_standard_eV': float(self.Q_a),
                'Q_a_defect_eV': float(defect_params['Q_a']),
                'Q_eff_standard_eV': float(Q_eff_standard),
                'Q_eff_defect_eV': float(Q_eff_defect),
                'reduction_standard_eV': float(self.omega_eV_per_GPa * sigma_abs),
                'reduction_defect_eV': float(self.omega_eV_per_GPa * defect_params['omega_factor'] * sigma_abs),
                'reduction_percentage': float((self.omega_eV_per_GPa * sigma_abs) / self.Q_a * 100)
            },
            'diffusion_analysis': {
                'D0_standard': 1e-6,
                'D0_defect': float(defect_params['D0']),
                'D_enhancement_300K': float(D_enhancement_300K),
                'relative_diffusion_rate': float(defect_params['D0'] / 1e-6 * D_enhancement_300K)
            },
            'system_classification': {
                'system': system_info[0],
                'T_range_k': system_info[1],
                'T_range_c': (system_info[1][0] - 273.15, system_info[1][1] - 273.15),
                'predicted_T_k': float(system_info[2]),
                'predicted_T_c': float(system_info[2] - 273.15)
            },
            'defect_parameters': {
                'type': defect_type,
                'description': defect_params['description'],
                'eigen_strain': defect_params['eigen_strain'],
                'omega_factor': defect_params['omega_factor'],
                'D0_multiplier': float(defect_params['D0'] / 1e-6)
            },
            'orientation_effects': {
                'orientation_deg': float(orientation_deg),
                'habit_plane_distance': float(abs(orientation_deg - 54.7)),
                'is_near_habit_plane': abs(orientation_deg - 54.7) < 5.0,
                'habit_plane_factor': np.exp(-abs(orientation_deg - 54.7) / 15.0)
            }
        }
        
        return results
    
    def map_system_to_temperature(self, sigma_h):
        """Map hydrostatic stress to system classification"""
        sigma_abs = np.abs(sigma_h)
        
        if sigma_abs < 5.0:
            system = "System 1 (Perfect Crystal)"
            T_range = (620, 630)  # K
        elif sigma_abs < 20.0:
            system = "System 2 (Stacking Faults/Twins)"
            T_range = (450, 550)  # K
        else:
            system = "System 3 (Plastic Deformation)"
            T_range = (350, 400)  # K
            
        T_sinter = self.compute_sintering_temperature_exponential(sigma_abs)
        return system, T_range, T_sinter
    
    def get_theoretical_curve(self, max_stress=35.0, n_points=100):
        """Generate theoretical curve of T_sinter vs |σ_h| for all defect types"""
        
        stresses = np.linspace(0, max_stress, n_points)
        
        curves = {
            'stresses': stresses.tolist(),
            'exponential_standard': self.compute_sintering_temperature_exponential(stresses).tolist(),
            'arrhenius_standard': self.compute_sintering_temperature_arrhenius(stresses).tolist(),
        }
        
        # Add defect-specific curves
        for defect_type in ['ISF', 'ESF', 'Twin', 'No Defect']:
            curves[f'arrhenius_{defect_type.lower()}'] = [
                float(self.compute_sintering_temperature_arrhenius_defect(s, defect_type))
                for s in stresses
            ]
        
        # Add model parameters
        curves['parameters'] = {
            'T0': self.T0,
            'T_min': self.T_min,
            'sigma_peak': self.sigma_peak,
            'beta': self.beta,
            'G': self.G,
            'Q_a': self.Q_a,
            'omega_eV_per_GPa': self.omega_eV_per_GPa
        }
        
        return curves
    
    def create_comprehensive_sintering_plot(self, stresses, temperatures, defect_type='Twin',
                                           title="Comprehensive Sintering Analysis"):
        """Create detailed sintering temperature plot with multiple models"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Main sintering curve
        ax1.plot(stresses, temperatures, 'b-', linewidth=3, label='Exponential Model')
        
        # Add defect-specific Arrhenius curves
        colors = {'ISF': 'orange', 'ESF': 'red', 'Twin': 'green', 'No Defect': 'purple'}
        for defect in ['ISF', 'ESF', 'Twin', 'No Defect']:
            T_defect = [self.compute_sintering_temperature_arrhenius_defect(s, defect) for s in stresses]
            ax1.plot(stresses, T_defect, '--', linewidth=2, color=colors[defect],
                    label=f'Arrhenius ({defect})', alpha=0.7)
        
        # System boundaries
        ax1.axvspan(0, 5, alpha=0.1, color='green', label='System 1 (Perfect)')
        ax1.axvspan(5, 20, alpha=0.1, color='orange', label='System 2 (SF/Twin)')
        ax1.axvspan(20, 35, alpha=0.1, color='red', label='System 3 (Plastic)')
        
        ax1.set_xlabel('Absolute Hydrostatic Stress |σ_h| (GPa)', fontsize=11)
        ax1.set_ylabel('Sintering Temperature (K)', fontsize=11)
        ax1.set_title('Sintering Temperature vs Hydrostatic Stress', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        
        # 2. Activation energy reduction
        ax2.plot(stresses, self.omega_eV_per_GPa * stresses, 'r-', linewidth=3,
                label=f'ΔQ = ω·σ = {self.omega_eV_per_GPa:.3f} eV/GPa × σ')
        
        # Add defect-specific reductions
        for defect in ['ISF', 'ESF', 'Twin']:
            params = self.defect_diffusion_params[defect]
            reduction = self.omega_eV_per_GPa * params['omega_factor'] * stresses
            ax2.plot(stresses, reduction, '--', linewidth=2, color=colors[defect],
                    label=f'{defect} (ω-factor: {params["omega_factor"]})')
        
        ax2.axhline(self.Q_a, color='black', linestyle=':', linewidth=2, label=f'Q_a = {self.Q_a} eV')
        ax2.set_xlabel('Absolute Hydrostatic Stress |σ_h| (GPa)', fontsize=11)
        ax2.set_ylabel('Activation Energy Reduction ΔQ (eV)', fontsize=11)
        ax2.set_title('Stress-Induced Activation Energy Reduction', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        
        # 3. Diffusion enhancement
        T_ref = 300.0  # Room temperature
        D_enhancement = np.exp(self.omega_eV_per_GPa * stresses / (self.kB * T_ref))
        
        ax3.plot(stresses, D_enhancement, 'g-', linewidth=3)
        ax3.set_yscale('log')
        ax3.set_xlabel('Absolute Hydrostatic Stress |σ_h| (GPa)', fontsize=11)
        ax3.set_ylabel('Diffusion Rate Enhancement (D/D₀)', fontsize=11)
        ax3.set_title(f'Diffusion Enhancement at T = {T_ref} K', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, which='both')
        
        # Add annotations for significant enhancement levels
        for stress, enhancement in [(5, D_enhancement[stresses.tolist().index(5)] if 5 in stresses else 10),
                                   (20, D_enhancement[stresses.tolist().index(20)] if 20 in stresses else 1000)]:
            ax3.annotate(f'{enhancement:.0f}×', xy=(stress, enhancement),
                        xytext=(stress+2, enhancement*1.5),
                        arrowprops=dict(arrowstyle='->', color='gray'),
                        fontsize=9)
        
        # 4. System classification distribution
        system_stresses = [2.5, 12.5, 27.5]  # Representative stresses for each system
        system_temperatures = [self.compute_sintering_temperature_exponential(s) for s in system_stresses]
        system_labels = ['System 1\n(Perfect)', 'System 2\n(SF/Twin)', 'System 3\n(Plastic)']
        system_colors = ['green', 'orange', 'red']
        
        bars = ax4.bar(system_labels, system_temperatures, color=system_colors, edgecolor='black', alpha=0.7)
        ax4.set_ylabel('Sintering Temperature (K)', fontsize=11)
        ax4.set_title('System Classification - Representative Temperatures', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, temp in zip(bars, system_temperatures):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{temp:.0f} K\n({temp-273.15:.0f}°C)',
                    ha='center', va='bottom', fontsize=9)
        
        # Add Celsius scale on secondary axis
        ax1_2 = ax1.twinx()
        celsius_ticks = ax1.get_yticks()
        ax1_2.set_ylim(ax1.get_ylim())
        ax1_2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
        ax1_2.set_ylabel('Temperature (°C)', fontsize=11)
        
        plt.suptitle(f'{title} - Defect Type: {defect_type}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig

# =============================================
# ENHANCED SOLUTION LOADER WITH PHYSICS AWARENESS
# =============================================

class EnhancedSolutionLoader:
    """Enhanced solution loader with physics-aware processing"""
    
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
            print(f"Error loading {file_path}: {e}")
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
                
                # Perform physics analysis if we have history
                if standardized['history']:
                    last_frame = standardized['history'][-1]
                    if isinstance(last_frame, dict):
                        eta = last_frame.get('eta')
                        stresses = last_frame.get('stresses', {})
                        
                        if eta is not None and stresses:
                            # Analyze all regions
                            physics_results = {}
                            for region in ['defect', 'interface', 'bulk']:
                                region_results = self.physics_analyzer.extract_all_stress_components(
                                    eta, stresses, region
                                )
                                if region_results:
                                    physics_results[region] = region_results
                            
                            standardized['physics_analysis'] = physics_results
                            standardized['metadata']['physics_processed'] = True
                
                # Add eigen strain based on defect type
                params = standardized['params']
                if 'defect_type' in params:
                    defect_type = params['defect_type']
                    eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
                    params['eigen_strain'] = eigen_strain
                    
                    # Update eps0 if not set or different from eigen strain
                    if 'eps0' not in params or abs(params['eps0'] - eigen_strain) > 0.1:
                        params['eps0'] = eigen_strain
            
            # Convert tensors to numpy arrays
            self._convert_tensors(standardized)
            
        except Exception as e:
            print(f"Standardization error: {e}")
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
    
    def get_solution_statistics(self, solutions):
        """Get comprehensive statistics about loaded solutions"""
        if not solutions:
            return {}
        
        stats = {
            'total_solutions': len(solutions),
            'defect_types': {},
            'orientations': [],
            'eigen_strains': [],
            'has_physics_analysis': 0,
            'regions_analyzed': set()
        }
        
        for sol in solutions:
            params = sol.get('params', {})
            
            # Defect type statistics
            defect_type = params.get('defect_type', 'Unknown')
            stats['defect_types'][defect_type] = stats['defect_types'].get(defect_type, 0) + 1
            
            # Orientation statistics
            theta = params.get('theta', 0.0)
            if theta is not None:
                theta_deg = np.degrees(theta) % 360
                stats['orientations'].append(theta_deg)
            
            # Eigen strain statistics
            eps0 = params.get('eps0', 0.0)
            stats['eigen_strains'].append(eps0)
            
            # Physics analysis statistics
            if sol.get('physics_analysis'):
                stats['has_physics_analysis'] += 1
                for region in sol['physics_analysis'].keys():
                    stats['regions_analyzed'].add(region)
        
        # Calculate additional statistics
        if stats['orientations']:
            stats['orientation_stats'] = {
                'min': float(np.min(stats['orientations'])),
                'max': float(np.max(stats['orientations'])),
                'mean': float(np.mean(stats['orientations'])),
                'std': float(np.std(stats['orientations']))
            }
        
        if stats['eigen_strains']:
            stats['eigen_strain_stats'] = {
                'min': float(np.min(stats['eigen_strains'])),
                'max': float(np.max(stats['eigen_strains'])),
                'mean': float(np.mean(stats['eigen_strains'])),
                'std': float(np.std(stats['eigen_strains']))
            }
        
        stats['regions_analyzed'] = list(stats['regions_analyzed'])
        
        return stats

# =============================================
# PHYSICS-AWARE INTERPOLATOR WITH COMBINED ATTENTION
# =============================================

class PhysicsAwareInterpolator:
    """Physics-aware interpolator with combined attention and Gaussian regularization"""
    
    def __init__(self, sigma=0.3, attention_dim=32, num_heads=4,
                 attention_blend=0.7, use_spatial=True):
        self.sigma = sigma
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.attention_blend = attention_blend  # Blend ratio: attention vs spatial
        self.use_spatial = use_spatial
        
        # Physics-based constraints
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        self.habit_plane_angle = 54.7
        
        # Parameter mappings with physics significance
        self.defect_map = {
            'ISF': [1, 0, 0, 0, self.physics_analyzer.get_eigen_strain('ISF')],
            'ESF': [0, 1, 0, 0, self.physics_analyzer.get_eigen_strain('ESF')],
            'Twin': [0, 0, 1, 0, self.physics_analyzer.get_eigen_strain('Twin')],
            'No Defect': [0, 0, 0, 1, self.physics_analyzer.get_eigen_strain('No Defect')]
        }
        
        self.shape_map = {
            'Square': [1, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0],
            'Vertical Fault': [0, 0, 1, 0],
            'Rectangle': [0, 0, 0, 1]
        }
        
        # Initialize attention layers
        self.query_projection = nn.Linear(15, attention_dim)
        self.key_projection = nn.Linear(15, attention_dim)
        self.value_projection = nn.Linear(15, attention_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.output_projection = nn.Linear(attention_dim, 15)
    
    def compute_parameter_vector(self, params, target_angle_deg=None):
        """Convert parameters to numerical vector with physics-aware encoding"""
        vector = []
        
        if not isinstance(params, dict):
            return np.zeros(15, dtype=np.float32)
        
        # Defect type with eigen strain (5 dimensions)
        defect = params.get('defect_type', 'Twin')
        defect_vector = self.defect_map.get(defect, [0, 0, 0, 0, 0])
        vector.extend(defect_vector)
        
        # Shape (4 dimensions)
        shape = params.get('shape', 'Square')
        vector.extend(self.shape_map.get(shape, [0, 0, 0, 0]))
        
        # Numeric parameters with physics normalization (6 dimensions)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        # Normalize with physics constraints
        vector.append(eps0 / 3.0)  # eps0 normalized 0-1 (max eigen strain ~3)
        vector.append((kappa - 0.1) / (2.0 - 0.1))  # kappa normalized 0-1
        vector.append(theta / np.pi)  # theta normalized 0-1 (0 to π)
        
        # Additional physics features
        # Orientation proximity to habit plane
        if target_angle_deg is not None:
            theta_deg = target_angle_deg
        else:
            theta_deg = np.degrees(theta) if theta is not None else 0.0
        
        habit_distance = abs(theta_deg - self.habit_plane_angle)
        vector.append(np.exp(-habit_distance / 45.0))  # Habit plane proximity
        
        # Stress concentration factor (simplified)
        stress_factor = 1.0 + eps0 * 0.5  # Higher eigen strain → higher stress
        vector.append(stress_factor / 2.0)  # Normalized
        
        # Crystal symmetry factor
        symmetry_factor = np.abs(np.sin(np.radians(2 * theta_deg)))
        vector.append(symmetry_factor)
        
        return np.array(vector, dtype=np.float32)
    
    def compute_physics_weights(self, source_params, target_params, target_angle_deg):
        """Compute physics-based weights considering eigen strains and habit plane"""
        
        physics_weights = []
        target_vector = self.compute_parameter_vector(target_params, target_angle_deg)
        
        for src_params in source_params:
            # 1. Defect type compatibility
            src_defect = src_params.get('defect_type', 'Unknown')
            target_defect = target_params.get('defect_type', 'Unknown')
            
            if src_defect == target_defect:
                defect_weight = 1.0
            elif src_defect in ['ISF', 'ESF', 'Twin'] and target_defect in ['ISF', 'ESF', 'Twin']:
                defect_weight = 0.7  # Similar defect types
            else:
                defect_weight = 0.3
            
            # 2. Eigen strain similarity
            src_eps0 = src_params.get('eps0', 0.707)
            target_eps0 = target_params.get('eps0', 0.707)
            eps0_diff = np.abs(src_eps0 - target_eps0)
            eps0_weight = np.exp(-eps0_diff / 0.2)  # Exponential decay
            
            # 3. Orientation similarity (especially near habit plane)
            src_theta = src_params.get('theta', 0.0)
            src_deg = np.degrees(src_theta) % 360 if src_theta is not None else 0.0
            
            # Distance to target angle
            angle_diff = min(
                abs(src_deg - target_angle_deg),
                abs(src_deg - (target_angle_deg + 360)),
                abs(src_deg - (target_angle_deg - 360))
            )
            
            # Special emphasis near habit plane
            habit_distance_src = abs(src_deg - self.habit_plane_angle)
            habit_distance_target = abs(target_angle_deg - self.habit_plane_angle)
            
            if habit_distance_src < 10.0 and habit_distance_target < 10.0:
                # Both near habit plane
                habit_weight = 1.0
            elif habit_distance_src < 10.0 or habit_distance_target < 10.0:
                # One near habit plane
                habit_weight = 0.8
            else:
                habit_weight = 0.5
            
            # Combined angle weight
            angle_weight = np.exp(-angle_diff / 45.0) * habit_weight
            
            # 4. Combined physics weight
            physics_weight = defect_weight * eps0_weight * angle_weight
            physics_weights.append(physics_weight)
        
        # Normalize
        if physics_weights and np.sum(physics_weights) > 0:
            physics_weights = np.array(physics_weights) / np.sum(physics_weights)
        
        return physics_weights
    
    def compute_attention_weights(self, source_vectors, target_vector):
        """Compute attention weights using transformer-like attention"""
        
        if len(source_vectors) == 0:
            return np.array([])
        
        # Convert to PyTorch tensors
        source_tensor = torch.FloatTensor(source_vectors).unsqueeze(0)  # (1, N, 15)
        target_tensor = torch.FloatTensor(target_vector).unsqueeze(0).unsqueeze(1)  # (1, 1, 15)
        
        # Project to attention space
        query = self.query_projection(target_tensor)  # (1, 1, attention_dim)
        keys = self.key_projection(source_tensor)     # (1, N, attention_dim)
        values = self.value_projection(source_tensor) # (1, N, attention_dim)
        
        # Multi-head attention
        attention_output, attention_weights = self.multihead_attention(
            query, keys, values
        )
        
        # Get attention weights (averaged over heads)
        attention_weights = attention_weights.squeeze().detach().numpy()
        
        return attention_weights
    
    def compute_spatial_weights(self, source_vectors, target_vector):
        """Compute spatial locality weights using Gaussian kernel"""
        if len(source_vectors) == 0:
            return np.array([])
        
        # Calculate Euclidean distances in parameter space
        distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
        
        # Apply Gaussian kernel
        weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
        
        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights
    
    def interpolate_stress_components(self, sources, target_angle_deg, target_params,
                                     region_type='bulk', use_physics_constraints=True):
        """Interpolate all stress components with combined regularization"""
        
        # Validate inputs
        if not sources:
            return None
        
        # Convert angle to radians for target params
        target_params = target_params.copy()
        target_params['theta'] = np.radians(target_angle_deg)
        
        # Compute target parameter vector
        target_vector = self.compute_parameter_vector(target_params, target_angle_deg)
        
        # Prepare source data
        valid_sources = []
        source_vectors = []
        source_stresses = {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []}
        
        for src in sources:
            if not isinstance(src, dict) or 'params' not in src or 'history' not in src:
                continue
            
            valid_sources.append(src)
            
            # Get source parameter vector
            src_params = src['params']
            src_theta = src_params.get('theta', 0.0)
            src_deg = np.degrees(src_theta) if src_theta is not None else 0.0
            src_vector = self.compute_parameter_vector(src_params, src_deg)
            source_vectors.append(src_vector)
            
            # Extract stress components from physics analysis
            physics_analysis = src.get('physics_analysis', {})
            if region_type in physics_analysis:
                region_data = physics_analysis[region_type]
                for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                    if comp in region_data:
                        source_stresses[comp].append(region_data[comp]['max_abs'])
                    else:
                        source_stresses[comp].append(0.0)
            else:
                # Fallback: analyze last frame
                history = src.get('history', [])
                if history:
                    last_frame = history[-1]
                    if isinstance(last_frame, dict):
                        eta = last_frame.get('eta')
                        stresses = last_frame.get('stresses', {})
                        if eta is not None and stresses:
                            analyzer = PhysicsBasedStressAnalyzer()
                            region_data = analyzer.extract_all_stress_components(
                                eta, stresses, region_type
                            )
                            for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                                if comp in region_data:
                                    source_stresses[comp].append(region_data[comp]['max_abs'])
                                else:
                                    source_stresses[comp].append(0.0)
                        else:
                            for comp in source_stresses.keys():
                                source_stresses[comp].append(0.0)
                    else:
                        for comp in source_stresses.keys():
                            source_stresses[comp].append(0.0)
                else:
                    for comp in source_stresses.keys():
                        source_stresses[comp].append(0.0)
        
        if not valid_sources:
            return None
        
        source_vectors = np.array(source_vectors)
        
        # Compute weights using combined approach
        weights = np.zeros(len(valid_sources))
        
        if self.use_spatial:
            # Compute spatial weights
            spatial_weights = self.compute_spatial_weights(source_vectors, target_vector)
            
            # Compute attention weights
            attention_weights = self.compute_attention_weights(source_vectors, target_vector)
            
            if len(attention_weights) > 0 and len(spatial_weights) > 0:
                # Combine attention and spatial weights
                combined_weights = (self.attention_blend * attention_weights +
                                   (1 - self.attention_blend) * spatial_weights)
            else:
                combined_weights = spatial_weights
        else:
            # Use only attention weights
            attention_weights = self.compute_attention_weights(source_vectors, target_vector)
            combined_weights = attention_weights if len(attention_weights) > 0 else np.ones(len(valid_sources))
        
        # Apply physics constraints if requested
        if use_physics_constraints:
            source_params = [src['params'] for src in valid_sources]
            physics_weights = self.compute_physics_weights(source_params, target_params, target_angle_deg)
            
            if physics_weights is not None:
                # Blend with combined weights (50% physics, 50% ML)
                final_weights = 0.5 * combined_weights + 0.5 * physics_weights
            else:
                final_weights = combined_weights
        else:
            final_weights = combined_weights
        
        # Normalize final weights
        if np.sum(final_weights) > 0:
            final_weights = final_weights / np.sum(final_weights)
        else:
            final_weights = np.ones_like(final_weights) / len(final_weights)
        
        # Perform weighted interpolation for each stress component
        interpolated_stresses = {}
        for comp, stresses in source_stresses.items():
            if stresses and len(stresses) == len(final_weights):
                stresses_array = np.array(stresses)
                interpolated_stresses[comp] = float(np.sum(final_weights * stresses_array))
            else:
                interpolated_stresses[comp] = 0.0
        
        # Compute sintering temperatures
        sintering_calculator = EnhancedSinteringCalculator()
        sigma_h = interpolated_stresses.get('sigma_hydro', 0.0)
        defect_type = target_params.get('defect_type', 'Twin')
        
        sintering_analysis = sintering_calculator.compute_detailed_sintering_analysis(
            sigma_h, defect_type, target_angle_deg
        )
        
        # Prepare result
        result = {
            'target_angle_deg': float(target_angle_deg),
            'target_params': target_params,
            'region_type': region_type,
            'interpolated_stresses': interpolated_stresses,
            'sintering_analysis': sintering_analysis,
            'weights': {
                'final_weights': final_weights.tolist(),
                'num_sources': len(valid_sources),
                'weight_entropy': float(-np.sum(final_weights * np.log(final_weights + 1e-10)))
            },
            'physics_metrics': {
                'habit_plane_distance': float(abs(target_angle_deg - self.habit_plane_angle)),
                'is_near_habit_plane': abs(target_angle_deg - self.habit_plane_angle) < 5.0,
                'eigen_strain': target_params.get('eps0', 0.0)
            }
        }
        
        return result
    
    def create_vicinity_sweep(self, sources, target_params, vicinity_range=10.0,
                             n_points=50, region_type='bulk'):
        """Create stress sweep in vicinity of habit plane"""
        
        center_angle = self.habit_plane_angle
        min_angle = center_angle - vicinity_range
        max_angle = center_angle + vicinity_range
        
        angles = np.linspace(min_angle, max_angle, n_points)
        
        results = {
            'angles': angles.tolist(),
            'stresses': {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []},
            'sintering_temps': {'exponential': [], 'arrhenius_defect': []},
            'weights_matrix': [],
            'defect_type': target_params.get('defect_type', 'Twin')
        }
        
        for angle in angles:
            result = self.interpolate_stress_components(
                sources, float(angle), target_params, region_type
            )
            
            if result:
                stresses = result['interpolated_stresses']
                sintering = result['sintering_analysis']['temperature_predictions']
                
                results['stresses']['sigma_hydro'].append(stresses['sigma_hydro'])
                results['stresses']['von_mises'].append(stresses['von_mises'])
                results['stresses']['sigma_mag'].append(stresses['sigma_mag'])
                
                results['sintering_temps']['exponential'].append(sintering['exponential_model_k'])
                results['sintering_temps']['arrhenius_defect'].append(sintering['arrhenius_defect_k'])
                
                results['weights_matrix'].append(result['weights']['final_weights'])
            else:
                # Fill with zeros if interpolation fails
                for comp in results['stresses'].keys():
                    results['stresses'][comp].append(0.0)
                for model in results['sintering_temps'].keys():
                    results['sintering_temps'][model].append(0.0)
                results['weights_matrix'].append([0.0] * len(sources) if sources else [])
        
        return results
    
    def compare_defect_types(self, sources, angle_range=(0, 360), n_points=100,
                            region_type='bulk', shapes=None):
        """Compare different defect types across orientation range"""
        
        if shapes is None:
            shapes = ['Square']  # Default shape
        
        defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
        colors = {'ISF': '#FF6B6B', 'ESF': '#4ECDC4', 'Twin': '#45B7D1', 'No Defect': '#96CEB4'}
        
        min_angle, max_angle = angle_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        comparison_results = {}
        
        for defect in defect_types:
            for shape in shapes:
                key = f"{defect}_{shape}"
                target_params = {
                    'defect_type': defect,
                    'shape': shape,
                    'eps0': self.physics_analyzer.get_eigen_strain(defect),
                    'kappa': 0.6
                }
                
                stresses = {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []}
                sintering_temps = []
                
                for angle in angles:
                    result = self.interpolate_stress_components(
                        sources, float(angle), target_params, region_type
                    )
                    
                    if result:
                        stresses['sigma_hydro'].append(result['interpolated_stresses']['sigma_hydro'])
                        stresses['von_mises'].append(result['interpolated_stresses']['von_mises'])
                        stresses['sigma_mag'].append(result['interpolated_stresses']['sigma_mag'])
                        
                        sintering = result['sintering_analysis']['temperature_predictions']
                        sintering_temps.append(sintering['arrhenius_defect_k'])
                    else:
                        for comp in stresses.keys():
                            stresses[comp].append(0.0)
                        sintering_temps.append(0.0)
                
                comparison_results[key] = {
                    'defect_type': defect,
                    'shape': shape,
                    'angles': angles.tolist(),
                    'stresses': stresses,
                    'sintering_temps': sintering_temps,
                    'color': colors.get(defect, '#000000'),
                    'eigen_strain': target_params['eps0']
                }
        
        return comparison_results

# =============================================
# ENHANCED VISUALIZER FOR HABIT PLANE VICINITY WITH PUBLICATION QUALITY
# =============================================

class EnhancedHabitPlaneVisualizer:
    """Enhanced visualizer with publication-quality settings"""
    
    def __init__(self, habit_angle=54.7, publication_mode=True):
        self.habit_angle = habit_angle
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        self.publication_mode = publication_mode
        
        # Color schemes
        self.stress_colors = {
            'sigma_hydro': 'rgb(31, 119, 180)',
            'von_mises': 'rgb(255, 127, 14)',
            'sigma_mag': 'rgb(44, 160, 44)'
        }
        
        self.defect_colors = {
            'ISF': 'rgb(255, 187, 120)',
            'ESF': 'rgb(255, 152, 150)',
            'Twin': 'rgb(152, 223, 138)',
            'No Defect': 'rgb(174, 199, 232)'
        }
        
        self.system_colors = {
            'System 1 (Perfect Crystal)': 'rgb(46, 204, 113)',
            'System 2 (Stacking Faults/Twins)': 'rgb(241, 196, 15)',
            'System 3 (Plastic Deformation)': 'rgb(231, 76, 60)'
        }
        
        # Publication quality settings
        if publication_mode:
            # Larger fonts for publication
            self.title_font_size = 24
            self.axis_label_font_size = 18
            self.tick_label_font_size = 16
            self.legend_font_size = 14
            self.annotation_font_size = 14
            self.line_width = 3
            self.marker_size = 10
        else:
            # Default settings
            self.title_font_size = 20
            self.axis_label_font_size = 14
            self.tick_label_font_size = 12
            self.legend_font_size = 12
            self.annotation_font_size = 12
            self.line_width = 2
            self.marker_size = 6
    
    def create_publication_quality_figure(self, fig=None, title="", xlabel="", ylabel=""):
        """Apply publication quality formatting to a matplotlib figure"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            ax = fig.gca()
        
        # Set publication quality formatting
        ax.set_title(title, fontsize=self.title_font_size, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=self.axis_label_font_size, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=self.axis_label_font_size, fontweight='bold')
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', 
                      labelsize=self.tick_label_font_size, width=2)
        ax.tick_params(axis='both', which='minor', 
                      labelsize=self.tick_label_font_size-2, width=1)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        if ax.get_legend():
            ax.legend(fontsize=self.legend_font_size, frameon=True, 
                     framealpha=0.9, edgecolor='black')
        
        # Spines
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        return fig
    
    def create_publication_plotly_layout(self, title="", width=1000, height=800):
        """Create publication quality layout for Plotly figures"""
        return dict(
            title=dict(
                text=title,
                font=dict(
                    size=self.title_font_size,
                    family="Arial",
                    color='black',
                    weight='bold'
                ),
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            font=dict(
                family="Arial",
                size=self.legend_font_size,
                color="black"
            ),
            legend=dict(
                font=dict(size=self.legend_font_size),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                x=1.02,
                y=0.5
            ),
            width=width,
            height=height,
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='closest'
        )
    #
    def create_enhanced_vicinity_sunburst(self, angles, stresses, stress_component='sigma_hydro',
                                         title="Habit Plane Vicinity Analysis",
                                         radius_scale=1.0, show_grid=True):
        """Enhanced sunburst chart with publication quality"""
       
        # Ensure angles are numpy arrays
        angles = np.array(angles)
        stresses = np.array(stresses)
       
        # Check if data is empty
        if len(angles) == 0 or len(stresses) == 0:
            return self._create_empty_figure(title)
       
        # Filter to vicinity
        vicinity_range = 45.0
        mask = (angles >= self.habit_angle - vicinity_range) & (angles <= self.habit_angle + vicinity_range)
        vic_angles = angles[mask]
        vic_stresses = stresses[mask]
       
        if len(vic_angles) == 0:
            return self._create_empty_figure(title)
       
        # Create polar plot
        fig = go.Figure()
       
        # Add main stress distribution with enhanced styling
        fig.add_trace(go.Scatterpolar(
            r=vic_stresses * radius_scale,
            theta=vic_angles,
            mode='markers+lines',
            marker=dict(
                size=self.marker_size + 4,
                color=vic_stresses,
                colorscale='RdBu_r',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(size=self.legend_font_size)
                    ),
                    x=1.1,
                    thickness=25,
                    len=0.8
                ),
                line=dict(width=2, color='black'),
                symbol='circle'
            ),
            line=dict(
                color='rgba(100, 100, 100, 0.5)',
                width=self.line_width,
                shape='spline'
            ),
            name='Stress Distribution',
            hovertemplate=(
                '<b>Orientation:</b> %{theta:.1f}°<br>' +
                '<b>Stress:</b> %{r:.3f} GPa<br>' +
                '<b>Component:</b> ' + stress_component.replace('_', ' ').title() +
                '<extra></extra>'
            )
        ))
       
        # Highlight habit plane with enhanced styling
        habit_idx = np.argmin(np.abs(vic_angles - self.habit_angle))
        if habit_idx < len(vic_stresses):
            habit_stress = vic_stresses[habit_idx]
            fig.add_trace(go.Scatterpolar(
                r=[habit_stress * radius_scale],
                theta=[vic_angles[habit_idx]],
                mode='markers+text',
                marker=dict(
                    size=self.marker_size * 3,
                    color='rgb(46, 204, 113)',
                    symbol='star',
                    line=dict(width=3, color='black')
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(
                    size=self.annotation_font_size + 4,
                    color='black',
                    family='Arial Black',
                    weight='bold'
                ),
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate=(
                    f'<b>Habit Plane ({self.habit_angle}°)</b><br>' +
                    f'<b>Stress:</b> {habit_stress:.3f} GPa<br>' +
                    '<extra></extra>'
                )
            ))
       
        # Create publication quality layout
        layout = self.create_publication_plotly_layout(title=title)
       
        # Enhanced polar layout
        layout.update(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.5)" if show_grid else "transparent",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(
                        size=self.tick_label_font_size,
                        color='black',
                        family='Arial'
                    ),
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(
                            size=self.axis_label_font_size,
                            color='black',
                            family='Arial',
                            weight='bold'
                        )
                    ),
                    range=[0, max(vic_stresses) * radius_scale * 1.2] if len(vic_stresses) > 0 else [0, 1]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.5)" if show_grid else "transparent",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)) if len(vic_angles) > 0 else [],
                    ticktext=[f'{i}°' for i in range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)] if len(vic_angles) > 0 else [],
                    tickfont=dict(
                        size=self.tick_label_font_size,
                        color='black',
                        family='Arial'
                    ),
                    period=360,
                    thetaunit="degrees"
                ),
                bgcolor="rgba(240, 240, 240, 0.1)",
                sector=[vic_angles[0], vic_angles[-1]] if len(vic_angles) > 0 else [0, 360]
            ),
            showlegend=True,
            legend=dict(
                x=1.15,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(
                    size=self.legend_font_size,
                    family='Arial',
                    color='black'
                ),
                title=dict(
                    text='Legend',
                    font=dict(
                        size=self.legend_font_size + 2,
                        weight='bold'
                    )
                )
            ),
            margin=dict(l=100, r=200, t=120, b=100)
        )
       
        fig.update_layout(layout)
        
        # Add annotation for angular axis label
        fig.add_annotation(
            dict(
                font=dict(
                    size=self.axis_label_font_size,
                    color='black',
                    family='Arial',
                    weight='bold'
                ),
                text='Orientation (°)',
                x=0.5,
                y=-0.1,
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='top'
            )
        )
        
        return fig
    
    
    def create_enhanced_stress_comparison_radar(self, comparison_data, 
                                               title="Stress Component Comparison",
                                               show_grid=True, show_legend=True):
        """Enhanced radar chart with publication quality and better visibility"""
        
        fig = go.Figure()
        
        # Check if comparison_data is empty
        if not comparison_data:
            return self._create_empty_figure(title)
        
        # Track max stress for axis scaling
        max_stress = 0
        
        # Add traces for each entry in comparison_data with enhanced styling
        for entry_name, entry_data in comparison_data.items():
            if not isinstance(entry_data, dict):
                continue
                
            if 'angles' not in entry_data or 'stresses' not in entry_data:
                continue
            
            try:
                angles = np.array(entry_data['angles'])
                stresses = entry_data['stresses']
                
                if stresses is None:
                    continue
                
                # Convert stresses to numpy array
                if isinstance(stresses, (list, np.ndarray)):
                    stresses_array = np.array(stresses)
                elif isinstance(stresses, dict) and stresses:
                    first_key = list(stresses.keys())[0]
                    if isinstance(stresses[first_key], (list, np.ndarray)):
                        stresses_array = np.array(stresses[first_key])
                    else:
                        continue
                else:
                    continue
                
                # Check if arrays are empty
                if len(angles) == 0 or len(stresses_array) == 0:
                    continue
                
                # Check array lengths
                if len(angles) != len(stresses_array):
                    min_len = min(len(angles), len(stresses_array))
                    if min_len == 0:
                        continue
                    angles = angles[:min_len]
                    stresses_array = stresses_array[:min_len]
                
                # Close the loop for radar chart
                angles_closed = np.append(angles, angles[0])
                stresses_closed = np.append(stresses_array, stresses_array[0])
                
                # Update max stress
                try:
                    current_max = np.nanmax(stresses_array)
                    if not np.isnan(current_max):
                        max_stress = max(max_stress, current_max)
                except:
                    pass
                
                # Get color and name
                color = entry_data.get('color', 'rgba(100, 100, 100, 0.2)')
                display_name = entry_data.get('defect_type', entry_name)
                if 'component' in entry_data:
                    display_name = f"{display_name} - {entry_data['component']}"
                
                # Add trace with enhanced styling
                fig.add_trace(go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                    line=dict(
                        color=color,
                        width=self.line_width,
                        dash=None if 'twin' in display_name.lower() else 'dash'
                    ),
                    marker=dict(
                        size=self.marker_size,
                        color=color,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    name=display_name,
                    hovertemplate=(
                        '<b>Orientation:</b> %{theta:.1f}°<br>' +
                        '<b>Stress:</b> %{r:.3f} GPa<br>' +
                        f'<b>Type:</b> {display_name}' +
                        '<extra></extra>'
                    ),
                    showlegend=show_legend
                ))
            except Exception as e:
                print(f"Error processing entry {entry_name}: {e}")
                continue
        
        # If no traces were added
        if len(fig.data) == 0:
            return self._create_empty_figure(title)
        
        # Highlight habit plane with enhanced styling
        if max_stress > 0:
            fig.add_trace(go.Scatterpolar(
                r=[0, max_stress * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(
                    color='rgb(46, 204, 113)',
                    width=4,
                    dash='dashdot'
                ),
                name=f'Habit Plane ({self.habit_angle}°)',
                hoverinfo='skip',
                showlegend=show_legend
            ))
        
        # Create publication quality layout
        layout = self.create_publication_plotly_layout(title=title)
        
        # Enhanced polar layout for radar chart
        layout.update(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.5)" if show_grid else "transparent",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(
                        size=self.tick_label_font_size,
                        color='black',
                        family='Arial'
                    ),
                    title=dict(
                        text='Stress (GPa)',
                        font=dict(
                            size=self.axis_label_font_size,
                            color='black',
                            family='Arial',
                            weight='bold'
                        )
                    ),
                    range=[0, max_stress * 1.2 if max_stress > 0 else 1],
                    tickangle=0,
                    tickformat='.1f'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.5)" if show_grid else "transparent",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                    tickfont=dict(
                        size=self.tick_label_font_size,
                        color='black',
                        family='Arial'
                    ),
                    title=dict(
                        text='Orientation (°)',
                        font=dict(
                            size=self.axis_label_font_size,
                            color='black',
                            family='Arial',
                            weight='bold'
                        )
                    ),
                    period=360
                ),
                bgcolor="rgba(240, 240, 240, 0.05)",
                hole=0.1  # Creates a donut-like radar for better label visibility
            ),
            showlegend=show_legend,
            legend=dict(
                x=1.2 if show_legend else 0,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(
                    size=self.legend_font_size,
                    family='Arial',
                    color='black'
                ),
                title=dict(
                    text='Defect Types',
                    font=dict(
                        size=self.legend_font_size + 2,
                        weight='bold'
                    )
                ),
                itemsizing='constant'
            ),
            margin=dict(l=150, r=250 if show_legend else 100, t=120, b=150)
        )
        
        # Add annotations for clarity
        if len(fig.data) > 0:
            layout.update(
                annotations=[
                    dict(
                        text="Radar View: Closer to center = lower stress",
                        x=0.5,
                        y=-0.15,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=self.annotation_font_size, color='gray'),
                        align="center"
                    )
                ]
            )
        
        fig.update_layout(layout)
        
        # Add interactive features
        fig.update_traces(
            hoverinfo="text",
            hovertemplate=(
                '<b>%{fullData.name}</b><br>' +
                'Orientation: %{theta:.1f}°<br>' +
                'Stress: %{r:.3f} GPa<br>' +
                '<extra></extra>'
            )
        )
        
        return fig
    
    def create_enhanced_sintering_temperature_chart(self, vicinity_sweep, defect_type='Twin',
                                                   title="Sintering Temperature Analysis"):
        """Create enhanced temperature charts with publication quality"""
        
        if not vicinity_sweep or 'angles' not in vicinity_sweep:
            return self._create_empty_figure(title)
        
        angles = np.array(vicinity_sweep['angles'])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
        
        # Plot 1: Temperature vs Orientation
        if 'sintering_temps' in vicinity_sweep:
            temps_exp = vicinity_sweep['sintering_temps'].get('exponential', [])
            temps_arr = vicinity_sweep['sintering_temps'].get('arrhenius_defect', [])
            
            if len(temps_exp) == len(angles):
                ax1.plot(angles, temps_exp, 
                        color='red', 
                        linewidth=self.line_width + 1,
                        marker='o',
                        markersize=self.marker_size,
                        markevery=5,
                        label='Exponential Model',
                        zorder=3)
            
            if len(temps_arr) == len(angles):
                ax1.plot(angles, temps_arr,
                        color='blue',
                        linewidth=self.line_width + 1,
                        linestyle='--',
                        marker='s',
                        markersize=self.marker_size,
                        markevery=5,
                        label='Arrhenius Model (Defect)',
                        zorder=3)
        
        # Highlight habit plane
        ax1.axvline(self.habit_angle, color='green', linestyle=':', 
                   linewidth=self.line_width, label=f'Habit Plane ({self.habit_angle}°)',
                   zorder=2)
        
        # Add fill for System classification
        ax1.axvspan(0, 5, alpha=0.1, color='green', label='System 1 (Perfect)', zorder=1)
        ax1.axvspan(5, 20, alpha=0.1, color='orange', label='System 2 (SF/Twin)', zorder=1)
        ax1.axvspan(20, 35, alpha=0.1, color='red', label='System 3 (Plastic)', zorder=1)
        
        # Apply publication quality formatting
        ax1 = self._format_axis_for_publication(
            ax1,
            title='Sintering Temperature vs Orientation',
            xlabel='',
            ylabel='Temperature (K)',
            grid=True
        )
        
        # Add second y-axis for Celsius
        ax1_celsius = ax1.twinx()
        ylim_k = ax1.get_ylim()
        ax1_celsius.set_ylim(ylim_k[0] - 273.15, ylim_k[1] - 273.15)
        ax1_celsius.set_ylabel('Temperature (°C)', 
                              fontsize=self.axis_label_font_size,
                              fontweight='bold')
        ax1_celsius.tick_params(axis='y', labelsize=self.tick_label_font_size)
        
        # Plot 2: Stress vs Temperature relationship
        if 'stresses' in vicinity_sweep and 'sigma_hydro' in vicinity_sweep['stresses']:
            stresses = vicinity_sweep['stresses']['sigma_hydro']
            if len(stresses) == len(angles):
                # Color by stress magnitude
                colors = stresses
                scatter = ax2.scatter(angles, stresses, 
                                     c=colors, 
                                     cmap='RdBu_r',
                                     s=self.marker_size * 20,
                                     edgecolor='black',
                                     linewidth=1,
                                     alpha=0.8,
                                     zorder=3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Hydrostatic Stress (GPa)', 
                              fontsize=self.axis_label_font_size,
                              fontweight='bold')
                cbar.ax.tick_params(labelsize=self.tick_label_font_size)
        
        # Highlight habit plane
        ax2.axvline(self.habit_angle, color='green', linestyle=':', 
                   linewidth=self.line_width,
                   zorder=2)
        
        # Apply publication quality formatting
        ax2 = self._format_axis_for_publication(
            ax2,
            title='Hydrostatic Stress Distribution',
            xlabel='Orientation (°)',
            ylabel='Stress (GPa)',
            grid=True
        )
        
        # Add annotations
        if len(temps_arr) == len(angles):
            habit_idx = np.argmin(np.abs(angles - self.habit_angle))
            habit_temp = temps_arr[habit_idx]
            ax1.annotate(f'Habit Plane: {habit_temp:.0f} K\n({habit_temp-273.15:.0f}°C)',
                        xy=(self.habit_angle, habit_temp),
                        xytext=(self.habit_angle + 5, habit_temp + 50),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=self.annotation_font_size,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                        zorder=4)
        
        plt.suptitle(f'{title} - {defect_type}', 
                    fontsize=self.title_font_size + 4,
                    fontweight='bold',
                    y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    #
    def create_enhanced_defect_comparison_radar(self, defect_comparison,
                                               stress_component='sigma_hydro',
                                               title="Defect Type Comparison Radar"):
        """Enhanced defect comparison radar with publication quality"""
       
        fig = go.Figure()
       
        if not defect_comparison:
            return self._create_empty_figure(title)
       
        # Collect all data for consistent scaling
        all_stresses = []
        for key, data in defect_comparison.items():
            if isinstance(data, dict) and 'stresses' in data and stress_component in data['stresses']:
                stresses = data['stresses'][stress_component]
                if stresses:
                    all_stresses.extend(stresses)
       
        if not all_stresses:
            return self._create_empty_figure(title)
       
        max_stress = max(all_stresses) if all_stresses else 1
       
        # Add traces for each defect type
        for key, data in defect_comparison.items():
            if not isinstance(data, dict):
                continue
               
            defect_type = data.get('defect_type', 'Unknown')
            if 'angles' in data and 'stresses' in data and stress_component in data['stresses']:
                angles = data['angles']
                stresses = data['stresses'][stress_component]
               
                if angles is None or stresses is None:
                    continue
               
                try:
                    angles_array = np.array(angles)
                    stresses_array = np.array(stresses)
                except:
                    continue
               
                if len(angles_array) == 0 or len(stresses_array) == 0:
                    continue
               
                # Close the loop
                angles_closed = np.append(angles_array, angles_array[0])
                stresses_closed = np.append(stresses_array, stresses_array[0])
               
                color = data.get('color', self.defect_colors.get(defect_type, 'black'))
               
                fig.add_trace(go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.15)'),
                    line=dict(
                        color=color,
                        width=self.line_width,
                        dash='solid'
                    ),
                    marker=dict(
                        size=self.marker_size,
                        color=color,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                    hovertemplate=(
                        f'<b>{defect_type}</b><br>' +
                        'Orientation: %{theta:.1f}°<br>' +
                        f'{stress_component.replace("_", " ").title()}: %{r:.3f} GPa<br>' +
                        f"Eigen Strain: {data.get('eigen_strain', 0):.2f}" +
                        '<extra></extra>'
                    ),
                    showlegend=True
                ))
       
        if len(fig.data) == 0:
            return self._create_empty_figure(title)
       
        # Highlight habit plane
        fig.add_trace(go.Scatterpolar(
            r=[0, max_stress * 1.2],
            theta=[self.habit_angle, self.habit_angle],
            mode='lines',
            line=dict(
                color='rgb(46, 204, 113)',
                width=4,
                dash='dashdot'
            ),
            name=f'Habit Plane ({self.habit_angle}°)',
            hoverinfo='skip',
            showlegend=True
        ))
       
        # Create publication quality layout
        layout = self.create_publication_plotly_layout(title=f"{title} - {stress_component.replace('_', ' ').title()}")
       
        # Enhanced radar layout
        layout.update(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.5)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(
                        size=self.tick_label_font_size,
                        color='black',
                        family='Arial'
                    ),
                    title=dict(
                        text=f'{stress_component.replace("_", " ").title()} Stress (GPa)',
                        font=dict(
                            size=self.axis_label_font_size,
                            color='black',
                            family='Arial',
                            weight='bold'
                        )
                    ),
                    range=[0, max_stress * 1.2],
                    tickangle=45,
                    tickformat='.2f',
                    ticksuffix=' GPa'
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.5)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                    tickfont=dict(
                        size=self.tick_label_font_size,
                        color='black',
                        family='Arial'
                    ),
                    period=360
                ),
                bgcolor="rgba(240, 240, 240, 0.05)",
                hole=0.15 # Larger hole for better label visibility
            ),
            showlegend=True,
            legend=dict(
                x=1.25,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(
                    size=self.legend_font_size,
                    family='Arial',
                    color='black'
                ),
                title=dict(
                    text='Defect Types',
                    font=dict(
                        size=self.legend_font_size + 2,
                        weight='bold'
                    )
                ),
                itemwidth=30,
                itemsizing='constant'
            ),
            margin=dict(l=200, r=300, t=150, b=200)
        )
       
        # Add annotation for angular axis label
        fig.add_annotation(
            dict(
                font=dict(
                    size=self.axis_label_font_size,
                    color='black',
                    family='Arial',
                    weight='bold'
                ),
                text='Orientation (°)',
                x=0.5,
                y=-0.1,
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='top'
            )
        )
       
        fig.update_layout(layout)
       
        # Add interactive features
        fig.update_traces(
            hoverinfo="text",
            hovertemplate=(
                '<b>%{fullData.name}</b><br>' +
                'Orientation: %{theta:.1f}°<br>' +
                'Stress: %{r:.3f} GPa<br>' +
                '<extra></extra>'
            )
        )
       
        return fig
    
        
    def create_defect_comparison_plot(self, defect_comparison, stress_component='sigma_hydro',
                                     title="Defect Type Comparison"):
        """Create comparison plot for different defect types"""
        
        fig = go.Figure()
        
        # Check if defect_comparison is empty
        if not defect_comparison:
            fig.update_layout(
                title=dict(
                    text=f"{title} - No Data Available",
                    font=dict(size=self.title_font_size, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text="No defect comparison data available",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=self.annotation_font_size)
                    )
                ],
                width=1000,
                height=600
            )
            return fig
        
        # Add traces for each defect type
        for defect_key, data in defect_comparison.items():
            if not isinstance(data, dict):
                continue
                
            if 'angles' in data and 'stresses' in data and stress_component in data['stresses']:
                defect_type = data.get('defect_type', 'Unknown')
                angles = data['angles']
                stresses = data['stresses'][stress_component]
                
                # Check if data is valid
                if angles is None or stresses is None:
                    continue
                    
                # Convert to arrays
                try:
                    angles_array = np.array(angles)
                    stresses_array = np.array(stresses)
                except:
                    continue
                
                # Check if arrays are empty
                if len(angles_array) == 0 or len(stresses_array) == 0:
                    continue
                
                color = data.get('color', self.defect_colors.get(defect_type, 'black'))
                
                fig.add_trace(go.Scatter(
                    x=angles_array,
                    y=stresses_array,
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(size=self.marker_size, color=color),
                    name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                    hovertemplate='Orientation: %{x:.2f}°<br>Stress: %{y:.4f} GPa<extra></extra>',
                    showlegend=True
                ))
        
        # If no traces were added, return empty figure with message
        if len(fig.data) == 0:
            fig.update_layout(
                title=dict(
                    text=f"{title} - No Data Available",
                    font=dict(size=self.title_font_size, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text=f"No data available for stress component: {stress_component}",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=self.annotation_font_size)
                    )
                ],
                width=1000,
                height=600
            )
            return fig
        
        # Highlight habit plane
        fig.add_vline(x=self.habit_angle, line_width=3, line_dash="dashdot",
                     line_color="green", annotation_text=f"Habit Plane ({self.habit_angle}°)",
                     annotation_position="top right")
        
        # Add system boundaries
        fig.add_vrect(x0=0, x1=5, fillcolor="green", opacity=0.1, line_width=0,
                     annotation_text="System 1", annotation_position="top left")
        fig.add_vrect(x0=5, x1=20, fillcolor="orange", opacity=0.1, line_width=0,
                     annotation_text="System 2", annotation_position="top left")
        fig.add_vrect(x0=20, x1=35, fillcolor="red", opacity=0.1, line_width=0,
                     annotation_text="System 3", annotation_position="top left")
        
        # Update layout with publication quality
        layout = self.create_publication_plotly_layout(
            title=f"{title} - {stress_component.replace('_', ' ').title()}"
        )
        
        layout.update(
            xaxis=dict(
                title=dict(text='Orientation (°)', font=dict(size=self.axis_label_font_size, color='black')),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1,
                range=[0, 360],
                tickfont=dict(size=self.tick_label_font_size)
            ),
            yaxis=dict(
                title=dict(text=f'{stress_component.replace("_", " ").title()} Stress (GPa)',
                          font=dict(size=self.axis_label_font_size, color='black')),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1,
                tickfont=dict(size=self.tick_label_font_size)
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=self.legend_font_size, family='Arial')
            ),
            width=1200,
            height=700,
            hovermode='x unified'
        )
        
        fig.update_layout(layout)
        
        return fig
    
    def create_vicinity_sunburst(self, angles, stresses, stress_component='sigma_hydro',
                                title="Habit Plane Vicinity Analysis", radius_scale=1.0):
        """Create sunburst chart focused on habit plane vicinity"""
        
        # Ensure angles are numpy arrays
        angles = np.array(angles)
        stresses = np.array(stresses)
        
        # Check if data is empty
        if len(angles) == 0 or len(stresses) == 0:
            # Return empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text=f"{title} - No Data Available",
                    font=dict(size=self.title_font_size, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text="No data available for the selected vicinity range",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=self.annotation_font_size)
                    )
                ],
                width=900,
                height=700
            )
            return fig
        
        # Filter to vicinity (habit_angle ± 45° for better visualization)
        vicinity_range = 45.0
        mask = (angles >= self.habit_angle - vicinity_range) & (angles <= self.habit_angle + vicinity_range)
        vic_angles = angles[mask]
        vic_stresses = stresses[mask]
        
        if len(vic_angles) == 0:
            # Return empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text=f"{title} - No Data in Vicinity",
                    font=dict(size=self.title_font_size, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text=f"No data in vicinity range ({self.habit_angle - vicinity_range:.1f}° to {self.habit_angle + vicinity_range:.1f}°)",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=self.annotation_font_size)
                    )
                ],
                width=900,
                height=700
            )
            return fig
        
        # Create polar plot
        fig = go.Figure()
        
        # Add main stress distribution
        fig.add_trace(go.Scatterpolar(
            r=vic_stresses * radius_scale,
            theta=vic_angles,
            mode='markers+lines',
            marker=dict(
                size=self.marker_size,
                color=vic_stresses,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(
                    title=f"{stress_component.replace('_', ' ').title()} (GPa)",
                    x=1.1,
                    thickness=20
                ),
                line=dict(width=1, color='black')
            ),
            line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
            name='Stress Distribution',
            hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>'
        ))
        
        # Highlight habit plane
        habit_idx = np.argmin(np.abs(vic_angles - self.habit_angle))
        if habit_idx < len(vic_stresses):
            habit_stress = vic_stresses[habit_idx]
            fig.add_trace(go.Scatterpolar(
                r=[habit_stress * radius_scale],
                theta=[vic_angles[habit_idx]],
                mode='markers+text',
                marker=dict(
                    size=self.marker_size * 2.5,
                    color='rgb(46, 204, 113)',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(size=self.annotation_font_size, color='black', family='Arial Black'),
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate=f'Habit Plane ({self.habit_angle}°)<br>Stress: {habit_stress:.4f} GPa<extra></extra>'
            ))
        
        # Add radial lines for important orientations
        important_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]
        for angle in important_angles:
            if angle >= vic_angles[0] and angle <= vic_angles[-1]:
                fig.add_trace(go.Scatterpolar(
                    r=[0, max(vic_stresses) * radius_scale * 1.1],
                    theta=[angle, angle],
                    mode='lines',
                    line=dict(color='rgba(255, 0, 0, 0.1)', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.title_font_size, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(size=self.tick_label_font_size, color='black'),
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(size=self.axis_label_font_size, color='black')
                    ),
                    range=[0, max(vic_stresses) * radius_scale * 1.2] if len(vic_stresses) > 0 else [0, 1]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)) if len(vic_angles) > 0 else [],
                    ticktext=[f'{i}°' for i in range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)] if len(vic_angles) > 0 else [],
                    tickfont=dict(size=self.tick_label_font_size, color='black'),
                    period=360,
                    thetaunit="degrees"
                ),
                bgcolor="rgba(240, 240, 240, 0.3)",
                sector=[vic_angles[0], vic_angles[-1]] if len(vic_angles) > 0 else [0, 360]
            ),
            showlegend=True,
            legend=dict(
                x=1.2,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=self.legend_font_size, family='Arial')
            ),
            width=900,
            height=700,
            margin=dict(l=100, r=200, t=100, b=100)
        )
        
        return fig
    
    def _create_empty_figure(self, title="No Data"):
        """Create an empty figure with message"""
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.title_font_size, family="Arial", color='darkgray'),
                x=0.5
            ),
            annotations=[
                dict(
                    text="No data available",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=self.annotation_font_size, color='gray')
                )
            ],
            width=800,
            height=600,
            paper_bgcolor='white'
        )
        return fig
    
    def _format_axis_for_publication(self, ax, title="", xlabel="", ylabel="", grid=True):
        """Format axis for publication quality"""
        ax.set_title(title, fontsize=self.title_font_size, fontweight='bold', pad=15)
        ax.set_xlabel(xlabel, fontsize=self.axis_label_font_size, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=self.axis_label_font_size, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', 
                      labelsize=self.tick_label_font_size, width=2)
        ax.tick_params(axis='both', which='minor', 
                      labelsize=self.tick_label_font_size-2, width=1)
        
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        if ax.get_legend():
            ax.legend(fontsize=self.legend_font_size, 
                     frameon=True, 
                     framealpha=0.9,
                     edgecolor='black',
                     loc='best')
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        return ax

# =============================================
# ENHANCED RESULTS MANAGER WITH COMPREHENSIVE EXPORT
# =============================================

class EnhancedResultsManager:
    """Enhanced results manager with comprehensive export capabilities"""
    
    def __init__(self):
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        self.sintering_calculator = EnhancedSinteringCalculator()
    
    def prepare_vicinity_analysis_report(self, vicinity_sweep, defect_comparison,
                                        target_params, analysis_params):
        """Prepare comprehensive vicinity analysis report"""
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'habit_plane_vicinity',
                'software_version': '5.0.0',
                'habit_plane_angle': 54.7,
                'description': 'Comprehensive Ag FCC twin habit plane vicinity analysis'
            },
            'analysis_parameters': {
                'target_params': target_params,
                'analysis_params': analysis_params,
                'vicinity_range': analysis_params.get('vicinity_range', 10.0),
                'n_points': analysis_params.get('n_points', 50),
                'region_type': analysis_params.get('region_type', 'bulk')
            },
            'vicinity_sweep': self._prepare_vicinity_sweep_data(vicinity_sweep),
            'defect_comparison': self._prepare_defect_comparison_data(defect_comparison),
            'physics_analysis': self._prepare_physics_analysis(vicinity_sweep, target_params),
            'sintering_analysis': self._prepare_sintering_analysis(vicinity_sweep, target_params),
            'statistics': self._calculate_comprehensive_statistics(vicinity_sweep, defect_comparison)
        }
        
        return report
    
    def _prepare_vicinity_sweep_data(self, vicinity_sweep):
        """Prepare vicinity sweep data for export"""
        if not vicinity_sweep:
            return {}
        
        data = {
            'angles': vicinity_sweep.get('angles', []),
            'stresses': vicinity_sweep.get('stresses', {}),
            'sintering_temps': vicinity_sweep.get('sintering_temps', {}),
            'defect_type': vicinity_sweep.get('defect_type', 'Unknown')
        }
        
        # Add habit plane specific data
        angles = np.array(data['angles'])
        habit_angle = 54.7
        
        if len(angles) > 0:
            habit_idx = np.argmin(np.abs(angles - habit_angle))
            if habit_idx < len(angles):
                data['habit_plane_data'] = {
                    'angle': float(angles[habit_idx]),
                    'sigma_hydro': float(data['stresses'].get('sigma_hydro', [])[habit_idx]),
                    'von_mises': float(data['stresses'].get('von_mises', [])[habit_idx]),
                    'sigma_mag': float(data['stresses'].get('sigma_mag', [])[habit_idx]),
                    'T_sinter_exponential': float(data['sintering_temps'].get('exponential', [])[habit_idx]),
                    'T_sinter_arrhenius': float(data['sintering_temps'].get('arrhenius_defect', [])[habit_idx])
                }
        
        return data
    
    def _prepare_defect_comparison_data(self, defect_comparison):
        """Prepare defect comparison data for export"""
        if not defect_comparison:
            return {}
        
        data = {}
        for key, comparison in defect_comparison.items():
            data[key] = {
                'defect_type': comparison.get('defect_type'),
                'shape': comparison.get('shape'),
                'eigen_strain': comparison.get('eigen_strain', 0.0),
                'angles': comparison.get('angles', []),
                'stresses': comparison.get('stresses', {}),
                'sintering_temps': comparison.get('sintering_temps', []),
                'color': comparison.get('color')
            }
        
        return data
    
    def _prepare_physics_analysis(self, vicinity_sweep, target_params):
        """Prepare physics analysis for export"""
        analysis = {
            'defect_parameters': {
                'type': target_params.get('defect_type', 'Unknown'),
                'eigen_strain': target_params.get('eps0', 0.0),
                'description': self._get_defect_description(target_params.get('defect_type', 'Unknown'))
            },
            'crystal_orientation': {
                'habit_plane_angle': 54.7,
                'schmid_factor': 0.5,
                'primary_slip_systems': 12
            },
            'material_properties': self.physics_analyzer.ag_properties
        }
        
        # Add stress intensity factors if available
        if 'stresses' in vicinity_sweep and 'sigma_hydro' in vicinity_sweep['stresses']:
            stresses = vicinity_sweep['stresses']['sigma_hydro']
            if stresses:
                max_stress = max(stresses)
                K = self.physics_analyzer.compute_stress_intensity_factor(
                    {'sigma_hydro': {'max_abs': max_stress}},
                    target_params.get('eps0', 0.0),
                    target_params.get('defect_type', 'Twin')
                )
                analysis['fracture_mechanics'] = {
                    'max_stress_intensity_factor_mpa_sqrtm': float(K),
                    'max_hydrostatic_stress_gpa': float(max_stress)
                }
        
        return analysis
    
    def _prepare_sintering_analysis(self, vicinity_sweep, target_params):
        """Prepare sintering analysis for export"""
        analysis = {
            'model_parameters': {
                'T0_k': self.sintering_calculator.T0,
                'T_min_k': self.sintering_calculator.T_min,
                'sigma_peak_gpa': self.sintering_calculator.sigma_peak,
                'beta': self.sintering_calculator.beta,
                'G_gpa': self.sintering_calculator.G,
                'Q_a_eV': self.sintering_calculator.Q_a,
                'omega_eV_per_GPa': self.sintering_calculator.omega_eV_per_GPa
            },
            'defect_specific_parameters': self.sintering_calculator.defect_diffusion_params
        }
        
        # Add habit plane sintering data
        if 'angles' in vicinity_sweep and 'sintering_temps' in vicinity_sweep:
            angles = vicinity_sweep['angles']
            sintering_temps = vicinity_sweep['sintering_temps']
            habit_angle = 54.7
            
            if len(angles) > 0 and 'arrhenius_defect' in sintering_temps:
                habit_idx = np.argmin(np.abs(np.array(angles) - habit_angle))
                if habit_idx < len(sintering_temps['arrhenius_defect']):
                    analysis['habit_plane_sintering'] = {
                        'angle_deg': float(angles[habit_idx]),
                        'T_sinter_arrhenius_k': float(sintering_temps['arrhenius_defect'][habit_idx]),
                        'T_sinter_arrhenius_c': float(sintering_temps['arrhenius_defect'][habit_idx] - 273.15),
                        'T_sinter_exponential_k': float(sintering_temps.get('exponential', [])[habit_idx]),
                        'temperature_reduction_k': float(self.sintering_calculator.T0 - sintering_temps['arrhenius_defect'][habit_idx]),
                        'temperature_reduction_percent': float((self.sintering_calculator.T0 - sintering_temps['arrhenius_defect'][habit_idx]) / self.sintering_calculator.T0 * 100)
                    }
        
        return analysis
    
    def _calculate_comprehensive_statistics(self, vicinity_sweep, defect_comparison):
        """Calculate comprehensive statistics"""
        stats = {
            'vicinity_sweep': {},
            'defect_comparison': {},
            'sintering_analysis': {}
        }
        
        # Vicinity sweep statistics
        if vicinity_sweep and 'stresses' in vicinity_sweep:
            for comp, stresses in vicinity_sweep['stresses'].items():
                if stresses:
                    stats['vicinity_sweep'][comp] = {
                        'min': float(np.min(stresses)),
                        'max': float(np.max(stresses)),
                        'mean': float(np.mean(stresses)),
                        'std': float(np.std(stresses)),
                        'range': float(np.max(stresses) - np.min(stresses))
                    }
            
            if 'sintering_temps' in vicinity_sweep:
                for model, temps in vicinity_sweep['sintering_temps'].items():
                    if temps:
                        stats['sintering_analysis'][model] = {
                            'min_k': float(np.min(temps)),
                            'max_k': float(np.max(temps)),
                            'mean_k': float(np.mean(temps)),
                            'range_k': float(np.max(temps) - np.min(temps)),
                            'min_c': float(np.min(temps) - 273.15),
                            'max_c': float(np.max(temps) - 273.15)
                        }
        
        # Defect comparison statistics
        if defect_comparison:
            for key, data in defect_comparison.items():
                if 'stresses' in data and 'sigma_hydro' in data['stresses']:
                    stresses = data['stresses']['sigma_hydro']
                    if stresses:
                        stats['defect_comparison'][key] = {
                            'defect_type': data.get('defect_type', 'Unknown'),
                            'max_stress_gpa': float(np.max(stresses)),
                            'mean_stress_gpa': float(np.mean(stresses)),
                            'stress_range_gpa': float(np.max(stresses) - np.min(stresses))
                        }
        
        return stats
    
    def _get_defect_description(self, defect_type):
        """Get description for defect type"""
        descriptions = {
            'ISF': 'Intrinsic Stacking Fault - Missing one {111} atomic plane',
            'ESF': 'Extrinsic Stacking Fault - Extra {111} atomic plane',
            'Twin': 'Coherent Twin Boundary - Mirror symmetry across {111} plane',
            'No Defect': 'Perfect Face-Centered Cubic Crystal'
        }
        return descriptions.get(defect_type, 'Unknown defect type')
    
    def create_comprehensive_export(self, report, include_raw_data=True):
        """Create comprehensive export package"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        defect_type = report['analysis_parameters']['target_params'].get('defect_type', 'unknown')
        
        # Create ZIP archive
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # 1. Main JSON report
            report_json = json.dumps(report, indent=2, default=self._json_serializer)
            zip_file.writestr(f'vicinity_analysis_{defect_type}_{timestamp}.json', report_json)
            
            # 2. CSV data files
            self._add_csv_files(zip_file, report, timestamp)
            
            # 3. README with analysis details
            readme = self._create_readme(report)
            zip_file.writestr('README.txt', readme)
            
            # 4. Python script for data processing
            script = self._create_processing_script()
            zip_file.writestr('process_data.py', script)
            
            # 5. Configuration file
            config = self._create_config_file(report)
            zip_file.writestr('analysis_config.json', json.dumps(config, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def _add_csv_files(self, zip_file, report, timestamp):
        """Add CSV data files to ZIP archive"""
        
        # Vicinity sweep data
        if 'vicinity_sweep' in report:
            sweep_data = report['vicinity_sweep']
            
            rows = []
            if 'angles' in sweep_data:
                angles = sweep_data['angles']
                n_points = len(angles)
                
                for i in range(n_points):
                    row = {
                        'angle_deg': angles[i],
                        'angle_rad': np.radians(angles[i])
                    }
                    
                    # Add stresses
                    for comp, stresses in sweep_data.get('stresses', {}).items():
                        if i < len(stresses):
                            row[f'{comp}_gpa'] = stresses[i]
                    
                    # Add sintering temperatures
                    for model, temps in sweep_data.get('sintering_temps', {}).items():
                        if i < len(temps):
                            row[f'T_sinter_{model}_k'] = temps[i]
                            row[f'T_sinter_{model}_c'] = temps[i] - 273.15
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                csv_data = df.to_csv(index=False)
                zip_file.writestr(f'vicinity_sweep_data_{timestamp}.csv', csv_data)
        
        # Defect comparison data
        if 'defect_comparison' in report:
            comparison_data = []
            for key, data in report['defect_comparison'].items():
                if 'angles' in data and 'stresses' in data:
                    angles = data['angles']
                    stresses = data['stresses']
                    
                    for i in range(min(len(angles), len(stresses.get('sigma_hydro', [])))):
                        row = {
                            'defect_type': data.get('defect_type', 'Unknown'),
                            'shape': data.get('shape', 'Unknown'),
                            'eigen_strain': data.get('eigen_strain', 0.0),
                            'angle_deg': angles[i]
                        }
                        
                        for comp, comp_stresses in stresses.items():
                            if i < len(comp_stresses):
                                row[f'{comp}_gpa'] = comp_stresses[i]
                        
                        comparison_data.append(row)
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                csv_data = df.to_csv(index=False)
                zip_file.writestr(f'defect_comparison_data_{timestamp}.csv', csv_data)
    
    def _create_readme(self, report):
        """Create README file with analysis details"""
        
        target_params = report['analysis_parameters']['target_params']
        analysis_params = report['analysis_parameters']['analysis_params']
        
        readme = f"""# AG FCC TWIN HABIT PLANE VICINITY ANALYSIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## ANALYSIS SUMMARY
- **Defect Type**: {target_params.get('defect_type', 'Unknown')}
- **Eigen Strain (ε*)**: {target_params.get('eps0', 0.0):.3f}
- **Shape**: {target_params.get('shape', 'Unknown')}
- **Region Analyzed**: {analysis_params.get('region_type', 'bulk')}
- **Vicinity Range**: ±{analysis_params.get('vicinity_range', 10.0)}° around habit plane
- **Number of Points**: {analysis_params.get('n_points', 50)}

## HABIT PLANE PHYSICS
- **Ag FCC Twin Habit Plane**: 54.7°
- **Crystal Planes**: {{111}} family
- **Schmid Factor for Twinning**: 0.5
- **Primary Slip Systems**: 12

## STRESS COMPONENTS ANALYZED
1. **Hydrostatic Stress (σ_hydro)**: Trace of stress tensor/3, critical for sintering
2. **Von Mises Stress (σ_vm)**: Equivalent tensile stress, indicates yield onset
3. **Stress Magnitude (σ_mag)**: Overall stress intensity

## SINTERING TEMPERATURE MODELS
1. **Exponential Model**: T_sinter(σ) = T₀ × exp(-β × |σ| / G)
   - T₀ = {self.sintering_calculator.T0} K (reference at σ=0)
   - β = {self.sintering_calculator.beta} (calibration factor)
   - G = {self.sintering_calculator.G} GPa (Ag shear modulus)

2. **Arrhenius Model (Defect-Specific)**: D = D₀ × exp[-(Q_a - ωσ)/(k_B T)]
   - Q_a = {self.sintering_calculator.Q_a} eV (activation energy)
   - ω = {self.sintering_calculator.omega_eV_per_GPa:.3f} eV/GPa (activation volume)
   - Defect-specific parameters applied based on defect type

## SYSTEM CLASSIFICATION
1. **System 1 (Perfect Crystal)**: σ < 5 GPa, T ≈ 620-630 K
2. **System 2 (Stacking Faults/Twins)**: 5 ≤ σ < 20 GPa, T ≈ 450-550 K
3. **System 3 (Plastic Deformation)**: σ ≥ 20 GPa, T ≈ 350-400 K

## FILES INCLUDED
1. JSON report - Complete analysis data
2. CSV files - Tabular data for plotting
3. This README - Analysis documentation
4. Python script - For data reprocessing
5. Configuration - Analysis parameters

## INTERPOLATION METHOD
- **Combined Attention & Gaussian Regularization**
- **Attention Blend Ratio**: {analysis_params.get('attention_blend', 0.7)}
- **Spatial Sigma (σ)**: {analysis_params.get('sigma', 0.3)}
- **Physics Constraints**: Applied based on eigen strain and defect type

## MATERIAL PROPERTIES (Ag)
- Shear Modulus: 30 GPa
- Young's Modulus: 83 GPa
- Poisson's Ratio: 0.37
- Lattice Constant: 0.408 nm
- Stacking Fault Energy: 22 mJ/m²

## CITATION
If you use this analysis in your research, please cite:
"Stress-Mediated Low-Temperature Sintering of Ag Nanoparticles via Defect Engineering"
"""
        return readme
    
    def _create_processing_script(self):
        """Create Python script for data processing"""
        
        script = """#!/usr/bin/env python3
'''
Data processing script for Ag FCC Twin Habit Plane Analysis
'''

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

def load_analysis_data(json_file):
    '''Load analysis data from JSON file'''
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_vicinity_sweep(data, output_file='vicinity_sweep.png'):
    '''Plot vicinity sweep data'''
    
    sweep = data.get('vicinity_sweep', {})
    if not sweep:
        print("No vicinity sweep data found")
        return
    
    angles = sweep.get('angles', [])
    if not angles:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot stresses
    if 'stresses' in sweep:
        stresses = sweep['stresses']
        ax = axes[0, 0]
        for comp, values in stresses.items():
            if len(values) == len(angles):
                ax.plot(angles, values, label=comp, linewidth=2)
        ax.set_xlabel('Orientation (°)')
        ax.set_ylabel('Stress (GPa)')
        ax.set_title('Stress Components vs Orientation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(54.7, color='green', linestyle='--', label='Habit Plane')
    
    # Plot sintering temperatures
    if 'sintering_temps' in sweep:
        temps = sweep['sintering_temps']
        ax = axes[0, 1]
        for model, values in temps.items():
            if len(values) == len(angles):
                ax.plot(angles, values, label=model, linewidth=2)
        ax.set_xlabel('Orientation (°)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Sintering Temperature vs Orientation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(54.7, color='green', linestyle='--')
    
    # Plot stress vs temperature
    ax = axes[1, 0]
    if 'stresses' in sweep and 'sigma_hydro' in sweep['stresses']:
        stresses = sweep['stresses']['sigma_hydro']
        if 'sintering_temps' in sweep and 'arrhenius_defect' in sweep['sintering_temps']:
            temps = sweep['sintering_temps']['arrhenius_defect']
            if len(stresses) == len(temps):
                ax.scatter(stresses, temps, c=angles, cmap='viridis', s=50)
                ax.set_xlabel('Hydrostatic Stress (GPa)')
                ax.set_ylabel('Sintering Temperature (K)')
                ax.set_title('Stress vs Sintering Temperature')
                ax.grid(True, alpha=0.3)
    
    # Plot habit plane highlight
    ax = axes[1, 1]
    habit_angle = 54.7
    if angles:
        habit_idx = np.argmin(np.abs(np.array(angles) - habit_angle))
        ax.text(0.5, 0.5, 
                f'Habit Plane Analysis\\nAngle: {angles[habit_idx]:.1f}°\\nσ_hydro: {sweep["stresses"]["sigma_hydro"][habit_idx]:.3f} GPa\\nT_sinter: {sweep["sintering_temps"]["arrhenius_defect"][habit_idx]:.1f} K',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

def create_interactive_plot(data, output_file='interactive_plot.html'):
    '''Create interactive Plotly visualization'''
    
    sweep = data.get('vicinity_sweep', {})
    if not sweep or 'angles' not in sweep:
        return
    
    angles = sweep['angles']
    
    fig = go.Figure()
    
    # Add stress traces
    if 'stresses' in sweep:
        for comp, values in sweep['stresses'].items():
            if len(values) == len(angles):
                fig.add_trace(go.Scatter(
                    x=angles,
                    y=values,
                    mode='lines+markers',
                    name=f'{comp} (GPa)',
                    hovertemplate='Orientation: %{x:.2f}°<br>Stress: %{y:.4f} GPa<extra></extra>'
                ))
    
    # Add habit plane line
    fig.add_vline(x=54.7, line_width=3, line_dash="dash", line_color="green",
                  annotation_text="Habit Plane (54.7°)", annotation_position="top right")
    
    fig.update_layout(
        title='Habit Plane Vicinity Analysis - Interactive',
        xaxis_title='Orientation (°)',
        yaxis_title='Stress (GPa)',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")

if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'vicinity_analysis.json'
    
    try:
        data = load_analysis_data(json_file)
        plot_vicinity_sweep(data)
        create_interactive_plot(data)
        print("Data processing complete!")
    except Exception as e:
        print(f"Error processing data: {e}")
"""
        return script
    
    def _create_config_file(self, report):
        """Create configuration file"""
        
        config = {
            'analysis_type': 'habit_plane_vicinity',
            'generated_at': datetime.now().isoformat(),
            'parameters': report['analysis_parameters'],
            'physics_constants': {
                'habit_plane_angle': 54.7,
                'ag_shear_modulus': 30.0,
                'ag_youngs_modulus': 83.0,
                'ag_poissons_ratio': 0.37
            },
            'sintering_models': {
                'exponential': 'T_sinter(σ) = T₀ × exp(-β × |σ| / G)',
                'arrhenius': 'D = D₀ × exp[-(Q_a - ωσ)/(k_B T)]'
            }
        }
        
        return config
    
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
        else:
            return str(obj)

# =============================================
# MAIN APPLICATION WITH ENHANCED VISUALIZATIONS
# =============================================

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Ag FCC Twin: Enhanced Habit Plane Analysis",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS for publication quality
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .physics-header {
        font-size: 2.0rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 2.0rem;
        margin-bottom: 1.2rem;
    }
    .habit-plane-highlight {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 1.0rem;
        color: white;
        font-weight: bold;
        border: 4px solid #047857;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        margin: 1.5rem 0;
    }
    .defect-card {
        border: 3px solid;
        border-radius: 0.8rem;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: transform 0.3s;
    }
    .defect-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .isf-card {
        border-color: #FF6B6B;
        background-color: rgba(255, 107, 107, 0.12);
    }
    .esf-card {
        border-color: #4ECDC4;
        background-color: rgba(78, 205, 196, 0.12);
    }
    .twin-card {
        border-color: #45B7D1;
        background-color: rgba(69, 183, 209, 0.12);
    }
    .perfect-card {
        border-color: #96CEB4;
        background-color: rgba(150, 206, 180, 0.12);
    }
    .stress-metric {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.8rem;
        font-size: 1.1rem;
    }
    .temperature-metric {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.8rem;
        font-size: 1.1rem;
    }
    .system-metric {
        padding: 1.0rem;
        border-radius: 0.7rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.0rem;
    }
    .system-1 {
        background-color: #10B981 !important;
    }
    .system-2 {
        background-color: #F59E0B !important;
    }
    .system-3 {
        background-color: #EF4444 !important;
    }
    .latex-equation {
        font-family: "Cambria Math", "Times New Roman", serif;
        font-size: 1.4rem;
        padding: 1.5rem;
        background-color: #F8FAFC;
        border-radius: 0.8rem;
        border-left: 6px solid #3B82F6;
        margin: 1.5rem 0;
        text-align: center;
    }
    .tab-content {
        padding: 2.0rem;
        background-color: white;
        border-radius: 0.7rem;
        border: 2px solid #E5E7EB;
        margin-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0 0;
        gap: 1.2rem;
        padding-top: 12px;
        padding-bottom: 12px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .publication-mode-indicator {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.7rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .large-label {
        font-size: 1.3rem !important;
        font-weight: bold !important;
    }
    .stPlotlyChart {
        border: 3px solid #e9ecef;
        border-radius: 12px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .visualization-controls {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.7rem;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Ag FCC Twin: Enhanced Publication-Quality Habit Plane Analysis</h1>', unsafe_allow_html=True)
    
    # Publication mode toggle
    col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
    with col_header2:
        publication_mode = st.checkbox(
            "📊 Enable Publication Quality Mode", 
            value=True,
            help="Larger labels, better visibility for journal publications"
        )
    
    if publication_mode:
        st.markdown("""
        <div class="publication-mode-indicator">
        📈 <strong>PUBLICATION QUALITY MODE ENABLED</strong><br>
        <small>All visualizations use larger labels and enhanced visibility suitable for journal publications</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Physics equations with enhanced styling
    st.markdown("""
    <div class="latex-equation">
    <strong style="font-size: 1.6rem;">Physics of Stress-Modified Sintering:</strong><br>
    <div style="margin: 15px 0; font-size: 1.3rem;">
    <strong>Arrhenius Equation with Stress:</strong> D = D₀ exp[-(Q_a - Ωσ)/(k_BT)]
    </div>
    <div style="margin: 15px 0; font-size: 1.3rem;">
    <strong>Sintering Temperature:</strong> T_sinter(σ) = (Q_a - Ω|σ|) / [k_B ln(D₀/D_crit)]
    </div>
    <div style="margin: 15px 0; font-size: 1.3rem;">
    <strong>Hydrostatic Stress:</strong> σ_h = (σ₁ + σ₂ + σ₃)/3
    </div>
    <div style="margin: 15px 0; font-size: 1.3rem;">
    <strong>Von Mises Stress:</strong> σ_vm = √[½((σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²)]
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Habit plane information with enhanced styling
    st.markdown("""
    <div class="habit-plane-highlight">
    <div style="display: flex; align-items: center; justify-content: space-between;">
    <div>
    <h3 style="margin: 0; color: white; font-size: 1.8rem;">🎯 Ag FCC Twin Habit Plane: 54.7°</h3>
    <p style="margin: 0.8rem 0 0 0; color: white; opacity: 0.95; font-size: 1.1rem;">
    • {111} crystal planes • Maximum stress concentration • Optimal for defect engineering • Enhanced visualization for publications
    </p>
    </div>
    <div style="font-size: 3.5rem;">⚛️</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state with enhanced visualizer
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'physics_analyzer' not in st.session_state:
        st.session_state.physics_analyzer = PhysicsBasedStressAnalyzer()
    if 'sintering_calculator' not in st.session_state:
        st.session_state.sintering_calculator = EnhancedSinteringCalculator()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator(
            sigma=0.3,
            attention_dim=32,
            num_heads=4,
            attention_blend=0.7,
            use_spatial=True
        )
    if 'enhanced_visualizer' not in st.session_state:
        st.session_state.enhanced_visualizer = EnhancedHabitPlaneVisualizer(
            habit_angle=54.7,
            publication_mode=publication_mode
        )
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    
    # Update visualizer mode if changed
    st.session_state.enhanced_visualizer.publication_mode = publication_mode
    
    # Sidebar configuration with enhanced controls
    with st.sidebar:
        st.markdown('<h2 class="physics-header">⚙️ Enhanced Analysis Configuration</h2>', unsafe_allow_html=True)
        
        # Analysis mode
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["Habit Plane Vicinity", "Defect Type Comparison", "Comprehensive Dashboard", "Single Point Analysis"],
            index=0,
            help="Choose the type of analysis to perform"
        )
        
        # Publication quality controls
        st.markdown("#### 📊 Visualization Settings")
        
        # Visualization quality settings
        if publication_mode:
            st.info("📈 Publication quality mode enabled")
            
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                show_grid = st.checkbox(
                    "Show Grid",
                    value=True,
                    help="Display grid lines in visualizations"
                )
            
            with col_viz2:
                show_legend = st.checkbox(
                    "Show Legend",
                    value=True,
                    help="Display legend in visualizations"
                )
            
            # Chart customization
            chart_style = st.selectbox(
                "Chart Style",
                ["Standard", "High Contrast", "Colorblind Friendly"],
                index=0,
                help="Select color scheme for visualizations"
            )
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load Solutions", use_container_width=True):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    if st.session_state.solutions:
                        st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                    else:
                        st.warning("No solutions found in directory")
        
        with col_load2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.cache_data.clear()
                st.success("Cache cleared")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            stats = st.session_state.loader.get_solution_statistics(st.session_state.solutions)
            
            with st.expander(f"📊 Loaded Solutions ({stats['total_solutions']})", expanded=False):
                # Defect type distribution
                st.write("**Defect Types:**")
                for defect, count in stats.get('defect_types', {}).items():
                    col_d1, col_d2 = st.columns([2, 1])
                    with col_d1:
                        st.write(f"- {defect}")
                    with col_d2:
                        st.write(f"{count}")
                
                # Orientation statistics
                if 'orientation_stats' in stats:
                    st.write("**Orientation Statistics:**")
                    st.write(f"- Range: {stats['orientation_stats']['min']:.1f}° to {stats['orientation_stats']['max']:.1f}°")
                    st.write(f"- Mean: {stats['orientation_stats']['mean']:.1f}° ± {stats['orientation_stats']['std']:.1f}°")
        
        # Target parameters
        st.markdown("#### 🎯 Target Parameters")
        
        # Defect type with auto eigen strain
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for analysis"
        )
        
        # Show defect cards in sidebar
        st.markdown("##### 🔬 Defect Properties")
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        
        col_def1, col_def2 = st.columns(2)
        with col_def1:
            eps0 = st.number_input(
                "Eigen Strain (ε*)",
                min_value=0.0,
                max_value=3.0,
                value=default_eps0,
                step=0.01,
                help="Eigen strain value (auto-set based on defect type)"
            )
        
        with col_def2:
            kappa = st.slider(
                "Interface Energy (κ)",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.01,
                help="Interface energy parameter"
            )
        
        shape = st.selectbox(
            "Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
            index=0,
            help="Geometric shape of the defect"
        )
        
        # Region selection
        st.markdown("#### 📍 Analysis Region")
        region_type = st.selectbox(
            "Select Region for Analysis",
            ["bulk", "interface", "defect"],
            index=0,
            help="Material region to analyze: bulk (η<0.4), interface (0.4≤η≤0.6), defect (η>0.6)"
        )
        
        # Vicinity settings
        if analysis_mode == "Habit Plane Vicinity":
            st.markdown("#### 🎯 Vicinity Settings")
            
            vicinity_range = st.slider(
                "Vicinity Range (± degrees)",
                min_value=1.0,
                max_value=45.0,
                value=10.0,
                step=1.0,
                help="Range around habit plane to analyze"
            )
            
            n_points = st.slider(
                "Number of Points",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Number of orientation points in sweep"
            )
        
        # Interpolator settings
        st.markdown("#### 🧠 Interpolation Settings")
        
        col_int1, col_int2 = st.columns(2)
        with col_int1:
            attention_blend = st.slider(
                "Attention Blend",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Blend ratio: attention vs spatial weights (0=spatial only, 1=attention only)"
            )
        
        with col_int2:
            sigma = st.slider(
                "Spatial Sigma",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Spatial regularization parameter"
            )
        
        use_physics_constraints = st.checkbox(
            "Apply Physics Constraints",
            value=True,
            help="Apply physics-based constraints on interpolation"
        )
        
        # Sintering model settings
        st.markdown("#### 🔥 Sintering Model")
        
        sintering_model = st.radio(
            "Primary Sintering Model",
            ["Arrhenius (Physics-based)", "Exponential (Empirical)", "Both"],
            index=0,
            help="Select primary model for sintering temperature prediction"
        )
        
        # Enhanced visualization controls
        st.markdown("#### 🎨 Visualization Controls")
        
        # Radar chart settings
        with st.expander("Radar Chart Settings", expanded=False):
            radar_hole_size = st.slider(
                "Radar Hole Size",
                min_value=0.0,
                max_value=0.3,
                value=0.15,
                step=0.01,
                help="Center hole size in radar charts (0=no hole, 0.3=large hole)"
            )
            
            radar_line_width = st.slider(
                "Radar Line Width",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="Line width in radar charts"
            )
        
        # Generate button
        st.markdown("---")
        generate_text = "🚀 Generate Enhanced Analysis" if analysis_mode != "Comprehensive Dashboard" else "📊 Generate Enhanced Dashboard"
        if st.button(generate_text, type="primary", use_container_width=True):
            st.session_state.generate_analysis = True
        else:
            st.session_state.generate_analysis = False
    
    # Main content area
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        
        # Show directory information
        with st.expander("📁 Directory Information", expanded=True):
            st.info(f"**Solutions Directory:** {SOLUTIONS_DIR}")
            st.write("Expected file formats: .pkl, .pickle, .pt, .pth")
            st.write("""
            **Expected data structure:**
            - Each file should contain a dictionary with:
              - 'params': Dictionary of simulation parameters
              - 'history': List of simulation frames
              - Each frame should contain 'eta' (phase field) and 'stresses' (stress fields)
            """)
        
        # Quick start guide
        st.markdown("""
        ## 🚀 Quick Start Guide
        
        1. **Prepare Data**: Place your simulation files in the `numerical_solutions` directory
        2. **Load Solutions**: Click the "Load Solutions" button in the sidebar
        3. **Configure Analysis**: Set your analysis parameters in the sidebar
        4. **Generate Analysis**: Click "Generate Analysis" to start
        
        ## 🔬 Key Features
        
        ### Enhanced Publication Quality Visualizations
        - **Larger labels** suitable for journal publications
        - **Enhanced radar views** with adjustable parameters
        - **High-resolution outputs** for print quality
        - **Colorblind-friendly** color schemes available
        
        ### Habit Plane Vicinity Analysis
        - Focus on 54.7° ± specified range
        - Combined attention and Gaussian regularization
        - Physics-aware interpolation with eigen strain constraints
        
        ### Stress Components
        - **Hydrostatic Stress (σ_h)**: Critical for sintering (trace of stress tensor/3)
        - **Von Mises Stress (σ_vm)**: Indicates yield onset
        - **Stress Magnitude (σ_mag)**: Overall stress intensity
        
        ### Sintering Temperature Prediction
        - **Arrhenius Model**: Physics-based with defect-specific parameters
        - **Exponential Model**: Empirical correlation
        - **System Classification**: Maps stress to AgNP sintering systems
        
        ### Enhanced Visualization
        - Sunburst charts with publication-quality labels
        - Radar charts with larger, editable labels
        - Interactive plots with enhanced hover information
        - Comprehensive dashboards with multiple views
        """)
        
    else:
        # Create tabs for different analysis modes
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Overview",
            "📈 Enhanced Vicinity Analysis",
            "🔬 Enhanced Defect Comparison",
            "📊 Enhanced Dashboard"
        ])
        
        with tab1:
            st.markdown('<h2 class="physics-header">🏠 Enhanced Analysis Overview</h2>', unsafe_allow_html=True)
            
            # Quick statistics with enhanced styling
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Loaded Solutions", len(st.session_state.solutions), 
                         help="Total number of simulation solutions loaded")
            
            with col_stat2:
                defect_types = set()
                for sol in st.session_state.solutions:
                    params = sol.get('params', {})
                    defect_types.add(params.get('defect_type', 'Unknown'))
                st.metric("Defect Types", len(defect_types),
                         help="Number of different defect types in loaded solutions")
            
            with col_stat3:
                orientations = []
                for sol in st.session_state.solutions:
                    params = sol.get('params', {})
                    theta = params.get('theta', 0.0)
                    if theta is not None:
                        orientations.append(np.degrees(theta) % 360)
                if orientations:
                    st.metric("Orientation Range", f"{min(orientations):.1f}° - {max(orientations):.1f}°",
                             help="Range of crystal orientations in loaded solutions")
                else:
                    st.metric("Orientation Range", "N/A")
            
            with col_stat4:
                has_physics = sum(1 for sol in st.session_state.solutions if sol.get('physics_analysis'))
                st.metric("Physics Analyzed", f"{has_physics}/{len(st.session_state.solutions)}",
                         help="Number of solutions with physics analysis")
            
            # Defect type cards with enhanced styling
            st.markdown("#### 🔬 Defect Types with Eigen Strains")
            
            col_def1, col_def2, col_def3, col_def4 = st.columns(4)
            
            defect_info = [
                ("ISF", "Intrinsic Stacking Fault", 0.71, "#FF6B6B", "Missing one {111} atomic plane"),
                ("ESF", "Extrinsic Stacking Fault", 1.41, "#4ECDC4", "Extra {111} atomic plane"),
                ("Twin", "Coherent Twin Boundary", 2.12, "#45B7D1", "Mirror symmetry across {111} plane"),
                ("No Defect", "Perfect Crystal", 0.0, "#96CEB4", "Perfect Face-Centered Cubic crystal")
            ]
            
            for i, (name, desc, strain, color, detail) in enumerate(defect_info):
                with [col_def1, col_def2, col_def3, col_def4][i]:
                    st.markdown(f"""
                    <div class="defect-card {'isf' if name == 'ISF' else 'esf' if name == 'ESF' else 'twin' if name == 'Twin' else 'perfect'}-card">
                    <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{name}</div>
                    <div style="font-size: 1.0rem; color: #666; margin: 0.8rem 0;">{desc}</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #333; margin: 0.5rem 0;">ε* = {strain}</div>
                    <div style="font-size: 0.9rem; color: #888; font-style: italic;">{detail}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # System classification with enhanced styling
            st.markdown("#### 🏷️ System Classification")
            
            col_sys1, col_sys2, col_sys3 = st.columns(3)
            
            system_info = [
                ("System 1", "Perfect Crystal", "σ < 5 GPa", "620-630 K", "#10B981", "Low stress, high temperature"),
                ("System 2", "SF/Twins", "5 ≤ σ < 20 GPa", "450-550 K", "#F59E0B", "Medium stress, medium temperature"),
                ("System 3", "Plastic Deformation", "σ ≥ 20 GPa", "350-400 K", "#EF4444", "High stress, low temperature")
            ]
            
            for i, (name, desc, stress_range, temp_range, color, detail) in enumerate(system_info):
                with [col_sys1, col_sys2, col_sys3][i]:
                    st.markdown(f"""
                    <div class="system-metric" style="background-color: {color};">
                    <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem;">{name}</div>
                    <div style="font-size: 1.1rem; opacity: 0.95; margin-bottom: 0.5rem;">{desc}</div>
                    <div style="font-size: 0.9rem; margin: 0.3rem 0; padding: 0.2rem; background: rgba(255,255,255,0.2); border-radius: 0.3rem;">{stress_range}</div>
                    <div style="font-size: 0.9rem; margin: 0.3rem 0; padding: 0.2rem; background: rgba(255,255,255,0.2); border-radius: 0.3rem;">{temp_range}</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem; font-style: italic;">{detail}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Quick analysis buttons with enhanced styling
            st.markdown("#### ⚡ Quick Analysis")
            
            col_q1, col_q2, col_q3 = st.columns(3)
            
            with col_q1:
                if st.button("🎯 Analyze Habit Plane", use_container_width=True, 
                           help="Perform enhanced habit plane vicinity analysis"):
                    st.session_state.quick_analysis = "habit_plane"
            
            with col_q2:
                if st.button("🔬 Compare Defects", use_container_width=True,
                           help="Compare different defect types with enhanced visualizations"):
                    st.session_state.quick_analysis = "defect_compare"
            
            with col_q3:
                if st.button("📊 Generate Dashboard", use_container_width=True,
                           help="Generate comprehensive enhanced dashboard"):
                    st.session_state.quick_analysis = "dashboard"
        
        with tab2:
            st.markdown('<h2 class="physics-header">📈 Enhanced Habit Plane Vicinity Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'habit_plane':
                
                with st.spinner("Performing enhanced vicinity analysis..."):
                    # Prepare target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa
                    }
                    
                    # Prepare analysis parameters
                    analysis_params = {
                        'vicinity_range': vicinity_range if 'vicinity_range' in locals() else 10.0,
                        'n_points': n_points if 'n_points' in locals() else 50,
                        'region_type': region_type,
                        'attention_blend': attention_blend,
                        'sigma': sigma,
                        'use_physics_constraints': use_physics_constraints,
                        'publication_mode': publication_mode
                    }
                    
                    # Update interpolator settings
                    st.session_state.interpolator.attention_blend = attention_blend
                    st.session_state.interpolator.sigma = sigma
                    
                    # Perform vicinity sweep
                    vicinity_sweep = st.session_state.interpolator.create_vicinity_sweep(
                        st.session_state.solutions,
                        target_params,
                        vicinity_range=analysis_params['vicinity_range'],
                        n_points=analysis_params['n_points'],
                        region_type=analysis_params['region_type']
                    )
                    
                    if vicinity_sweep:
                        st.success(f"✅ Generated enhanced vicinity sweep with {analysis_params['n_points']} points")
                        
                        # Store in session state
                        st.session_state.vicinity_sweep = vicinity_sweep
                        st.session_state.current_target_params = target_params
                        st.session_state.current_analysis_params = analysis_params
                        
                        # Display results with enhanced styling
                        st.markdown("#### 📊 Enhanced Analysis Results")
                        
                        # Create columns for metrics with enhanced styling
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        
                        # Habit plane stress
                        angles = np.array(vicinity_sweep['angles'])
                        habit_angle = 54.7
                        habit_idx = np.argmin(np.abs(angles - habit_angle))
                        
                        with col_res1:
                            sigma_hydro = vicinity_sweep['stresses']['sigma_hydro'][habit_idx]
                            st.markdown(f"""
                            <div class="stress-metric">
                            <div style="font-size: 1.3rem;">Habit Plane σ_h</div>
                            <div style="font-size: 2.0rem; margin: 0.5rem 0;">{sigma_hydro:.3f} GPa</div>
                            <div style="font-size: 1.0rem;">Hydrostatic Stress</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_res2:
                            von_mises = vicinity_sweep['stresses']['von_mises'][habit_idx]
                            st.markdown(f"""
                            <div class="stress-metric">
                            <div style="font-size: 1.3rem;">Habit Plane σ_vm</div>
                            <div style="font-size: 2.0rem; margin: 0.5rem 0;">{von_mises:.3f} GPa</div>
                            <div style="font-size: 1.0rem;">Von Mises Stress</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_res3:
                            T_sinter = vicinity_sweep['sintering_temps']['arrhenius_defect'][habit_idx]
                            st.markdown(f"""
                            <div class="temperature-metric">
                            <div style="font-size: 1.3rem;">Habit Plane T_sinter</div>
                            <div style="font-size: 2.0rem; margin: 0.5rem 0;">{T_sinter:.1f} K</div>
                            <div style="font-size: 1.0rem;">({T_sinter-273.15:.1f}°C)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_res4:
                            system_info = st.session_state.sintering_calculator.map_system_to_temperature(sigma_hydro)
                            system_name = system_info[0].split('(')[0].strip()
                            system_temp_range = f"{system_info[1][0]:.0f}-{system_info[1][1]:.0f} K"
                            
                            st.markdown(f"""
                            <div class="system-metric system-{1 if '1' in system_name else 2 if '2' in system_name else 3}">
                            <div style="font-size: 1.3rem;">System Classification</div>
                            <div style="font-size: 1.8rem; margin: 0.5rem 0;">{system_name}</div>
                            <div style="font-size: 1.0rem;">{system_temp_range}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Create enhanced visualizations
                        st.markdown("#### 📈 Enhanced Visualizations")
                        
                        # Enhanced Sunburst for hydrostatic stress
                        fig_enhanced_sunburst = st.session_state.enhanced_visualizer.create_enhanced_vicinity_sunburst(
                            vicinity_sweep['angles'],
                            vicinity_sweep['stresses']['sigma_hydro'],
                            stress_component='sigma_hydro',
                            title=f"Enhanced Habit Plane Vicinity - {defect_type}",
                            radius_scale=1.0,
                            show_grid=show_grid if 'show_grid' in locals() else True
                        )
                        
                        if fig_enhanced_sunburst:
                            st.plotly_chart(fig_enhanced_sunburst, use_container_width=True)
                        
                        # Enhanced temperature chart
                        fig_enhanced_temp = st.session_state.enhanced_visualizer.create_enhanced_sintering_temperature_chart(
                            vicinity_sweep,
                            defect_type=defect_type,
                            title=f"Enhanced Sintering Temperature Analysis - {defect_type}"
                        )
                        
                        if fig_enhanced_temp:
                            st.pyplot(fig_enhanced_temp)
                            plt.close(fig_enhanced_temp)
                        
                        # Line plots for all stress components with enhanced styling
                        st.markdown("#### 📊 Stress Component Comparison")
                        
                        fig_enhanced_line, ax_enhanced_line = plt.subplots(figsize=(14, 7))
                        
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                        line_styles = ['-', '--', '-.']
                        line_widths = [3, 3, 3]
                        
                        for idx, (comp, color, style, width) in enumerate(zip(
                            ['sigma_hydro', 'von_mises', 'sigma_mag'], 
                            colors, line_styles, line_widths)):
                            
                            ax_enhanced_line.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses'][comp],
                                color=color,
                                linestyle=style,
                                linewidth=width,
                                label=comp.replace('_', ' ').title(),
                                marker='o' if idx == 0 else 's' if idx == 1 else '^',
                                markersize=8 if publication_mode else 6,
                                markevery=10
                            )
                        
                        # Enhanced habit plane line
                        ax_enhanced_line.axvline(habit_angle, color='green', linestyle=':', 
                                               linewidth=4 if publication_mode else 2,
                                               label=f'Habit Plane ({habit_angle}°)',
                                               alpha=0.8)
                        
                        # Enhanced axis labels
                        ax_enhanced_line.set_xlabel('Orientation (°)', 
                                                  fontsize=18 if publication_mode else 14,
                                                  fontweight='bold')
                        ax_enhanced_line.set_ylabel('Stress (GPa)', 
                                                  fontsize=18 if publication_mode else 14,
                                                  fontweight='bold')
                        
                        # Enhanced title
                        ax_enhanced_line.set_title('Enhanced Stress Components in Habit Plane Vicinity', 
                                                 fontsize=20 if publication_mode else 16,
                                                 fontweight='bold',
                                                 pad=20)
                        
                        # Enhanced legend
                        ax_enhanced_line.legend(fontsize=14 if publication_mode else 11,
                                              frameon=True,
                                              framealpha=0.9,
                                              edgecolor='black',
                                              loc='upper right')
                        
                        # Enhanced grid
                        ax_enhanced_line.grid(True, alpha=0.3, linestyle='--')
                        
                        # Enhanced ticks
                        ax_enhanced_line.tick_params(axis='both', which='major', 
                                                   labelsize=16 if publication_mode else 12)
                        
                        # Enhanced spines
                        for spine in ax_enhanced_line.spines.values():
                            spine.set_linewidth(3 if publication_mode else 2)
                        
                        plt.tight_layout()
                        st.pyplot(fig_enhanced_line)
                        plt.close(fig_enhanced_line)
                        
                        # Export options with enhanced styling
                        st.markdown("#### 📤 Enhanced Export Options")
                        
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            if st.button("💾 Export Enhanced JSON", use_container_width=True,
                                       help="Export complete analysis data in JSON format"):
                                # Prepare report
                                report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                    vicinity_sweep,
                                    {},
                                    target_params,
                                    analysis_params
                                )
                                
                                json_str = json.dumps(report, indent=2, default=st.session_state.results_manager._json_serializer)
                                
                                st.download_button(
                                    label="📥 Download Enhanced JSON",
                                    data=json_str,
                                    file_name=f"enhanced_vicinity_analysis_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        
                        with col_exp2:
                            if st.button("📊 Export High-Res CSV", use_container_width=True,
                                       help="Export data in CSV format for high-resolution plotting"):
                                # Create CSV data
                                rows = []
                                angles = vicinity_sweep['angles']
                                
                                for i in range(len(angles)):
                                    row = {
                                        'angle_deg': angles[i],
                                        'sigma_hydro_gpa': vicinity_sweep['stresses']['sigma_hydro'][i],
                                        'von_mises_gpa': vicinity_sweep['stresses']['von_mises'][i],
                                        'sigma_mag_gpa': vicinity_sweep['stresses']['sigma_mag'][i],
                                        'T_sinter_exponential_k': vicinity_sweep['sintering_temps']['exponential'][i],
                                        'T_sinter_arrhenius_k': vicinity_sweep['sintering_temps']['arrhenius_defect'][i]
                                    }
                                    rows.append(row)
                                
                                df = pd.DataFrame(rows)
                                csv = df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download High-Res CSV",
                                    data=csv,
                                    file_name=f"enhanced_vicinity_data_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        with col_exp3:
                            if st.button("📦 Export Complete Package", use_container_width=True,
                                       help="Export complete analysis package with all files"):
                                report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                    vicinity_sweep,
                                    {},
                                    target_params,
                                    analysis_params
                                )
                                
                                zip_buffer = st.session_state.results_manager.create_comprehensive_export(report)
                                
                                st.download_button(
                                    label="📥 Download Complete ZIP",
                                    data=zip_buffer.getvalue(),
                                    file_name=f"enhanced_vicinity_analysis_package_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                    
                    else:
                        st.error("Failed to generate vicinity sweep. Please check your data and parameters.")
            
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Enhanced Analysis'")
                
                # Show enhanced example visualization
                st.markdown("#### 📊 Enhanced Example Visualization")
                
                # Create enhanced example data
                example_angles = np.linspace(44.7, 64.7, 50)
                example_stress = 20 * np.exp(-(example_angles - 54.7)**2 / (2*5**2)) + 5
                example_temp = 623 * np.exp(-example_stress / 30) + 50 * np.sin(np.radians(example_angles))
                
                # Create enhanced example figure
                fig_enhanced_example, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Enhanced stress plot
                ax1.plot(example_angles, example_stress, 'b-', linewidth=3)
                ax1.axvline(54.7, color='green', linestyle='--', linewidth=3, label='Habit Plane (54.7°)')
                ax1.fill_between(example_angles, example_stress, alpha=0.2, color='blue')
                ax1.set_xlabel('Orientation (°)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Hydrostatic Stress (GPa)', fontsize=14, fontweight='bold')
                ax1.set_title('Enhanced Example: Stress Concentration at Habit Plane', fontsize=16, fontweight='bold')
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.tick_params(axis='both', labelsize=12)
                
                # Enhanced temperature plot
                ax2.plot(example_angles, example_temp, 'r-', linewidth=3)
                ax2.axvline(54.7, color='green', linestyle='--', linewidth=3, label='Habit Plane (54.7°)')
                ax2.fill_between(example_angles, example_temp, alpha=0.2, color='red')
                ax2.set_xlabel('Orientation (°)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Sintering Temperature (K)', fontsize=14, fontweight='bold')
                ax2.set_title('Enhanced Example: Temperature Reduction at Habit Plane', fontsize=16, fontweight='bold')
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.tick_params(axis='both', labelsize=12)
                
                # Add Celsius on secondary axis with enhanced styling
                ax2_2 = ax2.twinx()
                celsius_ticks = ax2.get_yticks()
                ax2_2.set_ylim(ax2.get_ylim())
                ax2_2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks], fontsize=12)
                ax2_2.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
                
                plt.suptitle('Enhanced Publication-Quality Visualization Example', fontsize=18, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig_enhanced_example)
                plt.close(fig_enhanced_example)
        
        with tab3:
            st.markdown('<h2 class="physics-header">🔬 Enhanced Defect Type Comparison</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'defect_compare':
                
                with st.spinner("Comparing defect types with enhanced visualizations..."):
                    # Compare all defect types
                    defect_comparison = st.session_state.interpolator.compare_defect_types(
                        st.session_state.solutions,
                        angle_range=(0, 360),
                        n_points=100,
                        region_type=region_type,
                        shapes=[shape]
                    )
                    
                    if defect_comparison:
                        st.success(f"✅ Generated enhanced comparison of {len(defect_comparison)} defect types")
                        
                        # Store in session state
                        st.session_state.defect_comparison = defect_comparison
                        
                        # Display defect comparison with enhanced styling
                        st.markdown("#### 📊 Enhanced Defect Comparison Results")
                        
                        # Create tabs for different enhanced views
                        comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Enhanced Stress Comparison", "Enhanced Sintering Comparison", "Enhanced Radar View"])
                        
                        with comp_tab1:
                            # Enhanced stress comparison plot
                            fig_enhanced_comp = st.session_state.enhanced_visualizer.create_defect_comparison_plot(
                                defect_comparison,
                                stress_component='sigma_hydro',
                                title="Enhanced Hydrostatic Stress Comparison"
                            )
                            
                            if fig_enhanced_comp:
                                st.plotly_chart(fig_enhanced_comp, use_container_width=True)
                        
                        with comp_tab2:
                            # Enhanced sintering temperature comparison
                            fig_enhanced_sinter_comp, ax_sinter = plt.subplots(figsize=(14, 7))
                            
                            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                            line_styles = ['-', '--', '-.', ':']
                            line_widths = [3, 3, 3, 3]
                            
                            for idx, (key, data) in enumerate(defect_comparison.items()):
                                if idx < len(colors):
                                    ax_sinter.plot(
                                        data['angles'],
                                        data['sintering_temps'],
                                        color=colors[idx],
                                        linestyle=line_styles[idx],
                                        linewidth=line_widths[idx],
                                        label=f"{data['defect_type']} (ε*={data['eigen_strain']})",
                                        marker='o' if idx == 0 else 's' if idx == 1 else '^' if idx == 2 else 'd',
                                        markersize=8 if publication_mode else 6,
                                        markevery=20
                                    )
                            
                            # Enhanced habit plane line
                            ax_sinter.axvline(54.7, color='green', linestyle=':', linewidth=4,
                                            label='Habit Plane (54.7°)', alpha=0.8)
                            
                            # Enhanced axis labels
                            ax_sinter.set_xlabel('Orientation (°)', 
                                               fontsize=18 if publication_mode else 14,
                                               fontweight='bold')
                            ax_sinter.set_ylabel('Sintering Temperature (K)', 
                                               fontsize=18 if publication_mode else 14,
                                               fontweight='bold')
                            
                            # Enhanced title
                            ax_sinter.set_title('Enhanced Sintering Temperature Comparison by Defect Type', 
                                              fontsize=20 if publication_mode else 16,
                                              fontweight='bold',
                                              pad=20)
                            
                            # Enhanced legend
                            ax_sinter.legend(fontsize=14 if publication_mode else 10,
                                           frameon=True,
                                           framealpha=0.9,
                                           edgecolor='black',
                                           loc='upper right')
                            
                            # Enhanced grid
                            ax_sinter.grid(True, alpha=0.3, linestyle='--')
                            
                            # Enhanced ticks
                            ax_sinter.tick_params(axis='both', which='major', 
                                                labelsize=16 if publication_mode else 12)
                            
                            # Enhanced spines
                            for spine in ax_sinter.spines.values():
                                spine.set_linewidth(3 if publication_mode else 2)
                            
                            # Add Celsius on secondary axis with enhanced styling
                            ax_sinter2 = ax_sinter.twinx()
                            celsius_ticks = ax_sinter.get_yticks()
                            ax_sinter2.set_ylim(ax_sinter.get_ylim())
                            ax_sinter2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks], 
                                                      fontsize=16 if publication_mode else 12)
                            ax_sinter2.set_ylabel('Temperature (°C)', 
                                                fontsize=18 if publication_mode else 14,
                                                fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig_enhanced_sinter_comp)
                            plt.close(fig_enhanced_sinter_comp)
                        
                        with comp_tab3:
                            # Enhanced radar comparison for habit plane vicinity
                            habit_range = 30.0
                            min_angle = 54.7 - habit_range
                            max_angle = 54.7 + habit_range
                            
                            # Filter data for habit plane vicinity
                            vicinity_comparison = {}
                            for key, data in defect_comparison.items():
                                angles = np.array(data['angles'])
                                mask = (angles >= min_angle) & (angles <= max_angle)
                                
                                if np.any(mask):
                                    vicinity_comparison[key] = {
                                        'angles': angles[mask].tolist(),
                                        'stresses': {comp: np.array(vals)[mask].tolist() 
                                                   for comp, vals in data['stresses'].items()},
                                        'defect_type': data['defect_type'],
                                        'color': data['color']
                                    }
                            
                            # Create enhanced radar comparison
                            fig_enhanced_radar = st.session_state.enhanced_visualizer.create_enhanced_defect_comparison_radar(
                                defect_comparison,
                                stress_component='sigma_hydro',
                                title="Enhanced Defect Type Comparison - Radar View"
                            )
                            
                            if fig_enhanced_radar:
                                st.plotly_chart(fig_enhanced_radar, use_container_width=True)
                        
                        # Enhanced summary statistics
                        st.markdown("#### 📈 Enhanced Summary Statistics")
                        
                        # Calculate enhanced statistics for each defect
                        summary_data = []
                        for key, data in defect_comparison.items():
                            if 'stresses' in data and 'sigma_hydro' in data['stresses']:
                                stresses = data['stresses']['sigma_hydro']
                                if stresses:
                                    summary_data.append({
                                        'Defect Type': data['defect_type'],
                                        'Eigen Strain': data['eigen_strain'],
                                        'Max Stress (GPa)': f"{max(stresses):.3f}",
                                        'Mean Stress (GPa)': f"{np.mean(stresses):.3f}",
                                        'Stress Range (GPa)': f"{max(stresses) - min(stresses):.3f}",
                                        'Min T_sinter (K)': f"{min(data['sintering_temps']):.1f}",
                                        'Max T_sinter (K)': f"{max(data['sintering_temps']):.1f}",
                                        'T_sinter Range (K)': f"{max(data['sintering_temps']) - min(data['sintering_temps']):.1f}"
                                    })
                        
                        if summary_data:
                            df_summary = pd.DataFrame(summary_data)
                            
                            # Apply enhanced styling to dataframe
                            st.dataframe(
                                df_summary.style
                                .background_gradient(subset=['Max Stress (GPa)', 'Mean Stress (GPa)'], cmap='Reds')
                                .background_gradient(subset=['Min T_sinter (K)', 'Max T_sinter (K)'], cmap='Blues_r')
                                .format(precision=3),
                                use_container_width=True,
                                height=400
                            )
                        
                        # Enhanced export comparison data
                        st.markdown("#### 📤 Enhanced Export Options")
                        
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            # Enhanced JSON export
                            comparison_json = json.dumps(
                                st.session_state.defect_comparison,
                                indent=2,
                                default=st.session_state.results_manager._json_serializer
                            )
                            
                            st.download_button(
                                label="💾 Export Enhanced JSON",
                                data=comparison_json,
                                file_name=f"enhanced_defect_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col_exp2:
                            # Enhanced CSV export with more data
                            rows = []
                            for key, data in defect_comparison.items():
                                angles = data['angles']
                                for i in range(len(angles)):
                                    row = {
                                        'defect_type': data['defect_type'],
                                        'eigen_strain': data['eigen_strain'],
                                        'shape': data.get('shape', 'Unknown'),
                                        'angle_deg': angles[i],
                                        'angle_rad': np.radians(angles[i])
                                    }
                                    
                                    for comp, stresses in data['stresses'].items():
                                        if i < len(stresses):
                                            row[f'{comp}_gpa'] = stresses[i]
                                    
                                    if i < len(data['sintering_temps']):
                                        row['T_sinter_k'] = data['sintering_temps'][i]
                                        row['T_sinter_c'] = data['sintering_temps'][i] - 273.15
                                    
                                    rows.append(row)
                            
                            if rows:
                                df = pd.DataFrame(rows)
                                csv = df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📊 Export Enhanced CSV",
                                    data=csv,
                                    file_name=f"enhanced_defect_comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                    
                    else:
                        st.error("Failed to generate defect comparison. Please check your data.")
            
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Enhanced Analysis'")
                
                # Show enhanced defect comparison info
                st.markdown("""
                #### 🔬 Enhanced Defect Comparison Analysis
                
                This enhanced analysis compares different defect types (ISF, ESF, Twin, No Defect) across all orientations with publication-quality visualizations.
                
                **Key enhanced comparisons:**
                1. **Stress distribution** - How each defect concentrates stress with larger, clearer labels
                2. **Sintering temperature** - Temperature reduction capability with enhanced visualizations
                3. **Habit plane effects** - Special behavior at 54.7° with improved radar views
                4. **Enhanced radar charts** - Larger labels, better visibility, adjustable parameters
                
                **Expected insights:**
                - Twin boundaries show maximum stress concentration
                - ISF/ESF have intermediate effects with clear differentiation
                - Perfect crystals have minimal stress, easily distinguishable
                - Habit plane shows peak effects for twins with enhanced visualization
                
                **Publication-quality features:**
                - Larger font sizes for all labels
                - Enhanced color contrast for better visibility
                - Adjustable radar chart parameters
                - High-resolution export options
                """)
        
        with tab4:
            st.markdown('<h2 class="physics-header">📊 Enhanced Comprehensive Dashboard</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'dashboard':
                
                with st.spinner("Generating enhanced comprehensive dashboard..."):
                    # Check if we have both vicinity sweep and defect comparison
                    if ('vicinity_sweep' not in st.session_state or 
                        'defect_comparison' not in st.session_state):
                        
                        st.warning("Please run both Enhanced Vicinity Analysis and Enhanced Defect Comparison first.")
                        
                        col_run1, col_run2 = st.columns(2)
                        with col_run1:
                            if st.button("🏃 Run Enhanced Vicinity Analysis", use_container_width=True):
                                st.session_state.quick_analysis = "habit_plane"
                                st.rerun()
                        
                        with col_run2:
                            if st.button("🏃 Run Enhanced Defect Comparison", use_container_width=True):
                                st.session_state.quick_analysis = "defect_compare"
                                st.rerun()
                    
                    else:
                        # Generate enhanced comprehensive dashboard
                        vicinity_sweep = st.session_state.vicinity_sweep
                        defect_comparison = st.session_state.defect_comparison
                        
                        # Create enhanced comprehensive visualization
                        st.markdown("#### 🎨 Enhanced Dashboard Visualizations")
                        
                        # Create multiple enhanced visualizations
                        col_dash1, col_dash2 = st.columns(2)
                        
                        with col_dash1:
                            # Enhanced sunburst
                            fig_enhanced_dash1 = st.session_state.enhanced_visualizer.create_enhanced_vicinity_sunburst(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses']['sigma_hydro'],
                                stress_component='sigma_hydro',
                                title=f"Enhanced Hydrostatic Stress - {st.session_state.current_target_params.get('defect_type', 'Unknown')}",
                                radius_scale=1.0,
                                show_grid=True
                            )
                            
                            if fig_enhanced_dash1:
                                st.plotly_chart(fig_enhanced_dash1, use_container_width=True)
                        
                        with col_dash2:
                            # Enhanced defect comparison radar
                            fig_enhanced_dash2 = st.session_state.enhanced_visualizer.create_enhanced_defect_comparison_radar(
                                defect_comparison,
                                stress_component='sigma_hydro',
                                title="Enhanced Defect Comparison Radar"
                            )
                            
                            if fig_enhanced_dash2:
                                st.plotly_chart(fig_enhanced_dash2, use_container_width=True)
                        
                        # Enhanced temperature chart
                        st.markdown("#### 🔥 Enhanced Sintering Analysis")
                        
                        fig_enhanced_temp_dash = st.session_state.enhanced_visualizer.create_enhanced_sintering_temperature_chart(
                            vicinity_sweep,
                            defect_type=st.session_state.current_target_params.get('defect_type', 'Unknown'),
                            title="Enhanced Comprehensive Sintering Analysis"
                        )
                        
                        if fig_enhanced_temp_dash:
                            st.pyplot(fig_enhanced_temp_dash)
                            plt.close(fig_enhanced_temp_dash)
                        
                        # Additional enhanced analysis
                        st.markdown("#### 📈 Enhanced Advanced Analysis")
                        
                        # Create tabs for different enhanced analyses
                        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Enhanced Physics Analysis", "Enhanced Sintering Optimization", "Enhanced Export Package"])
                        
                        with adv_tab1:
                            # Enhanced physics-based analysis
                            st.markdown("##### 🔬 Enhanced Physics-Based Analysis")
                            
                            # Calculate enhanced stress intensity factors
                            st.write("**Enhanced Stress Intensity Factors (K):**")
                            
                            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                            
                            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
                            k_values = []
                            
                            for i, defect in enumerate(defect_types):
                                with [col_k1, col_k2, col_k3, col_k4][i]:
                                    # Find max stress for this defect
                                    max_stress = 0
                                    for key, data in defect_comparison.items():
                                        if data['defect_type'] == defect:
                                            if 'sigma_hydro' in data['stresses']:
                                                max_stress = max(data['stresses']['sigma_hydro'])
                                                break
                                    
                                    # Calculate K
                                    K = st.session_state.physics_analyzer.compute_stress_intensity_factor(
                                        {'sigma_hydro': {'max_abs': max_stress}},
                                        st.session_state.physics_analyzer.get_eigen_strain(defect),
                                        defect
                                    )
                                    
                                    k_values.append(K)
                                    
                                    # Enhanced metric display
                                    color = "#10B981" if K < 10 else "#F59E0B" if K < 20 else "#EF4444"
                                    
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 1rem; border-radius: 0.7rem; border: 2px solid {color}; background: rgba(255,255,255,0.1);">
                                    <div style="font-size: 1.2rem; font-weight: bold; color: #333;">K for {defect}</div>
                                    <div style="font-size: 2.0rem; font-weight: bold; color: {color}; margin: 0.5rem 0;">{K:.2f} MPa√m</div>
                                    <div style="font-size: 1.0rem; color: #666;">Stress Intensity</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Enhanced crystal orientation analysis
                            st.markdown("##### 🧊 Enhanced Crystal Orientation Effects")
                            
                            orientation_effects = []
                            for angle in [0, 30, 45, 54.7, 60, 90]:
                                effect = st.session_state.physics_analyzer.analyze_crystal_orientation_effects(
                                    {},  # Empty stress data for basic analysis
                                    angle
                                )
                                orientation_effects.append(effect)
                            
                            if orientation_effects:
                                df_orientation = pd.DataFrame(orientation_effects)
                                
                                # Apply enhanced styling
                                styled_df = df_orientation.style \
                                    .background_gradient(subset=['schmid_factor', 'habit_plane_proximity'], cmap='YlOrRd') \
                                    .applymap(lambda x: 'background-color: #90EE90' if x else 'background-color: #FFCCCB', 
                                            subset=['is_near_habit_plane'])
                                
                                st.dataframe(styled_df, use_container_width=True)
                        
                        with adv_tab2:
                            # Enhanced sintering optimization
                            st.markdown("##### 🔥 Enhanced Sintering Optimization Analysis")
                            
                            # Find optimal orientation for each defect
                            optimal_data = []
                            for key, data in defect_comparison.items():
                                if 'angles' in data and 'sintering_temps' in data:
                                    temps = data['sintering_temps']
                                    angles = data['angles']
                                    
                                    # Find minimum sintering temperature
                                    min_temp_idx = np.argmin(temps)
                                    min_temp = temps[min_temp_idx]
                                    opt_angle = angles[min_temp_idx]
                                    
                                    optimal_data.append({
                                        'Defect Type': data['defect_type'],
                                        'Optimal Angle (°)': f"{opt_angle:.1f}",
                                        'Min T_sinter (K)': f"{min_temp:.1f}",
                                        'Min T_sinter (°C)': f"{min_temp-273.15:.1f}",
                                        'Temperature Reduction (K)': f"{623.0 - min_temp:.1f}",
                                        'Is Near Habit Plane': abs(opt_angle - 54.7) < 5.0,
                                        'Eigen Strain': data.get('eigen_strain', 0.0)
                                    })
                            
                            if optimal_data:
                                df_optimal = pd.DataFrame(optimal_data)
                                
                                # Apply enhanced styling
                                styled_optimal = df_optimal.style \
                                    .background_gradient(subset=['Min T_sinter (K)', 'Temperature Reduction (K)'], cmap='Blues_r') \
                                    .applymap(lambda x: 'background-color: #90EE90' if x == 'True' else 'background-color: #FFCCCB', 
                                            subset=['Is Near Habit Plane'])
                                
                                st.dataframe(styled_optimal, use_container_width=True)
                                
                                # Enhanced recommendation
                                st.markdown("##### 💡 Enhanced Optimization Recommendation")
                                
                                best_defect = min(optimal_data, key=lambda x: float(x['Min T_sinter (K)'].split()[0]))
                                
                                # Enhanced recommendation display
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); 
                                            padding: 1.5rem; 
                                            border-radius: 1rem; 
                                            color: white;
                                            margin: 1rem 0;">
                                <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem;">🎯 Recommended Configuration</div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                <div style="font-size: 1.1rem; opacity: 0.9;">Defect Type</div>
                                <div style="font-size: 1.4rem; font-weight: bold;">{best_defect['Defect Type']}</div>
                                </div>
                                <div>
                                <div style="font-size: 1.1rem; opacity: 0.9;">Optimal Orientation</div>
                                <div style="font-size: 1.4rem; font-weight: bold;">{best_defect['Optimal Angle (°)']}°</div>
                                </div>
                                <div>
                                <div style="font-size: 1.1rem; opacity: 0.9;">Min Temperature</div>
                                <div style="font-size: 1.4rem; font-weight: bold;">{best_defect['Min T_sinter (K)']} K</div>
                                <div style="font-size: 1.0rem;">({best_defect['Min T_sinter (°C)']}°C)</div>
                                </div>
                                <div>
                                <div style="font-size: 1.1rem; opacity: 0.9;">Reduction</div>
                                <div style="font-size: 1.4rem; font-weight: bold;">{best_defect['Temperature Reduction (K)']} K</div>
                                <div style="font-size: 1.0rem;">from reference</div>
                                </div>
                                </div>
                                <div style="margin-top: 1rem; font-size: 1.1rem; padding: 0.8rem; background: rgba(255,255,255,0.2); border-radius: 0.5rem;">
                                <strong>Note:</strong> {best_defect['Defect Type']} provides the lowest sintering temperature among all analyzed defect types.
                                </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with adv_tab3:
                            # Enhanced comprehensive export
                            st.markdown("##### 📦 Enhanced Comprehensive Export Package")
                            
                            st.markdown("""
                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.8rem; border: 2px solid #dee2e6;">
                            <div style="font-size: 1.4rem; font-weight: bold; margin-bottom: 1rem; color: #333;">The enhanced comprehensive export package includes:</div>
                            <ol style="font-size: 1.1rem; line-height: 2.0;">
                            <li><strong>Complete JSON report</strong> with all enhanced analysis data</li>
                            <li><strong>High-resolution CSV files</strong> for all datasets suitable for publication plotting</li>
                            <li><strong>Enhanced README</strong> with detailed analysis documentation</li>
                            <li><strong>Python script</strong> for enhanced data processing and visualization</li>
                            <li><strong>Configuration file</strong> with all analysis parameters</li>
                            <li><strong>Publication-quality figures</strong> in multiple formats (PNG, PDF, SVG)</li>
                            </ol>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Prepare enhanced comprehensive report
                            if st.button("🛠️ Prepare Enhanced Comprehensive Report", use_container_width=True,
                                       help="Prepare enhanced report with publication-quality outputs"):
                                with st.spinner("Preparing enhanced comprehensive report..."):
                                    report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                        vicinity_sweep,
                                        defect_comparison,
                                        st.session_state.current_target_params,
                                        st.session_state.current_analysis_params
                                    )
                                    
                                    # Add enhanced metadata
                                    report['metadata']['enhanced_analysis'] = True
                                    report['metadata']['publication_quality'] = publication_mode
                                    report['metadata']['generated_with'] = "Enhanced Habit Plane Visualizer v2.0"
                                    
                                    zip_buffer = st.session_state.results_manager.create_comprehensive_export(report)
                                    
                                    st.download_button(
                                        label="📥 Download Enhanced Complete Package",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"enhanced_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip",
                                        use_container_width=True
                                    )
            
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Enhanced Analysis'")
                
                # Enhanced dashboard features
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 2rem; 
                            border-radius: 1rem; 
                            border: 3px solid #3B82F6;
                            margin: 1rem 0;">
                <div style="font-size: 1.8rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1.5rem; text-align: center;">📊 Enhanced Comprehensive Dashboard Features</div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
                <div style="background: white; padding: 1.5rem; border-radius: 0.7rem; border: 2px solid #E5E7EB;">
                <div style="font-size: 1.4rem; font-weight: bold; color: #3B82F6; margin-bottom: 1rem;">🎨 Multi-Panel Visualization</div>
                <ul style="font-size: 1.1rem; line-height: 1.8;">
                <li>Enhanced sunburst charts for polar stress visualization</li>
                <li>Publication-quality line plots with larger labels</li>
                <li>Enhanced radar charts with adjustable parameters</li>
                <li>Defect comparison across all types with improved visibility</li>
                </ul>
                </div>
                
                <div style="background: white; padding: 1.5rem; border-radius: 0.7rem; border: 2px solid #E5E7EB;">
                <div style="font-size: 1.4rem; font-weight: bold; color: #10B981; margin-bottom: 1rem;">🔬 Advanced Analysis</div>
                <ul style="font-size: 1.1rem; line-height: 1.8;">
                <li>Physics-based stress intensity calculations</li>
                <li>Crystal orientation effects with enhanced visualization</li>
                <li>Sintering temperature optimization algorithms</li>
                <li>System classification mapping with clear boundaries</li>
                </ul>
                </div>
                
                <div style="background: white; padding: 1.5rem; border-radius: 0.7rem; border: 2px solid #E5E7EB;">
                <div style="font-size: 1.4rem; font-weight: bold; color: #F59E0B; margin-bottom: 1rem;">📤 Comprehensive Export</div>
                <ul style="font-size: 1.1rem; line-height: 1.8;">
                <li>Complete data package with enhanced outputs</li>
                <li>Processing scripts for further analysis</li>
                <li>Documentation and configuration files</li>
                <li>Publication-quality figure generation</li>
                </ul>
                </div>
                </div>
                
                <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(59, 130, 246, 0.1); border-radius: 0.7rem;">
                <div style="font-size: 1.3rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1rem;">🚀 To generate the enhanced dashboard:</div>
                <ol style="font-size: 1.1rem; line-height: 2.0;">
                <li>Run both Enhanced Vicinity Analysis and Enhanced Defect Comparison</li>
                <li>Click "Generate Enhanced Dashboard" in the sidebar</li>
                <li>Explore the comprehensive enhanced results</li>
                <li>Export publication-quality visualizations</li>
                </ol>
                </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick status check with enhanced styling
                st.markdown("#### 🔍 Enhanced Analysis Status")
                
                col_status1, col_status2 = st.columns(2)
                
                with col_status1:
                    if 'vicinity_sweep' in st.session_state:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                                    padding: 1rem; 
                                    border-radius: 0.7rem; 
                                    color: white;
                                    text-align: center;">
                        <div style="font-size: 1.3rem; font-weight: bold;">✅ Enhanced Vicinity Analysis</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Data available and ready for dashboard</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                                    padding: 1rem; 
                                    border-radius: 0.7rem; 
                                    color: white;
                                    text-align: center;">
                        <div style="font-size: 1.3rem; font-weight: bold;">⚠️ Enhanced Vicinity Analysis</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Not yet run - click button above</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_status2:
                    if 'defect_comparison' in st.session_state:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
                                    padding: 1rem; 
                                    border-radius: 0.7rem; 
                                    color: white;
                                    text-align: center;">
                        <div style="font-size: 1.3rem; font-weight: bold;">✅ Enhanced Defect Comparison</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Data available and ready for dashboard</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                                    padding: 1rem; 
                                    border-radius: 0.7rem; 
                                    color: white;
                                    text-align: center;">
                        <div style="font-size: 1.3rem; font-weight: bold;">⚠️ Enhanced Defect Comparison</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Not yet run - click button above</div>
                        </div>
                        """, unsafe_allow_html=True)

# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
