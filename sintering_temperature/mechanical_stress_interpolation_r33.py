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
        if 'sigma_hydro' in stress_data and 'max_abs' in stress_data['sigma_hydro']:
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
        if 'sigma_hydro' in stress_data and 'max_abs' in stress_data['sigma_hydro']:
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
                if 'params' in 
                    standardized['params'] = data['params']
                elif 'parameters' in 
                    standardized['params'] = data['parameters']
                
                # Extract history
                if 'history' in 
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
                if 'metadata' in 
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
                    if comp in region_
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
                                if comp in region_
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
# ENHANCED VISUALIZER FOR HABIT PLANE VICINITY
# =============================================
class HabitPlaneVisualizer:
    """Specialized visualizer for habit plane vicinity analysis"""
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        
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
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text="No data available for the selected vicinity range",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14)
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
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text=f"No data in vicinity range ({self.habit_angle - vicinity_range:.1f}° to {self.habit_angle + vicinity_range:.1f}°)",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14)
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
                size=10,
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
                    size=25,
                    color='rgb(46, 204, 113)',
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(size=14, color='black', family='Arial Black'),
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
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(size=12, color='black'),
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(size=14, color='black')
                    ),
                    range=[0, max(vic_stresses) * radius_scale * 1.2]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)),
                    ticktext=[f'{i}°' for i in range(int(vic_angles[0]), int(vic_angles[-1]) + 1, 15)],
                    tickfont=dict(size=12, color='black'),
                    period=360,
                    thetaunit="degrees"
                ),
                bgcolor="rgba(240, 240, 240, 0.3)",
                sector=[vic_angles[0], vic_angles[-1]]
            ),
            showlegend=True,
            legend=dict(
                x=1.2,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=900,
            height=700,
            margin=dict(l=100, r=200, t=100, b=100)
        )
        
        return fig

    def create_stress_comparison_radar(
        self,
        comparison_data,
        title="Stress Component Comparison",
        show_labels=True,
        radial_tick_color='black',
        radial_tick_width=2,
        angular_tick_color='black',
        angular_tick_width=2,
        angular_tick_step=45,
        colormap='default',
        custom_component_labels=None,
        font_size_title=20,
        font_size_axis=14,
        font_size_tick=12,
        bgcolor="rgba(240, 240, 240, 0.3)"
    ):
        """
        Create radar chart comparing different stress components with extensive customization.
        """
        import itertools
        import numpy as np
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt

        fig = go.Figure()

        if not comparison_
            fig.update_layout(
                title=dict(text=f"{title} - No Data Available", font=dict(size=font_size_title, family="Arial Black", color='darkblue'), x=0.5),
                annotations=[dict(text="No data available for the selected vicinity range", x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
                width=900, height=700
            )
            return fig

        max_stress = 0
        color_cycle = itertools.cycle(['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#D4A5A5', '#88D8B0', '#FFB347', '#B388EB'])

        # Resolve colormap
        if colormap == 'default':
            use_custom_colors = True
        else:
            use_custom_colors = False
            if colormap in ['viridis', 'plasma', 'RdBu', 'hot']:
                color_scale = plt.get_cmap(colormap)
                n_traces = len(comparison_data)
                color_list = [f'rgb{tuple(int(c*255) for c in color_scale(i/n_traces)[:3])}' for i in range(n_traces)]
                color_iter = iter(color_list)
            else:
                color_iter = itertools.cycle(colormap) if isinstance(colormap, list) else color_cycle

        for idx, (entry_name, entry_data) in enumerate(comparison_data.items()):
            if not isinstance(entry_data, dict):
                continue
            if 'angles' not in entry_data or 'stresses' not in entry_
                continue

            try:
                angles = np.array(entry_data['angles'])
                stresses_dict = entry_data['stresses']
                if not stresses_dict:
                    continue

                # Assume first stress component if multiple
                comp_key = list(stresses_dict.keys())[0]
                stresses = np.array(stresses_dict[comp_key])

                if len(angles) == 0 or len(stresses) == 0 or len(angles) != len(stresses):
                    continue

                # Close loop
                angles_closed = np.append(angles, angles[0])
                stresses_closed = np.append(stresses, stresses[0])

                current_max = np.nanmax(stresses)
                if not np.isnan(current_max):
                    max_stress = max(max_stress, current_max)

                # Determine color
                if use_custom_colors:
                    color = entry_data.get('color', next(color_cycle))
                else:
                    color = next(color_iter)

                # Custom label
                display_name = entry_data.get('defect_type', entry_name)
                if custom_component_labels and comp_key in custom_component_labels:
                    display_name += f" – {custom_component_labels[comp_key]}"

                fig.add_trace(go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.3)'),
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color),
                    name=display_name if show_labels else None,
                    hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                    showlegend=show_labels
                ))
            except Exception as e:
                print(f"Error processing entry {entry_name}: {e}")
                continue

        if len(fig.data) == 0:
            fig.update_layout(
                title=dict(text=f"{title} - No Valid Data", font=dict(size=font_size_title, family="Arial Black", color='darkblue'), x=0.5),
                annotations=[dict(text="No stress data available for the selected defect types in this vicinity range", x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
                width=900, height=700
            )
            return fig

        # Habit plane line
        if max_stress > 0:
            fig.add_trace(go.Scatterpolar(
                r=[0, max_stress * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
                name=f'Habit Plane ({self.habit_angle}°)' if show_labels else None,
                hoverinfo='skip',
                showlegend=show_labels
            ))

        # Angular tick values
        angular_tickvals = list(range(0, 360, angular_tick_step))
        angular_ticktext = [f'{v}°' for v in angular_tickvals]

        fig.update_layout(
            title=dict(text=title, font=dict(size=font_size_title, family="Arial Black", color='darkblue'), x=0.5),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor=radial_tick_color,
                    linewidth=radial_tick_width,
                    tickfont=dict(size=font_size_tick, color=radial_tick_color),
                    title=dict(text='Stress (GPa)', font=dict(size=font_size_axis, color='black')),
                    range=[0, max_stress * 1.2 if max_stress > 0 else 1]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor=angular_tick_color,
                    linewidth=angular_tick_width,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=angular_tickvals,
                    ticktext=angular_ticktext,
                    tickfont=dict(size=font_size_tick, color=angular_tick_color),
                    period=360
                ),
                bgcolor=bgcolor
            ),
            showlegend=show_labels,
            legend=dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=font_size_tick, family='Arial')
            ),
            width=900,
            height=700
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
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text="No defect comparison data available",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14)
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
                    marker=dict(size=6, color=color),
                    name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                    hovertemplate='Orientation: %{x:.2f}°<br>Stress: %{y:.4f} GPa<extra></extra>',
                    showlegend=True
                ))
        
        # If no traces were added, return empty figure with message
        if len(fig.data) == 0:
            fig.update_layout(
                title=dict(
                    text=f"{title} - No Data Available",
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text=f"No data available for stress component: {stress_component}",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14)
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
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} - {stress_component.replace('_', ' ').title()}",
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text='Orientation (°)', font=dict(size=14, color='black')),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1,
                range=[0, 360]
            ),
            yaxis=dict(
                title=dict(text=f'{stress_component.replace("_", " ").title()} Stress (GPa)',
                         font=dict(size=14, color='black')),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=1000,
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def create_sintering_temperature_radar(self, sintering_data, defect_type='Twin',
                                         title="Sintering Temperature Prediction"):
        """Create radar chart for sintering temperature prediction"""
        fig = go.Figure()
        
        # Check if sintering_data is empty
        if not sintering_
            fig.update_layout(
                title=dict(
                    text=f"{title} - No Data Available",
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text="No sintering data available",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                ],
                width=900,
                height=700
            )
            return fig
        
        # Add sintering temperature trace
        if 'angles' in sintering_data and 'sintering_temps' in sintering_
            try:
                angles = np.array(sintering_data['angles'])
                temps = sintering_data['sintering_temps'].get('arrhenius_defect', [])
                if temps is None:
                    temps = []
                temps_array = np.array(temps)
                
                # Check if arrays are empty
                if len(angles) == 0 or len(temps_array) == 0:
                    fig.update_layout(
                        title=dict(
                            text=f"{title} - No Data Available",
                            font=dict(size=20, family="Arial Black", color='darkblue'),
                            x=0.5
                        ),
                        annotations=[
                            dict(
                                text="No sintering temperature data available",
                                x=0.5,
                                y=0.5,
                                showarrow=False,
                                font=dict(size=14)
                            )
                        ],
                        width=900,
                        height=700
                    )
                    return fig
                
                # Close the loop for radar chart
                angles_closed = np.append(angles, angles[0])
                temps_closed = np.append(temps_array, temps_array[0])
                
                fig.add_trace(go.Scatterpolar(
                    r=temps_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor='rgba(255, 140, 0, 0.3)',
                    line=dict(color='rgb(255, 140, 0)', width=3),
                    marker=dict(size=6, color='rgb(255, 140, 0)'),
                    name='Sintering Temperature (K)',
                    hovertemplate='Orientation: %{theta:.2f}°<br>T_sinter: %{r:.1f} K<extra></extra>',
                    showlegend=True
                ))
            except Exception as e:
                print(f"Error creating sintering temperature trace: {e}")
        
        # Add stress trace on secondary radial axis
        if 'stresses' in sintering_data and 'sigma_hydro' in sintering_data['stresses']:
            try:
                stresses = sintering_data['stresses']['sigma_hydro']
                if stresses is None:
                    stresses = []
                stresses_array = np.array(stresses)
                
                # Check if stresses array is not empty
                if len(stresses_array) > 0 and 'angles' in sintering_
                    angles = np.array(sintering_data['angles'])
                    if len(angles) > 0:
                        stresses_closed = np.append(stresses_array, stresses_array[0])
                        
                        # Scale stresses for visualization
                        stress_max = max(stresses_array) if len(stresses_array) > 0 else 1
                        temp_max = max(temps_array) if 'temps_array' in locals() and len(temps_array) > 0 else 1
                        stress_scale = temp_max / (stress_max * 2) if stress_max > 0 else 1
                        
                        fig.add_trace(go.Scatterpolar(
                            r=stresses_closed * stress_scale,
                            theta=angles_closed,
                            line=dict(color='rgb(31, 119, 180)', width=3, dash='dot'),
                            marker=dict(size=6, color='rgb(31, 119, 180)'),
                            name='Hydrostatic Stress (scaled)',
                            hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{custom.4f} GPa<extra></extra>',
                            customdata=stresses_closed,
                            showlegend=True
                        ))
            except Exception as e:
                print(f"Error creating stress trace: {e}")
        
        # Highlight habit plane
        fig.add_trace(go.Scatterpolar(
            r=[0, 1000],
            theta=[self.habit_angle, self.habit_angle],
            mode='lines',
            line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
            name=f'Habit Plane ({self.habit_angle}°)',
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Add reference temperature lines
        T0 = 623.0  # Reference temperature
        T_min = 367.0  # Minimum temperature
        
        if 'angles' in sintering_
            try:
                angles = np.array(sintering_data['angles'])
                if len(angles) > 0:
                    angles_closed = np.append(angles, angles[0])
                    fig.add_trace(go.Scatterpolar(
                        r=[T0] * len(angles_closed),
                        theta=angles_closed,
                        mode='lines',
                        line=dict(color='rgba(0, 0, 0, 0.3)', width=2, dash='dash'),
                        name=f'T₀ = {T0} K',
                        hoverinfo='skip',
                        showlegend=True
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=[T_min] * len(angles_closed),
                        theta=angles_closed,
                        mode='lines',
                        line=dict(color='rgba(0, 0, 0, 0.3)', width=2, dash='dash'),
                        name=f'T_min = {T_min} K',
                        hoverinfo='skip',
                        showlegend=True
                    ))
            except:
                pass
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} - {defect_type}",
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(size=12, color='black'),
                    title=dict(text='Sintering Temperature (K)', font=dict(size=14, color='black')),
                    range=[0, 1000]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                    tickfont=dict(size=12, color='black'),
                    period=360
                ),
                bgcolor="rgba(240, 240, 240, 0.3)"
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
    
    def create_comprehensive_dashboard(self, vicinity_sweep, defect_comparison=None,
                                     title="Comprehensive Habit Plane Analysis"):
        """Create comprehensive dashboard with multiple visualizations"""
        # Check if data is empty
        if not vicinity_sweep or 'angles' not in vicinity_sweep:
            # Return empty figure with message
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=('No Data Available'),
            )
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=24, family="Arial Black", color='darkblue'),
                    x=0.5,
                    y=0.98
                ),
                annotations=[
                    dict(
                        text="No vicinity sweep data available",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16)
                    )
                ],
                height=600,
                width=800,
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Hydrostatic Stress Vicinity',
                'Von Mises Stress Vicinity',
                'Stress Magnitude Vicinity',
                'Defect Type Comparison',
                'Sintering Temperature Prediction',
                'Stress Components Radar'
            ),
            specs=[
                [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
                [{'type': 'scatter'}, {'type': 'polar'}, {'type': 'polar'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        angles = vicinity_sweep['angles']
        
        # 1. Hydrostatic stress sunburst
        if 'stresses' in vicinity_sweep and 'sigma_hydro' in vicinity_sweep['stresses']:
            sigma_hydro = vicinity_sweep['stresses']['sigma_hydro']
            if sigma_hydro is not None and len(sigma_hydro) > 0 and len(sigma_hydro) == len(angles):
                fig.add_trace(
                    go.Scatterpolar(
                        r=sigma_hydro,
                        theta=angles,
                        mode='lines+markers',
                        marker=dict(size=4, color=sigma_hydro, colorscale='RdBu'),
                        line=dict(color='rgba(31, 119, 180, 0.7)', width=2),
                        name='σ_hydro',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. Von Mises stress sunburst
        if 'stresses' in vicinity_sweep and 'von_mises' in vicinity_sweep['stresses']:
            von_mises = vicinity_sweep['stresses']['von_mises']
            if von_mises is not None and len(von_mises) > 0 and len(von_mises) == len(angles):
                fig.add_trace(
                    go.Scatterpolar(
                        r=von_mises,
                        theta=angles,
                        mode='lines+markers',
                        marker=dict(size=4, color=von_mises, colorscale='Viridis'),
                        line=dict(color='rgba(255, 127, 14, 0.7)', width=2),
                        name='σ_vm',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Stress magnitude sunburst
        if 'stresses' in vicinity_sweep and 'sigma_mag' in vicinity_sweep['stresses']:
            sigma_mag = vicinity_sweep['stresses']['sigma_mag']
            if sigma_mag is not None and len(sigma_mag) > 0 and len(sigma_mag) == len(angles):
                fig.add_trace(
                    go.Scatterpolar(
                        r=sigma_mag,
                        theta=angles,
                        mode='lines+markers',
                        marker=dict(size=4, color=sigma_mag, colorscale='Plasma'),
                        line=dict(color='rgba(44, 160, 44, 0.7)', width=2),
                        name='σ_mag',
                        showlegend=False
                    ),
                    row=1, col=3
                )
        
        # 4. Defect type comparison
        if defect_comparison:
            added_defects = 0
            for defect_key, data in list(defect_comparison.items())[:4]:  # Limit to 4 defects
                if not isinstance(data, dict):
                    continue
                
                defect_type = data.get('defect_type', 'Unknown')
                if ('angles' in data and 'stresses' in data and
                    'sigma_hydro' in data['stresses'] and
                    data['angles'] is not None and
                    data['stresses']['sigma_hydro'] is not None and
                    len(data['angles']) > 0 and
                    len(data['stresses']['sigma_hydro']) > 0):
                    
                    try:
                        angles_array = np.array(data['angles'])
                        stresses_array = np.array(data['stresses']['sigma_hydro'])
                        if len(angles_array) == len(stresses_array):
                            fig.add_trace(
                                go.Scatter(
                                    x=angles_array,
                                    y=stresses_array,
                                    mode='lines',
                                    line=dict(width=2, color=data.get('color', 'black')),
                                    name=defect_type,
                                    showlegend=False
                                ),
                                row=2, col=1
                            )
                            added_defects += 1
                    except:
                        continue
            
            # If no defects were added, add a placeholder
            if added_defects == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[0, 360],
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 5. Sintering temperature radar
        if 'sintering_temps' in vicinity_sweep and 'arrhenius_defect' in vicinity_sweep['sintering_temps']:
            sintering_temps = vicinity_sweep['sintering_temps']['arrhenius_defect']
            if sintering_temps is not None and len(sintering_temps) > 0 and len(sintering_temps) == len(angles):
                fig.add_trace(
                    go.Scatterpolar(
                        r=sintering_temps,
                        theta=angles,
                        mode='lines+markers',
                        marker=dict(size=4, color=sintering_temps, colorscale='Hot'),
                        line=dict(color='rgba(255, 140, 0, 0.7)', width=2),
                        name='T_sinter',
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # 6. Stress components radar (normalized)
        if 'stresses' in vicinity_sweep:
            # Find max stress for normalization
            max_stress = 0
            for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                if comp in vicinity_sweep['stresses']:
                    stresses = vicinity_sweep['stresses'][comp]
                    if stresses is not None and len(stresses) > 0:
                        try:
                            current_max = max(stresses)
                            max_stress = max(max_stress, current_max)
                        except:
                            continue
            
            if max_stress > 0:
                added_components = 0
                for comp_name in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                    if comp_name in vicinity_sweep['stresses']:
                        stresses = vicinity_sweep['stresses'][comp_name]
                        if stresses is not None and len(stresses) > 0 and len(stresses) == len(angles):
                            try:
                                normalized = np.array(stresses) / max_stress
                                fig.add_trace(
                                    go.Scatterpolar(
                                        r=normalized,
                                        theta=angles,
                                        mode='lines',
                                        line=dict(width=2, color=self.stress_colors.get(comp_name, 'black')),
                                        name=comp_name,
                                        showlegend=False
                                    ),
                                    row=2, col=3
                                )
                                added_components += 1
                            except:
                                continue
                
                # If no components were added, add a placeholder
                if added_components == 0:
                    fig.add_trace(
                        go.Scatterpolar(
                            r=[0, 1],
                            theta=[0, 0],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ),
                        row=2, col=3
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.98
            ),
            height=1200,
            width=1600,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Update polar subplot layouts
        for i in range(1, 4):
            fig.update_polars(
                radialaxis_range=[0, 1.2],
                angularaxis_rotation=90,
                angularaxis_direction="clockwise",
                sector=[min(angles), max(angles)] if len(angles) > 0 else [0, 360],
                row=1, col=i
            )
        
        fig.update_polars(
            radialaxis_range=[0, 1.2],
            angularaxis_rotation=90,
            angularaxis_direction="clockwise",
            sector=[0, 360],
            row=2, col=2
        )
        
        fig.update_polars(
            radialaxis_range=[0, 1.2],
            angularaxis_rotation=90,
            angularaxis_direction="clockwise",
            sector=[0, 360],
            row=2, col=3
        )
        
        return fig

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
            
            if 'angles' in sweep_
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
                if 'angles' in data and 'stresses' in 
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
            
            if comparison_
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
               f'Habit Plane Analysis\\n'
               f'Angle: {angles[habit_idx]:.1f}°\\n'
               f'σ_hydro: {sweep["stresses"]["sigma_hydro"][habit_idx]:.3f} GPa\\n'
               f'T_sinter: {sweep["sintering_temps"]["arrhenius_defect"][habit_idx]:.1f} K',
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
        print(f"Error processing  {e}")
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
# MAIN APPLICATION WITH COMPREHENSIVE FEATURES
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Ag FCC Twin: Habit Plane Vicinity Analysis",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.0rem !important;
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
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        font-weight: bold;
        border: 3px solid #047857;
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
    .isf-card {
        border-color: #FF6B6B;
        background-color: rgba(255, 107, 107, 0.1);
    }
    .esf-card {
        border-color: #4ECDC4;
        background-color: rgba(78, 205, 196, 0.1);
    }
    .twin-card {
        border-color: #45B7D1;
        background-color: rgba(69, 183, 209, 0.1);
    }
    .perfect-card {
        border-color: #96CEB4;
        background-color: rgba(150, 206, 180, 0.1);
    }
    .stress-metric {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1rem;
        border-radius: 0.6rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
    }
    .temperature-metric {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        padding: 1rem;
        border-radius: 0.6rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
    }
    .system-metric {
        padding: 0.8rem;
        border-radius: 0.5rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.3rem;
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
        font-size: 1.2rem;
        padding: 1.2rem;
        background-color: #F8FAFC;
        border-radius: 0.6rem;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
        text-align: center;
    }
    .tab-content {
        padding: 1.5rem;
        background-color: white;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px 5px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Ag FCC Twin: Habit Plane Vicinity Analysis</h1>', unsafe_allow_html=True)
    
    # Physics equations
    st.markdown("""
    <div class="latex-equation">
    <strong>Physics of Stress-Modified Sintering:</strong><br>
    <div style="margin: 10px 0;">
    <strong>Arrhenius Equation with Stress:</strong> D = D₀ exp[-(Q_a - Ωσ)/(k_BT)]
    </div>
    <div style="margin: 10px 0;">
    <strong>Sintering Temperature:</strong> T_sinter(σ) = (Q_a - Ω|σ|) / [k_B ln(D₀/D_crit)]
    </div>
    <div style="margin: 10px 0;">
    <strong>Hydrostatic Stress:</strong> σ_h = (σ₁ + σ₂ + σ₃)/3
    </div>
    <div style="margin: 10px 0;">
    <strong>Von Mises Stress:</strong> σ_vm = √[½((σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²)]
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Habit plane information
    st.markdown("""
    <div class="habit-plane-highlight">
    <div style="display: flex; align-items: center; justify-content: space-between;">
    <div>
    <h3 style="margin: 0; color: white;">🎯 Ag FCC Twin Habit Plane: 54.7°</h3>
    <p style="margin: 0.5rem 0 0 0; color: white; opacity: 0.9;">
    • {111} crystal planes • Maximum stress concentration • Optimal for defect engineering
    </p>
    </div>
    <div style="font-size: 2.5rem;">⚛️</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
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
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HabitPlaneVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="physics-header">⚙️ Analysis Configuration</h2>', unsafe_allow_html=True)
        
        # Analysis mode
        analysis_mode = st.radio(
            "Select Analysis Mode:",
            ["Habit Plane Vicinity", "Defect Type Comparison", "Comprehensive Dashboard", "Single Point Analysis"],
            index=0,
            help="Choose the type of analysis to perform"
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
                    st.write(f"- {defect}: {count}")
                
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
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        
        # Show defect cards
        st.markdown("##### 🔬 Defect Properties")
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
        
        # Generate button
        st.markdown("---")
        generate_text = "🚀 Generate Analysis" if analysis_mode != "Comprehensive Dashboard" else "📊 Generate Dashboard"
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
        
        ### Visualization
        - Sunburst charts for polar visualization
        - Radar charts for component comparison
        - Interactive plots with Plotly
        - Comprehensive dashboards
        """)
    else:
        # Create tabs for different analysis modes
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Overview",
            "📈 Vicinity Analysis",
            "🔬 Defect Comparison",
            "📊 Comprehensive Dashboard"
        ])
        
        with tab1:
            st.markdown('<h2 class="physics-header">🏠 Analysis Overview</h2>', unsafe_allow_html=True)
            
            # Quick statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Loaded Solutions", len(st.session_state.solutions))
            with col_stat2:
                defect_types = set()
                for sol in st.session_state.solutions:
                    params = sol.get('params', {})
                    defect_types.add(params.get('defect_type', 'Unknown'))
                st.metric("Defect Types", len(defect_types))
            with col_stat3:
                orientations = []
                for sol in st.session_state.solutions:
                    params = sol.get('params', {})
                    theta = params.get('theta', 0.0)
                    if theta is not None:
                        orientations.append(np.degrees(theta) % 360)
                if orientations:
                    st.metric("Orientation Range", f"{min(orientations):.1f}° - {max(orientations):.1f}°")
                else:
                    st.metric("Orientation Range", "N/A")
            with col_stat4:
                has_physics = sum(1 for sol in st.session_state.solutions if sol.get('physics_analysis'))
                st.metric("Physics Analyzed", f"{has_physics}/{len(st.session_state.solutions)}")
            
            # Defect type cards
            st.markdown("#### 🔬 Defect Types with Eigen Strains")
            col_def1, col_def2, col_def3, col_def4 = st.columns(4)
            defect_info = [
                ("ISF", "Intrinsic Stacking Fault", 0.71, "#FF6B6B"),
                ("ESF", "Extrinsic Stacking Fault", 1.41, "#4ECDC4"),
                ("Twin", "Coherent Twin Boundary", 2.12, "#45B7D1"),
                ("No Defect", "Perfect Crystal", 0.0, "#96CEB4")
            ]
            for i, (name, desc, strain, color) in enumerate(defect_info):
                with [col_def1, col_def2, col_def3, col_def4][i]:
                    st.markdown(f"""
                    <div class="defect-card {'isf' if name == 'ISF' else 'esf' if name == 'ESF' else 'twin' if name == 'Twin' else 'perfect'}-card">
                        <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{name}</div>
                        <div style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">{desc}</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #333;">ε* = {strain}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # System classification
            st.markdown("#### 🏷️ System Classification")
            col_sys1, col_sys2, col_sys3 = st.columns(3)
            system_info = [
                ("System 1", "Perfect Crystal", "σ < 5 GPa", "620-630 K", "#10B981"),
                ("System 2", "SF/Twins", "5 ≤ σ < 20 GPa", "450-550 K", "#F59E0B"),
                ("System 3", "Plastic Deformation", "σ ≥ 20 GPa", "350-400 K", "#EF4444")
            ]
            for i, (name, desc, stress_range, temp_range, color) in enumerate(system_info):
                with [col_sys1, col_sys2, col_sys3][i]:
                    st.markdown(f"""
                    <div class="system-metric" style="background-color: {color};">
                        <div style="font-size: 1.2rem; font-weight: bold;">{name}</div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">{desc}</div>
                        <div style="font-size: 0.8rem; margin-top: 0.5rem;">{stress_range}</div>
                        <div style="font-size: 0.8rem;">{temp_range}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Quick analysis buttons
            st.markdown("#### ⚡ Quick Analysis")
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                if st.button("🎯 Analyze Habit Plane", use_container_width=True):
                    st.session_state.quick_analysis = "habit_plane"
            with col_q2:
                if st.button("🔬 Compare Defects", use_container_width=True):
                    st.session_state.quick_analysis = "defect_compare"
            with col_q3:
                if st.button("📊 Generate Dashboard", use_container_width=True):
                    st.session_state.quick_analysis = "dashboard"
        
        with tab2:
            st.markdown('<h2 class="physics-header">📈 Habit Plane Vicinity Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'habit_plane':
                with st.spinner("Performing vicinity analysis..."):
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
                        'use_physics_constraints': use_physics_constraints
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
                        st.success(f"✅ Generated vicinity sweep with {analysis_params['n_points']} points")
                        # Store in session state
                        st.session_state.vicinity_sweep = vicinity_sweep
                        st.session_state.current_target_params = target_params
                        st.session_state.current_analysis_params = analysis_params
                        
                        # Display results
                        st.markdown("#### 📊 Analysis Results")
                        # Create columns for metrics
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        
                        # Habit plane stress
                        angles = np.array(vicinity_sweep['angles'])
                        habit_angle = 54.7
                        habit_idx = np.argmin(np.abs(angles - habit_angle))
                        
                        with col_res1:
                            sigma_hydro = vicinity_sweep['stresses']['sigma_hydro'][habit_idx]
                            st.metric(
                                "Habit Plane σ_h",
                                f"{sigma_hydro:.3f} GPa",
                                "Hydrostatic Stress"
                            )
                        with col_res2:
                            von_mises = vicinity_sweep['stresses']['von_mises'][habit_idx]
                            st.metric(
                                "Habit Plane σ_vm",
                                f"{von_mises:.3f} GPa",
                                "Von Mises Stress"
                            )
                        with col_res3:
                            T_sinter = vicinity_sweep['sintering_temps']['arrhenius_defect'][habit_idx]
                            st.metric(
                                "Habit Plane T_sinter",
                                f"{T_sinter:.1f} K",
                                f"{T_sinter-273.15:.1f}°C"
                            )
                        with col_res4:
                            system_info = st.session_state.sintering_calculator.map_system_to_temperature(sigma_hydro)
                            st.metric(
                                "System Classification",
                                system_info[0].split('(')[0].strip(),
                                system_info[1][1] - system_info[1][0]
                            )
                        
                        # Create visualizations
                        st.markdown("#### 📈 Visualizations")
                        
                        # Sunburst for hydrostatic stress
                        fig_sunburst = st.session_state.visualizer.create_vicinity_sunburst(
                            vicinity_sweep['angles'],
                            vicinity_sweep['stresses']['sigma_hydro'],
                            stress_component='sigma_hydro',
                            title=f"Habit Plane Vicinity - {defect_type}",
                            radius_scale=1.0
                        )
                        if fig_sunburst:
                            st.plotly_chart(fig_sunburst, use_container_width=True)
                        
                        # Line plots for all stress components
                        fig_line, ax_line = plt.subplots(figsize=(12, 6))
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                        for idx, (comp, color) in enumerate(zip(['sigma_hydro', 'von_mises', 'sigma_mag'], colors)):
                            ax_line.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses'][comp],
                                color=color,
                                linewidth=3,
                                label=comp.replace('_', ' ').title()
                            )
                        ax_line.axvline(habit_angle, color='green', linestyle='--', linewidth=2,
                                      label=f'Habit Plane ({habit_angle}°)')
                        ax_line.set_xlabel('Orientation (°)', fontsize=12)
                        ax_line.set_ylabel('Stress (GPa)', fontsize=12)
                        ax_line.set_title('Stress Components in Habit Plane Vicinity', fontsize=14, fontweight='bold')
                        ax_line.legend(fontsize=11)
                        ax_line.grid(True, alpha=0.3)
                        col_viz1, col_viz2 = st.columns(2)
                        with col_viz1:
                            st.pyplot(fig_line)
                            plt.close(fig_line)
                        # Sintering temperature plot
                        with col_viz2:
                            fig_temp, ax_temp = plt.subplots(figsize=(10, 6))
                            ax_temp.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['sintering_temps']['exponential'],
                                color='red',
                                linewidth=3,
                                label='Exponential Model'
                            )
                            ax_temp.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['sintering_temps']['arrhenius_defect'],
                                color='blue',
                                linewidth=3,
                                linestyle='--',
                                label='Arrhenius Model (Defect)'
                            )
                            ax_temp.axvline(habit_angle, color='green', linestyle='--', linewidth=2
ax_temp.axvline(habit_angle, color='green', linestyle='--', linewidth=2,
                label=f'Habit Plane ({habit_angle}°)')
ax_temp.set_xlabel('Orientation (°)', fontsize=12)
ax_temp.set_ylabel('Sintering Temperature (K)', fontsize=12)
ax_temp.set_title('Sintering Temperature Prediction in Habit Plane Vicinity', fontsize=14, fontweight='bold')
ax_temp.legend(fontsize=11)
ax_temp.grid(True, alpha=0.3)

# Add Celsius scale on secondary axis
ax_temp2 = ax_temp.twinx()
celsius_ticks = ax_temp.get_yticks()
ax_temp2.set_ylim(ax_temp.get_ylim())
ax_temp2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
ax_temp2.set_ylabel('Temperature (°C)', fontsize=12)

st.pyplot(fig_temp)
plt.close(fig_temp)

# Export options
st.markdown("#### 📤 Export Results")
col_exp1, col_exp2, col_exp3 = st.columns(3)
with col_exp1:
    if st.button("💾 Export JSON", use_container_width=True):
        # Prepare report
        report = st.session_state.results_manager.prepare_vicinity_analysis_report(
            vicinity_sweep,
            {},
            target_params,
            analysis_params
        )
        json_str = json.dumps(report, indent=2, default=st.session_state.results_manager._json_serializer)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"vicinity_analysis_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
with col_exp2:
    if st.button("📊 Export CSV", use_container_width=True):
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
            label="📥 Download CSV",
            data=csv,
            file_name=f"vicinity_data_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
with col_exp3:
    if st.button("📦 Export Complete Package", use_container_width=True):
        report = st.session_state.results_manager.prepare_vicinity_analysis_report(
            vicinity_sweep,
            {},
            target_params,
            analysis_params
        )
        zip_buffer = st.session_state.results_manager.create_comprehensive_export(report)
        st.download_button(
            label="📥 Download ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"vicinity_analysis_package_{defect_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
else:
    st.error("Failed to generate vicinity sweep. Please check your data and parameters.")
else:
    st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Analysis'")

# Show example visualization when no analysis is performed
if not st.session_state.get('generate_analysis', False) and not st.session_state.get('quick_analysis'):
    st.markdown("#### 📊 Example Visualization")
    # Create example data
    example_angles = np.linspace(44.7, 64.7, 50)
    example_stress = 20 * np.exp(-(example_angles - 54.7)**2 / (2*5**2)) + 5
    example_temp = 623 * np.exp(-example_stress / 30) + 50 * np.sin(np.radians(example_angles))
    
    fig_example, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stress plot
    ax1.plot(example_angles, example_stress, 'b-', linewidth=3)
    ax1.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane (54.7°)')
    ax1.fill_between(example_angles, example_stress, alpha=0.2, color='blue')
    ax1.set_xlabel('Orientation (°)', fontsize=12)
    ax1.set_ylabel('Hydrostatic Stress (GPa)', fontsize=12)
    ax1.set_title('Example: Stress Concentration at Habit Plane', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature plot
    ax2.plot(example_angles, example_temp, 'r-', linewidth=3)
    ax2.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane (54.7°)')
    ax2.fill_between(example_angles, example_temp, alpha=0.2, color='red')
    ax2.set_xlabel('Orientation (°)', fontsize=12)
    ax2.set_ylabel('Sintering Temperature (K)', fontsize=12)
    ax2.set_title('Example: Temperature Reduction at Habit Plane', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add Celsius on secondary axis
    ax2_2 = ax2.twinx()
    celsius_ticks = ax2.get_yticks()
    ax2_2.set_ylim(ax2.get_ylim())
    ax2_2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
    ax2_2.set_ylabel('Temperature (°C)', fontsize=12)
    
    st.pyplot(fig_example)
    plt.close(fig_example)

with tab3:
    st.markdown('<h2 class="physics-header">🔬 Defect Type Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'defect_compare':
        with st.spinner("Comparing defect types..."):
            # Compare all defect types
            defect_comparison = st.session_state.interpolator.compare_defect_types(
                st.session_state.solutions,
                angle_range=(0, 360),
                n_points=100,
                region_type=region_type,
                shapes=[shape]
            )
            
            if defect_comparison:
                st.success(f"✅ Generated comparison of {len(defect_comparison)} defect types")
                # Store in session state
                st.session_state.defect_comparison = defect_comparison
                
                # Display defect comparison
                st.markdown("#### 📊 Defect Comparison Results")
                
                # Create tabs for different views
                comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Stress Comparison", "Sintering Comparison", "Radar View"])
                
                with comp_tab1:
                    # Stress comparison plot
                    fig_comp = st.session_state.visualizer.create_defect_comparison_plot(
                        defect_comparison,
                        stress_component='sigma_hydro',
                        title="Hydrostatic Stress Comparison Across Defect Types"
                    )
                    if fig_comp:
                        st.plotly_chart(fig_comp, use_container_width=True)
                
                with comp_tab2:
                    # Sintering temperature comparison
                    fig_sinter_comp, ax_sinter = plt.subplots(figsize=(12, 6))
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    for idx, (key, data) in enumerate(defect_comparison.items()):
                        if idx < len(colors):
                            ax_sinter.plot(
                                data['angles'],
                                data['sintering_temps'],
                                color=colors[idx],
                                linewidth=3,
                                label=f"{data['defect_type']} (ε*={data['eigen_strain']})"
                            )
                    
                    ax_sinter.axvline(54.7, color='green', linestyle='--', linewidth=2,
                                    label='Habit Plane (54.7°)')
                    ax_sinter.set_xlabel('Orientation (°)', fontsize=12)
                    ax_sinter.set_ylabel('Sintering Temperature (K)', fontsize=12)
                    ax_sinter.set_title('Sintering Temperature Comparison by Defect Type', fontsize=14, fontweight='bold')
                    ax_sinter.legend(fontsize=10)
                    ax_sinter.grid(True, alpha=0.3)
                    
                    # Add Celsius on secondary axis
                    ax_sinter2 = ax_sinter.twinx()
                    celsius_ticks = ax_sinter.get_yticks()
                    ax_sinter2.set_ylim(ax_sinter.get_ylim())
                    ax_sinter2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
                    ax_sinter2.set_ylabel('Temperature (°C)', fontsize=12)
                    
                    st.pyplot(fig_sinter_comp)
                    plt.close(fig_sinter_comp)
                
                with comp_tab3:
                    # Radar comparison for habit plane vicinity
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
                                'sintering_temps': np.array(data['sintering_temps'])[mask].tolist(),
                                'defect_type': data['defect_type'],
                                'color': data['color'],
                                'eigen_strain': data['eigen_strain']
                            }
                    
                    # Create radar comparison with customization
                    if vicinity_comparison:
                        st.markdown("##### ⚙️ Radar View Customization")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            show_labels_radar = st.checkbox("Show Labels", value=True)
                            radial_tick_color = st.color_picker("Radial Tick Color", "#000000")
                            radial_tick_width = st.slider("Radial Tick Width", 1, 5, 2)
                        with col_r2:
                            angular_tick_color = st.color_picker("Angular Tick Color", "#000000")
                            angular_tick_width = st.slider("Angular Tick Width", 1, 5, 2)
                            angular_tick_step = st.select_slider("Angular Tick Step (°)", options=[15, 30, 45, 60, 90], value=45)
                        with col_r3:
                            colormap_choice = st.selectbox("Colormap", ["default", "viridis", "plasma", "RdBu", "hot"])
                            font_size_title = st.slider("Title Font Size", 12, 32, 20)
                            font_size_axis = st.slider("Axis Label Font Size", 10, 24, 14)
                            font_size_tick = st.slider("Tick Font Size", 8, 20, 12)
                        
                        custom_label_input = st.text_input(
                            "Custom Stress Component Label (e.g., 'Hydrostatic Stress')",
                            value="Hydrostatic Stress"
                        )
                        custom_component_labels = {'sigma_hydro': custom_label_input} if custom_label_input.strip() else None
                        
                        fig_radar = st.session_state.visualizer.create_stress_comparison_radar(
                            vicinity_comparison,
                            title="Stress Components in Habit Plane Vicinity",
                            show_labels=show_labels_radar,
                            radial_tick_color=radial_tick_color,
                            radial_tick_width=radial_tick_width,
                            angular_tick_color=angular_tick_color,
                            angular_tick_width=angular_tick_width,
                            angular_tick_step=angular_tick_step,
                            colormap=colormap_choice,
                            custom_component_labels=custom_component_labels,
                            font_size_title=font_size_title,
                            font_size_axis=font_size_axis,
                            font_size_tick=font_size_tick
                        )
                        if fig_radar:
                            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Summary statistics
            st.markdown("#### 📈 Summary Statistics")
            # Calculate statistics for each defect
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
                            'Max T_sinter (K)': f"{max(data['sintering_temps']):.1f}"
                        })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
            
            # Export comparison data
            st.markdown("#### 📤 Export Comparison Data")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                # JSON export
                comparison_json = json.dumps(
                    st.session_state.defect_comparison,
                    indent=2,
                    default=st.session_state.results_manager._json_serializer
                )
                st.download_button(
                    label="💾 Export JSON",
                    data=comparison_json,
                    file_name=f"defect_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_exp2:
                # CSV export
                rows = []
                for key, data in defect_comparison.items():
                    angles = data['angles']
                    for i in range(len(angles)):
                        row = {
                            'defect_type': data['defect_type'],
                            'eigen_strain': data['eigen_strain'],
                            'angle_deg': angles[i]
                        }
                        for comp, stresses in data['stresses'].items():
                            if i < len(stresses):
                                row[f'{comp}_gpa'] = stresses[i]
                        if i < len(data['sintering_temps']):
                            row['T_sinter_k'] = data['sintering_temps'][i]
                        rows.append(row)
                
                if rows:
                    df = pd.DataFrame(rows)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Export CSV",
                        data=csv,
                        file_name=f"defect_comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    else:
        st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Analysis'")
        
        # Show defect comparison info
        st.markdown("""
        #### 🔬 Defect Comparison Analysis
        This analysis compares different defect types (ISF, ESF, Twin, No Defect) across all orientations.
        
        **Key comparisons:**
        1. **Stress distribution** - How each defect concentrates stress
        2. **Sintering temperature** - Temperature reduction capability
        3. **Habit plane effects** - Special behavior at 54.7°
        
        **Expected insights:**
        - Twin boundaries show maximum stress concentration
        - ISF/ESF have intermediate effects
        - Perfect crystals have minimal stress
        - Habit plane shows peak effects for twins
        """)

with tab4:
    st.markdown('<h2 class="physics-header">📊 Comprehensive Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('generate_analysis', False) or st.session_state.get('quick_analysis') == 'dashboard':
        with st.spinner("Generating comprehensive dashboard..."):
            # Check if we have both vicinity sweep and defect comparison
            if ('vicinity_sweep' not in st.session_state or 
                'defect_comparison' not in st.session_state):
                st.warning("Please run both Vicinity Analysis and Defect Comparison first.")
                col_run1, col_run2 = st.columns(2)
                with col_run1:
                    if st.button("🏃 Run Vicinity Analysis", use_container_width=True):
                        st.session_state.quick_analysis = "habit_plane"
                        st.rerun()
                with col_run2:
                    if st.button("🏃 Run Defect Comparison", use_container_width=True):
                        st.session_state.quick_analysis = "defect_compare"
                        st.rerun()
            else:
                # Generate comprehensive dashboard
                vicinity_sweep = st.session_state.vicinity_sweep
                defect_comparison = st.session_state.defect_comparison
                
                # Create comprehensive visualization
                fig_dashboard = st.session_state.visualizer.create_comprehensive_dashboard(
                    vicinity_sweep,
                    defect_comparison,
                    title=f"Comprehensive Analysis - {st.session_state.current_target_params.get('defect_type', 'Unknown')}"
                )
                if fig_dashboard:
                    st.plotly_chart(fig_dashboard, use_container_width=True)
                
                # Additional analysis
                st.markdown("#### 📈 Advanced Analysis")
                
                # Create tabs for different analyses
                adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Physics Analysis", "Sintering Optimization", "Export Package"])
                
                with adv_tab1:
                    # Physics-based analysis
                    st.markdown("##### 🔬 Physics-Based Analysis")
                    
                    # Calculate stress intensity factors
                    st.write("**Stress Intensity Factors (K):**")
                    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                    defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
                    
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
                            
                            st.metric(
                                f"K for {defect}",
                                f"{K:.2f} MPa√m",
                                "Stress Intensity"
                            )
                    
                    # Crystal orientation analysis
                    st.markdown("##### 🧊 Crystal Orientation Effects")
                    orientation_effects = []
                    for angle in [0, 30, 45, 54.7, 60, 90]:
                        effect = st.session_state.physics_analyzer.analyze_crystal_orientation_effects(
                            {},  # Empty stress data for basic analysis
                            angle
                        )
                        orientation_effects.append(effect)
                    
                    if orientation_effects:
                        df_orientation = pd.DataFrame(orientation_effects)
                        st.dataframe(df_orientation, use_container_width=True)
                
                with adv_tab2:
                    # Sintering optimization
                    st.markdown("##### 🔥 Sintering Optimization Analysis")
                    
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
                                'Is Near Habit Plane': abs(opt_angle - 54.7) < 5.0
                            })
                    
                    if optimal_data:
                        df_optimal = pd.DataFrame(optimal_data)
                        st.dataframe(df_optimal, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("##### 💡 Optimization Recommendation")
                    if optimal_data:
                        best_defect = min(optimal_data, key=lambda x: float(x['Min T_sinter (K)'].split()[0]))
                        
                        st.info(f"""
                        **Recommended Configuration:**
                        - **Defect Type:** {best_defect['Defect Type']}
                        - **Optimal Orientation:** {best_defect['Optimal Angle (°)']}°
                        - **Minimum Sintering Temperature:** {best_defect['Min T_sinter (K)']} K ({best_defect['Min T_sinter (°C)']}°C)
                        - **Temperature Reduction:** {best_defect['Temperature Reduction (K)']} K from reference
                        
                        **Note:** {best_defect['Defect Type']} provides the lowest sintering temperature
                        among all analyzed defect types at an optimal orientation.
                        """)
                
                with adv_tab3:
                    # Comprehensive export
                    st.markdown("##### 📦 Comprehensive Export Package")
                    st.write("""
                    The comprehensive export package includes:
                    1. Complete JSON report with all analysis data
                    2. CSV files for all datasets
                    3. README with analysis documentation
                    4. Python script for data processing
                    5. Configuration file
                    """)
                    
                    # Prepare comprehensive report
                    if st.button("🛠️ Prepare Comprehensive Report", use_container_width=True):
                        with st.spinner("Preparing comprehensive report..."):
                            report = st.session_state.results_manager.prepare_vicinity_analysis_report(
                                vicinity_sweep,
                                defect_comparison,
                                st.session_state.current_target_params,
                                st.session_state.current_analysis_params
                            )
                            
                            zip_buffer = st.session_state.results_manager.create_comprehensive_export(report)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            st.download_button(
                                label="📥 Download Complete Package",
                                data=zip_buffer.getvalue(),
                                file_name=f"comprehensive_analysis_{timestamp}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
    else:
        st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Analysis'")
        
        # Dashboard features
        st.markdown("""
        #### 📊 Comprehensive Dashboard Features
        The comprehensive dashboard provides an integrated view of all analysis results:
        
        **1. Multi-Panel Visualization**
        - Sunburst charts for polar stress visualization
        - Line plots for detailed orientation dependence
        - Radar charts for component comparison
        - Defect comparison across all types
        
        **2. Advanced Analysis**
        - Physics-based stress intensity calculations
        - Crystal orientation effects
        - Sintering temperature optimization
        - System classification mapping
        
        **3. Comprehensive Export**
        - Complete data package with all results
        - Processing scripts for further analysis
        - Documentation and configuration files
        
        **To generate the dashboard:**
        1. Run both Vicinity Analysis and Defect Comparison
        2. Click "Generate Dashboard" in the sidebar
        3. Explore the comprehensive results
        """)
    
    # Quick status check
    st.markdown("#### 📊 Analysis Status")
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if 'vicinity_sweep' in st.session_state:
            st.success("✅ Vicinity analysis data available")
        else:
            st.warning("⚠️ Vicinity analysis not yet run")
    with col_status2:
        if 'defect_comparison' in st.session_state:
            st.success("✅ Defect comparison data available")
        else:
            st.warning("⚠️ Defect comparison not yet run")

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
