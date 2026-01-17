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
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
import plotly.express as px
import matplotlib.figure

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================
# ENHANCED PHYSICS-BASED STRESS ANALYZER WITH REAL GEOMETRY VISUALIZATION
# =============================================

class PhysicsBasedStressAnalyzer:
    """Enhanced physics-based stress analyzer with real geometry support."""
    
    def __init__(self):
        self.elastic_constants = {
            'C11': 124.0,  # GPa for Ag
            'C12': 93.4,
            'C44': 46.1
        }
        self.defect_properties = {
            'ISF': {'gamma': 16.0, 'epsilon_star': 0.71, 'color': '#FF6B6B'},
            'ESF': {'gamma': 32.0, 'epsilon_star': 1.41, 'color': '#4ECDC4'},
            'Twin': {'gamma': 8.0, 'epsilon_star': 2.12, 'color': '#45B7D1'},
            'No Defect': {'gamma': 0.0, 'epsilon_star': 0.0, 'color': '#96CEB4'}
        }
    
    def compute_strain_energy_density(self, stress_fields):
        """Compute strain energy density from stress tensor components."""
        # Check for required stress components
        required_keys = {'sigma_xx', 'sigma_yy', 'sigma_xy'}
        if not required_keys.issubset(stress_fields.keys()):
            missing_keys = required_keys - set(stress_fields.keys())
            st.warning(f"Missing required stress components for strain energy calculation: {missing_keys}")
            return None
            
        sigma_xx = stress_fields['sigma_xx']
        sigma_yy = stress_fields['sigma_yy']
        sigma_xy = stress_fields['sigma_xy']
        
        # For plane stress: U = 0.5 * (σ:ε) = 0.5 * (σ_xx*ε_xx + σ_yy*ε_yy + 2*σ_xy*ε_xy)
        # Using isotropic approximation for simplicity
        E = 83.0  # Young's modulus for Ag in GPa
        nu = 0.37  # Poisson's ratio for Ag
        
        epsilon_xx = (sigma_xx - nu * sigma_yy) / E
        epsilon_yy = (sigma_yy - nu * sigma_xx) / E
        epsilon_xy = (1 + nu) * sigma_xy / E
        
        strain_energy = 0.5 * (
            sigma_xx * epsilon_xx + 
            sigma_yy * epsilon_yy + 
            2 * sigma_xy * epsilon_xy
        )
        
        return strain_energy
    
    def analyze_crystal_orientation_effects(self, stats, orientation_deg):
        """Analyze crystal orientation effects on stress distribution."""
        results = {}
        
        # Habit plane analysis (54.7° for FCC)
        habit_angle = 54.7
        angle_diff = abs(orientation_deg - habit_angle)
        
        results['habit_plane_proximity'] = {
            'angle_difference': angle_diff,
            'is_near_habit': angle_diff < 10.0,
            'orientation_factor': np.cos(np.radians(angle_diff)),
            'max_schmid_factor': 0.5 * np.abs(np.sin(2 * np.radians(orientation_deg)))
        }
        
        # Compute anisotropy effects
        C11, C12, C44 = self.elastic_constants.values()
        theta_rad = np.radians(orientation_deg)
        
        # Orientation-dependent elastic modulus (Zener anisotropy)
        S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
        S44 = 1 / C44
        
        E_orientation = 1 / (
            S11 - 
            2 * (S11 - S44 - 0.5 * S11) * 
            (np.sin(theta_rad)**2 * np.cos(theta_rad)**2)
        )
        
        results['anisotropy_analysis'] = {
            'E_orientation_GPa': float(E_orientation),
            'anisotropy_ratio': float(C44 / ((C11 - C12) / 2)),
            'zener_ratio': float(2 * C44 / (C11 - C12)),
            'stiffness_direction': 'Soft' if E_orientation < 80 else 'Hard'
        }
        
        # Defect stress enhancement factor
        if 'defect' in stats and 'bulk' in stats:
            if 'sigma_hydro' in stats['defect'] and 'sigma_hydro' in stats['bulk']:
                defect_mean = stats['defect']['sigma_hydro']['mean']
                bulk_mean = stats['bulk']['sigma_hydro']['mean']
                
                if bulk_mean != 0:
                    enhancement = defect_mean / bulk_mean
                    results['stress_enhancement'] = {
                        'factor': float(enhancement),
                        'orientation_sensitivity': float(enhancement * np.cos(np.radians(angle_diff))),
                        'classification': 'High' if enhancement > 2.0 else 'Moderate' if enhancement > 1.5 else 'Low'
                    }
        
        return results
    
    def compute_real_geometry_stress(self, eta, stress_fields, orientation_deg=54.7):
        """
        Compute stress visualization for real geometry domain.
        
        Args:
            eta: Phase field order parameter (2D array)
            stress_fields: Dictionary of stress components
            orientation_deg: Crystal orientation angle
            
        Returns:
            Dictionary containing comprehensive stress analysis
        """
        results = {}
        
        if eta is None or not isinstance(eta, np.ndarray) or eta.ndim != 2:
            st.warning("Invalid phase field data for real geometry analysis")
            return results
        
        # Ensure stress fields are numpy arrays
        processed_stresses = {}
        for key in stress_fields:
            if torch.is_tensor(stress_fields[key]):
                processed_stresses[key] = stress_fields[key].cpu().numpy()
            else:
                processed_stresses[key] = stress_fields[key]
                
            # Ensure correct shape
            if processed_stresses[key].shape != eta.shape:
                # Try to reshape if possible
                try:
                    processed_stresses[key] = processed_stresses[key].reshape(eta.shape)
                except:
                    st.warning(f"Cannot reshape {key} from {processed_stresses[key].shape} to {eta.shape}")
                    continue
        
        # Region masks based on phase field
        defect_mask = eta > 0.6
        interface_mask = (eta >= 0.4) & (eta <= 0.6)
        bulk_mask = eta < 0.4
        
        results['region_masks'] = {
            'defect': defect_mask,
            'interface': interface_mask,
            'bulk': bulk_mask
        }
        
        # Compute region statistics for each stress component
        region_stats = {}
        for region_name, mask in [('defect', defect_mask),
                                  ('interface', interface_mask),
                                  ('bulk', bulk_mask)]:
            region_stats[region_name] = {}
            
            for stress_name, stress_data in processed_stresses.items():
                if stress_data.shape != mask.shape:
                    continue
                    
                if mask.any():
                    region_stress = stress_data[mask]
                    
                    # Compute statistics
                    region_stats[region_name][stress_name] = {
                        'mean': float(np.nanmean(region_stress)),
                        'std': float(np.nanstd(region_stress)),
                        'max': float(np.nanmax(region_stress)),
                        'min': float(np.nanmin(region_stress)),
                        'abs_max': float(np.nanmax(np.abs(region_stress))),
                        'percentile_95': float(np.nanpercentile(np.abs(region_stress), 95)),
                        'area_fraction': float(mask.sum() / mask.size),
                        'skewness': float(pd.Series(region_stress.flatten()).skew()) if len(region_stress.flatten()) > 0 else 0.0,
                        'kurtosis': float(pd.Series(region_stress.flatten()).kurtosis()) if len(region_stress.flatten()) > 0 else 0.0
                    }
                    
                    # Additional metrics for hydrostatic stress
                    if stress_name == 'sigma_hydro':
                        compressive_fraction = np.sum(region_stress < 0) / len(region_stress) if len(region_stress) > 0 else 0
                        tensile_fraction = np.sum(region_stress > 0) / len(region_stress) if len(region_stress) > 0 else 0
                        region_stats[region_name][stress_name].update({
                            'compressive_fraction': float(compressive_fraction),
                            'tensile_fraction': float(tensile_fraction),
                            'mean_compressive': float(np.nanmean(region_stress[region_stress < 0])) if np.any(region_stress < 0) else 0.0,
                            'mean_tensile': float(np.nanmean(region_stress[region_stress > 0])) if np.any(region_stress > 0) else 0.0
                        })
        
        results['region_stats'] = region_stats
        
        # Compute orientation-dependent metrics
        orientation_results = self.analyze_crystal_orientation_effects(
            region_stats, orientation_deg
        )
        results['orientation_analysis'] = orientation_results
        
        # Compute stress concentration factors
        if defect_mask.any() and bulk_mask.any():
            for stress_name in processed_stresses.keys():
                if (stress_name in region_stats['defect'] and 
                    stress_name in region_stats['bulk']):
                    defect_mean = region_stats['defect'][stress_name]['mean']
                    bulk_mean = region_stats['bulk'][stress_name]['mean']
                    
                    if bulk_mean != 0:
                        concentration_factor = defect_mean / bulk_mean
                        results[f'{stress_name}_concentration_factor'] = {
                            'factor': float(concentration_factor),
                            'defect_mean': float(defect_mean),
                            'bulk_mean': float(bulk_mean),
                            'enhancement': float(concentration_factor - 1)
                        }
        
        # Compute strain energy density
        strain_energy = self.compute_strain_energy_density(processed_stresses)
        if strain_energy is not None:
            results['strain_energy'] = {
                'total': float(np.nansum(strain_energy)),
                'mean': float(np.nanmean(strain_energy)),
                'max': float(np.nanmax(strain_energy)),
                'min': float(np.nanmin(strain_energy)),
                'defect_mean': float(np.nanmean(strain_energy[defect_mask])) if defect_mask.any() else 0.0,
                'interface_mean': float(np.nanmean(strain_energy[interface_mask])) if interface_mask.any() else 0.0,
                'bulk_mean': float(np.nanmean(strain_energy[bulk_mask])) if bulk_mask.any() else 0.0,
                'energy_density_GPa': float(np.nanmean(strain_energy))
            }
        else:
            st.warning("Strain energy calculation skipped: missing required stress components (sigma_xx, sigma_yy, sigma_xy).")
            results['strain_energy'] = {}  # Set to empty dictionary
        
        # Compute additional stress invariants if not already present
        if 'sigma_hydro' in processed_stresses and 'sigma_xx' in processed_stresses:
            sigma_hydro = processed_stresses['sigma_hydro']
            sigma_xx = processed_stresses['sigma_xx']
            sigma_yy = processed_stresses.get('sigma_yy', np.zeros_like(sigma_xx))
            sigma_xy = processed_stresses.get('sigma_xy', np.zeros_like(sigma_xx))
            
            # Von Mises stress (2D plane stress)
            von_mises = np.sqrt(
                sigma_xx**2 + sigma_yy**2 - sigma_xx*sigma_yy + 3*sigma_xy**2
            )
            results['computed_von_mises'] = von_mises
            
            # Stress magnitude
            sigma_mag = np.sqrt(sigma_xx**2 + sigma_yy**2 + 2*sigma_xy**2)
            results['computed_sigma_mag'] = sigma_mag
            
            # Max shear stress
            max_shear = 0.5 * np.sqrt((sigma_xx - sigma_yy)**2 + 4*sigma_xy**2)
            results['max_shear_stress'] = max_shear
        
        # Store raw data for visualization
        results['geometry_data'] = {
            'eta': eta,
            'coordinates': np.indices(eta.shape).transpose(1, 2, 0),
            'region_masks': {
                'defect': defect_mask,
                'interface': interface_mask,
                'bulk': bulk_mask
            }
        }
        
        results['stress_data'] = processed_stresses
        
        # Add metadata
        results['metadata'] = {
            'analysis_time': datetime.now().isoformat(),
            'grid_shape': eta.shape,
            'total_pixels': int(eta.size),
            'defect_pixels': int(defect_mask.sum()),
            'interface_pixels': int(interface_mask.sum()),
            'bulk_pixels': int(bulk_mask.sum())
        }
        
        return results

# =============================================
# ENHANCED SINTERING CALCULATOR
# =============================================

class EnhancedSinteringCalculator:
    """Enhanced calculator for sintering temperature prediction."""
    
    def __init__(self):
        # Material constants for Ag
        self.Q_a_eV = 0.95  # Activation energy for Ag diffusion (eV)
        self.Omega = 1.2e-29  # Activation volume (m³)
        self.kB = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.D0 = 1e-5  # Pre-exponential factor (m²/s)
        self.D_crit = 1e-18  # Critical diffusivity for sintering (m²/s)
        
        # System classification parameters
        self.system_thresholds = {
            'System 1 (Perfect)': {'min_stress': 0, 'max_stress': 5, 'base_temp': 630},
            'System 2 (SF/Twin)': {'min_stress': 5, 'max_stress': 20, 'base_temp': 550},
            'System 3 (Plastic)': {'min_stress': 20, 'max_stress': 100, 'base_temp': 400}
        }
        
        # Defect-specific adjustments
        self.defect_adjustments = {
            'ISF': {'Q_reduction': 0.05, 'temp_reduction': 30},
            'ESF': {'Q_reduction': 0.08, 'temp_reduction': 45},
            'Twin': {'Q_reduction': 0.12, 'temp_reduction': 60},
            'No Defect': {'Q_reduction': 0.0, 'temp_reduction': 0}
        }
    
    def compute_arrhenius_temperature(self, stress_GPa, defect_type='Twin'):
        """Compute sintering temperature using Arrhenius equation."""
        # Convert stress from GPa to Pa
        stress_Pa = stress_GPa * 1e9
        
        # Compute effective activation energy (eV)
        Q_eff = self.Q_a_eV - (self.Omega * stress_Pa) / 1.602e-19
        
        # Apply defect-specific reduction
        defect_adj = self.defect_adjustments.get(defect_type, self.defect_adjustments['No Defect'])
        Q_eff *= (1 - defect_adj['Q_reduction'])
        
        # Compute temperature from Arrhenius: D = D0 * exp(-Q_eff/(kB*T))
        # Solve for T: T = Q_eff / (kB * ln(D0/D_crit))
        if Q_eff > 0:
            T_k = Q_eff / (self.kB * np.log(self.D0 / self.D_crit))
        else:
            T_k = 300  # Minimum temperature
        
        # Apply defect-specific temperature reduction
        T_k -= defect_adj['temp_reduction']
        
        # Ensure reasonable bounds
        T_k = max(300, min(1000, T_k))
        T_c = T_k - 273.15
        
        return {
            'T_k': float(T_k),
            'T_c': float(T_c),
            'Q_eff_eV': float(Q_eff),
            'Q_reduction_eV': float(self.Q_a_eV - Q_eff),
            'Q_reduction_percent': float((self.Q_a_eV - Q_eff) / self.Q_a_eV * 100)
        }
    
    def compute_exponential_temperature(self, stress_GPa, defect_type='Twin'):
        """Compute sintering temperature using exponential model."""
        # Base temperatures for different systems
        base_temps = {
            'System 1 (Perfect)': 630,
            'System 2 (SF/Twin)': 550,
            'System 3 (Plastic)': 400
        }
        
        # Determine system based on stress
        system = self.classify_system(stress_GPa)
        base_T = base_temps.get(system, 550)
        
        # Exponential decay with stress
        # T = T_base * exp(-k * stress)
        k = 0.04  # Decay constant
        
        # Apply defect-specific adjustment
        defect_adj = self.defect_adjustments.get(defect_type, self.defect_adjustments['No Defect'])
        T_k = base_T * np.exp(-k * stress_GPa) - defect_adj['temp_reduction']
        
        # Ensure reasonable bounds
        T_k = max(300, min(1000, T_k))
        T_c = T_k - 273.15
        
        return {
            'T_k': float(T_k),
            'T_c': float(T_c),
            'base_system': system,
            'base_temperature': float(base_T),
            'stress_factor': float(np.exp(-k * stress_GPa))
        }
    
    def classify_system(self, stress_GPa):
        """Classify the sintering system based on stress level."""
        for system_name, thresholds in self.system_thresholds.items():
            if thresholds['min_stress'] <= stress_GPa < thresholds['max_stress']:
                return system_name
        return 'System 2 (SF/Twin)'  # Default
    
    def get_theoretical_curve(self, defect_type='Twin'):
        """Generate theoretical sintering temperature curves."""
        stresses = np.linspace(0, 30, 50)  # GPa
        arrhenius_temps = []
        exponential_temps = []
        
        for stress in stresses:
            arrhenius = self.compute_arrhenius_temperature(stress, defect_type)
            exponential = self.compute_exponential_temperature(stress, defect_type)
            arrhenius_temps.append(arrhenius['T_k'])
            exponential_temps.append(exponential['T_k'])
        
        return {
            'stresses': stresses.tolist(),
            'arrhenius_temps': arrhenius_temps,
            'exponential_temps': exponential_temps,
            'system_1_boundary': 5.0,
            'system_2_boundary': 20.0
        }
    
    def create_comprehensive_sintering_plot(self, stresses, temps, defect_type, title, settings=None):
        """Create comprehensive sintering analysis plot."""
        if settings is None:
            settings = {}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot experimental/theoretical data
        ax.plot(stresses, temps, 'o-', linewidth=2, markersize=8,
                label=f"{defect_type} Model", color='#3B82F6')
        
        # Add system classification regions
        ax.axvspan(0, 5, alpha=0.1, color='green', label='System 1 (Perfect)')
        ax.axvspan(5, 20, alpha=0.1, color='orange', label='System 2 (SF/Twin)')
        ax.axvspan(20, 30, alpha=0.1, color='red', label='System 3 (Plastic)')
        
        # Add theoretical curves
        theoretical = self.get_theoretical_curve(defect_type)
        ax.plot(theoretical['stresses'], theoretical['arrhenius_temps'], '--',
                label='Arrhenius Model', color='#10B981', linewidth=1.5)
        ax.plot(theoretical['stresses'], theoretical['exponential_temps'], ':',
                label='Exponential Model', color='#8B5CF6', linewidth=1.5)
        
        # Add vertical lines for system boundaries
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel("Hydrostatic Stress (GPa)", fontsize=settings.get('label_size', 12))
        ax.set_ylabel("Sintering Temperature (K)", fontsize=settings.get('label_size', 12))
        ax.set_title(title, fontsize=settings.get('title_size', 14), fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=settings.get('grid_alpha', 0.3), linestyle='--')
        ax.legend(loc='upper right', fontsize=settings.get('tick_size', 10))
        
        # Set limits
        ax.set_xlim(0, 30)
        ax.set_ylim(300, 700)
        
        # Add text annotations
        ax.text(2.5, 650, 'System 1\n(Perfect)', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax.text(12.5, 550, 'System 2\n(SF/Twin)', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax.text(25, 450, 'System 3\n(Plastic)', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        return fig

# =============================================
# PHYSICS-AWARE INTERPOLATOR
# =============================================

class PhysicsAwareInterpolator:
    """Physics-aware interpolator using attention and spatial weighting."""
    
    def __init__(self):
        self.defect_map = {
            "ISF": [1, 0, 0, 0],
            "ESF": [0, 1, 0, 0], 
            "Twin": [0, 0, 1, 0],
            "No Defect": [0, 0, 0, 1]
        }
        self.shape_map = {
            "Square": [1, 0, 0, 0],
            "Horizontal Fault": [0, 1, 0, 0],
            "Vertical Fault": [0, 0, 1, 0],
            "Rectangle": [0, 0, 0, 1]
        }
        
    def compute_parameter_vector(self, params, orientation_deg):
        """Convert parameters to a feature vector."""
        defect_vec = self.defect_map.get(params.get('defect_type', 'No Defect'), [0, 0, 0, 0])
        shape_vec = self.shape_map.get(params.get('shape', 'Square'), [0, 0, 0, 0])
        
        # Normalize orientation to sine/cosine for circular continuity
        orientation_rad = np.radians(orientation_deg)
        orientation_vec = [np.sin(orientation_rad), np.cos(orientation_rad)]
        
        # Additional physical parameters
        physical_params = [
            params.get('eps0', 0.0) / 3.0,  # Normalized
            params.get('kappa', 0.6) / 2.0,  # Normalized
            orientation_deg / 360.0  # Normalized
        ]
        
        # Combine all features
        feature_vector = np.array(
            defect_vec + shape_vec + orientation_vec + physical_params,
            dtype=np.float32
        )
        
        return feature_vector
    
    def compute_attention_weights(self, source_vectors, target_vector):
        """Compute attention weights using cosine similarity."""
        if len(source_vectors) == 0:
            return np.array([])
        
        # Compute cosine similarities
        similarities = []
        for src_vec in source_vectors:
            # Cosine similarity
            dot_product = np.dot(src_vec, target_vector)
            norm_product = np.linalg.norm(src_vec) * np.linalg.norm(target_vector)
            similarity = dot_product / (norm_product + 1e-8)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Softmax for attention weights
        exp_similarities = np.exp(similarities - np.max(similarities))  # Numerical stability
        weights = exp_similarities / (exp_similarities.sum() + 1e-8)
        
        return weights
    
    def compute_spatial_weights(self, source_vectors, target_vector, sigma=0.5):
        """Compute spatial weights using Gaussian RBF."""
        if len(source_vectors) == 0:
            return np.array([])
        
        # Compute Euclidean distances
        distances = np.linalg.norm(source_vectors - target_vector, axis=1)
        
        # Gaussian RBF weights
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def compute_physics_weights(self, source_params_list, target_params, orientation_deg):
        """Compute physics-based weights."""
        weights = []
        target_defect = target_params.get('defect_type', 'No Defect')
        
        for src_params in source_params_list:
            src_defect = src_params.get('defect_type', 'No Defect')
            
            # Base weight based on defect type match
            if src_defect == target_defect:
                base_weight = 1.0
            elif src_defect in ['ISF', 'ESF'] and target_defect in ['ISF', 'ESF']:
                base_weight = 0.7  # Partial match
            else:
                base_weight = 0.1
            
            # Orientation similarity bonus
            src_theta = src_params.get('theta', 0.0)
            if src_theta is not None:
                src_deg = np.degrees(src_theta)
                orientation_diff = min(
                    abs(src_deg - orientation_deg),
                    abs(src_deg - orientation_deg + 360),
                    abs(src_deg - orientation_deg - 360)
                )
                orientation_weight = np.exp(-orientation_diff / 45.0)  # 45° half-width
                base_weight *= (0.5 + 0.5 * orientation_weight)
            
            weights.append(base_weight)
        
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def interpolate_stress_components(self, solutions, orientation_deg, target_params, region_type='all'):
        """Interpolate stress components from source solutions."""
        if not solutions:
            return None
        
        # Prepare source data
        source_vectors = []
        source_stresses = []
        source_params_list = []
        
        for sol in solutions:
            if 'params' not in sol or 'history' not in sol or len(sol['history']) == 0:
                continue
            
            # Get parameters
            params = sol['params']
            source_params_list.append(params)
            
            # Compute feature vector
            if 'theta' in params:
                src_theta = params['theta']
                src_deg = np.degrees(src_theta) if src_theta is not None else 0.0
            else:
                src_deg = 0.0
            
            src_vector = self.compute_parameter_vector(params, src_deg)
            source_vectors.append(src_vector)
            
            # Get stress data from last frame
            last_frame = sol['history'][-1]
            stresses = last_frame.get('stresses', {})
            
            # Extract relevant stress components
            stress_data = {}
            for key in ['sigma_hydro', 'sigma_xx', 'sigma_yy', 'sigma_xy']:
                if key in stresses:
                    if torch.is_tensor(stresses[key]):
                        stress_data[key] = stresses[key].cpu().numpy()
                    else:
                        stress_data[key] = stresses[key]
            
            source_stresses.append(stress_data)
        
        if not source_vectors:
            return None
        
        source_vectors = np.array(source_vectors)
        
        # Compute target vector
        target_vector = self.compute_parameter_vector(target_params, orientation_deg)
        
        # Compute combined weights
        attention_weights = self.compute_attention_weights(source_vectors, target_vector)
        spatial_weights = self.compute_spatial_weights(source_vectors, target_vector)
        physics_weights = self.compute_physics_weights(source_params_list, target_params, orientation_deg)
        
        # Combine weights (adjustable blend)
        alpha = 0.4  # Attention weight
        beta = 0.3   # Spatial weight  
        gamma = 0.3  # Physics weight
        
        combined_weights = (
            alpha * attention_weights +
            beta * spatial_weights +
            gamma * physics_weights
        )
        combined_weights = combined_weights / (combined_weights.sum() + 1e-8)
        
        # Interpolate stress fields
        interpolated_stress = {}
        stress_keys = source_stresses[0].keys() if source_stresses else []
        
        for key in stress_keys:
            # Weighted average of stress fields
            weighted_sum = None
            for i, stress_data in enumerate(source_stresses):
                if key in stress_data:
                    if weighted_sum is None:
                        weighted_sum = combined_weights[i] * stress_data[key]
                    else:
                        weighted_sum += combined_weights[i] * stress_data[key]
            
            if weighted_sum is not None:
                interpolated_stress[key] = weighted_sum
        
        # Compute derived quantities
        if 'sigma_xx' in interpolated_stress and 'sigma_yy' in interpolated_stress:
            sigma_xx = interpolated_stress['sigma_xx']
            sigma_yy = interpolated_stress['sigma_yy']
            sigma_xy = interpolated_stress.get('sigma_xy', np.zeros_like(sigma_xx))
            
            # Hydrostatic stress
            sigma_hydro = (sigma_xx + sigma_yy) / 2.0
            
            # Von Mises stress (2D plane stress)
            von_mises = np.sqrt(
                sigma_xx**2 + sigma_yy**2 - sigma_xx*sigma_yy + 3*sigma_xy**2
            )
            
            # Stress magnitude
            sigma_mag = np.sqrt(sigma_xx**2 + sigma_yy**2 + 2*sigma_xy**2)
            
            interpolated_stress.update({
                'sigma_hydro': sigma_hydro,
                'von_mises': von_mises,
                'sigma_mag': sigma_mag
            })
        
        # Generate synthetic phase field for visualization
        if source_stresses and 'eta' in solutions[0]['history'][-1]:
            # Use weighted average of phase fields
            eta_sum = None
            for i, sol in enumerate(solutions):
                if 'history' in sol and sol['history']:
                    last_frame = sol['history'][-1]
                    if 'eta' in last_frame:
                        eta_data = last_frame['eta']
                        if torch.is_tensor(eta_data):
                            eta_data = eta_data.cpu().numpy()
                        
                        if eta_sum is None:
                            eta_sum = combined_weights[i] * eta_data
                        else:
                            eta_sum += combined_weights[i] * eta_data
            
            if eta_sum is not None:
                interpolated_stress['eta'] = eta_sum
        
        # Sintering analysis
        sintering_analysis = self._compute_sintering_analysis(
            interpolated_stress, target_params.get('defect_type', 'Twin')
        )
        
        return {
            'interpolated_stress': interpolated_stress,
            'sintering_analysis': sintering_analysis,
            'weights': {
                'attention': attention_weights.tolist(),
                'spatial': spatial_weights.tolist(),
                'physics': physics_weights.tolist(),
                'combined': combined_weights.tolist()
            },
            'metadata': {
                'num_sources': len(solutions),
                'target_params': target_params,
                'orientation_deg': orientation_deg
            }
        }
    
    def _compute_sintering_analysis(self, stress_data, defect_type):
        """Compute sintering analysis from stress data."""
        if 'sigma_hydro' not in stress_data:
            return None
        
        sigma_hydro = stress_data['sigma_hydro']
        
        # Get mean absolute stress in high-stress regions
        if 'eta' in stress_data:
            eta = stress_data['eta']
            defect_mask = eta > 0.6
            if defect_mask.any():
                avg_stress = np.mean(np.abs(sigma_hydro[defect_mask]))
            else:
                avg_stress = np.mean(np.abs(sigma_hydro))
        else:
            avg_stress = np.mean(np.abs(sigma_hydro))
        
        # Initialize sintering calculator
        sintering_calc = EnhancedSinteringCalculator()
        
        # Compute temperatures using both models
        arrhenius = sintering_calc.compute_arrhenius_temperature(avg_stress, defect_type)
        exponential = sintering_calc.compute_exponential_temperature(avg_stress, defect_type)
        
        # System classification
        system = sintering_calc.classify_system(avg_stress)
        
        return {
            'temperature_predictions': {
                'arrhenius_defect_k': arrhenius['T_k'],
                'arrhenius_defect_c': arrhenius['T_c'],
                'exponential_model_k': exponential['T_k'],
                'exponential_model_c': exponential['T_c']
            },
            'activation_energy_analysis': {
                'Q_a_standard_eV': sintering_calc.Q_a_eV,
                'Q_eff_defect_eV': arrhenius['Q_eff_eV'],
                'reduction_defect_eV': arrhenius['Q_reduction_eV'],
                'reduction_percentage': arrhenius['Q_reduction_percent']
            },
            'system_classification': {
                'system': system,
                'predicted_T_k': min(arrhenius['T_k'], exponential['T_k']),
                'stress_level_GPa': float(avg_stress)
            }
        }

# =============================================
# ENHANCED SOLUTION LOADER
# =============================================

class EnhancedSolutionLoader:
    """Enhanced loader for numerical simulation solutions."""
    
    def __init__(self, solutions_dir):
        self.solutions_dir = solutions_dir
        self.cache = {}
    
    def load_all_solutions(self):
        """Load all .pkl files from directory."""
        solutions = []
        
        if not os.path.exists(self.solutions_dir):
            st.error(f"Solutions directory not found: {self.solutions_dir}")
            return solutions
        
        pkl_files = [f for f in os.listdir(self.solutions_dir) if f.endswith('.pkl')]
        
        if not pkl_files:
            st.warning(f"No .pkl files found in {self.solutions_dir}")
            return solutions
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, filename in enumerate(pkl_files):
            status_text.text(f"Loading {filename} ({i+1}/{len(pkl_files)})...")
            
            try:
                filepath = os.path.join(self.solutions_dir, filename)
                
                # Check cache first
                if filename in self.cache:
                    solution = self.cache[filename]
                else:
                    with open(filepath, 'rb') as f:
                        solution = pickle.load(f)
                    self.cache[filename] = solution
                
                # Add metadata
                if 'metadata' not in solution:
                    solution['metadata'] = {}
                
                solution['metadata'].update({
                    'filename': filename,
                    'filepath': filepath,
                    'loaded_at': datetime.now().isoformat()
                })
                
                # Ensure required fields
                if 'params' not in solution:
                    solution['params'] = {}
                if 'history' not in solution:
                    solution['history'] = []
                
                # Validate stress fields
                if solution['history']:
                    last_frame = solution['history'][-1]
                    stresses = last_frame.get('stresses', {})
                    required_keys = {'sigma_xx', 'sigma_yy', 'sigma_xy'}
                    if not required_keys.issubset(stresses.keys()):
                        st.warning(f"Solution {filename} missing required stress components: {required_keys - set(stresses.keys())}")
                
                solutions.append(solution)
                
            except Exception as e:
                st.warning(f"Failed to load {filename}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(pkl_files))
        
        progress_bar.empty()
        status_text.empty()
        
        if solutions:
            st.success(f"Successfully loaded {len(solutions)} solutions")
            
            # Display summary
            defect_types = []
            orientations = []
            
            for sol in solutions:
                params = sol.get('params', {})
                defect_types.append(params.get('defect_type', 'Unknown'))
                
                if 'theta' in params:
                    theta = params['theta']
                    if theta is not None:
                        orientations.append(np.degrees(theta))
            
            if defect_types:
                unique_defects = set(defect_types)
                st.info(f"Defect types: {', '.join(unique_defects)}")
            
            if orientations:
                st.info(f"Orientation range: {min(orientations):.1f}° to {max(orientations):.1f}°")
        
        return solutions
    
    def clear_cache(self):
        """Clear the solution cache."""
        self.cache = {}

# =============================================
# UNIVERSAL VISUALIZATION ENHANCER CLASS
# =============================================

class UniversalVisualizationEnhancer:
    """
    Comprehensive visualization enhancer with 50+ colormaps, font controls,
    line thickness adjustments, and universal figure enhancements
    """
    def __init__(self):
        # Comprehensive colormap collection (50+ options)
        self.colormaps = {
            # Sequential colormaps
            'viridis': 'viridis', 'plasma': 'plasma', 'inferno': 'inferno',
            'magma': 'magma', 'cividis': 'cividis', 'summer': 'summer',
            'wistia': 'wistia', 'autumn': 'autumn', 'spring': 'spring',
            'cool': 'cool', 'Wistia': 'Wistia', 'hot': 'hot',
            'afmhot': 'afmhot', 'gist_heat': 'gist_heat', 'copper': 'copper',
            # Diverging colormaps
            'Spectral': 'Spectral', 'coolwarm': 'coolwarm', 'bwr': 'bwr',
            'seismic': 'seismic', 'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn',
            'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG',
            'PuOr': 'PuOr', 'RdGy': 'RdGy',
            # Qualitative colormaps
            'tab10': 'tab10', 'tab20': 'tab20', 'Set1': 'Set1',
            'Set2': 'Set2', 'Set3': 'Set3', 'tab20b': 'tab20b',
            'tab20c': 'tab20c', 'Pastel1': 'Pastel1', 'Pastel2': 'Pastel2',
            'Paired': 'Paired', 'Accent': 'Accent', 'Dark2': 'Dark2',
            # Cyclic colormaps
            'twilight': 'twilight', 'twilight_shifted': 'twilight_shifted',
            'hsv': 'hsv',
            # Legacy/perceptually problematic but requested
            'jet': 'jet', 'rainbow': 'rainbow', 'turbo': 'turbo',
            'nipy_spectral': 'nipy_spectral', 'gist_ncar': 'gist_ncar',
            'gist_rainbow': 'gist_rainbow',
            # Custom engineered colormaps
            'thermal_stress': self._create_thermal_stress_cmap(),
            'defect_gradient': self._create_defect_gradient_cmap(),
            'crystal_orientation': self._create_crystal_orientation_cmap(),
            'stress_tensile_compressive': self._create_stress_tensile_compressive_cmap(),
        }
        # Font options
        self.font_families = ['Arial', 'Times New Roman', 'Helvetica',
                              'Courier New', 'Verdana', 'Georgia', 'Cambria']
        # Default visualization parameters
        self.default_params = {
            'font_size': 12,
            'title_size': 14,
            'label_size': 11,
            'tick_size': 10,
            'legend_size': 10,
            'line_width': 2.0,
            'marker_size': 6,
            'grid_alpha': 0.3,
            'figure_dpi': 150,
            'colorbar_width': 0.02,
            'colorbar_pad': 0.05,
        }
        # Stress visualization specific parameters
        self.stress_params = {
            'hydrostatic_cmap': 'RdBu_r',
            'vonmises_cmap': 'viridis',
            'magnitude_cmap': 'plasma',
            'defect_cmap': 'Set1',
            'interface_alpha': 0.7,
            'bulk_alpha': 0.3,
            'defect_alpha': 0.9,
            'contour_levels': 20,
            'vector_scale': 50,
            'quiver_density': 10,
        }

    def _create_thermal_stress_cmap(self):
        """Create custom thermal stress colormap (blue to red)"""
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1),
                  (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
        positions = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        return LinearSegmentedColormap.from_list('thermal_stress', list(zip(positions, colors)))

    def _create_defect_gradient_cmap(self):
        """Create custom defect gradient colormap"""
        colors = [(0.1, 0.1, 0.1), (0.3, 0, 0.5), (0.6, 0, 0.8),
                  (0.8, 0.3, 0.1), (1, 0.6, 0), (1, 0.9, 0.3)]
        return LinearSegmentedColormap.from_list('defect_gradient', colors)

    def _create_crystal_orientation_cmap(self):
        """Create crystal orientation colormap (cyclic)"""
        colors = [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 1),
                  (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0),
                  (1, 0, 0), (0.5, 0, 0), (0, 0, 0.5)]
        return LinearSegmentedColormap.from_list('crystal_orientation', colors, N=256)

    def _create_stress_tensile_compressive_cmap(self):
        """Create tensile (red) to compressive (blue) colormap"""
        colors = [(0, 0, 1), (0.2, 0.2, 1), (0.4, 0.4, 1),
                  (0.8, 0.8, 1), (1, 1, 1), (1, 0.8, 0.8),
                  (1, 0.4, 0.4), (1, 0.2, 0.2), (1, 0, 0)]
        return LinearSegmentedColormap.from_list('stress_tensile_compressive', colors)

    def create_visualization_controls(self, container=None):
        """
        Create comprehensive visualization controls in Streamlit
        Args:
            container: Streamlit container to place controls in
        """
        if container is None:
            container = st.sidebar
        container.markdown("---")
        container.markdown("### 🎨 Visualization Controls")
        # Colormap selection with categories
        colormap_category = container.selectbox(
            "Colormap Category",
            ["Sequential", "Diverging", "Qualitative", "Cyclic", "Legacy", "Custom"],
            index=0,
            help="Select colormap category for visualization"
        )
        # Filter colormaps by category
        category_maps = {
            "Sequential": ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                           'summer', 'wistia', 'autumn', 'spring', 'cool',
                           'hot', 'afmhot', 'gist_heat', 'copper'],
            "Diverging": ['Spectral', 'coolwarm', 'bwr', 'seismic', 'RdYlBu',
                          'RdYlGn', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy'],
            "Qualitative": ['tab10', 'tab20', 'Set1', 'Set2', 'Set3',
                            'tab20b', 'tab20c', 'Pastel1', 'Pastel2',
                            'Paired', 'Accent', 'Dark2'],
            "Cyclic": ['twilight', 'twilight_shifted', 'hsv'],
            "Legacy": ['jet', 'rainbow', 'turbo', 'nipy_spectral',
                       'gist_ncar', 'gist_rainbow'],
            "Custom": ['thermal_stress', 'defect_gradient',
                       'crystal_orientation', 'stress_tensile_compressive']
        }
        selected_cmap = container.selectbox(
            "Select Colormap",
            category_maps[colormap_category],
            index=0,
            help="Choose from 50+ colormaps including rainbow, jet, turbo, inferno"
        )
        # Font controls
        container.markdown("#### 📝 Font Controls")
        col1, col2 = container.columns(2)
        with col1:
            font_family = col1.selectbox(
                "Font Family",
                self.font_families,
                index=0
            )
            title_size = col1.slider(
                "Title Font Size",
                min_value=8,
                max_value=24,
                value=self.default_params['title_size'],
                step=1
            )
        with col2:
            label_size = col2.slider(
                "Label Font Size",
                min_value=6,
                max_value=20,
                value=self.default_params['label_size'],
                step=1
            )
            tick_size = col2.slider(
                "Tick Font Size",
                min_value=6,
                max_value=18,
                value=self.default_params['tick_size'],
                step=1
            )
        # Line and marker controls
        container.markdown("#### 📈 Line & Marker Controls")
        col3, col4 = container.columns(2)
        with col3:
            line_width = col3.slider(
                "Line Width",
                min_value=0.5,
                max_value=10.0,
                value=self.default_params['line_width'],
                step=0.5
            )
            marker_size = col3.slider(
                "Marker Size",
                min_value=1,
                max_value=20,
                value=self.default_params['marker_size'],
                step=1
            )
        with col4:
            grid_alpha = col4.slider(
                "Grid Opacity",
                min_value=0.0,
                max_value=1.0,
                value=self.default_params['grid_alpha'],
                step=0.05
            )
            figure_dpi = col4.slider(
                "Figure DPI",
                min_value=72,
                max_value=300,
                value=self.default_params['figure_dpi'],
                step=12
            )
        # Colorbar controls
        container.markdown("#### 🎨 Colorbar Controls")
        col5, col6 = container.columns(2)
        with col5:
            colorbar_width = col5.slider(
                "Colorbar Width",
                min_value=0.01,
                max_value=0.1,
                value=self.default_params['colorbar_width'],
                step=0.005
            )
        with col6:
            colorbar_pad = col6.slider(
                "Colorbar Padding",
                min_value=0.01,
                max_value=0.2,
                value=self.default_params['colorbar_pad'],
                step=0.01
            )
        # Style options
        container.markdown("#### 🎭 Style Options")
        col7, col8 = container.columns(2)
        with col7:
            use_latex = col7.checkbox("Use LaTeX Rendering", value=False)
            dark_theme = col7.checkbox("Dark Theme", value=False)
        with col8:
            transparent_bg = col8.checkbox("Transparent Background", value=False)
            tight_layout = col8.checkbox("Tight Layout", value=True)
        # Return all settings
        return {
            'colormap': selected_cmap,
            'font_family': font_family,
            'title_size': title_size,
            'label_size': label_size,
            'tick_size': tick_size,
            'line_width': line_width,
            'marker_size': marker_size,
            'grid_alpha': grid_alpha,
            'figure_dpi': figure_dpi,
            'colorbar_width': colorbar_width,
            'colorbar_pad': colorbar_pad,
            'use_latex': use_latex,
            'dark_theme': dark_theme,
            'transparent_bg': transparent_bg,
            'tight_layout': tight_layout,
        }

    def apply_visualization_settings(self, fig=None, ax=None, settings=None):
        """
        Apply visualization settings to matplotlib figure/axes
        Args:
            fig: matplotlib figure object
            ax: matplotlib axes object
            settings: dictionary of visualization settings
        Returns:
            Updated figure and axes
        """
        if settings is None:
            settings = self.default_params
        if fig is not None:
            # Set figure DPI
            fig.set_dpi(settings['figure_dpi'])
            # Set background
            if settings['transparent_bg']:
                fig.patch.set_alpha(0.0)
            elif settings['dark_theme']:
                fig.patch.set_facecolor('black')
            # Apply tight layout
            if settings['tight_layout']:
                fig.tight_layout()
        if ax is not None:
            # Set font properties
            font_props = FontProperties(family=settings['font_family'])
            # Apply to title
            title = ax.get_title()
            if title:
                ax.set_title(title, fontsize=settings['title_size'],
                             fontproperties=font_props)
            # Apply to labels
            xlabel = ax.get_xlabel()
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=settings['label_size'],
                              fontproperties=font_props)
            ylabel = ax.get_ylabel()
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=settings['label_size'],
                              fontproperties=font_props)
            # Apply to ticks
            ax.tick_params(axis='both', which='major',
                           labelsize=settings['tick_size'])
            # Apply grid
            ax.grid(True, alpha=settings['grid_alpha'])
            # Apply to legend
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontproperties(font_props)
                    text.set_fontsize(settings.get('legend_size', settings['label_size']))
            # Set background
            if settings['dark_theme']:
                ax.set_facecolor('black')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
        return fig, ax

    def create_styled_colormap(self, cmap_name, n_colors=256):
        """
        Create styled colormap with enhanced properties
        Args:
            cmap_name: Name of the colormap
            n_colors: Number of colors in the colormap
        Returns:
            Colormap object
        """
        if cmap_name in self.colormaps:
            if isinstance(self.colormaps[cmap_name], str):
                # Matplotlib built-in colormap
                return plt.get_cmap(self.colormaps[cmap_name], n_colors)
            else:
                # Custom colormap
                return self.colormaps[cmap_name]
        else:
            # Fallback to viridis
            return plt.get_cmap('viridis', n_colors)

    def create_stress_visualization(self, geometry_data, stress_data, settings,
                                    orientation_angle=54.7, defect_type='Twin'):
        """
        Create comprehensive stress visualization for real geometry domain
        Args:
            geometry_data: Dictionary containing geometry information (eta, coordinates)
            stress_data: Dictionary containing stress fields
            settings: Visualization settings
            orientation_angle: Crystal orientation angle in degrees
            defect_type: Type of defect
        Returns:
            matplotlib figure object
        """
        # Extract data
        eta = geometry_data.get('eta', None)
        if eta is None or stress_data is None:
            st.warning("No geometry or stress data available for visualization")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        axes = axes.flatten()
        
        # Get colormaps
        cmap_hydro = self.create_styled_colormap('RdBu_r' if settings.get('stress_cmap_hydro') is None else settings['stress_cmap_hydro'])
        cmap_vonmises = self.create_styled_colormap('viridis' if settings.get('stress_cmap_vonmises') is None else settings['stress_cmap_vonmises'])
        cmap_magnitude = self.create_styled_colormap('plasma' if settings.get('stress_cmap_magnitude') is None else settings['stress_cmap_magnitude'])
        cmap_defect = self.create_styled_colormap('Set1' if settings.get('stress_cmap_defect') is None else settings['stress_cmap_defect'])
        
        # Plot 1: Phase Field (Eta) - Defect Regions
        ax1 = axes[0]
        if eta.ndim == 2:
            im1 = ax1.imshow(eta, cmap=cmap_defect, origin='lower',
                             extent=[0, eta.shape[1], 0, eta.shape[0]])
            ax1.set_title('Phase Field: Defect Regions', fontsize=settings['title_size'])
            ax1.set_xlabel('X Position', fontsize=settings['label_size'])
            ax1.set_ylabel('Y Position', fontsize=settings['label_size'])
            plt.colorbar(im1, ax=ax1, label='Order Parameter (η)', 
                        fraction=settings['colorbar_width'], pad=settings['colorbar_pad'])
        
        # Plot 2: Hydrostatic Stress
        ax2 = axes[1]
        sigma_hydro = stress_data.get('sigma_hydro', None)
        if sigma_hydro is not None and sigma_hydro.ndim == 2:
            vmax = np.nanmax(np.abs(sigma_hydro))
            vmin = -vmax if vmax > 0 else -1
            
            im2 = ax2.imshow(sigma_hydro, cmap=cmap_hydro, origin='lower',
                             vmin=vmin, vmax=vmax,
                             extent=[0, sigma_hydro.shape[1], 0, sigma_hydro.shape[0]])
            ax2.set_title(f'Hydrostatic Stress (σ_h)\nOrientation: {orientation_angle}°',
                          fontsize=settings['title_size'])
            ax2.set_xlabel('X Position', fontsize=settings['label_size'])
            ax2.set_ylabel('Y Position', fontsize=settings['label_size'])
            cbar2 = plt.colorbar(im2, ax=ax2, label='Stress (GPa)',
                                fraction=settings['colorbar_width'], pad=settings['colorbar_pad'])
            cbar2.ax.tick_params(labelsize=settings['tick_size'])
        
        # Plot 3: Von Mises Stress
        ax3 = axes[2]
        von_mises = stress_data.get('von_mises', None)
        if von_mises is not None and von_mises.ndim == 2:
            im3 = ax3.imshow(von_mises, cmap=cmap_vonmises, origin='lower',
                             extent=[0, von_mises.shape[1], 0, von_mises.shape[0]])
            ax3.set_title('Von Mises Stress (σ_vm)', fontsize=settings['title_size'])
            ax3.set_xlabel('X Position', fontsize=settings['label_size'])
            ax3.set_ylabel('Y Position', fontsize=settings['label_size'])
            cbar3 = plt.colorbar(im3, ax=ax3, label='Equivalent Stress (GPa)',
                                fraction=settings['colorbar_width'], pad=settings['colorbar_pad'])
            cbar3.ax.tick_params(labelsize=settings['tick_size'])
        
        # Plot 4: Stress Magnitude
        ax4 = axes[3]
        sigma_mag = stress_data.get('sigma_mag', None)
        if sigma_mag is not None and sigma_mag.ndim == 2:
            im4 = ax4.imshow(sigma_mag, cmap=cmap_magnitude, origin='lower',
                             extent=[0, sigma_mag.shape[1], 0, sigma_mag.shape[0]])
            ax4.set_title('Stress Magnitude (|σ|)', fontsize=settings['title_size'])
            ax4.set_xlabel('X Position', fontsize=settings['label_size'])
            ax4.set_ylabel('Y Position', fontsize=settings['label_size'])
            cbar4 = plt.colorbar(im4, ax=ax4, label='Stress Magnitude (GPa)',
                                fraction=settings['colorbar_width'], pad=settings['colorbar_pad'])
            cbar4.ax.tick_params(labelsize=settings['tick_size'])
        
        # Plot 5: Region-specific stress analysis
        ax5 = axes[4]
        if eta is not None:
            # Define regions based on eta
            defect_mask = eta > 0.6
            interface_mask = (eta >= 0.4) & (eta <= 0.6)
            bulk_mask = eta < 0.4
            
            # Create RGB image showing regions
            region_rgb = np.zeros((*eta.shape, 3))
            region_rgb[defect_mask] = [1, 0, 0]  # Red for defect
            region_rgb[interface_mask] = [0, 1, 0]  # Green for interface
            region_rgb[bulk_mask] = [0, 0, 1]  # Blue for bulk
            
            ax5.imshow(region_rgb, origin='lower',
                       extent=[0, eta.shape[1], 0, eta.shape[0]])
            ax5.set_title('Material Regions', fontsize=settings['title_size'])
            ax5.set_xlabel('X Position', fontsize=settings['label_size'])
            ax5.set_ylabel('Y Position', fontsize=settings['label_size'])
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.6, label='Defect (η > 0.6)'),
                Patch(facecolor='green', alpha=0.6, label='Interface (0.4 ≤ η ≤ 0.6)'),
                Patch(facecolor='blue', alpha=0.6, label='Bulk (η < 0.4)')
            ]
            ax5.legend(handles=legend_elements, loc='upper right',
                       fontsize=settings['tick_size'])
        
        # Plot 6: Stress distribution by region
        ax6 = axes[5]
        if eta is not None and sigma_hydro is not None:
            regions = []
            stresses = []
            
            defect_mask = eta > 0.6
            interface_mask = (eta >= 0.4) & (eta <= 0.6)
            bulk_mask = eta < 0.4
            
            for region_name, mask in [('Defect', defect_mask),
                                      ('Interface', interface_mask),
                                      ('Bulk', bulk_mask)]:
                if np.any(mask):
                    region_stresses = sigma_hydro[mask]
                    # Sample to avoid too many points
                    if len(region_stresses) > 1000:
                        region_stresses = np.random.choice(region_stresses, 1000)
                    regions.extend([region_name] * len(region_stresses))
                    stresses.extend(region_stresses)
            
            if regions and stresses:
                # Create box plot
                box_data = []
                labels = []
                colors = []
                
                for region, color in [('Defect', 'red'), ('Interface', 'green'), ('Bulk', 'blue')]:
                    region_indices = [i for i, r in enumerate(regions) if r == region]
                    if region_indices:
                        region_stresses = [stresses[i] for i in region_indices]
                        box_data.append(region_stresses)
                        labels.append(region)
                        colors.append(color)
                
                if box_data:
                    bp = ax6.boxplot(box_data, labels=labels, patch_artist=True)
                    
                    # Color boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                    
                    ax6.set_title('Stress Distribution by Region', fontsize=settings['title_size'])
                    ax6.set_ylabel('Hydrostatic Stress (GPa)', fontsize=settings['label_size'])
                    ax6.grid(True, alpha=settings['grid_alpha'], axis='y')
        
        # Apply universal settings to all axes
        for ax in axes:
            if ax.has_data():
                self.apply_visualization_settings(ax=ax, settings=settings)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=settings['title_size'])
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add main title
        fig.suptitle(f'Stress Analysis for {defect_type} Defect\n'
                     f'Orientation: {orientation_angle}°, '
                     f'Colormap: {settings.get("colormap", "Default")}',
                     fontsize=settings['title_size'] + 4,
                     fontweight='bold')
        
        return fig

    def create_3d_stress_visualization(self, geometry_data, stress_data, settings):
        """
        Create 3D stress visualization using Plotly
        Args:
            geometry_data: Dictionary containing geometry information
            stress_data: Dictionary containing stress fields
            settings: Visualization settings
        Returns:
            Plotly figure object
        """
        # Extract data
        eta = geometry_data.get('eta', None)
        sigma_hydro = stress_data.get('sigma_hydro', None)
        
        if eta is None or sigma_hydro is None:
            return None
        
        # Limit size for performance
        if eta.shape[0] * eta.shape[1] > 10000:
            # Downsample for performance
            step_x = max(1, eta.shape[1] // 100)
            step_y = max(1, eta.shape[0] // 100)
            eta = eta[::step_y, ::step_x]
            sigma_hydro = sigma_hydro[::step_y, ::step_x]
        
        # Create meshgrid
        x = np.arange(eta.shape[1])
        y = np.arange(eta.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Phase Field (η)', 'Hydrostatic Stress',
                            'Von Mises Stress', 'Stress Magnitude'),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Plot 1: Phase Field
        fig.add_trace(
            go.Surface(z=eta, x=X, y=Y,
                       colorscale=settings.get('colormap', 'viridis'),
                       showscale=True,
                       colorbar=dict(title="η", len=0.4, y=0.8),
                       name="Phase Field"),
            row=1, col=1
        )
        
        # Plot 2: Hydrostatic Stress
        fig.add_trace(
            go.Surface(z=sigma_hydro, x=X, y=Y,
                       colorscale='RdBu',
                       showscale=True,
                       colorbar=dict(title="σ_h (GPa)", len=0.4, y=0.8),
                       name="Hydrostatic Stress"),
            row=1, col=2
        )
        
        # Plot 3: Von Mises Stress
        von_mises = stress_data.get('von_mises', None)
        if von_mises is not None:
            if von_mises.shape[0] * von_mises.shape[1] > 10000:
                von_mises = von_mises[::step_y, ::step_x]
            
            fig.add_trace(
                go.Surface(z=von_mises, x=X, y=Y,
                           colorscale='viridis',
                           showscale=True,
                           colorbar=dict(title="σ_vm (GPa)", len=0.4, y=0.3),
                           name="Von Mises Stress"),
                row=2, col=1
            )
        
        # Plot 4: Stress Magnitude
        sigma_mag = stress_data.get('sigma_mag', None)
        if sigma_mag is not None:
            if sigma_mag.shape[0] * sigma_mag.shape[1] > 10000:
                sigma_mag = sigma_mag[::step_y, ::step_x]
            
            fig.add_trace(
                go.Surface(z=sigma_mag, x=X, y=Y,
                           colorscale='plasma',
                           showscale=True,
                           colorbar=dict(title="|σ| (GPa)", len=0.4, y=0.3),
                           name="Stress Magnitude"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="3D Stress Visualization",
            title_font_size=settings['title_size'] + 4,
            height=800,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            scene2=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            scene3=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            scene4=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            ),
            showlegend=False
        )
        
        return fig

# =============================================
# ENHANCED PHYSICS-BASED STRESS ANALYZER WITH REAL GEOMETRY VISUALIZATION
# =============================================

class EnhancedPhysicsBasedStressAnalyzer(PhysicsBasedStressAnalyzer):
    """Enhanced physics-based analyzer with real geometry visualization"""
    def __init__(self):
        super().__init__()
        self.visualizer = UniversalVisualizationEnhancer()

    def create_comprehensive_stress_report(self, analysis_results, settings=None,
                                           defect_type='Twin', orientation_deg=54.7):
        """
        Create comprehensive stress analysis report with visualizations
        Args:
            analysis_results: Results from compute_real_geometry_stress
            settings: Visualization settings
            defect_type: Type of defect
            orientation_deg: Orientation angle
        Returns:
            Dictionary with figures and analysis
        """
        if settings is None:
            settings = self.visualizer.default_params
        
        report = {
            'metadata': {
                'defect_type': defect_type,
                'orientation_deg': orientation_deg,
                'analysis_time': datetime.now().isoformat()
            },
            'figures': {},
            'analysis': {}
        }
        
        # Create 2D stress visualization
        fig_2d = self.visualizer.create_stress_visualization(
            analysis_results.get('geometry_data', {}),
            analysis_results.get('stress_data', {}),
            settings,
            orientation_deg,
            defect_type
        )
        
        if fig_2d is not None:
            report['figures']['stress_2d'] = fig_2d
        
        # Create 3D visualization if data is available
        try:
            fig_3d = self.visualizer.create_3d_stress_visualization(
                analysis_results.get('geometry_data', {}),
                analysis_results.get('stress_data', {}),
                settings
            )
            if fig_3d is not None:
                report['figures']['stress_3d'] = fig_3d
        except Exception as e:
            st.warning(f"3D visualization failed: {str(e)}")
        
        # Create region comparison plots
        fig_regions = self._create_region_comparison_plots(
            analysis_results.get('region_stats', {}),
            settings,
            defect_type,
            orientation_deg
        )
        
        if fig_regions is not None:
            report['figures']['region_comparison'] = fig_regions
        
        # Store analysis results
        report['analysis'] = {
            'region_stats': analysis_results.get('region_stats', {}),
            'orientation_analysis': analysis_results.get('orientation_analysis', {}),
            'strain_energy': analysis_results.get('strain_energy', {}),
            'metadata': analysis_results.get('metadata', {})
        }
        
        # Add concentration factors
        for key, value in analysis_results.items():
            if 'concentration_factor' in key:
                report['analysis'][key] = value
        
        return report

    def _create_region_comparison_plots(self, region_stats, settings, defect_type, orientation_deg):
        """
        Create comparison plots for different regions
        Args:
            region_stats: Statistics for each region
            settings: Visualization settings
            defect_type: Type of defect
            orientation_deg: Orientation angle
        Returns:
            matplotlib figure object
        """
        if not region_stats:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Prepare data for plotting
        regions = ['defect', 'interface', 'bulk']
        colors = ['red', 'green', 'blue']
        region_labels = ['Defect', 'Interface', 'Bulk']
        
        # Plot 1: Mean stress comparison
        ax1 = axes[0]
        stress_components = ['sigma_hydro', 'von_mises', 'sigma_mag']
        
        for i, region in enumerate(regions):
            if region in region_stats:
                means = []
                stds = []
                
                for comp in stress_components:
                    if comp in region_stats[region]:
                        means.append(region_stats[region][comp]['mean'])
                        stds.append(region_stats[region][comp]['std'])
                    else:
                        means.append(0)
                        stds.append(0)
                
                x_pos = np.arange(len(stress_components)) + i * 0.25
                bars = ax1.bar(x_pos, means, width=0.2, color=colors[i], 
                              alpha=0.7, label=region_labels[i], yerr=stds, capsize=3)
        
        ax1.set_xlabel('Stress Component', fontsize=settings['label_size'])
        ax1.set_ylabel('Mean Stress (GPa)', fontsize=settings['label_size'])
        ax1.set_title('Mean Stress by Region and Component', fontsize=settings['title_size'])
        ax1.set_xticks(np.arange(len(stress_components)) + 0.25)
        ax1.set_xticklabels(['σ_h', 'σ_vm', '|σ|'])
        ax1.legend(fontsize=settings['tick_size'])
        ax1.grid(True, alpha=settings['grid_alpha'], axis='y')
        
        # Plot 2: Maximum stress comparison
        ax2 = axes[1]
        for i, region in enumerate(regions):
            if region in region_stats:
                max_stresses = []
                
                for comp in stress_components:
                    if comp in region_stats[region]:
                        max_stresses.append(region_stats[region][comp]['max'])
                    else:
                        max_stresses.append(0)
                
                x_pos = np.arange(len(stress_components)) + i * 0.25
                ax2.bar(x_pos, max_stresses, width=0.2, color=colors[i], alpha=0.7)
        
        ax2.set_xlabel('Stress Component', fontsize=settings['label_size'])
        ax2.set_ylabel('Maximum Stress (GPa)', fontsize=settings['label_size'])
        ax2.set_title('Maximum Stress by Region and Component', fontsize=settings['title_size'])
        ax2.set_xticks(np.arange(len(stress_components)) + 0.25)
        ax2.set_xticklabels(['σ_h', 'σ_vm', '|σ|'])
        ax2.grid(True, alpha=settings['grid_alpha'], axis='y')
        
        # Plot 3: Area fractions
        ax3 = axes[2]
        area_fractions = []
        area_labels = []
        area_colors = []
        
        for region, color in zip(regions, colors):
            if region in region_stats and 'sigma_hydro' in region_stats[region]:
                area_fractions.append(region_stats[region]['sigma_hydro']['area_fraction'] * 100)
                area_labels.append(region.capitalize())
                area_colors.append(color)
        
        if area_fractions:
            wedges, texts, autotexts = ax3.pie(area_fractions, labels=area_labels,
                                               colors=area_colors, autopct='%1.1f%%',
                                               startangle=90)
            
            # Improve label appearance
            for text in texts:
                text.set_fontsize(settings['tick_size'])
            for autotext in autotexts:
                autotext.set_fontsize(settings['tick_size'] - 2)
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            ax3.set_title('Area Fraction of Each Region', fontsize=settings['title_size'])
        
        # Plot 4: Stress concentration factors
        ax4 = axes[3]
        concentration_factors = []
        concentration_labels = []
        concentration_colors = []
        
        # Compute concentration factors for defect and interface regions
        if 'defect' in region_stats and 'bulk' in region_stats:
            if 'sigma_hydro' in region_stats['defect'] and 'sigma_hydro' in region_stats['bulk']:
                defect_mean = region_stats['defect']['sigma_hydro']['mean']
                bulk_mean = region_stats['bulk']['sigma_hydro']['mean']
                
                if bulk_mean != 0:
                    concentration_factors.append(defect_mean / bulk_mean)
                    concentration_labels.append('Defect/Bulk')
                    concentration_colors.append('red')
        
        if 'interface' in region_stats and 'bulk' in region_stats:
            if 'sigma_hydro' in region_stats['interface'] and 'sigma_hydro' in region_stats['bulk']:
                interface_mean = region_stats['interface']['sigma_hydro']['mean']
                bulk_mean = region_stats['bulk']['sigma_hydro']['mean']
                
                if bulk_mean != 0:
                    concentration_factors.append(interface_mean / bulk_mean)
                    concentration_labels.append('Interface/Bulk')
                    concentration_colors.append('green')
        
        if concentration_factors:
            x_pos = np.arange(len(concentration_factors))
            bars = ax4.bar(x_pos, concentration_factors, color=concentration_colors, alpha=0.7)
            
            ax4.set_xlabel('Region Comparison', fontsize=settings['label_size'])
            ax4.set_ylabel('Concentration Factor', fontsize=settings['label_size'])
            ax4.set_title('Stress Concentration Factors\n(Relative to Bulk)', fontsize=settings['title_size'])
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(concentration_labels)
            ax4.grid(True, alpha=settings['grid_alpha'], axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, concentration_factors):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.2f}x', ha='center', va='bottom',
                         fontsize=settings['tick_size'])
        
        # Apply universal settings
        for ax in axes:
            self.visualizer.apply_visualization_settings(ax=ax, settings=settings)
        
        # Add overall title
        fig.suptitle(f'Region Analysis: {defect_type} at {orientation_deg}°',
                     fontsize=settings['title_size'] + 2, fontweight='bold')
        
        plt.tight_layout()
        return fig

# =============================================
# ENHANCED WORKFLOW PRESENTER
# =============================================

class EnhancedWorkflowPresenter:
    """
    Enhanced workflow presenter with detailed captions, explanations,
    and step-by-step guidance for stress interpolation and sintering prediction
    """
    def __init__(self):
        self.captions = self._initialize_captions()
        self.physics_explanations = self._initialize_physics_explanations()
        self.visualizer = UniversalVisualizationEnhancer()

    def _initialize_captions(self):
        """Initialize comprehensive captions for all visualizations"""
        return {
            'phase_field': {
                'title': "Phase Field Visualization of Material Defects",
                'description': "Shows the spatial distribution of the order parameter η, distinguishing between defect regions (η > 0.6), interfaces (0.4 ≤ η ≤ 0.6), and bulk material (η < 0.4).",
                'physics': "The phase field parameter η characterizes the local material state, with higher values indicating stronger defect presence and corresponding eigen strain fields.",
                'interpretation': "Red regions indicate strong defect presence, green shows transitional interfaces, and blue represents relatively defect-free bulk material."
            },
            'hydrostatic_stress': {
                'title': "Hydrostatic Stress Distribution",
                'description': "Visualizes the mean normal stress σ_h = (σ₁ + σ₂ + σ₃)/3, which governs volume changes and is critical for sintering processes.",
                'physics': "Compressive hydrostatic stress (negative, blue) promotes densification, while tensile stress (positive, red) can lead to cracking.",
                'interpretation': "Blue regions experience compressive stress beneficial for sintering, while red regions face tensile stress that may inhibit densification."
            },
            'von_mises_stress': {
                'title': "Von Mises Equivalent Stress",
                'description': "Shows the scalar stress measure σ_vm = √[½((σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²)] indicating potential for plastic deformation.",
                'physics': "Von Mises stress predicts yield onset according to the distortion energy theory, with higher values suggesting increased likelihood of plastic flow.",
                'interpretation': "Bright yellow regions indicate high shear stress concentrations where plastic deformation may initiate during sintering."
            },
            'stress_magnitude': {
                'title': "Total Stress Magnitude",
                'description': "Displays the overall stress intensity |σ| = √(σ:σ), combining all stress tensor components.",
                'physics': "Stress magnitude indicates the total elastic energy stored in the material, influencing diffusion kinetics and defect mobility.",
                'interpretation': "Regions with high stress magnitude act as drivers for stress-assisted diffusion and accelerated sintering."
            },
            'region_comparison': {
                'title': "Stress Analysis by Material Region",
                'description': "Compares stress statistics across defect, interface, and bulk regions to quantify defect-induced stress enhancement.",
                'physics': "Defects create eigen strain fields that locally amplify stress, with the degree of amplification quantified by concentration factors.",
                'interpretation': "High concentration factors in defect regions indicate significant stress amplification, crucial for low-temperature sintering."
            },
            'sintering_prediction': {
                'title': "Sintering Temperature Prediction",
                'description': "Predicts optimal sintering temperatures based on stress-modified diffusion kinetics using Arrhenius and exponential models.",
                'physics': "Hydrostatic stress reduces activation energy for diffusion: Q_eff = Q_a - Ωσ, enabling sintering at lower temperatures.",
                'interpretation': "Lower predicted temperatures indicate more favorable conditions for low-temperature sintering, with defect engineering enabling significant reductions."
            }
        }

    def _initialize_physics_explanations(self):
        """Initialize detailed physics explanations"""
        return {
            'stress_interpolation': """
## Stress Interpolation Methodology
**Physics-Aware Interpolation with Combined Regularization:**

1. **Parameter Space Embedding:** Each simulation is represented as a 15-dimensional vector encoding:
   - Defect type (one-hot encoded: ISF, ESF, Twin, No Defect)
   - Shape parameters (Square, Fault, Rectangle)
   - Physical parameters (ε*, κ, θ)
   - Physics-derived features (habit plane proximity, stress concentration factors)

2. **Attention Mechanism:** Multi-head attention weights simulations based on similarity to target parameters:
   - Query: Target parameter vector
   - Key/Value: Source simulation vectors
   - Output: Weighted combination of source stresses

3. **Gaussian Spatial Regularization:** Applies locality constraint in parameter space:
   - Weights decay exponentially with Euclidean distance
   - Preserves smoothness in orientation-dependent responses

4. **Physics Constraints:** Incorporates domain knowledge:
   - Eigen strain compatibility
   - Habit plane symmetry (54.7° for Ag FCC twins)
   - Defect interaction energies
""",
            'sintering_prediction_workflow': """
## Sintering Temperature Prediction Workflow

**Step 1: Stress Extraction & Interpolation**
- Extract hydrostatic stress from interpolated stress fields
- Focus on maximum absolute stress in defect regions
- Account for orientation effects near habit plane

**Step 2: Activation Energy Reduction**
- Compute effective activation energy: Q_eff = Q_a - Ω|σ|
- Ω = activation volume (converts stress to energy)
- Defect-specific adjustments (twin boundaries reduce Q_a more)

**Step 3: Arrhenius Temperature Calculation**
- Solve Arrhenius equation: D = D₀ exp(-Q_eff/kT)
- Set D = D_critical (required for sintering)
- Compute T_sinter = Q_eff / [k ln(D₀/D_crit)]

**Step 4: System Classification**
- System 1 (Perfect): σ < 5 GPa, T ≈ 620-630 K
- System 2 (SF/Twin): 5 ≤ σ < 20 GPa, T ≈ 450-550 K
- System 3 (Plastic): σ ≥ 20 GPa, T ≈ 350-400 K
""",
            'defect_physics': """
## Defect Physics & Stress Enhancement

**Eigen Strain Origins:**
- **ISF (ε* = 0.71):** Missing {111} atomic plane creates localized compression
- **ESF (ε* = 1.41):** Extra {111} plane introduces tensile strain field
- **Twin (ε* = 2.12):** Mirror symmetry creates alternating compressive/tensile fields

**Stress Concentration Mechanisms:**
1. **Elastic Mismatch:** Different elastic properties at defect interfaces
2. **Eigen Strain:** Incompatible deformation within defects
3. **Interface Curvature:** Geometric stress intensification
4. **Crystal Anisotropy:** Orientation-dependent elastic response

**Habit Plane Effects (54.7°):**
- Maximum Schmid factor for twinning
- Optimal stress transmission across {111} planes
- Enhanced defect-defect interactions
""",
            'sintering_mechanisms': """
## Stress-Modified Sintering Mechanisms

**Key Physical Principles:**

1. **Stress-Assisted Diffusion:**
   - Hydrostatic stress reduces energy barriers for atomic motion
   - Driving force: ∇μ = Ω∇σ (chemical potential gradient)
   - Enhanced vacancy migration to pore surfaces

2. **Dislocation Climb Acceleration:**
   - Stress aids dislocation motion through climb
   - Faster grain boundary migration
   - Reduced recovery time

3. **Interface Reaction Enhancement:**
   - Stress lowers activation energy for surface reactions
   - Faster neck growth between particles
   - Improved densification kinetics

4. **Defect-Mediated Pathways:**
   - Twins and stacking faults provide fast diffusion paths
   - Reduced effective diffusion distances
   - Lower overall sintering temperatures
"""
        }

    def create_caption_box(self, caption_key, analysis_results=None):
        """
        Create styled caption box for visualization
        Args:
            caption_key: Key identifying the caption type
            analysis_results: Optional analysis results for dynamic content
        Returns:
            Streamlit markdown with styled caption
        """
        if caption_key not in self.captions:
            return ""
        
        caption = self.captions[caption_key]
        
        # Add dynamic content if analysis results provided
        dynamic_content = ""
        if analysis_results is not None:
            if caption_key == 'hydrostatic_stress' and 'region_stats' in analysis_results:
                defect_stats = analysis_results['region_stats'].get('defect', {})
                if 'sigma_hydro' in defect_stats:
                    defect_stress = defect_stats['sigma_hydro']['mean']
                    dynamic_content = f"\n**Current Analysis:** Defect region shows mean hydrostatic stress of {defect_stress:.2f} GPa."
            
            elif caption_key == 'sintering_prediction' and 'temperature_predictions' in analysis_results:
                temps = analysis_results['temperature_predictions']
                dynamic_content = f"\n**Prediction:** Arrhenius model suggests {temps.get('arrhenius_defect_k', 0):.0f} K ({temps.get('arrhenius_defect_c', 0):.0f}°C) sintering temperature."
        
        caption_html = f"""
<div style="background-color: #f0f7ff; border-left: 5px solid #3b82f6; padding: 15px; margin: 10px 0; border-radius: 5px;">
<h4 style="color: #1e40af; margin-top: 0;">{caption['title']}</h4>
<p><strong>Description:</strong> {caption['description']}</p>
<p><strong>Physics:</strong> {caption['physics']}</p>
<p><strong>Interpretation:</strong> {caption['interpretation']}{dynamic_content}</p>
</div>
"""
        return caption_html

    def create_physics_explanation(self, explanation_key, container=None):
        """
        Create detailed physics explanation
        Args:
            explanation_key: Key identifying the explanation type
            container: Streamlit container to place explanation in
        """
        if container is None:
            container = st
        
        if explanation_key in self.physics_explanations:
            with container.expander(f"📚 Detailed Physics: {explanation_key.replace('_', ' ').title()}", expanded=False):
                container.markdown(self.physics_explanations[explanation_key])

# =============================================
# ENHANCED MAIN APPLICATION WITH COMPREHENSIVE VISUALIZATION CONTROLS
# =============================================

def main():
    # Configure Streamlit page with enhanced settings
    st.set_page_config(
        page_title="Ag FCC Twin: Enhanced Stress & Sintering Analysis",
        layout="wide",
        page_icon="🔬⚙️",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourrepo',
            'Report a bug': "https://github.com/yourrepo/issues",
            'About': "# Enhanced Ag FCC Twin Analysis\nComprehensive stress visualization with 50+ colormaps and universal controls"
        }
    )
    
    # Enhanced CSS with more styling options
    st.markdown("""
<style>
/* Main header styling */
.main-header {
font-size: 3.5rem !important;
background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #F59E0B);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
text-align: center;
font-weight: 900 !important;
margin-bottom: 1.5rem;
padding: 1rem;
text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

/* Physics equation styling */
.physics-equation {
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
padding: 1.5rem;
border-radius: 15px;
color: white;
font-family: "Cambria Math", "Times New Roman", serif;
font-size: 1.3rem;
margin: 1.5rem 0;
box-shadow: 0 6px 12px rgba(0,0,0,0.15);
border: 3px solid rgba(255,255,255,0.2);
}

/* Caption box styling */
.caption-box {
background: linear-gradient(135deg, #f0f7ff 0%, #e1f5fe 100%);
border-left: 6px solid #3b82f6;
padding: 1.2rem;
margin: 1rem 0;
border-radius: 10px;
box-shadow: 0 4px 6px rgba(0,0,0,0.05);
transition: transform 0.2s;
}

.caption-box:hover {
transform: translateY(-2px);
box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}

/* Metric card styling */
.metric-card {
background: white;
border-radius: 12px;
padding: 1.2rem;
margin: 0.5rem;
box-shadow: 0 4px 6px rgba(0,0,0,0.05);
border: 1px solid #e5e7eb;
transition: all 0.3s ease;
}

.metric-card:hover {
box-shadow: 0 8px 15px rgba(0,0,0,0.1);
transform: translateY(-3px);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
gap: 2rem;
padding: 0 1rem;
}

.stTabs [data-baseweb="tab"] {
padding: 1rem 2rem;
border-radius: 8px 8px 0 0;
background-color: #f3f4f6;
border: 1px solid #e5e7eb;
font-weight: 600;
transition: all 0.3s;
}

.stTabs [data-baseweb="tab"]:hover {
background-color: #e5e7eb;
}

.stTabs [aria-selected="true"] {
background-color: #3b82f6 !important;
color: white !important;
box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
}

/* Button enhancements */
.stButton > button {
border-radius: 8px !important;
padding: 0.5rem 1.5rem !important;
font-weight: 600 !important;
transition: all 0.3s !important;
border: 2px solid transparent !important;
}

.stButton > button:hover {
transform: translateY(-2px);
box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

/* Sidebar enhancements */
.css-1d391kg {
background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}

/* Custom scrollbar */
::-webkit-scrollbar {
width: 10px;
}

::-webkit-scrollbar-track {
background: #f1f1f1;
border-radius: 5px;
}

::-webkit-scrollbar-thumb {
background: #888;
border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
background: #555;
}
</style>
""", unsafe_allow_html=True)
    
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">🔬⚙️ Enhanced Ag FCC Twin Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">Comprehensive stress visualization with 50+ colormaps • Universal enhancement controls • Physics-aware interpolation</p>', unsafe_allow_html=True)
    
    # Physics equations showcase
    st.markdown("""
<div class="physics-equation">
<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
<div>
<strong>Stress-Modified Diffusion:</strong><br>
D(σ,T) = D₀ exp[-(Qₐ - Ωσ)/(k<sub>B</sub>T)]
</div>
<div>
<strong>Sintering Temperature:</strong><br>
T<sub>sinter</sub>(σ) = (Qₐ - Ω|σ|)/[k<sub>B</sub> ln(D₀/D<sub>crit</sub>)]
</div>
<div>
<strong>Hydrostatic Stress:</strong><br>
σ<sub>h</sub> = ⅓(σ₁₁ + σ₂₂ + σ₃₃)
</div>
</div>
</div>
""", unsafe_allow_html=True)
    
    # Initialize session state
    if 'enhanced_analyzer' not in st.session_state:
        st.session_state.enhanced_analyzer = EnhancedPhysicsBasedStressAnalyzer()
    
    if 'visualization_enhancer' not in st.session_state:
        st.session_state.visualization_enhancer = UniversalVisualizationEnhancer()
    
    if 'workflow_presenter' not in st.session_state:
        st.session_state.workflow_presenter = EnhancedWorkflowPresenter()
    
    if 'physics_analyzer' not in st.session_state:
        st.session_state.physics_analyzer = PhysicsBasedStressAnalyzer()
    
    if 'sintering_calculator' not in st.session_state:
        st.session_state.sintering_calculator = EnhancedSinteringCalculator()
    
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator()
    
    # Set up directories
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
    
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    
    # Enhanced sidebar with visualization controls
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")
        
        # Analysis mode selection
        analysis_mode = st.radio(
            "**Analysis Mode:**",
            ["Real Geometry Visualization", "Habit Plane Vicinity",
             "Defect Comparison", "Comprehensive Dashboard", "Workflow Analysis"],
            index=0,
            help="Select the primary analysis mode"
        )
        
        # Universal visualization controls (always available)
        st.markdown("### 🎨 Universal Visualization Controls")
        viz_settings = st.session_state.visualization_enhancer.create_visualization_controls()
        
        # Stress-specific visualization controls
        with st.expander("🔧 Stress Visualization Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                stress_cmap_hydro = st.selectbox(
                    "Hydrostatic CMAP",
                    ["RdBu_r", "coolwarm", "bwr", "seismic", "stress_tensile_compressive"],
                    index=0
                )
                contour_levels = st.slider("Contour Levels", 5, 50, 20)
            with col2:
                stress_cmap_vonmises = st.selectbox(
                    "Von Mises CMAP",
                    ["viridis", "plasma", "inferno", "magma", "hot"],
                    index=0
                )
                vector_scale = st.slider("Vector Scale", 10, 100, 50)
            
            # Add stress-specific settings to viz_settings
            viz_settings.update({
                'stress_cmap_hydro': stress_cmap_hydro,
                'stress_cmap_vonmises': stress_cmap_vonmises,
                'contour_levels': contour_levels,
                'vector_scale': vector_scale
            })
        
        # Data management
        st.markdown("### 📂 Data Management")
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            if st.button("🔄 Load Solutions", use_container_width=True,
                         help="Load all simulation solutions from the solutions directory"):
                with st.spinner("Loading solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
        
        with col_load2:
            if st.button("🧹 Clear Cache", use_container_width=True,
                         help="Clear all cached data and reload"):
                st.session_state.solutions = []
                st.cache_data.clear()
                st.rerun()
        
        # Target parameters
        st.markdown("### 🎯 Target Parameters")
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for analysis"
        )
        
        # Auto-set eigen strain
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            eps0 = st.number_input(
                "Eigen Strain (ε*)",
                min_value=0.0,
                max_value=3.0,
                value=default_eps0,
                step=0.01,
                help="Eigen strain magnitude"
            )
            orientation_angle = st.slider(
                "Orientation Angle (°)",
                min_value=0.0,
                max_value=360.0,
                value=54.7,
                step=0.1,
                help="Crystal orientation angle relative to habit plane"
            )
        
        with col_param2:
            kappa = st.slider(
                "Interface Energy (κ)",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.01
            )
            shape = st.selectbox(
                "Geometry Shape",
                ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
                index=0
            )
        
        # Region selection for analysis
        region_type = st.selectbox(
            "Analysis Region",
            ["defect", "interface", "bulk", "all"],
            index=0,
            help="Select material region for detailed analysis"
        )
        
        # Generate analysis button
        st.markdown("---")
        generate_text = "🚀 Generate Enhanced Analysis" if analysis_mode != "Workflow Analysis" else "📋 Show Complete Workflow"
        
        if st.button(generate_text, type="primary", use_container_width=True):
            st.session_state.generate_analysis = True
            st.session_state.current_analysis_mode = analysis_mode
        else:
            st.session_state.generate_analysis = False
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Create tabs for different analysis sections
        if st.session_state.solutions:
            tab_names = ["🏠 Dashboard", "📊 Real Geometry", "🎯 Habit Plane",
                         "🔬 Defect Comparison", "📈 Sintering Analysis", "📋 Workflow"]
            tabs = st.tabs(tab_names)
            
            with tabs[0]:  # Dashboard
                st.markdown("## 🏠 Comprehensive Dashboard")
                
                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Solutions Loaded", len(st.session_state.solutions))
                with col2:
                    defect_count = len(set(s.get('params', {}).get('defect_type', 'Unknown')
                                           for s in st.session_state.solutions))
                    st.metric("Unique Defects", defect_count)
                with col3:
                    has_physics = sum(1 for s in st.session_state.solutions
                                      if s.get('physics_analysis'))
                    st.metric("Physics Analyzed", f"{has_physics}/{len(st.session_state.solutions)}")
                with col4:
                    st.metric("Current Mode", analysis_mode)
                
                # Recent analyses
                st.markdown("### 📋 Recent Analyses")
                if 'recent_analyses' not in st.session_state:
                    st.session_state.recent_analyses = []
                
                for analysis in st.session_state.recent_analyses[-3:]:
                    st.info(f"**{analysis['type']}** - {analysis['time']}: {analysis['description']}")
                
                # Quick actions
                st.markdown("### ⚡ Quick Actions")
                col_act1, col_act2, col_act3 = st.columns(3)
                with col_act1:
                    if st.button("🔄 Re-run Last Analysis", use_container_width=True):
                        if st.session_state.recent_analyses:
                            last_analysis = st.session_state.recent_analyses[-1]
                            st.info(f"Re-running: {last_analysis['type']}")
                
                with col_act2:
                    if st.button("📊 Export All Data", use_container_width=True):
                        # Create export package
                        export_data = {
                            'solutions_count': len(st.session_state.solutions),
                            'analysis_mode': analysis_mode,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Convert to JSON for download
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="Download Export",
                            data=json_str,
                            file_name="analysis_export.json",
                            mime="application/json"
                        )
                
                with col_act3:
                    if st.button("📈 Generate Report", use_container_width=True):
                        st.info("Report generation feature coming soon!")
            
            with tabs[1]:  # Real Geometry Visualization
                st.markdown("## 📊 Real Geometry Stress Visualization")
                
                if st.session_state.get('generate_analysis', False) and \
                        st.session_state.get('current_analysis_mode') == "Real Geometry Visualization":
                    
                    # Show workflow steps
                    st.markdown("### 🔄 Analysis Workflow")
                    
                    # Step 1: Select solution
                    st.markdown("#### Step 1: Select Solution for Analysis")
                    solution_options = [s.get('metadata', {}).get('filename', f"Solution {i}")
                                        for i, s in enumerate(st.session_state.solutions)]
                    selected_solution = st.selectbox("Select Solution", solution_options, index=0)
                    
                    if selected_solution:
                        solution_idx = solution_options.index(selected_solution)
                        solution = st.session_state.solutions[solution_idx]
                        
                        # Step 2: Extract data
                        st.markdown("#### Step 2: Extract and Process Data")
                        
                        history = solution.get('history', [])
                        if history:
                            last_frame = history[-1]
                            eta = last_frame.get('eta')
                            stress_fields = last_frame.get('stresses', {})
                            
                            if eta is not None and stress_fields:
                                # Convert tensors to numpy if needed
                                if torch.is_tensor(eta):
                                    eta = eta.cpu().numpy()
                                
                                processed_stresses = {}
                                for key, value in stress_fields.items():
                                    if torch.is_tensor(value):
                                        processed_stresses[key] = value.cpu().numpy()
                                    else:
                                        processed_stresses[key] = value
                                
                                # Step 3: Compute analysis
                                st.markdown("#### Step 3: Compute Stress Analysis")
                                
                                analysis_results = st.session_state.enhanced_analyzer.compute_real_geometry_stress(
                                    eta, processed_stresses, orientation_angle
                                )
                                
                                # Create visualization report
                                report = st.session_state.enhanced_analyzer.create_comprehensive_stress_report(
                                    analysis_results, viz_settings, defect_type, orientation_angle
                                )
                                
                                # Display figures with captions
                                if 'figures' in report:
                                    for fig_name, fig in report['figures'].items():
                                        st.markdown(f"### 📈 {fig_name.replace('_', ' ').title()}")
                                        
                                        # Display caption
                                        caption_key = fig_name.split('_')[0] if '_' in fig_name else fig_name
                                        if caption_key in ['stress', 'region']:
                                            caption_key = 'region_comparison' if 'region' in fig_name else 'phase_field'
                                        
                                        st.markdown(
                                            st.session_state.workflow_presenter.create_caption_box(
                                                caption_key, analysis_results
                                            ), unsafe_allow_html=True
                                        )
                                        
                                        # Display figure
                                        if fig_name == 'stress_3d':
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.pyplot(fig)
                                        
                                        # Add download buttons with type checking
                                        col_fig1, col_fig2 = st.columns(2)
                                        
                                        with col_fig1:
                                            if fig is None:
                                                st.warning("Figure is not available")
                                            elif isinstance(fig, plt.Figure) or isinstance(fig, matplotlib.figure.Figure):
                                                buf = BytesIO()
                                                fig.savefig(buf, format="png", dpi=viz_settings['figure_dpi'])
                                                buf.seek(0)
                                                st.download_button(
                                                    label="📥 Download PNG",
                                                    data=buf,
                                                    file_name=f"{fig_name}_{defect_type}_{orientation_angle}deg.png",
                                                    mime="image/png",
                                                    use_container_width=True
                                                )
                                            elif isinstance(fig, go.Figure):
                                                try:
                                                    # Try to save as PNG
                                                    img_bytes = fig.to_image(format="png", scale=2)
                                                    st.download_button(
                                                        label="📥 Download PNG",
                                                        data=img_bytes,
                                                        file_name=f"{fig_name}_{defect_type}_{orientation_angle}deg.png",
                                                        mime="image/png",
                                                        use_container_width=True
                                                    )
                                                except Exception as e:
                                                    st.warning(f"Could not save Plotly figure as PNG: {e}")
                                        
                                        with col_fig2:
                                            if fig is None:
                                                st.warning("Figure is not available")
                                            elif isinstance(fig, plt.Figure) or isinstance(fig, matplotlib.figure.Figure):
                                                buf_pdf = BytesIO()
                                                fig.savefig(buf_pdf, format="pdf")
                                                buf_pdf.seek(0)
                                                st.download_button(
                                                    label="📥 Download PDF",
                                                    data=buf_pdf,
                                                    file_name=f"{fig_name}_{defect_type}_{orientation_angle}deg.pdf",
                                                    mime="application/pdf",
                                                    use_container_width=True
                                                )
                                            elif isinstance(fig, go.Figure):
                                                try:
                                                    # Try to save as PDF
                                                    pdf_bytes = fig.to_image(format="pdf", scale=2)
                                                    st.download_button(
                                                        label="📥 Download PDF",
                                                        data=pdf_bytes,
                                                        file_name=f"{fig_name}_{defect_type}_{orientation_angle}deg.pdf",
                                                        mime="application/pdf",
                                                        use_container_width=True
                                                    )
                                                except Exception as e:
                                                    st.warning(f"Could not save Plotly figure as PDF: {e}")
                                        
                                        st.markdown("---")
                                
                                # Display analysis results
                                st.markdown("### 📊 Quantitative Analysis Results")
                                
                                if 'analysis' in report:
                                    # Region statistics
                                    st.markdown("#### 📍 Region Statistics")
                                    region_stats = report['analysis'].get('region_stats', {})
                                    
                                    for region_name, stats in region_stats.items():
                                        with st.expander(f"{region_name.capitalize()} Region Analysis", expanded=False):
                                            if 'sigma_hydro' in stats:
                                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                                with col_stat1:
                                                    st.metric("Mean σ_h", f"{stats['sigma_hydro']['mean']:.3f} GPa")
                                                with col_stat2:
                                                    st.metric("Max σ_h", f"{stats['sigma_hydro']['max']:.3f} GPa")
                                                with col_stat3:
                                                    st.metric("Std Dev", f"{stats['sigma_hydro']['std']:.3f} GPa")
                                                with col_stat4:
                                                    st.metric("Area Fraction", f"{stats['sigma_hydro']['area_fraction']:.1%}")
                                    
                                    # Strain energy analysis - FIXED: Check if strain_energy exists and is non-empty
                                    if 'strain_energy' in report['analysis'] and report['analysis']['strain_energy']:
                                        st.markdown("#### ⚡ Strain Energy Analysis")
                                        energy = report['analysis']['strain_energy']
                                        col_energy1, col_energy2, col_energy3 = st.columns(3)
                                        with col_energy1:
                                            st.metric("Total Energy", f"{energy['total']:.3e} J/m³")
                                        with col_energy2:
                                            st.metric("Mean Energy", f"{energy['mean']:.3e} J/m³")
                                        with col_energy3:
                                            st.metric("Max Energy", f"{energy['max']:.3e} J/m³")
                                    else:
                                        st.info("Strain energy analysis unavailable (insufficient stress component data).")
                                    
                                    # Orientation analysis
                                    if 'orientation_analysis' in report['analysis']:
                                        st.markdown("#### 🧭 Orientation Analysis")
                                        orientation = report['analysis']['orientation_analysis']
                                        
                                        if 'habit_plane_proximity' in orientation:
                                            habit = orientation['habit_plane_proximity']
                                            col_orient1, col_orient2 = st.columns(2)
                                            with col_orient1:
                                                st.metric("Angle to Habit Plane", f"{habit['angle_difference']:.1f}°")
                                                st.metric("Near Habit Plane", "Yes" if habit['is_near_habit'] else "No")
                                            with col_orient2:
                                                st.metric("Orientation Factor", f"{habit['orientation_factor']:.3f}")
                                                st.metric("Max Schmid Factor", f"{habit['max_schmid_factor']:.3f}")
                                    
                                    # Store in recent analyses
                                    st.session_state.recent_analyses.append({
                                        'type': 'Real Geometry Analysis',
                                        'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        'description': f"{defect_type} at {orientation_angle}°"
                                    })
                                
                                st.success("✅ Real geometry analysis completed successfully!")
                            else:
                                st.error("No valid geometry or stress data found in selected solution.")
                        else:
                            st.warning("Selected solution has no history data.")
                else:
                    # Show information about real geometry visualization
                    st.info("👈 Configure parameters and click 'Generate Enhanced Analysis' to visualize stress in real geometry.")
                    
                    # Example visualization
                    st.markdown("""
### 📊 Real Geometry Visualization Features
This module provides comprehensive stress visualization for actual material geometry:

**Key Features:**
1. **Multi-component Stress Visualization:**
   - Hydrostatic stress (σ_h) with tensile/compressive coloring
   - Von Mises equivalent stress (σ_vm)
   - Total stress magnitude (|σ|)
   - Phase field visualization (η)

2. **Region-specific Analysis:**
   - Automatic detection of defect, interface, and bulk regions
   - Statistical comparison across regions
   - Stress concentration factor calculation

3. **Advanced Visualization Options:**
   - 50+ colormaps including custom engineered maps
   - Adjustable line thickness and font sizes
   - 3D surface plots for depth perception
   - Interactive plotly visualizations

**Physics Insights:**
- Visualize how defects create local stress concentrations
- Understand stress distribution across material interfaces
- Quantify stress amplification in defect regions
- Correlate stress fields with sintering temperature reduction
""")
            
            with tabs[2]:  # Habit Plane Vicinity
                st.markdown("## 🎯 Habit Plane Vicinity Analysis")
                st.info("Habit plane analysis module - configure parameters and generate analysis")
                # Add habit plane specific controls and visualizations here
            
            with tabs[3]:  # Defect Comparison
                st.markdown("## 🔬 Defect Type Comparison")
                st.info("Defect comparison module - configure parameters and generate analysis")
                # Add defect comparison specific controls and visualizations here
            
            with tabs[4]:  # Sintering Analysis
                st.markdown("## 📈 Enhanced Sintering Analysis")
                
                if st.session_state.get('generate_analysis', False):
                    # Show detailed physics explanations
                    st.session_state.workflow_presenter.create_physics_explanation(
                        'sintering_prediction_workflow', st
                    )
                    
                    # Prepare target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa
                    }
                    
                    # Perform interpolation for sintering prediction
                    with st.spinner("Performing physics-aware interpolation..."):
                        interpolation_result = st.session_state.interpolator.interpolate_stress_components(
                            st.session_state.solutions,
                            orientation_angle,
                            target_params,
                            region_type
                        )
                    
                    if interpolation_result:
                        # Display sintering analysis
                        sintering_analysis = interpolation_result.get('sintering_analysis', {})
                        
                        if sintering_analysis:
                            # Temperature predictions
                            st.markdown("### 🔥 Sintering Temperature Predictions")
                            temp_predictions = sintering_analysis.get('temperature_predictions', {})
                            
                            col_temp1, col_temp2, col_temp3 = st.columns(3)
                            with col_temp1:
                                st.metric(
                                    "Exponential Model",
                                    f"{temp_predictions.get('exponential_model_k', 0):.1f} K",
                                    f"{temp_predictions.get('exponential_model_c', 0):.1f} °C"
                                )
                            with col_temp2:
                                st.metric(
                                    "Arrhenius Model",
                                    f"{temp_predictions.get('arrhenius_defect_k', 0):.1f} K",
                                    f"{temp_predictions.get('arrhenius_defect_c', 0):.1f} °C"
                                )
                            with col_temp3:
                                system_info = sintering_analysis.get('system_classification', {})
                                st.metric(
                                    "System Classification",
                                    system_info.get('system', 'Unknown'),
                                    f"Predicted: {system_info.get('predicted_T_k', 0):.1f} K"
                                )
                            
                            # Activation energy analysis
                            st.markdown("### ⚡ Activation Energy Analysis")
                            activation_analysis = sintering_analysis.get('activation_energy_analysis', {})
                            
                            if activation_analysis:
                                col_act1, col_act2, col_act3 = st.columns(3)
                                with col_act1:
                                    st.metric(
                                        "Q_a (Standard)",
                                        f"{activation_analysis.get('Q_a_standard_eV', 0):.3f} eV",
                                        "Base activation energy"
                                    )
                                with col_act2:
                                    st.metric(
                                        "Q_eff (Defect)",
                                        f"{activation_analysis.get('Q_eff_defect_eV', 0):.3f} eV",
                                        f"Reduction: {activation_analysis.get('reduction_defect_eV', 0):.3f} eV"
                                    )
                                with col_act3:
                                    reduction_pct = activation_analysis.get('reduction_percentage', 0)
                                    st.metric(
                                        "Activation Energy Reduction",
                                        f"{reduction_pct:.1f}%",
                                        f"{activation_analysis.get('reduction_standard_eV', 0):.3f} eV"
                                    )
                            
                            # Create comprehensive sintering plot
                            st.markdown("### 📊 Comprehensive Sintering Analysis Plot")
                            
                            # Generate theoretical curves
                            theoretical_curves = st.session_state.sintering_calculator.get_theoretical_curve(defect_type)
                            
                            if theoretical_curves:
                                # Get stress level from analysis
                                stress_level = sintering_analysis.get('system_classification', {}).get('stress_level_GPa', 10.0)
                                
                                # Create plot
                                fig_sintering = st.session_state.sintering_calculator.create_comprehensive_sintering_plot(
                                    [stress_level],
                                    [temp_predictions.get('exponential_model_k', 550)],
                                    defect_type,
                                    f"Sintering Analysis for {defect_type}",
                                    viz_settings
                                )
                                
                                # Apply visualization settings
                                st.session_state.visualization_enhancer.apply_visualization_settings(
                                    fig=fig_sintering, settings=viz_settings
                                )
                                
                                st.pyplot(fig_sintering)
                                
                                # Add caption
                                st.markdown(
                                    st.session_state.workflow_presenter.create_caption_box(
                                        'sintering_prediction', sintering_analysis
                                    ), unsafe_allow_html=True
                                )
                            
                            # Store in recent analyses
                            st.session_state.recent_analyses.append({
                                'type': 'Sintering Analysis',
                                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'description': f"{defect_type} - {temp_predictions.get('exponential_model_k', 0):.0f}K predicted"
                            })
                else:
                    st.info("👈 Configure parameters and generate analysis to see sintering predictions")
            
            with tabs[5]:  # Workflow Analysis
                st.markdown("## 📋 Complete Workflow Analysis")
                st.info("Workflow analysis module - shows complete analysis pipeline")
                # Add workflow visualization here
        
        else:
            # No solutions loaded - show enhanced welcome screen
            st.markdown("""
## 🚀 Welcome to Enhanced Ag FCC Twin Analysis
**A comprehensive platform for stress analysis and sintering prediction with universal visualization controls.**

### 🔑 Key Features:

**1. Universal Visualization Controls**
- **50+ Colormaps:** Choose from sequential, diverging, qualitative, cyclic, and custom colormaps
- **Font Customization:** Adjust title, label, tick, and legend font sizes
- **Line & Marker Controls:** Customize line widths, marker sizes, and styles
- **Colorbar Settings:** Fine-tune colorbar width, padding, and positioning
- **Theme Options:** Switch between light/dark themes and transparent backgrounds

**2. Real Geometry Stress Visualization**
- **Multi-component Analysis:** Visualize hydrostatic, von Mises, and magnitude stresses
- **Region Detection:** Automatic identification of defect, interface, and bulk regions
- **3D Visualizations:** Interactive 3D surface plots for depth perception
- **Statistical Comparison:** Quantitative analysis across material regions

**3. Physics-Aware Interpolation**
- **Combined Attention:** ML-based similarity weighting with physics constraints
- **Habit Plane Focus:** Specialized analysis around 54.7° orientation
- **Defect-specific Models:** Custom parameters for ISF, ESF, Twin, and perfect crystals

**4. Enhanced Sintering Prediction**
- **Multiple Models:** Exponential empirical and Arrhenius physics-based predictions
- **Defect Engineering:** Quantify temperature reduction from different defect types
- **System Classification:** Map stress levels to AgNP sintering systems

**5. Comprehensive Workflow Presentation**
- **Step-by-Step Guidance:** Visual workflow with completion tracking
- **Physics Explanations:** Detailed explanations of underlying mechanisms
- **Interactive Captions:** Context-aware help for each visualization

### 🎯 Getting Started:

1. **Prepare Data:** Place your simulation files in the `numerical_solutions` directory
2. **Load Solutions:** Click the "Load Solutions" button in the sidebar
3. **Configure Analysis:** Set your parameters and visualization preferences
4. **Generate Analysis:** Click "Generate Enhanced Analysis" to begin

### 📁 Expected Data Format:
Each simulation file should contain:
- `params`: Dictionary of simulation parameters (defect_type, eps0, theta, etc.)
- `history`: List of simulation frames
- Each frame should contain:
  - `eta`: Phase field order parameter (numpy array)
  - `stresses`: Dictionary of stress components (sigma_hydro, von_mises, sigma_mag, etc.)

### 🎨 Visualization Tips:
- Use the **Universal Visualization Controls** in the sidebar to customize all plots
- Try different **colormap categories** for optimal stress visualization
- Adjust **font sizes** for better readability in publications
- Enable **dark theme** for reduced eye strain during extended analysis sessions
- Use **3D visualizations** for understanding complex stress distributions
""")
            
            # Quick start button
            if st.button("🚀 Quick Start Tutorial", type="primary", use_container_width=True):
                st.info("""
**Quick Start Tutorial:**

1. **Sample Data:** Download sample data and place in `numerical_solutions` folder
2. **Load Data:** Click "Load Solutions" in the sidebar
3. **Basic Analysis:** Select "Real Geometry Visualization" mode
4. **Visualization:** Adjust colormaps and font sizes to your preference
5. **Analysis:** Click "Generate Enhanced Analysis" to see results

For detailed tutorials, check the documentation section.
""")
    
    # Footer with enhanced information
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    with col_footer1:
        st.markdown("**🔬 Enhanced Ag FCC Twin Analysis**")
        st.markdown("Version 2.0 • Universal Visualization")
    with col_footer2:
        st.markdown("**⚙️ Features:**")
        st.markdown("• 50+ Colormaps • Physics-Aware Interpolation • Real Geometry Visualization")
    with col_footer3:
        st.markdown("**📊 Output:**")
        st.markdown("• PNG/PDF Export • Interactive 3D • Comprehensive Reports")

# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================

if __name__ == "__main__":
    main()
