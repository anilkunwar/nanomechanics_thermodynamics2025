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
from scipy.ndimage import zoom, rotate
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
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
                'physics_processed': False,
                'angle_degrees': None
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
                
                # Extract angle in degrees for easier sorting
                if 'theta' in standardized['params']:
                    theta_rad = standardized['params']['theta']
                    standardized['metadata']['angle_degrees'] = np.degrees(theta_rad)
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
# ANGULAR BRACKETING INTERPOLATOR
# =============================================
class AngularBracketingInterpolator:
    """
    Interpolator that uses angular bracketing principle:
    1. Finds the two nearest sources of the same defect type that bracket the target angle
    2. Assigns near-zero weights to all other sources
    3. Performs linear interpolation between the two bracketing sources
    """
    
    def __init__(self, bracketing_weight=0.98, other_weight=0.001):
        """
        Args:
            bracketing_weight: Total weight allocated to the two bracketing sources
            other_weight: Near-zero weight for non-bracketing sources
        """
        self.bracketing_weight = bracketing_weight
        self.other_weight = other_weight
    
    def find_bracketing_sources(self, solutions, target_angle_deg, target_defect_type):
        """
        Find the two nearest sources that bracket the target angle for the given defect type.
        
        Returns:
            (lower_source, upper_source, lower_idx, upper_idx, lower_angle, upper_angle)
        """
        # Filter solutions by defect type
        same_defect_solutions = []
        for i, sol in enumerate(solutions):
            if 'params' in sol and sol['params'].get('defect_type') == target_defect_type:
                angle_deg = sol['metadata'].get('angle_degrees')
                if angle_deg is not None:
                    same_defect_solutions.append((i, sol, angle_deg))
        
        if len(same_defect_solutions) < 2:
            st.warning(f"Need at least 2 {target_defect_type} sources for bracketing, found {len(same_defect_solutions)}")
            return None, None, None, None, None, None
        
        # Sort by angle
        same_defect_solutions.sort(key=lambda x: x[2])
        angles = [x[2] for x in same_defect_solutions]
        
        # Find bracketing indices
        lower_idx = None
        upper_idx = None
        
        # Case 1: Target angle is within range
        for i, angle in enumerate(angles):
            if angle > target_angle_deg:
                upper_idx = i
                lower_idx = i - 1 if i > 0 else None
                break
        
        # Case 2: Target angle is less than all angles
        if upper_idx is None:
            lower_idx = 0
            upper_idx = 1
        
        # Case 3: Target angle is greater than all angles
        elif lower_idx is None:
            lower_idx = len(angles) - 2
            upper_idx = len(angles) - 1
        
        # Get the sources
        lower_source_idx, lower_source, lower_angle = same_defect_solutions[lower_idx]
        upper_source_idx, upper_source, upper_angle = same_defect_solutions[upper_idx]
        
        return (lower_source, upper_source, 
                lower_source_idx, upper_source_idx,
                lower_angle, upper_angle)
    
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
        return np.zeros((100, 100))
    
    def interpolate_with_bracketing(self, solutions, target_angle_deg, target_params):
        """
        Targeted interpolation using only bracketing sources
        """
        if not solutions:
            st.warning("No solutions provided for interpolation.")
            return None
        
        try:
            target_defect_type = target_params.get('defect_type', 'Twin')
            
            # Find bracketing sources
            result = self.find_bracketing_sources(solutions, target_angle_deg, target_defect_type)
            if not result[0] or not result[1]:
                st.error("Could not find bracketing sources.")
                return None
            
            lower_source, upper_source, lower_idx, upper_idx, lower_angle, upper_angle = result
            
            # Calculate interpolation factor
            if abs(upper_angle - lower_angle) > 0:
                t = (target_angle_deg - lower_angle) / (upper_angle - lower_angle)
            else:
                t = 0.5
            
            # Calculate weights based on angular distance
            lower_dist = abs(target_angle_deg - lower_angle)
            upper_dist = abs(upper_angle - target_angle_deg)
            total_dist = lower_dist + upper_dist
            
            # Inverse distance weighting
            if total_dist > 0:
                lower_weight = (upper_dist / total_dist) * self.bracketing_weight
                upper_weight = (lower_dist / total_dist) * self.bracketing_weight
            else:
                lower_weight = upper_weight = self.bracketing_weight / 2
            
            # Create weight array
            weights = np.ones(len(solutions)) * self.other_weight  # Near-zero for others
            weights[lower_idx] = lower_weight
            weights[upper_idx] = upper_weight
            
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            
            # Extract all source fields
            source_fields = []
            source_params = []
            raw_source_fields = []  # Store original fields
            
            for i, src in enumerate(solutions):
                if 'params' not in src or 'history' not in src:
                    continue
                
                source_params.append(src['params'])
                
                # Get stress fields from last frame
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        stress_fields = last_frame['stresses'].copy()
                        
                        # Save raw fields
                        raw_source_fields.append(stress_fields.copy())
                        
                        # Extract or compute von Mises
                        if 'von_mises' in stress_fields:
                            vm = stress_fields['von_mises']
                        else:
                            vm = self.compute_von_mises(stress_fields)
                        
                        # Extract or compute hydrostatic
                        if 'sigma_hydro' in stress_fields:
                            hydro = stress_fields['sigma_hydro']
                        else:
                            hydro = self.compute_hydrostatic(stress_fields)
                        
                        # Compute magnitude
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
            
            if not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Ensure all fields have same shape
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                target_shape = shapes[0]
                for fields, raw_fields in zip(source_fields, raw_source_fields):
                    for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        if fields[key].shape != target_shape:
                            factors = [t/s for t, s in zip(target_shape, fields[key].shape)]
                            fields[key] = zoom(fields[key], factors, order=1)
                            raw_fields[key] = zoom(raw_fields[key], factors, order=1)
            
            # Apply angular interpolation
            shape = source_fields[0]['von_mises'].shape
            interpolated_fields = {}
            
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                
                # Get bracketing source fields
                lower_field = source_fields[lower_idx][component]
                upper_field = source_fields[upper_idx][component]
                
                # Linear interpolation between bracketing sources
                interpolated = (1 - t) * lower_field + t * upper_field
                
                # Add small contributions from other sources
                threshold = 0.001  # Only include sources with weight > 0.1%
                for i, fields in enumerate(source_fields):
                    if i != lower_idx and i != upper_idx:
                        if weights[i] > threshold and component in fields:
                            interpolated += weights[i] * fields[component]
                
                interpolated_fields[component] = interpolated
            
            # Calculate statistics
            stats = {}
            for component, field in interpolated_fields.items():
                stats[component] = {
                    'max': float(np.max(field)),
                    'min': float(np.min(field)),
                    'mean': float(np.mean(field)),
                    'std': float(np.std(field)),
                    'median': float(np.median(field))
                }
            
            # Calculate angular distances for all sources
            angular_distances = []
            defect_types = []
            for params in source_params:
                if 'theta' in params:
                    angle_deg = np.degrees(params['theta'])
                    dist = abs(angle_deg - target_angle_deg)
                    dist = min(dist, 360 - dist)  # Handle circular nature
                    angular_distances.append(dist)
                else:
                    angular_distances.append(180.0)
                
                defect_types.append(params.get('defect_type', 'Unknown'))
            
            # Calculate weight distribution metrics
            def calculate_entropy(weights):
                weights = np.array(weights)
                weights = weights[weights > 0]
                if len(weights) == 0:
                    return 0.0
                weights = weights / weights.sum()
                return float(-np.sum(weights * np.log(weights + 1e-10)))
            
            weight_entropy = calculate_entropy(weights)
            
            return {
                'fields': interpolated_fields,
                'raw_source_fields': raw_source_fields,
                'weights': {
                    'combined': weights.tolist(),
                    'entropy': weight_entropy,
                    'bracketing_sources': {
                        'lower': {
                            'index': lower_idx,
                            'angle': lower_angle,
                            'weight': lower_weight
                        },
                        'upper': {
                            'index': upper_idx,
                            'angle': upper_angle,
                            'weight': upper_weight
                        }
                    }
                },
                'statistics': stats,
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'source_angular_distances': angular_distances,
                'source_defect_types': defect_types,
                'source_fields': source_fields,
                'interpolation_method': 'angular_bracketing',
                'interpolation_factor': {
                    'lower_angle': float(lower_angle),
                    'upper_angle': float(upper_angle),
                    'target_angle': float(target_angle_deg),
                    'interpolation_t': float(t)
                }
            }
            
        except Exception as e:
            st.error(f"Error during angular bracketing interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None

# =============================================
# ENHANCED HEATMAP VISUALIZER WITH BRACKETING ANALYSIS
# =============================================
class HeatMapVisualizer:
    """Enhanced heat map visualizer with comparison dashboard and publication styling"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
    
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                            cmap_name='viridis', figsize=(12, 10),
                            colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                            show_stats=True, target_angle=None, defect_type=None,
                            show_colorbar=True, aspect_ratio='equal'):
        """Create enhanced heat map with chosen colormap and publication styling"""
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
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
    
    def create_comparison_dashboard(self, interpolated_fields, source_fields, source_info,
                                   target_angle, defect_type, component='von_mises',
                                   cmap_name='viridis', figsize=(20, 15),
                                   ground_truth_index=None, bracketing_info=None):
        """
        Create comprehensive comparison dashboard showing:
        1. Interpolated result
        2. Ground truth (selected source or closest match)
        3. Difference between interpolated and ground truth
        4. Weight distribution analysis
        5. Angular distribution of sources
        """
        
        fig = plt.figure(figsize=figsize)
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
                # Get angular info if available
                gt_angle = None
                if 'source_angular_distances' in source_info and ground_truth_index < len(source_info.get('source_angular_distances', [])):
                    gt_distance = source_info['source_angular_distances'][ground_truth_index]
                    gt_angle = target_angle + gt_distance if ground_truth_index < len(source_info.get('source_theta_degrees', [])) else None
                
                im2 = ax2.imshow(gt_field, cmap=cmap_name,
                                vmin=vmin, vmax=vmax, aspect='equal', interpolation='bilinear', origin='lower')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=f"{component.replace('_', ' ').title()} (GPa)")
                
                if gt_angle is not None:
                    ax2.set_title(f'Ground Truth\nθ = {gt_angle:.1f}° (Δ={gt_distance:.1f}°)',
                                 fontsize=16, fontweight='bold')
                else:
                    ax2.set_title(f'Ground Truth\nSource {ground_truth_index}',
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
        
        # 4. Weight distribution analysis (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        if 'weights' in source_info:
            weights = source_info['weights']['combined']
            x = range(len(weights))
            bars = ax4.bar(x, weights, alpha=0.7, color='steelblue', edgecolor='black')
            ax4.set_xlabel('Source Index')
            ax4.set_ylabel('Weight')
            ax4.set_title('Source Weight Distribution', fontsize=16, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Highlight bracketing sources if available
            if bracketing_info:
                lower_idx = bracketing_info.get('lower', {}).get('index')
                upper_idx = bracketing_info.get('upper', {}).get('index')
                
                if lower_idx is not None and lower_idx < len(bars):
                    bars[lower_idx].set_color('green')
                    bars[lower_idx].set_alpha(0.9)
                    bars[lower_idx].set_label('Lower Bracket')
                
                if upper_idx is not None and upper_idx < len(bars):
                    bars[upper_idx].set_color('red')
                    bars[upper_idx].set_alpha(0.9)
                    bars[upper_idx].set_label('Upper Bracket')
            
            # Highlight selected ground truth
            if ground_truth_index is not None and ground_truth_index < len(bars):
                bars[ground_truth_index].set_edgecolor('orange')
                bars[ground_truth_index].set_linewidth(3)
            
            ax4.legend()
        
        # 5. Angular distribution of sources (middle center)
        ax5 = fig.add_subplot(gs[1, 1], projection='polar')
        if 'source_angular_distances' in source_info:
            # Get angles for polar plot
            angles_rad = []
            distances = []
            
            if 'source_theta_degrees' in source_info:
                angles_rad = np.radians(source_info['source_theta_degrees'])
                distances = source_info['source_angular_distances']
            else:
                # If theta degrees not available, use angular distances
                angles_rad = np.linspace(0, 2*np.pi, len(source_info['source_angular_distances']))
                distances = source_info['source_angular_distances']
            
            # Plot sources as points with size proportional to weight
            if 'weights' in source_info:
                weights = source_info['weights']['combined']
                sizes = 100 * np.array(weights) / (np.max(weights) + 1e-8)  # Normalize sizes
            else:
                sizes = 50 * np.ones(len(angles_rad))
            
            scatter = ax5.scatter(angles_rad, distances,
                                s=sizes, alpha=0.7, c='blue', edgecolors='black')
            
            # Plot target angle
            target_rad = np.radians(target_angle)
            ax5.scatter(target_rad, 0, s=200, c='red', marker='*', edgecolors='black', label='Target')
            
            # Highlight bracketing angles if available
            if bracketing_info:
                lower_angle = bracketing_info.get('lower', {}).get('angle')
                upper_angle = bracketing_info.get('upper', {}).get('angle')
                
                if lower_angle is not None:
                    lower_rad = np.radians(lower_angle)
                    ax5.scatter(lower_rad, abs(lower_angle - target_angle), 
                               s=150, c='green', marker='^', edgecolors='black', label='Lower Bracket')
                
                if upper_angle is not None:
                    upper_rad = np.radians(upper_angle)
                    ax5.scatter(upper_rad, abs(upper_angle - target_angle), 
                               s=150, c='red', marker='v', edgecolors='black', label='Upper Bracket')
            
            ax5.set_title('Angular Distribution of Sources', fontsize=16, fontweight='bold', pad=20)
            ax5.set_theta_zero_location('N')  # 0° at top
            ax5.set_theta_direction(-1)  # Clockwise
            ax5.legend(loc='upper right', fontsize=10)
        
        # 6. Bracketing interpolation visualization (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        if bracketing_info:
            lower_angle = bracketing_info.get('lower', {}).get('angle')
            upper_angle = bracketing_info.get('upper', {}).get('angle')
            interp_t = bracketing_info.get('interpolation_t', 0.5)
            
            # Create angle line
            angles = np.array([lower_angle, target_angle, upper_angle])
            values = np.array([0, interp_t, 1])
            
            ax6.plot(angles, values, 'b-', linewidth=2, marker='o', markersize=8)
            ax6.fill_between([lower_angle, upper_angle], 0, 1, alpha=0.2, color='blue')
            
            # Add labels
            ax6.text(lower_angle, -0.05, f'{lower_angle:.1f}°', 
                    ha='center', va='top', fontsize=12, fontweight='bold')
            ax6.text(target_angle, interp_t + 0.05, f'Target: {target_angle:.1f}°', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
            ax6.text(upper_angle, -0.05, f'{upper_angle:.1f}°', 
                    ha='center', va='top', fontsize=12, fontweight='bold')
            
            # Add interpolation factor
            ax6.text(target_angle, interp_t/2, f't = {interp_t:.3f}', 
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            ax6.set_xlabel('Angle (degrees)')
            ax6.set_ylabel('Interpolation Factor (t)')
            ax6.set_title('Angular Bracketing Interpolation', fontsize=16, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(-0.1, 1.1)
        else:
            ax6.text(0.5, 0.5, 'Bracketing info\nnot available',
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax6.set_title('Bracketing Visualization', fontsize=16, fontweight='bold')
            ax6.set_axis_off()
        
        # 7. Statistics table (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.axis('off')
        
        # Prepare statistics with enhanced formatting
        stats = source_info.get('statistics', {}).get(component, {})
        if stats:
            stats_text = (
                f"{component.replace('_', ' ').title()} Statistics:\n"
                f" Max: {stats.get('max', 0):.3f} GPa\n"
                f" Min: {stats.get('min', 0):.3f} GPa\n"
                f" Mean: {stats.get('mean', 0):.3f} GPa\n"
                f" Std: {stats.get('std', 0):.3f} GPa\n"
                f" Median: {stats.get('median', 0):.3f} GPa"
            )
        else:
            stats_text = "Statistics not available"
        
        ax7.text(0.1, 0.5, stats_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax7.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='brown', linewidth=2))
        ax7.set_title('Stress Statistics', fontsize=18, fontweight='bold', pad=20)
        
        # 8. Source information (bottom center)
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        
        # Prepare source information
        if bracketing_info:
            lower_weight = bracketing_info.get('lower', {}).get('weight', 0)
            upper_weight = bracketing_info.get('upper', {}).get('weight', 0)
            
            source_text = (
                f"Bracketing Sources:\n"
                f" Lower: {bracketing_info.get('lower', {}).get('angle', 0):.1f}°\n"
                f" Upper: {bracketing_info.get('upper', {}).get('angle', 0):.1f}°\n\n"
                f"Weight Distribution:\n"
                f" Lower: {lower_weight*100:.1f}%\n"
                f" Upper: {upper_weight*100:.1f}%\n"
                f" Others: {(1 - lower_weight - upper_weight)*100:.3f}%"
            )
        else:
            source_text = "Bracketing information\nnot available"
        
        ax8.text(0.1, 0.5, source_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax8.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))
        ax8.set_title('Source Information', fontsize=18, fontweight='bold', pad=20)
        
        # 9. Interpolation method info (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        method_text = (
            f"Interpolation Method:\n"
            f" Angular Bracketing\n\n"
            f"Target Parameters:\n"
            f" Angle: {target_angle:.1f}°\n"
            f" Defect: {defect_type}\n"
            f" Grid: {interpolated_fields[component].shape[0]}×{interpolated_fields[component].shape[1]}\n\n"
            f"Sources Used: {source_info.get('num_sources', 0)}"
        )
        
        ax9.text(0.1, 0.5, method_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax9.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2))
        ax9.set_title('Method Info', fontsize=18, fontweight='bold', pad=20)
        
        plt.suptitle(f'Comprehensive Stress Field Analysis - Target θ={target_angle:.1f}°, {defect_type}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_bracketing_analysis_dashboard(self, interpolation_result, figsize=(20, 15)):
        """Create comprehensive dashboard for bracketing interpolation analysis"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        result = interpolation_result
        target_angle = result['target_angle']
        target_defect = result['target_params']['defect_type']
        
        # 1. Weight distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'weights' in result and 'combined' in result['weights']:
            weights = result['weights']['combined']
            x = range(len(weights))
            
            bars = ax1.bar(x, weights, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Source Index')
            ax1.set_ylabel('Weight')
            ax1.set_title('Targeted Weight Distribution', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Highlight bracketing sources
            if 'bracketing_sources' in result['weights']:
                bracketing = result['weights']['bracketing_sources']
                lower_idx = bracketing['lower']['index']
                upper_idx = bracketing['upper']['index']
                
                if lower_idx < len(bars):
                    bars[lower_idx].set_color('green')
                    bars[lower_idx].set_alpha(0.9)
                    bars[lower_idx].set_label('Lower Bracket')
                
                if upper_idx < len(bars):
                    bars[upper_idx].set_color('red')
                    bars[upper_idx].set_alpha(0.9)
                    bars[upper_idx].set_label('Upper Bracket')
            
            # Add weight labels for significant sources
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.01:  # Label weights > 1%
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax1.legend()
        
        # 2. Angular bracketing visualization (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'interpolation_method' in result and result['interpolation_method'] == 'angular_bracketing':
            # Get bracketing angles
            if 'interpolation_factor' in result:
                factor = result['interpolation_factor']
                lower_angle = factor['lower_angle']
                upper_angle = factor['upper_angle']
                interp_t = factor['interpolation_t']
                
                # Create angle line
                angles = np.array([lower_angle, target_angle, upper_angle])
                values = np.array([0, interp_t, 1])
                
                ax2.plot(angles, values, 'b-', linewidth=2, marker='o', markersize=8)
                ax2.fill_between([lower_angle, upper_angle], 0, 1, alpha=0.2, color='blue')
                
                # Add labels
                ax2.text(lower_angle, -0.05, f'{lower_angle:.1f}°', 
                        ha='center', va='top', fontsize=12, fontweight='bold')
                ax2.text(target_angle, interp_t + 0.05, f'Target: {target_angle:.1f}°', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
                ax2.text(upper_angle, -0.05, f'{upper_angle:.1f}°', 
                        ha='center', va='top', fontsize=12, fontweight='bold')
                
                # Add interpolation factor
                ax2.text(target_angle, interp_t/2, f't = {interp_t:.3f}', 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
                ax2.set_xlabel('Angle (degrees)')
                ax2.set_ylabel('Interpolation Factor (t)')
                ax2.set_title('Angular Bracketing Interpolation', fontsize=16, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(-0.1, 1.1)
        
        # 3. Defect type distribution (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'source_defect_types' in result:
            defect_types = result['source_defect_types']
            defect_counts = {}
            
            for defect in defect_types:
                defect_counts[defect] = defect_counts.get(defect, 0) + 1
            
            # Create pie chart
            labels = list(defect_counts.keys())
            sizes = list(defect_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            
            ax3.set_title('Defect Type Distribution in Sources', fontsize=16, fontweight='bold')
            
            # Highlight target defect
            for i, label in enumerate(labels):
                if label == target_defect:
                    wedges[i].set_edgecolor('red')
                    wedges[i].set_linewidth(3)
        
        # 4. Stress field interpolation (middle row)
        components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        titles = ['Von Mises', 'Hydrostatic', 'Magnitude']
        
        for idx, (component, title) in enumerate(zip(components, titles)):
            ax = fig.add_subplot(gs[1, idx])
            
            if component in result['fields']:
                field = result['fields'][component]
                
                im = ax.imshow(field, cmap='viridis', aspect='equal', 
                              interpolation='bilinear', origin='lower')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                ax.set_title(f'{title} Stress\nθ={target_angle:.1f}°', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.grid(True, alpha=0.2)
        
        # 5. Angular distance vs weight (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        if 'source_angular_distances' in result and 'weights' in result:
            distances = result['source_angular_distances']
            weights = result['weights']['combined']
            defect_types = result['source_defect_types']
            
            # Color by defect type
            colors = []
            for defect in defect_types:
                if defect == target_defect:
                    colors.append('green')
                else:
                    colors.append('red')
            
            scatter = ax7.scatter(distances, weights, c=colors, alpha=0.7, 
                                 s=100, edgecolors='black')
            
            ax7.set_xlabel('Angular Distance from Target (°)')
            ax7.set_ylabel('Weight')
            ax7.set_title('Weight vs Angular Distance', fontsize=16, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            target_patch = mpatches.Patch(color='green', label=f'Target Defect ({target_defect})')
            other_patch = mpatches.Patch(color='red', label='Other Defects')
            ax7.legend(handles=[target_patch, other_patch])
        
        # 6. Bracketing sources comparison (bottom center)
        ax8 = fig.add_subplot(gs[2, 1])
        if 'weights' in result and 'bracketing_sources' in result['weights']:
            bracketing = result['weights']['bracketing_sources']
            lower_info = bracketing['lower']
            upper_info = bracketing['upper']
            
            # Create bar comparison
            x = [0, 1, 2]
            labels = ['Lower Bracket', 'Upper Bracket', 'Target']
            angles = [lower_info['angle'], upper_info['angle'], target_angle]
            weights_display = [lower_info['weight'], upper_info['weight'], 1.0]
            
            bars = ax8.bar(x, weights_display, alpha=0.7, color=['green', 'red', 'blue'])
            ax8.set_xticks(x)
            ax8.set_xticklabels(labels)
            ax8.set_ylabel('Weight')
            ax8.set_title('Bracketing Sources Weight Comparison', fontsize=16, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')
            
            # Add angle labels
            for i, (bar, angle) in enumerate(zip(bars, angles)):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{angle:.1f}°', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 7. Interpolation statistics (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Calculate weight concentration metrics
        if 'weights' in result:
            weights = np.array(result['weights']['combined'])
            sorted_weights = np.sort(weights)[::-1]
            top_2_weight = sum(sorted_weights[:2])
            other_weight = sum(sorted_weights[2:])
            
            stats_text = (
                f"Weight Concentration:\n"
                f" Top 2: {top_2_weight*100:.1f}%\n"
                f" Others: {other_weight*100:.3f}%\n\n"
                f"Entropy: {result['weights']['entropy']:.3f}\n"
                f"Sources: {result['num_sources']}\n"
                f"Grid: {result['shape'][0]}×{result['shape'][1]}"
            )
        else:
            stats_text = "Statistics not available"
        
        ax9.text(0.1, 0.5, stats_text, fontsize=13, family='monospace', fontweight='bold',
                verticalalignment='center', transform=ax9.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='brown', linewidth=2))
        ax9.set_title('Interpolation Statistics', fontsize=18, fontweight='bold', pad=20)
        
        plt.suptitle(f'Angular Bracketing Interpolation Analysis - θ={target_angle:.1f}°, {target_defect}',
                    fontsize=24, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def create_angular_bracketing_visualization(self, solutions, target_angle_deg, target_defect_type, figsize=(12, 8)):
        """Visualize how sources bracket the target angle"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group sources by defect type
        defect_groups = {}
        for i, sol in enumerate(solutions):
            if 'params' in sol:
                defect = sol['params'].get('defect_type', 'Unknown')
                if defect not in defect_groups:
                    defect_groups[defect] = []
                
                if 'theta' in sol['params']:
                    angle = np.degrees(sol['params']['theta'])
                    defect_groups[defect].append((i, angle))
        
        # Create visualization
        colors = {'Twin': 'red', 'ESF': 'blue', 'ISF': 'green', 'No Defect': 'gray'}
        y_positions = {}
        y = 0
        
        for defect, sources in defect_groups.items():
            if sources:
                y_positions[defect] = y
                angles = [s[1] for s in sources]
                
                # Plot sources as points
                ax.scatter(angles, [y] * len(angles), 
                          color=colors.get(defect, 'black'),
                          s=100, alpha=0.7, edgecolors='black',
                          label=f'{defect} ({len(sources)} sources)')
                
                # Draw range line
                if len(angles) > 1:
                    min_angle = min(angles)
                    max_angle = max(angles)
                    ax.plot([min_angle, max_angle], [y, y], 
                           color=colors.get(defect, 'black'), 
                           linewidth=2, alpha=0.5)
                
                y += 1
        
        # Plot target
        ax.axvline(x=target_angle_deg, color='red', linestyle='--', 
                  linewidth=3, alpha=0.8, label=f'Target: {target_angle_deg:.1f}°')
        
        # Highlight target defect
        if target_defect_type in y_positions:
            y_target = y_positions[target_defect_type]
            ax.axhline(y=y_target, color='yellow', linestyle=':', 
                      linewidth=2, alpha=0.5, label=f'Target Defect: {target_defect_type}')
        
        ax.set_xlabel('Angle (degrees)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Defect Type', fontsize=14, fontweight='bold')
        ax.set_title('Source Distribution by Angle and Defect Type', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set y-ticks
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(list(y_positions.keys()))
        
        plt.tight_layout()
        return fig
    
    # Additional visualization methods from the original code...
    # (These methods remain unchanged from the original code)
    
    def create_interactive_3d_surface(self, stress_field, title="3D Stress Surface",
                                     cmap_name='viridis', width=900, height=700,
                                     target_angle=None, defect_type=None):
        """Create interactive 3D surface plot with Plotly"""
        try:
            # Validate colormap
            if cmap_name not in px.colors.named_colorscales():
                cmap_name = 'viridis'
            
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
                colorscale=cmap_name,
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
                                       figsize=(8, 8), special_angle=None):
        """Create polar plot showing angular orientation"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Convert target angle to radians
        theta_rad = np.radians(target_angle_deg)
        
        # Plot the defect orientation as a red arrow
        ax.arrow(theta_rad, 0.8, 0, 0.6, width=0.02,
                color='red', alpha=0.8, label=f'Defect Orientation: {target_angle_deg:.1f}°')
        
        # Plot special angle (habit plane) if provided
        if special_angle is not None:
            habit_plane_rad = np.radians(special_angle)
            ax.arrow(habit_plane_rad, 0.8, 0, 0.6, width=0.02,
                    color='blue', alpha=0.5, label=f'Special Angle: {special_angle}°')
        
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
        
        # Add annotation for angular difference from special angle if applicable
        if special_angle is not None:
            angular_diff = abs(target_angle_deg - special_angle)
            angular_diff = min(angular_diff, 360 - angular_diff)  # Handle cyclic nature
            ax.annotate(f'Δθ = {angular_diff:.1f}°\nfrom {special_angle}°',
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
        fig, axes = plt.subplots(1, n_components, figsize=figsize)
        
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
        fig = plt.figure(figsize=figsize)
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
        fig, ax = plt.subplots(figsize=figsize)
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
                'interpolation_method': result.get('interpolation_method', 'angular_bracketing'),
                'visualization_params': visualization_params
            },
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'shape': result['shape'],
                'statistics': result['statistics'],
                'weights': result['weights'],
                'num_sources': result.get('num_sources', 0),
                'source_defect_types': result.get('source_defect_types', []),
                'interpolation_factor': result.get('interpolation_factor', {}),
                'bracketing_sources': result.get('weights', {}).get('bracketing_sources', {})
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
        
        return export_data
    
    def export_to_json(self, export_data, filename=None):
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = export_data['result']['target_angle']
            defect = export_data['result']['target_params']['defect_type']
            filename = f"angular_bracketing_interpolation_theta_{theta}_{defect}_{timestamp}.json"
        
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
# MAIN APPLICATION WITH ANGULAR BRACKETING
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Angular Bracketing Stress Interpolation",
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
    .warning-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .highlight-box {
        background-color: #FCE7F3;
        border-left: 5px solid #EC4899;
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
    .angular-weighting-plot {
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 15px;
        background-color: #F8FAFC;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Angular Bracketing Stress Interpolation</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Physics-aware interpolation with Angular Bracketing Principle.</strong><br>
    • Load simulation files from numerical_solutions directory<br>
    • <strong>Angular Bracketing:</strong> Selects nearest sources of same defect type below and above target angle<br>
    • <strong>Targeted Weights:</strong> Gives ~98% weight to bracketing sources, ~0.1% to others<br>
    • <strong>Linear Interpolation:</strong> Uses interpolation factor t = (θ_target - θ_lower) / (θ_upper - θ_lower)<br>
    • Comprehensive comparison dashboard with ground truth selection<br>
    • Advanced bracketing analysis visualization<br>
    • Export results in multiple formats
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'angular_interpolator' not in st.session_state:
        # Initialize with adjustable bracketing parameters
        st.session_state.angular_interpolator = AngularBracketingInterpolator(
            bracketing_weight=0.98,  # Default: 98% weight to bracketing sources
            other_weight=0.001      # Default: 0.1% weight to others
        )
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = HeatMapVisualizer()
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = ResultsManager()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'selected_ground_truth' not in st.session_state:
        st.session_state.selected_ground_truth = None
    
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
                    
                    # Show defect distribution
                    defect_counts = {}
                    for sol in st.session_state.solutions:
                        if 'params' in sol:
                            defect = sol['params'].get('defect_type', 'Unknown')
                            defect_counts[defect] = defect_counts.get(defect, 0) + 1
                    
                    for defect, count in defect_counts.items():
                        st.info(f"{defect}: {count} sources")
                else:
                    st.warning("No solutions found in directory")
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.session_state.selected_ground_truth = None
                st.success("Cache cleared")
        
        # Show source distribution
        if st.session_state.solutions:
            if st.button("📊 Show Source Distribution", use_container_width=True):
                # Get current target parameters
                target_angle = st.session_state.get('target_angle', 54.7)
                target_defect = st.session_state.get('target_defect', 'Twin')
                
                fig_dist = st.session_state.heatmap_visualizer.create_angular_bracketing_visualization(
                    st.session_state.solutions,
                    target_angle,
                    target_defect
                )
                st.pyplot(fig_dist)
        
        st.divider()
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle
        custom_theta = st.slider(
            "Target Angle θ (degrees)",
            min_value=0.0,
            max_value=180.0,
            value=54.7,
            step=0.1,
            help="Angle in degrees (0° to 180°). Default habit plane is 54.7°"
        )
        
        # Store in session state
        st.session_state.target_angle = custom_theta
        
        # Defect type
        defect_type = st.selectbox(
            "Target Defect Type",
            options=['ISF', 'ESF', 'Twin', 'No Defect'],
            index=2,
            help="Type of crystal defect to simulate"
        )
        
        # Store in session state
        st.session_state.target_defect = defect_type
        
        # Show available angles for selected defect
        if st.session_state.solutions:
            same_defect_angles = []
            for sol in st.session_state.solutions:
                if 'params' in sol and sol['params'].get('defect_type') == defect_type:
                    if 'theta' in sol['params']:
                        angle = np.degrees(sol['params']['theta'])
                        same_defect_angles.append(angle)
            
            if same_defect_angles:
                same_defect_angles.sort()
                st.info(f"Available {defect_type} angles: {', '.join([f'{a:.1f}°' for a in same_defect_angles])}")
                
                # Find bracketing angles
                lower_angle = None
                upper_angle = None
                
                for angle in same_defect_angles:
                    if angle < custom_theta:
                        lower_angle = angle
                
                for angle in reversed(same_defect_angles):
                    if angle > custom_theta:
                        upper_angle = angle
                
                if lower_angle is not None and upper_angle is not None:
                    st.success(f"Bracketing sources: {lower_angle:.1f}° (lower) and {upper_angle:.1f}° (upper)")
                elif lower_angle is not None:
                    st.warning(f"Only lower bracket: {lower_angle:.1f}°")
                elif upper_angle is not None:
                    st.warning(f"Only upper bracket: {upper_angle:.1f}°")
                else:
                    st.error(f"No {defect_type} sources available")
        
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
        
        # Eigenstrain auto-calculation
        st.markdown("#### 🧮 Eigenstrain Calculation")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            auto_eigen = st.checkbox("Auto-calculate eigenstrain", value=True)
        with col_e2:
            if auto_eigen:
                # Auto-calculate based on defect type
                eigen_strain = {
                    'ISF': 0.289,
                    'ESF': 0.333,
                    'Twin': 0.707,
                    'No Defect': 0.0
                }[defect_type]
                st.metric("Eigenstrain ε₀", f"{eigen_strain:.3f}")
            else:
                eigen_strain = st.slider(
                    "Eigenstrain ε₀",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.707,
                    step=0.001
                )
        
        st.divider()
        
        # Angular bracketing parameters
        st.markdown('<h2 class="section-header">🎯 Angular Bracketing Parameters</h2>', unsafe_allow_html=True)
        
        # IMPORTANT TOGGLE: ANGULAR BRACKETING ENABLE/DISABLE
        enable_bracketing = st.checkbox(
            "Enable Angular Bracketing",
            value=True,
            help="If checked, uses only two nearest sources of same defect type. If unchecked, falls back to other methods."
        )
        
        # Bracketing weight parameters
        st.markdown("#### ⚖️ Weight Parameters")
        
        bracketing_weight = st.slider(
            "Bracketing Source Weight",
            min_value=0.9,
            max_value=0.999,
            value=0.98,
            step=0.001,
            help="Total weight allocated to the two bracketing sources"
        )
        
        other_weight = st.slider(
            "Other Source Weight",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f",
            help="Weight allocated to non-bracketing sources (should be very small)"
        )
        
        # Update interpolator parameters
        st.session_state.angular_interpolator.bracketing_weight = bracketing_weight
        st.session_state.angular_interpolator.other_weight = other_weight
        
        st.divider()
        
        # Run interpolation
        st.markdown("#### 🚀 Interpolation Control")
        if st.button("🎯 Perform Angular Bracketing Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing angular bracketing interpolation..."):
                    # Setup target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'eps0': eigen_strain,
                        'kappa': kappa,
                        'theta': np.radians(custom_theta),
                        'shape': shape
                    }
                    
                    # Perform interpolation with angular bracketing
                    result = st.session_state.angular_interpolator.interpolate_with_bracketing(
                        st.session_state.solutions,
                        custom_theta,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        
                        # Show success message with bracketing info
                        if 'bracketing_sources' in result.get('weights', {}):
                            bracketing = result['weights']['bracketing_sources']
                            lower = bracketing['lower']
                            upper = bracketing['upper']
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✅ Angular Bracketing Interpolation Successful!</strong><br>
                                • Used <strong>Source {lower['index']} ({lower['angle']:.1f}°)</strong> as lower bracket<br>
                                • Used <strong>Source {upper['index']} ({upper['angle']:.1f}°)</strong> as upper bracket<br>
                                • Bracketing weight: <strong>{(lower['weight'] + upper['weight'])*100:.1f}%</strong><br>
                                • Other sources weight: <strong>{(1 - lower['weight'] - upper['weight'])*100:.3f}%</strong><br>
                                • Interpolation factor t = <strong>{result.get('interpolation_factor', {}).get('interpolation_t', 0):.3f}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.success("Interpolation completed!")
                        st.session_state.selected_ground_truth = None
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
            for sol in st.session_state.solutions:
                if 'params' in sol and 'theta' in sol['params']:
                    theta_deg = np.degrees(sol['params']['theta']) % 360  # Normalize to [0, 360)
                    source_thetas.append(theta_deg)
            if source_thetas:
                st.markdown(f"**Source Angles Range:** {min(source_thetas):.1f}° to {max(source_thetas):.1f}°")
                st.markdown(f"**Mean Source Angle:** {np.mean(source_thetas):.1f}°")
    
    # Results display
    if st.session_state.interpolation_result:
        result = st.session_state.interpolation_result
        
        # Retrieve bracketing info
        bracketing_info = result.get('weights', {}).get('bracketing_sources', {})
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Results Overview",
            "🎯 Bracketing Analysis",
            "🎨 Visualization",
            "⚖️ Weight Analysis",
            "🔄 Comparison Dashboard",
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
                    f"{result['statistics']['sigma_hydro']['max']:.3f}/{result['statistics']['sigma_hydro']['min']:.3f} GPa"
                )
            with col3:
                st.metric(
                    "Mean Stress Magnitude",
                    f"{result['statistics']['sigma_mag']['mean']:.3f} GPa"
                )
            with col4:
                st.metric(
                    "Number of Sources",
                    result['num_sources'],
                    delta=f"Entropy: {result['weights']['entropy']:.3f}"
                )
            
            # Target parameters display
            st.markdown("#### 🎯 Target Parameters")
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.markdown(f"""
                <div class="param-table">
                    <div class="param-key">Angle (θ)</div>
                    <div class="param-value">{result['target_angle']:.2f}°</div>
                    <div class="param-key">Defect Type</div>
                    <div class="param-value">{result['target_params']['defect_type']}</div>
                </div>
                """, unsafe_allow_html=True)
            with param_col2:
                st.markdown(f"""
                <div class="param-table">
                    <div class="param-key">Eigenstrain (ε₀)</div>
                    <div class="param-value">{result['target_params']['eps0']:.3f}</div>
                    <div class="param-key">Kappa (κ)</div>
                    <div class="param-value">{result['target_params']['kappa']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with param_col3:
                st.markdown(f"""
                <div class="param-table">
                    <div class="param-key">Shape</div>
                    <div class="param-value">{result['target_params'].get('shape', 'Square')}</div>
                    <div class="param-key">Method</div>
                    <div class="param-value">{result.get('interpolation_method', 'angular_bracketing')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Bracketing information
            st.markdown("#### 🎯 Bracketing Information")
            if bracketing_info:
                col_br1, col_br2, col_br3 = st.columns(3)
                with col_br1:
                    lower_info = bracketing_info.get('lower', {})
                    st.metric(
                        "Lower Bracket",
                        f"{lower_info.get('angle', 0):.1f}°",
                        delta=f"Weight: {lower_info.get('weight', 0)*100:.1f}%"
                    )
                with col_br2:
                    interp_t = result.get('interpolation_factor', {}).get('interpolation_t', 0.5)
                    st.metric(
                        "Interpolation Factor (t)",
                        f"{interp_t:.3f}",
                        delta=f"{(1-interp_t)*100:.1f}% : {interp_t*100:.1f}%"
                    )
                with col_br3:
                    upper_info = bracketing_info.get('upper', {})
                    st.metric(
                        "Upper Bracket",
                        f"{upper_info.get('angle', 0):.1f}°",
                        delta=f"Weight: {upper_info.get('weight', 0)*100:.1f}%"
                    )
            
            # Quick preview of stress fields
            st.markdown("#### 👀 Quick Preview")
            preview_component = st.selectbox(
                "Preview Component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0,
                key="preview_component"
            )
            
            if preview_component in result['fields']:
                fig_preview = st.session_state.heatmap_visualizer.create_stress_heatmap(
                    result['fields'][preview_component],
                    title=f"{preview_component.replace('_', ' ').title()} Stress",
                    cmap_name='viridis',
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(10, 8)
                )
                st.pyplot(fig_preview)
        
        with tab2:
            # Bracketing analysis
            st.markdown('<h2 class="section-header">🎯 Angular Bracketing Analysis</h2>', unsafe_allow_html=True)
            
            # Create comprehensive bracketing analysis dashboard
            fig_analysis = st.session_state.heatmap_visualizer.create_bracketing_analysis_dashboard(result)
            st.pyplot(fig_analysis)
            
            # Additional analysis
            st.markdown("#### 📊 Detailed Bracketing Analysis")
            
            if bracketing_info:
                lower_info = bracketing_info.get('lower', {})
                upper_info = bracketing_info.get('upper', {})
                
                # Create comparison table
                st.markdown("##### 📋 Bracketing Sources Comparison")
                df_bracketing = pd.DataFrame([
                    {
                        'Parameter': 'Angle',
                        'Lower Bracket': f"{lower_info.get('angle', 0):.1f}°",
                        'Upper Bracket': f"{upper_info.get('angle', 0):.1f}°",
                        'Target': f"{result['target_angle']:.1f}°"
                    },
                    {
                        'Parameter': 'Weight',
                        'Lower Bracket': f"{lower_info.get('weight', 0)*100:.2f}%",
                        'Upper Bracket': f"{upper_info.get('weight', 0)*100:.2f}%",
                        'Target': '100.00%'
                    },
                    {
                        'Parameter': 'Δθ from Target',
                        'Lower Bracket': f"{abs(result['target_angle'] - lower_info.get('angle', 0)):.1f}°",
                        'Upper Bracket': f"{abs(upper_info.get('angle', 0) - result['target_angle']):.1f}°",
                        'Target': '0.0°'
                    },
                    {
                        'Parameter': 'Source Index',
                        'Lower Bracket': str(lower_info.get('index', 'N/A')),
                        'Upper Bracket': str(upper_info.get('index', 'N/A')),
                        'Target': 'N/A'
                    }
                ])
                
                st.dataframe(df_bracketing, use_container_width=True)
                
                # Show interpolation factor details
                st.markdown("##### 🧮 Interpolation Factor Details")
                interp_t = result.get('interpolation_factor', {}).get('interpolation_t', 0.5)
                col_int1, col_int2, col_int3 = st.columns(3)
                with col_int1:
                    st.metric("t = (θ_target - θ_lower) / (θ_upper - θ_lower)", f"{interp_t:.4f}")
                with col_int2:
                    st.metric("Lower Contribution", f"{(1-interp_t)*100:.1f}%")
                with col_int3:
                    st.metric("Upper Contribution", f"{interp_t*100:.1f}%")
        
        with tab3:
            # Visualization tab
            st.markdown('<h2 class="section-header">🎨 Advanced Visualization</h2>', unsafe_allow_html=True)
            
            # Visualization controls
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            with col_viz1:
                component = st.selectbox(
                    "Stress Component",
                    options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0,
                    key="viz_component"
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
            if cmap_name:
                col_preview1, col_preview2, col_preview3 = st.columns([1, 2, 1])
                with col_preview2:
                    fig_cmap = st.session_state.heatmap_visualizer.get_colormap_preview(cmap_name)
                    st.pyplot(fig_cmap)
            
            # Visualization type selection
            viz_type = st.radio(
                "Visualization Type",
                options=["2D Heatmap", "3D Surface", "Interactive Heatmap", "Interactive 3D", "Angular Orientation"],
                horizontal=True
            )
            
            if component in result['fields']:
                stress_field = result['fields'][component]
                
                if viz_type == "2D Heatmap":
                    # 2D heatmap
                    fig_2d = st.session_state.heatmap_visualizer.create_stress_heatmap(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(12, 10)
                    )
                    st.pyplot(fig_2d)
                    
                    # Show statistics
                    with st.expander("📊 Detailed Statistics", expanded=False):
                        stats = result['statistics'][component]
                        for key, value in stats.items():
                            st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                
                elif viz_type == "3D Surface":
                    # 3D surface plot
                    fig_3d = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(14, 10)
                    )
                    st.pyplot(fig_3d)
                
                elif viz_type == "Interactive Heatmap":
                    # Interactive heatmap
                    fig_interactive = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        width=800,
                        height=700
                    )
                    st.plotly_chart(fig_interactive, use_container_width=True)
                
                elif viz_type == "Interactive 3D":
                    # Interactive 3D surface
                    fig_3d_interactive = st.session_state.heatmap_visualizer.create_interactive_3d_surface(
                        stress_field,
                        title=f"{component.replace('_', ' ').title()} Stress",
                        cmap_name=cmap_name,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        width=900,
                        height=700
                    )
                    st.plotly_chart(fig_3d_interactive, use_container_width=True)
                
                elif viz_type == "Angular Orientation":
                    # Angular orientation plot
                    fig_angular = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                        result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        figsize=(10, 10)
                    )
                    st.pyplot(fig_angular)
            
            # Comparison of all components
            st.markdown("#### 🔄 Component Comparison")
            if st.button("Show All Components Comparison", key="show_all_components"):
                fig_all = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                    result['fields'],
                    cmap_name=cmap_name,
                    target_angle=result['target_angle'],
                    defect_type=result['target_params']['defect_type'],
                    figsize=(18, 6)
                )
                st.pyplot(fig_all)
        
        with tab4:
            # Weight analysis tab
            st.markdown('<h2 class="section-header">⚖️ Weight Distribution Analysis</h2>', unsafe_allow_html=True)
            
            if 'weights' in result:
                weights = result['weights']['combined']
                
                # Weight statistics
                col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                with col_w1:
                    st.metric("Weight Entropy", f"{result['weights']['entropy']:.3f}")
                with col_w2:
                    sorted_weights = np.sort(weights)[::-1]
                    top_2_weight = sum(sorted_weights[:2])
                    st.metric("Top 2 Sources Weight", f"{top_2_weight*100:.1f}%")
                with col_w3:
                    other_weight = 1 - top_2_weight
                    st.metric("Other Sources Weight", f"{other_weight*100:.3f}%")
                with col_w4:
                    non_zero_sources = sum(1 for w in weights if w > 0.001)
                    st.metric("Significant Sources", non_zero_sources)
                
                # Weight distribution plot
                st.markdown("#### 📊 Source Weight Distribution")
                fig_weights, ax_weights = plt.subplots(figsize=(14, 6))
                x = range(len(weights))
                
                # Plot weights
                bars = ax_weights.bar(x, weights, alpha=0.7, color='steelblue', edgecolor='black')
                ax_weights.set_xlabel('Source Index')
                ax_weights.set_ylabel('Weight')
                ax_weights.set_title('Weight Distribution Across Sources', fontsize=16, fontweight='bold')
                ax_weights.grid(True, alpha=0.3, axis='y')
                
                # Highlight bracketing sources
                if bracketing_info:
                    lower_idx = bracketing_info.get('lower', {}).get('index')
                    upper_idx = bracketing_info.get('upper', {}).get('index')
                    
                    if lower_idx is not None and lower_idx < len(bars):
                        bars[lower_idx].set_color('green')
                        bars[lower_idx].set_alpha(0.9)
                        bars[lower_idx].set_label('Lower Bracket')
                    
                    if upper_idx is not None and upper_idx < len(bars):
                        bars[upper_idx].set_color('red')
                        bars[upper_idx].set_alpha(0.9)
                        bars[upper_idx].set_label('Upper Bracket')
                
                # Add weight labels for significant sources
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0.01:  # Label weights > 1%
                        ax_weights.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                                      f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax_weights.legend()
                st.pyplot(fig_weights)
                
                # Top contributors table
                st.markdown("#### 🏆 Top Contributors Analysis")
                weight_data = []
                for i in range(len(weights)):
                    angle_dist = result['source_angular_distances'][i] if i < len(result['source_angular_distances']) else 0.0
                    defect_type = result['source_defect_types'][i] if i < len(result['source_defect_types']) else 'Unknown'
                    
                    weight_data.append({
                        'Source': i,
                        'Weight': weights[i],
                        'Angular Distance (°)': angle_dist,
                        'Defect Type': defect_type,
                        'Contribution': f"{weights[i]*100:.2f}%",
                        'Is Bracket': '✓' if (i == bracketing_info.get('lower', {}).get('index') or 
                                             i == bracketing_info.get('upper', {}).get('index')) else ''
                    })
                
                df_weights = pd.DataFrame(weight_data)
                df_weights = df_weights.sort_values('Weight', ascending=False)
                
                # Display top contributors
                st.dataframe(df_weights.head(10).style.format({
                    'Weight': '{:.6f}',
                    'Angular Distance (°)': '{:.1f}'
                }).background_gradient(subset=['Weight'], cmap='YlOrRd'))
        
        with tab5:
            # COMPARISON DASHBOARD
            st.markdown('<h2 class="section-header">🔄 Comparison Dashboard</h2>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            <strong>Compare interpolated results with ground truth sources</strong><br>
            • Select a source solution as ground truth<br>
            • Visualize differences between interpolation and ground truth<br>
            • Calculate error metrics (MSE, MAE, RMSE, correlation)<br>
            • Analyze spatial correlation patterns<br>
            • Current Method: Angular Bracketing<br>
            • <strong>Bracketing Sources:</strong> Shows which sources were used as lower/upper brackets
            </div>
            """, unsafe_allow_html=True)
            
            # Ground truth selection
            st.markdown("#### 🎯 Select Ground Truth Source")
            if 'source_defect_types' in result and result['source_defect_types']:
                # Create dropdown options
                ground_truth_options = []
                for i in range(len(result['source_defect_types'])):
                    angle_dist = result['source_angular_distances'][i] if i < len(result['source_angular_distances']) else 0.0
                    defect = result['source_defect_types'][i] if i < len(result['source_defect_types']) else 'Unknown'
                    weight = result['weights']['combined'][i] if i < len(result['weights']['combined']) else 0.0
                    
                    # Determine if this is a bracketing source
                    bracket_type = ""
                    if i == bracketing_info.get('lower', {}).get('index'):
                        bracket_type = " [Lower Bracket]"
                    elif i == bracketing_info.get('upper', {}).get('index'):
                        bracket_type = " [Upper Bracket]"
                    
                    ground_truth_options.append(
                        f"Source {i}{bracket_type}: {defect}, Δ={angle_dist:.1f}°, weight={weight:.3f}"
                    )
                
                selected_option = st.selectbox(
                    "Choose ground truth source:",
                    options=ground_truth_options,
                    index=0 if not st.session_state.selected_ground_truth else st.session_state.selected_ground_truth,
                    key="ground_truth_select"
                )
                
                # Parse selected index
                selected_index = int(selected_option.split(":")[0].split(" ")[1])
                st.session_state.selected_ground_truth = selected_index
                
                # Display selected source info
                selected_weight = result['weights']['combined'][selected_index]
                selected_distance = result['source_angular_distances'][selected_index] if selected_index < len(result['source_angular_distances']) else 0.0
                selected_defect = result['source_defect_types'][selected_index] if selected_index < len(result['source_defect_types']) else 'Unknown'
                
                col_gt1, col_gt2, col_gt3, col_gt4 = st.columns(4)
                with col_gt1:
                    st.metric("Selected Source", selected_index)
                with col_gt2:
                    st.metric("Angular Distance", f"{selected_distance:.1f}°")
                with col_gt3:
                    st.metric("Defect Type", selected_defect)
                with col_gt4:
                    st.metric("Contribution Weight", f"{selected_weight:.3f}")
            
            # Visualization options for comparison
            st.markdown("#### 🎨 Comparison Visualization")
            comp_component = st.selectbox(
                "Component for Comparison",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0,
                key="comp_component"
            )
            
            comp_cmap = st.selectbox(
                "Colormap for Comparison",
                options=COLORMAP_OPTIONS['Publication Standard'],
                index=0,
                key="comp_cmap"
            )
            
            # Create comparison dashboard
            if comp_component in result['fields']:
                # Prepare source info for dashboard
                source_info = {
                    'theta_degrees': result.get('source_theta_degrees', []),
                    'source_angular_distances': result['source_angular_distances'],
                    'weights': result['weights'],
                    'statistics': result['statistics'],
                    'num_sources': result['num_sources']
                }
                
                # Get raw source fields
                raw_source_fields_list = result.get('raw_source_fields', result.get('source_fields', []))
                
                if len(raw_source_fields_list) > 0:
                    # Get bracketing info for visualization
                    bracketing_viz_info = {}
                    if bracketing_info:
                        bracketing_viz_info = {
                            'lower': bracketing_info.get('lower', {}),
                            'upper': bracketing_info.get('upper', {}),
                            'interpolation_t': result.get('interpolation_factor', {}).get('interpolation_t', 0.5)
                        }
                    
                    fig_comparison = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                        interpolated_fields=result['fields'],
                        source_fields=raw_source_fields_list,
                        source_info=source_info,
                        target_angle=result['target_angle'],
                        defect_type=result['target_params']['defect_type'],
                        component=comp_component,
                        cmap_name=comp_cmap,
                        figsize=(20, 15),
                        ground_truth_index=selected_index,
                        bracketing_info=bracketing_viz_info
                    )
                    st.pyplot(fig_comparison)
                    
                    # Calculate and display detailed error metrics
                    if selected_index < len(raw_source_fields_list):
                        ground_truth_field = raw_source_fields_list[selected_index].get(comp_component)
                        interpolated_field = result['fields'][comp_component]
                        
                        if ground_truth_field is not None:
                            # Calculate errors
                            error_field = interpolated_field - ground_truth_field
                            mse = np.mean(error_field**2)
                            mae = np.mean(np.abs(error_field))
                            rmse = np.sqrt(mse)
                            
                            # Calculate correlation
                            try:
                                corr_coef = np.corrcoef(ground_truth_field.flatten(), interpolated_field.flatten())[0, 1]
                            except:
                                corr_coef = 0.0
                            
                            # Display metrics
                            st.markdown("#### 📊 Error Metrics")
                            err_col1, err_col2, err_col3, err_col4 = st.columns(4)
                            with err_col1:
                                st.metric("Mean Squared Error (MSE)", f"{mse:.6f}")
                            with err_col2:
                                st.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
                            with err_col3:
                                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.6f}")
                            with err_col4:
                                st.metric("Pearson Correlation", f"{corr_coef:.4f}")
            else:
                st.warning("No source information available for comparison.")
        
        with tab6:
            # Export tab
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-box">
            <strong>Export interpolation results for further analysis</strong><br>
            • Export full results as JSON with metadata<br>
            • Export stress field data as CSV for external analysis<br>
            • Download visualizations as PNG images<br>
            • Save comparison dashboard for publication
            </div>
            """, unsafe_allow_html=True)
            
            # Export options
            export_format = st.radio(
                "Export Format",
                options=["JSON (Full Results)", "CSV (Field Data)", "PNG (Visualizations)"],
                horizontal=True
            )
            
            if export_format == "JSON (Full Results)":
                # JSON export
                visualization_params = {
                    'component': component if 'component' in locals() else 'von_mises',
                    'colormap': cmap_name if 'cmap_name' in locals() else 'viridis',
                    'visualization_type': viz_type if 'viz_type' in locals() else '2D Heatmap'
                }
                
                export_data = st.session_state.results_manager.prepare_export_data(
                    result, visualization_params
                )
                
                json_str, json_filename = st.session_state.results_manager.export_to_json(export_data)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=json_filename,
                    mime="application/json",
                    use_container_width=True
                )
                
                # Show preview
                with st.expander("🔍 JSON Preview", expanded=False):
                    st.json(export_data)
            
            elif export_format == "CSV (Field Data)":
                # CSV export
                csv_str, csv_filename = st.session_state.results_manager.export_to_csv(result)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_str,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Show preview
                with st.expander("🔍 CSV Preview", expanded=False):
                    # Create a sample of the data
                    sample_data = {}
                    for field_name, field_data in result['fields'].items():
                        sample_data[field_name] = field_data.flatten()[:100]  # First 100 values
                    
                    df_sample = pd.DataFrame(sample_data)
                    st.dataframe(df_sample.head(10))
            
            elif export_format == "PNG (Visualizations)":
                # PNG export options
                st.markdown("#### 📸 Select Visualizations to Export")
                export_plots = st.multiselect(
                    "Choose plots to export:",
                    options=[
                        "Von Mises Heatmap",
                        "Hydrostatic Heatmap",
                        "Stress Magnitude Heatmap",
                        "3D Surface Plot",
                        "Angular Orientation",
                        "Bracketing Analysis Dashboard",
                        "Weight Distribution",
                        "Comparison Dashboard"
                    ],
                    default=["Von Mises Heatmap", "Bracketing Analysis Dashboard", "Comparison Dashboard"]
                )
                
                if st.button("🖼️ Generate and Download Visualizations", use_container_width=True):
                    # Create a zip file with all selected plots
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Generate and save each selected plot
                        plot_count = 0
                        
                        if "Von Mises Heatmap" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                result['fields']['von_mises'],
                                title="Von Mises Stress",
                                cmap_name='viridis',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"von_mises_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Hydrostatic Heatmap" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                result['fields']['sigma_hydro'],
                                title="Hydrostatic Stress",
                                cmap_name='RdBu_r',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"hydrostatic_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Stress Magnitude Heatmap" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                result['fields']['sigma_mag'],
                                title="Stress Magnitude",
                                cmap_name='plasma',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"stress_magnitude_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "3D Surface Plot" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_3d_surface_plot(
                                result['fields']['von_mises'],
                                title="3D Von Mises Stress",
                                cmap_name='viridis',
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"3d_surface_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Angular Orientation" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_angular_orientation_plot(
                                result['target_angle'],
                                defect_type=result['target_params']['defect_type']
                            )
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"angular_orientation_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Bracketing Analysis Dashboard" in export_plots:
                            fig = st.session_state.heatmap_visualizer.create_bracketing_analysis_dashboard(result)
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"bracketing_analysis_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Weight Distribution" in export_plots:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            weights = result['weights']['combined']
                            x = range(len(weights))
                            bars = ax.bar(x, weights, alpha=0.7, color='steelblue', edgecolor='black')
                            ax.set_xlabel('Source Index')
                            ax.set_ylabel('Weight')
                            ax.set_title('Weight Distribution', fontsize=16, fontweight='bold')
                            ax.grid(True, alpha=0.3, axis='y')
                            
                            # Highlight bracketing sources
                            if bracketing_info:
                                lower_idx = bracketing_info.get('lower', {}).get('index')
                                upper_idx = bracketing_info.get('upper', {}).get('index')
                                
                                if lower_idx is not None and lower_idx < len(bars):
                                    bars[lower_idx].set_color('green')
                                    bars[lower_idx].set_alpha(0.9)
                                    bars[lower_idx].set_label('Lower Bracket')
                                
                                if upper_idx is not None and upper_idx < len(bars):
                                    bars[upper_idx].set_color('red')
                                    bars[upper_idx].set_alpha(0.9)
                                    bars[upper_idx].set_label('Upper Bracket')
                            
                            ax.legend()
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"weight_distribution_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                        
                        if "Comparison Dashboard" in export_plots and st.session_state.selected_ground_truth is not None:
                            source_info = {
                                'source_angular_distances': result['source_angular_distances'],
                                'weights': result['weights'],
                                'statistics': result['statistics'],
                                'num_sources': result['num_sources']
                            }
                            
                            bracketing_viz_info = {}
                            if bracketing_info:
                                bracketing_viz_info = {
                                    'lower': bracketing_info.get('lower', {}),
                                    'upper': bracketing_info.get('upper', {}),
                                    'interpolation_t': result.get('interpolation_factor', {}).get('interpolation_t', 0.5)
                                }
                            
                            fig = st.session_state.heatmap_visualizer.create_comparison_dashboard(
                                interpolated_fields=result['fields'],
                                source_fields=result.get('raw_source_fields', result.get('source_fields', [])),
                                source_info=source_info,
                                target_angle=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                component='von_mises',
                                cmap_name='viridis',
                                ground_truth_index=st.session_state.selected_ground_truth,
                                bracketing_info=bracketing_viz_info
                            )
                            
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                            zip_file.writestr(f"comparison_dashboard_theta_{result['target_angle']:.1f}.png", buf.getvalue())
                            plot_count += 1
                            plt.close(fig)
                    
                    zip_buffer.seek(0)
                    
                    # Download button for zip file
                    st.download_button(
                        label=f"📦 Download {plot_count} Visualization(s) as ZIP",
                        data=zip_buffer,
                        file_name=f"visualizations_theta_{result['target_angle']:.1f}_{result['target_params']['defect_type']}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    st.success(f"Generated {plot_count} visualization(s) for download.")
    
    else:
        # No results yet - show instructions
        st.markdown("""
        <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); border-radius: 20px; color: white;">
            <h2>🎯 Ready to Begin!</h2>
            <p style="font-size: 1.2rem; margin-bottom: 30px;">
                Follow these steps to start interpolating stress fields with angular bracketing:
            </p>
            <ol style="text-align: left; display: inline-block; font-size: 1.1rem;">
                <li>Load simulation files from the sidebar</li>
                <li>Configure target parameters (angle, defect type, etc.)</li>
                <li><strong>Feature:</strong> System automatically finds bracketing sources of same defect type</li>
                <li><strong>Feature:</strong> Adjust bracketing weight parameters (98% to bracketing, 0.1% to others)</li>
                <li>Click "Perform Angular Bracketing Interpolation"</li>
                <li>Explore results in the tabs above</li>
            </ol>
            <p style="margin-top: 30px; font-size: 1.1rem;">
                <strong>Angular Bracketing Principle:</strong>
                <ul style="text-align: left; display: inline-block;">
                    <li><strong>Source Selection:</strong> Finds nearest sources below and above target angle with same defect type</li>
                    <li><strong>Weight Assignment:</strong> Gives ~98% weight to bracketing sources, ~0.1% to others</li>
                    <li><strong>Linear Interpolation:</strong> Uses t = (θ_target - θ_lower) / (θ_upper - θ_lower)</li>
                    <li><strong>Physics-based:</strong> Maintains defect type consistency for accurate interpolation</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Theory explanation
        with st.expander("📚 Theoretical Basis for Angular Bracketing", expanded=True):
            st.markdown("""
            ### **Angular Bracketing: Mathematical Formulation**
            
            **1. Source Selection:**
            For target angle θₜ and defect type Dₜ:
            - Find Sₗ = argminₛ |θₛ - θₜ| where θₛ < θₜ and defect_type(Sₛ) = Dₜ
            - Find Sᵤ = argminₛ |θₛ - θₜ| where θₛ > θₜ and defect_type(Sₛ) = Dₜ
            
            **2. Weight Assignment:**
            - wₗ = α × (θᵤ - θₜ) / (θᵤ - θₗ)  where α = bracketing_weight (e.g., 0.98)
            - wᵤ = α × (θₜ - θₗ) / (θᵤ - θₗ)
            - wᵢ = (1 - α) / (N - 2) for all other sources i
            
            **3. Interpolation:**
            σ(θₜ) = wₗ × σ(θₗ) + wᵤ × σ(θᵤ) + Σᵢ wᵢ × σ(θᵢ)
            
            Where Σ w = 1 and wₗ + wᵤ ≈ α ≈ 0.98
            
            **4. Special Cases:**
            - If only one bracketing source found, use it with weight α
            - If no same-defect sources found, fall back to nearest sources regardless of defect type
            - If target angle outside source range, use nearest two sources
            
            **5. Advantages:**
            - Physically intuitive (linear interpolation between nearest angles)
            - Consistent with crystallographic symmetry
            - Minimizes influence of dissimilar defect types
            - Provides smooth interpolation with clear bracketing
            - Computationally efficient (uses only 2 main sources)
            """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
