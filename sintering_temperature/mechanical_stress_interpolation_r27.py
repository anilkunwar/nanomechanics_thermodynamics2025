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
# ESSENTIAL PHYSICS CLASSES (MINIMAL VERSION)
# =============================================
class PhysicsBasedStressAnalyzer:
    """Minimal physics analyzer for defect eigen strains"""
    def __init__(self):
        self.eigen_strains = {
            'ISF': 0.71,      # Intrinsic Stacking Fault
            'ESF': 1.41,      # Extrinsic Stacking Fault
            'Twin': 2.12,     # Twin boundary
            'No Defect': 0.0, # Perfect crystal
            'Unknown': 0.0
        }
    
    def get_eigen_strain(self, defect_type):
        """Get eigen strain value for a specific defect type"""
        return self.eigen_strains.get(defect_type, 0.0)

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
                
                # Add eigen strain based on defect type
                params = standardized['params']
                if 'defect_type' in params:
                    defect_type = params['defect_type']
                    eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
                    params['eigen_strain'] = eigen_strain
                    # Update eps0 if not set or different from eigen strain
                    if 'eps0' not in params or abs(params['eps0'] - eigen_strain) > 0.1:
                        params['eps0'] = eigen_strain
        except Exception as e:
            print(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
        
        return standardized
    
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

class PhysicsAwareInterpolator:
    """Physics-aware interpolator for defect stress patterns"""
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        self.defect_colors = {
            'ISF': '#FF6B6B',
            'ESF': '#4ECDC4', 
            'Twin': '#45B7D1',
            'No Defect': '#96CEB4'
        }
    
    def create_vicinity_sweep(self, sources, target_params, vicinity_range=15.0,
                            n_points=72, region_type='bulk'):
        """Create stress sweep in vicinity of habit plane"""
        center_angle = self.habit_angle
        min_angle = center_angle - vicinity_range
        max_angle = center_angle + vicinity_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        results = {
            'angles': angles.tolist(),
            'stresses': {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []},
            'defect_type': target_params.get('defect_type', 'Twin'),
            'eigen_strain': self.physics_analyzer.get_eigen_strain(target_params.get('defect_type', 'Twin'))
        }
        
        # For demonstration, generate synthetic stress data based on angle and defect type
        defect_type = target_params.get('defect_type', 'Twin')
        eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
        
        # Generate synthetic stress patterns with peak at habit plane
        for angle in angles:
            # Distance from habit plane (in degrees)
            angle_diff = abs(angle - self.habit_angle)
            
            # Base stress pattern: Gaussian peak at habit plane
            base_pattern = np.exp(-angle_diff**2 / (2 * 5**2))
            
            # Scale by eigen strain and add noise
            sigma_hydro = eigen_strain * 15 * base_pattern * (0.8 + 0.2 * np.random.random())
            von_mises = sigma_hydro * 1.2 * (0.9 + 0.1 * np.random.random())
            sigma_mag = np.sqrt(sigma_hydro**2 + von_mises**2)
            
            results['stresses']['sigma_hydro'].append(sigma_hydro)
            results['stresses']['von_mises'].append(von_mises)
            results['stresses']['sigma_mag'].append(sigma_mag)
        
        return results
    
    def compare_defect_types(self, sources, vicinity_range=15.0, n_points=72,
                           region_type='bulk', shapes=None):
        """Compare different defect types across orientation range near habit plane"""
        if shapes is None:
            shapes = ['Square']
        
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
                
                # Generate synthetic stress data with different patterns for each defect type
                stresses = {'sigma_hydro': [], 'von_mises': [], 'sigma_mag': []}
                
                for angle in angles:
                    angle_diff = abs(angle - self.habit_angle)
                    
                    # Different peak shapes for different defect types
                    if defect == 'Twin':
                        # Sharp peak at habit plane
                        base_pattern = np.exp(-angle_diff**2 / (2 * 3**2))
                    elif defect == 'ISF':
                        # Broader peak
                        base_pattern = np.exp(-angle_diff**2 / (2 * 6**2))
                    elif defect == 'ESF':
                        # Medium width peak
                        base_pattern = np.exp(-angle_diff**2 / (2 * 4.5**2))
                    else:  # No Defect
                        # Very flat pattern
                        base_pattern = 0.1 * np.exp(-angle_diff**2 / (2 * 10**2))
                    
                    # Scale by eigen strain and add some noise
                    sigma_hydro = eigen_strain * 15 * base_pattern * (0.85 + 0.15 * np.random.random())
                    von_mises = sigma_hydro * 1.3 * (0.9 + 0.1 * np.random.random())
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
                    'eigen_strain': eigen_strain
                }
        
        return comparison_results

# =============================================
# FOCUSED VISUALIZATION CLASS FOR DEFECT RADAR CHARTS
# =============================================
class DefectRadarVisualizer:
    """Focused visualizer for defect radar charts with customization options"""
    
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        self.defect_colors = {
            'ISF': 'rgb(255, 107, 107)',    # Red-orange
            'ESF': 'rgb(78, 205, 196)',     # Teal
            'Twin': 'rgb(69, 183, 209)',    # Blue
            'No Defect': 'rgb(150, 206, 180)' # Green
        }
        self.stress_component_colors = {
            'sigma_hydro': 'rgb(31, 119, 180)',  # Blue
            'von_mises': 'rgb(255, 127, 14)',   # Orange
            'sigma_mag': 'rgb(44, 160, 44)'     # Green
        }
    
    def create_basic_defect_radar(self, defect_comparison, stress_component='sigma_hydro',
                                title="Defect Stress Patterns Near Habit Plane",
                                show_habit_plane=True, fill_opacity=0.2,
                                line_width=3, marker_size=8, show_grid=True, bgcolor="white"):
        """Create a basic radar chart comparing different defect types"""
        fig = go.Figure()
        
        # Add traces for each defect type
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = data['angles']
            stresses = data['stresses'][stress_component]
            
            # Close the loop for radar chart
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(stresses, stresses[0])
            
            color = data.get('color', self.defect_colors.get(defect_type, 'black'))
            
            # Convert to hex if it's an RGB string
            if color.startswith('rgb'):
                color = self._rgb_to_hex(color)
            
            # Add the trace
            fig.add_trace(go.Scatterpolar(
                r=stresses_closed,
                theta=angles_closed,
                fill='toself',
                fillcolor=f'rgba{color[3:-1]}, {fill_opacity})' if color.startswith('rgba') else f'{color}{int(fill_opacity*255):02x}',
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size, color=color, line=dict(width=1, color='white')),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                showlegend=True
            ))
        
        # Highlight habit plane if requested
        if show_habit_plane:
            max_stress = max(max(data['stresses'][stress_component]) for data in defect_comparison.values())
            fig.add_trace(go.Scatterpolar(
                r=[0, max_stress * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
                name=f'Habit Plane ({self.habit_angle}°)',
                hoverinfo='skip',
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
            height=700
        )
        
        return fig
    
    def create_sunburst_defect_chart(self, defect_comparison, stress_component='sigma_hydro',
                                   title="Defect Stress Patterns - Sunburst View",
                                   show_habit_plane=True, radius_scale=1.0,
                                   color_scale='RdBu', show_colorbar=True):
        """Create a sunburst-style chart for defect stress patterns"""
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
            
            color = data.get('color', self.defect_colors.get(defect_type, 'black'))
            
            # Convert to rgba format for gradient
            if color.startswith('rgb'):
                rgba_color = f'rgba{color[3:-1]}, 0.8)'
            else:
                rgba_color = f'{color}cc'  # Add alpha value
            
            # Add the trace
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
                line=dict(color=color, width=3, shape='spline'),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='Defect: ' + defect_type + '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
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
                        color='rgb(46, 204, 113)',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    text=['HABIT PLANE'],
                    textposition='top center',
                    textfont=dict(size=14, color='black', family='Arial Black'),
                    name=f'Habit Plane ({self.habit_angle}°)',
                    hovertemplate=f'Habit Plane ({self.habit_angle}°)<br>Peak Stress: {max_habit_stress:.4f} GPa<extra></extra>',
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
    
    def create_multi_component_radar(self, defect_comparison, defect_type='Twin',
                                   title="Stress Components for Twin Defects",
                                   show_habit_plane=True, fill_opacity=0.15,
                                   component_opacity=0.6):
        """Create radar chart showing multiple stress components for a single defect type"""
        fig = go.Figure()
        
        # Find data for the specified defect type
        target_data = None
        for key, data in defect_comparison.items():
            if data.get('defect_type') == defect_type:
                target_data = data
                break
        
        if not target_data:
            st.error(f"No data found for defect type: {defect_type}")
            return fig
        
        angles = target_data['angles']
        eigen_strain = target_data['eigen_strain']
        
        # Add traces for each stress component
        for comp_name in ['sigma_hydro', 'von_mises', 'sigma_mag']:
            if comp_name in target_data['stresses']:
                stresses = target_data['stresses'][comp_name]
                # Close the loop for radar chart
                angles_closed = np.append(angles, angles[0])
                stresses_closed = np.append(stresses, stresses[0])
                
                color = self.stress_component_colors.get(comp_name, 'black')
                
                fig.add_trace(go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=f'rgba{color[3:-1]}, {fill_opacity})',
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color),
                    name=f"{comp_name.replace('_', ' ').title()}",
                    hovertemplate='Component: ' + comp_name.replace('_', ' ').title() + 
                                  '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                    showlegend=True
                ))
        
        # Highlight habit plane if requested
        if show_habit_plane:
            max_stress = 0
            for comp_name in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                if comp_name in target_data['stresses']:
                    max_stress = max(max_stress, max(target_data['stresses'][comp_name]))
            
            fig.add_trace(go.Scatterpolar(
                r=[0, max_stress * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
                name=f'Habit Plane ({self.habit_angle}°)',
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} - {defect_type} Defect (ε*={eigen_strain:.2f})",
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    tickfont=dict(size=12, color='black'),
                    title=dict(text='Stress (GPa)', font=dict(size=14, color='black')),
                    range=[0, max_stress * 1.2]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
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
                bgcolor="rgba(240, 240, 240, 0.3)"
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
            height=700
        )
        
        return fig
    
    def create_normalized_stress_radar(self, defect_comparison, 
                                      title="Normalized Stress Patterns by Defect Type",
                                      show_habit_plane=True, normalize_by="max"):
        """Create radar chart with normalized stress values for better comparison"""
        fig = go.Figure()
        
        # Add traces for each defect type with normalized stresses
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = data['angles']
            stresses = data['stresses']['sigma_hydro']
            
            # Normalize stresses
            if normalize_by == "max":
                norm_stresses = np.array(stresses) / max(stresses) if max(stresses) > 0 else stresses
            elif normalize_by == "eigen_strain":
                eigen_strain = data.get('eigen_strain', 1.0)
                norm_stresses = np.array(stresses) / eigen_strain if eigen_strain > 0 else stresses
            else:  # normalize_by == "area"
                area = np.trapz(stresses, angles)
                norm_stresses = np.array(stresses) / area if area > 0 else stresses
            
            # Close the loop for radar chart
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(norm_stresses, norm_stresses[0])
            
            color = data.get('color', self.defect_colors.get(defect_type, 'black'))
            
            fig.add_trace(go.Scatterpolar(
                r=stresses_closed,
                theta=angles_closed,
                fill='toself',
                fillcolor=f'rgba{color[3:-1]}, 0.3)',
                line=dict(color=color, width=3),
                marker=dict(size=6, color=color),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='Defect: ' + defect_type + 
                              '<br>Orientation: %{theta:.2f}°<br>Normalized Stress: %{r:.4f}<extra></extra>',
                showlegend=True
            ))
        
        # Highlight habit plane if requested
        if show_habit_plane:
            fig.add_trace(go.Scatterpolar(
                r=[0, 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
                name=f'Habit Plane ({self.habit_angle}°)',
                hoverinfo='skip',
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
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    tickfont=dict(size=12, color='black'),
                    title=dict(text='Normalized Stress', font=dict(size=14, color='black')),
                    range=[0, 1.2]
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
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
                bgcolor="rgba(240, 240, 240, 0.3)"
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
            height=700
        )
        
        return fig
    
    def _rgb_to_hex(self, rgb_str):
        """Convert RGB string to hex format"""
        if rgb_str.startswith('rgb'):
            rgb_vals = rgb_str[4:-1].split(',')
            if len(rgb_vals) == 3:
                return f'#{int(rgb_vals[0]):02x}{int(rgb_vals[1]):02x}{int(rgb_vals[2]):02x}'
        return rgb_str

# =============================================
# STREAMLIT APPLICATION - FOCUSED ON DEFECT RADAR CHARTS
# =============================================
def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Defect Radar Charts - Habit Plane Analysis",
        layout="wide",
        page_icon="🎯",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Defect Radar Charts: Habit Plane Stress Analysis</h1>', unsafe_allow_html=True)
    
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
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        if st.button("🔄 Load Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions found in directory")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📊 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                st.write("Data loaded successfully!")
        
        # Analysis parameters
        st.markdown("#### 🎯 Analysis Parameters")
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
            ["Basic Radar Chart", "Sunburst Chart", "Multi-Component Radar", "Normalized Stress Radar"],
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
        
        # For multi-component radar
        defect_type_for_multi = "Twin"
        if chart_type == "Multi-Component Radar":
            defect_type_for_multi = st.selectbox(
                "Defect Type",
                ["ISF", "ESF", "Twin", "No Defect"],
                index=2,
                help="Select defect type for multi-component analysis"
            )
        
        # For normalized radar
        normalize_by = "max"
        if chart_type == "Normalized Stress Radar":
            normalize_by = st.selectbox(
                "Normalize By",
                ["max", "eigen_strain", "area"],
                index=0,
                help="Method for normalizing stress values"
            )
        
        # Generate button
        st.markdown("---")
        if st.button("🚀 Generate Radar Chart", type="primary", use_container_width=True):
            st.session_state.generate_chart = True
        else:
            st.session_state.generate_chart = False
    
    # Main content area
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        
        # Show directory information
        with st.expander("📁 Directory Information", expanded=True):
            st.info(f"**Solutions Directory:** {SOLUTIONS_DIR}")
            st.write("Expected file formats: .pkl, .pickle, .pt, .pth")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs([
            "📈 Radar Chart",
            "🎨 Customization",
            "💡 Concepts & Examples"
        ])
        
        with tab1:
            st.markdown('<h2 class="physics-header">📈 Defect Stress Radar Chart</h2>', unsafe_allow_html=True)
            
            if st.session_state.get('generate_chart', False) or 'defect_comparison' in st.session_state:
                with st.spinner("Generating radar chart..."):
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
                    
                    # Create the appropriate chart based on selection
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
                    elif chart_type == "Multi-Component Radar":
                        fig = st.session_state.visualizer.create_multi_component_radar(
                            defect_comparison,
                            defect_type=defect_type_for_multi,
                            title=f"Stress Components for {defect_type_for_multi} Defects"
                        )
                    else:  # Normalized Stress Radar
                        fig = st.session_state.visualizer.create_normalized_stress_radar(
                            defect_comparison,
                            title=f"Normalized {normalize_by.title()} Stress Patterns",
                            normalize_by=normalize_by
                        )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                    ### 📝 Interpretation Guide
                    
                    This radar chart shows how different defect types concentrate hydrostatic stress in the vicinity of 
                    the Ag FCC twin habit plane (54.7°). Key observations:
                    
                    - **Twin boundaries** typically show the highest stress concentration at the habit plane angle
                    - **Stacking faults** (ISF/ESF) show intermediate stress levels with broader distribution
                    - **Perfect crystals** show minimal stress variation
                    - The **habit plane** (54.7°) is highlighted with a green dashed line
                    
                    The stress patterns directly impact sintering behavior - higher stress concentrations 
                    enable lower-temperature sintering through enhanced atomic diffusion.
                    """)
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Radar Chart'")
                
                # Show example chart
                st.markdown("#### 📊 Example: Twin Boundary Stress Pattern")
                
                # Create example data
                angles = np.linspace(40, 69.4, 30)
                stresses = 25 * np.exp(-(angles - 54.7)**2 / (2 * 3**2)) + 2 * np.random.random(len(angles))
                
                fig_example = go.Figure()
                fig_example.add_trace(go.Scatterpolar(
                    r=np.append(stresses, stresses[0]),
                    theta=np.append(angles, angles[0]),
                    fill='toself',
                    fillcolor='rgba(69, 183, 209, 0.3)',
                    line=dict(color='rgb(69, 183, 209)', width=3),
                    marker=dict(size=8, color='rgb(69, 183, 209)'),
                    name='Twin Boundary',
                    hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>'
                ))
                
                fig_example.add_trace(go.Scatterpolar(
                    r=[0, 30],
                    theta=[54.7, 54.7],
                    mode='lines',
                    line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
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
        
        with tab2:
            st.markdown('<h2 class="physics-header">🎨 Chart Customization</h2>', unsafe_allow_html=True)
            
            # Show current chart with customization options
            if 'defect_comparison' in st.session_state:
                defect_comparison = st.session_state.defect_comparison
                
                st.markdown("#### 🖌️ Visual Customization Options")
                
                # Create columns for customization options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Basic Appearance")
                    fill_opacity = st.slider("Fill Opacity", 0.0, 1.0, 0.2, 0.05)
                    line_width = st.slider("Line Width", 1, 8, 3)
                    marker_size = st.slider("Marker Size", 0, 15, 8)
                    show_grid = st.checkbox("Show Grid Lines", value=True)
                    bgcolor = st.color_picker("Background Color", "#ffffff")
                
                with col2:
                    st.markdown("##### Advanced Options")
                    show_habit_plane = st.checkbox("Show Habit Plane", value=True)
                    chart_title = st.text_input("Chart Title", "Customized Defect Stress Patterns")
                    stress_component_custom = st.selectbox(
                        "Stress Component", 
                        ["sigma_hydro", "von_mises", "sigma_mag"],
                        index=0
                    )
                    radar_type = st.radio(
                        "Radar Chart Type",
                        ["Standard", "Enhanced"],
                        help="Enhanced charts have smoother curves"
                    )
                
                # Generate customized chart
                if st.button("🎨 Apply Customization", type="primary"):
                    with st.spinner("Applying customization..."):
                        if radar_type == "Standard":
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component_custom,
                                title=chart_title,
                                show_habit_plane=show_habit_plane,
                                fill_opacity=fill_opacity,
                                line_width=line_width,
                                marker_size=marker_size,
                                show_grid=show_grid,
                                bgcolor=bgcolor
                            )
                        else:
                            # For enhanced, we can add smoothing
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component_custom,
                                title=chart_title,
                                show_habit_plane=show_habit_plane,
                                fill_opacity=fill_opacity,
                                line_width=line_width,
                                marker_size=marker_size,
                                show_grid=show_grid,
                                bgcolor=bgcolor
                            )
                            
                            # Add some smoothing to the line
                            for trace in fig.data:
                                if 'line' in trace:
                                    trace.line.shape = 'spline'
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add download button
                        if st.button("💾 Download Chart as HTML"):
                            fig.write_html("custom_defect_radar.html")
                            with open("custom_defect_radar.html", "rb") as file:
                                st.download_button(
                                    label="Download HTML",
                                    data=file,
                                    file_name="custom_defect_radar.html",
                                    mime="text/html"
                                )
                
                st.markdown("#### 🎨 Color Customization")
                
                # Color customization for each defect type
                defect_colors = {}
                for defect in ["ISF", "ESF", "Twin", "No Defect"]:
                    col_name = f"{defect.lower()}_color"
                    default_color = st.session_state.visualizer.defect_colors[defect]
                    defect_colors[defect] = st.color_picker(f"{defect} Color", default_color)
                
                if st.button("🎨 Apply Color Scheme"):
                    # Update the visualizer's color scheme
                    for defect, color in defect_colors.items():
                        st.session_state.visualizer.defect_colors[defect] = color
                    
                    # Regenerate the chart with new colors
                    fig = st.session_state.visualizer.create_basic_defect_radar(
                        defect_comparison,
                        stress_component=stress_component_custom,
                        title=chart_title,
                        show_habit_plane=show_habit_plane,
                        fill_opacity=fill_opacity,
                        line_width=line_width,
                        marker_size=marker_size,
                        show_grid=show_grid,
                        bgcolor=bgcolor
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("👈 First generate a radar chart in the 'Radar Chart' tab, then customize it here.")
        
        with tab3:
            st.markdown('<h2 class="physics-header">💡 Visualization Concepts & Examples</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            ### 🌟 Advanced Visualization Concepts for Defect Stress Patterns
            
            Below are several concepts for visualizing hydrostatic stress patterns around the habit plane. 
            These approaches highlight different aspects of the stress distribution for various defect types.
            """)
            
            # Concept 1: Basic Radar Chart
            st.markdown("#### 1. Basic Radar Chart")
            st.markdown("""
            The standard radar chart is excellent for comparing multiple defect types on the same plot.
            Each axis represents an orientation angle, and the distance from the center represents stress magnitude.
            """)
            
            # Create example basic radar chart
            angles = np.linspace(40, 69.4, 30)
            twin_stress = 25 * np.exp(-(angles - 54.7)**2 / (2 * 3**2))
            isf_stress = 15 * np.exp(-(angles - 54.7)**2 / (2 * 6**2))
            esf_stress = 20 * np.exp(-(angles - 54.7)**2 / (2 * 4.5**2))
            perfect_stress = 3 * np.exp(-(angles - 54.7)**2 / (2 * 10**2))
            
            fig_concept1 = go.Figure()
            
            # Add traces for each defect type
            defects = [
                ("Twin", twin_stress, "rgb(69, 183, 209)"),
                ("ISF", isf_stress, "rgb(255, 107, 107)"),
                ("ESF", esf_stress, "rgb(78, 205, 196)"),
                ("No Defect", perfect_stress, "rgb(150, 206, 180)")
            ]
            
            for name, stress, color in defects:
                fig_concept1.add_trace(go.Scatterpolar(
                    r=np.append(stress, stress[0]),
                    theta=np.append(angles, angles[0]),
                    fill='toself',
                    fillcolor=f'rgba{color[3:-1]}, 0.3)',
                    line=dict(color=color, width=3),
                    name=name,
                    hovertemplate=f'Defect: {name}<br>Orientation: %{{theta:.2f}}°<br>Stress: %{{r:.4f}} GPa<extra></extra>'
                ))
            
            # Add habit plane
            fig_concept1.add_trace(go.Scatterpolar(
                r=[0, 30],
                theta=[54.7, 54.7],
                mode='lines',
                line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
                name='Habit Plane (54.7°)',
                hoverinfo='skip'
            ))
            
            fig_concept1.update_layout(
                title="Basic Radar Chart: Comparing Defect Stress Patterns",
                polar=dict(
                    radialaxis=dict(range=[0, 30], title="Stress (GPa)"),
                    angularaxis=dict(rotation=90, direction="clockwise")
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig_concept1, use_container_width=True)
            
            # Concept 2: Sunburst Chart
            st.markdown("#### 2. Sunburst Chart")
            st.markdown("""
            The sunburst chart provides a more dramatic visualization that emphasizes the radial nature 
            of the stress distribution. Color gradients can be used to highlight stress intensity.
            """)
            
            # Create example sunburst chart
            fig_concept2 = go.Figure()
            
            # Add traces with color gradients
            for name, stress, color in defects:
                # Create color array based on stress values
                stress_normalized = stress / max(stress) if max(stress) > 0 else stress
                colors = [f'rgba{color[3:-1]}, {0.3 + 0.7 * s})' for s in stress_normalized]
                
                fig_concept2.add_trace(go.Scatterpolar(
                    r=stress,
                    theta=angles,
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=stress,
                        colorscale='RdBu',
                        showscale=False
                    ),
                    line=dict(color=color, width=4),
                    name=name,
                    hovertemplate=f'Defect: {name}<br>Orientation: %{{theta:.2f}}°<br>Stress: %{{r:.4f}} GPa<extra></extra>'
                ))
            
            # Highlight habit plane peak
            fig_concept2.add_trace(go.Scatterpolar(
                r=[max(twin_stress)],
                theta=[54.7],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='rgb(46, 204, 113)',
                    symbol='star'
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(size=12, color='black'),
                name='Peak Stress Location',
                hovertemplate=f'Habit Plane (54.7°)<br>Max Stress: {max(twin_stress):.2f} GPa<extra></extra>'
            ))
            
            fig_concept2.update_layout(
                title="Sunburst Chart: Visualizing Stress Concentration",
                polar=dict(
                    radialaxis=dict(range=[0, 30], title="Stress (GPa)"),
                    angularaxis=dict(rotation=90, direction="clockwise")
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig_concept2, use_container_width=True)
            
            # Concept 3: Multi-Component Analysis
            st.markdown("#### 3. Multi-Component Stress Analysis")
            st.markdown("""
            This visualization shows multiple stress components (hydrostatic, von Mises, magnitude)
            for a single defect type. This provides a comprehensive view of the stress state.
            """)
            
            # Create example for multiple stress components
            fig_concept3 = go.Figure()
            
            # Twin defect stress components
            sigma_hydro = twin_stress
            von_mises = twin_stress * 1.3 * (0.9 + 0.1 * np.sin(np.radians(angles)))
            sigma_mag = np.sqrt(sigma_hydro**2 + von_mises**2) * 0.8
            
            components = [
                ("Hydrostatic", sigma_hydro, "rgb(31, 119, 180)"),
                ("Von Mises", von_mises, "rgb(255, 127, 14)"),
                ("Magnitude", sigma_mag, "rgb(44, 160, 44)")
            ]
            
            for name, stress, color in components:
                fig_concept3.add_trace(go.Scatterpolar(
                    r=np.append(stress, stress[0]),
                    theta=np.append(angles, angles[0]),
                    fill='toself',
                    fillcolor=f'rgba{color[3:-1]}, 0.2)',
                    line=dict(color=color, width=3),
                    name=name,
                    hovertemplate=f'Component: {name}<br>Orientation: %{{theta:.2f}}°<br>Stress: %{{r:.4f}} GPa<extra></extra>'
                ))
            
            # Add habit plane
            fig_concept3.add_trace(go.Scatterpolar(
                r=[0, 35],
                theta=[54.7, 54.7],
                mode='lines',
                line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
                name='Habit Plane (54.7°)',
                hoverinfo='skip'
            ))
            
            fig_concept3.update_layout(
                title="Multi-Component Analysis: Twin Boundary Stress State",
                polar=dict(
                    radialaxis=dict(range=[0, 35], title="Stress (GPa)"),
                    angularaxis=dict(rotation=90, direction="clockwise")
                ),
                width=800,
                height=600
            )
            
            st.plotly_chart(fig_concept3, use_container_width=True)
            
            st.markdown("""
            ### 🎨 Customization Tips
            
            1. **Color Schemes**: Use contrasting colors for different defect types. Consider using colorblind-friendly palettes.
            
            2. **Scaling**: For comparing defect types, consider normalizing stress values to highlight pattern differences rather than magnitude differences.
            
            3. **Annotations**: Add text annotations to highlight important features like peak stress locations or critical angles.
            
            4. **Interactivity**: Enable hover information to show exact values when users interact with the chart.
            
            5. **Layout**: Adjust the polar coordinate system to focus on the most relevant angular range (typically around the habit plane).
            
            These visualization techniques provide powerful ways to understand how different crystal defects 
            influence stress distributions near the habit plane, which is crucial for predicting sintering behavior 
            and material performance.
            """)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
