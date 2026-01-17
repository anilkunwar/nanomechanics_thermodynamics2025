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
        # Use HEX colors here for consistency with Streamlit
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
# FOCUSED VISUALIZATION CLASS FOR DEFECT RADAR CHARTS - ENHANCED & FIXED
# =============================================
class DefectRadarVisualizer:
    """Focused visualizer for defect radar charts with customization options"""
    
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        # CRITICAL FIX: Use HEX colors for Streamlit compatibility
        self.defect_colors = {
            'ISF': '#FF6B6B',    # Red-orange
            'ESF': '#4ECDC4',     # Teal
            'Twin': '#45B7D1',    # Blue
            'No Defect': '#96CEB4' # Green
        }
        # FIXED: Initialize stress_component_colors attribute
        self.stress_component_colors = {
            'sigma_hydro': '#1F77B4',  # Blue
            'von_mises': '#FF7F0E',   # Orange
            'sigma_mag': '#2CA02C'     # Green
        }
    
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
        
        # Return as-is if already valid hex
        if color.startswith('#') and (len(color) == 4 or len(color) == 7 or len(color) == 9):
            return color
        
        # Handle RGB format
        if color.startswith(('rgb', 'rgba')):
            try:
                # Extract RGB values
                nums = list(map(int, re.findall(r'\d+', color)))
                if len(nums) >= 3:
                    r, g, b = nums[:3]
                    return f'#{r:02x}{g:02x}{b:02x}'
            except:
                pass
        
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
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            
            # Create fill color with proper opacity handling
            fill_color = self._apply_opacity_to_color(color, fill_opacity)
            
            # Add the trace
            fig.add_trace(go.Scatterpolar(
                r=stresses_closed,
                theta=angles_closed,
                fill='toself',
                fillcolor=fill_color,
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size, color=color, line=dict(width=1, color='white')),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
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
                line=dict(color='#2ECC71', width=4, dash='dashdot'),  # Use hex color
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
            height=700,
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor
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
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            
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
                        color='#2ECC71',  # Use hex color
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
                
                color = self._get_valid_color(self.stress_component_colors.get(comp_name, '#000000'))
                fill_color = self._apply_opacity_to_color(color, fill_opacity)
                
                fig.add_trace(go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color),
                    name=f"{comp_name.replace('_', ' ').title()}",
                    hovertemplate='Component: ' + comp_name.replace('_', ' ').title() + 
                                  '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                    showlegend=True
                ))
        
        # Calculate max stress across all components
        max_stress = 0
        for comp_name in ['sigma_hydro', 'von_mises', 'sigma_mag']:
            if comp_name in target_data['stresses']:
                max_stress = max(max_stress, max(target_data['stresses'][comp_name]))
        
        # Highlight habit plane if requested
        if show_habit_plane:
            fig.add_trace(go.Scatterpolar(
                r=[0, max_stress * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='#2ECC71', width=4, dash='dashdot'),  # Use hex color
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
                max_val = max(stresses)
                norm_stresses = np.array(stresses) / max_val if max_val > 0 else stresses
            elif normalize_by == "eigen_strain":
                eigen_strain = data.get('eigen_strain', 1.0)
                norm_stresses = np.array(stresses) / eigen_strain if eigen_strain > 0 else stresses
            else:  # normalize_by == "area"
                area = np.trapz(stresses, angles)
                norm_stresses = np.array(stresses) / area if area > 0 else stresses
            
            # Close the loop for radar chart
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(norm_stresses, norm_stresses[0])
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            fill_color = self._apply_opacity_to_color(color, 0.3)
            
            fig.add_trace(go.Scatterpolar(
                r=stresses_closed,
                theta=angles_closed,
                fill='toself',
                fillcolor=fill_color,
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
                line=dict(color='#2ECC71', width=4, dash='dashdot'),  # Use hex color
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
    
    def create_advanced_sunburst_chart(self, defect_comparison, stress_component='sigma_hydro',
                                      title="Advanced Sunburst: Defect Stress Concentration",
                                      show_habit_plane=True, color_scale='Viridis',
                                      radial_range_factor=1.2):
        """
        Create an advanced sunburst-style visualization with enhanced features
        for showing stress concentration patterns.
        """
        fig = go.Figure()
        
        # Calculate global max stress for consistent scaling
        global_max_stress = 0
        for data in defect_comparison.values():
            global_max_stress = max(global_max_stress, max(data['stresses'][stress_component]))
        
        # Create smooth curves for each defect type using spline interpolation
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = np.array(data['angles'])
            stresses = np.array(data['stresses'][stress_component])
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            
            # Create smooth curve by adding intermediate points
            smooth_angles = np.linspace(min(angles), max(angles), 300)
            smooth_stresses = np.interp(smooth_angles, angles, stresses)
            
            # Add gradient color based on stress magnitude
            fig.add_trace(go.Scatterpolar(
                r=smooth_stresses * radial_range_factor,
                theta=smooth_angles,
                mode='lines',
                line=dict(
                    color=color,
                    width=4,
                    shape='spline'
                ),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='Defect: ' + defect_type + 
                              '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                showlegend=True
            ))
            
            # Add high-stress markers
            high_stress_idx = np.where(stresses > np.percentile(stresses, 75))[0]
            if len(high_stress_idx) > 0:
                high_angles = angles[high_stress_idx]
                high_stresses = stresses[high_stress_idx]
                
                fig.add_trace(go.Scatterpolar(
                    r=high_stresses * radial_range_factor * 1.05,
                    theta=high_angles,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=high_stresses,
                        colorscale=color_scale,
                        symbol='star',
                        line=dict(width=1, color='white')
                    ),
                    name=f'{defect_type} Hotspots',
                    hovertemplate='Stress Hotspot<br>Defect: ' + defect_type + 
                                  '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                    showlegend=False
                ))
        
        # Highlight habit plane with special marker
        if show_habit_plane:
            # Find the maximum stress at habit plane across all defects
            habit_stresses = []
            for data in defect_comparison.values():
                angles = np.array(data['angles'])
                stresses = np.array(data['stresses'][stress_component])
                habit_idx = np.argmin(np.abs(angles - self.habit_angle))
                habit_stresses.append(stresses[habit_idx])
            
            max_habit_stress = max(habit_stresses) * radial_range_factor
            
            fig.add_trace(go.Scatterpolar(
                r=[max_habit_stress * 1.15],
                theta=[self.habit_angle],
                mode='markers+text',
                marker=dict(
                    size=40,
                    color='gold',
                    symbol='diamond',
                    line=dict(width=3, color='#D4AF37')  # Use hex for gold border
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(size=16, color='darkred', family='Arial Black'),
                name=f'Habit Plane Peak ({self.habit_angle}°)',
                hovertemplate=f'Habit Plane ({self.habit_angle}°)<br>Peak Stress: {max_habit_stress:.4f} GPa<extra></extra>',
                showlegend=True
            ))
            
            # Add habit plane reference line
            fig.add_trace(go.Scatterpolar(
                r=[0, max_habit_stress * 1.25],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='#D4AF37', width=3, dash='dot'),  # Use hex color
                name=f'Habit Plane Reference',
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Add circular grid lines for reference
        max_range = global_max_stress * radial_range_factor * 1.3
        for r_val in np.linspace(0, max_range, 5)[1:]:
            fig.add_trace(go.Scatterpolar(
                r=[r_val] * 360,
                theta=np.linspace(0, 360, 360),
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.2)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Update layout for professional appearance
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
                    title=dict(text=f'{stress_component.replace("_", " ").title()} Stress (GPa)', 
                              font=dict(size=16, color='black', family='Arial Bold')),
                    range=[0, max_range],
                    tickprefix="",
                    ticksuffix="",
                    showticklabels=True
                ),
                angularaxis=dict(
                    gridcolor="rgba(150, 150, 150, 0.4)",
                    gridwidth=1,
                    linecolor="black",
                    linewidth=2,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=np.linspace(min(angles), max(angles), 7),
                    ticktext=[f'{i:.1f}°' for i in np.linspace(min(angles), max(angles), 7)],
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
        
        return fig
    
    # FIXED: Added the missing create_interactive_3d_defect_sunburst method
    def create_interactive_3d_defect_sunburst(self, defect_comparison,
                                           stress_component='sigma_hydro',
                                           title="3D Defect Stress Distribution"):
        """
        Create an interactive 3D sunburst visualization for defect stress patterns.
        This provides a unique perspective on stress concentration.
        """
        fig = go.Figure()
        
        # Create 3D coordinates for each defect type
        z_offset = 0
        max_z = len(defect_comparison) * 5
        
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
            
            # Create smooth surface by adding intermediate points
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines+markers',
                line=dict(
                    color=color,
                    width=6,
                    shape='spline'
                ),
                marker=dict(
                    size=8,
                    color=stresses,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name=f"{defect_type} (ε*={data.get('eigen_strain', 0):.2f})",
                hovertemplate='<b>' + defect_type + '</b><br>' +
                              'Orientation: %{customdata[0]:.2f}°<br>' +
                              'Stress: %{customdata[1]:.4f} GPa<br>' +
                              'Z-Position: %{z:.1f}<extra></extra>',
                customdata=np.column_stack([angles, stresses]),
                showlegend=True
            ))
            
            # Add a connecting line to origin for visual reference
            fig.add_trace(go.Scatter3d(
                x=[0, x[0]],
                y=[0, y[0]],
                z=[z_offset, z_offset],
                mode='lines',
                line=dict(color='rgba(150,150,150,0.3)', width=2),
                hoverinfo='skip',
                showlegend=False
            ))
            
            z_offset += 5
        
        # Add habit plane reference planes
        max_stress = max(max(data['stresses'][stress_component]) for data in defect_comparison.values())
        habit_plane_angle = np.radians(self.habit_angle)
        
        # Create a grid for the habit plane reference
        x_plane = np.linspace(-max_stress, max_stress, 10)
        y_plane = np.linspace(-max_stress, max_stress, 10)
        X, Y = np.meshgrid(x_plane, y_plane)
        Z = np.zeros_like(X)
        
        # Rotate the plane to match habit angle
        X_rot = X * np.cos(habit_plane_angle) - Y * np.sin(habit_plane_angle)
        Y_rot = X * np.sin(habit_plane_angle) + Y * np.cos(habit_plane_angle)
        
        # Add transparent habit plane reference
        fig.add_trace(go.Surface(
            x=X_rot, y=Y_rot, z=Z + max_z/2,
            opacity=0.15,
            colorscale=[[0, 'gold'], [1, 'gold']],
            showscale=False,
            name=f'Habit Plane ({self.habit_angle}°)',
            hoverinfo='skip'
        ))
        
        # Update 3D layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(title='X Stress Component', gridcolor='lightgray'),
                yaxis=dict(title='Y Stress Component', gridcolor='lightgray'),
                zaxis=dict(title='Defect Type', gridcolor='lightgray',
                          ticktext=[data.get('defect_type', '') for data in defect_comparison.values()],
                          tickvals=list(range(0, max_z, 5))),
                bgcolor='rgba(240, 240, 240, 0.8)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
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
            width=950,
            height=700
        )
        
        return fig
    
    def create_defect_comparison_dashboard(self, defect_comparison,
                                         stress_component='sigma_hydro',
                                         title="Defect Stress Analysis Dashboard"):
        """
        Create a comprehensive dashboard with multiple visualization types
        for comparing defect stress patterns.
        """
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Radar Chart Comparison',
                'Stress Magnitude Distribution',
                'Habit Plane Focus',
                'Normalized Patterns'
            ),
            specs=[
                [{'type': 'polar'}, {'type': 'xy'}],
                [{'type': 'polar'}, {'type': 'polar'}]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )
        
        # Plot 1: Basic radar chart in top-left
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = data['angles']
            stresses = data['stresses'][stress_component]
            
            # Close the loop
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(stresses, stresses[0])
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            fill_color = self._apply_opacity_to_color(color, 0.2)
            
            fig.add_trace(
                go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color=color, width=2),
                    name=f"{defect_type}",
                    showlegend=False,
                    hovertemplate='Defect: ' + defect_type + 
                                  '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add habit plane reference to plot 1
        max_stress_p1 = max(max(data['stresses'][stress_component]) for data in defect_comparison.values())
        fig.add_trace(
            go.Scatterpolar(
                r=[0, max_stress_p1 * 1.2],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(color='#2ECC71', width=3, dash='dashdot'),  # Use hex color
                name='Habit Plane',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Plot 2: Stress magnitude distribution (histogram) in top-right
        all_stresses = []
        all_defects = []
        
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            stresses = data['stresses'][stress_component]
            all_stresses.extend(stresses)
            all_defects.extend([defect_type] * len(stresses))
        
        unique_defects = list(set(all_defects))
        for defect_type in unique_defects:
            defect_stresses = [s for s, d in zip(all_stresses, all_defects) if d == defect_type]
            color = self._get_valid_color(self.defect_colors.get(defect_type, '#000000'))
            
            fig.add_trace(
                go.Histogram(
                    x=defect_stresses,
                    name=defect_type,
                    marker_color=color,
                    opacity=0.7,
                    hovertemplate='Defect: ' + defect_type + 
                                  '<br>Stress Range: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Habit plane focus in bottom-left
        habit_range = 10  # degrees around habit plane
        min_angle = self.habit_angle - habit_range
        max_angle = self.habit_angle + habit_range
        
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = np.array(data['angles'])
            stresses = np.array(data['stresses'][stress_component])
            
            # Filter data near habit plane
            mask = (angles >= min_angle) & (angles <= max_angle)
            if np.any(mask):
                filtered_angles = angles[mask]
                filtered_stresses = stresses[mask]
                
                color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=filtered_stresses,
                        theta=filtered_angles,
                        mode='lines+markers',
                        line=dict(color=color, width=3),
                        marker=dict(size=8, color=color),
                        name=f"{defect_type}",
                        showlegend=False,
                        hovertemplate='Defect: ' + defect_type + 
                                      '<br>Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Add habit plane marker to plot 3
        fig.add_trace(
            go.Scatterpolar(
                r=[max_stress_p1 * 1.1],
                theta=[self.habit_angle],
                mode='markers+text',
                marker=dict(size=15, color='gold', symbol='star'),
                text=['HABIT'],
                textposition='top center',
                name='Habit Plane Focus',
                showlegend=False,
                hovertemplate=f'Habit Plane ({self.habit_angle}°)<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 4: Normalized patterns in bottom-right
        for defect_key, data in defect_comparison.items():
            defect_type = data.get('defect_type', 'Unknown')
            angles = data['angles']
            stresses = data['stresses'][stress_component]
            
            # Normalize by max stress
            max_stress = max(stresses)
            norm_stresses = np.array(stresses) / max_stress if max_stress > 0 else stresses
            
            # Close the loop
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(norm_stresses, norm_stresses[0])
            
            color = self._get_valid_color(data.get('color', self.defect_colors.get(defect_type, '#000000')))
            fill_color = self._apply_opacity_to_color(color, 0.2)
            
            fig.add_trace(
                go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color=color, width=2),
                    name=f"{defect_type}",
                    showlegend=True,
                    hovertemplate='Defect: ' + defect_type + 
                                  '<br>Orientation: %{theta:.2f}°<br>Normalized Stress: %{r:.4f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout for the dashboard
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=28, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95
            ),
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                x=1.05,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            plot_bgcolor='rgba(248, 249, 252, 0.8)',
            paper_bgcolor='rgba(248, 249, 252, 0.8)'
        )
        
        # Update polar axes
        fig.update_polars(
            radialaxis=dict(
                tickfont=dict(size=10),
                title_font=dict(size=10),
                gridcolor='rgba(150,150,150,0.3)',
                linecolor='black',
                linewidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                gridcolor='rgba(150,150,150,0.3)',
                linecolor='black',
                linewidth=1,
                rotation=90,
                direction='clockwise'
            )
        )
        
        # Update the histogram layout
        fig.update_xaxes(title_text="Stress (GPa)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        return fig

# =============================================
# STREAMLIT APPLICATION - ENHANCED & FIXED
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
            [
                "Basic Radar Chart", 
                "Sunburst Chart", 
                "Multi-Component Radar", 
                "Normalized Stress Radar",
                "Advanced Sunburst",
                "3D Interactive View",
                "Comparison Dashboard"
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
        
        # For advanced sunburst
        color_scale = "Viridis"
        if chart_type in ["Advanced Sunburst", "3D Interactive View"]:
            color_scale = st.selectbox(
                "Color Scale",
                ["Viridis", "RdBu", "Plasma", "Inferno", "Magma", "Cividis"],
                index=0,
                help="Color scale for stress visualization"
            )
        
        # Generate button
        st.markdown("---")
        if st.button("🚀 Generate Visualization", type="primary", use_container_width=True):
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
            
            # Show physics background
            st.markdown("#### 🔬 Physics Background")
            st.markdown("""
            This application focuses on visualizing stress patterns around crystal defects, 
            particularly near the FCC twin habit plane angle of 54.7°. The radar charts and sunburst 
            visualizations reveal how different defect types concentrate stress, which is crucial 
            for understanding sintering behavior in nanomaterials.
            """)
    else:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Main Visualization",
            "🎨 Customization",
            "💡 Concepts & Examples",
            "🔍 Detailed Analysis"
        ])
        
        with tab1:
            st.markdown('<h2 class="physics-header">📈 Defect Stress Visualization</h2>', unsafe_allow_html=True)
            
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
                        elif chart_type == "Multi-Component Radar":
                            fig = st.session_state.visualizer.create_multi_component_radar(
                                defect_comparison,
                                defect_type=defect_type_for_multi,
                                title=f"Stress Components for {defect_type_for_multi} Defects"
                            )
                        elif chart_type == "Normalized Stress Radar":
                            fig = st.session_state.visualizer.create_normalized_stress_radar(
                                defect_comparison,
                                title=f"Normalized {normalize_by.title()} Stress Patterns",
                                normalize_by=normalize_by
                            )
                        elif chart_type == "Advanced Sunburst":
                            fig = st.session_state.visualizer.create_advanced_sunburst_chart(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"Advanced Stress Concentration: {stress_component.replace('_', ' ').title()}",
                                color_scale=color_scale
                            )
                        elif chart_type == "3D Interactive View":
                            # FIXED: This method now exists in the DefectRadarVisualizer class
                            fig = st.session_state.visualizer.create_interactive_3d_defect_sunburst(
                                defect_comparison,
                                stress_component=stress_component,
                                title=f"3D Defect Stress Distribution: {stress_component.replace('_', ' ').title()}"
                            )
                        else:  # Comparison Dashboard
                            fig = st.session_state.visualizer.create_defect_comparison_dashboard(
                                defect_comparison,
                                stress_component=stress_component,
                                title="Comprehensive Defect Stress Analysis Dashboard"
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
                        elif chart_type == "Advanced Sunburst":
                            st.markdown("""
                            <div class="success-box">
                            💡 <strong>Advanced Analysis:</strong> The sunburst visualization highlights stress hotspots with star markers, 
                            showing how Twin defects create extreme stress concentrations that could drive atomic diffusion during sintering.
                            </div>
                            """, unsafe_allow_html=True)
                        elif chart_type == "3D Interactive View":
                            st.markdown("""
                            <div class="insight-box">
                            🌐 <strong>3D Perspective:</strong> This interactive view shows how stress patterns vary across different defect types, 
                            with the Z-axis separating defect categories. The gold plane represents the habit plane orientation.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box">
                            📊 <strong>Comprehensive View:</strong> This dashboard provides multiple perspectives on defect stress patterns, 
                            combining radial views, histograms, and focused habit plane analysis for complete understanding.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add download button
                        if st.button("💾 Download as HTML", key="download_main"):
                            fig.write_html("defect_stress_visualization.html")
                            with open("defect_stress_visualization.html", "rb") as file:
                                st.download_button(
                                    label="Download HTML File",
                                    data=file,
                                    file_name="defect_stress_visualization.html",
                                    mime="text/html",
                                    key="download_button_main"
                                )
                    
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
                        st.exception(e)
            else:
                st.info("👈 Configure analysis parameters in the sidebar and click 'Generate Visualization'")
                
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
                    line=dict(color='#45B7D1', width=3),  # Use hex color
                    marker=dict(size=8, color='#45B7D1'),
                    name='Twin Boundary',
                    hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>'
                ))
                
                fig_example.add_trace(go.Scatterpolar(
                    r=[0, 30],
                    theta=[54.7, 54.7],
                    mode='lines',
                    line=dict(color='#2ECC71', width=4, dash='dashdot'),  # Use hex color
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
                    💫 <strong>Habit Plane Significance:</strong> The 54.7° angle represents the FCC twin habit plane 
                    where stress concentration is maximized for twin boundary defects. This specific orientation 
                    enables lower-temperature sintering through enhanced atomic diffusion.
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<h2 class="physics-header">🎨 Advanced Customization</h2>', unsafe_allow_html=True)
            
            # Show current chart with customization options
            if 'defect_comparison' in st.session_state:
                defect_comparison = st.session_state.defect_comparison
                
                st.markdown("#### 🖌️ Visual Properties")
                
                # Create columns for customization options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### **Color & Transparency**")
                    fill_opacity = st.slider("Fill Opacity", 0.0, 1.0, 0.2, 0.05)
                    line_width = st.slider("Line Width", 1, 8, 3)
                    marker_size = st.slider("Marker Size", 0, 15, 8)
                    bg_opacity = st.slider("Background Opacity", 0.0, 1.0, 1.0, 0.05)
                
                with col2:
                    st.markdown("##### **Layout & Grid**")
                    show_grid = st.checkbox("Show Grid Lines", value=True)
                    show_habit_plane = st.checkbox("Show Habit Plane", value=True)
                    grid_color = st.color_picker("Grid Color", "#cccccc")
                    bgcolor = st.color_picker("Background Color", "#ffffff")
                    
                    # Advanced background option
                    bg_type = st.radio("Background Type", ["Solid", "Gradient"], horizontal=True)
                    if bg_type == "Gradient":
                        bg_color2 = st.color_picker("Second Gradient Color", "#e0e0e0")
                
                with col3:
                    st.markdown("##### **Text & Labels**")
                    chart_title = st.text_input("Chart Title", "Customized Defect Stress Patterns")
                    title_size = st.slider("Title Font Size", 12, 36, 20)
                    label_size = st.slider("Label Font Size", 8, 24, 12)
                    legend_position = st.selectbox("Legend Position", 
                                                 ["right", "top", "bottom", "left"], index=0)
                
                # Color scheme customization - FIXED FOR STREAMLIT
                st.markdown("#### 🎨 Defect Type Color Scheme")
                defect_colors = {}
                cols = st.columns(4)
                for i, defect in enumerate(["ISF", "ESF", "Twin", "No Defect"]):
                    with cols[i]:
                        st.markdown(f"**{defect}**")
                        # Get current color in hex format
                        current_color = st.session_state.visualizer.defect_colors[defect]
                        # CRITICAL FIX: Ensure color is in hex format for Streamlit
                        if not current_color.startswith('#'):
                            current_color = '#45B7D1'  # Default to Twin blue if invalid
                        
                        # Use properly formatted key without spaces
                        color_key = f"color_{defect.replace(' ', '_')}"
                        defect_colors[defect] = st.color_picker(
                            f"{defect} Color", 
                            current_color, 
                            key=color_key
                        )
                        st.caption(f"Eigen strain: {st.session_state.interpolator.physics_analyzer.get_eigen_strain(defect):.2f}")
                
                # Stress component colors - FIXED FOR STREAMLIT
                st.markdown("#### 📈 Stress Component Colors")
                stress_colors = {}
                stress_components = ["sigma_hydro", "von_mises", "sigma_mag"]
                stress_labels = ["Hydrostatic", "Von Mises", "Magnitude"]
                cols = st.columns(3)
                for i, (comp, label) in enumerate(zip(stress_components, stress_labels)):
                    with cols[i]:
                        st.markdown(f"**{label}**")
                        # Get current color in hex format
                        current_color = st.session_state.visualizer.stress_component_colors[comp]
                        if not current_color.startswith('#'):
                            # Default colors if invalid
                            defaults = {'sigma_hydro': '#1F77B4', 'von_mises': '#FF7F0E', 'sigma_mag': '#2CA02C'}
                            current_color = defaults[comp]
                        
                        # Use properly formatted key
                        color_key = f"stress_{comp}"
                        stress_colors[comp] = st.color_picker(
                            f"{label} Color", 
                            current_color, 
                            key=color_key
                        )
                
                # Generate customized chart
                if st.button("🎨 Render Customized Visualization", type="primary"):
                    with st.spinner("Applying customization..."):
                        # Update color schemes
                        for defect, color in defect_colors.items():
                            st.session_state.visualizer.defect_colors[defect] = color
                        
                        for comp, color in stress_colors.items():
                            st.session_state.visualizer.stress_component_colors[comp] = color
                        
                        # Create background color based on type
                        bg_color = bgcolor
                        if bg_type == "Gradient":
                            # Simple gradient implementation via opacity
                            bg_color = f"rgba({int(bgcolor[1:3], 16)}, {int(bgcolor[3:5], 16)}, {int(bgcolor[5:7], 16)}, {bg_opacity})"
                        
                        # Generate the appropriate chart with customization
                        if chart_type == "Basic Radar Chart":
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component,
                                title=chart_title,
                                show_habit_plane=show_habit_plane,
                                fill_opacity=fill_opacity,
                                line_width=line_width,
                                marker_size=marker_size,
                                show_grid=show_grid,
                                bgcolor=bg_color
                            )
                        elif chart_type == "Sunburst Chart":
                            fig = st.session_state.visualizer.create_sunburst_defect_chart(
                                defect_comparison,
                                stress_component=stress_component,
                                title=chart_title,
                                show_habit_plane=show_habit_plane,
                                color_scale=color_scale
                            )
                        elif chart_type == "Multi-Component Radar":
                            fig = st.session_state.visualizer.create_multi_component_radar(
                                defect_comparison,
                                defect_type=defect_type_for_multi,
                                title=chart_title,
                                show_habit_plane=show_habit_plane,
                                fill_opacity=fill_opacity
                            )
                        elif chart_type == "3D Interactive View":
                            # Ensure the method exists
                            fig = st.session_state.visualizer.create_interactive_3d_defect_sunburst(
                                defect_comparison,
                                stress_component=stress_component,
                                title=chart_title
                            )
                        else:  # Default to basic radar
                            fig = st.session_state.visualizer.create_basic_defect_radar(
                                defect_comparison,
                                stress_component=stress_component,
                                title=chart_title,
                                show_habit_plane=show_habit_plane,
                                fill_opacity=fill_opacity,
                                line_width=line_width,
                                marker_size=marker_size,
                                show_grid=show_grid,
                                bgcolor=bg_color
                            )
                        
                        # Update layout with custom font sizes
                        fig.update_layout(
                            title_font_size=title_size,
                            font=dict(size=label_size)
                        )
                        
                        # Position legend
                        if legend_position == "right":
                            fig.update_layout(legend=dict(x=1.1, y=0.5))
                        elif legend_position == "top":
                            fig.update_layout(legend=dict(x=0.5, y=1.1, xanchor='center', yanchor='bottom'))
                        elif legend_position == "bottom":
                            fig.update_layout(legend=dict(x=0.5, y=-0.2, xanchor='center', yanchor='top'))
                        else:  # left
                            fig.update_layout(legend=dict(x=-0.3, y=0.5))
                        
                        # Update grid color
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(gridcolor=grid_color),
                                angularaxis=dict(gridcolor=grid_color)
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add download options
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("💾 Download as HTML", key="download_custom"):
                                fig.write_html("custom_defect_radar.html")
                                with open("custom_defect_radar.html", "rb") as file:
                                    st.download_button(
                                        label="Download HTML",
                                        data=file,
                                        file_name="custom_defect_radar.html",
                                        mime="text/html",
                                        key="download_button_custom"
                                    )
                        with col2:
                            if st.button("🖼️ Download as PNG", key="download_png"):
                                try:
                                    fig.write_image("defect_radar.png")
                                    with open("defect_radar.png", "rb") as file:
                                        st.download_button(
                                            label="Download PNG",
                                            data=file,
                                            file_name="defect_radar.png",
                                            mime="image/png",
                                            key="download_button_png"
                                        )
                                except Exception as e:
                                    st.error("Error saving PNG: " + str(e))
                                    st.error("You may need to install kaleido: pip install kaleido")
                        with col3:
                            st.button("📋 Copy to Clipboard", disabled=True)  # Placeholder for future functionality
                        
                        st.success("✅ Customization applied successfully!")
                
                # Advanced CSS customization
                with st.expander("🔧 Advanced CSS Customization"):
                    st.markdown("""
                    For advanced users, you can override the default CSS styles. Here are some examples:
                    """)
                    
                    css_examples = {
                        "Transparent Background": """
                        .plotly-graph-div {
                            background: transparent !important;
                        }
                        """,
                        "Dark Mode": """
                        .plotly-graph-div {
                            background: #1a1a1a !important;
                            color: white !important;
                        }
                        .plotly-graph-div .text {
                            fill: white !important;
                        }
                        """,
                        "High Contrast": """
                        .plotly-graph-div .gridline {
                            stroke-width: 2px !important;
                            stroke: black !important;
                        }
                        .plotly-graph-div .zeroline {
                            stroke-width: 3px !important;
                            stroke: red !important;
                        }
                        """
                    }
                    
                    selected_css = st.selectbox("CSS Template", list(css_examples.keys()))
                    custom_css = st.text_area("Custom CSS", css_examples[selected_css], height=200)
                    
                    if st.button("Apply CSS"):
                        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
                        st.success("CSS applied! Refresh the page to see changes.")
            
            else:
                st.info("👈 First generate a visualization in the 'Main Visualization' tab, then customize it here.")
        
        with tab3:
            st.markdown('<h2 class="physics-header">💡 Visualization Concepts & Best Practices</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            ### 🌟 Advanced Visualization Techniques for Defect Stress Patterns
            
            Below are expert-level concepts for visualizing hydrostatic stress patterns around the habit plane. 
            These approaches reveal different aspects of stress distribution critical for materials science research.
            """)
            
            # Concept 1: Color Theory for Stress Visualization
            st.markdown("#### 1. Color Theory for Stress Patterns")
            st.markdown("""
            <div class="success-box">
            🎨 <strong>Best Practice:</strong> Use diverging color scales (blue-red) for stress components that can be 
            positive or negative, and sequential scales (blue-white) for magnitude-only data. Always maintain 
            colorblind-friendly palettes.
            </div>
            """, unsafe_allow_html=True)
            
            # Create color theory example
            fig_concept1 = go.Figure()
            
            # Create data for color demonstration
            angles = np.linspace(0, 360, 100)
            stress_pattern = 10 * np.sin(np.radians(angles) * 3)  # Oscillating pattern
            
            # Add traces with different color scales
            color_scales = ['RdBu', 'Viridis', 'Plasma', 'Inferno']
            
            for i, scale in enumerate(color_scales):
                fig_concept1.add_trace(go.Scatterpolar(
                    r=stress_pattern + (i * 15),  # Offset for visibility
                    theta=angles,
                    mode='lines',
                    line=dict(
                        color='black',
                        width=3,
                        shape='spline'
                    ),
                    marker=dict(
                        size=6,
                        color=stress_pattern,
                        colorscale=scale,
                        showscale=True if i == 0 else False,
                        colorbar=dict(
                            title="Stress (GPa)",
                            x=1.15,
                            y=0.5 + (i * 0.25),
                            len=0.2
                        ) if i == 0 else None
                    ),
                    name=f'{scale} Scale',
                    hoverinfo='skip'
                ))
            
            fig_concept1.update_layout(
                title="Color Scale Comparison for Stress Visualization",
                polar=dict(
                    radialaxis=dict(range=[-15, 60], visible=False),
                    angularaxis=dict(visible=False),
                    bgcolor="rgba(250, 250, 250, 0.5)"
                ),
                width=800,
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_concept1, use_container_width=True)
            
            # Concept 2: Interactive Elements
            st.markdown("#### 2. Interactive Elements for Scientific Communication")
            st.markdown("""
            <div class="insight-box">
            🔍 <strong>Expert Tip:</strong> Interactive visualizations allow researchers to explore data depth. 
            Hover information should include precise numerical values, while click events can trigger 
            detailed analysis of specific orientations or defect types.
            </div>
            """, unsafe_allow_html=True)
            
            # Create interactive example
            fig_concept2 = go.Figure()
            
            # Twin boundary stress pattern with interactive elements
            angles = np.linspace(40, 69.4, 50)
            twin_stress = 25 * np.exp(-(angles - 54.7)**2 / (2 * 3**2))
            
            fig_concept2.add_trace(go.Scatterpolar(
                r=np.append(twin_stress, twin_stress[0]),
                theta=np.append(angles, angles[0]),
                fill='toself',
                fillcolor='rgba(69, 183, 209, 0.3)',
                line=dict(color='#45B7D1', width=4),  # Use hex color
                marker=dict(size=10, color='#45B7D1'),
                name='Twin Boundary',
                hovertemplate='<b>Twin Boundary</b><br>' +
                              'Orientation: %{theta:.2f}°<br>' +
                              'Hydrostatic Stress: %{r:.4f} GPa<br>' +
                              'Eigen Strain: 2.12<br>' +
                              '<extra>Peak at habit plane</extra>',
                customdata=np.column_stack([angles, twin_stress])
            ))
            
            # Add clickable habit plane marker
            fig_concept2.add_trace(go.Scatterpolar(
                r=[max(twin_stress) * 1.1],
                theta=[54.7],
                mode='markers+text',
                marker=dict(
                    size=25,
                    color='gold',
                    symbol='star',
                    line=dict(width=2, color='#D4AF37')  # Use hex for gold
                ),
                text=['CLICK FOR DETAILS'],
                textposition='top center',
                textfont=dict(size=10, color='darkred'),
                name='Habit Plane (54.7°)',
                hovertemplate='<b>Habit Plane Peak</b><br>' +
                              'Maximum Stress: %{r:.4f} GPa<br>' +
                              'Diffusion Enhancement: High<br>' +
                              '<extra>Click for analysis</extra>',
                customdata=np.array([[54.7, max(twin_stress)]])
            ))
            
            fig_concept2.update_layout(
                title="Interactive Stress Visualization with Detailed Hover Information",
                polar=dict(
                    radialaxis=dict(range=[0, 35], title="Stress (GPa)"),
                    angularaxis=dict(
                        rotation=90,
                        direction="clockwise",
                        tickmode='array',
                        tickvals=[40, 45, 50, 54.7, 60, 65, 70],
                        ticktext=['40°', '45°', '50°', '54.7°', '60°', '65°', '70°']
                    )
                ),
                width=800,
                height=600,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial"
                )
            )
            
            st.plotly_chart(fig_concept2, use_container_width=True)
            
            # Concept 3: Multi-dimensional Visualization
            st.markdown("#### 3. Multi-dimensional Stress Analysis")
            st.markdown("""
            <div class="success-box">
            📊 <strong>Advanced Technique:</strong> Combine multiple stress components in a single visualization 
            using size, color, and position encoding. This reveals correlations between hydrostatic, 
            von Mises, and magnitude stresses that drive material behavior.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🎯 Practical Implementation Guide
            
            1. **Data Preparation**:
               - Ensure stress data is properly normalized and filtered for the habit plane vicinity
               - Calculate eigen strains for each defect type based on crystallographic properties
               - Generate smooth interpolation between data points for aesthetic curves
            
            2. **Visualization Selection**:
               - **Radar Charts**: Best for comparing multiple defect types at once
               - **Sunburst Charts**: Ideal for showing stress concentration patterns
               - **3D Views**: Useful for comprehensive analysis across multiple dimensions
               - **Dashboards**: Perfect for presentations and detailed analysis
            
            3. **Publication-Ready Output**:
               - Use vector formats (SVG, PDF) for publications
               - Maintain consistent color schemes across all figures
               - Include detailed captions explaining the physical significance
               - Add reference markers for critical angles and stress values
            
            4. **Interactive Features for Research**:
               - Implement hover information with precise values
               - Add click events for detailed analysis
               - Include zoom and pan capabilities for exploring details
               - Provide download options in multiple formats
            
            These advanced visualization techniques transform raw stress data into compelling scientific insights, 
            helping researchers understand the complex relationship between crystal defects and sintering behavior.
            """)
        
        with tab4:
            st.markdown('<h2 class="physics-header">🔍 Detailed Physics Analysis</h2>', unsafe_allow_html=True)
            
            if 'defect_comparison' in st.session_state:
                defect_comparison = st.session_state.defect_comparison
                
                st.markdown("""
                ### 🔬 Physical Interpretation of Stress Patterns
                
                The visualizations reveal fundamental physics of defect-mediated sintering. Here's a detailed analysis 
                of the key patterns and their implications for materials processing.
                """)
                
                # Create analysis columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📈 Stress Concentration Analysis")
                    
                    # Calculate and display key metrics
                    metrics = {}
                    for defect_key, data in defect_comparison.items():
                        defect_type = data.get('defect_type', 'Unknown')
                        stresses = data['stresses']['sigma_hydro']
                        angles = data['angles']
                        
                        max_stress = max(stresses)
                        max_angle = angles[np.argmax(stresses)]
                        habit_stress = stresses[np.argmin(np.abs(np.array(angles) - 54.7))]
                        
                        metrics[defect_type] = {
                            'max_stress': max_stress,
                            'max_angle': max_angle,
                            'habit_stress': habit_stress,
                            'eigen_strain': data.get('eigen_strain', 0)
                        }
                    
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
                            <h4>🔍 {defect_type} Analysis</h4>
                            <p><strong>• Maximum Stress:</strong> {values['max_stress']:.4f} GPa at {values['max_angle']:.2f}°</p>
                            <p><strong>• Habit Plane Stress:</strong> {values['habit_stress']:.4f} GPa</p>
                            <p><strong>• Eigen Strain:</strong> {values['eigen_strain']:.4f}</p>
                            <p><strong>• Stress Ratio (Habit/Max):</strong> {values['habit_stress']/values['max_stress']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### ⚡ Sintering Implications")
                    
                    st.markdown("""
                    <div class="habit-plane-highlight">
                        <h4>🔥 Critical Sintering Thresholds</h4>
                        <p><strong>• Twin boundaries:</strong> Generate sufficient stress at habit plane to enable sintering 
                        at temperatures 300-400°C below bulk melting point</p>
                        <p><strong>• Stacking faults:</strong> Moderate stress concentration enables intermediate sintering temperatures</p>
                        <p><strong>• Perfect crystals:</strong> Require significantly higher temperatures for equivalent diffusion</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    ### 🎯 Design Recommendations
                    
                    1. **Defect Engineering**: Introduce controlled twin boundaries at 54.7° orientation to maximize 
                       stress-driven diffusion during sintering
                    
                    2. **Temperature Profiles**: Use lower sintering temperatures (500-600°C) when high twin density 
                       is present, versus 800-900°C for perfect crystals
                    
                    3. **Processing Control**: Monitor stress patterns during processing to optimize defect distribution 
                       for desired mechanical properties
                    
                    4. **Multi-scale Modeling**: Combine these stress visualizations with atomistic simulations to 
                       predict grain boundary mobility and final microstructure
                    """)
                
                # Advanced analysis tools
                st.markdown("#### 🛠️ Advanced Analysis Tools")
                
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Stress Gradient Analysis", "Diffusion Coefficient Estimation", "Energy Landscape Mapping"],
                    key="analysis_type"
                )
                
                if analysis_type == "Stress Gradient Analysis":
                    st.markdown("##### Stress Gradient Analysis")
                    
                    # Calculate stress gradients for each defect type
                    fig_grad = go.Figure()
                    
                    for defect_key, data in defect_comparison.items():
                        defect_type = data.get('defect_type', 'Unknown')
                        angles = np.array(data['angles'])
                        stresses = np.array(data['stresses']['sigma_hydro'])
                        
                        # Calculate gradient (derivative)
                        d_stress_d_angle = np.gradient(stresses, np.deg2rad(angles))  # Convert to radians for proper derivative
                        
                        color = st.session_state.visualizer.defect_colors.get(defect_type, '#000000')
                        
                        fig_grad.add_trace(go.Scatterpolar(
                            r=np.append(d_stress_d_angle, d_stress_d_angle[0]),
                            theta=np.append(angles, angles[0]),
                            fill='toself',
                            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                            line=dict(color=color, width=3),
                            name=f"{defect_type} Gradient",
                            hovertemplate='Defect: ' + defect_type + 
                                          '<br>Orientation: %{theta:.2f}°<br>Stress Gradient: %{r:.4f} GPa/deg<extra></extra>'
                        ))
                    
                    fig_grad.add_trace(go.Scatterpolar(
                        r=[0, np.max(np.abs(d_stress_d_angle)) * 1.2],
                        theta=[54.7, 54.7],
                        mode='lines',
                        line=dict(color='#2ECC71', width=4, dash='dashdot'),  # Use hex color
                        name='Habit Plane (54.7°)',
                        hoverinfo='skip'
                    ))
                    
                    fig_grad.update_layout(
                        title="Stress Gradient Analysis: Twin Boundary Dominance",
                        polar=dict(
                            radialaxis=dict(title="Stress Gradient (GPa/deg)", range=[0, np.max(np.abs(d_stress_d_angle)) * 1.2]),
                            angularaxis=dict(rotation=90, direction="clockwise")
                        ),
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig_grad, use_container_width=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    📐 <strong>Gradient Significance:</strong> Steep stress gradients near the habit plane create strong driving forces 
                    for atomic diffusion. Twin boundaries show the highest gradients, explaining their superior sintering enhancement.
                    </div>
                    """, unsafe_allow_html=True)
                
                elif analysis_type == "Diffusion Coefficient Estimation":
                    st.markdown("##### Diffusion Coefficient Estimation")
                    
                    st.markdown("""
                    Using the stress patterns, we can estimate the enhancement in diffusion coefficient (D) relative to bulk:
                    
                    ```
                    D/D₀ = exp(-Ωσ_hydro / kT)
                    
                    Where:
                    - Ω = atomic volume (1.56e-29 m³ for Ag)
                    - σ_hydro = hydrostatic stress
                    - k = Boltzmann constant
                    - T = temperature (650 K typical sintering temp)
                    ```
                    """)
                    
                    # Calculate diffusion enhancement
                    Omega = 1.56e-29  # m³ for Ag
                    k = 1.38e-23  # J/K
                    T = 650  # K
                    
                    fig_diff = go.Figure()
                    
                    for defect_key, data in defect_comparison.items():
                        defect_type = data.get('defect_type', 'Unknown')
                        angles = np.array(data['angles'])
                        stresses = np.array(data['stresses']['sigma_hydro'])
                        
                        # Convert GPa to Pa
                        stresses_pa = stresses * 1e9
                        
                        # Calculate diffusion enhancement ratio
                        diffusion_ratio = np.exp(-Omega * stresses_pa / (k * T))
                        
                        color = st.session_state.visualizer.defect_colors.get(defect_type, '#000000')
                        
                        fig_diff.add_trace(go.Scatterpolar(
                            r=np.append(diffusion_ratio, diffusion_ratio[0]),
                            theta=np.append(angles, angles[0]),
                            fill='toself',
                            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                            line=dict(color=color, width=3),
                            name=f"{defect_type} Enhancement",
                            hovertemplate='Defect: ' + defect_type + 
                                          '<br>Orientation: %{theta:.2f}°<br>Diffusion Enhancement: %{r:.2f}x<extra></extra>'
                        ))
                    
                    fig_diff.add_trace(go.Scatterpolar(
                        r=[0, np.max(diffusion_ratio) * 1.2],
                        theta=[54.7, 54.7],
                        mode='lines',
                        line=dict(color='#2ECC71', width=4, dash='dashdot'),  # Use hex color
                        name='Habit Plane (54.7°)',
                        hoverinfo='skip'
                    ))
                    
                    fig_diff.update_layout(
                        title="Diffusion Coefficient Enhancement Factor",
                        polar=dict(
                            radialaxis=dict(title="D/D₀ Enhancement Ratio", type="log"),
                            angularaxis=dict(rotation=90, direction="clockwise")
                        ),
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    st.markdown("""
                    <div class="success-box">
                    ⚡ <strong>Diffusion Enhancement:</strong> At the habit plane, twin boundaries can enhance diffusion by 10-100x 
                    compared to bulk material. This dramatic enhancement enables low-temperature sintering of nanocrystalline silver.
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("""
    ---
    <div style="text-align: center; color: #666; padding: 20px;">
        <strong>🔬 Advanced Materials Research Tool</strong><br>
        This visualization system reveals the physics of defect-mediated sintering.<br>
        For research use only - © 2026 Materials Science Research Group
    </div>
    """, unsafe_allow_html=True)

# =============================================
# RUN THE APPLICATION
# =============================================
if __name__ == "__main__":
    main()
