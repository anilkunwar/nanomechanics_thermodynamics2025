import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
import warnings
import pickle
import torch
import sqlite3
from pathlib import Path
import tempfile
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import re
from scipy.interpolate import interp1d
import torch.nn as nn
from matplotlib.colors import Normalize, LogNorm
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
FIGURE_DIR = os.path.join(SCRIPT_DIR, "figures")
DB_PATH = os.path.join(SCRIPT_DIR, "sunburst_data.db")
if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR, exist_ok=True)

# =============================================
# CUSTOM COLORMAPS
# =============================================
def create_custom_colormaps():
    """Create custom colormaps for stress visualization"""
    # Stress colormap (blue to red)
    stress_cmap = LinearSegmentedColormap.from_list(
        'stress_cmap',
        ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
    )
  
    # Attention weight colormap
    attention_cmap = LinearSegmentedColormap.from_list(
        'attention_cmap',
        ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    )
  
    # Comparison colormap (diverging)
    compare_cmap = LinearSegmentedColormap.from_list(
        'compare_cmap',
        ['#0066CC', '#66CCFF', '#FFFFFF', '#FF9999', '#CC0000']
    )
  
    # Sunburst specific colormaps
    sunburst_cmap = LinearSegmentedColormap.from_list(
        'sunburst_cmap',
        ['#1a1334', '#26294a', '#01545a', '#017351', '#03c383', 
         '#aad962', '#fbbf45', '#ef6a32', '#ed0345', '#a12a5e', '#710162']
    )
    
    return stress_cmap, attention_cmap, compare_cmap, sunburst_cmap

# =============================================
# EXTENDED COLORMAPS FOR SUNBURST
# =============================================
EXTENDED_CMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
    'twilight', 'twilight_shifted', 'hsv',
    'tab10', 'tab20', 'Set1', 'Set2', 'Set3',
    'jet', 'turbo', 'nipy_spectral', 'gist_ncar', 'gist_rainbow'
]

# =============================================
# PREDICTION SUNBURST VISUALIZATION MANAGER
# =============================================
class PredictionSunburstVisualizer:
    """
    Specialized visualizer for creating sunburst and radar charts
    that showcase the purpose of prediction visualization
    """
    
    def __init__(self, interpolator=None):
        self.interpolator = interpolator
        self.stress_cmap, self.attention_cmap, self.compare_cmap, self.sunburst_cmap = create_custom_colormaps()
        
    def generate_prediction_sunburst_data(self, source_simulations, defect_types=None, 
                                         theta_range=None, stress_components=None):
        """
        Generate comprehensive prediction data for sunburst visualization
        
        Args:
            source_simulations: List of source simulation data
            defect_types: List of defect types to analyze (ISF, ESF, Twin)
            theta_range: Range of orientations in degrees
            stress_components: List of stress components to analyze
            
        Returns:
            Dictionary with organized prediction data
        """
        if defect_types is None:
            defect_types = ['ISF', 'ESF', 'Twin']
        
        if theta_range is None:
            theta_range = np.arange(0, 91, 10)  # 0 to 90 degrees in 10° steps
        
        if stress_components is None:
            stress_components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        
        # Convert to radians
        theta_rad = np.deg2rad(theta_range)
        
        # Initialize data structure
        sunburst_data = {
            'defect_types': defect_types,
            'theta_degrees': theta_range,
            'theta_radians': theta_rad,
            'stress_components': stress_components,
            'predictions': {},
            'statistics': {}
        }
        
        # Generate predictions for each combination
        for defect_type in defect_types:
            sunburst_data['predictions'][defect_type] = {}
            sunburst_data['statistics'][defect_type] = {}
            
            for theta_deg, theta_rad in zip(theta_range, theta_rad):
                # Create target parameters
                target_params = {
                    'defect_type': defect_type,
                    'theta': float(theta_rad),
                    'orientation': self.interpolator.get_orientation_from_angle(theta_deg),
                    'shape': 'Square',  # Default shape
                    'eps0': 0.707,  # Default material parameter
                    'kappa': 0.6  # Default material parameter
                }
                
                # Generate prediction
                prediction = self._generate_prediction(source_simulations, target_params)
                
                if prediction:
                    # Store prediction
                    key = f"theta_{theta_deg:.0f}"
                    sunburst_data['predictions'][defect_type][key] = prediction
                    
                    # Calculate and store statistics
                    stats = self._calculate_stress_statistics(prediction)
                    sunburst_data['statistics'][defect_type][key] = stats
        
        return sunburst_data
    
    def _generate_prediction(self, source_simulations, target_params):
        """Generate prediction using attention interpolation"""
        try:
            # Prepare source data
            source_param_vectors = []
            source_stress_data = []
            
            for sim_data in source_simulations:
                param_vector, _ = self.interpolator.compute_parameter_vector(sim_data)
                source_param_vectors.append(param_vector)
                
                # Get stress from final frame
                history = sim_data.get('history', [])
                if history:
                    eta, stress_fields = history[-1]
                    stress_components = np.stack([
                        stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                        stress_fields.get('sigma_mag', np.zeros_like(eta)),
                        stress_fields.get('von_mises', np.zeros_like(eta))
                    ], axis=0)
                    source_stress_data.append(stress_components)
            
            if not source_param_vectors:
                return None
            
            source_param_vectors = np.array(source_param_vectors)
            source_stress_data = np.array(source_stress_data)
            
            # Compute target parameter vector
            target_vector, _ = self.interpolator.compute_parameter_vector(
                {'params': target_params}
            )
            
            # Calculate distances and attention weights
            distances = np.sqrt(np.sum((source_param_vectors - target_vector) ** 2, axis=1))
            weights = np.exp(-0.5 * (distances / 0.3) ** 2)
            weights = weights / (np.sum(weights) + 1e-8)
            
            # Weighted combination
            weighted_stress = np.sum(
                source_stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis],
                axis=0
            )
            
            return {
                'sigma_hydro': weighted_stress[0],
                'sigma_mag': weighted_stress[1],
                'von_mises': weighted_stress[2],
                'target_params': target_params,
                'attention_weights': weights,
                'source_count': len(source_simulations)
            }
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
            return None
    
    def _calculate_stress_statistics(self, prediction):
        """Calculate comprehensive stress statistics"""
        stats = {}
        
        for comp_name in ['sigma_hydro', 'sigma_mag', 'von_mises']:
            if comp_name in prediction:
                data = prediction[comp_name]
                stats[comp_name] = {
                    'max': float(np.max(data)),
                    'min': float(np.min(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'median': float(np.median(data)),
                    'percentile_95': float(np.percentile(data, 95)),
                    'percentile_99': float(np.percentile(data, 99)),
                    'area_above_threshold': float(np.sum(data > 0.5))  # Area above 0.5 GPa
                }
        
        return stats
    
    def create_comprehensive_sunburst_plot(self, sunburst_data, component='von_mises',
                                          title="Stress Field Prediction Sunburst"):
        """
        Create comprehensive sunburst plot showing stress predictions across
        defect types and orientations
        
        Args:
            sunburst_data: Dictionary with prediction data
            component: Stress component to visualize
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Prepare data for sunburst plot
        labels = []
        parents = []
        values = []
        colors = []
        custom_data = []
        
        # Root level
        root_label = "Stress Predictions"
        labels.append(root_label)
        parents.append("")
        values.append(0)  # Root doesn't have a value
        colors.append('#ffffff')
        custom_data.append({'type': 'root', 'info': 'All predictions'})
        
        # Defect type level
        for defect_type in sunburst_data['defect_types']:
            defect_label = f"Defect: {defect_type}"
            labels.append(defect_label)
            parents.append(root_label)
            
            # Calculate average stress for this defect type
            avg_stress = 0
            count = 0
            for theta_key, prediction in sunburst_data['predictions'][defect_type].items():
                if component in prediction:
                    avg_stress += np.mean(prediction[component])
                    count += 1
            
            if count > 0:
                values.append(avg_stress / count)
            else:
                values.append(0)
            
            # Color based on defect type
            defect_colors = {'ISF': '#FF6B6B', 'ESF': '#4ECDC4', 'Twin': '#45B7D1'}
            colors.append(defect_colors.get(defect_type, '#95A5A6'))
            custom_data.append({'type': 'defect', 'defect': defect_type})
            
            # Orientation level
            for theta_deg in sunburst_data['theta_degrees']:
                theta_key = f"theta_{theta_deg:.0f}"
                if theta_key in sunburst_data['predictions'][defect_type]:
                    prediction = sunburst_data['predictions'][defect_type][theta_key]
                    
                    if component in prediction:
                        # Orientation label
                        orient_label = f"{theta_deg}°"
                        labels.append(orient_label)
                        parents.append(defect_label)
                        
                        # Calculate value (average stress at this orientation)
                        stress_value = float(np.mean(prediction[component]))
                        values.append(stress_value)
                        
                        # Color based on stress value
                        max_stress = max(values[2:]) if len(values) > 2 else 1
                        normalized_stress = stress_value / max_stress if max_stress > 0 else 0
                        color_idx = int(normalized_stress * (len(EXTENDED_CMAPS) - 1))
                        colors.append(px.colors.sequential.Plasma[color_idx % len(px.colors.sequential.Plasma)])
                        
                        # Custom data for hover
                        stats = sunburst_data['statistics'][defect_type].get(theta_key, {})
                        comp_stats = stats.get(component, {})
                        custom_data.append({
                            'type': 'orientation',
                            'defect': defect_type,
                            'theta': theta_deg,
                            'stress_value': stress_value,
                            'max_stress': comp_stats.get('max', 0),
                            'mean_stress': comp_stats.get('mean', 0),
                            'area_above_threshold': comp_stats.get('area_above_threshold', 0)
                        })
        
        # Create sunburst plot
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                colorscale='Plasma',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Stress: %{value:.3f} GPa<br>' +
                         '%{customdata.info}<extra></extra>',
            customdata=custom_data,
            maxdepth=3
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br>{component.replace('_', ' ').title()}",
                font=dict(size=24, family="Arial", color="black"),
                x=0.5,
                y=0.95
            ),
            margin=dict(l=20, r=20, t=100, b=20),
            height=700,
            showlegend=False
        )
        
        return fig
    
    def create_radar_comparison_plot(self, sunburst_data, defect_types=None,
                                    theta_values=None, component='von_mises',
                                    title="Stress Component Radar Comparison"):
        """
        Create radar plot comparing stress components across defect types
        and orientations
        
        Args:
            sunburst_data: Dictionary with prediction data
            defect_types: List of defect types to include
            theta_values: List of specific theta values to compare
            component: Stress component to visualize
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if defect_types is None:
            defect_types = sunburst_data['defect_types']
        
        if theta_values is None:
            theta_values = [0, 30, 60, 90]
        
        # Prepare data for radar plot
        categories = [f"{theta}°" for theta in theta_values]
        
        fig = go.Figure()
        
        # Color mapping for defect types
        defect_colors = {
            'ISF': '#FF6B6B',
            'ESF': '#4ECDC4',
            'Twin': '#45B7D1'
        }
        
        # Plot each defect type
        for defect_type in defect_types:
            if defect_type in sunburst_data['predictions']:
                stress_values = []
                
                for theta in theta_values:
                    theta_key = f"theta_{theta:.0f}"
                    if theta_key in sunburst_data['predictions'][defect_type]:
                        prediction = sunburst_data['predictions'][defect_type][theta_key]
                        if component in prediction:
                            stress_values.append(float(np.mean(prediction[component])))
                        else:
                            stress_values.append(0)
                    else:
                        stress_values.append(0)
                
                # Close the radar loop
                stress_values.append(stress_values[0])
                categories_cyclic = categories + [categories[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=stress_values,
                    theta=categories_cyclic,
                    name=defect_type,
                    line=dict(color=defect_colors.get(defect_type, '#95A5A6'), width=3),
                    marker=dict(size=8),
                    fill='toself',
                    opacity=0.6,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Orientation: %{theta}<br>' +
                                 'Stress: %{r:.3f} GPa<br>' +
                                 '<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br>{component.replace('_', ' ').title()}",
                font=dict(size=20, family="Arial", color="black"),
                x=0.5,
                y=0.95
            ),
            polar=dict(
                radialaxis=dict(
                    title=dict(text="Stress (GPa)", font=dict(size=14)),
                    tickfont=dict(size=12),
                    gridcolor="lightgray",
                    linecolor="black",
                    showline=True,
                    angle=90
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    gridcolor="lightgray",
                    linecolor="black",
                    rotation=90,
                    direction="clockwise"
                ),
                bgcolor="white"
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            height=600,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        return fig
    
    def create_orientation_sweep_plot(self, sunburst_data, defect_type='ISF',
                                     components=None, title="Orientation Sweep Analysis"):
        """
        Create comprehensive plot showing stress evolution across orientations
        for different stress components
        
        Args:
            sunburst_data: Dictionary with prediction data
            defect_type: Specific defect type to analyze
            components: List of stress components to include
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if components is None:
            components = ['von_mises', 'sigma_hydro', 'sigma_mag']
        
        # Prepare data
        theta_values = sunburst_data['theta_degrees']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Von Mises Stress',
                'Hydrostatic Stress',
                'Stress Magnitude',
                'Component Comparison'
            ),
            specs=[[{'type': 'polar'}, {'type': 'polar'}],
                  [{'type': 'polar'}, {'type': 'polar'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Color mapping for components
        component_colors = {
            'von_mises': '#FF6B6B',
            'sigma_hydro': '#4ECDC4',
            'sigma_mag': '#45B7D1'
        }
        
        component_names = {
            'von_mises': 'Von Mises',
            'sigma_hydro': 'Hydrostatic',
            'sigma_mag': 'Magnitude'
        }
        
        # Plot each component in its own subplot
        for idx, component in enumerate(components[:3]):  # First three components
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            if defect_type in sunburst_data['predictions']:
                stress_values = []
                
                for theta in theta_values:
                    theta_key = f"theta_{theta:.0f}"
                    if theta_key in sunburst_data['predictions'][defect_type]:
                        prediction = sunburst_data['predictions'][defect_type][theta_key]
                        if component in prediction:
                            stress_values.append(float(np.mean(prediction[component])))
                        else:
                            stress_values.append(0)
                    else:
                        stress_values.append(0)
                
                # Convert to cyclic for polar plot
                stress_cyclic = stress_values + [stress_values[0]]
                theta_cyclic = list(theta_values) + [theta_values[0]]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=stress_cyclic,
                        theta=theta_cyclic,
                        mode='lines+markers',
                        name=component_names[component],
                        line=dict(color=component_colors[component], width=3),
                        marker=dict(size=6),
                        fill='toself',
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Combined comparison in bottom right
        row, col = 2, 2
        
        for component in components:
            if defect_type in sunburst_data['predictions']:
                stress_values = []
                
                for theta in theta_values:
                    theta_key = f"theta_{theta:.0f}"
                    if theta_key in sunburst_data['predictions'][defect_type]:
                        prediction = sunburst_data['predictions'][defect_type][theta_key]
                        if component in prediction:
                            stress_values.append(float(np.mean(prediction[component])))
                        else:
                            stress_values.append(0)
                    else:
                        stress_values.append(0)
                
                # Convert to cyclic for polar plot
                stress_cyclic = stress_values + [stress_values[0]]
                theta_cyclic = list(theta_values) + [theta_values[0]]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=stress_cyclic,
                        theta=theta_cyclic,
                        mode='lines',
                        name=component_names[component],
                        line=dict(color=component_colors[component], width=2),
                        opacity=0.8,
                        showlegend=True
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title} - {defect_type}",
                font=dict(size=24, family="Arial", color="black"),
                x=0.5,
                y=0.98
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=800,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update polar subplots
        for i in range(1, 5):
            fig.update_polars(
                dict(
                    radialaxis=dict(
                        title=dict(text="Stress (GPa)", font=dict(size=10)),
                        tickfont=dict(size=8),
                        gridcolor="lightgray"
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=8),
                        gridcolor="lightgray",
                        rotation=90
                    )
                ),
                row=(i-1)//2 + 1,
                col=(i-1)%2 + 1
            )
        
        return fig
    
    def create_prediction_summary_dashboard(self, sunburst_data):
        """
        Create comprehensive dashboard showing prediction summary
        with multiple visualization types
        
        Args:
            sunburst_data: Dictionary with prediction data
            
        Returns:
            Dictionary of Plotly figures
        """
        dashboard = {}
        
        # 1. Sunburst plot for Von Mises
        dashboard['sunburst_von_mises'] = self.create_comprehensive_sunburst_plot(
            sunburst_data, component='von_mises',
            title="Stress Field Prediction Analysis"
        )
        
        # 2. Radar comparison for all defect types
        dashboard['radar_comparison'] = self.create_radar_comparison_plot(
            sunburst_data, defect_types=sunburst_data['defect_types'],
            theta_values=[0, 30, 60, 90], component='von_mises',
            title="Defect Type Comparison"
        )
        
        # 3. Orientation sweep for each defect type
        for defect_type in sunburst_data['defect_types']:
            dashboard[f'orientation_sweep_{defect_type}'] = self.create_orientation_sweep_plot(
                sunburst_data, defect_type=defect_type,
                title=f"Orientation Analysis"
            )
        
        # 4. Statistical summary table
        dashboard['statistics_table'] = self._create_statistics_table(sunburst_data)
        
        # 5. Stress distribution across orientations
        dashboard['stress_distribution'] = self._create_stress_distribution_plot(sunburst_data)
        
        return dashboard
    
    def _create_statistics_table(self, sunburst_data):
        """Create statistics table for predictions"""
        stats_data = []
        
        for defect_type in sunburst_data['defect_types']:
            for theta_deg in sunburst_data['theta_degrees']:
                theta_key = f"theta_{theta_deg:.0f}"
                
                if (defect_type in sunburst_data['statistics'] and 
                    theta_key in sunburst_data['statistics'][defect_type]):
                    
                    stats = sunburst_data['statistics'][defect_type][theta_key]
                    
                    row = {
                        'Defect Type': defect_type,
                        'Orientation (°)': theta_deg,
                    }
                    
                    # Add statistics for each component
                    for comp in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                        if comp in stats:
                            comp_stats = stats[comp]
                            row[f'{comp}_max'] = f"{comp_stats['max']:.3f}"
                            row[f'{comp}_mean'] = f"{comp_stats['mean']:.3f}"
                            row[f'{comp}_std'] = f"{comp_stats['std']:.3f}"
                    
                    stats_data.append(row)
        
        # Create DataFrame
        df_stats = pd.DataFrame(stats_data)
        
        # Create interactive table with Plotly
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_stats.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df_stats[col] for col in df_stats.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="Prediction Statistics Summary",
                font=dict(size=16, family="Arial", color="black"),
                x=0.5,
                y=0.95
            ),
            height=400,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig
    
    def _create_stress_distribution_plot(self, sunburst_data):
        """Create plot showing stress distribution across orientations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Maximum Stress',
                'Mean Stress',
                'Standard Deviation',
                'Area Above Threshold'
            ),
            shared_xaxes=True
        )
        
        # Color mapping for defect types
        defect_colors = {
            'ISF': '#FF6B6B',
            'ESF': '#4ECDC4',
            'Twin': '#45B7D1'
        }
        
        theta_values = sunburst_data['theta_degrees']
        
        # Plot for each metric
        metrics = ['max', 'mean', 'std', 'area_above_threshold']
        titles = ['Maximum Stress (GPa)', 'Mean Stress (GPa)', 
                 'Standard Deviation (GPa)', 'Area Above 0.5 GPa']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            for defect_type in sunburst_data['defect_types']:
                values = []
                
                for theta_deg in theta_values:
                    theta_key = f"theta_{theta_deg:.0f}"
                    
                    if (defect_type in sunburst_data['statistics'] and 
                        theta_key in sunburst_data['statistics'][defect_type]):
                        
                        stats = sunburst_data['statistics'][defect_type][theta_key]
                        if 'von_mises' in stats:
                            values.append(stats['von_mises'].get(metric, 0))
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                
                fig.add_trace(
                    go.Scatter(
                        x=theta_values,
                        y=values,
                        mode='lines+markers',
                        name=defect_type,
                        line=dict(color=defect_colors[defect_type], width=2),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{defect_type}</b><br>' +
                                     'Orientation: %{x}°<br>' +
                                     f'{title}: %{{y:.3f}}<br>' +
                                     '<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Stress Distribution Analysis",
                font=dict(size=20, family="Arial", color="black"),
                x=0.5,
                y=0.98
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=600,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Orientation (°)", row=2, col=1)
        fig.update_xaxes(title_text="Orientation (°)", row=2, col=2)
        fig.update_yaxes(title_text="Stress (GPa)", row=1, col=1)
        fig.update_yaxes(title_text="Stress (GPa)", row=1, col=2)
        fig.update_yaxes(title_text="Std Dev (GPa)", row=2, col=1)
        fig.update_yaxes(title_text="Area (pixels)", row=2, col=2)
        
        return fig

# =============================================
# ENHANCED SUNBURST TAB WITH PREDICTION VISUALIZATION
# =============================================
def create_sunburst_prediction_tab():
    """
    Create comprehensive sunburst and radar visualization tab
    that showcases the purpose of prediction visualization
    """
    st.header("🌅 Prediction Sunburst & Radar Visualization")
    
    # Check if we have source simulations
    if 'source_simulations' not in st.session_state or not st.session_state.source_simulations:
        st.warning("⚠️ Please load source simulations first in the '📤 Load Source Data' tab.")
        st.info("""
        ### Purpose of This Visualization:
        
        This tab demonstrates the **power of attention-based interpolation** by showing:
        
        1. **Orientation Dependence**: How stress fields change with different crystal orientations
        2. **Defect Type Comparison**: Differences between ISF, ESF, and Twin defects
        3. **Stress Component Analysis**: Von Mises, Hydrostatic, and Magnitude stress variations
        4. **Prediction Confidence**: Visualization of interpolation quality across parameter space
        
        ### How It Works:
        
        - Uses attention weights from your trained model
        - Interpolates stress fields for orientations not in your training data
        - Creates comprehensive visualizations showing the entire parameter space
        - Helps identify critical orientations and defect configurations
        
        **Please load at least 2 source simulations to begin.**
        """)
        return
    
    # Initialize visualizer
    if 'sunburst_visualizer' not in st.session_state:
        st.session_state.sunburst_visualizer = PredictionSunburstVisualizer(
            interpolator=st.session_state.interpolator
        )
    
    # Sidebar configuration
    st.sidebar.header("🌅 Sunburst & Radar Settings")
    
    with st.sidebar.expander("🎯 Prediction Parameters", expanded=True):
        # Defect types to analyze
        defect_types = st.multiselect(
            "Defect Types to Analyze",
            ["ISF", "ESF", "Twin"],
            default=["ISF", "ESF", "Twin"],
            help="Select defect types for comparison"
        )
        
        # Orientation range
        col1, col2, col3 = st.columns(3)
        with col1:
            theta_min = st.number_input("Min Angle (°)", 0.0, 90.0, 0.0, 1.0)
        with col2:
            theta_max = st.number_input("Max Angle (°)", 0.0, 90.0, 90.0, 1.0)
        with col3:
            theta_step = st.number_input("Step (°)", 1.0, 30.0, 10.0, 1.0)
        
        # Material parameters
        eps0 = st.slider(
            "ε* (Strain Amplitude)",
            min_value=0.3,
            max_value=3.0,
            value=0.707,
            step=0.1,
            help="Material parameter for prediction"
        )
        
        kappa = st.slider(
            "κ (Shape Parameter)",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.05,
            help="Material parameter for prediction"
        )
    
    with st.sidebar.expander("📊 Visualization Options", expanded=True):
        # Stress components to visualize
        stress_components = st.multiselect(
            "Stress Components",
            ["von_mises", "sigma_hydro", "sigma_mag"],
            default=["von_mises", "sigma_hydro", "sigma_mag"],
            help="Select stress components to visualize"
        )
        
        # Visualization mode
        viz_mode = st.radio(
            "Visualization Mode",
            ["Comprehensive Dashboard", "Sunburst Only", "Radar Only", "Orientation Sweep"],
            index=0,
            help="Choose the type of visualization"
        )
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Plasma", "Viridis", "Inferno", "Magma", "Rainbow", "Custom"],
            index=0
        )
        
        # Interactive features
        show_hover = st.checkbox("Show Detailed Hover Info", value=True)
        show_legend = st.checkbox("Show Legend", value=True)
        auto_scale = st.checkbox("Auto-scale Colors", value=True)
    
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        # Prediction quality
        quality_level = st.slider(
            "Prediction Quality",
            min_value=1,
            max_value=10,
            value=7,
            step=1,
            help="Higher values = more accurate but slower"
        )
        
        # Cache settings
        use_cache = st.checkbox("Use Prediction Cache", value=True)
        clear_cache = st.button("Clear Cache")
        
        if clear_cache:
            if 'sunburst_data' in st.session_state:
                del st.session_state.sunburst_data
            st.success("Cache cleared!")
    
    # Main content area
    st.subheader("Generate Prediction Visualizations")
    
    col_gen1, col_gen2 = st.columns([3, 1])
    
    with col_gen1:
        if st.button("🚀 Generate Comprehensive Predictions", type="primary", use_container_width=True):
            with st.spinner("Generating predictions across defect types and orientations..."):
                try:
                    # Generate theta range
                    theta_range = np.arange(theta_min, theta_max + theta_step/2, theta_step)
                    
                    # Generate sunburst data
                    sunburst_data = st.session_state.sunburst_visualizer.generate_prediction_sunburst_data(
                        source_simulations=st.session_state.source_simulations,
                        defect_types=defect_types,
                        theta_range=theta_range,
                        stress_components=stress_components
                    )
                    
                    # Store in session state
                    st.session_state.sunburst_data = sunburst_data
                    st.session_state.sunburst_params = {
                        'defect_types': defect_types,
                        'theta_range': theta_range,
                        'eps0': eps0,
                        'kappa': kappa,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    st.success(f"✅ Generated predictions for {len(defect_types)} defect types and {len(theta_range)} orientations!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
    
    with col_gen2:
        if 'sunburst_data' in st.session_state:
            data = st.session_state.sunburst_data
            st.metric("Defect Types", len(data['defect_types']))
            st.metric("Orientations", len(data['theta_degrees']))
            st.metric("Predictions", len(data['defect_types']) * len(data['theta_degrees']))
    
    # Display visualizations if data exists
    if 'sunburst_data' in st.session_state:
        sunburst_data = st.session_state.sunburst_data
        
        # Information box about what's being shown
        with st.expander("📋 Prediction Summary", expanded=True):
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Total Predictions", 
                         len(sunburst_data['defect_types']) * len(sunburst_data['theta_degrees']))
            
            with col_info2:
                avg_vm = 0
                count = 0
                for defect_type in sunburst_data['defect_types']:
                    for theta_key in sunburst_data['predictions'][defect_type]:
                        if 'von_mises' in sunburst_data['predictions'][defect_type][theta_key]:
                            avg_vm += np.mean(sunburst_data['predictions'][defect_type][theta_key]['von_mises'])
                            count += 1
                
                if count > 0:
                    st.metric("Avg Von Mises", f"{avg_vm/count:.3f} GPa")
            
            with col_info3:
                # Calculate prediction confidence (based on attention weight variance)
                confidence_scores = []
                for defect_type in sunburst_data['defect_types']:
                    for theta_key in sunburst_data['predictions'][defect_type]:
                        pred = sunburst_data['predictions'][defect_type][theta_key]
                        if 'attention_weights' in pred:
                            weights = pred['attention_weights']
                            # Confidence is higher when weights are more focused
                            confidence = 1 - np.std(weights) / (np.mean(weights) + 1e-8)
                            confidence_scores.append(confidence)
                
                if confidence_scores:
                    avg_confidence = np.mean(confidence_scores) * 100
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Main visualization section based on selected mode
        if viz_mode == "Comprehensive Dashboard":
            st.subheader("📊 Comprehensive Prediction Dashboard")
            
            # Generate dashboard
            dashboard = st.session_state.sunburst_visualizer.create_prediction_summary_dashboard(
                sunburst_data
            )
            
            # Display all dashboard components
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🌅 Sunburst Overview",
                "📡 Radar Comparison",
                "📈 Orientation Analysis",
                "📊 Statistics",
                "📉 Distribution"
            ])
            
            with tab1:
                st.plotly_chart(dashboard['sunburst_von_mises'], use_container_width=True)
                st.caption("**Sunburst Chart**: Hierarchical view of stress predictions showing defect types (inner ring) and orientations (outer ring). Color intensity indicates stress magnitude.")
            
            with tab2:
                st.plotly_chart(dashboard['radar_comparison'], use_container_width=True)
                st.caption("**Radar Comparison**: Direct comparison of stress values across defect types at key orientations (0°, 30°, 60°, 90°).")
            
            with tab3:
                # Show orientation sweep for each defect type
                selected_defect = st.selectbox(
                    "Select Defect Type for Detailed Analysis",
                    sunburst_data['defect_types'],
                    index=0
                )
                
                if f'orientation_sweep_{selected_defect}' in dashboard:
                    st.plotly_chart(dashboard[f'orientation_sweep_{selected_defect}'], use_container_width=True)
                    st.caption(f"**Orientation Sweep Analysis for {selected_defect}**: Detailed view showing how different stress components vary with orientation.")
            
            with tab4:
                st.plotly_chart(dashboard['statistics_table'], use_container_width=True)
                st.caption("**Statistics Table**: Numerical summary of predictions including maximum, mean, and standard deviation of stress values.")
            
            with tab5:
                st.plotly_chart(dashboard['stress_distribution'], use_container_width=True)
                st.caption("**Stress Distribution**: Analysis of how stress metrics (maximum, mean, standard deviation, high-stress area) vary with orientation.")
        
        elif viz_mode == "Sunburst Only":
            st.subheader("🌅 Sunburst Visualization")
            
            # Component selection for sunburst
            sunburst_component = st.selectbox(
                "Select Stress Component for Sunburst",
                stress_components,
                index=0,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Generate sunburst
            fig_sunburst = st.session_state.sunburst_visualizer.create_comprehensive_sunburst_plot(
                sunburst_data, component=sunburst_component,
                title="Stress Prediction Hierarchy"
            )
            
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            # Sunburst interpretation guide
            with st.expander("🎓 How to Interpret This Sunburst Chart"):
                st.markdown("""
                ### Understanding the Sunburst Visualization:
                
                **Hierarchy Levels:**
                1. **Center**: All predictions
                2. **Middle Ring**: Defect types (ISF, ESF, Twin)
                3. **Outer Ring**: Orientations (0° to 90°)
                
                **Color Coding:**
                - **Defect Types**: Fixed colors for easy identification
                - **Orientations**: Color intensity shows stress magnitude
                - **Hover**: Detailed information for each segment
                
                **Key Insights:**
                - Larger segments = higher average stress
                - Color gradient = stress variation with orientation
                - Segment size = relative importance/impact
                
                **Interpreting Results:**
                - Look for **bright outer rings** = high stress at those orientations
                - Compare **segment sizes** between defect types
                - Identify **orientation clusters** with similar stress levels
                """)
        
        elif viz_mode == "Radar Only":
            st.subheader("📡 Radar Chart Comparison")
            
            # Configuration for radar
            col_radar1, col_radar2 = st.columns(2)
            
            with col_radar1:
                radar_component = st.selectbox(
                    "Stress Component for Radar",
                    stress_components,
                    index=0,
                    key="radar_component"
                )
            
            with col_radar2:
                radar_thetas = st.multiselect(
                    "Orientations to Include",
                    list(sunburst_data['theta_degrees']),
                    default=[0, 30, 60, 90]
                )
            
            # Generate radar
            fig_radar = st.session_state.sunburst_visualizer.create_radar_comparison_plot(
                sunburst_data, defect_types=defect_types,
                theta_values=radar_thetas, component=radar_component,
                title="Defect Type Comparison"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Radar interpretation
            with st.expander("🎯 Radar Chart Interpretation"):
                st.markdown("""
                ### Reading Radar Charts:
                
                **Radial Axis:**
                - Distance from center = stress magnitude
                - Further from center = higher stress
                
                **Angular Axis:**
                - Each spoke = specific orientation
                - Full circle = 0° to 360° (shown as 0° to 90°)
                
                **Shape Analysis:**
                - **Symmetrical shapes** = consistent stress across orientations
                - **Asymmetrical shapes** = orientation-dependent stress
                - **Larger area** = generally higher stress
                
                **Comparison Tips:**
                1. **Overlap areas** = similar stress patterns
                2. **Distinct shapes** = different mechanical behavior
                3. **Peak locations** = critical orientations
                
                **Practical Applications:**
                - Identify **worst-case orientations**
                - Compare **defect type sensitivity**
                - Guide **material design decisions**
                """)
        
        elif viz_mode == "Orientation Sweep":
            st.subheader("📈 Orientation Sweep Analysis")
            
            # Select defect type
            sweep_defect = st.selectbox(
                "Select Defect Type",
                sunburst_data['defect_types'],
                index=0,
                key="sweep_defect"
            )
            
            # Generate orientation sweep
            fig_sweep = st.session_state.sunburst_visualizer.create_orientation_sweep_plot(
                sunburst_data, defect_type=sweep_defect,
                components=stress_components,
                title=f"Orientation Analysis - {sweep_defect}"
            )
            
            st.plotly_chart(fig_sweep, use_container_width=True)
            
            # Orientation analysis insights
            with st.expander("🔬 Orientation Analysis Insights"):
                st.markdown(f"""
                ### Orientation Dependence Analysis for {sweep_defect}
                
                **Subplot Breakdown:**
                1. **Top Left**: Von Mises stress variation
                2. **Top Right**: Hydrostatic stress variation
                3. **Bottom Left**: Stress magnitude variation
                4. **Bottom Right**: Combined comparison
                
                **Key Observations:**
                - **Peak orientations**: Where stress is maximum
                - **Minimum orientations**: Where stress is lowest
                - **Symmetry patterns**: Crystal symmetry effects
                - **Component correlation**: How different stresses relate
                
                **Engineering Significance:**
                - **Design guidance**: Orient components to minimize stress
                - **Failure prediction**: Identify critical orientations
                - **Material selection**: Choose materials based on orientation sensitivity
                
                **For {sweep_defect}:**
                - Look for orientations with **simultaneous peaks** in all components
                - Identify **safe zones** where all stresses are low
                - Note any **unusual patterns** that might indicate special behavior
                """)
        
        # Export section
        st.subheader("📤 Export Prediction Visualizations")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Export as HTML (interactive)
            if st.button("💾 Export Interactive Dashboard", use_container_width=True):
                if viz_mode == "Comprehensive Dashboard":
                    # Create HTML file with all visualizations
                    html_content = """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Prediction Visualization Dashboard</title>
                        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 20px; }
                            .chart { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
                            .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
                        </style>
                    </head>
                    <body>
                        <h1>Prediction Visualization Dashboard</h1>
                        <div class="summary">
                            <p><strong>Generated:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                            <p><strong>Defect Types:</strong> """ + ", ".join(defect_types) + """</p>
                            <p><strong>Orientations:</strong> """ + f"{theta_min}° to {theta_max}° in {theta_step}° steps" + """</p>
                        </div>
                    </body>
                    </html>
                    """
                    
                    st.download_button(
                        label="📥 Download HTML",
                        data=html_content,
                        file_name=f"prediction_dashboard_{datetime.now():%Y%m%d_%H%M%S}.html",
                        mime="text/html"
                    )
        
        with col_exp2:
            # Export data as JSON
            if st.button("📊 Export Prediction Data", use_container_width=True):
                # Prepare data for export
                export_data = {
                    'metadata': {
                        'defect_types': defect_types,
                        'theta_range': list(sunburst_data['theta_degrees']),
                        'stress_components': stress_components,
                        'generated_at': datetime.now().isoformat(),
                        'source_count': len(st.session_state.source_simulations)
                    },
                    'statistics': sunburst_data['statistics']
                }
                
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"prediction_data_{datetime.now():%Y%m%d_%H%M%S}.json",
                    mime="application/json"
                )
        
        with col_exp3:
            # Export as PDF report
            if st.button("📄 Generate PDF Report", use_container_width=True):
                st.info("PDF report generation would be implemented here. This would include all visualizations and analysis.")
        
        # Advanced analysis section
        st.subheader("🔬 Advanced Analysis")
        
        with st.expander("📊 Prediction Quality Metrics", expanded=False):
            # Calculate and display quality metrics
            quality_metrics = {}
            
            for defect_type in sunburst_data['defect_types']:
                attention_variance = []
                stress_consistency = []
                
                for theta_key in sunburst_data['predictions'][defect_type]:
                    pred = sunburst_data['predictions'][defect_type][theta_key]
                    
                    # Attention weight quality (lower variance = more confident)
                    if 'attention_weights' in pred:
                        weights = pred['attention_weights']
                        attention_variance.append(np.std(weights))
                    
                    # Stress field consistency
                    if 'von_mises' in pred and 'sigma_hydro' in pred:
                        vm = pred['von_mises']
                        sh = pred['sigma_hydro']
                        # Check if patterns are physically consistent
                        correlation = np.corrcoef(vm.flatten(), sh.flatten())[0, 1]
                        stress_consistency.append(abs(correlation))
                
                if attention_variance:
                    quality_metrics[defect_type] = {
                        'avg_attention_variance': np.mean(attention_variance),
                        'avg_stress_consistency': np.mean(stress_consistency) if stress_consistency else 0,
                        'prediction_count': len(attention_variance)
                    }
            
            # Display metrics
            if quality_metrics:
                df_quality = pd.DataFrame(quality_metrics).T
                st.dataframe(df_quality.style.format({
                    'avg_attention_variance': '{:.4f}',
                    'avg_stress_consistency': '{:.3f}'
                }), use_container_width=True)
                
                st.caption("**Quality Metrics**: Lower attention variance = more confident predictions. Higher stress consistency = more physically plausible results.")
        
        # Conclusion and insights
        st.subheader("💡 Key Insights & Recommendations")
        
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.info("""
            **Critical Findings:**
            
            1. **Worst-case Orientations**: 
               - Identify orientations with maximum stress
               - These are critical for design
               
            2. **Defect Sensitivity**:
               - Which defect types are most sensitive to orientation
               - Relative risk assessment
               
            3. **Safe Operating Zones**:
               - Orientations with consistently low stress
               - Recommended for critical applications
            """)
        
        with col_ins2:
            st.success("""
            **Practical Recommendations:**
            
            1. **Design Guidance**:
               - Orient components to avoid critical angles
               - Consider defect type in material selection
               
            2. **Testing Focus**:
               - Prioritize testing at identified critical orientations
               - Focus on most sensitive defect types
               
            3. **Further Analysis**:
               - Investigate anomalies in prediction patterns
               - Validate with additional simulations
            """)
    
    else:
        # Show instructions if no data generated yet
        st.info("""
        ### Ready to Generate Visualizations?
        
        Click the **"Generate Comprehensive Predictions"** button above to:
        
        1. **Create predictions** across all selected defect types and orientations
        2. **Generate interactive visualizations** showing stress patterns
        3. **Analyze orientation dependence** and defect sensitivity
        4. **Export results** for further analysis
        
        ### What You'll See:
        
        - **🌅 Sunburst Charts**: Hierarchical view of stress predictions
        - **📡 Radar Plots**: Direct comparison across defect types
        - **📈 Orientation Analysis**: Detailed stress vs. angle plots
        - **📊 Statistical Summary**: Numerical analysis of predictions
        - **📉 Distribution Plots**: How stress metrics vary with orientation
        
        ### Customization Options:
        
        Use the sidebar to:
        - Select defect types to analyze
        - Set orientation range and step size
        - Choose visualization modes
        - Adjust material parameters
        - Configure display options
        """)

# =============================================
# UPDATE THE MAIN ATTENTION INTERFACE
# =============================================
def create_attention_interface():
    st.header("🤖 Spatial-Attention Stress Interpolation")
  
    # Initialize managers (existing code remains)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
  
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager()
  
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
  
    if 'prediction_results_manager' not in st.session_state:
        st.session_state.prediction_results_manager = PredictionResultsManager()
  
    if 'visualization_manager' not in st.session_state:
        st.session_state.visualization_manager = VisualizationManager()
  
    if 'time_frame_manager' not in st.session_state:
        st.session_state.time_frame_manager = TimeFrameVisualizationManager(
            st.session_state.visualization_manager
        )
  
    # Initialize sunburst visualizer
    if 'sunburst_visualizer' not in st.session_state:
        st.session_state.sunburst_visualizer = None
  
    # Initialize data storage (existing code remains)
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
  
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
  
    if 'matplotlib_figures' not in st.session_state:
        st.session_state.matplotlib_figures = {}
  
    if 'sunburst_data' not in st.session_state:
        st.session_state.sunburst_data = None
  
    extent = get_grid_extent()
  
    # Sidebar (existing code remains, add sunburst section)
    st.sidebar.header("🔮 Attention Interpolator Settings")
  
    with st.sidebar.expander("⚙️ Model Parameters", expanded=False):
        num_heads = st.slider("Number of Attention Heads", 1, 8, 4, 1)
        sigma_spatial = st.slider("Spatial Sigma (σ_spatial)", 0.05, 1.0, 0.2, 0.05)
        sigma_param = st.slider("Parameter Sigma (σ_param)", 0.05, 1.0, 0.3, 0.05)
        use_gaussian = st.checkbox("Use Gaussian Spatial Regularization", True)
      
        if st.button("🔄 Update Model Parameters"):
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
                num_heads=num_heads,
                sigma_spatial=sigma_spatial,
                sigma_param=sigma_param,
                use_gaussian=use_gaussian
            )
            st.success("Model parameters updated!")
  
    with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
        viz_library = st.radio(
            "Primary Visualization Library",
            ["Plotly (Interactive)", "Matplotlib (Static)"],
            index=0
        )
      
        default_colormap = st.selectbox(
            "Default Colormap",
            ["viridis", "plasma", "coolwarm", "RdBu", "Spectral", "custom_stress"],
            index=5
        )
      
        include_contours = st.checkbox("Include Contour Lines", value=True)
        figure_dpi = st.slider("Figure DPI", 100, 300, 150, 10)
  
    # Update tab structure to include sunburst visualization
    tab1, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict",
        "📊 Results & Visualization",
        "⏱️ Time Frame Analysis",
        "💾 Export Results",
        "🌅 Prediction Sunburst"  # Updated tab name and icon
    ])
  
    # Existing tabs 1-7 (code remains the same)
    # ... (your existing code for tabs 1-7)
  
    # New tab 8: Prediction Sunburst Visualization
    with tab8:
        create_sunburst_prediction_tab()

if __name__ == "__main__":
    create_attention_interface()
st.caption(f"🔬 Attention Interpolation • Unified Download Pattern • {datetime.now().year}")
