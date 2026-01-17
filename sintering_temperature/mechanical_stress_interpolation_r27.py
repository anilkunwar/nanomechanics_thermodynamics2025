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
# ENHANCED DEFECT RADAR CHART VISUALIZER
# =============================================

class DefectRadarVisualizer:
    """Enhanced radar chart visualizer for defect comparison with extensive customization"""
    
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        
        # Comprehensive colormap library (50+ colormaps)
        self.colormaps = {
            # Sequential colormaps
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'magma': 'magma',
            'cividis': 'cividis',
            'turbo': 'turbo',
            'jet': 'jet',
            'rainbow': 'rainbow',
            'hsv': 'hsv',
            'hot': 'hot',
            'cool': 'cool',
            'spring': 'spring',
            'summer': 'summer',
            'autumn': 'autumn',
            'winter': 'winter',
            'bone': 'bone',
            'copper': 'copper',
            'pink': 'pink',
            'gray': 'gray',
            
            # Diverging colormaps
            'RdBu': 'RdBu',
            'RdYlBu': 'RdYlBu',
            'Spectral': 'Spectral',
            'coolwarm': 'coolwarm',
            'bwr': 'bwr',
            'seismic': 'seismic',
            'PiYG': 'PiYG',
            'PRGn': 'PRGn',
            'BrBG': 'BrBG',
            'PuOr': 'PuOr',
            
            # Qualitative colormaps
            'tab10': 'tab10',
            'tab20': 'tab20',
            'Set1': 'Set1',
            'Set2': 'Set2',
            'Set3': 'Set3',
            'Pastel1': 'Pastel1',
            'Pastel2': 'Pastel2',
            'Paired': 'Paired',
            'Accent': 'Accent',
            'Dark2': 'Dark2',
            
            # Custom colormaps
            'deep_thermal': [
                [0, 'rgb(0, 0, 128)'],      # Deep blue
                [0.2, 'rgb(0, 0, 255)'],    # Blue
                [0.4, 'rgb(0, 255, 255)'],  # Cyan
                [0.6, 'rgb(255, 255, 0)'],  # Yellow
                [0.8, 'rgb(255, 128, 0)'],  # Orange
                [1, 'rgb(255, 0, 0)']       # Red
            ],
            'stress_gradient': [
                [0, 'rgb(0, 255, 0)'],      # Green (low stress)
                [0.5, 'rgb(255, 255, 0)'],  # Yellow (medium)
                [1, 'rgb(255, 0, 0)']       # Red (high stress)
            ],
            'defect_specific': [
                [0, 'rgb(152, 223, 138)'],  # Twin - Light green
                [0.33, 'rgb(255, 187, 120)'], # ISF - Light orange
                [0.66, 'rgb(255, 152, 150)'], # ESF - Light red
                [1, 'rgb(174, 199, 232)']   # Perfect - Light blue
            ],
            'crystal_thermal': [
                [0, 'rgb(0, 128, 128)'],    # Teal
                [0.25, 'rgb(0, 191, 255)'], # Deep sky blue
                [0.5, 'rgb(255, 215, 0)'],  # Gold
                [0.75, 'rgb(255, 140, 0)'], # Dark orange
                [1, 'rgb(220, 20, 60)']     # Crimson
            ]
        }
        
        # Default defect colors
        self.defect_colors = {
            'TWIN': 'rgba(152, 223, 138, 0.8)',
            'ESF': 'rgba(255, 152, 150, 0.8)',
            'ISF': 'rgba(255, 187, 120, 0.8)',
            'No Defect': 'rgba(174, 199, 232, 0.8)',
            'Twin': 'rgba(152, 223, 138, 0.8)',  # Alternative spelling
            'Unknown': 'rgba(189, 189, 189, 0.8)'
        }
        
        # Stress component definitions
        self.stress_components = {
            'sigma_hydro': {
                'name': 'Hydrostatic Stress',
                'symbol': 'σ_h',
                'unit': 'GPa',
                'description': 'Average of three principal stresses, critical for sintering'
            },
            'von_mises': {
                'name': 'Von Mises Stress',
                'symbol': 'σ_vm',
                'unit': 'GPa',
                'description': 'Equivalent tensile stress, indicates yield onset'
            },
            'sigma_mag': {
                'name': 'Stress Magnitude',
                'symbol': '|σ|',
                'unit': 'GPa',
                'description': 'Overall stress intensity magnitude'
            }
        }
        
        # Publication quality defaults
        self.publication_styles = {
            'title_font': dict(size=24, family="Arial Black", color='black'),
            'axis_font': dict(size=16, family="Arial", color='black'),
            'legend_font': dict(size=14, family="Arial", color='black'),
            'tick_font': dict(size=12, family="Arial", color='black'),
            'line_width': 3,
            'marker_size': 10,
            'grid_width': 1.5,
            'grid_color': 'rgba(100, 100, 100, 0.3)',
            'bg_color': 'rgba(240, 240, 240, 0.1)'
        }
    
    def create_enhanced_defect_radar(self, defect_data, title="Defect Stress Radar",
                                    vicinity_range=10.0, center_angle=None,
                                    colormap='turbo', normalization='max',
                                    show_components=True, radial_range=None,
                                    font_settings=None, grid_settings=None):
        """
        Create enhanced radar chart for defect comparison
        
        Parameters:
        -----------
        defect_data : dict
            Dictionary with defect types as keys and stress data as values
        vicinity_range : float
            Range around center_angle to display
        center_angle : float or None
            Center angle for radar display (default: habit_angle)
        colormap : str
            Colormap name from self.colormaps
        normalization : str
            'max', 'minmax', 'zscore', or 'none'
        show_components : bool
            Show all stress components or just selected ones
        radial_range : tuple or None
            Manual radial range (min, max)
        font_settings : dict or None
            Custom font settings
        grid_settings : dict or None
            Custom grid settings
        """
        
        if center_angle is None:
            center_angle = self.habit_angle
        
        # Get colormap
        if colormap in self.colormaps:
            if isinstance(self.colormaps[colormap], list):
                colorscale = self.colormaps[colormap]
            else:
                colorscale = self.colormaps[colormap]
        else:
            colorscale = 'turbo'
        
        # Merge font settings with defaults
        if font_settings is None:
            font_settings = self.publication_styles.copy()
        else:
            for key in self.publication_styles:
                if key not in font_settings:
                    font_settings[key] = self.publication_styles[key]
        
        # Prepare data
        processed_data = {}
        max_values = {}
        min_values = {}
        
        # First pass: collect statistics
        for defect_name, data in defect_data.items():
            if isinstance(data, dict) and 'stresses' in data:
                for comp, values in data['stresses'].items():
                    if comp not in max_values:
                        max_values[comp] = []
                        min_values[comp] = []
                    if values:
                        max_values[comp].append(max(values))
                        min_values[comp].append(min(values))
        
        # Determine radial ranges
        if radial_range is None:
            radial_max = 0
            radial_min = 0
            for comp in max_values:
                if max_values[comp]:
                    radial_max = max(radial_max, max(max_values[comp]))
                if min_values[comp]:
                    radial_min = min(radial_min, min(min_values[comp]))
            radial_range = [radial_min * 0.9, radial_max * 1.1]
        
        # Create figure
        fig = go.Figure()
        
        # Process each defect
        for defect_name, data in defect_data.items():
            if not isinstance(data, dict) or 'angles' not in data or 'stresses' not in data:
                continue
                
            angles = data['angles']
            stresses = data['stresses']
            
            # Filter to vicinity
            mask = (np.array(angles) >= center_angle - vicinity_range) & \
                   (np.array(angles) <= center_angle + vicinity_range)
            
            if not np.any(mask):
                continue
                
            vic_angles = np.array(angles)[mask]
            
            # Get defect color
            defect_color = self.defect_colors.get(defect_name, 
                                                f'rgba({np.random.randint(50,200)}, '
                                                f'{np.random.randint(50,200)}, '
                                                f'{np.random.randint(50,200)}, 0.8)')
            
            # Add each stress component
            for comp_idx, (comp_name, comp_values) in enumerate(stresses.items()):
                if not show_components and comp_idx > 0:
                    continue
                    
                if comp_values is None or len(comp_values) == 0:
                    continue
                    
                vic_stresses = np.array(comp_values)[mask]
                
                if len(vic_stresses) == 0:
                    continue
                
                # Normalize if requested
                if normalization == 'max' and max(vic_stresses) > 0:
                    display_values = vic_stresses / max(vic_stresses)
                elif normalization == 'minmax' and (max(vic_stresses) - min(vic_stresses)) > 0:
                    display_values = (vic_stresses - min(vic_stresses)) / \
                                   (max(vic_stresses) - min(vic_stresses))
                elif normalization == 'zscore' and np.std(vic_stresses) > 0:
                    display_values = (vic_stresses - np.mean(vic_stresses)) / np.std(vic_stresses)
                else:
                    display_values = vic_stresses
                
                # Close the loop for radar
                display_closed = np.append(display_values, display_values[0])
                angles_closed = np.append(vic_angles, vic_angles[0])
                
                # Add trace
                trace_name = f"{defect_name} - {self.stress_components.get(comp_name, {}).get('name', comp_name)}"
                
                fig.add_trace(go.Scatterpolar(
                    r=display_closed,
                    theta=angles_closed,
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(
                        color=defect_color,
                        width=font_settings['line_width'],
                        shape='spline'
                    ),
                    marker=dict(
                        size=font_settings['marker_size'],
                        color=vic_stresses,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(
                            title=f"{comp_name} (GPa)",
                            thickness=20,
                            len=0.5,
                            x=1.15
                        ) if comp_idx == 0 else None
                    ),
                    fill='toself' if show_components else 'none',
                    fillcolor=defect_color.replace('0.8', '0.2') if show_components else None,
                    hovertemplate=f"""
                    <b>{defect_name}</b><br>
                    Orientation: %{{theta:.1f}}°<br>
                    {comp_name}: %{{r:.3f}} GPa<br>
                    <extra></extra>
                    """,
                    showlegend=True
                ))
        
        # Add habit plane reference line
        fig.add_trace(go.Scatterpolar(
            r=[0, radial_range[1]],
            theta=[center_angle, center_angle],
            mode='lines',
            line=dict(
                color='rgb(46, 204, 113)',
                width=4,
                dash='dashdot'
            ),
            name=f'Habit Plane ({center_angle}°)',
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Update layout with enhanced styling
        fig.update_layout(
            title=dict(
                text=title,
                font=font_settings['title_font'],
                x=0.5,
                y=0.95
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor=grid_settings['grid_color'] if grid_settings else font_settings['grid_color'],
                    gridwidth=font_settings['grid_width'],
                    linecolor='black',
                    linewidth=2,
                    tickfont=font_settings['tick_font'],
                    title=dict(
                        text='Stress (GPa)' if normalization == 'none' else 'Normalized Stress',
                        font=font_settings['axis_font']
                    ),
                    range=radial_range,
                    angle=90,
                    tickangle=0
                ),
                angularaxis=dict(
                    gridcolor=grid_settings['grid_color'] if grid_settings else font_settings['grid_color'],
                    gridwidth=font_settings['grid_width'],
                    linecolor='black',
                    linewidth=2,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(int(center_angle - vicinity_range), 
                                      int(center_angle + vicinity_range) + 1, 
                                      max(1, int(vicinity_range/5)))),
                    ticktext=[f'{i}°' for i in range(int(center_angle - vicinity_range), 
                                                   int(center_angle + vicinity_range) + 1, 
                                                   max(1, int(vicinity_range/5)))],
                    tickfont=font_settings['tick_font'],
                    period=360
                ),
                bgcolor=font_settings['bg_color'],
                sector=[center_angle - vicinity_range, center_angle + vicinity_range]
            ),
            showlegend=True,
            legend=dict(
                x=1.15,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=1,
                font=font_settings['legend_font'],
                itemsizing='constant'
            ),
            width=1200,
            height=800,
            margin=dict(l=150, r=250, t=100, b=100),
            hovermode='closest',
            hoverlabel=dict(
                font_size=14,
                font_family="Arial"
            )
        )
        
        return fig
    
    def create_comparison_radar_matrix(self, defect_data, vicinity_range=10.0,
                                      colormap='defect_specific', title="Defect Radar Matrix"):
        """
        Create matrix of radar charts for comprehensive defect comparison
        """
        
        stress_components = list(self.stress_components.keys())
        defect_types = list(defect_data.keys())
        
        n_rows = len(stress_components)
        n_cols = len(defect_types)
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f"{defect}<br>{self.stress_components[comp]['name']}" 
                          for comp in stress_components for defect in defect_types],
            specs=[[{'type': 'polar'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        for i, comp in enumerate(stress_components):
            for j, defect in enumerate(defect_types):
                row = i + 1
                col = j + 1
                
                if defect in defect_data and comp in defect_data[defect]['stresses']:
                    data = defect_data[defect]
                    angles = np.array(data['angles'])
                    stresses = np.array(data['stresses'][comp])
                    
                    # Filter to vicinity
                    mask = (angles >= self.habit_angle - vicinity_range) & \
                           (angles <= self.habit_angle + vicinity_range)
                    
                    if np.any(mask):
                        vic_angles = angles[mask]
                        vic_stresses = stresses[mask]
                        
                        # Close loop
                        angles_closed = np.append(vic_angles, vic_angles[0])
                        stresses_closed = np.append(vic_stresses, vic_stresses[0])
                        
                        fig.add_trace(
                            go.Scatterpolar(
                                r=stresses_closed,
                                theta=angles_closed,
                                mode='lines+markers',
                                line=dict(
                                    color=self.defect_colors.get(defect, 'blue'),
                                    width=2
                                ),
                                marker=dict(
                                    size=4,
                                    color=vic_stresses,
                                    colorscale=self.colormaps[colormap],
                                    showscale=False
                                ),
                                fill='toself',
                                fillcolor=self.defect_colors.get(defect, 'blue').replace('0.8', '0.2'),
                                showlegend=False,
                                hovertemplate=f"Angle: %{{theta:.1f}}°<br>Stress: %{{r:.3f}} GPa<extra></extra>"
                            ),
                            row=row, col=col
                        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=self.publication_styles['title_font'],
                x=0.5,
                y=0.98
            ),
            height=400 * n_rows,
            width=400 * n_cols,
            showlegend=False
        )
        
        # Update polar subplots
        for i in range(1, n_rows * n_cols + 1):
            fig.update_polars(
                radialaxis_showticklabels=True,
                radialaxis_tickfont_size=10,
                angularaxis_tickfont_size=10,
                row=(i-1)//n_cols + 1,
                col=(i-1)%n_cols + 1
            )
        
        return fig
    
    def create_sunburst_defect_chart(self, defect_data, max_layers=3,
                                    title="Spatial Hydrostatic Stress Distribution"):
        """
        Create sunburst chart for spatial defect analysis
        """
        
        # Prepare hierarchical data
        labels = []
        parents = []
        values = []
        colors = []
        
        # Root
        labels.append("Defects")
        parents.append("")
        values.append(0)  # Will be sum of children
        colors.append('rgb(200, 200, 200)')
        
        # Add defect types
        for defect_idx, (defect_name, data) in enumerate(defect_data.items()):
            labels.append(defect_name)
            parents.append("Defects")
            
            # Calculate average hydrostatic stress
            if 'stresses' in data and 'sigma_hydro' in data['stresses']:
                avg_stress = np.mean(data['stresses']['sigma_hydro']) if data['stresses']['sigma_hydro'] else 0
            else:
                avg_stress = 0
                
            values.append(abs(avg_stress))
            colors.append(self.defect_colors.get(defect_name, f'rgb({100 + defect_idx*50}, {100}, {100})'))
            
            # Add stress components
            if max_layers > 1:
                for comp_idx, (comp_name, comp_data) in enumerate(data.get('stresses', {}).items()):
                    comp_label = f"{defect_name}_{comp_name}"
                    labels.append(self.stress_components.get(comp_name, {}).get('name', comp_name))
                    parents.append(defect_name)
                    
                    if comp_data:
                        comp_value = np.mean(comp_data)
                    else:
                        comp_value = 0
                        
                    values.append(abs(comp_value))
                    
                    # Color based on stress magnitude
                    norm_value = (comp_value - min(values)) / (max(values) - min(values)) if max(values) > min(values) else 0.5
                    colors.append(f'rgba({int(255*norm_value)}, {int(255*(1-norm_value))}, 100, 0.8)')
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                line=dict(width=2, color='white')
            ),
            textinfo="label+percent entry",
            hoverinfo="label+value+percent parent",
            maxdepth=max_layers
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=self.publication_styles['title_font'],
                x=0.5
            ),
            width=900,
            height=700,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    def create_interactive_radar_dashboard(self, defect_data, vicinity_range=20.0):
        """
        Create interactive dashboard with multiple radar visualizations
        """
        
        tabs = []
        
        # Tab 1: Enhanced Radar Chart
        fig_radar = self.create_enhanced_defect_radar(
            defect_data,
            title="Enhanced Defect Radar - Habit Plane Vicinity",
            vicinity_range=vicinity_range,
            colormap='turbo',
            show_components=True
        )
        tabs.append(("Enhanced Radar", fig_radar))
        
        # Tab 2: Component-wise Radar
        fig_component = self.create_comparison_radar_matrix(
            defect_data,
            vicinity_range=vicinity_range,
            colormap='stress_gradient',
            title="Component-wise Defect Comparison"
        )
        tabs.append(("Component Matrix", fig_component))
        
        # Tab 3: Sunburst Chart
        fig_sunburst = self.create_sunburst_defect_chart(
            defect_data,
            max_layers=3,
            title="Spatial Stress Distribution - Sunburst View"
        )
        tabs.append(("Sunburst View", fig_sunburst))
        
        # Tab 4: Normalized Comparison
        fig_normalized = self.create_enhanced_defect_radar(
            defect_data,
            title="Normalized Stress Comparison",
            vicinity_range=vicinity_range,
            colormap='RdBu',
            normalization='minmax',
            show_components=False
        )
        tabs.append(("Normalized View", fig_normalized))
        
        return tabs
    
    def get_colormap_preview(self, selected_colormaps=None):
        """
        Generate colormap preview visualization
        """
        if selected_colormaps is None:
            selected_colormaps = list(self.colormaps.keys())[:20]  # Preview first 20
        
        n_colors = 256
        gradient = np.linspace(0, 1, n_colors)
        
        fig, axes = plt.subplots(len(selected_colormaps), 1, 
                                figsize=(10, len(selected_colormaps) * 0.5),
                                constrained_layout=True)
        
        if len(selected_colormaps) == 1:
            axes = [axes]
        
        for ax, cmap_name in zip(axes, selected_colormaps):
            if cmap_name in self.colormaps:
                if isinstance(self.colormaps[cmap_name], list):
                    # Custom colormap
                    cmap = LinearSegmentedColormap.from_list(cmap_name, 
                                                           [(p, c) for p, c in self.colormaps[cmap_name]])
                else:
                    # Matplotlib colormap
                    cmap = plt.get_cmap(self.colormaps[cmap_name])
                
                gradient_img = np.vstack([gradient])
                ax.imshow(gradient_img, aspect='auto', cmap=cmap)
                ax.text(-0.01, 0.5, cmap_name, va='center', ha='right', fontsize=10,
                       transform=ax.transAxes, fontweight='bold')
                ax.set_axis_off()
        
        fig.suptitle('Available Colormaps', fontsize=14, fontweight='bold', y=1.02)
        return fig

# =============================================
# ENHANCED STREAMLIT INTERFACE FOR RADAR CHARTS
# =============================================

def create_defect_radar_interface():
    """
    Create Streamlit interface for enhanced defect radar charts
    """
    
    st.set_page_config(
        page_title="Enhanced Defect Radar Visualization",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .radar-header {
        font-size: 2.8rem !important;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900 !important;
        margin-bottom: 1.5rem;
        padding: 1rem;
    }
    .habit-plane-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        border: 2px solid #047857;
    }
    .defect-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .twin-tag { background-color: rgba(152, 223, 138, 0.3); color: #2e7d32; }
    .esf-tag { background-color: rgba(255, 152, 150, 0.3); color: #c62828; }
    .isf-tag { background-color: rgba(255, 187, 120, 0.3); color: #ef6c00; }
    .perfect-tag { background-color: rgba(174, 199, 232, 0.3); color: #1565c0; }
    .control-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="radar-header">📊 Enhanced Defect Radar Visualization</h1>', unsafe_allow_html=True)
    
    # Habit plane badge
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
    <span class="habit-plane-badge">
    🎯 AG FCC Twin Habit Plane: 54.7° | {111} Crystal Planes | Maximum Stress Concentration
    </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualizer
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = DefectRadarVisualizer()
    
    # Sample defect data (in real app, this would come from your interpolator)
    @st.cache_data
    def generate_sample_defect_data():
        """Generate sample defect data for demonstration"""
        angles = np.linspace(0, 360, 360)
        
        defect_data = {
            'TWIN': {
                'angles': angles.tolist(),
                'stresses': {
                    'sigma_hydro': (20 * np.exp(-(angles - 54.7)**2 / (2*30**2)) + 
                                   10 * np.sin(np.radians(angles)) + 15).tolist(),
                    'von_mises': (25 * np.exp(-(angles - 54.7)**2 / (2*25**2)) + 
                                 5 * np.sin(np.radians(2*angles)) + 20).tolist(),
                    'sigma_mag': (30 * np.exp(-(angles - 54.7)**2 / (2*20**2)) + 
                                 8 * np.cos(np.radians(angles)) + 25).tolist()
                }
            },
            'ESF': {
                'angles': angles.tolist(),
                'stresses': {
                    'sigma_hydro': (15 * np.exp(-(angles - 54.7)**2 / (2*40**2)) + 
                                   8 * np.sin(np.radians(angles)) + 10).tolist(),
                    'von_mises': (20 * np.exp(-(angles - 54.7)**2 / (2*35**2)) + 
                                 4 * np.sin(np.radians(2*angles)) + 15).tolist(),
                    'sigma_mag': (25 * np.exp(-(angles - 54.7)**2 / (2*30**2)) + 
                                 6 * np.cos(np.radians(angles)) + 20).tolist()
                }
            },
            'ISF': {
                'angles': angles.tolist(),
                'stresses': {
                    'sigma_hydro': (12 * np.exp(-(angles - 54.7)**2 / (2*50**2)) + 
                                   6 * np.sin(np.radians(angles)) + 8).tolist(),
                    'von_mises': (18 * np.exp(-(angles - 54.7)**2 / (2*45**2)) + 
                                 3 * np.sin(np.radians(2*angles)) + 12).tolist(),
                    'sigma_mag': (22 * np.exp(-(angles - 54.7)**2 / (2*40**2)) + 
                                 5 * np.cos(np.radians(angles)) + 18).tolist()
                }
            },
            'No Defect': {
                'angles': angles.tolist(),
                'stresses': {
                    'sigma_hydro': (5 * np.exp(-(angles - 54.7)**2 / (2*100**2)) + 
                                   2 * np.sin(np.radians(angles)) + 3).tolist(),
                    'von_mises': (8 * np.exp(-(angles - 54.7)**2 / (2*80**2)) + 
                                 1 * np.sin(np.radians(2*angles)) + 5).tolist(),
                    'sigma_mag': (10 * np.exp(-(angles - 54.7)**2 / (2*60**2)) + 
                                 2 * np.cos(np.radians(angles)) + 8).tolist()
                }
            }
        }
        
        return defect_data
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<div class="control-panel">⚙️ Radar Control Panel</div>', unsafe_allow_html=True)
        
        # Data selection
        st.markdown("#### 📁 Data Configuration")
        use_sample_data = st.checkbox("Use Sample Data", value=True, 
                                     help="Use pre-generated sample defect data")
        
        # Defect selection
        st.markdown("#### 🔬 Defect Selection")
        selected_defects = st.multiselect(
            "Select Defects to Display",
            options=['TWIN', 'ESF', 'ISF', 'No Defect'],
            default=['TWIN', 'ESF', 'ISF', 'No Defect'],
            help="Choose which defect types to include in the radar chart"
        )
        
        # Radar configuration
        st.markdown("#### 🎯 Radar Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            vicinity_range = st.slider(
                "Vicinity Range (°)",
                min_value=5.0,
                max_value=45.0,
                value=20.0,
                step=1.0,
                help="Range around habit plane to display"
            )
        
        with col2:
            center_angle = st.number_input(
                "Center Angle (°)",
                min_value=0.0,
                max_value=360.0,
                value=54.7,
                step=0.1,
                help="Center angle for radar display"
            )
        
        # Colormap selection
        st.markdown("#### 🎨 Colormap Selection")
        colormap_options = list(st.session_state.visualizer.colormaps.keys())
        selected_colormap = st.selectbox(
            "Choose Colormap",
            options=colormap_options,
            index=colormap_options.index('turbo'),
            help="Select colormap for stress visualization"
        )
        
        # Normalization
        st.markdown("#### 📊 Normalization")
        normalization = st.radio(
            "Normalization Method",
            options=['none', 'max', 'minmax', 'zscore'],
            index=1,
            help="How to normalize stress values"
        )
        
        # Display options
        st.markdown("#### 👁️ Display Options")
        show_components = st.checkbox("Show All Components", value=True,
                                     help="Display all stress components or just selected ones")
        show_legend = st.checkbox("Show Legend", value=True)
        show_grid = st.checkbox("Show Grid", value=True)
        
        # Font and styling controls
        st.markdown("#### ✏️ Styling Controls")
        
        with st.expander("Font Settings", expanded=False):
            title_size = st.slider("Title Font Size", 16, 36, 24)
            axis_size = st.slider("Axis Font Size", 10, 24, 16)
            legend_size = st.slider("Legend Font Size", 10, 20, 14)
            tick_size = st.slider("Tick Font Size", 8, 18, 12)
            
            font_family = st.selectbox(
                "Font Family",
                options=['Arial', 'Arial Black', 'Times New Roman', 'Courier New', 'Verdana'],
                index=0
            )
            
        with st.expander("Line and Marker Settings", expanded=False):
            line_width = st.slider("Line Width", 1, 8, 3)
            marker_size = st.slider("Marker Size", 3, 20, 8)
            grid_width = st.slider("Grid Width", 0.5, 3.0, 1.5)
            
        with st.expander("Color Adjustments", expanded=False):
            bg_color = st.color_picker("Background Color", "#F0F0F0")
            grid_color = st.color_picker("Grid Color", "#646464")
            
            # Custom defect colors
            st.write("Defect Colors:")
            twin_color = st.color_picker("TWIN Color", "#98DF8A")
            esf_color = st.color_picker("ESF Color", "#FF9896")
            isf_color = st.color_picker("ISF Color", "#FFBB78")
            perfect_color = st.color_picker("No Defect Color", "#AEC7E8")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Enhanced Radar",
        "🧩 Component Matrix",
        "☀️ Sunburst View",
        "🎨 Customization",
        "📋 Export & Save"
    ])
    
    # Get data
    if use_sample_data:
        defect_data = generate_sample_defect_data()
        # Filter selected defects
        defect_data = {k: v for k, v in defect_data.items() if k in selected_defects}
    else:
        # In real implementation, load from session state or files
        defect_data = {}
        st.info("Please load defect data from your analysis or use sample data.")
    
    with tab1:
        # Enhanced Radar Chart
        st.markdown("### 📊 Enhanced Defect Radar Chart")
        st.markdown("""
        This radar chart visualizes stress components for different defect types 
        in the vicinity of the habit plane. Use the controls in the sidebar to 
        customize the visualization.
        """)
        
        if defect_data:
            # Prepare font settings
            font_settings = {
                'title_font': dict(size=title_size, family=font_family, color='black'),
                'axis_font': dict(size=axis_size, family=font_family, color='black'),
                'legend_font': dict(size=legend_size, family=font_family, color='black'),
                'tick_font': dict(size=tick_size, family=font_family, color='black'),
                'line_width': line_width,
                'marker_size': marker_size,
                'grid_width': grid_width,
                'grid_color': grid_color,
                'bg_color': bg_color
            }
            
            # Update defect colors
            st.session_state.visualizer.defect_colors.update({
                'TWIN': f'rgba({int(twin_color[1:3], 16)}, {int(twin_color[3:5], 16)}, {int(twin_color[5:7], 16)}, 0.8)',
                'ESF': f'rgba({int(esf_color[1:3], 16)}, {int(esf_color[3:5], 16)}, {int(esf_color[5:7], 16)}, 0.8)',
                'ISF': f'rgba({int(isf_color[1:3], 16)}, {int(isf_color[3:5], 16)}, {int(isf_color[5:7], 16)}, 0.8)',
                'No Defect': f'rgba({int(perfect_color[1:3], 16)}, {int(perfect_color[3:5], 16)}, {int(perfect_color[5:7], 16)}, 0.8)'
            })
            
            # Create enhanced radar
            fig = st.session_state.visualizer.create_enhanced_defect_radar(
                defect_data,
                title=f"Defect Stress Radar - {vicinity_range}° Vicinity",
                vicinity_range=vicinity_range,
                center_angle=center_angle,
                colormap=selected_colormap,
                normalization=normalization,
                show_components=show_components,
                font_settings=font_settings
            )
            
            # Update legend visibility
            fig.update_layout(showlegend=show_legend)
            
            # Update grid visibility
            if not show_grid:
                fig.update_polars(
                    radialaxis_gridcolor='rgba(0,0,0,0)',
                    angularaxis_gridcolor='rgba(0,0,0,0)'
                )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            
            # Statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            for idx, (defect_name, data) in enumerate(defect_data.items()):
                with [col_stat1, col_stat2, col_stat3, col_stat4][idx % 4]:
                    if 'stresses' in data and 'sigma_hydro' in data['stresses']:
                        stresses = data['stresses']['sigma_hydro']
                        if stresses:
                            avg_stress = np.mean(stresses)
                            max_stress = max(stresses)
                            st.metric(
                                f"{defect_name} σ_h",
                                f"{avg_stress:.2f} GPa",
                                f"Max: {max_stress:.2f} GPa"
                            )
    
    with tab2:
        # Component Matrix
        st.markdown("### 🧩 Component-wise Radar Matrix")
        st.markdown("""
        Matrix view showing each stress component for every defect type. 
        This allows for direct comparison across components and defects.
        """)
        
        if defect_data:
            fig_matrix = st.session_state.visualizer.create_comparison_radar_matrix(
                defect_data,
                vicinity_range=vicinity_range,
                colormap=selected_colormap,
                title="Defect Stress Component Matrix"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
    
    with tab3:
        # Sunburst View
        st.markdown("### ☀️ Spatial Stress Distribution - Sunburst Chart")
        st.markdown("""
        Hierarchical sunburst chart showing spatial distribution of hydrostatic stress
        across defect types and stress components.
        """)
        
        if defect_data:
            fig_sunburst = st.session_state.visualizer.create_sunburst_defect_chart(
                defect_data,
                max_layers=3,
                title="Spatial Hydrostatic Stress Distribution"
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            # Explanation
            with st.expander("ℹ️ Sunburst Chart Interpretation", expanded=False):
                st.markdown("""
                **How to read this sunburst chart:**
                1. **Outer ring**: Shows defect types (TWIN, ESF, ISF, No Defect)
                2. **Inner rings**: Show stress components within each defect
                3. **Segment size**: Represents relative stress magnitude
                4. **Color intensity**: Indicates stress level (darker = higher stress)
                
                **Insights:**
                - Larger segments indicate higher stress contributions
                - Color gradients show stress distribution patterns
                - Hierarchy reveals component contributions to total stress
                """)
    
    with tab4:
        # Customization
        st.markdown("### 🎨 Advanced Customization")
        
        col_cust1, col_cust2 = st.columns(2)
        
        with col_cust1:
            st.markdown("#### 🎯 Radar Customization")
            
            # Radar type selection
            radar_type = st.radio(
                "Radar Chart Type",
                options=["Standard", "Stacked", "Nested", "Parallel"],
                index=0,
                help="Different radar chart configurations"
            )
            
            # Component selection
            selected_components = st.multiselect(
                "Select Stress Components",
                options=list(st.session_state.visualizer.stress_components.keys()),
                default=list(st.session_state.visualizer.stress_components.keys()),
                help="Choose which stress components to display"
            )
            
            # Radial scale
            radial_scale = st.selectbox(
                "Radial Scale",
                options=["Linear", "Logarithmic", "Square Root"],
                index=0
            )
            
        with col_cust2:
            st.markdown("#### 🎨 Colormap Preview")
            
            # Show colormap preview
            preview_colormaps = st.multiselect(
                "Preview Colormaps",
                options=list(st.session_state.visualizer.colormaps.keys()),
                default=[selected_colormap, 'rainbow', 'jet', 'RdBu', 'deep_thermal'],
                help="Select colormaps to preview"
            )
            
            if preview_colormaps:
                fig_preview = st.session_state.visualizer.get_colormap_preview(preview_colormaps)
                st.pyplot(fig_preview)
        
        # Create custom visualization based on selections
        if defect_data and st.button("🔄 Apply Customizations", use_container_width=True):
            st.success("Customizations applied! Switch to Enhanced Radar tab to see changes.")
    
    with tab5:
        # Export and Save
        st.markdown("### 📋 Export & Save Visualizations")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.markdown("#### 💾 Save Configuration")
            
            config_name = st.text_input("Configuration Name", "my_radar_config")
            
            if st.button("💾 Save Current Settings", use_container_width=True):
                config = {
                    'vicinity_range': vicinity_range,
                    'center_angle': center_angle,
                    'colormap': selected_colormap,
                    'normalization': normalization,
                    'show_components': show_components,
                    'font_settings': {
                        'title_size': title_size,
                        'axis_size': axis_size,
                        'legend_size': legend_size,
                        'tick_size': tick_size
                    },
                    'defect_colors': {
                        'TWIN': twin_color,
                        'ESF': esf_color,
                        'ISF': isf_color,
                        'No Defect': perfect_color
                    }
                }
                
                # Save to JSON
                json_str = json.dumps(config, indent=2)
                st.download_button(
                    label="📥 Download Configuration",
                    data=json_str,
                    file_name=f"{config_name}_radar_config.json",
                    mime="application/json"
                )
        
        with col_exp2:
            st.markdown("#### 📤 Export Visualization")
            
            export_format = st.radio(
                "Export Format",
                options=["PNG", "SVG", "PDF", "HTML"],
                index=0
            )
            
            if defect_data:
                # Create figure for export
                fig_export = st.session_state.visualizer.create_enhanced_defect_radar(
                    defect_data,
                    title=f"Defect Radar - {vicinity_range}° Vicinity",
                    vicinity_range=vicinity_range,
                    colormap=selected_colormap
                )
                
                # Export buttons
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    if st.button("🖼️ Export Image", use_container_width=True):
                        # In production, use: fig_export.write_image(...)
                        st.info(f"Image export ({export_format}) would be implemented here")
                
                with col_btn2:
                    if st.button("📊 Export Data", use_container_width=True):
                        # Export underlying data
                        export_data = []
                        for defect_name, data in defect_data.items():
                            export_data.append({
                                'defect': defect_name,
                                'angles': data['angles'],
                                'stresses': data['stresses']
                            })
                        
                        df = pd.DataFrame(export_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"defect_radar_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        # Quick export presets
        st.markdown("#### ⚡ Quick Export Presets")
        
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        
        with col_preset1:
            if st.button("🎯 Habit Plane Focus", use_container_width=True):
                st.session_state.vicinity_range = 10.0
                st.session_state.center_angle = 54.7
                st.session_state.selected_colormap = 'deep_thermal'
                st.rerun()
        
        with col_preset2:
            if st.button("📈 Publication Quality", use_container_width=True):
                st.session_state.title_size = 24
                st.session_state.axis_size = 18
                st.session_state.line_width = 4
                st.session_state.colormap = 'viridis'
                st.rerun()
        
        with col_preset3:
            if st.button("🔬 Defect Comparison", use_container_width=True):
                st.session_state.show_components = False
                st.session_state.normalization = 'minmax'
                st.session_state.vicinity_range = 30.0
                st.rerun()

# =============================================
# INTEGRATION WITH EXISTING SYSTEM
# =============================================

def integrate_radar_visualization(main_app_function):
    """
    Integrate radar visualization into existing app
    """
    
    # Add radar visualization tab to existing app
    def enhanced_main():
        # Create tabs including radar visualization
        tab_names = ["Main Analysis", "Radar Visualization", "Export"]
        
        # Run radar interface
        create_defect_radar_interface()
    
    return enhanced_main

# =============================================
# EXAMPLE USAGE
# =============================================

if __name__ == "__main__":
    # Option 1: Run standalone radar visualization
    create_defect_radar_interface()
    
    # Option 2: Integrate with existing app
    # main_app = integrate_radar_visualization(main)
    # main_app()
