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
import plotly.express as px
import cmasher as cmr  # For additional colormaps

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# =============================================
# ENHANCED VISUALIZATION SETTINGS
# =============================================

class PublicationVisualizationSettings:
    """Settings for publication-quality visualizations"""
    
    def __init__(self):
        # Font settings for publication quality
        self.font_settings = {
            'family': 'Arial',
            'size': {
                'title': 22,
                'axis_label': 18,
                'axis_tick': 14,
                'legend': 16,
                'annotation': 14,
                'radar_label': 16,
                'radar_tick': 14
            },
            'weight': {
                'title': 'bold',
                'axis_label': 'bold',
                'legend': 'normal'
            }
        }
        
        # Color settings
        self.color_settings = {
            'defect_colors': {
                'ISF': '#FF6B6B',  # Red
                'ESF': '#4ECDC4',   # Teal
                'Twin': '#45B7D1',  # Blue
                'No Defect': '#96CEB4'  # Green
            },
            'stress_colors': {
                'sigma_hydro': '#1F77B4',
                'von_mises': '#FF7F0E',
                'sigma_mag': '#2CA02C'
            },
            'system_colors': {
                'System 1': '#10B981',
                'System 2': '#F59E0B',
                'System 3': '#EF4444'
            }
        }
        
        # Figure dimensions for publication
        self.figure_sizes = {
            'single_column': (8, 6),      # inches
            'double_column': (12, 8),
            'wide': (16, 10),
            'square': (8, 8)
        }
        
        # DPI settings
        self.dpi = 300
        
    def get_font_dict(self, element_type='axis_label'):
        """Get font dictionary for specific element"""
        return {
            'family': self.font_settings['family'],
            'size': self.font_settings['size'].get(element_type, 12),
            'weight': self.font_settings['weight'].get(element_type, 'normal')
        }
    
    def get_colormap_list(self):
        """Get comprehensive list of colormaps"""
        colormaps = {
            'Perceptually Uniform Sequential': [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'rocket', 'mako', 'flare', 'crest', 'icefire',
                'vlag', 'turbo'
            ],
            'Sequential': [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
            ],
            'Sequential (2)': [
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                'pink', 'spring', 'summer', 'autumn', 'winter',
                'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
            ],
            'Diverging': [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
            ],
            'Cyclic': [
                'twilight', 'twilight_shifted', 'hsv'
            ],
            'Qualitative': [
                'tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3',
                'pastel1', 'pastel2', 'Paired', 'Dark2', 'Accent'
            ],
            'Miscellaneous': [
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                'nipy_spectral', 'gist_ncar'
            ]
        }
        return colormaps
    
    def get_plotly_colorscale(self, colormap_name='viridis'):
        """Get plotly colorscale from matplotlib colormap"""
        import matplotlib.cm as cm
        try:
            cmap = cm.get_cmap(colormap_name)
            colorscale = []
            for i in range(256):
                r, g, b, a = cmap(i)
                colorscale.append([i/255, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
            return colorscale
        except:
            # Default to viridis
            return px.colors.sequential.Viridis

# =============================================
# ENHANCED RADAR VISUALIZER
# =============================================

class EnhancedRadarVisualizer:
    """Enhanced radar visualizer with extensive customization options"""
    
    def __init__(self):
        self.pub_settings = PublicationVisualizationSettings()
        self.habit_angle = 54.7
        
    def create_customizable_radar(self, data_dict, title="Defect Comparison Radar",
                                 colormap='viridis', font_size=14, show_grid=True,
                                 grid_color='rgba(150, 150, 150, 0.3)',
                                 bg_color='rgba(240, 240, 240, 0.1)',
                                 line_width=3, marker_size=8,
                                 fill_opacity=0.2, show_legend=True,
                                 legend_position='right', radial_range=None,
                                 angular_range=None, annotations=None,
                                 custom_labels=None, show_habit_plane=True,
                                 habit_plane_color='green',
                                 habit_plane_width=3,
                                 radial_tick_count=5):
        """Create highly customizable radar chart"""
        
        fig = go.Figure()
        
        # Get colors from colormap
        colorscale = self.pub_settings.get_plotly_colorscale(colormap)
        num_traces = len(data_dict)
        
        # Add each trace
        for idx, (trace_name, trace_data) in enumerate(data_dict.items()):
            # Get color from colormap
            if isinstance(colorscale, list) and len(colorscale) > 0:
                color_idx = int(idx * 255 / max(1, num_traces - 1))
                color = colorscale[color_idx][1]
            else:
                color = self.pub_settings.color_settings['defect_colors'].get(
                    trace_name.split('_')[0], '#000000'
                )
            
            # Prepare data
            if 'angles' in trace_data and 'stresses' in trace_data:
                angles = np.array(trace_data['angles'])
                stresses = np.array(trace_data['stresses'])
                
                # Close the loop for radar chart
                if len(angles) > 0 and len(stresses) > 0:
                    angles_closed = np.append(angles, angles[0])
                    stresses_closed = np.append(stresses, stresses[0])
                    
                    # Add trace
                    fig.add_trace(go.Scatterpolar(
                        r=stresses_closed,
                        theta=angles_closed,
                        fill='toself' if fill_opacity > 0 else 'none',
                        fillcolor=color.replace('rgb', 'rgba').replace(')', f', {fill_opacity})'),
                        line=dict(color=color, width=line_width),
                        marker=dict(size=marker_size, color=color),
                        name=custom_labels.get(trace_name, trace_name) if custom_labels else trace_name,
                        hovertemplate=(
                            f"<b>{trace_name}</b><br>" +
                            "Angle: %{theta:.1f}°<br>" +
                            "Value: %{r:.4f}<br>" +
                            "<extra></extra>"
                        ),
                        showlegend=show_legend
                    ))
        
        # Set radial range if provided
        if radial_range:
            radial_range = radial_range
        else:
            # Auto calculate range
            all_stresses = []
            for trace_data in data_dict.values():
                if 'stresses' in trace_data:
                    all_stresses.extend(trace_data['stresses'])
            if all_stresses:
                max_stress = max(all_stresses)
                radial_range = [0, max_stress * 1.2]
            else:
                radial_range = [0, 1]
        
        # Set angular range if provided
        if angular_range:
            angular_range = angular_range
        else:
            angular_range = [0, 360]
        
        # Configure polar layout
        polar_layout = dict(
            radialaxis=dict(
                visible=True,
                gridcolor=grid_color,
                gridwidth=1 if show_grid else 0,
                linecolor='black',
                linewidth=2,
                tickfont=dict(
                    size=font_size,
                    family=self.pub_settings.font_settings['family'],
                    color='black'
                ),
                title=dict(
                    text='Stress (GPa)',
                    font=dict(
                        size=font_size + 2,
                        family=self.pub_settings.font_settings['family'],
                        weight='bold',
                        color='black'
                    )
                ),
                range=radial_range,
                tickmode='linear',
                tick0=radial_range[0],
                dtick=(radial_range[1] - radial_range[0]) / radial_tick_count,
                nticks=radial_tick_count + 1
            ),
            angularaxis=dict(
                gridcolor=grid_color,
                gridwidth=1 if show_grid else 0,
                linecolor='black',
                linewidth=2,
                rotation=90,
                direction="clockwise",
                tickmode='array',
                tickvals=list(range(0, 361, 45)),
                ticktext=[f'{i}°' for i in range(0, 361, 45)],
                tickfont=dict(
                    size=font_size,
                    family=self.pub_settings.font_settings['family'],
                    color='black'
                ),
                period=360
            ),
            bgcolor=bg_color
        )
        
        # Add habit plane line
        if show_habit_plane:
            fig.add_trace(go.Scatterpolar(
                r=[radial_range[0], radial_range[1]],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(
                    color=habit_plane_color,
                    width=habit_plane_width,
                    dash='dashdot'
                ),
                name=f'Habit Plane ({self.habit_angle}°)',
                hoverinfo='skip',
                showlegend=True
            ))
        
        # Add custom annotations if provided
        if annotations:
            for annotation in annotations:
                fig.add_annotation(
                    dict(
                        x=annotation.get('x', 0.5),
                        y=annotation.get('y', 1.05),
                        text=annotation.get('text', ''),
                        showarrow=annotation.get('showarrow', False),
                        font=dict(
                            size=annotation.get('font_size', font_size),
                            color=annotation.get('color', 'black')
                        ),
                        align=annotation.get('align', 'center')
                    )
                )
        
        # Configure layout
        legend_positions = {
            'right': dict(x=1.1, y=0.5),
            'left': dict(x=-0.1, y=0.5),
            'top': dict(x=0.5, y=1.1),
            'bottom': dict(x=0.5, y=-0.1)
        }
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=font_size + 8,
                    family=self.pub_settings.font_settings['family'],
                    weight='bold',
                    color='darkblue'
                ),
                x=0.5,
                y=0.95
            ),
            polar=polar_layout,
            showlegend=show_legend,
            legend=dict(
                **legend_positions.get(legend_position, legend_positions['right']),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(
                    size=font_size,
                    family=self.pub_settings.font_settings['family']
                ),
                title=dict(
                    text='Defect Types',
                    font=dict(
                        size=font_size + 2,
                        weight='bold'
                    )
                )
            ),
            width=1000,
            height=800,
            margin=dict(l=100, r=150, t=100, b=100),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_habit_plane_vicinity_radar(self, vicinity_data, defect_comparison=None,
                                         title="Habit Plane Vicinity Analysis",
                                         colormap='turbo', show_all_defects=True):
        """Create specialized radar for habit plane vicinity analysis"""
        
        # Focus on vicinity around 54.7 degrees
        vicinity_range = 30.0  # ±30 degrees
        min_angle = self.habit_angle - vicinity_range
        max_angle = self.habit_angle + vicinity_range
        
        # Prepare data dictionary
        data_dict = {}
        
        # Add vicinity sweep data
        if vicinity_data and 'angles' in vicinity_data:
            angles = np.array(vicinity_data['angles'])
            # Filter for vicinity range
            mask = (angles >= min_angle) & (angles <= max_angle)
            if np.any(mask):
                vicinity_angles = angles[mask]
                
                # Add each stress component
                for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                    if comp in vicinity_data.get('stresses', {}):
                        stresses = np.array(vicinity_data['stresses'][comp])[mask]
                        data_dict[f'Vicinity_{comp}'] = {
                            'angles': vicinity_angles,
                            'stresses': stresses
                        }
        
        # Add defect comparison data if available and requested
        if show_all_defects and defect_comparison:
            for key, data in defect_comparison.items():
                if 'angles' in data and 'stresses' in data:
                    angles = np.array(data['angles'])
                    mask = (angles >= min_angle) & (angles <= max_angle)
                    if np.any(mask):
                        defect_angles = angles[mask]
                        defect_stresses = np.array(data['stresses']['sigma_hydro'])[mask]
                        defect_type = data.get('defect_type', key)
                        data_dict[f'{defect_type}'] = {
                            'angles': defect_angles,
                            'stresses': defect_stresses
                        }
        
        if not data_dict:
            # Return empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=24, family="Arial", weight='bold', color='darkblue'),
                    x=0.5
                ),
                annotations=[
                    dict(
                        text="No data available for radar visualization",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16, color='red')
                    )
                ],
                width=800,
                height=600
            )
            return fig
        
        # Custom labels
        custom_labels = {}
        for key in data_dict.keys():
            if key.startswith('Vicinity_'):
                comp = key.replace('Vicinity_', '')
                custom_labels[key] = f"{comp.replace('_', ' ').title()} Sweep"
            else:
                custom_labels[key] = key
        
        # Create radar with enhanced settings
        fig = self.create_customizable_radar(
            data_dict,
            title=title,
            colormap=colormap,
            font_size=16,  # Larger for publication
            show_grid=True,
            grid_color='rgba(100, 100, 100, 0.2)',
            bg_color='rgba(240, 240, 240, 0.05)',
            line_width=3,
            marker_size=10,
            fill_opacity=0.15,
            show_legend=True,
            legend_position='right',
            radial_range=None,  # Auto-calculate
            angular_range=[min_angle, max_angle],
            custom_labels=custom_labels,
            show_habit_plane=True,
            habit_plane_color='rgb(46, 204, 113)',
            habit_plane_width=4,
            radial_tick_count=6
        )
        
        # Update layout for habit plane focus
        fig.update_polars(
            angularaxis=dict(
                tickmode='array',
                tickvals=list(range(int(min_angle), int(max_angle) + 1, 15)),
                ticktext=[f'{i}°' for i in range(int(min_angle), int(max_angle) + 1, 15)],
                tickfont=dict(size=14),
                range=[min_angle, max_angle]
            )
        )
        
        # Add annotation for habit plane
        fig.add_annotation(
            dict(
                x=0.5,
                y=1.05,
                text=f"Focus: {self.habit_angle}° ± {vicinity_range}°",
                showarrow=False,
                font=dict(size=14, color='darkblue', weight='bold'),
                align='center'
            )
        )
        
        return fig
    
    def create_interactive_radar_dashboard(self, vicinity_data, defect_comparison):
        """Create interactive radar dashboard with multiple views"""
        
        tabs = []
        
        # Tab 1: Full range radar
        full_radar = self.create_customizable_radar(
            self._prepare_full_range_data(defect_comparison),
            title="Full Orientation Range - All Defects",
            colormap='viridis',
            font_size=14,
            show_legend=True
        )
        tabs.append(('Full Range', full_radar))
        
        # Tab 2: Habit plane vicinity
        habit_radar = self.create_habit_plane_vicinity_radar(
            vicinity_data,
            defect_comparison,
            title="Habit Plane Vicinity (54.7° ± 30°)",
            colormap='turbo',
            show_all_defects=True
        )
        tabs.append(('Habit Vicinity', habit_radar))
        
        # Tab 3: Stress component comparison
        stress_radar = self._create_stress_component_radar(vicinity_data)
        tabs.append(('Stress Components', stress_radar))
        
        # Tab 4: Sintering temperature radar
        if vicinity_data and 'sintering_temps' in vicinity_data:
            temp_radar = self._create_sintering_radar(vicinity_data)
            tabs.append(('Sintering Temp', temp_radar))
        
        return tabs
    
    def _prepare_full_range_data(self, defect_comparison):
        """Prepare data for full range radar"""
        data_dict = {}
        
        if defect_comparison:
            for key, data in defect_comparison.items():
                defect_type = data.get('defect_type', key)
                if 'angles' in data and 'stresses' in data:
                    data_dict[defect_type] = {
                        'angles': data['angles'],
                        'stresses': data['stresses'].get('sigma_hydro', [])
                    }
        
        return data_dict
    
    def _create_stress_component_radar(self, vicinity_data):
        """Create radar comparing different stress components"""
        data_dict = {}
        
        if vicinity_data and 'angles' in vicinity_data and 'stresses' in vicinity_data:
            angles = vicinity_data['angles']
            stresses = vicinity_data['stresses']
            
            # Focus on habit plane vicinity
            mask = (np.array(angles) >= 54.7 - 30) & (np.array(angles) <= 54.7 + 30)
            if np.any(mask):
                filtered_angles = np.array(angles)[mask]
                
                for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                    if comp in stresses:
                        filtered_stresses = np.array(stresses[comp])[mask]
                        data_dict[comp] = {
                            'angles': filtered_angles,
                            'stresses': filtered_stresses
                        }
        
        return self.create_customizable_radar(
            data_dict,
            title="Stress Component Comparison in Habit Plane Vicinity",
            colormap='RdYlBu_r',
            font_size=15,
            fill_opacity=0.1,
            show_legend=True
        )
    
    def _create_sintering_radar(self, vicinity_data):
        """Create radar for sintering temperatures"""
        data_dict = {}
        
        if vicinity_data and 'angles' in vicinity_data and 'sintering_temps' in vicinity_data:
            angles = vicinity_data['angles']
            sintering_temps = vicinity_data['sintering_temps']
            
            # Focus on habit plane vicinity
            mask = (np.array(angles) >= 54.7 - 30) & (np.array(angles) <= 54.7 + 30)
            if np.any(mask):
                filtered_angles = np.array(angles)[mask]
                
                for model in ['exponential', 'arrhenius_defect']:
                    if model in sintering_temps:
                        filtered_temps = np.array(sintering_temps[model])[mask]
                        data_dict[model] = {
                            'angles': filtered_angles,
                            'stresses': filtered_temps  # Using stresses key for compatibility
                        }
        
        return self.create_customizable_radar(
            data_dict,
            title="Sintering Temperature Models in Habit Plane Vicinity",
            colormap='hot',
            font_size=15,
            fill_opacity=0.1,
            show_legend=True,
            radial_range=[300, 700]  # Reasonable temperature range for sintering
        )

# =============================================
# ENHANCED HABIT PLANE VISUALIZER
# =============================================

class EnhancedHabitPlaneVisualizer(HabitPlaneVisualizer):
    """Enhanced visualizer with publication-quality settings"""
    
    def __init__(self, habit_angle=54.7):
        super().__init__(habit_angle)
        self.pub_settings = PublicationVisualizationSettings()
        self.radar_visualizer = EnhancedRadarVisualizer()
        
    def create_publication_quality_chart(self, angles, stresses, stress_component='sigma_hydro',
                                        title="Habit Plane Vicinity Analysis",
                                        figsize=(12, 8), dpi=300):
        """Create publication-quality matplotlib chart"""
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Convert to numpy arrays
        angles = np.array(angles)
        stresses = np.array(stresses)
        
        # Plot with enhanced settings
        ax.plot(angles, stresses, 
               linewidth=3, 
               color=self.pub_settings.color_settings['stress_colors'].get(stress_component, '#1F77B4'),
               marker='o', 
               markersize=8,
               markeredgecolor='black',
               markeredgewidth=1,
               label=f'{stress_component.replace("_", " ").title()}')
        
        # Highlight habit plane
        ax.axvline(self.habit_angle, 
                  color='green', 
                  linestyle='--', 
                  linewidth=2.5,
                  alpha=0.8,
                  label=f'Habit Plane ({self.habit_angle}°)')
        
        # Add shaded region for vicinity
        vicinity_range = 10.0
        ax.axvspan(self.habit_angle - vicinity_range, 
                  self.habit_angle + vicinity_range, 
                  alpha=0.1, 
                  color='green',
                  label=f'±{vicinity_range}° Vicinity')
        
        # Set labels with publication-quality fonts
        ax.set_xlabel('Orientation Angle (°)', 
                     fontdict=self.pub_settings.get_font_dict('axis_label'))
        ax.set_ylabel(f'{stress_component.replace("_", " ").title()} (GPa)', 
                     fontdict=self.pub_settings.get_font_dict('axis_label'))
        ax.set_title(title, 
                    fontdict=self.pub_settings.get_font_dict('title'),
                    pad=20)
        
        # Configure ticks
        ax.tick_params(axis='both', which='major', 
                      labelsize=self.pub_settings.font_settings['size']['axis_tick'])
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(fontsize=self.pub_settings.font_settings['size']['legend'],
                 frameon=True,
                 framealpha=0.9,
                 loc='best')
        
        # Tight layout
        plt.tight_layout()
        
        return fig, ax
    
    def create_enhanced_sunburst(self, angles, stresses, stress_component='sigma_hydro',
                                title="Habit Plane Vicinity Analysis",
                                colormap='turbo', show_annotations=True):
        """Create enhanced sunburst chart with publication-quality labels"""
        
        # Ensure arrays
        angles = np.array(angles)
        stresses = np.array(stresses)
        
        # Create figure with enhanced settings
        fig = go.Figure()
        
        # Add main trace
        fig.add_trace(go.Scatterpolar(
            r=stresses,
            theta=angles,
            mode='lines+markers',
            marker=dict(
                size=12,  # Larger markers
                color=stresses,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(
                            size=self.pub_settings.font_settings['size']['axis_label'],
                            family=self.pub_settings.font_settings['family']
                        )
                    ),
                    x=1.15,
                    thickness=25,
                    len=0.8,
                    tickfont=dict(
                        size=self.pub_settings.font_settings['size']['axis_tick'],
                        family=self.pub_settings.font_settings['family']
                    )
                ),
                line=dict(width=2, color='black')
            ),
            line=dict(color='rgba(100, 100, 100, 0.5)', width=2),
            name='Stress Distribution',
            hovertemplate=(
                "<b>Habit Plane Vicinity</b><br>" +
                f"<b>{stress_component.replace('_', ' ').title()}</b><br>" +
                "Angle: %{theta:.2f}°<br>" +
                "Stress: %{r:.4f} GPa<br>" +
                "<extra></extra>"
            )
        ))
        
        # Highlight habit plane
        habit_idx = np.argmin(np.abs(angles - self.habit_angle))
        if habit_idx < len(stresses):
            fig.add_trace(go.Scatterpolar(
                r=[stresses[habit_idx]],
                theta=[angles[habit_idx]],
                mode='markers+text',
                marker=dict(
                    size=30,
                    color='rgb(46, 204, 113)',
                    symbol='star',
                    line=dict(width=3, color='black')
                ),
                text=['HABIT PLANE'],
                textposition='top center',
                textfont=dict(
                    size=self.pub_settings.font_settings['size']['annotation'] + 2,
                    color='black',
                    family=self.pub_settings.font_settings['family'],
                    weight='bold'
                ),
                name=f'Habit Plane ({self.habit_angle}°)',
                hovertemplate=(
                    f"<b>Habit Plane ({self.habit_angle}°)</b><br>" +
                    f"Angle: {angles[habit_idx]:.2f}°<br>" +
                    f"Stress: {stresses[habit_idx]:.4f} GPa<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(
                    size=self.pub_settings.font_settings['size']['title'],
                    family=self.pub_settings.font_settings['family'],
                    weight='bold',
                    color='darkblue'
                ),
                x=0.5,
                y=0.95
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    tickfont=dict(
                        size=self.pub_settings.font_settings['size']['axis_tick'],
                        family=self.pub_settings.font_settings['family'],
                        color='black'
                    ),
                    title=dict(
                        text=f"{stress_component.replace('_', ' ').title()} (GPa)",
                        font=dict(
                            size=self.pub_settings.font_settings['size']['axis_label'],
                            family=self.pub_settings.font_settings['family'],
                            weight='bold',
                            color='black'
                        )
                    )
                ),
                angularaxis=dict(
                    gridcolor="rgba(100, 100, 100, 0.3)",
                    gridwidth=2,
                    linecolor="black",
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(0, 361, 30)),
                    ticktext=[f'{i}°' for i in range(0, 361, 30)],
                    tickfont=dict(
                        size=self.pub_settings.font_settings['size']['axis_tick'],
                        family=self.pub_settings.font_settings['family'],
                        color='black'
                    ),
                    period=360
                ),
                bgcolor="rgba(240, 240, 240, 0.1)"
            ),
            showlegend=True,
            legend=dict(
                x=1.15,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                font=dict(
                    size=self.pub_settings.font_settings['size']['legend'],
                    family=self.pub_settings.font_settings['family']
                ),
                title=dict(
                    text='Legend',
                    font=dict(
                        size=self.pub_settings.font_settings['size']['legend'] + 2,
                        weight='bold'
                    )
                )
            ),
            width=1000,
            height=800,
            margin=dict(l=100, r=200, t=100, b=100),
            paper_bgcolor='white'
        )
        
        # Add annotations if requested
        if show_annotations:
            fig.add_annotation(
                dict(
                    x=0.5,
                    y=-0.1,
                    text=f"Analysis focused on {self.habit_angle}° habit plane vicinity",
                    showarrow=False,
                    font=dict(
                        size=self.pub_settings.font_settings['size']['annotation'],
                        color='darkblue'
                    ),
                    align='center'
                )
            )
        
        return fig

# =============================================
# ENHANCED MAIN APPLICATION
# =============================================

def main():
    # Configure Streamlit page with enhanced settings
    st.set_page_config(
        page_title="Ag FCC Twin: Enhanced Habit Plane Analysis",
        layout="wide",
        page_icon="🔬",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': "# Enhanced Habit Plane Analysis\nPublication-quality visualizations for Ag FCC twin analysis"
        }
    )
    
    # Enhanced CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2.0rem !important;
        color: #374151 !important;
        font-weight: 700 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05));
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .pub-quality {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        border: 3px solid #4F46E5;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    .visualization-controls {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #E5E7EB;
        margin: 1rem 0;
    }
    .radar-customization {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #F59E0B;
        margin: 1rem 0;
    }
    .defect-highlight {
        border: 3px solid;
        border-radius: 0.8rem;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .defect-highlight:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
    }
    .isf-highlight { border-color: #FF6B6B; background-color: rgba(255, 107, 107, 0.08); }
    .esf-highlight { border-color: #4ECDC4; background-color: rgba(78, 205, 196, 0.08); }
    .twin-highlight { border-color: #45B7D1; background-color: rgba(69, 183, 209, 0.08); }
    .perfect-highlight { border-color: #96CEB4; background-color: rgba(150, 206, 180, 0.08); }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 0.8rem;
        border: 2px solid #E5E7EB;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.15);
    }
    .metric-value {
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        color: #1E3A8A !important;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem !important;
        color: #6B7280 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .habit-plane-banner {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        font-weight: bold;
        border: 3px solid #047857;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
        margin: 1.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        padding: 0 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 2rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        background-color: #F3F4F6;
        border: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E5E7EB;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        border-color: #2563EB;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    .tab-content {
        padding: 2rem;
        background-color: white;
        border-radius: 0.5rem;
        border: 2px solid #E5E7EB;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .custom-slider {
        padding: 1rem 0;
    }
    .stButton > button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🔬 Ag FCC Twin: Enhanced Habit Plane Analysis</h1>', unsafe_allow_html=True)
    
    # Publication quality banner
    st.markdown("""
    <div class="pub-quality">
    <div style="display: flex; align-items: center; justify-content: center; gap: 1rem;">
    <div style="font-size: 3rem;">📊</div>
    <div>
    <h2 style="margin: 0; color: white;">Publication-Quality Visualizations</h2>
    <p style="margin: 0.5rem 0 0 0; color: white; opacity: 0.9; font-size: 1.1rem;">
    Enhanced charts with larger labels, customizable radar views, and extensive colormap options
    </p>
    </div>
    <div style="font-size: 3rem;">🎨</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize enhanced components
    pub_settings = PublicationVisualizationSettings()
    radar_visualizer = EnhancedRadarVisualizer()
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedHabitPlaneVisualizer()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator()
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Enhanced Controls</h2>', unsafe_allow_html=True)
        
        # Visualization Settings Section
        st.markdown('<div class="visualization-controls">', unsafe_allow_html=True)
        st.markdown("#### 🎨 Visualization Settings")
        
        # Font size controls
        font_size_mode = st.radio(
            "Font Size Mode",
            ["Publication", "Presentation", "Custom"],
            index=0,
            help="Select font size preset"
        )
        
        if font_size_mode == "Custom":
            col_font1, col_font2 = st.columns(2)
            with col_font1:
                title_size = st.slider("Title Size", 12, 32, 22, 1)
            with col_font2:
                label_size = st.slider("Label Size", 10, 24, 16, 1)
        else:
            presets = {
                "Publication": {"title": 22, "label": 16},
                "Presentation": {"title": 28, "label": 20}
            }
            title_size = presets[font_size_mode]["title"]
            label_size = presets[font_size_mode]["label"]
        
        # Colormap selection
        st.markdown("##### 🌈 Colormap Selection")
        colormap_categories = pub_settings.get_colormap_list()
        
        colormap_category = st.selectbox(
            "Colormap Category",
            list(colormap_categories.keys()),
            index=0
        )
        
        selected_colormap = st.selectbox(
            "Select Colormap",
            colormap_categories[colormap_category],
            index=0 if 'viridis' in colormap_categories[colormap_category] else 0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Radar Customization Section
        st.markdown('<div class="radar-customization">', unsafe_allow_html=True)
        st.markdown("#### 📡 Radar View Customization")
        
        radar_font_size = st.slider(
            "Radar Font Size",
            min_value=10,
            max_value=24,
            value=16,
            step=1,
            help="Font size for radar chart labels"
        )
        
        line_width = st.slider(
            "Line Width",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="Line width for radar traces"
        )
        
        marker_size = st.slider(
            "Marker Size",
            min_value=4,
            max_value=20,
            value=10,
            step=1,
            help="Marker size for data points"
        )
        
        fill_opacity = st.slider(
            "Fill Opacity",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="Opacity of filled areas"
        )
        
        show_grid = st.checkbox("Show Grid", value=True)
        show_legend = st.checkbox("Show Legend", value=True)
        
        legend_position = st.selectbox(
            "Legend Position",
            ["right", "left", "top", "bottom"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Defect Selection
        st.markdown("#### 🔬 Defect Selection")
        
        defect_type = st.selectbox(
            "Defect Type",
            ["Twin", "ISF", "ESF", "No Defect"],
            index=0,
            help="Select defect type for analysis"
        )
        
        # Show defect properties
        defect_properties = {
            "Twin": {"eigen_strain": 2.12, "color": "#45B7D1", "desc": "Coherent Twin Boundary"},
            "ISF": {"eigen_strain": 0.71, "color": "#FF6B6B", "desc": "Intrinsic Stacking Fault"},
            "ESF": {"eigen_strain": 1.41, "color": "#4ECDC4", "desc": "Extrinsic Stacking Fault"},
            "No Defect": {"eigen_strain": 0.0, "color": "#96CEB4", "desc": "Perfect Crystal"}
        }
        
        defect_info = defect_properties[defect_type]
        st.markdown(f"""
        <div class="defect-highlight {'twin' if defect_type == 'Twin' else 'isf' if defect_type == 'ISF' else 'esf' if defect_type == 'ESF' else 'perfect'}-highlight">
        <div style="display: flex; align-items: center; gap: 1rem;">
        <div style="width: 20px; height: 20px; background-color: {defect_info['color']}; border-radius: 50%;"></div>
        <div>
        <strong style="font-size: 1.1rem;">{defect_type}</strong><br>
        <span style="font-size: 0.9rem; color: #666;">{defect_info['desc']}</span>
        </div>
        </div>
        <div style="margin-top: 0.8rem;">
        <strong>Eigen Strain (ε*):</strong> {defect_info['eigen_strain']}<br>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Vicinity Settings
        st.markdown("#### 🎯 Vicinity Settings")
        
        vicinity_range = st.slider(
            "Vicinity Range (± degrees)",
            min_value=1.0,
            max_value=45.0,
            value=30.0,
            step=1.0,
            help="Range around habit plane for detailed analysis"
        )
        
        n_points = st.slider(
            "Number of Points",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
            help="Points in orientation sweep"
        )
        
        # Analysis Type
        analysis_type = st.radio(
            "Analysis Type",
            ["Habit Plane Vicinity", "Defect Comparison", "Comprehensive Dashboard", "Custom Radar"],
            index=0
        )
        
        # Generate Button
        st.markdown("---")
        generate_col1, generate_col2 = st.columns(2)
        with generate_col1:
            if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):
                st.session_state.generate_analysis = True
                st.session_state.analysis_type = analysis_type
        with generate_col2:
            if st.button("🔄 Reset Settings", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
    # Main content area
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "📊 Overview",
        "🎯 Habit Plane Analysis",
        "📡 Radar Views",
        "📈 Comprehensive Analysis"
    ])
    
    with main_tab1:
        st.markdown('<h2 class="sub-header">📊 Analysis Overview</h2>', unsafe_allow_html=True)
        
        # Habit plane banner
        st.markdown(f"""
        <div class="habit-plane-banner">
        <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
        <h2 style="margin: 0; color: white;">🎯 Ag FCC Twin Habit Plane: 54.7°</h2>
        <p style="margin: 0.5rem 0 0 0; color: white; opacity: 0.9;">
        • {{111}} crystal planes • Maximum Schmid factor • Optimal defect engineering
        </p>
        </div>
        <div style="font-size: 3rem;">⚛️</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">Publication Quality</div>
            <div class="metric-value">300 DPI</div>
            <div style="font-size: 0.9rem; color: #6B7280;">High-resolution output</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">Colormaps</div>
            <div class="metric-value">50+</div>
            <div style="font-size: 0.9rem; color: #6B7280;">Available options</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">Defect Types</div>
            <div class="metric-value">4</div>
            <div style="font-size: 0.9rem; color: #6B7280;">TWIN, ESF, ISF, Perfect</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-label">Vicinity Range</div>
            <div class="metric-value">±30°</div>
            <div style="font-size: 0.9rem; color: #6B7280;">Around habit plane</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Defect comparison
        st.markdown('<h3 class="sub-header">🔬 Defect Properties Comparison</h3>', unsafe_allow_html=True)
        
        defect_cols = st.columns(4)
        defect_data = [
            ("TWIN", 2.12, "Maximum strain, optimal for sintering", "#45B7D1"),
            ("ESF", 1.41, "Intermediate strain, good diffusion", "#4ECDC4"),
            ("ISF", 0.71, "Lower strain, moderate effects", "#FF6B6B"),
            ("Perfect", 0.0, "Reference, minimal effects", "#96CEB4")
        ]
        
        for idx, (name, strain, desc, color) in enumerate(defect_data):
            with defect_cols[idx]:
                st.markdown(f"""
                <div class="defect-highlight {'twin' if name == 'TWIN' else 'esf' if name == 'ESF' else 'isf' if name == 'ISF' else 'perfect'}-highlight">
                <div style="text-align: center;">
                <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin-bottom: 0.5rem;">{name}</div>
                <div style="font-size: 1.3rem; font-weight: bold; color: #333; margin: 0.8rem 0;">ε* = {strain}</div>
                <div style="font-size: 0.9rem; color: #666; line-height: 1.4;">{desc}</div>
                </div>
                </div>
                """, unsafe_allow_html=True)
    
    with main_tab2:
        st.markdown('<h2 class="sub-header">🎯 Habit Plane Vicinity Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.get('generate_analysis', False) and st.session_state.get('analysis_type') == "Habit Plane Vicinity":
            
            with st.spinner("🚀 Performing enhanced habit plane analysis..."):
                # Load solutions if not already loaded
                if not st.session_state.solutions:
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                
                if st.session_state.solutions:
                    # Prepare target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'shape': 'Square',
                        'eps0': defect_info['eigen_strain'],
                        'kappa': 0.6
                    }
                    
                    # Perform vicinity sweep
                    vicinity_sweep = st.session_state.interpolator.create_vicinity_sweep(
                        st.session_state.solutions,
                        target_params,
                        vicinity_range=vicinity_range,
                        n_points=n_points,
                        region_type='bulk'
                    )
                    
                    if vicinity_sweep:
                        st.success(f"✅ Generated vicinity analysis with {n_points} points (±{vicinity_range}°)")
                        
                        # Store in session state
                        st.session_state.vicinity_sweep = vicinity_sweep
                        
                        # Display key metrics
                        st.markdown('<h3 class="sub-header">📈 Key Metrics at Habit Plane</h3>', unsafe_allow_html=True)
                        
                        # Calculate habit plane metrics
                        angles = np.array(vicinity_sweep['angles'])
                        habit_idx = np.argmin(np.abs(angles - 54.7))
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            sigma_h = vicinity_sweep['stresses']['sigma_hydro'][habit_idx]
                            st.metric(
                                "σ_hydro",
                                f"{sigma_h:.3f} GPa",
                                "Hydrostatic Stress"
                            )
                        
                        with metric_cols[1]:
                            sigma_vm = vicinity_sweep['stresses']['von_mises'][habit_idx]
                            st.metric(
                                "σ_von Mises",
                                f"{sigma_vm:.3f} GPa",
                                "Equivalent Stress"
                            )
                        
                        with metric_cols[2]:
                            T_sinter = vicinity_sweep['sintering_temps']['arrhenius_defect'][habit_idx]
                            st.metric(
                                "T_sinter",
                                f"{T_sinter:.1f} K",
                                f"{T_sinter-273.15:.1f}°C"
                            )
                        
                        with metric_cols[3]:
                            temp_reduction = 623.0 - T_sinter
                            st.metric(
                                "ΔT Reduction",
                                f"{temp_reduction:.1f} K",
                                "From reference"
                            )
                        
                        # Create enhanced visualizations
                        st.markdown('<h3 class="sub-header">📊 Enhanced Visualizations</h3>', unsafe_allow_html=True)
                        
                        # Sunburst chart
                        st.markdown("##### 🌟 Polar Visualization (Sunburst)")
                        fig_sunburst = st.session_state.visualizer.create_enhanced_sunburst(
                            vicinity_sweep['angles'],
                            vicinity_sweep['stresses']['sigma_hydro'],
                            stress_component='sigma_hydro',
                            title=f"Habit Plane Vicinity - {defect_type}",
                            colormap=selected_colormap,
                            show_annotations=True
                        )
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                        
                        # Line plots
                        st.markdown("##### 📈 Stress Component Comparison")
                        
                        fig_line, axes = plt.subplots(1, 3, figsize=(18, 6))
                        
                        components = ['sigma_hydro', 'von_mises', 'sigma_mag']
                        colors = ['#1F77B4', '#FF7F0E', '#2CA02C']
                        
                        for idx, (comp, color) in enumerate(zip(components, colors)):
                            ax = axes[idx]
                            ax.plot(
                                vicinity_sweep['angles'],
                                vicinity_sweep['stresses'][comp],
                                color=color,
                                linewidth=3,
                                marker='o',
                                markersize=6,
                                markeredgecolor='black'
                            )
                            ax.axvline(54.7, color='green', linestyle='--', linewidth=2, alpha=0.7)
                            ax.set_xlabel('Orientation (°)', fontsize=label_size)
                            ax.set_ylabel(f'{comp.replace("_", " ").title()} (GPa)', fontsize=label_size)
                            ax.set_title(f'{comp.replace("_", " ").title()} Stress', fontsize=title_size, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.tick_params(axis='both', labelsize=label_size-2)
                            
                            # Add vicinity shading
                            ax.axvspan(54.7 - vicinity_range, 54.7 + vicinity_range, alpha=0.1, color='green')
                        
                        plt.tight_layout()
                        st.pyplot(fig_line)
                        plt.close(fig_line)
                        
                        # Sintering temperature analysis
                        st.markdown("##### 🔥 Sintering Temperature Analysis")
                        
                        fig_temp, ax_temp = plt.subplots(figsize=(12, 6))
                        
                        ax_temp.plot(
                            vicinity_sweep['angles'],
                            vicinity_sweep['sintering_temps']['exponential'],
                            color='red',
                            linewidth=3,
                            label='Exponential Model',
                            marker='s',
                            markersize=6
                        )
                        
                        ax_temp.plot(
                            vicinity_sweep['angles'],
                            vicinity_sweep['sintering_temps']['arrhenius_defect'],
                            color='blue',
                            linewidth=3,
                            linestyle='--',
                            label='Arrhenius Model',
                            marker='^',
                            markersize=6
                        )
                        
                        ax_temp.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane')
                        ax_temp.axvspan(54.7 - vicinity_range, 54.7 + vicinity_range, alpha=0.1, color='green')
                        
                        ax_temp.set_xlabel('Orientation (°)', fontsize=label_size)
                        ax_temp.set_ylabel('Sintering Temperature (K)', fontsize=label_size)
                        ax_temp.set_title('Sintering Temperature Prediction', fontsize=title_size, fontweight='bold')
                        ax_temp.legend(fontsize=label_size-2)
                        ax_temp.grid(True, alpha=0.3)
                        ax_temp.tick_params(axis='both', labelsize=label_size-2)
                        
                        # Add Celsius secondary axis
                        ax_temp2 = ax_temp.twinx()
                        celsius_ticks = ax_temp.get_yticks()
                        ax_temp2.set_ylim(ax_temp.get_ylim())
                        ax_temp2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
                        ax_temp2.set_ylabel('Temperature (°C)', fontsize=label_size)
                        ax_temp2.tick_params(axis='y', labelsize=label_size-2)
                        
                        plt.tight_layout()
                        st.pyplot(fig_temp)
                        plt.close(fig_temp)
                        
                    else:
                        st.error("Failed to generate vicinity analysis. Please check your data.")
                else:
                    st.warning("No solutions loaded. Please load data first.")
        else:
            st.info("👈 Configure analysis settings in the sidebar and click 'Generate Analysis'")
            
            # Show example visualization
            st.markdown("#### 📊 Example Visualization")
            
            # Create example data for demonstration
            example_angles = np.linspace(54.7 - 30, 54.7 + 30, 100)
            example_stress = 25 * np.exp(-(example_angles - 54.7)**2 / (2*10**2)) + 3 * np.sin(np.radians(example_angles * 2))
            example_temp = 623 * np.exp(-example_stress / 30) + 50 * np.cos(np.radians(example_angles - 54.7))
            
            fig_example, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Stress plot
            ax1.plot(example_angles, example_stress, 'b-', linewidth=3)
            ax1.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane (54.7°)')
            ax1.fill_between(example_angles, example_stress, alpha=0.2, color='blue')
            ax1.set_xlabel('Orientation (°)', fontsize=14)
            ax1.set_ylabel('Hydrostatic Stress (GPa)', fontsize=14)
            ax1.set_title('Example: Stress Concentration at Habit Plane', fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Temperature plot
            ax2.plot(example_angles, example_temp, 'r-', linewidth=3)
            ax2.axvline(54.7, color='green', linestyle='--', linewidth=2, label='Habit Plane (54.7°)')
            ax2.fill_between(example_angles, example_temp, alpha=0.2, color='red')
            ax2.set_xlabel('Orientation (°)', fontsize=14)
            ax2.set_ylabel('Sintering Temperature (K)', fontsize=14)
            ax2.set_title('Example: Temperature Reduction at Habit Plane', fontsize=16, fontweight='bold')
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_example)
            plt.close(fig_example)
    
    with main_tab3:
        st.markdown('<h2 class="sub-header">📡 Enhanced Radar Views</h2>', unsafe_allow_html=True)
        
        if st.session_state.get('generate_analysis', False) and (
            st.session_state.get('analysis_type') == "Custom Radar" or 
            st.session_state.get('analysis_type') == "Defect Comparison"
        ):
            
            with st.spinner("🔄 Creating enhanced radar visualizations..."):
                # Load or generate data
                if 'vicinity_sweep' not in st.session_state:
                    # Need to generate data first
                    if not st.session_state.solutions:
                        st.session_state.solutions = st.session_state.loader.load_all_solutions()
                    
                    target_params = {
                        'defect_type': defect_type,
                        'shape': 'Square',
                        'eps0': defect_info['eigen_strain'],
                        'kappa': 0.6
                    }
                    
                    vicinity_sweep = st.session_state.interpolator.create_vicinity_sweep(
                        st.session_state.solutions,
                        target_params,
                        vicinity_range=vicinity_range,
                        n_points=n_points,
                        region_type='bulk'
                    )
                    st.session_state.vicinity_sweep = vicinity_sweep
                
                # Generate defect comparison if needed
                if st.session_state.get('analysis_type') == "Defect Comparison":
                    defect_comparison = st.session_state.interpolator.compare_defect_types(
                        st.session_state.solutions,
                        angle_range=(54.7 - vicinity_range, 54.7 + vicinity_range),
                        n_points=n_points,
                        region_type='bulk'
                    )
                    st.session_state.defect_comparison = defect_comparison
                
                # Create radar dashboard
                st.markdown("#### 📊 Radar Visualization Dashboard")
                
                # Radar type selection
                radar_type = st.radio(
                    "Select Radar View",
                    ["Habit Plane Vicinity", "Defect Comparison", "Stress Components", "Sintering Temperature"],
                    horizontal=True
                )
                
                if radar_type == "Habit Plane Vicinity":
                    # Prepare data for habit plane vicinity radar
                    vicinity_data = st.session_state.vicinity_sweep
                    
                    # Filter for vicinity range
                    angles = np.array(vicinity_data['angles'])
                    mask = (angles >= 54.7 - vicinity_range) & (angles <= 54.7 + vicinity_range)
                    
                    if np.any(mask):
                        filtered_data = {
                            'angles': angles[mask].tolist(),
                            'stresses': {
                                comp: np.array(vals)[mask].tolist() 
                                for comp, vals in vicinity_data['stresses'].items()
                            }
                        }
                        
                        # Create data dictionary for radar
                        data_dict = {}
                        for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                            if comp in filtered_data['stresses']:
                                data_dict[comp] = {
                                    'angles': filtered_data['angles'],
                                    'stresses': filtered_data['stresses'][comp]
                                }
                        
                        # Create radar
                        fig_radar = radar_visualizer.create_customizable_radar(
                            data_dict,
                            title=f"Habit Plane Vicinity Analysis - {defect_type}",
                            colormap=selected_colormap,
                            font_size=radar_font_size,
                            show_grid=show_grid,
                            line_width=line_width,
                            marker_size=marker_size,
                            fill_opacity=fill_opacity,
                            show_legend=show_legend,
                            legend_position=legend_position,
                            angular_range=[54.7 - vicinity_range, 54.7 + vicinity_range],
                            show_habit_plane=True,
                            habit_plane_color='rgb(46, 204, 113)',
                            habit_plane_width=4
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)
                
                elif radar_type == "Defect Comparison" and 'defect_comparison' in st.session_state:
                    # Prepare data for defect comparison radar
                    defect_comparison = st.session_state.defect_comparison
                    
                    # Create data dictionary
                    data_dict = {}
                    for key, data in defect_comparison.items():
                        defect_name = data.get('defect_type', key)
                        if 'angles' in data and 'stresses' in data:
                            data_dict[defect_name] = {
                                'angles': data['angles'],
                                'stresses': data['stresses'].get('sigma_hydro', [])
                            }
                    
                    # Create radar
                    fig_radar = radar_visualizer.create_customizable_radar(
                        data_dict,
                        title="Defect Type Comparison in Habit Plane Vicinity",
                        colormap=selected_colormap,
                        font_size=radar_font_size,
                        show_grid=show_grid,
                        line_width=line_width,
                        marker_size=marker_size,
                        fill_opacity=fill_opacity,
                        show_legend=show_legend,
                        legend_position=legend_position,
                        angular_range=[54.7 - vicinity_range, 54.7 + vicinity_range],
                        custom_labels={k: f"{k} (ε*={defect_properties.get(k, {}).get('eigen_strain', 0):.2f})" 
                                     for k in data_dict.keys()},
                        show_habit_plane=True
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                elif radar_type == "Stress Components" and 'vicinity_sweep' in st.session_state:
                    # Create stress component radar
                    fig_radar = radar_visualizer._create_stress_component_radar(
                        st.session_state.vicinity_sweep
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                elif radar_type == "Sintering Temperature" and 'vicinity_sweep' in st.session_state:
                    # Create sintering temperature radar
                    fig_radar = radar_visualizer._create_sintering_radar(
                        st.session_state.vicinity_sweep
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Download options
                st.markdown("#### 💾 Export Options")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    if st.button("📸 Save as PNG", use_container_width=True):
                        st.info("Use Plotly's camera icon in the chart to save as PNG")
                
                with col_dl2:
                    if st.button("📊 Export Data", use_container_width=True):
                        # Prepare data for export
                        if 'vicinity_sweep' in st.session_state:
                            data = st.session_state.vicinity_sweep
                            df = pd.DataFrame({
                                'angle_deg': data['angles'],
                                'sigma_hydro_gpa': data['stresses']['sigma_hydro'],
                                'von_mises_gpa': data['stresses']['von_mises'],
                                'sigma_mag_gpa': data['stresses']['sigma_mag'],
                                'T_sinter_exp_k': data['sintering_temps']['exponential'],
                                'T_sinter_arr_k': data['sintering_temps']['arrhenius_defect']
                            })
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"radar_data_{defect_type}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                
                with col_dl3:
                    if st.button("🖼️ High-Res Export", use_container_width=True):
                        st.info("For publication-quality figures, use 300 DPI PNG or PDF format")
        
        else:
            st.info("👈 Select 'Custom Radar' or 'Defect Comparison' analysis type and click 'Generate Analysis'")
            
            # Show radar customization options
            st.markdown("""
            #### 🎨 Radar Customization Features
            
            The enhanced radar visualizer includes:
            
            1. **Extensive Colormap Support**: 50+ colormaps including:
               - Perceptually uniform (viridis, plasma, turbo)
               - Sequential (Blues, Reds, Greens)
               - Diverging (RdBu, RdYlBu, coolwarm)
               - Cyclic (twilight, hsv)
               - Qualitative (Set1, Set2, Set3)
            
            2. **Customizable Appearance**:
               - Adjustable font sizes (10-24 pt)
               - Line width control (1-8 px)
               - Marker size customization (4-20 px)
               - Fill opacity control
               - Grid visibility toggle
            
            3. **Habit Plane Focus**:
               - Automatic focus on 54.7° ± custom range
               - Habit plane highlight with customizable color
               - Angular range control
            
            4. **Interactive Features**:
               - Hover information with defect details
               - Legend positioning (right, left, top, bottom)
               - Download options for publication
            """)
    
    with main_tab4:
        st.markdown('<h2 class="sub-header">📈 Comprehensive Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        if st.session_state.get('generate_analysis', False) and st.session_state.get('analysis_type') == "Comprehensive Dashboard":
            
            with st.spinner("🔄 Building comprehensive dashboard..."):
                # Check if we have all required data
                needs_data = []
                if 'vicinity_sweep' not in st.session_state:
                    needs_data.append("vicinity sweep")
                if 'defect_comparison' not in st.session_state:
                    needs_data.append("defect comparison")
                
                if needs_data:
                    st.warning(f"Please run {' and '.join(needs_data)} analysis first")
                else:
                    # Create comprehensive dashboard
                    vicinity_sweep = st.session_state.vicinity_sweep
                    defect_comparison = st.session_state.defect_comparison
                    
                    st.success("✅ Comprehensive dashboard generated")
                    
                    # Create tabs for different views
                    dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Overview", "Comparative Analysis", "Export"])
                    
                    with dash_tab1:
                        # Overview dashboard
                        st.markdown("#### 📊 Comprehensive Overview")
                        
                        # Create subplots
                        fig = make_subplots(
                            rows=2, cols=3,
                            subplot_titles=(
                                'Hydrostatic Stress Vicinity',
                                'Von Mises Stress Vicinity',
                                'Stress Magnitude Vicinity',
                                'Defect Comparison',
                                'Sintering Temperature',
                                'Habit Plane Radar'
                            ),
                            specs=[
                                [{'type': 'polar'}, {'type': 'polar'}, {'type': 'polar'}],
                                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'polar'}]
                            ],
                            vertical_spacing=0.1,
                            horizontal_spacing=0.15
                        )
                        
                        # Add polar plots for stress components
                        angles = vicinity_sweep['angles']
                        
                        for idx, (comp, color) in enumerate([
                            ('sigma_hydro', '#1F77B4'),
                            ('von_mises', '#FF7F0E'),
                            ('sigma_mag', '#2CA02C')
                        ]):
                            if comp in vicinity_sweep['stresses']:
                                fig.add_trace(
                                    go.Scatterpolar(
                                        r=vicinity_sweep['stresses'][comp],
                                        theta=angles,
                                        mode='lines',
                                        line=dict(color=color, width=2),
                                        name=comp.replace('_', ' ').title()
                                    ),
                                    row=1, col=idx+1
                                )
                        
                        # Add defect comparison
                        if defect_comparison:
                            for key, data in list(defect_comparison.items())[:4]:  # Limit to 4
                                defect_type = data.get('defect_type', key)
                                color = data.get('color', '#000000')
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=data['angles'],
                                        y=data['stresses']['sigma_hydro'],
                                        mode='lines',
                                        line=dict(color=color, width=2),
                                        name=defect_type,
                                        showlegend=False
                                    ),
                                    row=2, col=1
                                )
                        
                        # Add sintering temperature
                        if 'sintering_temps' in vicinity_sweep:
                            fig.add_trace(
                                go.Scatter(
                                    x=angles,
                                    y=vicinity_sweep['sintering_temps']['arrhenius_defect'],
                                    mode='lines',
                                    line=dict(color='red', width=2),
                                    name='Sintering Temp',
                                    showlegend=False
                                ),
                                row=2, col=2
                            )
                        
                        # Add habit plane radar
                        # Prepare data for radar subplot
                        radar_data = {}
                        for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                            if comp in vicinity_sweep['stresses']:
                                radar_data[comp] = {
                                    'angles': angles,
                                    'stresses': vicinity_sweep['stresses'][comp]
                                }
                        
                        # Add radar traces
                        for comp, color in [
                            ('sigma_hydro', '#1F77B4'),
                            ('von_mises', '#FF7F0E'),
                            ('sigma_mag', '#2CA02C')
                        ]:
                            if comp in radar_data:
                                data = radar_data[comp]
                                angles_closed = np.append(data['angles'], data['angles'][0])
                                stresses_closed = np.append(data['stresses'], data['stresses'][0])
                                
                                fig.add_trace(
                                    go.Scatterpolar(
                                        r=stresses_closed,
                                        theta=angles_closed,
                                        mode='lines',
                                        line=dict(color=color, width=1.5),
                                        name=comp.replace('_', ' ').title(),
                                        showlegend=False
                                    ),
                                    row=2, col=3
                                )
                        
                        # Update layout
                        fig.update_layout(
                            height=900,
                            showlegend=True,
                            legend=dict(
                                x=1.02,
                                y=0.5,
                                font=dict(size=12)
                            ),
                            title=dict(
                                text=f"Comprehensive Analysis Dashboard - {defect_type}",
                                font=dict(size=20, weight='bold')
                            )
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
                            row=2, col=3
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with dash_tab2:
                        # Comparative analysis
                        st.markdown("#### 🔬 Comparative Analysis")
                        
                        # Create comparative metrics
                        col_comp1, col_comp2, col_comp3 = st.columns(3)
                        
                        with col_comp1:
                            # Stress maxima comparison
                            st.markdown("##### 📈 Maximum Stresses")
                            max_stresses = {}
                            for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                                if comp in vicinity_sweep['stresses']:
                                    max_stresses[comp] = max(vicinity_sweep['stresses'][comp])
                            
                            df_max = pd.DataFrame({
                                'Component': list(max_stresses.keys()),
                                'Max Stress (GPa)': list(max_stresses.values())
                            })
                            st.dataframe(df_max, use_container_width=True)
                        
                        with col_comp2:
                            # Temperature comparison
                            st.markdown("##### 🌡️ Temperature Range")
                            temp_data = {}
                            for model in ['exponential', 'arrhenius_defect']:
                                if model in vicinity_sweep['sintering_temps']:
                                    temps = vicinity_sweep['sintering_temps'][model]
                                    temp_data[model] = {
                                        'min': min(temps),
                                        'max': max(temps),
                                        'range': max(temps) - min(temps)
                                    }
                            
                            df_temp = pd.DataFrame(temp_data).T
                            st.dataframe(df_temp, use_container_width=True)
                        
                        with col_comp3:
                            # Defect comparison summary
                            st.markdown("##### 🔬 Defect Comparison")
                            defect_summary = []
                            if defect_comparison:
                                for key, data in defect_comparison.items():
                                    defect_type = data.get('defect_type', key)
                                    stresses = data['stresses']['sigma_hydro']
                                    defect_summary.append({
                                        'Defect': defect_type,
                                        'Max σ_h (GPa)': f"{max(stresses):.3f}",
                                        'Avg σ_h (GPa)': f"{np.mean(stresses):.3f}"
                                    })
                            
                            if defect_summary:
                                df_defect = pd.DataFrame(defect_summary)
                                st.dataframe(df_defect, use_container_width=True)
                    
                    with dash_tab3:
                        # Export options
                        st.markdown("#### 📤 Comprehensive Export")
                        
                        st.write("""
                        Export comprehensive analysis package including:
                        1. All visualization figures (PNG, PDF)
                        2. Raw data files (CSV)
                        3. Analysis report (PDF)
                        4. Configuration settings (JSON)
                        """)
                        
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            export_format = st.selectbox(
                                "Export Format",
                                ["PNG (300 DPI)", "PDF", "SVG", "All Formats"],
                                index=0
                            )
                        
                        with col_exp2:
                            include_data = st.checkbox("Include Raw Data", value=True)
                            include_report = st.checkbox("Include Analysis Report", value=True)
                        
                        with col_exp3:
                            if st.button("📦 Generate Export Package", use_container_width=True):
                                with st.spinner("Generating export package..."):
                                    # Create timestamp
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    
                                    # Prepare data for export
                                    export_data = {
                                        'metadata': {
                                            'generated_at': timestamp,
                                            'defect_type': defect_type,
                                            'vicinity_range': vicinity_range,
                                            'n_points': n_points
                                        },
                                        'vicinity_data': vicinity_sweep,
                                        'defect_comparison': list(defect_comparison.keys()) if defect_comparison else []
                                    }
                                    
                                    # Create JSON export
                                    json_str = json.dumps(export_data, indent=2)
                                    
                                    st.download_button(
                                        label="📥 Download JSON",
                                        data=json_str,
                                        file_name=f"comprehensive_analysis_{defect_type}_{timestamp}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
        else:
            st.info("👈 Select 'Comprehensive Dashboard' analysis type and click 'Generate Analysis'")
            
            # Show dashboard features
            st.markdown("""
            #### 📊 Comprehensive Dashboard Features
            
            The comprehensive dashboard provides:
            
            1. **Multi-Panel Visualization**:
               - Polar plots for stress components
               - Defect comparison charts
               - Sintering temperature analysis
               - Integrated radar views
            
            2. **Comparative Analysis**:
               - Maximum stress comparisons
               - Temperature range analysis
               - Defect performance metrics
               - Habit plane specific metrics
            
            3. **Export Capabilities**:
               - High-resolution figure export (300 DPI)
               - Raw data export (CSV format)
               - Comprehensive report generation
               - Configuration file export
            
            4. **Publication-Ready Output**:
               - Proper font sizing and labeling
               - Consistent color schemes
               - Professional layout
               - Clear annotations
            """)

# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
