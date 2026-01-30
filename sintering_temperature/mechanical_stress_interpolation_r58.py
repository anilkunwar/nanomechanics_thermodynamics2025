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
from scipy.ndimage import zoom
import re
import time
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# =============================================
# GLOBAL STYLING CONFIGURATION
# =============================================
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
# ENHANCED WEIGHT VISUALIZER WITH SANKEY AND CHORD DIAGRAMS
# =============================================
class WeightVisualizer:
    """Enhanced visualizer for weight analysis with hierarchical attributes"""
    
    def __init__(self):
        # Color scheme for different components
        self.color_scheme = {
            # Defect types
            'Twin': '#FF6B6B',
            'ISF': '#4ECDC4',
            'ESF': '#95E1D3',
            'No Defect': '#FFD93D',
            
            # Weight components
            'spatial_kernel': '#45B7D1',
            'learned_attention': '#FFA07A',
            'defect_mask': '#95E1D3',
            'combined_weight': '#9D4EDD',
            
            # Special
            'query': '#9D4EDD',
            'target': '#FF1493',
            'match': '#2E8B57',
            'mismatch': '#DC143C'
        }
        
        # Domain configuration
        self.domain_size_nm = 12.8
        
    def prepare_hierarchical_data(self, sources_data, target_params):
        """Prepare hierarchical data structure for visualization"""
        hierarchical_data = {
            'defect_types': {},
            'angular_bins': {},
            'sources': []
        }
        
        # Group by defect type
        for source in sources_data:
            defect_type = source['defect_type']
            if defect_type not in hierarchical_data['defect_types']:
                hierarchical_data['defect_types'][defect_type] = {
                    'total_weight': 0,
                    'sources': [],
                    'count': 0
                }
            
            # Add source to defect type group
            hierarchical_data['defect_types'][defect_type]['total_weight'] += source['combined_weight']
            hierarchical_data['defect_types'][defect_type]['sources'].append(source)
            hierarchical_data['defect_types'][defect_type]['count'] += 1
            
            # Group by angular bins (30-degree intervals)
            angle_bin = f"{int(source['theta_deg']/30)*30}°-{int(source['theta_deg']/30)*30+30}°"
            if angle_bin not in hierarchical_data['angular_bins']:
                hierarchical_data['angular_bins'][angle_bin] = {
                    'total_weight': 0,
                    'sources': [],
                    'defect_distribution': {}
                }
            
            hierarchical_data['angular_bins'][angle_bin]['total_weight'] += source['combined_weight']
            hierarchical_data['angular_bins'][angle_bin]['sources'].append(source)
            
            # Track defect distribution within angular bin
            if defect_type not in hierarchical_data['angular_bins'][angle_bin]['defect_distribution']:
                hierarchical_data['angular_bins'][angle_bin]['defect_distribution'][defect_type] = 0
            hierarchical_data['angular_bins'][angle_bin]['defect_distribution'][defect_type] += source['combined_weight']
            
            # Store individual source
            hierarchical_data['sources'].append(source)
        
        return hierarchical_data
    
    def create_hierarchical_radar_chart(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create hierarchical radar chart with:
        Tier 1: Angular variation (outer ring)
        Tier 2: Defect type weight components (middle ring)
        Tier 3: Individual sources (inner ring)
        
        FIXED: Plotly API compatibility issues
        """
        if not sources_data:
            return None
        
        # Prepare data for hierarchical radar
        fig = go.Figure()
        
        # Group by defect type
        defect_groups = {}
        for source in sources_data:
            defect = source['defect_type']
            if defect not in defect_groups:
                defect_groups[defect] = []
            defect_groups[defect].append(source)
        
        # Define categories for radar chart
        categories = ['Spatial Kernel', 'Defect Match', 'Combined Weight', 'Angular Proximity']
        
        # Add traces for each defect type
        for defect_type, sources in defect_groups.items():
            # Calculate average values for this defect type
            avg_values = []
            
            # 1. Average Spatial Kernel
            avg_spatial = np.mean([s['spatial_weight'] for s in sources])
            avg_values.append(avg_spatial)
            
            # 2. Defect Match (1 if matches target, 0 otherwise)
            defect_match = 1.0 if defect_type == target_defect else 0.0
            avg_values.append(defect_match)
            
            # 3. Average Combined Weight
            avg_combined = np.mean([s['combined_weight'] for s in sources])
            avg_values.append(avg_combined)
            
            # 4. Angular Proximity to target
            avg_angular_proximity = np.mean([
                np.exp(-0.5 * (abs(s['theta_deg'] - target_angle) / spatial_sigma) ** 2)
                for s in sources
            ])
            avg_values.append(avg_angular_proximity)
            
            # Normalize for radar display
            max_val = max(avg_values) if max(avg_values) > 0 else 1
            normalized_values = [v/max_val for v in avg_values]
            
            # Close the polygon
            values = normalized_values + [normalized_values[0]]
            theta = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=defect_type,
                line=dict(color=self.color_scheme[defect_type], width=2),
                marker=dict(size=8),
                opacity=0.7,
                hoverinfo='text',
                text=f"{defect_type}<br>Spatial: {avg_spatial:.3f}<br>Combined: {avg_combined:.3f}<br>Match: {'Yes' if defect_type == target_defect else 'No'}"
            ))
        
        # Add target reference line
        fig.add_trace(go.Scatterpolar(
            r=[1, 1, 1, 1, 1],
            theta=categories + [categories[0]],
            mode='lines',
            name='Target Reference',
            line=dict(color='black', width=1, dash='dot'),
            opacity=0.3,
            hoverinfo='none'
        ))
        
        # FIXED: Update layout without problematic parameters
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=12)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=90
                )
            ),
            title=dict(
                text=f'Hierarchical Weight Analysis<br>Target: {target_angle:.1f}° {target_defect} | σ={spatial_sigma}°',
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            width=900,
            height=700,
            margin=dict(t=100, l=50, r=50, b=50),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_weight_comparison_radar(self, sources_data, target_defect):
        """Create dual radar chart comparing weights with and without defect mask"""
        if not sources_data:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'polar'}, {'type': 'polar'}]],
            subplot_titles=('Without Defect Filter', 'With Defect Filter')
        )
        
        # Prepare data
        angles = [s['theta_deg'] for s in sources_data]
        
        # Calculate weights without defect filter
        weights_without_filter = []
        weights_with_filter = []
        
        for source in sources_data:
            # Without defect filter: just spatial and learned components
            w_without = source['spatial_weight'] * source.get('learned_attention', 1.0)
            weights_without_filter.append(w_without)
            
            # With defect filter: include defect mask
            w_with = source['combined_weight']
            weights_with_filter.append(w_with)
        
        # Normalize
        max_without = max(weights_without_filter) if weights_without_filter else 1
        max_with = max(weights_with_filter) if weights_with_filter else 1
        
        weights_without_norm = [w/max_without for w in weights_without_filter]
        weights_with_norm = [w/max_with for w in weights_with_filter]
        
        # Colors based on defect match
        colors = []
        for source in sources_data:
            if source['defect_type'] == target_defect:
                colors.append(self.color_scheme['match'])
            else:
                colors.append(self.color_scheme['mismatch'])
        
        # Plot without defect filter
        fig.add_trace(go.Scatterpolar(
            r=weights_without_norm,
            theta=angles,
            mode='markers+lines',
            name='Weight (no defect filter)',
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=2, color='white')
            ),
            line=dict(width=1, color='#4682B4', dash='dash'),
            hoverinfo='text',
            text=[f"Angle: {a}°<br>Weight: {w:.3f}<br>Defect: {s['defect_type']}"
                  for a, w, s in zip(angles, weights_without_filter, sources_data)]
        ), row=1, col=1)
        
        # Plot with defect filter
        fig.add_trace(go.Scatterpolar(
            r=weights_with_norm,
            theta=angles,
            mode='markers+lines',
            name='Weight (with defect filter)',
            marker=dict(
                size=12,
                color=colors,
                symbol='star' if target_defect == 'Twin' else 'circle',
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color='#FF8C00'),
            hoverinfo='text',
            text=[f"Angle: {a}°<br>Weight: {w:.3f}<br>Defect: {s['defect_type']}<br>Match: {s['defect_type'] == target_defect}"
                  for a, w, s in zip(angles, weights_with_filter, sources_data)]
        ), row=1, col=2)
        
        # Update polar layouts
        for col in [1, 2]:
            fig.update_polars(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=10)
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            title=dict(
                text=f'Impact of Defect Type Filtering<br>Target Defect: {target_defect}',
                font=dict(size=18, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            width=1200,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_sankey_diagram(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create Sankey diagram showing flow of attention weights
        from sources through components to target
        """
        if not sources_data:
            return None
        
        # Prepare Sankey data
        labels = []
        source_indices = []
        target_indices = []
        values = []
        colors = []
        
        # Node 0: Target
        labels.append(f"Target<br>{target_defect}<br>{target_angle:.1f}°")
        
        # Nodes 1-N: Sources
        for i, source in enumerate(sources_data):
            labels.append(f"Source {i}<br>{source['defect_type']}<br>{source['theta_deg']:.1f}°")
        
        # Nodes for components
        comp_start = len(labels)
        labels.append("Spatial Kernel")
        labels.append("Defect Match")
        labels.append("Combined Weight")
        
        # Build flows
        for i, source in enumerate(sources_data):
            source_idx = i + 1  # +1 for target node
            
            # Flow from source to spatial kernel
            source_indices.append(source_idx)
            target_indices.append(comp_start)
            values.append(source['spatial_weight'] * 100)  # Scale for visibility
            colors.append(self.color_scheme['spatial_kernel'])
            
            # Flow from source to defect match
            source_indices.append(source_idx)
            target_indices.append(comp_start + 1)
            values.append((1.0 if source['defect_type'] == target_defect else 0.1) * 100)
            colors.append(self.color_scheme['defect_mask'])
            
            # Flow from components to combined weight
            source_indices.append(comp_start)
            target_indices.append(comp_start + 2)
            values.append(source['spatial_weight'] * 50)
            colors.append(self.color_scheme['spatial_kernel'])
            
            source_indices.append(comp_start + 1)
            target_indices.append(comp_start + 2)
            values.append((1.0 if source['defect_type'] == target_defect else 0.1) * 50)
            colors.append(self.color_scheme['defect_mask'])
            
            # Flow from combined weight to target
            source_indices.append(comp_start + 2)
            target_indices.append(0)  # Target node
            values.append(source['combined_weight'] * 100)
            colors.append(self.color_scheme['combined_weight'])
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=[self.color_scheme['target']] + 
                      [self.color_scheme[s['defect_type']] for s in sources_data] +
                      [self.color_scheme['spatial_kernel'], 
                       self.color_scheme['defect_mask'], 
                       self.color_scheme['combined_weight']]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='Flow: %{value:.1f}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=f'Attention Flow Sankey Diagram<br>σ={spatial_sigma}°, Target: {target_defect} at {target_angle:.1f}°',
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            font=dict(size=12),
            width=1000,
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_chord_diagram(self, sources_data, target_angle, target_defect):
        """
        Create chord diagram showing relationships between sources based on weights
        """
        if not sources_data:
            return None
        
        # Create a matrix of relationships
        n = len(sources_data)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = sources_data[i]['combined_weight']
                else:
                    # Relationship based on angle similarity and defect match
                    angle_diff = min(
                        abs(sources_data[i]['theta_deg'] - sources_data[j]['theta_deg']),
                        360 - abs(sources_data[i]['theta_deg'] - sources_data[j]['theta_deg'])
                    )
                    angle_sim = np.exp(-angle_diff / 30.0)
                    
                    defect_match = 1.0 if sources_data[i]['defect_type'] == sources_data[j]['defect_type'] else 0.2
                    
                    matrix[i, j] = angle_sim * defect_match * 0.5
        
        # Create chord diagram using plotly
        fig = go.Figure()
        
        # Create nodes around circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 0.9
        
        node_x = radius * np.cos(angles)
        node_y = radius * np.sin(angles)
        
        # Add nodes
        for i in range(n):
            fig.add_trace(go.Scatter(
                x=[node_x[i]],
                y=[node_y[i]],
                mode='markers+text',
                marker=dict(
                    size=30 + sources_data[i]['combined_weight'] * 50,
                    color=self.color_scheme[sources_data[i]['defect_type']],
                    line=dict(width=2, color='white')
                ),
                text=[f"S{i}"],
                textposition="middle center",
                hoverinfo='text',
                hovertext=f"Source {i}<br>Angle: {sources_data[i]['theta_deg']:.1f}°<br>Defect: {sources_data[i]['defect_type']}<br>Weight: {sources_data[i]['combined_weight']:.3f}",
                showlegend=False
            ))
        
        # Add chords (edges)
        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0.1 and i != j:  # Only show significant connections
                    # Create curved line
                    t = np.linspace(0, 1, 50)
                    x = (1-t)*node_x[i] + t*node_x[j]
                    y = (1-t)*node_y[i] + t*node_y[j]
                    
                    # Add some curvature
                    mid_x = (node_x[i] + node_x[j]) / 2
                    mid_y = (node_y[i] + node_y[j]) / 2
                    dist = np.sqrt((node_x[j] - node_x[i])**2 + (node_y[j] - node_y[i])**2)
                    
                    # Apply curvature
                    if dist > 0.1:
                        control_x = mid_x + (node_y[j] - node_y[i]) * 0.3
                        control_y = mid_y - (node_x[j] - node_x[i]) * 0.3
                        
                        # Bezier curve
                        t = np.linspace(0, 1, 50)
                        x = (1-t)**2 * node_x[i] + 2*(1-t)*t*control_x + t**2 * node_x[j]
                        y = (1-t)**2 * node_y[i] + 2*(1-t)*t*control_y + t**2 * node_y[j]
                    
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        line=dict(
                            width=matrix[i, j] * 5,
                            color=f'rgba(157, 78, 221, {matrix[i, j]})'  # Purple with opacity
                        ),
                        hoverinfo='text',
                        hovertext=f"Source {i} → Source {j}<br>Strength: {matrix[i, j]:.3f}",
                        showlegend=False
                    ))
        
        # Add target node in center
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0],
            mode='markers+text',
            marker=dict(
                size=40,
                color=self.color_scheme['target'],
                symbol='star',
                line=dict(width=3, color='white')
            ),
            text=["T"],
            textposition="middle center",
            hoverinfo='text',
            hovertext=f"Target<br>Angle: {target_angle:.1f}°<br>Defect: {target_defect}",
            showlegend=False
        ))
        
        # Connect sources to target
        for i in range(n):
            weight = sources_data[i]['combined_weight']
            if weight > 0.05:  # Only show significant connections
                fig.add_trace(go.Scatter(
                    x=[node_x[i], 0],
                    y=[node_y[i], 0],
                    mode='lines',
                    line=dict(
                        width=weight * 10,
                        color=self.color_scheme[sources_data[i]['defect_type']],
                        dash='dash'
                    ),
                    hoverinfo='text',
                    hovertext=f"Source {i} → Target<br>Weight: {weight:.3f}",
                    showlegend=False
                ))
        
        fig.update_layout(
            title=dict(
                text=f'Weight Relationship Chord Diagram<br>Target: {target_angle:.1f}° {target_defect}',
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
            width=800,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )
        
        return fig
    
    def create_hierarchical_sunburst(self, hierarchical_data, target_angle, target_defect):
        """
        Create hierarchical sunburst chart showing weight distribution
        across defect types and angular bins
        """
        # Prepare data for sunburst
        labels = []
        parents = []
        values = []
        custom_data = []
        
        # Root node
        labels.append('All Sources')
        parents.append('')
        values.append(1.0)
        custom_data.append(['', 0, 0])
        
        # Add defect type level
        for defect_type, data in hierarchical_data['defect_types'].items():
            labels.append(defect_type)
            parents.append('All Sources')
            values.append(data['total_weight'])
            custom_data.append([defect_type, data['count'], 0])
            
            # Add angular bin level for each defect type
            for angle_bin, bin_data in hierarchical_data['angular_bins'].items():
                # Filter sources in this defect type and angular bin
                bin_sources = [s for s in bin_data['sources'] if s['defect_type'] == defect_type]
                if bin_sources:
                    bin_weight = sum(s['combined_weight'] for s in bin_sources)
                    if bin_weight > 0:
                        # Create combined label
                        label = f"{angle_bin}"
                        labels.append(label)
                        parents.append(defect_type)
                        values.append(bin_weight)
                        custom_data.append([defect_type, len(bin_sources), np.mean([s['theta_deg'] for s in bin_sources])])
                        
                        # Add individual sources
                        for i, source in enumerate(bin_sources[:5]):  # Limit to top 5 per bin
                            source_label = f"S{i}: {source['theta_deg']:.1f}°"
                            labels.append(source_label)
                            parents.append(label)
                            values.append(source['combined_weight'])
                            custom_data.append([defect_type, 1, source['theta_deg']])
        
        # Create sunburst
        fig = px.sunburst(
            names=labels,
            parents=parents,
            values=values,
            title=f'Hierarchical Weight Distribution<br>Target: {target_angle:.1f}° {target_defect}',
            width=900,
            height=900,
            color_continuous_scale='RdYlBu',
            color=[d[2] for d in custom_data]  # Color by average angle
        )
        
        # Update hover template
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>' +
                         'Weight: %{value:.3f}<br>' +
                         'Defect: %{customdata[0]}<br>' +
                         'Count: %{customdata[1]}<br>' +
                         'Avg Angle: %{customdata[2]:.1f}°<extra></extra>',
            customdata=custom_data
        )
        
        fig.update_layout(
            title=dict(
                font=dict(size=20, family="Arial Black", color='#1E3A8A'),
                x=0.5
            ),
            margin=dict(t=100, l=0, r=0, b=0)
        )
        
        return fig
    
    def create_weight_formula_breakdown(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create visualization showing the weight formula breakdown
        w_i(θ*) = [ᾱ_i(θ*) · exp(-(Δφ_i)²/(2σ²)) · 𝟙(τ_i = τ*)] / Σ[...] + 10⁻⁶
        """
        if not sources_data:
            return None
        
        # Sort sources by angle
        sources_data = sorted(sources_data, key=lambda x: x['theta_deg'])
        
        # Prepare data
        angles = [s['theta_deg'] for s in sources_data]
        spatial_weights = [s['spatial_weight'] for s in sources_data]
        defect_weights = [1.0 if s['defect_type'] == target_defect else 0.0 for s in sources_data]
        combined_weights = [s['combined_weight'] for s in sources_data]
        
        # Calculate the formula components for each source
        formula_components = []
        for i, source in enumerate(sources_data):
            # Spatial component: exp(-(Δφ_i)²/(2σ²))
            delta_phi = min(
                abs(source['theta_deg'] - target_angle),
                360 - abs(source['theta_deg'] - target_angle)
            )
            spatial_component = np.exp(-0.5 * (delta_phi / spatial_sigma) ** 2)
            
            # Defect component: 𝟙(τ_i = τ*)
            defect_component = 1.0 if source['defect_type'] == target_defect else 0.0
            
            # Combined (learned attention would be here in full formula)
            formula_components.append({
                'source_idx': i,
                'angle': source['theta_deg'],
                'defect': source['defect_type'],
                'spatial_component': spatial_component,
                'defect_component': defect_component,
                'combined_weight': source['combined_weight'],
                'delta_phi': delta_phi
            })
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Spatial Kernel Component',
                'Defect Match Component',
                'Combined Weight Distribution',
                'Weight Formula Breakdown',
                'Angular Distance vs Weight',
                'Cumulative Weight Distribution'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Spatial kernel component
        fig.add_trace(go.Bar(
            x=angles,
            y=[c['spatial_component'] for c in formula_components],
            name='Spatial Kernel',
            marker_color=self.color_scheme['spatial_kernel'],
            text=[f'Δφ={c["delta_phi"]:.1f}°' for c in formula_components],
            textposition='outside'
        ), row=1, col=1)
        
        # 2. Defect match component
        fig.add_trace(go.Bar(
            x=angles,
            y=[c['defect_component'] for c in formula_components],
            name='Defect Match',
            marker_color=self.color_scheme['defect_mask'],
            text=[f'{c["defect"]}' for c in formula_components],
            textposition='outside'
        ), row=1, col=2)
        
        # 3. Combined weight distribution
        fig.add_trace(go.Bar(
            x=angles,
            y=[c['combined_weight'] for c in formula_components],
            name='Combined Weight',
            marker_color=self.color_scheme['combined_weight'],
            text=[f'{c["combined_weight"]:.3f}' for c in formula_components],
            textposition='outside'
        ), row=2, col=1)
        
        # 4. Weight formula breakdown (stacked bar)
        fig.add_trace(go.Bar(
            x=angles,
            y=[c['spatial_component'] for c in formula_components],
            name='Spatial',
            marker_color=self.color_scheme['spatial_kernel']
        ), row=2, col=2)
        
        fig.add_trace(go.Bar(
            x=angles,
            y=[c['defect_component'] for c in formula_components],
            name='Defect',
            marker_color=self.color_scheme['defect_mask']
        ), row=2, col=2)
        
        # 5. Angular distance vs weight
        fig.add_trace(go.Scatter(
            x=[c['delta_phi'] for c in formula_components],
            y=[c['combined_weight'] for c in formula_components],
            mode='markers+text',
            name='Δφ vs Weight',
            marker=dict(
                size=12,
                color=[self.color_scheme[c['defect']] for c in formula_components],
                line=dict(width=2, color='white')
            ),
            text=[f"S{i}" for i in range(len(formula_components))],
            textposition="top center"
        ), row=3, col=1)
        
        # Add theoretical Gaussian curve
        delta_phi_range = np.linspace(0, 180, 100)
        gaussian_curve = np.exp(-0.5 * (delta_phi_range / spatial_sigma) ** 2)
        fig.add_trace(go.Scatter(
            x=delta_phi_range,
            y=gaussian_curve,
            mode='lines',
            name='Theoretical Gaussian',
            line=dict(color='red', width=2, dash='dash')
        ), row=3, col=1)
        
        # 6. Cumulative weight distribution
        sorted_weights = sorted([c['combined_weight'] for c in formula_components], reverse=True)
        cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative,
            mode='lines+markers',
            name='Cumulative Weight',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ), row=3, col=2)
        
        # Add 80% threshold line
        threshold_idx = next((i for i, val in enumerate(cumulative) if val >= 0.8), len(cumulative)-1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=3, col=2)
        fig.add_vline(x=threshold_idx+1, line_dash="dash", line_color="red", row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Weight Formula Breakdown: wᵢ(θ*) = [ᾱᵢ(θ*) · exp(-(Δφᵢ)²/(2σ²)) · 𝟙(τᵢ=τ*)] / Σ[...] + 10⁻⁶<br>Target: {target_angle:.1f}° {target_defect}, σ={spatial_sigma}°',
                font=dict(size=16, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            showlegend=True,
            barmode='stack',
            height=1200,
            width=1200,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Angle (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Spatial Kernel Value", row=1, col=1)
        fig.update_xaxes(title_text="Angle (degrees)", row=1, col=2)
        fig.update_yaxes(title_text="Defect Match (0 or 1)", row=1, col=2)
        fig.update_xaxes(title_text="Angle (degrees)", row=2, col=1)
        fig.update_yaxes(title_text="Combined Weight", row=2, col=1)
        fig.update_xaxes(title_text="Angle (degrees)", row=2, col=2)
        fig.update_yaxes(title_text="Component Value", row=2, col=2)
        fig.update_xaxes(title_text="Angular Distance Δφ (degrees)", row=3, col=1)
        fig.update_yaxes(title_text="Weight", row=3, col=1)
        fig.update_xaxes(title_text="Number of Top Sources", row=3, col=2)
        fig.update_yaxes(title_text="Cumulative Weight Fraction", row=3, col=2)
        
        return fig
    
    def create_aggregated_weight_dashboard(self, sources_data, target_angle, target_defect, spatial_sigma):
        """
        Create comprehensive dashboard with all weight visualizations
        """
        if not sources_data:
            return None
        
        # Prepare hierarchical data
        hierarchical_data = self.prepare_hierarchical_data(sources_data, target_defect)
        
        # Create all visualizations
        fig1 = self.create_hierarchical_radar_chart(sources_data, target_angle, target_defect, spatial_sigma)
        fig2 = self.create_weight_comparison_radar(sources_data, target_defect)
        fig3 = self.create_sankey_diagram(sources_data, target_angle, target_defect, spatial_sigma)
        fig4 = self.create_chord_diagram(sources_data, target_angle, target_defect)
        fig5 = self.create_hierarchical_sunburst(hierarchical_data, target_angle, target_defect)
        fig6 = self.create_weight_formula_breakdown(sources_data, target_angle, target_defect, spatial_sigma)
        
        return {
            'hierarchical_radar': fig1,
            'weight_comparison': fig2,
            'sankey': fig3,
            'chord': fig4,
            'sunburst': fig5,
            'formula_breakdown': fig6
        }

# =============================================
# ENHANCED TRANSFORMER INTERPOLATOR WITH WEIGHT DATA PREPARATION
# =============================================
class EnhancedTransformerInterpolator(TransformerSpatialInterpolator):
    """Enhanced interpolator with weight data preparation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_visualizer = WeightVisualizer()
    
    def prepare_weight_analysis_data(self, result):
        """Prepare comprehensive weight analysis data"""
        sources_data = []
        
        for i, field in enumerate(result['source_fields']):
            source_params = field['source_params']
            theta_rad = source_params.get('theta', 0.0)
            theta_deg = np.degrees(theta_rad) % 360
            
            # Calculate angular distance to target
            target_angle = result['target_angle']
            angular_dist = min(
                abs(theta_deg - target_angle),
                360 - abs(theta_deg - target_angle)
            )
            
            sources_data.append({
                'source_index': i,
                'theta_deg': theta_deg,
                'angular_dist': angular_dist,
                'defect_type': source_params.get('defect_type', 'Unknown'),
                'spatial_weight': result['weights']['spatial_kernel'][i],
                'learned_attention': result['weights'].get('learned_attention', [0.0] * len(result['weights']['combined']))[i],
                'defect_weight': result['weights']['defect_mask'][i],
                'combined_weight': result['weights']['combined'][i],
                'target_defect_match': source_params.get('defect_type') == result['target_params']['defect_type']
            })
        
        return sources_data

# =============================================
# MAIN APPLICATION WITH EXPANDED VISUALIZATIONS
# =============================================
def main():
    st.set_page_config(
        page_title="Weight Component Visualization Dashboard",
        layout="wide",
        page_icon="🎯",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem !important;
        color: #374151 !important;
        font-weight: 800 !important;
        border-left: 5px solid #3B82F6;
        padding-left: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .weight-formula {
        background-color: #F0F7FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .visualization-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Weight Component Visualization Dashboard</h1>', unsafe_allow_html=True)
    
    # Weight formula
    st.markdown(r"""
    <div class="weight-formula">
    <strong>Attention Weight Formula:</strong><br>
    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*)}{\sum_k \bar{\alpha}_k \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*)} + 10^{-6}$$
    
    <strong>Components:</strong>
    • $\bar{\alpha}_i$: Learned attention score<br>
    • $\exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right)$: Spatial kernel (angular bracketing)<br>
    • $\mathbb{I}(\tau_i = \tau^*)$: Defect type indicator (1 if match, 0 otherwise)<br>
    • $\sigma$: Angular kernel width (controls bracketing window)
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = EnhancedTransformerInterpolator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = WeightVisualizer()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'weight_dashboard' not in st.session_state:
        st.session_state.weight_dashboard = None
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Data loading
        if st.button("📂 Load Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"Loaded {len(st.session_state.solutions)} solutions")
                else:
                    st.warning("No solutions found")
        
        # Target parameters
        st.markdown("#### 🎯 Target Parameters")
        target_angle = st.slider("Target Angle (degrees)", 0.0, 180.0, 54.7, 0.1)
        target_defect = st.selectbox("Target Defect Type", ['Twin', 'ISF', 'ESF', 'No Defect'], index=2)
        spatial_sigma = st.slider("Spatial Kernel Width σ (degrees)", 1.0, 45.0, 10.0, 0.5)
        
        # Run interpolation
        if st.button("🚀 Generate Weight Analysis", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("Please load solutions first!")
            else:
                with st.spinner("Performing interpolation and weight analysis..."):
                    target_params = {
                        'defect_type': target_defect,
                        'eps0': PhysicsParameters.get_eigenstrain(target_defect),
                        'theta': np.radians(target_angle),
                        'shape': 'Square'
                    }
                    
                    result = st.session_state.interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        target_angle,
                        target_params
                    )
                    
                    if result:
                        st.session_state.interpolation_result = result
                        
                        # Prepare weight analysis data
                        sources_data = st.session_state.interpolator.prepare_weight_analysis_data(result)
                        
                        # Generate dashboard
                        st.session_state.weight_dashboard = st.session_state.visualizer.create_aggregated_weight_dashboard(
                            sources_data, target_angle, target_defect, spatial_sigma
                        )
                        
                        st.success("Weight analysis completed!")
    
    # Main content
    if st.session_state.weight_dashboard:
        dashboard = st.session_state.weight_dashboard
        
        # Create tabs for different visualizations
        tab_names = [
            "📊 Hierarchical Radar",
            "⚖️ Weight Comparison", 
            "🕸️ Sankey Diagram",
            "🎯 Chord Diagram",
            "🌳 Sunburst Chart",
            "📈 Formula Breakdown"
        ]
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            st.markdown('<h3 class="section-header">Hierarchical Radar Chart</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Visualization Strategy:**
            - **Tier 1 (Radial Axis):** Normalized weight components
            - **Tier 2 (Angular Axis):** Different weight components
            - **Color Coding:** Defect type specific
            - **Polygon Area:** Overall attention strength for each defect type
            """)
            st.plotly_chart(dashboard['hierarchical_radar'], use_container_width=True)
        
        with tabs[1]:
            st.markdown('<h3 class="section-header">Weight Comparison: With vs Without Defect Mask</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Comparison Analysis:**
            - **Left Radar:** Shows weights without defect type filtering
            - **Right Radar:** Shows weights with defect type filtering applied
            - **Color Coding:** Green = matching defect type, Red = mismatched defect type
            - **Key Insight:** Visualizes the impact of defect type as a hard constraint
            """)
            st.plotly_chart(dashboard['weight_comparison'], use_container_width=True)
        
        with tabs[2]:
            st.markdown('<h3 class="section-header">Sankey Diagram: Attention Flow</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Flow Visualization:**
            - **Nodes:** Sources (left), Components (middle), Target (right)
            - **Flow Width:** Proportional to contribution strength
            - **Color Coding:** Defect type and component specific
            - **Key Insight:** Shows how attention flows from sources through components to target
            """)
            st.plotly_chart(dashboard['sankey'], use_container_width=True)
        
        with tabs[3]:
            st.markdown('<h3 class="section-header">Chord Diagram: Weight Relationships</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Network Relationships:**
            - **Nodes:** Sources arranged around circle, Target at center
            - **Chords:** Weight relationships between sources
            - **Node Size:** Proportional to combined weight
            - **Key Insight:** Shows complex relationships and clustering patterns
            """)
            st.plotly_chart(dashboard['chord'], use_container_width=True)
        
        with tabs[4]:
            st.markdown('<h3 class="section-header">Sunburst Chart: Hierarchical Distribution</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Hierarchical Analysis:**
            - **Level 1:** Defect types
            - **Level 2:** Angular bins (30° intervals)
            - **Level 3:** Individual sources
            - **Segment Size:** Proportional to weight contribution
            - **Color:** Indicates angular position
            """)
            st.plotly_chart(dashboard['sunburst'], use_container_width=True)
        
        with tabs[5]:
            st.markdown('<h3 class="section-header">Weight Formula Component Breakdown</h3>', unsafe_allow_html=True)
            st.markdown("""
            **Formula Analysis:**
            - **Spatial Kernel:** Gaussian angular proximity weight
            - **Defect Match:** Binary constraint (0 or 1)
            - **Combined Weight:** Final normalized attention
            - **Angular Distance:** Shows relationship between angular separation and weight
            - **Cumulative Distribution:** Shows how many sources contribute to total weight
            """)
            st.plotly_chart(dashboard['formula_breakdown'], use_container_width=True)
        
        # Add summary statistics
        st.markdown("---")
        st.markdown('<h3 class="section-header">📊 Weight Analysis Summary</h3>', unsafe_allow_html=True)
        
        if st.session_state.interpolation_result:
            result = st.session_state.interpolation_result
            sources_data = st.session_state.interpolator.prepare_weight_analysis_data(result)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_weight = sum(s['combined_weight'] for s in sources_data)
                st.metric("Total Weight", f"{total_weight:.3f}")
            
            with col2:
                matching_sources = [s for s in sources_data if s['target_defect_match']]
                mismatch_sources = [s for s in sources_data if not s['target_defect_match']]
                st.metric("Matching Sources", f"{len(matching_sources)}/{len(sources_data)}")
            
            with col3:
                if matching_sources:
                    avg_matching_weight = np.mean([s['combined_weight'] for s in matching_sources])
                    st.metric("Avg Match Weight", f"{avg_matching_weight:.3f}")
                else:
                    st.metric("Avg Match Weight", "N/A")
            
            with col4:
                entropy = result['weights']['entropy']
                st.metric("Attention Entropy", f"{entropy:.3f}")
            
            # Weight distribution table
            st.markdown("#### 📋 Detailed Weight Distribution")
            weight_df = pd.DataFrame(sources_data)
            st.dataframe(
                weight_df[['source_index', 'theta_deg', 'defect_type', 'spatial_weight', 
                          'defect_weight', 'combined_weight', 'target_defect_match']].style
                .background_gradient(subset=['combined_weight'], cmap='Blues')
                .format({
                    'spatial_weight': '{:.4f}',
                    'defect_weight': '{:.4f}',
                    'combined_weight': '{:.4f}'
                })
            )
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Weight Component Visualization Dashboard
        
        This tool provides comprehensive visualization of attention weights in Angular Bracketing Theory.
        
        ### Key Visualizations:
        1. **Hierarchical Radar Chart**: Multi-tier weight analysis with defect type grouping
        2. **Weight Comparison Radar**: Shows impact of defect type filtering
        3. **Sankey Diagram**: Flow visualization of attention from sources to target
        4. **Chord Diagram**: Network relationships between sources based on weights
        5. **Sunburst Chart**: Hierarchical distribution across defect types and angles
        6. **Formula Breakdown**: Detailed analysis of each weight formula component
        
        ### Getting Started:
        1. Click **"Load Solutions"** in the sidebar
        2. Configure target parameters (angle, defect type)
        3. Adjust spatial kernel width (σ) to control angular bracketing
        4. Click **"Generate Weight Analysis"** to create visualizations
        
        ### Physics Interpretation:
        All visualizations highlight:
        - **Hierarchical attributes**: Defect type → Angular position → Individual sources
        - **Dynamic weight computation**: Shows how weights change with query/target
        - **Component breakdown**: Separates spatial, defect, and learned components
        - **Domain awareness**: 12.8 nm × 12.8 nm domain context
        """)

if __name__ == "__main__":
    main()
