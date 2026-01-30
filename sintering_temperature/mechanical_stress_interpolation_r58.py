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

# =============================================
# ENHANCED TRANSFORMER SPATIAL INTERPOLATOR
# =============================================
class TransformerSpatialInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                spatial_sigma=10.0, temperature=1.0, locality_weight_factor=0.5):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(15, d_model)
        
    def compute_weight_components(self, result):
        """
        Compute the mathematical components of the weight formula:
        w_i(θ*) = [ᾱ_i(θ*) · exp(-(Δφ_i)²/(2σ²)) · 𝕀(τ_i = τ*)] / 
                  [∑_{k=1}^{N} ᾱ_k(θ*) · exp(-(Δφ_k)²/(2σ²)) · 𝕀(τ_k = τ*)] + 10^{-6}
        
        Returns:
            Dictionary with all weight components for each source
        """
        weights = result['weights']
        source_angles = result['source_theta_degrees']
        target_angle = result['target_angle']
        target_defect = result['target_params']['defect_type']
        
        # Extract source defect types
        source_defects = [field['source_params']['defect_type'] 
                         for field in result['source_fields']]
        
        N = len(source_angles)
        components = []
        
        for i in range(N):
            # Compute angular difference
            delta_phi = abs(source_angles[i] - target_angle)
            delta_phi = min(delta_phi, 360 - delta_phi)
            
            # Extract components
            alpha_bar_i = weights['pre_mask'][i]  # Pre-mask attention
            angular_weight = np.exp(-(delta_phi**2) / (2 * self.spatial_sigma**2))
            defect_indicator = 1.0 if source_defects[i] == target_defect else 0.0
            
            # Compute numerator
            numerator = alpha_bar_i * angular_weight * defect_indicator
            
            # Compute denominator (sum over all sources)
            denominator = 0.0
            for k in range(N):
                delta_phi_k = abs(source_angles[k] - target_angle)
                delta_phi_k = min(delta_phi_k, 360 - delta_phi_k)
                angular_weight_k = np.exp(-(delta_phi_k**2) / (2 * self.spatial_sigma**2))
                defect_indicator_k = 1.0 if source_defects[k] == target_defect else 0.0
                denominator += weights['pre_mask'][k] * angular_weight_k * defect_indicator_k
            
            # Compute normalized weight without offset
            normalized_weight = numerator / (denominator + 1e-10)
            
            # Final weight with offset
            final_weight = normalized_weight + 1e-6
            
            components.append({
                'source_index': i,
                'theta_deg': source_angles[i],
                'delta_phi': delta_phi,
                'defect_type': source_defects[i],
                'defect_match': source_defects[i] == target_defect,
                'alpha_bar': alpha_bar_i,
                'angular_weight': angular_weight,
                'defect_indicator': defect_indicator,
                'numerator': numerator,
                'denominator': denominator,
                'normalized_weight': normalized_weight,
                'final_weight': final_weight,
                'spatial_sigma': self.spatial_sigma
            })
        
        return components
    
    def prepare_component_data(self, result, query_index=None):
        """
        Prepare data for component-based visualizations
        """
        components = self.compute_weight_components(result)
        
        # Create DataFrame for easy manipulation
        df_data = []
        for comp in components:
            df_data.append({
                'source_index': comp['source_index'],
                'theta_deg': comp['theta_deg'],
                'delta_phi': comp['delta_phi'],
                'defect_type': comp['defect_type'],
                'defect_match': comp['defect_match'],
                'alpha_bar': comp['alpha_bar'],
                'angular_weight': comp['angular_weight'],
                'defect_indicator': comp['defect_indicator'],
                'numerator': comp['numerator'],
                'denominator': comp['denominator'] if comp['source_index'] == 0 else None,
                'normalized_weight': comp['normalized_weight'],
                'final_weight': comp['final_weight'],
                'is_query': comp['source_index'] == query_index if query_index is not None else False,
                'spatial_sigma': comp['spatial_sigma']
            })
        
        df = pd.DataFrame(df_data)
        
        # Add habit plane distance
        df['habit_distance'] = abs(df['theta_deg'] - 54.7)
        df['habit_distance'] = df['habit_distance'].apply(lambda x: min(x, 360 - x))
        
        return df

# =============================================
# ENHANCED HEAT MAP VISUALIZER WITH MATHEMATICAL VISUALIZATIONS
# =============================================
class HeatMapVisualizer:
    def __init__(self):
        self.colormaps = {
            'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'Diverging': ['RdBu', 'RdYlBu', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
            'Qualitative': ['Set1', 'Set2', 'Set3', 'tab10', 'tab20'],
            'Publication': ['viridis', 'plasma', 'RdBu', 'Spectral', 'coolwarm']
        }
    
    def create_mathematical_radar(self, component_df, query_index=None, title="Mathematical Weight Components Radar"):
        """
        Create radar chart showing all components of the weight formula for each source
        w_i(θ*) = [ᾱ_i(θ*) · exp(-(Δφ_i)²/(2σ²)) · 𝕀(τ_i = τ*)] / 
                  [∑_{k=1}^{N} ᾱ_k(θ*) · exp(-(Δφ_k)²/(2σ²)) · 𝕀(τ_k = τ*)] + 10^{-6}
        """
        # Define the components to show
        components = ['alpha_bar', 'angular_weight', 'defect_indicator', 
                     'numerator', 'normalized_weight', 'final_weight']
        component_names = ['ᾱ (Pre-mask)', 'exp(-Δφ²/2σ²)', '𝕀(τ=τ*)', 
                          'Numerator', 'Normalized', 'Final + 10⁻⁶']
        
        # Create figure
        fig = go.Figure()
        
        # Normalize each component for better visualization
        normalized_data = {}
        for comp in components:
            values = component_df[comp].values
            if comp == 'defect_indicator':
                normalized_data[comp] = values  # Binary, keep as is
            else:
                min_val = values.min()
                max_val = values.max()
                if max_val > min_val:
                    normalized_data[comp] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[comp] = np.ones_like(values) * 0.5
        
        # Add traces for each source
        colors = px.colors.qualitative.Set3
        for idx, row in component_df.iterrows():
            values = [normalized_data[comp][idx] for comp in components] + [normalized_data[components[0]][idx]]
            theta = component_names + [component_names[0]]
            
            # Highlight query source
            is_query = row['is_query']
            color = '#FF1493' if is_query else colors[idx % len(colors)]
            line_width = 3 if is_query else 1.5
            opacity = 1.0 if is_query else 0.7
            name = f"Source {idx} (θ={row['theta_deg']:.1f}°)" + (" [QUERY]" if is_query else "")
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill='toself',
                name=name,
                line=dict(color=color, width=line_width),
                opacity=opacity,
                hoverinfo='text',
                text=f"""
                Source {idx}<br>
                θ = {row['theta_deg']:.1f}°<br>
                ᾱ = {row['alpha_bar']:.4f}<br>
                exp(-Δφ²/2σ²) = {row['angular_weight']:.4f}<br>
                𝕀(τ=τ*) = {row['defect_indicator']}<br>
                Numerator = {row['numerator']:.6f}<br>
                Normalized = {row['normalized_weight']:.6f}<br>
                Final = {row['final_weight']:.6f}
                """
            ))
        
        # Add mathematical formula as annotation
        formula_text = r"$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*) }{ \sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*) } + 10^{-6}$"
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.1],
                    tickfont=dict(size=11),
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    direction='clockwise',
                    rotation=90
                ),
                bgcolor='rgba(245, 245, 245, 0.8)'
            ),
            title=dict(
                text=f"{title}<br><span style='font-size: 14px; color: gray;'>{formula_text}</span>",
                font=dict(size=20, family="Arial", color='#1E3A8A'),
                x=0.5,
                y=0.98
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                font=dict(size=10)
            ),
            width=1000,
            height=800,
            margin=dict(l=100, r=200, t=120, b=50)
        )
        
        # Add summary statistics
        fig.add_annotation(
            text=f"N = {len(component_df)} sources | σ = {component_df['spatial_sigma'].iloc[0]:.1f}°",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=12, color="darkblue"),
            bgcolor="lightyellow",
            bordercolor="darkblue",
            borderwidth=1
        )
        
        return fig
    
    def create_mathematical_sunburst(self, component_df, query_index=None, title="Mathematical Decomposition Sunburst"):
        """
        Create hierarchical sunburst showing the mathematical decomposition of weights
        """
        # Build hierarchical data
        labels = ["Weight Formula"]
        parents = [""]
        values = [1.0]
        colors = []
        
        # Level 1: Components of the formula
        formula_parts = ["Numerator ∑", "Denominator", "Offset"]
        for part in formula_parts:
            labels.append(part)
            parents.append("Weight Formula")
            values.append(0.33)  # Equal distribution for visualization
            colors.append("#4ECDC4")
        
        # Level 2: For numerator, break down into sources
        for idx, row in component_df.iterrows():
            source_label = f"Source {idx}<br>θ={row['theta_deg']:.1f}°"
            labels.append(source_label)
            parents.append("Numerator ∑")
            values.append(row['numerator'])
            
            # Color by defect type
            if row['defect_match']:
                colors.append("#FF6B6B")  # Red for matching defect
            else:
                colors.append("#95E1D3")  # Teal for non-matching
        
        # Level 3: For each source, show components
        for idx, row in component_df.iterrows():
            source_label = f"Source {idx}<br>θ={row['theta_deg']:.1f}°"
            
            # Components
            components = [
                ("ᾱ", row['alpha_bar']),
                ("exp(-Δφ²/2σ²)", row['angular_weight']),
                ("𝕀(τ=τ*)", row['defect_indicator'])
            ]
            
            for comp_name, comp_value in components:
                comp_label = f"{comp_name} = {comp_value:.4f}"
                labels.append(comp_label)
                parents.append(source_label)
                values.append(comp_value)
                colors.append("#FFD166")  # Yellow for components
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors + ["white"] * (len(labels) - len(colors)),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.6f}<br>Parent: %{parent}<extra></extra>',
            maxdepth=3,
            insidetextorientation='radial'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=22, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            width=900,
            height=900,
            margin=dict(t=80, l=20, r=20, b=20)
        )
        
        # Add formula annotation
        formula_text = r"$w_i = \frac{\bar{\alpha}_i \cdot e^{-\Delta\phi_i^2/2\sigma^2} \cdot \mathbb{I}(\tau_i=\tau^*)}{\sum_k \bar{\alpha}_k \cdot e^{-\Delta\phi_k^2/2\sigma^2} \cdot \mathbb{I}(\tau_k=\tau^*)} + 10^{-6}$"
        fig.add_annotation(
            text=formula_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=14, color="darkblue"),
            bgcolor="white",
            bordercolor="darkblue",
            borderwidth=1
        )
        
        return fig
    
    def create_weight_sankey(self, component_df, query_index=None, title="Attention Weight Flow Sankey Diagram"):
        """
        Create Sankey diagram showing the flow of attention weights through the mathematical pipeline
        """
        # Define nodes
        nodes = []
        node_labels = []
        
        # Level 0: Sources
        for idx, row in component_df.iterrows():
            node_labels.append(f"Source {idx}<br>θ={row['theta_deg']:.1f}°<br>τ={row['defect_type']}")
            nodes.append({
                "color": "#FF6B6B" if row['defect_match'] else "#4ECDC4",
                "pad": 15,
                "thickness": 20
            })
        
        # Level 1: Mathematical operations
        operations = ["ᾱ → Pre-mask", "exp(-Δφ²/2σ²)", "𝕀(τ=τ*)", "Multiply", "Normalize", "Add 10⁻⁶"]
        for op in operations:
            node_labels.append(op)
            nodes.append({
                "color": "#118AB2",
                "pad": 10,
                "thickness": 15
            })
        
        # Level 2: Final weights
        node_labels.append("Final Weights")
        nodes.append({
            "color": "#06D6A0",
            "pad": 20,
            "thickness": 25
        })
        
        # Create links
        links = {
            'source': [],
            'target': [],
            'value': [],
            'color': []
        }
        
        num_sources = len(component_df)
        num_ops = len(operations)
        
        # Connect sources to first operation (alpha bar)
        for i in range(num_sources):
            links['source'].append(i)
            links['target'].append(num_sources)  # First operation
            links['value'].append(component_df.iloc[i]['alpha_bar'])
            links['color'].append("rgba(255, 107, 107, 0.6)" if component_df.iloc[i]['defect_match'] else "rgba(78, 205, 196, 0.6)")
        
        # Connect through the mathematical pipeline
        # Alpha bar to angular weight multiplication
        for i in range(num_sources):
            links['source'].append(num_sources)  # Alpha bar
            links['target'].append(num_sources + 1)  # Angular weight
            links['value'].append(component_df.iloc[i]['alpha_bar'] * component_df.iloc[i]['angular_weight'])
            links['color'].append("rgba(17, 138, 178, 0.6)")
        
        # Angular weight to defect indicator
        for i in range(num_sources):
            links['source'].append(num_sources + 1)  # Angular weight
            links['target'].append(num_sources + 2)  # Defect indicator
            links['value'].append(component_df.iloc[i]['alpha_bar'] * component_df.iloc[i]['angular_weight'] * component_df.iloc[i]['defect_indicator'])
            links['color'].append("rgba(17, 138, 178, 0.6)")
        
        # Defect indicator to multiplication
        for i in range(num_sources):
            links['source'].append(num_sources + 2)  # Defect indicator
            links['target'].append(num_sources + 3)  # Multiply
            links['value'].append(component_df.iloc[i]['numerator'])
            links['color'].append("rgba(17, 138, 178, 0.6)")
        
        # Multiplication to normalization
        total_numerator = component_df['numerator'].sum()
        links['source'].append(num_sources + 3)  # Multiply
        links['target'].append(num_sources + 4)  # Normalize
        links['value'].append(total_numerator)
        links['color'].append("rgba(17, 138, 178, 0.8)")
        
        # Normalization to offset addition
        for i in range(num_sources):
            links['source'].append(num_sources + 4)  # Normalize
            links['target'].append(num_sources + 5)  # Add offset
            links['value'].append(component_df.iloc[i]['normalized_weight'])
            links['color'].append("rgba(6, 214, 160, 0.6)")
        
        # Offset addition to final weights
        for i in range(num_sources):
            links['source'].append(num_sources + 5)  # Add offset
            links['target'].append(num_sources + num_ops)  # Final weights
            links['value'].append(component_df.iloc[i]['final_weight'])
            links['color'].append("rgba(6, 214, 160, 0.8)" if component_df.iloc[i]['defect_match'] else "rgba(149, 225, 211, 0.8)")
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=[node["color"] for node in nodes]
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                color=links['color'],
                hovertemplate='Flow: %{value:.6f}<extra></extra>'
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            font=dict(size=12),
            width=1200,
            height=800,
            margin=dict(t=100, l=50, r=50, b=50)
        )
        
        # Add formula and statistics
        formula_text = r"Mathematical Pipeline: ᾱ → exp(-Δφ²/2σ²) → 𝕀(τ=τ*) → Multiply → Normalize → +10⁻⁶"
        fig.add_annotation(
            text=formula_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=14, color="darkblue"),
            bgcolor="white",
            bordercolor="darkblue",
            borderwidth=1
        )
        
        # Add summary
        total_final = component_df['final_weight'].sum()
        matching_sources = component_df[component_df['defect_match']]
        non_matching_sources = component_df[~component_df['defect_match']]
        
        summary_text = f"""
        Total Weight: {total_final:.4f} | 
        Matching Sources: {len(matching_sources)} | 
        Non-matching: {len(non_matching_sources)} | 
        Max Weight: {component_df['final_weight'].max():.4f} | 
        Min Weight: {component_df['final_weight'].min():.4f}
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=12, color="darkgreen"),
            bgcolor="lightyellow",
            bordercolor="darkgreen",
            borderwidth=1
        )
        
        return fig
    
    def create_component_comparison_matrix(self, component_df, query_index=None, title="Component Comparison Matrix"):
        """
        Create a heatmap matrix comparing all components across all sources
        """
        # Select components for comparison
        components = ['alpha_bar', 'angular_weight', 'defect_indicator', 
                     'numerator', 'normalized_weight', 'final_weight']
        component_labels = ['ᾱ', 'exp(-Δφ²/2σ²)', '𝕀(τ=τ*)', 'Numerator', 'Normalized', 'Final']
        
        # Prepare data matrix
        data = []
        for comp in components:
            data.append(component_df[comp].values)
        data = np.array(data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=[f"Source {i}<br>θ={component_df.iloc[i]['theta_deg']:.1f}°" for i in range(len(component_df))],
            y=component_labels,
            colorscale='RdBu',
            zmid=0.5,
            colorbar=dict(title="Value"),
            hovertemplate='<b>%{y}</b><br>Source: %{x}<br>Value: %{z:.6f}<extra></extra>',
            text=np.round(data, 4),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br>Color: Red = Higher, Blue = Lower",
                font=dict(size=20, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                tickfont=dict(size=12)
            ),
            width=1000,
            height=600,
            margin=dict(t=100, l=100, r=50, b=100)
        )
        
        # Highlight query column if exists
        if query_index is not None:
            fig.add_vline(
                x=query_index - 0.5,
                line_width=3,
                line_dash="dash",
                line_color="#FF1493",
                annotation_text="Query Source",
                annotation_position="top"
            )
        
        # Add statistics
        stats_text = f"""
        Matrix Statistics:<br>
        • Mean ᾱ: {component_df['alpha_bar'].mean():.4f}<br>
        • Mean Angular Weight: {component_df['angular_weight'].mean():.4f}<br>
        • Matching Defects: {component_df['defect_indicator'].sum():.0f}/{len(component_df)}<br>
        • Weight Sum: {component_df['final_weight'].sum():.4f}
        """
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=1.05, y=0.5,
            showarrow=False,
            font=dict(size=11, color="darkblue"),
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="darkblue",
            borderwidth=1,
            align="left"
        )
        
        return fig
    
    def create_component_parallel_coordinates(self, component_df, query_index=None, title="Component Parallel Coordinates"):
        """
        Create parallel coordinates plot for comparing all components
        """
        # Select and normalize components
        components = ['alpha_bar', 'angular_weight', 'defect_indicator', 
                     'numerator', 'normalized_weight', 'final_weight']
        
        # Normalize data for parallel coordinates
        normalized_df = component_df.copy()
        for comp in components:
            if comp != 'defect_indicator':  # Keep binary as is
                min_val = normalized_df[comp].min()
                max_val = normalized_df[comp].max()
                if max_val > min_val:
                    normalized_df[comp] = (normalized_df[comp] - min_val) / (max_val - min_val)
        
        # Create parallel coordinates
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=normalized_df['final_weight'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Final Weight')
                ),
                dimensions=[dict(
                    range=[0, 1],
                    label=comp,
                    values=normalized_df[comp]
                ) for comp in components],
                hoverinfo='text',
                hovertext=[f"""
                Source {idx}<br>
                θ = {row['theta_deg']:.1f}°<br>
                ᾱ = {row['alpha_bar']:.4f}<br>
                exp(-Δφ²/2σ²) = {row['angular_weight']:.4f}<br>
                𝕀(τ=τ*) = {row['defect_indicator']}<br>
                Final Weight = {row['final_weight']:.6f}
                """ for idx, row in component_df.iterrows()]
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br>Each line represents one source, colored by final weight",
                font=dict(size=20, family="Arial", color='#1E3A8A'),
                x=0.5
            ),
            width=1200,
            height=600,
            margin=dict(t=100, l=50, r=50, b=50)
        )
        
        # Highlight query source if exists
        if query_index is not None:
            # Add annotation
            query_row = component_df.iloc[query_index]
            fig.add_annotation(
                text=f"QUERY SOURCE {query_index}<br>θ={query_row['theta_deg']:.1f}°<br>Final Weight={query_row['final_weight']:.6f}",
                xref="paper", yref="paper",
                x=0.98, y=0.95,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#FF1493",
                font=dict(size=12, color="#FF1493", weight="bold"),
                bgcolor="white",
                bordercolor="#FF1493",
                borderwidth=2
            )
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Mathematical Weight Visualization", layout="wide", page_icon="📊")
    
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
    }
    .section-header { 
        font-size: 2.2rem !important; 
        color: #374151 !important; 
        font-weight: 800 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .math-formula {
        background-color: #F7F9FC;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        font-family: "Cambria Math", serif;
        font-size: 1.4rem;
        text-align: center;
        color: #1E3A8A;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1.2rem 0;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2.5rem; }
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 Mathematical Weight Component Visualizer</h1>', unsafe_allow_html=True)
    
    # Display the mathematical formula
    st.markdown("""
    <div class="math-formula">
        <b>Weight Formula:</b><br>
        $$w_i(\boldsymbol{\theta}^*) = \\frac{\\bar{\\alpha}_i(\\boldsymbol{\\theta}^*) \\cdot \\exp\\left(-\\frac{(\\Delta\\phi_i)^2}{2\\sigma^2}\\right) \\cdot \\mathbb{I}(\\tau_i = \\tau^*) }{ \\sum_{k=1}^{N} \\bar{\\alpha}_k(\\boldsymbol{\\theta}^*) \\cdot \\exp\\left(-\\frac{(\\Delta\\phi_k)^2}{2\\sigma^2}\\right) \\cdot \\mathbb{I}(\\tau_k = \\tau^*) } + 10^{-6}$$
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'component_data' not in st.session_state:
        st.session_state.component_data = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HeatMapVisualizer()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = TransformerSpatialInterpolator()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Simulation parameters
        st.markdown("#### 🎯 Simulation Parameters")
        num_sources = st.slider("Number of Sources", 3, 20, 8)
        spatial_sigma = st.slider("Spatial Sigma (σ)", 5.0, 30.0, 10.0, 0.5)
        
        # Generate synthetic data
        if st.button("🎲 Generate Synthetic Data", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic component data..."):
                # Create synthetic component data
                np.random.seed(42)
                
                # Generate angles
                angles = np.sort(np.random.uniform(0, 180, num_sources))
                target_angle = np.random.uniform(0, 180)
                
                # Generate defect types
                defect_types = ['Twin', 'ISF', 'ESF', 'No Defect']
                source_defects = np.random.choice(defect_types, num_sources, p=[0.4, 0.3, 0.2, 0.1])
                target_defect = np.random.choice(defect_types)
                
                # Generate component values
                component_df = pd.DataFrame({
                    'source_index': range(num_sources),
                    'theta_deg': angles,
                    'delta_phi': np.minimum(np.abs(angles - target_angle), 360 - np.abs(angles - target_angle)),
                    'defect_type': source_defects,
                    'defect_match': source_defects == target_defect,
                    'alpha_bar': np.random.beta(2, 5, num_sources),
                    'angular_weight': np.exp(-(np.minimum(np.abs(angles - target_angle), 360 - np.abs(angles - target_angle))**2) / (2 * spatial_sigma**2)),
                    'defect_indicator': (source_defects == target_defect).astype(float),
                    'spatial_sigma': spatial_sigma
                })
                
                # Compute derived components
                component_df['numerator'] = component_df['alpha_bar'] * component_df['angular_weight'] * component_df['defect_indicator']
                denominator = component_df['numerator'].sum()
                component_df['normalized_weight'] = component_df['numerator'] / (denominator + 1e-10)
                component_df['final_weight'] = component_df['normalized_weight'] + 1e-6
                
                # Randomly select a query source
                query_idx = np.random.randint(0, num_sources)
                component_df['is_query'] = component_df['source_index'] == query_idx
                
                st.session_state.component_data = component_df
                st.session_state.target_angle = target_angle
                st.session_state.target_defect = target_defect
                st.session_state.query_index = query_idx
                
                st.success(f"Generated {num_sources} sources with target θ={target_angle:.1f}°, τ={target_defect}")
        
        st.divider()
        
        # Visualization options
        st.markdown("#### 📊 Visualization Options")
        query_source = st.selectbox(
            "Select Query Source",
            options=list(range(num_sources)) if st.session_state.component_data is not None else [0],
            index=st.session_state.query_index if hasattr(st.session_state, 'query_index') else 0
        )
        
        if st.session_state.component_data is not None:
            st.session_state.component_data['is_query'] = st.session_state.component_data['source_index'] == query_source
    
    # Main content area
    if st.session_state.component_data is not None:
        component_df = st.session_state.component_data
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📐 Mathematical Radar", 
            "🌅 Formula Sunburst", 
            "🌊 Weight Sankey", 
            "📊 Component Matrix",
            "📈 Parallel Coordinates"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📐 Mathematical Component Radar Chart</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Radar Chart Interpretation:</strong><br>
            Each polygon represents one source. The six axes show the components of the weight formula:<br>
            • <strong>ᾱ (Pre-mask):</strong> Raw attention score before spatial and defect masking<br>
            • <strong>exp(-Δφ²/2σ²):</strong> Angular proximity weight (Gaussian kernel)<br>
            • <strong>𝕀(τ=τ*):</strong> Defect type indicator (1 for match, 0 for mismatch)<br>
            • <strong>Numerator:</strong> Product of the three components above<br>
            • <strong>Normalized:</strong> Numerator divided by sum of all numerators<br>
            • <strong>Final + 10⁻⁶:</strong> Normalized weight with small offset
            </div>
            """, unsafe_allow_html=True)
            
            fig1 = st.session_state.visualizer.create_mathematical_radar(
                component_df, 
                query_index=query_source,
                title="Weight Formula Component Analysis"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display component values table
            st.markdown("#### 📋 Component Values Table")
            display_df = component_df.copy()
            display_df = display_df[[
                'source_index', 'theta_deg', 'defect_type', 'defect_match',
                'alpha_bar', 'angular_weight', 'defect_indicator',
                'numerator', 'normalized_weight', 'final_weight', 'is_query'
            ]]
            display_df.columns = [
                'Source', 'θ (°)', 'Defect', 'Match?',
                'ᾱ', 'exp(-Δφ²/2σ²)', '𝕀(τ=τ*)',
                'Numerator', 'Normalized', 'Final', 'Query?'
            ]
            st.dataframe(
                display_df.style.format({
                    'θ (°)': '{:.1f}',
                    'ᾱ': '{:.4f}',
                    'exp(-Δφ²/2σ²)': '{:.4f}',
                    '𝕀(τ=τ*)': '{:.0f}',
                    'Numerator': '{:.6f}',
                    'Normalized': '{:.6f}',
                    'Final': '{:.6f}'
                }).apply(
                    lambda x: ['background-color: #FFE5E5' if v else '' for v in x == 'Query?'],
                    axis=0
                ),
                use_container_width=True
            )
        
        with tab2:
            st.markdown('<h2 class="section-header">🌅 Mathematical Formula Sunburst</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Sunburst Interpretation:</strong><br>
            Hierarchical decomposition of the weight formula:<br>
            • <strong>Center:</strong> Complete weight formula<br>
            • <strong>Middle Ring:</strong> Mathematical components (Numerator, Denominator, Offset)<br>
            • <strong>Outer Rings:</strong> Individual sources and their component values<br>
            • <strong>Area:</strong> Proportional to component value<br>
            • <strong>Color:</strong> Red for matching defects, Teal for non-matching, Yellow for components
            </div>
            """, unsafe_allow_html=True)
            
            fig2 = st.session_state.visualizer.create_mathematical_sunburst(
                component_df,
                query_index=query_source,
                title="Hierarchical Formula Decomposition"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_weight = component_df['final_weight'].sum()
                st.metric("Total Weight", f"{total_weight:.4f}")
            with col2:
                matching_count = component_df['defect_match'].sum()
                st.metric("Matching Sources", f"{matching_count}/{len(component_df)}")
            with col3:
                max_weight = component_df['final_weight'].max()
                st.metric("Max Weight", f"{max_weight:.4f}")
        
        with tab3:
            st.markdown('<h2 class="section-header">🌊 Attention Weight Flow Sankey Diagram</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Sankey Diagram Interpretation:</strong><br>
            Flow of attention through the mathematical pipeline:<br>
            • <strong>Left:</strong> Source nodes with angle and defect type<br>
            • <strong>Middle:</strong> Mathematical operations (ᾱ → Angular Weight → Defect Indicator → Multiply → Normalize → Offset)<br>
            • <strong>Right:</strong> Final weight distribution<br>
            • <strong>Flow Width:</strong> Proportional to value at each stage<br>
            • <strong>Color:</strong> Red for matching defects, Blue for operations, Green for final weights
            </div>
            """, unsafe_allow_html=True)
            
            fig3 = st.session_state.visualizer.create_weight_sankey(
                component_df,
                query_index=query_source,
                title="Mathematical Pipeline Flow"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Pipeline statistics
            st.markdown("#### 🎯 Pipeline Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Normalization:**")
                st.write(f"- Total Numerator: {component_df['numerator'].sum():.6f}")
                st.write(f"- Max Numerator: {component_df['numerator'].max():.6f}")
                st.write(f"- Min Numerator: {component_df['numerator'].min():.6f}")
            
            with col2:
                st.write("**After Normalization:**")
                st.write(f"- Total Weight: {component_df['final_weight'].sum():.6f}")
                st.write(f"- Weight Range: [{component_df['final_weight'].min():.6f}, {component_df['final_weight'].max():.6f}]")
                st.write(f"- Mean Weight: {component_df['final_weight'].mean():.6f}")
        
        with tab4:
            st.markdown('<h2 class="section-header">📊 Component Comparison Matrix</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Matrix Interpretation:</strong><br>
            Heatmap comparing all components across all sources:<br>
            • <strong>Rows:</strong> Mathematical components<br>
            • <strong>Columns:</strong> Sources with their angles<br>
            • <strong>Color:</strong> Red = higher values, Blue = lower values<br>
            • <strong>Query Column:</strong> Highlighted with pink dashed line<br>
            • <strong>Values:</strong> Displayed in each cell
            </div>
            """, unsafe_allow_html=True)
            
            fig4 = st.session_state.visualizer.create_component_comparison_matrix(
                component_df,
                query_index=query_source,
                title="Component Value Comparison"
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Correlation analysis
            st.markdown("#### 🔗 Component Correlations")
            
            # Compute correlations between components
            correlation_df = component_df[[
                'alpha_bar', 'angular_weight', 'defect_indicator', 
                'numerator', 'normalized_weight', 'final_weight'
            ]].corr()
            
            fig_corr = px.imshow(
                correlation_df,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                title="Correlation Matrix Between Components"
            )
            fig_corr.update_layout(width=800, height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab5:
            st.markdown('<h2 class="section-header">📈 Component Parallel Coordinates</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Parallel Coordinates Interpretation:</strong><br>
            Each colored line represents one source through all component dimensions:<br>
            • <strong>Vertical Axes:</strong> Mathematical components (normalized 0-1)<br>
            • <strong>Lines:</strong> Individual sources<br>
            • <strong>Color:</strong> Based on final weight (yellow=low, purple=high)<br>
            • <strong>Patterns:</strong> Clustered lines indicate similar component profiles<br>
            • <strong>Query Source:</strong> Annotated with pink callout
            </div>
            """, unsafe_allow_html=True)
            
            fig5 = st.session_state.visualizer.create_component_parallel_coordinates(
                component_df,
                query_index=query_source,
                title="Multi-dimensional Component Analysis"
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            # Cluster analysis
            st.markdown("#### 🎯 Source Clustering by Component Profile")
            
            # Simple k-means clustering based on components
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data for clustering
            cluster_data = component_df[[
                'alpha_bar', 'angular_weight', 'defect_indicator', 
                'normalized_weight'
            ]].values
            
            # Standardize
            scaler = StandardScaler()
            cluster_scaled = scaler.fit_transform(cluster_data)
            
            # Apply k-means
            kmeans = KMeans(n_clusters=min(3, len(component_df)), random_state=42)
            component_df['cluster'] = kmeans.fit_predict(cluster_scaled)
            
            # Display cluster results
            cluster_summary = component_df.groupby('cluster').agg({
                'source_index': 'count',
                'final_weight': 'mean',
                'defect_match': 'mean',
                'theta_deg': ['min', 'max', 'mean']
            }).round(3)
            
            st.write("**Cluster Analysis Results:**")
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Visualize clusters
            fig_cluster = px.scatter(
                component_df,
                x='theta_deg',
                y='final_weight',
                color='cluster',
                size='alpha_bar',
                hover_data=['defect_type', 'defect_match', 'angular_weight'],
                title="Source Clustering by Angle and Final Weight",
                labels={'theta_deg': 'Angle (°)', 'final_weight': 'Final Weight'}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Export functionality
        st.markdown("---")
        st.markdown("#### 💾 Export Data")
        
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("📥 Export Component Data as CSV", use_container_width=True):
                csv = component_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"weight_components_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            if st.button("📊 Export All Visualizations as HTML", use_container_width=True):
                # Create combined HTML report
                import plotly.io as pio
                
                html_content = f"""
                <html>
                <head>
                    <title>Weight Component Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .header {{ text-align: center; margin-bottom: 40px; }}
                        .formula {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; }}
                        .chart {{ margin: 30px 0; }}
                        .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Weight Component Analysis Report</h1>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="formula">
                        <h3>Weight Formula</h3>
                        <p style="font-size: 18px;">
                            $$w_i(\\boldsymbol{{\\theta}}^*) = \\frac{{\\bar{{\\alpha}}_i(\\boldsymbol{{\\theta}}^*) \\cdot \\exp\\left(-\\frac{{(\\Delta\\phi_i)^2}}{{2\\sigma^2}}\\right) \\cdot \\mathbb{{I}}(\\tau_i = \\tau^*) }}{{ \\sum_{{k=1}}^{{N}} \\bar{{\\alpha}}_k(\\boldsymbol{{\\theta}}^*) \\cdot \\exp\\left(-\\frac{{(\\Delta\\phi_k)^2}}{{2\\sigma^2}}\\right) \\cdot \\mathbb{{I}}(\\tau_k = \\tau^*) }} + 10^{{-6}}$$
                        </p>
                    </div>
                    
                    <div class="stats">
                        <h3>Summary Statistics</h3>
                        <p>Total Sources: {len(component_df)} | Query Source: {query_source}</p>
                        <p>Total Weight: {component_df['final_weight'].sum():.4f} | Mean Weight: {component_df['final_weight'].mean():.4f}</p>
                        <p>Matching Defects: {component_df['defect_match'].sum()}/{len(component_df)}</p>
                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download HTML Report",
                    data=html_content,
                    file_name=f"weight_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
    
    else:
        # Initial state - no data generated yet
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>📊 Mathematical Weight Component Visualizer</h2>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">
                Visualize the mathematical decomposition of attention weights using the formula:
            </p>
            
            <div style="background-color: #f8f9fa; padding: 30px; border-radius: 15px; margin: 20px auto; max-width: 800px;">
                <p style="font-size: 1.4rem; color: #1E3A8A; margin-bottom: 20px;">
                    $$w_i(\boldsymbol{\theta}^*) = \frac{\bar{\alpha}_i(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_i)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_i = \tau^*) }{ \sum_{k=1}^{N} \bar{\alpha}_k(\boldsymbol{\theta}^*) \cdot \exp\left(-\frac{(\Delta\phi_k)^2}{2\sigma^2}\right) \cdot \mathbb{I}(\tau_k = \tau^*) } + 10^{-6}$$
                </p>
                
                <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-top: 30px;">
                    <h3>🎯 Formula Components:</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                        <div style="text-align: left;">
                            <p><strong>ᾱ_i(θ*)</strong>: Pre-mask attention score</p>
                            <p><strong>exp(-Δφ²/2σ²)</strong>: Angular proximity weight</p>
                            <p><strong>𝕀(τ_i = τ*)</strong>: Defect type indicator</p>
                        </div>
                        <div style="text-align: left;">
                            <p><strong>Numerator</strong>: Product of three components</p>
                            <p><strong>Denominator</strong>: Sum over all sources</p>
                            <p><strong>10⁻⁶</strong>: Small offset for numerical stability</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 40px;">
                <p style="font-size: 1.1rem; color: #555;">
                    Click <strong>"Generate Synthetic Data"</strong> in the sidebar to begin visualization
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
