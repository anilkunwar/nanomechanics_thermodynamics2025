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
from math import cos, sin, pi
warnings.filterwarnings('ignore')

# =============================================
# SET PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# =============================================
st.set_page_config(
    page_title="Advanced Weight Analysis - Angular Bracketing Theory",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

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
# ENHANCED WEIGHT VISUALIZER WITH CUSTOMIZABLE SANKEY
# =============================================
class EnhancedWeightVisualizer:
    def __init__(self):
        # Vibrant color scheme for better contrast
        self.color_scheme = {
            'Twin': '#FF6B6B',
            'ISF': '#4ECDC4',
            'ESF': '#95E1D3',
            'No Defect': '#FFD93D',
            'Query': '#9D4EDD',
            'Spatial': '#36A2EB',
            'Defect': '#FF6384',
            'Attention': '#4BC0C0',
            'Combined': '#9966FF'
        }
        
    def create_enhanced_sankey_diagram(self, sources_data, target_angle, target_defect, 
                                       spatial_sigma, font_config, visual_params):
        """
        Create enhanced Sankey diagram with adjustable font sizes and visual parameters
        
        Args:
            sources_data: List of source data dictionaries
            target_angle: Target angle in degrees
            target_defect: Target defect type
            spatial_sigma: Angular kernel width
            font_config: Dictionary with font size configurations
            visual_params: Dictionary with visual enhancement parameters
        """
        # Create nodes for Sankey diagram
        labels = ['Target']
        
        # Add source nodes
        for source in sources_data:
            angle = source['theta_deg']
            defect = source['defect_type']
            labels.append(f"S{source['source_index']}\n{defect}\n{angle:.1f}°")
        
        # Add component nodes
        component_start = len(labels)
        labels.extend(['Spatial Kernel', 'Defect Match', 'Attention Score', 'Combined Weight'])
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        link_labels = []
        
        # Calculate link values
        for i, source in enumerate(sources_data):
            source_idx = i + 1
            
            # Spatial kernel link
            source_indices.append(source_idx)
            target_indices.append(component_start)
            spatial_value = source['spatial_weight'] * 100
            values.append(spatial_value)
            colors.append('rgba(54, 162, 235, 0.8)')  # Blue
            link_labels.append(f"Spatial: {source['spatial_weight']:.3f}")
            
            # Defect match link
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            defect_value = source['defect_weight'] * 100
            values.append(defect_value)
            colors.append('rgba(255, 99, 132, 0.8)')  # Pink
            link_labels.append(f"Defect: {source['defect_weight']:.3f}")
            
            # Attention score link
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_value = attention_w * 100
            values.append(attention_value)
            colors.append('rgba(75, 192, 192, 0.8)')  # Cyan
            link_labels.append(f"Attention: {attention_w:.3f}")
            
            # Combined weight link
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            combined_value = source['combined_weight'] * 100
            values.append(combined_value)
            colors.append('rgba(153, 102, 255, 0.8)')  # Purple
            link_labels.append(f"Combined: {source['combined_weight']:.3f}")
        
        # Links from components to target
        for comp_idx in range(4):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)
            
            # Sum of all flows into this component
            comp_value = sum(v for s, t, v in zip(source_indices, target_indices, values) 
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.5)
            
            # Component-specific colors
            if comp_idx == 0:
                colors.append('rgba(54, 162, 235, 0.6)')
            elif comp_idx == 1:
                colors.append('rgba(255, 99, 132, 0.6)')
            elif comp_idx == 2:
                colors.append('rgba(75, 192, 192, 0.6)')
            else:
                colors.append('rgba(153, 102, 255, 0.6)')
            
            link_labels.append(f"Component {comp_idx} → Target")
        
        # Adjust colors based on visual parameters
        if visual_params.get('enhance_colors', False):
            # Enhance color saturation
            colors = [color.replace('0.8', '0.95').replace('0.6', '0.8') for color in colors]
        
        # Create Sankey diagram with configurable parameters
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=visual_params.get('node_padding', 25),
                thickness=visual_params.get('node_thickness', 30),
                line=dict(
                    color="black", 
                    width=visual_params.get('border_width', 2)
                ),
                label=labels,
                color=self._get_node_colors(labels, sources_data),
                hoverinfo='label+value',
                hoverlabel=dict(
                    font=dict(
                        size=font_config.get('hover_font_size', 14),
                        family='Arial, sans-serif'
                    ),
                    bgcolor='rgba(44, 62, 80, 0.9)',
                    bordercolor='white'
                )
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<extra></extra>',
                line=dict(
                    width=0.5, 
                    color='rgba(255,255,255,0.3)'
                )
            )
        )])
        
        # Enhanced layout with configurable font sizes
        fig.update_layout(
            title=dict(
                text=f'<b>ENHANCED SANKEY DIAGRAM</b><br>Target: {target_angle}° {target_defect} | σ={spatial_sigma}°',
                font=dict(
                    family='Arial, sans-serif',
                    size=font_config.get('title_font_size', 24),
                    color='#2C3E50',
                    weight='bold'
                ),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            font=dict(
                family='Arial, sans-serif',
                size=font_config.get('label_font_size', 16),
                color='#2C3E50'
            ),
            width=visual_params.get('width', 1400),
            height=visual_params.get('height', 900),
            plot_bgcolor=visual_params.get('plot_bgcolor', 'rgba(240, 240, 245, 0.9)'),
            paper_bgcolor=visual_params.get('paper_bgcolor', 'white'),
            margin=dict(
                t=font_config.get('title_font_size', 24) + 50,
                l=50, 
                r=50, 
                b=50
            ),
            hoverlabel=dict(
                font=dict(
                    family='Arial, sans-serif',
                    size=font_config.get('hover_font_size', 14),
                    color='white'
                ),
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white'
            )
        )
        
        # Add annotations with configurable font size
        annotations = self._create_sankey_annotations(font_config)
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def _get_node_colors(self, labels, sources_data):
        """Get color palette for nodes"""
        colors = []
        
        # Target node
        colors.append(self.color_scheme['Query'])
        
        # Source nodes - use defect-specific colors
        for i in range(len(sources_data)):
            defect = sources_data[i]['defect_type']
            colors.append(self.color_scheme.get(defect, '#CCCCCC'))
        
        # Component nodes
        colors.extend([
            self.color_scheme['Spatial'],
            self.color_scheme['Defect'],
            self.color_scheme['Attention'],
            self.color_scheme['Combined']
        ])
        
        return colors
    
    def _create_sankey_annotations(self, font_config):
        """Create Sankey diagram annotations"""
        return [
            dict(
                x=0.02, y=1.05,
                xref='paper', yref='paper',
                text='<b>COLOR CODING:</b>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 14),
                    color='darkblue',
                    weight='bold'
                )
            ),
            dict(
                x=0.02, y=1.02,
                xref='paper', yref='paper',
                text='• Spatial: <span style="color:#36A2EB">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 14),
                    color='#36A2EB'
                )
            ),
            dict(
                x=0.15, y=1.02,
                xref='paper', yref='paper',
                text='• Defect: <span style="color:#FF6384">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 14),
                    color='#FF6384'
                )
            ),
            dict(
                x=0.28, y=1.02,
                xref='paper', yref='paper',
                text='• Attention: <span style="color:#4BC0C0">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 14),
                    color='#4BC0C0'
                )
            ),
            dict(
                x=0.41, y=1.02,
                xref='paper', yref='paper',
                text='• Combined: <span style="color:#9966FF">█</span>',
                showarrow=False,
                font=dict(
                    size=font_config.get('annotation_font_size', 14),
                    color='#9966FF'
                )
            )
        ]
    
    def create_visualization_dashboard(self, sources_data, target_angle, target_defect, 
                                       spatial_sigma, user_config):
        """
        Create comprehensive visualization dashboard with user-configurable parameters
        
        Args:
            user_config: Dictionary with user configuration from sidebar
        """
        # Extract font configuration from user config
        font_config = {
            'title_font_size': user_config.get('title_font_size', 24),
            'label_font_size': user_config.get('label_font_size', 16),
            'annotation_font_size': user_config.get('annotation_font_size', 14),
            'hover_font_size': user_config.get('hover_font_size', 12),
            'legend_font_size': user_config.get('legend_font_size', 14)
        }
        
        # Extract visual parameters from user config
        visual_params = {
            'node_padding': user_config.get('node_padding', 25),
            'node_thickness': user_config.get('node_thickness', 30),
            'border_width': user_config.get('border_width', 2),
            'width': user_config.get('width', 1400),
            'height': user_config.get('height', 900),
            'plot_bgcolor': user_config.get('plot_bgcolor', 'rgba(240, 240, 245, 0.9)'),
            'paper_bgcolor': user_config.get('paper_bgcolor', 'white'),
            'enhance_colors': user_config.get('enhance_colors', False),
            'show_flow_values': user_config.get('show_flow_values', True)
        }
        
        # Create the enhanced Sankey diagram
        fig = self.create_enhanced_sankey_diagram(
            sources_data=sources_data,
            target_angle=target_angle,
            target_defect=target_defect,
            spatial_sigma=spatial_sigma,
            font_config=font_config,
            visual_params=visual_params
        )
        
        return fig

# =============================================
# HELPER FUNCTIONS
# =============================================
def create_demo_data():
    """Create demo data for visualization testing"""
    sources = []
    defect_types = ['Twin', 'ISF', 'ESF', 'No Defect']
    
    for i in range(6):
        defect = defect_types[i % len(defect_types)]
        sources.append({
            'source_index': i,
            'theta_deg': np.random.uniform(0, 180),
            'defect_type': defect,
            'spatial_weight': np.random.uniform(0.1, 0.5),
            'defect_weight': 1.0 if defect == 'Twin' else 0.1,
            'attention_weight': np.random.uniform(0.2, 0.8),
            'combined_weight': np.random.uniform(0.1, 0.6),
            'target_defect_match': defect == 'Twin'
        })
    
    return {
        'sources': sources,
        'target_angle': 54.7,
        'target_defect': 'Twin',
        'spatial_sigma': 10.0
    }

def save_configuration(config):
    """Save current configuration to file"""
    config['saved_at'] = datetime.now().isoformat()
    
    with open('sankey_config.json', 'w') as f:
        json.dump(config, f, indent=2)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    # Enhanced CSS with custom styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #EF4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sankey-controls {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .font-control-group {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .visual-control-group {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .control-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: block;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .stSlider > div > div > div {
        color: white !important;
    }
    .section-header {
        font-size: 2.2rem !important;
        color: #2C3E50 !important;
        font-weight: 800 !important;
        border-left: 8px solid #3B82F6;
        padding-left: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(to right, #F0F9FF, white);
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 Enhanced Sankey Visualization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Customize font sizes and visual parameters for optimal viewing")
    
    # Initialize session state
    if 'weight_visualizer' not in st.session_state:
        st.session_state.weight_visualizer = EnhancedWeightVisualizer()
    
    # Generate demo data
    demo_data = create_demo_data()
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.markdown('<div class="sankey-controls">', unsafe_allow_html=True)
        st.markdown("### 🎨 Sankey Diagram Controls")
        
        # Font size controls
        st.markdown('<div class="font-control-group">', unsafe_allow_html=True)
        st.markdown('<span class="control-label">📝 Font Sizes</span>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            title_font_size = st.slider(
                "Title Size",
                min_value=16,
                max_value=40,
                value=24,
                step=1,
                help="Title font size in points"
            )
        
        with col2:
            label_font_size = st.slider(
                "Label Size",
                min_value=10,
                max_value=30,
                value=16,
                step=1,
                help="Node label font size in points"
            )
        
        annotation_font_size = st.slider(
            "Annotation Size",
            min_value=10,
            max_value=24,
            value=14,
            step=1,
            help="Annotation text font size"
        )
        
        hover_font_size = st.slider(
            "Hover Text Size",
            min_value=10,
            max_value=20,
            value=12,
            step=1,
            help="Hover text font size"
        )
        
        legend_font_size = st.slider(
            "Legend Font Size",
            min_value=10,
            max_value=24,
            value=14,
            step=1,
            help="Legend font size"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual controls
        st.markdown('<div class="visual-control-group">', unsafe_allow_html=True)
        st.markdown('<span class="control-label">🎨 Visual Parameters</span>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            node_padding = st.slider(
                "Node Padding",
                min_value=10,
                max_value=50,
                value=25,
                step=1,
                help="Padding between nodes"
            )
        
        with col4:
            node_thickness = st.slider(
                "Node Thickness",
                min_value=15,
                max_value=50,
                value=30,
                step=1,
                help="Thickness of nodes"
            )
        
        border_width = st.slider(
            "Border Width",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Border width of nodes"
        )
        
        # Color enhancement
        enhance_colors = st.checkbox(
            "🎨 Enhance Colors",
            value=True,
            help="Increase color saturation for better visibility"
        )
        
        show_flow_values = st.checkbox(
            "🔢 Show Flow Values",
            value=True,
            help="Display flow values in hover text"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Layout controls
        st.markdown('<div class="visual-control-group">', unsafe_allow_html=True)
        st.markdown('<span class="control-label">📐 Layout Settings</span>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        with col5:
            width = st.slider(
                "Width",
                min_value=800,
                max_value=2000,
                value=1400,
                step=50,
                help="Diagram width in pixels"
            )
        
        with col6:
            height = st.slider(
                "Height",
                min_value=600,
                max_value=1200,
                value=900,
                step=50,
                help="Diagram height in pixels"
            )
        
        # Background color
        bg_color = st.color_picker(
            "Background Color",
            value="#F0F0F5",
            help="Select plot background color"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Preset buttons
        st.markdown("### 🚀 Quick Presets")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("📱 Mobile", use_container_width=True):
                st.session_state.mobile_preset = True
                st.rerun()
        
        with col_p2:
            if st.button("🖥️ Desktop", use_container_width=True):
                st.session_state.desktop_preset = True
                st.rerun()
        
        with col_p3:
            if st.button("📊 Presentation", use_container_width=True):
                st.session_state.presentation_preset = True
                st.rerun()
    
    # Handle presets
    if 'mobile_preset' in st.session_state and st.session_state.mobile_preset:
        title_font_size = 20
        label_font_size = 12
        annotation_font_size = 10
        width = 800
        height = 600
        st.session_state.mobile_preset = False
    
    if 'desktop_preset' in st.session_state and st.session_state.desktop_preset:
        title_font_size = 24
        label_font_size = 16
        annotation_font_size = 14
        width = 1200
        height = 800
        st.session_state.desktop_preset = False
    
    if 'presentation_preset' in st.session_state and st.session_state.presentation_preset:
        title_font_size = 28
        label_font_size = 18
        annotation_font_size = 16
        width = 1600
        height = 1000
        enhance_colors = True
        st.session_state.presentation_preset = False
    
    # Get user configuration
    user_config = {
        'title_font_size': title_font_size,
        'label_font_size': label_font_size,
        'annotation_font_size': annotation_font_size,
        'hover_font_size': hover_font_size,
        'legend_font_size': legend_font_size,
        'node_padding': node_padding,
        'node_thickness': node_thickness,
        'border_width': border_width,
        'width': width,
        'height': height,
        'plot_bgcolor': bg_color,
        'paper_bgcolor': 'white',
        'enhance_colors': enhance_colors,
        'show_flow_values': show_flow_values
    }
    
    # Main content area
    # Display current configuration
    st.markdown('<h2 class="section-header">📋 Current Configuration</h2>', unsafe_allow_html=True)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f'<div class="metric-card">Title Font Size<br>{title_font_size} pt</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Label Font Size<br>{label_font_size} pt</div>', unsafe_allow_html=True)
    
    with col_c2:
        st.markdown(f'<div class="metric-card">Diagram Width<br>{width} px</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Diagram Height<br>{height} px</div>', unsafe_allow_html=True)
    
    with col_c3:
        st.markdown(f'<div class="metric-card">Node Thickness<br>{node_thickness} px</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Border Width<br>{border_width} px</div>', unsafe_allow_html=True)
    
    # Create and display the Sankey diagram
    st.markdown('<h2 class="section-header">🌀 Enhanced Sankey Diagram</h2>', unsafe_allow_html=True)
    
    try:
        fig = st.session_state.weight_visualizer.create_visualization_dashboard(
            sources_data=demo_data['sources'],
            target_angle=demo_data['target_angle'],
            target_defect=demo_data['target_defect'],
            spatial_sigma=demo_data['spatial_sigma'],
            user_config=user_config
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown('<h2 class="section-header">💾 Export Options</h2>', unsafe_allow_html=True)
        
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            if st.button("💾 Save Configuration", use_container_width=True, type="primary"):
                save_configuration(user_config)
                st.success("Configuration saved to 'sankey_config.json'!")
        
        with col_e2:
            # Export as HTML
            html = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📥 Export as HTML",
                data=html,
                file_name="enhanced_sankey.html",
                mime="text/html",
                use_container_width=True,
                type="secondary"
            )
        
        with col_e3:
            # Export as PNG
            if st.button("🖼️ Export as PNG", use_container_width=True, type="secondary"):
                with st.spinner("Generating PNG..."):
                    img_bytes = fig.to_image(format="png", width=width, height=height)
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name="enhanced_sankey.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.info("Try adjusting the parameters or using different data.")
    
    # Source data preview
    with st.expander("📊 View Source Data", expanded=False):
        st.dataframe(pd.DataFrame(demo_data['sources']), use_container_width=True)

# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
