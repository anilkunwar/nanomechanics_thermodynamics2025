import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================
# ENHANCED SANKEY VISUALIZER (Focused Component)
# =============================================
class EnhancedSankeyVisualizer:
    """Focused visualizer for Enhanced Sankey diagrams with customization options"""
    
    def __init__(self):
        self.color_scheme = {
            'Spatial': '#36A2EB',    # Bright Blue
            'Defect': '#FF6384',     # Pink
            'Attention': '#4BC0C0',  # Cyan
            'Combined': '#9966FF',   # Purple
            'Target': '#FF6B6B',     # Bright Red
            'Source': '#4ECDC4'      # Turquoise
        }
    
    def create_enhanced_sankey(
        self,
        sources_data: List[Dict[str, Any]],
        target_angle: float,
        target_defect: str,
        spatial_sigma: float,
        # Customization parameters
        label_font_size: int = 16,
        title_font_size: int = 24,
        node_thickness: int = 30,
        node_padding: int = 25,
        link_opacity: float = 0.8,
        show_annotations: bool = True,
        diagram_width: int = 1400,
        diagram_height: int = 900,
        use_dark_mode: bool = False
    ) -> go.Figure:
        """
        Create enhanced Sankey diagram with full customization options
        
        Parameters:
        -----------
        sources_data : List of source weight data dictionaries
        target_angle : Target angle in degrees
        target_defect : Target defect type
        spatial_sigma : Angular kernel width
        label_font_size : Font size for node labels (default: 16)
        title_font_size : Font size for title (default: 24)
        node_thickness : Thickness of Sankey nodes (default: 30)
        node_padding : Padding between nodes (default: 25)
        link_opacity : Opacity of links (0.0 to 1.0, default: 0.8)
        show_annotations : Show color legend annotations (default: True)
        diagram_width : Width of diagram in pixels (default: 1400)
        diagram_height : Height of diagram in pixels (default: 900)
        use_dark_mode : Use dark mode color scheme (default: False)
        """
        # Create nodes for Sankey diagram
        labels = ['Target']  # Start with target node
        
        # Add source nodes with enhanced labels
        for source in sources_data:
            angle = source['theta_deg']
            defect = source['defect_type']
            labels.append(f"S{source['source_index']}\n{defect}\n{angle:.1f}°")
        
        # Add component nodes
        component_start = len(labels)
        component_labels = ['Spatial Kernel', 'Defect Match', 'Attention Score', 'Combined Weight']
        labels.extend(component_labels)
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        link_labels = []
        
        # Links from sources to components
        for i, source in enumerate(sources_data):
            source_idx = i + 1  # +1 because index 0 is target
            
            # Link to spatial kernel
            source_indices.append(source_idx)
            target_indices.append(component_start)
            spatial_value = source['spatial_weight'] * 100
            values.append(spatial_value)
            colors.append(f'rgba(54, 162, 235, {link_opacity})')
            link_labels.append(f"Spatial: {source['spatial_weight']:.3f}")
            
            # Link to defect match
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            defect_value = source['defect_weight'] * 100
            values.append(defect_value)
            colors.append(f'rgba(255, 99, 132, {link_opacity})')
            link_labels.append(f"Defect: {source['defect_weight']:.3f}")
            
            # Link to attention score
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_value = attention_w * 100
            values.append(attention_value)
            colors.append(f'rgba(75, 192, 192, {link_opacity})')
            link_labels.append(f"Attention: {attention_w:.3f}")
            
            # Link to combined weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            combined_value = source['combined_weight'] * 100
            values.append(combined_value)
            colors.append(f'rgba(153, 102, 255, {link_opacity})')
            link_labels.append(f"Combined: {source['combined_weight']:.3f}")
        
        # Links from components to target
        for comp_idx in range(4):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)  # Target node
            
            # Sum of all flows into this component
            comp_value = sum(v for s, t, v in zip(source_indices[:-4], target_indices[:-4], values[:-4])
                           if t == component_start + comp_idx)
            values.append(comp_value * 0.5)  # Reduce flow to target for visual clarity
            
            # Component-specific colors with opacity
            base_colors = [
                'rgba(54, 162, 235',    # Spatial
                'rgba(255, 99, 132',    # Defect
                'rgba(75, 192, 192',    # Attention
                'rgba(153, 102, 255'    # Combined
            ]
            colors.append(f'{base_colors[comp_idx]}, {link_opacity * 0.7})')
            link_labels.append(f"{component_labels[comp_idx]} → Target")
        
        # Set up color scheme based on mode
        bg_color = 'rgb(30, 30, 30)' if use_dark_mode else 'rgba(245, 247, 250, 0.95)'
        paper_color = 'rgb(20, 20, 20)' if use_dark_mode else 'white'
        #text_color = 'white' if use_dark_mode else '#2C3E50' # Dark blue-gray
        text_color = 'white' if use_dark_mode else '#000000' # Pure black
        grid_color = 'rgba(255, 255, 255, 0.1)' if use_dark_mode else 'rgba(0, 0, 0, 0.1)'
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=node_padding,
                thickness=node_thickness,
                line=dict(color="white" if use_dark_mode else "black", width=2),
                label=labels,
                color=[
                    self.color_scheme['Target']  # Target node
                ] + [
                    self.color_scheme['Source'] for _ in range(len(sources_data))  # Source nodes
                ] + [
                    self.color_scheme['Spatial'],
                    self.color_scheme['Defect'],
                    self.color_scheme['Attention'],
                    self.color_scheme['Combined']
                ],
                hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<br>%{customdata}<extra></extra>',
                customdata=link_labels,
                line=dict(width=0.5, color='rgba(255,255,255,0.3)')
            ),
            hoverinfo='all'
        )])
        
        # Enhanced layout with customizable fonts
        fig.update_layout(
            title=dict(
                text=f'<b>SANKEY DIAGRAM: ATTENTION COMPONENT FLOW</b><br>'
                     f'<span style="font-size: {title_font_size-4}px; font-weight: normal;">'
                     f'Target: {target_angle}° {target_defect} | Δ={spatial_sigma}°</span>',
                font=dict(
                    family='Arial, sans-serif',
                    size=title_font_size,
                    color=text_color,
                    weight='bold'
                ),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            font=dict(
                family='Arial, sans-serif',
                size=label_font_size,
                color=text_color
            ),
            width=diagram_width,
            height=diagram_height,
            plot_bgcolor=bg_color,
            paper_bgcolor=paper_color,
            margin=dict(t=100, l=50, r=50, b=80),
            hoverlabel=dict(
                font=dict(
                    family='Arial, sans-serif',
                    size=label_font_size,
                    color='white' if use_dark_mode else 'black'
                ),
                bgcolor='rgba(44, 62, 80, 0.95)' if not use_dark_mode else 'rgba(200, 200, 200, 0.9)',
                bordercolor='white' if use_dark_mode else 'black'
            )
        )
        
        # Add customizable annotations for color coding
        if show_annotations:
            annotations = [
                dict(
                    x=0.02, y=-0.08,
                    xref='paper', yref='paper',
                    text='<b style="font-size: 20px;">COLOR CODING:</b>',
                    showarrow=False,
                    font=dict(size=label_font_size, color=text_color, family='Arial')
                ),
                dict(
                    x=0.18, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Spatial"]}; font-size: 28px;">█</span> Spatial',
                    showarrow=False,
                    font=dict(size=label_font_size+2, color=text_color, family='Arial')
                ),
                dict(
                    x=0.32, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Defect"]}; font-size: 28px;">█</span> Defect',
                    showarrow=False,
                    font=dict(size=label_font_size+2, color=text_color, family='Arial')
                ),
                dict(
                    x=0.46, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Attention"]}; font-size: 28px;">█</span> Attention',
                    showarrow=False,
                    font=dict(size=label_font_size+2, color=text_color, family='Arial')
                ),
                dict(
                    x=0.62, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Combined"]}; font-size: 28px;">█</span> Combined',
                    showarrow=False,
                    font=dict(size=label_font_size+2, color=text_color, family='Arial')
                )
            ]
            fig.update_layout(annotations=annotations)
        
        return fig


# =============================================
# STREAMLIT APPLICATION (Sankey-Focused)
# =============================================
def main_sankey_visualization():
    st.set_page_config(
        page_title="Enhanced Sankey Diagram - Weight Visualization",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 1.5rem;
    }
    .config-section {
        background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #3B82F6;
    }
    .viz-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 ENHANCED SANKEY DIAGRAM VISUALIZATION</h1>', unsafe_allow_html=True)
    
    # Initialize session state for demo data
    if 'sankey_demo_data' not in st.session_state:
        st.session_state.sankey_demo_data = [
            {'source_index': 0, 'theta_deg': 0.0, 'defect_type': 'Twin', 
             'spatial_weight': 0.35, 'defect_weight': 1.0, 'attention_weight': 0.42, 'combined_weight': 0.38,
             'target_defect_match': True, 'angular_dist': 9.7},
            {'source_index': 1, 'theta_deg': 30.0, 'defect_type': 'Twin', 
             'spatial_weight': 0.48, 'defect_weight': 1.0, 'attention_weight': 0.51, 'combined_weight': 0.45,
             'target_defect_match': True, 'angular_dist': 4.7},
            {'source_index': 2, 'theta_deg': 60.0, 'defect_type': 'Twin', 
             'spatial_weight': 0.42, 'defect_weight': 1.0, 'attention_weight': 0.38, 'combined_weight': 0.35,
             'target_defect_match': True, 'angular_dist': 5.3},
            {'source_index': 3, 'theta_deg': 30.0, 'defect_type': 'ISF', 
             'spatial_weight': 0.15, 'defect_weight': 1e-6, 'attention_weight': 0.12, 'combined_weight': 0.001,
             'target_defect_match': False, 'angular_dist': 24.7},
        ]
    
    # Sidebar configuration - FOCUSED ON SANKEY CUSTOMIZATION
    with st.sidebar:
        st.markdown('<h2 style="color: #3B82F6; border-bottom: 3px solid #3B82F6; padding-bottom: 0.5rem;">🎨 SANKEY CUSTOMIZATION</h2>', unsafe_allow_html=True)
        
        st.markdown("### 📏 Font & Label Settings")
        label_font_size = st.slider(
            "Node Label Font Size",
            min_value=8,
            max_value=24,
            value=16,
            step=1,
            help="Adjust the font size of node labels in the Sankey diagram"
        )
        
        title_font_size = st.slider(
            "Title Font Size",
            min_value=16,
            max_value=36,
            value=24,
            step=1,
            help="Adjust the font size of the diagram title"
        )
        
        st.markdown("### 🧱 Node Appearance")
        node_thickness = st.slider(
            "Node Thickness",
            min_value=15,
            max_value=50,
            value=30,
            step=1,
            help="Control the thickness of Sankey nodes"
        )
        
        node_padding = st.slider(
            "Node Padding",
            min_value=10,
            max_value=40,
            value=25,
            step=1,
            help="Adjust spacing between nodes"
        )
        
        st.markdown("### 🔗 Link Appearance")
        link_opacity = st.slider(
            "Link Opacity",
            min_value=0.3,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Control transparency of connection links"
        )
        
        st.markdown("### 🖼️ Diagram Layout")
        col1, col2 = st.columns(2)
        with col1:
            diagram_width = st.number_input(
                "Width (px)",
                min_value=800,
                max_value=2000,
                value=1400,
                step=100,
                help="Width of the diagram in pixels"
            )
        with col2:
            diagram_height = st.number_input(
                "Height (px)",
                min_value=600,
                max_value=1500,
                value=900,
                step=100,
                help="Height of the diagram in pixels"
            )
        
        st.markdown("### 🎨 Display Options")
        show_annotations = st.toggle(
            "Show Color Legend",
            value=True,
            help="Display color coding legend below the diagram"
        )
        
        use_dark_mode = st.toggle(
            "Dark Mode",
            value=False,
            help="Switch to dark color scheme for better contrast"
        )
        
        st.markdown("### 🎯 Target Configuration")
        target_angle = st.number_input(
            "Target Angle (°)",
            min_value=0.0,
            max_value=180.0,
            value=54.7,
            step=0.1,
            help="Target angle for interpolation"
        )
        
        target_defect = st.selectbox(
            "Target Defect Type",
            options=['Twin', 'ISF', 'ESF', 'No Defect'],
            index=0,
            help="Defect type for target configuration"
        )
        
        spatial_sigma = st.slider(
            "Kernel Width σ (°)",
            min_value=1.0,
            max_value=45.0,
            value=10.0,
            step=0.5,
            help="Angular bracketing kernel width"
        )
        
        st.divider()
        st.markdown("💡 **Tip**: Increase font sizes for presentations or publications. Adjust node thickness for better visibility on large screens.")
    
    # Main content area
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Current Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Label Font Size", f"{label_font_size}px")
    with col2:
        st.metric("Node Thickness", f"{node_thickness}px")
    with col3:
        st.metric("Link Opacity", f"{link_opacity:.1f}")
    with col4:
        st.metric("Diagram Size", f"{diagram_width}×{diagram_height}px")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create and display Sankey diagram
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    st.markdown("### 🌊 Enhanced Sankey Diagram")
    
    # Initialize visualizer and create diagram with current settings
    visualizer = EnhancedSankeyVisualizer()
    fig = visualizer.create_enhanced_sankey(
        sources_data=st.session_state.sankey_demo_data,
        target_angle=target_angle,
        target_defect=target_defect,
        spatial_sigma=spatial_sigma,
        label_font_size=label_font_size,
        title_font_size=title_font_size,
        node_thickness=node_thickness,
        node_padding=node_padding,
        link_opacity=link_opacity,
        show_annotations=show_annotations,
        diagram_width=diagram_width,
        diagram_height=diagram_height,
        use_dark_mode=use_dark_mode
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Explanation section
    st.markdown("### ℹ️ Visualization Guide")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Node Types:**
        - 🔴 **Red (Target)**: Query configuration being interpolated
        - 🟢 **Teal (Sources)**: Training configurations used for interpolation
        - 🔵 **Blue (Spatial)**: Angular proximity contribution
        - 🟣 **Purple (Combined)**: Final attention weights
        
        **Link Thickness:**
        - Proportional to weight magnitude
        - Thicker links = higher contribution to interpolation
        """)
    with col2:
        st.markdown("""
        **Weight Components:**
        - **Spatial Kernel**: Angular proximity (Gaussian decay)
        - **Defect Match**: Hard constraint (1.0 if match, ~0 if mismatch)
        - **Attention Score**: Learned similarity from transformer
        - **Combined Weight**: Final normalized attention weight
        
        **Interactive Features:**
        - Hover over nodes/links for detailed values
        - Zoom and pan for detailed inspection
        - Full-screen mode available (↗️ icon)
        """)

if __name__ == "__main__":
    main_sankey_visualization()
