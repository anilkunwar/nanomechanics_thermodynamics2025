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
# UPDATED MAIN APPLICATION WITH ENHANCED CONTROLS
# =============================================
def main():
    st.set_page_config(
        page_title="Advanced Weight Analysis - Enhanced Visualization",
        layout="wide",
        page_icon="📊"
    )
    
    # Enhanced CSS with custom styling
    st.markdown("""
    <style>
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
    .stSlider > div > div > div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📊 Enhanced Sankey Visualization Dashboard")
    st.markdown("### Customize font sizes and visual parameters for optimal viewing")
    
    # Initialize session state
    if 'weight_visualizer' not in st.session_state:
        st.session_state.weight_visualizer = EnhancedWeightVisualizer()
    
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
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("📱 Mobile View", use_container_width=True):
                # Reset to mobile-friendly settings
                title_font_size = 20
                label_font_size = 12
                annotation_font_size = 10
                width = 800
                height = 600
        
        with col_p2:
            if st.button("🖥️ Presentation", use_container_width=True):
                # Reset to presentation settings
                title_font_size = 28
                label_font_size = 18
                annotation_font_size = 16
                width = 1600
                height = 1000
                enhance_colors = True
    
    # Main content area
    # Generate sample data for demonstration
    # In practice, this would come from your interpolation result
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = create_demo_data()
    
    # Get user configuration
    user_config = {
        'title_font_size': title_font_size,
        'label_font_size': label_font_size,
        'annotation_font_size': annotation_font_size,
        'hover_font_size': hover_font_size,
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
    
    # Display current configuration
    with st.expander("📋 Current Configuration", expanded=False):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Title Font Size", f"{title_font_size} pt")
            st.metric("Label Font Size", f"{label_font_size} pt")
            st.metric("Diagram Width", f"{width} px")
        with col_c2:
            st.metric("Annotation Font Size", f"{annotation_font_size} pt")
            st.metric("Node Thickness", f"{node_thickness} px")
            st.metric("Diagram Height", f"{height} px")
    
    # Create and display the Sankey diagram
    try:
        fig = st.session_state.weight_visualizer.create_visualization_dashboard(
            sources_data=st.session_state.demo_data['sources'],
            target_angle=st.session_state.demo_data['target_angle'],
            target_defect=st.session_state.demo_data['target_defect'],
            spatial_sigma=st.session_state.demo_data['spatial_sigma'],
            user_config=user_config
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.markdown("---")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            if st.button("💾 Save Configuration", use_container_width=True):
                save_configuration(user_config)
                st.success("Configuration saved!")
        
        with col_e2:
            # Export as HTML
            html = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📥 Export as HTML",
                data=html,
                file_name="sankey_diagram.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col_e3:
            # Export as PNG
            if st.button("🖼️ Export as PNG", use_container_width=True):
                img_bytes = fig.to_image(format="png", width=width, height=height)
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="sankey_diagram.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.info("Try adjusting the parameters or using different data.")

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
    import json
    config['saved_at'] = datetime.now().isoformat()
    
    with open('sankey_config.json', 'w') as f:
        json.dump(config, f, indent=2)

# =============================================
# RUN THE ENHANCED APPLICATION
# =============================================
if __name__ == "__main__":
    main()
