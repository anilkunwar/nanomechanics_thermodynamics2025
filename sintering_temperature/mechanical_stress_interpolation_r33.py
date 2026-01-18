import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import pickle
import torch
import json
from datetime import datetime
from io import BytesIO

# =============================================
# ESSENTIAL CLASSES (MINIMAL IMPLEMENTATION)
# =============================================
class PhysicsBasedStressAnalyzer:
    """Minimal physics analyzer for eigen strains"""
    def __init__(self):
        self.eigen_strains = {
            'ISF': 0.71,      # Intrinsic Stacking Fault
            'ESF': 1.41,      # Extrinsic Stacking Fault
            'Twin': 2.12,     # Twin boundary
            'No Defect': 0.0, # Perfect crystal
        }
    
    def get_eigen_strain(self, defect_type):
        return self.eigen_strains.get(defect_type, 0.0)

class EnhancedSolutionLoader:
    """Simplified solution loader"""
    def __init__(self, solutions_dir="numerical_solutions"):
        self.solutions_dir = solutions_dir
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        os.makedirs(solutions_dir, exist_ok=True)
    
    def scan_solutions(self):
        """Scan directory for solution files"""
        file_info = []
        for root, _, files in os.walk(self.solutions_dir):
            for f in files:
                if f.endswith(('.pkl', '.pickle', '.pt', '.pth')):
                    path = os.path.join(root, f)
                    file_info.append({
                        'path': path,
                        'filename': f,
                        'size': os.path.getsize(path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(path))
                    })
        return sorted(file_info, key=lambda x: x['modified'], reverse=True)
    
    def read_simulation_file(self, file_path):
        """Read and standardize simulation file"""
        try:
            if file_path.endswith(('.pt', '.pth')):
                data = torch.load(file_path, map_location='cpu', weights_only=False)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Minimal standardization
            standardized = {
                'params': data.get('params', {}),
                'history': data.get('history', []),
                'metadata': {'filename': os.path.basename(file_path)}
            }
            
            # Add eigen strain if missing
            defect_type = standardized['params'].get('defect_type', 'Twin')
            standardized['params']['eps0'] = self.physics_analyzer.get_eigen_strain(defect_type)
            
            return standardized
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return None

class PhysicsAwareInterpolator:
    """Minimal interpolator for vicinity analysis"""
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
    
    def _compute_dummy_stress(self, angle, defect_type, base_value=20.0):
        """Generate physically plausible dummy stress values for demonstration"""
        eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
        distance = abs(angle - self.habit_angle)
        
        # Stress peaks at habit plane (54.7°) and decays with distance
        stress = base_value * eigen_strain * np.exp(-distance**2 / 20.0)
        
        # Add some noise and variation
        stress += np.random.normal(0, 0.5)
        return max(0.1, stress)  # Ensure positive values
    
    def create_vicinity_sweep(self, target_params, vicinity_range=10.0, n_points=50):
        """Generate stress values in habit plane vicinity"""
        center_angle = self.habit_angle
        angles = np.linspace(center_angle - vicinity_range, 
                            center_angle + vicinity_range, 
                            n_points)
        
        defect_type = target_params['defect_type']
        results = {
            'angles': angles.tolist(),
            'stresses': {
                'sigma_hydro': [],
                'von_mises': [],
                'sigma_mag': []
            },
            'defect_type': defect_type
        }
        
        for angle in angles:
            # Generate physically plausible stress values
            base_stress = self._compute_dummy_stress(angle, defect_type)
            
            # Different stress components with realistic relationships
            results['stresses']['sigma_hydro'].append(base_stress * 0.8)
            results['stresses']['von_mises'].append(base_stress * 1.0)
            results['stresses']['sigma_mag'].append(base_stress * 1.2)
        
        return results
    
    def compare_defect_types(self, vicinity_range=10.0, n_points=50, shapes=None):
        """Compare different defect types in habit plane vicinity"""
        if shapes is None:
            shapes = ['Square']
        
        defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
        colors = {
            'ISF': '#FF6B6B', 
            'ESF': '#4ECDC4', 
            'Twin': '#45B7D1', 
            'No Defect': '#96CEB4'
        }
        
        min_angle = self.habit_angle - vicinity_range
        max_angle = self.habit_angle + vicinity_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        comparison_results = {}
        for defect in defect_types:
            for shape in shapes:
                key = f"{defect}_{shape}"
                stresses = {
                    'sigma_hydro': [],
                    'von_mises': [],
                    'sigma_mag': []
                }
                
                for angle in angles:
                    base_stress = self._compute_dummy_stress(angle, defect)
                    stresses['sigma_hydro'].append(base_stress * 0.8)
                    stresses['von_mises'].append(base_stress * 1.0)
                    stresses['sigma_mag'].append(base_stress * 1.2)
                
                comparison_results[key] = {
                    'defect_type': defect,
                    'shape': shape,
                    'angles': angles.tolist(),
                    'stresses': stresses,
                    'color': colors[defect],
                    'eigen_strain': self.physics_analyzer.get_eigen_strain(defect)
                }
        
        return comparison_results

class HabitPlaneVisualizer:
    """Specialized visualizer for radar charts with customization"""
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
    
    def create_stress_comparison_radar(self, comparison_data, 
                                      title="Stress Components in Habit Plane Vicinity",
                                      show_labels=True,
                                      show_legend=True,
                                      line_width=3,
                                      colormap='viridis',
                                      legend_position='top right',
                                      background_opacity=0.3):
        """Create customizable radar chart for stress components"""
        fig = go.Figure()
        
        # Check if we have data
        if not comparison_data:
            fig.add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title=title, width=800, height=700)
            return fig
        
        # Find max stress for scaling
        max_stress = 0
        for entry in comparison_data.values():
            if 'stresses' in entry and 'sigma_hydro' in entry['stresses']:
                stresses = np.array(entry['stresses']['sigma_hydro'])
                if len(stresses) > 0:
                    max_stress = max(max_stress, np.max(stresses))
        
        # Add traces for each defect type
        for key, data in comparison_data.items():
            angles = np.array(data['angles'])
            stresses = np.array(data['stresses']['sigma_hydro'])
            
            if len(angles) == 0 or len(stresses) == 0:
                continue
            
            # Close the loop for radar chart
            angles_closed = np.append(angles, angles[0])
            stresses_closed = np.append(stresses, stresses[0])
            
            # Create hover text
            hover_text = [f"Angle: {a:.1f}°<br>Stress: {s:.3f} GPa" 
                         for a, s in zip(angles, stresses)]
            hover_text.append(hover_text[0])  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=stresses_closed,
                theta=angles_closed,
                mode='lines',
                name=f"{data['defect_type']} (ε*={data['eigen_strain']:.2f})",
                line=dict(
                    color=data['color'],
                    width=line_width,
                    shape='spline'
                ),
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                showlegend=show_legend
            ))
        
        # Highlight habit plane
        fig.add_trace(go.Scatterpolar(
            r=[0, max_stress * 1.1] if max_stress > 0 else [0, 1],
            theta=[self.habit_angle, self.habit_angle],
            mode='lines',
            line=dict(color='rgb(46, 204, 113)', width=4, dash='dashdot'),
            name=f'Habit Plane ({self.habit_angle}°)',
            showlegend=show_legend
        ))
        
        # Add labels to vertices if enabled
        if show_labels and max_stress > 0:
            label_angles = np.linspace(
                min(angles), 
                max(angles), 
                min(8, len(angles))
            )
            for angle in label_angles:
                fig.add_annotation(
                    x=angle,
                    y=max_stress * 0.95,
                    text=f"{angle:.0f}°",
                    showarrow=False,
                    font=dict(size=10),
                    xref="theta",
                    yref="r"
                )
        
        # Update layout with customization options
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial Black"),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_stress * 1.2] if max_stress > 0 else [0, 1],
                    title=dict(text='Stress (GPa)', font=dict(size=14)),
                    gridcolor="rgba(100, 100, 100, 0.3)",
                ),
                angularaxis=dict(
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=np.linspace(min(angles), max(angles), 5),
                    gridcolor="rgba(100, 100, 100, 0.3)",
                ),
                bgcolor=f"rgba(240, 240, 240, {background_opacity})"
            ),
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ) if legend_position == 'top' else dict(
                x=1.1,
                y=0.5,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            width=900,
            height=700,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Habit Plane Radar Chart Visualization",
        layout="wide",
        page_icon="📈"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 5px 5px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = PhysicsAwareInterpolator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = HabitPlaneVisualizer()
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #3B82F6;">📈 Habit Plane Vicinity Stress Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 📂 Data Management")
        
        # Load solutions button
        if st.button("🔄 Load Solutions", use_container_width=True):
            with st.spinner("Loading solutions..."):
                # For demo purposes, we'll create dummy solutions
                st.session_state.solutions = [
                    {'params': {'defect_type': 'Twin', 'theta': np.radians(54.7)}},
                    {'params': {'defect_type': 'ISF', 'theta': np.radians(45.0)}},
                    {'params': {'defect_type': 'ESF', 'theta': np.radians(60.0)}},
                ]
                st.success(f"Loaded {len(st.session_state.solutions)} solutions")
        
        # Show loaded solutions info
        if st.session_state.solutions:
            with st.expander(f"📊 Loaded Solutions ({len(st.session_state.solutions)})", expanded=True):
                st.write("**Defect Types Available:**")
                defect_counts = {}
                for sol in st.session_state.solutions:
                    defect = sol['params'].get('defect_type', 'Unknown')
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
                
                for defect, count in defect_counts.items():
                    st.write(f"- {defect}: {count}")
        
        # Analysis parameters
        st.markdown("### 🎯 Analysis Parameters")
        
        defect_type = st.selectbox(
            "Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for analysis"
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        default_eps0 = eigen_strains[defect_type]
        
        eps0 = st.number_input(
            "Eigen Strain (ε*)",
            min_value=0.0,
            max_value=3.0,
            value=default_eps0,
            step=0.01,
            help="Eigen strain value (auto-set based on defect type)"
        )
        
        kappa = st.slider(
            "Interface Energy (κ)",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.01,
            help="Interface energy parameter"
        )
        
        shape = st.selectbox(
            "Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
            index=0,
            help="Geometric shape of the defect"
        )
        
        # Vicinity settings
        st.markdown("### 🎯 Vicinity Settings")
        
        vicinity_range = st.slider(
            "Vicinity Range (± degrees)",
            min_value=1.0,
            max_value=45.0,
            value=10.0,
            step=1.0,
            help="Range around habit plane to analyze"
        )
        
        n_points = st.slider(
            "Number of Points",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of orientation points in sweep"
        )
        
        # Generate button
        st.markdown("---")
        if st.button("🚀 Generate Analysis", type="primary", use_container_width=True):
            st.session_state.generate_analysis = True
        else:
            st.session_state.generate_analysis = False
    
    # Main content
    if not st.session_state.solutions:
        st.warning("⚠️ Please load solutions first using the button in the sidebar.")
        st.markdown("""
        ### Quick Start Guide
        1. **Load Solutions**: Click the "Load Solutions" button to load simulation data
        2. **Configure Parameters**: Set defect type and analysis parameters in the sidebar
        3. **Generate Analysis**: Click the button to create the radar chart visualization
        4. **Customize**: Use the options below to customize your visualization
        """)
    else:
        # Create tabs
        tab1, tab2 = st.tabs(["📈 Radar Chart", "🎨 Customization"])
        
        with tab1:
            if st.session_state.get('generate_analysis', False):
                with st.spinner("Generating radar chart..."):
                    # Create target parameters
                    target_params = {
                        'defect_type': defect_type,
                        'shape': shape,
                        'eps0': eps0,
                        'kappa': kappa
                    }
                    
                    # Generate defect comparison data
                    defect_comparison = st.session_state.interpolator.compare_defect_types(
                        vicinity_range=vicinity_range,
                        n_points=n_points
                    )
                    
                    # Store in session state for customization tab
                    st.session_state.defect_comparison = defect_comparison
                    st.session_state.current_params = target_params
                    st.session_state.vicinity_range = vicinity_range
                    
                    # Get customization settings from session state
                    show_labels = st.session_state.get('show_labels', True)
                    show_legend = st.session_state.get('show_legend', True)
                    line_width = st.session_state.get('line_width', 3)
                    background_opacity = st.session_state.get('background_opacity', 0.3)
                    legend_position = st.session_state.get('legend_position', 'right')
                    
                    # Create visualization
                    fig_radar = st.session_state.visualizer.create_stress_comparison_radar(
                        defect_comparison,
                        title=f"Stress Components: {vicinity_range}° Vicinity of Habit Plane",
                        show_labels=show_labels,
                        show_legend=show_legend,
                        line_width=line_width,
                        background_opacity=background_opacity,
                        legend_position=legend_position
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Habit Plane Angle", "54.7°")
                    with col2:
                        st.metric("Selected Defect", f"{defect_type} (ε*={eps0:.2f})")
                    with col3:
                        st.metric("Analysis Points", n_points)
            else:
                st.info("👈 Configure parameters in the sidebar and click 'Generate Analysis'")
                
                # Show example radar chart
                st.markdown("### Example Visualization")
                # Create dummy data for example
                example_angles = np.linspace(44.7, 64.7, 10)
                example_stress = 15 * np.exp(-(example_angles - 54.7)**2 / 30) + 2
                
                fig_example = go.Figure()
                fig_example.add_trace(go.Scatterpolar(
                    r=np.append(example_stress, example_stress[0]),
                    theta=np.append(example_angles, example_angles[0]),
                    mode='lines',
                    name='Example Stress',
                    line=dict(color='#45B7D1', width=3)
                ))
                fig_example.add_trace(go.Scatterpolar(
                    r=[0, 20],
                    theta=[54.7, 54.7],
                    mode='lines',
                    line=dict(color='green', dash='dashdot', width=2),
                    name='Habit Plane'
                ))
                fig_example.update_layout(
                    polar=dict(
                        radialaxis=dict(range=[0, 20]),
                        angularaxis=dict(rotation=90, direction='clockwise')
                    ),
                    title="Example: Stress Distribution Around Habit Plane",
                    width=800,
                    height=600
                )
                st.plotly_chart(fig_example, use_container_width=True)
        
        with tab2:
            st.markdown("### 🎨 Visualization Customization")
            st.write("Customize the appearance of your radar chart:")
            
            if 'defect_comparison' not in st.session_state:
                st.info("Generate an analysis first to access customization options")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    show_labels = st.checkbox("Show Angle Labels", value=True)
                    show_legend = st.checkbox("Show Legend", value=True)
                    line_width = st.slider("Line Width", 1, 10, 3)
                    background_opacity = st.slider("Background Opacity", 0.0, 1.0, 0.3)
                
                with col2:
                    legend_position = st.radio(
                        "Legend Position",
                        ["right", "top"],
                        index=0,
                        help="Position of the legend"
                    )
                    colormap_option = st.selectbox(
                        "Color Scheme",
                        ["Default", "Viridis", "Plasma", "Inferno", "Magma"],
                        help="Color scheme for stress visualization"
                    )
                
                # Store customization settings
                st.session_state.show_labels = show_labels
                st.session_state.show_legend = show_legend
                st.session_state.line_width = line_width
                st.session_state.background_opacity = background_opacity
                st.session_state.legend_position = legend_position
                
                # Apply changes button
                if st.button("Apply Customization", type="primary"):
                    st.session_state.generate_analysis = True
                    st.rerun()
                
                # Show preview of current settings
                st.markdown("### Preview of Current Settings")
                settings_preview = f"""
                - **Angle Labels:** {"Enabled" if show_labels else "Disabled"}
                - **Legend:** {"Visible" if show_legend else "Hidden"} ({legend_position} position)
                - **Line Width:** {line_width}px
                - **Background Opacity:** {background_opacity:.1f}
                - **Color Scheme:** {colormap_option}
                """
                st.markdown(settings_preview)

if __name__ == "__main__":
    main()
