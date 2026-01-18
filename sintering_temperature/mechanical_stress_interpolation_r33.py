import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
import json
from datetime import datetime
from typing import List, Dict, Any
import torch

# Directory configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

class PhysicsBasedStressAnalyzer:
    """Simplified physics analyzer for eigen strain values"""
    def __init__(self):
        # Eigen strains for different defect types
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
    """Simplified solution loader"""
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        
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
        """Read simulation file"""
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
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

class PhysicsAwareInterpolator:
    """Simplified interpolator for stress components"""
    def __init__(self):
        self.physics_analyzer = PhysicsBasedStressAnalyzer()
        self.habit_plane_angle = 54.7
        
        # Parameter mappings
        self.defect_map = {
            'ISF': [1, 0, 0, 0, self.physics_analyzer.get_eigen_strain('ISF')],
            'ESF': [0, 1, 0, 0, self.physics_analyzer.get_eigen_strain('ESF')],
            'Twin': [0, 0, 1, 0, self.physics_analyzer.get_eigen_strain('Twin')],
            'No Defect': [0, 0, 0, 1, self.physics_analyzer.get_eigen_strain('No Defect')]
        }
    
    def create_vicinity_sweep(self, sources, target_params, vicinity_range=10.0,
                             n_points=50, region_type='bulk'):
        """Create stress sweep in vicinity of habit plane"""
        center_angle = self.habit_plane_angle
        min_angle = center_angle - vicinity_range
        max_angle = center_angle + vicinity_range
        angles = np.linspace(min_angle, max_angle, n_points)
        
        # Mock interpolation - in real code this would use sophisticated methods
        # Here we create synthetic data for demonstration
        defect_type = target_params.get('defect_type', 'Twin')
        eigen_strain = self.physics_analyzer.get_eigen_strain(defect_type)
        
        # Create synthetic stress data with peak at habit plane
        sigma_hydro = eigen_strain * 10 * np.exp(-(angles - center_angle)**2 / (2 * 3**2)) + 1.0
        von_mises = eigen_strain * 12 * np.exp(-(angles - center_angle)**2 / (2 * 4**2)) + 2.0
        sigma_mag = eigen_strain * 15 * np.exp(-(angles - center_angle)**2 / (2 * 2**2)) + 0.5
        
        # Create synthetic sintering temperatures
        T0 = 623.0  # Reference temperature in K
        sintering_exponential = T0 * np.exp(-0.05 * sigma_hydro)
        sintering_arrhenius = T0 * np.exp(-0.07 * sigma_hydro)
        
        results = {
            'angles': angles.tolist(),
            'stresses': {
                'sigma_hydro': sigma_hydro.tolist(),
                'von_mises': von_mises.tolist(),
                'sigma_mag': sigma_mag.tolist()
            },
            'sintering_temps': {
                'exponential': sintering_exponential.tolist(),
                'arrhenius_defect': sintering_arrhenius.tolist()
            },
            'defect_type': defect_type,
            'eigen_strain': eigen_strain
        }
        return results

class HabitPlaneVisualizer:
    """Specialized visualizer for habit plane vicinity analysis"""
    def __init__(self, habit_angle=54.7):
        self.habit_angle = habit_angle
        # Color schemes
        self.stress_colors = {
            'sigma_hydro': 'rgb(31, 119, 180)',
            'von_mises': 'rgb(255, 127, 14)',
            'sigma_mag': 'rgb(44, 160, 44)'
        }
        
    def create_stress_comparison_radar(self, angles, stresses, custom_settings=None):
        """
        Create radar chart for stress components in habit plane vicinity
        with customizable visualization settings
        """
        # Default settings that can be customized
        default_settings = {
            'line_colors': {
                'sigma_hydro': 'rgb(31, 119, 180)',
                'von_mises': 'rgb(255, 127, 14)',
                'sigma_mag': 'rgb(44, 160, 44)'
            },
            'line_width': 3,
            'line_styles': {
                'sigma_hydro': 'solid',
                'von_mises': 'solid',
                'sigma_mag': 'solid'
            },
            'show_labels': True,
            'show_markers': True,
            'marker_size': 6,
            'show_legend': True,
            'legend_position': {'x': 1.1, 'y': 0.5},
            'show_habit_plane': True,
            'habit_plane_color': 'rgb(46, 204, 113)',
            'habit_plane_dash': 'dashdot',
            'habit_plane_width': 4,
            'fill_area': False,
            'fill_opacity': 0.1,
            'bgcolor': 'rgba(240, 240, 240, 0.3)',
            'radial_range': None,
            'angular_range': [0, 360],
            'title': 'Stress Components in Habit Plane Vicinity',
            'title_font_size': 20,
            'title_font_color': 'darkblue',
            'radial_axis_title': 'Stress (GPa)',
            'radial_axis_color': 'black',
            'angular_axis_color': 'black',
            'grid_color': 'rgba(100, 100, 100, 0.3)',
            'figure_width': 900,
            'figure_height': 700
        }
        
        # Update defaults with custom settings if provided
        if custom_settings:
            for key, value in custom_settings.items():
                if key in default_settings:
                    if isinstance(value, dict) and isinstance(default_settings[key], dict):
                        default_settings[key].update(value)
                    else:
                        default_settings[key] = value
        
        settings = default_settings
        
        # Convert inputs to numpy arrays for processing
        angles = np.array(angles)
        if len(angles) == 0:
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="No Data Available",
                    font=dict(size=20, family="Arial Black", color='darkblue'),
                    x=0.5
                ),
                annotations=[dict(
                    text="No data available for the selected vicinity range",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=14)
                )],
                width=settings['figure_width'],
                height=settings['figure_height']
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Determine radial range if not specified
        max_stress = 0
        if settings['radial_range'] is None:
            for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
                if comp in stresses:
                    max_val = max(stresses[comp])
                    if max_val > max_stress:
                        max_stress = max_val
            radial_max = max_stress * 1.2 if max_stress > 0 else 1.0
        else:
            radial_max = settings['radial_range'][1]
        
        # Add traces for each stress component
        for comp in ['sigma_hydro', 'von_mises', 'sigma_mag']:
            if comp in stresses:
                # Convert to arrays and close the loop for radar chart
                stress_vals = np.array(stresses[comp])
                if len(stress_vals) != len(angles):
                    continue
                    
                angles_closed = np.append(angles, angles[0])
                stresses_closed = np.append(stress_vals, stress_vals[0])
                
                # Create line style dictionary
                line_dict = {
                    'color': settings['line_colors'].get(comp, 'black'),
                    'width': settings['line_width'],
                    'dash': settings['line_styles'].get(comp, 'solid')
                }
                
                # Create marker dictionary if enabled
                marker_dict = None
                if settings['show_markers']:
                    marker_dict = {
                        'size': settings['marker_size'],
                        'color': settings['line_colors'].get(comp, 'black')
                    }
                
                # Add trace
                trace = go.Scatterpolar(
                    r=stresses_closed,
                    theta=angles_closed,
                    mode='lines' + ('+markers' if settings['show_markers'] else ''),
                    line=line_dict,
                    marker=marker_dict,
                    name=comp.replace('_', ' ').title(),
                    hovertemplate='Orientation: %{theta:.2f}°<br>Stress: %{r:.4f} GPa<extra></extra>',
                    showlegend=settings['show_legend']
                )
                
                # Add fill if enabled
                if settings['fill_area']:
                    trace.fill = 'toself'
                    trace.fillcolor = f"rgba{tuple(int(settings['line_colors'].get(comp, '#000000')[i:i+2], 16) for i in (1, 3, 5)) + (settings['fill_opacity'],)}"
                
                fig.add_trace(trace)
        
        # Highlight habit plane if enabled
        if settings['show_habit_plane']:
            fig.add_trace(go.Scatterpolar(
                r=[0, radial_max],
                theta=[self.habit_angle, self.habit_angle],
                mode='lines',
                line=dict(
                    color=settings['habit_plane_color'],
                    width=settings['habit_plane_width'],
                    dash=settings['habit_plane_dash']
                ),
                name=f'Habit Plane ({self.habit_angle}°)',
                hoverinfo='skip',
                showlegend=settings['show_legend']
            ))
        
        # Update layout with custom settings
        fig.update_layout(
            title=dict(
                text=settings['title'],
                font=dict(
                    size=settings['title_font_size'],
                    family="Arial Black",
                    color=settings['title_font_color']
                ),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor=settings['grid_color'],
                    gridwidth=2,
                    linecolor=settings['radial_axis_color'],
                    linewidth=3,
                    tickfont=dict(size=12, color='black'),
                    title=dict(
                        text=settings['radial_axis_title'],
                        font=dict(size=14, color=settings['radial_axis_color'])
                    ),
                    range=[0, radial_max]
                ),
                angularaxis=dict(
                    gridcolor=settings['grid_color'],
                    gridwidth=2,
                    linecolor=settings['angular_axis_color'],
                    linewidth=3,
                    rotation=90,
                    direction="clockwise",
                    tickmode='array',
                    tickvals=list(range(int(min(angles)), int(max(angles)) + 1, 15)),
                    ticktext=[f'{i}°' for i in range(int(min(angles)), int(max(angles)) + 1, 15)],
                    tickfont=dict(size=12, color='black'),
                    period=360,
                    thetaunit="degrees"
                ),
                bgcolor=settings['bgcolor'],
                sector=[min(angles), max(angles)]
            ),
            showlegend=settings['show_legend'],
            legend=dict(
                x=settings['legend_position']['x'],
                y=settings['legend_position']['y'],
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12, family='Arial')
            ),
            width=settings['figure_width'],
            height=settings['figure_height'],
            margin=dict(l=100, r=200, t=100, b=100)
        )
        
        return fig

def load_data_and_create_radar(vicinity_range=10.0, n_points=100):
    """Load data and create radar chart for habit plane vicinity"""
    # Initialize components
    loader = EnhancedSolutionLoader()
    interpolator = PhysicsAwareInterpolator()
    visualizer = HabitPlaneVisualizer()
    
    # Load solutions
    solutions = []
    file_info = loader.scan_solutions()
    
    # For demo purposes, we'll use mock data if no solutions are found
    if not file_info:
        st.warning("No solution files found. Using demo data for visualization.")
    
    # Create vicinity sweep for a specific defect type
    target_params = {
        'defect_type': 'Twin',
        'shape': 'Square',
        'eps0': 2.12,
        'kappa': 0.6
    }
    
    # Create synthetic vicinity sweep
    vicinity_sweep = interpolator.create_vicinity_sweep(
        solutions,
        target_params,
        vicinity_range=vicinity_range,
        n_points=n_points,
        region_type='bulk'
    )
    
    return vicinity_sweep, visualizer

def create_radar_chart(vicinity_sweep, visualizer, custom_settings=None):
    """Create radar chart from vicinity sweep data"""
    if not vicinity_sweep:
        return None
    
    # Extract data
    angles = vicinity_sweep['angles']
    stresses = vicinity_sweep['stresses']
    
    # Create radar chart
    fig = visualizer.create_stress_comparison_radar(angles, stresses, custom_settings)
    return fig

# Streamlit app
def main():
    st.set_page_config(
        page_title="Habit Plane Radar Chart Visualization",
        layout="wide",
        page_icon="🔬"
    )
    
    st.title("🔬 Ag FCC Twin: Habit Plane Vicinity Radar Chart")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data loading options
        st.subheader("Data Options")
        vicinity_range = st.slider("Vicinity Range (± degrees)", 1.0, 45.0, 10.0, 1.0)
        n_points = st.slider("Number of Points", 10, 200, 50, 10)
        
        # Trigger data loading
        if st.button("Load Data & Generate Chart", type="primary"):
            with st.spinner("Loading data and generating visualization..."):
                st.session_state.vicinity_sweep, st.session_state.visualizer = load_data_and_create_radar(
                    vicinity_range, n_points
                )
                st.success("Data loaded and processed successfully!")
        
        # Customization options
        st.subheader("Chart Customization")
        
        # Line customization
        st.markdown("### Line Properties")
        line_width = st.slider("Line Width", 1, 10, 3)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma_hydro_color = st.color_picker("Hydrostatic Stress", "#1f77b4")
            sigma_hydro_style = st.selectbox("Style", ["solid", "dash", "dot", "dashdot"], key="hyd_style")
        with col2:
            von_mises_color = st.color_picker("Von Mises Stress", "#ff7f0e")
            von_mises_style = st.selectbox("Style", ["solid", "dash", "dot", "dashdot"], key="vm_style")
        with col3:
            sigma_mag_color = st.color_picker("Stress Magnitude", "#2ca02c")
            sigma_mag_style = st.selectbox("Style", ["solid", "dash", "dot", "dashdot"], key="mag_style")
        
        # Display options
        st.markdown("### Display Options")
        show_markers = st.checkbox("Show Markers", True)
        marker_size = st.slider("Marker Size", 2, 15, 6) if show_markers else 6
        show_labels = st.checkbox("Show Labels", True)
        show_legend = st.checkbox("Show Legend", True)
        show_habit_plane = st.checkbox("Show Habit Plane", True)
        fill_area = st.checkbox("Fill Area Under Curves", False)
        fill_opacity = st.slider("Fill Opacity", 0.0, 1.0, 0.1) if fill_area else 0.1
        
        # Axis and layout options
        st.markdown("### Axis & Layout")
        radial_max = st.number_input("Max Stress (GPa)", 1.0, 100.0, 30.0)
        title = st.text_input("Chart Title", "Stress Components in Habit Plane Vicinity")
        figure_width = st.slider("Figure Width", 600, 1500, 900)
        figure_height = st.slider("Figure Height", 500, 1000, 700)
        
        # Generate custom settings dictionary
        custom_settings = {
            'line_colors': {
                'sigma_hydro': sigma_hydro_color,
                'von_mises': von_mises_color,
                'sigma_mag': sigma_mag_color
            },
            'line_width': line_width,
            'line_styles': {
                'sigma_hydro': sigma_hydro_style,
                'von_mises': von_mises_style,
                'sigma_mag': sigma_mag_style
            },
            'show_labels': show_labels,
            'show_markers': show_markers,
            'marker_size': marker_size,
            'show_legend': show_legend,
            'legend_position': {'x': 1.1, 'y': 0.5},
            'show_habit_plane': show_habit_plane,
            'habit_plane_color': 'rgb(46, 204, 113)',
            'habit_plane_dash': 'dashdot',
            'habit_plane_width': 4,
            'fill_area': fill_area,
            'fill_opacity': fill_opacity,
            'bgcolor': 'rgba(240, 240, 240, 0.3)',
            'radial_range': [0, radial_max],
            'title': title,
            'figure_width': figure_width,
            'figure_height': figure_height
        }
    
    # Main content area
    if 'vicinity_sweep' not in st.session_state:
        st.info("👈 Configure options in the sidebar and click 'Load Data & Generate Chart' to start")
        st.markdown("""
        ### What this visualization shows:
        
        This radar chart displays stress components in the vicinity of the Ag FCC twin habit plane (54.7°):
        
        - **Hydrostatic Stress (σ_hydro)**: Trace of stress tensor/3, critical for sintering
        - **Von Mises Stress (σ_vm)**: Equivalent tensile stress, indicates yield onset
        - **Stress Magnitude (σ_mag)**: Overall stress intensity
        
        The habit plane (54.7°) is highlighted, showing where maximum stress concentration typically occurs.
        """)
    else:
        # Create and display radar chart
        fig = create_radar_chart(st.session_state.vicinity_sweep, 
                               st.session_state.visualizer, 
                               custom_settings)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export as HTML
                html_str = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="Download HTML",
                    data=html_str,
                    file_name=f"radar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            
            with col2:
                # Export as PNG
                st.download_button(
                    label="Download PNG",
                    data=fig.to_image(format="png"),
                    file_name=f"radar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            
            with col3:
                # Export data as CSV
                angles = st.session_state.vicinity_sweep['angles']
                stresses = st.session_state.vicinity_sweep['stresses']
                
                df = pd.DataFrame({
                    'Angle_deg': angles,
                    'Sigma_hydro_GPa': stresses['sigma_hydro'],
                    'Von_Mises_GPa': stresses['von_mises'],
                    'Sigma_Mag_GPa': stresses['sigma_mag']
                })
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Data (CSV)",
                    data=csv,
                    file_name=f"radar_chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
