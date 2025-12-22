import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# Color schemes
STRESS_CMAP = LinearSegmentedColormap.from_list(
    'stress_cmap', ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
)
SUNBURST_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'rainbow']

# =============================================
# DATA LOADER
# =============================================
class SolutionLoader:
    """Loads simulation files from numerical_solutions directory"""
    
    @staticmethod
    def load_all_solutions():
        """Load all simulation files from directory"""
        solutions = []
        
        if not os.path.exists(SOLUTIONS_DIR):
            st.warning(f"Directory {SOLUTIONS_DIR} not found. Creating it.")
            os.makedirs(SOLUTIONS_DIR, exist_ok=True)
            return solutions
        
        for fname in os.listdir(SOLUTIONS_DIR):
            if fname.endswith(('.pkl', '.pt', '.pickle')):
                path = os.path.join(SOLUTIONS_DIR, fname)
                try:
                    if fname.endswith('.pt'):
                        sim = torch.load(path, map_location='cpu')
                    else:
                        with open(path, 'rb') as f:
                            sim = pickle.load(f)
                    
                    # Standardize structure
                    if 'params' in sim and 'history' in sim:
                        sim['filename'] = fname
                        solutions.append(sim)
                        st.success(f"✅ Loaded: {fname}")
                    else:
                        st.warning(f"⚠️ Skipped {fname}: Missing params/history")
                        
                except Exception as e:
                    st.error(f"❌ Error loading {fname}: {str(e)}")
        
        return solutions

# =============================================
# ATTENTION INTERPOLATOR
# =============================================
class AttentionInterpolator(nn.Module):
    """Simple attention-based interpolator for stress fields"""
    
    def __init__(self, sigma=0.3):
        super().__init__()
        self.sigma = sigma
    
    def compute_parameter_vector(self, params):
        """Convert parameters to numerical vector"""
        defect_map = {'ISF': [1,0,0], 'ESF': [0,1,0], 'Twin': [0,0,1]}
        shape_map = {'Square': [1,0,0,0,0], 'Horizontal Fault': [0,1,0,0,0], 
                    'Vertical Fault': [0,0,1,0,0], 'Rectangle': [0,0,0,1,0], 
                    'Ellipse': [0,0,0,0,1]}
        
        vector = []
        
        # Defect type
        defect = params.get('defect_type', 'ISF')
        vector.extend(defect_map.get(defect, [0,0,0]))
        
        # Shape
        shape = params.get('shape', 'Square')
        vector.extend(shape_map.get(shape, [0,0,0,0,0]))
        
        # Numeric parameters (normalized)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        vector.append((eps0 - 0.3) / (3.0 - 0.3))  # eps0 normalized
        vector.append((kappa - 0.1) / (2.0 - 0.1))  # kappa normalized
        vector.append(theta / (np.pi / 2))  # theta normalized 0-pi/2
        
        return np.array(vector, dtype=np.float32)
    
    def interpolate(self, sources, target_params):
        """Interpolate stress field using attention weights"""
        
        # Get source parameter vectors
        source_vectors = []
        source_stresses = []
        
        for src in sources:
            src_vec = self.compute_parameter_vector(src['params'])
            source_vectors.append(src_vec)
            
            # Get stress from final frame
            if src['history']:
                _, stress_fields = src['history'][-1]
                source_stresses.append({
                    'von_mises': stress_fields.get('von_mises', np.zeros((128, 128))),
                    'sigma_hydro': stress_fields.get('sigma_hydro', np.zeros((128, 128))),
                    'sigma_mag': stress_fields.get('sigma_mag', np.zeros((128, 128)))
                })
        
        if not source_vectors:
            return None
        
        source_vectors = np.array(source_vectors)
        target_vector = self.compute_parameter_vector(target_params)
        
        # Compute attention weights (Gaussian similarity)
        distances = np.sqrt(np.sum((source_vectors - target_vector) ** 2, axis=1))
        weights = np.exp(-0.5 * (distances / self.sigma) ** 2)
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Weighted combination
        result = {}
        for key in ['von_mises', 'sigma_hydro', 'sigma_mag']:
            combined = np.zeros_like(source_stresses[0][key])
            for w, stress in zip(weights, source_stresses):
                combined += w * stress[key]
            result[key] = combined
        
        return {
            'stress_fields': result,
            'attention_weights': weights,
            'target_params': target_params
        }

# =============================================
# SUNBURST & RADAR VISUALIZER
# =============================================
class SunburstRadarVisualizer:
    """Creates sunburst and radar charts for stress visualization"""
    
    @staticmethod
    def create_sunburst_plot(stress_matrix, times, thetas, title, cmap='plasma'):
        """Create polar heatmap (sunburst) visualization"""
        
        # Create polar plot
        theta_deg = np.deg2rad(thetas)
        theta_mesh, time_mesh = np.meshgrid(theta_deg, times)
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        # Plot heatmap
        im = ax.pcolormesh(theta_mesh, time_mesh, stress_matrix, 
                          cmap=cmap, shading='auto')
        
        # Customize
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Orientation (degrees)', labelpad=20)
        ax.set_xticks(theta_deg)
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.1)
        cbar.set_label('Stress (GPa)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_plotly_sunburst(stress_matrix, times, thetas, title, cmap='Plasma'):
        """Interactive sunburst with Plotly"""
        
        # Prepare data for polar scatter
        theta_deg = np.deg2rad(thetas)
        theta_grid, time_grid = np.meshgrid(theta_deg, times)
        
        fig = go.Figure(data=go.Scatterpolar(
            r=time_grid.flatten(),
            theta=np.rad2deg(theta_grid).flatten(),
            mode='markers',
            marker=dict(
                size=8,
                color=stress_matrix.flatten(),
                colorscale=cmap,
                showscale=True,
                colorbar=dict(title="Stress (GPa)")
            ),
            hovertemplate='Time: %{r:.1f}s<br>' +
                         'Orientation: %{theta:.1f}°<br>' +
                         'Stress: %{marker.color:.3f} GPa<br>'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            polar=dict(
                radialaxis=dict(title="Time (s)", gridcolor="lightgray"),
                angularaxis=dict(gridcolor="lightgray", rotation=90)
            ),
            height=600,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_radar_plot(stress_values, thetas, component_name, time_point):
        """Create radar/spider chart"""
        
        # Close the loop
        angles = np.linspace(0, 2*np.pi, len(thetas), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        values = np.concatenate([stress_values, [stress_values[0]]])
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, markersize=6)
        ax.fill(angles, values, alpha=0.25)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'{t:.0f}°' for t in thetas])
        ax.set_title(f'{component_name} at t={time_point:.1f}s', fontsize=14, pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return fig

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(page_title="Stress Interpolation Visualizer", layout="wide")
    
    st.title("🔬 Stress Field Interpolation with Sunburst & Radar Charts")
    st.markdown("""
    This app loads simulation data, performs attention-based interpolation, 
    and visualizes results as sunburst and radar charts.
    """)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = AttentionInterpolator(sigma=0.3)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = SunburstRadarVisualizer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Load solutions
        if st.button("📂 Load Solutions from Directory", use_container_width=True):
            with st.spinner("Loading solutions..."):
                st.session_state.solutions = SolutionLoader.load_all_solutions()
        
        if st.session_state.solutions:
            st.success(f"Loaded {len(st.session_state.solutions)} solutions")
            
            # Show solution info
            with st.expander("📋 Solution Details"):
                for i, sol in enumerate(st.session_state.solutions):
                    params = sol['params']
                    st.write(f"**{i+1}.** {params.get('defect_type', 'Unknown')} - "
                            f"θ={np.rad2deg(params.get('theta', 0)):.1f}° - "
                            f"ε*={params.get('eps0', 0):.3f}")
        
        st.divider()
        
        # Interpolation settings
        st.subheader("🎯 Interpolation Target")
        
        defect_type = st.selectbox("Defect Type", ["ISF", "ESF", "Twin"], index=0)
        
        # Orientation sweep settings
        st.subheader("🌐 Orientation Sweep")
        theta_min = st.slider("Min Angle (°)", 0, 90, 0, 5)
        theta_max = st.slider("Max Angle (°)", 0, 90, 90, 5)
        theta_step = st.slider("Step (°)", 5, 45, 15, 5)
        
        # Material parameters
        eps0 = st.slider("ε* (Strain)", 0.3, 3.0, 0.707, 0.1)
        kappa = st.slider("κ (Shape)", 0.1, 2.0, 0.6, 0.1)
        
        # Visualization settings
        st.subheader("🎨 Visualization")
        viz_type = st.radio("Chart Type", ["Sunburst", "Radar", "Both"])
        cmap = st.selectbox("Color Map", SUNBURST_CMAPS, index=1)
        use_plotly = st.checkbox("Use Interactive Plotly", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🚀 Generate Visualizations")
        
        if not st.session_state.solutions:
            st.warning("Please load solutions first using the button in the sidebar.")
            st.info("""
            **Expected file structure:**
            - Place `.pkl` or `.pt` files in `numerical_solutions/` directory
            - Each file should contain simulation data with:
              - `params` dictionary (defect_type, theta, eps0, kappa, shape)
              - `history` list with stress fields
            """)
        else:
            if st.button("✨ Generate Sunburst & Radar Charts", type="primary", use_container_width=True):
                with st.spinner("Generating orientation sweep..."):
                    try:
                        # Generate theta range
                        thetas = np.arange(theta_min, theta_max + theta_step, theta_step)
                        theta_rad = np.deg2rad(thetas)
                        
                        # Generate time points (simulated)
                        n_times = 50
                        times = np.linspace(0, 200, n_times)
                        
                        # Generate predictions for each orientation
                        predictions = []
                        
                        progress_bar = st.progress(0)
                        for i, theta in enumerate(theta_rad):
                            # Target parameters
                            target_params = {
                                'defect_type': defect_type,
                                'theta': float(theta),
                                'eps0': eps0,
                                'kappa': kappa,
                                'shape': 'Square'
                            }
                            
                            # Interpolate
                            result = st.session_state.interpolator.interpolate(
                                st.session_state.solutions, target_params
                            )
                            
                            if result:
                                # Extract stress evolution at center point
                                center_i, center_j = 64, 64  # Assuming 128x128 grid
                                time_evolution = []
                                
                                # Create synthetic time evolution
                                for t in times:
                                    # Simulate stress build-up over time
                                    base_stress = result['stress_fields']['von_mises'][center_i, center_j]
                                    stress_at_t = base_stress * (1 - np.exp(-t / 50))
                                    time_evolution.append(stress_at_t)
                                
                                predictions.append(time_evolution)
                            
                            progress_bar.progress((i + 1) / len(theta_rad))
                        
                        progress_bar.empty()
                        
                        # Create stress matrix (time x theta)
                        if predictions:
                            stress_matrix = np.array(predictions).T  # Shape: (n_times, n_thetas)
                            
                            # Store for visualization
                            st.session_state.stress_matrix = stress_matrix
                            st.session_state.times = times
                            st.session_state.thetas = thetas
                            
                            st.success(f"✅ Generated {len(thetas)} orientations × {len(times)} time points")
                            
                            # Display results
                            if viz_type in ["Sunburst", "Both"]:
                                st.subheader("🌅 Sunburst Chart")
                                
                                if use_plotly:
                                    fig = st.session_state.visualizer.create_plotly_sunburst(
                                        stress_matrix, times, thetas,
                                        title=f"Von Mises Stress - {defect_type}",
                                        cmap=cmap
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    fig = st.session_state.visualizer.create_sunburst_plot(
                                        stress_matrix, times, thetas,
                                        title=f"Von Mises Stress - {defect_type}",
                                        cmap=cmap
                                    )
                                    st.pyplot(fig)
                                    
                                    # Download button
                                    buf = BytesIO()
                                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                                    st.download_button(
                                        "📥 Download Sunburst PNG",
                                        data=buf.getvalue(),
                                        file_name=f"sunburst_{defect_type}.png",
                                        mime="image/png"
                                    )
                            
                            if viz_type in ["Radar", "Both"]:
                                st.subheader("📡 Radar Charts")
                                
                                # Select time point
                                time_idx = st.slider("Select Time Point", 0, len(times)-1, len(times)//2)
                                selected_time = times[time_idx]
                                
                                # Create radar for each stress component
                                cols = st.columns(3)
                                component_names = ['Von Mises', 'Hydrostatic', 'Magnitude']
                                
                                for idx, (col, name) in enumerate(zip(cols, component_names)):
                                    with col:
                                        # For demonstration, use von_mises data
                                        radar_values = stress_matrix[time_idx, :]
                                        
                                        fig_radar = st.session_state.visualizer.create_radar_plot(
                                            radar_values, thetas, name, selected_time
                                        )
                                        st.pyplot(fig_radar)
                            
                            # Statistics
                            st.subheader("📊 Statistics")
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            
                            with col_stat1:
                                st.metric("Max Stress", f"{np.max(stress_matrix):.3f} GPa")
                            with col_stat2:
                                st.metric("Mean Stress", f"{np.mean(stress_matrix):.3f} GPa")
                            with col_stat3:
                                st.metric("Orientation Range", f"{theta_min}° to {theta_max}°")
                            
                            # Data export
                            st.subheader("📤 Export Data")
                            
                            # CSV export
                            if st.button("💾 Export as CSV"):
                                # Create DataFrame
                                export_data = []
                                for t_idx, time_val in enumerate(times):
                                    for theta_idx, theta_val in enumerate(thetas):
                                        export_data.append({
                                            'time_s': time_val,
                                            'orientation_deg': theta_val,
                                            'stress_gpa': stress_matrix[t_idx, theta_idx]
                                        })
                                
                                df = pd.DataFrame(export_data)
                                csv = df.to_csv(index=False)
                                
                                st.download_button(
                                    "📥 Download CSV",
                                    data=csv,
                                    file_name=f"stress_data_{defect_type}.csv",
                                    mime="text/csv"
                                )
                            
                            # JSON export
                            if st.button("📊 Export as JSON"):
                                export_dict = {
                                    'metadata': {
                                        'defect_type': defect_type,
                                        'theta_range': f"{theta_min}-{theta_max}°",
                                        'eps0': eps0,
                                        'kappa': kappa,
                                        'generated_at': datetime.now().isoformat()
                                    },
                                    'times': times.tolist(),
                                    'thetas': thetas.tolist(),
                                    'stress_matrix': stress_matrix.tolist()
                                }
                                
                                json_str = json.dumps(export_dict, indent=2)
                                st.download_button(
                                    "📥 Download JSON",
                                    data=json_str,
                                    file_name=f"stress_data_{defect_type}.json",
                                    mime="application/json"
                                )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        if 'stress_matrix' in st.session_state:
            stress_matrix = st.session_state.stress_matrix
            
            st.metric("Time Points", len(st.session_state.times))
            st.metric("Orientations", len(st.session_state.thetas))
            st.metric("Max Value", f"{np.max(stress_matrix):.3f} GPa")
            st.metric("Mean Value", f"{np.mean(stress_matrix):.3f} GPa")
            
            # Stress distribution
            st.subheader("📊 Distribution")
            
            fig_dist, ax_dist = plt.subplots(figsize=(3, 2))
            ax_dist.hist(stress_matrix.flatten(), bins=20, edgecolor='black', alpha=0.7)
            ax_dist.set_xlabel('Stress (GPa)')
            ax_dist.set_ylabel('Count')
            ax_dist.set_title('Stress Distribution')
            plt.tight_layout()
            st.pyplot(fig_dist)
        else:
            st.info("No data generated yet. Click 'Generate' to create visualizations.")

# =============================================
# RUN APPLICATION
# =============================================
if __name__ == "__main__":
    main()
