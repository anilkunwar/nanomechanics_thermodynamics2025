import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import warnings
import pickle
import torch
import sqlite3
from io import StringIO
import traceback
import h5py
import msgpack
import dill
import joblib
from pathlib import Path
import tempfile
import base64
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")

if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)

# =============================================
# STRESS ANALYSIS MANAGER
# =============================================
class StressAnalysisManager:
    """Manager for stress value analysis and visualization"""
    
    @staticmethod
    def compute_max_stress_values(stress_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute maximum stress values from stress fields
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            
        Returns:
            Dictionary with max stress values
        """
        results = {}
        
        # Hydrostatic stress - take absolute maximum
        if 'sigma_hydro' in stress_fields:
            hydro_data = stress_fields['sigma_hydro']
            results['max_abs_hydrostatic'] = float(np.max(np.abs(hydro_data)))
            results['max_hydrostatic'] = float(np.max(hydro_data))
            results['min_hydrostatic'] = float(np.min(hydro_data))
            results['mean_abs_hydrostatic'] = float(np.mean(np.abs(hydro_data)))
        
        # Stress magnitude - take maximum
        if 'sigma_mag' in stress_fields:
            mag_data = stress_fields['sigma_mag']
            results['max_stress_magnitude'] = float(np.max(mag_data))
            results['mean_stress_magnitude'] = float(np.mean(mag_data))
        
        # Von Mises stress - take maximum
        if 'von_mises' in stress_fields:
            vm_data = stress_fields['von_mises']
            results['max_von_mises'] = float(np.max(vm_data))
            results['mean_von_mises'] = float(np.mean(vm_data))
            results['min_von_mises'] = float(np.min(vm_data))
        
        # Principal stresses if available
        if 'sigma_1' in stress_fields and 'sigma_2' in stress_fields and 'sigma_3' in stress_fields:
            sigma1 = stress_fields['sigma_1']
            sigma2 = stress_fields['sigma_2']
            sigma3 = stress_fields['sigma_3']
            
            results['max_principal_1'] = float(np.max(sigma1))
            results['max_principal_2'] = float(np.max(sigma2))
            results['max_principal_3'] = float(np.max(sigma3))
            results['max_principal_abs'] = float(np.max(np.abs(sigma1)))
            
            # Maximum shear stress (Tresca)
            max_shear = 0.5 * np.max(np.abs(sigma1 - sigma3))
            results['max_shear_tresca'] = float(max_shear)
        
        # Additional statistical measures
        if 'sigma_hydro' in stress_fields:
            hydro_data = stress_fields['sigma_hydro']
            results['hydro_std'] = float(np.std(hydro_data))
            results['hydro_skewness'] = float(stats.skew(hydro_data.flatten()))
            results['hydro_kurtosis'] = float(stats.kurtosis(hydro_data.flatten()))
        
        if 'von_mises' in stress_fields:
            vm_data = stress_fields['von_mises']
            # Percentiles
            results['von_mises_p95'] = float(np.percentile(vm_data, 95))
            results['von_mises_p99'] = float(np.percentile(vm_data, 99))
            results['von_mises_p99_9'] = float(np.percentile(vm_data, 99.9))
        
        return results
    
    @staticmethod
    def extract_stress_peaks(stress_fields: Dict[str, np.ndarray], 
                           threshold_percentile: float = 95) -> Dict[str, Dict]:
        """
        Extract stress peak locations and values
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            threshold_percentile: Percentile for peak detection
            
        Returns:
            Dictionary with peak information for each stress component
        """
        peaks = {}
        
        for component_name, stress_data in stress_fields.items():
            if not isinstance(stress_data, np.ndarray):
                continue
                
            # Calculate threshold based on percentile
            threshold = np.percentile(stress_data, threshold_percentile)
            
            # Find peaks (indices where value > threshold)
            peak_indices = np.where(stress_data > threshold)
            
            if len(peak_indices[0]) > 0:
                # Get peak values
                peak_values = stress_data[peak_indices]
                
                # Find global maximum
                max_idx = np.argmax(peak_values)
                max_pos = (peak_indices[0][max_idx], peak_indices[1][max_idx])
                
                peaks[component_name] = {
                    'num_peaks': len(peak_values),
                    'max_value': float(np.max(peak_values)),
                    'max_position': max_pos,
                    'mean_peak_value': float(np.mean(peak_values)),
                    'peak_indices': peak_indices,
                    'peak_values': peak_values,
                    'threshold': float(threshold)
                }
        
        return peaks
    
    @staticmethod
    def compute_stress_gradients(stress_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute stress gradients for each stress component
        
        Args:
            stress_fields: Dictionary containing stress component arrays
            
        Returns:
            Dictionary with gradient magnitudes
        """
        gradients = {}
        
        for component_name, stress_data in stress_fields.items():
            if not isinstance(stress_data, np.ndarray):
                continue
            
            # Compute gradients using numpy's gradient
            grad_y, grad_x = np.gradient(stress_data)
            
            # Compute gradient magnitude
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            gradients[f'{component_name}_grad_x'] = grad_x
            gradients[f'{component_name}_grad_y'] = grad_y
            gradients[f'{component_name}_grad_mag'] = grad_magnitude
            
            # Statistics
            gradients[f'{component_name}_max_grad'] = float(np.max(grad_magnitude))
            gradients[f'{component_name}_mean_grad'] = float(np.mean(grad_magnitude))
        
        return gradients
    
    @staticmethod
    def create_stress_summary_dataframe(source_simulations: List[Dict], 
                                      predictions: Dict) -> pd.DataFrame:
        """
        Create comprehensive stress summary DataFrame
        
        Args:
            source_simulations: List of source simulation data
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with stress summary for all simulations and predictions
        """
        summary_rows = []
        
        # Process source simulations
        for i, sim_data in enumerate(source_simulations):
            params = sim_data.get('params', {})
            history = sim_data.get('history', [])
            
            if history:
                # Use final frame
                eta, stress_fields = history[-1]
                
                # Compute max stress values
                max_stress = StressAnalysisManager.compute_max_stress_values(stress_fields)
                
                # Create row
                row = {
                    'id': f'source_{i}',
                    'type': 'source',
                    'defect_type': params.get('defect_type', 'Unknown'),
                    'shape': params.get('shape', 'Unknown'),
                    'orientation': params.get('orientation', 'Unknown'),
                    'eps0': params.get('eps0', np.nan),
                    'kappa': params.get('kappa', np.nan),
                    'theta_deg': np.deg2rad(params.get('theta', 0)) if params.get('theta') else np.nan,
                    **max_stress
                }
                summary_rows.append(row)
        
        # Process predictions
        if predictions:
            for pred_key, pred_data in predictions.items():
                if isinstance(pred_data, dict) and 'target_params' in pred_data:
                    params = pred_data['target_params']
                    
                    # Compute max stress values
                    max_stress = StressAnalysisManager.compute_max_stress_values(pred_data)
                    
                    # Create row
                    row = {
                        'id': pred_key,
                        'type': 'prediction',
                        'defect_type': params.get('defect_type', 'Unknown'),
                        'shape': params.get('shape', 'Unknown'),
                        'orientation': params.get('orientation', 'Unknown'),
                        'eps0': params.get('eps0', np.nan),
                        'kappa': params.get('kappa', np.nan),
                        'theta_deg': np.deg2rad(params.get('theta', 0)) if params.get('theta') else np.nan,
                        **max_stress
                    }
                    summary_rows.append(row)
        
        # Create DataFrame
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            # Calculate additional metrics
            if 'max_von_mises' in df.columns and 'max_abs_hydrostatic' in df.columns:
                df['stress_ratio_vm_hydro'] = df['max_von_mises'] / (df['max_abs_hydrostatic'] + 1e-10)
            return df
        else:
            return pd.DataFrame()

# =============================================
# SUNBURST CHART MANAGER
# =============================================
class SunburstChartManager:
    """Manager for creating sunburst charts and other hierarchical visualizations"""
    
    @staticmethod
    def get_all_colormaps() -> List[str]:
        """Get list of all available matplotlib colormaps"""
        colormaps = sorted([m for m in plt.colormaps() if not m.endswith('_r')])
        return colormaps
    
    @staticmethod
    def create_sunburst_chart(df: pd.DataFrame, 
                            path_columns: List[str],
                            value_column: str,
                            title: str = "Sunburst Chart",
                            colormap: str = "viridis") -> go.Figure:
        """
        Create a sunburst chart using plotly
        
        Args:
            df: DataFrame containing the data
            path_columns: List of column names for hierarchical path
            value_column: Column name for values
            title: Chart title
            colormap: Matplotlib colormap name
            
        Returns:
            Plotly Figure object
        """
        # Prepare data for sunburst
        df_plot = df.copy()
        
        # Create path string
        df_plot['path'] = df_plot[path_columns].astype(str).agg(' / '.join, axis=1)
        
        # Create sunburst chart
        fig = px.sunburst(
            df_plot,
            path=path_columns,
            values=value_column,
            color=value_column,
            color_continuous_scale=colormap,
            title=title,
            hover_data={col: True for col in df.columns if col not in path_columns + [value_column]}
        )
        
        fig.update_traces(
            textinfo="label+percent entry",
            hovertemplate="<b>%{label}</b><br>" +
                         f"{value_column}: %{{value:.3f}}<br>" +
                         "%{parent}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            height=600,
            title_x=0.5
        )
        
        return fig
    
    @staticmethod
    def create_treemap_chart(df: pd.DataFrame,
                           path_columns: List[str],
                           value_column: str,
                           title: str = "Treemap Chart",
                           colormap: str = "viridis") -> go.Figure:
        """
        Create a treemap chart as alternative to sunburst
        
        Args:
            df: DataFrame containing the data
            path_columns: List of column names for hierarchical path
            value_column: Column name for values
            title: Chart title
            colormap: Matplotlib colormap name
            
        Returns:
            Plotly Figure object
        """
        fig = px.treemap(
            df,
            path=path_columns,
            values=value_column,
            color=value_column,
            color_continuous_scale=colormap,
            title=title,
            hover_data={col: True for col in df.columns if col not in path_columns + [value_column]}
        )
        
        fig.update_traces(
            textinfo="label+value+percent entry",
            hovertemplate="<b>%{label}</b><br>" +
                         f"{value_column}: %{{value:.3f}}<br>" +
                         "%{parent}<br>" +
                         "<extra></extra>"
        )
        
        fig.update_layout(
            margin=dict(t=50, l=0, r=0, b=0),
            height=600,
            title_x=0.5
        )
        
        return fig
    
    @staticmethod
    def create_parallel_categories(df: pd.DataFrame,
                                 dimensions: List[str],
                                 color_column: str,
                                 title: str = "Parallel Categories") -> go.Figure:
        """
        Create parallel categories diagram
        
        Args:
            df: DataFrame containing the data
            dimensions: List of categorical columns to include
            color_column: Numerical column for coloring
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = px.parallel_categories(
            df,
            dimensions=dimensions,
            color=df[color_column],
            color_continuous_scale='viridis',
            title=title,
            labels={col: col.replace('_', ' ').title() for col in dimensions}
        )
        
        fig.update_layout(
            height=500,
            title_x=0.5
        )
        
        return fig
    
    @staticmethod
    def create_radial_bar_chart(df: pd.DataFrame,
                              categories: List[str],
                              values: List[str],
                              title: str = "Radial Bar Chart") -> go.Figure:
        """
        Create radial bar chart for comparison
        
        Args:
            df: DataFrame containing the data
            categories: Column name for categories
            values: List of value columns to plot
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        for i, value_col in enumerate(values):
            fig.add_trace(go.Barpolar(
                r=df[value_col].values,
                theta=df[categories].values,
                name=value_col,
                marker_color=px.colors.sequential.Viridis[i/len(values)],
                opacity=0.8
            ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df[values].max().max() * 1.1]
                ),
                angularaxis=dict(
                    direction="clockwise",
                    period=len(df)
                )
            ),
            showlegend=True,
            height=500
        )
        
        return fig

# =============================================
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator:
    """Enhanced attention-based interpolator with spatial locality regularization"""
    
    def __init__(self, input_dim=15, num_heads=4, d_model=32, output_dim=3, 
                 sigma_spatial=0.2, sigma_param=0.2, use_gaussian=True):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.output_dim = output_dim
        self.sigma_spatial = sigma_spatial
        self.sigma_param = sigma_param
        self.use_gaussian = use_gaussian
        
        self.model = self._build_model()
        
        # Initialize stress analysis manager
        self.stress_analyzer = StressAnalysisManager()
        self.sunburst_manager = SunburstChartManager()
        
        self.readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
    
    def _build_model(self):
        # ... (keep existing model building code) ...
        model = torch.nn.ModuleDict({
            'param_embedding': torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.d_model)
            ),
            'attention': torch.nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1
            ),
            'feed_forward': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.d_model * 4, self.d_model)
            ),
            'output_projection': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.output_dim)
            ),
            'spatial_regularizer': torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, self.num_heads)
            ) if self.use_gaussian else None,
            'norm1': torch.nn.LayerNorm(self.d_model),
            'norm2': torch.nn.LayerNorm(self.d_model)
        })
        return model
    
    # ... (keep existing reader methods) ...

# =============================================
# ENHANCED ATTENTION INTERFACE WITH STRESS ANALYSIS
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with enhanced stress analysis"""
    
    st.header("🤖 Spatial-Attention Stress Interpolation with Analysis")
    
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
            num_heads=4,
            sigma_spatial=0.2,
            sigma_param=0.3
        )
    
    # Initialize numerical solutions manager
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
    
    # Initialize multi-target manager
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
    
    # Initialize stress analysis manager
    if 'stress_analyzer' not in st.session_state:
        st.session_state.stress_analyzer = StressAnalysisManager()
    
    # Initialize sunburst chart manager
    if 'sunburst_manager' not in st.session_state:
        st.session_state.sunburst_manager = SunburstChartManager()
    
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    
    # Initialize multi-target predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
    
    # Initialize stress summary DataFrame
    if 'stress_summary_df' not in st.session_state:
        st.session_state.stress_summary_df = pd.DataFrame()
    
    # Get grid extent for visualization
    extent = get_grid_extent()
    
    # Sidebar configuration
    st.sidebar.header("🔮 Attention Interpolator Settings")
    
    with st.sidebar.expander("⚙️ Model Parameters", expanded=False):
        num_heads = st.slider("Number of Attention Heads", 1, 8, 4, 1)
        sigma_spatial = st.slider("Spatial Sigma (σ_spatial)", 0.05, 1.0, 0.2, 0.05)
        sigma_param = st.slider("Parameter Sigma (σ_param)", 0.05, 1.0, 0.3, 0.05)
        use_gaussian = st.checkbox("Use Gaussian Spatial Regularization", True)
        
        if st.button("🔄 Update Model Parameters"):
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
                num_heads=num_heads,
                sigma_spatial=sigma_spatial,
                sigma_param=sigma_param,
                use_gaussian=use_gaussian
            )
            st.success("Model parameters updated!")
    
    # Main interface tabs - ADDED NEW TAB FOR STRESS ANALYSIS
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 Load Source Data", 
        "🎯 Configure Target", 
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict", 
        "📊 Results & Export",
        "📁 Manage Files",
        "📈 Stress Analysis & Sunburst"  # NEW TAB
    ])
    
    # ... (keep existing tabs 1-6 code exactly as in your provided code) ...
    # Tab 1: Load Source Data
    with tab1:
        # ... (existing tab1 code) ...
        pass
    
    # Tab 2: Configure Target
    with tab2:
        # ... (existing tab2 code) ...
        pass
    
    # Tab 3: Configure Multiple Targets
    with tab3:
        # ... (existing tab3 code) ...
        pass
    
    # Tab 4: Train & Predict
    with tab4:
        # ... (existing tab4 code) ...
        pass
    
    # Tab 5: Results & Export
    with tab5:
        # ... (existing tab5 code) ...
        pass
    
    # Tab 6: Manage Files
    with tab6:
        # ... (existing tab6 code) ...
        pass
    
    # =============================================
    # NEW TAB 7: STRESS ANALYSIS & SUNBURST CHARTS
    # =============================================
    with tab7:
        st.header("📈 Stress Analysis and Sunburst Visualization")
        
        # Update stress summary DataFrame
        if st.button("🔄 Update Stress Summary", type="secondary"):
            with st.spinner("Computing stress statistics..."):
                st.session_state.stress_summary_df = (
                    st.session_state.stress_analyzer.create_stress_summary_dataframe(
                        st.session_state.source_simulations,
                        st.session_state.multi_target_predictions
                    )
                )
                if not st.session_state.stress_summary_df.empty:
                    st.success(f"✅ Stress summary updated with {len(st.session_state.stress_summary_df)} entries")
                else:
                    st.warning("No data available for stress analysis")
        
        # Display stress summary if available
        if not st.session_state.stress_summary_df.empty:
            st.subheader("📋 Stress Summary Statistics")
            
            # Show DataFrame
            st.dataframe(
                st.session_state.stress_summary_df.style.format({
                    col: "{:.3f}" for col in st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns
                }),
                use_container_width=True,
                height=400
            )
            
            # Download stress summary
            csv_buffer = BytesIO()
            st.session_state.stress_summary_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Stress Summary CSV",
                data=csv_buffer,
                file_name=f"stress_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # =============================================
            # SUNBURST CHART CONFIGURATION
            # =============================================
            st.subheader("🌀 Sunburst Chart Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Select hierarchical levels
                available_columns = list(st.session_state.stress_summary_df.columns)
                categorical_cols = ['defect_type', 'shape', 'orientation', 'type']
                categorical_cols = [c for c in categorical_cols if c in available_columns]
                
                level1 = st.selectbox(
                    "First Level (Center)",
                    categorical_cols,
                    index=0 if 'type' in categorical_cols else 0
                )
                
                level2_options = [c for c in categorical_cols if c != level1]
                level2 = st.selectbox(
                    "Second Level",
                    ['None'] + level2_options,
                    index=0
                )
                
                level3_options = [c for c in level2_options if c != level2 and level2 != 'None']
                level3 = st.selectbox(
                    "Third Level",
                    ['None'] + level3_options,
                    index=0
                )
            
            with col2:
                # Select value column for sizing
                numeric_cols = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                stress_value_cols = [c for c in numeric_cols if 'max' in c or 'mean' in c]
                
                value_column = st.selectbox(
                    "Value Metric",
                    stress_value_cols,
                    index=0 if 'max_von_mises' in stress_value_cols else 0
                )
            
            with col3:
                # Select colormap
                colormaps = st.session_state.sunburst_manager.get_all_colormaps()
                selected_colormap = st.selectbox(
                    "Colormap",
                    colormaps,
                    index=colormaps.index('viridis') if 'viridis' in colormaps else 0
                )
                
                # Chart type selection
                chart_type = st.radio(
                    "Chart Type",
                    ["Sunburst", "Treemap", "Parallel Categories", "Radial Bar"],
                    horizontal=True
                )
            
            # Build path columns list
            path_columns = [level1]
            if level2 != 'None':
                path_columns.append(level2)
            if level3 != 'None':
                path_columns.append(level3)
            
            # Additional filters
            st.subheader("🔍 Filter Options")
            
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # Filter by defect type
                defect_types = st.session_state.stress_summary_df['defect_type'].unique() if 'defect_type' in st.session_state.stress_summary_df.columns else []
                selected_defects = st.multiselect(
                    "Filter by Defect Type",
                    defect_types,
                    default=defect_types.tolist() if len(defect_types) > 0 else []
                )
            
            with filter_col2:
                # Filter by shape
                shapes = st.session_state.stress_summary_df['shape'].unique() if 'shape' in st.session_state.stress_summary_df.columns else []
                selected_shapes = st.multiselect(
                    "Filter by Shape",
                    shapes,
                    default=shapes.tolist() if len(shapes) > 0 else []
                )
            
            with filter_col3:
                # Filter by simulation type
                sim_types = st.session_state.stress_summary_df['type'].unique() if 'type' in st.session_state.stress_summary_df.columns else []
                selected_types = st.multiselect(
                    "Filter by Simulation Type",
                    sim_types,
                    default=sim_types.tolist() if len(sim_types) > 0 else []
                )
            
            # Apply filters
            df_filtered = st.session_state.stress_summary_df.copy()
            
            if len(selected_defects) > 0:
                df_filtered = df_filtered[df_filtered['defect_type'].isin(selected_defects)]
            
            if len(selected_shapes) > 0:
                df_filtered = df_filtered[df_filtered['shape'].isin(selected_shapes)]
            
            if len(selected_types) > 0:
                df_filtered = df_filtered[df_filtered['type'].isin(selected_types)]
            
            # Generate chart button
            if st.button("🌀 Generate Visualization", type="primary"):
                if len(df_filtered) == 0:
                    st.warning("No data available after filtering")
                elif len(path_columns) == 0:
                    st.warning("Please select at least one hierarchical level")
                elif value_column not in df_filtered.columns:
                    st.warning(f"Value column '{value_column}' not found in data")
                else:
                    with st.spinner("Generating visualization..."):
                        try:
                            if chart_type == "Sunburst":
                                # Create sunburst chart
                                fig = st.session_state.sunburst_manager.create_sunburst_chart(
                                    df=df_filtered,
                                    path_columns=path_columns,
                                    value_column=value_column,
                                    title=f"Stress Analysis: {value_column.replace('_', ' ').title()}",
                                    colormap=selected_colormap
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif chart_type == "Treemap":
                                # Create treemap chart
                                fig = st.session_state.sunburst_manager.create_treemap_chart(
                                    df=df_filtered,
                                    path_columns=path_columns,
                                    value_column=value_column,
                                    title=f"Stress Analysis: {value_column.replace('_', ' ').title()}",
                                    colormap=selected_colormap
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                            elif chart_type == "Parallel Categories":
                                # For parallel categories, need at least 2 dimensions
                                if len(path_columns) >= 2:
                                    dimensions = path_columns[:min(4, len(path_columns))]  # Limit to 4 dimensions
                                    fig = st.session_state.sunburst_manager.create_parallel_categories(
                                        df=df_filtered,
                                        dimensions=dimensions,
                                        color_column=value_column,
                                        title=f"Stress Analysis: {value_column.replace('_', ' ').title()}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Parallel Categories requires at least 2 dimensions")
                            
                            elif chart_type == "Radial Bar":
                                # For radial bar, need categories and multiple value columns
                                if len(path_columns) > 0:
                                    category_col = path_columns[0]
                                    # Select multiple value columns
                                    selected_values = st.multiselect(
                                        "Select Value Columns for Radial Bar",
                                        stress_value_cols,
                                        default=stress_value_cols[:min(3, len(stress_value_cols))]
                                    )
                                    
                                    if len(selected_values) > 0:
                                        fig = st.session_state.sunburst_manager.create_radial_bar_chart(
                                            df=df_filtered,
                                            categories=category_col,
                                            values=selected_values,
                                            title="Stress Component Comparison"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("Please select at least one value column")
                                else:
                                    st.warning("Please select at least one category column")
                        
                        except Exception as e:
                            st.error(f"Error generating chart: {str(e)}")
                            print(traceback.format_exc())
            
            # =============================================
            # ADDITIONAL VISUALIZATIONS
            # =============================================
            st.subheader("📊 Additional Visualizations")
            
            viz_tabs = st.tabs(["Correlation Matrix", "3D Scatter Plot", "Heatmap", "Box Plots"])
            
            with viz_tabs[0]:
                # Correlation matrix
                if len(df_filtered.select_dtypes(include=[np.number]).columns) > 1:
                    corr_matrix = df_filtered.select_dtypes(include=[np.number]).corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=selected_colormap,
                        title="Correlation Matrix of Stress Metrics"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation matrix")
            
            with viz_tabs[1]:
                # 3D Scatter plot
                if len(stress_value_cols) >= 3:
                    col_x, col_y, col_z = st.columns(3)
                    
                    with col_x:
                        x_col = st.selectbox("X-axis", stress_value_cols, index=0)
                    with col_y:
                        y_col = st.selectbox("Y-axis", stress_value_cols, index=1)
                    with col_z:
                        z_col = st.selectbox("Z-axis", stress_value_cols, index=2)
                    
                    color_by = st.selectbox(
                        "Color by",
                        ['defect_type', 'shape', 'orientation', 'type'] + stress_value_cols,
                        index=0
                    )
                    
                    if st.button("Generate 3D Scatter"):
                        fig_3d = px.scatter_3d(
                            df_filtered,
                            x=x_col,
                            y=y_col,
                            z=z_col,
                            color=color_by if color_by in df_filtered.columns else None,
                            hover_name='id',
                            title="3D Stress Metric Visualization",
                            color_continuous_scale=selected_colormap,
                            opacity=0.7
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.info("Need at least 3 numeric columns for 3D scatter plot")
            
            with viz_tabs[2]:
                # Heatmap by parameter combinations
                if len(path_columns) >= 2:
                    heatmap_data = df_filtered.pivot_table(
                        index=path_columns[0],
                        columns=path_columns[1] if len(path_columns) > 1 else 'type',
                        values=value_column,
                        aggfunc='mean'
                    )
                    
                    fig_heat = px.imshow(
                        heatmap_data,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=selected_colormap,
                        title=f"Heatmap: {value_column.replace('_', ' ').title()} by {path_columns[0]} and {path_columns[1]}"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Need at least 2 hierarchical levels for heatmap")
            
            with viz_tabs[3]:
                # Box plots
                if len(path_columns) > 0:
                    group_by = st.selectbox(
                        "Group by for Box Plot",
                        path_columns,
                        index=0
                    )
                    
                    box_values = st.multiselect(
                        "Select metrics for Box Plot",
                        stress_value_cols,
                        default=stress_value_cols[:min(5, len(stress_value_cols))]
                    )
                    
                    if len(box_values) > 0:
                        # Create subplots
                        fig_boxes = make_subplots(
                            rows=len(box_values),
                            cols=1,
                            subplot_titles=[v.replace('_', ' ').title() for v in box_values],
                            vertical_spacing=0.1
                        )
                        
                        for i, value_col in enumerate(box_values):
                            # Create box plot for each group
                            for group in df_filtered[group_by].unique():
                                group_data = df_filtered[df_filtered[group_by] == group][value_col].dropna()
                                
                                fig_boxes.add_trace(
                                    go.Box(
                                        y=group_data,
                                        name=str(group),
                                        boxpoints='outliers',
                                        jitter=0.3,
                                        pointpos=-1.8,
                                        marker_color=px.colors.sequential.Viridis[i/len(box_values)]
                                    ),
                                    row=i+1,
                                    col=1
                                )
                        
                        fig_boxes.update_layout(
                            height=300 * len(box_values),
                            showlegend=True,
                            title_text=f"Box Plots by {group_by}"
                        )
                        
                        st.plotly_chart(fig_boxes, use_container_width=True)
                else:
                    st.info("Please configure hierarchical levels first")
            
            # =============================================
            # STRESS PEAK ANALYSIS
            # =============================================
            st.subheader("🏔️ Stress Peak Analysis")
            
            if 'prediction_results' in st.session_state:
                stress_fields = st.session_state.prediction_results.get('stress_fields', {})
                
                if any(isinstance(v, np.ndarray) for v in stress_fields.values()):
                    threshold = st.slider("Peak Detection Threshold (%ile)", 90.0, 99.9, 95.0, 0.1)
                    
                    if st.button("Analyze Stress Peaks"):
                        peaks = st.session_state.stress_analyzer.extract_stress_peaks(
                            stress_fields,
                            threshold_percentile=threshold
                        )
                        
                        if peaks:
                            st.write("**Detected Stress Peaks:**")
                            for comp, peak_info in peaks.items():
                                with st.expander(f"{comp} - {peak_info['num_peaks']} peaks"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Max Value", f"{peak_info['max_value']:.3f} GPa")
                                        st.metric("Mean Peak Value", f"{peak_info['mean_peak_value']:.3f} GPa")
                                    with col2:
                                        st.metric("Threshold", f"{peak_info['threshold']:.3f} GPa")
                                        st.metric("Peak Position", str(peak_info['max_position']))
                            
                            # Plot peak locations
                            fig_peaks, axes = plt.subplots(1, len(peaks), figsize=(5*len(peaks), 4))
                            if len(peaks) == 1:
                                axes = [axes]
                            
                            for ax, (comp, peak_info) in zip(axes, peaks.items()):
                                stress_data = stress_fields.get(comp)
                                if stress_data is not None:
                                    im = ax.imshow(stress_data, extent=extent, cmap='hot', origin='lower')
                                    ax.set_title(f"{comp} Peaks")
                                    ax.set_xlabel('x (nm)')
                                    ax.set_ylabel('y (nm)')
                                    
                                    # Plot peaks
                                    peak_indices = peak_info['peak_indices']
                                    if len(peak_indices[0]) > 0:
                                        ax.scatter(
                                            peak_indices[1] - stress_data.shape[1]/2,
                                            peak_indices[0] - stress_data.shape[0]/2,
                                            c='cyan', s=10, alpha=0.6, edgecolors='white'
                                        )
                                    
                                    plt.colorbar(im, ax=ax, label='Stress (GPa)')
                            
                            st.pyplot(fig_peaks)
                        else:
                            st.info("No significant peaks detected above threshold")
            
            # =============================================
            # ADVANCED STATISTICAL ANALYSIS
            # =============================================
            st.subheader("📈 Advanced Statistical Analysis")
            
            if len(df_filtered) > 5:  # Need enough data for meaningful stats
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    # Select metric for distribution analysis
                    dist_metric = st.selectbox(
                        "Select metric for distribution",
                        stress_value_cols,
                        index=0
                    )
                    
                    if dist_metric in df_filtered.columns:
                        fig_dist, ax = plt.subplots(2, 1, figsize=(10, 8))
                        
                        # Histogram
                        ax[0].hist(df_filtered[dist_metric].dropna(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                        ax[0].set_xlabel(dist_metric.replace('_', ' ').title())
                        ax[0].set_ylabel('Frequency')
                        ax[0].set_title(f'Distribution of {dist_metric.replace("_", " ").title()}')
                        ax[0].grid(True, alpha=0.3)
                        
                        # Q-Q plot
                        stats.probplot(df_filtered[dist_metric].dropna(), dist="norm", plot=ax[1])
                        ax[1].set_title(f'Q-Q Plot for {dist_metric.replace("_", " ").title()}')
                        ax[1].grid(True, alpha=0.3)
                        
                        st.pyplot(fig_dist)
                
                with stat_col2:
                    # Pairwise comparisons
                    compare_col1 = st.selectbox("Compare metric 1", stress_value_cols, index=0)
                    compare_col2 = st.selectbox("Compare metric 2", stress_value_cols, index=1)
                    
                    if compare_col1 in df_filtered.columns and compare_col2 in df_filtered.columns:
                        fig_scatter, ax = plt.subplots(figsize=(8, 6))
                        
                        scatter = ax.scatter(
                            df_filtered[compare_col1],
                            df_filtered[compare_col2],
                            c=df_filtered.get('max_von_mises', 0) if 'max_von_mises' in df_filtered.columns else None,
                            cmap='viridis',
                            alpha=0.7,
                            s=50
                        )
                        
                        ax.set_xlabel(compare_col1.replace('_', ' ').title())
                        ax.set_ylabel(compare_col2.replace('_', ' ').title())
                        ax.set_title(f'{compare_col1} vs {compare_col2}')
                        ax.grid(True, alpha=0.3)
                        
                        if 'max_von_mises' in df_filtered.columns:
                            plt.colorbar(scatter, ax=ax, label='Max Von Mises (GPa)')
                        
                        st.pyplot(fig_scatter)
                        
                        # Calculate correlation
                        correlation = df_filtered[[compare_col1, compare_col2]].corr().iloc[0, 1]
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        else:
            st.info("👈 Please load simulations and generate predictions first to enable stress analysis")

# =============================================
# ENHANCED MAIN APPLICATION
# =============================================
def main():
    """Main application with enhanced stress analysis"""
    
    st.sidebar.header("📁 Directory Information")
    st.sidebar.write(f"**App Directory:** `{SCRIPT_DIR}`")
    st.sidebar.write(f"**Solutions Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
    
    if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
        st.sidebar.warning("⚠️ Solutions directory not found")
        if st.sidebar.button("📁 Create Directory"):
            os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
            st.sidebar.success("✅ Directory created")
            st.rerun()
    
    st.sidebar.header("🔧 Operation Mode")
    
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["Attention Interpolation", "Run New Simulation", 
         "Compare Saved Simulations", "Single Simulation View",
         "Stress Analysis Dashboard"],  # Added new mode
        index=0
    )
    
    if operation_mode == "Attention Interpolation":
        create_attention_interface()
    elif operation_mode == "Stress Analysis Dashboard":
        st.header("📊 Stress Analysis Dashboard")
        
        # Initialize managers
        if 'stress_analyzer' not in st.session_state:
            st.session_state.stress_analyzer = StressAnalysisManager()
        if 'sunburst_manager' not in st.session_state:
            st.session_state.sunburst_manager = SunburstChartManager()
        
        # Load all available data
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
        
        # Load all simulations from directory
        all_files = st.session_state.solutions_manager.get_all_files()
        all_simulations = []
        
        if st.button("📥 Load All Simulations for Analysis"):
            with st.spinner("Loading all simulations..."):
                for file_info in all_files[:50]:  # Limit to 50 files
                    try:
                        sim_data = st.session_state.solutions_manager.load_simulation(
                            file_info['path'],
                            st.session_state.interpolator
                        )
                        all_simulations.append(sim_data)
                    except:
                        continue
                
                if all_simulations:
                    # Create comprehensive stress summary
                    stress_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                        all_simulations, {}
                    )
                    
                    if not stress_df.empty:
                        st.session_state.stress_summary_df = stress_df
                        st.success(f"✅ Loaded {len(all_simulations)} simulations for analysis")
                    else:
                        st.warning("No stress data found in loaded simulations")
                else:
                    st.error("No simulations could be loaded")
        
        # Display analysis interface if data available
        if not st.session_state.stress_summary_df.empty:
            # Quick stats
            st.subheader("📈 Quick Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Simulations", len(st.session_state.stress_summary_df))
            with col2:
                max_vm = st.session_state.stress_summary_df['max_von_mises'].max() if 'max_von_mises' in st.session_state.stress_summary_df.columns else 0
                st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
            with col3:
                mean_vm = st.session_state.stress_summary_df['max_von_mises'].mean() if 'max_von_mises' in st.session_state.stress_summary_df.columns else 0
                st.metric("Avg Max Von Mises", f"{mean_vm:.2f} GPa")
            with col4:
                defect_counts = st.session_state.stress_summary_df['defect_type'].value_counts().to_dict() if 'defect_type' in st.session_state.stress_summary_df.columns else {}
                st.metric("Unique Defect Types", len(defect_counts))
            
            # Show the stress analysis tab interface
            create_attention_interface()  # This will show tab7 content
        else:
            st.info("Please load simulations first to enable the stress analysis dashboard")
    
    else:
        st.warning("⚠️ This mode is not fully integrated with attention interpolation.")
        st.info("Please use 'Attention Interpolation' or 'Stress Analysis Dashboard' mode.")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Enhanced Theoretical Analysis: Stress Metrics and Visualization", expanded=False):
    st.markdown(f"""
    ## 📊 **Enhanced Stress Analysis and Visualization**
    
    ### **🏔️ Maximum Stress Value Capture**
    
    **New Stress Metrics:**
    1. **Hydrostatic Stress (σ_hydro):**
       - Max Absolute Value: `max_abs_hydrostatic`
       - Maximum Value: `max_hydrostatic`
       - Minimum Value: `min_hydrostatic`
       - Mean Absolute Value: `mean_abs_hydrostatic`
       - Standard Deviation: `hydro_std`
       - Skewness: `hydro_skewness`
       - Kurtosis: `hydro_kurtosis`
    
    2. **Stress Magnitude:**
       - Maximum: `max_stress_magnitude`
       - Mean: `mean_stress_magnitude`
    
    3. **Von Mises Stress (σ_vM):**
       - Maximum: `max_von_mises`
       - Mean: `mean_von_mises`
       - Minimum: `min_von_mises`
       - 95th Percentile: `von_mises_p95`
       - 99th Percentile: `von_mises_p99`
       - 99.9th Percentile: `von_mises_p99_9`
    
    4. **Principal Stresses (σ₁, σ₂, σ₃):**
       - Max Principal 1: `max_principal_1`
       - Max Principal 2: `max_principal_2`
       - Max Principal 3: `max_principal_3`
       - Max Absolute Principal: `max_principal_abs`
       - Maximum Shear (Tresca): `max_shear_tresca`
    
    ### **🌀 Sunburst Chart Features**
    
    **Hierarchical Visualization:**
    1. **Multi-level Hierarchy:**
       - First Level (Center): Defect type, Shape, or Simulation type
       - Second Level: Orientation, ε*, κ, etc.
       - Third Level: Additional parameters or categories
    
    2. **Value Metrics:** Any stress metric can be used for:
       - Area sizing in sunburst
       - Color mapping
       - Value display
    
    3. **50+ Colormaps:** Full matplotlib colormap support:
       - Sequential: viridis, plasma, inferno, magma, cividis
       - Diverging: RdBu, PiYG, PRGn, RdYlBu, RdYlGn
       - Cyclic: twilight, twilight_shifted, hsv
       - Qualitative: Set1, Set2, Set3, tab10, tab20
    
    **Advanced Chart Types:**
    1. **Sunburst Chart:**
       - Radial hierarchical visualization
       - Interactive hover information
       - Click to drill down/up
       - Value-based coloring
    
    2. **Treemap Chart:**
       - Rectangular hierarchical visualization
       - Better for comparison of leaf nodes
       - Space-efficient display
    
    3. **Parallel Categories:**
       - Multi-dimensional categorical visualization
       - Shows relationships between parameters
       - Color by stress values
    
    4. **Radial Bar Chart:**
       - Circular bar chart for comparisons
       - Multiple stress metrics simultaneously
       - Visual comparison across categories
    
    ### **📈 Additional Visualizations**
    
    **1. Correlation Matrix:**
    - Shows relationships between all stress metrics
    - Identifies correlated stress components
    - Heatmap visualization with numerical values
    
    **2. 3D Scatter Plot:**
    - Interactive 3D visualization of stress metrics
    - Color by defect type or stress values
    - Rotate, zoom, and pan for exploration
    
    **3. Heatmaps:**
    - 2D matrix of stress values by parameter combinations
    - Quick identification of high-stress regions
    - Color-coded by stress magnitude
    
    **4. Box Plots:**
    - Distribution analysis of stress metrics
    - Comparison across different categories
    - Outlier detection
    
    ### **🏔️ Stress Peak Analysis**
    
    **Peak Detection Algorithm:**
    1. **Threshold-based Detection:**
       - User-defined percentile threshold (90-99.9%)
       - Automatic peak identification
       - Peak counting and characterization
    
    2. **Peak Characterization:**
       - Number of peaks above threshold
       - Maximum peak value and position
       - Mean peak value
       - Peak spatial distribution
    
    3. **Visualization:**
       - Overlay peaks on stress field maps
       - Color-coded by peak intensity
       - Interactive exploration
    
    ### **📊 Statistical Analysis**
    
    **Distribution Analysis:**
    1. **Histograms:** Frequency distribution of stress metrics
    2. **Q-Q Plots:** Normality testing
    3. **Correlation Analysis:** Pearson correlation between metrics
    4. **Descriptive Statistics:** Mean, median, std, min, max
    
    **Comparative Analysis:**
    1. **Pairwise Scatter Plots:** Compare any two stress metrics
    2. **Grouped Analysis:** Compare across defect types, shapes, orientations
    3. **Trend Analysis:** Identify patterns in stress development
    
    ### **🔬 Scientific Applications**
    
    **1. Stress Concentration Analysis:**
    - Identify critical stress locations
    - Quantify stress intensity factors
    - Study defect interactions
    
    **2. Material Design Optimization:**
    - Find parameter combinations that minimize stress
    - Optimize defect geometry for stress relief
    - Design stress-resistant microstructures
    
    **3. Failure Prediction:**
    - Use maximum stress values for failure criteria
    - Predict crack initiation locations
    - Estimate fatigue life
    
    **4. Comparative Studies:**
    - Compare stress fields across different defect types
    - Study orientation effects on stress distribution
    - Analyze shape influence on stress concentrations
    
    **5. Data Mining:**
    - Discover hidden patterns in stress data
    - Cluster simulations by stress characteristics
    - Identify outlier simulations
    
    ### **🚀 Implementation Details**
    
    **Data Pipeline:**
    1. **Data Collection:** Extract stress fields from all simulations
    2. **Metric Computation:** Calculate all stress metrics
    3. **Data Aggregation:** Combine into comprehensive DataFrame
    4. **Visualization:** Generate interactive charts
    
    **Performance Optimization:**
    - Vectorized NumPy operations for speed
    - Caching of computed metrics
    - Lazy loading of visualization components
    - Efficient memory management
    
    **User Interface:**
    - Intuitive dropdown selections
    - Real-time filtering
    - Interactive chart controls
    - Export capabilities
    
    **Integration:**
    - Seamless integration with existing attention interpolation
    - Support for both source simulations and predictions
    - Unified data management
    - Consistent user experience
    
    This enhanced stress analysis system provides a comprehensive toolkit for
    post-processing simulation results, enabling deep insights into stress
    distributions, peak stresses, and parameter-stress relationships.
    """)

if __name__ == "__main__":
    main()

st.caption(f"🔬 Enhanced Multi-Target Spatial-Attention Stress Interpolation • Stress Analysis Dashboard • Sunburst Visualization • 50+ Colormaps • 2025")
