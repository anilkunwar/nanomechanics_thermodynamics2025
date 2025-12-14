# Add these imports at the top
import io
from io import BytesIO
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import os
import zipfile
import json
import pickle
from datetime import datetime

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# =============================================
# ENHANCED SUNBURST CHART MANAGER WITH ADVANCED VISUALIZATIONS
# =============================================
class EnhancedSunburstChartManager:
    """Enhanced manager for advanced hierarchical visualizations"""
   
    @staticmethod
    def create_stress_diffusion_sunburst(df: pd.DataFrame,
                                       stress_metric: str = 'max_von_mises',
                                       diffusion_metric: str = 'diff_factor',
                                       sintering_metric: str = 'Ts',
                                       title: str = "Stress-Diffusion-Sintering Analysis") -> go.Figure:
        """
        Create a comprehensive sunburst chart for stress, diffusion, and sintering
       
        Args:
            df: DataFrame containing stress, diffusion, and sintering data
            stress_metric: Column name for stress metric
            diffusion_metric: Column name for diffusion factor
            sintering_metric: Column name for sintering temperature
            title: Chart title
           
        Returns:
            Plotly Figure object
        """
        # Create aggregated DataFrame for hierarchical visualization
        df_agg = df.copy()
       
        # Create hierarchical path: defect_type -> shape -> orientation
        df_agg['path'] = df_agg['defect_type'] + ' / ' + df_agg['shape'] + ' / ' + df_agg['orientation']
       
        # Calculate aggregated metrics
        agg_metrics = df_agg.groupby(['defect_type', 'shape', 'orientation']).agg({
            stress_metric: 'mean',
            diffusion_metric: 'mean',
            sintering_metric: 'mean',
            'eps0': 'mean',
            'kappa': 'mean'
        }).reset_index()
       
        # Create multi-level sunburst
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'sunburst'}, {'type': 'sunburst'}, {'type': 'sunburst'}]],
            subplot_titles=[f"Stress ({stress_metric})",
                          f"Diffusion ({diffusion_metric})",
                          f"Sintering Temp ({sintering_metric})"]
        )
       
        # Stress sunburst
        fig.add_trace(go.Sunburst(
            labels=agg_metrics['orientation'],
            parents=[f"{row['defect_type']}/{row['shape']}" for _, row in agg_metrics.iterrows()],
            values=agg_metrics[stress_metric],
            branchvalues="total",
            marker=dict(
                colors=agg_metrics[stress_metric],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Stress (GPa)")
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         f'{stress_metric}: %{{value:.2f}} GPa<br>' +
                         'Path: %{parent}<extra></extra>',
            name="Stress"
        ), row=1, col=1)
       
        # Diffusion sunburst
        fig.add_trace(go.Sunburst(
            labels=agg_metrics['orientation'],
            parents=[f"{row['defect_type']}/{row['shape']}" for _, row in agg_metrics.iterrows()],
            values=agg_metrics[diffusion_metric],
            branchvalues="total",
            marker=dict(
                colors=agg_metrics[diffusion_metric],
                colorscale='Greens',
                showscale=True,
                colorbar=dict(title="Diffusion Factor")
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         f'{diffusion_metric}: %{{value:.2f}}x<br>' +
                         'Path: %{parent}<extra></extra>',
            name="Diffusion"
        ), row=1, col=2)
       
        # Sintering temperature sunburst
        fig.add_trace(go.Sunburst(
            labels=agg_metrics['orientation'],
            parents=[f"{row['defect_type']}/{row['shape']}" for _, row in agg_metrics.iterrows()],
            values=agg_metrics[sintering_metric],
            branchvalues="total",
            marker=dict(
                colors=agg_metrics[sintering_metric],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Temperature (K)")
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         f'{sintering_metric}: %{{value:.1f}} K<br>' +
                         'Path: %{parent}<extra></extra>',
            name="Sintering Temp"
        ), row=1, col=3)
       
        fig.update_layout(
            title_text=title,
            height=600,
            showlegend=False
        )
       
        return fig
   
    @staticmethod
    def create_radar_chart_comparison(df: pd.DataFrame,
                                    categories: List[str],
                                    metrics: List[str],
                                    group_by: str = 'defect_type',
                                    title: str = "Radar Chart Comparison") -> go.Figure:
        """
        Create radar chart comparing multiple metrics across categories
       
        Args:
            df: DataFrame containing the data
            categories: List of category values to compare
            metrics: List of metric columns to include in radar
            group_by: Column to group by
            title: Chart title
           
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
       
        # Normalize metrics for fair comparison
        df_normalized = df.copy()
        for metric in metrics:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()
                if max_val > min_val:
                    df_normalized[metric] = (df[metric] - min_val) / (max_val - min_val)
       
        # Add traces for each category
        colors = px.colors.qualitative.Set3
       
        for idx, category in enumerate(categories):
            if group_by in df_normalized.columns:
                category_data = df_normalized[df_normalized[group_by] == category]
                if not category_data.empty:
                    # Calculate mean values for each metric
                    mean_values = [category_data[metric].mean() if metric in category_data.columns else 0
                                 for metric in metrics]
                   
                    # Close the radar chart
                    theta = metrics + [metrics[0]]
                    r = mean_values + [mean_values[0]]
                   
                    fig.add_trace(go.Scatterpolar(
                        r=r,
                        theta=theta,
                        name=str(category),
                        fill='toself',
                        line_color=colors[idx % len(colors)],
                        opacity=0.7
                    ))
       
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title=title,
            height=500
        )
       
        return fig
   
    @staticmethod
    def create_3d_scatter_plot(df: pd.DataFrame,
                             x_col: str,
                             y_col: str,
                             z_col: str,
                             color_col: str,
                             size_col: str = None,
                             title: str = "3D Scatter Plot") -> go.Figure:
        """
        Create interactive 3D scatter plot
       
        Args:
            df: DataFrame containing the data
            x_col: Column for x-axis
            y_col: Column for y-axis
            z_col: Column for z-axis
            color_col: Column for color encoding
            size_col: Column for size encoding (optional)
            title: Chart title
           
        Returns:
            Plotly Figure object
        """
        if size_col:
            fig = px.scatter_3d(
                df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_col,
                size=size_col,
                hover_name='id' if 'id' in df.columns else None,
                title=title,
                color_continuous_scale='Viridis',
                opacity=0.7,
                labels={col: col.replace('_', ' ').title() for col in [x_col, y_col, z_col, color_col]}
            )
        else:
            fig = px.scatter_3d(
                df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_col,
                hover_name='id' if 'id' in df.columns else None,
                title=title,
                color_continuous_scale='Viridis',
                opacity=0.7,
                labels={col: col.replace('_', ' ').title() for col in [x_col, y_col, z_col, color_col]}
            )
       
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                zaxis_title=z_col.replace('_', ' ').title()
            ),
            height=600
        )
       
        return fig
   
    @staticmethod
    def create_parallel_coordinates(df: pd.DataFrame,
                                  dimensions: List[str],
                                  color_column: str,
                                  title: str = "Parallel Coordinates") -> go.Figure:
        """
        Create parallel coordinates plot for high-dimensional data
       
        Args:
            df: DataFrame containing the data
            dimensions: List of columns to include as dimensions
            color_column: Column for coloring
            title: Chart title
           
        Returns:
            Plotly Figure object
        """
        fig = px.parallel_coordinates(
            df,
            dimensions=dimensions,
            color=color_column,
            color_continuous_scale='Viridis',
            title=title,
            labels={col: col.replace('_', ' ').title() for col in dimensions}
        )
       
        fig.update_layout(height=500)
       
        return fig
# =============================================
# ENHANCED DATA EXPORT MANAGER
# =============================================
class EnhancedDataExportManager:
    """Manager for exporting machine learning data in multiple formats"""
   
    @staticmethod
    def export_dataframe(df: pd.DataFrame, format_type: str = 'csv') -> bytes:
        """
        Export DataFrame to various formats
       
        Args:
            df: DataFrame to export
            format_type: Export format ('csv', 'excel', 'json', 'parquet', 'feather', 'hdf5', 'pickle')
           
        Returns:
            Bytes of exported data
        """
        buffer = BytesIO()
       
        try:
            if format_type == 'csv':
                df.to_csv(buffer, index=False)
                buffer.seek(0)
               
            elif format_type == 'excel':
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                buffer.seek(0)
               
            elif format_type == 'json':
                # Convert to JSON with proper handling of numpy types
                json_str = df.to_json(orient='records', date_format='iso', default_handler=str)
                buffer.write(json_str.encode('utf-8'))
                buffer.seek(0)
               
            elif format_type == 'parquet':
                df.to_parquet(buffer, index=False, compression='snappy')
                buffer.seek(0)
               
            elif format_type == 'feather':
                df.to_feather(buffer)
                buffer.seek(0)
               
            elif format_type == 'hdf5':
                if HAS_H5PY:
                    df.to_hdf(buffer, key='data', mode='w', format='table')
                    buffer.seek(0)
                else:
                    raise ValueError("HDF5 support not available (h5py not installed)")
               
            elif format_type == 'pickle':
                pickle.dump(df, buffer)
                buffer.seek(0)
               
            elif format_type == 'msgpack':
                if HAS_MSGPACK:
                    # Convert DataFrame to dict for msgpack
                    data_dict = df.to_dict(orient='records')
                    buffer.write(msgpack.packb(data_dict))
                    buffer.seek(0)
                else:
                    raise ValueError("Msgpack support not available (msgpack not installed)")
               
            elif format_type == 'html':
                html_content = df.to_html(index=False)
                buffer.write(html_content.encode('utf-8'))
                buffer.seek(0)
               
            else:
                raise ValueError(f"Unsupported format: {format_type}")
               
        except Exception as e:
            raise Exception(f"Error exporting data to {format_type}: {str(e)}")
       
        return buffer.getvalue()
   
    @staticmethod
    def export_simulation_data(simulation_data: Dict[str, Any],
                             format_type: str = 'json') -> bytes:
        """
        Export simulation data in various formats
       
        Args:
            simulation_data: Dictionary containing simulation data
            format_type: Export format
           
        Returns:
            Bytes of exported data
        """
        buffer = BytesIO()
       
        try:
            if format_type == 'json':
                def json_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                        return str(obj)
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    else:
                        return str(obj)
               
                json.dump(simulation_data, buffer, default=json_serializer, indent=2)
                buffer.seek(0)
               
            elif format_type == 'pickle':
                pickle.dump(simulation_data, buffer)
                buffer.seek(0)
               
            elif format_type == 'npz':
                # Convert data to numpy arrays
                np_data = {}
                for key, value in simulation_data.items():
                    if isinstance(value, np.ndarray):
                        np_data[key] = value
                    elif isinstance(value, (int, float, list)):
                        np_data[key] = np.array(value)
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, np.ndarray):
                                np_data[f"{key}_{subkey}"] = subvalue
               
                np.savez(buffer, **np_data)
                buffer.seek(0)
               
            elif format_type == 'hdf5':
                if HAS_H5PY:
                    with h5py.File(buffer, 'w') as h5f:
                        for key, value in simulation_data.items():
                            if isinstance(value, np.ndarray):
                                h5f.create_dataset(key, data=value)
                            elif isinstance(value, dict):
                                group = h5f.create_group(key)
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, np.ndarray):
                                        group.create_dataset(subkey, data=subvalue)
                    buffer.seek(0)
                else:
                    raise ValueError("HDF5 support not available (h5py not installed)")
               
            else:
                raise ValueError(f"Unsupported format for simulation data: {format_type}")
               
        except Exception as e:
            raise Exception(f"Error exporting simulation data: {str(e)}")
       
        return buffer.getvalue()
   
    @staticmethod
    def create_export_bundle(df: pd.DataFrame,
                           simulation_data: Dict[str, Any] = None,
                           charts: List[go.Figure] = None) -> bytes:
        """
        Create a comprehensive export bundle with data, metadata, and charts
       
        Args:
            df: Main DataFrame
            simulation_data: Additional simulation data
            charts: List of Plotly figures to include
           
        Returns:
            Bytes of zipped export bundle
        """
        zip_buffer = BytesIO()
       
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            try:
                # Export DataFrame in multiple formats
                for fmt in ['csv', 'excel', 'json', 'parquet']:
                    data_bytes = EnhancedDataExportManager.export_dataframe(df, fmt)
                    zip_file.writestr(f'data/data.{fmt}', data_bytes)
               
                # Export simulation data if available
                if simulation_data:
                    sim_bytes = EnhancedDataExportManager.export_simulation_data(simulation_data, 'json')
                    zip_file.writestr('simulation_data/simulation.json', sim_bytes)
               
                # Export charts as HTML and PNG
                if charts:
                    for i, chart in enumerate(charts):
                        # Export as HTML
                        html_str = chart.to_html(include_plotlyjs='cdn')
                        zip_file.writestr(f'charts/chart_{i}.html', html_str)
                       
                        # Export as PNG
                        try:
                            img_bytes = chart.to_image(format="png", width=1200, height=800)
                            zip_file.writestr(f'charts/chart_{i}.png', img_bytes)
                        except:
                            pass # Skip if PNG export fails
               
                # Add metadata file
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'data_shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': {col: str(df[col].dtype) for col in df.columns},
                    'summary_stats': {
                        col: {
                            'mean': float(df[col].mean()) if df[col].dtype in [np.float64, np.float32, np.int64, np.int32] else None,
                            'std': float(df[col].std()) if df[col].dtype in [np.float64, np.float32] else None,
                            'min': float(df[col].min()) if df[col].dtype in [np.float64, np.float32, np.int64, np.int32] else None,
                            'max': float(df[col].max()) if df[col].dtype in [np.float64, np.float32, np.int64, np.int32] else None
                        }
                        for col in df.select_dtypes(include=[np.number]).columns
                    }
                }
               
                metadata_bytes = json.dumps(metadata, indent=2).encode('utf-8')
                zip_file.writestr('metadata.json', metadata_bytes)
               
                # Add README file
                readme_content = f"""
                Export Bundle - Machine Learning Data
                ===================================
               
                Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
               
                Contents:
                1. data/ - Main dataset in multiple formats (CSV, Excel, JSON, Parquet)
                2. simulation_data/ - Raw simulation data (if available)
                3. charts/ - Visualizations (HTML and PNG)
                4. metadata.json - Dataset metadata and statistics
               
                Dataset Information:
                - Shape: {df.shape[0]} rows × {df.shape[1]} columns
                - Columns: {', '.join(df.columns.tolist())}
               
                """
                zip_file.writestr('README.txt', readme_content)
               
            except Exception as e:
                error_content = f"Error creating export bundle: {str(e)}"
                zip_file.writestr('ERROR.txt', error_content)
       
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
# =============================================
# ENHANCED VISUALIZATION INTERFACE
# =============================================
def create_enhanced_visualization_interface():
    """Create enhanced visualization interface for stress, diffusion, and sintering analysis"""
   
    st.header("📊 Enhanced Visualization Dashboard")
   
    # Initialize enhanced managers
    if 'enhanced_sunburst_manager' not in st.session_state:
        st.session_state.enhanced_sunburst_manager = EnhancedSunburstChartManager()
   
    if 'export_manager' not in st.session_state:
        st.session_state.export_manager = EnhancedDataExportManager()
   
    # Check if we have data
    if 'stress_summary_df' not in st.session_state or st.session_state.stress_summary_df.empty:
        st.warning("⚠️ Please load simulations and generate predictions first")
        return
   
    df = st.session_state.stress_summary_df
   
    # Create tabs for different visualization types
    viz_tabs = st.tabs([
        "🌀 Multi-Metric Sunburst",
        "📡 Radar Charts",
        "📊 3D Visualization",
        "🔗 Parallel Coordinates",
        "📈 Statistical Analysis",
        "💾 Data Export"
    ])
   
    # Tab 1: Multi-Metric Sunburst
    with viz_tabs[0]:
        st.subheader("🌀 Multi-Metric Sunburst Analysis")
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            # Select metrics for comparison
            stress_metrics = [col for col in df.columns if 'stress' in col.lower() or 'von_mises' in col or 'hydro' in col]
            diffusion_metrics = [col for col in df.columns if 'diff' in col.lower() or 'diff_factor' in col]
            sintering_metrics = [col for col in df.columns if 'Ts' in col or 'sinter' in col.lower()]
           
            selected_stress = st.selectbox(
                "Stress Metric",
                stress_metrics,
                index=stress_metrics.index('max_von_mises') if 'max_von_mises' in stress_metrics else 0
            )
           
            selected_diffusion = st.selectbox(
                "Diffusion Metric",
                diffusion_metrics,
                index=diffusion_metrics.index('diff_factor') if 'diff_factor' in diffusion_metrics else 0
            )
           
            selected_sintering = st.selectbox(
                "Sintering Metric",
                sintering_metrics,
                index=sintering_metrics.index('Ts') if 'Ts' in sintering_metrics else 0
            )
       
        with col2:
            # Configuration options
            chart_title = st.text_input("Chart Title", "Stress-Diffusion-Sintering Analysis")
            color_scheme = st.selectbox(
                "Color Scheme",
                ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Rainbow'],
                index=0
            )
       
        if st.button("🌀 Generate Multi-Metric Sunburst", type="primary"):
            with st.spinner("Generating comprehensive sunburst visualization..."):
                try:
                    fig = st.session_state.enhanced_sunburst_manager.create_stress_diffusion_sunburst(
                        df=df,
                        stress_metric=selected_stress,
                        diffusion_metric=selected_diffusion,
                        sintering_metric=selected_sintering,
                        title=chart_title
                    )
                   
                    st.plotly_chart(fig, use_container_width=True)
                   
                    # Add insights
                    with st.expander("📋 Analysis Insights", expanded=True):
                        # Calculate correlations
                        if all(m in df.columns for m in [selected_stress, selected_diffusion, selected_sintering]):
                            stress_diff_corr = df[selected_stress].corr(df[selected_diffusion])
                            stress_temp_corr = df[selected_stress].corr(df[selected_sintering])
                           
                            st.markdown(f"""
                            **Key Correlations:**
                            - Stress vs Diffusion: `{stress_diff_corr:.3f}`
                            - Stress vs Sintering Temp: `{stress_temp_corr:.3f}`
                           
                            **Interpretation:**
                            - Positive correlation (close to 1): Metrics increase together
                            - Negative correlation (close to -1): One metric increases while the other decreases
                            - Near zero: Little to no linear relationship
                            """)
                       
                        # Top performers by category
                        if 'defect_type' in df.columns:
                            defect_stats = df.groupby('defect_type').agg({
                                selected_stress: 'mean',
                                selected_diffusion: 'mean',
                                selected_sintering: 'mean'
                            }).round(3)
                           
                            st.markdown("**Average Metrics by Defect Type:**")
                            st.dataframe(defect_stats)
               
                except Exception as e:
                    st.error(f"Error generating sunburst: {str(e)}")
   
    # Tab 2: Radar Charts
    with viz_tabs[1]:
        st.subheader("📡 Radar Chart Analysis")
       
        col1, col2 = st.columns(2)
       
        with col1:
            # Select metrics for radar chart
            all_metrics = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_metrics = st.multiselect(
                "Select Metrics for Radar Axes",
                all_metrics,
                default=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts', 'eps0', 'kappa']
            )
           
            # Grouping options
            group_options = [col for col in df.columns if df[col].nunique() < 20]
            selected_group = st.selectbox(
                "Group By",
                group_options,
                index=group_options.index('defect_type') if 'defect_type' in group_options else 0
            )
       
        with col2:
            # Radar chart configuration
            radar_title = st.text_input("Radar Chart Title", "Multi-Metric Comparison")
            fill_opacity = st.slider("Fill Opacity", 0.0, 1.0, 0.3, 0.05)
            normalize_data = st.checkbox("Normalize Metrics", value=True)
       
        if selected_metrics and len(selected_metrics) >= 3:
            if st.button("📡 Generate Radar Chart", type="primary"):
                with st.spinner("Generating radar chart..."):
                    try:
                        # Get unique groups
                        unique_groups = df[selected_group].unique() if selected_group in df.columns else ['All']
                       
                        fig = st.session_state.enhanced_sunburst_manager.create_radar_chart_comparison(
                            df=df,
                            categories=unique_groups[:6], # Limit to 6 groups for clarity
                            metrics=selected_metrics,
                            group_by=selected_group,
                            title=radar_title
                        )
                       
                        st.plotly_chart(fig, use_container_width=True)
                       
                        # Add metric statistics
                        with st.expander("📊 Metric Statistics", expanded=True):
                            stat_cols = st.columns(3)
                           
                            for idx, metric in enumerate(selected_metrics):
                                if metric in df.columns:
                                    col_idx = idx % 3
                                    with stat_cols[col_idx]:
                                        st.metric(
                                            label=metric.replace('_', ' ').title(),
                                            value=f"{df[metric].mean():.3f}",
                                            delta=f"±{df[metric].std():.3f}"
                                        )
                       
                    except Exception as e:
                        st.error(f"Error generating radar chart: {str(e)}")
        else:
            st.warning("Please select at least 3 metrics for radar chart")
   
    # Tab 3: 3D Visualization
    with viz_tabs[2]:
        st.subheader("📊 3D Visualization")
       
        col1, col2, col3 = st.columns(3)
       
        with col1:
            # X-axis selection
            x_options = df.select_dtypes(include=[np.number]).columns.tolist()
            x_axis = st.selectbox(
                "X-Axis",
                x_options,
                index=x_options.index('eps0') if 'eps0' in x_options else 0
            )
       
        with col2:
            # Y-axis selection
            y_options = [col for col in x_options if col != x_axis]
            y_axis = st.selectbox(
                "Y-Axis",
                y_options,
                index=y_options.index('kappa') if 'kappa' in y_options else 0
            )
       
        with col3:
            # Z-axis selection
            z_options = [col for col in y_options if col != y_axis]
            z_axis = st.selectbox(
                "Z-Axis",
                z_options,
                index=z_options.index('max_von_mises') if 'max_von_mises' in z_options else 0
            )
       
        # Color and size encoding
        col4, col5 = st.columns(2)
       
        with col4:
            color_options = ['defect_type', 'shape', 'orientation', 'type'] + df.select_dtypes(include=[np.number]).columns.tolist()
            color_by = st.selectbox(
                "Color By",
                color_options,
                index=color_options.index('defect_type') if 'defect_type' in color_options else 0
            )
       
        with col5:
            size_options = ['None'] + df.select_dtypes(include=[np.number]).columns.tolist()
            size_by = st.selectbox(
                "Size By",
                size_options,
                index=size_options.index('diff_factor') if 'diff_factor' in size_options else 1
            )
       
        if st.button("📊 Generate 3D Plot", type="primary"):
            with st.spinner("Generating 3D visualization..."):
                try:
                    fig = st.session_state.enhanced_sunburst_manager.create_3d_scatter_plot(
                        df=df,
                        x_col=x_axis,
                        y_col=y_axis,
                        z_col=z_axis,
                        color_col=color_by,
                        size_col=size_by if size_by != 'None' else None,
                        title=f"3D Analysis: {x_axis} vs {y_axis} vs {z_axis}"
                    )
                   
                    st.plotly_chart(fig, use_container_width=True)
                   
                    # Add PCA analysis
                    with st.expander("🧮 PCA Analysis", expanded=False):
                        # Select numerical columns for PCA
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        pca_cols = st.multiselect(
                            "Select columns for PCA",
                            numeric_cols,
                            default=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts', 'eps0', 'kappa']
                        )
                       
                        if len(pca_cols) >= 2:
                            pca_data = df[pca_cols].dropna()
                            if len(pca_data) > 1:
                                # Standardize data
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(pca_data)
                               
                                # Perform PCA
                                pca = PCA(n_components=min(3, len(pca_cols)))
                                pca_result = pca.fit_transform(scaled_data)
                               
                                # Create PCA DataFrame
                                pca_df = pd.DataFrame(
                                    data=pca_result,
                                    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
                                )
                                pca_df['defect_type'] = df.loc[pca_data.index, 'defect_type'].values if 'defect_type' in df.columns else 'Unknown'
                               
                                # Plot PCA results
                                if pca_result.shape[1] >= 3:
                                    fig_pca = px.scatter_3d(
                                        pca_df,
                                        x='PC1',
                                        y='PC2',
                                        z='PC3',
                                        color='defect_type',
                                        title='PCA Analysis (3D)',
                                        labels={
                                            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                                            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'
                                        }
                                    )
                                    st.plotly_chart(fig_pca, use_container_width=True)
                                else:
                                    fig_pca = px.scatter(
                                        pca_df,
                                        x='PC1',
                                        y='PC2',
                                        color='defect_type',
                                        title='PCA Analysis (2D)',
                                        labels={
                                            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
                                        }
                                    )
                                    st.plotly_chart(fig_pca, use_container_width=True)
                               
                                # Show explained variance
                                exp_var_df = pd.DataFrame({
                                    'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                    'Explained Variance': pca.explained_variance_ratio_,
                                    'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
                                })
                               
                                st.dataframe(exp_var_df.style.format({
                                    'Explained Variance': '{:.2%}',
                                    'Cumulative Variance': '{:.2%}'
                                }))
               
                except Exception as e:
                    st.error(f"Error generating 3D plot: {str(e)}")
   
    # Tab 4: Parallel Coordinates
    with viz_tabs[3]:
        st.subheader("🔗 Parallel Coordinates Analysis")
       
        # Select dimensions for parallel coordinates
        all_cols = df.columns.tolist()
        dimension_options = [col for col in all_cols if df[col].nunique() > 1]
       
        selected_dimensions = st.multiselect(
            "Select Dimensions for Parallel Coordinates",
            dimension_options,
            default=['defect_type', 'shape', 'eps0', 'kappa', 'max_von_mises', 'diff_factor', 'Ts']
        )
       
        color_options = ['defect_type', 'shape', 'orientation', 'type'] + df.select_dtypes(include=[np.number]).columns.tolist()
        color_by = st.selectbox(
            "Color Dimension",
            color_options,
            index=color_options.index('defect_type') if 'defect_type' in color_options else 0
        )
       
        if len(selected_dimensions) >= 2:
            if st.button("🔗 Generate Parallel Coordinates", type="primary"):
                with st.spinner("Generating parallel coordinates plot..."):
                    try:
                        # Create parallel coordinates plot
                        fig = st.session_state.enhanced_sunburst_manager.create_parallel_coordinates(
                            df=df,
                            dimensions=selected_dimensions,
                            color_column=color_by,
                            title="Multi-Dimensional Analysis"
                        )
                       
                        st.plotly_chart(fig, use_container_width=True)
                       
                        # Add correlation matrix
                        with st.expander("📈 Correlation Matrix", expanded=True):
                            numeric_dimensions = [dim for dim in selected_dimensions if dim in df.select_dtypes(include=[np.number]).columns]
                           
                            if len(numeric_dimensions) >= 2:
                                corr_matrix = df[numeric_dimensions].corr()
                               
                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=True,
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r',
                                    title="Correlation Matrix",
                                    zmin=-1,
                                    zmax=1
                                )
                               
                                st.plotly_chart(fig_corr, use_container_width=True)
                               
                                # Highlight strong correlations
                                strong_corrs = []
                                for i in range(len(corr_matrix.columns)):
                                    for j in range(i+1, len(corr_matrix.columns)):
                                        corr_val = corr_matrix.iloc[i, j]
                                        if abs(corr_val) > 0.7:
                                            strong_corrs.append({
                                                'Variables': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                                                'Correlation': corr_val
                                            })
                               
                                if strong_corrs:
                                    st.markdown("**Strong Correlations (|r| > 0.7):**")
                                    st.dataframe(pd.DataFrame(strong_corrs).sort_values('Correlation', key=abs, ascending=False))
                   
                    except Exception as e:
                        st.error(f"Error generating parallel coordinates: {str(e)}")
        else:
            st.warning("Please select at least 2 dimensions for parallel coordinates")
   
    # Tab 5: Statistical Analysis
    with viz_tabs[4]:
        st.subheader("📈 Advanced Statistical Analysis")
       
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Descriptive Statistics", "Regression Analysis", "ANOVA", "Time Series Analysis", "Cluster Analysis"]
        )
       
        if analysis_type == "Descriptive Statistics":
            st.markdown("### 📊 Descriptive Statistics")
           
            # Select columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_stats_cols = st.multiselect(
                "Select columns for statistical analysis",
                numeric_cols,
                default=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts', 'eps0', 'kappa']
            )
           
            if selected_stats_cols:
                # Calculate statistics
                stats_df = df[selected_stats_cols].describe().T
                stats_df['skewness'] = df[selected_stats_cols].skew()
                stats_df['kurtosis'] = df[selected_stats_cols].kurtosis()
                stats_df['cv'] = stats_df['std'] / stats_df['mean'] # Coefficient of variation
               
                st.dataframe(stats_df.style.format("{:.4f}"))
               
                # Create distribution plots
                st.markdown("### 📈 Distribution Plots")
               
                plot_cols = st.columns(min(3, len(selected_stats_cols)))
               
                for idx, col in enumerate(selected_stats_cols[:9]): # Limit to 9 plots
                    with plot_cols[idx % 3]:
                        fig = px.histogram(
                            df,
                            x=col,
                            nbins=30,
                            title=f"Distribution of {col}",
                            marginal="box",
                            color_discrete_sequence=['steelblue']
                        )
                        st.plotly_chart(fig, use_container_width=True)
       
        elif analysis_type == "Regression Analysis":
            st.markdown("### 📈 Regression Analysis")
           
            # Select variables for regression
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
           
            col1, col2 = st.columns(2)
           
            with col1:
                independent_vars = st.multiselect(
                    "Independent Variables (X)",
                    numeric_cols,
                    default=['eps0', 'kappa']
                )
           
            with col2:
                dependent_var = st.selectbox(
                    "Dependent Variable (Y)",
                    numeric_cols,
                    index=numeric_cols.index('max_von_mises') if 'max_von_mises' in numeric_cols else 0
                )
           
            if independent_vars and dependent_var:
                # Simple linear regression visualization
                for indep_var in independent_vars:
                    if indep_var in df.columns and dependent_var in df.columns:
                        fig = px.scatter(
                            df,
                            x=indep_var,
                            y=dependent_var,
                            trendline="ols",
                            title=f"Regression: {dependent_var} vs {indep_var}",
                            hover_name='id' if 'id' in df.columns else None
                        )
                       
                        # Get regression results
                        results = px.get_trendline_results(fig)
                        if not results.empty:
                            model = results.iloc[0]["px_fit_results"]
                            r_squared = model.rsquared
                           
                            fig.update_layout(
                                annotations=[
                                    dict(
                                        x=0.05,
                                        y=0.95,
                                        xref="paper",
                                        yref="paper",
                                        text=f"R² = {r_squared:.4f}",
                                        showarrow=False,
                                        font=dict(size=14)
                                    )
                                ]
                            )
                       
                        st.plotly_chart(fig, use_container_width=True)
       
        elif analysis_type == "ANOVA":
            st.markdown("### 📊 Analysis of Variance (ANOVA)")
           
            # Select categorical and numerical variables
            categorical_cols = ['defect_type', 'shape', 'orientation', 'type']
            available_categorical = [col for col in categorical_cols if col in df.columns]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
           
            if available_categorical and numeric_cols:
                cat_var = st.selectbox("Categorical Variable", available_categorical)
                num_var = st.selectbox("Numerical Variable", numeric_cols)
               
                if cat_var and num_var:
                    # Calculate group statistics
                    group_stats = df.groupby(cat_var)[num_var].agg(['mean', 'std', 'count']).round(4)
                    group_stats['ci_95'] = 1.96 * group_stats['std'] / np.sqrt(group_stats['count'])
                   
                    st.markdown(f"**Group Statistics for {num_var} by {cat_var}:**")
                    st.dataframe(group_stats)
                   
                    # Create ANOVA visualization
                    fig = px.box(
                        df,
                        x=cat_var,
                        y=num_var,
                        title=f"ANOVA: {num_var} by {cat_var}",
                        points="all"
                    )
                   
                    st.plotly_chart(fig, use_container_width=True)
   
    # Tab 6: Data Export
    with viz_tabs[5]:
        st.subheader("💾 Advanced Data Export")
       
        # Export options
        export_options = st.multiselect(
            "Select Data to Export",
            ["Stress Summary Data", "Simulation Data", "Visualizations", "Model Parameters"],
            default=["Stress Summary Data"]
        )
       
        # Export format selection
        st.markdown("### 📁 Export Formats")
       
        format_cols = st.columns(4)
       
        with format_cols[0]:
            csv_export = st.checkbox("CSV", value=True)
        with format_cols[1]:
            excel_export = st.checkbox("Excel", value=True)
        with format_cols[2]:
            json_export = st.checkbox("JSON", value=True)
        with format_cols[3]:
            parquet_export = st.checkbox("Parquet", value=False)
       
        # Additional formats
        format_cols2 = st.columns(4)
        with format_cols2[0]:
            pickle_export = st.checkbox("Pickle", value=False)
        with format_cols2[1]:
            feather_export = st.checkbox("Feather", value=False)
        with format_cols2[2]:
            html_export = st.checkbox("HTML", value=False)
        with format_cols2[3]:
            bundle_export = st.checkbox("Complete Bundle", value=True)
       
        # Export configuration
        st.markdown("### ⚙️ Export Configuration")
       
        col_config1, col_config2 = st.columns(2)
       
        with col_config1:
            include_metadata = st.checkbox("Include Metadata", value=True)
            include_charts = st.checkbox("Include Charts", value=True)
       
        with col_config2:
            compress_data = st.checkbox("Compress Data", value=True)
            timestamp_export = st.checkbox("Add Timestamp", value=True)
       
        # Custom filename
        custom_filename = st.text_input(
            "Custom Filename (optional)",
            value=f"stress_analysis_export_{datetime.now().strftime('%Y%m%d')}"
        )
       
        # Prepare export data
        export_data = {}
        charts_to_export = []
       
        if "Stress Summary Data" in export_options and not df.empty:
            export_data['stress_summary'] = df
       
        if "Simulation Data" in export_options:
            # Gather simulation data from session state
            sim_data = {
                'source_simulations': st.session_state.get('source_simulations', []),
                'predictions': st.session_state.get('prediction_results', {}),
                'multi_target_predictions': st.session_state.get('multi_target_predictions', {})
            }
            export_data['simulation_data'] = sim_data
       
        if "Visualizations" in export_options:
            # Generate some charts for export
            try:
                # Create a sunburst chart
                sunburst_fig = st.session_state.enhanced_sunburst_manager.create_stress_diffusion_sunburst(
                    df=df,
                    title="Stress-Diffusion-Sintering Analysis"
                )
                charts_to_export.append(sunburst_fig)
               
                # Create a radar chart
                if 'defect_type' in df.columns and 'max_von_mises' in df.columns:
                    radar_fig = st.session_state.enhanced_sunburst_manager.create_radar_chart_comparison(
                        df=df,
                        categories=df['defect_type'].unique()[:5],
                        metrics=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts'],
                        group_by='defect_type',
                        title="Defect Type Comparison"
                    )
                    charts_to_export.append(radar_fig)
            except:
                pass
       
        # Export button
        if st.button("🚀 Generate Export", type="primary"):
            with st.spinner("Preparing export package..."):
                try:
                    # Create export bundle
                    if bundle_export and export_data.get('stress_summary') is not None:
                        bundle_bytes = st.session_state.export_manager.create_export_bundle(
                            df=export_data['stress_summary'],
                            simulation_data=export_data.get('simulation_data'),
                            charts=charts_to_export
                        )
                       
                        # Generate filename
                        filename = f"{custom_filename}_bundle.zip"
                       
                        # Create download button
                        st.download_button(
                            label="📥 Download Complete Bundle (ZIP)",
                            data=bundle_bytes,
                            file_name=filename,
                            mime="application/zip"
                        )
                   
                    # Individual format exports
                    if export_data.get('stress_summary') is not None:
                        st.markdown("### 📄 Individual Format Exports")
                       
                        export_cols = st.columns(4)
                       
                        if csv_export:
                            with export_cols[0]:
                                csv_data = st.session_state.export_manager.export_dataframe(
                                    export_data['stress_summary'], 'csv'
                                )
                                st.download_button(
                                    label="📥 CSV",
                                    data=csv_data,
                                    file_name=f"{custom_filename}.csv",
                                    mime="text/csv"
                                )
                       
                        if excel_export:
                            with export_cols[1]:
                                excel_data = st.session_state.export_manager.export_dataframe(
                                    export_data['stress_summary'], 'excel'
                                )
                                st.download_button(
                                    label="📥 Excel",
                                    data=excel_data,
                                    file_name=f"{custom_filename}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                       
                        if json_export:
                            with export_cols[2]:
                                json_data = st.session_state.export_manager.export_dataframe(
                                    export_data['stress_summary'], 'json'
                                )
                                st.download_button(
                                    label="📥 JSON",
                                    data=json_data,
                                    file_name=f"{custom_filename}.json",
                                    mime="application/json"
                                )
                       
                        if parquet_export:
                            with export_cols[3]:
                                parquet_data = st.session_state.export_manager.export_dataframe(
                                    export_data['stress_summary'], 'parquet'
                                )
                                st.download_button(
                                    label="📥 Parquet",
                                    data=parquet_data,
                                    file_name=f"{custom_filename}.parquet",
                                    mime="application/octet-stream"
                                )
                   
                    st.success("✅ Export package ready! Click the download buttons above.")
                   
                    # Show export summary
                    with st.expander("📋 Export Summary", expanded=True):
                        summary_data = {
                            'Data Type': [],
                            'Rows': [],
                            'Columns': [],
                            'Size (approx)': []
                        }
                       
                        if export_data.get('stress_summary') is not None:
                            summary_data['Data Type'].append('Stress Summary')
                            summary_data['Rows'].append(len(export_data['stress_summary']))
                            summary_data['Columns'].append(len(export_data['stress_summary'].columns))
                            summary_data['Size (approx)'].append(f"{export_data['stress_summary'].memory_usage(deep=True).sum() / 1024:.1f} KB")
                       
                        if export_data.get('simulation_data'):
                            summary_data['Data Type'].append('Simulation Data')
                            summary_data['Rows'].append('Variable')
                            summary_data['Columns'].append('Multiple')
                            summary_data['Size (approx)'].append('Variable')
                       
                        if charts_to_export:
                            summary_data['Data Type'].append('Visualizations')
                            summary_data['Rows'].append(len(charts_to_export))
                            summary_data['Columns'].append('-')
                            summary_data['Size (approx)'].append(f"{len(charts_to_export)} charts")
                       
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
               
                except Exception as e:
                    st.error(f"❌ Error during export: {str(e)}")
# =============================================
# UPDATE THE MAIN FUNCTION TO INCLUDE ENHANCED VISUALIZATION
# =============================================
def main():
    """Main application with enhanced stress analysis and visualization"""
   
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
        ["Attention Interpolation", "Stress Analysis Dashboard", "Enhanced Visualization"],
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
       
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
       
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
       
        all_files = st.session_state.solutions_manager.get_all_files()
       
        if st.button("📥 Load All Simulations for Analysis"):
            with st.spinner("Loading all simulations..."):
                all_simulations = []
                for file_info in all_files[:50]:
                    sim_data = st.session_state.solutions_manager.load_simulation(
                        file_info['path'],
                        st.session_state.interpolator
                    )
                    if sim_data is not None:
                        all_simulations.append(sim_data)
               
                if all_simulations:
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
       
        if not st.session_state.stress_summary_df.empty:
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
            create_attention_interface()
        else:
            st.info("Please load simulations first to enable the stress analysis dashboard")
   
    else: # Enhanced Visualization Mode
        create_enhanced_visualization_interface()
# =============================================
# UPDATE THEORETICAL ANALYSIS SECTION
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
   
    ### **🌀 Enhanced Sunburst Chart Features**
   
    **Multi-Metric Sunburst:**
    1. **Three-Panel Visualization:** Simultaneous display of Stress, Diffusion, and Sintering metrics
    2. **Interactive Comparison:** Click to drill down into hierarchical levels
    3. **Color-Coded Metrics:** Different colormaps for different physical phenomena
   
    **Advanced Chart Types:**
    1. **Radar Charts:** Multi-axis comparison of stress, diffusion, and sintering metrics
    2. **3D Scatter Plots:** Interactive 3D visualization with size and color encoding
    3. **Parallel Coordinates:** High-dimensional data visualization
    4. **PCA Analysis:** Dimensionality reduction for pattern discovery
   
    ### **📈 Statistical Analysis Suite**
   
    **Comprehensive Analysis Tools:**
    1. **Descriptive Statistics:** Mean, median, std, skewness, kurtosis
    2. **Regression Analysis:** Linear regression with R² values
    3. **ANOVA:** Analysis of variance between groups
    4. **Correlation Matrix:** Heatmap visualization of relationships
   
    ### **💾 Advanced Data Export**
   
    **Export Formats:**
    1. **Multiple Formats:** CSV, Excel, JSON, Parquet, Feather, Pickle, HTML
    2. **Complete Bundles:** ZIP files with data, metadata, and visualizations
    3. **Compression Options:** Optimized file sizes for large datasets
    4. **Custom Metadata:** Export summaries with statistical information
   
    ### **🔥 Sintering and Diffusion Mapping**
   
    **Physics-Based Transformations:**
    1. **Sintering Temperature:** `Ts ≈ Ts0 * (1 - (Ω * |σ_h|) / Qa)`
    2. **Diffusion Factor:** `D/D0 = exp(Ω * |σ_h| / (k_B * T))`
    3. **Parameter Customization:** Adjust Ts0, Qa, Ω, and T for different materials
   
    **Visualization Integration:**
    1. **Sunburst Charts:** Hierarchical visualization of Ts and diffusion factors
    2. **Radar Charts:** Multi-metric comparison including diffusion effects
    3. **3D Plots:** Explore relationships between stress, diffusion, and sintering
    4. **Statistical Analysis:** Quantify the impact of different defects on sintering
    """)
if __name__ == "__main__":
    main()
st.caption(f"🔬 Enhanced Multi-Target Spatial-Attention Stress Interpolation • Advanced Visualization Dashboard • Comprehensive Data Export • 2025")
