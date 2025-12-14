# part2_stress_analysis.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import h5py
import json
import os
import glob
from datetime import datetime
from io import BytesIO
import base64
import zipfile
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_RESULTS_DIR = os.path.join(SCRIPT_DIR, "ml_results")

# =============================================
# RESULTS LOADER
# =============================================
class ResultsLoader:
    """Load attention interpolation results from ml_results directory"""
    
    def __init__(self, results_dir=ML_RESULTS_DIR):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
    
    def list_available_results(self) -> List[Dict[str, Any]]:
        """List all available result files in ml_results directory"""
        result_files = []
        
        for ext_pattern in ['*.pkl', '*.h5', '*.npz', '*.parquet', '*.csv']:
            pattern = os.path.join(self.results_dir, ext_pattern)
            files = glob.glob(pattern)
            for file_path in files:
                file_info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'format': os.path.splitext(file_path)[1].lstrip('.'),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'size_mb': os.path.getsize(file_path) / (1024 * 1024)
                }
                result_files.append(file_info)
        
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: x['modified'], reverse=True)
        return result_files
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """Load results from a specific file"""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pkl':
                return self._load_pickle(file_path)
            elif ext == '.h5' or ext == '.hdf5':
                return self._load_hdf5(file_path)
            elif ext == '.npz':
                return self._load_npz(file_path)
            elif ext == '.parquet':
                return self._load_parquet(file_path)
            elif ext == '.csv':
                return self._load_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")
    
    def _load_pickle(self, file_path: str) -> Dict[str, Any]:
        """Load results from pickle file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Ensure consistent structure
        results = {
            'source_simulations': data.get('source_simulations', []),
            'predictions': data.get('predictions', {}),
            'stress_summary_df': data.get('stress_summary_df', pd.DataFrame()),
            'metadata': data.get('metadata', {}),
            'loaded_from': file_path,
            'loaded_at': datetime.now().isoformat()
        }
        
        return results
    
    def _load_hdf5(self, file_path: str) -> Dict[str, Any]:
        """Load results from HDF5 file"""
        results = {
            'source_simulations': [],
            'predictions': {},
            'stress_summary_df': pd.DataFrame(),
            'metadata': {},
            'loaded_from': file_path,
            'loaded_at': datetime.now().isoformat()
        }
        
        with h5py.File(file_path, 'r') as f:
            # Load metadata
            if 'metadata' in f.attrs:
                results['metadata'] = dict(f.attrs)
            
            # Load stress summary if available
            if 'stress_summary' in f:
                stress_data = {}
                for col in f['stress_summary'].keys():
                    stress_data[col] = f[f'stress_summary/{col}'][()]
                
                if stress_data:
                    results['stress_summary_df'] = pd.DataFrame(stress_data)
        
        return results
    
    def _load_npz(self, file_path: str) -> Dict[str, Any]:
        """Load results from NPZ file"""
        data = np.load(file_path, allow_pickle=True)
        
        results = {
            'source_simulations': data['source_simulations'].tolist() if 'source_simulations' in data else [],
            'predictions': data['predictions'].tolist() if 'predictions' in data else {},
            'metadata': {},
            'loaded_from': file_path,
            'loaded_at': datetime.now().isoformat()
        }
        
        # Reconstruct stress summary DataFrame
        if 'stress_summary_data' in data and 'stress_summary_columns' in data:
            columns = data['stress_summary_columns'].tolist()
            values = data['stress_summary_data']
            if len(values) > 0:
                results['stress_summary_df'] = pd.DataFrame(values, columns=columns)
        
        return results
    
    def _load_parquet(self, file_path: str) -> Dict[str, Any]:
        """Load results from Parquet file"""
        results = {
            'source_simulations': [],
            'predictions': {},
            'stress_summary_df': pd.read_parquet(file_path),
            'metadata': {},
            'loaded_from': file_path,
            'loaded_at': datetime.now().isoformat()
        }
        return results
    
    def _load_csv(self, file_path: str) -> Dict[str, Any]:
        """Load results from CSV file"""
        results = {
            'source_simulations': [],
            'predictions': {},
            'stress_summary_df': pd.read_csv(file_path),
            'metadata': {},
            'loaded_from': file_path,
            'loaded_at': datetime.now().isoformat()
        }
        return results

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
                json_str = df.to_json(orient='records', date_format='iso', default_handler=str)
                buffer.write(json_str.encode('utf-8'))
                buffer.seek(0)
                
            elif format_type == 'parquet':
                df.to_parquet(buffer, index=False, compression='snappy')
                buffer.seek(0)
                
            elif format_type == 'pickle':
                pickle.dump(df, buffer)
                buffer.seek(0)
                
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Error exporting data to {format_type}: {str(e)}")
        
        return buffer.getvalue()
    
    @staticmethod
    def create_export_bundle(df: pd.DataFrame, 
                           charts: List[go.Figure] = None,
                           metadata: Dict[str, Any] = None) -> bytes:
        """
        Create a comprehensive export bundle with data, metadata, and charts
        
        Args:
            df: Main DataFrame
            charts: List of Plotly figures to include
            metadata: Additional metadata
            
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
                
                # Export charts as HTML and PNG
                if charts:
                    for i, chart in enumerate(charts):
                        # Export as HTML
                        html_str = chart.to_html(include_plotlyjs='cdn')
                        zip_file.writestr(f'charts/chart_{i}.html', html_str)
                
                # Add metadata file
                metadata = metadata or {}
                metadata['export_timestamp'] = datetime.now().isoformat()
                metadata['data_shape'] = df.shape
                metadata['columns'] = list(df.columns)
                
                metadata_bytes = json.dumps(metadata, indent=2).encode('utf-8')
                zip_file.writestr('metadata.json', metadata_bytes)
                
                # Add README file
                readme_content = f"""
                Export Bundle - Machine Learning Data Analysis
                ==============================================
                
                Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Contents:
                1. data/ - Main dataset in multiple formats (CSV, Excel, JSON, Parquet)
                2. charts/ - Visualizations (HTML)
                3. metadata.json - Dataset metadata and statistics
                
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
# STRESS ANALYSIS EXTENSIONS
# =============================================
class StressAnalysisExtensions:
    """Extended stress analysis methods"""
    
    @staticmethod
    def compute_sintering_metrics(df: pd.DataFrame, Ts0: float, Qa: float, Omega: float, T: float, R: float) -> pd.DataFrame:
        """
        Compute sintering temperature and diffusion factor based on hydrostatic stress.
        
        Args:
            df: Stress summary DataFrame
            Ts0: Zero-stress sintering temperature (K)
            Qa: Activation energy (kJ/mol)
            Omega: Activation volume (m³/mol)
            T: Temperature for diffusion calculation (K)
            R: Gas constant (J/mol·K)
        
        Returns:
            Updated DataFrame with 'Ts' and 'diff_factor' columns
        """
        if 'max_abs_hydrostatic' in df.columns:
            # Convert stress from GPa to Pa
            df['sigma_h_abs_pa'] = df['max_abs_hydrostatic'] * 1e9
            
            # Compute delta_Q in kJ/mol
            df['delta_Q'] = (Omega * df['sigma_h_abs_pa']) / 1000  # J to kJ
            
            # Sintering temperature approximation
            df['Ts'] = Ts0 * (1 - df['delta_Q'] / Qa)
            
            # Diffusion factor exp(delta_Q / (R T)), with delta_Q in kJ/mol, R in kJ/mol·K
            R_kj = R / 1000  # Convert R to kJ/mol·K
            df['diff_factor'] = np.exp(df['delta_Q'] / (R_kj * T))
        
        return df
    
    @staticmethod
    def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for numerical columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            return df[numeric_cols].corr()
        return pd.DataFrame()

# =============================================
# STREAMLIT DASHBOARD
# =============================================
def create_dashboard():
    """Create Streamlit dashboard for stress analysis"""
    
    st.set_page_config(
        page_title="Stress Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Stress Analysis Dashboard")
    st.markdown("---")
    
    # Initialize session state
    if 'results_loader' not in st.session_state:
        st.session_state.results_loader = ResultsLoader()
    
    if 'enhanced_sunburst_manager' not in st.session_state:
        st.session_state.enhanced_sunburst_manager = EnhancedSunburstChartManager()
    
    if 'export_manager' not in st.session_state:
        st.session_state.export_manager = EnhancedDataExportManager()
    
    if 'stress_analyzer' not in st.session_state:
        st.session_state.stress_analyzer = StressAnalysisExtensions()
    
    # Sidebar: Load Results
    with st.sidebar:
        st.header("📁 Load Results")
        
        # List available results
        result_files = st.session_state.results_loader.list_available_results()
        
        if not result_files:
            st.warning(f"No result files found in {ML_RESULTS_DIR}")
            st.info("Run Part 1 to generate results first")
        else:
            # Create selection options
            file_options = {f"{f['filename']} ({f['size_mb']:.2f} MB, {f['modified'].strftime('%Y-%m-%d %H:%M')})": f['path'] 
                          for f in result_files}
            
            selected_file = st.selectbox(
                "Select result file to load:",
                options=list(file_options.keys()),
                index=0
            )
            
            if st.button("📥 Load Selected File", type="primary"):
                with st.spinner("Loading results..."):
                    try:
                        file_path = file_options[selected_file]
                        results = st.session_state.results_loader.load_results(file_path)
                        
                        # Store in session state
                        st.session_state.loaded_results = results
                        st.session_state.stress_summary_df = results.get('stress_summary_df', pd.DataFrame())
                        st.session_state.metadata = results.get('metadata', {})
                        
                        st.success(f"✅ Loaded {selected_file}")
                        
                        # Show metadata
                        with st.expander("📋 File Metadata", expanded=False):
                            st.json(results.get('metadata', {}))
                            
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        # Sintering parameters
        st.subheader("🔥 Sintering Parameters")
        Ts0 = st.number_input("Ts0 (Zero-stress sintering temp, K)", value=623.0, step=1.0)
        Qa = st.number_input("Qa (Activation energy, kJ/mol)", value=90.0, step=1.0)
        Omega = st.number_input("Ω (Activation volume, m³/mol)", value=6e-6, format="%.2e")
        T_diff = st.number_input("T (Diffusion temp, K)", value=623.0, step=1.0)
        R = 8.314  # J/mol·K
        
        if st.button("🧮 Compute Sintering Metrics"):
            if 'stress_summary_df' in st.session_state and not st.session_state.stress_summary_df.empty:
                with st.spinner("Computing sintering metrics..."):
                    st.session_state.stress_summary_df = st.session_state.stress_analyzer.compute_sintering_metrics(
                        st.session_state.stress_summary_df, Ts0, Qa, Omega, T_diff, R
                    )
                st.success("✅ Sintering metrics computed!")
    
    # Main content area
    if 'stress_summary_df' not in st.session_state or st.session_state.stress_summary_df.empty:
        st.info("👈 Please load results from the sidebar to begin analysis")
        
        # Show available files
        if result_files:
            st.subheader("📋 Available Result Files")
            file_info = pd.DataFrame([{
                'Filename': f['filename'],
                'Size (MB)': f'{f["size_mb"]:.2f}',
                'Modified': f['modified'].strftime('%Y-%m-%d %H:%M'),
                'Format': f['format']
            } for f in result_files])
            st.dataframe(file_info, use_container_width=True)
    else:
        df = st.session_state.stress_summary_df
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entries", len(df))
        with col2:
            defect_types = df['defect_type'].nunique() if 'defect_type' in df.columns else 0
            st.metric("Defect Types", defect_types)
        with col3:
            max_vm = df['max_von_mises'].max() if 'max_von_mises' in df.columns else 0
            st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
        with col4:
            avg_vm = df['max_von_mises'].mean() if 'max_von_mises' in df.columns else 0
            st.metric("Avg Von Mises", f"{avg_vm:.2f} GPa")
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs([
            "📋 Data Overview",
            "🌀 Multi-Metric Sunburst",
            "📡 Radar Charts",
            "📊 3D Visualization",
            "🔗 Parallel Coordinates",
            "📈 Statistical Analysis",
            "💾 Data Export"
        ])
        
        # Tab 1: Data Overview
        with viz_tabs[0]:
            st.subheader("📋 Data Overview")
            
            # Show DataFrame
            st.dataframe(
                df.style.format({
                    col: "{:.3f}" for col in df.select_dtypes(include=[np.number]).columns
                }),
                use_container_width=True,
                height=400
            )
            
            # Basic statistics
            st.subheader("📊 Basic Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_df = df[numeric_cols].describe().T
                st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)
        
        # Tab 2: Multi-Metric Sunburst
        with viz_tabs[1]:
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
                        
                    except Exception as e:
                        st.error(f"Error generating sunburst: {str(e)}")
        
        # Tab 3: Radar Charts
        with viz_tabs[2]:
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
            
            if selected_metrics and len(selected_metrics) >= 3:
                if st.button("📡 Generate Radar Chart", type="primary"):
                    with st.spinner("Generating radar chart..."):
                        try:
                            # Get unique groups
                            unique_groups = df[selected_group].unique() if selected_group in df.columns else ['All']
                            
                            fig = st.session_state.enhanced_sunburst_manager.create_radar_chart_comparison(
                                df=df,
                                categories=unique_groups[:6],  # Limit to 6 groups for clarity
                                metrics=selected_metrics,
                                group_by=selected_group,
                                title=radar_title
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating radar chart: {str(e)}")
            else:
                st.warning("Please select at least 3 metrics for radar chart")
        
        # Tab 4: 3D Visualization
        with viz_tabs[3]:
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
                        
                    except Exception as e:
                        st.error(f"Error generating 3D plot: {str(e)}")
        
        # Tab 5: Parallel Coordinates
        with viz_tabs[4]:
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
                            fig = st.session_state.enhanced_sunburst_manager.create_parallel_coordinates(
                                df=df,
                                dimensions=selected_dimensions,
                                color_column=color_by,
                                title="Multi-Dimensional Analysis"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating parallel coordinates: {str(e)}")
            else:
                st.warning("Please select at least 2 dimensions for parallel coordinates")
        
        # Tab 6: Statistical Analysis
        with viz_tabs[5]:
            st.subheader("📈 Advanced Statistical Analysis")
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Descriptive Statistics", "Correlation Analysis", "Distribution Analysis", "PCA Analysis"]
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
                    stats_df['cv'] = stats_df['std'] / stats_df['mean']  # Coefficient of variation
                    
                    st.dataframe(stats_df.style.format("{:.4f}"))
                    
                    # Create distribution plots
                    st.markdown("### 📈 Distribution Plots")
                    
                    plot_cols = st.columns(min(3, len(selected_stats_cols)))
                    
                    for idx, col in enumerate(selected_stats_cols[:9]):
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
            
            elif analysis_type == "Correlation Analysis":
                st.markdown("### 📊 Correlation Analysis")
                
                # Compute correlation matrix
                corr_matrix = st.session_state.stress_analyzer.compute_correlations(df)
                
                if not corr_matrix.empty:
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix",
                        zmin=-1,
                        zmax=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strong correlations
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
            
            elif analysis_type == "PCA Analysis":
                st.markdown("### 🧮 Principal Component Analysis (PCA)")
                
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
        
        # Tab 7: Data Export
        with viz_tabs[6]:
            st.subheader("💾 Data Export")
            
            # Export options
            export_formats = st.multiselect(
                "Select Export Formats",
                ["CSV", "Excel", "JSON", "Parquet", "Complete Bundle (ZIP)"],
                default=["CSV", "Complete Bundle (ZIP)"]
            )
            
            # Custom filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"stress_analysis_{timestamp}"
            custom_filename = st.text_input(
                "Custom Filename (without extension)",
                value=default_filename
            )
            
            # Generate charts for export
            charts_to_export = []
            if st.checkbox("Include visualizations in export", value=True):
                # Generate a sunburst chart
                try:
                    sunburst_fig = st.session_state.enhanced_sunburst_manager.create_stress_diffusion_sunburst(
                        df=df,
                        title="Stress-Diffusion-Sintering Analysis"
                    )
                    charts_to_export.append(sunburst_fig)
                except:
                    pass
            
            if st.button("🚀 Generate Export", type="primary"):
                with st.spinner("Preparing export..."):
                    try:
                        # Individual format exports
                        for fmt in export_formats:
                            if fmt == "CSV":
                                csv_data = st.session_state.export_manager.export_dataframe(df, 'csv')
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"{custom_filename}.csv",
                                    mime="text/csv"
                                )
                            
                            elif fmt == "Excel":
                                excel_data = st.session_state.export_manager.export_dataframe(df, 'excel')
                                st.download_button(
                                    label="📥 Download Excel",
                                    data=excel_data,
                                    file_name=f"{custom_filename}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            elif fmt == "JSON":
                                json_data = st.session_state.export_manager.export_dataframe(df, 'json')
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_data,
                                    file_name=f"{custom_filename}.json",
                                    mime="application/json"
                                )
                            
                            elif fmt == "Parquet":
                                parquet_data = st.session_state.export_manager.export_dataframe(df, 'parquet')
                                st.download_button(
                                    label="📥 Download Parquet",
                                    data=parquet_data,
                                    file_name=f"{custom_filename}.parquet",
                                    mime="application/octet-stream"
                                )
                            
                            elif fmt == "Complete Bundle (ZIP)":
                                bundle_data = st.session_state.export_manager.create_export_bundle(
                                    df=df,
                                    charts=charts_to_export,
                                    metadata=st.session_state.get('metadata', {})
                                )
                                st.download_button(
                                    label="📥 Download Complete Bundle (ZIP)",
                                    data=bundle_data,
                                    file_name=f"{custom_filename}_bundle.zip",
                                    mime="application/zip"
                                )
                        
                        st.success("✅ Export ready! Click the download buttons above.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during export: {str(e)}")
        
        # Footer
        st.markdown("---")
        st.caption(f"🔬 Stress Analysis Dashboard • Data loaded from: {ML_RESULTS_DIR} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    create_dashboard()
