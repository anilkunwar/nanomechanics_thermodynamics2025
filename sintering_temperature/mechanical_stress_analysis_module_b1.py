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
import io
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_RESULTS_DIR = os.path.join(SCRIPT_DIR, "ml_results")
if not os.path.exists(ML_RESULTS_DIR):
    os.makedirs(ML_RESULTS_DIR, exist_ok=True)
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
            df['delta_Q'] = (Omega * df['sigma_h_abs_pa']) / 1000 # J to kJ
           
            # Sintering temperature approximation
            df['Ts'] = Ts0 * (1 - df['delta_Q'] / Qa)
           
            # Diffusion factor exp(delta_Q / (R T)), with delta_Q in kJ/mol, R in kJ/mol·K
            R_kj = R / 1000 # Convert R to kJ/mol·K
            df['diff_factor'] = np.exp(df['delta_Q'] / (R_kj * T))
       
        return df
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
                df.to_hdf(buffer, key='data', mode='w', format='table')
                buffer.seek(0)
               
            elif format_type == 'pickle':
                pickle.dump(df, buffer)
                buffer.seek(0)
               
            elif format_type == 'msgpack':
                # Convert DataFrame to dict for msgpack
                data_dict = df.to_dict(orient='records')
                buffer.write(msgpack.packb(data_dict))
                buffer.seek(0)
               
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
# NUMERICAL SOLUTIONS MANAGER
# =============================================
class NumericalSolutionsManager:
    def __init__(self, solutions_dir: str = ML_RESULTS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
   
    def _ensure_directory(self):
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
   
    def scan_directory(self) -> Dict[str, List[str]]:
        file_formats = {
            'pkl': [],
            'pt': [],
            'h5': [],
            'npz': [],
            'sql': [],
            'json': []
        }
       
        for format_type, extensions in [
            ('pkl', ['*.pkl', '*.pickle']),
            ('pt', ['*.pt', '*.pth']),
            ('h5', ['*.h5', '*.hdf5']),
            ('npz', ['*.npz']),
            ('sql', ['*.sql', '*.db']),
            ('json', ['*.json'])
        ]:
            for ext in extensions:
                pattern = os.path.join(self.solutions_dir, ext)
                files = glob.glob(pattern)
                if files:
                    files.sort(key=os.path.getmtime, reverse=True)
                    file_formats[format_type].extend(files)
       
        return file_formats
   
    def get_all_files(self) -> List[Dict[str, Any]]:
        all_files = []
        file_formats = self.scan_directory()
       
        for format_type, files in file_formats.items():
            for file_path in files:
                file_info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'format': format_type,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    'relative_path': os.path.relpath(file_path, self.solutions_dir)
                }
                all_files.append(file_info)
       
        all_files.sort(key=lambda x: x['filename'].lower())
        return all_files
   
    def get_file_by_name(self, filename: str) -> Optional[str]:
        for file_info in self.get_all_files():
            if file_info['filename'] == filename:
                return file_info['path']
        return None
   
    def load_simulation(self, file_path: str, interpolator) -> Dict[str, Any]:
        try:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if ext in ['pkl', 'pickle']:
                format_type = 'pkl'
            elif ext in ['pt', 'pth']:
                format_type = 'pt'
            elif ext in ['h5', 'hdf5']:
                format_type = 'h5'
            elif ext == 'npz':
                format_type = 'npz'
            elif ext in ['sql', 'db']:
                format_type = 'sql'
            elif ext == 'json':
                format_type = 'json'
            else:
                format_type = 'auto'
           
            with open(file_path, 'rb') as f:
                file_content = f.read()
           
            sim_data = interpolator.read_simulation_file(file_content, format_type)
            sim_data['loaded_from'] = 'ml_results'
            return sim_data
           
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            raise
   
    def save_simulation(self, data: Dict[str, Any], filename: str, format_type: str = 'pkl'):
        if not filename.endswith(f'.{format_type}'):
            filename = f"{filename}.{format_type}"
       
        file_path = os.path.join(self.solutions_dir, filename)
       
        try:
            if format_type == 'pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
           
            elif format_type == 'pt':
                torch.save(data, file_path)
           
            elif format_type == 'json':
                def convert_for_json(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    else:
                        return obj
               
                json_data = convert_for_json(data)
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
           
            else:
                st.warning(f"Format {format_type} not supported for saving")
                return False
           
            st.success(f"✅ Saved simulation to: {filename}")
            return True
           
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return False
# =============================================
# SPATIAL LOCALITY ATTENTION INTERPOLATOR (FOR LOADING)
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
       
        self.readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json
        }
   
    def _build_model(self):
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
   
    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        return pickle.load(buffer)
   
    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))
   
    def _read_h5(self, file_content):
        buffer = BytesIO(file_content)
        with h5py.File(buffer, 'r') as f:
            data = {}
            def read_h5_obj(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
                elif isinstance(obj, h5py.Group):
                    data[name] = {}
                    for key in obj.keys():
                        read_h5_obj(f"{name}/{key}", obj[key])
            for key in f.keys():
                read_h5_obj(key, f[key])
        return data
   
    def _read_npz(self, file_content):
        buffer = BytesIO(file_content)
        data = np.load(buffer, allow_pickle=True)
        return {key: data[key] for key in data.files}
   
    def _read_sql(self, file_content):
        with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
       
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
           
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
           
            data = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                data[table_name] = {
                    'columns': columns,
                    'rows': rows
                }
           
            conn.close()
            os.unlink(tmp_path)
            return data
        except Exception as e:
            os.unlink(tmp_path)
            raise e
   
    def _read_json(self, file_content):
        return json.loads(file_content.decode('utf-8'))
   
    def read_simulation_file(self, file_content, format_type='auto'):
        if format_type == 'auto':
            format_type = 'pkl'
       
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            return self._standardize_data(data, format_type, "loaded_file")
        else:
            raise ValueError(f"Unsupported format: {format_type}")
   
    def _standardize_data(self, data, format_type, file_path):
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path) if isinstance(file_path, str) else "loaded"
        }
       
        if format_type == 'pkl':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
               
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                        standardized['history'].append((eta, stresses))
       
        elif format_type == 'pt':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
               
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                       
                        if torch.is_tensor(eta):
                            eta = eta.numpy()
                       
                        stress_dict = {}
                        for key, value in stresses.items():
                            if torch.is_tensor(value):
                                stress_dict[key] = value.numpy()
                            else:
                                stress_dict[key] = value
                       
                        standardized['history'].append((eta, stress_dict))
       
        elif format_type == 'h5':
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            for key in data.keys():
                if 'history' in key.lower():
                    standardized['history'] = data[key]
                    break
       
        elif format_type == 'npz':
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            if 'history' in data:
                standardized['history'] = data['history']
       
        elif format_type == 'json':
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                standardized['history'] = data.get('history', [])
       
        return standardized
   
    def compute_parameter_vector(self, sim_data):
        params = sim_data.get('params', {})
       
        param_vector = []
        param_names = []
       
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        param_names.extend(['defect_ISF', 'defect_ESF', 'defect_Twin'])
       
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
        shape = params.get('shape', 'Square')
        param_vector.extend(shape_encoding.get(shape, [0, 0, 0, 0, 0]))
        param_names.extend(['shape_square', 'shape_horizontal', 'shape_vertical',
                           'shape_rectangle', 'shape_ellipse'])
       
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
       
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3) if eps0 is not None else 0.5
        param_vector.append(eps0_norm)
        param_names.append('eps0_norm')
       
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1) if kappa is not None else 0.5
        param_vector.append(kappa_norm)
        param_names.append('kappa_norm')
       
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi) if theta is not None else 0.0
        param_vector.append(theta_norm)
        param_names.append('theta_norm')
       
        orientation = params.get('orientation', 'Horizontal {111} (0°)')
        orientation_encoding = {
            'Horizontal {111} (0°)': [1, 0, 0, 0],
            'Tilted 30° (1¯10 projection)': [0, 1, 0, 0],
            'Tilted 60°': [0, 0, 1, 0],
            'Vertical {111} (90°)': [0, 0, 0, 1]
        }
       
        if orientation.startswith('Custom ('):
            param_vector.extend([0, 0, 0, 0]) 
        else:
            param_vector.extend(orientation_encoding.get(orientation, [0, 0, 0, 0]))
           
        param_names.extend(['orient_0deg', 'orient_30deg', 'orient_60deg', 'orient_90deg'])
       
        return np.array(param_vector, dtype=np.float32), param_names
   
    @staticmethod
    def get_orientation_from_angle(angle_deg: float) -> str:
        """Convert angle in degrees to orientation string with custom support"""
        if 0 <= angle_deg <= 15:
            return 'Horizontal {111} (0°)'
        elif 15 < angle_deg <= 45:
            return 'Tilted 30° (1¯10 projection)'
        elif 45 < angle_deg <= 75:
            return 'Tilted 60°'
        elif 75 < angle_deg <= 90:
            return 'Vertical {111} (90°)'
        else:
            angle_deg = angle_deg % 90
            return f"Custom ({angle_deg:.1f}°)"
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
   
    # Initialize stress analysis manager
    if 'stress_analyzer' not in st.session_state:
        st.session_state.stress_analyzer = StressAnalysisManager()
   
    # Initialize sunburst chart manager
    if 'sunburst_manager' not in st.session_state:
        st.session_state.sunburst_manager = SunburstChartManager()
   
    # Initialize numerical solutions manager
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager()
   
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
   
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.loaded_from_numerical = []
   
    # Initialize predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
   
    # Initialize stress summary DataFrame
    if 'stress_summary_df' not in st.session_state:
        st.session_state.stress_summary_df = pd.DataFrame()
   
    # Load data from ml_results
    st.subheader("📂 Load Results from ml_results Directory")
    st.info(f"Loading from: `{ML_RESULTS_DIR}`")
   
    file_formats = st.session_state.solutions_manager.scan_directory()
    all_files_info = st.session_state.solutions_manager.get_all_files()
   
    if not all_files_info:
        st.warning(f"No files found in `{ML_RESULTS_DIR}`")
    else:
        file_groups = {}
        for file_info in all_files_info:
            format_type = file_info['format']
            if format_type not in file_groups:
                file_groups[format_type] = []
            file_groups[format_type].append(file_info)
       
        for format_type, files in file_groups.items():
            with st.expander(f"{format_type.upper()} Files ({len(files)})", expanded=True):
                file_options = {}
                for file_info in files:
                    display_name = f"{file_info['filename']} ({file_info['size'] // 1024}KB)"
                    file_options[display_name] = file_info['path']
               
                selected_files = st.multiselect(
                    f"Select {format_type} files",
                    options=list(file_options.keys()),
                    key=f"select_analysis_{format_type}"
                )
               
                if selected_files:
                    if st.button(f"📥 Load Selected {format_type} Files", key=f"load_analysis_{format_type}"):
                        with st.spinner(f"Loading {len(selected_files)} files..."):
                            loaded_count = 0
                            for display_name in selected_files:
                                file_path = file_options[display_name]
                                try:
                                    sim_data = st.session_state.solutions_manager.load_simulation(
                                        file_path,
                                        st.session_state.interpolator
                                    )
                                   
                                    if file_path not in st.session_state.loaded_from_numerical:
                                        st.session_state.source_simulations.append(sim_data)
                                        st.session_state.loaded_from_numerical.append(file_path)
                                        loaded_count += 1
                                        st.success(f"✅ Loaded: {os.path.basename(file_path)}")
                                    else:
                                        st.warning(f"⚠️ Already loaded: {os.path.basename(file_path)}")
                                       
                                except Exception as e:
                                    st.error(f"❌ Error loading {os.path.basename(file_path)}: {str(e)}")
                           
                            if loaded_count > 0:
                                st.success(f"Successfully loaded {loaded_count} new files!")
                                st.rerun()
   
    # Display loaded simulations
    if st.session_state.source_simulations:
        st.subheader("📋 Loaded Results")
       
        summary_data = []
        for i, sim_data in enumerate(st.session_state.source_simulations):
            params = sim_data.get('params', {})
            metadata = sim_data.get('metadata', {})
            source = sim_data.get('loaded_from', 'unknown')
           
            summary_data.append({
                'ID': i+1,
                'Source': source,
                'Defect Type': params.get('defect_type', 'Unknown'),
                'Shape': params.get('shape', 'Unknown'),
                'Orientation': params.get('orientation', 'Unknown'),
                'ε*': params.get('eps0', 'Unknown'),
                'κ': params.get('kappa', 'Unknown'),
                'Frames': len(sim_data.get('history', [])),
                'Format': sim_data.get('format', 'Unknown')
            })
       
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
           
            # Clear button
            if st.button("🗑️ Clear All Loaded Results", type="secondary"):
                st.session_state.source_simulations = []
                st.session_state.loaded_from_numerical = []
                st.session_state.stress_summary_df = pd.DataFrame()
                st.success("All loaded results cleared!")
                st.rerun()
   
    # Generate stress summary
    if st.session_state.source_simulations:
        if st.button("🔄 Generate Stress Summary", type="primary"):
            with st.spinner("Generating stress summary..."):
                st.session_state.stress_summary_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                    st.session_state.source_simulations, st.session_state.multi_target_predictions
                )
                st.success("✅ Stress summary generated!")
   
    # Sintering metrics
    if not st.session_state.stress_summary_df.empty:
        st.subheader("🔥 Compute Sintering Metrics")
       
        col1, col2 = st.columns(2)
       
        with col1:
            Ts0 = st.number_input("Zero-stress Ts0 (K)", value=623.0, step=1.0)
            Qa = st.number_input("Activation energy Qa (kJ/mol)", value=90.0, step=1.0)
       
        with col2:
            Omega = st.number_input("Activation volume Ω (m³/mol)", value=6e-6, format="%.2e")
            T = st.number_input("Temperature T (K)", value=623.0, step=1.0)
       
        R = 8.314  # J/mol·K
       
        if st.button("🧮 Compute Metrics"):
            st.session_state.stress_summary_df = st.session_state.stress_analyzer.compute_sintering_metrics(
                st.session_state.stress_summary_df, Ts0, Qa, Omega, T, R
            )
            st.success("✅ Metrics computed!")
       
        st.subheader("📋 Stress Summary DataFrame")
        st.dataframe(st.session_state.stress_summary_df, use_container_width=True)
   
    # Enhanced visualization interface
    if not st.session_state.stress_summary_df.empty:
        # Create tabs for different visualization types
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
            "🌀 Multi-Metric Sunburst",
            "📡 Radar Charts",
            "📊 3D Visualization",
            "🔗 Parallel Coordinates",
            "📈 Statistical Analysis"
        ])
       
        # Tab 1: Multi-Metric Sunburst
        with viz_tab1:
            st.subheader("🌀 Multi-Metric Sunburst Analysis")
           
            col1, col2 = st.columns([2, 1])
           
            with col1:
                # Select metrics for comparison
                stress_metrics = [col for col in st.session_state.stress_summary_df.columns if 'stress' in col.lower() or 'von_mises' in col or 'hydro' in col]
                diffusion_metrics = [col for col in st.session_state.stress_summary_df.columns if 'diff' in col.lower() or 'diff_factor' in col]
                sintering_metrics = [col for col in st.session_state.stress_summary_df.columns if 'Ts' in col or 'sinter' in col.lower()]
               
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
                            df=st.session_state.stress_summary_df,
                            stress_metric=selected_stress,
                            diffusion_metric=selected_diffusion,
                            sintering_metric=selected_sintering,
                            title=chart_title
                        )
                        st.plotly_chart(fig, use_container_width=True)
                       
                        # Add insights
                        with st.expander("📋 Analysis Insights", expanded=True):
                            # Calculate correlations
                            if all(m in st.session_state.stress_summary_df.columns for m in [selected_stress, selected_diffusion, selected_sintering]):
                                stress_diff_corr = st.session_state.stress_summary_df[selected_stress].corr(st.session_state.stress_summary_df[selected_diffusion])
                                stress_temp_corr = st.session_state.stress_summary_df[selected_stress].corr(st.session_state.stress_summary_df[selected_sintering])
                               
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
                            if 'defect_type' in st.session_state.stress_summary_df.columns:
                                defect_stats = st.session_state.stress_summary_df.groupby('defect_type').agg({
                                    selected_stress: 'mean',
                                    selected_diffusion: 'mean',
                                    selected_sintering: 'mean'
                                }).round(3)
                               
                                st.markdown("**Average Metrics by Defect Type:**")
                                st.dataframe(defect_stats)
                   
                    except Exception as e:
                        st.error(f"Error generating sunburst: {str(e)}")
       
        # Tab 2: Radar Charts
        with viz_tab2:
            st.subheader("📡 Radar Chart Analysis")
           
            col1, col2 = st.columns(2)
           
            with col1:
                # Select metrics for radar chart
                all_metrics = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                selected_metrics = st.multiselect(
                    "Select Metrics for Radar Axes",
                    all_metrics,
                    default=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts', 'eps0', 'kappa']
                )
               
                # Grouping options
                group_options = [col for col in st.session_state.stress_summary_df.columns if st.session_state.stress_summary_df[col].nunique() < 20]
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
                            unique_groups = st.session_state.stress_summary_df[selected_group].unique() if selected_group in st.session_state.stress_summary_df.columns else ['All']
                           
                            fig = st.session_state.enhanced_sunburst_manager.create_radar_chart_comparison(
                                df=st.session_state.stress_summary_df,
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
                                    if metric in st.session_state.stress_summary_df.columns:
                                        col_idx = idx % 3
                                        with stat_cols[col_idx]:
                                            st.metric(
                                                label=metric.replace('_', ' ').title(),
                                                value=f"{st.session_state.stress_summary_df[metric].mean():.3f}",
                                                delta=f"±{st.session_state.stress_summary_df[metric].std():.3f}"
                                            )
                           
                        except Exception as e:
                            st.error(f"Error generating radar chart: {str(e)}")
            else:
                st.warning("Please select at least 3 metrics for radar chart")
       
        # Tab 3: 3D Visualization
        with viz_tab3:
            st.subheader("📊 3D Visualization")
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                # X-axis selection
                x_options = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
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
                color_options = ['defect_type', 'shape', 'orientation', 'type'] + st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                color_by = st.selectbox(
                    "Color By",
                    color_options,
                    index=color_options.index('defect_type') if 'defect_type' in color_options else 0
                )
           
            with col5:
                size_options = ['None'] + st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                size_by = st.selectbox(
                    "Size By",
                    size_options,
                    index=size_options.index('diff_factor') if 'diff_factor' in size_options else 1
                )
           
            if st.button("📊 Generate 3D Plot", type="primary"):
                with st.spinner("Generating 3D visualization..."):
                    try:
                        fig = st.session_state.enhanced_sunburst_manager.create_3d_scatter_plot(
                            df=st.session_state.stress_summary_df,
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
                            numeric_cols = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                            pca_cols = st.multiselect(
                                "Select columns for PCA",
                                numeric_cols,
                                default=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts', 'eps0', 'kappa']
                            )
                           
                            if len(pca_cols) >= 2:
                                pca_data = st.session_state.stress_summary_df[pca_cols].dropna()
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
                                    pca_df['defect_type'] = st.session_state.stress_summary_df.loc[pca_data.index, 'defect_type'].values if 'defect_type' in st.session_state.stress_summary_df.columns else 'Unknown'
                                   
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
        with viz_tab4:
            st.subheader("🔗 Parallel Coordinates Analysis")
           
            # Select dimensions for parallel coordinates
            all_cols = st.session_state.stress_summary_df.columns.tolist()
            dimension_options = [col for col in all_cols if st.session_state.stress_summary_df[col].nunique() > 1]
           
            selected_dimensions = st.multiselect(
                "Select Dimensions for Parallel Coordinates",
                dimension_options,
                default=['defect_type', 'shape', 'eps0', 'kappa', 'max_von_mises', 'diff_factor', 'Ts']
            )
           
            color_options = ['defect_type', 'shape', 'orientation', 'type'] + st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
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
                                df=st.session_state.stress_summary_df,
                                dimensions=selected_dimensions,
                                color_column=color_by,
                                title="Multi-Dimensional Analysis"
                            )
                           
                            st.plotly_chart(fig, use_container_width=True)
                           
                            # Add correlation matrix
                            with st.expander("📈 Correlation Matrix", expanded=True):
                                numeric_dimensions = [dim for dim in selected_dimensions if dim in st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns]
                               
                                if len(numeric_dimensions) >= 2:
                                    corr_matrix = st.session_state.stress_summary_df[numeric_dimensions].corr()
                                   
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
        with viz_tab5:
            st.subheader("📈 Advanced Statistical Analysis")
           
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Descriptive Statistics", "Regression Analysis", "ANOVA", "Time Series Analysis", "Cluster Analysis"]
            )
           
            if analysis_type == "Descriptive Statistics":
                st.markdown("### 📊 Descriptive Statistics")
               
                # Select columns for analysis
                numeric_cols = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
                selected_stats_cols = st.multiselect(
                    "Select columns for statistical analysis",
                    numeric_cols,
                    default=['max_von_mises', 'max_abs_hydrostatic', 'diff_factor', 'Ts', 'eps0', 'kappa']
                )
               
                if selected_stats_cols:
                    # Calculate statistics
                    stats_df = st.session_state.stress_summary_df[selected_stats_cols].describe().T
                    stats_df['skewness'] = st.session_state.stress_summary_df[selected_stats_cols].skew()
                    stats_df['kurtosis'] = st.session_state.stress_summary_df[selected_stats_cols].kurtosis()
                    stats_df['cv'] = stats_df['std'] / stats_df['mean'] # Coefficient of variation
                   
                    st.dataframe(stats_df.style.format("{:.4f}"))
                   
                    # Create distribution plots
                    st.markdown("### 📈 Distribution Plots")
                   
                    plot_cols = st.columns(min(3, len(selected_stats_cols)))
                   
                    for idx, col in enumerate(selected_stats_cols[:9]): # Limit to 9 plots
                        with plot_cols[idx % 3]:
                            fig = px.histogram(
                                st.session_state.stress_summary_df,
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
                numeric_cols = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
               
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
                        if indep_var in st.session_state.stress_summary_df.columns and dependent_var in st.session_state.stress_summary_df.columns:
                            fig = px.scatter(
                                st.session_state.stress_summary_df,
                                x=indep_var,
                                y=dependent_var,
                                trendline="ols",
                                title=f"Regression: {dependent_var} vs {indep_var}",
                                hover_name='id' if 'id' in st.session_state.stress_summary_df.columns else None
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
                available_categorical = [col for col in categorical_cols if col in st.session_state.stress_summary_df.columns]
                numeric_cols = st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns.tolist()
               
                if available_categorical and numeric_cols:
                    cat_var = st.selectbox("Categorical Variable", available_categorical)
                    num_var = st.selectbox("Numerical Variable", numeric_cols)
                   
                    if cat_var and num_var:
                        # Calculate group statistics
                        group_stats = st.session_state.stress_summary_df.groupby(cat_var)[num_var].agg(['mean', 'std', 'count']).round(4)
                        group_stats['ci_95'] = 1.96 * group_stats['std'] / np.sqrt(group_stats['count'])
                       
                        st.markdown(f"**Group Statistics for {num_var} by {cat_var}:**")
                        st.dataframe(group_stats)
                       
                        # Create ANOVA visualization
                        fig = px.box(
                            st.session_state.stress_summary_df,
                            x=cat_var,
                            y=num_var,
                            title=f"ANOVA: {num_var} by {cat_var}",
                            points="all"
                        )
                       
                        st.plotly_chart(fig, use_container_width=True)
           
            # Add export section
            st.subheader("💾 Export Analysis")
           
            if st.button("📥 Export Current DataFrame as CSV"):
                csv_buffer = BytesIO()
                st.session_state.stress_summary_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer,
                    file_name="stress_analysis.csv",
                    mime="text/csv"
                )
else:
    st.info("👈 Load results from ml_results directory to enable analysis")
# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application for stress analysis"""
   
    st.sidebar.header("📁 Directory Information")
    st.sidebar.write(f"**App Directory:** `{SCRIPT_DIR}`")
    st.sidebar.write(f"**Results Directory:** `{ML_RESULTS_DIR}`")
   
    if not os.path.exists(ML_RESULTS_DIR):
        st.sidebar.warning("⚠️ Results directory not found")
        if st.sidebar.button("📁 Create Directory"):
            os.makedirs(ML_RESULTS_DIR, exist_ok=True)
            st.sidebar.success("✅ Directory created")
            st.rerun()
   
    create_enhanced_visualization_interface()

if __name__ == "__main__":
    main()
st.caption(f"🔬 Stress Analysis • Multi-Metric Sunburst • Radar Charts • 3D Visualization • Parallel Coordinates • Statistical Analysis • {datetime.now().year}")
