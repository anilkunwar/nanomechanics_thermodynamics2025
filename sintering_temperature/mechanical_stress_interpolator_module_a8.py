import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
import warnings
import pickle
import torch
import sqlite3
from pathlib import Path
import tempfile
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
warnings.filterwarnings('ignore')
# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
# =============================================
# CUSTOM COLORMAPS
# =============================================
def create_custom_colormaps():
    """Create custom colormaps for stress visualization"""
    # Stress colormap (blue to red)
    stress_cmap = LinearSegmentedColormap.from_list(
        'stress_cmap',
        ['#00008B', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#8B0000']
    )
  
    # Attention weight colormap
    attention_cmap = LinearSegmentedColormap.from_list(
        'attention_cmap',
        ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    )
  
    # Comparison colormap (diverging)
    compare_cmap = LinearSegmentedColormap.from_list(
        'compare_cmap',
        ['#0066CC', '#66CCFF', '#FFFFFF', '#FF9999', '#CC0000']
    )
  
    return stress_cmap, attention_cmap, compare_cmap
# =============================================
# VISUALIZATION MANAGER
# =============================================
class VisualizationManager:
    """Manager for creating and exporting visualizations"""
  
    def __init__(self):
        self.stress_cmap, self.attention_cmap, self.compare_cmap = create_custom_colormaps()
        self.output_dir = VISUALIZATION_OUTPUT_DIR
        self._ensure_directories()
  
    def _ensure_directories(self):
        """Create necessary directories"""
        subdirs = ['stress_fields', 'attention_plots', 'comparisons', 'animations']
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
  
    def create_stress_field_plot_matplotlib(self, stress_data, title, component_name,
                                          extent=None, vmin=None, vmax=None,
                                          include_contour=True, include_colorbar=True):
        """
        Create matplotlib plot for stress field
      
        Args:
            stress_data: 2D numpy array
            title: Plot title
            component_name: Name of stress component
            extent: Plot extent [xmin, xmax, ymin, ymax]
            vmin/vmax: Color limits
            include_contour: Whether to add contour lines
            include_colorbar: Whether to add colorbar
          
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
      
        if extent is None:
            extent = [-64, 64, -64, 64]
      
        if vmin is None:
            vmin = np.nanmin(stress_data)
        if vmax is None:
            vmax = np.nanmax(stress_data)
      
        # Create heatmap
        im = ax.imshow(stress_data,
                      cmap=self.stress_cmap,
                      extent=extent,
                      origin='lower',
                      aspect='equal',
                      vmin=vmin,
                      vmax=vmax)
      
        # Add contour lines
        if include_contour and not np.all(stress_data == stress_data[0,0]):
            try:
                X, Y = np.meshgrid(np.linspace(extent[0], extent[1], stress_data.shape[1]),
                                  np.linspace(extent[2], extent[3], stress_data.shape[0]))
              
                levels = np.linspace(vmin, vmax, 12)
                contour = ax.contour(X, Y, stress_data,
                                    levels=levels,
                                    colors='black',
                                    linewidths=0.5,
                                    alpha=0.7)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
            except:
                pass
      
        # Add colorbar
        if include_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Stress (GPa)', rotation=270, labelpad=15)
      
        # Set labels and title
        ax.set_xlabel('x (nm)', fontsize=12)
        ax.set_ylabel('y (nm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
      
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
      
        plt.tight_layout()
        return fig
  
    def create_attention_weights_plot_matplotlib(self, weights, source_names=None,
                                               title="Attention Weights Distribution"):
        """
        Create matplotlib bar plot for attention weights
      
        Args:
            weights: Array of attention weights
            source_names: List of source names
            title: Plot title
          
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
      
        if source_names is None:
            source_names = [f'Source {i+1}' for i in range(len(weights))]
      
        # Create bar plot
        x_pos = np.arange(len(weights))
        bars = ax.bar(x_pos, weights,
                     color=self.attention_cmap(np.linspace(0, 1, len(weights))),
                     edgecolor='black',
                     linewidth=1,
                     alpha=0.8)
      
        # Add value labels on top of bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
      
        # Add percentage labels
        total = np.sum(weights)
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            percentage = (weight / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., -0.02,
                   f'{percentage:.1f}%', ha='center', va='top', fontsize=8)
      
        # Customize plot
        ax.set_xlabel('Source Simulations', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(source_names, rotation=45, ha='right')
        ax.set_ylim([0, max(weights) * 1.3])
      
        # Add horizontal line for average
        avg_weight = np.mean(weights)
        ax.axhline(y=avg_weight, color='red', linestyle='--', alpha=0.7,
                  label=f'Average: {avg_weight:.3f}')
        ax.legend()
      
        # Add grid
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
      
        plt.tight_layout()
        return fig
  
    def create_training_loss_plot_matplotlib(self, losses, title="Training Loss Curve"):
        """
        Create matplotlib line plot for training losses
      
        Args:
            losses: Array of training losses
            title: Plot title
          
        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
      
        epochs = range(1, len(losses) + 1)
      
        # Plot loss curve
        line = ax.plot(epochs, losses,
                      color='#C73E1D',
                      linewidth=2,
                      marker='o',
                      markersize=4,
                      markerfacecolor='white',
                      markeredgecolor='#C73E1D',
                      markeredgewidth=1)
      
        # Add smoothing line
        if len(losses) > 5:
            window_size = max(3, len(losses) // 10)
            smooth_losses = pd.Series(losses).rolling(window=window_size,
                                                     center=True,
                                                     min_periods=1).mean()
            ax.plot(epochs, smooth_losses,
                   color='#2E86AB',
                   linewidth=2,
                   linestyle='--',
                   alpha=0.7,
                   label='Smoothed')
      
        # Customize plot
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
      
        # Add annotation for final loss
        final_loss = losses[-1]
        ax.annotate(f'Final: {final_loss:.4f}',
                   xy=(epochs[-1], final_loss),
                   xytext=(epochs[-1] * 0.7, losses[0] * 0.8),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
      
        plt.tight_layout()
        return fig
  
    def create_multi_frame_comparison(self, frames_data, titles, component_name,
                                    nrows=2, ncols=3, figsize=(20, 12)):
        """
        Create multi-frame comparison plot
      
        Args:
            frames_data: List of 2D numpy arrays for each frame
            titles: List of titles for each frame
            component_name: Stress component name
            nrows, ncols: Grid dimensions
            figsize: Figure size
          
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=figsize,
                                dpi=150,
                                constrained_layout=True)
      
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()
      
        # Find global min/max for consistent color scaling
        all_data = np.concatenate([d.flatten() for d in frames_data if d is not None])
        vmin, vmax = np.nanpercentile(all_data, [1, 99])
      
        for idx, (ax, data, title) in enumerate(zip(axes, frames_data, titles)):
            if data is None or idx >= len(frames_data):
                ax.axis('off')
                continue
              
            im = ax.imshow(data,
                          cmap=self.stress_cmap,
                          extent=[-64, 64, -64, 64],
                          origin='lower',
                          aspect='equal',
                          vmin=vmin,
                          vmax=vmax)
          
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('x (nm)')
            ax.set_ylabel('y (nm)')
            ax.grid(True, alpha=0.2)
          
            # Add frame number
            ax.text(0.02, 0.98, f'Frame {idx+1}',
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
      
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'{component_name} (GPa)', rotation=270, labelpad=20)
      
        fig.suptitle(f'{component_name.replace("_", " ").title()} - Frame Comparison',
                    fontsize=16, fontweight='bold', y=1.02)
      
        return fig
  
    def save_figure(self, fig, filename, format='png', dpi=300):
        """
        Save matplotlib figure to file
      
        Args:
            fig: matplotlib figure
            filename: Output filename
            format: Image format
            dpi: Resolution
          
        Returns:
            filepath
        """
        if not filename.endswith(f'.{format}'):
            filename = f'{filename}.{format}'
      
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return filepath
  
    def get_image_download_link(self, fig, filename, caption="Download Image"):
        """
        Generate download link for matplotlib figure
      
        Args:
            fig: matplotlib figure
            filename: Download filename
            caption: Link caption
          
        Returns:
            HTML download link
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
      
        b64 = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{caption}</a>'
        return href
# =============================================
# TIME FRAME VISUALIZATION MANAGER
# =============================================
class TimeFrameVisualizationManager:
    """Manager for visualizing all time frames from simulations"""
  
    def __init__(self, vis_manager=None):
        self.vis_manager = vis_manager or VisualizationManager()
  
    def extract_time_frames(self, simulation_data, max_frames=10):
        """
        Extract time frames from simulation history
      
        Args:
            simulation_data: Simulation data dictionary
            max_frames: Maximum number of frames to extract
          
        Returns:
            Dictionary with frames for each stress component
        """
        history = simulation_data.get('history', [])
        if not history:
            return {}
      
        # Limit number of frames
        n_frames = min(len(history), max_frames)
        frame_indices = np.linspace(0, len(history)-1, n_frames, dtype=int)
      
        frames = {
            'sigma_hydro': [],
            'sigma_mag': [],
            'von_mises': [],
            'time_points': [],
            'frame_numbers': []
        }
      
        for idx in frame_indices:
            eta, stress_fields = history[idx]
            frames['sigma_hydro'].append(stress_fields.get('sigma_hydro', np.zeros_like(eta)))
            frames['sigma_mag'].append(stress_fields.get('sigma_mag', np.zeros_like(eta)))
            frames['von_mises'].append(stress_fields.get('von_mises', np.zeros_like(eta)))
            frames['time_points'].append(idx) # Use index as time proxy
            frames['frame_numbers'].append(idx + 1)
      
        return frames
  
    def create_time_series_plot(self, frames_data, component_name,
                              title="Time Evolution of Stress Field"):
        """
        Create time series plot showing stress evolution
      
        Args:
            frames_data: Dictionary with frames data
            component_name: Stress component name
            title: Plot title
          
        Returns:
            matplotlib figure
        """
        component_frames = frames_data.get(component_name, [])
        frame_numbers = frames_data.get('frame_numbers', [])
      
        if not component_frames:
            return None
      
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150,
                                constrained_layout=True)
        axes = axes.flatten()
      
        # Get consistent color scaling
        all_data = np.concatenate([f.flatten() for f in component_frames])
        vmin, vmax = np.nanpercentile(all_data, [1, 99])
      
        # Plot selected frames
        selected_frames = np.linspace(0, len(component_frames)-1, 6, dtype=int)
      
        for idx, (frame_idx, ax) in enumerate(zip(selected_frames, axes)):
            if frame_idx < len(component_frames):
                data = component_frames[frame_idx]
                frame_num = frame_numbers[frame_idx]
              
                im = ax.imshow(data,
                              cmap=self.vis_manager.stress_cmap,
                              extent=[-64, 64, -64, 64],
                              origin='lower',
                              aspect='equal',
                              vmin=vmin,
                              vmax=vmax)
              
                ax.set_title(f'Frame {frame_num}', fontsize=10, fontweight='bold')
                ax.set_xlabel('x (nm)')
                ax.set_ylabel('y (nm)')
                ax.grid(True, alpha=0.2)
      
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'{component_name} (GPa)', rotation=270, labelpad=20)
      
        fig.suptitle(f'{title}\n{component_name.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
      
        return fig
  
    def create_stress_evolution_metrics(self, frames_data):
        """
        Create plots showing stress evolution metrics
      
        Args:
            frames_data: Dictionary with frames data
          
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150,
                                constrained_layout=True)
        axes = axes.flatten()
      
        frame_numbers = frames_data.get('frame_numbers', [])
      
        # Plot 1: Maximum stress evolution
        ax1 = axes[0]
        for comp_name, comp_data in [('von_mises', 'Von Mises'),
                                     ('sigma_hydro', 'Hydrostatic'),
                                     ('sigma_mag', 'Magnitude')]:
            if comp_name in frames_data:
                max_stress = [np.max(frame) for frame in frames_data[comp_name]]
                ax1.plot(frame_numbers, max_stress, 'o-', label=comp_data, linewidth=2)
      
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Maximum Stress (GPa)')
        ax1.set_title('Maximum Stress Evolution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
      
        # Plot 2: Mean stress evolution
        ax2 = axes[1]
        for comp_name, comp_data in [('von_mises', 'Von Mises'),
                                     ('sigma_hydro', 'Hydrostatic'),
                                     ('sigma_mag', 'Magnitude')]:
            if comp_name in frames_data:
                mean_stress = [np.mean(frame) for frame in frames_data[comp_name]]
                ax2.plot(frame_numbers, mean_stress, 's-', label=comp_data, linewidth=2)
      
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Mean Stress (GPa)')
        ax2.set_title('Mean Stress Evolution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
      
        # Plot 3: Stress volume above threshold
        ax3 = axes[2]
        threshold = 0.5 # 0.5 GPa threshold
        for comp_name, comp_data in [('von_mises', 'Von Mises'),
                                     ('sigma_hydro', 'Hydrostatic'),
                                     ('sigma_mag', 'Magnitude')]:
            if comp_name in frames_data:
                volume_above = [np.sum(frame > threshold) for frame in frames_data[comp_name]]
                ax3.plot(frame_numbers, volume_above, '^-', label=comp_data, linewidth=2)
      
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel(f'Pixels > {threshold} GPa')
        ax3.set_title(f'High Stress Volume Evolution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
      
        # Plot 4: Stress distribution width
        ax4 = axes[3]
        for comp_name, comp_data in [('von_mises', 'Von Mises'),
                                     ('sigma_hydro', 'Hydrostatic'),
                                     ('sigma_mag', 'Magnitude')]:
            if comp_name in frames_data:
                std_stress = [np.std(frame) for frame in frames_data[comp_name]]
                ax4.plot(frame_numbers, std_stress, 'D-', label=comp_data, linewidth=2)
      
        ax4.set_xlabel('Frame Number')
        ax4.set_ylabel('Standard Deviation (GPa)')
        ax4.set_title('Stress Distribution Width', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
      
        fig.suptitle('Stress Evolution Metrics Analysis', fontsize=14, fontweight='bold')
      
        return fig
# =============================================
# PREDICTION RESULTS SAVING AND DOWNLOAD MANAGER
# =============================================
class PredictionResultsManager:
    """Manager for saving and downloading prediction results"""
  
    @staticmethod
    def prepare_prediction_data_for_saving(prediction_results: Dict[str, Any],
                                         source_simulations: List[Dict]) -> Dict[str, Any]:
        """
        Prepare prediction results for saving to file
      
        Args:
            prediction_results: Dictionary of prediction results
            source_simulations: List of source simulation data
          
        Returns:
            Structured dictionary ready for saving
        """
        # Create metadata
        metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'num_sources': len(source_simulations),
            'software_version': '1.0.0',
            'data_type': 'attention_interpolation_results'
        }
      
        # Extract source parameters
        source_params = []
        for i, sim_data in enumerate(source_simulations):
            params = sim_data.get('params', {})
            source_params.append({
                'id': i,
                'defect_type': params.get('defect_type'),
                'shape': params.get('shape'),
                'orientation': params.get('orientation'),
                'eps0': float(params.get('eps0', 0)),
                'kappa': float(params.get('kappa', 0)),
                'theta': float(params.get('theta', 0))
            })
      
        # Structure the data
        save_data = {
            'metadata': metadata,
            'source_parameters': source_params,
            'prediction_results': prediction_results.copy() # Create a copy to avoid modifications
        }
      
        if 'attention_weights' in prediction_results:
            weights = prediction_results['attention_weights']
            save_data['attention_analysis'] = {
                'weights': weights.tolist() if hasattr(weights, 'tolist') else weights,
                'source_names': [f'S{i+1}' for i in range(len(source_simulations))],
                'dominant_source': int(np.argmax(weights)) if hasattr(weights, '__len__') else 0,
                'weight_entropy': float(-np.sum(weights * np.log(weights + 1e-10)))
            }
      
        # Add stress statistics if available
        if 'stress_fields' in prediction_results:
            stress_stats = {}
            for field_name, field_data in prediction_results['stress_fields'].items():
                if isinstance(field_data, np.ndarray):
                    stress_stats[field_name] = {
                        'max': float(np.max(field_data)),
                        'min': float(np.min(field_data)),
                        'mean': float(np.mean(field_data)),
                        'std': float(np.std(field_data)),
                        'percentile_95': float(np.percentile(field_data, 95)),
                        'percentile_99': float(np.percentile(field_data, 99))
                    }
            save_data['stress_statistics'] = stress_stats
      
        return save_data
  
    @staticmethod
    def create_single_prediction_archive(prediction_results: Dict[str, Any],
                                       source_simulations: List[Dict]) -> BytesIO:
        """
        Create a comprehensive archive for single prediction
      
        Args:
            prediction_results: Single prediction results
            source_simulations: List of source simulations
          
        Returns:
            BytesIO buffer containing the archive
        """
        # Create in-memory zip file
        zip_buffer = BytesIO()
      
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save main prediction data as PKL
            save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                prediction_results, source_simulations
            )
          
            # PKL format
            pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
            zip_file.writestr('prediction_results.pkl', pkl_data)
          
            # 2. Save as PT (PyTorch) format
            pt_buffer = BytesIO()
            torch.save(save_data, pt_buffer)
            pt_buffer.seek(0)
            zip_file.writestr('prediction_results.pt', pt_buffer.read())
          
            # 3. Save stress fields as separate NPZ files
            stress_fields = prediction_results.get('stress_fields', {})
            for field_name, field_data in stress_fields.items():
                if isinstance(field_data, np.ndarray):
                    npz_buffer = BytesIO()
                    np.savez_compressed(npz_buffer, data=field_data)
                    npz_buffer.seek(0)
                    zip_file.writestr(f'stress_{field_name}.npz', npz_buffer.read())
          
            # 4. Save attention weights as CSV
            if 'attention_weights' in prediction_results:
                weights = prediction_results['attention_weights']
                if hasattr(weights, 'flatten'):
                    weights = weights.flatten()
              
                weight_df = pd.DataFrame({
                    'source_id': [f'S{i+1}' for i in range(len(weights))],
                    'weight': weights,
                    'percent_contribution': 100 * weights / (np.sum(weights) + 1e-10)
                })
                csv_data = weight_df.to_csv(index=False)
                zip_file.writestr('attention_weights.csv', csv_data)
          
            # 5. Save target parameters as JSON
            target_params = prediction_results.get('target_params', {})
            if target_params:
                # Convert numpy types to Python types for JSON
                def convert_for_json(obj):
                    if isinstance(obj, (np.float32, np.float64, np.float16)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    else:
                        return obj
              
                json_data = json.dumps(target_params, default=convert_for_json, indent=2)
                zip_file.writestr('target_parameters.json', json_data)
          
            # 6. Save summary statistics
            if 'stress_fields' in prediction_results:
                stats_rows = []
                for field_name, field_data in stress_fields.items():
                    if isinstance(field_data, np.ndarray):
                        stats_rows.append({
                            'field': field_name,
                            'max': float(np.max(field_data)),
                            'min': float(np.min(field_data)),
                            'mean': float(np.mean(field_data)),
                            'std': float(np.std(field_data)),
                            'percentile_95': float(np.percentile(field_data, 95)),
                            'percentile_99': float(np.percentile(field_data, 99)),
                            'area_above_threshold': float(np.sum(field_data > np.mean(field_data)))
                        })
              
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    stats_csv = stats_df.to_csv(index=False)
                    zip_file.writestr('stress_statistics.csv', stats_csv)
          
            # 7. Save a README file
            readme_content = f"""# Prediction Results Archive
Generated: {datetime.now().isoformat()}
Number of source simulations: {len(source_simulations)}
Prediction mode: Single target
Files included:
1. prediction_results.pkl - Main prediction data (Python pickle format)
2. prediction_results.pt - PyTorch format
3. stress_*.npz - Individual stress fields (NumPy compressed)
4. attention_weights.csv - Attention weights distribution
5. target_parameters.json - Target parameters
6. stress_statistics.csv - Statistical summary
For more information, see the documentation.
"""
            zip_file.writestr('README.txt', readme_content)
      
        zip_buffer.seek(0)
        return zip_buffer
# =============================================
# NUMERICAL SOLUTIONS MANAGER (UPDATED FOR NUMERICAL_SOLUTIONS)
# =============================================
class NumericalSolutionsManager:
    def __init__(self, solutions_dir: str = NUMERICAL_SOLUTIONS_DIR):
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
            sim_data['loaded_from'] = 'numerical_solutions'
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
            'spatial_regularizer': torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, self.num_heads)
            ) if self.use_gaussian else None,
            'norm1': torch.nn.LayerNorm(self.d_model),
        })
        return model

    def get_attention_weights(self, target_param, source_params):
        """Get attention weights between target and source parameters"""
        if source_params.size(0) == 0:
            raise ValueError("No source parameters provided for attention weights computation")
        
        # Prepare embeddings with correct dimensions
        target_embed = self.model.param_embedding(target_param).unsqueeze(1)  # (1, 1, d_model)
        source_embeds = self.model.param_embedding(source_params).unsqueeze(1)  # (N, 1, d_model)
        
        # Transpose for multi-head attention (batch_first=True expects [batch, seq_len, d_model])
        target_embed = target_embed.transpose(0, 1)  # (1, 1, d_model) -> (1, 1, d_model) no change needed
        source_embeds = source_embeds.transpose(0, 1)  # (N, 1, d_model) -> (1, N, d_model)
        
        # Compute attention
        _, attn_weights = self.model.attention(
            target_embed, 
            source_embeds, 
            source_embeds, 
            average_attn_weights=False
        )  # (num_heads, 1, N)
        
        # Average across heads to get single weight per source
        weights = attn_weights.mean(dim=0)  # (1, N)
        weights = weights.squeeze(0)  # (N,)
        
        # Normalize weights
        weights = torch.softmax(weights, dim=0)
        return weights

    def train(self, source_params, source_stress, epochs=50, lr=0.001):
        """Train the interpolator using leave-one-out cross-validation"""
        if source_params.size(0) < 2:
            raise ValueError("Need at least 2 source simulations for leave-one-out training")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        N = source_params.size(0)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(N):
                # Leave-one-out: use all except i-th as sources
                target_param = source_params[i].unsqueeze(0)  # (1, input_dim)
                target_stress = source_stress[i]  # (3, H, W)
                
                # Create mask for sources
                src_mask = torch.ones(N, dtype=torch.bool)
                src_mask[i] = False
                
                src_params = source_params[src_mask]  # (N-1, input_dim)
                src_stress = source_stress[src_mask]  # (N-1, 3, H, W)
                
                if len(src_params) < 1:
                    continue
                
                # Get attention weights
                weights = self.get_attention_weights(target_param, src_params)  # (N-1,)
                
                # Check dimensions before einsum
                if weights.dim() != 1:
                    raise ValueError(f"Expected 1D weights tensor, got shape {weights.shape}")
                if src_stress.dim() != 4:
                    raise ValueError(f"Expected 4D stress tensor, got shape {src_stress.shape}")
                if weights.size(0) != src_stress.size(0):
                    raise ValueError(f"Dimension mismatch: weights ({weights.shape}) vs stress ({src_stress.shape})")
                
                # Compute weighted sum: sum over n dimension
                predicted_stress = torch.einsum('n,nchw->chw', weights, src_stress)
                
                # Compute loss
                loss = torch.mean((predicted_stress - target_stress) ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if N > 0:
                losses.append(epoch_loss / N)
            else:
                losses.append(0.0)
        
        return losses

    def predict(self, target_param, source_params, source_stress):
        """Predict stress field for target parameters"""
        if source_params.size(0) == 0:
            raise ValueError("No source parameters provided for prediction")
        
        # Get attention weights
        weights = self.get_attention_weights(target_param, source_params)
        
        # Check dimension compatibility
        if weights.size(0) != source_stress.size(0):
            raise ValueError(
                f"Dimension mismatch: weights ({weights.shape[0]}) vs "
                f"stress ({source_stress.shape[0]}) sources"
            )
        
        # Compute weighted sum
        predicted_stress = torch.einsum('n,nchw->chw', weights, source_stress)
        
        return predicted_stress.numpy(), weights.numpy()

    def _read_pkl(self, file_content):
        buffer = BytesIO(file_content)
        return pickle.load(buffer)

    def _read_pt(self, file_content):
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))

    def _read_h5(self, file_content):
        import h5py
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
            # Try to auto-detect based on content or extension
            format_type = 'pkl'  # Default
      
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            return self._standardize_data(data, format_type, "uploaded_file")
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _standardize_data(self, data, format_type, file_path):
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path) if isinstance(file_path, str) else "uploaded"
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
        """Compute parameter vector from simulation data"""
        params = sim_data.get('params', {})
      
        param_vector = []
        param_names = []
      
        # Encode defect type (3 dimensions)
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        param_names.extend(['defect_ISF', 'defect_ESF', 'defect_Twin'])
      
        # Encode shape (5 dimensions)
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
      
        # Normalize continuous parameters
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
      
        # Encode orientation (4 dimensions)
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
# GRID AND EXTENT CONFIGURATION
# =============================================
def get_grid_extent(N=128, dx=0.1):
    """Get grid extent for visualization"""
    return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
# =============================================
# ATTENTION INTERFACE
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with save/download"""
  
    st.header("🤖 Spatial-Attention Stress Interpolation")
  
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
  
    # Initialize numerical solutions manager
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager()
  
    # Initialize prediction results manager
    if 'prediction_results_manager' not in st.session_state:
        st.session_state.prediction_results_manager = PredictionResultsManager()
  
    # Initialize visualization manager
    if 'visualization_manager' not in st.session_state:
        st.session_state.visualization_manager = VisualizationManager()
  
    # Initialize time frame visualization manager
    if 'time_frame_manager' not in st.session_state:
        st.session_state.time_frame_manager = TimeFrameVisualizationManager(
            st.session_state.visualization_manager
        )
  
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
  
    # Initialize saving options
    if 'save_format' not in st.session_state:
        st.session_state.save_format = 'both'
  
    # Initialize download data in session state
    if 'download_pkl_data' not in st.session_state:
        st.session_state.download_pkl_data = None
    if 'download_pt_data' not in st.session_state:
        st.session_state.download_pt_data = None
    if 'download_zip_data' not in st.session_state:
        st.session_state.download_zip_data = None
    if 'download_zip_filename' not in st.session_state:
        st.session_state.download_zip_filename = None
  
    # Initialize matplotlib figure storage
    if 'matplotlib_figures' not in st.session_state:
        st.session_state.matplotlib_figures = {}
  
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
  
    with st.sidebar.expander("🎨 Visualization Settings", expanded=False):
        viz_library = st.radio(
            "Primary Visualization Library",
            ["Plotly (Interactive)", "Matplotlib (Static)"],
            index=0,
            help="Choose between interactive Plotly charts or static Matplotlib figures"
        )
      
        default_colormap = st.selectbox(
            "Default Colormap",
            ["viridis", "plasma", "coolwarm", "RdBu", "Spectral", "custom_stress"],
            index=5,
            help="Colormap for stress field visualizations"
        )
      
        include_contours = st.checkbox("Include Contour Lines", value=True)
        include_grid = st.checkbox("Include Grid", value=True)
        figure_dpi = st.slider("Figure DPI", 100, 300, 150, 10)
  
    with st.sidebar.expander("💾 Download Options", expanded=True):
        st.session_state.save_format = st.radio(
            "Download Format",
            ["PKL only", "PT only", "Both PKL & PT", "Archive (ZIP)"],
            index=2,
            key="save_format_radio"
        )
      
        # Time frame download options
        st.markdown("---")
        st.markdown("**Time Frame Download**")
        time_frame_format = st.radio(
            "Time Frame Format",
            ["PNG", "PDF", "SVG", "All formats"],
            index=0,
            help="Format for downloading time frame visualizations"
        )
  
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Target",
        "🚀 Train & Predict",
        "📊 Results & Visualization",
        "⏱️ Time Frame Analysis",
        "💾 Export Results"
    ])
  
    # Tab 1: Load Source Data
    with tab1:
        st.subheader("Load Source Simulation Files")
      
        col1, col2 = st.columns([1, 1])
      
        with col1:
            st.markdown("### 📂 From numerical_solutions Directory")
            st.info(f"Loading from: `{NUMERICAL_SOLUTIONS_DIR}`")
          
            file_formats = st.session_state.solutions_manager.scan_directory()
            all_files_info = st.session_state.solutions_manager.get_all_files()
          
            if not all_files_info:
                st.warning(f"No simulation files found in `{NUMERICAL_SOLUTIONS_DIR}`")
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
                            key=f"select_{format_type}"
                        )
                      
                        if selected_files:
                            if st.button(f"📥 Load Selected {format_type} Files", key=f"load_{format_type}"):
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
  
        with col2:
            st.markdown("### 📤 Upload Local Files")
          
            uploaded_files = st.file_uploader(
                "Upload simulation files (PKL, PT, H5, NPZ, SQL, JSON)",
                type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'sql', 'db', 'json'],
                accept_multiple_files=True
            )
          
            format_type = st.selectbox(
                "File Format (for upload)",
                ["Auto Detect", "PKL", "PT", "H5", "NPZ", "SQL", "JSON"],
                index=0
            )
          
            if uploaded_files and st.button("📥 Load Uploaded Files", type="primary"):
                with st.spinner("Loading uploaded files..."):
                    loaded_sims = []
                    for uploaded_file in uploaded_files:
                        try:
                            file_content = uploaded_file.getvalue()
                            actual_format = format_type.lower() if format_type != "Auto Detect" else "auto"
                            if actual_format == "auto":
                                filename = uploaded_file.name.lower()
                                if filename.endswith('.pkl'):
                                    actual_format = 'pkl'
                                elif filename.endswith('.pt'):
                                    actual_format = 'pt'
                                elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                                    actual_format = 'h5'
                                elif filename.endswith('.npz'):
                                    actual_format = 'npz'
                                elif filename.endswith('.sql') or filename.endswith('.db'):
                                    actual_format = 'sql'
                                elif filename.endswith('.json'):
                                    actual_format = 'json'
                          
                            sim_data = st.session_state.interpolator.read_simulation_file(
                                file_content, actual_format
                            )
                            sim_data['loaded_from'] = 'upload'
                          
                            file_id = f"{uploaded_file.name}_{hashlib.md5(file_content).hexdigest()[:8]}"
                            st.session_state.uploaded_files[file_id] = {
                                'filename': uploaded_file.name,
                                'data': sim_data,
                                'format': actual_format
                            }
                          
                            st.session_state.source_simulations.append(sim_data)
                            loaded_sims.append(uploaded_file.name)
                          
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                  
                    if loaded_sims:
                        st.success(f"Successfully loaded {len(loaded_sims)} uploaded files!")
      
        # Display loaded simulations
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Source Simulations")
          
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
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Clear All Source Simulations", type="secondary"):
                        st.session_state.source_simulations = []
                        st.session_state.uploaded_files = {}
                        st.session_state.loaded_from_numerical = []
                        st.success("All source simulations cleared!")
                        st.rerun()
                with col2:
                    st.info(f"**Total loaded simulations:** {len(st.session_state.source_simulations)}")
  
    # Tab 2: Configure Target
    with tab2:
        st.subheader("Configure Target Parameters")
      
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            col1, col2 = st.columns(2)
          
            with col1:
                target_defect = st.selectbox(
                    "Target Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="target_defect"
                )
              
                target_shape = st.selectbox(
                    "Target Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="target_shape"
                )
              
                target_eps0 = st.slider(
                    "Target ε*",
                    0.3, 3.0, 1.414, 0.01,
                    key="target_eps0"
                )
          
            with col2:
                target_kappa = st.slider(
                    "Target κ",
                    0.1, 2.0, 0.7, 0.05,
                    key="target_kappa"
                )
              
                orientation_mode = st.radio(
                    "Orientation Mode",
                    ["Predefined", "Custom Angle"],
                    horizontal=True,
                    key="orientation_mode"
                )
              
                if orientation_mode == "Predefined":
                    target_orientation = st.selectbox(
                        "Target Orientation",
                        ["Horizontal {111} (0°)",
                         "Tilted 30° (1¯10 projection)",
                         "Tilted 60°",
                         "Vertical {111} (90°)"],
                        index=0,
                        key="target_orientation"
                    )
                  
                    angle_map = {
                        "Horizontal {111} (0°)": 0,
                        "Tilted 30° (1¯10 projection)": 30,
                        "Tilted 60°": 60,
                        "Vertical {111} (90°)": 90,
                    }
                    target_theta = np.deg2rad(angle_map.get(target_orientation, 0))
                    st.info(f"**Target θ:** {np.rad2deg(target_theta):.1f}°")
                  
                else:
                    target_angle = st.slider(
                        "Target Angle (degrees)",
                        0.0, 90.0, 0.0, 0.5,
                        key="target_angle_custom"
                    )
                    target_theta = np.deg2rad(target_angle)
                  
                    target_orientation = st.session_state.interpolator.get_orientation_from_angle(target_angle)
                    st.info(f"**Target θ:** {target_angle:.1f}°")
                    st.info(f"**Orientation:** {target_orientation}")
          
            target_params = {
                'defect_type': target_defect,
                'shape': target_shape,
                'eps0': target_eps0,
                'kappa': target_kappa,
                'orientation': target_orientation,
                'theta': target_theta
            }
          
            st.session_state.target_params = target_params
  
    # Tab 3: Train & Predict
    with tab3:
        st.subheader("Train Model and Predict")
      
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        elif 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure target parameters first")
        else:
            col1, col2 = st.columns(2)
          
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
          
            with col2:
                batch_size = st.slider("Batch Size", 1, 16, 4, 1)
                validation_split = st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05)
          
            if st.button("🚀 Train & Predict", type="primary"):
                with st.spinner("Training attention model and predicting..."):
                    try:
                        param_vectors = []
                        stress_data = []
                      
                        for sim_data in st.session_state.source_simulations:
                            param_vector, _ = st.session_state.interpolator.compute_parameter_vector(sim_data)
                            param_vectors.append(param_vector)
                          
                            history = sim_data.get('history', [])
                            if history:
                                eta, stress_fields = history[-1]
                                stress_components = np.stack([
                                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                                    stress_fields.get('von_mises', np.zeros_like(eta))
                                ], axis=0)
                                stress_data.append(stress_components)
                      
                        if len(param_vectors) < 1:
                            raise ValueError("No valid source simulations available for prediction")
                      
                        param_vectors = np.array(param_vectors)
                        stress_data = np.stack(stress_data) # (N, 3, H, W)
                      
                        source_params = torch.from_numpy(param_vectors).float()
                        source_stress = torch.from_numpy(stress_data).float()
                      
                        losses = st.session_state.interpolator.train(
                            source_params, source_stress, epochs=epochs, lr=learning_rate
                        )
                      
                        target_vector, _ = st.session_state.interpolator.compute_parameter_vector(
                            {'params': st.session_state.target_params}
                        )
                        target_param = torch.from_numpy(target_vector).float().unsqueeze(0)
                      
                        weights = st.session_state.interpolator.get_attention_weights(target_param, source_params)
                      
                        predicted_stress = torch.einsum('n,nchw->chw', weights, source_stress).numpy()
                      
                        predicted = {
                            'sigma_hydro': predicted_stress[0],
                            'sigma_mag': predicted_stress[1],
                            'von_mises': predicted_stress[2],
                            'predicted': True
                        }
                      
                        attention_weights = weights.numpy()
                      
                        st.session_state.prediction_results = {
                            'stress_fields': predicted,
                            'attention_weights': attention_weights,
                            'target_params': st.session_state.target_params,
                            'training_losses': losses,
                            'source_count': len(st.session_state.source_simulations),
                            'mode': 'single'
                        }
                      
                        st.success("✅ Training and prediction complete!")
                      
                    except Exception as e:
                        st.error(f"❌ Error during training/prediction: {str(e)}")
  
    # Tab 4: Results & Visualization
    with tab4:
        st.subheader("Prediction Results Visualization")
      
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            results = st.session_state.prediction_results
            stress_fields = results.get('stress_fields', {})
            attention_weights = results.get('attention_weights')
            target_params = results.get('target_params', {})
            training_losses = results.get('training_losses')
          
            # Visualization controls
            col_viz1, col_viz2, col_viz3, col_viz4 = st.columns(4)
            with col_viz1:
                stress_component = st.selectbox(
                    "Select Stress Component",
                    ['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0
                )
            with col_viz2:
                if viz_library == "Plotly (Interactive)":
                    colormap = st.selectbox(
                        "Colormap",
                        ['viridis', 'plasma', 'coolwarm', 'RdBu', 'Spectral'],
                        index=0
                    )
                else:
                    colormap = st.selectbox(
                        "Colormap",
                        ['custom_stress', 'viridis', 'plasma', 'coolwarm', 'RdBu', 'Spectral'],
                        index=0
                    )
            with col_viz3:
                show_contour = st.checkbox("Show Contour Lines", value=include_contours)
            with col_viz4:
                viz_option = st.radio(
                    "View",
                    ["Single View", "Comparison View"],
                    horizontal=True,
                    key="viz_option"
                )
          
            # Store matplotlib figures for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
          
            if stress_component in stress_fields:
                stress_data = stress_fields[stress_component]
              
                if viz_library == "Plotly (Interactive)":
                    # Interactive heatmap with Plotly
                    fig_heat = px.imshow(
                        stress_data,
                        color_continuous_scale=colormap,
                        origin='lower',
                        aspect='equal',
                        title=f'{stress_component.replace("_", " ").title()} (GPa) - Orientation: {target_params.get("orientation", "Unknown")}'
                    )
                    fig_heat.update_layout(
                        xaxis_title='x (nm)',
                        yaxis_title='y (nm)',
                        coloraxis_colorbar=dict(title='Stress (GPa)')
                    )
                  
                    # Add contours if enabled
                    if show_contour:
                        fig_heat.add_trace(
                            go.Contour(
                                z=stress_data,
                                showscale=False,
                                contours=dict(
                                    coloring='lines',
                                    showlabels=True,
                                    labelfont=dict(size=8)
                                ),
                                line=dict(color='black', width=0.5),
                                ncontours=10
                            )
                        )
                  
                    st.plotly_chart(fig_heat, use_container_width=True)
                  
                else:
                    # Static visualization with Matplotlib
                    title = f'{stress_component.replace("_", " ").title()} (GPa)\nOrientation: {target_params.get("orientation", "Unknown")}'
                  
                    fig_matplotlib = st.session_state.visualization_manager.create_stress_field_plot_matplotlib(
                        stress_data=stress_data,
                        title=title,
                        component_name=stress_component,
                        extent=extent,
                        include_contour=show_contour,
                        include_colorbar=True
                    )
                  
                    # Display the figure
                    st.pyplot(fig_matplotlib)
                  
                    # Store for download
                    st.session_state.matplotlib_figures['stress_field'] = fig_matplotlib
                  
                    # Download button for matplotlib figure
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        buf = BytesIO()
                        fig_matplotlib.savefig(buf, format="png", dpi=figure_dpi, bbox_inches='tight')
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf.getvalue(),
                            file_name=f"stress_field_{stress_component}_{timestamp}.png",
                            mime="image/png",
                            key=f"download_stress_png_{timestamp}"
                        )
                    with col_dl2:
                        buf = BytesIO()
                        fig_matplotlib.savefig(buf, format="pdf", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download PDF",
                            data=buf.getvalue(),
                            file_name=f"stress_field_{stress_component}_{timestamp}.pdf",
                            mime="application/pdf",
                            key=f"download_stress_pdf_{timestamp}"
                        )
                    with col_dl3:
                        buf = BytesIO()
                        fig_matplotlib.savefig(buf, format="svg", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download SVG",
                            data=buf.getvalue(),
                            file_name=f"stress_field_{stress_component}_{timestamp}.svg",
                            mime="image/svg+xml",
                            key=f"download_stress_svg_{timestamp}"
                        )
          
            # Attention weights visualization
            st.subheader("🔍 Attention Weights")
          
            if attention_weights is not None:
                weights = attention_weights
                source_names = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
              
                if viz_library == "Plotly (Interactive)":
                    fig_weights = px.bar(
                        x=source_names,
                        y=weights,
                        labels={'x': 'Source Simulations', 'y': 'Attention Weight'},
                        title='Attention Weights Distribution'
                    )
                    fig_weights.update_traces(marker_color='steelblue', opacity=0.7)
                    fig_weights.update_layout(yaxis_range=[0, max(weights) * 1.2])
                  
                    # Add text labels
                    for i, weight in enumerate(weights):
                        fig_weights.add_annotation(
                            x=source_names[i],
                            y=weight,
                            text=f'{weight:.3f}',
                            showarrow=False,
                            yshift=10,
                            font=dict(size=9)
                        )
                  
                    st.plotly_chart(fig_weights, use_container_width=True)
                else:
                    fig_weights = st.session_state.visualization_manager.create_attention_weights_plot_matplotlib(
                        weights=weights,
                        source_names=source_names,
                        title="Attention Weights Distribution"
                    )
                  
                    st.pyplot(fig_weights)
                  
                    # Store for download
                    st.session_state.matplotlib_figures['attention_weights'] = fig_weights
                  
                    # Download button for attention weights
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        buf = BytesIO()
                        fig_weights.savefig(buf, format="png", dpi=figure_dpi, bbox_inches='tight')
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf.getvalue(),
                            file_name=f"attention_weights_{timestamp}.png",
                            mime="image/png",
                            key=f"download_attention_png_{timestamp}"
                        )
                    with col_dl2:
                        buf = BytesIO()
                        fig_weights.savefig(buf, format="pdf", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download PDF",
                            data=buf.getvalue(),
                            file_name=f"attention_weights_{timestamp}.pdf",
                            mime="application/pdf",
                            key=f"download_attention_pdf_{timestamp}"
                        )
          
            # Training losses visualization
            st.subheader("📉 Training Losses")
            if training_losses is not None:
                if viz_library == "Plotly (Interactive)":
                    fig_losses = px.line(
                        x=range(len(training_losses)),
                        y=training_losses,
                        labels={'x': 'Epoch', 'y': 'Loss'},
                        title='Training Loss Curve'
                    )
                    fig_losses.update_traces(line_color='firebrick')
                    st.plotly_chart(fig_losses, use_container_width=True)
                else:
                    fig_losses = st.session_state.visualization_manager.create_training_loss_plot_matplotlib(
                        losses=training_losses,
                        title="Training Loss Curve"
                    )
                  
                    st.pyplot(fig_losses)
                  
                    # Store for download
                    st.session_state.matplotlib_figures['training_losses'] = fig_losses
                  
                    # Download button for training losses
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        buf = BytesIO()
                        fig_losses.savefig(buf, format="png", dpi=figure_dpi, bbox_inches='tight')
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf.getvalue(),
                            file_name=f"training_losses_{timestamp}.png",
                            mime="image/png",
                            key=f"download_losses_png_{timestamp}"
                        )
                    with col_dl2:
                        buf = BytesIO()
                        fig_losses.savefig(buf, format="pdf", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download PDF",
                            data=buf.getvalue(),
                            file_name=f"training_losses_{timestamp}.pdf",
                            mime="application/pdf",
                            key=f"download_losses_pdf_{timestamp}"
                        )
          
            # Statistics table
            st.subheader("📊 Stress Statistics")
          
            stats_data = []
            for comp_name, comp_data in stress_fields.items():
                if isinstance(comp_data, np.ndarray):
                    stats_data.append({
                        'Component': comp_name.replace('_', ' ').title(),
                        'Max (GPa)': float(np.max(comp_data)),
                        'Min (GPa)': float(np.min(comp_data)),
                        'Mean (GPa)': float(np.mean(comp_data)),
                        'Std Dev': float(np.std(comp_data)),
                        '95th %ile': float(np.percentile(comp_data, 95))
                    })
          
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats.style.format({
                    'Max (GPa)': '{:.3f}',
                    'Min (GPa)': '{:.3f}',
                    'Mean (GPa)': '{:.3f}',
                    'Std Dev': '{:.3f}',
                    '95th %ile': '{:.3f}'
                }), use_container_width=True)
              
                # Download statistics as CSV
                csv = df_stats.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics CSV",
                    data=csv,
                    file_name=f"stress_statistics_{timestamp}.csv",
                    mime="text/csv",
                    key=f"download_stats_{timestamp}"
                )
  
    # Tab 5: Time Frame Analysis
    with tab5:
        st.subheader("⏱️ Time Frame Analysis")
      
        if not st.session_state.source_simulations:
            st.warning("⚠️ Please load source simulations first")
        else:
            # Select source simulation for time frame analysis
            sim_options = {}
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                sim_name = f"Sim {i+1}: {params.get('defect_type', 'Unknown')} - {params.get('shape', 'Unknown')}"
                sim_options[sim_name] = i
          
            selected_sim_name = st.selectbox(
                "Select Source Simulation for Time Analysis",
                options=list(sim_options.keys()),
                index=0,
                key="time_sim_selector"
            )
          
            selected_sim_idx = sim_options[selected_sim_name]
            selected_sim = st.session_state.source_simulations[selected_sim_idx]
          
            # Extract time frames
            max_frames = st.slider("Maximum number of frames to analyze", 5, 50, 10, 1)
          
            if st.button("📊 Extract Time Frames", type="primary"):
                with st.spinner("Extracting time frames..."):
                    time_frames = st.session_state.time_frame_manager.extract_time_frames(
                        selected_sim, max_frames
                    )
                    st.session_state.time_frames = time_frames
                    st.success(f"✅ Extracted {len(time_frames.get('frame_numbers', []))} time frames!")
          
            if 'time_frames' in st.session_state and st.session_state.time_frames:
                time_frames = st.session_state.time_frames
              
                # Component selection
                time_component = st.selectbox(
                    "Select Stress Component for Time Analysis",
                    ['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0,
                    key="time_component"
                )
              
                # Display time series plot
                st.subheader("📈 Time Evolution")
              
                fig_time_series = st.session_state.time_frame_manager.create_time_series_plot(
                    time_frames, time_component
                )
              
                if fig_time_series:
                    st.pyplot(fig_time_series)
                  
                    # Download time series plot
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        buf = BytesIO()
                        fig_time_series.savefig(buf, format="png", dpi=figure_dpi, bbox_inches='tight')
                        st.download_button(
                            label="📥 Download Time Series PNG",
                            data=buf.getvalue(),
                            file_name=f"time_series_{time_component}_{timestamp}.png",
                            mime="image/png",
                            key=f"download_time_png_{timestamp}"
                        )
                    with col_dl2:
                        buf = BytesIO()
                        fig_time_series.savefig(buf, format="pdf", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download Time Series PDF",
                            data=buf.getvalue(),
                            file_name=f"time_series_{time_component}_{timestamp}.pdf",
                            mime="application/pdf",
                            key=f"download_time_pdf_{timestamp}"
                        )
                    with col_dl3:
                        buf = BytesIO()
                        fig_time_series.savefig(buf, format="svg", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download Time Series SVG",
                            data=buf.getvalue(),
                            file_name=f"time_series_{time_component}_{timestamp}.svg",
                            mime="image/svg+xml",
                            key=f"download_time_svg_{timestamp}"
                        )
                  
                    plt.close(fig_time_series)
              
                # Display stress evolution metrics
                st.subheader("📊 Stress Evolution Metrics")
              
                fig_metrics = st.session_state.time_frame_manager.create_stress_evolution_metrics(time_frames)
              
                if fig_metrics:
                    st.pyplot(fig_metrics)
                  
                    # Download metrics plot
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        buf = BytesIO()
                        fig_metrics.savefig(buf, format="png", dpi=figure_dpi, bbox_inches='tight')
                        st.download_button(
                            label="📥 Download Metrics PNG",
                            data=buf.getvalue(),
                            file_name=f"stress_metrics_{timestamp}.png",
                            mime="image/png",
                            key=f"download_metrics_png_{timestamp}"
                        )
                    with col_dl2:
                        buf = BytesIO()
                        fig_metrics.savefig(buf, format="pdf", bbox_inches='tight')
                        st.download_button(
                            label="📥 Download Metrics PDF",
                            data=buf.getvalue(),
                            file_name=f"stress_metrics_{timestamp}.pdf",
                            mime="application/pdf",
                            key=f"download_metrics_pdf_{timestamp}"
                        )
                  
                    plt.close(fig_metrics)
              
                # Individual frame download
                st.subheader("📥 Download Individual Time Frames")
              
                if time_component in time_frames:
                    frames = time_frames[time_component]
                    frame_numbers = time_frames.get('frame_numbers', [])
                  
                    # Create grid layout for frame selection
                    cols = st.columns(4)
                    for idx, (frame_num, frame_data) in enumerate(zip(frame_numbers, frames)):
                        col_idx = idx % 4
                        with cols[col_idx]:
                            # Create mini preview
                            fig_mini, ax_mini = plt.subplots(figsize=(3, 3), dpi=80)
                            im = ax_mini.imshow(frame_data, cmap='viridis', aspect='auto')
                            ax_mini.axis('off')
                            ax_mini.set_title(f"Frame {frame_num}", fontsize=8)
                            plt.colorbar(im, ax=ax_mini, fraction=0.046, pad=0.04)
                            plt.tight_layout()
                          
                            st.pyplot(fig_mini)
                            plt.close(fig_mini)
                          
                            # Download button for individual frame
                            buf = BytesIO()
                            np.save(buf, frame_data)
                            st.download_button(
                                label=f"📥 Frame {frame_num} (NPY)",
                                data=buf.getvalue(),
                                file_name=f"frame_{frame_num}_{time_component}_{timestamp}.npy",
                                mime="application/octet-stream",
                                key=f"download_frame_{frame_num}_{timestamp}"
                            )
  
    # Tab 6: Save & Export Results
    with tab6:
        st.subheader("💾 Export Prediction Results")
      
        # Check if we have predictions to save
        has_prediction = 'prediction_results' in st.session_state
      
        if not has_prediction:
            st.warning("⚠️ No prediction results available to save. Please run predictions first.")
        else:
            st.success("✅ Prediction results available for export!")
          
            # Display what's available
            st.info(f"**Single Target Prediction:** Available")
            single_params = st.session_state.prediction_results.get('target_params', {})
            st.write(f"- Target: {single_params.get('defect_type', 'Unknown')}, "
                    f"ε*={single_params.get('eps0', 0):.3f}, "
                    f"κ={single_params.get('kappa', 0):.3f}")
          
            st.divider()
          
            # Save options
            st.subheader("📁 Export Options")
          
            save_col1, save_col2, save_col3 = st.columns(3)
          
            with save_col1:
                save_mode = "Current Single Prediction"
                st.write("Exporting single prediction")
          
            with save_col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = st.text_input(
                    "Base filename",
                    value=f"prediction_{timestamp}",
                    help="Files will be saved with this base name plus appropriate extensions"
                )
          
            with save_col3:
                include_source_info = st.checkbox("Include source simulations info", value=True)
                include_visualizations = st.checkbox("Include visualization data", value=True)
          
            st.divider()
          
            # Save/Download buttons
            st.subheader("⬇️ Download Options")
          
            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
          
            # PKL download
            with dl_col1:
                st.markdown("**PKL Format**")
                prepare_pkl = st.button("💾 Prepare PKL", type="secondary", use_container_width=True, key="prepare_pkl")
              
                if prepare_pkl:
                    with st.spinner("Preparing PKL file..."):
                        try:
                            save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                st.session_state.prediction_results,
                                st.session_state.source_simulations
                            )
                          
                            # Add metadata
                            save_data['save_info'] = {
                                'format': 'pkl',
                                'timestamp': timestamp,
                                'mode': 'single'
                            }
                          
                            # Create download link
                            pkl_buffer = BytesIO()
                            pickle.dump(save_data, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                            pkl_buffer.seek(0)
                            st.session_state.download_pkl_data = pkl_buffer.getvalue()
                            st.success("✅ PKL file prepared!")
                          
                        except Exception as e:
                            st.error(f"❌ Error preparing PKL: {str(e)}")
              
                # Always show download button if data exists
                if st.session_state.download_pkl_data:
                    st.download_button(
                        label="📥 Download PKL",
                        data=st.session_state.download_pkl_data,
                        file_name=f"{base_filename}.pkl",
                        mime="application/octet-stream",
                        key="download_pkl_final",
                        use_container_width=True
                    )
                else:
                    st.caption("Click 'Prepare PKL' first")
          
            # PT download
            with dl_col2:
                st.markdown("**PT Format**")
                prepare_pt = st.button("💾 Prepare PT", type="secondary", use_container_width=True, key="prepare_pt")
              
                if prepare_pt:
                    with st.spinner("Preparing PT file..."):
                        try:
                            save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                st.session_state.prediction_results,
                                st.session_state.source_simulations
                            )
                          
                            # Add metadata
                            save_data['save_info'] = {
                                'format': 'pt',
                                'timestamp': timestamp,
                                'mode': 'single'
                            }
                          
                            # Create download link
                            pt_buffer = BytesIO()
                            torch.save(save_data, pt_buffer)
                            pt_buffer.seek(0)
                            st.session_state.download_pt_data = pt_buffer.getvalue()
                            st.success("✅ PT file prepared!")
                          
                        except Exception as e:
                            st.error(f"❌ Error preparing PT: {str(e)}")
              
                # Always show download button if data exists
                if st.session_state.download_pt_data:
                    st.download_button(
                        label="📥 Download PT",
                        data=st.session_state.download_pt_data,
                        file_name=f"{base_filename}.pt",
                        mime="application/octet-stream",
                        key="download_pt_final",
                        use_container_width=True
                    )
                else:
                    st.caption("Click 'Prepare PT' first")
          
            # ZIP download
            with dl_col3:
                st.markdown("**ZIP Archive**")
                prepare_zip = st.button("📦 Prepare ZIP", type="primary", use_container_width=True, key="prepare_zip")
              
                if prepare_zip:
                    with st.spinner("Creating comprehensive archive..."):
                        try:
                            zip_buffer = st.session_state.prediction_results_manager.create_single_prediction_archive(
                                st.session_state.prediction_results,
                                st.session_state.source_simulations
                            )
                            st.session_state.download_zip_data = zip_buffer.getvalue()
                            st.session_state.download_zip_filename = f"{base_filename}_complete.zip"
                            st.success("✅ ZIP archive prepared!")
                          
                        except Exception as e:
                            st.error(f"❌ Error creating archive: {str(e)}")
              
                # Always show download button if data exists
                if st.session_state.download_zip_data and st.session_state.download_zip_filename:
                    st.download_button(
                        label="📥 Download ZIP",
                        data=st.session_state.download_zip_data,
                        file_name=st.session_state.download_zip_filename,
                        mime="application/zip",
                        key="download_zip_final",
                        use_container_width=True
                    )
                else:
                    st.caption("Click 'Prepare ZIP' first")
          
            # Clear all button
            with dl_col4:
                st.markdown("**Clear All**")
                if st.button("🗑️ Clear All Prepared", type="secondary", use_container_width=True):
                    st.session_state.download_pkl_data = None
                    st.session_state.download_pt_data = None
                    st.session_state.download_zip_data = None
                    st.session_state.download_zip_filename = None
                    st.success("✅ All prepared files cleared!")
                    st.rerun()
                st.caption("Clears prepared download files from memory")
          
            st.divider()
          
            # Advanced options
            with st.expander("⚙️ Advanced Export Options", expanded=False):
                col_adv1, col_adv2 = st.columns(2)
              
                with col_adv1:
                    # Export stress fields as separate files
                    st.markdown("**Separate Stress Fields**")
                    stress_fields = st.session_state.prediction_results.get('stress_fields', {})
                  
                    for field_name, field_data in stress_fields.items():
                        if isinstance(field_data, np.ndarray):
                            npz_buffer = BytesIO()
                            np.savez_compressed(npz_buffer, data=field_data)
                            npz_buffer.seek(0)
                          
                            st.download_button(
                                label=f"📥 {field_name}.npz",
                                data=npz_buffer.getvalue(),
                                file_name=f"{base_filename}_{field_name}.npz",
                                mime="application/octet-stream",
                                key=f"download_npz_{field_name}_{timestamp}"
                            )
              
                with col_adv2:
                    # Export to other formats
                    st.markdown("**Export Formats**")
                  
                    # JSON export
                    target_params = st.session_state.prediction_results.get('target_params', {})
                    if target_params:
                        # Helper function to convert numpy types for JSON
                        def convert_for_json(obj):
                            if isinstance(obj, (np.float32, np.float64, np.float16)):
                                return float(obj)
                            elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                                return int(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, np.generic):
                                return obj.item()
                            else:
                                return obj
                      
                        json_str = json.dumps(target_params, indent=2, default=convert_for_json)
                      
                        st.download_button(
                            label="📥 Parameters JSON",
                            data=json_str,
                            file_name=f"{base_filename}_params.json",
                            mime="application/json",
                            key=f"download_json_{timestamp}"
                        )
                  
                    # CSV export
                    if 'stress_fields' in st.session_state.prediction_results:
                        stress_fields = st.session_state.prediction_results['stress_fields']
                        stats_rows = []
                        for field_name, field_data in stress_fields.items():
                            if isinstance(field_data, np.ndarray):
                                stats_rows.append({
                                    'field': field_name,
                                    'max': float(np.max(field_data)),
                                    'min': float(np.min(field_data)),
                                    'mean': float(np.mean(field_data)),
                                    'std': float(np.std(field_data)),
                                    '95th_percentile': float(np.percentile(field_data, 95))
                                })
                      
                        if stats_rows:
                            stats_df = pd.DataFrame(stats_rows)
                            csv_data = stats_df.to_csv(index=False)
                          
                            st.download_button(
                                label="📥 Statistics CSV",
                                data=csv_data,
                                file_name=f"{base_filename}_stats.csv",
                                mime="text/csv",
                                key=f"download_csv_{timestamp}"
                            )
          
            # EXPANDED SECTION: Download saved files from directory
            st.divider()
            st.subheader("📥 Download Files from numerical_solutions Directory")
            all_files_info = st.session_state.solutions_manager.get_all_files()
            all_files_info.sort(key=lambda x: datetime.fromisoformat(x['modified']), reverse=True) # Newest first
          
            if not all_files_info:
                st.info("No files found in the numerical_solutions directory.")
            else:
                with st.expander("📂 All Files in Directory (Newest First, Up to 20)", expanded=True):
                    for file_info in all_files_info[:20]: # Limit to 20 to avoid UI clutter
                        file_path = file_info['path']
                        file_name = file_info['filename']
                        file_size_kb = file_info['size'] // 1024
                        modified_time = file_info['modified']
                      
                        col_file1, col_file2 = st.columns([3, 1])
                        with col_file1:
                            st.write(f"**{file_name}** ({file_size_kb}KB, modified {modified_time})")
                        with col_file2:
                            try:
                                with open(file_path, 'rb') as f:
                                    file_data = f.read()
                                st.download_button(
                                    label="📥 Download",
                                    data=file_data,
                                    file_name=file_name,
                                    mime="application/octet-stream",
                                    key=f"download_dir_{file_name}_{hash(file_path)}" # Unique key
                                )
                            except Exception as e:
                                st.error(f"Error reading {file_name}: {str(e)}")
if __name__ == "__main__":
    create_attention_interface()
st.caption(f"🔬 Attention Interpolation • PKL/PT/ZIP Support • Matplotlib & Plotly Visualizations • {datetime.now().year}")
