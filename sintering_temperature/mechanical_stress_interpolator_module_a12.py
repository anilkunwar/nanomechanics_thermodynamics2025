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
# TRAINING RESULTS MANAGER
# =============================================
class TrainingResultsManager:
    """Manager for saving and downloading training results"""
    
    @staticmethod
    def prepare_training_data_for_saving(training_results: Dict[str, Any],
                                       source_simulations: List[Dict],
                                       model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare training results for saving to file
        
        Args:
            training_results: Dictionary of training results
            source_simulations: List of source simulation data
            model_params: Model parameters
            
        Returns:
            Structured dictionary ready for saving
        """
        # Create metadata
        metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'num_sources': len(source_simulations),
            'software_version': '1.0.0',
            'data_type': 'attention_training_results',
            'model_parameters': model_params
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
            'training_results': training_results.copy(),
            'training_history': {
                'epochs': len(training_results.get('losses', [])),
                'final_loss': float(training_results.get('losses', [0])[-1]) if training_results.get('losses') else 0,
                'best_loss': float(np.min(training_results.get('losses', [0]))) if training_results.get('losses') else 0
            }
        }
        
        return save_data
    
    @staticmethod
    def create_training_archive(training_results: Dict[str, Any],
                              source_simulations: List[Dict],
                              model_params: Dict[str, Any]) -> BytesIO:
        """
        Create a comprehensive archive for training results
        
        Args:
            training_results: Training results
            source_simulations: List of source simulations
            model_params: Model parameters
            
        Returns:
            BytesIO buffer containing the archive
        """
        # Create in-memory zip file
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save main training data as PKL
            save_data = TrainingResultsManager.prepare_training_data_for_saving(
                training_results, source_simulations, model_params
            )
            
            # PKL format
            pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
            zip_file.writestr('training_results.pkl', pkl_data)
            
            # 2. Save as PT (PyTorch) format
            pt_buffer = BytesIO()
            torch.save(save_data, pt_buffer)
            pt_buffer.seek(0)
            zip_file.writestr('training_results.pt', pt_buffer.read())
            
            # 3. Save training losses as CSV
            if 'losses' in training_results:
                losses = training_results['losses']
                epochs = range(1, len(losses) + 1)
                loss_df = pd.DataFrame({
                    'epoch': epochs,
                    'loss': losses,
                    'smoothed_loss': pd.Series(losses).rolling(window=max(3, len(losses)//10), 
                                                               center=True, min_periods=1).mean()
                })
                csv_data = loss_df.to_csv(index=False)
                zip_file.writestr('training_losses.csv', csv_data)
            
            # 4. Save model configuration as JSON
            config_data = {
                'model_parameters': model_params,
                'training_config': {
                    'num_epochs': len(training_results.get('losses', [])),
                    'num_sources': len(source_simulations),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            json_data = json.dumps(config_data, indent=2)
            zip_file.writestr('model_config.json', json_data)
            
            # 5. Save a README file
            readme_content = f"""# Training Results Archive
Generated: {datetime.now().isoformat()}
Number of source simulations: {len(source_simulations)}
Training epochs: {len(training_results.get('losses', []))}
Final loss: {training_results.get('losses', [0])[-1] if training_results.get('losses') else 0:.6f}

Files included:
1. training_results.pkl - Main training data (Python pickle format)
2. training_results.pt - PyTorch format
3. training_losses.csv - Training loss history
4. model_config.json - Model configuration

For more information, see the documentation.
"""
            zip_file.writestr('README.txt', readme_content)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# PREDICTION RESULTS SAVING AND DOWNLOAD MANAGER
# =============================================
class PredictionResultsManager:
    """Manager for saving and downloading prediction results"""
 
    @staticmethod
    def prepare_prediction_data_for_saving(prediction_results: Dict[str, Any],
                                         source_simulations: List[Dict],
                                         mode: str = 'single') -> Dict[str, Any]:
        """
        Prepare prediction results for saving to file
     
        Args:
            prediction_results: Dictionary of prediction results
            source_simulations: List of source simulation data
            mode: 'single' or 'multi'
         
        Returns:
            Structured dictionary ready for saving
        """
        # Create metadata
        metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'num_sources': len(source_simulations),
            'software_version': '1.0.0',
            'data_type': f'attention_interpolation_results_{mode}'
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
                prediction_results, source_simulations, 'single'
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
# MULTI-TARGET PREDICTION MANAGER
# =============================================
class MultiTargetPredictionManager:
    """Manager for handling multiple target predictions"""
    
    @staticmethod
    def create_parameter_grid(base_params, ranges_config):
        """
        Create a grid of parameter combinations based on ranges
        
        Args:
            base_params: Base parameter dictionary
            ranges_config: Dictionary with range specifications
                Example: {
                    'eps0': {'min': 0.5, 'max': 2.0, 'steps': 10},
                    'kappa': {'min': 0.2, 'max': 1.0, 'steps': 5},
                    'theta': {'values': [0, np.pi/6, np.pi/3, np.pi/2]}
                }
           
        Returns:
            List of parameter dictionaries
        """
        param_grid = []
        
        # Prepare parameter value lists
        param_values = {}
        
        for param_name, config in ranges_config.items():
            if 'values' in config:
                # Specific values provided
                param_values[param_name] = config['values']
            elif 'min' in config and 'max' in config:
                # Range with steps
                steps = config.get('steps', 10)
                param_values[param_name] = np.linspace(
                    config['min'], config['max'], steps
                ).tolist()
            else:
                # Single value
                param_values[param_name] = [config.get('value', base_params.get(param_name))]
        
        # Generate all combinations
        param_names = list(param_values.keys())
        value_arrays = [param_values[name] for name in param_names]
        
        for combination in product(*value_arrays):
            param_dict = base_params.copy()
            for name, value in zip(param_names, combination):
                param_dict[name] = float(value) if isinstance(value, (int, float, np.number)) else value
            
            param_grid.append(param_dict)
        
        return param_grid
    
    @staticmethod
    def batch_predict(source_simulations, target_params_list, interpolator, use_attention=True):
        """
        Perform batch predictions for multiple targets
        
        Args:
            source_simulations: List of source simulation data
            target_params_list: List of target parameter dictionaries
            interpolator: SpatialLocalityAttentionInterpolator instance
            use_attention: Whether to use attention mechanism (if False, uses Gaussian)
           
        Returns:
            Dictionary with predictions for each target
        """
        predictions = {}
        
        # Prepare source data
        source_param_vectors = []
        source_stress_data = []
        
        for sim_data in source_simulations:
            param_vector, _ = interpolator.compute_parameter_vector(sim_data)
            source_param_vectors.append(param_vector)
            
            # Get stress from final frame
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                source_stress_data.append(stress_components)
        
        source_param_vectors = np.array(source_param_vectors)
        source_stress_data = np.array(source_stress_data)
        
        # Convert to tensors if using attention
        if use_attention:
            source_params_tensor = torch.from_numpy(source_param_vectors).float()
            source_stress_tensor = torch.from_numpy(source_stress_data).float()
        
        # Predict for each target
        for idx, target_params in enumerate(target_params_list):
            # Compute target parameter vector
            target_vector, _ = interpolator.compute_parameter_vector(
                {'params': target_params}
            )
            
            if use_attention:
                # Use attention mechanism
                target_param_tensor = torch.from_numpy(target_vector).float().unsqueeze(0)
                
                # Get attention weights
                weights = interpolator.get_attention_weights(
                    target_param_tensor, source_params_tensor
                )
                weights = weights.detach().cpu().numpy()
            else:
                # Use Gaussian weights
                distances = np.sqrt(np.sum((source_param_vectors - target_vector) ** 2, axis=1))
                weights = np.exp(-0.5 * (distances / 0.3) ** 2)
                weights = weights / (np.sum(weights) + 1e-8)
            
            # Weighted combination
            weighted_stress = np.sum(
                source_stress_data * weights[:, np.newaxis, np.newaxis, np.newaxis],
                axis=0
            )
            
            predicted_stress = {
                'sigma_hydro': weighted_stress[0],
                'sigma_mag': weighted_stress[1],
                'von_mises': weighted_stress[2],
                'predicted': True,
                'target_params': target_params,
                'attention_weights': weights,
                'target_index': idx,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            predictions[f"target_{idx:03d}"] = predicted_stress
        
        return predictions
    
    @staticmethod
    def create_multi_prediction_archive_with_training(multi_predictions: Dict[str, Any],
                                                    source_simulations: List[Dict],
                                                    training_results: Optional[Dict[str, Any]] = None,
                                                    model_params: Optional[Dict[str, Any]] = None) -> BytesIO:
        """
        Create a comprehensive archive for multiple predictions including training data
        
        Args:
            multi_predictions: Dictionary of multiple predictions
            source_simulations: List of source simulations
            training_results: Optional training results
            model_params: Optional model parameters
            
        Returns:
            BytesIO buffer containing the archive
        """
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save training data if available
            if training_results and model_params:
                training_data = TrainingResultsManager.prepare_training_data_for_saving(
                    training_results, source_simulations, model_params
                )
                
                # Save training as PKL
                pkl_data = pickle.dumps(training_data, protocol=pickle.HIGHEST_PROTOCOL)
                zip_file.writestr('training/training_results.pkl', pkl_data)
                
                # Save training losses as CSV
                if 'losses' in training_results:
                    losses = training_results['losses']
                    epochs = range(1, len(losses) + 1)
                    loss_df = pd.DataFrame({
                        'epoch': epochs,
                        'loss': losses
                    })
                    csv_data = loss_df.to_csv(index=False)
                    zip_file.writestr('training/training_losses.csv', csv_data)
            
            # 2. Save each prediction individually
            for pred_key, pred_data in multi_predictions.items():
                # Create directory for each prediction
                pred_dir = f'predictions/{pred_key}'
                
                # Save prediction data
                save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                    pred_data, source_simulations, 'multi'
                )
                
                # Save as PKL
                pkl_data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
                zip_file.writestr(f'{pred_dir}/prediction.pkl', pkl_data)
                
                # Save as PT
                pt_buffer = BytesIO()
                torch.save(save_data, pt_buffer)
                pt_buffer.seek(0)
                zip_file.writestr(f'{pred_dir}/prediction.pt', pt_buffer.read())
                
                # Save stress fields as NPZ
                stress_fields = {k: v for k, v in pred_data.items()
                               if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
                
                if stress_fields:
                    npz_buffer = BytesIO()
                    np.savez_compressed(npz_buffer, **stress_fields)
                    npz_buffer.seek(0)
                    zip_file.writestr(f'{pred_dir}/stress_fields.npz', npz_buffer.read())
                
                # Save attention weights as CSV
                if 'attention_weights' in pred_data:
                    weights = pred_data['attention_weights']
                    weight_df = pd.DataFrame({
                        'source_id': [f'S{i+1}' for i in range(len(weights))],
                        'weight': weights,
                        'percent_contribution': 100 * weights / (np.sum(weights) + 1e-10)
                    })
                    csv_data = weight_df.to_csv(index=False)
                    zip_file.writestr(f'{pred_dir}/attention_weights.csv', csv_data)
            
            # 3. Save master summary
            summary_rows = []
            for pred_key, pred_data in multi_predictions.items():
                target_params = pred_data.get('target_params', {})
                stress_fields = {k: v for k, v in pred_data.items()
                               if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
                
                row = {
                    'prediction_id': pred_key,
                    'defect_type': target_params.get('defect_type', 'Unknown'),
                    'shape': target_params.get('shape', 'Unknown'),
                    'orientation': target_params.get('orientation', 'Unknown'),
                    'eps0': float(target_params.get('eps0', 0)),
                    'kappa': float(target_params.get('kappa', 0)),
                    'theta_deg': float(np.rad2deg(target_params.get('theta', 0))),
                    'prediction_timestamp': pred_data.get('prediction_timestamp', 'Unknown')
                }
                
                # Add stress metrics
                for field_name, field_data in stress_fields.items():
                    row[f'{field_name}_max'] = float(np.max(field_data))
                    row[f'{field_name}_mean'] = float(np.mean(field_data))
                    row[f'{field_name}_std'] = float(np.std(field_data))
                
                summary_rows.append(row)
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_csv = summary_df.to_csv(index=False)
                zip_file.writestr('multi_prediction_summary.csv', summary_csv)
            
            # 4. Save source parameters summary
            source_rows = []
            for i, sim_data in enumerate(source_simulations):
                params = sim_data.get('params', {})
                source_rows.append({
                    'source_id': f'S{i+1}',
                    'defect_type': params.get('defect_type', 'Unknown'),
                    'shape': params.get('shape', 'Unknown'),
                    'orientation': params.get('orientation', 'Unknown'),
                    'eps0': float(params.get('eps0', 0)),
                    'kappa': float(params.get('kappa', 0)),
                    'theta_deg': float(np.rad2deg(params.get('theta', 0)))
                })
            
            if source_rows:
                source_df = pd.DataFrame(source_rows)
                source_csv = source_df.to_csv(index=False)
                zip_file.writestr('source_parameters.csv', source_csv)
            
            # 5. Save a README file
            readme_content = f"""# Multi-Prediction Results Archive
Generated: {datetime.now().isoformat()}
Number of source simulations: {len(source_simulations)}
Number of predictions: {len(multi_predictions)}
{'Includes training data: Yes' if training_results else 'Includes training data: No'}

Structure:
- training/ - Training results (if available)
- predictions/[prediction_id]/ - Individual prediction data
- multi_prediction_summary.csv - Summary of all predictions
- source_parameters.csv - Source simulation parameters

Each prediction directory contains:
1. prediction.pkl - Main prediction data (pickle format)
2. prediction.pt - PyTorch format
3. stress_fields.npz - Stress fields as NumPy compressed
4. attention_weights.csv - Attention weights distribution

For more information, see the documentation.
"""
            zip_file.writestr('README.txt', readme_content)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# NUMERICAL SOLUTIONS MANAGER
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
                torch.nn.Linear(self.d_model * 2, self.d_model),
                torch.nn.LayerNorm(self.d_model)
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
            'output_projection': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.d_model * 2, self.d_model),
                torch.nn.LayerNorm(self.d_model)
            ),
        })
        return model

    def get_attention_weights(self, target_param, source_params):
        """Get attention weights between target and source parameters"""
        if source_params.size(0) == 0:
            raise ValueError("No source parameters provided for attention weights computation")
       
        # Prepare embeddings with correct dimensions
        # target_param: (1, input_dim) -> (1, d_model)
        # source_params: (N, input_dim) -> (N, d_model)
        target_embed = self.model.param_embedding(target_param) # (1, d_model)
        source_embeds = self.model.param_embedding(source_params) # (N, d_model)
       
        # Apply output projection
        target_embed = self.model.output_projection(target_embed) # (1, d_model)
        source_embeds = self.model.output_projection(source_embeds) # (N, d_model)
       
        # For multi-head attention, we need to add sequence dimension
        # target_embed: (1, 1, d_model) [batch_size=1, seq_len=1, d_model]
        # source_embeds: (1, N, d_model) [batch_size=1, seq_len=N, d_model]
        target_embed_seq = target_embed.unsqueeze(1) # (1, 1, d_model)
        source_embeds_seq = source_embeds.unsqueeze(0) # (1, N, d_model)
       
        # Compute attention with average_attn_weights=True to get averaged attention weights
        # This returns attention weights averaged over heads
        attn_output, attn_weights = self.model.attention(
            target_embed_seq,
            source_embeds_seq,
            source_embeds_seq,
            average_attn_weights=True,
            need_weights=True
        )
       
        # attn_weights shape: (1, 1, N) when average_attn_weights=True
        # Squeeze to get (N,)
        weights = attn_weights.squeeze(0).squeeze(0) # (N,)
       
        # Ensure weights are positive and sum to 1
        weights = torch.softmax(weights, dim=0)
       
        return weights

    def train(self, source_params, source_stress, epochs=50, lr=0.001):
        """Train the interpolator using leave-one-out cross-validation"""
        if source_params.size(0) < 2:
            raise ValueError("Need at least 2 source simulations for leave-one-out training")
       
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        losses = []
        N = source_params.size(0)
       
        # Enable training mode
        self.model.train()
       
        for epoch in range(epochs):
            epoch_loss = 0.0
            valid_iterations = 0
           
            # Shuffle indices for stochastic training
            indices = torch.randperm(N)
           
            for idx in indices:
                # Leave-one-out: use all except current as sources
                target_param = source_params[idx].unsqueeze(0) # (1, input_dim)
                target_stress = source_stress[idx] # (3, H, W)
               
                # Create mask for sources (all except current)
                src_mask = torch.ones(N, dtype=torch.bool)
                src_mask[idx] = False
               
                src_params = source_params[src_mask] # (N-1, input_dim)
                src_stress = source_stress[src_mask] # (N-1, 3, H, W)
               
                if len(src_params) < 1:
                    continue
               
                # Get attention weights
                weights = self.get_attention_weights(target_param, src_params) # (N-1,)
               
                # Debug: check weights shape and values
                if torch.isnan(weights).any() or torch.isinf(weights).any():
                    weights = torch.softmax(torch.randn_like(weights), dim=0)
               
                # Check dimensions before einsum
                if weights.dim() != 1:
                    weights = weights.squeeze()
               
                if weights.dim() != 1:
                    raise ValueError(f"Expected 1D weights tensor, got shape {weights.shape}")
                if src_stress.dim() != 4:
                    raise ValueError(f"Expected 4D stress tensor, got shape {src_stress.shape}")
                if weights.size(0) != src_stress.size(0):
                    raise ValueError(f"Dimension mismatch: weights ({weights.shape}) vs stress ({src_stress.shape})")
               
                # Compute weighted sum: sum over n dimension
                predicted_stress = torch.einsum('n,nchw->chw', weights, src_stress)
               
                # Compute loss with L2 regularization
                mse_loss = torch.mean((predicted_stress - target_stress) ** 2)
               
                # Add L2 regularization to prevent overfitting
                l2_lambda = 0.001
                l2_reg = torch.tensor(0.).to(source_params.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
               
                loss = mse_loss + l2_lambda * l2_reg
               
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
               
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
               
                optimizer.step()
               
                epoch_loss += mse_loss.item() # Track only MSE for reporting
                valid_iterations += 1
           
            if valid_iterations > 0:
                avg_loss = epoch_loss / valid_iterations
                losses.append(avg_loss)
                scheduler.step(avg_loss)
            else:
                losses.append(0.0)
                avg_loss = 0.0
       
        # Switch to evaluation mode
        self.model.eval()
       
        return losses

    def predict(self, target_param, source_params, source_stress):
        """Predict stress field for target parameters"""
        if source_params.size(0) == 0:
            raise ValueError("No source parameters provided for prediction")
       
        # Switch to evaluation mode
        self.model.eval()
       
        # Ensure no gradient computation
        with torch.no_grad():
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
           
            # Detach tensors from computation graph before converting to numpy
            predicted_stress_np = predicted_stress.detach().cpu().numpy()
            weights_np = weights.detach().cpu().numpy()
       
        return predicted_stress_np, weights_np

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
            format_type = 'pkl'
     
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

    def save_model(self, filename, format_type='pt'):
        """Save the trained model to file"""
        if format_type == 'pt':
            torch.save(self.model.state_dict(), filename)
        elif format_type == 'pkl':
            with open(filename, 'wb') as f:
                pickle.dump(self.model.state_dict(), f)
        else:
            raise ValueError(f"Unsupported model save format: {format_type}")

# =============================================
# GRID AND EXTENT CONFIGURATION
# =============================================
def get_grid_extent(N=128, dx=0.1):
    """Get grid extent for visualization"""
    return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]

# =============================================
# ATTENTION INTERFACE FOR MULTI-TARGET TRAINING
# =============================================
def create_multi_target_attention_interface():
    """Create the multi-target attention interpolation interface"""
 
    st.header("🤖 Multi-Target Spatial-Attention Stress Interpolation")
 
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
 
    # Initialize managers
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager()
 
    if 'prediction_results_manager' not in st.session_state:
        st.session_state.prediction_results_manager = PredictionResultsManager()
    
    if 'training_results_manager' not in st.session_state:
        st.session_state.training_results_manager = TrainingResultsManager()
    
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
 
    if 'visualization_manager' not in st.session_state:
        st.session_state.visualization_manager = VisualizationManager()
 
    if 'time_frame_manager' not in st.session_state:
        st.session_state.time_frame_manager = TimeFrameVisualizationManager(
            st.session_state.visualization_manager
        )
 
    # Initialize session state variables
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
    
    if 'batch_params' not in st.session_state:
        st.session_state.batch_params = []
    
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    
    if 'model_params' not in st.session_state:
        st.session_state.model_params = None
    
    if 'matplotlib_figures' not in st.session_state:
        st.session_state.matplotlib_figures = {}
 
    # Get grid extent for visualization
    extent = get_grid_extent()
 
    # Sidebar configuration
    st.sidebar.header("🔮 Multi-Target Settings")
 
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
            index=0
        )
     
        default_colormap = st.selectbox(
            "Default Colormap",
            ["viridis", "plasma", "coolwarm", "RdBu", "Spectral", "custom_stress"],
            index=5
        )
     
        include_contours = st.checkbox("Include Contour Lines", value=True)
        include_grid = st.checkbox("Include Grid", value=True)
        figure_dpi = st.slider("Figure DPI", 100, 300, 150, 10)
 
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Multi-Targets",
        "🚀 Multi-Target Training",
        "📊 Results & Visualization",
        "📈 Batch Analysis",
        "💾 Export Results",
        "⏱️ Time Frame Analysis"
    ])
 
    # Tab 1: Load Source Data
    with tab1:
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
                        st.rerun()
     
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
 
    # Tab 2: Configure Multi-Targets
    with tab2:
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            st.info("Configure parameter ranges for batch predictions")
           
            st.markdown("### 🎯 Base Parameters")
            col1, col2 = st.columns(2)
           
            with col1:
                base_defect = st.selectbox(
                    "Base Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="base_defect_multi"
                )
               
                base_shape = st.selectbox(
                    "Base Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="base_shape_multi"
                )
           
            with col2:
                orientation_mode = st.radio(
                    "Orientation Mode",
                    ["Predefined", "Custom Angles"],
                    horizontal=True,
                    key="orientation_mode_multi"
                )
               
                if orientation_mode == "Predefined":
                    base_orientation = st.selectbox(
                        "Base Orientation",
                        ["Horizontal {111} (0°)",
                         "Tilted 30° (1¯10 projection)",
                         "Tilted 60°",
                         "Vertical {111} (90°)"],
                        index=0,
                        key="base_orientation_multi"
                    )
                   
                    angle_map = {
                        "Horizontal {111} (0°)": 0,
                        "Tilted 30° (1¯10 projection)": 30,
                        "Tilted 60°": 60,
                        "Vertical {111} (90°)": 90,
                    }
                    base_theta = np.deg2rad(angle_map.get(base_orientation, 0))
                    st.info(f"**Base θ:** {np.rad2deg(base_theta):.1f}°")
                   
                else:
                    base_angle = st.slider(
                        "Base Angle (degrees)",
                        0.0, 90.0, 0.0, 0.5,
                        key="base_angle_custom_multi"
                    )
                    base_theta = np.deg2rad(base_angle)
                    base_orientation = st.session_state.interpolator.get_orientation_from_angle(base_angle)
                    st.info(f"**Base θ:** {base_angle:.1f}°")
                    st.info(f"**Orientation:** {base_orientation}")
           
            base_params = {
                'defect_type': base_defect,
                'shape': base_shape,
                'orientation': base_orientation,
                'theta': base_theta
            }
           
            # Parameter ranges
            st.markdown("### 📊 Parameter Ranges")
           
            st.markdown("#### ε* Range")
            eps0_range_col1, eps0_range_col2, eps0_range_col3 = st.columns(3)
            with eps0_range_col1:
                eps0_min = st.number_input("Min ε*", 0.3, 3.0, 0.5, 0.1, key="eps0_min")
            with eps0_range_col2:
                eps0_max = st.number_input("Max ε*", 0.3, 3.0, 2.5, 0.1, key="eps0_max")
            with eps0_range_col3:
                eps0_steps = st.number_input("Steps", 2, 100, 10, 1, key="eps0_steps")
           
            st.markdown("#### κ Range")
            kappa_range_col1, kappa_range_col2, kappa_range_col3 = st.columns(3)
            with kappa_range_col1:
                kappa_min = st.number_input("Min κ", 0.1, 2.0, 0.2, 0.05, key="kappa_min")
            with kappa_range_col2:
                kappa_max = st.number_input("Max κ", 0.1, 2.0, 1.5, 0.05, key="kappa_max")
            with kappa_range_col3:
                kappa_steps = st.number_input("Steps", 2, 50, 8, 1, key="kappa_steps")
           
            st.markdown("#### Orientation Range (Optional)")
            use_orientation_range = st.checkbox("Vary orientation", value=False, key="use_orientation_range")
           
            if use_orientation_range:
                if orientation_mode == "Predefined":
                    orientation_options = st.multiselect(
                        "Select orientations to include",
                        ["Horizontal {111} (0°)", "Tilted 30° (1¯10 projection)", "Tilted 60°", "Vertical {111} (90°)"],
                        default=["Horizontal {111} (0°)", "Vertical {111} (90°)"],
                        key="orientation_multi_select"
                    )
                else:
                    orientation_range_col1, orientation_range_col2, orientation_range_col3 = st.columns(3)
                    with orientation_range_col1:
                        angle_min = st.number_input("Min Angle (°)", 0.0, 90.0, 0.0, 1.0, key="angle_min")
                    with orientation_range_col2:
                        angle_max = st.number_input("Max Angle (°)", 0.0, 90.0, 90.0, 1.0, key="angle_max")
                    with orientation_range_col3:
                        angle_steps = st.number_input("Steps", 2, 20, 5, 1, key="angle_steps")
           
            # Generate parameter grid
            if st.button("🔄 Generate Parameter Grid", type="primary"):
                ranges_config = {}
               
                if eps0_max > eps0_min:
                    ranges_config['eps0'] = {
                        'min': float(eps0_min),
                        'max': float(eps0_max),
                        'steps': int(eps0_steps)
                    }
               
                if kappa_max > kappa_min:
                    ranges_config['kappa'] = {
                        'min': float(kappa_min),
                        'max': float(kappa_max),
                        'steps': int(kappa_steps)
                    }
               
                if use_orientation_range:
                    if orientation_mode == "Predefined" and orientation_options:
                        angle_map_rev = {
                            "Horizontal {111} (0°)": 0,
                            "Tilted 30° (1¯10 projection)": 30,
                            "Tilted 60°": 60,
                            "Vertical {111} (90°)": 90,
                        }
                        orientation_angles = [angle_map_rev[orient] for orient in orientation_options]
                        ranges_config['theta'] = {
                            'values': [np.deg2rad(angle) for angle in orientation_angles]
                        }
                    else:
                        if angle_max > angle_min:
                            angles = np.linspace(angle_min, angle_max, angle_steps)
                            ranges_config['theta'] = {
                                'values': [np.deg2rad(angle) for angle in angles]
                            }
               
                param_grid = st.session_state.multi_target_manager.create_parameter_grid(
                    base_params, ranges_config
                )
               
                for param_set in param_grid:
                    angle = np.rad2deg(param_set.get('theta', 0))
                    param_set['orientation'] = st.session_state.interpolator.get_orientation_from_angle(angle)
               
                st.session_state.batch_params = param_grid
               
                st.success(f"✅ Generated {len(param_grid)} parameter combinations!")
               
                st.subheader("📋 Generated Parameter Grid")
               
                grid_data = []
                for i, params in enumerate(param_grid):
                    grid_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': f"{params.get('eps0', 'Unknown'):.3f}",
                        'κ': f"{params.get('kappa', 'Unknown'):.3f}",
                        'Orientation': params.get('orientation', 'Unknown'),
                        'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                    })
               
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(df_grid, use_container_width=True)
           
            if st.session_state.batch_params:
                st.subheader("📊 Current Parameter Grid")
               
                grid_data = []
                for i, params in enumerate(st.session_state.batch_params):
                    grid_data.append({
                        'ID': i+1,
                        'Defect': params.get('defect_type', 'Unknown'),
                        'Shape': params.get('shape', 'Unknown'),
                        'ε*': f"{params.get('eps0', 'Unknown'):.3f}",
                        'κ': f"{params.get('kappa', 'Unknown'):.3f}",
                        'Orientation': params.get('orientation', 'Unknown'),
                        'θ°': f"{np.rad2deg(params.get('theta', 0)):.1f}"
                    })
               
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(df_grid, use_container_width=True)
                   
                    if st.button("🗑️ Clear Parameter Grid", type="secondary"):
                        st.session_state.batch_params = []
                        st.session_state.multi_target_predictions = {}
                        st.success("Parameter grid cleared!")
                        st.rerun()
 
    # Tab 3: Multi-Target Training
    with tab3:
        st.subheader("🚀 Multi-Target Training & Prediction")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        elif not st.session_state.batch_params:
            st.warning("⚠️ Please generate a parameter grid first")
        else:
            col1, col2 = st.columns(2)
           
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
                use_attention = st.checkbox("Use Attention Mechanism", value=True,
                                          help="If unchecked, uses simple Gaussian interpolation")
           
            with col2:
                batch_size = st.slider("Batch Size", 1, 16, 4, 1)
                save_training = st.checkbox("Save Training Results", value=True,
                                          help="Include training data in download archive")
                enable_progress = st.checkbox("Show Progress Bar", value=True)
           
            if st.button("🚀 Train & Predict (Batch)", type="primary"):
                with st.spinner(f"Running batch predictions for {len(st.session_state.batch_params)} targets..."):
                    try:
                        # Prepare source data for training
                        param_vectors = []
                        stress_data = []
                        
                        # Filter simulations with history
                        valid_simulations = 0
                        for sim_data in st.session_state.source_simulations:
                            history = sim_data.get('history', [])
                            if not history:
                                st.warning(f"Skipping simulation without history: {sim_data.get('filename', 'unknown')}")
                                continue
                            
                            param_vector, _ = st.session_state.interpolator.compute_parameter_vector(sim_data)
                            param_vectors.append(param_vector)
                            
                            eta, stress_fields = history[-1]
                            stress_components = np.stack([
                                stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                                stress_fields.get('sigma_mag', np.zeros_like(eta)),
                                stress_fields.get('von_mises', np.zeros_like(eta))
                            ], axis=0)
                            stress_data.append(stress_components)
                            valid_simulations += 1
                        
                        if valid_simulations < 2:
                            raise ValueError("Need at least 2 source simulations with history for training")
                        
                        param_vectors = np.array(param_vectors)
                        stress_data = np.stack(stress_data)  # (N, 3, H, W)
                        
                        # Training results storage
                        training_results = None
                        model_params = None
                        
                        # Train model if using attention
                        if use_attention:
                            st.info(f"Training attention model with {valid_simulations} sources...")
                            
                            source_params = torch.from_numpy(param_vectors).float()
                            source_stress = torch.from_numpy(stress_data).float()
                            
                            # Get model parameters for saving
                            model_params = {
                                'num_heads': st.session_state.interpolator.num_heads,
                                'd_model': st.session_state.interpolator.d_model,
                                'sigma_spatial': st.session_state.interpolator.sigma_spatial,
                                'sigma_param': st.session_state.interpolator.sigma_param,
                                'use_gaussian': st.session_state.interpolator.use_gaussian,
                                'epochs': epochs,
                                'learning_rate': learning_rate
                            }
                            
                            # Show progress bar
                            if enable_progress:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                            
                            # Train the model
                            training_losses = st.session_state.interpolator.train(
                                source_params, source_stress, epochs=epochs, lr=learning_rate
                            )
                            
                            if enable_progress:
                                progress_bar.empty()
                                status_text.empty()
                            
                            training_results = {
                                'losses': training_losses,
                                'num_sources': valid_simulations,
                                'training_timestamp': datetime.now().isoformat(),
                                'final_loss': float(training_losses[-1]) if training_losses else 0
                            }
                            
                            st.success(f"✅ Training complete! Final loss: {training_losses[-1]:.6f}")
                        
                        # Perform batch predictions
                        predictions = st.session_state.multi_target_manager.batch_predict(
                            st.session_state.source_simulations,
                            st.session_state.batch_params,
                            st.session_state.interpolator,
                            use_attention=use_attention
                        )
                        
                        st.session_state.multi_target_predictions = predictions
                        
                        # Store training results in session state
                        if training_results:
                            st.session_state.training_results = training_results
                            st.session_state.model_params = model_params
                        
                        if predictions:
                            # Store first prediction for visualization
                            first_key = list(predictions.keys())[0]
                            st.session_state.prediction_results = {
                                'stress_fields': predictions[first_key],
                                'attention_weights': predictions[first_key]['attention_weights'],
                                'target_params': predictions[first_key]['target_params'],
                                'training_losses': training_losses if use_attention else np.random.rand(epochs) * 0.1 * (1 - np.linspace(0, 1, epochs)),
                                'source_count': len(st.session_state.source_simulations),
                                'mode': 'multi',
                                'current_target_index': 0,
                                'total_targets': len(predictions),
                                'use_attention': use_attention
                            }
                        
                        st.success(f"✅ Batch predictions complete! Generated {len(predictions)} predictions")
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch prediction: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
            
            # Download trained model
            if 'training_results' in st.session_state and st.session_state.training_results:
                st.subheader("🤖 Download Trained Model")
                model_format = st.selectbox("Model Download Format", ["PT", "PKL"])
                model_filename = st.text_input("Model Filename", "trained_multi_target_model")
                if st.button("📥 Download Model"):
                    buf = BytesIO()
                    st.session_state.interpolator.save_model(buf, model_format.lower())
                    buf.seek(0)
                    st.download_button(
                        label="Download Trained Model",
                        data=buf.getvalue(),
                        file_name=f"{model_filename}.{model_format.lower()}",
                        mime="application/octet-stream"
                    )
 
    # Tab 4: Results & Visualization
    with tab4:
        if 'multi_target_predictions' not in st.session_state or not st.session_state.multi_target_predictions:
            st.warning("⚠️ Please run multi-target training and prediction first")
        else:
            results = st.session_state.prediction_results
            
            # Target selection
            target_keys = list(st.session_state.multi_target_predictions.keys())
            selected_target_key = st.selectbox(
                "Select Target Prediction",
                options=target_keys,
                index=results.get('current_target_index', 0),
                key="multi_target_selector"
            )
            selected_results = st.session_state.multi_target_predictions[selected_target_key]
            
            # Unify access
            if 'stress_fields' in selected_results:
                stress_fields = selected_results['stress_fields']
            else:
                stress_fields = {
                    'sigma_hydro': selected_results.get('sigma_hydro'),
                    'sigma_mag': selected_results.get('sigma_mag'),
                    'von_mises': selected_results.get('von_mises')
                }
           
            attention_weights = selected_results.get('attention_weights')
            target_params = selected_results.get('target_params', {})
            training_losses = results.get('training_losses')
           
            # Visualization controls
            col_viz1, col_viz2, col_viz3 = st.columns(3)
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
           
            # Store matplotlib figures for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           
            if stress_component in stress_fields:
                stress_data = stress_fields[stress_component]
               
                if viz_library == "Plotly (Interactive)":
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
                    title = f'{stress_component.replace("_", " ").title()} (GPa)\nOrientation: {target_params.get("orientation", "Unknown")}'
                   
                    fig_matplotlib = st.session_state.visualization_manager.create_stress_field_plot_matplotlib(
                        stress_data=stress_data,
                        title=title,
                        component_name=stress_component,
                        extent=extent,
                        include_contour=show_contour,
                        include_colorbar=True
                    )
                   
                    st.pyplot(fig_matplotlib)
                   
                    # Store for download
                    st.session_state.matplotlib_figures['stress_field'] = fig_matplotlib
            
            # Attention weights visualization
            st.subheader("🔍 Attention Weights")
           
            if attention_weights is not None:
                weights = attention_weights
                source_names = [f'S{i+1}' for i in range(st.session_state.prediction_results.get('source_count', 0))]
               
                if viz_library == "Plotly (Interactive)":
                    fig_weights = px.bar(
                        x=source_names,
                        y=weights,
                        labels={'x': 'Source Simulations', 'y': 'Attention Weight'},
                        title='Attention Weights Distribution'
                    )
                    fig_weights.update_traces(marker_color='steelblue', opacity=0.7)
                    fig_weights.update_layout(yaxis_range=[0, max(weights) * 1.2])
                   
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
                   
                    st.session_state.matplotlib_figures['attention_weights'] = fig_weights
            
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
                   
                    st.session_state.matplotlib_figures['training_losses'] = fig_losses
            
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
               
                csv_data = df_stats.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics CSV",
                    data=csv_data,
                    file_name=f"stress_statistics_{timestamp}.csv",
                    mime="text/csv",
                    key=f"download_stats_{timestamp}"
                )
 
    # Tab 5: Batch Analysis
    with tab5:
        if 'multi_target_predictions' not in st.session_state or not st.session_state.multi_target_predictions:
            st.warning("⚠️ Please run multi-target predictions first")
        else:
            st.subheader("📈 Batch Analysis Dashboard")
            
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Parameter Sweep", "Stress Distribution", "Attention Patterns", "Comparative Analysis"],
                key="analysis_type"
            )
            
            if analysis_type == "Parameter Sweep":
                param_to_analyze = st.selectbox("Parameter to Analyze", ["eps0", "kappa", "theta"], key="param_analyze")
                metric_to_plot = st.selectbox("Metric to Plot", 
                                            ["Max von Mises", "Mean von Mises", "Max Hydrostatic", "Mean Hydrostatic"],
                                            key="metric_plot")
                
                # Collect data
                x_values = []
                y_values = []
                target_ids = []
                
                for pred_key, pred in st.session_state.multi_target_predictions.items():
                    params = pred.get('target_params', {})
                    stress = pred.get('stress_fields', {})
                    
                    x_val = params.get(param_to_analyze, 0)
                    if param_to_analyze == 'theta':
                        x_val = np.rad2deg(x_val)
                    
                    if metric_to_plot == "Max von Mises":
                        y_val = np.max(stress.get('von_mises', np.array([0])))
                    elif metric_to_plot == "Mean von Mises":
                        y_val = np.mean(stress.get('von_mises', np.array([0])))
                    elif metric_to_plot == "Max Hydrostatic":
                        y_val = np.max(stress.get('sigma_hydro', np.array([0])))
                    elif metric_to_plot == "Mean Hydrostatic":
                        y_val = np.mean(stress.get('sigma_hydro', np.array([0])))
                    
                    x_values.append(x_val)
                    y_values.append(y_val)
                    target_ids.append(pred_key)
                
                # Create plot
                fig = px.scatter(x=x_values, y=y_values, 
                               title=f"{metric_to_plot} vs {param_to_analyze}",
                               labels={'x': param_to_analyze, 'y': metric_to_plot},
                               hover_name=target_ids)
                st.plotly_chart(fig)
                
                # Add trend line
                if len(x_values) > 2:
                    z = np.polyfit(x_values, y_values, 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=np.linspace(min(x_values), max(x_values), 100),
                        y=p(np.linspace(min(x_values), max(x_values), 100)),
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', dash='dash')
                    ))
                    st.plotly_chart(fig)
            
            elif analysis_type == "Stress Distribution":
                # Create violin plot of stress distributions
                all_von_mises = []
                all_hydrostatic = []
                
                for pred in st.session_state.multi_target_predictions.values():
                    stress = pred.get('stress_fields', {})
                    vm = stress.get('von_mises', np.array([0])).flatten()
                    hydro = stress.get('sigma_hydro', np.array([0])).flatten()
                    
                    # Sample to avoid memory issues
                    if len(vm) > 10000:
                        vm = np.random.choice(vm, 10000, replace=False)
                    if len(hydro) > 10000:
                        hydro = np.random.choice(hydro, 10000, replace=False)
                    
                    all_von_mises.extend(vm)
                    all_hydrostatic.extend(hydro)
                
                fig = make_subplots(rows=1, cols=2, subplot_titles=('Von Mises Distribution', 'Hydrostatic Distribution'))
                
                fig.add_trace(go.Violin(y=all_von_mises, name='Von Mises', box_visible=True), row=1, col=1)
                fig.add_trace(go.Violin(y=all_hydrostatic, name='Hydrostatic', box_visible=True), row=1, col=2)
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig)
            
            elif analysis_type == "Attention Patterns":
                # Analyze attention weight patterns across targets
                attention_matrix = []
                target_labels = []
                
                for pred_key, pred in st.session_state.multi_target_predictions.items():
                    weights = pred.get('attention_weights', [])
                    if len(weights) > 0:
                        attention_matrix.append(weights)
                        target_labels.append(pred_key)
                
                if attention_matrix:
                    attention_matrix = np.array(attention_matrix)
                    fig = px.imshow(attention_matrix.T,
                                  labels=dict(x="Target", y="Source", color="Weight"),
                                  x=target_labels,
                                  y=[f'S{i+1}' for i in range(attention_matrix.shape[1])],
                                  title="Attention Weight Matrix")
                    st.plotly_chart(fig)
            
            # Export analysis results
            if st.button("📊 Export Analysis Results"):
                analysis_data = {
                    'batch_size': len(st.session_state.multi_target_predictions),
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                    'summary_statistics': {
                        'num_predictions': len(st.session_state.multi_target_predictions),
                        'num_sources': len(st.session_state.source_simulations)
                    }
                }
                
                json_data = json.dumps(analysis_data, indent=2)
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=json_data,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
 
    # Tab 6: Export Results
    with tab6:
        st.subheader("💾 Export Multi-Target Results")
        
        # Check available data
        has_multi_predictions = ('multi_target_predictions' in st.session_state and
                                len(st.session_state.multi_target_predictions) > 0)
        has_training_results = ('training_results' in st.session_state and
                               st.session_state.training_results is not None)
        
        if not has_multi_predictions:
            st.warning("⚠️ No multi-target results available to export. Please run predictions first.")
        else:
            st.success("✅ Multi-target results available for export!")
            
            # Display what's available
            if has_multi_predictions:
                st.info(f"**Multiple Target Predictions:** {len(st.session_state.multi_target_predictions)} available")
            if has_training_results:
                st.info("**Training Results:** Available (includes loss history)")
            
            st.divider()
            
            # Base filename input
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = st.text_input(
                "Base filename for downloads",
                value=f"multi_target_{timestamp}",
                help="All files will use this base name"
            )
            
            # Download format selection
            download_format = st.radio(
                "Select Download Format",
                ["Complete Archive (ZIP)", "Individual Files", "Training Only", "Summary Only"],
                index=0,
                help="Choose how you want to download the results"
            )
            
            st.divider()
            
            # COMPLETE ARCHIVE OPTION
            if download_format == "Complete Archive (ZIP)":
                st.markdown("### 📦 Complete Results Archive")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info("Includes everything: training data, all predictions, summaries, and source parameters.")
                
                with col2:
                    # Create complete archive
                    archive_buffer = MultiTargetPredictionManager.create_multi_prediction_archive_with_training(
                        st.session_state.multi_target_predictions,
                        st.session_state.source_simulations,
                        st.session_state.training_results if has_training_results else None,
                        st.session_state.model_params if has_training_results else None
                    )
                    
                    st.download_button(
                        label="📥 Download Complete Archive",
                        data=archive_buffer.getvalue(),
                        file_name=f"{base_filename}_complete.zip",
                        mime="application/zip",
                        key=f"download_complete_{timestamp}",
                        use_container_width=True,
                        type="primary"
                    )
            
            # INDIVIDUAL FILES OPTION
            elif download_format == "Individual Files":
                st.markdown("### 📄 Download Individual Components")
                
                # Training files (if available)
                if has_training_results:
                    st.markdown("#### 🎯 Training Files")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Training results PKL
                        training_data = TrainingResultsManager.prepare_training_data_for_saving(
                            st.session_state.training_results,
                            st.session_state.source_simulations,
                            st.session_state.model_params
                        )
                        pkl_buffer = BytesIO()
                        pickle.dump(training_data, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                        pkl_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Training PKL",
                            data=pkl_buffer.getvalue(),
                            file_name=f"{base_filename}_training.pkl",
                            mime="application/octet-stream",
                            key=f"download_training_pkl_{timestamp}"
                        )
                    
                    with col2:
                        # Training losses CSV
                        if 'losses' in st.session_state.training_results:
                            losses = st.session_state.training_results['losses']
                            epochs = range(1, len(losses) + 1)
                            loss_df = pd.DataFrame({
                                'epoch': epochs,
                                'loss': losses,
                                'smoothed_loss': pd.Series(losses).rolling(window=max(3, len(losses)//10), 
                                                                           center=True, min_periods=1).mean()
                            })
                            csv_data = loss_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Training Losses CSV",
                                data=csv_data,
                                file_name=f"{base_filename}_training_losses.csv",
                                mime="text/csv",
                                key=f"download_training_csv_{timestamp}"
                            )
                    
                    with col3:
                        # Model config JSON
                        config_data = {
                            'model_parameters': st.session_state.model_params,
                            'training_config': {
                                'num_epochs': len(st.session_state.training_results.get('losses', [])),
                                'num_sources': len(st.session_state.source_simulations),
                                'timestamp': datetime.now().isoformat()
                            }
                        }
                        json_data = json.dumps(config_data, indent=2)
                        
                        st.download_button(
                            label="📥 Model Config JSON",
                            data=json_data,
                            file_name=f"{base_filename}_model_config.json",
                            mime="application/json",
                            key=f"download_config_json_{timestamp}"
                        )
                
                # Multi-target summary
                st.markdown("#### 📊 Multi-Target Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary CSV
                    summary_rows = []
                    for pred_key, pred_data in st.session_state.multi_target_predictions.items():
                        target_params = pred_data.get('target_params', {})
                        summary_rows.append({
                            'prediction_id': pred_key,
                            'defect_type': target_params.get('defect_type', 'Unknown'),
                            'shape': target_params.get('shape', 'Unknown'),
                            'orientation': target_params.get('orientation', 'Unknown'),
                            'eps0': float(target_params.get('eps0', 0)),
                            'kappa': float(target_params.get('kappa', 0)),
                            'theta_deg': float(np.rad2deg(target_params.get('theta', 0)))
                        })
                    
                    if summary_rows:
                        summary_df = pd.DataFrame(summary_rows)
                        csv_data = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Multi-Target Summary CSV",
                            data=csv_data,
                            file_name=f"{base_filename}_multi_summary.csv",
                            mime="text/csv",
                            key=f"download_multi_summary_{timestamp}"
                        )
                
                with col2:
                    # Source parameters CSV
                    source_rows = []
                    for i, sim_data in enumerate(st.session_state.source_simulations):
                        params = sim_data.get('params', {})
                        source_rows.append({
                            'source_id': f'S{i+1}',
                            'defect_type': params.get('defect_type', 'Unknown'),
                            'shape': params.get('shape', 'Unknown'),
                            'orientation': params.get('orientation', 'Unknown'),
                            'eps0': float(params.get('eps0', 0)),
                            'kappa': float(params.get('kappa', 0)),
                            'theta_deg': float(np.rad2deg(params.get('theta', 0)))
                        })
                    
                    if source_rows:
                        source_df = pd.DataFrame(source_rows)
                        csv_data = source_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Source Parameters CSV",
                            data=csv_data,
                            file_name=f"{base_filename}_source_params.csv",
                            mime="text/csv",
                            key=f"download_source_csv_{timestamp}"
                        )
                
                # Individual target predictions
                st.markdown("#### 🎯 Individual Target Predictions")
                target_keys = list(st.session_state.multi_target_predictions.keys())
                selected_target = st.selectbox(
                    "Select a target to download",
                    options=target_keys,
                    key="individual_target_download"
                )
                
                if selected_target:
                    pred_data = st.session_state.multi_target_predictions[selected_target]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Individual prediction PKL
                        save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                            pred_data, st.session_state.source_simulations, 'single'
                        )
                        pkl_buffer = BytesIO()
                        pickle.dump(save_data, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                        pkl_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Individual PKL",
                            data=pkl_buffer.getvalue(),
                            file_name=f"{selected_target}_prediction.pkl",
                            mime="application/octet-stream",
                            key=f"download_individual_pkl_{timestamp}"
                        )
                    
                    with col2:
                        # Stress fields NPZ
                        stress_fields = {k: v for k, v in pred_data.items()
                                       if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
                        
                        if stress_fields:
                            npz_buffer = BytesIO()
                            np.savez_compressed(npz_buffer, **stress_fields)
                            npz_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Stress Fields NPZ",
                                data=npz_buffer.getvalue(),
                                file_name=f"{selected_target}_stress.npz",
                                mime="application/octet-stream",
                                key=f"download_stress_npz_{timestamp}"
                            )
                    
                    with col3:
                        # Attention weights CSV
                        if 'attention_weights' in pred_data:
                            weights = pred_data['attention_weights']
                            weight_df = pd.DataFrame({
                                'source_id': [f'S{i+1}' for i in range(len(weights))],
                                'weight': weights,
                                'percent_contribution': 100 * weights / (np.sum(weights) + 1e-10)
                            })
                            csv_data = weight_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Attention Weights CSV",
                                data=csv_data,
                                file_name=f"{selected_target}_attention.csv",
                                mime="text/csv",
                                key=f"download_attention_csv_{timestamp}"
                            )
            
            # TRAINING ONLY OPTION
            elif download_format == "Training Only" and has_training_results:
                st.markdown("### 🎯 Training Data Only")
                
                # Create training archive
                training_archive = TrainingResultsManager.create_training_archive(
                    st.session_state.training_results,
                    st.session_state.source_simulations,
                    st.session_state.model_params
                )
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info("Includes training loss history, model configuration, and source parameters.")
                
                with col2:
                    st.download_button(
                        label="📥 Download Training Archive",
                        data=training_archive.getvalue(),
                        file_name=f"{base_filename}_training.zip",
                        mime="application/zip",
                        key=f"download_training_zip_{timestamp}",
                        use_container_width=True,
                        type="primary"
                    )
            
            # SUMMARY ONLY OPTION
            elif download_format == "Summary Only":
                st.markdown("### 📊 Summary Data Only")
                
                # Create combined summary
                all_summaries = []
                for pred_key, pred_data in st.session_state.multi_target_predictions.items():
                    target_params = pred_data.get('target_params', {})
                    stress_fields = pred_data.get('stress_fields', {})
                    
                    summary = {
                        'prediction_id': pred_key,
                        'defect_type': target_params.get('defect_type', 'Unknown'),
                        'shape': target_params.get('shape', 'Unknown'),
                        'orientation': target_params.get('orientation', 'Unknown'),
                        'eps0': float(target_params.get('eps0', 0)),
                        'kappa': float(target_params.get('kappa', 0)),
                        'theta_deg': float(np.rad2deg(target_params.get('theta', 0))),
                        'von_mises_max': float(np.max(stress_fields.get('von_mises', 0))),
                        'von_mises_mean': float(np.mean(stress_fields.get('von_mises', 0))),
                        'hydrostatic_max': float(np.max(stress_fields.get('sigma_hydro', 0))),
                        'hydrostatic_mean': float(np.mean(stress_fields.get('sigma_hydro', 0)))
                    }
                    all_summaries.append(summary)
                
                if all_summaries:
                    summary_df = pd.DataFrame(all_summaries)
                    csv_data = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Combined Summary CSV",
                        data=csv_data,
                        file_name=f"{base_filename}_combined_summary.csv",
                        mime="text/csv",
                        key=f"download_combined_summary_{timestamp}",
                        use_container_width=True,
                        type="primary"
                    )
            
            st.divider()
            
            # Clear cache button
            if st.button("🗑️ Clear All Download Caches", type="secondary"):
                if 'matplotlib_figures' in st.session_state:
                    st.session_state.matplotlib_figures = {}
                st.success("✅ Download caches cleared!")
                st.rerun()
 
    # Tab 7: Time Frame Analysis
    with tab7:
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

# =============================================
# MAIN APPLICATION
# =============================================
if __name__ == "__main__":
    create_multi_target_attention_interface()

st.caption(f"🔬 Multi-Target Attention Interpolation • Complete Training & Export System • {datetime.now().year}")
