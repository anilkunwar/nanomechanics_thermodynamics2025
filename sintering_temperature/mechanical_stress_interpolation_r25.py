import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm, ListedColormap
from matplotlib.cm import get_cmap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from io import BytesIO
import warnings
import json
import zipfile
import itertools
from typing import List, Dict, Any, Optional, Tuple, Union
import seaborn as sns
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION - ENHANCED
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
VISUALIZATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualization_outputs")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.figsize': (10, 8),
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Enhanced Colormap options with perceptual uniformity
COLORMAP_OPTIONS = {
    'Sequential (Perceptually Uniform)': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo'],
    'Diverging': ['RdBu_r', 'coolwarm', 'bwr', 'seismic', 'Spectral_r', 'PiYG', 'PRGn', 'PuOr'],
    'Scientific': ['jet', 'hot', 'hot_r', 'afmhot', 'afmhot_r', 'gnuplot', 'gnuplot2'],
    'Rainbow': ['rainbow', 'gist_rainbow', 'hsv', 'twilight', 'twilight_shifted'],
    'Grayscale': ['gray', 'bone', 'gist_gray', 'gist_yarg', 'binary']
}

# Define aspect ratios for different plot types
ASPECT_RATIOS = {
    'heatmap': 'equal',  # 1:1 aspect ratio
    'line_plot': 'auto',
    '3d_plot': 'auto',
    'stat_plot': 'auto'
}

# =============================================
# ENHANCED SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    """Enhanced solution loader with physics-aware processing and visualization support"""
    
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        self.cache = {}
        self.source_previews = {}  # Cache for source simulation previews

    def _ensure_directory(self):
        """Create solutions directory if it doesn't exist"""
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)

    def scan_solutions(self) -> List[Dict[str, Any]]:
        """Scan directory for solution files with enhanced metadata"""
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
                    'format': 'pkl' if file_path.endswith(('.pkl', '.pickle')) else 'pt',
                    'preview_generated': False
                }
                file_info.append(info)
            except Exception as e:
                st.warning(f"Error processing file {file_path}: {e}")
                continue
        
        return file_info

    def generate_source_preview(self, solution_index):
        """Generate preview visualization for a source solution"""
        if solution_index in self.source_previews:
            return self.source_previews[solution_index]
        
        try:
            solution = st.session_state.solutions[solution_index]
            params = solution.get('params', {})
            history = solution.get('history', [])
            
            if not history:
                return None
                
            last_frame = history[-1]
            stresses = last_frame.get('stresses', {})
            
            if not stresses:
                return None
            
            # Extract stress fields
            von_mises = stresses.get('von_mises')
            sigma_hydro = stresses.get('sigma_hydro')
            sigma_mag = stresses.get('sigma_mag')
            
            if von_mises is None:
                # Compute von Mises if not directly available
                if all(k in stresses for k in ['sigma_xx', 'sigma_yy', 'sigma_zz', 'tau_xy']):
                    sxx = stresses['sigma_xx']
                    syy = stresses['sigma_yy']
                    szz = stresses.get('sigma_zz', np.zeros_like(sxx))
                    txy = stresses['tau_xy']
                    tyz = stresses.get('tau_yz', np.zeros_like(sxx))
                    tzx = stresses.get('tau_zx', np.zeros_like(sxx))
                    
                    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 +
                                             6*(txy**2 + tyz**2 + tzx**2)))
            
            # Create preview figure with publication-quality styling
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12), 
                                                        gridspec_kw={'height_ratios': [3, 2]})
            
            # Plot von Mises stress
            if von_mises is not None:
                im1 = ax1.imshow(von_mises, cmap='viridis', aspect='equal')
                plt.colorbar(im1, ax=ax1, label='Von Mises Stress (GPa)', pad=0.02)
                ax1.set_title('Von Mises Stress Field', fontsize=16, fontweight='bold', pad=15)
                ax1.set_xlabel('X Position', fontsize=14)
                ax1.set_ylabel('Y Position', fontsize=14)
            
            # Create orientation diagram
            theta_deg = np.degrees(params.get('theta', 0))
            self._create_orientation_diagram(ax2, theta_deg, params.get('defect_type', 'Unknown'))
            
            # Plot hydrostatic stress if available
            if sigma_hydro is not None:
                im3 = ax3.imshow(sigma_hydro, cmap='RdBu_r', aspect='equal', 
                                vmin=-np.max(np.abs(sigma_hydro)), vmax=np.max(np.abs(sigma_hydro)))
                plt.colorbar(im3, ax=ax3, label='Hydrostatic Stress (GPa)', pad=0.02)
                ax3.set_title('Hydrostatic Stress Field', fontsize=16, fontweight='bold', pad=15)
                ax3.set_xlabel('X Position', fontsize=14)
                ax3.set_ylabel('Y Position', fontsize=14)
            
            # Plot statistics
            if von_mises is not None:
                self._plot_stress_statistics(ax4, von_mises, sigma_hydro, sigma_mag)
            
            # Add title with parameters
            defect_type = params.get('defect_type', 'Unknown')
            eps0 = params.get('eps0', 0.0)
            kappa = params.get('kappa', 0.0)
            shape = params.get('shape', 'Unknown')
            
            fig.suptitle(f'Source Simulation: {defect_type} at θ={theta_deg:.1f}° | ε₀={eps0:.2f}, κ={kappa:.2f}, Shape={shape}', 
                         fontsize=18, fontweight='bold', y=0.95)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Convert to Plotly figure for interactivity
            plotly_fig = go.Figure()
            
            # Store the matplotlib figure and close it to free memory
            self.source_previews[solution_index] = fig
            plt.close(fig)
            
            return fig
            
        except Exception as e:
            st.error(f"Error generating preview for solution {solution_index}: {e}")
            return None
    
    def _create_orientation_diagram(self, ax, theta_deg, defect_type):
        """Create a diagram showing the angular orientation"""
        # Create circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1.5)
        ax.add_patch(circle)
        
        # Draw orientation line
        theta_rad = np.radians(theta_deg)
        x_end = np.cos(theta_rad)
        y_end = np.sin(theta_rad)
        
        # Draw orientation line with arrow
        ax.arrow(0, 0, x_end*0.95, y_end*0.95, 
                 head_width=0.08, head_length=0.1, fc='blue', ec='blue', linewidth=2.5)
        
        # Add angle annotation
        arc_radius = 0.4
        arc_theta = np.linspace(0, theta_rad, 100)
        arc_x = arc_radius * np.cos(arc_theta)
        arc_y = arc_radius * np.sin(arc_theta)
        ax.plot(arc_x, arc_y, 'r-', linewidth=2)
        
        # Add angle text
        mid_angle = theta_rad / 2
        text_x = (arc_radius + 0.1) * np.cos(mid_angle)
        text_y = (arc_radius + 0.1) * np.sin(mid_angle)
        ax.text(text_x, text_y, f'θ = {theta_deg:.1f}°', 
                ha='center', va='center', fontsize=14, fontweight='bold', color='red')
        
        # Add defect type annotation
        ax.text(0, -1.2, f'Defect: {defect_type}', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Set limits and remove axes
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Orientation Diagram', fontsize=16, fontweight='bold', pad=15)
    
    def _plot_stress_statistics(self, ax, von_mises, sigma_hydro=None, sigma_mag=None):
        """Plot stress statistics for the source simulation"""
        data = []
        labels = []
        colors = []
        
        if von_mises is not None:
            data.append(von_mises.flatten())
            labels.append('Von Mises')
            colors.append('blue')
        
        if sigma_hydro is not None:
            data.append(sigma_hydro.flatten())
            labels.append('Hydrostatic')
            colors.append('green')
        
        if sigma_mag is not None:
            data.append(sigma_mag.flatten())
            labels.append('Magnitude')
            colors.append('red')
        
        if not data:
            return
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        
        # Style the box plot
        for i, box in enumerate(bp['boxes']):
            box.set(color=colors[i], linewidth=2)
            box.set(facecolor=colors[i], alpha=0.3)
        
        for i, whisker in enumerate(bp['whiskers']):
            whisker.set(color=colors[i//2], linewidth=1.5)
        
        for i, cap in enumerate(bp['caps']):
            cap.set(color=colors[i//2], linewidth=1.5)
        
        for i, median in enumerate(bp['medians']):
            median.set(color='black', linewidth=2.5)
        
        ax.set_ylabel('Stress (GPa)', fontsize=14)
        ax.set_title('Stress Distribution Statistics', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def read_simulation_file(self, file_path, format_type='auto'):
        """Read simulation file with physics-aware processing"""
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
            
            # Standardize data structure
            standardized = self._standardize_data(data, file_path)
            return standardized
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None

    def _standardize_data(self, data, file_path):
        """Standardize simulation data with physics metadata"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {
                'filename': os.path.basename(file_path),
                'loaded_at': datetime.now().isoformat(),
                'physics_processed': False
            }
        }
        
        try:
            if isinstance(data, dict):
                # Extract parameters
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                
                # Extract history
                if 'history' in data:
                    history = data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        # Convert dict to list
                        history_list = []
                        for key in sorted(history.keys()):
                            if isinstance(history[key], dict):
                                history_list.append(history[key])
                        standardized['history'] = history_list
                
                # Extract additional metadata
                if 'metadata' in data:
                    standardized['metadata'].update(data['metadata'])
                
                # Convert tensors to numpy arrays
                self._convert_tensors(standardized)
        except Exception as e:
            st.error(f"Standardization error: {e}")
            standardized['metadata']['error'] = str(e)
        
        return standardized

    def _convert_tensors(self, data):
        """Convert PyTorch tensors to numpy arrays recursively"""
        if isinstance(data, dict):
            for key, value in data.items():
                if torch.is_tensor(value):
                    data[key] = value.cpu().numpy()
                elif isinstance(value, (dict, list)):
                    self._convert_tensors(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if torch.is_tensor(item):
                    data[i] = item.cpu().numpy()
                elif isinstance(item, (dict, list)):
                    self._convert_tensors(item)

    def load_all_solutions(self, use_cache=True, max_files=None):
        """Load all solutions with physics processing"""
        solutions = []
        file_info = self.scan_solutions()
        
        if max_files:
            file_info = file_info[:max_files]
        
        if not file_info:
            return solutions
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_info_item in enumerate(file_info):
            progress = (i + 1) / len(file_info)
            progress_bar.progress(progress)
            status_text.text(f"Loading {file_info_item['filename']} ({i+1}/{len(file_info)})")
            
            cache_key = file_info_item['filename']
            if use_cache and cache_key in self.cache:
                solutions.append(self.cache[cache_key])
                continue
            
            solution = self.read_simulation_file(file_info_item['path'])
            if solution:
                self.cache[cache_key] = solution
                solutions.append(solution)
        
        progress_bar.empty()
        status_text.empty()
        return solutions

# =============================================
# POSITIONAL ENCODING FOR TRANSFORMER
# =============================================
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer with improved stability"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass with proper dimension handling"""
        batch_size, seq_len, d_model = x.shape
        return x + self.pe[:seq_len, :].unsqueeze(0)

# =============================================
# TRANSFORMER SPATIAL INTERPOLATOR
# =============================================
class TransformerSpatialInterpolator:
    """Transformer-inspired stress interpolator with spatial locality regularization and orientation awareness"""
    
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=0.2, temperature=1.0):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection - expects exactly 18 input features with enhanced orientation encoding
        self.input_proj = nn.Linear(18, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Orientation-aware attention weights
        self.orientation_weights = None

    def debug_feature_dimensions(self, params_list, target_angle_deg):
        """Debug method to check feature dimensions"""
        encoded = self.encode_parameters(params_list, target_angle_deg)
        print(f"Debug: Encoded shape: {encoded.shape}")
        print(f"Debug: Number of features: {encoded.shape[1]}")
        
        # Print first encoded vector
        if len(params_list) > 0:
            print(f"Debug: First encoded vector: {encoded[0]}")
            print(f"Debug: Number of non-zero elements: {torch.sum(encoded[0] != 0).item()}")
        
        return encoded.shape

    def compute_positional_weights(self, source_params, target_params):
        """Compute spatial locality weights based on parameter similarity with orientation awareness"""
        weights = []
        target_theta = target_params.get('theta', 0.0)
        target_theta_deg = np.degrees(target_theta)
        
        for src in source_params:
            # Compute parameter distance
            param_dist = 0.0
            weight_factors = {}
            
            # Compare key parameters with appropriate scaling
            key_params = [
                ('eps0', 3.0, 0.3),    # Eigenstrain, max value 3.0, weight 0.3
                ('kappa', 2.0, 0.2),   # Material parameter, max value 2.0, weight 0.2
                ('theta', np.pi, 0.4), # Orientation angle, max value π, weight 0.4
                ('defect_type', None, 0.1) # Defect type categorical similarity, weight 0.1
            ]
            
            for param, max_val, weight in key_params:
                if param == 'theta':
                    # Angular distance (cyclic)
                    src_theta = src.get('theta', 0.0)
                    tgt_theta = target_params.get('theta', 0.0)
                    diff = abs(src_theta - tgt_theta)
                    diff = min(diff, 2*np.pi - diff)  # Handle periodicity
                    normalized_diff = diff / np.pi
                    param_dist += weight * normalized_diff
                    weight_factors[param] = normalized_diff
                    
                elif param == 'defect_type':
                    # Categorical similarity
                    src_defect = src.get('defect_type', 'Unknown')
                    tgt_defect = target_params.get('defect_type', 'Unknown')
                    defect_diff = 0.0 if src_defect == tgt_defect else 1.0
                    param_dist += weight * defect_diff
                    weight_factors[param] = defect_diff
                    
                elif param in src and param in target_params and max_val is not None:
                    # Normalized Euclidean distance
                    src_val = src.get(param, 0)
                    tgt_val = target_params.get(param, 0)
                    normalized_diff = abs(src_val - tgt_val) / max_val
                    param_dist += weight * normalized_diff
                    weight_factors[param] = normalized_diff
            
            # Compute orientation-specific weight
            src_theta_deg = np.degrees(src.get('theta', 0.0))
            angle_diff = abs(src_theta_deg - target_theta_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)  # Handle circularity
            orientation_weight = np.exp(-0.5 * (angle_diff / 30.0) ** 2)  # 30° is characteristic width
            
            # Apply Gaussian kernel with orientation awareness
            spatial_weight = np.exp(-0.5 * (param_dist / self.spatial_sigma) ** 2)
            combined_weight = spatial_weight * orientation_weight
            
            weights.append({
                'weight': combined_weight,
                'factors': weight_factors,
                'orientation_angle': src_theta_deg,
                'orientation_diff': angle_diff,
                'orientation_weight': orientation_weight
            })
        
        # Extract just the weights for return
        weight_values = np.array([w['weight'] for w in weights])
        
        # Normalize weights
        if np.sum(weight_values) > 0:
            weight_values = weight_values / np.sum(weight_values)
        
        return weight_values, weights

    def encode_parameters(self, params_list, target_angle_deg):
        """Encode parameters into transformer input with enhanced orientation representation"""
        encoded = []
        target_angle_rad = np.radians(target_angle_deg)
        
        for params in params_list:
            # Create feature vector with 18 features
            features = []
            
            # 1. Numeric features (scaled)
            features.append(params.get('eps0', 0.707) / 3.0)  # Eigenstrain
            features.append(params.get('kappa', 0.6) / 2.0)   # Material parameter
            
            # 2. Orientation features (multiple representations for robustness)
            theta = params.get('theta', 0.0)
            theta_deg = np.degrees(theta)
            
            # Basic scaled orientation
            features.append(theta / np.pi)  # 0 to 1 scaling
            
            # Angular difference to target
            angle_diff = abs(theta_deg - target_angle_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)  # Handle circularity
            features.append(angle_diff / 180.0)  # Normalized to 0-1
            
            # Periodic features (sin/cos) for better angular representation
            features.append(np.sin(theta))
            features.append(np.cos(theta))
            features.append(np.sin(2*theta))  # Double angle for symmetry
            features.append(np.cos(2*theta))
            
            # Gaussian kernel features for orientation similarity
            kernel_width = 30.0  # degrees
            orientation_similarity = np.exp(-0.5 * (angle_diff / kernel_width) ** 2)
            features.append(orientation_similarity)
            
            # 3. One-hot encoding for defect type (4 features)
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            
            # 4. Shape encoding (4 features)
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
            
            # 5. Additional orientation context features
            # Distance to habit plane (54.7° for FCC materials)
            habit_plane_angle = 54.7
            habit_distance = abs(theta_deg - habit_plane_angle)
            habit_distance = min(habit_distance, 360 - habit_distance)
            features.append(np.exp(-habit_distance / 15.0))
            
            # Orientation quadrant encoding
            quadrant = int(theta_deg // 90) % 4
            quadrant_features = [0.0, 0.0, 0.0, 0.0]
            quadrant_features[quadrant] = 1.0
            features.extend(quadrant_features)
            
            # Verify we have exactly 18 features
            if len(features) != 18:
                st.warning(f"Warning: Expected 18 features, got {len(features)}. Padding or truncating.")
                # Pad with zeros if fewer than 18
                while len(features) < 18:
                    features.append(0.0)
                # Truncate if more than 18
                features = features[:18]
            
            encoded.append(features)
        
        return torch.FloatTensor(encoded)

    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        """Interpolate full spatial stress fields using transformer attention with orientation awareness"""
        if not sources:
            st.warning("No sources provided for interpolation.")
            return None
        
        try:
            # Extract source parameters and fields
            source_params = []
            source_fields = []
            valid_sources = []
            
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    st.warning(f"Skipping source {i}: missing params or history")
                    continue
                
                history = src['history']
                if not history or not isinstance(history[-1], dict):
                    st.warning(f"Skipping source {i}: invalid history format")
                    continue
                
                last_frame = history[-1]
                if 'stresses' not in last_frame:
                    st.warning(f"Skipping source {i}: no stress fields found")
                    continue
                
                stress_fields = last_frame['stresses']
                
                # Get von Mises stress
                if 'von_mises' in stress_fields:
                    vm = stress_fields['von_mises']
                else:
                    # Compute von Mises from components
                    vm = self.compute_von_mises(stress_fields)
                
                # Get hydrostatic stress
                if 'sigma_hydro' in stress_fields:
                    hydro = stress_fields['sigma_hydro']
                else:
                    hydro = self.compute_hydrostatic(stress_fields)
                
                # Get stress magnitude
                if 'sigma_mag' in stress_fields:
                    mag = stress_fields['sigma_mag']
                else:
                    mag = np.sqrt(vm**2 + hydro**2)
                
                source_params.append(src['params'])
                source_fields.append({
                    'von_mises': vm,
                    'sigma_hydro': hydro,
                    'sigma_mag': mag
                })
                valid_sources.append(src)
            
            if not source_params or not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Get common shape (use median shape to avoid extreme resizing)
            shapes = [f['von_mises'].shape for f in source_fields]
            median_shape = tuple(int(np.median([s[i] for s in shapes])) for i in range(2))
            
            # Resize fields to common shape if needed
            resized_fields = []
            for fields in source_fields:
                resized = {}
                for key, field in fields.items():
                    if field.shape != median_shape:
                        factors = [t/s for t, s in zip(median_shape, field.shape)]
                        resized[key] = zoom(field, factors, order=1)
                    else:
                        resized[key] = field
                resized_fields.append(resized)
            
            source_fields = resized_fields
            shape = median_shape
            
            # Compute positional weights with orientation awareness
            pos_weights, weight_details = self.compute_positional_weights(source_params, target_params)
            
            # Encode parameters for transformer
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Prepare transformer input
            seq_len = len(source_features) + 1  # Sources + target
            
            # Create sequence: [source1, source2, ..., target]
            all_features = torch.cat([source_features, target_features], dim=0).unsqueeze(0)
            
            # Apply input projection and positional encoding
            proj_features = self.input_proj(all_features)
            proj_features = self.pos_encoder(proj_features)
            
            # Transformer encoding
            transformer_output = self.transformer(proj_features)
            
            # Extract attention weights from the last layer
            # For simplicity, we'll use the attention to the target token
            # In a real implementation, we would extract the attention weights from the transformer layers
            target_rep = transformer_output[:, -1, :]  # Target is last in sequence
            source_reps = transformer_output[:, :-1, :]  # All sources
            
            # Compute attention scores
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1) / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            transformer_weights = torch.softmax(attn_scores, dim=-1).squeeze().detach().numpy()
            
            # Combine positional and transformer weights
            combined_weights = 0.6 * transformer_weights + 0.4 * pos_weights
            combined_weights = combined_weights / np.sum(combined_weights)
            
            # Interpolate spatial fields
            interpolated_fields = {}
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                interpolated = np.zeros(shape)
                for i, fields in enumerate(source_fields):
                    if component in fields:
                        interpolated += combined_weights[i] * fields[component]
                interpolated_fields[component] = interpolated
            
            # Compute statistics
            stats = {}
            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                if component in interpolated_fields:
                    field = interpolated_fields[component]
                    if component == 'sigma_hydro':
                        stats[component] = {
                            'max_tension': float(np.max(field)),
                            'max_compression': float(np.min(field)),
                            'mean': float(np.mean(field)),
                            'std': float(np.std(field))
                        }
                    else:
                        stats[component] = {
                            'max': float(np.max(field)),
                            'min': float(np.min(field)),
                            'mean': float(np.mean(field)),
                            'std': float(np.std(field))
                        }
            
            # Create orientation details
            orientation_details = {
                'target_angle_deg': target_angle_deg,
                'target_angle_rad': np.radians(target_angle_deg),
                'source_angles': [np.degrees(p.get('theta', 0)) for p in source_params],
                'angular_distances': [w['orientation_diff'] for w in weight_details],
                'orientation_weights': [w['orientation_weight'] for w in weight_details]
            }
            
            return {
                'fields': interpolated_fields,
                'weights': {
                    'transformer': transformer_weights.tolist(),
                    'positional': pos_weights.tolist(),
                    'combined': combined_weights.tolist(),
                    'details': weight_details
                },
                'statistics': stats,
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'shape': shape,
                'num_sources': len(source_fields),
                'orientation': orientation_details,
                'source_params': source_params,
                'valid_sources': valid_sources
            }
            
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None

    def compute_von_mises(self, stress_fields):
        """Compute von Mises stress from stress components"""
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz', 'tau_xy']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            txy = stress_fields['tau_xy']
            tyz = stress_fields.get('tau_yz', np.zeros_like(sxx))
            tzx = stress_fields.get('tau_zx', np.zeros_like(sxx))
            
            von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 +
                                     6*(txy**2 + tyz**2 + tzx**2)))
            return von_mises
        
        # Default fallback
        default_shape = (100, 100)
        if 'sigma_xx' in stress_fields:
            default_shape = stress_fields['sigma_xx'].shape
        return np.zeros(default_shape)

    def compute_hydrostatic(self, stress_fields):
        """Compute hydrostatic stress from stress components"""
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields['sigma_zz']
            return (sxx + syy + szz) / 3
        
        # Default fallback
        default_shape = (100, 100)
        if 'sigma_xx' in stress_fields:
            default_shape = stress_fields['sigma_xx'].shape
        return np.zeros(default_shape)

# =============================================
# ADVANCED HEATMAP VISUALIZER
# =============================================
class AdvancedHeatMapVisualizer:
    """Enhanced heat map visualizer with publication-quality output and interactive features"""
    
    def __init__(self):
        self.colormaps = COLORMAP_OPTIONS
        self.plotly_templates = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
        
    def create_stress_heatmap(self, stress_field, title="Stress Heat Map",
                            cmap_name='viridis', figsize=(10, 8),
                            colorbar_label="Stress (GPa)", vmin=None, vmax=None,
                            show_stats=True, show_colorbar=True, aspect_ratio='equal',
                            theta=None, defect_type=None, show_orientation=True):
        """Create publication-quality heat map with orientation indicator"""
        # Create figure with publication-quality settings
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        # Get colormap with fallback
        if cmap_name not in plt.colormaps():
            cmap_name = 'viridis'
            st.warning(f"Colormap {cmap_name} not available. Using viridis instead.")
        
        cmap = plt.get_cmap(cmap_name)
        
        # Determine vmin and vmax if not provided
        if vmin is None:
            vmin = np.nanmin(stress_field)
        if vmax is None:
            vmax = np.nanmax(stress_field)
        
        # Create heatmap
        im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect=aspect_ratio, interpolation='bilinear')
        
        # Add colorbar with proper formatting
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(colorbar_label, fontsize=16, fontweight='bold', labelpad=10)
            cbar.ax.tick_params(labelsize=14)
        
        # Set title with orientation information if available
        full_title = title
        if theta is not None:
            full_title += f" (θ = {theta:.1f}°)"
        if defect_type:
            full_title += f" - {defect_type}"
        
        ax.set_title(full_title, fontsize=18, fontweight='bold', pad=15)
        
        # Set axis labels with larger font
        ax.set_xlabel('X Position (nm)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y Position (nm)', fontsize=16, fontweight='bold', labelpad=10)
        
        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add grid with subtle styling
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
        
        # Add orientation indicator if requested
        if show_orientation and theta is not None:
            self._add_orientation_indicator(ax, theta, stress_field.shape)
        
        # Add statistics annotation
        if show_stats:
            stats_text = (f"Max: {np.max(stress_field):.3f} GPa\n"
                         f"Min: {np.min(stress_field):.3f} GPa\n"
                         f"Mean: {np.mean(stress_field):.3f} GPa")
            
            # Add text box with statistics
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=14, verticalalignment='top', bbox=props,
                   fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _add_orientation_indicator(self, ax, theta_deg, field_shape):
        """Add visual indicator of orientation angle to the plot"""
        # Calculate center position
        center_x = field_shape[1] / 2
        center_y = field_shape[0] / 2
        
        # Calculate end point based on angle
        theta_rad = np.radians(theta_deg)
        radius = min(field_shape) / 3  # Radius is 1/3 of the smaller dimension
        end_x = center_x + radius * np.cos(theta_rad)
        end_y = center_y - radius * np.sin(theta_rad)  # Negative because y increases downward in image coordinates
        
        # Draw orientation line with arrow
        ax.arrow(center_x, center_y, end_x - center_x, end_y - center_y,
                head_width=radius/10, head_length=radius/8, fc='red', ec='darkred',
                linewidth=2.5, length_includes_head=True, alpha=0.8)
        
        # Draw circle at center
        circle = plt.Circle((center_x, center_y), radius/15, color='red', alpha=0.8)
        ax.add_patch(circle)
        
        # Add angle annotation
        arc_radius = radius / 2
        arc_theta = np.linspace(0, theta_rad, 100)
        if theta_deg > 0:
            arc_x = center_x + arc_radius * np.cos(arc_theta)
            arc_y = center_y - arc_radius * np.sin(arc_theta)  # Negative for image coordinates
            ax.plot(arc_x, arc_y, 'r-', linewidth=2, alpha=0.8)
            
            # Add angle text at midpoint of arc
            mid_angle = theta_rad / 2
            text_x = center_x + (arc_radius + 5) * np.cos(mid_angle)
            text_y = center_y - (arc_radius + 5) * np.sin(mid_angle)  # Negative for image coordinates
            ax.text(text_x, text_y, f'{theta_deg:.1f}°', color='red', fontsize=14,
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    def create_interactive_heatmap(self, stress_field, title="Stress Heat Map",
                                 cmap_name='viridis', width=800, height=700,
                                 theta=None, defect_type=None, template="plotly_white"):
        """Create interactive heatmap with Plotly with proper hover data and orientation display"""
        # Create hover text with coordinates and values
        hover_text = np.empty(stress_field.shape, dtype=object)
        for i in range(stress_field.shape[0]):
            for j in range(stress_field.shape[1]):
                hover_text[i, j] = f"Position: ({j}, {i})<br>Stress: {stress_field[i, j]:.4f} GPa"
        
        # Determine color scale limits
        zmin = np.nanmin(stress_field)
        zmax = np.nanmax(stress_field)
        
        # Create heatmap trace with proper hover information
        heatmap_trace = go.Heatmap(
            z=stress_field,
            colorscale=cmap_name,
            zmin=zmin,
            zmax=zmax,
            text=hover_text,
            hoverinfo='text',
            colorbar=dict(
                title=dict(
                    text="Stress (GPa)",
                    font=dict(size=16, family='Arial', color='black')
                ),
                tickfont=dict(size=14, family='Arial'),
                len=0.8,
                thickness=20
            ),
            showscale=True
        )
        
        # Create figure with template
        fig = go.Figure(data=[heatmap_trace])
        
        # Update layout with publication-quality settings
        full_title = title
        if theta is not None:
            full_title += f" (θ = {theta:.1f}°)"
        if defect_type:
            full_title += f" - {defect_type}"
        
        fig.update_layout(
            title=dict(
                text=full_title,
                font=dict(size=22, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            width=width,
            height=height,
            template=template,
            xaxis=dict(
                title=dict(
                    text="X Position (nm)",
                    font=dict(size=18, family='Arial', color='black')
                ),
                tickfont=dict(size=14, family='Arial'),
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title=dict(
                    text="Y Position (nm)",
                    font=dict(size=18, family='Arial', color='black')
                ),
                tickfont=dict(size=14, family='Arial'),
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=False,
                autorange="reversed"  # Match matplotlib's coordinate system
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Arial"
            ),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='white'
        )
        
        # Add orientation indicator if theta is provided
        if theta is not None:
            self._add_plotly_orientation_indicator(fig, theta, stress_field.shape)
        
        return fig
    
    def _add_plotly_orientation_indicator(self, fig, theta_deg, field_shape):
        """Add orientation indicator to Plotly figure"""
        # Calculate center and end points
        center_x = field_shape[1] / 2
        center_y = field_shape[0] / 2
        
        theta_rad = np.radians(theta_deg)
        radius = min(field_shape) / 3
        
        end_x = center_x + radius * np.cos(theta_rad)
        end_y = center_y - radius * np.sin(theta_rad)  # Negative for image coordinates
        
        # Add orientation line
        fig.add_shape(
            type="line",
            x0=center_x, y0=center_y,
            x1=end_x, y1=end_y,
            line=dict(color="red", width=3, dash="solid"),
            opacity=0.8
        )
        
        # Add arrowhead (triangle)
        arrow_size = radius / 10
        arrow_angle = theta_rad - np.pi
        
        # Calculate arrowhead points
        arrow_x1 = end_x + arrow_size * np.cos(arrow_angle + np.pi/6)
        arrow_y1 = end_y - arrow_size * np.sin(arrow_angle + np.pi/6)
        
        arrow_x2 = end_x + arrow_size * np.cos(arrow_angle - np.pi/6)
        arrow_y2 = end_y - arrow_size * np.sin(arrow_angle - np.pi/6)
        
        # Add arrowhead triangle
        fig.add_shape(
            type="path",
            path=f"M {end_x},{end_y} L {arrow_x1},{arrow_y1} L {arrow_x2},{arrow_y2} Z",
            fillcolor="red",
            line=dict(color="darkred", width=1),
            opacity=0.8
        )
        
        # Add angle annotation
        if theta_deg > 0:
            arc_radius = radius / 2
            mid_angle = theta_rad / 2
            text_x = center_x + (arc_radius + 5) * np.cos(mid_angle)
            text_y = center_y - (arc_radius + 5) * np.sin(mid_angle)
            
            fig.add_annotation(
                x=text_x,
                y=text_y,
                text=f"{theta_deg:.1f}°",
                showarrow=False,
                font=dict(size=16, color="red", family="Arial"),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="red",
                borderwidth=1
            )
    
    def create_interactive_3d_surface(self, stress_field, title="3D Stress Surface",
                                     cmap_name='viridis', width=900, height=800,
                                     theta=None, defect_type=None, template="plotly_white"):
        """Create interactive 3D surface plot with Plotly"""
        # Create grid coordinates
        x = np.arange(stress_field.shape[1])
        y = np.arange(stress_field.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=stress_field,
            colorscale=cmap_name,
            colorbar=dict(
                title=dict(
                    text="Stress (GPa)",
                    font=dict(size=16, family='Arial', color='black')
                ),
                tickfont=dict(size=14, family='Arial'),
                len=0.8
            ),
            contours={
                "x": {"show": True, "start": 0, "end": stress_field.shape[1], "size": stress_field.shape[1]//10, "color": "white"},
                "y": {"show": True, "start": 0, "end": stress_field.shape[0], "size": stress_field.shape[0]//10, "color": "white"},
                "z": {"show": True, "start": np.min(stress_field), "end": np.max(stress_field), "size": (np.max(stress_field)-np.min(stress_field))/10, "color": "white"}
            },
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.2,
                roughness=0.3
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=200
            )
        )])
        
        # Set title with orientation information
        full_title = title
        if theta is not None:
            full_title += f" (θ = {theta:.1f}°)"
        if defect_type:
            full_title += f" - {defect_type}"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=full_title,
                font=dict(size=22, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            width=width,
            height=height,
            template=template,
            scene=dict(
                xaxis=dict(
                    title=dict(
                        text="X Position (nm)",
                        font=dict(size=16, family='Arial', color='black')
                    ),
                    tickfont=dict(size=14, family='Arial'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    title=dict(
                        text="Y Position (nm)",
                        font=dict(size=16, family='Arial', color='black')
                    ),
                    tickfont=dict(size=14, family='Arial'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                zaxis=dict(
                    title=dict(
                        text="Stress (GPa)",
                        font=dict(size=16, family='Arial', color='black')
                    ),
                    tickfont=dict(size=14, family='Arial'),
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            margin=dict(l=80, r=80, t=100, b=80),
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Arial"
            )
        )
        
        return fig
    
    def create_comparison_heatmaps(self, stress_fields_dict, cmap_name='viridis',
                                 figsize=(18, 12), titles=None, theta=None, defect_type=None):
        """Create comparison heatmaps for multiple stress components with orientation indicators"""
        n_components = len(stress_fields_dict)
        fig, axes = plt.subplots(1, n_components, figsize=figsize, dpi=300)
        
        if n_components == 1:
            axes = [axes]
        
        if titles is None:
            titles = list(stress_fields_dict.keys())
        
        # Custom titles and colorbars for each component
        component_info = {
            'von_mises': {'title': 'Von Mises Stress', 'cmap': cmap_name, 'label': 'Von Mises Stress (GPa)'},
            'sigma_hydro': {'title': 'Hydrostatic Stress', 'cmap': 'RdBu_r', 'label': 'Hydrostatic Stress (GPa)'},
            'sigma_mag': {'title': 'Stress Magnitude', 'cmap': cmap_name, 'label': 'Stress Magnitude (GPa)'}
        }
        
        for idx, ((component_name, stress_field), title) in enumerate(zip(stress_fields_dict.items(), titles)):
            ax = axes[idx]
            
            # Get component-specific info
            comp_info = component_info.get(component_name, 
                                          {'title': title, 'cmap': cmap_name, 'label': 'Stress (GPa)'})
            
            # Use component-specific colormap if available
            cmap = comp_info['cmap']
            
            # Special handling for hydrostatic stress (diverging colormap)
            vmin, vmax = None, None
            if component_name == 'sigma_hydro':
                abs_max = np.max(np.abs(stress_field))
                vmin, vmax = -abs_max, abs_max
            
            # Create heatmap
            im = ax.imshow(stress_field, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect='equal', interpolation='bilinear')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(comp_info['label'], fontsize=14, fontweight='bold', labelpad=10)
            cbar.ax.tick_params(labelsize=12)
            
            # Set title with component name
            plot_title = comp_info['title']
            if theta is not None:
                plot_title += f" (θ = {theta:.1f}°)"
            ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=10)
            
            # Set axis labels
            ax.set_xlabel('X Position (nm)', fontsize=14, labelpad=8)
            ax.set_ylabel('Y Position (nm)', fontsize=14, labelpad=8)
            
            # Add orientation indicator
            if theta is not None:
                self._add_orientation_indicator(ax, theta, stress_field.shape)
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Set overall title
        overall_title = f"Stress Component Comparison"
        if theta is not None:
            overall_title += f" (θ = {theta:.1f}°)"
        if defect_type:
            overall_title += f" - {defect_type}"
        
        fig.suptitle(overall_title, fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig
    
    def create_interactive_comparison(self, stress_fields, theta, defect_type, 
                                     cmap_name='viridis', template="plotly_white"):
        """Create interactive comparison of multiple stress fields"""
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Von Mises Stress", "Hydrostatic Stress", "Stress Magnitude"),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Component information
        components = [
            ('von_mises', 'Von Mises Stress', cmap_name),
            ('sigma_hydro', 'Hydrostatic Stress', 'RdBu_r'),
            ('sigma_mag', 'Stress Magnitude', cmap_name)
        ]
        
        # Add each component as a heatmap
        for i, (comp_name, title, cmap) in enumerate(components, 1):
            if comp_name not in stress_fields:
                continue
            
            field = stress_fields[comp_name]
            
            # Special handling for hydrostatic stress
            zmin, zmax = None, None
            if comp_name == 'sigma_hydro':
                abs_max = np.max(np.abs(field))
                zmin, zmax = -abs_max, abs_max
            
            # Create hover text
            hover_text = np.empty(field.shape, dtype=object)
            for r in range(field.shape[0]):
                for c in range(field.shape[1]):
                    hover_text[r, c] = f"Position: ({c}, {r})<br>Stress: {field[r, c]:.4f} GPa"
            
            # Add heatmap trace
            heatmap = go.Heatmap(
                z=field,
                colorscale=cmap,
                zmin=zmin,
                zmax=zmax,
                text=hover_text,
                hoverinfo='text',
                colorbar=dict(
                    title=dict(text="Stress (GPa)", font=dict(size=14)),
                    tickfont=dict(size=12),
                    len=0.7,
                    x=1.0 if i == 3 else (0.33 if i == 1 else 0.66),  # Position colorbars
                    y=0.5,
                    thickness=15
                ),
                name=title,
                showscale=(i == 3)  # Only show colorbar for last plot
            )
            
            fig.add_trace(heatmap, row=1, col=i)
            
            # Add orientation indicator to each subplot
            if theta is not None:
                self._add_plotly_orientation_indicator_subplot(fig, theta, field.shape, row=1, col=i)
        
        # Set title
        full_title = f"Stress Field Comparison - θ = {theta:.1f}°"
        if defect_type:
            full_title += f" - {defect_type}"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=full_title,
                font=dict(size=24, family="Arial Black", color='darkblue'),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            width=1200,
            height=500,
            template=template,
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(
                title_text="X Position (nm)",
                row=1, col=i,
                title_font=dict(size=14),
                tickfont=dict(size=12),
                scaleanchor=f"y{i}",
                scaleratio=1
            )
            fig.update_yaxes(
                title_text="Y Position (nm)" if i == 1 else None,
                row=1, col=i,
                title_font=dict(size=14),
                tickfont=dict(size=12),
                autorange="reversed"
            )
        
        return fig
    
    def _add_plotly_orientation_indicator_subplot(self, fig, theta_deg, field_shape, row=1, col=1):
        """Add orientation indicator to a subplot in a Plotly figure"""
        # Calculate center and end points
        center_x = field_shape[1] / 2
        center_y = field_shape[0] / 2
        
        theta_rad = np.radians(theta_deg)
        radius = min(field_shape) / 3
        
        end_x = center_x + radius * np.cos(theta_rad)
        end_y = center_y - radius * np.sin(theta_rad)  # Negative for image coordinates
        
        # Add orientation line
        fig.add_shape(
            type="line",
            x0=center_x, y0=center_y,
            x1=end_x, y1=end_y,
            line=dict(color="red", width=2, dash="solid"),
            opacity=0.8,
            row=row, col=col
        )
    
    def create_source_comparison_dashboard(self, sources, interpolated_result, 
                                         cmap_name='viridis', figsize=(20, 15)):
        """Create comprehensive dashboard comparing source simulations with interpolated result"""
        if not sources or not interpolated_result:
            return None
        
        # Get fields from interpolated result
        stress_fields = interpolated_result['fields']
        target_theta = interpolated_result['target_angle']
        defect_type = interpolated_result['target_params'].get('defect_type', 'Unknown')
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Interpolated von Mises (main plot)
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(stress_fields['von_mises'], cmap=cmap_name, aspect='equal')
        
        # Add colorbar
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar1.set_label('Von Mises Stress (GPa)', fontsize=14, fontweight='bold')
        
        # Set title with orientation
        ax1.set_title(f'Interpolated Von Mises Stress\nθ = {target_theta:.1f}° - {defect_type}', 
                     fontsize=16, fontweight='bold', pad=15)
        
        # Add orientation indicator
        self._add_orientation_indicator(ax1, target_theta, stress_fields['von_mises'].shape)
        
        ax1.set_xlabel('X Position (nm)', fontsize=14)
        ax1.set_ylabel('Y Position (nm)', fontsize=14)
        ax1.grid(True, alpha=0.2)
        
        # 2. Source contributions
        ax2 = fig.add_subplot(gs[0, 2:])
        weights = interpolated_result['weights']['combined']
        source_angles = [np.degrees(p.get('theta', 0)) for p in interpolated_result['source_params']]
        
        bars = ax2.bar(range(len(weights)), weights, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Source Simulation Index', fontsize=14)
        ax2.set_ylabel('Contribution Weight', fontsize=14)
        ax2.set_title('Source Contributions to Interpolation', fontsize=16, fontweight='bold', pad=15)
        
        # Annotate bars with angle information
        for i, (bar, angle) in enumerate(zip(bars, source_angles)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'θ={angle:.1f}°', 
                    ha='center', va='bottom', fontsize=12, rotation=45)
        
        ax2.set_ylim(0, max(0.2, max(weights) * 1.2))
        ax2.grid(True, alpha=0.3)
        
        # 3-5. Top 3 source simulations (von Mises)
        top_sources_idx = np.argsort(weights)[-3:][::-1]  # Get indices of top 3 sources
        
        for i, src_idx in enumerate(top_sources_idx):
            if src_idx >= len(sources):
                continue
            
            ax = fig.add_subplot(gs[1, i])
            source = sources[src_idx]
            
            # Get von Mises stress from source
            history = source['history']
            if history and 'stresses' in history[-1]:
                stresses = history[-1]['stresses']
                if 'von_mises' in stresses:
                    vm_field = stresses['von_mises']
                    
                    # Resize if needed
                    if vm_field.shape != stress_fields['von_mises'].shape:
                        factors = [t/s for t, s in zip(stress_fields['von_mises'].shape, vm_field.shape)]
                        vm_field = zoom(vm_field, factors, order=1)
                    
                    im = ax.imshow(vm_field, cmap=cmap_name, aspect='equal')
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_label('Stress (GPa)', fontsize=10)
                    
                    # Get source angle
                    src_theta = np.degrees(source['params'].get('theta', 0))
                    
                    ax.set_title(f'Source {src_idx+1}: θ={src_theta:.1f}°\nWeight: {weights[src_idx]:.3f}', 
                               fontsize=14, fontweight='bold', pad=10)
                    
                    # Add orientation indicator
                    self._add_orientation_indicator(ax, src_theta, vm_field.shape)
                    
                    ax.set_xlabel('X Position', fontsize=12)
                    ax.set_ylabel('Y Position', fontsize=12)
        
        # 6. Angular distribution of sources
        ax6 = fig.add_subplot(gs[2, 0])
        angles = source_angles
        weights_arr = np.array(weights)
        
        # Create polar plot
        theta_rad = np.radians(angles)
        ax6 = plt.subplot(gs[2, 0], polar=True)
        bars = ax6.bar(theta_rad, weights_arr, width=0.2, alpha=0.7, color='teal')
        
        # Highlight target angle
        target_rad = np.radians(target_theta)
        ax6.axvline(target_rad, color='red', linestyle='--', linewidth=2, label=f'Target θ={target_theta:.1f}°')
        
        ax6.set_title('Source Distribution by Orientation', fontsize=14, fontweight='bold', pad=15)
        ax6.legend(loc='upper right', fontsize=10)
        
        # 7-8. Statistics comparison
        ax7 = fig.add_subplot(gs[2, 1])
        ax8 = fig.add_subplot(gs[2, 2])
        
        # Extract statistics
        source_stats = []
        for src_idx in top_sources_idx:
            if src_idx < len(sources):
                source = sources[src_idx]
                history = source['history']
                if history and 'stresses' in history[-1]:
                    stresses = history[-1]['stresses']
                    if 'von_mises' in stresses:
                        vm = stresses['von_mises']
                        source_stats.append({
                            'max': np.max(vm),
                            'mean': np.mean(vm),
                            'std': np.std(vm)
                        })
        
        # Plot max values comparison
        x = np.arange(len(top_sources_idx))
        width = 0.35
        
        target_max = np.max(stress_fields['von_mises'])
        target_mean = np.mean(stress_fields['von_mises'])
        
        if source_stats:
            source_max = [stats['max'] for stats in source_stats]
            source_mean = [stats['mean'] for stats in source_stats]
            
            ax7.bar(x, source_max, width, label='Source Max', alpha=0.7, color='steelblue')
            ax7.axhline(y=target_max, color='red', linestyle='--', linewidth=2, 
                       label=f'Target Max: {target_max:.2f} GPa')
            ax7.set_title('Maximum Von Mises Stress Comparison', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Source Index')
            ax7.set_ylabel('Max Stress (GPa)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
            
            ax8.bar(x, source_mean, width, label='Source Mean', alpha=0.7, color='darkgreen')
            ax8.axhline(y=target_mean, color='red', linestyle='--', linewidth=2,
                       label=f'Target Mean: {target_mean:.2f} GPa')
            ax8.set_title('Mean Von Mises Stress Comparison', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Source Index')
            ax8.set_ylabel('Mean Stress (GPa)')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Parameter similarity heatmap
        ax9 = fig.add_subplot(gs[2, 3])
        
        if interpolated_result['weights']['details']:
            # Extract parameter similarities
            params = ['eps0', 'kappa', 'theta', 'defect_type']
            similarities = []
            source_labels = [f'Source {i+1}' for i in range(len(weights))]
            
            for detail in interpolated_result['weights']['details']:
                sim_row = []
                factors = detail['factors']
                for param in params:
                    sim_row.append(1.0 - factors.get(param, 0.0))  # Convert distance to similarity
                similarities.append(sim_row)
            
            # Create heatmap
            im9 = ax9.imshow(similarities, cmap='YlGnBu', aspect='auto')
            
            # Add colorbar
            divider = make_axes_locatable(ax9)
            cax9 = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im9, cax=cax9, label='Similarity')
            
            # Set labels
            ax9.set_xticks(np.arange(len(params)))
            ax9.set_xticklabels(params, rotation=45, ha='right')
            ax9.set_yticks(np.arange(len(source_labels)))
            ax9.set_yticklabels(source_labels)
            
            # Add text annotations
            for i in range(len(similarities)):
                for j in range(len(params)):
                    ax9.text(j, i, f'{similarities[i][j]:.2f}',
                            ha='center', va='center', color='black' if similarities[i][j] < 0.7 else 'white',
                            fontweight='bold')
            
            ax9.set_title('Parameter Similarity to Target', fontsize=14, fontweight='bold', pad=15)
        
        # Set overall title
        fig.suptitle(f'Source Simulations vs. Interpolated Result - θ = {target_theta:.1f}° - {defect_type}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def get_colormap_preview(self, cmap_name, figsize=(12, 1.5)):
        """Generate high-quality preview of a colormap"""
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        
        ax.set_title(f"Colormap: {cmap_name}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add value labels with background
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        
        ax.text(0, 0.5, "0.0", transform=ax.transAxes,
               va='center', ha='right', fontsize=14, fontweight='bold',
               bbox=props, pad=0.5)
        
        ax.text(1, 0.5, "1.0", transform=ax.transAxes,
               va='center', ha='left', fontsize=14, fontweight='bold',
               bbox=props, pad=0.5)
        
        plt.tight_layout()
        return fig
    
    def save_publication_figure(self, fig, filename, format='png', dpi=600):
        """Save figure with publication-quality settings"""
        fig.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, filename), 
                   format=format, dpi=dpi, bbox_inches='tight', pad_inches=0.1,
                   facecolor='white')
        return os.path.join(VISUALIZATION_OUTPUT_DIR, filename)

# =============================================
# ENHANCED RESULTS MANAGER
# =============================================
class EnhancedResultsManager:
    """Enhanced manager for exporting interpolation results with visualization support"""
    
    def __init__(self):
        pass
    
    def prepare_export_data(self, interpolation_result, visualization_params):
        """Prepare data for export with enhanced metadata"""
        result = interpolation_result.copy()
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'interpolation_method': 'transformer_spatial',
                'visualization_params': visualization_params,
                'source_files': [src['metadata']['filename'] for src in result.get('valid_sources', [])]
            },
            'result': {
                'target_angle': result['target_angle'],
                'target_params': result['target_params'],
                'shape': result['shape'],
                'statistics': result['statistics'],
                'weights': result['weights'],
                'num_sources': result.get('num_sources', 0),
                'orientation': result['orientation']
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for field_name, field_data in result['fields'].items():
            export_data['result'][f'{field_name}_data'] = field_data.tolist()
        
        return export_data
    
    def export_to_json(self, export_data, filename=None):
        """Export results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = export_data['result']['target_angle']
            defect = export_data['result']['target_params']['defect_type']
            filename = f"transformer_interpolation_theta_{theta:.1f}_{defect}_{timestamp}.json"
        
        json_str = json.dumps(export_data, indent=2, default=self._json_serializer)
        return json_str, filename
    
    def export_to_csv(self, interpolation_result, filename=None):
        """Export flattened field data to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename = f"stress_fields_theta_{theta:.1f}_{defect}_{timestamp}.csv"
        
        # Create DataFrame with flattened data
        data_dict = {}
        for field_name, field_data in interpolation_result['fields'].items():
            data_dict[field_name] = field_data.flatten()
        
        df = pd.DataFrame(data_dict)
        csv_str = df.to_csv(index=False)
        return csv_str, filename
    
    def export_fields_as_images(self, interpolation_result, visualizer, filename_prefix=None, 
                              cmap_name='viridis'):
        """Export stress fields as high-quality images"""
        if filename_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            theta = interpolation_result['target_angle']
            defect = interpolation_result['target_params']['defect_type']
            filename_prefix = f"stress_fields_theta_{theta:.1f}_{defect}_{timestamp}"
        
        saved_files = []
        
        # Export each stress component
        components = {
            'von_mises': 'Von Mises Stress',
            'sigma_hydro': 'Hydrostatic Stress',
            'sigma_mag': 'Stress Magnitude'
        }
        
        for comp_name, title in components.items():
            if comp_name in interpolation_result['fields']:
                field = interpolation_result['fields'][comp_name]
                theta = interpolation_result['target_angle']
                defect_type = interpolation_result['target_params']['defect_type']
                
                # Create high-quality figure
                fig = visualizer.create_stress_heatmap(
                    field,
                    title=title,
                    cmap_name=cmap_name,
                    figsize=(10, 8),
                    colorbar_label=f"{title} (GPa)",
                    theta=theta,
                    defect_type=defect_type,
                    show_stats=True
                )
                
                # Save with high DPI
                filename = f"{filename_prefix}_{comp_name}.png"
                saved_path = visualizer.save_publication_figure(fig, filename, dpi=600)
                saved_files.append(saved_path)
                plt.close(fig)
        
        return saved_files
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return str(obj)

# =============================================
# HELPER FUNCTIONS
# =============================================
def _calculate_entropy(weights):
    """Calculate entropy of weight distribution"""
    weights = np.array(weights)
    weights = weights[weights > 0]  # Remove zeros
    if len(weights) == 0:
        return 0.0
    weights = weights / weights.sum()
    return -np.sum(weights * np.log(weights + 1e-10))  # Add small epsilon to avoid log(0)

def _angular_distance(theta1, theta2):
    """Calculate angular distance between two angles in degrees"""
    diff = abs(theta1 - theta2) % 360
    return min(diff, 360 - diff)

def _create_angle_indicator(ax, angle_deg, radius=0.4, center=(0.5, 0.5), color='red'):
    """Create a visual indicator for an angle on a matplotlib axis"""
    angle_rad = np.radians(angle_deg)
    
    # Draw arc
    theta = np.linspace(0, angle_rad, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color=color, linewidth=2)
    
    # Draw lines
    ax.plot([center[0], center[0] + radius * np.cos(0)], 
            [center[1], center[1] + radius * np.sin(0)], 
            color=color, linewidth=2)
    ax.plot([center[0], center[0] + radius * np.cos(angle_rad)], 
            [center[1], center[1] + radius * np.sin(angle_rad)], 
            color=color, linewidth=2)
    
    # Add text
    mid_angle = angle_rad / 2
    text_x = center[0] + (radius + 0.05) * np.cos(mid_angle)
    text_y = center[1] + (radius + 0.05) * np.sin(mid_angle)
    ax.text(text_x, text_y, f'{angle_deg:.1f}°', color=color, fontsize=12,
           ha='center', va='center', fontweight='bold')

# =============================================
# MAIN APPLICATION - ENHANCED
# =============================================
def main():
    # Configure Streamlit page with better layout
    st.set_page_config(
        page_title="Advanced Transformer Stress Field Interpolation",
        layout="wide",
        page_icon="🔍",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 2.0rem !important;
        color: #1E293B !important;
        font-weight: 800 !important;
        border-left: 6px solid #3B82F6;
        padding-left: 1.2rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(to right, #f8fafc, #e2e8f0);
        border-radius: 0 8px 8px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .info-box {
        background-color: #F0F9FF;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
        background: #f8fafc;
        padding: 0.5rem;
        border-radius: 0.8rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 8px 8px 0 0;
        gap: 1.2rem;
        padding-top: 12px;
        padding-bottom: 12px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        box-shadow: 0 -2px 0 #1E3A8A inset;
    }
    
    .source-preview-container {
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        background: #f8fafc;
        transition: border-color 0.3s ease;
    }
    
    .source-preview-container:hover {
        border-color: #3B82F6;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 2px solid #3b82f6;
    }
    
    .orientation-diagram {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.5rem 0;
    }
    
    .parameter-card {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem;
        border: 1px solid #cbd5e1;
    }
    
    .weight-visualization {
        height: 400px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with enhanced styling
    st.markdown('<h1 class="main-header">🔍 Advanced Transformer-Based Stress Field Interpolation</h1>', unsafe_allow_html=True)
    
    # Description with enhanced styling
    st.markdown("""
    <div class="info-box">
        <strong>🔬 Physics-aware stress interpolation using transformer architecture with spatial locality regularization and orientation awareness.</strong><br>
        <ul style="margin-top: 0.8rem; line-height: 1.6;">
            <li><strong>Load simulation files</strong> from numerical_solutions directory</li>
            <li><strong>Interpolate stress fields</strong> at arbitrary crystallographic orientations</li>
            <li><strong>Visualize stress components</strong> with publication-quality heatmaps and 3D plots</li>
            <li><strong>Compare source simulations</strong> with interpolated results</li>
            <li><strong>Export results</strong> in multiple formats for further analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader(SOLUTIONS_DIR)
    if 'transformer_interpolator' not in st.session_state:
        st.session_state.transformer_interpolator = TransformerSpatialInterpolator()
    if 'heatmap_visualizer' not in st.session_state:
        st.session_state.heatmap_visualizer = AdvancedHeatMapVisualizer()  # Updated class name
    if 'results_manager' not in st.session_state:
        st.session_state.results_manager = EnhancedResultsManager()  # Updated class name
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'selected_sources' not in st.session_state:
        st.session_state.selected_sources = []
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Data loading
        st.markdown("#### 📂 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Solutions", use_container_width=True):
                with st.spinner("Loading and processing solutions..."):
                    st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions")
                    # Auto-select all sources initially
                    st.session_state.selected_sources = list(range(len(st.session_state.solutions)))
                else:
                    st.warning("⚠️ No solutions found in directory")
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.session_state.solutions = []
                st.session_state.interpolation_result = None
                st.session_state.selected_sources = []
                st.success("✅ Cache cleared")
        
        # Debug button with enhanced functionality
        if st.button("🔍 Debug Feature Dimensions", use_container_width=True):
            if st.session_state.solutions:
                source_params = [sol['params'] for sol in st.session_state.solutions[:1]]
                shape = st.session_state.transformer_interpolator.debug_feature_dimensions(
                    source_params, 54.7
                )
                st.write(f"Feature dimensions: {shape}")
                st.write(f"Number of solutions: {len(st.session_state.solutions)}")
        
        # Show loaded solutions info with enhanced preview
        if st.session_state.solutions:
            with st.expander(f"📚 Loaded Solutions ({len(st.session_state.solutions)})", expanded=False):
                st.markdown("### Select source simulations for interpolation:")
                selected_indices = []
                
                for i, sol in enumerate(st.session_state.solutions):
                    params = sol.get('params', {})
                    theta_deg = np.degrees(params.get('theta', 0))
                    defect_type = params.get('defect_type', 'Unknown')
                    eps0 = params.get('eps0', 0.0)
                    
                    # Create checkbox for each solution
                    is_selected = st.checkbox(
                        f"Solution {i+1}: {defect_type} at θ={theta_deg:.1f}° (ε₀={eps0:.2f})",
                        value=i in st.session_state.selected_sources,
                        key=f"solution_{i}"
                    )
                    
                    if is_selected:
                        selected_indices.append(i)
                    
                    # Add preview button
                    if st.button(f"🖼️ Preview Solution {i+1}", key=f"preview_{i}", use_container_width=True):
                        with st.spinner(f"Generating preview for Solution {i+1}..."):
                            preview_fig = st.session_state.loader.generate_source_preview(i)
                        
                        if preview_fig:
                            st.pyplot(preview_fig, use_container_width=True)
                            plt.close(preview_fig)
                
                # Update selected sources
                st.session_state.selected_sources = selected_indices
                
                st.info(f"Selected {len(st.session_state.selected_sources)} source simulations for interpolation")
        
        # Target parameters
        st.markdown('<h2 class="section-header">🎯 Target Parameters</h2>', unsafe_allow_html=True)
        
        # Custom polar angle with visual indicator
        st.markdown("#### Orientation Angle")
        col_angle1, col_angle2 = st.columns([3, 1])
        with col_angle1:
            custom_theta = st.slider(
                "Polar Angle θ (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=54.7,
                step=0.1,
                help="Set custom polar angle for interpolation (default: 54.7° - habit plane for FCC materials)"
            )
        with col_angle2:
            st.metric("Angle", f"{custom_theta:.1f}°")
        
        # Visual angle indicator
        fig_angle, ax_angle = plt.subplots(figsize=(2, 2))
        circle = plt.Circle((0.5, 0.5), 0.4, fill=False, color='black', linewidth=2)
        ax_angle.add_patch(circle)
        
        theta_rad = np.radians(custom_theta)
        ax_angle.plot([0.5, 0.5 + 0.4*np.cos(theta_rad)], 
                     [0.5, 0.5 + 0.4*np.sin(theta_rad)], 
                     'r-', linewidth=3)
        
        _create_angle_indicator(ax_angle, custom_theta, radius=0.3, color='red')
        
        ax_angle.set_xlim(0, 1)
        ax_angle.set_ylim(0, 1)
        ax_angle.set_aspect('equal')
        ax_angle.axis('off')
        ax_angle.set_title("Orientation", fontsize=12, fontweight='bold')
        
        st.pyplot(fig_angle, use_container_width=True)
        plt.close(fig_angle)
        
        # Defect type selection with visual guide
        st.markdown("#### Defect Type")
        defect_type = st.selectbox(
            "Crystal Defect Type",
            ["ISF", "ESF", "Twin", "No Defect"],
            index=2,
            help="Select the defect type for interpolation:\n- ISF: Intrinsic Stacking Fault\n- ESF: Extrinsic Stacking Fault\n- Twin: Deformation Twin"
        )
        
        # Auto-set eigen strain based on defect type
        eigen_strains = {"ISF": 0.71, "ESF": 1.41, "Twin": 2.12, "No Defect": 0.0}
        
        col_def1, col_def2 = st.columns(2)
        with col_def1:
            eps0 = eigen_strains[defect_type]
            st.metric("Eigen Strain (ε₀)", f"{eps0:.2f}")
        with col_def2:
            kappa = st.slider(
                "Kappa Parameter",
                min_value=0.1,
                max_value=2.0,
                value=0.6,
                step=0.1,
                help="Material parameter controlling stress distribution"
            )
        
        # Shape selection
        st.markdown("#### Defect Geometry")
        shape = st.selectbox(
            "Defect Shape",
            ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle"],
            index=0,
            help="Geometry of the defect region"
        )
        
        # Transformer parameters
        st.markdown('<h2 class="section-header">🧠 Transformer Parameters</h2>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            spatial_sigma = st.slider(
                "Spatial Locality (σ)",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Controls how quickly similarity decreases with parameter distance. Lower values make interpolation more local."
            )
        with col_t2:
            attention_temp = st.slider(
                "Attention Temperature",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Temperature for attention scaling. Lower values make attention more focused, higher values more uniform."
            )
        
        # Visualization parameters
        st.markdown('<h2 class="section-header">🎨 Visualization</h2>', unsafe_allow_html=True)
        
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            colormap_category = st.selectbox(
                "Colormap Category",
                list(COLORMAP_OPTIONS.keys()),
                index=0
            )
        with col_v2:
            colormap_name = st.selectbox(
                "Colormap Selection",
                COLORMAP_OPTIONS[colormap_category],
                index=0
            )
        
        # Colormap preview
        if st.button("🖼️ Preview Colormap", use_container_width=True):
            fig_preview = st.session_state.heatmap_visualizer.get_colormap_preview(colormap_name)
            st.pyplot(fig_preview, use_container_width=True)
            plt.close(fig_preview)
        
        visualization_type = st.selectbox(
            "Visualization Type",
            ["Static 2D Heatmap", "Interactive 2D Heatmap", "Interactive 3D Surface", 
             "Stress Component Comparison", "Source vs Target Dashboard"],
            index=1
        )
        
        st.markdown("#### Plot Template (Interactive)")
        plotly_template = st.selectbox(
            "Plotly Template",
            ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
            index=0
        )
        
        # Interpolation button
        st.markdown("---")
        if st.button("🚀 Perform Advanced Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.error("❌ Please load solutions first!")
            elif not st.session_state.selected_sources:
                st.error("❌ Please select at least one source simulation!")
            else:
                # Update transformer parameters
                st.session_state.transformer_interpolator.spatial_sigma = spatial_sigma
                st.session_state.transformer_interpolator.temperature = attention_temp
                
                # Prepare target parameters
                target_params = {
                    'defect_type': defect_type,
                    'eps0': eps0,
                    'kappa': kappa,
                    'theta': np.radians(custom_theta),
                    'shape': shape
                }
                
                # Get selected sources
                selected_sources = [st.session_state.solutions[i] for i in st.session_state.selected_sources]
                
                # Perform interpolation
                with st.spinner("🧠 Performing transformer-based spatial interpolation with orientation awareness..."):
                    try:
                        result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                            selected_sources,
                            custom_theta,
                            target_params
                        )
                        
                        if result:
                            st.session_state.interpolation_result = result
                            st.success(f"✅ Successfully interpolated stress fields at θ={custom_theta:.1f}° using {result['num_sources']} sources")
                        else:
                            st.error("❌ Failed to interpolate stress fields. Check data compatibility.")
                    except Exception as e:
                        st.error(f"❌ Error during interpolation: {str(e)}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
    
    # Main content
    if not st.session_state.solutions:
        st.warning("📁 Please load solutions first using the button in the sidebar.")
    
    # Directory information
    with st.expander("📁 Directory Information", expanded=True):
        st.info(f"**Solutions Directory:** `{SOLUTIONS_DIR}`")
        st.write("""
        **Expected file formats:** .pkl, .pickle, .pt, .pth
        
        **Expected data structure:**
        - Each file should contain a dictionary with:
        - `params`: Dictionary of simulation parameters including:
            - `theta`: Orientation angle in radians
            - `defect_type`: Type of crystal defect (ISF, ESF, Twin, No Defect)
            - `eps0`: Eigenstrain magnitude
            - `kappa`: Material parameter
            - `shape`: Geometry of defect region
        - `history`: List of simulation frames
        - Each frame should contain `stresses` dictionary with stress fields:
            - `von_mises`: Von Mises stress field
            - `sigma_hydro`: Hydrostatic stress field
            - `sigma_mag`: Stress magnitude field
            - Or individual components: `sigma_xx`, `sigma_yy`, `sigma_zz`, `tau_xy`, etc.
        """)
    
    # Quick guide with enhanced formatting
    st.markdown("""
    ## 📋 Quick Start Guide
    1. **Prepare Data**: Place your simulation files in the `numerical_solutions` directory
    2. **Load Solutions**: Click the "🔄 Load Solutions" button in the sidebar
    3. **Select Sources**: Choose which simulations to use as sources for interpolation
    4. **Set Target Parameters**: Configure orientation angle and defect properties
    5. **Perform Interpolation**: Click "🚀 Perform Advanced Interpolation"
    6. **Visualize Results**: Explore different visualization options
    
    ## 🔬 Key Features
    ### Transformer Architecture
    - Multi-head attention across source simulations with orientation awareness
    - Spatial locality regularization for physically meaningful interpolation
    - Enhanced parameter encoding with angular features
    
    ### Stress Components
    - **Von Mises Stress (σ_vm)**: Equivalent tensile stress for yielding criteria
    - **Hydrostatic Stress (σ_h)**: Mean normal stress (trace/3), drives volumetric changes
    - **Stress Magnitude (σ_mag)**: Overall stress intensity combining deviatoric and hydrostatic parts
    
    ### Visualization Options
    - **Static 2D Heatmaps**: Publication-quality figures with orientation indicators
    - **Interactive 2D Heatmaps**: Zoom, pan, and hover capabilities with Plotly
    - **Interactive 3D Surfaces**: Rotate and explore stress fields in three dimensions
    - **Component Comparison**: Side-by-side visualization of all stress components
    - **Source vs Target Dashboard**: Comprehensive comparison showing source contributions
    
    ### Scientific Features
    - Orientation-aware interpolation respecting crystallographic symmetries
    - Weight analysis showing contribution of each source simulation
    - Statistical comparison between sources and interpolated result
    - Parameter similarity metrics for physical interpretation
    """)
    
    if st.session_state.solutions:
        # Create tabs with enhanced styling
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Results Overview",
            "🎨 Stress Visualization", 
            "⚖️ Source Contributions",
            "📈 Detailed Analysis",
            "💾 Export Results"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">📊 Interpolation Results Overview</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Display key metrics with enhanced cards
                st.markdown("### Key Stress Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    vm_stats = result['statistics']['von_mises']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">Von Mises Stress</div>
                        <div style="font-size: 1.8rem; margin: 0.5rem 0;">{vm_stats['max']:.3f} GPa</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Mean: {vm_stats['mean']:.3f} GPa</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Std: {vm_stats['std']:.3f} GPa</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    hydro_stats = result['statistics']['sigma_hydro']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">Hydrostatic Stress</div>
                        <div style="font-size: 1.5rem; margin: 0.5rem 0;">{hydro_stats['max_tension']:.3f}</div>
                        <div style="font-size: 1.5rem; margin: 0.2rem 0;">{hydro_stats['max_compression']:.3f} GPa</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Tension/Compression</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    mag_stats = result['statistics']['sigma_mag']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">Stress Magnitude</div>
                        <div style="font-size: 1.8rem; margin: 0.5rem 0;">{mag_stats['max']:.3f} GPa</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Mean: {mag_stats['mean']:.3f} GPa</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Min: {mag_stats['min']:.3f} GPa</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">Interpolation Parameters</div>
                        <div style="font-size: 1.6rem; margin: 0.5rem 0;">θ = {result['target_angle']:.1f}°</div>
                        <div style="font-size: 1.2rem; margin: 0.3rem 0;">{result['target_params']['defect_type']}</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Sources: {result.get('num_sources', 0)}</div>
                        <div style="font-size: 1.0rem; opacity: 0.9;">Grid: {result['shape'][0]}×{result['shape'][1]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display parameters in expandable section
                with st.expander("🔬 Detailed Interpolation Parameters", expanded=True):
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        st.markdown("#### Target Parameters")
                        target_params = result['target_params']
                        params_html = ""
                        for key, value in target_params.items():
                            if key == 'theta':
                                params_html += f"<div style='margin: 0.5rem 0;'><strong>{key}:</strong> {np.degrees(value):.2f}°</div>"
                            else:
                                params_html += f"<div style='margin: 0.5rem 0;'><strong>{key}:</strong> {value}</div>"
                        st.markdown(params_html, unsafe_allow_html=True)
                    
                    with col_p2:
                        st.markdown("#### Interpolation Settings")
                        settings_html = f"""
                        <div style='margin: 0.5rem 0;'><strong>Spatial Locality (σ):</strong> {spatial_sigma}</div>
                        <div style='margin: 0.5rem 0;'><strong>Attention Temperature:</strong> {attention_temp}</div>
                        <div style='margin: 0.5rem 0;'><strong>Number of Sources:</strong> {result.get('num_sources', len(result['weights']['combined']))}</div>
                        <div style='margin: 0.5rem 0;'><strong>Feature Dimension:</strong> 18</div>
                        """
                        st.markdown(settings_html, unsafe_allow_html=True)
                
                # Orientation diagram with enhanced styling
                st.markdown("### Crystallographic Orientation")
                fig_orient, ax_orient = plt.subplots(figsize=(8, 6))
                
                # Create stereographic projection style plot
                circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
                ax_orient.add_patch(circle)
                
                # Plot source orientations
                source_angles = result['orientation']['source_angles']
                source_weights = result['weights']['combined']
                
                for angle, weight in zip(source_angles, source_weights):
                    rad = np.radians(angle)
                    x = 0.9 * np.cos(rad)
                    y = 0.9 * np.sin(rad)
                    
                    # Size based on weight
                    size = 50 + weight * 300
                    alpha = 0.5 + weight * 0.5
                    
                    ax_orient.scatter(x, y, s=size, alpha=alpha, color='blue', 
                                    edgecolors='navy', linewidth=1,
                                    label=f'Source (θ={angle:.1f}°)')
                
                # Plot target orientation
                target_rad = np.radians(result['target_angle'])
                target_x = np.cos(target_rad)
                target_y = np.sin(target_rad)
                
                ax_orient.scatter(target_x, target_y, s=200, color='red', edgecolors='darkred', 
                                linewidth=2, marker='*', zorder=10, label='Target Orientation')
                
                # Add angle indicators
                for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
                    rad = np.radians(angle)
                    x = 1.1 * np.cos(rad)
                    y = 1.1 * np.sin(rad)
                    ax_orient.text(x, y, f'{angle}°', ha='center', va='center', fontsize=10, fontweight='bold')
                
                ax_orient.set_xlim(-1.3, 1.3)
                ax_orient.set_ylim(-1.3, 1.3)
                ax_orient.set_aspect('equal')
                ax_orient.axis('off')
                ax_orient.set_title('Source and Target Orientations in Stereographic Projection', 
                                  fontsize=16, fontweight='bold', pad=15)
                
                ax_orient.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), 
                               ncol=3, fontsize=12, frameon=True, framealpha=0.9)
                
                st.pyplot(fig_orient, use_container_width=True)
                plt.close(fig_orient)
                
                # Quick preview with enhanced styling
                st.markdown("### Stress Field Preview")
                
                # Create a quick preview figure with orientation indicators
                fig_preview, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
                
                components = ['von_mises', 'sigma_hydro', 'sigma_mag']
                titles = ['Von Mises Stress', 'Hydrostatic Stress', 'Stress Magnitude']
                cmaps = [colormap_name, 'RdBu_r', colormap_name]
                
                for idx, (comp, title, cmap) in enumerate(zip(components, titles, cmaps)):
                    ax = axes[idx]
                    
                    # Get appropriate vmin/vmax for hydrostatic
                    vmin, vmax = None, None
                    if comp == 'sigma_hydro':
                        abs_max = np.max(np.abs(result['fields'][comp]))
                        vmin, vmax = -abs_max, abs_max
                    
                    im = ax.imshow(result['fields'][comp], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = fig_preview.colorbar(im, cax=cax)
                    cbar.set_label("Stress (GPa)", fontsize=12)
                    cbar.ax.tick_params(labelsize=10)
                    
                    # Add orientation indicator
                    self._add_orientation_indicator(ax, result['target_angle'], result['fields'][comp].shape)
                    
                    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                    ax.set_xlabel('X Position (nm)', fontsize=12)
                    ax.set_ylabel('Y Position (nm)', fontsize=12)
                    ax.grid(True, alpha=0.2)
                
                plt.suptitle(f"Stress Fields at θ={result['target_angle']:.1f}° - {result['target_params']['defect_type']}", 
                           fontsize=18, fontweight='bold', y=0.95)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                
                st.pyplot(fig_preview, use_container_width=True)
                plt.close(fig_preview)
                
            else:
                st.info("💡 Configure parameters and click '🚀 Perform Advanced Interpolation' to generate results")
        
        with tab2:
            st.markdown('<h2 class="section-header">🎨 Stress Field Visualization</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Component selection with icons
                st.markdown("#### Stress Component Selection")
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    show_von_mises = st.checkbox("🟦 Von Mises Stress", value=True, key="show_vm")
                with col_comp2:
                    show_hydrostatic = st.checkbox("🟥 Hydrostatic Stress", value=True, key="show_hydro")
                with col_comp3:
                    show_magnitude = st.checkbox("🟩 Stress Magnitude", value=True, key="show_mag")
                
                # Create tabs for different visualization types
                viz_tabs = st.tabs(["📈 2D Visualization", "🌐 3D Visualization", "📊 Component Comparison"])
                
                with viz_tabs[0]:
                    # 2D Visualization
                    if visualization_type in ["Static 2D Heatmap", "Interactive 2D Heatmap"]:
                        st.markdown("### 2D Stress Field Visualization")
                        
                        components_to_show = []
                        if show_von_mises: components_to_show.append(('von_mises', 'Von Mises Stress', colormap_name))
                        if show_hydrostatic: components_to_show.append(('sigma_hydro', 'Hydrostatic Stress', 'RdBu_r'))
                        if show_magnitude: components_to_show.append(('sigma_mag', 'Stress Magnitude', colormap_name))
                        
                        if not components_to_show:
                            st.warning("⚠️ Please select at least one stress component to visualize")
                        else:
                            for comp_name, title, cmap in components_to_show:
                                st.markdown(f"#### {title} at θ={result['target_angle']:.1f}°")
                                
                                if visualization_type == "Static 2D Heatmap":
                                    fig = st.session_state.heatmap_visualizer.create_stress_heatmap(
                                        result['fields'][comp_name],
                                        title=title,
                                        cmap_name=cmap,
                                        colorbar_label=f"{title} (GPa)",
                                        theta=result['target_angle'],
                                        defect_type=result['target_params']['defect_type'],
                                        figsize=(10, 8)
                                    )
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)
                                
                                elif visualization_type == "Interactive 2D Heatmap":
                                    fig = st.session_state.heatmap_visualizer.create_interactive_heatmap(
                                        result['fields'][comp_name],
                                        title=title,
                                        cmap_name=cmap,
                                        theta=result['target_angle'],
                                        defect_type=result['target_params']['defect_type'],
                                        template=plotly_template
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[1]:
                    # 3D Visualization
                    st.markdown("### 3D Stress Field Visualization")
                    
                    components_to_show = []
                    if show_von_mises: components_to_show.append(('von_mises', 'Von Mises Stress', colormap_name))
                    if show_hydrostatic: components_to_show.append(('sigma_hydro', 'Hydrostatic Stress', 'RdBu_r'))
                    if show_magnitude: components_to_show.append(('sigma_mag', 'Stress Magnitude', colormap_name))
                    
                    if not components_to_show:
                        st.warning("⚠️ Please select at least one stress component to visualize")
                    else:
                        for comp_name, title, cmap in components_to_show:
                            st.markdown(f"#### {title} at θ={result['target_angle']:.1f}°")
                            
                            fig = st.session_state.heatmap_visualizer.create_interactive_3d_surface(
                                result['fields'][comp_name],
                                title=title,
                                cmap_name=cmap,
                                theta=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                template=plotly_template
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                with viz_tabs[2]:
                    # Component Comparison
                    st.markdown("### Stress Component Comparison")
                    
                    comparison_fields = {}
                    if show_von_mises: comparison_fields['von_mises'] = result['fields']['von_mises']
                    if show_hydrostatic: comparison_fields['sigma_hydro'] = result['fields']['sigma_hydro']
                    if show_magnitude: comparison_fields['sigma_mag'] = result['fields']['sigma_mag']
                    
                    if not comparison_fields:
                        st.warning("⚠️ Please select at least one stress component to compare")
                    else:
                        if visualization_type == "Interactive 2D Heatmap":
                            fig = st.session_state.heatmap_visualizer.create_interactive_comparison(
                                comparison_fields,
                                result['target_angle'],
                                result['target_params']['defect_type'],
                                cmap_name=colormap_name,
                                template=plotly_template
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = st.session_state.heatmap_visualizer.create_comparison_heatmaps(
                                comparison_fields,
                                cmap_name=colormap_name,
                                theta=result['target_angle'],
                                defect_type=result['target_params']['defect_type'],
                                figsize=(18, 6)
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                
                # Statistics section
                with st.expander("📈 Detailed Statistics", expanded=False):
                    if components_to_show:
                        stats_tabs = st.tabs([title for _, title, _ in components_to_show])
                        
                        for tab_idx, (comp_name, title, cmap) in enumerate(components_to_show):
                            with stats_tabs[tab_idx]:
                                st.markdown(f"### {title} Statistics")
                                
                                # Get statistics
                                stats = result['statistics'][comp_name]
                                
                                # Display metrics
                                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                                with col_s1:
                                    st.metric("Maximum", f"{stats.get('max', stats.get('max_tension', 0)):.3f} GPa")
                                with col_s2:
                                    st.metric("Minimum", f"{stats.get('min', stats.get('max_compression', 0)):.3f} GPa")
                                with col_s3:
                                    st.metric("Mean", f"{stats['mean']:.3f} GPa")
                                with col_s4:
                                    st.metric("Std Dev", f"{stats['std']:.3f} GPa")
                                
                                # Histogram
                                fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
                                ax_hist.hist(result['fields'][comp_name].flatten(), bins=50, alpha=0.7, color='blue')
                                ax_hist.set_xlabel(f'{title} (GPa)', fontsize=14)
                                ax_hist.set_ylabel('Frequency', fontsize=14)
                                ax_hist.set_title(f'Distribution of {title}', fontsize=16, fontweight='bold')
                                ax_hist.grid(True, alpha=0.3)
                                
                                st.pyplot(fig_hist, use_container_width=True)
                                plt.close(fig_hist)
                                
                                # Spatial statistics
                                st.markdown("#### Spatial Statistics")
                                field = result['fields'][comp_name]
                                spatial_stats = {
                                    'Center Region (25%)': field[int(0.375*field.shape[0]):int(0.625*field.shape[0]), 
                                                              int(0.375*field.shape[1]):int(0.625*field.shape[1])],
                                    'Edge Region': np.concatenate([
                                        field[:int(0.25*field.shape[0]), :],
                                        field[int(0.75*field.shape[0]):, :],
                                        field[int(0.25*field.shape[0]):int(0.75*field.shape[0]), :int(0.25*field.shape[1])],
                                        field[int(0.25*field.shape[0]):int(0.75*field.shape[0]), int(0.75*field.shape[1]):]
                                    ]),
                                    'Full Field': field
                                }
                                
                                spatial_df = pd.DataFrame({
                                    'Region': list(spatial_stats.keys()),
                                    'Mean': [np.mean(region) for region in spatial_stats.values()],
                                    'Max': [np.max(region) for region in spatial_stats.values()],
                                    'Std': [np.std(region) for region in spatial_stats.values()]
                                })
                                
                                st.dataframe(spatial_df.style.format({
                                    'Mean': '{:.3f}',
                                    'Max': '{:.3f}',
                                    'Std': '{:.3f}'
                                }), use_container_width=True)
            
            else:
                st.info("💡 No interpolation results available. Please perform interpolation first.")
        
        with tab3:
            st.markdown('<h2 class="section-header">⚖️ Source Contributions & Weight Analysis</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Source selection summary
                st.markdown("### Selected Source Simulations")
                source_info = []
                for i, src_idx in enumerate(st.session_state.selected_sources):
                    sol = st.session_state.solutions[src_idx]
                    params = sol['params']
                    theta_deg = np.degrees(params.get('theta', 0))
                    defect_type = params.get('defect_type', 'Unknown')
                    source_info.append({
                        'Source Index': i+1,
                        'Original Index': src_idx+1,
                        'Defect Type': defect_type,
                        'Orientation (θ)': f"{theta_deg:.1f}°",
                        'Eigenstrain (ε₀)': params.get('eps0', 0.0),
                        'Weight': result['weights']['combined'][i] if i < len(result['weights']['combined']) else 0.0
                    })
                
                source_df = pd.DataFrame(source_info)
                st.dataframe(source_df.style.format({
                    'Eigenstrain (ε₀)': '{:.3f}',
                    'Weight': '{:.3f}'
                }), use_container_width=True)
                
                # Weights visualization with enhanced styling
                st.markdown("### Weight Distribution Analysis")
                
                weights = result['weights']
                combined_weights = np.array(weights['combined'])
                
                # Create weight visualization tabs
                weight_tabs = st.tabs(["📊 Weight Comparison", "🎯 Orientation Analysis", "🔬 Parameter Similarity"])
                
                with weight_tabs[0]:
                    # Weight comparison plot
                    fig_weights, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    x = range(len(weights['combined']))
                    width = 0.25
                    
                    # Bar plot comparison
                    ax1.bar([i - width for i in x], weights['transformer'], width, 
                           label='Transformer Attention', alpha=0.8, color='skyblue', edgecolor='navy')
                    ax1.bar(x, weights['positional'], width, 
                           label='Spatial Locality', alpha=0.8, color='salmon', edgecolor='darkred')
                    ax1.bar([i + width for i in x], weights['combined'], width, 
                           label='Combined Weight', alpha=0.9, color='gold', edgecolor='goldenrod')
                    
                    ax1.set_xlabel('Source Index', fontsize=14)
                    ax1.set_ylabel('Weight', fontsize=14)
                    ax1.set_title('Weight Component Comparison', fontsize=16, fontweight='bold')
                    ax1.legend(fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    
                    # Pie chart of combined weights
                    labels = [f'Source {i+1}\n(θ={result["orientation"]["source_angles"][i]:.1f}°)' 
                             for i in range(len(combined_weights))]
                    ax2.pie(combined_weights, labels=labels, autopct='%1.1f%%', startangle=90,
                           colors=plt.cm.Paired(np.linspace(0, 1, len(combined_weights))))
                    ax2.set_title('Combined Weight Distribution', fontsize=16, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig_weights, use_container_width=True)
                    plt.close(fig_weights)
                    
                    # Weight statistics
                    st.markdown("#### Weight Statistics")
                    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                    
                    with col_w1:
                        entropy_trans = _calculate_entropy(weights['transformer'])
                        st.metric("Transformer Entropy", f"{entropy_trans:.3f}")
                    with col_w2:
                        entropy_pos = _calculate_entropy(weights['positional'])
                        st.metric("Positional Entropy", f"{entropy_pos:.3f}")
                    with col_w3:
                        entropy_comb = _calculate_entropy(weights['combined'])
                        st.metric("Combined Entropy", f"{entropy_comb:.3f}")
                    with col_w4:
                        max_weight_idx = np.argmax(combined_weights)
                        st.metric("Dominant Source", f"{max_weight_idx+1}")
                    
                    # Top contributors table
                    st.markdown("### Top Contributing Sources")
                    top_indices = np.argsort(combined_weights)[-5:][::-1]
                    
                    top_contributors = []
                    for idx in top_indices:
                        if idx < len(weights['combined']):
                            contributor = {
                                'Rank': len(top_contributors) + 1,
                                'Source Index': idx + 1,
                                'Combined Weight': weights['combined'][idx],
                                'Transformer Weight': weights['transformer'][idx],
                                'Positional Weight': weights['positional'][idx],
                                'Orientation (θ)': f"{result['orientation']['source_angles'][idx]:.1f}°",
                                'Angular Distance': f"{result['orientation']['angular_distances'][idx]:.1f}°"
                            }
                            top_contributors.append(contributor)
                    
                    contributors_df = pd.DataFrame(top_contributors)
                    st.dataframe(contributors_df.style.format({
                        'Combined Weight': '{:.3f}',
                        'Transformer Weight': '{:.3f}',
                        'Positional Weight': '{:.3f}'
                    }), use_container_width=True)
                
                with weight_tabs[1]:
                    # Orientation analysis
                    st.markdown("### Orientation-Based Weight Analysis")
                    
                    # Create polar plot of weights by orientation
                    fig_polar, ax_polar = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
                    
                    source_angles_rad = np.radians(result['orientation']['source_angles'])
                    combined_weights = np.array(weights['combined'])
                    
                    # Plot bars
                    bars = ax_polar.bar(source_angles_rad, combined_weights, width=0.2, alpha=0.7, color='teal')
                    
                    # Highlight target orientation
                    target_angle_rad = np.radians(result['target_angle'])
                    ax_polar.axvline(target_angle_rad, color='red', linestyle='--', linewidth=2, 
                                   label=f'Target θ={result["target_angle"]:.1f}°')
                    
                    # Add labels
                    ax_polar.set_title('Source Contribution by Orientation', fontsize=16, fontweight='bold', pad=20)
                    ax_polar.legend(loc='upper right', fontsize=12)
                    
                    # Add weight values to bars
                    for i, (bar, angle) in enumerate(zip(bars, result['orientation']['source_angles'])):
                        height = bar.get_height()
                        if height > 0.05:  # Only label significant contributions
                            ax_polar.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                        f'{height:.2f}', 
                                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    
                    st.pyplot(fig_polar, use_container_width=True)
                    plt.close(fig_polar)
                    
                    # Angular distance vs weight plot
                    fig_angle_weight, ax_aw = plt.subplots(figsize=(10, 6))
                    
                    angular_distances = result['orientation']['angular_distances']
                    orientation_weights = result['orientation']['orientation_weights']
                    
                    scatter = ax_aw.scatter(angular_distances, combined_weights, 
                                          s=[w*500 for w in combined_weights], 
                                          c=orientation_weights, cmap='viridis',
                                          alpha=0.8, edgecolors='black')
                    
                    ax_aw.set_xlabel('Angular Distance to Target (°)', fontsize=14)
                    ax_aw.set_ylabel('Combined Weight', fontsize=14)
                    ax_aw.set_title('Weight vs. Angular Distance', fontsize=16, fontweight='bold')
                    ax_aw.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax_aw)
                    cbar.set_label('Orientation Similarity', fontsize=12)
                    
                    st.pyplot(fig_angle_weight, use_container_width=True)
                    plt.close(fig_angle_weight)
                
                with weight_tabs[2]:
                    # Parameter similarity analysis
                    st.markdown("### Parameter Similarity Analysis")
                    
                    if 'details' in weights and weights['details']:
                        # Create similarity heatmap
                        params = ['eps0', 'kappa', 'theta', 'defect_type']
                        similarities = []
                        source_labels = [f'Source {i+1}' for i in range(len(weights['combined']))]
                        
                        for detail in weights['details']:
                            sim_row = []
                            factors = detail['factors']
                            for param in params:
                                # Convert distance to similarity (1 - normalized distance)
                                dist = factors.get(param, 0.0)
                                sim = 1.0 - dist
                                sim_row.append(sim)
                            similarities.append(sim_row)
                        
                        # Create heatmap
                        fig_sim, ax_sim = plt.subplots(figsize=(12, 8))
                        im = ax_sim.imshow(similarities, cmap='YlGnBu', aspect='auto')
                        
                        # Add colorbar
                        cbar = fig_sim.colorbar(im, ax=ax_sim)
                        cbar.set_label('Parameter Similarity', fontsize=14)
                        
                        # Set labels
                        ax_sim.set_xticks(np.arange(len(params)))
                        ax_sim.set_xticklabels(params, fontsize=12, fontweight='bold')
                        ax_sim.set_yticks(np.arange(len(source_labels)))
                        ax_sim.set_yticklabels(source_labels, fontsize=12)
                        
                        # Rotate x-axis labels
                        plt.setp(ax_sim.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        
                        # Add text annotations
                        for i in range(len(similarities)):
                            for j in range(len(params)):
                                ax_sim.text(j, i, f'{similarities[i][j]:.2f}',
                                          ha='center', va='center', 
                                          color='black' if similarities[i][j] < 0.7 else 'white',
                                          fontweight='bold', fontsize=12)
                        
                        ax_sim.set_title('Parameter Similarity to Target', fontsize=16, fontweight='bold', pad=15)
                        
                        st.pyplot(fig_sim, use_container_width=True)
                        plt.close(fig_sim)
                        
                        # Parameter importance analysis
                        st.markdown("#### Parameter Importance")
                        
                        # Calculate average similarity for each parameter
                        avg_similarities = np.mean(similarities, axis=0)
                        param_importance = pd.DataFrame({
                            'Parameter': params,
                            'Average Similarity': avg_similarities,
                            'Importance Score': 1 - avg_similarities  # Higher score means more discriminative
                        })
                        
                        st.dataframe(param_importance.style.format({
                            'Average Similarity': '{:.3f}',
                            'Importance Score': '{:.3f}'
                        }).background_gradient(subset=['Importance Score'], cmap='YlOrRd'), use_container_width=True)
            
            else:
                st.info("💡 No interpolation results available. Please perform interpolation first.")
        
        with tab4:
            st.markdown('<h2 class="section-header">📈 Comprehensive Analysis Dashboard</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Source comparison dashboard
                st.markdown("### Source Simulations vs. Interpolated Result")
                
                dashboard_tabs = st.tabs([
                    "📊 Comprehensive Dashboard", 
                    "🔍 Source Field Comparison",
                    "📐 Line Profile Analysis",
                    "🔍 Local Stress Concentration"
                ])
                
                with dashboard_tabs[0]:
                    # Create comprehensive dashboard
                    fig_dashboard = st.session_state.heatmap_visualizer.create_source_comparison_dashboard(
                        [st.session_state.solutions[i] for i in st.session_state.selected_sources],  # Pass selected source objects
                        result,
                        cmap_name=colormap_name,
                        figsize=(22, 16)
                    )
                    
                    if fig_dashboard:
                        st.pyplot(fig_dashboard, use_container_width=True)
                        plt.close(fig_dashboard)
                
                with dashboard_tabs[1]:
                    # Source field comparison
                    st.markdown("#### Compare Source Stress Fields")
                    
                    # Source selection
                    selected_source_idx = st.selectbox(
                        "Select Source to Compare",
                        range(len(st.session_state.selected_sources)),
                        format_func=lambda x: f"Source {x+1} (θ={np.degrees(st.session_state.solutions[st.session_state.selected_sources[x]]['params'].get('theta', 0)):.1f}°)"
                    )
                    
                    if selected_source_idx < len(st.session_state.selected_sources):
                        source_idx = st.session_state.selected_sources[selected_source_idx]
                        source_sol = st.session_state.solutions[source_idx]
                        
                        if 'history' in source_sol and source_sol['history']:
                            last_frame = source_sol['history'][-1]
                            if 'stresses' in last_frame:
                                source_stresses = last_frame['stresses']
                                
                                # Component selection
                                comp_to_compare = st.selectbox(
                                    "Stress Component to Compare",
                                    ['von_mises', 'sigma_hydro', 'sigma_mag'],
                                    format_func=lambda x: {
                                        'von_mises': 'Von Mises Stress',
                                        'sigma_hydro': 'Hydrostatic Stress', 
                                        'sigma_mag': 'Stress Magnitude'
                                    }[x]
                                )
                                
                                if comp_to_compare in source_stresses and comp_to_compare in result['fields']:
                                    # Get fields
                                    source_field = source_stresses[comp_to_compare]
                                    target_field = result['fields'][comp_to_compare]
                                    
                                    # Resize source field if needed
                                    if source_field.shape != target_field.shape:
                                        factors = [t/s for t, s in zip(target_field.shape, source_field.shape)]
                                        source_field = zoom(source_field, factors, order=1)
                                    
                                    # Create comparison figure
                                    fig_comp, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                                    
                                    # Source field
                                    im1 = ax1.imshow(source_field, cmap=colormap_name, aspect='equal')
                                    plt.colorbar(im1, ax=ax1, label='Stress (GPa)')
                                    source_theta = np.degrees(source_sol['params'].get('theta', 0))
                                    ax1.set_title(f'Source Field\nθ={source_theta:.1f}°', fontsize=14, fontweight='bold')
                                    ax1.set_xlabel('X Position (nm)')
                                    ax1.set_ylabel('Y Position (nm)')
                                    
                                    # Target field
                                    im2 = ax2.imshow(target_field, cmap=colormap_name, aspect='equal')
                                    plt.colorbar(im2, ax=ax2, label='Stress (GPa)')
                                    ax2.set_title(f'Interpolated Field\nθ={result["target_angle"]:.1f}°', fontsize=14, fontweight='bold')
                                    ax2.set_xlabel('X Position (nm)')
                                    
                                    # Difference field
                                    diff_field = target_field - source_field
                                    abs_max = np.max(np.abs(diff_field))
                                    im3 = ax3.imshow(diff_field, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, aspect='equal')
                                    plt.colorbar(im3, ax=ax3, label='Stress Difference (GPa)')
                                    ax3.set_title('Difference Field', fontsize=14, fontweight='bold')
                                    ax3.set_xlabel('X Position (nm)')
                                    
                                    plt.suptitle(f'{comp_to_compare.replace("_", " ").title()} Comparison', 
                                               fontsize=18, fontweight='bold')
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig_comp, use_container_width=True)
                                    plt.close(fig_comp)
                
                with dashboard_tabs[2]:
                    # Line profile analysis
                    st.markdown("#### Line Profile Analysis")
                    
                    # Field selection
                    profile_component = st.selectbox(
                        "Select Stress Component for Line Profiles",
                        ['von_mises', 'sigma_hydro', 'sigma_mag'],
                        format_func=lambda x: {
                            'von_mises': 'Von Mises Stress',
                            'sigma_hydro': 'Hydrostatic Stress', 
                            'sigma_mag': 'Stress Magnitude'
                        }[x],
                        key="profile_component"
                    )
                    
                    # Line position selection
                    line_position = st.slider(
                        "Line Position (Y-coordinate)",
                        0, result['shape'][0]-1,
                        result['shape'][0]//2,
                        key="line_position"
                    )
                    
                    # Create figure
                    fig_profile, ax_profile = plt.subplots(figsize=(12, 8))
                    
                    # Get interpolated field profile
                    target_field = result['fields'][profile_component]
                    target_profile = target_field[line_position, :]
                    
                    # Plot interpolated profile
                    ax_profile.plot(target_profile, 'r-', linewidth=3, label=f'Interpolated θ={result["target_angle"]:.1f}°')
                    
                    # Plot source profiles
                    for i, src_idx in enumerate(st.session_state.selected_sources):
                        source = st.session_state.solutions[src_idx]
                        if 'history' in source and source['history']:
                            last_frame = source['history'][-1]
                            if 'stresses' in last_frame and profile_component in last_frame['stresses']:
                                src_field = last_frame['stresses'][profile_component]
                                # Resize if needed
                                if src_field.shape != target_field.shape:
                                    factors = [t/s for t, s in zip(target_field.shape, src_field.shape)]
                                    src_field = zoom(src_field, factors, order=1)
                                src_profile = src_field[line_position, :]
                                src_theta = np.degrees(source['params'].get('theta', 0))
                                weight = result['weights']['combined'][i] if i < len(result['weights']['combined']) else 0
                                alpha = 0.4 + 0.6 * weight
                                ax_profile.plot(src_profile, linestyle='--', alpha=alpha,
                                              label=f'Source {i+1} θ={src_theta:.1f}° (w={weight:.2f})')
                    
                    ax_profile.set_xlabel('X Position (nm)', fontsize=14)
                    ax_profile.set_ylabel('Stress (GPa)', fontsize=14)
                    ax_profile.set_title(f'Line Profile at Y={line_position} - {profile_component.replace("_", " ").title()}', 
                                       fontsize=16, fontweight='bold')
                    ax_profile.legend(fontsize=12)
                    ax_profile.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_profile, use_container_width=True)
                    plt.close(fig_profile)
                
                with dashboard_tabs[3]:
                    # Local stress concentration analysis
                    st.markdown("#### Local Stress Concentration Analysis")
                    
                    # Region selection
                    st.markdown("##### Select Region of Interest")
                    col_roi1, col_roi2 = st.columns(2)
                    
                    with col_roi1:
                        roi_x_start = st.slider("X Start", 0, result['shape'][1]-1, max(0, result['shape'][1]//2 - 10))
                        roi_x_end = st.slider("X End", 0, result['shape'][1]-1, min(result['shape'][1]-1, result['shape'][1]//2 + 10))
                    
                    with col_roi2:
                        roi_y_start = st.slider("Y Start", 0, result['shape'][0]-1, max(0, result['shape'][0]//2 - 10))
                        roi_y_end = st.slider("Y End", 0, result['shape'][0]-1, min(result['shape'][0]-1, result['shape'][0]//2 + 10))
                    
                    # Component selection
                    roi_component = st.selectbox(
                        "Stress Component for ROI Analysis",
                        ['von_mises', 'sigma_hydro', 'sigma_mag'],
                        format_func=lambda x: {
                            'von_mises': 'Von Mises Stress',
                            'sigma_hydro': 'Hydrostatic Stress', 
                            'sigma_mag': 'Stress Magnitude'
                        }[x],
                        key="roi_component"
                    )
                    
                    # Extract ROI data
                    field = result['fields'][roi_component]
                    roi = field[roi_y_start:roi_y_end+1, roi_x_start:roi_x_end+1]
                    
                    # Create ROI visualization
                    fig_roi, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                    
                    # Full field with ROI box
                    im1 = ax1.imshow(field, cmap=colormap_name, aspect='equal')
                    rect = plt.Rectangle((roi_x_start, roi_y_start), 
                                       roi_x_end-roi_x_start+1, 
                                       roi_y_end-roi_y_start+1,
                                       fill=False, edgecolor='red', linewidth=2, linestyle='--')
                    ax1.add_patch(rect)
                    plt.colorbar(im1, ax=ax1, label='Stress (GPa)')
                    ax1.set_title('Full Field with ROI', fontsize=16, fontweight='bold')
                    ax1.set_xlabel('X Position (nm)')
                    ax1.set_ylabel('Y Position (nm)')
                    
                    # ROI close-up
                    im2 = ax2.imshow(roi, cmap=colormap_name, aspect='equal')
                    plt.colorbar(im2, ax=ax2, label='Stress (GPa)')
                    ax2.set_title(f'ROI ({roi_y_start}:{roi_y_end}, {roi_x_start}:{roi_x_end})', fontsize=16, fontweight='bold')
                    ax2.set_xlabel('X Position (nm)')
                    ax2.set_ylabel('Y Position (nm)')
                    
                    plt.tight_layout()
                    st.pyplot(fig_roi, use_container_width=True)
                    plt.close(fig_roi)
                    
                    # ROI statistics
                    st.markdown("##### ROI Statistics")
                    roi_stats = {
                        'Maximum': np.max(roi),
                        'Minimum': np.min(roi),
                        'Mean': np.mean(roi),
                        'Standard Deviation': np.std(roi),
                        '95th Percentile': np.percentile(roi, 95),
                        '5th Percentile': np.percentile(roi, 5)
                    }
                    
                    stats_df = pd.DataFrame([roi_stats])
                    st.dataframe(stats_df.style.format('{:.3f}'), use_container_width=True)
            
            else:
                st.info("💡 No interpolation results available. Please perform interpolation first.")
        
        with tab5:
            st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)
            
            if st.session_state.interpolation_result:
                result = st.session_state.interpolation_result
                
                # Export options with enhanced styling
                st.markdown("### Export Formats")
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                
                with col_e1:
                    # Export as JSON
                    if st.button("📄 Export as JSON", use_container_width=True, key="export_json"):
                        visualization_params = {
                            'colormap': colormap_name,
                            'visualization_type': visualization_type,
                            'plotly_template': plotly_template
                        }
                        export_data = st.session_state.results_manager.prepare_export_data(
                            result, visualization_params
                        )
                        json_str, filename = st.session_state.results_manager.export_to_json(export_data)
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_str,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True,
                            key="download_json"
                        )
                
                with col_e2:
                    # Export as CSV
                    if st.button("📊 Export as CSV", use_container_width=True, key="export_csv"):
                        csv_str, filename = st.session_state.results_manager.export_to_csv(result)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_str,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv"
                        )
                
                with col_e3:
                    # Export individual plots
                    if st.button("🖼️ Export Plots", use_container_width=True, key="export_plots"):
                        saved_files = st.session_state.results_manager.export_fields_as_images(
                            result, st.session_state.heatmap_visualizer, cmap_name=colormap_name
                        )
                        
                        if saved_files:
                            st.success(f"✅ Exported {len(saved_files)} image files")
                            for file_path in saved_files:
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label=f"⬇️ Download {os.path.basename(file_path)}",
                                        data=f.read(),
                                        file_name=os.path.basename(file_path),
                                        mime="image/png",
                                        key=f"download_{os.path.basename(file_path)}"
                                    )
                
                with col_e4:
                    # Export comprehensive report
                    if st.button("📋 Export Report", use_container_width=True, key="export_report"):
                        # Create comprehensive report figure
                        fig_report = st.session_state.heatmap_visualizer.create_source_comparison_dashboard(
                            [st.session_state.solutions[i] for i in st.session_state.selected_sources],
                            result,
                            cmap_name=colormap_name,
                            figsize=(22, 16)
                        )
                        
                        if fig_report:
                            # Save to buffer
                            buf = BytesIO()
                            fig_report.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                            buf.seek(0)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"analysis_report_theta_{result['target_angle']:.1f}_{timestamp}.png"
                            
                            st.download_button(
                                label="⬇️ Download Report PNG",
                                data=buf,
                                file_name=filename,
                                mime="image/png",
                                use_container_width=True,
                                key="download_report"
                            )
                            plt.close(fig_report)
                
                # Bulk export
                st.markdown("---")
                st.markdown("### 📦 Bulk Export Options")
                
                bulk_tabs = st.tabs(["🔄 All Components", "📊 Statistical Data", "🔍 Source Data"])
                
                with bulk_tabs[0]:
                    st.markdown("#### Export All Stress Components")
                    
                    if st.button("🚀 Export All Components", use_container_width=True, type="secondary", key="export_all"):
                        # Create zip file with all components
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Export each component as CSV
                            for component in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                                if component in result['fields']:
                                    component_data = result['fields'][component]
                                    df = pd.DataFrame(component_data)
                                    csv_str = df.to_csv(index=False)
                                    zip_file.writestr(f"{component}_theta_{result['target_angle']:.1f}.csv", csv_str)
                            
                            # Export metadata
                            metadata = {
                                'target_angle': result['target_angle'],
                                'target_params': result['target_params'],
                                'statistics': result['statistics'],
                                'weights': result['weights'],
                                'orientation': result['orientation'],
                                'exported_at': datetime.now().isoformat(),
                                'sources_used': [i for i in st.session_state.selected_sources]
                            }
                            json_str = json.dumps(metadata, indent=2)
                            zip_file.writestr("metadata.json", json_str)
                            
                            # Export visualization parameters
                            viz_params = {
                                'colormap': colormap_name,
                                'visualization_type': visualization_type,
                                'plotly_template': plotly_template
                            }
                            zip_file.writestr("visualization_params.json", json.dumps(viz_params, indent=2))
                        
                        zip_buffer.seek(0)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"stress_analysis_theta_{result['target_angle']:.1f}_{timestamp}.zip"
                        
                        st.download_button(
                            label="⬇️ Download Complete Dataset",
                            data=zip_buffer.getvalue(),
                            file_name=filename,
                            mime="application/zip",
                            use_container_width=True,
                            key="download_complete"
                        )
                
                with bulk_tabs[1]:
                    st.markdown("#### Export Statistical Summary")
                    
                    if st.button("📊 Export Statistics", use_container_width=True, key="export_stats"):
                        # Create comprehensive statistics
                        stats_data = []
                        
                        # Add stress component statistics
                        for comp_name, display_name in [
                            ('von_mises', 'Von Mises Stress'),
                            ('sigma_hydro', 'Hydrostatic Stress'),
                            ('sigma_mag', 'Stress Magnitude')
                        ]:
                            if comp_name in result['statistics']:
                                stats = result['statistics'][comp_name]
                                row = {
                                    'Component': display_name,
                                    'Maximum (GPa)': stats.get('max', stats.get('max_tension', 0)),
                                    'Minimum (GPa)': stats.get('min', stats.get('max_compression', 0)),
                                    'Mean (GPa)': stats['mean'],
                                    'Std Dev (GPa)': stats['std']
                                }
                                stats_data.append(row)
                        
                        # Add weight statistics
                        weights = result['weights']['combined']
                        weight_stats = {
                            'Component': 'Interpolation Weights',
                            'Maximum (GPa)': max(weights),
                            'Minimum (GPa)': min(weights),
                            'Mean (GPa)': np.mean(weights),
                            'Std Dev (GPa)': np.std(weights)
                        }
                        stats_data.append(weight_stats)
                        
                        # Create DataFrame
                        stats_df = pd.DataFrame(stats_data)
                        
                        # Export as CSV
                        csv_stats = stats_df.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"statistical_summary_theta_{result['target_angle']:.1f}_{timestamp}.csv"
                        
                        st.download_button(
                            label="⬇️ Download Statistics CSV",
                            data=csv_stats,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True,
                            key="download_stats_csv"
                        )
                        
                        # Display the statistics
                        st.markdown("##### Statistical Summary")
                        st.dataframe(stats_df.style.format({
                            'Maximum (GPa)': '{:.3f}',
                            'Minimum (GPa)': '{:.3f}',
                            'Mean (GPa)': '{:.3f}',
                            'Std Dev (GPa)': '{:.3f}'
                        }), use_container_width=True)
                
                with bulk_tabs[2]:
                    st.markdown("#### Export Source Simulation Data")
                    
                    if st.button("🔍 Export Source Data", use_container_width=True, key="export_source"):
                        # Create zip file with source data
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Export each selected source
                            for i, src_idx in enumerate(st.session_state.selected_sources):
                                source = st.session_state.solutions[src_idx]
                                params = source['params']
                                theta_deg = np.degrees(params.get('theta', 0))
                                defect_type = params.get('defect_type', 'Unknown')
                                
                                # Extract stress fields from last frame
                                if 'history' in source and source['history']:
                                    last_frame = source['history'][-1]
                                    if 'stresses' in last_frame:
                                        stresses = last_frame['stresses']
                                        
                                        # Export each stress component as CSV
                                        for comp_name in ['von_mises', 'sigma_hydro', 'sigma_mag']:
                                            if comp_name in stresses:
                                                comp_data = stresses[comp_name]
                                                df = pd.DataFrame(comp_data)
                                                csv_str = df.to_csv(index=False)
                                                zip_file.writestr(f"source_{i+1}_{comp_name}_theta_{theta_deg:.1f}.csv", csv_str)
                                        
                                        # Export source metadata
                                        source_metadata = {
                                            'index': i,
                                            'source_params': params,
                                            'theta_degrees': theta_deg,
                                            'defect_type': defect_type,
                                            'weight_in_interpolation': result['weights']['combined'][i] if i < len(result['weights']['combined']) else 0
                                        }
                                        json_str = json.dumps(source_metadata, indent=2)
                                        zip_file.writestr(f"source_{i+1}_metadata.json", json_str)
                        
                        zip_buffer.seek(0)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"source_simulation_data_{timestamp}.zip"
                        
                        st.download_button(
                            label="⬇️ Download Source Data",
                            data=zip_buffer.getvalue(),
                            file_name=filename,
                            mime="application/zip",
                            use_container_width=True,
                            key="download_source_data"
                        )
                
                # Statistics table
                st.markdown("---")
                st.markdown("### 📊 Detailed Stress Statistics")
                
                try:
                    stats_data = []
                    for stat_key, display_name in [('von_mises', 'Von Mises Stress'),
                                                  ('sigma_hydro', 'Hydrostatic Stress'),
                                                  ('sigma_mag', 'Stress Magnitude')]:
                        if stat_key in result['statistics']:
                            component_stats = result['statistics'][stat_key]
                            max_val = component_stats['max']
                            min_val = component_stats['min']
                            mean_val = component_stats.get('mean', 0.0)
                            std_val = component_stats.get('std', 0.0)
                            stats_data.append({
                                'Component': display_name,
                                'Max (GPa)': f"{max_val:.3f}",
                                'Min (GPa)': f"{min_val:.3f}",
                                'Mean (GPa)': f"{mean_val:.3f}",
                                'Std (GPa)': f"{std_val:.3f}"
                            })
                        else:
                            st.warning(f"Statistics not found for {stat_key}")
                    
                    if stats_data:
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True)
                        
                        # Export statistics as CSV
                        csv_stats = df_stats.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Statistics CSV",
                            data=csv_stats,
                            file_name=f"statistics_theta_{result['target_angle']:.1f}.csv",
                            mime="text/csv",
                            key="download_stats"
                        )
                    else:
                        st.info("No statistics data available for export.")
                except KeyError as e:
                    st.error(f"KeyError accessing statistics: {e}")
                    st.info("Statistics structure may be different than expected.")
                    st.json(result['statistics'] if 'statistics' in result else {})
                except Exception as e:
                    st.error(f"Error creating statistics table: {e}")
                    
            else:
                st.info("No interpolation results available. Please perform interpolation first.")
    
    # Add visualization of source data
    with st.expander("🔍 Source Data Visualization", expanded=False):
        if st.session_state.solutions:
            st.markdown("### Source Simulation Data")
            
            # Source selection
            col_src1, col_src2 = st.columns(2)
            with col_src1:
                source_idx = st.selectbox(
                    "Select Source Simulation",
                    range(len(st.session_state.solutions)),
                    format_func=lambda x: f"Source {x+1} (θ={np.degrees(st.session_state.solutions[x]['params'].get('theta', 0)):.1f}°)"
                )
            
            with col_src2:
                src_component = st.selectbox(
                    "Select Stress Component",
                    ["von_mises", "sigma_hydro", "sigma_mag"],
                    format_func=lambda x: {
                        "von_mises": "Von Mises Stress",
                        "sigma_hydro": "Hydrostatic Stress",
                        "sigma_mag": "Stress Magnitude"
                    }[x],
                    key="source_component"
                )
            
            # Get the selected source
            source = st.session_state.solutions[source_idx]
            if 'history' in source and source['history']:
                last_frame = source['history'][-1]
                if 'stresses' in last_frame and src_component in last_frame['stresses']:
                    src_field = last_frame['stresses'][src_component]
                    src_theta = np.degrees(source['params'].get('theta', 0))
                    src_defect = source['params'].get('defect_type', 'Unknown')
                    
                    # Visualize the source field
                    fig_src = st.session_state.heatmap_visualizer.create_stress_heatmap(
                        src_field,
                        title=f"{src_component.replace('_', ' ').title()} - Source {source_idx+1} (θ={src_theta:.1f}°, {src_defect})",
                        cmap_name=colormap_name,
                        colorbar_label=f"{src_component.replace('_', ' ').title()} (GPa)"
                    )
                    st.pyplot(fig_src, use_container_width=True)
                    plt.close(fig_src)
                    
                    # Source statistics
                    with st.expander("📋 Source Statistics", expanded=False):
                        if src_component in last_frame['stresses']:
                            field = last_frame['stresses'][src_component]
                            col_src_stat1, col_src_stat2, col_src_stat3, col_src_stat4 = st.columns(4)
                            with col_src_stat1:
                                st.metric("Max", f"{np.max(field):.3f} GPa")
                            with col_src_stat2:
                                st.metric("Min", f"{np.min(field):.3f} GPa")
                            with col_src_stat3:
                                st.metric("Mean", f"{np.mean(field):.3f} GPa")
                            with col_src_stat4:
                                st.metric("Std", f"{np.std(field):.3f} GPa")
                            
                            # Distribution plot
                            fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
                            ax_dist.hist(field.flatten(), bins=50, alpha=0.7, color='blue')
                            ax_dist.set_xlabel(f'{src_component.replace("_", " ").title()} (GPa)')
                            ax_dist.set_ylabel('Frequency')
                            ax_dist.set_title(f'Distribution for Source {source_idx+1}')
                            ax_dist.grid(True, alpha=0.3)
                            st.pyplot(fig_dist, use_container_width=True)
                            plt.close(fig_dist)
                else:
                    st.warning(f"Stress component '{src_component}' not found in source data")
        else:
            st.info("No source simulations loaded. Please load solutions first.")
    
    # Add advanced analysis section
    with st.expander("🔬 Advanced Analysis", expanded=False):
        if st.session_state.interpolation_result:
            result = st.session_state.interpolation_result
            
            # Orientation sensitivity analysis
            st.markdown("### Orientation Sensitivity Analysis")
            
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                angle_range = st.slider(
                    "Angle Range for Analysis",
                    min_value=0,
                    max_value=180,
                    value=90,
                    step=10,
                    help="Range of angles to analyze around target orientation"
                )
            
            with col_o2:
                num_angles = st.slider(
                    "Number of Angles",
                    min_value=5,
                    max_value=37,
                    value=13,
                    step=2,
                    help="Number of angles to sample in the range"
                )
            
            if st.button("🚀 Run Sensitivity Analysis", key="sensitivity_analysis"):
                with st.spinner("Performing orientation sensitivity analysis..."):
                    # Generate angles
                    target_angle = result['target_angle']
                    angles = np.linspace(
                        max(0, target_angle - angle_range/2),
                        min(360, target_angle + angle_range/2),
                        num_angles
                    )
                    
                    # Store results
                    sensitivity_results = {
                        'angles': angles.tolist(),
                        'max_vm': [],
                        'mean_vm': [],
                        'max_hydro_tension': [],
                        'max_hydro_compression': [],
                        'mean_hydro': []
                    }
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Analyze each angle
                    for i, angle in enumerate(angles):
                        progress = (i + 1) / len(angles)
                        progress_bar.progress(progress)
                        status_text.text(f"Analyzing angle {angle:.1f}° ({i+1}/{len(angles)})")
                        
                        # Create target parameters for this angle
                        target_params = result['target_params'].copy()
                        target_params['theta'] = np.radians(angle)
                        
                        # Interpolate at this angle
                        try:
                            angle_result = st.session_state.transformer_interpolator.interpolate_spatial_fields(
                                [st.session_state.solutions[i] for i in st.session_state.selected_sources],
                                angle,
                                target_params
                            )
                            
                            if angle_result:
                                # Extract statistics
                                vm_stats = angle_result['statistics']['von_mises']
                                hydro_stats = angle_result['statistics']['sigma_hydro']
                                
                                sensitivity_results['max_vm'].append(vm_stats['max'])
                                sensitivity_results['mean_vm'].append(vm_stats['mean'])
                                sensitivity_results['max_hydro_tension'].append(hydro_stats['max_tension'])
                                sensitivity_results['max_hydro_compression'].append(hydro_stats['max_compression'])
                                sensitivity_results['mean_hydro'].append(hydro_stats['mean'])
                        except Exception as e:
                            st.warning(f"Error at angle {angle:.1f}°: {str(e)}")
                            # Append zeros to maintain array length
                            sensitivity_results['max_vm'].append(0)
                            sensitivity_results['mean_vm'].append(0)
                            sensitivity_results['max_hydro_tension'].append(0)
                            sensitivity_results['max_hydro_compression'].append(0)
                            sensitivity_results['mean_hydro'].append(0)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results in session state
                    st.session_state.sensitivity_results = sensitivity_results
                    
                    # Create visualization
                    fig_sens, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Max Von Mises vs angle
                    ax1.plot(angles, sensitivity_results['max_vm'], 'b-', linewidth=2.5, marker='o')
                    ax1.set_xlabel('Orientation Angle (°)', fontsize=12)
                    ax1.set_ylabel('Max Von Mises Stress (GPa)', fontsize=12)
                    ax1.set_title('Max Von Mises vs Orientation', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    ax1.axvline(x=target_angle, color='r', linestyle='--', label=f'Target θ={target_angle:.1f}°')
                    ax1.legend()
                    
                    # Mean Von Mises vs angle
                    ax2.plot(angles, sensitivity_results['mean_vm'], 'g-', linewidth=2.5, marker='o')
                    ax2.set_xlabel('Orientation Angle (°)', fontsize=12)
                    ax2.set_ylabel('Mean Von Mises Stress (GPa)', fontsize=12)
                    ax2.set_title('Mean Von Mises vs Orientation', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    ax2.axvline(x=target_angle, color='r', linestyle='--', label=f'Target θ={target_angle:.1f}°')
                    ax2.legend()
                    
                    # Hydrostatic tension vs angle
                    ax3.plot(angles, sensitivity_results['max_hydro_tension'], 'r-', linewidth=2.5, marker='o', label='Max Tension')
                    ax3.plot(angles, sensitivity_results['max_hydro_compression'], 'b-', linewidth=2.5, marker='o', label='Max Compression')
                    ax3.set_xlabel('Orientation Angle (°)', fontsize=12)
                    ax3.set_ylabel('Hydrostatic Stress (GPa)', fontsize=12)
                    ax3.set_title('Hydrostatic Stress vs Orientation', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                    ax3.axvline(x=target_angle, color='k', linestyle='--', label=f'Target θ={target_angle:.1f}°')
                    ax3.legend()
                    
                    # Mean hydrostatic vs angle
                    ax4.plot(angles, sensitivity_results['mean_hydro'], 'purple', linewidth=2.5, marker='o')
                    ax4.set_xlabel('Orientation Angle (°)', fontsize=12)
                    ax4.set_ylabel('Mean Hydrostatic Stress (GPa)', fontsize=12)
                    ax4.set_title('Mean Hydrostatic vs Orientation', fontsize=14, fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    ax4.axvline(x=target_angle, color='r', linestyle='--', label=f'Target θ={target_angle:.1f}°')
                    ax4.legend()
                    
                    plt.suptitle(f'Orientation Sensitivity Analysis - {defect_type} Defect', 
                               fontsize=18, fontweight='bold', y=0.95)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    
                    st.pyplot(fig_sens, use_container_width=True)
                    plt.close(fig_sens)
                    
                    # Save figure option
                    if st.button("💾 Save Sensitivity Analysis Plot", key="save_sensitivity"):
                        buf = BytesIO()
                        fig_sens.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                        buf.seek(0)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"sensitivity_analysis_{defect_type}_{timestamp}.png"
                        st.download_button(
                            label="⬇️ Download Sensitivity Plot",
                            data=buf,
                            file_name=filename,
                            mime="image/png",
                            key="download_sensitivity"
                        )
        else:
            st.info("No interpolation results available. Please perform interpolation first.")
    
    # Add about section
    with st.expander("ℹ️ About this Application", expanded=False):
        st.markdown("""
        ### Transformer-Based Stress Field Interpolation
        
        This application performs physics-aware interpolation of stress fields from molecular dynamics simulations using a transformer-based architecture.
        
        #### Key Features:
        - **Transformer Architecture**: Uses multi-head attention with spatial locality regularization
        - **Orientation Awareness**: Properly handles angular parameters with periodic boundary conditions
        - **Multiple Stress Components**: Interpolates von Mises, hydrostatic, and stress magnitude fields
        - **Interactive Visualization**: Supports 2D heatmaps, 3D surfaces, and comprehensive dashboards
        - **Colormap Options**: 50+ colormaps including scientific and perceptually uniform options
        - **Export Capabilities**: JSON, CSV, PNG, and ZIP exports for further analysis
        
        #### Technical Details:
        - **Input Features**: 15-dimensional parameter encoding including:
          - Numeric parameters (eigenstrain, kappa)
          - One-hot encoded categorical features (defect type, shape)
          - Orientation features with angular awareness
          - Habit plane proximity metrics
        - **Transformer Configuration**: 
          - 64-dimensional embedding
          - 8 attention heads
          - 3 transformer layers
        - **Spatial Regularization**: Gaussian kernel-based weighting with configurable sigma parameter
        
        #### Citation:
        If you use this tool in your research, please cite:
        ```
        [Citation information would go here]
        ```
        
        #### Contact:
        For questions, bug reports, or feature requests, please contact: [Contact information would go here]
        """)
    
    # Add footer
    st.markdown("""
    ---
    <div style="text-align: center; padding: 20px; color: #64748b; font-size: 0.9rem;">
        <strong>Transformer Stress Field Interpolation</strong> • Version 1.0 • 
        Physics-Informed Machine Learning for Materials Science
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
