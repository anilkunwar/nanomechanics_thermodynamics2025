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
import shutil

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION PATHS
# =============================================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")

# Create solution directory if it doesn't exist
if not os.path.exists(SOLUTION_DIR):
    os.makedirs(SOLUTION_DIR, exist_ok=True)
    st.info(f"📁 Created numerical solutions directory: {SOLUTION_DIR}")

# =============================================
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator:
    """Enhanced attention-based interpolator with spatial locality regularization"""
    
    def __init__(self, input_dim=15, num_heads=4, d_model=32, output_dim=3, 
                 sigma_spatial=0.2, sigma_param=0.2, use_gaussian=True):
        """
        Args:
            input_dim: Dimension of parameter vector (defect + geometry + orientation)
            num_heads: Number of attention heads
            d_model: Dimension of model
            output_dim: Number of output stress components (hydrostatic, magnitude, vonMises)
            sigma_spatial: Spatial locality parameter for Gaussian weighting
            sigma_param: Parameter space locality parameter
            use_gaussian: Whether to use Gaussian spatial regularization
        """
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.output_dim = output_dim
        self.sigma_spatial = sigma_spatial
        self.sigma_param = sigma_param
        self.use_gaussian = use_gaussian
        
        # Initialize model
        self.model = self._build_model()
        
        # File format readers
        self.readers = {
            'pkl': self._read_pkl,
            'pt': self._read_pt,
            'h5': self._read_h5,
            'npz': self._read_npz,
            'sql': self._read_sql,
            'json': self._read_json,
            'npy': self._read_npy,
            'mat': self._read_mat,
            'csv': self._read_csv
        }
    
    def _build_model(self):
        """Build the attention model with spatial regularization"""
        model = torch.nn.ModuleDict({
            # Parameter embeddings with positional encoding for spatial awareness
            'param_embedding': torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.d_model)
            ),
            
            # Multi-head attention with spatial bias
            'attention': torch.nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1
            ),
            
            # Feed-forward with skip connections
            'feed_forward': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 4),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.d_model * 4, self.d_model)
            ),
            
            # Output projection for stress fields
            'output_projection': torch.nn.Sequential(
                torch.nn.Linear(self.d_model, self.d_model * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.d_model * 2, self.output_dim)
            ),
            
            # Spatial regularization network
            'spatial_regularizer': torch.nn.Sequential(
                torch.nn.Linear(2, 32),  # x,y coordinates
                torch.nn.ReLU(),
                torch.nn.Linear(32, self.num_heads)
            ) if self.use_gaussian else None,
            
            # Layer norms
            'norm1': torch.nn.LayerNorm(self.d_model),
            'norm2': torch.nn.LayerNorm(self.d_model)
        })
        
        return model
    
    def _read_pkl(self, file_content):
        """Read pickle format file"""
        return pickle.loads(file_content)
    
    def _read_pt(self, file_content):
        """Read PyTorch format file"""
        buffer = BytesIO(file_content)
        return torch.load(buffer, map_location=torch.device('cpu'))
    
    def _read_h5(self, file_content):
        """Read HDF5 format file"""
        buffer = BytesIO(file_content)
        with h5py.File(buffer, 'r') as f:
            data = {}
            def read_h5_obj(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
                elif isinstance(obj, h5py.Group):
                    data[name] = {}
            f.visititems(read_h5_obj)
        return data
    
    def _read_npz(self, file_content):
        """Read numpy compressed format"""
        buffer = BytesIO(file_content)
        return dict(np.load(buffer, allow_pickle=True))
    
    def _read_sql(self, file_content):
        """Read SQL dump format"""
        buffer = StringIO(file_content.decode('utf-8'))
        conn = sqlite3.connect(':memory:')
        conn.executescript(buffer.read())
        return conn
    
    def _read_json(self, file_content):
        """Read JSON format"""
        return json.loads(file_content.decode('utf-8'))
    
    def _read_npy(self, file_content):
        """Read numpy array format"""
        buffer = BytesIO(file_content)
        return np.load(buffer, allow_pickle=True)
    
    def _read_mat(self, file_content):
        """Read MATLAB format"""
        buffer = BytesIO(file_content)
        from scipy.io import loadmat
        return loadmat(buffer)
    
    def _read_csv(self, file_content):
        """Read CSV format"""
        buffer = StringIO(file_content.decode('utf-8'))
        return pd.read_csv(buffer)
    
    def read_simulation_file(self, file_path, format_type='auto'):
        """
        Read simulation file from path in various formats
        
        Args:
            file_path: Path to simulation file
            format_type: 'auto' or specific format ('pkl', 'pt', etc.)
            
        Returns:
            Dictionary with simulation data
        """
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        filename = os.path.basename(file_path).lower()
        
        # Auto-detect format
        if format_type == 'auto':
            if filename.endswith('.pkl'):
                format_type = 'pkl'
            elif filename.endswith('.pt'):
                format_type = 'pt'
            elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                format_type = 'h5'
            elif filename.endswith('.npz'):
                format_type = 'npz'
            elif filename.endswith('.sql') or filename.endswith('.db'):
                format_type = 'sql'
            elif filename.endswith('.json'):
                format_type = 'json'
            elif filename.endswith('.npy'):
                format_type = 'npy'
            elif filename.endswith('.mat'):
                format_type = 'mat'
            elif filename.endswith('.csv'):
                format_type = 'csv'
            else:
                raise ValueError(f"Unrecognized file format: {filename}")
        
        # Read file
        if format_type in self.readers:
            data = self.readers[format_type](file_content)
            
            # Convert to standardized format
            return self._standardize_data(data, format_type, file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _standardize_data(self, data, format_type, file_path):
        """Convert different formats to standardized structure"""
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'format': format_type,
            'file_path': file_path,
            'filename': os.path.basename(file_path)
        }
        
        if format_type == 'pkl':
            # PKL format from your export
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                
                # Convert history
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                        standardized['history'].append((eta, stresses))
        
        elif format_type == 'pt':
            # PyTorch format
            if isinstance(data, dict):
                standardized['params'] = data.get('params', {})
                standardized['metadata'] = data.get('metadata', {})
                
                # Convert tensors to numpy
                for frame in data.get('history', []):
                    if isinstance(frame, dict):
                        eta = frame.get('eta')
                        stresses = frame.get('stresses', {})
                        
                        # Convert tensors
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
            # HDF5 format
            if 'params' in data:
                standardized['params'] = data['params']
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            if 'history' in data:
                standardized['history'] = data['history']
        
        elif format_type == 'csv':
            # CSV format - assuming it's a flattened simulation
            try:
                # Try to extract metadata from filename
                filename = os.path.basename(file_path)
                if 'ISF' in filename:
                    standardized['params']['defect_type'] = 'ISF'
                elif 'ESF' in filename:
                    standardized['params']['defect_type'] = 'ESF'
                elif 'Twin' in filename:
                    standardized['params']['defect_type'] = 'Twin'
                
                # Extract orientation from filename
                if 'orient0deg' in filename:
                    standardized['params']['orientation'] = 'Horizontal {111} (0°)'
                elif 'orient30deg' in filename:
                    standardized['params']['orientation'] = 'Tilted 30° (1¯10 projection)'
                elif 'orient60deg' in filename:
                    standardized['params']['orientation'] = 'Tilted 60°'
                elif 'orient90deg' in filename:
                    standardized['params']['orientation'] = 'Vertical {111} (90°)'
                
                # Store the CSV data
                standardized['csv_data'] = data
        
        # Add timestamp if missing
        if 'created_at' not in standardized['metadata']:
            standardized['metadata']['created_at'] = datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat()
        
        return standardized
    
    def compute_parameter_vector(self, sim_data):
        """
        Compute parameter vector from simulation data
        
        Args:
            sim_data: Standardized simulation data
            
        Returns:
            parameter_vector: Normalized parameter vector
            param_names: Names of parameters
        """
        params = sim_data.get('params', {})
        
        # Parameter encoding scheme
        param_vector = []
        param_names = []
        
        # 1. Defect type encoding
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        param_names.extend(['defect_ISF', 'defect_ESF', 'defect_Twin'])
        
        # 2. Shape encoding
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
        
        # 3. Numerical parameters (normalized)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        # Normalize eps0 to [0,1] range (0.3-3.0)
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3)
        param_vector.append(eps0_norm)
        param_names.append('eps0_norm')
        
        # Normalize kappa to [0,1] range (0.1-2.0)
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1)
        param_vector.append(kappa_norm)
        param_names.append('kappa_norm')
        
        # Normalize theta (angle) to [0,1] range (0-2π)
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        param_vector.append(theta_norm)
        param_names.append('theta_norm')
        
        # 4. Orientation encoding
        orientation = params.get('orientation', 'Horizontal {111} (0°)')
        orientation_encoding = {
            'Horizontal {111} (0°)': [1, 0, 0, 0],
            'Tilted 30° (1¯10 projection)': [0, 1, 0, 0],
            'Tilted 60°': [0, 0, 1, 0],
            'Vertical {111} (90°)': [0, 0, 0, 1]
        }
        param_vector.extend(orientation_encoding.get(orientation, [0, 0, 0, 0]))
        param_names.extend(['orient_0deg', 'orient_30deg', 'orient_60deg', 'orient_90deg'])
        
        return np.array(param_vector, dtype=np.float32), param_names
    
    def compute_spatial_weights(self, source_coords, target_coords):
        """
        Compute Gaussian spatial weights based on coordinate similarity
        
        Args:
            source_coords: Source simulation coordinates (normalized)
            target_coords: Target simulation coordinates (normalized)
            
        Returns:
            spatial_weights: Gaussian weights for each source
        """
        # Calculate Euclidean distances in parameter space
        distances = np.sqrt(np.sum((source_coords - target_coords)**2, axis=1))
        
        # Apply Gaussian kernel
        spatial_weights = np.exp(-0.5 * (distances / self.sigma_param)**2)
        
        # Normalize weights
        spatial_weights = spatial_weights / (np.sum(spatial_weights) + 1e-8)
        
        return spatial_weights
    
    def prepare_training_data(self, source_simulations):
        """
        Prepare training data from source simulations
        
        Args:
            source_simulations: List of standardized simulation data
            
        Returns:
            X: Parameter vectors (n_sources, n_features)
            Y_stress: Stress fields (n_sources, n_components, H, W)
            spatial_info: Spatial coordinates for regularization
        """
        X_list = []
        Y_list = []
        spatial_coords = []
        
        for sim_data in source_simulations:
            # Get parameter vector
            param_vector, _ = self.compute_parameter_vector(sim_data)
            X_list.append(param_vector)
            
            # Get stress fields from final frame
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]  # Use final frame
                
                # Extract stress components
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                Y_list.append(stress_components)
                
                # Use parameter vector as spatial coordinates (normalized)
                spatial_coords.append(param_vector)
        
        return (
            np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(spatial_coords, dtype=np.float32)
        )
    
    def train(self, source_simulations, epochs=50, lr=0.001):
        """
        Train the attention model on source simulations
        
        Args:
            source_simulations: List of standardized simulation data
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            training_losses: List of losses during training
        """
        # Prepare training data
        X, Y, spatial_coords = self.prepare_training_data(source_simulations)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        Y_tensor = torch.FloatTensor(Y).unsqueeze(0)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Randomly select a target from source for training (leave-one-out)
            target_idx = torch.randint(0, len(source_simulations), (1,)).item()
            
            # Prepare source and target
            source_mask = torch.ones(len(source_simulations), dtype=torch.bool)
            source_mask[target_idx] = False
            
            X_source = X_tensor[:, source_mask, :]
            Y_source = Y_tensor[:, source_mask, :, :, :]
            X_target = X_tensor[:, target_idx:target_idx+1, :]
            Y_target = Y_tensor[:, target_idx:target_idx+1, :, :, :]
            
            # Forward pass
            Y_pred, _ = self.forward(X_source, Y_source, X_target, spatial_coords[source_mask])
            
            # Compute loss
            loss = criterion(Y_pred, Y_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return losses
    
    def forward(self, X_source, Y_source, X_target, spatial_coords=None):
        """
        Forward pass with spatial regularization
        
        Args:
            X_source: Source parameter vectors (batch, n_sources, n_features)
            Y_source: Source stress fields (batch, n_sources, n_components, H, W)
            X_target: Target parameter vector (batch, 1, n_features)
            spatial_coords: Spatial coordinates for regularization
            
        Returns:
            Y_pred: Predicted stress fields
            attention_weights: Attention weights for interpretability
        """
        batch_size, n_sources, _ = X_source.shape
        
        # 1. Embed parameters
        source_embeddings = self.model['param_embedding'](X_source)
        target_embeddings = self.model['param_embedding'](X_target)
        
        # 2. Compute attention with spatial regularization
        if self.use_gaussian and spatial_coords is not None:
            # Add spatial bias to attention
            spatial_coords_tensor = torch.FloatTensor(spatial_coords).unsqueeze(0)
            spatial_bias = self.model['spatial_regularizer'](spatial_coords_tensor)
            spatial_bias = spatial_bias.permute(0, 2, 1)  # (batch, n_heads, n_sources)
        else:
            spatial_bias = None
        
        # Compute attention
        attended, attention_weights = self.model['attention'](
            query=target_embeddings,
            key=source_embeddings,
            value=source_embeddings,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=True
        )
        
        # Apply spatial regularization if available
        if spatial_bias is not None:
            attention_weights = attention_weights * spatial_bias
            attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # 3. Residual connection and normalization
        attended = self.model['norm1'](attended + target_embeddings)
        
        # 4. Feed-forward
        ff_output = self.model['feed_forward'](attended)
        encoded = self.model['norm2'](ff_output + attended)
        
        # 5. Project to stress space weights
        stress_weights = self.model['output_projection'](encoded)
        
        # 6. Apply weights to source stress fields
        stress_weights = torch.softmax(stress_weights, dim=1)
        stress_weights = stress_weights.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        
        # Weighted combination of source stress fields
        Y_pred = torch.sum(Y_source * stress_weights, dim=1)
        
        return Y_pred, attention_weights
    
    def predict(self, source_simulations, target_params):
        """
        Predict stress fields for target parameters
        
        Args:
            source_simulations: List of standardized source simulation data
            target_params: Dictionary of target parameters
            
        Returns:
            predicted_stress: Dictionary of predicted stress fields
            attention_weights: Attention weights for interpretability
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare source data
            X_source, Y_source, spatial_coords = self.prepare_training_data(source_simulations)
            
            # Compute target parameter vector
            # Create temporary simulation data for target
            target_sim_data = {'params': target_params}
            X_target, _ = self.compute_parameter_vector(target_sim_data)
            
            # Convert to tensors
            X_source_tensor = torch.FloatTensor(X_source).unsqueeze(0)
            Y_source_tensor = torch.FloatTensor(Y_source).unsqueeze(0)
            X_target_tensor = torch.FloatTensor(X_target).unsqueeze(0).unsqueeze(0)
            
            # Forward pass
            Y_pred, attention_weights = self.forward(
                X_source_tensor, Y_source_tensor, X_target_tensor, spatial_coords
            )
            
            # Convert to numpy
            Y_pred = Y_pred.squeeze().numpy()
            attention_weights = attention_weights.squeeze().numpy()
            
            # Format output
            predicted_stress = {
                'sigma_hydro': Y_pred[0],
                'sigma_mag': Y_pred[1],
                'von_mises': Y_pred[2],
                'predicted': True
            }
            
            return predicted_stress, attention_weights
    
    def visualize_attention(self, attention_weights, source_names):
        """Visualize attention weights"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of attention weights
        x_pos = np.arange(len(source_names))
        bars = ax1.bar(x_pos, attention_weights, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Source Simulations')
        ax1.set_ylabel('Attention Weight')
        ax1.set_title('Attention Weights for Stress Interpolation')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(source_names, rotation=45, ha='right')
        
        # Add value labels
        for bar, weight in zip(bars, attention_weights):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Heatmap of attention across heads (if multi-head)
        if attention_weights.ndim > 1:
            im = ax2.imshow(attention_weights.T, aspect='auto', cmap='viridis')
            ax2.set_xlabel('Source Simulations')
            ax2.set_ylabel('Attention Heads')
            ax2.set_title('Multi-head Attention Heatmap')
            ax2.set_xticks(range(len(source_names)))
            ax2.set_xticklabels([f'S{i+1}' for i in range(len(source_names))], rotation=45)
            plt.colorbar(im, ax=ax2)
        else:
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'Single-head attention', 
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        return fig

# =============================================
# ERROR HANDLING DECORATOR
# =============================================
def handle_errors(func):
    """Decorator to handle errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"❌ Error in {func.__name__}: {str(e)}")
            st.error("Please check the console for detailed error information.")
            print(f"Error in {func.__name__}: {str(e)}")
            print(traceback.format_exc())
            return None
    return wrapper

# =============================================
# METADATA MANAGEMENT CLASS
# =============================================
class MetadataManager:
    """Centralized metadata management to ensure consistency"""
    
    @staticmethod
    def create_metadata(sim_params, history, run_time=None, **kwargs):
        """Create standardized metadata dictionary"""
        if run_time is None:
            run_time = 0.0
            
        metadata = {
            'run_time': run_time,
            'frames': len(history) if history else 0,
            'grid_size': kwargs.get('grid_size', 128),
            'dx': kwargs.get('dx', 0.1),
            'created_at': datetime.now().isoformat(),
            'colormaps': kwargs.get('colormaps', {
                'eta': sim_params.get('eta_cmap', 'viridis'),
                'sigma': sim_params.get('sigma_cmap', 'hot'),
                'hydro': sim_params.get('hydro_cmap', 'coolwarm'),
                'vm': sim_params.get('vm_cmap', 'plasma')
            }),
            'material_properties': {
                'C11': 124.0,
                'C12': 93.4,
                'C44': 46.1,
                'lattice_constant': 0.4086
            },
            'simulation_parameters': {
                'dt': 0.004,
                'N': kwargs.get('grid_size', 128),
                'dx': kwargs.get('dx', 0.1)
            }
        }
        return metadata
    
    @staticmethod
    def validate_metadata(metadata):
        """Validate metadata structure and add missing fields"""
        if not isinstance(metadata, dict):
            metadata = {}
        
        required_fields = [
            'run_time', 'frames', 'grid_size', 'dx', 'created_at'
        ]
        
        for field in required_fields:
            if field not in metadata:
                if field == 'created_at':
                    metadata[field] = datetime.now().isoformat()
                elif field == 'run_time':
                    metadata[field] = 0.0
                elif field == 'frames':
                    metadata[field] = 0
                elif field == 'grid_size':
                    metadata[field] = 128
                elif field == 'dx':
                    metadata[field] = 0.1
        
        # Ensure colormaps exist
        if 'colormaps' not in metadata:
            metadata['colormaps'] = {
                'eta': 'viridis',
                'sigma': 'hot',
                'hydro': 'coolwarm',
                'vm': 'plasma'
            }
        
        return metadata
    
    @staticmethod
    def get_metadata_field(metadata, field, default=None):
        """Safely get metadata field with default"""
        try:
            return metadata.get(field, default)
        except:
            return default

# Configure page with better styling
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer with Attention", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Analyzer with Spatial-Attention Interpolation")
st.markdown("""
**Run simulations • Upload existing data • Predict stress fields using spatial-attention interpolation**
**Support for PKL, PT, H5, NPZ, SQL, JSON formats • Access to numerical_solutions directory**
""")

# =============================================
# Material & Grid
# =============================================
a = 0.4086
b = a / np.sqrt(6)
d111 = a / np.sqrt(3)

# Elastic constants for FCC Ag (experimental, in GPa)
C11 = 124.0
C12 = 93.4
C44 = 46.1

N = 128
dx = 0.1  # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# ATTENTION INTERPOLATOR INTERFACE WITH NUMERICAL SOLUTIONS
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with numerical solutions directory"""
    
    st.header("🤖 Spatial-Attention Stress Interpolation")
    
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
            num_heads=4,
            sigma_spatial=0.2,
            sigma_param=0.3
        )
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.local_files_cache = {}
    
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
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload Source Data", 
        "📁 Numerical Solutions",
        "🎯 Configure Target", 
        "🚀 Train & Predict", 
        "📊 Results & Export"
    ])
    
    with tab1:
        st.subheader("Upload Source Simulation Files")
        
        # File upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload simulation files (PKL, PT, H5, NPZ, SQL, JSON)",
                type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'sql', 'db', 'json', 'csv', 'npy', 'mat'],
                accept_multiple_files=True,
                help="Upload precomputed simulation files for interpolation basis"
            )
        
        with col2:
            format_type = st.selectbox(
                "File Format",
                ["Auto Detect", "PKL", "PT", "H5", "NPZ", "SQL", "JSON", "CSV", "NPY", "MAT"],
                index=0
            )
            
            if st.button("📥 Load Uploaded Files", type="primary"):
                if uploaded_files:
                    with st.spinner("Loading simulation files..."):
                        loaded_sims = []
                        for uploaded_file in uploaded_files:
                            try:
                                # Read file
                                file_content = uploaded_file.getvalue()
                                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                                with open(temp_path, 'wb') as f:
                                    f.write(file_content)
                                
                                sim_data = st.session_state.interpolator.read_simulation_file(
                                    temp_path, 
                                    format_type.lower() if format_type != "Auto Detect" else "auto"
                                )
                                
                                # Store in session state
                                file_id = f"{uploaded_file.name}_{hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]}"
                                st.session_state.uploaded_files[file_id] = {
                                    'filename': uploaded_file.name,
                                    'data': sim_data,
                                    'format': format_type.lower() if format_type != "Auto Detect" else "auto"
                                }
                                
                                # Add to source simulations
                                st.session_state.source_simulations.append(sim_data)
                                loaded_sims.append(uploaded_file.name)
                                
                                # Clean up temp file
                                os.remove(temp_path)
                                
                            except Exception as e:
                                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                        
                        if loaded_sims:
                            st.success(f"Successfully loaded {len(loaded_sims)} files!")
                            st.write("**Loaded files:**")
                            for filename in loaded_sims:
                                st.write(f"- {filename}")
                else:
                    st.warning("Please upload files first")
        
        # Display loaded simulations
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Source Simulations")
            
            # Create summary table
            summary_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                metadata = sim_data.get('metadata', {})
                
                summary_data.append({
                    'ID': i+1,
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
                
                # Parameter space visualization
                st.subheader("🎯 Parameter Space Visualization")
                
                try:
                    # Extract parameter vectors
                    param_vectors = []
                    for sim_data in st.session_state.source_simulations:
                        param_vector, _ = st.session_state.interpolator.compute_parameter_vector(sim_data)
                        param_vectors.append(param_vector[:3])  # Use first 3 dimensions for visualization
                    
                    if param_vectors:
                        param_vectors = np.array(param_vectors)
                        
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # Scatter plot
                        scatter = ax.scatter(
                            param_vectors[:, 0],  # Defect encoding
                            param_vectors[:, 1],  # Shape encoding
                            param_vectors[:, 2],  # eps0_norm
                            c=range(len(param_vectors)),
                            cmap='viridis',
                            s=100,
                            alpha=0.7
                        )
                        
                        ax.set_xlabel('Defect Encoding')
                        ax.set_ylabel('Shape Encoding')
                        ax.set_zlabel('ε* (normalized)')
                        ax.set_title('Source Simulations in Parameter Space')
                        
                        # Add labels
                        for i, (x, y, z) in enumerate(param_vectors):
                            ax.text(x, y, z, f'S{i+1}', fontsize=8)
                        
                        plt.colorbar(scatter, ax=ax, label='Simulation Index')
                        st.pyplot(fig)
                
                except Exception as e:
                    st.warning(f"Could not visualize parameter space: {str(e)}")
        
        # Clear button
        if st.session_state.source_simulations:
            if st.button("🗑️ Clear All Source Simulations", type="secondary"):
                st.session_state.source_simulations = []
                st.session_state.uploaded_files = {}
                st.success("All source simulations cleared!")
                st.rerun()
    
    with tab2:
        st.subheader("📁 Numerical Solutions Directory")
        st.info(f"**Directory:** `{SOLUTION_DIR}`")
        
        # Check if directory exists
        if not os.path.exists(SOLUTION_DIR):
            st.warning(f"Directory not found: {SOLUTION_DIR}")
            if st.button("Create Directory"):
                os.makedirs(SOLUTION_DIR, exist_ok=True)
                st.success(f"Created directory: {SOLUTION_DIR}")
                st.rerun()
        else:
            # List files in directory
            file_patterns = ['*.pkl', '*.pt', '*.h5', '*.hdf5', '*.npz', '*.sql', '*.db', '*.json', '*.csv', '*.npy', '*.mat']
            all_files = []
            for pattern in file_patterns:
                all_files.extend(glob.glob(os.path.join(SOLUTION_DIR, pattern)))
            
            all_files = sorted(all_files, key=lambda x: os.path.getmtime(x), reverse=True)
            
            if all_files:
                st.success(f"Found {len(all_files)} simulation files")
                
                # File selection
                file_options = {os.path.basename(f): f for f in all_files}
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_files = st.multiselect(
                        "Select files from numerical_solutions directory",
                        options=list(file_options.keys()),
                        help="Hold Ctrl/Cmd to select multiple files"
                    )
                
                with col2:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    load_numerical = st.button("📂 Load Selected Files", type="primary")
                
                if selected_files and load_numerical:
                    with st.spinner(f"Loading {len(selected_files)} files..."):
                        loaded_count = 0
                        for filename in selected_files:
                            try:
                                file_path = file_options[filename]
                                
                                # Check if already loaded
                                file_key = f"numerical_{file_path}"
                                if file_key in st.session_state.local_files_cache:
                                    sim_data = st.session_state.local_files_cache[file_key]
                                else:
                                    # Read file
                                    sim_data = st.session_state.interpolator.read_simulation_file(
                                        file_path, 
                                        "auto"
                                    )
                                    st.session_state.local_files_cache[file_key] = sim_data
                                
                                # Add to source simulations
                                st.session_state.source_simulations.append(sim_data)
                                loaded_count += 1
                                
                            except Exception as e:
                                st.error(f"Error loading {filename}: {str(e)}")
                        
                        if loaded_count > 0:
                            st.success(f"Successfully loaded {loaded_count} files from numerical_solutions!")
                            st.rerun()
                
                # File management section
                st.subheader("🗃️ File Management")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Refresh File List"):
                        st.rerun()
                
                with col2:
                    # Upload to numerical_solutions
                    st.write("Upload to directory:")
                    upload_file = st.file_uploader(
                        "Upload file to numerical_solutions",
                        type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'sql', 'db', 'json', 'csv', 'npy', 'mat'],
                        key="upload_to_numerical"
                    )
                    
                    if upload_file and st.button("💾 Save to Directory"):
                        save_path = os.path.join(SOLUTION_DIR, upload_file.name)
                        with open(save_path, 'wb') as f:
                            f.write(upload_file.getvalue())
                        st.success(f"Saved {upload_file.name} to numerical_solutions!")
                        st.rerun()
                
                with col3:
                    # Create sample files if directory is empty
                    if len(all_files) == 0:
                        st.info("Directory is empty")
                        if st.button("📝 Create Sample Files"):
                            create_sample_simulations()
                            st.success("Created sample simulation files!")
                            st.rerun()
                
                # Display file details
                st.subheader("📋 File Details")
                
                file_details = []
                for file_path in all_files[:20]:  # Show first 20 files
                    stat = os.stat(file_path)
                    file_details.append({
                        'Filename': os.path.basename(file_path),
                        'Size (KB)': f"{stat.st_size / 1024:.1f}",
                        'Modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                        'Type': os.path.splitext(file_path)[1][1:].upper()
                    })
                
                if file_details:
                    df_files = pd.DataFrame(file_details)
                    st.dataframe(df_files, use_container_width=True)
                
                # Quick load by file type
                st.subheader("⚡ Quick Load by File Type")
                
                file_types = {}
                for file_path in all_files:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext not in file_types:
                        file_types[ext] = []
                    file_types[ext].append(file_path)
                
                if file_types:
                    col1, col2, col3 = st.columns(3)
                    type_cols = [col1, col2, col3]
                    
                    for i, (ext, files) in enumerate(list(file_types.items())[:3]):
                        with type_cols[i]:
                            if st.button(f"📚 Load all {ext.upper()} files ({len(files)})"):
                                with st.spinner(f"Loading {len(files)} {ext.upper()} files..."):
                                    for file_path in files:
                                        try:
                                            file_key = f"numerical_{file_path}"
                                            if file_key not in st.session_state.local_files_cache:
                                                sim_data = st.session_state.interpolator.read_simulation_file(
                                                    file_path, "auto"
                                                )
                                                st.session_state.local_files_cache[file_key] = sim_data
                                            
                                            st.session_state.source_simulations.append(
                                                st.session_state.local_files_cache[file_key]
                                            )
                                        except Exception as e:
                                            st.error(f"Error loading {os.path.basename(file_path)}: {str(e)}")
                                    
                                    st.success(f"Loaded {len(files)} {ext.upper()} files!")
                                    st.rerun()
            else:
                st.info("No simulation files found in numerical_solutions directory")
                
                # Create sample files
                if st.button("📝 Create Sample Simulation Files", type="primary"):
                    create_sample_simulations()
                    st.success("Created sample simulation files!")
                    st.rerun()
    
    with tab3:
        st.subheader("Configure Target Parameters")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please upload at least 2 source simulations first")
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
                
                target_orientation = st.selectbox(
                    "Target Orientation",
                    ["Horizontal {111} (0°)", 
                     "Tilted 30° (1¯10 projection)", 
                     "Tilted 60°", 
                     "Vertical {111} (90°)"],
                    index=0,
                    key="target_orientation"
                )
                
                # Map orientation to angle
                angle_map = {
                    "Horizontal {111} (0°)": 0,
                    "Tilted 30° (1¯10 projection)": 30,
                    "Tilted 60°": 60,
                    "Vertical {111} (90°)": 90,
                }
                target_theta = np.deg2rad(angle_map.get(target_orientation, 0))
                
                st.info(f"**Target θ:** {np.rad2deg(target_theta):.1f}°")
            
            # Store target parameters
            target_params = {
                'defect_type': target_defect,
                'shape': target_shape,
                'eps0': target_eps0,
                'kappa': target_kappa,
                'orientation': target_orientation,
                'theta': target_theta
            }
            
            st.session_state.target_params = target_params
            
            # Show parameter comparison
            st.subheader("📊 Parameter Comparison")
            
            # Create comparison table
            comparison_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                comparison_data.append({
                    'Source': f'S{i+1}',
                    'Defect': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'ε*': params.get('eps0', 'Unknown'),
                    'κ': params.get('kappa', 'Unknown'),
                    'Orientation': params.get('orientation', 'Unknown')
                })
            
            # Add target
            comparison_data.append({
                'Source': '🎯 TARGET',
                'Defect': target_defect,
                'Shape': target_shape,
                'ε*': target_eps0,
                'κ': target_kappa,
                'Orientation': target_orientation
            })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison.style.apply(
                lambda x: ['background-color: #ffd700' if x.name == len(comparison_data)-1 else '' for _ in x],
                axis=1
            ), use_container_width=True)
    
    with tab4:
        st.subheader("Train Model and Predict")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please upload at least 2 source simulations and configure target")
        elif 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure target parameters first")
        else:
            # Training configuration
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider("Training Epochs", 10, 200, 50, 10)
                learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            
            with col2:
                batch_size = st.slider("Batch Size", 1, 16, 4, 1)
                validation_split = st.slider("Validation Split", 0.0, 0.5, 0.2, 0.05)
            
            # Training button
            if st.button("🚀 Train & Predict", type="primary"):
                with st.spinner("Training attention model and predicting..."):
                    try:
                        # Train model
                        losses = st.session_state.interpolator.train(
                            st.session_state.source_simulations,
                            epochs=epochs,
                            lr=learning_rate
                        )
                        
                        # Store losses
                        st.session_state.training_losses = losses
                        
                        # Make prediction
                        predicted_stress, attention_weights = st.session_state.interpolator.predict(
                            st.session_state.source_simulations,
                            st.session_state.target_params
                        )
                        
                        # Store results
                        st.session_state.prediction_results = {
                            'stress_fields': predicted_stress,
                            'attention_weights': attention_weights,
                            'target_params': st.session_state.target_params,
                            'training_losses': losses
                        }
                        
                        st.success("✅ Training and prediction complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during training/prediction: {str(e)}")
                        print(traceback.format_exc())
        
        # Display training results if available
        if 'training_losses' in st.session_state:
            st.subheader("📈 Training Progress")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(st.session_state.training_losses, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Training Loss Convergence')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            st.pyplot(fig)
    
    with tab5:
        st.subheader("Prediction Results")
        
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            results = st.session_state.prediction_results
            
            # Display attention weights
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔍 Attention Analysis")
                
                # Create source names
                source_names = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
                
                # Visualize attention
                fig_attention = st.session_state.interpolator.visualize_attention(
                    results['attention_weights'],
                    source_names
                )
                st.pyplot(fig_attention)
            
            with col2:
                st.subheader("📊 Attention Statistics")
                
                # Calculate statistics
                attn_weights = results['attention_weights'].flatten()
                
                st.metric("Max Weight", f"{np.max(attn_weights):.3f}")
                st.metric("Min Weight", f"{np.min(attn_weights):.3f}")
                st.metric("Mean Weight", f"{np.mean(attn_weights):.3f}")
                st.metric("Std Dev", f"{np.std(attn_weights):.3f}")
                
                # Dominant source
                if attn_weights.ndim == 1:
                    dominant_idx = np.argmax(attn_weights)
                    st.success(f"**Dominant Source:** S{dominant_idx + 1}")
            
            # Display predicted stress fields
            st.subheader("🎯 Predicted Stress Fields")
            
            stress_fields = results['stress_fields']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            titles = ['Hydrostatic Stress (GPa)', 'Stress Magnitude (GPa)', 'Von Mises Stress (GPa)']
            components = ['sigma_hydro', 'sigma_mag', 'von_mises']
            
            for ax, title, comp in zip(axes, titles, components):
                if comp in stress_fields:
                    im = ax.imshow(stress_fields[comp], extent=extent, cmap='coolwarm',
                                  origin='lower', aspect='equal')
                    ax.set_title(title)
                    ax.set_xlabel('x (nm)')
                    ax.set_ylabel('y (nm)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(title)
            
            st.pyplot(fig)
            
            # Stress statistics
            st.subheader("📊 Stress Field Statistics")
            
            stats_data = []
            for comp in components:
                if comp in stress_fields:
                    data = stress_fields[comp]
                    stats_data.append({
                        'Component': comp,
                        'Max (GPa)': float(np.nanmax(data)),
                        'Min (GPa)': float(np.nanmin(data)),
                        'Mean (GPa)': float(np.nanmean(data)),
                        'Std Dev': float(np.nanstd(data))
                    })
            
            if stats_data:
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats.style.format({
                    'Max (GPa)': '{:.3f}',
                    'Min (GPa)': '{:.3f}',
                    'Mean (GPa)': '{:.3f}',
                    'Std Dev': '{:.3f}'
                }), use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Results")
            
            export_col1, export_col2, export_col3, export_col4 = st.columns(4)
            
            with export_col1:
                if st.button("💾 Save as PKL", type="secondary"):
                    # Create export data
                    export_data = {
                        'prediction_results': results,
                        'source_simulations_count': len(st.session_state.source_simulations),
                        'target_params': st.session_state.target_params,
                        'interpolator_config': {
                            'num_heads': st.session_state.interpolator.num_heads,
                            'sigma_spatial': st.session_state.interpolator.sigma_spatial,
                            'sigma_param': st.session_state.interpolator.sigma_param
                        },
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    # Create download button
                    pkl_buffer = BytesIO()
                    pickle.dump(export_data, pkl_buffer)
                    pkl_buffer.seek(0)
                    
                    filename = f"attention_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    st.download_button(
                        label="Download PKL",
                        data=pkl_buffer,
                        file_name=filename,
                        mime="application/octet-stream"
                    )
            
            with export_col2:
                if st.button("⚡ Save as PT", type="secondary"):
                    # Convert to PyTorch format
                    torch_data = {
                        'predicted_stress': {k: torch.FloatTensor(v) for k, v in results['stress_fields'].items() 
                                           if k != 'predicted'},
                        'attention_weights': torch.FloatTensor(results['attention_weights']),
                        'target_params': st.session_state.target_params,
                        'training_losses': torch.FloatTensor(results['training_losses'])
                    }
                    
                    pt_buffer = BytesIO()
                    torch.save(torch_data, pt_buffer)
                    pt_buffer.seek(0)
                    
                    filename = f"attention_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                    st.download_button(
                        label="Download PT",
                        data=pt_buffer,
                        file_name=filename,
                        mime="application/octet-stream"
                    )
            
            with export_col3:
                if st.button("📊 Export Report", type="secondary"):
                    # Create comprehensive report
                    report = f"""
                    SPATIAL-ATTENTION STRESS PREDICTION REPORT
                    ============================================
                    
                    Generated: {datetime.now().isoformat()}
                    
                    1. MODEL CONFIGURATION
                    -----------------------
                    - Number of attention heads: {st.session_state.interpolator.num_heads}
                    - Spatial sigma (σ_spatial): {st.session_state.interpolator.sigma_spatial}
                    - Parameter sigma (σ_param): {st.session_state.interpolator.sigma_param}
                    - Gaussian regularization: {st.session_state.interpolator.use_gaussian}
                    
                    2. SOURCE SIMULATIONS
                    ---------------------
                    Total sources: {len(st.session_state.source_simulations)}
                    
                    """
                    
                    # Add source details
                    for i, sim_data in enumerate(st.session_state.source_simulations):
                        params = sim_data.get('params', {})
                        report += f"\nSource S{i+1}:"
                        report += f"\n  - Defect: {params.get('defect_type', 'Unknown')}"
                        report += f"\n  - Shape: {params.get('shape', 'Unknown')}"
                        report += f"\n  - ε*: {params.get('eps0', 'Unknown')}"
                        report += f"\n  - κ: {params.get('kappa', 'Unknown')}"
                        report += f"\n  - Frames: {len(sim_data.get('history', []))}"
                    
                    # Add target details
                    target = st.session_state.target_params
                    report += f"\n\n3. TARGET PARAMETERS\n-------------------"
                    report += f"\n- Defect: {target.get('defect_type', 'Unknown')}"
                    report += f"\n- Shape: {target.get('shape', 'Unknown')}"
                    report += f"\n- ε*: {target.get('eps0', 'Unknown')}"
                    report += f"\n- κ: {target.get('kappa', 'Unknown')}"
                    report += f"\n- Orientation: {target.get('orientation', 'Unknown')}"
                    
                    # Add attention weights
                    report += f"\n\n4. ATTENTION WEIGHTS\n-------------------\n"
                    if results['attention_weights'].ndim == 1:
                        for i, weight in enumerate(results['attention_weights']):
                            report += f"S{i+1}: {weight:.4f}\n"
                    else:
                        report += str(results['attention_weights'])
                    
                    # Add stress statistics
                    report += f"\n\n5. PREDICTED STRESS STATISTICS (GPa)\n-----------------------------------\n"
                    for comp in ['sigma_hydro', 'sigma_mag', 'von_mises']:
                        if comp in results['stress_fields']:
                            data = results['stress_fields'][comp]
                            report += f"\n{comp}:"
                            report += f"\n  Max: {np.nanmax(data):.3f}"
                            report += f"\n  Min: {np.nanmin(data):.3f}"
                            report += f"\n  Mean: {np.nanmean(data):.3f}"
                            report += f"\n  Std: {np.nanstd(data):.3f}"
                    
                    # Create download
                    report_buffer = BytesIO()
                    report_buffer.write(report.encode('utf-8'))
                    report_buffer.seek(0)
                    
                    filename = f"attention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    st.download_button(
                        label="Download Report",
                        data=report_buffer,
                        file_name=filename,
                        mime="text/plain"
                    )
            
            with export_col4:
                if st.button("📁 Save to Numerical Solutions", type="secondary"):
                    # Save prediction to numerical_solutions directory
                    filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    filepath = os.path.join(SOLUTION_DIR, filename)
                    
                    export_data = {
                        'prediction_results': results,
                        'source_simulations_count': len(st.session_state.source_simulations),
                        'target_params': st.session_state.target_params,
                        'interpolator_config': {
                            'num_heads': st.session_state.interpolator.num_heads,
                            'sigma_spatial': st.session_state.interpolator.sigma_spatial,
                            'sigma_param': st.session_state.interpolator.sigma_param
                        },
                        'export_timestamp': datetime.now().isoformat()
                    }
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(export_data, f)
                    
                    st.success(f"✅ Prediction saved to: {filepath}")

def create_sample_simulations():
    """Create sample simulation files in the numerical_solutions directory"""
    
    # Create sample data structure
    sample_data = []
    
    # Sample defect configurations
    defect_configs = [
        {'defect_type': 'ISF', 'eps0': 0.707, 'kappa': 0.6, 'shape': 'Square', 'orientation': 'Horizontal {111} (0°)'},
        {'defect_type': 'ESF', 'eps0': 1.414, 'kappa': 0.7, 'shape': 'Rectangle', 'orientation': 'Tilted 30° (1¯10 projection)'},
        {'defect_type': 'Twin', 'eps0': 2.121, 'kappa': 0.3, 'shape': 'Ellipse', 'orientation': 'Vertical {111} (90°)'},
        {'defect_type': 'ISF', 'eps0': 0.850, 'kappa': 0.5, 'shape': 'Horizontal Fault', 'orientation': 'Tilted 60°'},
        {'defect_type': 'ESF', 'eps0': 1.200, 'kappa': 0.8, 'shape': 'Vertical Fault', 'orientation': 'Horizontal {111} (0°)'}
    ]
    
    for i, config in enumerate(defect_configs):
        # Create sample stress fields (simplified for demo)
        np.random.seed(42 + i)
        
        # Create random stress field
        stress_field = np.random.randn(N, N) * 2.0
        stress_field = gaussian_filter(stress_field, sigma=2)
        
        # Create sample data
        sample_sim = {
            'params': config,
            'history': [
                {
                    'eta': np.random.rand(N, N) * 0.5 + 0.3,
                    'stresses': {
                        'sigma_hydro': stress_field,
                        'sigma_mag': np.abs(stress_field) * 1.5,
                        'von_mises': stress_field * 1.2
                    }
                }
            ],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'grid_size': N,
                'dx': dx,
                'frames': 1,
                'run_time': 1.23
            }
        }
        
        # Save in different formats
        base_name = f"{config['defect_type']}_orient{config['orientation'][:2].replace(' ', '')}_{config['shape']}_eps0-{config['eps0']}_kappa-{config['kappa']}"
        
        # PKL format
        pkl_path = os.path.join(SOLUTION_DIR, f"{base_name}.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(sample_sim, f)
        
        # PT format
        pt_path = os.path.join(SOLUTION_DIR, f"{base_name}.pt")
        torch_data = {
            'params': config,
            'metadata': sample_sim['metadata'],
            'history': [
                {
                    'eta': torch.FloatTensor(sample_sim['history'][0]['eta']),
                    'stresses': {k: torch.FloatTensor(v) for k, v in sample_sim['history'][0]['stresses'].items()}
                }
            ]
        }
        torch.save(torch_data, pt_path)
        
        # NPZ format
        npz_path = os.path.join(SOLUTION_DIR, f"{base_name}.npz")
        np.savez_compressed(
            npz_path,
            params=config,
            eta=sample_sim['history'][0]['eta'],
            stresses=sample_sim['history'][0]['stresses']
        )
        
        # JSON format
        json_path = os.path.join(SOLUTION_DIR, f"{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'params': config,
                'metadata': sample_sim['metadata'],
                'data_info': {
                    'eta_shape': sample_sim['history'][0]['eta'].shape,
                    'stresses_keys': list(sample_sim['history'][0]['stresses'].keys())
                }
            }, f, indent=2)

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    """Main application with attention interpolation"""
    
    # Sidebar operation mode
    st.sidebar.header("🔧 Operation Mode")
    
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["Run New Simulation", "Compare Saved Simulations", 
         "Single Simulation View", "Attention Interpolation"],
        index=3  # Default to Attention Interpolation
    )
    
    if operation_mode == "Attention Interpolation":
        # Run attention interpolation interface
        create_attention_interface()
    
    else:
        # Original simulation interface (kept for compatibility)
        st.warning("⚠️ This mode is not fully integrated with attention interpolation.")
        st.info("Please use 'Attention Interpolation' mode for spatial-attention predictions.")
        
        # Placeholder for original interface
        st.header("Original Simulation Interface")
        st.write("This interface is available but separate from attention interpolation.")
        
        if operation_mode == "Run New Simulation":
            st.subheader("Run New Simulation")
            
        elif operation_mode == "Compare Saved Simulations":
            st.subheader("Compare Saved Simulations")
            
        elif operation_mode == "Single Simulation View":
            st.subheader("Single Simulation View")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Analysis: Spatial-Attention Interpolation", expanded=False):
    st.markdown("""
    ## 🎯 **Spatial Locality Regularized Attention Interpolation with Numerical Solutions**
    
    ### **📁 Numerical Solutions Directory Integration**
    
    The system now includes seamless integration with the `numerical_solutions` directory:
    
    **Directory Structure:**
    ```
    your_project/
    ├── main_script.py
    └── numerical_solutions/
        ├── ISF_orient0deg_Square_eps0-0.707_kappa-0.6.pkl
        ├── ESF_orient30deg_Rectangle_eps0-1.414_kappa-0.7.pt
        ├── Twin_orient90deg_Ellipse_eps0-2.121_kappa-0.3.npz
        └── prediction_20241215_143022.pkl
    ```
    
    **Key Features:**
    1. **Auto-discovery**: Automatically finds simulation files in the directory
    2. **Multi-format support**: Reads PKL, PT, NPZ, JSON, CSV, H5, MAT formats
    3. **File management**: Upload, organize, and manage simulation files
    4. **Caching**: Intelligent caching for fast reloading
    5. **Sample generation**: Create sample simulations for testing
    
    **Workflow Integration:**
    1. **Store**: Save simulation results to `numerical_solutions/`
    2. **Load**: Select files from dropdown for attention interpolation
    3. **Predict**: Generate new predictions using stored simulations
    4. **Export**: Save predictions back to the directory
    
    **Advanced capabilities for efficient scientific workflow management!**
    """)
    
    # Display directory statistics
    if os.path.exists(SOLUTION_DIR):
        file_patterns = ['*.pkl', '*.pt', '*.h5', '*.hdf5', '*.npz', '*.sql', '*.db', '*.json', '*.csv', '*.npy', '*.mat']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(SOLUTION_DIR, pattern)))
        
        total_size = sum(os.path.getsize(f) for f in all_files)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(all_files))
        with col2:
            st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
        with col3:
            st.metric("Directory", os.path.basename(SOLUTION_DIR))
        with col4:
            st.metric("Path", "..." + SOLUTION_DIR[-30:])

# Run the main application
if __name__ == "__main__":
    main()

st.caption("🔬 Spatial-Attention Stress Interpolation • Numerical Solutions Directory • 2025")
