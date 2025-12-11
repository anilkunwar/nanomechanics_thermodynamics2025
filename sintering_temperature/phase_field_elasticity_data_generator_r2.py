import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import pandas as pd
import zipfile
from io import BytesIO, StringIO
import time
import hashlib
import json
from datetime import datetime
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import warnings
import pickle
import torch
import torch.nn as nn
import h5py
import tempfile
import base64

warnings.filterwarnings('ignore')

# =============================================
# PHASE FIELD DATASET GENERATOR
# =============================================
class PhaseFieldDatasetGenerator:
    """Generate comprehensive datasets from phase field simulations for ML training"""
    
    def __init__(self):
        self.simulations = {}
        self.datasets = {}
        self.attention_interpolator = None
        
    def generate_parameter_space(self, num_simulations=50):
        """Generate diverse parameter combinations for dataset"""
        parameter_space = []
        
        # Define parameter ranges
        defect_types = ["ISF", "ESF", "Twin"]
        shapes = ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"]
        orientations = ["Horizontal {111} (0°)", "Tilted 30° (1¯10 projection)", 
                       "Tilted 60°", "Vertical {111} (90°)"]
        
        # Create parameter combinations
        np.random.seed(42)
        
        for i in range(num_simulations):
            params = {
                'defect_type': np.random.choice(defect_types),
                'shape': np.random.choice(shapes),
                'eps0': np.random.uniform(0.3, 3.0),
                'kappa': np.random.uniform(0.1, 2.0),
                'orientation': np.random.choice(orientations),
                'steps': 100,
                'save_every': 20
            }
            
            # Set theta based on orientation
            angle_map = {
                "Horizontal {111} (0°)": 0,
                "Tilted 30° (1¯10 projection)": 30,
                "Tilted 60°": 60,
                "Vertical {111} (90°)": 90,
            }
            params['theta'] = np.deg2rad(angle_map[params['orientation']])
            
            parameter_space.append(params)
        
        return parameter_space
    
    def create_ml_ready_dataset(self, simulations, include_evolution=False):
        """Create ML-ready dataset from simulations"""
        
        # Initialize data structures
        X_params = []  # Parameter vectors
        Y_stress = []  # Stress fields
        Y_defect = []  # Defect fields
        metadata = []
        
        for sim_id, sim_data in simulations.items():
            params = sim_data['params']
            history = sim_data['history']
            
            # Encode parameters
            param_vector = self._encode_parameters(params)
            
            if include_evolution:
                # Include all frames
                for frame_idx, (eta, stress_fields) in enumerate(history):
                    X_params.append(param_vector)
                    Y_stress.append([
                        stress_fields['sigma_hydro'],
                        stress_fields['sigma_mag'],
                        stress_fields['von_mises']
                    ])
                    Y_defect.append(eta)
                    metadata.append({
                        'sim_id': sim_id,
                        'frame_idx': frame_idx,
                        'total_frames': len(history)
                    })
            else:
                # Only final frame
                eta, stress_fields = history[-1]
                X_params.append(param_vector)
                Y_stress.append([
                    stress_fields['sigma_hydro'],
                    stress_fields['sigma_mag'],
                    stress_fields['von_mises']
                ])
                Y_defect.append(eta)
                metadata.append({
                    'sim_id': sim_id,
                    'frame_idx': len(history) - 1,
                    'total_frames': len(history)
                })
        
        # Convert to arrays
        X_params = np.array(X_params, dtype=np.float32)
        Y_stress = np.array(Y_stress, dtype=np.float32)  # Shape: (n_samples, 3, N, N)
        Y_defect = np.array(Y_defect, dtype=np.float32)  # Shape: (n_samples, N, N)
        
        # Create dataset dictionary
        dataset = {
            'X_params': X_params,
            'Y_stress': Y_stress,
            'Y_defect': Y_defect,
            'metadata': metadata,
            'param_names': self._get_parameter_names(),
            'stress_names': ['hydrostatic', 'magnitude', 'von_mises'],
            'grid_info': {
                'N': N,
                'dx': dx,
                'extent': extent.tolist()
            }
        }
        
        return dataset
    
    def _encode_parameters(self, params):
        """Encode simulation parameters as ML-ready vector"""
        
        # Normalize numerical parameters
        eps0_norm = (params['eps0'] - 0.3) / (3.0 - 0.3)
        kappa_norm = (params['kappa'] - 0.1) / (2.0 - 0.1)
        theta_norm = params['theta'] / (2 * np.pi)
        
        # One-hot encode defect type
        defect_encoding = np.zeros(3)
        defect_map = {"ISF": 0, "ESF": 1, "Twin": 2}
        defect_encoding[defect_map[params['defect_type']]] = 1
        
        # One-hot encode shape
        shape_encoding = np.zeros(5)
        shape_map = {
            "Square": 0, 
            "Horizontal Fault": 1, 
            "Vertical Fault": 2, 
            "Rectangle": 3, 
            "Ellipse": 4
        }
        shape_encoding[shape_map[params['shape']]] = 1
        
        # One-hot encode orientation
        orientation_encoding = np.zeros(4)
        orientation_map = {
            "Horizontal {111} (0°)": 0,
            "Tilted 30° (1¯10 projection)": 1,
            "Tilted 60°": 2,
            "Vertical {111} (90°)": 3
        }
        orientation_encoding[orientation_map[params['orientation']]] = 1
        
        # Combine all features
        param_vector = np.concatenate([
            [eps0_norm, kappa_norm, theta_norm],
            defect_encoding,
            shape_encoding,
            orientation_encoding
        ])
        
        return param_vector
    
    def _get_parameter_names(self):
        """Get parameter names for ML dataset"""
        return [
            'eps0_norm', 'kappa_norm', 'theta_norm',
            'defect_ISF', 'defect_ESF', 'defect_Twin',
            'shape_square', 'shape_horizontal', 'shape_vertical', 
            'shape_rectangle', 'shape_ellipse',
            'orient_0deg', 'orient_30deg', 'orient_60deg', 'orient_90deg'
        ]
    
    def export_dataset(self, dataset, format='h5'):
        """Export dataset in various formats"""
        
        if format == 'h5':
            return self._export_hdf5(dataset)
        elif format == 'npz':
            return self._export_npz(dataset)
        elif format == 'pt':
            return self._export_torch(dataset)
        elif format == 'pkl':
            return self._export_pickle(dataset)
        elif format == 'csv':
            return self._export_csv(dataset)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_hdf5(self, dataset):
        """Export to HDF5 format"""
        buffer = BytesIO()
        
        with h5py.File(buffer, 'w') as f:
            # Store datasets with compression
            f.create_dataset('X_params', data=dataset['X_params'], compression='gzip')
            f.create_dataset('Y_stress', data=dataset['Y_stress'], compression='gzip')
            f.create_dataset('Y_defect', data=dataset['Y_defect'], compression='gzip')
            
            # Store metadata
            metadata_group = f.create_group('metadata')
            for i, meta in enumerate(dataset['metadata']):
                meta_group = metadata_group.create_group(f'sample_{i:06d}')
                for key, value in meta.items():
                    meta_group.attrs[key] = value
            
            # Store grid information
            grid_group = f.create_group('grid')
            grid_group.attrs['N'] = dataset['grid_info']['N']
            grid_group.attrs['dx'] = dataset['grid_info']['dx']
            grid_group.attrs['extent'] = json.dumps(dataset['grid_info']['extent'])
            
            # Store parameter and stress names
            f.create_dataset('param_names', data=np.array(dataset['param_names'], dtype='S'))
            f.create_dataset('stress_names', data=np.array(dataset['stress_names'], dtype='S'))
            
            # Store creation timestamp
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['num_samples'] = len(dataset['X_params'])
        
        buffer.seek(0)
        return buffer
    
    def _export_npz(self, dataset):
        """Export to compressed numpy format"""
        buffer = BytesIO()
        
        # Save with compression
        np.savez_compressed(
            buffer,
            X_params=dataset['X_params'],
            Y_stress=dataset['Y_stress'],
            Y_defect=dataset['Y_defect'],
            metadata=np.array(dataset['metadata'], dtype=object),
            param_names=np.array(dataset['param_names'], dtype='U'),
            stress_names=np.array(dataset['stress_names'], dtype='U'),
            grid_info=np.array([json.dumps(dataset['grid_info'])], dtype='U')
        )
        
        buffer.seek(0)
        return buffer
    
    def _export_torch(self, dataset):
        """Export to PyTorch format"""
        buffer = BytesIO()
        
        # Convert to tensors
        X_tensor = torch.from_numpy(dataset['X_params'])
        Y_stress_tensor = torch.from_numpy(dataset['Y_stress'])
        Y_defect_tensor = torch.from_numpy(dataset['Y_defect'])
        
        # Create dataset dictionary
        torch_dataset = {
            'X_params': X_tensor,
            'Y_stress': Y_stress_tensor,
            'Y_defect': Y_defect_tensor,
            'metadata': dataset['metadata'],
            'param_names': dataset['param_names'],
            'stress_names': dataset['stress_names'],
            'grid_info': dataset['grid_info']
        }
        
        torch.save(torch_dataset, buffer)
        buffer.seek(0)
        return buffer
    
    def _export_pickle(self, dataset):
        """Export to pickle format"""
        buffer = BytesIO()
        pickle.dump(dataset, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        buffer.seek(0)
        return buffer
    
    def _export_csv(self, dataset):
        """Export to CSV format (flattened)"""
        buffer = StringIO()
        
        # Flatten stress and defect fields
        num_samples = dataset['X_params'].shape[0]
        grid_size = N * N
        
        # Create column names
        param_names = dataset['param_names']
        hydro_columns = [f'hydro_{i}' for i in range(grid_size)]
        mag_columns = [f'mag_{i}' for i in range(grid_size)]
        vm_columns = [f'vm_{i}' for i in range(grid_size)]
        defect_columns = [f'defect_{i}' for i in range(grid_size)]
        
        all_columns = param_names + hydro_columns + mag_columns + vm_columns + defect_columns
        
        # Create DataFrame
        data = np.hstack([
            dataset['X_params'],
            dataset['Y_stress'][:, 0].reshape(num_samples, -1),  # hydrostatic
            dataset['Y_stress'][:, 1].reshape(num_samples, -1),  # magnitude
            dataset['Y_stress'][:, 2].reshape(num_samples, -1),  # von mises
            dataset['Y_defect'].reshape(num_samples, -1)         # defect field
        ])
        
        df = pd.DataFrame(data, columns=all_columns)
        
        # Add metadata columns
        for i, meta in enumerate(dataset['metadata']):
            for key, value in meta.items():
                if i == 0:
                    df.insert(i, key, [value] + ['']*(num_samples-1))
                else:
                    df.at[i, key] = value
        
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        return BytesIO(buffer.getvalue().encode())
    
    def create_attention_interpolator(self, dataset, num_heads=4, d_model=32):
        """Create attention-based interpolator model"""
        
        input_dim = dataset['X_params'].shape[1]
        output_dim = 3  # hydrostatic, magnitude, von mises
        
        self.attention_interpolator = StressAttentionInterpolator(
            input_dim=input_dim,
            num_heads=num_heads,
            d_model=d_model,
            output_dim=output_dim
        )
        
        return self.attention_interpolator
    
    def train_interpolator(self, dataset, epochs=100, lr=0.001, batch_size=32):
        """Train the attention interpolator"""
        if self.attention_interpolator is None:
            self.create_attention_interpolator(dataset)
        
        # Prepare data
        X = torch.FloatTensor(dataset['X_params'])
        Y = torch.FloatTensor(dataset['Y_stress'])
        
        # Split into training and validation
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        split_idx = int(0.8 * n_samples)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_val, Y_val = X[val_indices], Y[val_indices]
        
        # Training
        optimizer = torch.optim.Adam(self.attention_interpolator.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        self.attention_interpolator.train()
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_Y = Y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                predicted, _ = self.attention_interpolator(batch_X.unsqueeze(1), 
                                                          batch_Y.unsqueeze(1), 
                                                          batch_X)
                loss = criterion(predicted, batch_Y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.attention_interpolator.eval()
            with torch.no_grad():
                val_predicted, _ = self.attention_interpolator(X_val.unsqueeze(1), 
                                                              Y_val.unsqueeze(1), 
                                                              X_val)
                val_loss = criterion(val_predicted, Y_val).item()
            
            train_losses.append(train_loss / (len(X_train) // batch_size))
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
        
        return train_losses, val_losses

# =============================================
# ATTENTION-BASED STRESS INTERPOLATOR
# =============================================
class StressAttentionInterpolator(nn.Module):
    """Transformer-inspired attention mechanism for stress field interpolation"""
    
    def __init__(self, input_dim=15, num_heads=4, d_model=32, output_dim=3):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Parameter embedding
        self.param_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, output_dim * N * N),
            nn.Tanh()
        )
        
    def forward(self, source_params, source_stress, target_params):
        """
        Args:
            source_params: (batch_size, num_sources, input_dim)
            source_stress: (batch_size, num_sources, output_dim, N, N)
            target_params: (batch_size, input_dim)
            
        Returns:
            predicted_stress: (batch_size, output_dim, N, N)
            attention_weights: (batch_size, num_sources)
        """
        batch_size = source_params.shape[0]
        
        # Embed parameters
        source_embeddings = self.param_embedding(source_params)
        target_embeddings = self.param_embedding(target_params).unsqueeze(1)
        
        # Compute attention weights
        attn_output, attn_weights = self.attention(
            query=target_embeddings,
            key=source_embeddings,
            value=source_embeddings
        )
        
        # Apply FFN
        ff_output = self.ffn(attn_output)
        
        # Project to stress space
        stress_flat = self.output_projection(ff_output)
        predicted_stress = stress_flat.view(batch_size, self.output_dim, N, N)
        
        # Apply attention-weighted combination from source stresses
        attn_weights = attn_weights.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weighted_source_stress = (source_stress * attn_weights).sum(dim=1)
        
        # Blend with predicted stress
        alpha = 0.7  # Weight for attention-based interpolation
        final_stress = alpha * predicted_stress + (1 - alpha) * weighted_source_stress
        
        return final_stress, attn_weights.squeeze(1)

# =============================================
# MATERIAL & GRID (KEEP ORIGINAL)
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
# INITIALIZE DATASET GENERATOR
# =============================================
if 'dataset_generator' not in st.session_state:
    st.session_state.dataset_generator = PhaseFieldDatasetGenerator()

# =============================================
# ADD DATASET GENERATION SIDEBAR
# =============================================
st.sidebar.header("🤖 ML Dataset Generation")

with st.sidebar.expander("📊 Generate Training Dataset", expanded=False):
    # Dataset generation settings
    num_simulations = st.slider("Number of simulations", 5, 100, 20)
    include_evolution = st.checkbox("Include evolution frames", False)
    
    # Parameter space options
    st.subheader("Parameter Space")
    vary_defect = st.checkbox("Vary defect type", True)
    vary_shape = st.checkbox("Vary seed shape", True)
    vary_eps = st.checkbox("Vary eigenstrain", True)
    vary_kappa = st.checkbox("Vary interface energy", True)
    vary_orientation = st.checkbox("Vary orientation", True)
    
    # Generate dataset button
    if st.button("🔬 Generate Dataset", type="primary"):
        with st.spinner(f"Generating {num_simulations} simulations..."):
            # Generate parameter space
            parameter_space = st.session_state.dataset_generator.generate_parameter_space(
                num_simulations
            )
            
            # Run simulations
            simulations = {}
            progress_bar = st.progress(0)
            
            for i, params in enumerate(parameter_space):
                # Run simulation
                history = run_simulation(params)
                
                # Generate ID
                sim_id = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
                
                # Store simulation
                simulations[sim_id] = {
                    'params': params,
                    'history': history,
                    'created_at': datetime.now().isoformat()
                }
                
                # Update progress
                progress = (i + 1) / len(parameter_space)
                progress_bar.progress(progress)
            
            # Create ML-ready dataset
            dataset = st.session_state.dataset_generator.create_ml_ready_dataset(
                simulations, include_evolution
            )
            
            # Store in session state
            st.session_state.dataset = dataset
            st.session_state.dataset_simulations = simulations
            
            st.success(f"""
            ✅ Dataset Generated!
            - **Samples**: {len(dataset['X_params'])}
            - **Parameters**: {dataset['X_params'].shape[1]}
            - **Grid Size**: {N}×{N}
            - **Stress Components**: {len(dataset['stress_names'])}
            """)

with st.sidebar.expander("📤 Export Dataset", expanded=False):
    if 'dataset' in st.session_state:
        export_format = st.selectbox(
            "Export Format",
            ["HDF5 (.h5)", "NumPy (.npz)", "PyTorch (.pt)", "Pickle (.pkl)", "CSV (.csv)"]
        )
        
        include_grid = st.checkbox("Include grid information", True)
        compress_data = st.checkbox("Compress data", True)
        
        if st.button("💾 Export Dataset"):
            format_key = export_format.split(' ')[1][1:-1]
            buffer = st.session_state.dataset_generator.export_dataset(
                st.session_state.dataset, format_key
            )
            
            # Download button
            filename = f"phase_field_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_key}"
            mime_type = {
                'h5': 'application/x-hdf5',
                'npz': 'application/x-npz',
                'pt': 'application/octet-stream',
                'pkl': 'application/octet-stream',
                'csv': 'text/csv'
            }[format_key]
            
            st.sidebar.download_button(
                label=f"Download {export_format}",
                data=buffer.getvalue(),
                file_name=filename,
                mime=mime_type
            )

with st.sidebar.expander("🧠 Train Attention Model", expanded=False):
    if 'dataset' in st.session_state:
        st.subheader("Attention Model Training")
        
        num_heads = st.slider("Number of attention heads", 1, 8, 4)
        d_model = st.slider("Model dimension", 16, 128, 32)
        epochs = st.slider("Training epochs", 10, 500, 100)
        learning_rate = st.number_input("Learning rate", 1e-4, 1e-1, 1e-3, format="%.4f")
        
        if st.button("🚀 Train Attention Model", type="primary"):
            with st.spinner("Training attention model..."):
                # Create interpolator
                interpolator = st.session_state.dataset_generator.create_attention_interpolator(
                    st.session_state.dataset, num_heads, d_model
                )
                
                # Train model
                train_losses, val_losses = st.session_state.dataset_generator.train_interpolator(
                    st.session_state.dataset, epochs, learning_rate
                )
                
                # Store results
                st.session_state.training_results = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'interpolator': interpolator
                }
                
                st.success("Model training complete!")

# =============================================
# ADD DATASET VISUALIZATION TO MAIN CONTENT
# =============================================
if 'dataset' in st.session_state:
    st.header("📊 Generated Dataset Analysis")
    
    dataset = st.session_state.dataset
    
    # Display dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(dataset['X_params']))
    with col2:
        st.metric("Parameter Dimension", dataset['X_params'].shape[1])
    with col3:
        st.metric("Grid Size", f"{N}×{N}")
    with col4:
        st.metric("Stress Components", len(dataset['stress_names']))
    
    # Display parameter distribution
    with st.expander("📈 Parameter Distributions", expanded=True):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.set_constrained_layout(True)
        
        # Extract specific parameters
        eps_values = dataset['X_params'][:, 0] * (3.0 - 0.3) + 0.3  # Denormalize
        kappa_values = dataset['X_params'][:, 1] * (2.0 - 0.1) + 0.1
        
        # Plot distributions
        axes[0, 0].hist(eps_values, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel("Eigenstrain ε*")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Eigenstrain Distribution")
        
        axes[0, 1].hist(kappa_values, bins=20, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel("Interface energy κ")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Interface Energy Distribution")
        
        # Defect type distribution
        defect_counts = dataset['X_params'][:, 3:6].argmax(axis=1)
        unique_defects, defect_counts = np.unique(defect_counts, return_counts=True)
        defect_labels = ['ISF', 'ESF', 'Twin']
        axes[1, 0].bar(defect_labels, defect_counts, alpha=0.7, color=['steelblue', 'coral', 'green'])
        axes[1, 0].set_xlabel("Defect Type")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Defect Type Distribution")
        
        # Orientation distribution
        orient_counts = dataset['X_params'][:, 11:15].argmax(axis=1)
        unique_orients, orient_counts = np.unique(orient_counts, return_counts=True)
        orient_labels = ['0°', '30°', '60°', '90°']
        axes[1, 1].bar(orient_labels, orient_counts, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel("Orientation")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Orientation Distribution")
        
        # Stress statistics
        hydro_mean = dataset['Y_stress'][:, 0].mean()
        hydro_std = dataset['Y_stress'][:, 0].std()
        axes[2, 0].hist(dataset['Y_stress'][:, 0].flatten(), bins=50, alpha=0.7, color='red')
        axes[2, 0].set_xlabel("Hydrostatic Stress (GPa)")
        axes[2, 0].set_ylabel("Frequency")
        axes[2, 0].set_title(f"Hydrostatic Stress\nMean: {hydro_mean:.2f} ± {hydro_std:.2f} GPa")
        
        vm_mean = dataset['Y_stress'][:, 2].mean()
        vm_std = dataset['Y_stress'][:, 2].std()
        axes[2, 1].hist(dataset['Y_stress'][:, 2].flatten(), bins=50, alpha=0.7, color='green')
        axes[2, 1].set_xlabel("Von Mises Stress (GPa)")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].set_title(f"Von Mises Stress\nMean: {vm_mean:.2f} ± {vm_std:.2f} GPa")
        
        st.pyplot(fig)
    
    # Display sample data
    with st.expander("👁️ Sample Data Visualization", expanded=True):
        sample_idx = st.slider("Sample Index", 0, len(dataset['X_params'])-1, 0)
        
        # Get sample data
        sample_params = dataset['X_params'][sample_idx]
        sample_stress = dataset['Y_stress'][sample_idx]
        sample_defect = dataset['Y_defect'][sample_idx]
        sample_meta = dataset['metadata'][sample_idx]
        
        # Decode parameters
        eps_denorm = sample_params[0] * (3.0 - 0.3) + 0.3
        kappa_denorm = sample_params[1] * (2.0 - 0.1) + 0.1
        defect_idx = sample_params[3:6].argmax()
        defect_type = ['ISF', 'ESF', 'Twin'][defect_idx]
        orient_idx = sample_params[11:15].argmax()
        orientation = ['0°', '30°', '60°', '90°'][orient_idx]
        
        # Display sample info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Defect Type", defect_type)
            st.metric("ε*", f"{eps_denorm:.3f}")
        with col2:
            st.metric("Orientation", f"{orientation}")
            st.metric("κ", f"{kappa_denorm:.3f}")
        
        # Visualize stress fields
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        titles = ['Hydrostatic Stress', 'Stress Magnitude', 'Von Mises Stress', 
                  'Defect Field', 'Stress Correlation', 'Radial Profile']
        
        for i in range(3):
            im = axes[0, i].imshow(sample_stress[i], extent=extent, cmap='coolwarm',
                                  origin='lower', aspect='equal')
            axes[0, i].set_title(titles[i])
            axes[0, i].set_xlabel("x (nm)")
            axes[0, i].set_ylabel("y (nm)")
            plt.colorbar(im, ax=axes[0, i], shrink=0.8)
        
        # Defect field
        im = axes[1, 0].imshow(sample_defect, extent=extent, cmap='viridis',
                              origin='lower', aspect='equal')
        axes[1, 0].set_title(titles[3])
        axes[1, 0].set_xlabel("x (nm)")
        axes[1, 0].set_ylabel("y (nm)")
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
        
        # Stress correlation
        axes[1, 1].scatter(sample_stress[0].flatten(), sample_stress[2].flatten(),
                          alpha=0.1, s=1, color='blue')
        axes[1, 1].set_xlabel("Hydrostatic Stress (GPa)")
        axes[1, 1].set_ylabel("Von Mises Stress (GPa)")
        axes[1, 1].set_title(titles[4])
        
        # Radial profile
        center_x, center_y = N//2, N//2
        radius = np.sqrt((X - X[center_x, center_y])**2 + (Y - Y[center_x, center_y])**2)
        radial_bins = np.linspace(0, np.max(radius), 20)
        radial_hydro = []
        radial_vm = []
        
        for r_min, r_max in zip(radial_bins[:-1], radial_bins[1:]):
            mask = (radius >= r_min) & (radius < r_max)
            if np.any(mask):
                radial_hydro.append(sample_stress[0][mask].mean())
                radial_vm.append(sample_stress[2][mask].mean())
        
        axes[1, 2].plot(radial_bins[1:], radial_hydro, label='Hydrostatic', linewidth=2)
        axes[1, 2].plot(radial_bins[1:], radial_vm, label='Von Mises', linewidth=2)
        axes[1, 2].set_xlabel("Radius (nm)")
        axes[1, 2].set_ylabel("Stress (GPa)")
        axes[1, 2].set_title(titles[5])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Training results visualization
    if 'training_results' in st.session_state:
        with st.expander("📈 Training Results", expanded=True):
            results = st.session_state.training_results
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss curves
            axes[0].plot(results['train_losses'], label='Training Loss', linewidth=2)
            axes[0].plot(results['val_losses'], label='Validation Loss', linewidth=2)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("MSE Loss")
            axes[0].set_title("Training Progress")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Final loss values
            final_train_loss = results['train_losses'][-1]
            final_val_loss = results['val_losses'][-1]
            
            axes[1].bar(['Training', 'Validation'], [final_train_loss, final_val_loss],
                       alpha=0.7, color=['steelblue', 'coral'])
            axes[1].set_ylabel("Final MSE Loss")
            axes[1].set_title(f"Final Loss: Train={final_train_loss:.6f}, Val={final_val_loss:.6f}")
            
            st.pyplot(fig)

# =============================================
# ORIGINAL CODE CONTINUES WITH MINIMAL MODIFICATIONS
# =============================================
# Keep all your original classes and functions here...
# EnhancedLineProfiler, JournalTemplates, FigureStyler, etc.

# =============================================
# MODIFY SIMULATION FUNCTIONS TO COLLECT DATA
# =============================================
# These functions should already exist in your original code
# I'll add data collection capabilities

def create_initial_eta(shape, defect_type):
    """Create initial defect configuration"""
    # Set initial amplitude based on defect type
    amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
    init_amplitude = amplitudes[defect_type]
    
    eta = np.zeros((N, N))
    cx, cy = N//2, N//2
    w, h = (24, 12) if shape in ["Rectangle", "Horizontal Fault"] else (16, 16)
    
    if shape == "Square":
        eta[cy-h:cy+h, cx-h:cx+h] = init_amplitude
    elif shape == "Horizontal Fault":
        eta[cy-4:cy+4, cx-w:cx+w] = init_amplitude
    elif shape == "Vertical Fault":
        eta[cy-w:cy+w, cx-4:cx+4] = init_amplitude
    elif shape == "Rectangle":
        eta[cy-h:cy+h, cx-w:cx+w] = init_amplitude
    elif shape == "Ellipse":
        mask = ((X/(w*1.5))**2 + (Y/(h*1.5))**2) <= 1
        eta[mask] = init_amplitude
    
    eta += 0.02 * np.random.randn(N, N)
    return np.clip(eta, 0.0, 1.0)

@jit(nopython=True, parallel=True)
def evolve_phase_field(eta, kappa, dt, dx, N):
    """Phase field evolution with Allen-Cahn equation"""
    eta_new = eta.copy()
    dx2 = dx * dx
    for i in prange(1, N-1):
        for j in prange(1, N-1):
            lap = (eta[i+1,j] + eta[i-1,j] + eta[i,j+1] + eta[i,j-1] - 4*eta[i,j]) / dx2
            dF = 2*eta[i,j]*(1-eta[i,j])*(eta[i,j]-0.5)
            eta_new[i,j] = eta[i,j] + dt * (-dF + kappa * lap)
            eta_new[i,j] = np.maximum(0.0, np.minimum(1.0, eta_new[i,j]))
    eta_new[0,:] = eta_new[-2,:]; eta_new[-1,:] = eta_new[1,:]
    eta_new[:,0] = eta_new[:,-2]; eta_new[:,-1] = eta_new[:,1]
    return eta_new

@st.cache_data
def compute_stress_fields(eta, eps0, theta):
    """FFT-based stress solver with rotated eigenstrain"""
    # Plane-strain reduced constants (Pa)
    C11_p = (C11 - C12**2 / C11) * 1e9
    C12_p = (C12 - C12**2 / C11) * 1e9
    C44_p = C44 * 1e9
    
    # Wavevectors
    kx = np.fft.fftfreq(N, d=dx)
    ky = np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(2 * np.pi * kx, 2 * np.pi * ky)
    K2 = KX**2 + KY**2
    K2[0, 0] = 1e-12
    mask = K2 > 0
    
    n1 = np.zeros_like(KX)
    n2 = np.zeros_like(KX)
    n1[mask] = KX[mask] / np.sqrt(K2[mask])
    n2[mask] = KY[mask] / np.sqrt(K2[mask])
    
    # Acoustic tensor components
    A11 = np.zeros_like(KX)
    A22 = np.zeros_like(KX)
    A12 = np.zeros_like(KX)
    A11[mask] = C11_p * n1[mask]**2 + C44_p * n2[mask]**2
    A22[mask] = C11_p * n2[mask]**2 + C44_p * n1[mask]**2
    A12[mask] = (C12_p + C44_p) * n1[mask] * n2[mask]
    
    det = A11 * A22 - A12**2
    G11 = np.zeros_like(KX)
    G22 = np.zeros_like(KX)
    G12 = np.zeros_like(KX)
    G11[mask] = A22[mask] / det[mask]
    G22[mask] = A11[mask] / det[mask]
    G12[mask] = -A12[mask] / det[mask]
    
    # Eigenstrain (rotated)
    gamma = eps0
    ct, st = np.cos(theta), np.sin(theta)
    n = np.array([ct, st])
    s = np.array([-st, ct])
    delta = 0.02  # Small dilatation
    eps_local = delta * np.outer(n, n) + gamma * (np.outer(n, s) + np.outer(s, n)) / 2
    R = np.array([[ct, -st], [st, ct]])
    eps_star = R @ eps_local @ R.T
    
    eps_xx_star = eps_star[0,0] * eta
    eps_yy_star = eps_star[1,1] * eta
    eps_xy_star = eps_star[0,1] * eta
    
    # Polarization stress tau = C : eps*
    tau_xx = C11_p * eps_xx_star + C12_p * eps_yy_star
    tau_yy = C12_p * eps_xx_star + C11_p * eps_yy_star
    tau_xy = 2 * C44_p * eps_xy_star
    
    tau_hat_xx = np.fft.fft2(tau_xx)
    tau_hat_yy = np.fft.fft2(tau_yy)
    tau_hat_xy = np.fft.fft2(tau_xy)
    
    S_hat_x = KX * tau_hat_xx + KY * tau_hat_xy
    S_hat_y = KX * tau_hat_xy + KY * tau_hat_yy
    
    u_hat_x = np.zeros_like(KX, dtype=complex)
    u_hat_y = np.zeros_like(KX, dtype=complex)
    u_hat_x[mask] = -1j * (G11[mask] * S_hat_x[mask] + G12[mask] * S_hat_y[mask])
    u_hat_y[mask] = -1j * (G12[mask] * S_hat_x[mask] + G22[mask] * S_hat_y[mask])
    
    u_hat_x[0, 0] = 0
    u_hat_y[0, 0] = 0
    
    # Displacements
    ux = np.real(np.fft.ifft2(u_hat_x))
    uy = np.real(np.fft.ifft2(u_hat_y))
    
    # Elastic strains
    exx = np.real(np.fft.ifft2(1j * KX * u_hat_x))
    eyy = np.real(np.fft.ifft2(1j * KY * u_hat_y))
    exy = 0.5 * np.real(np.fft.ifft2(1j * (KX * u_hat_y + KY * u_hat_x)))
    
    # Elastic stresses (Pa → GPa)
    sxx = (C11_p * (exx - eps_xx_star) + C12_p * (eyy - eps_yy_star)) / 1e9
    syy = (C12_p * (exx - eps_xx_star) + C11_p * (eyy - eps_yy_star)) / 1e9
    sxy = 2 * C44_p * (exy - eps_xy_star) / 1e9
    szz = (C12 / (C11 + C12)) * (sxx + syy)  # Plane strain approximation
    
    # Derived quantities (GPa)
    sigma_mag = np.sqrt(sxx**2 + syy**2 + 2*sxy**2)
    sigma_hydro = (sxx + syy) / 2
    von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
    
    return {
        'sxx': sxx, 'syy': syy, 'sxy': sxy, 'szz': szz,
        'sigma_mag': sigma_mag, 'sigma_hydro': sigma_hydro, 'von_mises': von_mises
    }

def run_simulation(sim_params):
    """Run a complete simulation with given parameters"""
    # Create initial defect
    eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
    
    # Run evolution
    history = []
    for step in range(sim_params['steps'] + 1):
        if step > 0:
            eta = evolve_phase_field(eta, sim_params['kappa'], dt=0.004, dx=dx, N=N)
        if step % sim_params['save_every'] == 0 or step == sim_params['steps']:
            stress_fields = compute_stress_fields(eta, sim_params['eps0'], sim_params['theta'])
            history.append((eta.copy(), stress_fields))
    
    return history

# =============================================
# ENHANCED PUBLICATION-QUALITY PLOTTING FUNCTIONS
# =============================================
def create_publication_heatmaps(simulations, frames, config, style_params):
    """Publication-quality heatmap comparison"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    n_sims = len(simulations)
    cols = min(3, n_sims)
    rows = (n_sims + cols - 1) // cols
    
    # Create figure with journal sizing
    journal_styles = JournalTemplates.get_journal_styles()
    journal = style_params.get('journal_style', 'nature')
    fig_width = journal_styles[journal]['figure_width_double'] / 2.54  # Convert cm to inches
    
    fig, axes = plt.subplots(rows, cols, 
                            figsize=(fig_width, fig_width * 0.8 * rows/cols),
                            constrained_layout=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    
    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get data
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Apply smoothing for better visualization
        if style_params.get('apply_smoothing', True):
            stress_data = gaussian_filter(stress_data, sigma=1)
        
        # Choose colormap
        cmap_name = sim['params']['sigma_cmap']
        if cmap_name in enhanced_cmaps:
            cmap = enhanced_cmaps[cmap_name]
        else:
            cmap = plt.cm.get_cmap(COLORMAPS.get(cmap_name, 'viridis'))
        
        # Create heatmap with enhanced settings
        im = ax.imshow(stress_data, extent=extent, cmap=cmap, 
                      origin='lower', aspect='equal')  # Fixed: aspect='equal' for proper scaling
        
        # Add contour lines for defect boundary
        contour = ax.contour(X, Y, eta, levels=[0.5], colors='white', 
                           linewidths=1, linestyles='--', alpha=0.8)
        
        # Add scale bar
        PublicationEnhancer.add_scale_bar(ax, 5.0, location='lower right')
        
        # Enhanced title
        title = f"{sim['params']['defect_type']}"
        if sim['params']['orientation'] != "Horizontal {111} (0°)":
            title += f"\n{sim['params']['orientation'].split(' ')[0]}"
        
        ax.set_title(title, fontsize=style_params.get('title_font_size', 10),
                    fontweight='semibold', pad=10)
        
        # Axis labels only on edge plots
        if row == rows - 1:
            ax.set_xlabel("x (nm)", fontsize=style_params.get('label_font_size', 9))
        if col == 0:
            ax.set_ylabel("y (nm)", fontsize=style_params.get('label_font_size', 9))
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label(f"{config['stress_component']} (GPa)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    # Hide empty subplots
    for idx in range(n_sims, rows*cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
    
    return fig

def create_enhanced_line_profiles(simulations, frames, config, style_params):
    """Enhanced line profile comparison with multiple directions"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    # Get profile configuration
    profile_direction = config.get('profile_direction', 'Horizontal')
    selected_profiles = config.get('selected_profiles', ['Horizontal'])
    position_ratio = config.get('position_ratio', 0.5)
    custom_angle = config.get('custom_angle', 45.0)
    
    # Create figure layout based on number of profiles
    if profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        n_profiles = len(selected_profiles)
        fig = plt.figure(figsize=(16, 12))
        fig.set_constrained_layout(True)
        
        # Create subplot grid: 2 columns for profiles, 1 for location map
        gs = fig.add_gridspec(3, 3)
        
        # Main profile plot (spanning 2 rows, 2 columns)
        ax_profiles = fig.add_subplot(gs[0:2, 0:2])
        
        # Statistical plot
        ax_stats = fig.add_subplot(gs[0, 2])
        
        # Location map
        ax_location = fig.add_subplot(gs[1, 2])
        
        # Individual profile plots
        ax_individual = fig.add_subplot(gs[2, :])
        
        axes = [ax_profiles, ax_stats, ax_location, ax_individual]
        
    else:
        # Single profile mode
        fig = plt.figure(figsize=(14, 10))
        fig.set_constrained_layout(True)
        
        gs = fig.add_gridspec(2, 3)
        ax_profiles = fig.add_subplot(gs[0, 0:2])
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_location = fig.add_subplot(gs[1, 0:2])
        ax_individual = fig.add_subplot(gs[1, 2])
        
        axes = [ax_profiles, ax_stats, ax_location, ax_individual]
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Prepare data storage
    all_profiles = {profile_type: [] for profile_type in selected_profiles}
    
    # Define colors for different profile types
    profile_colors = {
        'Horizontal': 'red',
        'Vertical': 'blue',
        'Diagonal': 'green',
        'Anti-Diagonal': 'purple',
        'Custom': 'orange'
    }
    
    # Extract and plot profiles
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Extract profiles for all selected types
        for profile_type in selected_profiles:
            # Handle custom angle
            if profile_type == "Custom Angle":
                profile_type_key = "custom"
                angle = custom_angle
            else:
                profile_type_key = profile_type.lower().replace(" ", "_").replace("-", "")
                angle = custom_angle
            
            # Extract profile
            if profile_type_key == "custom":
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    stress_data, 'custom', position_ratio, angle
                )
            else:
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    stress_data, profile_type_key, position_ratio, angle
                )
            
            # Store for statistics
            all_profiles[profile_type].append({
                'distance': distance,
                'profile': profile,
                'endpoints': endpoints,
                'color': color,
                'label': f"{sim['params']['defect_type']}"
            })
            
            # Plot on main axes if single profile mode
            if profile_direction != "Multiple Profiles" or len(selected_profiles) == 1:
                line_style = config.get('line_style', 'solid')
                ax_profiles.plot(distance, profile, color=color,
                               linewidth=style_params.get('line_width', 1.5),
                               linestyle=line_style,
                               label=f"{sim['params']['defect_type']}",
                               alpha=0.8)
    
    # Enhanced axis labeling for main profile plot
    if profile_direction != "Multiple Profiles" or len(selected_profiles) == 1:
        ax_profiles.set_xlabel("Position (nm)", fontsize=style_params.get('label_font_size', 10))
        ax_profiles.set_ylabel(f"{config['stress_component']} (GPa)", 
                              fontsize=style_params.get('label_font_size', 10))
        
        profile_name = selected_profiles[0]
        if profile_name == "Custom Angle":
            profile_name = f"Custom ({custom_angle:.0f}°)"
        ax_profiles.set_title(f"{profile_name} Stress Profile", 
                             fontsize=style_params.get('title_font_size', 12),
                             fontweight='bold')
        
        # Add legend
        PublicationEnhancer.create_fancy_legend(ax_profiles, *ax_profiles.get_legend_handles_labels(),
                                              loc='upper right', frameon=True,
                                              fancybox=True, shadow=False)
    
    # Multiple profiles mode
    elif profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        # Plot all profiles for each simulation
        line_styles = ['-', '--', '-.', ':']
        
        for idx, profile_type in enumerate(selected_profiles):
            profile_data = all_profiles[profile_type]
            
            for sim_idx, data in enumerate(profile_data):
                # Use different line styles for different profile types
                linestyle = line_styles[idx % len(line_styles)]
                
                ax_profiles.plot(data['distance'], data['profile'],
                               color=data['color'],
                               linewidth=style_params.get('line_width', 1.5),
                               linestyle=linestyle,
                               alpha=0.7,
                               label=f"{data['label']} - {profile_type}" if sim_idx == 0 else "")
        
        ax_profiles.set_xlabel("Position (nm)", fontsize=style_params.get('label_font_size', 10))
        ax_profiles.set_ylabel(f"{config['stress_component']} (GPa)", 
                              fontsize=style_params.get('label_font_size', 10))
        ax_profiles.set_title("Multiple Stress Profiles", 
                             fontsize=style_params.get('title_font_size', 12),
                             fontweight='bold')
        
        # Simplify legend for multiple profiles
        handles, labels = ax_profiles.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        ax_profiles.legend(unique_handles, unique_labels, fontsize=8, 
                          loc='upper right', frameon=True)
    
    # Panel B: Statistical summary
    if all_profiles:
        # Calculate statistics for each simulation
        stats_data = []
        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
            eta, stress_fields = sim['history'][frame]
            stress_data = stress_fields[stress_key]
            
            # Calculate basic statistics
            stats_data.append({
                'Defect': sim['params']['defect_type'],
                'Max': float(np.nanmax(stress_data)),
                'Mean': float(np.nanmean(stress_data)),
                'Std': float(np.nanstd(stress_data)),
                'color': colors[idx]
            })
        
        # Create bar plot
        defect_names = [stats['Defect'] for stats in stats_data]
        max_stresses = [stats['Max'] for stats in stats_data]
        colors_list = [stats['color'] for stats in stats_data]
        
        x_pos = np.arange(len(defect_names))
        bars = ax_stats.bar(x_pos, max_stresses, color=colors_list, alpha=0.7)
        
        ax_stats.set_xticks(x_pos)
        ax_stats.set_xticklabels(defect_names, rotation=45, ha='right')
        ax_stats.set_ylabel("Maximum Stress (GPa)", fontsize=9)
        ax_stats.set_title("Peak Stress Comparison", 
                          fontsize=10, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, max_stresses):
            ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Panel C: Show profile locations
    if simulations and selected_profiles:
        sim = simulations[0]
        eta, _ = sim['history'][frames[0]]
        
        # Prepare profile config for plotting
        profile_configs = {}
        for profile_type in selected_profiles:
            if profile_type == "Custom Angle":
                profile_type_key = "custom"
                angle = custom_angle
            else:
                profile_type_key = profile_type.lower().replace(" ", "_").replace("-", "")
                angle = custom_angle
            
            # Extract profile for visualization
            if profile_type_key == "custom":
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    eta, 'custom', position_ratio, angle
                )
            else:
                distance, profile, endpoints = EnhancedLineProfiler.extract_profile(
                    eta, profile_type_key, position_ratio, angle
                )
            
            profile_configs[profile_type] = {
                'profiles': {profile_type: {'endpoints': endpoints}}
            }
        
        # Plot with profile locations
        im, ax_location = EnhancedLineProfiler.plot_profile_locations(
            ax_location, eta, profile_configs, 
            cmap=enhanced_cmaps['plasma_enhanced'], alpha=0.7
        )
        ax_location.set_title("Profile Locations", 
                             fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_location, shrink=0.8)
        cbar.set_label('Defect Parameter η', fontsize=9)
    
    # Panel D: Individual profile comparison (for multiple profiles mode)
    if profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        # Plot each profile type in separate subplot
        n_cols = min(4, len(selected_profiles))
        n_rows = (len(selected_profiles) + n_cols - 1) // n_cols
        
        # Clear the individual axis and create subplots
        ax_individual.clear()
        fig.delaxes(ax_individual)
        
        # Create subplots for individual profiles
        for idx, profile_type in enumerate(selected_profiles):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            profile_data = all_profiles[profile_type]
            
            for sim_idx, data in enumerate(profile_data):
                ax.plot(data['distance'], data['profile'],
                       color=data['color'],
                       linewidth=style_params.get('line_width', 1.0),
                       alpha=0.7,
                       label=data['label'] if sim_idx == 0 else "")
            
            ax.set_title(f"{profile_type} Profile", fontsize=9)
            ax.set_xlabel("Position (nm)", fontsize=8)
            ax.set_ylabel("Stress (GPa)", fontsize=8)
            
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
    
    # Apply publication styling
    if profile_direction == "Multiple Profiles" and len(selected_profiles) > 1:
        # Get all axes for styling
        all_axes = [ax_profiles, ax_stats, ax_location]
        for i in range(len(selected_profiles)):
            all_axes.append(fig.axes[i + 3])  # Individual profile axes
        
        fig = EnhancedFigureStyler.apply_publication_styling(fig, all_axes, style_params)
        
        # Add panel labels
        for ax, label in zip([ax_profiles, ax_stats, ax_location], ['A', 'B', 'C']):
            ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='top')
    else:
        fig = EnhancedFigureStyler.apply_publication_styling(fig, axes, style_params)
        
        # Add panel labels
        for ax, label in zip([ax_profiles, ax_stats, ax_location, ax_individual], ['A', 'B', 'C', 'D']):
            if ax is not None:
                ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='top')
    
    return fig

def create_publication_statistics(simulations, frames, config, style_params):
    """Publication-quality statistical analysis"""
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(14, 10))
    fig.set_constrained_layout(True)
    
    # Define subplots
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2)  # Box plot
    ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)  # Violin plot
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=2)  # Histogram
    ax4 = plt.subplot2grid((3, 4), (1, 2), colspan=2)  # Cumulative distribution
    ax5 = plt.subplot2grid((3, 4), (2, 0), colspan=4)  # Statistical table
    
    # Get colors
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Collect data
    all_data = []
    labels = []
    
    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key].flatten()
        stress_data = stress_data[np.isfinite(stress_data)]
        
        all_data.append(stress_data)
        labels.append(f"{sim['params']['defect_type']}\n({sim['params']['orientation'][:10]}...)")
    
    # Panel 1: Enhanced box plot
    bp = ax1.boxplot(all_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False,
                    meanprops=dict(color='white', linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='gray', linewidth=1),
                    capprops=dict(color='gray', linewidth=1),
                    boxprops=dict(linewidth=1))
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title(f"Distribution of {config['stress_component']}", 
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel("Stress (GPa)", fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    
    # Add mean values as text
    for i, data in enumerate(all_data):
        mean_val = np.mean(data)
        ax1.text(i + 1, mean_val, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Panel 2: Violin plot
    parts = ax2.violinplot(all_data, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    ax2.set_title("Probability Density", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Stress (GPa)", fontsize=10)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels([sim['params']['defect_type'] for sim in simulations])
    
    # Panel 3: Histogram with KDE
    ax3.hist(all_data, bins=30, density=True, stacked=True, 
            label=[sim['params']['defect_type'] for sim in simulations],
            color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Add KDE
    for data, color, label in zip(all_data, colors, labels):
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data.min() for data in all_data), 
                             max(data.max() for data in all_data), 100)
        ax3.plot(x_range, kde(x_range), color=color, linewidth=2, label=label.split('\n')[0])
    
    ax3.set_title("Histogram with KDE", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Stress (GPa)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=8)
    
    # Panel 4: Cumulative distribution
    for data, color, label in zip(all_data, colors, labels):
        sorted_data = np.sort(data)
        y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, y_vals, color=color, linewidth=2, label=label.split('\n')[0])
    
    ax4.set_title("Cumulative Distribution", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Stress (GPa)", fontsize=10)
    ax4.set_ylabel("Cumulative Probability", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 5: Statistical table
    ax5.axis('off')
    
    # Create comprehensive statistics table
    table_data = []
    columns = ['Defect', 'N', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max', 'Skew', 'Kurtosis']
    
    for idx, (data, sim) in enumerate(zip(all_data, simulations)):
        table_data.append([
            sim['params']['defect_type'],
            len(data),
            f"{np.mean(data):.3f}",
            f"{np.std(data):.3f}",
            f"{np.min(data):.3f}",
            f"{np.percentile(data, 25):.3f}",
            f"{np.median(data):.3f}",
            f"{np.percentile(data, 75):.3f}",
            f"{np.max(data):.3f}",
            f"{stats.skew(data):.3f}",
            f"{stats.kurtosis(data):.3f}"
        ])
    
    # Create table
    table = ax5.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#f2f2f2']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code cells
    for i in range(len(table_data)):
        for j in range(1, len(columns)):  # Skip first column (Defect)
            table[(i+1, j)].set_facecolor(mpl.colors.to_rgba(colors[i], 0.3))  # Add alpha
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5], style_params)
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4], ['A', 'B', 'C', 'D']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
    
    return fig

def create_publication_correlation(simulations, frames, config, style_params):
    """Publication-quality correlation analysis"""
    # Component mapping
    component_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises',
        "Defect Parameter η": 'eta'
    }
    
    x_key = component_map[config.get('correlation_x', 'Defect Parameter η')]
    y_key = component_map[config.get('correlation_y', 'Stress Magnitude |σ|')]
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(15, 12))
    fig.set_constrained_layout(True)
    
    # Define subplot grid
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # Scatter with regression
    ax2 = plt.subplot2grid((3, 3), (0, 2))              # Correlation coefficients
    ax3 = plt.subplot2grid((3, 3), (1, 0))              # Residuals
    ax4 = plt.subplot2grid((3, 3), (1, 1))              # QQ plot
    ax5 = plt.subplot2grid((3, 3), (1, 2))              # Histogram of residuals
    ax6 = plt.subplot2grid((3, 3), (2, 0), colspan=3)   # Regression parameters
    
    # Get enhanced colormaps
    enhanced_cmaps = PublicationEnhancer.create_custom_colormaps()
    colors = enhanced_cmaps['defect_categorical'].colors[:len(simulations)]
    
    # Store regression results
    regression_results = []
    
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
        
        # Prepare x data
        if x_key == 'eta':
            x_data = eta.flatten()
        else:
            x_data = stress_fields[x_key].flatten()
        
        # Prepare y data
        if y_key == 'eta':
            y_data = eta.flatten()
        else:
            y_data = stress_fields[y_key].flatten()
        
        # Sample data for clarity
        sample_size = min(5000, len(x_data))
        indices = np.random.choice(len(x_data), sample_size, replace=False)
        x_sampled = x_data[indices]
        y_sampled = y_data[indices]
        
        # Remove outliers
        q_low, q_high = np.percentile(x_sampled, [1, 99])
        mask = (x_sampled > q_low) & (x_sampled < q_high)
        x_sampled = x_sampled[mask]
        y_sampled = y_sampled[mask]
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_sampled, y_sampled)
        
        # Store results
        regression_results.append({
            'defect': sim['params']['defect_type'],
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'n': len(x_sampled)
        })
        
        # Panel 1: Scatter with regression line
        scatter = ax1.scatter(x_sampled, y_sampled, color=color, alpha=0.3,
                            s=10, edgecolors='none', label=sim['params']['defect_type'])
        
        # Add regression line
        x_range = np.linspace(np.min(x_sampled), np.max(x_sampled), 100)
        y_pred = slope * x_range + intercept
        ax1.plot(x_range, y_pred, color=color, linewidth=2, alpha=0.8,
                label=f"R = {r_value:.3f}")
        
        # Panel 3: Residuals
        y_pred_points = slope * x_sampled + intercept
        residuals = y_sampled - y_pred_points
        
        ax3.scatter(y_pred_points, residuals, color=color, alpha=0.3, s=10)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 4: QQ plot
        if idx == 0:  # Plot QQ for first simulation
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.get_lines()[0].set_marker('.')
            ax4.get_lines()[0].set_markersize(5)
            ax4.get_lines()[0].set_alpha(0.5)
            ax4.get_lines()[1].set_color('red')
            ax4.get_lines()[1].set_linewidth(2)
        
        # Panel 5: Histogram of residuals
        ax5.hist(residuals, bins=30, density=True, alpha=0.5, color=color)
    
    # Enhance Panel 1
    ax1.set_xlabel(config.get('correlation_x', 'X Component'), fontsize=11)
    ax1.set_ylabel(config.get('correlation_y', 'Y Component'), fontsize=11)
    ax1.set_title(f"Scatter Plot with Linear Regression", fontsize=12, fontweight='bold')
    
    # Create enhanced legend
    PublicationEnhancer.create_fancy_legend(ax1, *ax1.get_legend_handles_labels(),
                                          loc='upper left', frameon=True,
                                          fancybox=True, shadow=True, ncol=2)
    
    # Panel 2: Correlation coefficients
    defect_names = [sim['params']['defect_type'] for sim in simulations]
    r_values = [result['r_value'] for result in regression_results]
    
    bars = ax2.bar(range(len(defect_names)), r_values, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(defect_names)))
    ax2.set_xticklabels(defect_names, rotation=45, ha='right')
    ax2.set_ylabel("Correlation Coefficient (R)", fontsize=10)
    ax2.set_title("Correlation Strength", fontsize=11, fontweight='bold')
    ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, r_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Enhance Panel 3: Residuals
    ax3.set_xlabel("Predicted Values", fontsize=10)
    ax3.set_ylabel("Residuals", fontsize=10)
    ax3.set_title("Residual Plot", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Enhance Panel 4: QQ Plot
    ax4.set_title("Q-Q Plot of Residuals", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax4.set_ylabel("Sample Quantiles", fontsize=10)
    
    # Enhance Panel 5: Histogram of residuals
    ax5.set_title("Distribution of Residuals", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Residuals", fontsize=10)
    ax5.set_ylabel("Density", fontsize=10)
    ax5.legend([sim['params']['defect_type'] for sim in simulations], fontsize=8)
    
    # Panel 6: Regression parameters table
    ax6.axis('off')
    
    # Create detailed table
    table_data = []
    columns = ['Defect', 'Slope', 'Intercept', 'R', 'R²', 'p-value', 'Std Error', 'N']
    
    for result in regression_results:
        table_data.append([
            result['defect'],
            f"{result['slope']:.4f}",
            f"{result['intercept']:.4f}",
            f"{result['r_value']:.4f}",
            f"{result['r_squared']:.4f}",
            f"{result['p_value']:.3e}",
            f"{result['std_err']:.4f}",
            f"{result['n']:,}"
        ])
    
    table = ax6.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#f2f2f2']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code p-values
    for i in range(len(table_data)):
        p_val = float(table_data[i][5].replace('e-', 'E-'))
        if p_val < 0.001:
            table[(i+1, 5)].set_text_props(fontweight='bold', color='green')
        elif p_val < 0.01:
            table[(i+1, 5)].set_text_props(fontweight='bold', color='orange')
    
    # Apply publication styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, [ax1, ax2, ax3, ax4, ax5, ax6], style_params)
    
    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3, ax4, ax5], ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
               fontsize=14, fontweight='bold', va='top')
    
    return fig

def create_enhanced_comparison_plot(simulations, frames, config, style_params):
    """Create publication-quality comparison plots"""
    
    # Create figure based on comparison type
    if config['type'] == "Side-by-Side Heatmaps":
        return create_publication_heatmaps(simulations, frames, config, style_params)
    elif config['type'] == "Overlay Line Profiles":
        return create_enhanced_line_profiles(simulations, frames, config, style_params)
    elif config['type'] == "Statistical Summary":
        return create_publication_statistics(simulations, frames, config, style_params)
    elif config['type'] == "Defect-Stress Correlation":
        return create_publication_correlation(simulations, frames, config, style_params)
    else:
        # Fall back to simpler visualization for other types
        return create_simple_comparison_plot(simulations, frames, config, style_params)

def create_simple_comparison_plot(simulations, frames, config, style_params):
    """Simple comparison plot for unsupported types"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_constrained_layout(True)
    
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
    
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Simple line plot of mean stress
        mean_stress = np.mean(stress_data)
        ax.bar(idx, mean_stress, color=color, alpha=0.7, 
               label=f"{sim['params']['defect_type']}")
    
    ax.set_xlabel("Simulation", fontsize=style_params.get('label_font_size', 12))
    ax.set_ylabel(f"Mean {config['stress_component']} (GPa)", 
                  fontsize=style_params.get('label_font_size', 12))
    ax.set_title(f"{config['type']} Comparison", 
                 fontsize=style_params.get('title_font_size', 14),
                 fontweight='bold')
    ax.legend(fontsize=style_params.get('legend_fontsize', 10))
    
    # Apply styling
    fig = EnhancedFigureStyler.apply_publication_styling(fig, ax, style_params)
    
    return fig

# =============================================
# ORIGINAL COMPARISON PLOTTING FUNCTIONS (for backward compatibility)
# =============================================
def create_defect_stress_correlation_plot(simulations, frames, config, style_params):
    """Create defect-stress correlation plot for multiple simulations"""
    return create_publication_correlation(simulations, frames, config, style_params)

def create_stress_cross_correlation_plot(simulations, frames, config, style_params):
    """Create stress component cross-correlation plot"""
    st.subheader("📈 Stress Component Cross-Correlation")
    
    # Component mapping
    component_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    
    x_key = component_map[config.get('correlation_x', 'Stress Magnitude |σ|')]
    y_key = component_map[config.get('correlation_y', 'von Mises σ_vM')]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
    
    for idx, (sim, frame, color) in enumerate(zip(simulations, frames, colors)):
        # Get data
        eta, stress_fields = sim['history'][frame]
        
        x_data = stress_fields[x_key].flatten()
        y_data = stress_fields[y_key].flatten()
        
        # Sample data
        sample_size = int(len(x_data) * config.get('correlation_sample', 20) / 100)
        if sample_size < len(x_data):
            indices = np.random.choice(len(x_data), sample_size, replace=False)
            x_sampled = x_data[indices]
            y_sampled = y_data[indices]
        else:
            x_sampled = x_data
            y_sampled = y_data
        
        # Scatter plot
        axes[0].scatter(x_sampled, y_sampled, 
                       color=color, 
                       alpha=config.get('correlation_alpha', 0.5),
                       s=config.get('correlation_point_size', 10),
                       label=f"{sim['params']['defect_type']} - {sim['params']['orientation']}")
        
        # Calculate correlation
        mask = np.isfinite(x_sampled) & np.isfinite(y_sampled)
        if np.sum(mask) > 10:
            corr = np.corrcoef(x_sampled[mask], y_sampled[mask])[0, 1]
            # Add to legend
            axes[0].plot([], [], ' ', label=f"R = {corr:.3f}")
    
    axes[0].set_xlabel(config.get('correlation_x', 'Stress Magnitude |σ|'), 
                      fontsize=style_params.get('label_font_size', 14))
    axes[0].set_ylabel(config.get('correlation_y', 'von Mises σ_vM'), 
                      fontsize=style_params.get('label_font_size', 14))
    axes[0].set_title(f"{config.get('correlation_x')} vs {config.get('correlation_y')}", 
                     fontsize=style_params.get('title_font_size', 16),
                     fontweight=style_params.get('title_weight', 'bold'))
    axes[0].legend(fontsize=style_params.get('legend_fontsize', 12))
    
    # Create correlation matrix
    if len(simulations) > 1:
        components = ['sigma_mag', 'sigma_hydro', 'von_mises']
        component_names = ['|σ|', 'σ_h', 'σ_vM']
        
        # Prepare correlation matrix
        corr_matrix = np.zeros((3, 3))
        
        for i, comp_i in enumerate(components):
            for j, comp_j in enumerate(components):
                # Average correlation across simulations
                corrs = []
                for sim, frame in zip(simulations, frames):
                    eta, stress_fields = sim['history'][frame]
                    data_i = stress_fields[comp_i].flatten()
                    data_j = stress_fields[comp_j].flatten()
                    mask = np.isfinite(data_i) & np.isfinite(data_j)
                    if np.sum(mask) > 10:
                        corr = np.corrcoef(data_i[mask], data_j[mask])[0, 1]
                        corrs.append(corr)
                
                if corrs:
                    corr_matrix[i, j] = np.mean(corrs)
        
        # Plot correlation matrix
        im = axes[1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = axes[1].text(j, i, f'{corr_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="white",
                                   fontsize=style_params.get('label_font_size', 14),
                                   fontweight='bold')
        
        axes[1].set_title("Stress Component Correlation Matrix", 
                         fontsize=style_params.get('title_font_size', 16),
                         fontweight=style_params.get('title_weight', 'bold'))
        axes[1].set_xticks(range(3))
        axes[1].set_yticks(range(3))
        axes[1].set_xticklabels(component_names)
        axes[1].set_yticklabels(component_names)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], shrink=0.8)
    
    # Apply styling
    fig = EnhancedFigureStyler.apply_advanced_styling(fig, axes, style_params)
    
    return fig

def create_evolution_timeline_plot(simulations, config, style_params):
    """Create evolution timeline comparison plot"""
    st.subheader("⏱️ Evolution Timeline Comparison")
    
    # Get evolution metrics
    evolution_data = {}
    
    for sim in simulations:
        history = sim['history']
        params = sim['params']
        
        # Calculate evolution metrics
        eta_evolution = []
        stress_evolution = []
        
        stress_map = {
            "Stress Magnitude |σ|": 'sigma_mag',
            "Hydrostatic σ_h": 'sigma_hydro',
            "von Mises σ_vM": 'von_mises'
        }
        stress_key = stress_map[config['stress_component']]
        
        for frame, (eta, stress_fields) in enumerate(history):
            eta_evolution.append(np.mean(eta))
            stress_evolution.append(np.mean(stress_fields[stress_key]))
        
        evolution_data[sim['id']] = {
            'defect_type': params['defect_type'],
            'orientation': params['orientation'],
            'eta': eta_evolution,
            'stress': stress_evolution,
            'frames': len(history)
        }
    
    # Create evolution plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.set_constrained_layout(True)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
    
    # Plot 1: η evolution
    ax1 = axes[0, 0]
    for idx, (sim_id, data) in enumerate(evolution_data.items()):
        frames = range(data['frames'])
        ax1.plot(frames, data['eta'], 
                color=colors[idx], 
                linewidth=style_params.get('line_width', 2.0),
                linestyle=config.get('line_style', 'solid'),
                label=f"{data['defect_type']} - {data['orientation']}")
    
    ax1.set_xlabel("Frame Number", fontsize=style_params.get('label_font_size', 14))
    ax1.set_ylabel("Average η", fontsize=style_params.get('label_font_size', 14))
    ax1.set_title("Defect Evolution (η)", 
                  fontsize=style_params.get('title_font_size', 16),
                  fontweight=style_params.get('title_weight', 'bold'))
    ax1.legend(fontsize=style_params.get('legend_fontsize', 12))
    
    # Plot 2: Stress evolution
    ax2 = axes[0, 1]
    for idx, (sim_id, data) in enumerate(evolution_data.items()):
        frames = range(data['frames'])
        ax2.plot(frames, data['stress'], 
                color=colors[idx], 
                linewidth=style_params.get('line_width', 2.0),
                linestyle=config.get('line_style', 'solid'),
                label=f"{data['defect_type']} - {data['orientation']}")
    
    ax2.set_xlabel("Frame Number", fontsize=style_params.get('label_font_size', 14))
    ax2.set_ylabel(f"Average {config['stress_component']} (GPa)", 
                  fontsize=style_params.get('label_font_size', 14))
    ax2.set_title(f"Stress Evolution ({config['stress_component']})", 
                  fontsize=style_params.get('title_font_size', 16),
                  fontweight=style_params.get('title_weight', 'bold'))
    ax2.legend(fontsize=style_params.get('legend_fontsize', 12))
    
    # Plot 3: Correlation between η and stress evolution
    ax3 = axes[1, 0]
    for idx, (sim_id, data) in enumerate(evolution_data.items()):
        # Calculate moving correlation
        eta_array = np.array(data['eta'])
        stress_array = np.array(data['stress'])
        
        window_size = min(10, len(eta_array))
        if window_size > 3:
            correlations = []
            for i in range(len(eta_array) - window_size + 1):
                window_eta = eta_array[i:i+window_size]
                window_stress = stress_array[i:i+window_size]
                corr = np.corrcoef(window_eta, window_stress)[0, 1]
                correlations.append(corr)
            
            frames = range(len(correlations))
            ax3.plot(frames, correlations, 
                    color=colors[idx], 
                    linewidth=style_params.get('line_width', 2.0),
                    label=f"{data['defect_type']} - {data['orientation']}")
    
    ax3.set_xlabel("Frame Window", fontsize=style_params.get('label_font_size', 14))
    ax3.set_ylabel("Moving Correlation (η vs Stress)", 
                  fontsize=style_params.get('label_font_size', 14))
    ax3.set_title("Evolution Correlation", 
                  fontsize=style_params.get('title_font_size', 16),
                  fontweight=style_params.get('title_weight', 'bold'))
    ax3.legend(fontsize=style_params.get('legend_fontsize', 12))
    
    # Plot 4: Evolution rate
    ax4 = axes[1, 1]
    for idx, (sim_id, data) in enumerate(evolution_data.items()):
        eta_array = np.array(data['eta'])
        stress_array = np.array(data['stress'])
        
        # Calculate rates of change
        eta_rate = np.diff(eta_array)
        stress_rate = np.diff(stress_array)
        frames = range(1, len(eta_array))
        ax4.scatter(frames, eta_rate, 
                   color=colors[idx], 
                   alpha=0.6, s=20,
                   label=f"{data['defect_type']} - η rate")
        
        frames = range(1, len(stress_array))
        ax4.scatter(frames, stress_rate, 
                   color=colors[idx], 
                   alpha=0.6, s=20,
                   marker='s',
                   label=f"{data['defect_type']} - stress rate")
    
    ax4.set_xlabel("Frame Number", fontsize=style_params.get('label_font_size', 14))
    ax4.set_ylabel("Rate of Change", fontsize=style_params.get('label_font_size', 14))
    ax4.set_title("Evolution Rates", 
                  fontsize=style_params.get('title_font_size', 16),
                  fontweight=style_params.get('title_weight', 'bold'))
    
    # Apply styling
    fig = EnhancedFigureStyler.apply_advanced_styling(fig, axes, style_params)
    
    return fig

def create_contour_comparison_plot(simulations, frames, config, style_params):
    """Create contour comparison plot"""
    st.subheader("🌀 Contour Level Comparison")
    
    stress_map = {
        "Stress Magnitude |σ|": 'sigma_mag',
        "Hydrostatic σ_h": 'sigma_hydro',
        "von Mises σ_vM": 'von_mises'
    }
    stress_key = stress_map[config['stress_component']]
    
    n_sims = len(simulations)
    cols = min(2, n_sims)
    rows = (n_sims + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), constrained_layout=True)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(simulations)))
    
    for idx, (sim, frame) in enumerate(zip(simulations, frames)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get data
        eta, stress_fields = sim['history'][frame]
        stress_data = stress_fields[stress_key]
        
        # Create contour plot
        levels = config.get('contour_levels', 10)
        contour = ax.contour(X, Y, stress_data, 
                            levels=levels,
                            linewidths=config.get('contour_linewidth', 1.5),
                            cmap=plt.cm.get_cmap(COLORMAPS.get(sim['params']['sigma_cmap'], 'viridis')))
        
        # Add contour labels
        ax.clabel(contour, inline=True, fontsize=style_params.get('tick_font_size', 12))
        
        # Add defect contour
        eta_contour = ax.contour(X, Y, eta, levels=[0.5], 
                                colors='black', linewidths=2, linestyles='--')
        
        ax.set_title(f"{sim['params']['defect_type']} - {sim['params']['orientation']}", 
                    fontsize=style_params.get('title_font_size', 16),
                    fontweight=style_params.get('title_weight', 'bold'))
        ax.set_xlabel("x (nm)", fontsize=style_params.get('label_font_size', 14))
        ax.set_ylabel("y (nm)", fontsize=style_params.get('label_font_size', 14))
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, shrink=0.8)
    
    # Hide empty subplots
    for idx in range(n_sims, rows*cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Apply styling
    fig = EnhancedFigureStyler.apply_advanced_styling(fig, axes, style_params)
    
    return fig

# =============================================
# MAIN CONTENT AREA
# =============================================
if operation_mode == "Run New Simulation":
    # Show simulation preview
    st.header("🎯 New Simulation Preview")
    
    if 'sim_params' in st.session_state:
        sim_params = st.session_state.sim_params
        
        # Display simulation parameters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Defect Type", sim_params['defect_type'])
        with col2:
            st.metric("ε*", f"{sim_params['eps0']:.3f}")
        with col3:
            st.metric("κ", f"{sim_params['kappa']:.2f}")
        with col4:
            st.metric("Orientation", sim_params['orientation'])
        
        # Show initial configuration
        init_eta = create_initial_eta(sim_params['shape'], sim_params['defect_type'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Apply styling
        fig = EnhancedFigureStyler.apply_advanced_styling(fig, [ax1, ax2], advanced_styling)
        
        # Initial defect
        im1 = ax1.imshow(init_eta, extent=extent, 
                        cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['eta_cmap'], 'viridis')), 
                        origin='lower', aspect='equal')  # Fixed: aspect='equal'
        ax1.set_title(f"Initial {sim_params['defect_type']} - {sim_params['shape']}")
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        plt.colorbar(im1, ax=ax1, shrink=advanced_styling.get('colorbar_shrink', 0.8))
        
        # Stress preview (calculated from initial state)
        stress_preview = compute_stress_fields(init_eta, sim_params['eps0'], sim_params['theta'])
        im2 = ax2.imshow(stress_preview['sigma_mag'], extent=extent, 
                        cmap=plt.cm.get_cmap(COLORMAPS.get(sim_params['sigma_cmap'], 'hot')), 
                        origin='lower', aspect='equal')  # Fixed: aspect='equal'
        ax2.set_title(f"Initial Stress Magnitude")
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        plt.colorbar(im2, ax=ax2, shrink=advanced_styling.get('colorbar_shrink', 0.8))
        
        st.pyplot(fig)
        
        # Run simulation button
        if st.button("▶️ Start Full Simulation", type="primary"):
            with st.spinner(f"Running {sim_params['defect_type']} simulation..."):
                start_time = time.time()
                
                # Run simulation
                history = run_simulation(sim_params)
                
                # Create metadata
                metadata = {
                    'run_time': time.time() - start_time,
                    'frames': len(history),
                    'grid_size': N,
                    'dx': dx,
                    'colormaps': {
                        'eta': sim_params['eta_cmap'],
                        'sigma': sim_params['sigma_cmap'],
                        'hydro': sim_params['hydro_cmap'],
                        'vm': sim_params['vm_cmap']
                    }
                }
                
                # Save to database
                sim_id = SimulationDB.save_simulation(sim_params, history, metadata)
                
                st.success(f"""
                ✅ Simulation Complete!
                - **ID**: `{sim_id}`
                - **Frames**: {len(history)}
                - **Time**: {metadata['run_time']:.1f} seconds
                - **Saved to database**
                """)
                
                # Show final frame with post-processing options
                with st.expander("📊 Post-Process Final Results", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        show_defect = st.checkbox("Show Defect Field", True)
                        show_stress = st.checkbox("Show Stress Field", True)
                    with col2:
                        custom_cmap = st.selectbox("Custom Colormap", cmap_list, 
                                                  index=cmap_list.index('viridis'))
                    
                    if show_defect or show_stress:
                        final_eta, final_stress = history[-1]
                        
                        n_plots = (1 if show_defect else 0) + (1 if show_stress else 0)
                        fig2, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
                        
                        if n_plots == 1:
                            axes = [axes]
                        
                        plot_idx = 0
                        if show_defect:
                            im = axes[plot_idx].imshow(final_eta, extent=extent, 
                                                      cmap=plt.cm.get_cmap(COLORMAPS.get(custom_cmap, 'viridis')), 
                                                      origin='lower', aspect='equal')  # Fixed
                            axes[plot_idx].set_title(f"Final {sim_params['defect_type']}")
                            axes[plot_idx].set_xlabel("x (nm)")
                            axes[plot_idx].set_ylabel("y (nm)")
                            plt.colorbar(im, ax=axes[plot_idx], shrink=0.8)
                            plot_idx += 1
                        
                        if show_stress:
                            im = axes[plot_idx].imshow(final_stress['sigma_mag'], extent=extent,
                                                      cmap=plt.cm.get_cmap(COLORMAPS.get(custom_cmap, 'viridis')), 
                                                      origin='lower', aspect='equal')  # Fixed
                            axes[plot_idx].set_title(f"Final Stress Magnitude")
                            axes[plot_idx].set_xlabel("x (nm)")
                            axes[plot_idx].set_ylabel("y (nm)")
                            plt.colorbar(im, ax=axes[plot_idx], shrink=0.8)
                        
                        # Apply advanced styling
                        fig2 = EnhancedFigureStyler.apply_advanced_styling(fig2, axes, advanced_styling)
                        st.pyplot(fig2)
                
                # Clear the run flag
                if 'run_new_simulation' in st.session_state:
                    del st.session_state.run_new_simulation
    
    else:
        st.info("Configure simulation parameters in the sidebar and click 'Run & Save Simulation'")
    
    # Show saved simulations
    st.header("📋 Saved Simulations")
    simulations = SimulationDB.get_simulation_list()
    
    if simulations:
        # Create a dataframe of saved simulations
        sim_data = []
        for sim in simulations:
            params = sim['params']
            sim_data.append({
                'ID': sim['id'],
                'Defect Type': params['defect_type'],
                'Orientation': params['orientation'],
                'ε*': params['eps0'],
                'κ': params['kappa'],
                'Shape': params['shape'],
                'Steps': params['steps'],
                'Frames': len(SimulationDB.get_simulation(sim['id'])['history'])
            })
        
        df = pd.DataFrame(sim_data)
        st.dataframe(df, use_container_width=True)
        
        # Delete option
        with st.expander("🗑️ Delete Simulations"):
            delete_options = [f"{sim['name']} (ID: {sim['id']})" for sim in simulations]
            to_delete = st.multiselect("Select simulations to delete", delete_options)
            
            if st.button("Delete Selected", type="secondary"):
                for sim_name in to_delete:
                    # Extract ID from string
                    sim_id = sim_name.split("ID: ")[1].replace(")", "")
                    if SimulationDB.delete_simulation(sim_id):
                        st.success(f"Deleted simulation {sim_id}")
                st.rerun()
    else:
        st.info("No simulations saved yet. Run a simulation to see it here!")

else:  # COMPARE SAVED SIMULATIONS
    st.header("🔬 Multi-Simulation Comparison")
    
    if 'run_comparison' in st.session_state and st.session_state.run_comparison:
        config = st.session_state.comparison_config
        
        # Load selected simulations
        simulations = []
        valid_sim_ids = []
        
        for sim_id in config['sim_ids']:
            sim_data = SimulationDB.get_simulation(sim_id)
            if sim_data:
                simulations.append(sim_data)
                valid_sim_ids.append(sim_id)
            else:
                st.warning(f"Simulation {sim_id} not found!")
        
        if not simulations:
            st.error("No valid simulations selected for comparison!")
        else:
            st.success(f"Loaded {len(simulations)} simulations for comparison")
            
            # Determine frame index
            frame_idx = config['frame_idx']
            if config['frame_selection'] == "Final Frame":
                # Use final frame for each simulation
                frames = [len(sim['history']) - 1 for sim in simulations]
            elif config['frame_selection'] == "Same Evolution Time":
                # Use same evolution time (percentage of total steps)
                target_percentage = 0.8  # 80% of evolution
                frames = [int(len(sim['history']) * target_percentage) for sim in simulations]
            else:
                # Specific frame index
                frames = [min(frame_idx, len(sim['history']) - 1) for sim in simulations]
            
            # Get stress component mapping
            stress_map = {
                "Stress Magnitude |σ|": 'sigma_mag',
                "Hydrostatic σ_h": 'sigma_hydro',
                "von Mises σ_vM": 'von_mises'
            }
            stress_key = stress_map[config['stress_component']]
            
            # Create comparison based on type
            if config['type'] in ["Side-by-Side Heatmaps", "Overlay Line Profiles", 
                                 "Statistical Summary", "Defect-Stress Correlation"]:
                # Use enhanced publication-quality plotting
                st.subheader(f"📰 Publication-Quality {config['type']}")
                
                # Create enhanced plot
                fig = create_enhanced_comparison_plot(simulations, frames, config, advanced_styling)
                
                # Display with enhanced options
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.pyplot(fig)
                
                with col2:
                    # Quick export info
                    st.info(f"""
                    **Publication Ready:**
                    - Journal: {advanced_styling.get('journal_style', 'custom').title()}
                    - DPI: {advanced_styling.get('figure_dpi', 600)}
                    - Vector: {'Yes' if advanced_styling.get('vector_output', True) else 'No'}
                    """)
                
                with col3:
                    # Show figure info
                    fig_size = fig.get_size_inches()
                    st.metric("Figure Size", f"{fig_size[0]:.1f} × {fig_size[1]:.1f} in")
                    st.metric("Resolution", f"{advanced_styling.get('figure_dpi', 600)} DPI")
                
                # Additional statistics for certain plot types
                if config['type'] in ["Statistical Summary", "Defect-Stress Correlation"]:
                    with st.expander("📊 Detailed Statistics", expanded=False):
                        # Generate detailed statistics
                        stats_data = []
                        for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                            eta, stress_fields = sim['history'][frame]
                            stress_data = stress_fields[stress_key].flatten()
                            stress_data = stress_data[np.isfinite(stress_data)]
                            
                            stats_data.append({
                                'Simulation': f"{sim['params']['defect_type']} - {sim['params']['orientation']}",
                                'N': len(stress_data),
                                'Max (GPa)': float(np.nanmax(stress_data)),
                                'Mean (GPa)': float(np.nanmean(stress_data)),
                                'Median (GPa)': float(np.nanmedian(stress_data)),
                                'Std Dev': float(np.nanstd(stress_data)),
                                'Skewness': float(stats.skew(stress_data)),
                                'Kurtosis': float(stats.kurtosis(stress_data))
                            })
                        
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats.style.format({
                            'Max (GPa)': '{:.3f}',
                            'Mean (GPa)': '{:.3f}',
                            'Median (GPa)': '{:.3f}',
                            'Std Dev': '{:.3f}',
                            'Skewness': '{:.3f}',
                            'Kurtosis': '{:.3f}'
                        }), use_container_width=True)
            
            elif config['type'] == "Overlay Line Profiles":
                # This is now handled by create_enhanced_line_profiles
                pass
            
            # Handle other comparison types
            elif config['type'] == "Stress Component Cross-Correlation":
                fig = create_stress_cross_correlation_plot(simulations, frames, config, advanced_styling)
                st.pyplot(fig)
            
            elif config['type'] == "Evolution Timeline":
                fig = create_evolution_timeline_plot(simulations, config, advanced_styling)
                st.pyplot(fig)
            
            elif config['type'] == "Contour Comparison":
                fig = create_contour_comparison_plot(simulations, frames, config, advanced_styling)
                st.pyplot(fig)
            
            # 3D Surface Comparison (simplified 2D version)
            elif config['type'] == "3D Surface Comparison":
                st.subheader("🗻 3D Surface Comparison (2D Projection)")
                
                # Create 2D surface plots
                n_sims = len(simulations)
                cols = min(2, n_sims)
                rows = (n_sims + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), constrained_layout=True)
                
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for idx, (sim, frame) in enumerate(zip(simulations, frames)):
                    row = idx // cols
                    col = idx % cols
                    ax = axes[row, col]
                    
                    # Get data
                    eta, stress_fields = sim['history'][frame]
                    stress_data = stress_fields[stress_key]
                    
                    # Create surface plot (simplified 2D)
                    im = ax.imshow(stress_data, extent=extent, 
                                  cmap=plt.cm.get_cmap(COLORMAPS.get(sim['params']['sigma_cmap'], 'viridis')), 
                                  origin='lower', aspect='equal')  # Fixed
                    
                    ax.set_title(f"{sim['params']['defect_type']} - {sim['params']['orientation']}")
                    ax.set_xlabel("x (nm)")
                    ax.set_ylabel("y (nm)")
                    
                    plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Hide empty subplots
                for idx in range(n_sims, rows*cols):
                    row = idx // cols
                    col = idx % cols
                    axes[row, col].axis('off')
                
                # Apply styling
                fig = EnhancedFigureStyler.apply_advanced_styling(fig, axes, advanced_styling)
                st.pyplot(fig)
            
            # Post-processing options
            with st.expander("🔄 Real-time Post-Processing", expanded=False):
                st.subheader("Live Figure Customization")
                
                col1, col2 = st.columns(2)
                with col1:
                    update_fonts = st.checkbox("Update Font Sizes", True)
                    update_lines = st.checkbox("Update Line Styles", True)
                with col2:
                    update_colors = st.checkbox("Update Colors", True)
                    update_grid = st.checkbox("Update Grid", True)
                
                if st.button("🔄 Refresh with New Styling", type="secondary"):
                    st.rerun()
            
            # Clear comparison flag
            if 'run_comparison' in st.session_state:
                del st.session_state.run_comparison
    
    else:
        st.info("Select simulations in the sidebar and click 'Run Comparison' to start!")
        
        # Show available simulations
        simulations = SimulationDB.get_simulation_list()
        
        if simulations:
            st.subheader("📚 Available Simulations")
            
            # Group by defect type
            defect_groups = {}
            for sim in simulations:
                defect = sim['params']['defect_type']
                if defect not in defect_groups:
                    defect_groups[defect] = []
                defect_groups[defect].append(sim)
            
            for defect_type, sims in defect_groups.items():
                with st.expander(f"{defect_type} ({len(sims)} simulations)"):
                    for sim in sims:
                        params = sim['params']
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.text(f"ID: {sim['id']}")
                        with col2:
                            st.text(f"Orientation: {params['orientation']}")
                        with col3:
                            st.text(f"ε*={params['eps0']:.2f}, κ={params['kappa']:.2f}")
        else:
            st.warning("No simulations available. Run some simulations first!")

# =============================================
# EXPORT FUNCTIONALITY WITH POST-PROCESSING
# =============================================
st.sidebar.header("💾 Export Options")

with st.sidebar.expander("📥 Advanced Export"):
    export_format = st.selectbox(
        "Export Format",
        ["Complete Package (JSON + CSV + PNG)", "JSON Parameters Only", 
         "Publication-Ready Figures", "Raw Data CSV"]
    )
    
    include_styling = st.checkbox("Include Styling Parameters", True)
    high_resolution = st.checkbox("High Resolution Figures", True)
    
    if st.button("📥 Generate Custom Export", type="primary"):
        simulations = SimulationDB.get_all_simulations()
        
        if not simulations:
            st.sidebar.warning("No simulations to export!")
        else:
            with st.spinner("Creating custom export package..."):
                buffer = BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Export each simulation
                    for sim_id, sim_data in simulations.items():
                        sim_dir = f"simulation_{sim_id}"
                        
                        # Export parameters
                        params_json = json.dumps(sim_data['params'], indent=2)
                        zf.writestr(f"{sim_dir}/parameters.json", params_json)
                        
                        # Export metadata
                        metadata_json = json.dumps(sim_data['metadata'], indent=2)
                        zf.writestr(f"{sim_dir}/metadata.json", metadata_json)
                        
                        # Export styling if requested
                        if include_styling:
                            styling_json = json.dumps(advanced_styling, indent=2)
                            zf.writestr(f"{sim_dir}/styling_parameters.json", styling_json)
                        
                        # Export data frames
                        if export_format in ["Complete Package (JSON + CSV + PNG)", "Raw Data CSV"]:
                            for i, (eta, stress_fields) in enumerate(sim_data['history']):
                                df = pd.DataFrame({
                                    'eta': eta.flatten(order='F'),
                                    'sxx': stress_fields['sxx'].flatten(order='F'),
                                    'syy': stress_fields['syy'].flatten(order='F'),
                                    'sxy': stress_fields['sxy'].flatten(order='F'),
                                    'sigma_mag': stress_fields['sigma_mag'].flatten(order='F'),
                                    'sigma_hydro': stress_fields['sigma_hydro'].flatten(order='F'),
                                    'von_mises': stress_fields['von_mises'].flatten(order='F')
                                })
                                zf.writestr(f"{sim_dir}/frame_{i:04d}.csv", df.to_csv(index=False))
                    
                    # Create summary file
                    summary = f"""MULTI-SIMULATION EXPORT SUMMARY
========================================
Generated: {datetime.now().isoformat()}
Total Simulations: {len(simulations)}
Export Format: {export_format}
Includes Styling: {include_styling}
High Resolution: {high_resolution}

STYLING PARAMETERS:
-------------------
{json.dumps(advanced_styling, indent=2)}

SIMULATIONS:
------------
"""
                    for sim_id, sim_data in simulations.items():
                        params = sim_data['params']
                        summary += f"\nSimulation {sim_id}:"
                        summary += f"\n  Defect: {params['defect_type']}"
                        summary += f"\n  Orientation: {params['orientation']}"
                        summary += f"\n  ε*: {params['eps0']}"
                        summary += f"\n  κ: {params['kappa']}"
                        summary += f"\n  Frames: {len(sim_data['history'])}"
                        summary += f"\n  Created: {sim_data['created_at']}\n"
                    
                    zf.writestr("EXPORT_SUMMARY.txt", summary)
                
                buffer.seek(0)
                
                # Determine file name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ag_np_analysis_export_{timestamp}.zip"
                
                st.sidebar.download_button(
                    "📥 Download Export Package",
                    buffer.getvalue(),
                    filename,
                    "application/zip"
                )
                st.sidebar.success("Export package ready!")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("🔬 Theoretical Soundness & Advanced Analysis", expanded=False):
    st.markdown("""
    ### 🎯 **Enhanced Multi-Simulation Comparison Platform**
    
    #### **📊 NEW: Enhanced Line Profile Analysis**
    
    **Multiple Profile Directions:**
    - **Horizontal Profiles**: Traditional stress analysis along x-axis
    - **Vertical Profiles**: Stress variation along y-axis (now properly scaled!)
    - **Diagonal Profiles**: 45° profiles for crystallographic analysis
    - **Anti-Diagonal Profiles**: 135° profiles for complete coverage
    - **Custom Angle Profiles**: Any angle from -180° to 180°
    - **Multiple Profile Mode**: Compare all profiles simultaneously
    
    **FIXED: Aspect Ratio Scaling Issue**
    - **Problem**: Horizontal profiles appeared stretched compared to vertical
    - **Solution**: All plots now use `aspect='equal'` for proper scaling
    - **Result**: Equal physical dimensions in x and y directions
    - **Benefit**: Accurate distance measurements in all directions
    
    **Enhanced Profile Features:**
    - **Position Control**: Adjust profile position relative to center (0-100%)
    - **Proper Distance Calculation**: Correct scaling for diagonal distances (√2 factor)
    - **Bilinear Interpolation**: Smooth profiles for custom angles
    - **Visual Overlay**: Clear indication of profile locations on heatmaps
    - **Color-coded Profiles**: Different colors for different profile types
    
    #### **📈 Scientific Benefits of Enhanced Profiles:**
    
    **Crystallographic Accuracy:**
    - **Diagonal Profiles**: Capture stress along {110} directions
    - **Custom Angles**: Align with specific crystallographic planes
    - **Proper Scaling**: Accurate stress gradients in all directions
    - **Orientation Analysis**: Study anisotropic stress distributions
    
    **Physical Insights:**
    - **Anisotropy Detection**: Compare horizontal vs vertical stress gradients
    - **Symmetry Analysis**: Diagonal vs anti-diagonal comparisons
    - **Habit Plane Effects**: Profile alignment with defect orientation
    - **Stress Concentration**: Identify hotspots in specific directions
    
    #### **🔧 Technical Implementation:**
    
    **EnhancedLineProfiler Class:**
    - **Robust Extraction**: Handles all profile types with proper boundary conditions
    - **Distance Calculation**: Correct physical distances for all orientations
    - **Interpolation**: Smooth profiles with bilinear interpolation
    - **Visualization**: Clear overlay of profile locations
    
    **Aspect Ratio Fix:**
    - **Heatmaps**: `aspect='equal'` for proper scaling
    - **Line Plots**: Correct distance calculations
    - **Consistency**: Uniform scaling across all visualizations
    - **Publication Quality**: Proper aspect ratios for scientific figures
    
    **Multiple Profile Mode:**
    - **Simultaneous Comparison**: View all profiles in single figure
    - **Statistical Summary**: Compare peak stresses across profiles
    - **Visual Correlation**: See how stress varies with direction
    - **Publication Layout**: Multi-panel figures for comprehensive analysis
    
    ### **🎨 Updated Styling & Visualization:**
    
    **Enhanced Color Schemes:**
    - **Profile-specific Colors**: Red (horizontal), Blue (vertical), Green (diagonal)
    - **Clear Legends**: Distinguish between simulation and profile types
    - **Consistent Styling**: Maintain publication-quality throughout
    
    **Improved Layout:**
    - **Multi-panel Figures**: A) Main profiles, B) Statistics, C) Locations, D) Individual profiles
    - **Clear Labeling**: Panel labels (A, B, C, D) for publication
    - **Optimized Spacing**: Balanced layout for clarity
    - **Responsive Design**: Adapts to number of profiles and simulations
    
    #### **📊 Key Physical Insights from Enhanced Analysis:**
    
    **Horizontal vs Vertical Stress:**
    - **Anisotropy Detection**: Different stress magnitudes in x vs y directions
    - **Habit Plane Effects**: Orientation-dependent stress distributions
    - **Defect Shape Influence**: How defect geometry affects stress anisotropy
    
    **Diagonal Stress Analysis:**
    - **Crystallographic Alignment**: Stress along {110} family directions
    - **Shear Stress Components**: Diagonal profiles capture shear components
    - **Symmetry Breaking**: Detect deviations from cubic symmetry
    
    **Custom Angle Profiles:**
    - **Specific Crystallographic Directions**: Align with experimental measurements
    - **Gradient Analysis**: Stress gradients in arbitrary directions
    - **Complete Coverage**: Full 360° stress characterization
    
    ### **🔬 Enhanced Scientific Workflow:**
    
    1. **Run Simulations** with different defect types and orientations
    2. **Select Profile Type** (Horizontal, Vertical, Diagonal, Custom, Multiple)
    3. **Adjust Position** to explore different regions of the defect
    4. **Compare Profiles** across multiple simulations
    5. **Analyze Anisotropy** by comparing different profile directions
    6. **Export Results** for publication with proper scaling
    
    #### **📈 Enhanced Publication Output:**
    
    **Multi-profile Figures:**
    - **Panel A**: Overlay of selected profiles
    - **Panel B**: Statistical comparison of peak stresses
    - **Panel C**: Visual map showing profile locations
    - **Panel D**: Individual profile plots (for multiple profiles mode)
    
    **Proper Scaling:**
    - **Equal Aspect Ratio**: True physical dimensions maintained
    - **Correct Distance Labels**: Accurate nanometer measurements
    - **Consistent Units**: Uniform scaling across all plots
    
    **Enhanced Annotation:**
    - **Profile Type Labels**: Clear indication of each profile
    - **Distance Markers**: Proper scaling for all orientations
    - **Color Coding**: Intuitive color scheme for different profiles
    
    ### **🔬 Platform Capabilities Summary:**
    
    **Enhanced Line Profile Analysis:**
    - **5 Profile Types**: Horizontal, Vertical, Diagonal, Anti-Diagonal, Custom
    - **Position Control**: 0-100% from center
    - **Angle Control**: -180° to 180° for custom profiles
    - **Multiple Mode**: Simultaneous comparison of all profiles
    
    **Fixed Scaling Issues:**
    - **Aspect Ratio Correction**: Proper `aspect='equal'` implementation
    - **Distance Calculation**: Correct scaling for all orientations
    - **Consistent Visualization**: Uniform scaling across all plots
    
    **Scientific Accuracy:**
    - **Crystallographic Alignment**: Profiles along specific crystallographic directions
    - **Physical Distance**: Correct nanometer measurements
    - **Stress Gradients**: Accurate calculation in all directions
    
    **Publication Quality:**
    - **Multi-panel Layout**: Comprehensive analysis in single figure
    - **Professional Styling**: Journal-ready formatting
    - **Clear Visualization**: Intuitive presentation of complex data
    
    **Advanced crystallographic stress analysis platform with enhanced line profile capabilities and proper scaling!**
    """)
    
    # Display platform statistics
    simulations = SimulationDB.get_all_simulations()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulations", len(simulations))
    with col2:
        total_frames = sum([len(sim['history']) for sim in simulations.values()]) if simulations else 0
        st.metric("Total Frames", f"{total_frames:,}")
    with col3:
        st.metric("Profile Types", "5+")
    with col4:
        st.metric("Fixed Scaling", "✓ Aspect Ratio")

#st.caption("🔬 Enhanced Multi-Defect Comparison • Multi-direction Line Profiles • Fixed Scaling • 2025")

# Update the page title to reflect new capabilities
st.title("🔬 Ag Nanoparticle Multi-Defect Analysis & ML Dataset Generator")
st.markdown("""
**Run multiple simulations • Generate ML datasets • Train attention models • Compare results**
**Phase Field → ML-Ready Data → Attention-Based Interpolation**
""")
