import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
import warnings
import glob
from typing import List, Dict, Any, Optional
from scipy.ndimage import zoom

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================
# GLOBAL CONFIGURATION
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# =============================================
# DOMAIN & PHYSICS CLASSES
# =============================================
class DomainConfiguration:
    """Configuration for the 12.8 nm × 12.8 nm simulation domain"""
    N = 128
    dx = 0.1
    DOMAIN_LENGTH = N * dx  # 12.8 nm
    DOMAIN_HALF = DOMAIN_LENGTH / 2.0

    @classmethod
    def get_extent(cls):
        return [-cls.DOMAIN_HALF, cls.DOMAIN_HALF, -cls.DOMAIN_HALF, cls.DOMAIN_HALF]

class PhysicsParameters:
    """Physics parameters with eigenstrain values"""
    EIGENSTRAIN_VALUES = {
        'Twin': 2.12,
        'ISF': 0.289,
        'ESF': 0.333,
        'No Defect': 0.0
    }
    
    @staticmethod
    def get_eigenstrain(defect_type: str) -> float:
        return PhysicsParameters.EIGENSTRAIN_VALUES.get(defect_type, 0.0)

# =============================================
# SPATIAL INTERPOLATOR (Logic Focus)
# =============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)

class TransformerSpatialInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3,
                spatial_sigma=10.0, temperature=1.0, locality_weight_factor=0.5):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        self.locality_weight_factor = locality_weight_factor
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(15, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
    def set_spatial_parameters(self, spatial_sigma=None, locality_weight_factor=None):
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
        if locality_weight_factor is not None:
            self.locality_weight_factor = locality_weight_factor
            
    def compute_angular_bracketing_kernel(self, source_params, target_params):
        spatial_weights = []
        defect_mask = []
        angular_distances = []
        
        target_theta = target_params.get('theta', 0.0)
        target_theta_deg = np.degrees(target_theta) % 360
        target_defect = target_params.get('defect_type', 'Twin')
        
        for src in source_params:
            src_theta = src.get('theta', 0.0)
            src_theta_deg = np.degrees(src_theta) % 360
            
            raw_diff = abs(src_theta_deg - target_theta_deg)
            angular_dist = min(raw_diff, 360 - raw_diff)
            angular_distances.append(angular_dist)
            
            # SPATIAL AND MASK LOGIC
            if src.get('defect_type') == target_defect:
                defect_mask.append(1.0)  # Mask: PASS
            else:
                defect_mask.append(1e-6)  # Mask: BLOCK (near zero)
                
            weight = np.exp(-0.5 * (angular_dist / self.spatial_sigma) ** 2)
            spatial_weights.append(weight)
            
        return np.array(spatial_weights), np.array(defect_mask), np.array(angular_distances)
    
    def encode_parameters(self, params_list, target_angle_deg):
        encoded = []
        for params in params_list:
            features = []
            features.append(params.get('eps0', 0.707) / 3.0)
            features.append(params.get('kappa', 0.6) / 2.0)
            theta = params.get('theta', 0.0)
            features.append(theta / np.pi)
            
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
                
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
                
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)
            features.append(np.exp(-angle_diff / 45.0))
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))
            
            habit_distance = abs(theta_deg - 54.7)
            habit_distance = min(habit_distance, 360 - habit_distance)
            features.append(np.exp(-habit_distance / 15.0))
            
            while len(features) < 15:
                features.append(0.0)
            features = features[:15]
            encoded.append(features)
        return torch.FloatTensor(encoded)
        
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        if not sources:
            return None
            
        try:
            source_params = []
            source_fields = []
            source_indices = []
            
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    continue
                source_params.append(src['params'])
                source_indices.append(i)
                
                history = src['history']
                if history and isinstance(history[-1], dict):
                    last_frame = history[-1]
                    if 'stresses' in last_frame:
                        stress_fields = last_frame['stresses']
                        vm = stress_fields.get('von_mises', self.compute_von_mises(stress_fields))
                        hydro = stress_fields.get('sigma_hydro', self.compute_hydrostatic(stress_fields))
                        mag = stress_fields.get('sigma_mag', np.sqrt(vm**2 + hydro**2))
                        
                        source_fields.append({
                            'von_mises': vm, 'sigma_hydro': hydro, 'sigma_mag': mag,
                            'source_index': i, 'source_params': src['params']
                        })
                        
            if not source_params or not source_fields:
                # Mock data if no valid fields found for demonstration
                st.warning("No valid source fields found. Using mock data for visualization demonstration.")
                for i, src in enumerate(sources):
                    source_fields.append({
                        'von_mises': np.zeros((10,10)), 
                        'source_index': i, 
                        'source_params': src['params']
                    })

            # Ensure uniform shapes
            shapes = [f['von_mises'].shape for f in source_fields]
            if len(set(shapes)) > 1:
                target_shape = shapes[0]
                resized_fields = []
                for fields in source_fields:
                    resized = {}
                    for key, field in fields.items():
                        if isinstance(field, np.ndarray) and field.shape != target_shape:
                            factors = [t/s for t, s in zip(target_shape, field.shape)]
                            resized[key] = zoom(field, factors, order=1)
                        else:
                            resized[key] = field
                    resized_fields.append(resized)
                source_fields = resized_fields
                
            # Compute Weights
            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )
            
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Padding check
            if source_features.shape[1] < 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            # Transformer Pass
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            proj_features = self.input_proj(all_features)
            proj_features = self.pos_encoder(proj_features)
            transformer_output = self.transformer(proj_features)
            
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
            
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            
            # Combine Weights: Spatial * Mask * Attention
            spatial_kernel_tensor = torch.FloatTensor(spatial_kernel).unsqueeze(0)
            defect_mask_tensor = torch.FloatTensor(defect_mask).unsqueeze(0)
            
            biased_scores = attn_scores * spatial_kernel_tensor * defect_mask_tensor
            final_attention_weights = torch.softmax(biased_scores, dim=-1).squeeze().detach().cpu().numpy()
            entropy_final = self._calculate_entropy(final_attention_weights)
            
            sources_data = []
            source_theta_degrees = [np.degrees(src.get('theta', 0.0)) % 360 for src in source_params]
            
            for i, (field, theta_deg, angular_dist, spatial_w, defect_w, combined_w) in enumerate(zip(
                source_fields, source_theta_degrees, angular_distances, 
                spatial_kernel, defect_mask, final_attention_weights
            )):
                attention_weight = attn_scores.squeeze().detach().cpu().numpy()[i] if i < len(attn_scores.squeeze()) else 0.0
                
                sources_data.append({
                    'source_index': i,
                    'theta_deg': theta_deg,
                    'angular_dist': angular_dist,
                    'defect_type': field['source_params'].get('defect_type', 'Unknown'),
                    'spatial_weight': spatial_w,       # Kernel
                    'defect_weight': defect_w,         # Mask
                    'attention_weight': attention_weight,
                    'combined_weight': combined_w,
                    'target_defect_match': field['source_params'].get('defect_type') == target_params['defect_type'],
                    'is_query': False
                })
            
            return {
                'fields': {'dummy': np.zeros((10,10))}, # Minimal placeholder
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist(),
                    'learned_attention': attn_scores.squeeze().detach().cpu().numpy().tolist(),
                    'entropy': entropy_final
                },
                'sources_data': sources_data,
                'target_params': target_params,
                'target_angle': target_angle_deg,
            }
            
        except Exception as e:
            st.error(f"Interpolation Error: {str(e)}")
            return None

    def compute_von_mises(self, stress_fields):
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz', 'tau_xy']):
            sxx = stress_fields['sigma_xx']
            syy = stress_fields['sigma_yy']
            szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
            txy = stress_fields['tau_xy']
            tyz = stress_fields.get('tau_yz', np.zeros_like(sxx))
            tzx = stress_fields.get('tau_zx', np.zeros_like(sxx))
            von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(txy**2 + tyz**2 + tzx**2)))
            return von_mises
        return np.zeros((10, 10))

    def compute_hydrostatic(self, stress_fields):
        if all(k in stress_fields for k in ['sigma_xx', 'sigma_yy', 'sigma_zz']):
            return (stress_fields['sigma_xx'] + stress_fields['sigma_yy'] + stress_fields.get('sigma_zz', np.zeros_like(stress_fields['sigma_xx']))) / 3
        return np.zeros((10, 10))
        
    def _calculate_entropy(self, weights):
        weights = np.array(weights)
        weights = weights[weights > 0]
        if len(weights) == 0:
            return 0.0
        weights = weights / weights.sum()
        return -np.sum(weights * np.log(weights + 1e-10))

# =============================================
# SOLUTION LOADER
# =============================================
class EnhancedSolutionLoader:
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
        self._ensure_directory()
        
    def _ensure_directory(self):
        if not os.path.exists(self.solutions_dir):
            os.makedirs(self.solutions_dir, exist_ok=True)
            
    def scan_solutions(self) -> List[Dict[str, Any]]:
        # For this demonstration, we will generate mock data if directory is empty
        # so the user can see the visualization immediately.
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth']:
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        if not all_files:
            return [] # Return empty, main function will handle mock generation
            
        file_info = []
        for file_path in all_files:
            try:
                info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path))
                }
                file_info.append(info)
            except:
                continue
        return file_info
    
    def load_all_solutions(self):
        solutions = []
        file_info = self.scan_solutions()
        
        if not file_info:
            return solutions
            
        for file_info_item in file_info:
            try:
                with open(file_info_item['path'], 'rb') as f:
                    data = pickle.load(f)
                    # Standardize structure
                    standardized = {
                        'params': data.get('params', {}),
                        'history': data.get('history', [])
                    }
                    solutions.append(standardized)
            except Exception as e:
                continue
        return solutions

# =============================================
# ENHANCED SANKEY VISUALIZER
# =============================================
class WeightVisualizer:
    def __init__(self):
        self.color_scheme = {
            'Twin': '#FF6B6B',
            'ISF': '#4ECDC4',
            'ESF': '#95E1D3',
            'No Defect': '#FFD93D',
            'Query': '#9D4EDD',
            'Spatial': '#36A2EB',
            'Defect': '#FF6384',
            'Attention': '#4BC0C0',
            'Combined': '#9966FF'
        }
        
    def get_colormap(self, cmap_name, n_colors=10):
        """Get color palette from colormap"""
        try:
            cmap = plt.get_cmap(cmap_name)
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
                   for r, g, b, _ in [cmap(i/n_colors) for i in range(n_colors)]]
        except:
            return [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
                   for r, g, b, _ in [plt.get_cmap('viridis')(i/n_colors) for i in range(n_colors)]]

    def create_enhanced_sankey_diagram(
        self, 
        sources_data, 
        target_angle, 
        target_defect, 
        spatial_sigma,
        label_font_size=16,      # User Input
        title_font_size=24,     # User Input
        legend_font_size=14,    # User Input
        node_pad=20,             # User Input
        node_thickness=20,       # User Input
        link_opacity=0.5         # User Input
    ):
        """
        Create enhanced Sankey diagram with user-configurable styles
        """
        # Create nodes
        labels = ['Target']  # Index 0
        
        # Add source nodes
        for source in sources_data:
            angle = source['theta_deg']
            defect = source['defect_type']
            labels.append(f"S{source['source_index']}\n{defect}\n{angle:.1f}°")
        
        # Component nodes (Indices after sources)
        component_start = len(labels)
        labels.extend(['Spatial Kernel', 'Defect Match', 'Attention Score', 'Combined Weight'])
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        link_labels = []
        
        color_palette = self.get_colormap('viridis', len(sources_data) + 4)
        
        # Links from sources to components
        for i, source in enumerate(sources_data):
            source_idx = i + 1 # +1 for Target at index 0
            source_color = color_palette[i % len(color_palette)]
            
            # Helper to adjust opacity in color string
            def adjust_rgba(hex_color, opacity):
                hex_color = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f'rgba({r}, {g}, {b}, {opacity})'

            # Link to spatial kernel
            source_indices.append(source_idx)
            target_indices.append(component_start)
            spatial_value = source['spatial_weight'] * 100
            values.append(spatial_value)
            colors.append(adjust_rgba('#36A2EB', link_opacity))
            link_labels.append(f"Spatial: {source['spatial_weight']:.3f}")
            
            # Link to defect match (MASK VISUALIZATION)
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            defect_value = source['defect_weight'] * 100
            values.append(defect_value)
            colors.append(adjust_rgba('#FF6384', link_opacity))
            link_labels.append(f"Defect Mask: {source['defect_weight']:.3f}")
            
            # Link to attention score
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_value = attention_w * 100
            values.append(attention_value)
            colors.append(adjust_rgba('#4BC0C0', link_opacity))
            link_labels.append(f"Attention: {attention_w:.3f}")
            
            # Link to combined weight
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            combined_value = source['combined_weight'] * 100
            values.append(combined_value)
            colors.append(adjust_rgba('#9966FF', link_opacity))
            link_labels.append(f"Combined: {source['combined_weight']:.3f}")
        
        # Links from components to target
        for comp_idx, comp_name in enumerate(['Spatial', 'Defect', 'Attention', 'Combined']):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)
            
            # Calculate flow out of component into target
            comp_value = 0
            for s_idx, t_idx, v in zip(source_indices, target_indices, values):
                if s_idx == component_start + comp_idx and t_idx != 0:
                    comp_value += v
            
            # Scale down slightly for visual balance
            values.append(comp_value * 0.5) 
            
            comp_colors = ['#36A2EB', '#FF6384', '#4BC0C0', '#9966FF']
            colors.append(adjust_rgba(comp_colors[comp_idx], link_opacity))
            link_labels.append(f"{comp_name} → Target")

        # Create Figure
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=node_pad,                      # Dynamic Input
                thickness=node_thickness,          # Dynamic Input
                line=dict(color="black", width=1),
                label=labels,
                color=["#FF6B6B"] + 
                      [color_palette[i % len(color_palette)] for i in range(len(sources_data))] + 
                      ["#36A2EB", "#FF6384", "#4BC0C0", "#9966FF"],
                customdata = [f"Node: {l}" for l in labels],
                hovertemplate='<b>%{label}</b><extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                customdata=link_labels,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>%{customdata}<extra></extra>'
            )
        )])
        
        # Apply Dynamic Layout Settings
        fig.update_layout(
            title=dict(
                text=f'<b>ENHANCED SANKEY: SPATIAL & MASK ATTENTION FLOW</b><br>Target: {target_angle}° {target_defect} | σ={spatial_sigma}°',
                font=dict(size=title_font_size, color='#2C3E50'), # Dynamic Input
                x=0.5, y=0.95
            ),
            font=dict(
                size=label_font_size, # Dynamic Input
                family='Arial, sans-serif',
                color='#2C3E50'
            ),
            width=1400,
            height=900,
            margin=dict(t=100, l=50, r=50, b=50),
            hoverlabel=dict(
                font=dict(size=legend_font_size), # Dynamic Input
                bgcolor='rgba(44, 62, 80, 0.9)',
                bordercolor='white'
            )
        )

        # Add Annotations for Legend (Also use dynamic font size)
        annotations = [
            dict(
                x=0.02, y=1.05, xref='paper', yref='paper',
                text='<b>COLOR CODING:</b>',
                showarrow=False,
                font=dict(size=legend_font_size + 2, color='darkblue', weight='bold') # Dynamic
            ),
            dict(
                x=0.02, y=1.02, xref='paper', yref='paper',
                text='• <span style="color:#36A2EB">Spatial Kernel</span>',
                showarrow=False,
                font=dict(size=legend_font_size)
            ),
            dict(
                x=0.20, y=1.02, xref='paper', yref='paper',
                text='• <span style="color:#FF6384">Defect Match (Mask)</span>',
                showarrow=False,
                font=dict(size=legend_font_size)
            ),
            dict(
                x=0.45, y=1.02, xref='paper', yref='paper',
                text='• <span style="color:#4BC0C0">Attention Score</span>',
                showarrow=False,
                font=dict(size=legend_font_size)
            ),
            dict(
                x=0.65, y=1.02, xref='paper', yref='paper',
                text='• <span style="color:#9966FF">Combined Weight</span>',
                showarrow=False,
                font=dict(size=legend_font_size)
            )
        ]
        
        fig.update_layout(annotations=annotations)
        
        return fig

# =============================================
# MAIN APP
# =============================================
def main():
    st.set_page_config(
        page_title="Enhanced Sankey Visualization",
        layout="wide",
        page_icon="🌀"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌀 SPATIAL INTERPOLATION & MASK ANALYSIS</h1>', unsafe_allow_html=True)
    
    # Init Session State
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'loader' not in st.session_state:
        st.session_state.loader = EnhancedSolutionLoader()
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = TransformerSpatialInterpolator(spatial_sigma=10.0)
    if 'result' not in st.session_state:
        st.session_state.result = None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # 1. Data Loading / Generation
        st.markdown("#### 📂 Data Source")
        if st.button("Load/Generate Solutions", use_container_width=True):
            with st.spinner("Preparing simulation data..."):
                sols = st.session_state.loader.load_all_solutions()
                if not sols:
                    st.info("No files found. Generating 8 Mock Solutions for visualization.")
                    # Generate Mock Data for demonstration
                    mock_defs = ['Twin', 'Twin', 'ISF', 'ISF', 'ESF', 'ESF', 'Twin', 'No Defect']
                    mock_thetas = [20.0, 50.0, 45.0, 80.0, 10.0, 60.0, 90.0, 30.0]
                    
                    for i in range(8):
                        st.session_state.solutions.append({
                            'params': {
                                'defect_type': mock_defs[i],
                                'theta': np.radians(mock_thetas[i]),
                                'eps0': PhysicsParameters.get_eigenstrain(mock_defs[i])
                            },
                            'history': [{'stresses': {'sigma_xx': np.zeros((10,10))}}]
                        })
                else:
                    st.session_state.solutions = sols
                st.success(f"Ready with {len(st.session_state.solutions)} solutions.")

        st.divider()
        
        # 2. Physics Controls
        st.markdown("#### 🎯 Physics Parameters")
        target_angle = st.slider("Target Angle (°)", 0.0, 180.0, 54.7, 0.1)
        target_defect = st.selectbox("Target Defect", ['Twin', 'ISF', 'ESF', 'No Defect'])
        spatial_sigma = st.slider("Spatial Sigma (°)", 1.0, 45.0, 15.0, 0.5,
                                  help="Width of the angular bracketing kernel.")

        st.divider()
        
        # 3. VISUALIZATION ENHANCEMENT OPTIONS (Requested Feature)
        st.markdown("#### 🎨 Sankey Enhancement")
        label_font = st.slider("Label Font Size", 10, 30, 16)
        title_font = st.slider("Title Font Size", 14, 40, 24)
        legend_font = st.slider("Legend/Info Font Size", 10, 24, 14)
        node_pad = st.slider("Node Padding", 5, 50, 20)
        node_thick = st.slider("Node Thickness", 5, 50, 20)
        link_opac = st.slider("Link Opacity", 0.1, 1.0, 0.5)
        
        st.divider()
        
        # 4. Execution
        if st.button("🚀 Run Interpolation", type="primary", use_container_width=True):
            if not st.session_state.solutions:
                st.warning("Load solutions first.")
            else:
                with st.spinner("Calculating spatial mask & attention weights..."):
                    target_params = {
                        'defect_type': target_defect,
                        'eps0': PhysicsParameters.get_eigenstrain(target_defect),
                        'kappa': 0.6,
                        'theta': np.radians(target_angle),
                        'shape': 'Square'
                    }
                    
                    st.session_state.interpolator.set_spatial_parameters(spatial_sigma=spatial_sigma)
                    result = st.session_state.interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        target_angle,
                        target_params
                    )
                    st.session_state.result = result

    # Main Content
    if st.session_state.result:
        result = st.session_state.result
        
        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Target Angle", f"{result['target_angle']:.1f}°")
        c2.metric("Target Defect", result['target_params']['defect_type'])
        c3.metric("Attention Entropy", f"{result['weights']['entropy']:.3f}")

        st.divider()
        
        # VISUALIZATION: ENHANCED SANKEY ONLY
        visualizer = WeightVisualizer()
        
        # Pass all the enhancement options from sidebar
        fig = visualizer.create_enhanced_sankey_diagram(
            sources_data=result['sources_data'],
            target_angle=result['target_angle'],
            target_defect=result['target_params']['defect_type'],
            spatial_sigma=spatial_sigma,
            label_font_size=label_font,
            title_font_size=title_font,
            legend_font_size=legend_font,
            node_pad=node_pad,
            node_thickness=node_thick,
            link_opacity=link_opac
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Data View
        with st.expander("📊 View Weight Data Table"):
            df = pd.DataFrame(result['sources_data'])
            st.dataframe(df[['source_index', 'defect_type', 'theta_deg', 
                             'spatial_weight', 'defect_weight', 'combined_weight']])

    else:
        st.info("👈 Configure parameters in the sidebar and click 'Run Interpolation' to generate the Enhanced Sankey visualization.")

if __name__ == "__main__":
    main()
