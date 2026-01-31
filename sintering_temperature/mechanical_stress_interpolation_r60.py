import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION PATHS
# =============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
os.makedirs(SOLUTIONS_DIR, exist_ok=True)

# =============================================
# PHYSICS PARAMETERS
# =============================================
class PhysicsParameters:
    EIGENSTRAIN_VALUES = {'Twin': 2.12, 'ISF': 0.289, 'ESF': 0.333, 'No Defect': 0.0}
    
    @staticmethod
    def get_eigenstrain(defect_type: str) -> float:
        return PhysicsParameters.EIGENSTRAIN_VALUES.get(defect_type, 0.0)

# =============================================
# POSITIONAL ENCODING
# =============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)

# =============================================
# TRANSFORMER SPATIAL INTERPOLATOR
# =============================================
class TransformerSpatialInterpolator:
    def __init__(self, d_model=64, nhead=8, num_layers=3, spatial_sigma=10.0, temperature=1.0):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.spatial_sigma = spatial_sigma
        self.temperature = temperature
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(15, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
    
    def set_spatial_parameters(self, spatial_sigma=None):
        if spatial_sigma is not None:
            self.spatial_sigma = spatial_sigma
    
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
            
            # DEFECT MASK: Hard constraint
            if src.get('defect_type') == target_defect:
                defect_mask.append(1.0)
            else:
                defect_mask.append(1e-6)
            
            # SPATIAL KERNEL: Gaussian angular bracketing
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
            
            # One-hot encoding for defect type
            defect_types = ['ISF', 'ESF', 'Twin', 'No Defect']
            defect = params.get('defect_type', 'Twin')
            for dt in defect_types:
                features.append(1.0 if dt == defect else 0.0)
            
            # One-hot encoding for shape
            shapes = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
            shape = params.get('shape', 'Square')
            for s in shapes:
                features.append(1.0 if s == shape else 0.0)
            
            # Angular features
            theta_deg = np.degrees(theta) if theta is not None else 0.0
            angle_diff = abs(theta_deg - target_angle_deg)
            angle_diff = min(angle_diff, 360 - angle_diff)
            features.append(np.exp(-angle_diff / 45.0))
            features.append(np.sin(np.radians(2 * theta_deg)))
            features.append(np.cos(np.radians(2 * theta_deg)))
            
            # Distance to habit plane
            habit_distance = abs(theta_deg - 54.7)
            habit_distance = min(habit_distance, 360 - habit_distance)
            features.append(np.exp(-habit_distance / 15.0))
            
            # Pad to 15 features
            while len(features) < 15:
                features.append(0.0)
            features = features[:15]
            encoded.append(features)
        return torch.FloatTensor(encoded)
    
    def interpolate_spatial_fields(self, sources, target_angle_deg, target_params):
        if not sources:
            st.error("No sources provided for interpolation.")
            return None
        
        try:
            source_params = []
            source_fields = []
            source_indices = []
            
            # Extract valid sources with stress fields
            for i, src in enumerate(sources):
                if 'params' not in src or 'history' not in src:
                    continue
                
                history = src['history']
                if not history or not isinstance(history[-1], dict):
                    continue
                
                last_frame = history[-1]
                if 'stresses' not in last_frame:
                    continue
                
                # Extract stress fields
                stress_fields = last_frame['stresses']
                sxx = stress_fields.get('sigma_xx', np.zeros((128, 128)))
                syy = stress_fields.get('sigma_yy', np.zeros_like(sxx))
                szz = stress_fields.get('sigma_zz', np.zeros_like(sxx))
                txy = stress_fields.get('tau_xy', np.zeros_like(sxx))
                
                # Compute von Mises stress
                von_mises = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*txy**2))
                hydrostatic = (sxx + syy + szz) / 3.0
                sigma_mag = np.sqrt(von_mises**2 + hydrostatic**2)
                
                source_params.append(src['params'])
                source_indices.append(i)
                source_fields.append({
                    'von_mises': von_mises,
                    'sigma_hydro': hydrostatic,
                    'sigma_mag': sigma_mag,
                    'source_index': i,
                    'source_params': src['params']
                })
            
            if not source_params or not source_fields:
                st.error("No valid sources with stress fields found.")
                return None
            
            # Encode parameters
            source_features = self.encode_parameters(source_params, target_angle_deg)
            target_features = self.encode_parameters([target_params], target_angle_deg)
            
            # Pad features if needed
            if source_features.shape[1] < 15:
                padding = torch.zeros(source_features.shape[0], 15 - source_features.shape[1])
                source_features = torch.cat([source_features, padding], dim=1)
            if target_features.shape[1] < 15:
                padding = torch.zeros(target_features.shape[0], 15 - target_features.shape[1])
                target_features = torch.cat([target_features, padding], dim=1)
            
            # Compute spatial kernel and defect mask
            spatial_kernel, defect_mask, angular_distances = self.compute_angular_bracketing_kernel(
                source_params, target_params
            )
            
            # Transformer processing
            all_features = torch.cat([target_features, source_features], dim=0).unsqueeze(0)
            proj_features = self.input_proj(all_features)
            proj_features = self.pos_encoder(proj_features)
            transformer_output = self.transformer(proj_features)
            
            target_rep = transformer_output[:, 0, :]
            source_reps = transformer_output[:, 1:, :]
            
            # Compute attention scores
            attn_scores = torch.matmul(target_rep.unsqueeze(1), source_reps.transpose(1, 2)).squeeze(1)
            attn_scores = attn_scores / np.sqrt(self.d_model)
            attn_scores = attn_scores / self.temperature
            
            # APPLY PHYSICS PRIORS: Spatial kernel × Defect mask
            spatial_kernel_tensor = torch.FloatTensor(spatial_kernel).unsqueeze(0)
            defect_mask_tensor = torch.FloatTensor(defect_mask).unsqueeze(0)
            biased_scores = attn_scores * spatial_kernel_tensor * defect_mask_tensor
            
            final_attention_weights = torch.softmax(biased_scores, dim=-1).squeeze().detach().cpu().numpy()
            
            # Prepare weight analysis data
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
                    'spatial_weight': spatial_w,
                    'defect_weight': defect_w,
                    'attention_weight': attention_weight,
                    'combined_weight': combined_w,
                    'target_defect_match': field['source_params'].get('defect_type') == target_params['defect_type'],
                    'source_params': field['source_params']
                })
            
            return {
                'sources_data': sources_data,
                'target_params': target_params,
                'target_angle': target_angle_deg,
                'weights': {
                    'combined': final_attention_weights.tolist(),
                    'spatial_kernel': spatial_kernel.tolist(),
                    'defect_mask': defect_mask.tolist()
                }
            }
        except Exception as e:
            st.error(f"Interpolation error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

# =============================================
# ENHANCED SANKEY VISUALIZER
# =============================================
class EnhancedSankeyVisualizer:
    def __init__(self):
        self.color_scheme = {
            'Spatial': '#36A2EB',    # Bright Blue
            'Defect': '#FF6384',     # Pink
            'Attention': '#4BC0C0',  # Cyan
            'Combined': '#9966FF',   # Purple
            'Target': '#FF6B6B',     # Bright Red
            'Source': '#4ECDC4'      # Turquoise
        }
    
    def create_enhanced_sankey(
        self,
        sources_data: List[Dict[str, Any]],
        target_angle: float,
        target_defect: str,
        spatial_sigma: float,
        target_params: Dict[str, Any],
        # CUSTOMIZATION PARAMETERS
        label_font_size: int = 16,
        title_font_size: int = 24,
        legend_font_size: int = 14,
        node_thickness: int = 30,
        node_padding: int = 25,
        link_opacity: float = 0.8,
        show_annotations: bool = True,
        diagram_width: int = 1400,
        diagram_height: int = 900,
        use_dark_mode: bool = False,
        show_link_labels: bool = True,
        link_line_width: float = 0.5
    ) -> go.Figure:
        
        # Check if sources_data is empty
        if not sources_data:
            st.error("No source data available for Sankey diagram.")
            # Create an empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No source data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=24, color="red")
            )
            fig.update_layout(
                title="Error: No Data Available",
                paper_bgcolor='white',
                plot_bgcolor='white',
                width=diagram_width,
                height=diagram_height
            )
            return fig
        
        # Get target kappa safely
        target_kappa = target_params.get('kappa', 0.6)
        target_shape = target_params.get('shape', 'Square')
        
        # Create nodes
        labels = ['Target']
        for source in sources_data:
            angle = source['theta_deg']
            defect = source['defect_type']
            # Get source kappa safely
            source_kappa = source.get('source_params', {}).get('kappa', 0.6)
            labels.append(f"S{source['source_index']}\n{defect}\n{angle:.1f}°\nκ={source_kappa:.2f}")
        
        component_start = len(labels)
        component_labels = ['Spatial Kernel', 'Defect Match', 'Attention Score', 'Combined Weight']
        labels.extend(component_labels)
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        colors = []
        link_labels = []
        
        for i, source in enumerate(sources_data):
            source_idx = i + 1
            
            # Spatial kernel link
            source_indices.append(source_idx)
            target_indices.append(component_start)
            spatial_value = source['spatial_weight'] * 100
            values.append(spatial_value)
            colors.append(f'rgba(54, 162, 235, {link_opacity})')
            link_labels.append(f"Spatial: {source['spatial_weight']:.3f}")
            
            # Defect match link
            source_indices.append(source_idx)
            target_indices.append(component_start + 1)
            defect_value = source['defect_weight'] * 100
            values.append(defect_value)
            colors.append(f'rgba(255, 99, 132, {link_opacity})')
            link_labels.append(f"Defect: {source['defect_weight']:.3f}")
            
            # Attention score link
            source_indices.append(source_idx)
            target_indices.append(component_start + 2)
            attention_w = source.get('attention_weight', source['combined_weight'] * 0.5)
            attention_value = attention_w * 100
            values.append(attention_value)
            colors.append(f'rgba(75, 192, 192, {link_opacity})')
            link_labels.append(f"Attention: {attention_w:.3f}")
            
            # Combined weight link
            source_indices.append(source_idx)
            target_indices.append(component_start + 3)
            combined_value = source['combined_weight'] * 100
            values.append(combined_value)
            colors.append(f'rgba(153, 102, 255, {link_opacity})')
            link_labels.append(f"Combined: {source['combined_weight']:.3f}")
        
        # Links from components to target
        for comp_idx in range(4):
            source_indices.append(component_start + comp_idx)
            target_indices.append(0)
            
            # Calculate component value safely
            comp_value = 0
            for s_idx, t_idx, val in zip(source_indices[:-4], target_indices[:-4], values[:-4]):
                if t_idx == component_start + comp_idx:
                    comp_value += val
            
            values.append(comp_value * 0.5)
            
            base_colors = [
                'rgba(54, 162, 235',    # Spatial
                'rgba(255, 99, 132',    # Defect
                'rgba(75, 192, 192',    # Attention
                'rgba(153, 102, 255'    # Combined
            ]
            colors.append(f'{base_colors[comp_idx]}, {link_opacity * 0.7})')
            link_labels.append(f"{component_labels[comp_idx]} → Target")
        
        # Color scheme based on mode
        bg_color = 'rgb(30, 30, 30)' if use_dark_mode else 'rgba(245, 247, 250, 0.95)'
        paper_color = 'rgb(20, 20, 20)' if use_dark_mode else 'white'
        text_color = 'white' if use_dark_mode else '#2C3E50'
        grid_color = 'rgba(255, 255, 255, 0.1)' if use_dark_mode else 'rgba(0, 0, 0, 0.1)'
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=node_padding,
                thickness=node_thickness,
                line=dict(color="white" if use_dark_mode else "black", width=2),
                label=labels,
                color=[
                    self.color_scheme['Target']
                ] + [
                    self.color_scheme['Source'] for _ in range(len(sources_data))
                ] + [
                    self.color_scheme['Spatial'],
                    self.color_scheme['Defect'],
                    self.color_scheme['Attention'],
                    self.color_scheme['Combined']
                ],
                hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=colors,
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Flow: %{value:.2f}<br>%{customdata}<extra></extra>',
                customdata=link_labels if show_link_labels else None,
                line=dict(width=link_line_width, color='rgba(255,255,255,0.3)')
            ),
            hoverinfo='all'
        )])
        
        # Layout with customizable fonts
        fig.update_layout(
            title=dict(
                text=f'<b>SANKEY DIAGRAM: ATTENTION COMPONENT FLOW</b><br>'
                     f'<span style="font-size: {title_font_size-4}px; font-weight: normal;">'
                     f'Target: {target_angle}° {target_defect} | σ={spatial_sigma}° | '
                     f'κ={target_kappa:.2f} | Shape: {target_shape}</span>',
                font=dict(
                    family='Arial, sans-serif',
                    size=title_font_size,
                    color=text_color,
                    weight='bold'
                ),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            font=dict(
                family='Arial, sans-serif',
                size=label_font_size,
                color=text_color
            ),
            width=diagram_width,
            height=diagram_height,
            plot_bgcolor=bg_color,
            paper_bgcolor=paper_color,
            margin=dict(t=100, l=50, r=50, b=80),
            hoverlabel=dict(
                font=dict(
                    family='Arial, sans-serif',
                    size=label_font_size,
                    color='white' if use_dark_mode else 'black'
                ),
                bgcolor='rgba(44, 62, 80, 0.95)' if not use_dark_mode else 'rgba(200, 200, 200, 0.9)',
                bordercolor='white' if use_dark_mode else 'black'
            )
        )
        
        # Color legend annotations with configurable font size
        if show_annotations:
            annotations = [
                dict(
                    x=0.02, y=-0.08,
                    xref='paper', yref='paper',
                    text='<b style="font-size: 16px;">COLOR CODING:</b>',
                    showarrow=False,
                    font=dict(size=legend_font_size + 2, color=text_color, family='Arial', weight='bold')
                ),
                dict(
                    x=0.18, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Spatial"]}; font-size: 20px;">█</span> Spatial',
                    showarrow=False,
                    font=dict(size=legend_font_size, color=text_color, family='Arial')
                ),
                dict(
                    x=0.32, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Defect"]}; font-size: 20px;">█</span> Defect',
                    showarrow=False,
                    font=dict(size=legend_font_size, color=text_color, family='Arial')
                ),
                dict(
                    x=0.46, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Attention"]}; font-size: 20px;">█</span> Attention',
                    showarrow=False,
                    font=dict(size=legend_font_size, color=text_color, family='Arial')
                ),
                dict(
                    x=0.62, y=-0.08,
                    xref='paper', yref='paper',
                    text=f'<span style="color:{self.color_scheme["Combined"]}; font-size: 20px;">█</span> Combined',
                    showarrow=False,
                    font=dict(size=legend_font_size, color=text_color, family='Arial')
                )
            ]
            fig.update_layout(annotations=annotations)
        
        return fig

# =============================================
# SOLUTION LOADER (LOAD ALL SOLUTIONS)
# =============================================
class SolutionLoader:
    def __init__(self, solutions_dir: str = SOLUTIONS_DIR):
        self.solutions_dir = solutions_dir
    
    def scan_solutions(self) -> List[Dict[str, Any]]:
        all_files = []
        for ext in ['*.pkl', '*.pickle', '*.pt', '*.pth']:
            import glob
            pattern = os.path.join(self.solutions_dir, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        all_files.sort(key=os.path.getmtime, reverse=True)
        file_info = []
        for file_path in all_files:
            try:
                info = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'format': 'pkl' if file_path.endswith(('.pkl', '.pickle')) else 'pt'
                }
                file_info.append(info)
            except:
                continue
        return file_info
    
    def load_solution(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                if file_path.endswith(('.pt', '.pth')):
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=True)
                    except:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                else:
                    data = pickle.load(f)
                return self._standardize_data(data)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None
    
    def _standardize_data(self, data):
        standardized = {'params': {}, 'history': []}
        try:
            if isinstance(data, dict):
                if 'params' in data:
                    standardized['params'] = data['params']
                elif 'parameters' in data:
                    standardized['params'] = data['parameters']
                elif 'param' in data:
                    standardized['params'] = data['param']
                elif 'config' in data:
                    standardized['params'] = data['config']
                
                if 'history' in data:
                    history = data['history']
                    if isinstance(history, list):
                        standardized['history'] = history
                    elif isinstance(history, dict):
                        history_list = []
                        for key in sorted(history.keys()):
                            if isinstance(history[key], dict):
                                history_list.append(history[key])
                        standardized['history'] = history_list
                elif 'results' in data:
                    standardized['history'] = [{'stresses': data['results']}]
                elif 'stresses' in data:
                    standardized['history'] = [{'stresses': data['stresses']}]
                
                # Convert tensors to numpy
                self._convert_tensors(standardized)
                
                # Ensure required parameters exist
                if 'theta' not in standardized['params']:
                    standardized['params']['theta'] = 0.0
                if 'defect_type' not in standardized['params']:
                    standardized['params']['defect_type'] = 'Twin'
                if 'eps0' not in standardized['params']:
                    standardized['params']['eps0'] = PhysicsParameters.get_eigenstrain(
                        standardized['params'].get('defect_type', 'Twin')
                    )
                if 'kappa' not in standardized['params']:
                    standardized['params']['kappa'] = 0.6
                if 'shape' not in standardized['params']:
                    standardized['params']['shape'] = 'Square'
                    
        except Exception as e:
            st.error(f"Standardization error: {e}")
        return standardized
    
    def _convert_tensors(self, data):
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
    
    def load_all_solutions(self, max_files=None):
        solutions = []
        file_info = self.scan_solutions()
        
        if max_files:
            file_info = file_info[:max_files]
        
        if file_info:
            st.info(f"📁 Found {len(file_info)} solution files. Loading all...")
        
        for i, info in enumerate(file_info):
            with st.spinner(f"Loading file {i+1}/{len(file_info)}: {info['filename']}"):
                solution = self.load_solution(info['path'])
                if solution:
                    solutions.append(solution)
        
        return solutions

# =============================================
# MAIN APPLICATION
# =============================================
def main():
    st.set_page_config(
        page_title="Enhanced Sankey Visualization - Weight Components",
        layout="wide",
        page_icon="🌊",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem !important;
        color: #1E3A8A !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #6366F1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 1.5rem;
    }
    .config-section {
        background: linear-gradient(135deg, #F0F9FF, #E0F2FE);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #3B82F6;
    }
    .viz-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin-top: 1.5rem;
    }
    .sidebar-title {
        font-size: 1.5rem !important;
        color: #3B82F6 !important;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #FFF3CD, #FFEAA7);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #FFC107;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌊 ENHANCED SANKEY VISUALIZATION - WEIGHT COMPONENTS</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #E0F7FA, #E3F2FD); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
        <strong>Physics-Aware Attention Weights:</strong> Spatial Kernel (Angular Bracketing) × Defect Mask (Hard Constraint) × Learned Attention × Kappa (Material Parameter)
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'solutions' not in st.session_state:
        st.session_state.solutions = []
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = TransformerSpatialInterpolator(spatial_sigma=10.0)
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EnhancedSankeyVisualizer()
    if 'interpolation_result' not in st.session_state:
        st.session_state.interpolation_result = None
    if 'loader' not in st.session_state:
        st.session_state.loader = SolutionLoader(SOLUTIONS_DIR)
    
    # SIDEBAR CONFIGURATION
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">⚙️ INTERPOLATION CONFIGURATION</h2>', unsafe_allow_html=True)
        
        # Load solutions button - LOAD ALL
        if st.button("📤 LOAD ALL SOLUTIONS", type="primary", use_container_width=True):
            with st.spinner("Loading ALL solutions from numerical_solutions directory..."):
                st.session_state.solutions = st.session_state.loader.load_all_solutions()
                if st.session_state.solutions:
                    st.success(f"✅ Loaded {len(st.session_state.solutions)} solutions")
                    # Show summary
                    angles = []
                    defects = []
                    shapes = []
                    kappas = []
                    
                    for sol in st.session_state.solutions:
                        if 'params' in sol:
                            params = sol['params']
                            if 'theta' in params:
                                angles.append(np.degrees(params['theta']) % 360)
                            if 'defect_type' in params:
                                defects.append(params['defect_type'])
                            if 'shape' in params:
                                shapes.append(params['shape'])
                            if 'kappa' in params:
                                kappas.append(params['kappa'])
                    
                    if angles:
                        st.info(f"Angles: {min(angles):.1f}°–{max(angles):.1f}°")
                    if defects:
                        st.info(f"Defects: {', '.join(set(defects))}")
                    if shapes:
                        st.info(f"Shapes: {', '.join(set(shapes))}")
                    if kappas:
                        st.info(f"Kappa range: {min(kappas):.2f}–{max(kappas):.2f}")
                else:
                    st.warning(f"⚠️ No valid solution files found in: {SOLUTIONS_DIR}")
                    st.markdown("""
                    <div class="warning-box">
                    💡 <strong>Place .pkl/.pt files with simulation results in:</strong><br>
                    <code>numerical_solutions/</code><br><br>
                    <strong>Expected format:</strong><br>
                    - Dictionary with 'params' and 'history' keys<br>
                    - 'params' should include: theta, defect_type, eps0, kappa, shape<br>
                    - 'history' should contain stress field data
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Target parameters
        st.markdown("#### 🎯 Target Configuration")
        col1, col2 = st.columns(2)
        with col1:
            target_angle = st.number_input(
                "Target Angle (°)",
                min_value=0.0,
                max_value=180.0,
                value=54.7,
                step=0.1,
                help="Target orientation for interpolation (habit plane = 54.7°)"
            )
        with col2:
            if st.button("↺ Habit Plane", use_container_width=True):
                target_angle = 54.7
                st.rerun()
        
        target_defect = st.selectbox(
            "Target Defect Type",
            options=['Twin', 'ISF', 'ESF', 'No Defect'],
            index=0,
            help="Defect type constraint - only matching sources contribute significantly"
        )
        
        # SHAPE PARAMETER INPUT
        shape_options = ['Square', 'Horizontal Fault', 'Vertical Fault', 'Rectangle']
        target_shape = st.selectbox(
            "Target Shape",
            options=shape_options,
            index=0,
            help="Geometry shape of the defect configuration"
        )
        
        # KAPPA PARAMETER INPUT (Learned from user request)
        target_kappa = st.number_input(
            "Kappa (κ) - Material Parameter",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.01,
            help="Material parameter controlling stress field characteristics. Typical range: 0.1-2.0"
        )
        
        spatial_sigma = st.slider(
            "Spatial Kernel Width σ (°)",
            min_value=1.0,
            max_value=45.0,
            value=10.0,
            step=0.5,
            help="Angular bracketing window width - controls spatial locality"
        )
        
        st.divider()
        
        # Sankey customization
        st.markdown('<h2 class="sidebar-title">🎨 SANKEY CUSTOMIZATION</h2>', unsafe_allow_html=True)
        
        st.markdown("##### 📏 Font Sizes")
        label_font_size = st.slider(
            "Node Label Font Size",
            min_value=8,
            max_value=58, #28
            value=16,
            step=1,
            help="Font size for source/target node labels"
        )
        title_font_size = st.slider(
            "Title Font Size",
            min_value=16,
            max_value=40,
            value=24,
            step=1,
            help="Font size for diagram title"
        )
        legend_font_size = st.slider(
            "Legend Font Size",
            min_value=8,
            max_value=44, #24
            value=14,
            step=1,
            help="Font size for color coding legend"
        )
        
        st.markdown("##### 🧱 Node Appearance")
        node_thickness = st.slider(
            "Node Thickness",
            min_value=10,
            max_value=60,
            value=30,
            step=2,
            help="Thickness of Sankey flow nodes"
        )
        node_padding = st.slider(
            "Node Padding",
            min_value=5,
            max_value=50,
            value=25,
            step=1,
            help="Spacing between nodes"
        )
        
        st.markdown("##### 🔗 Link Appearance")
        link_opacity = st.slider(
            "Link Opacity",
            min_value=0.2,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Transparency of flow links"
        )
        link_line_width = st.slider(
            "Link Border Width",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Width of link border lines"
        )
        show_link_labels = st.toggle(
            "Show Link Hover Labels",
            value=True,
            help="Display weight values on link hover"
        )
        
        st.markdown("##### 🖼️ Layout")
        col1, col2 = st.columns(2)
        with col1:
            diagram_width = st.number_input(
                "Width (px)",
                min_value=800,
                max_value=2500,
                value=1400,
                step=100,
                help="Diagram width in pixels"
            )
        with col2:
            diagram_height = st.number_input(
                "Height (px)",
                min_value=600,
                max_value=1800,
                value=900,
                step=100,
                help="Diagram height in pixels"
            )
        
        st.markdown("##### 🎨 Display Options")
        show_annotations = st.toggle(
            "Show Color Legend",
            value=True,
            help="Display color coding explanation below diagram"
        )
        use_dark_mode = st.toggle(
            "Dark Mode",
            value=False,
            help="Switch to dark color scheme"
        )
        
        st.divider()
        
        # Run interpolation
        if st.button("🚀 PERFORM INTERPOLATION", type="primary", use_container_width=True, disabled=len(st.session_state.solutions)==0):
            if not st.session_state.solutions:
                st.error("❌ No solutions loaded! Click 'LOAD ALL SOLUTIONS' first.")
            else:
                with st.spinner("Performing physics-aware interpolation with KAPPA parameter..."):
                    # Setup target parameters WITH SHAPE AND KAPPA
                    target_params = {
                        'defect_type': target_defect,
                        'eps0': PhysicsParameters.get_eigenstrain(target_defect),
                        'kappa': target_kappa,  # INCLUDE KAPPA PARAMETER
                        'theta': np.radians(target_angle),
                        'shape': target_shape  # INCLUDE SHAPE PARAMETER
                    }
                    
                    # Update interpolator parameters
                    st.session_state.interpolator.set_spatial_parameters(spatial_sigma=spatial_sigma)
                    
                    # Perform REAL interpolation with physics priors
                    result = st.session_state.interpolator.interpolate_spatial_fields(
                        st.session_state.solutions,
                        target_angle,
                        target_params
                    )
                    
                    if result and result['sources_data']:
                        st.session_state.interpolation_result = result
                        st.success(f"✅ Interpolation complete! {len(result['sources_data'])} sources processed")
                        st.session_state.target_angle = target_angle
                        st.session_state.target_defect = target_defect
                        st.session_state.spatial_sigma = spatial_sigma
                        st.session_state.target_kappa = target_kappa
                        st.session_state.target_shape = target_shape
                    else:
                        st.error("❌ Interpolation failed - no valid sources found for the given parameters")
    
    # MAIN CONTENT AREA
    if not st.session_state.solutions:
        st.markdown("""
        ## 📌 GETTING STARTED
        
        ### 1. Prepare Your Solution Files
        Place your simulation results in the `numerical_solutions` directory with this structure:
        ```python
        {
            'params': {
                'theta': 0.95,           # radians
                'defect_type': 'Twin',   # 'Twin', 'ISF', 'ESF', 'No Defect'
                'eps0': 2.12,            # eigenstrain
                'kappa': 0.6,            # material parameter (configurable)
                'shape': 'Square'        # geometry (configurable)
            },
            'history': [{
                'stresses': {
                    'sigma_xx': np.array(...),
                    'sigma_yy': np.array(...),
                    'sigma_zz': np.array(...),
                    'tau_xy': np.array(...)
                }
            }]
        }
        ```
        
        ### 2. Workflow
        1. Click **"LOAD ALL SOLUTIONS"** in sidebar to scan directory
        2. Configure target parameters (angle, defect type, **shape**, **kappa**, σ)
        3. Click **"PERFORM INTERPOLATION"** to compute physics-aware weights
        4. Customize Sankey visualization using sidebar controls
        5. Adjust font sizes, node appearance, and layout for publications
        
        ### 3. Physics Principles Visualized
        - **Spatial Kernel**: Gaussian weighting based on angular proximity (`exp(-(Δφ)²/2σ²)`)
        - **Defect Mask**: Hard constraint - mismatched defects get ~0 weight (`10⁻⁶`)
        - **Kappa (κ)**: Material parameter controlling stress field characteristics
        - **Shape**: Geometry configuration affecting stress distribution
        - **Combined Weight**: Final attention = Spatial × Defect × Learned Attention
        """)
        st.info(f"📁 Solution directory: `{SOLUTIONS_DIR}`")
        return
    
    # Show solution summary
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Loaded Solutions Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Solutions", len(st.session_state.solutions))
    
    with col2:
        angles = []
        for sol in st.session_state.solutions:
            if 'params' in sol and 'theta' in sol['params']:
                angles.append(np.degrees(sol['params']['theta']) % 360)
        if angles:
            st.metric("Angle Range", f"{min(angles):.1f}°–{max(angles):.1f}°")
        else:
            st.metric("Angle Range", "N/A")
    
    with col3:
        defects = []
        for sol in st.session_state.solutions:
            if 'params' in sol and 'defect_type' in sol['params']:
                defects.append(sol['params']['defect_type'])
        if defects:
            st.metric("Defect Types", len(set(defects)))
        else:
            st.metric("Defect Types", "N/A")
    
    with col4:
        shapes = []
        for sol in st.session_state.solutions:
            if 'params' in sol and 'shape' in sol['params']:
                shapes.append(sol['params']['shape'])
        if shapes:
            st.metric("Shape Types", len(set(shapes)))
        else:
            st.metric("Shape Types", "N/A")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display Sankey diagram if interpolation performed
    if st.session_state.interpolation_result and st.session_state.interpolation_result['sources_data']:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.markdown("### 🌊 Enhanced Sankey Diagram: Attention Component Flow")
        
        # Create and display diagram with current settings
        fig = st.session_state.visualizer.create_enhanced_sankey(
            sources_data=st.session_state.interpolation_result['sources_data'],
            target_angle=st.session_state.target_angle,
            target_defect=st.session_state.target_defect,
            spatial_sigma=st.session_state.spatial_sigma,
            target_params=st.session_state.interpolation_result['target_params'],
            label_font_size=label_font_size,
            title_font_size=title_font_size,
            legend_font_size=legend_font_size,
            node_thickness=node_thickness,
            node_padding=node_padding,
            link_opacity=link_opacity,
            show_annotations=show_annotations,
            diagram_width=diagram_width,
            diagram_height=diagram_height,
            use_dark_mode=use_dark_mode,
            show_link_labels=show_link_labels,
            link_line_width=link_line_width
        )
        
        st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': True, 'scrollZoom': True})
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Weight table with KAPPA information
        st.markdown("### 📋 Weight Components Breakdown")
        df = pd.DataFrame(st.session_state.interpolation_result['sources_data'])
        
        # Ensure all required columns exist
        required_columns = ['source_index', 'theta_deg', 'defect_type', 'angular_dist', 
                           'spatial_weight', 'defect_weight', 'attention_weight', 
                           'combined_weight', 'target_defect_match']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Add kappa column from source_params if available
        df['kappa'] = df.apply(
            lambda row: row.get('source_params', {}).get('kappa', 0.6) if isinstance(row.get('source_params'), dict) else 0.6,
            axis=1
        )
        
        # Display the dataframe
        display_cols = ['source_index', 'theta_deg', 'defect_type', 'kappa', 'angular_dist', 
                       'spatial_weight', 'defect_weight', 'attention_weight', 
                       'combined_weight', 'target_defect_match']
        
        st.dataframe(
            df[display_cols].style
            .background_gradient(subset=['combined_weight'], cmap='viridis')
            .format({
                'theta_deg': '{:.1f}°',
                'angular_dist': '{:.1f}°',
                'kappa': '{:.2f}',
                'spatial_weight': '{:.4f}',
                'defect_weight': '{:.2e}',
                'attention_weight': '{:.4f}',
                'combined_weight': '{:.4f}'
            })
        )
        
        # Physics interpretation with KAPPA
        st.markdown(f"""
        ### 🔬 Physics Interpretation
        
        **Configuration Parameters:**
        - **Target Angle:** {st.session_state.target_angle:.1f}°
        - **Defect Type:** {st.session_state.target_defect}
        - **Shape:** {st.session_state.target_shape}
        - **Kappa (κ):** {st.session_state.target_kappa:.2f} (Material parameter)
        - **Spatial Sigma (σ):** {st.session_state.spatial_sigma:.1f}°
        
        **Defect Type Gating (Hard Constraint):**
        - Sources with matching defect types receive full weight (`1.0`)
        - Mismatched defects receive near-zero weight (`10⁻⁶`) - physically invalid for interpolation
        - This ensures stress field compatibility during interpolation
        
        **Angular Bracketing (Spatial Kernel):**
        - Gaussian weighting centered on target angle: `exp(-(Δφ)²/2σ²)`
        - Sources within ±σ degrees receive highest weights
        - Creates natural "bracketing" structure optimal for interpolation
        
        **Kappa (κ) Parameter:**
        - Controls material response characteristics
        - Affects stress field magnitude and distribution
        - Typical range: 0.1 (soft) to 2.0 (stiff)
        - Influences attention through learned transformer features
        
        **Combined Attention:**
        - Final weight = Spatial Kernel × Defect Mask × Learned Attention
        - Physics priors (spatial + defect) modulate learned attention
        - Ensures physically valid interpolation while leveraging pattern recognition
        """)
    elif st.session_state.interpolation_result and not st.session_state.interpolation_result['sources_data']:
        st.warning("⚠️ Interpolation completed but no valid sources were found for the given parameters.")
        st.info("Try adjusting the target parameters or load more diverse solution files.")
    else:
        st.info("👈 Configure target parameters in the sidebar (including **Shape** and **Kappa**) and click **'PERFORM INTERPOLATION'** to generate the Sankey diagram")

if __name__ == "__main__":
    main()
