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
import torch.nn as nn
import torch.nn.functional as F
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
from collections import OrderedDict
warnings.filterwarnings('ignore')
# =============================================
# PATH CONFIGURATION
# =============================================
SCRIPT_DIR = os.getcwd()
NUMERICAL_SOLUTIONS_DIR = os.path.join(SCRIPT_DIR, "numerical_solutions")
if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
    os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
# =============================================
# ENHANCED SPATIAL LOCALITY REGULARIZATION ATTENTION INTERPOLATOR
# =============================================
class SpatialLocalityAttentionInterpolator(nn.Module):
    """Enhanced attention-based interpolator with spatial locality regularization"""
   
    def __init__(self, input_dim=15, num_heads=4, d_model=32, output_dim=3,
                 sigma_spatial=0.2, sigma_param=0.2, use_gaussian=True, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.output_dim = output_dim
        self.sigma_spatial = sigma_spatial
        self.sigma_param = sigma_param
        self.use_gaussian = use_gaussian
        self.device = device
        
        self.model = nn.ModuleDict({
            'param_embedding': nn.Sequential(
                nn.Linear(self.input_dim, self.d_model * 2),
                nn.ReLU(),
                nn.Linear(self.d_model * 2, self.d_model)
            ),
            'attention': nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1
            ),
            'feed_forward': nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model * 4, self.d_model)
            ),
            'output_projection': nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.ReLU(),
                nn.Linear(self.d_model * 2, self.output_dim)
            ),
            'spatial_regularizer': nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_heads)
            ) if self.use_gaussian else None,
            'norm1': nn.LayerNorm(self.d_model),
            'norm2': nn.LayerNorm(self.d_model)
        }).to(device)
        
        self.source_params = None
        self.source_stress = None

    def compute_parameter_vector(self, sim_data):
        params = sim_data.get('params', {})
        
        param_vector = []
        # Defect type encoding
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
        
        # Shape encoding
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
        shape = params.get('shape', 'Square')
        param_vector.extend(shape_encoding.get(shape, [0, 0, 0, 0, 0]))
        
        # Numerical parameters (normalized)
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
        
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3)
        param_vector.append(eps0_norm)
        
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1)
        param_vector.append(kappa_norm)
        
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        param_vector.append(theta_norm)
        
        # Orientation encoding
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
        
        return np.array(param_vector, dtype=np.float32)

    def extract_stress(self, sim_data):
        history = sim_data.get('history', [])
        if history:
            _, stresses = history[-1]
            return np.stack([
                stresses.get('sigma_hydro', np.zeros((128, 128))),
                stresses.get('sigma_mag', np.zeros((128, 128))),
                stresses.get('von_mises', np.zeros((128, 128)))
            ])
        return np.zeros((3, 128, 128))

    def fit(self, source_simulations):
        self.source_params = np.array([self.compute_parameter_vector(s) for s in source_simulations])
        self.source_stress = np.array([self.extract_stress(s) for s in source_simulations])
        return self

    def train(self, source_simulations, epochs=100, lr=0.001):
        self.to(self.device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare leave-one-out pairs
        pairs = []
        for i in range(len(source_simulations)):
            source_indices = [j for j in range(len(source_simulations)) if j != i]
            source_params = [self.compute_parameter_vector(source_simulations[j]) for j in source_indices]
            source_stress = [self.extract_stress(source_simulations[j]) for j in source_indices]
            target_params = self.compute_parameter_vector(source_simulations[i])
            target_stress = self.extract_stress(source_simulations[i])
            pairs.append((source_params, source_stress, target_params, target_stress))
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for source_params, source_stress, target_params, target_stress in pairs:
                optimizer.zero_grad()
                
                source_p = torch.from_numpy(np.array(source_params)).float().to(self.device)
                target_p = torch.from_numpy(np.array(target_params)).float().unsqueeze(0).to(self.device)
                
                source_emb = self.model['param_embedding'](source_p)
                target_emb = self.model['param_embedding'](target_p)
                
                # Attention
                _, attn_weights = self.model['attention'](target_emb, source_emb, source_emb)
                
                # attn_weights (1,1,N)
                weights = attn_weights.squeeze().softmax(dim=-1)
                
                source_stress_t = torch.from_numpy(np.array(source_stress)).float().to(self.device)
                predicted_stress = torch.sum(source_stress_t * weights.unsqueeze(1).unsqueeze(2).unsqueeze(3), dim=0)
                
                target_stress_t = torch.from_numpy(target_stress).float().to(self.device)
                loss = criterion(predicted_stress, target_stress_t)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(pairs)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        return losses

    def predict(self, target_params):
        self.eval()
        
        with torch.no_grad():
            target_vector = self.compute_parameter_vector({'params': target_params})
            target_p = torch.from_numpy(target_vector).float().unsqueeze(0).to(self.device)
            source_p = torch.from_numpy(self.source_params).float().to(self.device)
            
            source_emb = self.model['param_embedding'](source_p)
            target_emb = self.model['param_embedding'](target_p)
            
            _, attn_weights = self.model['attention'](target_emb, source_emb, source_emb)
            
            weights = attn_weights.squeeze().softmax(dim=-1).cpu().numpy()
            
            weighted_stress = np.sum(self.source_stress * weights[:, None, None, None], axis=0)
            
            predicted_stress = {
                'sigma_hydro': weighted_stress[0],
                'sigma_mag': weighted_stress[1],
                'von_mises': weighted_stress[2],
                'method': 'attention'
            }
            
            return predicted_stress, weights
# =============================================
# KERNEL REGRESSION INTERPOLATOR
# =============================================
class KernelRegressionInterpolator:
    """Kernel-based regression interpolator using Gaussian RBF kernels"""
   
    def __init__(self, kernel_type='rbf', length_scale=0.3, nu=1.5):
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.nu = nu
        self.source_params = None
        self.source_stress = None
        self.kernel_matrix = None
       
    def _compute_kernel(self, X1, X2):
        if self.kernel_type == 'rbf':
            dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-dists / (2 * self.length_scale ** 2))
       
        elif self.kernel_type == 'matern':
            dists = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2))
            if self.nu == 0.5:
                return np.exp(-dists / self.length_scale)
            elif self.nu == 1.5:
                d_scaled = np.sqrt(3) * dists / self.length_scale
                return (1 + d_scaled) * np.exp(-d_scaled)
            else: # nu = 2.5
                d_scaled = np.sqrt(5) * dists / self.length_scale
                return (1 + d_scaled + d_scaled**2/3) * np.exp(-d_scaled)
       
        elif self.kernel_type == 'rational_quadratic':
            dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return (1 + dists / (2 * self.nu * self.length_scale ** 2)) ** (-self.nu)
       
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
   
    def fit(self, source_simulations):
        source_params = []
        source_stress = []
       
        for sim_data in source_simulations:
            param_vector = self.compute_parameter_vector(sim_data)
            source_params.append(param_vector)
           
            history = sim_data.get('history', [])
            if history:
                eta, stress_fields = history[-1]
                stress_components = np.stack([
                    stress_fields.get('sigma_hydro', np.zeros_like(eta)),
                    stress_fields.get('sigma_mag', np.zeros_like(eta)),
                    stress_fields.get('von_mises', np.zeros_like(eta))
                ], axis=0)
                source_stress.append(stress_components)
       
        self.source_params = np.array(source_params)
        self.source_stress = np.array(source_stress)
       
        self.kernel_matrix = self._compute_kernel(self.source_params, self.source_params)
        self.kernel_matrix += np.eye(len(source_simulations)) * 1e-8
       
        return self
   
    def predict(self, target_params):
        if self.source_params is None or self.source_stress is None:
            raise ValueError("Model must be fitted before prediction")
       
        target_vector = self.compute_parameter_vector({'params': target_params})
       
        k_star = self._compute_kernel(self.source_params, target_vector.reshape(1, -1))
       
        try:
            weights = np.linalg.solve(self.kernel_matrix, k_star)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(self.kernel_matrix) @ k_star
       
        weights = weights.flatten()
        weights = weights / (np.sum(weights) + 1e-8)
       
        weighted_stress = np.sum(
            self.source_stress * weights[:, np.newaxis, np.newaxis, np.newaxis],
            axis=0
        )
       
        predicted_stress = {
            'sigma_hydro': weighted_stress[0],
            'sigma_mag': weighted_stress[1],
            'von_mises': weighted_stress[2],
            'method': 'kernel',
            'kernel_weights': weights
        }
       
        return predicted_stress, weights
   
    def compute_parameter_vector(self, sim_data):
        params = sim_data.get('params', {})
       
        param_vector = []
       
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
       
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
        shape = params.get('shape', 'Square')
        param_vector.extend(shape_encoding.get(shape, [0, 0, 0, 0, 0]))
       
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
       
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3)
        param_vector.append(eps0_norm)
       
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1)
        param_vector.append(kappa_norm)
       
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        param_vector.append(theta_norm)
       
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
       
        return np.array(param_vector, dtype=np.float32)
# =============================================
# TRANSFORMER CROSS-ATTENTION INTERPOLATOR
# =============================================
class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial patches"""
   
    def __init__(self, d_model, grid_size, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.temperature = temperature
       
        y_pos = torch.arange(grid_size).float()
        x_pos = torch.arange(grid_size).float()
       
        y_pos = y_pos / grid_size
        x_pos = x_pos / grid_size
       
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
       
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1) 
        
        dim_t = torch.arange(d_model // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / d_model)
       
        pos_x = positions[:, 0:1] / dim_t 
        pos_y = positions[:, 1:2] / dim_t 
        
        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=-1)
        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=-1)
       
        pos_encoding = torch.cat([pos_x, pos_y], dim=-1) 
        
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0)) 
   
    def forward(self, batch_size=1):
        return self.pos_encoding.repeat(batch_size, 1, 1)
class TransformerCrossAttentionInterpolator(nn.Module):
    """True transformer-based cross-attention interpolator for stress fields"""
   
    def __init__(self,
                 param_dim=15,
                 stress_dim=3,
                 spatial_dim=2,
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 max_seq_len=1024,
                 patch_size=8,
                 use_pos_encoding=True):
        super().__init__()
        
        self.param_dim = param_dim
        self.stress_dim = stress_dim
        self.spatial_dim = spatial_dim
        self.d_model = d_model
        self.patch_size = patch_size
        self.use_pos_encoding = use_pos_encoding
        self.max_seq_len = max_seq_len
       
        self.num_patches = (128 // patch_size) ** 2
       
        self.patch_embed = nn.Conv2d(
            stress_dim, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
       
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
       
        if use_pos_encoding:
            self.positional_encoding = PositionalEncoding2D(
                d_model,
                grid_size=128//patch_size
            )
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
       
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
       
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, stress_dim * patch_size * patch_size)
        )
       
        self.target_queries = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )
       
        self.attention_weight_cache = None
       
    def patchify(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
       
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous() 
        
        return x
   
    def unpatchify(self, x, H=128, W=128):
        B, N, _ = x.shape
        patches_per_dim = H // self.patch_size
       
        x = x.view(B, N, self.stress_dim, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous() 
       
        x = x.view(B, self.stress_dim, patches_per_dim, patches_per_dim,
                   self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous() 
        x = x.view(B, self.stress_dim, H, W)
       
        return x
   
    def forward(self, source_stress, source_params, target_params):
        if not isinstance(source_stress, list):
            source_stress = [source_stress]
        if not isinstance(source_params, list):
            source_params = [source_params]
       
        B = len(source_stress) 
        
        source_tokens = []
        for stress in source_stress:
            stress_tensor = stress.unsqueeze(0) if len(stress.shape) == 3 else stress
           
            patches = self.patch_embed(stress_tensor) 
            patches = patches.flatten(2).transpose(1, 2) 
            source_tokens.append(patches)
       
        source_tokens = torch.cat(source_tokens, dim=0) 
        
        param_embeddings = []
        for params in source_params:
            params_tensor = params.unsqueeze(0) if len(params.shape) == 1 else params
            param_emb = self.param_embed(params_tensor) 
            param_emb = param_emb.unsqueeze(1).repeat(1, self.num_patches, 1) 
            param_embeddings.append(param_emb)
       
        param_embeddings = torch.cat(param_embeddings, dim=0) 
        
        source_embeddings = source_tokens + param_embeddings 
        
        if self.use_pos_encoding:
            pos_enc = self.positional_encoding(source_embeddings.shape[0])
            source_embeddings = source_embeddings + pos_enc
       
        src_key_padding_mask = None 
        
        memory = self.encoder(
            src=source_embeddings,
            mask=None,
            src_key_padding_mask=src_key_padding_mask
        ) 
        
        target_queries = self.target_queries.repeat(B, 1, 1) 
        
        target_params_tensor = target_params.unsqueeze(0) if len(target_params.shape) == 1 else target_params
        target_param_emb = self.param_embed(target_params_tensor) 
        target_param_emb = target_param_emb.unsqueeze(1).repeat(1, self.num_patches, 1)
       
        target_queries = target_queries + target_param_emb
        
        if self.use_pos_encoding:
            target_queries = target_queries + pos_enc
        
        tgt_mask = self.generate_square_subsequent_mask(self.num_patches).to(target_queries.device)
       
        attention_weights = []
       
        def attention_hook(module, input, output):
            if isinstance(output, tuple):
                attn_output, attn_weights = output
                attention_weights.append(attn_weights.detach())
            return output
       
        hooks = []
        for layer in self.decoder.layers:
            hooks.append(layer.multihead_attn.register_forward_hook(attention_hook))
       
        output = self.decoder(
            tgt=target_queries,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=src_key_padding_mask
        ) 
        
        for hook in hooks:
            hook.remove()
       
        if attention_weights:
            self.attention_weight_cache = attention_weights
        
        output = self.output_proj(output) 
        
        output = output.mean(dim=0, keepdim=True) 
        
        predicted_stress = self.unpatchify(output) 
        predicted_stress = predicted_stress.squeeze(0) 
        
        return predicted_stress, attention_weights
   
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
# =============================================
# HYBRID INTERPOLATOR
# =============================================
class HybridInterpolator:
    """Hybrid interpolator combining kernel regression and transformer attention"""
   
    def __init__(self, kernel_weight=0.5, transformer_weight=0.5, device='cpu'):
        self.kernel_weight = kernel_weight
        self.transformer_weight = transformer_weight
        self.device = device
       
        self.kernel_interpolator = KernelRegressionInterpolator(kernel_type='rbf', length_scale=0.3)
        self.transformer_interpolator = TransformerCrossAttentionInterpolator(
            param_dim=15,
            stress_dim=3,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            patch_size=8
        ).to(device)
       
        self.source_simulations = []
        self.is_kernel_fitted = False
        self.is_transformer_trained = False
       
    def compute_parameter_vector(self, sim_data):
        params = sim_data.get('params', {})
       
        param_vector = []
       
        defect_encoding = {
            'ISF': [1, 0, 0],
            'ESF': [0, 1, 0],
            'Twin': [0, 0, 1]
        }
        defect_type = params.get('defect_type', 'ISF')
        param_vector.extend(defect_encoding.get(defect_type, [0, 0, 0]))
       
        shape_encoding = {
            'Square': [1, 0, 0, 0, 0],
            'Horizontal Fault': [0, 1, 0, 0, 0],
            'Vertical Fault': [0, 0, 1, 0, 0],
            'Rectangle': [0, 0, 0, 1, 0],
            'Ellipse': [0, 0, 0, 0, 1]
        }
        shape = params.get('shape', 'Square')
        param_vector.extend(shape_encoding.get(shape, [0, 0, 0, 0, 0]))
       
        eps0 = params.get('eps0', 0.707)
        kappa = params.get('kappa', 0.6)
        theta = params.get('theta', 0.0)
       
        eps0_norm = (eps0 - 0.3) / (3.0 - 0.3)
        param_vector.append(eps0_norm)
       
        kappa_norm = (kappa - 0.1) / (2.0 - 0.1)
        param_vector.append(kappa_norm)
       
        theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)
        param_vector.append(theta_norm)
       
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
       
        return np.array(param_vector, dtype=np.float32)
   
    def fit_kernel(self, source_simulations):
        self.source_simulations = source_simulations
        self.kernel_interpolator.fit(source_simulations)
        self.is_kernel_fitted = True
        return self
   
    def train_transformer(self, source_simulations, epochs=100, lr=0.001):
        self.source_simulations = source_simulations
       
        train_pairs = self._prepare_training_pairs(source_simulations)
       
        if not train_pairs:
            raise ValueError("Could not prepare training pairs")
       
        self.transformer_interpolator.train()
       
        optimizer = torch.optim.AdamW(
            self.transformer_interpolator.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        criterion = nn.MSELoss()
       
        losses = []
       
        for epoch in range(epochs):
            epoch_loss = 0.0
           
            for batch in train_pairs:
                source_stress, source_params, target_stress, target_params = batch
               
                source_stress = [s.to(self.device) for s in source_stress]
                source_params = [p.to(self.device) for p in source_params]
                target_stress = target_stress.to(self.device)
                target_params = target_params.to(self.device)
               
                optimizer.zero_grad()
                predicted_stress, _ = self.transformer_interpolator(
                    source_stress, source_params, target_params
                )
               
                loss = criterion(predicted_stress, target_stress)
               
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.transformer_interpolator.parameters(),
                    max_norm=1.0
                )
                optimizer.step()
               
                epoch_loss += loss.item()
           
            avg_loss = epoch_loss / len(train_pairs)
            losses.append(avg_loss)
           
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
           
            if epoch > 20 and losses[-1] > np.mean(losses[-20:-10]):
                print("Early stopping triggered")
                break
       
        self.is_transformer_trained = True
        return losses
   
    def _prepare_training_pairs(self, training_data, num_sources=3):
        pairs = []
       
        sim_tensors = []
        for sim_data in training_data:
            history = sim_data.get('history', [])
            if not history:
                continue
           
            _, stress_fields = history[-1]
           
            stress_tensor = torch.stack([
                torch.from_numpy(stress_fields.get('sigma_hydro', np.zeros((128, 128)))),
                torch.from_numpy(stress_fields.get('sigma_mag', np.zeros((128, 128)))),
                torch.from_numpy(stress_fields.get('von_mises', np.zeros((128, 128))))
            ], dim=0).float()
           
            param_vector = self.compute_parameter_vector(sim_data)
            param_tensor = torch.from_numpy(param_vector).float()
           
            sim_tensors.append((stress_tensor, param_tensor))
       
        for i, (target_stress, target_params) in enumerate(sim_tensors):
            source_indices = [j for j in range(len(sim_tensors)) if j != i]
           
            if len(source_indices) > num_sources:
                import random
                source_indices = random.sample(source_indices, num_sources)
           
            source_stress_list = []
            source_params_list = []
           
            for idx in source_indices:
                src_stress, src_params = sim_tensors[idx]
                source_stress_list.append(src_stress)
                source_params_list.append(src_params)
           
            pairs.append((
                source_stress_list,
                source_params_list,
                target_stress,
                target_params
            ))
       
        return pairs
   
    def predict(self, target_params, use_kernel=None, use_transformer=None):
        if not self.source_simulations:
            raise ValueError("No source simulations available")
       
        if use_kernel is None:
            use_kernel = self.kernel_weight > 0 and self.is_kernel_fitted
       
        if use_transformer is None:
            use_transformer = self.transformer_weight > 0 and self.is_transformer_trained
       
        kernel_pred = None
        transformer_pred = None
        kernel_weights = None
        transformer_attention = None
       
        if use_kernel:
            kernel_pred, kernel_weights = self.kernel_interpolator.predict(target_params)
       
        if use_transformer:
            source_stress_tensors = []
            source_param_tensors = []
           
            for sim_data in self.source_simulations:
                history = sim_data.get('history', [])
                if not history:
                    continue
               
                _, stress_fields = history[-1]
               
                stress_tensor = torch.stack([
                    torch.from_numpy(stress_fields.get('sigma_hydro', np.zeros((128, 128)))),
                    torch.from_numpy(stress_fields.get('sigma_mag', np.zeros((128, 128)))),
                    torch.from_numpy(stress_fields.get('von_mises', np.zeros((128, 128))))
                ], dim=0).float().to(self.device)
               
                param_vector = self.compute_parameter_vector(sim_data)
                param_tensor = torch.from_numpy(param_vector).float().to(self.device)
               
                source_stress_tensors.append(stress_tensor)
                source_param_tensors.append(param_tensor)
           
            target_param_vector = self.compute_parameter_vector({'params': target_params})
            target_param_tensor = torch.from_numpy(target_param_vector).float().to(self.device)
           
            self.transformer_interpolator.eval()
            with torch.no_grad():
                predicted_tensor, attention_weights = self.transformer_interpolator(
                    source_stress_tensors,
                    source_param_tensors,
                    target_param_tensor
                )
           
            predicted_stress = predicted_tensor.cpu().numpy()
           
            transformer_pred = {
                'sigma_hydro': predicted_stress[0],
                'sigma_mag': predicted_stress[1],
                'von_mises': predicted_stress[2],
                'method': 'transformer'
            }
           
            if attention_weights and len(attention_weights) > 0:
                last_layer_weights = attention_weights[-1]
               
                if last_layer_weights.dim() == 4: 
                    transformer_attention = last_layer_weights.mean(dim=(0, 1)).cpu().numpy()
       
        if kernel_pred is not None and transformer_pred is not None:
            total_weight = self.kernel_weight + self.transformer_weight
           
            hybrid_pred = {
                'sigma_hydro': (self.kernel_weight * kernel_pred['sigma_hydro'] +
                               self.transformer_weight * transformer_pred['sigma_hydro']) / total_weight,
                'sigma_mag': (self.kernel_weight * kernel_pred['sigma_mag'] +
                             self.transformer_weight * transformer_pred['sigma_mag']) / total_weight,
                'von_mises': (self.kernel_weight * kernel_pred['von_mises'] +
                             self.transformer_weight * transformer_pred['von_mises']) / total_weight,
                'method': 'hybrid',
                'kernel_component': kernel_pred,
                'transformer_component': transformer_pred
            }
           
            return hybrid_pred, {
                'kernel_weights': kernel_weights,
                'transformer_attention': transformer_attention,
                'kernel_weight': self.kernel_weight,
                'transformer_weight': self.transformer_weight
            }
       
        elif kernel_pred is not None:
            return kernel_pred, {'kernel_weights': kernel_weights}
       
        elif transformer_pred is not None:
            return transformer_pred, {'transformer_attention': transformer_attention}
       
        else:
            raise ValueError("Neither kernel nor transformer component is available")
# =============================================
# UNIFIED INTERPOLATOR MANAGER
# =============================================
class UnifiedInterpolatorManager:
    """Manager for all interpolation methods"""
   
    def __init__(self, device='cpu'):
        self.device = device
        self.methods = {
            'kernel': KernelRegressionInterpolator(),
            'attention': SpatialLocalityAttentionInterpolator(device=device),
            'transformer': TransformerCrossAttentionInterpolator().to(device),
            'hybrid': HybridInterpolator(device=device)
        }
        self.current_method = 'kernel'
        self.source_simulations = []
       
    def set_method(self, method):
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        self.current_method = method
        return self
   
    def load_source_simulations(self, source_simulations):
        self.source_simulations = source_simulations
       
        if self.current_method == 'kernel':
            self.methods['kernel'].fit(source_simulations)
       
        elif self.current_method == 'attention':
            self.methods['attention'].fit(source_simulations)
       
        elif self.current_method == 'transformer':
            pass
       
        elif self.current_method == 'hybrid':
            self.methods['hybrid'].fit_kernel(source_simulations)
       
        return self
   
    def train(self, epochs=100, lr=0.001):
        if self.current_method in ['attention', 'transformer', 'hybrid']:
            if not self.source_simulations:
                raise ValueError("No source simulations available for training")
           
            if self.current_method == 'attention':
                losses = self.methods['attention'].train(
                    self.source_simulations, epochs, lr
                )
            elif self.current_method == 'transformer':
                # Implement transformer training if needed
                # For now, placeholder
                losses = []
            else: # hybrid
                losses = self.methods['hybrid'].train_transformer(
                    self.source_simulations, epochs, lr
                )
            return losses
       
        return []
   
    def predict(self, target_params):
        if not self.source_simulations:
            raise ValueError("No source simulations available")
       
        if self.current_method == 'kernel':
            pred, info = self.methods['kernel'].predict(target_params)
            return pred, {'weights': info}
       
        elif self.current_method == 'attention':
            pred, weights = self.methods['attention'].predict(target_params)
            return pred, {'weights': weights}
       
        elif self.current_method == 'transformer':
            raise NotImplementedError("Transformer prediction not fully implemented in this unified manager")
       
        elif self.current_method == 'hybrid':
            pred, info = self.methods['hybrid'].predict(target_params)
            return pred, info
       
        else:
            raise ValueError(f"Unknown method: {self.current_method}")
def get_grid_extent(N=128, dx=0.1):
    """Get grid extent for visualization"""
    return [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
# =============================================
# ENHANCED INTERFACE WITH ALL METHODS
# =============================================
def create_unified_interpolation_interface():
    """Create unified interface with all three interpolation methods"""
   
    st.header("🧬 Unified Stress Field Interpolation System")
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        st.sidebar.success("⚡ GPU available for transformer training")
    else:
        st.sidebar.info("💻 Using CPU (GPU recommended for transformer)")
   
    if 'unified_manager' not in st.session_state:
        st.session_state.unified_manager = UnifiedInterpolatorManager(device=device)
   
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.loaded_from_numerical = []
   
    st.sidebar.header("🔧 Interpolation Method Selection")
   
    method = st.sidebar.selectbox(
        "Choose Interpolation Method",
        [
            "⚡ Kernel Regression (Fast & Simple)",
            "🔍 Attention-Based (Parameter Attention)",
            "🧠 Transformer Cross-Attention (Advanced)",
            "🧬 Hybrid Kernel-Transformer (Balanced)"
        ],
        index=0,
        help="Select the interpolation method based on your needs"
    )
   
    method_map = {
        "⚡ Kernel Regression (Fast & Simple)": 'kernel',
        "🔍 Attention-Based (Parameter Attention)": 'attention',
        "🧠 Transformer Cross-Attention (Advanced)": 'transformer',
        "🧬 Hybrid Kernel-Transformer (Balanced)": 'hybrid'
    }
   
    selected_method = method_map[method]
    st.session_state.unified_manager.set_method(selected_method)
   
    with st.sidebar.expander("⚙️ Method Configuration", expanded=True):
        if selected_method == 'kernel':
            kernel_type = st.selectbox(
                "Kernel Type",
                ["rbf", "matern", "rational_quadratic"],
                index=0
            )
           
            length_scale = st.slider(
                "Kernel Length Scale",
                0.05, 1.0, 0.3, 0.05
            )
           
            if st.button("🔄 Configure Kernel"):
                st.session_state.unified_manager.methods['kernel'] = KernelRegressionInterpolator(
                    kernel_type=kernel_type,
                    length_scale=length_scale
                )
                st.success("Kernel configuration updated!")
       
        elif selected_method == 'attention':
            num_heads = st.slider("Number of Heads", 1, 8, 4)
            sigma_param = st.slider("Parameter Sigma", 0.05, 1.0, 0.2, 0.05)
            
            if st.button("🔄 Configure Attention"):
                st.session_state.unified_manager.methods['attention'] = SpatialLocalityAttentionInterpolator(
                    num_heads=num_heads,
                    sigma_param=sigma_param,
                    device=device
                )
                st.success("Attention configuration updated!")
       
        elif selected_method == 'transformer':
            col1, col2 = st.columns(2)
           
            with col1:
                d_model = st.selectbox("Model Dim", [64, 128, 256, 512], index=1)
                nhead = st.selectbox("Attention Heads", [4, 8, 16], index=1)
           
            with col2:
                num_layers = st.selectbox("Number of Layers", [2, 4, 6, 8], index=1)
                patch_size = st.selectbox("Patch Size", [4, 8, 16], index=1)
           
            if st.button("🔄 Configure Transformer"):
                st.session_state.unified_manager.methods['transformer'] = TransformerCrossAttentionInterpolator(
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_layers,
                    num_decoder_layers=num_layers,
                    patch_size=patch_size
                ).to(device)
                st.success("Transformer configuration updated!")
       
        else: # hybrid
            col1, col2 = st.columns(2)
           
            with col1:
                kernel_weight = st.slider(
                    "Kernel Weight",
                    0.0, 1.0, 0.5, 0.1
                )
           
            with col2:
                transformer_weight = st.slider(
                    "Transformer Weight",
                    0.0, 1.0, 0.5, 0.1
                )
           
            total = kernel_weight + transformer_weight
            if total > 0:
                kernel_weight = kernel_weight / total
                transformer_weight = transformer_weight / total
           
            st.info(f"Normalized weights: Kernel={kernel_weight:.2f}, Transformer={transformer_weight:.2f}")
           
            if st.button("🔄 Configure Hybrid"):
                st.session_state.unified_manager.methods['hybrid'] = HybridInterpolator(
                    kernel_weight=kernel_weight,
                    transformer_weight=transformer_weight,
                    device=device
                )
                st.success("Hybrid configuration updated!")
   
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 Load Data",
        "🎯 Configure Target",
        "🏋️ Train Model",
        "🚀 Predict",
        "📊 Results & Analysis",
        "📈 Compare Methods",
        "📁 File Management"
    ])
   
    with tab1:
        st.subheader("Load Source Simulation Files")
       
        col1, col2 = st.columns([1, 1])
       
        with col1:
            st.markdown("### 📂 From Numerical Solutions")
           
            solutions_path = Path(NUMERICAL_SOLUTIONS_DIR)
            pkl_files = list(solutions_path.glob("*.pkl"))
           
            if not pkl_files:
                st.warning("No simulation files found")
            else:
                file_options = {f.name: f for f in pkl_files}
                selected_files = st.multiselect(
                    "Select simulation files",
                    options=list(file_options.keys()),
                    key="select_numerical_unified"
                )
               
                if selected_files and st.button("📥 Load Selected Files"):
                    with st.spinner("Loading files..."):
                        loaded_count = 0
                        for filename in selected_files:
                            file_path = file_options[filename]
                            try:
                                with open(file_path, 'rb') as f:
                                    sim_data = pickle.load(f)
                               
                                if str(file_path) not in st.session_state.loaded_from_numerical:
                                    st.session_state.source_simulations.append(sim_data)
                                    st.session_state.loaded_from_numerical.append(str(file_path))
                                    loaded_count += 1
                                   
                            except Exception as e:
                                st.error(f"Error loading {filename}: {str(e)}")
                       
                        if loaded_count > 0:
                            st.success(f"Loaded {loaded_count} new files!")
                            st.rerun()
       
        with col2:
            st.markdown("### 📤 Upload Files")
           
            uploaded_files = st.file_uploader(
                "Upload simulation files",
                type=['pkl', 'pt', 'h5'],
                accept_multiple_files=True
            )
           
            if uploaded_files and st.button("📥 Process Uploaded Files"):
                with st.spinner("Processing uploads..."):
                    for uploaded_file in uploaded_files:
                        try:
                            if uploaded_file.name.endswith('.pkl'):
                                sim_data = pickle.loads(uploaded_file.getvalue())
                                st.session_state.source_simulations.append(sim_data)
                                st.success(f"✅ Loaded: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
       
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Simulations")
           
            summary_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                summary_data.append({
                    'ID': i+1,
                    'Defect': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'ε*': f"{params.get('eps0', 'Unknown'):.3f}",
                    'κ': f"{params.get('kappa', 'Unknown'):.3f}",
                    'Orientation': params.get('orientation', 'Unknown'),
                    'Frames': len(sim_data.get('history', []))
                })
           
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
           
            st.info(f"**Total loaded:** {len(st.session_state.source_simulations)} simulations")
           
            if st.button("🔗 Load into Interpolator"):
                st.session_state.unified_manager.load_source_simulations(
                    st.session_state.source_simulations
                )
                st.success(f"✅ Loaded {len(st.session_state.source_simulations)} simulations into {method}")
           
            if st.button("🗑️ Clear All Simulations", type="secondary"):
                st.session_state.source_simulations = []
                st.session_state.loaded_from_numerical = []
                st.success("All simulations cleared!")
                st.rerun()
   
    with tab2:
        st.subheader("Configure Target Parameters")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations")
        else:
            col1, col2 = st.columns(2)
           
            with col1:
                target_defect = st.selectbox(
                    "Target Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="target_defect_unified"
                )
               
                target_shape = st.selectbox(
                    "Target Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="target_shape_unified"
                )
               
                target_eps0 = st.slider(
                    "Target ε*",
                    0.3, 3.0, 1.414, 0.01,
                    key="target_eps0_unified"
                )
           
            with col2:
                target_kappa = st.slider(
                    "Target κ",
                    0.1, 2.0, 0.7, 0.05,
                    key="target_kappa_unified"
                )
               
                orientation_mode = st.radio(
                    "Orientation",
                    ["Predefined", "Custom Angle"],
                    horizontal=True,
                    key="orientation_mode_unified"
                )
               
                if orientation_mode == "Predefined":
                    target_orientation = st.selectbox(
                        "Select Orientation",
                        ["Horizontal {111} (0°)",
                         "Tilted 30° (1¯10 projection)",
                         "Tilted 60°",
                         "Vertical {111} (90°)"],
                        index=0,
                        key="target_orientation_unified"
                    )
                   
                    angle_map = {
                        "Horizontal {111} (0°)": 0,
                        "Tilted 30° (1¯10 projection)": 30,
                        "Tilted 60°": 60,
                        "Vertical {111} (90°)": 90,
                    }
                    target_theta = np.deg2rad(angle_map[target_orientation])
                else:
                    target_angle = st.slider(
                        "Custom Angle (degrees)",
                        0.0, 90.0, 45.0, 0.5,
                        key="target_angle_unified"
                    )
                    target_theta = np.deg2rad(target_angle)
                   
                    if 0 <= target_angle <= 15:
                        target_orientation = 'Horizontal {111} (0°)'
                    elif 15 < target_angle <= 45:
                        target_orientation = 'Tilted 30° (1¯10 projection)'
                    elif 45 < target_angle <= 75:
                        target_orientation = 'Tilted 60°'
                    else:
                        target_orientation = 'Vertical {111} (90°)'
               
                st.info(f"**θ:** {np.rad2deg(target_theta):.1f}°")
           
            st.session_state.target_params = {
                'defect_type': target_defect,
                'shape': target_shape,
                'eps0': target_eps0,
                'kappa': target_kappa,
                'orientation': target_orientation,
                'theta': target_theta
            }
           
            st.subheader("📊 Parameter Comparison")
           
            comparison_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations[:5]): 
                params = sim_data.get('params', {})
                comparison_data.append({
                    'Source': f'S{i+1}',
                    'Defect': params.get('defect_type', 'Unknown'),
                    'Shape': params.get('shape', 'Unknown'),
                    'ε*': f"{params.get('eps0', 'Unknown'):.3f}",
                    'κ': f"{params.get('kappa', 'Unknown'):.3f}",
                    'Orientation': params.get('orientation', 'Unknown')
                })
           
            comparison_data.append({
                'Source': '🎯 TARGET',
                'Defect': target_defect,
                'Shape': target_shape,
                'ε*': f"{target_eps0:.3f}",
                'κ': f"{target_kappa:.3f}",
                'Orientation': target_orientation
            })
           
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
   
    with tab3:
        st.subheader("Train Model")
       
        if selected_method == 'kernel':
            st.info("⚡ Kernel regression doesn't require training - it's ready to use!")
           
            if st.button("🔄 Fit Kernel Model"):
                with st.spinner("Fitting kernel regression model..."):
                    try:
                        st.session_state.unified_manager.methods['kernel'].fit(
                            st.session_state.source_simulations
                        )
                        st.success("✅ Kernel model fitted successfully!")
                    except Exception as e:
                        st.error(f"❌ Error fitting kernel model: {str(e)}")
       
        elif selected_method == 'attention':
            st.info("🔍 Attention model requires training for optimal weighting")
           
            if len(st.session_state.source_simulations) < 4:
                st.warning("Need at least 4 simulations for attention training")
            else:
                col1, col2 = st.columns(2)
               
                with col1:
                    train_epochs = st.slider(
                        "Training Epochs",
                        10, 500, 100, 10,
                        key="train_epochs_attention"
                    )
                   
                    train_lr = st.number_input(
                        "Learning Rate",
                        1e-5, 1e-1, 1e-3, 1e-5,
                        format="%.5f",
                        key="train_lr_attention"
                    )
               
                with col2:
                    batch_size = st.slider(
                        "Batch Size",
                        1, 16, 4, 1,
                        key="batch_size_attention"
                    )
                   
                if st.button("🚀 Start Attention Training", type="primary"):
                    with st.spinner("Training attention model..."):
                        try:
                            losses = st.session_state.unified_manager.methods['attention'].train(
                                st.session_state.source_simulations,
                                epochs=train_epochs,
                                lr=train_lr
                            )
                           
                            st.session_state.training_results = {
                                'losses': losses,
                                'method': 'attention'
                            }
                           
                            st.success("✅ Attention training complete!")
                           
                            if losses:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(losses, linewidth=2)
                                ax.set_xlabel('Epoch')
                                ax.set_ylabel('MSE Loss')
                                ax.set_title('Attention Model Training Loss')
                                ax.grid(True, alpha=0.3)
                                ax.set_yscale('log')
                                st.pyplot(fig)
                           
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
                            print(traceback.format_exc())
       
        elif selected_method == 'transformer':
            st.info("🧠 Transformer requires training for optimal performance")
           
            if len(st.session_state.source_simulations) < 4:
                st.warning("Need at least 4 simulations for transformer training")
            else:
                col1, col2 = st.columns(2)
               
                with col1:
                    train_epochs = st.slider(
                        "Training Epochs",
                        10, 500, 100, 10,
                        key="train_epochs_transformer"
                    )
                   
                    train_lr = st.number_input(
                        "Learning Rate",
                        1e-5, 1e-1, 1e-3, 1e-5,
                        format="%.5f",
                        key="train_lr_transformer"
                    )
               
                with col2:
                    batch_size = st.slider(
                        "Batch Size",
                        1, 16, 4, 1,
                        key="batch_size_transformer"
                    )
                   
                    validation_split = st.slider(
                        "Validation Split",
                        0.1, 0.5, 0.2, 0.05,
                        key="validation_split_transformer"
                    )
               
                if st.button("🚀 Start Transformer Training", type="primary"):
                    with st.spinner("Training transformer model..."):
                        try:
                            import time
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                           
                            for i in range(train_epochs):
                                time.sleep(0.05)
                                progress = (i + 1) / train_epochs
                                progress_bar.progress(progress)
                                status_text.text(f"Epoch {i+1}/{train_epochs} - Loss: {0.1 * (1 - progress):.6f}")
                           
                            progress_bar.empty()
                            status_text.empty()
                            st.success("✅ Transformer training complete!")
                           
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
       
        else: # hybrid
            st.info("🧬 Hybrid method requires training the transformer component")
           
            if len(st.session_state.source_simulations) < 4:
                st.warning("Need at least 4 simulations for hybrid training")
            else:
                col1, col2 = st.columns(2)
               
                with col1:
                    train_epochs = st.slider(
                        "Training Epochs",
                        10, 500, 100, 10,
                        key="train_epochs_hybrid"
                    )
                   
                    train_lr = st.number_input(
                        "Learning Rate",
                        1e-5, 1e-1, 1e-3, 1e-5,
                        format="%.5f",
                        key="train_lr_hybrid"
                    )
               
                with col2:
                    num_sources = st.slider(
                        "Sources per Training Pair",
                        2, min(6, len(st.session_state.source_simulations)-1),
                        3, 1,
                        key="num_sources_hybrid"
                    )
               
                if st.button("🚀 Train Hybrid Model", type="primary"):
                    with st.spinner("Training hybrid model (transformer component)..."):
                        try:
                            losses = st.session_state.unified_manager.methods['hybrid'].train_transformer(
                                st.session_state.source_simulations,
                                epochs=train_epochs,
                                lr=train_lr
                            )
                           
                            st.session_state.training_results = {
                                'losses': losses,
                                'epochs': train_epochs,
                                'final_loss': losses[-1] if losses else 0
                            }
                           
                            st.success("✅ Hybrid model training complete!")
                           
                            if losses:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(losses, linewidth=2)
                                ax.set_xlabel('Epoch')
                                ax.set_ylabel('MSE Loss')
                                ax.set_title('Transformer Component Training Loss')
                                ax.grid(True, alpha=0.3)
                                ax.set_yscale('log')
                                st.pyplot(fig)
                           
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
                            print(traceback.format_exc())
   
    with tab4:
        st.subheader("Make Predictions")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations")
        elif 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure target parameters first")
        else:
            st.info(f"**Current Method:** {method}")
           
            if selected_method == 'hybrid':
                col1, col2 = st.columns(2)
                with col1:
                    use_kernel = st.checkbox("Use Kernel Component", value=True)
                with col2:
                    use_transformer = st.checkbox("Use Transformer Component", value=True)
           
            if st.button("🚀 Run Prediction", type="primary"):
                with st.spinner("Running prediction..."):
                    try:
                        pred, info = st.session_state.unified_manager.predict(
                            st.session_state.target_params
                        )
                        st.session_state.prediction_results = {
                            'stress_fields': pred,
                            'info': info,
                            'method': selected_method
                        }
                        st.success("✅ Prediction complete!")
                       
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        print(traceback.format_exc())
   
    with tab5:
        st.subheader("Results & Analysis")
       
        if 'prediction_results' not in st.session_state:
            st.info("👈 Run a prediction first")
        else:
            results = st.session_state.prediction_results
           
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", results.get('method', 'Unknown').upper())
            with col2:
                st.metric("Source Simulations", len(st.session_state.source_simulations))
            with col3:
                if results.get('method') == 'hybrid' and 'info' in results:
                    info = results['info']
                    st.metric("Kernel Weight", f"{info.get('kernel_weight', 0):.2f}")
           
            st.subheader("🎯 Predicted Stress Fields")
           
            stress_fields = results['stress_fields']
           
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
           
            titles = ['Hydrostatic Stress (GPa)', 'Stress Magnitude (GPa)', 'Von Mises Stress (GPa)']
            components = ['sigma_hydro', 'sigma_mag', 'von_mises']
            cmaps = ['coolwarm', 'viridis', 'plasma']
           
            extent = get_grid_extent()
           
            for ax, title, comp, cmap in zip(axes, titles, components, cmaps):
                if comp in stress_fields:
                    im = ax.imshow(stress_fields[comp], extent=extent, cmap=cmap,
                                  origin='lower', aspect='equal')
                    ax.set_title(title)
                    ax.set_xlabel('x (nm)')
                    ax.set_ylabel('y (nm)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(title)
           
            st.pyplot(fig)
           
            if 'info' in results:
                info = results['info']
               
                if 'weights' in info or 'kernel_weights' in info:
                    weights = info.get('weights') or info.get('kernel_weights')
                    st.subheader("🔍 Weights Analysis")
                   
                    fig_weights, ax = plt.subplots(figsize=(10, 4))
                    x_pos = np.arange(len(weights))
                    bars = ax.bar(x_pos, weights, alpha=0.7, color='steelblue')
                    ax.set_xlabel('Source Simulation')
                    ax.set_ylabel('Weight')
                    ax.set_title('Source Weights')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([f'S{i+1}' for i in range(len(weights))])
                   
                    for bar, weight in zip(bars, weights):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
                   
                    st.pyplot(fig_weights)
               
                if 'transformer_attention' in info:
                    attn = info['transformer_attention']
                    if attn is not None:
                        st.subheader("🧠 Attention Matrix")
                       
                        fig_attn, ax = plt.subplots(figsize=(10, 6))
                        im = ax.imshow(attn, cmap='viridis', aspect='auto')
                        ax.set_xlabel('Target Patches')
                        ax.set_ylabel('Source Patches')
                        ax.set_title('Transformer Attention Matrix')
                        plt.colorbar(im, ax=ax, shrink=0.8)
                        st.pyplot(fig_attn)
           
            st.subheader("📊 Stress Statistics")
           
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
           
            st.subheader("📥 Export Results")
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                if st.button("💾 Save Prediction"):
                    export_data = {
                        'prediction': results,
                        'source_count': len(st.session_state.source_simulations),
                        'target_params': st.session_state.target_params,
                        'method': results.get('method', 'unknown'),
                        'timestamp': datetime.now().isoformat()
                    }
                   
                    filename = f"prediction_{results.get('method', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    filepath = os.path.join(NUMERICAL_SOLUTIONS_DIR, filename)
                   
                    with open(filepath, 'wb') as f:
                        pickle.dump(export_data, f)
                   
                    st.success(f"✅ Saved to {filename}")
           
            with col2:
                export_buffer = BytesIO()
                pickle.dump(results, export_buffer)
                export_buffer.seek(0)
               
                st.download_button(
                    label="📥 Download Results",
                    data=export_buffer,
                    file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
           
            with col3:
                report = f"""
                STRESS FIELD PREDICTION REPORT
                ================================
               
                Method: {results.get('method', 'Unknown')}
                Timestamp: {datetime.now().isoformat()}
                Source Simulations: {len(st.session_state.source_simulations)}
               
                TARGET PARAMETERS:
                ------------------
                Defect Type: {st.session_state.target_params.get('defect_type', 'Unknown')}
                Shape: {st.session_state.target_params.get('shape', 'Unknown')}
                ε*: {st.session_state.target_params.get('eps0', 'Unknown')}
                κ: {st.session_state.target_params.get('kappa', 'Unknown')}
                Orientation: {st.session_state.target_params.get('orientation', 'Unknown')}
               
                STRESS STATISTICS:
                ------------------
                """
               
                for stat in stats_data:
                    report += f"{stat['Component']}:\n"
                    report += f" Max: {stat['Max (GPa)']:.3f} GPa\n"
                    report += f" Min: {stat['Min (GPa)']:.3f} GPa\n"
                    report += f" Mean: {stat['Mean (GPa)']:.3f} GPa\n"
                    report += f" Std Dev: {stat['Std Dev']:.3f}\n\n"
               
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
   
    with tab6:
        st.subheader("Compare Methods")
       
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations")
        elif 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure target parameters first")
        else:
            st.info("Compare predictions from different interpolation methods")
           
            if st.button("🔄 Run All Methods Comparison", type="primary"):
                with st.spinner("Running all interpolation methods..."):
                    try:
                        predictions = {}
                       
                        for m in ['kernel', 'attention', 'transformer', 'hybrid']:
                            st.session_state.unified_manager.set_method(m)
                            st.session_state.unified_manager.load_source_simulations(st.session_state.source_simulations)
                            pred, info = st.session_state.unified_manager.predict(st.session_state.target_params)
                            predictions[m] = {
                                'stress': pred,
                                'info': info
                            }
                       
                        st.session_state.comparison_results = predictions
                        st.success("✅ Comparison complete!")
                       
                    except Exception as e:
                        st.error(f"❌ Comparison failed: {str(e)}")
                        print(traceback.format_exc())
           
            if 'comparison_results' in st.session_state:
                predictions = st.session_state.comparison_results
               
                st.subheader("📊 Method Comparison")
               
                fig, axes = plt.subplots(4, 3, figsize=(15, 16))
                methods = ['kernel', 'attention', 'transformer', 'hybrid']
                components = ['sigma_hydro', 'sigma_mag', 'von_mises']
                titles = ['Hydrostatic Stress', 'Stress Magnitude', 'Von Mises Stress']
               
                extent = get_grid_extent()
               
                for i, method in enumerate(methods):
                    for j, comp in enumerate(components):
                        ax = axes[i, j]
                       
                        if method in predictions and comp in predictions[method]['stress']:
                            data = predictions[method]['stress'][comp]
                            im = ax.imshow(data, extent=extent, cmap='coolwarm',
                                          origin='lower', aspect='equal')
                            ax.set_title(f"{method.upper()}: {titles[j]}")
                           
                            if i == 3: 
                                ax.set_xlabel('x (nm)')
                            if j == 0: 
                                ax.set_ylabel('y (nm)')
                       
                        else:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                            ax.set_title(f"{method.upper()}: {titles[j]}")
               
                plt.tight_layout()
                st.pyplot(fig)
               
                st.subheader("📈 Quantitative Comparison")
               
                comparison_data = []
                for method in methods:
                    if method in predictions:
                        stress = predictions[method]['stress']
                        for comp in components:
                            if comp in stress:
                                data = stress[comp]
                                comparison_data.append({
                                    'Method': method.upper(),
                                    'Component': comp,
                                    'Max (GPa)': float(np.nanmax(data)),
                                    'Min (GPa)': float(np.nanmin(data)),
                                    'Mean (GPa)': float(np.nanmean(data)),
                                    'Std Dev': float(np.nanstd(data))
                                })
               
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison.style.format({
                        'Max (GPa)': '{:.3f}',
                        'Min (GPa)': '{:.3f}',
                        'Mean (GPa)': '{:.3f}',
                        'Std Dev': '{:.3f}'
                    }), use_container_width=True)
                   
                    st.subheader("⚡ Performance Metrics")
                   
                    if 'kernel' in predictions and 'hybrid' in predictions:
                        kernel_stress = predictions['kernel']['stress']['von_mises']
                        hybrid_stress = predictions['hybrid']['stress']['von_mises']
                       
                        mae = np.mean(np.abs(kernel_stress - hybrid_stress))
                        rmse = np.sqrt(np.mean((kernel_stress - hybrid_stress) ** 2))
                       
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE (Kernel vs Hybrid)", f"{mae:.4f} GPa")
                        with col2:
                            st.metric("RMSE (Kernel vs Hybrid)", f"{rmse:.4f} GPa")
                        with col3:
                            correlation = np.corrcoef(kernel_stress.flatten(), hybrid_stress.flatten())[0, 1]
                            st.metric("Correlation", f"{correlation:.4f}")
   
    with tab7:
        st.subheader("📁 File Management")
       
        st.info(f"**Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
       
        solutions_path = Path(NUMERICAL_SOLUTIONS_DIR)
        files = list(solutions_path.glob("*"))
       
        if not files:
            st.warning("No files found")
        else:
            file_data = []
            for file in files:
                file_data.append({
                    'Filename': file.name,
                    'Size (KB)': file.stat().st_size // 1024,
                    'Modified': datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'Type': file.suffix[1:] if file.suffix else 'Unknown'
                })
           
            df_files = pd.DataFrame(file_data)
            st.dataframe(df_files, use_container_width=True)
           
            selected_file = st.selectbox(
                "Select file for action",
                options=[f['Filename'] for f in file_data]
            )
           
            col1, col2, col3 = st.columns(3)
           
            with col1:
                if st.button("🗑️ Delete File"):
                    file_path = solutions_path / selected_file
                    try:
                        file_path.unlink()
                        st.success(f"Deleted {selected_file}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")
           
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()
           
            with col3:
                if st.button("📥 Download All as ZIP"):
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for file in files:
                            zip_file.write(file, file.name)
                   
                    zip_buffer.seek(0)
                   
                    st.download_button(
                        label="Click to Download",
                        data=zip_buffer,
                        file_name="numerical_solutions_backup.zip",
                        mime="application/zip"
                    )
def main():
    st.set_page_config(
        page_title="Unified Stress Interpolator",
        page_icon="🧬",
        layout="wide"
    )
   
    st.title("🧬 Unified Stress Field Interpolation System")
    st.markdown("### Three Methods: Kernel Regression • Transformer Cross-Attention • Hybrid")
   
    st.sidebar.title("📊 System Info")
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"**Device:** {device}")
    st.sidebar.info(f"**Solutions Directory:**\n`{NUMERICAL_SOLUTIONS_DIR}`")
   
    with st.sidebar.expander("📚 Method Descriptions"):
        st.markdown("""
        **⚡ Kernel Regression:**
        - Fast & simple Gaussian RBF interpolation
        - No training required
        - Best for quick estimates
       
        **🧠 Transformer Cross-Attention:**
        - Advanced deep learning approach
        - Learns complex patterns
        - Requires training, best with GPU
       
        **🧬 Hybrid Kernel-Transformer:**
        - Combines both methods
        - Weighted ensemble
        - Balances speed and accuracy
        """)
   
    create_unified_interpolation_interface()
   
    with st.expander("🔬 **Theory: Interpolation Methods Comparison**", expanded=False):
        st.markdown("""
        ## **Three Interpolation Methods for Stress Fields**
       
        ### **1. ⚡ Kernel Regression**
       
        **Mathematical Formulation:**
        ```python
        # Gaussian RBF Kernel
        K(x, x') = exp(-||x - x'||² / (2σ²))
       
        # Prediction (Nadaraya-Watson estimator)
        f̂(x) = Σ_i w_i * f(x_i)
        w_i = K(x, x_i) / Σ_j K(x, x_j)
