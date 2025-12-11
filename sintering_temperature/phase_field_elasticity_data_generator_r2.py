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
warnings.filterwarnings('ignore')

# NEW IMPORTS FOR DATA EXPORT
import pickle
import sqlite3
import torch
import h5py
import msgpack
import tempfile
import os
from pathlib import Path

# Configure page with better styling
st.set_page_config(page_title="Ag NP Multi-Defect Analyzer", layout="wide")
st.title("🔬 Ag Nanoparticle Multi-Defect Comparison Platform")
st.markdown("""
**Run multiple simulations • Compare ISF/ESF/Twin with different orientations • Cloud-style storage**
**Run → Save → Compare • 50+ Colormaps • Publication-ready comparison plots • Advanced Post-Processing**
**NEW: Machine Learning-Ready Data Export • Multiple Formats (PyTorch, SQLite, HDF5, Pickle) • GitHub Repository Compatible**
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
dx = 0.1 # nm
extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
X, Y = np.meshgrid(np.linspace(extent[0], extent[1], N),
                   np.linspace(extent[2], extent[3], N))

# =============================================
# ENHANCED SIMULATION DATA STRUCTURE
# =============================================
class SimulationDataStructure:
    """Structured data container for ML-ready simulation data"""
    
    @staticmethod
    def create_simulation_data(sim_id, params, history, metadata, grid_params):
        """
        Create structured simulation data for ML consumption
        
        Returns:
        --------
        dict: Structured data with the following keys:
            - metadata: Simulation metadata
            - parameters: All simulation parameters
            - grid: Grid configuration
            - frames: List of frame data
            - statistics: Derived statistics
        """
        # Create structured frame data
        frames_data = []
        for frame_idx, (eta, stress_fields) in enumerate(history):
            frame_data = {
                'frame_index': frame_idx,
                'time_step': frame_idx * params.get('dt', 0.004),
                'defect_field': eta.astype(np.float32),
                'stress_components': {
                    'sxx': stress_fields['sxx'].astype(np.float32),
                    'syy': stress_fields['syy'].astype(np.float32),
                    'sxy': stress_fields['sxy'].astype(np.float32),
                    'szz': stress_fields['szz'].astype(np.float32),
                    'sigma_mag': stress_fields['sigma_mag'].astype(np.float32),
                    'sigma_hydro': stress_fields['sigma_hydro'].astype(np.float32),
                    'von_mises': stress_fields['von_mises'].astype(np.float32)
                },
                'strain_components': SimulationDataStructure.calculate_strain_components(
                    eta, stress_fields, params
                ),
                'derived_quantities': {
                    'defect_volume': np.sum(eta > 0.5) * (dx**2),
                    'avg_stress_magnitude': np.mean(stress_fields['sigma_mag']),
                    'max_stress_magnitude': np.max(stress_fields['sigma_mag']),
                    'stress_gradient': SimulationDataStructure.calculate_stress_gradient(
                        stress_fields['sigma_mag'], dx
                    )
                }
            }
            frames_data.append(frame_data)
        
        # Calculate global statistics
        statistics = SimulationDataStructure.calculate_simulation_statistics(frames_data)
        
        # Create complete data structure
        structured_data = {
            'simulation_id': sim_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'run_time': metadata.get('run_time', 0),
                'total_frames': len(history),
                'grid_size': N,
                'grid_spacing': dx,
                'total_grid_points': N * N,
                'simulation_version': '1.0'
            },
            'parameters': {
                'defect_type': params['defect_type'],
                'defect_type_code': SimulationDataStructure.defect_type_to_code(params['defect_type']),
                'seed_shape': params['shape'],
                'seed_shape_code': SimulationDataStructure.shape_to_code(params['shape']),
                'habit_plane_angle': float(params.get('theta', 0)),
                'habit_plane_orientation': params['orientation'],
                'eigenstrain_magnitude': float(params['eps0']),
                'interface_energy_coefficient': float(params['kappa']),
                'total_steps': params['steps'],
                'save_frequency': params['save_every'],
                'physical_constants': {
                    'lattice_parameter': a,
                    'elastic_C11': C11,
                    'elastic_C12': C12,
                    'elastic_C44': C44
                },
                'simulation_parameters': {
                    'dt': 0.004,
                    'dx': dx,
                    'N': N
                }
            },
            'grid': {
                'X': X.astype(np.float32),
                'Y': Y.astype(np.float32),
                'dx': dx,
                'extent': extent,
                'grid_shape': (N, N)
            },
            'frames': frames_data,
            'statistics': statistics,
            'line_profiles': SimulationDataStructure.extract_line_profiles(frames_data, grid_params)
        }
        
        return structured_data
    
    @staticmethod
    def defect_type_to_code(defect_type):
        """Convert defect type to numeric code"""
        codes = {'ISF': 0, 'ESF': 1, 'Twin': 2}
        return codes.get(defect_type, -1)
    
    @staticmethod
    def shape_to_code(shape):
        """Convert shape to numeric code"""
        codes = {'Square': 0, 'Horizontal Fault': 1, 'Vertical Fault': 2, 
                'Rectangle': 3, 'Ellipse': 4}
        return codes.get(shape, -1)
    
    @staticmethod
    def calculate_strain_components(eta, stress_fields, params):
        """Calculate strain components from stress"""
        # Plane-strain reduced constants (Pa)
        C11_p = (C11 - C12**2 / C11) * 1e9
        C12_p = (C12 - C12**2 / C11) * 1e9
        C44_p = C44 * 1e9
        
        # Inverse of stiffness matrix for plane strain
        S11 = C11_p / (C11_p**2 - C12_p**2)
        S12 = -C12_p / (C11_p**2 - C12_p**2)
        S44 = 1.0 / (2 * C44_p)
        
        # Convert stress from GPa to Pa for strain calculation
        sxx = stress_fields['sxx'] * 1e9
        syy = stress_fields['syy'] * 1e9
        sxy = stress_fields['sxy'] * 1e9
        
        # Calculate elastic strains
        exx = S11 * sxx + S12 * syy
        eyy = S12 * sxx + S11 * syy
        exy = S44 * sxy
        
        return {
            'exx': exx.astype(np.float32),
            'eyy': eyy.astype(np.float32),
            'exy': exy.astype(np.float32),
            'volumetric_strain': (exx + eyy).astype(np.float32),
            'deviatoric_strain': np.sqrt(((exx - eyy)**2 + 4*exy**2)/2).astype(np.float32)
        }
    
    @staticmethod
    def calculate_stress_gradient(stress_field, dx):
        """Calculate stress gradient magnitude"""
        grad_y, grad_x = np.gradient(stress_field, dx)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return {
            'grad_x': grad_x.astype(np.float32),
            'grad_y': grad_y.astype(np.float32),
            'magnitude': gradient_magnitude.astype(np.float32)
        }
    
    @staticmethod
    def calculate_simulation_statistics(frames_data):
        """Calculate comprehensive statistics across all frames"""
        if not frames_data:
            return {}
        
        # Collect statistics across frames
        defect_volumes = []
        avg_stresses = []
        max_stresses = []
        stress_gradients = []
        
        for frame in frames_data:
            defect_volumes.append(frame['derived_quantities']['defect_volume'])
            avg_stresses.append(frame['derived_quantities']['avg_stress_magnitude'])
            max_stresses.append(frame['derived_quantities']['max_stress_magnitude'])
            stress_gradients.append(np.mean(frame['derived_quantities']['stress_gradient']['magnitude']))
        
        return {
            'defect_volume': {
                'mean': float(np.mean(defect_volumes)),
                'std': float(np.std(defect_volumes)),
                'min': float(np.min(defect_volumes)),
                'max': float(np.max(defect_volumes)),
                'evolution': np.array(defect_volumes, dtype=np.float32)
            },
            'stress_magnitude': {
                'mean': float(np.mean(avg_stresses)),
                'std': float(np.std(avg_stresses)),
                'min': float(np.min(avg_stresses)),
                'max': float(np.max(avg_stresses)),
                'evolution': np.array(avg_stresses, dtype=np.float32)
            },
            'peak_stress': {
                'mean': float(np.mean(max_stresses)),
                'std': float(np.std(max_stresses)),
                'min': float(np.min(max_stresses)),
                'max': float(np.max(max_stresses)),
                'evolution': np.array(max_stresses, dtype=np.float32)
            },
            'stress_gradient': {
                'mean': float(np.mean(stress_gradients)),
                'std': float(np.std(stress_gradients)),
                'evolution': np.array(stress_gradients, dtype=np.float32)
            }
        }
    
    @staticmethod
    def extract_line_profiles(frames_data, grid_params):
        """Extract line profiles for ML training"""
        if not frames_data:
            return {}
        
        # Use final frame for line profiles
        final_frame = frames_data[-1]
        sigma_mag = final_frame['stress_components']['sigma_mag']
        
        profiles = {}
        
        # Horizontal profile through center
        center_idx = sigma_mag.shape[0] // 2
        profiles['horizontal'] = {
            'position': center_idx,
            'distance': np.linspace(grid_params['extent'][0], grid_params['extent'][1], 
                                   sigma_mag.shape[1]).astype(np.float32),
            'stress': sigma_mag[center_idx, :].astype(np.float32),
            'defect': final_frame['defect_field'][center_idx, :].astype(np.float32)
        }
        
        # Vertical profile through center
        profiles['vertical'] = {
            'position': center_idx,
            'distance': np.linspace(grid_params['extent'][2], grid_params['extent'][3],
                                   sigma_mag.shape[0]).astype(np.float32),
            'stress': sigma_mag[:, center_idx].astype(np.float32),
            'defect': final_frame['defect_field'][:, center_idx].astype(np.float32)
        }
        
        # Diagonal profile
        diag_length = min(sigma_mag.shape)
        diag_indices = np.arange(diag_length)
        profiles['diagonal'] = {
            'distance': np.linspace(0, np.sqrt(2)*diag_length*dx, diag_length).astype(np.float32),
            'stress': sigma_mag[diag_indices, diag_indices].astype(np.float32),
            'defect': final_frame['defect_field'][diag_indices, diag_indices].astype(np.float32)
        }
        
        return profiles

# =============================================
# MULTI-FORMAT DATA EXPORTER
# =============================================
class DataExporter:
    """Export simulation data in multiple ML-ready formats"""
    
    @staticmethod
    def export_pytorch(simulation_data, filepath=None):
        """Export simulation data as PyTorch tensor dictionary"""
        # Convert numpy arrays to torch tensors
        torch_data = {}
        
        # Metadata and parameters
        torch_data['metadata'] = simulation_data['metadata']
        torch_data['parameters'] = simulation_data['parameters']
        
        # Grid data
        torch_data['grid_X'] = torch.from_numpy(simulation_data['grid']['X'])
        torch_data['grid_Y'] = torch.from_numpy(simulation_data['grid']['Y'])
        
        # Collect all frames into tensors
        n_frames = len(simulation_data['frames'])
        n, m = simulation_data['grid']['grid_shape']
        
        # Initialize tensors for all frames
        defect_fields = torch.zeros((n_frames, n, m), dtype=torch.float32)
        stress_components = {}
        for key in simulation_data['frames'][0]['stress_components'].keys():
            stress_components[key] = torch.zeros((n_frames, n, m), dtype=torch.float32)
        
        strain_components = {}
        for key in simulation_data['frames'][0]['strain_components'].keys():
            strain_components[key] = torch.zeros((n_frames, n, m), dtype=torch.float32)
        
        # Fill tensors
        for i, frame in enumerate(simulation_data['frames']):
            defect_fields[i] = torch.from_numpy(frame['defect_field'])
            for key, value in frame['stress_components'].items():
                stress_components[key][i] = torch.from_numpy(value)
            for key, value in frame['strain_components'].items():
                strain_components[key][i] = torch.from_numpy(value)
        
        torch_data['defect_fields'] = defect_fields
        torch_data['stress_components'] = stress_components
        torch_data['strain_components'] = strain_components
        
        # Line profiles
        torch_data['line_profiles'] = {}
        for profile_name, profile_data in simulation_data['line_profiles'].items():
            torch_data['line_profiles'][profile_name] = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in profile_data.items()
            }
        
        # Statistics
        torch_data['statistics'] = simulation_data['statistics']
        
        # Save to file or return bytes
        if filepath:
            torch.save(torch_data, filepath)
            return filepath
        else:
            buffer = BytesIO()
            torch.save(torch_data, buffer)
            buffer.seek(0)
            return buffer
    
    @staticmethod
    def export_sqlite(simulation_data, filepath=None):
        """Export simulation data to SQLite database"""
        if filepath is None:
            filepath = ':memory:'
        
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_metadata (
                sim_id TEXT PRIMARY KEY,
                defect_type TEXT,
                defect_type_code INTEGER,
                seed_shape TEXT,
                seed_shape_code INTEGER,
                habit_plane_angle REAL,
                eigenstrain_magnitude REAL,
                interface_energy_coeff REAL,
                total_steps INTEGER,
                total_frames INTEGER,
                grid_size INTEGER,
                grid_spacing REAL,
                timestamp TEXT,
                simulation_version TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id TEXT,
                frame_index INTEGER,
                time_step REAL,
                defect_volume REAL,
                avg_stress_magnitude REAL,
                max_stress_magnitude REAL,
                defect_field BLOB,
                stress_sxx BLOB,
                stress_syy BLOB,
                stress_sxy BLOB,
                stress_sigma_mag BLOB,
                stress_sigma_hydro BLOB,
                stress_von_mises BLOB,
                strain_exx BLOB,
                strain_eyy BLOB,
                strain_exy BLOB,
                strain_volumetric BLOB,
                strain_deviatoric BLOB,
                FOREIGN KEY (sim_id) REFERENCES simulation_metadata (sim_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS line_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id TEXT,
                profile_type TEXT,
                distance BLOB,
                stress_profile BLOB,
                defect_profile BLOB,
                FOREIGN KEY (sim_id) REFERENCES simulation_metadata (sim_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_statistics (
                sim_id TEXT PRIMARY KEY,
                defect_volume_mean REAL,
                defect_volume_std REAL,
                stress_mean REAL,
                stress_std REAL,
                peak_stress_mean REAL,
                peak_stress_std REAL,
                FOREIGN KEY (sim_id) REFERENCES simulation_metadata (sim_id)
            )
        ''')
        
        # Insert simulation metadata
        metadata = simulation_data['metadata']
        params = simulation_data['parameters']
        
        cursor.execute('''
            INSERT INTO simulation_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            simulation_data['simulation_id'],
            params['defect_type'],
            params['defect_type_code'],
            params['seed_shape'],
            params['seed_shape_code'],
            params['habit_plane_angle'],
            params['eigenstrain_magnitude'],
            params['interface_energy_coefficient'],
            params['total_steps'],
            metadata['total_frames'],
            metadata['grid_size'],
            metadata['grid_spacing'],
            simulation_data['timestamp'],
            metadata['simulation_version']
        ))
        
        # Insert frames data
        for frame in simulation_data['frames']:
            # Convert arrays to bytes
            defect_bytes = frame['defect_field'].tobytes()
            sxx_bytes = frame['stress_components']['sxx'].tobytes()
            syy_bytes = frame['stress_components']['syy'].tobytes()
            sxy_bytes = frame['stress_components']['sxy'].tobytes()
            sigma_mag_bytes = frame['stress_components']['sigma_mag'].tobytes()
            sigma_hydro_bytes = frame['stress_components']['sigma_hydro'].tobytes()
            von_mises_bytes = frame['stress_components']['von_mises'].tobytes()
            
            exx_bytes = frame['strain_components']['exx'].tobytes()
            eyy_bytes = frame['strain_components']['eyy'].tobytes()
            exy_bytes = frame['strain_components']['exy'].tobytes()
            vol_bytes = frame['strain_components']['volumetric_strain'].tobytes()
            dev_bytes = frame['strain_components']['deviatoric_strain'].tobytes()
            
            cursor.execute('''
                INSERT INTO simulation_frames 
                (sim_id, frame_index, time_step, defect_volume, avg_stress_magnitude, 
                 max_stress_magnitude, defect_field, stress_sxx, stress_syy, stress_sxy,
                 stress_sigma_mag, stress_sigma_hydro, stress_von_mises,
                 strain_exx, strain_eyy, strain_exy, strain_volumetric, strain_deviatoric)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                simulation_data['simulation_id'],
                frame['frame_index'],
                frame['time_step'],
                frame['derived_quantities']['defect_volume'],
                frame['derived_quantities']['avg_stress_magnitude'],
                frame['derived_quantities']['max_stress_magnitude'],
                sqlite3.Binary(defect_bytes),
                sqlite3.Binary(sxx_bytes),
                sqlite3.Binary(syy_bytes),
                sqlite3.Binary(sxy_bytes),
                sqlite3.Binary(sigma_mag_bytes),
                sqlite3.Binary(sigma_hydro_bytes),
                sqlite3.Binary(von_mises_bytes),
                sqlite3.Binary(exx_bytes),
                sqlite3.Binary(eyy_bytes),
                sqlite3.Binary(exy_bytes),
                sqlite3.Binary(vol_bytes),
                sqlite3.Binary(dev_bytes)
            ))
        
        # Insert line profiles
        for profile_name, profile_data in simulation_data['line_profiles'].items():
            cursor.execute('''
                INSERT INTO line_profiles (sim_id, profile_type, distance, stress_profile, defect_profile)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                simulation_data['simulation_id'],
                profile_name,
                sqlite3.Binary(profile_data['distance'].tobytes()),
                sqlite3.Binary(profile_data['stress'].tobytes()),
                sqlite3.Binary(profile_data['defect'].tobytes())
            ))
        
        # Insert statistics
        stats = simulation_data['statistics']
        cursor.execute('''
            INSERT INTO simulation_statistics 
            (sim_id, defect_volume_mean, defect_volume_std, stress_mean, stress_std, peak_stress_mean, peak_stress_std)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            simulation_data['simulation_id'],
            stats['defect_volume']['mean'],
            stats['defect_volume']['std'],
            stats['stress_magnitude']['mean'],
            stats['stress_magnitude']['std'],
            stats['peak_stress']['mean'],
            stats['peak_stress']['std']
        ))
        
        conn.commit()
        
        if filepath == ':memory:':
            # Save in-memory database to bytes
            buffer = BytesIO()
            for line in conn.iterdump():
                buffer.write(f'{line}\n'.encode('utf-8'))
            buffer.seek(0)
            conn.close()
            return buffer
        else:
            conn.close()
            return filepath
    
    @staticmethod
    def export_hdf5(simulation_data, filepath=None):
        """Export simulation data to HDF5 format"""
        if filepath is None:
            filepath = tempfile.NamedTemporaryFile(suffix='.h5', delete=False).name
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            metadata_group = f.create_group('metadata')
            parameters_group = f.create_group('parameters')
            grid_group = f.create_group('grid')
            frames_group = f.create_group('frames')
            profiles_group = f.create_group('line_profiles')
            stats_group = f.create_group('statistics')
            
            # Store metadata
            for key, value in simulation_data['metadata'].items():
                if isinstance(value, (str, int, float)):
                    metadata_group.attrs[key] = value
            
            metadata_group.attrs['simulation_id'] = simulation_data['simulation_id']
            metadata_group.attrs['timestamp'] = simulation_data['timestamp']
            
            # Store parameters
            for key, value in simulation_data['parameters'].items():
                if isinstance(value, dict):
                    param_subgroup = parameters_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (str, int, float)):
                            param_subgroup.attrs[subkey] = subvalue
                elif isinstance(value, (str, int, float)):
                    parameters_group.attrs[key] = value
            
            # Store grid
            grid_group.create_dataset('X', data=simulation_data['grid']['X'])
            grid_group.create_dataset('Y', data=simulation_data['grid']['Y'])
            grid_group.attrs['dx'] = simulation_data['grid']['dx']
            grid_group.attrs['extent'] = simulation_data['grid']['extent']
            
            # Store frames
            for i, frame in enumerate(simulation_data['frames']):
                frame_group = frames_group.create_group(f'frame_{i:04d}')
                frame_group.attrs['frame_index'] = frame['frame_index']
                frame_group.attrs['time_step'] = frame['time_step']
                
                # Store defect field
                frame_group.create_dataset('defect_field', data=frame['defect_field'])
                
                # Store stress components
                stress_group = frame_group.create_group('stress_components')
                for key, value in frame['stress_components'].items():
                    stress_group.create_dataset(key, data=value)
                
                # Store strain components
                strain_group = frame_group.create_group('strain_components')
                for key, value in frame['strain_components'].items():
                    strain_group.create_dataset(key, data=value)
                
                # Store derived quantities
                derived_group = frame_group.create_group('derived_quantities')
                for key, value in frame['derived_quantities'].items():
                    if isinstance(value, (int, float)):
                        derived_group.attrs[key] = value
                    elif isinstance(value, dict):
                        # For stress gradient
                        grad_group = derived_group.create_group(key)
                        for subkey, subvalue in value.items():
                            grad_group.create_dataset(subkey, data=subvalue)
            
            # Store line profiles
            for profile_name, profile_data in simulation_data['line_profiles'].items():
                profile_group = profiles_group.create_group(profile_name)
                for key, value in profile_data.items():
                    if isinstance(value, np.ndarray):
                        profile_group.create_dataset(key, data=value)
                    else:
                        profile_group.attrs[key] = value
            
            # Store statistics
            for stat_name, stat_data in simulation_data['statistics'].items():
                stat_group = stats_group.create_group(stat_name)
                for key, value in stat_data.items():
                    if isinstance(value, (int, float)):
                        stat_group.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        stat_group.create_dataset(key, data=value)
        
        if filepath.startswith(tempfile.gettempdir()):
            # Read file into bytes and delete temp file
            with open(filepath, 'rb') as f:
                buffer = BytesIO(f.read())
            os.unlink(filepath)
            buffer.seek(0)
            return buffer
        else:
            return filepath
    
    @staticmethod
    def export_pickle(simulation_data, filepath=None):
        """Export simulation data using pickle"""
        # Optimize data for pickling (remove large arrays if needed)
        optimized_data = DataExporter._optimize_for_pickle(simulation_data)
        
        if filepath:
            with open(filepath, 'wb') as f:
                pickle.dump(optimized_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return filepath
        else:
            buffer = BytesIO()
            pickle.dump(optimized_data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            buffer.seek(0)
            return buffer
    
    @staticmethod
    def export_npz(simulation_data, filepath=None):
        """Export simulation data as compressed numpy arrays"""
        if filepath is None:
            filepath = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name
        
        # Prepare data for npz
        save_dict = {}
        
        # Store metadata as JSON string
        save_dict['metadata'] = json.dumps(simulation_data['metadata'])
        save_dict['parameters'] = json.dumps(simulation_data['parameters'])
        
        # Store grid
        save_dict['grid_X'] = simulation_data['grid']['X']
        save_dict['grid_Y'] = simulation_data['grid']['Y']
        
        # Store all frames' defect fields as 3D array
        n_frames = len(simulation_data['frames'])
        n, m = simulation_data['grid']['grid_shape']
        
        defect_fields = np.zeros((n_frames, n, m), dtype=np.float32)
        stress_sigma_mag = np.zeros((n_frames, n, m), dtype=np.float32)
        
        for i, frame in enumerate(simulation_data['frames']):
            defect_fields[i] = frame['defect_field']
            stress_sigma_mag[i] = frame['stress_components']['sigma_mag']
        
        save_dict['defect_fields'] = defect_fields
        save_dict['stress_sigma_mag'] = stress_sigma_mag
        
        # Store final frame stress components
        final_frame = simulation_data['frames'][-1]
        for key, value in final_frame['stress_components'].items():
            save_dict[f'final_{key}'] = value
        
        # Store line profiles
        for profile_name, profile_data in simulation_data['line_profiles'].items():
            for key, value in profile_data.items():
                if isinstance(value, np.ndarray):
                    save_dict[f'profile_{profile_name}_{key}'] = value
        
        # Save
        np.savez_compressed(filepath, **save_dict)
        
        if filepath.startswith(tempfile.gettempdir()):
            # Read file into bytes and delete temp file
            with open(filepath, 'rb') as f:
                buffer = BytesIO(f.read())
            os.unlink(filepath)
            buffer.seek(0)
            return buffer
        else:
            return filepath
    
    @staticmethod
    def export_msgpack(simulation_data, filepath=None):
        """Export simulation data using MessagePack (compact binary format)"""
        # Convert numpy arrays to lists for msgpack
        def convert_for_msgpack(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_msgpack(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_msgpack(item) for item in obj]
            else:
                return obj
        
        packed_data = convert_for_msgpack(simulation_data)
        
        if filepath:
            with open(filepath, 'wb') as f:
                msgpack.pack(packed_data, f)
            return filepath
        else:
            buffer = BytesIO()
            msgpack.pack(packed_data, buffer)
            buffer.seek(0)
            return buffer
    
    @staticmethod
    def _optimize_for_pickle(data):
        """Optimize data structure for pickling"""
        optimized = data.copy()
        
        # For pickling, we might want to reduce data size
        # Keep only final frame and statistics for smaller files
        if len(optimized['frames']) > 10:
            # Keep only every 10th frame and the final frame
            indices_to_keep = list(range(0, len(optimized['frames']), 10))
            if len(optimized['frames']) - 1 not in indices_to_keep:
                indices_to_keep.append(len(optimized['frames']) - 1)
            
            optimized['frames'] = [optimized['frames'][i] for i in sorted(indices_to_keep)]
            optimized['metadata']['total_frames'] = len(optimized['frames'])
        
        return optimized
    
    @staticmethod
    def export_all_formats(simulation_data, base_filename):
        """Export simulation data in all available formats"""
        temp_dir = tempfile.mkdtemp()
        files_created = []
        
        # Export in all formats
        formats = {
            'pytorch': DataExporter.export_pytorch,
            'sqlite': DataExporter.export_sqlite,
            'hdf5': DataExporter.export_hdf5,
            'pickle': DataExporter.export_pickle,
            'npz': DataExporter.export_npz,
            'msgpack': DataExporter.export_msgpack
        }
        
        for format_name, export_func in formats.items():
            try:
                filename = f"{base_filename}_{simulation_data['simulation_id']}.{format_name}"
                filepath = os.path.join(temp_dir, filename)
                
                # Special handling for in-memory formats
                if format_name in ['pytorch', 'pickle', 'msgpack']:
                    result = export_func(simulation_data)
                    if isinstance(result, BytesIO):
                        with open(filepath, 'wb') as f:
                            f.write(result.getvalue())
                    else:
                        filepath = result
                else:
                    export_func(simulation_data, filepath)
                
                files_created.append(filepath)
            except Exception as e:
                st.warning(f"Failed to export {format_name}: {str(e)}")
        
        # Create a README file
        readme_content = f"""
        Simulation Data Export
        ======================
        
        Simulation ID: {simulation_data['simulation_id']}
        Defect Type: {simulation_data['parameters']['defect_type']}
        Timestamp: {simulation_data['timestamp']}
        
        Files included:
        """
        
        for i, filepath in enumerate(files_created):
            filename = os.path.basename(filepath)
            readme_content += f"\n{i+1}. {filename}"
        
        readme_content += """
        
        Format descriptions:
        - pytorch: PyTorch tensor format, ideal for ML training
        - sqlite: SQLite database, queryable with SQL
        - hdf5: Hierarchical Data Format, good for large datasets
        - pickle: Python pickle format, easy to load in Python
        - npz: Compressed numpy format, efficient for arrays
        - msgpack: Compact binary format, good for transmission
        
        Data structure:
        - Metadata: Simulation parameters and settings
        - Frames: Time evolution of defect and stress fields
        - Grid: Spatial coordinates
        - Line profiles: 1D stress profiles for analysis
        - Statistics: Derived metrics and statistics
        """
        
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        files_created.append(readme_path)
        
        # Create zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filepath in files_created:
                zip_file.write(filepath, os.path.basename(filepath))
        
        # Cleanup temp files
        for filepath in files_created:
            try:
                os.unlink(filepath)
            except:
                pass
        os.rmdir(temp_dir)
        
        zip_buffer.seek(0)
        return zip_buffer

# =============================================
# BATCH EXPORT SYSTEM FOR MULTIPLE SIMULATIONS
# =============================================
class BatchExporter:
    """Export multiple simulations in ML-ready formats"""
    
    @staticmethod
    def export_all_simulations(simulations_dict, format='zip'):
        """
        Export all simulations in the database
        
        Args:
            simulations_dict: Dictionary of simulations from SimulationDB
            format: Output format ('zip', 'hdf5', 'sqlite')
        
        Returns:
            BytesIO buffer or filepath
        """
        if not simulations_dict:
            raise ValueError("No simulations to export")
        
        if format == 'zip':
            return BatchExporter._export_zip(simulations_dict)
        elif format == 'hdf5':
            return BatchExporter._export_hdf5_collection(simulations_dict)
        elif format == 'sqlite':
            return BatchExporter._export_sqlite_collection(simulations_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _export_zip(simulations_dict):
        """Export all simulations as individual files in a zip"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        base_dir = os.path.join(temp_dir, "ag_np_simulations")
        os.makedirs(base_dir, exist_ok=True)
        
        # Create metadata file for the collection
        collection_metadata = {
            'total_simulations': len(simulations_dict),
            'export_timestamp': datetime.now().isoformat(),
            'simulation_ids': list(simulations_dict.keys()),
            'format_versions': {
                'data_structure': '1.0',
                'exporter_version': '1.0'
            }
        }
        
        metadata_path = os.path.join(base_dir, "collection_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(collection_metadata, f, indent=2)
        
        # Export each simulation
        for sim_id, sim_data in simulations_dict.items():
            sim_dir = os.path.join(base_dir, sim_id)
            os.makedirs(sim_dir, exist_ok=True)
            
            # Create structured data
            grid_params = {
                'extent': [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2],
                'dx': dx,
                'N': N
            }
            
            structured_data = SimulationDataStructure.create_simulation_data(
                sim_id, sim_data['params'], sim_data['history'], 
                sim_data['metadata'], grid_params
            )
            
            # Export in multiple formats
            formats_to_export = ['pytorch', 'npz', 'pickle']
            
            for fmt in formats_to_export:
                try:
                    if fmt == 'pytorch':
                        filepath = os.path.join(sim_dir, f"{sim_id}.pt")
                        DataExporter.export_pytorch(structured_data, filepath)
                    elif fmt == 'npz':
                        filepath = os.path.join(sim_dir, f"{sim_id}.npz")
                        DataExporter.export_npz(structured_data, filepath)
                    elif fmt == 'pickle':
                        filepath = os.path.join(sim_dir, f"{sim_id}.pkl")
                        DataExporter.export_pickle(structured_data, filepath)
                except Exception as e:
                    st.warning(f"Failed to export {sim_id} in {fmt}: {str(e)}")
            
            # Also save parameters as JSON for quick reference
            params_path = os.path.join(sim_dir, f"{sim_id}_parameters.json")
            with open(params_path, 'w') as f:
                json.dump(structured_data['parameters'], f, indent=2)
        
        # Create README
        readme_content = """
        Ag Nanoparticle Defect Simulation Collection
        ============================================
        
        This collection contains phase-field simulations of defects in Ag nanoparticles.
        
        Directory Structure:
        - collection_metadata.json: Overview of all simulations
        - [simulation_id]/: Directory for each simulation
            - [simulation_id].pt: PyTorch tensor format
            - [simulation_id].npz: Compressed numpy arrays
            - [simulation_id].pkl: Python pickle format
            - [simulation_id]_parameters.json: Simulation parameters
        
        Simulation Types:
        - ISF: Intrinsic Stacking Fault (code: 0)
        - ESF: Extrinsic Stacking Fault (code: 1)
        - Twin: Coherent Twin Boundary (code: 2)
        
        Seed Shapes:
        - Square (code: 0)
        - Horizontal Fault (code: 1)
        - Vertical Fault (code: 2)
        - Rectangle (code: 3)
        - Ellipse (code: 4)
        
        Data Structure in PyTorch files:
        - metadata: Simulation metadata
        - parameters: All simulation parameters
        - grid_X, grid_Y: Spatial coordinates
        - defect_fields: 3D tensor (frames, height, width)
        - stress_components: Dictionary of 3D stress tensors
        - strain_components: Dictionary of 3D strain tensors
        - line_profiles: 1D profiles through center
        - statistics: Derived statistics
        
        For ML applications, use:
        - Input features: defect_fields, parameters (as one-hot encoded)
        - Output labels: stress_components, strain_components
        
        Created with Ag NP Multi-Defect Analyzer
        """
        
        readme_path = os.path.join(base_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Create zip
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_file.write(file_path, arcname)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def _export_hdf5_collection(simulations_dict):
        """Export all simulations into a single HDF5 file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        temp_file.close()
        
        with h5py.File(temp_file.name, 'w') as f:
            # Create collection metadata
            f.attrs['total_simulations'] = len(simulations_dict)
            f.attrs['export_timestamp'] = datetime.now().isoformat()
            f.attrs['format_version'] = '1.0'
            
            # Create index group
            index_group = f.create_group('index')
            
            # Add each simulation
            for sim_id, sim_data in simulations_dict.items():
                sim_group = f.create_group(f'simulations/{sim_id}')
                
                # Store parameters as attributes
                for key, value in sim_data['params'].items():
                    if isinstance(value, (str, int, float)):
                        sim_group.attrs[key] = value
                
                # Store metadata
                for key, value in sim_data['metadata'].items():
                    if isinstance(value, (str, int, float)):
                        sim_group.attrs[f'meta_{key}'] = value
                
                # Add to index
                index_group.attrs[sim_id] = json.dumps({
                    'defect_type': sim_data['params']['defect_type'],
                    'orientation': sim_data['params']['orientation'],
                    'eps0': sim_data['params']['eps0'],
                    'kappa': sim_data['params']['kappa']
                })
        
        # Read file into bytes
        with open(temp_file.name, 'rb') as f:
            buffer = BytesIO(f.read())
        
        os.unlink(temp_file.name)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def _export_sqlite_collection(simulations_dict):
        """Export all simulations into a single SQLite database"""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create master tables
        cursor.execute('''
            CREATE TABLE collection_metadata (
                export_id TEXT PRIMARY KEY,
                total_simulations INTEGER,
                export_timestamp TEXT,
                format_version TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE simulation_index (
                sim_id TEXT PRIMARY KEY,
                defect_type TEXT,
                defect_type_code INTEGER,
                seed_shape TEXT,
                seed_shape_code INTEGER,
                habit_plane_angle REAL,
                eps0 REAL,
                kappa REAL,
                total_frames INTEGER,
                FOREIGN KEY (sim_id) REFERENCES simulation_data (sim_id)
            )
        ''')
        
        # Create a table for each simulation's metadata
        cursor.execute('''
            CREATE TABLE simulation_data (
                sim_id TEXT PRIMARY KEY,
                parameters_json TEXT,
                metadata_json TEXT
            )
        ''')
        
        # Insert collection metadata
        export_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]
        cursor.execute('''
            INSERT INTO collection_metadata VALUES (?, ?, ?, ?)
        ''', (export_id, len(simulations_dict), datetime.now().isoformat(), '1.0'))
        
        # Insert each simulation
        for sim_id, sim_data in simulations_dict.items():
            # Insert into index
            params = sim_data['params']
            cursor.execute('''
                INSERT INTO simulation_index 
                (sim_id, defect_type, defect_type_code, seed_shape, seed_shape_code,
                 habit_plane_angle, eps0, kappa, total_frames)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sim_id,
                params['defect_type'],
                SimulationDataStructure.defect_type_to_code(params['defect_type']),
                params['shape'],
                SimulationDataStructure.shape_to_code(params['shape']),
                float(params.get('theta', 0)),
                float(params['eps0']),
                float(params['kappa']),
                len(sim_data['history'])
            ))
            
            # Insert into simulation_data
            cursor.execute('''
                INSERT INTO simulation_data (sim_id, parameters_json, metadata_json)
                VALUES (?, ?, ?)
            ''', (
                sim_id,
                json.dumps(params),
                json.dumps(sim_data['metadata'])
            ))
        
        conn.commit()
        
        # Export to bytes
        buffer = BytesIO()
        for line in conn.iterdump():
            buffer.write(f'{line}\n'.encode('utf-8'))
        
        conn.close()
        buffer.seek(0)
        return buffer

# =============================================
# ENHANCED EXPORT UI
# =============================================
def create_export_ui():
    """Create comprehensive export interface"""
    st.sidebar.header("💾 ML-Ready Data Export")
    
    with st.sidebar.expander("📤 Export Single Simulation", expanded=False):
        # Get available simulations
        simulations = SimulationDB.get_simulation_list()
        
        if simulations:
            sim_options = {f"{sim['name']} (ID: {sim['id']})": sim['id'] for sim in simulations}
            selected_sim_name = st.selectbox("Select Simulation", list(sim_options.keys()))
            selected_sim_id = sim_options[selected_sim_name]
            
            export_format = st.selectbox(
                "Export Format",
                [
                    "PyTorch (.pt)", 
                    "SQLite (.db)", 
                    "HDF5 (.h5)", 
                    "Pickle (.pkl)",
                    "Compressed NumPy (.npz)",
                    "MessagePack (.msgpack)",
                    "All Formats (ZIP)"
                ]
            )
            
            if st.button("📥 Export Single Simulation", type="primary"):
                with st.spinner("Preparing export..."):
                    # Get simulation data
                    sim_data = SimulationDB.get_simulation(selected_sim_id)
                    if sim_data:
                        # Create structured data
                        grid_params = {
                            'extent': [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2],
                            'dx': dx,
                            'N': N
                        }
                        
                        structured_data = SimulationDataStructure.create_simulation_data(
                            selected_sim_id,
                            sim_data['params'],
                            sim_data['history'],
                            sim_data['metadata'],
                            grid_params
                        )
                        
                        # Export based on format
                        format_map = {
                            "PyTorch (.pt)": ("pytorch", "pt"),
                            "SQLite (.db)": ("sqlite", "db"),
                            "HDF5 (.h5)": ("hdf5", "h5"),
                            "Pickle (.pkl)": ("pickle", "pkl"),
                            "Compressed NumPy (.npz)": ("npz", "npz"),
                            "MessagePack (.msgpack)": ("msgpack", "msgpack"),
                            "All Formats (ZIP)": ("all", "zip")
                        }
                        
                        fmt_key, ext = format_map[export_format]
                        
                        if fmt_key == "pytorch":
                            buffer = DataExporter.export_pytorch(structured_data)
                        elif fmt_key == "sqlite":
                            buffer = DataExporter.export_sqlite(structured_data)
                        elif fmt_key == "hdf5":
                            buffer = DataExporter.export_hdf5(structured_data)
                        elif fmt_key == "pickle":
                            buffer = DataExporter.export_pickle(structured_data)
                        elif fmt_key == "npz":
                            buffer = DataExporter.export_npz(structured_data)
                        elif fmt_key == "msgpack":
                            buffer = DataExporter.export_msgpack(structured_data)
                        elif fmt_key == "all":
                            base_name = f"simulation_{selected_sim_id}"
                            buffer = DataExporter.export_all_formats(structured_data, base_name)
                        
                        # Offer download
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{selected_sim_id}_{timestamp}.{ext}"
                        
                        st.sidebar.download_button(
                            "⬇️ Download Export",
                            buffer.getvalue() if hasattr(buffer, 'getvalue') else buffer,
                            filename,
                            f"application/{ext}"
                        )
                        
                        # Show export summary
                        st.sidebar.success(f"""
                        ✅ Export ready!
                        - Format: {export_format}
                        - File: {filename}
                        - Defect: {sim_data['params']['defect_type']}
                        - Frames: {len(sim_data['history'])}
                        """)
        
        else:
            st.info("No simulations available for export")
    
    with st.sidebar.expander("📚 Export All Simulations", expanded=False):
        simulations = SimulationDB.get_all_simulations()
        
        if simulations:
            st.info(f"Found {len(simulations)} simulations")
            
            export_format_all = st.selectbox(
                "Export Format for Collection",
                ["ZIP Archive", "Single HDF5 File", "Single SQLite Database"],
                key="export_all_format"
            )
            
            include_format_map = {
                "ZIP Archive": "zip",
                "Single HDF5 File": "hdf5",
                "Single SQLite Database": "sqlite"
            }
            
            if st.button("📦 Export All Simulations", type="primary"):
                with st.spinner(f"Exporting {len(simulations)} simulations..."):
                    try:
                        buffer = BatchExporter.export_all_simulations(
                            simulations,
                            format=include_format_map[export_format_all]
                        )
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        ext = include_format_map[export_format_all]
                        filename = f"ag_np_simulations_collection_{timestamp}.{ext}"
                        
                        st.sidebar.download_button(
                            "⬇️ Download Collection",
                            buffer.getvalue() if hasattr(buffer, 'getvalue') else buffer,
                            filename,
                            f"application/{ext}"
                        )
                        
                        # Show collection summary
                        defect_types = {}
                        for sim_id, sim_data in simulations.items():
                            defect_type = sim_data['params']['defect_type']
                            defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
                        
                        st.sidebar.success(f"""
                        ✅ Collection exported!
                        - Total simulations: {len(simulations)}
                        - Format: {export_format_all}
                        - File: {filename}
                        
                        Simulation breakdown:
                        {', '.join([f'{k}: {v}' for k, v in defect_types.items()])}
                        """)
                        
                    except Exception as e:
                        st.sidebar.error(f"Export failed: {str(e)}")
        else:
            st.info("No simulations available for export")
    
    with st.sidebar.expander("🔧 Export Settings", expanded=False):
        st.markdown("**Data Compression Options**")
        compress_data = st.checkbox("Enable compression", True)
        compression_level = st.slider("Compression level", 1, 9, 6)
        
        st.markdown("**Data Reduction**")
        keep_all_frames = st.checkbox("Keep all frames", True)
        if not keep_all_frames:
            frames_to_keep = st.slider("Frames to keep", 1, 100, 10)
        
        st.markdown("**Metadata Options**")
        include_full_metadata = st.checkbox("Include full metadata", True)
        include_line_profiles = st.checkbox("Include line profiles", True)
        include_statistics = st.checkbox("Include statistics", True)

# =============================================
# GITHUB REPOSITORY COMPATIBILITY
# =============================================
def create_github_ready_export():
    """Create export optimized for GitHub repository"""
    simulations = SimulationDB.get_all_simulations()
    
    if not simulations:
        st.warning("No simulations to export")
        return
    
    st.header("🚀 GitHub Repository Export")
    st.markdown("""
    Create a complete dataset for your GitHub repository with:
    - Organized directory structure
    - README files with documentation
    - Multiple data formats
    - Jupyter notebook examples
    - Requirements file
    """)
    
    # Repository configuration
    repo_name = st.text_input("Repository Name", "ag-nanoparticle-defect-dataset")
    repo_description = st.text_area("Repository Description", 
                                   "Phase-field simulations of defects in silver nanoparticles for machine learning applications")
    
    col1, col2 = st.columns(2)
    with col1:
        include_notebooks = st.checkbox("Include Jupyter Notebooks", True)
        include_requirements = st.checkbox("Include requirements.txt", True)
    with col2:
        include_ml_examples = st.checkbox("Include ML Examples", True)
        split_train_test = st.checkbox("Split train/test sets", False)
    
    if st.button("🛠️ Create GitHub Repository Package", type="primary"):
        with st.spinner("Creating GitHub-ready package..."):
            # Create temporary directory structure
            temp_dir = tempfile.mkdtemp()
            repo_dir = os.path.join(temp_dir, repo_name)
            os.makedirs(repo_dir, exist_ok=True)
            
            # Create directory structure
            dirs = [
                "data/raw",
                "data/processed",
                "notebooks",
                "scripts",
                "models",
                "docs"
            ]
            
            for dir_path in dirs:
                os.makedirs(os.path.join(repo_dir, dir_path), exist_ok=True)
            
            # Export simulations to data/raw
            for sim_id, sim_data in simulations.items():
                sim_dir = os.path.join(repo_dir, "data/raw", sim_id)
                os.makedirs(sim_dir, exist_ok=True)
                
                # Create structured data
                grid_params = {
                    'extent': [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2],
                    'dx': dx,
                    'N': N
                }
                
                structured_data = SimulationDataStructure.create_simulation_data(
                    sim_id,
                    sim_data['params'],
                    sim_data['history'],
                    sim_data['metadata'],
                    grid_params
                )
                
                # Export in PyTorch format (ML-ready)
                torch_path = os.path.join(sim_dir, f"{sim_id}.pt")
                DataExporter.export_pytorch(structured_data, torch_path)
                
                # Export parameters as JSON
                params_path = os.path.join(sim_dir, f"{sim_id}_params.json")
                with open(params_path, 'w') as f:
                    json.dump(structured_data['parameters'], f, indent=2)
            
            # Create README.md
            readme_content = f"""# {repo_name}

{repo_description}

## Dataset Overview

This dataset contains phase-field simulations of defects in silver nanoparticles.

### Simulation Parameters

- **Defect Types**: ISF (Intrinsic Stacking Fault), ESF (Extrinsic Stacking Fault), Twin
- **Grid Size**: {N} × {N} points
- **Grid Spacing**: {dx} nm
- **Total Simulations**: {len(simulations)}

### Data Structure
