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

# =============================================
# NEW IMPORTS FOR DATA EXPORT
# =============================================
import torch
import pickle
import sqlite3
import h5py
import tempfile
import os
from pathlib import Path
import msgpack

# =============================================
# ENHANCED DATA STRUCTURE CLASSES
# =============================================
class MLDataStructure:
    """Structured data container for machine learning applications"""
    
    @staticmethod
    def create_simulation_data(sim_id, params, history, metadata, grid_info):
        """
        Create structured simulation data with all attributes for ML
        
        Args:
            sim_id: Unique simulation ID
            params: Simulation parameters dictionary
            history: List of (eta, stress_fields) tuples
            metadata: Additional metadata
            grid_info: Grid configuration dictionary
        
        Returns:
            dict: Structured data ready for ML consumption
        """
        # Convert defect type to numeric code
        defect_codes = {"ISF": 0, "ESF": 1, "Twin": 2}
        shape_codes = {
            "Square": 0, 
            "Horizontal Fault": 1, 
            "Vertical Fault": 2, 
            "Rectangle": 3, 
            "Ellipse": 4
        }
        
        # Create frames data
        frames_data = []
        for frame_idx, (eta, stress_fields) in enumerate(history):
            # Calculate time step
            time_step = frame_idx * 0.004  # Assuming dt=0.004
            
            frame_data = {
                'frame_index': frame_idx,
                'time_step': time_step,
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
                'strain_components': {
                    'exx': stress_fields.get('exx', np.zeros_like(eta)).astype(np.float32),
                    'eyy': stress_fields.get('eyy', np.zeros_like(eta)).astype(np.float32),
                    'exy': stress_fields.get('exy', np.zeros_like(eta)).astype(np.float32)
                },
                'eigenstrain_components': {
                    'eps_xx_star': stress_fields.get('eps_xx_star', np.zeros_like(eta)).astype(np.float32),
                    'eps_yy_star': stress_fields.get('eps_yy_star', np.zeros_like(eta)).astype(np.float32),
                    'eps_xy_star': stress_fields.get('eps_xy_star', np.zeros_like(eta)).astype(np.float32)
                },
                'derived_quantities': {
                    'defect_volume': np.sum(eta > 0.5) * (grid_info['dx']**2),
                    'avg_stress_magnitude': np.mean(stress_fields['sigma_mag']),
                    'max_stress_magnitude': np.max(stress_fields['sigma_mag']),
                    'stress_gradient': MLDataStructure._calculate_gradient(
                        stress_fields['sigma_mag'], grid_info['dx']
                    )
                }
            }
            frames_data.append(frame_data)
        
        # Calculate comprehensive statistics
        statistics = MLDataStructure._calculate_statistics(frames_data)
        
        # Create complete structured data
        structured_data = {
            'simulation_id': sim_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'run_time': metadata.get('run_time', 0),
                'total_frames': len(history),
                'grid_size': grid_info['N'],
                'grid_spacing': grid_info['dx'],
                'total_points': grid_info['N'] * grid_info['N'],
                'export_version': '1.0'
            },
            'parameters': {
                # Defect information
                'defect_type': params['defect_type'],
                'defect_type_code': defect_codes.get(params['defect_type'], -1),
                
                # Seed information
                'seed_shape': params['shape'],
                'seed_shape_code': shape_codes.get(params['shape'], -1),
                
                # Orientation information
                'habit_plane_angle_deg': float(params.get('angle_deg', np.rad2deg(params.get('theta', 0)))),
                'habit_plane_angle_rad': float(params.get('theta', 0)),
                'habit_plane_orientation': params.get('orientation', 'Horizontal {111} (0°)'),
                
                # Simulation parameters
                'eigenstrain_magnitude': float(params['eps0']),
                'interface_energy_coeff': float(params['kappa']),
                'total_steps': params['steps'],
                'save_frequency': params['save_every'],
                'initial_amplitude': MLDataStructure._get_initial_amplitude(params['defect_type']),
                
                # Physical constants
                'physical_constants': {
                    'lattice_parameter': 0.4086,
                    'elastic_C11': 124.0,
                    'elastic_C12': 93.4,
                    'elastic_C44': 46.1
                },
                
                # Simulation settings
                'simulation_settings': {
                    'dt': 0.004,
                    'dx': grid_info['dx'],
                    'N': grid_info['N']
                }
            },
            'grid': {
                'X': grid_info['X'].astype(np.float32),
                'Y': grid_info['Y'].astype(np.float32),
                'dx': grid_info['dx'],
                'extent': grid_info['extent'],
                'grid_shape': (grid_info['N'], grid_info['N'])
            },
            'frames': frames_data,
            'statistics': statistics,
            'line_profiles': MLDataStructure._extract_profiles(frames_data, grid_info)
        }
        
        return structured_data
    
    @staticmethod
    def _get_initial_amplitude(defect_type):
        """Get initial amplitude based on defect type"""
        amplitudes = {"ISF": 0.70, "ESF": 0.75, "Twin": 0.90}
        return amplitudes.get(defect_type, 0.75)
    
    @staticmethod
    def _calculate_gradient(field, dx):
        """Calculate gradient magnitude of a field"""
        grad_y, grad_x = np.gradient(field, dx)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return {
            'grad_x': grad_x.astype(np.float32),
            'grad_y': grad_y.astype(np.float32),
            'magnitude': gradient_magnitude.astype(np.float32)
        }
    
    @staticmethod
    def _calculate_statistics(frames_data):
        """Calculate comprehensive statistics across frames"""
        if not frames_data:
            return {}
        
        # Collect metrics across frames
        metrics = {
            'defect_volume': [],
            'avg_stress': [],
            'max_stress': [],
            'stress_gradient': []
        }
        
        for frame in frames_data:
            metrics['defect_volume'].append(frame['derived_quantities']['defect_volume'])
            metrics['avg_stress'].append(frame['derived_quantities']['avg_stress_magnitude'])
            metrics['max_stress'].append(frame['derived_quantities']['max_stress_magnitude'])
            metrics['stress_gradient'].append(
                np.mean(frame['derived_quantities']['stress_gradient']['magnitude'])
            )
        
        # Calculate statistics
        stats = {}
        for key, values in metrics.items():
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'evolution': np.array(values, dtype=np.float32)
                }
        
        return stats
    
    @staticmethod
    def _extract_profiles(frames_data, grid_info):
        """Extract line profiles from final frame"""
        if not frames_data:
            return {}
        
        final_frame = frames_data[-1]
        stress_field = final_frame['stress_components']['sigma_mag']
        defect_field = final_frame['defect_field']
        N = grid_info['N']
        dx = grid_info['dx']
        
        profiles = {}
        
        # Horizontal profile through center
        center_idx = N // 2
        profiles['horizontal'] = {
            'distance': np.linspace(grid_info['extent'][0], grid_info['extent'][1], N).astype(np.float32),
            'stress': stress_field[center_idx, :].astype(np.float32),
            'defect': defect_field[center_idx, :].astype(np.float32),
            'position': center_idx
        }
        
        # Vertical profile through center
        profiles['vertical'] = {
            'distance': np.linspace(grid_info['extent'][2], grid_info['extent'][3], N).astype(np.float32),
            'stress': stress_field[:, center_idx].astype(np.float32),
            'defect': defect_field[:, center_idx].astype(np.float32),
            'position': center_idx
        }
        
        # Diagonal profile
        diag_length = N
        profiles['diagonal'] = {
            'distance': np.linspace(0, np.sqrt(2) * diag_length * dx, diag_length).astype(np.float32),
            'stress': np.diag(stress_field).astype(np.float32),
            'defect': np.diag(defect_field).astype(np.float32)
        }
        
        return profiles

# =============================================
# MULTI-FORMAT DATA EXPORTER
# =============================================
class DataExporter:
    """Export simulation data in multiple ML-ready formats"""
    
    @staticmethod
    def export_pytorch(sim_data, filepath=None):
        """
        Export simulation data as PyTorch tensors
        
        Args:
            sim_data: Structured simulation data
            filepath: Optional file path to save directly
        
        Returns:
            BytesIO buffer or filepath
        """
        # Convert to PyTorch tensors
        torch_data = {}
        
        # Metadata and parameters
        torch_data['metadata'] = sim_data['metadata']
        torch_data['parameters'] = sim_data['parameters']
        
        # Grid data
        torch_data['grid_X'] = torch.from_numpy(sim_data['grid']['X'])
        torch_data['grid_Y'] = torch.from_numpy(sim_data['grid']['Y'])
        
        # Collect all frames into 3D tensors
        n_frames = len(sim_data['frames'])
        n, m = sim_data['grid']['grid_shape']
        
        # Initialize tensors
        defect_fields = torch.zeros((n_frames, n, m), dtype=torch.float32)
        
        # Stress components
        stress_keys = list(sim_data['frames'][0]['stress_components'].keys())
        stress_tensors = {}
        for key in stress_keys:
            stress_tensors[key] = torch.zeros((n_frames, n, m), dtype=torch.float32)
        
        # Strain components
        strain_keys = list(sim_data['frames'][0]['strain_components'].keys())
        strain_tensors = {}
        for key in strain_keys:
            strain_tensors[key] = torch.zeros((n_frames, n, m), dtype=torch.float32)
        
        # Fill tensors
        for i, frame in enumerate(sim_data['frames']):
            defect_fields[i] = torch.from_numpy(frame['defect_field'])
            
            for key in stress_keys:
                stress_tensors[key][i] = torch.from_numpy(frame['stress_components'][key])
            
            for key in strain_keys:
                strain_tensors[key][i] = torch.from_numpy(frame['strain_components'][key])
        
        torch_data['defect_fields'] = defect_fields
        torch_data['stress_components'] = stress_tensors
        torch_data['strain_components'] = strain_tensors
        
        # Line profiles
        torch_data['line_profiles'] = {}
        for profile_name, profile_data in sim_data['line_profiles'].items():
            profile_dict = {}
            for key, value in profile_data.items():
                if isinstance(value, np.ndarray):
                    profile_dict[key] = torch.from_numpy(value)
                else:
                    profile_dict[key] = value
            torch_data['line_profiles'][profile_name] = profile_dict
        
        # Statistics
        torch_data['statistics'] = sim_data['statistics']
        
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
    def export_sqlite(sim_data, filepath=None):
        """
        Export simulation data to SQLite database
        
        Args:
            sim_data: Structured simulation data
            filepath: Optional file path to save directly
        
        Returns:
            BytesIO buffer or filepath
        """
        if filepath is None:
            filepath = ':memory:'
        
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        
        # Create tables
        tables_sql = [
            '''CREATE TABLE IF NOT EXISTS simulation_metadata (
                sim_id TEXT PRIMARY KEY,
                defect_type TEXT,
                defect_type_code INTEGER,
                seed_shape TEXT,
                seed_shape_code INTEGER,
                habit_plane_angle REAL,
                orientation TEXT,
                eps0 REAL,
                kappa REAL,
                steps INTEGER,
                save_every INTEGER,
                total_frames INTEGER,
                grid_size INTEGER,
                grid_spacing REAL,
                timestamp TEXT
            )''',
            
            '''CREATE TABLE IF NOT EXISTS simulation_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id TEXT,
                frame_index INTEGER,
                time_step REAL,
                defect_volume REAL,
                avg_stress REAL,
                max_stress REAL,
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
                eps_xx_star BLOB,
                eps_yy_star BLOB,
                eps_xy_star BLOB,
                FOREIGN KEY (sim_id) REFERENCES simulation_metadata(sim_id)
            )''',
            
            '''CREATE TABLE IF NOT EXISTS line_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_id TEXT,
                profile_type TEXT,
                distance BLOB,
                stress_profile BLOB,
                defect_profile BLOB,
                FOREIGN KEY (sim_id) REFERENCES simulation_metadata(sim_id)
            )'''
        ]
        
        for sql in tables_sql:
            cursor.execute(sql)
        
        # Insert simulation metadata
        params = sim_data['parameters']
        metadata = sim_data['metadata']
        
        cursor.execute('''
            INSERT INTO simulation_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sim_data['simulation_id'],
            params['defect_type'],
            params['defect_type_code'],
            params['seed_shape'],
            params['seed_shape_code'],
            params['habit_plane_angle_deg'],
            params['habit_plane_orientation'],
            params['eigenstrain_magnitude'],
            params['interface_energy_coeff'],
            params['total_steps'],
            params['save_frequency'],
            metadata['total_frames'],
            metadata['grid_size'],
            metadata['grid_spacing'],
            sim_data['timestamp']
        ))
        
        # Insert frames data
        for frame in sim_data['frames']:
            # Convert arrays to bytes
            arrays = [
                sqlite3.Binary(frame['defect_field'].tobytes()),
                sqlite3.Binary(frame['stress_components']['sxx'].tobytes()),
                sqlite3.Binary(frame['stress_components']['syy'].tobytes()),
                sqlite3.Binary(frame['stress_components']['sxy'].tobytes()),
                sqlite3.Binary(frame['stress_components']['sigma_mag'].tobytes()),
                sqlite3.Binary(frame['stress_components']['sigma_hydro'].tobytes()),
                sqlite3.Binary(frame['stress_components']['von_mises'].tobytes()),
                sqlite3.Binary(frame['strain_components']['exx'].tobytes()),
                sqlite3.Binary(frame['strain_components']['eyy'].tobytes()),
                sqlite3.Binary(frame['strain_components']['exy'].tobytes()),
                sqlite3.Binary(frame['eigenstrain_components']['eps_xx_star'].tobytes()),
                sqlite3.Binary(frame['eigenstrain_components']['eps_yy_star'].tobytes()),
                sqlite3.Binary(frame['eigenstrain_components']['eps_xy_star'].tobytes())
            ]
            
            cursor.execute('''
                INSERT INTO simulation_frames 
                (sim_id, frame_index, time_step, defect_volume, avg_stress, max_stress,
                 defect_field, stress_sxx, stress_syy, stress_sxy, stress_sigma_mag,
                 stress_sigma_hydro, stress_von_mises, strain_exx, strain_eyy, strain_exy,
                 eps_xx_star, eps_yy_star, eps_xy_star)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sim_data['simulation_id'],
                frame['frame_index'],
                frame['time_step'],
                frame['derived_quantities']['defect_volume'],
                frame['derived_quantities']['avg_stress_magnitude'],
                frame['derived_quantities']['max_stress_magnitude'],
                *arrays
            ))
        
        # Insert line profiles
        for profile_name, profile_data in sim_data['line_profiles'].items():
            cursor.execute('''
                INSERT INTO line_profiles (sim_id, profile_type, distance, stress_profile, defect_profile)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                sim_data['simulation_id'],
                profile_name,
                sqlite3.Binary(profile_data['distance'].tobytes()),
                sqlite3.Binary(profile_data['stress'].tobytes()),
                sqlite3.Binary(profile_data['defect'].tobytes())
            ))
        
        conn.commit()
        
        # Return bytes if in-memory database
        if filepath == ':memory:':
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
    def export_pickle(sim_data, filepath=None):
        """
        Export simulation data using Python pickle
        
        Args:
            sim_data: Structured simulation data
            filepath: Optional file path to save directly
        
        Returns:
            BytesIO buffer or filepath
        """
        # Optimize data size for pickling
        optimized_data = DataExporter._optimize_for_export(sim_data)
        
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
    def export_npz(sim_data, filepath=None):
        """
        Export simulation data as compressed numpy arrays
        
        Args:
            sim_data: Structured simulation data
            filepath: Optional file path to save directly
        
        Returns:
            BytesIO buffer or filepath
        """
        if filepath is None:
            filepath = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name
        
        save_dict = {}
        
        # Store metadata as JSON strings
        save_dict['metadata'] = json.dumps(sim_data['metadata'])
        save_dict['parameters'] = json.dumps(sim_data['parameters'])
        
        # Store grid
        save_dict['grid_X'] = sim_data['grid']['X']
        save_dict['grid_Y'] = sim_data['grid']['Y']
        
        # Store defect fields as 3D array
        n_frames = len(sim_data['frames'])
        n, m = sim_data['grid']['grid_shape']
        
        defect_fields = np.zeros((n_frames, n, m), dtype=np.float32)
        stress_mag = np.zeros((n_frames, n, m), dtype=np.float32)
        
        for i, frame in enumerate(sim_data['frames']):
            defect_fields[i] = frame['defect_field']
            stress_mag[i] = frame['stress_components']['sigma_mag']
        
        save_dict['defect_fields'] = defect_fields
        save_dict['stress_magnitude'] = stress_mag
        
        # Store final frame data
        final_frame = sim_data['frames'][-1]
        save_dict['final_defect'] = final_frame['defect_field']
        
        for key, value in final_frame['stress_components'].items():
            save_dict[f'final_stress_{key}'] = value
        
        for key, value in final_frame['strain_components'].items():
            save_dict[f'final_strain_{key}'] = value
        
        # Store line profiles
        for profile_name, profile_data in sim_data['line_profiles'].items():
            for key, value in profile_data.items():
                if isinstance(value, np.ndarray):
                    save_dict[f'profile_{profile_name}_{key}'] = value
        
        # Save compressed file
        np.savez_compressed(filepath, **save_dict)
        
        # Read into bytes if temporary file
        if filepath.startswith(tempfile.gettempdir()):
            with open(filepath, 'rb') as f:
                buffer = BytesIO(f.read())
            os.unlink(filepath)
            buffer.seek(0)
            return buffer
        else:
            return filepath
    
    @staticmethod
    def export_hdf5(sim_data, filepath=None):
        """
        Export simulation data to HDF5 format
        
        Args:
            sim_data: Structured simulation data
            filepath: Optional file path to save directly
        
        Returns:
            BytesIO buffer or filepath
        """
        if filepath is None:
            filepath = tempfile.NamedTemporaryFile(suffix='.h5', delete=False).name
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            f.attrs['simulation_id'] = sim_data['simulation_id']
            f.attrs['timestamp'] = sim_data['timestamp']
            
            # Metadata group
            meta_group = f.create_group('metadata')
            for key, value in sim_data['metadata'].items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
            
            # Parameters group
            params_group = f.create_group('parameters')
            for key, value in sim_data['parameters'].items():
                if isinstance(value, dict):
                    sub_group = params_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (str, int, float)):
                            sub_group.attrs[sub_key] = sub_value
                elif isinstance(value, (str, int, float)):
                    params_group.attrs[key] = value
            
            # Grid group
            grid_group = f.create_group('grid')
            grid_group.create_dataset('X', data=sim_data['grid']['X'])
            grid_group.create_dataset('Y', data=sim_data['grid']['Y'])
            grid_group.attrs['dx'] = sim_data['grid']['dx']
            grid_group.attrs['extent'] = sim_data['grid']['extent']
            
            # Frames group
            frames_group = f.create_group('frames')
            for i, frame in enumerate(sim_data['frames']):
                frame_group = frames_group.create_group(f'frame_{i:04d}')
                frame_group.attrs['frame_index'] = frame['frame_index']
                frame_group.attrs['time_step'] = frame['time_step']
                
                # Store arrays
                frame_group.create_dataset('defect', data=frame['defect_field'])
                
                # Stress components
                stress_group = frame_group.create_group('stress')
                for key, value in frame['stress_components'].items():
                    stress_group.create_dataset(key, data=value)
                
                # Strain components
                strain_group = frame_group.create_group('strain')
                for key, value in frame['strain_components'].items():
                    strain_group.create_dataset(key, data=value)
            
            # Line profiles group
            profiles_group = f.create_group('line_profiles')
            for profile_name, profile_data in sim_data['line_profiles'].items():
                profile_group = profiles_group.create_group(profile_name)
                for key, value in profile_data.items():
                    if isinstance(value, np.ndarray):
                        profile_group.create_dataset(key, data=value)
        
        # Return bytes if temporary file
        if filepath.startswith(tempfile.gettempdir()):
            with open(filepath, 'rb') as f:
                buffer = BytesIO(f.read())
            os.unlink(filepath)
            buffer.seek(0)
            return buffer
        else:
            return filepath
    
    @staticmethod
    def export_all_formats(sim_data, base_name=None):
        """
        Export simulation data in all available formats as a zip file
        
        Args:
            sim_data: Structured simulation data
            base_name: Base name for files
        
        Returns:
            BytesIO buffer with zip file
        """
        if base_name is None:
            base_name = f"simulation_{sim_data['simulation_id']}"
        
        temp_dir = tempfile.mkdtemp()
        files_created = []
        
        # Export in all formats
        formats = [
            ('pytorch', '.pt', DataExporter.export_pytorch),
            ('sqlite', '.db', DataExporter.export_sqlite),
            ('hdf5', '.h5', DataExporter.export_hdf5),
            ('pickle', '.pkl', DataExporter.export_pickle),
            ('npz', '.npz', DataExporter.export_npz)
        ]
        
        for format_name, ext, export_func in formats:
            try:
                filename = f"{base_name}{ext}"
                filepath = os.path.join(temp_dir, filename)
                
                # Export the data
                result = export_func(sim_data, filepath)
                
                # If result is a filepath, we already saved it
                # If result is a buffer, we need to save it
                if isinstance(result, BytesIO):
                    with open(filepath, 'wb') as f:
                        f.write(result.getvalue())
                
                files_created.append(filepath)
            except Exception as e:
                st.warning(f"Failed to export {format_name}: {str(e)}")
        
        # Create a README file
        readme_content = f"""
        Simulation Data Export
        ======================
        
        Simulation ID: {sim_data['simulation_id']}
        Defect Type: {sim_data['parameters']['defect_type']}
        Export Timestamp: {sim_data['timestamp']}
        
        Files included:
        """
        
        for i, filepath in enumerate(files_created):
            filename = os.path.basename(filepath)
            readme_content += f"\n{i+1}. {filename}"
        
        readme_content += """
        
        Format descriptions:
        - .pt: PyTorch tensor format, ideal for ML training
        - .db: SQLite database, queryable with SQL
        - .h5: Hierarchical Data Format (HDF5), good for large datasets
        - .pkl: Python pickle format, easy to load in Python
        - .npz: Compressed numpy format, efficient for arrays
        
        Data structure:
        - Metadata: Simulation parameters and settings
        - Grid: Spatial coordinates (X, Y)
        - Frames: Time evolution of defect and stress fields
        - Line profiles: 1D stress profiles for analysis
        
        For ML applications, use the PyTorch (.pt) format.
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
        
        # Cleanup
        for filepath in files_created:
            try:
                os.unlink(filepath)
            except:
                pass
        os.rmdir(temp_dir)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def _optimize_for_export(sim_data):
        """Optimize data structure for export"""
        optimized = sim_data.copy()
        
        # For smaller files, we can reduce the number of frames
        if len(optimized['frames']) > 20:
            # Keep every 5th frame and the final frame
            indices = list(range(0, len(optimized['frames']), 5))
            if len(optimized['frames']) - 1 not in indices:
                indices.append(len(optimized['frames']) - 1)
            
            optimized['frames'] = [optimized['frames'][i] for i in sorted(indices)]
            optimized['metadata']['total_frames'] = len(optimized['frames'])
        
        return optimized

# =============================================
# BATCH EXPORTER FOR ALL SIMULATIONS
# =============================================
class BatchExporter:
    """Export all simulations in the database"""
    
    @staticmethod
    def export_all_simulations(simulations_dict, format='zip'):
        """
        Export all simulations from the database
        
        Args:
            simulations_dict: Dictionary of simulations from SimulationDB
            format: Output format ('zip', 'hdf5', 'sqlite')
        
        Returns:
            BytesIO buffer with exported data
        """
        if not simulations_dict:
            raise ValueError("No simulations to export")
        
        if format == 'zip':
            return BatchExporter._export_as_zip(simulations_dict)
        elif format == 'hdf5':
            return BatchExporter._export_as_hdf5(simulations_dict)
        elif format == 'sqlite':
            return BatchExporter._export_as_sqlite(simulations_dict)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def _export_as_zip(simulations_dict):
        """Export all simulations as a zip file"""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        export_dir = os.path.join(temp_dir, "ag_np_simulations")
        os.makedirs(export_dir, exist_ok=True)
        
        # Grid information (common to all simulations)
        grid_info = {
            'N': 128,
            'dx': 0.1,
            'extent': [-6.4, 6.4, -6.4, 6.4],
            'X': np.linspace(-6.4, 6.4, 128),
            'Y': np.linspace(-6.4, 6.4, 128)
        }
        
        # Export each simulation
        exported_simulations = []
        
        for sim_id, sim_data in simulations_dict.items():
            sim_export_dir = os.path.join(export_dir, sim_id)
            os.makedirs(sim_export_dir, exist_ok=True)
            
            # Create structured data
            structured_data = MLDataStructure.create_simulation_data(
                sim_id,
                sim_data['params'],
                sim_data['history'],
                sim_data['metadata'],
                grid_info
            )
            
            # Export in PyTorch format (most useful for ML)
            torch_file = os.path.join(sim_export_dir, f"{sim_id}.pt")
            DataExporter.export_pytorch(structured_data, torch_file)
            
            # Export parameters as JSON for quick reference
            params_file = os.path.join(sim_export_dir, f"{sim_id}_params.json")
            with open(params_file, 'w') as f:
                json.dump(structured_data['parameters'], f, indent=2)
            
            exported_simulations.append({
                'id': sim_id,
                'defect_type': sim_data['params']['defect_type'],
                'orientation': sim_data['params'].get('orientation', 'Unknown')
            })
        
        # Create collection metadata
        collection_metadata = {
            'total_simulations': len(simulations_dict),
            'export_timestamp': datetime.now().isoformat(),
            'simulations': exported_simulations,
            'grid_info': grid_info
        }
        
        metadata_file = os.path.join(export_dir, "collection_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(collection_metadata, f, indent=2)
        
        # Create README
        readme_content = """
        Ag Nanoparticle Defect Simulation Collection
        ============================================
        
        This collection contains phase-field simulations of defects in Ag nanoparticles.
        
        Each simulation folder contains:
        - [sim_id].pt: PyTorch tensor data for ML
        - [sim_id]_params.json: Simulation parameters
        
        Data structure in PyTorch files:
        - metadata: Simulation metadata
        - parameters: All simulation parameters
        - grid_X, grid_Y: Spatial coordinates
        - defect_fields: 3D tensor (frames, height, width)
        - stress_components: Dictionary of stress tensors
        - strain_components: Dictionary of strain tensors
        - line_profiles: 1D profiles through center
        
        For ML training, use:
        - Input: defect_fields, parameters
        - Target: stress_components, strain_components
        
        Created with Ag NP Multi-Defect Analyzer
        """
        
        readme_file = os.path.join(export_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(export_dir):
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
    def _export_as_hdf5(simulations_dict):
        """Export all simulations into a single HDF5 file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        temp_file.close()
        
        with h5py.File(temp_file.name, 'w') as f:
            # Collection metadata
            f.attrs['total_simulations'] = len(simulations_dict)
            f.attrs['export_timestamp'] = datetime.now().isoformat()
            
            # Create simulations group
            sims_group = f.create_group('simulations')
            
            grid_info = {
                'N': 128,
                'dx': 0.1,
                'extent': [-6.4, 6.4, -6.4, 6.4]
            }
            
            for sim_id, sim_data in simulations_dict.items():
                sim_group = sims_group.create_group(sim_id)
                
                # Store parameters as attributes
                for key, value in sim_data['params'].items():
                    if isinstance(value, (str, int, float)):
                        sim_group.attrs[key] = value
                
                # Store metadata
                for key, value in sim_data['metadata'].items():
                    if isinstance(value, (str, int, float)):
                        sim_group.attrs[f'meta_{key}'] = value
                
                # Store a sample frame (final frame)
                if sim_data['history']:
                    final_eta, final_stress = sim_data['history'][-1]
                    sim_group.create_dataset('final_defect', data=final_eta)
                    
                    # Store stress components
                    stress_group = sim_group.create_group('final_stress')
                    for key, value in final_stress.items():
                        if isinstance(value, np.ndarray):
                            stress_group.create_dataset(key, data=value)
        
        # Read file into bytes
        with open(temp_file.name, 'rb') as f:
            buffer = BytesIO(f.read())
        
        os.unlink(temp_file.name)
        buffer.seek(0)
        return buffer
    
    @staticmethod
    def _export_as_sqlite(simulations_dict):
        """Export all simulations into a single SQLite database"""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create master tables
        cursor.execute('''
            CREATE TABLE collection_metadata (
                export_id TEXT PRIMARY KEY,
                total_simulations INTEGER,
                export_timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE simulation_index (
                sim_id TEXT PRIMARY KEY,
                defect_type TEXT,
                orientation TEXT,
                eps0 REAL,
                kappa REAL,
                total_frames INTEGER
            )
        ''')
        
        # Insert collection metadata
        export_id = hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8]
        cursor.execute('''
            INSERT INTO collection_metadata VALUES (?, ?, ?)
        ''', (export_id, len(simulations_dict), datetime.now().isoformat()))
        
        # Insert each simulation into index
        for sim_id, sim_data in simulations_dict.items():
            params = sim_data['params']
            cursor.execute('''
                INSERT INTO simulation_index VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                sim_id,
                params['defect_type'],
                params.get('orientation', 'Unknown'),
                float(params['eps0']),
                float(params['kappa']),
                len(sim_data['history'])
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
# ENHANCED SIMULATIONDB WITH EXPORT CAPABILITIES
# =============================================
class EnhancedSimulationDB:
    """Extended simulation database with export methods"""
    
    @staticmethod
    def get_simulation_dataframe(sim_id):
        """Get simulation parameters as pandas DataFrame"""
        sim_data = SimulationDB.get_simulation(sim_id)
        if not sim_data:
            return None
        
        params = sim_data['params']
        
        df_data = {
            'simulation_id': [sim_id],
            'defect_type': [params['defect_type']],
            'seed_shape': [params['shape']],
            'habit_plane_angle_deg': [float(params.get('angle_deg', np.rad2deg(params.get('theta', 0))))],
            'orientation': [params.get('orientation', 'Unknown')],
            'eigenstrain_magnitude': [float(params['eps0'])],
            'interface_energy_coeff': [float(params['kappa'])],
            'total_steps': [params['steps']],
            'save_frequency': [params['save_every']],
            'total_frames': [len(sim_data['history'])],
            'run_time': [sim_data['metadata'].get('run_time', 0)],
            'created_at': [sim_data['created_at']]
        }
        
        return pd.DataFrame(df_data)
    
    @staticmethod
    def get_all_simulations_dataframe():
        """Get all simulations as a single DataFrame"""
        simulations = SimulationDB.get_all_simulations()
        
        if not simulations:
            return pd.DataFrame()
        
        all_data = []
        for sim_id, sim_data in simulations.items():
            params = sim_data['params']
            
            sim_dict = {
                'simulation_id': sim_id,
                'defect_type': params['defect_type'],
                'seed_shape': params['shape'],
                'habit_plane_angle_deg': float(params.get('angle_deg', np.rad2deg(params.get('theta', 0)))),
                'orientation': params.get('orientation', 'Unknown'),
                'eigenstrain_magnitude': float(params['eps0']),
                'interface_energy_coeff': float(params['kappa']),
                'total_steps': params['steps'],
                'save_frequency': params['save_every'],
                'total_frames': len(sim_data['history']),
                'run_time': sim_data['metadata'].get('run_time', 0),
                'created_at': sim_data['created_at']
            }
            all_data.append(sim_dict)
        
        return pd.DataFrame(all_data)

# =============================================
# STREAMLIT UI FOR DATA EXPORT
# =============================================
def create_export_ui():
    """Create the export interface in Streamlit"""
    
    st.sidebar.header("💾 ML Data Export")
    
    with st.sidebar.expander("📤 Export Single Simulation", expanded=True):
        simulations = SimulationDB.get_simulation_list()
        
        if simulations:
            # Create mapping for dropdown
            sim_options = {f"{sim['name']} (ID: {sim['id']})": sim['id'] for sim in simulations}
            selected_sim_name = st.selectbox("Select Simulation", list(sim_options.keys()))
            selected_sim_id = sim_options[selected_sim_name]
            
            # Export format selection
            export_format = st.selectbox(
                "Export Format",
                [
                    "PyTorch (.pt)",
                    "SQLite (.db)", 
                    "HDF5 (.h5)",
                    "Pickle (.pkl)",
                    "Compressed NumPy (.npz)",
                    "All Formats (ZIP)"
                ]
            )
            
            # Additional options
            with st.expander("Export Options"):
                include_all_frames = st.checkbox("Include all frames", True)
                compress_data = st.checkbox("Compress data", True)
                include_profiles = st.checkbox("Include line profiles", True)
            
            if st.button("📥 Export Selected Simulation", type="primary"):
                with st.spinner("Preparing export..."):
                    # Get simulation data
                    sim_data = SimulationDB.get_simulation(selected_sim_id)
                    
                    if sim_data:
                        # Prepare grid info
                        N = 128
                        dx = 0.1
                        extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
                        X, Y = np.meshgrid(
                            np.linspace(extent[0], extent[1], N),
                            np.linspace(extent[2], extent[3], N)
                        )
                        
                        grid_info = {
                            'N': N,
                            'dx': dx,
                            'extent': extent,
                            'X': X,
                            'Y': Y
                        }
                        
                        # Create structured data
                        structured_data = MLDataStructure.create_simulation_data(
                            selected_sim_id,
                            sim_data['params'],
                            sim_data['history'],
                            sim_data['metadata'],
                            grid_info
                        )
                        
                        # Export based on format
                        format_map = {
                            "PyTorch (.pt)": ("pytorch", "pt"),
                            "SQLite (.db)": ("sqlite", "db"),
                            "HDF5 (.h5)": ("hdf5", "h5"),
                            "Pickle (.pkl)": ("pickle", "pkl"),
                            "Compressed NumPy (.npz)": ("npz", "npz"),
                            "All Formats (ZIP)": ("all", "zip")
                        }
                        
                        fmt_key, ext = format_map[export_format]
                        
                        try:
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
                            
                            # Show success message
                            st.sidebar.success(f"""
                            ✅ Export ready!
                            - Format: {export_format}
                            - File: {filename}
                            - Size: {len(buffer.getvalue()) / 1024:.1f} KB
                            """)
                            
                        except Exception as e:
                            st.sidebar.error(f"Export failed: {str(e)}")
        
        else:
            st.info("No simulations available for export")
    
    with st.sidebar.expander("📚 Export All Simulations", expanded=False):
        simulations = SimulationDB.get_all_simulations()
        
        if simulations:
            st.info(f"Found {len(simulations)} simulations")
            
            export_format_all = st.selectbox(
                "Export Format for Collection",
                ["ZIP Archive", "Single HDF5 File", "Single SQLite Database"],
                key="export_all"
            )
            
            if st.button("📦 Export All Simulations", type="primary"):
                with st.spinner(f"Exporting {len(simulations)} simulations..."):
                    try:
                        format_map = {
                            "ZIP Archive": "zip",
                            "Single HDF5 File": "hdf5",
                            "Single SQLite Database": "sqlite"
                        }
                        
                        buffer = BatchExporter.export_all_simulations(
                            simulations,
                            format=format_map[export_format_all]
                        )
                        
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        ext = format_map[export_format_all]
                        filename = f"ag_np_simulations_collection_{timestamp}.{ext}"
                        
                        st.sidebar.download_button(
                            "⬇️ Download Collection",
                            buffer.getvalue() if hasattr(buffer, 'getvalue') else buffer,
                            filename,
                            f"application/{ext}"
                        )
                        
                        # Show summary
                        defect_counts = {}
                        for sim_id, sim_data in simulations.items():
                            defect_type = sim_data['params']['defect_type']
                            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                        
                        st.sidebar.success(f"""
                        ✅ Collection exported!
                        - Total simulations: {len(simulations)}
                        - Format: {export_format_all}
                        - File: {filename}
                        
                        Breakdown:
                        {', '.join([f'{k}: {v}' for k, v in defect_counts.items()])}
                        """)
                        
                    except Exception as e:
                        st.sidebar.error(f"Export failed: {str(e)}")
        else:
            st.info("No simulations available for export")

# =============================================
# MAIN APPLICATION INTEGRATION
# =============================================

# Add this after your existing sidebar code, before the simulation execution

# In your main sidebar section, add:
st.sidebar.header("🚀 Data Export & ML Integration")

export_tab = st.sidebar.selectbox(
    "Export Mode",
    ["Single Simulation Export", "Batch Export", "ML Data Preview"],
    index=0
)

if export_tab == "Single Simulation Export":
    create_export_ui()
    
elif export_tab == "Batch Export":
    simulations = SimulationDB.get_all_simulations()
    
    if simulations:
        st.sidebar.info(f"Database contains {len(simulations)} simulations")
        
        # Show summary statistics
        df = EnhancedSimulationDB.get_all_simulations_dataframe()
        
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.metric("Total", len(df))
        with col2:
            st.metric("ISF", len(df[df['defect_type'] == 'ISF']))
        with col3:
            st.metric("ESF", len(df[df['defect_type'] == 'ESF']))
        
        # Quick export options
        if st.sidebar.button("📊 Export Summary CSV", type="secondary"):
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                "⬇️ Download CSV",
                csv,
                "simulations_summary.csv",
                "text/csv"
            )
    
else:  # ML Data Preview
    st.sidebar.subheader("🧠 ML Data Structure Preview")
    
    simulations = SimulationDB.get_simulation_list()
    if simulations:
        selected_sim = st.sidebar.selectbox(
            "Preview Simulation",
            [sim['name'] for sim in simulations]
        )
        
        if st.sidebar.button("Show Data Structure", type="secondary"):
            sim_id = simulations[[sim['name'] for sim in simulations].index(selected_sim)]['id']
            sim_data = SimulationDB.get_simulation(sim_id)
            
            if sim_data:
                # Show data structure
                st.sidebar.code(f"""
                Simulation Data Structure:
                -------------------------
                ID: {sim_id}
                Defect: {sim_data['params']['defect_type']}
                Frames: {len(sim_data['history'])}
                
                Available Data:
                - η (defect field): {sim_data['history'][0][0].shape}
                - Stress components: {list(sim_data['history'][0][1].keys())}
                - Strain components: exx, eyy, exy
                - Eigenstrains: eps_xx_star, eps_yy_star, eps_xy_star
                
                Export Formats:
                - PyTorch (.pt): Direct ML training
                - SQLite (.db): Queryable database
                - HDF5 (.h5): Hierarchical storage
                - Pickle (.pkl): Python object
                - NPZ (.npz): Compressed arrays
                """)

# =============================================
# EXAMPLE ML USAGE SECTION
# =============================================
with st.expander("🧠 ML Usage Examples", expanded=False):
    st.markdown("""
    ### How to Use Exported Data for Machine Learning
    
    **PyTorch Format (.pt):**
    ```python
    import torch
    
    # Load simulation data
    data = torch.load('simulation_abc123.pt')
    
    # Access different components
    defect_fields = data['defect_fields']  # Shape: (frames, 128, 128)
    stress = data['stress_components']['sigma_mag']  # Same shape
    params = data['parameters']  # Simulation parameters
    
    # Create ML dataset
    class DefectDataset(torch.utils.data.Dataset):
        def __init__(self, defect_fields, stress_fields):
            self.defect = defect_fields
            self.stress = stress_fields
            
        def __len__(self):
            return len(self.defect)
        
        def __getitem__(self, idx):
            return {
                'defect': self.defect[idx].unsqueeze(0),  # Add channel dim
                'stress': self.stress[idx].unsqueeze(0),
                'defect_type': torch.tensor([params['defect_type_code']]),
                'eps0': torch.tensor([params['eigenstrain_magnitude']])
            }
    
    dataset = DefectDataset(defect_fields, stress)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    ```
    
    **SQLite Format (.db):**
    ```python
    import sqlite3
    import numpy as np
    
    conn = sqlite3.connect('simulation.db')
    cursor = conn.cursor()
    
    # Query metadata
    cursor.execute('SELECT * FROM simulation_metadata')
    metadata = cursor.fetchone()
    
    # Query frames
    cursor.execute('SELECT defect_field, stress_sigma_mag FROM simulation_frames WHERE sim_id=?', ('abc123',))
    frames = cursor.fetchall()
    
    # Convert BLOB to numpy array
    for defect_blob, stress_blob in frames:
        defect = np.frombuffer(defect_blob, dtype=np.float32).reshape((128, 128))
        stress = np.frombuffer(stress_blob, dtype=np.float32).reshape((128, 128))
    ```
    
    **Training a CNN for Stress Prediction:**
    ```python
    import torch.nn as nn
    
    class StressPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1)
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    # Training loop
    model = StressPredictor()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for batch in dataloader:
        defect = batch['defect']
        target_stress = batch['stress']
        
        predicted = model(defect)
        loss = criterion(predicted, target_stress)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```
    
    **GitHub Repository Structure:**
    ```
    ag-nanoparticle-ml/
    ├── data/
    │   ├── raw/          # Raw .pt files
    │   ├── processed/    # Processed datasets
    │   └── metadata.csv  # Simulation parameters
    ├── models/
    │   ├── cnn_stress.py
    │   └── unet_defect.py
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   └── 02_model_training.ipynb
    ├── scripts/
    │   ├── train.py
    │   └── evaluate.py
    └── README.md
    ```
    """)

# =============================================
# INTEGRATE WITH EXISTING MAIN CONTENT
# =============================================

# In your existing main content area, after running simulations,
# you can add an export section:

# After successful simulation run, add export options:
def add_simulation_export_section(sim_id, sim_data):
    """Add export section after simulation completion"""
    with st.expander("💾 Export Simulation Data", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Quick export buttons
            if st.button("📦 Export as PyTorch", key=f"export_pt_{sim_id}"):
                # Prepare grid info
                N = 128
                dx = 0.1
                extent = [-N*dx/2, N*dx/2, -N*dx/2, N*dx/2]
                X, Y = np.meshgrid(
                    np.linspace(extent[0], extent[1], N),
                    np.linspace(extent[2], extent[3], N)
                )
                
                grid_info = {
                    'N': N,
                    'dx': dx,
                    'extent': extent,
                    'X': X,
                    'Y': Y
                }
                
                structured_data = MLDataStructure.create_simulation_data(
                    sim_id,
                    sim_data['params'],
                    sim_data['history'],
                    sim_data['metadata'],
                    grid_info
                )
                
                buffer = DataExporter.export_pytorch(structured_data)
                st.download_button(
                    "⬇️ Download .pt file",
                    buffer.getvalue(),
                    f"{sim_id}.pt",
                    "application/octet-stream"
                )
        
        with col2:
            # Show data statistics
            params = sim_data['params']
            st.info(f"""
            **Simulation Data:**
            - Defect: {params['defect_type']}
            - Shape: {params['shape']}
            - Angle: {params.get('angle_deg', np.rad2deg(params.get('theta', 0))):.1f}°
            - Frames: {len(sim_data['history'])}
            - Grid: 128×128 ({sim_data['history'][0][0].size} points/frame)
            """)
