# =============================================
# ADD IMPORTS AT THE TOP OF YOUR FILE
# =============================================
import pickle
import torch
import h5py
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import traceback
import os
import sys

# If these are in your original imports, keep them too:
import streamlit as st
from numba import jit, prange
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import zipfile
import time
import hashlib
from scipy import stats, interpolate
from scipy.ndimage import gaussian_filter, rotate
import warnings
import tempfile
import base64
import glob

# =============================================
# ENHANCED FILE READER FOR SIMULATION OUTPUTS - FIXED VERSION
# =============================================
class EnhancedSimulationFileReader:
    """Enhanced reader for simulation output files with robust error handling"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.file_history = {}
        self._initialize_dependencies()
        
    def _initialize_dependencies(self):
        """Lazy import of optional dependencies"""
        self.dependencies = {
            'dill_available': False,
            'joblib_available': False,
            'msgpack_available': False,
            'dill': None,
            'joblib': None,
            'msgpack': None
        }
        
        try:
            import dill
            self.dependencies['dill'] = dill
            self.dependencies['dill_available'] = True
        except ImportError:
            if self.debug:
                print("⚠️ Dill not available, using pickle fallback")
        
        try:
            import joblib
            self.dependencies['joblib'] = joblib
            self.dependencies['joblib_available'] = True
        except ImportError:
            if self.debug:
                print("⚠️ Joblib not available")
        
        try:
            import msgpack
            self.dependencies['msgpack'] = msgpack
            self.dependencies['msgpack_available'] = True
        except ImportError:
            if self.debug:
                print("⚠️ Msgpack not available")
    
    def read_file(self, file_path_or_content, file_format=None):
        """
        Universal file reader for simulation outputs
        
        Args:
            file_path_or_content: Either a file path (str/Path) or file content (bytes)
            file_format: Optional format hint (pkl, pt, h5, etc.)
            
        Returns:
            Standardized simulation data
        """
        try:
            # Determine if input is file path or content
            if isinstance(file_path_or_content, (str, Path)):
                file_path = Path(file_path_or_content)
                return self.read_file_by_path(file_path, file_format)
            else:
                # Assume it's file content (bytes)
                return self.read_file_content(file_path_or_content, file_format)
                
        except Exception as e:
            if self.debug:
                print(f"Error reading file: {e}")
                import traceback
                traceback.print_exc()
            raise
    
    def read_file_by_path(self, file_path, format_hint=None):
        """Read simulation file from disk"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine format
        if format_hint:
            file_format = format_hint.lower()
        else:
            file_format = self.detect_format(file_path)
        
        # Read based on format
        if file_format == 'pkl':
            return self._read_pkl_file(file_path)
        elif file_format == 'pt':
            return self._read_pt_file(file_path)
        elif file_format in ['h5', 'hdf5']:
            return self._read_h5_file(file_path)
        elif file_format == 'npz':
            return self._read_npz_file(file_path)
        elif file_format == 'json':
            return self._read_json_file(file_path)
        elif file_format in ['sql', 'db']:
            return self._read_sql_file(file_path)
        elif file_format == 'npy':
            return self._read_npy_file(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    def read_file_content(self, file_content, format_hint=None):
        """Read simulation file from memory"""
        # Create a temporary file-like object
        buffer = BytesIO(file_content)
        
        # Try to detect format from content
        if format_hint:
            file_format = format_hint.lower()
        else:
            # Try to guess from content
            file_format = self.detect_format_from_content(file_content[:100])
        
        # Read based on format
        if file_format == 'pkl':
            try:
                return pickle.load(buffer)
            except Exception as e:
                # Try alternative pickle readers
                return self._read_pkl_alternative_buffer(buffer)
        elif file_format == 'pt':
            buffer.seek(0)
            try:
                return torch.load(buffer, map_location=torch.device('cpu'))
            except Exception as e:
                if self.debug:
                    print(f"Error loading PT from buffer: {e}")
                raise
        elif file_format in ['h5', 'hdf5']:
            buffer.seek(0)
            try:
                with h5py.File(buffer, 'r') as f:
                    return self._extract_h5_data(f)
            except Exception as e:
                if self.debug:
                    print(f"Error loading H5 from buffer: {e}")
                raise
        elif file_format == 'npz':
            buffer.seek(0)
            try:
                return dict(np.load(buffer, allow_pickle=True))
            except Exception as e:
                if self.debug:
                    print(f"Error loading NPZ from buffer: {e}")
                raise
        elif file_format == 'json':
            try:
                return json.loads(file_content.decode('utf-8'))
            except Exception as e:
                if self.debug:
                    print(f"Error loading JSON from buffer: {e}")
                raise
        elif file_format in ['sql', 'db']:
            try:
                return self._read_sql_content(file_content.decode('utf-8'))
            except Exception as e:
                if self.debug:
                    print(f"Error loading SQL from buffer: {e}")
                raise
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    def _read_pkl_alternative_buffer(self, buffer):
        """Alternative method for reading problematic pickle files from buffer"""
        buffer.seek(0)
        content = buffer.read()
        
        # Try with dill
        if self.dependencies['dill_available']:
            try:
                buffer.seek(0)
                return self.dependencies['dill'].load(buffer)
            except:
                pass
        
        # Try with joblib
        if self.dependencies['joblib_available']:
            try:
                from io import BytesIO
                temp_buffer = BytesIO(content)
                return self.dependencies['joblib'].load(temp_buffer)
            except:
                pass
        
        # Try different pickle protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                buffer.seek(0)
                unpickler = pickle.Unpickler(buffer)
                return unpickler.load()
            except:
                continue
        
        # Try as raw bytes if all else fails
        try:
            return content
        except:
            raise ValueError("Could not read pickle file with any method")
    
    def detect_format(self, file_path):
        """Detect file format from extension"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        format_map = {
            '.pkl': 'pkl',
            '.pickle': 'pkl',
            '.pt': 'pt',
            '.pth': 'pt',
            '.h5': 'h5',
            '.hdf5': 'h5',
            '.npz': 'npz',
            '.json': 'json',
            '.sql': 'sql',
            '.db': 'sql',
            '.npy': 'npy'
        }
        
        if ext in format_map:
            return format_map[ext]
        
        # Try to guess from content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(100)
            return self.detect_format_from_content(header)
        except:
            raise ValueError(f"Cannot detect format for file: {file_path}")
    
    def detect_format_from_content(self, header_bytes):
        """Detect format from file header"""
        if len(header_bytes) < 8:
            return 'unknown'
        
        # Check for pickle/PyTorch (common pickle magic numbers)
        pickle_magic_numbers = [
            b'\x80\x02',  # Protocol 2
            b'\x80\x03',  # Protocol 3
            b'\x80\x04',  # Protocol 4
            b'\x80\x05',  # Protocol 5
        ]
        
        for magic in pickle_magic_numbers:
            if header_bytes.startswith(magic):
                return 'pkl'
        
        # Check for JSON
        try:
            # Try to decode as UTF-8 and check if it starts with { or [
            decoded = header_bytes.decode('utf-8', errors='ignore').strip()
            if decoded.startswith('{') or decoded.startswith('['):
                return 'json'
        except:
            pass
        
        # Check for HDF5
        if header_bytes[:8] == b'\x89HDF\r\n\x1a\n':
            return 'h5'
        
        # Check for NumPy NPY format
        if header_bytes[:6] == b'\x93NUMPY':
            return 'npy'
        
        # Check for NPZ (compressed numpy)
        if header_bytes[:2] == b'PK':  # ZIP file header
            return 'npz'
        
        # Default to pickle (most common for your simulations)
        return 'pkl'
    
    def _read_pkl_file(self, file_path):
        """Read pickle file with enhanced error handling"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if self.debug:
                print(f"PKL file loaded successfully: {file_path.name}")
                print(f"Data type: {type(data)}")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                    if 'history' in data:
                        print(f"History length: {len(data['history'])}")
            
            return self._standardize_pkl_data(data)
            
        except Exception as e:
            if self.debug:
                print(f"Error reading PKL file {file_path}: {e}")
            # Try alternative pickle reading methods
            return self._read_pkl_alternative(file_path)
    
    def _read_pkl_alternative(self, file_path):
        """Alternative method for reading problematic pickle files"""
        # Try with dill (handles more complex objects)
        if self.dependencies['dill_available']:
            try:
                with open(file_path, 'rb') as f:
                    data = self.dependencies['dill'].load(f)
                return self._standardize_pkl_data(data)
            except Exception as e:
                if self.debug:
                    print(f"Dill failed: {e}")
        
        # Try with joblib
        if self.dependencies['joblib_available']:
            try:
                data = self.dependencies['joblib'].load(file_path)
                return self._standardize_pkl_data(data)
            except Exception as e:
                if self.debug:
                    print(f"Joblib failed: {e}")
        
        # Try with different pickle protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                with open(file_path, 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    data = unpickler.load()
                return self._standardize_pkl_data(data)
            except:
                continue
        
        # If all else fails, try to read as binary and guess
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Try to decode as JSON if it looks like text
            try:
                decoded = content.decode('utf-8')
                if decoded.strip().startswith('{') or decoded.strip().startswith('['):
                    data = json.loads(decoded)
                    return self._standardize_pkl_data(data)
            except:
                pass
            
            # Return raw bytes as last resort
            return {'raw_bytes': content, 'error': 'Could not parse with any method'}
            
        except Exception as e:
            raise ValueError(f"All reading methods failed for {file_path}: {e}")
    
    def _read_pt_file(self, file_path):
        """Read PyTorch file with tensor conversion"""
        try:
            # Load PyTorch file
            data = torch.load(file_path, map_location=torch.device('cpu'))
            
            if self.debug:
                print(f"PT file loaded successfully: {file_path.name}")
                print(f"Data type: {type(data)}")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
            
            # Convert all tensors to numpy
            return self._convert_tensors_to_numpy(data)
            
        except Exception as e:
            if self.debug:
                print(f"Error reading PT file {file_path}: {e}")
            raise
    
    def _read_h5_file(self, file_path):
        """Read HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                data = self._extract_h5_data(f)
            
            if self.debug:
                print(f"H5 file loaded successfully: {file_path.name}")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
            
            return data
            
        except Exception as e:
            if self.debug:
                print(f"Error reading H5 file {file_path}: {e}")
            raise
    
    def _extract_h5_data(self, h5_file):
        """Extract data from HDF5 file"""
        data = {}
        
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Convert to numpy array
                data[name] = obj[()]
            elif isinstance(obj, h5py.Group):
                data[name] = {}
                for key in obj.keys():
                    if isinstance(obj[key], h5py.Dataset):
                        data[name][key] = obj[key][()]
        
        h5_file.visititems(visitor)
        return data
    
    def _read_npz_file(self, file_path):
        """Read numpy compressed file"""
        try:
            data = dict(np.load(file_path, allow_pickle=True))
            
            if self.debug:
                print(f"NPZ file loaded successfully: {file_path.name}")
                print(f"Keys: {list(data.keys())}")
            
            return data
            
        except Exception as e:
            if self.debug:
                print(f"Error reading NPZ file {file_path}: {e}")
            raise
    
    def _read_json_file(self, file_path):
        """Read JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.debug:
                print(f"JSON file loaded successfully: {file_path.name}")
                print(f"Keys: {list(data.keys())}")
            
            return data
            
        except Exception as e:
            if self.debug:
                print(f"Error reading JSON file {file_path}: {e}")
            raise
    
    def _read_sql_file(self, file_path):
        """Read SQLite database file"""
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            data = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                data[table_name] = {
                    'columns': columns,
                    'rows': rows
                }
            
            conn.close()
            
            if self.debug:
                print(f"SQL file loaded successfully: {file_path.name}")
                print(f"Tables: {list(data.keys())}")
            
            return data
            
        except Exception as e:
            if self.debug:
                print(f"Error reading SQL file {file_path}: {e}")
            raise
    
    def _read_sql_content(self, sql_content):
        """Read SQL dump content"""
        try:
            conn = sqlite3.connect(':memory:')
            conn.executescript(sql_content)
            cursor = conn.cursor()
            
            # Similar extraction as _read_sql_file
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            data = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [col[1] for col in cursor.fetchall()]
                
                data[table_name] = {
                    'columns': columns,
                    'rows': rows
                }
            
            conn.close()
            return data
            
        except Exception as e:
            if self.debug:
                print(f"Error reading SQL content: {e}")
            raise
    
    def _read_npy_file(self, file_path):
        """Read numpy array file"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            if self.debug:
                print(f"NPY file loaded successfully: {file_path.name}")
                print(f"Data type: {type(data)}")
                if isinstance(data, np.ndarray):
                    print(f"Shape: {data.shape}")
            
            return data
            
        except Exception as e:
            if self.debug:
                print(f"Error reading NPY file {file_path}: {e}")
            raise
    
    def _convert_tensors_to_numpy(self, obj):
        """Recursively convert PyTorch tensors to numpy arrays"""
        if torch.is_tensor(obj):
            try:
                return obj.numpy()
            except:
                # For tensors that can't be converted to numpy (e.g., on GPU)
                return obj.cpu().numpy() if obj.is_cuda else obj.detach().numpy()
        elif isinstance(obj, dict):
            return {k: self._convert_tensors_to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_tensors_to_numpy(item) for item in obj)
        else:
            return obj
    
    def _standardize_pkl_data(self, data):
        """
        Standardize PKL data structure to match your simulation format
        
        Your simulation exports data in this structure:
        {
            'params': {...},
            'history': [
                {
                    'eta': numpy_array,
                    'stresses': {
                        'sxx': numpy_array,
                        'syy': numpy_array,
                        'sxy': numpy_array,
                        'szz': numpy_array,
                        'sigma_mag': numpy_array,
                        'sigma_hydro': numpy_array,
                        'von_mises': numpy_array
                    }
                },
                ...
            ],
            'metadata': {...}
        }
        """
        standardized = {
            'params': {},
            'history': [],
            'metadata': {},
            'raw_data': data  # Keep original for reference
        }
        
        if isinstance(data, dict):
            # Extract parameters
            if 'params' in data:
                standardized['params'] = data['params']
            elif 'simulation_parameters' in data:
                standardized['params'] = data['simulation_parameters']
            elif 'parameters' in data:
                standardized['params'] = data['parameters']
            
            # Extract metadata
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            elif 'info' in data:
                standardized['metadata'] = data['info']
            
            # Extract history
            if 'history' in data:
                history_data = data['history']
                
                if isinstance(history_data, list):
                    for frame in history_data:
                        if isinstance(frame, dict):
                            # Your format: {'eta': ..., 'stresses': ...}
                            eta = frame.get('eta')
                            stresses = frame.get('stresses', {})
                            standardized['history'].append((eta, stresses))
                        elif isinstance(frame, tuple) and len(frame) == 2:
                            # Alternative format: (eta, stresses)
                            eta, stresses = frame
                            standardized['history'].append((eta, stresses))
                        elif isinstance(frame, np.ndarray):
                            # Just eta array
                            standardized['history'].append((frame, {}))
            
            # Try to extract from raw data if not found in standard structure
            if not standardized['params']:
                # Look for common parameter fields
                param_candidates = ['defect_type', 'shape', 'eps0', 'kappa', 'orientation', 
                                  'theta', 'steps', 'save_every', 'dt', 'dx', 'N']
                for key in param_candidates:
                    if key in data:
                        standardized['params'][key] = data[key]
            
            # Try to extract stress fields directly
            if not standardized['history']:
                stress_keys = ['sigma_hydro', 'sigma_mag', 'von_mises', 'sxx', 'syy', 'sxy', 'szz']
                found_stress = any(key in data for key in stress_keys)
                
                if found_stress:
                    eta = data.get('eta', data.get('eta_field', np.zeros((128, 128))))
                    stresses = {}
                    for key in stress_keys:
                        if key in data:
                            stresses[key] = data[key]
                    standardized['history'].append((eta, stresses))
        
        elif isinstance(data, np.ndarray):
            # Single array
            standardized['history'].append((data, {}))
            standardized['metadata']['array_shape'] = str(data.shape)
            standardized['metadata']['array_dtype'] = str(data.dtype)
        
        # Ensure required fields
        if not standardized['metadata']:
            standardized['metadata'] = {
                'loaded_at': datetime.now().isoformat(),
                'frames': len(standardized['history']),
                'standardized': True
            }
        
        # Add default grid parameters if missing
        if 'N' not in standardized['params']:
            standardized['params']['N'] = 128
        if 'dx' not in standardized['params']:
            standardized['params']['dx'] = 0.1
        
        if self.debug:
            print(f"Standardized data structure:")
            print(f"  Parameters: {len(standardized['params'])} keys")
            print(f"  History: {len(standardized['history'])} frames")
            print(f"  Metadata: {len(standardized['metadata'])} keys")
        
        return standardized
    
    def analyze_file_structure(self, file_path):
        """
        Analyze and report file structure for debugging
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}
        
        analysis = {
            'file_path': str(file_path),
            'file_size': f"{file_path.stat().st_size / 1024:.2f} KB",
            'file_format': self.detect_format(file_path),
            'analysis_time': datetime.now().isoformat()
        }
        
        try:
            # Load file
            data = self.read_file(file_path)
            
            # Analyze structure
            analysis['data_type'] = type(data).__name__
            
            if isinstance(data, dict):
                analysis['keys'] = list(data.keys())
                analysis['key_count'] = len(data.keys())
                
                # Check for simulation structure
                if 'history' in data:
                    history = data['history']
                    analysis['history_type'] = type(history).__name__
                    analysis['history_length'] = len(history) if hasattr(history, '__len__') else 'N/A'
                    
                    if history and len(history) > 0:
                        first_frame = history[0]
                        analysis['frame_type'] = type(first_frame).__name__
                        
                        if isinstance(first_frame, dict):
                            analysis['frame_keys'] = list(first_frame.keys())
                        elif isinstance(first_frame, tuple):
                            analysis['frame_length'] = len(first_frame)
                
                if 'params' in data:
                    params = data['params']
                    analysis['params_type'] = type(params).__name__
                    if isinstance(params, dict):
                        analysis['param_keys'] = list(params.keys())
                
                if 'metadata' in data:
                    analysis['has_metadata'] = True
                    metadata = data['metadata']
                    if isinstance(metadata, dict):
                        analysis['metadata_keys'] = list(metadata.keys())
            
            elif isinstance(data, np.ndarray):
                analysis['array_shape'] = str(data.shape)
                analysis['array_dtype'] = str(data.dtype)
            
            analysis['success'] = True
            
        except Exception as e:
            analysis['success'] = False
            analysis['error'] = str(e)
            analysis['error_type'] = type(e).__name__
        
        return analysis
    
    def batch_load_directory(self, directory_path, pattern="*", max_files=None):
        """
        Batch load files from directory
        """
        directory = Path(directory_path)
        if not directory.exists():
            return []
        
        files = list(directory.glob(pattern))
        if max_files:
            files = files[:max_files]
        
        loaded_files = []
        failed_files = []
        
        for file_path in files:
            try:
                if self.debug:
                    print(f"Loading: {file_path.name}")
                
                data = self.read_file(file_path)
                loaded_files.append({
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'data': data,
                    'success': True
                })
                
            except Exception as e:
                failed_files.append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
                if self.debug:
                    print(f"Failed to load {file_path.name}: {e}")
        
        if self.debug:
            print(f"Batch load completed:")
            print(f"  Success: {len(loaded_files)} files")
            print(f"  Failed: {len(failed_files)} files")
        
        return {
            'successful': loaded_files,
            'failed': failed_files,
            'total_attempted': len(files)
        }
    
    def export_summary_report(self, directory_path, output_file=None):
        """
        Generate summary report of all files in directory
        """
        directory = Path(directory_path)
        if not directory.exists():
            return None
        
        files = list(directory.glob("*"))
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'directory': str(directory),
            'total_files': len(files),
            'file_analysis': []
        }
        
        for file_path in files:
            try:
                analysis = self.analyze_file_structure(file_path)
                report['file_analysis'].append(analysis)
            except Exception as e:
                report['file_analysis'].append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
        
        # Group by format
        formats = {}
        for analysis in report['file_analysis']:
            fmt = analysis.get('file_format', 'unknown')
            if fmt not in formats:
                formats[fmt] = 0
            formats[fmt] += 1
        
        report['format_summary'] = formats
        
        # Save report if output file specified
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report

# =============================================
# FIXED MAIN APPLICATION STRUCTURE
# =============================================
def create_file_reader_interface():
    """Create Streamlit interface for reading and analyzing simulation files"""
    
    st.header("📂 Simulation File Reader")
    
    # Initialize enhanced reader
    if 'file_reader' not in st.session_state:
        st.session_state.file_reader = EnhancedSimulationFileReader(debug=True)
    
    # Initialize scanner
    if 'directory_scanner' not in st.session_state:
        scanner = NumericalSolutionsScanner()
        scanner.file_reader.debug = True
        st.session_state.directory_scanner = scanner
    
    # Operation mode
    mode = st.radio(
        "Select operation mode:",
        ["📤 Upload & Read Files", 
         "📁 Load from Directory", 
         "🔍 Analyze File Structure",
         "📊 Batch Processing"],
        horizontal=True
    )
    
    if mode == "📤 Upload & Read Files":
        st.subheader("Upload Simulation Files")
        
        uploaded_files = st.file_uploader(
            "Upload simulation files",
            type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'json', 'npy', 'sql', 'db'],
            accept_multiple_files=True,
            help="Upload files generated by the simulation"
        )
        
        if uploaded_files:
            # File format selection
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_file = st.selectbox(
                    "Select file to analyze",
                    [f.name for f in uploaded_files]
                )
            with col2:
                format_hint = st.selectbox(
                    "Format hint",
                    ["Auto", "PKL", "PT", "H5", "NPZ", "JSON"],
                    index=0
                )
            
            if st.button("📖 Read Selected File", type="primary"):
                # Find the selected file
                for uploaded_file in uploaded_files:
                    if uploaded_file.name == selected_file:
                        with st.spinner(f"Reading {uploaded_file.name}..."):
                            try:
                                # Read file
                                if format_hint == "Auto":
                                    data = st.session_state.file_reader.read_file(
                                        uploaded_file.getvalue()
                                    )
                                else:
                                    data = st.session_state.file_reader.read_file(
                                        uploaded_file.getvalue(),
                                        format_hint.lower()
                                    )
                                
                                # Display success
                                st.success(f"✅ Successfully read {uploaded_file.name}")
                                
                                # Store in session state
                                if 'loaded_data' not in st.session_state:
                                    st.session_state.loaded_data = {}
                                st.session_state.loaded_data[uploaded_file.name] = data
                                
                                # Display summary
                                self._display_file_summary(data, uploaded_file.name)
                                
                            except Exception as e:
                                st.error(f"❌ Error reading file: {str(e)}")
                                st.code(traceback.format_exc())
                        break
    
    elif mode == "📁 Load from Directory":
        # ... (rest of your directory loading code remains the same)
        pass
    
    elif mode == "🔍 Analyze File Structure":
        # ... (rest of your analysis code remains the same)
        pass
    
    elif mode == "📊 Batch Processing":
        # ... (rest of your batch processing code remains the same)
        pass
    
    # Add debugging information
    with st.expander("🐛 Debug Information", expanded=False):
        st.subheader("File Reader Status")
        
        if 'loaded_data' in st.session_state:
            st.write(f"Loaded files: {len(st.session_state.loaded_data)}")
        
        st.write("### Reader Configuration")
        st.write(f"Debug mode: {st.session_state.file_reader.debug}")
        
        if st.button("Clear Cache"):
            if 'loaded_data' in st.session_state:
                del st.session_state.loaded_data
            if 'directory_scanner' in st.session_state:
                st.session_state.directory_scanner.clear_cache()
            st.success("Cache cleared!")
            st.rerun()

    def _display_file_summary(self, data, filename):
        """Helper method to display file summary"""
        with st.expander("📋 File Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            if isinstance(data, dict):
                with col1:
                    st.metric("Keys", len(data.keys()))
                with col2:
                    if 'history' in data:
                        st.metric("Frames", len(data['history']))
                    else:
                        st.metric("Frames", 0)
                with col3:
                    if 'params' in data:
                        st.metric("Parameters", len(data['params']))
                    else:
                        st.metric("Parameters", 0)
                
                # Show parameters
                if 'params' in data:
                    st.subheader("Simulation Parameters")
                    params_df = pd.DataFrame(
                        list(data['params'].items()),
                        columns=['Parameter', 'Value']
                    )
                    st.dataframe(params_df, use_container_width=True)
                
                # Show metadata
                if 'metadata' in data:
                    st.subheader("Metadata")
                    metadata_df = pd.DataFrame(
                        list(data['metadata'].items()),
                        columns=['Key', 'Value']
                    )
                    st.dataframe(metadata_df, use_container_width=True)
            
            # Visualize if there's image data
            if isinstance(data, dict) and 'history' in data and data['history']:
                self._visualize_data(data)

    def _visualize_data(self, data):
        """Helper method to visualize simulation data"""
        st.subheader("📊 Data Visualization")
        
        frame_idx = st.slider(
            "Select frame",
            0, len(data['history']) - 1, 
            len(data['history']) - 1
        )
        
        if data['history']:
            if isinstance(data['history'][0], tuple):
                eta, stresses = data['history'][frame_idx]
                
                # Display eta field
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(eta, cmap='viridis', aspect='auto')
                ax.set_title(f"η Field - Frame {frame_idx}")
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                
                # Display stress fields if available
                if stresses and isinstance(stresses, dict):
                    st.subheader("Stress Fields")
                    stress_keys = list(stresses.keys())
                    cols = min(3, len(stress_keys))
                    rows = (len(stress_keys) + cols - 1) // cols
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
                    if rows == 1 and cols == 1:
                        axes = np.array([[axes]])
                    elif rows == 1:
                        axes = axes.reshape(1, -1)
                    elif cols == 1:
                        axes = axes.reshape(-1, 1)
                    
                    for idx, key in enumerate(stress_keys[:rows*cols]):
                        row = idx // cols
                        col = idx % cols
                        ax = axes[row, col]
                        
                        if key in stresses:
                            stress_data = stresses[key]
                            im = ax.imshow(stress_data, cmap='coolwarm', aspect='auto')
                            ax.set_title(key)
                            plt.colorbar(im, ax=ax)
                        else:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                            ax.set_title(key)
                    
                    # Hide empty subplots
                    for idx in range(len(stress_keys), rows*cols):
                        row = idx // cols
                        col = idx % cols
                        axes[row, col].axis('off')
                    
                    st.pyplot(fig)

# =============================================
# FIXED MAIN ENTRY POINT
# =============================================
def main():
    """Main application with enhanced file reading"""
    
    # Set page config FIRST
    st.set_page_config(
        page_title="Ag NP Multi-Defect Analyzer with Attention",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar for mode selection
    with st.sidebar:
        st.header("🔧 Operation Mode")
        
        operation_mode = st.radio(
            "Select Mode",
            ["Run New Simulation", 
             "Compare Saved Simulations", 
             "Single Simulation View", 
             "Attention Interpolation",
             "📂 File Reader"],
            index=4  # Default to File Reader
        )
    
    # Main content based on selection
    if operation_mode == "📂 File Reader":
        create_file_reader_interface()
    else:
        # For other modes, show a placeholder or call the appropriate function
        st.warning(f"Mode '{operation_mode}' is under development. Please use '📂 File Reader' for now.")

# Update the main call
if __name__ == "__main__":
    main()
