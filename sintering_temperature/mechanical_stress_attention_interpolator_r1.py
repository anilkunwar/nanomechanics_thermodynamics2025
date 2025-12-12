# =============================================
# ENHANCED FILE READER FOR SIMULATION OUTPUTS
# =============================================
class EnhancedSimulationFileReader:
    """Enhanced reader for simulation output files with robust error handling"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.file_history = {}
        
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
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    def read_file_content(self, file_content, format_hint=None):
        """Read simulation file from memory"""
        from io import BytesIO, StringIO
        
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
            return pickle.load(buffer)
        elif file_format == 'pt':
            buffer.seek(0)
            return torch.load(buffer, map_location=torch.device('cpu'))
        elif file_format in ['h5', 'hdf5']:
            buffer.seek(0)
            with h5py.File(buffer, 'r') as f:
                return self._extract_h5_data(f)
        elif file_format == 'npz':
            buffer.seek(0)
            return dict(np.load(buffer, allow_pickle=True))
        elif file_format == 'json':
            return json.loads(file_content.decode('utf-8'))
        elif file_format in ['sql', 'db']:
            buffer.seek(0)
            return self._read_sql_content(file_content.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
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
            '.db': 'sql'
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
        # Check for pickle
        if b'defect_type' in header_bytes or b'params' in header_bytes:
            return 'pkl'
        # Check for PyTorch (magic number)
        if len(header_bytes) >= 8 and header_bytes[:8] == b'\x80\x02\x7d\x71\x00\x58':
            return 'pkl'
        # Check for JSON
        if header_bytes.startswith(b'{') or header_bytes.startswith(b'['):
            return 'json'
        # Check for HDF5
        if header_bytes[:8] == b'\x89HDF\r\n\x1a\n':
            return 'h5'
        # Default to pickle
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
        try:
            # Try with dill (handles more complex objects)
            import dill
            with open(file_path, 'rb') as f:
                data = dill.load(f)
            return self._standardize_pkl_data(data)
        except:
            # Try with joblib
            try:
                import joblib
                data = joblib.load(file_path)
                return self._standardize_pkl_data(data)
            except:
                # Try with different protocols
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    try:
                        with open(file_path, 'rb') as f:
                            unpickler = pickle.Unpickler(f)
                            data = unpickler.load()
                        return self._standardize_pkl_data(data)
                    except:
                        continue
                raise
    
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
            import sqlite3
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
            import sqlite3
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
    
    def _convert_tensors_to_numpy(self, obj):
        """Recursively convert PyTorch tensors to numpy arrays"""
        if torch.is_tensor(obj):
            return obj.numpy()
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
            
            # Extract metadata
            if 'metadata' in data:
                standardized['metadata'] = data['metadata']
            
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
                param_candidates = ['defect_type', 'shape', 'eps0', 'kappa', 'orientation']
                for key in param_candidates:
                    if key in data:
                        standardized['params'][key] = data[key]
            
            # Try to extract stress fields directly
            if not standardized['history']:
                stress_keys = ['sigma_hydro', 'sigma_mag', 'von_mises', 'sxx', 'syy', 'sxy']
                found_stress = any(key in data for key in stress_keys)
                
                if found_stress:
                    eta = data.get('eta', np.zeros((128, 128)))
                    stresses = {}
                    for key in stress_keys:
                        if key in data:
                            stresses[key] = data[key]
                    standardized['history'].append((eta, stresses))
        
        elif isinstance(data, np.ndarray):
            # Single array
            standardized['history'].append((data, {}))
            standardized['metadata']['array_shape'] = str(data.shape)
        
        # Ensure required fields
        if not standardized['metadata']:
            standardized['metadata'] = {
                'loaded_at': datetime.now().isoformat(),
                'frames': len(standardized['history']),
                'standardized': True
            }
        
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
# INTEGRATE ENHANCED READER INTO YOUR EXISTING CODE
# =============================================

# Update the NumericalSolutionsScanner class to use EnhancedSimulationFileReader
class NumericalSolutionsScanner:
    """Scan and load simulation files from numerical_solutions directory"""
    
    def __init__(self, base_dir=None):
        """
        Args:
            base_dir: Base directory containing numerical solutions (overrides environment variable)
        """
        # Read SIM_DIR environment variable or fallback
        env_dir = os.getenv('SIM_DIR', 'numerical_solutions')

        # Choose base_dir → priority: provided > ENV > default
        self.base_dir = Path(base_dir or env_dir).resolve()
        
        # Create folder if missing
        if not self.base_dir.exists():
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {self.base_dir}")
            except Exception as e:
                print(f"⚠️ Unable to create directory {self.base_dir}: {e}")

        # Initialize enhanced reader
        self.file_reader = EnhancedSimulationFileReader(debug=False)
        
        self.supported_formats = {
            '.pkl': self._read_pkl_file,
            '.pt': self._read_pt_file,
            '.h5': self._read_h5_file,
            '.hdf5': self._read_h5_file,
            '.npz': self._read_npz_file,
            '.json': self._read_json_file,
            '.npy': self._read_npy_file
        }

        # Cache for loaded simulations
        self._cache = {}
        self._metadata_cache = {}
        
        # Debug flag
        self.debug = os.getenv('SCANNER_DEBUG', 'False').lower() in ('true', '1', 't')
    
    def _read_pkl_file(self, file_path):
        """Read pickle file using enhanced reader"""
        return self.file_reader._read_pkl_file(file_path)
    
    def _read_pt_file(self, file_path):
        """Read PyTorch file using enhanced reader"""
        return self.file_reader._read_pt_file(file_path)
    
    def _read_h5_file(self, file_path):
        """Read HDF5 file using enhanced reader"""
        return self.file_reader._read_h5_file(file_path)
    
    def _read_npz_file(self, file_path):
        """Read numpy compressed file"""
        return self.file_reader._read_npz_file(file_path)
    
    def _read_json_file(self, file_path):
        """Read JSON file"""
        return self.file_reader._read_json_file(file_path)
    
    def _read_npy_file(self, file_path):
        """Read numpy array file"""
        try:
            return np.load(file_path, allow_pickle=True)
        except Exception as e:
            if self.debug:
                print(f"Error reading NPY file {file_path}: {e}")
            raise
    
    def load_simulation(self, file_path, use_cache=True):
        """
        Load a simulation from file path
        
        Args:
            file_path: Path to simulation file
            use_cache: Whether to use cache for faster loading
            
        Returns:
            Standardized simulation data
        """
        file_path = Path(file_path)
        
        if self.debug:
            print(f"🔍 Loading file: {file_path}")
            print(f"🔍 File exists: {file_path.exists()}")
        
        # Check cache
        cache_key = str(file_path.resolve())
        if use_cache and cache_key in self._cache:
            if self.debug:
                print(f"🔍 Using cached version")
            return self._cache[cache_key]
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in self.supported_formats:
            # Try with enhanced reader directly
            try:
                data = self.file_reader.read_file(file_path)
            except:
                raise ValueError(f"Unsupported file format: {ext}")
        else:
            # Use format-specific reader
            data = self.supported_formats[ext](file_path)
        
        if self.debug:
            print(f"🔍 Raw data type: {type(data)}")
            if isinstance(data, dict):
                print(f"🔍 Raw data keys: {list(data.keys())}")
        
        # Standardize data - use enhanced reader's standardization for PKL/PT
        if ext in ['.pkl', '.pt']:
            if ext == '.pkl':
                standardized_data = self.file_reader._standardize_pkl_data(data)
            else:  # .pt
                # PT files are already converted to numpy by _read_pt_file
                standardized_data = self.file_reader._standardize_pkl_data(data)
        else:
            # For other formats, use existing standardization
            standardized_data = self._standardize_data(data, file_path)
        
        # Add file metadata
        standardized_data['file_metadata'] = {
            'path': str(file_path),
            'filename': file_path.name,
            'size_bytes': file_path.stat().st_size,
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'format': ext[1:].upper() if ext.startswith('.') else ext
        }
        
        # Add source information
        if 'metadata' not in standardized_data:
            standardized_data['metadata'] = {}
        standardized_data['metadata']['loaded_from'] = str(file_path)
        standardized_data['metadata']['load_time'] = datetime.now().isoformat()
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = standardized_data
        
        if self.debug:
            print(f"🔍 Successfully loaded and standardized")
            print(f"🔍 History frames: {len(standardized_data.get('history', []))}")
        
        return standardized_data
    
    # Keep your existing _standardize_data method for other formats
    def _standardize_data(self, raw_data, file_path):
        """
        Standardize raw data from different formats (for non-PKL/PT formats)
        """
        # ... (keep your existing _standardize_data method for H5, NPZ, JSON, etc.)
        pass

# =============================================
# STREAMLIT INTERFACE FOR FILE READING
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
                                
                                # Display summary
                                with st.expander("📋 File Summary", expanded=True):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    if isinstance(data, dict):
                                        with col1:
                                            st.metric("Keys", len(data.keys()))
                                        with col2:
                                            if 'history' in data:
                                                st.metric("Frames", len(data['history']))
                                        with col3:
                                            if 'params' in data:
                                                st.metric("Parameters", len(data['params']))
                                        
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
                                        
                            except Exception as e:
                                st.error(f"❌ Error reading file: {str(e)}")
                                st.code(traceback.format_exc())
                        
                        break
    
    elif mode == "📁 Load from Directory":
        st.subheader("Load Files from Directory")
        
        # Directory selection
        col1, col2 = st.columns([3, 1])
        with col1:
            directory_path = st.text_input(
                "Directory path",
                value=str(st.session_state.directory_scanner.base_dir),
                help="Path to directory containing simulation files"
            )
        with col2:
            if st.button("📂 Browse", help="Select directory"):
                # Note: Streamlit doesn't have native directory browser
                # This would need a custom component or file browser
                st.info("Directory browsing requires custom component")
        
        if directory_path and os.path.exists(directory_path):
            # Scan directory
            if st.button("🔍 Scan Directory"):
                with st.spinner("Scanning directory..."):
                    scanner = st.session_state.directory_scanner
                    scanner.base_dir = Path(directory_path)
                    
                    # Scan for files
                    files_by_format = scanner.scan_directory()
                    total_files = len(files_by_format['all'])
                    
                    if total_files == 0:
                        st.warning("No simulation files found in directory!")
                    else:
                        st.success(f"Found {total_files} files")
                        
                        # Show file breakdown
                        st.subheader("📊 File Breakdown by Format")
                        format_data = []
                        for ext, files in files_by_format.items():
                            if ext != 'all' and files:
                                format_data.append({
                                    'Format': ext,
                                    'Count': len(files),
                                    'Files': ', '.join([Path(f).name for f in files[:3]]) + 
                                            ('...' if len(files) > 3 else '')
                                })
                        
                        if format_data:
                            df_formats = pd.DataFrame(format_data)
                            st.dataframe(df_formats, use_container_width=True)
                        
                        # File selection for loading
                        st.subheader("📥 Load Files")
                        all_files = files_by_format['all']
                        
                        selected_files = st.multiselect(
                            "Select files to load",
                            [Path(f).name for f in all_files],
                            default=[Path(f).name for f in all_files[:min(5, len(all_files))]]
                        )
                        
                        if selected_files and st.button("🚀 Load Selected Files"):
                            # Map back to full paths
                            file_map = {Path(f).name: f for f in all_files}
                            selected_paths = [file_map[name] for name in selected_files]
                            
                            # Load files
                            progress_bar = st.progress(0)
                            loaded_simulations = []
                            
                            for i, file_path in enumerate(selected_paths):
                                st.write(f"Loading {Path(file_path).name}...")
                                try:
                                    sim_data = scanner.load_simulation(file_path)
                                    loaded_simulations.append({
                                        'file': Path(file_path).name,
                                        'data': sim_data,
                                        'success': True
                                    })
                                except Exception as e:
                                    loaded_simulations.append({
                                        'file': Path(file_path).name,
                                        'error': str(e),
                                        'success': False
                                    })
                                
                                progress_bar.progress((i + 1) / len(selected_paths))
                            
                            # Display results
                            st.subheader("📋 Loading Results")
                            
                            success_count = sum(1 for s in loaded_simulations if s['success'])
                            failed_count = len(loaded_simulations) - success_count
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Successfully Loaded", success_count)
                            with col2:
                                st.metric("Failed", failed_count)
                            
                            # Store in session state
                            if 'loaded_simulations' not in st.session_state:
                                st.session_state.loaded_simulations = []
                            st.session_state.loaded_simulations.extend(
                                [s for s in loaded_simulations if s['success']]
                            )
                            
                            # Show failed files
                            if failed_count > 0:
                                with st.expander("❌ Failed Files", expanded=False):
                                    for sim in loaded_simulations:
                                        if not sim['success']:
                                            st.write(f"- {sim['file']}: {sim['error']}")
        
        else:
            st.warning("Please enter a valid directory path")
    
    elif mode == "🔍 Analyze File Structure":
        st.subheader("Analyze File Structure")
        
        uploaded_file = st.file_uploader(
            "Upload a file for structure analysis",
            type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'json', 'npy'],
            help="Upload any simulation file to analyze its structure"
        )
        
        if uploaded_file:
            if st.button("🔬 Analyze Structure", type="primary"):
                with st.spinner("Analyzing file structure..."):
                    try:
                        # Analyze file structure
                        analysis = st.session_state.file_reader.analyze_file_structure(
                            uploaded_file.name
                        )
                        
                        # Display results
                        st.success(f"✅ Analysis complete for {uploaded_file.name}")
                        
                        # Show basic info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Size", analysis.get('file_size', 'N/A'))
                        with col2:
                            st.metric("Format", analysis.get('file_format', 'N/A'))
                        with col3:
                            if analysis.get('success', False):
                                st.metric("Status", "✅ Success")
                            else:
                                st.metric("Status", "❌ Failed")
                        
                        # Show detailed analysis
                        with st.expander("📊 Detailed Analysis", expanded=True):
                            # Remove raw data if present to avoid display issues
                            analysis_display = {k: v for k, v in analysis.items() 
                                              if k not in ['raw_data', 'data']}
                            st.json(analysis_display)
                        
                        # If it's a dictionary, show interactive exploration
                        if analysis.get('data_type') == 'dict':
                            st.subheader("🔍 Interactive Data Explorer")
                            
                            # Simulate data loading for exploration
                            try:
                                data = st.session_state.file_reader.read_file(
                                    uploaded_file.getvalue()
                                )
                                
                                # Create interactive tree viewer
                                def display_dict_tree(data, parent_key='', depth=0):
                                    if isinstance(data, dict):
                                        for key, value in data.items():
                                            with st.expander(f"{'  ' * depth}📁 {key}", 
                                                           expanded=depth < 2):
                                                if isinstance(value, dict):
                                                    display_dict_tree(value, key, depth + 1)
                                                elif isinstance(value, (list, tuple)):
                                                    st.write(f"Type: {type(value).__name__}")
                                                    st.write(f"Length: {len(value)}")
                                                    if value and isinstance(value[0], (dict, list, tuple)):
                                                        display_dict_tree(value[0], f"{key}[0]", depth + 1)
                                                elif isinstance(value, np.ndarray):
                                                    st.write(f"Type: numpy.ndarray")
                                                    st.write(f"Shape: {value.shape}")
                                                    st.write(f"Dtype: {value.dtype}")
                                                else:
                                                    st.write(f"Value: {value}")
                                    elif isinstance(data, (list, tuple)):
                                        st.write(f"Length: {len(data)}")
                                        if data and len(data) > 0:
                                            display_dict_tree(data[0], f"{parent_key}[0]", depth + 1)
                                
                                display_dict_tree(data)
                                
                            except Exception as e:
                                st.warning(f"Cannot explore data: {e}")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
    
    elif mode == "📊 Batch Processing":
        st.subheader("Batch Process Directory")
        
        directory_path = st.text_input(
            "Directory to process",
            value=str(st.session_state.directory_scanner.base_dir),
            help="Path to directory for batch processing"
        )
        
        if directory_path and os.path.exists(directory_path):
            col1, col2 = st.columns(2)
            
            with col1:
                max_files = st.number_input(
                    "Maximum files to process",
                    min_value=1, 
                    max_value=1000, 
                    value=50
                )
            
            with col2:
                pattern = st.text_input(
                    "File pattern",
                    value="*",
                    help="Glob pattern for file selection"
                )
            
            if st.button("🚀 Run Batch Processing", type="primary"):
                with st.spinner(f"Processing files in {directory_path}..."):
                    try:
                        # Run batch processing
                        results = st.session_state.file_reader.batch_load_directory(
                            directory_path,
                            pattern=pattern,
                            max_files=max_files
                        )
                        
                        # Display results
                        st.success(f"✅ Batch processing complete!")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Attempted", results['total_attempted'])
                        with col2:
                            st.metric("Successful", len(results['successful']))
                        with col3:
                            st.metric("Failed", len(results['failed']))
                        
                        # Show successful files
                        with st.expander("✅ Successful Files", expanded=True):
                            if results['successful']:
                                # Create summary table
                                summary_data = []
                                for file_info in results['successful']:
                                    data = file_info['data']
                                    if isinstance(data, dict):
                                        frames = len(data.get('history', []))
                                        params = len(data.get('params', {}))
                                    else:
                                        frames = 0
                                        params = 0
                                    
                                    summary_data.append({
                                        'File': file_info['file_name'],
                                        'Frames': frames,
                                        'Parameters': params,
                                        'Type': type(data).__name__
                                    })
                                
                                if summary_data:
                                    df_summary = pd.DataFrame(summary_data)
                                    st.dataframe(df_summary, use_container_width=True)
                            else:
                                st.info("No files were successfully loaded")
                        
                        # Show failed files
                        if results['failed']:
                            with st.expander("❌ Failed Files", expanded=False):
                                for file_info in results['failed']:
                                    st.write(f"**{Path(file_info['file_path']).name}**: {file_info['error']}")
                        
                        # Generate and display report
                        if st.button("📋 Generate Detailed Report"):
                            report = st.session_state.file_reader.export_summary_report(
                                directory_path,
                                output_file=None  # Don't save to file
                            )
                            
                            if report:
                                st.subheader("📊 Directory Analysis Report")
                                
                                # Format summary
                                st.write("### File Format Distribution")
                                format_data = []
                                for fmt, count in report.get('format_summary', {}).items():
                                    format_data.append({
                                        'Format': fmt,
                                        'Count': count,
                                        'Percentage': f"{(count/report['total_files'])*100:.1f}%"
                                    })
                                
                                if format_data:
                                    df_formats = pd.DataFrame(format_data)
                                    st.dataframe(df_formats, use_container_width=True)
                                
                                # Success rate
                                success_count = sum(1 for a in report['file_analysis'] 
                                                  if a.get('success', False))
                                success_rate = (success_count / report['total_files']) * 100
                                
                                st.metric("Overall Success Rate", f"{success_rate:.1f}%")
                    
                    except Exception as e:
                        st.error(f"❌ Batch processing failed: {str(e)}")
        
        else:
            st.warning("Please enter a valid directory path")
    
    # Add debugging information
    with st.expander("🐛 Debug Information", expanded=False):
        st.subheader("File Reader Status")
        
        if 'loaded_simulations' in st.session_state:
            st.write(f"Loaded simulations: {len(st.session_state.loaded_simulations)}")
        
        st.write("### Reader Configuration")
        st.write(f"Debug mode: {st.session_state.file_reader.debug}")
        
        if st.button("Clear Cache"):
            if 'loaded_simulations' in st.session_state:
                del st.session_state.loaded_simulations
            st.success("Cache cleared!")
            st.rerun()

# =============================================
# INTEGRATE INTO MAIN APPLICATION
# =============================================
def main_enhanced():
    """Main application with enhanced file reading"""
    
    # Sidebar for mode selection
    st.sidebar.header("🔧 Operation Mode")
    
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["Run New Simulation", 
         "Compare Saved Simulations", 
         "Single Simulation View", 
         "Attention Interpolation",
         "📂 File Reader"],
        index=4  # Default to File Reader
    )
    
    if operation_mode == "📂 File Reader":
        create_file_reader_interface()
    else:
        # Original modes (unchanged)
        if operation_mode == "Attention Interpolation":
            create_attention_interface()
        else:
            st.warning("Original simulation modes not shown. Use '📂 File Reader' mode for file operations.")

# Update the main call
if __name__ == "__main__":
    main_enhanced()
