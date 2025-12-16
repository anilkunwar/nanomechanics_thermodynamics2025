import streamlit as st
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from io import BytesIO
import time
import hashlib
import json
from datetime import datetime
import warnings
import pickle
import torch
import sqlite3
from pathlib import Path
import tempfile
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Union
from itertools import product
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import requests
from github import Github, GithubIntegration, InputGitTreeElement
import h5py
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import uuid

warnings.filterwarnings('ignore')

# =============================================
# LOGGING CONFIGURATION
# =============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================
# GITHUB CLOUD CONFIGURATION
# =============================================
class GitHubConfig:
    """Configuration for GitHub cloud storage"""
    
    # Repository configuration
    REPO_OWNER = "your-username"  # Change to your GitHub username
    REPO_NAME = "ml-simulation-results"  # Repository for storing ML results
    BRANCH = "main"
    
    # Paths within repository
    ML_RESULTS_PATH = "ml_results"
    PREDICTIONS_PATH = f"{ML_RESULTS_PATH}/predictions"
    DATASETS_PATH = f"{ML_RESULTS_PATH}/datasets"
    MODELS_PATH = f"{ML_RESULTS_PATH}/models"
    
    # File size limits (GitHub has 100MB limit for individual files)
    MAX_FILE_SIZE = 95 * 1024 * 1024  # 95MB to be safe
    
    @staticmethod
    def get_repo_url():
        return f"https://github.com/{GitHubConfig.REPO_OWNER}/{GitHubConfig.REPO_NAME}"
    
    @staticmethod
    def get_api_url():
        return f"https://api.github.com/repos/{GitHubConfig.REPO_OWNER}/{GitHubConfig.REPO_NAME}"

# =============================================
# GITHUB CLOUD MANAGER
# =============================================
class GitHubCloudManager:
    """Manager for storing and retrieving ML results from GitHub cloud"""
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub cloud manager
        
        Args:
            github_token: GitHub personal access token (or use Streamlit secrets)
        """
        self.github_token = github_token or self._get_github_token()
        self.g = Github(self.github_token) if self.github_token else None
        self.repo = None
        self._initialize_repository()
    
    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from Streamlit secrets or environment"""
        try:
            # Try Streamlit secrets first
            return st.secrets["github"]["token"]
        except:
            # Fall back to environment variable
            return os.getenv("GITHUB_TOKEN")
    
    def _initialize_repository(self):
        """Initialize or verify GitHub repository exists"""
        try:
            if self.g:
                self.repo = self.g.get_repo(f"{GitHubConfig.REPO_OWNER}/{GitHubConfig.REPO_NAME}")
                logger.info(f"Connected to GitHub repository: {GitHubConfig.get_repo_url()}")
            else:
                logger.warning("GitHub token not provided. Cloud features will be limited.")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub repository: {str(e)}")
            st.warning("⚠️ GitHub repository not accessible. Some cloud features may be unavailable.")
    
    def upload_file_to_cloud(self, 
                           file_content: Union[bytes, str],
                           file_path: str,
                           commit_message: str = "Upload ML results",
                           branch: str = GitHubConfig.BRANCH) -> Dict[str, Any]:
        """
        Upload a file to GitHub cloud
        
        Args:
            file_content: File content as bytes or string
            file_path: Path in repository (e.g., "ml_results/predictions/result.pkl")
            commit_message: Commit message
            branch: Branch to commit to
            
        Returns:
            Dictionary with upload status and metadata
        """
        if not self.repo:
            return {"success": False, "error": "GitHub repository not initialized"}
        
        try:
            # Convert string to bytes if needed
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            
            # Check file size
            file_size = len(file_content)
            if file_size > GitHubConfig.MAX_FILE_SIZE:
                return {
                    "success": False,
                    "error": f"File too large ({file_size / 1024 / 1024:.2f}MB). GitHub limit is {GitHubConfig.MAX_FILE_SIZE / 1024 / 1024:.2f}MB"
                }
            
            # Encode content in base64 for GitHub API
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            # Check if file already exists
            try:
                existing_file = self.repo.get_contents(file_path, ref=branch)
                # Update existing file
                result = self.repo.update_file(
                    path=file_path,
                    message=commit_message,
                    content=encoded_content,
                    sha=existing_file.sha,
                    branch=branch
                )
                action = "updated"
            except:
                # Create new file
                result = self.repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=encoded_content,
                    branch=branch
                )
                action = "created"
            
            # Get download URL
            download_url = result["content"].download_url
            
            return {
                "success": True,
                "action": action,
                "commit": result["commit"].sha,
                "download_url": download_url,
                "file_path": file_path,
                "size": file_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error uploading to GitHub: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def upload_multiple_files(self, 
                            files: List[Dict[str, Union[bytes, str]]],
                            base_path: str = GitHubConfig.ML_RESULTS_PATH,
                            commit_message: str = "Upload multiple ML results") -> Dict[str, Any]:
        """
        Upload multiple files in a single commit
        
        Args:
            files: List of dicts with 'path' and 'content' keys
            base_path: Base path in repository
            commit_message: Commit message
            
        Returns:
            Upload results
        """
        if not self.repo:
            return {"success": False, "error": "GitHub repository not initialized"}
        
        try:
            # Get current commit
            ref = self.repo.get_git_ref(f"heads/{GitHubConfig.BRANCH}")
            commit = self.repo.get_git_commit(ref.object.sha)
            
            # Create tree elements
            tree_elements = []
            
            for file_info in files:
                file_path = os.path.join(base_path, file_info["path"])
                content = file_info["content"]
                
                if isinstance(content, str):
                    content = content.encode('utf-8')
                
                encoded_content = base64.b64encode(content).decode('utf-8')
                
                # Check if file exists
                try:
                    existing_file = self.repo.get_contents(file_path, ref=GitHubConfig.BRANCH)
                    mode = "100644"  # Regular file
                    tree_elements.append(InputGitTreeElement(
                        path=file_path,
                        mode=mode,
                        type="blob",
                        sha=existing_file.sha
                    ))
                except:
                    # File doesn't exist, will be created
                    pass
            
            # Create new tree
            tree = self.repo.create_git_tree(tree_elements, base_tree=commit.tree)
            
            # Create new commit
            new_commit = self.repo.create_git_commit(commit_message, tree, [commit])
            
            # Update branch reference
            ref.edit(new_commit.sha)
            
            return {
                "success": True,
                "commit": new_commit.sha,
                "files_uploaded": len(files),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error uploading multiple files: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def download_file_from_cloud(self, 
                               file_path: str,
                               branch: str = GitHubConfig.BRANCH) -> Optional[bytes]:
        """
        Download file from GitHub cloud
        
        Args:
            file_path: Path in repository
            branch: Branch to download from
            
        Returns:
            File content as bytes or None if failed
        """
        if not self.repo:
            return None
        
        try:
            file_content = self.repo.get_contents(file_path, ref=branch)
            decoded_content = base64.b64decode(file_content.content)
            return decoded_content
        except Exception as e:
            logger.error(f"Error downloading from GitHub: {str(e)}")
            return None
    
    def list_files_in_directory(self, 
                              directory_path: str = GitHubConfig.ML_RESULTS_PATH,
                              branch: str = GitHubConfig.BRANCH) -> List[Dict[str, Any]]:
        """
        List files in a GitHub directory
        
        Args:
            directory_path: Directory path in repository
            branch: Branch to list from
            
        Returns:
            List of file information dictionaries
        """
        if not self.repo:
            return []
        
        try:
            contents = self.repo.get_contents(directory_path, ref=branch)
            files = []
            
            for item in contents:
                files.append({
                    "name": item.name,
                    "path": item.path,
                    "type": item.type,  # "file" or "dir"
                    "size": item.size,
                    "sha": item.sha,
                    "download_url": item.download_url,
                    "last_modified": item.last_modified
                })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing directory: {str(e)}")
            return []
    
    def delete_file_from_cloud(self,
                             file_path: str,
                             commit_message: str = "Delete file",
                             branch: str = GitHubConfig.BRANCH) -> bool:
        """
        Delete file from GitHub cloud
        
        Args:
            file_path: Path in repository
            commit_message: Commit message
            branch: Branch to delete from
            
        Returns:
            Success status
        """
        if not self.repo:
            return False
        
        try:
            file = self.repo.get_contents(file_path, ref=branch)
            self.repo.delete_file(
                path=file_path,
                message=commit_message,
                sha=file.sha,
                branch=branch
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str, branch: str = GitHubConfig.BRANCH) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a file
        
        Args:
            file_path: Path in repository
            branch: Branch to check
            
        Returns:
            File information dictionary
        """
        if not self.repo:
            return None
        
        try:
            file = self.repo.get_contents(file_path, ref=branch)
            commits = list(self.repo.get_commits(path=file_path))
            
            return {
                "name": file.name,
                "path": file.path,
                "size": file.size,
                "sha": file.sha,
                "download_url": file.download_url,
                "html_url": file.html_url,
                "last_modified": file.last_modified,
                "commit_count": len(commits),
                "last_commit": commits[0].commit.message if commits else None
            }
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return None
    
    def create_release(self,
                      tag_name: str,
                      release_name: str,
                      body: str,
                      files: List[Tuple[str, bytes]] = None,
                      draft: bool = False,
                      prerelease: bool = False) -> Dict[str, Any]:
        """
        Create a GitHub release with attached files
        
        Args:
            tag_name: Git tag for release
            release_name: Release name
            body: Release description
            files: List of (filename, content) tuples
            draft: Whether release is a draft
            prerelease: Whether release is a prerelease
            
        Returns:
            Release information
        """
        if not self.repo:
            return {"success": False, "error": "GitHub repository not initialized"}
        
        try:
            # Create release
            release = self.repo.create_git_release(
                tag=tag_name,
                name=release_name,
                message=body,
                draft=draft,
                prerelease=prerelease
            )
            
            # Upload assets if provided
            assets = []
            if files:
                for filename, content in files:
                    asset = release.upload_asset(
                        name=filename,
                        label=filename,
                        content_type="application/octet-stream",
                        data=content
                    )
                    assets.append({
                        "name": filename,
                        "download_url": asset.browser_download_url
                    })
            
            return {
                "success": True,
                "release_id": release.id,
                "tag_name": release.tag_name,
                "name": release.name,
                "html_url": release.html_url,
                "assets": assets,
                "created_at": release.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating release: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get information about the repository
        
        Returns:
            Repository information
        """
        if not self.repo:
            return {"error": "Repository not initialized"}
        
        try:
            return {
                "name": self.repo.name,
                "full_name": self.repo.full_name,
                "description": self.repo.description,
                "html_url": self.repo.html_url,
                "stargazers_count": self.repo.stargazers_count,
                "forks_count": self.repo.forks_count,
                "size": self.repo.size,
                "updated_at": self.repo.updated_at.isoformat() if self.repo.updated_at else None,
                "default_branch": self.repo.default_branch
            }
        except Exception as e:
            logger.error(f"Error getting repository info: {str(e)}")
            return {"error": str(e)}
    
    def create_download_link(self, file_path: str) -> str:
        """
        Create a direct download link for a file
        
        Args:
            file_path: Path in repository
            
        Returns:
            Direct download URL
        """
        if not self.repo:
            return ""
        
        try:
            file = self.repo.get_contents(file_path, ref=GitHubConfig.BRANCH)
            return file.download_url
        except:
            # Fallback to raw URL
            return f"https://raw.githubusercontent.com/{GitHubConfig.REPO_OWNER}/{GitHubConfig.REPO_NAME}/{GitHubConfig.BRANCH}/{file_path}"

# =============================================
# ENHANCED PREDICTION RESULTS MANAGER WITH GITHUB
# =============================================
class CloudPredictionResultsManager(PredictionResultsManager):
    """Enhanced manager with GitHub cloud integration"""
    
    @staticmethod
    def save_prediction_to_cloud(prediction_data: Dict[str, Any],
                               filename: str,
                               github_manager: GitHubCloudManager,
                               format_type: str = 'pkl',
                               metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Save prediction to GitHub cloud
        
        Args:
            prediction_data: Prediction data to save
            filename: Base filename (without extension)
            github_manager: GitHubCloudManager instance
            format_type: File format ('pkl', 'pt', 'json', 'zip')
            metadata: Additional metadata
            
        Returns:
            Dictionary with save results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            cloud_filename = f"{filename}_{timestamp}_{unique_id}"
            
            # Prepare file content based on format
            file_content = None
            file_extension = format_type
            
            if format_type == 'pkl':
                file_content = pickle.dumps(prediction_data, protocol=pickle.HIGHEST_PROTOCOL)
                cloud_filename = f"{cloud_filename}.pkl"
                
            elif format_type == 'pt':
                buffer = BytesIO()
                torch.save(prediction_data, buffer)
                file_content = buffer.getvalue()
                cloud_filename = f"{cloud_filename}.pt"
                
            elif format_type == 'json':
                def convert_for_json(obj):
                    if isinstance(obj, (np.float32, np.float64, np.float16)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    else:
                        return obj
                
                json_data = json.dumps(prediction_data, default=convert_for_json, indent=2)
                file_content = json_data.encode('utf-8')
                cloud_filename = f"{cloud_filename}.json"
                
            elif format_type == 'zip':
                # Create a comprehensive archive
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add prediction data as JSON
                    json_data = json.dumps(prediction_data, default=str, indent=2)
                    zip_file.writestr('prediction_data.json', json_data)
                    
                    # Add stress fields as separate files if they exist
                    if 'stress_fields' in prediction_data:
                        for field_name, field_data in prediction_data['stress_fields'].items():
                            if isinstance(field_data, np.ndarray):
                                npz_buffer = BytesIO()
                                np.savez_compressed(npz_buffer, data=field_data)
                                npz_buffer.seek(0)
                                zip_file.writestr(f'stress_{field_name}.npz', npz_buffer.read())
                
                file_content = zip_buffer.getvalue()
                cloud_filename = f"{cloud_filename}.zip"
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format_type}"
                }
            
            # Upload to GitHub
            cloud_path = f"{GitHubConfig.PREDICTIONS_PATH}/{cloud_filename}"
            commit_message = f"Save prediction: {filename}"
            
            if metadata:
                commit_message += f" - {json.dumps(metadata)}"
            
            upload_result = github_manager.upload_file_to_cloud(
                file_content=file_content,
                file_path=cloud_path,
                commit_message=commit_message
            )
            
            if upload_result["success"]:
                upload_result.update({
                    "cloud_filename": cloud_filename,
                    "cloud_path": cloud_path,
                    "format": format_type,
                    "download_url": github_manager.create_download_link(cloud_path)
                })
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error saving to cloud: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def save_prediction_archive_to_cloud(prediction_results: Dict[str, Any],
                                       source_simulations: List[Dict],
                                       github_manager: GitHubCloudManager,
                                       archive_name: str = "prediction_archive") -> Dict[str, Any]:
        """
        Save comprehensive prediction archive to GitHub cloud
        
        Args:
            prediction_results: Prediction results dictionary
            source_simulations: List of source simulations
            github_manager: GitHubCloudManager instance
            archive_name: Base name for archive
            
        Returns:
            Dictionary with save results
        """
        try:
            # Create the archive
            if prediction_results.get('mode') == 'multi':
                archive_buffer = PredictionResultsManager.create_multi_prediction_archive(
                    prediction_results, source_simulations
                )
            else:
                archive_buffer = PredictionResultsManager.create_single_prediction_archive(
                    prediction_results, source_simulations
                )
            
            # Prepare metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            cloud_filename = f"{archive_name}_{timestamp}_{unique_id}.zip"
            
            # Upload to GitHub
            cloud_path = f"{GitHubConfig.PREDICTIONS_PATH}/archives/{cloud_filename}"
            
            upload_result = github_manager.upload_file_to_cloud(
                file_content=archive_buffer.getvalue(),
                file_path=cloud_path,
                commit_message=f"Save prediction archive: {archive_name}"
            )
            
            if upload_result["success"]:
                upload_result.update({
                    "cloud_filename": cloud_filename,
                    "cloud_path": cloud_path,
                    "download_url": github_manager.create_download_link(cloud_path)
                })
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error saving archive to cloud: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def load_prediction_from_cloud(filename: str,
                                 github_manager: GitHubCloudManager) -> Optional[Dict[str, Any]]:
        """
        Load prediction from GitHub cloud
        
        Args:
            filename: Filename in cloud (with extension)
            github_manager: GitHubCloudManager instance
            
        Returns:
            Loaded prediction data or None
        """
        try:
            cloud_path = f"{GitHubConfig.PREDICTIONS_PATH}/{filename}"
            file_content = github_manager.download_file_from_cloud(cloud_path)
            
            if not file_content:
                return None
            
            # Determine format from extension
            if filename.endswith('.pkl'):
                buffer = BytesIO(file_content)
                return pickle.load(buffer)
            elif filename.endswith('.pt'):
                buffer = BytesIO(file_content)
                return torch.load(buffer, map_location=torch.device('cpu'))
            elif filename.endswith('.json'):
                return json.loads(file_content.decode('utf-8'))
            elif filename.endswith('.zip'):
                # Handle zip archive
                zip_buffer = BytesIO(file_content)
                with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                    # Read main prediction file
                    if 'prediction_data.json' in zip_file.namelist():
                        json_data = zip_file.read('prediction_data.json')
                        return json.loads(json_data.decode('utf-8'))
            else:
                logger.warning(f"Unknown file format: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading from cloud: {str(e)}")
            return None
    
    @staticmethod
    def list_cloud_predictions(github_manager: GitHubCloudManager,
                             filter_format: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List predictions stored in GitHub cloud
        
        Args:
            github_manager: GitHubCloudManager instance
            filter_format: Optional format filter ('pkl', 'pt', 'json', 'zip')
            
        Returns:
            List of prediction file information
        """
        try:
            files = github_manager.list_files_in_directory(GitHubConfig.PREDICTIONS_PATH)
            
            if filter_format:
                files = [f for f in files if f["name"].endswith(f".{filter_format}")]
            
            # Sort by last modified (newest first)
            files.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing cloud predictions: {str(e)}")
            return []

# =============================================
# CLOUD DATASET MANAGER
# =============================================
class CloudDatasetManager:
    """Manager for storing datasets in GitHub cloud"""
    
    @staticmethod
    def save_dataset_to_cloud(dataset: Union[pd.DataFrame, np.ndarray, Dict],
                            dataset_name: str,
                            github_manager: GitHubCloudManager,
                            description: str = "",
                            tags: List[str] = None) -> Dict[str, Any]:
        """
        Save dataset to GitHub cloud
        
        Args:
            dataset: Dataset to save
            dataset_name: Name for the dataset
            github_manager: GitHubCloudManager instance
            description: Dataset description
            tags: Optional tags for categorization
            
        Returns:
            Save results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            cloud_filename = f"{dataset_name}_{timestamp}_{unique_id}"
            
            # Convert dataset to bytes based on type
            file_content = None
            metadata = {
                "dataset_name": dataset_name,
                "description": description,
                "tags": tags or [],
                "created_at": timestamp,
                "data_type": type(dataset).__name__
            }
            
            if isinstance(dataset, pd.DataFrame):
                # Save as CSV and Parquet
                csv_buffer = BytesIO()
                dataset.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                
                parquet_buffer = BytesIO()
                dataset.to_parquet(parquet_buffer, index=False)
                parquet_content = parquet_buffer.getvalue()
                
                # Upload both formats
                results = []
                
                # CSV
                csv_path = f"{GitHubConfig.DATASETS_PATH}/{cloud_filename}.csv"
                csv_result = github_manager.upload_file_to_cloud(
                    file_content=csv_content,
                    file_path=csv_path,
                    commit_message=f"Save dataset: {dataset_name} (CSV)"
                )
                if csv_result["success"]:
                    csv_result["format"] = "csv"
                    results.append(csv_result)
                
                # Parquet
                parquet_path = f"{GitHubConfig.DATASETS_PATH}/{cloud_filename}.parquet"
                parquet_result = github_manager.upload_file_to_cloud(
                    file_content=parquet_content,
                    file_path=parquet_path,
                    commit_message=f"Save dataset: {dataset_name} (Parquet)"
                )
                if parquet_result["success"]:
                    parquet_result["format"] = "parquet"
                    results.append(parquet_result)
                
                # Save metadata
                metadata_path = f"{GitHubConfig.DATASETS_PATH}/{cloud_filename}_metadata.json"
                metadata_result = github_manager.upload_file_to_cloud(
                    file_content=json.dumps(metadata, indent=2).encode('utf-8'),
                    file_path=metadata_path,
                    commit_message=f"Save dataset metadata: {dataset_name}"
                )
                
                return {
                    "success": any(r["success"] for r in results),
                    "formats_saved": [r["format"] for r in results if r["success"]],
                    "results": results,
                    "metadata": metadata
                }
                
            elif isinstance(dataset, np.ndarray):
                # Save as NPZ
                npz_buffer = BytesIO()
                np.savez_compressed(npz_buffer, data=dataset)
                file_content = npz_buffer.getvalue()
                cloud_filename = f"{cloud_filename}.npz"
                
            elif isinstance(dataset, dict):
                # Save as JSON
                file_content = json.dumps(dataset, indent=2, default=str).encode('utf-8')
                cloud_filename = f"{cloud_filename}.json"
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported dataset type: {type(dataset)}"
                }
            
            # Upload single file
            cloud_path = f"{GitHubConfig.DATASETS_PATH}/{cloud_filename}"
            
            upload_result = github_manager.upload_file_to_cloud(
                file_content=file_content,
                file_path=cloud_path,
                commit_message=f"Save dataset: {dataset_name}"
            )
            
            if upload_result["success"]:
                upload_result.update({
                    "metadata": metadata,
                    "format": cloud_filename.split('.')[-1]
                })
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error saving dataset to cloud: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def create_dataset_release(datasets: Dict[str, Any],
                             release_name: str,
                             github_manager: GitHubCloudManager,
                             description: str = "") -> Dict[str, Any]:
        """
        Create a GitHub release for datasets
        
        Args:
            datasets: Dictionary of dataset name to data
            release_name: Name for the release
            github_manager: GitHubCloudManager instance
            description: Release description
            
        Returns:
            Release results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag_name = f"dataset-release-{timestamp}"
            
            # Prepare files for release
            release_files = []
            dataset_metadata = {}
            
            for dataset_name, data in datasets.items():
                if isinstance(data, pd.DataFrame):
                    # Save as CSV
                    csv_buffer = BytesIO()
                    data.to_csv(csv_buffer, index=False)
                    release_files.append((f"{dataset_name}.csv", csv_buffer.getvalue()))
                    dataset_metadata[dataset_name] = {
                        "rows": len(data),
                        "columns": list(data.columns),
                        "type": "dataframe"
                    }
            
            # Create release
            release_body = f"{description}\n\nDatasets included:\n"
            for name in datasets.keys():
                release_body += f"- {name}\n"
            
            release_result = github_manager.create_release(
                tag_name=tag_name,
                release_name=release_name,
                body=release_body,
                files=release_files
            )
            
            if release_result["success"]:
                release_result["dataset_metadata"] = dataset_metadata
            
            return release_result
            
        except Exception as e:
            logger.error(f"Error creating dataset release: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# =============================================
# ENHANCED CREATE ATTENTION INTERFACE WITH GITHUB
# =============================================
def create_attention_interface_with_github():
    """Create the attention interpolation interface with GitHub cloud integration"""
    
    st.header("🤖 Spatial-Attention Stress Interpolation (GitHub Cloud)")
    
    # Initialize GitHub Cloud Manager
    if 'github_manager' not in st.session_state:
        with st.spinner("Initializing GitHub cloud connection..."):
            try:
                st.session_state.github_manager = GitHubCloudManager()
                st.session_state.cloud_prediction_manager = CloudPredictionResultsManager()
                st.session_state.cloud_dataset_manager = CloudDatasetManager()
            except Exception as e:
                st.error(f"Failed to initialize GitHub connection: {str(e)}")
                st.session_state.github_manager = None
    
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
    
    # Initialize numerical solutions manager (for local files only)
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager()
    
    # Initialize multi-target manager
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
    
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    
    # Initialize multi-target predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
    
    # Initialize cloud predictions list
    if 'cloud_predictions' not in st.session_state:
        st.session_state.cloud_predictions = []
    
    # Get grid extent for visualization
    extent = get_grid_extent()
    
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
    
    with st.sidebar.expander("☁️ GitHub Cloud Settings", expanded=True):
        if st.session_state.github_manager and st.session_state.github_manager.repo:
            repo_info = st.session_state.github_manager.get_repository_info()
            
            st.success("✅ Connected to GitHub Cloud")
            st.markdown(f"""
            **Repository:** [{repo_info.get('full_name', 'Unknown')}]({repo_info.get('html_url', '#')})
            **Branch:** {repo_info.get('default_branch', 'main')}
            **Stars:** {repo_info.get('stargazers_count', 0)} ⭐
            """)
            
            if st.button("🔄 Refresh Cloud Connection"):
                with st.spinner("Refreshing..."):
                    st.session_state.github_manager = GitHubCloudManager()
                    st.rerun()
            
            # List cloud predictions
            if st.button("📥 Load Cloud Predictions"):
                with st.spinner("Loading predictions from cloud..."):
                    predictions = st.session_state.cloud_prediction_manager.list_cloud_predictions(
                        st.session_state.github_manager
                    )
                    st.session_state.cloud_predictions = predictions
                    st.success(f"Loaded {len(predictions)} predictions from cloud")
                    st.rerun()
        else:
            st.warning("⚠️ GitHub connection not configured")
            st.markdown("""
            To use cloud features, configure GitHub access:
            1. Create a GitHub Personal Access Token with `repo` scope
            2. Set it as environment variable `GITHUB_TOKEN`
            3. Or add to Streamlit secrets in `.streamlit/secrets.toml`:
            ```
            [github]
            token = "your_token_here"
            ```
            """)
    
    # Main interface tabs - ADDED GITHUB CLOUD TAB
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 Load Source Data",
        "🎯 Configure Target",
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict",
        "📊 Results & Visualization",
        "☁️ Save to GitHub Cloud",
        "📥 Load from GitHub Cloud"
    ])
    
    # Tab 1: Load Source Data (UNCHANGED - keep existing code)
    with tab1:
        st.subheader("Load Source Simulation Files")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📂 From numerical_solutions Directory")
            st.info(f"Loading from: `{NUMERICAL_SOLUTIONS_DIR}`")
            
            file_formats = st.session_state.solutions_manager.scan_directory()
            all_files_info = st.session_state.solutions_manager.get_all_files()
            
            if not all_files_info:
                st.warning(f"No simulation files found in `{NUMERICAL_SOLUTIONS_DIR}`")
            else:
                file_groups = {}
                for file_info in all_files_info:
                    format_type = file_info['format']
                    if format_type not in file_groups:
                        file_groups[format_type] = []
                    file_groups[format_type].append(file_info)
                
                for format_type, files in file_groups.items():
                    with st.expander(f"{format_type.upper()} Files ({len(files)})", expanded=True):
                        file_options = {}
                        for file_info in files:
                            display_name = f"{file_info['filename']} ({file_info['size'] // 1024}KB)"
                            file_options[display_name] = file_info['path']
                        
                        selected_files = st.multiselect(
                            f"Select {format_type} files",
                            options=list(file_options.keys()),
                            key=f"select_{format_type}"
                        )
                        
                        if selected_files:
                            if st.button(f"📥 Load Selected {format_type} Files", key=f"load_{format_type}"):
                                with st.spinner(f"Loading {len(selected_files)} files..."):
                                    loaded_count = 0
                                    for display_name in selected_files:
                                        file_path = file_options[display_name]
                                        try:
                                            sim_data = st.session_state.solutions_manager.load_simulation(
                                                file_path,
                                                st.session_state.interpolator
                                            )
                                            
                                            if file_path not in st.session_state.loaded_from_numerical:
                                                st.session_state.source_simulations.append(sim_data)
                                                st.session_state.loaded_from_numerical.append(file_path)
                                                loaded_count += 1
                                                st.success(f"✅ Loaded: {os.path.basename(file_path)}")
                                            else:
                                                st.warning(f"⚠️ Already loaded: {os.path.basename(file_path)}")
                                                
                                        except Exception as e:
                                            st.error(f"❌ Error loading {os.path.basename(file_path)}: {str(e)}")
                                    
                                    if loaded_count > 0:
                                        st.success(f"Successfully loaded {loaded_count} new files!")
                                        st.rerun()
        
        with col2:
            st.markdown("### 📤 Upload Local Files")
            
            uploaded_files = st.file_uploader(
                "Upload simulation files (PKL, PT, H5, NPZ, SQL, JSON)",
                type=['pkl', 'pt', 'h5', 'hdf5', 'npz', 'sql', 'db', 'json'],
                accept_multiple_files=True
            )
            
            format_type = st.selectbox(
                "File Format (for upload)",
                ["Auto Detect", "PKL", "PT", "H5", "NPZ", "SQL", "JSON"],
                index=0
            )
            
            if uploaded_files and st.button("📥 Load Uploaded Files", type="primary"):
                with st.spinner("Loading uploaded files..."):
                    loaded_sims = []
                    for uploaded_file in uploaded_files:
                        try:
                            file_content = uploaded_file.getvalue()
                            actual_format = format_type.lower() if format_type != "Auto Detect" else "auto"
                            if actual_format == "auto":
                                filename = uploaded_file.name.lower()
                                if filename.endswith('.pkl'):
                                    actual_format = 'pkl'
                                elif filename.endswith('.pt'):
                                    actual_format = 'pt'
                                elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                                    actual_format = 'h5'
                                elif filename.endswith('.npz'):
                                    actual_format = 'npz'
                                elif filename.endswith('.sql') or filename.endswith('.db'):
                                    actual_format = 'sql'
                                elif filename.endswith('.json'):
                                    actual_format = 'json'
                            
                            sim_data = st.session_state.interpolator.read_simulation_file(
                                file_content, actual_format
                            )
                            sim_data['loaded_from'] = 'upload'
                            
                            file_id = f"{uploaded_file.name}_{hashlib.md5(file_content).hexdigest()[:8]}"
                            st.session_state.uploaded_files[file_id] = {
                                'filename': uploaded_file.name,
                                'data': sim_data,
                                'format': actual_format
                            }
                            
                            st.session_state.source_simulations.append(sim_data)
                            loaded_sims.append(uploaded_file.name)
                            
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                    
                    if loaded_sims:
                        st.success(f"Successfully loaded {len(loaded_sims)} uploaded files!")
        
        # Display loaded simulations
        if st.session_state.source_simulations:
            st.subheader("📋 Loaded Source Simulations")
            
            summary_data = []
            for i, sim_data in enumerate(st.session_state.source_simulations):
                params = sim_data.get('params', {})
                metadata = sim_data.get('metadata', {})
                source = sim_data.get('loaded_from', 'unknown')
                
                summary_data.append({
                    'ID': i+1,
                    'Source': source,
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
                
                # Clear button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Clear All Source Simulations", type="secondary"):
                        st.session_state.source_simulations = []
                        st.session_state.uploaded_files = {}
                        st.session_state.loaded_from_numerical = []
                        st.success("All source simulations cleared!")
                        st.rerun()
                with col2:
                    st.info(f"**Total loaded simulations:** {len(st.session_state.source_simulations)}")
    
    # Tab 2: Configure Target (UNCHANGED - keep existing code)
    with tab2:
        st.subheader("Configure Single Target Parameters")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                target_defect = st.selectbox(
                    "Target Defect Type",
                    ["ISF", "ESF", "Twin"],
                    index=0,
                    key="target_defect_single"
                )
                
                target_shape = st.selectbox(
                    "Target Shape",
                    ["Square", "Horizontal Fault", "Vertical Fault", "Rectangle", "Ellipse"],
                    index=0,
                    key="target_shape_single"
                )
                
                target_eps0 = st.slider(
                    "Target ε*",
                    0.3, 3.0, 1.414, 0.01,
                    key="target_eps0_single"
                )
            
            with col2:
                target_kappa = st.slider(
                    "Target κ",
                    0.1, 2.0, 0.7, 0.05,
                    key="target_kappa_single"
                )
                
                orientation_mode = st.radio(
                    "Orientation Mode",
                    ["Predefined", "Custom Angle"],
                    horizontal=True,
                    key="orientation_mode_single"
                )
                
                if orientation_mode == "Predefined":
                    target_orientation = st.selectbox(
                        "Target Orientation",
                        ["Horizontal {111} (0°)",
                         "Tilted 30° (1¯10 projection)",
                         "Tilted 60°",
                         "Vertical {111} (90°)"],
                        index=0,
                        key="target_orientation_single"
                    )
                    
                    angle_map = {
                        "Horizontal {111} (0°)": 0,
                        "Tilted 30° (1¯10 projection)": 30,
                        "Tilted 60°": 60,
                        "Vertical {111} (90°)": 90,
                    }
                    target_theta = np.deg2rad(angle_map.get(target_orientation, 0))
                    st.info(f"**Target θ:** {np.rad2deg(target_theta):.1f}°")
                    
                else:
                    target_angle = st.slider(
                        "Target Angle (degrees)",
                        0.0, 90.0, 0.0, 0.5,
                        key="target_angle_custom_single"
                    )
                    target_theta = np.deg2rad(target_angle)
                    
                    target_orientation = st.session_state.interpolator.get_orientation_from_angle(target_angle)
                    st.info(f"**Target θ:** {target_angle:.1f}°")
                    st.info(f"**Orientation:** {target_orientation}")
            
            target_params = {
                'defect_type': target_defect,
                'shape': target_shape,
                'eps0': target_eps0,
                'kappa': target_kappa,
                'orientation': target_orientation,
                'theta': target_theta
            }
            
            st.session_state.target_params = target_params
    
    # Tab 3: Configure Multiple Targets (UNCHANGED - keep existing code)
    with tab3:
        st.subheader("Configure Multiple Target Parameters")
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        else:
            # ... (keep existing Tab 3 code exactly as is) ...
            pass
    
    # Tab 4: Train & Predict (UNCHANGED - keep existing code)
    with tab4:
        st.subheader("Train Model and Predict")
        
        prediction_mode = st.radio(
            "Select Prediction Mode",
            ["Single Target", "Multiple Targets (Batch)"],
            index=0,
            key="prediction_mode"
        )
        
        if len(st.session_state.source_simulations) < 2:
            st.warning("⚠️ Please load at least 2 source simulations first")
        elif prediction_mode == "Single Target" and 'target_params' not in st.session_state:
            st.warning("⚠️ Please configure single target parameters first")
        elif prediction_mode == "Multiple Targets" and not st.session_state.multi_target_params:
            st.warning("⚠️ Please generate a parameter grid first")
        else:
            # ... (keep existing Tab 4 code exactly as is) ...
            pass
    
    # Tab 5: Results & Visualization (UNCHANGED - keep existing code)
    with tab5:
        st.subheader("Prediction Results Visualization")
        
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            # ... (keep existing Tab 5 code exactly as is) ...
            pass
    
    # Tab 6: NEW - Save to GitHub Cloud
    with tab6:
        st.subheader("☁️ Save Results to GitHub Cloud")
        
        if not st.session_state.github_manager or not st.session_state.github_manager.repo:
            st.error("⚠️ GitHub connection not available. Please configure GitHub access in the sidebar.")
            st.stop()
        
        # Check if we have predictions to save
        has_single_prediction = 'prediction_results' in st.session_state
        has_multi_predictions = ('multi_target_predictions' in st.session_state and
                                len(st.session_state.multi_target_predictions) > 0)
        
        if not has_single_prediction and not has_multi_predictions:
            st.warning("⚠️ No prediction results available to save. Please run predictions first.")
        else:
            st.success("✅ Prediction results available for cloud storage!")
            
            # Repository information
            repo_info = st.session_state.github_manager.get_repository_info()
            st.markdown(f"""
            **Repository:** [{repo_info.get('full_name')}]({repo_info.get('html_url')})
            **Branch:** {repo_info.get('default_branch', 'main')}
            **ML Results Path:** `{GitHubConfig.ML_RESULTS_PATH}`
            """)
            
            # Save options
            st.subheader("📤 Cloud Save Options")
            
            save_col1, save_col2 = st.columns(2)
            
            with save_col1:
                save_mode = st.radio(
                    "Select results to save",
                    ["Current Single Prediction", "All Multiple Predictions", "Comprehensive Archive"],
                    index=0 if has_single_prediction else 1,
                    disabled=not has_single_prediction and not has_multi_predictions
                )
            
            with save_col2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = st.text_input(
                    "Base filename",
                    value=f"prediction_{timestamp}",
                    help="Files will be saved with this base name plus appropriate extensions"
                )
            
            # Format selection
            st.subheader("📁 Select Cloud Formats")
            
            format_col1, format_col2, format_col3, format_col4 = st.columns(4)
            
            with format_col1:
                save_pkl = st.checkbox("PKL Format", value=True)
            with format_col2:
                save_pt = st.checkbox("PyTorch Format", value=True)
            with format_col3:
                save_json = st.checkbox("JSON Format", value=False)
            with format_col4:
                save_zip = st.checkbox("ZIP Archive", value=True)
            
            # Metadata
            with st.expander("📝 Add Metadata", expanded=False):
                metadata_col1, metadata_col2 = st.columns(2)
                with metadata_col1:
                    author = st.text_input("Author", value="ML Interpolator")
                    description = st.text_area("Description", value="Prediction results from spatial-attention interpolator")
                with metadata_col2:
                    tags = st.multiselect(
                        "Tags",
                        ["simulation", "stress-analysis", "crystal-defects", "attention-ml", "interpolation"],
                        default=["simulation", "attention-ml"]
                    )
                    additional_tags = st.text_input("Additional Tags (comma-separated)")
                    if additional_tags:
                        tags.extend([tag.strip() for tag in additional_tags.split(",")])
            
            # Save buttons
            st.subheader("🚀 Cloud Save Actions")
            
            save_buttons_col1, save_buttons_col2, save_buttons_col3 = st.columns(3)
            
            with save_buttons_col1:
                if st.button("💾 Save All Selected Formats", type="primary", use_container_width=True):
                    if has_single_prediction and save_mode in ["Current Single Prediction", "Comprehensive Archive"]:
                        with st.spinner("Saving to GitHub cloud..."):
                            results = []
                            metadata = {
                                "author": author,
                                "description": description,
                                "tags": tags,
                                "save_mode": save_mode,
                                "source_count": len(st.session_state.source_simulations)
                            }
                            
                            # Save in selected formats
                            if save_pkl:
                                pkl_result = st.session_state.cloud_prediction_manager.save_prediction_to_cloud(
                                    prediction_data=st.session_state.prediction_results,
                                    filename=base_filename,
                                    github_manager=st.session_state.github_manager,
                                    format_type='pkl',
                                    metadata=metadata
                                )
                                results.append(("PKL", pkl_result))
                            
                            if save_pt:
                                pt_result = st.session_state.cloud_prediction_manager.save_prediction_to_cloud(
                                    prediction_data=st.session_state.prediction_results,
                                    filename=base_filename,
                                    github_manager=st.session_state.github_manager,
                                    format_type='pt',
                                    metadata=metadata
                                )
                                results.append(("PyTorch", pt_result))
                            
                            if save_json:
                                json_result = st.session_state.cloud_prediction_manager.save_prediction_to_cloud(
                                    prediction_data=st.session_state.prediction_results,
                                    filename=base_filename,
                                    github_manager=st.session_state.github_manager,
                                    format_type='json',
                                    metadata=metadata
                                )
                                results.append(("JSON", json_result))
                            
                            if save_zip:
                                zip_result = st.session_state.cloud_prediction_manager.save_prediction_archive_to_cloud(
                                    prediction_results=st.session_state.prediction_results,
                                    source_simulations=st.session_state.source_simulations,
                                    github_manager=st.session_state.github_manager,
                                    archive_name=base_filename
                                )
                                results.append(("ZIP Archive", zip_result))
                            
                            # Display results
                            st.subheader("📊 Save Results")
                            
                            for format_name, result in results:
                                if result.get("success"):
                                    st.success(f"✅ {format_name}: Saved successfully!")
                                    st.markdown(f"""
                                    **File:** `{result.get('cloud_filename')}`  
                                    **Path:** `{result.get('cloud_path')}`  
                                    **Size:** {result.get('size', 0) / 1024:.2f} KB  
                                    **Download:** [Link]({result.get('download_url', '#')})
                                    """)
                                else:
                                    st.error(f"❌ {format_name}: Failed - {result.get('error', 'Unknown error')}")
            
            with save_buttons_col2:
                if st.button("📦 Create GitHub Release", type="secondary", use_container_width=True):
                    if has_single_prediction:
                        with st.spinner("Creating GitHub release..."):
                            # Prepare files for release
                            release_files = []
                            
                            # Add prediction data
                            if 'prediction_results' in st.session_state:
                                # Save as JSON for release
                                json_data = json.dumps(
                                    st.session_state.prediction_results,
                                    default=str,
                                    indent=2
                                ).encode('utf-8')
                                release_files.append((f"{base_filename}_prediction.json", json_data))
                            
                            # Add attention weights as CSV
                            if 'attention_weights' in st.session_state.prediction_results:
                                weights = st.session_state.prediction_results['attention_weights']
                                weight_df = pd.DataFrame({
                                    'source_id': [f'S{i+1}' for i in range(len(weights))],
                                    'weight': weights,
                                    'percent_contribution': 100 * weights / (np.sum(weights) + 1e-10)
                                })
                                csv_data = weight_df.to_csv(index=False).encode('utf-8')
                                release_files.append((f"{base_filename}_attention_weights.csv", csv_data))
                            
                            # Create release
                            release_result = st.session_state.github_manager.create_release(
                                tag_name=f"v{timestamp}",
                                release_name=f"Prediction Release - {base_filename}",
                                body=f"""Prediction results generated by Spatial-Attention Interpolator

**Author:** {author}
**Description:** {description}
**Tags:** {', '.join(tags)}
**Source Simulations:** {len(st.session_state.source_simulations)}
**Generated:** {datetime.now().isoformat()}
""",
                                files=release_files
                            )
                            
                            if release_result["success"]:
                                st.success("✅ GitHub release created successfully!")
                                st.markdown(f"""
                                **Release:** [{release_result['name']}]({release_result['html_url']})  
                                **Tag:** {release_result['tag_name']}  
                                **Assets:** {len(release_result.get('assets', []))} files
                                """)
                            else:
                                st.error(f"❌ Failed to create release: {release_result.get('error')}")
            
            with save_buttons_col3:
                if st.button("📊 Save Dataset to Cloud", type="secondary", use_container_width=True):
                    # Create dataset from predictions
                    if has_single_prediction:
                        with st.spinner("Creating and saving dataset..."):
                            # Extract stress statistics as dataset
                            stress_stats = []
                            if 'stress_fields' in st.session_state.prediction_results:
                                for field_name, field_data in st.session_state.prediction_results['stress_fields'].items():
                                    if isinstance(field_data, np.ndarray):
                                        stats = {
                                            'field': field_name,
                                            'max': float(np.max(field_data)),
                                            'min': float(np.min(field_data)),
                                            'mean': float(np.mean(field_data)),
                                            'std': float(np.std(field_data)),
                                            'percentile_95': float(np.percentile(field_data, 95)),
                                            'skewness': float(pd.Series(field_data.flatten()).skew()),
                                            'kurtosis': float(pd.Series(field_data.flatten()).kurtosis())
                                        }
                                        stats.update(st.session_state.prediction_results.get('target_params', {}))
                                        stress_stats.append(stats)
                            
                            if stress_stats:
                                df_dataset = pd.DataFrame(stress_stats)
                                
                                dataset_result = st.session_state.cloud_dataset_manager.save_dataset_to_cloud(
                                    dataset=df_dataset,
                                    dataset_name=f"stress_stats_{base_filename}",
                                    github_manager=st.session_state.github_manager,
                                    description=f"Stress statistics from prediction {base_filename}",
                                    tags=tags + ["statistics", "stress-analysis"]
                                )
                                
                                if dataset_result["success"]:
                                    st.success("✅ Dataset saved to cloud!")
                                    if "formats_saved" in dataset_result:
                                        st.info(f"Formats saved: {', '.join(dataset_result['formats_saved'])}")
                                else:
                                    st.error(f"❌ Failed to save dataset: {dataset_result.get('error')}")
                            else:
                                st.warning("No stress statistics available to save as dataset.")
    
    # Tab 7: NEW - Load from GitHub Cloud
    with tab7:
        st.subheader("📥 Load Results from GitHub Cloud")
        
        if not st.session_state.github_manager or not st.session_state.github_manager.repo:
            st.error("⚠️ GitHub connection not available. Please configure GitHub access in the sidebar.")
            st.stop()
        
        # List available predictions
        st.subheader("📁 Available Cloud Predictions")
        
        # Refresh button
        if st.button("🔄 Refresh Cloud Files"):
            with st.spinner("Loading from cloud..."):
                predictions = st.session_state.cloud_prediction_manager.list_cloud_predictions(
                    st.session_state.github_manager
                )
                st.session_state.cloud_predictions = predictions
                st.success(f"Loaded {len(predictions)} predictions from cloud")
        
        if not st.session_state.cloud_predictions:
            st.info("No predictions found in cloud. Save some predictions first or refresh the list.")
        else:
            # Filter options
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                filter_format = st.selectbox(
                    "Filter by format",
                    ["All", "PKL", "PyTorch", "JSON", "ZIP"],
                    index=0
                )
            
            with filter_col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Newest", "Oldest", "Name (A-Z)", "Name (Z-A)", "Size"],
                    index=0
                )
            
            with filter_col3:
                search_term = st.text_input("Search by name", "")
            
            # Filter predictions
            filtered_predictions = st.session_state.cloud_predictions.copy()
            
            if filter_format != "All":
                filtered_predictions = [p for p in filtered_predictions 
                                      if p["name"].endswith(f".{filter_format.lower()}")]
            
            if search_term:
                filtered_predictions = [p for p in filtered_predictions 
                                      if search_term.lower() in p["name"].lower()]
            
            # Sort predictions
            if sort_by == "Newest":
                filtered_predictions.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
            elif sort_by == "Oldest":
                filtered_predictions.sort(key=lambda x: x.get("last_modified", ""))
            elif sort_by == "Name (A-Z)":
                filtered_predictions.sort(key=lambda x: x["name"].lower())
            elif sort_by == "Name (Z-A)":
                filtered_predictions.sort(key=lambda x: x["name"].lower(), reverse=True)
            elif sort_by == "Size":
                filtered_predictions.sort(key=lambda x: x.get("size", 0), reverse=True)
            
            # Display predictions
            st.markdown(f"**Found {len(filtered_predictions)} predictions**")
            
            for pred in filtered_predictions[:20]:  # Show first 20
                with st.expander(f"📄 {pred['name']} ({pred.get('size', 0) // 1024}KB)", expanded=False):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **Path:** `{pred['path']}`  
                        **Type:** {pred['type']}  
                        **Last Modified:** {pred.get('last_modified', 'Unknown')}  
                        **SHA:** `{pred['sha'][:8]}...`
                        """)
                    
                    with col2:
                        # Download button
                        download_url = pred.get('download_url', '')
                        if download_url:
                            st.markdown(f"[⬇️ Download]({download_url})", unsafe_allow_html=True)
                    
                    with col3:
                        # Load button
                        if st.button("📥 Load", key=f"load_{pred['sha']}"):
                            with st.spinner(f"Loading {pred['name']}..."):
                                loaded_data = st.session_state.cloud_prediction_manager.load_prediction_from_cloud(
                                    pred['name'],
                                    st.session_state.github_manager
                                )
                                
                                if loaded_data:
                                    st.session_state.prediction_results = loaded_data
                                    st.success(f"✅ Loaded {pred['name']} from cloud!")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to load {pred['name']}")
            
            if len(filtered_predictions) > 20:
                st.info(f"Showing 20 of {len(filtered_predictions)} predictions. Use filters to narrow down results.")
            
            # Bulk operations
            st.subheader("🔄 Bulk Operations")
            
            bulk_col1, bulk_col2 = st.columns(2)
            
            with bulk_col1:
                if st.button("📥 Load Latest Prediction", use_container_width=True):
                    if filtered_predictions:
                        latest = filtered_predictions[0]
                        with st.spinner(f"Loading {latest['name']}..."):
                            loaded_data = st.session_state.cloud_prediction_manager.load_prediction_from_cloud(
                                latest['name'],
                                st.session_state.github_manager
                            )
                            
                            if loaded_data:
                                st.session_state.prediction_results = loaded_data
                                st.success(f"✅ Loaded latest prediction: {latest['name']}")
                                st.rerun()
            
            with bulk_col2:
                # Export selected predictions as dataset
                selected_files = st.multiselect(
                    "Select predictions to export as dataset",
                    options=[p["name"] for p in filtered_predictions],
                    help="Combine multiple predictions into a single dataset"
                )
                
                if selected_files and st.button("📊 Export as Dataset", use_container_width=True):
                    with st.spinner("Creating combined dataset..."):
                        all_data = []
                        for filename in selected_files:
                            data = st.session_state.cloud_prediction_manager.load_prediction_from_cloud(
                                filename,
                                st.session_state.github_manager
                            )
                            if data:
                                all_data.append(data)
                        
                        if all_data:
                            # Create combined dataset
                            combined_stats = []
                            for data in all_data:
                                if 'stress_fields' in data:
                                    for field_name, field_data in data['stress_fields'].items():
                                        if isinstance(field_data, np.ndarray):
                                            stats = {
                                                'source_file': filename,
                                                'field': field_name,
                                                'max': float(np.max(field_data)),
                                                'mean': float(np.mean(field_data)),
                                                'std': float(np.std(field_data))
                                            }
                                            stats.update(data.get('target_params', {}))
                                            combined_stats.append(stats)
                            
                            if combined_stats:
                                df_combined = pd.DataFrame(combined_stats)
                                
                                dataset_result = st.session_state.cloud_dataset_manager.save_dataset_to_cloud(
                                    dataset=df_combined,
                                    dataset_name=f"combined_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    github_manager=st.session_state.github_manager,
                                    description=f"Combined dataset from {len(selected_files)} predictions",
                                    tags=["combined", "bulk-export", "dataset"]
                                )
                                
                                if dataset_result["success"]:
                                    st.success(f"✅ Combined dataset saved with {len(combined_stats)} records!")
                                else:
                                    st.error(f"❌ Failed to save dataset: {dataset_result.get('error')}")

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        create_attention_interface_with_github()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.exception("Application crashed")

st.caption(f"🔬 Attention Interpolation • GitHub Cloud • {datetime.now().year}")
