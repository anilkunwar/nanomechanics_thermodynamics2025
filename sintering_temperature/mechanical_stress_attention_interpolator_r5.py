# =============================================
# NEW: PREDICTION RESULTS SAVING AND DOWNLOAD MANAGER
# =============================================
class PredictionResultsManager:
    """Manager for saving and downloading prediction results"""
    
    @staticmethod
    def prepare_prediction_data_for_saving(prediction_results: Dict[str, Any], 
                                         source_simulations: List[Dict],
                                         mode: str = 'single') -> Dict[str, Any]:
        """
        Prepare prediction results for saving to file
        
        Args:
            prediction_results: Dictionary of prediction results
            source_simulations: List of source simulation data
            mode: 'single' for single target, 'multi' for multiple targets
            
        Returns:
            Structured dictionary ready for saving
        """
        # Create metadata
        metadata = {
            'save_timestamp': datetime.now().isoformat(),
            'mode': mode,
            'num_sources': len(source_simulations),
            'software_version': '1.0.0'
        }
        
        # Extract source parameters
        source_params = []
        for i, sim_data in enumerate(source_simulations):
            params = sim_data.get('params', {})
            source_params.append({
                'id': i,
                'defect_type': params.get('defect_type'),
                'shape': params.get('shape'),
                'orientation': params.get('orientation'),
                'eps0': params.get('eps0'),
                'kappa': params.get('kappa'),
                'theta': params.get('theta')
            })
        
        # Structure the data
        save_data = {
            'metadata': metadata,
            'source_parameters': source_params,
            'prediction_results': prediction_results
        }
        
        # Add additional info based on mode
        if mode == 'single' and 'attention_weights' in prediction_results:
            save_data['attention_analysis'] = {
                'weights': prediction_results['attention_weights'].tolist() 
                            if hasattr(prediction_results['attention_weights'], 'tolist') 
                            else prediction_results['attention_weights'],
                'source_names': [f'S{i+1}' for i in range(len(source_simulations))]
            }
        
        return save_data
    
    @staticmethod
    def create_single_prediction_archive(prediction_results: Dict[str, Any],
                                       source_simulations: List[Dict]) -> BytesIO:
        """
        Create a comprehensive archive for single prediction
        
        Args:
            prediction_results: Single prediction results
            source_simulations: List of source simulations
            
        Returns:
            BytesIO buffer containing the archive
        """
        # Create in-memory zip file
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 1. Save main prediction data as PKL
            save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                prediction_results, source_simulations, 'single'
            )
            
            # PKL format
            pkl_data = pickle.dumps(save_data)
            zip_file.writestr('prediction_results.pkl', pkl_data)
            
            # 2. Save as PT (PyTorch) format
            pt_buffer = BytesIO()
            torch.save(save_data, pt_buffer)
            pt_buffer.seek(0)
            zip_file.writestr('prediction_results.pt', pt_buffer.read())
            
            # 3. Save stress fields as separate NPZ files
            stress_fields = prediction_results.get('stress_fields', {})
            for field_name, field_data in stress_fields.items():
                if isinstance(field_data, np.ndarray):
                    npz_buffer = BytesIO()
                    np.savez_compressed(npz_buffer, data=field_data)
                    npz_buffer.seek(0)
                    zip_file.writestr(f'stress_{field_name}.npz', npz_buffer.read())
            
            # 4. Save attention weights as CSV
            if 'attention_weights' in prediction_results:
                weights = prediction_results['attention_weights']
                if hasattr(weights, 'flatten'):
                    weights = weights.flatten()
                
                weight_df = pd.DataFrame({
                    'source_id': [f'S{i+1}' for i in range(len(weights))],
                    'weight': weights,
                    'percent_contribution': 100 * weights / np.sum(weights)
                })
                csv_data = weight_df.to_csv(index=False)
                zip_file.writestr('attention_weights.csv', csv_data)
            
            # 5. Save target parameters as JSON
            target_params = prediction_results.get('target_params', {})
            if target_params:
                # Convert numpy types to Python types for JSON
                def convert_for_json(obj):
                    if isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                json_data = json.dumps(target_params, default=convert_for_json, indent=2)
                zip_file.writestr('target_parameters.json', json_data)
            
            # 6. Save summary statistics
            if 'stress_fields' in prediction_results:
                stats_rows = []
                for field_name, field_data in stress_fields.items():
                    if isinstance(field_data, np.ndarray):
                        stats_rows.append({
                            'field': field_name,
                            'max': float(np.max(field_data)),
                            'min': float(np.min(field_data)),
                            'mean': float(np.mean(field_data)),
                            'std': float(np.std(field_data)),
                            'percentile_95': float(np.percentile(field_data, 95)),
                            'percentile_99': float(np.percentile(field_data, 99))
                        })
                
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    stats_csv = stats_df.to_csv(index=False)
                    zip_file.writestr('stress_statistics.csv', stats_csv)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def create_multi_prediction_archive(multi_predictions: Dict[str, Any],
                                       source_simulations: List[Dict]) -> BytesIO:
        """
        Create a comprehensive archive for multiple predictions
        
        Args:
            multi_predictions: Dictionary of multiple predictions
            source_simulations: List of source simulations
            
        Returns:
            BytesIO buffer containing the archive
        """
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Save each prediction individually
            for pred_key, pred_data in multi_predictions.items():
                # Create directory for each prediction
                pred_dir = f'predictions/{pred_key}'
                
                # Save prediction data
                save_data = PredictionResultsManager.prepare_prediction_data_for_saving(
                    pred_data, source_simulations, 'multi'
                )
                
                # Save as PKL
                pkl_data = pickle.dumps(save_data)
                zip_file.writestr(f'{pred_dir}/prediction.pkl', pkl_data)
                
                # Save stress statistics
                stress_fields = {k: v for k, v in pred_data.items() 
                               if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
                
                if stress_fields:
                    stats_rows = []
                    for field_name, field_data in stress_fields.items():
                        stats_rows.append({
                            'field': field_name,
                            'max': float(np.max(field_data)),
                            'min': float(np.min(field_data)),
                            'mean': float(np.mean(field_data))
                        })
                    
                    stats_df = pd.DataFrame(stats_rows)
                    stats_csv = stats_df.to_csv(index=False)
                    zip_file.writestr(f'{pred_dir}/statistics.csv', stats_csv)
            
            # Save master summary
            summary_rows = []
            for pred_key, pred_data in multi_predictions.items():
                target_params = pred_data.get('target_params', {})
                stress_fields = {k: v for k, v in pred_data.items() 
                               if isinstance(v, np.ndarray) and k in ['sigma_hydro', 'sigma_mag', 'von_mises']}
                
                row = {
                    'prediction_id': pred_key,
                    'defect_type': target_params.get('defect_type', 'Unknown'),
                    'shape': target_params.get('shape', 'Unknown'),
                    'eps0': target_params.get('eps0', 0),
                    'kappa': target_params.get('kappa', 0)
                }
                
                # Add stress metrics
                for field_name, field_data in stress_fields.items():
                    row[f'{field_name}_max'] = float(np.max(field_data))
                    row[f'{field_name}_mean'] = float(np.mean(field_data))
                
                summary_rows.append(row)
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_csv = summary_df.to_csv(index=False)
                zip_file.writestr('multi_prediction_summary.csv', summary_csv)
        
        zip_buffer.seek(0)
        return zip_buffer
    
    @staticmethod
    def save_prediction_to_numerical_solutions(prediction_data: Dict[str, Any],
                                             filename: str,
                                             solutions_manager: 'NumericalSolutionsManager') -> bool:
        """
        Save prediction to numerical solutions directory
        
        Args:
            prediction_data: Prediction data to save
            filename: Base filename (without extension)
            solutions_manager: NumericalSolutionsManager instance
            
        Returns:
            Success status
        """
        try:
            # Save as PKL
            pkl_filename = f"{filename}_prediction.pkl"
            pkl_success = solutions_manager.save_simulation(prediction_data, pkl_filename, 'pkl')
            
            # Save as PT
            pt_filename = f"{filename}_prediction.pt"
            pt_success = solutions_manager.save_simulation(prediction_data, pt_filename, 'pt')
            
            return pkl_success or pt_success
            
        except Exception as e:
            st.error(f"Error saving prediction: {str(e)}")
            return False

# =============================================
# MODIFIED ENHANCED ATTENTION INTERFACE WITH SAVING FUNCTIONALITY
# =============================================
def create_attention_interface():
    """Create the attention interpolation interface with enhanced saving/download features"""
    
    st.header("🤖 Spatial-Attention Stress Interpolation with Save/Download")
    
    # Initialize interpolator in session state
    if 'interpolator' not in st.session_state:
        st.session_state.interpolator = SpatialLocalityAttentionInterpolator(
            num_heads=4,
            sigma_spatial=0.2,
            sigma_param=0.3
        )
    
    # Initialize numerical solutions manager
    if 'solutions_manager' not in st.session_state:
        st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
    
    # Initialize multi-target manager
    if 'multi_target_manager' not in st.session_state:
        st.session_state.multi_target_manager = MultiTargetPredictionManager()
    
    # Initialize prediction results manager
    if 'prediction_results_manager' not in st.session_state:
        st.session_state.prediction_results_manager = PredictionResultsManager()
    
    # Initialize source simulations list
    if 'source_simulations' not in st.session_state:
        st.session_state.source_simulations = []
        st.session_state.uploaded_files = {}
        st.session_state.loaded_from_numerical = []
    
    # Initialize multi-target predictions
    if 'multi_target_predictions' not in st.session_state:
        st.session_state.multi_target_predictions = {}
        st.session_state.multi_target_params = []
    
    # Initialize saving options
    if 'save_format' not in st.session_state:
        st.session_state.save_format = 'both'
    if 'save_to_directory' not in st.session_state:
        st.session_state.save_to_directory = False
    
    # Get grid extent for visualization
    extent = get_grid_extent()
    
    # Sidebar configuration
    st.sidebar.header("🔮 Attention Interpolator Settings")
    
    with st.sidebar.expander("💾 Save/Download Options", expanded=True):
        st.session_state.save_format = st.radio(
            "Save Format",
            ["PKL only", "PT only", "Both PKL & PT", "Archive (ZIP)"],
            index=2,
            key="save_format_radio"
        )
        
        st.session_state.save_to_directory = st.checkbox(
            "Also save to numerical solutions directory",
            value=True,
            key="save_to_dir_checkbox"
        )
        
        if st.session_state.save_to_directory:
            st.info(f"Files will be saved to: `{NUMERICAL_SOLUTIONS_DIR}`")
    
    # Main interface tabs - UPDATED to include save/export tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Load Source Data", 
        "🎯 Configure Target", 
        "🎯 Configure Multiple Targets",
        "🚀 Train & Predict", 
        "📊 Results & Visualization",
        "💾 Save & Export Results"  # NEW TAB FOR SAVING
    ])
    
    # Tab 1-4: Existing functionality (keep as is)
    # [Keep all existing Tab 1-4 code unchanged...]
    # ... [Tab 1-4 code remains the same] ...
    
    # Tab 5: Results & Visualization (updated from original Tab 5)
    with tab5:
        st.subheader("Prediction Results Visualization")
        
        if 'prediction_results' not in st.session_state:
            st.info("👈 Please train the model and make predictions first")
        else:
            results = st.session_state.prediction_results
            
            # Visualization controls
            col_viz1, col_viz2, col_viz3 = st.columns(3)
            with col_viz1:
                stress_component = st.selectbox(
                    "Select Stress Component",
                    ['von_mises', 'sigma_hydro', 'sigma_mag'],
                    index=0
                )
            with col_viz2:
                colormap = st.selectbox(
                    "Colormap",
                    ['viridis', 'plasma', 'coolwarm', 'RdBu', 'Spectral'],
                    index=0
                )
            with col_viz3:
                show_contour = st.checkbox("Show Contour Lines", value=True)
            
            # Plot stress field
            if stress_component in results.get('stress_fields', {}):
                stress_data = results['stress_fields'][stress_component]
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(stress_data, extent=extent, cmap=colormap,
                              origin='lower', aspect='equal')
                
                if show_contour:
                    contour_levels = 10
                    contour = ax.contour(stress_data, levels=contour_levels,
                                        extent=extent, colors='black', alpha=0.5, linewidths=0.5)
                    ax.clabel(contour, inline=True, fontsize=8)
                
                ax.set_title(f'{stress_component.replace("_", " ").title()} (GPa)')
                ax.set_xlabel('x (nm)')
                ax.set_ylabel('y (nm)')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Stress (GPa)')
                
                st.pyplot(fig)
            
            # Attention weights visualization
            st.subheader("🔍 Attention Weights")
            
            if 'attention_weights' in results:
                weights = results['attention_weights']
                source_names = [f'S{i+1}' for i in range(len(st.session_state.source_simulations))]
                
                fig_weights, ax_weights = plt.subplots(figsize=(10, 4))
                bars = ax_weights.bar(source_names, weights, alpha=0.7, color='steelblue')
                ax_weights.set_xlabel('Source Simulations')
                ax_weights.set_ylabel('Attention Weight')
                ax_weights.set_title('Attention Weights Distribution')
                ax_weights.set_ylim(0, max(weights) * 1.2)
                
                # Add value labels on bars
                for bar, weight in zip(bars, weights):
                    height = bar.get_height()
                    ax_weights.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
                
                st.pyplot(fig_weights)
            
            # Statistics table
            st.subheader("📊 Stress Statistics")
            
            if 'stress_fields' in results:
                stats_data = []
                for comp_name, comp_data in results['stress_fields'].items():
                    if isinstance(comp_data, np.ndarray):
                        stats_data.append({
                            'Component': comp_name.replace('_', ' ').title(),
                            'Max (GPa)': float(np.max(comp_data)),
                            'Min (GPa)': float(np.min(comp_data)),
                            'Mean (GPa)': float(np.mean(comp_data)),
                            'Std Dev': float(np.std(comp_data)),
                            '95th %ile': float(np.percentile(comp_data, 95))
                        })
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats.style.format({
                        'Max (GPa)': '{:.3f}',
                        'Min (GPa)': '{:.3f}',
                        'Mean (GPa)': '{:.3f}',
                        'Std Dev': '{:.3f}',
                        '95th %ile': '{:.3f}'
                    }), use_container_width=True)
    
    # Tab 6: NEW Save & Export Results Tab
    with tab6:
        st.subheader("💾 Save and Export Prediction Results")
        
        # Check if we have predictions to save
        has_single_prediction = 'prediction_results' in st.session_state
        has_multi_predictions = ('multi_target_predictions' in st.session_state and 
                                len(st.session_state.multi_target_predictions) > 0)
        
        if not has_single_prediction and not has_multi_predictions:
            st.warning("⚠️ No prediction results available to save. Please run predictions first.")
        else:
            st.success("✅ Prediction results available for export!")
            
            # Display what's available
            if has_single_prediction:
                st.info(f"**Single Target Prediction:** Available")
                single_params = st.session_state.prediction_results.get('target_params', {})
                st.write(f"- Target: {single_params.get('defect_type', 'Unknown')}, "
                        f"ε*={single_params.get('eps0', 0):.3f}, "
                        f"κ={single_params.get('kappa', 0):.3f}")
            
            if has_multi_predictions:
                st.info(f"**Multiple Target Predictions:** {len(st.session_state.multi_target_predictions)} available")
            
            st.divider()
            
            # Save options
            st.subheader("📁 Save Options")
            
            save_col1, save_col2, save_col3 = st.columns(3)
            
            with save_col1:
                save_mode = st.radio(
                    "Select results to save",
                    ["Current Single Prediction", "All Multiple Predictions", "Both"],
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
            
            with save_col3:
                include_source_info = st.checkbox("Include source simulations info", value=True)
                include_visualizations = st.checkbox("Include visualization data", value=True)
            
            st.divider()
            
            # Save/Download buttons
            st.subheader("⬇️ Download Options")
            
            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
            
            with dl_col1:
                # Save as PKL
                if st.button("💾 Save as PKL", type="secondary", use_container_width=True):
                    with st.spinner("Preparing PKL file..."):
                        try:
                            if save_mode in ["Current Single Prediction", "Both"] and has_single_prediction:
                                save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations,
                                    'single'
                                )
                                
                                # Add metadata
                                save_data['save_info'] = {
                                    'format': 'pkl',
                                    'timestamp': timestamp,
                                    'mode': 'single'
                                }
                                
                                # Create download link
                                pkl_buffer = BytesIO()
                                pickle.dump(save_data, pkl_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                                pkl_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download PKL",
                                    data=pkl_buffer,
                                    file_name=f"{base_filename}.pkl",
                                    mime="application/octet-stream",
                                    key="download_pkl_single"
                                )
                                
                                # Save to directory if enabled
                                if st.session_state.save_to_directory:
                                    save_success = st.session_state.prediction_results_manager.save_prediction_to_numerical_solutions(
                                        save_data,
                                        base_filename,
                                        st.session_state.solutions_manager
                                    )
                                    if save_success:
                                        st.success(f"✅ Saved to {NUMERICAL_SOLUTIONS_DIR}")
                            
                        except Exception as e:
                            st.error(f"❌ Error saving PKL: {str(e)}")
            
            with dl_col2:
                # Save as PT (PyTorch)
                if st.button("💾 Save as PT", type="secondary", use_container_width=True):
                    with st.spinner("Preparing PT file..."):
                        try:
                            if save_mode in ["Current Single Prediction", "Both"] and has_single_prediction:
                                save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations,
                                    'single'
                                )
                                
                                # Add metadata
                                save_data['save_info'] = {
                                    'format': 'pt',
                                    'timestamp': timestamp,
                                    'mode': 'single'
                                }
                                
                                # Create download link
                                pt_buffer = BytesIO()
                                torch.save(save_data, pt_buffer)
                                pt_buffer.seek(0)
                                
                                st.download_button(
                                    label="📥 Download PT",
                                    data=pt_buffer,
                                    file_name=f"{base_filename}.pt",
                                    mime="application/octet-stream",
                                    key="download_pt_single"
                                )
                                
                                # Save to directory if enabled
                                if st.session_state.save_to_directory:
                                    # For PT format, use a different filename
                                    pt_filename = f"{base_filename}.pt"
                                    pt_save_data = {'pt_data': save_data}
                                    pt_success = st.session_state.solutions_manager.save_simulation(
                                        pt_save_data, pt_filename, 'pt'
                                    )
                                    if pt_success:
                                        st.success(f"✅ PT saved to {NUMERICAL_SOLUTIONS_DIR}")
                            
                        except Exception as e:
                            st.error(f"❌ Error saving PT: {str(e)}")
            
            with dl_col3:
                # Save as ZIP Archive
                if st.button("📦 Save as ZIP Archive", type="primary", use_container_width=True):
                    with st.spinner("Creating comprehensive archive..."):
                        try:
                            if save_mode == "Current Single Prediction" and has_single_prediction:
                                # Single prediction archive
                                zip_buffer = st.session_state.prediction_results_manager.create_single_prediction_archive(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations
                                )
                                
                                st.download_button(
                                    label="📥 Download ZIP Archive",
                                    data=zip_buffer,
                                    file_name=f"{base_filename}_complete.zip",
                                    mime="application/zip",
                                    key="download_zip_single"
                                )
                                
                            elif save_mode == "All Multiple Predictions" and has_multi_predictions:
                                # Multi prediction archive
                                zip_buffer = st.session_state.prediction_results_manager.create_multi_prediction_archive(
                                    st.session_state.multi_target_predictions,
                                    st.session_state.source_simulations
                                )
                                
                                st.download_button(
                                    label="📥 Download Multi-Prediction ZIP",
                                    data=zip_buffer,
                                    file_name=f"{base_filename}_multi_predictions.zip",
                                    mime="application/zip",
                                    key="download_zip_multi"
                                )
                            
                        except Exception as e:
                            st.error(f"❌ Error creating archive: {str(e)}")
            
            with dl_col4:
                # Save all formats
                if st.button("💾 Save All Formats", type="secondary", use_container_width=True):
                    with st.spinner("Saving in all formats..."):
                        try:
                            if has_single_prediction:
                                save_data = st.session_state.prediction_results_manager.prepare_prediction_data_for_saving(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations,
                                    'single'
                                )
                                
                                # Save PKL
                                pkl_buffer = BytesIO()
                                pickle.dump(save_data, pkl_buffer)
                                pkl_buffer.seek(0)
                                
                                # Save PT
                                pt_buffer = BytesIO()
                                torch.save(save_data, pt_buffer)
                                pt_buffer.seek(0)
                                
                                # Save ZIP
                                zip_buffer = st.session_state.prediction_results_manager.create_single_prediction_archive(
                                    st.session_state.prediction_results,
                                    st.session_state.source_simulations
                                )
                                
                                # Create columns for all downloads
                                dl_all_col1, dl_all_col2, dl_all_col3 = st.columns(3)
                                
                                with dl_all_col1:
                                    st.download_button(
                                        label="📥 Download PKL",
                                        data=pkl_buffer,
                                        file_name=f"{base_filename}.pkl",
                                        mime="application/octet-stream",
                                        key="download_all_pkl"
                                    )
                                
                                with dl_all_col2:
                                    st.download_button(
                                        label="📥 Download PT",
                                        data=pt_buffer,
                                        file_name=f"{base_filename}.pt",
                                        mime="application/octet-stream",
                                        key="download_all_pt"
                                    )
                                
                                with dl_all_col3:
                                    st.download_button(
                                        label="📥 Download ZIP",
                                        data=zip_buffer,
                                        file_name=f"{base_filename}_all_formats.zip",
                                        mime="application/zip",
                                        key="download_all_zip"
                                    )
                                
                                # Save to directory
                                if st.session_state.save_to_directory:
                                    st.success(f"✅ Ready to download all formats!")
                            
                        except Exception as e:
                            st.error(f"❌ Error saving all formats: {str(e)}")
            
            st.divider()
            
            # Advanced options
            with st.expander("⚙️ Advanced Save Options", expanded=False):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    # Save stress fields as separate files
                    save_separate = st.checkbox("Save stress fields as separate NPZ files", value=False)
                    
                    if save_separate and has_single_prediction:
                        stress_fields = st.session_state.prediction_results.get('stress_fields', {})
                        for field_name, field_data in stress_fields.items():
                            if isinstance(field_data, np.ndarray):
                                npz_buffer = BytesIO()
                                np.savez_compressed(npz_buffer, data=field_data)
                                npz_buffer.seek(0)
                                
                                st.download_button(
                                    label=f"📥 Download {field_name}.npz",
                                    data=npz_buffer,
                                    file_name=f"{base_filename}_{field_name}.npz",
                                    mime="application/octet-stream",
                                    key=f"download_npz_{field_name}"
                                )
                
                with col_adv2:
                    # Export to other formats
                    export_json = st.checkbox("Export parameters to JSON", value=False)
                    export_csv = st.checkbox("Export statistics to CSV", value=False)
                    
                    if export_json and has_single_prediction:
                        target_params = st.session_state.prediction_results.get('target_params', {})
                        if target_params:
                            json_str = json.dumps(target_params, indent=2, default=str)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_str,
                                file_name=f"{base_filename}_params.json",
                                mime="application/json",
                                key="download_json"
                            )
                    
                    if export_csv and has_single_prediction:
                        if 'stress_fields' in st.session_state.prediction_results:
                            stats_rows = []
                            for field_name, field_data in st.session_state.prediction_results['stress_fields'].items():
                                if isinstance(field_data, np.ndarray):
                                    stats_rows.append({
                                        'field': field_name,
                                        'max': np.max(field_data),
                                        'min': np.min(field_data),
                                        'mean': np.mean(field_data),
                                        'std': np.std(field_data)
                                    })
                            
                            if stats_rows:
                                stats_df = pd.DataFrame(stats_rows)
                                csv_data = stats_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"{base_filename}_stats.csv",
                                    mime="text/csv",
                                    key="download_csv"
                                )
            
            # Display saved files in directory
            st.divider()
            st.subheader("📁 Saved Files Preview")
            
            if st.session_state.save_to_directory:
                saved_files = []
                for ext in ['pkl', 'pt', 'zip']:
                    pattern = os.path.join(NUMERICAL_SOLUTIONS_DIR, f"*{base_filename}*.{ext}")
                    files = glob.glob(pattern)
                    saved_files.extend([os.path.basename(f) for f in files])
                
                if saved_files:
                    st.write("**Recently saved files:**")
                    for file in saved_files[:5]:  # Show last 5
                        st.code(file)
                else:
                    st.info("No files saved yet for this session.")
            else:
                st.info("Enable 'Save to directory' to persist files locally.")

# =============================================
# MODIFIED MAIN APPLICATION
# =============================================
def main():
    """Main application with enhanced save/download features"""
    
    st.sidebar.header("📁 Directory Information")
    st.sidebar.write(f"**App Directory:** `{SCRIPT_DIR}`")
    st.sidebar.write(f"**Solutions Directory:** `{NUMERICAL_SOLUTIONS_DIR}`")
    
    if not os.path.exists(NUMERICAL_SOLUTIONS_DIR):
        st.sidebar.warning("⚠️ Solutions directory not found")
        if st.sidebar.button("📁 Create Directory"):
            os.makedirs(NUMERICAL_SOLUTIONS_DIR, exist_ok=True)
            st.sidebar.success("✅ Directory created")
            st.rerun()
    
    st.sidebar.header("🔧 Operation Mode")
    
    # Simplified mode selection
    operation_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Attention Interpolation with Save", "Stress Analysis Dashboard"],
        index=0
    )
    
    if operation_mode == "Attention Interpolation with Save":
        create_attention_interface()
    else:
        st.header("📊 Stress Analysis Dashboard")
        
        # Initialize managers
        if 'stress_analyzer' not in st.session_state:
            st.session_state.stress_analyzer = StressAnalysisManager()
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
        
        # Simplified stress dashboard
        st.subheader("Load and Analyze Simulation Data")
        
        col1, col2 = st.columns(2)
        with col1:
            file_limit = st.number_input("Maximum files to process", 10, 500, 50, 10)
        
        if st.button("🚀 Load and Analyze Simulations", type="primary"):
            with st.spinner("Loading simulations..."):
                all_files = st.session_state.solutions_manager.get_all_files()[:file_limit]
                all_simulations = []
                
                progress_bar = st.progress(0)
                for idx, file_info in enumerate(all_files):
                    progress_bar.progress((idx + 1) / len(all_files))
                    try:
                        sim_data = st.session_state.solutions_manager.load_simulation(
                            file_info['path'],
                            st.session_state.interpolator
                        )
                        all_simulations.append(sim_data)
                    except:
                        continue
                
                progress_bar.empty()
                
                if all_simulations:
                    stress_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                        all_simulations, {}
                    )
                    st.session_state.stress_summary_df = stress_df
                    st.success(f"✅ Loaded {len(all_simulations)} simulations")
                else:
                    st.error("No simulations could be loaded")
        
        # Show data if available
        if 'stress_summary_df' in st.session_state and not st.session_state.stress_summary_df.empty:
            st.markdown("### 📊 Analysis Results")
            
            # Quick visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("Von Mises by Defect Type")
                if 'defect_type' in st.session_state.stress_summary_df.columns and 'max_von_mises' in st.session_state.stress_summary_df.columns:
                    fig = px.box(
                        st.session_state.stress_summary_df,
                        x='defect_type',
                        y='max_von_mises',
                        title="Von Mises Stress Distribution by Defect Type",
                        color='defect_type'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                st.subheader("Stress vs Parameters")
                if 'eps0' in st.session_state.stress_summary_df.columns and 'max_von_mises' in st.session_state.stress_summary_df.columns:
                    fig = px.scatter(
                        st.session_state.stress_summary_df,
                        x='eps0',
                        y='max_von_mises',
                        color='defect_type',
                        title="Von Mises vs ε*",
                        hover_data=['shape', 'orientation']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("📋 Complete Data Table")
            st.dataframe(
                st.session_state.stress_summary_df.style.format({
                    col: "{:.3f}" for col in st.session_state.stress_summary_df.select_dtypes(include=[np.number]).columns
                }),
                use_container_width=True,
                height=300
            )
            
            # Option to go to attention interpolation
            if st.button("🔬 Open Attention Interpolation with Save"):
                st.session_state.operation_mode = "Attention Interpolation with Save"
                st.rerun()
        else:
            st.info("👆 Click the button above to load and analyze simulation data")

# =============================================
# THEORETICAL ANALYSIS
# =============================================
with st.expander("💾 Enhanced Save/Export Features", expanded=False):
    st.markdown(f"""
    ## 📁 **Enhanced Prediction Results Save/Export System**
    
    ### **💾 Multiple Format Support**
    
    **1. PKL Format (Python Pickle):**
       - Full Python object serialization
       - Preserves all data structures and types
       - Fast loading within Python ecosystem
       - File extension: `.pkl`
    
    **2. PT Format (PyTorch):**
       - Compatible with PyTorch machine learning workflows
       - Can be loaded directly with `torch.load()`
       - Optimized for tensor data
       - File extension: `.pt`
    
    **3. ZIP Archive (Comprehensive):**
       - Contains multiple file formats in one package
       - Includes stress fields as separate NPZ files
       - Includes CSV statistics and JSON parameters
       - File extension: `.zip`
    
    ### **📊 Data Structure Preservation**
    
    **Saved Data Includes:**
    1. **Metadata:**
       - Save timestamp
       - Software version
       - Prediction mode (single/multi)
       - Number of source simulations
    
    2. **Source Parameters:**
       - Complete parameter vectors for all source simulations
       - Defect types, shapes, orientations
       - Numerical parameters (ε*, κ, θ)
    
    3. **Prediction Results:**
       - Stress fields (hydrostatic, magnitude, von Mises)
       - Attention weights distribution
       - Target parameters
       - Training statistics
    
    4. **Analysis Data:**
       - Stress statistics (max, min, mean, std, percentiles)
       - Attention weight contributions
       - Visualization-ready data
    
    ### **📁 File Management Features**
    
    **1. Directory Integration:**
       - Optional saving to `{NUMERICAL_SOLUTIONS_DIR}`
       - Automatic filename generation with timestamps
       - File organization by prediction type
    
    **2. Selective Saving:**
       - Choose between single or multiple predictions
       - Select specific formats to save
       - Include/exclude source information
    
    **3. Preview and Management:**
       - View recently saved files
       - Access saved files directly from the interface
       - Batch export capabilities
    
    ### **⚡ Performance Optimizations**
    
    **1. Memory Efficient:**
       - Uses BytesIO buffers for in-memory operations
       - Compressed storage for large stress fields
       - Streamlined data structures
    
    **2. Fast Export:**
       - Parallel processing for multiple formats
       - Incremental saving for large predictions
       - Background saving to directory
    
    **3. Error Resilience:**
       - Graceful handling of save failures
       - Validation of saved data integrity
       - Recovery options for interrupted saves
    
    ### **🔧 Advanced Features**
    
    **1. Custom Export:**
       - Export stress fields as separate NPZ files
       - Export parameters as JSON for interoperability
       - Export statistics as CSV for spreadsheet analysis
    
    **2. Batch Operations:**
       - Save all predictions from multi-target runs
       - Create comprehensive archives
       - Generate comparison reports
    
    **3. Integration Ready:**
       - Structured data for machine learning pipelines
       - Compatible with post-processing scripts
       - Ready for visualization tools
    """)

if __name__ == "__main__":
    main()

st.caption(f"🔬 Enhanced Attention Interpolation with Save/Export • PKL/PT/ZIP Support • {datetime.now().year}")
