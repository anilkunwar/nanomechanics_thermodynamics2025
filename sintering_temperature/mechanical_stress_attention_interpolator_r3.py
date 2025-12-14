# =============================================
# COMPREHENSIVE STRESS DASHBOARD
# =============================================
class ComprehensiveStressDashboard:
    """Comprehensive stress analysis and visualization dashboard"""
    
    def __init__(self):
        self.colormaps = plt.colormaps()
        
    def create_dashboard_layout(self):
        """Create the complete stress dashboard layout"""
        
        st.header("📊 Comprehensive Stress Analysis Dashboard")
        
        # Dashboard overview metrics
        self._display_dashboard_overview()
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📈 Summary Metrics",
            "📊 Distribution Analysis",
            "🏔️ Peak Analysis",
            "🔍 Gradient Analysis",
            "📋 Data Explorer",
            "📊 Comparative Analysis",
            "📈 Time Series",
            "📤 Export & Report"
        ])
        
        with tab1:
            self._display_summary_metrics()
        
        with tab2:
            self._display_distribution_analysis()
        
        with tab3:
            self._display_peak_analysis()
        
        with tab4:
            self._display_gradient_analysis()
        
        with tab5:
            self._display_data_explorer()
        
        with tab6:
            self._display_comparative_analysis()
        
        with tab7:
            self._display_time_series_analysis()
        
        with tab8:
            self._display_export_section()
    
    def _display_dashboard_overview(self):
        """Display dashboard overview metrics"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not st.session_state.stress_summary_df.empty:
                total_simulations = len(st.session_state.stress_summary_df)
                st.metric("Total Simulations", total_simulations)
            else:
                st.metric("Total Simulations", 0)
        
        with col2:
            if not st.session_state.stress_summary_df.empty and 'max_von_mises' in st.session_state.stress_summary_df.columns:
                max_vm = st.session_state.stress_summary_df['max_von_mises'].max()
                st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
            else:
                st.metric("Max Von Mises", "N/A")
        
        with col3:
            if not st.session_state.stress_summary_df.empty and 'max_abs_hydrostatic' in st.session_state.stress_summary_df.columns:
                max_hydro = st.session_state.stress_summary_df['max_abs_hydrostatic'].max()
                st.metric("Max Hydrostatic", f"{max_hydro:.2f} GPa")
            else:
                st.metric("Max Hydrostatic", "N/A")
        
        with col4:
            if not st.session_state.stress_summary_df.empty and 'defect_type' in st.session_state.stress_summary_df.columns:
                defect_types = st.session_state.stress_summary_df['defect_type'].nunique()
                st.metric("Defect Types", defect_types)
            else:
                st.metric("Defect Types", 0)
        
        # Quick actions row
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Dashboard", type="secondary"):
                st.rerun()
        with col2:
            if st.button("📊 Update All Statistics", type="primary"):
                with st.spinner("Updating all stress statistics..."):
                    self._update_all_statistics()
        with col3:
            if st.button("🧹 Clear Dashboard", type="secondary"):
                st.session_state.stress_summary_df = pd.DataFrame()
                st.success("Dashboard cleared!")
                st.rerun()
    
    def _update_all_statistics(self):
        """Update all stress statistics"""
        if st.session_state.source_simulations or st.session_state.multi_target_predictions:
            st.session_state.stress_summary_df = (
                st.session_state.stress_analyzer.create_stress_summary_dataframe(
                    st.session_state.source_simulations,
                    st.session_state.multi_target_predictions
                )
            )
            st.success(f"✅ Updated statistics for {len(st.session_state.stress_summary_df)} simulations")
        else:
            st.warning("No simulations loaded")
    
    def _display_summary_metrics(self):
        """Display comprehensive summary metrics"""
        
        st.subheader("📈 Stress Summary Metrics")
        
        if st.session_state.stress_summary_df.empty:
            st.info("No stress data available. Please load simulations first.")
            return
        
        # Key metrics selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_metrics = st.multiselect(
                "Select metrics to display",
                options=[
                    'max_von_mises', 'mean_von_mises', 'max_abs_hydrostatic',
                    'max_stress_magnitude', 'max_shear_tresca', 'hydro_std',
                    'von_mises_p95', 'von_mises_p99', 'stress_ratio_vm_hydro'
                ],
                default=['max_von_mises', 'max_abs_hydrostatic', 'max_shear_tresca']
            )
        
        with col2:
            group_by = st.selectbox(
                "Group by",
                options=['defect_type', 'shape', 'orientation', 'type', 'None'],
                index=0
            )
        
        if selected_metrics:
            # Create summary table
            summary_data = []
            for metric in selected_metrics:
                if metric in st.session_state.stress_summary_df.columns:
                    if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
                        # Grouped statistics
                        groups = st.session_state.stress_summary_df[group_by].unique()
                        for group in groups:
                            group_df = st.session_state.stress_summary_df[
                                st.session_state.stress_summary_df[group_by] == group
                            ]
                            summary_data.append({
                                'Metric': f"{metric} - {group}",
                                'Mean': group_df[metric].mean(),
                                'Std': group_df[metric].std(),
                                'Min': group_df[metric].min(),
                                'Max': group_df[metric].max(),
                                'Count': len(group_df)
                            })
                    else:
                        # Overall statistics
                        summary_data.append({
                            'Metric': metric,
                            'Mean': st.session_state.stress_summary_df[metric].mean(),
                            'Std': st.session_state.stress_summary_df[metric].std(),
                            'Min': st.session_state.stress_summary_df[metric].min(),
                            'Max': st.session_state.stress_summary_df[metric].max(),
                            'Count': len(st.session_state.stress_summary_df[metric].dropna())
                        })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(
                    df_summary.style.format({
                        'Mean': '{:.3f}',
                        'Std': '{:.3f}',
                        'Min': '{:.3f}',
                        'Max': '{:.3f}'
                    }),
                    use_container_width=True
                )
        
        # KPI Cards
        st.subheader("📊 Key Performance Indicators")
        
        if not st.session_state.stress_summary_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'max_von_mises' in st.session_state.stress_summary_df.columns:
                    vm_data = st.session_state.stress_summary_df['max_von_mises'].dropna()
                    if len(vm_data) > 0:
                        st.metric("Von Mises Range", 
                                 f"{vm_data.min():.2f} - {vm_data.max():.2f} GPa")
            
            with col2:
                if 'stress_ratio_vm_hydro' in st.session_state.stress_summary_df.columns:
                    ratio_data = st.session_state.stress_summary_df['stress_ratio_vm_hydro'].dropna()
                    if len(ratio_data) > 0:
                        st.metric("VM/Hydro Ratio", 
                                 f"{ratio_data.mean():.2f} ± {ratio_data.std():.2f}")
            
            with col3:
                if 'hydro_std' in st.session_state.stress_summary_df.columns:
                    hydro_std = st.session_state.stress_summary_df['hydro_std'].mean()
                    st.metric("Avg Hydro Variability", f"{hydro_std:.2f} GPa")
            
            with col4:
                if 'defect_type' in st.session_state.stress_summary_df.columns:
                    defect_counts = st.session_state.stress_summary_df['defect_type'].value_counts()
                    if len(defect_counts) > 0:
                        dominant_defect = defect_counts.index[0]
                        st.metric("Dominant Defect", dominant_defect)
    
    def _display_distribution_analysis(self):
        """Display stress distribution analysis"""
        
        st.subheader("📊 Stress Distribution Analysis")
        
        if st.session_state.stress_summary_df.empty:
            st.info("No stress data available.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution type selection
            dist_type = st.selectbox(
                "Select distribution type",
                ["Histogram", "KDE Plot", "Box Plot", "Violin Plot", "ECDF"],
                index=0
            )
            
            # Metric selection
            available_metrics = [
                col for col in st.session_state.stress_summary_df.columns 
                if st.session_state.stress_summary_df[col].dtype in [np.float64, np.float32, np.int64]
            ]
            metric = st.selectbox(
                "Select metric",
                available_metrics,
                index=available_metrics.index('max_von_mises') if 'max_von_mises' in available_metrics else 0
            )
        
        with col2:
            # Grouping options
            group_by = st.selectbox(
                "Group by (for box/violin)",
                ['None', 'defect_type', 'shape', 'orientation', 'type'],
                index=0
            )
            
            # Color palette
            palette = st.selectbox(
                "Color palette",
                ['viridis', 'plasma', 'coolwarm', 'Set1', 'Set2', 'tab10'],
                index=0
            )
        
        if metric:
            # Create distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if dist_type == "Histogram":
                if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
                    groups = st.session_state.stress_summary_df[group_by].unique()
                    for i, group in enumerate(groups):
                        group_data = st.session_state.stress_summary_df[
                            st.session_state.stress_summary_df[group_by] == group
                        ][metric].dropna()
                        ax.hist(group_data, alpha=0.5, label=str(group), 
                               bins=30, density=True)
                    ax.legend(title=group_by)
                else:
                    ax.hist(st.session_state.stress_summary_df[metric].dropna(), 
                           bins=30, alpha=0.7, density=True)
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
            
            elif dist_type == "KDE Plot":
                if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
                    groups = st.session_state.stress_summary_df[group_by].unique()
                    for i, group in enumerate(groups):
                        group_data = st.session_state.stress_summary_df[
                            st.session_state.stress_summary_df[group_by] == group
                        ][metric].dropna()
                        if len(group_data) > 1:
                            sns.kdeplot(group_data, label=str(group), ax=ax)
                    ax.legend(title=group_by)
                else:
                    sns.kdeplot(st.session_state.stress_summary_df[metric].dropna(), ax=ax)
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_ylabel('Density')
                ax.set_title(f'KDE of {metric.replace("_", " ").title()}')
            
            elif dist_type == "Box Plot":
                if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
                    sns.boxplot(data=st.session_state.stress_summary_df, 
                               x=group_by, y=metric, ax=ax, palette=palette)
                else:
                    ax.boxplot(st.session_state.stress_summary_df[metric].dropna())
                    ax.set_xticks([])
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'Box Plot of {metric.replace("_", " ").title()}')
            
            elif dist_type == "Violin Plot":
                if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
                    sns.violinplot(data=st.session_state.stress_summary_df, 
                                  x=group_by, y=metric, ax=ax, palette=palette)
                else:
                    sns.violinplot(y=st.session_state.stress_summary_df[metric].dropna(), ax=ax)
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'Violin Plot of {metric.replace("_", " ").title()}')
            
            elif dist_type == "ECDF":
                from scipy import stats
                data = st.session_state.stress_summary_df[metric].dropna().sort_values()
                y = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, y, marker='.', linestyle='none')
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_ylabel('ECDF')
                ax.set_title(f'ECDF of {metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        # Statistical tests
        st.subheader("📊 Statistical Tests")
        
        if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
            groups = st.session_state.stress_summary_df[group_by].unique()
            if len(groups) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Run ANOVA Test"):
                        from scipy import stats
                        group_data = []
                        for group in groups:
                            group_values = st.session_state.stress_summary_df[
                                st.session_state.stress_summary_df[group_by] == group
                            ][metric].dropna()
                            if len(group_values) > 0:
                                group_data.append(group_values)
                        
                        if len(group_data) >= 2:
                            f_stat, p_value = stats.f_oneway(*group_data)
                            st.write(f"**ANOVA Results for {metric}:**")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_value:.4e}")
                            if p_value < 0.05:
                                st.success("✅ Significant differences detected (p < 0.05)")
                            else:
                                st.info("⚠️ No significant differences detected")
                
                with col2:
                    if st.button("Run Kruskal-Wallis Test"):
                        from scipy import stats
                        group_data = []
                        for group in groups:
                            group_values = st.session_state.stress_summary_df[
                                st.session_state.stress_summary_df[group_by] == group
                            ][metric].dropna()
                            if len(group_values) > 0:
                                group_data.append(group_values)
                        
                        if len(group_data) >= 2:
                            h_stat, p_value = stats.kruskal(*group_data)
                            st.write(f"**Kruskal-Wallis Results for {metric}:**")
                            st.write(f"H-statistic: {h_stat:.4f}")
                            st.write(f"p-value: {p_value:.4e}")
    
    def _display_peak_analysis(self):
        """Display stress peak analysis"""
        
        st.subheader("🏔️ Stress Peak Analysis")
        
        if not st.session_state.source_simulations:
            st.info("Please load simulations to perform peak analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            simulation_idx = st.selectbox(
                "Select simulation",
                options=range(len(st.session_state.source_simulations)),
                format_func=lambda x: f"Simulation {x+1}",
                index=0
            )
            
            stress_component = st.selectbox(
                "Select stress component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0
            )
            
            threshold_percentile = st.slider(
                "Peak detection threshold (percentile)",
                75.0, 99.9, 95.0, 0.5
            )
        
        with col2:
            min_peak_distance = st.slider(
                "Minimum peak distance (pixels)",
                1, 50, 5, 1
            )
            
            peak_display_option = st.radio(
                "Display options",
                ["Show all peaks", "Show top N peaks", "Show peaks above absolute threshold"],
                index=0
            )
            
            if peak_display_option == "Show top N peaks":
                top_n = st.number_input("Number of peaks", 1, 100, 10, 1)
            elif peak_display_option == "Show peaks above absolute threshold":
                abs_threshold = st.number_input("Absolute threshold (GPa)", 0.0, 100.0, 5.0, 0.1)
        
        if st.button("🔍 Analyze Peaks"):
            with st.spinner("Analyzing stress peaks..."):
                try:
                    sim_data = st.session_state.source_simulations[simulation_idx]
                    history = sim_data.get('history', [])
                    
                    if history:
                        eta, stress_fields = history[-1]
                        
                        if stress_component in stress_fields:
                            stress_data = stress_fields[stress_component]
                            
                            # Compute peaks using StressAnalysisManager
                            peaks = st.session_state.stress_analyzer.extract_stress_peaks(
                                {stress_component: stress_data},
                                threshold_percentile
                            )
                            
                            if peaks and stress_component in peaks:
                                peak_info = peaks[stress_component]
                                num_peaks = peak_info['num_peaks']
                                max_value = peak_info['max_value']
                                max_position = peak_info['max_position']
                                threshold = peak_info['threshold']
                                
                                # Display peak statistics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Peaks", num_peaks)
                                with col2:
                                    st.metric("Max Value", f"{max_value:.2f} GPa")
                                with col3:
                                    st.metric("Threshold", f"{threshold:.2f} GPa")
                                with col4:
                                    st.metric("Peak Density", f"{num_peaks/(stress_data.shape[0]*stress_data.shape[1]):.4f}")
                                
                                # Visualize peaks
                                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # Original stress field
                                im1 = axes[0].imshow(stress_data, cmap='hot', origin='lower')
                                axes[0].set_title(f'Original {stress_component}')
                                plt.colorbar(im1, ax=axes[0], label='GPa')
                                
                                # Peaks overlay
                                im2 = axes[1].imshow(stress_data, cmap='hot', origin='lower', alpha=0.7)
                                peak_indices = peak_info['peak_indices']
                                
                                # Filter peaks based on display option
                                if peak_display_option == "Show top N peaks":
                                    top_indices = np.argsort(peak_info['peak_values'])[-top_n:]
                                    filtered_indices = (peak_indices[0][top_indices], peak_indices[1][top_indices])
                                elif peak_display_option == "Show peaks above absolute threshold":
                                    mask = peak_info['peak_values'] > abs_threshold
                                    filtered_indices = (peak_indices[0][mask], peak_indices[1][mask])
                                else:
                                    filtered_indices = peak_indices
                                
                                axes[1].scatter(filtered_indices[1], filtered_indices[0], 
                                               c='cyan', s=30, marker='o', edgecolors='white', 
                                               label=f'{len(filtered_indices[0])} peaks')
                                axes[1].set_title(f'Peaks > {threshold_percentile:.1f}th percentile')
                                axes[1].legend()
                                plt.colorbar(im2, ax=axes[1], label='GPa')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Peak statistics table
                                st.subheader("📋 Peak Statistics")
                                
                                if len(filtered_indices[0]) > 0:
                                    peak_values = stress_data[filtered_indices]
                                    sorted_indices = np.argsort(peak_values)[::-1]  # Descending
                                    
                                    peak_table = []
                                    for i, idx in enumerate(sorted_indices[:20]):  # Top 20 peaks
                                        y, x = filtered_indices[0][idx], filtered_indices[1][idx]
                                        peak_table.append({
                                            'Rank': i + 1,
                                            'X Position': x,
                                            'Y Position': y,
                                            'Value (GPa)': peak_values[idx],
                                            'Relative to Threshold': f"{(peak_values[idx]/threshold - 1)*100:.1f}%"
                                        })
                                    
                                    df_peaks = pd.DataFrame(peak_table)
                                    st.dataframe(df_peaks, use_container_width=True)
                                
                                else:
                                    st.warning("No peaks found with current criteria")
                            
                            else:
                                st.warning(f"No peaks detected for {stress_component} above {threshold_percentile}th percentile")
                    
                    else:
                        st.warning("No history data available for selected simulation")
                        
                except Exception as e:
                    st.error(f"Error analyzing peaks: {str(e)}")
    
    def _display_gradient_analysis(self):
        """Display stress gradient analysis"""
        
        st.subheader("🔍 Stress Gradient Analysis")
        
        if not st.session_state.source_simulations:
            st.info("Please load simulations to perform gradient analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            simulation_idx = st.selectbox(
                "Select simulation for gradient analysis",
                options=range(len(st.session_state.source_simulations)),
                format_func=lambda x: f"Simulation {x+1}",
                index=0,
                key="grad_sim_select"
            )
            
            stress_component = st.selectbox(
                "Select stress component",
                options=['von_mises', 'sigma_hydro', 'sigma_mag'],
                index=0,
                key="grad_component_select"
            )
        
        with col2:
            gradient_type = st.selectbox(
                "Gradient type",
                ["Magnitude", "X-component", "Y-component", "Direction"],
                index=0
            )
            
            smoothing_sigma = st.slider(
                "Smoothing sigma (for gradient calculation)",
                0.0, 2.0, 0.5, 0.1
            )
        
        if st.button("📐 Compute Gradients"):
            with st.spinner("Computing stress gradients..."):
                try:
                    sim_data = st.session_state.source_simulations[simulation_idx]
                    history = sim_data.get('history', [])
                    
                    if history:
                        eta, stress_fields = history[-1]
                        
                        if stress_component in stress_fields:
                            stress_data = stress_fields[stress_component]
                            
                            # Apply smoothing if requested
                            if smoothing_sigma > 0:
                                stress_data = gaussian_filter(stress_data, sigma=smoothing_sigma)
                            
                            # Compute gradients
                            gradients = st.session_state.stress_analyzer.compute_stress_gradients(
                                {stress_component: stress_data}
                            )
                            
                            # Display gradient statistics
                            st.subheader("📊 Gradient Statistics")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                max_grad_key = f'{stress_component}_max_grad'
                                if max_grad_key in gradients:
                                    st.metric("Max Gradient", f"{gradients[max_grad_key]:.3f} GPa/px")
                            with col2:
                                mean_grad_key = f'{stress_component}_mean_grad'
                                if mean_grad_key in gradients:
                                    st.metric("Mean Gradient", f"{gradients[mean_grad_key]:.3f} GPa/px")
                            with col3:
                                # Calculate gradient variance
                                grad_mag_key = f'{stress_component}_grad_mag'
                                if grad_mag_key in gradients:
                                    grad_mag = gradients[grad_mag_key]
                                    st.metric("Gradient Std", f"{np.std(grad_mag):.3f} GPa/px")
                            with col4:
                                # Calculate gradient entropy (measure of complexity)
                                if grad_mag_key in gradients:
                                    from scipy import stats
                                    grad_flat = gradients[grad_mag_key].flatten()
                                    if len(grad_flat) > 0:
                                        hist, _ = np.histogram(grad_flat, bins=50, density=True)
                                        hist = hist[hist > 0]
                                        entropy = -np.sum(hist * np.log(hist))
                                        st.metric("Gradient Entropy", f"{entropy:.3f}")
                            
                            # Visualize gradients
                            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                            
                            # Original stress field
                            im0 = axes[0, 0].imshow(stress_data, cmap='viridis', origin='lower')
                            axes[0, 0].set_title(f'Original {stress_component}')
                            plt.colorbar(im0, ax=axes[0, 0], label='GPa')
                            
                            # Gradient magnitude
                            if gradient_type == "Magnitude":
                                grad_data = gradients[f'{stress_component}_grad_mag']
                                title = f'Gradient Magnitude ({stress_component})'
                                cmap = 'hot'
                            elif gradient_type == "X-component":
                                grad_data = gradients[f'{stress_component}_grad_x']
                                title = f'X-Gradient ({stress_component})'
                                cmap = 'RdBu_r'
                            elif gradient_type == "Y-component":
                                grad_data = gradients[f'{stress_component}_grad_y']
                                title = f'Y-Gradient ({stress_component})'
                                cmap = 'RdBu_r'
                            else:  # Direction
                                grad_x = gradients[f'{stress_component}_grad_x']
                                grad_y = gradients[f'{stress_component}_grad_y']
                                grad_data = np.arctan2(grad_y, grad_x)
                                title = f'Gradient Direction ({stress_component})'
                                cmap = 'hsv'
                            
                            im1 = axes[0, 1].imshow(grad_data, cmap=cmap, origin='lower')
                            axes[0, 1].set_title(title)
                            cbar1 = plt.colorbar(im1, ax=axes[0, 1])
                            if gradient_type == "Direction":
                                cbar1.set_label('Angle (rad)')
                            else:
                                cbar1.set_label('GPa/px')
                            
                            # Gradient histogram
                            axes[1, 0].hist(grad_data.flatten(), bins=50, alpha=0.7, edgecolor='black')
                            axes[1, 0].set_xlabel('Gradient Value' if gradient_type != "Direction" else 'Angle (rad)')
                            axes[1, 0].set_ylabel('Frequency')
                            axes[1, 0].set_title('Gradient Distribution')
                            axes[1, 0].grid(True, alpha=0.3)
                            
                            # Quiver plot (for vector visualization)
                            if gradient_type in ["X-component", "Y-component", "Magnitude"]:
                                y, x = np.mgrid[0:stress_data.shape[0]:5, 0:stress_data.shape[1]:5]
                                grad_x_subsampled = gradients[f'{stress_component}_grad_x'][::5, ::5]
                                grad_y_subsampled = gradients[f'{stress_component}_grad_y'][::5, ::5]
                                
                                axes[1, 1].quiver(x, y, grad_x_subsampled, grad_y_subsampled, 
                                                 angles='xy', scale_units='xy', scale=0.5, 
                                                 width=0.003, color='red')
                                axes[1, 1].set_xlim(0, stress_data.shape[1])
                                axes[1, 1].set_ylim(0, stress_data.shape[0])
                                axes[1, 1].set_title('Gradient Vector Field')
                                axes[1, 1].set_aspect('equal')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Gradient feature extraction
                            st.subheader("📋 Gradient Features")
                            
                            if grad_mag_key in gradients:
                                grad_mag = gradients[grad_mag_key]
                                features = {
                                    'Max Gradient Location': f"({np.unravel_index(np.argmax(grad_mag), grad_mag.shape)[1]}, "
                                                           f"{np.unravel_index(np.argmax(grad_mag), grad_mag.shape)[0]})",
                                    '90th Percentile': f"{np.percentile(grad_mag, 90):.3f} GPa/px",
                                    'Gradient Skewness': f"{stats.skew(grad_mag.flatten()):.3f}",
                                    'Gradient Kurtosis': f"{stats.kurtosis(grad_mag.flatten()):.3f}",
                                    'High Gradient Area (%)': f"{(np.sum(grad_mag > np.mean(grad_mag)) / grad_mag.size * 100):.1f}%"
                                }
                                
                                df_features = pd.DataFrame(list(features.items()), 
                                                          columns=['Feature', 'Value'])
                                st.dataframe(df_features, use_container_width=True)
                    
                    else:
                        st.warning("No history data available for selected simulation")
                        
                except Exception as e:
                    st.error(f"Error computing gradients: {str(e)}")
    
    def _display_data_explorer(self):
        """Interactive data explorer"""
        
        st.subheader("📋 Stress Data Explorer")
        
        if st.session_state.stress_summary_df.empty:
            st.info("No stress data available. Please load simulations first.")
            return
        
        # Data filtering options
        st.subheader("🔍 Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by defect type
            defect_types = st.session_state.stress_summary_df['defect_type'].unique() if 'defect_type' in st.session_state.stress_summary_df.columns else []
            selected_defects = st.multiselect(
                "Defect Types",
                defect_types,
                default=list(defect_types)
            )
        
        with col2:
            # Filter by shape
            shapes = st.session_state.stress_summary_df['shape'].unique() if 'shape' in st.session_state.stress_summary_df.columns else []
            selected_shapes = st.multiselect(
                "Shapes",
                shapes,
                default=list(shapes)
            )
        
        with col3:
            # Filter by simulation type
            sim_types = st.session_state.stress_summary_df['type'].unique() if 'type' in st.session_state.stress_summary_df.columns else []
            selected_types = st.multiselect(
                "Simulation Types",
                sim_types,
                default=list(sim_types)
            )
        
        # Numerical filters
        st.subheader("📊 Numerical Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'max_von_mises' in st.session_state.stress_summary_df.columns:
                vm_min, vm_max = st.slider(
                    "Von Mises Range (GPa)",
                    float(st.session_state.stress_summary_df['max_von_mises'].min()),
                    float(st.session_state.stress_summary_df['max_von_mises'].max()),
                    (float(st.session_state.stress_summary_df['max_von_mises'].min()),
                     float(st.session_state.stress_summary_df['max_von_mises'].max()))
                )
        
        with col2:
            if 'max_abs_hydrostatic' in st.session_state.stress_summary_df.columns:
                hydro_min, hydro_max = st.slider(
                    "Hydrostatic Range (GPa)",
                    float(st.session_state.stress_summary_df['max_abs_hydrostatic'].min()),
                    float(st.session_state.stress_summary_df['max_abs_hydrostatic'].max()),
                    (float(st.session_state.stress_summary_df['max_abs_hydrostatic'].min()),
                     float(st.session_state.stress_summary_df['max_abs_hydrostatic'].max()))
                )
        
        with col3:
            if 'eps0' in st.session_state.stress_summary_df.columns:
                eps0_min, eps0_max = st.slider(
                    "ε* Range",
                    float(st.session_state.stress_summary_df['eps0'].min()),
                    float(st.session_state.stress_summary_df['eps0'].max()),
                    (float(st.session_state.stress_summary_df['eps0'].min()),
                     float(st.session_state.stress_summary_df['eps0'].max()))
                )
        
        # Apply filters
        filtered_df = st.session_state.stress_summary_df.copy()
        
        if selected_defects:
            filtered_df = filtered_df[filtered_df['defect_type'].isin(selected_defects)]
        
        if selected_shapes:
            filtered_df = filtered_df[filtered_df['shape'].isin(selected_shapes)]
        
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        if 'max_von_mises' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['max_von_mises'] >= vm_min) & 
                (filtered_df['max_von_mises'] <= vm_max)
            ]
        
        if 'max_abs_hydrostatic' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['max_abs_hydrostatic'] >= hydro_min) & 
                (filtered_df['max_abs_hydrostatic'] <= hydro_max)
            ]
        
        if 'eps0' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['eps0'] >= eps0_min) & 
                (filtered_df['eps0'] <= eps0_max)
            ]
        
        # Display filtered data
        st.subheader(f"📋 Filtered Data ({len(filtered_df)} simulations)")
        
        # Column selection
        all_columns = list(filtered_df.columns)
        default_columns = ['id', 'type', 'defect_type', 'shape', 'max_von_mises', 
                          'max_abs_hydrostatic', 'max_stress_magnitude', 'eps0', 'kappa']
        selected_columns = st.multiselect(
            "Select columns to display",
            all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if selected_columns:
            display_df = filtered_df[selected_columns]
            
            # Pagination
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
            total_pages = max(1, len(display_df) // page_size + (1 if len(display_df) % page_size > 0 else 0))
            
            if 'page' not in st.session_state:
                st.session_state.page = 0
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Previous") and st.session_state.page > 0:
                    st.session_state.page -= 1
            with col2:
                st.write(f"Page {st.session_state.page + 1} of {total_pages}")
            with col3:
                if st.button("Next ▶") and st.session_state.page < total_pages - 1:
                    st.session_state.page += 1
            
            start_idx = st.session_state.page * page_size
            end_idx = min((st.session_state.page + 1) * page_size, len(display_df))
            
            st.dataframe(
                display_df.iloc[start_idx:end_idx].style.format({
                    col: "{:.3f}" for col in display_df.select_dtypes(include=[np.number]).columns
                }),
                use_container_width=True,
                height=400
            )
            
            # Download filtered data
            csv_buffer = BytesIO()
            display_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_buffer,
                file_name=f"filtered_stress_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Interactive statistics
        st.subheader("📊 Interactive Statistics")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_stat = st.selectbox(
                    "Select statistic to compute",
                    ["Correlation Matrix", "Pairwise Relationships", "Statistical Summary"]
                )
            
            with col2:
                if selected_stat == "Correlation Matrix":
                    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                    selected_numeric = st.multiselect(
                        "Select numeric columns",
                        numeric_cols,
                        default=numeric_cols[:min(10, len(numeric_cols))]
                    )
                    
                    if selected_numeric and len(selected_numeric) >= 2:
                        corr_matrix = filtered_df[selected_numeric].corr()
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                        ax.set_xticks(range(len(selected_numeric)))
                        ax.set_yticks(range(len(selected_numeric)))
                        ax.set_xticklabels(selected_numeric, rotation=45, ha='right')
                        ax.set_yticklabels(selected_numeric)
                        
                        # Add correlation values
                        for i in range(len(selected_numeric)):
                            for j in range(len(selected_numeric)):
                                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                              ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                        
                        plt.colorbar(im, ax=ax)
                        ax.set_title('Correlation Matrix')
                        st.pyplot(fig)
    
    def _display_comparative_analysis(self):
        """Display comparative analysis between simulations"""
        
        st.subheader("📊 Comparative Stress Analysis")
        
        if st.session_state.stress_summary_df.empty:
            st.info("No stress data available for comparative analysis.")
            return
        
        analysis_type = st.selectbox(
            "Select analysis type",
            ["Parameter vs Stress", "Stress Component Comparison", 
             "Defect Type Comparison", "Shape Comparison", "Custom Comparison"]
        )
        
        if analysis_type == "Parameter vs Stress":
            self._display_parameter_vs_stress()
        elif analysis_type == "Stress Component Comparison":
            self._display_stress_component_comparison()
        elif analysis_type == "Defect Type Comparison":
            self._display_defect_type_comparison()
        elif analysis_type == "Shape Comparison":
            self._display_shape_comparison()
        else:
            self._display_custom_comparison()
    
    def _display_parameter_vs_stress(self):
        """Display parameter vs stress relationships"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox(
                "X-axis parameter",
                ['eps0', 'kappa', 'theta_deg', 'max_abs_hydrostatic', 'max_von_mises'],
                index=0
            )
            
            y_param = st.selectbox(
                "Y-axis parameter",
                ['max_von_mises', 'max_abs_hydrostatic', 'max_stress_magnitude', 
                 'mean_von_mises', 'mean_abs_hydrostatic', 'hydro_std'],
                index=0
            )
        
        with col2:
            color_by = st.selectbox(
                "Color by",
                ['defect_type', 'shape', 'orientation', 'type', 'None'],
                index=0
            )
            
            plot_type = st.selectbox(
                "Plot type",
                ["Scatter", "Line", "Hexbin", "Regression"],
                index=0
            )
        
        if x_param in st.session_state.stress_summary_df.columns and y_param in st.session_state.stress_summary_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Scatter":
                if color_by != 'None' and color_by in st.session_state.stress_summary_df.columns:
                    groups = st.session_state.stress_summary_df[color_by].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
                    
                    for i, group in enumerate(groups):
                        group_data = st.session_state.stress_summary_df[
                            st.session_state.stress_summary_df[color_by] == group
                        ]
                        ax.scatter(group_data[x_param], group_data[y_param], 
                                  label=str(group), color=colors[i], alpha=0.7, s=50)
                    ax.legend(title=color_by)
                else:
                    ax.scatter(st.session_state.stress_summary_df[x_param], 
                              st.session_state.stress_summary_df[y_param], 
                              alpha=0.6, s=50)
            
            elif plot_type == "Line":
                sorted_df = st.session_state.stress_summary_df.sort_values(x_param)
                ax.plot(sorted_df[x_param], sorted_df[y_param], 'o-', alpha=0.7)
            
            elif plot_type == "Hexbin":
                hb = ax.hexbin(st.session_state.stress_summary_df[x_param].values,
                              st.session_state.stress_summary_df[y_param].values,
                              gridsize=30, cmap='viridis', bins='log')
                plt.colorbar(hb, ax=ax, label='Count')
            
            elif plot_type == "Regression":
                from scipy import stats
                
                x_data = st.session_state.stress_summary_df[x_param].values
                y_data = st.session_state.stress_summary_df[y_param].values
                
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_clean = x_data[mask]
                y_clean = y_data[mask]
                
                if len(x_clean) > 1:
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                    
                    ax.scatter(x_clean, y_clean, alpha=0.6, s=50)
                    
                    # Regression line
                    x_range = np.linspace(min(x_clean), max(x_clean), 100)
                    y_pred = slope * x_range + intercept
                    ax.plot(x_range, y_pred, 'r-', linewidth=2, 
                           label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}')
                    
                    ax.legend()
                    
                    # Display regression statistics
                    st.write(f"**Regression Statistics:**")
                    st.write(f"R² = {r_value**2:.3f}")
                    st.write(f"p-value = {p_value:.3e}")
                    st.write(f"Standard Error = {std_err:.3f}")
            
            ax.set_xlabel(x_param.replace('_', ' ').title())
            ax.set_ylabel(y_param.replace('_', ' ').title())
            ax.set_title(f'{y_param.replace("_", " ").title()} vs {x_param.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    def _display_stress_component_comparison(self):
        """Display comparison between different stress components"""
        
        components = ['max_von_mises', 'max_abs_hydrostatic', 'max_stress_magnitude']
        available_components = [c for c in components if c in st.session_state.stress_summary_df.columns]
        
        if len(available_components) >= 2:
            selected_components = st.multiselect(
                "Select stress components to compare",
                available_components,
                default=available_components[:min(3, len(available_components))]
            )
            
            group_by = st.selectbox(
                "Group comparison by",
                ['defect_type', 'shape', 'orientation', 'type', 'None'],
                index=0
            )
            
            if len(selected_components) >= 2:
                # Create parallel coordinates plot
                fig = plt.figure(figsize=(12, 6))
                
                if group_by != 'None' and group_by in st.session_state.stress_summary_df.columns:
                    # Grouped parallel coordinates
                    groups = st.session_state.stress_summary_df[group_by].unique()
                    
                    for group in groups:
                        group_data = st.session_state.stress_summary_df[
                            st.session_state.stress_summary_df[group_by] == group
                        ]
                        
                        # Normalize data for parallel coordinates
                        normalized_data = []
                        for comp in selected_components:
                            comp_data = group_data[comp].values
                            if len(comp_data) > 0:
                                norm_data = (comp_data - comp_data.min()) / (comp_data.max() - comp_data.min() + 1e-10)
                                normalized_data.append(norm_data)
                        
                        if len(normalized_data) == len(selected_components):
                            # Plot lines
                            for i in range(min(50, len(normalized_data[0]))):  # Limit to 50 lines
                                values = [norm_data[i] for norm_data in normalized_data]
                                plt.plot(range(len(selected_components)), values, 
                                        alpha=0.3, linewidth=1)
                    
                    plt.xticks(range(len(selected_components)), 
                              [comp.replace('_', ' ').title() for comp in selected_components])
                    plt.ylabel('Normalized Value')
                    plt.title(f'Parallel Coordinates: {group_by}')
                    plt.grid(True, alpha=0.3)
                
                else:
                    # Simple parallel coordinates without grouping
                    normalized_data = []
                    for comp in selected_components:
                        comp_data = st.session_state.stress_summary_df[comp].values
                        norm_data = (comp_data - comp_data.min()) / (comp_data.max() - comp_data.min() + 1e-10)
                        normalized_data.append(norm_data)
                    
                    # Plot lines
                    for i in range(min(100, len(normalized_data[0]))):  # Limit to 100 lines
                        values = [norm_data[i] for norm_data in normalized_data]
                        plt.plot(range(len(selected_components)), values, 
                                alpha=0.1, linewidth=0.5, color='blue')
                    
                    # Plot mean line
                    mean_values = [np.mean(norm_data) for norm_data in normalized_data]
                    plt.plot(range(len(selected_components)), mean_values, 
                            'r-', linewidth=3, label='Mean')
                    
                    plt.xticks(range(len(selected_components)), 
                              [comp.replace('_', ' ').title() for comp in selected_components])
                    plt.ylabel('Normalized Value')
                    plt.title('Parallel Coordinates: Stress Components')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                st.pyplot(fig)
    
    def _display_defect_type_comparison(self):
        """Display comparison between defect types"""
        
        if 'defect_type' in st.session_state.stress_summary_df.columns:
            defect_types = st.session_state.stress_summary_df['defect_type'].unique()
            
            if len(defect_types) > 1:
                selected_metric = st.selectbox(
                    "Select metric for comparison",
                    ['max_von_mises', 'max_abs_hydrostatic', 'max_stress_magnitude', 
                     'mean_von_mises', 'hydro_std', 'stress_ratio_vm_hydro'],
                    index=0
                )
                
                if selected_metric in st.session_state.stress_summary_df.columns:
                    # Create comparison plot
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Box plot
                    sns.boxplot(data=st.session_state.stress_summary_df, 
                               x='defect_type', y=selected_metric, ax=axes[0])
                    axes[0].set_xlabel('Defect Type')
                    axes[0].set_ylabel(selected_metric.replace('_', ' ').title())
                    axes[0].set_title(f'{selected_metric.replace("_", " ").title()} by Defect Type')
                    
                    # Violin plot
                    sns.violinplot(data=st.session_state.stress_summary_df, 
                                  x='defect_type', y=selected_metric, ax=axes[1])
                    axes[1].set_xlabel('Defect Type')
                    axes[1].set_ylabel(selected_metric.replace('_', ' ').title())
                    axes[1].set_title(f'Distribution by Defect Type')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistical comparison
                    st.subheader("📊 Statistical Comparison")
                    
                    defect_data = []
                    for defect in defect_types:
                        defect_values = st.session_state.stress_summary_df[
                            st.session_state.stress_summary_df['defect_type'] == defect
                        ][selected_metric].dropna()
                        
                        if len(defect_values) > 0:
                            defect_data.append((defect, defect_values))
                    
                    if len(defect_data) >= 2:
                        from scipy import stats
                        
                        # Run ANOVA
                        anova_data = [values for _, values in defect_data]
                        f_stat, p_value = stats.f_oneway(*anova_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("**ANOVA Test:**")
                            st.write(f"F-statistic: {f_stat:.3f}")
                            st.write(f"p-value: {p_value:.3e}")
                        
                        with col2:
                            # Pairwise t-tests
                            st.write("**Pairwise Comparisons:**")
                            pairwise_results = []
                            for i in range(len(defect_data)):
                                for j in range(i+1, len(defect_data)):
                                    defect1, values1 = defect_data[i]
                                    defect2, values2 = defect_data[j]
                                    t_stat, p_val = stats.ttest_ind(values1, values2)
                                    pairwise_results.append({
                                        'Comparison': f"{defect1} vs {defect2}",
                                        't-statistic': t_stat,
                                        'p-value': p_val
                                    })
                            
                            if pairwise_results:
                                df_pairwise = pd.DataFrame(pairwise_results)
                                st.dataframe(df_pairwise.style.format({
                                    't-statistic': '{:.3f}',
                                    'p-value': '{:.3e}'
                                }), use_container_width=True)
                        
                        with col3:
                            # Effect sizes
                            st.write("**Effect Sizes:**")
                            effect_sizes = []
                            for i in range(len(defect_data)):
                                for j in range(i+1, len(defect_data)):
                                    defect1, values1 = defect_data[i]
                                    defect2, values2 = defect_data[j]
                                    cohens_d = (np.mean(values1) - np.mean(values2)) / np.sqrt(
                                        (np.std(values1)**2 + np.std(values2)**2) / 2
                                    )
                                    effect_sizes.append({
                                        'Comparison': f"{defect1} vs {defect2}",
                                        "Cohen's d": cohens_d
                                    })
                            
                            if effect_sizes:
                                df_effects = pd.DataFrame(effect_sizes)
                                st.dataframe(df_effects.style.format({
                                    "Cohen's d": '{:.3f}'
                                }), use_container_width=True)
    
    def _display_shape_comparison(self):
        """Display comparison between shapes"""
        
        if 'shape' in st.session_state.stress_summary_df.columns:
            shapes = st.session_state.stress_summary_df['shape'].unique()
            
            if len(shapes) > 1:
                selected_metric = st.selectbox(
                    "Select metric for shape comparison",
                    ['max_von_mises', 'max_abs_hydrostatic', 'max_stress_magnitude', 
                     'mean_von_mises', 'hydro_std'],
                    index=0
                )
                
                if selected_metric in st.session_state.stress_summary_df.columns:
                    # Create radar chart
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                    
                    # Calculate statistics for each shape
                    shape_stats = []
                    for shape in shapes:
                        shape_data = st.session_state.stress_summary_df[
                            st.session_state.stress_summary_df['shape'] == shape
                        ][selected_metric].dropna()
                        
                        if len(shape_data) > 0:
                            shape_stats.append({
                                'shape': shape,
                                'mean': np.mean(shape_data),
                                'std': np.std(shape_data),
                                'count': len(shape_data)
                            })
                    
                    if shape_stats:
                        # Prepare data for radar chart
                        angles = np.linspace(0, 2 * np.pi, len(shape_stats), endpoint=False).tolist()
                        values = [stat['mean'] for stat in shape_stats]
                        stds = [stat['std'] for stat in shape_stats]
                        
                        # Complete the loop
                        values += values[:1]
                        stds += stds[:1]
                        angles += angles[:1]
                        
                        # Plot
                        ax.plot(angles, values, 'o-', linewidth=2, label='Mean')
                        ax.fill(angles, values, alpha=0.25)
                        
                        # Add error bars
                        ax.errorbar(angles[:-1], values[:-1], yerr=stds[:-1], 
                                   fmt='o', capsize=5, alpha=0.5)
                        
                        # Set labels
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels([stat['shape'] for stat in shape_stats])
                        ax.set_title(f'{selected_metric.replace("_", " ").title()} by Shape')
                        ax.grid(True)
                        
                        st.pyplot(fig)
    
    def _display_custom_comparison(self):
        """Display custom comparison interface"""
        
        st.info("Create custom comparisons by selecting specific simulations")
        
        # Select simulations to compare
        simulation_ids = st.session_state.stress_summary_df['id'].tolist()
        selected_ids = st.multiselect(
            "Select simulations to compare",
            simulation_ids,
            default=simulation_ids[:min(5, len(simulation_ids))]
        )
        
        if len(selected_ids) >= 2:
            comparison_df = st.session_state.stress_summary_df[
                st.session_state.stress_summary_df['id'].isin(selected_ids)
            ]
            
            # Select metrics for comparison
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_metrics = st.multiselect(
                "Select metrics to compare",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_metrics:
                # Create comparison table
                st.dataframe(
                    comparison_df[['id', 'defect_type', 'shape'] + selected_metrics].style.format({
                        col: "{:.3f}" for col in selected_metrics
                    }),
                    use_container_width=True
                )
                
                # Create spider/radar chart for comparison
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
                
                angles = np.linspace(0, 2 * np.pi, len(selected_metrics), endpoint=False).tolist()
                
                for idx, row in comparison_df.iterrows():
                    values = [row[metric] for metric in selected_metrics]
                    
                    # Normalize values for radar chart
                    max_vals = comparison_df[selected_metrics].max().values
                    min_vals = comparison_df[selected_metrics].min().values
                    norm_values = [(v - min_v) / (max_v - min_v + 1e-10) 
                                  for v, min_v, max_v in zip(values, min_vals, max_vals)]
                    
                    # Complete the loop
                    norm_values += norm_values[:1]
                    plot_angles = angles + angles[:1]
                    
                    ax.plot(plot_angles, norm_values, 'o-', linewidth=2, 
                           label=f"{row['id']} ({row['defect_type']})")
                    ax.fill(plot_angles, norm_values, alpha=0.1)
                
                ax.set_xticks(angles)
                ax.set_xticklabels([metric.replace('_', ' ').title() for metric in selected_metrics])
                ax.set_title('Normalized Comparison of Selected Simulations')
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
                
                st.pyplot(fig)
    
    def _display_time_series_analysis(self):
        """Display time series analysis for simulations with history"""
        
        st.subheader("📈 Time Series Analysis")
        
        if not st.session_state.source_simulations:
            st.info("No simulations with time history available.")
            return
        
        # Select simulation with history
        simulations_with_history = []
        for i, sim in enumerate(st.session_state.source_simulations):
            if 'history' in sim and len(sim['history']) > 1:
                simulations_with_history.append(i)
        
        if not simulations_with_history:
            st.info("No simulations with multiple time steps available.")
            return
        
        simulation_idx = st.selectbox(
            "Select simulation with time history",
            simulations_with_history,
            format_func=lambda x: f"Simulation {x+1}",
            index=0
        )
        
        sim_data = st.session_state.source_simulations[simulation_idx]
        history = sim_data.get('history', [])
        
        if len(history) < 2:
            st.warning("Selected simulation doesn't have enough time steps.")
            return
        
        # Extract time series data
        time_points = list(range(len(history)))
        
        stress_metrics_over_time = {
            'Max Von Mises': [],
            'Mean Von Mises': [],
            'Max Hydrostatic': [],
            'Mean Hydrostatic': [],
            'Stress Magnitude': []
        }
        
        for frame_idx, (eta, stress_fields) in enumerate(history):
            if 'von_mises' in stress_fields:
                vm_data = stress_fields['von_mises']
                stress_metrics_over_time['Max Von Mises'].append(np.max(vm_data))
                stress_metrics_over_time['Mean Von Mises'].append(np.mean(vm_data))
            
            if 'sigma_hydro' in stress_fields:
                hydro_data = stress_fields['sigma_hydro']
                stress_metrics_over_time['Max Hydrostatic'].append(np.max(np.abs(hydro_data)))
                stress_metrics_over_time['Mean Hydrostatic'].append(np.mean(np.abs(hydro_data)))
            
            if 'sigma_mag' in stress_fields:
                mag_data = stress_fields['sigma_mag']
                stress_metrics_over_time['Stress Magnitude'].append(np.max(mag_data))
        
        # Plot time series
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            list(stress_metrics_over_time.keys()),
            default=list(stress_metrics_over_time.keys())[:3]
        )
        
        if selected_metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for metric in selected_metrics:
                if metric in stress_metrics_over_time and stress_metrics_over_time[metric]:
                    ax.plot(time_points, stress_metrics_over_time[metric], 
                           'o-', label=metric, linewidth=2, markersize=4)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Stress (GPa)')
            ax.set_title('Stress Evolution Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Time series analysis
            st.subheader("📊 Time Series Analysis")
            
            for metric in selected_metrics:
                if metric in stress_metrics_over_time and len(stress_metrics_over_time[metric]) > 1:
                    data = stress_metrics_over_time[metric]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{metric} Start", f"{data[0]:.3f} GPa")
                    with col2:
                        st.metric(f"{metric} End", f"{data[-1]:.3f} GPa")
                    with col3:
                        change = ((data[-1] - data[0]) / (abs(data[0]) + 1e-10)) * 100
                        st.metric(f"{metric} Change", f"{change:.1f}%")
        
        # Additional time series features
        st.subheader("🔄 Time Series Features")
        
        if st.button("Analyze Time Series Patterns"):
            # Calculate time series features
            features = {}
            
            for metric, data in stress_metrics_over_time.items():
                if len(data) > 1:
                    # Rate of change
                    rates = np.diff(data)
                    features[f'{metric} - Max Rate'] = np.max(np.abs(rates))
                    features[f'{metric} - Mean Rate'] = np.mean(np.abs(rates))
                    
                    # Stationarity test (simplified)
                    from scipy import stats
                    if len(data) > 10:
                        # Split into two halves
                        split = len(data) // 2
                        half1 = data[:split]
                        half2 = data[split:]
                        
                        # Compare means
                        t_stat, p_val = stats.ttest_ind(half1, half2)
                        features[f'{metric} - Stationarity p'] = p_val
                    
                    # Autocorrelation (lag 1)
                    if len(data) > 2:
                        correlation = np.corrcoef(data[:-1], data[1:])[0, 1]
                        features[f'{metric} - Autocorrelation'] = correlation
            
            if features:
                df_features = pd.DataFrame(list(features.items()), 
                                          columns=['Feature', 'Value'])
                st.dataframe(df_features.style.format({'Value': '{:.3f}'}), 
                           use_container_width=True)
    
    def _display_export_section(self):
        """Display export and reporting section"""
        
        st.subheader("📤 Export & Report Generation")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input("Report Title", 
                                        "Stress Analysis Report")
            
            report_author = st.text_input("Author", 
                                         "Stress Analysis Dashboard")
            
            include_sections = st.multiselect(
                "Include Sections",
                ["Summary Statistics", "Distribution Analysis", "Peak Analysis", 
                 "Gradient Analysis", "Comparative Analysis", "Time Series", 
                 "Visualizations", "Recommendations"],
                default=["Summary Statistics", "Distribution Analysis", "Visualizations"]
            )
        
        with col2:
            report_format = st.selectbox(
                "Report Format",
                ["HTML", "PDF", "Markdown", "Word"],
                index=0
            )
            
            export_data = st.checkbox("Include raw data", True)
            export_plots = st.checkbox("Include plots", True)
            export_statistics = st.checkbox("Include statistics", True)
        
        # Generate report
        if st.button("📄 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    report_content = self._generate_report_content(
                        report_title, report_author, include_sections,
                        export_data, export_plots, export_statistics
                    )
                    
                    # Create download button based on format
                    if report_format == "HTML":
                        self._export_html_report(report_content, report_title)
                    elif report_format == "PDF":
                        self._export_pdf_report(report_content, report_title)
                    elif report_format == "Markdown":
                        self._export_markdown_report(report_content, report_title)
                    elif report_format == "Word":
                        self._export_word_report(report_content, report_title)
                    
                    st.success("✅ Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Data export options
        st.subheader("📊 Data Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Summary Data (CSV)"):
                if not st.session_state.stress_summary_df.empty:
                    csv_buffer = BytesIO()
                    st.session_state.stress_summary_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv_buffer,
                        file_name=f"stress_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📊 Export Visualizations (ZIP)"):
                self._export_visualizations_zip()
        
        with col3:
            if st.button("📋 Export Configuration (JSON)"):
                self._export_configuration_json()
        
        # Advanced export options
        with st.expander("⚙️ Advanced Export Options"):
            # Custom data transformation
            st.write("**Custom Data Transformation**")
            
            transform_option = st.selectbox(
                "Apply transformation",
                ["None", "Normalize", "Standardize", "Log Transform", "Power Transform"],
                index=0
            )
            
            if transform_option != "None" and not st.session_state.stress_summary_df.empty:
                if st.button("Apply Transformation"):
                    transformed_df = self._apply_transformation(
                        st.session_state.stress_summary_df.copy(), 
                        transform_option
                    )
                    
                    csv_buffer = BytesIO()
                    transformed_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label=f"Download {transform_option} Data",
                        data=csv_buffer,
                        file_name=f"stress_data_{transform_option}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    def _generate_report_content(self, title, author, sections, 
                               include_data=True, include_plots=True, 
                               include_stats=True):
        """Generate report content"""
        
        report = f"""
# {title}
**Author:** {author}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Simulations:** {len(st.session_state.stress_summary_df) if not st.session_state.stress_summary_df.empty else 0}

---
"""
        
        if "Summary Statistics" in sections:
            report += "\n## Summary Statistics\n"
            if not st.session_state.stress_summary_df.empty:
                report += "### Key Metrics\n"
                
                if 'max_von_mises' in st.session_state.stress_summary_df.columns:
                    vm_mean = st.session_state.stress_summary_df['max_von_mises'].mean()
                    vm_std = st.session_state.stress_summary_df['max_von_mises'].std()
                    vm_min = st.session_state.stress_summary_df['max_von_mises'].min()
                    vm_max = st.session_state.stress_summary_df['max_von_mises'].max()
                    
                    report += f"- **Von Mises Stress:** {vm_mean:.2f} ± {vm_std:.2f} GPa (Range: {vm_min:.2f} - {vm_max:.2f} GPa)\n"
                
                if 'max_abs_hydrostatic' in st.session_state.stress_summary_df.columns:
                    hydro_mean = st.session_state.stress_summary_df['max_abs_hydrostatic'].mean()
                    hydro_std = st.session_state.stress_summary_df['max_abs_hydrostatic'].std()
                    report += f"- **Hydrostatic Stress:** {hydro_mean:.2f} ± {hydro_std:.2f} GPa\n"
        
        if "Distribution Analysis" in sections:
            report += "\n## Distribution Analysis\n"
            report += "Stress distributions were analyzed across all simulations.\n"
        
        if "Recommendations" in sections:
            report += "\n## Recommendations\n"
            report += "Based on the analysis:\n"
            report += "1. Monitor high stress concentration areas\n"
            report += "2. Consider stress relief strategies for critical components\n"
            report += "3. Validate predictions with additional simulations\n"
        
        return report
    
    def _export_html_report(self, content, title):
        """Export report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                .metric {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
            </style>
        </head>
        <body>
            {content.replace('#', '<h1>').replace('##', '<h2>').replace('###', '<h3>')
                     .replace('- ', '<li>').replace('\n', '<br>')}
        </body>
        </html>
        """
        
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{title}.html">Download HTML Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def _export_pdf_report(self, content, title):
        """Export report as PDF (simplified - in production, use a proper PDF library)"""
        st.info("PDF export requires additional libraries. Exporting as HTML instead.")
        self._export_html_report(content, title)
    
    def _export_markdown_report(self, content, title):
        """Export report as Markdown"""
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{title}.md">Download Markdown Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def _export_word_report(self, content, title):
        """Export report as Word document (simplified)"""
        st.info("Word export requires additional libraries. Exporting as HTML instead.")
        self._export_html_report(content, title)
    
    def _export_visualizations_zip(self):
        """Export visualizations as ZIP file"""
        import zipfile
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add summary statistics plot
            if not st.session_state.stress_summary_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                if 'max_von_mises' in st.session_state.stress_summary_df.columns:
                    ax.hist(st.session_state.stress_summary_df['max_von_mises'].dropna(), 
                           bins=30, alpha=0.7)
                    ax.set_xlabel('Max Von Mises (GPa)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Max Von Mises Stress')
                
                # Save to buffer
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                zip_file.writestr('von_mises_distribution.png', img_buffer.read())
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Visualizations ZIP",
            data=zip_buffer,
            file_name=f"stress_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
    
    def _export_configuration_json(self):
        """Export configuration as JSON"""
        config = {
            'dashboard_config': {
                'timestamp': datetime.now().isoformat(),
                'total_simulations': len(st.session_state.stress_summary_df) if not st.session_state.stress_summary_df.empty else 0,
                'metrics_available': list(st.session_state.stress_summary_df.columns) if not st.session_state.stress_summary_df.empty else []
            }
        }
        
        json_buffer = BytesIO()
        json_buffer.write(json.dumps(config, indent=2).encode())
        json_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Configuration JSON",
            data=json_buffer,
            file_name=f"stress_dashboard_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _apply_transformation(self, df, transform_type):
        """Apply data transformation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if transform_type == "Normalize":
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif transform_type == "Standardize":
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        elif transform_type == "Log Transform":
            for col in numeric_cols:
                # Add small constant to avoid log(0)
                df[col] = np.log(df[col] - df[col].min() + 1)
        
        elif transform_type == "Power Transform":
            from scipy import stats
            for col in numeric_cols:
                df[col], _ = stats.boxcox(df[col] - df[col].min() + 1)
        
        return df


# =============================================
# UPDATE MAIN FUNCTION FOR STRESS DASHBOARD
# =============================================
def main():
    """Main application with enhanced stress dashboard"""
    
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
    
    operation_mode = st.sidebar.radio(
        "Select Mode",
        ["Attention Interpolation", "Comprehensive Stress Dashboard", "Stress Analysis Dashboard"],
        index=0
    )
    
    if operation_mode == "Attention Interpolation":
        create_attention_interface()
    elif operation_mode == "Comprehensive Stress Dashboard":
        # Initialize the comprehensive stress dashboard
        if 'stress_dashboard' not in st.session_state:
            st.session_state.stress_dashboard = ComprehensiveStressDashboard()
        
        # Initialize required components
        if 'stress_analyzer' not in st.session_state:
            st.session_state.stress_analyzer = StressAnalysisManager()
        
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
        
        # Load data if needed
        if st.session_state.stress_summary_df.empty:
            st.info("📥 Loading simulations for dashboard...")
            
            # Quick load button
            if st.button("Load Available Simulations"):
                with st.spinner("Loading simulations..."):
                    all_files = st.session_state.solutions_manager.get_all_files()
                    all_simulations = []
                    
                    for file_info in all_files[:20]:  # Limit to 20 for performance
                        try:
                            sim_data = st.session_state.solutions_manager.load_simulation(
                                file_info['path'],
                                st.session_state.interpolator
                            )
                            all_simulations.append(sim_data)
                        except:
                            continue
                    
                    if all_simulations:
                        stress_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                            all_simulations, {}
                        )
                        
                        if not stress_df.empty:
                            st.session_state.stress_summary_df = stress_df
                            st.success(f"✅ Loaded {len(all_simulations)} simulations")
                            st.rerun()
                    else:
                        st.error("No simulations could be loaded")
        
        # Create the dashboard
        st.session_state.stress_dashboard.create_dashboard_layout()
        
    else:  # Original Stress Analysis Dashboard
        st.header("📊 Stress Analysis Dashboard")
        
        # Initialize managers
        if 'stress_analyzer' not in st.session_state:
            st.session_state.stress_analyzer = StressAnalysisManager()
        if 'sunburst_manager' not in st.session_state:
            st.session_state.sunburst_manager = SunburstChartManager()
        
        if 'solutions_manager' not in st.session_state:
            st.session_state.solutions_manager = NumericalSolutionsManager(NUMERICAL_SOLUTIONS_DIR)
        
        if 'interpolator' not in st.session_state:
            st.session_state.interpolator = SpatialLocalityAttentionInterpolator()
        
        all_files = st.session_state.solutions_manager.get_all_files()
        
        if st.button("📥 Load All Simulations for Analysis"):
            with st.spinner("Loading all simulations..."):
                all_simulations = []
                for file_info in all_files[:50]:
                    try:
                        sim_data = st.session_state.solutions_manager.load_simulation(
                            file_info['path'],
                            st.session_state.interpolator
                        )
                        all_simulations.append(sim_data)
                    except:
                        continue
                
                if all_simulations:
                    stress_df = st.session_state.stress_analyzer.create_stress_summary_dataframe(
                        all_simulations, {}
                    )
                    
                    if not stress_df.empty:
                        st.session_state.stress_summary_df = stress_df
                        st.success(f"✅ Loaded {len(all_simulations)} simulations for analysis")
                    else:
                        st.warning("No stress data found in loaded simulations")
                else:
                    st.error("No simulations could be loaded")
        
        if not st.session_state.stress_summary_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Simulations", len(st.session_state.stress_summary_df))
            with col2:
                max_vm = st.session_state.stress_summary_df['max_von_mises'].max() if 'max_von_mises' in st.session_state.stress_summary_df.columns else 0
                st.metric("Max Von Mises", f"{max_vm:.2f} GPa")
            with col3:
                mean_vm = st.session_state.stress_summary_df['max_von_mises'].mean() if 'max_von_mises' in st.session_state.stress_summary_df.columns else 0
                st.metric("Avg Max Von Mises", f"{mean_vm:.2f} GPa")
            with col4:
                defect_counts = st.session_state.stress_summary_df['defect_type'].value_counts().to_dict() if 'defect_type' in st.session_state.stress_summary_df.columns else {}
                st.metric("Unique Defect Types", len(defect_counts))
            
            # Show the stress analysis tab interface
            create_attention_interface()
        else:
            st.info("Please load simulations first to enable the stress analysis dashboard")
