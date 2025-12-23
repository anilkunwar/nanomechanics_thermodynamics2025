# =============================================
# SINTERING TEMPERATURE CALCULATOR
# =============================================

class SinteringTemperatureCalculator:
    """Calculate sintering temperature based on hydrostatic stress at habit plane"""
    
    def __init__(self, T0=623.0, beta=0.95, G=30.0, sigma_peak=28.5):
        self.T0 = T0  # Reference temperature at zero stress (K)
        self.beta = beta  # Calibration factor
        self.G = G  # Shear modulus of Ag (GPa)
        self.sigma_peak = sigma_peak  # Peak hydrostatic stress (GPa)
        self.T_min = 367.0  # Minimum sintering temperature at peak stress (K)
        
        # Material properties for Ag
        self.kB = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.Q_a = 1.1  # Activation energy for Ag diffusion (eV)
        self.omega = 0.85 * (0.408e-9)**3  # Activation volume (m³)
        self.omega_eV_per_GPa = self.omega * 6.242e18  # Convert to eV/GPa
        
    def compute_sintering_temperature_exponential(self, sigma_h):
        """Compute sintering temperature using exponential empirical model"""
        sigma_abs = np.abs(sigma_h)
        T_sinter = self.T0 * np.exp(-self.beta * sigma_abs / self.G)
        return T_sinter
    
    def compute_sintering_temperature_arrhenius(self, sigma_h, D0=1e-6, D_crit=1e-10):
        """Compute sintering temperature using stress-modified Arrhenius equation"""
        sigma_abs = np.abs(sigma_h)
        Q_eff = self.Q_a - self.omega_eV_per_GPa * sigma_abs
        T_sinter = Q_eff / (self.kB * np.log(D0 / D_crit))
        return T_sinter
    
    def compute_stress_for_temperature(self, T_sinter):
        """Compute required hydrostatic stress to achieve given sintering temperature"""
        if T_sinter <= 0:
            return 0.0
        sigma_h = -(self.G / self.beta) * np.log(T_sinter / self.T0)
        return sigma_h
    
    def compute_peak_stress_from_temperature(self, T_min=None):
        """Compute peak hydrostatic stress from minimum sintering temperature"""
        if T_min is None:
            T_min = self.T_min
        sigma_peak = -(self.G / self.beta) * np.log(T_min / self.T0)
        return sigma_peak
    
    def map_system_to_temperature(self, sigma_h):
        """Map hydrostatic stress to system classification"""
        sigma_abs = np.abs(sigma_h)
        
        if sigma_abs < 5.0:
            system = "System 1 (Perfect Crystal)"
            T_range = (620, 630)  # K
        elif sigma_abs < 20.0:
            system = "System 2 (Stacking Faults/Twins)"
            T_range = (450, 550)  # K
        else:
            system = "System 3 (Plastic Deformation)"
            T_range = (350, 400)  # K
            
        T_sinter = self.compute_sintering_temperature_exponential(sigma_abs)
        return system, T_range, T_sinter
    
    def get_theoretical_curve(self, max_stress=35.0, n_points=100):
        """Generate theoretical curve of T_sinter vs |σ_h|"""
        stresses = np.linspace(0, max_stress, n_points)
        T_exp = self.compute_sintering_temperature_exponential(stresses)
        T_arr = self.compute_sintering_temperature_arrhenius(stresses)
        
        return {
            'stresses': stresses,
            'T_exponential': T_exp,
            'T_arrhenius': T_arr,
            'T0': self.T0,
            'T_min': self.T_min,
            'sigma_peak': self.sigma_peak
        }
    
    def create_sintering_plot(self, stresses, temperatures, title="Sintering Temperature vs Hydrostatic Stress"):
        """Create detailed sintering temperature plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Main curve
        ax.plot(stresses, temperatures, 'b-', linewidth=3, label='Empirical Model')
        
        # System boundaries
        ax.axvspan(0, 5, alpha=0.1, color='green', label='System 1 (Perfect)')
        ax.axvspan(5, 20, alpha=0.1, color='orange', label='System 2 (SF/Twin)')
        ax.axvspan(20, 35, alpha=0.1, color='red', label='System 3 (Plastic)')
        
        # Reference points
        ax.plot(0, self.T0, 'go', markersize=12, label=f'System 1: {self.T0}K at 0 GPa')
        ax.plot(12.5, self.compute_sintering_temperature_exponential(12.5), 'yo', markersize=12, 
                label=f'System 2: ~475K at 12.5 GPa')
        ax.plot(self.sigma_peak, self.T_min, 'ro', markersize=12, 
                label=f'System 3: {self.T_min}K at {self.sigma_peak:.1f} GPa')
        
        # Lines for habit plane reference
        ax.axhline(self.T0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(self.T_min, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(self.sigma_peak, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Absolute Hydrostatic Stress |σ_h| (GPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sintering Temperature (K)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add second y-axis for Celsius
        ax2 = ax.twinx()
        celsius_ticks = ax.get_yticks()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticklabels([f'{t-273.15:.0f}°C' for t in celsius_ticks])
        ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        
        # Add annotations
        ax.text(0.02, 0.98, f'T₀ = {self.T0} K ({self.T0-273.15:.0f}°C) at σ_h = 0',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
        
        ax.text(0.02, 0.90, f'T_min = {self.T_min} K ({self.T_min-273.15:.0f}°C) at σ_h = {self.sigma_peak:.1f} GPa',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        
        return fig

# =============================================
# ENHANCED VISUALIZATION WITH SINTERING SUPPORT
# =============================================

class EnhancedSinteringVisualizer:
    """Enhanced visualizer for sintering temperature analysis"""
    
    def __init__(self):
        self.sintering_calculator = SinteringTemperatureCalculator()
    
    def create_comprehensive_sintering_dashboard(self, solutions, region_type='bulk',
                                                stress_component='sigma_hydro',
                                                stress_type='max_abs'):
        """Create comprehensive dashboard for sintering temperature analysis"""
        
        # Analyze all solutions
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(solutions, region_type, 
                                                 stress_component, stress_type)
        
        if not analyses:
            return None
        
        # Extract stresses and compute sintering temperatures
        stresses = []
        sintering_temps = []
        orientations = []
        systems = []
        
        for analysis in analyses:
            stress = analysis['region_stress']
            T_sinter = self.sintering_calculator.compute_sintering_temperature_exponential(abs(stress))
            system_info = self.sintering_calculator.map_system_to_temperature(stress)
            
            stresses.append(abs(stress))
            sintering_temps.append(T_sinter)
            orientations.append(analysis['theta_deg'])
            systems.append(system_info[0])
        
        # Create dashboard figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Sintering temperature vs stress
        ax1 = fig.add_subplot(2, 3, 1)
        scatter = ax1.scatter(stresses, sintering_temps, c=orientations, 
                             cmap='hsv', s=50, alpha=0.7, edgecolors='black')
        
        # Add theoretical curve
        theory_data = self.sintering_calculator.get_theoretical_curve()
        ax1.plot(theory_data['stresses'], theory_data['T_exponential'], 
                'k--', alpha=0.5, label='Theoretical')
        
        ax1.set_xlabel('|σ_h| (GPa)', fontsize=10)
        ax1.set_ylabel('T_sinter (K)', fontsize=10)
        ax1.set_title('Sintering Temperature vs Hydrostatic Stress', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for orientation
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Orientation (°)', fontsize=9)
        
        # 2. Histogram of sintering temperatures
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(sintering_temps, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(self.sintering_calculator.T0, color='green', linestyle='--', 
                   label=f'T₀ = {self.sintering_calculator.T0}K')
        ax2.axvline(self.sintering_calculator.T_min, color='red', linestyle='--',
                   label=f'T_min = {self.sintering_calculator.T_min}K')
        ax2.set_xlabel('Sintering Temperature (K)', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Distribution of Sintering Temperatures', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. System classification
        ax3 = fig.add_subplot(2, 3, 3)
        system_counts = {}
        for system in systems:
            system_counts[system] = system_counts.get(system, 0) + 1
        
        colors = {'System 1': 'green', 'System 2': 'orange', 'System 3': 'red'}
        bar_colors = [colors.get(sys.split()[0], 'gray') for sys in system_counts.keys()]
        
        ax3.bar(range(len(system_counts)), list(system_counts.values()), 
               color=bar_colors, edgecolor='black', alpha=0.7)
        ax3.set_xticks(range(len(system_counts)))
        ax3.set_xticklabels(list(system_counts.keys()), rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Number of Solutions', fontsize=10)
        ax3.set_title('System Classification Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Orientation vs sintering temperature
        ax4 = fig.add_subplot(2, 3, 4)
        scatter2 = ax4.scatter(orientations, sintering_temps, c=stresses, 
                              cmap='plasma', s=50, alpha=0.7, edgecolors='black')
        ax4.axvline(54.7, color='green', linestyle='--', alpha=0.5, label='Habit Plane (54.7°)')
        ax4.set_xlabel('Orientation (°)', fontsize=10)
        ax4.set_ylabel('T_sinter (K)', fontsize=10)
        ax4.set_title('Sintering Temperature vs Orientation', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        
        cbar2 = plt.colorbar(scatter2, ax=ax4)
        cbar2.set_label('|σ_h| (GPa)', fontsize=9)
        
        # 5. Temperature reduction factor
        ax5 = fig.add_subplot(2, 3, 5)
        temp_reduction = [(self.sintering_calculator.T0 - T) / self.sintering_calculator.T0 * 100 
                         for T in sintering_temps]
        ax5.scatter(stresses, temp_reduction, c='purple', s=50, alpha=0.7, edgecolors='black')
        ax5.set_xlabel('|σ_h| (GPa)', fontsize=10)
        ax5.set_ylabel('Temperature Reduction (%)', fontsize=10)
        ax5.set_title('Stress-Induced Temperature Reduction', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics table
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        stats_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Number of Solutions', f'{len(analyses)}', ''],
            ['Mean |σ_h|', f'{np.mean(stresses):.2f}', 'GPa'],
            ['Max |σ_h|', f'{np.max(stresses):.2f}', 'GPa'],
            ['Mean T_sinter', f'{np.mean(sintering_temps):.1f}', 'K'],
            ['Min T_sinter', f'{np.min(sintering_temps):.1f}', 'K'],
            ['Max T_sinter', f'{np.max(sintering_temps):.1f}', 'K'],
            ['T₀ (reference)', f'{self.sintering_calculator.T0}', 'K'],
            ['T_min (peak)', f'{self.sintering_calculator.T_min}', 'K'],
            ['Temperature Range', f'{np.min(sintering_temps):.0f}-{np.max(sintering_temps):.0f}', 'K'],
            ['Mean Reduction', f'{np.mean(temp_reduction):.1f}', '%']
        ]
        
        table = ax6.table(cellText=stats_data, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style table
        for i in range(len(stats_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(color='white')
                elif i % 2 == 1:
                    cell.set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_sintering_plot(self, solutions, region_type='bulk',
                                         stress_component='sigma_hydro',
                                         stress_type='max_abs'):
        """Create interactive Plotly visualization for sintering analysis"""
        
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(solutions, region_type,
                                                 stress_component, stress_type)
        
        if not analyses:
            return None
        
        # Prepare data
        stresses = []
        sintering_temps = []
        orientations = []
        filenames = []
        systems = []
        colors = []
        
        for analysis in analyses:
            stress = abs(analysis['region_stress'])
            T_sinter = self.sintering_calculator.compute_sintering_temperature_exponential(stress)
            system_info = self.sintering_calculator.map_system_to_temperature(stress)
            
            stresses.append(stress)
            sintering_temps.append(T_sinter)
            orientations.append(analysis['theta_deg'])
            filenames.append(analysis['filename'])
            systems.append(system_info[0])
            
            # Assign colors based on system
            if 'System 1' in system_info[0]:
                colors.append('green')
            elif 'System 2' in system_info[0]:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=stresses,
            y=sintering_temps,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=[f"File: {f}<br>Orientation: {o:.1f}°<br>System: {s}<br>Stress: {σ:.2f} GPa<br>T_sinter: {T:.1f} K ({T-273.15:.0f}°C)"
                  for f, o, s, σ, T in zip(filenames, orientations, systems, stresses, sintering_temps)],
            hoverinfo='text',
            name='Solutions'
        ))
        
        # Add theoretical curve
        theory_data = self.sintering_calculator.get_theoretical_curve()
        fig.add_trace(go.Scatter(
            x=theory_data['stresses'],
            y=theory_data['T_exponential'],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Theoretical Curve'
        ))
        
        # Add system boundaries
        fig.add_vrect(x0=0, x1=5, fillcolor="green", opacity=0.1, line_width=0,
                     annotation_text="System 1", annotation_position="top left")
        fig.add_vrect(x0=5, x1=20, fillcolor="orange", opacity=0.1, line_width=0,
                     annotation_text="System 2", annotation_position="top left")
        fig.add_vrect(x0=20, x1=35, fillcolor="red", opacity=0.1, line_width=0,
                     annotation_text="System 3", annotation_position="top left")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Sintering Temperature Analysis - {region_type}",
                font=dict(size=20, family="Arial Black", color='darkblue'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text='Absolute Hydrostatic Stress |σ_h| (GPa)', font=dict(size=14)),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1
            ),
            yaxis=dict(
                title=dict(text='Sintering Temperature (K)', font=dict(size=14)),
                gridcolor='rgba(100, 100, 100, 0.2)',
                gridwidth=1
            ),
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1
            ),
            width=1000,
            height=600
        )
        
        # Add second y-axis for Celsius
        fig.update_layout(
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying="y",
                side="right",
                tickmode='sync',
                tickvals=fig.data[0].y,
                ticktext=[f"{t-273.15:.0f}" for t in fig.data[0].y],
                range=[min(sintering_temps)-273.15, max(sintering_temps)-273.15]
            )
        )
        
        return fig
    
    def create_sintering_temperature_sweep(self, solutions, base_params, angle_range,
                                          region_type='bulk', stress_component='sigma_hydro',
                                          stress_type='max_abs', n_points=100):
        """Create sintering temperature sweep across orientation range"""
        
        interpolator = AttentionSpatialInterpolator()
        
        # Get stress sweep
        sweep_result = interpolator.create_orientation_sweep(
            solutions, base_params, angle_range, n_points,
            region_type, stress_component, stress_type
        )
        
        if not sweep_result:
            return None
        
        # Compute sintering temperatures
        stresses = np.array(sweep_result['stresses'])
        sintering_temps = self.sintering_calculator.compute_sintering_temperature_exponential(np.abs(stresses))
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot stress
        ax1.plot(sweep_result['angles'], stresses, 'b-', linewidth=3, label='Hydrostatic Stress')
        ax1.set_xlabel('Orientation (°)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Hydrostatic Stress (GPa)', fontsize=12, fontweight='bold', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        # Plot sintering temperature on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(sweep_result['angles'], sintering_temps, 'r-', linewidth=3, label='Sintering Temperature')
        ax2.set_ylabel('Sintering Temperature (K)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Highlight habit plane
        ax1.axvline(54.7, color='green', linestyle='--', linewidth=2, 
                   label='Habit Plane (54.7°)', alpha=0.7)
        
        # Add Celsius scale
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(sweep_result['angles'], [t-273.15 for t in sintering_temps], 
                'r--', alpha=0.5, linewidth=2, label='Sintering Temp (°C)')
        ax3.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold', color='darkred')
        ax3.tick_params(axis='y', labelcolor='darkred')
        
        # Title and legend
        ax1.set_title(f'Sintering Temperature Sweep: {region_type}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                  loc='upper left', fontsize=10)
        
        # Add statistics box
        stats_text = f"""Statistics:
        Min Stress: {np.min(stresses):.2f} GPa
        Max Stress: {np.max(stresses):.2f} GPa
        Min T_sinter: {np.min(sintering_temps):.1f} K ({np.min(sintering_temps)-273.15:.0f}°C)
        Max T_sinter: {np.max(sintering_temps):.1f} K ({np.max(sintering_temps)-273.15:.0f}°C)
        Habit Plane T: {sintering_temps[np.argmin(np.abs(np.array(sweep_result['angles'])-54.7))]:.1f} K
        """
        
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

# =============================================
# UPDATE MAIN APPLICATION WITH SINTERING ANALYSIS
# =============================================

def main():
    # ... [Previous main() code remains the same until analysis mode selection]
    
    # Add sintering analysis to the analysis mode options
    analysis_mode = st.radio(
        "Select analysis mode:",
        ["Precise Single Orientation", "Orientation Sweep", 
         "Compare Original vs Interpolated", "Heatmap Analysis",
         "Sintering Temperature Analysis"],  # Added new mode
        index=0,
        help="Choose analysis mode including sintering temperature prediction"
    )
    
    # ... [Previous sidebar code remains the same]
    
    # Add sintering-specific sidebar controls
    if analysis_mode == "Sintering Temperature Analysis":
        st.markdown("#### 🔥 Sintering Temperature Parameters")
        
        col_sint1, col_sint2 = st.columns(2)
        with col_sint1:
            T0 = st.number_input(
                "T₀ (reference temperature at σ=0)", 
                min_value=300.0, max_value=1000.0, value=623.0, step=1.0,
                help="Reference sintering temperature at zero stress"
            )
            T_min = st.number_input(
                "T_min (minimum temperature at peak stress)",
                min_value=300.0, max_value=1000.0, value=367.0, step=1.0,
                help="Minimum sintering temperature at peak hydrostatic stress"
            )
        
        with col_sint2:
            beta = st.number_input(
                "β (calibration factor)",
                min_value=0.1, max_value=2.0, value=0.95, step=0.01,
                help="Dimensionless calibration factor"
            )
            G = st.number_input(
                "G (shear modulus, GPa)",
                min_value=10.0, max_value=100.0, value=30.0, step=1.0,
                help="Shear modulus of Ag for stress normalization"
            )
        
        # Sintering analysis type
        sintering_type = st.radio(
            "Sintering Analysis Type",
            ["Single Solution Analysis", "Bulk Analysis", "Orientation Sweep", 
             "Theoretical Curve", "System Mapping"],
            horizontal=True,
            help="Select type of sintering analysis"
        )
        
        # Region selection for sintering
        region_type_display_sinter = st.selectbox(
            "Select region for sintering analysis:",
            ["Defect Region (η > 0.6)", "Interface Region (0.4 ≤ η ≤ 0.6)", "Bulk Ag Material (η < 0.4)"],
            index=0,
            help="Select region for stress extraction in sintering analysis"
        )
        region_key_sinter = region_map[region_type_display_sinter]
    
    # ... [Continue with previous main() code]
    
    # In the main content area, add sintering analysis section
    if analysis_mode == "Sintering Temperature Analysis" and st.session_state.solutions:
        st.markdown('<h2 class="sub-header">🔥 Sintering Temperature Analysis</h2>', unsafe_allow_html=True)
        
        # Initialize sintering visualizer
        if 'sintering_visualizer' not in st.session_state:
            st.session_state.sintering_visualizer = EnhancedSinteringVisualizer()
        
        # Update calculator parameters
        st.session_state.sintering_visualizer.sintering_calculator.T0 = T0
        st.session_state.sintering_visualizer.sintering_calculator.T_min = T_min
        st.session_state.sintering_visualizer.sintering_calculator.beta = beta
        st.session_state.sintering_visualizer.sintering_calculator.G = G
        
        if sintering_type == "Single Solution Analysis":
            st.info("Analyzing sintering temperature for individual solutions")
            
            # Solution selection
            if st.session_state.solutions:
                solution_names = [f"{i+1}. {sol.get('filename', 'Unknown')}" 
                                 for i, sol in enumerate(st.session_state.solutions)]
                selected_solution = st.selectbox(
                    "Select Solution for Analysis",
                    solution_names,
                    index=0
                )
                
                if st.button("Compute Sintering Temperature", type="primary"):
                    solution_idx = int(selected_solution.split('.')[0]) - 1
                    solution = st.session_state.solutions[solution_idx]
                    
                    # Analyze solution
                    analyzer = OriginalFileAnalyzer()
                    analysis = analyzer.analyze_solution(
                        solution, region_key_sinter, 'sigma_hydro', 'max_abs'
                    )
                    
                    if analysis:
                        stress = abs(analysis['region_stress'])
                        T_sinter = st.session_state.sintering_visualizer.sintering_calculator.compute_sintering_temperature_exponential(stress)
                        system_info = st.session_state.sintering_visualizer.sintering_calculator.map_system_to_temperature(stress)
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Hydrostatic Stress", f"{stress:.3f} GPa")
                        with col2:
                            st.metric("Sintering Temperature", f"{T_sinter:.1f} K")
                        with col3:
                            st.metric("Sintering Temperature", f"{T_sinter-273.15:.1f} °C")
                        with col4:
                            st.metric("System Type", system_info[0].split('(')[0])
                        
                        # Show theoretical curve with point
                        theory_data = st.session_state.sintering_visualizer.sintering_calculator.get_theoretical_curve()
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(theory_data['stresses'], theory_data['T_exponential'], 
                               'b-', linewidth=2, label='Theoretical Curve')
                        ax.plot(stress, T_sinter, 'ro', markersize=12, 
                               label=f'Solution: {stress:.2f} GPa, {T_sinter:.0f} K')
                        
                        ax.set_xlabel('|σ_h| (GPa)', fontsize=11)
                        ax.set_ylabel('T_sinter (K)', fontsize=11)
                        ax.set_title('Solution on Theoretical Curve', fontsize=12, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # Add Celsius axis
                        ax2 = ax.twinx()
                        ax2.set_ylabel('Temperature (°C)', fontsize=11)
                        ax2.set_ylim([t-273.15 for t in ax.get_ylim()])
                        
                        st.pyplot(fig)
                        plt.close(fig)
        
        elif sintering_type == "Bulk Analysis":
            st.info("Analyzing sintering temperatures for all loaded solutions")
            
            if st.button("Perform Bulk Sintering Analysis", type="primary"):
                with st.spinner("Analyzing all solutions..."):
                    # Create comprehensive dashboard
                    fig = st.session_state.sintering_visualizer.create_comprehensive_sintering_dashboard(
                        st.session_state.solutions, region_key_sinter, 'sigma_hydro', 'max_abs'
                    )
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Create interactive plot
                    fig_interactive = st.session_state.sintering_visualizer.create_interactive_sintering_plot(
                        st.session_state.solutions, region_key_sinter, 'sigma_hydro', 'max_abs'
                    )
                    
                    if fig_interactive:
                        st.plotly_chart(fig_interactive, use_container_width=True)
        
        elif sintering_type == "Orientation Sweep":
            st.info("Performing sintering temperature sweep across orientations")
            
            # Sweep parameters
            col_sweep1, col_sweep2 = st.columns(2)
            with col_sweep1:
                min_angle = st.number_input(
                    "Min Angle (°)", value=50.0, min_value=0.0, max_value=360.0
                )
            with col_sweep2:
                max_angle = st.number_input(
                    "Max Angle (°)", value=60.0, min_value=0.0, max_value=360.0
                )
            
            n_points = st.slider("Number of Points", 20, 200, 100)
            
            # Base parameters
            base_params = {
                'defect_type': defect_type,
                'shape': shape,
                'eps0': eps0,
                'kappa': kappa,
                'theta': 0.0
            }
            
            if st.button("Perform Sintering Sweep", type="primary"):
                with st.spinner("Performing orientation sweep..."):
                    fig = st.session_state.sintering_visualizer.create_sintering_temperature_sweep(
                        st.session_state.solutions, base_params, (min_angle, max_angle),
                        region_key_sinter, 'sigma_hydro', 'max_abs', n_points
                    )
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
        
        elif sintering_type == "Theoretical Curve":
            st.info("Displaying theoretical sintering temperature curve")
            
            # Generate and plot theoretical curve
            theory_data = st.session_state.sintering_visualizer.sintering_calculator.get_theoretical_curve()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Exponential model
            ax.plot(theory_data['stresses'], theory_data['T_exponential'], 
                   'b-', linewidth=3, label='Exponential Model')
            
            # Arrhenius model
            ax.plot(theory_data['stresses'], theory_data['T_arrhenius'], 
                   'g--', linewidth=2, label='Arrhenius Model')
            
            # System boundaries
            ax.axvspan(0, 5, alpha=0.1, color='green', label='System 1')
            ax.axvspan(5, 20, alpha=0.1, color='orange', label='System 2')
            ax.axvspan(20, 35, alpha=0.1, color='red', label='System 3')
            
            # Reference points
            ax.plot(0, T0, 'go', markersize=15, label=f'System 1: {T0} K')
            ax.plot(12.5, st.session_state.sintering_visualizer.sintering_calculator.compute_sintering_temperature_exponential(12.5), 
                   'yo', markersize=15, label='System 2: ~475 K')
            ax.plot(28.5, T_min, 'ro', markersize=15, label=f'System 3: {T_min} K')
            
            ax.set_xlabel('Absolute Hydrostatic Stress |σ_h| (GPa)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Sintering Temperature (K)', fontsize=12, fontweight='bold')
            ax.set_title('Theoretical Sintering Temperature Models', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            # Add Celsius axis
            ax2 = ax.twinx()
            ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            ax2.set_ylim([t-273.15 for t in ax.get_ylim()])
            
            # Add formula annotation
            formula_text = r"$T_{\text{sinter}}(\sigma_h) = T_0 \exp\left(-\beta \frac{|\sigma_h|}{G}\right)$"
            ax.text(0.05, 0.95, formula_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Display parameters
            with st.expander("Model Parameters", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("T₀ (reference)", f"{T0} K")
                    st.metric("β (calibration)", f"{beta}")
                with col2:
                    st.metric("G (shear modulus)", f"{G} GPa")
                    st.metric("T_min (minimum)", f"{T_min} K")
                with col3:
                    sigma_peak = -G/beta * np.log(T_min/T0)
                    st.metric("σ_peak (theoretical)", f"{sigma_peak:.1f} GPa")
                    activation_energy = 1.1  # eV (typical for Ag)
                    st.metric("Q_a (activation)", f"{activation_energy} eV")
        
        elif sintering_type == "System Mapping":
            st.info("Mapping solutions to AgNP system classification")
            
            if st.button("Generate System Map", type="primary"):
                # Analyze all solutions
                analyzer = OriginalFileAnalyzer()
                analyses = analyzer.analyze_all_solutions(
                    st.session_state.solutions, region_key_sinter, 'sigma_hydro', 'max_abs'
                )
                
                if analyses:
                    # Create classification table
                    system_data = []
                    for analysis in analyses:
                        stress = abs(analysis['region_stress'])
                        T_sinter = st.session_state.sintering_visualizer.sintering_calculator.compute_sintering_temperature_exponential(stress)
                        system_info = st.session_state.sintering_visualizer.sintering_calculator.map_system_to_temperature(stress)
                        
                        system_data.append({
                            'Filename': analysis['filename'],
                            'Orientation (°)': f"{analysis['theta_deg']:.1f}",
                            '|σ_h| (GPa)': f"{stress:.3f}",
                            'T_sinter (K)': f"{T_sinter:.1f}",
                            'T_sinter (°C)': f"{T_sinter-273.15:.1f}",
                            'System': system_info[0],
                            'Defect Type': analysis['params'].get('defect_type', 'Unknown')
                        })
                    
                    # Display as dataframe
                    df = pd.DataFrame(system_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("#### 📊 System Distribution Summary")
                    system_counts = df['System'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("System 1 (Perfect)", 
                                 system_counts.get('System 1 (Perfect Crystal)', 0))
                    with col2:
                        st.metric("System 2 (SF/Twin)", 
                                 system_counts.get('System 2 (Stacking Faults/Twins)', 0))
                    with col3:
                        st.metric("System 3 (Plastic)", 
                                 system_counts.get('System 3 (Plastic Deformation)', 0))
                    
                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download System Mapping Data",
                        data=csv,
                        file_name="agnp_system_mapping.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    # ... [Rest of main() function remains the same]

# =============================================
# ENHANCED RESULTS MANAGER WITH SINTERING SUPPORT
# =============================================

class EnhancedResultsManagerWithSintering(EnhancedResultsManager):
    """Extended results manager with sintering temperature support"""
    
    @staticmethod
    def prepare_sintering_analysis_data(solutions, region_type='bulk',
                                       sintering_calculator=None):
        """Prepare sintering analysis data for export"""
        if sintering_calculator is None:
            sintering_calculator = SinteringTemperatureCalculator()
        
        analyzer = OriginalFileAnalyzer()
        analyses = analyzer.analyze_all_solutions(
            solutions, region_type, 'sigma_hydro', 'max_abs'
        )
        
        sintering_data = []
        for analysis in analyses:
            stress = abs(analysis['region_stress'])
            T_sinter_exp = sintering_calculator.compute_sintering_temperature_exponential(stress)
            T_sinter_arr = sintering_calculator.compute_sintering_temperature_arrhenius(stress)
            system_info = sintering_calculator.map_system_to_temperature(stress)
            
            sintering_data.append({
                'filename': analysis['filename'],
                'orientation_deg': float(analysis['theta_deg']),
                'hydrostatic_stress_gpa': float(stress),
                'sintering_temp_exponential_k': float(T_sinter_exp),
                'sintering_temp_arrhenius_k': float(T_sinter_arr),
                'sintering_temp_celsius': float(T_sinter_exp - 273.15),
                'system_classification': system_info[0],
                'defect_type': analysis['params'].get('defect_type', 'Unknown'),
                'eps0': float(analysis['params'].get('eps0', 0)),
                'kappa': float(analysis['params'].get('kappa', 0))
            })
        
        # Calculate statistics
        if sintering_data:
            stresses = [d['hydrostatic_stress_gpa'] for d in sintering_data]
            temps = [d['sintering_temp_exponential_k'] for d in sintering_data]
            
            statistics = {
                'num_solutions': len(sintering_data),
                'mean_stress_gpa': float(np.mean(stresses)),
                'max_stress_gpa': float(np.max(stresses)),
                'min_stress_gpa': float(np.min(stresses)),
                'mean_sintering_temp_k': float(np.mean(temps)),
                'max_sintering_temp_k': float(np.max(temps)),
                'min_sintering_temp_k': float(np.min(temps)),
                'temp_range_k': float(np.max(temps) - np.min(temps)),
                'system_distribution': {
                    sys: len([d for d in sintering_data if sys in d['system_classification']])
                    for sys in ['System 1', 'System 2', 'System 3']
                }
            }
        else:
            statistics = {}
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'sintering_temperature',
                'model_parameters': {
                    'T0_k': float(sintering_calculator.T0),
                    'beta': float(sintering_calculator.beta),
                    'G_gpa': float(sintering_calculator.G),
                    'sigma_peak_gpa': float(sintering_calculator.sigma_peak),
                    'T_min_k': float(sintering_calculator.T_min)
                },
                'region_type': region_type,
                'description': 'AgNP sintering temperature analysis based on hydrostatic stress'
            },
            'sintering_data': sintering_data,
            'statistics': statistics
        }
        
        return export_data
    
    @staticmethod
    def create_sintering_analysis_report(solutions, region_type='bulk',
                                        sintering_calculator=None):
        """Create comprehensive sintering analysis report"""
        export_data = EnhancedResultsManagerWithSintering.prepare_sintering_analysis_data(
            solutions, region_type, sintering_calculator
        )
        
        # Generate markdown report
        report = f"""# AGNP SINTERING TEMPERATURE ANALYSIS REPORT

## Analysis Summary
- **Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Number of Solutions Analyzed**: {export_data['statistics'].get('num_solutions', 0)}
- **Analysis Region**: {region_type}
- **Model Used**: Stress-modified exponential model

## Model Parameters
- **Reference Temperature (T₀)**: {export_data['metadata']['model_parameters']['T0_k']} K
- **Calibration Factor (β)**: {export_data['metadata']['model_parameters']['beta']}
- **Shear Modulus (G)**: {export_data['metadata']['model_parameters']['G_gpa']} GPa
- **Peak Stress (σ_peak)**: {export_data['metadata']['model_parameters']['sigma_peak_gpa']} GPa
- **Minimum Temperature (T_min)**: {export_data['metadata']['model_parameters']['T_min_k']} K

## Key Findings
- **Average Sintering Temperature**: {export_data['statistics'].get('mean_sintering_temp_k', 0):.1f} K
- **Temperature Range**: {export_data['statistics'].get('temp_range_k', 0):.1f} K
- **Maximum Stress Observed**: {export_data['statistics'].get('max_stress_gpa', 0):.2f} GPa

## System Distribution
"""
        
        if 'system_distribution' in export_data['statistics']:
            for system, count in export_data['statistics']['system_distribution'].items():
                report += f"- **{system}**: {count} solutions\n"
        
        report += """
## Methodology
The sintering temperature is calculated using the empirical exponential model:
T_sinter(σ_h) = T₀ * exp(-β * |σ_h| / G)

Where:
- T₀ = Reference temperature at zero stress (623 K for Ag)
- β = Calibration factor (0.95)
- G = Shear modulus of Ag (30 GPa)
- |σ_h| = Absolute hydrostatic stress at habit plane (54.7°)

## Interpretation
- **System 1 (Perfect Crystal)**: σ_h < 5 GPa, T_sinter ≈ 600-630 K
- **System 2 (Stacking Faults/Twins)**: 5 GPa ≤ σ_h < 20 GPa, T_sinter ≈ 450-550 K
- **System 3 (Plastic Deformation)**: σ_h ≥ 20 GPa, T_sinter ≈ 350-400 K

## Implications for AgNP Bonding
The analysis demonstrates how defect engineering through controlled stress fields can significantly reduce sintering temperatures, enabling low-temperature AgNP bonding for advanced electronic packaging applications.

---
*Report generated by Ag FCC Twin Sintering Analysis System*
"""
        
        return export_data, report
